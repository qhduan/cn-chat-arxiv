# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enforcing Code Reasoning in Agentic Scaffold for More Effective Issue Resolution](https://arxiv.org/abs/2610.23782) | 该研究通过分析7,955条智能体编程轨迹揭示代码推理的形式及其缺乏引导的问题，并提出了DeskCheck智能体脚手架，通过模拟执行路径定位根因、模拟候选补丁等方式系统性引导代码推理，从而提升仓库级议题解决的效果。 |
| [^2] | [Empirical Insights from AI-Powered Software Engineering in Practice](https://arxiv.org/abs/2610.23632) | 本文通过三个案例研究实证分析了生成式AI在软件工程集成中遇到的实际问题与挑战，并据此提出了管理风险、确保可靠成果的针对性建议，为有效且负责任地采用GenAI提供了指导。 |
| [^3] | [An Interactive GraphRAG Workspace for Reifying, Auditing, and Steering Conversational AI Memories](https://arxiv.org/abs/2610.23559) | 该论文提出了CognitiveCanvas，一个将对话式AI记忆从临时文本日志转化为持久、可操作且可审计知识对象的交互式GraphRAG工作区，通过分层语义事实图谱解决多会话软件开发中上下文复用面临的检索失效、时间漂移污染和提示词臃肿三大挑战。 |
| [^4] | [Recall-Oriented Precise Binary-to-Source Function Matching](https://arxiv.org/abs/2610.23546) | BScout通过利用已有匹配的位置信息将搜索范围扩展到初始候选集之外，并采用组合感知训练与大语言模型审查，从而在处理内联函数的同时提高二进制-源代码函数匹配的精确率和召回率。 |
| [^5] | [MLOps-Bench: Benchmarking AI Agents on Production ML Engineering in Real Repositories](https://arxiv.org/abs/2610.23506) | 提出MLOps-Bench基准，基于31个真实开源机器学习仓库的212个经验证任务，评估AI智能体在生产级机器学习工程全生命周期（数据管道、特征工程、模型训练与服务、监控、CI/CD及治理）中的可执行变更能力，发现当前最强智能体的表现仍然有限。 |
| [^6] | [Monitor-Validity Failures: LLM Safety Evidence Paths at Evolving Framework Boundaries](https://arxiv.org/abs/2610.23362) | 该论文提出“金丝雀契约”测试协议，用于发现LLM安全框架中的一种新型故障——“监控有效性失效”，即风险证据已被检测器识别却在框架内部丢失、未能送达阻断决策点——并在11个LLM/智能体安全项目组中确认了18个此类证据路径失效案例。 |
| [^7] | [Robust Function Calling for LLM Agents Under Interface Drift](https://arxiv.org/abs/2610.23342) | 提出了一种无需训练的轻量级修复层 SchemaShield-Lite，通过模式校验、候选集重试和一次性受限修复，在接口漂移场景下显著恢复LLM智能体的函数调用准确率。 |
| [^8] | [SpinControl: Control Software for Prototyping and Benchmarking of Spin-Qubit Devices and Measurement Protocols](https://arxiv.org/abs/2610.23316) | 本文提出了自旋量子比特控制软件SpinControl的架构设计，通过数据与行为分离、功能模块化以及CI流水线与输入验证三大指标，同时满足实验者低层调校与算法研究者高层电路访问的需求，支持器件原型设计与测量协议基准测试。 |
| [^9] | [Towards Agentic Proof Harness Engineering](https://arxiv.org/abs/2610.23241) | 该论文提出一种智能体化的证明测试桩工程器，能够根据软件项目和指定的验证目标自动构建可供验证器使用的证明测试桩，从而大幅减少形式化验证中的人工工作量。 |
| [^10] | [Code Skill Server: A Large-Scale Search Engine for Code Generation Skills](https://arxiv.org/abs/2610.23186) | 提出了Code Skill Server（CSS），一个从多个异构来源增量收集并索引公开代码生成技能的大规模搜索引擎，能够自动为目标任务检索最有用的技能，解决了手动搜索技能在大规模场景下不可行的问题。 |
| [^11] | [Efficient Program Reduction for Library Fuzzing Harnesses](https://arxiv.org/abs/2610.23054) | 本文提出HarnessMinimizer系统，通过复用头文件分析和预编译的库与运行时来高效精简C/C++库模糊测试工具，帮助开发者快速判断崩溃源于库缺陷还是工具误用。 |
| [^12] | [VulnSAGE: Adjudicating SAST Reports by Connecting LLM Reasoning to Program Structure](https://arxiv.org/abs/2610.23035) | VulnSAGE提出了一种裁决SAST报告的系统，通过双向连接大语言模型推理与程序结构，利用以报告为中心的切片器恢复决定性代码的依赖关系，并通过CLAIM中间表示将模型解释记录为可被程序分析验证的语义事实，从而实现可核验的缺陷判断。 |
| [^13] | [MergeShuffle: Reconstruct Shared Libraries for Improved Performance and Security](https://arxiv.org/abs/2610.22952) | 本文提出MergeShuffle工具，通过在识别出共享库中被使用的函数后对其进行恰当重构以消除未使用代码，从而提升共享库瘦身在性能与安全性方面的效果。 |
| [^14] | [No One-Size-Fits-All: Adaptive Code Editing with Feature-Based Strategy Selection](https://arxiv.org/abs/2610.22881) | 该研究构建了CEBench基准并系统对比三种代码编辑策略，发现没有任何单一策略在所有任务上都最优，策略效果显著受任务特征影响，因此代码编辑应根据任务特征自适应地选择编辑策略。 |
| [^15] | [Between Spoilers and Guesswork: Avoiding Over-Specification in Coding Benchmarks](https://arxiv.org/abs/2610.22760) | 该论文揭示了编程基准测试中通过向任务描述泄露真实答案信息来解决任务描述不足的做法反而会导致过度具体化（剧透）问题，使解决方案绕过基准本应考察的难点，并主张在剧透与臆测之间取得平衡以延长基准的使用寿命。 |
| [^16] | [Enjoying the Product but Not Joining the Builders: Investigating Why Students Don't Contribute to OSS](https://arxiv.org/abs/2610.22664) | 本研究通过对122名计算专业学生的调查发现，尽管学生频繁使用开源软件，但仅有12.3%积极参与贡献，技能差距、时间限制、代码库复杂性和冒名顶替综合征是主要障碍，而导师指导、新手友好任务和课程激励可能促进参与。 |
| [^17] | [SIEVE: A Parameterized GitHub Corpus Builder for Software Engineering Research](https://arxiv.org/abs/2610.22544) | SIEVE是一个开源、参数化的GitHub语料库构建工具，能够在函数、方法和类的代码粒度上，根据编程语言、仓库质量和创建日期等标准提取带标注的代码语料库，为软件工程研究提供了对语料库组成的直接控制能力。 |
| [^18] | [ForTress: Bringing Assurance and Determinism for LLM-Driven Fortran-to-Python Translation](https://arxiv.org/abs/2610.22528) | ForTress将LLM翻译决策固化为受保护的可执行规则以实现确定性重用，并通过Lean机械化证明为部分规则提供语义保证，从而提升Fortran到Python翻译的可靠性。 |
| [^19] | [Understanding LLM Hallucinations Caused by Soft Errors](https://arxiv.org/abs/2610.22520) | 本文提出LLMEI工具在推理过程中向大语言模型注入硬件软错误，首次从硬件层面研究LLM幻觉问题，建立了失败模式分类体系，并评估了软错误在不同架构、任务和错误率下的影响。 |
| [^20] | [AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems](https://arxiv.org/abs/2610.22511) | AuditKIT是一个统一的评估框架，将基准测试、评判评估、运行时性能、工具使用和检索质量整合到单一结果表示中，解决了大语言模型与智能体系统评估中接口和指标语义不一致的问题。 |
| [^21] | [EnterpriseSynth: Zero-Execution SFT and Evaluation Data from OpenAPI Specs](https://arxiv.org/abs/2610.22412) | EnterpriseSynth提出了一种零执行框架，无需实际调用企业API即可将OpenAPI规范转换为工具使用型大语言模型智能体的监督微调训练轨迹和成对评估实例，并通过静态验证与对抗性细化保证数据质量。 |
| [^22] | [Eliciting Specifications: When Does Interactive Clarification Help?](https://arxiv.org/abs/2610.22382) | 本文提出一种引出代理（EA）方法，在形式化之前向模拟开发者提出有预算限制的澄清问题以改进LLM生成的Lean规格说明，并系统评估了在不同问题预算（k=3与k=10）及同模型/跨模型配置下交互式澄清何时有效。 |
| [^23] | [Are We Testing the Wrong Quantum Bugs? A Study of Fault Representation Gaps in Quantum Software Testing](https://arxiv.org/abs/2610.22089) | 该研究通过审查量子变异测试工具和人工整理的缺陷基准，并收集两年内410个真实缺陷，揭示量子软件测试评估中存在显著的缺陷表示差距——现有变异算子与基准仅覆盖极少数缺陷族，无法充分代表真实量子软件中的缺陷。 |
| [^24] | [WEAVE: Weaving Real-World Code Evolution into Dependency-Structured Tasks for Benchmarking and Training Code Agents](https://arxiv.org/abs/2610.22039) | WEAVE 框架将真实世界中经过验证的多 PR 代码演进历史转化为依赖结构化的可执行任务，并引入基于测试的确定性度量 Graph Credit，以更好地对长程代码智能体进行基准评估和训练。 |
| [^25] | [Benchmarking and Enhancing Design-Aware Code Review in the Linux Kernel](https://arxiv.org/abs/2610.21925) | 本文提出了DesignLore知识增强方法和DesignReviewBench基准，通过从历史审查讨论中提取、组织并复用设计决策知识，系统性评估并增强了Linux内核中的设计感知代码审查。 |
| [^26] | [Recursive Game Creator: An Agentic Product-Level Experience-Oriented Game Harness](https://arxiv.org/abs/2610.08621) | 该论文提出递归游戏创造者框架，通过设计者、构建者、玩家和评审者四个智能体的递归协作，将粗糙的游戏原型迭代开发为真正注重玩家体验的有趣游戏。 |
| [^27] | [Trust Is Not a Score: Runtime Assurance Contracts for High-Risk AI Agents](https://arxiv.org/abs/2609.39717) | 提出了运行时保障契约（RAC），用非补偿性强制门控而非综合分数来决定高风险AI智能体在关键任务中的权限动态转换，填补了观察证据与智能体授权之间的保障转换缺口。 |
| [^28] | [SWE-Prometheus: Measuring Engineering Governance Improvements in Real-World Repositories](https://arxiv.org/abs/2609.29465) | 该论文提出了SWE-Prometheus基准，首次系统评估大语言模型编程智能体在开放式仓库工程治理任务中的能力，要求智能体自主识别风险、排序干预优先级并验证变更，通过六个治理维度和多重验证机制对十个模型进行了评测。 |
| [^29] | [Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness](https://arxiv.org/abs/2609.00050) | 提出了一个智能体云工作流工程框架，通过将图工程（长时程工作流推进）、循环工程（有界诊断与修复重试）和零信任智能体套件（受限执行）三个关注点分离，将自然语言云工程任务自动转化为经过验证的代码仓库和可验证的云部署。 |
| [^30] | [CIDR: A Large-Scale Industrial Source Code Dataset for Software Engineering Research](https://arxiv.org/abs/2605.12153) | 本文发布了CIDR——一个包含4,687个仓库、84种编程语言、超10亿行代码的大规模匿名化工业源代码数据集，为代码智能和软件工程研究提供真实企业级代码资源。 |
| [^31] | [Epistemological Debt: Cognitive Atrophy and Systemic Collapse in AI-Dependent Software Engineering](https://arxiv.org/abs/2604.26855) | 本文提出“认识论债务”概念，指出AI依赖的软件开发会导致认知萎缩和系统性崩溃，并主张通过人机协同教学标准来维持工程韧性。 |
| [^32] | [A Longitudinal Study of Dependency Reclassifications in JavaScript Projects](https://arxiv.org/abs/2604.08747) | 本研究通过对33,087个JavaScript项目package.json文件的提交级分析，首次系统揭示了依赖项重新分类（包括移除和角色重分配）是一种普遍的维护活动，存在于79.1%的项目中，占所有依赖项维护提交的19.4%。 |
| [^33] | [Superset Decompilation](https://arxiv.org/abs/2603.28002) | 该论文提出“超集反编译”新范式：用单调 Datalog 规则在 IR 层次结构中保留多种候选解释、以函数级加权 MaxSMT 求解器统一裁决，从而克服传统反编译器因过早确定单一解释而无法对真实程序生成可编译输出的问题。 |
| [^34] | [A Theoretical Analysis of Test-Driven Code Generation](https://arxiv.org/abs/2602.06098) | 本文建立了测试驱动代码生成的概率理论框架，证明了基于模糊功能相似性的代码选择估计器在信噪比上严格优于基于功能等价性的估计器，并将回溯提示形式化为汤普森采样的上下文内近似，推导出新的遗憾界以解释其有效性受限于歧义性的原因。 |
| [^35] | [Veri-Sure: Multi-Agent RTL Code Generation with Temporal Tracing, Slicing and Formal Verification](https://arxiv.org/abs/2601.19747) | Veri-Sure是一个多智能体RTL代码生成框架，通过设计契约对齐智能体意图、静态依赖切片引导的局部化补丁修复以及形式化验证与时序追踪，解决了LLM生成RTL代码中反馈稀疏、整文件重生成引发回归和语义漂移三大瓶颈。 |
| [^36] | [FeatureFuzz: Treating Bug-prone Features as Core Primitives for Compiler Fuzzing](https://arxiv.org/abs/2601.12360) | 提出FeatureFuzz，一种将从历史缺陷中提取的易错语义显式建模为特性和特性组并加以重用的编译器模糊测试工具，从而更有效地发现依赖程序语义的编译器缺陷。 |
| [^37] | [Pluvio: Assembly Clone Search for Out-of-domain Architectures and Libraries through Transfer Learning and Conditional Variational Information Bottleneck.](http://arxiv.org/abs/2307.10631) | 本论文介绍了一种通过迁移学习和条件变分信息瓶颈实现对领域外结构和库的汇编克隆搜索的方法。已有的方法往往限制于少数训练过的工具链变体，而本研究首次解决了未见过架构和库的情况，提出了一种新的解决方案。 |

# 详细

[^1]: 在智能体脚手架中强化代码推理以实现更高效的议题解决

    Enforcing Code Reasoning in Agentic Scaffold for More Effective Issue Resolution

    [https://arxiv.org/abs/2610.23782](https://arxiv.org/abs/2610.23782)

    该研究通过分析7,955条智能体编程轨迹揭示代码推理的形式及其缺乏引导的问题，并提出了DeskCheck智能体脚手架，通过模拟执行路径定位根因、模拟候选补丁等方式系统性引导代码推理，从而提升仓库级议题解决的效果。

    

    推理是基于ReAct的编程智能体的核心，然而这些智能体是否以及如何对代码进行推理仍不清楚。我们在SWE-bench-Verified、SWE-bench-Pro和SWE-bench-Lite上，研究了来自六种单智能体与多智能体脚手架、涵盖四种编程语言的7,955条议题解决轨迹。我们在66.95%的轨迹中识别出12种代码推理形式。代码推理与更高的解决率相关，且其强度随任务复杂度增加而上升。然而，代码推理在很大程度上缺乏引导：智能体在定位问题时可能只关注表层症状，无法完善接近正确的补丁，忽略议题需求，并生成行为多样性有限的测试。为解决这些局限，我们提出了DeskCheck，一个在仓库级议题解决全过程中系统性引导代码推理的智能体脚手架。DeskCheck通过模拟执行路径来定位根本原因，通过模拟候选补丁来完善……（摘要在此处截断）

    arXiv:2610.23782v1 Announce Type: new  Abstract: Reasoning is central to ReAct-based programming agents, yet whether and how they reason about code remains unclear. We study 7,955 issue-resolution trajectories from six single- and multi-agent scaffolds across four programming languages on SWE-bench-Verified, SWE-bench-Pro, and SWE-bench-Lite. We identify 12 forms of code reasoning in 66.95% of trajectories. Code reasoning is associated with higher resolution rates, and its intensity increases with task complexity. However, it remains largely unguided: agents may focus on surface-level symptoms during localization, fail to refine near-correct patches, overlook issue requirements, and generate tests with limited behavioral diversity. To address these limitations, we propose DeskCheck, an agentic scaffold that systematically guides code reasoning throughout repository-level issue resolution. DeskCheck simulates execution paths to localize root causes, simulates candidate patches to refine
    
[^2]: AI赋能软件工程实践中的实证洞察

    Empirical Insights from AI-Powered Software Engineering in Practice

    [https://arxiv.org/abs/2610.23632](https://arxiv.org/abs/2610.23632)

    本文通过三个案例研究实证分析了生成式AI在软件工程集成中遇到的实际问题与挑战，并据此提出了管理风险、确保可靠成果的针对性建议，为有效且负责任地采用GenAI提供了指导。

    

    生成式人工智能正在未来的软件工程中变得不可或缺。其应用范围十分广泛——从辅助开发到自主软件工程，这要求组织应对相关的风险与挑战。本文通过三个案例研究对这些议题进行调查，重点揭示了在GenAI集成过程中遇到的实际问题。基于这些研究发现，本文为管理风险和确保可靠成果提供了有针对性的建议。本文基于实证洞察，为在软件工程中有效且负责任地采用GenAI提供了指导。

    arXiv:2610.23632v1 Announce Type: new  Abstract: Generative Artificial Intelligence (GenAI) is becoming indispensable in future software engineering. Its application spans a wide spectrum - from assisted development to autonomous software engineering, requiring organizations to address associated risks and challenges. This paper investigates these issues through three case studies, highlighting practical problems encountered during GenAI integration. Based on these findings, it provides targeted recommendations for managing risks and ensuring reliable outcomes. This paper provides guidance for effective and responsible GenAI adoption in software engineering based on empirical insights.
    
[^3]: 一个用于具体化、审计和引导对话式AI记忆的交互式GraphRAG工作区

    An Interactive GraphRAG Workspace for Reifying, Auditing, and Steering Conversational AI Memories

    [https://arxiv.org/abs/2610.23559](https://arxiv.org/abs/2610.23559)

    该论文提出了CognitiveCanvas，一个将对话式AI记忆从临时文本日志转化为持久、可操作且可审计知识对象的交互式GraphRAG工作区，通过分层语义事实图谱解决多会话软件开发中上下文复用面临的检索失效、时间漂移污染和提示词臃肿三大挑战。

    

    对话式大语言模型（LLM）被广泛应用于软件工程中的代码生成、调试和架构设计。然而，随着开发者参与多会话开发任务，技术知识被孤立地存储在分散的线性聊天历史中。跨会话复用上下文面临三个关键挑战：（1）平面向量检索无法遍历相距较远的架构决策之间的多跳关系依赖；（2）软件包和API的时间漂移导致在复用过期对话日志时产生静默的上下文污染；（3）原始对话拼接会导致提示词臃肿和注意力退化。我们提出了CognitiveCanvas，这是一个交互式工具，它将对话式AI记忆从临时性的文本日志转变为持久化、可操作且可审计的知识对象。CognitiveCanvas集成了：（1）一种分层语义事实图谱（HSFG），用于提炼……（摘要原文在此处截断）

    arXiv:2610.23559v1 Announce Type: new  Abstract: Conversational Large Language Models (LLMs) are widely employed in software engineering for code generation, debugging, and architectural design. However, as developers engage in multi-session development tasks, technical knowledge becomes siloed in isolated, linear chat histories. Reusing context across sessions leads to three critical challenges: (1) flat-vector retrieval fails to traverse multi-hop relational dependencies between distant architectural decisions; (2) temporal drift in software packages and APIs results in silent context poisoning when reusing aged dialogue logs; and (3) raw conversation concatenation induces prompt bloat and attention degradation.   We present CognitiveCanvas, an interactive tool that transforms conversational AI memories from ephemeral text logs into persistent, manipulable, and audited knowledge objects. \textsc{CognitiveCanvas} integrates: (1) a Hierarchical Semantic Fact-Graph (HSFG) that distills 
    
[^4]: 面向召回率的精确二进制-源代码函数匹配

    Recall-Oriented Precise Binary-to-Source Function Matching

    [https://arxiv.org/abs/2610.23546](https://arxiv.org/abs/2610.23546)

    BScout通过利用已有匹配的位置信息将搜索范围扩展到初始候选集之外，并采用组合感知训练与大语言模型审查，从而在处理内联函数的同时提高二进制-源代码函数匹配的精确率和召回率。

    

    二进制到源代码的函数匹配通过识别被复用的第三方代码来支持软件成分分析，但现有的基于嵌入的方法将后续的精细化处理限制在最初检索到的候选集中。当正确的源函数不在这个集合中时，对现有候选进行选择或过滤无法恢复匹配。内联技术通过将多个源函数组合成一个二进制函数，进一步增加了检索的复杂性。在本文中，我们提出BScout来提高匹配精度并发现额外的匹配。BScout利用已有匹配的位置信息为未解析的二进制函数补充候选，将搜索范围扩展到初始候选集之外，并采用组合感知训练来解决内联组合匹配问题。该方法采用大语言模型来选择和审查匹配结果，同时位置信息也有助于过滤不一致的结果。实验……

    arXiv:2610.23546v1 Announce Type: new  Abstract: Binary-to-source function matching supports software composition analysis by identifying reused third-party code, but existing embedding-based approaches restrict subsequent refinement to initially retrieved candidates. When the correct source function is absent from this set, selecting or filtering the available candidates cannot recover the match. Inlining further complicates the retrieval by combining multiple source functions into one binary function. In this paper, we propose BScout to improve matching precision and uncover additional matches. BScout uses the positional information of existing matches to supplement candidates for unresolved binary functions, extending the search beyond the initial candidate set, and adopts composition-aware training to address the inlined composition matching. Large language models are adopted to select and review matches, while positional information also helps filter inconsistent results. Experime
    
[^5]: MLOps-Bench：在真实仓库中对AI智能体的生产级机器学习工程能力进行基准测试

    MLOps-Bench: Benchmarking AI Agents on Production ML Engineering in Real Repositories

    [https://arxiv.org/abs/2610.23506](https://arxiv.org/abs/2610.23506)

    提出MLOps-Bench基准，基于31个真实开源机器学习仓库的212个经验证任务，评估AI智能体在生产级机器学习工程全生命周期（数据管道、特征工程、模型训练与服务、监控、CI/CD及治理）中的可执行变更能力，发现当前最强智能体的表现仍然有限。

    

    AI智能体越来越多地被期望修改和维护生产环境中的机器学习系统，然而现有的基准测试主要评估竞赛式的机器学习实验或通用的仓库问题解决。我们提出了MLOps-Bench，一个面向生产级机器学习工程的基于真实仓库的基准测试。与先前的机器学习智能体基准不同，MLOps-Bench评估真实机器学习仓库中可执行的生命周期变更，涵盖数据管道、特征工程、模型训练、模型服务、监控、CI/CD与治理以及端到端集成。每个任务都基于仓库证据构建，包含自然语言需求描述、标准补丁、合理的错误基线、fail-to-pass测试、pass-to-pass测试以及来源元数据。该基准包含来自31个公开机器学习仓库的212个经过验证的任务，共有636个fail-to-pass测试和856个pass-to-pass测试。我们评估了六个智能体，发现表现最佳的系统的解决能力仍然有限。

    arXiv:2610.23506v1 Announce Type: new  Abstract: AI agents are increasingly expected to modify and maintain machine-learning systems in production, yet existing benchmarks primarily evaluate either competition-style ML experimentation or generic repository issue resolution. We introduce MLOps-Bench, a repository-grounded benchmark for production ML engineering. Unlike prior ML-agent benchmarks, MLOps-Bench evaluates executable lifecycle changes inside real ML repositories, spanning data pipelines, feature engineering, model training, model serving, monitoring, CI/CD and governance, and end-to-end integration. Each task is grounded in repository evidence and includes a natural-language requirement, oracle patch, plausible buggy baseline, fail-to-pass tests, pass-to-pass tests, and provenance metadata. The benchmark contains 212 validated tasks from 31 public ML repositories, with 636 fail-to-pass tests and 856 pass-to-pass tests. We evaluate six agents and find that the best system solv
    
[^6]: 监控有效性失效：演化框架边界处的LLM安全证据路径

    Monitor-Validity Failures: LLM Safety Evidence Paths at Evolving Framework Boundaries

    [https://arxiv.org/abs/2610.23362](https://arxiv.org/abs/2610.23362)

    该论文提出“金丝雀契约”测试协议，用于发现LLM安全框架中的一种新型故障——“监控有效性失效”，即风险证据已被检测器识别却在框架内部丢失、未能送达阻断决策点——并在11个LLM/智能体安全项目组中确认了18个此类证据路径失效案例。

    

    现代大语言模型应用依赖运行时监控器来检查工具调用、执行轨迹、数据流、检测器输出和验证器输入。这带来了一个接口可靠性问题：框架路径可能仍在正常运行，但与策略相关的证据却从未到达其决策点。在一个代表性案例中，正则表达式检测器返回了阳性匹配结果，但框架在阻断决策之前将该结果搁置了。这不同于越狱攻击和检测器误判，因为证据已经被识别，却与执行机制断开了连接。我们使用“金丝雀契约”来研究这种修复边界特征：这是一种父级/固定确认单元，它仅在发生变更的证据表面上放置已知载荷，并记录监控器的可见性以及动作边界情况下的下游决策。将该协议应用于公开的框架修复中，我们在11个大语言模型或智能体安全项目组中确认了18个证据路径失效，并覆盖四个实现……（原文摘要在此处截断）

    arXiv:2610.23362v1 Announce Type: new  Abstract: Modern LLM applications rely on runtime monitors that inspect tool calls, traces, streams, detector outputs, and verifier inputs. This creates an interface-reliability problem: a framework path may remain operational while policy-relevant evidence never reaches its decision point. In one representative case, a regex detector returned a positive match, but the framework stranded the result before the block decision. This differs from jailbreaks and detector errors because the evidence is already recognized yet disconnected from enforcement. We study this repair-boundary signature with canary contracts: parent/fixed confirmation units that place a known payload only on the changed evidence surface and record monitor visibility together with the downstream decision for action-boundary cases. Applying the protocol to public framework repairs, we confirm 18 evidence-path failures across 11 LLM or agent safety project groups and four implement
    
[^7]: 接口漂移下LLM智能体的鲁棒函数调用

    Robust Function Calling for LLM Agents Under Interface Drift

    [https://arxiv.org/abs/2610.23342](https://arxiv.org/abs/2610.23342)

    提出了一种无需训练的轻量级修复层 SchemaShield-Lite，通过模式校验、候选集重试和一次性受限修复，在接口漂移场景下显著恢复LLM智能体的函数调用准确率。

    

    工具增强的语言模型智能体依赖接口文本：工具描述、参数模式（schema）以及推理时展示的候选工具列表。即使底层工具语义保持不变，这些接口在部署后也经常发生变化，这给函数调用系统带来了实际的可靠性问题。我们将这种故障模式称为“接口漂移”并加以研究，评估一个轻量级、无需训练的修复层能否在不重新训练基础模型的情况下恢复鲁棒性。我们的方法 SchemaShield-Lite 结合了模式校验、针对缺失工具名称的候选集重试以及一次性受限修复。在 Qwen3.5-9B 上，该方法将 DICE-300 的准确率从 0.717 提升至 0.813，将 BFCL-200 的准确率从 0.580 提升至 0.805，分别恢复了 19/22 和 33/45 个由漂移引起的失败案例，且未观察到修复带来的负面影响。简单的重试机制无法解释这些提升。消融实验表明，候选集漂移（尤其是干扰性工具）是造成大部分观察到的失败的原因……

    arXiv:2610.23342v1 Announce Type: new  Abstract: Tool-augmented language-model agents rely on interface text: tool descriptions, argument schemas, and the candidate tool list shown at inference time. These interfaces often change after deployment even when the underlying tool semantics remain fixed, creating a practical reliability problem for function-calling systems. We study this failure mode as interface drift and evaluate whether a lightweight, training-free repair layer can recover robustness without retraining the base model. Our method, SchemaShield-Lite, combines schema validation, candidate-set retry for missing tool names, and one-shot constrained repair. On Qwen3.5-9B, it raises DICE-300 accuracy from 0.717 to 0.813 and BFCL-200 accuracy from 0.580 to 0.805, recovering 19/22 and 33/45 drift-induced failures with no observed repair harms. Naive retry does not explain the gains. Ablations show that candidate-set drift, especially distractor tools, accounts for most observed d
    
[^8]: SpinControl：用于自旋量子比特器件与测量协议原型设计和基准测试的控制软件

    SpinControl: Control Software for Prototyping and Benchmarking of Spin-Qubit Devices and Measurement Protocols

    [https://arxiv.org/abs/2610.23316](https://arxiv.org/abs/2610.23316)

    本文提出了自旋量子比特控制软件SpinControl的架构设计，通过数据与行为分离、功能模块化以及CI流水线与输入验证三大指标，同时满足实验者低层调校与算法研究者高层电路访问的需求，支持器件原型设计与测量协议基准测试。

    

    量子计算机需要软件来控制量子比特的状态。随着对量子比特的操作在电路执行、校准和基准测试方面的复杂性不断增加，控制软件的复杂性也随之增长。自旋量子比特量子计算机正处于这样一个阶段：实验人员需要低级别的访问权限来进行调谐和校准，而量子算法研究人员则需要高级别的电路级访问。在本文中，我们介绍了SpinControl的软件设计。SpinControl融合了三项架构指标：1) 可定制性，通过分离数据与行为来实现，以支持器件和测量协议的原型设计；2) 模块化和可扩展性，通过将功能分离到不同的组件中来实现；3) 健壮性，通过CI流水线和输入验证来实现，以便在多个量子比特上可靠地执行量子电路。SpinControl支持不同类型的测量，同时支持自定义……

    arXiv:2610.23316v1 Announce Type: cross  Abstract: Quantum computers require software to control the states of the qubits. As the actions on qubits grow in complexity with circuit execution, calibration, and benchmarking, the complexity of the control software grows in tandem. Spin-qubit quantum computers are at a stage where experimentalists need low-level access for tuning and calibration, and quantum algorithm researchers want high-level circuit-level access. In this article, we present the software design of SpinControl. SpinControl integrates three architectural metrics: 1) customizability, to allow prototyping devices and measurement protocols, is achieved by separating data and behaviour; 2) modularity and extensibility are achieved by separating functionalities into different components; and 3) robustness, to reliably execute quantum circuits on many qubits, is achieved by CI pipelines and input validation. SpinControl allows different types of measurements while customizing th
    
[^9]: 迈向智能体化的证明测试桩工程

    Towards Agentic Proof Harness Engineering

    [https://arxiv.org/abs/2610.23241](https://arxiv.org/abs/2610.23241)

    该论文提出一种智能体化的证明测试桩工程器，能够根据软件项目和指定的验证目标自动构建可供验证器使用的证明测试桩，从而大幅减少形式化验证中的人工工作量。

    

    对真实世界项目进行形式化验证需要大量人工投入，从指定正确性属性到构建检查这些属性所需的基础设施。我们设想由验证智能体在整个过程中协助开发者。在涉及的众多任务中，有一项任务既具有基础性又适合自动化：证明测试桩工程。证明测试桩通过提供符号化输入、初始化代码以及其依赖项的模型，为一个程序单元（如函数或模块）定义一个隔离的验证环境。如今，证明测试桩工程在很大程度上仍是由验证工程师手动完成的任务。为探索编程智能体在此任务中的潜力，我们提出了一种智能体化的证明测试桩工程器，它能够根据一个软件项目和指定的验证目标自动构建可供验证器直接使用的测试桩。我们将该智能体与形式化的可靠性和精度保障相结合……（摘要原文在此处截断）

    arXiv:2610.23241v1 Announce Type: new  Abstract: Formal verification of real-world projects requires substantial manual effort, from specifying correctness properties to constructing the infrastructure needed to check them. We envision verification agents that assist developers throughout this process. Among the many tasks involved, one stands out as both foundational and amenable to automation: proof-harness engineering. A proof harness defines an isolated verification environment for a program unit, such as a function or module, by providing symbolic inputs, initialization code, and models of its dependencies. Today, proof-harness engineering remains largely a manual task performed by verification engineers. To explore the potential of coding agents for this task, we propose an agentic proof-harness engineer that automatically constructs verifier-ready harnesses from a software project and a designated verification target. We combine the agent with formal soundness and precision guar
    
[^10]: 代码技能服务器：面向代码生成技能的大规模搜索引擎

    Code Skill Server: A Large-Scale Search Engine for Code Generation Skills

    [https://arxiv.org/abs/2610.23186](https://arxiv.org/abs/2610.23186)

    提出了Code Skill Server（CSS），一个从多个异构来源增量收集并索引公开代码生成技能的大规模搜索引擎，能够自动为目标任务检索最有用的技能，解决了手动搜索技能在大规模场景下不可行的问题。

    

    智能体技能已被证明是提升智能体在特定任务（包括软件工程、自然科学和数学）上能力的有效技术。在这一过程中，使用智能体技能通常需要从异构来源中手动发现和整理技能，而现有的技能检索方法主要侧重于从已有技能库中选取相关技能。然而，随着可复用智能体技能的数量和种类快速增长，在异构来源中手动搜索相关技能在大规模场景下已变得不切实际。在本文中，我们提出了代码技能服务器，这是一个大规模搜索引擎，用于为目标任务找到最有用的代码生成技能。CSS由一个技能服务器组成，该服务器从多个来源增量式地收集并索引公开可用的代码生成技能，本工作所用语料库中包含993个技能。

    arXiv:2610.23186v1 Announce Type: cross  Abstract: Agent skills have proven to be an effective technique for improving agent ability on specific tasks, including software engineering, natural science, and mathematics. During such a process, using agent skills often involves manually discovering and organizing skills from heterogeneous sources, while existing skill retrieval methods mainly focus on selecting relevant skills from established skill libraries. However, with the rapidly increasing number and variety of reusable agent skills, manually searching for relevant skills across heterogeneous sources is becoming impractical at scale. In this paper, we introduce the Code Skill Server (CSS), a large-scale search engine for finding the most useful code generation skills for a target task. CSS consists of a skill server that incrementally collects and indexes publicly available code generation skills from multiple sources, with 993 skills included in the corpus used in this work. Extens
    
[^11]: 面向库模糊测试工具的高效程序精简

    Efficient Program Reduction for Library Fuzzing Harnesses

    [https://arxiv.org/abs/2610.23054](https://arxiv.org/abs/2610.23054)

    本文提出HarnessMinimizer系统，通过复用头文件分析和预编译的库与运行时来高效精简C/C++库模糊测试工具，帮助开发者快速判断崩溃源于库缺陷还是工具误用。

    

    库模糊测试工具将生成的输入转化为API调用。当复杂的测试工具触发崩溃时，开发人员必须确定原因是库本身的缺陷还是工具的无效使用。一个更小的复现程序可以简化这一调查过程，但获得它需要反复构建和执行修改后的测试工具并分析其崩溃，使得手动甚至自动化的精简过程非常耗时。此外，数据依赖的语句会阻碍精简器删除不必要的代码，而库中繁重的堆栈跟踪符号化也会拖慢候选程序的执行速度。为了应对这些挑战，我们提出了HarnessMinimizer，一个用于高效精简C/C++库模糊测试工具的系统。其优化的流水线通过复用头文件分析结果以及预先准备的库和运行时，减少了精简过程中构建和执行每个简化测试工具所需的工作量。该系统还会记录并重放输入值。

    arXiv:2610.23054v1 Announce Type: new  Abstract: Library fuzzing harnesses translate generated inputs into API calls. When a complex harness triggers a crash, developers must determine whether the cause is a library defect or invalid harness usage. A smaller reproducer can simplify this investigation, but obtaining one requires repeatedly building and executing modified harnesses and analyzing their crashes, making manual or even automatic minimization time-consuming. Moreover, data-dependent statements prevent the reducer to remove unnecessary code and the heavy stack trace symbolization in libraries also slows down the candidate execution. To address these challenges, we present HarnessMinimizer, a system for efficient minimization of C/C++ library fuzzing harnesses. Its optimized pipeline reuses header analysis and prepared libraries and runtimes to reduce the work required to build and execute each simplified harness during the reduction process. It records and replays input values
    
[^12]: VulnSAGE：通过连接大语言模型推理与程序结构来裁决SAST报告

    VulnSAGE: Adjudicating SAST Reports by Connecting LLM Reasoning to Program Structure

    [https://arxiv.org/abs/2610.23035](https://arxiv.org/abs/2610.23035)

    VulnSAGE提出了一种裁决SAST报告的系统，通过双向连接大语言模型推理与程序结构，利用以报告为中心的切片器恢复决定性代码的依赖关系，并通过CLAIM中间表示将模型解释记录为可被程序分析验证的语义事实，从而实现可核验的缺陷判断。

    

    arXiv:2610.23035v1 公告类型：新 摘要：静态应用安全测试（SAST）报告会标记那些安全性依赖于项目特定上下文的操作，维护者必须判断哪些报告描述的是真实缺陷。大语言模型（LLM）能够解释这些上下文，但用它们裁决报告需要决定性的代码，而这些代码往往远离被报告的代码行，且需要一个可供程序分析验证的判断依据。我们提出了VulnSAGE，一个针对SAST报告的裁决系统，它在两个方向上将大语言模型推理与程序结构相连接。其以报告为中心的切片器（slicer）能够恢复被报告操作的依赖关系，以及解释这些依赖关系的程序结构，例如辅助函数所作用的调用者对象。CLAIM是一种用于报告判断的中间表示，它将模型的每一条解释记录为一条语义事实，并引用其假定的程序关系。程序分析会验证这些前提条件，最终裁决由事实推导得出。

    arXiv:2610.23035v1 Announce Type: new  Abstract: Static Application Security Testing (SAST) reports flag operations whose safety depends on project-specific context, and maintainers must decide which reports describe real defects. Large language models (LLMs) can interpret this context, but adjudicating a report with them requires the decisive code, which often lies far from the reported line, and a decision basis that program analysis can check. We present VulnSAGE, an adjudication system for SAST reports that connects LLM reasoning to program structure in both directions. Its report-centered slicer recovers the dependencies of the reported operation together with the program structure that explains them, such as the caller object on which a helper acts. CLAIM, an intermediate representation for report judgments, records each model interpretation as a semantic fact that cites the program relations it assumes. Program analysis checks these premises, and the verdict follows from the fac
    
[^13]: MergeShuffle：重构共享库以提升性能与安全性

    MergeShuffle: Reconstruct Shared Libraries for Improved Performance and Security

    [https://arxiv.org/abs/2610.22952](https://arxiv.org/abs/2610.22952)

    本文提出MergeShuffle工具，通过在识别出共享库中被使用的函数后对其进行恰当重构以消除未使用代码，从而提升共享库瘦身在性能与安全性方面的效果。

    

    共享库瘦身多年来一直是一个活跃的研究课题。共享库瘦身工具依赖静态或动态分析技术来发现共享库中哪些部分被使用。一旦找到这些部分，这些工具会采取两种路径之一：要么移除库中未使用的部分（这需要对库进行重构），要么直接覆写未使用的函数。然而，最近一项针对瘦身工具的广泛研究表明，其中一些工具要么会增加静态二进制文件的大小，要么在处理复杂工作负载时总体效果较差。在本文中，我们认为一旦识别出共享库中被使用的函数，通过对共享库进行恰当的重构，可以提升共享库瘦身工具的性能。我们提出了MergeShuffle，一种新颖的工具，可以根据给定的文件范围集合来重构共享库，从而消除未使用的代码。

    arXiv:2610.22952v1 Announce Type: new  Abstract: Debloating shared libraries has been an active research topic for years. Tools for shared library debloating rely on either static or dynamic analysis techniques to discover which parts of the shared library is used. Once these parts are found, these tools take one of two routes, either removing parts of the library that are not used, requiring reconstruction of the library; or they simply overwriting the unused functions. However, recently a broad study~\cite{brown2024broad} on debloating tools has shown that some of these tools either increase the static binary size, or are generally less effective for complex workloads.   In this paper, we argue that shared-library debloaters' performance can be improved by proper reconstruction of the shared-libraries once the used functions in the library are identifies. We propose \myapp, a novel tool for reconstructing shared libraries from a given set of file ranges to eliminate unused code based
    
[^14]: 没有一劳永逸的方案：基于特征策略选择的自适应代码编辑

    No One-Size-Fits-All: Adaptive Code Editing with Feature-Based Strategy Selection

    [https://arxiv.org/abs/2610.22881](https://arxiv.org/abs/2610.22881)

    该研究构建了CEBench基准并系统对比三种代码编辑策略，发现没有任何单一策略在所有任务上都最优，策略效果显著受任务特征影响，因此代码编辑应根据任务特征自适应地选择编辑策略。

    

    大语言模型在自动化代码编辑（CE）任务中展现出强大潜力，但现有系统大多在各种场景下依赖固定的CE策略（即单步生成、搜索替换和计划应用）。在实践中，CE任务在代码上下文、编辑局部性和语义复杂性方面差异巨大，导致没有任何单一策略能够始终表现良好。此外，关于策略与CE任务之间关系的系统性研究仍然稀缺，限制了基于LLM的代码编辑策略的进一步优化。为填补这一空白，我们构建了CEBench，其中包含从SWE-bench中精选的500个真实世界CE任务。随后我们比较了三种代表性的CE策略（单步生成、搜索替换、计划应用）。我们的研究揭示了两个关键发现：（1）每种策略都针对特定的CE任务展现出独特优势，不存在普遍最优的策略；（2）策略的有效性受到四个核心任务特征的显著影响：总行（摘要在此处被截断）

    arXiv:2610.22881v1 Announce Type: new  Abstract: Large language models have shown strong potential in automating code editing (CE) tasks, yet most existing systems rely on a fixed CE strategy (i.e., one-step, search-replace, and plan-apply) across diverse scenarios. In practice, CE tasks vary widely in code context, edit locality, and semantic complexity, causing no single strategy to perform consistently well. Moreover, systematic studies on the relationship between strategies and CE tasks remain scarce, limiting further optimization of LLM-based code editing strategies. To address this gap, we construct CEBench, comprising 500 real-world CE tasks curated from SWE-bench. Then we compare three representative CE strategies (one-step, search-replace, plan-apply). Our study reveals two key findings: (1) each strategy exhibits distinct advantages for specific CE tasks, with no universal optimality; (2) strategy effectiveness is significantly influenced by four core task features: total lin
    
[^15]: 在剧透与臆测之间：避免编程基准测试中的过度具体化

    Between Spoilers and Guesswork: Avoiding Over-Specification in Coding Benchmarks

    [https://arxiv.org/abs/2610.22760](https://arxiv.org/abs/2610.22760)

    该论文揭示了编程基准测试中通过向任务描述泄露真实答案信息来解决任务描述不足的做法反而会导致过度具体化（剧透）问题，使解决方案绕过基准本应考察的难点，并主张在剧透与臆测之间取得平衡以延长基准的使用寿命。

    

    基准测试推动着人工智能的进步，但却受到任务具体化不足和过度具体化的困扰。任务具体化不足会导致臆测，即解决方案不得不凭空编造，或者更糟糕的是，诉诸数据污染或奖励作弊。任务过度具体化则会导致剧透，即解决方案可以绕过基准本应衡量的一些较难步骤。由于面向未来的基准测试应当具有挑战性，过度具体化会缩短基准在社区中被采用的保质期。仓库级编程基准测试帮助该领域取得了进步，但由于过度具体化和具体化不足的双重问题，它们经历了不必要的频繁更迭。例如，SWE-bench Pro、FEA-bench、Doc2Feat-bench 和 Multi-SWE-bench 都通过使用从真实答案中提取的提示来扩充任务描述，以解决具体化不足的问题。我们表明，将真实答案信息泄露到任务描述中会导致其中一些基准出现过度具体化的问题。我们进行了实验……

    arXiv:2610.22760v1 Announce Type: new  Abstract: Benchmarks drive AI progress, but are hampered by task under-specification and over-specification. Task under-specification causes guesswork, where solutions have to make things up, or worse, resort to contamination or reward hacking. Task over-specification causes spoilers, where solutions can bypass some of the harder steps the benchmark was supposed to measure. Since future-looking benchmarks should be difficult, over-specification reduces a benchmark's shelflife of community adoption. Repository-level coding benchmarks have helped the field progress but experience unnecessary churn due to both over- and under-specification. For instance, SWE-bench Pro, FEA-bench, Doc2Feat-bench, and Multi-SWE-bench all tackle under-specification by augmenting task statements with hints derived from ground truth. We show that leaking ground-truth information into the task statement causes over-specification in some of these benchmarks. We experiment w
    
[^16]: 享受产品却不加入建设者行列：探究学生为何不为开源软件做贡献

    Enjoying the Product but Not Joining the Builders: Investigating Why Students Don't Contribute to OSS

    [https://arxiv.org/abs/2610.22664](https://arxiv.org/abs/2610.22664)

    本研究通过对122名计算专业学生的调查发现，尽管学生频繁使用开源软件，但仅有12.3%积极参与贡献，技能差距、时间限制、代码库复杂性和冒名顶替综合征是主要障碍，而导师指导、新手友好任务和课程激励可能促进参与。

    

    参与开源软件（OSS）为学生提供了接触协作式、真实世界软件开发环境的宝贵机会，然而学生的积极贡献仍然有限。本短文对夏威夷大学马诺阿分校计算专业学生对OSS的认知、知识和参与经历进行了一项探索性研究。我们开展了问卷调查，并采用定量与定性相结合的方法分析了122份回复。学生表示对OSS有中等程度的熟悉度，并在课程作业和个人项目中频繁使用OSS，但仅有12.3%的学生积极做出贡献，与OSS社区的互动也较为有限。主要障碍包括技能差距（占非贡献者的72.89%）、时间限制（59.81%）和代码库复杂性（占贡献者的66.67%），且常伴随冒名顶替综合征（36.45%）。学生指出，更多的时间、导师指导、对初学者友好的任务、更好的入门引导/文档以及课程激励机制（如课程学分……）将有助于提高他们的贡献意愿。

    arXiv:2610.22664v1 Announce Type: new  Abstract: Open-source software (OSS) participation provides students with authentic exposure to collaborative, real-world software development, yet active contributions remain limited. This short paper presents an exploratory study of computing students' perceptions, knowledge, and experiences with OSS at the University of Hawai'i at Manoa. We administered a survey and analyzed 122 responses using both quantitative and qualitative methods. Students reported moderate OSS familiarity and frequent OSS use in coursework and personal projects, yet only 12.3% actively contributed and engagement with OSS communities was limited. Key barriers included skill gaps (72.89% of non-contributors), time constraints (59.81%), and codebase complexity (66.67% of contributors), often alongside imposter syndrome (36.45%). Students indicated that more time, mentorship, beginner-friendly issues, better onboarding/documentation, and curricular incentives (e.g., course c
    
[^17]: SIEVE：一个面向软件工程研究的参数化GitHub语料库构建工具

    SIEVE: A Parameterized GitHub Corpus Builder for Software Engineering Research

    [https://arxiv.org/abs/2610.22544](https://arxiv.org/abs/2610.22544)

    SIEVE是一个开源、参数化的GitHub语料库构建工具，能够在函数、方法和类的代码粒度上，根据编程语言、仓库质量和创建日期等标准提取带标注的代码语料库，为软件工程研究提供了对语料库组成的直接控制能力。

    

    软件工程研究日益依赖于大型且具有丰富标注的真实世界代码语料库。然而，研究人员通常使用临时编写的脚本或静态的预构建数据集来组装这些语料库，无法对语料库的组成进行控制。这就留下了一个空白：需要一种可复用的工具，能够深入到仓库级别以下，按照研究人员指定的标准构建带标注的函数级和类级代码语料库。我们提出了SIEVE（Software Ingestion & Extraction for Verifiable Evaluation，用于可验证评估的软件摄取与提取工具），这是一个开源的、参数化的GitHub语料库构建工具。与仓库级挖掘工具不同，它在代码粒度上工作：它能够在仓库中定位单个函数、方法和类，并将其提取为记录，从而基于多项标准直接控制语料库的构建，这些标准包括广泛使用的编程语言（Python、Java、JavaScript、C++）、仓库质量和创建日期。为了支持软件工程研究……

    arXiv:2610.22544v1 Announce Type: new  Abstract: Software engineering research increasingly relies on large, richly annotated corpora of real-world code. Still, researchers typically assemble these with ad-hoc scripts or static, pre-built datasets that offer no control over corpus composition. This leaves a gap for a reusable tool that goes below the repository level and builds annotated, function- and class-level code corpora under researcher-specified criteria. We present SIEVE (Software Ingestion & Extraction for Verifiable Evaluation), an open-source, parameterized GitHub corpus builder that, unlike repository-level mining tools, works at code granularity: it locates individual functions, methods, and classes in repositories and extracts them as records, giving direct control over corpus construction based on several criteria, including widely used programming languages (Python, Java, JavaScript, C++), repository quality, and creation date. To support software engineering research 
    
[^18]: ForTress：为LLM驱动的Fortran到Python翻译带来保证与确定性

    ForTress: Bringing Assurance and Determinism for LLM-Driven Fortran-to-Python Translation

    [https://arxiv.org/abs/2610.22528](https://arxiv.org/abs/2610.22528)

    ForTress将LLM翻译决策固化为受保护的可执行规则以实现确定性重用，并通过Lean机械化证明为部分规则提供语义保证，从而提升Fortran到Python翻译的可靠性。

    

    将成熟的科学软件从Fortran现代化迁移到Python可以提高可访问性并更好地与现代计算工具集成，但手动重写成本高昂。大型语言模型（LLM）使翻译变得更加实用，然而生成的代码可能在通过回归测试的同时违反源语言的语义，且上下文相关的修正可能无法在后续翻译中持久保留。我们提出ForTress，它将反复出现的LLM翻译决策固化为受保护的可执行规则，以实现确定性重用，并为这些规则中的一个非平凡子集提供语义保证。其LLM辅助的审计过程在Lean中基于机械化的Fortran和Python语义表达全称量化的保持义务。人类专家审查语义保真度以及与生产规则的对应关系，Lean的内核在明确的假设下检查证明与反驳。这些审计记录揭示了支持……（或反驳规则所依赖）的条件。

    arXiv:2610.22528v1 Announce Type: new  Abstract: Modernizing mature scientific software from Fortran to Python can improve accessibility and integration with modern computing tools, but manual rewriting is expensive. Large language models (LLMs) make translation more practical, yet generated code can pass regression tests while violating source-language semantics, and contextual corrections may not persist across subsequent translations. We present ForTress, which solidifies recurring LLM translation decisions into guarded executable rules for deterministic reuse and provides semantic assurance for a non-trivial subset of these rules. Its LLM-assisted auditing process expresses universally quantified preservation obligations over mechanized Fortran and Python semantics in Lean. Human experts review semantic fidelity and correspondence with production rules, and Lean's kernel checks proofs and refutations under explicit assumptions. These audit records expose the conditions either suppo
    
[^19]: 理解由软错误引起的大语言模型幻觉

    Understanding LLM Hallucinations Caused by Soft Errors

    [https://arxiv.org/abs/2610.22520](https://arxiv.org/abs/2610.22520)

    本文提出LLMEI工具在推理过程中向大语言模型注入硬件软错误，首次从硬件层面研究LLM幻觉问题，建立了失败模式分类体系，并评估了软错误在不同架构、任务和错误率下的影响。

    

    虽然大语言模型（LLM）的幻觉已从软件角度得到广泛研究，但对硬件层面因素的关注仍然有限。在本文中，我们进行了一项案例研究，以理解由硬件软错误引起的LLM幻觉。我们引入了LLMEI工具，在推理过程中向LLM注入软错误，并基于此评估了软错误在多种LLM架构和任务（包括数学推理和代码生成）中的影响。我们基于LLM幻觉行为总结了不同失败模式的分类体系，并分析了它们在不同错误率下的分布情况。我们的研究结果强调了在LLM硬件实现的可靠性中针对幻觉的特定诊断的重要性。

    arXiv:2610.22520v1 Announce Type: new  Abstract: While hallucinations in Large Language Models (LLMs) have been extensively studied from a software perspective, limited attention has been paid to hardware-level factors. In this paper, we performed a case study to understand LLM hallucination caused by soft errors on hardware. We introduce LLMEI to inject soft errors into LLMs during inference, based on what we evaluate the impact of soft error across diverse LLM architectures and tasks, including mathematical reasoning and code generation. We summarize the taxonomy of different failure modes based on LLM hallucination behaviors and analyze their distributions under different error rates. Our find- ings underscore the hallucination-specific diagnostics within the reliability of LLM implementation on hardware.
    
[^20]: AuditKIT：一个用于大语言模型与智能体系统基准测试、评判与比较的统一评估主干

    AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems

    [https://arxiv.org/abs/2610.22511](https://arxiv.org/abs/2610.22511)

    AuditKIT是一个统一的评估框架，将基准测试、评判评估、运行时性能、工具使用和检索质量整合到单一结果表示中，解决了大语言模型与智能体系统评估中接口和指标语义不一致的问题。

    

    评估语言模型和智能体需要综合基准测试准确率、基于评判模型的评估、运行时性能、工具使用行为以及检索质量等多方面指标。接口、指标语义和轨迹格式的差异使这些比较变得复杂，并可能掩盖智能体工作流中的失败。我们提出了AuditKIT，一个源码可用的评估库，它将这些测量统一记录在单一的结果表示中，具备方向感知的指标契约、配置指纹以及可扩展的指标和后端。针对智能体，它提供回合感知的工具调用匹配，区分并行与顺序执行，并将经过验证的任务结果与轨迹级诊断区分开来。缺失或不可观察的证据会被标记为“未评分”，而不是被计为失败。针对检索增强生成（RAG），它结合了排序检索指标、评判式的依据性评估以及确定性的证据检查。七个案例研究涵盖模型（比较）……

    arXiv:2610.22511v1 Announce Type: cross  Abstract: Evaluating language models and agents requires combining benchmark accuracy, judge-based assessment, runtime performance, tool-use behavior, and retrieval quality. Differences in interfaces, metric semantics, and trace formats complicate these comparisons and can hide failures in agentic workflows. We present AuditKIT, a source-available evaluation library that records these measurements in one result representation, with direction-aware metric contracts, configuration fingerprints, and extensible metrics and backends. For agents, it provides turn-aware tool-call matching, separates parallel from sequential execution, and distinguishes verified task outcomes from trajectory-level diagnostics. Missing or unobservable evidence is marked unscored rather than counted as failure. For retrieval-augmented generation, it combines ranked retrieval metrics, judged grounding, and deterministic evidence checks. Seven case studies spanning model co
    
[^21]: EnterpriseSynth：基于OpenAPI规范的零执行SFT与评估数据

    EnterpriseSynth: Zero-Execution SFT and Evaluation Data from OpenAPI Specs

    [https://arxiv.org/abs/2610.22412](https://arxiv.org/abs/2610.22412)

    EnterpriseSynth提出了一种零执行框架，无需实际调用企业API即可将OpenAPI规范转换为工具使用型大语言模型智能体的监督微调训练轨迹和成对评估实例，并通过静态验证与对抗性细化保证数据质量。

    

    使用工具的大型语言模型智能体需要大量的训练与评估数据，然而由于访问控制、合规性约束、速率限制以及修改生产数据的风险，从企业API收集执行轨迹通常是不安全、昂贵或不可行的。我们提出了EnterpriseSynth，一个零执行框架，它能够在不调用底层API的情况下，将OpenAPI规范转换为监督微调（SFT）轨迹和成对的评估实例。该框架采用包含规范解析、意图生成、轨迹合成和静态验证的四阶段流水线。每条生成的轨迹在被纳入数据集之前，都会对照源规范检查端点、参数、类型和模式的一致性。我们在17个真实和合成API上对EnterpriseSynth进行了评估。对抗性细化将验证器对植入模式违规的检测率从57-8……（摘要在此处被截断）

    arXiv:2610.22412v1 Announce Type: new  Abstract: Tool-using large language model agents require substantial training and evaluation data, yet collecting execution traces from enterprise APIs is often unsafe, expensive, or infeasible because of access controls, compliance constraints, rate limits, and the risk of modifying production data. We present EnterpriseSynth, a zero-execution framework that converts OpenAPI specifications into supervised fine-tuning trajectories and paired evaluation instances without invoking the underlying APIs. The framework uses a four-stage pipeline for specification parsing, intent generation, trajectory synthesis, and static validation. Each generated trajectory is checked against the source specification for endpoint, parameter, type, and schema consistency before being admitted to the dataset. We evaluate EnterpriseSynth on 17 real and synthetic APIs. Adversarial refinement increases the validator's detection rate for planted schema violations from 57-8
    
[^22]: 引出规格说明：交互式澄清何时有所帮助？

    Eliciting Specifications: When Does Interactive Clarification Help?

    [https://arxiv.org/abs/2610.22382](https://arxiv.org/abs/2610.22382)

    本文提出一种引出代理（EA）方法，在形式化之前向模拟开发者提出有预算限制的澄清问题以改进LLM生成的Lean规格说明，并系统评估了在不同问题预算（k=3与k=10）及同模型/跨模型配置下交互式澄清何时有效。

    

    LLM可以从任务描述中生成Lean规格说明，但这些描述往往不够明确或存在歧义，并且可能遗漏与验证器相关的边界情况，从而产生不完善的形式化规格说明。我们研究了一个引出代理（Elicitation Agent, EA）能否通过在形式化之前向基于参考资料构建的模拟开发者寻求澄清，来改进经过验证器检查的Lean规格说明。该代理对任务进行分解，识别尚未解决的决策点，提出有数量限制的澄清问题，并利用答案撰写精炼后的自然语言规格说明，再由LLM形式化工具将其翻译为Lean。在189个Verina Lean 4规格说明生成任务上，我们比较了单次生成与引出式方法的效果，问题预算分别设为k=3和k=10（即每个任务最多允许EA提出三个或十个问题），并涵盖了同模型与跨模型两种配置。同模型的Claude Sonnet 4.6流水线在两种……（原文摘要在此处被截断）

    arXiv:2610.22382v1 Announce Type: cross  Abstract: LLMs can generate Lean specifications from task descriptions, but those descriptions are often underspecified or ambiguous and may omit verifier-relevant edge cases, yielding imperfect formal specifications. We study whether an elicitation agent (EA) can improve verifier-checked Lean specifications by seeking clarification from a reference-grounded simulated developer before formalization. The agent decomposes the task, identifies unresolved decisions, asks bounded clarification questions, and uses the answers to write a refined natural-language specification that an LLM formalizer then translates into Lean. On 189 Verina Lean 4 specification-generation tasks, we compare single-shot generation with elicitation under question budgets of k=3 and k=10, allowing at most three or ten EA questions per task, across same-model and cross-model configurations. The same-model Claude Sonnet 4.6 pipeline has negative strict pass@1 estimates at both
    
[^23]: 我们是否在测试错误的量子缺陷？量子软件测试中缺陷表示差距的研究

    Are We Testing the Wrong Quantum Bugs? A Study of Fault Representation Gaps in Quantum Software Testing

    [https://arxiv.org/abs/2610.22089](https://arxiv.org/abs/2610.22089)

    该研究通过审查量子变异测试工具和人工整理的缺陷基准，并收集两年内410个真实缺陷，揭示量子软件测试评估中存在显著的缺陷表示差距——现有变异算子与基准仅覆盖极少数缺陷族，无法充分代表真实量子软件中的缺陷。

    

    量子软件依赖编译器、中间表示和执行框架，这些组件会在程序执行前对其进行转换。这些组件中的缺陷可能会改变原本正确的量子程序的行为。然而，量子测试技术通常使用变异生成的缺陷和人工整理的缺陷基准来评估。我们研究了这些工件所表示的缺陷能在多大程度上反映现代量子软件中实际遇到的缺陷。我们审查了四个量子变异测试工具和三个人工整理的缺陷基准，并在两年内从十个量子软件仓库中收集了410个已验证的缺陷。我们将来自这三个来源的缺陷统一归类到一个包含56个缺陷族的分类体系中，并比较了它们在变异算子、人工整理基准和近期真实世界缺陷中的覆盖情况。变异测试工具支持20个缺陷族，人工整理的基准又增加了15个。然而，410个真实世界缺陷中仅有56个（13.7%）属于……（原文摘要在此处截断）

    arXiv:2610.22089v1 Announce Type: new  Abstract: Quantum software relies on compilers, intermediate representations, and execution frameworks that transform programs before execution. Faults in these components can change the behavior of otherwise correct quantum programs. Yet quantum testing techniques are often evaluated using mutation-generated faults and curated bug benchmarks. We investigate how well the faults represented by these artifacts reflect those encountered in modern quantum software. We audit four quantum mutation-testing tools and three curated bug benchmarks, and collect 410 verified bugs from ten quantum software repositories over two years. We classify faults from all three sources into a common taxonomy of 56 bug families and compare their representation across mutation operators, curated benchmarks, and recent real-world bugs. The mutation tools support 20 bug families, and curated benchmarks add 15 more. Yet only 56 of the 410 real-world bugs (13.7%) belong to fa
    
[^24]: WEAVE：将真实世界代码演进编织为依赖结构化任务，用于代码智能体的基准测试与训练

    WEAVE: Weaving Real-World Code Evolution into Dependency-Structured Tasks for Benchmarking and Training Code Agents

    [https://arxiv.org/abs/2610.22039](https://arxiv.org/abs/2610.22039)

    WEAVE 框架将真实世界中经过验证的多 PR 代码演进历史转化为依赖结构化的可执行任务，并引入基于测试的确定性度量 Graph Credit，以更好地对长程代码智能体进行基准评估和训练。

    

    真实世界的软件工程很少由孤立的变更构成。功能开发、迁移和系统性修复往往跨越多个拉取请求（PR）展开，需要对整个代码库进行协调一致的修改。然而，大多数可执行的代码库级基准测试仍以单个 issue 或 PR 作为任务单元，而现有的多 PR 设置则依赖粗粒度的聚合和与结构无关的评估。这种错位掩盖了变更之间的依赖关系，限制了长程代码智能体的进展衡量和学习信号。我们提出了 WEAVE，一个将经过验证的 PR 历史转化为可执行的、依赖结构化任务的框架。WEAVE 利用代码库校准的类型化变更位点，将相关联的 PR 组织成有界 DAG（有向无环图），并将每个 DAG 实例化为统一的可执行任务。我们进一步引入了 Graph Credit，这是一种确定性的、基于测试的度量指标，用于衡量符合依赖关系的部分进展，并支持更密集的……

    arXiv:2610.22039v1 Announce Type: cross  Abstract: Real-world software engineering rarely consists of isolated changes. Features, migrations, and systematic fixes often unfold across multiple pull requests (PRs), requiring coordinated modifications across a repository. Yet most executable repository-level benchmarks retain a single issue or PR as the task unit, while existing multi-PR settings rely on coarse aggregation and structure-agnostic evaluation. This mismatch obscures dependencies among changes and limits progress measurement and learning signals for long-horizon code agents. We introduce WEAVE, a framework that transforms verified PR histories into executable, dependency-structured tasks. WEAVE leverages repository-calibrated typed change loci to organize related PRs into bounded DAGs and instantiate each as a unified executable task. We further introduce Graph Credit, a deterministic, test-grounded metric that measures dependency-consistent partial progress and supports dens
    
[^25]: 基准测试与增强Linux内核中的设计感知代码审查

    Benchmarking and Enhancing Design-Aware Code Review in the Linux Kernel

    [https://arxiv.org/abs/2610.21925](https://arxiv.org/abs/2610.21925)

    本文提出了DesignLore知识增强方法和DesignReviewBench基准，通过从历史审查讨论中提取、组织并复用设计决策知识，系统性评估并增强了Linux内核中的设计感知代码审查。

    

    代码审查往往需要超越局部正确性的推理。在成熟的软件系统中，审查者必须识别所提议变更引入的设计权衡，比较备选解决方案，并推理其与项目特定约束的兼容性。我们将这一问题称为“设计感知代码审查”。然而，现有的自动化审查方法主要关注缺陷和正确性，而历史审查讨论中积累的设计知识仍然难以复用。为了应对这些挑战，我们提出了DesignLore，这是一种知识增强方法，它从历史审查讨论中提取设计决策，将其组织成自适应的设计分类体系，并复用相关的历史知识来指导代码审查。我们进一步构建了DesignReviewBench，这是一个源自真实补丁审查历史的基准，用于系统地评估设计感知代码审查。我们的评估表明……

    arXiv:2610.21925v1 Announce Type: new  Abstract: Code review often requires reasoning beyond local correctness. In mature software systems, reviewers must recognize design trade-offs introduced by a proposed change, compare alternative solutions, and reason about their compatibility with project-specific constraints. We refer to this problem as design-aware code review. However, existing automated review approaches primarily focus on defects and correctness, while the design knowledge accumulated in historical review discussions remains difficult to reuse.   To address these challenges, we present DesignLore, a knowledge-augmented approach that extracts design decisions from historical review discussions, organizes them into an adaptive design taxonomy, and reuses relevant historical knowledge to guide code review. We further construct DesignReviewBench, a benchmark derived from real patch-review histories for systematically evaluating design-aware code review. Our evaluation shows tha
    
[^26]: 递归游戏创造者：一个面向体验的智能体级产品游戏开发框架

    Recursive Game Creator: An Agentic Product-Level Experience-Oriented Game Harness

    [https://arxiv.org/abs/2610.08621](https://arxiv.org/abs/2610.08621)

    该论文提出递归游戏创造者框架，通过设计者、构建者、玩家和评审者四个智能体的递归协作，将粗糙的游戏原型迭代开发为真正注重玩家体验的有趣游戏。

    

    近期的游戏设计智能体在生成可玩游戏方面取得了长足进步。然而，程序的正确性并不能保证玩家获得愉快的游戏体验。我们提出了递归游戏创造者，这是一个面向体验的框架，旨在将智能体游戏开发从粗糙的游戏原型推进为有趣的游戏。递归游戏创造者围绕四个组件组织递归式开发：设计者、构建者、玩家和评审者。设计者将用户指令和评审者的反馈转化为详细的计划。构建者将这些计划转化为候选游戏。基于代码原生的玩家通过编程接口创建并执行可复用的策略，以高效收集多样化的游戏玩法轨迹，缓解了基于图形界面（GUI）的缓慢收集方式所导致的评估偏差。评审者使用精心设计的基于轨迹的指标来推断玩家偏好，并结合视觉证据和明确的文本偏好（进行综合评估）。

    arXiv:2610.08621v1 Announce Type: new  Abstract: Recent game design agents have made substantial progress in generating playable games. However, program correctness does not ensure an enjoyable experience for players. We present Recursive Game Creator, an experience-oriented harness to advance agentic game development from rough game prototypes into entertaining games. Recursive Game Creator organizes recursive development around four components: Designer, Builder, Player, and Reviewer. The Designer translates user instructions and Reviewer's feedback into detailed plans. The Builder turns these plans into candidate games. The coding-native Player creates and executes reusable policies through programmatic interfaces to efficiently collect diverse gameplay trajectories, mitigating evaluation bias caused by slow GUI-based collection. The Reviewer uses carefully designed trajectory-based metrics to induce player preferences, integrating with visual evidence and explicit textual preferenc
    
[^27]: 信任不是分数：面向高风险AI智能体的运行时保障契约

    Trust Is Not a Score: Runtime Assurance Contracts for High-Risk AI Agents

    [https://arxiv.org/abs/2609.39717](https://arxiv.org/abs/2609.39717)

    提出了运行时保障契约（RAC），用非补偿性强制门控而非综合分数来决定高风险AI智能体在关键任务中的权限动态转换，填补了观察证据与智能体授权之间的保障转换缺口。

    

    基准测试、审计和智能体协议描述了性能、权限和修复机制，但并未说明观察到的证据应如何在重大任务中改变智能体的权限。我们将这一现象称为保障转换缺口。我们提出了运行时保障契约（RAC），这是一种策略级的形式化模式，将自主性边界、组件资格、证据状态、转换策略、人工审查容量以及非补偿性门控绑定在一起。在RAC下，软性指标可以用于路由决策，而失败或状态未知的强制门控则会强制触发重试、切换、升级、延迟或停止；总体性能不能作为行动授权。我们定义了契约、证据记录、权限规则以及五个不变量，并在临床、工业和司法领域的失败探测案例中加以说明。随后我们报告了一项针对智能体编程的确定性故障注入研究：280个构造案例分别由门控合取规则、仅分数规则和受限的…进行评估。

    arXiv:2609.39717v1 Announce Type: new  Abstract: Benchmarks, audits, and agent protocols describe performance, permissions, and repair, but not how observed evidence should change an agent's authority during a consequential task. We call this the assurance-transition gap. We propose a Runtime Assurance Contract (RAC), a policy-level formal schema binding autonomy boundaries, component eligibility, evidence state, transition policy, human-review capacity, and non-compensatory gates. Under RAC, soft metrics may inform routing, whereas a failed or unknown mandatory gate forces retry, switch, escalation, deferral, or stop; aggregate performance cannot authorize action. We define the contract, an evidence record, a permission rule, and five invariants, and illustrate them in clinical, industrial, and judicial failure probes. We then report a deterministic failure-injection study in agentic coding: 280 constructed cases evaluated by a gate conjunction, a score-only rule, and a restricted pro
    
[^28]: SWE-Prometheus：衡量真实世界代码仓库中的工程治理改进

    SWE-Prometheus: Measuring Engineering Governance Improvements in Real-World Repositories

    [https://arxiv.org/abs/2609.29465](https://arxiv.org/abs/2609.29465)

    该论文提出了SWE-Prometheus基准，首次系统评估大语言模型编程智能体在开放式仓库工程治理任务中的能力，要求智能体自主识别风险、排序干预优先级并验证变更，通过六个治理维度和多重验证机制对十个模型进行了评测。

    

    基于大语言模型的编程智能体在仓库级软件工程任务上已取得显著进展。然而，现有的仓库基准测试通常从一个由人类确定的问题出发，评估补丁是否满足某个功能性信号。我们提出了SWE-Prometheus，一个针对更广泛任务——改进代码仓库工程治理——的基准测试。每个任务提供一个固定的代码快照和一个开放式的目标，要求智能体识别风险、确定干预措施的优先级并验证由此产生的变更。SWE-Prometheus通过配对证据、干净环境探测、行为门控以及对同一证据的两项独立教师评分，来评估六个治理维度。该基准包含60个仓库；十个模型在一个共享的22仓库公共子集上进行评估，其中平均归一化治理改进得分介于0.0568到0.5760之间，观察到的行为破坏率介于0%到23%之间。

    arXiv:2609.29465v1 Announce Type: new  Abstract: Large language model based coding agents have made substantial progress on repository-level software engineering tasks. Existing repository benchmarks, however, usually start from a human-identified issue and evaluate whether a patch satisfies a functional signal. We present SWE-Prometheus, a benchmark for the broader task of improving repository engineering governance. Each task provides a fixed snapshot and an open-ended objective, requiring the agent to identify risks, prioritize interventions, and verify the resulting changes. SWE-Prometheus evaluates six governance dimensions through paired evidence, clean-environment probes, behavior gates, and two independent teacher ratings of the same evidence. The benchmark contains 60 repositories; ten models are evaluated on a shared 22-repository public subset, where mean Normalized Governance Improvement ranges from 0.0568 to 0.5760 and observed behavior-breakage rates range from 0% to 23%.
    
[^29]: 迈向智能体化云工程：基于零信任智能体套件的图工程与循环工程

    Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness

    [https://arxiv.org/abs/2609.00050](https://arxiv.org/abs/2609.00050)

    提出了一个智能体云工作流工程框架，通过将图工程（长时程工作流推进）、循环工程（有界诊断与修复重试）和零信任智能体套件（受限执行）三个关注点分离，将自然语言云工程任务自动转化为经过验证的代码仓库和可验证的云部署。

    

    智能体AI正在推动基于云的工作流的发展，其中自主智能体可以对运营状态进行推理、调用授权工具、修改软件和基础设施、部署服务、验证执行结果，并在长时程、多步骤任务中进行自适应调整。构建此类工作流需要针对工作流推进、受限执行、故障恢复和可验证完成等环节的显式机制。我们提出了智能体云工作流工程，这是一个智能体AI框架，它将自然语言描述的智能体云工程任务转化为经过验证的代码仓库和经过验证的运营性云部署，从而实现基于云的智能体工作流自动化。该框架分离了三个互补的关注点：图工程负责指定长时程工作流推进以及依赖验证的状态转移；循环工程提供有界的诊断、修复或重新规划、重试和重新验证；智能体套件工程则（负责执行零信任的受限执行控制）。

    arXiv:2609.00050v1 Announce Type: cross  Abstract: Agentic AI is enabling cloud-based workflows in which autonomous agents reason over operational state, invoke authorized tools, modify software and infrastructure, deploy services, verify execution outcomes, and adapt across long-horizon, multistep tasks. Engineering such workflows requires explicit mechanisms for workflow progression, constrained execution, failure recovery, and verifiable completion. We present Agentic Cloud Workflow Engineering, an agentic AI framework that transforms natural-language agentic cloud-engineering tasks into validated code repositories and verified operational cloud deployments for automating cloud-based agentic workflows. The framework separates three complementary concerns: graph engineering specifies long-horizon workflow progression and verification-dependent transitions; loop engineering provides bounded diagnosis, repair or re-planning, retry, and re-verification; and agent harness engineering enf
    
[^30]: CIDR：一个用于软件工程研究的大规模工业源代码数据集

    CIDR: A Large-Scale Industrial Source Code Dataset for Software Engineering Research

    [https://arxiv.org/abs/2605.12153](https://arxiv.org/abs/2605.12153)

    本文发布了CIDR——一个包含4,687个仓库、84种编程语言、超10亿行代码的大规模匿名化工业源代码数据集，为代码智能和软件工程研究提供真实企业级代码资源。

    

    我们提出了精选工业开发者仓库数据集（CIDR），这是一个从工业合作伙伴处收集的大规模真实世界软件仓库数据集。该数据集包含4,687个仓库，涵盖84种编程语言，总计10.33亿行原始代码（6.703亿行逻辑代码），同时提供仓库级别的结构化元数据、完整的版本控制历史，以及持续集成使用情况和自动化测试存在与否等工程实践属性。所有仓库均通过专门为此目的开发的多阶段流水线进行收集、过滤和匿名化处理。我们还报告了一项探索性微调研究，将一个30亿参数的代码语言模型适配到CIDR上，并量化了其对留出的企业代码的影响。CIDR旨在支持代码智能、软件质量分析、开发者工具及相关软件工程领域的研究。

    arXiv:2605.12153v3 Announce Type: replace-cross  Abstract: We present the Curated Industrial Developer Repository (CIDR), a large-scale dataset of real-world software repositories collected from industrial partners. The dataset comprises 4,687 repositories spanning 84 programming languages, totaling 1,033 million raw lines of code (670,3 million logical lines), along with structured metadata at the repository level, full version control history, and engineering-practice attributes such as continuous integration usage and the presence of automated tests. All repositories were collected, filtered, and anonymized through a multi-stage pipeline developed specifically for this purpose. We additionally report an exploratory fine-tuning study that adapts a 3-billion-parameter code language model to CIDR and quantifies the effect on held-out enterprise code. CIDR is intended to support research in code intelligence, software quality analysis, developer tooling, and related software engineering
    
[^31]: 认识论债务：人工智能依赖的软件工程中的认知萎缩与系统性崩溃

    Epistemological Debt: Cognitive Atrophy and Systemic Collapse in AI-Dependent Software Engineering

    [https://arxiv.org/abs/2604.26855](https://arxiv.org/abs/2604.26855)

    本文提出“认识论债务”概念，指出AI依赖的软件开发会导致认知萎缩和系统性崩溃，并主张通过人机协同教学标准来维持工程韧性。

    

    大型语言模型（LLMs）融入软件开发生命周期（SDLC）掩盖了一个关键的社会技术失败：认知-系统性崩溃。本文引入了“认识论债务”，即当工程师用被动的AI验证取代逻辑推导时所承担的隐藏携带成本。这种债务侵蚀了根本原因分析所必需的心智模型，扩大了系统复杂性与人类理解之间的差距。此外，对合成代码的递归训练威胁到全球软件资源的同质化，减少了稳健工程所需的多样性。以2026年亚马逊中断事件为案例研究，本研究展示了“机械化趋同”如何导致系统性脆弱。为保持长期韧性，工程领导者必须超越基于提示的开发，实施严格的人机协同教学标准。该框架平衡了AI驱动生产力与人类认知参与。

    arXiv:2604.26855v3 Announce Type: replace  Abstract: The integration of Large Language Models (LLMs) into the software development lifecycle (SDLC) masks a critical socio-technical failure: Cognitive-Systemic Collapse. This paper introduces "Epistemological Debt," the hidden carrying cost incurred when engineers substitute logical derivation with passive AI verification. This debt erodes the mental models essential for root-cause analysis, widening the gap between system complexity and human comprehension. Furthermore, recursive training on synthetic code threatens to homogenize the global software reservoir, diminishing the variance required for robust engineering. Using the 2026 Amazon outages as a case study, this research illustrates how "mechanized convergence" leads to systemic fragility. To preserve long-term resilience, engineering leaders must move beyond prompt-based development to implement rigorous human-in-the-loop pedagogical standards. This framework balances AI-driven p
    
[^32]: JavaScript项目中依赖项重新分类的纵向研究

    A Longitudinal Study of Dependency Reclassifications in JavaScript Projects

    [https://arxiv.org/abs/2604.08747](https://arxiv.org/abs/2604.08747)

    本研究通过对33,087个JavaScript项目package.json文件的提交级分析，首次系统揭示了依赖项重新分类（包括移除和角色重分配）是一种普遍的维护活动，存在于79.1%的项目中，占所有依赖项维护提交的19.4%。

    

    现代软件项目依赖第三方依赖项，随着项目的演进，这些依赖项的声明必须得到持续维护。以往的研究主要关注依赖项的版本更新，而对于开发者如何随时间推移将依赖项分配到不同角色的研究则相对匮乏。在本文中，我们研究了JavaScript项目的开发者如何对其依赖项进行重新分类，包括移除依赖项和重新分配其角色。通过分析package.json文件在提交级别的修改，我们重建了依赖项角色变更历史，并识别出反复出现的重新分类实践。我们对33,087个积极维护依赖项的JavaScript项目的分析表明，依赖项重新分类是一种普遍存在的维护活动，出现在79.1%的所研究项目中，占所有依赖项维护提交的19.4%。在这些项目中，几乎所有项目（97.2%）都会在某些时候移除依赖项，而38.0%的项目经历了依赖项角色的重新分配。

    arXiv:2604.08747v2 Announce Type: replace  Abstract: Modern software projects depend on third-party dependencies, whose declarations must be maintained as projects evolve. Prior work has focused on dependency version updates, while much less is known about how developers assign dependencies to different roles over time. In this paper, we investigate how developers of JavaScript projects reclassify their dependencies, including removal and role reassignment. By analyzing commit-level modifications to package.json files, we reconstruct dependency role histories and identify recurring reclassification practices. Our analysis of 33,087 JavaScript projects with active dependency maintenance reveals that dependency reclassification is a prevalent maintenance activity, occurring in 79.1% of the studied projects, and accounting for 19.4% of all dependency-maintenance commits. Of these projects, nearly all (97.2%) remove dependencies at some point, while 38.0% undergo role reassignments across 
    
[^33]: 超集反编译

    Superset Decompilation

    [https://arxiv.org/abs/2603.28002](https://arxiv.org/abs/2603.28002)

    该论文提出“超集反编译”新范式：用单调 Datalog 规则在 IR 层次结构中保留多种候选解释、以函数级加权 MaxSMT 求解器统一裁决，从而克服传统反编译器因过早确定单一解释而无法对真实程序生成可编译输出的问题。

    

    反编译是从编译后的二进制文件中恢复高级源代码的任务。传统上，反编译被实现为在单一表示上进行的一系列分析和重写，每一步都只确定一种解释并丢弃其余的。我们认为，这种过早的承诺正是反编译器难以在完整的真实程序上生成可编译输出的原因，而可编译输出对于后续测试是必要的。编译是多对一的：同一个二进制文件可能对应多个源程序，而用于消解歧义的证据往往出现在距离歧义产生处很远的地方。因此，当竞争性解释的消解依赖于非局部证据时，我们选择保留这些解释，而将纯局部性的歧义留到最终的求解阶段处理。我们将该方法形式化为超集反编译：通过单调 Datalog 规则在 IR（中间表示）层次结构中不断积累候选解释，再由针对每个函数的加权 MaxSMT 求解器联合选择……

    arXiv:2603.28002v2 Announce Type: replace-cross  Abstract: Decompilation is the task of recovering high-level source from a compiled binary. Conventionally, it is implemented as a series of analyses and rewrites over a single representation, committing at each step to a single interpretation and discarding the rest. We argue this premature commitment is why decompilers struggle to produce compilable output on whole real-world programs, which is necessary for further testing. Compilation is many-to-one: a single binary may admit many source programs, and the evidence that resolves an ambiguity often lies far from where it arises. We therefore preserve competing interpretations when their resolution depends on non-local evidence, while resolving purely local ambiguities in a final solving stage. We formalize this approach as Superset Decompilation: monotone Datalog rules accumulate candidate interpretations across an IR hierarchy, and a per-function weighted MaxSMT solver jointly selects
    
[^34]: 测试驱动代码生成的理论分析

    A Theoretical Analysis of Test-Driven Code Generation

    [https://arxiv.org/abs/2602.06098](https://arxiv.org/abs/2602.06098)

    本文建立了测试驱动代码生成的概率理论框架，证明了基于模糊功能相似性的代码选择估计器在信噪比上严格优于基于功能等价性的估计器，并将回溯提示形式化为汤普森采样的上下文内近似，推导出新的遗憾界以解释其有效性受限于歧义性的原因。

    

    代码助手在测试驱动的软件开发中的应用日益广泛，但其环境交互策略背后的理论机制仍然缺乏深入研究。我们为两种主流范式提供了概率框架：利用执行环境在生成后进行代码选择，以及以环境反馈为条件的代码生成。首先，我们将若干成熟的选择启发式方法形式化为环境感知的代码正确性估计器。我们从理论上证明，基于模糊功能相似性的估计器引入了归纳偏置，并且在信噪比方面严格优于基于功能等价性的估计器。其次，我们将回溯提示（backprompting）框架化为汤普森采样的上下文内近似。我们为包含不可观测成分的奖励函数推导出了一个新的遗憾界，从理论上解释了为什么回溯提示的有效性受限于歧义性。

    arXiv:2602.06098v4 Announce Type: replace-cross  Abstract: Code assistants are increasingly utilized in test-driven software development, yet the theoretical mechanisms behind their environment-interaction strategies remain underexplored. We provide a probabilistic framework for two dominant paradigms: code selection after generation using the execution environment, and code generation conditioned on environment feedback. First, we formalize several well-established selection heuristics as environment-aware estimators of code correctness. We theoretically prove that estimators based on fuzzy functional similarity add an inductive bias and strictly dominate estimators based on functional equivalence in terms of signal-to-noise ratio. Second, we frame backprompting as an in-context approximation of Thompson sampling. We derive a novel regret bound for reward functions with unobservable components, theoretically explaining why the effectiveness of backprompting is limited by the ambiguity
    
[^35]: Veri-Sure：基于时序追踪、切片与形式化验证的多智能体RTL代码生成

    Veri-Sure: Multi-Agent RTL Code Generation with Temporal Tracing, Slicing and Formal Verification

    [https://arxiv.org/abs/2601.19747](https://arxiv.org/abs/2601.19747)

    Veri-Sure是一个多智能体RTL代码生成框架，通过设计契约对齐智能体意图、静态依赖切片引导的局部化补丁修复以及形式化验证与时序追踪，解决了LLM生成RTL代码中反馈稀疏、整文件重生成引发回归和语义漂移三大瓶颈。

    

    在快速发展的电子设计自动化（EDA）领域中，部署大语言模型（LLM）进行寄存器传输级（RTL）代码生成已成为一个有前景的方向。然而，由于RTL代码描述的是具有严格周期级时序的并发硬件行为，与软件代码有着本质区别，基于LLM的RTL代码生成仍受制于以下瓶颈：（i）以仿真为中心的评估方法反馈稀疏且测试覆盖有限；（ii）调试过程中的整文件重新生成引入了回归问题和修复幻觉；（iii）随着意图在修复迭代和智能体交接过程中被反复重新解释而产生语义漂移。在本工作中，我们提出了Veri-Sure，这是一个多智能体框架，它通过建立设计契约来对齐各智能体的意图，并采用由静态依赖切片引导的补丁机制来执行精确的局部化修复。通过集成多分支验证……

    arXiv:2601.19747v2 Announce Type: replace-cross  Abstract: In the rapidly evolving field of Electronic Design Automation (EDA), the deployment of Large Language Models (LLMs) for Register-Transfer Level (RTL) code generation has emerged as a promising direction. However, because RTL code describes concurrent hardware behavior with strict cycle-level timing and thus differs fundamentally from software code, LLM-based RTL code generation remains bottlenecked by (i) sparse feedback and limited test coverage of simulation-centric evaluation, (ii) regressions and repair hallucinations introduced by whole-file regeneration during debugging, and (iii) semantic drift as intent is reinterpreted across repair iterations and agent handoffs. In this work, we propose Veri-Sure, a multi-agent framework that establishes a design contract to align agents' intent and uses a patching mechanism guided by static dependency slicing to perform precise, localized repairs. By integrating a multi-branch verifi
    
[^36]: FeatureFuzz：将易错特性作为编译器模糊测试的核心原语

    FeatureFuzz: Treating Bug-prone Features as Core Primitives for Compiler Fuzzing

    [https://arxiv.org/abs/2601.12360](https://arxiv.org/abs/2601.12360)

    提出FeatureFuzz，一种将从历史缺陷中提取的易错语义显式建模为特性和特性组并加以重用的编译器模糊测试工具，从而更有效地发现依赖程序语义的编译器缺陷。

    

    CI/CD流水线通常以高权限执行构建和测试任务，这使得编译阶段具有安全敏感性。在这种环境下，恶意拉取请求可能利用精心构造的源代码在编译过程中触发任意代码执行或过度的资源消耗，从而使编译器漏洞具有安全相关性。近期的工作利用从历史缺陷中提取的易错知识来驱动编译器探索与缺陷发现。然而，许多编译器缺陷依赖于程序语义而非语法，现有方法要么依赖语法变异，要么将此类知识编码在模型参数中，难以保留并重组真正触发编译器缺陷的语义。我们提出FeatureFuzz，一种通过特性和特性组显式重用易错语义的编译器模糊测试工具。FeatureFuzz从历史缺陷中提取这些易错语义作为特性，并对其实施……

    arXiv:2601.12360v3 Announce Type: replace  Abstract: CI/CD pipelines often execute build and test tasks with high privileges, making the compilation stage security-sensitive. In this setting, malicious pull requests may exploit carefully crafted source code to trigger arbitrary code execution or excessive resource consumption during compilation, making compiler vulnerabilities security-relevant. Recent work has leveraged bug-prone knowledge from historical bugs to drive compiler exploration and bug discovery. However, many compiler bugs depend on program semantics rather than syntax, and existing approaches either rely on syntactic mutation or encode this knowledge in model parameters, making it difficult to preserve and recombine the semantics that actually trigger compiler bugs. We propose FeatureFuzz, a compiler fuzzer that explicitly reuses bug-prone semantics through features and feature groups. FeatureFuzz extracts these bug-prone semantics from historical bugs as features, synth
    
[^37]: Pluvio:通过迁移学习和条件变分信息瓶颈实现对领域外结构和库的汇编克隆搜索

    Pluvio: Assembly Clone Search for Out-of-domain Architectures and Libraries through Transfer Learning and Conditional Variational Information Bottleneck. (arXiv:2307.10631v1 [cs.SE])

    [http://arxiv.org/abs/2307.10631](http://arxiv.org/abs/2307.10631)

    本论文介绍了一种通过迁移学习和条件变分信息瓶颈实现对领域外结构和库的汇编克隆搜索的方法。已有的方法往往限制于少数训练过的工具链变体，而本研究首次解决了未见过架构和库的情况，提出了一种新的解决方案。

    

    代码重用在软件开发中至关重要，可以加快和提高开发生命周期的效率。然而，实际中，代码重用存在着控制不当的问题，导致漏洞传播和知识产权侵权等问题。汇编克隆搜索是一种重要的右移防御机制，已经在识别由已发布的可执行文件中的代码重用导致的漏洞方面取得了良好效果。最近关于汇编克隆搜索的研究表明，越来越多采用基于机器学习的方法来匹配由不同工具链生成的汇编代码变体。然而，这些方法仅限于从训练中学习到的少数工具链变体，无法应用于未见过的架构及其相应的编译工具链变体。本文首次研究了在未见过的架构和库中的汇编克隆搜索问题。我们提议通过人类共有知识来解决这个问题。

    The practice of code reuse is crucial in software development for a faster and more efficient development lifecycle. In reality, however, code reuse practices lack proper control, resulting in issues such as vulnerability propagation and intellectual property infringements. Assembly clone search, a critical shift-right defence mechanism, has been effective in identifying vulnerable code resulting from reuse in released executables. Recent studies on assembly clone search demonstrate a trend towards using machine learning-based methods to match assembly code variants produced by different toolchains. However, these methods are limited to what they learn from a small number of toolchain variants used in training, rendering them inapplicable to unseen architectures and their corresponding compilation toolchain variants.  This paper presents the first study on the problem of assembly clone search with unseen architectures and libraries. We propose incorporating human common knowledge throu
    

