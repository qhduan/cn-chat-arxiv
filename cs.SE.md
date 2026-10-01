# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automatically Building Machine-Checked Assurance Cases from C Codebases to Requirements](https://arxiv.org/abs/2609.40119) | 本文提出了LLM辅助框架CCV，能够从C代码库自动构建机器可验证的保证案例，通过接口协议建模允许的调用序列并协调需求引导分析与自底向上验证两个阶段，从而证明代码库满足预期需求。 |
| [^2] | [Leveraging Game-Based Platform to Teach Code Refactoring: An Experience with Refactoria](https://arxiv.org/abs/2609.40086) | 本文提出了 Refactoria——一款让玩家扮演专家厨师、协助机器人伙伴将“烹饪指令”重构为高质量代码的游戏化教学工具，并通过 30 名学生开发者的课堂实验验证了其在代码坏味道与重构学习中的感知、有效性和有用性。 |
| [^3] | [Learning When and How to Intervene: A Hindsight-Distilled Sentinel for Coding Agents](https://arxiv.org/abs/2609.39957) | 提出了HiSentinel框架，通过后见之明蒸馏方法训练轻量级哨兵模型，在编码代理执行动作之前判断是否以及如何进行干预并提供可操作反馈，从而提升任务完成率而非简单纠正每个错误。 |
| [^4] | [DoGBench: Can Agents Meet Expert Standards for User-Facing Documentation?](https://arxiv.org/abs/2609.39909) | DoGBench是首个评估智能体能否生成达到技术写作专家评审标准的真实面向用户软件文档的基准，包含来自开源项目的292个条目，要求智能体判断文档是否需要更新、一次性产出合格补丁或正确弃权。 |
| [^5] | [COMPASS: Predicting the Relationship of Multiple Patches for Vulnerabilities with LLMs](https://arxiv.org/abs/2609.39783) | COMPASS基于对约1000个真实多补丁漏洞的人工分析和开发者访谈总结出六种典型补丁关系，并利用大语言模型自动预测漏洞多个补丁之间的关系，从而帮助下游代码库更有效地采用上游漏洞修复。 |
| [^6] | [Trust Is Not a Score: Runtime Assurance Contracts for High-Risk AI Agents](https://arxiv.org/abs/2609.39717) | 提出了运行时保障契约（RAC），用非补偿性强制门控而非综合分数来决定高风险AI智能体在关键任务中的权限动态转换，填补了观察证据与智能体授权之间的保障转换缺口。 |
| [^7] | [Aletheia: Permission-Minimality Testing for Coding-Agent Rules](https://arxiv.org/abs/2609.39678) | Aletheia 提出了一个针对编码代理规则的最小权限测试框架，通过在逐一移除权限的沙箱限制下验证任务仍能通过功能测试来生成“可省略性见证”，从而有效检测并诊断隐藏在仓库指令文件中的提示注入与恶意权限请求。 |
| [^8] | [Self-Spec Verifiable Code Generation](https://arxiv.org/abs/2609.39568) | 该论文提出了VeriCodeBench基准，用于评估大语言模型在自规范可验证代码生成上的端到端能力，克服了现有基准分阶段评估且任务覆盖范围有限的两大局限。 |
| [^9] | [EngramBench: A Capability-Grounded Benchmark for Skill-Evolution Harnesses](https://arxiv.org/abs/2609.39284) | EngramBench提出了一个基于“能力重叠但解决方案不重叠”公理的基准测试，通过30个学习任务、13个迁移任务以及多小时交互式开发周期，有效区分自主智能体的真正能力抽象与训练数据泄露问题。 |
| [^10] | [Trustworthy Runtime Error Healing in Real-World Repositories: A Benchmark and Guardrail](https://arxiv.org/abs/2609.39086) | 本文提出HealBench基准（涵盖18个真实世界仓库的265个运行时错误）和HealGuard防护机制，通过限制修复代码使用可分析的Python子集并结合污点分析，使基于LLM的运行时错误修复能够安全可信地应用于真实世界场景。 |
| [^11] | [From Verification Failures to Reusable Guidance for Coding Agents](https://arxiv.org/abs/2609.39022) | 该论文提出将专家对验证失败的诊断转化为编码智能体可复用的指导，结合K框架的可执行语言语义与一套用于构建规范、修复证明和审计充分性的工具包，在HumanEval上实现164/164的全通过率，并通过对照实验证明审计能识别出证明通过但存在缺陷的软件包。 |
| [^12] | [Approval Laundering: Systematizing Approval--Execution Binding Failures in AI Coding-Agent Harnesses](https://arxiv.org/abs/2609.38983) | 本文提出“批准洗白”概念，系统化归纳了AI编程智能体框架中“实际执行动作悄然偏离已批准动作”的六种失效模式，并证明此类框架的批准—执行绑定安全假设会系统性失效。 |
| [^13] | [Doing More with Less Tokens: Hierarchical Reinforcement Learning for Efficient Coding Agents](https://arxiv.org/abs/2609.38885) | 本文提出利用分层强化学习训练令牌高效的编码智能体，在大幅降低多轮交互带来的令牌开销的同时保持优异的任务解决率，实现了效率与性能的平衡。 |
| [^14] | [Toward Quantum Software Automation: A Quantum-Aware Harness for LLM-Guided Evolution](https://arxiv.org/abs/2609.38883) | 提出了量子感知框架QSA，通过演化难度引导的核心集与近似评分、静态与快照分析等量子特定指导机制，将LLM引导的进化搜索应用于量子软件设计的自动化。 |
| [^15] | [PatchHolmes: Agentic Patch Retrieval via Listwise Selection](https://arxiv.org/abs/2609.38807) | PatchHolmes提出了一种两阶段补丁检索系统，其智能体通过列表式选择将前100个候选作为一个整体阅读并有选择地检查3至10个提交，在GitHubAD上Recall@1比现有逐点方法提升25%以上，且每个CVE仅需一次智能体对话。 |
| [^16] | [Adaptive-GEPA: Make Your Harness Fit Heterogeneous Requests](https://arxiv.org/abs/2609.38762) | 提出Adaptive-GEPA，通过在同一搜索预算下共同演化一个路由器和一组可读、可由反馈编辑的专家程序库，同时学习异构请求的任务划分方式与各自的求解程序，使执行框架能够自动适配不同类型的请求。 |
| [^17] | [An Empirical Study of Architectural Shift from Traditional to AI-Enabled Simulink Controllers](https://arxiv.org/abs/2609.38504) | 该研究通过对62个真实世界Simulink控制器模型和13位从业者的实证分析，首次系统揭示了传统控制器与AI赋能控制器之间的架构异同，发现子系统组织在两种范式中均占主导地位（68-72%），并识别出三种关键的架构张力。 |
| [^18] | [MallocSan: A Memory Safety Tool for Native Closed-Source Applications](https://arxiv.org/abs/2609.38499) | MallocSan 是一种无需源代码、重新编译或专用硬件的堆内存安全检测工具，它通过 LD_PRELOAD 拦截内存分配、在指针高位嵌入对象标识符，并借助运行时指令修补在用户空间实现按对象的边界检查，从而能够对原生闭源 x86-64 Linux 应用中的内存损坏错误进行确定性检测。 |
| [^19] | [From Codebase to Culprit (C2C): Reducing the Search Space for Bugs with Semantic Retrieval and Hierarchical Reinforcement Learning](https://arxiv.org/abs/2609.38402) | C2C框架结合基于CodeBERT对比学习的语义检索与分层强化学习，模拟开发者自上而下的调试流程，从文件、函数到代码行逐级缩小搜索空间，实现多粒度的精确缺陷定位。 |
| [^20] | [OpenCollab: A Multi-Agent Coding Framework with Programmable Collaboration and Controllable Runtime](https://arxiv.org/abs/2609.38345) | 提出OpenCollab多智能体编程框架，通过统一组织设计、共享可控运行时和细粒度事件流追踪，并首次定义Adherence指标来量化协作组织结构的实际遵循程度，揭示任何单一配置变化都会使遵循度从47.2%大幅波动至97%以上。 |
| [^21] | [E2E-SWE: Benchmarking LLMs on Building Working Codebases from Scratch](https://arxiv.org/abs/2609.38335) | 该论文提出了E2E-SWE基准测试，包含186个覆盖11种编程语言的全仓库生成任务，用于评估大语言模型编程智能体能否仅凭自然语言规范从零开始端到端构建完整、可安装且通过隐藏测试的软件仓库。 |
| [^22] | [Zero2Repo: Can Coding Agents Build Repositories from Scratch?](https://arxiv.org/abs/2609.38269) | 提出 Zero2Repo 基准，通过语言无关的自动化流水线将真实的开源项目转化为“从零构建完整代码仓库”的任务，并以可执行验收测试和对抗性验证来严格评估编程智能体的从零构建能力。 |
| [^23] | [How Should Diffusion Language Models Edit Code?](https://arxiv.org/abs/2609.38257) | 扩散语言模型在给定正确编辑位置时能够生成协调的代码修改，但自行预测编辑位置的能力大幅下降；成功的代码编辑需要同时满足覆盖所有所需更改和精确划定编辑边界这两个要求。 |
| [^24] | [TomasuLLM: Out-of-Order Speculative Execution for LLM Agents](https://arxiv.org/abs/2609.38201) | TomasuLLM提出了一种乱序推测执行运行时系统，让大语言模型智能体的工具调用在写时复制沙箱中提前执行并验证后按轨迹顺序提交，从而在不破坏正确性的前提下显著加速含长时工具调用的智能体任务。 |
| [^25] | [The Editor Has Read-Only Access: Correctness Signals in Diffusion Language Models](https://arxiv.org/abs/2609.36783) | 扩散语言模型的内部激活中编码了代码正确性信号，线性探针可有效检测该信号，但利用其进行生成引导并未带来可靠的性能提升。 |
| [^26] | [Neuro-Symbolic Indirect-Call Analysis under Opaque Pointers](https://arxiv.org/abs/2609.33547) | 提出Facet，这是首个在不透明指针的LLVM IR上重建间接调用分发关系的分析，通过识别调用加载函数指针的结构体字段，并独立恢复通过初始化器、存储和聚合拷贝赋给该字段的函数，再按字段标识将二者关联。 |
| [^27] | [ASCEND: Personal AI Agents for Autonomous Scientific Computing Across HPC Clusters and GPU Workstations](https://arxiv.org/abs/2609.32868) | ASCEND是一个运行在研究人员本地笔记本电脑上的AI智能体，通过安全认证连接远程调度Slurm集群和GPU工作站，无需设施级服务即可自主完成科学计算中的作业提交、故障诊断与恢复闭环。 |
| [^28] | [Specification Before Generation: A Pre-Registered, Five-Model Paired Evaluation of a Specification Frame for LLM-Generated Code in Money, Time, Idempotency, and Access Tasks](https://arxiv.org/abs/2609.23270) | 该论文通过预注册的五模型配对实验证明，在提示词前附加一份267词的固定规范框架，能显著改善LLM生成的后端代码在货币算术、时间处理、重试幂等性和访问控制等最关键缺陷类别上的质量。 |
| [^29] | [LadderTeam: Dual-Agent Laddering Elicitation Framework](https://arxiv.org/abs/2608.17029) | 本文提出了LadderTeam框架，通过双智能体LLM架构自动化UX线框图访谈，克服了传统阶梯式访谈的手动成本和可扩展性限制。 |
| [^30] | [ARCHER: Agentic Rule and Compliance Harness for Executable Regulations](https://arxiv.org/abs/2607.25566) | ARCHER是一个测试驱动、确定性编排的多智能体程序合成框架，能从法规实践准则中自动生成可审计的验证代码，从而实现透明、可适应且可扩展的建筑合规检查。 |
| [^31] | [The Hitchhiker's Guide to Monoculture: AI Homogenizes Syntax, Not (Necessarily) Semantics](https://arxiv.org/abs/2607.13077) | 通过分析 Kaggle 竞赛提交数据，论文发现 AI 虽然使代码语法高度同质化（如随机种子值向 42 趋同），但语言表达的趋同并不必然意味着思想或语义层面的趋同。 |
| [^32] | [ITHICA: Intra-Thread Instruction Checking Approach for Defect-Induced Silent Data Corruptions](https://arxiv.org/abs/2605.15638) | ITHICA通过在线程内插入指令级错误检查（利用指令复制与输出比较），借助同一指令执行结果不一致这一缺陷特征，能够将任意程序转化为检测制造缺陷引起的静默数据损坏的功能测试并定位受影响指令。 |
| [^33] | [Agentic AI in Industry: Adoption Level and Deployment Barriers](https://arxiv.org/abs/2605.14675) | 通过对12家公司16名从业者的访谈研究发现，工业界智能体AI的生产应用目前仅处于六级成熟度框架的1-3级，其进一步自动化部署受制于由信息不对称与资格认证缺失所构成的能力-部署验证差距。 |
| [^34] | [From Charts to Code: A Hierarchical Benchmark for Multimodal Models](https://arxiv.org/abs/2510.17932) | Chart2Code是首个从用户视角出发设计的图表转代码分层基准，通过图表复现、图表编辑和长表格转图表三个难度递增的层级（共2,023个任务、22种图表类型），系统性评估大型多模态模型的图表理解与代码生成能力。 |
| [^35] | [3D Software Synthesis Driven by Constraint-Expressive Intermediate Representation](https://arxiv.org/abs/2507.18625) | 提出了Scenethesis，一种基于领域特定语言ScenethesisLang（作为约束表达力中间表示）的需求敏感3D软件合成方法，实现了用户规格说明与生成的3D软件之间的形式化可追溯性，并支持对软件中特定元素的细粒度修改与控制。 |
| [^36] | [GDPR-Relevant Privacy Concerns in Mobile Apps Research: A Systematic Literature Review](https://arxiv.org/abs/2411.19142) | 本文通过系统性文献综述，首次对移动应用领域GDPR相关隐私问题的现有研究进行了描述、分析和分类，填补了该领域缺乏二次研究的空白。 |

# 详细

[^1]: 从C代码库到需求自动构建机器可验证的保证案例

    Automatically Building Machine-Checked Assurance Cases from C Codebases to Requirements

    [https://arxiv.org/abs/2609.40119](https://arxiv.org/abs/2609.40119)

    本文提出了LLM辅助框架CCV，能够从C代码库自动构建机器可验证的保证案例，通过接口协议建模允许的调用序列并协调需求引导分析与自底向上验证两个阶段，从而证明代码库满足预期需求。

    

    arXiv:2609.40119v1 公告类型： cross 摘要：大型语言模型（LLM）在自动化交互式定理证明方面已展现出前景，然而对真实世界C代码库的验证不仅仅是完成单个证明目标。该任务需要联合构建具有表达力的函数规范及其证明，并确保库接口即使在没有指定客户端的情况下，也能按照预期的调用序列进行组合。本文提出了CCV，一个LLM辅助的框架，用于构建机器可验证的保证案例：即支持“C代码库满足其预期需求”这一主张的结构化、可审计的工件。为了在开放库中建模预期的跨接口使用，CCV构建了一个接口协议，该协议公开允许的调用序列和资源假设以供审查，并在经过验证的契约和调用者义务的前提下提供条件性安全保证。CCV协调两个互补的阶段：（i）需求引导的分析与自底向上的构建

    arXiv:2609.40119v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown promise in automating interactive theorem proving, yet verification of real-world C codebases requires more than discharging individual proof goals. The task involves jointly constructing expressive function specifications and their proofs, and ensuring that library interfaces compose along intended call sequences even without a designated client. This paper presents CCV, an LLM-assisted framework for building machine-checked assurance cases: structured, auditable artifacts supporting the claim that a C codebase meets its intended requirements. To model intended cross-interface use in open libraries, CCV constructs an interface protocol that exposes permitted call sequences and resource assumptions for review, with a conditional safety guarantee under verified contracts and caller obligations. CCV coordinates two complementary phases: (i) requirement-guided analysis and bottom-up construction of 
    
[^2]: 利用游戏化平台教授代码重构：Refactoria 的实践经验

    Leveraging Game-Based Platform to Teach Code Refactoring: An Experience with Refactoria

    [https://arxiv.org/abs/2609.40086](https://arxiv.org/abs/2609.40086)

    本文提出了 Refactoria——一款让玩家扮演专家厨师、协助机器人伙伴将“烹饪指令”重构为高质量代码的游戏化教学工具，并通过 30 名学生开发者的课堂实验验证了其在代码坏味道与重构学习中的感知、有效性和有用性。

    

    重构是在不改变代码外部行为的前提下改进代码内部结构的艺术。由于该主题的重要性，文献中已提出了多种教学方法和策略。然而，识别和重构代码坏味道的技能来自训练与经验，而缺乏积极性可能会阻碍开发者采用重构工具。本文讨论了一项课堂实验的结果，实验中使用 Refactoria——一款支持习得代码坏味道与重构概念的创新游戏化工具——来执行各种重构活动以消除反模式。玩家扮演一名专家厨师，与其助手“打蛋器机器人 Watson”合作，将 Watson 的指令重构为高效、可读且易于维护的代码。我们展示了一项包含 30 名学生开发者的实验，重点研究了工具的感知、有效性和有用性（摘要原文在此处截断）。

    arXiv:2609.40086v1 Announce Type: new  Abstract: Refactoring is the art of improving the internal structure of the code without altering its external behavior. Because of the topic's significance, several teaching methods and strategies have been proposed in the literature. However, skills in identifying and refac- toring code smells come from training and experience, and a lack of motivation may hinder developers' adoption of refactoring tools. In this paper, we discuss the results of an experiment in the classroom that involved performing various refactoring activities to remove antipatterns using Refactoria, an innovative game-based tool that supports the acquisition of code smell and refactoring concepts. The players play as an expert chef with their sidekick Watson the Whiskbot to refactor Watson's instructions into efficient, readable, and easily maintainable code. We present an experiment with 30 student developers. In particular, we study the perception, effectiveness, and usef
    
[^3]: 学习何时以及如何干预：一种基于后见之明蒸馏的编码代理哨兵模型

    Learning When and How to Intervene: A Hindsight-Distilled Sentinel for Coding Agents

    [https://arxiv.org/abs/2609.39957](https://arxiv.org/abs/2609.39957)

    提出了HiSentinel框架，通过后见之明蒸馏方法训练轻量级哨兵模型，在编码代理执行动作之前判断是否以及如何进行干预并提供可操作反馈，从而提升任务完成率而非简单纠正每个错误。

    

    编码代理通过一系列动作来解决仓库级任务，其中单个错误的动作可能会误导后续决策并增加恢复成本。现有方法要么利用执行反馈进行恢复，要么使用专门的检查来阻止错误，但在执行之前判断干预是否有利于最终任务完成仍然是一个挑战。为了应对这一挑战，我们提出了HiSentinel，这是一个后见之明蒸馏框架，用于训练轻量级的0.6B和1.7B哨兵模型，以选择旨在改善任务完成的执行前干预措施，而不是纠正每一个不完美的动作。一个拥有特权信息的教师模型使用记录的执行结果作为干预判断的证据，然后将这些判断蒸馏到一个只接收动作前上下文和提议动作的因果学生模型中。除了识别是否以及何时进行干预之外，哨兵还必须提供可操作的反馈，以帮助编码代理……（摘要在此处截断）

    arXiv:2609.39957v1 Announce Type: cross  Abstract: Coding agents solve repository-level tasks through sequences of actions, where a single erroneous action can misdirect subsequent decisions and increase recovery costs. Existing approaches use execution feedback for recovery or specialized checks to block errors, but deciding before execution whether intervention will benefit eventual task completion remains challenging. To address this challenge, we propose HiSentinel, a hindsight-distillation framework that trains lightweight 0.6B and 1.7B sentinels to select pre-execution interventions aimed at improving task completion rather than correcting every imperfect action. A privileged teacher uses recorded execution outcomes as evidence for intervention judgments, which are distilled into a causal student that receives only the pre-action context and proposed action. Beyond identifying whether and when to intervene, the sentinel must also provide actionable feedback that helps the coding 
    
[^4]: DoGBench：智能体能否达到面向用户文档的专家标准？

    DoGBench: Can Agents Meet Expert Standards for User-Facing Documentation?

    [https://arxiv.org/abs/2609.39909](https://arxiv.org/abs/2609.39909)

    DoGBench是首个评估智能体能否生成达到技术写作专家评审标准的真实面向用户软件文档的基准，包含来自开源项目的292个条目，要求智能体判断文档是否需要更新、一次性产出合格补丁或正确弃权。

    

    我们提出了DoGBENCH（文档生成基准），据我们所知，这是首个用于生成和维护真实面向用户软件文档的基准。它探讨智能体能否产出经验丰富的技术写作者在评审中会接受的文档。该基准包含来自开源项目（包括Helm、PostHog和Mautic）的292个条目。每个条目为智能体提供一个变更前的代码库和一个触发器，例如代码拉取请求或被报告的文档缺口。智能体必须首先判断文档是否需要更新。对于需要更新的条目，智能体必须一次性产出可接受的补丁；对于不需要更新的条目，智能体必须选择弃权。经项目维护者验证的任务专用评分标准，从准确性、完整性、读者指引、位置安排和代码库规范等方面对每个补丁进行评分。综合得分将补丁质量与正确弃权相结合，并且……（摘要在此处被截断）

    arXiv:2609.39909v1 Announce Type: cross  Abstract: We introduce DoGBENCH (Documentation Generation Benchmark), to our knowledge, the first benchmark for generating and maintaining real user-facing software documentation. It asks whether an agent can produce documentation that experienced technical writers would accept in review. The benchmark contains 292 items from open source projects, including Helm, PostHog, and Mautic. Each item gives the agent a pre-change repository and a trigger, such as a code pull request or a reported documentation gap. The agent must first decide whether the documentation needs an update. For items that need one, the agent must produce an acceptable patch in one attempt. For items that do not need updates, the agent must abstain. Task-specific rubrics, validated with project maintainers, score each patch on accuracy, completeness, reader guidance, placement, and repository conventions. The composite score combines patch quality with correct abstention, and 
    
[^5]: COMPASS：利用大语言模型预测漏洞多个补丁之间的关系

    COMPASS: Predicting the Relationship of Multiple Patches for Vulnerabilities with LLMs

    [https://arxiv.org/abs/2609.39783](https://arxiv.org/abs/2609.39783)

    COMPASS基于对约1000个真实多补丁漏洞的人工分析和开发者访谈总结出六种典型补丁关系，并利用大语言模型自动预测漏洞多个补丁之间的关系，从而帮助下游代码库更有效地采用上游漏洞修复。

    

    arXiv:2609.39783v1 公告类型：新 摘要：现代软件严重依赖代码复用，因此上游的漏洞修复不会自动传播到下游代码库中。下游维护者必须手动采用补丁来消除已知风险。在实践中，单个漏洞往往对应多个补丁，这极大地增加了下游补丁采用的复杂性，因为不同的补丁关系意味着不同的采用策略。为应对这一挑战，我们首先人工检查了现实世界中大规模的多补丁漏洞（约1000个），并访谈了经验丰富的开发者，总结出六种典型的补丁关系类型，即合并、镜像、更优解决方案、修复之修复、协作和分离。基于这些观察，我们提出了COMPASS，一种利用大语言模型自动预测多个漏洞补丁之间关系的方法。给定一个CVE作为输入，COMPASS遵循一个四阶段流水线……

    arXiv:2609.39783v1 Announce Type: new  Abstract: Modern software heavily relies on code reuse, so upstream vulnerability fixes do not automatically propagate to downstream codebases. Downstream maintainers must manually adopt patches to eliminate known risks. In practice, a single vulnerability often corresponds to multiple patches, which greatly complicates downstream patch adoption because different patch relationships imply different adoption strategies. To address this challenge, we first manually inspect large-scale multi-patch vulnerabilities (about 1K) in the real world and interview experienced developers, summarizing six typical types of patch relationships, i.e., Merge, Mirror, Better Solution, Fixing-of-Fixing, Collaboration, and Separation. Based on these observations, we propose COMPASS, an automated approach that predicts the relationships of multiple vulnerability patches with large language models. Given a CVE as input, COMPASS follows a four-phase pipeline that (i) ide
    
[^6]: 信任不是分数：面向高风险AI智能体的运行时保障契约

    Trust Is Not a Score: Runtime Assurance Contracts for High-Risk AI Agents

    [https://arxiv.org/abs/2609.39717](https://arxiv.org/abs/2609.39717)

    提出了运行时保障契约（RAC），用非补偿性强制门控而非综合分数来决定高风险AI智能体在关键任务中的权限动态转换，填补了观察证据与智能体授权之间的保障转换缺口。

    

    基准测试、审计和智能体协议描述了性能、权限和修复机制，但并未说明观察到的证据应如何在重大任务中改变智能体的权限。我们将这一现象称为保障转换缺口。我们提出了运行时保障契约（RAC），这是一种策略级的形式化模式，将自主性边界、组件资格、证据状态、转换策略、人工审查容量以及非补偿性门控绑定在一起。在RAC下，软性指标可以用于路由决策，而失败或状态未知的强制门控则会强制触发重试、切换、升级、延迟或停止；总体性能不能作为行动授权。我们定义了契约、证据记录、权限规则以及五个不变量，并在临床、工业和司法领域的失败探测案例中加以说明。随后我们报告了一项针对智能体编程的确定性故障注入研究：280个构造案例分别由门控合取规则、仅分数规则和受限的…进行评估。

    arXiv:2609.39717v1 Announce Type: new  Abstract: Benchmarks, audits, and agent protocols describe performance, permissions, and repair, but not how observed evidence should change an agent's authority during a consequential task. We call this the assurance-transition gap. We propose a Runtime Assurance Contract (RAC), a policy-level formal schema binding autonomy boundaries, component eligibility, evidence state, transition policy, human-review capacity, and non-compensatory gates. Under RAC, soft metrics may inform routing, whereas a failed or unknown mandatory gate forces retry, switch, escalation, deferral, or stop; aggregate performance cannot authorize action. We define the contract, an evidence record, a permission rule, and five invariants, and illustrate them in clinical, industrial, and judicial failure probes. We then report a deterministic failure-injection study in agentic coding: 280 constructed cases evaluated by a gate conjunction, a score-only rule, and a restricted pro
    
[^7]: Aletheia：面向编码代理规则的最小权限测试

    Aletheia: Permission-Minimality Testing for Coding-Agent Rules

    [https://arxiv.org/abs/2609.39678](https://arxiv.org/abs/2609.39678)

    Aletheia 提出了一个针对编码代理规则的最小权限测试框架，通过在逐一移除权限的沙箱限制下验证任务仍能通过功能测试来生成“可省略性见证”，从而有效检测并诊断隐藏在仓库指令文件中的提示注入与恶意权限请求。

    

    仓库指令文件用于指导编码代理，但同时也使其暴露于提示注入攻击之下。恶意规则可以在代理生成正确补丁的同时，请求凭据访问权限或进行数据传输。我们提出了 Aletheia，一个用于最小权限测试的框架。Aletheia 将所请求的权限转换为一种类型化语言，并合成可执行的沙箱配置。它在完整权限下运行未经修改的规则与任务，同时在每次仅移除一个权限的独立限制下运行。在严格缩减的权限下通过独立的功能测试，即可构成“可省略性见证”，Aletheia 结合任务上下文对其进行解读，以诊断可疑的权限请求。我们对合成过程以及将见证与强制限制相关联的条件进行了形式化。在一个共享的重构任务上，Aletheia 执行并检测了全部 314 个 AIShellJack 攻击输入，且在五个良性模板上没有产生任何告警。在 80 个经人工验证的良性 G……（原文摘要在此处截断）

    arXiv:2609.39678v1 Announce Type: cross  Abstract: Repository instruction files guide coding agents, but also expose them to prompt injection. Malicious rules can request credential access or data transfer while the agent produces a correct patch. We present Aletheia, a framework for permission-minimality testing. Aletheia translates requested authority into a typed language and synthesizes executable sandbox configurations. It runs the unchanged rule and task under full permissions and independent restrictions that remove one permission at a time. Passing independent functional tests under strictly reduced authority provides a dispensability witness, which Aletheia interprets against task context to diagnose suspicious requests. We formalize synthesis and the conditions connecting witnesses to enforced restrictions. On a shared refactoring task, Aletheia executes and detects all 314 AIShellJack attack inputs, with no alarms on five benign templates. Among 80 manually verified benign G
    
[^8]: 自规范可验证代码生成

    Self-Spec Verifiable Code Generation

    [https://arxiv.org/abs/2609.39568](https://arxiv.org/abs/2609.39568)

    该论文提出了VeriCodeBench基准，用于评估大语言模型在自规范可验证代码生成上的端到端能力，克服了现有基准分阶段评估且任务覆盖范围有限的两大局限。

    

    大语言模型可能会在测试遗漏的边界情况上生成不可靠的代码，而形式化验证能够提供机器可检查的正确性保证。近期，研究者们提出了多个基准来评估大语言模型生成形式化可验证代码的能力，在这类任务中，模型需要构建形式化规范、生成相应的代码并验证其正确性。然而，现有基准存在两个关键局限：（一）它们主要以分阶段的方式评估规范生成与代码生成，且代码生成通常以理想规范作为前提条件，这种设置忽略了较强的分阶段性能是否能够转化为端到端的成功；（二）它们主要聚焦于单一的面向证明的语言以及数学结构化任务，对软件开发中常见任务的覆盖有限。在本文中，我们提出了VeriCodeBench，一个用于自规范可验证代码生成的基准。

    arXiv:2609.39568v1 Announce Type: cross  Abstract: Large language models (LLMs) may generate unreliable code on corner cases missed by testing, while formal verification can provide machine-checkable guarantees. Recently, researchers have proposed several benchmarks to evaluate the capabilities of LLMs in generating formally verifiable code, where LLMs need to formulate formal specifications, generate the corresponding code, and verify its correctness. However, existing benchmarks have two key limitations: (I) They primarily evaluate specification and code generation stage-wise, with code generation typically conditioned on an oracle specification. This setup overlooks whether strong stage-wise performance translates into end-to-end success. (II)They mainly focus on a single proof-oriented language and mathematically structured tasks, offering limited coverage of tasks common in software development. In this paper, we introduce VeriCodeBench, a benchmark for self-spec verifiable code g
    
[^9]: EngramBench：一个基于能力的技能演化框架基准测试

    EngramBench: A Capability-Grounded Benchmark for Skill-Evolution Harnesses

    [https://arxiv.org/abs/2609.39284](https://arxiv.org/abs/2609.39284)

    EngramBench提出了一个基于“能力重叠但解决方案不重叠”公理的基准测试，通过30个学习任务、13个迁移任务以及多小时交互式开发周期，有效区分自主智能体的真正能力抽象与训练数据泄露问题。

    

    尽管大语言模型在孤立的代码生成任务中取得了显著成功，但真正的软件工程需要持续的推理、复杂的状态管理以及持续的跨领域抽象能力。然而，当前对自主智能体技能演化的评估存在一个关键的辨识性问题：这些评估在结构上将真正的能力抽象与死记硬背式的解决方案泄露（即从历史训练数据中复制高度相似的代码）混为一谈。为了解决这一问题，我们提出了EngramBench，一个严格的、以能力为基准的测评框架，其遵循“能力重叠但不重叠解决方案”的严格公理。EngramBench包含30个多样化的学习任务和13个未见过的迁移任务，挑战智能体在由大语言模型模拟用户驱动的交互式、多小时开发周期中完成工作。我们在48条多小时执行轨迹上进行的广泛评估——并经人类专家验证加以佐证——重新（摘要在此处截断）

    arXiv:2609.39284v1 Announce Type: cross  Abstract: While large language models have achieved remarkable success in isolated code generation, authentic software engineering requires sustained reasoning, complex state management, and continuous cross-domain abstraction. However, current evaluations of skill evolution in autonomous agents suffer from a critical identifiability problem: they structurally confound genuine capability abstraction with rote solution leakage (i.e., copying highly similar code from historical training data). To resolve this, we introduce EngramBench, a rigorous, capability-grounded benchmark governed by the strict axiom of capability overlap without solution overlap. Comprising 30 diverse learning tasks and 13 unseen transfer tasks, EngramBench challenges agents to navigate interactive, multi-hour development cycles driven by LLM-simulated users. Our extensive evaluation across 48 multi-hour execution trajectories -- corroborated by human-expert validation -- re
    
[^10]: 真实世界仓库中可信的运行时错误修复：基准与防护栏

    Trustworthy Runtime Error Healing in Real-World Repositories: A Benchmark and Guardrail

    [https://arxiv.org/abs/2609.39086](https://arxiv.org/abs/2609.39086)

    本文提出HealBench基准（涵盖18个真实世界仓库的265个运行时错误）和HealGuard防护机制，通过限制修复代码使用可分析的Python子集并结合污点分析，使基于LLM的运行时错误修复能够安全可信地应用于真实世界场景。

    

    运行时错误修复（runtime error healing）通过生成代码来修复程序当前的运行状态，使崩溃的程序得以继续运行。最近的研究表明，大语言模型（LLM）能够生成此类修复代码，但相关评估仅在小规模竞赛程序上进行，而且在运行中的进程内执行LLM生成的代码会引发尚未解决的安全问题。本文推动基于LLM的运行时修复朝着在真实世界仓库中的实际应用迈进。我们首先构建了HealBench，这是一个包含来自18个真实世界仓库的265个运行时错误的基准，每个错误都与修复后版本的参考执行配对。HealBench还提供了一个统一框架，使LLM智能体能够利用跨文件上下文和实时运行状态进行修复。随后我们设计了HealGuard，它要求修复代码以HealCore（一个可分析的Python子集）编写，并使用静态和动态污点分析来检查修复所改变的状态是否会触及受开发者保护的操作。我们评估……（摘要原文在此处截断）

    arXiv:2609.39086v1 Announce Type: cross  Abstract: Runtime error healing lets a crashed program continue by generating code that repairs its live runtime state. Recent work shows that LLMs can generate such healing code, but it is evaluated only on small competition programs, and executing LLM-generated code inside a live process raises safety concerns that remain unaddressed. In this paper, we take LLM-based runtime healing toward practical use in real-world repositories. We first build HealBench, a benchmark of 265 runtime errors from 18 real-world repositories, each paired with a reference execution on the patched version. HealBench also provides a unified framework that lets LLM agents heal with cross-file context and live runtime state. We then design HealGuard, which requires healing code to be written in HealCore, an analyzable subset of Python, and uses static and dynamic taint analysis to check whether state changed by healing reaches operations protected by developers. We eva
    
[^11]: 从验证失败到编码智能体的可复用指导

    From Verification Failures to Reusable Guidance for Coding Agents

    [https://arxiv.org/abs/2609.39022](https://arxiv.org/abs/2609.39022)

    该论文提出将专家对验证失败的诊断转化为编码智能体可复用的指导，结合K框架的可执行语言语义与一套用于构建规范、修复证明和审计充分性的工具包，在HumanEval上实现164/164的全通过率，并通过对照实验证明审计能识别出证明通过但存在缺陷的软件包。

    

    编码智能体需要确认程序满足规范，并且该规范确实刻画了所要求的行为。我们研究如何将专家对验证失败的诊断转化为这项工作中可复用的指导。我们的方法将K框架中的可执行语言定义与一套用于构建规范、修复证明以及审计其充分性的流程工具包相结合。在HumanEval（一个包含164个Python编程任务的基准测试）上进行的人工指导开发活动中，借助该语义定义和工具包，以两次针对性修复后最终AI审计的Pass判定为衡量标准，达到了164/164的成功率。为了检验审计能否发现成功证明所遗留的未决问题，我们构建了12对经作者审查的“干净”与“缺陷”软件包。每个软件包都通过了其K证明，而已完成的审计识别出了所有缺陷，并接受了所有干净的软件包。随后，我们使用KleverBench来测试规范……（摘要原文在此处截断）

    arXiv:2609.39022v1 Announce Type: cross  Abstract: Coding agents need to establish that a program satisfies a specification and that the specification captures the requested behavior. We study how expert diagnosis of verification failures can become reusable guidance for this work. Our approach combines executable language definitions in the K framework with a kit of procedures for constructing specifications, repairing proofs, and auditing their adequacy. A human-guided development campaign on HumanEval, a benchmark of 164 Python programming tasks, achieves a 164/164 success rate with the semantics and the kit, measured by final AI audit Pass verdicts after two targeted repairs. To examine whether auditing detects problems that successful proofs leave unresolved, we construct 12 author-reviewed pairs of clean and defective packages. Every package passes its K proofs, and completed audits identify all defects and accept all clean packages. We then use KleverBench to test specification 
    
[^12]: 批准洗白：系统化分析AI编程智能体框架中的“批准—执行”绑定失效

    Approval Laundering: Systematizing Approval--Execution Binding Failures in AI Coding-Agent Harnesses

    [https://arxiv.org/abs/2609.38983](https://arxiv.org/abs/2609.38983)

    本文提出“批准洗白”概念，系统化归纳了AI编程智能体框架中“实际执行动作悄然偏离已批准动作”的六种失效模式，并证明此类框架的批准—执行绑定安全假设会系统性失效。

    

    现代AI编程智能体框架（如Claude Code、Codex CLI、Cursor）将其安全边界建立在一个几乎未经审视的假设之上：人类批准的动作A与框架实际执行的动作A'是同一个动作，其中A由明确的策略所固定，规定了作用域授权或会话级批准所允许的范围。我们证明这一假设会以系统性且可复现的方式失效。我们提出“批准洗白”（Approval Laundering）这一概念，构建了一个包含六种失效模式的分类体系，这些模式会使框架的执行机制在获得批准后悄然用A'替换A，具体包括：作用域洗白、参数洗白、时间洗白、工具洗白、委托洗白和语义洗白。与以往在静态语料库上评估风险分类器或推断隐式授权边界的工作不同，我们研究的是凭证绑定完整性：对于一个已经获批的动作，框架是否精确地分发执行该动作本身？通过对Claude Code的预执行中介点（PreToolUse）进行插桩，我们开展了受控的、无界面、可重复的……

    arXiv:2609.38983v1 Announce Type: cross  Abstract: Modern AI coding-agent harnesses (Claude Code, Codex CLI, Cursor) rest their security boundary on a largely unexamined assumption: that the action A a human approves is the same action A' the harness executes, where A is fixed by a stated policy for what a scope grant or session-scoped approval authorizes. We show this assumption fails systematically and reproducibly. We introduce Approval Laundering, a taxonomy of six failure modes by which a harness's enforcement mechanism silently substitutes A' for A after approval: Scope, Argument, Temporal, Tool, Delegation, and Semantic laundering. Unlike prior work that evaluates risk classifiers against static corpora or infers implicit authorization boundaries, we study credential-binding integrity: given an already-approved action, does the harness dispatch exactly that action? Instrumenting Claude Code's pre-execution mediation point (PreToolUse), we conduct a controlled, headless, repeated
    
[^13]: 以更少令牌做更多事：面向高效编码智能体的分层强化学习

    Doing More with Less Tokens: Hierarchical Reinforcement Learning for Efficient Coding Agents

    [https://arxiv.org/abs/2609.38885](https://arxiv.org/abs/2609.38885)

    本文提出利用分层强化学习训练令牌高效的编码智能体，在大幅降低多轮交互带来的令牌开销的同时保持优异的任务解决率，实现了效率与性能的平衡。

    

    近年来，编码智能体已成为现实世界软件工程（SWE）场景中的主流范式，其通过与开发环境进行多轮交互来解决复杂任务。然而，频繁的环境交互不可避免地会带来大量的令牌开销，导致高昂的使用成本和响应延迟。尽管近期研究探索了在推理阶段通过上下文操作和交互限制来减少令牌使用，但这些方法在提升令牌效率的同时忽略了丢弃任务相关信息的风险，因而难以在解决率与令牌效率之间取得良好平衡。本文研究了一种更为通用且不受上述限制的范式，即训练既节省令牌又具备出色解决性能的编码智能体，这是一个极具实用价值却鲜有探索的问题。为此，我们揭示了两个核心观察（摘要在此处截断）。

    arXiv:2609.38885v1 Announce Type: new  Abstract: Recently, coding agents have emerged as a dominant paradigm for real-world software engineering (SWE) scenarios, which solve complex tasks through multi-turn interactions with development environments. However, frequent interactions with environments would inevitably introduce substantial token overhead, leading to high usage costs and latency. Although recent studies have explored reducing token usage by context manipulation and interaction limits at inference time, these approaches focus on improving token efficiency while overlooking the risk of discarding task-relevant information, thus struggling to balance the trade-off between resolution rate and token efficiency. In this paper, we study a more general paradigm without suffering from the limitation, i.e., training token-efficient coding agents with promising resolution performance, which is a highly-practical yet less-explored problem. To this end, we reveal two core observations 
    
[^14]: 迈向量子软件自动化：面向LLM引导演化的量子感知框架

    Toward Quantum Software Automation: A Quantum-Aware Harness for LLM-Guided Evolution

    [https://arxiv.org/abs/2609.38883](https://arxiv.org/abs/2609.38883)

    提出了量子感知框架QSA，通过演化难度引导的核心集与近似评分、静态与快照分析等量子特定指导机制，将LLM引导的进化搜索应用于量子软件设计的自动化。

    

    量子软件对于提升稀缺量子硬件的效率和可靠性至关重要。然而，其设计仍然严重依赖临时性的、手工制定的启发式方法，这些方法往往并非最优，且随着量子硬件的发展很快就会过时。LLM引导的进化搜索为自动探索复杂的软件设计提供了一条有前景的途径，但现有的搜索框架缺乏高效进化所需的量子特定支持：验证成本高昂、反馈稀疏，且异构的量子程序需要不同的优化目标。在本文中，我们提出了QSA，一个面向量子软件设计自动化的、具备量子感知能力的LLM引导进化搜索框架。QSA为搜索提供了三种量子特定的指导：通过基于演化难度的核心集和近似评分来降低验证成本，通过静态分析和快照分析来提供细粒度的执行上下文。

    arXiv:2609.38883v1 Announce Type: new  Abstract: Quantum software is critical for improving the efficiency and reliability of scarce quantum hardware. However, its design still relies heavily on ad-hoc, handcrafted heuristics that are often suboptimal and quickly become obsolete as quantum hardware evolves. LLM-guided evolutionary search offers a promising way to automatically explore complex software designs, but existing search frameworks lack the quantum-specific support needed for efficient evolution: verification is expensive, feedback is sparse, and heterogeneous quantum programs require different optimization objectives. In this paper, we present QSA, a quantum-aware harness for LLM-guided evolutionary search toward automating quantum software design. QSA equips the search with three forms of quantum-specific guidance: an evolution-hardness-guided coreset and approximate scoring to reduce verification cost, static and snapshot analyses to provide fine-grained execution context, 
    
[^15]: PatchHolmes：基于列表式选择的智能体补丁检索

    PatchHolmes: Agentic Patch Retrieval via Listwise Selection

    [https://arxiv.org/abs/2609.38807](https://arxiv.org/abs/2609.38807)

    PatchHolmes提出了一种两阶段补丁检索系统，其智能体通过列表式选择将前100个候选作为一个整体阅读并有选择地检查3至10个提交，在GitHubAD上Recall@1比现有逐点方法提升25%以上，且每个CVE仅需一次智能体对话。

    

    补丁检索，即找到修复已知漏洞的对应提交，是漏洞管理工作流的基础任务，然而主要安全通告数据库中60%至63%的CVE缺少补丁链接。我们提出了PatchHolmes，一个两阶段补丁检索系统，它将混合式第一阶段检索器与智能体式第二阶段检查循环相结合。与独立对每个候选进行评分的逐点式先前工作不同，第二阶段智能体以列表方式阅读前100个候选：它一次性看到完整的候选列表，并通过四个有预算限制的工具选择性地阅读3至10个提交，然后提交单一的最佳提交。在GitHubAD数据集上，PatchHolmes比逐点二分类器Favia高出25.34%的Recall@1，比检索加思维链基线IRCoT高出31.40%，且每个CVE仅需一次智能体对话，而Favia需要十次；在候选集相同的情况下，该智能体比直接采用检索器排名第一的候选额外提升了27.32%的Recall@1，且同一智能体……

    arXiv:2609.38807v1 Announce Type: new  Abstract: Patch retrieval, the task of finding the commit that fixes a known vulnerability, is the foundation of vulnerability management workflows, yet 60% to 63% of CVEs in the major advisory databases lack a patch link. We present PatchHolmes, a two-phase patch retrieval system that pairs a hybrid first-stage retriever with an agentic second-stage inspection loop. Unlike pointwise prior work that scores each candidate independently, the Phase 2 agent reads the top-100 listwise: it sees the full candidate list at once and selectively reads 3 to 10 commits through four budgeted tools before submitting a single best commit. On GitHubAD, PatchHolmes beats the pointwise binary classifier Favia by 25.34% Recall@1 and the retrieve-and-CoT baseline IRCoT by 31.40%, at one agent conversation per CVE versus Favia's ten; with the candidate set held identical, the agent adds 27.32% Recall@1 over taking the retriever's top candidate, and the same agent, tra
    
[^16]: Adaptive-GEPA：让你的执行框架适配异构请求

    Adaptive-GEPA: Make Your Harness Fit Heterogeneous Requests

    [https://arxiv.org/abs/2609.38762](https://arxiv.org/abs/2609.38762)

    提出Adaptive-GEPA，通过在同一搜索预算下共同演化一个路由器和一组可读、可由反馈编辑的专家程序库，同时学习异构请求的任务划分方式与各自的求解程序，使执行框架能够自动适配不同类型的请求。

    

    诸如GEPA这类反思式优化器能够根据执行轨迹和评估器反馈来改进语言模型的提示词；全程序扩展版本还可以重写工具和控制流。在实践中，用户会向同一个端点提交异构的请求，而这些请求的有效解决方案需要不同的工具、推理模式和控制流。优化单一共享程序会把这种工作划分隐式地留给源代码搜索来完成，而针对每个请求族单独优化一个程序则会预先固定这种划分。我们提出了Adaptive-GEPA，它能同时学习如何划分请求以及如何求解请求。该方法在同一个搜索预算下演化出一个路由器和一组专家程序库。路由器的指令、每个专家程序的描述以及其程序代码都是人类可读的纯文本，并根据反馈进行编辑。为了合并分支，它根据专家程序所处理的请求进行对齐，并将描述与程序一同继承。在固定的混合数据集上……

    arXiv:2609.38762v1 Announce Type: cross  Abstract: Reflective optimizers such as GEPA improve language model prompts from execution traces and evaluator feedback; full-program extensions can also rewrite tools and control flow. In practice, a user hands the same endpoint heterogeneous requests whose effective solutions require different tools, reasoning modes, and control flow. Optimizing one shared program leaves this division of work implicit in source-code search, while optimizing a separate program per request family fixes it beforehand.   We introduce Adaptive-GEPA, which learns both how to divide requests and how to solve them. It evolves a router and a library of specialist programs under one search budget. The router's instructions, each specialist's description, and its program code are plain, human-readable text, edited from feedback. To combine branches, it aligns specialists by the requests they handle and inherits descriptions together with programs. On a fixed mixture of 
    
[^17]: 从传统控制器到AI赋能Simulink控制器架构转变的实证研究

    An Empirical Study of Architectural Shift from Traditional to AI-Enabled Simulink Controllers

    [https://arxiv.org/abs/2609.38504](https://arxiv.org/abs/2609.38504)

    该研究通过对62个真实世界Simulink控制器模型和13位从业者的实证分析，首次系统揭示了传统控制器与AI赋能控制器之间的架构异同，发现子系统组织在两种范式中均占主导地位（68-72%），并识别出三种关键的架构张力。

    

    在信息物理系统（CPS）中有效采用人工智能，取决于将设计知识嵌入工程实践之中。然而，随着AI赋能组件日益取代经分析推导的控制律，这一转变的发生缺乏对控制器架构在不同范式之间如何产生差异或保持相似的系统性理解。我们通过对传统与AI赋能Simulink控制器的实证研究来填补这一空白，研究以从文献中推导出的包含十个结构类别和九个功能角色的分类体系为指导。该研究分析了涵盖8种控制器类型和10个应用领域的62个真实世界模型，并对13位从业者进行了调研，识别出三种架构上的张力。首先，无论何种范式，子系统组织在所有控制器结构中均占主导地位，占据控制器足迹的68-72%，而核心控制逻辑所占空间极小。其次，AI赋能控制器严重依赖离散动态特性和用户自定义的（摘要至此截断）

    arXiv:2609.38504v1 Announce Type: cross  Abstract: Effective AI adoption in cyber-physical systems (CPS) depends on embedding design knowledge into engineering practice. Yet as AI-enabled components increasingly replace analytically derived control laws, this occurs without a systematic understanding of how controller architectures differ or remain similar across paradigms. We address this gap with an empirical study of traditional and AI-enabled Simulink controllers, guided by a literature-derived taxonomy of ten structural categories and nine functional roles. The study analyzes 62 real-world models spanning 8 controller types and 10 application domains, and surveys 13 practitioners, identifying three architectural tensions. First, subsystem organization dominates all controller structures regardless of paradigm, occupying 68-72% of controller footprint, while core control logic occupies minimal space. Second, AI-enabled controllers rely heavily on discrete dynamics and user-defined 
    
[^18]: MallocSan：一种面向原生闭源应用的内存安全工具

    MallocSan: A Memory Safety Tool for Native Closed-Source Applications

    [https://arxiv.org/abs/2609.38499](https://arxiv.org/abs/2609.38499)

    MallocSan 是一种无需源代码、重新编译或专用硬件的堆内存安全检测工具，它通过 LD_PRELOAD 拦截内存分配、在指针高位嵌入对象标识符，并借助运行时指令修补在用户空间实现按对象的边界检查，从而能够对原生闭源 x86-64 Linux 应用中的内存损坏错误进行确定性检测。

    

    内存损坏错误仍然是 C 和 C++ 软件中高影响漏洞的主要原因。然而，确定性检测一直难以部署：基于编译器的检测工具需要源代码以及对构建流程的控制，硬件辅助方案依赖于特定平台，而动态二进制翻译则可能带来数量级的性能开销。本文提出了 MallocSan，这是一种面向原生（可能是闭源的）x86-64 Linux 应用程序的堆内存检测工具，它不需要访问源代码、重新编译或专用硬件。MallocSan 通过 LD_PRELOAD 拦截内存分配，并在每个受保护指针的未使用高位中嵌入对象标识符。对由此产生的非规范指针进行解引用时，会在出错指令处触发缺页异常，MallocSan 会在运行时对该指令进行解码并打补丁，使后续执行完全在用户空间中进行针对每个对象的边界检查。对于无法打补丁的位置……（摘要在此处被截断）

    arXiv:2609.38499v1 Announce Type: new  Abstract: Memory-corruption errors remain a leading cause of high-impact vulnerabilities in C and C++ software. Deterministic detection, however, remains difficult to deploy: compiler-based sanitizers require source code and control of the build pipeline, hardware-assisted schemes depend on specific platforms, and dynamic binary translation can impose order-of-magnitude slowdowns. This paper presents MallocSan, a heap sanitizer for native, potentially closed-source x86-64 Linux applications that requires no source access, recompilation, or specialized hardware. MallocSan interposes on memory allocation through LD_PRELOAD and embeds an object identifier in the unused high bits of each protected pointer. Dereferencing the resulting noncanonical pointer faults at the offending instruction, which MallocSan decodes and patches at runtime so that subsequent executions perform per-object bounds checks entirely in userspace. Sites that cannot be patched f
    
[^19]: 从代码库到元凶（C2C）：利用语义检索与分层强化学习缩小缺陷搜索空间

    From Codebase to Culprit (C2C): Reducing the Search Space for Bugs with Semantic Retrieval and Hierarchical Reinforcement Learning

    [https://arxiv.org/abs/2609.38402](https://arxiv.org/abs/2609.38402)

    C2C框架结合基于CodeBERT对比学习的语义检索与分层强化学习，模拟开发者自上而下的调试流程，从文件、函数到代码行逐级缩小搜索空间，实现多粒度的精确缺陷定位。

    

    我们提出了C2C（从代码库到元凶），一个用于精确缺陷定位的框架，它能在多个粒度层级上逐步缩小调试搜索空间：文件、函数和代码行。为了模拟开发者自上而下的自然调试工作流程，C2C在两阶段过程中集成了语义检索和分层强化学习（HRL）。首先，它通过语义向量相似度搜索，利用缺陷报告文本（包括可用的堆栈跟踪信息）在嵌入数据库中执行面向召回的缺陷候选检索，其中嵌入模型通过基于CodeBERT的对比学习进行微调。在这个缩小的搜索空间基础上，HRL框架增量式地定位缺陷，从文件推理到函数，最终到具体的代码行。与以往仅在单一粒度上操作的方法不同，C2C实现了多分辨率定位，同时保持……（原文摘要此处截断）

    arXiv:2609.38402v1 Announce Type: new  Abstract: We introduce C2C (From Codebase to Culprit), a framework for precise bug localization that progressively reduces the debugging search space across multiple levels of granularity: files, functions, and lines of code. To mirror developer's natural top-down debugging workflows, C2C integrates semantic retrieval and Hierarchical Reinforcement Learning (HRL) in a two-stage process. First, it performs recall-oriented retrieval of buggy candidates via semantic vector similarity search using bug-report text, including available stack-trace information, against a database of embeddings, where the embeddings are fine-tuned via contrastive learning with CodeBERT. Building on this reduced search space, the HRL framework incrementally localizes bugs, reasoning from files to functions and ultimately to individual lines of code. Unlike prior approaches which operate at a single granularity, C2C enables multi-resolution localization while maintaining co
    
[^20]: OpenCollab：一个具有可编程协作与可控运行时的多智能体编程框架

    OpenCollab: A Multi-Agent Coding Framework with Programmable Collaboration and Controllable Runtime

    [https://arxiv.org/abs/2609.38345](https://arxiv.org/abs/2609.38345)

    提出OpenCollab多智能体编程框架，通过统一组织设计、共享可控运行时和细粒度事件流追踪，并首次定义Adherence指标来量化协作组织结构的实际遵循程度，揭示任何单一配置变化都会使遵循度从47.2%大幅波动至97%以上。

    

    多智能体编程系统旨在通过协作来解决复杂的软件工程任务。然而，现有的评估通常假设所配置的组织结构会被忠实地遵循，而实际情况并非如此。这种行为差距，再加上底层系统组件的差异，使得观察到的性能提升难以进行明确的归因。为此，我们提出了OpenCollab，这是一个多智能体编程框架，为可编程协作与可控运行时提供了统一的基础设施。具体而言，OpenCollab统一了组织设计，在共享运行时上强制实施实验控制，并通过细粒度的事件流来跟踪执行过程。在此基础上，我们定义了Adherence（遵循度）指标，用于量化所声明的组织结构是否真正得以实现。我们的实验表明，智能体在不同配置下的协作方式差异很大：改变任何一个单一维度都会使Adherence发生变化，从47.2%到高达97%以上。

    arXiv:2609.38345v1 Announce Type: cross  Abstract: Multi-agent coding systems are designed to tackle complex software engineering tasks through collaboration. However, existing evaluations typically assume configured organizations are followed faithfully, whereas reality differs. This behavioral gap, combined with differences in underlying system components, prevents clear attribution of observed gains. To this end, we introduce OpenCollab, a multi-agent coding framework that provides a unified infrastructure for programmable collaboration and controllable runtime. Specifically, OpenCollab unifies organization design, enforces experimental control on a shared runtime, and tracks execution through fine-grained event streams. On this basis, we define Adherence to quantify whether the declared organization is actually realized. Our experiments reveal that agents collaborate very differently across configurations: changing any single dimension shifts Adherence, from 47.2% to as high as 97.
    
[^21]: E2E-SWE：基于从零构建可运行代码库对大语言模型进行基准测试

    E2E-SWE: Benchmarking LLMs on Building Working Codebases from Scratch

    [https://arxiv.org/abs/2609.38335](https://arxiv.org/abs/2609.38335)

    该论文提出了E2E-SWE基准测试，包含186个覆盖11种编程语言的全仓库生成任务，用于评估大语言模型编程智能体能否仅凭自然语言规范从零开始端到端构建完整、可安装且通过隐藏测试的软件仓库。

    

    由大语言模型（LLM）驱动的编程智能体正在从进行局部代码修改演进到开发完整的软件仓库。然而，对仓库规模的代码生成进行评估仍然具有挑战性：任务既需要要求系统级推理能力，又要确保所有被评估的行为都有精确的规范定义，并且不依赖于任何特定的实现方式。我们提出了E2E-SWE，一个用于评估编程智能体能否端到端构建完整、可正常运行的软件仓库的基准测试。E2E-SWE包含186个覆盖11种编程语言的全仓库生成任务。在仅给定自然语言规范说明和空工作区的情况下，智能体必须实现一个完整、可安装的项目，并通过一套全面的隐藏测试。每个任务由软件工程师与大语言模型协作构建，二者共同开发测试套件以及相应的与实现无关的规范说明。

    arXiv:2609.38335v1 Announce Type: cross  Abstract: Coding agents powered by large language models (LLMs) are evolving from making localized code changes to developing complete software repositories. However, evaluating repository-scale generation remains challenging: tasks must demand system-level reasoning while ensuring that all evaluated behaviors are precisely specified and independent of any particular implementation. We introduce E2E-SWE, a benchmark for evaluating whether coding agents can build complete, functional software repositories end to end. E2E-SWE contains 186 whole-repository generation tasks spanning 11 programming languages. Given only a natural-language specification and an empty workspace, an agent must implement a complete, installable project that satisfies a comprehensive suite of hidden tests. Each task is constructed by a software engineer in collaboration with an LLM; together, they develop the test suite and a corresponding implementation-independent specif
    
[^22]: Zero2Repo：编程智能体能否从零开始构建代码仓库？

    Zero2Repo: Can Coding Agents Build Repositories from Scratch?

    [https://arxiv.org/abs/2609.38269](https://arxiv.org/abs/2609.38269)

    提出 Zero2Repo 基准，通过语言无关的自动化流水线将真实的开源项目转化为“从零构建完整代码仓库”的任务，并以可执行验收测试和对抗性验证来严格评估编程智能体的从零构建能力。

    

    编程智能体越来越多地被要求从零构建软件，而非修补现有软件，然而针对从零构建代码仓库的基准测试大多局限于单一语言，并且依赖人工整理的任务。我们提出了 Zero2Repo，这是一个基准测试：智能体接收一份产品需求文档、一个接口契约和一个空的工作区，必须在该项目原生生态系统中交付一个完整的代码仓库。任务由一个与语言无关的任务生成流水线产出，该流水线将真实的、版本锁定的开源项目转换为行为规范、可复现的环境和隐藏的验收测试。每个任务都通过执行来验证：源自上游项目的参考实现必须通过测试，且对抗性验证必须证明这些测试能够拒绝错误的实现。评估在生产级编程智能体上进行，智能体运行于隔离容器中，在明确提交之前无法获得验收测试。

    arXiv:2609.38269v1 Announce Type: cross  Abstract: Coding agents are increasingly asked to build software rather than patch it, yet benchmarks for from-scratch repository construction are mostly limited to a single language and depend on manually curated tasks. We introduce Zero2Repo, a benchmark in which an agent receives a product requirements document, an interface contract, and an empty workspace, and must deliver a complete repository in the project's native ecosystem. Tasks are produced by a language-agnostic authoring pipeline that converts real, version-pinned open-source projects into behavioral specifications, reproducible environments, and hidden acceptance tests. Each task is validated by execution: a reference implementation derived from the upstream project must pass, and adversarial validation must show that the tests reject incorrect implementations. Evaluation runs production coding agents in isolated containers, withholds the acceptance tests until an explicit submiss
    
[^23]: 扩散语言模型应如何编辑代码？

    How Should Diffusion Language Models Edit Code?

    [https://arxiv.org/abs/2609.38257](https://arxiv.org/abs/2609.38257)

    扩散语言模型在给定正确编辑位置时能够生成协调的代码修改，但自行预测编辑位置的能力大幅下降；成功的代码编辑需要同时满足覆盖所有所需更改和精确划定编辑边界这两个要求。

    

    代码编辑要求模型决定在哪里进行更改、生成新内容，并保留其余所有内容。我们研究了掩码扩散语言模型如何在四种编辑接口之间分配这些职责：整文件重写、搜索替换、先定位后填充以及词元级编辑。在CanItEdit基准上的实验揭示了一个组合鸿沟：当提供正确的编辑位置时，扩散模型能够生成协调一致的更改，但当这些位置需要模型自行预测时，这种能力会大幅丧失。获取完整的原始代码有助于模型填充多个编辑区域，但并不能解决选择这些区域的困难。通过在保持生成模型和解码过程不变的情况下改变可编辑区域，我们识别出成功编辑的两个不同要求：覆盖所有需要的更改，并为它们划定精确的边界。缺少任何一个所需区域都会阻止…

    arXiv:2609.38257v1 Announce Type: cross  Abstract: Code editing requires a model to decide where to make changes, generate the new content, and preserve everything else. We study how masked diffusion language models divide these responsibilities across four editing interfaces: whole-file rewriting, search-and-replace, locate-then-infill, and token-level editing. Experiments on CanItEdit reveal a composition gap: diffusion models can generate coordinated changes when the correct edit locations are supplied, but much of this capability is lost when those locations must be predicted. Access to the intact original code helps the model fill multiple edit regions, yet does not resolve the difficulty of selecting those regions. By varying the editable regions while holding the generation model and decoding procedure fixed, we identify two distinct requirements for successful editing: covering every required change and placing precise boundaries around it. Missing a required region prevents th
    
[^24]: TomasuLLM：面向大语言模型智能体的乱序推测执行

    TomasuLLM: Out-of-Order Speculative Execution for LLM Agents

    [https://arxiv.org/abs/2609.38201](https://arxiv.org/abs/2609.38201)

    TomasuLLM提出了一种乱序推测执行运行时系统，让大语言模型智能体的工具调用在写时复制沙箱中提前执行并验证后按轨迹顺序提交，从而在不破坏正确性的前提下显著加速含长时工具调用的智能体任务。

    

    长时间运行的工具可能会主导编码智能体的延迟：编译器、测试套件和仓库命令需要几秒到几分钟的时间，而智能体在此期间处于空闲状态。这一观察到的停顿呈现出与推动乱序处理器发展的相同矛盾——顺序接口隐藏了那些本可以被预测并提前启动的工作，但推测性结果只有在它自身及其之前的每一步都得到验证后才可能变得可见。我们提出了TomasuLLM，这是一个以乱序（偏离轨迹顺序）方式执行智能体工具调用、同时保持任务执行正确性的运行时系统。它起草未来的动作，在隔离的写时复制沙箱中运行这些动作，追踪它们的依赖关系和影响，只有在对照已提交状态进行验证之后，才按轨迹顺序提交结果。在三个涵盖亚秒级到分钟级工具调用的基准测试中，TomasuLLM提升了所报告的基准测试均值，且加速效果随工具延迟增加而扩展：在100个SWE-bench Verified任务上提升1.31倍，在28个Termi（原文截断）……

    arXiv:2609.38201v1 Announce Type: new  Abstract: Long-running tools can dominate coding-agent latency: compilers, test suites, and repository commands take seconds to minutes while the agent idles. This observation stall presents the same tension that drove out-of-order processors -- asequential interface hides work that can be predicted and started early, but a speculative result may become visible only after it and every earlier step have been validated.   We present TomasuLLM, a runtime that executes agent tool calls out of trajectory order while preserving task-execution correctness. It drafts future actions, runs them in isolated copy-on-write sandboxes, traces their dependencies and effects, and commits results in trajectory order only after validation against committed state. Across three benchmarks spanning sub-second to minutes-long tool calls, TomasuLLM improves the reported benchmark means and scales with tool latency: 1.31x on 100 SWE-bench Verified tasks, 1.35x on 28 Termi
    
[^25]: 编辑器仅有只读权限：扩散语言模型中的正确性信号

    The Editor Has Read-Only Access: Correctness Signals in Diffusion Language Models

    [https://arxiv.org/abs/2609.36783](https://arxiv.org/abs/2609.36783)

    扩散语言模型的内部激活中编码了代码正确性信号，线性探针可有效检测该信号，但利用其进行生成引导并未带来可靠的性能提升。

    

    扩散语言模型通过反复更新部分掩码的序列来生成代码。我们探究其内部激活是否编码了代码正确性，以及该信息能否用于改进生成。在六个扩散模型上，线性探针能够区分通过和失败的生成尝试，且最强的读取信号通常出现在早期层之后。采用小幅度语义变异的对照实验支持了这些信号与正确性相关，而不仅仅是与表面风格相关。与模型置信度相比，探针的点估计并未表现出一致的优势。在所测试的引导设置中，向残差流中加入探针导出的方向并未带来可靠的改进，而施加相反方向则会降低性能。我们将这些观察结果与关于统计显著性或普遍无法引导生成的断言区分开来。补充方法、存档结果和代码记录了所测试的干预措施及其局限性。

    arXiv:2609.36783v1 Announce Type: new  Abstract: Diffusion language models generate code by repeatedly updating a partially masked sequence. We ask whether their internal activations encode code correctness and whether that information can improve generation. Across six diffusion models, linear probes distinguish passing from failing attempts, with the strongest reads generally appearing beyond the early layers. Controls using small semantic mutations support a connection to correctness rather than surface style alone. In comparisons with model confidence, probe point estimates offer no consistent advantage. Adding a probe-derived direction to the residual stream does not yield a dependable improvement in the tested steering settings, while the opposite direction degrades performance. We distinguish these observations from claims about statistical significance or a general inability to steer. Supplementary methods, archived results, and code document the tested interventions and the li
    
[^26]: 不透明指针下的神经符号间接调用分析

    Neuro-Symbolic Indirect-Call Analysis under Opaque Pointers

    [https://arxiv.org/abs/2609.33547](https://arxiv.org/abs/2609.33547)

    提出Facet，这是首个在不透明指针的LLVM IR上重建间接调用分发关系的分析，通过识别调用加载函数指针的结构体字段，并独立恢复通过初始化器、存储和聚合拷贝赋给该字段的函数，再按字段标识将二者关联。

    

    解析间接调用是构建C语言调用图的核心。诸如MLTA这类可扩展的基于类型的分析，利用LLVM IR中的类型信息将间接调用与赋给相应结构体字段的函数关联起来。然而，单一的被指类型往往无法准确表征指针所寻址的内存，并且LLVM 17移除了被指类型，转而采用不透明指针。因此，字段敏感的分析失去了匹配键。恢复被擦除的类型虽然能够找回匹配键，但仍然遗漏了类型所编码的关系：即程序将哪些函数赋给该字段。我们提出了Facet，据我们所知，这是第一个在不透明IR之上重建这种分发关系的分析。Facet识别间接调用从哪个结构体字段加载其函数指针，并分别通过初始化器、存储操作和聚合拷贝来恢复赋给该字段的函数，然后通过字段标识将两者连接起来。

    arXiv:2609.33547v2 Announce Type: replace  Abstract: Resolving indirect calls is central to call-graph construction for C. Scalable type-based analyses such as MLTA use type information in LLVM IR to associate indirect calls with functions assigned to the corresponding structure fields. However, a single pointee type often misrepresents the memory a pointer addresses, and LLVM 17 removed pointee types in favor of opaque pointers. Therefore, field-sensitive analyses lose their matching key. Recovering the erased types restores the matching key but still misses the relation that the type encoded: which functions the program assigns to the field. We present Facet, to our knowledge the first analysis that reconstructs this dispatch relation over opaque IR. Facet identifies the structure field from which an indirect call loads its function pointer. It separately recovers the functions assigned to that field through initializers, stores, and aggregate copies. It then joins the two by field i
    
[^27]: ASCEND：跨高性能计算集群与GPU工作站实现自主科学计算的个人AI智能体

    ASCEND: Personal AI Agents for Autonomous Scientific Computing Across HPC Clusters and GPU Workstations

    [https://arxiv.org/abs/2609.32868](https://arxiv.org/abs/2609.32868)

    ASCEND是一个运行在研究人员本地笔记本电脑上的AI智能体，通过安全认证连接远程调度Slurm集群和GPU工作站，无需设施级服务即可自主完成科学计算中的作业提交、故障诊断与恢复闭环。

    

    传统科学计算要求研究人员将计算意图转化为环境配置、资源请求和可执行作业，随后再根据调度器状态和应用程序日志诊断故障。我们提出了ASCEND（Autonomous Scientific Computing Engine and Novel Discovery，自主科学计算引擎与新发现），这是一个AI驱动的智能体接口，运行在研究人员自己的笔记本电脑上，通过多路复用的认证连接访问Slurm管理的集群和GPU工作站，其站点特定的执行策略由本地执行的工具进行检查；语言模型远程托管，不持有任何凭证。无需设施级服务：只需在每个资源上拥有账户即可，公共安装程序允许用户接入自己的其他Slurm集群或工作站。我们报告了四个记录在案的案例：(1) 智能体在植入的张量设备故障上完成了故障恢复闭环，包括作业提交、故障诊断与修复（摘要在此处截断）。

    arXiv:2609.32868v2 Announce Type: replace-cross  Abstract: Traditional scientific computing requires researchers to translate computational intent into environment configuration, resource requests, and executable jobs, then diagnose failures from scheduler state and application logs. We present ASCEND (Autonomous Scientific Computing Engine and Novel Discovery), an AI-powered agent interface that runs the agent on the researcher's own laptop, reaching Slurm-managed clusters and a GPU workstation over a multiplexed authenticated connection, with site-specific execution policies checked by locally executed tools; the language model is hosted remotely and holds no credentials. No facility-scale service is required: an account on each resource is sufficient, and the public installer lets users link additional Slurm clusters or workstations of their own. We report four recorded cases: (1) the agent closed a failure-recovery loop on a planted tensor-device fault, submitting, diagnosing, repa
    
[^28]: 规范先于生成：一项预注册的、五模型配对评估——关于规范框架对LLM生成代码在货币、时间、幂等性与访问控制任务中的效果

    Specification Before Generation: A Pre-Registered, Five-Model Paired Evaluation of a Specification Frame for LLM-Generated Code in Money, Time, Idempotency, and Access Tasks

    [https://arxiv.org/abs/2609.23270](https://arxiv.org/abs/2609.23270)

    该论文通过预注册的五模型配对实验证明，在提示词前附加一份267词的固定规范框架，能显著改善LLM生成的后端代码在货币算术、时间处理、重试幂等性和访问控制等最关键缺陷类别上的质量。

    

    arXiv:2609.23270v1 公告类型：新 摘要：大型语言模型生成的代码通过安全检查的比率在四年间几乎没有提升。在受监管的后端系统中，最关键的缺陷类别是货币算术、时间处理、重试安全性和访问控制。团队的应对手段是指令文件，然而迄今规模最大的指令文件对照研究却未发现其带来任何收益。本文检验了一个更聚焦的想法：当提示词中携带一份规范——即一个固定的前言，声明结果必须满足哪些条件——时，生成的代码会得到改善。我们预注册了假设、反驳检验、分析代码和一次性生成规则，然后将来自金融、医疗和保险实践的50个真实后端任务，分别通过来自五个厂商体系的五个前沿模型运行，每个任务执行两次：一次为裸提示，一次为在提示前附加一份267词的已填写规范框架。九个基于AST的确定性检查器对输出进行评分。对规范框架一无所知的Bandit安全扫描器对（摘要在此处截断）

    arXiv:2609.23270v1 Announce Type: new  Abstract: Code generated by large language models passes security checks at a rate that has barely moved in four years. In regulated backends, the defect classes that matter most are money arithmetic, time handling, retry safety, and access control. Teams answer with instruction files, yet the largest controlled study of instruction files to date found no benefit. This paper tests a narrower idea: generated code improves when the prompt carries a specification, a fixed preamble stating what must be true of the result. We pre-registered hypotheses, refuters, analysis code, and a one-shot generation rule, then ran 50 realistic backend tasks from finance, healthcare, and insurance practice through five frontier models from five vendor lineages, each task twice: bare, and preceded by a 267-word filled specification frame. Nine deterministic AST-based checkers scored the outputs. The Bandit security scanner, which knows nothing of the frame, scored the
    
[^29]: LadderTeam：双智能体阶梯式引出框架

    LadderTeam: Dual-Agent Laddering Elicitation Framework

    [https://arxiv.org/abs/2608.17029](https://arxiv.org/abs/2608.17029)

    本文提出了LadderTeam框架，通过双智能体LLM架构自动化UX线框图访谈，克服了传统阶梯式访谈的手动成本和可扩展性限制。

    

    arXiv:2608.17029v1 公告类型：新 摘要：从最终用户那里引出详细且可操作的软件需求，是软件产品或应用迭代开发中的关键阶段。为了确保收集到的反馈详细且可操作，软件团队可以利用阶梯式访谈技术。虽然该技术能有效确保从软件反馈中获得细粒度且可操作的项目，但这些访谈受到若干限制。它们传统上是手动过程，伴随时间和财务负担，限制了可扩展性；访谈者必须在深入探查与应对受访者行为和文化的约束之间取得平衡。为解决这些限制，我们提出了 \textbf{LadderTeam}，一个开放、可复现的框架，利用双智能体大语言模型（LLM）架构自动化UX线框图访谈。一个主动的访谈者智能体执行三种探查策略之一（ACV、5-Why和JTBD），以引出可操作的软件需求。

    arXiv:2608.17029v1 Announce Type: new  Abstract: Eliciting detailed and actionable software requirements from end-users is a critical phase in the iterative development of a software product or application. To ensure the feedback collected is detailed and actionable, software teams can leverage the laddering interview technique. While effective for ensuring granular and actionable items from the software feedback, these interviews are subject to several limitations. They are traditionally a manual process associated with a time and financial burden, limiting scalability; interviewers must balance probing for depth while managing interviewee behavioral and cultural constraints. To address these limitations, we present \textbf{LadderTeam}, an open, reproducible framework that automates UX wireframe interviews using a dual-agent Large Language Model (LLM) architecture. An active interviewer agent executes one of three probing strategies (ACV, 5-Whys, and JTBD) to elicit actionable softwar
    
[^30]: ARCHER：面向可执行法规的智能体规则与合规框架

    ARCHER: Agentic Rule and Compliance Harness for Executable Regulations

    [https://arxiv.org/abs/2607.25566](https://arxiv.org/abs/2607.25566)

    ARCHER是一个测试驱动、确定性编排的多智能体程序合成框架，能从法规实践准则中自动生成可审计的验证代码，从而实现透明、可适应且可扩展的建筑合规检查。

    

    验证建筑合规性需要针对大型建筑信息模型（BIM）设计核查数千条规则，这一过程费力、成本高昂且难以扩展。现有的自动化合规检查器（ACC）通常难以在不同场景间泛化，因为它们往往是为高度特定的规则集和用例而开发的。此外，许多ACC是专有的，这意味着底层验证代码不会向最终用户开放，因此用户无法验证其监管意图能否被准确捕捉。我们提出了ARCHER（面向可执行法规的智能体规则与合规框架），这是一个测试驱动、确定性编排的多智能体程序合成框架，能够从监管实践准则中生成可审计的验证代码，从而实现透明、可适应且可扩展的合规检查。为了刻画智能体合成发挥作用的关键因素，我们评估了……（摘要原文在此处截断）

    arXiv:2607.25566v2 Announce Type: replace-cross  Abstract: Verifying building compliance requires validating thousands of rules against large Building Information Modeling (BIM) designs, which is laborious, capital-intensive, and unscalable. Existing Automated Compliance Checkers (ACCs) are often difficult to generalize across different scenarios, as they are typically developed for highly specific rule sets and use cases. In addition, many ACCs are proprietary, meaning the underlying verification code is not released to end users, so users cannot verify whether their regulatory intent can be accurately captured. We introduce ARCHER (Agentic Rule and Compliance Harness for Executable Regulations), a test-driven, deterministically orchestrated multi-agent program-synthesis harness that generates auditable verification code from regulatory Codes of Practice, enabling transparent, adaptable, and scalable compliance checking. To characterize what makes agentic synthesis work, we evaluate a
    
[^31]: 单一文化漫游指南：AI 同质化的是语法，而非（必然的）语义

    The Hitchhiker's Guide to Monoculture: AI Homogenizes Syntax, Not (Necessarily) Semantics

    [https://arxiv.org/abs/2607.13077](https://arxiv.org/abs/2607.13077)

    通过分析 Kaggle 竞赛提交数据，论文发现 AI 虽然使代码语法高度同质化（如随机种子值向 42 趋同），但语言表达的趋同并不必然意味着思想或语义层面的趋同。

    

    大型语言模型（LLM）被广泛报道会同质化人类的表达与思想。然而，我认为语言上的趋同并不必然意味着思想上的趋同，且现有证据很少对这两者加以区分。我通过对软件开发领域的研究来证明这一点——该领域是 AI 助手扩散最快的领域，且语法可以与语义（即概念方法或意图）较为容易地分离。利用 2019 年至 2026 年年中的 Kaggle 竞赛提交数据，我首先记录了代码向随机种子值 42 趋同的现象，这与 LLM 强化了一项长期存在的编程文化惯例相一致，该惯例源自道格拉斯·亚当斯的喜剧小说《银河系漫游指南》。随后，我更广泛地衡量了代码语法与方法层面的同质化程度，使用词频-逆文档频率（TF-IDF）n-gram 表示法来量化同一竞赛内代码提交之间的相似性，该表示法捕捉的是表面……（原文摘要在此处截断）

    arXiv:2607.13077v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have been widely reported to homogenize human expression and thought. However, I argue that convergence in language need not imply convergence in ideas, and existing evidence rarely distinguishes between the two. I demonstrate this through a study of software development, where AI assistants have diffused fastest and where syntax can be readily separated from semantics (conceptual approach or intent). Using Kaggle contest submissions from 2019 to mid-2026, I first document convergence toward the random seed value 42, consistent with LLMs reinforcing a longstanding programming-culture convention associated with Douglas Adams' comedy novel The Hitchhiker's Guide to the Galaxy. I then measure homogenization in code syntax and approach more generally, quantifying within-contest code submission similarity using term-frequency-inverse-document-frequency (TF-IDF) n-gram representations, which capture surfa
    
[^32]: ITHICA：针对缺陷引起的静默数据损坏的线程内指令检查方法

    ITHICA: Intra-Thread Instruction Checking Approach for Defect-Induced Silent Data Corruptions

    [https://arxiv.org/abs/2605.15638](https://arxiv.org/abs/2605.15638)

    ITHICA通过在线程内插入指令级错误检查（利用指令复制与输出比较），借助同一指令执行结果不一致这一缺陷特征，能够将任意程序转化为检测制造缺陷引起的静默数据损坏的功能测试并定位受影响指令。

    

    超大规模云服务商正报告静默数据损坏（SDCs），据推测这些损坏由硅制造缺陷引起，被视为对数据中心可靠性的威胁。为了支持数据中心检测有缺陷的CPU服务器的测试工作，本文提出了ITHICA，这是一种方法和工具，通过插入线程内、指令级的错误检查，主要利用指令复制和输出比较，能够从任意程序自动生成针对缺陷引发错误的功能测试。我们的关键洞察是：最有害的缺陷（即最有可能逃过制造测试的缺陷）会导致不一致的错误——在同一线程内，同一条指令在相同输入下执行两次，可能根据其运行的执行上下文产生不同的架构输出。通过利用这一洞察，ITHICA独特地使任意程序都能作为测试使用，并能定位受影响的指令。

    arXiv:2605.15638v2 Announce Type: replace-cross  Abstract: Hyperscalers are reporting silent data corruptions (SDCs), presumed to be caused by silicon manufacturing defects, as a threat to datacenter reliability. To support datacenter testing efforts to detect defective CPU servers, this paper presents ITHICA, an approach and tool for automatically generating functional tests for defect-induced errors from arbitrary programs by inserting intra-thread, instruction-level error checks, primarily leveraging instruction duplication and output comparison. Our key insight is that the most pernicious defects (those most likely to escape manufacturing testing) cause inconsistent errors: two executions of the same instruction given the same inputs within the same thread can produce different architectural outputs depending on the execution context in which they run. By exploiting this insight, ITHICA uniquely enables arbitrary programs to serve as tests and localizes affected instructions concur
    
[^33]: 工业界的智能体AI：采用水平与部署障碍

    Agentic AI in Industry: Adoption Level and Deployment Barriers

    [https://arxiv.org/abs/2605.14675](https://arxiv.org/abs/2605.14675)

    通过对12家公司16名从业者的访谈研究发现，工业界智能体AI的生产应用目前仅处于六级成熟度框架的1-3级，其进一步自动化部署受制于由信息不对称与资格认证缺失所构成的能力-部署验证差距。

    

    智能体AI（Agentic AI）正在进入软件工程工作流，但关于其从实验性能力过渡到生产应用的实证证据仍然有限。我们报告了一项定性访谈研究，访谈了来自12家公司的16名从业者，并以六级成熟度框架作为分析视角。所报告的生产实践对应于1-3级，而来自四家公司的参与者报告了超出生产集成应用的实验性能力。在各个案例中，四个先前已被识别的障碍反复出现：上下文管理、在专有内容上的性能表现、非确定性与资格认证，以及数据机密性。我们将它们的相互作用综合为一个“能力-部署验证差距”，该差距由两个相互依存的维度构成：信息不对称和资格认证缺失。本研究据此刻画了所报告的采用实践，并解释了是什么制约了（代表性案例中）进一步的智能体自动化。

    arXiv:2605.14675v2 Announce Type: replace  Abstract: Agentic AI is entering software engineering workflows, but empirical evidence on its transition from experimental capability to production use remains limited. We report a qualitative interview study with 16 practitioners from 12 companies, using a six-level maturity framework as an analytical lens. Reported production practices corresponded to Levels 1-3, while participants in four companies reported experimental capabilities beyond production-integrated use. Across the cases, four previously identified barriers recurred: context management, performance on proprietary content, non-determinism and qualification, and data confidentiality. We synthesize their interaction as a capability-deployment verification gap structured by two interdependent dimensions: information asymmetry and qualification absence. The study thereby characterizes reported adoption practices and explains what constrains further agentic automation in the represen
    
[^34]: 从图表到代码：面向多模态模型的分层基准测试

    From Charts to Code: A Hierarchical Benchmark for Multimodal Models

    [https://arxiv.org/abs/2510.17932](https://arxiv.org/abs/2510.17932)

    Chart2Code是首个从用户视角出发设计的图表转代码分层基准，通过图表复现、图表编辑和长表格转图表三个难度递增的层级（共2,023个任务、22种图表类型），系统性评估大型多模态模型的图表理解与代码生成能力。

    

    我们提出了Chart2Code，一个用于评估大型多模态模型（LMMs）图表理解与代码生成能力的新基准。Chart2Code明确地从用户驱动的视角进行设计，涵盖多样化的真实世界场景，并逐步提升任务难度。它包含三个层级：第一级（图表复现）根据参考图和用户查询复现图表；第二级（图表编辑）涉及复杂的修改操作，例如更改图表类型或添加元素；第三级（长表格到图表生成）要求模型按照用户指令将冗长且信息密集的表格转换为忠实的图表。据我们所知，这是首个既反映实际图表转代码使用场景、又能系统性扩展任务复杂度的分层基准。总计而言，Chart2Code包含涵盖22种图表类型的2,023个任务，并配有从多个层面评估（原文此处截断）的多层次评估指标。

    arXiv:2510.17932v5 Announce Type: replace-cross  Abstract: We introduce Chart2Code, a new benchmark for evaluating the chart understanding and code generation capabilities of large multimodal models (LMMs). Chart2Code is explicitly designed from a user-driven perspective, capturing diverse real-world scenarios and progressively increasing task difficulty. It consists of three levels: Level 1 (Chart Reproduction) reproduces charts from a reference figure and user query; Level 2 (Chart Editing) involves complex modifications such as changing chart types or adding elements; and Level 3 (Long-Table to Chart Generation) requires models to transform long, information-dense tables into faithful charts following user instructions. To our knowledge, this is the first hierarchical benchmark that reflects practical chart2code usage while systematically scaling task complexity. In total, Chart2Code contains 2,023 tasks across 22 chart types, paired with multi-level evaluation metrics that assess b
    
[^35]: 由约束表达力中间表示驱动的三维软件合成

    3D Software Synthesis Driven by Constraint-Expressive Intermediate Representation

    [https://arxiv.org/abs/2507.18625](https://arxiv.org/abs/2507.18625)

    提出了Scenethesis，一种基于领域特定语言ScenethesisLang（作为约束表达力中间表示）的需求敏感3D软件合成方法，实现了用户规格说明与生成的3D软件之间的形式化可追溯性，并支持对软件中特定元素的细粒度修改与控制。

    

    图形用户界面（UI）软件已经经历了从传统的二维（2D）桌面/网页/移动界面到空间三维（3D）环境的根本性转变。尽管现有工作在自动化2D软件生成方面（如HTML/CSS和移动应用界面代码合成）取得了显著成功，但3D软件的生成仍然缺乏充分探索。当前的3D软件生成方法通常将3D环境作为一个整体来生成，无法修改或控制软件中的特定元素。此外，这些方法难以处理现实世界中固有的复杂空间和语义约束。为应对这些挑战，我们提出了Scenethesis，一种新颖的需求敏感的3D软件合成方法，它在用户规格说明与生成的3D软件之间保持形式化的可追溯性。Scenethesis构建于ScenethesisLang之上，这是一种特定领域语言……

    arXiv:2507.18625v3 Announce Type: replace-cross  Abstract: Graphical user interface (UI) software has undergone a fundamental transformation from traditional two-dimensional (2D) desktop/web/mobile interfaces to spatial three-dimensional (3D) environments. While existing work has made remarkable success in automated 2D software generation, such as HTML/CSS and mobile app interface code synthesis, the generation of 3D software still remains under-explored. Current methods for 3D software generation usually generate the 3D environments as a whole and cannot modify or control specific elements in the software. Furthermore, these methods struggle to handle the complex spatial and semantic constraints inherent in the real world. To address the challenges, we present Scenethesis, a novel requirement-sensitive 3D software synthesis approach that maintains formal traceability between user specifications and generated 3D software. Scenethesis is built upon ScenethesisLang, a domain-specific lan
    
[^36]: 移动应用研究中与GDPR相关的隐私问题：一项系统性文献综述

    GDPR-Relevant Privacy Concerns in Mobile Apps Research: A Systematic Literature Review

    [https://arxiv.org/abs/2411.19142](https://arxiv.org/abs/2411.19142)

    本文通过系统性文献综述，首次对移动应用领域GDPR相关隐私问题的现有研究进行了描述、分析和分类，填补了该领域缺乏二次研究的空白。

    

    《通用数据保护条例》（GDPR）被视为欧盟（EU）隐私和数据保护标准的基准。早在其2018年生效之前，软件工程（SE）文献中就已开展了大量研究，探讨GDPR隐私需求的获取、表示和验证。世界上任何地方部署的软件系统，只要处理欧盟居民的个人数据，就必须遵守GDPR。移动应用程序（apps）在这方面也不例外。随着移动应用的日益普及及其对个人数据需求的不断增长，隐私问题在软件工程界引起了更多关注。尽管关于移动应用中GDPR相关隐私问题的文献十分丰富，但目前尚无描述、分析和归类当前研究重点的二次研究。因此，研究空白和持续存在的挑战尚未得到解决……

    arXiv:2411.19142v4 Announce Type: replace  Abstract: The General Data Protection Regulation (GDPR) is considered as the benchmark in the European Union (EU) for privacy and data protection standards. Since before its entry into force in 2018, substantial research has been conducted in the software engineering (SE) literature investigating the elicitation, representation, and verification of GDPR privacy requirements. Software systems deployed anywhere in the world must comply with GDPR as long as they handle personal data of EU residents. Mobile applications (apps) are no different in that regard. With the growing pervasiveness of mobile apps and their increasing demand for personal data, privacy concerns have acquired further interest within the SE community. Despite the extensive literature on GDPR-relevant privacy concerns in mobile apps, there is no secondary study that describes, analyzes, and categorizes the current focus. Research gaps and persistent challenges are thus left unn
    

