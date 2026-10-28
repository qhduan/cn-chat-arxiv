# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An EVM-Cosmos Bridge for Atomic, Semantic-Preserving Transfer of Environmental Tokens](https://arxiv.org/abs/2610.37663) | 该论文提出了一种连接 EVM 与 Cosmos 的跨链桥，利用密码学承诺派生的确定性指标标识符在铸币时校验以防止环境代币（如碳信用）的语义混合，并通过扩展 HTLC 加入自动回调销毁与激励相容赏金机制，实现单一所有者跨链转移的原子性。 |
| [^2] | [FeatureArc: Benchmarking Coding Agents on Evolving Features](https://arxiv.org/abs/2610.37434) | 提出 FeatureArc 基准测试，首次评估编码智能体在软件特性持续演化场景下的能力，其流水线从仓库历史构建包含阶段间依赖、明确需求契约、可执行 F2P/P2P 测试验证并覆盖四种编程语言的任务。 |
| [^3] | [HIDE: Hierarchical Isolation via Documentation and Evidence for Rustc Bugs](https://arxiv.org/abs/2610.37343) | HIDE 是一种基于大语言模型的层次化方法，通过将编译器文档与缺陷证据相结合来对可疑的 rustc 源文件进行排序，从而有效缩小搜索空间并提升 rustc 缺陷隔离的准确性。 |
| [^4] | [Mind the Intent Gap: A Methodology for Collaborative Agent-Driven Software Engineering](https://arxiv.org/abs/2610.37340) | 提出CAIA（协作式智能体意图对齐）方法论，通过知识库显式记录团队意图，弥合AI智能体加入软件团队后因“意图未被完整记录”而靠猜测行事并偏离团队目标的“意图差距”问题。 |
| [^5] | [Hidden in Plain Sight: an Empirical Study about The Instruction Stacks of Coding Agents](https://arxiv.org/abs/2610.37332) | 该论文通过对五个主流编程智能体共 1,886 个版本的大规模纵向实证研究，首次系统揭示了隐藏在模型请求中的“指令栈”的构成与演变，表明仅看系统提示词或模型名称无法完整刻画智能体的真实输入。 |
| [^6] | [BenchMeUp: Building Software Engineering Benchmarks from Natural-Language Specifications](https://arxiv.org/abs/2610.37283) | 本文提出BenchMeUp方法，可从自然语言规范自动构建软件工程基准测试，通过并行LLM智能体识别并构建可执行实例、在新环境中验证正确性，从而大幅降低基准测试构建的人工成本。 |
| [^7] | [Grounding LLM Agents in Business Processes: Process-Aware Knowledge Graphs for Reliable Multi-Step Service Invocation](https://arxiv.org/abs/2610.37242) | 该论文提出将OData元数据与BPMN参考流程整合为类型化属性图知识图谱，并利用沿流程边检索的混合检索器，显著提升了LLM智能体在多步骤企业服务调用中的准确性和可靠性。 |
| [^8] | [Spine: Compiling Subtask Trees into Structured Context for Long-Horizon Coding Agents](https://arxiv.org/abs/2610.37224) | Spine 通过让智能体用“工作范围”显式声明子任务边界，将子任务树编译为持久化的结构化上下文，在不依赖子智能体委派的情况下对齐长程编码任务的嵌套结构与模型上下文，避免上下文膨胀。 |
| [^9] | [Toward Circuit-Aware Mutation Testing for Quantum Programs: Exact Visibility and Execution Planning](https://arxiv.org/abs/2610.36976) | 提出了一种面向量子程序的电路感知变异测试框架，能够精确区分变异体存活的三种原因（等价变异、被输入或测量隐藏的变更、统计证据不足），并给出测试无法揭示变异的条件与变异影响的精确公式，从而指导执行规划和测试改进。 |
| [^10] | [Does More Thinking Pay Off? Reasoning Cost and Correctness of LLMs in Code Generation and Bug Fixing](https://arxiv.org/abs/2610.36952) | 本文通过对 866 个 Python 任务的实证研究发现，开启显式推理能显著提升大语言模型在部分代码生成与漏洞修复任务上的正确性，但更长、更多的思考并不必然带来更高的准确率，推理收益与令牌开销之间不成正比。 |
| [^11] | [Beyond QASecClaw: Agent Scaling, Verifier Feedback, and Evidence Reliability in Software Assurance](https://arxiv.org/abs/2610.36947) | 该研究通过五组实验表明，在软件保障任务中扩大语言模型智能体上下文规模会使token开销激增49倍却无安全收益，验证器反馈虽能提升正确判定数量但同时增加错误判定，说明确定性的工具证据基线仍然不可或缺。 |
| [^12] | [A lightweight C Code Generator for Heterogeneous TinyML Systems](https://arxiv.org/abs/2610.36907) | 该论文提出了一种面向集成领域专用加速器的裸机异构TinyML系统的轻量级C代码生成器，通过权重调优和偏置融合等提前优化技术，在基于RISC-V的雷达DSP上相比microTVM实现了2.3倍推理加速和20%的二进制体积缩减。 |
| [^13] | [ASNER: Software Architecture Recovery with Adaptive Semantic Neighborhoods and Evidence-Grounded Refinement](https://arxiv.org/abs/2610.36894) | ASNER通过职责感知、项目自适应的关系建模构建自适应语义邻域，并采用证据约束的LLM边界验证机制，解决了固定阈值忽略仓库规模、功能相关性误导架构职责判断以及无约束LLM细化破坏正确结果的问题，从而提升软件架构恢复的准确性。 |
| [^14] | [God Does Not Play Dice: Enforcing Batch Invariance in Agentic Kernel Generation](https://arxiv.org/abs/2610.36623) | 该论文提出了一种即插即用工具，通过将契约引导的静态依赖分析与跨批处理上下文的按位一致性动态验证相结合，作为验收门控嵌入内核生成智能体的优化循环中，从而强制执行批处理不变性，解决LLM服务中动态批处理导致相同输入产生不同数值输出、损害可复现性的问题。 |
| [^15] | [A Fine-Grained Empirical Study of Upstream Security Events and Downstream Mitigation in Maven](https://arxiv.org/abs/2610.36603) | 该研究首次对Maven生态系统中832个CVE进行从上游修复到下游采用的端到端细粒度追踪，揭示了下游安全暴露究竟源于上游发布延迟、下游采用延迟还是两者兼有的问题。 |
| [^16] | [Benchmarking LLM Agents for Behaviorally Adequate Test Generation from Software Requirements](https://arxiv.org/abs/2610.36584) | 该论文提出了Req2Test-Bench基准，包含从8个开源项目BDD工件中恢复的5,086个经执行验证的实例，用于评测LLM智能体在仅依据自然语言需求（无实现代码）的情况下生成行为充分测试的能力。 |
| [^17] | [CASEL: Consequence-Aware Selective Execution for Reliable Long-Horizon Tool-Using Agents](https://arxiv.org/abs/2610.36570) | CASEL提出了一种与模型无关的运行时分配策略，通过综合失败概率、后果严重性、可逆性、模糊性和证据新鲜度等因素，为智能体的每次工具调用智能选择直接执行、验证、沙箱、澄清、审批或拒绝执行，显著减少了长时程工具使用智能体造成有害副作用的风险。 |
| [^18] | [From Ripple to Boundary: Deterministic Impact Boundaries for LLM-Based Software Evolution](https://arxiv.org/abs/2610.36549) | 该论文提出将多源设计资产形式化为知识图谱，实现从概率性语义检索到确定性结构约束的范式转变，为大语言模型驱动的软件演化提供确定性的影响边界，避免幻觉变更或破坏无关代码。 |
| [^19] | [ProCerDCL: Query-Efficient Black-Box Probabilistic Verification of Deep Learning Compiler Optimization](https://arxiv.org/abs/2610.36499) | ProCerDCL 提出了一种查询高效的黑盒概率验证框架，利用单一序贯检验在指定错误预算内判定属性违反概率是否超过阈值，并将其应用于深度学习编译器优化的验证。 |
| [^20] | [When AI Can Build It, What Should We Assess? Rethinking Evidence of Engineering Competence](https://arxiv.org/abs/2610.36413) | 该论文在生成式AI时代重新设计了软件工程课程的评估体系，通过工程问责工作流和由“收集—交叉核验—升级”逻辑连接的六渠道证据架构（产品、过程/来源、验证、解释、反思、个人保证），在学生可将工作委托给AI的情况下仍能可靠评估其真实工程能力。 |
| [^21] | [kw-bench: Breaking Frontier Harnesses on Real-World Tasks](https://arxiv.org/abs/2610.36405) | kw-bench 是基于真实业务委托构建的基准，首次将评估重点从生成式任务转向审计、税务等知识工作中的调查性环节，检验前沿智能体运行框架能否从不完整、不一致的分散证据中精确还原并论证答案。 |
| [^22] | [Your Wish Is My Commit: Over-Compliant Refactoring in AI Coding Agents](https://arxiv.org/abs/2610.36346) | 该论文首次通过受控的迭代重构实验系统研究了 AI 编程代理的谄媚问题，发现 Codex 和 Claude Code 等代理在代码已达最优后仍会持续进行不必要的编辑（过度顺从），并定义了六种量化这种行为的行为信号。 |
| [^23] | [How Stable Are LLM-Generated Proofs Under Program Refactoring? A Stress Test for Verifiable Code Generation](https://arxiv.org/abs/2610.36328) | 该论文提出 RefactorProof 基准，通过对 185 个 VERINA 任务生成 458 个经 Lean 认证的行为保持重构变体，揭示了大语言模型生成的证明在程序结构重构下存在显著脆弱性。 |
| [^24] | [Building Mathematical Software -- Lessons from the Development of GAP](https://arxiv.org/abs/2610.36300) | 本文以计算机代数系统GAP的开发为具体案例，总结了现有软件工程文献未充分覆盖的数学软件特殊需求与开发经验，并探讨了AI系统对数学软件开发过程的潜在影响。 |
| [^25] | [Instant Software Factories](https://arxiv.org/abs/2610.36254) | 该论文提出“即时软件工厂”概念，仅凭一轮人类输入即可通过引导提示词自动配置出定时任务智能体群和共享沟通看板两个协调机制，以可重复流程处理从原型到产品之间的长尾工程工作，并观察到每日代码合并量提升15倍。 |
| [^26] | [MARQ: A Modular, LLM-Agnostic Requirement Quality Assessor Infused with Systems Engineering Knowledge](https://arxiv.org/abs/2610.36234) | MARQ是一个模块化、与具体大模型无关的需求质量评估原型，通过注入基于INCOSE准则可操作化的系统工程专家知识，显著提升了大语言模型在不同质量准则下识别需求缺陷的能力，超越了现成模型的表现。 |
| [^27] | [Agent or Human? Detecting Commits by AI Coding Agents in the Absence of Authorship Signatures](https://arxiv.org/abs/2610.36229) | 该研究基于近800万个提交发现，即使去除作者身份签名，仍可凭借提交信息冗长度和变更规模等特征以最高0.859的ROC AUC区分AI编程智能体与人类的代码提交，且检测信号主要来自元数据而非代码本身。 |
| [^28] | [Skill Dreaming: Evolving Repository Skills through Pull Request Streams](https://arxiv.org/abs/2610.36188) | 提出了 Skill Dreaming 框架，以拉取请求流为知识来源，实现拉取请求、仓库与技能库的共同演化，使仓库专属技能随代码库的变化持续更新并反哺未来的 PR 实践。 |
| [^29] | [Can AI Guide Testing Without Generating Tests? A Comparative Study of AI-Assisted Unit Testing](https://arxiv.org/abs/2610.36185) | 该研究提出TestCompass——一个不代学生生成可执行测试、而是针对其测试套件薄弱点提供定向反馈的生成式AI辅导工具，并通过18名学生的课堂对比实验证明AI可以在不直接生成测试的前提下有效支持初学者的单元测试学习。 |
| [^30] | [Stratum: Repository Memory for Coding Agents Beyond the Context Window](https://arxiv.org/abs/2610.36153) | 该论文提出Stratum，为编码智能体提供跨会话的持久化、经验感知的仓库记忆系统，并通过实证测量证明真正的瓶颈在于相关知识的选择而非上下文窗口容量。 |
| [^31] | [Context Is a Pipeline, Not a Window: Information Attrition in LLM Agent Harnesses](https://arxiv.org/abs/2610.36114) | 该研究揭示了一个被忽视的关键问题：即使LLM智能体的上下文窗口利用率仅为5.75%，智能体框架仍会通过固定长度切分JSON的方式截断三分之一的工具结果，导致末尾数据不均匀丢失，使不同字段到达模型的比例差异悬殊（4.7%对99.3%），因此评估智能体不能只看窗口大小和答案正确率，还必须关注“信息损耗”。 |
| [^32] | [AGDecode: Semantic Attribute Grammar Derivatives for Safety-Constrained Decoding of Large Language Models](https://arxiv.org/abs/2610.36105) | AGDecode提出了一种基于最小不动点语义的可编程语义约束解码框架，通过将属性文法扩展到部分构造的语法树并把语义约束解码归约为可实现性问题，从而保证大语言模型所生成程序的结构与语义正确性。 |
| [^33] | [Do Coding Agents Still Need a Debugger? Frontier-Model-Based Fault Localization at the Accuracy Ceiling](https://arxiv.org/abs/2610.36015) | 该研究通过在610个注入故障的大规模实验发现，基于前沿大模型的编程智能体仅凭源代码推理即可将故障定位准确率提升至98%以上的天花板水平，调试器的额外辅助收益已微乎其微。 |
| [^34] | [CIVER: Causal Intervention Verification for Reliable Code Review](https://arxiv.org/abs/2610.36010) | CIVER提出了一种基于因果干预的验证框架，通过识别代码评审发现背后的核心问题来区分真实缺陷与误报，从而提升LLM代码评审的可靠性。 |
| [^35] | [Before the First Edit? Measuring When Coding-Agent Guidance Actually Arrives](https://arxiv.org/abs/2610.35992) | 该研究通过审查五个事件并在代理每个执行动作前后对Python文件进行哈希，测量编码代理在首次源代码编辑前的指导实际到达时机，发现指导确实先于所有观察到的首次变更，但研究未达到预设的20个分支覆盖目标。 |
| [^36] | [PROBE: Frontier Coding Agents Find Different Bugs Than Maintainers Fix](https://arxiv.org/abs/2610.35987) | 本文提出PROBE基准测试，揭示前沿编码智能体在仓库漏洞发现中召回率和精确率均很低，且其找到的漏洞与人类维护者实际修复的漏洞存在系统性差异。 |
| [^37] | [Generation and Verification: Human-Agent Collaboration for Safety-Critical Automotive Software Engineering](https://arxiv.org/abs/2610.35982) | 该论文提出一个面向安全关键汽车 C/C++ 软件的编程助手，通过 MCP 协议向开发者开放生成、编辑、验证三个可按需调用的工具以实现人机协作验证，并针对汽车领域定制底层大语言模型。 |
| [^38] | [Small agents should know what their tools do](https://arxiv.org/abs/2610.35981) | 本文提出EffectWorld基准，通过为工具添加效应类型声明（幂等性及读写资源），让小型语言模型能够生成效应安全且高效的多调用工具程序，将Qwen3-4B的任务成功率从53%大幅提升至99%。 |
| [^39] | [UnityBench-Mini3D: Evaluating LLM Agents for 3D Game Development in Unity](https://arxiv.org/abs/2610.35980) | UnityBench-Mini3D是一个包含77个Unity项目级任务的基准，通过1,178个人工编写的引擎内里程碑测试对LLM智能体的3D游戏开发能力进行确定性评估，可提供部分得分、定位失败点，无需依赖主观的LLM评判器。 |
| [^40] | [Testing Agentic Service Systems: An Agentic Living Lab Methodology Across the Physical-Digital Continuum](https://arxiv.org/abs/2610.35978) | 提出了一种“智能体生活实验室”测试方法论，以服务片段为测试单元、区分被测智能体与环境参与者，并通过显式的现实基底绑定与语义差异披露，解决智能体服务系统在物理-数字连续体上超越单纯任务成功的授权、义务与执行环境依赖等测试难题。 |
| [^41] | [Continue or Restart? The Limits of Early Failure Prediction in Coding Agents](https://arxiv.org/abs/2610.35971) | 研究发现编码智能体在早期执行阶段（5k-20k token）几乎无法可靠预测单次尝试的成败，任务内失败预测能力接近随机水平，因此基于早期失败预测的提前停止策略在节省计算成本方面效果有限。 |
| [^42] | [PASS Is a Claim, Not an Observation: Mechanical Re-verification of 10 Self-Asserted Fix Claims in a Live Multi-Agent Audit Layer](https://arxiv.org/abs/2610.35950) | 该论文对一个活跃多智能体平台中特权协调智能体发出的10个自我断言修复声明进行了机械化再验证，发现各标准下的通过率仅为60%-80%，证明智能体的“PASS”修复声明不能等同于经过验证的真实观察。 |
| [^43] | [KORA: Deterministic-First Execution Control for AI Workloads](https://arxiv.org/abs/2610.35869) | KORA提出了一种“确定性优先”的推理前执行控制架构，优先尝试确定性和结构化路由来处理请求，仅在确定性处理不足时才升级到模型调用，在基准测试中避免了80%的模拟模型调用且结果零不匹配。 |
| [^44] | [The Agent in the Middle: How AI-Mediated Access Reshapes Web Vulnerability Exploitation](https://arxiv.org/abs/2610.35541) | 提出可执行基准测试VELA，系统研究了AI智能体中介访问对Web漏洞利用的影响，发现面向智能体的接口设计决定了漏洞可利用性，参考设计下91%的漏洞仍然可被利用。 |
| [^45] | [JANUS: Synthesizing Cross-Layer Defenses Against Web Deserialization Attacks](https://arxiv.org/abs/2610.35532) | JANUS提出了一种红蓝对抗框架，通过黑盒攻击者迭代探测攻击变体与防御者进行反事实分析并细化策略，实现针对Web反序列化攻击的跨层防御自动化合成。 |
| [^46] | [Silent and Costly: Characterizing and Detecting Prompt-Caching Mis-Integration in LLM Agents](https://arxiv.org/abs/2610.34817) | 该论文首次对10个开源智能体系统中120个提示缓存相关的拉取请求进行实证研究，揭示了提示缓存错误集成往往静默失败且难以察觉的问题，并据此提出了检测工具CACHELINT。 |
| [^47] | [Honesty Testing of LLMs and AI Agents: A Vision for Software Engineering](https://arxiv.org/abs/2610.34239) | 这篇愿景论文提出将LLM和AI智能体的诚实性测试作为软件工程的新研究领域，把忠实报告确立为可规定、可测试、可维护的软件质量需求，并定义了“诚实性缺陷”概念及初步测试框架。 |
| [^48] | [Beyond Correctness: How Novice Software Engineering Students Exercise Oversight of Generative AI](https://arxiv.org/abs/2610.32118) | 软件工程新手学生对生成式AI的监督主要聚焦于输出的适用性问题（74.4%）而非正确性问题（16.9%），表明学生监督的核心是判断看似合理的AI输出是否真正适合实际任务。 |
| [^49] | [Knowing When to Stop: The Evidence-Action Gap in Coding Agents](https://arxiv.org/abs/2610.31895) | 该论文揭示了编码智能体的“证据-行动差距”现象——即智能体即使已收集到表明任务无法独立完成的充分证据仍过度坚持或虚报成功，并提出ImpasseBench基准通过在可解决任务中注入环境阻碍并标记证据充分步骤来量化这一差距。 |
| [^50] | [CuraLoc: Repository-Level Fault Localization via Planned Code Inspection](https://arxiv.org/abs/2610.31809) | CuraLoc提出在源码检索前一次性规划代码检查的缺陷定位方法：模型在单次规划中同时选定起始符号、检查关系并形成修改假设，再由确定性检索完成最终排序，从而减少反复的模型调用和token消耗。 |
| [^51] | [From Code Generator to Engineering Companion: A Plant-aware Agent for PLC Programming](https://arxiv.org/abs/2610.28160) | 该论文提出了一个工厂感知的对话式PLC编程智能体，通过工厂特定上下文摄取、以对话为先的解释与调试交互模式，以及从聊天到部署的分级工具访问阶梯，将PLC编程从一次性代码生成升级为持续的工程辅助体验。 |
| [^52] | [Vibenix: Agentic Software Packaging and Maintenance with Nix](https://arxiv.org/abs/2610.24712) | Vibenix是一个开源AI智能体，利用大型语言模型通过规则引导的构建-修复反馈循环自动生成并迭代优化Nix打包表达式，在NixBench数据集上以Claude Haiku 4.5实现了95.3%的可构建软件包成功率，远超传统单次生成方法。 |
| [^53] | [When Does the Dependency Graph Help Code-to-Architecture Mapping?](https://arxiv.org/abs/2610.19769) | 该论文将代码到架构的映射建模为依赖图上的半监督节点分类问题，并通过GCN、GAT、MLP与朴素贝叶斯的系统对比实验，揭示了依赖图结构信息在何种条件下才能真正提升架构映射的效果。 |
| [^54] | [What Does an Agentic Software Engineering Benchmark Measure? Profiling Task Demands and Agent Behaviour Beyond What Category Labels Reveal](https://arxiv.org/abs/2609.01271) | 本文提出Spread-Novelty-Centrality（SNC）三轴画像方法来刻画仓库级编码任务的真实需求，发现类别标签是任务需求的不可靠代理指标，且智能体行为轨迹能揭示人工标准答案无法反映的任务需求信息。 |
| [^55] | [QTUNER: A Feature-Guided Adaptive Framework for Quantum-Compiler Optimization Pass Tuning](https://arxiv.org/abs/2607.04586) | 本文提出QTUNER，一种特征引导的自适应框架，通过引入与优化特性直接相关的量子特征来对量子编译器的完整优化Pass空间进行自适应调优，以克服现有方法仅调优有限Pass子集且依赖静态特征的局限，从而提升量子电路的执行保真度。 |
| [^56] | [AI Writes Faster Than Humans Can Review: A Longitudinal Study of an Enterprise 2x Mandate](https://arxiv.org/abs/2607.01904) | 本文通过对802名开发者和196,212个PR的纵向案例研究发现，企业“2倍”AI生产力指令最终在第10个月才达成，且实现路径与公告预期不符——指令发布时PR吞吐量几乎不变，开发者开始使用AI时仅提升约12%，此后随经验积累持续增长，表明AI编程收益需要时间逐步释放。 |
| [^57] | [Structuring license permissiveness from pairwise comparisons](https://arxiv.org/abs/2606.31032) | 本文利用大型语言模型，通过成对比较方法系统化地构建和验证许可证宽松度的部分排序，解决了大规模许可证比较的难题。 |
| [^58] | [Bit-Precise Conformance Testing of Simulink Model Checkers](https://arxiv.org/abs/2606.24719) | 本文通过对Simulink基本模块类型进行位级精确的形式化，并提出一种一致性测试方法，来验证Simulink模型检测器与仿真器之间行为的一致性和可靠性。 |
| [^59] | [LLM Code Smells: A Taxonomy and Detection Approach](https://arxiv.org/abs/2605.22976) | 本文提出了LLM代码坏味道的独立分类体系和九种坏味道目录，开发了静态分析工具SpecDetect4LLM进行检测，并通过对692个开源项目的实证评估发现73.5%的项目存在LLM代码坏味道。 |
| [^60] | [SemLoc: Structured Grounding of Free-Form LLM Reasoning for Fault Localization](https://arxiv.org/abs/2603.29109) | SemLoc将LLM的自由形式推理转化为附加在代码图锚点上的可执行、可验证的语义约束，通过运行时检查和反事实验证实现软件故障定位。 |
| [^61] | [Measuring and Exploiting Contextual Bias in LLM-Assisted Security Code Review](https://arxiv.org/abs/2603.18740) | 本研究揭示了框架效应会导致基于LLM的自动化代码审查系统在漏洞检测中产生系统性且普遍的偏差，并证明攻击者可以通过注入带有偏差的PR元数据，将其利用为针对ACR流水线的供应链攻击向量。 |
| [^62] | [SWE-Skills-Bench: Do Agent Skills Actually Help in Real-World Software Engineering?](https://arxiv.org/abs/2603.15401) | 该论文提出了SWE-Skills-Bench，一个以技能为中心、需求驱动的基准测试，通过3,920组有/无技能的配对评估，系统量化了单个智能体技能对真实世界软件工程任务的实际提升效果。 |
| [^63] | [Agentic Refactoring: An Empirical Study of AI Coding Agents](https://arxiv.org/abs/2511.04824) | 本文基于AIDev数据集对真实开源Java项目中AI编程智能体生成的重构进行了大规模实证研究，分析了51,412个重构实例，首次系统揭示了智能体重构在实际开发中的使用方式、与人工重构的差异及其对代码质量的影响。 |
| [^64] | [Optimizing Spectral Signature in Code Backdoor Detection](https://arxiv.org/abs/2510.13992) | 本文重新审视并系统评估了谱签名防御方法在检测代码模型后门攻击中的有效性，探索其在不同攻击场景和防御配置下的优化应用。 |
| [^65] | [From Legacy Fortran to Portable Kokkos: An Autonomous Agentic AI Workflow](https://arxiv.org/abs/2509.12443) | 本文提出一种自主智能体AI工作流，通过多个专门的LLM智能体协作，自动将遗留Fortran内核翻译、验证、测试、调试并优化为具有跨硬件性能可移植性的Kokkos C++代码。 |

# 详细

[^1]: 一种用于环境代币原子性、语义保持跨链传输的 EVM-Cosmos 跨链桥

    An EVM-Cosmos Bridge for Atomic, Semantic-Preserving Transfer of Environmental Tokens

    [https://arxiv.org/abs/2610.37663](https://arxiv.org/abs/2610.37663)

    该论文提出了一种连接 EVM 与 Cosmos 的跨链桥，利用密码学承诺派生的确定性指标标识符在铸币时校验以防止环境代币（如碳信用）的语义混合，并通过扩展 HTLC 加入自动回调销毁与激励相容赏金机制，实现单一所有者跨链转移的原子性。

    

    在异构区块链之间（例如 EVM 和 Cosmos 生态系统）传输具有语义类型的代币会引发两个问题：一是语义混合，即代币的领域特定含义在传输过程中丢失（例如在不同方法学下签发的 CO2 移除量）；二是在共识机制和时间假设不一致的情况下保持原子性。我们提出了一个同时解决这两个问题的跨链桥：通过从密码学承诺中派生的确定性指标标识符，在两条链上的类别注册表中镜像存储并在铸币时进行校验，从构造上防止语义混合。在原子性方面，我们将哈希时间锁定合约（HTLC）从两方交换扩展到单一所有者转移场景，加入了在目标链认领后自动回调销毁源链托管资金的机制，并通过协议最低赏金使回退销毁机制即使面对恶意发送方也能保持激励相容。我们使用 TLC 对协议的不变量进行了机器验证（若缺少该机制，供应量守恒将被破坏……）

    arXiv:2610.37663v1 Announce Type: new  Abstract: Transferring semantically typed tokens between heterogeneous blockchains (e.g. EVM and Cosmos ecosystems) raises two problems: semantic mixing, the loss of a token's domainspecific meaning in transit (e.g., CO2 removals issued under different methodologies), and atomicity despite divergent consensus and timing assumptions. We propose a bridge that addresses both. A deterministic indicator identity derived from cryptographic commitments, mirrored in class registries on both chains and checked at mint time, preventing semantic mixing by construction. For atomicity, we extend hash time-locked contracts (HTLCs) from two-party exchange to single-owner transfer with an automatic callback that burns the source escrow after a destination claim, and a protocol-minimum bounty that makes the fallback burn incentive-compatible even against adversarial senders. We machine-check the protocol's invariants with TLC (supply conservation fails without the
    
[^2]: FeatureArc：在演化特性上对编码智能体进行基准测试

    FeatureArc: Benchmarking Coding Agents on Evolving Features

    [https://arxiv.org/abs/2610.37434](https://arxiv.org/abs/2610.37434)

    提出 FeatureArc 基准测试，首次评估编码智能体在软件特性持续演化场景下的能力，其流水线从仓库历史构建包含阶段间依赖、明确需求契约、可执行 F2P/P2P 测试验证并覆盖四种编程语言的任务。

    

    一个软件特性通常会随时间演化：随着新用例和需求的到来，开发人员会在第一版的基础上进行扩展，而每次扩展都建立在之前编写的代码之上。大多数特性基准测试是在固定代码库上评估单个请求。特性演化则要求在智能体自己先前编写的代码上评估连续的扩展，并检查该特性是否保持正确。我们提出了 FeatureArc，用于评估编码智能体在演化特性上的表现，并配备了一个从仓库历史构建任务的流水线。该基准测试结合了同一特性各连续阶段之间的依赖关系、明确的需求与接口契约，以及覆盖四种编程语言的可执行检查。该流水线在参考代码的每个阶段前后，验证针对新行为的“失败到通过”（F2P）测试和针对既有行为的“通过到通过”（P2P）测试。我们在 73 个经人工审核的任务上评估了 10 个编码智能体。

    arXiv:2610.37434v1 Announce Type: cross  Abstract: A software feature usually evolves over time: developers extend its first version as new use cases and requirements arrive, and each extension builds on code written earlier. Most feature benchmarks evaluate a single request on a fixed codebase. Feature evolution requires evaluating successive extensions on an agent's own earlier code and checking whether the feature remains correct. We introduce FeatureArc to evaluate coding agents on evolving features, together with a pipeline that constructs its tasks from repository histories. The benchmark combines dependencies between successive stages of the same feature, explicit requirements and interface contracts, and executable checks across four programming languages. The pipeline validates fail-to-pass (F2P) tests for new behavior and pass-to-pass (P2P) tests for existing behavior on the reference code before and after each stage. We evaluated 10 coding agents on 73 manually reviewed task
    
[^3]: HIDE：基于文档与证据的 Rustc 缺陷层次化隔离方法

    HIDE: Hierarchical Isolation via Documentation and Evidence for Rustc Bugs

    [https://arxiv.org/abs/2610.37343](https://arxiv.org/abs/2610.37343)

    HIDE 是一种基于大语言模型的层次化方法，通过将编译器文档与缺陷证据相结合来对可疑的 rustc 源文件进行排序，从而有效缩小搜索空间并提升 rustc 缺陷隔离的准确性。

    

    Rustc 缺陷可能导致合法程序无法编译或产生错误的编译结果，因此识别出导致问题的源文件对于修复工作至关重要。使用大语言模型（LLM）进行 rustc 缺陷隔离面临三个挑战。首先，编译器功能具有分布式特性，需要同时理解各个文件的功能以及它们之间的协作方式。其次，庞大的源文件搜索空间使得对所有文件进行穷举分析的成本高昂。第三，在 LLM 提示中包含大量候选文件的文档和源代码会导致上下文过载，从而降低缺陷隔离的准确性。我们提出了 HIDE，一种基于 LLM 的层次化方法，它将编译器文档与缺陷证据相结合，对可疑的 rustc 源文件进行排序。给定一个 rustc 版本和缺陷上下文，HIDE 由两个阶段组成，分别用于生成编译器文档和缩小搜索空间。为了理解分布式的编译器功能……（原文摘要在此处截断）

    arXiv:2610.37343v1 Announce Type: new  Abstract: Rustc bugs can prevent valid programs from compiling or cause incorrect compilation results, making the identification of responsible source files essential for repair. Using large language models (LLMs) for rustc bug isolation raises three challenges. First, distributed compiler functionality requires understanding both what individual files do and how they work together. Second, a large source file search space makes exhaustive analysis of all files costly. Third, including documentation and source code for many candidate files in LLM prompts leads to context overload, which can reduce bug isolation accuracy. We propose HIDE, a hierarchical LLM-based approach that combines compiler documentation with bug evidence to rank suspicious rustc source files. Given a rustc revision and a bug context, HIDE consists of two stages that generate compiler documentation and narrow the search space. To understand distributed compiler functionality, t
    
[^4]: 注意意图差距：一种协作式智能体驱动的软件工程方法论

    Mind the Intent Gap: A Methodology for Collaborative Agent-Driven Software Engineering

    [https://arxiv.org/abs/2610.37340](https://arxiv.org/abs/2610.37340)

    提出CAIA（协作式智能体意图对齐）方法论，通过知识库显式记录团队意图，弥合AI智能体加入软件团队后因“意图未被完整记录”而靠猜测行事并偏离团队目标的“意图差距”问题。

    

    AI智能体已进入软件团队，最初作为工具，如今则作为团队成员。它们能在几分钟内完成过去需要数天的工作，然而预期的交付收益却往往未能实现。我们认为，这是因为它们进入的是为人设计的流程，而这些流程一直悄然依赖于人来弥补自身的三个弱点。团队的意图从未被完整地记录下来，因为同事之间会自然填补这些空白；AI智能体同样会填补这些空白，但只能靠猜测，从而偏离了团队原本的想法。每个交付物背后的推理过程在流程步骤之间早已丢失，而AI智能体现在会放大或压缩传递的文档。人们独自完成的推理有一部分是私密性的，如今这些推理转移到了天然将其保持私密的AI工具之中。我们提出了CAIA（Collaborative Agentic Intent Alignment，协作式智能体意图对齐），这是一种将智能体融入软件团队工作方式的方法论。该方法通过一个知识库来描述项目所维护的产品。

    arXiv:2610.37340v1 Announce Type: new  Abstract: AI agents have entered software teams, first as tools and now as team members. They produce in minutes what used to take days, yet the expected gain in delivery often fails to appear. We argue that they are entering processes designed for people, which quietly relied on people to cover three of their weaknesses. The team's intent was never fully written down, because colleagues filled the gaps. An AI agent fills them too, though by guessing, and drifts from what the team had in mind. The reasoning behind each deliverable was already lost from one step to the next, and AI agents now inflate and deflate the documents passed along. The reasoning people did alone was partly private and now moves into AI tools that keep it private by nature. We present CAIA, for Collaborative Agentic Intent Alignment, a methodology for software teams that brings agents into the way they work. A knowledge base describes the product that a project maintains. Fo
    
[^5]: 《藏于显眼之处：关于编程智能体指令栈的实证研究》

    Hidden in Plain Sight: an Empirical Study about The Instruction Stacks of Coding Agents

    [https://arxiv.org/abs/2610.37332](https://arxiv.org/abs/2610.37332)

    该论文通过对五个主流编程智能体共 1,886 个版本的大规模纵向实证研究，首次系统揭示了隐藏在模型请求中的“指令栈”的构成与演变，表明仅看系统提示词或模型名称无法完整刻画智能体的真实输入。

    

    编程智能体发送给模型的远不止用户的提示词：它们还会附加指令、工具定义和环境上下文，这些内容共同构成一个“指令栈”。该指令栈指导智能体的执行过程，并随版本发布不断变化，然而它对用户而言几乎不可见，在研究中也缺乏充分的刻画。我们对 Claude Code、Codex、Gemini CLI、Qwen Code 和 OpenCode 五个智能体共 1,886 个版本中首次模型请求所捕获的指令栈，开展了纵向、跨智能体的研究。在受控条件下，我们分析了这些指令栈的组成与演变，并将四个开源智能体的变化追溯至有据可查的动机。我们的研究结果表明，仅检查系统提示词或只报告所用模型，无法完整描述智能体的实际输入。工具定义在三个智能体的最新请求中占据主导地位，而偶发的集中式修订则会改变控制工具使用、署名和代码风格的指令。此类修订可能…

    arXiv:2610.37332v1 Announce Type: new  Abstract: Coding agents send their models more than the user's prompt: they add instructions, tool definitions, and environmental context that form an instruction stack. This stack guides execution and changes across releases, yet remains largely unexposed to users and insufficiently characterized in research. We present a longitudinal, cross-agent study of the instruction stacks captured in the first model requests of 1,886 releases of Claude Code, Codex, Gemini CLI, Qwen Code, and OpenCode. Under controlled conditions, we analyze their composition and evolution and trace changes in the four open-source agents to documented motivations. Our findings show why examining the system prompt or reporting the model alone provides an incomplete account of agent inputs. Tool definitions dominate three agents' latest requests, while occasional concentrated revisions alter instructions governing tool use, attribution, and coding style. Such revisions can co
    
[^6]: BenchMeUp：从自然语言规范构建软件工程基准测试

    BenchMeUp: Building Software Engineering Benchmarks from Natural-Language Specifications

    [https://arxiv.org/abs/2610.37283](https://arxiv.org/abs/2610.37283)

    本文提出BenchMeUp方法，可从自然语言规范自动构建软件工程基准测试，通过并行LLM智能体识别并构建可执行实例、在新环境中验证正确性，从而大幅降低基准测试构建的人工成本。

    

    软件工程研究依赖于在一系列问题实例上评估新技术的基准测试。遗憾的是，手动构建基准测试非常耗费人力，因为研究人员必须寻找合适的实例、准备执行环境、建立预期结果并验证正确性。本文提出了BenchMeUp，一种从自然语言规范自动创建软件工程基准测试的方法。给定一个定义了所需实例、数据源和评估流程的规范，BenchMeUp会生成搜索规则和构建计划，调用并行的大语言模型（LLM）智能体来识别候选实例并构建可执行的实例，在全新的执行环境中验证结果，并通过统一的命令行接口封装这些实例。我们在四项任务上评估了该方法：程序修复、问题复现、编译器自动调优以及修复静态分析（相关任务）。

    arXiv:2610.37283v1 Announce Type: new  Abstract: Software-engineering research relies on benchmarks that evaluate new techniques on a set of problem instances. Unfortunately, manually constructing benchmarks is labor-intensive, as researchers must find suitable instances, prepare execution environments, establish expected outcomes, and validate correctness.   This paper presents BenchMeUp, an approach for automatically creating software-engineering benchmarks from natural-language specifications. Given a specification defining the desired instances, data sources, and evaluation procedures, BenchMeUp generates search rules and a construction plan, invokes parallel LLM agents that identify candidate instances and construct executable instances, validates the results in fresh execution environments, and packages the instances behind a unified command-line interface. We evaluate the approach on four tasks: program repair, issue reproduction, compiler autotuning, and fixing static-analysis 
    
[^7]: 将大语言模型智能体锚定于业务流程：面向可靠多步骤服务调用的流程感知知识图谱

    Grounding LLM Agents in Business Processes: Process-Aware Knowledge Graphs for Reliable Multi-Step Service Invocation

    [https://arxiv.org/abs/2610.37242](https://arxiv.org/abs/2610.37242)

    该论文提出将OData元数据与BPMN参考流程整合为类型化属性图知识图谱，并利用沿流程边检索的混合检索器，显著提升了LLM智能体在多步骤企业服务调用中的准确性和可靠性。

    

    调用企业API的大语言模型（LLM）智能体必须选择正确的端点、遵循流程顺序、使用有效的载荷键，并从预定义列表中选择编码值。然而，当前的检索增强式服务发现方法将端点独立处理，因此忽略了流程和模式约束。我们通过将开放数据协议元数据与业务流程模型和标记法（BPMN）参考流程整合到类型化属性图中来解决这一局限。混合检索器使用稠密嵌入识别候选端点，并沿着流程边检索后续端点。我们在42个企业服务上结合流程链真值对该方法进行评估。对于查询与下游实体之间词汇重叠度较低的查询，该方法将Recall@3提升了5.7个百分点（n=74），且在不同的嵌入模型和截断阈值下均保持一致的增益。在其他查询上，其表现与稠密检索相当，并提供了……

    arXiv:2610.37242v1 Announce Type: new  Abstract: Large language model (LLM) agents that call enterprise APIs must select the correct endpoints, follow process order, use valid payload keys, and choose coded values from predefined lists. However, current retrieval-augmented service discovery treats endpoints independently and therefore misses process and schema constraints. We address this limitation by integrating Open Data Protocol (OData) metadata and Business Process Model and Notation (BPMN) reference processes in a typed property graph. A hybrid retriever identifies candidate endpoints using dense embeddings and follows process edges to retrieve subsequent endpoints. We evaluate the method on 42 enterprise services with process-chain ground truth. For queries with low lexical overlap between the query and downstream entities, it improves Recall@3 by 5.7 points (n=74), with consistent gains across embedding models and cut-offs. It matches dense retrieval on other queries and provid
    
[^8]: Spine：将子任务树编译为结构化上下文以支持长程编码智能体

    Spine: Compiling Subtask Trees into Structured Context for Long-Horizon Coding Agents

    [https://arxiv.org/abs/2610.37224](https://arxiv.org/abs/2610.37224)

    Spine 通过让智能体用“工作范围”显式声明子任务边界，将子任务树编译为持久化的结构化上下文，在不依赖子智能体委派的情况下对齐长程编码任务的嵌套结构与模型上下文，避免上下文膨胀。

    

    长程编码任务包含嵌套的子任务。通常情况下，智能体将交互按顺序记录到不断增长的历史中，导致上下文膨胀和性能下降。在缺乏显式子任务边界的情况下压缩或裁剪历史，可能会丢弃未完成子任务所需的信息。相反，递归的子智能体委派方式通过子智能体来隔离已完成的子任务，将每一个子任务边界都交由委派处理，这混淆了子任务粒度与执行智能体的选择，增加了执行与协调开销。我们提出了 Spine，这是一个将模型上下文与子任务嵌套及完成情况对齐的接口与运行时。智能体通过“工作范围”声明显式的子任务边界，使同一个智能体能够内联执行嵌套子任务，同时将委派保留给独立的并行任务。该运行时以 SpineTree 来组织执行过程——这是一棵持久化的树，记录子任务之间的关系、局部……

    arXiv:2610.37224v1 Announce Type: cross  Abstract: Long-horizon coding tasks contain nested subtasks. Typically, agents record interactions sequentially into a growing history, causing context inflation and performance degradation. Compressing or trimming the history without explicit subtask boundaries can discard information needed by unfinished subtasks. Conversely, recursive subagent delegation isolates completed subtasks through child agents by assigning every subtask boundary to delegation, which conflates subtask granularity with executing agent choice, adding execution and coordination overhead. We introduce Spine, an interface and runtime that align model context with subtask nesting and completion. An agent declares explicit subtask boundaries through work scopes, allowing the same agent to execute nested subtasks inline while reserving delegation for independent parallel tasks. The runtime organizes execution in SpineTree, a persistent tree recording subtask relations, local 
    
[^9]: 面向量子程序的电路感知变异测试：精确可见性与执行规划

    Toward Circuit-Aware Mutation Testing for Quantum Programs: Exact Visibility and Execution Planning

    [https://arxiv.org/abs/2610.36976](https://arxiv.org/abs/2610.36976)

    提出了一种面向量子程序的电路感知变异测试框架，能够精确区分变异体存活的三种原因（等价变异、被输入或测量隐藏的变更、统计证据不足），并给出测试无法揭示变异的条件与变异影响的精确公式，从而指导执行规划和测试改进。

    

    变异测试通过向程序引入变更并检查测试是否能检测到这些变更来评估测试套件的质量。在量子程序中，一个存活的变异体可能源于以下几种情况：变异没有改变程序行为、产生的变更被所选的输入或测量所隐藏、或者由于统计证据不足而未被检测到。我们开发了一个电路感知框架来区分这些不同的原因，并据此指导执行规划和测试变更。该框架涵盖理想的、无噪声的程序，以及替换单个门或子电路的变异。该方法追踪变异如何影响测量输出，并将已有的数学检查与统计估计相结合。对于所支持的案例，我们建立了测试无法揭示变异的条件，以及在所需量子态信息可用时描述变异影响的精确公式。在给定假设下，我们在保留存活变异体和未解决变异体的同时对变异得分进行界定，并识别……

    arXiv:2610.36976v1 Announce Type: new  Abstract: Mutation testing assesses test suites by introducing program changes and checking whether tests detect them. In quantum programs, a surviving mutant may leave program behavior unchanged, produce a change hidden by the chosen input or measurement, or remain undetected because statistical evidence is insufficient. We develop a circuit-aware framework to distinguish these causes and guide execution planning and test changes. The framework covers ideal, noiseless programs and mutations that replace one gate or subcircuit. The method traces how a mutation can affect measured outputs and combines established mathematical checks with statistical estimates. For supported cases, we establish conditions under which a test cannot reveal a mutation and exact formulas for its effect when the required state information is available. Under stated assumptions, we bound the mutation score while retaining surviving and unresolved mutants, and identify a r
    
[^10]: 更多思考是否值得？大语言模型在代码生成与漏洞修复中的推理成本与正确性

    Does More Thinking Pay Off? Reasoning Cost and Correctness of LLMs in Code Generation and Bug Fixing

    [https://arxiv.org/abs/2610.36952](https://arxiv.org/abs/2610.36952)

    本文通过对 866 个 Python 任务的实证研究发现，开启显式推理能显著提升大语言模型在部分代码生成与漏洞修复任务上的正确性，但更长、更多的思考并不必然带来更高的准确率，推理收益与令牌开销之间不成正比。

    

    如今许多大语言模型在编写代码之前会输出显式的推理轨迹，但代码生成和漏洞修复的基准测试结果通常只报告正确性。因此，额外的推理相对于其消耗的令牌（token）究竟能带来多少收益仍不清楚。我们使用来自 HumanEval Pro、MBPP Pro 和 BugsInPy 的 866 个 Python 任务，在五种模型设置下考察了这一问题。我们分析了推理带来的准确率提升与令牌开销、推理轨迹中出现的行为，以及被缩短的推理轨迹是否仍能得到正确的解决方案。在 Qwen3-32B 上开启推理并运行五次，HumanEval Pro 的平均 pass@1 从 71.1% 提升至 76.5%，BugsInPy 的平均测试通过率从 3.5% 提升至 7.9%。而在 MBPP Pro 上，准确率从 68.1% 提升到 69.5%，尽管模型每个任务中位数消耗超过 2000 个思考令牌，该提升在统计上并不显著。跨模型来看，中位数更长的推理轨迹并不一定可靠地意味着更高的准确率。

    arXiv:2610.36952v1 Announce Type: new  Abstract: Many LLMs now emit an explicit reasoning trace before they write code, but benchmark results for code generation and bug fixing usually report correctness alone. It is therefore unclear how much the extra reasoning contributes relative to the tokens it consumes. We examine this question with 866 Python tasks drawn from HumanEval Pro, MBPP Pro, and BugsInPy under five model settings. We look at the accuracy gain and token overhead of reasoning, the behaviors that appear inside reasoning traces, and whether a shortened trace still leads to a correct solution. Turning reasoning on in Qwen3-32B over five runs improves mean pass@1 on HumanEval Pro from 71.1% to 76.5% and mean tests pass on BugsInPy from 3.5% to 7.9%. On MBPP Pro the improvement from 68.1% to 69.5% is not statistically significant, even though the model uses a median of over 2,000 thinking tokens per task. Across models, a longer median trace does not reliably mean higher accu
    
[^11]: 超越QASecClaw：软件保障中的智能体扩展、验证器反馈与证据可靠性

    Beyond QASecClaw: Agent Scaling, Verifier Feedback, and Evidence Reliability in Software Assurance

    [https://arxiv.org/abs/2610.36947](https://arxiv.org/abs/2610.36947)

    该研究通过五组实验表明，在软件保障任务中扩大语言模型智能体上下文规模会使token开销激增49倍却无安全收益，验证器反馈虽能提升正确判定数量但同时增加错误判定，说明确定性的工具证据基线仍然不可或缺。

    

    arXiv:2610.36947v1 公告类型：cross 摘要：当确定性工具证据已经可用时，额外的专家级语言模型上下文能否改善软件保障？我们通过五个实验来研究这一问题，涵盖人工编写的Java警告控制、OWASP基准评估、Defects4J测试生成、SV-COMP断言判定，以及刻意误导性的证据包。所评估的系统使用Qwen2.5-Coder-7B-Instruct，并进行1.5B参数的敏感性分析，共完成13,638次模型请求。在八个相关的开发警告上，将上下文数量从1增加到50会使token使用量增加49倍，而未能改善不安全代码的放行情况。在160个OWASP案例中，各模型配置在放行与未解决结果之间进行权衡；十上下文配置仅评估了96个案例。在32个SV-COMP任务上，原始验证器反馈将模型判定的平均正确数从12.7提高到22.0，但也将错误判定从1个增加到7个；JBMC加JVM验证基线返回26个……（摘要在此处截断）

    arXiv:2610.36947v1 Announce Type: cross  Abstract: Can additional specialist language-model contexts improve software assurance when deterministic tool evidence is already available? We study this question through five experiments spanning authored Java warning controls, OWASP Benchmark assessment, Defects4J test generation, SV-COMP assertion verdicts, and deliberately misleading evidence packages. The evaluated systems use Qwen2.5-Coder-7B-Instruct, with a 1.5B sensitivity analysis, and complete 13,638 model requests. On eight related development warnings, increasing the context count from 1 to 50 increases token use 49-fold without improving unsafe clearance. On 160 OWASP cases, model configurations trade clearance against unresolved outcomes; the ten-context arm assesses only 96 cases. On 32 SV-COMP tasks, raw verifier feedback raises mean correct model verdicts from 12.7 to 22.0, but also raises incorrect verdicts from one to seven; a JBMC-plus-JVM validation baseline returns 26 co
    
[^12]: 面向异构TinyML系统的轻量级C代码生成器

    A lightweight C Code Generator for Heterogeneous TinyML Systems

    [https://arxiv.org/abs/2610.36907](https://arxiv.org/abs/2610.36907)

    该论文提出了一种面向集成领域专用加速器的裸机异构TinyML系统的轻量级C代码生成器，通过权重调优和偏置融合等提前优化技术，在基于RISC-V的雷达DSP上相比microTVM实现了2.3倍推理加速和20%的二进制体积缩减。

    

    在过去十年中，人工智能的重大进展与日益复杂的训练框架和不断强大的计算硬件密切相关。然而，整个生态系统仍然处于碎片化状态。一方面，模型以依赖于特定框架的格式表示；另一方面，用于加速新型算子的定制化硬件不断涌现。为了有效降低从训练到推理的迁移成本，我们提出了一种轻量级C代码生成器，专为集成领域专用加速器的裸机异构系统而设计。通过在基于RISC-V的雷达DSP上进行的全面案例研究，我们证明了通过采用带有权重调优和偏置融合等提前优化技术的加速器，与microTVM流程相比，我们的方法实现了2.3倍的推理速度提升和20%的二进制文件体积缩减。

    arXiv:2610.36907v1 Announce Type: new  Abstract: Over the past decade, the significant advancements in artificial intelligence have been closely tied to increasingly sophisticated training frameworks and steadily powerful computing hardware. However, the entire ecosystem remains fragmented. On one hand, models are represented with frameworks dependent formats, while on the other hand, new tailored hardware for accelerating novel operators has continued to emerge. To effectively diminish the transfer costs from training to inference, we propose a lightweight C code generator designed for bare-metal heterogeneous systems that incorporate domain-specific accelerators. Through a comprehensive case study on a RISC-V based radar DSP, we demonstrate that by employing an accelerator with ahead-of-time optimizations like weight tuning and bias fusion, our approach achieves a 2.3 times improvement in inference speed and a 20% reduction in binary size compared to the microTVM flow registered with
    
[^13]: ASNER：基于自适应语义邻域与证据支撑细化的软件架构恢复

    ASNER: Software Architecture Recovery with Adaptive Semantic Neighborhoods and Evidence-Grounded Refinement

    [https://arxiv.org/abs/2610.36894](https://arxiv.org/abs/2610.36894)

    ASNER通过职责感知、项目自适应的关系建模构建自适应语义邻域，并采用证据约束的LLM边界验证机制，解决了固定阈值忽略仓库规模、功能相关性误导架构职责判断以及无约束LLM细化破坏正确结果的问题，从而提升软件架构恢复的准确性。

    

    自动化软件架构恢复为软件工程和仓库级LLM智能体提供架构上下文。现有方法结合了结构和语义信号，但功能上的相关性并不一定意味着共享架构职责。固定阈值或Top-K邻域的方法也忽略了仓库规模和局部相似度分布，从而引入会在聚类过程中传播的错误。而不受约束的LLM细化可能会在没有充分证据的情况下改变原本正确的分配。我们提出了ASNER（基于自适应语义邻域与证据支撑细化的软件架构恢复方法），其基础是职责感知的、项目自适应的关系建模以及证据支撑的边界验证。ASNER利用LLM推导出结构化的文件职责，构建能够适应仓库规模和局部相似度分布的语义邻域，并对静态依赖……（原文摘要在此处截断）

    arXiv:2610.36894v1 Announce Type: new  Abstract: Automated software architecture recovery provides architectural context for software engineering and repository-level LLM agents. Existing methods combine structural and semantic signals, but functional relatedness does not necessarily imply shared architectural responsibility. Fixed thresholds or Top-K neighborhoods also ignore repository scale and local similarity distributions, introducing errors that propagate through clustering. Unconstrained LLM refinement may further alter correct assignments without sufficient evidence. We propose ASNER (Adaptive Semantic Neighborhoods and Evidence-Grounded Refinement for Software Architecture Recovery), based on responsibility-aware, project-adaptive relation modeling and evidence-grounded boundary verification. ASNER derives structured file responsibilities with an LLM, constructs semantic neighborhoods that adapt to repository scale and local similarity distributions, and augments a static dep
    
[^14]: 上帝不掷骰子：在智能体式内核生成中强制执行批处理不变性

    God Does Not Play Dice: Enforcing Batch Invariance in Agentic Kernel Generation

    [https://arxiv.org/abs/2610.36623](https://arxiv.org/abs/2610.36623)

    该论文提出了一种即插即用工具，通过将契约引导的静态依赖分析与跨批处理上下文的按位一致性动态验证相结合，作为验收门控嵌入内核生成智能体的优化循环中，从而强制执行批处理不变性，解决LLM服务中动态批处理导致相同输入产生不同数值输出、损害可复现性的问题。

    

    大语言模型（LLM）服务系统会动态地将用户请求分批以进行GPU执行，因此同一请求在不同运行中可能会与不同的请求一起被处理。如果GPU内核缺乏批处理不变性，这种不断变化的批处理上下文可能会改变相同输入的数值输出，从而损害可复现性。现有的内核生成智能体专注于正确性和效率，却忽视了批处理不变性。我们在83个适用于批处理的已发布产物中发现了19个经确认的批处理不变性违规。为解决这一局限，我们提出了一个工具，它将基于契约的静态依赖分析与跨批处理上下文的按位一致性动态验证相结合。该工具被设计为通用内核智能体的即插即用组件，能够在复用其原有优化能力的同时增加批处理不变性强制执行。我们将其作为优化循环中的连续验收门控进行集成，拒绝结构上有风险或动态验证不通过的内核（摘要原文在此处截断）。

    arXiv:2610.36623v1 Announce Type: new  Abstract: Large language model (LLM) serving systems dynamically batch user requests for GPU execution, so the same request may be processed alongside different requests across runs. If the GPU kernels lack batch invariance (BI), this changing batch context can alter the numerical output for an unchanged input, undermining reproducibility. Existing kernel generation agents focus on correctness and efficiency but \emph{overlook BI}. We find 19 confirmed BI violations among 83 batch-applicable released artifacts. To address this limitation, we propose \tool, which combines contract-guided static dependency analysis with dynamic validation of bitwise consistency across batch contexts. \tool is designed as a plug-in component for general kernel agents, allowing their original optimization capabilities to be reused while adding BI enforcement. We integrate it as successive acceptance gates in the optimization loop, rejecting structurally risky or dynam
    
[^15]: Maven生态中上游安全事件与下游漏洞缓解的细粒度实证研究

    A Fine-Grained Empirical Study of Upstream Security Events and Downstream Mitigation in Maven

    [https://arxiv.org/abs/2610.36603](https://arxiv.org/abs/2610.36603)

    该研究首次对Maven生态系统中832个CVE进行从上游修复到下游采用的端到端细粒度追踪，揭示了下游安全暴露究竟源于上游发布延迟、下游采用延迟还是两者兼有的问题。

    

    开源漏洞缓解涉及上游库以及依赖这些库的下游项目。在上游，漏洞会经历如下过程：发布公开报告（如果存在）、进行修复、发布修复版本以及NVD披露。在下游，项目通过采用修复版本来缓解漏洞，而这种采用的及时性是相对于上述事件来衡量的。以往的研究将这两方面分开研究，因此尚不清楚下游的安全暴露究竟源于上游发布延迟、下游采用延迟，还是两者兼有。我们通过追踪Maven生态系统中832个CVE从上游修复到每个受影响下游项目采用的全过程来填补这一研究空白。我们将上游修复分类为“静默式”或“透明式”，还原了报告、修复、版本发布和NVD披露的先后顺序，并将存在漏洞的构件与其下游依赖者相关联。随后，我们将这一时间顺序与下游的采用情况以及dependabot的使用情况进行对比分析。

    arXiv:2610.36603v1 Announce Type: new  Abstract: Open-source vulnerability mitigation spans upstream libraries and the downstream projects that depend on them. Upstream, a vulnerability progresses through a public report when one exists, a fix, a fixed release, and NVD disclosure. Downstream, a project mitigates it by adopting the fixed release, and the timeliness of this adoption is measured relative to these events. Prior work studies the two sides separately, so it is unclear whether downstream exposure stems from delayed upstream release, delayed downstream adoption, or both. We address this gap by following 832 CVEs in the Maven ecosystem from the upstream fix to its adoption by each exposed downstream project. We classify upstream fixes as \textit{silent} or \textit{transparent}, recover the order of report, fix, release, and NVD disclosure, and link vulnerable artifacts to their downstream dependents. We then compare this order with downstream adoption and dependabot usage. We f
    
[^16]: 面向基于软件需求的行为充分性测试生成的LLM智能体基准评测

    Benchmarking LLM Agents for Behaviorally Adequate Test Generation from Software Requirements

    [https://arxiv.org/abs/2610.36584](https://arxiv.org/abs/2610.36584)

    该论文提出了Req2Test-Bench基准，包含从8个开源项目BDD工件中恢复的5,086个经执行验证的实例，用于评测LLM智能体在仅依据自然语言需求（无实现代码）的情况下生成行为充分测试的能力。

    

    随着大型语言模型（LLM）智能体日益将自然语言需求转化为仓库级别的代码，“从需求生成测试”变得愈发重要，因为它能够为代码生成提供验证约束（即测试脚手架/harness）。然而，目前尚不清楚现有的智能体能否在对应实现代码被隐藏的情况下，生成能够验证所需行为的测试。现有的测试生成基准大多聚焦于白盒测试，而需求—测试相关的研究通常需要对给定文档与所生成测试之间的语义对齐程度进行评估。我们提出了Req2Test-Bench，这是一个包含5,086个经执行验证实例的基准数据集，这些实例从8个开源项目中开发者编写的行为驱动开发（BDD）工件中恢复而来，涵盖5种编程语言和4个BDD框架。每个实例将Gherkin格式的自然语言需求与开发者编写的测试（配对）……（摘要原文在此处截断）

    arXiv:2610.36584v1 Announce Type: cross  Abstract: As large language model agents increasingly compile natural-language requirements into repository-level code, requirement-to-test generation grows in importance to provide validation constraints (or harness) to guide code generation. Yet it remains unclear whether current agents can generate tests that validate the required behavior when the implementation of that behavior is withheld. Existing test-generation benchmarks mostly focus on white-box testing, whereas the requirement--test-related study usually requires evaluation on semantic alignment between the given documents and the generated tests.   We introduce Req2Test-Bench, a benchmark containing 5,086 execution-validated instances recovered from developer-authored behavior-driven development (BDD) artifacts in 8 open-source projects, spanning 5 programming languages and 4 BDD frameworks. Each instance pairs a natural-language requirement in Gherkin with its developer-written Tes
    
[^17]: CASEL：面向可靠长时程工具使用智能体的后果感知选择性执行

    CASEL: Consequence-Aware Selective Execution for Reliable Long-Horizon Tool-Using Agents

    [https://arxiv.org/abs/2610.36570](https://arxiv.org/abs/2610.36570)

    CASEL提出了一种与模型无关的运行时分配策略，通过综合失败概率、后果严重性、可逆性、模糊性和证据新鲜度等因素，为智能体的每次工具调用智能选择直接执行、验证、沙箱、澄清、审批或拒绝执行，显著减少了长时程工具使用智能体造成有害副作用的风险。

    

    使用工具的语言智能体越来越多地修改外部系统，然而大多数运行时安全防护机制仅根据通用的不确定性评分或粗略的变更启发式方法来分配审查力度。我们提出了CASEL（后果感知选择性执行层），这是一种与模型无关的运行时分配策略，它综合了校准后的失败概率、状态相关的后果严重性、可逆性、模糊性和证据新鲜度，通过最小化每种模式的成本，为每个提议的工具调用在直接执行、验证、沙箱隔离、澄清、人工审批和拒绝执行之间做出选择。我们将直接执行形式化为选择性预测，并陈述和检验了有限样本阈值与轨迹风险账本的有效性条件。在一个包含20个随机种子、每个种子12万个测试动作的受控仿真实验中（其中每种方法拥有相同的干预菜单和相同的计算成本），CASEL将有害副作用轨迹的比例从无防护基线的14.75%降低至……（原文截断）

    arXiv:2610.36570v1 Announce Type: new  Abstract: Tool-using language agents increasingly modify external systems, yet most runtime safeguards allocate scrutiny by generic uncertainty scores or coarse mutation heuristics. We introduce CASEL (Consequence-Aware Selective Execution Layer), a model-agnostic runtime allocation policy that combines calibrated failure probability, state-dependent consequence, reversibility, ambiguity, and evidence freshness to choose, for each proposed tool call, among direct execution, verification, sandboxing, clarification, approval, and abstention by minimizing a per-mode cost. We formalize direct execution as selective prediction and state and test the validity conditions of a finite-sample threshold and a trajectory risk ledger. In a controlled simulation over 20 seeds with 120,000 test actions each, in which every method has the same intervention menu and the same compute cost, CASEL reduces harmful-side-effect trajectories from 14.75 percent for an unc
    
[^18]: 从涟漪到边界：面向大语言模型软件演化的确定性影响边界

    From Ripple to Boundary: Deterministic Impact Boundaries for LLM-Based Software Evolution

    [https://arxiv.org/abs/2610.36549](https://arxiv.org/abs/2610.36549)

    该论文提出将多源设计资产形式化为知识图谱，实现从概率性语义检索到确定性结构约束的范式转变，为大语言模型驱动的软件演化提供确定性的影响边界，避免幻觉变更或破坏无关代码。

    

    arXiv:2610.36549v1 公告类型：new 摘要：在大语言模型时代，软件演化的格局正在经历一场根本性的范式转变。随着工业软件工程日益向设计驱动的代码生成转变，演化的瓶颈已从编写代码转移到维护多源设计资产与生成代码之间的确定性一致性。当设计资产发生变更时，目前的大语言模型智能体缺乏确定性影响边界来识别哪些代码必须演化，从而导致两种失败模式：幻觉出缺失的变更，或破坏无关的代码。概率性方法（如语义检索或基于文本的差异比对）无法捕捉高度互联的设计依赖中跨结构的涟漪效应。我们提出了一种从概率性语义检索到确定性结构约束的范式转变。我们将设计资产形式化为知识图谱，并将其组织为基于内容的……（原文摘要到此中断）

    arXiv:2610.36549v1 Announce Type: new  Abstract: The landscape of software evolution is undergoing a fundamental paradigm shift in the era of Large Language Models. As industrial software engineering increasingly transitions toward design-driven code generation, the bottleneck of evolution has shifted from writing code to maintaining deterministic consistency between multi-source design assets and the generated code. When a design asset changes, LLM agents currently lack a deterministic impact boundary to identify which code must be evolved, leading to two failure modes: hallucinating missing changes or breaking unrelated code. Probabilistic approaches, such as semantic retrieval or text-based diffing, fail to capture the structural ripple effects across highly interconnected design dependencies. We propose a paradigm shift from probabilistic semantic retrieval to deterministic structural constraints. We formalize the design assets as a Knowledge Graph and organize them into a content-
    
[^19]: ProCerDCL：面向深度学习编译器优化的查询高效黑盒概率验证

    ProCerDCL: Query-Efficient Black-Box Probabilistic Verification of Deep Learning Compiler Optimization

    [https://arxiv.org/abs/2610.36499](https://arxiv.org/abs/2610.36499)

    ProCerDCL 提出了一种查询高效的黑盒概率验证框架，利用单一序贯检验在指定错误预算内判定属性违反概率是否超过阈值，并将其应用于深度学习编译器优化的验证。

    

    我们提出了 ProCerDCL，一个查询高效的黑盒概率验证框架。给定一个固定的、可采样的输入分布和一个二元属性判定器，ProCerDCL 针对阈值 θ 评估该属性的违反概率 p。在独立采样且判定器无状态的假设下，接受结论断言 p≤θ，拒绝结论断言 p>θ+η（其中 η 为用户指定的分离裕度），而“不确定”的结论则表示没有足够的证据支持任一结论。每类错误结论的概率均被其指定的错误预算所约束。为了高效支持这种三值判定契约，ProCerDCL 采用单一序贯检验，消除了基线方法所需的区间族。该检验通过修正的精确二项置信区间或任意时点有效的置信序列来处理重复采样检查。我们将 ProCerDCL 应用于概率化的局部模型语义一致性……

    arXiv:2610.36499v1 Announce Type: cross  Abstract: We present \tool, a query-efficient framework for black-box probabilistic verification. Given a fixed, sampleable input distribution and a binary property oracle, \tool assesses the property's violation probability $p$ against a threshold $\theta$. Under independent sampling and stateless oracle behavior, acceptance asserts $p\le\theta$, rejection asserts $p>\theta+\eta$ for a user-specified separation margin, and an inconclusive verdict indicates insufficient evidence for either conclusion. The probability of each type of erroneous verdict is bounded by its specified error budget. To support this three-valued contract efficiently, \tool uses a single sequential test, eliminating the interval family required by the baseline. The test accounts for repeated sampling checks through either corrected exact binomial confidence intervals or an anytime-valid confidence sequence. We apply \tool to probabilistic local model semantic consistency,
    
[^20]: 当AI能够构建时，我们该评估什么？重新思考工程能力的证据

    When AI Can Build It, What Should We Assess? Rethinking Evidence of Engineering Competence

    [https://arxiv.org/abs/2610.36413](https://arxiv.org/abs/2610.36413)

    该论文在生成式AI时代重新设计了软件工程课程的评估体系，通过工程问责工作流和由“收集—交叉核验—升级”逻辑连接的六渠道证据架构（产品、过程/来源、验证、解释、反思、个人保证），在学生可将工作委托给AI的情况下仍能可靠评估其真实工程能力。

    

    生成式AI能够产生代码、测试、文档和技术解释，这些产物与传统上用作学生能力证据的成果极为相似，从而削弱了从产物质量推断其背后工程能力的可靠性。本文报告了一门为期13周、允许使用AI的软件密集型工程课程的重设计与首次部署，在该课程中学生可以将大量工作委托给AI，同时保留对规划、验证、评审、集成和解释的责任。该重设计将一个工程问责工作流（用于规范人机协作分工并保留人类的决策权威）与一个评估证据架构相结合，该架构包含六个互补渠道：产品、过程/来源、验证、解释、反思和个人保证。一个“收集—交叉核验—升级”（CCE）推理逻辑将这些渠道连接起来，而不是将任何单一产物或痕迹视为充分的证据。

    arXiv:2610.36413v1 Announce Type: new  Abstract: Generative AI can produce code, tests, documentation, and technical explanations that resemble artefacts traditionally used as evidence of student competence, weakening the inference from artefact quality to the engineering capability behind it. We report the redesign and first deployment of a 13-week, AI-permissive, software-intensive engineering course in which students may delegate substantial work to AI while retaining responsibility for planning, verification, review, integration, and explanation. The redesign couples an engineering accountability workflow, which structures human-AI work and preserves human decision authority, with an assessment evidence architecture comprising six complementary channels: product, process/origin, verification, explanation, reflection, and individual assurance. A collect--cross-check--escalate (CCE) reasoning logic connects these channels rather than treating any single artefact or trace as sufficien
    
[^21]: kw-bench：用真实世界任务击穿前沿智能体运行框架

    kw-bench: Breaking Frontier Harnesses on Real-World Tasks

    [https://arxiv.org/abs/2610.36405](https://arxiv.org/abs/2610.36405)

    kw-bench 是基于真实业务委托构建的基准，首次将评估重点从生成式任务转向审计、税务等知识工作中的调查性环节，检验前沿智能体运行框架能否从不完整、不一致的分散证据中精确还原并论证答案。

    

    每一代新的前沿模型都在不断拓展可能性的边界。在一个运行框架（harness）中——包含一个循环、一个 shell、一个文件系统，以及编写并运行自身代码的能力——模型能够承担越来越困难的工作。越来越多的研究将编码智能体的运行框架视为知识工作的基础设施，而非单纯的编码工具，新一波的基准测试也正是在这一场景下对其进行检验。这些基准大多是生成式的：根据简报构建财务模型、备忘录或电子表格，并由评分标准或另一个模型来评判。然而，很少有基准去测试知识工作中最昂贵的部分，即调查性工作。在审计、税务、尽职调查、监管报告、理赔或任何涉及数字的争议中，答案其实早已存在——只是隐藏在散布于各个系统之中、不完整且相互矛盾的证据里——必须将其还原出来，并精确到分地加以论证。我们推出了 kw-bench，这是一个基于专业人士向我们描述的真实业务委托所构建的基准。它的第一个任务族是一个五……（摘要在此处截断）

    arXiv:2610.36405v1 Announce Type: new  Abstract: Each new generation of frontier model pushes the edge of what is possible. Inside a harness -- a loop, a shell, a filesystem, the ability to write and run its own code -- a model can take on increasingly difficult work.   A growing body of work treats the coding-agent harness as infrastructure for knowledge work rather than a coding tool, and a new wave of benchmarks tests it there. Most of them are generative: build a financial model, a memo, a spreadsheet from a brief, judged by a rubric or by another model. Few test the expensive part of knowledge work, which is investigative. In audit, tax, due diligence, regulatory reporting, claims or any dispute over numbers, the answer already exists -- hidden in incomplete and inconsistent evidence spread across systems -- and has to be recovered and defended to the cent.   We introduce kw-bench, a benchmark built from real engagements specialists described to us. Its first task family is a five
    
[^22]: 你的愿望就是我的提交：AI 编程代理中的过度顺从式重构

    Your Wish Is My Commit: Over-Compliant Refactoring in AI Coding Agents

    [https://arxiv.org/abs/2610.36346](https://arxiv.org/abs/2610.36346)

    该论文首次通过受控的迭代重构实验系统研究了 AI 编程代理的谄媚问题，发现 Codex 和 Claude Code 等代理在代码已达最优后仍会持续进行不必要的编辑（过度顺从），并定义了六种量化这种行为的行为信号。

    

    众所周知，大型语言模型具有谄媚性，即使顺从用户毫无益处，它们也会过度迎合用户。当这些模型充当编程代理时，这种行为可能代价高昂：一个在代码已无需修改后仍继续编辑的代理，会将无意义的改动、技术债务和审查负担写入代码库。然而，谄媚性在软件工程中的后果在很大程度上仍未被充分研究。为了理解这些后果，我们对迭代式重构开展了一项受控研究——这是一项没有停止标准的开放式任务，当代码已经很干净时，代理所做的任何编辑都反映的是顺从而非改进。我们在算法类（n=50）和面向对象类（n=10）的测试套件上，跨五个模型反复要求两个广泛使用的代理 Codex 和 Claude Code 继续重构。在超过 3,000 轮交互中，我们定义了六种过度顺从的行为信号。我们假设一个有能力的代理在识别出代码库已被优化后会停止操作；但没有任何一个代理这样做。

    arXiv:2610.36346v1 Announce Type: new  Abstract: Large language models are known to be sycophantic, over-complying with users even when doing so is unproductive. When these models act as coding agents, this can be costly: an agent that keeps editing after no edit improves the code writes churn, technical debt, and review burden into the codebase. Yet the consequences of sycophancy in software engineering remain largely understudied. To understand them, we run a controlled study of iterative refactoring, an open-ended task with no stopping criterion, where any edit an agent makes once the code is already clean reflects compliance rather than improvement. Across algorithmic (n=50) and object-oriented (n=10) suites, we repeatedly ask two widely used agents, Codex and Claude Code, to refactor further across five models. Over 3,000 turns, we define six behavioral signals of over-compliance. We hypothesized that a capable agent would stop once it recognized an optimized codebase; none did. E
    
[^23]: 程序重构下大语言模型生成的证明有多稳定？可验证代码生成的压力测试

    How Stable Are LLM-Generated Proofs Under Program Refactoring? A Stress Test for Verifiable Code Generation

    [https://arxiv.org/abs/2610.36328](https://arxiv.org/abs/2610.36328)

    该论文提出 RefactorProof 基准，通过对 185 个 VERINA 任务生成 458 个经 Lean 认证的行为保持重构变体，揭示了大语言模型生成的证明在程序结构重构下存在显著脆弱性。

    

    arXiv:2610.36328v1 公告类型：新。摘要：可验证代码生成的基准测试通常只针对一个固定的实现来验证证明。我们提出这样一个问题：在保持程序行为不变的重构之后，同一个证明是否仍然有效？我们提出了 RefactorProof，它从在我们固定环境中可编译的 185 个 VERINA 任务中推导出 458 个经 Lean 认证的变体，同时将规范说明和被评估的证明逐字节保持不变。VERINA 为 46 个任务提供了参考证明，这些任务全部来自其基础分割集；主要的模型与参考证明的对比进一步匹配在 21-25 个简短的非递归任务上。结构性重构在各个证明来源中暴露了显著的脆弱性：任务归一化的结构 PSR（证明存活率）在提供的参考证明上为 4.3%，而在专用证明器上为 13.3%-26.8%。在匹配的任务上，专用证明器生成的证明总体上仍比提供的参考证明工件具有更高的未变更证明存活率：DeepSeek-Prover-V2-7B 高出 +15.6 个百分点，+18.8（原文截断）

    arXiv:2610.36328v1 Announce Type: new  Abstract: Benchmarks for verifiable code generation usually test a proof against one fixed implementation. We ask whether the same proof remains valid after a program refactoring that preserves behavior. We introduce RefactorProof, which derives 458 Lean-certified variants from the 185 VERINA tasks that compile in our pinned environment while holding the specification and evaluated proof fixed byte-for-byte. VERINA supplies reference proofs for 46 tasks, all from its basic split; the primary model-versus-reference comparisons further match on 21-25 short, non-recursive tasks. Structural refactorings expose substantial brittleness across proof sources: task-normalized structural PSR is 4.3% for the supplied references and 13.3-26.8% for the specialized provers. On the matched tasks, specialized prover proofs nevertheless have higher overall unchanged-proof survival than the supplied reference artifacts: +15.6 points for DeepSeek-Prover-V2-7B, +18.8
    
[^24]: 构建数学软件——来自GAP开发的经验教训

    Building Mathematical Software -- Lessons from the Development of GAP

    [https://arxiv.org/abs/2610.36300](https://arxiv.org/abs/2610.36300)

    本文以计算机代数系统GAP的开发为具体案例，总结了现有软件工程文献未充分覆盖的数学软件特殊需求与开发经验，并探讨了AI系统对数学软件开发过程的潜在影响。

    

    数学软件有着一些特殊的需求，而这些需求并未被现有的软件工程或算法设计文献完全涵盖。本文以计算机代数系统GAP为具体背景，探讨了其中的若干方面。本文还研究了人工智能系统可能对数学软件开发过程产生的潜在影响。

    arXiv:2610.36300v1 Announce Type: cross  Abstract: Mathematical software has particular needs that are not all covered by the existing literature on software engineering or algorithm design. This paper looks at some of these aspects, in the concrete context of the computer algebra system GAP. It also studies the potential impact AI systems could have on the development process for mathematical software.
    
[^25]: 即时软件工厂

    Instant Software Factories

    [https://arxiv.org/abs/2610.36254](https://arxiv.org/abs/2610.36254)

    该论文提出“即时软件工厂”概念，仅凭一轮人类输入即可通过引导提示词自动配置出定时任务智能体群和共享沟通看板两个协调机制，以可重复流程处理从原型到产品之间的长尾工程工作，并观察到每日代码合并量提升15倍。

    

    编码智能体可以快速构建原型，但从原型到产品之间还隔着一条由修复、优化、测试和发布组成的长尾工作。软件工厂通过可重复的流程来处理这些持续性工作。我们探究的问题是：这些工作是否需要专门的编排，以及应该如何评判这类工厂。给定一个经过测试的代码仓库和一个编码智能体，即时软件工厂仅使用两个协调输入：一群具有固定目标的定时任务（cron-job）智能体，以及一个共享的沟通看板（如GitHub），智能体在其中通过各自留下的痕迹进行协调。我们开发了一个引导提示词（bootstrap prompt），旨在仅通过一轮人类输入就能设置好这两个协调输入，并将治理作为可分离的独立层。该提示词是从两个手动配置的工业工厂中提炼而来的，这两个工厂分别被观察了15天和17天，而该提示词本身尚未经过评估。在GitHub协调的CLI工厂中，智能体部署期间每日代码合并（merge）次数增长了15倍，这是一个观察到的关联。

    arXiv:2610.36254v1 Announce Type: new  Abstract: Coding agents can quickly build prototypes, but a long tail of fixes, optimizations, tests, and releases separates a prototype from a product. A software factory handles this ongoing work through a repeatable process. We ask whether this work needs dedicated orchestration and how such factories should be judged. Given a tested repository and a coding agent, an instant software factory uses only two coordination inputs, a swarm of cron-job agents with standing objectives and a shared communication board, such as GitHub, where agents coordinate through the traces they leave. We developed a bootstrap prompt designed to set up both inputs from one round of human input, with governance as a separable layer. We distilled it from two manually configured industrial factories, observed for 15 and 17 days, and have not yet evaluated it. In the GitHub-coordinated CLI factory, daily merges rose 15x during agent rollout, an observed association. Its 
    
[^26]: MARQ：一个注入系统工程知识的模块化、大模型无关的需求质量评估器

    MARQ: A Modular, LLM-Agnostic Requirement Quality Assessor Infused with Systems Engineering Knowledge

    [https://arxiv.org/abs/2610.36234](https://arxiv.org/abs/2610.36234)

    MARQ是一个模块化、与具体大模型无关的需求质量评估原型，通过注入基于INCOSE准则可操作化的系统工程专家知识，显著提升了大语言模型在不同质量准则下识别需求缺陷的能力，超越了现成模型的表现。

    

    需求工程（RE）是系统工程（SE）的基础，因为需求缺陷会向下游传播并对开发结果产生不利影响。尽管大语言模型（LLM）在需求质量评估方面前景可观，但先前研究表明，现成模型会遗漏大量专家识别出的缺陷，在不同质量准则上的表现参差不齐，且多次执行之间结果存在差异。本研究探究注入专家提炼的系统工程知识能否使基于大语言模型的需求质量评估超越现成模型的能力。我们提出了MARQ，一个模块化、大模型无关的原型系统，它通过一个针对具体质量准则的知识栈来增强大语言模型，该知识栈对INCOSE需求质量准则进行了可操作化的实现。我们在一个由专家标注的NASA需求数据集上，使用Claude Opus 5、GPT-5.6 Sol以及开放权重的Gemma 3 12B进行评估（每个模型执行100次），并与现成基线进行对比。MARQ取得了中位数F1分……（原文摘要在此处截断）

    arXiv:2610.36234v1 Announce Type: new  Abstract: Requirements engineering (RE) is foundational to systems engineering (SE), as requirement defects can propagate downstream and adversely affect development outcomes. Although large language models (LLMs) are promising for requirement quality assessment, prior work shows that off-the-shelf models miss a large share of expert-identified defects, perform unevenly across quality criteria, and vary across executions. This study investigates whether infusing expert-derived SE knowledge can improve LLM-based requirement quality evaluation beyond off-the-shelf capability. We present MARQ, a modular, LLM-agnostic prototype that supplements LLMs with a criterion-specific knowledge stack operationalizing INCOSE requirement quality criteria and evaluated against off-the-shelf baselines on an expert-labeled NASA requirement set using Claude Opus 5, GPT-5.6 Sol, and the open-weight Gemma 3 12B, with 100 executions per model. MARQ achieves median F1 sc
    
[^27]: 智能体还是人类？在缺失作者身份签名的情况下检测AI编程智能体的提交

    Agent or Human? Detecting Commits by AI Coding Agents in the Absence of Authorship Signatures

    [https://arxiv.org/abs/2610.36229](https://arxiv.org/abs/2610.36229)

    该研究基于近800万个提交发现，即使去除作者身份签名，仍可凭借提交信息冗长度和变更规模等特征以最高0.859的ROC AUC区分AI编程智能体与人类的代码提交，且检测信号主要来自元数据而非代码本身。

    

    诸如 Claude Code、GitHub Copilot 编程智能体和 Cursor 等 AI 编程智能体现在可以直接向开源仓库提交代码。针对这些智能体的实证研究通常通过作者身份签名（如机器人账号和 Co-authored-by 尾注）来识别它们的提交。然而，这些签名是可选的、可以被移除的，并且通常附加在拉取请求上而非单个提交上。因此，一旦签名消失，能否将智能体提交与人类提交区分开来尚不清楚。我们使用来自 1,226 个仓库的 7,837,946 个提交（覆盖五个编程智能体及人类提交），研究了无签名检测的信号与局限。在签名被掩盖后，检测器对智能体提交的评分排名高于人类提交，ROC AUC 最高可达 0.859，而且大部分信号来自提交信息的冗长度和变更规模，而非代码本身。在共享样本上，gpt-6-sol 在不进行微调的情况下达到了与 XG（摘要在此处被截断，后续内容缺失）……

    arXiv:2610.36229v1 Announce Type: cross  Abstract: AI coding agents such as Claude Code, GitHub Copilot coding agent, and Cursor now commit directly to open-source repositories. Empirical studies of these agents identify their commits through authorship signatures, such as bot accounts and Co-authored-by trailers. These signatures are optional, can be removed, and are often attached to pull requests rather than to individual commits. Hence, it is unclear whether Agent commits can be told apart from Human commits once the signatures are gone. We investigate the signals and limits of signature-free detection using 7,837,946 commits from 1,226 repositories, covering five coding agents and Human commits. After signatures are masked, detectors rank Agent commits above Human commits with a ROC AUC of up to 0.859, and most of this signal comes from message verbosity and change size rather than from the code itself. On the shared sample, gpt-6-sol without fine-tuning reaches AUCs similar to XG
    
[^28]: 技能梦境：通过拉取请求流演化仓库技能

    Skill Dreaming: Evolving Repository Skills through Pull Request Streams

    [https://arxiv.org/abs/2610.36188](https://arxiv.org/abs/2610.36188)

    提出了 Skill Dreaming 框架，以拉取请求流为知识来源，实现拉取请求、仓库与技能库的共同演化，使仓库专属技能随代码库的变化持续更新并反哺未来的 PR 实践。

    

    大语言模型（LLM）智能体在复杂的软件工程任务中越来越依赖可复用的技能，这使得技能库对于仓库的开发与维护至关重要。然而，针对特定仓库的技能必须随着代码结构、实现规范和维护需求的变化而不断演化。拉取请求记录了仓库在实现与审查过程中的演变，为此类知识提供了天然的来源。同时，新的拉取请求必须遵循仓库特定的实现模式，一致地修改相关组件，并确保在为新功能添加测试覆盖的同时，现有测试仍能继续通过。这就形成了一个相互增强的循环：拉取请求为技能更新提供知识，而这些技能又能改进未来的拉取请求、维护仓库，并支持技能的进一步演化。为了实现拉取请求、仓库和技能的这种共同演化，我们提出了 Skill Dreaming，一个框架……

    arXiv:2610.36188v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly rely on reusable skills for complex software engineering tasks, making skill libraries essential for repository development and maintenance. However, repository-specific skills must evolve with changing code structures, implementation conventions, and maintenance requirements. Pull requests (PRs) provide a natural source of such knowledge by recording how repositories evolve through implementation and review. Meanwhile, new PRs must follow repository-specific implementation patterns, modify related components consistently, and ensure that existing tests continue to pass while adding coverage for new functionality. This creates a mutually reinforcing cycle: PRs provide knowledge for updating skills, while these skills improve future PRs, maintain repositories, and support further skill evolution. To enable this co-evolution of PRs, repositories, and skills, we propose Skill Dreaming, a framew
    
[^29]: AI能否在不生成测试的情况下指导测试？一项关于AI辅助单元测试的对比研究

    Can AI Guide Testing Without Generating Tests? A Comparative Study of AI-Assisted Unit Testing

    [https://arxiv.org/abs/2610.36185](https://arxiv.org/abs/2610.36185)

    该研究提出TestCompass——一个不代学生生成可执行测试、而是针对其测试套件薄弱点提供定向反馈的生成式AI辅导工具，并通过18名学生的课堂对比实验证明AI可以在不直接生成测试的前提下有效支持初学者的单元测试学习。

    

    生成式人工智能的最新进展使AI助手能够生成日益复杂的可执行单元测试。然而，初学者可能会直接接受这些输出，而不去独立思考应该测试什么、评估测试套件的完整性或验证其正确性，这可能限制他们发展关键测试技能的机会。尽管先前的研究已经探索了AI辅助测试生成以及面向反馈的编程辅导工具，但对于AI能否在不为初学者生成测试的情况下支持他们的测试实践，人们仍然知之甚少。为了探究这一问题，我们提出了TestCompass——一个生成式AI辅导工具，它针对学生测试套件中的薄弱环节提供有针对性的反馈，但并不生成可执行的测试。在一项包含18名学生的课堂研究中，我们在两个Java单元测试任务上将TestCompass与通用AI助手进行了对比，考察了学生的测试行为、测试套件质量以及主观体验。（摘要原文在此处中断）

    arXiv:2610.36185v1 Announce Type: cross  Abstract: Recent advances in generative AI have enabled assistants to generate increasingly sophisticated executable unit tests. However, novices may accept these outputs without independently identifying what to test, evaluating test-suite completeness, or verifying correctness, potentially limiting opportunities to develop essential testing skills. Although prior work has explored AI-assisted test generation and feedback-oriented programming tutors, little is known about whether AI can support novices' testing practices without generating tests for them. To investigate this, we introduce TestCompass, a generative AI tutor that provides targeted feedback on weaknesses in students' test suites without generating executable tests. In a classroom study with 18 students, we compared TestCompass with a general AI assistant across two Java unit-testing tasks, examining students' testing behaviors, test-suite quality, and subjective experiences. We fo
    
[^30]: Stratum：超越上下文窗口的编码智能体仓库记忆

    Stratum: Repository Memory for Coding Agents Beyond the Context Window

    [https://arxiv.org/abs/2610.36153](https://arxiv.org/abs/2610.36153)

    该论文提出Stratum，为编码智能体提供跨会话的持久化、经验感知的仓库记忆系统，并通过实证测量证明真正的瓶颈在于相关知识的选择而非上下文窗口容量。

    

    上下文窗口已经增长了一千倍，然而编码智能体在每次会话开始时仍然对过去的工作毫无记忆。它们会重新读取文件、重试失败的方法，并忘记几小时前已经确定的根本原因。这不是上下文长度问题，而是记忆架构问题。我们的假设是：编码智能体需要持久的、具有经验感知能力的仓库记忆，使知识能够从一个会话延续到下一个会话。在真实仓库上的三项测量结果支撑了这一设计。首先，中等规模的项目已经超过最大的生产级上下文窗口，并且将知识存储在模型的键值（KV）缓存中的字节开销，比将其存储为索引文本高出 $10^4$ 到 $10^5$ 倍。其次，在752个真实的缺陷修复提交中，修复操作的中位数只编辑两个文件，这些文件及其直接导入的内容合计不到仓库的10%；瓶颈在于选择，而非容量。第三，即使考虑传递性导入，……

    arXiv:2610.36153v1 Announce Type: cross  Abstract: Context windows have grown a thousand-fold, yet coding agents still start every session with no memory of their past work. They re-read files, retry failed approaches, and forget root causes they established hours earlier. This is not a context-length problem but a memory-architecture problem. Our hypothesis: coding agents need persistent, experience-aware repository memory, so that knowledge survives from one session to the next. Three measurements on real repositories motivate the design. First, mid-size projects already exceed the largest production context windows, and storing knowledge inside the model's key-value (KV) cache costs $10^4$ to $10^5$ times more bytes than storing it as indexed text. Second, across 752 real bug-fix commits, the median fix edits two files, and these files plus their direct imports amount to less than 10% of the repository; the bottleneck is selection, not capacity. Third, even with transitive import re
    
[^31]: 上下文是一条流水线，而非一扇窗口：LLM智能体框架中的信息损耗

    Context Is a Pipeline, Not a Window: Information Attrition in LLM Agent Harnesses

    [https://arxiv.org/abs/2610.36114](https://arxiv.org/abs/2610.36114)

    该研究揭示了一个被忽视的关键问题：即使LLM智能体的上下文窗口利用率仅为5.75%，智能体框架仍会通过固定长度切分JSON的方式截断三分之一的工具结果，导致末尾数据不均匀丢失，使不同字段到达模型的比例差异悬殊（4.7%对99.3%），因此评估智能体不能只看窗口大小和答案正确率，还必须关注“信息损耗”。

    

    人们通常用两个数字来评判LLM智能体：上下文窗口的大小，以及最终答案的正确率。但这两个数字都忽略了第三件事：工具的输出在模型看到之前损失了多少。我们在一个真实的工具使用型智能体中测量了这种损耗。该智能体通过读取Linux内核跟踪记录来查找注入故障的原因。上下文窗口几乎是空的：在720次运行中，单个最大的提示词仅占用了窗口的5.75%。然而，三分之一的工具结果（11,189个中的3,827个）在到达模型之前就被截断了，而且每一个被截断的结果都变成了损坏的JSON。该框架通过在固定长度处切分JSON文本来截断每个结果，这并不会均匀地缩短结果，而是删除了位于末尾附近的任何数据。其中一个字段仅在4.7%的调用中到达模型，而在完全相同的结果中，另一个字段在99.3%的调用中到达模型。哪个字段包含故障的证据取决于故障进程启动的时间：早期……

    arXiv:2610.36114v1 Announce Type: cross  Abstract: LLM agents are judged by two numbers: the size of the context window, and how often the final answer is correct. Both numbers miss a third thing: how much of a tool's output is lost before the model ever sees it. We measured this loss in a real tool-using agent. The agent finds the cause of injected faults by reading Linux kernel traces. The context window was almost empty: across 720 runs, the single largest prompt used only 5.75% of it. Yet a third of all tool results (3,827 of 11,189) were cut before they reached the model, and every cut result became broken JSON. The harness cut each result by slicing its JSON text at a fixed length. This does not shorten a result evenly. It deletes whatever data sits near the end. One field reached the model in only 4.7% of calls. A second field, in the very same results, reached the model in 99.3% of calls. Which field held the fault's evidence depended on when the faulty process started: early o
    
[^32]: AGDecode：用于大语言模型安全约束解码的语义属性文法导数

    AGDecode: Semantic Attribute Grammar Derivatives for Safety-Constrained Decoding of Large Language Models

    [https://arxiv.org/abs/2610.36105](https://arxiv.org/abs/2610.36105)

    AGDecode提出了一种基于最小不动点语义的可编程语义约束解码框架，通过将属性文法扩展到部分构造的语法树并把语义约束解码归约为可实现性问题，从而保证大语言模型所生成程序的结构与语义正确性。

    

    大语言模型（LLM）越来越多地被用于代码生成，这促使人们研究能够保证所生成程序结构与语义正确性的技术。约束解码通过在生成过程中将词元选择限制在至少存在一个有效补全的选项上，从而强制保证正确性。虽然现有方法可以强制执行句法约束以及某些选定的语义属性（如类型安全），但它们要么只针对特定属性，要么依赖于难以推广到数据相关、归纳性程序属性的操作性剪枝机制。我们提出了AGDecode，一个基于最小不动点语义的可编程语义约束解码框架。AGDecode将属性文法扩展到部分构造的语法树上，并将语义约束解码归约为这类树上的可实现性问题。该框架将语义规范编译为关系约束系统（原文摘要在此处截断）。

    arXiv:2610.36105v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for code generation, motivating techniques that guarantee the structural and semantic correctness of generated programs. Constrained decoding enforces correctness during generation by restricting token choices to those that admit at least one valid completion. While existing approaches can enforce syntactic constraints and selected semantic properties such as type safety, they either target specific properties or rely on operational pruning mechanisms that are difficult to generalize to data-dependent, inductive program properties. We present AGDecode, a programmable framework for semantic constrained decoding based on least-fixed-point semantics. AGDecode extends attribute grammars to partially constructed syntax trees, and reduces semantic constrained decoding to a realizability problem over such trees. The framework compiles semantic specifications into systems of relational constrain
    
[^33]: 编程智能体还需要调试器吗？基于前沿模型的故障定位逼近准确率上限

    Do Coding Agents Still Need a Debugger? Frontier-Model-Based Fault Localization at the Accuracy Ceiling

    [https://arxiv.org/abs/2610.36015](https://arxiv.org/abs/2610.36015)

    该研究通过在610个注入故障的大规模实验发现，基于前沿大模型的编程智能体仅凭源代码推理即可将故障定位准确率提升至98%以上的天花板水平，调试器的额外辅助收益已微乎其微。

    

    背景：为了定位故障，编程智能体可以仅凭源代码进行推理，也可以借助调试器，通过运行程序并在交互式调试器中单步执行来获得辅助。虽然早期的智能体在静态执行追踪和调试器操作方面表现不佳，但前沿大语言模型在这两方面均已显著提升。目标：我们评估编程智能体在不使用工具的情况下，在原始程序和变换后程序上定位故障的准确程度，调试器辅助能在多大程度上提升准确率以及代价是什么，以及当调试器可选时智能体如何使用它。方法：我们将610个单行故障注入144个LeetCodeDataset解答中，并用五种保持行为不变的变换配方重写每个故障程序。基于Sol 6.1、Luna 6和Qwen 3.8的智能体在无调试器辅助、可选调试器辅助或鼓励使用调试器辅助的条件下对每个故障进行定位。结果：在不使用工具的情况下，各智能体在原始程序中定位了98.2%至99.9%的故障，在变换后程序中定位了96.7%至99.9%的故障，损失了（原文摘要在此处截断）。

    arXiv:2610.36015v1 Announce Type: cross  Abstract: Background: To localize a fault, a coding agent can reason from source code alone or use debugger assistance by running the program and stepping through it in an interactive debugger. While earlier agents struggled with static execution tracing and debugger operation, frontier LLMs have improved at both. Objective: We evaluate how accurately coding agents localize faults without tools across original and transformed programs, how much debugger assistance improves accuracy and at what cost, and how agents use it when optional. Methods: We seed 610 single-line faults into 144 LeetCodeDataset solutions and rewrite each faulty program with five recipes of behavior-preserving transformations. Agents based on Sol 6.1, Luna 6, and Qwen 3.8 localize each fault with no, optional, or encouraged debugger assistance. Results: Without tools, the agents localized 98.2 to 99.9% of faults in original and 96.7 to 99.9% in transformed programs, losing a
    
[^34]: CIVER：面向可靠代码评审的因果干预验证

    CIVER: Causal Intervention Verification for Reliable Code Review

    [https://arxiv.org/abs/2610.36010](https://arxiv.org/abs/2610.36010)

    CIVER提出了一种基于因果干预的验证框架，通过识别代码评审发现背后的核心问题来区分真实缺陷与误报，从而提升LLM代码评审的可靠性。

    

    代码评审对于维护软件仓库至关重要，它通过识别缺陷、提高代码质量并支持可靠的软件演化发挥作用。随着基于大语言模型（LLM）的编码智能体日益普及，越来越多的代码贡献和仓库变更由这些智能体生成。因此，基于LLM的代码评审智能体正变得越来越普遍，用于评估这些贡献并维护软件质量。然而，LLM评审者常常生成包含大量潜在问题的冗长评审，其中许多是误报或缺乏充分证据。这些不可靠的评审会误导开发者和下游编码智能体，因此亟需区分可操作的发现与缺乏依据的断言。我们提出了CIVER，一个基于因果干预的框架，用于识别代码评审发现背后的核心问题，并区分真实缺陷与误报。CIVER融合……

    arXiv:2610.36010v1 Announce Type: cross  Abstract: Code review is essential for maintaining software repositories by identifying defects, improving code quality, and supporting reliable software evolution. With the growing adoption of LLM-based coding agents, an increasing number of code contributions and repository changes are generated by these agents. Consequently, LLM-based code review agents are becoming increasingly prevalent to evaluate these contributions and maintain software quality. However, LLM reviewers often generate lengthy reviews containing numerous potential issues, many of which are false positives or lack sufficient evidence. These unreliable reviews can mislead developers and downstream coding agents, motivating the need to distinguish actionable findings from unsupported claims. We propose CIVER, a causal intervention-based framework that identifies the core issues underlying code-review findings and distinguishes genuine defects from false positives. CIVER integr
    
[^35]: 首次编辑之前？测量编码代理指导实际到达的时机

    Before the First Edit? Measuring When Coding-Agent Guidance Actually Arrives

    [https://arxiv.org/abs/2610.35992](https://arxiv.org/abs/2610.35992)

    该研究通过审查五个事件并在代理每个执行动作前后对Python文件进行哈希，测量编码代理在首次源代码编辑前的指导实际到达时机，发现指导确实先于所有观察到的首次变更，但研究未达到预设的20个分支覆盖目标。

    

    当编码代理框架承诺在首次源代码编辑之前提供指导时，它实际拦截的是哪个事件？我们审查了五个独立的事件：策略分配、请求资格、触发、计划交付以及观察到的源内容变更。规定动作比较揭示了三种门控规则的覆盖范围，但无法告诉我们代理在收到计划后会如何继续。因此，我们在公开软件议题上运行本地代理，并在每个执行动作前后对Python文件进行哈希。在一项以试点为指导、使用Qwen2.5-Coder 14B的76个分支研究中，有15个分支在七个任务和三个代码库中改变了Python源代码。指导先于每一次观察到的首次变更，但每次变更都使用了结构化编辑器，且该研究未达到预设的20个分支覆盖目标。事后本地测试发现七个变更分支通过了测试目标和回归选择器；其中四个使用了不完整的回归列表。公开暂停（PAUSE）

    arXiv:2610.35992v1 Announce Type: new  Abstract: When a coding-agent harness promises guidance before the first source edit, what event does it actually intercept? We audit five separate events: policy assignment, request eligibility, trigger, plan delivery, and observed source-content change. A prescribed-action comparison exposes the coverage of three gate rules but cannot tell us how an agent would continue after receiving a plan. We therefore run local agents on public software issues and hash Python files before and after each executed action. In a pilot-guided, 76-arm study using Qwen2.5-Coder 14B, 15 arms change Python source across seven tasks and three repositories. Guidance precedes every observed first change, but every change uses a structured editor and the study misses its prespecified 20-arm coverage target. Post hoc local tests find seven changed arms that pass the tested target and regression selectors; four of those seven use incomplete regression lists. Public PAUSE 
    
[^36]: PROBE：前沿编码智能体发现的Bug与维护者修复的Bug不同

    PROBE: Frontier Coding Agents Find Different Bugs Than Maintainers Fix

    [https://arxiv.org/abs/2610.35987](https://arxiv.org/abs/2610.35987)

    本文提出PROBE基准测试，揭示前沿编码智能体在仓库漏洞发现中召回率和精确率均很低，且其找到的漏洞与人类维护者实际修复的漏洞存在系统性差异。

    

    主动式漏洞发现是软件工程中的一个重要问题。尽管编码智能体已在该领域被广泛部署，但其能力仍未得到充分研究。特别是，在整个代码仓库中找出人类维护者会修复的所有漏洞这一具有挑战性的任务尚未被研究过。为了填补这一空白，我们提出了PROBE，一个包含20个多样化GitHub代码仓库在18个月内修复的全部2,457个漏洞的基准测试，并配套了一种将候选发现与该部分参考集进行匹配的方法。我们评估了基于多个前沿模型的编码智能体，发现它们相对于参考集的召回率均低于21%，精确率低于9%。虽然更强的模型将召回率从8%大幅提升至17%，但并不能提高精确率。令人惊讶的是，尽管精确率很低，LLM和人类评估者仍然判定85%的发现是有效的。

    arXiv:2610.35987v1 Announce Type: cross  Abstract: Proactive bug discovery is an important software-engineering problem. Even though coding agents are already widely deployed in this domain, their capabilities remain understudied. In particular, the challenging task of finding all bugs that human maintainers would repair across an entire repository has not been studied yet. To close this gap, we introduce PROBE, a benchmark consisting of all 2,457 bugs fixed over the course of 18 months across 20 diverse GitHub repositories, together with a method for matching candidate findings to this partial reference set. We evaluate coding agents across frontier models and find they all achieve a recall of below $21\%$ and a precision of less than $9\%$ with respect to the reference set. While stronger models increase recall substantially from $8\%$ to $17\%$, they do not improve precision. Surprisingly, despite the low precision, LLM and human evaluators still judge $85\%$ of findings valid. Cruc
    
[^37]: 生成与验证：面向安全关键汽车软件工程的人机智能体协作

    Generation and Verification: Human-Agent Collaboration for Safety-Critical Automotive Software Engineering

    [https://arxiv.org/abs/2610.35982](https://arxiv.org/abs/2610.35982)

    该论文提出一个面向安全关键汽车 C/C++ 软件的编程助手，通过 MCP 协议向开发者开放生成、编辑、验证三个可按需调用的工具以实现人机协作验证，并针对汽车领域定制底层大语言模型。

    

    有关编程智能体的研究主要聚焦于生成质量。然而，安全关键的汽车软件提出了两项额外要求。其一，它高度重视验证，因为每一处代码变更都必须经过审查、追溯到需求，并通过 MISRA 和 AUTOSAR 等行业标准的检查后才能提交。其二，它是一个狭窄的领域，代码需遵循分层架构、嵌入式资源约束以及组织特定的接口规范，而这些内容在通用大语言模型（LLM）的训练数据中十分稀缺。我们提出一个面向汽车 C/C++ 的编程助手，以同时应对这两项要求。首先，它通过模型上下文协议向用户开放三个工具（生成、编辑和验证），使开发者能够按需对任意代码运行验证，而不必依赖智能体自行检查其输出。其次，我们不使用通用模型，而是通过提示工程等方式针对汽车领域对底层大语言模型进行定制（原文摘要在此处被截断）。

    arXiv:2610.35982v1 Announce Type: cross  Abstract: Research on coding agents mainly focuses on generation quality. Yet, safety-critical automotive software makes two additional demands. It prioritizes verification, since every change is reviewed, traced to a requirement, and checked against industry standards such as MISRA and AUTOSAR before it can be committed. It is also a narrow domain, written against layered architectures, embedded resource constraints, and organization-specific interface conventions that are scarce in the data general-purpose LLMs are trained on. We present a coding assistant for automotive C/C++ that addresses both demands. First, it exposes three tools to the user over the Model Context Protocol (generate, edit, and verify), allowing developers to run verify on any code on demand rather than relying on the agent to check its own output. Second, rather than using a general-purpose model, we customize the underlying LLM for the automotive domain through prompt en
    
[^38]: 小型智能体应当了解其工具的作用

    Small agents should know what their tools do

    [https://arxiv.org/abs/2610.35981](https://arxiv.org/abs/2610.35981)

    本文提出EffectWorld基准，通过为工具添加效应类型声明（幂等性及读写资源），让小型语言模型能够生成效应安全且高效的多调用工具程序，将Qwen3-4B的任务成功率从53%大幅提升至99%。

    

    小型开源语言模型能够调用工具，但它们并不对调用的效果进行推理：它们不加询问就删除、重复写入、重新读取本已掌握的状态，并且每轮只发出一次调用，而一段简短的程序本可完成任务。我们提出了EffectWorld——一个由效应类型工具构成的可执行世界，其中每个工具都声明其效应、是否幂等，以及它读取和写入哪些资源。这些声明使得重放验证器无需标注或评判模型即可判定一条轨迹是否效应安全，使执行器能在单轮中运行包含多个依赖调用的工具程序，并使一个编译器风格的优化器能够将浪费性轨迹转化为经过验证的训练信号；这些共同使EffectWorld成为一个衡量效应安全性与调用效率的基准。在该数据上训练的低秩适配器带来了显著提升：在Qwen3-4B模型上，任务成功率在分布内场景下从53%提升至99%，在包含未见任务类型和分布外元素的保留测试集上从52%提升至81%……

    arXiv:2610.35981v1 Announce Type: new  Abstract: Small open language models can call tools, but they do not reason about what a call does: they delete without asking, repeat writes, re-read state they already hold, and issue one call per turn where a short program would do. We introduce EffectWorld, an executable world of effect-typed tools in which every tool declares its effect, whether it is idempotent, and which resources it reads and writes. These declarations let a replay verifier decide whether a trace is effect-safe without labels or judge models, let an executor run tool programs of several dependent calls in one turn, and let a compiler-style optimizer turn wasteful traces into verified training signal; together they make EffectWorld a benchmark of effect safety and call efficiency. Low-rank adapters trained on this data deliver large gains: on Qwen3-4B, they raise task success from 53% to 99% in distribution, from 52% to 81% on a held-out split with unseen task kinds and dis
    
[^39]: UnityBench-Mini3D：评估Unity中3D游戏开发的LLM智能体

    UnityBench-Mini3D: Evaluating LLM Agents for 3D Game Development in Unity

    [https://arxiv.org/abs/2610.35980](https://arxiv.org/abs/2610.35980)

    UnityBench-Mini3D是一个包含77个Unity项目级任务的基准，通过1,178个人工编写的引擎内里程碑测试对LLM智能体的3D游戏开发能力进行确定性评估，可提供部分得分、定位失败点，无需依赖主观的LLM评判器。

    

    游戏开发对代码智能体的考验方式与一般软件工程不同：正确性取决于引擎API、序列化场景状态和运行时行为之间的交互，而不仅仅是源代码本身。现有基准主要针对Godot等平台，并且严重依赖LLM评判器或粗粒度的任务级结果。我们提出了UnityBench-Mini3D，这是一个包含77个项目级Unity任务的基准，围绕七个由专家定义的技能类别（而非教程或游戏类型）组织，并设有三个难度等级。每个任务将一个完整的Unity项目与自然语言需求相配对，并使用1,178个留出的、由人工编写的里程碑测试进行评估，这些测试涵盖场景结构、组件配置和运行时行为。这种引擎内评估提供了确定性的部分得分，可以定位失败点，并区分部分实现与完整解决方案，而无需依赖主观评判器。在21个模型和9个（摘要在此处截断）……

    arXiv:2610.35980v1 Announce Type: cross  Abstract: Game development stresses coding agents in ways general software engineering does not: correctness depends on how engine APIs, serialized scene state, and runtime behavior interact, not on source code alone. Existing benchmarks mainly target platforms such as Godot and heavily rely on LLM judges or coarse task-level outcomes. We present UnityBench-Mini3D, a benchmark of 77 project-level Unity tasks organized around seven expert-defined skill categories rather than tutorials or genres, with three difficulty tiers. Each task pairs a complete Unity project with natural-language requirements and is evaluated using 1,178 held-out, human-authored milestone tests covering scene structure, component configuration, and runtime behavior. This in-engine evaluation provides deterministic partial credit, localizes failures, and distinguishes partial implementations from complete solutions without relying on subjective judges. Across 21 models and 9
    
[^40]: 测试智能体服务系统：跨越物理-数字连续体的智能体生活实验室方法论

    Testing Agentic Service Systems: An Agentic Living Lab Methodology Across the Physical-Digital Continuum

    [https://arxiv.org/abs/2610.35978](https://arxiv.org/abs/2610.35978)

    提出了一种“智能体生活实验室”测试方法论，以服务片段为测试单元、区分被测智能体与环境参与者，并通过显式的现实基底绑定与语义差异披露，解决智能体服务系统在物理-数字连续体上超越单纯任务成功的授权、义务与执行环境依赖等测试难题。

    

    对于那些代表委托人行事、改变持久业务状态并影响物理过程的智能体而言，仅有任务成功是不够的：一个有用的结果可能是未经授权的、留下未解决的义务，或者依赖于执行基底。我们提出了一种“智能体生活实验室”方法论，围绕三个问题组织：什么是测试单元、谁和什么参与测试、以及在何种现实下执行测试？测试单元是一个服务片段，其中包含委托角色、义务、世界状态和证据。该方法将被测智能体与环境参与者分离，并区分真实人类、人类痕迹、规则替代物和基于智能体的代理。实现配置文件、声明能力的现实基底绑定、接地配置文件和语义差异披露使跨越软件、企业系统、仿真和物理环境的变更变得明确。一个四平面参考架构分离了实验控制、参与者……（摘要在此处截断）

    arXiv:2610.35978v1 Announce Type: new  Abstract: Task success is insufficient for testing agents that act for principals, change persistent business state, and affect physical processes: a useful result may be unauthorized, leave obligations unresolved, or depend on the execution substrate. We propose an Agentic Living Lab methodology organized around three questions: what is the test unit, who and what participate, and under what reality is the test executed? The unit is a service episode containing delegated roles, obligations, world state, and evidence. The method separates agents under test from environment actors and distinguishes real humans, human traces, rule surrogates, and agent-based proxies. Realization Profiles, capability-declared Reality Substrate bindings, Grounding Profiles, and Semantic Delta disclosures make changes across software, enterprise systems, simulation, and physical settings explicit. A four-plane reference architecture separates experiment control, partic
    
[^41]: 是继续还是重新开始？编码智能体中早期失败预测的局限性

    Continue or Restart? The Limits of Early Failure Prediction in Coding Agents

    [https://arxiv.org/abs/2610.35971](https://arxiv.org/abs/2610.35971)

    研究发现编码智能体在早期执行阶段（5k-20k token）几乎无法可靠预测单次尝试的成败，任务内失败预测能力接近随机水平，因此基于早期失败预测的提前停止策略在节省计算成本方面效果有限。

    

    提前停止有望通过及早终止没有前景的运行来降低自主编码智能体的计算成本。决定何时放弃一次运行，需要将基线问题难度与单次尝试的轨迹区分开来。我们在来自两个公开语料库、涵盖四种模型-框架配置的147,002条执行轨迹中研究了这一区别。通过对比执行前的问题上下文与早期执行前缀，我们评估了线性计数模型、首轮文本嵌入、任务条件排序器以及一个微调的Qwen3-0.6B监控器。尽管早期前缀能在跨任务结果排序上带来适度改进，但在早期节点（5k-20k token）上，任务内的判别能力仍接近随机水平：首轮嵌入和20k token的任务条件模型得到的AUC约为0.50，并且在20k token处对32,406次尝试进行的三种子神经评估显示，在不同解题率的任务中，平均任务内AUC为0.50 [0.48, 0.51]。

    arXiv:2610.35971v1 Announce Type: new  Abstract: Early stopping promises to reduce compute costs for autonomous coding agents by terminating unpromising runs early. Deciding when to abandon a run requires separating baseline issue difficulty from the trajectory of an individual attempt. We study the distinction in 147,002 execution traces from two public corpora spanning four model-framework settings. Comparing pre-execution issue context against early execution prefixes, we evaluate linear count models, initial-turn text embeddings, task-conditional rankers, and a fine-tuned Qwen3-0.6B monitor. Although early prefixes modestly improve cross-task outcome ranking, within-task discrimination remains near chance across early landmarks (5k-20k tokens): initial-turn embeddings and 20k-token task-conditional models yield AUCs around 0.50, and three-seed neural evaluations across 32,406 attempts at 20k tokens achieve a mean within-task AUC of 0.50 [0.48, 0.51] across tasks of varying solve ra
    
[^42]: “PASS”是一个声明，而非观察结果：在活跃多智能体审计层中对10个自我断言修复声明的机械化再验证

    PASS Is a Claim, Not an Observation: Mechanical Re-verification of 10 Self-Asserted Fix Claims in a Live Multi-Agent Audit Layer

    [https://arxiv.org/abs/2610.35950](https://arxiv.org/abs/2610.35950)

    该论文对一个活跃多智能体平台中特权协调智能体发出的10个自我断言修复声明进行了机械化再验证，发现各标准下的通过率仅为60%-80%，证明智能体的“PASS”修复声明不能等同于经过验证的真实观察。

    

    当智能体声称已经修复了自身缺陷时，我们如何知道修复是真实的？我们对来自一个在AITRAP（一个活跃的多智能体知识树平台）中运行的特权协调智能体（azone_god）的10个自我断言的代码级修复声明进行了机械化再验证。这10个声明是从一个包含179条协调广播的语料库中提取的（其中28条包含结构化修复断言，31个唯一的修复ID/消息对，18个不同的修复标识符）。修复声明使用英/中混合格式（PASS、对勾和“验证通过”声明）。每个代码级声明都根据三个标准进行检查：（1）文件修改时间戳（mtime）一致性——通过率为70%；（2）在指定标识符和调用点处声称的防御模式的存在性——在声称的文件中通过率为60%（在整个代码库的任意位置为80%）；（3）仅基于广播公告进行的确定性规则再验证——通过率为80%。关键发现：在c……（原文摘要在此处被截断）

    arXiv:2610.35950v1 Announce Type: cross  Abstract: When agents claim to have fixed their own defects, how do we know the fix is real? We mechanically re-verified 10 self-asserted code-level fix claims from a privileged coordination agent (azone_god) operating in AITRAP, a live multi-agent knowledge-tree platform. These 10 claims were extracted from a corpus of 179 coordination broadcasts (28 containing structured fix assertions, 31 unique fix-ID/message pairs, 18 distinct fix identifiers). Fix claims use mixed English/Chinese formats (PASS, checkmarks, and "verification passed" declarations). Each code-level claim was checked against three criteria: (1) file modification timestamp (mtime) consistency -- 70% pass; (2) presence of the claimed defensive pattern at the specified identifier and call site -- 60% pass in the claimed file (80% anywhere in the codebase); (3) deterministic rule-based re-verification from the broadcast announcement alone -- 80% pass. The critical finding: among c
    
[^43]: KORA：面向AI工作负载的确定性优先执行控制

    KORA: Deterministic-First Execution Control for AI Workloads

    [https://arxiv.org/abs/2610.35869](https://arxiv.org/abs/2610.35869)

    KORA提出了一种“确定性优先”的推理前执行控制架构，优先尝试确定性和结构化路由来处理请求，仅在确定性处理不足时才升级到模型调用，在基准测试中避免了80%的模拟模型调用且结果零不匹配。

    

    AI应用常常在判断推理是否必要之前就调用模型。这种默认做法对模糊的生成式任务很有用，但对于结构化、确定性、受策略约束、面向查找或可直接验证的工作来说并不匹配。KORA引入了确定性优先的执行控制：一个推理前的控制层，它将请求表示为任务路径，首先尝试确定性或结构化路由，验证路由结果，记录遥测数据，并且仅在确定性处理不足时才升级到模型候选路径。本文将KORA作为一种确定性优先的执行控制架构提出，并报告了来自可复现基准测试与验证工作负载的有界证据。在一个可复现的100任务确定性密集型基准测试中，与朴素的直接基线相比，KORA控制的执行避免了100次模拟模型调用中的80次，且在确定性处理中零不匹配。

    arXiv:2610.35869v1 Announce Type: cross  Abstract: AI applications often invoke a model before they determine whether inference is necessary. That default is useful for ambiguous generative tasks, but it is poorly matched to work that is structured, deterministic, policy-bound, lookup-oriented, or directly verifiable. KORA introduces deterministic-first execution control: a pre-inference control layer that represents requests as task paths, attempts deterministic or structured routes first, validates route outcomes, records telemetry, and escalates to a model-candidate path only when deterministic handling is insufficient. This paper presents KORA as a deterministic-first execution-control architecture and reports bounded evidence from reproducible benchmark and validation workloads. In a reproducible 100-task deterministic-heavy benchmark, KORA-controlled execution avoided 80 of 100 simulated model invocations versus a naive direct baseline, with zero mismatches in the deterministic s
    
[^44]: 中间的智能体：AI中介访问如何重塑Web漏洞利用

    The Agent in the Middle: How AI-Mediated Access Reshapes Web Vulnerability Exploitation

    [https://arxiv.org/abs/2610.35541](https://arxiv.org/abs/2610.35541)

    提出可执行基准测试VELA，系统研究了AI智能体中介访问对Web漏洞利用的影响，发现面向智能体的接口设计决定了漏洞可利用性，参考设计下91%的漏洞仍然可被利用。

    

    Web服务越来越多地暴露后端接口，允许用户将任务委托给AI智能体，这些智能体无需浏览网站即可直接调用服务操作。这种转变改变了现有Web漏洞被利用的方式。我们通过VELA来研究这些变化，VELA是一个可执行的基准测试，涵盖十二个Web应用家族中的44个漏洞。该基准保留了存在漏洞的后端，并从合法工作流中推导出面向智能体的接口。我们在多种接口配置下评估了三个商业智能体，使用受控比较来区分接口效应与智能体行为，并使用独立的判定标准来验证攻击结果。我们的结果表明，在参考配置下，智能体能够实现直接接口调用所能实现的所有攻击，因此可利用性主要由接口决定：在参考设计下，91%的漏洞仍然可被利用。

    arXiv:2610.35541v1 Announce Type: cross  Abstract: Web services increasingly expose backend interfaces that let users delegate tasks to AI agents, which invoke service operations directly without navigating websites. This shift changes how existing Web vulnerabilities can be exploited. We investigate these changes through VELA, an executable benchmark covering 44 vulnerabilities across twelve Web application families. The benchmark preserves vulnerable backends and derives agent-facing interfaces from legitimate workflows. We evaluate three commercial agents across multiple interface configurations, using controlled comparisons to separate interface effects from agent behavior and independent oracles to verify attack outcomes. Our results show that the agent realizes every attack that direct interface invocation realizes in the reference configuration, so exploitability is set mainly by the interface: 91% of vulnerabilities remain exploitable under the reference design. Interface restr
    
[^45]: JANUS：针对Web反序列化攻击的跨层防御自动合成方法

    JANUS: Synthesizing Cross-Layer Defenses Against Web Deserialization Attacks

    [https://arxiv.org/abs/2610.35532](https://arxiv.org/abs/2610.35532)

    JANUS提出了一种红蓝对抗框架，通过黑盒攻击者迭代探测攻击变体与防御者进行反事实分析并细化策略，实现针对Web反序列化攻击的跨层防御自动化合成。

    

    不安全的反序列化会将不受信任的数据转化为攻击者可控的执行，而单个漏洞可能允许多种攻击变体，从而绕过针对已知攻击定制的防御。现有防御通常在单一执行点运行，对完整攻击的可见性有限。跨层组合防御控制可以提供互补保护，但协调这些控制以保留合法功能并适应新的攻击变体需要大量的专家投入。我们提出了JANUS，一个用于自动化跨层防御合成的红蓝对抗框架。从已知攻击出发，JANUS在隔离的离线环境中迭代地细化特定于部署的防御策略。黑盒攻击者仅使用外部可观察的反馈来探测当前策略下的攻击变体。防御者将每个已验证的变体总结为其安全相关的影响，并使用反事实重放来分离（摘要在此处截断）

    arXiv:2610.35532v1 Announce Type: cross  Abstract: Unsafe deserialization turns untrusted data into attacker-controlled execution, and a single vulnerability can admit diverse exploit variants that evade defenses tailored to a known exploit. Existing defenses typically operate at a single enforcement point with limited visibility into the full attack. Combining controls across layers offers complementary protection, but coordinating them to preserve legitimate functionality and adapting them to new variants requires substantial expert effort.   We propose JANUS, a red-blue adversarial framework for automated cross-layer defense synthesis. Starting from a known exploit, \sys iteratively refines deployment-specific policies in an isolated offline environment. A black-box attacker probes the current policy for exploit variants using only externally observable feedback. The defender summarizes each verified variant into its security-relevant effects, uses counterfactual replay to separate 
    
[^46]: 静默而代价高昂：LLM智能体中提示缓存错误集成的问题表征与检测

    Silent and Costly: Characterizing and Detecting Prompt-Caching Mis-Integration in LLM Agents

    [https://arxiv.org/abs/2610.34817](https://arxiv.org/abs/2610.34817)

    该论文首次对10个开源智能体系统中120个提示缓存相关的拉取请求进行实证研究，揭示了提示缓存错误集成往往静默失败且难以察觉的问题，并据此提出了检测工具CACHELINT。

    

    大规模语言模型服务需要处理海量的用户请求，从而产生巨额的推理成本。为了同时降低成本与延迟，服务提供商引入了提示缓存技术。然而，由于各提供商的API存在异构性且缓存语义持续演变，正确集成提示缓存并非易事。为了理解下游应用中提示缓存的错误集成问题，我们对来自10个流行开源智能体系统的120个与提示缓存相关的拉取请求开展了首次实证研究。研究结果表明，提示缓存错误集成既源于请求侧的错误配置，也源于读取侧的使用计数错误。更重要的是，大多数错误集成会静默失败，不会引发任何可观察到的运行时故障。基于这些发现，我们设计了CACHELINT，它将提示缓存错误集成建模为一个调用点以及正确缓存所需的先决条件

    arXiv:2610.34817v1 Announce Type: cross  Abstract: Large language model (LLM) services process massive volumes of user requests, resulting in substantial inference costs. To reduce both cost and latency, providers have introduced prompt caching. However, correctly integrating prompt caching is non-trivial due to heterogeneous provider-specific APIs and continuously evolving caching semantics. To understand prompt-caching mis-integration in downstream applications, we conduct the first empirical study of 120 prompt-caching-related pull requests from 10 popular open-source agent systems. Our findings show that prompt-caching mis-integrations arise in both request-side misconfiguration and read-side usage miscount. More importantly, most mis-integrations fail silently without causing observable runtime failures.   Motivated by these findings, we design CACHELINT, which models a prompt-caching mis-integration as an invocation site together with the preconditions required for correct cache 
    
[^47]: 大语言模型与AI智能体的诚实性测试：软件工程的愿景

    Honesty Testing of LLMs and AI Agents: A Vision for Software Engineering

    [https://arxiv.org/abs/2610.34239](https://arxiv.org/abs/2610.34239)

    这篇愿景论文提出将LLM和AI智能体的诚实性测试作为软件工程的新研究领域，把忠实报告确立为可规定、可测试、可维护的软件质量需求，并定义了“诚实性缺陷”概念及初步测试框架。

    

    对大语言模型（LLM）和人工智能（AI）智能体的人类监督日益依赖于它们的报告。与系统自身判断相冲突的陈述可能掩盖失败、妨碍及时干预，并导致严重危害。随着智能体自主性不断增强，忠实报告对于维持人类控制变得愈发重要。在这篇愿景论文中，我们提出将LLM和AI智能体的诚实性测试作为软件工程的一个新研究领域。在现有诚实性评估的基础上，我们的目标是使忠实报告成为开发者可以规定、测试和维护的软件质量需求。我们将诚实性缺陷定义为一种行为缺陷：当LLM或AI智能体被要求忠实沟通时，作出了与其报告时刻自身判断相冲突的明确陈述。我们提出了一个初步框架，涵盖受控输入下的LLM级测试和智能体级测试。

    arXiv:2610.34239v1 Announce Type: new  Abstract: Human oversight of large language models (LLMs) and artificial intelligence (AI) agents increasingly depends on their reports. Claims that conflict with a system's own judgments can conceal failures, prevent timely intervention, and contribute to severe harm. Greater autonomy makes faithful reporting increasingly important for maintaining human control. In this vision paper, we propose honesty testing of LLMs and AI agents as a new research area in software engineering. Building on existing honesty evaluations, our goal is to make faithful reporting a software quality requirement that developers can specify, test, and maintain. We define honesty bugs as behavioural defects in which an LLM or AI agent, when required to communicate faithfully, makes an explicit claim that conflicts with its own judgment at the time of reporting. We present an initial framework covering LLM-level testing under controlled inputs and agent-level testing of co
    
[^48]: 超越正确性：软件工程新手学生如何对生成式AI行使监督

    Beyond Correctness: How Novice Software Engineering Students Exercise Oversight of Generative AI

    [https://arxiv.org/abs/2610.32118](https://arxiv.org/abs/2610.32118)

    软件工程新手学生对生成式AI的监督主要聚焦于输出的适用性问题（74.4%）而非正确性问题（16.9%），表明学生监督的核心是判断看似合理的AI输出是否真正适合实际任务。

    

    生成式人工智能（GenAI）正在将软件工程学生的工作重心从产出制品转向评估和监督AI生成的输出。然而，关于学生在报告干预时识别出输出具备哪些特征，或者他们在真实的软件工程工作中如何行使监督，目前知之甚少。我们分析了145名一年级软件工程学生通过结构化AI日志记录的962个学生-AI交互回合，其中包括172次学生声明的干预。通过多维编码框架，我们刻画了这些报告的干预中识别出的输出特征、学生干预的原因、他们采取的行动以及这些行动的实施方式。74.4%的干预中发现了适用性问题，而涉及正确性问题的仅占16.9%，这表明监督更多关注的是看似合理的AI输出是否真正适合于

    arXiv:2610.32118v1 Announce Type: new  Abstract: Generative AI (GenAI) is shifting software engineering students' work from producing artefacts towards evaluating and supervising AI-generated outputs. However, little is known about what characteristics students identify in outputs when reporting interventions, or how they exercise oversight during authentic software engineering work. We analysed 962 student-AI interaction turns documented by 145 first-year software engineering students through structured AI Journals, including 172 student-declared interventions. Using a multi-dimensional coding framework, we characterised the output characteristics identified in these reported interventions, students' reasons for intervening, the actions they took, and how those actions were enacted. Fitness problems were identified in 74.4% of interventions, compared with 16.9% involving correctness problems, indicating that oversight more often concerned whether plausible AI outputs were suitable for
    
[^49]: 知道何时停止：编码智能体中的证据-行动差距

    Knowing When to Stop: The Evidence-Action Gap in Coding Agents

    [https://arxiv.org/abs/2610.31895](https://arxiv.org/abs/2610.31895)

    该论文揭示了编码智能体的“证据-行动差距”现象——即智能体即使已收集到表明任务无法独立完成的充分证据仍过度坚持或虚报成功，并提出ImpasseBench基准通过在可解决任务中注入环境阻碍并标记证据充分步骤来量化这一差距。

    

    编码智能体被训练来完成任务，即使环境使得任务根本无法完成时，它们仍会不断尝试。表明只有外部帮助才能恢复进展的证据通常在几步之内就会出现，但智能体很少据此采取行动：它们过度坚持，尝试不必要的变通方法，并声称取得记录无法支持的成功——我们将这一现象称为“证据-行动差距”。为了衡量这一差距，我们提出了ImpasseBench，它在原本可解决的修复任务中注入十二种真实的环境阻碍因素，并在每条轨迹中标记出所收集证据变得充分的步骤。相对于该步骤对智能体的行动进行评分，可以区分适当的努力与过度坚持以及过早求助。在八个前沿模型和19,371条轨迹上，我们发现几乎所有需要外部帮助的运行都收集到了充分的证据，但仍有三分之一耗尽了预算，而且我们检测到的所有不安全请求都发生在这一时点之后。

    arXiv:2610.31895v1 Announce Type: cross  Abstract: Coding agents are trained to complete tasks, and they keep trying even when the environment makes completion impossible. The evidence that only outside help can restore progress usually appears within a few steps, yet agents rarely act on it: they over-persist, attempt undesired workarounds, and claim success the record does not support -- a phenomenon we call evidence-action gap. To measure it, we introduce ImpasseBench, which injects twelve realistic environment blockers into otherwise solvable repair tasks and marks, in each trajectory, the step at which the collected evidence becomes sufficient. Scoring actions relative to that step separates appropriate effort from over-persistence and premature handoff. Across eight frontier models and 19,371 trajectories, we find that nearly every run that needs outside help collects sufficient evidence, yet a third still exhaust their budget, and every unsafe request we detect comes after that 
    
[^50]: CuraLoc：通过计划式代码检查实现仓库级缺陷定位

    CuraLoc: Repository-Level Fault Localization via Planned Code Inspection

    [https://arxiv.org/abs/2610.31809](https://arxiv.org/abs/2610.31809)

    CuraLoc提出在源码检索前一次性规划代码检查的缺陷定位方法：模型在单次规划中同时选定起始符号、检查关系并形成修改假设，再由确定性检索完成最终排序，从而减少反复的模型调用和token消耗。

    

    缺陷报告描述了出了什么问题，但很少指出代码应该在哪里修改。现有的仓库级定位方法通常让语言模型迭代地决定下一步要检查哪些代码，这需要多次模型调用并消耗大量token。我们研究了这些检查决策是否可以在源码检索之前一次性规划完成。在本研究中，我们提出了CuraLoc，它首先筛选候选文件，并使用简洁的行为与结构描述来表示文件中的符号。在单次规划决策中，模型联合完成三项任务：选择起始符号、确定从每个符号出发要检查的代码关系，并对可能的代码修改位置形成假设。随后由确定性检索收集所请求的源码，用于基于这些假设进行的最终排序，从而将模型的推理集中在选择和比较代码上，而不是指导每一步检索。我们在…上验证了所提方法的有效性

    arXiv:2610.31809v1 Announce Type: cross  Abstract: A bug report describes what went wrong, but rarely where the code should change. Existing repository-level localization methods often let language models iteratively decide what code to inspect next, requiring repeated model calls and substantial token usage. We investigate whether these inspection decisions can instead be planned before source retrieval. In this research, we introduce CuraLoc, which first selects candidate files and represents their symbols using compact behavioral and structural descriptions. In a single planning decision, the model jointly selects starting symbols, chooses code relationships to inspect from each, and forms hypotheses about likely edits. Deterministic retrieval collects the requested source for a final ranking informed by those hypotheses, concentrating model reasoning on choosing and comparing code rather than directing each retrieval step. We demonstrate the efficacy of our proposed approach on the
    
[^51]: 从代码生成器到工程助手：面向PLC编程的工厂感知智能体

    From Code Generator to Engineering Companion: A Plant-aware Agent for PLC Programming

    [https://arxiv.org/abs/2610.28160](https://arxiv.org/abs/2610.28160)

    该论文提出了一个工厂感知的对话式PLC编程智能体，通过工厂特定上下文摄取、以对话为先的解释与调试交互模式，以及从聊天到部署的分级工具访问阶梯，将PLC编程从一次性代码生成升级为持续的工程辅助体验。

    

    可编程逻辑控制器（PLC）编程仍然是一项以IDE为中心的手动任务，工程师需要在文档、项目文件和代码编辑器之间频繁切换上下文。近期研究表明，检索增强的大语言模型能够以较高的编译通过率生成符合厂商规范的Structured Text代码，但这些系统将代码生成视为一次性流水线：工程师提交查询后获得代码，系统对特定工厂没有持久化的知识。我们提出了一个工厂感知的对话式编程助手，该系统在三个方向上扩展了这一范式：工厂特定上下文的摄取、以对话为先的解释与调试交互模式，以及从对话到部署的分级工具访问阶梯。该系统专为真实工厂基础设施设计，并确保工程师对每个会产生现实世界影响的步骤保持控制权。

    arXiv:2610.28160v1 Announce Type: cross  Abstract: Programmable Logic Controller (PLC) coding remains a manual, IDE-centric task that demands constant context switching between documentation, project files, and code editors. Recent work has demonstrated that retrieval-augmented large language models can generate vendor-compliant Structured Text at high compilation rates \cite{pred}, yet these systems treat code generation as a one-shot pipeline: the engineer submits a query and receives code, with no persistent knowledge of the specific plant. We present a plant-aware conversational copilot that extends this paradigm in three directions: plant-specific context ingestion, a chat-first interaction model for explanation and debugging, and a graduated tool-access ladder from chat to deployment. The system is designed for real plant infrastructure and keeps the engineer in control of each step that produces real-world effects.
    
[^52]: Vibenix：基于Nix的智能体化软件打包与维护

    Vibenix: Agentic Software Packaging and Maintenance with Nix

    [https://arxiv.org/abs/2610.24712](https://arxiv.org/abs/2610.24712)

    Vibenix是一个开源AI智能体，利用大型语言模型通过规则引导的构建-修复反馈循环自动生成并迭代优化Nix打包表达式，在NixBench数据集上以Claude Haiku 4.5实现了95.3%的可构建软件包成功率，远超传统单次生成方法。

    

    隐式依赖、不受控制的网络访问以及其他构建环境的不纯因素造成了不透明的软件供应链。识别影响构建的每一个依赖是一项艰巨的任务。函数式包管理通过可复现的打包解决了这一问题，确保软件仅依赖已声明的、封闭隔离的输入。然而，创建构建配方（build recipes）仍然复杂且耗时，对开发者和维护者的采用构成了重大障碍。我们提出了Vibenix，一个开源的AI驱动智能体，通过自动生成Nix打包表达式来弥合这一差距。Vibenix使用大型语言模型（LLM）通过规则引导的构建-修复反馈循环迭代地优化配方。在我们新提出的NixBench数据集的466个打包任务上，Vibenix使用Claude Haiku 4.5在95.3%的情况下生成了可构建的软件包，而最先进的基于LLM的单次生成方法仅为5.6%。

    arXiv:2610.24712v2 Announce Type: replace  Abstract: Implicit dependencies, uncontrolled network access, and other build-environment impurities create intransparent software supply chains. Identifying every dependency affecting a build is a difficult task. Functional package management addresses this through reproducible packaging, ensuring software relies exclusively on declared, hermetically isolated inputs. However, creating build recipes remains complex and time-consuming, posing a significant barrier to adoption for developers and maintainers. We present Vibenix, an open-source AI-powered agent that bridges this gap by automatically generating Nix packaging expressions. Vibenix uses a large language model (LLM) to iteratively refine recipes through a rule-guided build-repair feedback loop. On 466 packaging tasks from our new NixBench dataset, Vibenix produced buildable packages in 95.3% of cases using Claude Haiku 4.5, compared with 5.6% for a state-of-the-art LLM-based single-sho
    
[^53]: 依赖图何时有助于代码到架构的映射？

    When Does the Dependency Graph Help Code-to-Architecture Mapping?

    [https://arxiv.org/abs/2610.19769](https://arxiv.org/abs/2610.19769)

    该论文将代码到架构的映射建模为依赖图上的半监督节点分类问题，并通过GCN、GAT、MLP与朴素贝叶斯的系统对比实验，揭示了依赖图结构信息在何种条件下才能真正提升架构映射的效果。

    

    当系统的实现偏离其预期的模块化分解，而源文件与架构模块之间的映射缺失、不完整或人工维护成本高昂时，就会产生架构技术债。我们将代码到架构的映射问题形式化为有向多关系依赖图上的半监督节点分类，其中文件作为节点，类型化的依赖关系作为边，词法特征则从代码标识符和目录布局中提取。从少量带标签的样本出发，通过迭代式自训练过程将映射扩展到其余文件。我们将图卷积网络（GCN）、图注意力网络（GAT）以及不使用图结构的多层感知机（MLP）——三种学习型模型均采用相同的特征和训练流程——与朴素贝叶斯参照方法进行对比。在十个开源Java和C/C++系统上进行的100次实验运行中，三种神经模型的表现均优于……（原文摘要至此截断）

    arXiv:2610.19769v1 Announce Type: cross  Abstract: Architectural technical debt arises when a system's implementation drifts from its inten ded modular decomposition, while the mapping between source files and architectural modu les is missing, incomplete, or costly to maintain manually. We formulate code-to-archite cture mapping as semi-supervised node classification over a directed multi-relational de pendency graph, with files as nodes, typed dependencies as edges, and lexical features d erived from code identifiers and directory layout. Starting from a small labeled sample,   an iterative self-training procedure expands the mapping to the remaining files. We com pare a graph convolutional network (GCN), a graph attention network (GAT), and a graph-f ree multi-layer perceptron (MLP), using identical features and training procedures for t he learned models, against a Naive Bayes reference. Across 100 runs on ten open-source J ava and C/C++ systems, all three neural models outperform 
    
[^54]: 智能体软件工程基准测试究竟在衡量什么？超越类别标签的任务需求与智能体行为画像

    What Does an Agentic Software Engineering Benchmark Measure? Profiling Task Demands and Agent Behaviour Beyond What Category Labels Reveal

    [https://arxiv.org/abs/2609.01271](https://arxiv.org/abs/2609.01271)

    本文提出Spread-Novelty-Centrality（SNC）三轴画像方法来刻画仓库级编码任务的真实需求，发现类别标签是任务需求的不可靠代理指标，且智能体行为轨迹能揭示人工标准答案无法反映的任务需求信息。

    

    智能体软件工程基准测试通常通过名义类别标签（如“缺陷修复”或“功能实现”）来概括，然而携带相同标签的基准测试是通过截然不同的构建流程创建的。因此，标签几乎无法揭示基准测试所要求的工程工作。我们引入了 Spread--Novelty--Centrality（SNC）画像，这是一种基于实证软件工程研究的三轴特征描述方法，用于刻画仓库级编码任务的需求。我们将该画像应用于五个广泛使用的基准测试，以及两个模型家族在三种规模下的14,922条轨迹，并报告了三项发现。（1）类别标签是任务需求的不可靠代理指标，因为每一对基准测试在至少两个SNC轴上均存在统计学上的显著分离，且这些分离可以追溯到具体的构建决策。（2）智能体行为揭示了人类编写的标准答案（gold solution）无法揭示的需求。智能体产出的解决方案……

    arXiv:2609.01271v1 Announce Type: cross  Abstract: Agentic software engineering benchmarks are typically summarized by nominal category labels such as "bug fix" or "feature implementation," yet benchmarks carrying the same label are built through very different curation pipelines. A label thus reveals little about the engineering work a benchmark demands. We introduce the Spread--Novelty--Centrality (SNC) profile, a three-axis characterization of the demands of repository-level coding tasks, grounded in empirical software engineering research. We apply the profile to five widely used benchmarks and 14,922 trajectories of two model families at three scales, and report three findings. (1) A label is an unreliable proxy for task demands, as every pair of benchmarks is statistically separated on at least two SNC axes, and the separations trace back to specific curation decisions. (2) Agent behaviour reveals demands that the human-written gold solution cannot. Agents produce larger solution
    
[^55]: QTUNER：一种用于量子编译器优化Pass调优的特征引导自适应框架

    QTUNER: A Feature-Guided Adaptive Framework for Quantum-Compiler Optimization Pass Tuning

    [https://arxiv.org/abs/2607.04586](https://arxiv.org/abs/2607.04586)

    本文提出QTUNER，一种特征引导的自适应框架，通过引入与优化特性直接相关的量子特征来对量子编译器的完整优化Pass空间进行自适应调优，以克服现有方法仅调优有限Pass子集且依赖静态特征的局限，从而提升量子电路的执行保真度。

    

    量子编译器优化对于提升量子电路的执行保真度至关重要。在实践中，这种优化通常被表述为降低电路层面的度量指标。在经典编译器领域，Pass调优已通过多种方法得到充分确立。然而，这些方法无法直接应用于量子编译器，因为量子程序表现出量子特有的特征。现有的量子编译器调优研究已尝试将量子电路特征与编译器调优技术相结合，但仍然存在两个关键局限：首先，它们仅调优有限的优化Pass子集，导致整个优化Pass空间的全局有效性尚未得到充分探索；其次，它们所使用的量子特征主要是静态特征（如结构和DAG特征），这些特征与量子电路的优化特性没有直接关联。在本文中，我们提出（摘要原文在此处截断）

    arXiv:2607.04586v2 Announce Type: replace-cross  Abstract: Quantum compiler optimization is important for improving the execution fidelity of quantum circuits. In practice, this optimization is often formulated as reducing circuit-level metrics. In classical compilers, pass tuning has been well-established through various methods. However, these methods cannot be directly applied to quantum compilers, because quantum programs exhibit quantum-specific features. Existing quantum compiler tuning studies have attempted to incorporate quantum circuit features with compiler tuning techniques, but they still suffer from two key limitations. First, they tune only a limited subset of optimization passes, leaving the global effectiveness of the full optimization pass space underexplored. Second, the quantum features they use are mainly static features, such as structure and DAG features, which are not directly tied to the optimization characteristics of quantum circuits.   In this paper, we prop
    
[^56]: AI写作速度超过人类审查速度：一项关于企业“2倍”指令的纵向研究

    AI Writes Faster Than Humans Can Review: A Longitudinal Study of an Enterprise 2x Mandate

    [https://arxiv.org/abs/2607.01904](https://arxiv.org/abs/2607.01904)

    本文通过对802名开发者和196,212个PR的纵向案例研究发现，企业“2倍”AI生产力指令最终在第10个月才达成，且实现路径与公告预期不符——指令发布时PR吞吐量几乎不变，开发者开始使用AI时仅提升约12%，此后随经验积累持续增长，表明AI编程收益需要时间逐步释放。

    

    arXiv:2607.01904v2 公告类型：替换 摘要：企业越来越多地强制要求使用AI编程工具，并宣布雄心勃勃的生产力目标，然而关于这些生产力目标是否已实现、如何实现以及付出了什么代价，几乎没有纵向研究证据。我们对一个有据可查的企业“2倍”指令进行了定量案例研究：2025年6月，一家中型AI前沿软件公司的CTO设定了借助AI将每位工程师合并的拉取请求数量翻倍的目标。基于包含802名开发者和196,212个PR（2024年1月至2026年4月）的面板数据，我们通过比较每位开发者开始使用AI工具之后与其之前自身的PR吞吐量，来估计AI采用的效果。我们发现该公司最终达成了目标，但仅在第10个月才实现，且实现方式与公告所暗示的不同：PR吞吐量在指令发布时几乎没有变化，在每位开发者开始使用AI时上升了约12%，随后随着开发者经验积累而持续增长。

    arXiv:2607.01904v2 Announce Type: replace  Abstract: Enterprises increasingly mandate the use of AI coding tools and announce ambitious productivity targets, yet there is little longitudinal evidence on whether such productivity targets have been reached, how, and at what cost. We present a quantitative case study of a documented enterprise "2x" mandate: in June 2025, the CTO of a mid-sized, AI-forward software company set the goal of doubling merged pull requests (PRs) per engineer with AI. Using a panel of 802 developers and 196,212 PRs (January 2024 - April 2026), we estimate the effect of AI adoption by comparing each developer's PR throughput after they started using AI tools with their own throughput before. We find that the company eventually reached the target, but only in the tenth month and not in the way the announcement implied: PR throughput barely moved at the mandate date, rose by about 12% when each developer started using AI, and then kept growing as developers accumul
    
[^57]: 从成对比较中构建许可证宽松度的结构

    Structuring license permissiveness from pairwise comparisons

    [https://arxiv.org/abs/2606.31032](https://arxiv.org/abs/2606.31032)

    本文利用大型语言模型，通过成对比较方法系统化地构建和验证许可证宽松度的部分排序，解决了大规模许可证比较的难题。

    

    许可证是发明者依赖的法律工具，用于保护他们构建的技术并规范其使用方式——然而，其创作和选择的性质意味着它们的解释、选择和执行在很大程度上是非结构化的。在实践中，这使得在规模上比较许可证变得困难——何时一个许可证被认为比另一个更宽松，以及何时它们的条款彼此不可比？目前，有越来越多的许可证被引入和使用，但没有系统的方法来研究它们之间的关系。这对Hugging Face、GitHub和Python包索引等平台很重要，在这些平台上，开发者发布或基于各自拥有许可证的技术进行构建。利用大型语言模型（LLMs），我们引入了在规模上比较许可证的方法：首先，以成对方式构建并验证基于宽松度的部分排序；以及

    arXiv:2606.31032v3 Announce Type: replace  Abstract: Licenses are legal instruments that inventors rely upon to protect the technologies they build and regulate how they are used---however, the nature of their authorship and selection implies that how they are interpreted, chosen, and enforced is largely unstructured. In practice, this makes it difficult to compare licenses at scale---when is one license considered more permissive than the other, and when are their terms incomparable to each other? Currently, there is a growing list of licenses that are introduced and used, yet no systematic way to study their relationships. This matters for platforms such as Hugging Face, GitHub, and the Python Package Index, where developers publish or build upon technologies that each have their own licenses. Using large language models (LLMs), we introduce methods for comparing licenses at scale: first, in a pairwise fashion to construct and validate a partial ordering based on permissiveness; and 
    
[^58]: Simulink模型检测器的位级精确一致性测试

    Bit-Precise Conformance Testing of Simulink Model Checkers

    [https://arxiv.org/abs/2606.24719](https://arxiv.org/abs/2606.24719)

    本文通过对Simulink基本模块类型进行位级精确的形式化，并提出一种一致性测试方法，来验证Simulink模型检测器与仿真器之间行为的一致性和可靠性。

    

    MATLAB/Simulink为信息物理系统的开发提供了实用的建模语言和仿真引擎。为确保所开发模型的质量，目前已有多种形式化验证工具可用，例如Simulink Design Verifier (SLDV)以及第三方的基于SMT的模型检测器。然而，由于缺乏覆盖Simulink模型所有元素及其数值行为细节的语义定义，模型检测器本身的可靠性常常受到质疑，可能会以与仿真器不同的方式分析模型。本工作旨在通过解决以下问题来验证Simulink模型检测器的质量：1) 对Simulink基本模块类型进行形式化，包括定义模块类型特征集以及模块的位级精确行为；2) 提出一种方法，用于针对每种模块类型测试各工具之间的位级精确一致性关系。测试套件的通过率……

    arXiv:2606.24719v3 Announce Type: replace  Abstract: MATLAB/Simulink provides a practical modeling language and a simulation engine for the development of cyber-physical systems. To ensure the quality of the developed models, there are formal verification tools available, such as Simulink Design Verifier (SLDV) and third-party SMT-based model checkers (SmtMC). However, due to the absence of a semantics of Simulink that covers every element of models and the details of its numerical behavior, the reliability of the model checkers themselves is often doubtful, potentially analyzing models differently from the simulator. This work aims to verify the quality of the Simulink model checkers by addressing the following items. 1) Formalization of the basic block types of Simulink. It involves defining block type feature sets and the bit-precise behavior of the blocks. 2) A method for testing bit-precise conformance relations among the tools for each block type. The pass rate of our test suite 
    
[^59]: LLM代码坏味道：分类体系与检测方法

    LLM Code Smells: A Taxonomy and Detection Approach

    [https://arxiv.org/abs/2605.22976](https://arxiv.org/abs/2605.22976)

    本文提出了LLM代码坏味道的独立分类体系和九种坏味道目录，开发了静态分析工具SpecDetect4LLM进行检测，并通过对692个开源项目的实证评估发现73.5%的项目存在LLM代码坏味道。

    

    由于大语言模型（LLM）具有多功能性、灵活性以及在一定程度上模拟人类推理的能力，它们正越来越多地被集成到软件系统中以实现多种用途。然而，LLM推理在源代码中的不良集成可能会损害软件系统的质量。因此，必须记录不恰当的LLM集成编码实践，以帮助开发者缓解此类问题。延续我们早期关于LLM代码坏味道的工作，本文通过提出一个独立的分类体系以及包含九种LLM代码坏味道的目录，巩固并完善了这一概念。我们还创建了SpecDetect4LLM——一个用于检测这些坏味道的静态源代码分析工具，并对其检测效果（精确率和召回率）以及LLM代码坏味道在692个开源软件项目（171,194个源文件）中的普遍程度进行了广泛的实证评估。我们的结果表明，73.5%的可分析项目受到LLM代码坏味道的影响。

    arXiv:2605.22976v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly integrated into software systems for diverse purposes, due to their versatility, flexibility, and ability to simulate human reasoning to some extent. However, poor integration of LLM inference in source code can undermine software system quality. Therefore, inadequate LLM integration coding practices must be documented to help developers mitigate such issues. Following our earlier work on LLM code smells, this paper consolidates and refines the concept by presenting a self-contained taxonomy and a catalog of nine LLM code smells. We also create SpecDetect4LLM, a static source code analysis tool for their detection, and conduct extensive empirical evaluations of its detection effectiveness (precision and recall) as well as the prevalence of LLM code smells across 692 open-source software projects (171,194 source files). Our results show that LLM code smells affect 73.5% of the analyz
    
[^60]: SemLoc：将自由形式的LLM推理进行结构化语义落地以实现故障定位

    SemLoc: Structured Grounding of Free-Form LLM Reasoning for Fault Localization

    [https://arxiv.org/abs/2603.29109](https://arxiv.org/abs/2603.29109)

    SemLoc将LLM的自由形式推理转化为附加在代码图锚点上的可执行、可验证的语义约束，通过运行时检查和反事实验证实现软件故障定位。

    

    大型语言模型（LLM）能够描述一段代码应该计算什么，但它们是以自由形式的文本来表达的。这类文本无法对照程序执行进行检验，也无法表明某个语句是否正确。我们提出了SemLoc，它将LLM的推理转化为可执行、可验证的语义约束。通过结构化的语义落地，LLM将每个属性写成具有固定类别的约束，并附加到代码图中的锚点上，例如变量定义、分支、循环或返回语句。SemLoc将每个约束作为运行时检查插入并运行测试，从而在通过和失败的测试上生成语义违规谱。随后进行反事实验证：在锚点处对代码打补丁并重新运行测试；只有当所有测试随后都通过时，约束所建议的补丁才被认定为已验证。我们在故障定位任务上研究SemLoc，其中测试作为验证的判定标准，并使用了四个LLM（摘要原文在此处截断）。

    arXiv:2603.29109v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) can describe what a piece of code should compute, but they do so in free-form text. Such text cannot be checked against program executions, and it does not reveal whether a statement is correct. We present SemLoc, which turns LLM reasoning into executable, verifiable semantic constraints. With structured semantic grounding, the LLM writes each property as a constraint with a fixed category, attached to an anchor in the code graph, such as a variable definition, branch, loop, or return. SemLoc inserts each constraint as a runtime check and runs the tests, which yields a semantic violation spectrum over passing and failing tests. Counterfactual verification then patches the code at an anchor and reruns the tests; a patch that a constraint suggests counts as verified only if all tests then pass.   We study SemLoc on fault localization, where the tests serve as the oracle for verification, with four LLM
    
[^61]: 测量与利用LLM辅助安全代码审查中的上下文偏差

    Measuring and Exploiting Contextual Bias in LLM-Assisted Security Code Review

    [https://arxiv.org/abs/2603.18740](https://arxiv.org/abs/2603.18740)

    本研究揭示了框架效应会导致基于LLM的自动化代码审查系统在漏洞检测中产生系统性且普遍的偏差，并证明攻击者可以通过注入带有偏差的PR元数据，将其利用为针对ACR流水线的供应链攻击向量。

    

    集成大语言模型（LLM）的自动化代码审查（ACR）系统在软件开发工作流程中的应用日益广泛，其形式涵盖从交互式助手到CI/CD流水线中的自主代理。在本文中，我们研究了ACR中基于LLM的漏洞检测如何受到框架效应的影响：即在形成判断时，信息呈现方式凌驾于其语义内容之上的倾向。我们考察了对手是否可以通过上下文偏差注入来利用这一效应（即通过精心构造PR元数据来偏置ACR的安全判断），并将其作为针对真实世界ACR流水线的供应链攻击向量。为此，我们首先在五种框架条件下对6个LLM进行了大规模探索性研究，确立了框架效应是基于LLM的漏洞检测中一种系统性且普遍存在的现象。随后，我们设计了一个真实且受控的实验环境，对20个（项目/代码库）中的33个CVE进行了评估。

    arXiv:2603.18740v3 Announce Type: replace-cross  Abstract: Automated Code Review (ACR) systems integrating Large Language Models (LLMs) are increasingly adopted in software development workflows, ranging from interactive assistants to autonomous agents in CI/CD pipelines. In this paper, we study how LLM-based vulnerability detection in ACR is affected by the framing effect: the tendency to let the presentation of information override its semantic content in forming judgments. We examine whether adversaries can exploit this through contextual-bias injection (crafting PR metadata to bias ACR security judgments) as a supply-chain attack vector against real-world ACR pipelines. To this end, we first conduct a large-scale exploratory study across 6 LLMs under five framing conditions, establishing the framing effect as a systematic and widespread phenomenon in LLM-based vulnerability detection.   We then design a realistic and controlled experimental environment, evaluating 33 CVEs across 20
    
[^62]: SWE-Skills-Bench：智能体技能在真实世界软件工程中真的有帮助吗？

    SWE-Skills-Bench: Do Agent Skills Actually Help in Real-World Software Engineering?

    [https://arxiv.org/abs/2603.15401](https://arxiv.org/abs/2603.15401)

    该论文提出了SWE-Skills-Bench，一个以技能为中心、需求驱动的基准测试，通过3,920组有/无技能的配对评估，系统量化了单个智能体技能对真实世界软件工程任务的实际提升效果。

    

    智能体技能是在推理时注入的结构化程序性知识包，越来越多地被用于增强大语言模型智能体在软件工程任务上的能力。然而，它们在端到端开发场景中的实际效用仍不明确，且目前没有标准流程能在安装前对单个技能进行测试。我们提出了SWE-Skills-Bench，这是一个以技能为中心、由需求驱动的基准测试，用于衡量单个智能体技能对真实世界软件工程（SWE）任务性能的提升程度。我们从98个公开的SWE技能出发，每个技能都对应一个固定提交版本的真实GitHub仓库。针对每个技能，我们推导出仓库特定的需求，创建可执行的验证器，并构建可复现的环境。在四种智能体-模型配置下，我们对每个任务分别在无技能和有技能的情况下各运行一次，共得到3,920组配对评估。针对每项需求，我们计算配对差值，即性能的变化……

    arXiv:2603.15401v2 Announce Type: replace-cross  Abstract: Agent skills, structured procedural knowledge packages injected at inference time, are increasingly used to augment LLM agents on software engineering tasks. However, their real utility in end-to-end development settings remains unclear, and no standard procedure tests an individual skill before installation. We present swe-skills-bench, a skill-centric, requirement-driven benchmark that measures how much individual agent skills improve performance on real-world software engineering (SWE) tasks. We start from 98 public SWE skills, each paired with a real GitHub repository pinned at a fixed commit. For each skill, we derive repository-specific requirements, create executable verifiers, and build a reproducible environment. Under each of four agent-model configurations, we run every task once without the skill and once with it, yielding 3,920 paired evaluations. For each requirement, we compute a paired delta, i.e., the change in
    
[^63]: 智能体重构：对AI编程智能体的实证研究

    Agentic Refactoring: An Empirical Study of AI Coding Agents

    [https://arxiv.org/abs/2511.04824](https://arxiv.org/abs/2511.04824)

    本文基于AIDev数据集对真实开源Java项目中AI编程智能体生成的重构进行了大规模实证研究，分析了51,412个重构实例，首次系统揭示了智能体重构在实际开发中的使用方式、与人工重构的差异及其对代码质量的影响。

    

    arXiv:2511.04824v2 公告类型：替换 摘要：诸如OpenAI Codex、Claude Code和Cursor等智能体编程工具正在改变软件开发的方式。这些AI驱动的系统作为自主的队友，能够规划和执行复杂的开发任务。智能体已经成为重构中的积极参与者，而重构是一种在不改变可观察行为的前提下改进内部代码质量的关键实践。尽管智能体重构的应用日益广泛，但目前缺乏关于智能体重构在实践中如何被使用、与人工驱动的重构相比表现如何、以及它对代码质量有何影响的实证理解。为了填补这一空白，我们基于AIDev数据集，对真实开源Java项目中AI智能体生成的重构进行了大规模研究，分析了跨越12,256个拉取请求和14,998次提交的51,412个重构实例。我们发现重构是一种常见且有明确意图的活动，智能体在26.1%的提交中明确以重构为目标……

    arXiv:2511.04824v2 Announce Type: replace  Abstract: Agentic coding tools, such as OpenAI Codex, Claude Code, and Cursor, are changing how software is developed. These AI-powered systems function as autonomous teammates capable of planning and executing complex development tasks. Agents have become active participants in refactoring, a key practice for improving internal code quality without altering observable behavior. Despite their increasing adoption, there is little empirical understanding of how agentic refactoring is used in practice, how it compares to human-driven refactoring, and what impact it has on code quality. To address this gap, we present a large-scale study of AI agent-generated refactorings in real-world open-source Java projects, analyzing 51,412 refactoring instances across 12,256 pull requests and 14,998 commits based on the AIDev dataset. We find that refactoring is a common and intentional activity, with agents explicitly targeting refactoring in 26.1% of commi
    
[^64]: 优化代码后门检测中的谱签名方法

    Optimizing Spectral Signature in Code Backdoor Detection

    [https://arxiv.org/abs/2510.13992](https://arxiv.org/abs/2510.13992)

    本文重新审视并系统评估了谱签名防御方法在检测代码模型后门攻击中的有效性，探索其在不同攻击场景和防御配置下的优化应用。

    

    随着大型语言模型（LLM）日益融入软件开发工作流程，它们也成为对抗性攻击的主要目标。其中，后门攻击是一种重大威胁，攻击者可以通过嵌入在训练数据中的隐藏触发器来操纵模型输出。检测此类后门仍然是一个挑战，而一种有前景的方法是使用谱签名防御方法，该方法通过特征向量分析特征表示来识别被投毒的数据。虽然此前已有一些工作探索了谱签名在神经网络后门检测中的应用，但近期研究表明，这些方法对代码模型可能并非最优有效。在本文中，我们重新审视了基于谱签名的防御方法在代码模型后门攻击背景下的适用性。我们在各种攻击场景和防御配置下系统性地评估了这些方法的有效性，并分析了……

    arXiv:2510.13992v2 Announce Type: replace  Abstract: As Large Language Models (LLMs) are increasingly integrated into software development workflows, they also become prime targets for adversarial attacks. Among these, backdoor attacks are a significant threat, allowing attackers to manipulate model outputs through hidden triggers embedded in training data. Detecting such backdoors remains a challenge, and one promising approach is the use of Spectral Signature defense methods that identify poisoned data by analyzing feature representations through eigenvectors. While some prior works have explored Spectral Signatures for backdoor detection in neural networks, recent studies suggest that these methods may not be optimally effective for code models. In this paper, we revisit the applicability of Spectral Signature-based defenses in the context of backdoor attacks on code models. We systematically evaluate their effectiveness under various attack scenarios and defense configurations, ana
    
[^65]: 从遗留Fortran代码到可移植Kokkos：一种自主智能体AI工作流

    From Legacy Fortran to Portable Kokkos: An Autonomous Agentic AI Workflow

    [https://arxiv.org/abs/2509.12443](https://arxiv.org/abs/2509.12443)

    本文提出一种自主智能体AI工作流，通过多个专门的LLM智能体协作，自动将遗留Fortran内核翻译、验证、测试、调试并优化为具有跨硬件性能可移植性的Kokkos C++代码。

    

    科学应用持续依赖最初为同构、基于CPU的系统开发的遗留Fortran代码库。随着高性能计算（HPC）向异构GPU加速架构转变，许多加速器缺乏原生Fortran绑定，因此迫切需要对遗留代码进行现代化改造以实现可移植性。Kokkos等框架提供了性能可移植性和单源C++抽象，但手动将Fortran移植到Kokkos需要大量专业知识和时间。大型语言模型（LLM）在源到源代码生成方面已展现出潜力，但其在完全自主工作流中用于翻译和优化并行代码的应用在很大程度上仍未被探索，尤其是跨多种硬件的性能可移植性方面。本文提出了一种智能体AI工作流，其中专门的LLM“智能体”协同工作，将Fortran内核进行翻译、验证、编译、运行、测试、调试和优化为……

    arXiv:2509.12443v4 Announce Type: replace-cross  Abstract: Scientific applications continue to rely on legacy Fortran codebases originally developed for homogeneous, CPU-based systems. As High-Performance Computing (HPC) shifts toward heterogeneous GPU-accelerated architectures, many accelerators lack native Fortran bindings, creating an urgent need to modernize legacy codes for portability. Frameworks like Kokkos provide performance portability and a single-source C++ abstraction, but manual Fortran-to-Kokkos porting demands significant expertise and time. Large language models (LLMs) have shown promise in source-to-source code generation, yet their use in fully autonomous workflows for translating and optimizing parallel code remains largely unexplored, especially for performance portability across diverse hardware. This paper presents an agentic AI workflow where specialized LLM "agents" collaborate to translate, validate, compile, run, test, debug, and optimize Fortran kernels into
    

