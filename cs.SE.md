# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Function-level Dataset of Vulnerable and Fixed Source Code in JavaScript and TypeScript](https://arxiv.org/abs/2609.38012) | JsVul是一个针对JavaScript和TypeScript的函数级漏洞数据集，通过语言感知的预处理流水线、多阶段去重和启发式标注，从七个来源精选出高质量的修复前后代码对，为自动化漏洞检测提供了可靠的训练数据。 |
| [^2] | [Merged, Not Measured: An Empirical Study of Performance Issues Fixed by Coding Agents](https://arxiv.org/abs/2609.37985) | 该研究通过对582个仓库中1,262个由编码智能体修复的性能问题进行实证分析发现，合并决定更多取决于智能体历史记录和仓库惯例而非实际性能验证——被拒修复的性能声称大多无法通过重新执行验证，而已合并修复倾向于删除更多代码行。 |
| [^3] | [AgentBug-Smith: Automatically Reproducing Real-World Harness Bugs in Agentic Systems](https://arxiv.org/abs/2609.37864) | 提出了自动化框架缺陷复现方法AgentBug-Smith，其复现成功率比现有面向通用软件的缺陷复现技术高出10.67%-27.56%，并据此构建了目前包含200个可复现缺陷的实时可扩展基准Live-Harness-Bench。 |
| [^4] | [Is manual software optimization a thing of the past?](https://arxiv.org/abs/2609.37849) | 研究表明，基于大语言模型的智能体在人类仅定义范围与验证标准的情况下即可自主优化科学软件（包括已被人类深度优化的成熟实现），在t-SNE、ssGSEA和图元计数等问题上取得显著性能提升，预示手动软件优化可能正在成为历史。 |
| [^5] | [Retrieve, Reproduce, Reveal: Dissecting Retrieval-Augmented Software Vulnerability Detection](https://arxiv.org/abs/2609.37669) | 该研究通过在开放权重环境下复现六个开源RAG4SVD系统，并建立采用统一数据集、指标套件和模型池的统一基准，解决了检索增强软件漏洞检测中的可复现性与可比性问题。 |
| [^6] | [Beyond Productivity: Measuring Developers' Cognitive Load During GenAI-Supported Software Development](https://arxiv.org/abs/2609.37645) | 该研究通过在SAP两个站点开展为期四天的实地研究，首次测量生成式AI辅助开发中专业开发者的认知负荷，并证明可穿戴设备的生理数据能在任务情境之外提供额外信息，为以开发者为中心的GenAI评估开辟了新途径。 |
| [^7] | [Independent Verification Paths Are Not Independent: A Case Study of Common-Mode Failure in a Satellite Catalogue Pipeline](https://arxiv.org/abs/2609.37603) | 当两条基于不同技术的验证路径共享了包含同一误读的相同常量时，交叉校验会因共模失效而失效——本研究中七项交叉检查虽全部“一致”，却有三项出错，其中一项数据被高估了四倍多（932对220）。 |
| [^8] | [Profiling the Energy Consumption of Serverless Functions with Joule Profiler](https://arxiv.org/abs/2609.37531) | 本文设计了名为 Joule Profiler 的实验性基准测试环境，实证剖析了无服务器函数在容器引擎、虚拟机监控器、单内核及语言运行时等各执行层次上的能耗与能效影响。 |
| [^9] | [Exploring Emotional Intelligence in Software Testing](https://arxiv.org/abs/2609.37526) | 本研究通过对瑞典16名软件测试人员的半结构化访谈，揭示情绪智力在测试人员应对截止期限压力、团队沟通冲突和需求变更中发挥着关键作用。 |
| [^10] | [VeriWeave Govern: Evidence-Gated Deterministic Runtime Governance for Enterprise AI Agents](https://arxiv.org/abs/2609.37457) | VeriWeave Govern是一个确定性运行时治理层，通过版本化策略评估、类型化证据验证和固定的拒绝优先顺序，将企业AI智能体的动作生成与授权分离，在六万个评估案例中实现了零错误放行和零治理攻击成功率。 |
| [^11] | [Complexity-Aware Evaluation of LLM Comprehension](https://arxiv.org/abs/2609.37405) | 该论文提出一个基于圈复杂度、嵌套深度、分支因子和Halstead体积的复杂度感知评估框架，发现DeepSeek-Coder-V2在代码理解上整体准确率（78.33%）优于Llama（70.33%），但两者的准确率均随代码复杂度增加而显著下降。 |
| [^12] | [Mubric: Mutation Testing-Guided Rubric Generation for LLM Evaluation](https://arxiv.org/abs/2609.37322) | Mubric借鉴软件工程中变异测试的思想，通过从真实的偏好与非偏好响应对中挖掘常见缺陷并抽象为可复用的变异算子，自动生成能够可靠捕捉任务特定质量要求的LLM评估评分准则。 |
| [^13] | [Do Agent Benchmarks Do What They Say? An Executable-Contract Audit of Tool-Using Agent Environments](https://arxiv.org/abs/2609.37315) | 该论文提出将工具接口声明视为可执行契约来审计工具使用型智能体基准测试，发现基准测试盲目信任模拟工具的执行结果，并揭示了现有审计方法对隐藏在分数之下的工具缺陷普遍存在漏检问题。 |
| [^14] | [SafeLLM4SE: Statistical Evaluation and Reporting for LLM-based Software Engineering Systems](https://arxiv.org/abs/2609.37294) | SafeLLM4SE 提出了一套基于统计原则的评估方法与报告标准，将大语言模型在软件工程任务中的输出视为随机过程的实现，通过自适应采样、置信区间、分布感知统计比较和效应量来区分并评估系统的质量、稳定性与估计不确定性。 |
| [^15] | [CRJudgeBench: Can AI Detect Plausible but Invalid Code Reviews?](https://arxiv.org/abs/2609.37216) | 该论文提出了包含1199个实例的CRJudgeBench基准，用于评估AI能否识别看似合理但技术上错误的代码审查评论，并设计了基于仓库上下文、主动收集代码证据进行验证的智能体裁判Sentinel。 |
| [^16] | [LoLBench: Evaluating Coding Agents with Long-Horizon Proposals on Large Software Systems](https://arxiv.org/abs/2609.37143) | LoLBench是一个多语言基准测试，通过让编码智能体在大型软件系统上完成从人工编写的增强提案到代码实现的完整流程，首次同时评估其将用户意图转化为规范的感知能力和生成正确代码修改的实现能力。 |
| [^17] | [GitCF: Reducing Incomplete Changes by Exploiting Multiple Similarities Among Commits](https://arxiv.org/abs/2609.37045) | 提出GitCF方法，通过利用开发者当前变更与历史提交之间的多重相似性来推荐应一并修改的位置，从而减少不完整变更。 |
| [^18] | [Cross-Organizational SysML Model Integration: A Survey of Challenges and AI-Supported Tasks](https://arxiv.org/abs/2609.37000) | 该论文通过对29位MBSE利益相关者的问卷调查，首次系统揭示了跨组织SysML模型集成是一个涉及语义、行为、可追溯性和交换互操作性的多维度对齐问题，并评估了六类AI支持任务的作用及相关认知差异。 |
| [^19] | [XRepoSkill: Learning Transferable Skills for Software Engineering Agents](https://arxiv.org/abs/2609.36807) | 提出XRepoSkill方法，通过对比同一智能体解决同一问题时的成功与失败轨迹分歧点，提取带可执行谓词的可验证技能规则，从而解决软件工程智能体技能难以跨仓库迁移的问题。 |
| [^20] | [GitHarness: Git Init Your Harness Working Memory for Perpetual User Requirements](https://arxiv.org/abs/2609.36789) | 提出 GitHarness，一个可插拔的 Git 风格框架，将需求状态与工具工作状态组织成可分支的版本历史，使 LLM 智能体在长周期人机协作中能够持续跟踪动态需求变化并实现局部更新而非全局重写。 |
| [^21] | [The Editor Has Read-Only Access: Correctness Signals in Diffusion Language Models](https://arxiv.org/abs/2609.36783) | 扩散语言模型的内部激活中编码了代码正确性信号，线性探针可有效检测该信号，但利用其进行生成引导并未带来可靠的性能提升。 |
| [^22] | [Can Agents Design Libraries for Agents?](https://arxiv.org/abs/2609.36730) | 该论文提出LibraryDesignBench基准来评估智能体为其他智能体设计代码库的能力，发现智能体设计者在大多数任务上能复现人类生产级库的抽象，但下游智能体对库的利用不足，经常重新实现库中已有的功能。 |
| [^23] | [CTE-Bench: Counterfactual Trace Evaluation for Stateful Software Simulators](https://arxiv.org/abs/2609.36647) | 提出CTE-Bench基准，通过要求模型预测代码修补或状态覆盖等干预后有状态服务对40个固定未来调用的响应，并以实际执行验证预测结果，从而在不让模型选择行动的情况下评估其模拟有状态软件反事实行为的能力。 |
| [^24] | [WitnessGym: Benchmarking Coding Agents on the Construction of Bug Witnesses](https://arxiv.org/abs/2609.36635) | WitnessGym通过向真实项目自动注入缺陷、重建项目并验证见证可暴露性，构建了包含1,300个用例的缺陷验证基准，用于评测编码智能体生成可执行缺陷见证的能力。 |
| [^25] | [LatentSift: Policy-State Filtering for Token-Efficient Verification of Software Engineering Agents](https://arxiv.org/abs/2609.36371) | LatentSift复用策略自身生成的隐藏状态构建正负样本库来过滤候选轨迹，无需额外的LLM验证推理，从而大幅降低软件工程智能体验证阶段的token消耗。 |
| [^26] | [Strategies for Deploying AI Agents in Production at Scientific User Facilities](https://arxiv.org/abs/2609.36362) | 基于在先进光子源（APS）部署大语言模型驱动智能体的实践经验，本文提出了在光源、中子源等科学用户设施中将AI智能体推向生产环境的可复用策略与设计原则。 |
| [^27] | [Towards an AI Software Factory for Data Systems](https://arxiv.org/abs/2609.36323) | 该论文提出构建覆盖软件开发生命周期全阶段（目标定位、编码、审查、运维）的AI软件工厂，通过元数据驱动自我改进，在微软数十个代码仓库的实际部署中实现了相比代理式编程3倍的工程效率和高达22倍的token效率。 |
| [^28] | [How Much Prompt Is Enough? A Blackbox Minimization of Few-Shots in LLMs](https://arxiv.org/abs/2609.36289) | 该论文提出了一个黑盒提示最小化框架，实验证明少样本提示可在完全保持输出保真度的情况下平均缩减65.3%的字符量。 |
| [^29] | [SCOUT: Synergizing Reasoning and Tool-Use for Computer-Use Safety](https://arxiv.org/abs/2609.36201) | SCOUT提出了一种两阶段的代理式安全验证器，通过将推理密集的评分标准生成与工具密集的证据收集相结合，能够有效检测计算机使用代理在执行任务时产生的细微且隐蔽的有害行为。 |
| [^30] | [Assay: Claims That Decay With the Code. Content-Addressed Evidence Graphs for Accountable AI-Assisted Software Delivery](https://arxiv.org/abs/2609.36170) | Assay系统将AI编码代理做出的每项断言（如测试通过、无密钥泄露）绑定到其覆盖代码的依赖锥的Merkle哈希上，使断言在代码或其依赖发生变化时精确失效，从而为AI辅助软件交付提供可验证、可问责的证据图机制。 |
| [^31] | [From Dead Code and Static Requirements to Working Engines: Software Revival with Coding Agents](https://arxiv.org/abs/2609.36161) | 该论文提出ReviveBench基准，通过隐藏验证器评估编程智能体复活无法运行的软件（涵盖依赖不兼容、删除核心模块、遗留构建及GPU基础模型等十个任务）以及从开放规范重建工业软件引擎的能力，结果显示最强模型能够通过全部十个复活任务，且在防污染实验中表现稳健。 |
| [^32] | [The Invisible Throttle: Running on Borrowed Time in Scratch](https://arxiv.org/abs/2609.36152) | 本文揭示了 Scratch 中一条未记录的执行速度规则：脚本并非如普遍认为的“每帧迭代一次”，而是受全局重绘门控隐形限速——隐藏负责绘制的角色可让不绘制的循环快达 77,000 倍。 |
| [^33] | [Live Architecture Models for Cloud-Native Architecture-as-Code: Early Results from Kubernetes Conformance Checking](https://arxiv.org/abs/2609.36148) | 本文提出“实时架构模型”方法，通过Kubernetes部署语言（KDL）子集和VS Code原型Archer将可编辑架构模型与Kubernetes运行时事实相连接，利用快照恢复和只读一致性检查来检测集群与架构意图之间的漂移不一致。 |
| [^34] | [The Invisible Scheduler: Dragging a Sprite Can Change What a Scratch Program Does](https://arxiv.org/abs/2609.36147) | 本文发现Scratch中角色图层的堆叠顺序会隐性地决定并发脚本的启动顺序，导致“拖动角色即可改变程序行为”且该隐患会随项目保存，并提出StackSwap工具通过系统性枚举不同堆叠顺序下的运行结果来检测此类与顺序相关的隐性问题。 |
| [^35] | [When Does Correction Become Repair? Mechanistic Auditing of Internal Interventions in Tool-Using LLMs](https://arxiv.org/abs/2609.36138) | 提出SAKIKO审计框架，揭示工具使用型大语言模型内部干预中“行为改变不等于修复”——即便干预带来净收益，也可能损害超过一半的原始决策，因此必须通过目的地解析验证来审计干预的真实效果。 |
| [^36] | [The Uneven Decline of Collective Knowledge Production: Evidence from Stack Overflow After Generative AI](https://arxiv.org/abs/2609.36069) | 通过分析ChatGPT发布后Stack Overflow上的200多万个问题，研究发现简单问题急剧减少而困难问题日益增多，表明生成式AI对集体知识生产造成了不均衡的侵蚀。 |
| [^37] | [Irene: Equivalence Checking of Hybrid Quantum Programs via Structure-Preserving Symbolic Reduction](https://arxiv.org/abs/2609.36065) | Irene是一个通过门级代数化简、混合路径和类型化图同构验证以及密度核分析这三层推理，来实现有界混合量子程序等价性验证的结构保持符号规约框架。 |
| [^38] | [Evaluating Name-Only Directory Routing for One-Shot Code Search](https://arxiv.org/abs/2609.35918) | 该论文证明，仅依据目录和文件名称的语言模型路由方法在一次性代码搜索中显著优于FTS5和rg等固定词法查询，在八位候选文件内实现0.465的文件召回率，且计算成本远低于扁平路径等探索性方法。 |
| [^39] | [UNBIND: UNlearning By INference-time Directional Steering for Code LLMs](https://arxiv.org/abs/2609.35913) | UNBIND是一种代码大语言模型遗忘框架，通过分别建模目标代码对应的隐藏状态与抑制其复制的方式构建独立引导方向，在不修改模型权重的情况下于推理时实现选择性遗忘，同时保持通用编程能力。 |
| [^40] | [The Artifact Promotion Control Model: An Implementation Case Study. Build on Target Machines vs. Build Once and Promote Artifacts](https://arxiv.org/abs/2609.35891) | 本文将“工件晋升”（一次构建、将同一构建副本逐环境晋升）形式化为严格的控制模型，并论证其比在目标机器上构建更能直接满足FedRAMP、SOX 404、HIPAA等法规的完整性与变更控制要求。 |
| [^41] | [PoliVEM: a Python-driven virtual element framework for computational solid mechanics](https://arxiv.org/abs/2609.35878) | PoliVEM是一个Python驱动的虚拟单元法软件框架，采用模块化C++核心统一实现多种固体与结构力学问题，新公式只需提供投影、离散形式和稳定化项即可复用网格、组装和求解等现有基础设施。 |
| [^42] | [More Programs or More Rolls? Separating Coverage from Specialization in LLM Harnesses](https://arxiv.org/abs/2609.35873) | 该研究通过受控评估将答案覆盖与任务专业化分离，发现LLM测试装置的性能提升主要来自重复执行带来的答案覆盖而非真正的专业化——生成的程序主要持续暴露弱点而非稳定优势，预执行选择也几乎没有增益。 |
| [^43] | [Beyond Rule-Based Mutation Testing: Test-Aware Mutant Generation Using Large Language Models](https://arxiv.org/abs/2609.35841) | 本文提出测试感知的变异体生成方法，让大语言模型在提示中同时感知问题描述、标准解答和现有基础测试，并生成能通过这些测试的非平凡变异体，从而克服传统规则式变异测试产生琐碎冗余变异体以及现有LLM方法“测试盲”的局限。 |
| [^44] | [A Leakage-Safe, Cost-Aware Regression Testing Methodology for the Quantum Transpiler](https://arxiv.org/abs/2609.35834) | 该论文提出了一种面向 Qiskit 量子转译器的防泄漏、成本感知的回归测试选择方法论，其预先注册、预算绑定的评估表明，透明的风险评分选择器在固定预算下的缺陷检测效果并不优于简单的多样性优先级基线。 |
| [^45] | [Automated Evaluation of Multi-Turn Dialogues in In-Car Conversational Assistants](https://arxiv.org/abs/2609.35812) | 提出了一个自动化测试框架，通过闭环仿真结合策略引导的用户模拟器、对抗性策略管理器和双层LLM裁判，来评估车载对话助手在多轮对话中的约束处理、上下文保持和安全关键行为。 |
| [^46] | [Lookahead-R: Budget-Aware Tool Retrieval via Execution-Centric Planning](https://arxiv.org/abs/2609.35811) | Lookahead-R通过轻量级执行感知代理世界模型（无需调用真实API即可预测工具执行结果、延迟与语义效用），结合预算感知的蒙特卡洛树搜索，将工具检索转化为资源受限的序贯决策问题，实现了精度与效率的最优平衡。 |
| [^47] | [Agent-Callable Feature Coverage: Measuring Software Readiness for AI Agents](https://arxiv.org/abs/2609.35789) | 该论文提出GUI-API对等原则，并引入智能体可调用功能覆盖率（ACFC）指标与智能体就绪度符合性（ARC）评估框架，从可访问性、可发现性和可控性三个维度衡量软件对AI智能体的就绪支持程度。 |
| [^48] | [TokenCast: Forecasting Token Consumption During LLM Agent Execution](https://arxiv.org/abs/2609.35760) | 提出TokenCast方法，通过学习可组合的执行片段成本表示来预测LLM智能体执行任务时的Token消耗，并随执行进展动态刷新预测，无需额外的LLM调用。 |
| [^49] | [MCP Error Messages Written for Developers Hurt the Most Capable Agents Most](https://arxiv.org/abs/2609.35381) | 研究发现MCP服务器中面向人类开发者的错误信息会误导只能调用工具的AI智能体去执行无法完成的操作，且模型能力越强、越忠实遵循这些指令，受到的性能损失反而越大。 |
| [^50] | [After the Fix: Transfer of Corrected Agent Experience](https://arxiv.org/abs/2609.34603) | 本研究通过3,300次运行系统评估了修复后的智能体经验向后续任务迁移的效果，发现修正经验带来的收益在很大程度上源于未修正基线较弱而非记忆质量的真正提升，且并非所有修正机制（如APEX）都能产生可比的修正收益。 |
| [^51] | [Counterfactual Rollout Replay: Forkable Environments as Free Process Rewards for Software Engineering Agents](https://arxiv.org/abs/2609.33875) | 该论文提出反事实回放重演（CRR），利用可分叉的可执行环境在选定决策点采样替代动作并对比终端回报，从而在无需人工过程标注或过程奖励模型的情况下，为软件工程智能体提供免费的步级过程监督信号，并在多个 SWE 基准上提升了 pass@1。 |
| [^52] | [Evaluating System One Models for Agent Security Decisions: Reliability, Calibration, and Selective Automation](https://arxiv.org/abs/2609.33401) | 该研究系统评估了四款用于智能体安全决策的系统一模型，发现良好的整体表现和校准可能掩盖针对特定攻击类别的集中性失误，且适配配置并不总是优于其基础模型。 |
| [^53] | [Compositional Safety Failures in Harness Evolution: Identification and Runtime Monitoring](https://arxiv.org/abs/2609.33123) | 该研究首次系统揭示了Harness演化中的组合式安全失效——即各自安全且不损害效用的组件更新在交互后可能引发不安全的智能体行为——在三个安全基准上识别出43个两两组合和18个不可约的三路组合安全失效，并提出运行时监控方法以应对跨组件安全验证的组合复杂度难题。 |
| [^54] | [Relic: From Multi-Agent Collaboration to Persistent Organizational Capability](https://arxiv.org/abs/2609.32965) | Relic系统将多智能体协作中反复出现的失败转化为组织拥有的可执行协议，使经验教训能够在成员更替后持续约束团队行为，并通过360次受控实验证明了其提升完整契约交付率的有效性。 |
| [^55] | [ASCEND: Personal AI Agents for Autonomous Scientific Computing Across HPC Clusters and GPU Workstations](https://arxiv.org/abs/2609.32868) | ASCEND是一个运行在研究人员本地笔记本电脑上的AI智能体，通过安全认证连接远程调度Slurm集群和GPU工作站，无需设施级服务即可自主完成科学计算中的作业提交、故障诊断与恢复闭环。 |
| [^56] | [The Complexity Kink: A Prompt-Side Structural Complexity Index for Code-Generation Reliability](https://arxiv.org/abs/2609.19616) | 该论文提出一个生成前评分的六维提示侧结构复杂度指数，发现代码生成通过率在复杂度分数上存在非单调断点，但该断点并非通用的失败临界值，会随任务类型固定效应等因素发生移动。 |
| [^57] | [Root-Cause Attribution Is a Search Problem: Continual Search for Long-Horizon Agent Failures](https://arxiv.org/abs/2609.13463) | 本文将根因归因重新定义为一个大规模搜索问题，指出现有的一次性LLM判断方法在长轨迹中会过早下结论而遗漏关键证据，并提出了针对长时程智能体任务失败的持续搜索诊断方法。 |
| [^58] | [The reach of a verification tool decides its value: A controlled study of verification surface, artifact quality, and cost in AI coding agents](https://arxiv.org/abs/2608.28795) | 该研究通过控制单一变量的受控实验（六个模型、八种工具配置、1,116个Web应用）发现，为AI编程智能体配备验证工具能显著提升交付软件的质量，其中最廉价的启动探针即可消除绝大多数应用无法启动的失败。 |
| [^59] | [Active-SWE: Benchmarking Coding Agents for Proactive Bug Fixing without Issue Reports](https://arxiv.org/abs/2608.04682) | 该论文提出了Active-SWE基准测试，首次将评估重点从依赖问题报告的被动式Bug修复转向无报告指导下的主动式Bug发现与修复，涵盖1,663个任务、6种Bug类别和8种编程语言，并支持多Bug修复与潜在Bug发现等更深层次的评估场景。 |
| [^60] | [CURATE: Leveraging LLM Agents to Compose, Catalog, and Deploy Reproducible Workflows](https://arxiv.org/abs/2608.04270) | 本文提出CURATE，一种人在回路的多智能体系统，利用LLM智能体覆盖计算工作流从组合、复用、编目到部署的完整生命周期，并通过模块目录实现模块的存储、共享与复用以支持FAIR原则。 |
| [^61] | [Multi-Mode Debugging for FRP-Based Embedded Systems](https://arxiv.org/abs/2608.04264) | 本文提出了一种面向基于Emfrp的嵌入式系统的多模式调试框架，通过源代码映射技术，既支持在Emfrp抽象层面进行调试，又允许检查平台相关的C/C++ I/O代码，从而弥合了源代码级FRP程序与可执行系统之间的抽象鸿沟。 |
| [^62] | [Low Reasoning Effort Is Enough for Routine Office Work by Language-Model Agents](https://arxiv.org/abs/2608.03169) | 研究表明，语言模型智能体执行常规办公任务时，低推理强度与最高推理强度表现同样可靠（始终遵守规则、不使用禁止工具），却能节省约43%的输出token和20%的时间。 |
| [^63] | [ORCA-bench: How Ready Are Language Model Agents for Oncall?](https://arxiv.org/abs/2607.28545) | 该论文提出了ORCA-bench基准，将1,079个根因分析任务与真实可观测性工具接口及六天生产级遥测数据相结合，系统评估语言模型智能体在值班根因分析场景中的真实能力。 |
| [^64] | [SIGIL: Compiling Agent Skills into Typed Harnesses](https://arxiv.org/abs/2607.27309) | SIGIL通过技能编译范式，将自然语言技能确定性编译为可执行代码，显著提升了智能体在执行过程中对技能要求的合规性。 |
| [^65] | [CodeNib: A Multi-View Data System for Serving Repository Context to Coding Agents](https://arxiv.org/abs/2607.25431) | CodeNib 是首个将同一代码提交的词汇、稠密和结构视图编译在统一清单与源地址契约之下的多视图数据系统，使编程智能体能够相互校验预计算结果并通过单一成本可见的运行时获得搜索、导航和有界上下文服务。 |
| [^66] | [HEDGEHOG: Hierarchical Evaluation of Drug Generators Through Rigorous Filtration](https://arxiv.org/abs/2607.13155) | 提出了HEDGEHOG——一个模拟命中化合物鉴定工作流程的统一六阶段严格过滤评估基准，用于更真实地评估分子生成器的药用合理性，并在KRAS G12D案例中评估了22个生成模型。 |
| [^67] | [Is Agent Code Less Maintainable Than Human Code?](https://arxiv.org/abs/2606.21804) | 本研究提出 CodeThread 框架，通过受控实验发现智能体基于智能体生成代码解决任务的效率低于基于人类代码（任务解决率最多下降 13.1%），且传统可维护性指标无法解释这一差异。 |
| [^68] | [Code Lifespan Survival Analysis (CLSA): Predicting the Survival of Source Code Lines Using AST-Aware Mining](https://arxiv.org/abs/2606.04993) | 该论文提出了首个对单行代码删除风险进行建模的框架CLSA，通过对120个TypeScript仓库中3250万次代码行诞生事件的生存分析，发现AST结构和代码行长度等可静态计算的协变量是预测代码行删除风险的最强因子。 |
| [^69] | [Poking Around in the Dark: Why a Shared Understanding of Components Matters](https://arxiv.org/abs/2606.02442) | 该论文通过自底向上分析软件开发生命周期中的组件包含机制，并使用六种编程语言的真实数据系统评估五种主流SBOM生成工具，揭示了不同工具对组件的定义与识别存在显著差异，证明业界缺乏关于SBOM应包含哪些组件的共同理解，现有技术尚不足以保障软件供应链安全。 |
| [^70] | [Remember Your Trace: Memory-Guided Long-Horizon Agentic Framework for Consistent and Hierarchical Repository-Level Code Documentation](https://arxiv.org/abs/2605.14563) | 提出MemDocAgent长程智能体框架，通过依赖感知的遍历引导与基于共享记忆RepoMemory的记忆引导智能体交互，在覆盖整个仓库的单一集成上下文中生成一致且具有分层结构的仓库级代码文档。 |
| [^71] | [PlayCoder: Making LLM-Generated GUI Code Playable](https://arxiv.org/abs/2604.19742) | 该论文提出基于43个多语言GUI应用构建的PlayEval基准与Play@k指标，从交互流程和UI逻辑角度评估大语言模型生成的GUI代码能否真正可运行，弥补了传统测试用例评估方式的不足。 |
| [^72] | [CIRCLE: A Framework for Evaluating AI from a Real-World Lens](https://arxiv.org/abs/2602.24055) | CIRCLE是一个六阶段、基于生命周期的AI评估框架，通过将AI技术栈之外的利益相关者需求转化为可衡量的前瞻性信号，弥合了模型性能指标与AI系统真实部署效果之间的差距。 |
| [^73] | [Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?](https://arxiv.org/abs/2602.11988) | 本文首次严格评估了 AGENTS.md 等仓库级上下文文件的实际效果，发现它们并不能普遍提升编程智能体的任务成功率，反而平均增加超过 20% 的推理成本。 |
| [^74] | [Bridging User Feedback and System Diagnosis: Reproducing Mobile Performance Issues from Reviews](https://arxiv.org/abs/2508.11147) | 本文提出了首个方法RevPerf，通过语义检索和提示工程整合互补的应用评论信息，实现从用户评论自动复现移动应用性能问题，从而弥合用户反馈与系统诊断之间的鸿沟。 |

# 详细

[^1]: JavaScript和TypeScript中易受攻击与已修复源代码的函数级数据集

    A Function-level Dataset of Vulnerable and Fixed Source Code in JavaScript and TypeScript

    [https://arxiv.org/abs/2609.38012](https://arxiv.org/abs/2609.38012)

    JsVul是一个针对JavaScript和TypeScript的函数级漏洞数据集，通过语言感知的预处理流水线、多阶段去重和启发式标注，从七个来源精选出高质量的修复前后代码对，为自动化漏洞检测提供了可靠的训练数据。

    

    JavaScript和TypeScript在现代Web开发中被广泛使用，因此其安全性至关重要；然而，自动化漏洞检测通常受到高质量训练数据可用性的限制。在这里，我们提出了JsVul，这是一个从七个主要来源精选而成的数据集。与可能保留噪声（如压缩代码和装饰性编辑）的通用多语言数据集不同，JsVul采用了针对特定语言的流水线。我们收集了安全修复前后的文件版本，并通过过滤无关制品和应用自动语法规范化，隔离出与安全相关的更改。我们通过多阶段去重和基于启发式方法的标注确保了数据的完整性。JsVul以按时间排序的JSONL格式提供，支持在JavaScript和TypeScript生态系统中进行稳健的模型训练，并证明了语言感知预处理在构建漏洞数据集中的重要性。

    arXiv:2609.38012v1 Announce Type: cross  Abstract: JavaScript and TypeScript are widely used in modern web development, making their security critical; however, automated vulnerability detection is often constrained by the availability of high-quality training data. Here we present JsVul, a dataset curated from seven major sources. Unlike generic multi-language datasets that may retain noise -- such as minified code and cosmetic edits -- JsVul utilizes a language-specific pipeline. We collected pre-fix and post-fix versions of files around security fixes and, by filtering irrelevant artifacts and applying automated syntax normalization, isolated security-related changes. We ensured data integrity through multi-stage deduplication and heuristic-based labeling. Provided in a time-ordered JSONL format, JsVul supports robust model training in the JavaScript and TypeScript ecosystem and demonstrates the importance of language-aware preprocessing in building vulnerability datasets.
    
[^2]: 合并而非验证：编码智能体修复性能问题的实证研究

    Merged, Not Measured: An Empirical Study of Performance Issues Fixed by Coding Agents

    [https://arxiv.org/abs/2609.37985](https://arxiv.org/abs/2609.37985)

    该研究通过对582个仓库中1,262个由编码智能体修复的性能问题进行实证分析发现，合并决定更多取决于智能体历史记录和仓库惯例而非实际性能验证——被拒修复的性能声称大多无法通过重新执行验证，而已合并修复倾向于删除更多代码行。

    

    编码智能体提交的拉取请求声称能加速软件，但针对人类性能修复的研究很少说明维护者如何回应此类修复，或其声称是否属实。从AIDev v4数据集的71,677个智能体PR中，通过文本过滤以及语言模型和作者的手册式编码，筛选出六种智能体在582个仓库中修复的1,262个性能问题。研究者对每个问题及其测试进行编码，并重新执行了23个被拒绝和30个已合并的修复。(1) 已关闭的修复中有57%被合并，61%的拒绝未给出任何明确理由，且在对主要为智能体构建的工作负载进行的三次运行试点中，23个被重新执行的被拒修复中仅有6个的性能声称成立。(2) 接受率随智能体在该仓库的历史表现记录（从31-37%提升至70%）以及该仓库在其他智能体PR上的合并率（从33%提升至84%）而上升。已合并的修复删除了其所修改行中更大比例的代码（0.26 对 0.15），这一差异在单个智能体内部和单个仓库内部均保持一致。

    arXiv:2609.37985v1 Announce Type: new  Abstract: Coding agents open pull requests (PRs) that claim to speed up software, but studies of human performance fixes say little about how maintainers respond to such a fix or whether its claim holds. From the 71,677 agent PRs of AIDev v4, a text filter and codebook coding by language models and by the authors select 1,262 performance issues fixed by six agents in 582 repositories. We code each issue and its tests and re-execute 23 rejected and 30 merged fixes. (1) 57% of closed fixes are merged, 61% of rejections give no stated reason, and only 6 of the 23 re-executed rejected claims held under our three-run pilot on mostly agent-built workloads. (2) Acceptance rises with the agent's track record in the repository (31-37% to 70%) and with the repository's pre-opening merge rate on its other agent PRs (33% to 84%). Merged fixes delete a larger share of the lines they change (0.26 versus 0.15), a difference that holds within agent and within rep
    
[^3]: AgentBug-Smith：自动复现智能体系统中真实世界框架缺陷的方法

    AgentBug-Smith: Automatically Reproducing Real-World Harness Bugs in Agentic Systems

    [https://arxiv.org/abs/2609.37864](https://arxiv.org/abs/2609.37864)

    提出了自动化框架缺陷复现方法AgentBug-Smith，其复现成功率比现有面向通用软件的缺陷复现技术高出10.67%-27.56%，并据此构建了目前包含200个可复现缺陷的实时可扩展基准Live-Harness-Bench。

    

    智能体框架缺陷具有独特的特征，对于最先进的软件智能体来说仍然难以修复。该领域的进展还受到现有基准测试的阻碍，这些基准测试仅包含少量且固定的可执行框架缺陷，同时需要数百小时的人工时间来构建。本工作提出了AgentBug-Smith，一种自动化的框架缺陷复现方法，能够持续地从开源智能体系统中发现并复现真实世界的框架缺陷。在不同的骨干大语言模型上，AgentBug-Smith始终优于为通用软件设计的现有缺陷复现技术，在复现框架缺陷方面取得了10.67%至27.56%的更高成功率。通过将AgentBug-Smith应用于实际的开源智能体系统，我们构建了Live-Harness-Bench，这是一个实时且可扩展的基准测试，目前包含200个可复现的框架缺陷。我们进一步展示了Live-Harness-Bench的实用性。

    arXiv:2609.37864v1 Announce Type: cross  Abstract: Agent harness bugs exhibit unique characteristics and remain challenging for state-of-the-art software agents to repair. Progress in this area is further hindered by existing benchmarks, which contain only a small and fixed number of executable harness bugs while requiring hundreds of human hours to construct. This work presents AgentBug-Smith, an automated harness bug reproduction approach that continuously discovers and reproduces real-world harness bugs from open-source agentic systems. Across different backbone LLMs, AgentBug-Smith consistently outperforms existing bug reproduction techniques designed for general software, achieving 10.67% - 27.56% higher success rates of reproducing harness bugs. By applying AgentBug-Smith to open-source agentic systems in the wild, we construct Live-Harness-Bench, a live and extensible benchmark that currently contains 200 reproducible harness bugs. We further demonstrate the utility of Live-Harn
    
[^4]: 手动软件优化是否已成过去式？

    Is manual software optimization a thing of the past?

    [https://arxiv.org/abs/2609.37849](https://arxiv.org/abs/2609.37849)

    研究表明，基于大语言模型的智能体在人类仅定义范围与验证标准的情况下即可自主优化科学软件（包括已被人类深度优化的成熟实现），在t-SNE、ssGSEA和图元计数等问题上取得显著性能提升，预示手动软件优化可能正在成为历史。

    

    科学软件日益需要处理更庞大的数据集，同时保持可接受的执行时间。软件优化传统上需要在编程、算法和数值方法方面具备深厚的专业知识。大语言模型（LLM）的最新进展为自动化这一过程的大部分工作提供了可能。我们研究了基于LLM的智能体能否在科学软件中自主实现显著的性能提升，包括那些已经过人类开发者深度优化的成熟实现。我们让一个基于LLM的智能体对三个计算问题的软件进行优化：t-SNE、单样本基因集富集分析和图元计数。人类定义了优化范围、正确性标准和验证机制，之后智能体自主工作，在某些情况下持续工作数小时。代码维护者对智能体生成的每个实现进行了审查……（摘要在此处截断）

    arXiv:2609.37849v1 Announce Type: cross  Abstract: Scientific software is increasingly required to process larger datasets while maintaining acceptable execution times. Software optimization traditionally requires substantial expertise in programming, algorithms, and numerical methods. Recent advances in large language models (LLMs) offer the possibility of automating much of this process. We investigate whether LLM-based agents can autonomously achieve substantial performance improvements in scientific software, including mature implementations that have already been extensively optimized by human developers. We tasked an LLM-based agent with optimizing software for three computational problems: t-SNE, single-sample gene set enrichment analysis (ssGSEA), and graphlet counting. Humans defined the scope, correctness criteria, and a verification mechanism, after which the agent worked autonomously, in some cases for several hours. Code maintainers reviewed each resulting implementation a
    
[^5]: 检索、复现、揭示：剖析检索增强的软件漏洞检测

    Retrieve, Reproduce, Reveal: Dissecting Retrieval-Augmented Software Vulnerability Detection

    [https://arxiv.org/abs/2609.37669](https://arxiv.org/abs/2609.37669)

    该研究通过在开放权重环境下复现六个开源RAG4SVD系统，并建立采用统一数据集、指标套件和模型池的统一基准，解决了检索增强软件漏洞检测中的可复现性与可比性问题。

    

    检索增强生成（RAG）正被越来越多地用于增强基于大语言模型（LLM）的软件漏洞检测，其做法是将预测建立在检索到的漏洞知识（如漏洞报告）之上。然而，现有的基于RAG的软件漏洞检测（RAG4SVD）系统通常使用专有模型进行评估，这对开放科学和可复现性构成了挑战。此外，各项研究使用不同的数据集、自定义的知识库、不同的骨干模型以及多样化的评估指标，这阻碍了跨系统之间的有意义比较。在本工作中，我们研究了六个开源的RAG4SVD系统，并通过以下方式应对这些可复现性与可比性挑战：（i）在开放权重设置下复现它们的实验配置；（ii）构建一个使用统一数据集、统一指标套件和开放权重模型池的统一基准。此外，RAG4SVD系统通常由多个组件构成，但往往……

    arXiv:2609.37669v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) is increasingly used to enhance Large Language Model (LLM)-based software vulnerability detection by grounding predictions in retrieved vulnerability knowledge, such as vulnerability reports. However, existing RAG-based software vulnerability detection (RAG4SVD) systems are often evaluated using proprietary models, which challenges open science and reproducibility. Further, studies use different datasets, custom knowledge bases, different backbone models, and diverse metrics, which hinders meaningful cross-system comparison. In this work, we study six open-source RAG4SVD systems and address these reproducibility and comparability challenges through (i) reproduction of their experimental settings under an open-weight setting, and (ii) a unified benchmark using a common dataset, metric suite, and pool of open-weight models. Further, RAG4SVD systems typically consist of multiple components, yet are oft
    
[^6]: 超越生产力：测量生成式AI辅助软件开发中开发者的认知负荷

    Beyond Productivity: Measuring Developers' Cognitive Load During GenAI-Supported Software Development

    [https://arxiv.org/abs/2609.37645](https://arxiv.org/abs/2609.37645)

    该研究通过在SAP两个站点开展为期四天的实地研究，首次测量生成式AI辅助开发中专业开发者的认知负荷，并证明可穿戴设备的生理数据能在任务情境之外提供额外信息，为以开发者为中心的GenAI评估开辟了新途径。

    

    生成式人工智能（GenAI）正在改变软件开发的工作流程以及开发者的工作方式。业界对GenAI采用的评估通常关注生产力提升、使用情况和产出质量，但对GenAI技术的实际采用者和推动者——软件开发者——的交互体验和认知负荷关注有限。理解GenAI是否改变或转移了开发者在日常开发中的认知需求，对于以开发者为中心评估GenAI辅助的软件开发十分重要。这可以帮助组织设计和评估有效的AI辅助工作流程。在本研究中，我们探讨了GenAI的使用和任务情境与专业开发者感知认知负荷之间的关系，以及由可穿戴设备采集的生理特征是否能在这些情境信息之外提供额外的信息。在一项于SAP两个站点开展的为期四天的工业实地研究中，21名开发者记录了他们的任务

    arXiv:2609.37645v1 Announce Type: new  Abstract: Generative AI (GenAI) is changing software development workflows and how developers work. Industry evaluations of GenAI adoption often monitor productivity gains, usage, and output quality, but limited attention is paid to the interaction experience and cognitive load of the actual adopters and drivers of GenAI technology - the software developers. Understanding whether GenAI changes or shifts developers' cognitive demands during everyday development is important for a developer-centered evaluation of GenAI-supported software development. It can inform organizations in designing and evaluating effective AI-supported workflows. In this work, we study how GenAI use and task context relate to professional developers' perceived cognitive load and whether wearable-derived physiological characteristics provide additional information beyond this context. In a four-day industrial field study at two SAP sites, 21 developers documented their tasks
    
[^7]: 独立验证路径并非真正独立：卫星目录数据管道中共模失效的案例研究

    Independent Verification Paths Are Not Independent: A Case Study of Common-Mode Failure in a Satellite Catalogue Pipeline

    [https://arxiv.org/abs/2609.37603](https://arxiv.org/abs/2609.37603)

    当两条基于不同技术的验证路径共享了包含同一误读的相同常量时，交叉校验会因共模失效而失效——本研究中七项交叉检查虽全部“一致”，却有三项出错，其中一项数据被高估了四倍多（932对220）。

    

    数据管道的一种常见保障措施是冗余计算：通过两条基于不同技术构建的路径推导每个发布的数据，当两者不一致时拒绝输出。我们报告了这样一个校验关卡失效的案例，该案例发生在一项针对两个地球轨道天体开放登记册的跨目录完整性研究中。一个将基于集合的Python路径与对生成的RDF图进行SPARQL查询相比较的校验关卡，在七项计数上打印出“所有交叉检查一致”（ALL CROSS-CHECKS AGREE）。然而其中有三项是错误的，其中一项被高估了四倍多（932对220）。两条路径导入了相同的常量，而这些常量编码了对源数据状态词汇表的误读，因此该错误属于共模失效，校验关卡无法察觉。我们给出了这一错误的机制、一个对每个数字进行核对的对象级账目，以及三项追溯到源数据文档的检查，并在有缺陷的代码及其修正版本上进行了测量。随后，我们将该修正与每个天体的相位历史进行了核对，这些相位历史保存在一个源文件中……（原文摘要到此截断）

    arXiv:2609.37603v1 Announce Type: cross  Abstract: A common safeguard for a data pipeline is redundant computation: derive each published number by two routes built on different technology and refuse to exit when they disagree. We report one such gate failing, in a cross-catalogue integrity study of two open registers of Earth-orbiting objects. A gate comparing a set-based Python path with SPARQL queries over the emitted RDF graph printed ALL CROSS-CHECKS AGREE on seven counts. Three were wrong, one overstated more than fourfold (932 against 220). Both paths imported the same constants, which encoded a misreading of the source's status vocabulary, so the error was common-mode and the gate could not see it. We give the mechanism, an object-level ledger reconciling every figure, and three checks that go back to the source's documentation, measured on the defective code and on its correction. We then checked that correction against each object's phase history, held in a source file the pi
    
[^8]: 使用 Joule Profiler 剖析无服务器函数的能耗

    Profiling the Energy Consumption of Serverless Functions with Joule Profiler

    [https://arxiv.org/abs/2609.37531](https://arxiv.org/abs/2609.37531)

    本文设计了名为 Joule Profiler 的实验性基准测试环境，实证剖析了无服务器函数在容器引擎、虚拟机监控器、单内核及语言运行时等各执行层次上的能耗与能效影响。

    

    云提供商和客户已广泛采用无服务器计算作为一种便捷的范式，用于按需部署和执行函数。为此，无服务器平台需要在函数代码运行之前预先配置合适的执行环境。这些环境由多个层次组成，例如容器引擎、虚拟机监控器、单内核以及编程语言运行时。虽然现有文献已经研究了这些无服务器平台的性能，但它们将函数视为黑盒，社区对于将应用打包为无服务器函数所产生的环境影响缺乏关键见解。因此，本文实证研究了可部署在无服务器平台上的无服务器函数的能效。我们设计了一个实验性基准测试环境，使各利益相关方能够探索执行无服务器函数所涉及的各个层次带来的影响。

    arXiv:2609.37531v1 Announce Type: new  Abstract: Cloud providers and customers have widely adopted serverless computing as a convenient paradigm for deploying and executing functions on demand. To do so, serverless platforms require provisioning an appropriate execution environment before a single line of the function's code runs. These environments consist of several layers, such as container engines, hypervisors, unikernels, and programming language runtimes. While the literature has investigated the performance of these serverless platforms, it treats functions as black boxes, and the community lacks key insights into the environmental impacts of packaging applications as serverless functions. This paper therefore empirically studies the energy efficiency of serverless functions deployable on serverless platforms. We design an experimental benchmarking environment that lets stakeholders explore the impacts of the various layers involved in executing serverless functions. We use it t
    
[^9]: 探索软件测试中的情绪智力

    Exploring Emotional Intelligence in Software Testing

    [https://arxiv.org/abs/2609.37526](https://arxiv.org/abs/2609.37526)

    本研究通过对瑞典16名软件测试人员的半结构化访谈，揭示情绪智力在测试人员应对截止期限压力、团队沟通冲突和需求变更中发挥着关键作用。

    

    背景：情绪智力（EI）是指识别、理解和管理自身及他人情绪的能力。软件测试人员需要在他们无法控制的截止期限下对同事的工作做出评判，而此前软件工程领域关于情绪的研究大多聚焦于开发人员。目的：探索软件测试人员如何描述情绪智力在其日常工作、团队沟通与冲突处理、以及应对需求变更中所扮演的角色。方法：对瑞典16名在敏捷团队中工作的软件测试人员进行半结构化访谈，受访者来自航空、汽车、医疗、IT服务、行政管理、银行和制药等多个行业，并采用基于戈尔曼（Goleman）情绪智力框架的反思性主题分析法进行分析。结果：归纳出三个主题，测试人员描述了在截止期限压力下如何调节压力，并从认可、清晰度和自主性中汲取动力，以及如何管理日常交付……（摘要在此处截断）

    arXiv:2609.37526v1 Announce Type: new  Abstract: Background: Emotional Intelligence (EI) is the ability to recognise, understand, and manage one's own and others' emotions. Software testers deliver judgements about colleagues' work under deadlines they do not control, and prior work on emotion in software engineering has mostly studied developers.   Aims: To explore how software testers describe the part EI plays in their day-to-day work, in communication and conflict within the team, and in responding to requirements volatility.   Method: Semi-structured interviews with 16 software testers in Sweden working in teams that use agile practices, across aviation, automotive, healthcare, IT services, administration, banking and pharmaceuticals, analysed with reflexive thematic analysis informed by Goleman's EI framework.   Results: Three themes. Testers described regulating stress under deadline pressure and drawing motivation from recognition, clarity and autonomy; managing the daily deliv
    
[^10]: VeriWeave Govern：面向企业AI智能体的证据门控确定性运行时治理

    VeriWeave Govern: Evidence-Gated Deterministic Runtime Governance for Enterprise AI Agents

    [https://arxiv.org/abs/2609.37457](https://arxiv.org/abs/2609.37457)

    VeriWeave Govern是一个确定性运行时治理层，通过版本化策略评估、类型化证据验证和固定的拒绝优先顺序，将企业AI智能体的动作生成与授权分离，在六万个评估案例中实现了零错误放行和零治理攻击成功率。

    

    企业人工智能智能体越来越多地调用工具、修改基础设施并处理受保护数据，因此需要将动作生成与动作授权分离。本文提出VeriWeave Govern，一个确定性运行时治理层，它根据版本化策略评估结构化的智能体动作，验证类型化证据，应用固定的“拒绝 > 审查 > 允许”优先级顺序，将重大后果动作路由至可问责的人工审查，并记录可重放的防篡改审计状态。GovernBench在30个独立种子和60,000个带真值标签的案例上对该设计进行评估，涵盖五个企业领域、对抗性证据、分布外动作以及时间维度的策略演化。VeriWeave在评估案例中实现了0.9888的平均准确率、0.9836的宏F1值，观测到的总体错误放行为零，治理攻击成功率为零。六项消融实验表明，证据门控、拒绝优先（原文在此处截断）

    arXiv:2609.37457v1 Announce Type: new  Abstract: Enterprise artificial-intelligence agents increasingly call tools, modify infrastructure, and process protected data, creating a need to separate action generation from action authorization. This article presents VeriWeave Govern, a deterministic runtime governance layer that evaluates structured agent actions against versioned policies, validates typed evidence, applies fixed deny > review > allow precedence, routes consequential actions to accountable human review, and records replayable tamper-evident audit state. GovernBench evaluates the design over 30 independent seeds and 60,000 oracle-labelled cases spanning five enterprise domains, adversarial evidence, out-of-distribution actions, and temporal policy evolution. VeriWeave achieves 0.9888 mean accuracy, 0.9836 macro-F1, zero observed aggregate false allows, and zero observed Governance Attack Success Rate on the evaluated cases. Six ablations show that evidence gating, deny prece
    
[^11]: 复杂度感知的大语言模型代码理解能力评估

    Complexity-Aware Evaluation of LLM Comprehension

    [https://arxiv.org/abs/2609.37405](https://arxiv.org/abs/2609.37405)

    该论文提出一个基于圈复杂度、嵌套深度、分支因子和Halstead体积的复杂度感知评估框架，发现DeepSeek-Coder-V2在代码理解上整体准确率（78.33%）优于Llama（70.33%），但两者的准确率均随代码复杂度增加而显著下降。

    

    大语言模型（LLM）越来越多地被应用于需要理解现有源代码的软件工程任务，包括行为预测、函数解释、调试和代码审查。然而，总体基准准确率可能掩盖模型可靠性随源代码结构复杂化而变化的情况。本文提出了一个复杂度感知的大语言模型代码理解评估框架，采用圈复杂度、嵌套深度、分支因子和Halstead体积等指标。我们通过两个互补的任务评估了DeepSeek-Coder-V2和Llama：对300个Python函数进行自动输入-输出预测，以及对60个函数的均衡子集进行人工评估的语义理解。这些函数被划分为低、中、高三个复杂度区间。DeepSeek-Coder-V2的总体自动预测准确率为78.33%，而Llama为70.33%。然而，准确率随着复杂度提升显著下降（原摘要在此处被截断）。

    arXiv:2609.37405v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used for software engineering tasks that require understanding existing source code, including behavior prediction, function explanation, debugging, and code review. However, aggregate benchmark accuracy can conceal how model reliability changes as source code becomes structurally more complex. This paper presents a complexity-aware framework for evaluating LLM code comprehension using cyclomatic complexity, nesting depth, branching factor, and Halstead volume. We evaluate DeepSeek-Coder-V2 and Llama through two complementary tasks: automatic input-output prediction over 300 Python functions and manually assessed semantic comprehension over a balanced subset of 60 functions. The functions are grouped into Low-, Medium-, and High-complexity bands. DeepSeek-Coder-V2 achieves an overall automatic accuracy of 78.33%, compared with 70.33% for Llama. However, accuracy decreases substantially from
    
[^12]: Mubric：基于变异测试引导的LLM评估评分准则生成方法

    Mubric: Mutation Testing-Guided Rubric Generation for LLM Evaluation

    [https://arxiv.org/abs/2609.37322](https://arxiv.org/abs/2609.37322)

    Mubric借鉴软件工程中变异测试的思想，通过从真实的偏好与非偏好响应对中挖掘常见缺陷并抽象为可复用的变异算子，自动生成能够可靠捕捉任务特定质量要求的LLM评估评分准则。

    

    摘要：基于评分准则（rubric）的评估被广泛用于评估基于大语言模型（LLM）的系统，其方法是将响应质量分解为任务特定的评分标准。然而，自动生成能够可靠捕捉任务特定质量要求的评分准则仍然具有挑战性。我们提出了Mubric，一种由变异测试引导的评分准则生成方法。变异测试是一种经典的软件测试方法，通过向程序中注入故障并检查测试用例能否检测到这些故障来评估测试套件的有效性。我们在测试套件与评分准则之间建立了类比：如果一条评分准则捕捉到了某项重要的质量要求，那么向一个原本高质量的响应中引入相应的缺陷时，其得分应当会降低。Mubric首先从真实的偏好与非偏好响应对中挖掘常见缺陷，并将这些缺陷抽象为可复用的变异算子，每个变异算子指定了如何引入特定类型的响应缺陷。对于新任务，它会应用……（原文在此处截断）

    arXiv:2609.37322v1 Announce Type: new  Abstract: Rubric-based evaluation is widely used to assess LLM-based systems by decomposing response quality into task-specific scoring criteria. However, automatically generating rubrics that reliably capture task-specific quality requirements remains challenging. We introduce Mubric, a mutation testing-guided approach to rubric generation. Mutation testing, a classic software testing methodology, evaluates a test suite by injecting faults into programs and checking whether the tests detect them. We draw an analogy between test suites and rubrics: if a rubric captures an important quality requirement, introducing a corresponding defect into an otherwise high-quality response should reduce its score. Mubric first mines common defects from real pairs of preferred and dispreferred responses and abstracts these defects into reusable mutation operators, each specifying how to introduce a particular type of response defect. For a new task, it applies r
    
[^13]: 智能体基准测试真的名副其实吗？对工具使用型智能体环境的可执行契约审计

    Do Agent Benchmarks Do What They Say? An Executable-Contract Audit of Tool-Using Agent Environments

    [https://arxiv.org/abs/2609.37315](https://arxiv.org/abs/2609.37315)

    该论文提出将工具接口声明视为可执行契约来审计工具使用型智能体基准测试，发现基准测试盲目信任模拟工具的执行结果，并揭示了现有审计方法对隐藏在分数之下的工具缺陷普遍存在漏检问题。

    

    工具使用型智能体正在进入错误操作会带来真实代价的场景，而认证这些智能体的基准测试依据每个模拟工具调用所报告的执行结果进行评分，默认工具确实执行了其接口所宣称的操作。我们调研的审计分类体系中没有任何类别涵盖这一假设，而隐藏在分数之下的缺陷会在每次重跑时持续存在。我们将工具的对外声明视为可执行契约，对照该契约检查其实现，并通过任务文件和评估器代码追踪每个分数的来源，直至由缺陷工具本应写入的状态所导出的判定结果。在四个基准测试的34个被审计的状态修改型工具中，我们在固定提交版本上确认了7个工具缺陷和1个评估器属性问题。在注入缺陷的实验中，检查器在25次触发中零误报，但仅标记了5个阴性对照中的2个，并且大多数被遗漏：在33个未被发现的记分缺陷中有29个存在覆盖该缺陷的契约条款，但没有任何探针能将其暴露出来。

    arXiv:2609.37315v1 Announce Type: cross  Abstract: Tool-using agents are entering settings where a wrong action carries real cost, and the benchmarks certifying them grade what each simulated tool call reports having done, assuming the tool did what its interface advertises. The audit taxonomies we survey publish no category for that assumption, and a defect beneath a score is present on every rerun. We treat a tool's advertised surfaces as an executable contract, check the implementation against it, and trace each score's provenance through the task files and evaluator code to the verdicts that derive from state a defective tool should have written. Across 34 audited mutating tools in four benchmarks we confirm seven tool defects and one evaluator property at pinned commits. On injected defects the checker raised no false positive in 25 flags, flagged 2 of 5 negative controls, and missed most: in 29 of 33 scored misses a clause covered the defect but no probe revealed it. The checker'
    
[^14]: SafeLLM4SE：基于大语言模型的软件工程系统的统计评估与报告

    SafeLLM4SE: Statistical Evaluation and Reporting for LLM-based Software Engineering Systems

    [https://arxiv.org/abs/2609.37294](https://arxiv.org/abs/2609.37294)

    SafeLLM4SE 提出了一套基于统计原则的评估方法与报告标准，将大语言模型在软件工程任务中的输出视为随机过程的实现，通过自适应采样、置信区间、分布感知统计比较和效应量来区分并评估系统的质量、稳定性与估计不确定性。

    

    大语言模型（LLM）在软件工程任务中的应用日益广泛，但其随机性行为给评估的有效性、可重复性和可比性带来了挑战。传统做法如报告单一输出、平均分数、best-of-N 或 pass@k 性能，可能会掩盖变异性和估计不确定性，从而可能导致对系统可靠性的误导性结论。本文提出了 SafeLLM4SE，这是一种用于基于大语言模型的软件工程系统的、遵循统计原则的评估实用方法学与报告标准。SafeLLM4SE 不将生成的输出视为确定性的产物，而是将其视为随机过程的实现，并区分质量、稳定性和估计不确定性三个维度。它将自适应采样与置信区间、考虑分布的统计比较、效应量以及涵盖模……的最低报告标准相结合。

    arXiv:2609.37294v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used for software engineering tasks, yet their stochastic behavior challenges the validity, reproducibility, and comparability of their evaluations. Conventional practices such as reporting a single output, an average score, best-of-N, or pass@k performance can obscure variability and estimation uncertainty, potentially leading to misleading conclusions about system reliability. This article presents SafeLLM4SE, a practical methodology and reporting standard for statistically principled evaluation of LLM-based software engineering systems. Rather than treating generated outputs as deterministic artifacts, SafeLLM4SE treats them as realizations of a stochastic process and distinguishes quality, stability, and estimation uncertainty. It combines adaptive sampling with confidence intervals, distribution-aware statistical comparisons, effect sizes, and a minimum reporting standard covering mode
    
[^15]: CRJudgeBench：AI能否检测出看似合理但无效的代码审查？

    CRJudgeBench: Can AI Detect Plausible but Invalid Code Reviews?

    [https://arxiv.org/abs/2609.37216](https://arxiv.org/abs/2609.37216)

    该论文提出了包含1199个实例的CRJudgeBench基准，用于评估AI能否识别看似合理但技术上错误的代码审查评论，并设计了基于仓库上下文、主动收集代码证据进行验证的智能体裁判Sentinel。

    

    大型语言模型能够生成看似合理的代码审查评论，但这些评论可能包含技术上不正确的断言，从而误导开发者。我们研究技术可信度判断问题：即确定一条审查评论的核心技术断言是否正确，并适用于其仓库上下文中被审查的代码。现有的代码审查基准主要评估评论生成、问题发现或整体评论质量，但并未直接评估智能体能否判断单条审查评论的技术可信度。为填补这一空白，我们提出了CRJudgeBench，这是一个包含1199个实例的基准数据集，由真实的拉取请求和经专家验证的扰动构造而成，同时涵盖可信评论与看似合理但实际不可信的评论。我们进一步提出了Sentinel，一种基于仓库上下文的智能体裁判，它在做出判断之前会主动收集代码证据来验证审查评论。从Qwen3开始

    arXiv:2609.37216v1 Announce Type: cross  Abstract: Large language models can generate plausible code-review comments, but such comments may contain technically incorrect claims that mislead developers. We study technical trustworthiness judgment: determining whether a review comment's core technical claims are correct and applicable to the reviewed code in its repository context. Existing code-review benchmarks primarily evaluate review generation, issue discovery, or general comment quality, but do not directly assess whether an agent can determine the technical trustworthy of an individual review comment. To fill this gap, we introduce CRJudgeBench, a benchmark of 1199 instances constructed from real pull requests and expert-verified perturbations, covering both trustworthy and plausible but untrustworthy comments. We further present Sentinel, a repository-grounded agentic judge that actively gathers code evidence to verify review comments before making judgments. Starting from Qwen3
    
[^16]: LoLBench：基于大型软件系统上长周期提案的编码智能体评估

    LoLBench: Evaluating Coding Agents with Long-Horizon Proposals on Large Software Systems

    [https://arxiv.org/abs/2609.37143](https://arxiv.org/abs/2609.37143)

    LoLBench是一个多语言基准测试，通过让编码智能体在大型软件系统上完成从人工编写的增强提案到代码实现的完整流程，首次同时评估其将用户意图转化为规范的感知能力和生成正确代码修改的实现能力。

    

    现代编码智能体能够交付越来越大规模的仓库级代码变更，近期的基准测试也反映了这一趋势，强调具有大型参考实现的长周期任务。许多基准测试评估的是编码智能体的实现能力，即根据详细规范生成正确的代码修改。然而，实际的模块化开发任务还需要感知能力，即将用户意图和高层设计落地转化为规范。我们提出LoLBench，通过在大型软件系统上执行从提案到实现的完整流程来评估这两种能力。这是一个多语言基准测试，涵盖五个领域中29个软件系统上的100个任务。每个任务提供一份由人工编写的增强提案，其中包含用户意图和高层设计。平均而言，提案约含5,000个单词，软件系统包含240万行源代码（LoC），实现拉取请求（PR）所修改的代码量约……（原文在此处截断）

    arXiv:2609.37143v1 Announce Type: cross  Abstract: Modern coding agents can deliver increasingly large repository-level changes, and recent benchmarks reflect this by emphasizing long-horizon tasks with large reference implementations. Many benchmarks evaluate coding agents' implementation capability to produce correct code edits from detailed specifications. However, practical modular development tasks also require the perception capability of grounding user intent and high-level design to derive a specification. We introduce LoLBench to evaluate both capabilities through the entire proposal-to-implementation process on large software systems. It is a multilingual benchmark of 100 tasks across 29 software systems in five domains. Each task provides a human-written enhancement proposal with user intent and high-level design. On average, proposals contain about 5,000 words, software systems contain 2.4 million source lines of code (LoC), and implementation pull requests (PRs) change app
    
[^17]: GitCF：通过利用提交间的多重相似性减少不完整变更

    GitCF: Reducing Incomplete Changes by Exploiting Multiple Similarities Among Commits

    [https://arxiv.org/abs/2609.37045](https://arxiv.org/abs/2609.37045)

    提出GitCF方法，通过利用开发者当前变更与历史提交之间的多重相似性来推荐应一并修改的位置，从而减少不完整变更。

    

    软件开发人员常常难以识别其变更所影响的所有位置，不完整的变更往往由此类遗漏而产生。为了缓解这一问题，已有若干方法被提出，用于推荐应与开发者当前变更一并修改的额外位置。这类技术中的一大类别从代码库的版本历史中挖掘共同变更规则，但由于它们仅依赖于哪些元素曾被一起修改，因此无法推荐很少发生共同变更的元素，并且忽略了伴随提交产生的文本信息。第二类技术虽然确实利用了文本信息，但其文本信息来源于源代码或变更请求，而非提交历史，并且不以开发者的当前变更作为输入。为了解决这些局限性，我们提出了GitCF，一种变更推荐方法，它通过比较开发者的当前变更与（原文在此处被截断）

    arXiv:2609.37045v1 Announce Type: new  Abstract: Software developers often struggle to identify all locations that their changes affect, and incomplete changes frequently result from these omissions. To mitigate this problem, several approaches have been proposed to recommend additional locations that should be modified together with a developer's current change. A large class of these techniques mines co-change rules from repository revision histories, but because they rely solely on which elements have been modified together, they cannot recommend elements that have rarely co-changed, and they disregard the textual information that accompanies commits. A second class of techniques does exploit textual information, but it derives it from the source code or from a change request rather than from commit history, and it does not use the developer's current change as its input. To address these limitations, we propose GitCF, a change recommendation method that compares a developer's curre
    
[^18]: 跨组织SysML模型集成：挑战与AI支持任务的调查研究

    Cross-Organizational SysML Model Integration: A Survey of Challenges and AI-Supported Tasks

    [https://arxiv.org/abs/2609.37000](https://arxiv.org/abs/2609.37000)

    该论文通过对29位MBSE利益相关者的问卷调查，首次系统揭示了跨组织SysML模型集成是一个涉及语义、行为、可追溯性和交换互操作性的多维度对齐问题，并评估了六类AI支持任务的作用及相关认知差异。

    

    跨组织协作被广泛认为是基于SysML的基于模型的系统工程（MBSE）的一项关键承诺，然而从业者在交换和集成系统模型时仍然面临持续的挑战。与此同时，大型语言模型（LLM）提升了人们对AI辅助模型理解与集成的期望，但其可靠性以及所需的人工监督仍然是挑战。本文报告了一项在线问卷调查的结果，调查对象为29位参与跨组织协作的MBSE利益相关者。受访者采用五点李克特量表，对八个预定义的集成挑战类别和六种AI支持的任务类型进行了评分。结果表明，利益相关者将模型集成视为一个涉及语义、行为、可追溯性和交换互操作性等多个维度的对齐问题，且这些认知因组织角色和参与集成的频率而有所差异。

    arXiv:2609.37000v1 Announce Type: cross  Abstract: Cross-organizational collaboration is widely regarded as a key promise of SysML-based Model-Based Systems Engineering (MBSE), yet practitioners still face persistent challenges when exchanging and integrating system models. In parallel, Large Language Models (LLMs) raise expectations for AI-assisted model understanding and integration, while reliability and required human oversight continue to pose challenges. This paper reports the results of an online questionnaire survey with 29 MBSE stakeholders involved in cross-organizational collaboration. Respondents rated eight predefined integration challenge categories and six AI-supported task types on five-point Likert scales. The results indicate that stakeholders perceive model integration as a multi-dimensional alignment problem across semantics, behavior, traceability, and exchange interoperability. These perceptions vary by organizational role and frequency of integration involvement.
    
[^19]: XRepoSkill：为软件工程智能体学习可迁移技能

    XRepoSkill: Learning Transferable Skills for Software Engineering Agents

    [https://arxiv.org/abs/2609.36807](https://arxiv.org/abs/2609.36807)

    提出XRepoSkill方法，通过对比同一智能体解决同一问题时的成功与失败轨迹分歧点，提取带可执行谓词的可验证技能规则，从而解决软件工程智能体技能难以跨仓库迁移的问题。

    

    软件工程智能体越来越多地利用从以往经验中提炼出的可复用技能来解决仓库级别的问题，但这类技能往往难以在不同仓库之间迁移。一个核心挑战在于：成功轨迹中出现的某种行为并不一定对成功结果负责——它可能真正有用，也可能只是偶然现象，或仅仅是模型的一种习惯性做法。我们提出XRepoSkill，这是一种基于轨迹学习可迁移技能的方法。我们将技能表示为一组规则，每条规则规定了在问题解决过程中应采取何种行动以及何时采取该行动。XRepoSkill首先对比同一智能体在同一问题上成功与失败的轨迹，并从二者执行路径产生分歧之处推导出候选规则。每条规则都配有一个可执行的谓词，使其所规定的行为能够在其他轨迹上进行系统性评估。一条规则只有经过验证……

    arXiv:2609.36807v1 Announce Type: new  Abstract: Software engineering agents increasingly use reusable skills distilled from prior experience to resolve repository-level issues, yet such skills often fail to transfer across repositories. A central challenge is that a behavior appearing in a successful trajectory is not necessarily responsible for the successful outcome: it may be genuinely useful, merely incidental, or simply a recurring habit of the model. We introduce XRepoSkill, a trajectory-based approach for learning transferable skills. We represent a skill as a collection of rules, each specifying what action to take and when to take it during issue resolution. XRepoSkill first contrasts successful and failed trajectories of the same agent on the same issue and derives candidate rules from where their execution paths diverge. Each rule is paired with an executable predicate that enables its prescribed behavior to be evaluated systematically on other trajectories. A rule is verif
    
[^20]: GitHarness：用 Git 初始化你的工具工作记忆，实现用户需求的永续管理

    GitHarness: Git Init Your Harness Working Memory for Perpetual User Requirements

    [https://arxiv.org/abs/2609.36789](https://arxiv.org/abs/2609.36789)

    提出 GitHarness，一个可插拔的 Git 风格框架，将需求状态与工具工作状态组织成可分支的版本历史，使 LLM 智能体在长周期人机协作中能够持续跟踪动态需求变化并实现局部更新而非全局重写。

    

    基于 LLM 的智能体越来越多地与用户协作完成长周期任务，通过大量的搜索、推理和执行积累证据、代码和草稿。当用户检视这些结果时，可能会补充缺失的信息（需求补全）、引入新的需求（需求启发），或修改已有需求（需求变更）。这些变更往往只影响已积累工作的一部分，但智能体可能沿用过时的信息，或将局部修订扩大为全局重写。现有方法要么在不决定先前工作应如何改变的情况下澄清当前意图，要么在固定目标下复用执行历史。我们通过将动态需求协作形式化为需求跟踪与局部更新的联合任务来填补这一空白。我们提出了 GitHarness，一个可插拔的 Git 风格框架，它将需求状态及其对应的工具工作状态组织成可分支的版本历史。

    arXiv:2609.36789v1 Announce Type: cross  Abstract: LLM-based agents increasingly collaborate with users on long-horizon tasks, accumulating evidence, code, and drafts through extensive search, reasoning, and execution. As users inspect these results, they may supply missing information requirement completion, introduce new requirements requirement elicitation, or revise existing ones requirement shift. These changes often affect only part of the accumulated work, yet agents may carry forward obsolete information or turn local revisions into global rewrites. Existing approaches clarify current intent without determining how prior work should change, or reuse execution histories under a fixed objective. We address this gap by formulating dynamic-requirement collaboration as joint requirement tracking and local update. We introduce GitHarness, a pluggable Git-style framework that organizes requirement states and their corresponding harness work states into a branchable version history. A 
    
[^21]: 编辑器仅有只读权限：扩散语言模型中的正确性信号

    The Editor Has Read-Only Access: Correctness Signals in Diffusion Language Models

    [https://arxiv.org/abs/2609.36783](https://arxiv.org/abs/2609.36783)

    扩散语言模型的内部激活中编码了代码正确性信号，线性探针可有效检测该信号，但利用其进行生成引导并未带来可靠的性能提升。

    

    扩散语言模型通过反复更新部分掩码的序列来生成代码。我们探究其内部激活是否编码了代码正确性，以及该信息能否用于改进生成。在六个扩散模型上，线性探针能够区分通过和失败的生成尝试，且最强的读取信号通常出现在早期层之后。采用小幅度语义变异的对照实验支持了这些信号与正确性相关，而不仅仅是与表面风格相关。与模型置信度相比，探针的点估计并未表现出一致的优势。在所测试的引导设置中，向残差流中加入探针导出的方向并未带来可靠的改进，而施加相反方向则会降低性能。我们将这些观察结果与关于统计显著性或普遍无法引导生成的断言区分开来。补充方法、存档结果和代码记录了所测试的干预措施及其局限性。

    arXiv:2609.36783v1 Announce Type: new  Abstract: Diffusion language models generate code by repeatedly updating a partially masked sequence. We ask whether their internal activations encode code correctness and whether that information can improve generation. Across six diffusion models, linear probes distinguish passing from failing attempts, with the strongest reads generally appearing beyond the early layers. Controls using small semantic mutations support a connection to correctness rather than surface style alone. In comparisons with model confidence, probe point estimates offer no consistent advantage. Adding a probe-derived direction to the residual stream does not yield a dependable improvement in the tested steering settings, while the opposite direction degrades performance. We distinguish these observations from claims about statistical significance or a general inability to steer. Supplementary methods, archived results, and code document the tested interventions and the li
    
[^22]: 智能体能为智能体设计程序库吗？

    Can Agents Design Libraries for Agents?

    [https://arxiv.org/abs/2609.36730](https://arxiv.org/abs/2609.36730)

    该论文提出LibraryDesignBench基准来评估智能体为其他智能体设计代码库的能力，发现智能体设计者在大多数任务上能复现人类生产级库的抽象，但下游智能体对库的利用不足，经常重新实现库中已有的功能。

    

    智能体越来越多地基于其他智能体编写的代码进行构建，但它们往往重新实现而非复用现有代码，导致后续智能体必须工作的代码库不断膨胀。为了衡量智能体为其他智能体设计库的能力，我们提出了LibraryDesignBench，这是一个两阶段的基准测试：智能体根据一份规范实现一个功能齐全的库，该规范定义了所需能力和潜在用例，但不规定具体设计。我们通过来自不同模型家族的三个用户智能体所编写程序的正确性和简洁性来评估该库。该基准测试涵盖15个库设计任务中的242个经专家验证的编程问题，涉及四种编程语言。在15个任务中的11个上，智能体设计者能够复现人工编写的生产级库中的抽象。下游智能体对智能体编写的库和人工编写的库都有采用，但利用不足，常常重新实现库中已提供的功能。

    arXiv:2609.36730v1 Announce Type: new  Abstract: Agents increasingly build on code written by other agents, and they reimplement rather than reuse, growing the codebases later agents must work in. To measure how well agents design libraries for other agents, we introduce LibraryDesignBench, a two-phase benchmark in which an agent implements a full-featured library from a specification that defines required capabilities and potential use cases without prescribing the design. We evaluate the library through the correctness and simplicity of programs written by three user agents from different model families. The benchmark spans 242 expert-validated programming problems across 15 library-design tasks in four languages. On eleven of the fifteen tasks, agent designers reproduce the abstractions of the human-written production library. Downstream agents adopt agent- and human-written libraries alike but underuse them, reimplementing capabilities the library already provides. Our failure anal
    
[^23]: CTE-Bench：面向有状态软件模拟器的反事实轨迹评估

    CTE-Bench: Counterfactual Trace Evaluation for Stateful Software Simulators

    [https://arxiv.org/abs/2609.36647](https://arxiv.org/abs/2609.36647)

    提出CTE-Bench基准，通过要求模型预测代码修补或状态覆盖等干预后有状态服务对40个固定未来调用的响应，并以实际执行验证预测结果，从而在不让模型选择行动的情况下评估其模拟有状态软件反事实行为的能力。

    

    编码智能体会改变正在运行的软件：它们修补服务的代码或覆盖其存储的状态，然后基于自己对服务随后将如何响应的预期来采取行动。错误的预期可能要到多次调用之后才会显现出来。函数级的代码执行基准测试忽略了持久化的服务状态，而智能体基准测试则评分智能体采取的行动或其到达的最终状态。我们提出CTE-Bench，用于衡量模型能否预测一次干预将如何改变有状态服务的未来行为，而无需模型选择行动。每个场景为模型提供Python服务代码、干预前观察到的调用与响应、干预本身（源代码编辑或状态覆盖），以及40个固定的未来调用；模型需要预测每一次未来响应，预测结果通过实际执行服务来验证。三种记忆协议控制模型是否能看到正确的早期响应、完全看不到……（摘要原文在此处截断）

    arXiv:2609.36647v1 Announce Type: new  Abstract: Coding agents change running software: they patch a service's code or overwrite its stored state, and then act on their own expectation of how the service will respond afterwards. A wrong expectation may surface only several calls later. Function-level code-execution benchmarks omit persistent service state, and agent benchmarks score the actions an agent takes or the final state it reaches. We introduce CTE-Bench, which measures whether a model can predict how an intervention changes a stateful service's future behavior, without asking it to choose actions. Each scenario gives the model Python service code, the calls and responses observed before the intervention, the intervention itself (a source edit or a state overwrite), and 40 fixed future calls; the model predicts every future response, and predictions are checked by executing the service. Three memory protocols control whether the model sees the correct earlier responses, none of
    
[^24]: WitnessGym：面向缺陷见证构建任务的编码智能体基准测试

    WitnessGym: Benchmarking Coding Agents on the Construction of Bug Witnesses

    [https://arxiv.org/abs/2609.36635](https://arxiv.org/abs/2609.36635)

    WitnessGym通过向真实项目自动注入缺陷、重建项目并验证见证可暴露性，构建了包含1,300个用例的缺陷验证基准，用于评测编码智能体生成可执行缺陷见证的能力。

    

    缺陷验证要求编码智能体为一个已报告的缺陷生成可执行的见证（witness）。该见证将具体输入与测试工具框架相结合，并在执行过程中暴露出缺陷行为。这样的证据能够使审计发现具有可操作性，然而当基准评测用例复用公开的历史缺陷与见证、或需要人工构建时，评测就会变得十分困难。我们提出了WitnessGym，一个通过缺陷注入自动构建缺陷验证基准的框架。它将缺陷注入到真实项目中被测试覆盖的代码路径上，重新构建每个项目，并保留能被构建时见证所暴露的用例。缺陷规范与执行适配器使其可扩展至更多缺陷类型和编程语言。保缺陷变换能够在保持见证行为不变的前提下改变周围代码结构。基于带有测试套件的真实Java项目，WitnessGym自动构建了1,300个基准用例。注入的补丁（原文此处截断）……

    arXiv:2609.36635v1 Announce Type: cross  Abstract: Bug validation asks a coding agent to produce an executable witness for a reported bug. The witness combines a concrete input with a testing harness and exposes faulty behavior during execution. Such evidence makes audit findings actionable, yet benchmark evaluation is difficult when cases reuse public historical bugs and witnesses or require manual construction. We present WitnessGym, an automated framework for constructing bug-validation benchmarks through bug injection. It injects bugs into test-reached paths of real projects, rebuilds each project, and retains cases exposed by a construction-time witness. Bug specifications and execution adapters allow extension to additional bug types and languages. Bug-preserving transformations vary the surrounding structure while preserving the witness behavior. Based on real-world Java projects with test suites, WitnessGym automatically constructs 1,300 benchmark cases. The injected patches re
    
[^25]: LatentSift：面向软件工程智能体Token高效验证的策略状态过滤

    LatentSift: Policy-State Filtering for Token-Efficient Verification of Software Engineering Agents

    [https://arxiv.org/abs/2609.36371](https://arxiv.org/abs/2609.36371)

    LatentSift复用策略自身生成的隐藏状态构建正负样本库来过滤候选轨迹，无需额外的LLM验证推理，从而大幅降低软件工程智能体验证阶段的token消耗。

    

    测试时扩展通过生成多个候选轨迹并从中选出最佳轨迹来提升软件工程智能体的表现。然而，对这些长交互进行验证与选择所消耗的token可能与生成本身一样多。现有的混合工作流先使用基于LLM的无执行验证器在运行测试前过滤候选，这会对每条轨迹额外增加一次模型推理。我们提出LatentSift，这是一种无需消耗token、也无需执行的过滤器，它利用策略在生成候选时已经产生的隐藏状态来替代这一第一阶段。LatentSift通过推理、观察和函数调用状态来表示每个候选，将其与策略训练期间从成功和不成功轨迹中收集的正、负样本库进行比较，并将所得的距离分数与一个学习到的线性分数相融合，从而为后续基于执行的阶段保留有前景的候选。

    arXiv:2609.36371v1 Announce Type: cross  Abstract: Test-time scaling improves software engineering agents by generating multiple candidate trajectories and selecting the best one. Verifying and selecting among these long interactions can consume as many tokens as generation itself. Existing hybrid workflows first apply an LLM-based execution-free (EF) verifier to filter candidates before running tests, which adds another model pass over every trajectory. We introduce LatentSift, a token-free and execution-free filter that replaces this first stage with hidden states the policy already produces while generating the candidates. It represents each candidate through its reasoning, observation, and function-call states, compares them with positive and negative banks of such states collected from successful and unsuccessful trajectories during policy training, and fuses the resulting distance scores with a learned linear score to retain promising candidates for the execution-based stages. On
    
[^26]: 在科学用户设施中将AI智能体部署到生产环境的策略

    Strategies for Deploying AI Agents in Production at Scientific User Facilities

    [https://arxiv.org/abs/2609.36362](https://arxiv.org/abs/2609.36362)

    基于在先进光子源（APS）部署大语言模型驱动智能体的实践经验，本文提出了在光源、中子源等科学用户设施中将AI智能体推向生产环境的可复用策略与设计原则。

    

    智能体人工智能正在超越研究演示阶段，走向科学用户设施的生产环境应用，这些设施包括光源、中子源、纳米科学中心和自主实验室。其科学价值不仅限于提高通量。智能体能够执行校准、测量执行和质量控制等可重复任务，以及将数据转化为可审查证据的初步分析，使科学家能够专注于假设、意外观察和结果解释。基于在先进光子源部署大语言模型驱动智能体的实际经验，本观点文章提炼出实用策略，重点强调可在不同仪器和设施之间复用的要素。我们讨论了用于束线控制、设施知识检索和数据分析的智能体框架，同时保持底层设计原则独立于任何特定实现。这些原则涵盖推理端点等关键要素。

    arXiv:2609.36362v1 Announce Type: new  Abstract: Agentic artificial intelligence (AI) is moving beyond research demonstrations toward production use at scientific user facilities, including light sources, neutron sources, nanoscience centers, and autonomous laboratories. Its scientific value extends beyond increasing throughput. Agents can perform repeatable tasks in calibration, measurement execution, and quality control, as well as initial analyses that turn data into reviewable evidence, allowing scientists to focus on hypotheses, unexpected observations, and interpretation. Drawing on deployments of LLM-driven agents at the APS, this perspective distills practical strategies with an emphasis on elements that can be reused across instruments and facilities. We discuss agent harnesses for beamline control, facility knowledge retrieval, and data analysis while keeping the underlying design principles independent of any specific implementation. These principles cover inference endpoint
    
[^27]: 迈向面向数据系统的AI软件工厂

    Towards an AI Software Factory for Data Systems

    [https://arxiv.org/abs/2609.36323](https://arxiv.org/abs/2609.36323)

    该论文提出构建覆盖软件开发生命周期全阶段（目标定位、编码、审查、运维）的AI软件工厂，通过元数据驱动自我改进，在微软数十个代码仓库的实际部署中实现了相比代理式编程3倍的工程效率和高达22倍的token效率。

    

    AI辅助编程工具显著加速了编程环节，但在端到端软件开发生命周期（SDLC）中的整体影响却十分有限——这正是阿姆达尔定律效应！在本文中，我们讨论了构建AI软件工厂的进展，该工厂能够加速SDLC的所有阶段——目标定位、编码、审查和运维。AI软件工厂会产生元数据排放流，通过微调模型权重和更新我们的世界模型（一个丰富的数据基底）来实现自我改进。我们专注于数据系统领域以及一类重要的演化式编程任务（即具有可度量目标、可进行爬山优化的任务），并报告了两方面成果：1）在微软的大规模部署（覆盖数十个代码仓库），实现了相比代理式编程3倍的工程效率和高达22倍的token效率提升；2）若干开放性挑战。

    arXiv:2609.36323v1 Announce Type: new  Abstract: AI-assisted coding tools deliver significant acceleration of coding, but only limited impact across the end-to-end software development lifecycle (SDLC)--an Amdahl's law effect!   In this paper, we discuss our progress towards building an AI SW Factory that accelerates all the stages of SDLC-Targeting, Coding, Reviewing, and Ops. The AI SW Factory produces a metadata exhaust that enables self-improvement by fine-tuning model weights and updating our World Model (a rich data substrate).   We focus on Data Systems and the important class of Evolutionary Coding Tasks (i.e., those with a measurable objective to hill-climb) and report on 1) scaled deployments at Microsoft (tens of repositories) leading to 3x engineering efficiency above agentic coding and up to 22x token efficiency, and 2) several open challenges.
    
[^28]: 多少提示才足够？一种针对大语言模型少样本示例的黑盒最小化方法

    How Much Prompt Is Enough? A Blackbox Minimization of Few-Shots in LLMs

    [https://arxiv.org/abs/2609.36289](https://arxiv.org/abs/2609.36289)

    该论文提出了一个黑盒提示最小化框架，实验证明少样本提示可在完全保持输出保真度的情况下平均缩减65.3%的字符量。

    

    提示是引导大语言模型（LLM）行为的主要机制。然而，提示的内部结构和因果层级仍然鲜为人知：哪些部分是因果必要的、哪些是冗余的，仍是一个悬而未决的问题。这种不透明性可能带来严重后果：在关键软件系统中，微妙的提示变化可能会悄然改变模型输出，而工程师缺乏评估提示可靠性的技术。我们提出了一个黑盒提示最小化框架，能够将少样本提示缩减至其必要的最小子集。我们通过案例研究将该框架应用于一个少样本学习系统，并展示了该框架所能提供的洞察。实验表明，少样本示例在字符数量上平均可减少65.3% ± 15.8%，同时完全保留命题输出的保真度。模型会优先保留逻辑标识符和常量。

    arXiv:2609.36289v1 Announce Type: cross  Abstract: Prompts are the primary mechanism for directing the behavior of large language models (LLMs). Yet the internal structure and causal hierarchy of prompts remain poorly understood: which parts are causally necessary and which are redundant is an open question. This opacity can have severe consequences. Subtle prompt variations can silently shift model outputs in critical software systems, and engineers lack techniques to reason about prompt reliability.   We present \framework, a blackbox prompt-minimization framework that reduces few-shot prompts to their necessary minimal subset. We use a case study to apply \framework to a few-shot learning system and demonstrate the insights that this framework can provide.   Our experiments show that few-shot exemplars can be reduced by a mean of 65.3\%~$\pm$~15.8\% in character count while fully preserving propositional output fidelity. The models preferentially retain logical identifiers and const
    
[^29]: SCOUT：协同推理与工具使用以保障计算机使用安全

    SCOUT: Synergizing Reasoning and Tool-Use for Computer-Use Safety

    [https://arxiv.org/abs/2609.36201](https://arxiv.org/abs/2609.36201)

    SCOUT提出了一种两阶段的代理式安全验证器，通过将推理密集的评分标准生成与工具密集的证据收集相结合，能够有效检测计算机使用代理在执行任务时产生的细微且隐蔽的有害行为。

    

    计算机使用代理（CUA）虽然能够在日常和专业工作流程中完成计算机任务，但即使在良性指令和环境条件下也可能造成意外的伤害。然而，检测此类伤害仍然极具挑战性。首先，这需要细致的、针对特定任务的推理：仅依靠通用安全标准进行判断的验证器往往会忽略许多重要但微妙的有害行为。其次，这需要主动调查：历史轨迹截图只显示了代理做了什么，但并不总能反映环境中的实际变化，因此仅依赖截图的LLM-as-a-judge验证器可能无法确定行动的实际后果。为了应对这些挑战，我们提出了SCOUT，一个两阶段的代理式安全验证器，它将推理密集的评分标准生成与工具密集的证据收集相协同。首先，SCOUT评分标准生成器对任务和代理的执行轨迹进行深入的推理……

    arXiv:2609.36201v1 Announce Type: cross  Abstract: Computer-use agents (CUAs), while capable of completing computer tasks in everyday and professional workflows, can cause unintended harm even under benign instructions and environments. However, detecting such harm remains challenging. First, it requires careful, task-specific reasoning: verifiers guided only by general safety criteria often overlook many important but subtle harmful behaviors. Second, it requires active investigation: past trajectory screenshots show what the agent did but not always what actually changed in the environment, so LLM-as-a-judge verifiers that rely on screenshots alone may be unable to determine the actual consequences of actions. To address these challenges, we introduce SCOUT, a two-stage agentic safety verifier that synergizes reasoning-intensive rubric generation with tool-intensive evidence gathering. First, our SCOUT rubric generator extensively reasons over the task and the agent's trajectory to d
    
[^30]: Assay：随代码衰减的断言。用于可问责的AI辅助软件交付的内容寻址证据图

    Assay: Claims That Decay With the Code. Content-Addressed Evidence Graphs for Accountable AI-Assisted Software Delivery

    [https://arxiv.org/abs/2609.36170](https://arxiv.org/abs/2609.36170)

    Assay系统将AI编码代理做出的每项断言（如测试通过、无密钥泄露）绑定到其覆盖代码的依赖锥的Merkle哈希上，使断言在代码或其依赖发生变化时精确失效，从而为AI辅助软件交付提供可验证、可问责的证据图机制。

    

    AI编码代理以两种相互耦合的方式失败：它们花费大部分上下文窗口来重新发现代码的位置，并且当工作变得困难时，它们会在没有证据的情况下断言成功。仓库索引用廉价的上下文解决了第一个问题，而带有对抗性审查的编排框架则用问责制解决了第二个问题。两者实际上是在两个时间尺度上描述同一个对象——代码库的结构：即代码现在什么是真的，以及什么曾被验证为真的、在哪个修订版本、由谁验证。Assay使这一观察具有可操作性。代理做出的每一个断言（测试通过、无密钥泄露、行为保持不变）都与其覆盖代码的依赖锥的Merkle哈希相绑定，因此当该代码或其依赖的任何内容发生变化时，该断言恰好在那一刻失效。我们证明了这种绑定是可靠且最小的，并且变更的影响范围恰好是它所失效的断言集合。在该图上，我们放置了与风险成比例的证据……

    arXiv:2609.36170v1 Announce Type: new  Abstract: AI coding agents fail in two coupled ways. They spend most of their context window rediscovering where things live, and they assert success without evidence when the work gets hard. Repository indexes address the first with cheap context, and orchestration frameworks with adversarial review address the second with accountability. Both describe the same object, the structure of the codebase, at two timescales: what is true of the code now, and what was verified to be true, at which revision, by whom. Assay makes that observation operational. Every claim an agent makes (tests pass, no secrets, behavior preserved) is bound to the Merkle hash of the dependency cone of the code it covers, so the claim is stale exactly when that code or anything it depends on changes. We show the binding is sound and minimal, and that the blast radius of a change is precisely the set of claims it invalidates. On the graph we place a risk-proportional evidence 
    
[^31]: 从死代码与静态需求到可运行引擎：基于编程智能体的软件复活

    From Dead Code and Static Requirements to Working Engines: Software Revival with Coding Agents

    [https://arxiv.org/abs/2609.36161](https://arxiv.org/abs/2609.36161)

    该论文提出ReviveBench基准，通过隐藏验证器评估编程智能体复活无法运行的软件（涵盖依赖不兼容、删除核心模块、遗留构建及GPU基础模型等十个任务）以及从开放规范重建工业软件引擎的能力，结果显示最强模型能够通过全部十个复活任务，且在防污染实验中表现稳健。

    

    编程智能体能否在保留底层方法的前提下，恢复已经无法运行的软件，并根据开放规范重建工业软件引擎？我们提出了ReviveBench，一个包含两个任务族的基准，通过针对原生执行环境、成熟工程工具或专门构建的参考实现进行校准的隐藏验证器进行评估。复活任务族包含十个任务，涉及依赖不兼容、核心模块被删除、遗留构建系统以及一个基于GPU的基础模型。每个起始工作区都无法通过验证，而所评估的最强模型在每个任务中至少有一次运行通过了全部十个任务。在污染控制实验中，标识符混淆将代码行与原始实现的相似度从0.51–0.96降低至0.03–0.44，但没有降低任何被评估模型的观察通过率。在创建于所述知识截止日期之后的代码库上，最强的模型……（原文摘要在此处被截断）

    arXiv:2609.36161v1 Announce Type: cross  Abstract: Can coding agents restore software that no longer runs while preserving its underlying methods, and reconstruct industrial software engines from open specifications? Here we introduce ReviveBench, a benchmark with two task families evaluated by hidden verifiers calibrated against native execution environments, established engineering tools, or purpose-built reference implementations. The revival family comprises ten tasks involving dependency incompatibilities, deleted core modules, legacy builds, and a GPU-based foundation model. Every starting workspace fails verification, and the strongest evaluated model passes all ten tasks in at least one run each. In contamination-control experiments, identifier obfuscation reduces line similarity to the original implementations from 0.51--0.96 to 0.03--0.44 without reducing the observed pass rate of any evaluated model. On repositories created after the stated knowledge cutoffs, the strongest m
    
[^32]: 无形的限速器：Scratch 中在“借来的时间”上运行

    The Invisible Throttle: Running on Borrowed Time in Scratch

    [https://arxiv.org/abs/2609.36152](https://arxiv.org/abs/2609.36152)

    本文揭示了 Scratch 中一条未记录的执行速度规则：脚本并非如普遍认为的“每帧迭代一次”，而是受全局重绘门控隐形限速——隐藏负责绘制的角色可让不绘制的循环快达 77,000 倍。

    

    Scratch 拥有 1.35 亿注册用户（其中大多数是儿童）和 1.64 亿个共享项目。人们学到的关于脚本速度的知识只有一句话：一个循环每帧迭代一次。遗憾的是，这句话描述的只是特例。在公开的虚拟机中，一帧会反复执行脚本，直到出现以下情况之一：有可见对象请求屏幕重绘、没有脚本仍在运行，或者已耗尽该帧实际时间的四分之三。因此，一个不进行绘制的循环，其运行节奏由其他可见内容和机器本身决定。在一个双角色项目中，隐藏负责移动的角色，另一个角色的计数循环在笔记本电脑上的运行速度会快 77,000 倍。没有任何文档说明这一规则。我们的关键观察是：重绘门控是整个运行时共享的一个标志，因此可以在不改动循环代码的情况下改变循环的速度——隐藏负责绘制的角色、更改机器的时间预算，或者在没有渲染器的工具中运行程序。ThrottleCheck 实现……（原文摘要在此截断）

    arXiv:2609.36152v1 Announce Type: new  Abstract: Scratch has 135 million registered users, most of them children, and 164 million shared projects. What they are taught about the speed of a script fits in one sentence: a loop iterates once per frame. Unfortunately, that sentence describes the exception. In the public virtual machine a frame repeats the scripts until something visible asks the screen to redraw, no script is left running, or three quarters of the frame's wall-clock time are spent. Hence a loop that does not draw is paced by whatever else is visible and by the machine. Hide the moving sprite of a two-sprite project, and the other's counting loop runs 77,000 times faster on a laptop. No documentation states the rule. Our key observation is that the redraw gate is one flag for the whole runtime, so a loop's speed can be changed without touching its code: hide the sprite that draws, change the machine's budget, or run the program in a tool with no renderer. ThrottleCheck impl
    
[^33]: 面向云原生“架构即代码”的实时架构模型：Kubernetes一致性检查的早期结果

    Live Architecture Models for Cloud-Native Architecture-as-Code: Early Results from Kubernetes Conformance Checking

    [https://arxiv.org/abs/2609.36148](https://arxiv.org/abs/2609.36148)

    本文提出“实时架构模型”方法，通过Kubernetes部署语言（KDL）子集和VS Code原型Archer将可编辑架构模型与Kubernetes运行时事实相连接，利用快照恢复和只读一致性检查来检测集群与架构意图之间的漂移不一致。

    

    基础设施漂移会使运行中的Kubernetes系统偏离其文档化的架构意图。本文研究了实时架构模型：即通过显式对应关系和周期性一致性检查与选定运行时事实相连接的可编辑架构表示。该方法通过Kubernetes部署语言（KDL）的一个Archer专用子集以及Archer——一个具有同步文本视图和图形视图的VS Code原型——进行实例化。快照恢复将选定的Kubernetes事实提取为KDL；周期性和按需触发的只读检查报告模型与集群之间的不一致性，但不会强制执行或修复部署状态。我们在三个经过功能筛选的Kubernetes示例应用上，依照作者自定义的协议评估了该方法的可行性，报告了快照恢复和选定不一致性检测的精确率与召回率。恢复得分可通过保存的基准真值和恢复的模型进行复现。

    arXiv:2609.36148v1 Announce Type: new  Abstract: Infrastructure drift can separate a running Kubernetes system from its documented architectural intent. This paper investigates live architecture models: editable architectural representations connected to selected runtime facts through explicit correspondences and recurring conformance checks. The approach is instantiated through an Archer-specific subset of Kubernetes Deployment Language (KDL) and Archer, a VS Code prototype with synchronized textual and graphical views. Snapshot recovery extracts selected Kubernetes facts into KDL; periodic and on-demand read-only checks report model-cluster inconsistencies without enforcing or repairing deployment state.   We assess feasibility on three feature-selected Kubernetes example applications under an author-defined protocol, reporting precision and recall for snapshot recovery and selected inconsistency detection. Recovery scores can be reproduced from saved ground-truth and recovered model
    
[^34]: 看不见的调度器：拖动角色就能改变Scratch程序的行为

    The Invisible Scheduler: Dragging a Sprite Can Change What a Scratch Program Does

    [https://arxiv.org/abs/2609.36147](https://arxiv.org/abs/2609.36147)

    本文发现Scratch中角色图层的堆叠顺序会隐性地决定并发脚本的启动顺序，导致“拖动角色即可改变程序行为”且该隐患会随项目保存，并提出StackSwap工具通过系统性枚举不同堆叠顺序下的运行结果来检测此类与顺序相关的隐性问题。

    

    数以千万计的儿童在Scratch中编程，其程序是并发的：当绿旗被点击时，每个角色的脚本同时启动并共享项目的状态。哪个脚本先启动是由角色的前后堆叠顺序决定的，而没有任何积木块会读取这个顺序。拖动一个角色会将其置于最前层，且该顺序会随项目一起保存。因此，同一个程序可能在作者的屏幕上正常运行，却在老师的屏幕上出错，即使积木块完全相同。我们的关键观察是：在文件其余部分固定的情况下，将启动脚本的角色在其各自位置上重新排列，恰好能产生其初始堆叠顺序的所有排列。在固定输入和种子下，未经修改的虚拟机可以复现其中每一种排列。StackSwap工具在这些顺序下运行项目——对于不超过五个角色的情况穷举所有顺序，超出范围时对冲突图的每个等价类取一个代表，若两者均不可行则进行抽样——并从四个观察维度比较各次运行的结果……

    arXiv:2609.36147v1 Announce Type: new  Abstract: Tens of millions of children program in Scratch, whose programs are concurrent: when the green flag is clicked, every sprite's scripts start together and share the project's state. Which script starts first is decided by the sprites' front-to-back stacking order, which no block reads. Dragging a sprite brings it to the front, and the order is saved with the project. Hence, a program can work on the author's screen and fail on the teacher's, with the same blocks. Our key observation is that, with the rest of the file fixed, rearranging the sprites that start scripts among their positions produces exactly the permutations of their initial stacking order. Under a fixed input and seed the unmodified virtual machine reproduces every one of them. StackSwap runs a project under these orders, all of them for up to five sprites, one per class of a conflict graph beyond, and a sample where neither is feasible, and compares the runs under four obse
    
[^35]: 纠正何时才能成为修复？工具使用型大语言模型内部干预的机制审计

    When Does Correction Become Repair? Mechanistic Auditing of Internal Interventions in Tool-Using LLMs

    [https://arxiv.org/abs/2609.36138](https://arxiv.org/abs/2609.36138)

    提出SAKIKO审计框架，揭示工具使用型大语言模型内部干预中“行为改变不等于修复”——即便干预带来净收益，也可能损害超过一半的原始决策，因此必须通过目的地解析验证来审计干预的真实效果。

    

    在调用外部工具之前，智能体型大语言模型必须在包含K个选项的动作空间中做出选择：执行调用、请求澄清、直接回答或拒绝回答。虽然内部激活引导可以改变这些执行前的决策，但传统的聚合指标掩盖了被改变的状态落向何处，以及它们造成了何种附带损害。我们提出了SAKIKO——一个审计框架，通过方向性错误发现、基于路由器的条件干预、目的地解析验证以及前瞻性冻结的统计许可来形式化“表征修复”这一概念。在When2Call和MetaTool数据集上对七个大语言模型的实验中，通道键控干预在五个模型中诱导出方向特定的净增益；在三次密封评估中，59个预算匹配的随机方向无一能达到校准后的目标增益。至关重要的是，目的地审计表明行为上的移动并不等同于修复：一个实现+55净增益的干预会破坏超过一半的原始决策（原文摘要在此处截断）。

    arXiv:2609.36138v1 Announce Type: new  Abstract: Before invoking external tools, an agentic LLM must select among a K-way action space: executing a call, seeking clarification, answering directly, or declining. While internal activation steering can alter these pre-execution decisions, conventional aggregate metrics obscure where altered states land and what collateral damage they inflict. We present SAKIKO, an auditing framework that formalizes representation repair via directional error discovery, router-conditioned intervention, destination-resolved verification, and prospectively frozen statistical licensing. Across seven LLMs on When2Call and MetaTool, channel-keyed interventions induce direction-specific net gains in five models; across three sealed evaluations, none of 59 budget-matched random directions matches calibrated target gain. Crucially, destination auditing shows that behavioral movement does not equal repair: an intervention achieving +55 net gain corrupts over half o
    
[^36]: 集体知识生产的不均衡衰退：生成式AI出现后Stack Overflow上的证据

    The Uneven Decline of Collective Knowledge Production: Evidence from Stack Overflow After Generative AI

    [https://arxiv.org/abs/2609.36069](https://arxiv.org/abs/2609.36069)

    通过分析ChatGPT发布后Stack Overflow上的200多万个问题，研究发现简单问题急剧减少而困难问题日益增多，表明生成式AI对集体知识生产造成了不均衡的侵蚀。

    

    生成式人工智能（Gen AI）正在重塑个人学习和工作的方式，但其对集体知识——即在线社区共同生产的知识共享体系——的影响仍鲜为人知。先前的研究已经记录了知识共享平台上参与度的总体下降，但目前尚不清楚哪些特定类型的知识最先流失。我们利用最大的软件工程在线社区之一Stack Overflow来研究这一问题，并将ChatGPT-3.5的发布视为一次自然冲击。通过分析2020年至2025年间发布的超过200万个问题，我们追踪了生成式AI发布后集体知识的两个维度——难度和数据可用性——如何发生变化。通过使用多种方法和稳健性检验，我们发现了一致的模式：简单问题急剧减少，而困难问题变得更加常见，这一模式得到了代码复杂度上升的印证。数据丰富的……

    arXiv:2609.36069v1 Announce Type: cross  Abstract: Generative AI (Gen AI) is reshaping how individuals learn and work, but its consequences for collective knowledge, the shared body of knowledge that online communities produce together, remain poorly understood. Prior work has documented an aggregate decline in participation on knowledge-sharing platforms, but it remains unclear which specific kinds of knowledge are being lost first. We study this question using Stack Overflow, one of the largest online communities for software engineering, treating the release of ChatGPT-3.5 as a natural shock. Analyzing over two million questions posted between 2020 and 2025, we track how two dimensions of collective knowledge, difficulty and data availability, change following Gen AI's release. Using diverse methods and robust checks, we find consistent patterns. Easy questions decline sharply while difficult questions become more common, a pattern corroborated by rising code complexity. Data-rich t
    
[^37]: Irene：基于结构保持符号规约的混合量子程序等价性验证

    Irene: Equivalence Checking of Hybrid Quantum Programs via Structure-Preserving Symbolic Reduction

    [https://arxiv.org/abs/2609.36065](https://arxiv.org/abs/2609.36065)

    Irene是一个通过门级代数化简、混合路径和类型化图同构验证以及密度核分析这三层推理，来实现有界混合量子程序等价性验证的结构保持符号规约框架。

    

    等价性验证对于验证混合量子程序的编译器变换至关重要，混合量子程序结合了量子操作、测量和经典控制。依赖于测量的控制限制了酉推理的能力，而经典结果与量子操作之间的依赖关系可能会扩大中间符号状态。我们提出了Irene，一个基于结构保持符号规约的有界混合量子程序等价性验证框架。该框架通过三个层次的推理逐步简化等价性义务：在门级，利用代数恒等式简化酉区域；在混合路径和（HPS）级，将简化的符号执行状态表示为类型化图，其同构性即可证明等价性；剩余的义务则由密度核处理，密度核刻画了输入密度算子到可观测输出的变换，从而实现比较。

    arXiv:2609.36065v1 Announce Type: cross  Abstract: Equivalence checking is essential for validating compiler transformations of hybrid quantum programs, which combine quantum operations, measurements, and classical control. Measurement-dependent control limits unitary reasoning, while dependencies between classical outcomes and quantum operations can enlarge intermediate symbolic states. We present Irene, an equivalence-checking framework for bounded hybrid quantum programs based on structure-preserving symbolic reduction. The framework progressively simplifies equivalence obligations through three levels of reasoning. At the gate level, algebraic identities simplify unitary regions. At the hybrid path-sum (HPS) level, reduced symbolic execution states are represented as typed graphs, whose isomorphism certifies equivalence. Remaining obligations are handled by density kernels that characterize transformations of input density operators into observable outputs, allowing comparison even
    
[^38]: 评估用于一次性代码搜索的仅名称目录路由方法

    Evaluating Name-Only Directory Routing for One-Shot Code Search

    [https://arxiv.org/abs/2609.35918](https://arxiv.org/abs/2609.35918)

    该论文证明，仅依据目录和文件名称的语言模型路由方法在一次性代码搜索中显著优于FTS5和rg等固定词法查询，在八位候选文件内实现0.465的文件召回率，且计算成本远低于扁平路径等探索性方法。

    

    找到正确的文件是编程智能体面临的一项早期挑战。我们测试语言模型能否通过跟随目录和文件名称，找到固定词法查询所遗漏的带标注代码文件。在来自11个代码仓库（固定于修复前提交点）的82个经审计的议题中，仅名称目录路由在八位候选文件内找回了0.465的黄金文件，相比之下FTS5为0.352，固定的全议题rg查询为0.245。相对FTS5的配对增益为0.113（95%仓库聚类自助置信区间为0.053至0.168）。在共享的16K令牌上下文预算下，在55个标注完全对齐的案例中，路由交付了0.443的标注行，而FTS5为0.246。在相同的八文件限制下，将路由与FTS5结合达到了0.491的文件召回率，但其相对仅用路由的增益并不确定。一项探索性的扁平路径对照方法达到了0.572的召回率，但每个议题需要24.6次模型调用，而路由仅需8.9次。路由平均为8.9

    arXiv:2609.35918v1 Announce Type: cross  Abstract: Finding the right files is an early challenge for coding agents. We test whether a language model can follow directory and file names to find annotated code files missed by fixed lexical queries. Across 82 audited issues from 11 repositories at pinned pre-fix commits, name-only directory routing recovered 0.465 of gold files within eight candidates, compared with 0.352 for FTS5 and 0.245 for a fixed full-issue rg query. The paired gain over FTS5 was 0.113 (95% repository-cluster bootstrap interval, 0.053 to 0.168). Under a shared 16K-token context budget, routing delivered 0.443 of annotated lines versus 0.246 for FTS5 on 55 cases with fully aligned annotations. At the same eight-file limit, combining routing with FTS5 reached 0.491 file recall, but its gain over routing alone was uncertain. An exploratory flat path control reached 0.572 recall while using 24.6 model calls per issue, compared with 8.9 for routing. Routing averaged 8.9 
    
[^39]: UNBIND：基于推理时方向引导的代码大语言模型遗忘方法

    UNBIND: UNlearning By INference-time Directional Steering for Code LLMs

    [https://arxiv.org/abs/2609.35913](https://arxiv.org/abs/2609.35913)

    UNBIND是一种代码大语言模型遗忘框架，通过分别建模目标代码对应的隐藏状态与抑制其复制的方式构建独立引导方向，在不修改模型权重的情况下于推理时实现选择性遗忘，同时保持通用编程能力。

    

    代码大语言模型从大型代码语料库中获取编程能力，但也可能记住一些日后需要删除的代码实现。当出现版权或安全问题时，需要进行代码遗忘以控制这些实现的持续复制。然而，需要遗忘的目标代码与需要保留的代码共享计算模式，这在遗忘特定实现与保持通用编程能力之间造成了矛盾。我们提出了UNBIND，一个代码遗忘框架，它分别考虑哪些隐藏状态对应目标代码以及如何抑制其复制。通过为这些目标构建独立的方向，UNBIND在保持模型权重固定的情况下，于推理时实现选择性遗忘。我们的评估涵盖了两个代码模型和两个语料库上的十四个基线方法。UNBIND在所有设置中都取得了最高的遗忘与效用联合得分。它减少了目标代码的复制

    arXiv:2609.35913v1 Announce Type: cross  Abstract: Code large language models acquire programming capabilities from large code corpora, but can also memorize implementations that later require removal. Code unlearning is needed to control their continued reproduction when copyright or security concerns arise. However, targeted and retained code share computational patterns, creating a tension between forgetting specific implementations and preserving general programming ability. We propose \textbf{UNBIND}, a code unlearning framework that separately considers which hidden states correspond to the target code and how to suppress its reproduction. By constructing separate directions for these objectives, UNBIND achieves selective unlearning at inference time while keeping model weights fixed. Our evaluation covers fourteen baselines across two code models and two corpora. UNBIND achieves the highest joint forgetting and utility score in every setting. It reduces target code reproduction 
    
[^40]: 工件晋升控制模型：一个实施案例研究。在目标机器上构建 vs. 一次构建并晋升工件

    The Artifact Promotion Control Model: An Implementation Case Study. Build on Target Machines vs. Build Once and Promote Artifacts

    [https://arxiv.org/abs/2609.35891](https://arxiv.org/abs/2609.35891)

    本文将“工件晋升”（一次构建、将同一构建副本逐环境晋升）形式化为严格的控制模型，并论证其比在目标机器上构建更能直接满足FedRAMP、SOX 404、HIPAA等法规的完整性与变更控制要求。

    

    工件晋升是指软件只构建一次，然后让同一个构建副本依次通过测试和生产环境，而不是在每台服务器上重新构建。第一作者此前的论述以通俗易懂的工程文字将其呈现为一种云部署控制模型。本文第一部分以更严格的形式重述了该模型：发布对象、部署所跨越的控制域、工件与环境身份的区分、作为控制域问题的源码控制失陷，以及各项法规控制要求（FedRAMP/SI-7、SOX 404、FFIEC、DO-178C、FDA 21 CFR Part 11、HIPAA、DoD IL）——与在目标机器上构建相比，该模型的属性能更直接地满足这些法规的完整性与变更控制要求。该模型的各个组成部分在已有文献中均已存在：一次构建原则、构建/发布/运行分离、二进制授权、作为供应链属性的隔离性、NIST SP 800-204D。

    arXiv:2609.35891v1 Announce Type: new  Abstract: Artifact promotion means building the software once and moving that same built copy through the test and production environments, instead of building it again on each server. A previous treatment by the first author presented it as a control model for cloud deployment in accessible engineering prose. Part I of this paper restates the model in stricter form: the release object, the control domains a deployment crosses, the artifact-versus-environment identity distinction, source-control compromise as a control-domain question, and the regulatory controls (FedRAMP/SI-7, SOX 404, FFIEC, DO-178C, FDA 21 CFR Part 11, HIPAA, DoD IL) whose integrity and change-control requirements the model's properties meet more directly than build-on-target does. The parts of the model are each in the literature already: the build-once principle, the build/release/run split, binary authorization, separation as a supply-chain property, NIST SP 800-204D. The pa
    
[^41]: PoliVEM：一个Python驱动的计算固体力学虚拟单元框架

    PoliVEM: a Python-driven virtual element framework for computational solid mechanics

    [https://arxiv.org/abs/2609.35878](https://arxiv.org/abs/2609.35878)

    PoliVEM是一个Python驱动的虚拟单元法软件框架，采用模块化C++核心统一实现多种固体与结构力学问题，新公式只需提供投影、离散形式和稳定化项即可复用网格、组装和求解等现有基础设施。

    

    本工作提出了PoliVEM，一个面向计算固体与结构力学中虚拟单元法（VEM）的软件框架。通过C++17计算核心与Python接口，该框架将一维梁、二维与三维弹性力学、轴对称弹性、瞬态扩散以及有限应变超弹性问题纳入统一的实现中。框架将顶点、边、面和单元自由度存储于同一层次结构中，基于公共多项式数据构建能量投影、应变投影和L²投影，并在单元层面保持一致性—稳定性的分解结构。核心部分将网格、材料、单元、组装器和求解器的职责相互分离，并通过组合方式将它们结合起来。新的公式化只需提供其投影、离散形式和稳定化项，即可复用网格表示、自由度编号、边界条件处理、稀疏矩阵组装、代数求解器以及Python绑定等现有基础设施。

    arXiv:2609.35878v1 Announce Type: new  Abstract: This work presents PoliVEM, a software framework for the Virtual Element Method (VEM) in computational solid and structural mechanics. A C++17 computational core and a Python interface place one-dimensional beams, two- and three-dimensional elasticity, axisymmetric elasticity, transient diffusion, and finite-strain hyperelasticity in a common implementation. The framework stores vertex, edge, face, and cell degrees of freedom in one hierarchy, constructs the energy, strain, and $L^2$ projections from common polynomial data, and retains the consistency--stabilization split at the element level. The core separates the mesh, material, element, assembler, and solver responsibilities and combines them by composition. A new formulation supplies its projection, discrete form, and stabilization while reusing the mesh representation, degree-of-freedom numbering, boundary-condition treatment, sparse assembly, algebraic solvers, and Python binding 
    
[^42]: 更多程序还是更多掷骰？分离LLM测试装置中的答案覆盖与任务专业化

    More Programs or More Rolls? Separating Coverage from Specialization in LLM Harnesses

    [https://arxiv.org/abs/2609.35873](https://arxiv.org/abs/2609.35873)

    该研究通过受控评估将答案覆盖与任务专业化分离，发现LLM测试装置的性能提升主要来自重复执行带来的答案覆盖而非真正的专业化——生成的程序主要持续暴露弱点而非稳定优势，预执行选择也几乎没有增益。

    

    自动化生成LLM测试装置有望通过任务专业化来改进推理。然而，额外的答案覆盖可能来自同一程序的重复执行，这使得专业化难以被识别。我们引入了一种受控评估方法，将答案覆盖、可重复的任务优势以及预执行选择带来的收益分离开来。在386个MATH-500任务上，我们将八个生成的测试装置与一个包含九个字节完全相同的基线副本的基线进行比较，每个成员执行三次。完全相同的程序产生了2.16个百分点的重复平均oracle提升空间。生成的程序表现出明显更多可重复的分数模式，但这些模式主要揭示的是持续的弱点：在100个任务上，相对于基线的损失在全部三次重复中持续存在，而持续获胜仅出现在一个任务上，且对答案提取方式敏感。冻结选择器获得0.00个百分点的收益，且两个总体……（摘要截断）

    arXiv:2609.35873v1 Announce Type: new  Abstract: Automated generation of LLM harnesses promises to improve inference through task specialization. Yet additional answer coverage can arise from repeated execution of the same program, making specialization difficult to identify. We introduce a controlled evaluation that separates answer coverage, repeatable task advantages, and gains from pre-execution selection. On 386 MATH-500 tasks, we compare eight generated harnesses plus a baseline with nine byte-identical baseline copies, using three executions per member. Identical programs yield 2.16 percentage points of repeat-averaged oracle headroom. Generated programs exhibit substantially more repeatable score patterns, but these chiefly reveal persistent weaknesses: losses relative to the baseline persist across all three repeats on 100 tasks, while persistent wins occur on only one task and are sensitive to answer extraction. The frozen selector gains 0.00 percentage points, and both popul
    
[^43]: 超越基于规则的变异测试：利用大语言模型进行测试感知的变异体生成

    Beyond Rule-Based Mutation Testing: Test-Aware Mutant Generation Using Large Language Models

    [https://arxiv.org/abs/2609.35841](https://arxiv.org/abs/2609.35841)

    本文提出测试感知的变异体生成方法，让大语言模型在提示中同时感知问题描述、标准解答和现有基础测试，并生成能通过这些测试的非平凡变异体，从而克服传统规则式变异测试产生琐碎冗余变异体以及现有LLM方法“测试盲”的局限。

    

    变异测试通过向程序代码中注入人工故障来评估测试套件的充分性。然而，传统的基于规则的工具往往会生成大量琐碎、冗余或等效的变异体，限制了其在识别测试套件缺口方面的实际应用。尽管近期基于大语言模型（LLM）的方法能够生成更真实的故障，但大多数方法仍然是“测试盲”的：模型只能看到源代码，无法推理现有测试已经覆盖了哪些内容。我们提出了测试感知的变异体生成方法，其中LLM在单个提示中接收问题描述、标准解答和基础测试，并且必须生成一个能够通过基础单元测试的非平凡变异体。我们在HumanEval和MBPP基准上，使用五个LLM——Gemini 3.1 Pro、Gemini 3 Flash、GPT 5.1 Codex Mini、GPT 4.1 Mini和Qwen3-32B——对该方法进行了评估。扩展的EvalPlus测试套件作为自动化判定器，用于验证（原文摘要在此处截断）

    arXiv:2609.35841v1 Announce Type: cross  Abstract: Mutation testing evaluates test-suite adequacy by injecting synthetic faults into program code. However, traditional rule-based tools often generate large numbers of trivial, redundant, or equivalent mutants that limit their practical use for identifying gaps in a test suite. While recent large language model (LLM)-based approaches generate more realistic faults, most remain test-blind: The model sees only the source code and cannot reason about what existing tests already cover. We propose test-aware mutant generation, in which an LLM receives the problem statement, canonical solution and base tests in a single prompt, and must generate a nontrivial mutant that passes the base unit tests. We evaluate this approach across a set of five LLMs -- Gemini 3.1 Pro, Gemini 3 Flash, GPT 5.1 Codex Mini, GPT 4.1 Mini, Qwen3-32B -- on the HumanEval and MBPP benchmarks. The extended EvalPlus test suites serve as an automated oracle to verify wheth
    
[^44]: 面向量子转译器的一种防泄漏、成本感知的回归测试方法论

    A Leakage-Safe, Cost-Aware Regression Testing Methodology for the Quantum Transpiler

    [https://arxiv.org/abs/2609.35834](https://arxiv.org/abs/2609.35834)

    该论文提出了一种面向 Qiskit 量子转译器的防泄漏、成本感知的回归测试选择方法论，其预先注册、预算绑定的评估表明，透明的风险评分选择器在固定预算下的缺陷检测效果并不优于简单的多样性优先级基线。

    

    arXiv:2609.35834v1 公告类型：新论文。摘要：诸如 Qiskit 之类的量子软件开发工具包（SDK）在不断修订，而转换器通道（transpiler pass）的单一改动就可能引入软件回归缺陷，因此持续集成（CI）必须在固定预算下从庞大且成本差异巨大的测试套件中选择并优先排序测试。我们提出了一种面向 Qiskit 量子转译器的防泄漏、成本感知的回归测试选择方法论，并采用预先注册、预算绑定的评估来避免数据泄漏。该评估使用了成本异构的测试语料库（116 个单元，单元间成本差异高达 9,945 倍，T_full = 69.17 秒），其预算在任何测试预言机标签被读取之前就根据实测成本加以固定。一个透明的 risk_score 选择器并未超越简单的测试用例优先级基线：检测率对预算的平均 AUC 为 0.721 [95% CI 0.44, 0.94]，而仅基于多样性的基线为 0.874 [0.65, 1.00]（Cliff's δ = -0.64，差异较大），基于成本/历史/变更阶段的基线为 0.840，随机基线为 0.821。组件消融实验显示……

    arXiv:2609.35834v1 Announce Type: new  Abstract: Quantum SDKs such as Qiskit are revised continually, and a single transpiler-pass change can introduce a software regression, so continuous integration (CI) must select and prioritize tests from a large, cost-heterogeneous suite under a fixed budget. We present a leakage-safe, cost-aware regression-test-selection methodology for the Qiskit quantum transpiler, with a pre-registered, budget binding evaluation that avoids data leakage. The evaluation uses a cost-heterogeneous corpus (116 units, per-unit cost spread 9,945X, T_full = 69.17 s) whose budgets were fixed from measured cost before any test-oracle label was read. A transparent risk_score selector does not beat simple test case-prioritization baselines: mean detection-vs-budget AUC is 0.721 [95% CI 0.44, 0.94] versus 0.874 [0.65, 1.00] for diversity-only (Cliff's {\delta} = -0.64, large), with cost/history/change-stage baselines at 0.840 and random at 0.821. A component ablation sho
    
[^45]: 车载对话助手多轮对话的自动化评估

    Automated Evaluation of Multi-Turn Dialogues in In-Car Conversational Assistants

    [https://arxiv.org/abs/2609.35812](https://arxiv.org/abs/2609.35812)

    提出了一个自动化测试框架，通过闭环仿真结合策略引导的用户模拟器、对抗性策略管理器和双层LLM裁判，来评估车载对话助手在多轮对话中的约束处理、上下文保持和安全关键行为。

    

    车载对话助手（ICAs）日益被集成到车辆中，以支持路线规划、车辆控制和信息获取。由于多轮交互、缺乏明确的真实标准（ground truth）以及严格的安全约束，确保其可靠性极具挑战性。现有评估技术存在不足，因为它们针对单轮对话设置，无法捕捉跨轮次的约束处理、上下文保持以及安全关键行为。我们提出了一个用于测试车载对话助手多轮对话能力的自动化框架。该系统被视为黑盒，通过闭环仿真进行评估，框架包含一个策略引导的用户模拟器、一个对抗性策略管理器，以及一个双层LLM裁判，用于评估轮次级的故障和对话级的质量。我们在一个工业级车载对话助手上评估了该方法，使用了六个LLM后端和十二名人类标注者。自动化裁判显示出高度的一致性……

    arXiv:2609.35812v1 Announce Type: new  Abstract: In-car conversational assistants (ICAs) are increasingly integrated into vehicles to support route planning, vehicle control, and information access. Ensuring their reliability is challenging due to multi-turn interactions, the absence of explicit ground truth, and strict safety constraints. Existing evaluation techniques fall short, as they target single-turn settings and fail to capture constraint handling, context retention, and safety-critical behavior across turns. We propose an automated framework for testing the multi-turn conversational capabilities of ICAs. The system is treated as a black box and evaluated via closed-loop simulation with a strategy-guided user simulator, an adversarial strategy manager, and a two-tier LLM judge assessing turn-level failures and conversation-level quality. We evaluate the approach on an industrial ICA with six LLM backends and twelve human annotators. The automated judge shows substantial agreem
    
[^46]: Lookahead-R：基于以执行为中心规划的预算感知工具检索

    Lookahead-R: Budget-Aware Tool Retrieval via Execution-Centric Planning

    [https://arxiv.org/abs/2609.35811](https://arxiv.org/abs/2609.35811)

    Lookahead-R通过轻量级执行感知代理世界模型（无需调用真实API即可预测工具执行结果、延迟与语义效用），结合预算感知的蒙特卡洛树搜索，将工具检索转化为资源受限的序贯决策问题，实现了精度与效率的最优平衡。

    

    工具检索是基于大语言模型（LLM）的智能体在大型异构API生态系统中运行的关键瓶颈。现有方法面临固有的权衡困境：语义检索器速度快但受制于语义-功能鸿沟，而基于执行的验证虽能提升精度，却带来难以接受的延迟。我们提出Lookahead-R，这是一个基于规划的框架，将工具检索重新表述为资源受限的序贯决策问题。其核心在于，Lookahead-R引入了一个轻量级的执行感知代理世界模型，无需调用真实API即可联合预测工具执行成功率、延迟成本和语义效用。该世界模型驱动一个成本敏感、不确定性引导的蒙特卡洛树搜索，在严格预算约束下探索工具空间。在大规模ToolBench基准上的评估表明，Lookahead-R在所有测试场景中均实现了卓越的准确性-效率权衡。

    arXiv:2609.35811v1 Announce Type: cross  Abstract: Tool retrieval is a critical bottleneck for LLM-based agents operating over large, heterogeneous API ecosystems. Existing approaches face an inherent trade-off: semantic retrievers are fast but suffer from the semantic-functional gap, while execution-based validation improves precision at the cost of prohibitive latency. We propose Lookahead-R, a planning-based framework that reformulates tool retrieval as a resource-constrained sequential decision-making problem. At its core, Lookahead-R introduces a lightweight execution-aware surrogate world model that jointly predicts tool execution success, latency cost, and semantic utility---without invoking real APIs. This world model drives a cost-sensitive, uncertainty-guided Monte Carlo Tree Search that navigates the tool space under strict budget constraints. Evaluated on the large-scale ToolBench benchmark, Lookahead-R achieves a superior accuracy-efficiency trade-off across all test scena
    
[^47]: 智能体可调用功能覆盖率：衡量软件对AI智能体的就绪程度

    Agent-Callable Feature Coverage: Measuring Software Readiness for AI Agents

    [https://arxiv.org/abs/2609.35789](https://arxiv.org/abs/2609.35789)

    该论文提出GUI-API对等原则，并引入智能体可调用功能覆盖率（ACFC）指标与智能体就绪度符合性（ARC）评估框架，从可访问性、可发现性和可控性三个维度衡量软件对AI智能体的就绪支持程度。

    

    AI智能体已经能够通过基于截图的计算机操作方式来操作图形化软件，因此当前紧迫的问题不是智能体能否操作软件，而是软件能否通过结构化、可控的通道为智能体提供良好支持。我们将这一需求形式化为GUI-API对等原则：人类用户通过图形界面可用的每一项功能，也应该能够通过带有适当安全元数据的结构化可调用接口供智能体访问。为了将该原则落地实施，我们提出了两项贡献：第一，智能体可调用功能覆盖率（ACFC），这是一个产品级的就绪度指标，用于量化智能体能够访问和使用系统面向人类功能的程度；第二，智能体就绪度符合性（ARC），这是一个从三个维度对每项功能进行评分的框架，包括可访问性（智能体能否调用它）、可发现性（智能体能否找到并理解它）以及可控性（智能体能否安全地使用它）。

    arXiv:2609.35789v1 Announce Type: new  Abstract: AI agents already operate graphical software through screenshot-based computer use, so the pressing question is not whether agents can operate software, but how well software supports them through structured, controllable channels. We formalize this need as the GUI-API parity principle: every capability available to human users through a graphical interface should also be accessible to agents through a structured, callable interface with appropriate safety metadata. To operationalize this principle, we introduce two contributions. First, Agent-Callable Feature Coverage (ACFC), a product-level readiness metric quantifying how much of a system's human-facing functionality agents can access and use. Second, Agent Readiness Conformance (ARC), a framework that scores each capability on three axes: Accessibility (can an agent call it), Discoverability (can an agent find and understand it), and Controllability (can an agent use it safely), each
    
[^48]: TokenCast：预测LLM智能体执行过程中的Token消耗

    TokenCast: Forecasting Token Consumption During LLM Agent Execution

    [https://arxiv.org/abs/2609.35760](https://arxiv.org/abs/2609.35760)

    提出TokenCast方法，通过学习可组合的执行片段成本表示来预测LLM智能体执行任务时的Token消耗，并随执行进展动态刷新预测，无需额外的LLM调用。

    

    当大语言模型（LLM）智能体执行同一任务时，不同运行之间的Token消耗可能相差超过一个数量级。智能体根据工具反馈和中间结果选择下一步操作，而不断增长的上下文会持续膨胀每次后续调用的输入规模。因此，任务的总消耗在执行前难以预测，且预测必须随着运行的展开而不断修正。在本文中，我们提出TokenCast，它为每个执行片段学习一种可组合的成本表示，记录其自身的消耗以及其引入的上下文增长。将相邻片段组合起来可以得到一个累积估计，该估计能够捕捉后续每次调用重新读取早期片段上下文时所产生的额外输入成本。随着执行的展开，新观察到的证据会刷新预测，无需额外的LLM调用，平均累积预测时间为32.8（摘要在此处截断）。

    arXiv:2609.35760v2 Announce Type: replace-cross  Abstract: When a large language model (LLM) agent executes the same task, token consumption can vary by over an order of magnitude across runs. The agent chooses its next steps based on tool feedback and intermediate results, while the growing context steadily inflates the input size of every subsequent call. The total consumption of a task is therefore hard to predict before execution and the prediction must be revised as the run unfolds. In this paper, we propose TokenCast, which learns a composable cost representation for each execution segment, recording its own consumption and the context growth it introduces. Composing adjacent segments yields a cumulative estimate that captures the extra input cost incurred when context from earlier segments is re-read by every later call. As execution unfolds, newly observed evidence refreshes the forecast, requiring no additional LLM calls and incurring a mean cumulative prediction time of 32.8 
    
[^49]: 为开发者编写的MCP错误信息对能力最强的智能体伤害最大

    MCP Error Messages Written for Developers Hurt the Most Capable Agents Most

    [https://arxiv.org/abs/2609.35381](https://arxiv.org/abs/2609.35381)

    研究发现MCP服务器中面向人类开发者的错误信息会误导只能调用工具的AI智能体去执行无法完成的操作，且模型能力越强、越忠实遵循这些指令，受到的性能损失反而越大。

    

    许多模型上下文协议服务器封装了为人类开发者构建的网络API，其错误信息会告诉读者去运行命令、编辑配置、打开网页或等待，而许多读取这些信息的智能体只能调用服务器的工具。在150个广泛使用的MCP服务器中，3,001条错误信息里有949条会告诉调用者下一步该做什么，其中一半的步骤依赖于服务器无法感知的调用者相关信息。在凭证错误方面，67个步骤中有62个要求执行终端命令、修改配置或打开网页；在速率限制方面，30条中有20条要求等待并重试，却未指明应重复哪个调用。我们测试了五个仅通过伯克利函数调用排行榜任务工具来执行操作的OpenAI模型，智能体确实按照步骤的指示行动。在凭证过期的情况下，步骤中要求的终端命令导致只有45%的任务得以恢复，且由此造成的损失从GPT-5.5的18分增长到GPT-6 Astra的69分。在速率限制方面，G……（原文摘要在此处截断）

    arXiv:2609.35381v2 Announce Type: replace-cross  Abstract: Many Model Context Protocol (MCP) servers wrap web APIs built for human developers, and their error messages tell the reader to run a command, edit a configuration, open a web page or wait. Many agents that read them can only call the server's tools. In 150 widely used MCP servers, 949 of 3,001 error messages tell the caller what to do next, and half of these steps depend on something the server cannot see about the caller. On credential errors, 62 of 67 steps ask for a terminal command, a configuration change or a web page; on rate limits, 20 of 30 say to wait and retry without naming the call to repeat. We tested five OpenAI models that act only through the tools of Berkeley Function Calling Leaderboard tasks, and the agents did what the step said. On expired credentials, a terminal command in the step left 45% of tasks recovered, and the loss it caused grew from 18 points for GPT-5.5 to 69 for GPT-6 Astra. On a rate limit, G
    
[^50]: 修复之后：修正后智能体经验的迁移

    After the Fix: Transfer of Corrected Agent Experience

    [https://arxiv.org/abs/2609.34603](https://arxiv.org/abs/2609.34603)

    本研究通过3,300次运行系统评估了修复后的智能体经验向后续任务迁移的效果，发现修正经验带来的收益在很大程度上源于未修正基线较弱而非记忆质量的真正提升，且并非所有修正机制（如APEX）都能产生可比的修正收益。

    

    修复一个失败的回合能否使其经验成为下一个任务更好的记忆？我们将同一失败的源任务在修复被接受之前和之后迁移到一个固定的目标任务，并与独立执行进行对比。我们的3,300次运行涵盖了100对ThinkingBox任务以及相同的100对APEX任务（分别在有和没有源状态继承的条件下），共在十一种条件下进行实验。ThinkingBox的Full/Skill/Hybrid修正收益分别为44/29/32个百分点，修正后的性能比独立执行高出25/22/18个百分点；但在任务族层面，推断能力会减弱。然而，Full相比Skill多出的15个百分点修正差距中，有12个百分点来自未修正时表现更差，而非修正后的记忆质量更好。此外，Full的46次向上转变中有22次只是恢复了观察到的基线成功。两种APEX机制均未能展现出可比的整体修正收益。行动证据将工作流收益与可复用的义务联系起来，而约定冲突则与源任务本地的选择相关。（原文摘要在此处不完整）

    arXiv:2609.34603v2 Announce Type: replace  Abstract: Does repairing an episode make its experience a better memory for the next task? We transfer the same failed source before and after accepted repair to a fixed target, alongside independent execution. Our 3,300 runs cover 100 ThinkingBox pairs and the same 100 APEX pairs with and without source-state inheritance, under eleven conditions. ThinkingBox's Full/Skill/Hybrid correction gains are 44/29/32 percentage points, with corrected performance 25/22/18 points above independence; inference weakens at the task-family level. Yet 12 of Full's 15-point larger correction gap over Skill come from worse uncorrected performance, not better corrected memory. Moreover, 22 of Full's 46 upward transitions restore observed baseline success. Neither APEX regime establishes comparable aggregate correction benefits. Action evidence connects workflow gains with reusable obligations and convention conflicts with source-local choices. Text APEX's accept
    
[^51]: 反事实回放重演：可分叉环境作为软件工程智能体的免费过程奖励

    Counterfactual Rollout Replay: Forkable Environments as Free Process Rewards for Software Engineering Agents

    [https://arxiv.org/abs/2609.33875](https://arxiv.org/abs/2609.33875)

    该论文提出反事实回放重演（CRR），利用可分叉的可执行环境在选定决策点采样替代动作并对比终端回报，从而在无需人工过程标注或过程奖励模型的情况下，为软件工程智能体提供免费的步级过程监督信号，并在多个 SWE 基准上提升了 pass@1。

    

    仅有结果奖励的强化学习为软件工程（SWE）智能体提供了终端成功信号，但对其中间决策几乎没有直接指导。我们提出了反事实回放重演（Counterfactual Rollout Replay, CRR），这是一种训练时流程，利用可分叉的可执行环境来获得步级回报对比。CRR 选取一小部分决策点，恢复每个状态，采样一个替代动作，并在当前策略下将该分支向前推演。它保留实际发生的训练轨迹，并在选定步骤上把优势替换为该轨迹终端回报与采样得到的反事实回报之差。该方法不需要人工过程标注，也不需要学习得到的过程奖励模型；“免费”指的是这些监督成本，而非重演所需的计算开销。使用 14B 策略，CRR 在 SWE-bench Verified、SWE-bench Live 和 SWE-rebench 上提升了 pass@1，并且可以与过程奖励和轨迹搜索方法相结合。在 SWE-bench

    arXiv:2609.33875v2 Announce Type: replace-cross  Abstract: Outcome-only reinforcement learning gives software engineering (SWE) agents a terminal success signal but little direct guidance about intermediate decisions. We introduce Counterfactual Rollout Replay (CRR), a training-time procedure that uses forkable executable environments to obtain step-level return contrasts. CRR selects a small set of decision points, restores each state, samples an alternative action, and rolls the branch forward under the policy. It retains the realised training trajectory and replaces the advantage at selected steps with the difference between its terminal return and the sampled counterfactual return. The method needs no human process labels or learned process reward model; free refers to those supervision costs, not replay compute. With a 14B policy, CRR improves pass@1 on SWE-bench Verified, SWE-bench Live, and SWE-rebench, and combines with process-reward and trajectory-search methods. On SWE-bench
    
[^52]: 评估用于智能体安全决策的系统一模型：可靠性、校准与选择性自动化

    Evaluating System One Models for Agent Security Decisions: Reliability, Calibration, and Selective Automation

    [https://arxiv.org/abs/2609.33401](https://arxiv.org/abs/2609.33401)

    该研究系统评估了四款用于智能体安全决策的系统一模型，发现良好的整体表现和校准可能掩盖针对特定攻击类别的集中性失误，且适配配置并不总是优于其基础模型。

    

    基于模型的评判器通过检测提示注入、评估交互风险以及筛查有害请求来支持智能体安全。系统一模型从预定义答案中进行选择并报告概率，软件可利用这些概率来允许、阻止或审查输入，但这些自动化决策的可靠性仍不明确。我们评估了 Jev、Laya、Decider 和 Bespoke Nimble，并将其与专用分类器和语言模型评判器进行对比，考察决策准确性、概率校准以及选择性自动化。我们得出以下结论：(1) 良好的整体表现和有利的总体校准可能掩盖集中于特定攻击类别的失败，包括以高置信度被判为安全的攻击；(2) 所评估的适配配置在各项任务上并未持续优于其基础模型的分类表现；(3) 在所评估的最严格错误限制下，这些策略仅允许少量输入通过。

    arXiv:2609.33401v2 Announce Type: replace-cross  Abstract: Model-based judges support agent security by detecting prompt injections, assessing interaction risks, and screening harmful requests. System One models select from predefined answers and report probabilities that software can use to allow, block, or review inputs, but the reliability of these automated decisions remains unclear. We evaluate Jev, Laya, Decider, and Bespoke Nimble against specialized classifiers and language-model judges, examining decision accuracy, probability calibration, and selective automation. We draw the following conclusions. (1) Strong overall performance and favorable aggregate calibration can hide failures concentrated in particular attack groups, including attacks classified as safe with high confidence. (2) The evaluated adapted configurations do not consistently improve classification over their base models across tasks. (3) Under the strictest evaluated error limits, the policies allow few inputs
    
[^53]: Harness演化中的组合式安全失效：识别与运行时监控

    Compositional Safety Failures in Harness Evolution: Identification and Runtime Monitoring

    [https://arxiv.org/abs/2609.33123](https://arxiv.org/abs/2609.33123)

    该研究首次系统揭示了Harness演化中的组合式安全失效——即各自安全且不损害效用的组件更新在交互后可能引发不安全的智能体行为——在三个安全基准上识别出43个两两组合和18个不可约的三路组合安全失效，并提出运行时监控方法以应对跨组件安全验证的组合复杂度难题。

    

    自进化的智能体Harness会持续更新记忆、提示词、技能和工具等持久化组件，我们将这一过程称为Harness演化。然而，这种演化可能引入意想不到的安全风险。现有工作研究Harness的错误演化，并对候选Harness进行验证或将问题归因于单个组件的更新，而对跨组件更新交互的安全分析在很大程度上尚未被研究。为填补这一空白，我们研究了Harness演化中的组合式安全失效：即各个组件更新单独来看是安全且保持效用的，但它们之间的交互可能产生不良或不安全的智能体行为，这揭示了Harness演化中固有的安全风险。在三个与安全相关的基准测试中，我们识别出43个两两组合和18个不可约的三路组合安全失效。传统解决方案在验证跨组件交互时会带来组合爆炸式的复杂度，使得安全检查难以实际执行。（注：原文摘要在此处被截断）

    arXiv:2609.33123v2 Announce Type: replace  Abstract: Self-evolving agent harnesses continually update persistent components such as memory, prompts, skills, and tools. We call this process harness evolution. However, such evolution could introduce unexpected safety risks. Existing work studies harness misevolution and validates candidate harnesses or attributed individual component updates, leaving safety analysis of cross-component update interactions largely unexamined. To address this gap, we study compositional safety failures in harness evolution, where interactions among individually safe and utility-preserving component updates can produce undesirable or unsafe agent behavior, revealing a safety risk intrinsic to harness evolution. Across three safety-related benchmarks, we identify 43 pairwise and 18 irreducible 3-way compositional safety failures. Conventional solution incurs combinatorial complexity in validating cross-component interactions, leaving the safety checking impra
    
[^54]: Relic：从多智能体协作到持久化的组织能力

    Relic: From Multi-Agent Collaboration to Persistent Organizational Capability

    [https://arxiv.org/abs/2609.32965](https://arxiv.org/abs/2609.32965)

    Relic系统将多智能体协作中反复出现的失败转化为组织拥有的可执行协议，使经验教训能够在成员更替后持续约束团队行为，并通过360次受控实验证明了其提升完整契约交付率的有效性。

    

    多个智能体在组织中可能经常发生冲突：例如，一个编码智能体修改了仓库中的某个接口，而另一个智能体仍在旧版本上继续开发，导致现有测试失效。一次对话可以解决这一事件，但当参与者更换后，是什么让这一经验教训继续约束整个团队？我们提出了Relic，它将反复出现的协作失败转化为组织拥有的、可执行的协议。成员们对可见的工作进行反思、提出规则，并管理规则的采纳。被采纳的协议将触发条件、职责、所需证据和执行后果绑定到运行时环境中，同时保持可修订和可退役的开放性。在一个追踪案例中，反复出现的集成摩擦催生了一项接口审查规则，该规则约束着后续的拉取请求，并随着工作的推进不断修订。在涵盖十个软件工作负载和三个模型的360次受控运行中，Relic提升了完整契约交付（摘要在此处截断）

    arXiv:2609.32965v2 Announce Type: replace  Abstract: Multiple agents may often conflict in an organization: for example, one coding agent changes an interface in a repository, but another continues to develop on the old version where existing tests become stale. A conversation can resolve the episode, but when the participants change, what makes the lesson continue to govern the team? We introduce Relic, which turns recurring collaboration failures into organization-owned, executable protocols. Members reflect on visible work, propose rules, and govern their adoption. Adopted protocols bind triggers, responsibilities, required evidence, and execution consequences to the runtime, while remaining open to revision and retirement. In one traced case, repeated integration friction produces an interface-review rule that governs later pull requests and is revised as work continues. Across 360 controlled runs over ten software workloads and three models, Relic raises complete-contract delivery
    
[^55]: ASCEND：跨高性能计算集群与GPU工作站实现自主科学计算的个人AI智能体

    ASCEND: Personal AI Agents for Autonomous Scientific Computing Across HPC Clusters and GPU Workstations

    [https://arxiv.org/abs/2609.32868](https://arxiv.org/abs/2609.32868)

    ASCEND是一个运行在研究人员本地笔记本电脑上的AI智能体，通过安全认证连接远程调度Slurm集群和GPU工作站，无需设施级服务即可自主完成科学计算中的作业提交、故障诊断与恢复闭环。

    

    传统科学计算要求研究人员将计算意图转化为环境配置、资源请求和可执行作业，随后再根据调度器状态和应用程序日志诊断故障。我们提出了ASCEND（Autonomous Scientific Computing Engine and Novel Discovery，自主科学计算引擎与新发现），这是一个AI驱动的智能体接口，运行在研究人员自己的笔记本电脑上，通过多路复用的认证连接访问Slurm管理的集群和GPU工作站，其站点特定的执行策略由本地执行的工具进行检查；语言模型远程托管，不持有任何凭证。无需设施级服务：只需在每个资源上拥有账户即可，公共安装程序允许用户接入自己的其他Slurm集群或工作站。我们报告了四个记录在案的案例：(1) 智能体在植入的张量设备故障上完成了故障恢复闭环，包括作业提交、故障诊断与修复（摘要在此处截断）。

    arXiv:2609.32868v2 Announce Type: replace-cross  Abstract: Traditional scientific computing requires researchers to translate computational intent into environment configuration, resource requests, and executable jobs, then diagnose failures from scheduler state and application logs. We present ASCEND (Autonomous Scientific Computing Engine and Novel Discovery), an AI-powered agent interface that runs the agent on the researcher's own laptop, reaching Slurm-managed clusters and a GPU workstation over a multiplexed authenticated connection, with site-specific execution policies checked by locally executed tools; the language model is hosted remotely and holds no credentials. No facility-scale service is required: an account on each resource is sufficient, and the public installer lets users link additional Slurm clusters or workstations of their own. We report four recorded cases: (1) the agent closed a failure-recovery loop on a planted tensor-device fault, submitting, diagnosing, repa
    
[^56]: 复杂度拐点：面向代码生成可靠性的提示侧结构复杂度指数

    The Complexity Kink: A Prompt-Side Structural Complexity Index for Code-Generation Reliability

    [https://arxiv.org/abs/2609.19616](https://arxiv.org/abs/2609.19616)

    该论文提出一个生成前评分的六维提示侧结构复杂度指数，发现代码生成通过率在复杂度分数上存在非单调断点，但该断点并非通用的失败临界值，会随任务类型固定效应等因素发生移动。

    

    从生成代码中测量的复杂度依赖于失败情况：一个困难的提示可能产生一个简短的失败程序，从而被赋予较低的输出复杂度。我们提出了一个六维度的提示侧结构复杂度指数，在生成之前进行评分，并与正确性保持分离。我们从初步单评分者量表的六个区间中选择了5,000个Python提示。四个面板外的LLM评分者对锁定的提示重新评分，得到19,997条评分记录；在四个评分齐全的4,998个提示上，综合评分者间信度为ICC = 0.872。我们对每个提示评估21个模型，共产生105,000个生成结果。在未经调整的均值池化分析中，通过率在综合分数13.75处出现非单调断点，该分数及以下的通过率为79.9%，以上为87.6%。这并非一个通用的失败临界点。任务类型固定效应将断点移至10.75，并将区间差距从7.6个百分点缩小至2.1个百分点。构建框架控制将其移至8.50，原始差距为……

    arXiv:2609.19616v1 Announce Type: new  Abstract: Complexity measured from generated code is failure-dependent: a difficult prompt can yield a short failing program and be assigned low output complexity. We introduce a six-dimension prompt-side structural-complexity index scored before generation and kept separate from correctness. We select 5,000 Python prompts across six bands of a preliminary single-rater rubric. Four out-of-panel LLM raters rescore the locked prompts, giving 19,997 score rows; composite inter-rater reliability is ICC = 0.872 on the 4,998 prompts with all four ratings. We evaluate 21 models per prompt, yielding 105,000 generations. In the unadjusted mean-pooled analysis, pass rate has a nonmonotone breakpoint at composite 13.75, with 79.9% at or below and 87.6% above. This is not a universal failure cutoff. Task-type fixed effects shift the breakpoint to 10.75 and cut the regime gap from 7.6 to 2.1 points. A construction-frame control shifts it to 8.50 with a raw gap
    
[^57]: 根因归因是一个搜索问题：针对长时程智能体失败的持续搜索

    Root-Cause Attribution Is a Search Problem: Continual Search for Long-Horizon Agent Failures

    [https://arxiv.org/abs/2609.13463](https://arxiv.org/abs/2609.13463)

    本文将根因归因重新定义为一个大规模搜索问题，指出现有的一次性LLM判断方法在长轨迹中会过早下结论而遗漏关键证据，并提出了针对长时程智能体任务失败的持续搜索诊断方法。

    

    AI智能体在长时程任务中的日益广泛部署产生了海量的执行日志。诊断这些记录中的失败对于可靠性至关重要，因为它能将结果层面的信号转化为可操作的干预措施。数据的庞大规模使得人工审查变得不切实际，从而催生了对自动化根因归因的需求。然而，基于大语言模型（LLM）的自动化RCA方法存在诊断准确率低的问题，尤其是当执行轨迹变得更大时。这些方法之所以表现不佳，是因为相关信息往往稀疏、分散在相距遥远的动作中，且与可见的失败脱节，这使得根因归因演变成一个大规模搜索问题。现有的RCA方法通常依赖一次性的LLM判断来从执行轨迹中诊断失败。虽然这些判断器对于较短的轨迹是有效的，但它们往往过早地得出一个看似合理的诊断，导致较长轨迹中的关键证据未被审查。我们提出……

    arXiv:2609.13463v1 Announce Type: new  Abstract: The increasing deployment of AI agents in long-horizon tasks yields massive execution logs. Diagnosing failures within these records is crucial for reliability, as it transforms outcome-level signals into actionable interventions. The sheer scale of the data renders human review impractical, driving the need for automated root-cause attribution (RCA). However, automated RCA methods using LLMs suffer from low diagnostic accuracy, especially as execution traces grow larger. They struggle because relevant information is often sparse, distributed across distant actions, and disconnected from the visible failure, reducing root-cause attribution to a massive search problem. Existing RCA methods typically rely on one-shot LLM judgments to diagnose failures from execution traces. While effective for shorter trajectories, these judges tend to settle on a plausible diagnosis early, leaving critical evidence in longer traces unexamined. We introduc
    
[^58]: 验证工具的覆盖范围决定其价值：关于AI编程智能体中验证面、产物质量与成本的受控研究

    The reach of a verification tool decides its value: A controlled study of verification surface, artifact quality, and cost in AI coding agents

    [https://arxiv.org/abs/2608.28795](https://arxiv.org/abs/2608.28795)

    该研究通过控制单一变量的受控实验（六个模型、八种工具配置、1,116个Web应用）发现，为AI编程智能体配备验证工具能显著提升交付软件的质量，其中最廉价的启动探针即可消除绝大多数应用无法启动的失败。

    

    现代人工智能编程智能体可以配备用于检查自身工作的工具，例如代码检查器、启动探针、shell、截图工具。我们将这一工具集合称为智能体的“验证面”。本研究探讨的问题是：在其他一切条件保持固定的情况下，仅扩大这一验证面，能否带来智能体所交付软件质量的相应提升。我们构建了一个最小化的编程智能体，其工具列表是唯一的受控变量，并用它在六个模型和八种工具配置下实现了1,116个Web应用。由对实验条件不知情的人类评分员依据固定评分标准对每个应用进行评分，同时自动探针对API可观察行为进行压力测试。验证带来的最廉价收益最先显现，即确保应用能够成功启动：在没有任何工具的情况下，约七分之一的构建版本完全无法启动，而单一的启动探针就能消除几乎所有这些失败，其成本约为完整……（原文在此处截断）的35%。

    arXiv:2608.28795v1 Announce Type: cross  Abstract: Modern artificial-intelligence coding agents can be equipped with tools for checking their own work e.g. a linter, a boot probe, a shell, a screenshot tool. We call this set the agent's verification surface. This study asks whether increasing only that surface, with everything else held fixed, produces a matching growth in the quality of the software the agent ships. We built a minimal coding agent whose tool list is the single controlled variable and used it to implement 1,116 web applications across six models and eight tool configurations. A condition-blind human graded every application against a frozen rubric, and automatic probes stress-tested the API-observable behaviors. Verification's cheapest benefit arrives first, which is to make sure that the application comes up. Without any tools, about one build in seven fails to launch at all and a single boot probe removes nearly all of these failures at roughly 35 percent of a full s
    
[^59]: Active-SWE：面向无问题报告的主动式Bug修复的编码智能体基准测试

    Active-SWE: Benchmarking Coding Agents for Proactive Bug Fixing without Issue Reports

    [https://arxiv.org/abs/2608.04682](https://arxiv.org/abs/2608.04682)

    该论文提出了Active-SWE基准测试，首次将评估重点从依赖问题报告的被动式Bug修复转向无报告指导下的主动式Bug发现与修复，涵盖1,663个任务、6种Bug类别和8种编程语言，并支持多Bug修复与潜在Bug发现等更深层次的评估场景。

    

    基于大语言模型（LLM）的编码智能体在软件工程（SWE）场景中的应用日益广泛，能够修复大型代码库中的特定Bug。然而，现有的SWE基准测试通常假设带有详细信息的高质量问题报告总是可用，而由于报告获取和整理的复杂性，这一假设在实践中很容易被打破。为了解决这一问题，我们提出了Active-SWE，一个用于评估编码智能体在缺乏报告指导下主动发现并修复多个Bug能力的基准测试，涵盖6个Bug类别和8种编程语言共1,663个任务。除了将研究焦点从现有的被动式Bug修复转向主动式Bug修复之外，Active-SWE还通过将评估范围从修复单个已记录的Bug扩展到多Bug修复和潜在Bug发现场景，实现了更加深入的评估。为了构建Active-SWE，我们提出了一种新颖的难度感知（原文摘要此处截断）……

    arXiv:2608.04682v2 Announce Type: replace-cross  Abstract: Coding agents powered by large language models (LLMs) are increasingly adopted in software engineering (SWE) scenarios, capable of fixing a specific bug in large-scale codebase. However, existing SWE benchmarks typically assume that high-quality issue reports with detailed information are always available, which is easily violated in practice due to the complexity of report acquisition and curation. To address this, we introduce Active-SWE, a benchmark for evaluating coding agents on proactively discovering and fixing multiple bugs without report guidance, covering 1,663 tasks across six bug categories and eight languages. Beyond shifting the focus from existing reactive bug fixing to proactive bug fixing, Active-SWE enables a more in-depth evaluation by expanding the scope from fixing a specific recorded bug to multiple-bug fixing and potential bug discovery scenarios. To construct Active-SWE, we propose a novel difficulty-awa
    
[^60]: CURATE：利用LLM智能体组合、编目与部署可复现的工作流

    CURATE: Leveraging LLM Agents to Compose, Catalog, and Deploy Reproducible Workflows

    [https://arxiv.org/abs/2608.04270](https://arxiv.org/abs/2608.04270)

    本文提出CURATE，一种人在回路的多智能体系统，利用LLM智能体覆盖计算工作流从组合、复用、编目到部署的完整生命周期，并通过模块目录实现模块的存储、共享与复用以支持FAIR原则。

    

    智能体代码生成有潜力加速计算工作流的开发，同时降低使用门槛。然而，目前仍存在一个关键缺口：现有的编程智能体专注于代码生成，并未覆盖包括部署与共享在内的完整工作流生命周期。因此，用户需要独立开发和拼接模块，并自行管理部署。为填补这一缺口，我们提出了CURATE（组合、用户参与、复用与自动化任务执行），这是一种新颖的人在回路多智能体系统，利用LLM智能体在完整生命周期内管理和开发可组合的工作流。该系统的一个关键特性是模块目录，支持跨工作流地存储和复用模块。模块目录为支持FAIR原则提供了可扩展的基础，能够促进精选模块和子图的共享与复用。我们展示了……的可行性

    arXiv:2608.04270v2 Announce Type: replace  Abstract: Agentic code generation has the potential to accelerate the development of computational workflows while also reducing barriers to entry. However, a key gap remains: existing coding agents focus on code generation and do not address the entire workflow lifecycle, including deployment and sharing. As a result, users develop and stitch modules independently while managing deployment on their own. To address this gap, we propose CURATE (Composition, User-in-the-loop, Reuse, and Automated Task Execution), a novel human-in-the-loop multi-agent system that uses LLM agents to manage and develop composable workflows across their entire lifecycle. A key feature of the system is a catalog that allows for the storage and reuse of modules across workflows. Module catalogs provide a foundation that can be expanded to support FAIR principles by facilitating the sharing and reuse of curated modules and subgraphs. We demonstrate the feasibility of o
    
[^61]: 面向基于FRP的嵌入式系统的多模式调试

    Multi-Mode Debugging for FRP-Based Embedded Systems

    [https://arxiv.org/abs/2608.04264](https://arxiv.org/abs/2608.04264)

    本文提出了一种面向基于Emfrp的嵌入式系统的多模式调试框架，通过源代码映射技术，既支持在Emfrp抽象层面进行调试，又允许检查平台相关的C/C++ I/O代码，从而弥合了源代码级FRP程序与可执行系统之间的抽象鸿沟。

    

    Emfrp 是一种专为小型嵌入式系统设计的函数式响应式编程（FRP）语言。时变值是 FRP 中的核心抽象机制，能够简洁地描述响应式行为。然而在实践中，Emfrp 程序会被编译为 C 代码，并与用 C 或 C++ 编写的、依赖平台的输入/输出组件相结合。因此，尽管应用逻辑是用 Emfrp 编写的，开发人员仍必须使用 GDB 等传统调试器来调试由此产生的 C/C++ 混合程序。这种情况在源代码级的 FRP 程序与可执行系统之间造成了抽象鸿沟。本文提出了一种面向基于 Emfrp 的嵌入式应用的多模式调试框架。该框架支持在 Emfrp 抽象层面进行调试，同时允许检查特定平台的 C/C++ I/O 代码。我们的方法采用源代码映射技术，将 Emfrp 构造与底层程序相关联

    arXiv:2608.04264v2 Announce Type: replace-cross  Abstract: Emfrp is a functional reactive programming (FRP) language designed for small-scale embedded systems. Time-varying values are the primary abstraction mechanism in FRP and enable concise descriptions of reactive behavior. In practice, however, Emfrp programs are compiled into C and combined with platform-dependent input/output components written in C or C++. Consequently, developers must debug the resulting mixed C/C++ program using conventional debuggers such as GDB, even though the application logic is written in Emfrp. This situation creates an abstraction gap between the source-level FRP program and the executable system. This paper presents a multi-mode debugging framework for Emfrp-based embedded applications. The framework supports debugging at the level of Emfrp abstractions while also allowing inspection of platform-specific C/C++ I/O code. Our approach uses a source code mapping technique that relates Emfrp constructs t
    
[^62]: 语言模型智能体执行常规办公任务时，低推理强度已经足够

    Low Reasoning Effort Is Enough for Routine Office Work by Language-Model Agents

    [https://arxiv.org/abs/2608.03169](https://arxiv.org/abs/2608.03169)

    研究表明，语言模型智能体执行常规办公任务时，低推理强度与最高推理强度表现同样可靠（始终遵守规则、不使用禁止工具），却能节省约43%的输出token和20%的时间。

    

    目的：开发者需要选择语言模型智能体在行动前进行多少推理。有些人出于对低推理强度不够的担心而选择高强度，为此付出更多token、时间和过度思考的代价。我们测试了低推理强度对于常规办公工作是否足够。方法：两个层级的GPT-5.6分别在低推理强度和最高推理强度下完成14项常规办公任务，共840次运行。为了增加任务难度，每个任务都设置了一个按规则无法达成的目标，并提供一个本可以达成该目标但被禁止的工具。某些版本还告知智能体该禁止工具是有效的。一个程序根据每次运行的最终状态进行核查。结果：在两个推理强度下，智能体始终遵守规则，从未使用禁止工具，包括在被告知该工具有效且仍可使用它的30次运行中。低推理强度使用的输出token少约43%，耗时少约20%。结论：对于常规办公工作，低推理强度已经足够。

    arXiv:2608.03169v2 Announce Type: replace-cross  Abstract: Purpose: Developers choose how much a language-model agent reasons before it acts. Some pick a high level for fear that a low one is not enough, and pay for it in tokens, time and overthinking. We tested whether a low level is enough for routine office work. Methods: Two tiers of GPT-5.6 did 14 routine office tasks at the low and the max reasoning level, 840 runs in all. To make the tasks harder, each one sets a target that the rules make impossible to reach and offers a forbidden tool that would reach it. Some versions also tell the agent that the forbidden tool counts. A program checked every run from its final state. Results: At both levels the agent always followed the rules and never used the forbidden tool, including 30 runs in which it was told that the tool counts while it could still have used it. The low level used about 43% fewer output tokens and 20% less time. Conclusion: For routine office work, a low reasoning le
    
[^63]: ORCA-bench：语言模型智能体对值班排障（Oncall）的准备程度如何？

    ORCA-bench: How Ready Are Language Model Agents for Oncall?

    [https://arxiv.org/abs/2607.28545](https://arxiv.org/abs/2607.28545)

    该论文提出了ORCA-bench基准，将1,079个根因分析任务与真实可观测性工具接口及六天生产级遥测数据相结合，系统评估语言模型智能体在值班根因分析场景中的真实能力。

    

    大型语言模型能够编写、修补和搜索代码，但值班根因分析（RCA）需要的是不同的能力：从模糊的用户报告中出发，对嘈杂的指标、日志、追踪数据和源代码进行推理，且通常发生在事故开始数小时之后。我们提出了ORCA-bench，这是一个将通用编码智能体置于高保真生产环境值班场景中的基准测试。ORCA-bench将1,079个RCA任务与从持续模拟用户负载下、经过OpenTelemetry插桩的微服务系统中收集的六天指标、日志和追踪数据相配对。智能体通过真实的可观测性接口——通过Grafana访问的Prometheus、Jaeger和OpenSearch——调查这些记录的历史数据，并可完全访问应用程序源代码。任务系统地改变报告的具体程度、检测时间以及共现故障场景。真实症状由专家SRE（站点可靠性工程师）审核并签署确认，我们的LLM-as-judge是独立……

    arXiv:2607.28545v3 Announce Type: replace-cross  Abstract: Large language models can write, patch, and search code, but oncall root cause analysis (RCA) demands something different: reasoning over noisy metrics, logs, traces, and source code, starting from ambiguous user-facing reports, often hours after the incident began. We introduce ORCA-bench, a benchmark that puts general-purpose coding agents in a production-fidelity oncall setting. ORCA-bench pairs 1,079 RCA tasks with six days of metrics, logs, and traces collected from an OpenTelemetry-instrumented microservice system under continuous simulated user load. Agents investigate this recorded history through real observability interfaces---Prometheus, Jaeger, and OpenSearch via Grafana---with full access to application source code. Tasks systematically vary report specificity, time-to-detection, and co-occurring fault scenarios. Ground-truth symptoms are curated and signed off by expert SREs, and our LLM-as-judge is independently 
    
[^64]: SIGIL：将智能体技能编译为类型化框架

    SIGIL: Compiling Agent Skills into Typed Harnesses

    [https://arxiv.org/abs/2607.27309](https://arxiv.org/abs/2607.27309)

    SIGIL通过技能编译范式，将自然语言技能确定性编译为可执行代码，显著提升了智能体在执行过程中对技能要求的合规性。

    

    arXiv:2607.27309v2 公告类型：替换 摘要：智能体技能提供了一种可重用的方式来指定多步骤智能体行为，但它们仍然是模型在运行时解释的自然语言规范。因此，即使技能明确规定了所需的工具调用、顺序约束和检查，这些内容也可能被跳过。我们引入了技能编译（Skill Compilation）范式，它将自然语言技能转换为可执行的智能体程序，同时在需要语义决策的地方保留模型判断。我们通过SIGIL实现了这一理念。SIGIL提取基于来源的需求，使用封闭的智能体指令集（AIS）分解它们，将其组合成具有显式所有权、数据流和控制流的AG-IR，并确定性地将验证过的AG-IR降级为可执行代码。在33个公开可用的SKILL.md文件和三个运行时模型上，SIGIL将平均适用指令合规性（AMC）——即执行期间满足的适用技能要求的比例——提高了。

    arXiv:2607.27309v2 Announce Type: replace  Abstract: Agent skills provide a reusable way to specify multi-step agent behavior, but they remain natural-language specifications interpreted by the model at runtime. As a result, required tool calls, ordering constraints, and checks may be skipped even when explicitly prescribed by the skill. We introduce Skill Compilation, a paradigm that translates natural-language skills into executable agent programs while preserving model judgment where semantic decisions are required. We realize this idea in SIGIL. SIGIL extracts source-grounded requirements, decomposes them using a closed Agent Instruction Set (AIS), composes them into AG-IR with explicit ownership, data flow, and control flow, and deterministically lowers validated AG-IR into executable code. Across 33 publicly available SKILL.md files and three runtime models, SIGIL increases mean Applicable-Mandate Compliance (AMC), the fraction of applicable skill requirements satisfied during ex
    
[^65]: CodeNib：一种为编程智能体提供仓库上下文的多视图数据系统

    CodeNib: A Multi-View Data System for Serving Repository Context to Coding Agents

    [https://arxiv.org/abs/2607.25431](https://arxiv.org/abs/2607.25431)

    CodeNib 是首个将同一代码提交的词汇、稠密和结构视图编译在统一清单与源地址契约之下的多视图数据系统，使编程智能体能够相互校验预计算结果并通过单一成本可见的运行时获得搜索、导航和有界上下文服务。

    

    增量代码索引可以在每次更新中都无错误地完成，却仍有一半的时间返回错误的图。编程智能体依赖这类复用的视图进行搜索和导航，然而提供这些视图的系统从未衡量过预计算的答案在哪些地方可以替代可信的路径：定位智能体为每个任务重建图，工具服务器为每个请求查询实时语言服务器，而代码智能数据库服务于开发者而非智能体。CodeNib 是一个使这种衡量成为可能的多视图数据系统。它是首个在单一清单与源地址契约之下编译同一提交的词汇、稠密和结构视图的系统，使词汇命中、图中的出现和稠密块能够相互对照，并与重建结果或实时服务器进行核对，同时通过一个成本可见的运行时为智能体提供搜索、导航和有界上下文。比较实验推翻了三个假设。

    arXiv:2607.25431v2 Announce Type: replace  Abstract: An incremental code index can complete every update without error and still return the wrong graph half the time. Coding agents depend on such reused views for search and navigation, yet the systems that supply them never measure where a precomputed answer may stand in for the trusted route: localization agents rebuild a graph per task, tool servers query a live language server per request, and code-intelligence databases serve developers, not agents.   CodeNib is a multi-view data system that makes that measurement possible. It is the first to compile lexical, dense, and structural views of one commit behind a single manifest and source-address contract, so that a lexical hit, a graph occurrence, and a dense block can be checked against each other and against a rebuild or a live server, and it serves search, navigation, and bounded context to agents through one cost-visible runtime.   The comparisons overturn three assumptions. Exec
    
[^66]: HEDGEHOG：通过严格过滤的药物生成器分层评估

    HEDGEHOG: Hierarchical Evaluation of Drug Generators Through Rigorous Filtration

    [https://arxiv.org/abs/2607.13155](https://arxiv.org/abs/2607.13155)

    提出了HEDGEHOG——一个模拟命中化合物鉴定工作流程的统一六阶段严格过滤评估基准，用于更真实地评估分子生成器的药用合理性，并在KRAS G12D案例中评估了22个生成模型。

    

    生成式分子模型可以通过从头设计提出新的候选化合物来支持早期药物发现。在实践中，有用的候选化合物必须在靶点相关活性、理化性质以及其他多参数设计约束之间取得平衡。然而，通常用于评估分子生成器的标准指标只能较弱地反映所生成化合物是否具有药用合理性以及是否适合下游计算。这会导致对模型性能的认识不完整以及计算资源的低效使用。我们提出了HEDGEHOG，一个统一的六阶段过滤基准，其构建方式模拟命中化合物鉴定工作流程：(i) 预处理；(ii) 理化描述符筛选；(iii) 结构警报与图完整性检查；(iv) 合成可行性；(v) 分子对接；以及 (vi) 三维构象与相互作用检查。我们在KRAS G12D案例研究中评估了22个生成模型，使用三个r（原文摘要在此处被截断）

    arXiv:2607.13155v2 Announce Type: replace  Abstract: Generative molecular models can support early drug discovery by proposing new candidate compounds de novo. In practice, useful candidates must balance target-relevant activity, physicochemical properties, and other multiparameter design constraints. However, standard metrics commonly used to evaluate molecular generators only weakly reflect whether the generated compounds are medicinally plausible and suitable for downstream computation. This can produce an incomplete view of model performance and inefficient use of computational resources. We introduce HEDGEHOG, a unified six-stage filtration benchmark that is constructed as a hit identification workflow: (i) preprocessing; (ii) physicochemical descriptor screening; (iii) structural alerts and graph-sanity checks; (iv) synthesis feasibility; (v) docking; and (vi) three-dimensional pose and interaction checks. We evaluated 22 generative models in a KRAS G12D case study, using three r
    
[^67]: 智能体代码比人类代码更难维护吗？

    Is Agent Code Less Maintainable Than Human Code?

    [https://arxiv.org/abs/2606.21804](https://arxiv.org/abs/2606.21804)

    本研究提出 CodeThread 框架，通过受控实验发现智能体基于智能体生成代码解决任务的效率低于基于人类代码（任务解决率最多下降 13.1%），且传统可维护性指标无法解释这一差异。

    

    可维护性是软件工程的核心维度，塑造着代码随时间推移如何被编写、审查和开发。尽管编码智能体在单 issue 任务上已展现出强大性能，但当未来的智能体在其代码之上继续构建时，这些代码的可维护性如何仍不清楚，而这可能导致不断叠加的下游影响。我们研究了在这些维护场景中智能体代码与人类代码的对比，并提出了 CodeThread——一个从仓库级编码基准中构建受控实验的框架。将 CodeThread 应用于四个前沿编码智能体和四个基准后，我们发现智能体在基于智能体代码解决任务时效果不如基于人类代码，任务解决率下降幅度高达 13.1%。回归分析表明，许多传统软件工程可维护性指标无法解释这一差异；相反，最清晰的信号更为微妙……

    arXiv:2606.21804v2 Announce Type: replace-cross  Abstract: Maintainability is a core dimension of software engineering, shaping how code is written, reviewed, and developed over time. While coding agents have demonstrated strong performance on single-issue tasks, it remains unclear how maintainable their code is when future agents build on top of it, potentially leading to compounding downstream effects. We investigate how agent code compares to human code in these maintenance settings, presenting CodeThread, a framework to construct controlled experiments from repository-level coding benchmarks. Applying CodeThread to four frontier coding agents and four benchmarks, we find that agents are less effective at resolving tasks when building on agent code compared to human code, with task resolve rate drops of up to 13.1%. Regression analysis reveals that many traditional software engineering maintainability metrics do not explain this difference. Instead, the clearest signals are subtler 
    
[^68]: 代码寿命生存分析（CLSA）：基于AST感知挖掘预测源代码行的存活

    Code Lifespan Survival Analysis (CLSA): Predicting the Survival of Source Code Lines Using AST-Aware Mining

    [https://arxiv.org/abs/2606.04993](https://arxiv.org/abs/2606.04993)

    该论文提出了首个对单行代码删除风险进行建模的框架CLSA，通过对120个TypeScript仓库中3250万次代码行诞生事件的生存分析，发现AST结构和代码行长度等可静态计算的协变量是预测代码行删除风险的最强因子。

    

    背景：预测哪些源代码行将被删除——以及何时被删除——对于维护工作和代码审查的优先级排序具有重要意义。现有的MSR（挖掘软件仓库）方法在文件或方法粒度上进行分析，掩盖了单条语句层面的风险。目标：我们提出了代码寿命生存分析（CLSA），这是首个从协变量出发对单行代码删除风险进行建模的框架——此前的行粒度研究虽然估计了代码行的寿命，但未发现显著的预测因子。CLSA将每一行代码视为右删失的研究对象，并从结构、上下文和时间协变量来估计其删除风险；其最强的预测因子可以从单个文件静态计算得出（AST结构加上代码行的标记数量），而无需版本历史或缺陷数据。方法：我们从120个开源TypeScript仓库中挖掘了3250万次代码行诞生事件。通过一个五阶段匹配流水线，将真正的代码删除与重构噪声区分开来，避免了830万次虚假的“死亡”记录。我们拟合了Cox比例…

    arXiv:2606.04993v4 Announce Type: replace  Abstract: Context: Predicting which source lines will be deleted - and when - matters for maintenance and review prioritization. Existing MSR approaches work at file or method granularity, masking individual-statement risk. Objective: We introduce Code Lifespan Survival Analysis (CLSA), the first framework to model individual-line deletion risk from covariates - where prior line-granularity work estimated lifespans but found no significant predictors. CLSA treats each line as a right-censored subject and estimates deletion risk from structural, contextual, and temporal covariates; its strongest predictors are computable statically from one file (AST structure plus line token count), without version history or bug data. Method: We mine 32.5 million line birth events from 120 open-source TypeScript repositories. A 5-stage matching pipeline separates true deletions from refactoring noise, preventing 8.3 million false deaths. We fit a Cox Proporti
    
[^69]: 在黑暗中摸索：为什么对组件的共同理解至关重要

    Poking Around in the Dark: Why a Shared Understanding of Components Matters

    [https://arxiv.org/abs/2606.02442](https://arxiv.org/abs/2606.02442)

    该论文通过自底向上分析软件开发生命周期中的组件包含机制，并使用六种编程语言的真实数据系统评估五种主流SBOM生成工具，揭示了不同工具对组件的定义与识别存在显著差异，证明业界缺乏关于SBOM应包含哪些组件的共同理解，现有技术尚不足以保障软件供应链安全。

    

    软件物料清单（SBOM）通过列出应用程序中包含的组件，旨在支持及时识别存在漏洞的组件并确保软件供应链的安全。然而，我们对以下基本假设提出质疑：即业界对SBOM中应列出的组件存在共识，且当前技术足以保障软件供应链的安全。首先，我们对软件开发生命周期中的组件包含机制（Component Inclusion Mechanisms, CIM）进行了自底向上的分析。然后，我们系统地分析了四种流行的SBOM生成工具——cdxgen、syft、trivy、ORT以及微软的sbom-tool，以了解它们如何定义和识别相关组件。最后，我们使用涵盖Python、Java、Go、PHP、Rust和C等编程语言的真实数据（ground truth）对这些工具进行了评估。尽管当今的工具在识别组件方面迈出了一步，但我们的结果表明，没有任何工具能够覆盖所有识别出的组件（摘要原文在此处截断）。

    arXiv:2606.02442v2 Announce Type: replace  Abstract: By listing the components included in an application, Software Bills of Materials (SBOMs) are intended to support the timely identification of vulnerable components and ensure the security of the software supply chain. However, we question the underlying assumption that there is agreement on the components to be listed in an SBOM and that current technology is sufficient to secure the software supply chain.   First, we propose a ground-up analysis of Component Inclusion Mechanisms (CIM) in the software's development lifecycle. Then we systematically analyze the four popular SBOM generation tools, cdxgen, syft, trivy, ORT, and the Microsoft sbom-tool, to understand how they define and identify relevant components. Finally, we assess these using a ground truth across the programming languages Python, Java, Go, PHP, Rust, and C.   While today's tools are a step toward identifying components, our results show that no tool covers all iden
    
[^70]: 铭记你的足迹：面向一致性与分层结构的仓库级代码文档的记忆引导长程智能体框架

    Remember Your Trace: Memory-Guided Long-Horizon Agentic Framework for Consistent and Hierarchical Repository-Level Code Documentation

    [https://arxiv.org/abs/2605.14563](https://arxiv.org/abs/2605.14563)

    提出MemDocAgent长程智能体框架，通过依赖感知的遍历引导与基于共享记忆RepoMemory的记忆引导智能体交互，在覆盖整个仓库的单一集成上下文中生成一致且具有分层结构的仓库级代码文档。

    

    arXiv:2605.14563v3 公告类型： replace-cross 摘要：自动化代码文档生成对现代软件开发至关重要，它为人类开发者和编码智能体提供了赖以浏览大型代码库的上下文基础。现有的仓库级方法独立处理各个组件，导致冗余检索以及文档间描述冲突，同时生成的输出缺乏分层结构。因此，我们提出了MemDocAgent，这是一个长程智能体框架，能够在覆盖整个仓库的单一集成上下文中生成文档。它结合了两个组件：（i）依赖感知的遍历引导，其预先确定遵循依赖关系与粒度层级的遍历顺序；（ii）记忆引导的智能体交互，其中智能体与RepoMemory进行交互——这是一个通过读取、写入和验证操作来积累先前工作痕迹的共享记忆。通过深入的多维度评估，Me（摘要在此处被截断）

    arXiv:2605.14563v3 Announce Type: replace-cross  Abstract: Automated code documentation is essential for modern software development, providing the contextual grounding that both human developers and coding agents rely on to navigate large codebases. Existing repository-level approaches process components independently, causing redundant retrieval and conflicting descriptions across documents while producing outputs that lack hierarchical structure. Therefore, we propose MemDocAgent, a long-horizon agentic framework that generates documentation within a single, integrated context spanning the entire repository. It combines two components: (i) Dependency-Aware Traversal Guiding that predetermines a traversal order respecting dependency and granularity hierarchies; (ii) Memory-Guided Agentic Interaction, in which the agent interacts with RepoMemory, a shared memory accumulating prior work traces through read, write, and verify operations. Through an in-depth multi-criteria evaluation, Me
    
[^71]: PlayCoder：让大语言模型生成的GUI代码可运行

    PlayCoder: Making LLM-Generated GUI Code Playable

    [https://arxiv.org/abs/2604.19742](https://arxiv.org/abs/2604.19742)

    该论文提出基于43个多语言GUI应用构建的PlayEval基准与Play@k指标，从交互流程和UI逻辑角度评估大语言模型生成的GUI代码能否真正可运行，弥补了传统测试用例评估方式的不足。

    

    大语言模型（LLM）在代码生成方面已取得显著成果，但其在生成GUI应用程序（尤其是游戏）方面的能力仍未得到充分研究。现有基准主要通过测试用例来评估正确性，这对于GUI应用程序而言是不够的，因为这类系统是交互式的、事件驱动的，并且需要在用户操作序列中保持正确的状态转换。因此，对GUI应用的评估应考虑交互流程和UI逻辑，而不仅仅是通过与不通过的结果。为研究这一问题，我们提出了PlayEval，这是一个基于43个多语言（Python、TypeScript和JavaScript）GUI应用程序构建的、具有仓库感知能力的基准。与先前难以适配桌面环境的GUI基准不同，PlayEval涵盖了六大主要GUI应用类别，并直接支持代码生成评估。我们进一步提出了Play@k指标，用于衡量k个生成程序中是否至少有一个能够真正可运行（可玩）。

    arXiv:2604.19742v2 Announce Type: replace  Abstract: Large language models (LLMs) have achieved strong results in code generation, but their ability to generate GUI applications, especially games, remains insufficiently studied. Existing benchmarks mainly evaluate correctness through test cases, which are inadequate for GUI applications because these systems are interactive, event-driven, and require correct state transitions across sequences of user actions. Their evaluation therefore should consider interaction flows and UI logic rather than only pass/fail outcomes. To study this problem, we introduce PlayEval, a repository-aware benchmark built from 43 multilingual GUI applications in Python, TypeScript, and JavaScript. Unlike prior GUI benchmarks that are difficult to adapt to desktop environments, PlayEval covers six major GUI application categories and directly supports code-generation evaluation. We further propose Play@k, a metric that measures whether at least one of *k* gener
    
[^72]: CIRCLE：一个从现实世界视角评估人工智能的框架

    CIRCLE: A Framework for Evaluating AI from a Real-World Lens

    [https://arxiv.org/abs/2602.24055](https://arxiv.org/abs/2602.24055)

    CIRCLE是一个六阶段、基于生命周期的AI评估框架，通过将AI技术栈之外的利益相关者需求转化为可衡量的前瞻性信号，弥合了模型性能指标与AI系统真实部署效果之间的差距。

    

    本研究提出了CIRCLE，一个基于生命周期、包含六个阶段的框架，旨在弥合以模型为中心的性能指标与AI系统部署实际效果之间的现实差距。当前的方法，如MLOps框架和AI模型基准测试，虽然能为系统稳定性和模型能力提供细致的洞察，但无法为AI技术栈之外的决策者提供系统性证据，以说明这些系统在真实世界环境中的实际表现，以及随时间推移对其组织产生的影响。CIRCLE通过将技术栈之外的利益相关者的优先事项转化为可衡量的信号，使TEVV（测试、评估、验证与确认）中的验证（Validation）阶段得以落地实施。不同于通常局限于局部的参与式设计，也不同于往往具有回溯性的算法审计，CIRCLE提供了一个结构化的、前瞻性的协议，用于将情境敏感的定性洞察与可扩展的定量指标相联系。通过整合……

    arXiv:2602.24055v5 Announce Type: replace  Abstract: This study proposes CIRCLE, a six-stage, lifecycle-based framework to bridge the reality gap between model-centric performance metrics and AI system outcomes in deployment. Current approaches such as MLOps frameworks and AI model benchmarks offer detailed insights into system stability and model capabilities, but they do not provide decision makers outside the AI stack with systematic evidence of how these systems actually behave in real world contexts or affect their organizations over time. CIRCLE operationalizes the Validation phase of TEVV (Test, Evaluation, Verification, and Validation) by translating priorities of stakeholders outside the stack into measurable signals. Unlike participatory design which often remains localized, or algorithmic audits which are often retrospective, CIRCLE provides a structured, prospective protocol for linking context sensitive qualitative insights to scalable quantitative metrics. By integrating 
    
[^73]: 评估 AGENTS.md：仓库级上下文文件对编程智能体真的有帮助吗？

    Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?

    [https://arxiv.org/abs/2602.11988](https://arxiv.org/abs/2602.11988)

    本文首次严格评估了 AGENTS.md 等仓库级上下文文件的实际效果，发现它们并不能普遍提升编程智能体的任务成功率，反而平均增加超过 20% 的推理成本。

    

    软件开发中一种广泛流行的做法是使用上下文文件（如 AGENTS.md）针对仓库定制编程智能体。尽管智能体开发者强烈鼓励这种做法，但目前尚无严格的研究来调查此类上下文文件是否对真实任务确实有效。在这项工作中，我们研究了这一问题，并在两种互补的设置下评估了编程智能体的任务完成性能：一是来自流行仓库的既有 SWE-bench 任务（配以 LLM 生成的上下文文件），二是一个新颖的问题集合，这些仓库中含有开发者主动提交的上下文文件。令人惊讶的是，我们发现提供上下文文件通常并不能提高任务成功率，反而平均增加了超过 20% 的推理成本。这一观察结果在不同的 LLM、不同的编程智能体、以及 LLM 生成的和开发者提交的上下文文件中均成立。具体而言，我们发现虽然……

    arXiv:2602.11988v3 Announce Type: replace-cross  Abstract: A widespread practice in software development is to tailor coding agents to repositories using context files, such as AGENTS.md. Although this practice is strongly encouraged by agent developers, there is currently no rigorous investigation into whether such context files are actually effective for real-world tasks. In this work, we study this question and evaluate coding agents' task completion performance in two complementary settings: established SWE-bench tasks from popular repositories, with LLM-generated context files, and a novel collection of issues from repositories containing developer-committed context files. Surprisingly, we find that providing context files does not generally improve task success rates, while increasing inference cost by over 20% on average. This observation holds across different LLMs, coding agents, and for both LLM-generated and developer-committed context files. Specifically, we find that while
    
[^74]: 连接用户反馈与系统诊断：从应用评论中复现移动应用性能问题

    Bridging User Feedback and System Diagnosis: Reproducing Mobile Performance Issues from Reviews

    [https://arxiv.org/abs/2508.11147](https://arxiv.org/abs/2508.11147)

    本文提出了首个方法RevPerf，通过语义检索和提示工程整合互补的应用评论信息，实现从用户评论自动复现移动应用性能问题，从而弥合用户反馈与系统诊断之间的鸿沟。

    

    移动应用性能是影响用户体验的关键因素。然而，性能问题在开发环境中 notoriously 难以检测，它们在这些环境中往往表现得不够明显，使其诊断更具挑战性。在这种情况下，来自不同设备和不同使用场景的最终用户的应用评论，可以为新出现的性能问题提供及时且上下文丰富的信息。然而，与结构化的缺陷报告不同，应用评论由最终用户撰写，往往更加模糊，且单条评论通常只能提供潜在问题的部分描述。为了弥合这一差距，我们提出了RevPerf，这是首个通过利用和综合应用评论中的信息来自动复现移动应用性能问题的方法。RevPerf通过语义检索获取互补的评论，并使用提示工程将其整合，用补充信息丰富原始评论……

    arXiv:2508.11147v3 Announce Type: replace  Abstract: Mobile application performance is a vital factor for user experience. Yet, performance issues are notoriously difficult to detect in development environments, where they often manifest less conspicuously, making their diagnosis more challenging. In this setting, app reviews from end users across diverse devices and usage contexts can provide timely and context-rich information about emerging performance issues. However, unlike structured bug reports, app reviews are written by end-users and tend to be more ambiguous, with individual reviews often providing only partial descriptions of the underlying issue. To bridge this gap, we present RevPerf, the first approach to automatically reproduce mobile application performance issues by leveraging and synthesizing information from app reviews. RevPerf retrieves complementary reviews via semantic retrieval and uses prompt engineering to integrate them, enriching the original review with per
    

