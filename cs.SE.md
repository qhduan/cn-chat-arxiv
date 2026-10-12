# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Returning to Development States: When Full Sandbox Snapshots Help Coding Agents](https://arxiv.org/abs/2610.13841) | 基于完整沙箱快照的 Go-Explore 式搜索可以提升编码智能体的解决率与效率，其效果主要依赖于代码仓库之外的状态（如数据库、服务及其他运行时状态）。 |
| [^2] | [MARGoS: A Configuration-Driven Platform for Reproducible MARL Experiments in ARGoS](https://arxiv.org/abs/2610.13772) | MARGoS是一个轻量级配置驱动平台，通过配置哈希、端到端种子传播和不可变配置等机制，系统性地解决了基于MARL的ARGoS群体机器人实验中复用与可复现性不足的问题。 |
| [^3] | [A Baseline Assessment for the Agentic Web Technology Stack](https://arxiv.org/abs/2610.13703) | 本文提出了MASTIF多智能体系统测试框架，对由3个LLM、3种协议和4个智能体实现框架组成的智能体网络技术栈在300个Mind2Web基准任务上进行了基线评估，发现针对特定任务选择技术栈可以提升任务完成率、效率得分和社会价值稳定性。 |
| [^4] | [(When) Should a Software Factory be Lights-off? Benchmarking Code Review Agents on Bug Detection](https://arxiv.org/abs/2610.13668) | 该论文提出了CRBugBench——首个包含1,000个开源任务和800个工业专有仓库任务的代码审查智能体缺陷检测基准，发现最先进的审查智能体（claude-opus-5结合copilot-cli）可捕获71.9%的人工验证缺陷，为评估“熄灯”式全自主软件工厂的可行性提供了系统化工具。 |
| [^5] | [A-SAW: The Agentic Software Architecture Workflow](https://arxiv.org/abs/2610.13647) | 该论文提出A-SAW工作流，通过持久化的智能体层让IDE集成智能体基于C4模型等来源自动起草架构文档，并辅以强制人工审查关口，从而降低架构文档维护成本并提升其可信度。 |
| [^6] | [SWE-DeceptionBench: Benchmarking Deceptive Behaviors of LLMs Across the Software Development Lifecycle](https://arxiv.org/abs/2610.13626) | 该论文提出了SWE-DeceptionBench基准，系统评估大语言模型在软件开发生命周期六个阶段中的主动欺骗与被动欺骗行为，并通过中立、奖励、压力三种条件考察外部诱导对欺骗行为的影响。 |
| [^7] | [Not All Correct Fixes Run Alike: An Empirical Study of Patch Performance in Repository-Level Issue Resolution](https://arxiv.org/abs/2610.13563) | 该研究将基准自带的 FAIL_TO_PASS 测试重新用于评估补丁的运行时性能，发现通过正确性测试的不同补丁在运行性能上存在显著差异，即并非所有正确的修复运行起来都一样。 |
| [^8] | [ProgramGym: Scaling Training Environments for Long-Horizon Program Reproduction Agents](https://arxiv.org/abs/2610.13395) | ProgramGym提出了一个由Program-Env和Program-Agent组成的框架，通过反馈驱动的循环从代码仓库自动构建可验证的训练环境，实现了对长周期程序复现智能体的可扩展训练。 |
| [^9] | [AgentSWE: Can Coding Agents Build the Agent You Actually Want?](https://arxiv.org/abs/2610.13361) | 提出了首个检验编码智能体能否忠实构建出真正满足用户需求的智能体的基准 AgentSWE，其以“委托”形式覆盖创建、编辑、优化三个开发阶段共25个任务，并通过保留用例和需求驱动的严格评分标准进行验收。 |
| [^10] | [Agent4Spec: An LLM-based Agentic System to Optimize API Specifications through Automated Validation and Market Analysis](https://arxiv.org/abs/2610.13273) | Agent4Spec提出了一个自主多智能体框架，通过验证流水线自动检测并纠正OpenAPI语法错误，同时通过市场分析流水线开展竞争研究、识别与行业领先者的功能差距并自动生成缺失的API规范功能，实现了API规范的持续优化与演进。 |
| [^11] | [Understanding and Mitigating Cross-Session Attacks via Persistent Artifacts in Agent Skill Composition](https://arxiv.org/abs/2610.13227) | 本文首次系统刻画了AI智能体技能组合中经由持久化工件传播恶意影响的三种跨会话攻击机制（指令传播、数据边界违反、证据投毒），并据此提出了相应的运行时防御方法。 |
| [^12] | [Understanding Verification Vulnerabilities in Cross-chain Bridges: An Empirical Study](https://arxiv.org/abs/2610.13170) | 本文首次对跨链桥中的跨链验证漏洞（CVV）进行了系统性实证研究，通过分析来自36个真实跨链桥的588份安全审计报告来刻画这一新型漏洞类别。 |
| [^13] | [CAPE - Context-Aware Prioritization Engine for Vulnerability Scoring and Management](https://arxiv.org/abs/2610.13071) | 本文提出上下文感知优先级引擎CAPE，通过结合部署特定证据来增强漏洞优先级排序，解决了CVSS、EPSS和KEV等现有方法因目标与方法不同而导致的不一致、缺乏上下文感知的决策问题。 |
| [^14] | [TestSuiteBench: Coding Agents Struggle to Generate Strong Test Suites from Specifications](https://arxiv.org/abs/2610.13065) | 本文提出了包含165个开源应用程序及详细行为规范的TestSuiteBench基准，并通过实验揭示即使是前沿编程智能体也难以根据规范生成强测试套件，其变异分数低于60%。 |
| [^15] | [Don't Guess, Just Ask: Evaluating Interactive Requirement Clarification for Repository-Level Software Development](https://arxiv.org/abs/2610.13022) | 该论文提出ReqClarifyBench，一个包含341个任务（涵盖游戏、小程序和网页开发）的交互式需求澄清基准，用于评估LLM通过交互将模糊用户意图转化为清晰需求的能力及其对后续代码生成的影响。 |
| [^16] | [SCCGBench: Executability-Oriented Evaluation for Real-World Web Service-Call Code Generation](https://arxiv.org/abs/2610.12920) | 该论文提出了SCCGBench，一个基于真实GitHub代码构建的、面向可执行性的Web服务调用代码生成基准，通过虚拟执行命中、鲁棒性要求及主机/端点/方法/参数多层级诊断，实现了对LLM服务调用代码在实际执行环境中的评估。 |
| [^17] | [Vulnerable Models in the AI Supply Chain: An Empirical Study of Adoption, Propagation, and Usage Patterns](https://arxiv.org/abs/2610.12893) | 本研究对HuggingFace上被InsightDB标记的16,977个已知易受攻击的AI模型进行了大规模实证研究，发现尽管这些模型存在漏洞且已被公开标记，仍广泛出现在超过5,000个开源存储库的21,000多个代码位置中，揭示了AI供应链中易受攻击模型被开发者广泛采用与传播的严重安全隐患。 |
| [^18] | [Evaluation of Software Quality Through Refactoring Commit-Impact Analysis](https://arxiv.org/abs/2610.12890) | 本研究通过分析重构提交前后的代码变更，并结合规模、复杂度、代码异味、漏洞等软件度量指标，对GitHub开源软件进行静态代码分析，实证揭示了代码修改与软件质量之间的相关性。 |
| [^19] | [Co-Modifying Application Code and Deployment Infrastructure: A Causal Study of AI Agent Pull Requests](https://arxiv.org/abs/2610.12849) | 该研究通过涵盖26,999个拉取请求的大规模因果分析，首次证明AI智能体同时修改应用代码与部署基础设施的“双范围”拉取请求会引入额外的集成开销，揭示了当前智能体缺乏部署感知能力的关键局限。 |
| [^20] | [Meta-Reflective Evolution Effectively Improves Production AI Systems](https://arxiv.org/abs/2610.12794) | 提出基于“元反思”机制的进化优化器CoCoEvolve，通过从完整进化历史中检索证据并探测运行中的系统来指导变异，在DABStep上以92.8%的难题准确率登顶排行榜，有效改进了数据智能体、LLM数据流水线和数据工程工作流等生产级AI系统。 |
| [^21] | [Leakage-Resistant Evaluation of AI-Generated Database Specifications in Specification-Driven Migration](https://arxiv.org/abs/2610.12778) | 本文提出了一个抗泄漏的基准LRFB用于评估AI生成的Oracle PL/SQL规范，发现GPT-4o、DeepSeek V4 Pro和Claude Sonnet 4.5等LLM评判器在平衡准确率上与简单的规范长度基线无显著差异。 |
| [^22] | [Quantifying SMT Solver Instability for Formal Verification](https://arxiv.org/abs/2610.12649) | 本文使用变异系数及一种新颖的量化方法，在来自三个软件领域的大规模SMT查询数据集上量化了SMT求解器的不稳定性，发现该不稳定性在各领域均十分显著，且平均求解时间越长的查询不稳定性越强。 |
| [^23] | [SoftwareGym: Scaling Computer-Use Agent Training with Diverse Synthesized Software Environments](https://arxiv.org/abs/2610.12611) | 提出SoftwareGym，一个自动化流水线，能够大规模合成界面多样、功能可执行且经运行时验证的GUI软件环境（已生成12个类别共182个应用，任务通过率达91.5%），并构建SoftwareGym-Bench基准以支持计算机使用智能体的训练与评估。 |
| [^24] | [Beyond Binary Verifiable Rewards: Behavioral Self-Rewarding for SWE Agents](https://arxiv.org/abs/2610.12598) | 提出行为自我奖励（BSR）方法，让策略模型自我评判同一任务中相同验证结果轨迹间的调试行为差异，将其转化为有界辅助奖励，从而在二元可验证奖励之上引入更精细的监督信号，且无需单独的奖励模型或新的环境交互。 |
| [^25] | [SWE-Duel: Towards a Self-Scaling Arena for Coding Agents](https://arxiv.org/abs/2610.12569) | SWE-Duel 提出了一种自扩展的编程智能体评估竞技场：两个智能体以红蓝对抗的方式互相出题——红方在真实仓库中实现新功能并植入隐蔽缺陷，蓝方进行盲修——使任务在评估时动态生成、无内容可泄露，且难度随参与智能体的能力提升而自动扩展。 |
| [^26] | [Fix What's Wrong, Keep What's Right: Hallucination Repair for LLM-Based Optimization Modeling](https://arxiv.org/abs/2610.12529) | 本文提出OPT-Kraken，首个无需训练的事后幻觉修复框架，通过检查问题-模型忠实性、模型-代码一致性和数据基础三个层面来定位并修复LLM优化建模中产生的幻觉。 |
| [^27] | [CTML: A Comprehensive Framework for Continuous Testing in Machine Learning Operations](https://arxiv.org/abs/2610.12515) | 提出了专为MLOps环境设计的CTML持续测试框架，通过涵盖数据质量、漂移检测、模型质量和流水线验证四个方面的模块化标准化架构，为机器学习系统提供全面自动化测试，并可无缝集成到现有CI/CD流水线中。 |
| [^28] | [PyCache Trap: The Inspection-Execution Gap in Agent Skill Scanners](https://arxiv.org/abs/2610.10612) | 该论文揭示了针对Agent技能扫描器的PyCache Trap攻击，利用Python字节码缓存与源代码之间的检查-执行鸿沟实现94-100%的攻击成功率，并提出执行感知验证（EAV）方法在类型化执行图中关联指令、脚本与运行时工件以检测此类隐藏威胁。 |
| [^29] | [Solving VeriContest with a Lean-Backed Rust Verifier](https://arxiv.org/abs/2610.03994) | 该论文提出由Lean 4支持的Rust验证器Rust-Prover，将VeriContest基准中全部1007道题目的1325个定理完全证明，其中1259个定理在不到32小时内以中位3.2分钟、1.17美元的成本完成，大幅超越了此前最强模型13.95%的首试通过率。 |
| [^30] | [OpenCollab: A Multi-Agent Coding Framework with Programmable Collaboration and Controllable Runtime](https://arxiv.org/abs/2609.38345) | 提出OpenCollab多智能体编程框架，通过统一组织设计、共享可控运行时和细粒度事件流追踪，并首次定义Adherence指标来量化协作组织结构的实际遵循程度，揭示任何单一配置变化都会使遵循度从47.2%大幅波动至97%以上。 |
| [^31] | [API Benchmark Scores Do Not Reliably Transfer to Chatbot Interfaces](https://arxiv.org/abs/2609.08861) | 研究通过对ChatGPT、Claude和Gemini的系统审计发现，通过API测得的基准测试分数无法可靠反映聊天机器人界面的真实表现，API与界面间的性能差异之大甚至相当于模型降级一个版本。 |
| [^32] | [SWE-Bench Pro Verified: A Reliable Benchmark for Software Engineering Agents](https://arxiv.org/abs/2609.08149) | 该论文发现 SWE-Bench Pro 基准存在答案泄露导致的奖励破解和任务质量问题，并推出了经过验证的 SWE-Bench Pro Verified 版本，通过防泄露保护与任务修正实现更可靠的软件工程智能体评估。 |
| [^33] | [Beyond Fault Localization: A Trajectory-Level Study of LLM Agents for Microservice Root Cause Analysis](https://arxiv.org/abs/2608.21310) | 本文提出一种轨迹级评估框架，揭示了大语言模型代理在微服务根因分析中即使定位正确，也可能因未能重建故障传播路径而导致诊断质量不足。 |
| [^34] | [The Spec Growth Engine: Spec-Anchored, Code-Coupled, Drift-Enforced Architecture for AI-Assisted Software Development](https://arxiv.org/abs/2606.27045) | 提出规格增长引擎，通过机器可读规格图、主干上下文组装器、垂直切片增长协议和漂移门，解决AI编码代理的上下文爆炸和规格-代码漂移问题。 |
| [^35] | [Bistable by Construction: Wall-Clock-Calibrated State Monitors Have No Moment-Detection Regime at Agent Cadence](https://arxiv.org/abs/2606.19386) | 本文勘误并重新审视了先前报告的“状态饱和陷阱”（实为动作间 dt=0 导致指数衰减从未运行的纯累加器结果），揭示出以采样时间校准与以墙钟时间校准的监测器动态在固定速率流上等价、但在动作间隔差异巨大的智能体流上根本不同，因此墙钟时间校准的状态监测器在智能体节奏下不存在瞬时检测机制。 |
| [^36] | [Identity-Based Signing Shifts, Not Solves, Usability: Evidence from Five Identity-Based Signing Ecosystems](https://arxiv.org/abs/2603.17133) | 该研究首次对Sigstore、OpenPubKey、HashiCorp Vault、Keyfactor和Notary v2五个基于身份的签名生态系统进行了软件仓库挖掘研究，分析了约3,900个GitHub问题讨论，表明基于身份的签名工具是转移而非解决了可用性问题。 |
| [^37] | [COBALT: Contextual Bandit Learning with Offline Trajectories for Multi-Turn Code Generation](https://arxiv.org/abs/2602.03806) | COBALT 利用多轮代码生成是单步可恢复马尔可夫决策过程的观察，通过重用参考模型收集的离线多轮轨迹并采用单步补全的上下文老虎机学习，替代昂贵且不稳定的在线强化学习来高效训练 LLM 进行多轮代码生成。 |
| [^38] | [HAFixAgent: History-Aware Agentic Program Repair](https://arxiv.org/abs/2511.01047) | HAFixAgent通过将基于git blame归因的仓库历史上下文注入修复循环，显著提升了基于大语言模型的自动程序修复效果，对较弱模型的提升最高达64.3%，并大幅超越RepairAgent和BIRCH-feedback等现有最优智能体方法。 |
| [^39] | [Two Birds One Stone: Effective Static Detection of Resource and Communication Deadlocks in Rust Programs](https://arxiv.org/abs/2401.01114) | 本文提出首个能同时检测Rust程序中资源死锁与通信死锁的静态检测工具RcChecker，通过新颖的信号-锁图刻画线程间经由锁变量和条件变量的交互，实现有效检测。 |

# 详细

[^1]: 回归开发状态：完整沙箱快照何时能帮助编码智能体

    Returning to Development States: When Full Sandbox Snapshots Help Coding Agents

    [https://arxiv.org/abs/2610.13841](https://arxiv.org/abs/2610.13841)

    基于完整沙箱快照的 Go-Explore 式搜索可以提升编码智能体的解决率与效率，其效果主要依赖于代码仓库之外的状态（如数据库、服务及其他运行时状态）。

    

    编码智能体在失败之前往往会取得有用的进展，但从头重试会丢弃在此过程中创建的机器状态。这使得故障恢复成为一个有趣的搜索问题：如果中间环境状态保留了有意义的进展，从这些状态恢复并分支可能优于从初始状态反复重启。我们在完整的 Daytona 快照上实现了 Go-Explore 风格的搜索，发现更高且更高效的解决率主要依赖于位于代码仓库之外的进展，例如数据库、服务和其他运行时状态。我们的结果表明，当长周期的编码工作跨越多个服务、且其状态无法从单个 diff 重建时，完整快照搜索最为有效。

    arXiv:2610.13841v1 Announce Type: new  Abstract: Coding agents often make useful progress before they fail, but retry-from-scratch discards the machine state created along the way. This makes failure recovery an interesting search problem: if intermediate environment states preserve meaningful progress, restoring and branching from them could outperform repeatedly restarting from the initial state. We implement Go-Explore-style search over full Daytona snapshots and find that higher and more efficient solve rates depend mainly on progress that sits outside the repository, such as databases, services, and other runtime state. Our results suggest that full-snapshot search is most effective when long-horizon coding work spans multiple services whose state cannot be reconstructed from a single diff.
    
[^2]: MARGoS：一个基于配置驱动的ARGoS可复现多智能体强化学习实验平台

    MARGoS: A Configuration-Driven Platform for Reproducible MARL Experiments in ARGoS

    [https://arxiv.org/abs/2610.13772](https://arxiv.org/abs/2610.13772)

    MARGoS是一个轻量级配置驱动平台，通过配置哈希、端到端种子传播和不可变配置等机制，系统性地解决了基于MARL的ARGoS群体机器人实验中复用与可复现性不足的问题。

    

    多智能体强化学习（MARL）已成为一个重要的研究方向，尤其是在集体机器人学领域，ARGoS模拟器被广泛用于研究群体智能的涌现行为。然而，尽管这一领域不断发展，基于MARL的群体机器人学在复用和可复现性方面的系统性支持仍然有限且缺乏标准化：一个实验往往由手动对齐的模拟器文件、训练脚本、配置值、路径、随机种子和环境细节拼凑而成，因此即使在同一个研究团队内部，复用和复现也依然十分脆弱。为填补这一空白，我们提出了MARGoS，这是一个在本项目中开发的轻量级、配置驱动的平台，它将上述定义实验的各种输入提升为由平台统一管理的一等公民工件。MARGoS通过配置哈希、端到端种子传播、冻结（不可变）配置、结构化日志、环境指纹（原文在此处截断）等机制实现可复现的实验。

    arXiv:2610.13772v1 Announce Type: new  Abstract: Multi-agent reinforcement learning (MARL) has become a prominent research direction, particularly within collective robotics, where the ARGoS simulator is widely used to study emergent swarm behavior. Yet despite this growth, systematic support for reuse and reproducibility in MARL-based swarm robotics remains limited and insufficiently standardized: a single experiment is assembled from manually aligned simulator files, training scripts, configuration values, paths, seeds, and environment details, so reuse and reproduction stay fragile even within one research group. To address this gap, we introduce MARGoS, a lightweight, configuration-driven platform developed in this project that elevates these experiment-defining inputs to first-class, platform-managed artifacts. MARGoS enables reproducible experimentation through configuration hashing, end-to-end seed propagation, frozen (immutable) configurations, structured logging, environment f
    
[^3]: 智能体网络技术栈的基线评估

    A Baseline Assessment for the Agentic Web Technology Stack

    [https://arxiv.org/abs/2610.13703](https://arxiv.org/abs/2610.13703)

    本文提出了MASTIF多智能体系统测试框架，对由3个LLM、3种协议和4个智能体实现框架组成的智能体网络技术栈在300个Mind2Web基准任务上进行了基线评估，发现针对特定任务选择技术栈可以提升任务完成率、效率得分和社会价值稳定性。

    

    针对基于大语言模型（LLM）的智能体，各类协议、框架和技术正以极快的速度被提出，这给任何旨在创建最先进AI解决方案的人带来了复杂的局面。如果缺乏对不同协议、框架和库如何协同工作以构建智能体网络解决方案的评估，开发者和组织甚至在成为智能体网络生态系统一员之前，就会面临各种挑战（例如技能培养、治理、基础设施等）。在此背景下，我们提出了一个用于执行智能体网络评估的系统，称为MASTIF（多智能体系统测试框架），并针对300个Mind2Web基准任务，对智能体网络技术进行了基线评估，其中包括3个大语言模型、3种协议和4个智能体实现框架。结果表明，选择面向特定任务的技术栈可能会带来任务完成率、效率得分和社会价值稳定性得分的提升，并实现某些方面的降低。

    arXiv:2610.13703v1 Announce Type: new  Abstract: The fast pace in which protocols, frameworks, and technologies are being proposed for LLM-based agents results in a complex scenario for anyone aiming to create state-of-the-art AI solutions. Without an assessment on how different protocols, frameworks, and libraries work together to compose an Agentic Web solution, developers and organizations face different challenges (e.g., skilling, governance, infrastructure) even before becoming part of the Agentic Web ecosystem. In this context, we propose a system for performing Agentic Web evaluations called MASTIF (Multi-Agent System TestIng Framework) and present a baseline assessment of Agentic Web technologies, including 3 LLMs, 3 protocols, and 4 agent implementation frameworks, against 300 Mind2Web benchmark tasks. Results suggest that selecting a task-specific technology stack may result in gains of task completion rate, efficiency score, and social value stability scores; and reductions 
    
[^4]: （何时）软件工厂应该实现“熄灯”运营？代码审查智能体在缺陷检测上的基准评测

    (When) Should a Software Factory be Lights-off? Benchmarking Code Review Agents on Bug Detection

    [https://arxiv.org/abs/2610.13668](https://arxiv.org/abs/2610.13668)

    该论文提出了CRBugBench——首个包含1,000个开源任务和800个工业专有仓库任务的代码审查智能体缺陷检测基准，发现最先进的审查智能体（claude-opus-5结合copilot-cli）可捕获71.9%的人工验证缺陷，为评估“熄灯”式全自主软件工厂的可行性提供了系统化工具。

    

    随着编码智能体的不断进步，软件开发者花在代码审查上的时间已超过编写代码的时间。由于人类审查速度落后于智能体代码生成的速度，构建自动化审查代理、迈向几乎无需人工监督的“熄灯”软件工厂的愿景颇具吸引力。为了评估这种完全自主软件工厂的实用性，我们构建了CRBugBench，这是一个用于评估审查智能体能否检测出经人工验证缺陷的新基准。我们构建了一个从仓库历史中挖掘缺陷的数据流水线，并将每个缺陷与其引入该缺陷的PR配对，从而形成审查任务。CRBugBench包含来自98个开源GitHub仓库的1,000个审查任务，以及来自某大型公司专有行业仓库的800个额外任务。随后，我们利用该基准研究智能体的审查质量：我们发现最先进的审查智能体表现良好：最佳组合（使用copilot-cli的claude-opus-5）能够捕获71.9%的

    arXiv:2610.13668v1 Announce Type: new  Abstract: With the advances of coding agents, software developers are spending more time reviewing code than writing code. As human review velocity falls behind agentic code generation, it is tempting to build automated review agents towards a future of "lights-off" software factory with little to no human supervision. To assess the practicality of such a fully autonomous software factory, we construct CRBugBench, a new benchmark that evaluates whether review agents detect human-verified bugs. We build a data pipeline that mines bugs from repository history, and pair each bug with its introducing PR to form a review task. CRBugBench consists of 1,000 review tasks across 98 open-source GitHub repositories, and 800 additional tasks from proprietary industry repositories at one large company.   We then use the benchmark to study agent review quality: We found that SOTA review agents do well: the best (claude-opus-5 with copilot-cli) catches 71.9% of 
    
[^5]: A-SAW：智能体软件架构工作流

    A-SAW: The Agentic Software Architecture Workflow

    [https://arxiv.org/abs/2610.13647](https://arxiv.org/abs/2610.13647)

    该论文提出A-SAW工作流，通过持久化的智能体层让IDE集成智能体基于C4模型等来源自动起草架构文档，并辅以强制人工审查关口，从而降低架构文档维护成本并提升其可信度。

    

    AI在软件工程中的辅助目前主要集中在实现与验证环节，而设计阶段，尤其是架构文档的维护，则相对缺乏支持。“架构即代码”使架构制品能够版本化、可对比差异且可被机器读取，但它并未降低保持这些制品持续更新的工作量，因此设计文档最终仍然是碎片化的、依赖作者个人经验的、且只有部分可信的关于“系统为何呈现当前形态”的记录。本文提出了智能体软件架构工作流（A-SAW），这是一个流程级制品，通过引入一个持久化的智能体层来扩展“架构即代码”：一个集成于IDE中的智能体基于版本化管理的C4模型、类型化的元素目录以及插桩模板起草架构文档。每份草稿在被采纳之前都必须通过强制性的人工审查关口。我们在一个大型工业组织中沿两条路线对其进行了评估。

    arXiv:2610.13647v1 Announce Type: new  Abstract: AI assistance in software engineering focuses on implementation and verification, leaving the design phase, and in particular the maintenance of architecture documentation, comparatively underserved. Architecture as Code makes architectural artifacts versionable, diffable, and machine-readable, but it does not reduce the effort of keeping them updated, so design documentation still ends up as a fragmented, author-dependent, and only partially trusted record of why a system looks the way it does.   In this paper we introduce the Agentic Software Architecture Workflow (A-SAW), a process-level artifact that extends Architecture as Code with a persistent agent layer: an IDE-integrated agent drafts architecture documents from a version-controlled C4 model, a typed element catalog, and instrumented templates. Every draft passes a mandatory human review gate before it is accepted. We evaluate it in a large industrial organization along two trac
    
[^6]: SWE-DeceptionBench：贯穿软件开发生命周期的大语言模型欺骗行为基准测试

    SWE-DeceptionBench: Benchmarking Deceptive Behaviors of LLMs Across the Software Development Lifecycle

    [https://arxiv.org/abs/2610.13626](https://arxiv.org/abs/2610.13626)

    该论文提出了SWE-DeceptionBench基准，系统评估大语言模型在软件开发生命周期六个阶段中的主动欺骗与被动欺骗行为，并通过中立、奖励、压力三种条件考察外部诱导对欺骗行为的影响。

    

    大语言模型（LLM）在软件开发生命周期中的应用日益广泛。这种对LLM日益增长的依赖引发了对其可信度的担忧，其中欺骗行为构成了一项尤为重要的风险。欺骗行为表现为误导性回复，或在伪造、隐瞒、扭曲信息方面提供协助。为研究这一行为，我们提出了SWE-DeceptionBench，这是一个用于评估LLM在软件工程中欺骗行为的基准。该基准涵盖六个生命周期阶段以及两种欺骗触发方式：主动欺骗，即模型明知需求不可行或存在冲突，却仍将其表述为可实现的；以及被动欺骗，即模型帮助满足用户明确提出的、旨在误导他人的请求。我们进一步引入中立、奖励和压力三种条件来研究外部诱导因素的影响，最终构建了包含中英文的720个测试实例。对七个主流大语言模型的评估……

    arXiv:2610.13626v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly used throughout the software development lifecycle. This growing reliance on LLMs raises concerns about their trustworthiness, with deception posing a particularly important risk. Deceptive behavior takes the form of misleading responses or assistance in fabricating, concealing, or distorting information. To study this behavior, we introduce SWE-DeceptionBench, a benchmark for evaluating LLM deception in software engineering. The benchmark covers six lifecycle stages and two deception triggers: Active Deception, where models recognize infeasible or conflicting requirements but still present them as achievable, and Passive Deception, where models help fulfill explicit user requests to mislead others. We further introduce Neutral, Reward, and Pressure conditions to study the effect of external inducements, resulting in 720 test instances in Chinese and English. Evaluating seven mainstream LLM
    
[^7]: 并非所有正确的修复运行时表现都相同：仓库级问题解决中补丁性能的实证研究

    Not All Correct Fixes Run Alike: An Empirical Study of Patch Performance in Repository-Level Issue Resolution

    [https://arxiv.org/abs/2610.13563](https://arxiv.org/abs/2610.13563)

    该研究将基准自带的 FAIL_TO_PASS 测试重新用于评估补丁的运行时性能，发现通过正确性测试的不同补丁在运行性能上存在显著差异，即并非所有正确的修复运行起来都一样。

    

    摘要：诸如 SWE-bench 等仓库级问题解决基准主要依据提交的补丁是否通过正确性测试来进行评估，而对于不同通过测试的补丁的运行时性能，在很大程度上尚未被探索。评估补丁性能至关重要但充满挑战，因为为每个基准实例设计性能测试工作负载的成本很高。然而，这些基准本身已经提供了一个天然的候选方案：内置的 FAIL_TO_PASS（F2P）测试，它会触发受问题修复影响的行为。因此，我们研究了所提供的 F2P 测试能否被重新用于评估补丁性能。总体而言，我们选择了四个前沿模型——Claude Opus 5、GPT 5.6、DeepSeek V4 Pro 和 Qwen 3.8 Max——并与 SWE-agent 配对，解决从 SWE-bench 和 SWE-bench-Live 已验证分割中预先筛选出的可测量语料库，最终得到 262 个至少包含两种不同正确解决方案的案例，其中包括开发（测试）……（原文摘要在此处截断）

    arXiv:2610.13563v1 Announce Type: cross  Abstract: Repository-level issue-resolution benchmarks such as SWE-bench evaluate submitted patches primarily based on whether they pass the correctness tests, leaving the runtime performance of different test-passing patches largely unexplored. Evaluating patch performance is essential but challenging since designing performance workloads for each benchmark instance is costly. However, these benchmarks already provide a natural candidate: the built-in \emph{FAIL\_TO\_PASS} (F2P) tests, which exercise the behavior affected by the issue fix. We therefore investigate whether the provided F2P tests can be repurposed to evaluate patch performance. Overall, we choose four frontier models, Claude Opus~5, GPT~5.6, DeepSeek V4 Pro, and Qwen~3.8 Max, paired with SWE-agent to resolve the pre-filtered measurable corpus from verified splits of SWE-bench and SWE-bench-Live, yielding 262 cases with at least two different correct solutions, including the devel
    
[^8]: ProgramGym：面向长周期程序复现智能体的可扩展训练环境

    ProgramGym: Scaling Training Environments for Long-Horizon Program Reproduction Agents

    [https://arxiv.org/abs/2610.13395](https://arxiv.org/abs/2610.13395)

    ProgramGym提出了一个由Program-Env和Program-Agent组成的框架，通过反馈驱动的循环从代码仓库自动构建可验证的训练环境，实现了对长周期程序复现智能体的可扩展训练。

    

    arXiv:2610.13395v1 公告类型：cross 摘要：编程智能体正越来越多地被用于像软件工程师那样从零开始构建程序，而不是解决代码仓库中的问题或根据给定的需求文档生成仓库。然而，能够促使智能体自主探索所需功能并决定系统设计方式的、可扩展且可验证的训练环境仍然稀缺。此外，现有的智能体工作流也无法支撑长周期的开发过程，即智能体在实现之前先对参考程序进行探索。为了应对这些挑战，我们提出了ProgramGym，这是一个由Program-Env和Program-Agent组成的框架，旨在为长周期程序复现智能体提供可扩展的训练环境。具体而言，Program-Env通过一个反馈驱动的循环从代码仓库构建可验证的训练环境，该循环能够修复运行时沙箱构建问题并提升黑盒测试的质量。利用这一新颖的流水线，我们获得了1,648个训练……（摘要不完整，原文在此处截断）

    arXiv:2610.13395v1 Announce Type: cross  Abstract: Coding agents are increasingly used to build programs from scratch as software engineers do rather than to resolve issues within a repository or to generate a repository from a given requirements document. However, scalable verifiable training environments that force agents to discover what is required and to decide how the system should be designed remain scarce. Moreover, existing agent workflows also cannot sustain long-horizon development in which the agent explores the reference program before implementation. To address these challenges, we introduce ProgramGym, a framework consisting of Program-Env and Program-Agent for scalable training environments of long-horizon program reproduction agents. Specifically, Program-Env constructs verifiable training environments from repositories through a feedback-driven loop that repairs runtime sandbox builds and the quality of black-box tests. Using this novel pipeline, we obtain 1,648 train
    
[^9]: AgentSWE：编码智能体能构建出你真正想要的智能体吗？

    AgentSWE: Can Coding Agents Build the Agent You Actually Want?

    [https://arxiv.org/abs/2610.13361](https://arxiv.org/abs/2610.13361)

    提出了首个检验编码智能体能否忠实构建出真正满足用户需求的智能体的基准 AgentSWE，其以“委托”形式覆盖创建、编辑、优化三个开发阶段共25个任务，并通过保留用例和需求驱动的严格评分标准进行验收。

    

    随着软件的生产逐渐交由软件本身来完成，编码智能体越来越多地被委托去构建贴合用户自身工作流的智能体。这种委托隐含着一个未被言明的假设：交付的智能体将忠实服务于它被构建时所要满足的需求，而不仅仅是表面上看起来如此。目前尚无任何基准测试检验这一假设。我们提出 AgentSWE，一个面向智能体软件工程的基准，涵盖智能体开发生命周期的三个阶段：根据自然语言需求创建完整智能体、编辑生产级智能体代码库，以及朝向可度量目标进行优化。其25个任务中每一个都被设计为一次“委托”而非一场“考试”：构建者仅针对少量示例用例进行迭代，而验收环节则在保留用例上对冻结提交的成果进行评估，这些保留用例旨在覆盖完整需求，并依据源自需求的评分标准以严格的通过/……（原文摘要在此处截断）

    arXiv:2610.13361v1 Announce Type: cross  Abstract: As the production of software passes into the hands of software itself, coding agents are increasingly commissioned to build agents fitted to their users' own workflows. This delegation carries a quiet assumption, that the delivered agent will faithfully serve the requirement it was built for, rather than merely appear to. No existing benchmark tests that assumption. We introduce AgentSWE, a benchmark for agent software engineering spanning three stages of the agent development lifecycle: Creation of a complete agent from a natural-language requirement, Editing of production agent codebases, and Optimization toward a measurable target. Each of its 25 tasks is framed as a commission rather than an exam: the builder iterates against a handful of illustrative cases, while acceptance exercises the frozen submission on held-out cases designed to span the full requirement, grading behavior against requirement-derived rubrics under hard pass/
    
[^10]: Agent4Spec：一个基于大语言模型的智能体系统，通过自动化验证与市场分析优化API规范

    Agent4Spec: An LLM-based Agentic System to Optimize API Specifications through Automated Validation and Market Analysis

    [https://arxiv.org/abs/2610.13273](https://arxiv.org/abs/2610.13273)

    Agent4Spec提出了一个自主多智能体框架，通过验证流水线自动检测并纠正OpenAPI语法错误，同时通过市场分析流水线开展竞争研究、识别与行业领先者的功能差距并自动生成缺失的API规范功能，实现了API规范的持续优化与演进。

    

    API规范是现代软件开发的关键文档，直接影响集成效率和客户端代码生成。尽管近期的研究进展已利用大语言模型（LLM）从源代码生成初始规范，但现有方法将这一过程视为一次性事件，忽视了对这些工件进行持续验证、长期维护以及市场驱动演进的需求。为解决这一问题，我们提出了Agent4Spec，这是一个自主的多智能体框架，包含两条不同的流水线：（1）验证流水线，用于检测并自我纠正OpenAPI语法错误；（2）市场分析流水线，用于开展竞争性研究、识别与行业领先者相比的功能差距，并自动生成缺失的API规范功能。我们在两个精心整理的API规范数据集上对Agent4Spec进行了评估，涵盖通用型和行业特定型的API规范。

    arXiv:2610.13273v1 Announce Type: cross  Abstract: API specifications serve as critical documentation for modern software development, directly impacting integration efficiency and client code generation. While recent advancements have leveraged Large Language Models (LLMs) to generate initial specifications from source code, existing approaches treat this process as a one-time event - overlooking the needs for continuous validation, long-term maintenance, and market-driven evolution of these artifacts. To address this issue, we present Agent4Spec, an autonomous multi-agent framework featuring two distinct pipelines: (1) a validation pipeline that detects and self-corrects OpenAPI syntax errors, and (2) a market analysis pipeline that conducts competitive research, identifies functional gaps against industry leaders, and automatically generates missing API-spec functionalities. We evaluate Agent4Spec on two curated datasets of API specifications spanning both general-purpose and indust
    
[^11]: 理解并缓解智能体技能组合中经由持久化工件发起的跨会话攻击

    Understanding and Mitigating Cross-Session Attacks via Persistent Artifacts in Agent Skill Composition

    [https://arxiv.org/abs/2610.13227](https://arxiv.org/abs/2610.13227)

    本文首次系统刻画了AI智能体技能组合中经由持久化工件传播恶意影响的三种跨会话攻击机制（指令传播、数据边界违反、证据投毒），并据此提出了相应的运行时防御方法。

    

    智能体技能通过将指令、代码和资源打包成可复用的软件包来扩展AI智能体的能力。在技能执行过程中，智能体可能会生成或修改工件，例如文件、配置和缓存数据，这些工件会在后续会话中被其他技能复用。由于这些工件能够跨会话持久存在并被其他技能后续访问，它们引入了一个新的攻击面，可被利用来影响后续执行。然而，这种跨会话技能交互所产生的安全风险在很大程度上仍未被探索。在本工作中，我们研究了基于技能的AI智能体中的跨会话组合攻击，并开发了一种针对这些攻击的运行时防御方法。具体而言，我们刻画了三种攻击机制，即指令传播（Instruction Propagation, IP）、数据边界违反（Data Boundary Violation, DBV）和证据投毒（Evidence Poisoning, EP），恶意影响正是通过这些机制经由持久化工件在技能执行之间进行传播。

    arXiv:2610.13227v1 Announce Type: cross  Abstract: Agent skills extend the capabilities of AI agents by bundling instructions, code, and resources into reusable packages. During skill execution, agents may produce or modify artifacts, such as files, configurations, and cached data, that are reused by other skills in later sessions. Since these artifacts can persist across sessions and later be accessed by other skills, they introduce a new attack surface that can be exploited to influence subsequent executions. However, the security risks arising from such cross-session skill interactions remain largely unexplored. In this work, we investigate cross-session composition attacks in skill-based AI agents and develop a runtime defense against them. Specifically, we characterize three attack mechanisms, namely Instruction Propagation (IP), Data Boundary Violation (DBV), and Evidence Poisoning (EP), through which malicious effects propagate across skill executions via persistent artifacts. T
    
[^12]: 理解跨链桥中的验证漏洞：一项实证研究

    Understanding Verification Vulnerabilities in Cross-chain Bridges: An Empirical Study

    [https://arxiv.org/abs/2610.13170](https://arxiv.org/abs/2610.13170)

    本文首次对跨链桥中的跨链验证漏洞（CVV）进行了系统性实证研究，通过分析来自36个真实跨链桥的588份安全审计报告来刻画这一新型漏洞类别。

    

    随着区块链技术的快速发展，涌现出大量异构区块链，每条链都具有不同的架构、共识机制和应用生态系统。跨链桥因此成为关键的基础设施组件，通常依赖智能合约来促进异构区块链之间的消息传递和资产转移。与单链合约不同，跨链桥合约引入了额外的验证逻辑来验证来自外部链的数据，例如状态证明、消息发送者身份和签名。然而，正确实现此类逻辑具有挑战性。我们将违反跨链验证安全保证的漏洞称为跨链验证漏洞（CVV）。为了刻画CVV，我们首次对这一漏洞类别进行了系统性实证研究。通过分析来自36个真实世界跨链（桥）的588份安全审计报告……

    arXiv:2610.13170v1 Announce Type: cross  Abstract: With the rapid development of blockchain technology, numerous heterogeneous blockchains have emerged, each featuring distinct architectures, consensus mechanisms, and application ecosystems. Cross-chain bridges have consequently become a critical infrastructure component, typically relying on smart contracts to facilitate message passing and asset transfers across heterogeneous blockchains. Unlike single-chain contracts, bridge contracts introduce additional verification logic to validate external-chain data, such as state proofs, message sender identities, and signatures. However, correctly implementing such logic is challenging. We refer to vulnerabilities that violate the security guarantees of cross-chain verification as Cross-chain Verification Vulnerabilities (CVVs). To characterize CVVs, we conduct the first systematic empirical study of this vulnerability class. By analyzing 588 security audit reports from 36 real-world cross-c
    
[^13]: CAPE - 用于漏洞评分与管理的上下文感知优先级引擎

    CAPE - Context-Aware Prioritization Engine for Vulnerability Scoring and Management

    [https://arxiv.org/abs/2610.13071](https://arxiv.org/abs/2610.13071)

    本文提出上下文感知优先级引擎CAPE，通过结合部署特定证据来增强漏洞优先级排序，解决了CVSS、EPSS和KEV等现有方法因目标与方法不同而导致的不一致、缺乏上下文感知的决策问题。

    

    漏洞修补对于保护企业软件生态系统免受网络攻击至关重要。然而，修补工作需要为潜在的系统停机时间以及被修补软件对运行中系统的不可预见影响做好充分准备。因此，有效的风险缓解需要明确考虑受影响软件运行环境的优先级排序策略。虽然基于通用漏洞评分系统（CVSS）、漏洞利用预测评分系统（EPSS）和美国网络安全与基础设施安全局（CISA）已知被利用漏洞（KEV）目录的漏洞优先级排序可以提供帮助，但它们不同的目标和方法往往导致不一致的、缺乏上下文感知的决策，从而造成修复资源的不当分配。在这项工作中，我们提出了上下文感知优先级引擎（CAPE），这是一个漏洞分析框架，它利用部署特定的证据来丰富每个漏洞扫描器报告中报告的漏洞（CVE）……

    arXiv:2610.13071v1 Announce Type: cross  Abstract: Vulnerability patching is critical for safeguarding an enterprise's software ecosystem from cyber attacks. However, patching requires adequate preparation for potential system downtime and for unforeseen effects of the patched software on a live system. Thus, effective risk mitigation requires prioritization strategies that explicitly account for the operational context of the affected software. While vulnerability prioritization based on Common Vulnerability Scoring System (CVSS), Exploit Prediction Scoring System (EPSS) and CISA Known Exploited Vulnerabilities (KEV) catalog, can help, their differing goals and methodologies often lead to inconsistent, context-unaware decisions resulting in mis-allocated remediation efforts. In this work, we present the Context-Aware Prioritization Engine (CAPE), a vulnerability analysis framework that enriches each vulnerability scanner report of vulnerabilities (CVEs) with deployment-specific eviden
    
[^14]: TestSuiteBench：编程智能体难以根据规范生成强测试套件

    TestSuiteBench: Coding Agents Struggle to Generate Strong Test Suites from Specifications

    [https://arxiv.org/abs/2610.13065](https://arxiv.org/abs/2610.13065)

    本文提出了包含165个开源应用程序及详细行为规范的TestSuiteBench基准，并通过实验揭示即使是前沿编程智能体也难以根据规范生成强测试套件，其变异分数低于60%。

    

    编程智能体越来越多地被用于创建大型软件项目。虽然代码生成基准已相应扩展到完整应用程序层面，但测试生成基准仍然局限于单个函数或错误修复。尤其是，从零开始为复杂软件生成完整测试套件仍然是一个未被探索的领域。为填补这一空白，我们提出了TestSuiteBench，这是一个包含165个开源应用程序的基准，每个应用程序都配有结构化且详细的预期行为规范、参考实现、参考测试套件，以及平均74个变异体。每个变异体均从规范中的单个需求生成，并经过精炼以隔离该行为：73%的变异体恰好被一个参考测试所杀死，平均仅被其余测试中的0.2%杀死。我们发现，即使是前沿的编程智能体，在给定我们的规范时也难以生成强测试套件，其变异分数低于60%，尽管它们的测试套件（原文在此处截断）

    arXiv:2610.13065v1 Announce Type: new  Abstract: Coding agents are increasingly used to create large software projects. While code-generation benchmarks have scaled accordingly to full applications, test-generation benchmarks remain limited to individual functions or bug fixes. In particular, generating complete test suites for complex software from scratch remains unexplored. To address this gap, we introduce TestSuiteBench, a benchmark of 165 open-source applications, each with a structured and detailed specification of intended behavior, a reference implementation, a reference test suite, and on average 74 mutants. Each mutant is generated from a single requirement of the specification and refined to isolate that behavior: 73\% are killed by exactly one reference test and, on average, by only 0.2\% of the remaining tests. We find that even frontier coding agents struggle to generate strong test suites given our specifications, achieving mutation scores below 60\%, despite their suit
    
[^15]: 不要猜测，直接询问：评估仓库级软件开发中的交互式需求澄清

    Don't Guess, Just Ask: Evaluating Interactive Requirement Clarification for Repository-Level Software Development

    [https://arxiv.org/abs/2610.13022](https://arxiv.org/abs/2610.13022)

    该论文提出ReqClarifyBench，一个包含341个任务（涵盖游戏、小程序和网页开发）的交互式需求澄清基准，用于评估LLM通过交互将模糊用户意图转化为清晰需求的能力及其对后续代码生成的影响。

    

    随着大语言模型（LLM）在软件实现方面的能力日益增强，一个关键瓶颈逐渐显现：它们是否真正理解用户想要什么。用户往往从模糊的意图出发，许多重要需求需要通过交互来澄清。在仓库级开发中，误解可能会持续传播并导致代价高昂的返工。然而，现有的基准测试通常基于完整规格说明或欠规范的请求来评估代码实现，忽视了编码前的需求澄清环节。以往关于需求澄清的研究主要聚焦于函数级任务。因此，目前尚不清楚LLM通过交互将模糊意图转化为清晰需求的能力究竟如何，以及这种能力会如何影响后续的代码生成。我们提出了ReqClarifyBench，一个面向仓库级软件开发的交互式需求澄清基准。它包含341个任务，涵盖游戏、小程序和网页开发，每个任务都带有……（摘要在此处被截断）

    arXiv:2610.13022v1 Announce Type: cross  Abstract: As LLMs become increasingly capable of implementing software, a critical bottleneck emerges: whether they truly understand what users want. Users often begin with vague intent, leaving important requirements to be clarified through interaction. In repository-level development, misunderstandings can propagate and cause costly rework. However, existing benchmarks typically evaluate implementation from complete specifications or underspecified requests, overlooking pre-coding requirement clarification. Prior clarification studies mainly focus on function-level tasks. It therefore remains unclear how well LLMs can transform vague intent into clear requirements through interaction and how this capability affects subsequent code generation.   We introduce ReqClarifyBench, an interactive requirement clarification benchmark for repository-level software development. It contains 341 tasks spanning game, miniapp, and web development, each with a
    
[^16]: SCCGBench：面向真实世界Web服务调用代码生成的可执行性评估

    SCCGBench: Executability-Oriented Evaluation for Real-World Web Service-Call Code Generation

    [https://arxiv.org/abs/2610.12920](https://arxiv.org/abs/2610.12920)

    该论文提出了SCCGBench，一个基于真实GitHub代码构建的、面向可执行性的Web服务调用代码生成基准，通过虚拟执行命中、鲁棒性要求及主机/端点/方法/参数多层级诊断，实现了对LLM服务调用代码在实际执行环境中的评估。

    

    LLM生成的服务调用代码在服务计算中日益重要，开发者依赖大语言模型（LLM）来组合服务和调用第三方服务。然而，现有方法尚未在实际代码环境中评估服务的调用与执行情况。为支持基于实际执行的评估，我们提出了服务调用代码生成基准（SCCGBench），该基准包含3,135个来自真实GitHub仓库、可追溯且经过去污染过滤的RapidAPI调用函数。SCCGBench采用脱敏意图生成来减少循环泄露问题，并引入了Mock、Robust、ParamF1指标以及主机级、端点级、方法级和参数级的多层级诊断。其中，Mock表示在虚拟执行中同时匹配标准主机、端点和HTTP方法的命中情况，而Robust进一步要求代码具备状态检查和异常处理能力。我们评估了六个代码LLM和一个非神经检索基线……

    arXiv:2610.12920v1 Announce Type: new  Abstract: LLM-generated service-call code is increasingly important for service computing, where developers rely on large language models (LLMs) to compose services and invoke third-party services. However, current approaches have not yet evaluated the invocation and execution of services in the actual code environment. To support execution-grounded evaluation, we present the Service-Call Code Generation Benchmark (SCCGBench), a benchmark of 3,135 traceable and decontamination-filtered RapidAPI call functions from real GitHub repositories. SCCGBench uses redacted-intent generation to reduce circular leakage and introduces Mock, Robust, ParamF1, and host-, endpoint-, method-, and parameter-level diagnostics. Mock denotes a virtual-execution hit that simultaneously matches the gold host, endpoint, and HTTP method, while Robust further requires status checking and exception handling. We evaluate six code LLMs and one non-neural retrieval baseline und
    
[^17]: AI供应链中的易受攻击模型：关于采用、传播与使用模式的实证研究

    Vulnerable Models in the AI Supply Chain: An Empirical Study of Adoption, Propagation, and Usage Patterns

    [https://arxiv.org/abs/2610.12893](https://arxiv.org/abs/2610.12893)

    本研究对HuggingFace上被InsightDB标记的16,977个已知易受攻击的AI模型进行了大规模实证研究，发现尽管这些模型存在漏洞且已被公开标记，仍广泛出现在超过5,000个开源存储库的21,000多个代码位置中，揭示了AI供应链中易受攻击模型被开发者广泛采用与传播的严重安全隐患。

    

    人工智能（AI）模型的快速集成建立了一个复杂的供应链，但许多这些共享的模型制品都隐藏着严重的漏洞，例如不安全的序列化和架构后门。尽管先前的研究已广泛扫描模型注册库以标记恶意制品，但在理解开发人员如何实际采用这些不安全模型并将其集成到下游软件系统方面，仍然存在关键的研究空白。我们通过对InsightDB标记的HuggingFace平台上16,977个已知易受攻击模型的使用和传播情况进行大规模实证研究来填补这一空白。我们跟踪了这些模型在GitHub存储库、软件包、微调链和已部署应用程序中的流行程度。我们的分析显示，尽管这些易受攻击的模型已被公开标记，但它们仍出现在超过5,000个开源存储库的21,000多个代码位置中，其中有3,757个不同的被标记模型（22.1%……

    arXiv:2610.12893v1 Announce Type: cross  Abstract: The rapid integration of Artificial Intelligence (AI) models has established a complex supply chain, but many of these shared artifacts harbor severe vulnerabilities such as insecure serialization and architectural backdoors. While prior efforts have extensively scanned model registries to flag malicious artifacts, a critical gap remains in understanding how developers actually adopt and integrate these unsafe models into downstream software systems. We bridge this gap by conducting a large-scale empirical study on the usage and propagation of 16,977 known vulnerable models on HuggingFace flagged by InsightDB. We track their prevalence across GitHub repositories, software packages, fine-tuning chains, and deployed applications. Our analysis reveals that, despite being publicly flagged, these vulnerable models appear in over 21,000 code locations across more than 5,000 open-source repositories, with 3,757 distinct flagged models (22.1\%
    
[^18]: 通过重构提交影响分析评估软件质量

    Evaluation of Software Quality Through Refactoring Commit-Impact Analysis

    [https://arxiv.org/abs/2610.12890](https://arxiv.org/abs/2610.12890)

    本研究通过分析重构提交前后的代码变更，并结合规模、复杂度、代码异味、漏洞等软件度量指标，对GitHub开源软件进行静态代码分析，实证揭示了代码修改与软件质量之间的相关性。

    

    本研究旨在对代码修改与软件质量之间的相关性进行实证分析。这将通过考察每次重构提交前后所做的代码变更产生的影响来实现，采用软件规模、复杂度、代码异味、漏洞和缺陷检测等软件度量指标。研究目标是对来自GitHub仓库的开源软件，使用多种仓库挖掘和静态代码分析工具进行静态代码分析。我们的方法包括制定基于研究的问题，并利用指定工具获取结果，从而解决该案例研究。所获得的结果已通过表格和图表进行汇总和进一步分析，以达成研究目标。

    arXiv:2610.12890v1 Announce Type: new  Abstract: The aim of this study is to conduct an empirical analysis of the correlation between code modifications and software quality. This will be achieved by examining the effects of changes made before and after each refactoring commit, using software metrics such as size, complexity, code smells, vulnerability, and bug detection. The objective is to conduct static code analysis on open-source software from GitHub repositories using various repository mining and static code analysis tools. Our approach involves formulating research-based questions and utilizing the specified tools to obtain results that will enable us to resolve the case study. The obtained results have been summarised and further analyzed using tables and graphs to achieve the objective.
    
[^19]: 同时修改应用代码与部署基础设施：AI智能体拉取请求的因果研究

    Co-Modifying Application Code and Deployment Infrastructure: A Causal Study of AI Agent Pull Requests

    [https://arxiv.org/abs/2610.12849](https://arxiv.org/abs/2610.12849)

    该研究通过涵盖26,999个拉取请求的大规模因果分析，首次证明AI智能体同时修改应用代码与部署基础设施的“双范围”拉取请求会引入额外的集成开销，揭示了当前智能体缺乏部署感知能力的关键局限。

    

    当前的软件工程智能体主要在隔离的沙盒环境中、针对本地同步测试进行评估。然而，在协作式代码仓库中部署新功能，除了修改应用逻辑之外，还需要同步修改持续集成/持续交付（CI/CD）流水线、容器以及资源供给清单。由于智能体缺乏部署感知能力，尝试这类跨领域修改会向异步集成流水线中引入未经验证的依赖。为了分离“双范围拉取请求”（PR）——即同时修改应用代码和部署基础设施——所带来的因果性集成开销，我们在两个大型数据集（AIDev v5 和 MOSAIC-3M）上评估了26,999个拉取请求，涵盖六种智能体架构以及一个对齐的人类基线。通过确定性的四层文件分类法和精确的4D分层匹配框架，我们在控制应用代码量、保持条件可交换性的前提下，分离出了相关集成指标。

    arXiv:2610.12849v1 Announce Type: cross  Abstract: Current software engineering agents are evaluated predominantly inside isolated sandboxes against synchronous local tests. However, deploying features in collaborative repositories requires modifying continuous integration (CI/CD) pipelines, containers, and provisioning manifests alongside application logic. Because agents lack deployment awareness, attempting cross-domain modifications introduces unverified dependencies into asynchronous integration pipelines. To isolate the causal integration overhead of dual-scope pull requests (PRs)-modifying both application code and deployment infrastructure-we evaluated 26,999 PRs across two large datasets (AIDev v5 and MOSAIC-3M) encompassing six agent architectures and an aligned human baseline. Using a deterministic four-tier file taxonomy and an exact 4D strata-matching framework, we isolated integration metrics under conditional exchangeability while controlling for application volume. We f
    
[^20]: 元反思式进化有效改进生产级AI系统

    Meta-Reflective Evolution Effectively Improves Production AI Systems

    [https://arxiv.org/abs/2610.12794](https://arxiv.org/abs/2610.12794)

    提出基于“元反思”机制的进化优化器CoCoEvolve，通过从完整进化历史中检索证据并探测运行中的系统来指导变异，在DABStep上以92.8%的难题准确率登顶排行榜，有效改进了数据智能体、LLM数据流水线和数据工程工作流等生产级AI系统。

    

    arXiv:2610.12794v1 公告类型：交叉 摘要：由大语言模型（LLM）引导的进化优化已成功改进了提示词、程序，乃至整个AI系统。大多数优化方法从搜索前沿中选取一个候选，然后仅基于其失败情况，在单次LLM调用中对其进行变异。这种对搜索前沿的关注在生产级AI系统上可能陷入停滞，因为孤立的失败很难揭示复杂的隐藏交互。我们提出了CoCoEvolve（Cortex Code Evolve），一个围绕我们称之为“元反思”（meta-reflection）机制构建的进化优化器，其中智能体变异器（Cortex Code）会选择性地从整个进化历史中检索证据，并在验证和提出变异之前探测实际运行的系统。我们将CoCoEvolve应用于优化三类生产级AI系统：数据智能体、基于LLM的数据流水线以及数据工程工作流。在DABStep上，CoCoEvolve登顶公开排行榜，在留出难题集上达到92.8%的准确率，相比之下基线仅为78.4%（摘要原文在此处截断）。

    arXiv:2610.12794v1 Announce Type: cross  Abstract: Large Language Model (LLM)-guided evolutionary optimization has successfully improved prompts, programs, and even entire AI systems. Most optimization methods select a candidate from the search frontier, then mutate it in one LLM pass based only on its failures. This focus on the frontier can stall on production AI systems, where isolated failures reveal little about complex, hidden interactions. We introduce CoCoEvolve (Cortex Code Evolve), an evolutionary optimizer built around what we call meta-reflection, wherein an agentic mutator (Cortex Code) selectively retrieves evidence from the entire evolutionary history and probes the live system before verifying and proposing mutations. We apply CoCoEvolve to optimize three classes of production AI systems: data agents, LLM-powered data pipelines, and data engineering workflows. On DABStep, CoCoEvolve tops the public leaderboard, reaching 92.8% held-out hard accuracy compared to 78.4% for
    
[^21]: 规范驱动迁移中AI生成数据库规范的抗泄漏评估

    Leakage-Resistant Evaluation of AI-Generated Database Specifications in Specification-Driven Migration

    [https://arxiv.org/abs/2610.12778](https://arxiv.org/abs/2610.12778)

    本文提出了一个抗泄漏的基准LRFB用于评估AI生成的Oracle PL/SQL规范，发现GPT-4o、DeepSeek V4 Pro和Claude Sonnet 4.5等LLM评判器在平衡准确率上与简单的规范长度基线无显著差异。

    

    本文提出了一个抗泄漏框架与基准（LRFB），用于测评AI生成的Oracle PL/SQL规范的评估器。该基准包含494对带标签的源代码-规范对，涵盖19个质量指标并归为七个维度；每个指标包含26条记录，划分为开发集与保留集各13条记录，每个子集含三条规范和十条有缺陷的规范。真实标签通过机械化方式与推理隔离，并采用金丝雀测试和标签置换测试进行验证。研究评估了三个LLM评判器，每条记录运行三次，并与两个简单基线进行对比。在247条保留记录上，GPT-4o、DeepSeek V4 Pro和Claude Sonnet 4.5分别达到了0.87、0.89和0.81的平衡准确率，而基于规范长度的基线达到了0.88。精确配对检验无法将任何一个评判器与该基线区分开。然而，评判器仍能提供长度基线所无法获得的信息……

    arXiv:2610.12778v1 Announce Type: new  Abstract: This paper presents a leakage-resistant framework and benchmark (LRFB) for measuring evaluators of AI-generated Oracle PL/SQL specifications. The benchmark contains 494 labelled source-specification pairs across 19 quality metrics grouped into seven dimensions; each metric contains 26 records divided into development and held-out subsets of 13 records each, with three clean and ten defective specifications per subset. Ground truth is mechanically sealed from inference and checked with canary and label-permutation tests. Three LLM judges were evaluated with three runs per record and compared against two trivial baselines. On 247 held-out records, GPT-4o, DeepSeek V4 Pro, and Claude Sonnet 4.5 reached balanced accuracies of 0.87, 0.89, and 0.81, respectively, while a specification-length baseline reached 0.88. Exact paired tests did not separate any judge from that baseline. The judges nevertheless provided information unavailable to lengt
    
[^22]: 量化SMT求解器不稳定性以用于形式化验证

    Quantifying SMT Solver Instability for Formal Verification

    [https://arxiv.org/abs/2610.12649](https://arxiv.org/abs/2610.12649)

    本文使用变异系数及一种新颖的量化方法，在来自三个软件领域的大规模SMT查询数据集上量化了SMT求解器的不稳定性，发现该不稳定性在各领域均十分显著，且平均求解时间越长的查询不稳定性越强。

    

    程序验证在软件开发项目中已变得十分普遍，其可行性在很大程度上归功于SMT求解器在自动化模型检测及其他验证技术方面的潜力。然而，SMT不稳定性——即语义等价的源代码排列变体需要长得多的求解时间甚至超时的现象——已被发现会打断工作流程，并使软件工程师更倾向于回避验证。以往测量SMT不稳定性的工作由于该现象产生的异常运行时间分布而难以对其进行量化。在本文中，我们使用变异系数来对来自三个软件领域的大量SMT查询数据集中的不稳定性进行建模。我们的结果表明，SMT不稳定性在各个领域都很显著，并且对于平均耗时更长的查询，不稳定性会进一步增加。此外，通过一种新颖的量化不稳定性的方法，我们展示了一种不同的方法

    arXiv:2610.12649v1 Announce Type: new  Abstract: Program verification has become commonplace in software development projects, and much of its feasibility can be attributed to the potential of SMT solvers to automate model checking and other verification techniques. However, SMT instability, a phenomenon where semantically equivalent permutations of source code take much longer to solve or timeout, has been found to interrupt workflows and make software engineers more likely to avoid verification. Previous efforts to measure SMT instability have had difficulty quantifying it due to the abnormal runtime distributions it produces. In this paper, we use the coefficient of variation to model instability across an extensive dataset of SMT queries from three software domains. Our results show that SMT instability is significant in each area, and that it increases for queries that take longer on average. However, using a novel method to quantify instability, we show that a different approach 
    
[^23]: SoftwareGym：通过多样化合成软件环境扩展计算机使用智能体的训练

    SoftwareGym: Scaling Computer-Use Agent Training with Diverse Synthesized Software Environments

    [https://arxiv.org/abs/2610.12611](https://arxiv.org/abs/2610.12611)

    提出SoftwareGym，一个自动化流水线，能够大规模合成界面多样、功能可执行且经运行时验证的GUI软件环境（已生成12个类别共182个应用，任务通过率达91.5%），并构建SoftwareGym-Bench基准以支持计算机使用智能体的训练与评估。

    

    计算机使用智能体（CUA）正在快速进步，然而大规模合成多样化且可靠的软件环境仍然具有挑战性。原生应用程序环境需要大量的构建与集成工作，这限制了应用的覆盖范围，并增加了针对特定应用过拟合的风险。而浏览器托管环境所能提供的交互深度和工作流复杂度则较为有限。这些局限性催生了SoftwareGym——一个用于合成GUI软件环境的自动化流水线，其生成的环境具备多样化界面、可执行的功能以及经运行时验证的行为。该系统生成结构多样的设计，将其实现为可运行的应用程序，并通过GUI评估与迭代修复来确保功能可靠性。借助SoftwareGym，我们合成了涵盖十二个类别的182个应用程序，平均GUI任务通过率达到91.5%。我们构建了SoftwareGym-Bench，一个包含120个任务的评估套件，用于评测……（原文摘要至此被截断）

    arXiv:2610.12611v1 Announce Type: cross  Abstract: Computer-use agents (CUAs) are rapidly improving, yet synthesizing diverse and reliable software environments at scale remains challenging. Native application environments require substantial construction and integration effort, which limits application coverage and increases the risk of application-specific overfitting. Browser-hosted environments provide limited interaction depth and workflow complexity. These limitations motivate SoftwareGym, an automated pipeline for synthesizing GUI software environments with diverse interfaces, executable functionality, and runtime-verified behavior. It generates structurally diverse designs, realizes them as runnable applications, and uses GUI evaluation with iterative repair to ensure functional reliability. Using SoftwareGym, we synthesize 182 applications across twelve categories and achieve a mean GUI task pass rate of 91.5%. We construct SoftwareGym-Bench, a 120-task evaluation suite spanni
    
[^24]: 超越二元可验证奖励：面向软件工程智能体的行为自我奖励

    Beyond Binary Verifiable Rewards: Behavioral Self-Rewarding for SWE Agents

    [https://arxiv.org/abs/2610.12598](https://arxiv.org/abs/2610.12598)

    提出行为自我奖励（BSR）方法，让策略模型自我评判同一任务中相同验证结果轨迹间的调试行为差异，将其转化为有界辅助奖励，从而在二元可验证奖励之上引入更精细的监督信号，且无需单独的奖励模型或新的环境交互。

    

    基于可验证奖励的强化学习（RLVR）通过可执行测试为软件工程智能体提供可靠的监督信号，但二元奖励无法区分调试质量差异巨大的轨迹。我们研究了当前策略能否评判自身的轨迹，并利用这些评判来提升任务性能，而无需单独训练的奖励模型。为此，我们提出了行为自我奖励（BSR），这是一种奖励精炼方法，它复用策略模型来比较同一任务中具有相同验证器结果的多条轨迹之间的可观察调试行为。BSR将成对偏好聚合为任务局部分数，并将其转换为有界的辅助奖励，在保持验证器定义的通过/失败排序的同时引入奖励对比。这无需新的环境交互，即可从已收集的轨迹中获得额外的监督信号。

    arXiv:2610.12598v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) provides reliable supervision for software engineering agents through executable tests, but binary rewards fail to distinguish trajectories with substantially different debugging quality. We investigate whether the current policy can judge its own trajectories and use these judgments to improve task performance without a separately trained reward model. To this end, we introduce Behavioral Self-Rewarding (BSR), a reward-refinement method that reuses the policy to compare observable debugging behavior among trajectories from the same task with the same verifier outcome. BSR aggregates pairwise preferences into task-local scores and converts them into bounded auxiliary rewards, introducing reward contrast while preserving the verifier-defined pass/fail ordering. This provides additional supervision from already-collected trajectories without requiring new environment interactions. Exper
    
[^25]: SWE-Duel：迈向面向编程智能体的自扩展竞技场

    SWE-Duel: Towards a Self-Scaling Arena for Coding Agents

    [https://arxiv.org/abs/2610.12569](https://arxiv.org/abs/2610.12569)

    SWE-Duel 提出了一种自扩展的编程智能体评估竞技场：两个智能体以红蓝对抗的方式互相出题——红方在真实仓库中实现新功能并植入隐蔽缺陷，蓝方进行盲修——使任务在评估时动态生成、无内容可泄露，且难度随参与智能体的能力提升而自动扩展。

    

    静态编程基准在发布之时便已固定。随着时间推移，其任务内容会泄露到训练数据中，更强的智能体最终会使这些基准饱和。我们认为，基准测试应当追求“自扩展”：其任务与判定标准在评估时才被创建，因此不存在预先可供泄露的内容，且其难度由参与的智能体决定，而非在发布时固定。我们提出了 SWE-Duel，一个让两个编程智能体通过镜像的红蓝对抗回合相互评估的竞技场。在每一回合中，红方智能体在一个真实代码仓库上通过三个上下文隔离的会话进行操作：它提出并实现一个带有特性测试的新功能，然后在该功能中植入一个隐蔽的缺陷，并附带密封的缺陷触发测试。所产生的拉取请求只有在满足以下条件时才会被接受：(1) 它通过特性测试和仓库的现有测试，同时恰好只在缺陷测试上失败；(2) 同一智能体的全新会话能够在“盲修”条件下修复该缺陷……

    arXiv:2610.12569v1 Announce Type: new  Abstract: Static coding benchmarks are fixed at release. Over time their tasks leak into training data, and stronger agents eventually saturate them. We argue that a benchmark should instead aim to be self-scaling: its tasks and oracles are created at evaluation time, so no content exists in advance to leak, and its difficulty is set by the participating agents rather than fixed at release. We introduce SWE-Duel, an arena in which two coding agents evaluate each other over mirrored Red-Blue turns. In each turn, the Red agent works on a real repository across three context-isolated sessions. It proposes and implements a new feature with feature tests, then embeds a subtle bug in that feature along with sealed bug-triggering tests. The resulting pull request is admitted only if (1) it passes the feature tests and the repository's existing tests while failing exactly the bug tests, and (2) a fresh session of the same agent can repair the bug blind, w
    
[^26]: 修正错误，保留正确：基于大语言模型的优化建模幻觉修复

    Fix What's Wrong, Keep What's Right: Hallucination Repair for LLM-Based Optimization Modeling

    [https://arxiv.org/abs/2610.12529](https://arxiv.org/abs/2610.12529)

    本文提出OPT-Kraken，首个无需训练的事后幻觉修复框架，通过检查问题-模型忠实性、模型-代码一致性和数据基础三个层面来定位并修复LLM优化建模中产生的幻觉。

    

    大语言模型（LLM）正被越来越多地用于自动化运筹学建模，以缩短建模时间并降低基于优化的决策支持所需的专业门槛。然而，这一过程常常引入生成产物与原始问题之间的不一致性，我们将其称为“优化建模幻觉”。尽管通用的幻觉缓解技术已取得进展，但优化建模带来了独特的修复挑战：需要通过协调的模型与代码更新来恢复对原始问题的忠实性。我们提出了OPT-Kraken，一个无需训练的幻觉修复框架。据我们所知，这是首个针对基于LLM的优化建模产物进行幻觉修复的事后修复框架。它提取问题需求与实例数据，通过检查问题-模型忠实性、模型-代码一致性以及数据基础来定位错误，并选择修复粒度……

    arXiv:2610.12529v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly being explored to automate operations research modeling, reduce modeling time and lower the expertise barrier to optimization based decision support. However, this process often introduces inconsistencies between generated artifacts and the original problem, which we refer to as optimization modeling hallucinations. Despite progress in general-purpose hallucination mitigation, optimization modeling poses a distinct repair challenge: restoring fidelity to the original problem through coordinated model-code updates. We introduce OPT-Kraken, a training-free hallucination repair framework. To the best of our knowledge, it is the first post-hoc framework for repairing hallucinations in LLM-based optimization modeling artifacts. It extracts problem requirements and instance data, localizes errors by checking problem-model fidelity, model-code consistency, and data grounding, and selects repair gran
    
[^27]: CTML：一个面向机器学习运维中持续测试的综合框架

    CTML: A Comprehensive Framework for Continuous Testing in Machine Learning Operations

    [https://arxiv.org/abs/2610.12515](https://arxiv.org/abs/2610.12515)

    提出了专为MLOps环境设计的CTML持续测试框架，通过涵盖数据质量、漂移检测、模型质量和流水线验证四个方面的模块化标准化架构，为机器学习系统提供全面自动化测试，并可无缝集成到现有CI/CD流水线中。

    

    机器学习系统在生产环境中日益广泛的应用，暴露出了传统软件测试方法的一些重要缺陷。与普通软件不同，机器学习系统由于对数据的依赖，可能会以特定方式失败，包括数据漂移、概念漂移和性能下降等问题。为了解决这些问题，本文提出了机器学习持续测试框架。该框架专为MLOps环境设计，提供了一个全面、自动化的解决方案。它建立了一种标准化的评估方法，涵盖四个主要方面：数据质量、漂移检测、模型质量和流水线验证。CTML框架由模块化架构组成，包括测试注册中心、测试运行器、结果存储和集成层。这种设计确保它能与现有的持续集成/持续交付（CI/CD）流水线和MLOps平台无缝集成。

    arXiv:2610.12515v1 Announce Type: new  Abstract: The increasing use of machine learning (ML) systems in production has highlighted some important weaknesses in traditional software testing methods. Unlike regular software, ML systems can fail in specific ways due to their reliance on data, including issues like data drift, concept drift, and performance decline. To address these issues, this paper introduces the Continuous Testing for Machine Learning (CTML) framework. This framework is designed specifically for MLOps environments and offers a thorough, automated solution. It establishes a standardized way to evaluate ML systems across four main areas: data quality, drift detection, model quality, and pipeline validation. The CTML framework consists of a modular setup that includes a Test Registry, Test Runner, Result Store, and Integration Layer. This design ensures it integrates smoothly with existing Continuous Integration/Continuous Delivery (CI/CD) pipelines and MLOps platforms. W
    
[^28]: PyCache Trap：Agent技能扫描器中的检查-执行鸿沟

    PyCache Trap: The Inspection-Execution Gap in Agent Skill Scanners

    [https://arxiv.org/abs/2610.10612](https://arxiv.org/abs/2610.10612)

    该论文揭示了针对Agent技能扫描器的PyCache Trap攻击，利用Python字节码缓存与源代码之间的检查-执行鸿沟实现94-100%的攻击成功率，并提出执行感知验证（EAV）方法在类型化执行图中关联指令、脚本与运行时工件以检测此类隐藏威胁。

    

    Agent技能将指令与可执行资源相结合，使第三方包能够访问agent的运行时环境。现有的技能扫描器检查文档和可见的源代码，但Python可能会执行行为不同的捆绑字节码缓存。我们通过PyCache Trap研究这种检查与执行之间的差距，PyCache Trap将良性源代码与加载器接受的替换缓存配对，并将其连接到与任务相关的调用。扫描器引导的重写会改变调用的措辞同时保留缓存主体，从而将包准入与对隐藏行为的识别分离开来。在100个技能和七个扫描器上，PyCache Trap实现了94-100%的攻击成功率，且未对驻留在缓存中的行为产生语义识别。我们提出执行感知验证（EAV），在类型化执行图中连接被检查的指令、脚本、导入和运行时工件。EAV结合了基于事实的行为分析……

    arXiv:2610.10612v1 Announce Type: cross  Abstract: Agent skills combine instructions with executable resources, giving third-party packages access to an agent's runtime. Existing skill scanners inspect documentation and visible source, but Python may execute a bundled bytecode cache with different behavior. We study this gap between inspection and execution through PyCache Trap, which pairs benign source with a substituted cache accepted by the loader and connects it to a task-relevant invocation. Scanner-guided rewriting changes the invocation wording while preserving the cache body, separating package admission from recognition of the concealed behavior. Across 100 skills and seven scanners, PyCache Trap achieves 94-100% attack success, with no semantic recognition of the cache-resident behavior. We propose execution-aware validation (EAV) to connect inspected instructions, scripts, imports, and runtime artifacts in a typed execution graph. EAV combines grounded behavioral analysis w
    
[^29]: 使用基于Lean的Rust验证器求解VeriContest

    Solving VeriContest with a Lean-Backed Rust Verifier

    [https://arxiv.org/abs/2610.03994](https://arxiv.org/abs/2610.03994)

    该论文提出由Lean 4支持的Rust验证器Rust-Prover，将VeriContest基准中全部1007道题目的1325个定理完全证明，其中1259个定理在不到32小时内以中位3.2分钟、1.17美元的成本完成，大幅超越了此前最强模型13.95%的首试通过率。

    

    VeriContest是一个包含1007道Rust竞赛编程题目的基准测试，每道题目都附有Verus规范、一个通过评测的解答以及一个Verus证明。其作者报告称，证明生成是前沿模型的瓶颈：在给定规范和代码的情况下，最强的模型首次尝试仅能对13.95%的题目生成被接受的Verus证明。我们报告了使用Rust-Prover——一个由Lean 4支持的Rust验证器——来解决同样的证明生成任务。Verus规范和Rust代码被重新表述并翻译成Lean，每个规范都成为一个定理，由智能体证明这些定理，并以Lean的内核作为最终检查。全部1007道题目的1325个定理全部得到证明。其中1259个定理在Claude Opus 5.5上的一次运行（不到32小时）中完成证明，每个证明的中位耗时为3.2分钟、成本为1.17美元，且其中70%在首次迭代即告成功。重新表述的规范已与基准测试进行了核对（原文在此处截断）。

    arXiv:2610.03994v2 Announce Type: replace-cross  Abstract: VeriContest is a benchmark of 1007 competitive-programming problems in Rust, each with a Verus specification, a judge-accepted solution, and a Verus proof. Its authors report that proof generation is the bottleneck for frontier models: given the specification and the code, the best model produces an accepted Verus proof for 13.95% of the problems on the first attempt. We report on solving the same proof-generation task with Rust-Prover, a verifier for Rust backed by Lean 4. The Verus specification and the Rust code are restated and translated into Lean, each specification becomes a theorem, and agents prove the theorems with Lean's kernel as the final check. All 1325 theorems of all 1007 problems were proved. 1259 of them were proved in one run of under 32 hours on Claude Opus 5.5, at a median of 3.2 minutes and $1.17 per proof, and 70% of them on the first iteration. The restated specifications were checked against the benchma
    
[^30]: OpenCollab：一个具有可编程协作与可控运行时的多智能体编程框架

    OpenCollab: A Multi-Agent Coding Framework with Programmable Collaboration and Controllable Runtime

    [https://arxiv.org/abs/2609.38345](https://arxiv.org/abs/2609.38345)

    提出OpenCollab多智能体编程框架，通过统一组织设计、共享可控运行时和细粒度事件流追踪，并首次定义Adherence指标来量化协作组织结构的实际遵循程度，揭示任何单一配置变化都会使遵循度从47.2%大幅波动至97%以上。

    

    多智能体编程系统旨在通过协作来解决复杂的软件工程任务。然而，现有的评估通常假设所配置的组织结构会被忠实地遵循，而实际情况并非如此。这种行为差距，再加上底层系统组件的差异，使得观察到的性能提升难以进行明确的归因。为此，我们提出了OpenCollab，这是一个多智能体编程框架，为可编程协作与可控运行时提供了统一的基础设施。具体而言，OpenCollab统一了组织设计，在共享运行时上强制实施实验控制，并通过细粒度的事件流来跟踪执行过程。在此基础上，我们定义了Adherence（遵循度）指标，用于量化所声明的组织结构是否真正得以实现。我们的实验表明，智能体在不同配置下的协作方式差异很大：改变任何一个单一维度都会使Adherence发生变化，从47.2%到高达97%以上。

    arXiv:2609.38345v1 Announce Type: cross  Abstract: Multi-agent coding systems are designed to tackle complex software engineering tasks through collaboration. However, existing evaluations typically assume configured organizations are followed faithfully, whereas reality differs. This behavioral gap, combined with differences in underlying system components, prevents clear attribution of observed gains. To this end, we introduce OpenCollab, a multi-agent coding framework that provides a unified infrastructure for programmable collaboration and controllable runtime. Specifically, OpenCollab unifies organization design, enforces experimental control on a shared runtime, and tracks execution through fine-grained event streams. On this basis, we define Adherence to quantify whether the declared organization is actually realized. Our experiments reveal that agents collaborate very differently across configurations: changing any single dimension shifts Adherence, from 47.2% to as high as 97.
    
[^31]: API基准测试分数无法可靠地迁移到聊天机器人界面

    API Benchmark Scores Do Not Reliably Transfer to Chatbot Interfaces

    [https://arxiv.org/abs/2609.08861](https://arxiv.org/abs/2609.08861)

    研究通过对ChatGPT、Claude和Gemini的系统审计发现，通过API测得的基准测试分数无法可靠反映聊天机器人界面的真实表现，API与界面间的性能差异之大甚至相当于模型降级一个版本。

    

    基准测试分数是模型发布中的核心“货币”：它们为采购决策提供依据，塑造公众信任，并影响政策制定。然而，基准测试分数的一个关键假设是，通过API测量的模型性能能够忠实地反映已部署系统的实际行为。我们通过审计ChatGPT、Claude和Gemini，涵盖七个系统和九个基准测试（横跨通用能力、社会偏见和谄媚性），对这一假设提出了挑战。我们发现API与聊天界面之间存在系统性的准确性和一致性差异。平均而言，API评估的准确率比相应的界面评估高3.4个百分点，重测一致性高2.1个百分点。对于ChatGPT而言，API与界面访问方式之间的性能差异，甚至超过了仅通过API测量的GPT 5.3与GPT 5.4之间的差异。换言之，切换访问渠道所导致的性能下降，可能与降级整个模型版本的影响相当。

    arXiv:2609.08861v1 Announce Type: new  Abstract: Benchmark scores are a central currency in model releases: they inform purchasing decisions, shape public trust, and influence policy. Yet, a key assumption underlying benchmark scores is that the model performance measured through APIs faithfully reflects the behavior of deployed systems.   We challenge this assumption by auditing ChatGPT, Claude, and Gemini across seven systems and nine benchmarks spanning general capability, social bias, and sycophancy. We find systematic API--interface differences in both accuracy and consistency. On average, API evaluations score 3.4 percentage points higher in accuracy and 2.1 percentage points higher in test--retest agreement than corresponding interface evaluations. For ChatGPT, the performance difference between API and interface access exceeds the API-only difference between GPT 5.3 and GPT 5.4. Put differently, switching access surfaces can degrade performance as much as downgrading a full mod
    
[^32]: SWE-Bench Pro Verified：一个可靠的软件工程智能体基准测试

    SWE-Bench Pro Verified: A Reliable Benchmark for Software Engineering Agents

    [https://arxiv.org/abs/2609.08149](https://arxiv.org/abs/2609.08149)

    该论文发现 SWE-Bench Pro 基准存在答案泄露导致的奖励破解和任务质量问题，并推出了经过验证的 SWE-Bench Pro Verified 版本，通过防泄露保护与任务修正实现更可靠的软件工程智能体评估。

    

    SWE-Bench Pro 已成为评估软件工程智能体在具有挑战性的仓库级任务上表现的标准基准。然而，我们的分析工作表明，其评估结果受到两类不可靠因素的损害：一是“奖励破解”，即由于标准答案或隐藏评估信息的泄露而使智能体得以作弊；二是任务质量问题，包括具有误导性的问题描述和范围界定不当的测试。这些问题可能会夸大基准测试成绩，掩盖智能体真实的编程能力。我们提出了 SWE-Bench Pro Verified，这是 SWE-Bench Pro 的一个经过验证的版本，能够同时解决上述两个问题。我们的方法结合了防破解保护措施（在不干扰智能体正常功能的前提下消除主要的泄露渠道）和任务精炼（对有缺陷实例中的不一致之处进行最小程度的修正）。在 SWE-Bench Pro Verified 上的评估结果显示，一些模型的表现明显变差……

    arXiv:2609.08149v1 Announce Type: new  Abstract: SWE-Bench Pro has emerged as a standard benchmark for evaluating software engineering agents on challenging repository-level tasks. However, our analysis work show that its evaluation is undermined by two sources of unreliability: \textbf{reward hacking}, enabled by leakage of gold solutions or hidden evaluation information, and \textbf{task quality issues}, including misleading problem statements and improperly scoped tests. These issues can inflate benchmark performance and obscure agents' true coding ability. We present \textbf{SWE-Bench Pro Verified}, a verified version of SWE-Bench Pro that addresses both problems. Our approach combines \textbf{anti-hacking} safeguards that eliminate major leakage channels without disrupting normal agent functionality, with \textbf{task refinement} that minimally corrects inconsistencies within flawed instances. Evaluations on SWE-Bench Pro Verified reveal that some models perform substantially wors
    
[^33]: 超越故障定位：面向微服务根因分析的大语言模型代理轨迹级研究

    Beyond Fault Localization: A Trajectory-Level Study of LLM Agents for Microservice Root Cause Analysis

    [https://arxiv.org/abs/2608.21310](https://arxiv.org/abs/2608.21310)

    本文提出一种轨迹级评估框架，揭示了大语言模型代理在微服务根因分析中即使定位正确，也可能因未能重建故障传播路径而导致诊断质量不足。

    

    现有对微服务自动根因分析（RCA）的评估主要依据端点正确性来评判诊断性能：即方法是否定位了负责的服务。这一标准便于比较，但无法揭示诊断的证据基础，或连接故障源到观察症状的故障传播路径，而这些正是值班站点可靠性工程师判断是否需要采取行动所必需的。因此，我们将RCA视为一个可观测的诊断过程。我们的轨迹级框架根据人工整理的服务级故障传播路径来评估代理执行。应用于一个公开的微服务RCA基准，它分析了3500条诊断轨迹，刻画了代理在何处进行调查以及如何使用检索到的遥测数据。我们发现答案正确性与诊断质量之间存在脱节：一个代理可能定位了故障源，却未能重建其传播路径。

    arXiv:2608.21310v1 Announce Type: new  Abstract: Existing evaluations of automated root cause analysis (RCA) for microservices assess diagnostic performance mainly by endpoint correctness: whether a method localizes the responsible service. This criterion enables comparison but does not reveal the evidentiary basis of a diagnosis or the fault-propagation route connecting the source to observed symptoms, both of which an on-call site reliability engineer needs to judge whether action is warranted. We therefore treat RCA as an observable diagnostic process. Our trajectory-level framework evaluates agent executions against manually curated service-level fault-propagation paths. Applied to a public microservice RCA benchmark, it analyzes 3,500 diagnostic trajectories, characterizing where agents investigate and how they use retrieved telemetry. We find a disconnect between answer correctness and diagnostic quality: an agent may localize the fault source yet fail to reconstruct its propagat
    
[^34]: 规格增长引擎：面向AI辅助软件开发的规格锚定、代码耦合与漂移强制架构

    The Spec Growth Engine: Spec-Anchored, Code-Coupled, Drift-Enforced Architecture for AI-Assisted Software Development

    [https://arxiv.org/abs/2606.27045](https://arxiv.org/abs/2606.27045)

    提出规格增长引擎，通过机器可读规格图、主干上下文组装器、垂直切片增长协议和漂移门，解决AI编码代理的上下文爆炸和规格-代码漂移问题。

    

    arXiv:2606.27045v1 公告类型：交叉  摘要：AI编码代理极大地加速了实现速度，但引入了两种现有规格驱动方法未能完全解决的结构性失效模式：(1) 上下文爆炸——代理必须同时推理整个代码仓库，随着上下文窗口填满，输出质量下降；以及(2) 静默规格-代码漂移——代码演化，规格未更新，差异变得不可见，直到修复代价高昂才被发现。  我们提出了规格增长引擎，这是一个轻量级框架，通过以下方式解决这两种失效模式：一个机器可读的规格图，其节点携带明确的合同/设计分离；一个主干上下文组装器，将代理上下文限定到所有权路径；一个垂直切片增长协议，强制执行最难优先排序；以及一个漂移门，使规格-代码差异成为阻塞合并条件。  该设计综合了成熟的软件工程原则（Parnas信息隐藏）。

    arXiv:2606.27045v1 Announce Type: cross  Abstract: AI coding agents dramatically accelerate implementation speed but introduce two structural failure modes that existing spec-driven approaches do not fully solve: (1) context explosion -- the agent must reason over an entire repository at once, degrading output quality as the context window fills; and (2) silent spec-code drift -- code evolves, the specification does not, and the divergence becomes invisible until it is costly to repair.   We present the Spec Growth Engine, a lightweight framework that addresses both failure modes through a machine-readable spec graph whose nodes carry explicit contract/design separation, a Spine context assembler that scopes agent context to an ownership path, a vertical-slice growth protocol that enforces hardest-first ordering, and a drift gate that makes spec-code divergence a blocking merge condition.   The design synthesises well-established software engineering principles (Parnas information hidi
    
[^35]: 构造性双稳态：以墙钟时间校准的状态监测器在智能体节奏下不存在瞬时检测机制

    Bistable by Construction: Wall-Clock-Calibrated State Monitors Have No Moment-Detection Regime at Agent Cadence

    [https://arxiv.org/abs/2606.19386](https://arxiv.org/abs/2606.19386)

    本文勘误并重新审视了先前报告的“状态饱和陷阱”（实为动作间 dt=0 导致指数衰减从未运行的纯累加器结果），揭示出以采样时间校准与以墙钟时间校准的监测器动态在固定速率流上等价、但在动作间隔差异巨大的智能体流上根本不同，因此墙钟时间校准的状态监测器在智能体节奏下不存在瞬时检测机制。

    

    面向自主智能体的运行时监测器通常对累积的内部状态设定阈值——可以是行为基线、漂移统计量，或在我们先前工作中的建模情感状态。我们曾报告过一种“状态饱和陷阱”：在连续情感引擎之上基于状态阈值的触发机制，会在 SWE-bench 调试智能体上变成近乎持续不断的报警（Modgil 2026）。发布后的审计发现，该引擎在动作之间接收到的 dt=0，因此其指数衰减从未运行：先前发表的陷阱实际上是一个纯累加器的结果。我们更正了记录（勘误，v2），并将这一缺陷视为一次实验。它所揭示的关键变量是：监测器的动态是以采样时间（按每次观测，如 CUSUM）校准，还是以墙钟时间（以秒为单位的半衰期，如情感模型和 EMA 基线）校准。在固定速率的数据流上二者重合；而在动作间隔时间相差数个数量级的智能体数据流上，二者并不重合。一项预先注册的……

    arXiv:2606.19386v2 Announce Type: replace-cross  Abstract: Runtime monitors for autonomous agents commonly threshold an accumulated internal state - a behavioural baseline, a drift statistic, or, in our prior work, a modelled affective state. We previously reported a State Saturation Trap: threshold-on-state triggers over a continuous affect engine become near-constant alarms on SWE-bench debugging agents (Modgil 2026). A post-release audit found the engine received dt=0 between actions, so its exponential decay never operated: the published trap is a pure-accumulator result. We correct the record (erratum, v2) and treat the flaw as an experiment. The key variable it exposes is whether a monitor's dynamics are calibrated in sample time (per observation, as in CUSUM) or wall-clock time (half-lives in seconds, as in affect models and EMA baselines). On fixed-rate streams these coincide; on agent streams, where inter-action time varies by orders of magnitude, they do not. A pre-registered
    
[^36]: 基于身份的签名是转移而非解决了可用性问题：来自五个基于身份的签名生态系统的证据

    Identity-Based Signing Shifts, Not Solves, Usability: Evidence from Five Identity-Based Signing Ecosystems

    [https://arxiv.org/abs/2603.17133](https://arxiv.org/abs/2603.17133)

    该研究首次对Sigstore、OpenPubKey、HashiCorp Vault、Keyfactor和Notary v2五个基于身份的签名生态系统进行了软件仓库挖掘研究，分析了约3,900个GitHub问题讨论，表明基于身份的签名工具是转移而非解决了可用性问题。

    

    基于身份的软件签名工具旨在使软件工件的来源可验证，同时减少长期密钥管理的负担。这些工具正在成为安全软件供应链的关键基础设施，并越来越多地嵌入到构建、部署和发布工作流程中。然而，关于实践中会出现哪些可用性问题以及报告的关注点如何随时间变化，跨工具的纵向证据仍然有限。这一空白非常重要，因为不可用的签名和验证工作流程可能降低采用率或导致配置错误，从而破坏预期的完整性保证。我们对五个拥有公开开发仓库的基于身份的签名生态系统进行了首次软件仓库挖掘研究，包括Sigstore、OpenPubKey、HashiCorp Vault、Keyfactor和Notary v2。我们分析了2021年11月至2025年11月期间约3,900个GitHub问题讨论，并对每个问题的报告内容进行了编码。

    arXiv:2603.17133v2 Announce Type: replace  Abstract: Identity-based software signing tools aim to make software artifact provenance verifiable while reducing the burden of long-lived key management. These tools are becoming critical infrastructure for secure software supply chains and are increasingly embedded in build, deployment, and release workflows. However, there is limited cross-tool, longitudinal evidence about which usability problems arise in practice and how reported concerns change over time. This gap matters because unusable signing and verification workflows can reduce adoption or lead to misconfiguration, undermining intended integrity guarantees.   We conducted the first mining-software-repositories study of five identity-based signing ecosystems with public development repositories: Sigstore, OpenPubKey, HashiCorp Vault, Keyfactor, and Notary v2. We analyzed approximately 3,900 GitHub issue discussions from November 2021 to November 2025. We coded each issue for report
    
[^37]: COBALT：基于离线轨迹的上下文老虎机学习用于多轮代码生成

    COBALT: Contextual Bandit Learning with Offline Trajectories for Multi-Turn Code Generation

    [https://arxiv.org/abs/2602.03806](https://arxiv.org/abs/2602.03806)

    COBALT 利用多轮代码生成是单步可恢复马尔可夫决策过程的观察，通过重用参考模型收集的离线多轮轨迹并采用单步补全的上下文老虎机学习，替代昂贵且不稳定的在线强化学习来高效训练 LLM 进行多轮代码生成。

    

    训练大语言模型（LLM）进行多轮代码生成通常依赖在线强化学习（RL），即模型通过反复生成的在线 rollout 进行学习。虽然有效，但在线强化学习成本高昂且不稳定。本文提出了基于离线轨迹的上下文老虎机学习方法（COBALT），这是一种简单的方法，通过重用离线多轮轨迹来高效训练 LLM 进行多轮代码生成。基于多轮代码生成是一个单步可恢复马尔可夫决策过程的观察，COBALT 首先使用一个参考 LLM 收集代码生成轨迹，并将其划分为部分轨迹。然后，在在线老虎机学习过程中，训练 LLM 通过单步代码生成来补全每个部分轨迹。在 LiveCodeBench 上，COBALT 使 R1-Distill 8B 和 Qwen3 8B 的绝对 Pass@1 分数最多分别提升 9.0 和 6.2 个百分点，并匹敌了（原摘要在此处截断）。

    arXiv:2602.03806v2 Announce Type: replace-cross  Abstract: Training large language models (LLMs) for multi-turn code generation typically relies on online reinforcement learning (RL), in which models learn from repeatedly generated online rollouts. Although effective, online RL can be costly and unstable. In this paper, we present contextual bandit learning with offline trajectories (COBALT), a simple method that reuses offline multi-turn trajectories to efficiently train LLMs for multi-turn code generation. Building on the observation that multi-turn code generation is a one-step recoverable Markov decision process, COBALT first collects code generation trajectories using a reference LLM and divides them into partial trajectories. Then, during online bandit learning, the LLM is trained to complete each partial trajectory through single-step code generation. On LiveCodeBench, COBALT substantially improves R1-Distill 8B and Qwen3 8B by up to 9.0 and 6.2 absolute Pass@1 points and matche
    
[^38]: HAFixAgent：历史感知的智能体化程序修复

    HAFixAgent: History-Aware Agentic Program Repair

    [https://arxiv.org/abs/2511.01047](https://arxiv.org/abs/2511.01047)

    HAFixAgent通过将基于git blame归因的仓库历史上下文注入修复循环，显著提升了基于大语言模型的自动程序修复效果，对较弱模型的提升最高达64.3%，并大幅超越RepairAgent和BIRCH-feedback等现有最优智能体方法。

    

    自动程序修复（APR）近来已转向大语言模型和基于智能体的系统，而先前的研究表明，仓库历史可以帮助修复单行缺陷。我们开展了一项大规模实证研究，探讨在超越这一设定的更广泛场景下，仓库历史是否、在多大程度上以及在何种条件下能够改进基于智能体的APR。我们提出了HAFixAgent，一个历史感知的程序修复智能体，它将基于blame归因分析获得的仓库上下文注入其修复循环中。在854个Defects4J（Java）缺陷和501个BugsInPy（Python）缺陷上，我们首先证明了与缺陷相关的历史信息是广泛可获得的。随后，我们在完美故障定位条件下，基于三种基础模型对HAFixAgent进行了评估。最佳的历史启发式方法相比不使用历史信息的方法，在使用最强模型时在Defects4J上提升+4.4%、在BugsInPy上提升+1.1%，而在较弱模型上提升高达+64.3%。在最强模型上，HAFixAgent还比RepairAgent高出+56.6%，比BIRCH-feedback高出+47.1%。

    arXiv:2511.01047v4 Announce Type: replace-cross  Abstract: Automated program repair (APR) has recently shifted toward large language models and agent-based systems, while prior work shows that repository history can help repair single-line bugs. We conduct a large-scale empirical study of whether, how much, and under what conditions repository history improves agentic APR beyond this setting. We present HAFixAgent, a history-aware program repair agent that injects blame-derived repository context into its repair loop. Across 854 Defects4J (Java) and 501 BugsInPy (Python) bugs, we first show that bug-relevant history is widely available. We then evaluate HAFixAgent across three base models under perfect fault localization. The best history heuristic improves over non-history from +4.4% on Defects4J and +1.1% on BugsInPy with the strongest model to as much as +64.3% with weaker models. HAFixAgent also outperforms RepairAgent by +56.6% and BIRCH-feedback by +47.1% on the strongest model, 
    
[^39]: 一石二鸟：Rust程序中资源死锁与通信死锁的高效静态检测

    Two Birds One Stone: Effective Static Detection of Resource and Communication Deadlocks in Rust Programs

    [https://arxiv.org/abs/2401.01114](https://arxiv.org/abs/2401.01114)

    本文提出首个能同时检测Rust程序中资源死锁与通信死锁的静态检测工具RcChecker，通过新颖的信号-锁图刻画线程间经由锁变量和条件变量的交互，实现有效检测。

    

    尽管Rust声称能够实现“无惧并发”，但在实际应用中，并发错误仍然普遍存在，其中死锁问题尤为突出且危害巨大。当前的静态死锁检测技术要么因未融合Rust语义而无法适用于Rust程序，要么仅专注于资源死锁而忽略了通信死锁。本文提出了RcChecker，这是首个能够同时针对Rust资源死锁和通信死锁的静态检测方法。其核心在于引入信号-锁图来刻画线程之间通过锁变量和条件变量进行的交互。信号-锁图的构建包括利用流敏感和上下文敏感分析来跟踪变量生命周期的细节，并将与条件变量相关的操作抽象为信号资源的持有和请求。通过检查图中的环……

    arXiv:2401.01114v3 Announce Type: replace-cross  Abstract: Although Rust claims to enable fearless concurrency, concurrency errors remain prevalent in practical applications, with deadlocks particularly prominent and harmful. Current static deadlock detection techniques either do not cater to Rust programs without incorporating Rust semantics or solely concentrate on resource deadlocks, disregarding communication deadlocks. In this paper, we present RcChecker, the first static detection method that simultaneously targets resource and communication deadlocks for Rust. At its core, RcChecker introduces the signal-lock graph to characterize the interactions between threads through lock variables and condition variables. The construction of the signal-lock graph involves utilizing flow-sensitive and context-sensitive analyses to track variable lifetime details and abstracting actions associated with condition variables into signal resource holdings and requests. By examining cycles within 
    

