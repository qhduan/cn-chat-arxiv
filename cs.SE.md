# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Free to Look, Free to Reason: Simple LLM Harnesses Outdiagnose Hand-Crafted Root Cause Analysis Agents](https://arxiv.org/abs/2610.35767) | 该论文发现，不预设调查流程、让模型自由观察和推理的简单LLM框架，在微服务故障根因分析上比手工设计的LLM智能体和非LLM统计因果方法诊断得更准确。 |
| [^2] | [FLOWSTEP: A Framework for Learning to Predict GitHub Actions Workflow STEPs](https://arxiv.org/abs/2610.35650) | FLOWSTEP 通过明确区分可复用 Action 与项目特定 Script 两类工作流步骤，并结合大语言模型与对比学习，显著提升了 GitHub Actions 工作流步骤自动补全的准确性。 |
| [^3] | [GEAR: Feature-Based Configuration and Code Generation for Agentic Systems](https://arxiv.org/abs/2610.35592) | GEAR是一个开源工具，通过基于特征的统一规范来配置多智能体系统，并能自动生成适用于十个不同智能体框架的代码实现，解决了开发者需要学习不同框架API的问题。 |
| [^4] | [Making the Telemetry Pipeline Measurable: An Edge IoT Testbed and Dataset for End-to-End Latency, Provenance, and Energy-Aware Benchmarking](https://arxiv.org/abs/2610.35591) | 提出了一个以测量为先的边缘物联网测试床与数据集，通过基于角色的事件驱动架构和嵌入阶段级时间戳的遥测消息封装，首次将端到端延迟、溯源和能耗作为可测量的一等遥测维度进行基准测试。 |
| [^5] | [The Agent in the Middle: How AI-Mediated Access Reshapes Web Vulnerability Exploitation](https://arxiv.org/abs/2610.35541) | 提出可执行基准测试VELA，系统研究了AI智能体中介访问对Web漏洞利用的影响，发现面向智能体的接口设计决定了漏洞可利用性，参考设计下91%的漏洞仍然可被利用。 |
| [^6] | [JANUS: Synthesizing Cross-Layer Defenses Against Web Deserialization Attacks](https://arxiv.org/abs/2610.35532) | JANUS提出了一种红蓝对抗框架，通过黑盒攻击者迭代探测攻击变体与防御者进行反事实分析并细化策略，实现针对Web反序列化攻击的跨层防御自动化合成。 |
| [^7] | [Ranking Function Generation with LLMs for Termination](https://arxiv.org/abs/2610.35523) | 该论文提出RAINFALL神经符号框架，通过将排序函数合成转化为带槽位约束的代码补全任务，并结合符号验证器的反馈迭代改进，利用大语言模型自动生成可验证的排序函数以证明循环终止性。 |
| [^8] | [Governing Enterprise AI Traffic at Scale: Design and Operation of a Policy-Driven, Multi-Vendor LLM Gateway on Envoy](https://arxiv.org/abs/2610.35509) | PayPal的Cosmos LLM网关基于Envoy，通过将I/O密集型控制决策（进程外服务）与数据面转换（代理内Rust WASM过滤器）按延迟特征分离，实现了对多厂商企业LLM流量的统一策略治理、成本控制与无缝流式转发。 |
| [^9] | [The Hidden Costs of Forgetting: A Study of Trajectory Deletion in Software Engineering Agents](https://arxiv.org/abs/2610.35496) | 通过在SWE-bench Verified上对9,400次运行的大规模实证研究发现，软件工程智能体可以容忍删除大量旧轨迹内容（甚至100%的推理文本）而不影响整体有效性，但删除必须延迟执行，立即删除会显著降低智能体表现。 |
| [^10] | [FeatureLoop: Benchmarking Coding Agents under Evolving Requirements](https://arxiv.org/abs/2610.35488) | 该论文提出FeatureLoop基准，通过演化反馈与完成反馈模拟真实开发中需求不断演进的过程，对61个任务上的12个前沿模型进行评估，发现最强模型仅完成72.1%的任务，且绝大多数失败阶段难以在两次纠正性修订内恢复。 |
| [^11] | [Understanding Python Linters through a Comprehensive and Systematic Empirical Study](https://arxiv.org/abs/2610.35485) | 本文首次对Ruff、Pylint、Flake8、Mypy和Black五个主流Python代码检查工具开展了大规模系统实证研究，从检查规则、速度、配置方法和工具偏好四个维度揭示了它们的共性、差异与特点。 |
| [^12] | [Making Small Code LLMs More Efficient for Secure Code Review with Neuro-Symbolic Fine-Tuning](https://arxiv.org/abs/2610.35477) | 该论文提出secSemScan系统，通过神经符号微调将静态分析器与本地部署的小型代码模型结合，使其能生成可验证的、模式约束的安全审查解释，并构建了包含1,993个C/C++函数、覆盖22类CWE的OGSM数据集。 |
| [^13] | [Then there is prompt injection, a malicious email, ...: The Wild Wide West of OpenClaw Agentic System Security](https://arxiv.org/abs/2610.35451) | 本文通过对 Reddit 讨论与安全相关 GitHub issue 的大规模实证分析，系统刻画了 OpenClaw 智能体框架的安全现状，识别出 59 种弱点类型下的 521 个 CWE 实例，揭示了提示注入等风险在缺乏明确信任边界时对 LLM 智能体系统的普遍威胁。 |
| [^14] | [Toward Evidence-Based Agentic Software Engineering](https://arxiv.org/abs/2610.35348) | 本研究通过标注146篇论文中的325个实验，揭示智能体软件工程研究普遍缺乏对运行框架版本及配置的充分报告、且很少将运行框架影响与模型影响分离，并据此提出了参考运行框架的需求。 |
| [^15] | [RepoTrace: Improving Requirement Tracing in Multi-Round Agent Interactions](https://arxiv.org/abs/2610.35333) | 提出 RepoTrace，通过将仓库结构与需求语义结合构建需求-代码对应关系并注入智能体上下文，解决了代码智能体在多轮交互中因需求焦点错位导致的性能下降问题。 |
| [^16] | [Testing with Coding Agents: A Large-Scale Comparison of Agent-Supported and Human-Authored Pull Requests](https://arxiv.org/abs/2610.35297) | 该研究通过对391个Java仓库中114,134个拉取请求的大规模实证分析，首次系统比较了编码智能体支持的与人工编写的测试变更在频率、范围、类型、结构、坏味道和覆盖率五个维度上的差异，为理解智能体时代测试工作的分工及其对长期可维护性的影响提供了重要依据。 |
| [^17] | [Chart the Invisible: Interactive Perception for Hidden-Target Exploration in GUI Agents](https://arxiv.org/abs/2610.35237) | 该论文提出了首个用于可滚动区域感知的多模态模型，使GUI智能体能够在动作选择前探索隐藏区域，从而解决智能体在面对视口外隐藏目标时难以决定何时、何地及如何滑动的关键难题。 |
| [^18] | [Behavioral Regression Detection for Pull Requests via Structured Oracles](https://arxiv.org/abs/2610.35223) | ORION通过从PR意图合成结构化行为预言并引导可执行测试的生成与差分执行，实现了拉取请求级别的行为回归检测，同时发布了首个Python PR回归基准RegBench-Py。 |
| [^19] | [V$^3$AST-Bench: Verifiable, Valid, and Versatile Safety Evaluation for Coding Agents](https://arxiv.org/abs/2610.35188) | 提出V³AST-Bench基准，通过可执行检查验证代码智能体造成的实际危害与任务结果，利用成对孪生任务区分安全性与能力短板，并以212个跨18类风险的任务实现对智能体安全的全面评估。 |
| [^20] | [Priority-Aware Scheduler for Serverless Functions Provision and Load Balancing](https://arxiv.org/abs/2610.35073) | 本文提出SPES-Ext，一种面向内存受限分布式工作节点的基于规则的优先级感知调度器，通过利用无服务器系统集中的调用模式进行行为预测和函数分类，优化冷启动、函数驱逐和负载不均衡问题，在启动延迟与资源效率之间实现权衡。 |
| [^21] | [Toward Methods and Tools for Critical Examination of Disparate Systems for Discovery of Hidden Dependencies in Engineering of Autonomous and Semi-Autonomous Systems](https://arxiv.org/abs/2610.34974) | 本文提出一种通过对相关环境中异构系统视频进行主动对抗性批判分析的方法论和AI工具，用于发现隐藏的依赖关系和环境假设，帮助软件工程师在陌生领域开发自主/半自主系统时获取领域知识并提升系统质量与工程实践。 |
| [^22] | [Rivet: Utility-Preserving Context Control for Repository-Level Coding Agents](https://arxiv.org/abs/2610.34967) | 提出 Rivet 框架，通过直接保留目标位置、辅助定义、调用方用法和配置约束等仓库证据，解决现有上下文选择方法中“看似相关却缺失关键证据”的上下文效用稀释问题，从而提升仓库级编码智能体的代码生成效果。 |
| [^23] | [Decision Model Factory: Building Specialized Sovereign Decision Models](https://arxiv.org/abs/2610.34956) | 提出决策模型工厂，可按需将骨干模型转化为专业化、主权可控的决策模型，配合专用运行器在保持决策分数基本不变的情况下将服务速度提升一倍以上。 |
| [^24] | [Reasoning Patterns for Efficient AI Coding: An Empirical Study](https://arxiv.org/abs/2610.34943) | 该研究通过将四个编码任务的推理轨迹转化为结构化动作序列进行实证分析，识别出代码生成、执行推理、调试和翻译中的四种通用推理模式及其正负面模式，为系统性刻画和推理时控制大型推理模型提供了依据。 |
| [^25] | [How Do Coding-Agent Harnesses Self-Evolve? An Empirical Study of Retrospective Adaptation](https://arxiv.org/abs/2610.34938) | 该论文首次实证研究了编码智能体通过回顾已完成任务会话来提出并实施运行框架改进的“回溯性适应”机制，发现尽管大多数建议源于具体会话事件，但在七项有效评估中仅有两项更改真正消除了后续任务中的目标问题，表明自我进化的实际收益仍然有限。 |
| [^26] | [What Will Replace Pull Requests? Rethinking Software Collaboration for AI Agents](https://arxiv.org/abs/2610.34933) | 该研究通过分析31个组织或个人的50份灰色文献，系统调查了AI编码智能体时代从业者提出的拉取请求替代机制，发现大多数方案仍以某种形式保留拉取请求，而非完全取代它。 |
| [^27] | [SemWeaver: Refining LLM-Generated Static Vulnerability Checkers with Analyzer-Internal Evidence](https://arxiv.org/abs/2610.34928) | SemWeaver 提出了一个插件式精炼框架，通过提取静态分析器内部已有的证据（如控制流图、符号值、内存区域和检查器回调），并将其与源代码上下文和验证反馈组装成带来源标注的结构化信息，迭代修正 LLM 生成的静态漏洞检查器，使其从仅匹配补丁的表面修改提升为捕捉更深层的语义条件。 |
| [^28] | [AIPC: Evidence-Guided Agent Workflows for PyTorch Project Deployment](https://arxiv.org/abs/2610.34890) | 提出AIPC方法，通过将结构化技能与工具链、运行时和性能分析证据相结合，引导智能体进行跨层诊断与修复，实现PyTorch推理项目在专用加速器（尤其是端侧NPU）上的自动化部署与移植。 |
| [^29] | [Ethics by Configuration: AI Ethics Principles in Coding Agent Instructions](https://arxiv.org/abs/2610.34847) | 该研究通过分析8,893个GitHub仓库中的40,079个配置文件，首次系统揭示了开发者在配置AI编码智能体时会自发写入伦理相关声明（见于39.4%的文件），且这些声明集中于问责、透明度、隐私保护和可靠性四大原则。 |
| [^30] | [Large Language Models for Quantum Code Translation: An Exploratory Study](https://arxiv.org/abs/2610.34841) | 本研究提出了一种量子感知的基于大语言模型的代码翻译工作流，能够将Qiskit程序翻译为PennyLane和Q#，零样本Pass@1最高达72.33%，展示出LLM在跨量子SDK代码翻译方面的巨大潜力。 |
| [^31] | [Silent and Costly: Characterizing and Detecting Prompt-Caching Mis-Integration in LLM Agents](https://arxiv.org/abs/2610.34817) | 该论文首次对10个开源智能体系统中120个提示缓存相关的拉取请求进行实证研究，揭示了提示缓存错误集成往往静默失败且难以察觉的问题，并据此提出了检测工具CACHELINT。 |
| [^32] | [Arya Forge: Auditing Validation and Release Approval for Financial Instruction Data](https://arxiv.org/abs/2610.34778) | Arya Forge 将类型化记录、内容检查与绑定数据集清单的发布审批相结合，构建了一个能在金融指令数据流水线中拦截数值错误与政策违规、并区分内容有效性与数据发布许可的审计框架。 |
| [^33] | [The Physical Cost of Context: Quality-Constrained Resource Accounting for Coding Agents](https://arxiv.org/abs/2610.34763) | 该论文提出了一种质量门控的资源核算协议，指出编码代理上下文策略的真实成本取决于调用次数和前缀缓存复用而非仅仅提示词长度，并证明预加载检索块（Eager）策略在保证代码质量的前提下可减少35.7%的模型调用和33.2%的GPU能耗。 |
| [^34] | [Thinking-in-Bets at Test Time Reduces Sycophantic Revisions in LLM Agents](https://arxiv.org/abs/2610.34654) | 提出无需训练的“押注式思考”方法，让智能体在读取用户反馈前先私密记录其论断、验证依据与置信度，从而在受到质疑时保护已验证的成果、减少LLM智能体的谄媚性修改。 |
| [^35] | [AgentProofBench: Benchmarking Reusable Verifiers for Web Agents](https://arxiv.org/abs/2610.34634) | AgentProofBench是一个评估Web智能体可复用验证器的基准，涵盖97个任务和四个Web服务，实验表明部署状态面板正确性可以有效区分正确交付与错误交付的程序分配。 |
| [^36] | [CodeMeepo: A Multi-Agent Workflow for Repository-Level Code Migration and Optimization](https://arxiv.org/abs/2610.34633) | CodeMeepo提出了一种多智能体流水线，通过将翻译与验收测试的编写职责分离、拆分大型仓库任务以克服上下文窗口限制，并对翻译后的代码进行优化，实现了完整且高效的仓库级代码迁移。 |
| [^37] | [Built to Remember, Shipped to Forget: Operational Contracts, Failures, and Repairs in Containerized LLM Agent Deployments](https://arxiv.org/abs/2610.34620) | 该论文首次实证研究了LLM智能体框架的持久化需求与容器平台“容器一次性”假设之间的矛盾，提出14条“运维契约”，并通过433个开源智能体系统、7种诱导故障事件和122个开发者修复对契约进行了评估。 |
| [^38] | [Do AI Coding Agents Write Infrastructure as Code Like Humans? An Empirical Study of GitHub Pull Requests](https://arxiv.org/abs/2610.34613) | 该研究首次大规模实证分析了34.9万个GitHub IaC拉取请求，发现AI编码代理编写的PR规模约为人类的四倍且更常改动核心基础设施模块，尽管其总体合并率更高，但在同一仓库内却比人类PR更难获得维护者的合并认可。 |
| [^39] | [NovaFabric: A Black-Box Recorder for AI Agents with Sealed, Replayable, Diffable and Auditable Runs](https://arxiv.org/abs/2610.34594) | 提出了开源黑盒记录器NovaFabric，可将AI智能体的复杂运行过程封装为密封、可重放、可对比且可审计的证据胶囊，解决了传统日志可变、不完整且无法重放执行的问题。 |
| [^40] | [Software Engineering for AI-Driven Science](https://arxiv.org/abs/2610.34573) | 本文提出“面向AI驱动科学的软件工程”（SE4AI4Sci）这一新兴研究方向，通过“科学保证链”框架和六个真实失败案例分析，指出了保障AI驱动科学发现可靠性与可复现性的关键研究挑战。 |
| [^41] | [Agentic AI Orchestration Framework for Spatial Intelligence: Supply Chain Deployment](https://arxiv.org/abs/2610.34569) | 提出了ANASSA参考架构，通过类型化数据契约、权威空间基础验证、多智能体编排与两道人在环审批，解决工业空间决策中跨系统割裂、缺乏统一参考框架的问题，并已在供应链多灾害风险评估场景中实现实际部署。 |
| [^42] | [MOFjson: An Explicit Topological Representation of Metal-Organic Frameworks for Language Models](https://arxiv.org/abs/2610.34549) | MOFjson首次提出将MOF结构分离为包含显式键移矢量的原始原子层与暴露次级结构单元和网络拓扑的派生层，使语言模型在MOF结构理解的各项任务上大幅超越现有基线。 |
| [^43] | [Multimodal Requirements Traceability Link Recovery in Industrial Settings](https://arxiv.org/abs/2610.34503) | 该论文提出了一种多模态RAG扩展方法，通过分别检索文本和图像候选、利用视觉-语言模型进行分类并用言语化置信度过滤预测链接，从而在工业需求可追溯性链接恢复任务中将F1分数从基线的0.114提升至0.310。 |
| [^44] | [A Multi-Domain Study on the Effectiveness of LLMs at Writing Property-Based Tests](https://arxiv.org/abs/2610.34487) | 该研究首次将基于LLM的属性测试编写工具从Python库扩展评估到深度学习、数据库、系统软件、编译器和LLM服务五个领域的十个程序，发现Agentic-PBT的分支覆盖表现更优、可与领域专用模糊测试工具相媲美，并通过180份错误报告（65份已获确认）证明了其跨领域发现真实缺陷的能力。 |
| [^45] | [CyberDual: AI Attackers versus AI Code Reviewers in Real Software Maintenance](https://arxiv.org/abs/2610.34435) | 该论文提出CyberDual竞技场，让AI攻击者智能体在真实开源软件维护任务中隐藏可利用漏洞、AI审查者智能体负责发现并修复，并以重新执行攻击者PoC的结果裁决胜负，揭示现有审查智能体虽能拦截85-91%的攻击，但仍有约六分之一的分散式攻击可逃过全部审查，且开源权重的GLM 5.3是最强攻击者。 |
| [^46] | [SymLift: Neural Semantic Lifting for Compile-Free Vulnerability Detection in Context-Incomplete Code Fragments](https://arxiv.org/abs/2610.34431) | 提出SymLift框架，利用微调大语言模型将上下文不完整的代码片段提升为类LLVM中间表示中的漏洞相关程序关系，再通过Datalog规则实现无需编译整个项目、可审计且确定性的漏洞检测。 |
| [^47] | [From GPU to TPU with Coding Agents: What Works, What Fails, and What It Costs?](https://arxiv.org/abs/2610.34422) | 该研究构建了148个PyTorch研究项目的数据集，并使用Claude Code将其中10个项目从NVIDIA GPU迁移到Google Cloud TPU，通过独立评估发现迁移平均需2.4次人工干预、76%的执行尝试失败且近半数失败不报错，首次系统量化了智能体驱动代码迁移的可信度与成本。 |
| [^48] | [MulFCoder: Framework-conditioned Multi-agent for MLLM-based Multi-framework Front-end Code Generation](https://arxiv.org/abs/2610.34418) | MulFCoder提出了一种框架条件化的多智能体方法，通过显式编码框架约束并协调Grounder、Planner、Writer、Judger四个智能体，实现了从UI截图生成React/Vue/Angular等多框架的可运行前端代码。 |
| [^49] | [ComBeautify: Leveraging Code Change Summaries for Conventional Commit Generation](https://arxiv.org/abs/2610.34374) | 本文提出ComBeautify，利用大语言模型对Git差异进行保留上下文的摘要，自动生成符合规范化提交规范（CCS）的提交信息，帮助开发者更准确地表达代码变更意图。 |
| [^50] | [EAST: Embodied AI Safety Testing via Systematic Fault Injection in Digital Twins](https://arxiv.org/abs/2610.34358) | EAST是一个基于数字孪生的具身AI安全测试框架，它通过系统化的故障注入方法，将文献中记录的安全证据转化为覆盖感知、指令、环境、动作和时序的可执行、可验证的安全测试。 |
| [^51] | [Developer Task Delegation to LLMs and the Trust Factor: A Mixed Methods Study](https://arxiv.org/abs/2610.34317) | 该研究通过22次访谈和160名从业者的问卷调查，首次系统揭示了开发者向大语言模型委派任务的四种模式（完全委派、部分委派、自我委派和任务回避），并通过统计分析证明信任是影响任务委派决策的关键因素。 |
| [^52] | [Repairing Broken Access Control in Web Applications with LLMs: An Empirical Benchmark Study](https://arxiv.org/abs/2610.34290) | 该研究构建了首个用于修复Web应用失效访问控制漏洞的基准BAC-Bench，包含来自六个真实Web应用的46个漏洞，并提出了相应的修复策略框架，为评估大语言模型修复此类逻辑漏洞的能力提供了标准化平台。 |
| [^53] | [Beyond Benchmarks: Can Small Open-Weight Models Get Real Coding Work Done?](https://arxiv.org/abs/2610.34241) | 该实证经验报告基于4,165次真实编码代理会话、623个调试任务和三项委托研究，指出小型开放权重模型能否完成真实编码工作取决于工具调用格式、缓存复用等系统层面的工程细节，而非基准测试成绩。 |
| [^54] | [Honesty Testing of LLMs and AI Agents: A Vision for Software Engineering](https://arxiv.org/abs/2610.34239) | 这篇愿景论文提出将LLM和AI智能体的诚实性测试作为软件工程的新研究领域，把忠实报告确立为可规定、可测试、可维护的软件质量需求，并定义了“诚实性缺陷”概念及初步测试框架。 |
| [^55] | [$\Delta\tau$-Bench: Pushing Agents Toward a Moving Capability](https://arxiv.org/abs/2610.34238) | Δτ-Bench是一个模拟工具生态系统持续演化的可执行智能体基准测试，包含覆盖42个领域的1,588个多轮任务，评估显示最强模型的任务成功率也仅达42.3%，揭示了智能体在应对工具环境动态变化方面的重大不足。 |
| [^56] | [The Policies of Programming Language Ecosystems: A "Qualitative" Analysis](https://arxiv.org/abs/2610.34187) | 本文对八个编程语言生态系统的政策进行了定性分析，发现各生态系统的政策总体呈现分歧而非趋同，仅有恶意软件、知识产权和内容删除等少数类别获得普遍认可。 |
| [^57] | [SpeC2Rust: Specification-Driven Repository-Level C-to-Rust Translation](https://arxiv.org/abs/2610.34141) | SpeC2Rust提出了一种规格驱动的三阶段流水线，通过项目意图结构化生成持久化的仓库级与模块级规格，结合实时符号注册表的规格引导生成和分层反馈修复，实现了仓库级C代码到安全Rust代码的自动翻译。 |
| [^58] | [Bug Severity Classification with Fine-Tuned Open LLMs: A Paired Comparison with TF-IDF and CodeBERT Baselines on Eclipse Reports](https://arxiv.org/abs/2610.34127) | 该研究对可在笔记本电脑上运行的开源大语言模型与更便宜的TF-IDF和CodeBERT基线在Eclipse漏洞报告严重性分类任务上进行了配对比较，发现微调后的LLM性能优于传统方法。 |
| [^59] | [Securing Agent-Generated Code with Its Own Development History](https://arxiv.org/abs/2610.34098) | TRACER将编码智能体自身的开发历史（编辑轨迹与陈述假设）作为安全审查证据，通过符号级变更跟踪、验证智能体理由、评估并功能验证修复、附加可执行检查，来发现和修复智能体生成代码中被遗漏的安全防护步骤。 |
| [^60] | [How Do Small Coding Agents Use Code Graphs and Embeddings?](https://arxiv.org/abs/2610.33958) | 研究发现，代码图与嵌入搜索等结构化工具对小型编码智能体的帮助最大——在 1.7B-4B 参数规模可带来 17-21 个百分点的定位准确率提升，但收益随模型规模增大而递减，从而修正了“工具可以补足模型对仓库结构理解”的普遍假设。 |
| [^61] | [Why Does Structured Output Fail? An Empirical Study of Prompt-Enforced Pydantic Schemas Across Nine LLMs](https://arxiv.org/abs/2610.33949) | 该研究提出了一个细粒度的模式符合性失败分类法和确定性分类器，通过对九个大语言模型配置的3,997次生成实验，揭示了单一有效率指标所掩盖的多样化失败模式，并发现主导失败类别会随模式约束的不同而变化。 |
| [^62] | [DRILS: Dependency-based Requirement Change Impact Analysis Using LLMs](https://arxiv.org/abs/2610.33920) | DRILS是一个基于大语言模型的多智能体需求变更影响分析框架，通过RE-RAG检索增强技术提供聚焦领域上下文，并利用LLM-as-a-judge机制促进候选影响集的多样性，从而自动准确地识别需求变更的影响范围。 |
| [^63] | [The Computational Assessment Stack: An Architecture for AI-Native Measurement](https://arxiv.org/abs/2610.33901) | 本文提出计算化评估堆栈（CAS），一种以机器可读、模块化、可版本化的统一计算构念表示（知识图谱）为基础的四层AI原生测量架构，通过计算化内容生成、计算心理测量学与编排层，在智能体AI主导测试开发的新范式下保障测评质量。 |
| [^64] | [FlashScript: Compiling Recurring Agent Workflows into Reusable Neuro-Symbolic Programs](https://arxiv.org/abs/2610.33876) | FlashScript 提出一套将成功的智能体运行编译为可复用神经符号程序的准则——把动作与数据转换变为代码、实例特定值变为参数、需模型判断的决策变为局部化的 LLM 调用——从而将非确定性局部化，使确定性程序与神经调用可分别审查和测试，显著降低重复任务的成本与结果波动。 |
| [^65] | [Numina Fuse: Interactive Autoformalization in Lean 4](https://arxiv.org/abs/2610.33860) | Numina Fuse 是一个用于 Lean 4 数学自动形式化的智能体原生应用，其核心创新在于以人类监督为中心，通过蓝图、依赖图和代码等视图让用户监督 AI 编排器的工作，并采用将项目状态与编排逻辑解耦的架构，使智能体能随模型进步逐步自主接管更多形式化工作。 |
| [^66] | [ArchGuard-APR: Architecture-Constraint-Aware Multi-Agent Program Repair Using Repository Graphs and Execution Feedback](https://arxiv.org/abs/2610.33853) | ArchGuard-APR 提出了一个架构约束感知的多智能体程序修复原型，通过不可变的仓库图谱、显式架构配置与隔离执行反馈来引导和约束补丁生成，使修复结果不仅通过测试，还不违反架构边界。 |
| [^67] | [Software Engineering LLMs: Why We Need Them and How Software Engineering Helps Build Them](https://arxiv.org/abs/2610.33851) | 软件工程专业知识可有效指导编码大语言模型的能力选择、可验证训练数据构建与评估，本工业案例通过RefactorForge流水线为仓库级重构生成了4,572条经验证的教师轨迹，并以极小的数据成本微调了一个服务10万余名开发者的920亿参数编码模型。 |
| [^68] | [VulDec: Vulnerability Detection in Binaries via Neural Decompilation](https://arxiv.org/abs/2610.33716) | VulDec提出了一个大规模、带多架构CWE详细标注的汇编-C函数对齐数据集，并首次构建了利用LLM神经反编译技术实现二进制漏洞检测的端到端框架。 |
| [^69] | [Covered Is Not Checked: Which Branch Outcomes Would Your Tests Notice?](https://arxiv.org/abs/2610.33708) | 提出无需切片的Veritas方法，通过强制相反的分支结果并验证原本通过的测试是否因此失败，来判断被覆盖的分支结果是否真正被测试检验，从而无需动态依赖分析即可区分代码执行与实际缺陷检测能力。 |
| [^70] | [SWE-Ladder: Learning to Localize, Reproduce, and Patch via Progressive Privileged Self-Distillation](https://arxiv.org/abs/2610.33667) | 该论文提出SWE-Ladder框架，将已验证的修复过程分解为定位、复现和修补三个阶段的特权证据，通过基于规则的路由器和冻结教师模型按修复顺序进行渐进式自蒸馏，并结合GRPO优化完整轨迹，为SWE智能体提供了传统强化学习和OPSD所缺失的阶段特定训练指导。 |
| [^71] | [Build or Adopt? Custom-Built vs. General-Purpose LLM Agent Harnesses for Automotive GUI Testing](https://arxiv.org/abs/2610.33636) | 该研究在沃尔沃集团对比了自建LLM智能体框架与通用框架（Claude Code、OpenCode）在车载信息娱乐系统GUI测试中的表现，发现通用框架成功率相当但成本更高、波动更大，然而其节省的工程投入足以弥补成本劣势。 |
| [^72] | [Developers as Managers of AI Agents: A Workplace Shadowing Study](https://arxiv.org/abs/2610.33588) | 该研究通过对11名专业开发者的工作场所跟随观察，首次系统揭示了开发者在日常工作中管理自主AI编程智能体的三种模式——监督者、委派者和学习者，并发现开发者约三分之一的时间用于编写提示词和检查AI响应。 |
| [^73] | [Can Agentic Code Review Catch Vulnerabilities Before They Are Merged?](https://arxiv.org/abs/2610.33509) | 提出了SecPRBench——首个基于经过验证的真实漏洞引入提交来评估智能体安全代码审查的综合性基准，涵盖Firefox、V8、OpenSSL等九个关键系统的265个任务，并发现八个前沿模型的召回率仅为32.5%至78.1%。 |
| [^74] | [Secure and Functional Code Generation with Self-Evaluation](https://arxiv.org/abs/2610.33450) | SecGent是一个推理时框架，通过从CWE数据库检索弱点上下文提供任务特定的安全指导，并结合功能与安全测试的自我评估反馈，在不牺牲代码功能性的前提下提升生成代码的安全性。 |
| [^75] | [BoardSafety: AI-Ready Data from Manufacturer Documentation for Development-Board Safety](https://arxiv.org/abs/2610.33448) | 提出了BoardSafety基准，通过将厂商数据手册和硬件指南转换为板卡特定的安全规则库，构建了带标注的操作场景数据集，用于评估大语言模型判断开发板操作安全性的能力。 |
| [^76] | [OneCodeReviewBench: Typed Decisions for Code Review and Search](https://arxiv.org/abs/2610.33389) | 该论文提出OneCodeReviewBench基准，包含覆盖12种编程语言和YAML的1,968个具有类型化答案（布尔、类别、有序）的合成代码审查案例，并通过十个模型-适配器系统的评测揭示了当前领先模型在代码审查决策任务上的性能水平。 |
| [^77] | [PMU Foundation Model Bridge: A Tool for Connecting Synchrophasor Streams to Time Series Foundation Models](https://arxiv.org/abs/2610.33387) | 本文提出了一个开源工具PMU基础模型桥接器，能够将PMU/PDC同步相量数据流连接到多种公开可用的时间序列基础模型，并支持基于C37.118的在线工作流和离线分析。 |
| [^78] | [EVM-Anchored Provenance for AI-Assisted Code Evolution: A Smoke-Scale Deployment Report](https://arxiv.org/abs/2610.33350) | 该论文报告了一个面向AI辅助代码演进的EVM锚定溯源系统的冒烟规模部署，通过MCP服务器、Solidity合约、IPFS固定与可重建的SQLite索引，为AI生成的代码状态流提供链上存在性、排序性和不可否认性证明，并明确界定了锚定不证明真实性的边界。 |
| [^79] | [From Cosmetic to Structural: A Paired Experiment on Metric-Guided Agentic Refactoring](https://arxiv.org/abs/2610.33343) | 显式静态分析反馈能够引导编码智能体从表面修饰性编辑转向结构性重构，将有效重构率提升21至26个百分点，且其收益取决于初始代码质量，各语言的盈亏平衡点位于CH 4.9至7.1之间。 |
| [^80] | [Responsibility Boundaries for Agentic Systems: A Comparative Evaluation of Probabilistic Agentic Programming](https://arxiv.org/abs/2610.33312) | 该论文提出概率智能体编程（PAP），将责任边界作为显式编程抽象，通过中介内核、条件安全性归纳证明和反向模拟，为语言模型智能体的授权、审批与证据追踪提供形式化保障。 |
| [^81] | [Operationalizing Governed Agentic Workflows in Regulated Finance: A Customer Zero Case Study using Zafin AIOS](https://arxiv.org/abs/2610.33253) | 本文提出Zafin AIOS这一端到端编排与控制平台，并通过“零号客户”内部生产案例研究，验证了治理型智能体工作流在受监管金融行业中从试点走向生产落地的可行性与有效性。 |
| [^82] | [Position, Role or Label? A Controlled Study of Instruction Conflicts in LLM Agents](https://arxiv.org/abs/2610.33201) | 该论文构建了包含1,080个受控场景的WhoWins基准，通过对14个开放权重LLM智能体约189万条回复的分析，揭示了指令冲突中独立于角色和标签的强位置效应——首条指令在消息内和跨消息场景中均显著占优。 |
| [^83] | [Cross-Platform Application Migration to HarmonyOS: An Agentic Pipeline with Artifact-Derived Verification](https://arxiv.org/abs/2610.33197) | 该论文提出了HMOS Migrator，一个智能体流水线，能自动将Android/iOS应用源码转换为ArkTS项目，并通过在真实设备上重放录制交互，以及仅基于源应用行为和流水线产物的七个加权指标进行无参考实现的质量评估，从而验证迁移后的功能正确性。 |
| [^84] | [WebUniverse: Scaling Verified Training Data for Agentic Web Development](https://arxiv.org/abs/2610.33149) | WebUniverse从开源仓库和真实网页构建了包含15K个可验证任务环境和开发轨迹的数据集，用于训练前端Web开发智能体，显著提升了模型在Vision2Web等基准上的表现。 |
| [^85] | [Speculative Forking: Turning Idle Wait into Parallel Exploration for Repository Environment Setup](https://arxiv.org/abs/2610.33129) | SpecBuild通过“投机分叉”机制将环境命令的漫长等待时间用于并行探索多条安装分支，返回首个成功配置的环境并丢弃其余分支，从而显著提升基于LLM的代码仓库环境搭建的速度与成功率。 |
| [^86] | [Navigating Epistemic Parity in LLM Agents: A Benchmark for Cross-Source Conflict Resolution Between Memory and Procedural Skills](https://arxiv.org/abs/2610.33128) | 该论文提出了一个衡量LLM智能体如何裁决记忆与程序性技能之间跨源指令冲突的基准，包含记忆-技能矛盾的四分类法以及52个由确定性检查器评分的测试场景。 |
| [^87] | [ToolContractGate: Versioned Trusted Contracts for Drift-Safe Agent Tool Invocation](https://arxiv.org/abs/2610.33108) | 提出 ToolContractGate，通过版本化可信合约将代理工具调用请求与已批准的 OpenAPI 源工件确定性绑定，在无误阻断的前提下检测出全部 35 个漂移故障，且平均门控延迟仅 0.18 毫秒。 |
| [^88] | [Beyond Correctness: Attribution of Natural-Language Documentation for Formal Libraries](https://arxiv.org/abs/2610.33094) | 该论文提出“归因”概念以检验文档陈述是否真正依赖于函数的特定行为，并实现了 DocAuth 文档生成方法，通过过滤掉已证明但不可归因的性质，并为每句发布的文档附加可检查的形式化证据，从而提升形式化库自然语言文档的可信度。 |
| [^89] | [RADAgent: Automating RADseq Pipelines with Private Open-Source LLMs](https://arxiv.org/abs/2610.33009) | RADAgent利用私有开源大语言模型自动化RADseq分析流程，将传统上需要专家跨多个阶段协调的异构生物信息学工作流（包括状态跟踪、结果验证和故障处理）实现无人监督执行。 |
| [^90] | [What Open-Source Projects Tell Their Coding Agents: A Census of AGENTS.md and CLAUDE.md](https://arxiv.org/abs/2610.31715) | 这项研究首次对十三种语言的活跃 GitHub 仓库中的 AGENTS.md 和 CLAUDE.md 指令文件进行大规模普查，系统揭示了开源项目向 AI 编码智能体传达的指令构成（项目知识与代码、构建、工作流规则的混合），以及各项目在 AI 署名披露等问题上的立场分歧。 |
| [^91] | [Separate Field Scoring Protects Unused Tools When Agents Pool Tool-Use Experience](https://arxiv.org/abs/2610.30867) | 提出SYNAPSE方法，将工具使用记录作为独立字段进行评分，在智能体共享工具使用经验提升已使用工具检索效果的同时，有效避免了从未被使用过的工具检索性能的下降。 |
| [^92] | [Can Loan Processors and Underwriters Spot AI-Edited Statements? An Empirical Study](https://arxiv.org/abs/2610.30758) | 该研究通过实证实验发现，经验丰富的贷款审核员和处理员在仅凭文件的首次审阅中，仅有约10.6%的判断准确指出了AI伪造财务文件中的编辑痕迹（且全部依赖标识符不一致），表明仅靠人工文件审查很难发现AI编辑的伪造贷款文件。 |
| [^93] | [SpecAgent: Empowering Program Verification with Agentic Synthesis of Formal Program Specifications](https://arxiv.org/abs/2610.05132) | SpecAgent 提出一个智能体式框架，结合依赖感知规划与检索增强生成等组件，为真实世界 C 程序自动合成高质量 ACSL 形式化规约，克服了传统前向式流程和局部修复无法解决的级联验证失败问题。 |
| [^94] | [A2ABreak: Systematic Security Analysis of the A2A Protocol](https://arxiv.org/abs/2609.10871) | 本文提出A2ABreak框架，首次对A2A协议进行系统性安全分析，通过LLM辅助从自然语言规范中提取经验证的有限状态机模型（37个状态、76个转换），并利用对抗性验证系统性发现协议层面的安全漏洞。 |
| [^95] | [Broken on Arrival: Silently Defective LLM Artifacts in Public Model Registries and How to Catch Them](https://arxiv.org/abs/2609.05881) | 首次系统性冒烟测试了公共模型注册库中的327个量化GGUF模型制品，发现官方Ollama库中存在五个静默失效的量化模型转换制品，并提出了一套可有效捕获此类缺陷的检测流程。 |
| [^96] | [AgentLogs: A Dataset for Opening the Black Box of GitHub's Cloud Agent](https://arxiv.org/abs/2608.29204) | 该论文发布了AgentLogs——首个大规模GitHub智能体活动数据集，包含30余万个任务、54万余个会话及6400多万条逐步日志，首次打开了GitHub Copilot云智能体完成软件工程任务全过程的“黑盒”。 |
| [^97] | [SemaPLC: A Project-Grounded, Verification-Gated Agent Harness for PLC Code Generation](https://arxiv.org/abs/2608.18565) | SemaPLC通过严格的外部验证门控和项目集成，显著提升了大型语言模型生成的PLC代码在真实工业环境中的可靠性与通过率。 |
| [^98] | [EntailLLM: Verifying Agent-Generated Vulnerability Discovery Paths with Domain Knowledge via Logic Programming](https://arxiv.org/abs/2608.01763) | EntailLLM框架让专门的LLM智能体提出二进制漏洞的函数调用图遍历路径，并通过标注逻辑下的蕴含推导将其与独立的领域知识图谱和逻辑程序进行验证，从而提升安全关键场景中漏洞分析的可靠性。 |
| [^99] | [Change2Task: From Repository Changes to Executable Coding Agent Tasks and Environments](https://arxiv.org/abs/2607.28591) | Change2Task 提出了一个基于仓库历史的系统，能够将已合并的拉取请求自动转换为健康现代代码版本上经过验证的可执行编码智能体任务与环境，为编码智能体的训练、基准测试和持续评估提供可扩展的数据供给。 |
| [^100] | [In the Driver's Seat: A Multi-Company Study on the Reality of Autonomous Driving System Testing](https://arxiv.org/abs/2607.15820) | 该研究通过对六个国家九家公司专家的访谈，首次系统揭示了工业界自动驾驶系统测试的真实运作方式——一个由规划证据需求、产生与评估证据等反复活动构成的证据构建过程。 |
| [^101] | [Compaction as Epistemic Failure: How Agentic LLM Tools Fabricate Confirmed Results from Killed Processes](https://arxiv.org/abs/2607.13071) | 本文揭示了Claude Code等智能体LLM工具会将超时被杀进程的部分输出错误记录为“已确认结果”并在会话间传播假阳性，其根源在于混淆了“终端观察”与“持久化存储”这两种信息状态。 |
| [^102] | [MicroAgent: Context-Augmented Multi-Agent Framework for Automatic Microservice Decomposition](https://arxiv.org/abs/2606.29742) | 该论文提出了MicroAgent，一个上下文增强的多智能体框架，通过多智能体协作并融合丰富的上下文信息，自动化地将单体应用分解为高内聚的微服务，从而克服了传统自动化方法语义理解不足以及直接使用大语言模型时忽视上下文与设计原则的局限。 |
| [^103] | [A Difference Without a Distinction: A Measurement Study of Release Trust Evidence in Package Registries](https://arxiv.org/abs/2606.22593) | 对包注册表发布信任证据的大规模测量研究表明，pnpm的新证据策略主要误伤手动向后移植的安全修复（semver 6.3.1造成16个热门失败中的14个），并首次对比了npm、PyPI、Maven Central、crates.io和RubyGems五大注册表的发布证据状况。 |
| [^104] | [Matching Matters: A Fair Quality-Efficiency Benchmark for Command-Line Agents](https://arxiv.org/abs/2606.21140) | 本文提出AgentMeter基准测试及新指标AMS，通过联合评估任务质量、预算敏感性和资源密集型零奖励执行，实现对大语言模型与命令行界面（CLI）组合匹配效果的更公平、更全面的质量-效率评估。 |
| [^105] | [Tiny Machine-Learning Operations within Cyber-Physical Systems: a Field Study](https://arxiv.org/abs/2606.20952) | 该论文提出并实地验证了一个以知识为中心的端到端微型机器学习运维（TinyMLOps）管线，融合领域物理知识、专家推测与传感器数据流，为资源受限的网络物理系统提供可解释、低内存占用且可部署在设备端的模型。 |
| [^106] | [Enhancing Software Engineering Through Closed-Loop Memory Optimization](https://arxiv.org/abs/2606.05646) | 提出MemOp闭环记忆优化框架，将记忆效用定义为经过验证的下游影响，使其同时充当与任务无关的评估基准和无需标注的优化信号，从而让软件工程智能体能够跨任务保留、改进和复用经验。 |
| [^107] | [EviACT: An Evidence-to-Action Framework for Agentic Program Repair](https://arxiv.org/abs/2605.27238) | EviACT框架通过检索支架、编译门和测试驱动门三种证据驱动护栏协调修复流程，在四个基准上使智能体程序修复的解决率提升1.6-6.0个百分点，同时将每缺陷API成本降低70.1-88.6%。 |
| [^108] | [Firefly: Illuminating Large-Scale Verified Tool-Call Data Generation from Real APIs](https://arxiv.org/abs/2605.17558) | FireFly通过先让大模型探索真实API再从观察结果反向合成任务的新范式，从真实MCP服务器生成在构造上保证标签正确性的大规模工具调用训练数据。 |
| [^109] | [Web Agents Should Adopt the Compile-and-Execute Paradigm](https://arxiv.org/abs/2605.14290) | 该论文主张网页智能体应放弃 ReAct 这一默认架构，改用“编译-执行”范式——即在观察运行时网页内容之前先确定任务程序再执行，从而防止网页中多来源的不可信内容（如提示注入）篡改智能体的控制流。 |
| [^110] | [Terminus-4B: Can a Smaller Model Replace Frontier LLMs at Agentic Execution Tasks?](https://arxiv.org/abs/2605.03195) | 本文提出 Terminus-4B——一个通过 SFT 和基于量规的 LLM-as-judge 强化学习后训练的 Qwen3-4B 模型，证明了经过针对性微调的小型语言模型可以在智能体终端执行任务中达到与前沿大模型相当的性能，从而可作为子智能体替代前沿模型。 |
| [^111] | [Human-AI Synergy in Agentic Code Review](https://arxiv.org/abs/2603.15911) | 该论文通过对300个开源项目中278,790次代码审查对话的大规模实证分析，比较了人类审查者与AI智能体的反馈差异，揭示了人机协作模式及交互方式如何影响代码审查结果与建议采纳率。 |
| [^112] | [A Causal Perspective on the Role of Training Dynamics for Interpreting Privacy Risks in Code Models](https://arxiv.org/abs/2512.07814) | 该研究构建了包含多种PII类型的数据集并结合结构因果模型，首次从因果视角证明了个人身份信息的训练动态（可学习性）因果性地决定了其在代码大模型中被泄露的风险，且不同PII类型的泄露风险存在显著异质性。 |
| [^113] | [PBFuzz: Agentic Directed Fuzzing for PoV Generation](https://arxiv.org/abs/2512.04611) | 本文提出PBFuzz，一种模仿人类专家工作方式的智能体式定向模糊测试框架，通过迭代提取语义约束、形成触发假设并利用调试反馈，高效生成漏洞证明输入。 |
| [^114] | [GPU-Accelerated Belief Propagation for Program Analysis](https://arxiv.org/abs/2509.22337) | FastLBP是一个面向程序分析的GPU加速信念传播框架，通过统一表示灵活更新策略、依赖分析实现并行执行以及内存高效的局部结构实现，相比CPU方法最高取得17.42倍加速。 |
| [^115] | [What's in a Package? Getting Visibility Into Dependencies Using Security-Sensitive API Calls](https://arxiv.org/abs/2408.02846) | 本文提出了一种有文档记录且可重复的方法，用于推导特定生态系统的安全敏感API列表，帮助开发人员了解依赖包可访问的敏感资源（如文件系统、网络、进程等），从而实现更细粒度的依赖控制和风险识别。 |
| [^116] | [How Quickly Do Development Teams Update Their Vulnerable Dependencies?](https://arxiv.org/abs/2403.17382) | 本文指出现有的依赖项更新响应性指标MTTU和MTTR因忽略浮动版本和近期更新优先等关键细节而无法准确反映开发团队的更新实践，旨在提出更准确的改进衡量方案。 |

# 详细

[^1]: 自由观察，自由推理：简单的LLM运行框架在根因分析诊断上超越手工构建的智能体

    Free to Look, Free to Reason: Simple LLM Harnesses Outdiagnose Hand-Crafted Root Cause Analysis Agents

    [https://arxiv.org/abs/2610.35767](https://arxiv.org/abs/2610.35767)

    该论文发现，不预设调查流程、让模型自由观察和推理的简单LLM框架，在微服务故障根因分析上比手工设计的LLM智能体和非LLM统计因果方法诊断得更准确。

    

    微服务故障的根因分析（RCA）吸引了众多LLM智能体，这些智能体的调查过程是通过标准操作流程、固定阶段和角色智能体等方式手工构建的。由于每个智能体都在各自的数据上仅针对少量基线进行评估，因此尚不清楚这种设计是否优于让一个能力强大的模型自主开展调查。我们在RCAtlas基准上比较了三种不预设任何模型调用方式的简单框架（对证据包的单次调用、SQL智能体以及Codex编码智能体）、五个已发表的LLM RCA智能体，以及统计方法和因果方法。RCAtlas包含来自八个公开数据集的90个经验证的案例，覆盖六个模型和三种推理努力程度。在两个系列方法都运行过的每一个模型上，简单框架都是最准确的（在三个中端模型上PR@1达到65-77%），领先于已发表的智能体（16-53%）和非LLM方法（6-39%）。已发表的智能体在程序中将十分之一到全部的模型调用固定下来（摘要原文在此处中断）。

    arXiv:2610.35767v1 Announce Type: new  Abstract: Root cause analysis (RCA) of microservice failures has attracted many LLM agents whose investigation is hand-crafted, through standard operating procedures, fixed stages and role agents. Each is evaluated on its own data against few baselines, so it is unknown whether this design beats letting a capable model investigate on its own. We compare three simple harnesses that prescribe none of the model's calls (one call over an evidence package, a SQL agent and the Codex coding agent) with five published LLM RCA agents and with statistical and causal methods on RCAtlas, 90 validated cases from eight public datasets, across six models and three reasoning efforts. The simple harnesses are the most accurate (PR@1 65-77% on three mid-tier models), ahead of the published agents (16-53%) and the non-LLM methods (6-39%), on every model on which both families were run. The published agents fix from a tenth to all of their model calls in the program,
    
[^2]: FLOWSTEP：一个学习预测 GitHub Actions 工作流步骤的框架

    FLOWSTEP: A Framework for Learning to Predict GitHub Actions Workflow STEPs

    [https://arxiv.org/abs/2610.35650](https://arxiv.org/abs/2610.35650)

    FLOWSTEP 通过明确区分可复用 Action 与项目特定 Script 两类工作流步骤，并结合大语言模型与对比学习，显著提升了 GitHub Actions 工作流步骤自动补全的准确性。

    

    持续集成与持续交付（CI/CD）是现代软件开发的基础，然而配置诸如 GitHub Actions（GHA）工作流之类的流水线仍然复杂且容易出错。近期研究已探索了基于 Transformer 的模型用于 GHA 工作流推荐，但现有方法通常将工作流步骤建模为单一的同质生成任务。这种建模方式忽略了一个关键区别：可复用的 Actions 是结构化的生态系统级组件，而项目特定的 Scripts 则编码了本地构建、测试和部署的语义。因此，当前模型难以捕获准确完成工作流所需的不同形式的知识。在本工作中，我们提出了 FLOWSTEP，一个用于 GHA 工作流步骤补全的框架，它明确区分 Action 生成与 Script 生成。FLOWSTEP 将大语言模型（LLMs）与对比学习相结合，以更好地捕获语义关系……

    arXiv:2610.35650v1 Announce Type: new  Abstract: Continuous Integration and Delivery (CI/CD) are fundamental to modern software development, yet configuring pipelines such as GitHub Actions (GHA) workflows remains complex and error-prone. Recent work has explored Transformer-based models for GHA workflow recommendation, but existing approaches typically model workflow steps as a single homogeneous generation task. This formulation overlooks a key distinction: reusable Actions are structured, ecosystem-level components, whereas project-specific Scripts encode local build, test, and deployment semantics. Consequently, current models struggle to capture the different forms of knowledge required for accurate workflow completion. In this work, we present FLOWSTEP, a framework for GHA workflow step completion that explicitly distinguishes between Action and Script generation. FLOWSTEP combines large language models (LLMs) with contrastive learning to better capture semantic relationships amo
    
[^3]: GEAR：面向智能体系统的基于特征的配置与代码生成

    GEAR: Feature-Based Configuration and Code Generation for Agentic Systems

    [https://arxiv.org/abs/2610.35592](https://arxiv.org/abs/2610.35592)

    GEAR是一个开源工具，通过基于特征的统一规范来配置多智能体系统，并能自动生成适用于十个不同智能体框架的代码实现，解决了开发者需要学习不同框架API的问题。

    

    构建多智能体系统（MAS）需要选择智能体、编排策略、通信机制、推理方法和记忆配置。这些选择依赖于特定框架的能力和抽象，导致开发者在探索不同实现方案时需要学习不同的API。我们提出了GEAR，这是一个开源工具，用于配置多智能体系统，并能从统一的规范生成特定框架的实现。GEAR采用基于特征的表示方式，使配置选项及其约束显式化，从而支持用户在代码生成前对规范进行验证。通过为十个智能体框架提供的连接器，用户可以从同一规范生成项目，并随后查看描述框架特定适配情况的报告。对映射可行性的初步评估显示，在320个被评估的特征-框架对中，有271个（84.69%）确定了相应的翻译策略。

    arXiv:2610.35592v1 Announce Type: cross  Abstract: Engineering multi-agent systems (MAS) requires choosing agents, orchestration strategies, communication mechanisms, reasoning approaches, and memory configurations. These choices depend on framework-specific capabilities and abstractions, requiring developers to learn different APIs when exploring alternative implementations. We present GEAR, an open-source tool for configuring MAS and generating framework-specific implementations from a common specification. GEAR uses a feature-based representation to make configuration options and their constraints explicit, enabling users to validate specifications before code generation. Through connectors for ten agentic frameworks, users generate projects from the same specification and can then inspect reports describing framework-specific adaptations. A preliminary assessment of mapping feasibility identifies translation strategies for 271 of 320 evaluated feature-framework pairs (84.69%). GEAR
    
[^4]: 让遥测管道可测量：一个用于端到端延迟、溯源与能耗感知基准测试的边缘物联网测试床与数据集

    Making the Telemetry Pipeline Measurable: An Edge IoT Testbed and Dataset for End-to-End Latency, Provenance, and Energy-Aware Benchmarking

    [https://arxiv.org/abs/2610.35591](https://arxiv.org/abs/2610.35591)

    提出了一个以测量为先的边缘物联网测试床与数据集，通过基于角色的事件驱动架构和嵌入阶段级时间戳的遥测消息封装，首次将端到端延迟、溯源和能耗作为可测量的一等遥测维度进行基准测试。

    

    以边缘为中心的物联网系统越来越多地通过传感器数值或平台特性来进行评估，然而遥测管道本身——即感知、序列化、消息中介、摄取和持久化这一序列——很少作为一个可测量的对象被公开发布。这限制了对时效性、溯源和运营成本的可复现基准测试，尤其是在早期物联网设计中，此时真实的部署和高质量的开放数据往往难以获得。我提出了一个以测量为先的边缘物联网测试床和数据集，将时效性、溯源和能耗提升为一等遥测维度。该测试床实现了一种基于角色的事件驱动架构，具有明确的源端-代理-聚合器职责划分，以及一个由模式治理的遥测消息封装，其中嵌入了关联标识符和统一的、跨角色的阶段级时间戳词汇表。为了追踪消息中介的影响，我扩展了一个MQTT代理（Eclipse Mosquitto）……（摘要原文在此处截断）

    arXiv:2610.35591v1 Announce Type: cross  Abstract: Edge-centric IoT systems are increasingly evaluated through sensor values or platform features, yet the telemetry pipeline itself, the sequence of sensing, serialization, message mediation, ingestion, and persistence, is rarely released as a measurable object. This limits reproducible benchmarking of timeliness, provenance, and operational cost, especially in early-stage IoT design where real deployments and high-quality open data are often unavailable. I present a measurement-first edge IoT testbed and dataset that elevate timeliness, provenance, and energy to first-class telemetry dimensions. The testbed operationalizes a role-based event-driven architecture with explicit Source-Broker-Aggregator responsibilities and a schema-governed TelemetryMessage envelope (Avro) that embeds correlation identifiers and a uniform, cross-role vocabulary of stage-level timestamps. To trace mediation effects, I extend an MQTT broker (Eclipse Mosquitt
    
[^5]: 中间的智能体：AI中介访问如何重塑Web漏洞利用

    The Agent in the Middle: How AI-Mediated Access Reshapes Web Vulnerability Exploitation

    [https://arxiv.org/abs/2610.35541](https://arxiv.org/abs/2610.35541)

    提出可执行基准测试VELA，系统研究了AI智能体中介访问对Web漏洞利用的影响，发现面向智能体的接口设计决定了漏洞可利用性，参考设计下91%的漏洞仍然可被利用。

    

    Web服务越来越多地暴露后端接口，允许用户将任务委托给AI智能体，这些智能体无需浏览网站即可直接调用服务操作。这种转变改变了现有Web漏洞被利用的方式。我们通过VELA来研究这些变化，VELA是一个可执行的基准测试，涵盖十二个Web应用家族中的44个漏洞。该基准保留了存在漏洞的后端，并从合法工作流中推导出面向智能体的接口。我们在多种接口配置下评估了三个商业智能体，使用受控比较来区分接口效应与智能体行为，并使用独立的判定标准来验证攻击结果。我们的结果表明，在参考配置下，智能体能够实现直接接口调用所能实现的所有攻击，因此可利用性主要由接口决定：在参考设计下，91%的漏洞仍然可被利用。

    arXiv:2610.35541v1 Announce Type: cross  Abstract: Web services increasingly expose backend interfaces that let users delegate tasks to AI agents, which invoke service operations directly without navigating websites. This shift changes how existing Web vulnerabilities can be exploited. We investigate these changes through VELA, an executable benchmark covering 44 vulnerabilities across twelve Web application families. The benchmark preserves vulnerable backends and derives agent-facing interfaces from legitimate workflows. We evaluate three commercial agents across multiple interface configurations, using controlled comparisons to separate interface effects from agent behavior and independent oracles to verify attack outcomes. Our results show that the agent realizes every attack that direct interface invocation realizes in the reference configuration, so exploitability is set mainly by the interface: 91% of vulnerabilities remain exploitable under the reference design. Interface restr
    
[^6]: JANUS：针对Web反序列化攻击的跨层防御自动合成方法

    JANUS: Synthesizing Cross-Layer Defenses Against Web Deserialization Attacks

    [https://arxiv.org/abs/2610.35532](https://arxiv.org/abs/2610.35532)

    JANUS提出了一种红蓝对抗框架，通过黑盒攻击者迭代探测攻击变体与防御者进行反事实分析并细化策略，实现针对Web反序列化攻击的跨层防御自动化合成。

    

    不安全的反序列化会将不受信任的数据转化为攻击者可控的执行，而单个漏洞可能允许多种攻击变体，从而绕过针对已知攻击定制的防御。现有防御通常在单一执行点运行，对完整攻击的可见性有限。跨层组合防御控制可以提供互补保护，但协调这些控制以保留合法功能并适应新的攻击变体需要大量的专家投入。我们提出了JANUS，一个用于自动化跨层防御合成的红蓝对抗框架。从已知攻击出发，JANUS在隔离的离线环境中迭代地细化特定于部署的防御策略。黑盒攻击者仅使用外部可观察的反馈来探测当前策略下的攻击变体。防御者将每个已验证的变体总结为其安全相关的影响，并使用反事实重放来分离（摘要在此处截断）

    arXiv:2610.35532v1 Announce Type: cross  Abstract: Unsafe deserialization turns untrusted data into attacker-controlled execution, and a single vulnerability can admit diverse exploit variants that evade defenses tailored to a known exploit. Existing defenses typically operate at a single enforcement point with limited visibility into the full attack. Combining controls across layers offers complementary protection, but coordinating them to preserve legitimate functionality and adapting them to new variants requires substantial expert effort.   We propose JANUS, a red-blue adversarial framework for automated cross-layer defense synthesis. Starting from a known exploit, \sys iteratively refines deployment-specific policies in an isolated offline environment. A black-box attacker probes the current policy for exploit variants using only externally observable feedback. The defender summarizes each verified variant into its security-relevant effects, uses counterfactual replay to separate 
    
[^7]: 基于大语言模型的排序函数生成用于终止性证明

    Ranking Function Generation with LLMs for Termination

    [https://arxiv.org/abs/2610.35523](https://arxiv.org/abs/2610.35523)

    该论文提出RAINFALL神经符号框架，通过将排序函数合成转化为带槽位约束的代码补全任务，并结合符号验证器的反馈迭代改进，利用大语言模型自动生成可验证的排序函数以证明循环终止性。

    

    排序函数是证明循环终止性的基础技术，但手动构造排序函数具有挑战性且需要大量专业知识。我们提出了RAINFALL，这是一个神经符号框架，遵循生成-验证-改进范式，将大语言模型与符号验证紧密集成。RAINFALL将排序函数合成任务表述为代码补全任务，通过在输入程序中插入占位符，并约束大语言模型为每个循环填充四个预定义的槽位。补全后的程序随后由符号验证器进行检查。如果验证成功，则获得有效的排序函数；否则，验证器的诊断信息会被转换为自然语言反馈，以指导大语言模型输出的迭代改进。我们在两个公开基准测试套件上对RAINFALL进行了评估，共包含358个可终止的C程序。使用MiniMax V2.7，RAINFALL成功为323个程序（90……）合成了可验证的排序函数

    arXiv:2610.35523v1 Announce Type: cross  Abstract: Ranking functions are a fundamental technique for proving loop termination, but constructing them manually is challenging and requires substantial expertise. We propose RAINFALL, a neuro-symbolic framework which follows the generate-verify-refine paradigm and tightly integrates LLMs with symbolic verification. RAINFALL formulates ranking function synthesis as a code-completion task by inserting placeholders into the input program and constraining the LLM to fill four predefined slots for each loop. The completed program is then checked by a symbolic verifier. If verification succeeds, a valid ranking function is obtained; otherwise, the verifier's diagnostics are translated into natural-language feedback that guides iterative refinement of the LLM's output. We evaluate RAINFALL on two public benchmark suites comprising 358 terminating C programs. Using MiniMax V2.7, RAINFALL synthesizes verifiable ranking functions for 323 programs (90
    
[^8]: 大规模企业AI流量的治理：基于Envoy的策略驱动多厂商LLM网关的设计与运营

    Governing Enterprise AI Traffic at Scale: Design and Operation of a Policy-Driven, Multi-Vendor LLM Gateway on Envoy

    [https://arxiv.org/abs/2610.35509](https://arxiv.org/abs/2610.35509)

    PayPal的Cosmos LLM网关基于Envoy，通过将I/O密集型控制决策（进程外服务）与数据面转换（代理内Rust WASM过滤器）按延迟特征分离，实现了对多厂商企业LLM流量的统一策略治理、成本控制与无缝流式转发。

    

    大型组织同时使用来自多个厂商的大语言模型（LLM），而每个厂商都有各自不同的API方言、凭证模型、流式传输协议和定价方式。如果直接集成，这种异构性会导致凭证泛滥、格式锁定、无节制的支出以及不一致的治理。我们介绍了Cosmos LLM网关的设计与生产运营情况，它是PayPal企业LLM流量的受治理统一入口。该网关构建于Envoy之上，按延迟特征划分工作负载：I/O密集型的控制决策（认证、凭证代理、策略与预算评估）在进程外的外部处理服务中运行，而线上格式转换、token计量和提示缓存整形则以Rust WebAssembly过滤器的形式在代理内运行，因此默认情况下，流式响应体分块永远不会因进程外调用而等待。模型、提供商和价格在很大程度上是经过验证的数据而非代码。治理采用声明式策略评估机制。

    arXiv:2610.35509v1 Announce Type: cross  Abstract: Large organizations consume large language models (LLMs) from several vendors at once, each with its own API dialect, credential model, streaming protocol and pricing. Integrated directly, this heterogeneity yields credential sprawl, format lock-in, unbounded spend and inconsistent governance. We present the design and production operation of the Cosmos LLM Gateway, the governed entry point for enterprise LLM traffic at PayPal. Built on Envoy, it splits work by latency profile: I/O-bound control decisions (authentication, credential brokering, policy and budget evaluation) run in an out-of-process external-processing service, while wire-format translation, token metering and prompt-cache shaping run in-proxy as a Rust WebAssembly filter, so that, by default, streamed response body chunks never wait on an out-of-process call. Models, providers and prices are largely validated data rather than code. Governance is declarative policy evalu
    
[^9]: 遗忘的隐性代价：软件工程智能体中轨迹删除的研究

    The Hidden Costs of Forgetting: A Study of Trajectory Deletion in Software Engineering Agents

    [https://arxiv.org/abs/2610.35496](https://arxiv.org/abs/2610.35496)

    通过在SWE-bench Verified上对9,400次运行的大规模实证研究发现，软件工程智能体可以容忍删除大量旧轨迹内容（甚至100%的推理文本）而不影响整体有效性，但删除必须延迟执行，立即删除会显著降低智能体表现。

    

    软件工程智能体在每一轮交互中都会将其完整的轨迹重新发送给大语言模型（LLM），这促使人们研究在线轨迹缩减方法。先前的工作表明，随机删除旧回合中75%的token可能保持智能体的有效性，但其背后的原因、智能体如何补偿缺失的内容，以及这种做法是否真正降低成本仍不清楚。本文对42种轨迹缩减配置进行了实证研究，重点关注有效性、智能体行为和成本。我们在SWE-bench Verified基准上使用两个最先进的智能体，通过改变删除的范围、粒度、数量和时机，在9,400次运行中系统性地探索了从旧回合中删除内容的方式。我们发现，删除大量轨迹内容（例如，高达100%的推理文本）仍可以保持整体有效性。重要的是，这种删除必须延迟应用，因为立即删除往往会显著降低有效性。智能体通常会通过……（原文摘要在此处截断）

    arXiv:2610.35496v1 Announce Type: cross  Abstract: Software engineering agents resend their entire trajectory to an LLM at every turn, motivating online trajectory reduction. Prior work shows that randomly deleting 75% of tokens in old turns may preserve agent effectiveness, but it remains unclear why, how agents compensate for missing content, and whether it actually reduces costs. This paper presents an empirical study of 42 trajectory-reduction configurations, focusing on effectiveness, agent behavior, and cost. We systematically explore deleting content from old turns by varying the scope, granularity, amount, and timing of deletion across 9,400 runs with two state-of-the-art agents on SWE-bench Verified. We find deleting large amounts of trajectory content (e.g., up to 100% of reasoning text) can preserve overall effectiveness. Importantly, such deletion must be applied with a delay, as immediate deletion tends to significantly degrade effectiveness. Agents often compensate for de
    
[^10]: FeatureLoop：在需求不断演化下对编码智能体进行基准测试

    FeatureLoop: Benchmarking Coding Agents under Evolving Requirements

    [https://arxiv.org/abs/2610.35488](https://arxiv.org/abs/2610.35488)

    该论文提出FeatureLoop基准，通过演化反馈与完成反馈模拟真实开发中需求不断演进的过程，对61个任务上的12个前沿模型进行评估，发现最强模型仅完成72.1%的任务，且绝大多数失败阶段难以在两次纠正性修订内恢复。

    

    使用编码智能体进行软件开发通常始于一个粗略的请求，用户在尝试实现后会对其进行逐步完善。仅凭最终结果的正确性无法体现这一过程中所需的反复引导。我们提出了FeatureLoop，这是一个包含61个经人工审核任务的基准测试，任务来自34个代码仓库，涵盖五种主要编程语言和七种软件形态。每个任务在持久化工作区中围绕一个开发目标展开，通过演化反馈（增加或澄清需求）和完成反馈（报告已声明需求的失败情况）来推进。累积的可执行检查用于追踪新行为的实现、问题的修复以及早期功能的保留。我们评估了12个前沿模型，发现表现最强的模型在同时提供两类反馈的情况下完成了72.1%的任务。然而，纵观所有模型，观测到的失败阶段中仅有15.4%能在两次纠正性修订内恢复。进一步的分析揭示了修复不完整、功能回退以及模型特有的……（原文摘要在此处截断）

    arXiv:2610.35488v1 Announce Type: new  Abstract: Software development with coding agents often begins with a rough request that users refine after trying the implementation. Final correctness alone does not capture the repeated guidance this process requires. We introduce FeatureLoop, a benchmark of 61 human-reviewed tasks from 34 repositories across five primary programming languages and seven software forms. Each task follows one development goal in a persistent workspace through evolution feedback, which adds or clarifies requirements, and completion feedback, which reports failures of stated requirements. Cumulative executable checks track new behavior, repair, and preservation of earlier functionality. We evaluate 12 frontier models and find that the strongest completes 72.1% of tasks with both feedback types. Across models, however, only 15.4% of observed failed stages recover within two corrective revisions. Further analysis reveals incomplete repairs, regressions, and model-spe
    
[^11]: 通过全面且系统的实证研究理解Python代码检查工具

    Understanding Python Linters through a Comprehensive and Systematic Empirical Study

    [https://arxiv.org/abs/2610.35485](https://arxiv.org/abs/2610.35485)

    本文首次对Ruff、Pylint、Flake8、Mypy和Black五个主流Python代码检查工具开展了大规模系统实证研究，从检查规则、速度、配置方法和工具偏好四个维度揭示了它们的共性、差异与特点。

    

    代码检查工具是能够自动检测代码问题的工具，通过尽早发现错误来帮助开发者加速开发并提升代码质量。与以往主要关注单一检查工具或非正式比较的研究不同，本文首次对五个广泛使用的Python检查工具：Ruff、Pylint、Flake8、Mypy和Black进行了大规模实证研究，研究围绕四个关键的用户关注点展开：检查规则、速度、配置方法以及检查工具偏好。我们分析了1,551条检查规则，将其归类为五种类型，并构建了规则映射关系，发现超过40%的规则被多个检查工具共享，其中Ruff和Pylint共同覆盖了超过80%的所有规则。Pylint在前两个规则类别中拥有最多的规则，而Ruff在后三个类别中处于领先地位。接着，我们在三种设置（单规则、默认、全部规则）下测量了这些工具在1,000个GitHub项目上的运行时间，发现其速度排名保持一致……

    arXiv:2610.35485v1 Announce Type: new  Abstract: Code linters, tools that automatically detect code issues, help developers accelerate development and improve code quality by identifying errors early. Unlike previous studies that focused mainly on single linters or informal comparisons, this paper presents the first large-scale empirical study of five widely used Python linters: Ruff, Pylint, Flake8, Mypy and Black, through four key user concerns: linting rule checks, speed, configuration methods, and linter preferences. We analyze 1,551 linting rules, categorize them into five types, and construct rule mappings, finding that more than 40% are shared between multiple linters, with Ruff and Pylint together covering over 80% of all rules. Pylint has the most rules in the first two categories, and Ruff leads in the last three. Next, we measure their running times on 1,000 GitHub projects under three setups (single-rule, default, all-rules), finding a consistent speed ranking from fastest 
    
[^12]: 通过神经符号微调使小型代码大语言模型在安全代码审查中更高效

    Making Small Code LLMs More Efficient for Secure Code Review with Neuro-Symbolic Fine-Tuning

    [https://arxiv.org/abs/2610.35477](https://arxiv.org/abs/2610.35477)

    该论文提出secSemScan系统，通过神经符号微调将静态分析器与本地部署的小型代码模型结合，使其能生成可验证的、模式约束的安全审查解释，并构建了包含1,993个C/C++函数、覆盖22类CWE的OGSM数据集。

    

    随着AI编程助手生成越来越多的代码，对其安全缺陷进行审查的需求也不断增长。现有方法存在空白：静态分析器能够识别程序位置但几乎不提供解释，而语言模型生成的解释则难以验证。小型开源权重代码模型可以在不将专有代码发送到托管服务的情况下实现安全审查。我们研究微调是否能帮助这些模型将安全证据转化为开发者和工具可以核查的解释。我们提出了secSemScan，它将静态分析器与一个本地可部署的模型配对，该模型生成受模式约束的安全独白，指明程序位置、分配CWE编号，并给出机器可解析的判定结果。我们还介绍了OGSM（基于Oracle的安全独白）数据集，包含1,993个C/C++函数，涵盖污点类和生命周期类的22个CWE。每条记录将一个函数与来自分析器或修复差异的安全证据配对……

    arXiv:2610.35477v1 Announce Type: new  Abstract: As AI coding assistants generate more code, the need to review it for security flaws grows. Existing approaches leave a gap: static analyzers identify program sites but offer little explanation, while language-model explanations can be difficult to verify. Small open-weight code models could make security review available without sending proprietary code to a hosted service. We study whether fine-tuning helps these models turn security evidence into explanations that developers and tools can check. We present secSemScan, which pairs a static analyzer with a locally deployable model that writes a schema-constrained security monologue naming program sites, assigning a CWE, and giving a machine-parseable verdict. We also introduce OGSM (Oracle-Grounded Security Monologue), a dataset of 1,993 C/C\texttt{++} functions covering 22 CWEs in the taint and lifetime families. Each record pairs a function with analyzer-derived or fix-diff-derived ev
    
[^13]: 《然后是提示注入、一封恶意邮件……：OpenClaw 智能体系统安全的蛮荒西部》

    Then there is prompt injection, a malicious email, ...: The Wild Wide West of OpenClaw Agentic System Security

    [https://arxiv.org/abs/2610.35451](https://arxiv.org/abs/2610.35451)

    本文通过对 Reddit 讨论与安全相关 GitHub issue 的大规模实证分析，系统刻画了 OpenClaw 智能体框架的安全现状，识别出 59 种弱点类型下的 521 个 CWE 实例，揭示了提示注入等风险在缺乏明确信任边界时对 LLM 智能体系统的普遍威胁。

    

    诸如 OpenClaw 这类由 AI 驱动的个人助理框架结合了大语言模型（LLM）推理、持久化记忆、工具调用、本地执行以及基于网页的工作流。这些能力带来了有用的自动化，但当不可信输入、模型推理、工具、记忆与部署配置在没有明确信任边界的情况下相互交互时，也会产生安全风险。本文利用两个互补的数据来源——Reddit 讨论和带有安全标签的 GitHub issue——对 OpenClaw 中被报告的安全问题进行了实证研究。我们分析了来自 13 个 OpenClaw 相关 subreddit 的超过 2.5 万条 Reddit 帖子，识别出 162 条与安全相关的帖子及 1,787 条关联评论；并进一步收集了 358 个带安全标签的 GitHub issue，保留了其中 180 个描述具体安全问题的 issue。通过定性分析和基于证据的 CWE 映射，我们共识别出 59 种不同弱点类型下的 521 个 CWE 实例，主要集中在访问控制（原文此处被截断）。

    arXiv:2610.35451v1 Announce Type: cross  Abstract: AI-powered personal assistant frameworks such as OpenClaw combine LLM reasoning, persistent memory, tool invocation, local execution, and web-based workflows. These capabilities enable useful automation, but they also create security risks when untrusted inputs, model reasoning, tools, memory, and deployment configuration interact without explicit trust boundaries. This paper presents an empirical study of reported security concerns in OpenClaw using two complementary data sources: Reddit discussions and security-tagged GitHub issues. We analyze more than 25K Reddit posts from 13 OpenClaw-related subreddits and identify 162 security-relevant posts with 1,787 associated comments. We further collect 358 security-tagged GitHub issues and retain 180 issues describing concrete security concerns. Using qualitative analysis and evidence-grounded CWE mapping, we identify 521 CWE instances across 59 distinct weakness types, concentrated in acce
    
[^14]: 迈向基于证据的智能体软件工程

    Toward Evidence-Based Agentic Software Engineering

    [https://arxiv.org/abs/2610.35348](https://arxiv.org/abs/2610.35348)

    本研究通过标注146篇论文中的325个实验，揭示智能体软件工程研究普遍缺乏对运行框架版本及配置的充分报告、且很少将运行框架影响与模型影响分离，并据此提出了参考运行框架的需求。

    

    arXiv:2610.35348v1 公告类型：新 摘要：智能体编码工具的任务成功率和执行成本同时取决于模型和运行框架。运行框架负责管理上下文、提供工具并运行智能体循环。在实证软件工程（SE）研究中，同时改变模型和运行框架的比较无法分离两者的各自影响。我们对2025年至2026年间发表的146篇同行评审全文论文中报告的325个实验进行了定性标注。我们考察了所报告的配置，以及这些比较是否分离了运行框架的影响，包括运行框架的配置与版本、模型、任务和用户交互。结果发现：仅有15个实验报告了所用运行框架的确切版本；只有5篇论文报告了实验运行的时间；31篇论文的作者分享了每个会话的执行轨迹；在比较运行框架的226个实验中，有35个从未单独改变运行框架这一变量。基于这一分析，我们推导出了一个参考运行框架所应满足的需求……

    arXiv:2610.35348v1 Announce Type: new  Abstract: Task success rates and execution costs of agentic coding tools depend on both model and harness. The harness manages context, provides tools, and runs the agent loop. In empirical software engineering (SE) studies, comparisons that change model and harness together cannot isolate their effects. We qualitatively annotated 325 experiments reported in 146 peer-reviewed full papers published between 2025 and 2026. We examined the reported configurations and whether comparisons isolated harness effects, including harness configuration and version, model, tasks, and user interaction. The exact version of every harness used was reported for only 15 experiments. For five, the paper reported when the experiments were run, and for 31, authors shared execution traces for every session. Of the 226 experiments comparing harnesses, 35 never varied the harness in isolation. Based on this analysis, we derive requirements for a reference harness that the
    
[^15]: RepoTrace：改进多轮智能体交互中的需求追踪

    RepoTrace: Improving Requirement Tracing in Multi-Round Agent Interactions

    [https://arxiv.org/abs/2610.35333](https://arxiv.org/abs/2610.35333)

    提出 RepoTrace，通过将仓库结构与需求语义结合构建需求-代码对应关系并注入智能体上下文，解决了代码智能体在多轮交互中因需求焦点错位导致的性能下降问题。

    

    代码智能体通过与软件环境交互执行软件工程任务的能力日益增强。然而，随着交互轮数的增加，智能体的表现往往会下降，这使得在迭代修改过程中维护现有软件变得困难。我们分析了 EvoCode-Bench 中的 2,656 条智能体轨迹，发现“需求焦点错位”是一种反复出现的失败模式，即智能体难以区分需求的主要修改目标与无关代码。我们提出了 RepoTrace，一种需求驱动的仓库理解工具框架，它将仓库结构与需求语义相结合，构建需求-代码对应关系，并将其注入智能体的上下文中。这种对应关系使代码智能体能够遵循仓库架构、在多轮交互中追踪需求，并识别与当前修改相对应的核心代码元素。

    arXiv:2610.35333v1 Announce Type: cross  Abstract: Code agents have become increasingly capable of performing software engineering tasks through interaction with software environments. However, agents tend to degrade as the number of interaction rounds increases, making it difficult to maintain existing software during iterative modification. We analyze 2,656 agent trajectories in EvoCode-Bench and identify requirement focus misalignment as a recurring failure pattern, where agents struggle to distinguish the primary modification targets of a requirement from unrelated code.   We propose Repotrace a requirement-driven repository understanding harness that combines repository structure with requirement semantics to construct a requirement-code correspondence and injects it into the agent context. This correspondence enables code agents to follow the repository architecture, trace requirements across multi-round interactions, and identify the core code elements corresponding to the curre
    
[^16]: 使用编码智能体进行测试：智能体支持的与人工编写的拉取请求的大规模比较

    Testing with Coding Agents: A Large-Scale Comparison of Agent-Supported and Human-Authored Pull Requests

    [https://arxiv.org/abs/2610.35297](https://arxiv.org/abs/2610.35297)

    该研究通过对391个Java仓库中114,134个拉取请求的大规模实证分析，首次系统比较了编码智能体支持的与人工编写的测试变更在频率、范围、类型、结构、坏味道和覆盖率五个维度上的差异，为理解智能体时代测试工作的分工及其对长期可维护性的影响提供了重要依据。

    

    背景：编码智能体越来越多地通过拉取请求贡献测试代码，而这些代码随后由开发者维护，因此测试工作的分工方式会影响软件的长期可维护性。目标：我们从五个维度比较开发者与编码智能体协作的拉取请求（智能体支持型）与无可检测智能体参与的拉取请求（人工编写型）：测试变更的频率与范围、变更的类型与上下文、测试结构、测试坏味道以及代码覆盖率。方法：我们分析了391个Java仓库中2025年1月至2026年4月间合并的114,134个拉取请求，并通过拉取请求级别和提交级别的信号识别出五种编码智能体。由于智能体的采用率在不同项目之间存在差异，我们在仓库内部采用控制了变更规模的混合效应模型对两组进行比较。结果：跨仓库汇总来看，两组变更测试代码的频率几乎相同；在仓库内部，其几率（摘要在此处被截断）

    arXiv:2610.35297v1 Announce Type: new  Abstract: Context. Coding agents increasingly contribute tests through pull requests that developers then maintain, so how testing work is divided affects long-term maintainability.   Objective. We compare pull requests in which developers and coding agents collaborate (agent-supported) with pull requests without detectable agent involvement (human-authored) along five dimensions: the frequency and extent of test changes, their type and context, test structure, test smells, and code coverage.   Method. We analyze 114,134 pull requests merged between January 2025 and April 2026 in 391 Java repositories, identifying five coding agents through pull-request-level and commit-level signals. Since agent adoption varies across projects, we compare the two groups within repositories with mixed-effects models that control for change size.   Results. Pooled across repositories, both groups change test code almost equally often; within repositories, the odds 
    
[^17]: 绘制不可见之物：面向GUI智能体隐藏目标探索的交互式感知

    Chart the Invisible: Interactive Perception for Hidden-Target Exploration in GUI Agents

    [https://arxiv.org/abs/2610.35237](https://arxiv.org/abs/2610.35237)

    该论文提出了首个用于可滚动区域感知的多模态模型，使GUI智能体能够在动作选择前探索隐藏区域，从而解决智能体在面对视口外隐藏目标时难以决定何时、何地及如何滑动的关键难题。

    

    尽管在命令行界面（CLI）任务方面取得了快速进展，但近期基于大语言模型的智能体在通过图形用户界面（GUI）可靠执行任务方面仍然存在困难。一个关键挑战是有限视口内可见内容的局限性，尤其是在移动设备上，许多与任务相关的目标处于隐藏状态，只能通过滑动交互才能显现。因此，尽管大多数GUI智能体支持将滑动作为动作原语，但由于缺乏目标可见性，它们难以决定何时、何地以及如何滑动以显示隐藏内容。我们的定量研究进一步表明，在需要滑动探索的情形中，GUI智能体在47.8%的情况下错误地选择了点击操作。为弥合这一差距，本文提出为GUI智能体配备可滚动区域感知能力，以支持在动作选择之前进行隐藏区域探索。为此，我们训练了首个用于可滚动区域感知的多模态模型……

    arXiv:2610.35237v1 Announce Type: new  Abstract: Despite rapid progress in command-line interface (CLI) tasks, recent LLM-based agents still struggle with reliable task execution through graphical user interfaces (GUIs). A key challenge is the limited content visible within a bounded viewport, especially on mobile devices, where many task-relevant targets remain hidden and can only be revealed through swipe interactions. Therefore, although most GUI agents support swipe as an action primitive, the lack of target visibility makes it difficult for them to decide when, where, and how to swipe to reveal the hidden content. Our quantitative study further shows that GUI agents incorrectly choose a click action in 47.8% of cases that require swipe exploration. To bridge this gap, this paper proposes equipping GUI agents with scrollable-region perception to support hidden-area exploration before action selection. To this end, we trained the first multimodal model for scrollable-region percepti
    
[^18]: 基于结构化预言的拉取请求行为回归检测

    Behavioral Regression Detection for Pull Requests via Structured Oracles

    [https://arxiv.org/abs/2610.35223](https://arxiv.org/abs/2610.35223)

    ORION通过从PR意图合成结构化行为预言并引导可执行测试的生成与差分执行，实现了拉取请求级别的行为回归检测，同时发布了首个Python PR回归基准RegBench-Py。

    

    软件回归是指代码修改所引入的非预期行为变化，这类问题在拉取请求（PR）级别难以检测。现有方法依赖于不完整的测试，或者直接通过大语言模型（LLM）分析PR，缺乏结构化的行为推理和显式的预言。我们提出了ORION，一个用于PR级别回归检测的分阶段框架。ORION首先提取PR意图，并通过基于抽象语法树（AST）的影响分析识别受影响的函数，然后综合生成函数级别的结构化行为预言，这些预言从PR意图、静态风险信号和代码变更中编码了保留/变更的行为、设置约束和控制流提示。在这些预言的指导下，ORION生成可执行的测试，进行预言覆盖率引导的测试增强与静态修复，并在变更前后的代码上差异化地运行测试。它采用确定性规则与受约束的LLM推理相结合的方式对PR进行分类。我们构建了RegBench-Py，这是首个Python PR回归基准（测试集）。

    arXiv:2610.35223v1 Announce Type: new  Abstract: Software regressions are unintended behavioral changes introduced by code modifications and are hard to detect at pull-request (PR) level. Existing methods rely on incomplete tests or directly analyze PRs via LLMs, lacking structured behavioral reasoning and explicit oracles. We present ORION, a staged framework for PR-level regression detection. ORION extracts PR intent and identifies impacted functions via AST-based impact analysis, then synthesizes function-level structured behavioral oracles encoding preserved/changed behaviors, setup constraints and control-flow hints from PR intent, static risk signals and code changes. Guided by these oracles, ORION generates executable tests, conducts oracle-coverage-guided augmentation and static repair, and differentially runs tests on pre- and post-change code. It classifies PRs with deterministic rules and constrained LLM reasoning. We build RegBench-Py, the first Python PR regression benchma
    
[^19]: V³AST-Bench：面向代码智能体的可验证、有效且多样化的安全评估

    V$^3$AST-Bench: Verifiable, Valid, and Versatile Safety Evaluation for Coding Agents

    [https://arxiv.org/abs/2610.35188](https://arxiv.org/abs/2610.35188)

    提出V³AST-Bench基准，通过可执行检查验证代码智能体造成的实际危害与任务结果，利用成对孪生任务区分安全性与能力短板，并以212个跨18类风险的任务实现对智能体安全的全面评估。

    

    大型语言模型越来越多地通过代码来执行操作：修改代码仓库、运行程序，并以智能体的形式与服务进行交互。这些行为带来的风险无法仅凭模型生成的回复来评估。安全评估必须能够验证实际后果、区分“避免伤害”与“无能力完成任务”，并覆盖真实的工作流程。我们提出了V³AST-Bench，一个针对智能体安全性实现可验证结果、有效解释与多样化风险覆盖的基准。它包含212个风险任务，涵盖18个类别。专门的运行时环境通过对状态、记录和产物行为的可执行检查来验证危害与任务结果，同时通过成对的能力孪生任务来验证工作流程的胜任能力。我们的实验涵盖11个模型与五种测试框架的51种组合，共计86,496次试验和超过90亿token的模型交互。结果表明，即使是前沿模型，V³AST-Bench依然具有挑战性。

    arXiv:2610.35188v1 Announce Type: cross  Abstract: Large language models increasingly act through code, modifying repositories, executing programs, and interacting with services in agent form. Their actions introduce risks that cannot be assessed from generated responses alone. Safety evaluation must verify actual consequences, distinguish harm avoidance from task incapability, and cover realistic workflows. We introduce V$^3$AST-Bench, a benchmark for Verifiable results, Valid interpretation, and Versatile risk coverage of Agent Safety. It comprises 212 risk tasks spanning 18 categories. Dedicated runtime environments verify harm and task outcomes through executable checks over states, records, and artifact behavior, while paired capability twin tasks verify workflow competence. Our experiments span 51 combinations of 11 models and five harnesses, spanning 86,496 trials and over 9 billion tokens of model interaction. Results show that V$^3$AST-Bench remains challenging even for fronti
    
[^20]: 面向无服务器函数供应与负载均衡的优先级感知调度器

    Priority-Aware Scheduler for Serverless Functions Provision and Load Balancing

    [https://arxiv.org/abs/2610.35073](https://arxiv.org/abs/2610.35073)

    本文提出SPES-Ext，一种面向内存受限分布式工作节点的基于规则的优先级感知调度器，通过利用无服务器系统集中的调用模式进行行为预测和函数分类，优化冷启动、函数驱逐和负载不均衡问题，在启动延迟与资源效率之间实现权衡。

    

    无服务器计算因其高效利用按需资源而成为关键的云部署范式。然而，持续存在的冷启动问题仍然阻碍着系统性能。当前策略通常采用过于简化的预加载/卸载机制，未能充分利用调用模式，无法优化启动延迟与资源效率之间的关键权衡。此外，现有方法忽略了内存受限的多工作节点环境的实际约束，加剧了函数驱逐和负载不均衡问题。为解决这些局限性，我们提出了SPES-Ext，一个面向内存受限分布式工作节点的基于规则的优先级感知调度器。我们的方法源于两个关键观察：（1）无服务器系统表现出集中的调用模式，使行为具有可预测性；（2）这些模式允许对函数进行分类

    arXiv:2610.35073v1 Announce Type: cross  Abstract: Serverless computing has emerged as a pivotal cloud deployment paradigm due to its efficient utilization of on-demand resources. However, the persistent cold start problem continues to hinder system performance. Current strategies often employ oversimplified preloading/unloading mechanisms that inadequately exploit invocation patterns, failing to optimize the critical trade-off between startup latency and resource efficiency. Furthermore, existing approaches disregard practical constraints of memory-limited multi-worker environments, exacerbating function eviction and load imbalance issues. To address these limitations, we present SPES-Ext, a rule-based priority-aware scheduler designed for memory-constrained distributed worker nodes. Our methodology stems from two key observations: (1) serverless systems exhibit concentrated invocation patterns enabling behavioral predictability, and (2) these patterns permit function categorization f
    
[^21]: 面向自主与半自主系统工程中通过批判性考察异构系统以发现隐藏依赖关系的方法与工具

    Toward Methods and Tools for Critical Examination of Disparate Systems for Discovery of Hidden Dependencies in Engineering of Autonomous and Semi-Autonomous Systems

    [https://arxiv.org/abs/2610.34974](https://arxiv.org/abs/2610.34974)

    本文提出一种通过对相关环境中异构系统视频进行主动对抗性批判分析的方法论和AI工具，用于发现隐藏的依赖关系和环境假设，帮助软件工程师在陌生领域开发自主/半自主系统时获取领域知识并提升系统质量与工程实践。

    

    我们提出了一种方法论、一个初步的AI技能包以及初步成果，旨在增强软件工程师在开发其可能较为陌生的领域中的自主或半自主系统时，对领域知识的获取与应用能力以及系统级工程视角。该方法的核心是一个对相关环境中异构系统视频进行主动对抗性批判分析的过程。工程师和自动化AI工具在观察到的系统中寻找交互、依赖关系和环境假设，这些内容同样适用于目标系统，否则可能会在项目工件中一直未被明确表述。该方法及其支持工具能够提升目标系统的质量、工程师个人的技能以及组织的工程实践。

    arXiv:2610.34974v1 Announce Type: new  Abstract: We present a methodology, an initial AI Skill package, and emerging results, for enhancing software engineers' acquisition and application of domain knowledge and system-wide engineering perspectives towards the development of an autonomous or semi-autonomous system in a domain that may be new to them. At the core of the approach is a process involving proactive adversarial critiquing of videos of disparate systems in related environments. Among other things, the engineers and automated AI tools look for interactions, dependencies, and environment assumptions manifested in the observed systems, which are applicable also to the target system, but that otherwise might remain unarticulated in project artifacts. The approach and its supporting tools can enhance the quality of the target system, the individual engineer's skills and the organization's engineering practices.
    
[^22]: Rivet：面向仓库级编码智能体的效用保持式上下文控制

    Rivet: Utility-Preserving Context Control for Repository-Level Coding Agents

    [https://arxiv.org/abs/2610.34967](https://arxiv.org/abs/2610.34967)

    提出 Rivet 框架，通过直接保留目标位置、辅助定义、调用方用法和配置约束等仓库证据，解决现有上下文选择方法中“看似相关却缺失关键证据”的上下文效用稀释问题，从而提升仓库级编码智能体的代码生成效果。

    

    仓库级代码生成要求智能体在现有仓库中实现某项任务需求。为了产出能在仓库中正常运行的代码，智能体需要仓库证据，例如目标位置、辅助函数定义、调用方用法和配置约束，这些证据决定了代码应如何定位、集成和执行。由于上下文预算无法容纳整个仓库，现有的检索、压缩和结构感知方法依据任务相似度、上下文长度或结构邻近性来选择上下文，但这些标准并不能直接衡量所选上下文是否包含上述证据。因此，所选上下文可能看起来与任务相关，却遗漏了实现所依赖的关键证据。我们将这种失败模式称为“上下文效用稀释”。在本文中，我们提出了 Rivet，一个在（摘要截断处）为编码智能体保持仓库证据的上下文控制框架。

    arXiv:2610.34967v1 Announce Type: new  Abstract: Repository-level code generation asks an agent to implement a task requirement inside an existing repository. To produce code that works in the repository, the agent needs repository evidence, such as target locations, helper definitions, caller usages, and configuration constraints, which determine how the code should be located, integrated, and executed. Because the context budget cannot hold the entire repository, existing retrieval, compression, and structure-aware methods select context by task similarity, context length, or structural proximity, but these criteria do not directly measure whether the selected context contains this evidence. As a result, the selected context can appear relevant to the task yet miss the evidence that the implementation depends on. We call this failure mode context utility dilution. In this paper, we propose Rivet, a context control framework that preserves repository evidence for coding agents under a
    
[^23]: 决策模型工厂：构建专业化主权决策模型

    Decision Model Factory: Building Specialized Sovereign Decision Models

    [https://arxiv.org/abs/2610.34956](https://arxiv.org/abs/2610.34956)

    提出决策模型工厂，可按需将骨干模型转化为专业化、主权可控的决策模型，配合专用运行器在保持决策分数基本不变的情况下将服务速度提升一倍以上。

    

    决策需求因应用、语言和部署环境而各不相同。决策模型工厂（Decision Model Factory）能够按需将这些需求转化为专业化、主权可控的模型。一套简明的配方指导对选定骨干模型进行训练、校准、验证和评估，从而在操作者的掌控下交付一个可部署的决策模型。当需求发生变化时，同一工作流程即可创建新的专用模型。在日语决策任务、紧凑部署和长文本段落排序等场景中，五个由工厂构建的模型均优于其骨干模型；其中三个模型在固定的公开日语基准测试比较中超越了最强结果。一个专用的运行器仅计算允许答案的概率并复用共享上下文，在匹配的多问题工作负载上将服务速度提升一倍以上，而决策分数基本保持不变。总之，该工厂与运行器相结合，使快速构建和部署特定任务的决策模型成为现实。

    arXiv:2610.34956v1 Announce Type: new  Abstract: Decision needs vary with each application, language, and deployment environment. The Decision Model Factory turns those needs into specialized, sovereign models on demand. A brief recipe guides the training, calibration, verification, and evaluation of a selected backbone, delivering a deployable decision model under the operator's control. The same workflow creates a new specialist whenever the requirements change. Across Japanese decision tasks, compact deployment, and long-context passage ranking, five factory-built models improve upon their backbones; three surpass the strongest results in a fixed public Japanese benchmark comparison. A dedicated runner computes only permitted-answer probabilities and reuses shared context, more than doubling serving speed on matched multi-question workloads with essentially unchanged decision scores. Together, the factory and runner make fast, task-specific decision models practical to build and dep
    
[^24]: 高效AI编码的推理模式：一项实证研究

    Reasoning Patterns for Efficient AI Coding: An Empirical Study

    [https://arxiv.org/abs/2610.34943](https://arxiv.org/abs/2610.34943)

    该研究通过将四个编码任务的推理轨迹转化为结构化动作序列进行实证分析，识别出代码生成、执行推理、调试和翻译中的四种通用推理模式及其正负面模式，为系统性刻画和推理时控制大型推理模型提供了依据。

    

    arXiv:2610.34943v1 公告类型：交叉 摘要：大型推理模型（LRMs）会生成长篇推理轨迹，揭示其中间推理行为，为研究和控制AI编码中的推理提供了机会。然而，这些轨迹中的底层模式及其在分析和引导编码模型方面的潜力仍知之甚少，限制了对LRM推理的系统性刻画和推理时控制。我们对四个代表性编码任务中的推理模式进行了实证研究。我们将推理轨迹转换为结构化的动作序列，并利用任务内支持、跨任务对比以及有效-无效支持对比来挖掘反复出现的片段。我们刻画了编码智能体的四种通用推理模式：代码生成中的“设计到实现”、执行推理中的“状态追踪”、调试中的“证据-修复-复查”以及翻译中的“语义映射”。我们进一步识别了与编码性能相关的正面和负面推理模式

    arXiv:2610.34943v1 Announce Type: cross  Abstract: Large Reasoning Models (LRMs) generate long reasoning traces that expose intermediate inference behavior, offering opportunities to study and control reasoning in AI coding. However, the underlying patterns in these traces and their potential for analyzing and guiding coding models remain poorly understood, limiting systematic characterization and inference-time control of LRM reasoning. We conduct an empirical study of reasoning patterns across four representative coding tasks. We transform reasoning traces into structured action sequences and mine recurring fragments using within-task support, cross-task contrasts, and valid--invalid support contrasts. We characterize four general reasoning patterns by coding agents: design-to-implementation for code generation, state tracing for execution reasoning, evidence--fix--recheck for debugging, and semantic mapping for translation. We further identify positive and negative patterns associat
    
[^25]: 编码智能体运行框架如何自我进化？一项关于回溯性适应的实证研究

    How Do Coding-Agent Harnesses Self-Evolve? An Empirical Study of Retrospective Adaptation

    [https://arxiv.org/abs/2610.34938](https://arxiv.org/abs/2610.34938)

    该论文首次实证研究了编码智能体通过回顾已完成任务会话来提出并实施运行框架改进的“回溯性适应”机制，发现尽管大多数建议源于具体会话事件，但在七项有效评估中仅有两项更改真正消除了后续任务中的目标问题，表明自我进化的实际收益仍然有限。

    

    编码智能体可以根据早期任务的经验调整其运行框架（即用于运行它们的软件、配置和指令），但这些调整能否改善其在后续任务中的表现仍不清楚。我们研究了 Retro，这是一项要求智能体回顾已完成的会话并提出框架更改建议的技能，并探究它会提出哪些更改，以及实施这些更改是否能减少后续任务中的重复工作。我们在 30 个 SWE-bench Verified 问题上运行了 Claude Code，对 Retro 提出的 125 条建议进行了定性编码，实施了其中 24 条，并在来自相同代码仓库的后续问题上，分别在有和无每项更改的全新会话中评估了 12 项框架更改。大多数建议都与会话中的某个事件相关，但在 43 条与 lint 相关的建议中，有 18 条无论实际发生了什么都推荐某种做法。在未应用更改的会话确实遇到目标问题的七项评估中，有两项更改消除了该重复问题（原文摘要至此截断）。

    arXiv:2610.34938v1 Announce Type: cross  Abstract: Coding agents can adapt their harnesses, the software, configuration, and instructions used to run them, based on experience from earlier tasks, but whether these adaptations improve their behavior on later tasks remains unclear. We study \retro, a skill that asks an agent to review a completed session and propose harness changes, and ask what changes it proposes and whether implemented changes reduce repeated work on later tasks. We ran Claude Code on 30 SWE-bench Verified issues, qualitatively coded the 125 proposals from \retro, implemented 24, and evaluated 12 harness changes on later issues from the same repositories in fresh sessions with and without each change. Most proposals referred to an event in the session, but 18 of 43 lint-related proposals recommended a practice regardless of what had happened. Among the seven evaluations in which sessions without the change encountered the targeted problem, two changes eliminated the t
    
[^26]: 什么将取代拉取请求？重新思考面向AI智能体的软件协作

    What Will Replace Pull Requests? Rethinking Software Collaboration for AI Agents

    [https://arxiv.org/abs/2610.34933](https://arxiv.org/abs/2610.34933)

    该研究通过分析31个组织或个人的50份灰色文献，系统调查了AI编码智能体时代从业者提出的拉取请求替代机制，发现大多数方案仍以某种形式保留拉取请求，而非完全取代它。

    

    人工智能编码智能体的兴起使拉取请求面临压力：智能体每月创建数百万个拉取请求，而一些开源项目已开始限制或禁用外部贡献。我们调查了从业者提出或采用了哪些拉取请求的替代方案，以及这些变化对实证软件工程研究意味着什么。我们审查了2025年10月至2026年10月间发布的来自31个组织或个人的50份灰色文献文档，对每份文档指出的 问题、其提出或采用的机制、每种机制接管了拉取请求五项功能（集成、审查、决策理由、问责和外部贡献）中的哪些，以及所提供的证据进行了编码。大多数应对措施以某种形式保留了拉取请求，无论是通过添加工具、附加新的工件，还是限制访问权限，而限制访问是最常被报告为一种临时的应对…

    arXiv:2610.34933v1 Announce Type: new  Abstract: The emergence of artificial intelligence (AI) coding agents has put pull requests under pressure: agents open millions of pull requests per month, while some open-source projects have begun to restrict or disable external contributions. We investigate which alternatives to pull requests practitioners are proposing or adopting and what these changes mean for empirical software engineering research. We reviewed 50 gray-literature documents from 31 organizations or individuals published between October 2025 and October 2026, coding the problems each named, the mechanisms it proposed or adopted, which of five roles of pull requests (integration, review, rationale, accountability, and external contribution) each mechanism takes over, and the evidence offered. Most responses retained pull requests in some form, whether by adding tooling, attaching a new artifact, or restricting access, and restriction was the response most often reported as ad
    
[^27]: SemWeaver：利用分析器内部证据精炼大语言模型生成的静态漏洞检查器

    SemWeaver: Refining LLM-Generated Static Vulnerability Checkers with Analyzer-Internal Evidence

    [https://arxiv.org/abs/2610.34928](https://arxiv.org/abs/2610.34928)

    SemWeaver 提出了一个插件式精炼框架，通过提取静态分析器内部已有的证据（如控制流图、符号值、内存区域和检查器回调），并将其与源代码上下文和验证反馈组装成带来源标注的结构化信息，迭代修正 LLM 生成的静态漏洞检查器，使其从仅匹配补丁的表面修改提升为捕捉更深层的语义条件。

    

    最近的研究表明，大语言模型（LLM）能够从安全补丁中合成静态分析检查器，但一个可编译的检查器并不一定是正确的检查器。补丁只揭示了“改了什么”，而没有说明“为什么这样改是安全的”，因此生成的检查器往往只匹配表面的代码修改，而遗漏更深层的语义条件。现有的精炼迭代循环将编译错误和验证报告反馈给模型，然而这些信号只描述了症状，却无法解释检查器对代码的哪方面理解出现了偏差。我们观察到，分析框架本身其实已经计算出了弥合这一差距所需的信息：控制流图、符号值、内存区域和检查器回调等证据在分析器内部已经可用，只是目前的精炼策略尚未加以利用。SemWeaver 是一个插件式的精炼层，它提取与补丁相关的分析器证据，并将其与源代码上下文和验证反馈组装成结构化的、带有来源标注的信息包。一个迭代……

    arXiv:2610.34928v1 Announce Type: new  Abstract: Recent work shows that LLMs can synthesize static-analysis checkers from security patches, but a compilable checker is not necessarily a correct one. A patch reveals what changed, not why the change is safe, so generated checkers often match only the surface edit and miss the deeper semantic condition. Existing refinement loops feed back compilation errors and validation reports, yet these signals describe symptoms without explaining what the checker misunderstands about the code. We observe that the analysis framework itself already computes the information needed to close this gap. Control-flow graphs, symbolic values, memory regions and checker callbacks are available inside the analyzer but unused by current refinement strategies. SemWeaver is a plug-in refinement layer that extracts patch-relevant analyzer evidence and assembles it with source context and validation feedback into a structured, provenance-labelled bundle. An iterativ
    
[^28]: AIPC：面向PyTorch项目部署的证据引导式智能体工作流

    AIPC: Evidence-Guided Agent Workflows for PyTorch Project Deployment

    [https://arxiv.org/abs/2610.34890](https://arxiv.org/abs/2610.34890)

    提出AIPC方法，通过将结构化技能与工具链、运行时和性能分析证据相结合，引导智能体进行跨层诊断与修复，实现PyTorch推理项目在专用加速器（尤其是端侧NPU）上的自动化部署与移植。

    

    arXiv:2610.34890v1 公告类型：cross 摘要：在专用加速器上部署和优化PyTorch推理项目，尤其是端侧NPU，并非仅使用类似CUDA的框架设备API就能完成。项目必须被转换为硬件运行时工件，并适配目标工具链和应用程序接口。模型格式、量化流程、算子支持和工具链接口方面的差异会产生跨越PyTorch/ONNX模型、工具链、运行时和设备执行的跨层问题。LLM智能体可以调用工具来解决此类问题，但由于领域知识不足、配置和修复空间庞大或路径选择不佳，可能会出现偏差。我们提出AIPC（AI Porting & Conversion），一种用于部署PyTorch推理项目的智能体方法。AIPC将结构化技能与工具链、运行时和性能分析证据相结合，以驱动跨层诊断和修复，组织模型适配、工件构建、端上部署等环节。

    arXiv:2610.34890v1 Announce Type: cross  Abstract: Deploying and optimizing a PyTorch inference project on a dedicated accelerator, especially an on-device NPU, requires more than using a framework device API as with CUDA. The project must be converted into hardware-runtime artifacts and adapted to the target toolchain and application interface. Differences in model formats, quantization flows, operator support, and toolchain interfaces create cross-layer problems spanning PyTorch/ONNX models, toolchains, runtimes, and device execution. LLM Agents can invoke tools to address such problems, but may drift because of insufficient domain knowledge, large configuration and repair spaces, or poor path selection.   We propose AIPC (AI Porting & Conversion), an Agent method for deploying PyTorch inference projects. AIPC combines structured Skills with toolchain, runtime, and profiling evidence to drive cross-layer diagnosis and repair, organizing model adaptation, artifact building, on-target 
    
[^29]: 基于配置的伦理：编码智能体指令中的AI伦理原则

    Ethics by Configuration: AI Ethics Principles in Coding Agent Instructions

    [https://arxiv.org/abs/2610.34847](https://arxiv.org/abs/2610.34847)

    该研究通过分析8,893个GitHub仓库中的40,079个配置文件，首次系统揭示了开发者在配置AI编码智能体时会自发写入伦理相关声明（见于39.4%的文件），且这些声明集中于问责、透明度、隐私保护和可靠性四大原则。

    

    随着AI编码智能体成为软件开发中不可或缺的一部分，项目越来越多地利用配置文件（如上下文文件和技能文件）来规范智能体的行为。尽管AI伦理原则在研究和政策中被广泛使用，但关于开发者在配置编码智能体时是否以及如何体现这些原则，目前知之甚少。我们研究了8,893个GitHub仓库中40,079个配置文件里的伦理原则。通过使用经过验证的大语言模型（LLM）分类方法，我们测量了伦理相关声明的普遍程度，并对679个文件中的702个分层样本进行了定性分析，以揭示这些文件所反映的伦理考量及其涉及的各方。研究发现，伦理相关声明出现在39.4%的文件和61.5%的仓库中；主要集中于问责、透明度、隐私保护和可靠性方面；并且在技能等任务特定文件中比在全项目范围文件中更常出现。

    arXiv:2610.34847v1 Announce Type: new  Abstract: As AI coding agents become integral to software development, projects increasingly utilise configuration files, such as context files and skills, to govern agent behaviour. While AI ethics principles are widely used in research and policy, little is known about whether and how they are reflected when developers configure coding agents. We studied ethics principles in 40,079 configuration files across 8,893 GitHub repositories. Using a validated large language model (LLM) classification, we measured the prevalence of ethics-related statements, and qualitatively analysed 702 stratified samples across 679 files to uncover the ethical considerations reflected in these files and the parties they concern. Ethics-related statements appeared in 39.4% of files and 61.5% of repositories; concentrated on accountability, transparency, privacy protection, and reliability; and occurred more often in task-specific files such as skills than in project-w
    
[^30]: 面向量子代码翻译的大语言模型：一项探索性研究

    Large Language Models for Quantum Code Translation: An Exploratory Study

    [https://arxiv.org/abs/2610.34841](https://arxiv.org/abs/2610.34841)

    本研究提出了一种量子感知的基于大语言模型的代码翻译工作流，能够将Qiskit程序翻译为PennyLane和Q#，零样本Pass@1最高达72.33%，展示出LLM在跨量子SDK代码翻译方面的巨大潜力。

    

    量子软件开发面临着量子力学语义和碎片化软件开发工具包（SDK）带来的挑战，这使得跨SDK的代码翻译和执行变得复杂。本文提出了一种基于大语言模型的方法，用于在高级量子SDK之间翻译可执行程序。我们开发了一种量子感知的翻译工作流，其提示词设计和验证过程反映了量子语义，并能适应量子生态系统的版本演进。我们在43个任务上对4种大语言模型配置进行了初步评估，将Qiskit程序翻译为PennyLane和Q#。大语言模型实现了高达72.33%的零样本Pass@1，而单样本提示进一步提升了解译效果。这些结果表明大语言模型在量子代码翻译方面具有潜力。

    arXiv:2610.34841v1 Announce Type: new  Abstract: Quantum software development is challenged by quantum-mechanical semantics and fragmented software development kits (SDKs), which complicate cross-SDK code translation and execution. In this paper, we propose an LLM-based approach for translating executable programs between high-level quantum SDKs. We develop a quantum-aware translation workflow whose prompt design and validation reflect quantum semantics and accommodate the version evolution of quantum ecosystems. We conduct a preliminary evaluation of 4 LLM configurations on 43 tasks, translating Qiskit programs to PennyLane and Q#. LLMs achieve a zero-shot Pass@1 of up to 72.33%, while one-shot prompting further improves the translation effectiveness. These results suggest potential for quantum code translation by LLMs.
    
[^31]: 静默而代价高昂：LLM智能体中提示缓存错误集成的问题表征与检测

    Silent and Costly: Characterizing and Detecting Prompt-Caching Mis-Integration in LLM Agents

    [https://arxiv.org/abs/2610.34817](https://arxiv.org/abs/2610.34817)

    该论文首次对10个开源智能体系统中120个提示缓存相关的拉取请求进行实证研究，揭示了提示缓存错误集成往往静默失败且难以察觉的问题，并据此提出了检测工具CACHELINT。

    

    大规模语言模型服务需要处理海量的用户请求，从而产生巨额的推理成本。为了同时降低成本与延迟，服务提供商引入了提示缓存技术。然而，由于各提供商的API存在异构性且缓存语义持续演变，正确集成提示缓存并非易事。为了理解下游应用中提示缓存的错误集成问题，我们对来自10个流行开源智能体系统的120个与提示缓存相关的拉取请求开展了首次实证研究。研究结果表明，提示缓存错误集成既源于请求侧的错误配置，也源于读取侧的使用计数错误。更重要的是，大多数错误集成会静默失败，不会引发任何可观察到的运行时故障。基于这些发现，我们设计了CACHELINT，它将提示缓存错误集成建模为一个调用点以及正确缓存所需的先决条件

    arXiv:2610.34817v1 Announce Type: cross  Abstract: Large language model (LLM) services process massive volumes of user requests, resulting in substantial inference costs. To reduce both cost and latency, providers have introduced prompt caching. However, correctly integrating prompt caching is non-trivial due to heterogeneous provider-specific APIs and continuously evolving caching semantics. To understand prompt-caching mis-integration in downstream applications, we conduct the first empirical study of 120 prompt-caching-related pull requests from 10 popular open-source agent systems. Our findings show that prompt-caching mis-integrations arise in both request-side misconfiguration and read-side usage miscount. More importantly, most mis-integrations fail silently without causing observable runtime failures.   Motivated by these findings, we design CACHELINT, which models a prompt-caching mis-integration as an invocation site together with the preconditions required for correct cache 
    
[^32]: Arya Forge：金融指令数据的审计验证与发布审批

    Arya Forge: Auditing Validation and Release Approval for Financial Instruction Data

    [https://arxiv.org/abs/2610.34778](https://arxiv.org/abs/2610.34778)

    Arya Forge 将类型化记录、内容检查与绑定数据集清单的发布审批相结合，构建了一个能在金融指令数据流水线中拦截数值错误与政策违规、并区分内容有效性与数据发布许可的审计框架。

    

    金融指令数据流水线必须区分内容有效性与发布特定数据集的许可权限。Arya Forge 将类型化记录、当前内容检查以及与数据集清单绑定的审批覆盖相结合。我们在一个金融数据系统案例研究中评估其发布边界。完整性审计涵盖 100,000 条合成指令记录；一个独立的算术实现与十二个公式族中的全部 15,000 条数值记录保持一致。在包含 2,400 个处理行的 240 个场景中，在短指令过滤方案中加入金融与发布记录检查后，所有注入的数值错误和政策保留均被拒绝，同时保留了 240 个干净对照中的 220 个。丢失的二十个对照包括所有被测试的现金转换周期示例。十九个持久化发布案例显示，十一个选定的负面案例被阻止，四个等价表示被拒绝，以及外部证据在……之后的变化（原文在此处截断）。

    arXiv:2610.34778v1 Announce Type: new  Abstract: Financial instruction-data pipelines must distinguish content validity from permission to release a particular dataset. Arya Forge combines typed records, current-content checks, and approval coverage bound to a dataset manifest. We evaluate its release boundaries in a financial-data systems case study. An integrity audit covers 100,000 synthetic instruction records; a separate arithmetic implementation agrees with all 15,000 numeric records across twelve formula families. In 240 scenarios with 2,400 treatment rows, adding financial and release-record checks to a short-instruction filtering recipe rejects all injected numeric faults and policy holds while retaining 220 of 240 clean controls. The twenty lost controls comprise every tested cash-conversion-cycle example. Nineteen persisted-release cases show that eleven selected negative cases are blocked, four equivalent representations are rejected, and changes to external evidence after 
    
[^33]: 上下文的物理代价：面向编码代理的质量约束资源核算

    The Physical Cost of Context: Quality-Constrained Resource Accounting for Coding Agents

    [https://arxiv.org/abs/2610.34763](https://arxiv.org/abs/2610.34763)

    该论文提出了一种质量门控的资源核算协议，指出编码代理上下文策略的真实成本取决于调用次数和前缀缓存复用而非仅仅提示词长度，并证明预加载检索块（Eager）策略在保证代码质量的前提下可减少35.7%的模型调用和33.2%的GPU能耗。

    

    编码代理在每次模型调用时都会重新发送不断增长的上下文，而上下文策略通常仅以节省的提示词token数量来评判。但实际执行的工作量还取决于调用次数和前缀复用：启用缓存的运行时只会预填充最长已缓存前缀之后的token。因此，一个更短的提示词如果重写了已缓存的历史，反而可能代价更高。我们提出的质量门控协议将实际执行的预填充量拆分为“逻辑提示词减去复用前缀”，并测量KV缓存驻留、时间和能耗。在基于SWE-rebench V2的450个定位案例中，预加载十个检索块的策略（Eager）以及在此基础上增加锚定式历史压缩的策略（Compact）均通过了五点非劣效性质量门槛。在另一项900单元的实验中，与按需搜索（Grep）相比，Eager将模型调用次数减少了35.7%，GPU能耗代理指标减少了33.2%；而Compact未带来进一步节省。启用前缀缓存后，Eager的提示词总量基本持平（+8.3%；其95%置信区间包含零），但其未缓存预填充量降低了（原文在此处截断）

    arXiv:2610.34763v1 Announce Type: cross  Abstract: Coding agents resend a growing context on every model call, and context policies are usually judged by the prompt tokens they save. Executed work also depends on call count and prefix reuse: a caching runtime prefills only the tokens after the longest cached prefix. A shorter prompt can therefore cost more if it rewrites cached history. Our quality-gated protocol splits executed prefill into logical prompt minus reused prefix and measures KV residency, time, and energy. On 450 localization cases derived from SWE-rebench V2, preloading ten retrieved blocks (Eager) and adding anchored history compaction (Compact) both pass a five-point non-inferiority gate. In a separate 900-cell study, Eager cuts model calls by 35.7% and the GPU-energy proxy by 33.2% against on-demand search (Grep); Compact saves nothing further. Enabling prefix caching leaves Eager's prompt volume similar (+8.3%; its 95% CI spans zero) but cuts its uncached prefill by 
    
[^34]: 测试时押注式思考减少大语言模型智能体的谄媚性修改

    Thinking-in-Bets at Test Time Reduces Sycophantic Revisions in LLM Agents

    [https://arxiv.org/abs/2610.34654](https://arxiv.org/abs/2610.34654)

    提出无需训练的“押注式思考”方法，让智能体在读取用户反馈前先私密记录其论断、验证依据与置信度，从而在受到质疑时保护已验证的成果、减少LLM智能体的谄媚性修改。

    

    语言模型智能体在用户质疑其工作时常常放弃原本正确的结果，但简单地告诉它们坚持立场又会使它们忽视真正的错误。一旦智能体读到用户的抱怨，它往往为了迎合用户而低估自己之前的检查。我们提出了押注式思考，这是一种无需训练的方法：智能体在阅读用户反馈之前，先写下一份简短的私密记录，包含其主要论断、支持这些论断的检查依据以及自身的置信度。当受到质疑时，智能体会参考这份“论断清单”，在保持已验证工作完整的同时，去调查不确定的或新受到质疑的部分。我们在八个基准上评估了五个Gemini模型（3.5 Flash-Lite、3.5 Flash、3.6 Flash、3.8 Flash和4.0 Argon）：软件工程领域的SWE-Together、SWE-Interact和DeepSWE，专业报告领域的Agents' Last Exam，学术推理领域的Humanity's Last Exam和SycEval，金融分析领域的Vals Finance，以及终端任务的Terminal-Bench。

    arXiv:2610.34654v1 Announce Type: cross  Abstract: Language model agents often abandon correct work when a user challenges it, yet simply telling them to stand firm makes them ignore real mistakes. Once an agent reads a complaint, it tends to discount its own earlier checks in order to agree with the user. We introduce Thinking-in-Bets, a training-free method in which an agent writes a short private record of its main claims, the checks supporting them, and its confidence before reading user feedback. When challenged, the agent consults this Claim Checklist to keep verified work intact while investigating uncertain or newly challenged parts. We evaluate five Gemini models (3.5 Flash-Lite, 3.5 Flash, 3.6 Flash, 3.8 Flash, and 4.0 Argon) across eight benchmarks: SWE-Together, SWE-Interact, and DeepSWE for software engineering, Agents' Last Exam for professional reports, Humanity's Last Exam and SycEval for academic reasoning, Vals Finance for financial analysis, and Terminal-Bench for co
    
[^35]: AgentProofBench：面向Web智能体的可复用验证器基准测试

    AgentProofBench: Benchmarking Reusable Verifiers for Web Agents

    [https://arxiv.org/abs/2610.34634](https://arxiv.org/abs/2610.34634)

    AgentProofBench是一个评估Web智能体可复用验证器的基准，涵盖97个任务和四个Web服务，实验表明部署状态面板正确性可以有效区分正确交付与错误交付的程序分配。

    

    可复用的Web智能体验证需要一个能够跟随变化需求并支持后续执行的规则。我们提出了AgentProofBench，它将冻结验收规则的成对测试与来自相同保存起点的执行连接起来。该资源涵盖97个任务、79个任务族和四个Web服务。使用DeepSeek Flash/Pro的实验建立了两项关联测量。在48个任务族中，完整面板测试和决策回放测试分别保留了82/96个Flash和79/96个Pro源程序；成对的复用-重建BA区间分别为[-5.47, 1.57]和[-10.16, 0.00]个百分点。在22个执行任务族中，部署状态面板正确性能够有效区分两个保存起点均被正确交付的程序分配：33个无错误面板的分配中有32个成功交付，而11个存在面板错误的分配中仅有1个成功交付。所有参考定义的程序门控回归和扣留交付均发生在后一组中。中性延续获得了最高的正确性。

    arXiv:2610.34634v1 Announce Type: cross  Abstract: Reusable Web-agent verification requires a rule to follow changing requirements and support subsequent execution. We introduce AgentProofBench, which connects paired tests of a frozen acceptance rule to execution from identical saved openings. The resource spans 97 tasks, 79 families, and four Web services. Experiments with DeepSeek Flash/Pro establish two linked measurements. Across 48 families, complete-panel and decision-replay tests retain 82/96 Flash and 79/96 Pro source programs; paired reuse-rebuild BA intervals are [-5.47, 1.57] and [-10.16, 0.00] percentage points. Across 22 execution families, deployment-state panel correctness separates program assignments whose two saved openings are both delivered correctly: 32/33 assignments with error-free panels, versus 1/11 with panel errors. All reference-defined program-gate regressions and withheld deliveries occur in the latter group. Neutral continuation attains the highest correc
    
[^36]: CodeMeepo：一个用于仓库级代码迁移与优化的多智能体工作流

    CodeMeepo: A Multi-Agent Workflow for Repository-Level Code Migration and Optimization

    [https://arxiv.org/abs/2610.34633](https://arxiv.org/abs/2610.34633)

    CodeMeepo提出了一种多智能体流水线，通过将翻译与验收测试的编写职责分离、拆分大型仓库任务以克服上下文窗口限制，并对翻译后的代码进行优化，实现了完整且高效的仓库级代码迁移。

    

    仓库级代码迁移是指用一种新的编程语言重写整个项目，同时保持其可观察的行为不变。这项任务极具挑战性，因为程序行为横跨相互依赖的多个模块，依赖项和构建基础设施必须适配目标生态系统，而且源代码往往是最唯一的规范。大语言模型智能体现在已经可以端到端地尝试这类迁移，但仍有三个挑战存在：首先，同时编写翻译代码和验收测试的智能体可能会在两者中忽略同样的需求，导致缺失或错误的行为无法被察觉；其次，大型仓库可能超出模型的上下文窗口，而将工作拆分为较小的任务并不能保证所有所需行为都被迁移；第三，即使是行为正确的翻译代码，其运行速度也可能比原始实现更慢。我们提出了CodeMeepo，这是一个用于完整且高效迁移大型仓库的多智能体流水线。

    arXiv:2610.34633v1 Announce Type: cross  Abstract: Repository-level code migration rewrites an entire project in a new programming language while preserving its observable behavior. It is challenging because behavior spans interdependent modules, dependencies and build infrastructure must be adapted to the target ecosystem, and the source code is often the only specification.   Large language model agents can now attempt these migrations end to end, but three challenges remain. First, an agent that writes both the translation and its acceptance tests may overlook the same requirements in both, leaving missing or incorrect behavior undetected. Second, large repositories can exceed a model's context window, and dividing the work into smaller tasks does not ensure that all required behavior is migrated. Third, even behaviorally correct translations may be slower than the original implementation.   We present CodeMeepo, a multi-agent pipeline for complete and efficient migration of large r
    
[^37]: 为记忆而构建，交付后即遗忘：容器化LLM智能体部署中的运维契约、故障与修复

    Built to Remember, Shipped to Forget: Operational Contracts, Failures, and Repairs in Containerized LLM Agent Deployments

    [https://arxiv.org/abs/2610.34620](https://arxiv.org/abs/2610.34620)

    该论文首次实证研究了LLM智能体框架的持久化需求与容器平台“容器一次性”假设之间的矛盾，提出14条“运维契约”，并通过433个开源智能体系统、7种诱导故障事件和122个开发者修复对契约进行了评估。

    

    LLM智能体系统以容器镜像的形式进入生产环境，由Docker、Compose和Kubernetes等平台运行。框架与平台的文档表达了截然相反的期望：对平台而言，容器是一次性的，而框架却会持久化对话、暂停运行以等待人类输入，并从保存的状态中恢复被中断的工作；双方各自只规定了自己的行为，因此二者组合所产生的义务在任何地方都没有被说明。我们对这一边界开展了首个实证研究，将这些义务表述为“运维契约”：即当部署平台事件发生时，部署必须提供的保证，以维持智能体系统所需的行为。我们从LangGraph和三个平台的第一方文档中推导出14条契约，并利用三个独立的证据来源对其进行评估：433个已部署的开源智能体系统、在其中12个系统上诱导出的七种事件，以及从39,261个提交（commits）中挖掘出的122个开发者修复。

    arXiv:2610.34620v1 Announce Type: new  Abstract: LLM agent systems reach production as container images run by platforms such as Docker, Compose, and Kubernetes. Framework and platform document opposite expectations: to the platform a container is disposable, while the framework persists conversations, pauses runs for human input, and resumes interrupted work from saved state; each side specifies only its own behavior, so the obligations their composition creates are stated nowhere. We present the first empirical study of this boundary, stating these obligations as operational contracts: guarantees a deployment must provide to preserve an agent system's required behavior when a deployment-platform event occurs. From the first-party documentation of LangGraph and the three platforms we derive 14 contracts and evaluate them with three independent sources of evidence: 433 deployed open-source agent systems, seven events induced on 12 of them, and 122 developer repairs mined from 39,261 co
    
[^38]: AI编码代理像人类一样编写基础设施即代码吗？——基于GitHub拉取请求的实证研究

    Do AI Coding Agents Write Infrastructure as Code Like Humans? An Empirical Study of GitHub Pull Requests

    [https://arxiv.org/abs/2610.34613](https://arxiv.org/abs/2610.34613)

    该研究首次大规模实证分析了34.9万个GitHub IaC拉取请求，发现AI编码代理编写的PR规模约为人类的四倍且更常改动核心基础设施模块，尽管其总体合并率更高，但在同一仓库内却比人类PR更难获得维护者的合并认可。

    

    基础设施即代码配置定义了云资源、网络和访问控制，而AI编码代理正越来越多地编写这些配置。目前人们对代理的贡献与人类及自动化机器人有何不同、维护者如何评估这些贡献，以及它们是否引入不同的可持续性问题知之甚少。我们研究了来自49,241个GitHub仓库的349,156个IaC拉取请求（PR），区分了人类、自动化机器人、未披露代理和已披露代理。结果显示，机器人占表面上由人类提交的PR的32.9%，其余PR中有12.3%包含代理痕迹，且这一比例在研究期间持续增长。代理PR的规模约为人类PR的四倍，且更常修改核心基础设施模块，而机器人主要负责维护依赖项。未披露代理的PR的总体合并率高于人类PR（87.4%对78.9%），但在同一仓库内其合并率却低4.0个百分点。已披露代理的相关情况（原文在此处截断）。

    arXiv:2610.34613v1 Announce Type: new  Abstract: Infrastructure-as-Code (IaC) configurations define cloud resources, networking, and access control, and AI coding agents increasingly write them. Little is known about how agent contributions differ from those of humans and automation bots, how maintainers evaluate them, and whether they introduce different sustainability concerns.   We study 349,156 IaC pull requests (PRs) from 49,241 GitHub repositories, distinguishing humans, automation bots, silent agents, and disclosed agents. Bots account for 32.9% of ostensibly human PRs, and 12.3% of the remaining PRs contain agent traces, a share that grows over the study period. Agent PRs are about four times larger than human PRs and more often modify core infrastructure modules, whereas bots mainly maintain dependencies. Silent-agent PRs have higher overall merge rates than human PRs (87.4% vs. 78.9%) but are merged 4.0 percentage points less often within the same repositories. Disclosed agen
    
[^39]: NovaFabric：一种面向AI智能体的黑盒记录器，实现密封、可重放、可对比且可审计的运行记录

    NovaFabric: A Black-Box Recorder for AI Agents with Sealed, Replayable, Diffable and Auditable Runs

    [https://arxiv.org/abs/2610.34594](https://arxiv.org/abs/2610.34594)

    提出了开源黑盒记录器NovaFabric，可将AI智能体的复杂运行过程封装为密封、可重放、可对比且可审计的证据胶囊，解决了传统日志可变、不完整且无法重放执行的问题。

    

    AI智能体的一次运行不再是一个简单的函数调用，而是由提示词、模型调用、工具调用、文件与网络副作用以及子智能体组成的级联过程。当这样的运行出现异常行为，或审计人员询问究竟发生了什么时，日志和追踪信息往往难以给出令人满意的答案：它们是可变的、不完整的，且无法重新执行。本技术报告介绍了NovaFabric，一个开源（Apache-2.0协议）的AI智能体黑盒记录器，它能将一次运行转化为密封、可重放、可对比、可审计的证据胶囊。该设计仅基于两种格式：运行胶囊是一个扁平文件目录，作为唯一可信数据源；每个数据库都是基于胶囊可重建的索引。证据包则是提供给审计人员的签名导出文件。其余一切都遵循现有标准：OpenTelemetry GenAI、OpenLineage、W3C PROV、in-toto/DSSE、RFC 3161和CycloneDX。单节点运行是分布式运行的最小情形；该模式支持面向Slurm数组和Kubernetes的胶囊层级结构。

    arXiv:2610.34594v1 Announce Type: cross  Abstract: An AI agent run is no longer a function call. It is a cascade of prompts, model calls, tool invocations, file and network effects and sub-agents. When such a run misbehaves, or an auditor asks what happened, logs and traces answer poorly: they are mutable, partial and cannot be re-executed. This technical report describes NovaFabric, an open-source (Apache-2.0) black-box recorder for AI agents that turns a run into a sealed, replayable, diffable and auditable evidence capsule. The design rests on two formats only. The Run Capsule is a directory of flat files and the source of truth; every database is a rebuildable index over capsules. The Evidence Bundle is the signed export for an auditor. Everything else follows existing standards: OpenTelemetry GenAI, OpenLineage, W3C PROV, in-toto/DSSE, RFC 3161 and CycloneDX. A single-node run is the smallest case of a distributed one; the schema supports a capsule hierarchy for Slurm arrays and K
    
[^40]: 面向AI驱动科学的软件工程

    Software Engineering for AI-Driven Science

    [https://arxiv.org/abs/2610.34573](https://arxiv.org/abs/2610.34573)

    本文提出“面向AI驱动科学的软件工程”（SE4AI4Sci）这一新兴研究方向，通过“科学保证链”框架和六个真实失败案例分析，指出了保障AI驱动科学发现可靠性与可复现性的关键研究挑战。

    

    人工智能（AI）已为科学发现做出了重大贡献。自主智能体如今能够独立分析复杂问题、设计实验、编写并执行代码、分析研究结果并推导出科学结论。然而，AI生成的代码、实验流程或证据解读中的错误可能导致无效的研究结论，损害自动化科学发现的可靠性与可复现性。在这篇愿景论文中，我们提出“面向AI驱动科学的软件工程”（SE4AI4Sci）作为一个新兴研究领域，以应对这一挑战。作为组织框架，我们引入了“科学保证链”，将意图、实验设计、实现、执行、证据与结论断言连接起来。通过对近期AI for Science系统中六个有记录的失败案例的分析，我们展示了该链条可能在何处断裂、现有保证机制在何处不足，以及由此带来的新研究挑战。我们还……（原文摘要在此处截断）

    arXiv:2610.34573v1 Announce Type: new  Abstract: AI has made significant contributions to scientific discovery. Autonomous agents can now independently analyse complex problems, design experiments, write and execute code, analyse findings, and derive scientific conclusions. Nevertheless, errors in AI-generated code, experimental procedures, or evidence interpretation can lead to invalid research conclusions, undermining the reliability and reproducibility of automated scientific discovery. In this vision paper, we propose Software Engineering for AI-Driven Science (SE4AI4Sci) as an emerging research area for addressing this challenge. As an organising framework, we introduce the Scientific Assurance Chain, connecting intent, experimental design, implementation, execution, evidence, and claim. Through six documented failures in recent AI-for-science systems, we show where this chain can break, where existing assurance mechanisms fall short, and what new research challenges follow. We al
    
[^41]: 面向空间智能的智能体AI编排框架：供应链部署

    Agentic AI Orchestration Framework for Spatial Intelligence: Supply Chain Deployment

    [https://arxiv.org/abs/2610.34569](https://arxiv.org/abs/2610.34569)

    提出了ANASSA参考架构，通过类型化数据契约、权威空间基础验证、多智能体编排与两道人在环审批，解决工业空间决策中跨系统割裂、缺乏统一参考框架的问题，并已在供应链多灾害风险评估场景中实现实际部署。

    

    工业空间决策是在既不共享参考框架也不共享所有权归属的多个系统中组装而成的，分析师往往要花费大量精力协调这些系统，而非专注于对空间本身的推理。本文提出了ANASSA，一种集成结构化推理、多智能体工作流编排、权威空间基础锚定以及反馈驱动执行的参考架构，并报告了其在实际运行的供应链部署中的实例化。该架构在智能体边界处设置类型化数据契约，针对经过认证的权威基础数据验证空间声明，并在六步认知循环中强制执行两次阻塞性的人在环审批。该部署涵盖海运与内陆货运通道上的多灾害风险评估，包括基于集装箱遥测的冷链异常偏离检测。在为期三个月、包含三次实施迭代的概念验证（POC）过程中，三个部署成果……（原文摘要至此截断）

    arXiv:2610.34569v1 Announce Type: new  Abstract: Industrial spatial decisions are assembled across systems that share neither a reference frame nor an owner, and analysts spend substantial effort reconciling those systems rather than reasoning about space. This paper presents ANASSA, a reference architecture integrating structured reasoning, multi-agent workflow orchestration, authoritative spatial grounding, and feedback-driven execution, and reports its instantiation in an operating supply-chain deployment. The architecture places typed data contracts at agent boundaries, validates spatial claims against a certified authoritative foundation, and enforces two blocking human-in-the-loop approvals inside a six-step cognitive loop. The deployment covers multi-hazard risk assessment on an ocean-and-inland freight lane, including cold-chain excursion detection over container telemetry. Across a three-month proof of concept (POC) with three implementation iterations, three deployment outcom
    
[^42]: MOFjson：一种面向语言模型的金属有机框架显式拓扑表示

    MOFjson: An Explicit Topological Representation of Metal-Organic Frameworks for Language Models

    [https://arxiv.org/abs/2610.34549](https://arxiv.org/abs/2610.34549)

    MOFjson首次提出将MOF结构分离为包含显式键移矢量的原始原子层与暴露次级结构单元和网络拓扑的派生层，使语言模型在MOF结构理解的各项任务上大幅超越现有基线。

    

    用于金属有机框架（MOFs）的语言模型目前所依赖的格式将原子几何与结构拓扑相分离，迫使模型在非结构化的坐标列表和非空间性的字符串标识符之间做出选择。我们提出了MOFjson，这是首个将结构数据分离为两层的表示方法：一层是包含显式键移矢量的原始原子层，另一层是暴露次级结构单元（金属节点和有机连接体）及网络拓扑的派生层。此外，我们引入了一个涵盖四个问题类别、具有晶体来源真值标签的基准测试。在四个模型上的实验表明，MOFjson在所有类别中均超越了三个基线方法：在判断一个结构块的最近周期性副本任务上达到100%对18-47%，在晶系表征任务上达到96%对71%，在金属节点和有机连接体配位任务上达到60%对48%。消融实验显示，移除移位标签后准确率降至12.7%，证明模型读取的是周期性图结构而非表面格式。MOFjson为语言模型提供了……

    arXiv:2610.34549v1 Announce Type: cross  Abstract: Language models for metal-organic frameworks (MOFs) rely on formats that separate atomic geometry from structural topology, forcing models to choose between unstructured coordinate lists and non-spatial string identifiers[1]. We present MOFjson, the first representation separating structural data into a raw atomic layer with explicit bond shift vectors and a derived layer exposing secondary building units (metal nodes and organic linkers) and network topology. Additionally, we introduce a benchmark with crystal-derived ground truth across four question categories. Across four models, MOFjson beats all three baselines in every category: 100% vs. 18-47% on reaching a block's nearest periodic copy, 96% vs. 71% on crystal system characterisation, and 60% vs. 48% on metal node and organic linker coordination. Ablating shift labels drops accuracy to 12.7%, proving models read the periodic graph rather than the surface format. MOFjson gives l
    
[^43]: 工业环境中的多模态需求可追溯性链接恢复

    Multimodal Requirements Traceability Link Recovery in Industrial Settings

    [https://arxiv.org/abs/2610.34503](https://arxiv.org/abs/2610.34503)

    该论文提出了一种多模态RAG扩展方法，通过分别检索文本和图像候选、利用视觉-语言模型进行分类并用言语化置信度过滤预测链接，从而在工业需求可追溯性链接恢复任务中将F1分数从基线的0.114提升至0.310。

    

    需求可追溯性链接可以支持变更影响分析、覆盖率检查和验证等活动。工业需求中可能包含可追溯的图像，而纯文本方法无法直接处理这些图像。对于固定的分类器配置而言，仅依靠二元决策无法直接调整漏检链接与错误链接之间的平衡。我们扩展了一种基于检索增强生成（RAG）的需求间可追溯性链接恢复（TLR）方法。我们的扩展方法分别检索文本和图像候选，使用视觉-语言模型对每个检索到的配对进行分类，并通过言语化置信度对预测的链接进行过滤。我们在三个互不重叠的工业需求集合和六个公开的纯文本数据集上评估了我们的方法。在工业测试集上，我们的方法取得了0.310的F1分数，而我们对Hey等人基线方法的复现仅达到0.114。我们的方法恢复的链接（原文在此处截断）

    arXiv:2610.34503v1 Announce Type: cross  Abstract: Requirements traceability links support activities such as change-impact analysis, coverage checking, and validation. Industrial requirements can include traceable images, which text-only approaches cannot process directly. For a fixed classifier configuration, binary decisions alone offer no direct way to adjust the balance between missed and incorrect links. We extend a retrieval-augmented generation (RAG) approach for inter-requirements traceability link recovery (TLR). Our extension retrieves text and image candidates separately, classifies each retrieved pair with a vision-language model, and filters the predicted links by verbalized confidence. We evaluate our approach on three non-overlapping industrial requirement sets and six public text-only datasets. On the industrial test sets, our approach achieves an F1 score of 0.310, compared with 0.114 for our replication of the baseline approach by Hey et al. Our approach recovers lin
    
[^44]: 关于大语言模型编写基于属性测试有效性的多领域研究

    A Multi-Domain Study on the Effectiveness of LLMs at Writing Property-Based Tests

    [https://arxiv.org/abs/2610.34487](https://arxiv.org/abs/2610.34487)

    该研究首次将基于LLM的属性测试编写工具从Python库扩展评估到深度学习、数据库、系统软件、编译器和LLM服务五个领域的十个程序，发现Agentic-PBT的分支覆盖表现更优、可与领域专用模糊测试工具相媲美，并通过180份错误报告（65份已获确认）证明了其跨领域发现真实缺陷的能力。

    

    基于属性的测试（PBT）通过在大量生成的输入上检查程序的某个属性来验证程序，但编写这样的测试需要定义一个属性、一个有效输入生成器和一个预言机。基于大语言模型的PBT编写工具能够自动生成这三者，但此前仅在Python库上被评估过。我们研究了其中两个工具——Proptest-AI和Agentic-PBT——如何泛化到五个领域的十个程序上：深度学习库、数据库、系统软件、编译器和LLM服务引擎。这些程序大多接受其自身语言形式的文本输入，例如查询、配置文件和程序代码。在使用Opus 5模型时，Agentic-PBT在十个程序中的七个上比Proptest-AI覆盖了更多分支；在使用GLM-5.3-Flash时，Agentic-PBT在全部十个程序上均覆盖更多分支。领域专用的模糊测试工具在编译器和PostgreSQL上的覆盖率高于Agentic-PBT，而在PyTorch上的覆盖率则低于它。我们共提交了180份错误报告，其中44份与模型知识截止日期之后他人提交的报告重复；其余报告中有65份已获维护者确认，且覆盖了全部五个领域。

    arXiv:2610.34487v1 Announce Type: new  Abstract: A property-based test (PBT) checks a property of a program on many generated inputs, but writing one takes a property, a generator of valid inputs, and an oracle. LLM-based PBT writers produce all three, yet they have been evaluated only on Python libraries. We study how two of them, Proptest-AI and Agentic-PBT, generalize to ten programs in five domains: deep learning libraries, databases, system software, compilers, and LLM serving engines. Most of these programs take text in their own language, such as queries, configurations, and programs. Agentic-PBT covers more branches than Proptest-AI on seven of the ten programs with Opus 5 and on all ten with GLM-5.3-Flash. Domain-specific fuzzers cover more of the compilers and of PostgreSQL and less of PyTorch than Agentic-PBT. We filed 180 bug reports, 44 of which duplicate reports that others filed after the models' knowledge cutoffs; maintainers have confirmed 65 of the others, in all five
    
[^45]: CyberDual：真实软件维护中AI攻击者与AI代码审查者的对抗

    CyberDual: AI Attackers versus AI Code Reviewers in Real Software Maintenance

    [https://arxiv.org/abs/2610.34435](https://arxiv.org/abs/2610.34435)

    该论文提出CyberDual竞技场，让AI攻击者智能体在真实开源软件维护任务中隐藏可利用漏洞、AI审查者智能体负责发现并修复，并以重新执行攻击者PoC的结果裁决胜负，揭示现有审查智能体虽能拦截85-91%的攻击，但仍有约六分之一的分散式攻击可逃过全部审查，且开源权重的GLM 5.3是最强攻击者。

    

    开源项目如今已允许AI智能体审查提交的代码，但被审查的代码本身可能来自AI攻击者。我们提出了CyberDual，一个针对AI这种双重角色的竞技场：攻击者智能体在广泛使用的开源项目的真实维护工作中隐藏一个可利用的漏洞，而审查者智能体则尝试发现并修复它。每场比赛由执行结果裁决：我们在审查后的代码上重新运行攻击者的概念验证。每次攻击都是全新的，且审查窗口可以每个季度从最新的上游历史重建。三个智能体审查者捕获了来自六个攻击者的1,088个有效攻击中的85%至91%。然而，当攻击者将漏洞分散到真实的上游提交中时，大约每六个攻击中仍有一个在审查后成功留下可利用的漏洞。最强的攻击者拥有开放权重：GLM 5.3几乎从不拒绝执行攻击，其攻击在绕过全部三个审查者的28个攻击中占了13个。当审查者出现遗漏时，Claude审查者大多……（摘要原文在此处截断）

    arXiv:2610.34435v1 Announce Type: cross  Abstract: Open-source projects now let AI agents review incoming code, but the code under review may itself come from an AI attacker. We present CyberDual, an arena for this dual role of AI: attacker agents hide an exploitable bug in real maintenance work on widely used open-source projects, and reviewer agents try to find and repair it. Execution decides each match: we rerun the attacker's proof of concept (PoC) on the code that comes out of review. Every attack is new, and the windows can be rebuilt each quarter from the latest upstream history. Three agentic reviewers catch 85-91% of 1,088 valid attacks from six attackers. Yet when the attacker spreads the bug over real upstream commits, about one in six attack slots still ends with a working exploit after review. The strongest attacker has open weights: GLM 5.3 almost never refuses, and its attacks make up 13 of the 28 that get past all three reviewers. When they miss, Claude reviewers mostl
    
[^46]: SymLift：面向上下文不完整代码片段的免编译漏洞检测神经语义提升方法

    SymLift: Neural Semantic Lifting for Compile-Free Vulnerability Detection in Context-Incomplete Code Fragments

    [https://arxiv.org/abs/2610.34431](https://arxiv.org/abs/2610.34431)

    提出SymLift框架，利用微调大语言模型将上下文不完整的代码片段提升为类LLVM中间表示中的漏洞相关程序关系，再通过Datalog规则实现无需编译整个项目、可审计且确定性的漏洞检测。

    

    缺失的项目上下文会阻碍依赖构建的分析器构建漏洞检测所需的程序关系，而仅靠神经预测又无法建立可审计的推导过程。我们提出了SymLift，这是一个以面向漏洞的神经语义提升为核心的免编译框架，用于处理上下文不完整的代码片段。经过微调的大语言模型在一种受约束的类LLVM中间表示（IR）中重建与漏洞相关的控制流、数据流、内存和资源关系，而无需重建完整的项目上下文，也无需直接预测漏洞标签。确定性解析器验证该IR，并将其支持的操作和关系映射为规范化的证据事实。显式的Datalog规则推导出带有支持事实和推导踪迹的检测结果，确保在事实和规则固定的情况下推理具有确定性。我们在FormAI、Juliet和MegaVul数据集上评估了SymLift。Reduc（原文摘要至此处截断）

    arXiv:2610.34431v1 Announce Type: new  Abstract: Missing project context can prevent build-dependent analyzers from constructing program relations required for vulnerability detection, while neural predictions alone do not establish auditable derivations. We present SymLift, a compile-free framework centered on Vulnerability-Oriented Neural Semantic Lifting for context-incomplete code fragments. A fine-tuned large language model reconstructs vulnerability-relevant control-flow, data-flow, memory, and resource relations in a constrained LLVM-like intermediate representation (IR), without reconstructing the full project context or directly predicting vulnerability labels. A deterministic parser validates the IR and maps its supported operations and relations into normalized evidence facts. Explicit Datalog rules derive findings with supporting facts and derivation traces, ensuring deterministic reasoning for fixed facts and rules. We evaluate SymLift on FormAI, Juliet, and MegaVul. Reduc
    
[^47]: 使用编程智能体从GPU迁移到TPU：什么有效、什么失败、代价几何？

    From GPU to TPU with Coding Agents: What Works, What Fails, and What It Costs?

    [https://arxiv.org/abs/2610.34422](https://arxiv.org/abs/2610.34422)

    该研究构建了148个PyTorch研究项目的数据集，并使用Claude Code将其中10个项目从NVIDIA GPU迁移到Google Cloud TPU，通过独立评估发现迁移平均需2.4次人工干预、76%的执行尝试失败且近半数失败不报错，首次系统量化了智能体驱动代码迁移的可信度与成本。

    

    一个编程智能体可以迁移一个机器学习工作负载并报告成功，但仍然会改变模型的训练方式。不断上涨的计算成本、有限的加速器容量以及对供应商的依赖，正促使各组织寻求图形处理器（GPU）的替代方案；一个关键问题是，由智能体生成的向张量处理单元（TPU）的迁移是否可信、其代价如何。我们构建了一个包含148个PyTorch机器学习研究项目的数据集，并使用Claude Code将其中十个项目从NVIDIA GPU迁移到Google Cloud TPU。我们独立评估每次迁移是否能够运行、是否保持数值行为、是否在预期硬件上执行、是否维持项目目标，以及是否提供可接受的性能。我们还考察了失败情况、人工投入与成本。八个项目满足我们的项目目标标准，迁移平均需要2.4次人工干预。在70次执行尝试中，53次失败（76%），其中26次失败（49%）没有引发任何错误信息。

    arXiv:2610.34422v1 Announce Type: new  Abstract: A coding agent can migrate a machine learning workload, report success, and still change how the model trains. Rising compute costs, limited accelerator capacity, and vendor dependence are pushing organizations toward alternatives to graphics processing units (GPUs); a key question is whether agent-generated migrations to tensor processing units (TPUs) can be trusted and at what cost. We build a dataset of 148 PyTorch ML research projects and use Claude Code to migrate ten of them from NVIDIA GPUs to Google Cloud TPUs. We independently assess whether each migration runs, preserves numerical behavior, executes on the intended hardware, maintains the project's goal, and delivers acceptable performance. We also examine failures, human effort, and costs. Eight projects meet our project-goal criterion, and migrations require 2.4 human interventions on average. Of 70 execution attempts, 53 fail (76%), and 26 of these failures (49%) raise no er
    
[^48]: MulFCoder：面向基于多模态大模型的多框架前端代码生成的框架条件化多智能体方法

    MulFCoder: Framework-conditioned Multi-agent for MLLM-based Multi-framework Front-end Code Generation

    [https://arxiv.org/abs/2610.34418](https://arxiv.org/abs/2610.34418)

    MulFCoder提出了一种框架条件化的多智能体方法，通过显式编码框架约束并协调Grounder、Planner、Writer、Judger四个智能体，实现了从UI截图生成React/Vue/Angular等多框架的可运行前端代码。

    

    从UI截图生成可运行的前端代码是自动化软件工程领域的长期目标。现有基于多模态大语言模型（MLLM）的方法主要聚焦于HTML/CSS，而对React/Vue/Angular等多框架代码生成的研究仍然不足。简单地修改提示词会在多框架场景下造成显著的性能差距，并产生高度框架特有的错误模式。为解决这一问题，我们提出MulFCoder，这是一种框架条件化的多智能体方法，通过显式编码框架约束，将多框架之间的差异引入可判定的规则空间。MulFCoder协调四个智能体协同工作：Grounder从检测到的UI元素构建元素表（ElementTable）、内容表（ContentTable）和宏观布局区域；Planner构建类DOM的层次化布局树，生成任务调度，并推导出框架特定的文件契约；Writer在受限的编辑窗口内生成结构化的文件写入或补丁；Judger执行轻量级（摘要在此处截断，后续内容未提供）。

    arXiv:2610.34418v1 Announce Type: cross  Abstract: Generating runnable front-end code from UI screenshots is a long-standing goal in automated software engineering. Existing MLLM-based methods predominantly focused on HTML/CSS, leaving multi-framework generation for React/Vue/Angular underexplored. Naively modifying prompts leads to substantial performance gaps across multi-framework and highly framework-specific error modes. To address this, we propose MulFCoder, a framework-conditioned multi-agent method that explicitly encodes framework constraints to bring multi-framework differences into a decidable rule space. MulFCoder orchestrates four agents: Grounder constructs an ElementTable, ContentTable, and macro-layout regions from detected UI elements; Planner builds a DOM-like hierarchical layout tree, produces a task schedule, and derives a framework-specific file Contract; Writer generates structured file writes or patches within a restricted edit window; Judger enforces lightweight
    
[^49]: ComBeautify：利用代码变更摘要生成规范化提交信息

    ComBeautify: Leveraging Code Change Summaries for Conventional Commit Generation

    [https://arxiv.org/abs/2610.34374](https://arxiv.org/abs/2610.34374)

    本文提出ComBeautify，利用大语言模型对Git差异进行保留上下文的摘要，自动生成符合规范化提交规范（CCS）的提交信息，帮助开发者更准确地表达代码变更意图。

    

    现代软件开发依赖于诸如Git之类的版本控制系统来跟踪变更并促进协作。虽然提交信息对于传达开发者意图至关重要，但它们并不影响程序功能；因此，开发者往往缺乏撰写有意义提交信息的动力。规范化提交规范通过结构化格式和指导原则来解决这一问题，帮助开发者表达变更意图，提高清晰度并支持自动化。然而，开发者在遵循该格式的同时表达意图时常常面临困难。特别是，由于可选类别数量众多或缺乏项目特定的指导，他们可能会错误地分类提交。我们提出了ComBeautify，一种自动生成符合CCS规范提交信息的方法。我们的混合方法利用大语言模型（LLM）对Git差异进行保留上下文的摘要，以改进提交分类。

    arXiv:2610.34374v1 Announce Type: cross  Abstract: Modern software development relies on version control systems such as Git to track changes and facilitate collaboration. While commit messages are crucial for conveying developer intent, they do not affect program functionality; as a result, developers often lack motivation to write meaningful ones. The Conventional Commits Specification (CCS) addresses this with a structured format and guidelines that help developers express the intent of a change, improving clarity and enabling automation. Yet, developers frequently struggle to express intent while adhering to the format. In particular, they may misclassify commits due to the large number of possible categories or the lack of project-specific guidance. We introduce ComBeautify, an approach that automatically generates CCS-compliant commit messages. Our hybrid approach leverages context-preserving summarization of Git diffs with large language models (LLMs) to improve commit classific
    
[^50]: EAST：基于数字孪生系统化故障注入的具身人工智能安全测试

    EAST: Embodied AI Safety Testing via Systematic Fault Injection in Digital Twins

    [https://arxiv.org/abs/2610.34358](https://arxiv.org/abs/2610.34358)

    EAST是一个基于数字孪生的具身AI安全测试框架，它通过系统化的故障注入方法，将文献中记录的安全证据转化为覆盖感知、指令、环境、动作和时序的可执行、可验证的安全测试。

    

    任务成功并不是衡量具身人工智能（AI）安全性的充分标准，因为机器人可能在违反安全要求的情况下完成目标。现有的评估通常强调性能和鲁棒性，但缺乏从已记录的安全危害到可执行、可验证测试的系统性路径。我们提出了EAST（具身人工智能安全测试），这是一个数字孪生框架，能够将基于文献的安全证据转化为涵盖感知、指令、环境、动作和时序五个方面的故障注入测试。EAST结合了经专家审查的安全故障分类法以及八个带有适用性和有效性约束的可执行操作符。其三阶段工作流程可生成测试规范，在仿真执行过程中验证故障是否真正实现，并利用仿真器状态证据独立评估安全挑战、违规行为、任务结果和行为特征。该框架保持了执行……（原文摘要在此处截断）

    arXiv:2610.34358v1 Announce Type: new  Abstract: Task success is not a sufficient measure of safety for embodied artificial intelligence (AI), because robots may complete objectives while violating safety requirements. Existing evaluations often emphasize performance and robustness, but lack a systematic route from documented safety hazards to executable, verifiable tests. We present EAST (Embodied AI Safety Testing), a digital-twin framework that converts literature-grounded safety evidence into fault-injection tests across perception, instructions, environments, actions, and timing. EAST combines an expert-reviewed safety-fault taxonomy and eight executable operators with applicability and validity constraints. Its three-stage workflow generates test specifications, verifies fault realization during simulated execution, and independently assesses safety challenges, violations, task outcomes, and behavioral signatures using simulator-state evidence. The framework preserves execution t
    
[^51]: 开发者向大语言模型委派任务及信任因素：一项混合方法研究

    Developer Task Delegation to LLMs and the Trust Factor: A Mixed Methods Study

    [https://arxiv.org/abs/2610.34317](https://arxiv.org/abs/2610.34317)

    该研究通过22次访谈和160名从业者的问卷调查，首次系统揭示了开发者向大语言模型委派任务的四种模式（完全委派、部分委派、自我委派和任务回避），并通过统计分析证明信任是影响任务委派决策的关键因素。

    

    随着大语言模型（LLMs）的兴起，软件开发者迎来了一种强大的新方式：将软件开发任务委派给大语言模型。现有研究仅针对部分任务和因素探索了任务委派问题。关键在于，开发者将哪些任务委派给大语言模型、委派到何种程度、以及为什么委派这些任务，目前尚不明确。我们通过22次访谈和对160名软件从业者的问卷调查来探索这些问题。开发者报告了向大语言模型委派任务的四种模式：完全委派、部分委派、自我委派和任务回避。基于我们的定性研究结果，我们讨论了这些选择背后的原因，并探讨了任务难度、任务风险、开发者动机和信任如何与向大语言模型委派软件开发任务相关联。我们的统计分析表明，信任是与向大语言模型委派任务相关的一个显著因素。

    arXiv:2610.34317v1 Announce Type: cross  Abstract: With the rise of Large Language Models (LLMs), software developers are presented with a powerful new approach: delegating software development tasks to LLMs. Existing studies have explored task delegation for a subset of tasks and factors. Critically, what tasks developers delegate to LLMs, to what extent, and why they delegate them, is not yet well understood. We explore these questions through 22 interviews and a survey with 160 software practitioners. Developers reported employing four modes of task delegation to LLMs: full delegation, partial delegation, self-delegation, and task avoidance. Drawing on our qualitative findings, we discuss the rationales behind these choices and explore how task difficulty, task risk, developer motivation, and trust appear to be associated with software development task delegation to LLMs. Our statistical analysis reveals that trust is a significant factor associated with task delegation to LLMs. Fin
    
[^52]: 使用大语言模型修复Web应用程序中的失效访问控制：一项实证基准研究

    Repairing Broken Access Control in Web Applications with LLMs: An Empirical Benchmark Study

    [https://arxiv.org/abs/2610.34290](https://arxiv.org/abs/2610.34290)

    该研究构建了首个用于修复Web应用失效访问控制漏洞的基准BAC-Bench，包含来自六个真实Web应用的46个漏洞，并提出了相应的修复策略框架，为评估大语言模型修复此类逻辑漏洞的能力提供了标准化平台。

    

    失效访问控制是最严重的Web应用程序漏洞类别之一，但其自动化修复在很大程度上仍未被探索。与内存安全漏洞不同，失效访问控制是一种逻辑漏洞，其修复需要对应用程序的访问控制模型和业务语义有深入的理解，而传统的自动化漏洞修复技术对此处理不佳。大语言模型为这一问题开辟了一条有前景的途径，但目前尚无标准基准来研究其修复失效访问控制漏洞的能力。我们构建了BAC-Bench，这是首个用于失效访问控制修复的基准，包含46个来自六个广泛使用的Web应用程序的真实失效访问控制漏洞，涵盖四种编程语言。每个漏洞都配备了一个沙盒化的运行环境、一个PoC验证程序和一个功能测试程序。我们进一步提出了一个失效访问控制修复策略框架，对任何失效访问控制漏洞的特征进行刻画。

    arXiv:2610.34290v1 Announce Type: cross  Abstract: Broken access control (BAC) ranks among the most severe web-application vulnerability classes, yet its automated repair remains largely unexplored. Unlike memory-safety vulnerabilities, BAC is a logic vulnerability whose repair demands a deep understanding of an application's access-control model and business semantics, which traditional automated vulnerability repair techniques handle poorly. Large language models (LLMs) open a promising avenue for this problem, but no standard benchmark exists for studying how well they repair BAC vulnerabilities. We construct BAC-Bench, the first benchmark for BAC repair, consisting of 46 real-world BAC vulnerabilities drawn from six widely used web applications spanning four programming languages. Each vulnerability is equipped with a sandboxed runtime environment, a PoC validation program and a functional test program. We further propose a BAC repair-strategy framework that characterizes any BAC p
    
[^53]: 超越基准测试：小型开放权重模型能否完成真正的编码工作？

    Beyond Benchmarks: Can Small Open-Weight Models Get Real Coding Work Done?

    [https://arxiv.org/abs/2610.34241](https://arxiv.org/abs/2610.34241)

    该实证经验报告基于4,165次真实编码代理会话、623个调试任务和三项委托研究，指出小型开放权重模型能否完成真实编码工作取决于工具调用格式、缓存复用等系统层面的工程细节，而非基准测试成绩。

    

    让开放权重语言模型生成文本并不等于让它完成编码工作。本报告是一份回顾性的实证系统工程经验报告，记录了作者从2026年7月到10月在编码客户端、模型特定服务配置、vLLM、SGLang、llama.cpp和MLX方面的工程实践。研究发现，名义上兼容的API端点背后的模型在工具调用格式、回合控制、上下文行为、缓存复用和可靠性方面仍存在显著差异。模型被成功部署、生成连贯文本、能执行工具调用、以及代码通过独立行为检查，被证明是四个不同层次的进展阶段。证据包括来自作者自己会话的4,165次编码代理网关交互、623个调试任务，以及三项带有隐藏行为检查并使用固定的Claude Opus 5.5作为父代理的独立委托研究。保存的冷前缀请求的中位首字时间（time to first token）约为69.4秒，而高复用请求仅为2.8秒（观察性数据）。在一项45次试验的研究中，直接codi……（摘要原文在此处截断）

    arXiv:2610.34241v1 Announce Type: new  Abstract: Making an open-weight language model generate text is not the same as making it finish coding work. This retrospective empirical systems experience report follows my engineering work from July to October 2026 across coding clients, model-specific serving recipes, vLLM, SGLang, llama.cpp, and MLX. Models behind a nominally compatible endpoint still differed in tool-call formatting, turn control, context behavior, cache reuse, and reliability. A served model, coherent text, executed tool calls, and code that passes independent behavioral checks proved to be distinct levels of progress. The evidence includes 4,165 coding-agent gateway turns from my own sessions, 623 debug jobs, and three separate delegation studies with hidden behavioral checks and a fixed Claude Opus 5.5 parent. Saved cold-prefix requests had a median time to first token of about 69.4 s, versus 2.8 s for high-reuse requests (observational). In a 45-trial study, direct codi
    
[^54]: 大语言模型与AI智能体的诚实性测试：软件工程的愿景

    Honesty Testing of LLMs and AI Agents: A Vision for Software Engineering

    [https://arxiv.org/abs/2610.34239](https://arxiv.org/abs/2610.34239)

    这篇愿景论文提出将LLM和AI智能体的诚实性测试作为软件工程的新研究领域，把忠实报告确立为可规定、可测试、可维护的软件质量需求，并定义了“诚实性缺陷”概念及初步测试框架。

    

    对大语言模型（LLM）和人工智能（AI）智能体的人类监督日益依赖于它们的报告。与系统自身判断相冲突的陈述可能掩盖失败、妨碍及时干预，并导致严重危害。随着智能体自主性不断增强，忠实报告对于维持人类控制变得愈发重要。在这篇愿景论文中，我们提出将LLM和AI智能体的诚实性测试作为软件工程的一个新研究领域。在现有诚实性评估的基础上，我们的目标是使忠实报告成为开发者可以规定、测试和维护的软件质量需求。我们将诚实性缺陷定义为一种行为缺陷：当LLM或AI智能体被要求忠实沟通时，作出了与其报告时刻自身判断相冲突的明确陈述。我们提出了一个初步框架，涵盖受控输入下的LLM级测试和智能体级测试。

    arXiv:2610.34239v1 Announce Type: new  Abstract: Human oversight of large language models (LLMs) and artificial intelligence (AI) agents increasingly depends on their reports. Claims that conflict with a system's own judgments can conceal failures, prevent timely intervention, and contribute to severe harm. Greater autonomy makes faithful reporting increasingly important for maintaining human control. In this vision paper, we propose honesty testing of LLMs and AI agents as a new research area in software engineering. Building on existing honesty evaluations, our goal is to make faithful reporting a software quality requirement that developers can specify, test, and maintain. We define honesty bugs as behavioural defects in which an LLM or AI agent, when required to communicate faithfully, makes an explicit claim that conflicts with its own judgment at the time of reporting. We present an initial framework covering LLM-level testing under controlled inputs and agent-level testing of co
    
[^55]: Δτ-Bench：推动智能体迈向动态变化的能力边界

    $\Delta\tau$-Bench: Pushing Agents Toward a Moving Capability

    [https://arxiv.org/abs/2610.34238](https://arxiv.org/abs/2610.34238)

    Δτ-Bench是一个模拟工具生态系统持续演化的可执行智能体基准测试，包含覆盖42个领域的1,588个多轮任务，评估显示最强模型的任务成功率也仅达42.3%，揭示了智能体在应对工具环境动态变化方面的重大不足。

    

    本文提出了Δτ-Bench，一个可执行的智能体基准测试，用于建模大型工具生态系统的结构演化。在Δτ-Bench中，各自独立演化的服务不断重塑任务的完成方式：熟悉的工具可能消失或被替换，而更高效的替代方案则会在功能重叠和充满噪声的文档中涌现。面对这一不断移动的能力边界，我们评估智能体能否主动发现高效的工具使用路径，并及时修订其积累的程序性知识，从而汇集了现有基准测试未能考察的多重需求。Δτ-Bench包含1,588个经过验证的多轮任务，覆盖42个领域，以及通过能力保持变换生成的六个连续的生态系统状态。我们的评估显示，即使是最强的被评估模型或框架，其任务成功率也仅为42.3%，而现有方法带来的初始收益……

    arXiv:2610.34238v1 Announce Type: cross  Abstract: In this paper, we introduce $\Delta\tau$-Bench, an executable agent benchmark that models the structural evolution of large tool ecosystems. In $\Delta\tau$-Bench, independently evolving services continually reshape how tasks can be accomplished: familiar tools may disappear or be replaced, while more efficient alternatives emerge amid overlapping functionality and noisy documentation. Against this moving capability frontier, we evaluate whether agents can proactively discover efficient tool-use paths and promptly revise their accumulated procedural knowledge, bringing together demands that existing benchmarks fail to examine. $\Delta\tau$-Bench comprises 1,588 validated multi-turn tasks across 42 domains and six successive ecosystem states generated through capability-preserving transformations. Our evaluation shows that even the strongest evaluated model or harness achieves only 42.3% task success, while initial gains from existing s
    
[^56]: 编程语言生态系统的政策：一项“定性”分析

    The Policies of Programming Language Ecosystems: A "Qualitative" Analysis

    [https://arxiv.org/abs/2610.34187](https://arxiv.org/abs/2610.34187)

    本文对八个编程语言生态系统的政策进行了定性分析，发现各生态系统的政策总体呈现分歧而非趋同，仅有恶意软件、知识产权和内容删除等少数类别获得普遍认可。

    

    编程语言生态系统（PLE），例如R语言的CRAN，极大地加速了软件重用。与此同时，它们也增加了软件供应链的风险与事件。然而，问题并不局限于网络安全；从非法内容与内容审核到骚扰和性内容等各种问题都在许多流行的PLE政策中被提及。本文对这些政策进行了考察。基于三位作者对38个不同政策类别和八个PLE政策评析的适度一致性，这些政策总体上表现出分歧而非趋同。不过，某些政策类别，如涉及恶意软件、知识产权和内容删除的类别，得到了普遍认可。因此，分歧主要体现在更细粒度的政策类别上，例如与包大小限制、指标操纵、仿冒包名抢注（typo-squatting）和二进制上传相关的类别。这些以及其他相关的定性结果……（原文摘要在此处截断）

    arXiv:2610.34187v1 Announce Type: new  Abstract: Programming language ecosystems (PLEs), such as CRAN for R, have greatly accelerated software reuse. At the same time, they have increased software supply-chain risks and incidents. However, the problems are not limited to cyber security; everything from illegal content and content moderation to harassment and sexual content are mentioned in many policies of popular PLEs. The paper examines these policies. According to a moderate agreement between the three authors on 38 distinct policy categories and eight PLE policies, the policies generally exhibit divergence rather than convergence. However, some policy categories, such as those about malware, intellectual property, and content removal, are universally recognized. The divergence is thus seen in more fine-grained policy categories, such as those related to package size limits, manipulation of metrics, typo-squatting, and binary uploads. These and other related qualitative results patc
    
[^57]: SpeC2Rust：规格驱动的仓库级C到Rust翻译

    SpeC2Rust: Specification-Driven Repository-Level C-to-Rust Translation

    [https://arxiv.org/abs/2610.34141](https://arxiv.org/abs/2610.34141)

    SpeC2Rust提出了一种规格驱动的三阶段流水线，通过项目意图结构化生成持久化的仓库级与模块级规格，结合实时符号注册表的规格引导生成和分层反馈修复，实现了仓库级C代码到安全Rust代码的自动翻译。

    

    遗留的C语言代码库仍然是系统软件和性能关键型基础设施的核心，但其薄弱的内存安全性带来了安全和维护方面的风险。近期的C到Rust翻译系统（包括基于大语言模型的翻译器）使得自动化迁移日益可行，但它们仍然难以恢复仓库级的意图、避免使用原始指针和不安全构造、维护跨文件接口，以及高效修复编译和验证失败。我们提出了SpeC2Rust，一种规格驱动的仓库级C到Rust翻译流水线。它包含三个阶段：(1) 项目意图结构化，将C仓库分解为语义连贯的模块，并生成持久的仓库级和模块级规格；(2) 规格引导生成，依据这些规格翻译每个Rust模块，并借助实时符号注册表来保持跨文件一致性；(3) 分层反馈修复。

    arXiv:2610.34141v1 Announce Type: new  Abstract: Legacy C codebases remain central to system software and performance-critical infrastructure, but their weak memory safety imposes security and maintenance risks. Recent C-to-Rust translation systems, including LLM-based translators, make automated migration increasingly feasible, yet they still struggle to recover repository-level intent, avoid raw pointers and unsafe constructs, maintain cross-file interfaces, and repair compilation and validation failures efficiently. We propose SpeC2Rust, a specification-driven pipeline for repository-level C-to-Rust translation. It has three stages: (1) Project-Intent Structuring, which decomposes a C repository into semantically coherent modules and generates persistent repository- and module-level specifications; (2) Spec-Guided Generation, which translates each Rust module against these specifications with a live symbol registry to preserve cross-file consistency; and (3) Layered-Feedback Repair,
    
[^58]: 基于微调开源大语言模型的漏洞严重性分类：在Eclipse报告上与TF-IDF和CodeBERT基线的配对比较

    Bug Severity Classification with Fine-Tuned Open LLMs: A Paired Comparison with TF-IDF and CodeBERT Baselines on Eclipse Reports

    [https://arxiv.org/abs/2610.34127](https://arxiv.org/abs/2610.34127)

    该研究对可在笔记本电脑上运行的开源大语言模型与更便宜的TF-IDF和CodeBERT基线在Eclipse漏洞报告严重性分类任务上进行了配对比较，发现微调后的LLM性能优于传统方法。

    

    足够小可以在笔记本电脑上运行的开源大语言模型（LLMs）是自动漏洞严重性预测的显而易见的候选方案。我们在一个任务上测量了它们相对于更便宜的分类器所带来的增益：将Eclipse漏洞报告分配到四个Bugzilla严重性级别之一（Critical、Major、Minor、Trivial）。从一个包含361,006条唯一Eclipse报告的公开转储中，我们抽取了一个类别平衡的20,000行样本，并比较了九种模型类型的十一种配置：六个TF-IDF分类器、一个微调的CodeBERT编码器，以及Llama 3 8B和Qwen 2.5 7B（每个都以零样本方式和QLoRA微调后运行）。所有十一个模型都在相同的2,000条留出报告上进行评分，使用自助法置信区间和McNemar检验。零样本运行达到宏观F1值0.305和0.321，低于所有TF-IDF模型（0.430-0.489）。微调运行达到0.526和0.485，但零样本和微调设置在检查点、服务运行时和解码方面也存在差异，因此这一差距不能被归因于……

    arXiv:2610.34127v1 Announce Type: new  Abstract: Open large language models (LLMs) small enough to run on a laptop are an obvious candidate for automatic bug severity prediction. We measure what they add over cheaper classifiers on one task: assigning one of four Bugzilla severity levels (Critical, Major, Minor, Trivial) to Eclipse bug reports. From a public dump of 361,006 unique Eclipse reports we draw a class-balanced sample of 20,000 rows and compare eleven configurations of nine model types: six TF-IDF classifiers, a fine-tuned CodeBERT encoder, and Llama 3 8B and Qwen 2.5 7B, each run zero-shot and after QLoRA fine-tuning. All eleven are scored on the same 2,000 held-out reports, with bootstrap intervals and McNemar tests. The zero-shot runs reach macro-F1 0.305 and 0.321, below every TF-IDF model (0.430-0.489). The fine-tuned runs reach 0.526 and 0.485, but the zero-shot and fine-tuned setups also differ in checkpoint, serving runtime and decoding, so this gap cannot be credited
    
[^59]: 利用智能体自身的开发历史保障智能体生成代码的安全

    Securing Agent-Generated Code with Its Own Development History

    [https://arxiv.org/abs/2610.34098](https://arxiv.org/abs/2610.34098)

    TRACER将编码智能体自身的开发历史（编辑轨迹与陈述假设）作为安全审查证据，通过符号级变更跟踪、验证智能体理由、评估并功能验证修复、附加可执行检查，来发现和修复智能体生成代码中被遗漏的安全防护步骤。

    

    由编码智能体编写的拉取请求可能通过项目的测试套件，但同时却遗漏了某个保护性检查或验证步骤。为了找出被遗漏的条件，审查者目前只能依赖对最终代码的静态分析。然而，智能体生成的变更伴随着人类提交所不具备的开发历史：智能体的轨迹记录了每次编辑及其所陈述的假设，而结构化差异分析则展示了每个被修改函数的演变过程。我们提出了TRACER，这是一种将这一开发历史作为安全审查证据的工作流程。TRACER在符号级别跟踪变更，将智能体记录的理由视为需要对照当前代码加以验证的断言，只有在经过评估和功能验证之后才接受修复，并为每个被接受的修复附加一个后续变更必须满足的可执行检查。在SusVibes基准上，符号级别的跟踪将易受攻击的符号置于……

    arXiv:2610.34098v1 Announce Type: new  Abstract: Pull requests written by coding agents can pass a project's test suite while omitting a guard or a validation step. To find the missing condition, a reviewer currently relies on static analysis of the final code. An agent-generated change, however, is accompanied by a development history that a human commit does not provide. The agent's trajectory records each edit together with the agent's stated assumptions, and structural differencing shows how each changed function evolved. We present TRACER, a workflow that uses this development history as evidence for security review. TRACER tracks changes at the symbol level, treats the agent's recorded rationale as claims to be verified against the current code, accepts a repair only after assessment and functional validation, and attaches to each accepted repair an executable check that later changes must satisfy. On the SusVibes benchmark, symbol-level tracking places the vulnerable symbol in t
    
[^60]: 小型编码智能体如何使用代码图与嵌入？

    How Do Small Coding Agents Use Code Graphs and Embeddings?

    [https://arxiv.org/abs/2610.33958](https://arxiv.org/abs/2610.33958)

    研究发现，代码图与嵌入搜索等结构化工具对小型编码智能体的帮助最大——在 1.7B-4B 参数规模可带来 17-21 个百分点的定位准确率提升，但收益随模型规模增大而递减，从而修正了“工具可以补足模型对仓库结构理解”的普遍假设。

    

    编码智能体被赋予嵌入搜索和代码图工具，其前提假设是：仓库的结构信息是模型本身所缺乏、需要由工具来提供的。我们在 SWE-bench Verified 上使用小型开源权重模型检验这一前提，并依据一项配套论文所定义和验证的确定性跳数深度（即从议题中提及的代码到黄金补丁所修改代码之间的最短代码图路径）对任务进行分层。首先，我们在深度均衡的 160 个任务子集以及全部 500 个任务上，对定位智能体的工具集进行消融实验（仅 grep；+嵌入搜索；+图工具；全部），共涉及十个模型（六个规模从 0.6B 到 32B 的 Qwen3，每个规模五种解码方式，一个 FP8 对照，以及三个 Gemma 4 规模；共计 7.9 万个回合）。结构化工具使 Qwen3-8B 和 Gemma 4 E4B 的仅 grep 准确率翻倍；其增益在 1.7B-4B 规模达到 17-21 个百分点的峰值，而在 32B 时降至 4-5 个百分点（参数量每翻倍，几率比为 0.77-0.82）。从 8B 起，该增益中约有一半来自 t……（原文摘要在此处截断）

    arXiv:2610.33958v1 Announce Type: cross  Abstract: Coding agents are given embedding search and code-graph tools on the premise that a repository's structure is something the model lacks and the tool supplies. We test that premise on SWE-bench Verified with small open-weight models, stratifying tasks by a deterministic hop depth (the shortest code-graph path from code the issue mentions to code the gold patch changes) that a companion paper defines and validates. First, we ablate a localization agent's tool set (grep only; +embedding search; +graph tools; all) on a depth-balanced 160-task split and on all 500 tasks for ten models (six Qwen3 sizes from 0.6B to 32B with five decodings each, an FP8 control, three Gemma 4 sizes; 79k episodes). Structured tools double grep-only accuracy for Qwen3-8B and Gemma 4 E4B; their gain peaks at 17-21 points at 1.7-4B and falls to 4-5 points at 32B (odds ratio 0.77-0.82 per doubling of parameters). From 8B up, about half of that gain is the form of t
    
[^61]: 为什么结构化输出会失败？对九个大语言模型中提示词强制执行的Pydantic模式的实证研究

    Why Does Structured Output Fail? An Empirical Study of Prompt-Enforced Pydantic Schemas Across Nine LLMs

    [https://arxiv.org/abs/2610.33949](https://arxiv.org/abs/2610.33949)

    该研究提出了一个细粒度的模式符合性失败分类法和确定性分类器，通过对九个大语言模型配置的3,997次生成实验，揭示了单一有效率指标所掩盖的多样化失败模式，并发现主导失败类别会随模式约束的不同而变化。

    

    需要从托管语言模型获取结构化输出的流水线通常在提示词中指定Pydantic模式，对响应进行验证，并在验证失败时重试。尽管原生结构化输出模式提供了一种替代方案，但应用侧验证对于强制执行自定义约束仍然十分重要。然而，流水线可靠性通常仅用单一的有效率来概括，这掩盖了输出失败的原因以及哪些干预措施可能有效。我们引入了一种细粒度的模式符合性失败分类法，以及一个确定性分类器，该分类器根据模型响应和Pydantic验证错误分配失败类别。我们将此框架应用于九种OpenAI和Anthropic模型/API配置下的3,997次生成尝试，使用来自HotpotQA和FinanceBench的问题。我们的结果表明，相似的有效率可能掩盖不同的失败模式，且主导失败类别随模式约束而变化。

    arXiv:2610.33949v1 Announce Type: cross  Abstract: Pipelines that require structured outputs from hosted language models often specify a Pydantic schema in the prompt, validate the response, and retry when validation fails. Although native structured-output modes provide an alternative, application-side validation remains important for enforcing custom constraints. Yet pipeline reliability is often summarized by a single validity rate, which obscures why outputs fail and which interventions may help. We introduce a fine-grained taxonomy of schema-conformance failures and a deterministic classifier that assigns failure categories from model responses and Pydantic validation errors. We apply this framework to 3,997 generation attempts across nine OpenAI and Anthropic model/API configurations, using questions from HotpotQA and FinanceBench. Our results show that similar validity rates can mask different failure patterns, and that dominant failure categories vary with schema constraints an
    
[^62]: DRILS：基于依赖关系的大语言模型需求变更影响分析

    DRILS: Dependency-based Requirement Change Impact Analysis Using LLMs

    [https://arxiv.org/abs/2610.33920](https://arxiv.org/abs/2610.33920)

    DRILS是一个基于大语言模型的多智能体需求变更影响分析框架，通过RE-RAG检索增强技术提供聚焦领域上下文，并利用LLM-as-a-judge机制促进候选影响集的多样性，从而自动准确地识别需求变更的影响范围。

    

    变更在软件系统中不可避免，需求必须持续适应不断演化的利益相关者需求、法规和项目范围。管理此类变更是软件工程中的一项根本性挑战，因为在需求量大、自然语言复杂且相互依赖关系错综复杂的情况下，人工识别变更请求的影响既容易出错又耗时。大语言模型为需求变更影响分析（CIA）提供了有前景的解决方案，但其在结合先进技术增强后的有效性仍有待深入探索。我们提出了DRILS，一个基于大语言模型的智能体框架，用于需求变更影响分析，该框架采用多智能体架构生成候选影响集，并通过LLM作为评判者机制在选择过程中促进多样性。DRILS包含两个关键组件：RE-RAG，一种精心设计的检索增强生成技术，可提供聚焦的领域上下文；以及一个需求……（摘要在此处被截断）

    arXiv:2610.33920v1 Announce Type: new  Abstract: Change is inevitable in software systems, where requirements must continuously adapt to evolving stakeholder needs, regulations, and project scopes. Managing such changes is a fundamental software engineering challenge, as manually identifying the impact of a change request, given high requirements volume, Natural Language (NL) complexity, and intricate interdependencies, is error-prone and time-consuming. Large Language Models (LLMs) offer promising solutions for requirements change impact analysis (CIA), yet their effectiveness when enhanced with advanced techniques remains underexplored. We present DRILS, an LLM-based agentic framework for requirements Change Impact Analysis, employing a multi-agent architecture to generate candidate impact sets, with an LLM-as-a-judge promoting diversity during selection. DRILS features two key components: RE-RAG, a carefully engineered RAG technique providing focused domain context, and a requiremen
    
[^63]: 计算化评估堆栈：一种面向AI原生测量的架构

    The Computational Assessment Stack: An Architecture for AI-Native Measurement

    [https://arxiv.org/abs/2610.33901](https://arxiv.org/abs/2610.33901)

    本文提出计算化评估堆栈（CAS），一种以机器可读、模块化、可版本化的统一计算构念表示（知识图谱）为基础的四层AI原生测量架构，通过计算化内容生成、计算心理测量学与编排层，在智能体AI主导测试开发的新范式下保障测评质量。

    

    测试开发正在向自动化和智能体人工智能转变，在这一范式下，自主系统能够大规模起草试题、组装试卷、实施自适应测评、对复杂作答进行评分并监控测试质量。随着自主系统承担更多此类工作，测试开发的基础架构必须随之改变，以便在新范式下保持测评质量。本文提出了计算化评估堆栈（CAS），这是一种面向AI原生测量的四层架构，评估生命周期的每个阶段都共享一个统一的计算化构念表示作为其底层。该表示是一个机器可读、模块化且可版本管理的知识图谱。在其之上构建了三层：基于知识图谱的计算化内容生成层；直接从知识图谱推导实质结构的计算心理测量学层；以及一个编排层……

    arXiv:2610.33901v1 Announce Type: cross  Abstract: Test development is shifting toward automation and agentic artificial intelligence (AI), in which autonomous systems draft items at scale, assemble test forms, administer adaptive assessments, score complex responses, and monitor test quality. As autonomous systems take on more of this work, the architecture underlying test development must change to preserve assessment quality under the new paradigm. This paper proposes the Computational Assessment Stack (CAS), a four-layer architecture for AI-native measurement where every stage of the assessment lifecycle shares a unifying computational construct representation as its base layer. This representation is a machine-readable, modular, and versioned knowledge graph. Three layers are built above it: computational content generation that builds on the knowledge graph; computational psychometrics that derives substantive structure directly from the knowledge graph; an orchestration layer in
    
[^64]: FlashScript：将重复出现的智能体工作流编译为可复用的神经符号程序

    FlashScript: Compiling Recurring Agent Workflows into Reusable Neuro-Symbolic Programs

    [https://arxiv.org/abs/2610.33876](https://arxiv.org/abs/2610.33876)

    FlashScript 提出一套将成功的智能体运行编译为可复用神经符号程序的准则——把动作与数据转换变为代码、实例特定值变为参数、需模型判断的决策变为局部化的 LLM 调用——从而将非确定性局部化，使确定性程序与神经调用可分别审查和测试，显著降低重复任务的成本与结果波动。

    

    许多智能体任务会随着新数据或微小变化而重复出现，然而智能体在每次运行时都会重新生成推理和动作，产生不一致的结果。将先前的运行提炼成技能虽然能帮助智能体避免之前的错误，但每次仍需重新执行完整流程，导致运行结果多变并产生相应成本。我们提出一套称为 FlashScript 的准则，将成功的智能体运行编译为可复用的神经符号程序（NSP）：动作和数据转换变成代码，特定实例的值变成参数，而需要模型判断的决策变成 LLM 调用。我们识别出这些调用所属的三个层次，以及三种用代码替代其中大部分调用的技术。这种编译并未消除非确定性，而是将其局部化，使确定性程序与神经调用能够被分别审查和测试。在使用 Claude Haiku 4.5 于 120 个 WorkBench 分析任务上的实验中，这些 NSP 表现正确并做出了相同的工具调用（原文摘要在此处截断）。

    arXiv:2610.33876v1 Announce Type: cross  Abstract: Many agent tasks recur with new data or small changes, yet agents regenerate their reasoning and actions on each run, producing different results. Distilling a prior run into a skill helps the agent avoid earlier mistakes, but it still re-runs the full process each time, so runs can vary and incur that cost. We compile a successful agent run into a reusable neuro-symbolic program (NSP) with a set of guidelines we call FlashScript. Actions and data transformations become code, instance-specific values become parameters, and decisions that need model judgment become LLM calls. We identify three layers where these calls belong and three techniques that replace most of them with code. The compilation does not remove nondeterminism; it localizes it, so the deterministic program and its neural calls can be reviewed and tested separately. Across 120 WorkBench analytics tasks with Claude Haiku 4.5, the NSPs were correct and made identical tool
    
[^65]: Numina Fuse：Lean 4 中的交互式自动形式化

    Numina Fuse: Interactive Autoformalization in Lean 4

    [https://arxiv.org/abs/2610.33860](https://arxiv.org/abs/2610.33860)

    Numina Fuse 是一个用于 Lean 4 数学自动形式化的智能体原生应用，其核心创新在于以人类监督为中心，通过蓝图、依赖图和代码等视图让用户监督 AI 编排器的工作，并采用将项目状态与编排逻辑解耦的架构，使智能体能随模型进步逐步自主接管更多形式化工作。

    

    Numina Fuse 是一个智能体原生（agent-native）的应用程序，用于在 Lean 4 中进行数学形式化。由于 Lean 代码可以通过编译却并不表达其预期的非形式化含义，而且如今智能体编写 Lean 代码的速度已经超过了人类阅读的速度，我们围绕“人类监督智能体工作”这一理念构建了 Fuse。用户与一个 AI 编排器共同协作一个 Lean 项目，并通过蓝图视图、依赖关系图视图和 Lean 代码视图对其进行监督，而后端则维护项目的派生状态，并在用户不在场的情况下让智能体持续运行数小时。我们描述了 Fuse 在 2026 年间的开发历程：它从一条由人工审查把控节奏的预设智能体流水线，演进为一个能够自主调度专家智能体、证明器智能体和并行子智能体的编排器。我们为这一转变进行了前瞻性设计，将项目状态、工具和界面设计与编排逻辑相互解耦，以便随着模型的改进，智能体能够接管越来越多的工作。我们报告了 Fuse 在一次形式化（原文摘要在此处被截断）……

    arXiv:2610.33860v1 Announce Type: new  Abstract: Numina Fuse is an agent-native application for formalizing mathematics in Lean 4. Since Lean can compile while not representing its intended informal meaning, and agents now write Lean faster than people can read it, we built Fuse around human supervision of agent work. A user works with an AI orchestrator on a Lean project and supervises it through views of the blueprint, the dependency graph, and the Lean code, while the backend keeps the project's derived state and runs agents for hours without the user present. We describe the development of Fuse throughout 2026, as it moved from a prescribed pipeline of agents paced by human review to an orchestrator that dispatches specialists, provers, and parallel children on its own. We designed for this shift, keeping project state, tooling, and interface design separate from orchestration so that agents could take over more of the work as models improved. We report Fuse's use in a formalizatio
    
[^66]: ArchGuard-APR：基于仓库图谱与执行反馈的架构约束感知多智能体程序修复

    ArchGuard-APR: Architecture-Constraint-Aware Multi-Agent Program Repair Using Repository Graphs and Execution Feedback

    [https://arxiv.org/abs/2610.33853](https://arxiv.org/abs/2610.33853)

    ArchGuard-APR 提出了一个架构约束感知的多智能体程序修复原型，通过不可变的仓库图谱、显式架构配置与隔离执行反馈来引导和约束补丁生成，使修复结果不仅通过测试，还不违反架构边界。

    

    仓库级自动化程序修复所需考量的远不止一处局部化的失败语句。一个候选变更可能在通过狭窄测试的同时违反依赖边界、修改不安全的路径，或依赖于遗漏了相关仓库结构的上下文。ArchGuard-APR 是一个架构感知的修复原型，它集成了确定性的仓库分析、不可变的仓库图谱、显式的架构配置、图谱引导的上下文检索、基于角色的修复编排、补丁策略执行、隔离的执行反馈、提供商成本控制以及规范的免源码证据。我们通过一项探索性的配对可行性研究（而非验证性的有效性实验）来评估所实现的流水线。原计划执行六次；其中四次完成，两次 black-4 条件在进入提供商阶段前因基础设施损耗而终止。在两个有效的配对任务中，基线条件与 ArchGuard 条件各自产……（原文摘要在此截断）

    arXiv:2610.33853v1 Announce Type: new  Abstract: Repository-level automated program repair must reason about more than a localized failing statement. A candidate change can satisfy a narrow test while violating dependency boundaries, modifying an unsafe path, or relying on context that omits relevant repository structure. ArchGuard-APR is an architecture-aware repair prototype that integrates deterministic repository analysis, immutable repository graphs, explicit architecture profiles, graph-guided context retrieval, role-based repair orchestration, patch-policy enforcement, isolated execution feedback, provider cost controls, and canonical source-free evidence.   We evaluate the implemented pipeline through an exploratory paired feasibility study, not a confirmatory effectiveness experiment. Six executions were planned; four completed, and two black-4 conditions ended in pre-provider infrastructure attrition. Across two valid paired tasks, baseline and ArchGuard conditions each produ
    
[^67]: 软件工程大语言模型：为什么我们需要它们，以及软件工程如何助力构建它们

    Software Engineering LLMs: Why We Need Them and How Software Engineering Helps Build Them

    [https://arxiv.org/abs/2610.33851](https://arxiv.org/abs/2610.33851)

    软件工程专业知识可有效指导编码大语言模型的能力选择、可验证训练数据构建与评估，本工业案例通过RefactorForge流水线为仓库级重构生成了4,572条经验证的教师轨迹，并以极小的数据成本微调了一个服务10万余名开发者的920亿参数编码模型。

    

    软件工程（SE）专业知识可以塑造编码大语言模型（LLMs）的训练方式以及它们所获得的工程能力。我们通过对一个服务超过10万名开发者的920亿（92B）参数编码大语言模型进行能力导向训练的工业案例研究，来考察这一角色。我们展示了成熟的软件工程知识如何指导工程能力的选择、可验证训练数据的构建，以及模型改进的评估。为了实现仓库级重构这一目标能力，我们开发了RefactorForge——一条结合重构知识、成熟软件工程工具和验证技术的流水线，用于构建训练数据。在306个Python仓库上，RefactorForge生成了5,704个可执行任务，并保留了4,572条经过验证的教师轨迹，共计7,251万个token。基于这些轨迹进行的增量式监督微调，仅占先前编码混合数据总量的0.46%……

    arXiv:2610.33851v1 Announce Type: new  Abstract: Software engineering (SE) expertise can shape how coding large language models (LLMs) are trained and which engineering capabilities they acquire. We examine this role through an industrial case study of capability-focused training for a 92B-parameter coding LLM serving more than 100,000 developers. We demonstrate how established SE knowledge can guide the selection of engineering capabilities, the construction of verifiable training data, and the evaluation of model improvements. To target repository-level refactoring, we developed RefactorForge, a pipeline that combines refactoring knowledge, established SE tools, and verification techniques to construct training data. Across 306 Python repositories, RefactorForge generated 5,704 executable tasks and retained 4,572 verified teacher trajectories totaling 72.51 million tokens. Incremental supervised fine-tuning on these trajectories, accounting for 0.46% of the preceding coding mixture's
    
[^68]: VulDec：基于神经反编译的二进制漏洞检测

    VulDec: Vulnerability Detection in Binaries via Neural Decompilation

    [https://arxiv.org/abs/2610.33716](https://arxiv.org/abs/2610.33716)

    VulDec提出了一个大规模、带多架构CWE详细标注的汇编-C函数对齐数据集，并首次构建了利用LLM神经反编译技术实现二进制漏洞检测的端到端框架。

    

    二进制可执行文件中的自动化漏洞检测极具挑战性，原因在于源代码的缺失以及汇编层面可用的语义信息有限。尽管基于机器学习的代码分析已取得重大进展，该领域仍然缺乏相应的方法以及大型高质量数据集——即能够在汇编级和源级同时提供易受攻击代码并带有精确漏洞标签的数据集。为了填补这一空白，我们提出了VulDec，其包含两部分：（i）一个新的大规模、精心整理的数据集，将真实世界的汇编-C函数对与跨多种架构的详细CWE注释进行对齐；（ii）首个利用神经反编译技术进行二进制漏洞检测的端到端框架。在我们的数据集上微调的基于大语言模型（LLM）的神经反编译器在多种架构上实现了较强的源码级恢复能力（编辑相似度约为56%，BLEU-4约为43%），而我们微调的基于LLM的反编译器……

    arXiv:2610.33716v1 Announce Type: cross  Abstract: Automated vulnerability detection in binary executables is challenging due to the lack of source code and the limited semantic information available at the assembly level. Despite significant progress in machine-learning-based code analysis, the field still lacks methodologies and large, high-quality datasets that provide vulnerable code jointly at assembly- and source-level with precise vulnerability labels. To address this gap, we introduce VulDec, consisting of (i) a new large-scale, carefully curated dataset aligning real-world Assembly-C function pairs with detailed CWE annotations across multiple architectures, and (ii) the first end-to-end framework that leverages neural decompilation for binary vulnerability detection. Tuned LLM-based neural decompilers trained on our dataset achieve strong source-level recovery across architectures (Edit Similarity of approximately 56%, BLEU-4 of approximately 43%), while our tuned LLM-based d
    
[^69]: 覆盖不等于检验：你的测试能注意到哪些分支结果？

    Covered Is Not Checked: Which Branch Outcomes Would Your Tests Notice?

    [https://arxiv.org/abs/2610.33708](https://arxiv.org/abs/2610.33708)

    提出无需切片的Veritas方法，通过强制相反的分支结果并验证原本通过的测试是否因此失败，来判断被覆盖的分支结果是否真正被测试检验，从而无需动态依赖分析即可区分代码执行与实际缺陷检测能力。

    

    代码覆盖率报告的是测试执行了什么，而非测试能否检测出缺陷。检验覆盖通过动态切片追踪影响测试判定基准的代码来弥补这一差距，但切片成本高昂，且Java切片器难以处理现代语言特性和复杂程序。我们提出了Veritas，一种无需切片的方法，可直接测试被覆盖的行为是否真正被检验。我们针对对象分支覆盖（OBC），这是一种基于编译代码的结构化准则，其缺陷检测效果可与源代码级的修正条件/判定覆盖（MC/DC）相媲美，后者是DO-178C最高保证等级所要求的标准。对于每个被覆盖的对象分支结果，Veritas强制其产生相反的结果，并且只有当该变化被执行、原本通过的测试失败、且该失败可归因于该结果时，才将其计入检验。因此，Veritas无需动态依赖分析即可区分代码执行与已证实的缺陷检测。

    arXiv:2610.33708v1 Announce Type: new  Abstract: Code coverage reports what tests execute, not whether tests detect faults. Checked coverage addresses this gap by tracing code that influences test oracles through dynamic slicing, but slicing is costly, and Java slicers struggle with modern language features and complex programs. We present Veritas, a slicing-free approach that directly tests whether covered behavior is checked. We target object branch coverage (OBC), a compiled-code structural criterion with fault-detection effectiveness comparable to source-level Modified Condition/Decision Coverage (MC/DC), which is required at the highest assurance level of DO-178C. For each covered object-branch outcome, Veritas forces the opposite outcome and credits checking only when the change is executed, a previously passing test fails, and the failure is attributable to that outcome. Thus, Veritas distinguishes execution from demonstrated fault detection without dynamic dependence analysis. 
    
[^70]: SWE-Ladder：通过渐进式特权自蒸馏学习定位、复现与修补

    SWE-Ladder: Learning to Localize, Reproduce, and Patch via Progressive Privileged Self-Distillation

    [https://arxiv.org/abs/2610.33667](https://arxiv.org/abs/2610.33667)

    该论文提出SWE-Ladder框架，将已验证的修复过程分解为定位、复现和修补三个阶段的特权证据，通过基于规则的路由器和冻结教师模型按修复顺序进行渐进式自蒸馏，并结合GRPO优化完整轨迹，为SWE智能体提供了传统强化学习和OPSD所缺失的阶段特定训练指导。

    

    解决仓库级问题通常需要软件工程（SWE）智能体定位故障代码、复现失败并进行代码修补。强化学习通过最终补丁的单一测试结果来训练此类智能体，而策略内自蒸馏（OPSD）则从拥有特权证据的教师模型中添加token级监督，在每一步都暴露完整的参考补丁。然而，两者都无法为定位或复现阶段提供针对性的指导。我们提出SWE-Ladder，将每个训练实例的已验证修复转化为特定阶段的证据：用于定位的变更文件和行号、用于复现的fail-to-pass测试，以及用于修补的代码差异。基于规则的路由器根据工具调用将端到端rollout的每一步分配到相应阶段，冻结的教师模型只接收与该阶段匹配的证据，各阶段按修复顺序依次进行蒸馏，同时由GRPO优化完整轨迹。使用Qw…（原文摘要截断）

    arXiv:2610.33667v1 Announce Type: new  Abstract: Resolving a repository-level issue typically requires a software-engineering (SWE) agent to localize the faulty code, reproduce the failure, and patch the code. Reinforcement learning trains such agents from a single test result on the final patch, while on-policy self-distillation (OPSD) adds token-level supervision from a teacher with privileged evidence, exposing the full reference patch at every step. Neither provides stage-specific guidance for localization or reproduction. We propose SWE-Ladder, which turns the verified repair of each training instance into stage-specific evidence: the changed files and lines for localization, the fail-to-pass tests for reproduction, and the diff for patching. A rule-based router assigns each step of an end-to-end rollout to a stage based on its tool call, a frozen teacher sees only the matching evidence, and the stages are distilled in repair order while GRPO optimizes the full trajectory. With Qw
    
[^71]: 自建还是采用？面向汽车GUI测试的定制化与通用LLM智能体框架对比

    Build or Adopt? Custom-Built vs. General-Purpose LLM Agent Harnesses for Automotive GUI Testing

    [https://arxiv.org/abs/2610.33636](https://arxiv.org/abs/2610.33636)

    该研究在沃尔沃集团对比了自建LLM智能体框架与通用框架（Claude Code、OpenCode）在车载信息娱乐系统GUI测试中的表现，发现通用框架成功率相当但成本更高、波动更大，然而其节省的工程投入足以弥补成本劣势。

    

    随着通用LLM智能体框架日趋成熟，组织必须决定是继续开发内部系统还是采用现有的替代方案。我们在沃尔沃集团针对基于Android的车载信息娱乐（IVI）系统的GUI测试研究了这一决策。我们将一个生产环境中的自建框架与两个通用框架（Claude Code和OpenCode）进行比较，同时保持通过模型上下文协议访问的工具和所使用的模型一致。我们使用多达四种模型，在沃尔沃的IVI测试用例以及公开应用（Google Maps、Booking.com）中的开放式任务上对这些框架进行了评估，并采访了十位从业者，包括开发人员、产品负责人和测试人员。在沃尔沃的分步式IVI测试中，通用框架的成功率与自建框架相当，但在不同运行之间的波动更大。它们的单任务推理成本高出2.6至10.7倍，但其节省的工程工作量足以抵消这一成本：对于夜间IVI回归测试……

    arXiv:2610.33636v1 Announce Type: cross  Abstract: As general-purpose LLM agent harnesses mature, organizations must decide whether to continue developing in-house systems or adopt existing alternatives. We study this decision at Volvo Group for GUI testing of Android-based in-vehicle infotainment (IVI) systems. We compare a production, custom-built harness with two general-purpose harnesses, Claude Code and OpenCode, keeping the tools, accessed through the Model Context Protocol (MCP), and the models the same. We evaluate the harnesses with up to four models on Volvo's IVI test cases and on open-ended tasks in public apps (Google Maps, Booking.com), and interview ten practitioners: developers, the product owner, and testers. On Volvo's stepwise IVI tests, general-purpose harnesses match the custom-built harness in success but vary more across runs. Their inference cost per task is 2.6-10.7 times higher, yet the engineering effort they save outweighs this: for a nightly IVI regression 
    
[^72]: 开发者作为AI智能体的管理者：一项工作场所跟随观察研究

    Developers as Managers of AI Agents: A Workplace Shadowing Study

    [https://arxiv.org/abs/2610.33588](https://arxiv.org/abs/2610.33588)

    该研究通过对11名专业开发者的工作场所跟随观察，首次系统揭示了开发者在日常工作中管理自主AI编程智能体的三种模式——监督者、委派者和学习者，并发现开发者约三分之一的时间用于编写提示词和检查AI响应。

    

    随着AI编程智能体变得更加自主，开发者越来越多地需要指导、协调和检查它们的工作。然而，关于这些实践在日常专业开发中如何展开，目前知之甚少。我们对11名专业开发者进行了约15小时的工作跟随观察，对他们的活动和任务切换进行了编码，分析了255条提交的提示词，并开展了4.6小时的后续访谈。开发者平均将33.4%的观察时间用于编写提示词或检查AI响应，任务切换率在每小时0次到101.6次之间。除了产出工件之外，提示词还承担着促进理解、澄清需求、配置智能体以及协调沟通等功能。我们识别出三种反复出现的行为模式：*监督者*（Supervisor），即直接检查AI产出的工件；*委派者*（Delegator），即通过对话进行协调并将实现与检查工作委派给智能体；*学习者*（Learner），即主要借助智能体来构建……（原文摘要在此处截断）

    arXiv:2610.33588v1 Announce Type: new  Abstract: As AI coding agents become more autonomous, developers increasingly direct, coordinate, and check their work. Yet little is known about these practices in everyday professional development. We shadowed 11 professional developers across approximately 15 hours of work, coded their activities and task transitions, analyzed 255 submitted prompts, and conducted 4.6 hours of follow-up interviews.   Developers spent an average of 33.4% of observed time formulating prompts or checking AI responses, with task-switching rates ranging from 0 to 101.6 switches per hour. Beyond producing artifacts, prompts supported understanding, requirements clarification, agent configuration, and coordination. We identified three recurring patterns: \textit{Supervisor}, involving direct artifact inspection; \textit{Delegator}, involving dialogue-based coordination and delegated implementation and checking; and \textit{Learner}, involving agents primarily to build 
    
[^73]: 智能体代码审查能否在合并前捕获漏洞？

    Can Agentic Code Review Catch Vulnerabilities Before They Are Merged?

    [https://arxiv.org/abs/2610.33509](https://arxiv.org/abs/2610.33509)

    提出了SecPRBench——首个基于经过验证的真实漏洞引入提交来评估智能体安全代码审查的综合性基准，涵盖Firefox、V8、OpenSSL等九个关键系统的265个任务，并发现八个前沿模型的召回率仅为32.5%至78.1%。

    

    LLM智能体已被越来越多地用于以安全为核心的代码审查，但现有基准大多从后续的修复补丁中提取被审查的代码变更，而非真实的漏洞引入提交。在本文中，我们提出了SecPRBench，这是首个用于在经过验证的真实世界代码变更上评估智能体安全审查的综合性基准。更具体地说，我们构建了一个自动化智能体流水线，能够自主收集、构建并验证面向热门软件系统的大量真实漏洞引入提交。为确保评估的精确性，我们还采用LLM-as-a-judge（大语言模型作为裁判）对与真实漏洞匹配的发现给予评分。迄今为止，我们已从九个关键系统（包括拥有数百万行代码的Firefox、V8和OpenSSL）中收集了265个任务。在八个前沿模型上的评估显示，召回率介于32.5%至78.1%之间，其中Grok 4.6和GPT-6 sol位居前列。我们的研究还表明，大多数漏检情况并

    arXiv:2610.33509v1 Announce Type: cross  Abstract: LLM agents have been increasingly adopted for security-focused code review, but existing benchmarks mostly derive the reviewed changes from their later fixes instead of the real-world bug-introducing commits. In this paper, we present SecPRBench, the first comprehensive benchmark for evaluating agentic security review on validated real-world code changes. More specifically, we have built an automated agentic pipeline to autonomously harvest, build, and validate massive real-world bug-introducing commits for popular software systems. To ensure precise evaluation, we also adopt LLM-as-a-judge to credit the findings matching ground-truth vulnerabilities. To date, we have collected 265 tasks from nine critical systems, including Firefox, V8, and OpenSSL with millions of lines of code. Across eight frontier models, recall ranges from 32.5% to 78.1%, with Grok 4.6 and GPT-6 sol ranking at the top. Our study also shows that most misses are no
    
[^74]: 基于自我评估的安全且功能正确的代码生成

    Secure and Functional Code Generation with Self-Evaluation

    [https://arxiv.org/abs/2610.33450](https://arxiv.org/abs/2610.33450)

    SecGent是一个推理时框架，通过从CWE数据库检索弱点上下文提供任务特定的安全指导，并结合功能与安全测试的自我评估反馈，在不牺牲代码功能性的前提下提升生成代码的安全性。

    

    大型语言模型（LLM）能够生成功能正确但仍不安全的代码。以安全为核心的微调成本高昂且并非总是可行，这促使人们探索推理时（inference-time）方法。面向安全的提示和检索可以减少漏洞，但也可能移除所需的行为，或应用与任务不兼容的防御措施。挑战在于如何识别相关的安全知识，并利用它在不牺牲功能性的前提下提升安全性。我们提出了SecGent，这是一个推理时框架，它从通用缺陷枚举（CWE）数据库中检索弱点上下文并将其应用于安全代码生成。通过将任务规范与相关的CWE文本记录关联起来，SecGent提供针对特定任务的安全指导，而无需依赖检索到的易受攻击或已修补的代码示例。互补的功能测试和安全测试对生成的实现进行检查，其执行反馈指导……（原文在此处截断）

    arXiv:2610.33450v1 Announce Type: cross  Abstract: Large Language Models (LLMs) can generate functionally correct code that remains insecure. Security-focused fine-tuning can be expensive and is not always available, motivating inference-time approaches. Security-oriented prompting and retrieval can reduce vulnerabilities, but may also remove required behavior or apply defenses incompatible with the task. The challenge is to identify relevant security knowledge and use it to improve security without sacrificing functionality. We present SecGent, an inference-time framework that retrieves weakness context from the Common Weakness Enumeration (CWE) database and applies it to secure code generation. By connecting task specifications to relevant textual CWE records, SecGent provides task-specific security guidance without relying on retrieved vulnerable or patched code examples. Complementary functional and security tests check the resulting implementation, and their execution feedback gui
    
[^75]: BoardSafety：基于制造商文档构建的面向开发板安全的AI就绪数据

    BoardSafety: AI-Ready Data from Manufacturer Documentation for Development-Board Safety

    [https://arxiv.org/abs/2610.33448](https://arxiv.org/abs/2610.33448)

    提出了BoardSafety基准，通过将厂商数据手册和硬件指南转换为板卡特定的安全规则库，构建了带标注的操作场景数据集，用于评估大语言模型判断开发板操作安全性的能力。

    

    大语言模型（LLM）的发展使其能够辅助硬件实验。在此背景下，产生了一个实际问题：LLM能否判断某项拟议操作对特定开发板是否安全？文档中记载的操作限制和安全要求是针对特定板卡的，即使是同一制造商的不同产品之间也存在差异。这种差异性使得LLM进行板卡特定的安全判断难以评估。为解决这一问题，我们提出了BoardSafety基准。该基准基于厂商数据手册和硬件指南构建。在数据转换方面，这些文档被转换为针对特定板卡的安全规则库，并随后用于创建带标注的操作场景。每个数据项将一块开发板和一个拟议操作与适用的安全规则及真实标签配对，用于指示安全等级。每个操作的安全等级分类包括：安全、不安全或信息不足（原文摘要在此处截断）。

    arXiv:2610.33448v1 Announce Type: new  Abstract: The development of large language models (LLMs) are capable to assist with hardware experiments. In this scenario, this raises a practical question: Can LLMs determine whether a proposed operation is safe for a specific development board? Documented operating limits and safety requirements are board-specific, even across products from the same manufacturer. This variation makes board-specific safety judgments by an LLM difficult to evaluate. To address this problem, we introduce the BoardSafety Benchmark. The benchmark is built from vendor datasheets and hardware guides. For data transformation, these documents are converted into a board-specific safety rule base and are then used to create labeled operating scenarios. Each data item pairs a board and a proposed operation to an applicable safety rule with ground-truth label, indicating the level of safety. The categorized level for each operation includes safe, unsafe, or insufficient-in
    
[^76]: OneCodeReviewBench：面向代码审查与搜索的类型化决策基准

    OneCodeReviewBench: Typed Decisions for Code Review and Search

    [https://arxiv.org/abs/2610.33389](https://arxiv.org/abs/2610.33389)

    该论文提出OneCodeReviewBench基准，包含覆盖12种编程语言和YAML的1,968个具有类型化答案（布尔、类别、有序）的合成代码审查案例，并通过十个模型-适配器系统的评测揭示了当前领先模型在代码审查决策任务上的性能水平。

    

    代码审查需要在多种编程语言和软件栈中发现功能性缺陷、内存安全错误、安全弱点和性能问题。工具可以将这项工作中边界明确的部分表达为对给定代码的决策。我们提出OneCodeReviewBench，包含覆盖12种编程语言和YAML的1,968个合成案例，答案类型包括布尔型、类别型和有序型。明确的契约规定了运行时假设、源代码范围和答案评判标准；独立的证据记录支持参考审查。一位作者审查了150个案例，另外两名工程师审查了其中的100个案例。我们在相同的案例和契约上比较了十个模型-适配器系统：OpenAI Decisions达到81.3%的布尔型平衡准确率；Kev-27B达到76.8%的选择准确率和0.367的评分平均绝对误差。在排除两个契约不完整的模型家族后，这些领先系统依然保持优势。额外的比较涵盖586个使用原始符号搜索的案例以及40个……（原文摘要在此处截断）

    arXiv:2610.33389v1 Announce Type: new  Abstract: Code review requires finding functional bugs, memory-safety errors, security weaknesses, and performance problems across languages and software stacks. Tools can express bounded parts of this work as decisions about supplied code. We present OneCodeReviewBench, 1,968 synthetic cases across 12 programming languages and YAML, with Boolean, categorical, and ordered answers. Explicit contracts specify runtime assumptions, source scope, and answer criteria; separate evidence records support reference review. One author reviewed 150 cases; two additional engineers reviewed 100 of those cases. We compare ten model--adapter systems on the same cases and contracts: OpenAI Decisions reaches 81.3% Boolean balanced accuracy; Kev-27B reaches 76.8% Choice accuracy and Score mean absolute error of 0.367. Excluding two families with incomplete contracts preserves these leaders. Additional comparisons cover 586 cases with raw symbol search and 40 with a 
    
[^77]: PMU基础模型桥接器：一种将同步相量数据流连接到时间序列基础模型的工具

    PMU Foundation Model Bridge: A Tool for Connecting Synchrophasor Streams to Time Series Foundation Models

    [https://arxiv.org/abs/2610.33387](https://arxiv.org/abs/2610.33387)

    本文提出了一个开源工具PMU基础模型桥接器，能够将PMU/PDC同步相量数据流连接到多种公开可用的时间序列基础模型，并支持基于C37.118的在线工作流和离线分析。

    

    本文介绍了PMU基础模型桥接器，这是一个开源工具，可将在线相量测量单元（PMU）或相量数据集中器（PDC）数据流连接到公开可用的时间序列基础模型（TSFM）。用户可以通过桌面界面或命令行界面选择测量通道、定义预处理、选择TSFM并配置输出。在初始版本中，该工具支持4个公开可用的时间序列基础模型，包括Chronos-2、TimesFM 2.5、Timestral和TimesFM-3。该工具支持文件输出和同步相量流式输出，因此既可用于离线使用，也可集成到基于C37.118的在线工作流应用中。除了支持原始同步相量数据外，该框架还提供了一个预处理层，用于推导用于预测的定制附加量。该工具通过虚拟PMU和来自频率扰动记录仪（FDR）的实时实验室测量数据进行了验证。

    arXiv:2610.33387v1 Announce Type: cross  Abstract: This paper presents PMU Foundation Model Bridge, an open source tool that connects online phasor measurement unit (PMU) or phasor data concentrator (PDC) streams to publicly available time series foundation models (TSFM). Through a desktop interface or command line interface, users can select measurement channels, define preprocessing, choose TSFMs, and configure outputs. In the initial release, 4 publicly available TSFMs were supported, including Chronos-2, TimesFM 2.5, Timestral, and TimesFM-3. The tool supports both file and synchrophasor streaming outputs, therefore enabling both offline use and integration into C37.118-based online workflow applications. In addition to supporting raw synchrophasor data, the framework provides a preprocessing layer for deriving customized additional quantities for forecasting. The tool is validated using a virtual PMU and real-time laboratory measurements from a frequency disturbance recorder (FDR)
    
[^78]: 面向AI辅助代码演进的EVM锚定溯源：一项冒烟规模部署报告

    EVM-Anchored Provenance for AI-Assisted Code Evolution: A Smoke-Scale Deployment Report

    [https://arxiv.org/abs/2610.33350](https://arxiv.org/abs/2610.33350)

    该论文报告了一个面向AI辅助代码演进的EVM锚定溯源系统的冒烟规模部署，通过MCP服务器、Solidity合约、IPFS固定与可重建的SQLite索引，为AI生成的代码状态流提供链上存在性、排序性和不可否认性证明，并明确界定了锚定不证明真实性的边界。

    

    我们报告了一项针对AI辅助代码生成所产生的状态流的EVM锚定溯源框架的冒烟规模部署。该系统由两部分组成：codebase-state-manager，一个开源的模型上下文协议（MCP）服务器（MIT许可，v0.2.1版本），可向任何兼容MCP的助手暴露代码库状态、转换和奖励操作；以及CodebaseStateRegistry，一个部署在Base Sepolia测试网上的Solidity合约，用于发出每个状态的承诺事件。数据载荷被固定到星际文件系统（IPFS）上，并通过一个可通过重放合约事件来重新同步的可重建SQLite镜像进行索引。我们精确界定了锚定能证明什么和不能证明什么（提交者密钥的存在性、排序性以及不可否认性；绝不证明真实性或防止伪造），并概述了弥合这一差距所需的正交供应链控制措施。在本地Anvil模拟器上作者生成的合成工作负载下，中位数提交……（摘要在此处截断）

    arXiv:2610.33350v1 Announce Type: new  Abstract: We report a smoke-scale deployment of an EVM-anchored provenance framework for the state stream produced by AI-assisted code generation. The system comprises codebase-state-manager, an open-source Model Context Protocol (MCP) server (MIT, v0.2.1) that exposes codebase state, transition, and reward operations to any MCP-compatible assistant, and CodebaseStateRegistry, a Solidity contract deployed on Base Sepolia that emits per-state commitment events. Payloads are pinned to the InterPlanetary File System (IPFS) and indexed through a rebuildable SQLite mirror that resynchronizes by replaying contract events. We delimit precisely what anchoring does and does not attest (existence, ordering, and non-equivocation by the committer key; never truthfulness or fabrication-prevention) and outline the orthogonal supply-chain controls required to close that gap. Under an author-produced synthetic workload on a local Anvil simulator, the median commi
    
[^79]: 从表面修饰到结构重构：一项关于指标引导下智能体重构的配对实验

    From Cosmetic to Structural: A Paired Experiment on Metric-Guided Agentic Refactoring

    [https://arxiv.org/abs/2610.33343](https://arxiv.org/abs/2610.33343)

    显式静态分析反馈能够引导编码智能体从表面修饰性编辑转向结构性重构，将有效重构率提升21至26个百分点，且其收益取决于初始代码质量，各语言的盈亏平衡点位于CH 4.9至7.1之间。

    

    编码智能体的角色正在从编写新代码扩展到作为AI队友维护现有系统。然而，近期研究表明，智能体的重构倾向于表面化、一致性导向的编辑，而非解决技术债（TD）所需的结构性转换。我们研究了显式的静态分析反馈能否弥合这一差距。通过CodeScene MCP服务器，我们让开源智能体gptme能够访问代码坏味报告和代码健康度（CH）评分（1-10分，分数越高越好），并在一项涵盖24,620个C++、Java和Python单文件竞赛编程解决方案的配对实验中，比较了有引导与无引导的重构效果。我们发现，引导使智能体从执行表面修饰性编辑转向结构性操作，将有效重构率提高了21至26个百分点。其收益取决于初始代码质量，各语言存在不同的盈亏平衡点，位于CH 4.9至7.1之间。

    arXiv:2610.33343v1 Announce Type: new  Abstract: The role of coding agents is expanding beyond writing new code toward maintaining existing systems as AI teammates. Yet recent studies show that agent refactorings favor superficial, consistency-oriented edits over structural transformations needed to resolve Technical Debt (TD). We investigate if explicit static-analysis feedback closes this gap. Through the CodeScene MCP server, we give an open-source agent (gptme) access to code smell reports and CodeHealth (CH) scores (1--10, higher is better), and compare guided against unguided refactorings in a paired experiment covering $24{,}620$ single-file competitive programming solutions in C++, Java, and Python. We find that guidance shifts the agent from performing cosmetic edits toward structural operations, raising the rate of effective refactorings by $21$--$26$ percentage points. The benefit depends on initial code quality, with language-dependent break-even points at CH $4.9$--$7.1$, 
    
[^80]: 智能体系统的责任边界：概率智能体编程的比较评估

    Responsibility Boundaries for Agentic Systems: A Comparative Evaluation of Probabilistic Agentic Programming

    [https://arxiv.org/abs/2610.33312](https://arxiv.org/abs/2610.33312)

    该论文提出概率智能体编程（PAP），将责任边界作为显式编程抽象，通过中介内核、条件安全性归纳证明和反向模拟，为语言模型智能体的授权、审批与证据追踪提供形式化保障。

    

    语言模型智能体结合了推理、工具、记忆、委托和外部效应，而权限往往分散在提示词、宿主代码、配置和日志之中。概率智能体编程将责任边界作为显式编程抽象，连接授权、权限、效应与证据。我们从七个维度将PAP与编程范式、面向智能体和BDI系统、工作流与策略语言以及LLM工具使用框架进行了比较。中介内核将工具执行前的检查、基于操作的审批、资源限制和证据形式化。归纳证明在守卫正确且中介完备的假设下，为任意模型响应建立了条件安全性。反向模拟在定义的顺序Touba子集中将图执行与源执行相匹配。人工检查点负责释放程序延续，基于操作的审批则保持为独立内核。

    arXiv:2610.33312v1 Announce Type: cross  Abstract: Language-model agents combine reasoning, tools, memory, delegation, and external effects. Authority is often distributed across prompts, host code, configuration, and logs. Probabilistic Agentic Programming (PAP) makes responsibility boundaries explicit programming abstractions connecting mandate, authority, effects, and evidence. We compare PAP with programming paradigms, agent-oriented and BDI systems, workflow and policy languages, and LLM tool-use frameworks along seven dimensions. A mediation kernel formalizes checks before tool execution, operation-bound approval, resource limits, and evidence. Inductive proofs establish conditional safety for arbitrary model responses, assuming correct guards and complete mediation. A backward simulation matches graph executions to source executions in a defined sequential Touba subset. Its human checkpoints release program continuations; operation-bound approval remains a separate kernel contra
    
[^81]: 在受监管金融领域落地治理型智能体工作流：基于 Zafin AIOS 的“零号客户”案例研究

    Operationalizing Governed Agentic Workflows in Regulated Finance: A Customer Zero Case Study using Zafin AIOS

    [https://arxiv.org/abs/2610.33253](https://arxiv.org/abs/2610.33253)

    本文提出Zafin AIOS这一端到端编排与控制平台，并通过“零号客户”内部生产案例研究，验证了治理型智能体工作流在受监管金融行业中从试点走向生产落地的可行性与有效性。

    

    金融和医疗等受监管行业面临着关键的治理挑战，这些挑战阻碍了AI辅助软件开发的采用，也阻碍了从实验性试点向生产环境工作的过渡。本文介绍了Zafin AIOS，一个专为在金融等受监管行业中部署AI而设计的端到端编排平台和控制平面。我们提供了架构原则，以及AI辅助工作负载与人工主导工作负载之间的实证比较。为了对该框架进行实证验证，我们对Zafin的内部企业工作流程进行了案例研究分析，我们将其称为“零号客户”。本文为受监管行业中的AI辅助软件工程贡献了一个生产案例研究。我们在两个不同的生产代码库上评估了使用Zafin AIOS的结果，一个用于定价执行平台，另一个用于客户交易谈判平台。我们的贡献是一个……

    arXiv:2610.33253v1 Announce Type: cross  Abstract: Regulated industries such as finance and healthcare face critical governance challenges that hamper the adoption of AI-assisted software development and hinder the transition from experimental pilots to production work. This paper presents Zafin AIOS, an end-to-end orchestration platform and control plane designed to deploy AI within regulated industries such as finance. We provide the architectural principles and an empirical comparison between AI-assisted and human-led workloads. To validate this framework empirically, we conduct a case study analysis of Zafin's internal enterprise workflows, which we refer to as Customer Zero. This paper contributes a production case study of AI-assisted software engineering in regulated industries. We evaluate the results of using Zafin AIOS with two different production codebases, one for the pricing execution platform and the other for the client deal negotiation platform. Our contribution is a c
    
[^82]: 位置、角色还是标签？LLM智能体中指令冲突的受控研究

    Position, Role or Label? A Controlled Study of Instruction Conflicts in LLM Agents

    [https://arxiv.org/abs/2610.33201](https://arxiv.org/abs/2610.33201)

    该论文构建了包含1,080个受控场景的WhoWins基准，通过对14个开放权重LLM智能体约189万条回复的分析，揭示了指令冲突中独立于角色和标签的强位置效应——首条指令在消息内和跨消息场景中均显著占优。

    

    LLM智能体会从操作者、仓库配置、用户、工具结果和保存的记忆中接收指令，这些指令经常相互冲突。指令层级基准测试通常询问排名较高的来源是否获胜，但一个来源同时决定了位置、角色和标签。首先，我们构建了WhoWins，这是一个包含五类任务共1,080个由代码自动评分的场景的基准，每个场景最多包含五条互斥指令，并设计了七个仅改变指令传递方式的受控实验。其次，我们研究了14个开放权重模型（参数量从0.6B到122B，分别关闭和开启思考模式），共收集1,887,356条评分回复，并在保留场景上验证了主要发现。在同一条消息内，两条冲突行中的第一行以72.42%至94.66%的比率获胜；跨消息时，第一行仍保持22.95%至62.65%的胜率。在系统（或开发者）消息中首先发送的策略，在所有Qwen模型中与后续指令的冲突大多落败，但在gpt-oss执行操作时大多获胜。在工具和文件写入任务中，其获胜频率低于……

    arXiv:2610.33201v1 Announce Type: cross  Abstract: LLM agents take instructions from the operator, repository configuration, user, tool results and saved memory, which often conflict. Instruction-hierarchy benchmarks ask whether the higher-ranked source wins, but a source sets position, role and label at once. First, we build WhoWins, 1,080 code-scored scenarios in five task types, each with up to five mutually exclusive instructions, and seven controlled experiments that vary only how instructions are delivered. Second, we study 14 open-weight models (0.6B to 122B, thinking off and on) over 1,887,356 scored replies, checking main findings on held-out scenarios. Inside one message the first of two conflicting lines wins 72.42 to 94.66%, and across messages it keeps 22.95 to 62.65%. A policy sent first in the system (or developer) message loses most conflicts with later lines in every Qwen model but wins most on gpt-oss when it acts. In tool and file-write tasks it wins less often than 
    
[^83]: 跨平台应用迁移至鸿蒙系统：一种基于产物推导验证的智能体流水线

    Cross-Platform Application Migration to HarmonyOS: An Agentic Pipeline with Artifact-Derived Verification

    [https://arxiv.org/abs/2610.33197](https://arxiv.org/abs/2610.33197)

    该论文提出了HMOS Migrator，一个智能体流水线，能自动将Android/iOS应用源码转换为ArkTS项目，并通过在真实设备上重放录制交互，以及仅基于源应用行为和流水线产物的七个加权指标进行无参考实现的质量评估，从而验证迁移后的功能正确性。

    

    将为Android或iOS等不同平台编写的应用程序迁移到HarmonyOS NEXT需要使用ArkTS进行完全重写。尽管大语言模型能够自动将源代码结构翻译为ArkTS，但仅实现语法编译是不够的，功能保障是HarmonyOS NEXT迁移中一个关键的开放性挑战。我们提出了HMOS Migrator，这是一个智能体流水线，它读取应用程序的源代码，生成ArkTS项目，并通过在真实设备上重放录制的交互来验证结果。我们将评分问题建模为无参考实现的质量估计；我们仅使用源应用程序录制的运行行为和流水线产生的产物来评估迁移质量。我们定义了七个加权指标，每个指标的构建方式均确保减少测量项不会人为提高分数。在六个开源应用程序上进行的十二次运行中，该流水线的结构复现度较高（c

    arXiv:2610.33197v1 Announce Type: cross  Abstract: The migration of applications written for a different platforms like Android or iOS to HarmonyOS NEXT requires a complete rewrite in ArkTS. Although LLMs automate the translation of source constructs into ArkTS, syntactic compilation is not enough and functional assurance is a critical open challenge in HarmonyOS NEXT migration. We present HMOS~Migrator, an agentic pipeline that reads an application's source code, generates an ArkTS project, and validates the result by replaying recorded interactions on a real device. We cast the scoring problem as quality estimation without a reference implementation; we score a migration using only the source application's recorded behaviour and the artifacts produced by the pipeline. We define seven weighted metrics, each constructed so that reduced measurement cannot artificially raise the score. Over twelve runs across six open-source applications, the pipeline's structural reproduction is high (c
    
[^84]: WebUniverse：为智能体化Web开发扩展经过验证的训练数据

    WebUniverse: Scaling Verified Training Data for Agentic Web Development

    [https://arxiv.org/abs/2610.33149](https://arxiv.org/abs/2610.33149)

    WebUniverse从开源仓库和真实网页构建了包含15K个可验证任务环境和开发轨迹的数据集，用于训练前端Web开发智能体，显著提升了模型在Vision2Web等基准上的表现。

    

    训练前端Web开发智能体需要代码编辑、执行、浏览器检查与修改等环节的演示数据，然而公开的轨迹数据和可复用的任务环境仍然稀缺。我们提出了WebUniverse，这是一个从开源代码仓库和真实网页出发，构建可验证任务及经过验证的开发轨迹的流程。这些任务涵盖了独立和迭代设置下基于文本的应用程序开发，以及跨页面和视口的基于视觉的网页复现。我们为这些任务收集了智能体轨迹，并针对任务特定的功能或视觉要求对生成的应用程序进行验证。我们发布了包含15K个可执行任务环境的WebUniverse-Env，以及包含对应的15K条开发轨迹（附带验证记录和评分）的WebUniverse-Mix。在WebUniverse-Mix上微调Qwen3.5-35B-A3B模型，将Vision2Web平均分数从51.7提升至68.3。

    arXiv:2610.33149v1 Announce Type: new  Abstract: Training agents for front-end web development requires demonstrations of code editing, execution, browser inspection and revision, yet public trajectories and reusable task environments remain limited. We introduce WebUniverse, a pipeline that constructs verifiable tasks with verified development trajectories from open-source repositories and real web pages. The tasks cover text-grounded application development in standalone and iterative settings and vision-grounded reproduction across pages and viewports. We collect agent trajectories for these tasks and verify the resulting applications against task-specific functional or visual requirements. We release WebUniverse-Env, containing 15K executable task environments, and WebUniverse-Mix, containing the corresponding 15K development trajectories with verification records and scores. Fine-tuning Qwen3.5-35B-A3B on WebUniverse-Mix raises the Vision2Web average score from 51.7 to 68.3, WebGe
    
[^85]: 投机分叉：将闲置等待转化为代码仓库环境搭建中的并行探索

    Speculative Forking: Turning Idle Wait into Parallel Exploration for Repository Environment Setup

    [https://arxiv.org/abs/2610.33129](https://arxiv.org/abs/2610.33129)

    SpecBuild通过“投机分叉”机制将环境命令的漫长等待时间用于并行探索多条安装分支，返回首个成功配置的环境并丢弃其余分支，从而显著提升基于LLM的代码仓库环境搭建的速度与成功率。

    

    在开发人员能够使用一个软件项目之前，必须先让该项目运行起来。这通常需要安装正确的语言运行时、系统软件包、构建工具和测试命令。近期基于大语言模型（LLM）的智能体通过检查代码仓库、编写安装脚本、运行脚本并修复产生的错误，实现了环境搭建的自动化。然而，这些智能体仍然遵循串行的试错循环：它们先尝试一条安装路径，等待缓慢的命令执行完毕，观察失败结果，然后才决定下一步尝试什么。由于环境相关的命令往往需要很长时间运行，这种串行工作流使得环境搭建速度缓慢，且每次只能探索一条路径。我们提出了SpecBuild，一个通过投机分叉机制实现更快、更高成功率的基于LLM的环境搭建系统。SpecBuild并行探索多条候选环境搭建分支，返回第一个成功配置完成的环境，并丢弃其余分支。它将这一设计应用于……

    arXiv:2610.33129v1 Announce Type: cross  Abstract: Before developers can use a software project, they must first get it to run. This often requires installing the right language runtime, system packages, build tools, and test commands. Recent LLM-based agents automate environment setup by inspecting a repository, writing setup scripts, running them, and repairing the resulting errors. However, these agents still follow a sequential trial-and-error loop: they try one setup route, wait for slow commands to finish, observe the failure, and only then decide what to try next. Because environment commands often take long to run, this sequential workflow makes setup slow and limits exploration to one route at a time.   We present SpecBuild, an LLM-based environment setup system designed to be both faster and more successful through speculative forking. SpecBuild explores candidate setup branches in parallel, returns the first committed environment, and discards the rest. It applies this desig
    
[^86]: 驾驭LLM智能体中的认知对等问题：一个面向记忆与程序性技能间跨源冲突解决的基准

    Navigating Epistemic Parity in LLM Agents: A Benchmark for Cross-Source Conflict Resolution Between Memory and Procedural Skills

    [https://arxiv.org/abs/2610.33128](https://arxiv.org/abs/2610.33128)

    该论文提出了一个衡量LLM智能体如何裁决记忆与程序性技能之间跨源指令冲突的基准，包含记忆-技能矛盾的四分类法以及52个由确定性检查器评分的测试场景。

    

    生产环境中的LLM智能体日益同时运行情景记忆流水线和程序性技能模块（例如SKILL.md文件）。两者被注入到同一个上下文区域中且没有定义的优先级，因此它们可能发出任何指令层级都无法解决的矛盾指令，我们将这种状况称为“认知对等”。我们引入了一个基准，用于衡量LLM智能体如何裁决此类跨源冲突。我们（i）开发了记忆-技能矛盾的四分类法（事实类、格式类、程序类、权限类）；（ii）在三分测试用例结构（记忆注入、技能注入、模糊用户任务）下实例化了52个场景，几乎全部通过对工具调用载荷和输出结构的确定性检查器进行评分；（iii）在三种上下文条件下运行每个场景，以隔离位置偏差、衡量显式优先级指令的效果，并对披露行为进行分类，量化静默裁决行为。

    arXiv:2610.33128v1 Announce Type: new  Abstract: Production LLM agents increasingly run episodic memory pipelines and procedural skill modules (e.g., SKILL.md files) in tandem. Both are injected into the same context region with no defined precedence, so they can issue contradictory instructions that no instruction hierarchy resolves, a condition we term epistemic parity. We introduce a benchmark for measuring how LLM agents adjudicate such cross-source conflicts. We (i) develop a four-way taxonomy of memory-skill contradictions (factual, format, procedural, permission); (ii) instantiate 52 scenarios under a tripartite test-case anatomy (memory injection, skill injection, ambiguous user task), scored almost entirely by deterministic checkers over tool-call payloads and output structure; and (iii) run each scenario under three context conditions to isolate positional bias, measure the efficacy of an explicit precedence directive, and classify disclosure behavior, quantifying silent adju
    
[^87]: ToolContractGate：面向漂移安全的代理工具调用的版本化可信合约

    ToolContractGate: Versioned Trusted Contracts for Drift-Safe Agent Tool Invocation

    [https://arxiv.org/abs/2610.33108](https://arxiv.org/abs/2610.33108)

    提出 ToolContractGate，通过版本化可信合约将代理工具调用请求与已批准的 OpenAPI 源工件确定性绑定，在无误阻断的前提下检测出全部 35 个漂移故障，且平均门控延迟仅 0.18 毫秒。

    

    代理工具审批与输入验证本身并不能将请求绑定到已批准的源工件上。ToolContractGate 为基于 OpenAPI 派生的代理工具提供了一个确定性的运行时合约评估器。每个版本化的 ToolContract 将存储的 OpenAPI 指纹与访问、模式（schema）及行为声明相结合。完整评估器执行十项检查，其中包括缓存的启动时溯源结果；四种执行模式和五种评估配置支持基线、消融以及非阻塞的影子分析。在一个包含 45 个案例的合成基准测试中（涵盖 12 类漂移的 35 个不安全案例和 10 个安全案例），完整流水线检测出全部 35 个故障：29 个 BLOCK（阻断）、3 个 REVIEW（审查）和 3 个 WARN（警告）决策，且对安全案例没有产生误阻断。存档运行记录显示，采用整数毫秒测量时，门控平均延迟为 0.18 毫秒，第 95 百分位数（p95）为 2 毫秒。开源工件包含七个 OpenAPI 规范……

    arXiv:2610.33108v1 Announce Type: new  Abstract: Agent tool approval and input validation do not, by themselves, bind a request to an approved source artifact. ToolContractGate provides a deterministic runtime contract evaluator for OpenAPI-derived agent tools. Each versioned ToolContract combines a stored OpenAPI fingerprint with access, schema, and behavioral declarations. The full evaluator applies ten checks, including a cached boot-time provenance result; four execution modes and five evaluation configurations support baseline, ablation, and non-blocking shadow analysis. On a synthetic benchmark of 45 cases (35 unsafe cases across 12 drift classes and 10 safe cases), the full pipeline detects all 35 faults: 29 BLOCK, 3 REVIEW, and 3 WARN decisions, with no false blocks on the safe cases. The archived run reports a mean gate latency of 0.18 ms and a 95th percentile (p95) of 2 ms using integer-millisecond measurements. The open-source artifact includes seven OpenAPI specifications, 
    
[^88]: 超越正确性：形式化库自然语言文档的归因

    Beyond Correctness: Attribution of Natural-Language Documentation for Formal Libraries

    [https://arxiv.org/abs/2610.33094](https://arxiv.org/abs/2610.33094)

    该论文提出“归因”概念以检验文档陈述是否真正依赖于函数的特定行为，并实现了 DocAuth 文档生成方法，通过过滤掉已证明但不可归因的性质，并为每句发布的文档附加可检查的形式化证据，从而提升形式化库自然语言文档的可信度。

    

    形式化语言在软件工程中被广泛用于规约和验证程序性质。由于其数学符号和逻辑难以阅读，开发者通常依赖自然语言文档来理解形式化库中的函数。然而，一条文档陈述即使在函数行为发生改变后，仍可能被证明为真且依然成立。我们将陈述对函数特定行为的依赖性称为“归因”，并给出了一种检验归因的方法。我们设计并实现了 DocAuth——一种文档生成方法，它利用归因来决定发布哪些内容，并为每个发布的句子附加从中恢复出的形式化性质以及可检查的证据。在 ACL2 定理证明器 Community Books 库的 150 个函数上，三个大语言模型为约三分之一的函数生成了已证明但不可归因的性质。DocAuth 的归因表现与之相当或更优。

    arXiv:2610.33094v1 Announce Type: cross  Abstract: Formal languages are widely used in software engineering to specify and verify program properties. Because their mathematical notation and logic are hard to read, developers often rely on natural-language documentation to understand the functions in formal libraries. Yet a documentation statement can be provably true and still hold after the function's behavior changes. We call a statement's dependence on the function's specific behavior attribution, and we give a method to check it. We design and implement DocAuth, a documentation-generation method that uses attribution to decide what to publish and attaches to each published sentence the formal property recovered from it, together with checkable evidence. On 150 functions from the Community Books library of the ACL2 theorem prover, three large language models produce proved but non-attributable properties for about a third of the functions. DocAuth matches or exceeds the attribution 
    
[^89]: RADAgent：使用私有开源大语言模型实现RADseq分析流程自动化

    RADAgent: Automating RADseq Pipelines with Private Open-Source LLMs

    [https://arxiv.org/abs/2610.33009](https://arxiv.org/abs/2610.33009)

    RADAgent利用私有开源大语言模型自动化RADseq分析流程，将传统上需要专家跨多个阶段协调的异构生物信息学工作流（包括状态跟踪、结果验证和故障处理）实现无人监督执行。

    

    限制性位点相关DNA测序（RADseq）是一类简化表示测序方法。它利用限制性内切酶对基因组子集进行采样，因此能够以低成本同时对多个个体进行测序和基因分型。RADseq广泛应用于群体基因组学、进化生物学、分子标记发现以及SNP芯片面板的开发，但完整的分析仍需要专家在多个工具和处理阶段之间进行协调。传统的RADseq分析需要专家协调异构的生物信息学工具、中间产物、质量控制检查以及跨多个阶段的参数依赖决策。尽管单个步骤可以通过脚本实现，但执行完整的工作流程仍需要人工监督来进行状态跟踪、结果验证和故障处理。为了自动化这种由专家驱动的协调工作，我们提出了RADAgent，它引入了一个模块化的……

    arXiv:2610.33009v1 Announce Type: cross  Abstract: Restriction site-associated DNA sequencing (RADseq) is a family of reduced-representation sequencing methods. It uses restriction enzymes to sample subsets of the genome, therefore allowing multiple individuals to be sequenced and genotyped together at low cost. RADseq is widely used for population genomics, evolutionary biology, marker discovery, and the development of SNP panels, but complete analysis still requires expert coordination across multiple tools and processing stages. Traditional RADseq analysis requires expert coordination of heterogeneous bioinformatics tools, intermediate artifacts, quality-control checks, and parameter-dependent decisions across multiple stages. Although individual steps can be scripted, executing the complete workflow still requires human oversight for state tracking, result validation, and failure handling. To automate this expert-driven coordination, we present RADAgent, which introduces a modular 
    
[^90]: 开源项目对编码智能体说了什么：AGENTS.md 与 CLAUDE.md 文件普查

    What Open-Source Projects Tell Their Coding Agents: A Census of AGENTS.md and CLAUDE.md

    [https://arxiv.org/abs/2610.31715](https://arxiv.org/abs/2610.31715)

    这项研究首次对十三种语言的活跃 GitHub 仓库中的 AGENTS.md 和 CLAUDE.md 指令文件进行大规模普查，系统揭示了开源项目向 AI 编码智能体传达的指令构成（项目知识与代码、构建、工作流规则的混合），以及各项目在 AI 署名披露等问题上的立场分歧。

    

    编码智能体可以通过仓库中的指令文件获得引导。我们通过对十三种语言的热门活跃 GitHub 仓库进行普查来考察这一新兴实践，将每个仓库根目录下的 AGENTS.md 和 CLAUDE.md 视为一个指令集。智能体指令出现在一定比例的仓库以及星标最高的仓库之中；当两个文件同时存在时，CLAUDE.md 通常指向 AGENTS.md。我们采用模型辅助标注与人工审核相结合的方式分析这些指令语句。标注结果显示，指令内容是项目知识与关于代码、构建和工作流规则的实际混合体。协作规则会指明具体要求，从测试计划到提交格式，但相当一部分指令并未说明理由。各项目在 AI 署名问题上立场不一：提交信息规则禁止 AI 署名的情况多于要求署名，而拉取请求规则则更倾向于要求披露 AI 的参与。三分之一的指令集带有与生成器相关的签名、工具插入的内容块、共享文本或其他生成痕迹（摘要在此处截断）。

    arXiv:2610.31715v1 Announce Type: cross  Abstract: Coding agents can be guided with instruction files in a repository. We examine this emerging practice through a census of popular, active GitHub repositories in thirteen languages, treating each repository's root AGENTS.md and CLAUDE.md as one instruction set. Agent instructions appear in of repositories and of the most-starred; where both files exist, CLAUDE.md usually points to AGENTS.md. We analyse statements using model-assisted annotation and human review. The annotations reveal a practical mix of project knowledge and rules about code, building and workflow. Collaboration rules name concrete requirements, from test plans to commit formats, yet of directives give no reason. Projects disagree on AI attribution: commit-message rules more often prohibit it than require it, while pull-request rules more often require disclosure. A third of instruction sets have a generator-associated signature, a tool-inserted block, shared text or an
    
[^91]: 分离字段评分：当智能体汇集工具使用经验时保护未被使用过的工具

    Separate Field Scoring Protects Unused Tools When Agents Pool Tool-Use Experience

    [https://arxiv.org/abs/2610.30867](https://arxiv.org/abs/2610.30867)

    提出SYNAPSE方法，将工具使用记录作为独立字段进行评分，在智能体共享工具使用经验提升已使用工具检索效果的同时，有效避免了从未被使用过的工具检索性能的下降。

    

    从共享工具目录中调用工具的智能体可以汇集它们学到的关于哪种工具适合哪种请求的经验。我们测量了汇集的使用示例对工具检索的影响，并分别针对从未被使用过的预留工具和所有其他工具进行报告。将使用示例作为已使用工具的额外条目加入索引，会提高这些工具的Hit@5（即正确工具是否出现在前五名），但在三个工具目录中的两个以及我们测试的两种嵌入器上，会大幅降低预留工具的Hit@5。如果给已使用工具添加相同数量、等长但填充其他工具文本的条目，预留工具的Hit@5会下降相近的幅度，却几乎不带来任何增益。我们将这种把使用记录作为独立字段进行评分的方法称为SYNAPSE。在StableToolBench上，预留的正确工具进入前五名的频率高于为使用记录保留槽位的路由器方法（0.785对0.683），并且这一在预留工具上的优势在所有三个工具目录和两种嵌入器上均保持成立。

    arXiv:2610.30867v1 Announce Type: new  Abstract: Agents that call tools from a shared catalog can pool what they learn about which tool serves which request. We measure what pooled usage examples do to tool retrieval, reporting separately the held-out tools, for which no agent has usage, and all other tools. Adding usage examples to the index as extra entries for used tools raises Hit@5 (whether a gold tool is in the top five) for those tools and lowers it for held-out tools, by a large margin on two of three catalogs and with both embedders we tested. Giving used tools the same number of equally long entries filled with other tools' text lowers held-out Hit@5 about as much, but gives little or none of the gain. We call our method of scoring usage as its own field SYNAPSE. On StableToolBench, the held-out gold tool appears in its top five more often than with a router that reserves slots for usage (0.785 against 0.683), and this held-out lead holds on all three catalogs with both embed
    
[^92]: 贷款处理员与贷款审核员能识别AI编辑的报表吗？一项实证研究

    Can Loan Processors and Underwriters Spot AI-Edited Statements? An Empirical Study

    [https://arxiv.org/abs/2610.30758](https://arxiv.org/abs/2610.30758)

    该研究通过实证实验发现，经验丰富的贷款审核员和处理员在仅凭文件的首次审阅中，仅有约10.6%的判断准确指出了AI伪造财务文件中的编辑痕迹（且全部依赖标识符不一致），表明仅靠人工文件审查很难发现AI编辑的伪造贷款文件。

    

    AI辅助编辑可以在保留原发行方熟悉版式的同时更改申请人的身份，从而生成贷款文件。我们研究了经验丰富的贷款审核员和贷款处理员在首次审阅（仅基于文件本身）时能否识别此类伪造文件。通过Upwork招募的三位从业者，每人收到一份申请人材料包，这些材料包来自134份伪造的PDF文件，而这些PDF文件是在完全知情同意的前提下，基于47份真实财务文件生成的。编辑保留了金额、日期和余额；在分发之前，处理流程中引入的标识符不一致被有意保留。审阅者未被告知文件已被篡改，共返回了132份文件级判断。将他们的解释与原始文件核对后，共发现14处有编辑依据的疑虑（10.6%）：12个“可疑”判断和2个“信息不足”判断。所有有依据的疑虑均涉及文件内部或跨文件的标识符不一致，而没有报告的视觉…（摘要在此处被截断）

    arXiv:2610.30758v1 Announce Type: cross  Abstract: AI-assisted editing can produce loan documents that preserve familiar issuer layouts while changing the applicant's identity. We investigated whether experienced underwriters and loan processors identify such forgeries in first-pass, document-only review. Three practitioners recruited through Upwork each received one applicant package drawn from 134 forged PDFs derived from 47 genuine financial documents supplied with full consent. The edits preserved amounts, dates and balances; identifier inconsistencies introduced by the pipeline were knowingly retained before distribution. Reviewers were not told that files had been altered and returned 132 file-level judgements. Checking their explanations against the originals yielded 14 edit-supported concerns (10.6%): 12 Suspicious judgements and two Insufficient information judgements. All supported concerns involved identifier disagreements within or across documents, and no reported visual c
    
[^93]: SpecAgent：通过智能体式形式化程序规约合成赋能程序验证

    SpecAgent: Empowering Program Verification with Agentic Synthesis of Formal Program Specifications

    [https://arxiv.org/abs/2610.05132](https://arxiv.org/abs/2610.05132)

    SpecAgent 提出一个智能体式框架，结合依赖感知规划与检索增强生成等组件，为真实世界 C 程序自动合成高质量 ACSL 形式化规约，克服了传统前向式流程和局部修复无法解决的级联验证失败问题。

    

    形式化规约对于演绎式程序验证至关重要，它为复杂软件的组合式验证提供了语义抽象。然而，手动构建规约非常耗时费力，这促使人们寻求自动化规约合成方法。尽管大语言模型（LLM）近来取得了进展，但现有方法通常依赖仅前向的工作流程和局部化修复，这限制了它们在具有复杂函数间依赖关系和循环的真实世界程序上的有效性。验证失败可能源于先前生成的规约，从而导致局部细化无法解决的级联失败。为应对这些挑战，我们提出了 SpecAgent，一个用于为真实世界 C 程序合成高质量 ACSL 规约的智能体框架。SpecAgent 集成了四个组件：用于识别规约目标的依赖感知规划、用于提供相关规约信息的检索增强生成（RAG）……

    arXiv:2610.05132v2 Announce Type: replace  Abstract: Formal specifications are essential for deductive program verification, providing semantic abstractions for compositional verification of complex software. However, manually constructing specifications is labor-intensive, motivating automated synthesis. Despite recent advances in large language models (LLMs), existing approaches often rely on forward-only workflows and localized repair, limiting their effectiveness on real-world programs with complex dependencies among functions and loops. Verification failures may stem from previously generated specifications, causing cascading failures that local refinement cannot resolve. To address these challenges, we present SpecAgent, an agentic framework for synthesizing high-quality ACSL specifications for real-world C programs. SpecAgent integrates four components: dependency-aware planning to identify specification targets, retrieval-augmented generation (RAG) to provide relevant specifica
    
[^94]: A2ABreak：A2A协议的系统性安全分析

    A2ABreak: Systematic Security Analysis of the A2A Protocol

    [https://arxiv.org/abs/2609.10871](https://arxiv.org/abs/2609.10871)

    本文提出A2ABreak框架，首次对A2A协议进行系统性安全分析，通过LLM辅助从自然语言规范中提取经验证的有限状态机模型（37个状态、76个转换），并利用对抗性验证系统性发现协议层面的安全漏洞。

    

    由Linux基金会管理的Agent2Agent（A2A）协议是一项开放标准，使自主AI智能体能够跨越组织边界相互发现、进行身份认证并委派任务。A2A旨在与用于工具集成的模型上下文协议（MCP）互补，正迅速成为多智能体生态系统的水平通信层。然而，该协议的安全性尚未得到系统性分析。本文提出了A2ABreak，这是对A2A协议首次严谨的系统性安全分析。我们引入了一种新颖的框架，利用大语言模型（LLM）辅助从自然语言规范中直接提取经验证的有限状态机，从929条形式化语句中构建了一个包含37个状态和76个转换的统一模型，随后对该模型进行系统性推理，在完全符合规范的条件下通过对抗性验证来发现协议层面的漏洞。

    arXiv:2609.10871v1 Announce Type: cross  Abstract: The Agent2Agent (A2A) protocol, now governed by the Linux Foundation, is an open standard that enables autonomous AI agents to discover, authenticate with, and delegate tasks to one another across organizational boundaries. Designed to complement the Model Context Protocol (MCP) for tool integration, A2A is rapidly emerging as the horizontal communication layer of the multi-agent ecosystem. Yet the protocol's security has received no systematic analysis.   This paper presents A2ABreak, the first rigorous systematic security analysis of the A2A protocol. We introduce a novel framework that utilizes an LLM-assisted extraction of a verified finite-state machine directly from the natural-language specification, producing a unified model of 37 states and 76 transitions from 929 formalized statements, and then systematically reasons over this model to discover protocol-level vulnerabilities through adversarial verification, under a full-comp
    
[^95]: 出厂即坏：公共模型注册库中静默失效的大语言模型制品及其检测方法

    Broken on Arrival: Silently Defective LLM Artifacts in Public Model Registries and How to Catch Them

    [https://arxiv.org/abs/2609.05881](https://arxiv.org/abs/2609.05881)

    首次系统性冒烟测试了公共模型注册库中的327个量化GGUF模型制品，发现官方Ollama库中存在五个静默失效的量化模型转换制品，并提出了一套可有效捕获此类缺陷的检测流程。

    

    开发者越来越多地通过从公共模型注册库拉取量化GGUF制品来在本地运行大语言模型，然而整个分发流程中没有任何环节在用户获取这些转换产物之前对其进行功能性测试。我们共执行了327个具备代码能力的量化模型制品：其中305个来自官方Ollama库，涵盖15个模型系列在8GB及以下所有符合条件的量化级别；另外22个来自HuggingFace上下载量最高的社区仓库。每个制品运行一个包含15项任务的冒烟测试套件，该套件经过校准，健康的制品能够通过，而已知损坏的制品则会失败；疑似有问题的制品随后将接受完整的164项任务评估、第二种推理后端的测试、由独立分发商对同一模型和量化级别进行转换作为仲裁，对于社区文件，还会在制品自身的模板下重新测试。官方库中存在五个静默失效的制品，包括一批四个Qwen2.5-Coder-3B转换版本和一个phi3.5-mini转换版本……

    arXiv:2609.05881v1 Announce Type: cross  Abstract: Developers increasingly run large language models locally by pulling quantized GGUF artifacts from public registries, yet nothing in the distribution pipeline functionally tests these conversions before they reach users. We executed 327 quantized code-capable model artifacts: 305 from the official Ollama library, spanning 15 model lines at every eligible quantization level at or under 8 GB, and 22 from the most-downloaded community repositories on HuggingFace. Each ran a 15-task smoke suite calibrated so that healthy artifacts pass while a known-broken one fails; suspects then faced full 164-task evaluation, a second inference backend, an independent distributor's conversion of the same model and quantization as referee, and, for community files, re-testing under the artifact's own template. The official library carries five silently defective artifacts, a batch of four Qwen2.5-Coder-3B conversions and one phi3.5-mini conversion, that 
    
[^96]: AgentLogs：一个揭开GitHub云端智能体黑盒的数据集

    AgentLogs: A Dataset for Opening the Black Box of GitHub's Cloud Agent

    [https://arxiv.org/abs/2608.29204](https://arxiv.org/abs/2608.29204)

    该论文发布了AgentLogs——首个大规模GitHub智能体活动数据集，包含30余万个任务、54万余个会话及6400多万条逐步日志，首次打开了GitHub Copilot云智能体完成软件工程任务全过程的“黑盒”。

    

    基于生成式AI的软件工程智能体正日益成为真实世界软件项目的常规贡献者。在GitHub上，开发者可以将任务分配给Copilot云智能体，该智能体能够自主探索代码仓库、编辑代码、运行命令、创建或审查拉取请求，并生成记录每一步操作的详细日志。尽管现有数据集捕捉了智能体贡献的结果（例如智能体编写的拉取请求），但智能体产生这些贡献的过程在很大程度上仍未被探索。为填补这一空白，我们推出了AgentLogs，一个大规模的GitHub智能体活动数据集。AgentLogs包含307,416个智能体任务和549,239个智能体会话，覆盖我们扫描的1,812,362个热门公开仓库中的35,810个，并附带64,255,174条会话日志条目，逐条记录每次智能体运行的完整过程，包括提示词、中间推理、工具调用（例如文件编辑、git操作和GitHub API调用等）。

    arXiv:2608.29204v1 Announce Type: cross  Abstract: Generative AI-based software engineering agents are becoming routine contributors to real-world software projects. On GitHub, developers can assign tasks to the Copilot cloud agent, which autonomously explores the repository, edits code, runs commands, and opens or reviews pull requests, producing a detailed log of every step along the way. While existing datasets capture outcomes of agent contributions, such as agent-authored pull requests, the process by which agents produce these contributions remains largely unexplored. To address this gap, we introduce AgentLogs, a large-scale dataset of agent activity on GitHub. AgentLogs comprises 307,416 agent tasks and 549,239 agent sessions in 35,810 of the 1,812,362 popular public repositories that we scanned, together with 64,255,174 session log entries that record each agent run step by step, including prompts, intermediate reasoning, tool calls (e.g., file edits, git operations, and GitHu
    
[^97]: SemaPLC：一种面向PLC代码生成的、基于项目且验证门控的代理框架

    SemaPLC: A Project-Grounded, Verification-Gated Agent Harness for PLC Code Generation

    [https://arxiv.org/abs/2608.18565](https://arxiv.org/abs/2608.18565)

    SemaPLC通过严格的外部验证门控和项目集成，显著提升了大型语言模型生成的PLC代码在真实工业环境中的可靠性与通过率。

    

    可编程逻辑控制器（PLC）运行工业工厂，而大型语言模型已经能够为其生成独立的程序组织单元（POU）。此类逻辑是否能集成到现有PLC项目中并正确运行，仅在有限测试中得到了验证。我们提出了SemaPLC，这是一种基于项目且验证门控的代理框架，由常规工具组装而成，但受严格完成规则约束。SemaPLC并非在模型判断自身输出足够时停止，而是仅在记录的外部检查确认后才宣布任务完成。这些检查覆盖规范、编译以及实时运行时的行为。在117个匹配现有基准的独立POU任务中，它在所有七个模型上达到了最高的严格验证通过率（平均72.6%）。在65个任务的项目上下文轨道中，其生成的逻辑必须在真实项目内编译和运行，它实现了...

    arXiv:2608.18565v1 Announce Type: new  Abstract: Programmable logic controllers (PLCs) run industrial plants, and large language models can already generate independent program organization units (POUs) for them. Whether such logic integrates into an existing PLC project and then runs correctly has been checked only in limited tests. We present \textsc{SemaPLC}, a project-grounded and verification-gated agent harness assembled from conventional tools but governed by a strict completion rule. Rather than stopping when the model judges its own output adequate, \textsc{SemaPLC} declares a task complete only when logged external checks confirm it. Those checks cover the specification, the compilation, and the behavior on a live runtime. On 117 independent-POU tasks matching existing benchmarks, it attains the highest strict verified pass rate on all seven models (72.6\% mean). On a project-context track of 65 tasks whose generated logic must compile and run inside a real project, it attain
    
[^98]: EntailLLM：基于逻辑编程利用领域知识验证智能体生成的漏洞发现路径

    EntailLLM: Verifying Agent-Generated Vulnerability Discovery Paths with Domain Knowledge via Logic Programming

    [https://arxiv.org/abs/2608.01763](https://arxiv.org/abs/2608.01763)

    EntailLLM框架让专门的LLM智能体提出二进制漏洞的函数调用图遍历路径，并通过标注逻辑下的蕴含推导将其与独立的领域知识图谱和逻辑程序进行验证，从而提升安全关键场景中漏洞分析的可靠性。

    

    大型语言模型智能体正被越来越多地用于分析潜在的软件漏洞，但其输出可能会悄然违反领域知识，从而限制了其在医疗器械等安全关键场景中的可靠性。以往的工作要么将该输出视为需要评分的预测，要么将其约束在单一知识图谱内的游走路径上；两者都无法检查针对二进制文件的推理是否被一个独立的领域知识体系所蕴含。我们提出了EntailLLM，这是一个包含专门LLM智能体和验证机制的框架。该智能体针对某一漏洞提出函数调用图的遍历路径，该路径通过标注逻辑下的蕴含推导程序，与一个独立的知识图谱和逻辑程序进行核对验证。我们通过检查六个专门智能体在三个CWE类别上的表现来评估EntailLLM，并使用了函数数量从405到12,696不等的八个二进制文件。这些智能体提升了……（摘要在此处截断）

    arXiv:2608.01763v2 Announce Type: replace-cross  Abstract: Large language model agents are increasingly used to analyze potential software vulnerabilities, but their outputs can silently violate domain knowledge, limiting their reliability in safety-critical settings such as medical devices. Prior work either treats that output as a prediction to be scored or constrains it to walks within a single knowledge graph; neither checks whether reasoning over a binary is entailed by an independent body of domain knowledge. We present EntailLLM, a framework that includes a specialized LLM agent and a validation mechanism. The agent proposes a traversal path of the function call graph for a vulnerability, which is checked against a separate knowledge graph and a logic program by an entailment procedure under annotated logic. We evaluate EntailLLM by examining six specialized agents across three CWE classes, and eight binaries varying in size from 405 to 12,696 functions. The agents improve poole
    
[^99]: Change2Task：从仓库变更到可执行的编码智能体任务与环境

    Change2Task: From Repository Changes to Executable Coding Agent Tasks and Environments

    [https://arxiv.org/abs/2607.28591](https://arxiv.org/abs/2607.28591)

    Change2Task 提出了一个基于仓库历史的系统，能够将已合并的拉取请求自动转换为健康现代代码版本上经过验证的可执行编码智能体任务与环境，为编码智能体的训练、基准测试和持续评估提供可扩展的数据供给。

    

    扩展编码智能体需要持续供给可执行的数据，用于训练、基准测试和持续评估。每个任务必须将真实的软件状态与任务规范、开发工具和可靠的验证相结合。为了扩大这一供给，我们提出了 Change2Task，这是一个基于仓库历史的系统，它将已合并的拉取请求（PR）转换为同一仓库健康现代版本上经过验证的任务。它将历史证据与演进的代码进行对齐，通过补丁反转、代码映射或智能体重建三种方式重建任务状态，并验证从健康基础状态到任务状态再到恢复状态的完整生命周期。通过从维护良好的环境中推导出多个基于开发者证据的任务，Change2Task 为编码智能体的训练和评估提供了可执行数据，同时减少了重复的环境搭建、存储和任务构建成本。我们评估了五个编码智能体……（原文截断）

    arXiv:2607.28591v2 Announce Type: replace-cross  Abstract: Scaling coding agents requires a continuing supply of executable data for training, benchmarking, and continuous evaluation. Each task must couple a realistic software state with a specification, development tools, and reliable verification. To expand this supply, we present Change2Task, a system grounded in repository history that converts merged pull requests (PRs) into verified tasks on healthy modern revisions of the same repository. It aligns historical evidence with evolved code, reconstructs task states through Patch Reversal, Code Mapping, or Agent Reconstruction, and validates the lifecycle from a healthy base to a task state and a restored state. By deriving multiple tasks grounded in developer evidence from maintained environments, Change2Task provides executable data for coding agent training and evaluation while reducing repeated environment setup, storage, and task construction effort. We evaluate five coding agen
    
[^100]: 掌控方向盘：自动驾驶系统测试现实的多公司研究

    In the Driver's Seat: A Multi-Company Study on the Reality of Autonomous Driving System Testing

    [https://arxiv.org/abs/2607.15820](https://arxiv.org/abs/2607.15820)

    该研究通过对六个国家九家公司专家的访谈，首次系统揭示了工业界自动驾驶系统测试的真实运作方式——一个由规划证据需求、产生与评估证据等反复活动构成的证据构建过程。

    

    自动驾驶系统（ADS）日益被部署于复杂的现实环境中，这对建立其功能性和安全性所需的严格测试提出了强烈需求。然而，自动驾驶系统测试仍然困难重重，因为从业者不仅需要确定测试什么以及如何测试，还必须判断由此产生的证据是否足以支持推进和发布决策。为了解实践中如何应对这些问题，我们对来自六个国家九家公司的自动驾驶系统开发与测试专家进行了半结构化访谈。通过主题分析，我们考察了从业者如何组织自动驾驶系统测试以产生和评估证据，以及哪些因素限制了对这些证据的信心。我们的研究结果表明，工业界的自动驾驶系统测试可以理解为一个证据构建过程，其中涉及三种反复出现的活动：规划所需证据、跨互补渠道产生和评估证据……

    arXiv:2607.15820v2 Announce Type: replace  Abstract: Autonomous driving systems (ADS) are increasingly deployed in complex real-world environments, creating strong demands for rigorous testing to establish their functionality and safety. However, ADS testing remains difficult because practitioners must determine not only what and how to test, but also whether the resulting evidence is sufficient to support progression and release decisions. To understand how this is handled in practice, we conducted semi-structured interviews with experts involved in ADS development and testing across nine companies in six countries. Through thematic analysis, we examined how practitioners organize ADS testing to produce and assess evidence, and what limits confidence in that evidence. Our findings show that industrial ADS testing can be understood as an evidence-building process involving three recurring activities: planning what evidence is needed, producing and evaluating evidence across complementa
    
[^101]: 压缩即认知失败：智能体式LLM工具如何从被终止的进程中编造出“已确认的结果”

    Compaction as Epistemic Failure: How Agentic LLM Tools Fabricate Confirmed Results from Killed Processes

    [https://arxiv.org/abs/2607.13071](https://arxiv.org/abs/2607.13071)

    本文揭示了Claude Code等智能体LLM工具会将超时被杀进程的部分输出错误记录为“已确认结果”并在会话间传播假阳性，其根源在于混淆了“终端观察”与“持久化存储”这两种信息状态。

    

    智能体式LLM编程工具会将漫长的会话历史压缩为压缩摘要，后续会话会将其继承为“真实依据”。本文记录了Claude Code中的一种失败模式：超时被终止的命令（退出码143）产生的部分标准输出被记录进压缩摘要并被视为已确认的结果，从而在跨会话、跨模型版本之间传播假阳性，且未经重新验证。其底层机制是“观察”与“持久化”的混淆——出现在终端中的信息被等同于已写入持久存储的信息。这一发现扩展了先前关于LLM-as-judge评分非确定性研究中所报告的LLM自我评估失败的分析，表明智能体式工具在报告自身操作结果时表现出类似的可靠性缺陷。该失败对任何依赖智能体会话连续性进行数据处理的工作流具有直接影响。

    arXiv:2607.13071v2 Announce Type: replace-cross  Abstract: Agentic LLM coding tools compress long session histories into compaction summaries that subsequent sessions inherit as ground truth. This paper documents a failure mode in Claude Code where partial standard output from timed-out commands (exit code 143) is recorded in compaction summaries as confirmed results, propagating false positives across sessions and model versions without re-verification. The underlying mechanism is a conflation of observation and persistence, where information that appeared in the terminal is treated as equivalent to information written to durable storage. This finding extends the analysis of LLM self-evaluation failures reported in prior work on non-determinism in LLM-as-judge grading by showing that agentic tools exhibit analogous reliability deficits when reporting on their own operational outcomes. The failure has direct implications for any workflow that relies on agentic session continuity for da
    
[^102]: MicroAgent：面向自动微服务分解的上下文增强多智能体框架

    MicroAgent: Context-Augmented Multi-Agent Framework for Automatic Microservice Decomposition

    [https://arxiv.org/abs/2606.29742](https://arxiv.org/abs/2606.29742)

    该论文提出了MicroAgent，一个上下文增强的多智能体框架，通过多智能体协作并融合丰富的上下文信息，自动化地将单体应用分解为高内聚的微服务，从而克服了传统自动化方法语义理解不足以及直接使用大语言模型时忽视上下文与设计原则的局限。

    

    微服务架构（MSA）的采用通过提升可扩展性、敏捷性和可维护性，相比传统单体应用给软件工程带来了革命性变化。随着越来越多的开发者将遗留系统迁移到基于微服务的架构，有效的微服务分解——即将单体应用划分为高内聚的服务——变得至关重要。然而，这一分解任务面临重大挑战：手动方法耗时且劳动密集；现有的自动化方法往往难以从复杂应用中捕获必要的语义洞察；而对大语言模型（LLM）的简单直接应用则容易忽视关键的上下文信息和设计原则，导致结果欠佳。为了应对这些挑战，我们提出了MicroAgent，一个用于微服务分解的上下文增强多智能体框架。我们的框架将分解过程……

    arXiv:2606.29742v2 Announce Type: replace  Abstract: The adoption of Microservice Architecture (MSA) has revolutionized software engineering by enhancing scalability, agility, and maintainability over traditional monolithic applications. As more developers transition their legacy systems to microservice-based architectures, effective microservice decomposition-partitioning monolithic applications into highly cohesive services-becomes vital. However, this decomposition task presents significant challenges. Manual approaches are time-consuming and labor-intensive. Existing automated methods often fail to capture the necessary semantic insights from complex applications, while naive applications of Large Language Models tend to overlook crucial contextual information and design principles, leading to suboptimal results.   To address these challenges, we propose MicroAgent, a Context-Augmented Multi-Agent Framework for Microservice Decomposition. Our framework divides the decomposition pro
    
[^103]: 无实质区别的差异：包注册表中发布信任证据的测量研究

    A Difference Without a Distinction: A Measurement Study of Release Trust Evidence in Package Registries

    [https://arxiv.org/abs/2606.22593](https://arxiv.org/abs/2606.22593)

    对包注册表发布信任证据的大规模测量研究表明，pnpm的新证据策略主要误伤手动向后移植的安全修复（semver 6.3.1造成16个热门失败中的14个），并首次对比了npm、PyPI、Maven Central、crates.io和RubyGems五大注册表的发布证据状况。

    

    被盗的npm令牌或账户可使攻击者发布流行包的恶意版本。此类上传会绕过项目的CI工作流，并可能缺少其来源证明，因此自2025年11月起，一项可选启用的pnpm策略会拒绝任何证据弱于先前版本的发布。维护者也会手动发布向后移植版本；该策略阻断这些发布的频率以及它能标记哪些攻击尚属未知。我们在1,997个热门及近期更新的npm包的最新版本上测试该策略，在重构的凭据被盗事件上重放其策略变体和身份锁定，并分析npm恶意软件公告普查。我们还比较了五个包注册表（npm、PyPI、Maven Central、crates.io和RubyGems）的发布证据。使用pnpm 10.29.3时，1.6%的热门安装和7.6%的近期更新安装被阻断，其中主要由向后移植引起，而一个手动向后移植的安全修复semver 6.3.1造成了16个热门失败中的14个。该策略……

    arXiv:2606.22593v3 Announce Type: replace  Abstract: A stolen npm token or account lets an attacker publish a malicious version of a popular package. Such uploads skip the project's CI workflow and can lack its provenance, so since November 2025 an opt-in pnpm policy refuses any release with weaker evidence than an earlier one. Maintainers also publish backports by hand; how often the policy blocks them and which attacks it flags is unknown. We test the policy on the latest versions of 1,997 popular and recently updated npm packages, replay its variants and identity pins on reconstructed stolen-credential incidents, and analyze an npm malware advisory census. We also compare publication evidence across five package registries: npm, PyPI, Maven Central, crates.io, and RubyGems. With pnpm 10.29.3, 1.6% of popular and 7.6% of recently updated installs are blocked, mostly by backports, and one security fix backported by hand, semver 6.3.1, causes 14 of the 16 popular failures. The policy c
    
[^104]: 《匹配至关重要：面向命令行智能体的公平质量-效率基准测试》

    Matching Matters: A Fair Quality-Efficiency Benchmark for Command-Line Agents

    [https://arxiv.org/abs/2606.21140](https://arxiv.org/abs/2606.21140)

    本文提出AgentMeter基准测试及新指标AMS，通过联合评估任务质量、预算敏感性和资源密集型零奖励执行，实现对大语言模型与命令行界面（CLI）组合匹配效果的更公平、更全面的质量-效率评估。

    

    大语言模型的快速发展提升了基于命令行界面（CLI）的智能体的任务解决能力，而智能体的CLI决定了模型如何调用工具、维护交互历史以及从故障中恢复。因此，CLI与大语言模型之间的有效匹配变得至关重要。然而，现有的智能体基准测试主要强调成功率，却忽视了成本与效率等实际目标，以及语言模型-CLI组合的选择问题，而这些因素在真实部署场景中都至关重要。为此，我们提出了AgentMeter，一个质量-效率基准测试，并引入了新指标——AgentMeter分数（AMS），该指标联合表征任务质量、预算敏感性以及资源密集型的零奖励执行情况，从而能够对已部署的语言模型-CLI组合进行更全面的评估。此外，所收集的任务描述可能会无意中偏向于与某些CLI特别兼容的……（摘要截断）

    arXiv:2606.21140v3 Announce Type: replace-cross  Abstract: Rapid advances in large language models have improved the task-solving capabilities of command-line-interface (CLI)-based agents, whose CLIs determine how models invoke tools, maintain interaction history, and recover from failures. Consequently, effective matching between CLIs and LLMs has become essential. However, existing agent benchmarks largely emphasize success rate while overlooking practical objectives such as cost and efficiency, as well as the selection of LM-CLI combinations, all of which are critical in real-world deployment. We therefore introduce AgentMeter, a quality-efficiency benchmark with a new metric, the AgentMeter Score (AMS), that jointly characterizes task quality, budget sensitivity, and resource-intensive zero-reward execution, enabling a more complete assessment of deployed LM-CLI pairs. Furthermore, collected task descriptions may inadvertently favor LM-CLI pairs that are particularly compatible wit
    
[^105]: 网络物理系统中的微型机器学习运维：一项实地研究

    Tiny Machine-Learning Operations within Cyber-Physical Systems: a Field Study

    [https://arxiv.org/abs/2606.20952](https://arxiv.org/abs/2606.20952)

    该论文提出并实地验证了一个以知识为中心的端到端微型机器学习运维（TinyMLOps）管线，融合领域物理知识、专家推测与传感器数据流，为资源受限的网络物理系统提供可解释、低内存占用且可部署在设备端的模型。

    

    机器学习运维正在成熟为一门软件工程学科，然而其面向网络物理系统（CPS）中资源受限微控制器的微型版本在工业实践中仍鲜为人知。模型的不透明性、嘈杂的异构数据以及紧张的内存预算阻碍了其在安全关键场景中的落地，在这些场景中，大多数决策仍然依赖人类专家。我们报告了一项针对端到端、以知识为中心的微型机器学习运维管线的实地研究，该管线融合领域物理知识、专家推测和传感器数据流，提供可解释、低内存占用且可部署在设备端的模型。该管线涵盖异构时间序列的自动化采集与清洗、知识驱动的特征构建、可解释的正则化模型，以及概念漂移下的滚动时间交叉验证。我们在来自两次海上风电电缆开沟作业的4.4 GB数据上对其进行了评估。

    arXiv:2606.20952v2 Announce Type: replace  Abstract: Machine-Learning Operations (MLOps) is maturing into a software-engineering discipline, yet its tiny-scale variant (TinyMLOps)-targeting the resource-constrained microcontrollers embedded in cyber-physical systems (CPS)-remains poorly understood in industrial practice. Opaque models, noisy heterogeneous data, and tight memory budgets hinder adoption in safety-critical settings, where most decisions still rely on human experts. We report a field study of an end-to-end, knowledge-centered TinyMLOps pipeline that fuses domain physics, expert speculation, and sensor streams to deliver explainable, low-footprint models deployable on-device. The pipeline spans automated collection and cleaning of heterogeneous time series, knowledge-driven feature construction, interpretable regularized models, and rolling temporal cross-validation under concept drift. We evaluate it on 4.4 GB of data from two offshore-wind cable-trenching campaigns. The c
    
[^106]: 通过闭环记忆优化增强软件工程

    Enhancing Software Engineering Through Closed-Loop Memory Optimization

    [https://arxiv.org/abs/2606.05646](https://arxiv.org/abs/2606.05646)

    提出MemOp闭环记忆优化框架，将记忆效用定义为经过验证的下游影响，使其同时充当与任务无关的评估基准和无需标注的优化信号，从而让软件工程智能体能够跨任务保留、改进和复用经验。

    

    大型语言模型（LLM）已经催生出强大的软件工程（SE）智能体，能够浏览复杂的代码库并解决真实世界的问题。然而，这些智能体在本质上仍然是情景式的：它们无法跨任务保留、改进和复用经验，反复从零开始重建上下文，并重复犯下类似的错误。即使具备记忆支持，由于缺乏一种有原则的、与任务无关的“记忆效用”概念，这些问题也无从解决，使得它们难以被严格评估，也难以在不同智能体和设置之间进行泛化。为了解决这些局限，我们提出了MemOp，一个面向软件工程智能体记忆增强的闭环框架。MemOp将记忆效用建立在经过验证的下游影响之上，使效用既成为一个与任务无关的评估基准，又成为一种无需标注的优化信号。MemOp在四个基准测试、三种编程语言上进行了评估……

    arXiv:2606.05646v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) have enabled powerful software engineering (SE) agents capable of navigating complex codebases and resolving real-world issues. However, these agents remain fundamentally episodic: they fail to retain, refine, and reuse experiences across tasks, repeatedly reconstructing context from scratch and reproducing similar mistakes. Even with memory support, they offer no remedy for the absence of a principled, task-agnostic \textit{memory utility}, making them difficult to evaluate rigorously or generalize across agents and settings. To tackle these limitations, we introduce MemOp, a closed-loop framework for memory augmentation in SE agents. MemOp grounds memory utility in \textit{validated downstream impact}, establishing utility as both a task-agnostic \textbf{evaluation benchmark} and an annotation-free \textbf{optimization signal}. Evaluating MemOp across four benchmarks, three programming languages, 
    
[^107]: EviACT：一个用于智能体程序修复的证据到行动框架

    EviACT: An Evidence-to-Action Framework for Agentic Program Repair

    [https://arxiv.org/abs/2605.27238](https://arxiv.org/abs/2605.27238)

    EviACT框架通过检索支架、编译门和测试驱动门三种证据驱动护栏协调修复流程，在四个基准上使智能体程序修复的解决率提升1.6-6.0个百分点，同时将每缺陷API成本降低70.1-88.6%。

    

    基于大语言模型（LLM）的智能体已将自动化程序修复（APR）从固定上下文的补丁生成推进到交互式的仓库级修复。然而，现有的智能体APR系统在利用执行证据来指导定位、补丁生成和验证方面仍然存在困难。我们提出了EviACT（Evidence-to-Action，证据到行动），一个智能体APR框架，它在修复的各个阶段协调三种由证据驱动的护栏机制：检索支架为修复上下文提供依据，编译门过滤无效的编辑，测试驱动门在全量回归之前检查目标测试的恢复情况。在四个基准测试中，EviACT相比已报告的最强可比基线将解决率提升了1.6-6.0个百分点，并在有基线成本数据的情况下，报告的每缺陷API成本降低了70.1-88.6%。消融实验和诊断分析表明，这些提升与协调一致的证据到行动链条相关，使智能体APR更加高效有效。

    arXiv:2605.27238v2 Announce Type: replace  Abstract: LLM-based agents have moved automated program repair (APR) from fixed-context patch generation to interactive repository-level repair. However, existing agentic APR systems still struggle to use execution evidence to guide localization, patch generation, and validation. We propose EviACT (Evidence-to-Action), an agentic APR framework that coordinates three evidence-driven guardrails across repair stages. The retrieval scaffold grounds repair context, the compile gate filters invalid edits, and the test-driven gate checks target-test recovery before full regression. Across four benchmarks, EviACT improves resolve rate over the strongest reported comparable baselines by 1.6-6.0 percentage points and shows 70.1-88.6% lower reported per-bug API cost where baseline costs are available. Ablations and diagnostics suggest that these gains are associated with the coordinated evidence-to-action chain, making agentic APR more effective and effi
    
[^108]: Firefly：从真实API中点亮大规模可验证工具调用数据生成

    Firefly: Illuminating Large-Scale Verified Tool-Call Data Generation from Real APIs

    [https://arxiv.org/abs/2605.17558](https://arxiv.org/abs/2605.17558)

    FireFly通过先让大模型探索真实API再从观察结果反向合成任务的新范式，从真实MCP服务器生成在构造上保证标签正确性的大规模工具调用训练数据。

    

    训练工具调用智能体需要大规模带有可验证标签的轨迹数据，然而现有方法要么合成偏离真实API行为的环境，要么生成缺乏真实结果以供验证的任务。我们提出了FireFly，一个从真实世界MCP服务器生成可验证工具调用数据的流水线。我们的关键洞察是反转标准的合成流水线：不是先生成任务并期望它们可解，而是先让一个强大的大语言模型沿着图引导的DAG结构探索真实API，然后从观察到的结果反向合成任务，从而在构造上保证标签的正确性。为了应对真实世界工具空间的规模（约1000个工具），我们构建了一个成对工具图并采样子DAG，将探索聚焦在语义连贯的工作流上。为了解决在线API中的环境漂移问题，我们构建了一个检索增强的模拟器，缓存所有探索……

    arXiv:2605.17558v2 Announce Type: replace-cross  Abstract: Training tool-calling agents requires large-scale trajectory data with verifiable labels, yet existing approaches either synthesize environments that diverge from real API behavior or generate tasks without ground-truth outcomes for verification. We present FireFly, a pipeline for generating verified tool-call data from real-world MCP servers. Our key insight is to invert the standard synthesis pipeline: rather than generating tasks and hoping they are solvable, we first let a strong LLM explore real APIs along graph-guided DAG structures, then synthesize tasks backward from observed outcomes, guaranteeing label correctness by construction. To handle the scale of real-world tool spaces (${\sim}$1,000 tools), we build a pairwise tool graph and sample sub-DAGs to focus exploration on semantically coherent workflows. To address environment drift in live APIs, we construct a retrieval-augmented simulator that caches all exploration
    
[^109]: 网页智能体应采用“编译-执行”范式

    Web Agents Should Adopt the Compile-and-Execute Paradigm

    [https://arxiv.org/abs/2605.14290](https://arxiv.org/abs/2605.14290)

    该论文主张网页智能体应放弃 ReAct 这一默认架构，改用“编译-执行”范式——即在观察运行时网页内容之前先确定任务程序再执行，从而防止网页中多来源的不可信内容（如提示注入）篡改智能体的控制流。

    

    ReAct 已成为大语言模型（LLM）智能体的默认架构，许多现有网页智能体也遵循这一范式。我们认为，这对网页智能体而言是错误的默认选择。相反，网页智能体应默认采用“编译-执行”范式：在观察到运行时网页内容之前，先确定一个针对特定任务的程序，然后再执行它。原因在于网页内容混合了来自多方的输入。例如，一个电商产品页面可能同时包含卖家的商品信息、顾客评论、赞助广告以及平台生成的推荐内容。在 ReAct 范式下，所有这些内容在决定下一步动作时都会流入模型，从而为提示注入攻击直接操控智能体的控制流提供了途径。“编译-执行”范式改变了这一边界：不可信的数据或许会影响预定义执行图中的取值或分支，但它无法重新定义用户任务，也无法使模型在运行时合成新的动作。我们分析了 WebArena 这一流行的……（原文摘要至此被截断）

    arXiv:2605.14290v2 Announce Type: replace-cross  Abstract: ReAct has become the default architecture across LLM agents, and many existing web agents follow this paradigm. We argue that it is the wrong default for web agents. Instead, web agents should default to compile-and-execute: commit to a task-specific program before observing runtime web content, then execute it. The reason is that web content mixes inputs from many parties. An e-commerce product page may combine a seller's listing, customer reviews, sponsored advertisements, and platform-generated recommendations. Under ReAct, all of this content flows into the model when deciding on the next action, creating a direct path for prompt injections to steer the agent's control flow. Compile-and-execute changes this boundary: untrusted data may influence values or branches inside a predefined execution graph, but it cannot redefine the user task or cause the model to synthesize new actions at runtime. We analyze WebArena, a popular 
    
[^110]: Terminus-4B：更小的模型能否在智能体执行任务中取代前沿大语言模型？

    Terminus-4B: Can a Smaller Model Replace Frontier LLMs at Agentic Execution Tasks?

    [https://arxiv.org/abs/2605.03195](https://arxiv.org/abs/2605.03195)

    本文提出 Terminus-4B——一个通过 SFT 和基于量规的 LLM-as-judge 强化学习后训练的 Qwen3-4B 模型，证明了经过针对性微调的小型语言模型可以在智能体终端执行任务中达到与前沿大模型相当的性能，从而可作为子智能体替代前沿模型。

    

    现代编程智能体越来越多地将专门的子任务委托给子智能体，这些子智能体是规模更小、更专注的智能体循环，负责处理诸如搜索、调试或终端执行等狭窄的职责。这种架构模式通过在子智能体的上下文中隔离冗长的输出（例如构建日志、测试结果等），来保持主智能体上下文窗口的整洁。通常，当智能体使用子智能体执行此类任务时，它们会采用前沿模型作为这些子智能体。在本文中，我们研究了经过微调的小型语言模型（SLM）能否在智能体终端执行任务中达到与前沿模型相当的性能。我们提出了 Terminus-4B，这是一个针对该任务、通过监督微调（SFT）以及采用基于量规的 LLM-as-judge 奖励的强化学习（RL）进行后训练的 Qwen3-4B 模型。在我们涵盖多种前沿模型、训练消融实验和主智能体配置的广泛评估中……

    arXiv:2605.03195v3 Announce Type: replace  Abstract: Modern coding agents increasingly delegate specialized subtasks to subagents, which are smaller, focused agentic loops that handle narrow responsibilities like search, debugging or terminal execution. This architectural pattern keeps the main agent's context window clean by isolating verbose outputs (e.g. build logs, test results, etc.) within the subagent context. Typically when agents employ subagents for such tasks, they use frontier models as these subagents. In this paper, we investigate whether a finetuned small language model (SLM) can achieve comparable performance to frontier models in the task of agentic terminal execution. We present Terminus-4B, which is a post-trained Qwen3-4B model via Supervised Finetuning (SFT) and Reinforcement Learning (RL) using rubric-based LLM-as-judge reward, specifically for this task. In our extensive evaluation spanning various frontier models, training ablations and main agent configurations
    
[^111]: 智能体化代码审查中的人机协同

    Human-AI Synergy in Agentic Code Review

    [https://arxiv.org/abs/2603.15911](https://arxiv.org/abs/2603.15911)

    该论文通过对300个开源项目中278,790次代码审查对话的大规模实证分析，比较了人类审查者与AI智能体的反馈差异，揭示了人机协作模式及交互方式如何影响代码审查结果与建议采纳率。

    

    代码审查是一项关键的软件工程实践，开发人员在代码集成之前审查代码变更，以确保代码质量、检测缺陷并提升可维护性。近年来，能够理解代码上下文、规划审查行动并与开发环境交互的AI智能体被越来越多地集成到代码审查流程中。然而，目前缺乏实证证据来比较AI智能体与人类审查者在协作工作流中的有效性。为了填补这一空白，我们对300个开源GitHub项目中的278,790次代码审查对话进行了大规模实证分析。在本研究中，我们旨在比较人类审查者与AI智能体所提供的反馈差异。我们研究了审查对话中的人机协作模式，以理解交互如何塑造审查结果。此外，我们分析了人类审查者和AI智能体所提供的代码建议的采纳情况。

    arXiv:2603.15911v2 Announce Type: replace  Abstract: Code review is a critical software engineering practice where developers review code changes before integration to ensure code quality, detect defects, and improve maintainability. In recent years, AI agents that can understand code context, plan review actions, and interact with development environments have been increasingly integrated into the code review process. However, there is limited empirical evidence to compare the effectiveness of AI agents and human reviewers in collaborative workflows. To address this gap, we conduct a large-scale empirical analysis of 278,790 code review conversations across 300 open-source GitHub projects. In our study, we aim to compare the feedback differences provided by human reviewers and AI agents. We investigate human-AI collaboration patterns in review conversations to understand how interaction shapes review outcomes. Moreover, we analyze the adoption of code suggestions provided by human rev
    
[^112]: 基于因果视角解析训练动态在代码模型隐私风险解释中的作用

    A Causal Perspective on the Role of Training Dynamics for Interpreting Privacy Risks in Code Models

    [https://arxiv.org/abs/2512.07814](https://arxiv.org/abs/2512.07814)

    该研究构建了包含多种PII类型的数据集并结合结构因果模型，首次从因果视角证明了个人身份信息的训练动态（可学习性）因果性地决定了其在代码大模型中被泄露的风险，且不同PII类型的泄露风险存在显著异质性。

    

    面向代码的大语言模型（LLM4Code）极大地提升了开发者的生产力，但由于其训练依赖于包含大量个人身份信息（PII）的开源代码库，也引发了隐私方面的担忧。已有研究表明，商业模型能够复现敏感的个人身份信息，然而现有研究大多将PII视为单一类别，忽视了不同类型之间风险的异质性。我们研究了不同类型的PII在被LLM4Code学习和泄露的可能性上是否存在差异，以及这种关系是否具有因果性。我们的研究方法包括：构建一个包含多种PII类型的数据集，微调不同规模的代表性模型，在真实的PII数据上计算训练动态，并构建结构因果模型来估计可学习性对泄露的因果效应。结果表明，不同PII类型之间的泄露风险存在显著差异，且与其训练动态相关（摘要此处被截断）。

    arXiv:2512.07814v3 Announce Type: replace-cross  Abstract: Large language models for code (LLM4Code) have greatly improved developer productivity but also raise privacy concerns due to their reliance on open-source repositories containing abundant personally identifiable information (PII). Prior work shows that commercial models can reproduce sensitive PII, yet existing studies largely treat PII as a single category and overlook the heterogeneous risks among different types. We investigate whether distinct PII types vary in their likelihood of being learned and leaked by LLM4Code, and whether this relationship is causal. Our methodology includes building a dataset with diverse PII types, fine-tuning representative models of different scales, computing training dynamics on real PII data, and formulating a structural causal model to estimate the causal effect of learnability on leakage. Results show that leakage risks differ substantially across PII types and correlate with their trainin
    
[^113]: PBFuzz：面向PoV生成的智能体式定向模糊测试

    PBFuzz: Agentic Directed Fuzzing for PoV Generation

    [https://arxiv.org/abs/2512.04611](https://arxiv.org/abs/2512.04611)

    本文提出PBFuzz，一种模仿人类专家工作方式的智能体式定向模糊测试框架，通过迭代提取语义约束、形成触发假设并利用调试反馈，高效生成漏洞证明输入。

    

    漏洞证明输入生成是软件安全中的一项关键任务，支持漏洞复现和补丁验证等下游应用。生成PoV输入需要解决两组约束：（1）到达漏洞代码位置的可达性约束，（2）激活目标漏洞的触发约束。现有方法（包括定向灰盒模糊测试和LLM辅助模糊测试）难以高效地满足这些约束。本工作提出了一种模仿人类专家的智能体方法。人类分析师会迭代地研究代码以提取语义可达性和触发约束，形成关于PoV触发策略的假设，将其编码为测试输入，并利用调试反馈来完善他们的理解。我们通过一个名为PBFuzz的智能体式定向模糊测试框架将这一过程自动化。PBFuzz解决了智能体PoV生成中的四个挑战……

    arXiv:2512.04611v3 Announce Type: replace-cross  Abstract: Proof-of-Vulnerability (PoV) input generation is a critical task in software security and supports downstream applications such as bug reproduction and patch validation. Generating a PoV input requires solving two sets of constraints: (1) reachability constraints for reaching vulnerable code locations, and (2) triggering constraints for activating the target vulnerability. Existing approaches, including directed greybox fuzzing and LLM-assisted fuzzing, struggle to efficiently satisfy these constraints.   This work presents an agentic method that mimics human experts. Human analysts iteratively study code to extract semantic reachability and triggering constraints, form hypotheses about PoV triggering strategies, encode them as test inputs, and refine their understanding using debugging feedback. We automate this process with an agentic directed fuzzing framework called PBFuzz. PBFuzz tackles four challenges in agentic PoV gene
    
[^114]: 面向程序分析的GPU加速信念传播

    GPU-Accelerated Belief Propagation for Program Analysis

    [https://arxiv.org/abs/2509.22337](https://arxiv.org/abs/2509.22337)

    FastLBP是一个面向程序分析的GPU加速信念传播框架，通过统一表示灵活更新策略、依赖分析实现并行执行以及内存高效的局部结构实现，相比CPU方法最高取得17.42倍加速。

    

    信念传播（BP）是概率图模型（PGMs）中广泛使用的近似推理算法，但在应用于大规模程序分析时计算开销十分高昂。现有的基于GPU的方法无法支持灵活的更新策略，也尚未将逻辑约束与GPU加速相结合，导致在通用性和效率方面均面临挑战。我们提出了FastLBP，一个用于程序分析的GPU加速信念传播框架。我们提出了一种统一表示方法来指定程序分析中所需的灵活更新策略，并提出了一种依赖分析算法以实现并行执行。此外，我们通过为消息计算分配独立线程并采用节省内存的表示方式，在GPU上实现了带局部结构的信念传播。在SmartFL和BINGO上的实验表明，FastLBP相比基于CPU的方法分别实现了平均17.42倍和2.82倍的加速。

    arXiv:2509.22337v3 Announce Type: replace  Abstract: Belief Propagation (BP) is a widely used approximate inference algorithm in probabilistic graphical models (PGMs), but is computationally expensive when applied to large-scale program analysis. Existing GPU-based approaches are unable to support flexible update strategies and have yet to integrate logical constraints with GPU acceleration, leading to challenges in both generality and efficiency.   We present FastLBP, a GPU-accelerated BP framework for program analysis. We propose a unified representation for specifying flexible update strategies required in program analysis, along with a dependency analysis algorithm to enable parallel execution. Furthermore, we implement BP with local structures on GPUs by assigning individual threads to message computations and utilizing a memory-efficient representation. Experiments on SmartFL and BINGO show that FastLBP achieves average speedups of $17.42\times$ and $2.82\times$ over CPU-based ap
    
[^115]: 包里有什么？通过安全敏感API调用实现对依赖项的可见性

    What's in a Package? Getting Visibility Into Dependencies Using Security-Sensitive API Calls

    [https://arxiv.org/abs/2408.02846](https://arxiv.org/abs/2408.02846)

    本文提出了一种有文档记录且可重复的方法，用于推导特定生态系统的安全敏感API列表，帮助开发人员了解依赖包可访问的敏感资源（如文件系统、网络、进程等），从而实现更细粒度的依赖控制和风险识别。

    

    Log4Shell漏洞的发生是因为使用了JNDI（Java命名和目录接口）查找功能，尽管该功能对log4j的核心功能并非必需。这一案例说明了为什么开发人员可能希望了解依赖项所提供的敏感功能。但仅仅存在某个API调用，并不能确定该调用是不必要的或有风险的。安全敏感API提供对安全敏感资源的访问，例如文件系统、网络或与进程相关的资源。如果开发人员知道某个依赖项可以访问哪些类型的敏感资源，他们就可以更细粒度地控制其依赖项，并且当某个依赖项做出不必要或可疑的安全敏感API调用时，可以考虑使用替代的依赖项。然而，先前的工作尚未提供有文档记录、可重复的程序来推导特定生态系统的安全敏感API列表，因此开发人员缺乏系统性的方法……

    arXiv:2408.02846v3 Announce Type: replace-cross  Abstract: The Log4Shell vulnerability occurred because the JNDI (Java Naming and Directory Interface) lookup feature was used, even though it was not needed for log4j's core functionality. This case illustrates why developers may want visibility into the sensitive capabilities available through a dependency. The presence of a call alone does not establish that the call is unnecessary or risky. Security-sensitive APIs provide access to security-sensitive resources, such as filesystem, network, or process-related resources. If developers know which types of sensitive resources a dependency can access, they can control their dependencies at a finer grain, and they can consider an alternative dependency when a dependency makes unnecessary or suspicious security-sensitive API calls. However, prior work has not provided a documented, repeatable procedure for deriving an ecosystem-specific security-sensitive API list, so developers lack systema
    
[^116]: 开发团队更新其易受攻击依赖项的速度有多快？

    How Quickly Do Development Teams Update Their Vulnerable Dependencies?

    [https://arxiv.org/abs/2403.17382](https://arxiv.org/abs/2403.17382)

    本文指出现有的依赖项更新响应性指标MTTU和MTTR因忽略浮动版本和近期更新优先等关键细节而无法准确反映开发团队的更新实践，旨在提出更准确的改进衡量方案。

    

    业界从业者越来越关注包含易受攻击版本的第三方依赖项（包括直接引入和传递引入的）的软件。为解决这一问题，项目被鼓励（a）快速更新到无漏洞版本的依赖项，以及（b）留意其所选用依赖项的更新实践。为此，研究人员提出了衡量包的开发团队在保持依赖项更新方面的响应能力的指标：平均更新时间（MTTU）和平均修复时间（MTTR）。MTTU涵盖所有依赖项，而MTTR量化一个包更新其易受攻击依赖项所需的时间。然而，现有指标未能捕捉重要的细节，例如未考虑浮动版本以及未对最近的更新赋予更高权重，从而无法准确反映开发团队的更新实践。（原文摘要在此处截断）

    arXiv:2403.17382v5 Announce Type: replace  Abstract: Industry practitioners are increasingly concerned with software that contains vulnerable versions of third-party dependencies that are included both directly and transitively. To address this problem, projects are encouraged to both (a)~quickly update to non-vulnerable versions of dependencies and (b)~be mindful of the update practices of the dependencies they choose to use. To this end, researchers have proposed metrics to measure the responsiveness of the development teams of the packages in keeping their dependencies updated: Mean-Time-To-Update (MTTU) and Mean-Time-To-Remediate (MTTR). While MTTU covers all dependencies, MTTR quantifies the time needed for a package to update its vulnerable dependencies. However, existing metrics fail to capture important nuances, such as considering floating versions and prioritizing recent updates, leading to inaccurate reflections of a development team's update practices. \textit{The goal of t
    

