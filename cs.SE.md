# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ParanoiaEval: Benchmarking Unnecessary Defensive Work in Agentic Coding](https://arxiv.org/abs/2610.08662) | 提出了首个统一评估编程智能体风险应对能力的基准ParanoiaEval，基于风险管理中的规避-转移-缓解-接受框架，通过200对证据受控的仓库级任务对和专用评估指标来衡量智能体的防御性工作是否合理。 |
| [^2] | [A Case Study in Assuring AI-Written Software](https://arxiv.org/abs/2610.08651) | 本研究通过对一个由非软件专业操作员运用编码智能体构建并管理的生产级医疗平台进行案例研究，揭示了测试、监控器和审查智能体等AI软件监督机制本身的不可靠性，表明详尽的代码审查不能作为人类控制AI编写软件的唯一依据。 |
| [^3] | [Recursive Game Creator: An Agentic Product-Level Experience-Oriented Game Harness](https://arxiv.org/abs/2610.08621) | 该论文提出递归游戏创造者框架，通过设计者、构建者、玩家和评审者四个智能体的递归协作，将粗糙的游戏原型迭代开发为真正注重玩家体验的有趣游戏。 |
| [^4] | [How Much Evidence Should a Coding Agent's Self-Correction Carry? Adaptive Dirichlet Evidence for Self-Distillation](https://arxiv.org/abs/2610.08514) | 该论文提出有效证据自蒸馏（EESD），利用Dirichlet后验将执行相关性与证据量分开建模，为编码智能体的自我纠正生成经不确定性惩罚的学习权重，在八个观测下相比固定质量方法显著降低了未来结果的NLL。 |
| [^5] | [RAPO-Sol: Retrieval-Augmented Preference Optimization for Repository-Level Solidity Code Generation](https://arxiv.org/abs/2610.08429) | 该论文提出RAPO-Sol两阶段训练框架，将检索增强微调（RAFT）与基于语义锚点扰动（SAP）构建拒绝样本的直接偏好优化（DPO）相结合，以提升仓库级Solidity智能合约代码生成的正确性与语义一致性。 |
| [^6] | [Learning from Failures: A Failure-Driven Prompt Refinement for LLM-Based Vulnerability Analysis](https://arxiv.org/abs/2610.08405) | 本文提出失败驱动提示词优化方法（FDPR），通过分析大语言模型在漏洞分析中的反复失败模式来系统性地改进提示词，实验证明该方法显著提升了基于LLM的漏洞分析可靠性。 |
| [^7] | [Newer and Bigger, but Safer? A Longitudinal Study of the Functionality-Security Gap in LLM-Generated Code](https://arxiv.org/abs/2610.08240) | 本研究对七个模型家族共 32 个大语言模型进行纵向评估，发现新一代模型在绝对安全水平上有所提升，但没有任何模型家族能够弥合生成代码“功能通过却安全失败”的功能-安全差距。 |
| [^8] | [GPU Acceleration of Awkward Arrays: Using Python cuda.compute](https://arxiv.org/abs/2610.08238) | 本文基于 Python CUDA 核心计算库为高能物理中广泛使用的 Awkward Array 库构建了新的 CUDA 执行模型，无需自定义 CUDA 内核即可通过高级 Python 接口实现 GPU 加速，并将多个操作融合到更少的内核中，有效降低了内核启动开销。 |
| [^9] | [Beyond the Leaderboard: Multi-Dimensional Evaluation of Dense and Mixture-of-Experts Models for Automated Program Repair](https://arxiv.org/abs/2610.08173) | 该论文受ISO/IEC 25010启发提出加权质量指数（QI），对稠密与混合专家代码模型进行涵盖正确性、可维护性、安全性和效率的多维度评估，发现模型排名随权重方案变化，单一指标评估会掩盖关键权衡。 |
| [^10] | [When Tools Lie: Reliability of Mathematical Agents Under Corrupted Tool Feedback](https://arxiv.org/abs/2610.08097) | 该论文提出一个受控污染框架研究数学智能体检测和纠正被篡改工具反馈的能力，发现无验证时污染使准确率从100%降至72.4%，而强制同上下文反思可将性能完全恢复至100%。 |
| [^11] | [FC-SWE: Failure-Conditioned RL for Long-Horizon Software Engineering Agents](https://arxiv.org/abs/2610.07898) | 论文提出FC-SWE框架，通过将失败补丁的验证器反馈作为条件上下文，将恢复尝试纳入强化学习策略训练，从而提升长时程软件工程智能体从失败中学习的能力。 |
| [^12] | [Online Sign Language Interpretation System](https://arxiv.org/abs/2610.07872) | 本研究为解决比利时法语区手语译员严重短缺、聋人难以及时获得公共服务的问题，提出并成功测试了远程视频手语翻译方案，验证了其可行性并确定了至少256 kbps（理想为384 kbps）CIF视频的带宽需求。 |
| [^13] | [Harness Engineering for Software Engineering via Modular Executable Dev-Primitives](https://arxiv.org/abs/2610.07832) | 该论文提出Dev-Primitives，一种将代码库工件与常驻LLM配对的模块化可执行抽象，使软件组件从被动工件转变为具备智能体原生接口的主动参与者，从而解决LLM智能体在长程软件工程工作流中反复重建程序状态、上下文爆炸和语义漂移的问题。 |
| [^14] | [ES-Trace: Auditing Ethical-Sourcing Disclosure of Code Generation Models Beyond Model Cards](https://arxiv.org/abs/2610.07762) | 提出ES-Trace框架，利用模型文档可追溯性图将代码生成模型的道德采购披露审计扩展到模型卡之外，发现仅审计模型卡会低估披露水平（1.77/5），追溯引用文档后得分提升至2.82/5。 |
| [^15] | [Acquiring and Verifying Repository Norms for Coding Agents](https://arxiv.org/abs/2610.07757) | 提出RepoNorm框架，独立于编码任务从仓库证据和Git历史中获取并验证显性与隐性规范，并以规范包形式提供给编码代理，显著提升了各类规范合规率。 |
| [^16] | [When Old Facts Return: Re-Reads, Reverts, and the Limits of Temporal Memory](https://arxiv.org/abs/2610.07715) | 该论文揭示了时序记忆系统的一个关键歧义——旧信息的重读与真正的回退会产生相同的观测序列却需要相反的答案——并提出一个拒绝重新激活已淘汰值的守卫机制，该机制能有效防御重读攻击（准确率从10.8%恢复至97.7%），但需要额外的变更溯源信息才能区分合法回退。 |
| [^17] | [HarnessSecurity-Bench: Do Security Mechanisms Really Protect Coding Agent Harnesses?](https://arxiv.org/abs/2610.07639) | 该论文提出了首个针对编码智能体框架安全机制的系统性实证研究与基准 HarnessSecurity-Bench，揭示了约半数安全机制为默认关闭的可选项、闭源框架存在证据缺失，并通过覆盖五类攻击面的 23 个任务评估了六大主流框架中九种机制的真实防护效果。 |
| [^18] | [CISB-Bench: An Auditable Source--IR Dataset of Compiler-Introduced Security Bugs](https://arxiv.org/abs/2610.07635) | CISB-Bench 是一个从 GCC 和 LLVM 中挖掘出的可审计数据集，包含 429 条带有标准化 LLVM IR 分析、公开来源和二分类标注的 C 程序数据，涵盖 280 个编译器引入安全缺陷（CISB）和 149 个困难非 CISB 案例，为研究编译器引入的安全缺陷提供了可验证的基准。 |
| [^19] | [CheckerBench: Can Long-Horizon Agents Synthesize Static-Analysis Checkers?](https://arxiv.org/abs/2610.07557) | 该论文提出了首个可执行基准CheckerBench（包含源自297个CVE、167个仓库的300个任务），用于评估长程智能体能否在真实代码仓库中端到端合成可用的静态分析检查器，并配套CheckerLab统一评估框架衡量诊断对比度、补丁定位、误报率和工具使用等指标。 |
| [^20] | [The EPIC Framework for Spec-Driven Development](https://arxiv.org/abs/2610.07534) | 该研究基于ISO/IEC/IEEE 29148标准对114个开源SDD仓库进行评分分析，提出了包含10个质量维度、40项实践的EPIC框架，帮助开发人员在规范驱动开发中为编码智能体编写更明确、更完整的规范、计划和任务。 |
| [^21] | [CogAdapt: Cognition-informed Sparse Adaptation of Code LLMs](https://arxiv.org/abs/2610.07446) | 提出了CogAdapt框架，利用人类阅读代码时产生的认知信号来指导代码大模型的稀疏选择性适应，在不牺牲性能的前提下显著降低模型微调成本。 |
| [^22] | [A Validated Dataset and Benchmark for Coherent Multi-Diagram SysML Models](https://arxiv.org/abs/2610.07356) | 该论文提出了SEMAADB——一个包含3,000个工程情境、15,000张经过一致性和有效渲染验证的SysML多视图图的大规模数据集与基准，用于评估大语言模型生成连贯多图系统建模的能力。 |
| [^23] | [Catching Developers in the Flow: Low-Latency Agentic Program Repair at Google Scale](https://arxiv.org/abs/2610.07289) | 本文提出部署于Google的AI智能体FlowAgent，通过ReAct风格的生成-验证循环与弃权过滤器，在持续集成的提交前阶段以低延迟实时自动修复测试失败，使开发者无需切换上下文即可在心流中获得高质量修复建议。 |
| [^24] | [SAFESHIELD: A Decision-Organization Framework for Deployment-Time Safety of Small Language Models](https://arxiv.org/abs/2610.07276) | 本文提出SAFESHIELD框架，将小语言模型的部署时安全形式化为决策组织问题，通过组织准入、路由、证据和发布四种安全决策职责，并将决策记录于可审计的决策轨迹中，实现了安全决策的显式组织、协调与审计。 |
| [^25] | [Pattern-Guided Graph Synthesis for Suppressing Known Defects in DL Compiler Fuzzing](https://arxiv.org/abs/2610.06968) | 提出 Reprise——一种深度学习编译器模糊测试工具，它将每个已发现的缺陷提炼为包含算子、值约束、图上下文和数据流的语义图模式，并在图合成阶段重新生成会匹配已知模式的节点，从而在编译执行之前就从源头避免重复触发已知缺陷，而非依赖事后去重。 |
| [^26] | [ASAP: Assembly-Source Aligned Pseudocode Refinement For Binary Decompilation](https://arxiv.org/abs/2610.06900) | ASAP通过对比对齐学习汇编与源码的对应表示，并借助Q-Former压缩汇编特征、结合随机掩码与相对汇编优势损失，有效提升了LLM对二进制反编译伪代码的优化质量，尤其应对了激进优化带来的反编译错误。 |
| [^27] | [When Does a Second Model Help? Cross-Model Review in LLM Verification](https://arxiv.org/abs/2610.01471) | 在大语言模型输出验证中，跨模型审查与同模型审查发现的错误集合部分不同，且“一次同模型新会话审查加一次跨模型审查”的组合比两次同模型审查能匹配到更多埋设错误（56.7% vs. 42.7%）。 |
| [^28] | [CLAD: Constrained Abstract Domain for Neural Network Verification](https://arxiv.org/abs/2609.34628) | 提出了约束拉格朗日抽象域（CLAD），能够在Lp范数球附加额外约束的复杂输入区域上计算神经网络行为更紧致的可靠过近似，从而克服现有抽象域因输入区域描述受限而导致的验证失败或虚假反例问题。 |
| [^29] | [ZonoGPT: Towards An Abstract Domain for Verifying Large GPT Models](https://arxiv.org/abs/2609.34457) | ZonoGPT提出了一种空间复杂度与网络深度无关的抽象域，通过结构化zonotope、生成元约简机制以及针对Attention、LayerNorm和GELU的保精度变换，实现了对大型GPT模型的高效形式化验证。 |
| [^30] | [CLEAR: Causal Context-Based Agentic Reasoning for Vulnerability Detection](https://arxiv.org/abs/2608.03134) | 该论文提出CLEAR框架，通过构建建模入口点、前置条件、根本原因和修复意图之间因果链的漏洞因果知识图谱，并配合多智能体推理，克服现有方法仅关注表面相似性的局限，实现对源代码漏洞深层因果依赖的检测。 |
| [^31] | [Palette: A Modular, Controllable, and Efficient Framework for On-demand Authorized Safety Alignment Relaxation in LLMs](https://arxiv.org/abs/2605.24154) | Palette 提出了一个模块化、可控且高效的框架，通过多目标搜索识别拒绝方向并借助轻量级适配将其内化到模型中，从而按需放宽授权领域的安全拒绝行为，同时保持其他领域的标准安全性。 |
| [^32] | [PBT-Bench: Benchmarking AI Agents on Property-Based Testing](https://arxiv.org/abs/2605.15229) | PBT-Bench是一个包含100个覆盖40个真实Python库的基于属性测试问题的基准，通过注入默认随机输入几乎无法触发的语义bug，专门评估AI智能体从文档中推导语义不变量并设计精确输入生成策略的能力。 |
| [^33] | [Uncovering Business Logic Bugs via Semantics-Driven Unit Test Generation](https://arxiv.org/abs/2604.23509) | SeGa 通过从产品需求文档构建语义知识库，并推导出包含前置条件、触发动作、预期结果和语义约束的细粒度业务场景来指导大语言模型生成单元测试，从而比现有最先进技术多发现 22-25 个业务逻辑缺陷。 |
| [^34] | [Finding Memory Leaks in C/C++ Programs via Neuro-Symbolic Augmented Static Analysis](https://arxiv.org/abs/2603.27224) | MemHint结合大语言模型的代码语义理解与基于Z3的符号推理验证，识别项目自定义内存管理函数并过滤不可行的函数摘要，从而增强静态分析器检测C/C++程序内存泄漏的能力。 |
| [^35] | [ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization](https://arxiv.org/abs/2602.15983) | ReLoop通过结合结构化生成和行为验证，有效缩小了大语言模型在优化代码生成中的可行性与正确性差距。 |
| [^36] | [Theory building in software engineering: Operationalization](https://arxiv.org/abs/2412.02384) | 本文系统化了软件工程理论构建中的操作化阶段，将概念化阶段得到的概念和命题转化为明确的构念和可经验检验的假设，并通过三个维度（实际程序步骤、形式化数学规范和实证案例评估）进行阐述。 |

# 详细

[^1]: ParanoiaEval：智能体编程中不必要防御性工作的基准测试

    ParanoiaEval: Benchmarking Unnecessary Defensive Work in Agentic Coding

    [https://arxiv.org/abs/2610.08662](https://arxiv.org/abs/2610.08662)

    提出了首个统一评估编程智能体风险应对能力的基准ParanoiaEval，基于风险管理中的规避-转移-缓解-接受框架，通过200对证据受控的仓库级任务对和专用评估指标来衡量智能体的防御性工作是否合理。

    

    随着编程智能体日益自主地承担真实世界的工作，判断其风险应对措施是否合理已变得尤为重要。现有工作从各自独立的角度评估相关的智能体行为，但缺乏一个统一这些行为的系统性框架。为弥合这一差距，我们提出了ParanoiaEval——首个用于统一评估编程智能体风险应对能力的基准。该基准以软件工程风险管理中成熟的“规避-转移-缓解-接受”（Avoidance-Transfer-Mitigation-Acceptance）框架为基础，将这4种基本风险应对措施操作化到编程智能体场景中，并包含200对证据受控的仓库级任务对，每对任务仅在定义应对措施的证据上存在差异。我们进一步引入了针对风险应对违规和证据响应性的专用指标，并采用经过人类校准的智能体裁判以实现可靠评估。在8个代表性模型上进行的大规模实验……

    arXiv:2610.08662v1 Announce Type: new  Abstract: As coding agents increasingly undertake real-world work autonomously, judging whether their risk treatments are warranted has become important. Existing work evaluates related agent behaviors from separate perspectives, but lacks a systematic framework for unifying these behaviors. To bridge this gap, we introduce ParanoiaEval, the first benchmark for unified evaluation of risk-treatment capabilities in coding agents. Grounded in the well-established Avoidance-Transfer-Mitigation-Acceptance framework in software engineering risk management, ParanoiaEval operationalizes its 4 fundamental treatments for coding-agent settings and contains 200 evidence-controlled repository-level task pairs, each differing only in treatment-defining evidence. We further introduce dedicated metrics for risk-treatment violations and evidence responsiveness, using a human-calibrated agentic judge for reliable evaluation. Large-scale experiments on 8 representat
    
[^2]: 保障AI编写软件的案例研究

    A Case Study in Assuring AI-Written Software

    [https://arxiv.org/abs/2610.08651](https://arxiv.org/abs/2610.08651)

    本研究通过对一个由非软件专业操作员运用编码智能体构建并管理的生产级医疗平台进行案例研究，揭示了测试、监控器和审查智能体等AI软件监督机制本身的不可靠性，表明详尽的代码审查不能作为人类控制AI编写软件的唯一依据。

    

    软件工程智能体可以让没有接受过正规软件培训的人构建他们原本无法实现的系统，同时其生成的代码量也可能超出即使是最资深专家所能有效审查的范围。在这两种情况下，详尽的代码审查都不能作为人类控制的唯一可靠依据。我们报告了一项案例研究，研究对象是一个通过编码智能体构建、并由一名没有接受过正规软件工程培训的操作员管理的生产级医疗保健平台。随着时间推移，其工作流程演变为一个以人类为主导的元智能体系统：一个智能体负责编写代码，其他智能体负责监督和审查，项目规则则将经验教训传承下去。操作员发现，用于监督该系统的测试、监控器和审查智能体本身也是不可靠的：一些监控器测量的是代理指标而非实际结果，一些审计会静默失败，缺失的检查项会从报告结果中消失，还有一个自动化修复操作甚至造成了运营中断。在本案例中，

    arXiv:2610.08651v1 Announce Type: cross  Abstract: Software-engineering agents can enable people without formal software training to build systems they could not otherwise implement and simultaneously can produce more code than even experts can meaningfully inspect. In both cases, exhaustive code review is not reliable as the sole basis for human control. We report a case study of a production healthcare platform built through coding agents and governed by an operator without formal software-engineering training. Over time, its workflow grew into a human-led meta-agent system where one agent wrote code, other agents supervised and reviewed it, and project rules carried lessons forward. The operator found that tests, monitors and reviewing agents used to supervise the system were fallible. Some monitors measured proxies rather than outcomes, some audits failed silently, missing checks disappeared from reported results and one automated repair caused operational disruption. In this case,
    
[^3]: 递归游戏创造者：一个面向体验的智能体级产品游戏开发框架

    Recursive Game Creator: An Agentic Product-Level Experience-Oriented Game Harness

    [https://arxiv.org/abs/2610.08621](https://arxiv.org/abs/2610.08621)

    该论文提出递归游戏创造者框架，通过设计者、构建者、玩家和评审者四个智能体的递归协作，将粗糙的游戏原型迭代开发为真正注重玩家体验的有趣游戏。

    

    近期的游戏设计智能体在生成可玩游戏方面取得了长足进步。然而，程序的正确性并不能保证玩家获得愉快的游戏体验。我们提出了递归游戏创造者，这是一个面向体验的框架，旨在将智能体游戏开发从粗糙的游戏原型推进为有趣的游戏。递归游戏创造者围绕四个组件组织递归式开发：设计者、构建者、玩家和评审者。设计者将用户指令和评审者的反馈转化为详细的计划。构建者将这些计划转化为候选游戏。基于代码原生的玩家通过编程接口创建并执行可复用的策略，以高效收集多样化的游戏玩法轨迹，缓解了基于图形界面（GUI）的缓慢收集方式所导致的评估偏差。评审者使用精心设计的基于轨迹的指标来推断玩家偏好，并结合视觉证据和明确的文本偏好（进行综合评估）。

    arXiv:2610.08621v1 Announce Type: new  Abstract: Recent game design agents have made substantial progress in generating playable games. However, program correctness does not ensure an enjoyable experience for players. We present Recursive Game Creator, an experience-oriented harness to advance agentic game development from rough game prototypes into entertaining games. Recursive Game Creator organizes recursive development around four components: Designer, Builder, Player, and Reviewer. The Designer translates user instructions and Reviewer's feedback into detailed plans. The Builder turns these plans into candidate games. The coding-native Player creates and executes reusable policies through programmatic interfaces to efficiently collect diverse gameplay trajectories, mitigating evaluation bias caused by slow GUI-based collection. The Reviewer uses carefully designed trajectory-based metrics to induce player preferences, integrating with visual evidence and explicit textual preferenc
    
[^4]: 编码智能体的自我纠正应承载多少证据？面向自蒸馏的自适应Dirichlet证据

    How Much Evidence Should a Coding Agent's Self-Correction Carry? Adaptive Dirichlet Evidence for Self-Distillation

    [https://arxiv.org/abs/2610.08514](https://arxiv.org/abs/2610.08514)

    该论文提出有效证据自蒸馏（EESD），利用Dirichlet后验将执行相关性与证据量分开建模，为编码智能体的自我纠正生成经不确定性惩罚的学习权重，在八个观测下相比固定质量方法显著降低了未来结果的NLL。

    

    执行反馈使编码智能体能够修改程序并从自身的纠正中学习。一次纠正的学习权重应当同时反映其执行所支持的转移，以及该支持背后证据的数量。我们提出了有效证据自蒸馏，它将这两个量分开表示：归一化的执行相关性决定相对转移支持和有效伪计数质量；随后由Dirichlet后验产生一个经不确定性惩罚的权重，用于基于KL锚定的纠正学习。在对称先验下，改变质量可保持类别排序，且有效质量得到的监督系数以其匹配的固定质量对应值为上界。在四个模型-领域的历史扫描实验中，将可见观测次数从一增加到八，可使未来结果的NLL降低55.0–59.3%。在八次观测时，有效质量在全部四个对比中均取得了比固定质量更低的NLL。

    arXiv:2610.08514v1 Announce Type: new  Abstract: Execution feedback lets coding agents revise programs and learn from their own corrections. A correction's learning weight should reflect both the transitions supported by its executions and the amount of evidence behind that support. We introduce Effective-Evidence Self-Distillation (EESD), which represents these quantities separately. Normalized execution relevance determines relative transition support and an effective pseudo-count mass; a Dirichlet posterior then produces an uncertainty-penalized weight for KL-anchored correction learning. Under a symmetric prior, changing mass preserves category ordering, and effective mass yields a supervised coefficient bounded by its matched fixed-mass counterpart. Across four model-domain history sweeps, increasing visible observations from one to eight reduces future-outcome NLL by 55.0-59.3%. At eight observations, effective mass achieves lower NLL than fixed mass in all four comparisons. In t
    
[^5]: RAPO-Sol：面向仓库级Solidity代码生成的检索增强偏好优化

    RAPO-Sol: Retrieval-Augmented Preference Optimization for Repository-Level Solidity Code Generation

    [https://arxiv.org/abs/2610.08429](https://arxiv.org/abs/2610.08429)

    该论文提出RAPO-Sol两阶段训练框架，将检索增强微调（RAFT）与基于语义锚点扰动（SAP）构建拒绝样本的直接偏好优化（DPO）相结合，以提升仓库级Solidity智能合约代码生成的正确性与语义一致性。

    

    用Solidity编写的智能合约管理着资产、权限和不可逆的状态变更，这使得代码生成既具有实用价值又关乎安全关键。仓库级的Solidity生成极具挑战性，因为模型必须在合成完整合约或库的同时，保持状态变量、修饰符、事件、继承、外部调用和访问控制逻辑之间的一致性。我们提出了RAPO-Sol，这是一个用于仓库级Solidity代码生成的两阶段训练框架。首先，检索增强微调（RAFT）通过相似的Solidity示例来增强每个训练输入，帮助模型学习重复出现的合约级模式，同时在推理时无需检索。其次，直接偏好优化（DPO）训练模型更倾向于参考合约，而非那些接近但存在语义缺陷的替代方案。我们使用Solidity语义锚点扰动（SAP）构建被拒绝的样本，通过对有效……（摘要原文在此处截断）

    arXiv:2610.08429v1 Announce Type: new  Abstract: Smart contracts written in Solidity manage assets, permissions, and irreversible state changes, making code generation both useful and security-critical. Repository-level Solidity generation is challenging because models must synthesize complete contracts or libraries while preserving consistency across state variables, modifiers, events, inheritance, external calls, and access-control logic. We present RAPO-Sol, a two-stage training framework for repository-level Solidity code generation. First, Retrieval-Augmented Fine-Tuning (RAFT) augments each training input with similar Solidity examples, helping the model learn recurring contract-level patterns while remaining retrieval-free at inference time. Second, Direct Preference Optimization (DPO) trains the model to prefer reference contracts over close but semantically flawed alternatives. We construct rejected samples using Solidity Semantic-Anchor Perturbation (SAP), which perturbs vali
    
[^6]: 从失败中学习：一种面向基于大语言模型漏洞分析的失败驱动提示词优化方法

    Learning from Failures: A Failure-Driven Prompt Refinement for LLM-Based Vulnerability Analysis

    [https://arxiv.org/abs/2610.08405](https://arxiv.org/abs/2610.08405)

    本文提出失败驱动提示词优化方法（FDPR），通过分析大语言模型在漏洞分析中的反复失败模式来系统性地改进提示词，实验证明该方法显著提升了基于LLM的漏洞分析可靠性。

    

    大语言模型已成为软件漏洞分析领域颇具前景的工具，但其有效性在很大程度上取决于提示词的设计。现有研究主要使用聚合性能指标来比较各种提示策略，对于模型为何失败以及如何系统地改进提示词所提供的见解有限。我们提出了失败驱动提示词优化方法（FDPR），这是一种通过分析模型反复出现的失败来指导基于证据的提示词改进的方法论。基于Damn Vulnerable Java Application（DVJA），我们识别出反复出现的失败模式，包括误报、漏报、无依据推理和CWE错误分类，并将其转化为有针对性的提示词优化。随后，我们在Juliet测试套件上对优化后的提示词进行评估，并通过跨模型验证来评估其泛化能力。结果表明，失败驱动的优化方法提高了基于大语言模型的漏洞分析的可靠性。

    arXiv:2610.08405v1 Announce Type: cross  Abstract: Large Language Models have emerged as promising tools for software vulnerability analysis, but their effectiveness depends heavily on prompt design. Existing research primarily compares prompting strategies using aggregate performance metrics, providing limited insight into why models fail or how prompts can be improved systematically. We propose Failure-Driven Prompt Refinement (FDPR), a methodology that analyzes recurring model failures to guide evidence-based prompt refinement. Using the Damn Vulnerable Java Application (DVJA), we identify recurring failure modes, including false positives, false negatives, unsupported reasoning, and CWE misclassification, and translate them into targeted prompt refinements. We then evaluate the resulting prompt on the Juliet Test Suite and perform cross-model validation to assess generalizability. The results show that failure-driven refinement improves the reliability of LLM-based vulnerability an
    
[^7]: 更新更大，但更安全了吗？——LLM 生成代码中“功能-安全”差距的纵向研究

    Newer and Bigger, but Safer? A Longitudinal Study of the Functionality-Security Gap in LLM-Generated Code

    [https://arxiv.org/abs/2610.08240](https://arxiv.org/abs/2610.08240)

    本研究对七个模型家族共 32 个大语言模型进行纵向评估，发现新一代模型在绝对安全水平上有所提升，但没有任何模型家族能够弥合生成代码“功能通过却安全失败”的功能-安全差距。

    

    大语言模型（LLM）被广泛用于生成代码。尽管其功能合理性不断提升，但生成的代码往往包含安全漏洞。“功能-安全差距”指的就是那些能通过功能测试却无法通过安全测试的代码。最近一项针对三个模型家族的纵向研究得出结论：LLM 变得更聪明但没有变得更安全，其中唯一被考察的开源权重模型家族处于停滞状态。这一结论是否适用于其他（开源权重）模型家族，尤其是紧凑型模型，仍然悬而未决。我们开展了一项纵向研究，考察来自七个模型家族（其中五个为开源权重）的 32 个 LLM 在该差距上的表现，每个家族涵盖旗舰版与紧凑版共三个连续发布版本。借助 CWEval 基准（包含五种编程语言中的 119 个任务和 31 类 CWE），我们比较了不同家族、模型规模和编程语言之间的安全演进轨迹。研究发现：较新的模型在绝对意义上确实变得更加安全，但没有任何模型家族能够弥合这一差距。（注：原文摘要在此处被截断）

    arXiv:2610.08240v1 Announce Type: new  Abstract: Large Language Models (LLMs) are widely used to generate code. Although their functional plausibility keeps improving, the generated code often contains security vulnerabilities. The functionality-security gap captures code that passes functional tests but fails security tests. A recent longitudinal study of three model families concluded that LLMs become smarter but not safer, with the only considered open-weight family stagnating. Whether this holds for other (open-weight) families and particularly for compact models remains open. We present a longitudinal study of the gap across 32 LLMs from seven model families (five open-weight), covering three successive releases per family in flagship and compact variants. Using CWEval with 119 tasks in five programming languages and 31 CWEs, we compare trajectories across families, model sizes, and languages. Newer models do become safer in absolute terms, although no family closes the gap. Unlik
    
[^8]: Awkward 数组的 GPU 加速：使用 Python cuda.compute

    GPU Acceleration of Awkward Arrays: Using Python cuda.compute

    [https://arxiv.org/abs/2610.08238](https://arxiv.org/abs/2610.08238)

    本文基于 Python CUDA 核心计算库为高能物理中广泛使用的 Awkward Array 库构建了新的 CUDA 执行模型，无需自定义 CUDA 内核即可通过高级 Python 接口实现 GPU 加速，并将多个操作融合到更少的内核中，有效降低了内核启动开销。

    

    Awkward Array 是高能物理（HEP）领域广泛使用的 Python 库，用于表示和操作嵌套的、可变长度的数据。此前的 CHEP 会议工作已经探索了 Awkward Array 的 GPU 加速，展示了基于 CUDA 后端的可行性和性能优势，同时也指出了在不规则数据访问、细粒度内核启动以及操作可组合性方面的局限性。在本工作中，我们展示了直接建立在这些早期努力之上的最新进展，即基于 Python CUDA 核心计算库为 Awkward Array 引入了 CUDA 执行模型。使用 CCCL，我们无需编写自定义 CUDA 内核，而是可以使用高级 Python 接口。基于 CCCL 的方法还能够将多个 Awkward 操作融合到更少数量的 CUDA 内核中，从而解决了早期 GPU 实现中观察到的内核启动开销问题。惰性执行……（摘要在此处截断）

    arXiv:2610.08238v1 Announce Type: cross  Abstract: Awkward Array is a widely used library in high-energy physics (HEP) for representing and manipulating nested, variable-length data in Python. Previous CHEP contributions have explored GPU acceleration for Awkward Array, demonstrating the feasibility and performance benefits of CUDA-based backend while also identifying limitations related to irregular data access, fine-grained kernel launches, and composability of operations. In this contribution, we present recent developments that build directly on these earlier efforts by introducing a CUDA execution model for Awkward Array based on the Python CUDA Core Compute Libraries (CCCL).   Using CCCL, we eliminate the need for custom CUDA kernels and can instead use a high-level Python interface. The CCCL-based approach also enables fusion of multiple Awkward operations into a reduced number of CUDA kernels, addressing kernel launch overhead observed in earlier GPU implementations. Lazy execu
    
[^9]: 超越排行榜：面向自动程序修复的稠密模型与混合专家模型的多维度评估

    Beyond the Leaderboard: Multi-Dimensional Evaluation of Dense and Mixture-of-Experts Models for Automated Program Repair

    [https://arxiv.org/abs/2610.08173](https://arxiv.org/abs/2610.08173)

    该论文受ISO/IEC 25010启发提出加权质量指数（QI），对稠密与混合专家代码模型进行涵盖正确性、可维护性、安全性和效率的多维度评估，发现模型排名随权重方案变化，单一指标评估会掩盖关键权衡。

    

    使用语言模型进行自动程序修复（APR）通常仅通过生成的补丁能否通过测试套件来评估，这可能掩盖模型在可维护性、安全性和计算成本方面的差异。我们受ISO/IEC 25010软件质量模型的启发，提出了一个加权质量指数（QI），该指数在可配置的权重方案下综合考量功能正确性、可维护性、安全性和生成效率。我们在40个QuixBugs和90个Defects4J缺陷上评估了三个稠密Qwen2.5-Coder模型（3B、7B、14B）以及160亿参数的DeepSeek-Coder-V2-Lite混合专家（MoE）模型（24亿激活参数），所有实验均在相同硬件上本地运行，以控制基础设施因素的影响。结果显示模型排名随权重方案而变化，表明单一指标的评估可能掩盖模型间的权衡取舍。MoE模型在正确性方面与7B和14B稠密模型几乎没有统计学上的显著差异（McNemar精确检验）。

    arXiv:2610.08173v1 Announce Type: cross  Abstract: Automated Program Repair (APR) with language models is usually evaluated by whether a generated patch passes the test suite, which can hide differences in maintainability, security, and computational cost. We propose a Weighted Quality Index (QI), inspired by the ISO/IEC 25010 software quality model, that combines functional correctness, maintainability, security, and generation efficiency under configurable weighting schemes. We evaluate three dense Qwen2.5-Coder models (3B, 7B, 14B) and the 16B-parameter DeepSeek-Coder-V2-Lite Mixture-of-Experts (MoE) model (2.4B active parameters) on 40 QuixBugs and 90 Defects4J bugs, all run locally on identical hardware to control for infrastructure effects. Model rankings change with the weighting scheme, showing that single-metric evaluation can hide trade-offs. The MoE model shows almost no statistically significant difference in correctness from the 7B and 14B dense models (McNemar's exact tes
    
[^10]: 当工具撒谎时：受污染工具反馈下数学智能体的可靠性

    When Tools Lie: Reliability of Mathematical Agents Under Corrupted Tool Feedback

    [https://arxiv.org/abs/2610.08097](https://arxiv.org/abs/2610.08097)

    该论文提出一个受控污染框架研究数学智能体检测和纠正被篡改工具反馈的能力，发现无验证时污染使准确率从100%降至72.4%，而强制同上下文反思可将性能完全恢复至100%。

    

    数学问题求解通常需要确定性的计算步骤，智能体会将这些步骤委托给工具并隐式地信任它们。然而，工具可能会无声地失效，返回看似合理但错误的结果。智能体能在多大程度上检测并纠正被污染的工具调用输出？我们通过一个受控污染框架来研究这个问题：在该框架中，一个隐藏的拦截器会在特定问题上将工具调用结果替换为看似合理的错误信息。我们在31个问题上评估了智能体，采用四种验证设计，包括无验证（基线）、强制同上下文反思、可选的新上下文验证以及可选的结构化验证。在没有验证的情况下，污染导致准确率大幅下降，从100%降至72.4%。强制反思能将性能完全恢复至100%。只有当模型主动调用时，可选验证才能提升准确率。我们的结果表明，检查频率与（原文在此处截断）

    arXiv:2610.08097v1 Announce Type: cross  Abstract: Mathematical problem solving often requires deterministic computational steps that agents delegate to tools and implicitly trust. Yet tools can fail silently, returning plausible but incorrect results. How well can agents detect and correct corrupted tool call outputs? We study this through a controlled corruption framework where a hidden interceptor replaces tool call results with plausible incorrect information on targeted problems. We evaluate agents across 31 problems under four verification designs including no verification (baseline), mandatory same-context reflection, optional fresh-context verification, and optional structural verification. Without verification, corruption causes dramatic accuracy loss, from 100% down to 72.4%. Mandatory reflection fully recovers this performance to 100%. Optional verification improves accuracy only when models actively invoke it. Our results show that checking frequency is strongly associated 
    
[^11]: FC-SWE：面向长时程软件工程智能体的失败条件强化学习

    FC-SWE: Failure-Conditioned RL for Long-Horizon Software Engineering Agents

    [https://arxiv.org/abs/2610.07898](https://arxiv.org/abs/2610.07898)

    论文提出FC-SWE框架，通过将失败补丁的验证器反馈作为条件上下文，将恢复尝试纳入强化学习策略训练，从而提升长时程软件工程智能体从失败中学习的能力。

    

    仓库级软件工程（SWE）是一个具有挑战性的长时程任务场景：智能体需要在长时间的交互中进行推理、使用工具，并适应有状态的环境。近期的研究工作使用组相对策略优化（GRPO）等强化学习方法来训练SWE智能体，该方法针对每个问题独立采样多条轨迹，测试生成的补丁，并在固定组内比较终端奖励。然而，这种训练设置并未将失败补丁的验证器反馈作为后续尝试的上下文加以复用，尽管这些反馈包含了关于出错原因的宝贵诊断信息。在恢复轨迹上进行训练具有挑战性，因为前一次的结果决定了下一条轨迹是否会被生成，而失败的执行则决定了其条件上下文。我们提出了FC-SWE，这是一个将恢复尝试纳入策略训练的失败条件强化学习框架。

    arXiv:2610.07898v1 Announce Type: new  Abstract: Repository-level software engineering (SWE) is a challenging long-horizon setting: agents must reason over extended interactions, use tools, and adapt to stateful environments. Recent work trains SWE agents with reinforcement learning methods such as Group Relative Policy Optimization (GRPO), which independently sample multiple trajectories per issue, test the resulting patches, and compare terminal rewards within a fixed group. However, this training setup does not reuse verifier feedback from failed patches as context for subsequent attempts, even though this feedback contains valuable diagnostic information about what went wrong. Training on recovery trajectories is challenging because the preceding outcome determines whether the next trajectory is generated, while the failed execution determines its conditioning context. We introduce FC-SWE, a failure-conditioned RL framework that incorporates recovery attempts into policy training. 
    
[^12]: 在线手语翻译系统

    Online Sign Language Interpretation System

    [https://arxiv.org/abs/2610.07872](https://arxiv.org/abs/2610.07872)

    本研究为解决比利时法语区手语译员严重短缺、聋人难以及时获得公共服务的问题，提出并成功测试了远程视频手语翻译方案，验证了其可行性并确定了至少256 kbps（理想为384 kbps）CIF视频的带宽需求。

    

    arXiv:2610.07872v1 通告类型：cross 摘要：瓦隆大区委托开展了这项研究，旨在改善聋人在新千年获取公共服务的途径。研究聚焦于手语使用者群体——在法语社区约有25,000名成年人，对他们而言书面法语几乎相当于一门外语。核心问题在于手语译员严重短缺：在比利时法语区仅有约二十名译员在工作，因此预约往往需要数周时间安排，且无法应对紧急情况。该研究评估了三种方案，其中被推荐的方案为远程视频手语翻译，在撰写本文时即可部署：专业译员通过视频会议同时加入聋人与工作人员的对话，从而节省路途时间并能够提供紧急服务。类似服务已在瑞典、芬兰、荷兰和法国运行。2003年5月的原型测试取得了明显的成功，并表明手语翻译至少需要高于256 kbps的CIF视频质量，理想情况下为384 kbps。该系统适用于简单的……

    arXiv:2610.07872v1 Announce Type: cross  Abstract: The Walloon Region commissioned this study to improve deaf people's access to public services for the new millennium. It focuses on sign language users, about 25,000 adults in the French-speaking Community, for whom written French is close to a foreign language. The core problem is a severe shortage of interpreters: only about twenty work in French-speaking Belgium, so appointments take weeks to arrange and emergencies cannot be covered.   The study assessed three options. The recommended one, remote video interpretation, could be deployed at the time of writing: a professional interpreter joins the deaf person and the employee by videoconference, saving travel time and enabling an emergency service. Similar services already ran in Sweden, Finland, the Netherlands and France. Prototype tests in May 2003 were a clear success and showed that interpretation needs at least CIF video above 256 kbps, ideally 384 kbps. The system suits simple
    
[^13]: 通过模块化可执行的开发原语为软件工程构建工程化框架

    Harness Engineering for Software Engineering via Modular Executable Dev-Primitives

    [https://arxiv.org/abs/2610.07832](https://arxiv.org/abs/2610.07832)

    该论文提出Dev-Primitives，一种将代码库工件与常驻LLM配对的模块化可执行抽象，使软件组件从被动工件转变为具备智能体原生接口的主动参与者，从而解决LLM智能体在长程软件工程工作流中反复重建程序状态、上下文爆炸和语义漂移的问题。

    

    配备终端访问能力的大型语言模型（LLMs）在自动化软件工程任务方面已展现出强大的能力。然而，现有智能体在长程工作流中依然十分脆弱：它们必须反复重建分散在源代码文件、配置、测试、依赖项和运行时行为中的程序状态，导致交互历史不断膨胀、上下文爆炸以及语义漂移。大型代码库则进一步增加了识别与任务相关组件的难度。为了应对这些挑战，我们提出了Dev-Primitives（开发原语），这是一种模块化且可执行的抽象，它将代码库组件从被动的软件工件转变为软件工程中的主动参与者。每个Dev-Primitive将一个代码库工件与一个常驻LLM配对，从而赋予该工件一个基于其自身实现和依赖关系的智能体原生接口。

    arXiv:2610.07832v1 Announce Type: cross  Abstract: Large language models (LLMs) equipped with terminal access have demonstrated strong capabilities in automating software engineering tasks. However, existing agents remain brittle on long-horizon workflows, where they must repeatedly reconstruct program state scattered across source files, configurations, tests, dependencies, and runtime behavior, leading to increasingly long interaction histories, context explosion, and semantic drift. Large repositories further complicate the identification of task-relevant components. To address these challenges, we introduce \textbf{Dev-Primitives} (\emph{Development Primitives}), a modular and executable abstraction that transforms repository components from passive software artifacts into active participants in software engineering. Each Dev-Primitive pairs a repository artifact with a resident LLM, which gives the artifact an agent-native interface grounded in its own implementation and dependenc
    
[^14]: ES-Trace：超越模型卡的代码生成模型道德采购披露审计

    ES-Trace: Auditing Ethical-Sourcing Disclosure of Code Generation Models Beyond Model Cards

    [https://arxiv.org/abs/2610.07762](https://arxiv.org/abs/2610.07762)

    提出ES-Trace框架，利用模型文档可追溯性图将代码生成模型的道德采购披露审计扩展到模型卡之外，发现仅审计模型卡会低估披露水平（1.77/5），追溯引用文档后得分提升至2.82/5。

    

    代码生成模型在软件开发中的应用日益增多，但其开发引发了涉及知识产权、隐私、公平性、劳工实践和环境影响的道德采购方面的担忧。尽管已有研究为代码生成定义了道德采购标准，但现有模型究竟披露了多少证据、以及这些证据可以在哪里找到仍不清楚。我们提出了ES-Trace，一个用于道德采购披露审计的框架，它借助模型文档可追溯性图（MDTG）将披露证据的追溯范围扩展到模型卡之外，该图表示模型、版本和文档制品之间的关系。我们将ES-Trace应用于来自10个发布者的26个模型，涵盖77份文档和20个ES-CodeGen评估方面。仅审计模型卡时平均得分为1.77/5，而解析声明中引用的文档后得分提升至2.82/5，其中大部分提升来自发布者在模型卡之外提供的文档。

    arXiv:2610.07762v1 Announce Type: new  Abstract: Code generation models have been increasingly used in software development, but their development raises ethical-sourcing concerns involving intellectual property, privacy, fairness, labour practices, and environmental impact. Although prior work has defined ethical-sourcing criteria for code generation, it remains unclear how much evidence existing models disclose and where that evidence can be found. We introduce ES-Trace, a framework for ethical-sourcing disclosure audits that traces disclosed evidence beyond model cards using the Model Documentation Traceability Graph (MDTG), which represents relationships among models, versions, and documentation artifacts. We apply ES-Trace to 26 models from 10 publishers across 77 documents and 20 ES-CodeGen aspects. Model-card-only auditing yields a mean score of 1.77/5, while resolving the declared references increases it to 2.82/5, with most of the increase arising from documents that the publi
    
[^15]: 面向编码代理的仓库规范获取与验证

    Acquiring and Verifying Repository Norms for Coding Agents

    [https://arxiv.org/abs/2610.07757](https://arxiv.org/abs/2610.07757)

    提出RepoNorm框架，独立于编码任务从仓库证据和Git历史中获取并验证显性与隐性规范，并以规范包形式提供给编码代理，显著提升了各类规范合规率。

    

    编码代理产生的代码变更可能通过功能测试，却未能满足仓库的贡献要求。遵循仓库特定规范需要识别散布在仓库各处的指导信息，并解释其条件和例外情况。检索和文档方法可以提供通用上下文，但代理仍需自行判断哪些规范适用。我们提出RepoNorm，它可以独立于编码任务获取显性和隐性的仓库规范。该方法利用仓库证据检查规范内容和适用性，必要时查阅Git历史，并向现有编码代理提供规范包。我们的评估使用了三个编码模型和来自RepoNormBench的121个任务。与无额外生成指导的基线（Raw）相比，总体规范合规率（NCR）相对提升7.42-10.77%，贡献类NCR提升31.64-45.44%，提示省略类NCR提升11.34-17.69%。三个编码模型……

    arXiv:2610.07757v1 Announce Type: new  Abstract: Changes produced by coding agents can pass functional tests while leaving repository contribution requirements unmet. Following repository-specific norms requires identifying guidance dispersed across repository sources and interpreting its conditions and exceptions. Retrieval and documentation approaches supply general context, but agents must still determine which norms apply. We introduce RepoNorm to acquire explicit and implicit repository norms independently of coding tasks. It checks norm content and applicability using repository evidence, consults Git history when needed, and delivers norm packages to existing coding agents. Our evaluation uses three coding models and 121 tasks from RepoNormBench. Against the baseline with no additional generated guidance (Raw), relative improvements are 7.42-10.77% for Overall Norm Compliance Rate (NCR), 31.64-45.44% for Contribution NCR, and 11.34-17.69% for Prompt-omitted NCR. All three coding
    
[^16]: 当旧事实回归时：重读、回退与时序记忆的局限

    When Old Facts Return: Re-Reads, Reverts, and the Limits of Temporal Memory

    [https://arxiv.org/abs/2610.07715](https://arxiv.org/abs/2610.07715)

    该论文揭示了时序记忆系统的一个关键歧义——旧信息的重读与真正的回退会产生相同的观测序列却需要相反的答案——并提出一个拒绝重新激活已淘汰值的守卫机制，该机制能有效防御重读攻击（准确率从10.8%恢复至97.7%），但需要额外的变更溯源信息才能区分合法回退。

    

    记忆系统可能会淘汰一个过时的值，但随后仅仅因为同样的旧语句再次出现就将其恢复。对旧来源的逐字重读与真正的回退可以产生相同的观测值序列，却需要截然相反的当前答案。我们在从软件修复中提取的130个由抽取器选定的原子转换上研究这种歧义性。在普通转换条件下，基于身份的时序记忆在字面过时值代理指标下达到98.5%的模型评判准确率，且观测错误为零。而在追加一段旧语句的逐字重读后，准确率降至10.8%，过时值率升至88.5%。一个拒绝重新激活先前被淘汰值的守卫机制，在此构建的重读条件下将准确率恢复至97.7%，并将过时值率降至0.8%。然而，在没有额外的变更溯源信息的情况下，该守卫无法同时识别合法的回退操作。两项辅助研究考察了将已淘汰历史暴露给……（摘要在此处截断）

    arXiv:2610.07715v1 Announce Type: cross  Abstract: A memory system can retire an obsolete value and later restore it merely because the same old statement appears again. A re-read of an old source and a genuine revert can produce the same observed sequence of values while requiring opposite current answers. We study this ambiguity on 130 extractor-selected atomic transitions derived from software fixes. In the ordinary transition condition, identity-based temporal memory reaches 98.5% model-judged accuracy with zero observed errors under a literal stale-value proxy. Appending a verbatim re-read of the old statement reduces accuracy to 10.8% and raises the stale-value rate to 88.5%. A guard that refuses to reactivate a previously retired value restores accuracy to 97.7% and reduces that rate to 0.8% in this constructed re-read condition. The guard cannot also recognize a legitimate revert without additional change provenance. Two supporting studies examine exposing retired history to th
    
[^17]: HarnessSecurity-Bench：安全机制真的能保护编码智能体框架吗？

    HarnessSecurity-Bench: Do Security Mechanisms Really Protect Coding Agent Harnesses?

    [https://arxiv.org/abs/2610.07639](https://arxiv.org/abs/2610.07639)

    该论文提出了首个针对编码智能体框架安全机制的系统性实证研究与基准 HarnessSecurity-Bench，揭示了约半数安全机制为默认关闭的可选项、闭源框架存在证据缺失，并通过覆盖五类攻击面的 23 个任务评估了六大主流框架中九种机制的真实防护效果。

    

    编码智能体框架（harness）负责协调工具使用并授权各类操作，但其安全机制及运行时的实际效果尚未得到充分表征。我们提出了 HarnessSecurity，这是首个针对开源与闭源编码智能体框架的系统性实证研究与基准测试。首先，我们构建了一个包含十种安全机制的分类体系，并通过研究人员与大语言模型（LLM）评审员的独立评分，对 400 个“框架-机制”组合单元进行了评估。我们发现，在已确认的机制实现中，约半数为可选启用（opt-in），而闭源框架则存在大量证据缺失。其次，我们推出了 HarnessSecurity-Bench，这是一个涵盖 23 个任务、覆盖五类攻击面的基准测试，同时不牺牲合法的任务需求。通过使用相互独立的确定性预言机分别衡量任务效用与攻击效果，并结合不同安全设置的对比，我们评估了六大主流框架中的九种机制：Claude Code、Codex CLI、Gemini CLI、gptme、Qwe…（原文截断）

    arXiv:2610.07639v1 Announce Type: cross  Abstract: Coding agent harnesses mediate tool use and authorize actions, yet their security mechanisms and runtime effects remain incompletely characterized. We present HarnessSecurity, the first systematic empirical study and benchmark of open- and closed-source coding agent harnesses. First, we derive a ten-mechanism taxonomy and then assess 400 harness-mechanism cells using independent ratings by researchers and large language model (LLM) judges. We find that about half of confirmed mechanism implementations are opt-in, while closed-source harnesses exhibit substantial evidence gaps. Second, we introduce HarnessSecurity-Bench, a benchmark of 23 tasks across five attack surfaces without sacrificing legitimate task requirements. Using separate deterministic oracles to measure task utility and attack effects with security setting comparisons, we evaluate nine mechanisms across six leading harnesses: Claude Code, Codex CLI, Gemini CLI, gptme, Qwe
    
[^18]: CISB-Bench：一个可审计的编译器引入安全缺陷源码—IR数据集

    CISB-Bench: An Auditable Source--IR Dataset of Compiler-Introduced Security Bugs

    [https://arxiv.org/abs/2610.07635](https://arxiv.org/abs/2610.07635)

    CISB-Bench 是一个从 GCC 和 LLVM 中挖掘出的可审计数据集，包含 429 条带有标准化 LLVM IR 分析、公开来源和二分类标注的 C 程序数据，涵盖 280 个编译器引入安全缺陷（CISB）和 149 个困难非 CISB 案例，为研究编译器引入的安全缺陷提供了可验证的基准。

    

    编译器引入的安全缺陷（CISB）是指当优化、 lowering 或插桩决策改变了所生成程序的安全相关属性时产生的缺陷。这类缺陷难以研究，因为其证据分散在问题报告、精简测试用例、历史配置和编译器产物之中；而且一份安全相关的报告并不代表每一个相关联的精简测试都能确立一个涉及安全属性的编译器错误。我们提出了 CISB-Bench，这是一个可审计的数据集，包含从 GCC 和 LLVM 中挖掘出的 429 条精确的 C 程序数据行。每行数据包含其 C 语言精简用例、-O0 至 -O3 各优化级别下的标准化 LLVM IR 分析包、公开的来源信息、最终的二分类标签，以及主要机制或边界注释。两名评审员独立地对固定语料库进行了标注，在 369 行数据上达成一致（86.0%，Cohen's kappa=0.662）；60 处分歧经过裁决解决。最终数据集包含 280 个 CISB 和 149 个具有挑战性的非 CISB 案例。（注：原摘要在此处被截断）

    arXiv:2610.07635v1 Announce Type: cross  Abstract: Compiler-introduced security bugs (CISBs) arise when an optimization, lowering, or instrumentation decision changes a security-relevant property of the generated program. They are difficult to study because their evidence is distributed across issue reports, reduced tests, historical configurations, and compiler artifacts; a security-related report also does not imply that every associated reduction establishes a security-bearing compiler failure. We present CISB-Bench, an auditable dataset of 429 exact C-program rows mined from GCC and LLVM. Each row contains its C reduction, a standardized LLVM IR analysis bundle at -O0 through -O3, public provenance, a final binary label, and a primary mechanism or boundary annotation. Two reviewers independently labeled the fixed corpus, agreeing on 369 rows (86.0%, Cohen's kappa=0.662); the 60 disagreements were adjudicated. The final dataset comprises 280 CISBs and 149 hard non-CISB cases. The pr
    
[^19]: CheckerBench：长程智能体能否合成静态分析检查器？

    CheckerBench: Can Long-Horizon Agents Synthesize Static-Analysis Checkers?

    [https://arxiv.org/abs/2610.07557](https://arxiv.org/abs/2610.07557)

    该论文提出了首个可执行基准CheckerBench（包含源自297个CVE、167个仓库的300个任务），用于评估长程智能体能否在真实代码仓库中端到端合成可用的静态分析检查器，并配套CheckerLab统一评估框架衡量诊断对比度、补丁定位、误报率和工具使用等指标。

    

    静态分析检查器合成要求智能体理解缺陷规范、检查代码仓库、实现分析器特定的逻辑，并通过反复的编译和分析反馈来完善检查器。现有的编码智能体基准主要关注补丁生成或漏洞检测等任务，很少评估智能体能否在代码仓库中从头到尾开发出一个可用的检查器。我们提出了CheckerBench，一个包含300个任务的可执行基准，这些任务源自167个代码仓库中的297个CVE，涵盖85种CWE类型和五种语言生态系统。每个任务包含存在漏洞和已修复的代码版本、固定的分析环境以及检查器脚手架。我们还进一步推出了CheckerLab，这是一个统一的评估框架，可独立重建提交的检查器，并衡量漏洞-修复诊断对比度、补丁定位能力、误报率和工具使用情况。在21种模型-框架配置和三个独立……（摘要原文截断）

    arXiv:2610.07557v1 Announce Type: cross  Abstract: Static-analysis checker synthesis requires agents to interpret a defect specification, inspect a repository, implement analyzer-specific logic, and refine the checker through repeated compilation and analysis feedback. Existing coding-agent benchmarks focus on tasks such as patch generation or vulnerability detection and rarely assess whether an agent can develop a working checker in a repository from start to finish. We introduce CheckerBench, an executable benchmark of 300 tasks derived from 297 CVEs across 167 repositories, 85 CWEs, and five language ecosystems. Each task includes vulnerable and fixed revisions, a pinned analysis environment, and a checker scaffold. We further introduce CheckerLab, a common evaluation framework that independently rebuilds submitted checkers and measures vulnerable-fixed diagnostic contrast, patch localization, false positives, and tool use. Across 21 model-harness configurations and three independen
    
[^20]: 面向规范驱动开发的EPIC框架

    The EPIC Framework for Spec-Driven Development

    [https://arxiv.org/abs/2610.07534](https://arxiv.org/abs/2610.07534)

    该研究基于ISO/IEC/IEEE 29148标准对114个开源SDD仓库进行评分分析，提出了包含10个质量维度、40项实践的EPIC框架，帮助开发人员在规范驱动开发中为编码智能体编写更明确、更完整的规范、计划和任务。

    

    一位受访从业者表示，他们的团队在指导编码智能体时会写“必须”而不是“应该”，因为智能体可能将“应该”视为可选项。措辞上的细微选择之所以重要，是因为智能体常常会用自身的假设来填补指令中的空白。规范驱动开发要求开发人员在智能体编写代码之前先撰写规范、计划和任务。SDD框架为这些制品提供了模板，但模板并不能帮助开发人员判断所写内容是否足够充分或足够清晰。我们研究了优秀的SDD规范应包含哪些内容，依据ISO/IEC/IEEE 29148标准对114个开源SDD仓库的制品进行评分，并从得分最高的制品中总结出实践方法。由此形成的EPIC框架包含10个质量维度上的40项实践，指导开发人员在面向编码智能体的规范、计划和任务中明确表达期望与决策。大多数SDD（摘要在此处截断）

    arXiv:2610.07534v1 Announce Type: new  Abstract: One practitioner we interviewed said their team writes "must" instead of "should" when instructing a coding agent, because the agent may treat "should" as optional. Small wording choices matter because agents often fill gaps in their instructions with their own assumptions. Spec-driven development (SDD) asks developers to write a specification, plan, and tasks before the agent writes code. SDD frameworks provide templates for these artifacts, but the templates do not help developers judge whether they have written enough or clearly enough. We studied what good SDD specifications contain. We scored the artifacts of 114 open-source SDD repositories against ISO/IEC/IEEE 29148 and derived practices from the highest-scoring ones. The resulting framework, EPIC, has 40 practices in 10 quality dimensions that guide developers in making expectations and decisions explicit in specifications, plans, and tasks for coding agents. The majority of SDD 
    
[^21]: CogAdapt：基于认知启发的代码大语言模型稀疏适应方法

    CogAdapt: Cognition-informed Sparse Adaptation of Code LLMs

    [https://arxiv.org/abs/2610.07446](https://arxiv.org/abs/2610.07446)

    提出了CogAdapt框架，利用人类阅读代码时产生的认知信号来指导代码大模型的稀疏选择性适应，在不牺牲性能的前提下显著降低模型微调成本。

    

    大语言模型（LLM）生成代码的能力日益增强。然而，要获得更强的代码生成性能，通常仍依赖于代价高昂的模型适应，即对预训练模型参数进行微调。已有研究表明，人类处理代码的过程与神经模型的注意力或内部计算之间存在对应关系。基于人类对齐的学习方法利用认知信号来指导训练，但通常需要对模型的大部分参数进行适应，导致训练成本基本没有降低。人类认知信号不仅可能指示模型应该从什么内容中学习，还可能指示在何处进行适应最为有效。我们研究了人类在阅读代码时的反应是否与代码模型的行为相对应，并能否在不牺牲性能的前提下指导选择性适应。我们提出了CogAdapt，一个用于代码模型任务依赖的稀疏适应的认知启发框架。CogAdapt首先学习可迁移的程序……（原文摘要在此处截断）

    arXiv:2610.07446v1 Announce Type: new  Abstract: Large language models (LLMs) have become increasingly capable of generating code. However, achieving stronger code-generation performance still often relies on costly model adaptation, i.e., fine-tuning pretrained model parameters. Prior studies have shown correspondence between human code processing and neural models' attention or internal computation. Human-aligned learning approaches use cognitive signals to guide training, but typically adapt a large portion of the model, leaving training costs largely unchanged. Human cognitive signals may indicate not only what the model must learn from, but also where adaptation is most useful. We investigate whether human responses during code reading correspond to code-model behavior and can guide selective adaptation without sacrificing performance.   We present CogAdapt, a cognition-informed framework for task-dependent sparse adaptation of code models. CogAdapt first learns transferable progr
    
[^22]: 一套面向连贯多图SysML模型的验证数据集与基准

    A Validated Dataset and Benchmark for Coherent Multi-Diagram SysML Models

    [https://arxiv.org/abs/2610.07356](https://arxiv.org/abs/2610.07356)

    该论文提出了SEMAADB——一个包含3,000个工程情境、15,000张经过一致性和有效渲染验证的SysML多视图图的大规模数据集与基准，用于评估大语言模型生成连贯多图系统建模的能力。

    

    系统工程师使用多种图来描述系统的结构和行为。工程师会共同创建这些图，以确保它们使用相同的元素并保持彼此一致。大型语言模型能够以文本或代码的形式生成图，这使得自动创建系统图成为可能。然而，它们生成连贯图集的能力尚不清楚，且现有的数据集和基准无法大规模地直接衡量这一能力。我们提出了SEMAADB（Systems Engineering Modeling Assistant with AI Dataset and Benchmark，AI系统建模助手数据集与基准），这是一个包含3,000个工程情境和15,000张图的数据集。每个情境包含五个相互关联的SysML视图：需求图、块定义图、活动图、状态机图和序列图。在这里，视图是呈现系统某一个方面的图。我们对这些图集进行了一致性和有效渲染方面的检查。此外，其中100个情境的图集还经过了人工验证。

    arXiv:2610.07356v1 Announce Type: cross  Abstract: Systems engineers use several diagrams to describe the structure and behavior of systems. Engineers create these diagrams together to make sure that they use the same elements and remain consistent with one another. Large language models can generate diagrams as text or code, which makes it possible to create system diagrams automatically. However, their ability to generate coherent sets of diagrams is not well understood, and existing datasets and benchmarks do not directly measure this ability at scale. We introduce SEMAADB (Systems Engineering Modeling Assistant with AI Dataset and Benchmark), a dataset of 3,000 engineering contexts and 15,000 diagrams. Each context contains five connected SysML views: Requirement, Block Definition, Activity, State Machine, and Sequence. Here, a view is a diagram that presents one aspect of a system. We checked the diagram sets for consistency and valid rendering. A set of 100 contexts is also human
    
[^23]: 在心流中抓住开发者：Google规模下的低延迟智能体程序修复

    Catching Developers in the Flow: Low-Latency Agentic Program Repair at Google Scale

    [https://arxiv.org/abs/2610.07289](https://arxiv.org/abs/2610.07289)

    本文提出部署于Google的AI智能体FlowAgent，通过ReAct风格的生成-验证循环与弃权过滤器，在持续集成的提交前阶段以低延迟实时自动修复测试失败，使开发者无需切换上下文即可在心流中获得高质量修复建议。

    

    程序故障的手动修复对软件开发者来说既耗时又具有干扰性，尤其是在提交前（pre-submit）阶段，此时测试失败发生在持续集成系统中。尽管自动程序修复（Automated Program Repair）借助大语言模型已取得显著进展，但现有的最先进技术主要聚焦于提交后（post-submit）的工作流程，以离线方式运行，缺乏在开发者切换上下文之前实时辅助其工作流程所需的低延迟能力。在本文中，我们介绍了FlowAgent，这是部署于Google的一个AI智能体，用于在持续集成系统内的提交前外循环工作流程中自动修复测试失败。FlowAgent集成了Google的内部开发者工具Critique和Cider，采用ReAct风格的生成与验证循环，以及严格的执行前和执行后弃权（abstention）过滤器，以确保在严格条件下提供高质量的建议。

    arXiv:2610.07289v1 Announce Type: cross  Abstract: Manual repair of program failures is time-consuming and disruptive for software developers, particularly during the pre-submit phase where test failures occur within continuous integration systems. While Automated Program Repair has seen significant advancement through Large Language Models, existing state-of-the-art techniques primarily focus on post-submit workflows, operating offline without the low-latency requirements necessary to assist developers in real-time within their flow before they switch context.   In this paper, we introduce FlowAgent, an AI agent deployed at Google to automatically repair test failures in the pre-submit outer-loop workflow inside continuous integration systems. Integrated into Google's internal developer tools, Critique and Cider,FlowAgent utilizes a ReAct-style generate-and-validate loop, as well as rigorous pre-execution and post-execution abstention filters to ensure high-quality suggestions under s
    
[^24]: SAFESHIELD：面向小语言模型部署时安全性的决策组织框架

    SAFESHIELD: A Decision-Organization Framework for Deployment-Time Safety of Small Language Models

    [https://arxiv.org/abs/2610.07276](https://arxiv.org/abs/2610.07276)

    本文提出SAFESHIELD框架，将小语言模型的部署时安全形式化为决策组织问题，通过组织准入、路由、证据和发布四种安全决策职责，并将决策记录于可审计的决策轨迹中，实现了安全决策的显式组织、协调与审计。

    

    语言模型的部署时安全通常通过运行时护栏（如输入审核、路由、检索验证和输出过滤）来实现。现有的部署框架为这些功能提供了日益强大的机制，但对于这些框架所产生的安全决策应如何被显式地组织、协调和审计，所提供的指导却十分有限。我们将部署时安全形式化为一个决策组织问题，包含两个要素：面向职责的安全决策分解，以及各决策之间的显式协调。我们将这一形式化实例化为SAFESHIELD——一个面向小语言模型的部署时安全系统，它组织了四种反复出现的决策职责（准入、路由、证据和发布），并将已确定的决策记录在可审计的决策轨迹中。我们通过机制级实验、阶段级聚合消融实验以及受控协调实验（原文在此处截断）对SAFESHIELD进行了评估。

    arXiv:2610.07276v1 Announce Type: cross  Abstract: Deployment-time safety of language models is commonly implemented through runtime guardrails such as input moderation, routing, retrieval verification, and output filtering. Existing deployment frameworks provide increasingly capable mechanisms for these functions, but offer limited guidance on how the safety decisions they produce should be explicitly organized, coordinated, and audited. We formulate deployment-time safety as a decision-organization problem with two elements: responsibility-oriented decomposition of safety decisions and explicit coordination among them. We instantiate this formulation in SAFESHIELD, a deployment-time safety system for small language models that organizes four recurring decision responsibilities (admission, routing, evidence, and release) and records committed decisions in auditable Decision Traces. We evaluate SAFESHIELD through mechanism-level experiments, aggregate stage ablations, controlled coordi
    
[^25]: 面向深度学习编译器模糊测试中已知缺陷抑制的模式引导图合成

    Pattern-Guided Graph Synthesis for Suppressing Known Defects in DL Compiler Fuzzing

    [https://arxiv.org/abs/2610.06968](https://arxiv.org/abs/2610.06968)

    提出 Reprise——一种深度学习编译器模糊测试工具，它将每个已发现的缺陷提炼为包含算子、值约束、图上下文和数据流的语义图模式，并在图合成阶段重新生成会匹配已知模式的节点，从而在编译执行之前就从源头避免重复触发已知缺陷，而非依赖事后去重。

    

    arXiv:2610.06968v1 公告类型：cross 摘要：模糊测试在发现深度学习（DL）编译器缺陷方面非常有效，但现有的模糊测试工具会反复触发它们已经发现的错误。当前的模糊测试工具致力于使生成的程序多样化，下游工具则在事后对错误报告进行去重，但两者都无法阻止模糊测试工具生成会再次触发已知缺陷的程序。我们提出了 Reprise，这是一种能够在测试生成阶段抑制已知缺陷报告的深度学习编译器模糊测试工具。Reprise 使用一个刻意保持轻量化的生成器，在具有局部算子签名的统一中间表示（UIR）中构建计算图。Reprise 并不追求程序多样化，而是将每个已发现的缺陷提炼为一种语义图模式，该模式捕获了触发该缺陷所需的算子、值约束、图上下文和数据流。在图合成过程中，它会重新生成任何会完成已知模式的节点，从而在编译和执行之前就避免触发已知缺陷。我们进行了评估……

    arXiv:2610.06968v1 Announce Type: cross  Abstract: Fuzzing is effective at finding bugs in deep learning (DL) compilers, but existing fuzzers repeatedly trigger faults they have already uncovered. Current fuzzers diversify the generated programs, and downstream tools deduplicate bug reports post hoc, but neither stops a fuzzer from generating programs that re-trigger known defects. We present Reprise, a DL compiler fuzzer that suppresses reports of known defects during test generation. Reprise uses a deliberately lightweight generator that builds graphs in a unified intermediate representation (UIR) with local operator signatures. Instead of diversifying programs, Reprise distills each discovered defect into a semantic graph pattern that captures the operators, value constraints, graph context, and data flow required to trigger it. During graph synthesis, it regenerates any node that completes a known pattern, so known defect triggers are avoided before compilation and execution. We ev
    
[^26]: ASAP：面向二进制反编译的汇编-源码对齐伪代码优化

    ASAP: Assembly-Source Aligned Pseudocode Refinement For Binary Decompilation

    [https://arxiv.org/abs/2610.06900](https://arxiv.org/abs/2610.06900)

    ASAP通过对比对齐学习汇编与源码的对应表示，并借助Q-Former压缩汇编特征、结合随机掩码与相对汇编优势损失，有效提升了LLM对二进制反编译伪代码的优化质量，尤其应对了激进优化带来的反编译错误。

    

    大语言模型（LLMs）越来越多地被应用于二进制反编译中，用于优化传统基于规则的反编译器生成的类C伪代码。尽管这种伪代码很有用，但它是一种启发式且有损的抽象，而非源代码的忠实副本，其中常常包含反编译器的错误，尤其是在经过激进优化的二进制文件中，关键的底层细节往往被掩盖。我们提出了ASAP，一个面向二进制反编译的汇编-源码对齐伪代码优化框架。ASAP利用函数级与片段级联合的对比对齐方法，从配对的源代码和二进制函数中学习源码对齐的汇编表示。随后，一个Q-Former将块级汇编特征压缩为固定数量的汇编token，与反编译器生成的伪代码一起作为条件输入反编译LLM。在优化过程中，我们采用随机伪代码掩码和相对汇编优势损失来减少（原文摘要在此处截断）

    arXiv:2610.06900v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in binary decompilation to refine the C-like pseudocode produced by traditional rule-based decompilers. While this pseudocode is useful, it is a heuristic and lossy abstraction rather than a faithful copy of the source code. It often contains decompiler errors, especially for aggressively optimized binaries where critical low-level details are obscured. We present ASAP, an assembly-source aligned pseudocode refinement framework for binary decompilation. ASAP learns source-aligned assembly representations from paired source and binary functions using joint function-level and snippet-level contrastive alignment. A Q-Former then compresses chunk-level assembly features into a fixed number of assembly tokens that condition the decompilation LLM alongside the decompiler-produced pseudocode. During refinement, we use stochastic pseudocode masking and a relative assembly-advantage loss to red
    
[^27]: 第二个模型何时有帮助？大语言模型验证中的跨模型审查

    When Does a Second Model Help? Cross-Model Review in LLM Verification

    [https://arxiv.org/abs/2610.01471](https://arxiv.org/abs/2610.01471)

    在大语言模型输出验证中，跨模型审查与同模型审查发现的错误集合部分不同，且“一次同模型新会话审查加一次跨模型审查”的组合比两次同模型审查能匹配到更多埋设错误（56.7% vs. 42.7%）。

    

    大语言模型如今能够生成代码、文档和分析内容，并且越来越多地被用于审查这类输出。本文探究的问题是：由不同的模型进行第二次审查在何时才有帮助？在作者早期预印本研究（在单一模型内改变上下文、重复次数和角色结构）的基础上，我们通过一项受控实验来检验模型独立性：实验包含30个工件（其中埋设150个错误）、10种审查条件，以及由来自两个开发者的三个审查模型执行的900次审查会话。在该实验中，(1) 顶级跨模型审查者在F1分数上与同模型在全新会话中的审查（CCR）无显著差异，但这并不等同于二者等价；(2) 两者发现的错误部分不同（Jaccard相似度为41.2%）；(3) 在两次审查调用的设定下，一次CCR加一次跨模型审查所匹配到的埋设错误多于两次CCR审查（56.7% vs. 42.7%；经Holm校正的p=.006），但并不显著多于两次顶级跨模型审查。

    arXiv:2610.01471v1 Announce Type: cross  Abstract: Large language models now generate code, documentation, and analyses, and are increasingly used to review such output. We ask when a second review by a different model helps. Building on the author's earlier preprints, which varied context, repetition, and role structure within one model, we test model independence in a controlled experiment: 30 artifacts with 150 planted errors, 10 review conditions, and 900 review sessions with three reviewer models from two developers. In this experiment, (1) a top-tier cross-model reviewer is not significantly different in F1 from same-model review in a fresh session (CCR), which does not establish equivalence; (2) the two find partly different errors (Jaccard 41.2%); and (3) at two review calls, one CCR plus one cross-model review matches more planted errors than two CCR reviews (56.7% vs. 42.7%; Holm-adjusted p=.006), but not significantly more than two reviews by the top-tier cross-model reviewe
    
[^28]: CLAD：用于神经网络验证的约束抽象域

    CLAD: Constrained Abstract Domain for Neural Network Verification

    [https://arxiv.org/abs/2609.34628](https://arxiv.org/abs/2609.34628)

    提出了约束拉格朗日抽象域（CLAD），能够在Lp范数球附加额外约束的复杂输入区域上计算神经网络行为更紧致的可靠过近似，从而克服现有抽象域因输入区域描述受限而导致的验证失败或虚假反例问题。

    

    神经网络验证（NNV）用于形式化地验证一个网络对于定义区域内所有输入都满足给定的属性。现代神经网络验证工具采用抽象域从给定输入区域出发计算网络行为的可靠过近似，因此这些抽象的紧致程度本质上决定了验证的效率。学界已开发出一系列精度不断提升的抽象域，但它们都以同样受限的方式描述有效输入区域，例如Lp范数球。然而，实际中的输入区域很少是简单的Lp球，而往往是Lp球与额外约束的组合。在 such 区域上使用现有抽象方法验证网络会产生松散的过近似，导致无法验证属性或产生虚假反例。我们提出了约束拉格朗日抽象域（CLAD），这是一种新的抽象域，能够计算神经网络[行为的可靠过近似]（摘要在此处被截断）。

    arXiv:2609.34628v2 Announce Type: replace-cross  Abstract: Neural network verification (NNV) formally verifies that a network satisfies a specified property for all inputs within a defined region. Modern NNV tools employ abstract domains to compute a sound over-approximation of the network's behavior from the given input region, thus the tightness of these abstractions essentially determines efficiency. A long line of increasingly precise domains has been developed, but they all describe the valid input region in the same restrictive way, e.g., an Lp-norm ball. A practical input region is rarely a simple Lp ball, but rather a combination Lp ball with additional constraints. Verifying a network over such a region with existing abstraction produces a loose over-approximation, which results in either failing to verify a property or spurious counterexamples. We introduce Constrained Lagrangian Abstract Domain (CLAD), a new abstract domain that computes a sound over-approximation of neural 
    
[^29]: ZonoGPT：面向大型GPT模型验证的抽象域

    ZonoGPT: Towards An Abstract Domain for Verifying Large GPT Models

    [https://arxiv.org/abs/2609.34457](https://arxiv.org/abs/2609.34457)

    ZonoGPT提出了一种空间复杂度与网络深度无关的抽象域，通过结构化zonotope、生成元约简机制以及针对Attention、LayerNorm和GELU的保精度变换，实现了对大型GPT模型的高效形式化验证。

    

    arXiv:2609.34457v2 公告类型：替换 摘要：基于Transformer的模型被广泛应用于推理、编程和多模态智能体任务。为了对期望的行为（如鲁棒性、安全性和公平性）提供形式化保证，神经网络验证技术在部署前证明所需属性并提供可审计的保证。然而，先前的工作仍局限于小型或受限的Transformer模型，并且在深层模型中保持验证精度仍然具有挑战性。在本工作中，我们提出了ZonoGPT，一种用于验证大型Transformer的抽象域，其空间复杂度与网络深度无关。ZonoGPT使用结构化zonotope和生成元约简机制来高效地保留变量间的关联性。为了保持精度，它为Attention和LayerNorm引入了保留特征关系的块级特定融合变换，并为GELU引入了保留生成元关系的仿射变换。这些机制使ZonoGPT能够……

    arXiv:2609.34457v2 Announce Type: replace  Abstract: Transformer-based models are widely used for reasoning, coding, and multimodal agentic tasks. To provide formal assurance of desirable behaviors, such as robustness, safety, and fairness, neural network verification techniques prove required properties and provide auditable guarantees before deployment. However, prior work remains limited to small or restricted Transformers, and maintaining precision across deep models remains challenging. In this work, we introduce ZonoGpt, an abstract domain for verifying large transformers that maintains a space complexity independent of network depth. ZonoGpt uses a structured zonotope and a generator reduction mechanism to efficiently preserve correlations. To maintain precision, it introduces block-specific fused transformations for Attention and LayerNorm that retain feature relations, along with an affine transform for GELU that preserves generator relations. These mechanisms enable ZonoGpt t
    
[^30]: CLEAR：基于因果上下文的智能体推理漏洞检测方法

    CLEAR: Causal Context-Based Agentic Reasoning for Vulnerability Detection

    [https://arxiv.org/abs/2608.03134](https://arxiv.org/abs/2608.03134)

    该论文提出CLEAR框架，通过构建建模入口点、前置条件、根本原因和修复意图之间因果链的漏洞因果知识图谱，并配合多智能体推理，克服现有方法仅关注表面相似性的局限，实现对源代码漏洞深层因果依赖的检测。

    

    随着现代安全漏洞深植于执行流、控制条件和程序状态之间复杂的因果依赖关系中，检测源代码漏洞变得越来越困难。尽管大语言模型和多智能体框架近年来取得了进展，但现有方法主要关注良性函数与易受攻击函数之间的表面相似性，而未能捕捉安全漏洞中固有的复杂因果依赖关系。为解决这些局限性，我们提出了基于因果上下文的智能体推理框架，这是一种融合因果知识图谱的新型多智能体漏洞检测框架。CLEAR 系统地构建了漏洞因果知识图谱，对漏洞实例中入口点、前置条件、根本原因和修复意图之间的因果链进行建模。利用这种结构化知识，四个专门化的智能体，包括 Collec……（原文摘要在此处截断）

    arXiv:2608.03134v2 Announce Type: replace-cross  Abstract: Detecting source code vulnerabilities is increasingly difficult as modern security flaws are rooted in complex causal dependencies between execution flows, control conditions, and program states. Despite recent advances in Large Language Models (LLMs) and multi-agent frameworks, existing approaches primarily address superficial similarities between benign and vulnerable functions while failing to capture the complex causal dependencies inherent in security flaws. To address these limitations, we propose Causal Context-based Agentic Reasoning (CLEAR), a novel multi-agent vulnerability detection framework integrated with a causal knowledge graph. CLEAR systematically constructs a Vulnerability Causal Knowledge Graph (VCKG) that models the causal chains between entrypoints, preconditions, root causes, and fix intents across vulnerability instances. Leveraging this structured knowledge, four specialized agents, including the Collec
    
[^31]: Palette：一个模块化、可控、高效的大语言模型按需授权安全对齐放宽框架

    Palette: A Modular, Controllable, and Efficient Framework for On-demand Authorized Safety Alignment Relaxation in LLMs

    [https://arxiv.org/abs/2605.24154](https://arxiv.org/abs/2605.24154)

    Palette 提出了一个模块化、可控且高效的框架，通过多目标搜索识别拒绝方向并借助轻量级适配将其内化到模型中，从而按需放宽授权领域的安全拒绝行为，同时保持其他领域的标准安全性。

    

    当前基础模型的安全对齐主要遵循“一刀切”范式，即对所有用户和情境应用相同的拒绝策略。这导致模型可能会拒绝那些对普通用户不安全、但对授权专业人士而言合法的请求，从而限制了模型在专业场景中的实用性。现有方法要么需要代价高昂的重新对齐，要么依赖推理时的引导技术，但后者存在控制不精确和额外延迟的问题。为此，我们提出了 Palette，一个模块化、可控且高效的框架，能够有选择地放宽授权目标领域上的拒绝行为，同时在其他方面保持标准的安全性。我们的方法通过多目标搜索识别拒绝方向，并通过轻量级适配将其内化到模型中。Palette 还进一步支持模块化组合：它可以独立学习领域特定的安全控制并进行组合。

    arXiv:2605.24154v2 Announce Type: replace  Abstract: Current safety alignment of foundation models largely follows a \emph{one-size-fits-all} paradigm, applying the same refusal policy across users and contexts. As a result, models may refuse requests that are unsafe for general users but legitimate for authorized professionals, limiting helpfulness in specialized professional settings. Existing approaches either require costly realignment or rely on inference-time steering that suffers from imprecise control and added latency. To this end, we propose \textsc{Palette}, a modular, controllable, and efficient framework that selectively relaxes refusal behavior on authorized target domains while preserving standard safety elsewhere. Our method identifies a refusal direction via multi-objective search and internalizes it into the model through lightweight adaptation. \textsc{Palette} further supports modular composition: it learns domain-specific safety controls independently and composes 
    
[^32]: PBT-Bench：基于属性测试的AI智能体基准测试

    PBT-Bench: Benchmarking AI Agents on Property-Based Testing

    [https://arxiv.org/abs/2605.15229](https://arxiv.org/abs/2605.15229)

    PBT-Bench是一个包含100个覆盖40个真实Python库的基于属性测试问题的基准，通过注入默认随机输入几乎无法触发的语义bug，专门评估AI智能体从文档中推导语义不变量并设计精确输入生成策略的能力。

    

    现有的代码基准测试衡量的是智能体能否生成任何能重现已知bug的测试，或者能否生成修复所描述问题的补丁。这两者都没有隔离出基于属性测试这一独特技能：即从文档中推导出语义不变量，然后构建一个足够精确的输入生成策略，使得随机搜索能够揭示违规行为。我们提出了PBT-Bench，这是一个包含100个精心筛选的基于属性测试问题的基准测试，涵盖40个真实的Python库。每个问题注入一个或多个语义bug（共365个，平均每个问题3.65个），其设计使得默认策略的随机输入几乎从不触发这些bug；智能体必须阅读库的文档，识别相关的不变量，并指定一个Hypothesis @given策略，将概率质量集中在触发区域。bug按三个难度级别（L1-L3）进行分层，涵盖单约束边界……

    arXiv:2605.15229v4 Announce Type: replace-cross  Abstract: Existing code benchmarks measure whether an agent can produce any test that reproduces a known bug, or whether it can produce a   patch that fixes a described issue. Neither isolates the distinct skill of property-based testing: deriving a semantic invariant   from documentation, and then constructing an input-generation strategy precise enough to make a random search reveal the violation.   We introduce PBT-Bench, a benchmark of 100 curated property-based testing problems across 40 real Python libraries. Each problem   injects one or more semantic bugs (365 in total, mean 3.65 per problem) designed so that default-strategy random inputs almost   never trigger them; the agent must read the library's documentation, identify the relevant invariant, and specify a Hypothesis   @given strategy that concentrates mass in the trigger region. Bugs are stratified across three difficulty levels (L1-L3) spanning   single-constraint boundar
    
[^33]: 通过语义驱动的单元测试生成揭示业务逻辑缺陷

    Uncovering Business Logic Bugs via Semantics-Driven Unit Test Generation

    [https://arxiv.org/abs/2604.23509](https://arxiv.org/abs/2604.23509)

    SeGa 通过从产品需求文档构建语义知识库，并推导出包含前置条件、触发动作、预期结果和语义约束的细粒度业务场景来指导大语言模型生成单元测试，从而比现有最先进技术多发现 22-25 个业务逻辑缺陷。

    

    业务逻辑缺陷违背预期的业务语义，在企业软件中尤为普遍。然而，现有的单元测试生成技术大多以代码为中心，使得此类缺陷难以被暴露。我们提出了 SeGa，一种用于发现业务逻辑缺陷的语义驱动单元测试生成技术。SeGa 从产品需求文档中构建语义知识库，将其表示为一组功能条目，这些条目将相关需求按共同的业务意图进行分组。给定一个目标方法，SeGa 检索相关的功能条目，并推导出具有明确前置条件、触发动作、预期结果和语义约束的细粒度业务场景，以指导基于大语言模型（LLM）的测试生成。我们在包含 60 个真实业务逻辑缺陷的四个工业级 Go 项目上对 SeGa 进行了评估。结果表明，SeGa 比四种最先进的基于 LLM 的技术多检测出 22-25 个缺陷，并提高了精确率。

    arXiv:2604.23509v3 Announce Type: replace  Abstract: Business logic bugs violate intended business semantics and are particularly prevalent in enterprise software. Yet most existing unit test generation techniques are code-centric, making such bugs difficult to expose. We present SeGa, a semantics-driven unit test generation technique for uncovering business logic bugs. SeGa constructs a semantic knowledge base from product requirement documents, represented as a set of functionality entries that group related requirements under a common business intent. Given a focal method, SeGa retrieves the relevant functionality entries and derives fine-grained business scenarios with explicit preconditions, triggering actions, expected outcomes, and semantic constraints to guide LLM-based test generation. We evaluate SeGa on four industrial Go projects containing 60 real-world business logic bugs. SeGa detects 22-25 more bugs than four state-of-the-art LLM-based techniques and improves precision 
    
[^34]: 通过神经符号增强静态分析发现C/C++程序中的内存泄漏

    Finding Memory Leaks in C/C++ Programs via Neuro-Symbolic Augmented Static Analysis

    [https://arxiv.org/abs/2603.27224](https://arxiv.org/abs/2603.27224)

    MemHint结合大语言模型的代码语义理解与基于Z3的符号推理验证，识别项目自定义内存管理函数并过滤不可行的函数摘要，从而增强静态分析器检测C/C++程序内存泄漏的能力。

    

    内存泄漏在真实世界的C/C++软件中仍然普遍存在。诸如CodeQL之类的静态分析器提供了可扩展的程序分析能力，但经常遗漏此类缺陷，因为它们无法识别项目特定的自定义内存管理函数，且缺乏路径敏感的控制流建模。我们提出了MemHint，这是一个神经符号流水线，通过将大语言模型对代码的语义理解与基于Z3的符号推理相结合，来解决上述两个局限。MemHint解析目标代码库，并利用大语言模型将每个函数分类为内存分配器、内存释放器或两者皆非，生成记录哪个参数或返回值携带内存所有权的函数摘要，从而将分析器的内置知识扩展到malloc和free等标准原语之外。基于Z3的验证步骤将每个摘要与函数的控制流图进行核对，丢弃那些所声称的内存操作在任何可行路径上均不可达的摘要。

    arXiv:2603.27224v5 Announce Type: replace  Abstract: Memory leaks remain prevalent in real-world C/C++ software. Static analyzers such as CodeQL provide scalable program analysis but frequently miss such bugs because they cannot recognize project-specific custom memory-management functions and lack path-sensitive control-flow modeling. We present MemHint, a neuro-symbolic pipeline that addresses both limitations by combining LLMs' semantic understanding of code with Z3-based symbolic reasoning. MemHint parses the target codebase and applies an LLM to classify each function as a memory allocator, deallocator, or neither, producing function summaries that record which argument or return value carries memory ownership, extending the analyzer's built-in knowledge beyond standard primitives such as malloc and free. A Z3-based validation step checks each summary against the function's control-flow graph, discarding those whose claimed memory operation is unreachable on any feasible path. The
    
[^35]: ReLoop：面向可靠的大语言模型优化的结构化建模与行为验证

    ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization

    [https://arxiv.org/abs/2602.15983](https://arxiv.org/abs/2602.15983)

    ReLoop通过结合结构化生成和行为验证，有效缩小了大语言模型在优化代码生成中的可行性与正确性差距。

    

    大语言模型（LLMs）可以将自然语言转化为优化代码，但静默失败构成关键风险：能够执行并返回求解器可行解的代码可能编码了语义上错误的公式——这种可行性与正确性之间的差距在组合问题上高达90个百分点。我们引入了ReLoop，通过两种互补机制来解决这一差距。结构化生成将代码生产分解为四阶段推理链（理解、形式化、综合、验证），从源头防止公式错误。行为验证通过测试公式是否对基于求解器的参数扰动做出正确响应来检测生成过程中存活的错误——这是一种绕过LLM自我审查且无需真实标签的外部语义信号。这两种机制在错误结构上互补：结构化生成在组合问题上带来最大改进。

    arXiv:2602.15983v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) can translate natural language into optimization code, but silent failures pose a critical risk: code that executes and returns solver-feasible solutions may encode semantically incorrect formulations---a feasibility--correctness gap reaching 90 percentage points on compositional problems. We introduce ReLoop, which addresses this gap through two complementary mechanisms. Structured generation decomposes code production into a four-stage reasoning chain (understand, formalize, synthesize, verify), preventing formulation errors at their source. Behavioral verification detects errors that survive generation by testing whether the formulation responds correctly to solver-based parameter perturbation---an external semantic signal that bypasses LLM self-review and requires no ground truth. The two mechanisms are complementary by error structure: structured generation drives the largest gains on compositi
    
[^36]: 软件工程中的理论构建：操作化

    Theory building in software engineering: Operationalization

    [https://arxiv.org/abs/2412.02384](https://arxiv.org/abs/2412.02384)

    本文系统化了软件工程理论构建中的操作化阶段，将概念化阶段得到的概念和命题转化为明确的构念和可经验检验的假设，并通过三个维度（实际程序步骤、形式化数学规范和实证案例评估）进行阐述。

    

    这项工作是一个研究项目的一部分，该项目的最终目标是系统化软件工程中构建理论的过程。所提出的方法论包括四个阶段：概念化、操作化、测试和应用。在之前的工作中，我们描述了概念化过程。本文提出了一套用于系统化理论构建中操作化阶段的程序。具体而言，它将先前概念化阶段获得的概念和命题转化为构念和可经验检验的假设。操作化阶段在三个不同的维度上展开：实用的程序步骤、其严格的形式化数学规范，以及通过一个示例性实证案例研究进行的评估。基于批判实在论，我们将定性推导出的概念和命题操作化为明确的构念和逻辑假设。

    arXiv:2412.02384v2 Announce Type: replace  Abstract: This work is part of a research project whose ultimate goal is to systematize a procedure for constructing theories in software engineering. The proposed methodology involves four phases: conceptualization, operationalization, testing, and application. In previous work, we described the conceptualization process. This paper presents a set of procedures for systematizing the operationalization phase in theory building. Specifically, it translates the concepts and propositions obtained from the previous conceptualization into constructs and empirically testable hypotheses. The operationalization phase is structured across three distinct dimensions: the practical procedural steps, their strict formal mathematical specification, and their evaluation through an illustrative empirical case study. Grounded in critical realism, we operationalize the qualitatively derived concepts and propositions into explicit constructs and logical hypothes
    

