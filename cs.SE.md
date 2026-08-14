# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [QuoteBench: How Matched Scores Can Hide Command-Path Failures](https://arxiv.org/abs/2608.13547) | QuoteBench通过精确的最终状态验证揭示了LLM编码代理中匹配分数无法区分命令生成错误与执行传输故障，并证明了公开解析器边界能显著提高恢复率。 |
| [^2] | [Vero: Can AI Agents Build Formally Verified Software Repositories?](https://arxiv.org/abs/2608.13522) | Vero是首个在仓库级别评估AI代理联合实现和证明合成能力的基准，填补了现有基准在真实多模块代码库上的空白。 |
| [^3] | [CAPRI: Contract-Aware Proof Repair for Isabelle](https://arxiv.org/abs/2608.13459) | CAPRI通过引入契约感知的修复工作流，结合Isabelle检查和独立编辑契约验证，确保LLM生成的证明修复不越权修改受保护文本，在保证安全性的同时实现了高效修复。 |
| [^4] | [LLM-Assisted Dynamic Threat Analysis for Attacker-Reachable Software Weaknesses in Autonomous Vehicles](https://arxiv.org/abs/2608.13450) | 本文提出利用大语言模型自动化自动驾驶软件栈中攻击者可达弱点的动态可利用性确认，通过静态分析引导和编译器在环修复生成可执行测试工件，显著简化了漏洞验证过程。 |
| [^5] | [Does Fixing Break Security? An Empirical Study of Security Degradation in Iterative LLM-Driven Infrastructure-as-Code Repair](https://arxiv.org/abs/2608.13404) | 本研究首次揭示了迭代LLM修复基础设施即代码时，安全回归现象频繁发生，即修复其他问题可能导致先前通过的安全检查失败。 |
| [^6] | [Integration-First Structural Coverage for Embedded Software:Trace-Based Evidence, Hybrid Runtime Analysis, and Cross-Variant Consolidation](https://arxiv.org/abs/2608.13322) | 本文提出了一种面向嵌入式软件的一体化优先结构覆盖率方法，通过混合运行时分析和跨变体整合，解决集成测试覆盖率不准确的问题，确保完整性作为已覆盖或已论证来建立。 |
| [^7] | [Refine After Generation: Toward Correct and Concise Patches in LLM-based Program Repair](https://arxiv.org/abs/2608.13292) | 本文发现LLM-based程序修复生成的补丁存在严重的冗长性问题，并提出了生成后补丁精炼方法，旨在在保持正确性的同时产生更简洁的补丁。 |
| [^8] | [Can Formal Specifications Be Synthesized from Tests Alone?](https://arxiv.org/abs/2608.13240) | 本文提出一种仅基于测试代码和动态执行轨迹、无需访问源代码内部的白盒信息来合成形式化规格的LLM方法，并在SpecGenBench上初步验证了其有效性，同时指出了检查器兼容性和反馈诊断是主要挑战。 |
| [^9] | [Smart Contract Invariants Protect Against Cybercriminals](https://arxiv.org/abs/2608.13191) | 本文提出使用经典的程序不变量概念来保护智能合约，并通过包含28个真实以太坊漏洞的基准和重放框架验证，证明其能有效阻止网络攻击。 |
| [^10] | [How Powerful are LLMs in Generating Formal Program Specifications?](https://arxiv.org/abs/2608.13077) | 本文提出了Coins框架，通过实例化规范并生成具体证明义务来评估LLM生成形式化程序规范的质量，避免了传统方法中证明难度与规范质量混淆的问题。 |
| [^11] | [Static analysis-guided agentic AI translation enables Rust as a full stack bioinformatics language](https://arxiv.org/abs/2608.13029) | 本文提出利用静态分析引导的智能体AI将遗留生物信息学代码自动翻译为Rust语言，大幅缩减代码体积、提升构建和运行性能，并消除Unix依赖，实现了全栈式单细胞分析。 |
| [^12] | [Requirements-Augmented Generation for Trustworthy Acceptance Testing of LLM-Based Software](https://arxiv.org/abs/2608.12970) | 该论文提出了一种基于需求增强生成（REAG）的自动化验收测试框架，通过检索软件需求、领域知识和用户角色来校准LLM软件的裁决可靠性，解决了传统验收测试无法适应LBS上下文相关随机行为的问题。 |
| [^13] | [Dissecting Software Graphs: Structural Insights for Driver-Guided Fuzzing](https://arxiv.org/abs/2608.12859) | 本文提出了一种基于静态调用图的结构抽象方法，通过投影驱动特定覆盖率来分析多驱动模糊测试中的软件结构，并基于27个真实项目揭示了执行模式对模糊测试有效性的影响。 |
| [^14] | [Memorization Diagnostics for Code LLMs Should be Scale-Aware](https://arxiv.org/abs/2608.12771) | 本文指出传统记忆探测方法在大型代码模型上失效，并提出通过可逆数学变换将表示负载与记忆分离，以实现规模感知的记忆诊断。 |
| [^15] | [Does It Render Everywhere? A Study of Cross-Environment Compatibility in MLLM-Generated Webpages](https://arxiv.org/abs/2608.12518) | 本研究首次系统评估了AI生成网页的跨环境兼容性，发现68%的网页存在渲染问题，揭示了当前多模态大语言模型在网页生成中的可靠性缺陷。 |
| [^16] | [Specification-first convergence with an AI coding agent: a case study of dismantling a core architectural invariant across 189 files in a 717k-line codebase with no test oracle and no human code review](https://arxiv.org/abs/2608.12440) | 本研究展示了一种规范优先协议，使AI编码代理能够在无测试预言机和人工审查的情况下，成功完成对大型代码库中核心架构不变量的拆除，这一任务传统上被认为需要重写。 |
| [^17] | [SynWeaver: Website-Prior Task and Trajectory Co-Synthesis for Web Agents](https://arxiv.org/abs/2608.12429) | SynWeaver通过构建网站地图来获取网站先验知识，并协同合成任务与轨迹，从而提升网络代理对未见网站的泛化能力。 |
| [^18] | [Humans are Missing from AI Coding Agent Research](https://arxiv.org/abs/2608.12355) | 本文主张AI编码代理研究应从自主任务解决转向以人为中心的设计，强调通过任务对齐、可验证性、可引导性和适应性四个维度来增强人类与代理的协作。 |
| [^19] | [FlowLog: Re-thinking Datalog for Fast and Extensible Static Analysis](https://arxiv.org/abs/2607.23971) | FlowLog通过将Datalog程序编译为差分数据流，实现了静态分析的高效与可扩展性，在24个真实基准测试中优于现有引擎，并支持毫秒级增量更新和细粒度性能调优。 |
| [^20] | [Semantic Drift in Bug Resolution: How Behavioral Signals Propagate from Reports to Tests and Patches](https://arxiv.org/abs/2607.18550) | 本文提出Desc2Fix框架，通过结构化行为锚点和多种相似度指标测量缺陷报告、测试与补丁间的语义对齐，发现LLM能可靠提取信号但对齐高度依赖表示方式，完整差异是最稳定的判断基础。 |
| [^21] | [Alipay-PIBench: A Realistic Payment Integration Benchmark for Coding Agents](https://arxiv.org/abs/2607.14573) | 本文提出了Alipay-PIBench基准测试，通过现实支付宝支付集成任务评估编码代理，并展示了特定技能可显著提升代理性能（平均RPR提高10.31个百分点）。 |
| [^22] | [LLM-Based Test Oracles: Source-of-Authority Taxonomy -- A Systematic Literature Review](https://arxiv.org/abs/2607.05031) | 本综述首次按权威来源对LLM测试预言机进行分类，发现超过半数预言机在无规范情况下仅依赖模型训练知识作出判决，揭示了该领域信任基础的隐患。 |
| [^23] | [What Breaks When LLMs Code? Characterizing Operational Safety Failures of Agentic Code Assistants](https://arxiv.org/abs/2605.30777) | 本文通过分析大量论文和GitHub问题，首次系统性地分类了编码智能体在日常开发中发生的操作安全失败类型及其影响，并提出了一个多维度安全分类框架。 |
| [^24] | [Formal Verification of Imperative First-Class Functions in Move](https://arxiv.org/abs/2605.10007) | 本文为Move语言中的一等命令式函数提出了基于行为谓词和状态标签的正式验证方法，并实现了在Move Prover中的编码与验证。 |
| [^25] | [Foundation Models as Oracles for Refactoring Correctness Detection](https://arxiv.org/abs/2605.02096) | 本研究首次证明基础模型可通过零样本提示有效检测Java IDE中的重构错误，在226个真实案例上达到高准确率，为自动化重构验证提供了新路径。 |
| [^26] | [IACDM: Interactive Adversarial Convergence Development Methodology -- A Structured Framework for AI-Assisted Software Development](https://arxiv.org/abs/2604.16399) | 本文提出IACDM框架，通过外部状态机驱动的8阶段对抗验证过程，解决AI辅助软件开发中因验证差距导致的效率降低和数据泄露问题。 |
| [^27] | [Error Understanding in Program Code: A Systematic Study of LLM-DL Combinations for Multi-label Classification](https://arxiv.org/abs/2603.25005) | 本研究系统评估了32种LLM-DL组合在代码多标签错误分类中的性能，发现CodeT5+GRU组合表现最佳，并揭示了编码器-解码器组件间的相互作用对性能的关键影响。 |
| [^28] | [CangjieBench: Benchmarking LLMs on a Low-Resource General-Purpose Programming Language](https://arxiv.org/abs/2603.14501) | 本文提出了CangjieBench，一个针对低资源通用语言仓颉的无污染基准测试，系统评估发现语法约束生成在准确性和效率上表现最优。 |
| [^29] | [The Impact of Generative AI on Collaborative Open-Source Software Development: Evidence from GitHub Copilot](https://arxiv.org/abs/2410.02091) | 本研究通过GitHub Copilot的数据发现，生成式AI在开源软件开发中提高了代码贡献和开发者参与度，但同时也增加了协调时间和代码讨论，揭示出AI在扩展贡献范围与增加协作成本之间的权衡。 |

# 详细

[^1]: QuoteBench：匹配分数如何掩盖命令路径故障

    QuoteBench: How Matched Scores Can Hide Command-Path Failures

    [https://arxiv.org/abs/2608.13547](https://arxiv.org/abs/2608.13547)

    QuoteBench通过精确的最终状态验证揭示了LLM编码代理中匹配分数无法区分命令生成错误与执行传输故障，并证明了公开解析器边界能显著提高恢复率。

    

    arXiv:2608.13547v1 公告类型：新 摘要：LLM编码代理通过可能序列化、包装和重新解析模型输出的接口发出Bash命令。仅凭匹配执行分数无法区分命令生成错误与生成后引入的失败。QuoteBench通过14个事件衍生家族的56个一次性任务，以精确的最终状态验证来衡量这一边界，将生成契约与执行传输交叉，围绕一个故意未转义的添加解析器。在插值点进行转义可重现每个重放回复的原始路径结果，因此在公开边界下的任何恢复都必须来自模型改变其生成。在八个同窗口配置中，通过添加解析器重放相同回复会使成功率降低55.4至73.2个百分点；公开恢复对六个配置提高30.4至60.7个百分点，对另外两个配置为零或略有负增长。原始生成在边界处接近饱和；但...

    arXiv:2608.13547v1 Announce Type: new  Abstract: LLM coding agents issue Bash commands through interfaces that may serialize, wrap, and reparse model output. Matched execution scores alone cannot distinguish command-generation errors from failures introduced after generation. QuoteBench measures this boundary with exact final-state validation on 56 one-shot tasks from 14 incident-derived families, crossing the generation contract with the execution transport around one deliberately unescaped added parser. Escaping at the interpolation point reproduces each replayed reply's raw-path outcome, so any recovery under a disclosed boundary must come from the model changing its generation. Across eight same-window configurations, replaying the same reply through the added parser lowers success by 55.4 to 73.2 percentage points; disclosure recovers 30.4 to 60.7 points for six configurations, and zero or slightly negative for the other two. Raw generation is nearly saturated at the frontier; bou
    
[^2]: Vero：AI代理能否构建经过正式验证的软件仓库？

    Vero: Can AI Agents Build Formally Verified Software Repositories?

    [https://arxiv.org/abs/2608.13522](https://arxiv.org/abs/2608.13522)

    Vero是首个在仓库级别评估AI代理联合实现和证明合成能力的基准，填补了现有基准在真实多模块代码库上的空白。

    

    arXiv:2608.13522v1 公告类型：交叉 摘要：AI代理越来越多地用于编程，但对其生成代码的正确性不提供任何保证。经过验证的代码生成，即代理同时生成实现及其规范的机器检查证明，为可信赖的AI生成软件提供了一条更强的路径。现有这一方向的基准要么专注于单个函数，要么仅评估在提供实现情况下的证明生成。代理能否在真实的多模块代码库中做出连贯的实现和证明选择仍是一个未解决的问题。为填补这一空白，我们引入了Vero，这是首个在仓库级别评估联合实现和证明合成的基准。Vero包含43个多模块实例，源自真实世界的仓库，涵盖Python、Dafny、Verus和Coq，并覆盖从加密协议到分布式系统的多样领域。每个实例由一个多模块任务组成。

    arXiv:2608.13522v1 Announce Type: cross  Abstract: AI agents are increasingly used for programming, but do not provide any guarantee on the correctness of generated code. Verified code generation, in which an agent produces both an implementation and a machine-checked proof of its specification, offers a stronger path toward trustworthy AI-generated software. Existing benchmarks in this direction either focus on individual functions or only evaluate proof generation with provided implementations. It is still an open question whether agents can make coherent implementation and proof choices across real multi-module codebases. To bridge this gap, we introduce Vero, the first benchmark to evaluate joint implementation and proof synthesis at the repository level. Vero contains 43 multi-module instances sourced from real-world repositories spanning Python, Dafny, Verus, and Coq, and covering diverse domains from cryptographic protocols to distributed systems. Each instance consists of a mul
    
[^3]: CAPRI：面向Isabelle的契约感知证明修复

    CAPRI: Contract-Aware Proof Repair for Isabelle

    [https://arxiv.org/abs/2608.13459](https://arxiv.org/abs/2608.13459)

    CAPRI通过引入契约感知的修复工作流，结合Isabelle检查和独立编辑契约验证，确保LLM生成的证明修复不越权修改受保护文本，在保证安全性的同时实现了高效修复。

    

    arXiv:2608.13459v1 公告类型：交叉 摘要：我们探讨了使用大型语言模型（LLMs）来辅助发现Isabelle证明的方法。Isabelle构建过程确认了提交的理论被接受，但并不保证LLM仅修改了开发者授权的部分。我们提出了CAPRI，一种契约感知修复工作流，其中Isabelle检查证明，独立的检查器强制执行机器可读的编辑契约。提示、提议、候选仓库、诊断、判定和哈希值均被保留以供审计。我们在四个开发中的十二个失败证明上评估了五种工作流，每个任务和条件重复三次，共进行180次运行和138次有效修复。在Isabelle接受的144个终端候选中，有六个修改了受保护文本；所有这些都出现在可以编辑完整理论的迭代工作流中。仅证明体接口产生了29/36的有效修复且无契约违规，而相应的完整理论工作流则为31/36。单次修复...

    arXiv:2608.13459v1 Announce Type: cross  Abstract: We address the use of large language models (LLMs) to help discover Isabelle proofs. An Isabelle build establishes that the submitted theory is accepted, but not that an LLM changed only what the developer authorised. We present CAPRI, a contract-aware repair workflow in which Isabelle checks the proof and an independent checker enforces a machine-readable edit contract. Prompts, proposals, candidate repositories, diagnostics, verdicts, and hashes are retained for audit. We evaluate five workflows on twelve failed proofs from four developments, with three replicates per task and condition, giving 180 runs and 138 valid repairs. Of 144 terminal candidates accepted by Isabelle, six had modified protected text; all arose in iterative workflows that could edit a complete theory. A proof-body-only interface produced 29/36 valid repairs and no contract violations, compared with 31/36 for the corresponding full-theory workflow. One-shot repai
    
[^4]: 基于大语言模型的自动驾驶车辆攻击者可达软件弱点动态分析

    LLM-Assisted Dynamic Threat Analysis for Attacker-Reachable Software Weaknesses in Autonomous Vehicles

    [https://arxiv.org/abs/2608.13450](https://arxiv.org/abs/2608.13450)

    本文提出利用大语言模型自动化自动驾驶软件栈中攻击者可达弱点的动态可利用性确认，通过静态分析引导和编译器在环修复生成可执行测试工件，显著简化了漏洞验证过程。

    

    摘要：arXiv:2608.13450v1 公告类型：交叉 摘要：自动驾驶车辆依赖于庞大的安全关键软件栈，其中从对抗性输入可达的弱点可能影响转向、制动或其他控制决策。静态分析可以识别候选位置，但动态确认可利用性需要可执行的测试工件，而这些工件难以手动构建。我们研究了大语言模型（LLMs）是否能自动化这一过程，针对开源自动驾驶栈Autoware。我们在185个软件包上进行了编译器精确的静态分析，识别出1,375个决策规则、2,274个验证检查和482个输入到安全输出的数据流，从中我们推导出弱点分类并采样了740个可达位置。两个本地开源权重LLM、一个无静态上下文消融和一个简单模板基线生成了3,700个工件集，这些工件集在消毒器下针对真实构建进行编译，通过编译器在环反馈修复，并进行模糊测试。

    arXiv:2608.13450v1 Announce Type: cross  Abstract: Autonomous vehicles depend on large safety-critical software stacks, where weaknesses reachable from adversarial inputs may affect steering, braking, or other control decisions. Static analysis can identify candidate sites, but dynamically confirming exploitability requires executable test artifacts that are difficult to construct manually. We investigate whether large language models (LLMs) can automate this process for Autoware, an open-source autonomous-driving stack. We perform compiler-precise static analysis across 185 packages, identifying 1,375 decision rules, 2,274 validation checks, and 482 input-to-safety-output flows, from which we derive a weakness taxonomy and sample 740 reachable sites. Two local open-weight LLMs, a no-static-context ablation, and a naive-template baseline generate 3,700 artifact sets, which are compiled against the real build under sanitizers, repaired through compiler-in-the-loop feedback, and fuzzed w
    
[^5]: 修复是否破坏安全性？迭代LLM驱动的基础设施即代码修复中安全退化的实证研究

    Does Fixing Break Security? An Empirical Study of Security Degradation in Iterative LLM-Driven Infrastructure-as-Code Repair

    [https://arxiv.org/abs/2608.13404](https://arxiv.org/abs/2608.13404)

    本研究首次揭示了迭代LLM修复基础设施即代码时，安全回归现象频繁发生，即修复其他问题可能导致先前通过的安全检查失败。

    

    arXiv:2608.13404v1 公告类型：新 摘要：背景：迭代反馈循环是改进LLM生成的基础设施即代码（IaC）的主要范式：诸如Checkov和terraform validate等验证器会将错误信号反馈给后续的修复尝试。先前的工作报告了累积最佳指标，这些指标在构造上非递减，因此从未对IaC的逐迭代安全轨迹进行过检查。目的：我们研究安全回归（先前通过的CIS基准检查在修复迭代后失败），以确定迭代LLM修复在修复其他问题时是否以及多久会降低安全性。方法：我们分析了IaC-Eval基准中的5,968个场景时间线，每个场景通过一个配置运行最多5次修复迭代。15种配置（六种模型特定的RAG，九种模型聚合的非RAG，每种三个温度）产生了4,440次迭代转换，两侧均有Checkov数据。我们跟踪30个单独的CIS基准检查。

    arXiv:2608.13404v1 Announce Type: new  Abstract: Background: Iterative feedback loops are the dominant paradigm for improving LLM-generated Infrastructure-as-Code (IaC): validators such as Checkov and terraform validate feed error signals back for successive repair attempts. Prior work reports cumulative-best metrics, which are non-decreasing by construction, so the raw per-iteration security trajectory has never been examined for IaC. Aims: We study security regression (a previously-passing CIS Benchmark check that fails after a repair iteration) to determine whether and how often iterative LLM repair degrades security while fixing other issues. Method: We analyze 5,968 scenario timelines from the IaC-Eval benchmark, each one scenario run through one configuration for up to 5 repair iterations. The 15 configurations (six model-specific RAG, nine model-aggregated non-RAG, three temperatures each) yield 4,440 iteration transitions with Checkov data on both sides. We track 30 individual 
    
[^6]: 面向嵌入式软件的一体化优先结构覆盖率：基于追踪的证据、混合运行时分析与跨变体整合

    Integration-First Structural Coverage for Embedded Software:Trace-Based Evidence, Hybrid Runtime Analysis, and Cross-Variant Consolidation

    [https://arxiv.org/abs/2608.13322](https://arxiv.org/abs/2608.13322)

    本文提出了一种面向嵌入式软件的一体化优先结构覆盖率方法，通过混合运行时分析和跨变体整合，解决集成测试覆盖率不准确的问题，确保完整性作为已覆盖或已论证来建立。

    

    arXiv:2608.13322v1 公告类型：新 摘要：结构覆盖率被广泛用作测试完整性的证据，但在嵌入式项目中，它主要在单元级别收集，仅仅因为在那里进行插桩和可观测性成本低廉。这导致了不匹配。最具代表性的完整性信号应来自在待测设备上执行的集成和系统测试，但传统插桩会扰动时序、内存占用和并发行为，而纯粹基于追踪重建的覆盖率在编译器激进优化时对决策和条件的可靠性会降低。我们从两端解决这一不匹配。在流程方面，我们描述了一种一体化优先的覆盖率策略，将集成和系统测试作为基线测量，并通过显式闭合循环驱动残余缺口，从而将完整性确立为“已覆盖或已论证”，而非仅“已覆盖”。在技术方面，我们提出了一种混合运行时分析，结合插桩和追踪方法，以在优化环境中保持覆盖率评估的准确性，并引入跨变体整合，以在多个固件变体中一致地汇总覆盖率结果。

    arXiv:2608.13322v1 Announce Type: new  Abstract: Structural coverage is widely used as evidence that testing is complete, yet in embedded projects it is predominantly collected at unit level, simply because that is where instrumentation and observability are inexpensive. This produces a mismatch. The most representative completeness signal would come from integration and system tests executed on the device under test, but classical instrumentation perturbs timing, memory footprint and concurrency behaviour, while purely trace-reconstructed coverage loses reliability for decisions and conditions as soon as the compiler optimizes aggressively. We address this mismatch from both ends. On the process side we describe an integrationfirst coverage strategy that treats integration and system tests as the baseline measurement and drives the residual gaps through an explicit closure loop, so that completeness is established as covered or justified rather than as covered alone. On the technical 
    
[^7]: 生成后精炼：迈向基于LLM的程序修复中的正确且简洁补丁

    Refine After Generation: Toward Correct and Concise Patches in LLM-based Program Repair

    [https://arxiv.org/abs/2608.13292](https://arxiv.org/abs/2608.13292)

    本文发现LLM-based程序修复生成的补丁存在严重的冗长性问题，并提出了生成后补丁精炼方法，旨在在保持正确性的同时产生更简洁的补丁。

    

    arXiv:2608.13292v1 公告类型：新 摘要：大型语言模型（LLMs）已将自动程序修复（APR）推进到代理系统常规解决真实世界、仓库级问题的程度。然而，生成的补丁除了是否通过测试外，很少受到严格审查。在本文中，我们指出补丁冗长性是LLM-based APR中一个主要但被忽视的问题。通过对SWE-bench Verified上的28种最先进方法进行特征化分析，我们发现即使成功的补丁也始终比开发者编写的补丁更大且更复杂，中位数方法产生的总更改量多121.78%，净更改量多80.91%，圈复杂度高43.99%。我们进一步表明，这种冗长性源于以能力为导向的设计选择，如迭代精炼和广泛上下文，并且很难通过表面控制（如输出格式或简洁性提示）来减少。基于这些发现，我们提出了生成后补丁精炼框架。

    arXiv:2608.13292v1 Announce Type: new  Abstract: Large language models (LLMs) have advanced automatic program repair (APR) to the point where agentic systems routinely resolve real-world, repository-level issues. Yet the generated patch has received little scrutiny beyond whether it passes tests.   In this paper, we identify patch verbosity as a major yet overlooked concern in LLM-based APR. Characterizing 28 state-of-the-art approaches on SWE-bench Verified, we find that even successful patches are consistently larger and more complex than developer patches, with the median approach producing 121.78% more total changes, 80.91% more net changes, and 43.99% higher cyclomatic complexity. We further show that this verbosity is rooted in capability-oriented design choices such as iterative refinement and broad context, and can hardly be reduced by surface-level controls such as output format or minimality prompts. Motivated by these findings, we formulate post-generation patch refinement a
    
[^8]: 仅凭测试能否综合出形式化规格？

    Can Formal Specifications Be Synthesized from Tests Alone?

    [https://arxiv.org/abs/2608.13240](https://arxiv.org/abs/2608.13240)

    本文提出一种仅基于测试代码和动态执行轨迹、无需访问源代码内部的白盒信息来合成形式化规格的LLM方法，并在SpecGenBench上初步验证了其有效性，同时指出了检查器兼容性和反馈诊断是主要挑战。

    

    arXiv:2608.13240v1 公告类型：新 摘要：形式化规格提供了强有力的保证，但手工编写成本高昂。近期基于LLM的方法通过从源代码推断规格来实现自动化，但其对白盒访问的依赖因知识产权风险和部署成本而构成工业采用的障碍。我们的方法使用LLM仅从测试代码和动态执行轨迹中推断候选规格：LLM仅观察程序接口、选定输入及相应的输出或状态变化，而实现内部细节保持隐藏。候选规格通过有界模型检查在本地验证，并利用反馈指导迭代细化。在SpecGenBench基准上的初步结果表明，测试可以引导LLM生成有意义的Java建模语言规格，同时也凸显了检查器兼容性和诊断反馈作为可靠细化的关键挑战。

    arXiv:2608.13240v1 Announce Type: new  Abstract: Formal specifications offer strong guarantees, but remain costly to write manually. Recent LLM-based approaches automate this by inferring specifications from source code, yet their reliance on white-box access poses barriers to industrial adoption due to intellectual property risks and deployment costs. Our approach uses LLMs to infer candidate specifications solely from test code and dynamic execution traces: the LLM observes only the program interface, selected inputs, and corresponding outputs or state changes, while the implementation internals remain hidden. Candidate specifications are validated locally using bounded model checking, with feedback guiding iterative refinement. Initial results on the SpecGenBench benchmark suggest that tests can guide LLMs towards meaningful Java Modeling Language specifications, while also highlighting checker compatibility and diagnostic feedback as key challenges for reliable refinement.
    
[^9]: 智能合约不变量保护免受网络罪犯侵害

    Smart Contract Invariants Protect Against Cybercriminals

    [https://arxiv.org/abs/2608.13191](https://arxiv.org/abs/2608.13191)

    本文提出使用经典的程序不变量概念来保护智能合约，并通过包含28个真实以太坊漏洞的基准和重放框架验证，证明其能有效阻止网络攻击。

    

    arXiv:2608.13191v1 公告类型：交叉 摘要：区块链是计算领域中最具对抗性的环境之一。数十亿美元被利用漏洞的网络罪犯窃取。这是一个开放性问题，目前没有任何概念或技术被证明能真正带来改变。在本文中，我们声称经典的程序不变量概念可能是解决该问题的最有力方案。我们设计了一个原创的实验协议，用于1) 研究不变量如何能够防范过去的真实世界攻击，以及2) 评估最先进的自动化工具是否能够发现这些不变量。实验工具链非常复杂。它基于INVARIANTEVAL，这是一个包含28个真实以太坊漏洞利用的基准，每个漏洞都配有人工编写的不变量，用于阻止攻击。我们使用PONDEREPLAY验证每个不变量，这是一个重放框架，通过重新执行交易来证明智能合约不变量的正确性和健全性。我们证明了智能合约不变量能够阻止攻击。

    arXiv:2608.13191v1 Announce Type: cross  Abstract: Blockchains are among the most adversarial environments in computing. Billions are stolen by cybercriminals who exploit vulnerabilities. This is an open problem and no concept or technique has proven to really make a difference. In this paper, we claim that the classical notion of program invariant is perhaps the most powerful solution to the problem. We devise anoriginal experimental protocol to 1) study how invariants would have protected against past real-world attacks and 2) whether state-of-the-art automated tools can find them. The experimental toolchain is sophisticated. It is based on INVARIANTEVAL, a benchmark of 28 real Ethereum exploits, each paired with a human-authored invariant that blocks the attack. We validate every invariant with PONDEREPLAY, a replay framework that re-executes transactions in order to prove the correctness and soundness of smart contract invariants. We demonstrate that smart contract invariants block
    
[^10]: 大型语言模型在生成形式化程序规范方面有多强大？

    How Powerful are LLMs in Generating Formal Program Specifications?

    [https://arxiv.org/abs/2608.13077](https://arxiv.org/abs/2608.13077)

    本文提出了Coins框架，通过实例化规范并生成具体证明义务来评估LLM生成形式化程序规范的质量，避免了传统方法中证明难度与规范质量混淆的问题。

    

    形式化验证提供了软件正确性的强有力保证，但其应用受到编写精确形式化规范高昂成本的限制。尽管近期大型语言模型（LLMs）在定理证明和已验证代码生成方面展现出强大能力，但它们生成程序规范的真实能力仍不明确。现有评估要么需要验证实现一致性，要么需要证明规范之间的语义等价性，这两者都极其困难，且可能将证明难度与规范质量混为一谈。为解决此问题，我们引入了Coins，一个基于Rocq的评估框架，通过在被评估的规范上实例化可信测试用例并生成具体证明义务来评估规范质量。这种设计符合形式推理的非对称性，其中成功的证明提供可靠证据，而证明失败则不然。

    arXiv:2608.13077v1 Announce Type: new  Abstract: Formal verification provides strong guarantees of software correctness, but its adoption is limited by the high cost of writing precise formal specifications. While recent large language models (LLMs) have shown strong capabilities in theorem proving and verified code generation, their true ability to generate program specifications remains unclear. Existing evaluations require either verifying implementation conformance or proving semantic equivalence between specifications, both of which are formidably difficult and may conflate proof difficulty with specification quality. To address this problem, we introduce Coins, a Rocq based evaluation framework that assesses specification quality by instantiating specifications under evaluation on trusted test cases and generating concrete proof obligations. This design aligns with the asymmetric nature of formal reasoning, where successful proofs provide reliable evidence while proof failures ar
    
[^11]: 静态分析引导的智能体AI翻译使Rust成为全栈生物信息学语言

    Static analysis-guided agentic AI translation enables Rust as a full stack bioinformatics language

    [https://arxiv.org/abs/2608.13029](https://arxiv.org/abs/2608.13029)

    本文提出利用静态分析引导的智能体AI将遗留生物信息学代码自动翻译为Rust语言，大幅缩减代码体积、提升构建和运行性能，并消除Unix依赖，实现了全栈式单细胞分析。

    

    生物信息学领域长期受困于遗留代码——这些代码虽被广泛使用，但可能缺乏维护者，或用现已不熟悉的语言（如Perl、Fortran）编写。这带来了维护成本（技术债务），而动态类型语言也对环境产生负面影响，且未能充分利用现代硬件。遗留代码还可能存在安全或可靠性问题，使其不适合在临床环境中使用。在此，我们展示了结合静态分析的智能体AI可用于将遗留代码翻译成现代语言Rust。我们提供了提示词和支持软件以促进系统性翻译，并在常见的NGS和成像软件上进行了评估。我们在我们的软件Bascet上展示了结果：大小减少约80倍，构建时间缩短约10倍，关键步骤性能提升超过3倍。此外，Unix依赖被移除，使Bascet成为唯一能实现单细胞全流程的管道。

    arXiv:2608.13029v1 Announce Type: cross  Abstract: The field of bioinformatics struggles with legacy code - old code that is commonly used but may no longer have a maintainer, or may be written in an now-unfamiliar language (e.g. Perl, Fortran). This incurs maintenance cost (technical debt), but dynamically typed languages also negatively impacts the environment and fail to make use of modern hardware. Legacy code may also have security or safety problems that make it unsuited for use in clinical settings. Here we show that agentic AI, combined with static analysis, can be used to translate legacy code to the modern language Rust. We provide prompts and supporting software to aid systematic translation, and evaluate it on common software for NGS and imaging. We showcase the result on our software Bascet: Size was reduced by ~80x, build time decreased by ~10x, and performance of key steps improved >3x. Unix dependencies were also removed, making Bascet the only single-cell pipeline able
    
[^12]: 基于需求增强生成的LLM软件可信验收测试

    Requirements-Augmented Generation for Trustworthy Acceptance Testing of LLM-Based Software

    [https://arxiv.org/abs/2608.12970](https://arxiv.org/abs/2608.12970)

    该论文提出了一种基于需求增强生成（REAG）的自动化验收测试框架，通过检索软件需求、领域知识和用户角色来校准LLM软件的裁决可靠性，解决了传统验收测试无法适应LBS上下文相关随机行为的问题。

    

    arXiv:2608.12970v1 公告类型：新 摘要：基于LLM的软件（LBS）将大语言模型作为核心组件集成，以提供灵活、个性化的响应。与传统具有确定性输出的软件不同，LBS表现出依赖于上下文、随机的行为，这使得经典的验收测试和测试预言机不再适用：相同的查询可能根据用户角色和软件上下文需要根本不同的响应。这一差距催生了对自动化验收测试框架的迫切需求，该框架能够自主解释用户指令，同时在变化的环境中可靠地推断用户意图。在本文中，我们提出了一个用于LBS的自动化验收测试框架，通过两项技术贡献实现校准的裁决可靠性。首先，我们引入了需求增强生成（REAG），它通过自适应RAG和自推理检索相关软件需求、领域知识和用户角色来解释用户意图。

    arXiv:2608.12970v1 Announce Type: new  Abstract: LLM-based software (LBS) integrates large language models as core components to deliver flexible, personalised responses. Unlike traditional software with deterministic outputs, LBSs exhibit context-dependent, stochastic behaviour that renders classical acceptance testing and test oracles insufficient: the same query may require fundamentally different responses depending on user personas and software context. This gap creates an urgent need for automated acceptance testing frameworks that autonomously interpret user instructions, while reliably inferring user intentions in a changing environment. In this paper, we present an automated acceptance testing framework for LBS with calibrated verdict reliability via two technical contributions. First, we introduce Requirements-Augmented Generation (REAG), which interprets user intentions by retrieving relevant software requirements, domain knowledge, and personas via adaptive RAG and self-rea
    
[^13]: 剖析软件图：面向驱动引导模糊测试的结构洞察

    Dissecting Software Graphs: Structural Insights for Driver-Guided Fuzzing

    [https://arxiv.org/abs/2608.12859](https://arxiv.org/abs/2608.12859)

    本文提出了一种基于静态调用图的结构抽象方法，通过投影驱动特定覆盖率来分析多驱动模糊测试中的软件结构，并基于27个真实项目揭示了执行模式对模糊测试有效性的影响。

    

    arXiv:2608.12859v1 公告类型：新 摘要：许多软件系统通过命令行选项、子命令和配置标志暴露多种执行模式。对于此类程序，模糊测试既依赖于变异输入，也依赖于所调用的模式。然而，评估仍侧重于覆盖率和错误数量，未明确执行模式如何划分、重叠或遗漏软件结构，以及这些差异如何影响有效性。我们提出了一项关于多驱动模糊测试下软件结构的实证研究。我们提出了一种结构抽象，使用静态调用图作为共享主干，并将驱动特定的动态覆盖率投影到其上，以推导驱动诱导子图。基于此抽象，我们开发了一个四阶段方法论，包括主干构建、模糊测试与剖析、基于图的分析以及研究问题驱动的评估。我们将其应用于27个源自OSS-Fuzz的C/C++项目，涵盖43个可执行文件和854个驱动配置。在相同总预算下...

    arXiv:2608.12859v1 Announce Type: new  Abstract: Many software systems expose multiple execution modes through command-line options, subcommands, and configuration flags. For such programs, fuzzing depends on both mutated inputs and the invoked mode. Yet evaluations still focus on coverage and bug counts, leaving unclear how execution modes partition, overlap, and miss software structure, and how these differences affect effectiveness. We present an empirical study of software structure under multi-driver fuzzing. We propose a structural abstraction that uses a static call graph as a shared backbone and projects driver-specific dynamic coverage onto it to derive driver-induced subgraphs. Based on this abstraction, we develop a four-phase methodology for backbone construction, fuzzing and profiling, graph-based analysis, and research-question-driven evaluation. We apply it to 27 OSS-Fuzz-derived C/C++ projects, spanning 43 executables and 854 driver configurations. Under the same total 
    
[^14]: 代码大语言模型的记忆诊断应具备规模感知能力

    Memorization Diagnostics for Code LLMs Should be Scale-Aware

    [https://arxiv.org/abs/2608.12771](https://arxiv.org/abs/2608.12771)

    本文指出传统记忆探测方法在大型代码模型上失效，并提出通过可逆数学变换将表示负载与记忆分离，以实现规模感知的记忆诊断。

    

    arXiv:2608.12771v1 公告类型：交叉 摘要：大型代码语言模型在多大程度上依赖记忆而非真正理解，仍是一个高度争议的问题。尽管当前文献经常报道广泛的记忆现象，但评估跨密集架构的底层探测技术揭示，在大规模模型上这些技术的效用严重失效。传统的编码器式探测方法，如使用同义词模糊测试或死代码插入等扰动，在扩展模型中难以暴露记忆问题，即使在已知污染的基准测试上也是如此，而依赖对数概率的解码器式探测方法也表现出类似的性能退化。这些探测方法的具体失效模式，特别是为何这类技术能干扰较小模型但对较大模型无效，促使我们将表示负载与记忆解耦，而不是将其视为单一现象。通过对数值问题应用可逆数学变换，我们隔离了这两个因素。

    arXiv:2608.12771v1 Announce Type: cross  Abstract: The extent to which large language models for code rely on memorization over genuine understanding remains highly debated. While current literature frequently reports widespread memorization, evaluating the underlying probing techniques across dense architectures reveals a severe breakdown in their utility at scale. Traditional encoder-style probes using perturbations such as synonym fuzzing or dead-code insertion struggle to expose memorization in scaled models, even on known-contaminated benchmarks, and decoder-style probes that rely on log probabilities show similar performance degradation. The specific mode of failure for these probes, particularly why such techniques disrupt smaller models but fail to impact larger ones, motivates us to untangle representation load from memorization rather than treating them as a single phenomenon. By applying invertible mathematical transforms to numeric problems, we isolate these two factors and
    
[^15]: 它在所有环境中都能正确渲染吗？多模态大语言模型生成网页的跨环境兼容性研究

    Does It Render Everywhere? A Study of Cross-Environment Compatibility in MLLM-Generated Webpages

    [https://arxiv.org/abs/2608.12518](https://arxiv.org/abs/2608.12518)

    本研究首次系统评估了AI生成网页的跨环境兼容性，发现68%的网页存在渲染问题，揭示了当前多模态大语言模型在网页生成中的可靠性缺陷。

    

    多模态大语言模型（MLLMs）越来越多地被用于从视觉设计（如截图）自动生成网页。然而，现有评估仅限于在固定浏览器-设备配置下的视觉保真度评估，这种设置忽视了真实世界部署中的跨环境渲染兼容性问题。为填补这一空白，我们首次对AI生成网页的跨环境兼容性进行了系统性实证研究。具体而言，我们构建了WebCompat数据集，包含2032个标注实例，由8个代表性AI工具生成的网页组成，每个网页在9种浏览器和设备组合下进行渲染。我们分析了兼容性问题的普遍性、用户可感知的症状以及底层代码级根本原因。我们的研究结果显示，68%的生成网页至少存在一个兼容性问题，凸显了普遍存在的可靠性担忧。

    arXiv:2608.12518v1 Announce Type: new  Abstract: Multimodal Large Language Models (MLLMs) have been increasingly adopted to automate webpage generation from visual designs (e.g., screenshots). However, existing evaluations are limited to visual fidelity assessment under a fixed browser-device configuration. Such a setting overlooks the cross-environment rendering compatibility for real-world deployments.   To address this gap, we present the first systematic empirical study of cross-environment compatibility in AI-generated webpages. Specifically, we construct WebCompat, a dataset of 2,032 annotated instances, comprising webpages generated by 8 representative AI tools, each rendered across 9 browser-and-device combinations. We analyze the prevalence of compatibility issues, their user-perceptible symptoms, and underlying code-level root causes. Our findings reveal that 68% of generated webpages exhibit at least one compatibility issue, underscoring the pervasive reliability concerns su
    
[^16]: 以规范优先与AI编码代理实现收敛：一项在无测试预言机且无人工代码审查的717k行代码库中拆除核心架构不变量的案例研究

    Specification-first convergence with an AI coding agent: a case study of dismantling a core architectural invariant across 189 files in a 717k-line codebase with no test oracle and no human code review

    [https://arxiv.org/abs/2608.12440](https://arxiv.org/abs/2608.12440)

    本研究展示了一种规范优先协议，使AI编码代理能够在无测试预言机和人工审查的情况下，成功完成对大型代码库中核心架构不变量的拆除，这一任务传统上被认为需要重写。

    

    本文报告了一项单一、完全仪器化的案例研究，该研究涉及AI编码代理在规范优先协议下进行的大规模架构重构，且生成代码无人工审查，也无预先存在的预言机来验证目标行为。任务是在一个大型相互依赖的代码库中拆除一个核心不变量，作者评估该任务通过增量重构实际上不可行，这类变更通常需要重写。在本文所述的协议下，代理成功完成了该任务。系统是一个包含717,725行生产级TypeScript应用，跨越3,648个文件。任务要求拆除一个核心生命周期不变量：即UI面板在AI请求期间保持打开的保证。目标行为是流式生成在面板关闭后仍能存活，并在重新打开时能重新连接到同一实时流，且无需...

    arXiv:2608.12440v1 Announce Type: cross  Abstract: This paper reports a single, fully instrumented case study of a large-scale architectural refactoring by an AI coding agent under a specification-first protocol, with no human review of the generated code and no pre-existing oracle to validate the target behaviour. The task, dismantling a central invariant across a large interdependent codebase, was assessed by the author as effectively infeasible through incremental refactoring, the kind of change that conventionally calls for a rewrite instead. Under the protocol described here, the agent completed it successfully.   The system is a 717,725-line production TypeScript application across 3,648 files. The task required dismantling a core lifetime invariant: the guarantee that a UI panel remains open for the duration of an AI request. The target behaviour was that a streaming generation survives the closing of its panel and can be reattached, on reopening, to the same live stream with no
    
[^17]: SynWeaver：面向网络代理的网站先验任务与轨迹协同合成

    SynWeaver: Website-Prior Task and Trajectory Co-Synthesis for Web Agents

    [https://arxiv.org/abs/2608.12429](https://arxiv.org/abs/2608.12429)

    SynWeaver通过构建网站地图来获取网站先验知识，并协同合成任务与轨迹，从而提升网络代理对未见网站的泛化能力。

    

    arXiv:2608.12429v1 公告类型：交叉 摘要：网络代理通常难以泛化到未见过的网站，因为它们缺乏针对特定网站的监督。基于探索的数据合成方法减少了人工标注，但仍面临两个关键限制：它们往往无法覆盖网站的完整功能，并且缺乏足够的网站先验知识，容易提出虚构的任务，这反过来限制了后续轨迹合成的多样性和效率。我们提出了\textbf{SynWeaver}，一个网站先验任务-轨迹协同合成框架，旨在解决这些挑战。SynWeaver首先执行结构化的网站探索，构建一个网站地图，覆盖目标网站上功能不同的页面状态和可执行交互的广泛集合。然后，它从该地图中提取页面级和转换级监督，以训练一个具有网站特定先验的UI感知模型，从而实现更基于实际的任务提议。

    arXiv:2608.12429v1 Announce Type: cross  Abstract: Web agents often struggle to generalize to unseen websites because they lack website-specific supervision. Recent exploration-based data synthesis methods reduce manual annotation, but they still face two key limitations: they often fail to cover the full functionality of a website, and without sufficient website prior knowledge, they tend to propose hallucinated tasks, which in turn limits the diversity and efficiency of downstream trajectory synthesis. We present \textbf{SynWeaver}, a website-prior task-trajectory co-synthesis framework designed to address these challenges. SynWeaver first performs structured website exploration and constructs a website map that covers a broad set of functionally distinct page states and executable interactions on the target website. It then derives page-level and transition-level supervision from this map to train a UI-aware model with website-specific priors, enabling more grounded task proposals. 
    
[^18]: 人类在AI编码代理研究中被忽视

    Humans are Missing from AI Coding Agent Research

    [https://arxiv.org/abs/2608.12355](https://arxiv.org/abs/2608.12355)

    本文主张AI编码代理研究应从自主任务解决转向以人为中心的设计，强调通过任务对齐、可验证性、可引导性和适应性四个维度来增强人类与代理的协作。

    

    arXiv:2608.12355v1 公告类型：交叉 摘要：近年来，AI编码代理研究的进展显著提升了代理自主执行复杂软件工程任务的能力，从编辑大型代码库到执行长期开发工作流。然而，随着这些系统的进步，实际可用性的主要瓶颈逐渐从纯粹的任务解决能力转向用户如何与代理沟通、监督和信任方面的挑战。在本文中，我们主张从自主型编码代理转向以人为中心的编码代理：这些系统不仅旨在完成任务，还要与人类有效协作。我们识别了四个核心交互层面维度，表征了人类-代理任务解决循环：任务对齐、可验证性、可引导性和适应性。最后，我们概述了具体的研究方向以推进这些维度，包括用户参与的编码环境、全面的评估指标等。

    arXiv:2608.12355v1 Announce Type: cross  Abstract: Recent progress in AI coding agent research has led to rapid improvements in agents' ability to autonomously perform complex software engineering tasks, from editing large codebases to executing long-horizon development workflows. As these systems make strides, however, the primary bottleneck to practical usefulness increasingly shifts away from pure task-solving capability, and toward challenges in how users communicate with, supervise, and trust agents. In this position paper, we argue for a reorientation from autonomous to human-centered coding agents: systems designed not only to complete tasks, but to collaborate effectively with people. We identify four core interaction-level dimensions that characterize the human-agent task-solving loop: task alignment, verifiability, steerability, and adaptability. Finally, we outline concrete research directions to advance these dimensions, including user-involved coding environments, comprehe
    
[^19]: FlowLog：重新思考Datalog以支持快速且可扩展的静态分析

    FlowLog: Re-thinking Datalog for Fast and Extensible Static Analysis

    [https://arxiv.org/abs/2607.23971](https://arxiv.org/abs/2607.23971)

    FlowLog通过将Datalog程序编译为差分数据流，实现了静态分析的高效与可扩展性，在24个真实基准测试中优于现有引擎，并支持毫秒级增量更新和细粒度性能调优。

    

    arXiv:2607.23971v2 公告类型：替换交叉 摘要：Datalog被广泛用于构建静态分析器，然而现有引擎常常在效率和可扩展性之间被迫做出权衡。在实践中，静态分析并非运行一次就完事：用户会编辑事实、调整规则、诊断瓶颈，并且经常需要超出标准Datalog的语义，这些任务往往依赖临时工具或侵入式引擎重写。我们展示了FlowLog，一个将Soufflé风格程序转换为差分数据流可执行文件的Datalog编译器，以实现高效且可扩展的静态分析。在源自真实工作负载的24个基准测试中，FlowLog在运行时上持续优于最先进的引擎，同时保持内存高效并具有更好的扩展性。该演示使用DOOP指向分析。与会者可以运行它，将同一程序从一次性评估切换到增量评估，在毫秒内撤回事实并更新结果；还可以调整它，检查每个运算符的成本。

    arXiv:2607.23971v2 Announce Type: replace-cross  Abstract: Datalog is widely used to build static analyzers, yet existing engines often force a tradeoff between efficiency and extensibility. In practice, static analyses are not run once and forgotten: users edit facts, tune rules, diagnose bottlenecks, and often need semantics beyond standard Datalog, leaving these tasks to ad hoc tooling or invasive engine rewrites.   We demonstrate FlowLog, a Datalog compiler that turns Souffl\'e-style programs into Differential Dataflow executables for efficient and extensible static analysis. Across 24 benchmarks derived from real-world workloads, FlowLog consistently outperforms state-of-the-art engines in runtime while remaining memory-efficient and scaling better.   The demonstration uses a DOOP points-to analysis. Attendees run it, switching the same program from one-shot to incremental evaluation that retracts a fact and updates results in milliseconds; tune it, inspecting per-operator costs i
    
[^20]: 缺陷修复中的语义漂移：行为信号如何从报告传播到测试与补丁

    Semantic Drift in Bug Resolution: How Behavioral Signals Propagate from Reports to Tests and Patches

    [https://arxiv.org/abs/2607.18550](https://arxiv.org/abs/2607.18550)

    本文提出Desc2Fix框架，通过结构化行为锚点和多种相似度指标测量缺陷报告、测试与补丁间的语义对齐，发现LLM能可靠提取信号但对齐高度依赖表示方式，完整差异是最稳定的判断基础。

    

    arXiv:2607.18550v2 公告类型：替换  摘要：Desc2Fix是一个用于衡量缺陷报告、触发测试和开发者编写的修复之间语义对齐的框架。对齐通过结构化行为锚点（例如，复现步骤、API/异常线索、预期与实际行为）、确定性相似度指标（ROUGE、SBERT、CodeBERT、OpenAI嵌入）以及基于覆盖度、正确性和特异性的LLM判断来操作化。我们的分析涵盖了来自Defects4J和SWT-Bench的2,857个报告-测试-补丁三元组，使用了来自不同模型家族的两个广泛采用的指令调优LLM。LLM可靠地提取结构化信号（完整性高达90%），并表现出强大的跨模型一致性，为下游推理提供了稳定的语义输入契约。然而，对齐高度依赖于表示方式：仅词汇相似度是不够的，完整差异提供了判断报告-补丁对应关系的最稳定基础，而st

    arXiv:2607.18550v2 Announce Type: replace  Abstract: Desc2Fix is a framework for measuring semantic alignment between bug reports, triggering tests, and developer-written fixes. Alignment is operationalized through structured behavioral anchors (e.g., reproduction steps, API/exception cues, expected vs. actual behavior), deterministic similarity metrics (ROUGE, SBERT, CodeBERT, OpenAI embeddings), and LLM-based judgments grounded in coverage, correctness, and specificity. Our analysis covers 2,857 report-test-patch triplets from Defects4J and SWT-Bench using two widely adopted instruction-tuned LLMs from distinct model families. LLMs reliably extract structured signals (up to 90% completeness) and exhibit strong cross-model consistency, yielding a stable semantic input contract for downstream reasoning. However, alignment is highly representation-sensitive: lexical similarity alone is insufficient, full diffs provide the most stable basis for judging report-patch correspondence, and st
    
[^21]: Alipay-PIBench：面向编码代理的现实支付集成基准测试

    Alipay-PIBench: A Realistic Payment Integration Benchmark for Coding Agents

    [https://arxiv.org/abs/2607.14573](https://arxiv.org/abs/2607.14573)

    本文提出了Alipay-PIBench基准测试，通过现实支付宝支付集成任务评估编码代理，并展示了特定技能可显著提升代理性能（平均RPR提高10.31个百分点）。

    

    支付集成是一项要求很高的仓库级软件任务：代理必须选择合适的产品，实现协调的客户端-服务器流程，验证支付结果，并保持交易与业务状态之间的一致性。我们引入了Alipay-PIBench，这是一个用于评估编码代理在真实支付宝支付集成场景中表现的基准测试。它包含九个特定于产品的项目和18个任务实例，每个实例都分为基础功能完成和高级风险感知加固两种场景。场景特定的评分标准支持确定性静态检查、单元测试、集成测试和端到端检查，并辅以LLM辅助评估来满足语义需求。我们评估了六个编码代理模型，并报告了评分标准通过率（RPR）。在具备技能条件下，平均RPR范围从68.58%到91.37%。访问支付宝支付集成技能相对于无技能条件，平均提高了10.31个百分点的RPR。

    arXiv:2607.14573v4 Announce Type: replace  Abstract: Payment integration is a demanding repository-level software task: agents must select a suitable product, implement coordinated client-server flows, verify payment outcomes, and preserve consistency between transaction and business states. We introduce Alipay-PIBench, a benchmark for evaluating coding agents on realistic Alipay payment integration. It contains nine product-specific projects and 18 task instances, each organized into Basic functional-completion and Advanced risk-aware hardening scenarios. Scenario-specific rubrics support deterministic static, unit, integration, and end-to-end checks, supplemented by LLM-assisted assessment for semantic requirements. We evaluate six coding-agent models and report rubric pass rate (RPR). Under the with-skill condition, mean RPR ranges from 68.58% to 91.37%. Access to the alipay-payment-integration skill improves mean RPR by 10.31 percentage points on average relative to the without-ski
    
[^22]: 基于大语言模型的测试预言机：权威来源分类法——一项系统性文献综述

    LLM-Based Test Oracles: Source-of-Authority Taxonomy -- A Systematic Literature Review

    [https://arxiv.org/abs/2607.05031](https://arxiv.org/abs/2607.05031)

    本综述首次按权威来源对LLM测试预言机进行分类，发现超过半数预言机在无规范情况下仅依赖模型训练知识作出判决，揭示了该领域信任基础的隐患。

    

    摘要：大语言模型（LLMs）越来越多地通过编写测试预言机或直接充当预言机来决定软件行为是否正确。然而，两个预言机可能看起来相同，却基于不同的依据：一个断言编码了书面规范，另一个仅依赖于模型在训练中学到的内容。先前的二次研究按形式或技术对预言机进行分类，很少依据决定判决可信度的属性——即其权威来源。本系统性文献综述按照2020年系统综述和元分析首选报告项目（PRISMA）指南进行，筛选了2,436条记录至54项纳入研究，并通过引文搜索（滚雪球法）扩展至总计83项。我们沿着三个维度阅读了文献集：预言机权威的来源、其采取的形式以及裁决其的机制。语料库中略多于一半的预言机在没有规范的情况下做出判决。这就是关键所在。

    arXiv:2607.05031v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly decide whether software behaves correctly, either by writing a test oracle or by acting as one. Yet two oracles can look identical and rest on different ground: one assertion encodes a written specification, another only what the model learned in training. Prior secondary studies sort oracles by form or by technique, rarely by the property that governs how far a verdict can be trusted: where its authority comes from. This systematic literature review, reported under the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) 2020 guidelines, screens 2,436 records to 54 included studies, extended by citation searching (snowballing) to 83 in total. We read the corpus along three axes: the source of an oracle's authority, the form it takes, and the mechanism that adjudicates it. Just over half of the corpus reaches a verdict with no specification at all. That is what le
    
[^23]: 当大语言模型编写代码时，什么会出错？表征智能体代码助手的操作安全失败

    What Breaks When LLMs Code? Characterizing Operational Safety Failures of Agentic Code Assistants

    [https://arxiv.org/abs/2605.30777](https://arxiv.org/abs/2605.30777)

    本文通过分析大量论文和GitHub问题，首次系统性地分类了编码智能体在日常开发中发生的操作安全失败类型及其影响，并提出了一个多维度安全分类框架。

    

    基于大语言模型（LLMs）的自主编码智能体正迅速融入开发工作流程，然而，除了对明确恶意输入的评估外，其操作安全特性仍未被充分理解。在实践中，高影响失败常发生在良性的、目标导向的使用过程中，例如环境破坏、虚假成功报告等，而当前基准测试并未捕捉这些情况。当编码智能体用于日常开发任务时，实际会发生哪些类别的操作安全失败，其影响又是什么？我们提出了一项基于事件的实证研究，依托两个互补的证据流。我们筛选了来自22个顶级会议/期刊的68,816篇论文，精选出185项与安全相关的研究，并从广泛部署的LLM驱动的编码工具中挖掘了16,586个GitHub问题，人工确认了547个真实的安全失败案例。通过对两个语料库应用系统化的开放编码，我们推导出一个多维度的安全分类体系。

    arXiv:2605.30777v3 Announce Type: replace  Abstract: Autonomous coding agents built on large language models (LLMs) are rapidly being integrated into development workflows, yet their operational safety properties remain poorly understood beyond evaluations of explicitly malicious inputs. In practice, high-impact failures arise during benign, goal-directed use through environment breakage, fabricated success reports, etc. that current benchmarks do not capture. What categories of operational safety failures actually occur when coding agents are used for everyday development tasks and what is their impact? We present an incident-driven empirical study grounded in two complementary evidence streams. We screen 68,816 papers from 22 premier venues, curating 185 safety-relevant studies, and mine 16,586 GitHub issues from widely deployed LLM-powered coding tools, manually confirming 547 genuine safety failures. Applying systematic open coding over both corpora, we derive a multi-dimensional s
    
[^24]: Move语言中命令式一等函数的正式验证

    Formal Verification of Imperative First-Class Functions in Move

    [https://arxiv.org/abs/2605.10007](https://arxiv.org/abs/2605.10007)

    本文为Move语言中的一等命令式函数提出了基于行为谓词和状态标签的正式验证方法，并实现了在Move Prover中的编码与验证。

    

    arXiv:2605.10007v3 公告类型：替换交叉 摘要：Move Prover (MVP) 是一个用于验证用Move编程语言编写的智能合约的正式验证器。最近，Aptos上的Move扩展了高阶函数：作为一等值的命令式函数可以被传递、存储在数据结构中，并保留在持久存储中，从而实现动态分派。本文描述了Move规范语言中函数值的表示及其在MVP中的实现。我们引入了行为谓词，通过单状态或双状态谓词来表征Move函数（中止条件及前置/后置条件）。我们还引入了状态标签，用于命名表达式求值时的中间内存状态，并允许组合行为谓词以描述状态转换序列。在SMT层面，函数值通过对到达调用点的可能函数值进行区分来编码：当具体函数已知时...

    arXiv:2605.10007v3 Announce Type: replace-cross  Abstract: The Move Prover (MVP) is a formal verifier for smart contracts written in the Move programming language. Recently, Move on Aptos was extended with higher-order functions: imperative functions as first-class values that can be passed around, stored in data structs, and kept in persistent storage, enabling dynamic dispatch. This paper describes the representation of function values in the Move specification language and their implementation in MVP. We introduce behavioral predicates which characterize Move functions (aborts and pre/post conditions) by single-state or two-state predicates. We also introduce state labels for naming intermediate memory states in which expressions are evaluated and which allow to compose behavioral predicates to describe sequences of state transitions. On SMT level, function values are encoded by discriminating over the possible function values reaching a call site: when the concrete function is know
    
[^25]: 基础模型作为重构正确性检测的预言机

    Foundation Models as Oracles for Refactoring Correctness Detection

    [https://arxiv.org/abs/2605.02096](https://arxiv.org/abs/2605.02096)

    本研究首次证明基础模型可通过零样本提示有效检测Java IDE中的重构错误，在226个真实案例上达到高准确率，为自动化重构验证提供了新路径。

    

    arXiv:2605.02096v3 公告类型：替换 摘要：主流集成开发环境（IDE）中的重构工具可能引入意外的行为变化或编译错误，这一持续挑战削弱了开发者对自动化转换的信任。传统检测方法依赖于手工设计的先决条件以及静态和动态分析，但在适应性上仍有限，且可能遗漏细微的正确性问题。本研究探讨了基础模型作为检测Java程序中重构错误的预言机的潜力。我们在226个真实重构错误上评估了零样本提示（无需任务特定训练），这些错误收集自过去十多年广泛使用的Java IDE（IntelliJ-IDEA、Eclipse和NetBeans），涵盖47种重构类型。结果表明，基础模型在此任务上可以发挥作用，尽管不同模型的性能有所差异。在首次运行设置中，GPT-OSS-20B实现了80.5%的准确率，而GPT-5达到了更高水平。

    arXiv:2605.02096v3 Announce Type: replace  Abstract: Refactoring tools in popular Integrated Development Environments (IDEs) can introduce unintended behavioral changes or compilation errors, a persistent challenge that undermines developer trust in automated transformations. Traditional detection approaches rely on handcrafted preconditions, and static and dynamic analyses, yet remain limited in adaptability and can miss subtle correctness issues. This study examines the potential of foundation models to serve as oracles for detecting refactoring bugs in Java programs. We evaluate zero-shot prompting, without task-specific training, across 226 real refactoring bugs collected over more than a decade from widely used Java IDEs (IntelliJ-IDEA, Eclipse, and NetBeans), spanning 47 refactoring types. Our results indicate that foundation models can be effective for this task, although performance varies across models. In the first-run setting, GPT-OSS-20B achieved 80.5% accuracy, while GPT-5
    
[^26]: IACDM：交互式对抗收敛开发方法论——一个AI辅助软件开发的结构化框架

    IACDM: Interactive Adversarial Convergence Development Methodology -- A Structured Framework for AI-Assisted Software Development

    [https://arxiv.org/abs/2604.16399](https://arxiv.org/abs/2604.16399)

    本文提出IACDM框架，通过外部状态机驱动的8阶段对抗验证过程，解决AI辅助软件开发中因验证差距导致的效率降低和数据泄露问题。

    

    arXiv:2604.16399v3 公告类型：替换-交叉 摘要：2025年采用AI辅助开发暴露了一种与工具无关的失败模式：使用前沿模型的经验丰富的开发者实际速度更慢，却自认为更快，并且在一次生产展示中，10.3%的应用因配置不当的访问权限而泄露数据。这些失败有一个共同的结构性原因，即验证差距：在缺乏外部工具使用的情况下，没有任何语言模型能确定其生成内容是否正确。工具无关紧要；过程起决定性作用。我们提出了IACDM（交互式对抗收敛开发方法论），这是一个8阶段框架，其中生成器外部的验证代理在离散门控处运行，AI交替进行构建工件和通过专业批评视角攻击工件。它与所借鉴的审查和红队传统不同之处在于，门控由模型外的状态机强制执行：代理

    arXiv:2604.16399v3 Announce Type: replace-cross  Abstract: Adoption of AI-assisted development in 2025 exposed a tool-agnostic failure pattern: experienced developers using frontier models were measurably slower while believing they were faster, and 10.3% of applications in one production showcase leaked data through misconfigured access. These failures share a structural cause, the verification gap: absent external tool use, no language model can determine whether what it generated is correct. The tool is irrelevant; the process is determinative. We present IACDM (Interactive Adversarial Convergence Development Methodology), an 8-phase framework in which verification agents external to the generator operate at discrete gates, and the AI alternates between building artifacts and attacking them through specialized critique lenses. What distinguishes it from the review and red-teaming traditions it borrows from is that the gate is enforced by a state machine outside the model: the agent 
    
[^27]: 程序代码中的错误理解：LLM-DL组合用于多标签分类的系统研究

    Error Understanding in Program Code: A Systematic Study of LLM-DL Combinations for Multi-label Classification

    [https://arxiv.org/abs/2603.25005](https://arxiv.org/abs/2603.25005)

    本研究系统评估了32种LLM-DL组合在代码多标签错误分类中的性能，发现CodeT5+GRU组合表现最佳，并揭示了编码器-解码器组件间的相互作用对性能的关键影响。

    

    编程是计算机科学和软件工程中的核心技能，但识别和解决代码错误对从业者来说仍然具有挑战性。大型语言模型（LLMs）在自然语言理解方面展现出卓越能力，然而，当代码专用LLMs与深度学习（DL）序列解码器配对时，它们的行为方式以及该流程中哪个组件驱动性能，仍未得到充分探索。本研究对用于源代码多标签错误分类（MLEC）的LLM-DL组合进行了系统评估。八个微调后的LLMs，包括CodeT5、GraphCodeBERT、CodeT5+、UniXcoder、RoBERTa、具有收窄学习率范围的RoBERTa、PLBART和CoTexT，与GRU、LSTM、BiLSTM以及带加性注意力机制解码器的BiLSTM集成，在真实世界的Python代码错误数据集上进行测试。使用Optuna调优的32个模型变体，通过全面的多标签指标套件进行评估。在单次运行评估中，CodeT5+ GRU表现最佳，加权F分数达到...

    arXiv:2603.25005v2 Announce Type: replace  Abstract: Programming is a core skill in CS and SE, yet identifying and resolving code errors remains challenging for practitioners. LLMs have shown remarkable capabilities in NL understanding, but how code-specialized LLMs behave when paired with DL sequence decoders, and which component of such a pipeline drives performance, remains insufficiently explored. This study presents a systematic evaluation of LLM-DL combinations for multi-label error classification (MLEC) of source code. Eight fine-tuned LLMs, including CodeT5, GraphCodeBERT, CodeT5+, UniXcoder, RoBERTa, RoBERTa with a narrowed learning-rate range, PLBART, and CoTexT, are integrated with GRU, LSTM, BiLSTM, and BiLSTM with an additive attention mechanism decoder on a real-world Python code error dataset. The resulting 32 model variants, tuned with Optuna, are assessed on a comprehensive multi-label metric suite. In single-run evaluation, CodeT5+ GRU performs best, with a weighted F
    
[^28]: CangjieBench：在低资源通用编程语言上对大型语言模型进行基准测试

    CangjieBench: Benchmarking LLMs on a Low-Resource General-Purpose Programming Language

    [https://arxiv.org/abs/2603.14501](https://arxiv.org/abs/2603.14501)

    本文提出了CangjieBench，一个针对低资源通用语言仓颉的无污染基准测试，系统评估发现语法约束生成在准确性和效率上表现最优。

    

    大型语言模型在高资源编程语言上表现出色，但在低资源编程语言上则表现不佳。现有的低资源编程语言研究主要集中于领域特定语言（DSL），而忽视了因数据稀缺而受限的通用语言。为填补这一空白，我们引入了CangjieBench，这是一个针对仓颉（Cangjie）语言的无污染基准测试，仓颉是一种具有代表性的低资源通用语言。该基准测试包含248个从HumanEval和ClassEval手动翻译的高质量样本，涵盖文本到代码和代码到代码两类任务。我们在四种设置下对多种大型语言模型进行了系统评估：直接生成、语法约束生成、检索增强生成（RAG）和智能体。实验表明，直接生成表现较差，而语法约束生成在准确性和计算成本之间提供了最佳平衡。

    arXiv:2603.14501v2 Announce Type: replace-cross  Abstract: Large Language Models excel in high-resource programming languages but struggle with low-resource ones. Existing research related to low-resource programming languages primarily focuses on Domain-Specific Languages (DSLs), leaving general-purpose languages that suffer from data scarcity underexplored. To address this gap, we introduce CangjieBench, a contamination-free benchmark for Cangjie, a representative low-resource general-purpose language. The benchmark comprises 248 high-quality samples manually translated from HumanEval and ClassEval, covering both Text-to-Code and Code-to-Code tasks. We conduct a systematic evaluation of diverse LLMs under four settings: Direct Generation, Syntax-Constrained Generation, Retrieval-Augmented Generation (RAG), and Agent. Experiments reveal that Direct Generation performs poorly, whereas Syntax-Constrained Generation offers the best trade-off between accuracy and computational cost. Agent
    
[^29]: 生成式人工智能对协作式开源软件开发的影响：来自GitHub Copilot的证据

    The Impact of Generative AI on Collaborative Open-Source Software Development: Evidence from GitHub Copilot

    [https://arxiv.org/abs/2410.02091](https://arxiv.org/abs/2410.02091)

    本研究通过GitHub Copilot的数据发现，生成式AI在开源软件开发中提高了代码贡献和开发者参与度，但同时也增加了协调时间和代码讨论，揭示出AI在扩展贡献范围与增加协作成本之间的权衡。

    

    生成式人工智能（AI）促进了内容生产并增强了构思能力，对开发者生产力和软件开发参与度具有潜在的重要影响。为了探索其对协作式开源软件（OSS）开发的影响，我们研究了GitHub Copilot（一种生成式AI结对程序员）在多个分布式开发者自愿协作的OSS开发中的作用。利用GitHub专有的Copilot使用数据，结合从GitHub获取的公开OSS项目数据，我们发现Copilot的使用使项目级别的代码贡献增加了5.9%。这一增益伴随着开发者编码参与度增加3.4%和个人代码贡献增加2.1%。然而，Copilot的使用还与协调时间增加8%和更多的代码讨论相关。这揭示了一个重要的权衡：虽然AI扩展了谁可以贡献以及如何贡献，但它也增加了协作的协调成本。

    arXiv:2410.02091v4 Announce Type: replace-cross  Abstract: Generative artificial intelligence (AI) facilitates content production and enhances ideation, with potentially important implications for developer productivity and participation in software development. To explore its impact on collaborative open-source software (OSS) development, we investigate the role of GitHub Copilot, a generative AI pair programmer, in OSS development where multiple distributed developers voluntarily collaborate. Using GitHub's proprietary Copilot usage data, combined with public OSS project data obtained from GitHub, we find that Copilot use increases project-level code contributions by 5.9%. This gain is accompanied by a 3.4% increase in developer coding participation and a 2.1% increase in individual code contributions. However, Copilot use is also associated with an 8% increase in coordination time and more code discussions. This reveals an important tradeoff: While AI expands who can contribute and 
    

