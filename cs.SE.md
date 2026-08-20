# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SiNMULI: Novel Signed Network Approach for Malicious URL Identification](https://arxiv.org/abs/2608.19190) | 本文提出SiNMULI，一种基于符号网络和社交平衡理论的新型恶意URL识别方法，将URL识别建模为符号网络二分类问题，以克服传统静态分析方法对动态恶意实体的不足。 |
| [^2] | [Pre-Compiled Pipeline Shards for Distributed LLM Inference on Intel AI PC Fleets](https://arxiv.org/abs/2608.19147) | 通过将大语言模型按层预编译为OpenVINO分片并利用流水线并行，在多个英特尔AI PC上实现分布式推理，通过注入beam_idx Gather触发GPU优化和投机解码，达到与单体推理相当的性能。 |
| [^3] | [Grouping the Stochastic Machine: Precision, Not Capability, as the Frontier Metric for AI Systems](https://arxiv.org/abs/2608.19140) | 本文提出前沿AI系统的关键区别在于输出精度（重复请求下结果的一致性），而非传统能力指标，并论证了精度可低成本、无循环地测量。 |
| [^4] | [Tuning the Stochastic Machine: A Systems Engineer's Operating Model for Human-AI Engineering](https://arxiv.org/abs/2608.19125) | 本文提出将LLM系统视为需要系统工程师操作纪律的随机机器，并基于映射失败推导出七项原则，核心是持久化错误纠正的循环机制。 |
| [^5] | [SkillForge: Self-Distilling Agents for Project-Specific Issue Resolution](https://arxiv.org/abs/2608.18933) | SkillForge通过自蒸馏框架主动合成项目特定问题，将可复用的项目知识提炼为技能，从而无需依赖历史修复信号或高测试成本即可提升代理在特定代码库中的问题解决能力。 |
| [^6] | [Contract-Aware Rescue of a Drifted Isabelle Development: The Double-Tank Case Study](https://arxiv.org/abs/2608.18822) | 本文提出了一种契约感知的证明修复方法，通过结合Isabelle接受性和独立契约检查，成功修复了漂移的Isabelle开发，解决了双水箱案例中的所有未完成目标，同时揭示了大型语言模型提议证明可能导致的验证任务弱化问题。 |
| [^7] | [Metrics That Write Themselves: Evolving an Evaluator from Its Own Blind Spots](https://arxiv.org/abs/2608.18744) | 本文提出EvalCEGAR方法，通过反例引导抽象细化自动演化评估指标，利用碰撞对（正确与错误答案评分相同）作为作者请求，从自身盲点中生成可解释的缺陷检测操作符池，解决了报告生成等场景中自动评分指标缺失的问题。 |
| [^8] | [Flama: a Python framework for development and deployment of production-ready APIs, machine learning, and LLM services](https://arxiv.org/abs/2608.18733) | Flama是一个基于ASGI的Python框架，通过统一架构和七个子系统简化了生产级API、机器学习模型和LLM应用的开发与部署，实现了类型驱动和异步优先的编程体验。 |
| [^9] | [Code Health in LLM-Based Test Generation: Effectiveness and Token Efficiency](https://arxiv.org/abs/2608.18645) | 本研究发现代码可维护性（以CodeHealth衡量）对LLM生成的单元测试有效性提供弱但一致的信号，且与输入令牌数量负相关，表明高质量代码更有利于LLM测试生成。 |
| [^10] | [OdinEval: A Reproducible Benchmark for LLM-Based Program Repair in the Odin Programming Language](https://arxiv.org/abs/2608.18595) | OdinEval提出了一个针对Odin语言的可复现修复基准，通过严格准入和黑盒测试验证，评估了六种模型，其中Kimi-K3在修复成功率上领先。 |
| [^11] | [AppEval: A Unified Benchmark for LLM-Based Mobile Application Repair in ArkTS, Swift, and Kotlin](https://arxiv.org/abs/2608.18588) | AppEval提出了一个统一的移动应用修复基准和评估框架，解决了跨平台（HarmonyOS、iOS、Android）的构建-安装-启动-测试边界问题，确保修复验证不被基础设施故障误判。 |
| [^12] | [SemaPLC: A Project-Grounded, Verification-Gated Agent Harness for PLC Code Generation](https://arxiv.org/abs/2608.18565) | SemaPLC通过严格的外部验证门控和项目集成，显著提升了大型语言模型生成的PLC代码在真实工业环境中的可靠性与通过率。 |
| [^13] | [Building real-time digital twin instances with Function+Data Flow: user evaluation and extension for iterative pipelines](https://arxiv.org/abs/2608.18480) | 本文通过用户研究评估了函数+数据流（FDF）可视化DSL及其集成环境DesCartes Builder，证明它们能提高基于AI的数字孪生开发的易用性和可靠性，并扩展以支持迭代管线。 |
| [^14] | [When Do Microservices Save Energy? Evidence from Environmental Simulation Workflows](https://arxiv.org/abs/2608.18376) | 本文发现微服务在大型环境模拟工作流中通过事件驱动编排和选择性重执行可节省能源，但在小型或紧密耦合模型中会增加能耗。 |
| [^15] | [One Gate Is Not Enough: Composing Stateful Pre-Action Controls for Agentic AI](https://arxiv.org/abs/2608.18360) | 本文发现智能体AI中多个动作前控制之间存在补救耦合，导致控制失效，并提出了一种“补救-重新门控”协议来恢复健全性，同时证明补救操作符不可交换，使补救顺序成为控制语义的关键部分。 |
| [^16] | [Engine-Transfer-Bench: An Evidence-Based Benchmark for Document Compilation Engine Selection](https://arxiv.org/abs/2608.18329) | 该论文提出了一个名为Engine-Transfer-Bench的基准框架，用于系统比较不同文档编译引擎的性能和可靠性，并发现引擎失败主要源于架构问题而非包缺失。 |
| [^17] | [What Makes Software Issue Resolution Tasks Difficult for Agents?](https://arxiv.org/abs/2608.18280) | 本文提出一个测量框架，通过分析CoderForge-Preview数据集中的任务补丁、仓库和提示特征，系统量化了软件任务的结构属性对智能体解决成功率的影响，揭示了哪些静态属性可预测任务难度。 |
| [^18] | [Reproducibility is Not Enough: Artifact Verifiability in Decentralized-Build Package Ecosystems](https://arxiv.org/abs/2608.18180) | 本文提出了一种独立验证者模型和分层工件比较方法，系统评估了四个去中心化构建包生态系统中工件的可验证性，揭示了仅靠可复现性不足以确保工件可独立验证的挑战。 |
| [^19] | [Demo: tfdrift - A Severity Taxonomy and Risk Classification Framework for Infrastructure Drift Detection](https://arxiv.org/abs/2608.18173) | 本文提出了一种基础设施漂移的严重性分类法，并通过tfdrift框架实现，将警报量减少73%的同时保留94%的安全关键变更，有效缓解了警报疲劳问题。 |
| [^20] | [Adversarial Review: Structured Disagreement for Grounded Agentic Code Review](https://arxiv.org/abs/2608.18167) | 本文提出了一种名为对抗性审查（AR）的最小合作代码审查协议，通过引入批评者智能体进行结构化分歧审计，在仅使用三个智能体的情况下超越了五个智能体的基线性能，并揭示了朴素方法中的虚假共识问题。 |
| [^21] | [MicroPython and CircuitPython: Pythons Quiet Takeover of IoT and Robotics](https://arxiv.org/abs/2608.18160) | 本文通过数据分析和基准测试，论证了MicroPython和CircuitPython正以降低开发门槛和推动教育普及的方式，在物联网和机器人领域实现对嵌入式系统的悄然接管。 |
| [^22] | [FPGA Lifecycle Management for RISC-V Systems](https://arxiv.org/abs/2608.18156) | 该论文提出了一种基于Linux操作系统层的与主机无关的控制平面架构，使RISC-V处理器能作为FPGA生命周期管理主机，实现供应商中立且可扩展的位流部署。 |
| [^23] | [PowderLine: a programmatic powder diffraction analysis application](https://arxiv.org/abs/2608.17009) | PowderLine是一个Python工具，通过声明式配方和版本化验证，将粉末衍射精修过程标准化为机器可读的程序化流程，从而支持高通量和自主实验。 |
| [^24] | [From Documentation to Zero-day Vulnerabilities: LLM-Driven Fuzzing of JavaScript Engines in PDF Readers](https://arxiv.org/abs/2608.06641) | 提出PDFuzzer，利用大语言模型自动生成复杂API调用序列的PDF引擎模糊测试器，显著提升了对主流PDF阅读器中JavaScript引擎漏洞的检测能力。 |
| [^25] | [MetaInfer: A Knowledge Only LLM Inference Engine Generator SKILL Toolbox](https://arxiv.org/abs/2607.12875) | 本文提出MetaInfer，一种利用LLM作为编译器，通过多智能体协作和契约知识库，仅根据用户指定的运行时约束自动生成定制化推理框架的方法，以减少代码复杂性和性能开销。 |
| [^26] | [From Quality Properties to Practice: A Guideline and Workflow for Explainability Requirements](https://arxiv.org/abs/2606.10882) | 本文提出并评估了一种指南驱动的可解释性需求制定工作流，通过提炼十个核心质量属性并操作化为工具，以改善需求制定的清晰度和一致性。 |
| [^27] | [Do Privacy Policies Match with the Logs? An Empirical Study of Privacy Disclosure in Android Application Logs](https://arxiv.org/abs/2604.18552) | 本研究通过对1000个安卓应用的实证分析发现，隐私政策与日志披露存在显著不一致，多数应用泄露未在政策中声明的敏感信息。 |
| [^28] | [A Framework and Prototype for a Navigable Map of Datasets in Engineering Design and Systems Engineering](https://arxiv.org/abs/2603.15722) | 本文提出了一种系统框架和交互式工具原型，通过多维分类体系构建“EDSE数据集地图”，以解决工程设计与系统工程领域数据集碎片化和不可访问的问题，促进数据发现和复用。 |
| [^29] | [When Agents Fail: A Comprehensive Study of Bugs in LLM Agents with Automated Labeling](https://arxiv.org/abs/2601.15232) | 本研究首次系统性地分析了LLM代理软件中的缺陷类型、根本原因和影响，并探讨了自动化缺陷识别的可行性。 |
| [^30] | [Professional Software Developers Don't Vibe, They Control: AI Agent Use for Coding in 2025](https://arxiv.org/abs/2512.14012) | 经验丰富的开发者将AI代理视为生产力工具，但坚持保留设计控制权，通过专业知识策略性地引导代理行为，以确保软件质量。 |
| [^31] | [A Configuration-First Framework for Reproducible, Low-Code Localization](https://arxiv.org/abs/2510.25692) | 本文提出了一种配置优先的框架LOCALIZE，通过配置文件声明实验和自动化工作流，实现了低代码、可复现的机器学习实验管理，特别适用于无线定位研究。 |
| [^32] | [Toward Inclusive AI-Driven Development: Exploring Gender Differences in Code Generation Tool Interactions](https://arxiv.org/abs/2507.14770) | 本研究通过对照实验发现，性别差异显著影响开发者与代码生成工具的交互效果和认知负荷，强调AI开发工具需更具包容性设计。 |
| [^33] | [`From Prompt to Perturbation': An Adaptive Framework for Voice-Based Jailbreaks on Audio LLMs](https://arxiv.org/abs/2502.00735) | 本文提出了一种自适应越狱攻击框架，能在统一设置下系统评估级联流水线和端到端音频大语言模型，覆盖更广泛的音频攻击空间。 |

# 详细

[^1]: SiNMULI：一种用于恶意URL识别的新型符号网络方法

    SiNMULI: Novel Signed Network Approach for Malicious URL Identification

    [https://arxiv.org/abs/2608.19190](https://arxiv.org/abs/2608.19190)

    本文提出SiNMULI，一种基于符号网络和社交平衡理论的新型恶意URL识别方法，将URL识别建模为符号网络二分类问题，以克服传统静态分析方法对动态恶意实体的不足。

    

    arXiv:2608.19190v1 公告类型：跨领域 摘要：在当今人工智能快速发展的时代，计算机安全和在线防护措施已得到显著改进。然而，恶意网站仍在助长网络钓鱼、欺诈活动和垃圾信息的传播。传统的机器学习、深度学习和假冒网站检测方法主要依赖静态数据分析，这往往难以有效应对不断演变的恶意在线实体。针对这些挑战，在本工作中，我们提出了一种基于符号网络的恶意URL识别方法SiNMULI。我们引入了一个创新框架，将有害URL的识别概念化为一个基于符号网络的二分类问题，该问题深深植根于社交网络分析和社会平衡理论的基本原理。在这种方法中，符号网络被...

    arXiv:2608.19190v1 Announce Type: cross  Abstract: In today's era of rapid advancements in artificial intelligence, computer security and online safeguarding measures have undergone significant improvements. However, malicious websites continue to facilitate the spread of phishing schemes, fraudulent activities and unsolicited communications. Conventional methodologies in machine learning, deep learning and counterfeit website detection predominantly depend on static data analysis, which frequently proves ineffective against the evolving nature of malicious online entities. In response to these challenges, in this work, we propose a signed network-based approach for malicious URL identification, SiNMULI. We introduce an innovative framework that conceptualises the identification of harmful URLs as a signed network-based binary classification problem strongly rooted in the fundamental principles of social network analysis and social balance theory. In this approach, a signed network is 
    
[^2]: 预编译流水线分片用于英特尔AI PC机群上的分布式大语言模型推理

    Pre-Compiled Pipeline Shards for Distributed LLM Inference on Intel AI PC Fleets

    [https://arxiv.org/abs/2608.19147](https://arxiv.org/abs/2608.19147)

    通过将大语言模型按层预编译为OpenVINO分片并利用流水线并行，在多个英特尔AI PC上实现分布式推理，通过注入beam_idx Gather触发GPU优化和投机解码，达到与单体推理相当的性能。

    

    arXiv:2608.19147v1 公告类型：交叉 摘要：现代英特尔AI PC配备了功能强大的集成GPU和NPU，拥有16GB以上的统一内存，但这些设备大部分时间处于闲置状态。这些内存不足以容纳像700亿参数的大语言模型这样的大型模型。我们证明，通过普通网络协作的一小批AI PC，可以服务超过任何单台设备能力的模型。我们采用流水线并行：模型按层拆分为每阶段的分片，每个分片预编译为OpenVINO图，这样每台机器运行一个分片并将激活传递给下一个分片。三种技术使其足够快速以实用。首先，我们恢复了未拆分模型的速度：朴素的每阶段导出性能远低于单体推理，因为它缺少OpenVINO GPU优化，而向每个分片注入beam_idx Gather可触发该优化（IndirectKVCache融合），使分片性能达到一致。其次，我们利用有状态OpenVINO模型上的投机解码。

    arXiv:2608.19147v1 Announce Type: cross  Abstract: Modern Intel AI PCs ship capable integrated GPUs and NPUs with 16+ GB of unified memory, and they spend considerable time idle. That is not enough memory to fit a large model such as a 70B-parameter LLM. We show that a handful of AIPCs, working together over an ordinary network, can serve models beyond the capability of any single one. We use pipeline parallelism: a model is split by layer into per-stage shards, each pre-compiled into an OpenVINO graph, so that every machine runs one shard and passes activations to the next. Three techniques make this fast enough to be useful. First, we recover the speed of the unsplit model: a naive per-stage export runs well below monolithic inference because it misses an OpenVINO GPU optimization, and injecting a beam_idx Gather into each shard triggers that optimization (the IndirectKVCache fusion) and brings the shards to parity. Second, we leverage speculative decoding on stateful OpenVINO models
    
[^3]: 随机机器的分组：精度而非能力，作为AI系统的前沿指标

    Grouping the Stochastic Machine: Precision, Not Capability, as the Frontier Metric for AI Systems

    [https://arxiv.org/abs/2608.19140](https://arxiv.org/abs/2608.19140)

    本文提出前沿AI系统的关键区别在于输出精度（重复请求下结果的一致性），而非传统能力指标，并论证了精度可低成本、无循环地测量。

    

    arXiv:2608.19140v1 公告类型：新公告 摘要：前沿语言模型在能力上进行比较、营销和基准测试——即它们最佳或平均输出能达到的水平。我认为这衡量了错误的维度。这些模型已经达到精度饱和：它们的平均输出落在目标上。在实践中，现在区分一个系统与另一个系统的是精度：在重复、相同的请求中，输出围绕该目标的紧密集中程度。借用射击手的区分，能力是平均弹着点的位置；可靠性是弹着群的散布大小。我提出三个主张。第一，精度而非能力，是系统之间的前沿区分因素，而基准测试文化系统性地未能测量这一点，只报告集中趋势而非离散度。第二，精度是可测量的，成本低廉且无循环性，只需在固定温度下多次运行一组确定性评分的任务，并计算每个任务结果的一致性——无需——

    arXiv:2608.19140v1 Announce Type: new  Abstract: Frontier language models are compared, marketed, and benchmarked on capability -- what their best or average output can achieve. I argue this measures the wrong axis. The models have saturated accuracy: their mean output lands on the target. What now separates one system from another in practice is precision: how tightly concentrated their outputs are around that target across repeated, identical requests. Borrowing the marksman's distinction, capability is where the average shot lands; reliability is the size of the group. I make three claims. First, precision, not capability, is the frontier differentiator between systems, and benchmark culture systematically fails to measure it, reporting central tendency rather than spread. Second, precision is measurable, cheaply and without circularity, by running a fixed suite of deterministically scored tasks many times at fixed temperature and computing the per-task consistency of outcomes -- no
    
[^4]: 调谐随机机器：人类-AI工程的系统工程师操作模型

    Tuning the Stochastic Machine: A Systems Engineer's Operating Model for Human-AI Engineering

    [https://arxiv.org/abs/2608.19125](https://arxiv.org/abs/2608.19125)

    本文提出将LLM系统视为需要系统工程师操作纪律的随机机器，并基于映射失败推导出七项原则，核心是持久化错误纠正的循环机制。

    

    arXiv:2608.19125v1 公告类型：新 摘要：当专家纠正LLM助手的错误时，纠正通常随会话结束而消失，错误类别会再次出现。我认为这是一个操作问题，而非工具问题：持久化纠正的机制已经存在并正在部署，但管理它们的纪律——带有来源的版本控制、复发监控、反指标、过时规则的淘汰——却缺失。作为一位拥有三十年经验的系统工程师，我将LLM堆栈映射到我职业已经操作的机器（固化硅、固件、可加载模块、持久配置、易失内存），识别映射失败之处（随机生成、仅概率性绑定的配置、默认无通用淘汰（验证）阶段），并从失败中推导出以错误循环为核心的操作纪律的七项原则。我实践中的三个案例说明了该机制，其中包括一个控制案例。

    arXiv:2608.19125v1 Announce Type: new  Abstract: When an expert corrects an LLM assistant's error, the correction usually dies with the session, and the error class returns. I argue this is an operations problem, not a tooling problem: mechanisms for persisting corrections exist and are shipping, but the discipline for governing them -- versioning with provenance, recurrence monitoring, counter-metrics, retirement of stale rules -- does not. Writing as a systems engineer of thirty years, I map the LLM stack onto the machines my profession already operates (frozen silicon, firmware, loadable modules, persistent configuration, volatile memory), identify where the mapping fails (stochastic generation, configuration that binds only probabilistically, no general-purpose retirement (verification) stage by default), and derive from the failures a seven-principle operating discipline with an error loop at its core. Three cases from my own practice illustrate the mechanism, among them a control
    
[^5]: SkillForge：用于项目特定问题解决的自蒸馏代理

    SkillForge: Self-Distilling Agents for Project-Specific Issue Resolution

    [https://arxiv.org/abs/2608.18933](https://arxiv.org/abs/2608.18933)

    SkillForge通过自蒸馏框架主动合成项目特定问题，将可复用的项目知识提炼为技能，从而无需依赖历史修复信号或高测试成本即可提升代理在特定代码库中的问题解决能力。

    

    基于大型语言模型（LLM）的代理在自动化软件问题解决方面表现出显著的能力，但它们往往因缺乏项目特定知识而难以解决特定代码库中的问题。现有的自进化方法从代码库历史或在线修复轨迹中获取此类知识，但它们要么依赖于可用的历史问题解决信号，要么在每次问题解决时产生高昂的测试时探索成本。在本文中，我们提出了SkillForge，一种自蒸馏框架，它主动从代码库本身获取项目特定知识。SkillForge不是等待真实问题暴露项目特定知识缺口，而是通过重新实现代码库中测试覆盖的核心功能来合成项目特定问题。通过解决这些合成问题，SkillForge将可重用的项目特定知识蒸馏为基于实体的技能和

    arXiv:2608.18933v1 Announce Type: cross  Abstract: Large language model (LLM) based agents have demonstrated remarkable proficiency in automated software issue resolution, yet they often struggle to resolve issues in a specific repository because they lack project-specific knowledge. Existing self-evolving approaches acquire such knowledge from repository history or online repair trajectories, but they either depend on available historical issue-resolution signals or incur substantial per-issue test-time exploration cost. In this paper, we propose SkillForge, a self-distillation framework that proactively acquires project-specific knowledge from the repository itself. Instead of waiting for real issues to expose project-specific knowledge gaps, SkillForge synthesizes project-specific issues by re-implementing test-covered core functionalities of the repository. By resolving these synthetic issues, SkillForge distills reusable project-specific knowledge into entity-grounded skills and a
    
[^6]: 契约感知的漂移Isabelle开发修复：双水箱案例研究

    Contract-Aware Rescue of a Drifted Isabelle Development: The Double-Tank Case Study

    [https://arxiv.org/abs/2608.18822](https://arxiv.org/abs/2608.18822)

    本文提出了一种契约感知的证明修复方法，通过结合Isabelle接受性和独立契约检查，成功修复了漂移的Isabelle开发，解决了双水箱案例中的所有未完成目标，同时揭示了大型语言模型提议证明可能导致的验证任务弱化问题。

    

    大型语言模型可以为交互式定理证明器提出证明，但成功的构建并不能表明周围的验证任务被保留。我们在一个采样数据双水箱控制器的Isabelle开发中研究此问题。工作始于九个理论和十个未完成的目标，增长为一个包含16个理论的构建，没有使用sorry、oops、添加公理或预言机，并积累了23个稳定和36个破损的证明状态。回顾性审计发现100个原始声明中有16个发生了实质性变化，包括一个弱化的端到端保证定理，该定理在其结论中假设了四个要求中的三个。我们使用CAPRI，一种契约感知的证明修复工具，通过结合Isabelle接受性和对仓库更改与机器可读编辑契约的独立检查来管理重建。重建在原始九理论结构内解决了所有十个范围化目标。

    arXiv:2608.18822v1 Announce Type: new  Abstract: Large language models can propose proofs for interactive theorem provers, but a successful build does not show the surrounding verification task was preserved. We study this problem in an Isabelle development of a sampled-data double-tank controller. The work began with nine theories and ten unfinished obligations, grew to a 16-theory build without sorry, oops, added axiomatisation, or oracle use, and accumulated 23 stable and 36 broken proof states. A retrospective audit found material changes in 16 of the 100 original declarations, including a weakened end-to-end assurance theorem that assumed three of the four requirements in its conclusion. We used CAPRI, a contract-aware proof-repair tool, to govern a reconstruction by combining Isabelle acceptance with an independent check of repository changes against machine-readable edit contracts. The reconstruction discharged all ten scoped obligations within the original nine-theory structure
    
[^7]: 自我编写的指标：从自身盲点演化出评估器

    Metrics That Write Themselves: Evolving an Evaluator from Its Own Blind Spots

    [https://arxiv.org/abs/2608.18744](https://arxiv.org/abs/2608.18744)

    本文提出EvalCEGAR方法，通过反例引导抽象细化自动演化评估指标，利用碰撞对（正确与错误答案评分相同）作为作者请求，从自身盲点中生成可解释的缺陷检测操作符池，解决了报告生成等场景中自动评分指标缺失的问题。

    

    arXiv:2608.18744v1 公告类型：新 摘要：智能体在可靠自动指标的引导下能快速进步，而没有指标则会停滞不前；最需要这种指标的应用（如报告生成）恰恰是无人知道如何评分的领域。指标能自我编写吗？说清什么使答案优秀很难，但指出答案的问题则相对容易，因此我们演化的指标是一个小型Python操作符池，每个操作符为一个命名的缺陷标记候选答案，或弃权，并投票。直接让模型生成操作符是行不通的：183个候选仅实现96种不同行为，且来自一个巨大空间中的狭窄区域。EvalCEGAR转而借鉴程序验证中的反例引导抽象细化方法。它将操作符池视为一种抽象，并搜索碰撞——即两个答案在操作符评分下相同，但一个正确一个错误。该配对（而非提示）成为创作请求，当碰撞击败所有尝试时，循环会扩大操作符的定义范围。

    arXiv:2608.18744v1 Announce Type: new  Abstract: Agents improve quickly against a reliable automatic metric and stall without one, and the applications that need them most, report generation among them, are the ones nobody knows how to score. Can the metric write itself? Saying what makes an answer good is hard; pointing at something wrong with one is easier, so the metric we evolve is a pool of small Python operators that each flag a candidate for one named defect, or abstain, and vote. Asking a model for operators directly does not work: 183 candidates realise only 96 distinct behaviours, from one narrow region of an enormous space. EvalCEGAR instead borrows counterexample-guided abstraction refinement from program verification. It reads the pool as an abstraction and searches for a collision, two answers the operators score identically, one correct and one not. That pair, not a prompt, is the authoring request, and when a collision defeats every attempt the loop widens what an opera
    
[^8]: Flama：一个用于开发与部署生产级API、机器学习和LLM服务的Python框架

    Flama: a Python framework for development and deployment of production-ready APIs, machine learning, and LLM services

    [https://arxiv.org/abs/2608.18733](https://arxiv.org/abs/2608.18733)

    Flama是一个基于ASGI的Python框架，通过统一架构和七个子系统简化了生产级API、机器学习模型和LLM应用的开发与部署，实现了类型驱动和异步优先的编程体验。

    

    arXiv:2608.18733v1 公告类型：交叉 摘要：我们介绍了Flama，一个用于开发与部署生产级Web API、机器学习服务和大语言模型（LLM）应用的开源Python框架。Flama基于异步服务器网关接口（ASGI）构建，提供了一种类型驱动、异步优先的编程模型，将REST API开发、预测模型服务和生成式AI推理统一在一个架构中。它围绕七个子系统组织：一个基于组件的依赖注入系统，在启动时根据类型注解解析处理器参数；一个可插拔的模式层，通过单一适配器支持Pydantic、Marshmallow和Typesystem；一个自动CRUD生成器，将SQLAlchemy表和模式类转换为由仓储和工作单元模式支持的REST端点；一个可移植的二进制格式（.flm），用于打包scikit-learn、TensorFlow、PyTorch和Hugging Face Transformers模型及其元数据，实现零拷贝...

    arXiv:2608.18733v1 Announce Type: cross  Abstract: We present Flama, an open-source Python framework for developing and deploying production-ready web APIs, machine learning services, and large-language-model (LLM) applications. Built on the Asynchronous Server Gateway Interface (ASGI), Flama offers a type-driven, async-first programming model that unifies REST API development, predictive model serving, and generative AI inference in one architecture.   It is organised around seven subsystems: a component-based dependency injection system resolving handler parameters from type annotations at startup; a pluggable schema layer supporting Pydantic, Marshmallow and Typesystem behind a single adapter; an automatic CRUD generator turning a SQLAlchemy table and a schema class into REST endpoints backed by the Repository and Unit of Work patterns; a portable binary format (.flm) packaging models from scikit-learn, TensorFlow, PyTorch and Hugging Face Transformers with their metadata for zero-c
    
[^9]: 基于LLM的测试生成中的代码健康：有效性与令牌效率

    Code Health in LLM-Based Test Generation: Effectiveness and Token Efficiency

    [https://arxiv.org/abs/2608.18645](https://arxiv.org/abs/2608.18645)

    本研究发现代码可维护性（以CodeHealth衡量）对LLM生成的单元测试有效性提供弱但一致的信号，且与输入令牌数量负相关，表明高质量代码更有利于LLM测试生成。

    

    arXiv:2608.18645v1 公告类型：新 摘要：由大型语言模型（LLM）驱动的编码代理在软件工程领域日益突出。先前的研究表明，AI工具在高质量、易于维护的源代码上表现更好。在本研究中，我们调查了LLM生成的单元测试的有效性如何随可维护性水平变化，该水平由CodeScene的CodeHealth（CH）衡量。我们使用传统覆盖率指标和变异得分来评估Python、Java和C++中的测试有效性。此外，我们研究了不同CH水平的代码如何通过常见工业分词器转化为输入令牌。我们的结果表明，CH提供了LLM生成测试有效性的弱但一致的信号，并与输入令牌数量呈负相关。这些发现进一步证明了可维护性与基于LLM的软件开发之间的关系。

    arXiv:2608.18645v1 Announce Type: new  Abstract: Coding agents powered by Large Language Models (LLMs) are now prominent in software engineering. Previous work has shown that AI tools perform better on high-quality source code that is easy to maintain. In this study, we investigate how the effectiveness of LLM-generated unit tests varies across maintainability levels measured by CodeScene's CodeHealth (CH). We assess test effectiveness using traditional coverage metrics and mutation score across Python, Java, and C++. Moreover, we study how code with different levels of CH translates into input tokens using common industrial tokenizers. Our results suggest that CH provides a weak but consistent signal of LLM-generated test effectiveness and is negatively correlated with input-token count. These findings provide further evidence for a relationship between maintainability and LLM-based software development.
    
[^10]: OdinEval：一个基于Odin编程语言的可复现LLM程序修复基准

    OdinEval: A Reproducible Benchmark for LLM-Based Program Repair in the Odin Programming Language

    [https://arxiv.org/abs/2608.18595](https://arxiv.org/abs/2608.18595)

    OdinEval提出了一个针对Odin语言的可复现修复基准，通过严格准入和黑盒测试验证，评估了六种模型，其中Kimi-K3在修复成功率上领先。

    

    仓库级修复基准仍集中于少数主流语言，导致Odin等系统编程语言在很大程度上未被测试。我们提出了OdinEval，一个从公共Odin仓库中记录的缺陷构建的可复现基准。每个实例将问题与基础提交和修复提交、一个黄金补丁、一个特定于问题的回归测试、一个历史工具链以及执行记录绑定。准入要求测试在基础版本上失败，并在黄金修复后通过。当没有可用的开发者测试时，一个黑盒测试由同一模型的三个实例独立审查，在历史状态下执行，并根据版本化测试编写技能从记录反馈中修订。我们在统一协议下评估了168个过滤实例上的六种语言模型。Kimi-K3以66.7%的最高Resolved分数记录，而Qwen3.8-Max以96.4%的最高Repro分数。发布包括冻结数据、源代码存档和c（截断部分）

    arXiv:2608.18595v1 Announce Type: new  Abstract: Repository-level repair benchmarks still center on a few mainstream languages, leaving systems languages such as Odin largely untested. We present OdinEval, a reproducible benchmark built from documented defects in public Odin repositories. Each instance binds an issue to base and fix commits, a gold patch, an issue-specific regression test, a historical toolchain, and execution records. Admission requires the test to fail on the base revision and pass after the gold fix. When no usable developer test exists, a black-box test is reviewed independently by three instances of the same model, executed in both historical states, and revised from recorded feedback under a versioned Test Writing Skill. We evaluate six language models on 168 filtered instances under one shared protocol. Kimi-K3 records the highest Resolved score at 66.7%, while Qwen3.8-Max has the highest Repro score at 96.4%. The release includes frozen data, source archives, c
    
[^11]: AppEval：一个用于基于LLM的移动应用修复的统一基准，涵盖ArkTS、Swift和Kotlin

    AppEval: A Unified Benchmark for LLM-Based Mobile Application Repair in ArkTS, Swift, and Kotlin

    [https://arxiv.org/abs/2608.18588](https://arxiv.org/abs/2608.18588)

    AppEval提出了一个统一的移动应用修复基准和评估框架，解决了跨平台（HarmonyOS、iOS、Android）的构建-安装-启动-测试边界问题，确保修复验证不被基础设施故障误判。

    

    arXiv:2608.18588v1 公告类型：新 摘要：仓库级别的LLM代理通常是在其测试在构建主机上运行的项目上进行评估。目前尚不清楚它们的修复是否能跨越移动构建-安装-启动-测试的边界，在该边界中，缺失的SDK、离线设备或断言前的崩溃可能被误认为是程序失败。我们提出了AppEval，一个基准和原生工具链评估框架，用于跨HarmonyOS/ArkTS、iOS/Swift和Android/Kotlin的移动应用修复。每个任务将隐藏行为测试与参考生产修复分开，并且只有在相同已安装应用目标在缺陷修订版上达到断言失败并在修复后通过时才被接受；基础设施故障仍是一个独立的结果。一个通用模式将此契约映射到每个平台的构建系统、运行时和测试运行器。审计过的Android分区包含来自24个独立可构建仓库的200个已接受的仪器化任务。在这些任务上，五个（部分被截断）

    arXiv:2608.18588v1 Announce Type: new  Abstract: Repository-level LLM agents are typically evaluated on projects whose tests run on the build host. It remains unclear whether their repairs survive the mobile build-install-launch-test boundary, where a missing SDK, offline device, or pre-assertion crash can be mistaken for a program failure. We present AppEval, a benchmark and native-toolchain evaluation framework for mobile application repair across HarmonyOS/ArkTS, iOS/Swift, and Android/Kotlin. Each task separates a hidden behavior test from the reference production fix and is accepted only when the same installed-app target reaches an assertion failure on the defective revision and passes after the fix; infrastructure failures remain a distinct outcome. A common schema maps this contract to each platform's build system, runtime, and test runner. The audited Android partition contains 200 accepted instrumentation tasks from 24 independently buildable repositories. On these tasks, fiv
    
[^12]: SemaPLC：一种面向PLC代码生成的、基于项目且验证门控的代理框架

    SemaPLC: A Project-Grounded, Verification-Gated Agent Harness for PLC Code Generation

    [https://arxiv.org/abs/2608.18565](https://arxiv.org/abs/2608.18565)

    SemaPLC通过严格的外部验证门控和项目集成，显著提升了大型语言模型生成的PLC代码在真实工业环境中的可靠性与通过率。

    

    可编程逻辑控制器（PLC）运行工业工厂，而大型语言模型已经能够为其生成独立的程序组织单元（POU）。此类逻辑是否能集成到现有PLC项目中并正确运行，仅在有限测试中得到了验证。我们提出了SemaPLC，这是一种基于项目且验证门控的代理框架，由常规工具组装而成，但受严格完成规则约束。SemaPLC并非在模型判断自身输出足够时停止，而是仅在记录的外部检查确认后才宣布任务完成。这些检查覆盖规范、编译以及实时运行时的行为。在117个匹配现有基准的独立POU任务中，它在所有七个模型上达到了最高的严格验证通过率（平均72.6%）。在65个任务的项目上下文轨道中，其生成的逻辑必须在真实项目内编译和运行，它实现了...

    arXiv:2608.18565v1 Announce Type: new  Abstract: Programmable logic controllers (PLCs) run industrial plants, and large language models can already generate independent program organization units (POUs) for them. Whether such logic integrates into an existing PLC project and then runs correctly has been checked only in limited tests. We present \textsc{SemaPLC}, a project-grounded and verification-gated agent harness assembled from conventional tools but governed by a strict completion rule. Rather than stopping when the model judges its own output adequate, \textsc{SemaPLC} declares a task complete only when logged external checks confirm it. Those checks cover the specification, the compilation, and the behavior on a live runtime. On 117 independent-POU tasks matching existing benchmarks, it attains the highest strict verified pass rate on all seven models (72.6\% mean). On a project-context track of 65 tasks whose generated logic must compile and run inside a real project, it attain
    
[^13]: 构建实时数字孪生实例的函数+数据流：用户评估及迭代管线的扩展

    Building real-time digital twin instances with Function+Data Flow: user evaluation and extension for iterative pipelines

    [https://arxiv.org/abs/2608.18480](https://arxiv.org/abs/2608.18480)

    本文通过用户研究评估了函数+数据流（FDF）可视化DSL及其集成环境DesCartes Builder，证明它们能提高基于AI的数字孪生开发的易用性和可靠性，并扩展以支持迭代管线。

    

    数字孪生（DT）越来越多地利用人工智能（AI）和机器学习（ML）管线，既用于从高保真仿真构建实时DT，也用于使用历史数据实例化它们。然而，工程化这些管线在很大程度上仍是临时性的：管线难以指定、验证和重用，且缺乏专门的工具支持。函数+数据流（FDF）通过定义一种可视化的领域特定语言（DSL）来解决这一问题，该语言显式表示函数（ML模型），从而实现其组合和重用。我们在DesCartes Builder中实现了FDF，这是一个支持基于FDF的DT合成和验证的集成建模环境。本文报告了一项实证用户研究，评估FDF和DesCartes Builder是否能使基于AI的DT开发更易访问和更可靠。参与者在DesCartes Builder中实现了一个代表性的实时DT原型，我们测量了感知可用性。

    arXiv:2608.18480v1 Announce Type: cross  Abstract: Digital twins (DTs) increasingly leverage artificial intelligence (AI) and machine learning (ML) pipelines, both to build real-time DTs from high-fidelity simulations and to instantiate them with historical data. However, engineering these pipelines remains largely ad-hoc: pipelines are hard to specify, validate, and reuse, with scarce dedicated tooling. Function+Data Flow (FDF) addresses this by defining a visual domain-specific language (DSL) that represents functions (ML models) explicitly, enabling their composition and reuse. We implemented FDF in DesCartes Builder, an integrated modeling environment supporting FDF-based DT synthesis and validation.   In this paper, we report on an empirical user study evaluating whether FDF and DesCartes Builder can make AI-based DT development more accessible and reliable. Participants implemented a representative real-time DT prototype within DesCartes Builder, and we measured perceived usabili
    
[^14]: 微服务何时能节省能源？来自环境模拟工作流的证据

    When Do Microservices Save Energy? Evidence from Environmental Simulation Workflows

    [https://arxiv.org/abs/2608.18376](https://arxiv.org/abs/2608.18376)

    本文发现微服务在大型环境模拟工作流中通过事件驱动编排和选择性重执行可节省能源，但在小型或紧密耦合模型中会增加能耗。

    

    环境模拟模型支持情景分析、校准和决策制定，但重复执行可能会产生显著的能源成本。微服务提供了模块化和可扩展性，但其低碳影响仍不明确，因为分解引入了编排、通信、持久化和空闲服务开销。本文评估了四种环境模型作为容器化微服务工作流，比较了单体执行与基于轮询和事件驱动的编排方式。结果表明，对于较小或紧密耦合的模型，微服务会增加能源消耗，其中协调开销占主导地位。对于较大的工作流，事件驱动的编排在运行时间更长的情况下仍能减少能源使用，而在重复参数探索过程中，选择性下游重执行实现了41%的能源减少。

    arXiv:2608.18376v1 Announce Type: cross  Abstract: Environmental simulation models support scenario analysis, calibration, and decision-making, but repeated execution can incur significant energy costs. Microservices offer modularity and scalability, yet their low-carbon impact remains unclear because decomposition introduces orchestration, communication, persistence, and idle-service overheads. This paper evaluates four environmental models as containerised microservice workflows, comparing monolithic execution with polling-based and event-driven orchestration. Results show that microservices increase energy consumption for smaller or tightly coupled models, where coordination overhead dominates. For a larger workflow, event-driven orchestration reduces energy use despite longer runtime, while selective downstream re-execution achieves a 41% reduction during repeated parameter exploration.
    
[^15]: 一道门不够：为智能体AI组合有状态的动作前控制

    One Gate Is Not Enough: Composing Stateful Pre-Action Controls for Agentic AI

    [https://arxiv.org/abs/2608.18360](https://arxiv.org/abs/2608.18360)

    本文发现智能体AI中多个动作前控制之间存在补救耦合，导致控制失效，并提出了一种“补救-重新门控”协议来恢复健全性，同时证明补救操作符不可交换，使补救顺序成为控制语义的关键部分。

    

    摘要：arXiv:2608.18360v1 公告类型：交叉 摘要：智能体AI系统在采取关键行动时，同时受多个动作前控制约束：权限门、资源门和证据门，这些控制可以在行动执行前允许、降级或补救该行动。本文的核心对象是补救引发的控制耦合：一个控制应用的补救措施可能改变另一个控制所评估的行动、证据或上下文，从而使其先前的判断失效。我们形式化了这种耦合，并提出了一种“补救-重新门控”协议，该协议在给定的有界、幂等设置及其假设下恢复了每个动作的健全性。我们还进一步表明，两种已实现的补救操作符（证据替换和资源预算降级）不满足交换律——一个有限模型检查器发现了具体的反例实例——这使得补救顺序成为控制平面语义的一部分，而非实现细节。一个受治理的证据缓冲区信任其输入。

    arXiv:2608.18360v1 Announce Type: cross  Abstract: Agentic AI systems take consequential actions governed by more than one pre-action control at once: authority, resource, and evidence gates that can admit, degrade, or remediate an action before it executes. This paper's central object is remediation-induced control coupling: a remediation applied by one control can change the action, evidence, or context another control evaluates, invalidating that control's earlier judgment. We formalize this coupling and give a remediate-and-regate protocol that restores per-action soundness in the current bounded, idempotent setting under its stated assumptions. We further show that the two implemented remediation operators (evidence substitution and resource-budget downroute) do not commute -- a finite-model checker finds concrete counterexample instances -- making remediation order part of the control-plane semantics rather than an implementation detail. A governed evidence buffer that trusts its
    
[^16]: 引擎转换基准：文档编译引擎选择的证据基础基准

    Engine-Transfer-Bench: An Evidence-Based Benchmark for Document Compilation Engine Selection

    [https://arxiv.org/abs/2608.18329](https://arxiv.org/abs/2608.18329)

    该论文提出了一个名为Engine-Transfer-Bench的基准框架，用于系统比较不同文档编译引擎的性能和可靠性，并发现引擎失败主要源于架构问题而非包缺失。

    

    摘要：目前缺乏在文档编译引擎（如pdfLaTeX、XeLaTeX、LuaLaTeX、Tectonic、Typst和pandoc PDF后端）之间进行选择的共享框架。我们提出了引擎转换基准（ETB）：包含1,784篇开放文档，涵盖可靠性、延迟、文本一致性和失败情况四个任务，并配有固定测试框架和按主机标记的多操作系统结果。在GitHub Actions上（每个主机在macOS、Ubuntu和Windows上进行N=4,211次编译），Tectonic的成功率稳定在0.9个百分点内（96.3%-97.2%），而经典的TeX Live风格引擎根据发行策略（Ubuntu apt、MiKTeX自动安装或macOS BasicTeX）波动12-20个百分点。在702篇可移植LaTeX文档中，测试引擎均达到100%成功率，因此延迟成为主要选择因素；失败集中在107个引擎特定模板中。在ETB内，失败是架构性的，涉及字体、布局和资源，而非预配置主机上缺失包。一个50...

    arXiv:2608.18329v1 Announce Type: new  Abstract: There is no shared framework for selecting among document compilation engines (pdfLaTeX, XeLaTeX, LuaLaTeX, Tectonic, Typst, and pandoc PDF backends). We present Engine-Transfer-Bench (ETB): 1,784 open documents, four tasks covering reliability, latency, text consistency, and failures, a pinned harness, and host-tagged multi-OS results. On GitHub Actions (N=4,211 compiles per host across macOS, Ubuntu, and Windows), Tectonic success is stable within 0.9 percentage points (96.3-97.2%), whereas classic TeX Live-style engines vary by 12-20 percentage points according to distribution policy (Ubuntu apt, MiKTeX auto-install, or macOS BasicTeX). On 702 portable LaTeX documents, the tested engines succeed at 100%, making latency the primary selection factor; failures concentrate in 107 engine-specific templates. Within ETB, failures are architectural, involving fonts, layout, and assets, rather than missing packages on a provisioned host. A 50-
    
[^17]: 什么因素使得软件问题解决任务对智能体来说变得困难？

    What Makes Software Issue Resolution Tasks Difficult for Agents?

    [https://arxiv.org/abs/2608.18280](https://arxiv.org/abs/2608.18280)

    本文提出一个测量框架，通过分析CoderForge-Preview数据集中的任务补丁、仓库和提示特征，系统量化了软件任务的结构属性对智能体解决成功率的影响，揭示了哪些静态属性可预测任务难度。

    

    摘要：arXiv:2608.18280v1 公告类型：交叉 摘要：背景。智能体系统的进展同时且迅速地使基准测试趋于饱和。尽管这一现象常被讨论，但由于缺乏对任务难度的控制和表征，基准分数仍然难以解释。更具体地说，我们目前对什么使一个任务比另一个更难，以及任务难度在多大程度上可以从静态任务属性中预测，了解甚少。目标。我们提出了一个测量框架，以研究和系统量化软件任务的结构属性如何对应于智能体在问题解决任务中的成功率。方法。我们在CoderForge-Preview（迄今为止最大的编码智能体轨迹开放数据集）上进行了一项大规模实证研究，通过提取任务补丁、仓库和提示中的特征。我们使用集成方法、SHAP归因和效应评估了每个特征对任务结果预测能力。

    arXiv:2608.18280v1 Announce Type: cross  Abstract: Background. Advances in agentic systems are simultaneously, and rapidly, saturating benchmarks. Despite this often discussed phenomena, benchmark scores remain difficult to interpret due to the lack of control and characterization of task difficulty. More specifically, we currently have little understanding of what makes one task harder than another, and to what extent task difficulty is predictable from static task properties. Aims. We propose a measurement framework to investigate and systematically quantify what structural properties of software tasks correspond to agent success rates for issue resolution tasks. Method. We conducted a large scale empirical study on CoderForge-Preview, the largest open dataset of coding agent trajectories to date, by extracting features across task patch, repository and prompt. We evaluated the predictive power of each feature against task outcomes using ensemble methods, SHAP attribution, and effect
    
[^18]: 可复现性还不够：去中心化构建包生态系统中的工件可验证性

    Reproducibility is Not Enough: Artifact Verifiability in Decentralized-Build Package Ecosystems

    [https://arxiv.org/abs/2608.18180](https://arxiv.org/abs/2608.18180)

    本文提出了一种独立验证者模型和分层工件比较方法，系统评估了四个去中心化构建包生态系统中工件的可验证性，揭示了仅靠可复现性不足以确保工件可独立验证的挑战。

    

    arXiv:2608.18180v1 公告类型：新 摘要：可复现且可验证的构建通过使独立方能够检测由受损害的构建或发布管道产生的工件，从而增加对分布式软件工件的信任。然而，工件验证需要的不仅仅是确定性构建：验证者还必须恢复产生该工件的源代码状态、构建环境、依赖项和构建指令。去中心化构建生态系统使这变得困难，因为工件是通过异构工具、维护者控制的工作流和碎片化的元数据产生的。因此，这些生态系统中工件能被独立验证的频率仍不清楚。本文研究了四个流行的去中心化构建包生态系统中的工件可验证性。我们定义了一个仅依赖注册表可派生元数据的独立验证者模型，以及一个具有分层等价级别的工件比较模型。我们在中实现了这些模型

    arXiv:2608.18180v1 Announce Type: new  Abstract: Reproducible and verifiable builds increase trust in distributed software artifacts by enabling independent parties to detect artifacts produced by compromised build or release pipelines. However, artifact verification requires more than deterministic builds: a verifier must also recover the source state, build environment, dependencies, and build instructions that produced the artifact. Decentralized-build ecosystems make this difficult because artifacts are produced through heterogeneous tools, maintainer-controlled workflows, and fragmented metadata. As a result, it remains unclear how often artifacts in these ecosystems can be independently verified.   This paper studies artifact verifiability across four popular decentralized-build package ecosystems. We define an independent verifier model that relies only on registry-derivable metadata and an artifact comparison model with tiered equivalence levels. We implement these models in an
    
[^19]: 演示：tfdrift——基础设施漂移检测的严重性分类法与风险分类框架

    Demo: tfdrift - A Severity Taxonomy and Risk Classification Framework for Infrastructure Drift Detection

    [https://arxiv.org/abs/2608.18173](https://arxiv.org/abs/2608.18173)

    本文提出了一种基础设施漂移的严重性分类法，并通过tfdrift框架实现，将警报量减少73%的同时保留94%的安全关键变更，有效缓解了警报疲劳问题。

    

    摘要：像Terraform这样的基础设施即代码（IaC）工具已成为声明式云资源管理的标准，然而配置漂移（即部署的基础设施偏离其声明状态）仍然是一个持续存在的操作和安全挑战。当前的检测方法将所有变更等同对待，导致警报疲劳，使操作人员错过安全关键的修改。我们提出了一种针对基础设施漂移的通用严重性分类法，根据资源类型和属性级影响将变更分为四个风险等级。我们在tfdrift中实现了这一分类法，这是一个开源分类框架，包含60多条可配置规则，覆盖AWS、Azure和GCP资源模式（此处报告的评估以AWS为重点）。对150多个AWS Terraform工作区的评估表明，严重性过滤将警报量减少了73%，同时保留了94%的安全相关变更，提供了显著的改进。

    arXiv:2608.18173v1 Announce Type: new  Abstract: Infrastructure as Code (IaC) tools like Terraform have become the standard for declarative cloud resource management, yet configuration drift, where deployed infrastructure diverges from its declared state, remains a persistent operational and security challenge. Current detection approaches treat all changes equivalently, contributing to alert fatigue that causes operators to miss security-critical modifications. We propose a generalized severity taxonomy for infrastructure drift that classifies changes into four risk tiers based on resource type and attribute-level impact. We implement this taxonomy in tfdrift, an open-source classification framework with 60+ configurable rules covering AWS, Azure, and GCP resource patterns (evaluation reported here is AWS-focused). Evaluation across 150+ AWS Terraform workspaces demonstrates that severity filtering reduces alert volume by 73% while retaining 94% of security-relevant changes, offering 
    
[^20]: 对抗性审查：用于接地智能体代码审查的结构化分歧

    Adversarial Review: Structured Disagreement for Grounded Agentic Code Review

    [https://arxiv.org/abs/2608.18167](https://arxiv.org/abs/2608.18167)

    本文提出了一种名为对抗性审查（AR）的最小合作代码审查协议，通过引入批评者智能体进行结构化分歧审计，在仅使用三个智能体的情况下超越了五个智能体的基线性能，并揭示了朴素方法中的虚假共识问题。

    

    arXiv:2608.18167v1 公告类型：新 摘要：早期的多智能体LLM系统通常采用角色分离的团队，但在仓库级编码任务上，增加智能体数量会导致收益递减。最近的替代方案将智能体视为被动工具（子智能体），但这完全消除了智能体交互的好处。我们研究了子智能体范式是否能支持一种折中方案：在避免大型多智能体团队开销的同时，实现最小限度的智能体合作。我们引入了对抗性审查（AR），一种最小合作的代码审查协议，其中主编码智能体与一个审查者智能体和一个批评者智能体协作。审查者评估代码，而批评者通过结构化分歧审计审查结果，之后主智能体进行编辑。在LiveCodeBench上，AR在测试方法中实现了最高的通过率，仅使用三个智能体就超越了五个智能体的基线。在SWE-PRBench上，朴素的AR暴露了一种虚假共识失败模式，即智能体在没有充分依据的情况下达成一致。

    arXiv:2608.18167v1 Announce Type: new  Abstract: Early multi-agent LLM systems often used role-separated teams, yet scaling agent count yields diminishing returns on repository-level coding tasks. Recent alternatives treat agents as passive tools (subagents), yet this removes the benefits of agent interaction entirely. We study whether a subagent paradigm can support a middle ground: minimal agentic cooperation without the overhead of large multi-agent teams. We introduce Adversarial Review (AR), a minimal cooperative code-review protocol in which a main coding agent works with a reviewer and a critic agent. The reviewer evaluates code, while the critic audits the review through structured disagreement before the main agent edits. On LiveCodeBench, AR achieves the highest pass rate among tested methods, outperforming a five-agent baseline while using only three agents. On SWE-PRBench, naive AR exposes a false-consensus failure mode, where agents converge on agreement without sufficient
    
[^21]: MicroPython与CircuitPython：Python在物联网和机器人领域的悄然崛起

    MicroPython and CircuitPython: Pythons Quiet Takeover of IoT and Robotics

    [https://arxiv.org/abs/2608.18160](https://arxiv.org/abs/2608.18160)

    本文通过数据分析和基准测试，论证了MicroPython和CircuitPython正以降低开发门槛和推动教育普及的方式，在物联网和机器人领域实现对嵌入式系统的悄然接管。

    

    摘要：背景：Python已成为软件和数据科学领域的主导语言，但嵌入式系统由于性能和内存限制，仍与C/C++紧密相关。MicroPython和CircuitPython正通过将Python引入微控制器，降低物联网和机器人开发的门槛，从而改变这一现状。目的：本文考察这些平台是否正在实现对嵌入式系统的悄然接管，重点关注生态系统增长、实际应用、性能权衡、教育采用和未来前景。方法：采用混合方法设计，包括对GitHub、Stack Overflow和Google Trends数据的定量分析；从Hackster.io、Hackaday.io和Adafruit学习系统整理案例研究；并在ESP32和Raspberry Pi Pico上进行原始基准测试，比较MicroPython、CircuitPython和Arduino C++在GPIO、I2C、SPI、Wi-Fi和内存使用方面的表现。结果：指标显示持续增长，其中...

    arXiv:2608.18160v1 Announce Type: cross  Abstract: Background: Python has become the dominant language in software and data science, yet embedded systems have remained tied to C/C++ due to performance and memory constraints. MicroPython and CircuitPython are changing this by bringing Python to microcontrollers, lowering barriers for IoT and robotics development. Aim: This article examines whether these platforms are achieving a quiet takeover of embedded systems, focusing on ecosystem growth, practical applications, performance trade-offs, educational adoption, and prospects. Methods: A mixed-methods design was used, including quantitative analysis of GitHub, Stack Overflow, and Google Trends data; curation of case studies from Hackster.io, Hackaday.io, and the Adafruit Learning System; and original benchmarks on ESP32 and Raspberry Pi Pico comparing MicroPython, CircuitPython, and Arduino C++ across GPIO, I2C, SPI, Wi-Fi, and memory usage. Results: Metrics show sustained growth, with 
    
[^22]: RISC-V系统的FPGA生命周期管理

    FPGA Lifecycle Management for RISC-V Systems

    [https://arxiv.org/abs/2608.18156](https://arxiv.org/abs/2608.18156)

    该论文提出了一种基于Linux操作系统层的与主机无关的控制平面架构，使RISC-V处理器能作为FPGA生命周期管理主机，实现供应商中立且可扩展的位流部署。

    

    arXiv:2608.18156v1 公告类型：交叉 摘要：FPGA生命周期管理仍受限于专有工具链和主机架构，使得RISC-V缺乏可扩展位流部署的供应商中立模型。本文提出了一种与主机无关的控制平面架构，通过利用标准Linux功能将生命周期管理转移到操作系统层，从而将部署与特定ISA和供应商堆栈解耦。这使得具备Linux能力的RISC-V处理器能够作为异构FPGA系统中的控制主机。该架构在Zynq-7000 SoC上进行了原型验证，并可推广至RISC-V平台，为大规模FPGA管理提供了可移植的基础。

    arXiv:2608.18156v1 Announce Type: cross  Abstract: FPGA lifecycle management remains tied to proprietary toolchains and host architectures, leaving RISC-V without a vendor-neutral model for scalable bitstream deployment. A host-agnostic control-plane architecture is presented that shifts lifecycle management to the operating-system layer by leveraging standard Linux capabilities, thereby decoupling deployment from specific ISAs and vendor stacks. This enables Linux-capable RISC-V processors to serve as control hosts in heterogeneous FPGA systems. Prototyped on a Zynq-7000 SoC and generalizable to RISC-V platforms, the architecture provides a portable foundation for fleet-scale FPGA management.
    
[^23]: PowderLine：一个程序化的粉末衍射分析应用程序

    PowderLine: a programmatic powder diffraction analysis application

    [https://arxiv.org/abs/2608.17009](https://arxiv.org/abs/2608.17009)

    PowderLine是一个Python工具，通过声明式配方和版本化验证，将粉末衍射精修过程标准化为机器可读的程序化流程，从而支持高通量和自主实验。

    

    全谱拟合方法，如Rietveld精修，在从粉末衍射数据中提取详细的结构、化学和微观结构信息方面表现出色。获得可靠结果需要相当的专业知识和特定软件的知识，而大规模应用这些方法通常依赖于为每个应用编写的自定义脚本。高通量实验和自主自驱动实验室越来越多地利用粉末衍射分析以程序化方式进行，并返回结构化的、机器可读的结果。在这里，我们介绍了PowderLine，一个Python应用程序，它将完整的精修过程封装到一个声明式配方中，根据版本化模式验证该配方，并通过精修软件执行它以返回结构化结果。该精修配方是对Rietveld或单峰分析的全包含、机器可读和可写的描述。

    arXiv:2608.17009v1 Announce Type: cross  Abstract: Whole-pattern fitting methods, such as Rietveld refinement, excel at extracting detailed structural, chemical, and microstructural information from powder diffraction data. Obtaining reliable results requires both considerable expertise and software-specific knowledge, and applying these methods at scale typically relies on custom scripts written for each application. High-throughput experiments and autonomous self-driving laboratories increasingly utilize powder diffraction analysis to proceed programmatically and to return structured, machine-readable results. Here, we introduce PowderLine, a Python application that encapsulates a complete refinement into a single declarative recipe, validates that recipe against a versioned schema, and executes it through refinement software to return structured results. The refinement recipe is an all-inclusive, machine-readable and -writable description of either Rietveld or single peak analysis t
    
[^24]: 从文档到零日漏洞：基于大语言模型驱动的PDF阅读器中JavaScript引擎模糊测试

    From Documentation to Zero-day Vulnerabilities: LLM-Driven Fuzzing of JavaScript Engines in PDF Readers

    [https://arxiv.org/abs/2608.06641](https://arxiv.org/abs/2608.06641)

    提出PDFuzzer，利用大语言模型自动生成复杂API调用序列的PDF引擎模糊测试器，显著提升了对主流PDF阅读器中JavaScript引擎漏洞的检测能力。

    

    arXiv:2608.06641v2 公告类型：替换交叉 摘要：现有的PDF阅读器模糊测试器依赖于仅涉及单个API调用的简单测试用例，导致覆盖范围有限，可能遗漏需要API调用序列的漏洞。为解决这些限制，我们提出了PDFuzzer，一种新颖的PDF引擎模糊测试器，能够自动生成复杂且有意义的API调用序列。PDFuzzer首先使用大语言模型（LLM）从JavaScript API手册和执行轨迹中提取的规范构建上下文无关文法，并推断各个API调用之间的关系。基于这些文法和关系，PDFuzzer采用约束求解器生成具体的API调用序列用于模糊测试。我们的实验表明，PDFuzzer在三个主流PDF阅读器（Adobe Acrobat Reader、Foxit PDF Re）上显著优于最先进的PDF模糊测试器（TypeOracle、Favocado和Cooper）以及基于LLM的模糊测试器（Fuzz4All、朴素LLM）。

    arXiv:2608.06641v2 Announce Type: replace-cross  Abstract: Existing fuzzers for PDF readers rely on simple test cases that involve only individual API calls, leading to limited coverage and potentially missing vulnerabilities that require sequences of API calls. To address these limitations, we propose PDFuzzer, a novel PDF engine fuzzer that automatically generates complex and meaningful API call sequences. PDFuzzer first uses a Large Language Model (LLM) to construct context-free grammars and infer the relationships between individual API calls from specifications extracted from JavaScript API manuals and execution traces. Based on the grammars and relationships, PDFuzzer employs a constraint solver to generate concrete API call sequences for fuzzing. Our experiments show that PDFuzzer significantly outperforms state-of-the-art PDF fuzzers (TypeOracle, Favocado, and Cooper) and LLM-based fuzzers (Fuzz4All, naive LLM) on three mainstream PDF readers: Adobe Acrobat Reader, Foxit PDF Re
    
[^25]: MetaInfer：一个仅需知识即可生成LLM推理引擎的SKILL工具箱

    MetaInfer: A Knowledge Only LLM Inference Engine Generator SKILL Toolbox

    [https://arxiv.org/abs/2607.12875](https://arxiv.org/abs/2607.12875)

    本文提出MetaInfer，一种利用LLM作为编译器，通过多智能体协作和契约知识库，仅根据用户指定的运行时约束自动生成定制化推理框架的方法，以减少代码复杂性和性能开销。

    

    随着大语言模型技术的进步，模型家族、计算硬件、量化方案、并行化策略以及专用优化内核的空间持续扩大，这急剧增加了通用推理框架的代码复杂性和维护成本。传统软件工程通过多层抽象来支持多样化的应用场景，但这些抽象也增加了系统复杂性，并可能引入额外的性能开销。本文提出了metainfer，一种“LLM即编译器”的方法，用户只需指定推理程序的运行时约束。一个由LLM驱动的多智能体协作系统，结合契约知识库，自动生成一个满足这些约束的紧凑定制推理框架。我们从三个角度评估metainfer：源代码参考的效果、运行时行为以及...

    arXiv:2607.12875v2 Announce Type: replace-cross  Abstract: As LLM technology advances, the space of model families, compute hardware, quantization schemes, parallelization strategies, and specialized optimization kernels continues to expand, sharply increasing the code complexity and maintenance cost of general-purpose inference frameworks. Conventional software engineering uses multiple layers of abstraction to support diverse application scenarios, but these abstractions also increase system complexity and may introduce additional performance overhead. This paper presents metainfer, an 'LLM-as-Compiler' approach in which users specify only the runtime constraints of an inference program. An LLM-driven multi-agent collaboration system, coupled with a contract knowledge base, then automatically generates a compact customized inference framework that satisfies these constraints. We evaluate metainfer from three perspectives: the effect of source-code reference, the runtime behavior and 
    
[^26]: 从质量属性到实践：可解释性需求的指南与工作流

    From Quality Properties to Practice: A Guideline and Workflow for Explainability Requirements

    [https://arxiv.org/abs/2606.10882](https://arxiv.org/abs/2606.10882)

    本文提出并评估了一种指南驱动的可解释性需求制定工作流，通过提炼十个核心质量属性并操作化为工具，以改善需求制定的清晰度和一致性。

    

    可解释性在AI赋能的软件系统中日益被要求，以支持透明度、用户信任和合规性。然而，可解释性需求往往以临时方式编写，而无指导的大语言模型支持可能产生模糊、不一致或不完整的陈述。本文提出了一种顺序化、指南驱动的工作流，用于制定可解释性需求，并评估其基于工具的操作化实现。我们首先通过结构化文献综述和开发者访谈，筛选出候选的质量属性。然后，我们在一个面向从业者的在线调查（n=20）中对这些属性进行优先级排序，并提炼出一份包含十个核心属性的简明指南，附有可操作的定义说明。接下来，我们将该指南操作化于一个基于网页的工具中，该工具支持迭代工作流，包括起草、基于属性的检查和修订。我们通过两项互补研究评估了该工作流。在一项基于任务的研究中...

    arXiv:2606.10882v2 Announce Type: replace  Abstract: Explainability is increasingly required in AI-enabled software systems to support transparency, user trust, and compliance. Yet, explainability requirements are often written ad hoc, and unguided large language model support can yield vague, inconsistent, or incomplete statements. This paper presents a sequential, guideline-driven workflow for formulating explainability requirements and evaluates its tool-based operationalization. We first elicited candidate quality properties through a structured literature review and developer interviews. We then prioritized these properties in an online survey with practitioners (n = 20) and derived a concise guideline of ten core properties with actionable formulation instructions. Next, we operationalized the guideline in a web-based tool that supports an iterative workflow of drafting, property-based checks, and revision. We evaluated the workflow in two complementary studies. In a task-based s
    
[^27]: 隐私政策是否与日志匹配？安卓应用日志中隐私披露的实证研究

    Do Privacy Policies Match with the Logs? An Empirical Study of Privacy Disclosure in Android Application Logs

    [https://arxiv.org/abs/2604.18552](https://arxiv.org/abs/2604.18552)

    本研究通过对1000个安卓应用的实证分析发现，隐私政策与日志披露存在显著不一致，多数应用泄露未在政策中声明的敏感信息。

    

    隐私政策旨在告知用户软件系统如何收集和处理数据，但它们往往含糊不清或不完整。本文对隐私政策中与日志相关的陈述模式及其与安卓应用日志中观察到的隐私披露的一致性进行了实证研究。我们分析了跨多个类别的1，000个安卓应用，生成了86，836，964条日志记录。我们的发现表明，虽然大多数应用（88.0%）提供了隐私政策，但只有28.5%明确提及日志记录实践。在提及日志记录的应用中，大多数清楚地描述了记录了什么信息；然而，27.7%的日志相关陈述仍然过于简单或模糊，对实际数据收集提供的洞察有限。我们还观察到应用日志中普遍存在隐私泄露，67.6%的应用泄露了政策中未提及的敏感信息。令人担忧的是，仅有0.4%的应用完全符合政策与日志披露的一致性要求。

    arXiv:2604.18552v3 Announce Type: replace-cross  Abstract: Privacy policies are intended to inform users about how software systems collect and handle data, yet they often remain vague or incomplete. This paper presents an empirical study of patterns in log-related statements within privacy policies and their alignment with privacy disclosures observed in Android application logs. We analyzed 1,000 Android apps across multiple categories, generating 86,836,964 log entries. Our findings reveal that while most applications (88.0%) provide privacy policies, only 28.5% explicitly mention logging practices. Among those that reference logging, most clearly describe what information is logged; however, 27.7% of log-related statements remain overly simplistic or vague, offering limited insight into actual data collection. We further observed widespread privacy leakages in application logs, with 67.6% of apps leaking sensitive information not mentioned in their policies. Alarmingly, only 0.4% o
    
[^28]: 工程设计与系统工程中数据集可导航地图的框架与原型

    A Framework and Prototype for a Navigable Map of Datasets in Engineering Design and Systems Engineering

    [https://arxiv.org/abs/2603.15722](https://arxiv.org/abs/2603.15722)

    本文提出了一种系统框架和交互式工具原型，通过多维分类体系构建“EDSE数据集地图”，以解决工程设计与系统工程领域数据集碎片化和不可访问的问题，促进数据发现和复用。

    

    摘要：系统生命周期中数据的激增为工程设计与系统工程（EDSE）带来了重大机遇和挑战。虽然这种“数字主线”有潜力推动创新，但现有数据集的碎片化和不可访问性阻碍了方法验证、限制了可重复性，并减缓了研究进展。与受益于成熟基准生态系统的计算机视觉和自然语言处理等领域不同，工程设计研究往往依赖于小型、专有或临时数据集。本文通过提出一个“EDSE数据集地图”的系统框架来应对这一挑战。该框架基于一个多维分类体系，旨在按领域、生命周期阶段、数据类型和格式对工程数据集进行分类，从而实现分面发现。文中详细描述并演示了一个交互式发现工具的架构。

    arXiv:2603.15722v3 Announce Type: replace-cross  Abstract: The proliferation of data across the system lifecycle presents both a significant opportunity and a challenge for Engineering Design and Systems Engineering (EDSE). While this "digital thread" has the potential to drive innovation, the fragmented and inaccessible nature of existing datasets hinders method validation, limits reproducibility, and slows research progress. Unlike fields such as computer vision and natural language processing, which benefit from established benchmark ecosystems, engineering design research often relies on small, proprietary, or ad-hoc datasets. This paper addresses this challenge by proposing a systematic framework for a "Map of Datasets in EDSE." The framework is built upon a multi-dimensional taxonomy designed to classify engineering datasets by domain, lifecycle stage, data type, and format, enabling faceted discovery. An architecture for an interactive discovery tool is detailed and demonstrated
    
[^29]: 当代理失败时：LLM代理中缺陷的全面研究及其自动标注

    When Agents Fail: A Comprehensive Study of Bugs in LLM Agents with Automated Labeling

    [https://arxiv.org/abs/2601.15232](https://arxiv.org/abs/2601.15232)

    本研究首次系统性地分析了LLM代理软件中的缺陷类型、根本原因和影响，并探讨了自动化缺陷识别的可行性。

    

    大型语言模型（LLM）已经彻底改变了智能应用开发。虽然独立的LLM无法执行任何操作，但LLM代理通过集成工具解决了这一限制。然而，调试LLM代理既困难又昂贵，因为该领域仍处于早期阶段，社区发展尚不成熟。为了理解代理开发过程中遇到的缺陷，我们首次对基于LLM代理的软件中的缺陷类型、根本原因和影响进行了全面研究。我们收集并分析了来自Stack Overflow、GitHub和Hugging Face论坛的1,268个与缺陷相关的帖子和代码片段，重点关注使用七个广泛使用的LLM框架以及自定义实现构建的LLM代理。为了进行更深入的分析，我们还研究了缺陷发生的LLM代理组件，以及编程语言和框架。本研究还探讨了自动化缺陷识别的可行性。

    arXiv:2601.15232v3 Announce Type: replace  Abstract: Large Language Models (LLMs) have revolutionized intelligent application development. While standalone LLMs cannot perform any actions, LLM agents address the limitation by integrating tools. However, debugging LLM agents is difficult and costly, as the field is still in its early stages and the community is underdeveloped. To understand the bugs encountered during agent development, we present the first comprehensive study of bug types, root causes, and effects in LLM agent-based software. We collected and analyzed 1,268 bug-related posts and code snippets from Stack Overflow, GitHub, and Hugging Face forums, focused on LLM agents built with seven widely used LLM frameworks as well as custom implementations. For a deeper analysis, we have also studied the component of the LLM agent where the bug occurred, along with the programming language and framework. This study also investigates the feasibility of automating bug identification.
    
[^30]: 专业软件开发者不随波逐流，而是掌控：2025年AI代理在编码中的应用

    Professional Software Developers Don't Vibe, They Control: AI Agent Use for Coding in 2025

    [https://arxiv.org/abs/2512.14012](https://arxiv.org/abs/2512.14012)

    经验丰富的开发者将AI代理视为生产力工具，但坚持保留设计控制权，通过专业知识策略性地引导代理行为，以确保软件质量。

    

    arXiv:2512.14012v2 公告类型：替换-交叉 摘要：AI代理的兴起正在改变软件的构建方式。代理的承诺是开发者可以更快地编写代码，将多个任务委托给不同的代理，甚至仅通过自然语言编写完整的软件。然而，在实际中，代理在专业软件开发中扮演何种角色仍是个问题。本文调查了经验丰富的开发者如何在构建软件时使用代理，包括他们的动机、策略、任务适宜性和情感。通过现场观察（N=13）和定性调查（N=99），我们发现，虽然经验丰富的开发者重视代理作为生产力提升工具，但他们出于对基本软件质量属性的坚持，保留了自己在软件设计和实现中的主导权，并利用专业知识采用控制代理行为的策略。此外，经验丰富的开发者喜欢与代理合作，将其视为协作的源泉。

    arXiv:2512.14012v2 Announce Type: replace-cross  Abstract: The rise of AI agents is transforming how software can be built. The promise of agents is that developers might write code quicker, delegate multiple tasks to different agents, and even write a full piece of software purely out of natural language. In reality, what roles agents play in professional software development remains in question. This paper investigates how experienced developers use agents in building software, including their motivations, strategies, task suitability, and sentiments. Through field observations (N=13) and qualitative surveys (N=99), we find that while experienced developers value agents as a productivity boost, they retain their agency in software design and implementation out of insistence on fundamental software quality attributes, employing strategies for controlling agent behavior leveraging their expertise. In addition, experienced developers enjoy working with agents as source of collaboration 
    
[^31]: 面向可复现、低代码本地化的配置优先框架

    A Configuration-First Framework for Reproducible, Low-Code Localization

    [https://arxiv.org/abs/2510.25692](https://arxiv.org/abs/2510.25692)

    本文提出了一种配置优先的框架LOCALIZE，通过配置文件声明实验和自动化工作流，实现了低代码、可复现的机器学习实验管理，特别适用于无线定位研究。

    

    随着机器学习（ML）日益支撑关键应用，可信、可比较且可重复的实验结果变得更加重要。日常的工作流程应默认支持严格的实验规范和受控执行，同时在需要时允许高级实验操作。在实践中，研究人员仍需组合配置、执行、版本控制和评估等工具，并在其应用领域内重复常见的实现工作。在本文中，我们提出了一种面向特定应用机器学习实验框架的配置优先设计，并将其实现为用于无线定位研究的LOCALIZE系统。实验以人类可读的配置文件声明，工作流编排器执行具有明确输入和输出的隔离阶段，并对代码、数据、配置、环境规范和生成产物进行版本管理。LOCALIZE提供了预配置的...

    arXiv:2510.25692v4 Announce Type: replace-cross  Abstract: As machine learning (ML) increasingly underpins critical applications, credible, comparable, and repeatable experimental results become more important. Everyday workflows should make rigorous experiment specification and controlled execution the default while allowing advanced experimentation when required. In practice, researchers still have to combine tools for configuration, execution, versioning, and evaluation, and repeat common implementation work within their application domain. In this paper, we present a configuration-first design for application-specific ML experimentation frameworks and implement it as LOCALIZE for radio-localization research. Experiments are declared in human-readable configuration files, a workflow orchestrator executes isolated stages with explicit inputs and outputs, and code, data, configurations, environment specifications, and generated artifacts are versioned. LOCALIZE provides preconfigured 
    
[^32]: 迈向包容性人工智能驱动的开发：探索代码生成工具交互中的性别差异

    Toward Inclusive AI-Driven Development: Exploring Gender Differences in Code Generation Tool Interactions

    [https://arxiv.org/abs/2507.14770](https://arxiv.org/abs/2507.14770)

    本研究通过对照实验发现，性别差异显著影响开发者与代码生成工具的交互效果和认知负荷，强调AI开发工具需更具包容性设计。

    

    arXiv:2507.14770v2 公告类型：替换 摘要：对代码生成工具（CGTs）如Claude Code和GitHub Copilot的日益依赖正在重塑编程工作流程，并引发了关于人机协作中公平性和包容性的关键问题。尽管CGTs提供了潜在的生产力提升，但其在不同用户群体中的有效性尚未得到充分研究。我们假设开发者与CGTs的交互因性别而异，影响任务结果和认知负荷，因为先前研究表明性别差异可能影响技术使用和认知处理。本研究采用混合主体设计，共39名参与者，按性别均分以实现平衡设计。参与者使用两种不同处理方式完成两项中高难度编程任务：仅使用CGT辅助和仅使用互联网访问。任务顺序和条件经过平衡以减轻顺序效应。我们收集...

    arXiv:2507.14770v2 Announce Type: replace  Abstract: The increasing reliance on Code Generation Tools (CGTs), such as Claude Code and GitHub Copilot, is revamping programming workflows and raising critical questions about fairness and inclusivity in human-AI collaboration. While CGTs offer potential productivity enhancements, their effectiveness across diverse user groups have not been sufficiently investigated. We hypothesized that developers' interactions with CGTs vary based on gender, influencing task outcomes and cognitive load, as prior research suggests that gender differences can affect technology use and cognitive processing. This study employed a mixed-subjects design with 39 participants, evenly divided by gender for a counterbalanced design. Participants completed two programming tasks of medium to high difficulty using two distinct treatments: only CGT assistance and only internet access. Task orders and conditions were counterbalanced to mitigate order effects. We collect
    
[^33]: 从提示到扰动：针对音频大语言模型的基于语音的越狱攻击自适应框架

    `From Prompt to Perturbation': An Adaptive Framework for Voice-Based Jailbreaks on Audio LLMs

    [https://arxiv.org/abs/2502.00735](https://arxiv.org/abs/2502.00735)

    本文提出了一种自适应越狱攻击框架，能在统一设置下系统评估级联流水线和端到端音频大语言模型，覆盖更广泛的音频攻击空间。

    

    随着大语言模型（LLMs）越来越多地集成到基于音频的应用中，人们对其易受音频对抗攻击的脆弱性日益担忧。这些系统通常遵循两种架构范式：级联流水线（其中自动语音识别将音频输入转换为文本，再交由LLM处理）和端到端的大音频语言模型（LALMs，直接解释原始音频信号）。除了架构差异外，级联流水线主要容易受到通过语音传递的文本级越狱策略的攻击，而端到端LALMs则引入了额外的声学语义攻击向量。然而，现有研究往往聚焦于单一范式，对更广泛的音频攻击空间覆盖有限。为弥补这一差距，我们提出了一个自适应越狱攻击框架，用于在统一设置下对级联流水线和LALMs进行系统评估。

    arXiv:2502.00735v4 Announce Type: replace-cross  Abstract: As large language models (LLMs) are increasingly integrated into audio-based applications, growing concerns have emerged regarding their vulnerability to audio-based adversarial attacks. These systems typically follow two architectural paradigms: cascaded pipelines, where automatic speech recognition converts audio inputs into text before LLM processing, and end-to-end large audio-language models (LALMs), which directly interpret raw audio signals. Beyond architectural differences, cascaded pipelines are primarily vulnerable to text-level jailbreak strategies delivered through speech, whereas end-to-end LALMs introduce additional acoustic-semantic attack vectors. However, existing studies often focus on a single paradigm and provide limited coverage of the broader audio attack space. To bridge this gap, we propose an adaptive jailbreak attack framework for systematic evaluation of both cascaded pipelines and LALMs under a unifi
    

