# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Polyglot's Dilemma: Conformance Testing a Dozen Specs in as Many Languages](https://arxiv.org/abs/2608.18039) | 本文介绍了MongoDB通过统一YAML测试格式跨多种语言客户端库实现一致性测试，显著减少非符合性缺陷，并总结了声明式测试设计与架构演化的经验。 |
| [^2] | [What Does It Mean and Why Should I Bother? Motivating Students to Write Better Commit Messages](https://arxiv.org/abs/2608.17993) | 本文通过案例研究证实了学生提交信息存在沟通质量问题，并提出了一款教育游戏WDYM，有效提升学生的提交信息撰写意识和反思能力。 |
| [^3] | [Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection](https://arxiv.org/abs/2608.17965) | 本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。 |
| [^4] | [Reshaping the SDLC for Data- and AI-Centric Systems](https://arxiv.org/abs/2608.17824) | 本文通过整合DataOps、MLOps和LLMOps实践，系统性地重塑了数据与AI中心系统的软件开发生命周期，提出了一个覆盖需求、架构、开发、测试、部署、监控和治理的转型框架。 |
| [^5] | [SpecTrum: Specification-Guided Differential Fuzzing for Ethereum Consensus Clients](https://arxiv.org/abs/2608.17738) | SpecTrum通过机械化规范显式化以太坊共识的有效性条件，并引入前提覆盖率指标，以系统性指导差分模糊测试，增强共识客户端的测试充分性。 |
| [^6] | [What Aggregate Scores Miss: Measuring Item-Level Regressions in Commercial LLM API Migrations](https://arxiv.org/abs/2608.17719) | 该论文揭示了商业LLM API迁移中聚合基准分数掩盖了项目级改进与回归共存的现象，表明依赖单一分数可能误导迁移决策。 |
| [^7] | [GADR: Gathering Architecture Decision Records from Meeting Transcriptions](https://arxiv.org/abs/2608.17694) | GADR提出了一种多智能体自纠正工作流程，能从嘈杂的原始会议转录中提取架构决策并生成Nygard格式的ADR草稿，在稳定性和结构上优于现有基线。 |
| [^8] | [Benchmarking Automated Security Patch Backporting: How Far Are We?](https://arxiv.org/abs/2608.17671) | 本文提出了一个跨版本、跨分支和跨代码库的安全补丁回溯移植基准，揭示了对齐评估下现有工具性能显著下降，尤其影响某些工具。 |
| [^9] | [Unified Message Model for Heterogeneous Serial Data Exchange Protocols](https://arxiv.org/abs/2608.17642) | 本文提出了一种与协议无关的统一消息模型，通过正式定义数据类型、原子容器和消息结构，为异构串行通信协议提供了机器可处理的确定性描述基础，以支持嵌入式系统开发中的自动化工具链。 |
| [^10] | [TRUSS: Towards Task-Reliable and User-Safe Automated Agent Skill Generation](https://arxiv.org/abs/2608.17588) | TRUSS是一个证据引导的框架，通过静态安全属性检查和可控执行环境中的影子智能体验证，生成功能有效且安全可靠的智能体技能。 |
| [^11] | [REST API Testing with Verified LLM-Inferred Dependencies and Response-Driven Refinement](https://arxiv.org/abs/2608.17546) | 本文提出APIPilot框架，通过执行验证LLM推断的API依赖关系，避免虚假依赖并提高REST API测试的可行性。 |
| [^12] | [Agent Lightning v1.0: Towards Harnessed Agentic RL](https://arxiv.org/abs/2608.17528) | 本文介绍了Agent Lightning v1.0，提出了一种“受控代理强化学习”范式，其中部署时框架直接参与模型后训练，并解决了由此带来的重分词、样本合并等训练稳定性挑战。 |
| [^13] | [Beyond FLOPs: Energy-Aware Knowledge Distillation for Sustainable LLMs on Code-Related Task](https://arxiv.org/abs/2608.17515) | 本文提出并验证了能量感知知识蒸馏方法（Morph），用于提升软件工程任务中大语言模型的可持续性，同时质疑了FLOPs作为能源效率指标的可靠性。 |
| [^14] | [Agentic Porting, Construction and Initial Verification and Validation of Libraries within the Open Source Unified TRAnsient Multi-Phase Advanced Reactor simulation Kit (Outram Park) Part I: Thermal Hydraulics](https://arxiv.org/abs/2608.17504) | 本文提出了一种人机协同的智能体式库移植方法，将OpenFOAM库移植到Rust语言中，用于构建Outram Park模拟工具包，并初步验证了其热工水力模块的准确性。 |
| [^15] | [COMMITGUARD: Differential Slice Fuzzing for Commit-Induced Bug Detection](https://arxiv.org/abs/2608.17401) | COMMITGUARD提出了一种基于提交感知的差分切片模糊测试方法，利用提交前版本作为基线来有效检测提交引发的内存安全缺陷。 |
| [^16] | [SNIPTEST: Fuzzing Multi-Level Code Slices for Validating Vulnerabilities](https://arxiv.org/abs/2608.17396) | SNIPTEST通过生成并模糊测试围绕静态分析警告的多层级编译代码切片，实现了高效且可扩展的漏洞验证，避免了全项目模糊测试的计算开销。 |
| [^17] | [NeuroAbs: A Neuro-Symbolic RTL Abstraction Framework for Property Checking Acceleration](https://arxiv.org/abs/2608.17304) | NeuroAbs提出了一种结合大语言模型和抽象语法树符号表示的神经符号RTL抽象框架，通过LLM辅助信号识别和SMT正确性检查，实现了自动化且灵活的RTL抽象，显著加速了属性检查过程。 |
| [^18] | [Oracles That Cannot Fail: Anchoring and the Expectation That Moves With the Fault](https://arxiv.org/abs/2608.17214) | 该论文引入“预言机锚定”概念，指出测试预言机从被测系统获取期望值时，故障会因比较抵消而无法被检测，并通过实证测量展示了这一缺陷在现实系统中的影响。 |
| [^19] | [Graphectory Viewer: A Tool for Process-Centric Analysis of Agentic Software Trajectories](https://arxiv.org/abs/2608.17195) | 本文介绍了一个名为Graphectory Viewer的交互式Web工具，它通过将智能体轨迹转化为阶段感知图，支持过程中心的分析，从而帮助研究者深入理解行为模式并比较不同执行结果。 |
| [^20] | [Grounding AI Agents in Contracts: An Empirical Evaluation of Spec-Driven Test Generation](https://arxiv.org/abs/2608.17177) | 提出规格驱动测试生成方法，通过让LLM代理先显式推理和记录代码合同（前置/后置条件及未定义行为）作为认知脚手架，显著提升了生产环境中的缺陷检测率和分支覆盖率。 |
| [^21] | [A Multi-Surface Consistency Audit of Software Citation Metadata](https://arxiv.org/abs/2608.17159) | 本文首次系统审计了117个开源研究软件项目在不同元数据表面（如引文文件、DOI记录、包注册表等）之间的一致性，揭示了这些表面之间常存在分歧，可能影响学术贡献的认可和溯源。 |
| [^22] | [LadderTeam: Dual-Agent Laddering Elicitation Framework](https://arxiv.org/abs/2608.17029) | 本文提出了LadderTeam框架，通过双智能体LLM架构自动化UX线框图访谈，克服了传统阶梯式访谈的手动成本和可扩展性限制。 |
| [^23] | [ORCA: Observability-Grounded Program Repair for Microservice Incidents](https://arxiv.org/abs/2608.17018) | ORCA通过将遥测数据差异转化为故障签名，并利用修复图代理和探索代理生成补丁，配合遥测验证器，显著提升了微服务事件修复的成本效益。 |
| [^24] | [PowderLine: a programmatic powder diffraction analysis application](https://arxiv.org/abs/2608.17009) | PowderLine是一个Python工具，通过声明式配方和版本化验证，将粉末衍射精修过程标准化为机器可读的程序化流程，从而支持高通量和自主实验。 |
| [^25] | [Experiential Learning of Runtime Monitoring Using Pachinko](https://arxiv.org/abs/2608.16898) | 本文介绍了一种通过构建交互式弹珠机游戏来教授运行时监控的课堂作业，利用双核ESP32和RTLola规范，强调形式化方法的实践教学和跨课程的可移植性。 |
| [^26] | [Distribird: Literature-Informed Prior Distribution Design for Bayesian Model Calibration](https://arxiv.org/abs/2608.11210) | Distribird是一个自动化代理工具，通过文献搜索、加权提取和AIC模型选择，为贝叶斯模型校准生成信息丰富的先验分布，并在无文献时提供合理的非信息性替代方案。 |
| [^27] | [From Adoption to Deployment: A Qualitative Study on AI Integration in Software Development Practice](https://arxiv.org/abs/2607.16660) | 本研究通过22名从业者的半结构化访谈，揭示了软件实践中AI组件选择和集成时安全考虑不足的现状，并提出了改进决策流程的见解。 |
| [^28] | [ThinkLog: Leveraging Reasoning for Log Statement Generation](https://arxiv.org/abs/2607.11615) | ThinkLog通过将推理融入提示作为少量示例，提升了基于LLM的端到端日志语句生成的准确性，在9,619个Java方法上进行了验证。 |
| [^29] | [FVSpec: Real-World Property-Based Tests as Lean Challenges](https://arxiv.org/abs/2606.01008) | 本文提出FVSpec基准，通过自动将真实Python项目中的基于属性测试转译为Lean 4规范，并采用三智能体LLM流水线，为AI在正式软件验证任务上的能力评估提供了大规模挑战和基线。 |
| [^30] | [Post-Deployment Accountability in AI Governance: A Cross-Regulatory Empirical Analysis of AI Incidents](https://arxiv.org/abs/2605.16281) | 本研究通过实证分析发现，AI部署后问责存在显著缺口，内部监控机制显著提升合规率，而外部检测事件合规率极低。 |
| [^31] | [Epistemological Debt: Cognitive Atrophy and Systemic Collapse in AI-Dependent Software Engineering](https://arxiv.org/abs/2604.26855) | 本文提出“认识论债务”概念，指出AI依赖的软件开发会导致认知萎缩和系统性崩溃，并主张通过人机协同教学标准来维持工程韧性。 |
| [^32] | [Counterexample Classification for Signal Temporal Logic Specifications](https://arxiv.org/abs/2601.13743) | 本文提出了一种针对信号时态逻辑规范的反例分类方法，旨在减少工程师在调试中因重复反例检查所花费的精力。 |
| [^33] | [D-LiFT: Improving LLM-based Decompiler Backend via Code Quality-driven Fine-tuning](https://arxiv.org/abs/2506.10125) | 本文提出D-LIFT，一种通过代码质量驱动的强化学习微调LLM的反编译器增强方法，在保持代码准确性的同时提高可读性，并引入D-Score评估系统。 |
| [^34] | [RBCTest: Leveraging LLMs to Mine and Verify Oracles of API Response Bodies for RESTful API Testing](https://arxiv.org/abs/2504.17287) | 本文提出RBCTest，一种利用大型语言模型从API规范中静态挖掘响应体约束并生成测试用例的新方法，通过观察-确认方案有效降低幻觉，提高了预言机挖掘的准确性。 |
| [^35] | [LSem2Vec: A Simple yet Effective Two-Stage Approach for Source Code Embedding](https://arxiv.org/abs/2409.14644) | LSem2Vec通过两阶段方法（LLM提取语义+句子嵌入生成向量）实现无需监督训练或微调的源代码嵌入，有效处理错误信息并提升性能。 |

# 详细

[^1]: 多语言者的困境：以同样多的语言对十几种规范进行一致性测试

    The Polyglot's Dilemma: Conformance Testing a Dozen Specs in as Many Languages

    [https://arxiv.org/abs/2608.18039](https://arxiv.org/abs/2608.18039)

    本文介绍了MongoDB通过统一YAML测试格式跨多种语言客户端库实现一致性测试，显著减少非符合性缺陷，并总结了声明式测试设计与架构演化的经验。

    

    摘要：arXiv:2608.18039v1 公告类型：新 摘要：MongoDB维护着十二种编程语言的客户端库，被数万个组织和数百万开发者使用。大多数是原生实现，而非围绕共享核心的封装。确保这些包含数百万行代码的库之间行为一致，既困难又至关重要。十一年来，我们开发了一种基于规范的测试方法：测试用YAML编写一次，并由每个库的语言特定解释器执行。我们描述了从多种临时格式向统一测试格式的演变，这使我们删除了超过22,000行测试代码。在采用YAML测试的驱动中，不符合规范的缺陷率下降了高达86%（尽管结果因库而异）。我们报告了关于声明式测试设计、测试架构、模式演化和统一化局限性的经验教训。

    arXiv:2608.18039v1 Announce Type: new  Abstract: MongoDB maintains client libraries in a dozen programming languages, used by tens of thousands of organizations and millions of developers. Most are implemented natively rather than as wrappers around a shared core. Ensuring consistent behavior across these libraries, comprising millions of lines of code, is hard but essential. Over eleven years, we developed a specification-based testing approach: tests are written once in YAML and executed by language-specific interpreters for each library. We describe the evolution from many ad-hoc formats to a Unified Test Format, which allowed us to delete over 22,000 lines of test code. The rate of nonconformance bugs fell up to 86% in drivers that adopted YAML tests (though results varied). We report lessons learned about declarative test design, test architecture, schema evolution, and the limits of unification.
    
[^2]: 这是什么意思，为什么我要费心？激励学生写出更好的提交信息

    What Does It Mean and Why Should I Bother? Motivating Students to Write Better Commit Messages

    [https://arxiv.org/abs/2608.17993](https://arxiv.org/abs/2608.17993)

    本文通过案例研究证实了学生提交信息存在沟通质量问题，并提出了一款教育游戏WDYM，有效提升学生的提交信息撰写意识和反思能力。

    

    本文报告了一项基于本地动机的混合方法案例研究，旨在解决软件工程教师的一个教学相关疑虑：学生撰写的提交信息经常未能发挥其预期的沟通作用。为实证检验这一疑虑，我们使用一个已建立的提交信息质量分类法的部分复制，分析了来自学生和工业案例研究项目的提交信息。结果证实，沟通和质量问题在这两种情境中均反复出现，从而证实了教师最初的担忧。基于这一发现，我们设计了“你是什么意思？”（WDYM），一种轻量级、基于角色的教育游戏，旨在在大学课程的约束下揭示并解决提交信息沟通断裂问题。对游戏观察和参与者调查的分析表明，WDYM在提高意识和促进反思方面是有效的。

    arXiv:2608.17993v1 Announce Type: new  Abstract: This paper reports on a locally motivated mixed-methods case study addressing a teaching-related suspicion held by software engineering instructors: that commit messages written by students frequently fail to serve their intended communicative role. To examine this suspicion empirically, we analyzed commit messages from student and industrial case-study projects using a partial replication of an established commit-message quality taxonomy. The results confirm that communication and quality issues occur recurrently in both contexts, substantiating the instructors' initial concern. Motivated by this finding, we devised What Do You Mean? (WDYM), a lightweight, role-based educational game intended to surface and address commit-message communication breakdowns within the constraints of university coursework. Analysis of gameplay observations and participant surveys shows that WDYM is effective in raising awareness and fostering reflection on 
    
[^3]: 过于自信而不安全：用于可靠日志异常检测的模型校准

    Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection

    [https://arxiv.org/abs/2608.17965](https://arxiv.org/abs/2608.17965)

    本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。

    

    在线日志异常检测对于维护大规模计算系统的可靠性至关重要。尽管基于语言模型的日志异常检测器取得了强大的检测性能，但其置信度估计仍校准不佳。我们表明，这些检测器经常对错误预测赋予过高的置信度，尤其是在严重类别不平衡下的异常日志中。此外，即使传统校准指标显示校准良好，错误预测的置信度仍持续偏高，这为运维监控系统造成了关键可靠性缺口。为解决此问题，我们提出了日志重建与距离（LoRD），一种轻量级的事后校准框架，用于可靠的日志异常检测。LoRD从正确分类的验证样本的潜在表示中学习预测路径特定的可靠性模型，并估计预测可靠性阈值。

    arXiv:2608.17965v1 Announce Type: cross  Abstract: Online log anomaly detection is critical for maintaining the reliability of large-scale computing systems. Although recent language model-based log anomaly detectors achieve strong detection performance, their confidence estimates remain poorly calibrated. We show that these detectors frequently assign excessive confidence to incorrect predictions, particularly for anomalous logs under severe class imbalance. Moreover, confidence on erroneous predictions remains persistently high even when conventional calibration metrics indicate good calibration, creating a critical reliability gap for operational monitoring systems. To address this issue, we propose Log Reconstruction and Distance (LoRD), a lightweight post-hoc calibration framework for reliable log anomaly detection. LoRD learns prediction-route-specific reliability models from latent representations of correctly classified validation samples and estimates prediction reliability th
    
[^4]: 重塑面向数据与AI中心系统的软件开发生命周期

    Reshaping the SDLC for Data- and AI-Centric Systems

    [https://arxiv.org/abs/2608.17824](https://arxiv.org/abs/2608.17824)

    本文通过整合DataOps、MLOps和LLMOps实践，系统性地重塑了数据与AI中心系统的软件开发生命周期，提出了一个覆盖需求、架构、开发、测试、部署、监控和治理的转型框架。

    

    传统的软件开发生命周期（SDLC）假设系统行为主要由源代码决定，从而允许通过以代码为中心的实践来指定、实现和验证正确性。然而，数据密集型与AI赋能的系统挑战了这一假设，因为其行为源于代码、数据和已学习模型的相互作用，并且当现实条件偏离训练数据时，性能可能会下降。本文探讨了如何通过DataOps、MLOps和LLMOps实践来整合数据工程与软件工程实践，从而重塑这些系统的SDLC。我们做出了四项贡献。首先，我们将软件工程、数据管理、机器学习系统和人机交互领域的文献综合为一个阶段结构化的生命周期转型描述，涵盖需求、架构、开发、测试、部署、监控、治理等环节。

    arXiv:2608.17824v1 Announce Type: new  Abstract: The traditional Software Development Lifecycle (SDLC) assumes that system behavior is determined primarily by source code, allowing correctness to be specified, implemented, and verified through code-centric practices. Data-intensive and AI-enabled systems challenge this assumption because their behavior emerges from the interaction of code, data, and learned models, while performance may degrade as real-world conditions drift from training data. This paper examines how integrating data engineering and software engineering practices, operationalized through DataOps, MLOps, and LLMOps, reshapes the SDLC for these systems. We make four contributions. First, we synthesize literature across software engineering, data management, machine learning systems, and human-centered computing into a phase-structured account of lifecycle transformation spanning requirements, architecture, development, testing, deployment, monitoring, governance, and or
    
[^5]: SpecTrum：面向以太坊共识客户端的规范引导差分模糊测试

    SpecTrum: Specification-Guided Differential Fuzzing for Ethereum Consensus Clients

    [https://arxiv.org/abs/2608.17738](https://arxiv.org/abs/2608.17738)

    SpecTrum通过机械化规范显式化以太坊共识的有效性条件，并引入前提覆盖率指标，以系统性指导差分模糊测试，增强共识客户端的测试充分性。

    

    以太坊的共识安全性依赖于独立的共识客户端实现，在每次状态转换上保持一致。当这些实现因错误而出现分歧时，网络可能分叉，最终性可能停滞，并可能引发严重攻击。为防止此类共识分歧，以太坊提供了一份Python参考实现（共识规范），作为规范，以及一套手工制作的官方测试套件（spectests）。然而，作为可执行实现，以太坊的规范通过运行时行为隐式定义有效性。因此，它缺乏系统性方法，以确保所有有效性条件得到彻底评估。我们提出了SpecTrum，一个通过三个阶段解决此问题的框架。首先，我们引入了Consensus-SpecTec，一种以太坊共识算法的机械化规范，使有效性条件以前提条件的形式显式化。其次，我们定义了前提覆盖率，这是一种度量标准，用于衡量这些条件被测试覆盖的程度。

    arXiv:2608.17738v1 Announce Type: new  Abstract: Ethereum's consensus safety relies on independent consensus client implementations agreeing on every state transition. When they diverge due to implementation errors, the network can fork, finality can stall, and severe attacks are possible. To prevent such consensus divergences, Ethereum provides a Python reference implementation (consensus-spec), which acts as a specification, and a hand-crafted official test suite (spectests). However, as an executable implementation, Ethereum's specification defines validity implicitly through runtime behavior. As a result, it lacks a systematic way to ensure that all validity conditions are thoroughly evaluated.   We present SpecTrum, a framework that addresses this problem in three stages. First, we introduce Consensus-SpecTec, a mechanized specification of the Ethereum consensus algorithm, which makes validity conditions explicit as if-premises. Second, we define premise coverage, a metric that me
    
[^6]: 聚合分数遗漏了什么：在商业LLM API迁移中衡量项目级回归

    What Aggregate Scores Miss: Measuring Item-Level Regressions in Commercial LLM API Migrations

    [https://arxiv.org/abs/2608.17719](https://arxiv.org/abs/2608.17719)

    该论文揭示了商业LLM API迁移中聚合基准分数掩盖了项目级改进与回归共存的现象，表明依赖单一分数可能误导迁移决策。

    

    摘要：背景：依赖商业大型语言模型API的软件系统，在供应商弃用旧模型时必须迁移到后继版本。迁移决策通常依赖于聚合基准分数，这些分数将异构的项目级行为压缩成一个单一的净数字。目标：我们衡量这种压缩所掩盖的内容。方法：在GPT-5.4到GPT-5.6 Sol产品序列中的三次成对升级中，我们查询了900个公共基准项目（研究生水平知识、奥林匹克数学、指令遵循），每个项目每个模型查询50次，将每个项目分类为可靠改进、可靠回归、实际等效或不确定，在错误发现率控制下和实践显著性阈值下，并将结果与标签置换零模型进行校准。结果：在所有九个迁移-基准组合中，可靠改进和可靠回归共存。聚合增益高达7.3的边界。

    arXiv:2608.17719v1 Announce Type: cross  Abstract: Context: Software systems that depend on commercial large language model APIs must migrate to successor versions when vendors deprecate older models. Migration decisions typically rely on aggregate benchmark scores, which compress heterogeneous item-level behaviour into a single net figure. Objective: We measure what that compression conceals. Method: On three pairwise upgrades in the GPT-5.4 to GPT-5.6 Sol product sequence, we query 900 public benchmark items (graduate-level knowledge, olympiad mathematics, instruction following) 50 times per item per model, classify each item as reliably improved, reliably regressed, practically equivalent, or inconclusive under false-discovery-rate control and a practical-significance threshold, and calibrate the results against a label-permutation null. Results: Across all nine migration-benchmark cells, reliable improvements and reliable regressions coexist. Edges with aggregate gains of up to 7.3
    
[^7]: GADR：从会议转录中收集架构决策记录

    GADR: Gathering Architecture Decision Records from Meeting Transcriptions

    [https://arxiv.org/abs/2608.17694](https://arxiv.org/abs/2608.17694)

    GADR提出了一种多智能体自纠正工作流程，能从嘈杂的原始会议转录中提取架构决策并生成Nygard格式的ADR草稿，在稳定性和结构上优于现有基线。

    

    arXiv:2608.17694v1 公告类型：交叉 摘要：现有的基于大型语言模型的架构决策记录（ADR）生成方法共享一个关键且基本未经验证的假设：输入已经具有合理的结构。然而在实践中，架构决策源自非正式、嘈杂的会议，其中选择是隐式的、零散的，并与离题对话交织在一起，这正是单次提示生成效果下降的条件。本文提出了GADR，一种多智能体、自纠正的工作流程，能够从原始会议转录中提取架构决策，并生成Nygard格式的ADR草稿。一项可行性研究包括五个真实项目会议转录、四位资深架构师的专家评审以及十五名学生的评估，初步证据表明，该智能体工作流程能够捕捉大多数专家识别的决策，并生成参与者认为清晰且有用的草稿，在稳定性和结构上优于零样本和少样本基线。

    arXiv:2608.17694v1 Announce Type: cross  Abstract: Existing LLM-based approaches to Architecture Decision Record (ADR) generation share a critical and largely unexamined assumption: that input is already reasonably structured. In practice, architectural decisions emerge from informal, noisy meetings where choices are implicit, fragmented, and entangled with off-topic dialogue, precisely the conditions under which single-pass prompting degrades. This paper presents GADR, a multi-agent, self-correcting workflow that extracts architectural decisions from raw meeting transcriptions and generates Nygard-formatted ADR drafts. A feasibility study comprising five real project meeting transcripts, expert review by four senior architects, and evaluation by fifteen students provides initial evidence that the agentic workflow captures most expert-identified decisions and produces drafts participants found clear and useful, outperforming zero-shot and few-shot baselines in stability and structural 
    
[^8]: 基准测试自动化安全补丁回溯移植：我们进展如何？

    Benchmarking Automated Security Patch Backporting: How Far Are We?

    [https://arxiv.org/abs/2608.17671](https://arxiv.org/abs/2608.17671)

    本文提出了一个跨版本、跨分支和跨代码库的安全补丁回溯移植基准，揭示了对齐评估下现有工具性能显著下降，尤其影响某些工具。

    

    自动化安全补丁回溯移植对于缓解N-day漏洞至关重要。近期工具在其各自数据集上报告的成功率超过80%。然而，这些评估往往局限于同质环境，例如单一代码库或特定项目版本。因此，这些工具在其原始目标场景之外的泛化能力仍不明确。我们提出了Porting Benchmark，这是一个包含1,234个安全补丁回溯移植案例的精选数据集，涵盖跨版本、跨分支和跨代码库场景，并配套一个通用评估框架。利用该基准，我们在对齐设置下评估了五种工具，涵盖程序分析、LLM提示和LLM代理。我们的结果显示，对齐评估改变了表面性能格局：PortGPT和TSBPort在复制数据集上仍相对强劲，而FixMorph和Mystique在跨场景下性能显著下降。

    arXiv:2608.17671v1 Announce Type: cross  Abstract: Automated security patch backporting is critical for mitigating N-day vulnerabilities. Recent tools report success rates above 80% on their respective datasets. However, these evaluations are often confined to homogeneous environments, such as one repository or specific project versions. Consequently, it remains unclear how well these tools generalize beyond their originally targeted scenarios. We present Porting Benchmark, a curated dataset of 1,234 security patch backporting cases spanning cross-version, cross-branch, and cross-repository scenarios, paired with a common evaluation framework. Using this benchmark, we evaluate five tools spanning program analysis, LLM prompting, and LLM agents under aligned settings. Our results show that aligned evaluation changes the apparent performance landscape: PortGPT and TSBPort remain comparatively strong on the Replication Dataset, while FixMorph and Mystique degrade substantially under the c
    
[^9]: 异构串行数据交换协议的统一消息模型

    Unified Message Model for Heterogeneous Serial Data Exchange Protocols

    [https://arxiv.org/abs/2608.17642](https://arxiv.org/abs/2608.17642)

    本文提出了一种与协议无关的统一消息模型，通过正式定义数据类型、原子容器和消息结构，为异构串行通信协议提供了机器可处理的确定性描述基础，以支持嵌入式系统开发中的自动化工具链。

    

    arXiv:2608.17642v1 公告类型：新  摘要：现代嵌入式系统正变得越来越复杂，通常集成大量异构设备，如控制器、传感器、执行器和辅助子系统。因此，其开发和集成涉及多种串行通信协议，从标准化解决方案到部分标准化及完全项目自定义格式。此类系统的高效开发日益依赖于自动化工具链，而这些工具链反过来需要清晰、统一且机器可处理的正式基础。本文提出了一种统一的、与协议无关的消息模型，用于显式且确定性地描述串行消息。该模型基于数据类型、原子消息元素（容器）和完整消息结构的正式定义。除模型本身外，本文还介绍了实际使用该模型的方法，包括用于表达结构化的可配置消息类型。

    arXiv:2608.17642v1 Announce Type: new  Abstract: Modern embedded systems are becoming increasingly complex and typically integrate numerous heterogeneous devices, such as controllers, sensors, actuators, and supporting subsystems. As a result, their development and integration involve a wide variety of serial communication protocols, ranging from standardized solutions to partially standardized and fully project-defined formats. Efficient development of such systems increasingly depends on automation toolchains, which in turn require a clear, unified, and machine-processable formal basis. This paper proposes a unified, protocol-agnostic message model for explicit and deterministic description of serial messages. The model is based on formal definition of data types, atomic message elements (containers), and complete message structure. In addition to the model itself, the paper introduces methods for practical work with it, including configurable message types for expressing structural 
    
[^10]: TRUSS：迈向任务可靠且用户安全的自动化智能体技能生成

    TRUSS: Towards Task-Reliable and User-Safe Automated Agent Skill Generation

    [https://arxiv.org/abs/2608.17588](https://arxiv.org/abs/2608.17588)

    TRUSS是一个证据引导的框架，通过静态安全属性检查和可控执行环境中的影子智能体验证，生成功能有效且安全可靠的智能体技能。

    

    arXiv:2608.17588v1 公告类型：新 摘要：智能体技能将可复用的自然语言流程与可执行资源打包在一起，使软件智能体无需模型适配即可获得特定任务能力。自动生成此类技能可提升任务性能，然而仅根据候选技能的产物或最终任务结果进行评估，仍无法解决装备该技能的智能体会执行哪些操作以及这些操作会产生哪些副作用的问题。我们提出TRUSS，一个基于证据引导的框架，用于生成功能有效且安全可靠的智能体技能。TRUSS首先对照来源和领域证据检查功能声明，同时在九种预定义安全属性下评估完整产物。通过此静态门控的候选者由影子智能体在可控执行环境中加载，其中代理工具将请求的操作暴露给策略执行，并将其结果记录为保留来源的执行轨迹。

    arXiv:2608.17588v1 Announce Type: new  Abstract: Agent Skills package reusable natural language procedures with executable resources, enabling software agents to acquire task specific capabilities without model adaptation. Automatically generating such Skills can improve task performance, yet evaluating a candidate solely from its artifact or final task outcome leaves unresolved which actions the equipped agent will perform and which side effects those actions will produce. We present TRUSS, an evidence guided framework for generating functionally effective and safety reliable Agent Skills. TRUSS first inspects functional claims against source and domain evidence while evaluating the complete artifact under nine predefined safety properties. Candidates admitted by this static gate are loaded by a shadow agent inside a Controllable Execution Environment, where brokered tools expose requested actions to policy enforcement and record their results as provenance preserving execution traces
    
[^11]: 基于验证的LLM推断依赖与响应驱动精化的REST API测试

    REST API Testing with Verified LLM-Inferred Dependencies and Response-Driven Refinement

    [https://arxiv.org/abs/2608.17546](https://arxiv.org/abs/2608.17546)

    本文提出APIPilot框架，通过执行验证LLM推断的API依赖关系，避免虚假依赖并提高REST API测试的可行性。

    

    测试RESTful API需要生成满足操作、参数和运行时创建资源之间依赖关系的API调用序列。近期的基于LLM的方法从OpenAPI规范中推断此类依赖并生成测试序列，但它们通常将LLM推断的关系视为正确而不进行基于执行的验证。这可能导致引入虚假依赖、遗漏可行的操作链，并产生不可行的测试。在本文中，我们提出了APIPilot，一个执行验证的REST API测试框架。APIPilot首先使用结构启发式方法和基于LLM的语义推理，从OpenAPI规范中推导候选的生产者-消费者依赖关系。然后，它将这些依赖视为假设，并在用于测试生成之前，通过具体的API执行对其进行验证。验证后的依赖被组织成一个依赖图，APIPilot从中构建覆盖感知的测试序列。

    arXiv:2608.17546v1 Announce Type: new  Abstract: Testing RESTful APIs requires generating sequences of API calls that satisfy dependencies among operations, parameters, and runtime-created resources. Recent LLM-based approaches infer such dependencies and generate test sequences from OpenAPI specifications, but they often treat LLM-inferred relationships as correct without execution-based validation. This can introduce spurious dependencies, miss feasible operation chains, and produce infeasible tests. In this paper, we propose APIPilot}, an execution-validated framework for REST API testing. APIPilot first derives candidate producer-consumer dependencies from OpenAPI specifications using structural heuristics and LLM-based semantic reasoning. It then treats these dependencies as hypotheses and validates them through concrete API executions before using them for test generation. The validated dependencies are organized into a dependency graph from which APIPilot constructs coverage-awa
    
[^12]: Agent Lightning v1.0：迈向受控的代理强化学习

    Agent Lightning v1.0: Towards Harnessed Agentic RL

    [https://arxiv.org/abs/2608.17528](https://arxiv.org/abs/2608.17528)

    本文介绍了Agent Lightning v1.0，提出了一种“受控代理强化学习”范式，其中部署时框架直接参与模型后训练，并解决了由此带来的重分词、样本合并等训练稳定性挑战。

    

    arXiv:2608.17528v1 公告类型：新 摘要：现代代理在管理工具、上下文和控制流的代理框架（harness）内运行，使得框架成为代理系统的关键组成部分。我们最初的Agent Lightning引入了一种分离式架构，通过LLM端点代理将任意代理连接到强化学习训练，这一方法后来被诸如verl Uni-Agent、AReaL 2.0、slime和Polar等框架采用。我们将这种范式称为“受控代理强化学习”（harnessed agentic RL），其中部署时的框架直接参与模型的后训练。受控代理强化学习与传统代理强化学习有根本不同：框架而非训练引擎拥有环境交互循环，而训练器仅观察LLM请求-响应对序列。这带来了在重新分词、样本合并、优势计算、损失归一化和后端调度方面的挑战，这些可能显著影响训练的稳定性和有效性。我们提出了...

    arXiv:2608.17528v1 Announce Type: new  Abstract: Modern agents operate inside agent harnesses that manage tools, context, and control flow, making the harness a critical part of the agent system. Our original Agent Lightning introduced a disaggregated architecture that connects arbitrary agents to RL training through an LLM endpoint proxy, an approach later adopted by frameworks such as verl Uni-Agent, AReaL 2.0, slime, and Polar. We refer to this paradigm as harnessed agentic RL, where the deploy-time harness directly participates in model post-training. Harnessed agentic RL differs fundamentally from traditional agentic RL: the harness, rather than the training engine, owns the environment interaction loop, while the trainer observes only sequences of LLM request-response pairs. This introduces challenges in retokenization, sample merging, advantage calculation, loss normalization, and backend scheduling, which can substantially affect training stability and effectiveness. We present
    
[^13]: 超越FLOPs：面向代码相关任务的可持续大语言模型的能量感知知识蒸馏

    Beyond FLOPs: Energy-Aware Knowledge Distillation for Sustainable LLMs on Code-Related Task

    [https://arxiv.org/abs/2608.17515](https://arxiv.org/abs/2608.17515)

    本文提出并验证了能量感知知识蒸馏方法（Morph），用于提升软件工程任务中大语言模型的可持续性，同时质疑了FLOPs作为能源效率指标的可靠性。

    

    摘要：背景：大语言模型（LLMs）正越来越多地应用于软件工程（SE）任务，在克隆检测、漏洞预测和代码摘要等问题上取得了高精度。然而，它们的高计算需求和能耗引发了可持续性担忧，并阻碍了其在消费级硬件和资源受限平台上的使用。在文献和工业界中，报告LLM计算成本的一种常见方式是使用网络前向传播所需的浮点运算次数（FLOPs）。目的：本文研究了能量感知知识蒸馏对软件工程的影响，旨在提高模型效率的同时保持性能，并确定FLOPs是否是可靠的能源感知指标。方法：我们使用Morph（一种基于多目标优化的蒸馏方法）进行受控实验，以实证检验...

    arXiv:2608.17515v1 Announce Type: cross  Abstract: Background: Large Language Models (LLMs) are increasingly being applied to Software Engineering (SE) tasks, achieving high accuracy across problems such as clone detection, vulnerability prediction, and code summarization. However, their high computational demands and energy consumption raise sustainability concerns and hinder their use on consumer hardware and resource-constrained platforms. A common way to report the computational cost of an LLM in the literature and industry is to use the number of Floating Point Operations (FLOPs) required to perform a pass over the network. Aims: This paper investigates the implications of energy-aware knowledge distillation for SE, aiming to improve model efficiency while maintaining performance and to determine whether FLOPs is a reliable energy-aware metric. Method: We conduct a controlled experiment using Morph, a Many-Objective Optimization-based distillation methodology, to empirically exami
    
[^14]: 开源统一瞬态多相先进反应堆模拟工具包（Outram Park）中的智能体式移植、构建及初步验证与确认——第一部分：热工水力

    Agentic Porting, Construction and Initial Verification and Validation of Libraries within the Open Source Unified TRAnsient Multi-Phase Advanced Reactor simulation Kit (Outram Park) Part I: Thermal Hydraulics

    [https://arxiv.org/abs/2608.17504](https://arxiv.org/abs/2608.17504)

    本文提出了一种人机协同的智能体式库移植方法，将OpenFOAM库移植到Rust语言中，用于构建Outram Park模拟工具包，并初步验证了其热工水力模块的准确性。

    

    arXiv:2608.17504v1 公告类型：交叉 摘要：在开源统一瞬态多相先进反应堆模拟工具包（Outram Park）中，采用人机协同的智能体式方法，将多个开源库移植到Rust语言，用于构建模块。通过这种新方法，验证和确认工作侧重于人类专业知识，而非代码生成，这已成为开发可靠模拟代码的瓶颈。本研究展示了将OpenFOAM库移植到Outram-Foam Rust库的过程、其初步验证与确认（V&V）工作，以及随后在Outram Park中开发开源两相均匀平衡（HEM）阻塞流求解器的应用，这些求解器用于热工水力AI多相集成仿真系统（TAMPINES）库，如tampines-steam-tables。Outram-Foam的初步V&V结果显示，空腔和Sod激波管案例与文献值吻合良好。

    arXiv:2608.17504v1 Announce Type: cross  Abstract: Agentic porting of multiple open-source libraries into Rust, with human in the loop, has been performed for construction of modules within the Open-source Unified TRAnsient Multi-Phase Advanced Reactor simulation Kit (Outram Park). With this new methodology, verification and validation with human expertise, rather than code generation has become the bottleneck in developing reliable simulation codes. In this work, we present the porting of OpenFOAM libraries into the Outram-Foam Rust libraries, their preliminary verification and validation (V\&V) efforts, and their subsequent use in the development of open-source two-phase homogeneous-equilibrium (HEM) choked-flow solvers for the Thermo-hydraulic AI Multi-Phase INtegrated Emulator System (TAMPINES) libraries within Outram Park such as tampines-steam-tables. Preliminary V\&V efforts of Outram-Foam show that the cavity and Sod shock tube cases agree reasonably well with literature values
    
[^15]: COMMITGUARD：针对提交引发的缺陷检测的差分切片模糊测试

    COMMITGUARD: Differential Slice Fuzzing for Commit-Induced Bug Detection

    [https://arxiv.org/abs/2608.17401](https://arxiv.org/abs/2608.17401)

    COMMITGUARD提出了一种基于提交感知的差分切片模糊测试方法，利用提交前版本作为基线来有效检测提交引发的内存安全缺陷。

    

    现代软件系统通过频繁的提交（commits）来演化，这些提交实现了错误修复、功能增强和安全补丁。尽管代码审查和测试被广泛用于检查这些更改，但它们对内存安全问题提供的保证往往有限。代码审查者可能会遗漏细微的边界、生命周期或初始化错误，而现有测试可能无法覆盖提交所影响的特定路径。模糊测试在暴露此类缺陷方面非常有效，但对每个提交应用模糊测试仍不切实际，因为全程序模糊测试成本高昂、需要合适的测试工具（harnesses），并且可能无法触及提交所更改的代码。在本文中，我们引入了COMMITGUARD，一种针对验证代码更改的提交感知差分切片模糊测试方法。COMMITGUARD的关键洞察在于，修改后函数在提交前的版本可以作为解释提交后发现的缺陷的行为基线。对于每个目标提交...

    arXiv:2608.17401v1 Announce Type: new  Abstract: Modern software systems evolve through frequent commits that implement bug fixes, features, and security patches. Although code review and testing are widely used to check these changes, they often provide limited assurance for memory-safety issues. Code reviewers may miss subtle boundary, lifetime, or initialization errors, while existing tests may not exercise the specific paths affected by a commit. Fuzzing is effective at exposing such bugs, but applying it to every commit remains impractical because whole-program fuzzing is expensive, requires suitable harnesses, and may still fail to reach the code changed by a commit.   In this paper, we introduce COMMITGUARD, a commit-aware differential slice-based fuzzing approach for verifying code changes. The key insight behind COMMITGUARD is that the pre-commit version of a modified function can serve as a behavioral baseline for interpreting bugs found after the commit. For each target comm
    
[^16]: SNIPTEST：对多层级代码切片进行模糊测试以验证漏洞

    SNIPTEST: Fuzzing Multi-Level Code Slices for Validating Vulnerabilities

    [https://arxiv.org/abs/2608.17396](https://arxiv.org/abs/2608.17396)

    SNIPTEST通过生成并模糊测试围绕静态分析警告的多层级编译代码切片，实现了高效且可扩展的漏洞验证，避免了全项目模糊测试的计算开销。

    

    现代软件系统日益复杂，静态分析工具通常用于通过发出警告来识别潜在易受攻击的代码。然而，这些警告通常需要人工检查以确认报告的问题是否真实，这使得过程耗时且容易出错。定向模糊测试已成为一种强大的自动化技术，用于验证这些警告。然而，针对每个警告将其应用于整个项目在计算上是不可行的，通常需要数天的执行才能仅获得代码覆盖率的增量改进。我们提出了SNIPTEST，一种基于执行的警告分流框架，它生成并模糊测试围绕静态分析警告的编译代码切片。SNIPTEST并非证明完整程序中的可利用性，而是提供关于警告在逐步扩展的切片执行上下文中的行为证据。它采用逐层切片的方法。

    arXiv:2608.17396v1 Announce Type: new  Abstract: Modern software systems are increasingly complex, and static analysis tools are commonly used to identify potentially vulnerable code by issuing warnings. However, these warnings often require manual inspection to confirm whether the reported issues are real, making the process time-consuming and error-prone. Directed fuzzing has emerged as a powerful automated technique to validate the warnings. However, applying it to the entire project in response to each warning is computationally infeasible, often requiring days of execution to achieve only incremental improvements in code coverage.   We present SNIPTEST, an execution-based warning triage framework that generates and fuzzes compiled code slices centered around static-analysis warnings. Rather than proving exploitability in the full program, SNIPTEST provides evidence about how a warning behaves under progressively expanded sliced execution contexts. It employs a layer-by-layer slici
    
[^17]: NeuroAbs：一种用于属性检查加速的神经符号RTL抽象框架

    NeuroAbs: A Neuro-Symbolic RTL Abstraction Framework for Property Checking Acceleration

    [https://arxiv.org/abs/2608.17304](https://arxiv.org/abs/2608.17304)

    NeuroAbs提出了一种结合大语言模型和抽象语法树符号表示的神经符号RTL抽象框架，通过LLM辅助信号识别和SMT正确性检查，实现了自动化且灵活的RTL抽象，显著加速了属性检查过程。

    

    形式验证是确保硬件设计功能正确性的关键技术。在属性检查的背景下，一个关键挑战是如何在面对日益复杂的RTL设计时，高效地证明用户指定的属性。为了解决这一挑战，通常采用抽象技术来降低系统复杂性并加速验证过程。然而，先前的RTL抽象方法要么需要大量手动工作，要么依赖于缺乏灵活性的基于规则的技术。本文介绍了NeuroAbs，一种用于RTL抽象的神经符号框架。NeuroAbs首先使用LLM辅助的RTL分析来识别适合抽象的信号。然后，它结合基于LLM的抽象与基于AST的符号RTL表示，以更好地使生成的抽象与预期的变换对齐。每个抽象的正确性使用可满足性模理论进行检查。

    arXiv:2608.17304v1 Announce Type: cross  Abstract: Formal verification is a crucial technique for ensuring the functional correctness of hardware designs. In the context of property checking, a key challenge is how to efficiently prove a user-specified property in the face of increasingly complex RTL designs. To address this challenge, abstraction techniques are often employed to reduce system complexity and accelerate the verification process. However, prior RTL abstraction methods either require significant manual effort or rely on rule-based techniques that lack flexibility. This paper introduces NeuroAbs, a neuro-symbolic framework for RTL abstraction. NeuroAbs first uses LLM-assisted RTL analysis to identify signals suitable for abstraction. It then combines LLM-based abstraction with an AST-based symbolic RTL representation to better align the generated abstraction with the intended transformation. The soundness of each abstraction is checked using satisfiability modulo theories 
    
[^18]: 不会失败的预言机：锚定与随故障移动的期望值

    Oracles That Cannot Fail: Anchoring and the Expectation That Moves With the Fault

    [https://arxiv.org/abs/2608.17214](https://arxiv.org/abs/2608.17214)

    该论文引入“预言机锚定”概念，指出测试预言机从被测系统获取期望值时，故障会因比较抵消而无法被检测，并通过实证测量展示了这一缺陷在现实系统中的影响。

    

    arXiv:2608.17214v1 公告类型：新 摘要：一个从其评判的系统中获取期望值的测试预言机不会失败。如果故障同时移动测量值和期望值，比较会完全抵消，任何生成的输入都无法揭示该故障。缺陷在于预言机本身，而非输入空间。我们称这种现象为预言机锚定。当期望值由突变代码外部固定的值组成时，它是规范锚定的；当它直接或间接地从该代码流出时，则是状态锚定的。期望值形式在测试异味文献中被提及，但我们在检索到的任何研究中都未对其进行测量。我们进一步命名了这种值到达裁决结果的三种渠道，将谓词限制为从突变目标流出的值，并进行了测量。研究对象是一个已部署的空中交通管制模拟器，包含12个无模型属性套件。在4个模块和366个突变体中，与手写测试相比，这些增加了3个突变体的检测，同时剩余6到3个。

    arXiv:2608.17214v1 Announce Type: new  Abstract: A test oracle that obtains its expected value from the system it is judging cannot fail. If a fault moves measurement and expectation together the comparison cancels exactly, and no generated input will reveal it. The defect is in the oracle and not in the input space. We call this oracle anchoring. An expectation is specification-anchored when composed from values fixed outside the code under mutation, and state-anchored when it flows, directly or transitively, from that code. The expected-value form is named in the test-smell literature but not measured in any study we retrieved. We name three further channels by which such a value reaches a verdict, restrict the predicate to values flowing from the mutate target, and measure it. The subject is a deployed air traffic control simulator with 12 model-free property suites. Across 4 modules and 366 mutants these add 3 mutants of detection over the hand-written tests, while remaining 6 to 3
    
[^19]: Graphectory Viewer：面向智能体软件轨迹过程中心分析的工具

    Graphectory Viewer: A Tool for Process-Centric Analysis of Agentic Software Trajectories

    [https://arxiv.org/abs/2608.17195](https://arxiv.org/abs/2608.17195)

    本文介绍了一个名为Graphectory Viewer的交互式Web工具，它通过将智能体轨迹转化为阶段感知图，支持过程中心的分析，从而帮助研究者深入理解行为模式并比较不同执行结果。

    

    我们介绍了Graphectory Viewer，一个基于Web的交互式工具，用于对软件智能体轨迹进行以过程为中心的分析。该工具建立在我们先前工作中引入的Graphectory表示之上，将异构的原始轨迹转换为阶段感知的图，这些图连接了低层执行细节与高层行为结构。该工具支持来自多个智能体框架的轨迹，并提供交互式图构建；对思想、动作和观察的节点级检查；对大型轨迹集合的搜索和过滤；以及问题解决阶段转换的Sankey式摘要。这些能力使研究人员和实践者能够检查单个执行过程，识别重复出现的行为模式，比较成功和失败的运行，并分析超越最终任务结果的大型轨迹语料库。为支持可复现性和进一步研究，我们发布了Gra...

    arXiv:2608.17195v1 Announce Type: cross  Abstract: We present Graphectory Viewer, a web-based tool for interactive, process-centric analysis of software-agent trajectories. Building on the Graphectory representation introduced in our previous work, Graphectory Viewer transforms heterogeneous raw trajectories into phase-aware graphs that connect low-level execution details with higher-level behavioral structures. The tool supports trajectories from multiple agent frameworks and provides interactive graph construction; node-level inspection of thoughts, actions, and observations; search and filtering over large trajectory collections; and Sankey-style summaries of problem-solving phase transitions. These capabilities enable researchers and practitioners to inspect individual executions, identify recurring behavioral patterns, compare successful and failed runs, and analyze large trajectory corpora beyond final task outcomes. To support reproducibility and further research, we release Gra
    
[^20]: 将AI代理锚定在合同中：规格驱动测试生成的实证评估

    Grounding AI Agents in Contracts: An Empirical Evaluation of Spec-Driven Test Generation

    [https://arxiv.org/abs/2608.17177](https://arxiv.org/abs/2608.17177)

    提出规格驱动测试生成方法，通过让LLM代理先显式推理和记录代码合同（前置/后置条件及未定义行为）作为认知脚手架，显著提升了生产环境中的缺陷检测率和分支覆盖率。

    

    arXiv:2608.17177v1 公告类型：新 摘要：基于LLM的代理越来越多地用于编码任务，在这些任务中，它们已超越许多经典方法，并扩展到仓库级任务，如测试生成。然而，当直接提示生成测试时，这些代理可能无法推理代码及其底层合同，从而遗漏影响测试质量的边缘案例和行为边界。为解决这一限制，我们提出了规格驱动测试生成方法，即指示代理首先推理并显式记录代码的前置条件、后置条件和未定义行为。这种中间半形式化规格作为认知脚手架，指导后续的测试生成。我们在Google生产缺陷上的评估显示，与直接提示相比，规格驱动代理在缺陷检测率上提高了9.8个百分点（p = 0.0352），在分支覆盖率上提高了2.5个百分点（p = 0.0034）。

    arXiv:2608.17177v1 Announce Type: new  Abstract: LLM-based agents are increasingly used for coding tasks, where they have outperformed many classical approaches and scaled to repository-level tasks, such as test generation. However, when directly prompted to generate tests, these agents can fail to reason about the code and its underlying contracts, thereby missing edge cases and behavioral boundaries that affect test quality. To address this limitation, we propose Spec-Driven Test Generation, where we instruct an agent to first reason about -- and explicitly document -- code pre-conditions, post-conditions, and undefined behaviors. This intermediate semi-formal specification acts as a cognitive scaffold to guide subsequent test generation. Our evaluation on production bugs from Google shows that the spec-driven agent can deliver a 9.8 percentage points ($p = 0.0352$) improvement in bug detection rate and a 2.5 percentage point ($p = 0.0034$) improvement in branch coverage, compared to
    
[^21]: 软件引文元数据的多表面一致性审计

    A Multi-Surface Consistency Audit of Software Citation Metadata

    [https://arxiv.org/abs/2608.17159](https://arxiv.org/abs/2608.17159)

    本文首次系统审计了117个开源研究软件项目在不同元数据表面（如引文文件、DOI记录、包注册表等）之间的一致性，揭示了这些表面之间常存在分歧，可能影响学术贡献的认可和溯源。

    

    arXiv:2608.17159v1 公告类型：新 摘要：研究软件项目在多个地方同时描述自身：仓库中的引文文件、档案存储、DOI注册记录、软件包注册表以及README文本。我们将软件视为底层对象，并将这些机器可读的自我描述视为其表面：即人们和自动化系统读取项目所声明内容的关键点。引文指南、索引服务和自动化代理可能读取这些表面的不同子集，因此它们之间的不一致会悄然导致贡献和溯源的分裂。本文提出了一个简单且尚未直接测量的问题：当项目自身的元数据表面相互比较时，它们的一致性有多高？我们审计了117个开源研究软件项目，包括一个87个项目的高性能计算和量子计算语料库，以及一个30个项目的注册基线，该基线来自JOSS和pyOpenSci接受论文。

    arXiv:2608.17159v1 Announce Type: new  Abstract: Research software projects describe themselves in many places at once: citation files in the repository, archive deposits, DOI registry records, package registries, and README text. We treat the software as the underlying object and these machine-readable self-descriptions as its surfaces: the points where people and automated systems read what the project declares about the software. Citation guidance, indexing services, and automated agents may read a different subset of these surfaces, so disagreement between them can silently fragment credit and provenance. This paper asks a simple question that has not been measured directly: when a project's own metadata surfaces are compared with each other, how often do they agree? We audited 117 open-source research software projects, comprising an 87-project high-performance computing and quantum computing corpus and a 30-project registered baseline drawn from the JOSS and pyOpenSci accepted-pa
    
[^22]: LadderTeam：双智能体阶梯式引出框架

    LadderTeam: Dual-Agent Laddering Elicitation Framework

    [https://arxiv.org/abs/2608.17029](https://arxiv.org/abs/2608.17029)

    本文提出了LadderTeam框架，通过双智能体LLM架构自动化UX线框图访谈，克服了传统阶梯式访谈的手动成本和可扩展性限制。

    

    arXiv:2608.17029v1 公告类型：新 摘要：从最终用户那里引出详细且可操作的软件需求，是软件产品或应用迭代开发中的关键阶段。为了确保收集到的反馈详细且可操作，软件团队可以利用阶梯式访谈技术。虽然该技术能有效确保从软件反馈中获得细粒度且可操作的项目，但这些访谈受到若干限制。它们传统上是手动过程，伴随时间和财务负担，限制了可扩展性；访谈者必须在深入探查与应对受访者行为和文化的约束之间取得平衡。为解决这些限制，我们提出了 \textbf{LadderTeam}，一个开放、可复现的框架，利用双智能体大语言模型（LLM）架构自动化UX线框图访谈。一个主动的访谈者智能体执行三种探查策略之一（ACV、5-Why和JTBD），以引出可操作的软件需求。

    arXiv:2608.17029v1 Announce Type: new  Abstract: Eliciting detailed and actionable software requirements from end-users is a critical phase in the iterative development of a software product or application. To ensure the feedback collected is detailed and actionable, software teams can leverage the laddering interview technique. While effective for ensuring granular and actionable items from the software feedback, these interviews are subject to several limitations. They are traditionally a manual process associated with a time and financial burden, limiting scalability; interviewers must balance probing for depth while managing interviewee behavioral and cultural constraints. To address these limitations, we present \textbf{LadderTeam}, an open, reproducible framework that automates UX wireframe interviews using a dual-agent Large Language Model (LLM) architecture. An active interviewer agent executes one of three probing strategies (ACV, 5-Whys, and JTBD) to elicit actionable softwar
    
[^23]: ORCA：面向微服务事件的基于可观测性的程序修复

    ORCA: Observability-Grounded Program Repair for Microservice Incidents

    [https://arxiv.org/abs/2608.17018](https://arxiv.org/abs/2608.17018)

    ORCA通过将遥测数据差异转化为故障签名，并利用修复图代理和探索代理生成补丁，配合遥测验证器，显著提升了微服务事件修复的成本效益。

    

    arXiv:2608.17018v1 公告类型：新 摘要：微服务故障通常通过操作遥测数据进行诊断。然而，自动化程序修复系统通常从问题报告、局部代码上下文或失败的测试开始。这种不匹配在基于遥测的诊断和补丁生成之间留下了差距。我们提出了ORCA，一种针对微服务事件的基于可观测性的自动化程序修复（APR）流水线。ORCA首先将成对的失败和参考遥测数据的差异提炼为故障签名，然后使用该签名识别候选代码和部署配置位置。修复图代理和探索代理从这些位置生成统一差异补丁候选。ORCA使用基于遥测的补丁验证器评估生成的补丁，该验证器分离补丁有效性、语法和语义正确性、测试预言完整性和遥测回放。在575个案例的基准测试中，ORCA在成本效益方面优于所有评估的基线。结果显示

    arXiv:2608.17018v1 Announce Type: new  Abstract: Microservice failures are often diagnosed from operational telemetry. However, automated program repair systems usually start from issue reports, localized code context, or failing tests. This mismatch leaves a gap between telemetry-based diagnosis and patch generation. We present ORCA, an observability-grounded APR pipeline for microservice incidents. ORCA first distills the differences in paired failure and reference telemetry into a fault signature, then uses the signature to identify candidate code and deployment-configuration locations. Repair graph agents and an Exploration agent generate unified-diff patch candidates from these locations. ORCA evaluates generated patches with a Telemetry-Grounded Patch Verifier that separates patch validity, syntactic and semantic correctness, test-oracle integrity, and telemetry replay. On a 575-case benchmark, ORCA outperforms all evaluated baselines in terms of cost-effectiveness. Results show 
    
[^24]: PowderLine：一个程序化的粉末衍射分析应用程序

    PowderLine: a programmatic powder diffraction analysis application

    [https://arxiv.org/abs/2608.17009](https://arxiv.org/abs/2608.17009)

    PowderLine是一个Python工具，通过声明式配方和版本化验证，将粉末衍射精修过程标准化为机器可读的程序化流程，从而支持高通量和自主实验。

    

    全谱拟合方法，如Rietveld精修，在从粉末衍射数据中提取详细的结构、化学和微观结构信息方面表现出色。获得可靠结果需要相当的专业知识和特定软件的知识，而大规模应用这些方法通常依赖于为每个应用编写的自定义脚本。高通量实验和自主自驱动实验室越来越多地利用粉末衍射分析以程序化方式进行，并返回结构化的、机器可读的结果。在这里，我们介绍了PowderLine，一个Python应用程序，它将完整的精修过程封装到一个声明式配方中，根据版本化模式验证该配方，并通过精修软件执行它以返回结构化结果。该精修配方是对Rietveld或单峰分析的全包含、机器可读和可写的描述。

    arXiv:2608.17009v1 Announce Type: cross  Abstract: Whole-pattern fitting methods, such as Rietveld refinement, excel at extracting detailed structural, chemical, and microstructural information from powder diffraction data. Obtaining reliable results requires both considerable expertise and software-specific knowledge, and applying these methods at scale typically relies on custom scripts written for each application. High-throughput experiments and autonomous self-driving laboratories increasingly utilize powder diffraction analysis to proceed programmatically and to return structured, machine-readable results. Here, we introduce PowderLine, a Python application that encapsulates a complete refinement into a single declarative recipe, validates that recipe against a versioned schema, and executes it through refinement software to return structured results. The refinement recipe is an all-inclusive, machine-readable and -writable description of either Rietveld or single peak analysis t
    
[^25]: 利用弹珠机进行运行时监控的体验式学习

    Experiential Learning of Runtime Monitoring Using Pachinko

    [https://arxiv.org/abs/2608.16898](https://arxiv.org/abs/2608.16898)

    本文介绍了一种通过构建交互式弹珠机游戏来教授运行时监控的课堂作业，利用双核ESP32和RTLola规范，强调形式化方法的实践教学和跨课程的可移植性。

    

    arXiv:2608.16898v1 公告类型：交叉 摘要：我们展示了一个课堂作业的文档，该作业通过一个创造性的嵌入式系统构建——交互式弹珠机游戏——来教授运行时监控。该作业以双核ESP32工作流为中心，学生编写用于监控的RTLola规范，将这些监控器编译为C语言，并将其与传感器和执行器控制逻辑一起部署。弹珠机游戏事件会实时记录，并根据正式的时间逻辑规范用于触发声音、动画和电机行为。这项工作展示了如何在动手、基于项目的环境中，为创意和课堂规模的学习者教授形式化方法。我们还讨论了可移植性：作业模板、硬件栈、代码库和评估方法的设计和文档旨在可复制到其他嵌入式系统、创意计算或创客空间风格的课程中。该作业已分配给“创意嵌入式”课程的学生。

    arXiv:2608.16898v1 Announce Type: cross  Abstract: We present documentation of a classroom assignment that teaches runtime monitoring through a creative embedded systems build: an interactive Pachinko game. The assignment centers on a dual-core ESP32 workflow in which students write RTLola specifications for monitors, compile these monitors to C, and deploy them alongside sensor and actuator control logic. Pachinko game events are logged in real time and used to trigger sound, animation, and motor behavior according to formal temporal logic specifications.   This work showcases how formal methods can be taught in a hands-on, project-based setting for learners in a creative and classroom-scale setting. We also discuss portability: the assignment template, hardware stack, code base, and assessment approach are designed and documented to be replicated in other embedded systems, creative computing, or makerspace-style courses. This assignment was given to the students of Creative Embedded 
    
[^26]: Distribird：面向贝叶斯模型校准的文献知情先验分布设计

    Distribird: Literature-Informed Prior Distribution Design for Bayesian Model Calibration

    [https://arxiv.org/abs/2608.11210](https://arxiv.org/abs/2608.11210)

    Distribird是一个自动化代理工具，通过文献搜索、加权提取和AIC模型选择，为贝叶斯模型校准生成信息丰富的先验分布，并在无文献时提供合理的非信息性替代方案。

    

    arXiv:2608.11210v1 公告类型：新 摘要：基于过程的模型的贝叶斯校准需要为每个模型参数设定先验分布。尽管经过数十年的方法论研究，研究人员几乎总是退而使用均匀先验。主要原因是，从科学文献中构建信息丰富的先验分布过程缓慢，且需要领域和统计两方面的专业知识。我们提出了\textbf{Distribird}，一个代理型网络应用程序，可自动化此过程。给定参数名称、物理描述和领域背景，Distribird部署一个多代理流水线，搜索文献，根据领域相关性提取并加权报告值，并通过AIC模型选择拟合概率分布。当没有可用文献时，系统会退回到合理的非信息性替代方案，并清晰报告其生成的每个先验背后的证据和置信水平。它专为具有物理可解释参数的模型问题而设计。

    arXiv:2608.11210v1 Announce Type: new  Abstract: Bayesian calibration of process-based models requires a prior distribution for each model parameter. Despite decades of methodological work, researchers almost always fall back on uniform priors. The main reason is that building informative priors from scientific literature is slow and needs both domain and statistical expertise. We present \textbf{Distribird}, an agentic web application that automates this process. Given a parameter name, physical description, and domain context, Distribird deploys a multi-agent pipeline that searches the literature, extracts and weights reported values by domain relevance, and fits a probability distribution via AIC model selection. When no literature is available, the system falls back to sensible uninformative alternatives, and clearly reports both the evidence behind and the confidence level of every prior it produces. It is designed for the problems where the models have physically interpretable pa
    
[^27]: 从采纳到部署：软件开实践中AI整合的质性研究

    From Adoption to Deployment: A Qualitative Study on AI Integration in Software Development Practice

    [https://arxiv.org/abs/2607.16660](https://arxiv.org/abs/2607.16660)

    本研究通过22名从业者的半结构化访谈，揭示了软件实践中AI组件选择和集成时安全考虑不足的现状，并提出了改进决策流程的见解。

    

    arXiv:2607.16660v2 公告类型：替换-交叉 摘要：大型语言模型（LLMs）作为AI组件在现代软件系统中的日益普及，给软件供应链带来了独特的安全风险。尽管传统软件供应链的组件已有许多考虑和安全机制，但近期AI组件和平台的快速采纳却忽视了这些来之不易的经验教训。在选择和集成AI模型时，如果没有明确指导这些选择如何影响系统安全性，应用程序可能会面临恶意组件、数据泄露和意外行为等威胁。本研究的目标是通过探索性半结构化访谈研究，了解从业者在选择和集成AI组件时的决策过程和安全考虑。为此，我们对22名来自不同背景的软件开发者、架构师和AI从业者进行了半结构化访谈。

    arXiv:2607.16660v2 Announce Type: replace-cross  Abstract: The increasing adoption of Large Language Models (LLMs) as AI components in modern software systems introduces distinct security risks to the software supply chain. While many considerations and safety mechanisms are in place for components of the traditional software supply chain, the recent rapid adoption of AI components and platforms has overlooked these hard learned lessons. Selecting and integrating AI models without clear guidance on how these choices affect system security may leave applications vulnerable to threats, such as malicious components, data leakage, and unintended behavior. The goal of this study is to understand practitioners' decision making process and security considerations in selecting and integrating AI components through an exploratory semi-structured interview study. Toward this goal, we conducted semistructured interviews with 22 software developers, architects, and AI practitioners across diverse 
    
[^28]: ThinkLog：利用推理进行日志语句生成

    ThinkLog: Leveraging Reasoning for Log Statement Generation

    [https://arxiv.org/abs/2607.11615](https://arxiv.org/abs/2607.11615)

    ThinkLog通过将推理融入提示作为少量示例，提升了基于LLM的端到端日志语句生成的准确性，在9,619个Java方法上进行了验证。

    

    运行时日志是支持软件维护的重要信息来源。为了获得有用的日志，开发人员花费大量精力确定合适的日志位置、分配正确的严重性级别，并编写简洁而信息丰富的消息。因此，端到端的自动化日志语句生成可以帮助减轻这一负担，先前的工作已提出了许多针对此任务的方法。然而，现有方法仍表现出有限的准确性。为解决这一问题，我们提出了ThinkLog，一种基于LLM的端到端日志语句生成方法。ThinkLog的核心思想是融入推理，帮助LLM在日志插入、严重性级别分配和消息生成方面做出决策，从而提高日志语句生成的准确性。ThinkLog将推理作为少量示例注入提示中，并引导LLM生成合适的日志语句。我们在从pub提取的9,619个Java方法上进行了评估。

    arXiv:2607.11615v2 Announce Type: replace  Abstract: Runtime logs are an important source of information that supports software maintenance. To obtain useful logs, developers spend significant effort identifying appropriate log locations, assigning correct severity levels, and writing concise yet informative messages. Therefore, end-to-end automated log statement generation can help reduce this burden, and prior work has proposed many methods for this task. However, existing methods still exhibit limited accuracy. To address this problem, we propose ThinkLog, an LLM-based end-to-end log statement generation method. The core idea of ThinkLog is to incorporate reasoning that helps LLMs make decisions about log insertion, severity level assignment, and message generation, thereby improving log statement generation accuracy. ThinkLog injects reasoning into prompts as few-shot examples and guides LLMs to generate appropriate log statements. Evaluated on 9,619 Java methods extracted from pub
    
[^29]: FVSpec：将真实世界的基于属性的测试作为精益挑战

    FVSpec: Real-World Property-Based Tests as Lean Challenges

    [https://arxiv.org/abs/2606.01008](https://arxiv.org/abs/2606.01008)

    本文提出FVSpec基准，通过自动将真实Python项目中的基于属性测试转译为Lean 4规范，并采用三智能体LLM流水线，为AI在正式软件验证任务上的能力评估提供了大规模挑战和基线。

    

    我们提出了一个基准，用于评估AI模型和智能体在真实世界正式软件验证任务上的表现。首先，我们从真实世界的Python仓库中抓取了11,039个基于属性的测试（PBTs），然后自动将其中2,772个（25%）翻译成9,415个带有“sorry”占位符的Lean 4规范（每个PBT约对应3个形式化；当没有单一版本在质量指标上占优时，我们保留多个尝试）。将PBT翻译成Lean规范具有挑战性：它需要在Lean中建模Python语义，推断命令式PBT中编码的逻辑属性，并处理在一种少用语言中进行依赖类型编程的固有困难。我们描述了一个三智能体LLM流水线，用于将PBT转译成Lean规范，评估覆盖率和质量指标，并提供使用多种自动化和基于模型的方法进行证明生成的基线。所有代码（抓取器和智能体）和数据（PBT和Lean规范）均可用。

    arXiv:2606.01008v2 Announce Type: replace-cross  Abstract: We present a benchmark for evaluating AI models and agents on real-world formal software verification tasks. We first scrape 11,039 property-based tests (PBTs) from real-world Python repositories, then automatically translate 2,772 of them (25%) into 9,415 Lean 4 specifications with sorry placeholders (about 3 formalizations/PBT; we retain multiple attempts when none dominates on quality metrics). Translating PBTs into Lean specifications is challenging: it requires modeling Python semantics in Lean, inferring the logical property encoded in an imperative PBT, and handling the inherent difficulties of dependently-typed programming in a seldom-used language. We describe a three-agent LLM pipeline for transpiling PBTs into Lean specifications, evaluate coverage and quality metrics, and provide baselines for proof generation using several automated and model based approaches. All code (scraper and agents) and data (PBTs and Lean s
    
[^30]: AI治理中的部署后问责：跨监管实证分析AI事件

    Post-Deployment Accountability in AI Governance: A Cross-Regulatory Empirical Analysis of AI Incidents

    [https://arxiv.org/abs/2605.16281](https://arxiv.org/abs/2605.16281)

    本研究通过实证分析发现，AI部署后问责存在显著缺口，内部监控机制显著提升合规率，而外部检测事件合规率极低。

    

    部署后问责已成为AI治理的核心，但很少有实证证据表明，当AI系统失效时，监控、事件报告和影响评估义务是否可见。本研究分析了AI事件数据库（2020-2026年）中的真实世界AI事件，并根据欧盟AI法案、NIST AI风险管理框架和GDPR中的九项部署后条款对其进行编码。研究结果显示存在显著的问责缺口：77.1%的事件缺乏欧盟AI法案上市后监控的证据，99.6%的事件缺乏记录在案的数据保护影响评估证据。治理缺口也是系统性的，9.8%的事件同时违反两个或更多监管制度。通过内部监控检测到的事件显示出比外部检测事件高得多的合规率（在欧盟AI法案下为87.5%对5.3%；在NIST下为95.8%对58.1%），这表明监控能力是关键因素。

    arXiv:2605.16281v3 Announce Type: cross  Abstract: Post-deployment accountability has become central to AI governance, yet little empirical evidence shows whether monitoring, incident reporting, and impact assessment obligations are visible when AI systems fail. This study analyzes real-world AI incidents from the AI Incident Database (2020-2026) and codes them against nine post-deployment provisions from the EU AI Act, the NIST AI Risk Management Framework, and the GDPR. The findings show substantial accountability gaps: 77.1% of incidents lack evidence of EU AI Act post-market monitoring, and 99.6% lack documented Data-Protection Impact Assessment evidence. Governance gaps are also systemic, with 9.8% of incidents simultaneously non-compliant under two or more regimes. Incidents detected through internal monitoring show much higher compliance than externally detected incidents (87.5% vs. 5.3% under the EU AI Act; 95.8% vs. 58.1% under NIST), suggesting that monitoring capacity is a k
    
[^31]: 认识论债务：人工智能依赖的软件工程中的认知萎缩与系统性崩溃

    Epistemological Debt: Cognitive Atrophy and Systemic Collapse in AI-Dependent Software Engineering

    [https://arxiv.org/abs/2604.26855](https://arxiv.org/abs/2604.26855)

    本文提出“认识论债务”概念，指出AI依赖的软件开发会导致认知萎缩和系统性崩溃，并主张通过人机协同教学标准来维持工程韧性。

    

    大型语言模型（LLMs）融入软件开发生命周期（SDLC）掩盖了一个关键的社会技术失败：认知-系统性崩溃。本文引入了“认识论债务”，即当工程师用被动的AI验证取代逻辑推导时所承担的隐藏携带成本。这种债务侵蚀了根本原因分析所必需的心智模型，扩大了系统复杂性与人类理解之间的差距。此外，对合成代码的递归训练威胁到全球软件资源的同质化，减少了稳健工程所需的多样性。以2026年亚马逊中断事件为案例研究，本研究展示了“机械化趋同”如何导致系统性脆弱。为保持长期韧性，工程领导者必须超越基于提示的开发，实施严格的人机协同教学标准。该框架平衡了AI驱动生产力与人类认知参与。

    arXiv:2604.26855v3 Announce Type: replace  Abstract: The integration of Large Language Models (LLMs) into the software development lifecycle (SDLC) masks a critical socio-technical failure: Cognitive-Systemic Collapse. This paper introduces "Epistemological Debt," the hidden carrying cost incurred when engineers substitute logical derivation with passive AI verification. This debt erodes the mental models essential for root-cause analysis, widening the gap between system complexity and human comprehension. Furthermore, recursive training on synthetic code threatens to homogenize the global software reservoir, diminishing the variance required for robust engineering. Using the 2026 Amazon outages as a case study, this research illustrates how "mechanized convergence" leads to systemic fragility. To preserve long-term resilience, engineering leaders must move beyond prompt-based development to implement rigorous human-in-the-loop pedagogical standards. This framework balances AI-driven p
    
[^32]: 信号时态逻辑规范的反例分类

    Counterexample Classification for Signal Temporal Logic Specifications

    [https://arxiv.org/abs/2601.13743](https://arxiv.org/abs/2601.13743)

    本文提出了一种针对信号时态逻辑规范的反例分类方法，旨在减少工程师在调试中因重复反例检查所花费的精力。

    

    信号时态逻辑（STL）已被广泛用作指定混合系统所需行为的规范语言。STL最常见的用途之一是反例生成，即尝试生成反例信号，以展示系统如何违反给定的STL规范。已有多种反例生成方法和工具可用于高效生成反例，工程师可以检查这些反例以识别系统中的潜在缺陷。然而，其中一些反例可能被认为是相似的，因为它们描述的系统行为源于相同的根本原因或缺陷。由于检查反例可能是一项劳动密集型任务，一个能够呈现不同反例集合并避免显示重复反例的工具可以减少工程师在调试中所花费的精力。在本文中，我们提出了一种反例分类方法。

    arXiv:2601.13743v2 Announce Type: replace  Abstract: Signal Temporal Logic (STL) has been widely adopted as a specification language for specifying desirable behaviors of hybrid systems. One of the most common uses of STL is falsification, which attempts to generate counterexample signals that demonstrate how the system violates a given STL specification. A number of falsification methods and tools are available for efficient generation of counterexamples, which can be examined by the engineer to identify potential defects in the system. However, some of these counterexamples may be considered similar to each other in that they describe system behavior that stems from the same underlying causes or defects. Since examining counterexamples can be a labor-intensive task, a tool that presents a distinct set of counterexamples and avoids showing repetitive ones could reduce the amount of effort that the engineer spends in debugging.   In this paper, we propose a counterexample classificatio
    
[^33]: D-LiFT：通过代码质量驱动的微调改进基于LLM的反编译器后端

    D-LiFT: Improving LLM-based Decompiler Backend via Code Quality-driven Fine-tuning

    [https://arxiv.org/abs/2506.10125](https://arxiv.org/abs/2506.10125)

    本文提出D-LIFT，一种通过代码质量驱动的强化学习微调LLM的反编译器增强方法，在保持代码准确性的同时提高可读性，并引入D-Score评估系统。

    

    摘要：作为许多安全任务中的关键工具之一，反编译器从二进制文件中重建可读的源代码。然而，尽管近期取得了进展，其输出常常存在语法和语义错误，且难以阅读。最近，随着大型语言模型（LLMs）的出现，研究人员开始探索利用LLMs来优化反编译器输出的潜力。然而，我们对这些方法的研究揭示了它们的问题，例如引入新错误和依赖不可靠的准确性验证。在本文中，我们提出了D-LIFT，一种增强的反编译器-LLM流水线，它使用代码质量感知的强化学习进行微调的LLM。与先前忽略保持准确性的工作不同，D-LIFT遵循一个关键原则来提升反编译代码的质量：在提高可读性的同时保持准确性。D-LIFT的核心是我们提出了D-Score，一个集成的代码质量评估系统，用于评分...

    arXiv:2506.10125v4 Announce Type: replace-cross  Abstract: As one of the key tools in many security tasks, decompilers reconstruct human-readable source code from binaries. Yet, despite recent advances, their outputs often suffer from syntactic and semantic errors and remain difficult to read. Recently, with the advent of large language models (LLMs), researchers began to explore the potential of LLMs to refine decompiler output. Nevertheless, our study of these approaches reveals their problems, such as introducing new errors and relying on unreliable accuracy validation.   In this paper, we present D-LIFT, an enhanced decompiler-LLM pipeline with a fine-tuned LLM using code quality-aware reinforcement learning. Unlike prior work that overlooks preserving accuracy, D-LIFT adheres to a key principle for enhancing the quality of decompiled code: preserving accuracy while improving readability. Central to D-LIFT, we propose D-Score, an integrated code quality assessment system to score t
    
[^34]: RBCTest：利用大型语言模型挖掘并验证RESTful API测试中响应体的预言机

    RBCTest: Leveraging LLMs to Mine and Verify Oracles of API Response Bodies for RESTful API Testing

    [https://arxiv.org/abs/2504.17287](https://arxiv.org/abs/2504.17287)

    本文提出RBCTest，一种利用大型语言模型从API规范中静态挖掘响应体约束并生成测试用例的新方法，通过观察-确认方案有效降低幻觉，提高了预言机挖掘的准确性。

    

    在API测试中，推导API响应体上的逻辑约束以用作预言机，对于生成测试用例和执行RESTful API的自动化测试至关重要。然而，现有方法仅限于动态分析，即通过执行被测系统中的API来提取预言机。在本文中，我们提出了一种基于LLM的互补静态方法，其中API响应体的约束从API规范中挖掘。我们利用大型语言模型（LLMs）来理解API规范，挖掘响应体的约束，并生成测试用例。为了减少LLM的幻觉，我们应用了一种观察-确认（OC）方案，该方案使用初始提示来情境化约束，使后续提示能更准确地确认其存在。我们的实证结果表明，采用OC提示的RBCTest在约束挖掘中实现了高精度，平均值范围在...

    arXiv:2504.17287v4 Announce Type: replace  Abstract: In API testing, deriving logical constraints on API response bodies to be used as oracles is crucial for generating test cases and performing automated testing of RESTful APIs. However, existing approaches are restricted to dynamic analysis, in which oracles are extracted via the execution of APIs as part of the system under test. In this paper, we propose a complementary LLM-based static approach in which constraints for API response bodies are mined from API specifications. We leverage large language models (LLMs) to comprehend API specifications, mine constraints for response bodies, and generate test cases. To reduce LLM hallucination, we apply an Observation-Confirmation (OC) scheme that uses initial prompts to contextualize constraints, allowing subsequent prompts to more accurately confirm their presence. Our empirical results show that RBCTest with OC prompting achieves high precision in constraint mining, with averages rangi
    
[^35]: LSem2Vec：一种简单而有效的两阶段源代码嵌入方法

    LSem2Vec: A Simple yet Effective Two-Stage Approach for Source Code Embedding

    [https://arxiv.org/abs/2409.14644](https://arxiv.org/abs/2409.14644)

    LSem2Vec通过两阶段方法（LLM提取语义+句子嵌入生成向量）实现无需监督训练或微调的源代码嵌入，有效处理错误信息并提升性能。

    

    摘要：大型语言模型（LLMs）的出现显著推进了软件工程中的人工智能，源代码嵌入在诸如源代码克隆检测和源代码聚类等任务中扮演着关键角色。然而，现有的源代码嵌入方法，包括基于LLMs的方法，通常依赖昂贵的监督训练或微调来进行领域适应。本文提出了LSem2Vec（LLM提取的代码语义到向量嵌入），一种简单而有效的两阶段方法，通过结合大型语言模型和句子嵌入模型来嵌入源代码。具体来说，LSem2Vec利用LLM提取源代码的语义，然后使用句子嵌入模型生成表示向量。与之前的方法相比，LSem2Vec消除了对任务特定训练或微调的需求，并有效解决了源代码中常见的错误信息问题。

    arXiv:2409.14644v4 Announce Type: replace-cross  Abstract: The advent of large language models (LLMs) has significantly advanced artificial intelligence in software engineering, with source code embeddings playing a crucial role in tasks such as source code clone detection and source code clustering. However, existing methods for source code embedding, including those based on LLMs, often rely on costly supervised training or fine-tuning for domain adaptation. This paper proposes LSem2Vec (LLM-extracted code Semantics to Vector embedding), a simple yet effective two-stage approach to embedding source code by combining large language and sentence embedding models. Specifically, LSem2Vec leverages an LLM to extract the semantics of source code, and then uses a sentence embedding model to generate representation vectors. Compared with previous approaches, LSem2Vec eliminates the need for task-specific training or fine-tuning and effectively addresses erroneous information commonly found i
    

