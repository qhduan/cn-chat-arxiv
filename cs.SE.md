# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From Source Code to Network Profile: Automated and Traceable MUD Profile Generation for IoT Devices](https://arxiv.org/abs/2609.31594) | 提出了AutoMUD工具，通过分析物联网设备的固件和软件源代码自动生成准确、完整且可追溯的MUD网络访问控制配置文件，克服了传统基于流量监控方法难以覆盖罕见及依赖配置的通信行为的问题。 |
| [^2] | [Compact Documentation for Coding Agents: A Benchmark, an Optimizer, and Why It Does Not Transfer](https://arxiv.org/abs/2609.31587) | 本文提出了一个通过“重新生成代码能否通过原始测试”来评估代码描述的往返基准和优化器，发现完整性而非长度决定文档保真度，但令人意外的负面结果表明：当源代码可用时，更好的文档并不能帮助编程智能体解决真实的仓库问题。 |
| [^3] | [Context-Aware Functional Modeling for Android Third-Party Library Detection](https://arxiv.org/abs/2609.31409) | 提出了LibFan，一种基于上下文感知功能建模的学习型Android第三方库检测方法，通过方法级的上下文感知对比学习与库级功能划分两个互补组件，有效提升了在代码混淆、收缩、优化以及应用仅保留部分库内容场景下的检测鲁棒性。 |
| [^4] | [Beyond Approved Actions: Runtime Validation of Persistent Outcomes in Agent Workflows](https://arxiv.org/abs/2609.31301) | 提出了EffectMatch运行时系统，通过在受控执行边界内收集智能体操作产生的持久性变更，并将其与应用批准的内容进行比对验证后再决定提交，有效阻止了未批准的副作用在智能体工作流中传播。 |
| [^5] | [When the Model Retires: An Empirical Study of LLM Migration in Open-Source Applications](https://arxiv.org/abs/2609.31288) | 该研究通过分析 GitHub 上 22,555 个提交发现，约 82% 的开源应用对已退役 LLM 模型的迁移是在模型停用、应用已开始故障之后才完成的，表明开发者普遍未能及时响应模型退役通知。 |
| [^6] | [Semi-Automatic Quantification of Bayesian Networks for Software Decision Support: Comparing WSA and RNM in a Software R&D Organization](https://arxiv.org/abs/2609.31275) | 该研究在软件研发组织的物联网功能选择和界面设计选择两个真实决策场景中，比较了加权和算法（WSA）与排序节点法（RNM）作为贝叶斯网络半自动量化流程的表现，为数据有限条件下的软件决策支持提供了来自实际运营流程的比较证据。 |
| [^7] | [Joule-Profiler: Profiling the Energy Consumption of Build Automation Tools Made Easy](https://arxiv.org/abs/2609.31228) | 本文提出了开源命令行工具 Joule-Profiler，它通过 Intel RAPL 和 NVML 测量硬件能耗并结合标准输出监控，将 Maven 构建流水线的能耗精确归因到各个构建阶段，并以 Google Gson 为案例展示了冷构建与热构建之间的能耗差异。 |
| [^8] | [Rethinking Data Quality for AI-Driven Systems: Evidence from Practitioner Interviews](https://arxiv.org/abs/2609.31191) | 本文通过对16名从业者的访谈首次提供了实证证据，揭示了AI驱动系统中数据质量内涵的根本转变——可追溯性转向模型行为归因、智能体上下文与记忆成为数据对象、合成数据使真实性成为关注点、合法性成为训练数据的准入门槛。 |
| [^9] | [MetaPermit: Scalable and Auditable Access Control for AI Agents via LLM-Inferred Meta-Attributes](https://arxiv.org/abs/2609.31039) | 提出MetaPermit框架，利用LLM从智能体与用户的交互中推断元属性，并将语义推断与安全执行解耦，从而为AI智能体实现可扩展、一致且可审计的工具访问控制。 |
| [^10] | [CoCoRerank: Towards Conventional Commit Message Generation by Component and Candidate Consistency Reranking](https://arxiv.org/abs/2609.30953) | 该论文构建了一个包含86,688条高质量提交、遵循完整约定式提交规范的基准数据集，并提出CoCoRerank重排序框架，通过代码变更与提交信息各组件间的横向一致性以及候选间的纵向共识，提升大语言模型生成规范化提交信息的能力。 |
| [^11] | [Developing a Roadmap to an AI-first Organization: A Case Study in Embedded Software Development](https://arxiv.org/abs/2609.30863) | 本文通过对一家大型嵌入式系统公司40名从业者的研讨会数据进行混合方法分析，研究了该公司向AI优先组织转型的路线图，发现智能体AI预计将深刻影响团队结构、所需能力、组织战略及开发人员角色。 |
| [^12] | [AGATE: Provenance-Based Runtime Defense Against Compositional Attacks on LLM Agents](https://arxiv.org/abs/2609.30830) | AGATE通过在LLM智能体运行框架边界上部署结合授权管理与数据溯源的确定性门控机制，在决策路径中不依赖LLM即可在运行时防御由普通操作组合而成的攻击。 |
| [^13] | [Analyzing and Mitigating Cost-Inefficient Behaviors in Coding Agents](https://arxiv.org/abs/2609.30725) | 该论文首次系统研究编码智能体中的成本低效行为，识别出子集化检索、相似脚本生成与测试重复执行三种行为（影响79%–98%的任务、最高占任务成本的22.75%），并评估了结构感知检索、智能体自合成技能与开发者设计技能三种缓解策略的效果。 |
| [^14] | [A Framework for Identifying, Categorizing, and Explaining Bias in AI-Generated Code](https://arxiv.org/abs/2609.30642) | 本研究提出了一个基于分类体系的框架，用于识别、分类和解释AI生成代码中的偏见，并构建真实标准数据集评估了各类LLM作为自动化偏见检测与解释系统的可靠性。 |
| [^15] | [EA-Ops: Git-Native Architecture as Code for Continuous Enterprise Architecture Governance](https://arxiv.org/abs/2609.30593) | 本文提出EA-Ops，一个开源的Git原生企业架构即代码框架，通过将架构事实以YAML表示、按ArchiMate 3.2验证类型化关系、强制执行治理规则并进行图基变更影响分析，实现架构治理与软件工程工作流的深度集成，在240次故障注入试验中取得完美的精确率和召回率，并可扩展至5万个对象和10万个关系。 |
| [^16] | [Closing the Loop: Continuous Measurement-Driven Refinement of Offloading Predictions](https://arxiv.org/abs/2609.30429) | 该论文提出一种测量驱动的运行时闭环系统，通过轻量级多头神经网络的增量在线更新，持续重新校准车辆计算卸载中的绝对性能指标预测，从而解决离线训练预测器因时序变化、异构硬件等因素而漂移所导致的可靠性问题。 |
| [^17] | [Evaluating Code Recommender Systems: A Review](https://arxiv.org/abs/2609.30351) | 该综述通过分析2017–2024年的92篇文献发现，代码推荐系统的评估以系统为中心的离线评估为主，以用户为中心的评估（如在线评估和用户研究）严重不足，且评估活动主要集中在软件构造阶段。 |
| [^18] | [Untangling the Spaghetti Code in Game Development: A Review of Challenges and Academic Solutions](https://arxiv.org/abs/2609.30349) | 本文通过系统性文献综述分析了34篇研究，揭示了游戏开发中代码质量问题的根源（短期思维、需求频繁变更、代码复用不足、缺乏自动化测试），并总结了学术界提出的代码异味检测工具与测试实践等解决方案。 |
| [^19] | [What Will Remain Human in Software Architecture? A Focus Group Report](https://arxiv.org/abs/2609.30334) | 本研究通过EuroPLoP 2026上的焦点小组探讨AI开发智能体对软件架构实践的影响，发现架构决策、问责制和架构护栏编写仍是不可替代的人类核心职责，并提出“驾驭工程”这一新兴概念——即构建用于治理AI辅助系统创建的系统的学科。 |
| [^20] | [When Is a Multi-Agent Code Judge Actually Grounded? Two Label-Free Measurements, and a Judge That Declines to Guess](https://arxiv.org/abs/2609.30328) | 该论文指出多智能体验证框架在代码评判中因证据无法满足“独立于答案且能区分候选解”这两个条件而失效（在78%–95%的比较中判定两个解同样好、准确率仅4.4%），据此提出两种无需标签的度量方法来检验评判是否真正有据可依，并设计了一个在证据不足时拒绝猜测的评判器。 |
| [^21] | [Proceedings Tenth Symposium on Working Formal Methods](https://arxiv.org/abs/2609.30324) | 该论文集收录了2026年在罗马尼亚蒂米什瓦拉举行的第十届形式化方法工作研讨会（FROM 2026）经程序委员会评审录用的14篇论文。 |
| [^22] | [HyQDB: LLM-Assisted Debugging for Hybrid Quantum Workflows](https://arxiv.org/abs/2609.30313) | HyQDB是一个分层LLM代理调试工具，它通过将确定性的硬件、物理和优化证据注入修复过程来处理机械性故障，并在无证据时升级到意图重构层来处理概念性故障，同时推出了QFaultBench基准测试。 |
| [^23] | [Empty Intersection: Provenance Coverage Rose to 98% and Neither Verification Decision Moved](https://arxiv.org/abs/2609.30308) | 本文在194,620行生产数据快照上实测了行级溯源等级标签与单一写入入口两种结构性防御，发现尽管溯源覆盖率提升至98%，两项验证判定却均未改变，其贡献在于首次测量了这类防御对验证结论的实际影响。 |
| [^24] | [Silent Success: A Release Gate That Passed on Checks It Never Ran, and Eight More](https://arxiv.org/abs/2609.30307) | 论文揭示了一类发布质量门禁的“静默成功”缺陷——当检查未运行时，缺失数据被默认计为“无违规”，使几乎未执行任何检查的运行也能满分通过；通过引入“无法判定”这第三种状态可将静默通过转为失败，并在工业与开源环境中共发现了九例此类案例。 |
| [^25] | [Orchestrating AI-Assisted Code Remediation: Socio-Technical Bottlenecks in a Large Industrial Repository](https://arxiv.org/abs/2609.29172) | 本研究通过对大型工业C++代码库开展为期15天的实地案例研究，揭示了大规模AI辅助代码修复在持续集成、代码审查和团队协调等方面所面临的社会技术瓶颈。 |
| [^26] | [From Evidence to Effect: Authority Semantics and Runtime Infrastructure for Stateful Agents](https://arxiv.org/abs/2609.08472) | 该论文揭示了“跨基底权限鸿沟”问题——即授权信息存在于智能体可见的工作区和记忆之外，并通过受控消融实验证明，向系统提供原始权限回执可使语义成功率从 0/32 提升至 32/32。 |
| [^27] | [Scanning the Harness: A Repository Study of Configuration Exposures in AI Coding Agents](https://arxiv.org/abs/2609.07360) | 本研究对3,171个公开GitHub仓库开展了首次大规模系统性分析，识别出AI编码智能体配置中的六类安全暴露问题，发现15.4%-17.9%的设置存在未固定MCP包版本、宽泛执行权限授予等供应链风险。 |
| [^28] | [One Capability or Many? Testing the Economic Validity of Frontier AI Evaluation](https://arxiv.org/abs/2608.29420) | 本研究通过潜变量模型对421个模型配置和12个基准的分析发现，经济基准测量的并非独立的独特能力，而是与其他基准共享的同一通用能力维度（单一因子解释74.5%的共同方差），从而质疑了前沿AI经济评估的构念效度。 |
| [^29] | [AgentDV: Closed-Loop Agentic AI for Hardware Design Verification](https://arxiv.org/abs/2608.27148) | AgentDV是一个闭环智能体AI框架，通过可运行性过滤、CSR接地检查和覆盖率引导迭代，将LLM测试平台生成转化为可靠的RTL验证流水线。 |
| [^30] | [Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection](https://arxiv.org/abs/2608.17965) | 本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。 |
| [^31] | [Grid-Orch: An LLM-Powered Orchestrator for Distribution Grid Simulation and Analytics](https://arxiv.org/abs/2605.12728) | Grid-Orch通过MCP协议将大语言模型与OpenDSS配电网仿真相结合，提供36种领域工具，使工程师能用自然语言完成潮流计算、电压分析、QSTS仿真和自动化优化，并支持本地部署以满足电力系统安全隔离需求。 |
| [^32] | [VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation](https://arxiv.org/abs/2604.21375) | VLAA-GUI提出一个模块化GUI自动化框架，通过强制性完整性验证器杜绝无视觉证据的过早成功宣告、多层级循环断路器打破重复失败循环、以及按需在线搜索应对不熟悉元素，系统性地解决了GUI智能体的过早停止与重复循环两大核心难题。 |
| [^33] | [Mixed Choice in Asynchronous Multiparty Session Types](https://arxiv.org/abs/2602.23927) | 该论文提出了一个支持异步混合选择的多方会话类型框架，其核心构造允许分布式参与者间协议状态暂时不一致但保证最终达成一致，并通过进展性与操作对应性证明了正确性，同时实现了用于规范验证和Erlang/OTP协议编程的实用工具链，并以RabbitMQ的amqp_client为案例进行了验证。 |
| [^34] | [LLM-based Vulnerability Detection at Project Scale: An Empirical Study](https://arxiv.org/abs/2601.19239) | 本研究首次对项目级LLM漏洞检测器进行了大规模实证评估，发现通用智能体召回率最高但难以应对复杂代码，同时揭示了基于LLM的方法与传统方法在误报和不完备性方面的关键局限。 |
| [^35] | [BabelCoder: Agentic Code Translation with Specification Alignment](https://arxiv.org/abs/2512.06902) | 提出BabelCoder，一个将代码翻译任务分解为翻译、测试和精炼等多个专门智能体协同工作的智能体化框架，通过规范对齐显著提升了多语言代码翻译的准确性和质量。 |
| [^36] | [SLMFix: Leveraging Small Language Models for Domain Specific Language Error Fixing with Reinforcement Learning](https://arxiv.org/abs/2511.19422) | 该论文提出SLMFix流水线，利用强化学习微调的小语言模型根据解释器反馈修复LLM生成代码中的语法错误，在低资源编程语言上使验证器通过率提升了40%。 |
| [^37] | [Structural Enforcement of Statistical Rigor in AI-Driven Discovery: A Functional Architecture](https://arxiv.org/abs/2511.06701) | 该论文提出一种函数式架构，通过Haskell的Research monad、声明式脚手架、操作系统级沙箱以及机器验证的Lean 4形式化LORD在线FDR控制，从结构上强制保证AI驱动科学发现中的统计严谨性，防止AI科学家系统因不受控的多重检验而产生虚假发现。 |
| [^38] | [A First Look at the Self-Admitted Technical Debt in Test Code: Taxonomy and Detection](https://arxiv.org/abs/2510.22409) | 该研究首次系统性考察测试代码中的自承认技术债务（SATD），通过人工分析1,000个开源Java项目的5万条注释构建了包含11个类别的SATD分类体系，并评估了现有检测工具和大语言模型自动检测此类债务的能力。 |

# 详细

[^1]: 从源代码到网络画像：面向物联网设备的自动化且可追溯的MUD配置文件生成

    From Source Code to Network Profile: Automated and Traceable MUD Profile Generation for IoT Devices

    [https://arxiv.org/abs/2609.31594](https://arxiv.org/abs/2609.31594)

    提出了AutoMUD工具，通过分析物联网设备的固件和软件源代码自动生成准确、完整且可追溯的MUD网络访问控制配置文件，克服了传统基于流量监控方法难以覆盖罕见及依赖配置的通信行为的问题。

    

    制造商使用描述标准允许物联网制造商在MUD文件中定义预期的网络行为。该文件可以被转换为可执行的访问控制策略，将被入侵的设备限制为仅通过制造商定义的通信模式运行。然而，MUD的实际应用依赖于准确、完整且可维护的配置文件。现有方法使用基于流量的自动化方式，但需要部署设备并进行长时间监控，仅能捕获观察期间所表现出的行为。罕见的、由故障触发的或依赖配置的通信可能会缺失，导致生成不完整的策略，从而干扰合法操作，并且难以洞察每条规则所对应的软件组件。我们提出了AutoMUD，一个由源代码驱动的工具，可以从物联网设备的固件和软件源代码中生成可追溯的MUD配置文件。AutoMUD...

    arXiv:2609.31594v1 Announce Type: cross  Abstract: The Manufacturer Usage Description (MUD) standard allows IoT manufacturers to define expected network behaviors in a MUD file. This file can be translated into enforceable access-control policies, restricting compromised devices to operate solely through manufacturer-defined communication patterns. However, practical adoption of MUD depends on profiles that are accurate, complete, and maintainable. Existing approaches use traffic-based automation but require device deployment and prolonged monitoring, capturing only behavior exercised during observation. Rare, failure-triggered, or configuration-dependent communications may remain absent, producing incomplete policies that disrupt legitimate operation and offer limited insight into the software components responsible for each rule. We present AutoMUD, a source-code-driven tool that generates traceable MUD profiles for IoT devices from their firmware and software source code. AutoMUD co
    
[^2]: 面向编程智能体的紧凑文档：一个基准、一个优化器，以及它为何无法迁移

    Compact Documentation for Coding Agents: A Benchmark, an Optimizer, and Why It Does Not Transfer

    [https://arxiv.org/abs/2609.31587](https://arxiv.org/abs/2609.31587)

    本文提出了一个通过“重新生成代码能否通过原始测试”来评估代码描述的往返基准和优化器，发现完整性而非长度决定文档保真度，但令人意外的负面结果表明：当源代码可用时，更好的文档并不能帮助编程智能体解决真实的仓库问题。

    

    我们研究了自然语言文档是否能帮助编程智能体解决软件问题，并构建了用于生成和评估此类文档的工具。我们引入了一个往返式基准，通过“根据代码描述重新生成的代码能否通过原始测试”来为描述打分，并证明决定描述保真度的是完整性而非长度。以该基准作为优化信号，我们发现了一条描述撰写提示词，可达到完全保真度并能泛化到未见过的文件。随后，我们检验了这项工作最初的动机假设：更好的文档能帮助智能体解决真实的仓库问题。我们在两个模型家族和十个仓库上进行了测试，并设置了一个阳性对照以确认我们的评估能够检测出真正的改进，但结果表明情况并非如此。当源代码存在时，无论是静态紧凑文档还是检索到的上下文，都不如仅凭问题描述有效。我们将这一负面结果与……（原文截断）

    arXiv:2609.31587v1 Announce Type: cross  Abstract: We investigate whether natural-language documentation helps coding agents resolve software issues, and we build the tools to construct and evaluate it. We introduce a roundtrip benchmark that scores code descriptions by whether code regenerated from them passes the original tests, and show that completeness, not length, drives a description's fidelity. Using the benchmark as an optimization signal, we discover a description-writing prompt that reaches full fidelity and generalizes to unseen files. We then test the hypothesis that motivated the work: that better documentation helps an agent resolve real repository issues. Across two model families and ten repositories, and against a positive control confirming that our evaluation can detect a genuine improvement, we find that it does not. When the source is present, neither static compact documentation nor retrieved context beats the issue alone. We report this negative result together 
    
[^3]: 面向Android第三方库检测的上下文感知功能建模

    Context-Aware Functional Modeling for Android Third-Party Library Detection

    [https://arxiv.org/abs/2609.31409](https://arxiv.org/abs/2609.31409)

    提出了LibFan，一种基于上下文感知功能建模的学习型Android第三方库检测方法，通过方法级的上下文感知对比学习与库级功能划分两个互补组件，有效提升了在代码混淆、收缩、优化以及应用仅保留部分库内容场景下的检测鲁棒性。

    

    第三方库（TPLs）在Android应用中被广泛使用，但其复用可能引入安全风险，并干扰下游的程序分析。现有的Android第三方库检测方法面临两个关键局限：其手工设计的特征在激进的代码变换下较为脆弱，且其整库匹配策略在应用仅保留第三方库部分内容时效果不佳。在本文中，我们提出了LibFan，一种基于学习的、以上下文感知功能建模为核心的Android第三方库检测方法。它通过两个互补的组件来实现这种建模：方法级的上下文感知对比学习和库级的功能划分。在方法级，它通过对比训练学习语义表示，同时结合方法的对外调用关系和类级上下文，提升了对混淆、收缩和优化的鲁棒性。在库级，它对（原文摘要此处截断）

    arXiv:2609.31409v1 Announce Type: cross  Abstract: Third-party libraries (TPLs) are widely used in Android apps, but their reuse can introduce security risks and interfere with downstream program analyses. Existing Android TPL detection approaches face two key limitations: their hand-crafted features are fragile under aggressive code transformations, and their whole-library matching strategies are ineffective when apps retain only part of a TPL.   In this paper, we propose LibFan, a learning-based Android TPL detection approach based on context-aware functional modeling. It realizes this modeling through two complementary components: context-aware contrastive learning at the method level and functional partitioning at the library level. At the method level, it learns semantic representations through contrastive training while incorporating outgoing call relationships and class-level context, improving robustness to obfuscation, shrinking, and optimization. At the library level, it part
    
[^4]: 超越已批准的操作：智能体工作流中持久性结果的运行时验证

    Beyond Approved Actions: Runtime Validation of Persistent Outcomes in Agent Workflows

    [https://arxiv.org/abs/2609.31301](https://arxiv.org/abs/2609.31301)

    提出了EffectMatch运行时系统，通过在受控执行边界内收集智能体操作产生的持久性变更，并将其与应用批准的内容进行比对验证后再决定提交，有效阻止了未批准的副作用在智能体工作流中传播。

    

    大型语言模型智能体越来越多地在软件系统上执行操作，不再仅仅生成文本，而是直接改变数据库和在线服务。然而，一个已获批准的数据库更新可能成功执行，却留下一个未获批准的通知，因为执行过程可能产生超出所请求变更的持久性影响。当前的保障机制可以批准某个操作或记录其后果，但若不在继续执行前检查持久性结果，未获批准的结果可能被误认为成功并传播到后续步骤中。我们提出了EffectMatch，一个运行时系统，它在受控的执行边界内收集持久性变更，并将其与应用程序针对当前状态和执行所批准的内容进行比较，该比较决定了提交以及依赖执行能否继续进行。在206个公开业务任务上的对比评估中，EffectMatch保留了所有正常的执行，并阻止了所有经过测试的错误提交。六项20次运行的消融实验揭示了……（摘要原文在此处截断）

    arXiv:2609.31301v1 Announce Type: cross  Abstract: Large language model agents increasingly act on software systems, no longer merely generating text but also changing databases and online services. However, an approved database update may succeed yet leave an unapproved notification because execution can produce persistent effects beyond the requested change. Current safeguards can approve an action or record its aftermath, but without checking the persistent result before continuation, an unapproved outcome can be accepted as success and propagated to later steps. We present EffectMatch, a runtime that collects persistent changes within a controlled execution boundary and compares them with what the application approved for the current state and execution. The comparison governs commit and dependent execution. In comparative evaluation on 206 public business tasks, EffectMatch preserved all clean executions and prevented all tested incorrect commits. Six 20-run ablations exposed the 
    
[^5]: 当模型退役时：开源应用中大语言模型迁移的实证研究

    When the Model Retires: An Empirical Study of LLM Migration in Open-Source Applications

    [https://arxiv.org/abs/2609.31288](https://arxiv.org/abs/2609.31288)

    该研究通过分析 GitHub 上 22,555 个提交发现，约 82% 的开源应用对已退役 LLM 模型的迁移是在模型停用、应用已开始故障之后才完成的，表明开发者普遍未能及时响应模型退役通知。

    

    基于商业大语言模型（LLM）API 构建的应用依赖于服务提供者按自身计划退役的模型版本，其通知期限从一年到两周不等。我们探讨当模型退役时，应用实际上会发生什么。我们在 GitHub 上挖掘从 OpenAI、Anthropic 和 Google 官方弃用的模型和端点迁移出去的提交，并将每个提交与提供者公布的公告日期和停用日期进行匹配。在 17,703 个非分叉仓库（2024-2026 年）的 22,555 个提交中，有 5,139 个与官方事件相匹配；两名独立编码员对分层抽样的 300 个样本进行了验证（kappa = 0.89-0.95），并根据其标注对所有估计进行了重新加权。我们发现，据估计 82%（95% 置信区间 79-84）的从退役模型迁移的提交是在停用日期之后才完成的——即应用已经开始出现故障之后——且这一结果不受仓库热门程度、以往退役经验或存在……（摘要原文在此处截断）

    arXiv:2609.31288v1 Announce Type: new  Abstract: Applications built on commercial large language model (LLM) APIs depend on model versions that providers retire on their own schedule, with notice periods ranging from one year to two weeks. We ask what actually happens to applications when a model is retired. We mine GitHub for commits that migrate away from officially deprecated models and endpoints of OpenAI, Anthropic, and Google, matching each commit to the provider's published announcement and shutdown dates. From 22,555 commits in 17,703 non-fork repositories (2024-2026), 5,139 are matched to an official event; two independent coders validated a stratified sample of 300 (kappa = 0.89-0.95), and we reweight all estimates by their labels. We find that an estimated 82% (95% CI 79-84) of migrations away from retired models were committed after the shutdown date - after the application had started failing - regardless of repository popularity, prior retirement experience, or the presen
    
[^6]: 面向软件决策支持的贝叶斯网络半自动量化：在软件研发组织中比较WSA与RNM方法

    Semi-Automatic Quantification of Bayesian Networks for Software Decision Support: Comparing WSA and RNM in a Software R&D Organization

    [https://arxiv.org/abs/2609.31275](https://arxiv.org/abs/2609.31275)

    该研究在软件研发组织的物联网功能选择和界面设计选择两个真实决策场景中，比较了加权和算法（WSA）与排序节点法（RNM）作为贝叶斯网络半自动量化流程的表现，为数据有限条件下的软件决策支持提供了来自实际运营流程的比较证据。

    

    专家驱动的贝叶斯网络可以在历史数据有限的情况下支持重复出现的软件决策，但量化其条件概率表（CPT）需要大量的概率判断。半自动方法可以减少直接的专家引出工作量，但从业者在实际运营决策流程中几乎没有可参考的比较证据。我们报告了一项嵌入式案例研究，在单一软件研发（R&D）组织的两个场景中，将加权和算法（WSA）与排序节点法（RNM）作为完整的引出与量化流程进行比较：物联网项目的功能选择和用户界面设计选择。在每个场景中，两种流程共享相同的图结构、根节点先验和决策证据。我们通过15个专家定义的模型走查场景以及对决策会议中记录的备选方案的回顾性重构对两者进行评估。WSA在走查中匹配了7/7和6/8

    arXiv:2609.31275v1 Announce Type: new  Abstract: Expert-driven Bayesian networks can support recurring software decisions when historical data are limited, but quantifying their conditional probability tables (CPTs) requires many probability judgments. Semi-automatic methods reduce direct elicitation, yet practitioners have little comparative evidence from operational decision processes. We report an embedded case study comparing the Weighted Sum Algorithm (WSA) and Ranked Nodes Method (RNM) as complete elicitation-and-quantification pipelines in two contexts within a single software research and development (R&D) organization: feature selection for Internet of Things projects and user interface design selection. Within each context, the pipelines shared the graph, root priors, and decision evidence. We evaluated them through 15 expert-defined model-walkthrough scenarios and retrospective reconstructions of alternatives recorded in decision meetings. WSA matched 7/7 and 6/8 walkthrough
    
[^7]: Joule-Profiler：轻松剖析构建自动化工具的能耗

    Joule-Profiler: Profiling the Energy Consumption of Build Automation Tools Made Easy

    [https://arxiv.org/abs/2609.31228](https://arxiv.org/abs/2609.31228)

    本文提出了开源命令行工具 Joule-Profiler，它通过 Intel RAPL 和 NVML 测量硬件能耗并结合标准输出监控，将 Maven 构建流水线的能耗精确归因到各个构建阶段，并以 Google Gson 为案例展示了冷构建与热构建之间的能耗差异。

    

    构建流水线是现代软件开发中不可或缺的一部分，但其能源足迹对从业者而言在很大程度上是不可见的。现有的 CI 能源工具要么依赖基于模型的估算（由于云运行器的硬件访问限制），要么只报告流水线的总能耗，而无法将其分解为有意义的阶段。Joule-Profiler 是一款面向 Linux 的开源命令行工具，它通过 Intel RAPL（CPU）和 NVML（NVIDIA GPU）测量硬件能耗，并通过监控标准输出将能耗归因到用户定义的程序阶段。在这篇工具论文中，我们以 Google Gson 为案例研究，在 Maven 构建流水线的场景下演示了 Joule-Profiler。通过对 Maven 插件调用应用标记模式匹配，我们将最近 5 个 Gson 版本的构建分析为分阶段的能耗概况，并比较了冷构建（空本地仓库）与热构建（依赖已缓存）的差异。

    arXiv:2609.31228v1 Announce Type: new  Abstract: Build pipelines are integral to modern software development, yet their energy footprint remains largely invisible to practitioners. Existing CI energy tools either rely on model-based estimation (due to hardware access restrictions in cloud runners) or report only total pipeline energy without decomposing it into meaningful phases. Joule-Profiler is an open-source command-line tool for Linux that measures hardware energy consumption via Intel RAPL (CPU), NVML (NVIDIA GPU), and attributes it to user-defined program phases by monitoring standard output. In this tool paper, we demonstrate Joule-Profiler in the context of Maven build pipelines using Google Gson as a case study. By applying a token pattern-matching Maven plugin invocations, we analyze builds across the 5 most recent Gson releases into per-phase energy profiles, comparing cold builds (empty local repository) and warm builds (cached dependencies).
    
[^8]: 重新思考AI驱动系统的数据质量：来自从业者访谈的证据

    Rethinking Data Quality for AI-Driven Systems: Evidence from Practitioner Interviews

    [https://arxiv.org/abs/2609.31191](https://arxiv.org/abs/2609.31191)

    本文通过对16名从业者的访谈首次提供了实证证据，揭示了AI驱动系统中数据质量内涵的根本转变——可追溯性转向模型行为归因、智能体上下文与记忆成为数据对象、合成数据使真实性成为关注点、合法性成为训练数据的准入门槛。

    

    数据质量研究通常将数据视为被存储、处理和验证的输入。而在AI驱动的软件密集型系统中，数据还塑造着模型行为、评估方式和合法使用。关于从业者在此类条件下如何定义、评估和管理质量，目前的实证证据仍然有限。我们访谈了来自九个组织的16名从业者，采用反身性主题分析法对访谈记录进行分析，并从参与者的叙述中提炼出六个主题。在AI系统中，可追溯性从模块化调试转变为对模型行为的归因，而将模型用作质量评估者则引入了循环性问题。智能体的上下文和记忆成为数据对象，合成数据和伪标签数据使真实性成为新的质量关注点。在基础模型开发中，合法性成为训练数据的准入门槛，而代表性则通过系统必须安全运行的情境覆盖范围来评判。

    arXiv:2609.31191v1 Announce Type: cross  Abstract: Data quality research has usually treated data as an input that is stored, processed, and validated. In AI-driven software-intensive systems, data also shapes model behavior, evaluation, and lawful use. Empirical evidence remains limited on how practitioners define, assess, and manage quality under these conditions. We interviewed 16 practitioners from nine organizations and analyzed the transcripts using reflexive thematic analysis and developed six themes from participants' accounts. In AI systems, traceability shifted from modular debugging to attributing model behavior, while using models as quality assessors introduced circularity. Agent context and memory became data objects, and synthetic and pseudo-labeled data made authenticity a quality concern. In foundation-model development, lawfulness became a gate for training data, while representativeness was judged through coverage of situations in which the system must behave safely.
    
[^9]: MetaPermit：通过LLM推断的元属性实现可扩展且可审计的AI智能体访问控制

    MetaPermit: Scalable and Auditable Access Control for AI Agents via LLM-Inferred Meta-Attributes

    [https://arxiv.org/abs/2609.31039](https://arxiv.org/abs/2609.31039)

    提出MetaPermit框架，利用LLM从智能体与用户的交互中推断元属性，并将语义推断与安全执行解耦，从而为AI智能体实现可扩展、一致且可审计的工具访问控制。

    

    配备工具的自主AI智能体的兴起带来了重大的安全风险，包括无意的工具误用以及通过间接提示注入（IPI）攻击进行的对抗性操纵。在实践中，已部署的智能体系统（如OpenAI Codex和Claude Code）通过粗粒度权限规则与基于LLM对单个拟议操作的判断相结合的方式来保护工具调用。然而，这两个组件都存在重要局限：静态策略必须预先设想可能的用户意图，因此无法扩展到开放式任务；而LLM驱动的授权虽支持动态决策，但产生的结果不一致，且仍易受针对性IPI攻击。为了提供可扩展且更一致的授权，我们提出了MetaPermit，这是一个基于策略的工具访问控制框架，它将语义推断与安全执行解耦。通过分析智能体与用户的交互，我们推导……（原文摘要在此处截断）

    arXiv:2609.31039v1 Announce Type: cross  Abstract: The rise of autonomous AI agents equipped with tools has introduced significant security risks, ranging from unintended tool misuse to adversarial manipulation through Indirect Prompt Injection (IPI) attacks. In practice, deployed agent systems such as OpenAI Codex and Claude Code protect tool invocations through a combination of coarse-grained permission rules and LLM-based judgments about individual proposed actions. Both components, however, have important limitations: static policies must anticipate possible user intents and therefore do not scale to open-ended tasks, while LLM-driven authorization supports dynamic decisions but produces inconsistent outcomes and remains vulnerable to targeted IPI attacks. To provide scalable and more consistent authorization, we propose MetaPermit, a policy-based tool access-control framework that decouples semantic inference from security enforcement. By analyzing agent-user interactions, we deri
    
[^10]: CoCoRerank：通过组件与候选一致性重排序迈向约定式提交信息生成

    CoCoRerank: Towards Conventional Commit Message Generation by Component and Candidate Consistency Reranking

    [https://arxiv.org/abs/2609.30953](https://arxiv.org/abs/2609.30953)

    该论文构建了一个包含86,688条高质量提交、遵循完整约定式提交规范的基准数据集，并提出CoCoRerank重排序框架，通过代码变更与提交信息各组件间的横向一致性以及候选间的纵向共识，提升大语言模型生成规范化提交信息的能力。

    

    提交信息对于理解软件变更至关重要，然而现有的自动提交信息生成方法通常将消息视为非结构化的文本序列。这限制了其支持标准化开发工作流的能力——在这类工作流中，提交信息通常需要遵循约定式提交规范，即 type (scope): subject 的形式。本文研究了完整 CCS 格式下的约定式提交信息生成问题。我们构建了一个新的基准数据集，包含从开源 GitHub 仓库收集的 86,688 条高质量提交，并通过结构规范化与语义质量过滤，将每条消息规范化为 type、scope 和 subject 三个组成部分。基于该基准，我们提出了一种面向大语言模型生成的、基于二维一致性的重排序框架 CoCoRerank。CoCoRerank 利用了代码变更、type、scope 和 subject 之间的横向一致性，以及候选之间的纵向共识……

    arXiv:2609.30953v1 Announce Type: new  Abstract: Commit messages are essential for understanding software changes, yet automatic commit message generation typically treats a message as an unstructured text sequence. This limits its ability to support standardized development workflows, where commit messages are often expected to follow the Conventional Commits Specification (CCS) in the form type (scope): subject. In this paper, we study conventional commit message generation under the complete CCS format. We construct a new benchmark of 86,688 high-quality commits collected from open-source GitHub repositories, with each message normalized into type, scope, and subject through structural normalization and semantic quality filtering. Based on this benchmark, we propose a two-dimensional consistency-based reranking framework named CoCoRerank for LLM-based generation. CoCoRerank exploits horizontal consistency among the code change, type, scope, and subject, as well as vertical consensus
    
[^11]: 制定AI优先组织路线图：嵌入式软件开发案例研究

    Developing a Roadmap to an AI-first Organization: A Case Study in Embedded Software Development

    [https://arxiv.org/abs/2609.30863](https://arxiv.org/abs/2609.30863)

    本文通过对一家大型嵌入式系统公司40名从业者的研讨会数据进行混合方法分析，研究了该公司向AI优先组织转型的路线图，发现智能体AI预计将深刻影响团队结构、所需能力、组织战略及开发人员角色。

    

    AI智能体的出现预计将通过超越AI作为助手的角色，转向能够以日益增强的自主性来规划、执行和评估开发任务的系统，从而重塑软件工程。这一转变对于嵌入式软件组织尤为重要，因为这类组织通常对质量、可追溯性、验证和长期可维护性有着严格的要求。本文对一家大型嵌入式系统公司向AI优先组织转型的过程进行了案例研究。通过混合研究方法，我们分析了从一场有40名参与者（包括Scrum主管、架构师、管理层和产品负责人）参加的半结构化研讨会中收集的数据。研究结果表明，参与者预计智能体AI将影响团队结构、所需能力、组织战略以及开发人员在组织中的角色。基于这些发现，本文讨论了……（摘要原文在此处截断）

    arXiv:2609.30863v1 Announce Type: cross  Abstract: The emergence of AI agents is expected to reshape software engineering by moving beyond AI as assistants towards systems capable of planning, executing, and evaluating development tasks with increasing autonomy. This transition is particularly significant for embedded software organizations, where strict requirements for quality, traceability, verification, and long-term maintainability often apply. This paper presents a case study of a large embedded systems company and its transition toward becoming an AI-first organization. Through a mixed method, we analyzed data collected from a semi-structured workshop with 40 participants, including scrum masters, architects, management, and product owners. The findings show that the participants expect agentic AI to affect team structure, required competencies, organizational strategies, and developers' roles within the organization. Based on these findings, the paper discusses implications for
    
[^12]: AGATE：基于数据溯源的LLM智能体组合攻击运行时防御

    AGATE: Provenance-Based Runtime Defense Against Compositional Attacks on LLM Agents

    [https://arxiv.org/abs/2609.30830](https://arxiv.org/abs/2609.30830)

    AGATE通过在LLM智能体运行框架边界上部署结合授权管理与数据溯源的确定性门控机制，在决策路径中不依赖LLM即可在运行时防御由普通操作组合而成的攻击。

    

    LLM智能体可以通过一系列普通操作产生有害影响。判断此类行为既需要确定允许其执行的权限，也需要确定其所携带数据的来源。我们提出了AGATE，一个部署在经过插桩的智能体运行框架边界上的授权与数据溯源门控。操作者声明和宿主批准事件构成授权的基础；被委托的操作受到授权（grant）的约束，这些授权绑定精确参数、会过期、并限制使用次数。来源注册将观察到的输入与后续的数据传递关联起来，而效果账本则追踪重复的请求。确定性检查在决策路径中无需LLM参与即可做出判断，并保留执行证据以供取证回放。适配器集成了三个生产级运行框架——DeepSeek Harness、OpenCode和OpenClaw——且无需修改宿主代码，将每个宿主的原生观察点和否决点转换为统一的单一接口……

    arXiv:2609.30830v1 Announce Type: cross  Abstract: LLM agents can produce harmful effects through sequences of ordinary operations. Judging such actions requires establishing both the authority that permits them and the origin of the data they carry. We present AGATE, an authorization and data-provenance gate at instrumented agent-harness boundaries. Operator declarations and host approval events ground authorization; delegated actions are constrained by grants that bind to exact parameters, expire, and permit a limited number of uses. Source registration connects observed inputs to subsequent transfers, while an effect ledger tracks repeated requests. Deterministic checks make decisions without an LLM in the decision path and retain their grounds with execution evidence for forensic replay. Adapters integrate three production harnesses -- DeepSeek Harness, OpenCode, and OpenClaw -- without modifying host code, translating each host's native observation and veto points into a single sh
    
[^13]: 分析与缓解编码智能体中的成本低效行为

    Analyzing and Mitigating Cost-Inefficient Behaviors in Coding Agents

    [https://arxiv.org/abs/2609.30725](https://arxiv.org/abs/2609.30725)

    该论文首次系统研究编码智能体中的成本低效行为，识别出子集化检索、相似脚本生成与测试重复执行三种行为（影响79%–98%的任务、最高占任务成本的22.75%），并评估了结构感知检索、智能体自合成技能与开发者设计技能三种缓解策略的效果。

    

    arXiv:2609.30725v1 公告类型：新 摘要：编码智能体虽然效果显著，但往往会产生高昂的货币成本。其反复出现的成本低效行为至今仍缺乏充分研究。我们首次对编码智能体中的行为性成本低效开展了研究，分析了 Claude Code 和 Mini-SWE-Agent 在 SWE-bench Verified 上四种配置下的 1,200 条运行轨迹。我们识别出三种成本低效行为：子集化检索、相似脚本生成和测试重复执行。随后，我们在留出的 SWE-bench Verified 和 Pro 任务上基于 1 万条轨迹评估了三种缓解策略：结构感知检索、智能体自合成技能以及开发者设计技能。我们的主要发现包括：(1) 这三种行为影响 79.00%–98.00% 的编码任务，最高可占任务成本的 22.75%。(2) 结构感知检索可能引入检索开销并改变智能体的任务委派方式，导致检索效率提升不一致，且成本增加最高可达……（原文摘要至此截断）

    arXiv:2609.30725v1 Announce Type: new  Abstract: Although effective, coding agents often incur substantial monetary costs. Their recurring cost-inefficient behaviors remain underexplored. We conduct the first study of behavioral cost inefficiencies in coding agents, analyzing 1,200 trajectories from Claude Code and Mini-SWE-Agent across four configurations on SWE-bench Verified. We identify three cost-inefficient behaviors: subsumed retrieval, similar script generation, and test re-execution. We then evaluate three mitigation strategies: structure-aware retrieval, agent-synthesized skills, and developer-designed skills, over 10k trajectories on held-out SWE-bench Verified and Pro tasks. Our main findings are: (1) The three behaviors affect 79.00\%--98.00\% of coding tasks and account for up to 22.75\% of task cost. (2) Structure-aware retrieval can introduce retrieval overhead and alter agent delegation, causing inconsistent improvements in retrieval efficiency and cost increases of up
    
[^14]: 一个用于识别、分类和解释AI生成代码中偏见的框架

    A Framework for Identifying, Categorizing, and Explaining Bias in AI-Generated Code

    [https://arxiv.org/abs/2609.30642](https://arxiv.org/abs/2609.30642)

    本研究提出了一个基于分类体系的框架，用于识别、分类和解释AI生成代码中的偏见，并构建真实标准数据集评估了各类LLM作为自动化偏见检测与解释系统的可靠性。

    

    随着大型语言模型（LLM）被集成到软件开发工作流中，人们对AI生成代码中无意产生的偏见日益担忧。尽管有证据表明这些偏见确实存在，但系统性地识别、分类和解释这些偏见的研究仍然有限。本研究调查了AI生成代码中的偏见，并通过一个基于分类体系（taxonomy）的框架评估LLM能否可靠地识别和解释这些偏见。我们扩展了一个现有的包含偏见的AI生成Python代码数据集，并人工为代码片段标注偏见类别和人工编写的理由，从而建立了一个真实标准（ground-truth）数据集。利用该数据集，我们通过上下文学习（ICL）将专有和开源LLM评估为自动化偏见检测与理由生成系统。最后，我们使用结构化理由指标和代码识别指标，分析了LLM生成的解释与人工编写的理由之间的相似性。我们的研究结果表明……（原文摘要到此截断）

    arXiv:2609.30642v1 Announce Type: cross  Abstract: As Large Language Models (LLMs) become integrated into software development workflows, concerns regarding unintentional biases in AI-generated code. Although evidence suggests these biases exist, limited research has systematically identified, categorized, and explained them. This study investigates bias in AI-generated code and evaluates whether LLMs can reliably identify and explain it through a taxonomy-driven framework. We extended an existing dataset of biased AI-generated Python code and manually annotated snippets with bias categories and human-authored justifications to establish a ground-truth dataset. Using this dataset, we evaluated proprietary and open-source LLMs as automated bias detection and justification systems through ICL. Finally, we analyzed similarity between LLM-generated explanations and human-authored justifications using structured justification and code identification metrics.   Our findings demonstrate that 
    
[^15]: EA-Ops：基于Git原生的架构即代码，实现持续企业架构治理

    EA-Ops: Git-Native Architecture as Code for Continuous Enterprise Architecture Governance

    [https://arxiv.org/abs/2609.30593](https://arxiv.org/abs/2609.30593)

    本文提出EA-Ops，一个开源的Git原生企业架构即代码框架，通过将架构事实以YAML表示、按ArchiMate 3.2验证类型化关系、强制执行治理规则并进行图基变更影响分析，实现架构治理与软件工程工作流的深度集成，在240次故障注入试验中取得完美的精确率和召回率，并可扩展至5万个对象和10万个关系。

    

    企业架构（EA）知识库通常将架构模型与用于变更软件和基础设施的工程工作流相分离。本文提出了EA-Ops，一个开源的Git原生企业架构即代码框架，该框架将架构事实表示为YAML格式，基于ArchiMate 3.2配置文件验证类型化关系，执行组织特定的治理规则，执行基于图的变更影响分析，并从同一经过评审的源代码发布面向人员的报告和静态交互式门户。我们使用可复现的GitHub Actions测试框架对EA-Ops进行了评估。八类独立注入的结构、语义和治理故障各执行30次试验；所有240次试验均与真实情况完全匹配，精确率、召回率和F1值均达到1.000。经过30次重复测量的可扩展性实验达到了50,000个对象和100,000个关系：中位数验证时间为……

    arXiv:2609.30593v1 Announce Type: new  Abstract: Enterprise architecture (EA) repositories frequently separate architecture models from the engineering workflow used to change software and infrastructure. This article presents EA-Ops, an open-source Git-native Enterprise Architecture-as-Code framework that represents architecture facts as YAML, validates typed relationships against an ArchiMate 3.2 profile, enforces organization-specific governance rules, performs graph-based change-impact analysis, and publishes human-facing reports and a static interactive portal from the same reviewed source. We evaluate EA-Ops with a reproducible GitHub Actions harness. Eight independently injected structural, semantic, and governance fault classes were executed across 30 trials each; all 240 trials matched ground truth exactly, with precision, recall, and $F_1$ of 1.000. Scalability experiments with 30 measured repetitions reached 50,000 objects and 100,000 relationships: median validation time wa
    
[^16]: 闭环控制：基于持续测量驱动的卸载预测精细化

    Closing the Loop: Continuous Measurement-Driven Refinement of Offloading Predictions

    [https://arxiv.org/abs/2609.30429](https://arxiv.org/abs/2609.30429)

    该论文提出一种测量驱动的运行时闭环系统，通过轻量级多头神经网络的增量在线更新，持续重新校准车辆计算卸载中的绝对性能指标预测，从而解决离线训练预测器因时序变化、异构硬件等因素而漂移所导致的可靠性问题。

    

    现代车辆越来越多地将计算密集型的感知与决策功能卸载到后端服务器上，这需要对往返时延（RTT）、处理时间和利用率等绝对性能指标进行准确预测。在实际运行中，强烈的时序变化、异构的后端硬件以及多模态的延迟分布，会导致离线训练的预测器发生漂移，从而给时延敏感型功能带来可靠性缺口。我们通过一个可实际运行的、测量驱动的闭环来弥合这一缺口，该闭环在运行时持续重新校准绝对值预测器。该系统将真实执行测量值与预测值对齐，并在保持模型稳定性的同时，对轻量级多头神经网络进行增量式在线更新。该模型隐式地学习了输入指标广泛的非高斯分布特性，此外，我们在评估中采用基于sigma的误差分析来刻画残余变异性。

    arXiv:2609.30429v1 Announce Type: new  Abstract: Modern vehicles increasingly offload computation- ally intensive perception and decision functions to backend servers, requiring accurate predictions of absolute performance metrics such as Round-Trip Time (RTT), processing time, and utilization. In practice, strong temporal variability, heterogeneous backend hardware, and multimodal latency regimes cause offline- trained predictors to drift, creating a reliability gap for latency- sensitive functions. We address this gap with an operational, measurement-driven closed loop that continuously recalibrates absolute-value predictors during runtime. The system aligns real execution measurements with predicted values and performs incremental online updates of a lightweight multi-head neural network while preserving model stability. The model implicitly learns the broad, non-Gaussian spread of input metrics, and a sigma-based error analysis in our evaluation characterizes resid- ual variability
    
[^17]: 代码推荐系统的评估：一项综述

    Evaluating Code Recommender Systems: A Review

    [https://arxiv.org/abs/2609.30351](https://arxiv.org/abs/2609.30351)

    该综述通过分析2017–2024年的92篇文献发现，代码推荐系统的评估以系统为中心的离线评估为主，以用户为中心的评估（如在线评估和用户研究）严重不足，且评估活动主要集中在软件构造阶段。

    

    背景：代码推荐系统（CRSs）是一类专门作用于源代码工件的软件系统，能够在软件开发的各个阶段为开发人员提供自动生成的推荐。其目标是提升软件质量，同时增强开发人员的效率、有效性和使用体验。问题：尽管其重要性日益增长，但目前对这些系统以人为中心的评估现状知之甚少。研究方法：我们开展了一项系统性文献综述，以识别并综合评估代码推荐系统的原始研究（2017–2024年）。共纳入92篇文献，并对其进行了系统性内容分析。结果：我们的研究证实，离线评估是最常见的、以系统为中心的评估类型，而以用户为中心的评估（在线评估、用户研究）则很少被报道。评估活动集中于软件构造阶段。大多数研究明确……（原文在此处截断）

    arXiv:2609.30351v1 Announce Type: new  Abstract: Context: Code recommender systems (CRSs) are specialized software systems operating on source artifacts to provide automatically generated recommendations to software developers in all phases of development. The goal is to improve software quality while enhancing developers' efficiency, effectiveness, and experience. Problem: Despite the growing importance, little is known about the state of evaluating these systems in a human-centric manner. Research Approach: We conducted a systematic literature review to identify and to synthesize primary studies evaluating CRSs (2017-2024). Ninety-two publications were included and subjected to a systematic content analysis. Results: Our study confirms that Offline Evaluations are the most common, system-centric evaluation type, whereas user-centric evaluations (Online Evaluations, User Studies) are rarely reported. Evaluations are concentrated on the Software Construction phase. Most studies explici
    
[^18]: 解开游戏开发中的“意大利面条式代码”：挑战与学术解决方案综述

    Untangling the Spaghetti Code in Game Development: A Review of Challenges and Academic Solutions

    [https://arxiv.org/abs/2609.30349](https://arxiv.org/abs/2609.30349)

    本文通过系统性文献综述分析了34篇研究，揭示了游戏开发中代码质量问题的根源（短期思维、需求频繁变更、代码复用不足、缺乏自动化测试），并总结了学术界提出的代码异味检测工具与测试实践等解决方案。

    

    游戏领域的代码质量被认为比其他软件领域更缺乏结构、更加复杂混乱，这导致了较低的可维护性和更高的缺陷频率。在这项工作中，我们进行了一项系统性文献综述，以回答四个主要研究问题：（1）游戏开发中代码异味和技术债务的发生率与其他软件领域相比有何不同；（2）学术界提出了哪些解决方案来缓解这些代码质量问题；（3）研究人员是否与开发人员有效合作以实施这些解决方案；（4）所提出的解决方案是否足以解决已诊断出的问题。研究共对34项工作进行了映射和分析。我们发现游戏领域受到短期思维、持续的需求变更、代码复用减少以及极少或完全没有自动化测试的影响。作为潜在的解决方案，我们识别了用于检测代码异味的工具和模型，以及相关的测试实践（原文摘要在此处被截断）。

    arXiv:2609.30349v1 Announce Type: new  Abstract: The code quality in the game domain is perceived as less structured and more convoluted than in other domains, which results in lower maintainability and higher bug frequency. In this work, we conduct a systematic literature review to address four main research questions: (1) how the incidence of code smells and technical debt differs in game development compared to other software domains; (2) what academic solutions are proposed to mitigate these code quality issues; (3) whether researchers collaborate effectively with developers to implement these solutions; and (4) if the proposed solutions are sufficient to tackle the diagnosed problems. Thirty-four works were mapped and analyzed. We found that the game domain is affected by a short-term mentality, continuous requirements changes, reduced code reuse, and minimal to no automated testing. As potential solutions, we identified tools and models to detect code smells, along with testing p
    
[^19]: 软件架构中什么仍将由人类掌控？一份焦点小组报告

    What Will Remain Human in Software Architecture? A Focus Group Report

    [https://arxiv.org/abs/2609.30334](https://arxiv.org/abs/2609.30334)

    本研究通过EuroPLoP 2026上的焦点小组探讨AI开发智能体对软件架构实践的影响，发现架构决策、问责制和架构护栏编写仍是不可替代的人类核心职责，并提出“驾驭工程”这一新兴概念——即构建用于治理AI辅助系统创建的系统的学科。

    

    AI开发智能体正被越来越多地用于支持并部分自动化软件架构任务。为了探索从业者如何看待这一转变——具体而言，什么在改变、什么保持不变、以及哪些新职责正在涌现——我们在第31届欧洲模式、人员与实践语言会议（EuroPLoP 2026）上开展了一次焦点小组研究。来自工业界和学术界的22名参与者讨论了当前的实践、信任与验证策略、AI自主性的边界、治理挑战以及对教育的影响。除其他发现外，我们观察到与会者存在广泛共识：架构决策、问责制以及架构护栏的编写从本质上仍然是人类的任务。一个核心的新兴概念是“驾驭工程”：即构建用于治理AI辅助系统创建过程的系统的学科，它包含验证机制、知识层以及公司特定的标准。

    arXiv:2609.30334v1 Announce Type: cross  Abstract: AI development agents are increasingly used to support and partially automate software architecture tasks. To explore how practitioners perceive this shift, specifically what changes, what remains, and what new responsibilities emerge, we conducted a focus group at the 31st European Conference on Pattern Languages of Programs, People, and Practices (EuroPLoP 2026). Twenty-two participants from industry and academia discussed current practices, trust and validation strategies, the boundaries of AI autonomy, governance challenges, and implications for education. Among others, we found broad consensus that architectural decision-making, accountability, and the authoring of architectural guardrails remain fundamentally human tasks. A central emergent concept was harness engineering: the discipline of building the system that governs AI-assisted system creation, comprising validation mechanisms, knowledge lay- ers, and company-specific stan
    
[^20]: 多智能体代码评判器何时才真正有据可依？两种无需标签的度量方法，以及一个拒绝猜测的评判器

    When Is a Multi-Agent Code Judge Actually Grounded? Two Label-Free Measurements, and a Judge That Declines to Guess

    [https://arxiv.org/abs/2609.30328](https://arxiv.org/abs/2609.30328)

    该论文指出多智能体验证框架在代码评判中因证据无法满足“独立于答案且能区分候选解”这两个条件而失效（在78%–95%的比较中判定两个解同样好、准确率仅4.4%），据此提出两种无需标签的度量方法来检验评判是否真正有据可依，并设计了一个在证据不足时拒绝猜测的评判器。

    

    当一个语言模型评判另一个语言模型的代码是否正确时，它并不会报告证据的缺失。它会返回一个附带推理过程的、充满自信的判决，这与一个真正有据可依的判决难以区分。多智能体验证将判断分解为可核查的声明，并逐条对照证据加以验证，是一种颇有前景的应对方式，且在证据为一组检索文档时效果良好。我们认为这类方法对其证据有两个要求：证据必须独立于被评审的答案，并且必须在被比较的两个候选答案之间有所区分。第二个条件在检索文档场景下会自动满足，但在代码评判中不再成立。我们在两个代码评判基准上未经修改地运行已发表的框架MARCH，进行了80项按条件逐单元的测量，发现它在78%到95%的比较中判定两个解同样好，而在直接询问同一模型时准确率仅为4.4%。

    arXiv:2609.30328v1 Announce Type: new  Abstract: When one language model judges whether another's code is correct, it does not report the absence of evidence. It returns a confident verdict with reasoning attached, indistinguishable from a verdict it had grounds for. Multi-agent verification, which decomposes a judgment into checkable claims and verifies each against evidence, is a promising response and works well when the evidence is a set of retrieved documents.   We argue such methods require two things of their evidence: it must be independent of the answer under review, and it must differ between the two candidates being compared. The second condition holds automatically with retrieved documents and stops holding in code judging.   Running MARCH, a published framework unmodified over 80 condition-by-cell measurements on two code judging benchmarks, we find it declares both solutions equally good on 78 to 95% of comparisons, reaching 4.4% accuracy where the same model asked direct
    
[^21]: 第十届形式化方法工作研讨会论文集

    Proceedings Tenth Symposium on Working Formal Methods

    [https://arxiv.org/abs/2609.30324](https://arxiv.org/abs/2609.30324)

    该论文集收录了2026年在罗马尼亚蒂米什瓦拉举行的第十届形式化方法工作研讨会（FROM 2026）经程序委员会评审录用的14篇论文。

    

    第十届形式化方法工作研讨会（FROM 2026）于2026年9月15日至17日在罗马尼亚蒂米什瓦拉举行。该会议由蒂米什瓦拉西部大学信息学院、e-奥地利研究所以及逻辑与数据科学研究所（ILDS）联合举办，并与第28届科学计算符号与数值算法国际研讨会（SYNASC 2026）同期举行。本论文集收录了程序委员会录用的14篇论文的全文。

    arXiv:2609.30324v1 Announce Type: new  Abstract: The 10th Working Formal Methods Symposium (FROM 2026) was held in Timi\c{s}oara, Romania on September 15-17, 2026. It was organized jointly by the Faculty of Informatics of West University of Timi\c{s}oara, the e-Austria institute, and the Institute for Logic and Data Science (ILDS). This event was co-located with the 28th International Symposium on Symbolic and Numeric Algorithms for Scientific Computing (SYNASC 2026). This volume contains the texts of  the 14 papers accepted by the Program Committee for presentation.
    
[^22]: HyQDB：基于LLM辅助的混合量子工作流调试

    HyQDB: LLM-Assisted Debugging for Hybrid Quantum Workflows

    [https://arxiv.org/abs/2609.30313](https://arxiv.org/abs/2609.30313)

    HyQDB是一个分层LLM代理调试工具，它通过将确定性的硬件、物理和优化证据注入修复过程来处理机械性故障，并在无证据时升级到意图重构层来处理概念性故障，同时推出了QFaultBench基准测试。

    

    混合量子程序的故障经常无声地发生，然而现有调试工具在检测和修复这些故障方面提供的支持有限。这类故障在领域专家报告的故障中占主导地位，但现有工具评估所用的公开数据对这种故障模式的代表性不足。我们的关键洞察是，故障可分为两类，需要不同的处理策略：机械性故障，可以进行确定性分析；以及概念性故障，需要重构程序的意图。为应对这一挑战，我们提出了HyQDB，一个分层代理，它将确定性的硬件、物理和优化证据注入到LLM修复过程中。当未检测到任何证据时，代理将这种沉默视为信号，升级到第二层的意图重构层，该层推断程序的行为并将其与实际实现进行核对。为评估HyQDB，我们引入了QFaultBench，一个基于专家构建的基准测试……

    arXiv:2609.30313v1 Announce Type: new  Abstract: Hybrid quantum program failures frequently occur silently, yet existing debugging tools provide limited support for detecting and repairing them. These faults dominate the failures reported by domain experts, yet existing tools evaluate on public data that under-represents this failure mode. Our key insight is that faults divide into two classes that require different strategies: mechanical faults, which allow deterministic analysis, and conceptual faults, which require reconstructing the program's intent. To address this challenge, we present HyQDB, a tiered agent that injects deterministic hardware, physics and optimization evidence into the LLM repair process. When no evidence is detected, the agent treats this silence as a signal to escalate to a second intent-reconstruction tier, that infers the program's behaviour and reconciles it with the implementation. To evaluate HyQDB, we introduce QFaultBench, a benchmark built from an exper
    
[^23]: 空交集：溯源覆盖率升至98%，但两项验证判定均未改变

    Empty Intersection: Provenance Coverage Rose to 98% and Neither Verification Decision Moved

    [https://arxiv.org/abs/2609.30308](https://arxiv.org/abs/2609.30308)

    本文在194,620行生产数据快照上实测了行级溯源等级标签与单一写入入口两种结构性防御，发现尽管溯源覆盖率提升至98%，两项验证判定却均未改变，其贡献在于首次测量了这类防御对验证结论的实际影响。

    

    针对溯源的两种结构性防御——其一是在每一行数据上附加等级标签，使验证例程无法将系统自身写入的输出误当作外部观测值；其二是设置单一写入入口，使等级标签得以被强制执行而非仅流于惯例——在催生这些防御的生产部署上进行了测量，测量基于一个包含194,620行的冻结快照以及该快照所支持的两项验证判定。结果是，两种防御均未改变任何一项判定。这两种方案均由一篇相伴论文提出，该论文诊断了该部署的问题所在：其验证例程使用系统自身写入的值来判定结果。这两种方案并非新创：它们都是在互不引用的领域中早已确立的实践，且未发现任何先前工作测量过其中任何一种是否会改变验证结论，因此本文所提供的是测量结果而非方案本身。按等级标签过滤验证查询会使两项判定均从“通过”变为“未确定”；放宽……（原文摘要在此处截断）

    arXiv:2609.30308v1 Announce Type: new  Abstract: Two structural defenses for provenance, a grade on every row, so that a verification routine cannot mistake the system's own output for an observation, and a single write ingress, so that the grade is enforced rather than merely conventional, were measured against the production deployment that motivated them, over a frozen snapshot of 194,620 rows and the two verification decisions the snapshot supports. Neither reaches either decision. Both were prescribed by a companion paper, which diagnosed that deployment: its verification routines decided outcomes using values the system itself had written. Neither prescription is new: both are established practice in fields that do not cite one another, and no prior work measuring whether either changes a verdict was found, so what is offered here is the measurement and not the prescriptions. Filtering the verification queries by grade turns both decisions from pass to undetermined; widening the 
    
[^24]: 静默成功：一个在从未运行的检查上通过判定的发布门禁，以及另外八例

    Silent Success: A Release Gate That Passed on Checks It Never Ran, and Eight More

    [https://arxiv.org/abs/2609.30307](https://arxiv.org/abs/2609.30307)

    论文揭示了一类发布质量门禁的“静默成功”缺陷——当检查未运行时，缺失数据被默认计为“无违规”，使几乎未执行任何检查的运行也能满分通过；通过引入“无法判定”这第三种状态可将静默通过转为失败，并在工业与开源环境中共发现了九例此类案例。

    

    生产发布流水线中的一个阻塞式质量门禁在某次运行中报告了 PASS（通过），而其中一个子门禁的两项检查均未执行，另一个子门禁八项检查仅运行了六项。决定该次运行结果的两个键都询问“是否观察到违规”，而二者都是从已剔除未能运行案例的总体中计算这一答案的，因此缺失的数据回答了“否”，一次几乎什么都没检查的运行却获得了满分，带来了两周的绿色构建。引入第三个取值——通过、违规、无法判定——将这些静默通过转变为失败，并把检查缺口写入退出状态；随后对同一规范的独立工作发现一个检测器以 0.000177 个百分点的余量触发，其触发下限稳定保持到第六位小数。此后又发现了八个相同形式的实例：其中五个来自同一项目合作，两个出现在开源项目中，还有一个网关的配置缓存 TTL 被声明为……（原文摘要在此截断）

    arXiv:2609.30307v1 Announce Type: new  Abstract: A blocking quality gate in a production release pipeline reported PASS on a run in which one subgate had executed neither of its two checks and another had run six of its eight. Both keys that decided the run asked whether a violation had been observed, and both computed that from a population already stripped of the cases that failed to run, so absent data answered "no" and a run that checked almost nothing scored perfectly, for two weeks of green builds. Introducing a third value, pass, violate, and unable to determine, turned those silent passes into failures and put the shortfall into the exit status; separate work on the same specifications then found a detector firing with a margin of 0.000177 percentage points, its firing floor holding at the sixth decimal place. Eight further instances of the same form followed: five more from the same engagement, two in open-source projects, a gateway where a configured cache TTL was declared bu
    
[^25]: 编排AI辅助的代码修复：大型工业代码库中的社会技术瓶颈

    Orchestrating AI-Assisted Code Remediation: Socio-Technical Bottlenecks in a Large Industrial Repository

    [https://arxiv.org/abs/2609.29172](https://arxiv.org/abs/2609.29172)

    本研究通过对大型工业C++代码库开展为期15天的实地案例研究，揭示了大规模AI辅助代码修复在持续集成、代码审查和团队协调等方面所面临的社会技术瓶颈。

    

    背景：在大型、长期存续的代码库中，代码退化问题若通过手动重构和机会性清理来修复，成本十分高昂。基于大语言模型（LLM）的编程助手可以大规模执行机械化的代码修复，但其对工业工作流的影响尚未得到充分探索。目标：我们研究了大规模AI辅助代码修复如何影响一个大型工业代码库中基于提交构建的持续集成（CI）、代码审查和团队协调，以及当AI辅助使源代码修改变得低廉时，哪些社会技术瓶颈会制约此类修复。方法：我们报告了一项为期15天的探索性单案例实地研究，研究过程中一名经验丰富的开发者使用命令行AI编程助手来修复一个闭源工业C++代码库中普遍存在的问题。我们将Gerrit元数据与开发者日记和团队聊天记录进行三角互证，并通过描述性统计和定性编码进行分析。结果：AI辅助

    arXiv:2609.29172v1 Announce Type: new  Abstract: Background: Code degradation in large, long-lived codebases is costly to remediate through manual refactoring and opportunistic clean-ups. LLM-based coding assistants can perform mechanical remediation at scale, but their impact on industrial workflows is underexplored. Objective: We investigate how massive AI-assisted code remediation affects build-on-commit continuous integration (CI), code review, and team coordination in a large industrial repository, and which socio-technical bottlenecks constrain such remediation when source editing becomes cheap through AI assistance. Method: We report on a 15-day exploratory single-case field study in which an experienced developer used a command-line AI coding buddy to remediate widespread issues in a closed-source industrial C++ repository. We triangulate Gerrit metadata with a developer diary and team chat, analyzed through descriptive statistics and qualitative coding. Results: AI-assisted re
    
[^26]: 从证据到效果：面向有状态智能体的权限语义与运行时基础设施

    From Evidence to Effect: Authority Semantics and Runtime Infrastructure for Stateful Agents

    [https://arxiv.org/abs/2609.08472](https://arxiv.org/abs/2609.08472)

    该论文揭示了“跨基底权限鸿沟”问题——即授权信息存在于智能体可见的工作区和记忆之外，并通过受控消融实验证明，向系统提供原始权限回执可使语义成功率从 0/32 提升至 32/32。

    

    智能体系统在变更工作区的同时会持久化模型可见的记忆，而运行时、注册表或审批服务可能在两者之外持有权限状态。此时，相同的最终文件可能需要截然相反的安全操作。我们将这一现象称为“跨基底权限鸿沟”：与决策相关的授权信息存在于规划器可见的工作区或记忆状态之外。在两个受控迷你基准家族上，三个实验使用真实的 Git 谱系、持久记录的智能体执行尝试、确定性预言机以及两条模型路线，对规划器观察增强与执行时权限检查进行了比较。实验 1 是一个 128 单元格的受控证据消融实验：不具备权限感知的候选证据最终语义成功率为 0/32，而原始回执和类型化关系均达到 32/32。缺失的权限事实解释了这一性能增益；在相同条件下，类型化封装相较于（基线）未观察到规划准确性的提升。

    arXiv:2609.08472v2 Announce Type: replace-cross  Abstract: Agentic systems persist model-visible memory while mutating workspaces, while a runtime, registry, or approval service may hold authority state outside both. Identical final files can then require opposite safe actions. We call this the cross-substrate authority gap: decision- relevant authorization information resides outside the planner-visible workspace or memory state. Across two controlled mini-benchmark families, three experiments compare planner-observation augmentation with an execution-time authority check using real Git lineage, durably recorded agent execution attempts, deterministic oracles, and two model routes. Experiment 1 is a 128-cell controlled evidence ablation: authority-blind candidate evidence obtains 0/32 final semantic success, while raw receipts and a typed relation both obtain 32/32. The missing authority fact accounts for the gain; typed packaging provides no observed planning-accuracy gain over equal
    
[^27]: 《扫描智能体框架：AI编码智能体配置暴露问题的仓库研究》

    Scanning the Harness: A Repository Study of Configuration Exposures in AI Coding Agents

    [https://arxiv.org/abs/2609.07360](https://arxiv.org/abs/2609.07360)

    本研究对3,171个公开GitHub仓库开展了首次大规模系统性分析，识别出AI编码智能体配置中的六类安全暴露问题，发现15.4%-17.9%的设置存在未固定MCP包版本、宽泛执行权限授予等供应链风险。

    

    AI编码智能体依赖于仓库指令、技能、钩子、工具服务器声明和子智能体定义。这些工件同时分发行为逻辑和对可执行依赖项的访问权限，使得配置审查成为智能体软件供应链的重要组成部分。我们研究了3,171个公开的GitHub仓库：包括2,660个组装配置的设置和511个技能集合。通过确定性分析、机械重推导、模型辅助审查和平台文档核查，我们识别出六类配置暴露和一致性问题。未固定的MCP包声明出现在9.8%的设置中，宽泛的执行权限授予占2.5%，宽泛的技能工具预批准占3.8%。这三者的并集覆盖了409个设置（15.4%）；在包含MCP配置的设置中，24.5%含有未固定的声明。若将必填字段缺失和技能格式问题纳入统计，设置的问题率升至17.9%，技能集合的问题率为6.8%。在保持六类问题固定不变的前提下，情境化的……（摘要原文在此处截断）

    arXiv:2609.07360v2 Announce Type: replace  Abstract: AI coding agents rely on repository instructions, skills, hooks, tool-server declarations, and subagent definitions. These artifacts distribute both behavior and access to executable dependencies, making configuration review part of the agent software supply chain. We study 3,171 public GitHub repositories: 2,660 assembled setups and 511 skill collections. Deterministic analysis, mechanical re-derivation, model-assisted review, and platform documentation checks identify six categories of configuration exposure and conformance issues. Unpinned MCP package declarations occur in 9.8% of setups, broad execution grants in 2.5%, and broad skill tool preapproval in 3.8%. Their union covers 409 setups (15.4%); among setups with MCP configuration, 24.5% contain an unpinned declaration. Including required-field and skill-format issues brings the setup rate to 17.9% and the collection rate to 6.8%. Holding the six categories fixed, contextual r
    
[^28]: 一种能力还是多种能力？检验前沿AI评估的经济效度

    One Capability or Many? Testing the Economic Validity of Frontier AI Evaluation

    [https://arxiv.org/abs/2608.29420](https://arxiv.org/abs/2608.29420)

    本研究通过潜变量模型对421个模型配置和12个基准的分析发现，经济基准测量的并非独立的独特能力，而是与其他基准共享的同一通用能力维度（单一因子解释74.5%的共同方差），从而质疑了前沿AI经济评估的构念效度。

    

    前沿模型排行榜如今基于经济基准对系统进行排名，这些基准测试模型执行专业任务的能力，涵盖从软件工程到银行工作流程等领域，而这些排名影响着组织的采购决策、监管机构的审查方向，以及对工作方式将如何变化的预期。这类基准究竟衡量的是一种有别于一般应试能力的独特能力，还是仅仅是重新表达了随着模型改进所有基准都会随之提升的同一维度，这是一个尚未被研究的构念效度问题。我们在一个固定哈希值的排行榜快照上对这一问题进行了检验，该快照包含421个模型配置和十二个基准测试（其中四个为经济类基准），将基准视为题目、模型视为被试，在一个潜变量模型中检验四个假设，且这些假设及其判定阈值均在分析前预先设定。结果表明，单一因子解释了74.5%的共同方差，并与模型发布日期高度相关（R² = 0.505），因此能力的主导轴在很大程度上是一个时间趋势。

    arXiv:2608.29420v1 Announce Type: new  Abstract: Frontier-model leaderboards now rank systems based on economic benchmarks, tests of how well models carry out professional tasks from software engineering to banking workflows, and those rankings inform what organisations buy, what regulators scrutinise, and expectations of how work will change. Whether such benchmarks measure a capability distinct from general test-taking, or re-express the one axis along which every benchmark rises as models improve, is a question of construct validity that has not yet been studied. We test it on a hash-pinned leaderboard snapshot of 421 model configurations across twelve benchmarks, four of them economic, treating benchmarks as items and models as respondents in a latent-variable model with four hypotheses and their thresholds fixed before analysis. A single factor explains 74.5% of common variance and tracks model release date (R^2 = 0.505), so the leading axis of capability is substantially a time t
    
[^29]: AgentDV：用于硬件设计验证的闭环智能体AI

    AgentDV: Closed-Loop Agentic AI for Hardware Design Verification

    [https://arxiv.org/abs/2608.27148](https://arxiv.org/abs/2608.27148)

    AgentDV是一个闭环智能体AI框架，通过可运行性过滤、CSR接地检查和覆盖率引导迭代，将LLM测试平台生成转化为可靠的RTL验证流水线。

    

    寄存器传输级（RTL）验证在现代片上系统（SoC）开发中占据了主要工作量。然而，近期基于LLM的验证代码生成往往无法产生可运行、设计一致且能产生覆盖率的测试平台。我们提出了AgentDV，一个用于自动化RTL验证环境生成的闭环智能体AI框架。AgentDV通过结合LLM引导的分析、测试平台构建、仿真、覆盖率测量和迭代优化，将单次LLM测试平台生成转化为一个基于工具的验证流水线。该框架引入了三个关键思想：1）可运行性过滤，拒绝无效的生成环境；2）基于CSR的检查，减少幻觉信号和错误预期行为；3）基于覆盖率的迭代，根据测量的验证差距重新生成测试。我们使用三个LLM在挑战性DUT和公开的OpenTitan外设上评估了AgentDV。

    arXiv:2608.27148v1 Announce Type: new  Abstract: Register-transfer level (RTL) verification consumes a major part of modern system-on-chip (SoC) development effort. Yet, recent LLM-based verification-code generation often fails to produce runnable, design-consistent, and coverage-producing testbenches. We present AgentDV, a closed-loop agentic AI framework for automated RTL verification environment generation. AgentDV transforms single-shot LLM testbench generation into a tool-grounded verification pipeline by combining LLM-guided analysis, testbench construction, simulation, coverage measurement, and iterative refinement. The framework introduces three key ideas: 1) runnability filtering to reject invalid generated environments, 2) CSR-grounded checking to reduce hallucinated signals and incorrect expected behavior, and 3) coverage-guided iteration to regenerate tests based on measured verification gaps. We evaluate AgentDV using three LLMs on challenge DUTs and public OpenTitan perip
    
[^30]: 过于自信而不安全：用于可靠日志异常检测的模型校准

    Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection

    [https://arxiv.org/abs/2608.17965](https://arxiv.org/abs/2608.17965)

    本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。

    

    在线日志异常检测对于维护大规模计算系统的可靠性至关重要。尽管基于语言模型的日志异常检测器取得了强大的检测性能，但其置信度估计仍校准不佳。我们表明，这些检测器经常对错误预测赋予过高的置信度，尤其是在严重类别不平衡下的异常日志中。此外，即使传统校准指标显示校准良好，错误预测的置信度仍持续偏高，这为运维监控系统造成了关键可靠性缺口。为解决此问题，我们提出了日志重建与距离（LoRD），一种轻量级的事后校准框架，用于可靠的日志异常检测。LoRD从正确分类的验证样本的潜在表示中学习预测路径特定的可靠性模型，并估计预测可靠性阈值。

    arXiv:2608.17965v1 Announce Type: cross  Abstract: Online log anomaly detection is critical for maintaining the reliability of large-scale computing systems. Although recent language model-based log anomaly detectors achieve strong detection performance, their confidence estimates remain poorly calibrated. We show that these detectors frequently assign excessive confidence to incorrect predictions, particularly for anomalous logs under severe class imbalance. Moreover, confidence on erroneous predictions remains persistently high even when conventional calibration metrics indicate good calibration, creating a critical reliability gap for operational monitoring systems. To address this issue, we propose Log Reconstruction and Distance (LoRD), a lightweight post-hoc calibration framework for reliable log anomaly detection. LoRD learns prediction-route-specific reliability models from latent representations of correctly classified validation samples and estimates prediction reliability th
    
[^31]: Grid-Orch：基于大语言模型的配电网仿真与分析编排器

    Grid-Orch: An LLM-Powered Orchestrator for Distribution Grid Simulation and Analytics

    [https://arxiv.org/abs/2605.12728](https://arxiv.org/abs/2605.12728)

    Grid-Orch通过MCP协议将大语言模型与OpenDSS配电网仿真相结合，提供36种领域工具，使工程师能用自然语言完成潮流计算、电压分析、QSTS仿真和自动化优化，并支持本地部署以满足电力系统安全隔离需求。

    

    摘要：预计到2030年，配电工程领域将面临高达150万名工程师的劳动力短缺，这使得对更易用的分析工具的需求变得十分迫切。本文提出了Grid-Orch，这是一个通过模型上下文协议（MCP）将大语言模型（LLM）与电力系统仿真相连接的框架，使工程师能够通过自然语言执行复杂的配电分析。以OpenDSS作为参考实现，Grid-Orch提供了涵盖十一个类别的36种领域专用工具，覆盖潮流计算、电压分析、准静态时间序列（QSTS）仿真以及自动化优化。其与供应商无关的LLM层同时支持云端托管模型（Gemini、Claude）和本地部署模型（Ollama、llama-cpp），从而能够为安全敏感的公用事业环境提供物理隔离（气隙）运行。三种优化技能——电容器选址、电压越限分析与过电压缓解，……（摘要在此处被截断）

    arXiv:2605.12728v2 Announce Type: replace-cross  Abstract: The power distribution engineering workforce faces a projected shortage of up to 1.5 million engineers by 2030, creating urgent demand for more accessible analysis tools. This paper introduces Grid-Orch, a framework that bridges Large Language Models (LLMs) and power system simulation through the Model Context Protocol (MCP), enabling engineers to perform complex distribution analyses via natural language. Using OpenDSS as the reference implementation, Grid-Orch provides 36 domain-specific tools across eleven categories, covering power flow, voltage analysis, quasi-static time series (QSTS) simulation, and automated optimization. A provider-agnostic LLM layer supports both cloud-hosted (Gemini, Claude) and locally deployed (Ollama, llama-cpp) models, enabling air-gapped operation for security-sensitive utility environments. Three optimization skills, capacitor placement, voltage violation analysis, and overvoltage mitigation, e
    
[^32]: VLAA-GUI：知道何时停止、恢复与搜索——一个模块化的GUI自动化框架

    VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation

    [https://arxiv.org/abs/2604.21375](https://arxiv.org/abs/2604.21375)

    VLAA-GUI提出一个模块化GUI自动化框架，通过强制性完整性验证器杜绝无视觉证据的过早成功宣告、多层级循环断路器打破重复失败循环、以及按需在线搜索应对不熟悉元素，系统性地解决了GUI智能体的过早停止与重复循环两大核心难题。

    

    自主GUI智能体面临两个根本性挑战：一是过早停止，即智能体在没有可验证证据的情况下过早宣布任务成功；二是重复循环，即智能体在没有恢复机制的情况下反复执行相同的失败动作。我们提出了VLAA-GUI，这是一个模块化的GUI智能体框架，围绕三个集成组件构建，用以指导系统何时停止、恢复和搜索。第一，一个强制性的完整性验证器在每个完成步骤强制执行UI可观察的成功标准和验证——借助一个智能体级验证器，利用决策规则对完成声明进行交叉审查，拒绝缺乏直接视觉证据的声明。第二，一个强制性的循环断路器提供多层级过滤：在重复失败后切换交互模式，在屏幕状态持续重现后强制改变策略，并将反思信号与策略转换绑定。第三，一个按需触发的搜索智能体在线搜索不熟悉的……（摘要原文在此处截断）

    arXiv:2604.21375v3 Announce Type: replace-cross  Abstract: Autonomous GUI agents face two fundamental challenges: early stopping, where agents prematurely declare success without verifiable evidence, and repetitive loops, where agents cycle through the same failing actions without recovery. We present VLAA-GUI, a modular GUI agentic framework built around three integrated components that guide the system on when to Stop, Recover, and Search. First, a mandatory Completeness Verifier enforces UI-observable success criteria and verification at every finish step -- with an agent-level verifier that cross-examines completion claims with decision rules, rejecting those lacking direct visual evidence. Second, a mandatory Loop Breaker provides multi-tier filtering: switching interaction mode after repeated failures, forcing strategy changes after persistent screen-state recurrence, and binding reflection signals to strategy shifts. Third, an on-demand Search Agent searches online for unfamilia
    
[^33]: 异步多方会话类型中的混合选择

    Mixed Choice in Asynchronous Multiparty Session Types

    [https://arxiv.org/abs/2602.23927](https://arxiv.org/abs/2602.23927)

    该论文提出了一个支持异步混合选择的多方会话类型框架，其核心构造允许分布式参与者间协议状态暂时不一致但保证最终达成一致，并通过进展性与操作对应性证明了正确性，同时实现了用于规范验证和Erlang/OTP协议编程的实用工具链，并以RabbitMQ的amqp_client为案例进行了验证。

    

    我们提出了一种支持异步混合选择（MC）的多方会话类型（MST）框架。我们为混合选择提出了一种核心构造，该构造允许分布式参与者之间的协议状态出现暂时性不一致，但确保所有参与者始终能够最终达成相互一致的状态。我们通过建立进展性（progress）性质以及全局类型与分布式局部类型投影之间的操作对应关系，证明了系统的正确性。基于该理论，我们实现了一个实用的工具链，用于规范和验证具有混合选择的异步MST协议，并在Erlang/OTP中编写符合规范的gen_statem进程。我们通过使用该工具链对Erlang版RabbitMQ代理的amqp_client的一部分进行规范化和重新实现，对框架进行了测试。

    arXiv:2602.23927v2 Announce Type: replace-cross  Abstract: We present a multiparty session type (MST) framework with asynchronous mixed choice (MC). We propose a core construct for MC that allows transient inconsistencies in protocol state between distributed participants, but ensures all participants can always eventually reach a mutually consistent state. We prove the correctness of our system by establishing a progress property and an operational correspondence between global types and distributed local type projections. Based on our theory, we implement a practical toolchain for specifying and validating asynchronous MST protocols featuring MC, and programming compliant gen_statem processes in Erlang/OTP. We test our framework by using our toolchain to specify and reimplement part of the amqp_client of the RabbitMQ broker for Erlang.
    
[^34]: 基于大语言模型的项目级漏洞检测：一项实证研究

    LLM-based Vulnerability Detection at Project Scale: An Empirical Study

    [https://arxiv.org/abs/2601.19239](https://arxiv.org/abs/2601.19239)

    本研究首次对项目级LLM漏洞检测器进行了大规模实证评估，发现通用智能体召回率最高但难以应对复杂代码，同时揭示了基于LLM的方法与传统方法在误报和不完备性方面的关键局限。

    

    随着软件复杂度的不断增长，自动化漏洞检测变得日益重要。近期基于大语言模型（LLM）的检测器将语义推理与静态分析相结合以实现项目级扫描，但其实际有效性和失败原因仍不清楚。我们首次对项目级的专用LLM检测器开展了全面的实证研究，在265个已知C/C++和Java漏洞以及24个活跃的开源项目/模块上，评估了五种专用方法、两种通用智能体和四种传统静态分析器。通过采用Codex辅助标注并结合分层人工验证，我们分析了6,442条抽样警告，并基于从355份人工检查报告中归纳出的分类体系，对5,896条误报进行了分类。本研究得出三项发现：首先，通用智能体取得了最高的召回率，但在处理复杂代码时表现不佳，而基于LLM的方法和传统方法都受到不完备性问题的困扰。

    arXiv:2601.19239v2 Announce Type: replace  Abstract: As software complexity grows, automated vulnerability detection becomes increasingly important. Recent LLM-based detectors combine semantic reasoning with static analysis for project-scale scanning, but their practical effectiveness and failure causes remain unclear. We present the first comprehensive empirical study of specialized LLM-based detectors at project scale, evaluating five specialized methods, two general-purpose agents, and four traditional static analyzers on 265 known C/C++ and Java vulnerabilities and 24 active open-source projects/modules. Using Codex-assisted labeling with stratified manual validation, we analyze 6,442 sampled warnings and classify 5,896 false positives using a taxonomy derived from 355 manually inspected reports. Our study yields three findings. First, general-purpose agents achieve the highest recall but struggle with complex code, while both LLM-based and traditional methods suffer from incomplet
    
[^35]: BabelCoder：基于规范对齐的智能体化代码翻译

    BabelCoder: Agentic Code Translation with Specification Alignment

    [https://arxiv.org/abs/2512.06902](https://arxiv.org/abs/2512.06902)

    提出BabelCoder，一个将代码翻译任务分解为翻译、测试和精炼等多个专门智能体协同工作的智能体化框架，通过规范对齐显著提升了多语言代码翻译的准确性和质量。

    

    随着软件系统的不断演进，开发者越来越多地需要跨多种编程语言工作，并经常面临将代码从一种语言迁移到另一种语言的需求。尽管自动代码翻译提供了一种有前景的解决方案，但它长期以来一直是一项具有挑战性的任务。大型语言模型（LLM）的最新进展在这一任务上展现出了潜力，然而现有方法在准确性方面仍然有限，且未能有效利用代码中的上下文和结构线索。先前的工作已经探索了翻译和修复机制，但缺乏一个结构化的、智能体化的框架来让多个专门的智能体协同提升翻译质量。在本工作中，我们提出了BabelCoder，这是一个智能体化框架，它通过将代码翻译任务分解为多个专门智能体来执行，包括翻译、测试和精炼等环节，每个智能体负责特定方面的工作，例如生成代码、验证正确性或修复错误。

    arXiv:2512.06902v2 Announce Type: replace  Abstract: As software systems evolve, developers increasingly work across multiple programming languages and often face the need to migrate code from one language to another. While automatic code translation offers a promising solution, it has long remained a challenging task. Recent advancements in Large Language Models (LLMs) have shown potential for this task, yet existing approaches remain limited in accuracy and fail to effectively leverage contextual and structural cues within the code. Prior work has explored translation and repair mechanisms, but lacks a structured, agentic framework where multiple specialized agents collaboratively improve translation quality. In this work, we introduce BabelCoder, an agentic framework that performs code translation by decomposing the task into specialized agents for translation, testing, and refinement, each responsible for a specific aspect such as generating code, validating correctness, or repairi
    
[^36]: SLMFix：利用强化学习的小语言模型修复领域特定语言错误

    SLMFix: Leveraging Small Language Models for Domain Specific Language Error Fixing with Reinforcement Learning

    [https://arxiv.org/abs/2511.19422](https://arxiv.org/abs/2511.19422)

    该论文提出SLMFix流水线，利用强化学习微调的小语言模型根据解释器反馈修复LLM生成代码中的语法错误，在低资源编程语言上使验证器通过率提升了40%。

    

    大语言模型（LLMs）在多种编程语言的代码生成方面展现出了令人瞩目的能力，但即使是最先进的LLM也会生成包含语法错误的程序，无法完成给定的任务，尤其是在低资源编程语言（LRPLs）上。此外，高昂的训练成本使得计算资源受限的用户无法负担LLM的微调，进一步削弱了LLM在代码生成方面的有效性。在这项工作中，我们提出了SLMFix，一种新颖的代码生成流水线，它利用通过强化学习（RL）技术微调的小语言模型（SLM），基于解释器反馈来修复LLM生成的领域特定语言（DSLs）程序中的语法错误。我们的实验结果证明了该方法在多种DSL上的有效性和泛化能力，在低资源编程语言上将验证器通过率提高了40%，并消除……（原文摘要在此处截断）

    arXiv:2511.19422v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) have shown impressive capabilities in code generation across many programming languages but even state-of-the-art LLMs generate programs that contain syntactic errors and fail to complete the given tasks, especially for low-resource programming languages (LRPLs). In addition, the high cost of training makes finetuning LLMs unaffordable for those with constrained computational resources, further weakening the effectiveness of LLMs for code generation. In this work, we propose SLMFix, a novel code generation pipeline that leverages a small language model (SLM) finetuned using reinforcement learning (RL) techniques to fix syntactic errors in LLM-generated programs for domain-specific languages (DSLs) based on interpreter feedback. Our experimental results demonstrate the effectiveness and generalizability of our approach across multiple DSLs, improving the validator pass rates by 40% on LRPLs and elimi
    
[^37]: 人工智能驱动发现中统计严谨性的结构性保障：一种函数式架构

    Structural Enforcement of Statistical Rigor in AI-Driven Discovery: A Functional Architecture

    [https://arxiv.org/abs/2511.06701](https://arxiv.org/abs/2511.06701)

    该论文提出一种函数式架构，通过Haskell的Research monad、声明式脚手架、操作系统级沙箱以及机器验证的Lean 4形式化LORD在线FDR控制，从结构上强制保证AI驱动科学发现中的统计严谨性，防止AI科学家系统因不受控的多重检验而产生虚假发现。

    

    AI-Scientist系统存在通过不受控的多重检验而制造虚假发现的风险。我们提出一种在两个层面强制统计严谨性的功能架构：一是Haskell嵌入式领域专用语言（即Research monad），它使得在不更新错误预算的情况下无法检验假设；二是声明式脚手架，用于固定数据流与统计检验方法；此外还配有操作系统级沙箱，使验证数据在LLM生成代码的运行环境中物理上不可见。我们将FDR（错误发现率）控制视为形式化需求，并将其追溯至具体实现。我们以机器验证的Lean 4形式化为基础来支撑该设计，形式化了LORD在线错误发现率控制：我们推导了其错误预算，证明了边际FDR控制，以及当阈值不随先前拒绝结果自适应调整时的完全FDR控制。随后，我们在SPARK/Ada中验证了以IEEE 7（54浮点标准计算的LORD阈值）……

    arXiv:2511.06701v4 Announce Type: replace-cross  Abstract: AI-Scientist systems risk manufacturing spurious discoveries through uncontrolled multiple testing. We present a functional architecture that enforces statistical rigor at two levels: a Haskell embedded domain-specific language (the Research monad) that makes it impossible to test a hypothesis without updating the error budget, and a declarative scaffold that fixes the data flow and the statistical test, together with an OS-level sandbox that makes validation data physically absent from the environment in which LLM-generated code runs. We treat FDR control as a formal requirement and trace it to the implementation. We ground the design in a machine-checked Lean~4 formalization of LORD online false-discovery-rate (FDR) control: we derive its error budget and prove marginal FDR control, and full FDR control when thresholds do not adapt to earlier rejections. We then verify in SPARK/Ada that the LORD thresholds, computed in IEEE~7
    
[^38]: 测试代码中自承认技术债务的首次审视：分类体系与检测

    A First Look at the Self-Admitted Technical Debt in Test Code: Taxonomy and Detection

    [https://arxiv.org/abs/2510.22409](https://arxiv.org/abs/2510.22409)

    该研究首次系统性考察测试代码中的自承认技术债务（SATD），通过人工分析1,000个开源Java项目的5万条注释构建了包含11个类别的SATD分类体系，并评估了现有检测工具和大语言模型自动检测此类债务的能力。

    

    自承认技术债务（SATD）是指开发人员在注释中明确承认代码问题、变通方案或次优解决方案的情况。众所周知，SATD会显著增加软件维护的工作量。尽管已有大量研究考察了源代码中的SATD，但其在测试代码中的存在与影响尚未得到专门关注，这导致我们对SATD在测试场景中如何表现的理解存在重大空白。本研究通过人工分析从1,000个开源Java项目的160万条注释中随机抽取的50,000条注释，对测试代码中的SATD进行了调查研究。经过人工分析和筛选，我们从样本中识别出615条SATD注释，并将其划分为11个不同的类别，构建了测试代码SATD的分类体系。为探究测试代码中的SATD能否被自动检测，我们评估了现有的SATD检测工具，以及开源和专有的大语言模型（LLM）……

    arXiv:2510.22409v3 Announce Type: replace  Abstract: Self-admitted technical debt (SATD) refers to comments in which developers explicitly acknowledge code issues, workarounds, or suboptimal solutions. SATD is known to significantly increase software maintenance effort. While extensive research has examined SATD in source code, its presence and impact in test code have received no focused attention, leaving a significant gap in our understanding of how SATD manifests in testing contexts.   This study investigates SATD in test code by manually analyzing 50,000 comments randomly sampled from 1.6 million comments across 1,000 open-source Java projects. From this sample, after manual analysis and filtering, we identified 615 SATD comments and classified them into 11 distinct categories, building a taxonomy of test code SATD. To investigate whether test code SATD can be detected automatically, we evaluated existing SATD detection tools, as well as both open-source and proprietary LLMs. Amon
    

