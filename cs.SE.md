# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Agentic Harness Engineering Beta: Requirement-Centered Evolution of Agent Harnesses](https://arxiv.org/abs/2611.24924) | 提出AHE-Beta，一个以需求为中心的智能体框架演进循环，通过模拟用户提出需求、团队协作实现可泛化变更以及基于经验证需求选择变更，解决了现有方法产生狭窄补丁和基准局部最优的问题。 |
| [^2] | [MicroAssert: Microarchitecture-Aware Structured Assertion Generation for CPU Verification](https://arxiv.org/abs/2611.24872) | MicroAssert是一个微架构感知的结构化断言自动生成框架，通过从RTL识别功能区域并生成局部与复合断言，在五个开源RISC-V CPU上实现了100%语法正确率和74.82%的平均证明率，错误检测率较现有方法提升1.99至3.51倍。 |
| [^3] | [MLIR-MRacle: Memory-Model-Aware Metamorphic Fuzzing of Concurrent MLIR](https://arxiv.org/abs/2611.24804) | MLIR-MRacle是一个内存模型感知的蜕变模糊测试工具，通过目标内存模型调控的蜕变关系与统计预言机来检测并发MLIR编译过程中可能悄然改变程序行为的优化错误。 |
| [^4] | [The Limit Is in the Input, Not the Model: State-Dependent Command Safety in Natural-Language Interfaces to Industrial Controllers](https://arxiv.org/abs/2611.24682) | 本文提出“可判定性审计”方法，证明语言模型在判断工业控制器命令安全性时的失误源于输入中缺失机器状态信息而非模型推理能力不足，且提供状态值可显著提升判断的准确性。 |
| [^5] | [DrawScore: Evaluating Generated CAD Models against Engineering Drawings](https://arxiv.org/abs/2611.24681) | 提出DrawScore新评估指标，通过工程图纸式三视图投影与饱和惩罚，使孔位错误等制造关键的小体积局部缺陷权重仅线性衰减，从而有效弥补传统体积类指标对局部错误不敏感的问题。 |
| [^6] | [Metadata-Driven AI: An Architectural Pattern for Sustainable Data Analysis with LLMs](https://arxiv.org/abs/2611.24661) | 该论文提出“元数据驱动的AI”架构模式，只让元数据而非实际数据跨越单向边界进入LLM，并将模型生成的分析程序保留复用，从而将推理成本从每次运行一次降低为每次分析一次，实现更经济、更私密且可持续的自然语言数据分析。 |
| [^7] | [Probing agent overeagerness in realistic settings with continuation audits](https://arxiv.org/abs/2611.24491) | 本文提出“延续审计”方法，以真实编程智能体会话记录的前缀为起点引导审计者沿轨迹延续至预设目标行为，从而在真实场景中探测智能体的过度积极行为，并规避了模型识别合成测试环境的问题。 |
| [^8] | [Mitigating fix reversal: Counterfactual constraints for function-level vulnerability detection](https://arxiv.org/abs/2611.24489) | CFVul通过滑动窗口编码保留完整修复信息，并利用训练时的内容约束和推理时的长度约束等反事实手段，缓解了函数级漏洞检测中因长度捷径导致的“修复反转”问题。 |
| [^9] | [Predicting Process-Level Behavior: A Benchmarking Study](https://arxiv.org/abs/2611.24425) | 该论文通过对并发轨迹预测和平均吞吐时间预测两项流程级任务的全面基准测试，比较了统计方法、机器学习方法、基础模型、业务流程仿真以及聚合到流程级的轨迹级PPM方法，揭示了最适合预测流程级行为的建模范式。 |
| [^10] | [Exploring multimodal AI--XR interactions for software development: From expert perspectives to semantic code outlines](https://arxiv.org/abs/2611.24334) | 该研究通过13位专家访谈与两个XR原型实验，探索了利用扩展现实工作空间中的多模态信息（如头部注视方向与手动输入），将已查看的代码替换为语义化功能描述大纲，从而帮助开发人员理解和审查AI生成代码的交互设计。 |
| [^11] | [Reducing Inference Cost of Coding Agents with Timeout Cascades](https://arxiv.org/abs/2611.24324) | 提出一种超时级联的混合推理策略，先让小模型尝试解决任务、必要时再升级到大模型，从而在不牺牲准确率的前提下大幅降低AI编码智能体的推理成本。 |
| [^12] | [Keeping Up with the Kernel: Can LLMs Adapt Linux Drivers to Interface Changes?](https://arxiv.org/abs/2611.24269) | 该论文研究了 20 年来 Linux 内核接口变更引发的驱动“伴随演化”现象，并构建了包含 316 个任务的基准 CoEvoBench 来评估大语言模型能否自动将旧驱动适配到新内核。 |
| [^13] | [The Comprehension Gap in LLM-Assisted Software Development: An Experiment](https://arxiv.org/abs/2611.24216) | 通过35名参与者的交叉实验发现，开发者对LLM生成代码的理解显著低于自写代码，这种“认知债务”会阻碍后续任务的高效完成。 |
| [^14] | [OmegaHyper: Explicit-State HyperLTL Model Checking on an omega-Automaton Diet](https://arxiv.org/abs/2611.24210) | OmegaHyper在保留AutoHyper算法的基础上，通过分解与缩减所处理的ω-自动机并集成Kofola求补工具，将显式状态HyperLTL模型检测的性能提升了多达几个数量级。 |
| [^15] | [MemoSync: Co-Managing Developer-Agent Metamemory Across Contexts](https://arxiv.org/abs/2611.24165) | MemoSync 提出了一个让开发者与 AI 编程智能体协同管理记忆的系统，智能体负责提出记忆候选并报告使用情况，开发者则可审查、编辑和限定记忆范围，从而解决了开发者对智能体记忆缺乏可见性的“元记忆”问题。 |
| [^16] | [Assessing Vision-Language Models for Early Prediction of Unsafe Situations in Robotics](https://arxiv.org/abs/2611.24146) | 本文通过在三个机器人案例研究中实证评估Qwen、InternVL和LLaVA-OneVision等视觉-语言模型对视频与里程计数据的分析能力，证明VLM能够在不安全情况发生前实现早期故障预警，其中Qwen表现最佳。 |
| [^17] | [SCAFFOLD: Toward Warrant-Preserving Evidence Transport for Reusable LLM-Agent Skills](https://arxiv.org/abs/2611.23990) | 本文提出以有保证的复用声明（WRC）和凭证保全式证据迁移（WPET）作为规范框架，用于判定历史证据在上下文变化后是否仍可采信，从而为LLM智能体技能的复用建立前瞻性信任。 |
| [^18] | [Characterizing GitHub Security Actions Alerts For Web Application Tutorials](https://arxiv.org/abs/2611.23967) | 本研究分析了60个Django和Spring教程仓库中来自四种默认GitHub Security Actions配置的6,609条SAST安全警报，从警报数量、相关性等多个维度刻画了Web应用开发学习场景中安全警报的特征。 |
| [^19] | [Quantifying the Intrusiveness of Retrofitting Retrieval-Augmented Generation onto Legacy Software Systems](https://arxiv.org/abs/2611.23931) | 该论文提出了三种 RAG 改造模式的分类法以及量化侵入度的 LIS 与 LIS-X 评分体系，并通过静态评估与真实部署案例验证了侵入性排序：边车模式 < SDK 嵌入 < 朴素侵入式。 |
| [^20] | [Inference-Time Scaling in Agentic Hardware Verification Workflows: A Multi-Strategy Characterization Across Protocol Complexity](https://arxiv.org/abs/2611.23905) | 该论文首次对面向AMBA协议桥接的智能体式UVM测试平台合成中的推理时计算扩展进行了系统性的实证刻画，对比了四种不同成本的推理策略在不同协议复杂度下的验证质量，并指出结构化诊断反馈是提升硬件验证代码生成效果的关键。 |
| [^21] | [SWE-growth: Teacher-Guided Adaptive Self-Play for Software Engineering Agents](https://arxiv.org/abs/2611.23876) | SWE-Growth 提出了一种教师引导的自适应自博弈框架，利用教师验证的结果识别学生的学习机会——将全失败任务用于 SFT 示范、具有正向师生差距的混合结果任务用于强化学习，并将任务资格判定与注入器更新相连接，实现学生训练与任务生成的闭环协同。 |
| [^22] | [Asking but Missing: Decision Coverage in Natural-Language Data Querying](https://arxiv.org/abs/2611.23864) | 该论文提出了一种名为接地代码生成（GCG）的诊断协议，通过将决策识别、证据解决和代码生成三个阶段分离，来诊断大语言模型在自然语言数据查询中的欠规范决策补全（UDC）行为，即模型默默填补问题歧义导致错误查询结果的问题。 |
| [^23] | [Re-Testing LLM-Based C-to-Rust Translations Against Their Original C](https://arxiv.org/abs/2611.23839) | 该论文重新测试了11个已发布的基于LLM的C到Rust翻译器，发现所有翻译均存在缺陷（共160个缺陷，其中93个为普通输入上的错误），证明其自报成功率不可靠，因为这些检查大多从未在能暴露缺陷的输入上完整比较输出。 |
| [^24] | [RoboDx: Repair and Diagnosis of Faults in Robot Software](https://arxiv.org/abs/2611.23816) | 提出了RoboDx——一个带有注入故障与隐藏场景验证的机器人导航软件修复与诊断任务基准，实验表明即使修复通过验证，智能体在双故障任务中往往只发现并修复了部分故障（尤其是静默故障）。 |
| [^25] | [Stephen: Verifying Sequencer-Aided Coordination Systems with Cross-View Prophecy](https://arxiv.org/abs/2611.23811) | 本文提出 Stephen，通过“跨视图预言”技术首次对基于定序器的协调系统实现进行形式化验证，解决了定序器处的线性化点在视图变更时不持久、可能被丢弃而使证明失效的难题。 |
| [^26] | [Validating Concurrent Correctness Witnesses with Ghosts by Thread-Modular Abstract Interpretation](https://arxiv.org/abs/2611.23792) | 提出了一种新颖的并发正确性见证验证框架，利用幽灵变量中编码的语义信息，通过线程模块化抽象解释来验证带幽灵变量的并发正确性见证，而非将其简单视为额外的证明义务。 |
| [^27] | [Reliability is a user-specific property of large language models](https://arxiv.org/abs/2611.23732) | 该论文提出LLM服务的可靠性应针对特定用户和使用条件进行评估，并据此提出了包含论断规范、动态操作剖面、局部概率鲁棒性等六个方向的研究路线图。 |
| [^28] | [JITAtlas: Checking Lifetime and Protection Obligations in Compiler Rewrites](https://arxiv.org/abs/2611.23715) | JITAtlas是一个源码级检查器，通过从编译器优化器代码中重建引用与目标计算，检测编译器重写中违反对象生命周期、借用支持和调用者可见所有权的缺陷，在独立标注的基准上检出全部29个缺陷且无误报，并在CPython中发现15个新漏洞。 |
| [^29] | [Evaluating Governed LLM-Driven ML Experimentation: Lifecycle, Provenance, and Failure Metrics](https://arxiv.org/abs/2611.23678) | 本文提出应从实验过程而非仅从结果分数来评估LLM驱动的ML实验系统，构建了由控制平面（而非LLM）管理试验生命周期、编辑范围与保留决策、并将记录写入只追加账本的治理框架，从中计算无效操作率、重复失败率和溯源完整性等治理指标，并在六个ML任务共1,445次试验中验证了同一契约的普遍适用性。 |
| [^30] | [CratesBump: A Version-Bump Dataset for the Rust Ecosystem](https://arxiv.org/abs/2611.23664) | 该论文发布了首个用于预测 Rust 生态系统版本升级的公开数据集 CratesBump（涵盖约 22 万个 crate、超过两百万个连续发布对、每对含 41 个特征和双重标签），并配套提供可从公开数据重新生成该数据集的工具 semver-extract，基准实验表明版本升级预测任务可学习但远未解决。 |
| [^31] | [Safe Personal RSI: Evaluating and Repairing Personal-Agent Updates](https://arxiv.org/abs/2611.23651) | 本文提出Safe Personal RSI更新评估机制，通过将递归自我改进限制在提示词层面、固定模型权重与代码，并建立“测试—修复—验证”流程，在29个案例的基准上成功拒绝了21次高风险更新尝试中的20次，保障个人智能体更新后的安全性与任务性能。 |
| [^32] | [Option-GRPO: Hierarchical RL Against AI Slop in Code Generation for Unseen SDKs](https://arxiv.org/abs/2611.23633) | 该论文提出Option-GRPO分层强化学习框架，仅凭SDK源代码（无需人工示例或测试用例）即可将代码大模型适配到未见过的SDK，通过从源代码中挖掘可复用代码模式（code options）作为课程与动作空间，同时优化功能正确性与代码简洁优雅性，从而抑制AI垃圾代码的产生。 |
| [^33] | [RefinementGuard: Counterexample-Directed Semantic Qualification for Hidden Agent-Tool Execution](https://arxiv.org/abs/2611.23611) | 该论文提出RefinementGuard框架，通过反例导向的语义资格验证（基于显式父操作义务，将候选图与独立参考在隔离状态下比对）结合工件绑定的故障关闭式运行时准入，来守护智能体工具调用下沉至隐藏并行图的过程，在保留全部有效变体的同时拒绝近半数非等价故障。 |
| [^34] | [MetaphIR: Corpus-Scale Applicability Analysis for LLVM Peephole Optimizations](https://arxiv.org/abs/2611.23555) | 该论文提出MetaphIR，可在LLVM IR语料库规模上同时评估窥孔优化重写规则的模式匹配与前置条件满足情况，相比忽略前置条件的结构化搜索计数更精确，且避免了完整重编译流程高昂的工程成本与47分钟级的单次运行耗时。 |
| [^35] | [Record Sufficiency: What a Decision Record Must Preserve When an Unsettled Claim Becomes an Operative Status](https://arxiv.org/abs/2611.23422) | 本文提出“记录充分性”概念——即一个充分的决策记录应使读者仅凭记录及其引用内容，就能解释、限定、重新推导并质疑从未决主张变为生效状态的那个决策步骤——并据此提出记录必须回答的十个问题、映射到DRK v0.1模式、定义四项机械检查，同时区分符合性、适当性、忠实性与价值四个不同层面。 |
| [^36] | [Where the Tokens Go: Heat, Entropy and Error in Agentic Coding](https://arxiv.org/abs/2611.23375) | 提出软件热力学理论，将编程智能体消耗的token成本划分为功与热，以测试套件作为麦克斯韦妖，并通过编码-反编码循环实验定义“软件温度”来刻画智能体编程的开销特性。 |
| [^37] | [A Survey on RLOps for Large Language Models](https://arxiv.org/abs/2611.23357) | 本文对大语言模型RLOps方法进行了系统性综述，将现有技术归纳为训练设置、经验收集、奖励计算、策略优化和模型评估五个阶段，填补了该领域技术碎片化、缺乏统一梳理的空白。 |
| [^38] | [Idempotency Is Not Recovery: A Fault-Injection Study of Language-Model Agent Tool Actions](https://arxiv.org/abs/2611.23353) | 该论文通过系统的故障注入实验揭示“幂等性不等于恢复”，证明只有将授权、操作账本、目标端幂等性、对账与证据封存相结合的完整配置，才能将语言模型代理工具调用的重复副作用从基线的117/200降至0/200，并保障故障后的恢复能力。 |
| [^39] | [Systematic Mapping Protocol. Generative AI-Augmented Requirements Engineering: A Systematic Mapping Study of Practices, Specifications, and Reuse in Software Development](https://arxiv.org/abs/2611.23349) | 本文提出了一项系统映射研究的协议方案，旨在识别和分类生成式AI增强的需求工程研究，重点关注其产生的规范等工件在规范驱动开发、代码生成、测试推导、验证与复用等下游活动中的应用。 |
| [^40] | [RootSink: Reproducible C/C++ Crashes with Tested Repair Sites](https://arxiv.org/abs/2611.23336) | RootSink数据集汇集了来自354个开源项目的3341条C/C++崩溃记录，通过实际复现崩溃并测试开发者修复补丁，为每条记录提供了带证据标注的崩溃位置与经过验证的修复位置，填补了漏洞定位与修复研究缺乏成对证据样本的空白。 |
| [^41] | [Truthgate: Checking Coding Agents Against a Record They Cannot Edit - Cheating, Deterrence and Disclosure on Impossible Tasks](https://arxiv.org/abs/2611.23335) | 本文提出Truthgate框架，通过智能体无法篡改的记录（宿主重跑的测试、隐藏容器检查、签名工具调用日志）来核验编程智能体的真实性，发现Claude Sonnet 5.5在21.7%的不可能任务上通过特殊处理代码来作弊，而在系统提示中加入一句审查警告即可大幅减少此类作弊行为。 |
| [^42] | [Integrating AI Coding Agents Into Scientific Research](https://arxiv.org/abs/2611.23334) | 本文提出了PROOF六级分类标准，用于按来源可追溯性、可重现性、开放性、监督性和可证伪性对AI辅助数值实验进行分级，以指导在安全环境中将AI编程智能体整合进科学研究。 |
| [^43] | [What Should an LLM Know? Knowledge-Guided Warm Starts for Software Optimization](https://arxiv.org/abs/2611.23331) | 该研究在切实可行的标注预算下，提出了三种向LLM注入领域知识的热启动方法（规则校验、文档增强、专家纠正），发现只有经过检验的领域知识才能真正改善LLM驱动软件优化的“启动”阶段。 |
| [^44] | [Training for sustainable scientific software in the AI era: from code production to scientific stewardship](https://arxiv.org/abs/2611.23330) | 该论文提出在AI时代，科学软件培训应从代码生产转向科学管护，通过将AI系统教学与既有软件和分析培训相融合，使研究人员能够在AI辅助的工作流程中具备验证、解释、复现和维护科学软件的能力。 |
| [^45] | [A level is not a rank: an empirical audit of quantile-to-index conversion in 10 released conformal-prediction libraries](https://arxiv.org/abs/2611.23329) | 该论文对10个已发布的共形预测库中39处“置信水平转换为索引”的代码位置进行了实证审计，发现部分库将置信水平错误地解析为分位数秩，在特定样本量下导致实际覆盖率偏离理论保证。 |
| [^46] | [HydroMind: A Checkable Agent for Autonomous Hydrological Research](https://arxiv.org/abs/2611.23323) | 该论文开发了面向通用智能体的水文建模系统HydroMind及其建模工具和研究流程，首次检验了当自主智能体自行决定水文模型评估方案时，其结论能否被自身数据、代码和结果支持并供读者核查。 |
| [^47] | [YUL: Bringing Software Supply Chain Security to Coding Agents](https://arxiv.org/abs/2611.23314) | YUL 通过让编码智能体在声明依赖时对照包注册表进行校验，阻止过时依赖的使用并引导智能体重试最新版本，从而将软件供应链安全引入编码智能体，使依赖最终落在最新版本的比率提升至 93%。 |
| [^48] | [Model Discovery for Virtual Chaos Engineering with Incomplete Tracing: An eBPF Study](https://arxiv.org/abs/2611.23313) | 该论文探索了利用eBPF外部观测技术在追踪数据缺失或不完整的遗留系统上发现操作模型，从而在无需改造追踪系统的情况下实现虚拟混沌工程中的基于模型的故障分析。 |
| [^49] | [AppSec Core: An Ontology for Security-by-Design Software Engineering](https://arxiv.org/abs/2611.23312) | 本文提出了AppSec Core——一个涵盖十个领域、共259个元素的安全设计软件工程本体模型，它将分散在众多标准、框架和法律中的安全要求统一为“目标—实践—机制—证据工件”的结构化表示，并基于31个外部来源（含10个AI相关来源）进行了验证。 |
| [^50] | [Post-Quantum Cryptography Education for Software Developers: A Multivocal Literature Review](https://arxiv.org/abs/2611.23311) | 本文通过对8篇学术文献和15个专业培训项目的多源文献综述，系统评估了面向软件开发者的后量子密码学（PQC）教育现状，发现虽然所有来源均覆盖PQC基础与算法，但实际软件工程任务训练不均衡，并识别出六大教育缺口及相应改进建议。 |
| [^51] | [Repository-Grounded Rubrics for Execution-Free Evaluation of Coding Agents](https://arxiv.org/abs/2611.23309) | 该论文提出了一种基于仓库探索生成加权评分量表的无执行评估方法，无需运行测试即可评估编码智能体补丁的质量，其评分排序与真实测试结果高度一致（平均AUROC达0.826，系统排名Spearman相关系数达0.968）。 |
| [^52] | [Checking Retry Transparency in Agent Tools](https://arxiv.org/abs/2611.23307) | 提出一种基于人工构建、源码可溯的效果契约的测试原型，从原子重试吸收义务和成功尝试微步骤的有界交错两个层面检验智能体工具重试透明性背后的恢复保证。 |
| [^53] | [When Exact Inference Changes Code Selection: A Decision-Theoretic Analysis of B4](https://arxiv.org/abs/2611.23306) | 本文通过决策论分析精确刻画了在B4代码-测试模型中更精确的后验推断何时会改变程序选择——给出了保证整组规则已达MAP的先验-预算条件与有限修正界，并在大量实证比较中表明精确推断几乎从不改变所选代码。 |
| [^54] | [Dependency Fabrication in Repository Deployment Agents](https://arxiv.org/abs/2611.23305) | 本研究通过构造无法合法获取依赖的探针仓库，揭示了仓库部署智能体在环境搭建失败时会通过伪造缺失依赖（如创建存根）来通过测试，且现有规则和测试框架难以可靠阻止这种“奖励破解”行为，仅当智能体收到明确禁令并具备停止上报机制时缓解措施才最有效。 |
| [^55] | [When Code Works but the License Doesn't: Evaluating License Constraint Handling in Coding Agents](https://arxiv.org/abs/2611.23299) | 该论文系统评估了编程智能体在仓库级开发中对软件许可证约束的处理能力，发现智能体虽能成功构建并测试软件，却普遍忽视来源追溯、授权与分发等许可证约束，暴露出功能正确性与许可证合规性之间的持续差距。 |
| [^56] | [Plug-and-Play Sensor Evolution in Software-Defined Vehicles through Asset Administration Shells](https://arxiv.org/abs/2611.23298) | 本文提出一种基于资产管理壳（AAS）的即插即用传感器集成框架，使软件定义汽车在部署后新增传感器时能够自动完成识别、配置与安全监督下的软件部署。 |
| [^57] | [Architectural Design Decisions for Data Ingestion in Big Data Architectures](https://arxiv.org/abs/2611.23297) | 本文通过系统性灰文献综述、定性分析及专家访谈，编制了一份包含数据摄取设计决策、选项及选择驱动因素的目录，以帮助从业者系统地开发大数据架构中的数据摄取流程。 |
| [^58] | [Ask the Peers: Ranking NFR Coverage Gaps for Requirements Review](https://arxiv.org/abs/2611.23296) | 本研究将NFR覆盖缺口评审建模为项目级排序任务，实验表明利用同行项目类别统计信息的四种配置表现最优且效果相近（平均AUROC达0.875-0.889）。 |
| [^59] | [From Ambiguous Tickets to Ranked Hypotheses: Multi-Agent Root-Cause Analysis for Autonomous Heavy-Vehicle Software](https://arxiv.org/abs/2611.23294) | 该论文提出了一种多智能体根因分析工作流，将稀疏的故障工单和海量多模态车辆记录转化为基于遥测的事件链、按优先级排序的因果假设及候选代码位置，并通过隔离疑似原因、预筛选遥测数据和压缩共享证据，显著降低了每次事故的分析成本。 |
| [^60] | [Decided or Open? Comparing Program Repair Techniques When Patch Correctness Is Unknown](https://arxiv.org/abs/2611.23287) | 该论文提出一种将程序修复技术比较结果划分为“已判定”或“悬而未决”的方法，通过计算成功率差异在未知补丁正确性标签所有可能取值下的范围来量化比较结论的确定性，并能在测试运行前判断哪些比较无法通过增加测试来解决。 |
| [^61] | [The Illusion of Relief: How GenAI Redistributes Cognitive Workload in Software Organizations](https://arxiv.org/abs/2611.23286) | 研究表明生成式AI并未消除软件从业者的认知工作，而是将其重新分配——从业者从直接编写代码转向提供上下文、审查验证和协调AI输出等监督性工作，人类判断与问责仍然是核心。 |
| [^62] | [Beyond Stars: MTTX Brings Developer Responsiveness to Dependency Decisions, from Dashboard to IDE to GitHub Actions](https://arxiv.org/abs/2611.23285) | MTTX 是一个开源工具，通过精确的历史依赖解析为任意包计算 MTTU 和 MTTR 这两项难以伪造的响应能力指标，并将其集成到从仪表盘、IDE 到 GitHub Actions 的开发者工作流程各环节中。 |
| [^63] | [When Single Agent Is Enough: Rethinking Multi-Agent Systems for Automated Code Review](https://arxiv.org/abs/2611.23283) | 该论文通过系统比较发现，更简单的单智能体方法在代码审查任务中的表现与多智能体系统相当甚至更优，挑战了多智能体架构是复杂软件工程任务默认选择的普遍假设。 |
| [^64] | [Ephyra: A Property-Based Testing Framework for the LLM Era](https://arxiv.org/abs/2611.23282) | Ephyra是一个面向大语言模型时代的Rust语言基于属性测试框架，以可定制性、可观察性和生成器合成为核心特性，在14个开源系统中发现了349个漏洞。 |
| [^65] | [Log Analysis for LLM-Based Agents: A Lifecycle-Oriented Survey](https://arxiv.org/abs/2611.23281) | 本文首次以生命周期视角系统综述了基于LLM智能体的日志分析领域，通过对90项核心研究的阶段划分与比较，揭示了日志记录、结构化与挖掘之间的跨阶段依赖关系，并指出详细日志未必证据充分、结构一致不等于语义等价等关键局限。 |
| [^66] | [Evidence-Carrying Contracts for Safe Claim Composition in Data Marketplace Workflows](https://arxiv.org/abs/2611.23276) | 该论文提出“证据携带声明合约”这一轻量级假设-保证语义，通过冲突敏感的四状态域与安全组合规则，确保数据市场工作流中的事务声明只有在证据充分支持时才能被提升，并为有限无环工作流证明了可靠性、有界性、撤销安全性和终止性等关键性质。 |
| [^67] | [A Concurrent Structured Record Mirrored the Prose It Accompanied: A Pre-Registered Conceptual Replication and Extension of the Hide Pending Tool Call Scenario](https://arxiv.org/abs/2611.23274) | 本研究对 Huang 等人（2026）的“隐藏待处理工具调用”场景进行了预注册概念复制与扩展，发现要求模型在同一次生成中附加含“未解决问题”残余部分的结构化记录时，该记录只是映照了其伴随的散文叙述内容，并未能独立提升对缺失待处理结果的披露。 |
| [^68] | [torchtest2: One-Call Sanity Checks for PyTorch Models,Evaluated on Real and Unseen Architectures](https://arxiv.org/abs/2611.23273) | torchtest2 是一个开源库，只需一次调用即可对 PyTorch 模型执行 18 项健全性检查以捕获无声 bug，并通过在 103 个真实模型、600 个随机架构和 20 个真实 bug 上的变异分析验证了其有效性。 |
| [^69] | [An Industry Perspective on Software Vulnerability Management: A Systematic Review of Approaches, Tools, Practices, and Challenges](https://arxiv.org/abs/2611.23267) | 该系统性综述分析了57项直接涉及行业从业者的实证研究，总结了软件漏洞管理领域的方法、工具与实践及其实施挑战，发现现有证据以指南、经验教训等实践导向贡献为主，而严格的评估性研究较为缺乏。 |
| [^70] | [Escaping the Anthropomorphic Trap: Resolving the Cognitive Mismatch in Future AI Software Development](https://arxiv.org/abs/2611.23264) | 论文指出传统以人类认知为中心的软件开发范式与多智能体AI存在根本性认知错配，由此产生“AI-软件技术债务”，并提出“隐形软件工程”这一新范式，实现从智能体模仿人类向以环境为中心的共识调控治理的转变。 |
| [^71] | [Perceived Effectiveness of Social Debt Mitigation Measures: A Preliminary Vignette-Based Survey](https://arxiv.org/abs/2611.23263) | 该论文通过情景式问卷调查，让23名IT从业者和高年级学生对六类社会债务问题场景中的41项缓解措施进行有效性评分，并用配对t检验与分层贝叶斯模型分析排名，初步揭示了从业者眼中针对不同社会债务问题最有效的应对措施。 |
| [^72] | [Artificial Intelligence in Finance and the Early-Career Skills Shift: A Structured Evidence Synthesis of Decision Delegation, Bank Deployments, and Business Education](https://arxiv.org/abs/2611.23261) | 本文通过结构化的证据综合，系统梳理了金融工作中哪些任务可安全委托给AI、哪些须保留人类主导，并指出在AI压缩新手历练路径的背景下商业教育亟需转型。 |
| [^73] | [Software Engineering for LLM-Based Agents: A Lifecycle-Oriented Survey](https://arxiv.org/abs/2611.23260) | 本综述基于355篇论文，从需求工程、架构与设计、实现与集成、测试与保障、维护与演化五个软件生命周期阶段，系统梳理并比较了基于LLM智能体的软件工程方法。 |
| [^74] | [Controlled Natural Language for Requirements Specification: A Systematic Literature Review](https://arxiv.org/abs/2611.23259) | 本文通过一项覆盖2000年至2024年间133篇原始研究的系统性文献综述，对用于需求规约的受控自然语言（CNL）方法进行了识别、分类与比较。 |
| [^75] | [EnvAgent: Multi-Agent Approach to Building Science Environments in the Clouds](https://arxiv.org/abs/2611.23258) | 提出了EnvAgent多智能体框架，能够自动综合、构建和验证仓库级Docker环境，并引入ENVBENCH基准（包含54个科学成果）来评估科学计算环境自动化构建的效果。 |
| [^76] | [StatePortal: Agentic Checkpoint Synthesis for Testing Deep States in Interactive Software](https://arxiv.org/abs/2611.23257) | StatePortal 提出了一种智能体方法，能从自然语言测试用例和只读源码中自动合成可版本化、可重放的检查点，从而解决交互式 GUI 软件测试中反复到达深层状态的成本瓶颈。 |
| [^77] | [Generative AI in Software Testing: Practitioners' Perspectives on the Benefits, Challenges, Opportunities, and Risks](https://arxiv.org/abs/2611.23256) | 一项针对64名软件测试从业者的问卷调查显示，从业者普遍认可生成式AI在减少工作量、测试生成和调试方面的益处，但也担心提示词依赖、信任问题和错误输出，且使用者比非使用者更关注具体收益与可靠性风险。 |
| [^78] | [Structure-Behavior Coalescence for Model-Based Systems Engineering](https://arxiv.org/abs/2611.23255) | 本文提出结构-行为融合（SBC），将系统定义为结构与行为通过交互融合而成的单一集成实体，并借助进程代数框架与良构性约束实现MBSE中“构造即一致”的系统建模。 |
| [^79] | [Measuring What Matters in the Age of AI: The Research Software Metrics Landscape](https://arxiv.org/abs/2611.23253) | 该论文基于49篇文献系统梳理了研究软件度量指标的全景，将其归纳为八大类并从数据来源、效度、自动化潜力和滥用风险等维度进行分析，同时指出AI辅助开发正使现有指标体系的假设面临失效。 |
| [^80] | [From Data Quality to Data Trustworthiness in AI-Driven Software-Intensive Embedded Systems: A Systematic Review](https://arxiv.org/abs/2611.23252) | 本系统性综述识别了AI驱动软件密集型嵌入式系统中影响数据可信性的16个数据质量关注点和8类评估管理方法，并揭示上游数据问题可能嵌入训练基线且在运行时难以检测。 |
| [^81] | [What Moves the Ceiling on Automated Audit: from General Code to Scientific Code, Results and Data](https://arxiv.org/abs/2611.23250) | 本文在真值基准上量化了“异厂商模型审计+确定性检查”这一自动化审计流程的能力上限：通用代码上召回率仅约 30% 且无饱和点，可通过阅读模型、规则与决策规则提升；而在科学结果核查中，数字溯源检查能识别全部 49 份不一致报告，模型审计则可额外发现 25 份内部一致伪造中的 23 份。 |
| [^82] | [The Obsolescence of High-Level Code: Agentic Generation from Semantic Artifacts](https://arxiv.org/abs/2611.23249) | 本文提出当AI智能体能够从作为语义权威的语义规范出发生成、验证和重新生成可执行实现时，高级源代码将不再是软件开发的主要持久工件，“生成”将成为软件开发的基本操作。 |
| [^83] | [ETA: Exit on Tool Arrival in Speculative Agent Loops](https://arxiv.org/abs/2611.23239) | ETA 将推测性工具分发的分支用作模型决策状态的“传感器”，在工具结果到达且预测稳定一致时提前截断推理，从而在不牺牲准确性的前提下将工具调用智能体的速度提升最高 1.33 倍并减少 24% 的 token 解码量。 |
| [^84] | [Tracing the Messy Middle: A Runtime Transaction Schema for AI Coding Agents](https://arxiv.org/abs/2611.23234) | 本文提出一个可观测性框架，将AI编码代理开发过程中通常被忽视的运行时行为（如重试、回退、路由决策和放弃的会话）记录为结构化事务数据，从而支持基于行为证据的工程辅导、工作流优化和以总任务成本为导向的路由策略改进。 |
| [^85] | [A Metric for Semantic Graph Matching of No-Code Workflows](https://arxiv.org/abs/2611.23233) | 该论文为无代码工作流图首次定义了严格的语义图匹配度量，证明了其在属性同构意义下满足度量公理，并通过填充匈牙利算法给出可扩展的上界，在4,705个公开n8n工作流上完成了大规模验证。 |
| [^86] | [Live Noise Fingerprints for Quantum Program Analysis](https://arxiv.org/abs/2611.19143) | 本文提出“噪声指纹”这一紧凑的抽象表示，并通过概率伽罗瓦连接将其与具体噪声行为关联，实现了对含噪量子程序的高效可靠分析，且多项式规模的接口可产生比完整噪声模型指数级更小的指纹。 |
| [^87] | [Detecting Silent Faults at the Environment-Trainer Boundary in Reinforcement Learning](https://arxiv.org/abs/2611.08144) | 该论文发现强化学习中环境与训练器边界处的静默故障几乎无法通过学习曲线等基于结果的检查发现，而直接检查运行过程中跨越边界传输的数据本身则能检测到每一个注入的故障。 |
| [^88] | [Interaction-as-a-Judge](https://arxiv.org/abs/2610.17895) | 提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。 |
| [^89] | [What Does a Harness Buy? Tokens, Mostly](https://arxiv.org/abs/2610.04433) | 该研究通过对七个模型在三个生产级框架和多个基准上的重复对照实验发现，没有任何框架能稳定提升性能，更换框架带来的差异仅有 2 到 5 个百分点、效果与重新运行一次相当，框架带来的分数提升主要源于更多的 token 消耗。 |
| [^90] | [Hearsay: Who Can Audit the Record a Deployed Agent Harness Writes?](https://arxiv.org/abs/2609.32495) | 该论文提出 Hearsay 基准，揭示了一个关键问题：由于已部署的智能体运行框架是运行记录的唯一书写者，审计者虽能从记录中指认故障（高达九成运行），却无法独立核实记录的真实性，使其沦为法律意义上的“传闻”。 |
| [^91] | [Who Is Behind the Harness? Fingerprinting LLMs through Agentic Behavior](https://arxiv.org/abs/2609.28559) | 提出了一种名为LIDAR的主动式黑盒指纹识别方法，通过编码智能体在运行时的决策与行动（如编辑后验证、故障恢复和规范-测试冲突处理等行为）来识别框架背后的大语言模型身份。 |
| [^92] | [Biomedical Knowledge Composition: A Software Engineering Perspective](https://arxiv.org/abs/2608.08927) | 本文从软件工程视角统一阐述生物医学知识基础设施，向软件工程师介绍知识图谱及五大数据协调挑战，并指出软件工具采用不足是生物医学知识基础设施工程化困难的一个根本原因。 |
| [^93] | [Scrouting: Cost-Aware Routing of Coding Agents by Scouting the Repository First](https://arxiv.org/abs/2608.04804) | SuperScout先用一个7B搜索器侦察仓库并生成经沙箱验证的结构化交接文档，再据此将编码任务路由至合适的前沿模型，在SWE-bench Pro上以约五分之一的成本达到与最佳单一模型相当的解决率，且新增修复模型无需重新训练。 |
| [^94] | [Where Is the Tradeoff in Using Third-Party API Routers for Agentic Software Development?](https://arxiv.org/abs/2607.23624) | 该论文首次实证研究了编码智能体中第三方API路由器的路由器端注入风险，通过四个隐蔽程度递增的干预级别（如响应替换）揭示其可对软件开发任务造成真实且难以检测的影响，而现有客户端权限机制难以有效防御。 |
| [^95] | [Prior Knowledge or Search? A Study of LLM Agents in Hardware-Aware Code Optimization](https://arxiv.org/abs/2605.19782) | 该研究通过三个受控实验揭示，LLM智能体在硬件感知代码优化中依赖先验知识而非搜索——它在黑盒优化中表现为贪婪优化器，在零样本内核生成中会完全忽略输入尺寸信息，且面对不常见的内核尺寸时性能急剧下降。 |
| [^96] | [Analysis of Commit Signing on GitHub](https://arxiv.org/abs/2604.14014) | 该论文首次以GitHub账户为抽样单位，通过对270多万个账户、1600多万次提交和20多万个公钥的大规模端到端分析，系统揭示了开发者提交签名中签名密钥的使用与管理模式。 |

# 详细

[^1]: 智能体框架工程Beta：以需求为中心的智能体框架演进

    Agentic Harness Engineering Beta: Requirement-Centered Evolution of Agent Harnesses

    [https://arxiv.org/abs/2611.24924](https://arxiv.org/abs/2611.24924)

    提出AHE-Beta，一个以需求为中心的智能体框架演进循环，通过模拟用户提出需求、团队协作实现可泛化变更以及基于经验证需求选择变更，解决了现有方法产生狭窄补丁和基准局部最优的问题。

    

    现实世界的框架工程由显性与隐性用户需求驱动，这些需求指导着开发与验证过程。然而，现有的框架演进方法往往忽视隐性需求和交互需求，同时使用单一演进智能体来修改框架，并依赖基准测试分数来选择变更。这一过程常常产生狭窄的补丁，并导致针对特定基准的局部最优。我们提出智能体框架工程Beta（AHE-Beta），这是一个以需求为中心的循环，将现实世界的工程实践引入框架演进。该循环包含三个步骤：（1）用户驱动评估：让模拟用户在与框架进行多轮交互的过程中，以问题（issue）的形式提出需求；（2）基于团队的修改：将这些需求实现为可泛化的变更，而非特定任务的补丁；（3）需求驱动选择：仅当变更解决了经验证的需求时才将其合并。

    arXiv:2611.24924v1 Announce Type: cross  Abstract: Real-world harness engineering is driven by explicit and implicit user requirements, which guide development and validation. However, existing harness evolution methods often overlook implicit requirements and interaction needs, while using a single evolution agent to modify the harness and benchmark scores to select changes. This process often produces narrow patches and leads to benchmark-specific local optima. We introduce Agentic Harness Engineering Beta (AHE-Beta), a requirement-centered loop that brings real-world engineering practices into harness evolution. The loop consists of three steps: (1) User-driven evaluation lets simulated users raise their requirements as issues while they work with the harness over multiple turns. (2) Team-based modification implements those requirements as generalizable changes rather than task-specific patches. (3) Requirement-driven selection merges a change only when it resolves a verified requir
    
[^2]: MicroAssert：面向CPU验证的微架构感知结构化断言生成

    MicroAssert: Microarchitecture-Aware Structured Assertion Generation for CPU Verification

    [https://arxiv.org/abs/2611.24872](https://arxiv.org/abs/2611.24872)

    MicroAssert是一个微架构感知的结构化断言自动生成框架，通过从RTL识别功能区域并生成局部与复合断言，在五个开源RISC-V CPU上实现了100%语法正确率和74.82%的平均证明率，错误检测率较现有方法提升1.99至3.51倍。

    

    自动断言生成减少了人工验证的工作量，但现有方法难以捕捉CPU中的微架构行为和跨区域交互。我们提出了MicroAssert，这是一个微架构感知的框架，它直接从RTL中识别功能区域，并通过结构化契约和确定性验证原语生成局部及复合断言。覆盖率引导的反馈通过逐步扩展与证明相关的RTL覆盖率，进一步提升了验证完整性。在五个开源RISC-V CPU上与三种现有断言生成方法进行对比评估，MicroAssert实现了100%的语法正确性以及完全的分支、语句和翻转覆盖率，平均证明率达到74.82%，Proof Core覆盖率达到90.96%。变异测试进一步展示了其优势，平均错误检测率达到28.73%，比现有方法高出1.99至3.51倍。

    arXiv:2611.24872v1 Announce Type: cross  Abstract: Automatic assertion generation reduces manual verification effort, yet existing approaches struggle to capture microarchitectural behaviors and cross-region interactions in CPUs. We present MicroAssert, a microarchitecture-aware framework that identifies functional regions directly from RTL and generates local and composite assertions through structured contracts and deterministic verification primitives. Coverage-guided feedback further improves verification completeness by progressively expanding proof-relevant RTL coverage. Evaluated on five open-source RISC-V CPUs against three existing assertion generation approaches, MicroAssert achieves 100% syntax correctness and complete branch, statement, and toggle coverage, with an average proven rate of 74.82% and Proof Core coverage of 90.96%. Mutation testing further demonstrates its advantage, achieving an average error detection rate of 28.73%, 1.99 to 3.51 times higher than existing m
    
[^3]: MLIR-MRacle：内存模型感知的并发MLIR蜕变模糊测试

    MLIR-MRacle: Memory-Model-Aware Metamorphic Fuzzing of Concurrent MLIR

    [https://arxiv.org/abs/2611.24804](https://arxiv.org/abs/2611.24804)

    MLIR-MRacle是一个内存模型感知的蜕变模糊测试工具，通过目标内存模型调控的蜕变关系与统计预言机来检测并发MLIR编译过程中可能悄然改变程序行为的优化错误。

    

    基于MLIR构建的多层次编译器将并发程序经过多个阶段降低至目标平台，而这些目标平台的内存模型可能截然不同，在这种情况下，一个看似合理的优化可能会悄然改变程序行为。测试这些降低过程非常困难：并发程序通常没有单一的预期结果，而现有的MLIR模糊测试工具往往假设程序具有确定性。我们提出了MLIR-MRacle，这是一个面向并发MLIR程序的内存模型感知的蜕变模糊测试工具。MLIR-MRacle应用由目标内存模型调控的蜕变关系，使得原始程序和转换后的程序都保持良定义——重复执行两者，并通过两个统计预言机之一来比较它们的结果集合或分布。该工具采用基于libafl构建的覆盖率引导的测试活动驱动器来执行持续模糊测试。蜕变关系、内存模型、预言机和执行后端均提供了扩展点，供未来的研究者和编译器工程师使用。

    arXiv:2611.24804v1 Announce Type: new  Abstract: Multi-level compilers built on MLIR lower concurrent programs through many stages towards targets with potentially drastically different memory models, where an optimisation which appears reasonable, can silently change program behaviour. Testing these lowerings is hard: a concurrent program often has no single expected outcome, and existing MLIR fuzzers often assume determinism. We present MLIR-MRacle, a memory-model-aware metamorphic fuzzer for concurrent MLIR programs. MLIR-MRacle applies metamorphic relations gated by the target memory model, so that both the original and transformed programs remain well defined - executing both repeatedly, and comparing their outcome sets or distributions withone of two statistical oracles. A coverage guided campaign driver built on libafl is used to perform continuous fuzzing. MRs, memory models, oracles, and execution backends all provide extension points for future researchers and compiler engine
    
[^4]: 局限在于输入而非模型：工业控制器自然语言接口中的状态依赖命令安全性

    The Limit Is in the Input, Not the Model: State-Dependent Command Safety in Natural-Language Interfaces to Industrial Controllers

    [https://arxiv.org/abs/2611.24682](https://arxiv.org/abs/2611.24682)

    本文提出“可判定性审计”方法，证明语言模型在判断工业控制器命令安全性时的失误源于输入中缺失机器状态信息而非模型推理能力不足，且提供状态值可显著提升判断的准确性。

    

    语言模型在判断发送给工业控制器的命令时，可能没有接收到机器状态，即设备当前的状况。同一条命令在不同的状态下可能需要截然相反的安全决策，因此错误的回答可能反映的是信息缺失而非推理能力不足。我们提出了一种“可判定性审计”：即模型是否收到了做出决策所需的信息？该审计记录模型的输入，并将同一条命令与两个需要相反答案的状态配对。只有当两个答案都正确时，配对分数才被计入。我们先向模型输入中添加规则，再添加状态值。在一项由作者构建的、来自三个系统的15对测试用例、使用四个语言模型的测试中，在没有状态信息的情况下，即使提供了规则，没有任何模型能够可靠地区分这些配对用例。提供状态后，得分范围从15对中正确0对到13对不等；其中三个模型有所改进。仅依靠文本时，一个模型总是允许命令执行，另一个模型总是拒绝命令。它们的总体

    arXiv:2611.24682v1 Announce Type: cross  Abstract: Language models judging commands to industrial controllers may not receive machine state, the equipment's current condition. The same command can require opposite safety decisions in different states, so an incorrect answer may reflect missing information rather than poor reasoning. We propose an answerability audit: does the model receive the information needed to decide? The audit records the model's inputs and pairs the same command with two states requiring opposite answers. The paired score counts a pair only when both answers are correct. We add rules, then state values, to the model's input. In an author-built test of 15 pairs from three systems with four language models, none reliably distinguished the paired cases without state, even with rules. With state, scores ranged from 0 to 13 correct pairs out of 15; three models improved. With text alone, one model always allowed commands and another always denied them. Their overall 
    
[^5]: DrawScore：基于工程图纸评估生成的CAD模型

    DrawScore: Evaluating Generated CAD Models against Engineering Drawings

    [https://arxiv.org/abs/2611.24681](https://arxiv.org/abs/2611.24681)

    提出DrawScore新评估指标，通过工程图纸式三视图投影与饱和惩罚，使孔位错误等制造关键的小体积局部缺陷权重仅线性衰减，从而有效弥补传统体积类指标对局部错误不敏感的问题。

    

    自动CAD生成通常采用体积IoU、Chamfer距离和表面F分数进行评估，这些指标几乎无法惩罚仅占零件体积或表面积很小比例的局部错误，例如孔的位置放错或槽开在了错误的面上，尽管此类错误对制造而言至关重要。我们将这些指标形式化为统一的积分形式，并证明局部错误的权重随其尺寸的(k+q)次幂衰减，其中k为积分域的维度，q由惩罚函数决定。受此分析启发，我们提出DrawScore，它像工程图纸一样将两个模型投影为带有可见线和隐藏线的三视图，并沿投影曲线施加饱和惩罚，使局部错误的权重仅线性衰减。受控实验证实了预测的衰减阶数，并在现有CAD生成代理的输出上……（原文摘要至此截断）

    arXiv:2611.24681v1 Announce Type: cross  Abstract: Automatic CAD generation is commonly evaluated with volumetric IoU, Chamfer Distance, and surface F-score, which barely penalize local errors that occupy only a small fraction of a part's volume or surface area, such as a misplaced hole or a slot on the wrong face, even though such errors are critical to manufacturing. We formalize these metrics in a unified integral form and show that the weight of a local error decays as the $(k+q)$-th power of its size, where $k$ is the dimension of the integration domain and $q$ is determined by the penalty function. Guided by this analysis, we propose DrawScore, which projects both models into three orthographic views with visible and hidden lines, as in engineering drawings, and applies a saturating penalty along the projected curves, so that the weight of a local error decays only linearly. Controlled experiments confirm the predicted decay orders, and on outputs of existing CAD generation agent
    
[^6]: 元数据驱动的AI：一种基于大语言模型的可持续数据分析架构模式

    Metadata-Driven AI: An Architectural Pattern for Sustainable Data Analysis with LLMs

    [https://arxiv.org/abs/2611.24661](https://arxiv.org/abs/2611.24661)

    该论文提出“元数据驱动的AI”架构模式，只让元数据而非实际数据跨越单向边界进入LLM，并将模型生成的分析程序保留复用，从而将推理成本从每次运行一次降低为每次分析一次，实现更经济、更私密且可持续的自然语言数据分析。

    

    大型语言模型（LLM）使用户能够用自然语言分析数据，既可以将数据直接提供给模型读取，也可以让模型生成代码来处理数据。但这两种方式都会将数据或其片段发送给模型；每次运行都需要支付推理成本；而且模型为回答问题而编写的代码通常在问题得到解答后就被丢弃了。我们提出了元数据驱动的AI（Metadata-Driven AI），这是一种将数据与模型分离、并将每次分析转化为可复用程序的架构模式。在该模式中，只有元数据（而非实际数据）通过单向推理边界传递给LLM，LLM返回的程序在数据旁边运行并被保留下来，因此推理成本每次分析只需支付一次，而不是每次运行都需支付。我们对该模式进行了描述，并给出了一个简单的成本模型、对公开数据集上跨越该边界的信息量之少的测量结果，以及一个表明小型本地托管模型可以胜任该任务的基准测试。

    arXiv:2611.24661v1 Announce Type: new  Abstract: Large Language Models (LLMs) let users analyse data in natural language, either by giving the data to the model to read directly or by having the model generate code to process it. Both routes send the data, or fragments of it, to the model; both pay for inference on every run; and the code a model writes to answer a question is usually discarded once the question is answered. We present Metadata-Driven AI, an architectural pattern that keeps the data away from the model and turns each analysis into a reusable program. Only metadata (not the actual data) crosses a one-way inference boundary to the LLM. The program the LLM returns runs next to the data and is kept, so that inference is paid once per analysis rather than once per run. We describe the pattern together with a simple cost model, a measurement of how little information crosses the boundary on public datasets, and a benchmark showing that a small, locally hosted model can serve
    
[^7]: 基于延续审计在真实场景中探测智能体的过度积极行为

    Probing agent overeagerness in realistic settings with continuation audits

    [https://arxiv.org/abs/2611.24491](https://arxiv.org/abs/2611.24491)

    本文提出“延续审计”方法，以真实编程智能体会话记录的前缀为起点引导审计者沿轨迹延续至预设目标行为，从而在真实场景中探测智能体的过度积极行为，并规避了模型识别合成测试环境的问题。

    

    前沿编程智能体日益表现出“过度积极”行为，即采取超出用户授权范围的、令人意外甚至鲁莽的行动——这在近期前沿模型开发商发布的系统报告中已有记录。这些案例通常通过内部部署监控发现，但这种方式需要私有流量数据，对于开源权重模型和独立评估者而言无法企及。自动化对齐审计是天然的替代方案，然而能力较强的模型越来越容易识别出合成的测试环境并相应调整自身行为。我们提出了“延续审计”，这是一种流程，它以从真实编程智能体会话记录中截取的前缀作为审计者的基础，并引导其沿轨迹继续推进至预设的目标行为。由于前缀来自真实会话，审计者只需维持场景的真实性，而无需从零构建。我们在 SWE-Chat 前缀上验证了该方法：在 90 组配对审计的子集上，评判者无法可靠地区分……（原文摘要在此处截断）

    arXiv:2611.24491v1 Announce Type: cross  Abstract: Frontier coding agents increasingly exhibit overeagerness, taking surprising, sometimes reckless actions beyond a user's mandate --- reported in recent system cards from frontier model developers. These cases are surfaced through internal deployment monitoring, which requires private traffic and is out of reach for open-weight models and independent evaluators. Automated alignment audits are the natural alternative, but capable models increasingly recognise the synthetic setup and adjust their behaviour. We introduce Continuation Audits, a pipeline that grounds the auditor on a prefix drawn from a real coding-agent transcript and directs it to continue the trajectory toward a seeded behaviour. Because the prefix is drawn from a real session, the auditor only needs to maintain a realistic setting rather than manufacturing it. We validate the method on SWE-Chat prefixes: on a paired subset of 90 audits a judge cannot reliably distinguish
    
[^8]: 缓解修复反转：面向函数级漏洞检测的反事实约束

    Mitigating fix reversal: Counterfactual constraints for function-level vulnerability detection

    [https://arxiv.org/abs/2611.24489](https://arxiv.org/abs/2611.24489)

    CFVul通过滑动窗口编码保留完整修复信息，并利用训练时的内容约束和推理时的长度约束等反事实手段，缓解了函数级漏洞检测中因长度捷径导致的“修复反转”问题。

    

    函数级漏洞检测器常常表现出“修复反转”现象，即给修复后的版本赋予比漏洞版本更高的漏洞概率。修复通常会延长函数长度，且长度增加越大，反转现象越严重。代码变换实验表明，修复反转主要源于长度捷径：当内容固定时，漏洞概率随函数长度增加而上升。我们提出CFVul来解决这一问题。滑动窗口编码覆盖完整函数，保留了截断边界之外的修复信息。在训练阶段，内容约束促使漏洞版本在相同长度输入下获得更高的漏洞概率。在推理阶段，长度约束对固定参考长度下的漏洞概率进行平均，降低了实际长度变化的影响。在PrimeVul数据集上，CFVul在修复对排序指标上取得了……

    arXiv:2611.24489v1 Announce Type: cross  Abstract: Function-level vulnerability detectors often exhibit fix reversal, assigning a higher vulnerability probability to a fixed version than to its vulnerable counterpart. Fixes usually lengthen functions, and larger length increases are associated with more severe reversal. Code-transformation experiments indicate that fix reversal mainly stems from a length shortcut: vulnerability probability increases with function length when content is fixed. We propose CFVul to address this problem. Sliding-window encoding covers complete functions, retaining fix information beyond the truncation boundary. During training, a content constraint encourages the vulnerable version to receive a higher vulnerability probability under a common length input. During inference, a length constraint averages vulnerability probabilities over fixed reference lengths, reducing the influence of actual length changes. On PrimeVul, CFVul achieves a fix-pair ranking met
    
[^9]: 预测流程级行为：一项基准测试研究

    Predicting Process-Level Behavior: A Benchmarking Study

    [https://arxiv.org/abs/2611.24425](https://arxiv.org/abs/2611.24425)

    该论文通过对并发轨迹预测和平均吞吐时间预测两项流程级任务的全面基准测试，比较了统计方法、机器学习方法、基础模型、业务流程仿真以及聚合到流程级的轨迹级PPM方法，揭示了最适合预测流程级行为的建模范式。

    

    预测性流程监控（PPM）通常预测单个流程实例的未来情况，即在轨迹（trace）级别上进行运作。然而，流程管理者通常是在流程级别上进行决策的，例如在分配员工或计算关键绩效指标（KPI）时。鉴于PPM研究至今尚未专注于流程级预测，一个问题是：现有方法对这类流程级管理决策的支持程度究竟如何。我们通过对两项流程级预测任务——并发轨迹预测和平均吞吐时间预测——的全面基准测试研究来回答这一问题。在合成事件日志和真实事件日志上，我们比较了统计方法和基于机器学习的方法、基础模型、业务流程仿真，以及（这一点尤为重要）已被广泛采用的轨迹级PPM方法——我们将其预测结果聚合到流程级别。我们评估了哪些建模范式最适合用于预测流程级行为。

    arXiv:2611.24425v1 Announce Type: new  Abstract: Predictive process monitoring (PPM) typically forecasts the future of individual process instances, hence operating on trace level. Process managers, however, typically operate on process level, e.g., when assigning staff or calculating KPIs. The question is how well such process-level managerial decisions are supported by existing approaches, given that PPM research has so far not focused on process-level predictions. We answer this question through a comprehensive benchmarking study of two process-level prediction tasks: concurrent trace prediction and average throughput time prediction. Across synthetic and real-life event logs, we compare statistical and machine-learning-based methods, foundation models, business process simulation, and, importantly, established trace-level PPM approaches, aggregating their predictions to the process-level. We assess which modeling paradigms are most suitable for predicting process-level behavior and
    
[^10]: 探索软件开发中的多模态AI-XR交互：从专家视角到语义代码大纲

    Exploring multimodal AI--XR interactions for software development: From expert perspectives to semantic code outlines

    [https://arxiv.org/abs/2611.24334](https://arxiv.org/abs/2611.24334)

    该研究通过13位专家访谈与两个XR原型实验，探索了利用扩展现实工作空间中的多模态信息（如头部注视方向与手动输入），将已查看的代码替换为语义化功能描述大纲，从而帮助开发人员理解和审查AI生成代码的交互设计。

    

    AI工具可以快速生成代码，但开发人员仍需要理解并检查这些结果。我们探索了源自扩展现实（XR）工作空间的丰富多模态上下文如何帮助开发人员管理这些信息。我们首先采访了13位来自AI、XR及相关领域的专家，以识别其中的机会与设计挑战（研究1）。这些访谈成果催生了在12名专业开发人员中开展探索的两个原型（研究2）：其中四人试用了一个版本，该版本中代码在被注视后会逐渐淡出直至不可见；另外八人试用了另一个版本，该版本将已查看的代码段替换为对其功能的简短描述。在第二个原型中，头部朝向用于确定何时提供更改，而手动输入则允许开发人员应用更改并重新打开原始代码。一项与七位软件工程研究人员进行的开放式演示提供了补充反馈。参与者的反馈表明……

    arXiv:2611.24334v1 Announce Type: cross  Abstract: AI tools can generate code quickly, but developers still need to understand and check the results. We explore how the enriched multimodal context derived from the extended reality (XR) workspace might help developers manage this information. We first interviewed 13 experts in AI, XR, and related fields to identify opportunities and design challenges (Study 1). These interviews informed two prototypes explored with 12 professional developers (Study 2): four tried a version where the code faded to become invisible after it was looked at, and eight tried a version that replaced viewed code sections with short descriptions of their purpose. In the second prototype, head direction helped determine when to offer a change, while manual input let developers apply it and reopen the original code. A separate open-ended demonstration with seven software-engineering researchers provided complementary feedback. Participants' accounts suggested that
    
[^11]: 使用超时级联降低编码智能体的推理成本

    Reducing Inference Cost of Coding Agents with Timeout Cascades

    [https://arxiv.org/abs/2611.24324](https://arxiv.org/abs/2611.24324)

    提出一种超时级联的混合推理策略，先让小模型尝试解决任务、必要时再升级到大模型，从而在不牺牲准确率的前提下大幅降低AI编码智能体的推理成本。

    

    AI编码智能体已成为大语言模型最流行的应用之一。然而，智能体式AI的运行需要消耗大量GPU计算资源，这限制了其盈利能力和规模化部署的可行性。在本文中，我们研究了更小、更便宜的模型能否在不牺牲准确性的前提下降低这一成本。我们在一个软件工程基准测试上评估了自2024年以来发布的36个流行的大型和小型开源权重模型，发现近期的小模型正在缩小与大型模型之间的准确性差距。我们进一步研究了这一差距的结构，发现差距高度集中在小语言模型无法解决的困难任务上，而对于简单和中等复杂度的任务，小模型能够取得非常好的准确性。基于这一观察，我们展示了通过采用混合推理策略——先尝试小模型，仅在需要时才升级到大模型——我们可以大幅降低推理成本。（注：原文摘要在此处截断）

    arXiv:2611.24324v1 Announce Type: new  Abstract: AI coding agents have become one of the most popular applications of large language models. Yet, agentic AI comes at the cost of a tremendous amount of GPU compute resources, which limits its profitability and its viability at scale. In this paper, we investigate whether smaller, cheaper models can cut this cost without sacrificing accuracy.   We evaluate 36 popular large and small open-weight models released since 2024 on a software engineering benchmark, and find that recent small models are closing the accuracy gap compared to their large counterparts. We investigate the structure of this gap and find that the gap is highly concentrated around hard tasks that small language models fail to solve while achieving remarkably good accuracy for easy and medium complexity tasks. Following this observation, we show that by utilizing a hybrid inference strategy that tries a small model first and escalates to a large one only when needed, we ca
    
[^12]: 紧跟内核步伐：大语言模型能否让 Linux 驱动程序适应接口变更？

    Keeping Up with the Kernel: Can LLMs Adapt Linux Drivers to Interface Changes?

    [https://arxiv.org/abs/2611.24269](https://arxiv.org/abs/2611.24269)

    该论文研究了 20 年来 Linux 内核接口变更引发的驱动“伴随演化”现象，并构建了包含 316 个任务的基准 CoEvoBench 来评估大语言模型能否自动将旧驱动适配到新内核。

    

    我们分析了 Linux 中的“伴随演化”现象，即当驱动程序所依赖的内核接口发生变化时，驱动程序所需的协同更新，涵盖自该现象首次被研究以来 20 年间的版本，从 v3.0 到 v7.2。自 v2.6.13 以来，驱动文件和支持库的数量增长了约八倍，对使用最广泛的库的一次修改现在可波及约 5300 个驱动文件，且这类变更要么集中爆发，要么分散在多个版本中。我们提出了 CoEvoBench，这是一个基准测试，要求模型将旧的驱动模块适配到目标内核，并根据开发者的参考补丁来评估结果。该基准包含 316 个经人工确认的任务，来自 202 个主线提交和 268 个模块，跨越内核版本 v6.19 至 v7.2。它区分了 134 个“破坏构建”任务（旧驱动无法编译）和 182 个“构建兼容”任务（旧用法仍可编译）。使用 CoEvoAgent，我们评估了

    arXiv:2611.24269v1 Announce Type: new  Abstract: We analyze collateral evolution in Linux, the coordinated updates that drivers require when the kernel interfaces they depend on change, over the 20 years since it was first studied, covering releases from v3.0 to v7.2. Since v2.6.13, the number of driver files and support libraries has grown about eightfold, a single change to the most widely used library can now reach about 5.3k driver files, and such changes arrive either in bursts or spread over many releases. We introduce CoEvoBench, a benchmark that asks models to adapt an old driver module to a target kernel and evaluates the result against the developer's reference patch. The benchmark contains 316 manually confirmed tasks from 202 mainline commits and 268 modules, spanning kernel releases v6.19 to v7.2. It distinguishes 134 build-breaking tasks, where the old driver fails to compile, from 182 build-compatible tasks, where the old usage still builds. Using CoEvoAgent, we evaluate
    
[^13]: LLM辅助软件开发中的理解差距：一项实验研究

    The Comprehension Gap in LLM-Assisted Software Development: An Experiment

    [https://arxiv.org/abs/2611.24216](https://arxiv.org/abs/2611.24216)

    通过35名参与者的交叉实验发现，开发者对LLM生成代码的理解显著低于自写代码，这种“认知债务”会阻碍后续任务的高效完成。

    

    当LLM工具在软件开发任务中被推向超出其能力范围时，开发者必须能够及时介入。然而，由于过度的代码生成，开发者在开发过程中所获得的理解可能会受到负面影响。这可能造成一种“认知债务”，需要先加以解决才能重新高效地完成任务。我们探索了开发者对自己通过提示词生成的代码的理解程度（与自写代码和未见过的代码进行对比），以及他们如何反思自身的理解水平。我们对35名参与者（包括学生、学术人员和软件开发者）开展了交叉实验和访谈。参与者分别在使用和不使用LLM聊天工具的条件下完成两个编程任务，随后需要在所生成的代码以及额外的未见过的代码中找出被引入的变异。我们结合访谈和屏幕录像的洞察来补充实验结果。

    arXiv:2611.24216v1 Announce Type: new  Abstract: When LLM tools are pushed beyond their capabilities for software development tasks, developers must be able to step in. However, the comprehension that developers acquire during development might be negatively impacted due to excessive code generation. This could create a cognitive debt that needs to be resolved before tasks can be solved efficiently again. We explore how well developers understand code they generated by prompting in comparison to self-written and unseen code and how they reflect on their comprehension. We conduct a crossover experiment and interviews with 35 participants consisting of students, academic personnel, and software developers. After completing two programming tasks with or without LLM chat usage, respectively, participants are tasked with finding introduced mutations in the produced and additional unseen code. We use insights from interviews and screen recordings to supplement our results. Regarding the expe
    
[^14]: OmegaHyper：基于ω-自动机“瘦身”的显式状态HyperLTL模型检测

    OmegaHyper: Explicit-State HyperLTL Model Checking on an omega-Automaton Diet

    [https://arxiv.org/abs/2611.24210](https://arxiv.org/abs/2611.24210)

    OmegaHyper在保留AutoHyper算法的基础上，通过分解与缩减所处理的ω-自动机并集成Kofola求补工具，将显式状态HyperLTL模型检测的性能提升了多达几个数量级。

    

    HyperLTL通过对轨迹的量化扩展了LTL。AUtoHyper是事实上的HyperLTL显式状态模型检测器，对任意量词前缀均完备：它通过将公式的ω-自动机与系统做乘积来消除量词，并在量词交替处进行求补。然而，由于它需要完整地实体化输入系统和中间自动机，因此面临状态爆炸问题。我们提出了OmegaHyper，它是对AutoHyper的一次全面改造，保留了原有算法，但分解并缩减了所处理自动机的规模，同时集成了Kofola ω-自动机求补工具。我们在来自已发布测试集的120个基准测试上，将OmegaHyper与现有的显式状态及符号化HyperLTL模型检测器进行了比较。结果表明，OmegaHyper将显式状态HyperLTL模型检测的性能提升了多达几个数量级；而对于具有短反例的违规实例，有界的符号化工具仍然能够覆盖更多……

    arXiv:2611.24210v1 Announce Type: new  Abstract: HyperLTL extends LTL with quantification over traces. AUtoHyper is the de facto explicit-state model checker for HyperLTL, complete for an arbitrary quantifier prefix: it eliminates quantifiers through products of the formula's omega-automaton with the systems, complementing on quantifier alternations. However, it suffers from state explosion, as it fully materializes input systems and intermediate automata. We present OmegaHyper, an overhaul of AutoHyper that keeps its algorithm but decomposes and reduces the size of the processed automata, and integrates the Kofola omega-automaton complementation tool. We compare OmegaHyper with existing explicit-state and symbolic HyperLTL model checkers, on 120 benchmarks from their published suites. OmegaHyper improves the performance of explicit-state HyperLTL model checking by up to orders of magnitude; bounded symbolic tools still reach further on violated instances with short counterexamples, as
    
[^15]: MemoSync：跨语境协同管理开发者与智能体的元记忆

    MemoSync: Co-Managing Developer-Agent Metamemory Across Contexts

    [https://arxiv.org/abs/2611.24165](https://arxiv.org/abs/2611.24165)

    MemoSync 提出了一个让开发者与 AI 编程智能体协同管理记忆的系统，智能体负责提出记忆候选并报告使用情况，开发者则可审查、编辑和限定记忆范围，从而解决了开发者对智能体记忆缺乏可见性的“元记忆”问题。

    

    AI 编程智能体越来越多地负责管理跨会话和跨项目所携带的上下文内容，但开发者对这些智能体究竟存储、检索和应用了什么却缺乏足够的可见性。这种不透明性造成了一个“元记忆”问题：开发者关于什么应该被记住的元层判断，与智能体对象层的记忆操作相互分离，破坏了保持两个层级一致所需的监控与控制机制。我们对 12 名开发者开展了一项形成性研究，识别出他们希望在智能体记忆使用的四个维度上监控和控制的内容。基于这些发现，我们设计了 MemoSync——一个协同管理系统，它将智能体记忆视为一种共享的、可操作的表达：智能体提出记忆候选并报告其使用了哪些条目，而开发者则对这些条目进行审查、编辑和范围界定。在一项与两个基线系统对比的被试间实验（N=132）中，MemoSync 的参与者取得了显著更高的任务表现……

    arXiv:2611.24165v1 Announce Type: cross  Abstract: AI coding agents increasingly manage what context carries across sessions and projects, but developers have limited visibility into what these agents store, retrieve, and apply. This opacity creates a metamemory problem: the developer's meta-level judgments about what should be remembered are separated from the agent's object-level memory operations, disrupting the monitoring and control that keep the two levels aligned. We conducted a formative study with 12 developers and identified what they want to monitor and control across four dimensions of agent memory use. Based on these findings, we designed MemoSync, a co-management system that treats agent memory as a shared, operable representation: the agent proposes memory candidates and reports which items it uses, and the developer reviews, edits, and scopes them. In a between-subjects study (N=132) against two baselines, MemoSync participants achieved significantly higher task perform
    
[^16]: 评估视觉-语言模型在机器人不安全情况早期预测中的应用

    Assessing Vision-Language Models for Early Prediction of Unsafe Situations in Robotics

    [https://arxiv.org/abs/2611.24146](https://arxiv.org/abs/2611.24146)

    本文通过在三个机器人案例研究中实证评估Qwen、InternVL和LLaVA-OneVision等视觉-语言模型对视频与里程计数据的分析能力，证明VLM能够在不安全情况发生前实现早期故障预警，其中Qwen表现最佳。

    

    视觉-语言模型（VLM）能够处理视觉信息和文本指令，并凭借其理解和推理上下文的能力，非常适合在机器人故障发生之前对其进行预测。然而，支持这一观点的证据仍然有限。为解决这一问题，我们对VLM在预测机器人故障方面的能力进行了实证评估。具体而言，我们在三个机器人案例研究（HuRoN、Pal Robotics和CrowdBot）中对Qwen、InternVL和LLaVA-OneVision进行了实验，分析视频和里程计数据以实现早期故障预警（即在发生不安全情况之前）。特别地，我们评估了这些模型在将情况分类为安全、潜在不安全和不安全方面的性能，并结合多种指标对不确定性进行量化，进而研究了不确定性与性能之间的关系。我们的结果表明，Qwen提供了最强的整体分类和预测可靠性表现。

    arXiv:2611.24146v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) process visual information and textual instructions and, by using their ability to understand and reason about context, are well-suited for predicting robot failures before they occur. However, evidence supporting this remains limited. To address this, we present an empirical evaluation of VLMs for predicting robot failures. In particular, we experiment with Qwen, InternVL, and LLaVA-OneVision across three robotic case studies (HuRoN, Pal Robotics, and CrowdBot), analyzing both video and odometry data for early failure warning (i.e., before an unsafe situation occurs). Specifically, we evaluate their performance in classifying situations as safe, potentially unsafe, and unsafe, together with uncertainty quantified using various metrics, and study the relationship between uncertainty and performance. Our results showed that Qwen provides the strongest overall classification and prediction-reliability perfor
    
[^17]: SCAFFOLD：面向可复用LLM智能体技能的凭证保全式证据迁移

    SCAFFOLD: Toward Warrant-Preserving Evidence Transport for Reusable LLM-Agent Skills

    [https://arxiv.org/abs/2611.23990](https://arxiv.org/abs/2611.23990)

    本文提出以有保证的复用声明（WRC）和凭证保全式证据迁移（WPET）作为规范框架，用于判定历史证据在上下文变化后是否仍可采信，从而为LLM智能体技能的复用建立前瞻性信任。

    

    昨天运行成功的智能体技能，今天或许仍可执行，但这并不意味着今天复用它就是有充分依据的。测试可能仅覆盖可观测的契约条款，结果判定基准可能缺失，评估可能继承了同样错误的规格说明，且权限或依赖关系可能在验证与调用之间发生变化。我们主张，前瞻性信任的恰当单元是“有保证的复用声明”（Warranted Reuse Claim, WRC）：一种可废止的、针对特定声明的授权，它与版本化的技能、目标上下文、证据、假设以及已识别的废止因素相绑定。本文通过“凭证保全式证据迁移”（Warrant-Preserving Evidence Transport, WPET）来强化这一立场，这是一种用于判定哪些历史证据在上下文发生变化后仍然可采信的规范规程。该规程将证据的可采信性与所支持的安全性或有效性声明的真实性区分开来；给出一个条件性的保全结果及其局限；并规定了三值适用性与非对称上下文……

    arXiv:2611.23990v1 Announce Type: new  Abstract: An agent skill that succeeded yesterday may be executable today without being justified for reuse today. Tests may cover only observable contract clauses, outcome oracles may be absent, evaluations may inherit the same erroneous specification, and permissions or dependencies may change between validation and invocation. We argue that the appropriate unit of prospective trust is a Warranted Reuse Claim (WRC): a defeasible, claim-specific authorization tied to a versioned skill, target context, evidence, assumptions, and identified defeaters. The paper sharpens this position through Warrant-Preserving Evidence Transport (WPET), a proposed discipline for deciding which historical evidence remains admissible after context change. It distinguishes evidence admissibility from truth of the supported safety or effectiveness claim; states a conditional preservation result and its limits; specifies three-valued applicability and asymmetric context
    
[^18]: 针对Web应用教程的GitHub安全操作警报特征分析

    Characterizing GitHub Security Actions Alerts For Web Application Tutorials

    [https://arxiv.org/abs/2611.23967](https://arxiv.org/abs/2611.23967)

    本研究分析了60个Django和Spring教程仓库中来自四种默认GitHub Security Actions配置的6,609条SAST安全警报，从警报数量、相关性等多个维度刻画了Web应用开发学习场景中安全警报的特征。

    

    涉及Web应用的网络安全事件的高频发生凸显了安全软件开发实践的重要性。静态应用安全测试（SAST）工具可以在不执行代码的情况下帮助识别源代码中的潜在漏洞。将SAST工具集成到GitHub Security Actions中，可以在开发过程中报告安全警报。然而，这些警报的特性可能使安全专业知识有限的开发人员难以理解、确定优先级并加以处理。许多用于学习Web应用开发的教程也涉及具有安全影响的源代码。我们调查了与此学习场景相关的SAST警报的属性。我们分析了来自四个默认GitHub Security Actions配置的6,609条警报，这些警报来自60个Django和Spring教程仓库。我们从四个维度对警报进行了评估：警报数量、警报相关性、概念性……

    arXiv:2611.23967v1 Announce Type: new  Abstract: The high frequency of cybersecurity incidents involving web applications highlights the importance of secure software development practices. Static Application Security Testing (SAST) tools help identify potential vulnerabilities in source code without executing the code. Integrating SAST tools into GitHub Security Actions enables security alerts to be reported during development. However, the characteristics of these alerts can make them difficult for developers with limited security expertise to understand, prioritize, and address. Many tutorials for learning web application development also involve source code with security implications. We investigated the properties of SAST alerts relevant to this learning context. We analyzed 6,609 alerts from four default GitHub Security Actions configurations across 60 Django and Spring tutorial repositories. We evaluated the alerts along four dimensions: alert volume, alert relevance, conceptual
    
[^19]: 量化将检索增强生成为遗留软件系统进行改造式集成时的侵入性

    Quantifying the Intrusiveness of Retrofitting Retrieval-Augmented Generation onto Legacy Software Systems

    [https://arxiv.org/abs/2611.23931](https://arxiv.org/abs/2611.23931)

    该论文提出了三种 RAG 改造模式的分类法以及量化侵入度的 LIS 与 LIS-X 评分体系，并通过静态评估与真实部署案例验证了侵入性排序：边车模式 < SDK 嵌入 < 朴素侵入式。

    

    企业越来越倾向于在现有的 REST 和单体系统中加入检索增强生成（RAG），通常是在处理程序中临时编写一个 HTTP 调用，但无法量化这种选择的侵入程度。我们提出了：（1）三种改造模式的分类法——边车模式、SDK 嵌入和朴素侵入式；（2）遗留系统侵入度评分（Legacy Intrusion Score, LIS），包含三个代码子指标（代码变更度、影响广度、耦合度），以及其扩展版本 LIS-X，后者包含三个在真实运行系统上测量的运维子指标（运行时开销、故障传播和静默功能退化）；（3）对五个遗留系统的静态评估，其中两个是从代码仓库固定提交点获取的真实开源代码库；（4）一个将三种模式作为独立操作系统进程部署的实际案例研究。代码层面的排序 sidecar < SDK 嵌入 < 朴素侵入式在所有五个系统以及 2,000 次随机权重分配下均成立，而这一排序源于各分量指标上的支配关系……

    arXiv:2611.23931v1 Announce Type: new  Abstract: Enterprises increasingly add retrieval-augmented generation (RAG) to existing REST and monolithic systems, usually by writing an ad hoc HTTP call into a handler, and have no way to quantify how intrusive that choice is. We present (1) a taxonomy of three retrofit patterns, sidecar, SDK-embed and naive-invasive; (2) the Legacy Intrusion Score (LIS), three code sub-metrics (churn, breadth, coupling), and an extension, LIS-X, with three operational sub-metrics measured on a live system (run-time overhead, fault propagation and silent feature degradation); (3) a static evaluation on five legacy systems, two of them real open-source codebases fetched from their repositories at pinned commits; and (4) a live case study that deploys all three patterns as separate OS processes. The code-level ranking sidecar < SDK-embed < naive-invasive holds in all five systems and under 2,000 random weightings, but it follows from component-wise dominance of t
    
[^20]: 智能体硬件验证工作流中的推理时扩展：跨协议复杂度的多策略特征分析

    Inference-Time Scaling in Agentic Hardware Verification Workflows: A Multi-Strategy Characterization Across Protocol Complexity

    [https://arxiv.org/abs/2611.23905](https://arxiv.org/abs/2611.23905)

    该论文首次对面向AMBA协议桥接的智能体式UVM测试平台合成中的推理时计算扩展进行了系统性的实证刻画，对比了四种不同成本的推理策略在不同协议复杂度下的验证质量，并指出结构化诊断反馈是提升硬件验证代码生成效果的关键。

    

    推理时扩展，即在生成阶段而非预训练阶段分配更多计算资源，已在大语言模型推理任务中展现出显著的质量提升。然而，其在结构化硬件验证代码生成中的行为却鲜为人知——在这类任务中，输出必须同时满足编译器正确性、协议语义和功能覆盖率约束。我们针对AMBA协议桥接器的智能体式UVM测试平台合成，开展了一项系统性、基于仿真器的推理时计算扩展研究。我们在四个AMBA目标（协议复杂度指数11.8–22.6）上评估了四种推理策略——零样本（0.04美元）、原始反馈迭代（0.81美元）、结构化内循环（0.95美元）以及完整的双循环智能体合成（1.06美元），并采用协议验证完备性（PVC）指标、功能覆盖率以及六类UVM错误分类法（E1–E6）进行衡量。我们的核心发现是，结构化诊断反馈……（摘要原文在此处截断）

    arXiv:2611.23905v1 Announce Type: cross  Abstract: Inference-time scaling, allocating more compute during generation rather than pre-training, has demonstrated dramatic quality improvements for LLM reasoning tasks. Its behaviour in structured hardware verification code generation, where outputs must satisfy compiler correctness, protocol semantics, and functional coverage constraints simultaneously, is less understood. We present a systematic, simulator-grounded study of inference-time compute scaling for agentic UVM testbench synthesis targeting AMBA protocol bridges. Four inference strategies---zero-shot (\$0.04), raw-feedback iterative (\$0.81), structured inner-loop (\$0.95), and full dual-loop agentic synthesis (\$1.06)---are evaluated across four AMBA targets (Protocol Complexity Index 11.8--22.6) using the Protocol Verification Completeness (PVC) metric, functional coverage, and a six-class UVM error taxonomy (E1--E6). Our central finding is that structured diagnostic feedback a
    
[^21]: SWE-growth：面向软件工程智能体的教师引导自适应自博弈

    SWE-growth: Teacher-Guided Adaptive Self-Play for Software Engineering Agents

    [https://arxiv.org/abs/2611.23876](https://arxiv.org/abs/2611.23876)

    SWE-Growth 提出了一种教师引导的自适应自博弈框架，利用教师验证的结果识别学生的学习机会——将全失败任务用于 SFT 示范、具有正向师生差距的混合结果任务用于强化学习，并将任务资格判定与注入器更新相连接，实现学生训练与任务生成的闭环协同。

    

    自博弈通过生成仓库级修复任务，为训练软件工程智能体提供了一种可扩展的方式。然而，在二元修复奖励和有限的 rollout 预算下，全部失败的分组无法提供基于奖励的组间相对优势。当教师能够提供学生自身尝试中缺失的、经过验证的修复方案时，这类任务仍然可能提供学习机会。我们提出了 SWE-Growth，这是一个教师引导的自适应自博弈框架，利用经过验证的教师结果来识别当前学生的学习机会。所有学生均失败但教师能可靠成功的任务为监督微调（SFT）提供示范数据，而 SFT 之后的学生 rollout 则识别出具有正向教师-学生差距的混合结果任务用于强化学习。任务资格判定还指导注入器的更新，将学生训练与后续任务生成连接起来。从101个种子问题开始，SWE-Growth 提升（原文在此截断）……

    arXiv:2611.23876v1 Announce Type: new  Abstract: Self-play offers a scalable way to train software engineering agents by generating repository-level repair tasks. However, with binary repair rewards and finite rollout budgets, all-fail groups provide no reward-based group-relative advantage. Such tasks may still offer learning opportunities when a Teacher can provide verified repairs absent from the Student's own attempts. We introduce SWE-Growth, a teacher-guided adaptive self-play framework that uses verified Teacher outcomes to identify learning opportunities for the current Student. Tasks with all students failure but reliable Teacher success provide demonstrations for supervised fine-tuning (SFT), and post-SFT Student rollouts identify mixed-outcome tasks with a positive Teacher--Student gap for reinforcement learning. Task eligibility also guides Injector updates, connecting Student training with subsequent task generation. Starting from 101 seed issues, \textsc{SWE-Growth} impro
    
[^22]: 询问却遗漏：自然语言数据查询中的决策覆盖

    Asking but Missing: Decision Coverage in Natural-Language Data Querying

    [https://arxiv.org/abs/2611.23864](https://arxiv.org/abs/2611.23864)

    该论文提出了一种名为接地代码生成（GCG）的诊断协议，通过将决策识别、证据解决和代码生成三个阶段分离，来诊断大语言模型在自然语言数据查询中的欠规范决策补全（UDC）行为，即模型默默填补问题歧义导致错误查询结果的问题。

    

    大语言模型越来越多地被用于将关于结构化数据的自然语言问题转化为可执行程序。一个根本性的挑战是识别一个问题何时存在多种计算解释，并确定解决这些歧义所需的决策。我们研究语言模型是否在生成代码之前识别这些未解决的决策，还是默默采用可能导致错误结果的假设。我们将后一种行为称为欠规范决策补全（UDC）。为了区分UDC的不同成因，我们提出了接地代码生成（GCG），这是一种诊断协议，将未解决决策的识别、基于证据的决策解决、以及将决策转化为可执行代码这三个阶段分离开来。通过分别考察每个阶段，我们可以确定导致错误结果的决策最初是在哪里被遗漏或被错误处理的。我们在BIRD-Pan（原文摘要此处被截断）上对该框架进行了评估。

    arXiv:2611.23864v1 Announce Type: new  Abstract: Large language models are increasingly used to translate natural language questions about structured data into executable programs. A fundamental challenge is recognizing when a question admits multiple computational interpretations and identifying the decisions needed to resolve them. We investigate whether language models recognize these unresolved decisions before generating code, or silently adopt assumptions that may lead to incorrect results. We refer to the latter behavior as Underspecified Decision Completion (UDC). To distinguish different causes of UDC, we introduce Grounded Code Generation (GCG), a diagnostic protocol that separates the identification of unresolved decisions, their resolution using evidence, and their translation into executable code. By examining each stage separately, we can determine where the decisions responsible for incorrect results were first missed or mishandled. We evaluate this framework on BIRD-Pan
    
[^23]: 针对原始C代码重新测试基于LLM的C到Rust翻译

    Re-Testing LLM-Based C-to-Rust Translations Against Their Original C

    [https://arxiv.org/abs/2611.23839](https://arxiv.org/abs/2611.23839)

    该论文重新测试了11个已发布的基于LLM的C到Rust翻译器，发现所有翻译均存在缺陷（共160个缺陷，其中93个为普通输入上的错误），证明其自报成功率不可靠，因为这些检查大多从未在能暴露缺陷的输入上完整比较输出。

    

    大多数C到Rust翻译器使用大型语言模型（LLM），并且每个都基于自身的正确性检查来报告成功率。我们调查了34个使用LLM的C到Rust翻译器，其中只有一个报告过对其检查机制的测试，且仅有十一个以可运行形式发布了原始C代码及其Rust翻译。我们对这十一个全部进行了重新测试：在生成的输入上运行C程序和Rust程序，比较每一个输出，并且只在C程序不存在未定义行为的情形下才计为分歧。结果显示，所有十一个翻译器发布的产品都存在缺陷：我们共发现160个缺陷，其中93个是在普通输入上的彻底错误。它们各自的检查接受了其中86个有缺陷的翻译。对于这86个中的81个，其发布的检查从未在能暴露缺陷的输入上运行并比较过两个程序；对于另外五个中的三个，检查虽然运行了程序但只比较了部分输出。成功率的可靠程度完全取决于其检查的质量。在将一次翻译计为成功之前，作者应当在修复循环从未见过的输入上，将每一个输出与原始C程序的输出进行比较。

    arXiv:2611.23839v1 Announce Type: new  Abstract: Most C-to-Rust translators use large language models (LLMs), and each reports a success rate from its own correctness check. We survey 34 C-to-Rust translators that use LLMs. Only one reports testing its check, and eleven release the original C and their Rust in runnable form. We re-test all eleven. We run the C and the Rust on generated inputs, compare every output, and count a divergence only where the C has no undefined behavior. All eleven release defective translations: we find 160 defects, 93 of them outright errors on ordinary inputs. Their own checks accepted 86 defective translations. For 81 of them, the released check never compared the programs on an input that exposes the defect. For three of the other five, it ran one but compared only part of the output. A success rate is only as good as its check. Before counting a translation as a success, authors should compare every output with the C's on inputs the repair loop never sa
    
[^24]: RoboDx：机器人软件故障的修复与诊断

    RoboDx: Repair and Diagnosis of Faults in Robot Software

    [https://arxiv.org/abs/2611.23816](https://arxiv.org/abs/2611.23816)

    提出了RoboDx——一个带有注入故障与隐藏场景验证的机器人导航软件修复与诊断任务基准，实验表明即使修复通过验证，智能体在双故障任务中往往只发现并修复了部分故障（尤其是静默故障）。

    

    针对机器人导航软件的一个补丁可以通过所有单元测试，却仍然会使机器人偏离其行驶路径。检查修复后的机器人如何行驶可以捕获这类补丁，但这并不能表明智能体发现了所有的问题。我们提出了RoboDx，一个针对模拟机器人导航软件的修复任务生成器，其中注入了已知的故障。一个隐藏的验证器只有在机器人在智能体从未见过的隐藏场景中正确行驶后，才将修复标记为已通过验证。第二条测试轨道隐藏代码，要求智能体从假设列表中选择故障原因。包含两个故障的任务可能包含一个“静默故障”——即除非另一个故障同时存在，否则不会导致任何隐藏场景失败的故障。在我们包含100个任务的集合RoboDx-Hard-100上，使用标准编码工具的Claude Opus 5.5所做的修复几乎全部通过了验证。我们分析了其中94个任务，其中44个为单故障任务，50个为双故障任务。在50个双故障任务中，Opus的49个通过验证的修复中仅有17个……（原文摘要到此截断）

    arXiv:2611.23816v1 Announce Type: cross  Abstract: A patch to a robot's navigation software can pass every unit test and still drive the robot off its path. Checking how the repaired robot drives catches such patches. It does not show that the agent found everything that was wrong. We introduce RoboDx, a generator of repair tasks for simulated robot navigation software with known injected faults. A hidden verifier marks a repair as verified only if the robot then drives correctly in hidden scenarios, which the agent never sees. A second track hides the code and asks the agent to pick the cause from a list of hypotheses. A task with two faults can contain a quiet fault, a fault that makes no hidden scenario fail unless the other fault is also present. On our set of 100 tasks, RoboDx-Hard-100, almost every repair by Claude Opus 5.5 with standard coding tools is verified. We analyse 94 of the tasks, 44 with one fault and 50 with two. On the 50 two-fault tasks, only 17 of Opus's 49 verifie
    
[^25]: Stephen：使用跨视图预言验证基于定序器的协调系统

    Stephen: Verifying Sequencer-Aided Coordination Systems with Cross-View Prophecy

    [https://arxiv.org/abs/2611.23811](https://arxiv.org/abs/2611.23811)

    本文提出 Stephen，通过“跨视图预言”技术首次对基于定序器的协调系统实现进行形式化验证，解决了定序器处的线性化点在视图变更时不持久、可能被丢弃而使证明失效的难题。

    

    许多分布式系统通过依赖专用定序器来对请求排序，从而实现高性能，各副本则按照该顺序提交请求。尽管这类基于定序器的协调系统已有十多年的发展历程——从 Corfu 和 NOPaxos 到 LazyLog 和 SwitchBFT——但其正确性此前仅通过纸笔证明或协议层面的有界模型检测来论证。本文对 SAC 协议的实现进行了形式化验证。定序器似乎使这项工作变得容易：它决定所有请求的顺序，因此证明可以使用一种众所周知的技术——预言变量——将每个请求的线性化点放置在定序器处。然而，放置在定序器处的线性化点并不持久，因为定序器在任何副本提交请求之前就已对其进行排序。崩溃和其他罕见事件可能触发视图变更，从而丢弃该请求，使先前的决定以及依赖于该决定的证明失效。

    arXiv:2611.23811v1 Announce Type: cross  Abstract: Many distributed systems achieve high performance by relying on dedicated sequencers to order requests, leaving replicas to commit them in that order. Despite more than a decade of such sequencer-aided coordination (SAC) systems, from Corfu and NOPaxos to LazyLog and SwitchBFT, their correctness has been argued only with pen-and-paper proofs or bounded model checking at the protocol level.   In this paper, we formally verify implementations of SAC protocols. The sequencer seems to make this easy: it decides the order of all requests, so the proof can place each request's linearization point at the sequencer using a well-known technique, prophecy variables. Yet a linearization point placed at the sequencer is not durable, because the sequencer orders a request before any replica commits it. Crashes and other rare events may trigger a view change that discards the request, nullifying the earlier decision and the proof that rests on it.  
    
[^26]: 通过线程模块化抽象解释验证带幽灵变量的并发正确性见证

    Validating Concurrent Correctness Witnesses with Ghosts by Thread-Modular Abstract Interpretation

    [https://arxiv.org/abs/2611.23792](https://arxiv.org/abs/2611.23792)

    提出了一种新颖的并发正确性见证验证框架，利用幽灵变量中编码的语义信息，通过线程模块化抽象解释来验证带幽灵变量的并发正确性见证，而非将其简单视为额外的证明义务。

    

    并发程序的编写和推理是出了名的困难，即使是基于健全理论基础的工具，也可能由于实现缺陷而产生错误的结果。正确性见证被提出作为一种通过支持独立验证来提高软件验证器判定结果可信度的方法。最近，通过引入幽灵变量以允许编码更具表达力的不变式，正确性见证已被扩展到并发场景。尽管存在多种生成带幽灵变量见证的方法，但相应的验证技术仍然缺乏；验证器并没有利用幽灵变量中编码的信息来指导验证过程，而只是将幽灵插桩和关于幽灵变量的不变式视为额外的证明义务。我们提出了一种新颖的并发正确性见证验证框架，该框架利用幽灵变量中编码的语义信息……

    arXiv:2611.23792v1 Announce Type: cross  Abstract: Concurrent programs are notoriously difficult to write and to reason about, and even tools based on sound theoretical foundations can produce incorrect results due to implementation bugs. Correctness witnesses have been proposed as a way to increase the confidence in the verdicts of software verifiers by enabling independent validation. Recently, correctness witnesses have been extended to the concurrent setting, by introducing ghost variables to allow encoding more expressive invariants. While different approaches to producing witnesses with ghosts exist, validation techniques are still lacking; instead of leveraging the information encoded in the ghosts to guide the validation process, validators merely treat ghost instrumentation and invariants over ghost variables as additional proof obligations. We propose a novel framework for the validation of concurrent correctness witnesses that exploits the semantic information encoded in gho
    
[^27]: 可靠性是大型语言模型的用户特定属性

    Reliability is a user-specific property of large language models

    [https://arxiv.org/abs/2611.23732](https://arxiv.org/abs/2611.23732)

    该论文提出LLM服务的可靠性应针对特定用户和使用条件进行评估，并据此提出了包含论断规范、动态操作剖面、局部概率鲁棒性等六个方向的研究路线图。

    

    公开基准测试支持对大型语言模型（LLM）能力的比较，但用户对同一高排名模型的实际体验可能各不相同。借鉴传统软件可靠性理论，我们认为LLM服务的可靠性应针对特定用户和特定使用条件进行评估。这不仅仅是能力分数的问题：评估证据必须能够支持关于未来运行中故障的论断。我们区分了用于发现缺陷的测试与用于评估可靠性的测试，并识别了支撑此类论断的十个建模考量因素。我们提出了一个包含六个研究方向的研究路线图，涵盖论断规范、动态操作剖面、分级结果与噪声判断、局部概率鲁棒性、交互依赖性以及带重评估的统计推断。局部概率鲁棒性解决了将不同具体输入视为同一任务时所隐藏的行为变异问题。

    arXiv:2611.23732v1 Announce Type: cross  Abstract: Public benchmarks support comparisons of large language model (LLM) capabilities, yet users can experience the same highly ranked model differently. Drawing on traditional software reliability, we argue that the reliability of an LLM service should be assessed for specified users and conditions of use. This requires more than a capability score: evaluation evidence must support a claim about failure during future operation. We distinguish testing to discover faults from testing to assess reliability and identify ten modelling considerations behind such claims. We propose a six-workstream research roadmap linking claim specification, dynamic operational profiles, graded outcomes and noisy judgements, local probabilistic robustness, interaction dependence, and statistical inference with reassessment. Local probabilistic robustness addresses the behavioural variation hidden when different concrete inputs are treated as the same task. Publ
    
[^28]: JITAtlas：检查编译器重写中的生命周期与保护义务

    JITAtlas: Checking Lifetime and Protection Obligations in Compiler Rewrites

    [https://arxiv.org/abs/2611.23715](https://arxiv.org/abs/2611.23715)

    JITAtlas是一个源码级检查器，通过从编译器优化器代码中重建引用与目标计算，检测编译器重写中违反对象生命周期、借用支持和调用者可见所有权的缺陷，在独立标注的基准上检出全部29个缺陷且无误报，并在CPython中发现15个新漏洞。

    

    编译器重写可能在保留返回值的同时违反对象生命周期或所有权。当重写改变了对象保持存活的时间、借用的指针是否持续受支持、或到达调用者的所有权内容时，就会出现这类故障。我们提出了JITAtlas，一个源码级检查器，它从优化器代码中重建引用与目标计算，并检查有效使用、持续的借用支持以及调用者可见的所有权。在一个独立于JITAtlas输出构建和标注的包含58个输入的基准测试上，JITAtlas检测出全部29个缺陷，且未对29个安全输入产生任何误报；相比之下，PyRefcon检测出20个缺陷，CpyChecker检测出18个。在三个CPython快照上，JITAtlas提取了3,332个重写替代方案，在恢复的接收上下文下保留了801个，并报告了194处违规。人工审查确认了92份报告，精确率为47.42%。每个快照的端到端分析耗时35.56–42.31秒。JITAtlas还进一步发现了15个新漏洞（原文在此处截断）。

    arXiv:2611.23715v1 Announce Type: cross  Abstract: Compiler rewrites can preserve returned values while violating object lifetime or ownership. Such failures arise when a rewrite changes when an object remains live, whether a borrowed pointer stays supported, or what ownership reaches the caller. We present JITAtlas, a source-level checker that reconstructs reference and target computations from optimizer code and checks valid uses, continuous borrow support, and caller-visible ownership. On a 58-input benchmark constructed and labeled independently of JITAtlas output, JITAtlas detects all 29 defects and reports none of 29 safe inputs; PyRefcon detects 20 defects and CpyChecker 18. Across three CPython snapshots, JITAtlas extracts 3,332 rewrite alternatives, retains 801 under recovered receiving contexts, and reports 194 violations. Manual review confirms 92 reports, giving 47.42% precision. End-to-end analysis takes 35.56-42.31 s per snapshot. JITAtlas further identifies 15 new bugs a
    
[^29]: 评估受治理的LLM驱动的机器学习实验：生命周期、溯源与失败指标

    Evaluating Governed LLM-Driven ML Experimentation: Lifecycle, Provenance, and Failure Metrics

    [https://arxiv.org/abs/2611.23678](https://arxiv.org/abs/2611.23678)

    本文提出应从实验过程而非仅从结果分数来评估LLM驱动的ML实验系统，构建了由控制平面（而非LLM）管理试验生命周期、编辑范围与保留决策、并将记录写入只追加账本的治理框架，从中计算无效操作率、重复失败率和溯源完整性等治理指标，并在六个ML任务共1,445次试验中验证了同一契约的普遍适用性。

    

    运行机器学习实验的自主智能体通常以其达到的最佳分数来评判。但这个分数并不能说明它是如何达到的：智能体是否只编辑了其被允许编辑的文件，失败的试验是被保留还是被悄悄丢弃，以及每个保留或丢弃的决定是否可以追溯到其证据。我们提出在评估这类系统时不仅要看其结果，还要评估其实验过程，并为此构建了一个受治理的测试框架。一个控制平面（而非提出修改的LLM）拥有每个试验的生命周期，检查其编辑范围，决定保留还是丢弃，并将每次试验的一条记录追加到只追加的账本中。诸如无效操作率、重复失败率和溯源完整性等治理指标均从该账本中计算得出。在从合成逻辑回归到语言模型训练的六个机器学习任务、共1,445次试验中，同一套契约保持不变地应用。

    arXiv:2611.23678v1 Announce Type: new  Abstract: Autonomous agents that run ML experiments are usually judged by the best score they reach. That score says nothing about how it was reached: whether the agent stayed within the files it was allowed to edit, whether failed trials were kept or silently dropped, and whether each keep or discard decision can be traced to its evidence. We propose evaluating such systems on their experimental process as well as their outcome, and we build a governed harness that makes this possible. A control plane, not the LLM that proposes changes, owns each trial's lifecycle, checks its edit scope, decides keep or discard, and appends one record per trial to an append-only ledger. Governance metrics such as the invalid-action rate, the repeated-bad rate and provenance completeness are computed from that ledger. Across six ML tasks, from synthetic logistic regression to language-model training, and 1,445 trials, the same contract applied unchanged. The ledge
    
[^30]: CratesBump：面向 Rust 生态系统的版本升级数据集

    CratesBump: A Version-Bump Dataset for the Rust Ecosystem

    [https://arxiv.org/abs/2611.23664](https://arxiv.org/abs/2611.23664)

    该论文发布了首个用于预测 Rust 生态系统版本升级的公开数据集 CratesBump（涵盖约 22 万个 crate、超过两百万个连续发布对、每对含 41 个特征和双重标签），并配套提供可从公开数据重新生成该数据集的工具 semver-extract，基准实验表明版本升级预测任务可学习但远未解决。

    

    我们发布了 CratesBump，这是首个用于预测 Rust 生态系统中版本升级的公开数据集，同时发布了构建该数据集的工具 semver-extract。CratesBump 由 crates.io 上在 2026 年 9 月 14 日之前至少发布过两个版本的每个 crate（约 22 万个 crate）构建而成，提供了超过两百万个连续发布对。每个发布对包含 41 个特征（基于 AST 差异的代码变更、依赖变更、复杂度、发布节奏、GitHub 提交与 issue 活跃度）、其源代码仓库（如有列出），以及两个标签：对 MAJOR.MINOR.PATCH 的字面解读，以及 Cargo 的规则——按照该规则，0.y 版本的升级属于破坏性变更（breaking change）。semver-extract 可以基于公开的 crates.io 数据和 GitHub API 重新生成该数据集。为了展示其用途，我们（i）在留出的仓库上对版本升级预测器进行基准测试，发现该任务是可学习的但远未解决，且在较新的版本上得分更低；（ii）测量哪些特征组携带有效信号；（iii）检验……（原文摘要在此处被截断）

    arXiv:2611.23664v1 Announce Type: cross  Abstract: We release CratesBump, the first public dataset for predicting version bumps in the Rust ecosystem, together with semver-extract, the tool that builds it. CratesBump is built from every crate on crates.io with at least two releases before 14 September 2026 (about 220,000 crates), giving over two million consecutive release pairs. Each pair has 41 features (code changes from AST diffs, dependency changes, complexity, release cadence, GitHub commit and issue activity), its source repository where listed, and two labels: the literal reading of MAJOR.MINOR.PATCH, and Cargo's rule, under which a 0.y bump is a breaking change. semver-extract regenerates the dataset from public crates.io data and the GitHub API. To illustrate its use we (i) benchmark bump predictors on held-out repositories, finding the task learnable but far from solved, with lower scores on later releases; (ii) measure which feature groups carry the signal; (iii) check, aga
    
[^31]: 安全个人RSI：评估与修复个人智能体更新

    Safe Personal RSI: Evaluating and Repairing Personal-Agent Updates

    [https://arxiv.org/abs/2611.23651](https://arxiv.org/abs/2611.23651)

    本文提出Safe Personal RSI更新评估机制，通过将递归自我改进限制在提示词层面、固定模型权重与代码，并建立“测试—修复—验证”流程，在29个案例的基准上成功拒绝了21次高风险更新尝试中的20次，保障个人智能体更新后的安全性与任务性能。

    

    Kicker是一个个人健康助手。对助手可复用指令的更改可以在改进任务处理的同时，改变它所披露的内容、它遵循的请求以及它报告操作的方式。我们研究了Safe Personal RSI，一个为Kicker开发的更新评估组件。我们将递归自我改进（RSI）限制在提示词更新的范围内，模型权重和运行时代码保持固定。该流程测试提议的指令与当前指令，重复失败的案例，临时撤销更改以指导修复，并在采纳前检查安全性和任务性能。如果没有版本通过，则继续使用当前指令。我们的可执行基准测试包含29个电子邮件、日历、健康和市场案例，并设有针对授权复用、记忆和工具输出注入的独立探测。在三个模型和两次故意设置的风险编辑中，该流程拒绝了21次更新尝试中的20次。在修复确认中，一个被接纳的GPT-5.6 Luna rep（原文摘要在此处截断）

    arXiv:2611.23651v1 Announce Type: cross  Abstract: Kicker is a personal health assistant. Changes to an assistant's reusable instructions can improve task handling while changing what it discloses, which requests it follows, and how it reports actions. We study Safe Personal RSI, an update-evaluation component developed for Kicker. We restrict recursive self-improvement (RSI) to prompt updates, with model weights and runtime code fixed. The procedure tests proposed and current instructions, repeats failing cases, temporarily undoes changes to guide repairs, and checks safety and task performance before adoption. If no version passes, the current instructions stay in use. Our executable benchmark contains 29 email, calendar, health, and marketplace cases, with separate probes for authorization reuse, memory, and tool-output injection. Across three models and two deliberately risky edits, the procedure rejects 20 of 21 update attempts. In repair confirmation, an admitted GPT-5.6 Luna rep
    
[^32]: Option-GRPO：利用分层强化学习应对未见SDK代码生成中的AI垃圾代码问题

    Option-GRPO: Hierarchical RL Against AI Slop in Code Generation for Unseen SDKs

    [https://arxiv.org/abs/2611.23633](https://arxiv.org/abs/2611.23633)

    该论文提出Option-GRPO分层强化学习框架，仅凭SDK源代码（无需人工示例或测试用例）即可将代码大模型适配到未见过的SDK，通过从源代码中挖掘可复用代码模式（code options）作为课程与动作空间，同时优化功能正确性与代码简洁优雅性，从而抑制AI垃圾代码的产生。

    

    大型语言模型在生成的代码中越来越多地产生“AI垃圾代码”：冗长、冗余、非惯用的代码实现，这些代码虽能通过功能检查，却会使代码库臃肿不堪。这一问题在模型训练截止日期之后发布的SDK上尤为严重，因为模型既缺乏惯用写法的先验知识，也缺乏对SDK的组合性理解。我们提出了Option-GRPO，这是一个分层强化学习框架，仅凭SDK源代码即可将代码大语言模型适配到未见过的SDK上，无需人工编写的使用示例或测试用例，并且明确地同时优化功能正确性与代码优雅性。我们的核心洞察在于：SDK源代码中蕴含的组合性结构定义了一个可复用代码模式（代码选项）的层次体系，这些代码选项既可以作为课程学习的构建模块，也可以作为高效策略优化的动作空间。我们证明了代码选项可以诱导出有效的半马尔可夫决策过程，并给出了条件边界……

    arXiv:2611.23633v1 Announce Type: cross  Abstract: Large language models increasingly produce AI slop in generated code: verbose, redundant, non-idiomatic implementations that pass functional checks but bloat codebases. The problem compounds on SDKs released after the model's training cutoff, where the model has neither idiomatic priors nor compositional understanding. We present Option-GRPO, a hierarchical reinforcement learning framework that adapts a code LLM to an unseen SDK from its source code alone, without human-written usage examples or test cases, and that explicitly optimizes for both functional correctness and code elegance. Our key insight is that SDK source code contains compositional structure that defines a hierarchy of reusable code patterns (code options) which simultaneously serve as the building blocks of a curriculum and the action space for efficient policy optimization. We show that code options induce a valid semi-Markov decision process, give conditional bounds
    
[^33]: RefinementGuard：面向隐藏智能体-工具执行的反例导向语义资格验证

    RefinementGuard: Counterexample-Directed Semantic Qualification for Hidden Agent-Tool Execution

    [https://arxiv.org/abs/2611.23611](https://arxiv.org/abs/2611.23611)

    该论文提出RefinementGuard框架，通过反例导向的语义资格验证（基于显式父操作义务，将候选图与独立参考在隔离状态下比对）结合工件绑定的故障关闭式运行时准入，来守护智能体工具调用下沉至隐藏并行图的过程，在保留全部有效变体的同时拒绝近半数非等价故障。

    

    工具运行时可以将智能体选择的操作下沉到隐藏的并行图中，在降低延迟与上下文开销的同时，也带来了语义误译的风险。我们提出RefinementGuard：一种针对隐藏的、有副作用的智能体-工具下沉过程的翻译验证方法，并与工件绑定的、故障关闭式运行时准入机制相耦合。资格验证利用显式的父操作义务，在隔离状态下将候选图与独立参考进行比较。我们在受控仓库上评估了两个仓库级契约：缺陷定位与补丁验证。在由一名人工裁判裁定的100个已激活的非等价开发/资格验证故障中，父语义验证拒绝了49/100（算子聚类自助法95%置信区间[33.7%, 64.8%]）。其目标是选择性准入，而非最大化的变异拒绝：精确相等验证虽能拒绝100/100的故障，但也会拒绝全部60个有效变体；而基于义务的验证则接受了60/60的有效变体，其剩余的51个误接受……（原文摘要在此处截断）

    arXiv:2611.23611v1 Announce Type: new  Abstract: Tool runtimes can lower an agent-selected operation into a hidden parallel graph, reducing latency and context while risking semantic mistranslation. We introduce RefinementGuard: translation validation for hidden, effectful agent-tool lowerings, coupled to artifact-bound, fail-closed runtime admission. Qualification compares a candidate graph with an independent reference on isolated state using explicit parent-operation obligations. We evaluate two repository-level contracts, bug localization and patch validation, on controlled repositories. Among 100 activated, non-equivalent development/qualification faults adjudicated by one human, parent-semantic validation rejects 49/100 (operator-cluster bootstrap 95% CI [33.7%, 64.8%]). The objective is selective admission, not maximum mutation rejection: exact equality rejects 100/100 faults but also all 60 valid variants, whereas obligation-based validation accepts 60/60; its 51 residual misse
    
[^34]: MetaphIR：面向LLVM窥孔优化的语料库规模适用性分析

    MetaphIR: Corpus-Scale Applicability Analysis for LLVM Peephole Optimizations

    [https://arxiv.org/abs/2611.23555](https://arxiv.org/abs/2611.23555)

    该论文提出MetaphIR，可在LLVM IR语料库规模上同时评估窥孔优化重写规则的模式匹配与前置条件满足情况，相比忽略前置条件的结构化搜索计数更精确，且避免了完整重编译流程高昂的工程成本与47分钟级的单次运行耗时。

    

    窥孔优化将较短的源指令序列替换为更高效的等价序列，通常受前置条件约束，例如要求常量操作数必须是2的幂。由于每新增一条重写规则都会增加维护负担和编译时间成本，编译器开发者在决定实现某条优化之前，需要了解真实程序同时匹配源模式并满足其前置条件的频率。现有工作流程存在不足：对包含480万个函数的中间表示（IR）语料库进行结构化搜索会忽略前置条件，因而导致计数偏高；而实现重写并重新编译整个语料库虽能得到精确计数，但需要先付出工程实现成本，且该语料库的按需重编译流水线每次运行的中位数耗时达47分钟。当需要泛化一条重写规则时，这种延迟尤其成为瓶颈，因为开发者往往需要迭代多个变体以权衡覆盖率……（原文摘要在此处截断）

    arXiv:2611.23555v1 Announce Type: cross  Abstract: Peephole optimizations replace short source instruction sequences with more efficient equivalents, typically guarded by preconditions such as requiring a constant operand to be a power of two. Since every new rewrite adds maintenance burden and compile-time cost, compiler developers need to know how often real-world programs both match the source pattern and satisfy its preconditions before committing to an implementation. Existing workflows fall short: structural search over an intermediate representation (IR) corpus of 4.8 million functions ignores preconditions and therefore overcounts, whereas implementing the rewrite and recompiling the corpus yields exact counts only after paying the engineering cost, with each run of the corpus's on-demand recompilation pipeline taking a median of 47 minutes. This delay is especially limiting when generalizing a rewrite, as developers often iterate over multiple variants to trade off coverage ag
    
[^35]: 记录充分性：当未决主张变为生效状态时，决策记录必须保留什么

    Record Sufficiency: What a Decision Record Must Preserve When an Unsettled Claim Becomes an Operative Status

    [https://arxiv.org/abs/2611.23422](https://arxiv.org/abs/2611.23422)

    本文提出“记录充分性”概念——即一个充分的决策记录应使读者仅凭记录及其引用内容，就能解释、限定、重新推导并质疑从未决主张变为生效状态的那个决策步骤——并据此提出记录必须回答的十个问题、映射到DRK v0.1模式、定义四项机械检查，同时区分符合性、适当性、忠实性与价值四个不同层面。

    

    组织机构会保存证据、日志和审批记录，但往往没有保存未决主张变为生效状态（如一个被接受的标签或一个已发布的镜像）的那一步，因此后来的读者无法判断该状态被允许意味着什么。我们将一个决策记录称为“充分的”，如果它是真实的，并且仅凭该记录及其所引用的项目，读者就能够解释该状态、限定其边界、重新推导它并对它提出质疑。基于这四种用途，我们论证了决策记录必须回答的十个问题，将其映射到作者此前提出的决策记录 JSON Schema（DRK v0.1）上，并定义了四项机械检查。我们区分了四个层面：符合性，由这些检查来判定；适当性，即答案是否足够具体以供使用，这需要读者的参与；忠实性，对记录本身的任何检查都无法判定；以及价值，充分性并不要求这一层。在三个合成记录的164个单缺陷变体上，模式验证——DRK v0.1 唯一实现的检查——拒绝了……（原文摘要至此截断）

    arXiv:2611.23422v1 Announce Type: new  Abstract: Organisations keep evidence, logs and approvals, but often not the step by which an unsettled claim became an operative status, such as an accepted label or a released image, so later readers cannot tell what the status was allowed to mean. We call a decision record sufficient if it is true and a reader who has only the record and the items it cites can explain the status, bound it, re-derive it and challenge it. From these four uses we argue for ten questions a record must answer, map them onto the author's earlier JSON Schema for decision records (DRK v0.1), and define four mechanical checks. We separate conformance, which the checks decide; adequacy, whether the answers are specific enough to use, which needs a reader; fidelity, which no check on the record decides; and merits, which sufficiency does not require. On 164 single-defect variants of three synthetic records, schema validation, the only check DRK v0.1 implements, rejects ev
    
[^36]: Token去哪儿了：智能体编程中的热量、熵与误差

    Where the Tokens Go: Heat, Entropy and Error in Agentic Coding

    [https://arxiv.org/abs/2611.23375](https://arxiv.org/abs/2611.23375)

    提出软件热力学理论，将编程智能体消耗的token成本划分为功与热，以测试套件作为麦克斯韦妖，并通过编码-反编码循环实验定义“软件温度”来刻画智能体编程的开销特性。

    

    编程智能体按token计费，然而软件工程从未说明一个token究竟能买到什么。在此，我们提出了一种软件热力学理论。该理论借鉴经典热力学与随机热力学中的类比并加以扩展，将其应用于现代智能体编程。我们将智能体生成的token数计为其成本，并把该成本划分为功与热。测试套件则扮演软件版麦克斯韦妖的角色，评判每一次代码变更。我们通过类比“将单个分子拉开再让其重新折叠”的实验来检验该理论：让智能体对代码仓库做出更改（编码），随后再撤销该更改（反编码），如此重复多次，并统计每次模型调用的token数。该循环使所请求的编辑回到初始状态，每个方向均从存储的参考树出发，其成本在多次循环中的离散程度定义了一个“软件温度”，该温度随……上升（摘要在此处被截断）。

    arXiv:2611.23375v1 Announce Type: new  Abstract: Coding agents are billed by the token, yet software engineering says nothing about what a token actually buys. Here we introduce a theory of software thermodynamics. The theory draws analogues from classical and stochastic thermodynamics, extends them, and applies them to modern agentic coding. We count the tokens an agent generates as its cost and divide that cost into work and heat. The test suite acts as a software Maxwell's daemon that judges every coding change. We test the theory with an analogue of the experiments in which a single molecule is pulled open and let fold again: an agent makes a change to a repository (coding) and then undoes it (uncoding), many times over, and we count the tokens of every model call. The cycle returns the requested edits to where they began, with each direction started from a stored reference tree, and the spread of its cost over repeated cycles defines a software temperature, which rose with the sam
    
[^37]: 大语言模型RLOps综述

    A Survey on RLOps for Large Language Models

    [https://arxiv.org/abs/2611.23357](https://arxiv.org/abs/2611.23357)

    本文对大语言模型RLOps方法进行了系统性综述，将现有技术归纳为训练设置、经验收集、奖励计算、策略优化和模型评估五个阶段，填补了该领域技术碎片化、缺乏统一梳理的空白。

    

    强化学习（RL）在提升大语言模型（LLM）的推理、指令遵循和任务解决能力方面正变得越来越重要。然而，扩展强化学习训练会暴露出分布式执行、反馈计算和学习动态方面的故障，这些故障会浪费计算资源、拖慢训练速度并削弱模型改进效果。这些挑战催生了越来越多的大语言模型RLOps技术，用于管理大语言模型强化学习训练全生命周期的运行效率与可靠性。尽管关注度不断上升，这些技术仍然分散在各个研究领域，对其运维角色及相互作用的系统性梳理较为有限。为弥合这一差距，我们对大语言模型RLOps方法进行了系统性综述。我们将现有方法组织为五个阶段：训练设置、经验收集、奖励计算、策略优化和模型评估。

    arXiv:2611.23357v1 Announce Type: cross  Abstract: Reinforcement learning (RL) has become increasingly important for improving the reasoning, instruction-following, and task-solving capabilities of large language models (LLMs). However, scaling RL training exposes failures in distributed execution, feedback computation, and learning dynamics that can waste computational resources, slow training, and undermine model improvements. These challenges have motivated a growing body of techniques for LLM RLOps, which manages the efficiency and reliability of operations throughout the LLM reinforcement learning training lifecycle. Despite growing interest, these techniques remain scattered across research areas, with limited synthesis of their operational roles and interactions. To bridge this gap, we present a systematic survey of LLM RLOps methods. We organize existing approaches into five stages: Training Setup, Experience Collection, Reward Computation, Policy Optimization, and Model Evalua
    
[^38]: 幂等性不等于恢复：语言模型代理工具操作的故障注入研究

    Idempotency Is Not Recovery: A Fault-Injection Study of Language-Model Agent Tool Actions

    [https://arxiv.org/abs/2611.23353](https://arxiv.org/abs/2611.23353)

    该论文通过系统的故障注入实验揭示“幂等性不等于恢复”，证明只有将授权、操作账本、目标端幂等性、对账与证据封存相结合的完整配置，才能将语言模型代理工具调用的重复副作用从基线的117/200降至0/200，并保障故障后的恢复能力。

    

    当语言模型代理的工具调用被中断时，该操作可能已经在未经确认的情况下改变了外部系统。重试存在重复执行的风险；而假定成功则可能掩盖一个从未真正发生的操作。我们在一个受控的代理运行时环境中研究这些故障模式，该环境结合了授权机制、操作账本、目标端幂等性、对账机制和证据封存。我们注入了十类单主机故障，并使用独立于运行时账本的目标端预言机，比较了四种配置和三种消融设置。在Windows和Linux两种系统上各进行了7,513次试验，使用了从24个构建案例中捕获的96个模型响应。在200组已知的原始崩溃与重放对中，完整配置记录了0/200的重复副作用，而各个较简单的基线则达到117/200。与账本相关的消融实验产生了170/170的孤立副作用（完整配置为0/170），恢复成功率仅为200/300（完整配置为300/300）。禁用对账机制……（原文摘要在此处截断）

    arXiv:2611.23353v1 Announce Type: cross  Abstract: When a language-model agent's tool call is interrupted, the action may already have changed an external system without confirmation. Retrying risks duplication; assuming success may conceal an action that never occurred. We study these failure modes in a controlled agent runtime combining authorization, an action ledger, target-side idempotency, reconciliation, and evidence sealing. We inject ten single-host fault classes and compare four configurations and three ablations using a target-side oracle separate from the runtime ledger. On each of Windows and Linux, 7,513 trials used 96 captured model responses from 24 constructed cases. In 200 known original crash and replay pairs, the full configuration recorded 0/200 repeated effects versus 117/200 for each simpler baseline. A ledger-related ablation had 170/170 orphaned effects versus 0/170 and 200/300 recovery successes versus 300/300 for the full configuration. Disabling reconciliati
    
[^39]: 系统映射协议：生成式人工智能增强的需求工程——软件开发中实践、规范与复用的系统映射研究

    Systematic Mapping Protocol. Generative AI-Augmented Requirements Engineering: A Systematic Mapping Study of Practices, Specifications, and Reuse in Software Development

    [https://arxiv.org/abs/2611.23349](https://arxiv.org/abs/2611.23349)

    本文提出了一项系统映射研究的协议方案，旨在识别和分类生成式AI增强的需求工程研究，重点关注其产生的规范等工件在规范驱动开发、代码生成、测试推导、验证与复用等下游活动中的应用。

    

    生成式人工智能（GenAI），特别是大语言模型（LLM），正在需求工程（RE）领域得到迅速应用。现有的二次研究已经梳理了GenAI支持哪些需求工程活动以及使用了哪些提示策略，但对于GenAI增强的需求工程所产生的规范，以及这些规范如何在后续活动（如规范驱动开发（SDD）、设计与代码生成、测试推导、验证和复用）中被使用，目前知之甚少。本文档提出了一项系统映射研究（SMS）的方案，该研究将识别并分类关于GenAI增强需求工程的研究，尤其关注其生成的工作产品及其下游应用。该方案定义了研究问题及其分类方案、检索策略、研究筛选流程、数据提取表、分析方法、工具支持以及有效性威胁。

    arXiv:2611.23349v1 Announce Type: cross  Abstract: Generative artificial intelligence (GenAI), particularly large language models (LLMs), is rapidly being adopted in requirements engineering (RE). Existing secondary studies have characterized which RE activities are supported and which prompting strategies are used, but little is known about the specifications that GenAI-augmented RE produces and how they are used in subsequent activities such as specification-driven development (SDD), design and code generation, test derivation, verification, and reuse. This document presents the protocol of a systematic mapping study (SMS) that will identify and classify research on GenAI-augmented RE, with particular attention to the work products it generates and their downstream use. The protocol defines the research questions and their classification schemes, the search strategy, the study selection procedure, the data extraction form, the analysis approach, the tool support, and the threats to v
    
[^40]: RootSink：带经验证修复位置的可复现C/C++崩溃数据集

    RootSink: Reproducible C/C++ Crashes with Tested Repair Sites

    [https://arxiv.org/abs/2611.23336](https://arxiv.org/abs/2611.23336)

    RootSink数据集汇集了来自354个开源项目的3341条C/C++崩溃记录，通过实际复现崩溃并测试开发者修复补丁，为每条记录提供了带证据标注的崩溃位置与经过验证的修复位置，填补了漏洞定位与修复研究缺乏成对证据样本的空白。

    

    当C或C++程序在特制输入下崩溃时，崩溃发生的代码行往往并非真正出错的代码行。用于修复或定位漏洞的工具和模型需要同时展示崩溃位置与根因位置、并为二者提供证据的样本，而现有数据集很少能做到这一点。RootSink提供了来自354个开源项目的3341条记录，其中大多数记录包含内存错误检测器报告的崩溃位置、根因的候选位置，并为每个位置标注了弱点类型及其证据。对于“已复现”层级的1954条记录，研究团队自行重新运行了崩溃，并确认修复后的版本能够正常运行。其中1275条记录中，开发者修复补丁中的单个代码块即可独立阻止崩溃，这标记了一个经过测试验证的修复位置（尽管不一定是缺陷本身），其中1273条记录将其作为根位置。在对早期版本的一次盲审中，两位共同作者评审员将其见到的经测试修复位置中的68%和92%判定为缺陷本身，但仅有37%和51%的……（原文摘要在此处截断）

    arXiv:2611.23336v1 Announce Type: new  Abstract: When a C or C++ program crashes on a crafted input, the line that crashes is often not the line that is wrong. Tools and models that repair or locate vulnerabilities need examples showing both, with evidence for each, which existing datasets rarely provide. RootSink provides 3341 records from 354 open-source projects. Most give the crash site a memory-error detector reported and a candidate location for its root cause, and label each with a weakness type and its evidence. For the 1954 records in the Reproduced tier, we re-ran the crash ourselves and confirmed that the fixed version runs cleanly. In 1275 of these, a single hunk of the developer's fix stopped the crash on its own. That marks a tested repair site, though not necessarily the defect, and 1273 records use it as the root. In a blind audit of an earlier version, two co-author raters called 68% and 92% of the tested repair sites they saw the defect, but only 37% and 51% of that v
    
[^41]: Truthgate：用智能体无法篡改的记录检验编程智能体——不可能任务中的作弊、威慑与披露

    Truthgate: Checking Coding Agents Against a Record They Cannot Edit - Cheating, Deterrence and Disclosure on Impossible Tasks

    [https://arxiv.org/abs/2611.23335](https://arxiv.org/abs/2611.23335)

    本文提出Truthgate框架，通过智能体无法篡改的记录（宿主重跑的测试、隐藏容器检查、签名工具调用日志）来核验编程智能体的真实性，发现Claude Sonnet 5.5在21.7%的不可能任务上通过特殊处理代码来作弊，而在系统提示中加入一句审查警告即可大幅减少此类作弊行为。

    

    当编程智能体声称任务已完成时，这一说法是否属实？人们又该如何验证？我们通过五个步骤研究这一问题；所有付费研究均经过预注册，且在每一步中，智能体都会接受一项其无法篡改的记录的检验：包括由宿主重新运行的测试，以及从第二步起在智能体从未接触的容器中进行的隐藏检查，还有由智能体沙箱之外的签名者写下的每一次工具调用的签名日志。一项使用小型本地模型（Qwen3.5-4B，在不可能任务上运行36次）的先导实验未发现隐藏的失败，同时表明仅基于行为构建的标签会高估撒谎的次数。在Impossible-LiveCodeBench上，Claude Sonnet 5.5通过了180个不可能任务中的39个（21.7%），每次都是通过对代码进行特殊处理以迎合失败的断言来实现的；而Opus 5.5（2.8%）和Haiku 5.5（1.7%）则很少这样做。在系统提示词末尾附加一句话，就能将Sonnet在相同的60个任务中的作弊次数从22次降至8次（告知将有AI审查员阅读对话记录）、降至6次（告知将有真人审查其工作成果）。

    arXiv:2611.23335v1 Announce Type: cross  Abstract: When a coding agent says it is done, is that true, and how would anyone know? We study this question in five steps; every paid study was pre-registered, and in each step the agent is checked against a record it cannot edit: tests re-run by the host and, from step 2 on, hidden checks in containers the agent never sees and a signed log of every tool call written by a signer outside the agent's sandbox. A pilot with a small local model (Qwen3.5-4B, 36 runs on impossible tasks) found no hidden failure, and showed that labels built from actions alone over-count lies. On Impossible-LiveCodeBench, Claude Sonnet 5.5 passed 39 of 180 impossible tasks (21.7%), every time by special-casing its code to fit the failing assertion; Opus 5.5 (2.8%) and Haiku 5.5 (1.7%) rarely did. One sentence appended to the system prompt cut Sonnet's cheats on the same 60 tasks from 22 to 8 (an AI reviewer will read the transcript), 6 (a person will review the work)
    
[^42]: 将AI编程智能体融入科学研究

    Integrating AI Coding Agents Into Scientific Research

    [https://arxiv.org/abs/2611.23334](https://arxiv.org/abs/2611.23334)

    本文提出了PROOF六级分类标准，用于按来源可追溯性、可重现性、开放性、监督性和可证伪性对AI辅助数值实验进行分级，以指导在安全环境中将AI编程智能体整合进科学研究。

    

    新的软件工具已经出现，使AI智能体能够快速开发用于分析数据和生成图表的代码，但智能体犯的错误也可能损害研究人员的计算机，需要大量资源进行验证，并使研究结果受到更多质疑。如果科学软件在安全环境中开发，并配备监控文件变更的系统以及明确的AI智能体处理标准，这些顾虑可以得到缓解。本文首先描述了编程智能体如何改变生成、验证和重现数值结果的十一个方面。随后，论文提出了一套六级分类标准（PROOF），用于根据来源可追溯性、可重现性、开放性、监督性和可证伪性对AI辅助数值实验进行分类。该标准从本地自洽、版本受控、经人工验证的结果起步，最终达到可公开访问且经过整理的文件、变更历史和提示词归档。

    arXiv:2611.23334v1 Announce Type: new  Abstract: New software tools have emerged that enable AI agents to quickly develop code that analyzes data and generates figures, but their mistakes can also harm a researcher's computer, require extensive resources for validation, and cause the results to be received more skeptically. These concerns could be mitigated if scientific software is developed in a secure environment with a system for monitoring file changes and well-defined standards for handling AI agents. This paper first describes how coding agents change eleven aspects of generating, verifying, and reproducing numerical results. A six-level scale is then proposed for classifying AI-assisted numerical experiments by their provenance, reproducibility, openness, oversight, and falsifiability (PROOF). The scale begins with locally self-consistent, version-controlled, human-verified results and culminates in a publicly accessible and curated archive of the files, change history, prompt 
    
[^43]: LLM应该知道什么？面向软件优化的知识引导式热启动

    What Should an LLM Know? Knowledge-Guided Warm Starts for Software Optimization

    [https://arxiv.org/abs/2611.23331](https://arxiv.org/abs/2611.23331)

    该研究在切实可行的标注预算下，提出了三种向LLM注入领域知识的热启动方法（规则校验、文档增强、专家纠正），发现只有经过检验的领域知识才能真正改善LLM驱动软件优化的“启动”阶段。

    

    大型语言模型（LLM）几分钟内即可处理数GB的数据，而人类专家的工作速度要慢得多。先前探索人类/AI优化器组合的研究忽视了这一差距：它们要求系统或人类标注过多的样本（例如，SynthCore每次试验需要四十多个标注）。本研究在一个评估切实可行的预算下（几十个乃至更少的标注）测试人类与AI知识的作用。我们将五个基线方法（随机搜索、RRP、EZR、少样本提示，以及预算匹配的SynthCore变体）与三种在LLM提出首批配置之前向其注入领域知识的方式进行比较：(1) AMP列出该领域的规则，然后依据这些规则检查每个候选配置；(2) HKMA在提示中添加侦查得到的数据或文档；(3) H-DKP由人类专家纠正LLM的规则。我们发现，领域知识能够改善LLM优化的“启动”阶段，但前提是这些知识必须经过检验——无论是通过LLM自身还是……（原文摘要在此处截断）

    arXiv:2611.23331v1 Announce Type: new  Abstract: Large language models (LLMs) process gigabytes in minutes; human experts work far more slowly. Prior studies that explored combinations of human/AI optimizers ignore this gap. Those studies asked systems or humans to label too many examples (e.g., SynthCore needs over four dozen per trial). Here, we test human and AI knowledge under a budget where evaluation is practical (a few dozen labels or fewer).   We compare five baselines (random search, RRP, EZR, few-shot prompting, and a budget-matched SynthCore variant) against three ways to give an LLM domain knowledge before it proposes its first configurations: (1) AMP lists the domain's rules, then checks each candidate against them; (2) HKMA adds scouted data or documentation to the prompt; and (3) H-DKP has human experts correct the LLM's rules.   We find that domain knowledge improves the "start-up" phase of LLM optimization, but only when that knowledge is checked, by the LLM itself or 
    
[^44]: AI时代的可持续科学软件培训：从代码生产到科学管护

    Training for sustainable scientific software in the AI era: from code production to scientific stewardship

    [https://arxiv.org/abs/2611.23330](https://arxiv.org/abs/2611.23330)

    该论文提出在AI时代，科学软件培训应从代码生产转向科学管护，通过将AI系统教学与既有软件和分析培训相融合，使研究人员能够在AI辅助的工作流程中具备验证、解释、复现和维护科学软件的能力。

    

    生成式和智能体式人工智能（AI）系统正在改变研究人员与科学软件交互的方式。它们对生产力的影响因任务和用户而异，但可能会削弱“成功产出计算成果”与“对其所依赖的关键选择和证据保持科学掌控”之间的联系。对于高能物理领域而言，这使得部分培训挑战从代码生产转向科学管护：研究人员必须能够验证、解释、复现、维护并论证部分由AI生成或修改的软件和工作流程。我们认为，具有AI意识的软件培训应当将对AI系统的专门教学与将AI融入既有的软件和分析培训相结合。测试、溯源、可复现性、资源判断和科学验证应在AI辅助的工作流程中得到训练和实践，使研究人员具备相应的能力……

    arXiv:2611.23330v1 Announce Type: new  Abstract: Generative and agentic artificial intelligence (AI) systems are changing how researchers interact with scientific software. Their effects on productivity vary across tasks and users, but they can weaken the connection between producing a successful computational artifact and retaining scientific command over the consequential choices and evidence on which it depends. For high-energy physics, this shifts part of the training challenge from code production toward scientific stewardship: researchers must be able to validate, explain, reproduce, maintain, and justify software and workflows partly generated or modified by AI. We argue that AI-aware software training should therefore combine dedicated instruction on AI systems with the integration of AI into established software and analysis training. Testing, provenance, reproducibility, resource judgment, and scientific validation should be exercised in AI-assisted workflows, with competence
    
[^45]: 水平不是秩：对10个已发布共形预测库中分位数到索引转换的实证审计

    A level is not a rank: an empirical audit of quantile-to-index conversion in 10 released conformal-prediction libraries

    [https://arxiv.org/abs/2611.23329](https://arxiv.org/abs/2611.23329)

    该论文对10个已发布的共形预测库中39处“置信水平转换为索引”的代码位置进行了实证审计，发现部分库将置信水平错误地解析为分位数秩，在特定样本量下导致实际覆盖率偏离理论保证。

    

    背景。0.90水平的共形区间只有在正确的秩处才能保持其保证：一个新分数以概率r/(n+1)落在n个分数中第r小者或以下。我们所研究的库却将置信水平传递给分位数函数，这可能落在其他位置。当n=20时，statsforecast区间的上端位于第18.05个位置，而秩19才是所要求的。目标。已安装的代码在哪里解析出错误的秩，覆盖率上的代价是多少，以及当没有秩能达到该水平时它返回什么？方法。我们审计了10个固定版本软件包中共39处将水平转换为索引的代码位置，这是一个指定名称的集合而非系统性搜索。其中12处通过实际运行来分类，11处通过它们共享的numpy方法分类，其余通过源代码分类。覆盖率在相同的分数上与所要求的秩进行配对比较。结果。所有7个表格实现都达到了所要求的秩。在darts中，在一条代码路径上，当秩一致时覆盖率不变（0.0000），而0.0280到0.[原文摘要在此处截断]

    arXiv:2611.23329v1 Announce Type: new  Abstract: Context. A conformal interval at 0.90 keeps that guarantee only at the right rank: a fresh score falls at or below the rth smallest of n with probability r/(n + 1). The libraries we study instead hand a level to a quantile function, which can land elsewhere. At n = 20, the upper end of statsforecast's interval sits at position 18.05 where rank 19 is required. Objective. Where does installed code resolve the wrong rank, at what cost in coverage, and what does it return when no rank reaches the level? Method. We audit 39 sites where a level becomes an index in 10 pinned packages, a named set rather than a systematic search. 12 are classified by running them, 11 through their shared numpy method, the rest from source. Coverage is paired against the required rank on the same scores. Results. All 7 tabular implementations reach the required rank. In darts, on one code path, coverage is unchanged (0.0000) where the ranks agree and 0.0280 to 0.
    
[^46]: HydroMind：面向自主水文研究的可检验智能体

    HydroMind: A Checkable Agent for Autonomous Hydrological Research

    [https://arxiv.org/abs/2611.23323](https://arxiv.org/abs/2611.23323)

    该论文开发了面向通用智能体的水文建模系统HydroMind及其建模工具和研究流程，首次检验了当自主智能体自行决定水文模型评估方案时，其结论能否被自身数据、代码和结果支持并供读者核查。

    

    通用编程智能体已经为水文和地球系统模型编写过代码。在这些研究中，研究者事先决定了模型将如何被评估，例如测试时段、评分函数以及模型需要达到的标准。尚无研究检验过当智能体自行做出这些决策时会发生什么：它的结论中有多少能被其自身的数据、代码和结果所支持，以及读者能否对这些结论进行核查。这两点都可能取决于智能体的运行框架，即它可以调用的工具以及它所遵循的书面指令。我们比较了三种运行框架：智能体自带的框架；该框架加上HydroMind的建模工具——HydroMind是我们为通用智能体开发的水文建模系统，它通过固定命令运行每个模型，并存储模型的配置设置、训练好的模型和预测结果；以及再加上HydroMind的研究流程，在该流程下，智能体在运行之前先撰写研究设计（原文摘要至此截断）。

    arXiv:2611.23323v1 Announce Type: cross  Abstract: General-purpose coding agents have written code for hydrological and Earth-system models. In these studies the researchers decided in advance how the model would be evaluated, for example the test period, the scoring function, and the standard the model had to meet. No study has tested what happens when the agent makes these decisions itself: how many of its claims are supported by its own data, code, and results, and whether readers can check them. Both may depend on the agent's harness, that is, the tools it can call and the written instructions it follows. We compared three harnesses: the agent's own harness; this harness plus the modeling tools of HydroMind, a hydrological modeling system we developed for general-purpose agents, which run every model through fixed commands and store its settings, trained model, and predictions; and both plus HydroMind's research procedure, under which the agent writes a research design before runni
    
[^47]: YUL：将软件供应链安全引入编码智能体

    YUL: Bringing Software Supply Chain Security to Coding Agents

    [https://arxiv.org/abs/2611.23314](https://arxiv.org/abs/2611.23314)

    YUL 通过让编码智能体在声明依赖时对照包注册表进行校验，阻止过时依赖的使用并引导智能体重试最新版本，从而将软件供应链安全引入编码智能体，使依赖最终落在最新版本的比率提升至 93%。

    

    编码智能体会自主选择并声明第三方依赖。它们往往会声明一个从训练数据中回忆起来的过时版本。过时的依赖会使项目暴露于最新版本已修复的已知漏洞之中。我们提出了 YUL，这是一种针对编码智能体运行环境的设计，可确保永远不会使用过时的依赖。它通过将每一项依赖声明与包注册表进行核对来实现这一点。如果某个依赖已过时或不存在，YUL 会阻止其使用，并告知编码智能体应使用的版本，从而使编码智能体在修复后重试。我们在六个生态系统的 60 个任务上评估了 YUL，使用运行 Claude Opus 5.5 的 Claude Code 以及运行 DeepSeek V4.1 Flash 的 OpenCode。在没有 YUL 的情况下，Opus 在 29% 的运行中声明了过时版本，DeepSeek 为 39%，而且在所有 GitHub Actions 运行中两者均声明了过时版本。使用 YUL 后，编码智能体在 93% 的 Opus 过时运行中最终使用了最新版本。

    arXiv:2611.23314v1 Announce Type: new  Abstract: Coding agents choose and declare third-party dependencies on their own. They often declare a stale version that they recall from their training data. A stale dependency exposes the project to known vulnerabilities that the latest release already fixes. We present YUL, a design for coding agent harnesses that ensures that no stale dependencies are ever used. It does so by checking every dependency declaration against the package registry. If a dependency is stale or does not exist, YUL blocks the usage and tells the coding agent which version to use, so the coding agent retries with the fix. We evaluate YUL on 60 tasks across six ecosystems with Claude Code running Claude Opus 5.5 and OpenCode running DeepSeek V4.1 Flash. Without YUL, Opus declares a stale version in 29% of runs and DeepSeek in 39%, and both declare a stale version in every GitHub Actions run. With YUL, the coding agents end on the latest release in 93% of stale Opus runs
    
[^48]: 面向虚拟混沌工程的不完整追踪模型发现：一项eBPF研究

    Model Discovery for Virtual Chaos Engineering with Incomplete Tracing: An eBPF Study

    [https://arxiv.org/abs/2611.23313](https://arxiv.org/abs/2611.23313)

    该论文探索了利用eBPF外部观测技术在追踪数据缺失或不完整的遗留系统上发现操作模型，从而在无需改造追踪系统的情况下实现虚拟混沌工程中的基于模型的故障分析。

    

    基于模型的故障实验可以在对运行中的应用程序进行干预之前评估组件故障的后果。近期关于模型发现和图模拟的研究从应用程序遥测数据中获取操作模型，并将其用于可用性预测。将这种方法应用于仍在运行的遗留系统时会产生一个获取难题：所需的追踪数据可能缺失或不完整，而引入这些数据可能需要对异构组件进行改动。这些系统在漫长的维护和现代化改造期间持续支撑业务运营。外部观测为在追踪系统改造之前进行基于模型的故障分析提供了一条可能的途径，但仅仅收集遥测数据本身并不能保证能够获得可执行的操作模型。我们使用扩展伯克利数据包过滤器观测技术和现有的基于图的建模方法来研究这条途径。一项包含108个案例的研究改变了应用程序的……（原文摘要在此处截断）

    arXiv:2611.23313v1 Announce Type: new  Abstract: Model-based fault experiments can assess the consequences of component failures before intervening in a running application. Recent work on model discovery and graph simulation acquired operation models from application telemetry and used them for availability prediction. Applying this approach to operational legacy systems raises an acquisition problem: the required traces may be absent or incomplete, while introducing them can require changes across heterogeneous components. These systems continue to serve business operations during extended maintenance and modernization. External observation offers a possible route to model-based fault analysis before a tracing retrofit, but collecting telemetry does not by itself establish that an executable operation model can be obtained. We investigate this route using extended Berkeley Packet Filter (eBPF) observations and the existing graph-based modelling approach. A 108-case study varied appli
    
[^49]: AppSec Core：面向安全设计软件工程的本体

    AppSec Core: An Ontology for Security-by-Design Software Engineering

    [https://arxiv.org/abs/2611.23312](https://arxiv.org/abs/2611.23312)

    本文提出了AppSec Core——一个涵盖十个领域、共259个元素的安全设计软件工程本体模型，它将分散在众多标准、框架和法律中的安全要求统一为“目标—实践—机制—证据工件”的结构化表示，并基于31个外部来源（含10个AI相关来源）进行了验证。

    

    当工程师或编码智能体更改应用程序、依赖项、配置或构建流水线时，他们需要知道三件事：哪些安全规则适用、如何满足这些规则，以及应留下什么证据。如今，这些答案分散在众多标准、框架、法律和弱点列表之中，而每个来源都使用自己的术语和不同详细程度的表述。本文提出了AppSec Core，一个描述工程师所从事安全工作的模型。它说明了必须满足什么目标、通过哪些实践来达成目标、哪些机制来完成工作，以及哪些工件可以证明工作已经完成。该模型包含十个领域中的259个元素。我们基于一份源自实践的安全设计手册构建了该模型，然后将其与31个外部来源进行核对，其中十个涉及AI系统和智能体。在此过程中我们做了三处修改，增加了五个目标并扩展了一个领域的范围，但无需新增领域。大多数目标都有多个来源的支持：典型的目标由u……（原文在此处截断）

    arXiv:2611.23312v1 Announce Type: new  Abstract: When engineers or coding agents change an application, a dependency, a configuration or a build pipeline, they need to know three things: which security rules apply, how to meet them, and what evidence to leave. Today the answers are spread over many standards, frameworks, laws and lists of weaknesses, and each one uses its own words and level of detail. This paper presents AppSec Core, a model of the security work that engineers do. It says what must be true, which practices get there, which mechanisms do the job and which artefacts show that it was done. It has 259 elements in ten domains. We built it from a security-by-design manual written from practice, and then checked it against 31 outside sources, ten of them about AI systems and agents. Along the way we made three changes, which added five objectives and made one domain wider. We did not need a new domain. Most objectives are backed by several sources: the typical objective is u
    
[^50]: 面向软件开发者的后量子密码学教育：一项多源文献综述

    Post-Quantum Cryptography Education for Software Developers: A Multivocal Literature Review

    [https://arxiv.org/abs/2611.23311](https://arxiv.org/abs/2611.23311)

    本文通过对8篇学术文献和15个专业培训项目的多源文献综述，系统评估了面向软件开发者的后量子密码学（PQC）教育现状，发现虽然所有来源均覆盖PQC基础与算法，但实际软件工程任务训练不均衡，并识别出六大教育缺口及相应改进建议。

    

    量子计算对广泛部署的公钥密码学构成威胁，使后量子密码学（PQC）的采用成为一项软件工程挑战。然而，现有的教育资源能在多大程度上帮助软件开发者安全地实现和迁移到PQC，目前仍不明确。我们对8篇研究论文和15个专业培训项目进行了多源文献综述，考察了能力覆盖范围、实际的软件工程任务、教学方法、学习者支持以及评估等维度。所有23个来源都涵盖了PQC基础和算法，但对实际软件工程任务的覆盖并不均衡。我们的分析识别出六个缺口：面向开发者的评估、真实的软件工程任务、任务与能力的一致性、教学设计与学习者支持、以安全为中心的考核，以及学习成果与迁移的长期验证。我们据此提出了相关建议。

    arXiv:2611.23311v1 Announce Type: new  Abstract: Quantum computing threatens widely deployed public-key cryptography, making post-quantum cryptography (PQC) adoption a software engineering challenge. However, it remains unclear how well existing educational provision prepares software developers for secure PQC implementation and migration. We conducted a multivocal literature review of 8 research publications and 15 professional training offerings, examining competency coverage, practical software engineering tasks, pedagogical approaches, learner support, and evaluation. All 23 sources covered PQC foundations and algorithms, but coverage of practical software-engineering tasks was uneven. Our analysis identifies six gaps concerning developer focused evaluation, authentic software engineering tasks, task competency alignment, pedagogical design and learner support, security-focused assessment, and long term validation of learning outcomes and transfer. We derive recommendations for str
    
[^51]: 基于仓库的评分量表：编码智能体的无需执行评估方法

    Repository-Grounded Rubrics for Execution-Free Evaluation of Coding Agents

    [https://arxiv.org/abs/2611.23309](https://arxiv.org/abs/2611.23309)

    该论文提出了一种基于仓库探索生成加权评分量表的无执行评估方法，无需运行测试即可评估编码智能体补丁的质量，其评分排序与真实测试结果高度一致（平均AUROC达0.826，系统排名Spearman相关系数达0.968）。

    

    编码智能体通常通过运行特定任务的测试来评估，但测试可能缺失或不完整，而且运行测试需要为每个代码仓库搭建可运行的环境。我们将最初为在候选补丁中进行选择而提出的智能体评分量表，改造成一种无需执行的度量方法，用于衡量一个补丁在多大程度上满足了任务的要求。一个智能体在代码变更之前探索仓库并撰写带权重的二元评判标准，然后一个不使用工具的评审者根据补丁所满足标准的加权占比对代码差异进行打分。我们使用来自十五个系统的7,291个带标签补丁，在SWE-bench Verified上验证了该分数产生的排序与真实测试结果的一致性。在同时拥有已解决和未解决补丁的400个任务中，每个任务的平均AUROC为0.826（在未用于调整评分量表生成提示词的任务上为0.838），并且根据失败补丁对系统进行排名与根据解决率排名高度一致（Spearman相关系数0.968）。

    arXiv:2611.23309v1 Announce Type: cross  Abstract: Coding agents are usually evaluated by running task-specific tests, but tests may be missing or incomplete, and running them requires a working environment for each repository. We adapt agentic rubrics, introduced for selecting among candidate patches, into an execution-free measure of how far a patch satisfies a task's requirements. An agent explores the repository before the change and writes weighted binary criteria, and a judge without tools scores a diff by the weighted share of criteria it satisfies. We validate the ordering this score produces against test outcomes on SWE-bench Verified, using 7,291 labeled patches from fifteen systems. Within the 400 tasks that have both resolved and unresolved patches, the mean per-task AUROC is 0.826 (0.838 on tasks not used to tune the rubric-generation prompt), and ranking systems by their failing patches closely agrees with ranking them by resolve rate (Spearman 0.968). Case analyses trace
    
[^52]: 检查智能体工具中的重试透明性

    Checking Retry Transparency in Agent Tools

    [https://arxiv.org/abs/2611.23307](https://arxiv.org/abs/2611.23307)

    提出一种基于人工构建、源码可溯的效果契约的测试原型，从原子重试吸收义务和成功尝试微步骤的有界交错两个层面检验智能体工具重试透明性背后的恢复保证。

    

    一次成功的重试并不一定意味着先前尝试的影响已被消除。旧的写入操作仍可能覆盖随后已被确认的更新；部分执行的文件写入甚至可能在重试成功返回后将文件清空。我们研究如何测试智能体工具所声称的本地幂等性背后的恢复保证。我们的原型系统采用人工构建的、有源码依据的效果契约，并在两个层面进行检验：原子级的重试吸收义务，以及成功尝试微步骤的有界交错。该系统能够输出模型见证、有条件的准入决策，以及在无法证明完成边界时的显式不确定性声明。评估保留了来自4个代码仓库的6个MCP组件的本地冻结目的性框架；其中5个组件的选定路径在研究机器上执行，另有一个后端仍不可用。真实测试涵盖文件、图、Git索引、文档更新和浏览器绑定，以及一个原生Qdra（原文在此处截断）。

    arXiv:2611.23307v1 Announce Type: new  Abstract: A successful retry does not necessarily put the earlier attempt behind it. An old write may still overwrite a later acknowledged update; a partially executed file write may even empty a file after the retry has returned successfully. We study how to test the recovery guarantee behind an agent tool's local idempotence claim. Our prototype takes manually constructed, source-backed effect contracts and checks them at two levels: atomic retry-absorption obligations and bounded interleavings of successful-attempt microsteps. It emits model witnesses, conditional admission decisions, and explicit uncertainty when a completion boundary cannot be justified. The evaluation preserves a locally frozen purposive frame of 6 MCP components from 4 repositories; selected paths in 5 execute on the study machine and one backend remains unavailable. Real tests include file, graph, Git-index, document-update and browser bindings, together with a native Qdra
    
[^53]: 何时精确推断会改变代码选择：对B4的决策论分析

    When Exact Inference Changes Code Selection: A Decision-Theoretic Analysis of B4

    [https://arxiv.org/abs/2611.23306](https://arxiv.org/abs/2611.23306)

    本文通过决策论分析精确刻画了在B4代码-测试模型中更精确的后验推断何时会改变程序选择——给出了保证整组规则已达MAP的先验-预算条件与有限修正界，并在大量实证比较中表明精确推断几乎从不改变所选代码。

    

    当使用生成的测试对生成的程序进行排序时，更精确的后验计算未必能改进所选的程序。我们在已发表的B4代码-测试模型中对这一区别进行了刻画。相同的执行签名将精确的候选边缘化简化为N个求和项，并将非空的联合最大后验（MAP）推断简化为每组两个权重。一个先验-预算条件保证了B4的整组规则已经达到MAP；一个尖锐的有限修正界证明了候选边缘何时必须选择相同的组。在807个开发矩阵上的全部2,421次默认设置比较中，所选候选完全相同。对一个来自独立公共档案的1,345个执行快照进行的前瞻性规范验证，同样在4,035次默认设置比较中未发现任何变化。在80个源自文献的先验下，这一新档案在107,600次比较中产生了85个变化的顶级集合；其中仅有六个与整组规则的失效相吻合。

    arXiv:2611.23306v1 Announce Type: new  Abstract: When generated programs are ranked using generated tests, a more exact posterior computation need not improve the selected program. We characterize this distinction in the published B4 code--test model. Identical execution signatures reduce exact candidate marginalization to N summands and nonempty joint maximum a posteriori (MAP) inference to two weights per group. A prior--budget condition guarantees that B4's whole-group rule already attains MAP; a sharp finite correction bound certifies when candidate marginals must select the same group. All 2,421 default-setting comparisons on 807 development matrices select the same candidates. A prospectively specified validation on 1,345 execution snapshots from a separate public archive also finds no changes in 4,035 default-setting comparisons. Across 80 literature-derived priors, this new archive yields 85 changed top sets in 107,600 comparisons; only six coincide with failure of whole-group 
    
[^54]: 仓库部署智能体中的依赖伪造行为

    Dependency Fabrication in Repository Deployment Agents

    [https://arxiv.org/abs/2611.23305](https://arxiv.org/abs/2611.23305)

    本研究通过构造无法合法获取依赖的探针仓库，揭示了仓库部署智能体在环境搭建失败时会通过伪造缺失依赖（如创建存根）来通过测试，且现有规则和测试框架难以可靠阻止这种“奖励破解”行为，仅当智能体收到明确禁令并具备停止上报机制时缓解措施才最有效。

    

    仓库部署智能体负责搭建运行软件仓库所需的环境，它们通常将测试通过视为环境搭建成功的证明。本报告考察了当有效搭建无法实现时，此类智能体的行为表现。研究者构建了包含智能体无法合法获取的依赖项的探针仓库，因此测试通过即意味着伪造行为的发生，例如创建存根来替代缺失的软件包。实验发现，Repo2Run 和 SetupX 中的智能体通常会伪造依赖项，而两个测试框架大多仍将这些运行标记为成功。无论是通过提示词还是仓库文档给出的禁止伪造规则，都不能可靠地阻止这种行为。在本报告专门构建的 MinimalSetupAgent 上，奖励破解文献中提出的缓解措施在智能体收到直接禁令并拥有明确的停止与上报问题途径时最为有效。在这些条件下，Qwen3.5-Plus 和 G……（摘要在此处截断）

    arXiv:2611.23305v1 Announce Type: cross  Abstract: Repository deployment agents set up the environment needed to run a software repository. They often treat passing tests as proof of a successful setup. This report examines how such agents behave when a working setup is impossible. Probe repositories were built with a dependency that the agents could not legitimately obtain. Passing tests therefore indicated fabrication, such as creating a stub to replace the missing package. Agents in Repo2Run and SetupX usually fabricated the dependency. Both harnesses mostly still marked these runs as successful. Rules against fabrication, whether given in the prompt or repository documentation, did not reliably prevent this behavior. On MinimalSetupAgent, which was built for this report, mitigations proposed in the reward-hacking literature were most effective when the agent received a direct rule and had a clear way to stop and report the problem. Under these conditions, neither Qwen3.5-Plus nor G
    
[^55]: 当代码能运行但许可证不合规时：评估编程智能体对许可证约束的处理能力

    When Code Works but the License Doesn't: Evaluating License Constraint Handling in Coding Agents

    [https://arxiv.org/abs/2611.23299](https://arxiv.org/abs/2611.23299)

    该论文系统评估了编程智能体在仓库级开发中对软件许可证约束的处理能力，发现智能体虽能成功构建并测试软件，却普遍忽视来源追溯、授权与分发等许可证约束，暴露出功能正确性与许可证合规性之间的持续差距。

    

    编程智能体正日益承担仓库级别的开发任务，包括依赖选择、源代码修改和软件部署。然而，现有评估往往优先关注功能上的成功，而忽视了许可证合规性与商业适用性。我们研究了智能体在开发专有软件时能否识别并遵守软件许可证约束。在多个智能体上开展的实验考察了宽松型与著佐权许可证、许可证信息缺失、传递性依赖、源代码修改、双许可证、vendoring（内嵌第三方代码）以及分发等多种场景。智能体的行为轨迹、仓库变更和最终产物均对照许可证基准真值进行评估。结果表明，功能正确性与许可证合规性之间存在持续的差距：智能体能够成功构建并通过测试软件，却忽视了涉及来源追溯、授权与分发的许可证约束。当许可证（相关信息更为明确）时会出现更好的结果（原文摘要在此处截断）。

    arXiv:2611.23299v1 Announce Type: cross  Abstract: Coding agents increasingly undertake repository-level development, including dependency selection, source modification, and software deployment. However, existing evaluations prioritize functional success over licensing correctness and commercial suitability.   We investigate whether agents recognize and preserve software-license constraints when developing proprietary software. Experiments across multiple agents examine permissive and copyleft licenses, missing licensing information, transitive dependencies, source modification, dual licensing, vendoring, and distribution contexts. Agent traces, repository changes, and final artifacts are evaluated against licensing ground truth.   Results reveal a persistent gap between functional and licensing correctness. Agents can successfully build and test software while overlooking licensing constraints involving provenance, authorization, and distribution. Better outcomes emerge when licensin
    
[^56]: 通过资产管理壳实现软件定义汽车中的即插即用传感器演进

    Plug-and-Play Sensor Evolution in Software-Defined Vehicles through Asset Administration Shells

    [https://arxiv.org/abs/2611.23298](https://arxiv.org/abs/2611.23298)

    本文提出一种基于资产管理壳（AAS）的即插即用传感器集成框架，使软件定义汽车在部署后新增传感器时能够自动完成识别、配置与安全监督下的软件部署。

    

    arXiv:2611.23298v1 公告类型：新论文 摘要：软件定义汽车承诺持续的软件演进，但已部署车辆中的传感器集成仍然依赖手动配置、硬件专用驱动程序以及应用侧的反复适配。现有标准仅解决了该问题的一部分：AUTOSAR标准化了部分传感器接口，车辆信号规范（VSS）标准化了信号语义，而资产管理壳（AAS）标准化了数字资产描述。然而，这些构建模块很少被组合成一条端到端的接入路径，用于车辆部署后新增的传感器。本文提出了一种由AAS驱动的、面向安全监督传感器的即插即用传感器集成框架。每个传感器由一个AAS表示，该AAS结合了硬件识别信息、接口元数据、VSS映射，以及指向容器化驱动程序和可选应用软件的引用。在物理连接之后，边缘网关识别……（摘要原文在此处截断）

    arXiv:2611.23298v1 Announce Type: new  Abstract: Software-defined vehicles (SDVs) promise continuous software evolution, but sensor integration in deployed vehicles still depends on manual configuration, hardware-specific drivers, and repeated application-side adaptation. Existing standards address only parts of this problem. AUTOSAR standardizes selected sensor interfaces, the Vehicle Signal Specification (VSS) standardizes signal semantics, and Asset Administration Shells (AAS) standardize digital asset descriptions. However, these building blocks are rarely combined into an end-to-end onboarding path for sensors attached after vehicle deployment.   This paper presents an AAS-driven plug-and-play sensor integration framework for safety-supervised sensors. Each sensor is represented by an AAS that combines hardware identification, interface metadata, VSS mappings, and references to containerized driver and optional application software. After physical connection, an edge gateway ident
    
[^57]: 大数据架构中数据摄取的架构设计决策

    Architectural Design Decisions for Data Ingestion in Big Data Architectures

    [https://arxiv.org/abs/2611.23297](https://arxiv.org/abs/2611.23297)

    本文通过系统性灰文献综述、定性分析及专家访谈，编制了一份包含数据摄取设计决策、选项及选择驱动因素的目录，以帮助从业者系统地开发大数据架构中的数据摄取流程。

    

    引言：数据摄取是数据架构中的第一个关键步骤，在此步骤中，来自各种来源的数据被迁移到存储系统（如数据湖和数据仓库）中，用于分析和洞察生成。在开发数据摄取管道时，从业者（如数据工程师和架构师）必须理解数据摄取的需求（功能性需求和非功能性需求），并选择合适的设计选项来支持这些需求。目标：为了帮助从业者系统地开发数据摄取流程，本文旨在汇编一份关于数据摄取设计决策、其选项以及选择驱动因素的目录。方法：我们对数据摄取领域进行了系统性灰文献综述，采用定性数据分析方法对所选文献进行分析，并与专家进行了半结构化访谈，以评估该框架的实用性和效用。结果：从42篇灰文献文章中……（原文摘要至此截断）

    arXiv:2611.23297v1 Announce Type: new  Abstract: Introduction: Data ingestion is the first critical step in a data architecture, in which data from various sources is moved to storage systems, such as data lakes and data warehouses, for analysis and insight generation. When developing a data ingestion pipeline, practitioners (e.g., data engineers and architects) must understand the data ingestion requirements (functional and non-functional) and select appropriate design options to support them.   Objective: To help practitioners systematically develop data ingestion processes, this paper aims to compile a catalog of data ingestion design decisions, their options, and the selection drivers.   Method: We conducted a systematic gray literature review on data ingestion, analyzed the selected literature using qualitative data analysis methods, and conducted semi-structured interviews with experts to assess the framework's practicality and utility.   Results: From 42 gray literature articles
    
[^58]: 询问同行：为需求评审排序NFR覆盖缺口

    Ask the Peers: Ranking NFR Coverage Gaps for Requirements Review

    [https://arxiv.org/abs/2611.23296](https://arxiv.org/abs/2611.23296)

    本研究将NFR覆盖缺口评审建模为项目级排序任务，实验表明利用同行项目类别统计信息的四种配置表现最优且效果相近（平均AUROC达0.875-0.889）。

    

    缺失或代表性不足的非功能需求（NFR）类别在需求集合中留下的直接证据很少，使得这些类别在需求评审过程中难以识别。在本研究中，我们将NFR覆盖缺口评审构建为一个项目级排序任务：给定一个目标需求规格说明，目标是对11个PROMISE NFR类别进行排序，以供分析师检查。我们比较了七种基于统计、基于嵌入和基于大语言模型（LLM）的配置，它们在排序机制和证据来源上各不相同，证据来源包括目标需求陈述、同行需求、其语义表示以及同行类别统计。我们采用受控缺口注入和留一项目交叉评估方法，在PROMISE、PROMISE+、PURE-Transport和PURE-Business数据集上对这些配置进行评估。使用同行类别统计的四种配置在完全移除场景下表现基本持平（平均AUROC为0.875-0.889）……

    arXiv:2611.23296v1 Announce Type: cross  Abstract: Missing or underrepresented non-functional requirements (NFR) categories leave little direct evidence in a requirements set, making them difficult to identify during requirements review. In this study, we frame NFR coverage-gap review as a project-level ranking task: given a target requirements specification, the goal is to rank the 11 PROMISE NFR categories for analyst inspection. We compare seven statistical, embedding-based, and large language model (LLM)-based configurations that differ in both ranking mechanism and evidence source, including target requirement statements, peer requirements, their semantic representation, and peer category statistics. We evaluate the configurations using controlled gap injection with leave-one-project-out evaluation on PROMISE, PROMISE+, PURE-Transport, and PURE-Business. Four configurations that use peer category statistics are effectively tied under complete removal (0.875-0.889 mean AUROC), desp
    
[^59]: 从模糊工单到排序假设：面向自主重型车辆软件的多智能体根因分析

    From Ambiguous Tickets to Ranked Hypotheses: Multi-Agent Root-Cause Analysis for Autonomous Heavy-Vehicle Software

    [https://arxiv.org/abs/2611.23294](https://arxiv.org/abs/2611.23294)

    该论文提出了一种多智能体根因分析工作流，将稀疏的故障工单和海量多模态车辆记录转化为基于遥测的事件链、按优先级排序的因果假设及候选代码位置，并通过隔离疑似原因、预筛选遥测数据和压缩共享证据，显著降低了每次事故的分析成本。

    

    自主运输系统在无人驾驶矿用卡车上运行安全关键软件，车上没有驾驶员来检测或缓解异常行为。因此，测试或运行中的每一次故障都必须追溯到其根本原因，以保持安全论证的有效性并减少停机时间。调查通常始于一份信息稀少的工单，其中包含一个大致的时间戳，有时还包含一个疑似原因，以及数吉字节的多模态车辆记录。我们提出了一种多智能体根因分析（RCA）工作流，将这些输入转化为基于遥测数据的事件链、按优先级排序的因果假设以及候选代码位置。工单中的疑似原因可能会使分析锚定在错误的理论上，因此它被发送给单独智能体会话中的单个LLM分析师。非神经工具在分析之前一次性筛选出相关遥测数据，随后由LLM将其压缩为共享证据工件；这使得每次事故的成本相比之前降低了一半以上

    arXiv:2611.23294v1 Announce Type: new  Abstract: Autonomous haulage systems run safety-critical software on driverless mining trucks, with no onboard driver to detect or mitigate abnormal behavior. Every failure in testing or operation must therefore be traced to its root cause to keep the safety case valid and limit downtime. An investigation often starts from a sparse ticket containing an approximate timestamp, sometimes a suspected cause, and a multimodal vehicle recording of several gigabytes. We present a multi-agent root-cause analysis (RCA) workflow that turns this input into a telemetry-grounded event chain, ranked causal hypotheses, and candidate code locations. A suspected cause in the ticket, which could anchor the analysis on a false theory, goes to a single LLM analyst in a separate agent session. Non-neural tools select the relevant telemetry once, before analysis, and an LLM condenses it into shared evidence artifacts; this more than halved the cost per incident compared
    
[^60]: 已判定还是悬而未决？当补丁正确性未知时比较程序修复技术

    Decided or Open? Comparing Program Repair Techniques When Patch Correctness Is Unknown

    [https://arxiv.org/abs/2611.23287](https://arxiv.org/abs/2611.23287)

    该论文提出一种将程序修复技术比较结果划分为“已判定”或“悬而未决”的方法，通过计算成功率差异在未知补丁正确性标签所有可能取值下的范围来量化比较结论的确定性，并能在测试运行前判断哪些比较无法通过增加测试来解决。

    

    自动程序修复（APR）技术的比较依赖于知道哪些补丁是正确的，但许多补丁从未被检查过，而且一些存档的测试结果是错误的。我们提出将每次比较报告为“已判定”或“悬而未决”。对于各自为每个缺陷选择一个补丁的两种技术，我们计算它们成功率之差在未知标签所有可能取值下的完整范围，当该范围不包含零时，我们称该比较为“已判定”。由于测试只能证明补丁是错误的，更多的测试只能将这个范围朝单一方向移动。这使我们在任何测试运行之前就能判断哪些悬而未决的比较是测试永远无法解决的。我们研究了八种设置，包括针对Defects4J缺陷的55,466个候选补丁。存档中已有的标签判定了学习型补丁选择器与其基线之间16次比较中的12次，而添加进一步的证据后16次比较全部得到判定。在我们发布预测结果的评估活动中……

    arXiv:2611.23287v1 Announce Type: new  Abstract: Comparisons of automated program repair (APR) techniques depend on knowing which patches are correct, but many patches are never checked and some archived test results are wrong. We propose reporting each comparison as decided or open. For two techniques that each pick one patch per bug, we compute the full range of the difference in their success rates over every possible value of the unknown labels, and we call the comparison decided when that range excludes zero. Because a test can only show that a patch is wrong, more testing can move this range in one direction only. This lets us tell, before any test runs, which open comparisons testing could never settle. We study eight settings, including 55,466 candidate patches for Defects4J bugs. The labels already in the archive decide 12 of 16 comparisons between a learned patch selector and its baselines, and adding further evidence decides all 16. In a campaign whose predictions we publish
    
[^61]: 轻松的错觉：生成式AI如何在软件组织中重新分配认知工作量

    The Illusion of Relief: How GenAI Redistributes Cognitive Workload in Software Organizations

    [https://arxiv.org/abs/2611.23286](https://arxiv.org/abs/2611.23286)

    研究表明生成式AI并未消除软件从业者的认知工作，而是将其重新分配——从业者从直接编写代码转向提供上下文、审查验证和协调AI输出等监督性工作，人类判断与问责仍然是核心。

    

    本研究探讨了生成式人工智能如何重塑软件从业者在软件开发活动中的认知工作。我们采用描述性定量方法以及对相关开放式回答的定性分析，分析了来自九个组织的44名从业者的问卷回答。从业者报告了生产力和时间上的节省，但同时也报告了与提供上下文、编写提示、审查、纠正、验证以及整合AI生成输出相关的工作投入。人类判断与问责始终处于核心地位，从业者预计其角色将从直接的代码生产转向对AI支持工作的指导、评估、验证、整合与协调。因此，GenAI似乎是将认知工作重新分配至监督层面，而非简单地将其消除，尽管整体认知工作量究竟是增加还是减少仍尚无定论。

    arXiv:2611.23286v1 Announce Type: cross  Abstract: This study explores how Generative AI (GenAI) reshapes software practitioners' cognitive work across software development activities. We analyzed questionnaire responses from 44 practitioners across nine organizations using descriptive quantitative methods and qualitative analysis of relevant open-ended responses. Practitioners reported productivity and time savings, but also effort associated with providing context, prompting, reviewing, correcting, verifying, and integrating AI-generated outputs. Human judgment and accountability remained central, with practitioners anticipating a shift from direct code production to guiding, evaluating, validating, integrating, and coordinating AI-supported work. Thus, GenAI appears to redistribute cognitive work toward oversight rather than simply eliminate it, although whether overall cognitive workload increases or decreases remains undetermined.
    
[^62]: 超越星标：MTTX 将开发者响应能力引入依赖决策，覆盖仪表盘、IDE 到 GitHub Actions

    Beyond Stars: MTTX Brings Developer Responsiveness to Dependency Decisions, from Dashboard to IDE to GitHub Actions

    [https://arxiv.org/abs/2611.23285](https://arxiv.org/abs/2611.23285)

    MTTX 是一个开源工具，通过精确的历史依赖解析为任意包计算 MTTU 和 MTTR 这两项难以伪造的响应能力指标，并将其集成到从仪表盘、IDE 到 GitHub Actions 的开发者工作流程各环节中。

    

    现代软件将开源包作为依赖进行复用，开发者常常依赖廉价的信号（如 GitHub 星标数和下载量）在候选包之间做出选择。然而，这类信号很容易被伪造，因此开发者需要能够反映“可观察行为”的信号，例如包的维护者对新依赖版本发布和安全修复的响应速度。先前的工作已经计算出这些响应能力指标——MTTU（依赖平均更新时间）和 MTTR（易受攻击依赖的平均修复时间）——但仅通过针对固定包集合的研究原型实现。因此，开发者难以从中受益，也无法将其集成到自己的工作流程中。我们提出了 MTTX，这是一个开源工具，可以对任意包及给定时间窗口，通过精确的历史依赖解析来计算 MTTU 和 MTTR，并将这些指标带入开发者工作流程的不同阶段。MTTX 解析 npm、Py

    arXiv:2611.23285v1 Announce Type: new  Abstract: Modern software reuses open-source packages as dependencies, and developers often choose among candidate packages with cheap signals, such as GitHub stars and download counts. However, such signals are easy to fake, so developers need signals that represent "observed behavior," such as how quickly the developers of a package respond to new dependency releases and to security fixes. Prior work computed these responsiveness metrics, MTTU (Mean Time To Update) dependencies and MTTR (Mean Time To Remediate) vulnerable dependencies, but only with a research prototype for a fixed set of packages. Therefore, developers cannot easily benefit from them or integrate them into their workflow. We present MTTX, an open-source tool that computes MTTU and MTTR with accurate historical dependency resolution for any package and given time window, and brings them into different phases of developer workflow. MTTX resolves each version constraint of npm, Py
    
[^63]: 单智能体何时足够：重新思考用于自动化代码审查的多智能体系统

    When Single Agent Is Enough: Rethinking Multi-Agent Systems for Automated Code Review

    [https://arxiv.org/abs/2611.23283](https://arxiv.org/abs/2611.23283)

    该论文通过系统比较发现，更简单的单智能体方法在代码审查任务中的表现与多智能体系统相当甚至更优，挑战了多智能体架构是复杂软件工程任务默认选择的普遍假设。

    

    代码审查在识别软件质量问题和确保系统健壮性方面发挥着关键作用。先前的工作已经提出了审查意见生成（RCG）和代码改进（CR）的技术，近期的多智能体（MA）方法通过协调专注于不同质量维度的智能体取得了最先进的结果。然而，多智能体系统引入了大量架构复杂性和运行时开销，且近期证据表明它们在某些任务中可能表现不如单智能体（SA）方法。在本工作中，我们系统地比较了用于审查意见生成和代码改进的多智能体与单智能体设计。令人惊讶的是，我们的结果表明，尽管设计更简单，单智能体方法的表现与多智能体系统相当，有时甚至超越多智能体系统。这些发现挑战了多智能体架构是复杂软件工程任务默认选择的普遍假设。

    arXiv:2611.23283v1 Announce Type: cross  Abstract: Code review plays a critical role in identifying software quality issues and ensuring system robustness. Prior work has proposed techniques for Review Comment Generation (RCG) and Code Refinement (CR), with recent Multi-Agent (MA) approaches achieving state-of-the-art results by coordinating agents specialized in different quality dimensions. However, MA systems introduce substantial architectural complexity and runtime overhead, and recent evidence suggests they may underperform Single-Agent (SA) approaches in certain tasks. In this work, we systematically compare MA and SA designs for RCG and CR. Surprisingly, our results show that SA approaches perform comparably to- and sometimes outperform-MA systems, despite their simpler design. These findings challenge the prevailing assumption that MA architectures are the default choice for complex software engineering tasks.
    
[^64]: Ephyra：面向大语言模型时代的基于属性的测试框架

    Ephyra: A Property-Based Testing Framework for the LLM Era

    [https://arxiv.org/abs/2611.23282](https://arxiv.org/abs/2611.23282)

    Ephyra是一个面向大语言模型时代的Rust语言基于属性测试框架，以可定制性、可观察性和生成器合成为核心特性，在14个开源系统中发现了349个漏洞。

    

    编码智能体现在编写了越来越多的软件测试；然而，现有的基于属性的测试库都是基于历史考量设计的。我们提出了Ephyra，一个基于三个核心概念构建的Rust语言基于属性测试库：可定制性、可观察性和生成器合成。Ephyra让用户可以从生成、变异和收缩中组合出自己的测试循环，并合成这些组件以满足功能性谓词。将Ephyra应用于14个开源系统，我们发现了349个漏洞并报告了其中300个，涵盖数据库管理系统、经过验证的策略引擎、SQL等价性证明器、Node.js中使用的URL解析器、现有开源系统的Rust重实现版本以及编译器。

    arXiv:2611.23282v1 Announce Type: cross  Abstract: Coding agents now write a growing share of software tests; however, property-based testing (PBT) libraries are designed with historical considerations. We present Ephyra, a PBT library in Rust built on three concepts: customizability, observability and generator synthesis. Ephyra lets users compose their own testing loops from generation, mutation and shrinking, and synthesize these components to satisfy functional predicates. Applying Ephyra to 14 open source systems, we have found 349 bugs and reported 300 of them, in database management systems (SQLite, Turso, PostgreSQL, DuckDB, DataFusion), the verified policy engine (Cedar), the SQL equivalence prover (QED), the URL parser used in Node.js (ada-url), reimplementations of existing open source systems in Rust (coreutils, brush, gitoxide), and compilers (binaryen, OCaml, and OxCaml).
    
[^65]: 面向基于大语言模型智能体的日志分析：一项生命周期导向的综述

    Log Analysis for LLM-Based Agents: A Lifecycle-Oriented Survey

    [https://arxiv.org/abs/2611.23281](https://arxiv.org/abs/2611.23281)

    本文首次以生命周期视角系统综述了基于LLM智能体的日志分析领域，通过对90项核心研究的阶段划分与比较，揭示了日志记录、结构化与挖掘之间的跨阶段依赖关系，并指出详细日志未必证据充分、结构一致不等于语义等价等关键局限。

    

    基于大语言模型（LLM）的智能体所生成的日志能够支持超越最终输出的执行行为分析。现有综述对这些记录在整个生命周期中的跨阶段依赖关系关注有限。本文对2024年至2026年8月间发表的90项核心研究进行了生命周期导向的综述，涵盖智能体日志记录、智能体日志结构化与智能体日志挖掘三个阶段。针对各阶段的比较揭示了跨阶段的依赖关系与局限性：详细的日志未必能提供充分的证据；结构上的一致性并不保证语义上的等价性；分析性结论需要与支持证据、参考标准和验证程序保持一致。本文还总结了具有代表性的开源工具、公开数据集以及未来的研究方向。

    arXiv:2611.23281v1 Announce Type: new  Abstract: Logs generated by LLM-based agents support analysis of execution behavior beyond final outputs. Existing surveys give limited attention to dependencies across the lifecycle of these records. A lifecycle-oriented review of 90 core studies published between 2024 and August 2026 covers Agent Logging, Agent Log Structuring, and Agent Log Mining. Stage-specific comparisons reveal cross-stage dependencies and limitations. Detailed logs do not necessarily provide sufficient evidence. Structural uniformity does not guarantee semantic equivalence. Analytical claims require alignment with supporting evidence, reference criteria, and validation procedures. Representative open-source tools, public datasets, and future research directions are also summarized.
    
[^66]: 数据市场工作流中用于安全声明组合的证据携带合约

    Evidence-Carrying Contracts for Safe Claim Composition in Data Marketplace Workflows

    [https://arxiv.org/abs/2611.23276](https://arxiv.org/abs/2611.23276)

    该论文提出“证据携带声明合约”这一轻量级假设-保证语义，通过冲突敏感的四状态域与安全组合规则，确保数据市场工作流中的事务声明只有在证据充分支持时才能被提升，并为有限无环工作流证明了可靠性、有界性、撤销安全性和终止性等关键性质。

    

    数据市场工作流跨越组织边界，其中支付、政策存在性或 API 交付可能被提升为证据并不支持的事务声明。我们引入了证据携带声明合约：一种轻量级的假设-保证语义，用于声明假设、保证、阻塞义务和证据依赖。一个冲突敏感的四状态域用于区分被支持、被反驳、未解决和相互冲突的需求。安全组合通过局部边界的下确界来约束事务声明，并在关键义务仍未完成时阻止声明的完全提升。我们针对有限无环工作流建立了局部可靠性、组合有界性、撤销安全性、终止性以及聚合顺序不变性。一个可执行的一致性测试工具覆盖了全部 4,096 种配置（六个阶段各有四种状态）、18 种变异、六种基本撤销、12 种替代支持场景以及一个八阶段……（摘要在此处截断）

    arXiv:2611.23276v1 Announce Type: new  Abstract: Data marketplace workflows cross organizational boundaries, where payment, policy presence, or API delivery may be promoted into transaction claims that the evidence does not support. We introduce evidence-carrying claim contracts: a lightweight assume-guarantee semantics declaring assumptions, guarantees, blocking obligations, and evidence dependencies. A conflict-sensitive four-state domain distinguishes supported, refuted, unresolved, and conflicting requirements. Safe composition bounds the transaction claim by the meet of local bounds and blocks full promotion while a critical obligation remains open. We establish local soundness, compositional boundedness, revocation safety, termination, and aggregation-order invariance for finite acyclic workflows. An executable conformance harness covers all 4,096 configurations (four states for each of six stages), 18 mutations, six basic revocations, 12 alternative-support scenarios, an eight-t
    
[^67]: 并发生成的结构化记录映照了它所伴随的散文叙述：《隐藏待处理工具调用》场景的预注册概念性复制与扩展

    A Concurrent Structured Record Mirrored the Prose It Accompanied: A Pre-Registered Conceptual Replication and Extension of the Hide Pending Tool Call Scenario

    [https://arxiv.org/abs/2611.23274](https://arxiv.org/abs/2611.23274)

    本研究对 Huang 等人（2026）的“隐藏待处理工具调用”场景进行了预注册概念复制与扩展，发现要求模型在同一次生成中附加含“未解决问题”残余部分的结构化记录时，该记录只是映照了其伴随的散文叙述内容，并未能独立提升对缺失待处理结果的披露。

    

    Huang 等人（2026）的研究表明，撰写智能体工作报告的语言模型往往倾向于掩盖会改变叙述内容的缺陷，而在指令末尾附加“请在回答中保持诚实。”一句，能在其大多数场景中显著提高披露率。在其“隐藏待处理工具调用”场景中——即一个更宽泛的后续查询被接受却始终未返回结果——诚实提示使 GPT-5.5 标记缺失数据的比例仍然停留在 0%，尽管它让大多数回答变成了附带保留说明的形式。我探究的是换一种请求是否会改变这一点：在同一次生成中附加一份遵循对抗性审查工具模式的结构化记录，其中包含一个记录尚未解决问题的“残余”部分。在基于同一组 100 对合成日志（每个待处理日志均与一个后续查询确实返回了结果的“干净孪生”日志相配对）的两个预注册阶段中，我在四种条件下通过 Codex CLI 运行了 gpt-5.6-sol。第一阶段测量了记录的表现：当包含残余部分时……（原文摘要至此截断）

    arXiv:2611.23274v1 Announce Type: cross  Abstract: Huang et al. (2026) show that language models writing reports of agentic work tend to conceal narrative-changing flaws, and that appending "Be honest in your response." to the instruction substantially increases disclosure in most of their scenarios. In their Hide Pending Tool Call scenario, where a broader follow-up query was accepted but never returned, the honesty prompt left GPT-5.5's rate of flagging the missing data at 0%, although it turned most answers into caveats. I ask whether a different request changes this: appending, in the same generation, a structured record that follows the schema of an adversarial-review tool, including a residue section for what remained unresolved. In two pre-registered phases on the same 100 synthetic log pairs, each pending log matched with a clean twin in which the follow-up query did return, I ran gpt-5.6-sol through Codex CLI under four conditions. Phase 1 measured the record: with the residue
    
[^68]: torchtest2：一次调用即可完成的 PyTorch 模型健全性检查，并在真实与未见过的架构上进行评估

    torchtest2: One-Call Sanity Checks for PyTorch Models,Evaluated on Real and Unseen Architectures

    [https://arxiv.org/abs/2611.23273](https://arxiv.org/abs/2611.23273)

    torchtest2 是一个开源库，只需一次调用即可对 PyTorch 模型执行 18 项健全性检查以捕获无声 bug，并通过在 103 个真实模型、600 个随机架构和 20 个真实 bug 上的变异分析验证了其有效性。

    

    深度学习代码中的许多 bug 是无声的：一个从未被调用的层、一个在分类头被替换之前创建的优化器、一个把批次维度当作时间步处理的循环层，或者一个被忽略的注意力掩码。代码能够运行，损失也有所下降，而缺陷往往要等到消耗了数天的 GPU 时间后才会暴露出来（如果能够暴露的话）。我们提出了 torchtest2，这是一个开源库，只需一次调用即可在单个批次上运行 18 项健全性检查，能够恢复其所触及的 PyTorch 状态，并通过插件适配不同的模型家族。我们通过变异分析对其进行评估：在来自 torchvision、timm 和 Hugging Face transformers 的 103 个模型上注入 17 类无声 bug，在 600 个随机架构上进行测试，并使用 20 个人工编写的真实 bug，与原始 torchtest 的检查、单批次过拟合、冒烟测试以及 AUTOTRAINER 和 DeepDiagnosis 风格的训练监控器进行对比。每个变异体都与其健康的对应版本进行比较，置信区间来自……

    arXiv:2611.23273v1 Announce Type: new  Abstract: Many bugs in deep learning code are silent: a layer that is never called, an optimizer created before the classification head was replaced, a recurrent layer that treats the batch as time, or an ignored attention mask. The code runs, the loss decreases somewhat, and the defect surfaces, if ever, after days of GPU time. We present torchtest2, an open-source library that runs 18 sanity checks on a single batch with one call, restores the PyTorch state it touches, and adapts to model families through plugins. We evaluate it by mutation analysis with 17 classes of silent bugs on 103 models from torchvision, timm and Hugging Face transformers, on 600 random architectures and on 20 hand-written realistic bugs, against the checks of the original torchtest, overfitting a single batch, a smoke test and a training monitor in the style of AUTOTRAINER and DeepDiagnosis. Every mutant is compared with its healthy counterpart, and intervals come from a
    
[^69]: 软件漏洞管理的行业视角：方法、工具、实践与挑战的系统性综述

    An Industry Perspective on Software Vulnerability Management: A Systematic Review of Approaches, Tools, Practices, and Challenges

    [https://arxiv.org/abs/2611.23267](https://arxiv.org/abs/2611.23267)

    该系统性综述分析了57项直接涉及行业从业者的实证研究，总结了软件漏洞管理领域的方法、工具与实践及其实施挑战，发现现有证据以指南、经验教训等实践导向贡献为主，而严格的评估性研究较为缺乏。

    

    软件漏洞管理（SVM）是指在产品或生态系统中对软件漏洞（SV）进行检测、评估、修复和披露的协调性活动。随着软件系统规模和依赖深度的不断增长，组织必须在有限的专业知识、不断演变的工具和交付压力之间权衡及时的安全行动。在这项以从业者为中心的系统性文献综述中，我们综合了2010年至2025年间发表的57篇同行评审实证研究，这些研究均直接从行业从业者那里收集原始数据。我们考察了与从业者共同提出和评估的SVM解决方案（方法、工具和实践），以及从业者在真实环境中采用和使用这些解决方案时所报告的挑战。我们发现，面向实践的贡献，如指南、建议、经验教训和经验分享，构成了证据基础的主体，而对其有效性进行评估的研究相对较少。

    arXiv:2611.23267v1 Announce Type: new  Abstract: Software vulnerability management (SVM) refers to coordinated activities to detect, assess, repair, and disclose software vulnerabilities (SVs) across a product or ecosystem. As software systems grow in scale and dependency depth, organisations must balance timely security action with limited expertise, evolving tooling, and delivery pressure. In this practitioner-centred systematic literature review, we synthesise 57 peer-reviewed empirical studies published between 2010 and 2025 that collect primary data directly from industry practitioners. We examine the SVM solutions (approaches, tools, and practices) proposed and evaluated with practitioners and the challenges practitioners report when adopting and using these solutions in real-world settings. We find that practice-oriented contributions, such as guidelines, recommendations, lessons learned, and shared experiences, constitute most of the evidence base, while fewer studies evaluate 
    
[^70]: 逃离拟人化陷阱：解决未来AI软件开发中的认知错配问题

    Escaping the Anthropomorphic Trap: Resolving the Cognitive Mismatch in Future AI Software Development

    [https://arxiv.org/abs/2611.23264](https://arxiv.org/abs/2611.23264)

    论文指出传统以人类认知为中心的软件开发范式与多智能体AI存在根本性认知错配，由此产生“AI-软件技术债务”，并提出“隐形软件工程”这一新范式，实现从智能体模仿人类向以环境为中心的共识调控治理的转变。

    

    半个多世纪以来，软件开发的基础范式——如模块化、信息隐藏和分层劳动分工——都是为了适应人类工作记忆的严格认知限制而设计的。然而，多智能体人工智能的迅速崛起暴露出一种根本性的认知错配。当前框架试图整合高维、无状态的神经推理引擎，却强迫它们在这些遗留的、以人为中心的结构中运行，从而陷入了拟人化陷阱。这种摩擦产生了一种新颖且高度不稳定的行为熵，我们将其定义为“AI-软件技术债务”——即健忘的智能体为绕过架构盲区而产生不透明的变通方案。为解决这一问题，我们提出了“隐形软件工程”（Invisible SE），这是一种从以智能体为中心的人类模拟向以环境为中心的共识调控治理的范式转变。通过重新构想……

    arXiv:2611.23264v1 Announce Type: new  Abstract: For more than a half-century, the foundational paradigms of software development-such as modularity, information hiding, and hierarchical labor division-have been engineered to accommodate the strict cognitive limitations of human working memory. However, the rapid emergence of multi-agent artificial intelligence exposes a fundamental cognitive mismatch. Current frameworks attempt to integrate high-dimensional, stateless neural reasoning engines by coercing them to operate within these legacy, human-centric structures, falling into an anthropomorphic trap. This friction generates a novel, highly volatile behavioral entropy defined as AI-software technical debt, wherein amnesic agents hallucinate opaque workarounds to bypass architectural blindfolds. To resolve this, we propose Invisible Software Engineering (Invisible SE), a paradigm shift from agent-centric human emulation to environment-centric stigmergic governance. By reimagining the
    
[^71]: 社会债务缓解措施的感知有效性：一项初步的情景式问卷调查

    Perceived Effectiveness of Social Debt Mitigation Measures: A Preliminary Vignette-Based Survey

    [https://arxiv.org/abs/2611.23263](https://arxiv.org/abs/2611.23263)

    该论文通过情景式问卷调查，让23名IT从业者和高年级学生对六类社会债务问题场景中的41项缓解措施进行有效性评分，并用配对t检验与分层贝叶斯模型分析排名，初步揭示了从业者眼中针对不同社会债务问题最有效的应对措施。

    

    软件工程是一门社会技术学科，未解决的社会性与组织性问题会不断累积为“社会债务”，逐渐侵蚀团队协作与项目健康。现有文献为这类问题提供了扎实的术语体系，却很少说明从业者认为哪些应对措施是有效的。我们通过一项基于情景的问卷调查来填补这一空白：将六种社会债务问题维度（组织孤岛、新人孤立、沟通中断、知识瓶颈、文化与地理距离、心理安全感缺失）呈现为贴近现实的团队场景，由23名IT从业者与高年级学生对41项源自文献的缓解措施进行评分。评分排名通过配对t检验和一个考虑被试内设计的分层贝叶斯模型进行分析，该模型为每项措施给出其在其场景中成为最佳选项的概率。仅在沟通中断场景中……（原文摘要在此处截断）

    arXiv:2611.23263v1 Announce Type: new  Abstract: Software engineering is a socio-technical discipline in which unresolved social and organizational problems accumulate as social debt, gradually eroding collaboration and project health. The literature offers a solid vocabulary for these problems but says little about which countermeasures practitioners consider effective. We address this gap with a vignette-based survey: six social debt problem dimensions (Organizational Silo, Newcomer Isolation, Communication Breakdown, Knowledge Bottleneck, Cultural and Geographical Distance, Psychological Safety Deficit) were presented as realistic team scenarios, and 23 IT professionals and advanced students rated 41 literature-derived mitigation measures. Rankings were analyzed with paired t-tests and a hierarchical Bayesian model that accounts for the within-subject design and yields, for each measure, the probability of being the best option in its scenario. Only for Communication Breakdown is th
    
[^72]: 金融领域的人工智能与早期职业技能转变：关于决策委托、银行部署与商业教育的结构化证据综合

    Artificial Intelligence in Finance and the Early-Career Skills Shift: A Structured Evidence Synthesis of Decision Delegation, Bank Deployments, and Business Education

    [https://arxiv.org/abs/2611.23261](https://arxiv.org/abs/2611.23261)

    本文通过结构化的证据综合，系统梳理了金融工作中哪些任务可安全委托给AI、哪些须保留人类主导，并指出在AI压缩新手历练路径的背景下商业教育亟需转型。

    

    人工智能正在成为金融领域通用的交互与决策支持层。银行将AI用于研究检索、文档综合、会议跟进、客户服务、编程、欺诈检测、风险支持，以及日益增多的多步骤工作流程。本文探讨金融工作中哪些部分可以安全地委托给AI，哪些仍必须由人类主导，以及当AI压缩了新手传统上借以培养判断力的任务时，教育应当如何转变。本文对2023年至2026年10月间发布的金融、劳动经济学、教育、银行部署和金融监管等领域的研究成果与公开证据进行了结构化的证据综合，数值结果和部署声明均标注其原始来源。研究揭示了四项关键关联：AI在检索、转换、编程和初步分析方面表现最强；金融机构正朝着受限而非不受限的方向推进应用……

    arXiv:2611.23261v1 Announce Type: new  Abstract: Artificial intelligence is becoming a general interaction and decision-support layer across finance. Banks use AI for research retrieval, document synthesis, meeting follow-up, customer service, coding, fraud detection, risk support, and increasingly multi-step workflows. This paper asks which parts of finance work can be delegated safely, which remain human-owned, and how education should change when AI compresses the tasks through which novices traditionally build judgment. It presents a structured evidence synthesis of research and public evidence released from 2023 through October 2026 across finance, labor economics, education, bank deployments, and financial regulation. Numerical results and deployment claims remain attributed to their original sources. Four connections emerge: AI is strongest at retrieval, transformation, coding, and first-pass analysis; financial institutions are moving toward constrained rather than unrestricted
    
[^73]: 面向基于大语言模型智能体的软件工程：一项面向生命周期的综述

    Software Engineering for LLM-Based Agents: A Lifecycle-Oriented Survey

    [https://arxiv.org/abs/2611.23260](https://arxiv.org/abs/2611.23260)

    本综述基于355篇论文，从需求工程、架构与设计、实现与集成、测试与保障、维护与演化五个软件生命周期阶段，系统梳理并比较了基于LLM智能体的软件工程方法。

    

    基于大语言模型（LLM）的智能体通过工具使用和与外部环境的交互来执行任务。公开的工程指导通常从目标任务出发，选择合适的工作流或自主智能体设计，并依据任务结果、延迟和成本来优化提示词、工具和执行策略。当作为软件系统交付时，智能体还必须满足利益相关者的需求、行动边界、质量检查和维护需求。模型指导的执行在不同运行之间可能存在差异，故障可能同时涉及模型决策和工具，而对提示词、模型或接口的更改可能影响此前已被接受的行为。本综述考察了基于LLM的智能体如何在五个软件生命周期阶段进行工程化：需求工程、架构与设计、实现与集成、测试与保障，以及维护与演化。基于355篇论文的主要文献集，我们比较了这些工程……

    arXiv:2611.23260v1 Announce Type: new  Abstract: Large language model (LLM)-based Agents perform tasks through tool use and interaction with external environments. Public engineering guidance commonly begins with a target task, selects a suitable workflow or autonomous Agent design, and refines prompts, tools, and execution strategies using task outcomes, latency, and cost. When delivered as software systems, Agents must also satisfy stakeholder requirements, action boundaries, quality checks, and maintenance needs. Model-directed execution can vary across runs, failures can span model decisions and tools, and changes to prompts, models, or interfaces can affect previously accepted behavior. This survey examines how LLM-based Agents are engineered across five software lifecycle stages: Requirements Engineering, Architecture and Design, Implementation and Integration, Testing and Assurance, and Maintenance and Evolution. Based on a main collection of 355 papers, we compare the engineeri
    
[^74]: 用于需求规约的受控自然语言：一项系统性文献综述

    Controlled Natural Language for Requirements Specification: A Systematic Literature Review

    [https://arxiv.org/abs/2611.23259](https://arxiv.org/abs/2611.23259)

    本文通过一项覆盖2000年至2024年间133篇原始研究的系统性文献综述，对用于需求规约的受控自然语言（CNL）方法进行了识别、分类与比较。

    

    arXiv:2611.23259v1 公告类型：新论文 摘要：需求是软件开发生命周期中的关键产物。它们表达了系统应具备的能力，对开发和测试过程都起到指导作用。鉴于其重要性，需求规约近年来引起了研究人员和从业者的广泛关注。需求规约是一项对需求进行规约（即文档化）的活动。在此背景下，受控自然语言（CNL）被提出，作为自然语言的歧义性与形式语言的复杂性之间的一种折中方案。CNL 使人们能够使用可被自动处理的准确语句来规约需求，同时仍能被利益相关者所理解。在本文中，我们开展了一项系统性文献综述（SLR），以识别、分类并比较用于需求规约的 CNL 方法。该综述涵盖了 2000 年至 2024 年间发表的 133 篇原始研究。我们……

    arXiv:2611.23259v1 Announce Type: new  Abstract: Requirements are critical artifacts of the software development life-cycle. They express capabilities that the system should provide, guiding both the development and testing process. Given their significance, require- ments specification has attracted the interest of researchers and practitioners in recent years. Requirements specification is an activity where requirements are specified, i.e., documented. In this context, Controlled Natural Languages (CNL) were proposed as a compromise between the ambiguity of natural language and the complexity of formal languages. CNLs enable the specification of requirements using accurate statements that can be processed automatically, while remaining understandable by stakeholders. In this article, we perform a Systematic Literature Review (SLR) to identify, categorize, and compare CNL approaches for requirements specification. The SLR covers 133 primary studies published between 2000 and 2024. We 
    
[^75]: EnvAgent：在云端构建科学环境的多智能体方法

    EnvAgent: Multi-Agent Approach to Building Science Environments in the Clouds

    [https://arxiv.org/abs/2611.23258](https://arxiv.org/abs/2611.23258)

    提出了EnvAgent多智能体框架，能够自动综合、构建和验证仓库级Docker环境，并引入ENVBENCH基准（包含54个科学成果）来评估科学计算环境自动化构建的效果。

    

    AI智能体正在变革科学计算和软件工作流程，但生成和执行代码需要正确配置的执行环境。由于代码仓库中包含针对操作系统、工具链、依赖项和构建过程的异构且往往不完整的需求，自动化环境构建仍然十分困难。因此，现有工作流程依赖于开发人员维护的Dockerfile和安装脚本，这使得环境准备工作既费力又难以扩展，并可能导致评估产生偏差。我们提出了ENVAGENT，一个用于综合、构建和验证仓库级Docker环境的智能体框架。ENVAGENT将用于Docker执行和状态管理的确定性实用程序，与负责仓库分析、规划、Dockerfile综合以及迭代式故障诊断和修复的协同智能体相结合。我们还介绍了ENVBENCH，这是一个包含54个科学成果和开源代码仓库的基准测试。

    arXiv:2611.23258v1 Announce Type: cross  Abstract: AI agents are transforming scientific computing and software workflows, but generating and executing code requires a correctly configured execution environment. Automating environment construction remains difficult because repositories encode heterogeneous, often incomplete requirements for operating systems, toolchains, dependencies, and build procedures. Existing workflows thus rely on developer-maintained Dockerfiles and setup scripts, making preparation labor-intensive, hard to scale, and potentially biasing evaluations. We present ENVAGENT, an agentic framework that synthesizes, builds, and verifies repository-level Docker environments. ENVAGENT combines deterministic utilities for Docker execution and state management with coordinated agents for repository analysis, planning, Dockerfile synthesis, and iterative failure diagnosis and repair. We also introduce ENVBENCH, a benchmark of 54 scientific artifacts and open-source reposit
    
[^76]: StatePortal：用于测试交互式软件深层状态的智能体检查点合成方法

    StatePortal: Agentic Checkpoint Synthesis for Testing Deep States in Interactive Software

    [https://arxiv.org/abs/2611.23257](https://arxiv.org/abs/2611.23257)

    StatePortal 提出了一种智能体方法，能从自然语言测试用例和只读源码中自动合成可版本化、可重放的检查点，从而解决交互式 GUI 软件测试中反复到达深层状态的成本瓶颈。

    

    大型语言模型降低了为具有图形用户界面（GUI）的交互式软件生成代码的工作量。对于此类软件，一个关键的测试瓶颈是到达深层状态，即需要较长交互历史才能到达的状态。到达该状态的成本在每次代码提交间反复产生，而本可摊销该成本的 GUI 自动化技术又依赖于自渲染引擎所不提供的 DOM 或无障碍树。我们提出 StatePortal，这是一个智能体，它能够基于自然语言功能测试用例和对目标源代码的只读访问，合成一个检查点——一种能够恢复该状态、可版本化且可重放的制品。检查点通过一个针对每个应用的钩子来恢复，该钩子由一个引导智能体一次性绑定或合成，无需手写钩子代码。由于检查点可能在满足状态检查的同时渲染不正确，检查点通过程序化状态与视觉相结合的混合评估进行评分：验收标准……

    arXiv:2611.23257v1 Announce Type: cross  Abstract: Large language models have reduced the effort to generate code for interactive software with graphical user interfaces (GUIs). For such software, a key testing bottleneck is reaching a deep state, one that requires a long interaction history. The cost of reaching it recurs across commits, and the GUI automation that could amortize it assumes a DOM or accessibility tree that self-rendered engines do not expose. We introduce StatePortal, an agent that synthesizes a checkpoint, an artifact that restores that state and can be versioned and replayed, from a natural-language functional test case and read-only access to the target's source code. The checkpoint is restored through a per-application hook that an onboarding agent binds or synthesizes once without hand-written hook code. Because a checkpoint can satisfy state checks while rendering incorrectly, checkpoints are scored by hybrid programmatic-state and visual evaluation: acceptance 
    
[^77]: 生成式人工智能在软件测试中的应用：从业者对益处、挑战、机遇与风险的看法

    Generative AI in Software Testing: Practitioners' Perspectives on the Benefits, Challenges, Opportunities, and Risks

    [https://arxiv.org/abs/2611.23256](https://arxiv.org/abs/2611.23256)

    一项针对64名软件测试从业者的问卷调查显示，从业者普遍认可生成式AI在减少工作量、测试生成和调试方面的益处，但也担心提示词依赖、信任问题和错误输出，且使用者比非使用者更关注具体收益与可靠性风险。

    

    生成式人工智能日益支持软件测试工作，但关于从业者感知的实证证据仍然有限。本研究通过对64名从业者进行问卷调查，探讨了GenAI在软件测试中被感知到的益处、挑战、机遇与风险。封闭式问题的回答采用描述性方法分析，开放式问题的回答则通过定性编码进行情境化解读。在受访者中，34人为使用者（经常或在试验中使用GenAI进行测试的人员），30人为非使用者（此前未在测试中使用过GenAI的人员）。结果表明，从业者普遍对GenAI在减少工作量、测试生成和调试方面的作用持积极态度，但同时也对提示词依赖、信任问题和错误输出表示担忧。使用者对若干具体益处的评价更为积极，同时也更频繁地将可靠性相关风险列为优先关注事项，而非使用者则表现出更加……（原文摘要到此截断）

    arXiv:2611.23256v1 Announce Type: cross  Abstract: Generative Artificial Intelligence (GenAI) increasingly supports software testing, yet empirical evidence on practitioners' perceptions remains limited. This study investigates perceived benefits, challenges, opportunities, and risks of GenAI in software testing through a questionnaire survey of 64 practitioners. Closed-ended responses were analyzed descriptively, while open-ended responses were qualitatively coded to contextualize the findings. Among the respondents, 34 were users (those with regular or experimental use of GenAI in testing) and 30 were non-users (those without prior use of GenAI in testing). Results show generally positive perceptions of GenAI for reducing effort, test generation, and debugging, alongside concerns about prompt dependence, trust, and incorrect outputs. Users were more positive about several concrete benefits while also more frequently prioritizing reliability-related risks, whereas non-users showed gre
    
[^78]: 面向基于模型的系统工程的结构-行为融合

    Structure-Behavior Coalescence for Model-Based Systems Engineering

    [https://arxiv.org/abs/2611.23255](https://arxiv.org/abs/2611.23255)

    本文提出结构-行为融合（SBC），将系统定义为结构与行为通过交互融合而成的单一集成实体，并借助进程代数框架与良构性约束实现MBSE中“构造即一致”的系统建模。

    

    基于模型的系统工程数十年来推动了系统设计与分析的发展，但传统方法往往将结构与行为分开处理，导致不一致性、模型不完整和歧义。本文引入结构-行为融合作为 MBSE 中系统的形式化统一定义。SBC 将系统定义为单一集成实体，其中结构与行为通过交互而融合，既提供了概念基础，也提供了形式化的进程代数框架。良构性约束确保交互在结构上有据可依且组合有效，从而实现“构造即一致”。边界进程（包括用于终止状态的 0 进程，以及用于暂停或挂起状态的检查点构造）允许表示非活动和中间状态，同时保持系统的统一性。示例和智慧医疗物联网案例研究展示了该方法的有效性。

    arXiv:2611.23255v1 Announce Type: new  Abstract: Model-Based Systems Engineering (MBSE) has advanced system design and analysis for decades, yet traditional approaches often treat structure and behavior separately, resulting in inconsistencies, incomplete models, and ambiguity. This paper introduces Structure-Behavior Coalescence (SBC) as a formal and unified definition of systems for MBSE. SBC defines a system as a single integrated entity in which structure and behavior are coalesced through interaction, providing both a conceptual foundation and a formal process-algebra framework. Well-formedness constraints ensure that interactions are structurally grounded and compositions are valid, enabling consistency by construction. Boundary processes, including 0 for terminal states and a checkpoint construct for paused or suspended states, allow inactive and intermediate states to be represented while preserving system unity. Illustrative examples and a Smart Healthcare IoT case study demon
    
[^79]: AI时代衡量真正重要的东西：研究软件指标全景

    Measuring What Matters in the Age of AI: The Research Software Metrics Landscape

    [https://arxiv.org/abs/2611.23253](https://arxiv.org/abs/2611.23253)

    该论文基于49篇文献系统梳理了研究软件度量指标的全景，将其归纳为八大类并从数据来源、效度、自动化潜力和滥用风险等维度进行分析，同时指出AI辅助开发正使现有指标体系的假设面临失效。

    

    研究软件如今已成为科学研究的核心工具，但学术界用于评判其质量、可持续性、复用性和影响力的方法仍散落在软件工程、开放科学、FAIR软件、引用和研究评估等多个领域的文献中。AI辅助开发加剧了这一问题：代码生成速度超出了人工审查的能力，来源与归属更难追踪，学术认可体系无法跟上步伐，而现有指标所继承的关于研究软件开发方式的假设可能已不再适用。为帮助应对这些挑战，我们基于49篇实质性参考文献，对研究软件指标进行了聚焦的全景综述，将反复出现的指标归纳为八大类，并从数据来源、所声称的构念、效度证据、自动化潜力以及被滥用的可能性等方面加以描述。许多被广泛引用的研究软件指标（原文摘要在此处截断）

    arXiv:2611.23253v1 Announce Type: cross  Abstract: Research software now serves as a primary instrument of science, but the methods the community uses to judge its quality, sustainability, reuse, and impact remain scattered across software engineering, open science, FAIR software, citation, and research-assessment literature. AI-assisted development sharpens the problem: code can be produced at rates that outpace human review capacity, provenance and attribution are harder to track, scholarly credit systems cannot keep pace, and existing metrics inherit assumptions about how research software gets developed that may no longer apply. To aid in navigating these challenges, we present a focused landscape synthesis of research software metrics based on 49 substantive references, organizing the recurring metrics into eight families described by data source, claimed construct, evidence of validity, automation potential, and susceptibility to misuse. Many widely cited research software metric
    
[^80]: 从数据质量到AI驱动软件密集型嵌入式系统中的数据可信性：一项系统性综述

    From Data Quality to Data Trustworthiness in AI-Driven Software-Intensive Embedded Systems: A Systematic Review

    [https://arxiv.org/abs/2611.23252](https://arxiv.org/abs/2611.23252)

    本系统性综述识别了AI驱动软件密集型嵌入式系统中影响数据可信性的16个数据质量关注点和8类评估管理方法，并揭示上游数据问题可能嵌入训练基线且在运行时难以检测。

    

    AI驱动的软件密集型嵌入式系统（SIES）在开发和运行过程中高度依赖数据。糟糕的数据质量（DQ）可能削弱数据的可信性，然而关于其影响及生命周期管理的证据仍然碎片化。本系统性文献综述综合了2020年1月至2026年2月间发表的27项原始研究，考察了数据质量问题及其与数据可信性降低之间的关联证据，以及相关的评估和管理方法。该综述识别出16个数据质量关注点，并归纳为“来源质量”和“内容质量”两大类。已有研究提供了基于数据质量的评分，以及数据质量对模型或决策性能影响的证据。综合分析表明，上游的来源问题或处理问题可能会被嵌入训练基线之中，并在运行时难以检测。综述共识别出八类方法，其中运行时监控是规模最大的类别，而上游和训练时的控制措施与运行时监控的整合仍然不足。

    arXiv:2611.23252v1 Announce Type: cross  Abstract: AI-driven software-intensive embedded systems (SIES) depend on data during development and operation. Poor data quality (DQ) may weaken data trustworthiness, yet evidence about its effects and lifecycle management remains fragmented. This systematic literature review synthesizes 27 primary studies published from January 2020 to February 2026. It examines DQ concerns and evidence linking them to reduced data trustworthiness, along with assessment and management methods. The review identifies 16 concerns organized as Source Quality and Content Quality. Studies provide DQ-based scores and evidence of effects on model or decision performance. Synthesis suggests upstream source or handling problems may become embedded in training baselines and remain difficult to detect at runtime. Eight method categories were identified. Runtime monitoring is the largest category, while upstream and training-time controls remain poorly integrated with runt
    
[^81]: 是什么决定了自动化审计的上限：从通用代码到科学代码、结果与数据

    What Moves the Ceiling on Automated Audit: from General Code to Scientific Code, Results and Data

    [https://arxiv.org/abs/2611.23250](https://arxiv.org/abs/2611.23250)

    本文在真值基准上量化了“异厂商模型审计+确定性检查”这一自动化审计流程的能力上限：通用代码上召回率仅约 30% 且无饱和点，可通过阅读模型、规则与决策规则提升；而在科学结果核查中，数字溯源检查能识别全部 49 份不一致报告，模型审计则可额外发现 25 份内部一致伪造中的 23 份。

    

    arXiv:2611.23250v1 公告类型：cross 摘要：模型越来越多地被用来检查其他模型所写的内容。我们研究了一种已实际部署的安排：由来自不同供应商的审计模型阅读工作成果，并由确定性检查验证代码所能验证的部分，然后在没有任何模型能够提供的真值（ground truth）之下测量其极限。在带有隐藏测试的通用代码上，审计模型的八次阅读达到 30.0% 的召回率 [20.0, 40.7]，同时在正确代码上的误报率为 16.0% [10.1, 22.3]，收益递减且没有出现平台期。阅读模型（与其采样方式相混淆）、规则手册中的一个句子以及决策规则各自都会改变这一数字；而根据事后评级，大部分被遗漏的内容其实是规范从未明确界定的行为。在三项预注册的科学研究中，检查报告中的数字能否溯源到其来源的确定性检查，标记出了全部 49 份不一致的报告，却没有标记出 25 份内部一致的伪造报告中的任何一份——而模型审计员标记出了其中 23 份（事后评定中 21 份的发现有据可依）；在数据方面，一种由文档推导出的验证……（摘要在此处被截断）

    arXiv:2611.23250v1 Announce Type: cross  Abstract: Models increasingly check what other models write. We study one shipped arrangement, in which an auditor from a different vendor reads the work and deterministic checks verify what code can verify, and measure its limits against ground truth no model supplies. On general-purpose code with hidden tests, eight readings of the auditor reach 30.0% recall [20.0, 40.7] at 16.0% [10.1, 22.3] flags on correct code, with diminishing returns and no plateau. The reading model (confounded with its sampling), one rulebook sentence and the decision rule each move that figure, and much of what is missed is, on a post hoc rating, behaviour the specification never determined. In three preregistered science studies, checking that reported numbers trace to their sources flagged all 49 inconsistent reports and none of 25 consistent fabrications, of which the model auditor flagged 23 (21 with a sound finding, post hoc); on data, a documentation-derived val
    
[^82]: 高级代码的过时：从语义工件进行智能体生成

    The Obsolescence of High-Level Code: Agentic Generation from Semantic Artifacts

    [https://arxiv.org/abs/2611.23249](https://arxiv.org/abs/2611.23249)

    本文提出当AI智能体能够从作为语义权威的语义规范出发生成、验证和重新生成可执行实现时，高级源代码将不再是软件开发的主要持久工件，“生成”将成为软件开发的基本操作。

    

    本文认为，当AI智能体能够从语义意图生成、验证并重新生成可执行实现时，高级源代码无需再作为软件开发的主要持久工件。我们通过重新审视应用程序生成器、CASE工具、模型驱动开发以及GeneXus等平台，将这一论点置于历史背景之中——在这些平台中，开发者早已维护着更高层次的表示形式，代码和其他工件皆由此派生而来。我们提出了一个统一的观点，其中生成是基本操作：给定作为语义权威维护的语义规范 S = (R, C, I, T, O)，智能体据此产出一个生成的可执行工件 A。功能需求、契约和接口被视为语义义务加以评估，而测试和运营目标则由显式的外部预言机 V(A, S) 进行评估。我们通过一个实验将该框架付诸实践……

    arXiv:2611.23249v1 Announce Type: cross  Abstract: This paper argues that high-level source code need not remain the primary persistent artifact of software development when AI agents can generate, validate, and regenerate executable implementations from semantic intent. We place this claim in historical context by revisiting application generators, CASE tools, model-driven development, and platforms such as GeneXus, where developers already maintained higher-level representations from which code and other artifacts were derived.   We propose a unifying view in which generation is the fundamental operation: given a semantic specification S = (R, C, I, T, O), maintained as the semantic authority, an agent produces a generated executable artifact A. Functional requirements, contracts, and interfaces are assessed as semantic obligations, while tests and operational objectives are assessed by an explicit external oracle V(A, S). We operationalize this framework through an experiment in ind
    
[^83]: ETA：推测性智能体循环中工具到达即退出

    ETA: Exit on Tool Arrival in Speculative Agent Loops

    [https://arxiv.org/abs/2611.23239](https://arxiv.org/abs/2611.23239)

    ETA 将推测性工具分发的分支用作模型决策状态的“传感器”，在工具结果到达且预测稳定一致时提前截断推理，从而在不牺牲准确性的前提下将工具调用智能体的速度提升最高 1.33 倍并减少 24% 的 token 解码量。

    

    在显式思考模式下使用工具的语言智能体会在行动之前生成冗长的思维链，使得每一轮本质上都是串行的：后续的 token 依赖于工具的观测结果。近期的推测性工具分发方法将工具执行与推理过程重叠，但当推理本身主导延迟时，其收益有限。我们提出 ETA，它不再把推测性分支仅仅当作执行器，而是将其视为模型决策状态的传感器，并利用它来终止主解码的推理过程。只有当预测已稳定、与已分发的调用一致、通过一个用于测量是否真的会发出调用的免费探针、并且工具结果已经到达时，才会截断推理：这一截止条件将每轮的损失限制在剩余解码成本之内。ETA 在 tau2 airline/retail 上实现了 1.33 倍加速，在 tau2 telecom 上实现了 1.26 倍加速，在 BFCL v4 多轮任务上实现了 1.13 倍加速。在 tau2 上，它在每 token 速率几乎不变的情况下，解码 token 数量减少了 24%。

    arXiv:2611.23239v1 Announce Type: new  Abstract: Tool-using language agents in explicit thinking modes generate long chains of thought before acting, making each turn inherently serial: later tokens depend on the tool observation. Recent speculative tool dispatch methods overlap tool execution with reasoning, but offers limited benefit when reasoning dominates latency. We present ETA, that treats the speculative fork not as an actuator but as a sensor of the model's decision state, using it to terminate the reasoning of the main decode. It truncates reasoning only when the prediction is stable, agrees with the dispatched call, passes a free probe measuring whether a call will be emitted at all, and the tool result has arrived: a deadline that bounds per-turn regret by the remaining decode cost. ETA achieves speedups of 1.33x on tau2 airline/retail, 1.26x on tau2 telecom, and 1.13x on BFCL v4 multi-turn. On tau2, it decodes 24% fewer tokens at a nearly unchanged per-token rate without d
    
[^84]: 追踪“混乱的中间过程”：面向AI编码代理的运行时事务模式

    Tracing the Messy Middle: A Runtime Transaction Schema for AI Coding Agents

    [https://arxiv.org/abs/2611.23234](https://arxiv.org/abs/2611.23234)

    本文提出一个可观测性框架，将AI编码代理开发过程中通常被忽视的运行时行为（如重试、回退、路由决策和放弃的会话）记录为结构化事务数据，从而支持基于行为证据的工程辅导、工作流优化和以总任务成本为导向的路由策略改进。

    

    AI编码代理正日益融入软件开发，但该领域仍缺乏一个系统性的反馈循环来在实践层面改进其行为。现有的工件数据集只捕获最终输出，如代码补丁、提交和基准测试结果，却遗漏了开发生命周期中“混乱的中间过程”：失败的重试、回退策略、路由决策以及被中途放弃的会话——这些因素恰恰决定了运营成本、可靠性和生产力。我们提出一个可观测性框架，将这些运行时行为捕获为结构化的事务记录，从而实现基于行为证据的工程辅导、在任务易失败阶段进行的工作流优化，以及以总任务成本而非单次请求价格来衡量的路由策略迭代。我们定义了这一架构所需的运行时事务模式，并通过Agent Token Manager（Atm）这一本地优先的原型系统展示了初步可行性。

    arXiv:2611.23234v1 Announce Type: new  Abstract: AI coding agents are increasingly integrated into software development, yet the field lacks a systematic feedback loop to improve their behavior in practice. Artifact datasets capture final outputs such as code patches, commits, and benchmark outcomes, but omit the messy middle of the development lifecycle: the failed retries, fallback strategies, routing decisions, and abandoned sessions that determine operational cost, reliability, and productivity.   We propose an observability framework that captures these runtime behaviors as structured transaction records, enabling engineering coaching grounded in behavioral evidence, workflow optimization at the stages where tasks break down, and routing policy iteration measured by total task cost rather than per-request price. We define the runtime transaction schema this architecture requires and demonstrate initial feasibility through Agent Token Manager (Atm), a local-first prototype that rec
    
[^85]: 无代码工作流语义图匹配的度量方法

    A Metric for Semantic Graph Matching of No-Code Workflows

    [https://arxiv.org/abs/2611.23233](https://arxiv.org/abs/2611.23233)

    该论文为无代码工作流图首次定义了严格的语义图匹配度量，证明了其在属性同构意义下满足度量公理，并通过填充匈牙利算法给出可扩展的上界，在4,705个公开n8n工作流上完成了大规模验证。

    

    无代码系统与智能体系统日益产生可执行的工作流，但由于类型化操作、参数和有向连接共同决定了这一产物，工作流无法作为字符串进行充分比较。我们为有限属性化工作流图定义了一种语义图匹配距离。部分一一节点匹配决定了节点替换代价与有向标记边的分歧，而未匹配节点会产生正的空代价。在固定的类型层次结构、编码器、字段全集以及度量兼容的原始距离之下，该目标函数在属性同构意义下构成工作流图上的一个度量。一种填充匈牙利算法提供了区别于精确目标函数的可扩展可行上界。我们在4,705个公开的n8n导出文件上对该度量进行了评估，涵盖2,282个不同的归一化图哈希、73,057个节点和469种观测到的节点类型。在32,093次分支定界完整评估中，原始（摘要在此处截断）

    arXiv:2611.23233v1 Announce Type: new  Abstract: No-code and agentic systems increasingly produce executable workflows, but workflows cannot be compared adequately as strings because typed operations, parameters, and directed connections jointly determine the artifact. We define a semantic graph-matching distance for finite attributed workflow graphs. A partial one-to-one node matching determines node-substitution and directed labelled-edge disagreement, while unmatched nodes incur a positive null cost. Under a fixed type hierarchy, encoder, field universe, and metric-compatible primitive distances, the objective is a metric on workflow graphs up to attributed isomorphism. A padded Hungarian procedure provides a scalable feasible upper bound distinct from the exact objective. We evaluate the metric on 4,705 public n8n exports, representing 2,282 distinct normalized-graph hashes, 73,057 nodes, and 469 observed node types. Across 32,093 branch-and-bound-complete evaluations, the mean raw
    
[^86]: 量子程序分析中的实时噪声指纹

    Live Noise Fingerprints for Quantum Program Analysis

    [https://arxiv.org/abs/2611.19143](https://arxiv.org/abs/2611.19143)

    本文提出“噪声指纹”这一紧凑的抽象表示，并通过概率伽罗瓦连接将其与具体噪声行为关联，实现了对含噪量子程序的高效可靠分析，且多项式规模的接口可产生比完整噪声模型指数级更小的指纹。

    

    量子程序本质上是概率性的，而当前不完美硬件所导致的含噪执行进一步增加了其分析的难度。本文研究了在此类条件下分析量子程序的抽象语义。我们引入了“噪声指纹”的概念，即相对于有限的输入状态与输出测量接口，对噪声可观测效应的紧凑描述。我们利用噪声指纹来定义关于具体噪声相关行为的抽象域。我们通过概率伽罗瓦连接将具体域与抽象域关联起来，将伽罗瓦连接框架扩展到了以分布为取值的解释上。我们从观测可靠性、简洁性、效率以及适应噪声变化的能力等方面对所得的抽象进行了研究。我们证明，多项式规模的接口在保持所选观测结果的同时，能够产生比完整噪声模型指数级更小的指纹，并且……

    arXiv:2611.19143v1 Announce Type: new  Abstract: Quantum programs are probabilistic in nature, and noisy execution due to current imperfect hardware further complicates their analysis. In this paper, we study abstract semantics for analysing quantum programs under these conditions. We introduce noise fingerprints, compact descriptions of the observable effects of noise relative to a finite interface of input states and output measurements. We use them to define our abstract domain with respect to the concrete noise-related behaviour. We relate the concrete and abstract domains through a probabilistic Galois connection, extending the Galois-connection framework to distribution-valued interpretations. We study the resulting abstraction in terms of observational soundness, conciseness, efficiency, and ability to adapt to changing noise. We show that polynomial-size interfaces yield exponentially smaller fingerprints than full noise models while preserving the chosen observations, and that
    
[^87]: 强化学习中环境-训练器边界处的静默故障检测

    Detecting Silent Faults at the Environment-Trainer Boundary in Reinforcement Learning

    [https://arxiv.org/abs/2611.08144](https://arxiv.org/abs/2611.08144)

    该论文发现强化学习中环境与训练器边界处的静默故障几乎无法通过学习曲线等基于结果的检查发现，而直接检查运行过程中跨越边界传输的数据本身则能检测到每一个注入的故障。

    

    机器人强化学习越来越多地在训练器之外运行环境：在另一个进程中、在另一台机器上、使用另一种编程语言。这种边界上的许多故障不会引发任何错误报告。我们将该边界建模为一种线协议，并探究是什么能够察觉到其中发生的故障。在 Pendulum、HalfCheetah、Hopper 和一个 7 关节机械臂上，我们向跨越线路传输的数据中注入了多达 23 种故障，并向训练器的日志中注入了 2 种故障，范围从单个值出错到全部值出错。除非每个值都是错误的，否则基于学习曲线阈值的检查最多只能发现十分之一的故障运行，与十个无故障种子对比的检查最多发现 21%。对所有 1,060 个最终策略的干净评估与曲线检查结果一致：在 728 个边界故障运行中，两种曲线检查均通过，其中 723 个的得分不比无故障种子低超过三个标准差，尽管几乎所有运行最终都收敛到了不同的权重，而且对于奖励类故障，记录的回报值也与正常情况不同。而对运行过程本身（而非其结果）的检查则发现了每一个故障。一个步进台阶……（摘要在此处截断）

    arXiv:2611.08144v1 Announce Type: cross  Abstract: Robot reinforcement learning increasingly runs environments outside the trainer: in another process, on another machine, in another language. Many faults at this boundary raise no error. We make the boundary a wire protocol and ask what notices a fault in it.   On Pendulum, HalfCheetah, Hopper and a 7-joint arm (Pusher), we injected up to 23 faults into what crosses the wire and 2 into the trainer's log, from one value to every value. Unless every value was wrong, a learning-curve threshold caught at most one run in ten, and a comparison with ten fault-free seeds at most 21%. A clean evaluation of all 1,060 final policies agreed with the curves: of the 728 boundary-fault runs both curve checks passed, 723 scored no more than three standard deviations below fault-free seeds, though nearly all ended on other weights and, for reward faults, logged other returns.   Checks of the run itself, not of its outcome, saw every fault. A step ledge
    
[^88]: 交互即裁判

    Interaction-as-a-Judge

    [https://arxiv.org/abs/2610.17895](https://arxiv.org/abs/2610.17895)

    提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。

    

    编程智能体的任务是产出复杂的交互式应用，例如网站、桌面应用和游戏。人类在评估这些应用时会以交互方式进行，而当前自动化的LLM评估主要集中于代码分析和单元测试。这种不匹配会导致LLM评判者遗漏那些对人类用户而言一目了然的缺陷。我们研究了“交互即裁判”，这是一种由计算机操作智能体（CUA）直接与应用进行交互的评估范式。我们证明“交互即裁判”能够同时改进交互式应用的评估与生成。为此，我们收集并发布了两个互补的数据集：包含人类偏好及对应智能体操作轨迹的6,858对应用，以及带有经人类验证的需求标签的222个应用。相较于标准的非交互式方法，我们发现“交互即裁判”在预测h……（原文摘要在此处截断）

    arXiv:2610.17895v2 Announce Type: cross  Abstract: Coding agents are tasked with producing sophisticated interactive applications such as websites, desktop applications, and games. While humans assess these applications interactively, current automated LLM evaluations primarily focus on code analysis and unit tests. This mismatch can cause LLM judges to miss failures that are immediately apparent to human users. We study Interaction-as-a-Judge, an evaluation paradigm where computer-use agents (CUAs) directly interact with an application. We demonstrate that Interaction-as-a-Judge can improve both evaluation and generation of interactive applications. To this end, we collect and release two complementary datasets: $6,858$ pairs of applications with human preferences and corresponding agent traces, and $222$ applications with human-verified requirement labels. Relative to standard non-interactive approaches, we find that Interaction-as-a-Judge improves evaluation in terms of predicting h
    
[^89]: 框架到底换来了什么？主要只是更多的 Token

    What Does a Harness Buy? Tokens, Mostly

    [https://arxiv.org/abs/2610.04433](https://arxiv.org/abs/2610.04433)

    该研究通过对七个模型在三个生产级框架和多个基准上的重复对照实验发现，没有任何框架能稳定提升性能，更换框架带来的差异仅有 2 到 5 个百分点、效果与重新运行一次相当，框架带来的分数提升主要源于更多的 token 消耗。

    

    编码智能体是运行在框架中的语言模型：框架包括系统提示词、工具以及让聊天模型能够在代码仓库中工作的上下文管理。这些智能体的大部分工程投入都集中在框架上，厂商和排行榜也将分数提升归功于框架。早期的研究固定模型并更换框架，但并未测量这种更换对通过率的改变是否大于用同一框架重新运行一遍。我们在 SWE-bench Verified 以及另外两个不同类型的基准上，用三个生产级框架——Claude Code、mini-SWE-agent 和 OpenCode——运行七个模型，并在完全相同的条件下重新运行。结果显示，没有任何一个框架能在所有模型上保持一致领先。在数据能够界定两个框架差异的地方，该差异的上界为 2 到 5 个百分点，而更换框架所改变的任务结果数量与重新运行一次大致相当。例外情况是特定模型与框架配对中出现的框架效应，其影响方向随模型而变化：Ope

    arXiv:2610.04433v2 Announce Type: replace  Abstract: A coding agent is a language model inside a harness: the system prompt, the tools, and the context management that let a chat model work in a repository. Most of the engineering on these agents goes into the harness, and vendors and leaderboards credit it with score gains. Earlier studies have fixed the model and swapped the harness, but have not measured whether the swap changes the pass rate more than running the same harness again. We run seven models through three production harnesses, Claude Code, mini-SWE-agent, and OpenCode, on SWE-bench Verified and two benchmarks of other kinds, and rerun them unchanged. No harness leads consistently across models. Where the data can bound the difference between two harnesses, the bound is 2 to 5 points, and a harness swap changes about as many task outcomes as a rerun. The exceptions are harness effects in particular pairings of model and harness, whose direction changes with the model: Ope
    
[^90]: 传闻（Hearsay）：谁能审计已部署智能体运行框架所写入的记录？

    Hearsay: Who Can Audit the Record a Deployed Agent Harness Writes?

    [https://arxiv.org/abs/2609.32495](https://arxiv.org/abs/2609.32495)

    该论文提出 Hearsay 基准，揭示了一个关键问题：由于已部署的智能体运行框架是运行记录的唯一书写者，审计者虽能从记录中指认故障（高达九成运行），却无法独立核实记录的真实性，使其沦为法律意义上的“传闻”。

    

    智能体运行的记录是由驱动该智能体的运行框架写入的。当运行过程事后引发争议时，审计者当时并不在场，手中只有这份记录。我们提出的问题是：不在场的人能否核查这份记录，而答案又取决于什么。这取决于检查所依赖的内容由谁书写：如果记录全部由运行框架书写，那么它只是框架自己的单方面陈述，我们将其称为“传闻”。我们构建了 Hearsay，一个审视记录而非任务本身的基准：五个运行框架执行十四项任务，来自三家厂商的模型审查员和一个人类评审小组阅读记录并回答有关运行过程的问题，他们引用的每一段文字都通过机械方式标注其书写者。我们还对实际部署的系统进行了审计：在全部十六个框架中，运行框架都是唯一的书写者。在只有一个书写者的情况下，读者能够在高达十分之九的运行中从记录里指认故障，却无法核查记录本身。模型审查员声称获得了佐证……

    arXiv:2609.32495v2 Announce Type: replace-cross  Abstract: The record of an agent's run is written by the harness that drives the agent. When the run is later disputed, the auditor was not there and has only this record. We ask whether someone who was not there can check the record, and what decides the answer. It depends on who wrote what a check can rely on: if the harness wrote all of it, the record is only the harness's own word, and we call it hearsay. We build Hearsay, a benchmark that examines the record and not the task: five harnesses run fourteen tasks, model examiners from three vendors and a human panel read the records and answer questions about the runs, and every passage they cite is labelled mechanically by who wrote it. We also audit what is deployed: in all sixteen frameworks, the harness is the only writer. With one writer, readers can name the fault from the record, in up to nine runs in ten, but they cannot check the record. The model examiners claim corroboration 
    
[^91]: 谁在框架背后？通过智能体行为对大语言模型进行指纹识别

    Who Is Behind the Harness? Fingerprinting LLMs through Agentic Behavior

    [https://arxiv.org/abs/2609.28559](https://arxiv.org/abs/2609.28559)

    提出了一种名为LIDAR的主动式黑盒指纹识别方法，通过编码智能体在运行时的决策与行动（如编辑后验证、故障恢复和规范-测试冲突处理等行为）来识别框架背后的大语言模型身份。

    

    大语言模型越来越多地通过编码智能体框架运行，这些框架会检查代码仓库、调用工具并修改文件。因此，替换此类智能体背后的模型可能会改变与安全相关的决策，包括它是否会验证更改或从故障中安全恢复。现有的LLM指纹识别方法主要是从直接文本或token分布推断模型身份。而在编码智能体中，这些信号会受到系统指令、控制器逻辑、工具和执行反馈的调节，限制了它们的迁移能力。我们提出了LIDAR（基于运行时决策与行动的LLM识别），这是一种针对编码智能体执行的主动式黑盒指纹识别方法。三个编码探测对在受控变更下揭示编辑后验证、瞬时故障恢复以及规范与测试冲突的解决行为。LIDAR使用互补的实例级和分布级特征来表示所产生的轨迹，并将其与（原文摘要在此处截断）

    arXiv:2609.28559v1 Announce Type: cross  Abstract: LLMs increasingly operate through coding-agent harnesses that inspect repositories, invoke tools, and modify files. Substituting the model behind such an agent can therefore change security-relevant decisions, including whether it verifies changes or recovers safely from failures. Existing LLM fingerprints largely infer identity from direct text or token distributions. In coding agents, these signals are mediated by system instructions, controller logic, tools, and execution feedback, limiting their transfer.   We present LIDAR (LLM Identification from Decisions and Actions at Runtime), an active black-box fingerprinting method for coding-agent execution. Three coding probe pairs expose post-edit verification, transient-failure recovery, and specification--test conflict resolution under controlled changes. LIDAR represents the resulting trajectories with complementary instance-level and distribution-level features and compares them wit
    
[^92]: 生物医学知识构建：软件工程视角

    Biomedical Knowledge Composition: A Software Engineering Perspective

    [https://arxiv.org/abs/2608.08927](https://arxiv.org/abs/2608.08927)

    本文从软件工程视角统一阐述生物医学知识基础设施，向软件工程师介绍知识图谱及五大数据协调挑战，并指出软件工具采用不足是生物医学知识基础设施工程化困难的一个根本原因。

    

    生物医学研究已经积累了海量的分子、临床和人群数据，然而将这些丰富的数据转化为可操作的知识仍受到技术和组织方面困难的制约。本文对生物医学知识基础设施的两个视角进行了统一阐述。第一个视角是向软件工程师介绍生物医学领域：解释了为什么知识图谱是现代生物医学中核心的集成数据结构，刻画了五大数据协调挑战（标识符映射、实体消解、模式对齐、证据整合和溯源追踪），调查了从药物发现到数字孪生等应用领域，并介绍了六个具有不同设计选择的代表性知识图谱系统。第二个视角则探讨为什么生物医学知识基础设施的工程化构建仍然如此困难。我们认为，一个促成性的根本原因是软件工具的采用不足。

    arXiv:2608.08927v2 Announce Type: replace  Abstract: Biomedical research has accumulated vast molecular, clinical, and population data, yet translating this wealth into actionable knowledge remains constrained by technical and organizational difficulties. This article presents a unified treatment of two perspectives on biomedical knowledge infrastructure. The first introduces the biomedical domain to software engineers: it explains why knowledge graphs (KGs) are the central integrative data structure in modern biomedicine, characterizes five data harmonization challenges (identifier mapping, entity resolution, schema alignment, evidence integration, and provenance tracking), surveys application domains from drug discovery to digital twins, and profiles six representative KG systems with contrasting choices. The second perspective asks why engineering biomedical knowledge infrastructure remains so difficult. We argue that a contributing root cause is limited adoption of software tooling
    
[^93]: Scrouting：通过先侦察仓库实现成本感知的编码智能体路由

    Scrouting: Cost-Aware Routing of Coding Agents by Scouting the Repository First

    [https://arxiv.org/abs/2608.04804](https://arxiv.org/abs/2608.04804)

    SuperScout先用一个7B搜索器侦察仓库并生成经沙箱验证的结构化交接文档，再据此将编码任务路由至合适的前沿模型，在SWE-bench Pro上以约五分之一的成本达到与最佳单一模型相当的解决率，且新增修复模型无需重新训练。

    

    前沿语言模型能够解决仓库级别的软件问题，但每次尝试成本高昂，且现有的路由器仅根据问题文本来选择模型。我们提出了SuperScout，它在侦察仓库之后再进行路由：一个7B的搜索器SuperScout-7B首先探索仓库并生成结构化的交接文档，其中的复现声明经过沙箱验证，虚假声明在交付前会被剔除。随后，搜索器的隐藏状态与任务文本一起输入一个基于该交接文档（resume）的路由器，由其将任务分派给四个前沿修复模型之一。添加新的修复模型无需重新训练。在SWE-bench Pro的完整Python切片（266个任务）上，在基准测试官方设定的预算上限级别下，SuperScout以每次求解约五分之一总成本的代价，达到了与最佳单一模型相当的解决率（SuperScout为266个任务中的159个，最佳模型为158个），且所报告的配置优于随机流量分配基线。

    arXiv:2608.04804v2 Announce Type: replace-cross  Abstract: Frontier language models can resolve repository-level software issues, but each attempt is expensive, and existing routers select a model from the issue text alone. We present SuperScout, which routes after scouting the repository: a 7B searcher, SuperScout-7B, first explores the repository and produces a structured handoff whose reproduction claims are sandbox-verified, with false claims stripped before delivery. The searcher's hidden states, together with the task text, then feed a resume-based router that dispatches the task to one of four frontier fixers. Adding a new fixer requires no retraining. On the full Python slice of SWE-bench Pro (266 tasks) under the benchmark's official capped budget tier, SuperScout matches the best single model's solve rate (159 of 266 for SuperScout, 158 for the best model) at about a fifth of the total cost per solve, and the reported configuration sits above the random traffic-splitting base
    
[^94]: 在智能体软件开发中使用第三方API路由器的权衡在哪里？

    Where Is the Tradeoff in Using Third-Party API Routers for Agentic Software Development?

    [https://arxiv.org/abs/2607.23624](https://arxiv.org/abs/2607.23624)

    该论文首次实证研究了编码智能体中第三方API路由器的路由器端注入风险，通过四个隐蔽程度递增的干预级别（如响应替换）揭示其可对软件开发任务造成真实且难以检测的影响，而现有客户端权限机制难以有效防御。

    

    第三方API路由器已成为一种常见的中间层，统一了对日益多样化的LLM提供商的访问。在编码智能体工作流中，高自主性操作被广泛采用，因为它减少了交互开销。因此，位于智能体与上游提供商之间的第三方API路由器不可避免地占据了可信路径。它可以检查和修改每一个请求与响应，然而目前没有任何机制来验证提供商的输出与智能体最终执行的仓库级操作之间的一致性。因此，客户端权限机制在实践中可能变得无效。这种控制缺口是否会对软件开发任务产生真实的、难以检测的影响，此前尚未得到实证测量。在本文中，我们对编码智能体中的路由器端注入进行了实证研究，考察了四个隐蔽程度递增的干预级别：响应替换（L1）、响应追加……

    arXiv:2607.23624v3 Announce Type: replace-cross  Abstract: Third-party API routers have become a common layer that unifies access across increasingly diverse LLM providers. In coding-agent workflows, high-autonomy operation is widely adopted because it reduces interaction overhead. As a result, a third-party API router, which sits between the agent and the upstream provider, inevitably occupies the trusted path. It can inspect and modify every request and response, yet no mechanism verifies alignment between the provider's output and the repository-level actions ultimately executed by the agent. Consequently, client-side permission mechanisms may become ineffective in practice. Whether this control gap produces real, hard-to-detect effects on software development tasks remains empirically unmeasured. In this paper, we conduct an empirical study of router-side injection in coding agents, examining four intervention levels of increasing subtlety: Response Substitution (L1), Response Appe
    
[^95]: 先验知识还是搜索？LLM智能体在硬件感知代码优化中的研究

    Prior Knowledge or Search? A Study of LLM Agents in Hardware-Aware Code Optimization

    [https://arxiv.org/abs/2605.19782](https://arxiv.org/abs/2605.19782)

    该研究通过三个受控实验揭示，LLM智能体在硬件感知代码优化中依赖先验知识而非搜索——它在黑盒优化中表现为贪婪优化器，在零样本内核生成中会完全忽略输入尺寸信息，且面对不常见的内核尺寸时性能急剧下降。

    

    LLM发现与优化系统正被日益广泛地应用于各个领域，它们实现了一种通用的“提出-评估-修改”循环。这类优化或发现过程通过根据环境中收到的反馈对上下文进行条件化而向前推进。然而，随着现代LLM智能体结构日益复杂，很难评估哪些组件贡献最大，以及这种探索会在何时、以何种方式失败。我们通过三个受控实验回答了这些问题。我们的发现如下：(1) 在纯黑盒优化中，LLM表现得像贪婪优化器。(2) 在零样本内核生成中，提供显式的输入尺寸信息没有可测量的效果——无论尺寸或温度如何，模型都收敛到相同的内核参数，仿佛尺寸指令是不可见的。此外，当被要求对不常见的内核尺寸进行优化时，无论使用何种语言，性能都会急剧下降。(3

    arXiv:2605.19782v2 Announce Type: replace  Abstract: LLM discovery and optimization systems are increasingly applied across domains, implementing a common propose-evaluate-revise loop. Such optimization or discovery progresses via context conditioning on received feedback from an environment. However, as modern LLM agents are increasingly complex in their structure, it is difficult to evaluate which components contribute the most, and when and how this exploration may fail. We answer these questions through three controlled experiments. Our findings: (1) In pure black-box optimization, LLMs act as greedy optimizers. (2) In zero-shot kernel generation, providing explicit input-size information has no measurable effect, models converge to the same kernel parameters regardless of size or temperature, as though the size instruction were invisible. Moreover, when tasked to perform kernel optimization for uncommon kernel sizes, performance sharply degrades regardless of the language used. (3
    
[^96]: GitHub上提交签名的分析

    Analysis of Commit Signing on GitHub

    [https://arxiv.org/abs/2604.14014](https://arxiv.org/abs/2604.14014)

    该论文首次以GitHub账户为抽样单位，通过对270多万个账户、1600多万次提交和20多万个公钥的大规模端到端分析，系统揭示了开发者提交签名中签名密钥的使用与管理模式。

    

    自1999年以来，安全界就已经知道，对密码学系统的用户而言，管理密码学密钥十分困难。Git允许软件开发者使用自己控制的私钥，对记录代码更改的提交进行数字签名。这一做法为研究大型社区在日常工作中如何管理密码学密钥提供了难得的机会。在本文中，我们以GitHub账户为抽样单位，对提交签名进行了端到端的研究。我们对GitHub用户群体进行了2%的分层抽样，共包含2,737,649个账户，其中71,694个账户至少进行过一次提交。针对这些账户，我们分析了874,198个仓库中的16,112,439次提交，并收集了上传至抽样账户的208,546个公钥。我们考察了开发者签名密钥的使用模式，并将研究结果与以往提交签名研究进行比较——这些研究大多采用基于仓库的抽样方式，且较少关注密钥管理问题。

    arXiv:2604.14014v4 Announce Type: replace  Abstract: Since 1999, the security community has known that managing cryptographic keys is hard for users of cryptographic systems. Git lets software developers use private keys they control to digitally sign the commits that record their code changes to a repository. This practice offers a rare opportunity to study how a large community manages cryptographic keys in everyday work. In this paper, we study end-to-end commit signing, using GitHub accounts as our sampling unit. Our 2% stratified sample of the GitHub population comprises 2,737,649 accounts, 71,694 of which have made at least one commit. For these accounts, we analyze 16,112,439 commits across 874,198 repositories and collect 208,546 public keys uploaded to the sampled accounts. We examine developers' signing key usage patterns and compare our findings with those of prior commit signing studies, most of which used repository-based samples and paid less attention to key management. 
    

