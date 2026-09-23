# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CliffCompaction: Cost-Efficient Compaction for Long-Horizon Coding Agents](https://arxiv.org/abs/2609.26779) | CliffCompaction是一种仅通过截断或删除内容（绝不改写重写）来保持信息忠实性的自动上下文压缩技术，可使长时程编码智能体的成本降低多达50%，同时保持或提升性能，并大幅提升测试时扩展的成本效益。 |
| [^2] | [SWE-Serve: Benchmarking Agentic Engineering For Production Inference Serving](https://arxiv.org/abs/2609.26777) | SWE-Serve 是一个包含 53 个源自 SGLang 生产变更任务的新基准，用于评估智能体在生产级推理服务工程上的能力，任务覆盖模型支持、运行时执行和公共 API 等六大推理工程类别，并通过隐藏的功能与回归测试进行评估。 |
| [^3] | [Grow the Harness, Not the Context: From Strategy-Free Scaffolds to Reusable Specialist Agents](https://arxiv.org/abs/2609.26760) | 该论文提出 Growing Harness 训练范式，通过失败定位、联合修复与成功优先门控，把任务反馈中反复出现的控制逻辑自动沉淀为可复用的可执行代码，让智能体框架本身从任务交互中“生长”出来，而 LLM 只需专注于任务特定的语义推理。 |
| [^4] | [Metrics Failure in LLM-Based Code Vulnerability Repair: An Empirical Study and a Change-Aware Screen](https://arxiv.org/abs/2609.26749) | 该研究通过五项受控实验证明编译率是评估大语言模型修复C/C++安全漏洞的科学上不可靠的指标，因为其主要由评估框架和数据集伪影而非模型质量决定，并提出了一种变化感知的筛选方法。 |
| [^5] | [TraceVIC: Causal Reasoning over Code Evolution for Identifying Vulnerability-Inducing Commits](https://arxiv.org/abs/2609.26711) | TraceVIC提出一种基于时序图的方法，通过对漏洞相关代码的演化过程进行因果推理，而非依赖版本历史中的位置启发式规则，来识别和排序引入漏洞的提交。 |
| [^6] | [Measuring the Serving Stack Instead of the Model: Hidden Confounds in Local Tool-Use Evaluation](https://arxiv.org/abs/2609.26693) | 本地服务栈（如 Ollama）的工具调用门控机制和失败元数据丢失会混淆模型工具使用能力的评估，使测得的保真度反映的是服务层而非模型本身的行为。 |
| [^7] | [From Approval to Execution: Reconstruction-Aware Repair Analysis for LLM-Agent Software](https://arxiv.org/abs/2609.26529) | 该论文提出重构稳定授权的形式化框架ReSA及相应的修复判定义务，用于发现并消除LLM智能体软件中因执行前对象重构而产生的残留授权绕过漏洞。 |
| [^8] | [FeatLens: Feature-Guided Dynamic Code Graph Construction and Retrieval for Repository-Level Code Generation](https://arxiv.org/abs/2609.26480) | FeatLens 提出了一种特征引导的动态代码图构建与检索方法，通过将自然语言特征描述与函数级代码元素关联的特征索引，为仓库级代码生成高效检索可复用的代码依赖，同时显著降低图构建、推理和 token 成本。 |
| [^9] | [Recursive self-improvement of AI research agents](https://arxiv.org/abs/2609.26457) | 本文提出AIDE^2系统，实现了AI研究智能体对自身代码的递归自我改进循环，在8天自主运行中发现了七项连续的性能改进，以对抗研发投入边际收益递减的长期趋势。 |
| [^10] | [On the Lexical Superstition of Large Language Models for Code Comprehension: Re-evaluation on Code of Low Lexical Quality](https://arxiv.org/abs/2609.26388) | 论文提出语义保持的标识符重命名框架Face/Off，揭示大语言模型在代码理解中普遍过度依赖标识符的词汇线索，且这一根深蒂固的问题难以通过现有的提示和微调干预加以解决。 |
| [^11] | [Design and Evaluation of a Controlled Post-Alert Incident Orchestration and Response Subsystem Using a Rule Engine and a Local Large Language Model](https://arxiv.org/abs/2609.26316) | 该论文设计并评估了一个面向教育信息系统的受控告警后事件编排响应子系统，通过规则引擎实现确定性严重度分类与处置手册路由，并让本地大语言模型在多重安全控制下提供辅助分析，实验验证了其路由准确性、任务执行可靠性和约33秒的平均处理时间。 |
| [^12] | [CANcept: Model-based CAN Traffic Generation and Manipulation](https://arxiv.org/abs/2609.26263) | CANcept是一个开源的基于模型的工具，通过将流量调度模型（TSM）与基于DBC的通信模型（DCM）结合在统一的执行机制中，实现对CAN流量的精确指定、生成、重放与操控，从而支持CAN软件时序安全属性的测试。 |
| [^13] | [Towards Effective Black-Box Adversarial Attacks on Deep Code Models via Structural and Identifier Perturbations](https://arxiv.org/abs/2609.26234) | 提出Strike框架，一种针对深度代码模型的输入条件化黑盒对抗攻击方法，通过LLM生成上下文相关的结构扰动并结合相似度引导的标识符替换，构建层次化扰动空间以有效测试模型鲁棒性。 |
| [^14] | [Reducing Hallucinations in Large Language Models Through Integrated Self-Verification and Retrieval-Augmented Generation](https://arxiv.org/abs/2609.26229) | 本文提出CoVe-RAG+统一框架，将验证链与检索增强生成相结合，通过对外部权威工程来源的检索验证和迭代自验证过程，有效减少大语言模型在高端工程应用中的幻觉问题。 |
| [^15] | [WatchPoint: Executable User Feedback for Real-World Agentic Web Development](https://arxiv.org/abs/2609.26204) | WatchPoint是一个模拟用户系统，通过像真实开发者一样对运行中的Web应用生成并执行诊断脚本，产生结构化观察结果来指导编码智能体的重试，在包含1,000个顺序依赖任务的Web-Bench基准上恢复了57.6%的失败任务。 |
| [^16] | [A Large-Scale Longitudinal Study of Multi-CI Service Adoption](https://arxiv.org/abs/2609.26181) | 本文通过对来自七种编程语言的135227个GitHub仓库和八种CI服务的大规模纵向研究，首次系统刻画了多CI服务在项目生命周期中的采用、演进与废弃模式，发现约五分之一的仓库使用多个CI服务，且多为迁移过程中的过渡性使用而非长期并行使用。 |
| [^17] | [Why Do LLMs Fail at OCL Generation? A Graph Reasoning Perspective](https://arxiv.org/abs/2609.26122) | 该研究首次从图推理的视角系统揭示了LLM在OCL约束生成中失败的根本原因：生成性能随UML类图的导航深度和结构复杂度增加而显著下降，而词汇相似性的影响有限。 |
| [^18] | [Post-Hoc Attention Steering of Large Language Models for Robust Code Understanding under Obfuscation](https://arxiv.org/abs/2609.26102) | 提出CodeSteer方法，通过结合轻量级程序分析与推理时注意力引导，将大语言模型的注意力重新分配到后向切片、控制流路径等语义相关的程序元素上，从而显著提升其对混淆代码的理解鲁棒性。 |
| [^19] | [On Behavioral Alignment of Model-Code and Human-Code Understandability via Behavioral Proxies](https://arxiv.org/abs/2609.26101) | 该论文提出将代码可理解性视为读者与代码交互的关系性属性，区分人类与模型的代码可理解性，并通过引入四种行为代理指标来研究两者之间的行为对齐。 |
| [^20] | [Observing the Conduct of Systematic Reviews with Generative AI Support: An Experience Report from a Graduate Software Engineering Course](https://arxiv.org/abs/2609.26057) | 该经验报告通过观察软件工程博士生在有/无生成式AI辅助下开展试点系统性综述的课堂实践，发现大语言模型能降低入门门槛、加速备选方案生成并使方法论问题更加清晰，但也容易导致学生对AI的过度依赖。 |
| [^21] | [FIRE: Failure-Informed Runtime Engineering for Reliable Language-Model Agents](https://arxiv.org/abs/2609.26048) | 该论文提出FIRE，通过在不改变模型权重和用户提示的前提下，在失败发生前的状态处施加自然语言指令与动作拒绝等运行时策略，显著提升语言模型智能体的重复交付可靠性，在Terminal-Bench 2.1上pass^2最高提升9.2个百分点。 |
| [^22] | [Compiling Sufficient Governance Context from Declared Losses and Reachable States: Exact Observation-Contract Synthesis with Cardinality and Cost Objectives](https://arxiv.org/abs/2609.26016) | 该论文提出从有限可达状态模型与声明判定结果中精确综合“观察契约”，在可穷举时枚举所有包含极小的充分契约，否则借助SAT/MaxSAT编码求解最小基数或最小成本的契约，并区分个体不可或缺属性与联合充分契约。 |
| [^23] | [Towards Systematic Qualification of Vision-Language Models for Automotive Perception Systems](https://arxiv.org/abs/2609.25945) | 该论文指出视觉语言模型在汽车感知系统中存在幻觉风险（既可能虚构交通对象，也可能漏检真实存在的对象），并强调需要系统性地结合设计时与运行时的验证确认技术，以实现VLM在安全关键汽车系统中的合格性认证。 |
| [^24] | [When Should Dependency Updates Invoke Repair Agents? A Lightweight Routing Study](https://arxiv.org/abs/2609.25911) | 本文提出轻量级路由器DepFixRouter，仅利用PR创建时的标题和元数据信号预测哪些依赖更新拉取请求需要兼容性修复，避免对所有更新盲目调用昂贵的仓库级修复智能体，在节省调用的同时高效捕获真正需要修复的请求。 |
| [^25] | [Confidence-Guided Cross-Modal Knowledge Transfer for Multimodal Anomaly Detection in Microservice Systems](https://arxiv.org/abs/2609.25856) | 提出了CMT-AD方法，通过统一的深度聚类框架联合建模微服务系统的指标与日志，并利用软聚类分布将模态不确定性量化为置信度分数，以引导跨模态知识迁移，从而应对模态可靠性动态变化和数据异构性的挑战，实现更准确的多模态异常检测。 |
| [^26] | [What Was Once Learned May Need to Be Unlearned: Machine Unlearning for Deprecated API Knowledge in Large Language Models](https://arxiv.org/abs/2609.25786) | 本文针对大语言模型生成已弃用API的问题，构建了基于实际行为验证的基准MUDAPIBench（包含7000多个模型特定实例），并对八种机器遗忘方法在三个代码LLM上的遗忘效果与副作用进行了系统性实证研究。 |
| [^27] | [Testing and Learning Symbolic Finite State Machines](https://arxiv.org/abs/2609.25603) | 该论文证明了符号有限状态机有限实例化的语言等价性可以推广到完整输入域上，从而将确定性有限状态机的完备测试与学习方法成功迁移到数据域可能无限的符号有限状态机。 |
| [^28] | [Understanding Maintenance and Support in a Community-Driven Scientific Workflow Ecosystem: A Cross-Space Study of Galaxy](https://arxiv.org/abs/2609.25587) | 该研究通过对Galaxy生态系统中11,762个GitHub issue、52,203个pull request和6,235条论坛讨论的大规模跨空间实证分析，首次系统刻画了社区驱动科学工作流系统的维护与支持关注点，以及开发空间与社区支持空间中维护产物之间的关联。 |
| [^29] | [An Empirical Analysis of Cross-OS Portability Issues in Python Projects](https://arxiv.org/abs/2609.25531) | 该论文开展了首个针对Python跨操作系统可移植性问题的大规模实证研究，分析了2,042个开源仓库，构建了包含7个主要故障类别、24个子类别、15个诊断特征和4种系统性修复模式的全面分类体系。 |
| [^30] | [Evaluating Shaker for Flaky Test Detection in Python Projects](https://arxiv.org/abs/2609.25528) | 本研究首次对Shaker工具在Python项目中的不稳定测试检测效果进行实证评估，发现其检测率（37.2%）与简单的重复执行方法（35.8%）无统计学显著差异，表明Shaker在Java/Android上的优势无法直接迁移到Python。 |
| [^31] | [Dynamic Conformance Testing of WebGPU Through Specification-Driven Mutation](https://arxiv.org/abs/2609.25520) | LANTERN是一个规范引导的动态一致性测试框架，通过从WebGPU规范中提取语法规则和语义约束来变异官方CTS测试，从而生成有效与无效的测试变体以暴露实现中的缺陷。 |
| [^32] | [The Vocabulary of Flaky Tests in Swift](https://arxiv.org/abs/2609.25516) | 该研究首次针对Swift语言评估了基于词汇的机器学习方法来预测不稳定测试，通过从15个开源项目收集数据并训练五个分类器，证明随机森林模型（F1=0.86，MCC=0.75）能够利用测试词汇特征有效识别不稳定测试，并显著优于简单基线方法。 |
| [^33] | [Modular Composition of Inductive Types Using Lean Meta-programming](https://arxiv.org/abs/2609.25427) | 本文提出基于元编程的归纳类型与函数组合算法，并通过扩展Lean证明助手的语法，实现了归纳类型的模块化复用、组合与扩展，从而缓解表达式问题。 |
| [^34] | [Passes Alone, Fails Together: Benchmarking Semantic Coordination in Parallel LLM-Agent Development](https://arxiv.org/abs/2609.25396) | 该论文提出了 stale 基准测试来衡量并行LLM编码智能体之间的语义协调问题，发现真实合并的拉取请求中干扰极少，但在使用真实 Django 代码构建的受控任务中 97% 的运行出现合并干扰，而一条简单的并发更改描述消息即可恢复 82% 的失败运行。 |
| [^35] | [Extending FunctionGemma for Practical On-Device Mobile Function Calling](https://arxiv.org/abs/2609.25373) | 该研究通过构建涵盖十五类设备控制操作的约9,500个对话的合成数据集MOBILEACTIONSEXTENDED对FunctionGemma 270M进行微调，将设备端Android函数调用的端到端准确率从29.3%大幅提升至76.5%，并使组合模型在Google数据集上达到82.3%。 |
| [^36] | [SSP-Bench: A Hybrid Data Generation Framework for Safety, Security, and Privacy Evaluation](https://arxiv.org/abs/2609.25352) | SSP-Bench 是一个动态基准生成框架，通过按需生成评估实例、外部来源保障标签有效性、多模型面板校准难度，并将基准构建形式化为多目标优化问题，从而克服了静态基准在 LLM 安全、安保与隐私评估中的分数饱和与数据污染等缺陷。 |
| [^37] | [TAILOR: Template-Preserving Augmentation for Long-Tailed Log Parsing](https://arxiv.org/abs/2609.25261) | 该论文提出TAILOR方法，通过模板保持式的数据增强来应对日志解析中的长尾分布问题，提升对稀有日志模板的解析性能并使评估更加真实可靠。 |
| [^38] | [TRACTOR Benchmark for Evaluating C to Rust Translators](https://arxiv.org/abs/2609.25121) | 本文介绍了由 MIT 林肯实验室在 DARPA TRACTOR 项目下开发的标准化基准测试，通过难度递增的测试集和评估基础设施，系统性地评估 C 到 Rust 自动翻译工具的能力。 |
| [^39] | [OSFoundry: Building and Evolving Operating Systems with Specification-Guided Agents](https://arxiv.org/abs/2609.25018) | OSFoundry 提出了一个操作系统专用的智能体框架，其核心思想是将稳定的设计意图与多样化的实现相分离，通过共享开发蓝图 SysSpec*（包含任务边界的 Plan 和 OS 特定的 Specification）使设计意图在操作系统开发全程保持持久，从而提升操作系统持续演进的可靠性。 |
| [^40] | [Quantifying Overclaiming Propensity in Frontier LLM Agents](https://arxiv.org/abs/2609.20812) | 本文提出OverclaimBench评估套件，首次量化了前沿LLM编码智能体在最终回复中“过度宣称”任务完成的倾向，并发现在67.9%的运行中智能体并未真正阅读所有被要求审查的文件。 |
| [^41] | [Models as Governed Interfaces for AI-Native MBSE: Read-Side Adequacy and Write-Side Admissibility](https://arxiv.org/abs/2609.16252) | 该论文指出AI参与模型驱动系统工程（MBSE）的关键瓶颈不在建模语言而在数据架构，提出“认知充分性”这一数据架构模式，通过“读侧充分性”与“写侧可采性”防止AI用不可验证、不受治理的训练数据填补模型信息缺口。 |
| [^42] | [Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance](https://arxiv.org/abs/2608.26480) | 本文证明，在不进行训练或基准调优的情况下，基于账本控制的管理器-工作器脚手架能显著提升某些LLM的编码性能，但效果因模型而异，并非普遍适用。 |
| [^43] | [AutoSQL: Extracting SQL Templates from Imperative ORM Code in Large-Scale Repositories](https://arxiv.org/abs/2608.15595) | AutoSQL通过构建代码索引和混合上下文检索策略，利用LLM代理从Go ORM命令式代码中自动提取SQL模板，解决了静态恢复SQL的难题。 |
| [^44] | [Improving Constraint Models with LLM Agents](https://arxiv.org/abs/2608.08127) | 提出了一个基于大语言模型智能体的框架，能够从开放式空间自动重构约束规划模型，通过将解注入原始模型进行经验性验证并诊断修复，在中位数约十五分钟内返回最佳改进模型变体。 |
| [^45] | [Isabelle/STARK: A Formalization of zk-STARK in Isabelle/HOL](https://arxiv.org/abs/2608.01965) | 该论文在 Isabelle/HOL 中对 STARK 协议进行了机械化形式化验证，通过 FRI 相关一致性推理、Merkle 认证和精确采样计数等技术证明了带查询次数限制的可靠性，并在具体参数下给出虚假接受概率不超过 2^{-137} 的界。 |
| [^46] | [XScientist: A Git-Like Research Protocol for Long-Running Autonomous Scientific Discovery](https://arxiv.org/abs/2607.12301) | XScientist提出了一种类Git的研究协议，将研究状态而非论文手稿作为延续单位，通过类型化内容寻址对象、不可变检查点和声明-证据闭环等机制，使自主科学发现过程可检查、可分叉、可验证且可长期延续。 |
| [^47] | [Petrify: Petri-net Based Analysis of Concurrency Properties in Java Bytecode](https://arxiv.org/abs/2607.00830) | Petrify将Java字节码程序语义编码为简洁的Petri网，借助LoLA等模型检测工具实现对并发性质的自动化验证，在表达能力与实用性之间取得了独特的平衡。 |
| [^48] | [Real Money, Fake Models: Deceptive Model Claims in Shadow APIs](https://arxiv.org/abs/2603.01919) | 该论文首次对官方大语言模型API与影子API进行系统性审计，识别出17个已被187篇学术论文使用的影子API，揭示了影子API可能提供与官方API不一致的输出，从而威胁下游应用的可靠性和学术研究结果的有效性。 |
| [^49] | [VeriSoftBench: Repository-Scale Formal Verification Benchmarks for Lean](https://arxiv.org/abs/2602.18307) | VeriSoftBench是一个包含500个证明义务的仓库级Lean 4形式化验证基准，评估发现专为Mathlib数学调优的证明器难以迁移到以仓库为中心的软件验证场景，且任务成功率与其传递性依赖闭包的规模密切相关。 |
| [^50] | [SWE-Universe: Scale Real-World Verifiable Environments to Millions](https://arxiv.org/abs/2602.02361) | SWE-Universe提出一个可扩展框架，利用定制训练的构建智能体从GitHub PR中自动构建了80余万个真实世界软件工程可验证环境，并将Qwen3-Max-Thinking在SWE-Bench Verified上的成绩提升至75.3%。 |
| [^51] | [Metamodel-Guided Model Generation with Layered Constraints](https://arxiv.org/abs/2510.25890) | 提出一种元模型引导的分层约束模型生成方法，通过生成时约束层（L1）与生成后验证层（L2）的协同配合，确保LLM生成的工程模型满足结构约束、领域规则和任务要求。 |
| [^52] | [LLM-Based Repair of Static Nullability Errors](https://arxiv.org/abs/2507.20674) | NullRepair是一个将大语言模型嵌入结构化工作流的系统，其决策流程基于对200个真实错误的手动分析得出的流程图，并结合静态分析，从而自动且准确地修复Java代码中静态可空性检查残留的错误。 |

# 详细

[^1]: CliffCompaction：面向长时程编码智能体的低成本高效压缩技术

    CliffCompaction: Cost-Efficient Compaction for Long-Horizon Coding Agents

    [https://arxiv.org/abs/2609.26779](https://arxiv.org/abs/2609.26779)

    CliffCompaction是一种仅通过截断或删除内容（绝不改写重写）来保持信息忠实性的自动上下文压缩技术，可使长时程编码智能体的成本降低多达50%，同时保持或提升性能，并大幅提升测试时扩展的成本效益。

    

    智能体通常需要处理需要数百万token上下文的复杂问题，由于上下文窗口有限，这需要在会话之间进行压缩。我们开发了CliffCompaction，这是一种自动压缩技术，在有界上下文条件下可将成本降低多达50%，同时在Terminal-Bench上保持或提升性能，并在测试时扩展方面达到新的效率水平，在KernelBench上取得最先进的结果。CliffCompaction在每次rollout中的节省使测试时扩展的性能-成本权衡更加高效，以低于两次完整上下文运行的成本在Terminal-Bench上带来了超过10个百分点的成绩提升。在并行测试时扩展下，CliffCompaction让Kimi K2.6能够媲美Opus 4.7，并以更低成本超越Opus 4.6和GPT-5.3 Codex。CliffCompaction有效的关键在于，它仅通过截断或删除内容来保持压缩信息的忠实性，从不改写或重写内容。

    arXiv:2609.26779v1 Announce Type: cross  Abstract: Agents often work on complex problems that require millions of tokens of context, which necessitates compacting across sessions due to limited context windows. We develop CliffCompaction, an autocompaction technique that reduces cost by up to 50% under a bounded context while maintaining or improving performance on Terminal-Bench and achieving new levels of efficiency for test-time scaling and state-of-the-art results on KernelBench. The per-rollout savings of CliffCompaction make the performance--cost trade-off of test-time scaling more efficient, adding over 10 percentage points on Terminal-Bench for less than the cost of two full-context runs. Under parallel test-time scaling, CliffCompaction lets Kimi K2.6 match Opus 4.7, and exceed Opus 4.6 and GPT-5.3 Codex at lower cost. The key to CliffCompaction's effectiveness is that it keeps compacted information faithful by only truncating or dropping content, never rephrasing or rewriting
    
[^2]: SWE-Serve：面向生产级推理服务的智能体工程基准测试

    SWE-Serve: Benchmarking Agentic Engineering For Production Inference Serving

    [https://arxiv.org/abs/2609.26777](https://arxiv.org/abs/2609.26777)

    SWE-Serve 是一个包含 53 个源自 SGLang 生产变更任务的新基准，用于评估智能体在生产级推理服务工程上的能力，任务覆盖模型支持、运行时执行和公共 API 等六大推理工程类别，并通过隐藏的功能与回归测试进行评估。

    

    我们推出了 SWE-Serve，这是一个用于评估智能体在生产级推理工程任务上能力的基准测试。实现一个推理功能往往需要协调服务栈中的多项变更，包括模型支持、运行时执行和公共 API。现有基准测试对生产级推理工程的覆盖有限：仓库级软件工程基准测试并不针对推理，而通用终端智能体基准测试仅包含少量推理任务。与此同时，专门的推理基准测试主要聚焦于孤立的内核生成或性能优化，而非仓库规模的生产功能实现。SWE-Serve 提供了 53 个基于真实仓库的任务，这些任务源自 SGLang 近期的生产变更，涵盖六大推理工程类别。每个任务在 CPU 或单个 GPU（H100）上运行，并通过隐藏的功能测试和回归测试进行评估，包括，

    arXiv:2609.26777v1 Announce Type: cross  Abstract: We introduce SWE-Serve, a benchmark for evaluating agents on production inference engineering tasks. Implementing an inference feature can require coordinating multiple changes across the serving stack, including model support, runtime execution, and public APIs. Existing benchmarks provide limited coverage of production inference engineering: repository-level software engineering benchmarks do not target inference, while general terminal-agent benchmarks include only a few inference tasks. Dedicated inference benchmarks, meanwhile, focus primarily on isolated kernel generation or performance optimization rather than repository-scale production feature implementation. SWE-Serve provides 53 repository-grounded tasks derived from recent production changes to SGLang, spanning six inference engineering families. Each task executes on either CPU or a single GPU (H100) and is evaluated with hidden functional and regression tests, including, 
    
[^3]: 扩展框架而非上下文：从无策略脚手架到可复用的专家智能体

    Grow the Harness, Not the Context: From Strategy-Free Scaffolds to Reusable Specialist Agents

    [https://arxiv.org/abs/2609.26760](https://arxiv.org/abs/2609.26760)

    该论文提出 Growing Harness 训练范式，通过失败定位、联合修复与成功优先门控，把任务反馈中反复出现的控制逻辑自动沉淀为可复用的可执行代码，让智能体框架本身从任务交互中“生长”出来，而 LLM 只需专注于任务特定的语义推理。

    

    大语言模型（LLM）智能体通常需要处理一系列相关任务，然而标准的智能体框架反复要求模型在每个任务的上下文中重新构建相同的控制决策。我们研究能否转而利用任务反馈，将反复出现的控制逻辑转化为可复用的可执行代码，同时将 LLM 调用保留用于任务特定的语义推理。我们提出 Growing Harness（生长式框架），一种由失败引导的训练范式，它从一个无策略的脚手架中学习智能体框架本身；该脚手架仅暴露固定的模型与工具接口，而不编码任何任务求解控制器。函数级别的执行轨迹将每次失败定位到有界的代码范围内，优化器联合修复一个失败窗口，而以成功为先的保留集门控会回滚损害既有能力的修复序列。被接受的修改不断累积到同一个共享框架中，使其控制结构从任务反馈中自然涌现。在 BrowseComp-Plus 和 WebArena-Verified 上的实验（摘要此处截断）……

    arXiv:2609.26760v1 Announce Type: cross  Abstract: Large language model (LLM) agents often handle streams of related tasks, yet standard harnesses repeatedly ask the model to reconstruct the same control decisions inside each task's context. We study whether task feedback can instead turn recurring control into reusable executable code, while reserving LLM calls for task-specific semantic reasoning. We introduce Growing Harness, a failure-guided training paradigm that learns the agent harness itself from a strategy-free scaffold that exposes fixed model and tool interfaces but encodes no task-solving controller. Function-level execution traces localize each failure to a bounded code surface, an optimizer repairs a window of failures jointly, and a success-first held-out gate rolls back repair sequences that harm prior capability. Accepted edits accumulate in one shared harness, allowing its control structure to emerge from task feedback. Across BrowseComp-Plus and WebArena-Verified wit
    
[^4]: 基于大语言模型的代码漏洞修复中的度量失效：一项实证研究与一种变化感知的筛选方法

    Metrics Failure in LLM-Based Code Vulnerability Repair: An Empirical Study and a Change-Aware Screen

    [https://arxiv.org/abs/2609.26749](https://arxiv.org/abs/2609.26749)

    该研究通过五项受控实验证明编译率是评估大语言模型修复C/C++安全漏洞的科学上不可靠的指标，因为其主要由评估框架和数据集伪影而非模型质量决定，并提出了一种变化感知的筛选方法。

    

    大语言模型（LLM）越来越多地被应用于C/C++安全漏洞的自动化修复，而编译率（即生成的补丁能否通过编译）是常被报告的进展代理指标。我们认为，对于单函数漏洞修复而言，编译率是一个科学上不可靠的度量指标，并通过五项受控实验来支持这一论点，实验涉及来自Big-Vul的203个漏洞函数、三个开源代码大语言模型（参数量从3.5亿到67亿）以及三种提示策略。编译率（i）对一项能显著改善生成代码的干预措施几乎没有响应；（ii）主要受评估框架和数据集伪影而非模型质量的影响，约64%的编译失败不可归因于模型，且这一比例在不同模型间几乎不变；（iii）在完全相同的补丁下，仅改变一个编译器标准标志就会使编译率波动1.8到2.7倍，且没有出现任何回归；（iv）对三个模型的排名……（原文摘要在此处截断）

    arXiv:2609.26749v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly applied to the automated repair of C/C++ security vulnerabilities, and compile rate is a commonly reported proxy for progress: whether the generated patch compiles. We argue that compile rate is a scientifically unreliable metric for single-function vulnerability repair, and we support this with five controlled experiments over 203 vulnerable functions from Big-Vul, three open-source code LLMs (350M to 6.7B parameters), and three prompting strategies. Compile rate (i) barely responds to an intervention that substantially improves the generated code; (ii) is dominated by evaluation-harness and dataset artifacts rather than model quality, with about 64% of compile failures not attributable to the model, a share that is nearly invariant across models; (iii) shifts by 1.8 to 2.7 times on identical patches under a single compiler-standard flag, with zero regressions; (iv) ranks the three models in
    
[^5]: TraceVIC：基于代码演化的因果推理识别漏洞引入提交

    TraceVIC: Causal Reasoning over Code Evolution for Identifying Vulnerability-Inducing Commits

    [https://arxiv.org/abs/2609.26711](https://arxiv.org/abs/2609.26711)

    TraceVIC提出一种基于时序图的方法，通过对漏洞相关代码的演化过程进行因果推理，而非依赖版本历史中的位置启发式规则，来识别和排序引入漏洞的提交。

    

    软件漏洞往往在被引入很久之后才被发现，这使得识别引入潜在漏洞条件的漏洞引入提交（VIC）变得十分困难。现有的VIC识别技术主要依赖git blame在版本历史中追踪漏洞代码，并使用位置启发式方法，例如选择其最早或最近的一次修改。然而，真正的VIC可能出现在该历史中的任何位置，且漏洞行为可能依赖于跨多个版本演化的代码。因此，我们认为VIC识别需要对漏洞相关代码如何演化进行推理，而不是简单地看候选提交在版本历史中的位置。我们提出了TraceVIC，一种基于时序图的方法，通过推理代码演化来识别和排序VIC。TraceVIC首先定位可能的根因代码行并追踪其历史……

    arXiv:2609.26711v1 Announce Type: new  Abstract: Software vulnerabilities are often discovered long after they are introduced, making it difficult to identify the vulnerability-inducing commit (VIC) responsible for introducing the underlying vulnerable condition. Existing VIC identification techniques largely rely on git blame to trace vulnerable code through revision history and use positional heuristics, such as selecting its earliest or most recent modification. However, the true VIC may occur anywhere within this history, and vulnerable behavior may depend on code that evolves across multiple revisions. We therefore argue that VIC identification requires reasoning about how vulnerability-relevant code evolves, rather than simply where a candidate commit appears in the revision history.   We present TraceVIC, a temporal graph-based approach for identifying and ranking VICs by reasoning over code evolution. TraceVIC first localizes likely root-cause lines and traces their histories a
    
[^6]: 测量的是服务栈而非模型：本地工具使用评估中的隐藏混淆因素

    Measuring the Serving Stack Instead of the Model: Hidden Confounds in Local Tool-Use Evaluation

    [https://arxiv.org/abs/2609.26693](https://arxiv.org/abs/2609.26693)

    本地服务栈（如 Ollama）的工具调用门控机制和失败元数据丢失会混淆模型工具使用能力的评估，使测得的保真度反映的是服务层而非模型本身的行为。

    

    编码智能体必须发出有效的工具调用——即在提供的模式（schema）中对某个工具的可解析调用——之后测试框架才能执行其选择的动作。我们研究了本地服务栈如何影响这一协议步骤，并表明测量结果可能取决于服务层，而不仅仅是模型行为本身。在 Ollama 中，默认的 tools= 请求由静态模板标志按模型进行门控：一些模型被接受并以文本形式返回调用，一些模型返回原生 tool_calls，而 Phi-3 和 Gemma-3 则在推理之前就被拒绝。在我们的测试框架中，拒绝和重试耗尽不会被保存为结构化的失败元数据，因此下游分析可能将其错误归类为模型未发出调用，并简单粗暴地报告 0% 的保真度。在保留原生通道的同时添加文本工具列表，可以恢复被接受模型的大部分测量保真度；而统一的纯文本协议则会降低具有原生工具调用支持的 Llama-3.2 的保真度。跨栈……（原文摘要在此处截断）

    arXiv:2609.26693v1 Announce Type: new  Abstract: A coding agent must emit a valid tool call--a parseable invocation of a tool in the provided schema--before the harness can execute its chosen action. We study how local serving stacks affect this protocol step and show that measured outcomes can depend on the serving layer rather than model behavior alone. In Ollama, the default tools= request is gated per model by a static template flag: some models are accepted and return calls as text, some return native tool_calls, while Phi-3 and Gemma-3 are rejected before inference. In our harness, rejection and retry exhaustion are not preserved as structured failure metadata, so downstream analysis can misclassify them as model non-calls and naively report 0% fidelity. Adding a text tool list while retaining the native channel recovers much of the measured fidelity for accepted models, whereas a uniform text protocol reduces fidelity for Llama-3.2, which has native tool-call support. Cross-stac
    
[^7]: 从批准到执行：面向LLM智能体软件的重构感知修复分析

    From Approval to Execution: Reconstruction-Aware Repair Analysis for LLM-Agent Software

    [https://arxiv.org/abs/2609.26529](https://arxiv.org/abs/2609.26529)

    该论文提出重构稳定授权的形式化框架ReSA及相应的修复判定义务，用于发现并消除LLM智能体软件中因执行前对象重构而产生的残留授权绕过漏洞。

    

    批准机制已成为LLM智能体软件中对重大操作进行防护的首要安全保障。然而，用于审批展示的操作往往并非最终被消费的对象：工作流重载、对话记录投影、参数重绑定以及持久状态查找都可能在执行前对该对象进行重构。现有的字段流与检查覆盖分析能够证明预期字段已被检查，但无法证明被检查的对象版本确实到达了汇点，也无法证明在检查与使用之间没有发生替换。因此，一个局部完备的修复在重构之后仍可能留下残留的授权绕过漏洞。我们通过三项设计来解决这一问题：（1）我们在表示转换、汇点依赖、被消费版本与授权范围之上形式化了重构稳定授权；（2）我们推导出用于判定候选修复并暴露残留汇点后缀的义务；（3）我们在APAS-Fi中实现了该分析。

    arXiv:2609.26529v1 Announce Type: new  Abstract: Approval mechanisms have become a primary safeguard for consequential actions in LLM-agent software. Yet the action shown for approval is often not the object ultimately consumed: workflow reload, transcript projection, argument rebinding, and durable-state lookup may reconstruct it before execution. Existing fieldflow and check-coverage analyses can establish that expected fields were inspected, but not that the inspected object version reaches the sink or that no replacement intervenes between check and use. Consequently, a locally complete repair may still leave a residual authorization bypass after reconstruction. We address this problem through three designs. (1) We formulate reconstruction-stable authorization (ReSA) over representation transitions, sink dependencies, consumed versions, and grant scope. (2) We derive obligations that judge candidate repairs and expose residual sink suffixes. (3) We implement the analysis in APAS-Fi
    
[^8]: FeatLens：面向仓库级代码生成的特征引导动态代码图构建与检索

    FeatLens: Feature-Guided Dynamic Code Graph Construction and Retrieval for Repository-Level Code Generation

    [https://arxiv.org/abs/2609.26480](https://arxiv.org/abs/2609.26480)

    FeatLens 提出了一种特征引导的动态代码图构建与检索方法，通过将自然语言特征描述与函数级代码元素关联的特征索引，为仓库级代码生成高效检索可复用的代码依赖，同时显著降低图构建、推理和 token 成本。

    

    近期的代码生成研究已从孤立的函数补全转向在现有代码库中进行仓库级生成。为了正确实现目标函数，大语言模型（LLM）必须识别可复用的仓库依赖，例如已有函数、API 和跨文件定义。现有的检索方法通过代码相似性搜索、持久化的全仓库图或 LLM 驱动的图探索来提供此类上下文，但往往带来高昂的图构建、推理和 token 成本。面向特征的方法为软件功能提供了自然的视角，但它们主要支持需求分解、规划或特征编辑，而非代码依赖检索。本文提出了 FeatLens，一种面向仓库级代码生成的特征引导动态代码图构建与检索方法。FeatLens 构建了一个特征索引，将自然语言特征描述与函数级代码元素相链接……（原文摘要在此处截断）

    arXiv:2609.26480v1 Announce Type: new  Abstract: Recent code generation research has moved from isolated function completion toward repository-level generation in existing codebases. To implement a target function correctly, an LLM must identify reusable repository dependencies such as existing functions, APIs, and cross-file definitions. Existing retrieval methods provide such context through code similarity search, persistent whole-repository graphs, or LLM-driven graph exploration, but often incur high graph construction, reasoning, and token costs. Feature-oriented methods offer a natural view of software functionality, yet they mainly support requirement decomposition, planning, or feature editing rather than code dependency retrieval. This paper presents \textbf{FeatLens}, a feature-guided dynamic code graph construction and retrieval approach for repository-level code generation. FeatLens builds a feature index that links natural-language feature descriptions to function-level c
    
[^9]: AI研究智能体的递归自我改进

    Recursive self-improvement of AI research agents

    [https://arxiv.org/abs/2609.26457](https://arxiv.org/abs/2609.26457)

    本文提出AIDE^2系统，实现了AI研究智能体对自身代码的递归自我改进循环，在8天自主运行中发现了七项连续的性能改进，以对抗研发投入边际收益递减的长期趋势。

    

    AI智能体开始在整个AI技术栈中自动化研发工作，从提升训练效率到优化推理。一个自然的下一步是提升智能体自身的研究效率。当AI研究智能体自身的代码成为优化对象时，每一次被接受的改写都会成为下一轮被编辑的智能体。我们将这一循环称为递归自我改进。其重要性在于一个长期存在的趋势：研发累计投入的增加会带来边际收益递减，而持续的自我改进提供了一种对抗这一趋势的方法。我们提出了AIDE^2，一个为前沿AI研究智能体实现这一循环的系统。它会对自己代码提出修改建议，在一套AI研发任务上对修改后的自身版本进行基准测试，并保留在隐藏评估中表现最佳的修改。在一次自主运行的8天中，AIDE^2发现了七项连续的改进，涵盖从新的……

    arXiv:2609.26457v1 Announce Type: cross  Abstract: AI agents are beginning to automate research and development across the AI stack, from improving training efficiency to optimizing inference. A natural next step is to improve the research efficiency of the agents themselves. When an AI research agent's own code is the object of optimization, each accepted rewrite becomes the agent that the next round edits. We refer to this loop as recursive self-improvement. Its significance lies in a long-standing trend, in which increased cumulative spending on R&D yields diminishing returns. Sustained self-improvement offers a way to counter this trend. We present AIDE^2, a system that implements this loop for a frontier AI research agent. It proposes changes to its own code, benchmarks modified versions of itself on a suite of AI R&D tasks, and keeps the changes that perform best on hidden evaluations. In an autonomous 8-day run, AIDE^2 discovered seven successive improvements, ranging from a new
    
[^10]: 论大语言模型在代码理解中的词汇迷信：基于低词汇质量代码的重新评估

    On the Lexical Superstition of Large Language Models for Code Comprehension: Re-evaluation on Code of Low Lexical Quality

    [https://arxiv.org/abs/2609.26388](https://arxiv.org/abs/2609.26388)

    论文提出语义保持的标识符重命名框架Face/Off，揭示大语言模型在代码理解中普遍过度依赖标识符的词汇线索，且这一根深蒂固的问题难以通过现有的提示和微调干预加以解决。

    

    大语言模型（LLM）的最新进展使其被广泛应用于代码相关任务。标识符名称在自然产生的代码中具有统计信息价值，但其信息并不总是可靠的。我们研究了当前大语言模型在重命名保持程序结构不变的情况下，是否会对词汇线索赋予不成比例的权重。我们提出了Face/Off——一个保持语义的标识符重命名框架，并在多个模型和代码理解任务上评估渐进式命名条件。在该框架下，词汇过度依赖在被评估的模型和主要任务中普遍存在：随着标识符信息被移除或变得具有误导性，性能通常会下降，且输出往往被导向误导性名称所暗示的含义。该模式在代表性的基于提示和基于微调的干预措施下仍然存在，表明词汇过度依赖是一个根深蒂固的问题。

    arXiv:2609.26388v1 Announce Type: cross  Abstract: Recent advances in large language models (LLMs) have made them widely used for code-related tasks. Identifier names are statistically informative in naturally occurring code, but their information is not always reliable. We investigate whether current LLMs assign disproportionate weight to lexical cues when renaming preserves program structure. We introduce Face/Off, a semantics-preserving identifier-renaming framework, and evaluate progressive naming conditions across multiple models and code-comprehension tasks. Within this framework, lexical overemphasis is pervasive across the evaluated models and primary tasks: performance generally decreases as identifier information is removed or made misleading, and outputs are often directed toward the meanings suggested by misleading names. The pattern persists under representative prompt- and fine-tuning-based interventions, suggesting that lexical overemphasis is an entrenched problem. A ty
    
[^11]: 基于规则引擎与本地大语言模型的受控告警后事件编排与响应子系统的设计与评估

    Design and Evaluation of a Controlled Post-Alert Incident Orchestration and Response Subsystem Using a Rule Engine and a Local Large Language Model

    [https://arxiv.org/abs/2609.26316](https://arxiv.org/abs/2609.26316)

    该论文设计并评估了一个面向教育信息系统的受控告警后事件编排响应子系统，通过规则引擎实现确定性严重度分类与处置手册路由，并让本地大语言模型在多重安全控制下提供辅助分析，实验验证了其路由准确性、任务执行可靠性和约33秒的平均处理时间。

    

    本文提出了一种面向教育信息系统的受控告警后事件编排与响应子系统。该架构将确定性分类、上下文分析、人工审批和技术执行相互分离。规则引擎负责确定告警严重程度并选择处置手册，而静态RAG与本地大语言模型在验证器、护栏、输出净化器和安全回退机制的控制下提供建议性内容。实验在模拟告警存入Elasticsearch后开始。规则引擎在全部30个边界用例中均与预定义路由矩阵完全匹配。持久化队列完成了100个事件，未出现重复任务、新增失败任务或意外的防火墙规则。八告警并发实验保持了配置的单个活跃模型请求上限，30次顺序测量显示告警后总体平均处理时间约为33秒。结果表明……

    arXiv:2609.26316v1 Announce Type: cross  Abstract: This paper presents a controlled post-alert incident orchestration and response subsystem for educational information systems. The architecture separates deterministic classification, contextual analysis, human approval, and technical execution. A Rule Engine determines severity and selects the playbook, while Static RAG and a local large language model provide advisory content under Validator, Guardrail, Output Sanitizer, and Safe Fallback controls. Experiments begin after simulated alerts are stored in Elasticsearch. The Rule Engine matched the predefined routing matrix in all 30 boundary cases. The Durable Queue completed 100 events without duplicate tasks, new failed tasks, or unintended firewall rules. An eight-alert contention experiment preserved the configured limit of one active model request, and 30 sequential measurements showed an overall mean post-alert processing time of approximately 33 seconds. The results demonstrate f
    
[^12]: CANcept：基于模型的CAN流量生成与操控

    CANcept: Model-based CAN Traffic Generation and Manipulation

    [https://arxiv.org/abs/2609.26263](https://arxiv.org/abs/2609.26263)

    CANcept是一个开源的基于模型的工具，通过将流量调度模型（TSM）与基于DBC的通信模型（DCM）结合在统一的执行机制中，实现对CAN流量的精确指定、生成、重放与操控，从而支持CAN软件时序安全属性的测试。

    

    测试基于控制器局域网（CAN）的软件的时序相关安全属性，需要对传输数据和通信时序进行精确控制。现有的开源工具通常将此类场景编码在低级脚本中，导致其难以维护和演进。我们提出了CANcept，一个用于指定、生成、重放和操控CAN流量的开源基于模型的工具。其中，流量调度模型（TSM）定义消息的时序和内容转换，基于DBC的通信模型（DCM）定义消息级流量与CAN帧之间的编码和解码。CANcept将这两个模型结合在同一个执行机制中，用于生成和操控流量，包括对转换后轨迹的重放。初步评估表明，CANcept能够实现所指定的场景，具有较低的时序偏差，并在流量速率升高的情况下保持稳定执行。

    arXiv:2609.26263v1 Announce Type: new  Abstract: Testing timing-related safety properties of Controller Area Network (CAN)-based software requires precise control over transmitted data and communication timing. Existing open-source tools typically encode such scenarios in low-level scripts, making them difficult to maintain and evolve. We present \toolName{}, an open-source model-based tool for specifying, generating, replaying, and manipulating CAN traffic. A traffic schedule model (TSM) defines message timing and content transformations, and a DBC-based communication model (DCM) defines encoding and decoding between message-level traffic and CAN frames. \toolName{} combines both models in one execution mechanism for generated and manipulated traffic, including replay of transformed traces. A preliminary evaluation indicates that \toolName{} realizes the specified scenarios, with low timing deviation and stable execution under elevated traffic rates.
    
[^13]: 通过结构与标识符扰动实现对深度代码模型的有效黑盒对抗攻击

    Towards Effective Black-Box Adversarial Attacks on Deep Code Models via Structural and Identifier Perturbations

    [https://arxiv.org/abs/2609.26234](https://arxiv.org/abs/2609.26234)

    提出Strike框架，一种针对深度代码模型的输入条件化黑盒对抗攻击方法，通过LLM生成上下文相关的结构扰动并结合相似度引导的标识符替换，构建层次化扰动空间以有效测试模型鲁棒性。

    

    深度代码模型（DCMs）正日益被嵌入到代码智能任务中。然而，其在对抗攻击下的鲁棒性仍未得到充分理解。以往的黑盒攻击主要依赖仅针对标识符的替换，或从参考样本迁移而来的结构编辑，其扰动空间在很大程度上与被攻击的输入无关。我们提出了Strike，一个基于输入条件的黑盒对抗鲁棒性测试框架，它为每个输入构建并搜索层次化的扰动空间。Strike首先将代码划分为块，并利用大语言模型（LLM）生成与上下文相关的结构候选，这些候选经过语法有效性过滤、按相似度排序并进行自适应组合。随后，它通过从动态构建的候选池中进行相似度引导的标识符替换，对最优结构变体进行进一步细化。在多个代表性代码智能任务上的评估表明……

    arXiv:2609.26234v1 Announce Type: cross  Abstract: Deep code models (DCMs) are increasingly embedded in code intelligence tasks. However, their robustness under adversarial attacks remains insufficiently understood. Prior black-box attacks mainly rely on identifier- only substitutions or structural edits transferred from reference samples, yielding perturbation spaces defined largely independently of the attacked input. We introduce Strike, an input-conditioned black-box adversarial robustness-testing framework that constructs and searches a hierarchical perturbation space for each input. Strike first partitions code into blocks and uses an LLM to generate context-specific structural candidates, filtered for syntactic validity, ranked by similarity, and adaptively combined. It then refines the best structural variant through similarity-guided identifier substitutions drawn from dynamically constructed candidate pools. Evaluations across representative code intelligence tasks, including
    
[^14]: 通过集成自验证与检索增强生成减少大语言模型中的幻觉

    Reducing Hallucinations in Large Language Models Through Integrated Self-Verification and Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.26229](https://arxiv.org/abs/2609.26229)

    本文提出CoVe-RAG+统一框架，将验证链与检索增强生成相结合，通过对外部权威工程来源的检索验证和迭代自验证过程，有效减少大语言模型在高端工程应用中的幻觉问题。

    

    大语言模型（LLM）正逐步被应用于高级工程任务，包括计算机辅助设计（CAD）文档编制、标准合规性验证和知识检索。然而，它们容易产生幻觉，即看似可信但并非基于真实上下文的输出，这限制了其在对精度和合规性要求极高的高端工程应用中的可信度。本文提出了CoVe-RAG+，一个将验证链与检索增强生成（RAG）相集成的统一框架，用于减轻大语言模型生成结果中的幻觉。CoVe-RAG+支持大语言模型对外部权威来源（如工程标准、CAD信息和仿真报告）进行验证，同时应用迭代自验证过程来核实重要论断。CoVe-RAG+在CAD模型文档编制、标准合规性验证等工程任务上进行了评估。

    arXiv:2609.26229v1 Announce Type: new  Abstract: Large Language Models (LLMs) are progressively used for advanced engineering tasks, includes Computer-Aided Design (CAD) documentation, standards compliance verification, and knowledge retrieval. Still, they are prone to produce hallucinations, outputs that seem convincing but aren't based on context that limit their trustworthiness in high-end engineering applications where precision and compliance are crucial. The paper introduces CoVe-RAG+, a unified framework that integrates Chain-of-Verification (CoVe) with Retrieval-Augmented Generation (RAG) to mitigate hallucinations in the results generated by large language models (LLMs). CoVe-RAG+ supports LLM verification in external sources of authority, such as engineering standards, CAD information, and simulation reports, while applying an iterative self-verification process to validate important claims. CoVe-RAG+ is assessed on engineering activities such as CAD model documentation, stan
    
[^15]: WatchPoint：面向真实世界智能体Web开发的可执行用户反馈

    WatchPoint: Executable User Feedback for Real-World Agentic Web Development

    [https://arxiv.org/abs/2609.26204](https://arxiv.org/abs/2609.26204)

    WatchPoint是一个模拟用户系统，通过像真实开发者一样对运行中的Web应用生成并执行诊断脚本，产生结构化观察结果来指导编码智能体的重试，在包含1,000个顺序依赖任务的Web-Bench基准上恢复了57.6%的失败任务。

    

    当专业Web开发者的代码未通过测试时，他们不会仅仅重新阅读堆栈跟踪。他们会在浏览器中打开应用程序，点击按钮、检查计算样式并运行诊断命令，以了解问题出在哪里。现有的编码智能体反馈机制依赖于截图、LLM-as-a-judge评分或自然语言纠正，但很少有像开发者那样与实际运行的应用程序进行交互的。我们提出了WatchPoint，一个模拟用户系统，它通过针对运行中的应用程序生成并执行诊断脚本来模仿真实开发者的行为，产生结构化的观察结果来指导编码模型的重试。与以往针对单文件编辑或使用不可执行指标进行评估的方法不同，我们在Web-Bench上运行，这是一个包含50个多文件Web项目、共1,000个顺序依赖任务的基准，并通过确定性的端到端测试进行验证。WatchPoint恢复了57.6%……

    arXiv:2609.26204v1 Announce Type: cross  Abstract: When a professional web developer's code fails a test, they do not simply re-read the stack trace. They open the application in a browser, click buttons, inspect computed styles, and run diagnostic commands to understand what went wrong. Existing feedback mechanisms for coding agents rely on screenshots, LLM-as-a-judge scoring, or natural-language corrections, but few interact with the live application the way a developer would. We introduce WatchPoint, a simulated-user system that mimics real developer behavior by generating and executing diagnostic scripts against the running application, producing structured observations that guide the coding model's retry. Unlike prior approaches that target single-file edits or evaluate using non-executable metrics, we operate on Web-Bench, a benchmark of 50 multi-file web projects comprising 1,000 sequentially dependent tasks, verified by deterministic end-to-end tests. WatchPoint recovers 57.6% 
    
[^16]: 多CI服务采用的大规模纵向研究

    A Large-Scale Longitudinal Study of Multi-CI Service Adoption

    [https://arxiv.org/abs/2609.26181](https://arxiv.org/abs/2609.26181)

    本文通过对来自七种编程语言的135227个GitHub仓库和八种CI服务的大规模纵向研究，首次系统刻画了多CI服务在项目生命周期中的采用、演进与废弃模式，发现约五分之一的仓库使用多个CI服务，且多为迁移过程中的过渡性使用而非长期并行使用。

    

    持续集成（CI）的采用已经从单一实践演变为竞争服务林立的广阔格局，许多项目不再依赖单一的CI服务。然而，这种多服务行为在项目生命周期中如何演变仍然未知。以往的研究通过单个服务、特定编程语言或开发者访谈来考察CI的采用与迁移，使得多服务使用的生命周期仍未得到充分探索。本文对多CI服务的采用进行了一项大规模纵向研究，涵盖来自七种编程语言的135227个GitHub仓库，涉及八种CI服务。我们通过定量和定性分析刻画了CI服务随时间推移如何被采用、演进和放弃。大约五分之一的仓库使用多个CI服务，这往往是在迁移过程中的过渡性使用，而非长期的并行使用。CI服务的选择是采用过程中的主导因素……

    arXiv:2609.26181v1 Announce Type: new  Abstract: Continuous Integration (CI) adoption has evolved from a single practice into a broad landscape of competing services, and many projects no longer rely on a single CI service. However, how this multi-service behavior evolves over a project's lifetime remains unknown. Prior work has examined CI adoption and migration through individual services, specific languages, or developer interviews, leaving the lifecycle of multi-service usage underexplored. In this paper, we conduct a large-scale longitudinal study of multi-CI service adoption covering 135227 GitHub repositories from seven programming languages and involving eight CI services. We characterize how CI services are adopted, evolved, and abandoned over time through quantitative and qualitative analyses. About one in five repositories use multiple CI services, often transitional during migration rather than for sustained parallel use. CI service choice is the dominant factor across adop
    
[^17]: 为什么大语言模型在OCL生成中失败？一个图推理视角

    Why Do LLMs Fail at OCL Generation? A Graph Reasoning Perspective

    [https://arxiv.org/abs/2609.26122](https://arxiv.org/abs/2609.26122)

    该研究首次从图推理的视角系统揭示了LLM在OCL约束生成中失败的根本原因：生成性能随UML类图的导航深度和结构复杂度增加而显著下降，而词汇相似性的影响有限。

    

    大语言模型（LLM）日益被用于从自然语言规格说明和UML类图生成对象约束语言（OCL）约束。然而，现有工作主要聚焦于提升准确率，对这些模型为何失败的理解仍然有限。研究目标：本研究探究LLM在OCL生成中失败的根本原因，将该任务框定为对UML类图的图推理问题。研究方法：我们使用PathOCL数据集对六个最先进的LLM进行了实证评估，分析了UML结构特性（如导航深度和模型复杂度）、词汇相似性、提示排序策略以及图感知提示对OCL正确性的影响。研究结果：我们发现随着导航深度和结构复杂度的增加，OCL生成性能显著下降；词汇相似性的影响有限，而UML元素的文本排序方式则会产生（显著影响）。

    arXiv:2609.26122v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used to generate Object Constraint Language (OCL) constraints from natural language specifications and UML class diagrams. However, existing work mainly focuses on improving accuracy, with limited understanding of why these models fail. Aims. This study investigates the underlying causes of LLM failures in OCL generation, framing the task as a graph reasoning problem over UML class diagrams. Method. We conduct an empirical evaluation using the PathOCL dataset across six state-of-the-art LLMs. We analyze the impact of UML structural properties (e.g., navigation depth and model complexity), lexical similarity, prompt ordering strategies, and graph-aware prompting on OCL correctness. Results. We find that OCL generation performance significantly degrades with increasing navigation depth and structural complexity. Lexical similarity has limited influence, while textual ordering of UML elements af
    
[^18]: 面向混淆环境下鲁棒代码理解的大语言模型事后注意力引导方法

    Post-Hoc Attention Steering of Large Language Models for Robust Code Understanding under Obfuscation

    [https://arxiv.org/abs/2609.26102](https://arxiv.org/abs/2609.26102)

    提出CodeSteer方法，通过结合轻量级程序分析与推理时注意力引导，将大语言模型的注意力重新分配到后向切片、控制流路径等语义相关的程序元素上，从而显著提升其对混淆代码的理解鲁棒性。

    

    代码混淆被广泛应用于软件系统和恶意软件中，用于隐藏程序逻辑并阻碍分析，给人类开发者和自动化工具都带来了重大挑战。尽管大语言模型（LLMs）在代码理解方面展现出强大的能力，但它们对混淆的鲁棒性仍然缺乏充分研究。我们的初步研究表明，LLM在混淆代码上的性能显著下降，这表明其依赖于表层的词汇线索而非深层的语义推理。为了解决这一局限性，我们提出了CodeSteer，一种新颖的注意力引导方法，它将模型注意力重新分配到语义相关的程序元素上，包括用于输出预测的后向切片和用于执行推理的控制流路径。我们的方法将轻量级程序分析与推理时注意力引导相结合，引导LLM关注程序的核心输入到输出依赖关系。实验……

    arXiv:2609.26102v1 Announce Type: new  Abstract: Code obfuscation is widely used in software systems and malware to conceal program logic and hinder analysis, posing significant challenges for both human developers and automated tools. While large language models (LLMs) have shown strong capabilities in code understanding, their robustness to obfuscation remains poorly understood. Our preliminary study shows that LLM performance significantly degrades on obfuscated code, suggesting a reliance on superficial lexical cues rather than deep semantic reasoning. To address this limitation, we propose CodeSteer, a novel attention steering approach that reallocates model attention toward semantically relevant program elements, including backward slices for output prediction and control-flow paths for execution reasoning. Our method integrates lightweight program analysis with inference-time attention steering to guide LLMs toward the core input-to-output dependencies of a program. Experiments 
    
[^19]: 通过行为代理研究模型代码可理解性与人类代码可理解性的行为对齐

    On Behavioral Alignment of Model-Code and Human-Code Understandability via Behavioral Proxies

    [https://arxiv.org/abs/2609.26101](https://arxiv.org/abs/2609.26101)

    该论文提出将代码可理解性视为读者与代码交互的关系性属性，区分人类与模型的代码可理解性，并通过引入四种行为代理指标来研究两者之间的行为对齐。

    

    代码可理解性是软件质量的一个关键方面。以往的研究大多从以人为中心或以代码为中心的视角关注这一属性，而它实际上应被视为一种源于读者与代码之间交互的关系性属性。随着大语言模型在软件工程中的应用日益广泛，我们认为“读者”这一概念应被推广为同时涵盖人类和模型。基于这种关系性视角，我们将代码可理解性的概念加以扩展，区分人类代码可理解性与模型代码可理解性，旨在研究两者之间的行为对齐。为此，我们使用了一项先前研究的数据集，其中包含来自不同参与者群体对可理解性的人类评分判断，并评估了多个开源和闭源大语言模型。为了将这种行为对齐可操作化，我们引入了模型代码可理解性的四种行为代理指标。

    arXiv:2609.26101v1 Announce Type: new  Abstract: Code understandability is a critical aspect of software quality. Prior research has largely focused on this attribute from a human-centric or code-centric perspective, while it should be viewed as a relational property arising from the interaction between a reader and the code. With the increasing adoption of large language models in software engineering, we posit that the notion of "reader" should be generalized to encompass both humans and models. Building on this relational perspective, we extend the concept of code understandability to distinguish between human and model code understandability, aiming to investigate the behavioral alignment between the two. To this end, we use a dataset from a prior study containing human-rated judgments of understandability across diverse participant groups, and evaluate multiple open and closed-source LLMs. To operationalize such behavioral alignment, we introduce four behavioral proxies of model-c
    
[^20]: 观察生成式AI辅助下的系统性综述实施：来自软件工程研究生课程的经验报告

    Observing the Conduct of Systematic Reviews with Generative AI Support: An Experience Report from a Graduate Software Engineering Course

    [https://arxiv.org/abs/2609.26057](https://arxiv.org/abs/2609.26057)

    该经验报告通过观察软件工程博士生在有/无生成式AI辅助下开展试点系统性综述的课堂实践，发现大语言模型能降低入门门槛、加速备选方案生成并使方法论问题更加清晰，但也容易导致学生对AI的过度依赖。

    

    背景：二次研究（secondary studies）是循证软件工程中的基础实践，但教授这类研究需要设计能让学生面对真实方法论决策的活动。目标：本文报告了一门研究生课程中的教学经验，十名软件工程博士生分为三个小组，在有和没有生成式AI支持的情况下试行二次研究。方法：课程组织并观察了一次单日课堂活动，各小组在有和没有生成式AI辅助的条件下开展试点系统性综述；通过分析课堂观察记录、产出的成果以及与ChatGPT中配置的AI助手的交互对话，重建了各小组在整个活动中对该技术的采纳与使用方式。结果：大语言模型降低了初始门槛，加快了备选方案的生成，并使方法论问题更加显性化，但同时也助长了过度委托（对AI的过度依赖）……

    arXiv:2609.26057v1 Announce Type: cross  Abstract: Context: Secondary studies are fundamental practices in Evidence- Based Software Engineering, but teaching them requires activities that expose students to authentic methodological decisions. Objective: This paper reports an experience in a graduate course in which ten doctoral students in Software Engineering, organized into three groups, piloted secondary studies with and without support from generative AI. Method: A single-day classroom session was organized and observed, in which the groups conducted pilot systematic reviews with and without generative AI support. Classroom observations, produced artifacts, and interaction threads with assistants configured in ChatGPT were analyzed to reconstruct how each group appropriated the technology throughout the activity. Results: LLMs reduced initial barriers, accelerated the generation of alternatives, and made methodological problems more explicit, but they also favored excessive delegat
    
[^21]: FIRE：面向可靠语言模型智能体的故障知情运行时工程

    FIRE: Failure-Informed Runtime Engineering for Reliable Language-Model Agents

    [https://arxiv.org/abs/2609.26048](https://arxiv.org/abs/2609.26048)

    该论文提出FIRE，通过在不改变模型权重和用户提示的前提下，在失败发生前的状态处施加自然语言指令与动作拒绝等运行时策略，显著提升语言模型智能体的重复交付可靠性，在Terminal-Bench 2.1上pass^2最高提升9.2个百分点。

    

    语言模型智能体往往能够找到可行的解决方案，却无法稳定地将其交付。我们研究了运行时策略：由智能体框架在观测到失败之前的状态处施加的有针对性的自然语言指令和动作拒绝，且不改变模型权重或用户提示词。借助这一方法，在保持能力不变的情况下，我们观察到交付可靠性的显著提升。在完整的87个任务的Terminal-Bench 2.1套件上（每个任务尝试两次），策略使三个GPT-5.6层级的重复成功率（pass^2）均得到提升：Luna从50.6%提升至54.0%，Terra从55.2%提升至60.9%，Sol从64.4%提升至73.6%。Sol的两次尝试最佳成功率仅变化1.2个百分点，而重复成功率提升了9.2个百分点，这表明策略主要作用在于将可实现的解决方案转化为可靠的交付。我们进一步在Terra的冻结组合下测试了14个任务：策略引导的Terra达到71.4%，而无辅助的Sol为64.3%，成本约为其一半（原文摘要在此处截断）。

    arXiv:2609.26048v1 Announce Type: cross  Abstract: Language-model agents often reach a working solution and then fail to consistently deliver it. We study runtime policies: targeted natural-language instructions and action denials applied by the agent harness at states that preceded observed failures, without changing model weights or the user prompt. With this, keeping capability constant, we observe a meaningful unlock in delivered reliability. Across the complete 87-task Terminal-Bench 2.1 suite, with two attempts per task, policies increase repeated success (pass^2) in all three GPT-5.6 tiers: 50.6% to 54.0% for Luna, 55.2% to 60.9% for Terra, and 64.4% to 73.6% for Sol. Sol's best-of-two success changes by 1.2 points while repeated success rises by 9.2, showing that policies chiefly convert reachable solutions into dependable delivery. We further cover 14 tasks under Terra's frozen portfolio. Policy-guided Terra reaches 71.4%, compared with 64.3% for unassisted Sol, at about half 
    
[^22]: 从声明的损失与可达状态编译充分治理上下文：带基数与成本目标的精确观察契约综合

    Compiling Sufficient Governance Context from Declared Losses and Reachable States: Exact Observation-Contract Synthesis with Cardinality and Cost Objectives

    [https://arxiv.org/abs/2609.26016](https://arxiv.org/abs/2609.26016)

    该论文提出从有限可达状态模型与声明判定结果中精确综合“观察契约”，在可穷举时枚举所有包含极小的充分契约，否则借助SAT/MaxSAT编码求解最小基数或最小成本的契约，并区分个体不可或缺属性与联合充分契约。

    

    我们将本文推导并认证的对象称为最小充分治理上下文：给定一个有限可达状态模型、一个确定性的声明判定结果以及候选可观察属性，我们计算充分的观察集合，区分个体不可或缺的属性与联合充分的契约，并在基数或声明成本目标下从充分契约中进行选择。观察契约是一组候选属性，其取值在每一个可达状态上都能确定声明的判定结果；权威契约则是在某一目标下被选出并绑定到门控模式的契约。在穷举可行的情况下，我们综合出所有包含极小的充分契约；否则通过SAT/MaxSAT编码综合出最小基数或最小成本的契约，并直接检验其充分性。在一个构建的代码/云领域中，个体不可或缺的核心并不充分……

    arXiv:2609.26016v1 Announce Type: new  Abstract: We call the object this paper derives and certifies a minimal sufficient governance context: given a finite reachable-state model, a deterministic declared verdict, and candidate observable attributes, we compute sufficient observation sets, distinguish attributes that are individually indispensable from contracts that are jointly sufficient, and select among sufficient contracts under a cardinality or declared-cost objective. An observation contract is a set of candidate attributes whose values determine the declared verdict on every reachable state; an authority contract is one selected under an objective and bound to a gate schema. We synthesize every inclusion-minimal sufficient contract where exhaustive enumeration is affordable, and a minimum-cardinality or minimum-cost contract by SAT/MaxSAT encoding otherwise, checking sufficiency directly. On a constructed code/cloud domain, the individually-indispensable core is not sufficient 
    
[^23]: 面向汽车感知系统的视觉语言模型系统化认证研究

    Towards Systematic Qualification of Vision-Language Models for Automotive Perception Systems

    [https://arxiv.org/abs/2609.25945](https://arxiv.org/abs/2609.25945)

    该论文指出视觉语言模型在汽车感知系统中存在幻觉风险（既可能虚构交通对象，也可能漏检真实存在的对象），并强调需要系统性地结合设计时与运行时的验证确认技术，以实现VLM在安全关键汽车系统中的合格性认证。

    

    人工智能领域已被广泛应用于众多应用场景。视觉语言模型（VLM）是近期发展起来的先进AI技术之一，已被探索用于支持车辆感知和安全保障等汽车功能。然而，这类语言模型容易产生幻觉，对可能采用它们的汽车系统的安全性构成潜在威胁。在汽车领域，VLM不仅可能虚构出并不存在的交通对象，还可能无法识别实际存在的交通对象，这可能导致危险情况的发生。尽管我们观察到越来越多的文献提出了面向安全可信AI的验证与确认（V&V）技术，但这些方法往往是孤立研究的，要么专注于运行时阶段，要么专注于设计时阶段。这种孤立的技术在汽车等安全关键的现实场景中可能是不充分的。

    arXiv:2609.25945v1 Announce Type: cross  Abstract: The field of Artificial Intelligence has been adopted for many application domains. Vision Language Models are one of the recently advanced AI techniques that have been explored to support automotive features such as vehicle perception, and safety assurance. However, such language models are prone to hallucinations, posing a potential threat to the safety of automotive systems that may incorporate them. Within the automotive domain, VLMs could not only hallucinate traffic objects, but could also fail to identify traffic objects that are actually present, which may potentially lead to dangerous situations. Though we have observed a growing body of literature that proposes verification and validation techniques for safe and trustworthy AI, these methods are often studied in isolation, focusing either on run-time or design-time phases. Such isolated techniques could be insufficient in safety-critical, realistic contexts such as automotive
    
[^24]: 依赖更新何时应该调用修复智能体？一项轻量级路由研究

    When Should Dependency Updates Invoke Repair Agents? A Lightweight Routing Study

    [https://arxiv.org/abs/2609.25911](https://arxiv.org/abs/2609.25911)

    本文提出轻量级路由器DepFixRouter，仅利用PR创建时的标题和元数据信号预测哪些依赖更新拉取请求需要兼容性修复，避免对所有更新盲目调用昂贵的仓库级修复智能体，在节省调用的同时高效捕获真正需要修复的请求。

    

    依赖更新的拉取请求非常频繁且大多是例行性的，但其中一小部分需要非平凡的兼容性修复。近期的仓库级编码智能体使这类修复变得越来越可行，然而对每个依赖更新都调用它们会浪费模型调用次数、CI时间、仓库上下文以及审查注意力。我们将其框架化为一个智能体前置路由问题：在下游诊断或修复尝试之前，决定哪些依赖更新拉取请求应该被升级处理。我们提出了DepFixRouter，一个轻量级路由器，它使用创建时的文本和元数据信号，根据历史兼容性修复可能性对依赖更新进行排序。在497个带标签的GitHub依赖更新候选中，只有72个需要实质性修复。一个仅使用PR标题和机器人/依赖标志的、创建时安全的LinearSVC达到了0.488的修复F1分数，并在路由优先级最高的前20%拉取请求中捕获了51.4%的修复，提高了每次捕获修复所需的调用效率

    arXiv:2609.25911v1 Announce Type: new  Abstract: Dependency-update pull requests are frequent and mostly routine, but a small subset requires non-trivial compatibility repair. Recent repository-level coding agents make such repair increasingly plausible, yet invoking them on every dependency update wastes model calls, CI time, repository context, and review attention. We frame this as a pre-agent routing problem: deciding which dependency-update pull requests should be escalated before downstream diagnosis or repair attempts. We introduce DepFixRouter, a lightweight router that ranks dependency updates by historical compatibility-repair likelihood using creation-time textual and metadata signals. On 497 labeled GitHub dependency-update candidates, only 72 require substantive repair. A creation-time-safe LinearSVC using only PR titles and bot/dependency flags reaches 0.488 repair F1 and captures 51.4% of repairs within the top 20% routed pull requests, improving calls per captured repai
    
[^25]: 微服务系统中基于置信度引导的跨模态知识迁移多模态异常检测方法

    Confidence-Guided Cross-Modal Knowledge Transfer for Multimodal Anomaly Detection in Microservice Systems

    [https://arxiv.org/abs/2609.25856](https://arxiv.org/abs/2609.25856)

    提出了CMT-AD方法，通过统一的深度聚类框架联合建模微服务系统的指标与日志，并利用软聚类分布将模态不确定性量化为置信度分数，以引导跨模态知识迁移，从而应对模态可靠性动态变化和数据异构性的挑战，实现更准确的多模态异常检测。

    

    准确的异常检测对于微服务系统的可靠与安全运行至关重要。尽管越来越多的研究已从单模态建模转向多模态交互与融合，但如何有效利用可靠的跨模态信息仍然具有挑战性。该挑战主要源于两个方面：其一，不同模态受负载波动等因素影响，其可靠性会动态变化；其二，多模态数据在结构和语义上均表现出异构性。因此，我们提出了一种基于置信度引导的跨模态知识迁移多模态异常检测方法（CMT-AD）。该方法在统一的深度聚类框架内联合建模指标与日志，并通过软聚类分布估计模态可靠性，将聚类不确定性量化为置信度分数。在这些置信度分数的引导下，模型主动分析……

    arXiv:2609.25856v1 Announce Type: new  Abstract: Accurate anomaly detection is essential for reliable and secure operations of microservice systems. While an increasing number of studies have shifted from unimodal modeling to multimodal interaction and fusion, effectively leveraging reliable cross-modal information remains challenging. The challenge primarily stems from two aspects. Firstly, different modalities are influenced by factors like load fluctuations, leading to dynamically changing reliability. Secondly, multimodal data exhibit heterogeneity in both structure and semantics. Therefore, we propose a confidence-guided Cross-Modal knowledge Transfer method for multimodal Anomaly Detection (CMT-AD). It jointly models metrics and logs within a unified deep clustering framework and estimates modality reliability through the soft clustering distributions, where clustering uncertainty is quantified into confidence scores. Guided by these confidence scores, the model actively analyzes
    
[^26]: 曾经学到的可能需要被遗忘：大语言模型中已弃用API知识的机器遗忘

    What Was Once Learned May Need to Be Unlearned: Machine Unlearning for Deprecated API Knowledge in Large Language Models

    [https://arxiv.org/abs/2609.25786](https://arxiv.org/abs/2609.25786)

    本文针对大语言模型生成已弃用API的问题，构建了基于实际行为验证的基准MUDAPIBench（包含7000多个模型特定实例），并对八种机器遗忘方法在三个代码LLM上的遗忘效果与副作用进行了系统性实证研究。

    

    用于代码补全的大语言模型（LLMs）可能会生成已弃用的API，因为其预训练语料库中包含了历史版本的库代码。现有方法采用推理时干预、模型编辑或机器遗忘等技术，但由于存在多种合理的补全结果，预定义替换的方式具有较大局限性。此外，现有研究很少验证模型是否确实表现出目标弃用行为，也很少评估遗忘操作对其他API造成的意外影响。我们对已弃用API知识的机器遗忘进行了系统的实证研究，并构建了MUDAPIBench——一个基于行为的基准测试集，其中包含超过7,000个模型特定实例，这些实例源自八个Python库中145组从已弃用API到最新API的映射。只有当原始模型确实生成目标弃用API时，相应实例才会被保留。我们在三个代码大语言模型上评估了八种代表性的机器遗忘方法在弃用API遗忘、最新API保持等方面的表现。

    arXiv:2609.25786v1 Announce Type: new  Abstract: Large language models (LLMs) for code completion may generate deprecated APIs because their pre-training corpora contain code from historical library versions. Existing approaches use inference-time intervention, model editing, or machine unlearning, but multiple plausible completions make predefined replacements restrictive. Moreover, existing studies rarely verify whether models exhibit the targeted deprecated behavior or evaluate unintended changes to other APIs.   We conduct a systematic empirical study of machine unlearning for deprecated API knowledge and construct MUDAPIBench, a behavior-grounded benchmark with over 7,000 model-specific instances derived from 145 deprecated-to-up-to-date API mappings across eight Python libraries. Instances are retained only when the original model generates the target deprecated API. We evaluate eight representative unlearning methods across three code LLMs on deprecated API forgetting, up-to-dat
    
[^27]: 符号有限状态机的测试与学习

    Testing and Learning Symbolic Finite State Machines

    [https://arxiv.org/abs/2609.25603](https://arxiv.org/abs/2609.25603)

    该论文证明了符号有限状态机有限实例化的语言等价性可以推广到完整输入域上，从而将确定性有限状态机的完备测试与学习方法成功迁移到数据域可能无限的符号有限状态机。

    

    符号有限状态机（SFSMs）使用守卫条件和输出赋值来描述输入/输出行为，其数据域可能是无限的。我们研究确定性的、完全指定的符号有限状态机，其守卫条件和输出赋值仅依赖于当前输入。我们定义了有限的代表性输入集合，其中包含相关守卫重叠的见证，以及在这些重叠上取值不同的输出赋值的分离见证。我们的主要定理表明，有限实例化之间的语言等价性蕴含着在整个输入域上的语言等价性。这一结果将确定性有限状态机（DFSM）的完备测试方法转移到了符号有限状态机上，前提是已知可采纳守卫和输出赋值的有限集合，以及可区分可达状态数量的上界。在这些假设下，带有完备测试的DFSM学习器可以学习一个有限实例化，然后将其提升到符号有限状态机。

    arXiv:2609.25603v1 Announce Type: cross  Abstract: Symbolic finite state machines (SFSMs) describe input/output behaviour using guards and output assignments with possibly infinite data domains. We study deterministic and completely specified SFSMs whose guards and output assignments depend only on the current input. We define finite representative input sets that contain witnesses for relevant guard overlaps and separating witnesses for output assignments that differ on those overlaps. Our main theorem shows that language equivalence of the finite instantiations implies language equivalence over the full input domain. This result transfers complete testing methods for deterministic finite state machines (DFSMs) to SFSMs, provided finite sets of admissible guards and output assignments and an upper bound on the number of distinguishable reachable states are known. Under these assumptions, a DFSM learner with complete testing can learn a finite instantiation, which is then lifted to an 
    
[^28]: 理解社区驱动的科学工作流生态系统中的维护与支持：对Galaxy的跨空间研究

    Understanding Maintenance and Support in a Community-Driven Scientific Workflow Ecosystem: A Cross-Space Study of Galaxy

    [https://arxiv.org/abs/2609.25587](https://arxiv.org/abs/2609.25587)

    该研究通过对Galaxy生态系统中11,762个GitHub issue、52,203个pull request和6,235条论坛讨论的大规模跨空间实证分析，首次系统刻画了社区驱动科学工作流系统的维护与支持关注点，以及开发空间与社区支持空间中维护产物之间的关联。

    

    Galaxy是一个被广泛使用的、社区驱动的科学工作流系统，其可持续性依赖于对其软件、工具、工作流、基础设施、文档以及用户支持生态系统的持续维护。然而，Galaxy中的维护知识分散在开发空间和社区支持空间之中，这使得人们难以理解需要维护什么、维护产物如何被解决，以及用户关注的问题如何与仓库层面的开发相关联。我们对Galaxy开展了一项大规模实证研究，使用了11,762个GitHub issue、52,203个pull request以及6,235条社区论坛讨论。我们刻画了维护与支持方面的关注点，考察了与解决结果和解决时间相关的因素，并研究了这些空间中维护产物之间的显性联系和候选联系。利用BERTopic主题建模，我们识别出9个issue主题、14个pull request主题和14个论坛主题。

    arXiv:2609.25587v1 Announce Type: new  Abstract: Galaxy is a widely used, community-driven scientific workflow system whose sustainability depends on continuous maintenance across its software, tools, workflows, infrastructure, documentation, and user-support ecosystem. However, maintenance knowledge in Galaxy is distributed across development and community-support spaces, making it difficult to understand what is maintained, how maintenance artifacts are resolved, and how user-facing concerns connect to repository-level development. We conduct a large-scale empirical study of Galaxy using 11,762 GitHub issues, 52,203 pull requests, and 6,235 Community Forum discussions. We characterize maintenance and support concerns, examine factors associated with resolution outcomes and resolution time, and investigate explicit and candidate connections among maintenance artifacts across these spaces.   Using BERTopic modeling, we identify nine issue topics, 14 pull-request topics, and 14 forum to
    
[^29]: Python项目跨操作系统可移植性问题的实证分析

    An Empirical Analysis of Cross-OS Portability Issues in Python Projects

    [https://arxiv.org/abs/2609.25531](https://arxiv.org/abs/2609.25531)

    该论文开展了首个针对Python跨操作系统可移植性问题的大规模实证研究，分析了2,042个开源仓库，构建了包含7个主要故障类别、24个子类别、15个诊断特征和4种系统性修复模式的全面分类体系。

    

    尽管Python被设计为一种跨平台语言，但实际应用在部署到不同操作系统时会遇到可移植性故障。我们提出了首个针对Python跨操作系统可移植性问题的大规模实证研究，采用两种互补的方法分析了2,042个开源仓库：系统性的跨操作系统测试重执行以及对GitHub issue的人工分析。我们对500个项目的跨平台测试显示，11.2%的项目存在依赖于操作系统的测试失败。通过对240个GitHub issue的系统性分析，我们确认了102个真实的可移植性问题，涉及另外95个项目。我们构建了一个全面的分类体系，识别出7个主要故障类别——其中文件/目录操作、进程管理和库依赖最为普遍——以及24个不同的子类别、15个诊断特征和4种系统性修复模式。我们的评估表明，现有的静态分析工具（摘要在此处被截断）

    arXiv:2609.25531v1 Announce Type: new  Abstract: While Python is designed as a cross-platform language, real-world applications encounter portability failures when deployed across different operating systems. We present the first large-scale empirical study of cross-OS portability issues in Python, analyzing 2,042 open-source repositories using two complementary approaches: systematic cross-OS test reexecution and manual analysis of GitHub issues. Our cross-platform testing of 500 projects reveals that 11.2% exhibit OS-dependent test failures. Through systematic analysis of 240 GitHub issues, we confirm 102 genuine portability problems spanning 95 additional projects. We develop a comprehensive taxonomy identifying 7 primary failure categories - with file/directory operations, process management, and library dependencies being most prevalent - along with 24 distinct sub-categories, 15 diagnostic signatures, and 4 systematic repair patterns. Our evaluation reveals that existing static a
    
[^30]: 评估Shaker在Python项目中的不稳定测试检测效果

    Evaluating Shaker for Flaky Test Detection in Python Projects

    [https://arxiv.org/abs/2609.25528](https://arxiv.org/abs/2609.25528)

    本研究首次对Shaker工具在Python项目中的不稳定测试检测效果进行实证评估，发现其检测率（37.2%）与简单的重复执行方法（35.8%）无统计学显著差异，表明Shaker在Java/Android上的优势无法直接迁移到Python。

    

    不稳定测试是指在未更改的代码上非确定性地通过或失败的测试，这会削弱人们对测试套件的信任，并增加每次失败的处理成本。Shaker通过注入资源竞争（CPU、内存和I/O压力）来放大并发执行引起的非确定性，从而检测这类测试，据报道在Java和Android基准测试中能检测到95%的不稳定测试，而简单的重复执行方法（ReRun）只能检测到37.5%。我们首次对Shaker在Python中的表现进行了实证评估。我们从Gruber等人的基准真值数据集中提取非顺序依赖的不稳定测试，在配对设计中将Shaker与预算匹配的ReRun基线进行比较，为两种技术提供相同数量的测试执行次数：137个测试中的每一个在每种技术下均运行100次。按照为Java和Android配置的方式，Shaker相比简单的重复执行没有统计学上显著的检测优势（37.2% vs. 35.8%；McNemar精确检验p = 0.84）。两个发现解释了其中的原因。首先，更少的……

    arXiv:2609.25528v1 Announce Type: new  Abstract: Flaky tests pass or fail non-deterministically on unchanged code, eroding trust in test suites and inflating the cost of every failure. Shaker detects them by injecting resource contention (CPU, memory, and I/O stress) to amplify non-determinism caused by concurrent execution, and was reported to detect 95% of the flaky tests in a Java and Android benchmark against 37.5% for plain re-execution (ReRun). We present the first empirical evaluation of Shaker for Python. Drawing non-order-dependent flaky tests from the ground-truth dataset of Gruber et al., we compare Shaker against a budget-matched ReRun baseline in a paired design, giving both techniques the same number of test executions: Each of 137 tests is run 100 times under each. As configured for Java and Android, Shaker provides no statistically significant detection advantage over plain re-execution (37.2% vs. 35.8%; McNemar exact p = 0.84). Two findings explain why. First, fewer th
    
[^31]: 通过规范驱动的变异对WebGPU进行动态一致性测试

    Dynamic Conformance Testing of WebGPU Through Specification-Driven Mutation

    [https://arxiv.org/abs/2609.25520](https://arxiv.org/abs/2609.25520)

    LANTERN是一个规范引导的动态一致性测试框架，通过从WebGPU规范中提取语法规则和语义约束来变异官方CTS测试，从而生成有效与无效的测试变体以暴露实现中的缺陷。

    

    WebGPU是一种低级图形与计算API，它将现代GPU功能暴露给Web应用程序。官方的WebGPU一致性测试套件（CTS）专注于符合WebGPU规范的规范用法，但并未设计用于通过语义边缘情况或对抗性输入来对实现进行压力测试。相比之下，通用模糊测试工具由于WebGPU复杂的图形栈和多进程架构而难以对其有效测试。我们提出了LANTERN，这是一个规范引导的动态一致性测试框架，它使用从WebGPU规范中提取的约束来变异CTS测试。LANTERN从WebIDL定义中提取显式的语法API规则，并从自然语言规范文本中恢复语义约束，例如命令排序和对象生命周期。选定的规则指导基于AST定位的文本转换，生成有效和故意无效的CTS测试变体。我们执行由此生成的测试……（摘要在此处截断）

    arXiv:2609.25520v1 Announce Type: new  Abstract: WebGPU is a low-level graphics and compute API that exposes modern GPU functionality to web applications. While the official WebGPU Conformance Test Suite (CTS) focuses on well-formed usage under the WebGPU specification, it is not designed to stress implementations with semantic edge cases or adversarial inputs. General-purpose fuzzers, in contrast, struggle with WebGPU because of its complex graphics stack and multi-process architecture. We introduce LANTERN, a specification-guided dynamic conformance testing framework that mutates CTS tests using constraints extracted from the WebGPU specification. LANTERN extracts explicit syntactic API rules from WebIDL definitions and recovers semantic constraints, such as command ordering and object lifetimes, from natural-language specification text. Selected rules guide AST-located textual transformations that generate both valid and intentionally invalid CTS variants. We execute the resulting t
    
[^32]: Swift语言中不稳定测试的词汇特征

    The Vocabulary of Flaky Tests in Swift

    [https://arxiv.org/abs/2609.25516](https://arxiv.org/abs/2609.25516)

    该研究首次针对Swift语言评估了基于词汇的机器学习方法来预测不稳定测试，通过从15个开源项目收集数据并训练五个分类器，证明随机森林模型（F1=0.86，MCC=0.75）能够利用测试词汇特征有效识别不稳定测试，并显著优于简单基线方法。

    

    不稳定测试（Flaky tests）在代码未变更的情况下产生非确定性结果，削弱了持续集成（CI）的信心并延误交付。尽管基于词汇的机器学习预测方法已被证明对Java和JavaScript有效，但尚无研究针对Swift评估该方法，而Swift是一种测试风格以UI和异步代码为主的语言。我们通过重复执行和提交历史挖掘，从15个开源Swift项目中收集了91个不稳定测试和22,349个稳定测试，然后在TF-IDF一元加二元词组特征上，采用分层5折交叉验证训练了五个分类器（随机森林、决策树、朴素贝叶斯、支持向量机、K近邻）。随机森林取得了最佳性能（精确率=0.92，F1=0.86，AUC=0.95），并显著优于简单的基线方法，其中包括对信息量最大的词汇应用词汇阈值规则的方法，从而证实了真实的判别信号（MCC=0.75，而最佳基线仅为0.08）。信息增益分析揭示了两个互补的（摘要在此处被截断）

    arXiv:2609.25516v1 Announce Type: new  Abstract: Flaky tests produce non-deterministic outcomes without code change, eroding CI confidence and delaying deliveries. While vocabulary-based machine learning prediction has proven effective for Java and JavaScript, no study has evaluated it for Swift, a language whose testing style is dominated by UI and asynchronous code. We collect 91 flaky and 22,349 stable tests from 15 open-source Swift projects via re-execution and commit-history mining, then train five classifiers (Random Forest, Decision Tree, Naive Bayes, SVM, KNN) on TF-IDF unigram+bigram features under stratified 5-fold cross-validation. Random Forest achieves the best performance (Precision = 0.92, F1 = 0.86, AUC = 0.95) and substantially outperforms trivial baselines, among them a vocabulary-threshold rule applied to the most informative tokens, confirming a genuine discriminative signal (MCC = 0.75 vs. 0.08 for the best baseline). Information-gain analysis reveals two compleme
    
[^33]: 使用Lean元编程的归纳类型模块化组合

    Modular Composition of Inductive Types Using Lean Meta-programming

    [https://arxiv.org/abs/2609.25427](https://arxiv.org/abs/2609.25427)

    本文提出基于元编程的归纳类型与函数组合算法，并通过扩展Lean证明助手的语法，实现了归纳类型的模块化复用、组合与扩展，从而缓解表达式问题。

    

    归纳类型是许多编程语言和定理证明语言中普遍存在的基本构建块。归纳类型是一个封闭的构造器集合，通过这些构造器可以创建该类型的值。然而，一旦类型被定义，该集合便无法再扩展。这限制了在定义类型及其值上的函数时的可扩展性、复用性以及模块化的关注点分离。这一限制体现在表达式问题中：在不修改或重新编译已有语法构造器的情况下，用新的语法构造器扩展表达式语言，在几乎所有编程语言中都是一个难题。本文提出了基于元编程的归纳类型与函数实现组合算法，并给出了一组在Lean证明助手中实现这些算法的语法扩展。该框架允许对Lean类型和函数的一个子集进行模块化的复用、组合与扩展。

    arXiv:2609.25427v1 Announce Type: cross  Abstract: Inductive types are ubiquitous building blocks in many programming and theorem proving languages. An inductive type is a closed set of constructors from which values of the type can be created. That set cannot be extended though once a type is defined. This limits extensibility, reuse, and modular separation of concerns when defining types and functions operating over their values. This limitation is manifested in the expression problem, where extending an expression language with new syntactic constructors without having to modify or re-compile existing ones is a challenge in almost all programming languages.   This paper presents inductive type and function implementation composition algorithms based on meta-programming. In addition, a set of syntactic extensions to the Lean proof assistant implementing those algorithms are presented. This framework allows for modular reuse, composition, and extension of a subset of Lean type and fun
    
[^34]: 单独通过，合并失败：并行LLM智能体开发中的语义协调基准测试

    Passes Alone, Fails Together: Benchmarking Semantic Coordination in Parallel LLM-Agent Development

    [https://arxiv.org/abs/2609.25396](https://arxiv.org/abs/2609.25396)

    该论文提出了 stale 基准测试来衡量并行LLM编码智能体之间的语义协调问题，发现真实合并的拉取请求中干扰极少，但在使用真实 Django 代码构建的受控任务中 97% 的运行出现合并干扰，而一条简单的并发更改描述消息即可恢复 82% 的失败运行。

    

    并行编码智能体可以生成单独运行时有效的补丁，但在合并时却会失败。这种情况发生在当一个智能体更改了另一个智能体仍然依赖的接口或规则时。我们使用 stale——一个用于语义协调的基准测试——来研究这些失败。我们的评估方法是在每个补丁单独运行以及它们的组合上运行相同的测试，仅计算由合并补丁引入的失败。我们使用三个层级：具有受控接口更改的合成任务、成对合并的拉取请求，以及使用真实 Django 辅助函数构建的任务。在对 417 个挖掘的 Django 配对进行的 834 次运行中，在纠正评分程序后仅有一个案例显示出干扰。在使用 12 个 Django 辅助函数构建的任务中，97% 的运行出现了干扰。一条描述已完成的并发更改的消息可以恢复 82% 的失败运行。即使智能体在使用真实代码的受控任务上失败，经过审查的拉取请求中可能只包含很少的未解决并行更改。

    arXiv:2609.25396v1 Announce Type: new  Abstract: Parallel coding agents can produce patches that work alone but fail when merged. This happens when one agent changes an interface or rule that another agent still relies on. We study these failures with stale, a benchmark for semantic coordination. Our evaluation runs the same tests on each patch alone and on their combination, counting only failures introduced by combining the patches. We use three tiers: synthetic tasks with controlled interface changes, pairs of merged pull requests, and constructed tasks that use real Django helpers. Among 834 runs on 417 mined Django pairs, only one showed interference after correcting the grading procedure. On constructed tasks using 12 Django helpers, interference occurred in 97% of runs. A message describing the completed concurrent change recovered 82% of runs. Reviewed pull requests may contain few unresolved parallel changes, even when agents fail on controlled tasks using real code. The const
    
[^35]: 扩展FunctionGemma以实现实用的设备端移动函数调用

    Extending FunctionGemma for Practical On-Device Mobile Function Calling

    [https://arxiv.org/abs/2609.25373](https://arxiv.org/abs/2609.25373)

    该研究通过构建涵盖十五类设备控制操作的约9,500个对话的合成数据集MOBILEACTIONSEXTENDED对FunctionGemma 270M进行微调，将设备端Android函数调用的端到端准确率从29.3%大幅提升至76.5%，并使组合模型在Google数据集上达到82.3%。

    

    设备端助手需要能够将自然语言映射为本地系统操作的函数调用模型，但现有资源主要侧重于Web API或狭窄的移动操作目录。我们通过引入MOBILEACTIONSEXTENDED数据集，将FunctionGemma 270M-it扩展到实用的Android工作流程中。该数据集是一个合成的、经过模式校验的数据集，包含约9,500个对话，涵盖十五个设备控制类别，包括消息发送、电话呼叫、相机/截图、亮度控制、设备状态查询、手电筒控制以及应用程序管理。我们使用TRL监督微调方法（仅对完成部分计算损失）对270M模型进行微调，得到一个扩展的专用模型，以及一个与Google的MOBILEACTIONSGOOGLE联合训练的组合模型。在MOBILEACTIONSEXTENDED上，端到端准确率从基础模型的29.3%、Google Mobile-Actions变体的17.2%提升至76.5%。组合模型在MOBILEACTIONSEXTENDED上保持76.5%的准确率，并在MOBILEA（原文截断，应为Google数据集）上达到82.3%。

    arXiv:2609.25373v1 Announce Type: new  Abstract: On-device assistants require function-calling models that map natural language to local system actions, but existing resources emphasize web APIs or narrow mobile-action catalogs. We extend FunctionGemma 270M-it to practical Android workflows by introducing MOBILEACTIONSEXTENDED, a synthetic, schema-validated dataset of ~9,500 conversations covering fifteen device-control categories, including messaging, phone calls, camera/screenshot, brightness control, device-status queries, flashlight control, and application management. We fine-tune the 270M model with TRL supervised fine-tuning under completion-only loss, producing an extended specialist and a combined model trained jointly with Google's MOBILEACTIONSGOOGLE. On MOBILEACTIONSEXTENDED, end-to-end accuracy improves from 29.3% for the base model and 17.2% for Google's Mobile-Actions variant to 76.5%. The combined model retains 76.5% on MOBILEACTIONSEXTENDED and reaches 82.3% on MOBILEA
    
[^36]: SSP-Bench：一个用于安全、安保与隐私评估的混合数据生成框架

    SSP-Bench: A Hybrid Data Generation Framework for Safety, Security, and Privacy Evaluation

    [https://arxiv.org/abs/2609.25352](https://arxiv.org/abs/2609.25352)

    SSP-Bench 是一个动态基准生成框架，通过按需生成评估实例、外部来源保障标签有效性、多模型面板校准难度，并将基准构建形式化为多目标优化问题，从而克服了静态基准在 LLM 安全、安保与隐私评估中的分数饱和与数据污染等缺陷。

    

    大型语言模型（LLM）在安全、安保与隐私（SSP）方面的评估严重依赖静态基准，而这些基准存在分数饱和、数据污染和聚合伪影等问题，且无法捕捉模型对语言变化的敏感性。因此，在固定测试集上表现良好的模型，在语义等价的重述下往往表现不佳。我们提出了 SSP-Bench，一个动态基准测试框架，它能够按需生成评估实例，同时保持领域一致性。该框架通过外部可溯源的来源确保标签的有效性，通过针对特定服务的验证来强制执行范围约束，并利用多模型引导面板来校准难度。基准构建被形式化为一个关于难度、可分性、新颖性和多样性的多目标优化问题。在 24 个模型和四种 SSP 服务上的实验表明，SSP-Bench 揭示了静态评估的系统性失败，包括接近零的……

    arXiv:2609.25352v1 Announce Type: cross  Abstract: Evaluation of large language models (LLMs) for safety, security, and privacy (SSP) relies heavily on static benchmarks, which suffer from score saturation, data contamination, and aggregation artifacts, and fail to capture sensitivity to linguistic variation. As a result, models that perform well on fixed test sets often fail under semantically equivalent rephrasings. We introduce SSP-Bench, a dynamic benchmarking framework that generates evaluation instances on demand while preserving domain consistency. The framework ensures label validity through externally grounded sources, enforces scope via service-specific validation, and calibrates difficulty using a multi-model steering panel. Benchmark construction is formulated as a multi-objective optimization problem over difficulty, separability, novelty, and diversity. Across 24 models and four SSP services, SSP-Bench reveals systematic failures of static evaluation, including near-zero 
    
[^37]: TAILOR：面向长尾日志解析的模板保持式数据增强方法

    TAILOR: Template-Preserving Augmentation for Long-Tailed Log Parsing

    [https://arxiv.org/abs/2609.25261](https://arxiv.org/abs/2609.25261)

    该论文提出TAILOR方法，通过模板保持式的数据增强来应对日志解析中的长尾分布问题，提升对稀有日志模板的解析性能并使评估更加真实可靠。

    

    日志解析对于系统日志分析至关重要，它通过将非结构化的日志消息转换为结构化的日志模板，从而支持调试、监控和异常检测等任务。然而，现实世界的日志数据集呈现出高度不平衡的长尾分布，即少数频繁出现的模板占据主导地位，而许多稀有模板仅出现少数几次。这种不平衡导致评估结果过于乐观，因为频繁模板主导了基准测试指标，而在稀有但运维上十分重要的事件上的糟糕性能在很大程度上被掩盖了。在本文中，我们研究了稀有日志组（定义为实例数少于五个的日志组）的普遍性及其影响。我们对广泛使用的Loghub-2.0基准进行的实证研究表明，稀有日志组占所有模板的近20%，但在日志消息中占比不到0.01%。由于它们仅包含极少数实例，

    arXiv:2609.25261v1 Announce Type: new  Abstract: Log parsing is essential for system log analysis because it supports tasks such as debugging, monitoring, and anomaly detection by transforming unstructured log messages into structured log templates. However, real-world log datasets exhibit highly imbalanced, long-tailed distributions, where a small number of frequent templates dominate while many rare templates appear only a few times. This imbalance causes evaluation results to be overly optimistic by allowing frequent templates to dominate benchmark metrics, while poor performance on rare yet operationally important events remains largely hidden. In this paper, we investigate the prevalence and impact of rare log groups, defined as log groups with fewer than five instances. Our empirical study on the widely used Loghub-2.0 benchmark shows that rare log groups account for nearly 20% of all templates but less than 0.01% of log messages. Because they contain only a handful of instances,
    
[^38]: TRACTOR：用于评估 C 到 Rust 代码转换工具的基准测试

    TRACTOR Benchmark for Evaluating C to Rust Translators

    [https://arxiv.org/abs/2609.25121](https://arxiv.org/abs/2609.25121)

    本文介绍了由 MIT 林肯实验室在 DARPA TRACTOR 项目下开发的标准化基准测试，通过难度递增的测试集和评估基础设施，系统性地评估 C 到 Rust 自动翻译工具的能力。

    

    内存安全漏洞仍然是关键软件中持续存在的安全风险来源，其中许多软件是用 C 和 C++ 等内存不安全语言编写的。编程语言、程序分析和人工智能领域的最新进展，为通过自动翻译到 Rust 等内存安全语言来现代化改造这些遗留系统创造了新的机遇。DARPA 的“将所有 C 代码翻译为 Rust”（Translating All C to Rust，简称 TRACTOR）项目旨在开发可扩展的技术，将大型 C 代码库转换为安全、高性能且易于维护的 Rust 代码。麻省理工学院林肯实验室（MIT Lincoln Laboratory）作为该项目的独立测试与评估机构，开发了一个标准化基准，用于系统性评估各类 C 到 Rust 的翻译工具。本报告介绍了 TRACTOR 基准测试，包括难度逐步递增的测试集和更大规模的里程碑项目，以及配套的评估基础设施和相关指标。

    arXiv:2609.25121v1 Announce Type: new  Abstract: Memory-safety vulnerabilities remain a persistent source of security risk in critical software, much of which is implemented in memory-unsafe languages such as C and C++. Recent advances in programming languages, program analysis, and artificial intelligence have created new opportunities to modernize these legacy systems through automated translation to memory-safe languages such as Rust. The DARPA Translating All C to Rust (TRACTOR) program seeks to develop scalable techniques for translating large C codebases into safe, performant, and maintainable Rust. MIT Lincoln Laboratory serves as the program's independent test and evaluation organization and has developed a standardized benchmark for systematically assessing C-to-Rust translation tools. This report describes the TRACTOR benchmark, including progressively challenging test batteries and larger milestone projects, as well as the supporting evaluation infrastructure and metrics for
    
[^39]: OSFoundry：基于规范引导智能体的操作系统构建与演进

    OSFoundry: Building and Evolving Operating Systems with Specification-Guided Agents

    [https://arxiv.org/abs/2609.25018](https://arxiv.org/abs/2609.25018)

    OSFoundry 提出了一个操作系统专用的智能体框架，其核心思想是将稳定的设计意图与多样化的实现相分离，通过共享开发蓝图 SysSpec*（包含任务边界的 Plan 和 OS 特定的 Specification）使设计意图在操作系统开发全程保持持久，从而提升操作系统持续演进的可靠性。

    

    操作系统必须持续演进。然而其开发仍然以代码为中心且主要依赖人工：即使是一个局部性的改动，也可能需要恢复隐含假设、协调多个子系统，并反复进行构建、启动、测试和调试整个系统。通用编码智能体虽然能自动化单个编辑任务，但其以提示词为核心的工作流需要不断从分散的上下文中重建任务边界和操作系统语义，这限制了它们在持续操作系统演进中的可靠性。本文提出了 OSFoundry，一个面向操作系统的专用智能体框架，使设计意图在整个操作系统开发过程中保持持久。其关键洞察是将稳定的意图与多样化的实现相分离。OSFoundry 不依赖自由格式的自然语言提示词来传达设计意图，而是使用 SysSpec*，一个共享的开发蓝图，由界定任务范围的 Plan 和操作系统特定的 Specification 组成：Plan 界定一个改动

    arXiv:2609.25018v1 Announce Type: cross  Abstract: Operating systems must evolve continuously. Yet their development remains code-centric and largely manual: even a localized change can require recovering implicit assumptions, coordinating multiple subsystems, and repeatedly building, booting, testing, and debugging the complete system. General-purpose coding agents automate individual edits, but their prompt-centric workflows repeatedly reconstruct task boundaries and OS semantics from scattered context, limiting their reliability for sustained OS evolution. This paper presents OSFoundry, an OS-specialized agent harness that makes design intent persistent throughout OS development. Its key insight is to separate stable intent from diverse implementation. Instead of relying on free-form natural-language prompts to convey design intent, OSFoundry uses SysSpec*, a shared development blueprint comprising a task-bounding Plan and an OS-specific Specification: the Plan bounds what a change 
    
[^40]: 量化前沿大语言模型智能体的过度宣称倾向

    Quantifying Overclaiming Propensity in Frontier LLM Agents

    [https://arxiv.org/abs/2609.20812](https://arxiv.org/abs/2609.20812)

    本文提出OverclaimBench评估套件，首次量化了前沿LLM编码智能体在最终回复中“过度宣称”任务完成的倾向，并发现在67.9%的运行中智能体并未真正阅读所有被要求审查的文件。

    

    前沿编码智能体越来越被信任可以长时间自主工作，然而智能体的最终回复往往是用户能够看到的关于该工作的唯一记录。我们量化了前沿智能体“过度宣称”任务完成的倾向，这种失实陈述可能会误导用户。当智能体的最终回复与其上下文中的信息相矛盾时，即发生了过度宣称。这一定义无需对意图进行推断，且与任务是否成功无关。我们提出了OverclaimBench，这是一个由五个文件审查场景、基于对话记录的覆盖率测量以及预先登记的植入缺陷组成的评估套件。我们在八款专有前沿模型各自的生产级命令行界面中对其进行评估，并在单一固定测试框架下对四个开放权重模型进行评估，结果发现：1）在67.9%的运行中，智能体并未阅读所有被要求审查的文件；2）在未完整阅读文件的运行中，智能体……（摘要在此处被截断）

    arXiv:2609.20812v1 Announce Type: cross  Abstract: Frontier coding agents are increasingly trusted to work autonomously for long periods, yet an agent's final response is often the only account of that work a user sees. We quantify the propensity of frontier agents to \emph{overclaim} task completion, a misrepresentation that can mislead the user. An agent overclaims when its final response contradicts information in its context. This definition requires no inference about intent and is independent of task success. We introduce \emph{OverclaimBench}, an evaluation suite composed of five file-review scenarios, transcript-based coverage measurements, and registered planted defects. We evaluate eight proprietary frontier models in their own production command-line interfaces, and four open-weight models under a single fixed harness on OverclaimBench and find that 1) agents do not read all the files they were asked to review in 67.9\% of runs; 2) among runs where not all files are read, ag
    
[^41]: 模型作为AI原生MBSE的受治理接口：读侧充分性与写侧可采性

    Models as Governed Interfaces for AI-Native MBSE: Read-Side Adequacy and Write-Side Admissibility

    [https://arxiv.org/abs/2609.16252](https://arxiv.org/abs/2609.16252)

    该论文指出AI参与模型驱动系统工程（MBSE）的关键瓶颈不在建模语言而在数据架构，提出“认知充分性”这一数据架构模式，通过“读侧充分性”与“写侧可采性”防止AI用不可验证、不受治理的训练数据填补模型信息缺口。

    

    摘要：机器可读模型（如SysML v2）如今已可通过编程方式访问，越来越多的研究将这种可访问性视为AI参与系统工程的使能条件。然而，访问是必要的，但并不充分。剩余的工作不在于建模语言本身，而在于围绕它的数据架构。一个查询结构完整模型以进行推导的AI读取器，仍然会遇到推导链缺失、认知状态未标记、溯源信息缺失以及模型无法解析的证据等问题。面对这些缺口，AI不会选择弃权，而是从训练数据中填补——而训练数据这一来源既不可验证，也不受治理。为了在一个按当前实践标准堪称典范而非存在缺陷的模型上论证这一观点，我们探查了公开的阿波罗11号SysML v2重建模型。我们将这种缺失的属性命名为“认知充分性”，并将其作为一种候选数据架构模式分为两个部分。读侧充分性允许……（摘要在此处被截断）

    arXiv:2609.16252v1 Announce Type: cross  Abstract: Machine-readable models such as SysML v2 are now programmatically accessible, and a growing body of work treats that access as the enabling condition for AI participation in systems engineering. Access is necessary, but not sufficient. The remaining work lies not in the modelling language but in the data architecture around it. An AI reader that queries a structurally complete model for a derivation still runs into absent derivation chains, untagged epistemic status, missing provenance, and evidence that the model cannot resolve. Faced with these gaps, it does not abstain; it fills them from training data, a source that is neither verifiable nor governed. To make the case on a model that is exemplary by current practice rather than deficient, we probe the public Apollo 11 SysML v2 reconstruction. We name the missing property epistemic adequacy and offer it as a candidate data-architecture pattern in two halves. Read-side adequacy lets 
    
[^42]: 基于账本控制的零样本自编排提升LLM编码性能

    Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance

    [https://arxiv.org/abs/2608.26480](https://arxiv.org/abs/2608.26480)

    本文证明，在不进行训练或基准调优的情况下，基于账本控制的管理器-工作器脚手架能显著提升某些LLM的编码性能，但效果因模型而异，并非普遍适用。

    

    多智能体大语言模型系统被广泛报道能超越单模型基线，但证据不一，且比较通常存在混淆：流程同时改变令牌预算、工具调用和提示，因此总体增益很少能揭示真正有效的因素。我们研究了在共享文件系统工作区中引入管理器-工作器脚手架的效果，无需训练且无需针对基准进行调优，与同一模型单次回答进行对比。在九个模型上——五个开放权重模型，参数范围从9B到约2.8T，以及四个前沿封闭模型——针对LiveCodeBench最新的100个困难问题，脚手架的好处是真实但有条件的：对某些模型效果显著且统计显著（如Qwen3.8-27B提升23.4，GPT-5.6-Luna提升10.6，GPT-5.6-Terra提升8.0，各基于五次配对运行；Kimi-K3提升30.4，Minimax-M3提升11.0，基于五次配对运行且关闭推理，p值均小于10^-4，以及...）

    arXiv:2608.26480v1 Announce Type: cross  Abstract: Multi-agent large language model systems are widely reported to beat single-model baselines, but the evidence is mixed, and comparisons are usually confounded: pipelines change token budgets, tool calls, and prompts simultaneously, so an aggregate gain rarely reveals what actually helped. We investigate the effect of introducing the manager-worker scaffold over a shared filesystem workspace, with no training and no per-benchmark tuning, measured against the same model answering in a single pass. Across nine models -- five open-weight, spanning 9B to ~2.8T parameters, and four frontier closed models -- on the 100 latest hard LiveCodeBench problems, the scaffold's benefit is real but conditional: large and statistically significant for some (Qwen3.8-27B +23.4, GPT-5.6-Luna +10.6 and GPT-5.6-Terra +8.0, each over five paired passes; Kimi-K3 +30.4 and Minimax-M3 +11.0 over five paired passes with reasoning off, both at $p < 10^{-4}$, and +
    
[^43]: AutoSQL：从大型代码库中的命令式ORM代码提取SQL模板

    AutoSQL: Extracting SQL Templates from Imperative ORM Code in Large-Scale Repositories

    [https://arxiv.org/abs/2608.15595](https://arxiv.org/abs/2608.15595)

    AutoSQL通过构建代码索引和混合上下文检索策略，利用LLM代理从Go ORM命令式代码中自动提取SQL模板，解决了静态恢复SQL的难题。

    

    arXiv:2608.15595v1 公告类型：新 摘要：次优的SQL查询会显著降低云系统的性能，这促使在部署前提取和审计SQL语句。然而，Go ORM框架通过分散的方法调用序列以命令式方式构建SQL，使得静态恢复生成的SQL模板变得困难。我们提出了AutoSQL，一个从Go ORM代码重建SQL模板的系统。AutoSQL构建了一个代码索引，这是一个有向图，捕获函数、类型和全局变量之间的结构依赖关系作为可导航的边。然后，它从ORM调用点向上追踪调用链，以识别与数据库交互的函数作为入口点。对于每个入口点，一个LLM代理遍历代码索引以收集影响SQL生成的代码片段，当图无法解决检索目标时，切换到基于模式的搜索。我们称这种策略为混合上下文检索。一旦获得足够的上下文，...

    arXiv:2608.15595v1 Announce Type: new  Abstract: Suboptimal SQL queries can significantly degrade the performance of cloud systems, motivating the extraction and auditing of SQL statements before deployment. However, Go ORM frameworks construct SQL imperatively through scattered method-call sequences, making it difficult to statically recover the resulting SQL templates. We present AutoSQL, a system that reconstructs SQL templates from Go ORM code. AutoSQL constructs a Code Index, a directed graph that captures structural dependencies between functions, types, and global variables as navigable edges. It then traces upstream call chains from ORM invocation sites to identify database-interacting functions as entry points. For each entry point, an LLM agent traverses the Code Index to collect code slices that influence SQL generation, switching to pattern-based search when the graph cannot resolve a retrieval goal. We call this strategy Hybrid Context Retrieval. Once sufficient context is
    
[^44]: 使用大语言模型智能体改进约束模型

    Improving Constraint Models with LLM Agents

    [https://arxiv.org/abs/2608.08127](https://arxiv.org/abs/2608.08127)

    提出了一个基于大语言模型智能体的框架，能够从开放式空间自动重构约束规划模型，通过将解注入原始模型进行经验性验证并诊断修复，在中位数约十五分钟内返回最佳改进模型变体。

    

    约束规划（CP）求解器的运行时间对建模选择高度敏感，例如对称性破除、隐含约束、全局约束、约束重构和变量表示。改进这些约束模型传统上需要人类专业知识，而现有的自动重构系统仅限于预定义的手工制作转换规则库。我们引入了一个智能体框架，它从开放式的空间对约束模型进行重构，并通过经验验证而非构造方式来确立正确性：一个大语言模型（LLM）智能体在给定一个模型和三个训练实例的情况下，提出替代的公式化表述，通过将其解注入原始模型来验证每个方案，并诊断和修复失败，在中位数约十五分钟内返回其找到的最佳变体。这些模型使用CPMpy建模库表达。

    arXiv:2608.08127v2 Announce Type: replace-cross  Abstract: The runtime of Constraint Programming (CP) solvers is highly sensitive to modeling choices, such as symmetry breaking, implied constraints, global constraints, constraint reformulation, and variable representation. Improving these constraint models has traditionally required human expertise, and existing automated reformulation systems are restricted to a predefined library of hand-crafted transformation rules. We introduce an agentic framework that instead reformulates a constraint model from an open-ended space and establishes correctness empirically rather than by construction: a Large Language Model (LLM) agent, given a model and three training instances, proposes alternative formulations, validates each by injecting its solution back into the original model, and diagnoses and repairs failures, returning the best variant it finds in a median of about fifteen minutes. The models are expressed in the CPMpy modeling library, a
    
[^45]: Isabelle/STARK：在 Isabelle/HOL 中对 zk-STARK 的形式化验证

    Isabelle/STARK: A Formalization of zk-STARK in Isabelle/HOL

    [https://arxiv.org/abs/2608.01965](https://arxiv.org/abs/2608.01965)

    该论文在 Isabelle/HOL 中对 STARK 协议进行了机械化形式化验证，通过 FRI 相关一致性推理、Merkle 认证和精确采样计数等技术证明了带查询次数限制的可靠性，并在具体参数下给出虚假接受概率不超过 2^{-137} 的界。

    

    本报告在 Isabelle/HOL 中对一种 STARK 风格的协议进行了机械化验证，证明了带查询次数限制的可靠性。通过一个保持接受性的嵌入，将自适应的、私有随机化的 Fiat-Shamir 交互记录生成器与已建立的分阶段敌手实验以及原始的概率验证器连接起来。该证明结合了 FRI 相关一致性推理、Merkle 认证、精确的模采样器计数以及加权路径放大等技术。事件敏感的计数方法在不改变验证器、也不将重复的预言机调用视为免费的情况下，精化了完整的错误界限。对于一个经过认证的 192 位素数域、迹长度为 1024、查询重复次数为 640 的具体设定，该定理表明：对于每一个满足统一语法预言机调用界限 fs_query_bound (2^20) P 的被建模生成器，虚假端点接受概率的上界为 2^{-137}。此外，另一个单独的诚实完备性定理给出：在相同设定下，正确的平方序列端点的接受概率为 1。

    arXiv:2608.01965v2 Announce Type: replace-cross  Abstract: This report presents mechanized query-bounded soundness for a STARK-style protocol in Isabelle/HOL. An acceptance-preserving embedding connects adaptive, privately randomized Fiat-Shamir transcript producers to an established staged adversary experiment and the original probabilistic verifier. The proof combines FRI correlated-agreement reasoning, Merkle authentication, exact modulo-sampler accounting and weighted-path amplification. Event-sensitive accounting refines the complete error bound without changing the verifier or treating repeated oracle calls as free. For a certified 192-bit prime field, trace length 1024 and 640 query repetitions, the concrete theorem bounds false-endpoint acceptance by $2^{-137}$ for every modeled producer satisfying the uniform syntactic oracle-call bound fs_query_bound $(2^{20})$ P. A separate honest-completeness theorem gives acceptance one for the correct square-sequence endpoint in the same 
    
[^46]: XScientist：一种用于长期自主科学发现的类Git研究协议

    XScientist: A Git-Like Research Protocol for Long-Running Autonomous Scientific Discovery

    [https://arxiv.org/abs/2607.12301](https://arxiv.org/abs/2607.12301)

    XScientist提出了一种类Git的研究协议，将研究状态而非论文手稿作为延续单位，通过类型化内容寻址对象、不可变检查点和声明-证据闭环等机制，使自主科学发现过程可检查、可分叉、可验证且可长期延续。

    

    自主研究系统能够生成看似合理的论文，却会丢失检查或延续研究所需的决策记录、失败的分支和证据。我们提出XScientist，一种本地优先的类Git协议，它将研究状态而非论文手稿作为延续的基本单位。假设、实验尝试、观察结果、声明、评审和交接被表示为探索图中类型化的、内容寻址的对象。不可变检查点、明确的负面结果记录、声明-证据闭环、重放边界以及权限感知门控，使每一次状态转换都可被检查，同时不会将通过完整性检查误认为科学真理。该协议可导出一种可移植的智能体原生研究工件（ARA），供其他智能体或人类检查、分叉、验证和扩展。一个参考实现集成了规划、执行、评审、修复和受监督的长期运行，同时完整保留溯源信息。

    arXiv:2607.12301v2 Announce Type: replace  Abstract: Autonomous research systems can generate plausible papers while losing the decisions, failed branches, and evidence needed to inspect or continue the work. We present XScientist, a local-first, git-like protocol that treats research state, rather than a manuscript, as the unit of continuation. Hypotheses, experiment attempts, observations, claims, reviews, and handoffs are represented as typed, content-addressed objects in an exploration graph. Immutable checkpoints, explicit negative outcomes, claim-evidence closure, replay boundaries, and authority-aware gates make each transition inspectable without treating a passing integrity check as scientific truth. The protocol exports a portable Agent-Native Research Artifact (ARA) that another agent or human can inspect, fork, verify, and extend. A reference implementation integrates planning, execution, review, repair, and supervised long-running operation while preserving provenance acro
    
[^47]: Petrify：基于Petri网的Java字节码并发性质分析

    Petrify: Petri-net Based Analysis of Concurrency Properties in Java Bytecode

    [https://arxiv.org/abs/2607.00830](https://arxiv.org/abs/2607.00830)

    Petrify将Java字节码程序语义编码为简洁的Petri网，借助LoLA等模型检测工具实现对并发性质的自动化验证，在表达能力与实用性之间取得了独特的平衡。

    

    自动化形式化验证领域的技术在权衡取舍上存在显著差异：一些技术侧重于表达能力和精确性，支持对复杂性质的验证；另一些则倾向于可扩展性和实用性，以便能够应用于使用不同特性的较大规模程序。本文提出了Petrify，这是一种新颖的并发性质自动化验证技术，实现了独特的权衡。Petrify将Java字节码程序的语义编码为Petri网（PN），从而可以使用LoLA等最先进的模型检测工具进行分析。正如我们的实验所证明的，Petrify的方法提供了表达能力与实用性的有趣结合：Petri网是对程序并发行为相当精确的编码；同时，Petrify的Petri网编码十分简洁，因此其分析对参数规模的变化相当不敏感。另一个实际……

    arXiv:2607.00830v2 Announce Type: replace  Abstract: The landscape of automated formal verification is populated by techniques that make prominently different trade-offs: some focus on expressiveness and precision, supporting the verification of complex properties; others favor scalability and practicality, so that they are applicable to larger programs using different features. This paper presents Petrify, a novel automated verification technique for concurrency properties that achieves a distinctive trade-off. Petrify encodes the semantics of Java bytecode programs into Petri nets (PNs), which can be analyzed by state-of-the-art model checking tools such as LoLA. As our experiments demonstrate, Petrify's approach offers an interesting combination of expressiveness and practicality: PNs are a fairly precise encoding of the concurrent behavior of programs; at the same time, Petrify's PN encoding is succinct, so that its analysis remains quite insensitive to parameter size. Another prac
    
[^48]: 真金白银，虚假模型：影子API中的欺骗性模型声明

    Real Money, Fake Models: Deceptive Model Claims in Shadow APIs

    [https://arxiv.org/abs/2603.01919](https://arxiv.org/abs/2603.01919)

    该论文首次对官方大语言模型API与影子API进行系统性审计，识别出17个已被187篇学术论文使用的影子API，揭示了影子API可能提供与官方API不一致的输出，从而威胁下游应用的可靠性和学术研究结果的有效性。

    

    获取GPT-5和Gemini-2.5等前沿大语言模型（LLM）常常受到高定价、支付障碍和地区限制的阻碍。这些限制推动了“影子API”的泛滥——影子API是第三方服务，声称通过间接访问提供不受地区限制的官方模型服务。尽管影子API被广泛使用，但它们是否提供与官方API一致的输出仍不清楚，这引发了对依赖它们的下游应用可靠性以及相关研究发现有效性的担忧。在本文中，我们首次对官方LLM API和相应的影子API进行了系统性审计。我们首先识别出17个已被187篇学术论文使用的影子API，其中最受欢迎的一个截至2025年12月6日获得了超过5,900次引用和58,000个GitHub星标。通过多维度审计……

    arXiv:2603.01919v3 Announce Type: replace-cross  Abstract: Access to frontier large language models (LLMs), such as GPT-5 and Gemini-2.5, is often hindered by high pricing, payment barriers, and regional restrictions. These limitations drive the proliferation of $\textit{shadow APIs}$, third-party services that claim to provide access to official model services without regional limitations via indirect access. Despite their widespread use, it remains unclear whether shadow APIs deliver outputs consistent with those of the official APIs, raising concerns about the reliability of downstream applications and the validity of research findings that depend on them. In this paper, we present the first systematic audit between official LLM APIs and corresponding shadow APIs. We first identify 17 shadow APIs that have been utilized in 187 academic papers, with the most popular one reaching more than 5,900 citations and 58,000 GitHub stars by December 6, 2025. Through multidimensional auditing o
    
[^49]: VeriSoftBench：面向Lean的仓库级形式化验证基准

    VeriSoftBench: Repository-Scale Formal Verification Benchmarks for Lean

    [https://arxiv.org/abs/2602.18307](https://arxiv.org/abs/2602.18307)

    VeriSoftBench是一个包含500个证明义务的仓库级Lean 4形式化验证基准，评估发现专为Mathlib数学调优的证明器难以迁移到以仓库为中心的软件验证场景，且任务成功率与其传递性依赖闭包的规模密切相关。

    

    大型语言模型在交互式定理证明领域取得了显著成果，尤其是在Lean中。然而，大多数针对基于LLM的证明自动化的基准都取自Mathlib生态系统中的数学内容，而软件验证中的证明则是在包含丰富定义和大量项目专用库的代码库中开发的。我们提出了VeriSoftBench，这是一个包含500个Lean 4证明义务的基准，这些证明义务取自开源形式化方法项目，并经过打包处理以保留真实的仓库上下文和跨文件依赖关系。我们对前沿LLM和专业证明器的评估得出了三项观察结果。首先，针对Mathlib风格数学调优的证明器在这种以仓库为中心的环境中迁移效果不佳。其次，成功与传递性仓库依赖密切相关：其证明依赖于大型、多跳依赖闭包的任务更不容易被解决。第三，提供精选的上下文……

    arXiv:2602.18307v2 Announce Type: replace-cross  Abstract: Large language models have achieved striking results in interactive theorem proving, particularly in Lean. However, most benchmarks for LLM-based proof automation are drawn from mathematics in the Mathlib ecosystem, whereas proofs in software verification are developed inside definition-rich codebases with substantial project-specific libraries. We introduce VeriSoftBench, a benchmark of 500 Lean 4 proof obligations drawn from open-source formal-methods developments and packaged to preserve realistic repository context and cross-file dependencies. Our evaluation of frontier LLMs and specialized provers yields three observations. First, provers tuned for Mathlib-style mathematics transfer poorly to this repository-centric setting. Second, success is strongly correlated with transitive repository dependence: tasks whose proofs draw on large, multi-hop dependency closures are less likely to be solved. Third, providing curated cont
    
[^50]: SWE-Universe：将真实世界可验证环境扩展至百万规模

    SWE-Universe: Scale Real-World Verifiable Environments to Millions

    [https://arxiv.org/abs/2602.02361](https://arxiv.org/abs/2602.02361)

    SWE-Universe提出一个可扩展框架，利用定制训练的构建智能体从GitHub PR中自动构建了80余万个真实世界软件工程可验证环境，并将Qwen3-Max-Thinking在SWE-Bench Verified上的成绩提升至75.3%。

    

    我们提出了SWE-Universe，一个可扩展且高效的框架，用于从GitHub拉取请求（PR）中自动构建真实世界软件工程（SWE）可验证环境。为了克服自动构建中普遍存在的挑战，如低产出率、弱验证器和高昂成本，我们的框架利用一个由高效定制训练模型驱动的构建智能体。该智能体采用迭代式自我验证和循环内作弊检测机制，以确保可靠地生成高保真、可验证的任务。使用这种方法，我们将真实世界多语言SWE环境的数量扩展到百万规模（807,693个）。我们通过大规模智能体中期训练和强化学习展示了这些环境的深远价值。最后，我们将该技术应用于Qwen3-Max-Thinking，在SWE-Bench Verified上取得了75.3%的分数。我们的工作既提供了关键资源，也提供了稳健的方法。

    arXiv:2602.02361v2 Announce Type: replace  Abstract: We propose SWE-Universe, a scalable and efficient framework for automatically constructing real-world software engineering (SWE) verifiable environments from GitHub pull requests (PRs). To overcome the prevalent challenges of automatic building, such as low production yield, weak verifiers, and prohibitive cost, our framework utilizes a building agent powered by an efficient custom-trained model. This agent employs iterative self-verification and in-loop hacking detection to ensure the reliable generation of high-fidelity, verifiable tasks. Using this method, we scale the number of real-world multilingual SWE environments to a million scale (807,693). We demonstrate the profound value of our environments through large-scale agentic mid-training and reinforcement learning. Finally, we applied this technique to Qwen3-Max-Thinking and achieved a score of 75.3% on SWE-Bench Verified. Our work provides both a critical resource and a robus
    
[^51]: 基于分层约束的元模型引导模型生成方法

    Metamodel-Guided Model Generation with Layered Constraints

    [https://arxiv.org/abs/2510.25890](https://arxiv.org/abs/2510.25890)

    提出一种元模型引导的分层约束模型生成方法，通过生成时约束层（L1）与生成后验证层（L2）的协同配合，确保LLM生成的工程模型满足结构约束、领域规则和任务要求。

    

    大型语言模型（LLM）使工程建模中的自然语言交互成为可能，但生成的模型可能违反结构约束、领域规则或任务要求。我们提出了一种元模型引导的模型生成方法，该方法协调生成时约束与生成后验证。该方法对元模型信息进行转换，利用其术语引导从规格说明中进行结构化约束提取，并将约束链接到元模型元素，同时在集成约束模型（ICM）中记录其来源。针对每个任务，相关约束被绑定到具体的对象、值和引用上。生成时约束层（L1）限制候选内容；生成后验证层（L2）检查所构建的模型和序列化产物，任务验收检查在整个修复过程中始终保留原始需求。确定性程序负责构建并序列化模型。

    arXiv:2510.25890v4 Announce Type: replace  Abstract: Large language models (LLMs) enable natural-language interaction in engineering modeling, but generated models may violate structural constraints, domain rules, or task requirements. We propose a metamodel-guided model generation method that coordinates generation-time constraints and post-generation validation. The method transforms metamodel information, uses its terminology to guide structured constraint extraction from specifications, and links constraints to metamodel elements while recording their sources in an Integrated Constraint Model (ICM). For each task, relevant constraints are bound to concrete objects, values, and references. The generation-time constraint layer (L1) restricts candidate content. The post-generation validation layer (L2) checks constructed models and serialized artifacts, and task acceptance checks retain the original requirements throughout repair. Deterministic procedures construct and serialize model
    
[^52]: 基于大语言模型的静态可空性错误修复

    LLM-Based Repair of Static Nullability Errors

    [https://arxiv.org/abs/2507.20674](https://arxiv.org/abs/2507.20674)

    NullRepair是一个将大语言模型嵌入结构化工作流的系统，其决策流程基于对200个真实错误的手动分析得出的流程图，并结合静态分析，从而自动且准确地修复Java代码中静态可空性检查残留的错误。

    

    现代Java项目越来越多地采用静态分析工具，通过将空值视为类型属性来防止空指针异常。然而，将此类工具集成到大型现有代码库中仍然是一项重大挑战。虽然注解推断可以自动消除许多错误，但通常会残留一部分错误——通常是真实缺陷和误报的混合——只能通过代码修改来解决。手动处理这些错误既繁琐又容易出错。大语言模型（LLM）为自动化这些修复提供了一条有前景的路径，但简单提示的LLM往往会生成不正确、不符合上下文的编辑。我们提出了NullRepair，这是一个将LLM集成到结构化工作流中的系统，用于解决可空性检查器报告的错误。NullRepair的决策过程遵循一个流程图，该流程图源自对200个真实世界错误的手动分析。它利用静态分析……

    arXiv:2507.20674v3 Announce Type: replace  Abstract: Modern Java projects increasingly adopt static analysis tools that prevent null-pointer exceptions by treating nullness as a type property. However, integrating such tools into large, existing codebases remains a significant challenge. While annotation inference can eliminate many errors automatically, a subset of residual errors $-$ typically a mix of real bugs and false positives $-$ often persists and can only be resolved via code changes. Manually addressing these errors is tedious and error-prone. Large language models (LLMs) offer a promising path toward automating these repairs, but naively prompted LLMs often generate incorrect, contextually inappropriate edits. We present NullRepair, a system that integrates LLMs into a structured workflow for resolving the errors from a nullability checker. NullRepair's decision process follows a flowchart derived from manual analysis of 200 real-world errors. It leverages static analysis t
    

