# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient SWE Agent Benchmarking via Trajectory-Aware Evaluation](https://arxiv.org/abs/2609.01603) | 提出PTA-IRT框架，将历史执行轨迹作为特权信息融合过程与结果信号，在低校准预算下更准确地恢复软件工程智能体的完整基准分数与排名。 |
| [^2] | [Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation](https://arxiv.org/abs/2609.01601) | 该论文提出ACToR，通过识别LLM自回归代码生成过程中容易出错的关键token位置，并自适应地为这些位置检索细粒度的仓库上下文，从而提升仓库级代码生成的功能正确性。 |
| [^3] | [The Data Problem in Software Vulnerability Analysis: Artifacts, Quality, and Consumption](https://arxiv.org/abs/2609.01503) | 该论文提出了一个以数据集为中心的分类体系，从制品类型、数据质量和使用方式三个维度系统审视软件漏洞分析领域的数据问题，并通过深度编码1522篇论文语料库中的111篇锚点论文，系统评估了该领域数据集的研究现状与质量达成情况。 |
| [^4] | [HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?](https://arxiv.org/abs/2609.01437) | 提出HarnessDev基准，将评估单元从任务输出转移到可运行的基础设施，考察大语言模型能否从最小种子出发创建完整的智能体运行框架，并利用下游执行反馈迭代演化该框架以提升基准性能。 |
| [^5] | [What Does an Agentic Software Engineering Benchmark Measure? Profiling Task Demands and Agent Behaviour Beyond What Category Labels Reveal](https://arxiv.org/abs/2609.01271) | 本文提出Spread-Novelty-Centrality（SNC）三轴画像方法来刻画仓库级编码任务的真实需求，发现类别标签是任务需求的不可靠代理指标，且智能体行为轨迹能揭示人工标准答案无法反映的任务需求信息。 |
| [^6] | [Continuous Autonomous Refactoring: A Research Roadmap for AI-Driven Code Quality Maintenance](https://arxiv.org/abs/2609.01236) | 该论文提出将基于大语言模型的代码重构从偶发性的手动工具升级为软件维护中持续运行的自主组成部分，并围绕多目标优化、质量定义与评估、多时间尺度信号整合、架构与设计模式、自主重构信任这五个维度构建了研究路线图。 |
| [^7] | [Athena: Vulnerability-Affected Library Identification via Knowledge Graph Completion](https://arxiv.org/abs/2609.01187) | 提出了首个基于图的方法Athena，将漏洞数据库建模为知识图谱，通过知识图谱补全和链接预测自动识别并补全CVE缺失的受影响软件库信息。 |
| [^8] | [Smart Contracts Claimed Vulnerable by the CVE Database, with Labels and Source Locations](https://arxiv.org/abs/2609.01186) | 该论文发布了CVE-Smart-Contracts数据集，收录截至2026年7月与以太坊智能合约相关的CVE记录，提供漏洞工件、三种分类体系的标签和函数级漏洞位置，其构建流程85%自动化且完全可复现。 |
| [^9] | [Hints Help But Do They Teach? Evaluating Skills Transfer in Code Generation](https://arxiv.org/abs/2609.01106) | 研究发现，提示对失败代码生成的“挽救”效果大多可通过无提示的重复采样复现，且相关与无关提示共享同一激活方向，表明提示更多是引导模型已有能力而非传授新技能。 |
| [^10] | [Fine-Tuning Large Language Models to Classify Pull Request-Issue Alignments: Going Beyond Prompting](https://arxiv.org/abs/2609.01087) | 本研究超越了简单的提示工程方法，通过微调GPT-4o及多种开源大语言模型来自动分类拉取请求与问题之间的对齐关系，并结合可解释性分析揭示了PR-问题各字段对模型预测的影响。 |
| [^11] | [Does Fault Localization Beat a Fresh Attempt? A Placebo-Controlled Study of Test-Guided Code Repair](https://arxiv.org/abs/2609.00854) | 该安慰剂对照研究发现，故障定位在实际场景中很少可用（仅约9%的失败候选可定位），且即便可定位，基于频谱定位的片段填充修复也显著劣于盲目的整体重采样。 |
| [^12] | [Probabilistic Model Checking of Autoregressive Neural Sequence Models](https://arxiv.org/abs/2609.00838) | 本文提出一种基于概率模型检测的验证流程，从自回归神经序列模型的逐token生成中提取DTMC并用PRISM验证PCTL规约，从而给出约束违反概率的认证区间和构造性保守的输入空间覆盖率曲线，并借助CEGAR循环自适应收紧区间。 |
| [^13] | [Replacing Training with Memory: Listwise Selection for Text-to-SQL](https://arxiv.org/abs/2609.00834) | 该论文提出MaP-SQL，一种无需微调的Text-to-SQL列表式选择器，通过从训练数据蒸馏的可复用结构化记忆替代学习选择标准，并利用排名聚合缓解位置偏差，从而以更低成本实现候选查询选择。 |
| [^14] | [Towards a Reliable and Practical Eval Pipeline](https://arxiv.org/abs/2609.00805) | 该论文提出了一种结合评估检查清单创建与学习式聚合的端到端评估流水线，提高了LLM评判者之间的一致性及与人类判断的吻合度，同时提供自一致性、解释和预测不确定性。 |
| [^15] | [Predicting Program Exit Code with LLMs and Programming Language Semantics](https://arxiv.org/abs/2609.00579) | 该论文提出了程序可执行性预测这一新任务，并构建了由有效程序系统性生成无效变换的数据集，以研究大语言模型在判断程序有效性及其违反的形式化语义规则时，究竟是依赖预训练先验知识还是给定的程序语义。 |
| [^16] | [WiseSpec: Requirements-Driven Agents for Code Generation](https://arxiv.org/abs/2609.00568) | 该论文提出WiseSpec框架，借鉴软件需求工程思想，通过自动构建高质量结构化需求并结合基于执行的评估进行迭代优化，从而提升大语言模型在仓库级代码生成中的表现。 |
| [^17] | [Runtime-Independent Persistent Agents: Preserving Identity, Memory, and Code Across Models, Harnesses, and Servers](https://arxiv.org/abs/2609.00546) | 该论文提出一种运行时无关的持久化智能体架构，将身份、持久记忆和版本化代码作为连续性基底，与可替换的模型、执行框架、宿主服务器及交互表面解耦，使得更换这些运行时组件属于智能体迁移而非重新创建。 |
| [^18] | [What Survives the Next Model? Benchmarking LLM-Based Techniques Against Single-Prompts](https://arxiv.org/abs/2609.00468) | 该研究通过对 ICSE 2026 的 35 篇基于 LLM 的技术论文进行基准测试，发现在 37% 到 63% 的论文中，新一代模型仅凭一个自动生成的提示词就能超越一年前提出的复杂工程化工具，表明许多复杂 LLM 技术面临被下一代模型原生能力淘汰的风险。 |
| [^19] | [Audit-First Rollback Semantics for Safety-Critical Deployment Pipelines](https://arxiv.org/abs/2609.00406) | 提出审计优先回滚语义这一容错机制，保证在转换阶段发生故障停机崩溃时，部署运行时的审计链与实时状态在每个已提交的终止状态上保持一致。 |
| [^20] | [Federated Trust for Embodied Robot Capability Marketplaces](https://arxiv.org/abs/2609.00404) | 本文提出用本地信任目录与 Ed25519 分离签名实现的联邦信任机制，取代中心化 PKI，使具身机器人技能市场能够在离线、多监管环境下完成本地化的安全安装验证。 |
| [^21] | [Operation-Type-Aware Client Routing for Leader-Based Consensus Datastores](https://arxiv.org/abs/2609.00392) | 提出一种操作类型感知的客户端路由策略，通过将写操作固定路由至领导者、将读操作分散至健康节点读池，在降低延迟、提升吞吐量的同时还能自动检测并隔离静默劣化的跟随者节点。 |
| [^22] | [Bounded, Indeterminate, or a Bug: A Condition-Aware Oracle for Differential Testing of SQL Aggregates](https://arxiv.org/abs/2609.00381) | 该论文提出一个条件感知的差分测试判定器，以存储双精度浮点数的精确有理值为真值，将聚合结果差异分类为“精确、有界或不确定”，并推导出由算法条件数决定的可测试性边界 kappa* = (1/C_A)^(1/p)，从而能够区分真实缺陷与合理的浮点舍入误差。 |
| [^23] | [Deterministic LLM Inference Across GPU Kernels: Power-of-Two INT8 Quantization Scales and the Limits of Tolerance-Based Conformance](https://arxiv.org/abs/2609.00363) | 本文通过对INT8量化GEMM流水线系统性注入九种故障，证明了基于容差的符合性测试在构造上无法检测仅使输出偏移至多一个bfloat16间距的尾部计算故障，而采用二的幂次量化缩放因子是保障跨GPU核函数确定性推理的关键途径之一。 |
| [^24] | [Revisiting Feedback-Driven LLM Code Repair: A Replication and Exploratory Java Extension](https://arxiv.org/abs/2609.00362) | 该研究部分复现了FeedbackEval基准并探索性地将其扩展到Java语言，发现基于Python得出的反馈驱动代码修复结论可能缺乏跨语言的泛化性。 |
| [^25] | [Exploring Quantum Software Testing Across Research and Practice: Emerging Results from a Multivocal Literature Review](https://arxiv.org/abs/2609.00354) | 本文通过整合学术文献与灰色文献的多声部综述，首次系统刻画了研究界与工业界对量子软件测试的挑战、技术和工具的认知现状，揭示出该领域快速演进但碎片化的生态系统。 |
| [^26] | [Spec-Driven Development for Agentic Software Engineering: Harnessing Human-Agent Teamwork](https://arxiv.org/abs/2609.00252) | 本文提出规范驱动开发（SDD）作为团队规模智能体软件工程的使能学科，并刻画了团队治理自主智能体行为的“缰绳”机制，以应对个体生产力提升但团队吞吐量与稳定性下降的生产力悖论。 |
| [^27] | [Empirical Software Engineering in Practice: Insights from Google](https://arxiv.org/abs/2609.00247) | 本文作为“实践中的ESE”专栏首篇，通过采访谷歌开发者智能团队的从业者，揭示了实证软件工程在工业界的实际实施方式、研究方法选择、成果应用及面临的常见障碍。 |
| [^28] | [Beyond Locks and Thread IDs: Static Data Race Detection Off The Beaten Path (Extended Version)](https://arxiv.org/abs/2609.00246) | 该论文扩展了摘要框架，以支持线程屏障、pthread_once以及祖先线程锁集合等此前被静态数据竞争检测所忽视的并发构造与同步机制，并通过判定测试套件证明现有最先进的工具缺乏对这些特征的支持。 |
| [^29] | [Don't Let the Model Write the YAML: Deterministic, Minimal-Diff GitOps Remediation from LLM-Proposed Field Changes](https://arxiv.org/abs/2609.00227) | 该论文提出让 LLM 只负责做出字段级的语义决策，再由确定性工具将其转换为最小差异的配置修改，从而避免让模型直接生成 YAML 文件或 diff 所带来的静默损坏、不确定性和高开销问题。 |
| [^30] | [Commit-first LLM judging inherits the judge's own errors](https://arxiv.org/abs/2609.00088) | 研究发现“先答后判”式LLM评判会继承评判者自身的错误，而对八个主流评估框架的审计表明无一真正实现该方法，其中九个框架因复制同一祖先提示词而采用了已被证明无效的变体，导致大量错误代码被放行。 |
| [^31] | [Framework and Benchmark for Code-Driven Agentic Testing in Web Development](https://arxiv.org/abs/2609.00081) | 提出了代码驱动智能体测试（CAT）范式，通过CATJudge框架与CATTest基准让智能体编写Playwright代码自主探索Web应用以发现缺陷，实验揭示当前主流视觉语言模型的缺陷发现能力仍然薄弱。 |
| [^32] | [Beneath the Diff: Diagnosing and Mitigating Algorithmic Mode Collapse in Code-Level Autonomous Research Loops](https://arxiv.org/abs/2609.00077) | 论文系统性地诊断出代码级自主研究循环中一种名为“算法模式坍缩”的失效模式——即表层编辑多样性看似稳定但算法层面的语义与机制多样性已经坍缩，并提出了相应的缓解方法。 |
| [^33] | [Can MCP Clients Decide What to Do After Failure? A Result-Only Actionability Audit](https://arxiv.org/abs/2609.00072) | 该论文提出了一种仅基于MCP失败结果的可操作性审计框架，发现类型化字段虽能暴露失败和宽泛策略信息，但缺乏具体原因、目标和可执行修复方法，而自然语言描述虽含更丰富的信息却需要语义解释才能用于恢复。 |
| [^34] | [CUDA-Harness: Harnessing Agentic CUDA Kernel Generation and Optimization from Natural Language](https://arxiv.org/abs/2609.00058) | 该论文提出CUDA-Harness框架，通过智能体式方法直接从自然语言生成并优化高性能CUDA内核，克服了现有工作局限于PyTorch转译以及因依赖预定义测试输入而易受奖励欺骗的不足。 |
| [^35] | [Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness](https://arxiv.org/abs/2609.00050) | 提出了一个智能体云工作流工程框架，通过将图工程（长时程工作流推进）、循环工程（有界诊断与修复重试）和零信任智能体套件（受限执行）三个关注点分离，将自然语言云工程任务自动转化为经过验证的代码仓库和可验证的云部署。 |
| [^36] | [What Is a System? An Interaction-Based Account of Structure-Behavior Coalescence in General Systems Theory](https://arxiv.org/abs/2609.00043) | 本文提出“结构-行为融合”（SBC）观点，主张系统是结构化实体，其行为源于组成实体间的相互作用，且这些相互作用同时构成系统的结构组织与行为实现，从而为“什么是系统”提供了基于相互作用的统一解释。 |
| [^37] | [Structure-Behavior Coalescence and the Limits of Traditional Systems Theory](https://arxiv.org/abs/2609.00042) | 本文提出“结构-行为融合”（SBC）原则，主张结构与行为是同一系统过程中相互构成的两个方面，系统同一性源于二者的持续共同决定，从而克服传统系统论将结构与行为分离处理所带来的解释困难。 |
| [^38] | [trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories](https://arxiv.org/abs/2609.00038) | 仅看最终结果的LLM评判器无法发现智能体“答对但走错路”的问题——在可构造真值的确定性客服工具环境中，仅结果型评判器对静默故障的召回率仅45%且误报33%的正确轨迹，而基于逐步评分标准的评判器可将静默故障召回率提升至77%。 |
| [^39] | [SilentProbe: Measuring Silent Failure in Production APIs Used as Agent Tools](https://arxiv.org/abs/2609.00035) | 该论文首次大规模测量了LLM智能体调用生产API时的“静默失败”现象，发现API模式中机器可校验的约束（而非供应商身份）是预测服务器能否诚实报错的关键因素，而当前OpenAPI文档中这类约束严重缺失。 |
| [^40] | [Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents -- A Source-Code Study of Eleven Systems](https://arxiv.org/abs/2609.00006) | 本文通过对十一套生产级编码智能体 harness 的源代码解剖，定义了 harness 的七大标准子系统，总结出 13 条跨系统观察结论和 29 种常见设计模式，为新兴的 harness 工程学科建立了最全面的实证基础。 |
| [^41] | [RealSWE: A Compositional Evaluation of Coding Agents under Realistic User Requests](https://arxiv.org/abs/2608.27831) | 该论文提出RealSWE基准，通过381个源自SWE-bench的多变体任务族来模拟简短、随意、信息稀疏的真实用户请求，从而更真实地评估编程智能体，并揭示了现有基准与真实用户请求在信息完整度和语言风格上的显著差距。 |
| [^42] | [SWE-bench Science: Can Coding Agents Resolve Engineering Tasks in Science?](https://arxiv.org/abs/2608.19799) | 本文提出了SWE-bench Science，一个针对科学软件工程的仓库级基准，并揭示即使最佳代理在科学任务中成功率也低于50%，主要因科学知识不足等四种机制导致失败。 |
| [^43] | [MetaInfer: A Knowledge Only LLM Inference Engine Generator SKILL Toolbox](https://arxiv.org/abs/2607.12875) | 本文提出MetaInfer，一种利用LLM作为编译器，通过多智能体协作和契约知识库，仅根据用户指定的运行时约束自动生成定制化推理框架的方法，以减少代码复杂性和性能开销。 |
| [^44] | [SEDCoT: Enhancing LLM-Based COBOL Code Translation via Symbolic Execution and Delta Debugging](https://arxiv.org/abs/2607.04092) | SEDCoT是一个COBOL到C的代码翻译框架，它先利用大语言模型进行初始翻译，再结合符号执行生成测试套件迭代修复语义差异，并通过增量调试将失败测试最小化为简洁反例，从而提升翻译正确性。 |
| [^45] | [ChainSWE: Benchmarking Coding Agents on Multi-Bug Software Maintenance](https://arxiv.org/abs/2607.02606) | ChainSWE是首个评估编码智能体在共享代码库中进行顺序性、相互依赖缺陷修复的基准测试，通过收集54个Python项目的304个问题链条，揭示了随着链条长度增加智能体性能最多下降70%的现象。 |
| [^46] | [Steer, Don't Solve: Training Small Critic Models for Large Code Agents](https://arxiv.org/abs/2606.21811) | 通过训练专门负责高层次规划的小型评论模型（4B/8B）在推理时引导大型编码智能体识别并纠正错误，在SWE-Bench Verified上显著提升多个更大规模编码智能体的解决率（最高提升16.0%）并降低推理成本。 |
| [^47] | [Test vs Mutant: Adversarial LLM Agents for Robust Unit Test Generation](https://arxiv.org/abs/2602.08146) | 提出AdverTest框架，通过测试用例生成智能体与变异体生成智能体之间的对抗循环博弈，显著提升LLM生成单元测试在缺陷检测方面的鲁棒性。 |
| [^48] | [Measuring Computer Science Enthusiasm: A Questionnaire-Based Analysis of Age and Gender Effects on Students' Interest](https://arxiv.org/abs/2512.08472) | 本研究开发了一份28条目的问卷来测量学生对计算机科学的热情，发现青少年兴趣在青春期早期显著下降（尤其女生），推翻了早期接触能保证持久兴趣的常见假设。 |
| [^49] | [Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support for Incident Response](https://arxiv.org/abs/2511.15755) | 该论文提出MyAntFarm.ai框架，通过348次受控试验证明多智能体LLM编排相比单智能体方法可将可操作建议率从1.7%提升至100%，实现行动具体性提升80倍、方案正确性提升140倍且质量零方差的确定性事件响应决策支持。 |
| [^50] | [Oops!... I did it again. Analysing and Handling Conclusion (In-)Stability in Socio-Technical Software Engineering](https://arxiv.org/abs/2510.06844) | 本研究通过用四个独立挖掘工具正式复现三项已发表的社会技术软件分析研究，揭示了不同工具流水线在数据、结果与结论上的不稳定性，并为研究者和从业者提供了应对此类效度威胁的可操作建议。 |
| [^51] | [Essence Coach: A Bot for Software Practice Adoption](https://arxiv.org/abs/2508.16445) | 该论文提出 Essence Coach——一个结合大语言模型与检索增强生成（RAG）技术的聊天机器人，可作为教练工具帮助学习者和从业者理解并采纳 Essence 软件工程框架。 |
| [^52] | [ParaStudent: Closing the Sim2Real Gap in User Simulators for AI Tutor Evaluation](https://arxiv.org/abs/2507.12674) | ParaStudent是一个通过微调来模拟初学者编程修改的框架，其模拟结果更贴近真实学生代码分布，可用于AI导师部署前的反馈评估与筛选。 |
| [^53] | [The Popularity Hypothesis in Software Security: A Large-Scale Replication with PHP Packages](https://arxiv.org/abs/2502.16670) | 本文通过对近40万个PHP开源软件包和6000多个WordPress组件的大规模复现分析，验证了软件安全领域的流行度假设，即出现过漏洞报告的软件包通常比漏洞较少或没有漏洞的软件包更流行。 |

# 详细

[^1]: 基于轨迹感知评估的高效软件工程（SWE）智能体基准测试

    Efficient SWE Agent Benchmarking via Trajectory-Aware Evaluation

    [https://arxiv.org/abs/2609.01603](https://arxiv.org/abs/2609.01603)

    提出PTA-IRT框架，将历史执行轨迹作为特权信息融合过程与结果信号，在低校准预算下更准确地恢复软件工程智能体的完整基准分数与排名。

    

    在真实基准上评估软件工程智能体的成本很高，因为每个任务可能需要多步代码探索、修改和测试执行。现有的高效评估方法通过选择代表性子集来估计完整基准的性能，但它们在很大程度上仅依赖结果：它们拟合历史的通过/失败响应矩阵或静态任务语义，丢弃了智能体如何解决问题的信息。我们提出了PTA-IRT，一个融合过程与结果信号的特权轨迹感知项目反应理论框架。历史执行轨迹提供了超越通过/失败的过程级证据，例如探索的上下文、尝试的编辑和解题路径，PTA-IRT将这些作为特权信息用于校准子集选择和能力估计。在低校准预算下，PTA-IRT在四个软件工程基准上的分数和排名恢复方面始终优于现有的IRT基线方法。代码和数据均已公开。

    arXiv:2609.01603v1 Announce Type: cross  Abstract: Evaluating software engineering agents on realistic benchmarks is costly, since each task may require multi-step code exploration, modification, and test execution. Existing efficient evaluation methods select representative subsets to estimate full-benchmark performance, but are largely result-only: they fit historical pass/fail response matrices or static task semantics, discarding how agents solve problems. We propose PTA-IRT, a Privileged Trajectory-Aware Item Response Theory framework that fuses process and outcome signals. Historical execution trajectories supply process-level evidence beyond pass/fail, such as explored context, attempted edits, and solving paths, which PTA-IRT uses as privileged information for calibration subset selection and ability estimation. Under low calibration budgets, PTA-IRT consistently outperforms prior IRT baselines on score and ranking recovery across four SWE benchmarks. Code and data are publicly
    
[^2]: 面向仓库级代码生成的自适应关键Token感知检索

    Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation

    [https://arxiv.org/abs/2609.01601](https://arxiv.org/abs/2609.01601)

    该论文提出ACToR，通过识别LLM自回归代码生成过程中容易出错的关键token位置，并自适应地为这些位置检索细粒度的仓库上下文，从而提升仓库级代码生成的功能正确性。

    

    仓库级代码生成任务需要合成既满足任务要求、又与目标仓库上下文保持一致的代码。由于真实世界的仓库往往超出大语言模型（LLM）的输入长度限制，现有方法通常采用检索增强生成（RAG）来提供仓库特定的上下文。尽管这些方法改善了仓库上下文的检索，但它们通常将上下文作为任务级支持来提供，而没有显式识别生成过程中需要细粒度仓库上下文的关键token。在LLM的自回归生成过程中，错误往往集中在少数决定性位置上：一旦这些token被错误生成，后续代码可能沿着错误的语义路径发展，最终导致功能失效。我们将这些位置称为“关键token”。在本文中，我们提出了ACToR，一种自适应关键（token感知检索框架，摘要在此处截断）

    arXiv:2609.01601v1 Announce Type: cross  Abstract: The repository-level code generation task requires synthesizing code that satisfies task requirements while remaining consistent with the target repository context. Since real-world repositories often exceed the input length limits of LLMs, existing approaches commonly adopt retrieval-augmented generation (RAG) to provide repository-specific context. Despite improving repository-context retrieval, existing methods typically provide context as task-level support, without explicitly identifying the critical tokens that require fine-grained repository context during generation. During the autoregressive generation process of LLMs, errors often concentrate at a small number of decisive positions: once such tokens are generated incorrectly, subsequent code may follow an incorrect semantic path and eventually lead to functional failure. We refer to these positions as "critical tokens". In this paper, we propose ACToR, an adaptive critical to
    
[^3]: 软件漏洞分析中的数据问题：制品、质量与使用

    The Data Problem in Software Vulnerability Analysis: Artifacts, Quality, and Consumption

    [https://arxiv.org/abs/2609.01503](https://arxiv.org/abs/2609.01503)

    该论文提出了一个以数据集为中心的分类体系，从制品类型、数据质量和使用方式三个维度系统审视软件漏洞分析领域的数据问题，并通过深度编码1522篇论文语料库中的111篇锚点论文，系统评估了该领域数据集的研究现状与质量达成情况。

    

    基于机器学习和大型语言模型（LLM）的软件漏洞分析，其可信度完全取决于其训练和评估所使用的数据，然而这些数据却很少被当作一等对象加以审视。我们通过一个以数据集为中心的分类体系来研究漏洞分析背后的数据，该体系将数据区分为三个维度：制品是什么（代码、元数据、补丁、测试/PoC、推理、轨迹）、制品质量如何（真实性、标签证据、规模、多样性、数据泄漏、可获得性）以及制品的用途。从系统性收集的涵盖2016至2026年的1522篇论文语料库以及更早的基础性文献中，我们对分层的111篇锚点论文进行了深度编码，为每一个肯定性的评分标准取值提供了逐字文本片段作为支撑，并针对每个属性报告了两方面情况：该属性被研究的程度，以及数据集在该属性上的达成程度。结果呈现出一条证据阶梯：可执行制品是唯一一个主要制品类型，其中24个数据集中有15个既被评为真实世界数据又带有相应标签……（原文摘要在此处截断）

    arXiv:2609.01503v1 Announce Type: new  Abstract: Learning- and LLM-based software vulnerability analysis is only as trustworthy as the data it is trained and evaluated on, yet that data is rarely examined as a first-class object. We investigate the data behind vulnerability analysis through a dataset-centric taxonomy that separates what an artifact is (code, metadata, patches, tests/PoCs, reasoning, traces), how good it is (realism, label evidence, scale, diversity, leakage, availability), and what it is used for. From a systematically assembled corpus of 1522 papers covering 2016-2026 plus foundational earlier work we deep-code a tiered set of 111 anchor papers, backing every affirmative rubric-graded value with a verbatim span, and we report, per attribute, both how much it has been studied and how well datasets achieve it. The results trace an evidence ladder: executable artifacts are the only major type where 15 of the 24 datasets are both graded real-world and carry labels that re
    
[^4]: HarnessDev：大语言模型能否创造并演化自己的智能体运行框架（Agent Harness）？

    HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?

    [https://arxiv.org/abs/2609.01437](https://arxiv.org/abs/2609.01437)

    提出HarnessDev基准，将评估单元从任务输出转移到可运行的基础设施，考察大语言模型能否从最小种子出发创建完整的智能体运行框架，并利用下游执行反馈迭代演化该框架以提升基准性能。

    

    随着智能体从研究原型走向实际部署的工具，其能力越来越依赖于模型外部的执行基础设施，通常被称为智能体运行框架（agent harness）。在保持模型权重不变的情况下改变这一运行框架，可能会显著改变任务表现。当前的智能体评估通常只报告在选定运行框架下的下游性能，而模型自身开发运行框架的能力相对而言尚未得到充分探索。我们提出了 HarnessDev，这是一个将评估单元从任务输出转移到可运行基础设施的基准。HarnessDev 包含两个阶段：在“创造”阶段，智能体从一个最小化的种子和少量用例出发，构建一个完整的执行系统；在“演化”阶段，智能体从其自建的运行框架出发，利用下游执行反馈对其进行迭代修改，目标是提升基准性能。随后，我们在能力（任务成功率等指标）上评估每个构建出的运行框架（摘要在此处被截断）。

    arXiv:2609.01437v1 Announce Type: cross  Abstract: As agents move from research prototypes to deployed tools, their capability increasingly depends on model-external execution infrastructure, commonly termed the agent harness. Changing this harness while holding model weights fixed can substantially alter task performance. Current agent evaluations typically report downstream performance under a chosen harness, leaving a model's ability to develop the harness itself comparatively underexplored. We introduce HarnessDev, a benchmark that shifts the unit of evaluation from task outputs to runnable infrastructure. HarnessDev covers two stages. In Creation, the agent starts from a minimal seed and a small number of cases, then builds a complete execution system. In Evolution, it starts from its own created harness and iteratively revises it using downstream execution feedback, with the goal of improving benchmark performance. We then evaluate each constructed harness on capability (task suc
    
[^5]: 智能体软件工程基准测试究竟在衡量什么？超越类别标签的任务需求与智能体行为画像

    What Does an Agentic Software Engineering Benchmark Measure? Profiling Task Demands and Agent Behaviour Beyond What Category Labels Reveal

    [https://arxiv.org/abs/2609.01271](https://arxiv.org/abs/2609.01271)

    本文提出Spread-Novelty-Centrality（SNC）三轴画像方法来刻画仓库级编码任务的真实需求，发现类别标签是任务需求的不可靠代理指标，且智能体行为轨迹能揭示人工标准答案无法反映的任务需求信息。

    

    智能体软件工程基准测试通常通过名义类别标签（如“缺陷修复”或“功能实现”）来概括，然而携带相同标签的基准测试是通过截然不同的构建流程创建的。因此，标签几乎无法揭示基准测试所要求的工程工作。我们引入了 Spread--Novelty--Centrality（SNC）画像，这是一种基于实证软件工程研究的三轴特征描述方法，用于刻画仓库级编码任务的需求。我们将该画像应用于五个广泛使用的基准测试，以及两个模型家族在三种规模下的14,922条轨迹，并报告了三项发现。（1）类别标签是任务需求的不可靠代理指标，因为每一对基准测试在至少两个SNC轴上均存在统计学上的显著分离，且这些分离可以追溯到具体的构建决策。（2）智能体行为揭示了人类编写的标准答案（gold solution）无法揭示的需求。智能体产出的解决方案……

    arXiv:2609.01271v1 Announce Type: cross  Abstract: Agentic software engineering benchmarks are typically summarized by nominal category labels such as "bug fix" or "feature implementation," yet benchmarks carrying the same label are built through very different curation pipelines. A label thus reveals little about the engineering work a benchmark demands. We introduce the Spread--Novelty--Centrality (SNC) profile, a three-axis characterization of the demands of repository-level coding tasks, grounded in empirical software engineering research. We apply the profile to five widely used benchmarks and 14,922 trajectories of two model families at three scales, and report three findings. (1) A label is an unreliable proxy for task demands, as every pair of benchmarks is statistically separated on at least two SNC axes, and the separations trace back to specific curation decisions. (2) Agent behaviour reveals demands that the human-written gold solution cannot. Agents produce larger solution
    
[^6]: 持续自主重构：AI驱动的代码质量维护研究路线图

    Continuous Autonomous Refactoring: A Research Roadmap for AI-Driven Code Quality Maintenance

    [https://arxiv.org/abs/2609.01236](https://arxiv.org/abs/2609.01236)

    该论文提出将基于大语言模型的代码重构从偶发性的手动工具升级为软件维护中持续运行的自主组成部分，并围绕多目标优化、质量定义与评估、多时间尺度信号整合、架构与设计模式、自主重构信任这五个维度构建了研究路线图。

    

    大语言模型在代码重构方面已展现出可观的能力，但现有方法仍局限于方法级别的任务。在本文中，我们设想将基于大语言模型的重构作为软件维护的一个持续性组成部分，而非仅在偶发的手动重构中被调用的工具。在这一愿景下，AI智能体持续地监控、评估并改进代码库，以符合明确且不断演进的软件质量标准。我们提出了一个围绕五个维度构建的路线图：多目标优化问题、质量定义与评估、异构信号的多时间尺度整合、架构与设计模式，以及对自主重构的信任。我们进一步将持续交付流水线的集成与成本考量确定为横切关注点。针对每个维度，我们分析了其中的深层挑战并提出开放性研究问题。这些维度共同定义了一个研究议程。

    arXiv:2609.01236v1 Announce Type: new  Abstract: Large language models have shown promising capabilities in code refactoring, but existing approaches remain limited to method-level tasks. In this paper, we envision LLM-based refactoring as a continuous component of software maintenance rather than a tool invoked only for occasional manual refactoring. Under this vision, AI agents continuously monitor, evaluate, and improve codebases against explicit and evolving notions of software quality. We present a roadmap organized around five dimensions: the multi-objective optimization problem, quality definition and evaluation, multi-timescale integration of heterogeneous signals, architecture and design pattern, and trust in autonomous refactoring. We further identify integration into continuous delivery pipelines and cost considerations as cross-cutting concerns. For each dimension, we analyze the underlying challenges and pose open research questions. These dimensions define a research agen
    
[^7]: Athena：基于知识图谱补全的漏洞受影响软件库识别

    Athena: Vulnerability-Affected Library Identification via Knowledge Graph Completion

    [https://arxiv.org/abs/2609.01187](https://arxiv.org/abs/2609.01187)

    提出了首个基于图的方法Athena，将漏洞数据库建模为知识图谱，通过知识图谱补全和链接预测自动识别并补全CVE缺失的受影响软件库信息。

    

    一个被广泛使用的软件库中存在的单个漏洞可能会级联影响到数百万个依赖它的应用程序，然而超过一半的漏洞数据库条目包含缺失或不正确的受影响库信息。现有的自动化方法忽略了漏洞数据库的关系结构，将识别任务视为孤立的文本检索问题。在本文中，我们提出了Athena，这是首个基于图方法的漏洞受影响库识别方法。Athena将漏洞数据库建模为知识图谱，并将识别问题重新表述为知识图谱补全（KGC）问题。它包含三个关键模块：建模模块，构建一个集成了CVE、软件库、CWE弱点类型、CPE产品和软件生态系统的安全知识图谱；补全模块，应用模块化的KGC主干网络，通过链接预测来预测给定CVE所缺失的受影响库；以及重排序模块……

    arXiv:2609.01187v1 Announce Type: cross  Abstract: A single vulnerability in a widely used library can cascade through millions of dependent applications, yet more than half of vulnerability database entries contain missing or incorrect affected-library information. Existing automated approaches neglect the relational structure of vulnerability databases, treating identification as an isolated text retrieval problem. In this paper, we propose Athena, the first graph-based approach for vulnerability affected library identification. Athena models vulnerability databases as a knowledge graph and reformulates the identification problem as knowledge graph completion (KGC). It comprises three key modules: a Modeling module that constructs a security knowledge graph integrating CVEs, libraries, CWE weakness types, CPE products, and software ecosystems; a Completion module that applies a modular KGC backbone to predict missing affected libraries for a given CVE via link prediction; and a Re-ra
    
[^8]: CVE数据库中声称存在漏洞的智能合约：含标签与源码位置

    Smart Contracts Claimed Vulnerable by the CVE Database, with Labels and Source Locations

    [https://arxiv.org/abs/2609.01186](https://arxiv.org/abs/2609.01186)

    该论文发布了CVE-Smart-Contracts数据集，收录截至2026年7月与以太坊智能合约相关的CVE记录，提供漏洞工件、三种分类体系的标签和函数级漏洞位置，其构建流程85%自动化且完全可复现。

    

    通用漏洞与暴露数据库收录了硬件和软件中的漏洞声明，其中包括与区块链程序（即智能合约）相关的漏洞声明。我们提出了CVE-Smart-Contracts，这是一个精心整理的数据集，收录了截至2026年7月涉及以太坊智能合约的CVE记录。该数据集包含存在漏洞的工件（源代码和运行时字节码）、基于三种分类体系的标签，以及函数级别的漏洞位置。CVE记录的检索、额外证据的收集、记录与工件之间对应关系的验证、标签分配以及漏洞定位均由自动化流程完成，仅有15%需要人工分析。该数据集并不验证原始漏洞声明的真实性，但会将少量明显错误的记录标记为“已驳斥”。为保证可复现性，所有外部输入均被保留，因此重新运行整个流水线可以得到相同的输出。该数据集包含4……

    arXiv:2609.01186v1 Announce Type: cross  Abstract: The Common Vulnerabilities and Exposures (CVE) database catalogs vulnerability claims in hard- and software, among them those pertaining to blockchain programs a.k.a. smart contracts. We present CVE-Smart-Contracts, a curated dataset of CVE records up to July 2026 referring to Ethereum smart contracts. The dataset contains the vulnerable artifacts (source code and runtime bytecode), labels according to three taxonomies, and function-level locations. The retrieval of CVE records, collection of additional evidence, validation of the correspondence between records and artifacts, label assignment, and vulnerability localization are automated, leaving 15% to manual analysis. The dataset does not validate the original vulnerability claims, but marks a few records obviously wrong as `refuted'. For the sake of reproducibility, all external inputs are retained, so that rerunning the pipelines results in the same outputs. The dataset comprises 4
    
[^9]: 提示有帮助，但它们真的能“教会”模型吗？评估代码生成中的技能迁移

    Hints Help But Do They Teach? Evaluating Skills Transfer in Code Generation

    [https://arxiv.org/abs/2609.01106](https://arxiv.org/abs/2609.01106)

    研究发现，提示对失败代码生成的“挽救”效果大多可通过无提示的重复采样复现，且相关与无关提示共享同一激活方向，表明提示更多是引导模型已有能力而非传授新技能。

    

    当一条提示能把一个失败的生成程序变成通过测试的程序时，它究竟提供了缺失的信息，还是仅仅将模型引导至它本来就能得出的解？我们在 HumanEval+ 和 MBPP+ 上通过可执行评估来检验这些假设。对于 Qwen2.5-3B-Instruct，自适应的相关提示挽救了 79 个选定失败样例中的 36 个；无关提示挽救了 19 个；而在无提示条件下，8 次采样解决了 46 个样例，并覆盖了 36 个相关提示挽救中的 31 个。Phi-3.5-mini 呈现出相同的模式：相关提示挽救了 101 个失败中的 42 个，无关提示挽救了 17 个，无提示采样解决了 57 个，其中包括 42 个相关提示挽救中的 36 个。由于各提示条件使用了不同的尝试预算，这些比较并不能分离出纯粹的语义效应。在 Qwen 上进行的机制性测试发现，相关提示与无关提示共享一个稳定的激活方向。持续向该方向添加偏移会产生 14 次挽救和 18 次回退，且未检测到……（原文摘要在此处被截断）

    arXiv:2609.01106v1 Announce Type: cross  Abstract: When a hint turns a failing generated program into a passing one, does it provide missing information or merely steer the model toward a solution it could already produce? We test these hypotheses on HumanEval+ and MBPP+ using executable evaluation. For Qwen2.5-3B-Instruct, adaptive relevant hints rescue 36 of 79 selected failures; an unrelated hint rescues 19, while eight unhinted samples solve 46 and recover 31 of the 36 relevant-hint rescues. Phi-3.5-mini shows the same pattern: relevant hints rescue 42 of 101 failures, an unrelated hint rescues 17, and unhinted sampling solves 57, including 36 of the 42 relevant-hint rescues. Because the hint conditions use different attempt budgets, these comparisons do not isolate a purely semantic effect. Mechanistic tests on Qwen identify a stable activation direction shared by relevant and unrelated hints. Persistently adding this direction yields 14 rescues and 18 regressions, with no detecta
    
[^10]: 微调大语言模型以分类拉取请求与问题的对齐关系：超越提示工程

    Fine-Tuning Large Language Models to Classify Pull Request-Issue Alignments: Going Beyond Prompting

    [https://arxiv.org/abs/2609.01087](https://arxiv.org/abs/2609.01087)

    本研究超越了简单的提示工程方法，通过微调GPT-4o及多种开源大语言模型来自动分类拉取请求与问题之间的对齐关系，并结合可解释性分析揭示了PR-问题各字段对模型预测的影响。

    

    背景：拉取请求（PR）与对应问题（Issue）之间的准确对齐对于高效的软件开发和维护代码质量至关重要，因为错位对齐会降低可追溯性、阻碍缺陷定位并降低可维护性。目标：本研究旨在通过利用经过微调的大语言模型（LLM）来改进跨多个对齐类别的PR-问题对齐自动化分类，并进行可解释性分析以探究PR-问题各字段对微调后LLM预测结果的影响。方法：我们的方法论包括数据集准备、LLM微调和可解释性分析三个部分。我们首先扩展了现有数据集并应用数据增强来解决类别不平衡问题。随后通过指令微调对GPT-4o进行微调，并使用分类任务对开源大语言模型（包括CodeLlama-7B、CodeQwen1.5-7B、StableCode-3B、CodeGemma-7B和Deepseek-Coder-6.7B）进行微调……

    arXiv:2609.01087v1 Announce Type: new  Abstract: Context: Accurate alignment between pull requests (PRs) and corresponding issues is crucial for efficient software development and maintaining code quality, as misalignments can reduce traceability, hinder defect localization, and decrease maintainability.   Objective: This study aims to improve automated PR-issue alignment classification by leveraging fine-tuned large language models (LLMs) across multiple alignment categories, and conducts interpretability analysis to investigate the effects of PR-issue fields on the predictions of fine-tuned LLMs.   Method: Our methodology consists of dataset preparation, LLM fine-tuning, and interpretability analysis. We first extended an existing dataset and applied data augmentation to address class imbalance. GPT-4o was then fine-tuned via instruction tuning, and open-source LLMs including CodeLlama-7B, CodeQwen1.5-7B, StableCode-3B, CodeGemma-7B, and Deepseek-Coder-6.7B were fine-tuned using clas
    
[^11]: 故障定位能否胜过重新尝试？——一项关于测试引导代码修复的安慰剂对照研究

    Does Fault Localization Beat a Fresh Attempt? A Placebo-Controlled Study of Test-Guided Code Repair

    [https://arxiv.org/abs/2609.00854](https://arxiv.org/abs/2609.00854)

    该安慰剂对照研究发现，故障定位在实际场景中很少可用（仅约9%的失败候选可定位），且即便可定位，基于频谱定位的片段填充修复也显著劣于盲目的整体重采样。

    

    故障定位可以将代码模型的修复聚焦于失败测试所涉及的语句，但针对性的编辑可能仅仅因为改动较小而成功，而第二次模型调用也可能根本未利用失败信息就取得成功。我们通过对同一失败候选程序施加三种处理来区分这些解释：盲目的整方案重采样、基于频谱的定位后对可疑片段进行填充，以及在互不重叠的随机代码片段上进行等长填充（安慰剂对照）。在三个冻结的26-32B模型、三个基准数据集和488个失败候选上，外加一个单独声明的来自第三方家族的24B第四个模型，得到三个结果。首先，故障定位很少可用：只有9.0%的失败候选存在暴露失败的可公开测试及可用的频谱。其次，在来自强测试套件的177个可定位候选中，在匹配的尝试次数下，定位填充决定性地输给了盲重采样（3:40，p = 3.0 × 10^-9），这与我们的（摘要在此处截断）

    arXiv:2609.00854v1 Announce Type: cross  Abstract: Fault localization can focus a code model's repair on the statements a failing test implicates, but a targeted edit may succeed merely because it is small, and a second model call may succeed without using the failure at all. We separate these explanations with three arms applied to the same failed candidate: blind whole-solution resampling, spectrum-based localization followed by suspect-span infilling, and same-length infilling at a disjoint random code span. Across three frozen 26-32B models, three benchmarks and 488 failing candidates, plus a separately declared 24B fourth model from a third family, three results follow. First, localization is rarely available: only 9.0% of failing candidates expose a failing public test with a usable spectrum. Second, among the 177 candidates localizable from a strong suite, localized infilling loses decisively to blind resampling at a matched attempt count (3:40, p = 3.0 x 10^-9), opposite to our
    
[^12]: 自回归神经序列模型的概率模型检测

    Probabilistic Model Checking of Autoregressive Neural Sequence Models

    [https://arxiv.org/abs/2609.00838](https://arxiv.org/abs/2609.00838)

    本文提出一种基于概率模型检测的验证流程，从自回归神经序列模型的逐token生成中提取DTMC并用PRISM验证PCTL规约，从而给出约束违反概率的认证区间和构造性保守的输入空间覆盖率曲线，并借助CEGAR循环自适应收紧区间。

    

    测试集准确率对于部署自回归神经序列模型时两个重要问题是沉默的：被测系统（SUT）在采样可达的情况下对违反约束的备选方案分配了多少概率质量，以及输入总体中有多大比例满足领域要求。我们用概率模型检测来回答这两个问题。该流程从被测系统逐token的生成过程中提取离散时间马尔可夫链（DTMC），使用PRISM模型检测器验证形式化的PCTL规约，并将每个输入的判定结果聚合为输入空间上的覆盖率曲线。一个可靠性定理确立了该DTMC作为下近似，因此每个判定都能给出被测系统真实可达概率的认证区间。由这些判定构建的覆盖率因此是构造性保守的。反例引导的抽象细化（CEGAR）循环可自适应地收紧区间。

    arXiv:2609.00838v1 Announce Type: cross  Abstract: Test-set accuracy is silent on two issues that matter when deploying autoregressive neural sequence models: how much probability mass the system under test (SUT) places on constraint-violating alternatives that are reachable under sampling and what fraction of the input population satisfies a domain requirement. We answer both with probabilistic model checking. The pipeline extracts a discrete-time Markov chain (DTMC) from the SUT's token-by-token generation, verifies formal PCTL specifications with the PRISM model checker, and aggregates the per-input verdicts into a coverage curve over the input space. A soundness theorem establishes the DTMC as an under-approximation, so every verdict yields a certified interval on the SUT's true reachability probability. The coverage built from those verdicts is, therefore, conservative by construction. A counterexample-guided abstraction refinement (CEGAR) loop adaptively tightens the interval, an
    
[^13]: 用记忆取代训练：面向Text-to-SQL的列表式选择方法

    Replacing Training with Memory: Listwise Selection for Text-to-SQL

    [https://arxiv.org/abs/2609.00834](https://arxiv.org/abs/2609.00834)

    该论文提出MaP-SQL，一种无需微调的Text-to-SQL列表式选择器，通过从训练数据蒸馏的可复用结构化记忆替代学习选择标准，并利用排名聚合缓解位置偏差，从而以更低成本实现候选查询选择。

    

    现代Text-to-SQL系统通常遵循“生成-执行-选择”的流程，即先生成多个候选查询，再从中选出最优的一个。列表式选择通过联合比较多个候选查询，已被广泛采用，但微调列表式选择器的成本高昂。因此，我们提出了一种无需微调的列表式选择器。我们用推理时的策略取代了两个主要的微调目标：（1）将选择标准的学习视为排序学习；（2）缓解位置偏差。首先，我们构建可复用的结构化记忆，而不是将选择行为学习为模型参数。给定一个问题，MaP-SQL会检索从训练数据中蒸馏出的记忆，这些记忆编码了自然语言如何映射到模式元素、SQL操作以及预期输出。这些记忆作为显式的决策标准，用于以列表方式评估候选查询。其次，为了缓解列表式选择器的排序偏差，我们对多个排序结果进行排名聚合（摘要在此处被截断）。

    arXiv:2609.00834v1 Announce Type: cross  Abstract: Modern Text-to-SQL systems often follow generate-execute-select pipelines, generating multiple candidate queries then selecting the best one. Listwise selection, by jointly comparing multiple candidates, has been widely adopted, but fine-tuning listwise selectors is costly. We thus propose a fine-tuning-free listwise selector. We replace two major fine-tuning objectives with inference-time strategies: (1) learning selection criteria as ordering and (2) mitigating positional bias. First, we build reusable structured memories instead of learning selection behavior as model parameters. Given a question, MaP-SQL retrieves memories distilled from training data that encode how natural language maps to schema elements, SQL operations, and expected outputs. These memories serve as explicit decision criteria for evaluating candidates in a listwise manner. Second, to mitigate ordering bias of listwise selectors, we aggregate rankings across mult
    
[^14]: 迈向可靠且实用的评估流水线

    Towards a Reliable and Practical Eval Pipeline

    [https://arxiv.org/abs/2609.00805](https://arxiv.org/abs/2609.00805)

    该论文提出了一种结合评估检查清单创建与学习式聚合的端到端评估流水线，提高了LLM评判者之间的一致性及与人类判断的吻合度，同时提供自一致性、解释和预测不确定性。

    

    基于大语言模型（LLM）的软件系统日益需要在开发生命周期中将有效的“评估”作为质量门控。然而，现有工作通常仅解决评估可靠性的个别方面，而非全部实际需求。我们提出了一种端到端的评估流水线，将评估检查清单的创建与针对检查清单响应的学习式聚合相结合，以提高LLM评判者之间的一致性以及与人类判断相比的准确性。该框架还提供了自一致性、解释和预测不确定性，并通过实验证明了其有效性。

    arXiv:2609.00805v1 Announce Type: new  Abstract: LLM-based software systems increasingly require effective "evals" as quality gates in the development lifecycle. However, existing work typically addresses individual aspects of eval reliability rather than the full set of practical requirements. We present an end-to-end eval pipeline that combines eval checklist creation, with learned aggregation for checklist responses, to improve agreement across LLM judges and accuracy against human judgments. The framework additionally pro- vides self-consistency, explanations, and prediction uncertainty, and we empirically demonstrate its effectiveness.
    
[^15]: 基于大语言模型与程序设计语言语义的程序退出码预测

    Predicting Program Exit Code with LLMs and Programming Language Semantics

    [https://arxiv.org/abs/2609.00579](https://arxiv.org/abs/2609.00579)

    该论文提出了程序可执行性预测这一新任务，并构建了由有效程序系统性生成无效变换的数据集，以研究大语言模型在判断程序有效性及其违反的形式化语义规则时，究竟是依赖预训练先验知识还是给定的程序语义。

    

    大语言模型（LLM）在代码生成和翻译等多种软件工程任务中已展现出卓越能力。然而，其性能的一个关键局限可能在于对程序设计语言语义的理解（或缺乏理解）。即使给出了显式语义，LLM究竟是应用这些规则，还是依赖预训练期间学到的先验知识，目前仍不清楚。我们通过一项新颖任务——程序可执行性预测来研究LLM是依赖先验知识还是给定语义。该任务要求模型在给定程序语法和操作语义的情况下，预测程序在语义上是有效的还是无效的（如果是无效的，还需指出其违反了哪条形式化规则）。由于PrEx需要有效和无效的程序，我们构建了一个数据集，其中包含从有效程序系统性生成的无效变换。我们在两种语义形式体系和两种语义偏移下，跨Human-（评估开源代码LLM）。

    arXiv:2609.00579v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown proficiency in various software engineering tasks, such as code generation and translation. However, a key limitation in their performance may be their (lack of) understanding of programming-language semantics. Even when explicit semantics are given, it remains unclear whether LLMs apply those rules or lean on priors learned during pre-training instead. We study if LLMs lean on priors or given semantics with a novel task--Program Executability Prediction (PrEx)--that asks models to predict whether a program is semantically valid or invalid (and, if invalid, which formal rule it violates) given the program's syntax and operational semantics. Because PrEx requires both valid and invalid programs, we build a dataset with systematically generated invalid transformations derived from valid programs. We evaluate open-source coding LLMs under two semantic formalisms and two semantic shifts across Human-
    
[^16]: WiseSpec：面向代码生成的需求驱动智能体

    WiseSpec: Requirements-Driven Agents for Code Generation

    [https://arxiv.org/abs/2609.00568](https://arxiv.org/abs/2609.00568)

    该论文提出WiseSpec框架，借鉴软件需求工程思想，通过自动构建高质量结构化需求并结合基于执行的评估进行迭代优化，从而提升大语言模型在仓库级代码生成中的表现。

    

    代码生成旨在根据任务需求自动生成源代码，随着大语言模型（LLM）的快速发展而受到广泛关注。尽管取得了显著进展，但由于任务描述往往不完整、含糊不清或缺乏关键的上下文信息，大语言模型在处理复杂软件工程任务时常常难以生成正确的代码。现有方法主要通过更复杂的工具、技能和工作流程来提升编码智能体的能力，却在很大程度上忽视了任务需求本身的质量。为解决这一局限，我们从软件需求工程中汲取灵感，提出了WiseSpec——一个面向仓库级代码生成的新型需求驱动智能体框架。WiseSpec能够自动构建结构化且信息丰富的需求，通过基于执行的评估来衡量需求质量，并进行迭代式（原文摘要至此被截断）。

    arXiv:2609.00568v1 Announce Type: cross  Abstract: Code generation aims to automatically generate source code from task requirements and has attracted significant attention with the rapid advancement of large language models (LLMs). Despite remarkable progress, LLMs often struggle to generate correct code for complex software engineering tasks because task descriptions are frequently incomplete, ambiguous, or lack critical contextual information. Existing approaches primarily improve the capabilities of coding agents through more sophisticated tools, skills, and workflows, while largely overlooking the quality of the task requirements themselves. To address this limitation, we draw inspiration from software requirements engineering and propose WiseSpec, a novel requirements-driven agent framework for repository-level code generation. WiseSpec automatically constructs structured and information-rich requirements, assesses their quality through execution-based evaluation, and iteratively
    
[^17]: 运行时无关的持久化智能体：跨模型、执行框架与服务器保留身份、记忆与代码

    Runtime-Independent Persistent Agents: Preserving Identity, Memory, and Code Across Models, Harnesses, and Servers

    [https://arxiv.org/abs/2609.00546](https://arxiv.org/abs/2609.00546)

    该论文提出一种运行时无关的持久化智能体架构，将身份、持久记忆和版本化代码作为连续性基底，与可替换的模型、执行框架、宿主服务器及交互表面解耦，使得更换这些运行时组件属于智能体迁移而非重新创建。

    

    智能体系统通常由当前产生其行为的模型和执行框架来定义。这种界定方式对单次执行是有用的，但对于一个长期存续的智能体而言则不够充分——这样的智能体可能会更换模型、编排框架、交互会话和宿主服务器，同时仍保持同一身份、记忆和可执行代码谱系。我们提出了一种面向持久化智能体的运行时无关架构。一个承载连续性的基底 P_t=(I_t,M_t,B_t) 包含架构化的身份表示、私有持久记忆和版本化的软件体。一个可替换的部署绑定由执行基底 E_t=(R_t,H_t,D_t)（提供推理器、执行框架和宿主）以及一组交互表面 S_t（如聊天、API 或用户界面绑定）组成。一次已部署的执行表示为 A_t=P_t▷(E_t,S_t)；当授权协议保留至少……（摘要在此处被截断）

    arXiv:2609.00546v1 Announce Type: cross  Abstract: Agent systems are commonly described by the model and harness that currently produce their behavior. That boundary is useful for one execution but underspecifies a long-lived agent that may change models, orchestration harnesses, interaction sessions, and host servers while retaining one identity, memory, and executable code lineage. We present a runtime-independent architecture for persistent agents. A continuity-bearing substrate $P_t=(I_t,M_t,B_t)$ contains an architectural identity representation, private durable memory, and a versioned software body. A replaceable deployment binding comprises an execution substrate $E_t=(R_t,H_t,D_t)$, which supplies a reasoner, harness, and host, and a set of interaction surfaces $S_t$, such as chat, API, or user interface bindings. A deployed execution is $A_t=P_t\triangleright(E_t,S_t)$; changing either replaceable layer is migration, not agent creation, when an authorized protocol preserves at
    
[^18]: 什么能在下一代模型中幸存？将基于大语言模型的技术与单一提示词进行基准对比

    What Survives the Next Model? Benchmarking LLM-Based Techniques Against Single-Prompts

    [https://arxiv.org/abs/2609.00468](https://arxiv.org/abs/2609.00468)

    该研究通过对 ICSE 2026 的 35 篇基于 LLM 的技术论文进行基准测试，发现在 37% 到 63% 的论文中，新一代模型仅凭一个自动生成的提示词就能超越一年前提出的复杂工程化工具，表明许多复杂 LLM 技术面临被下一代模型原生能力淘汰的风险。

    

    软件工程研究界已经热情地拥抱将大语言模型（LLM）集成到复杂技术中，以解决各种各样的任务。然而，这种投入在多大程度上具有战略意义仍不清楚，因为新一代前沿模型的原生能力可能会迅速使现有技术过时。为了评估这项研究投入，我们分析了来自 ICSE 2026 的 35 篇基于 LLM 的技术论文。我们评估了这些复杂工具是否会被最简单的替代方案所超越：即在更新一代的模型上执行单个自动生成的提示词，且无需任何迭代改进。我们发现，在 37% 到 63% 的论文中，使用单个提示词的更新一代模型在原生状态下就能超越仅仅一年前提出的经过大量工程化设计的工具。我们发现，诸如代码生成或修复等构造性技术更容易被单...（原文摘要到此截断）

    arXiv:2609.00468v1 Announce Type: new  Abstract: The software engineering research community has enthusiastically embraced the integration of Large Language Models (LLMs) into complex techniques to solve a wide variety of tasks. However, the extent to which this investment is strategic remains unclear, as the native capabilities of successive frontier model generations can rapidly render existing techniques obsolete. To assess this research investment, we analyze 35 LLM-based technique papers from ICSE 2026. We evaluate whether their complex tools can be outperformed by the simplest possible alternative: a single, automatically generated prompt executed on a newer generation model, without any iterative refinement. We find that for between 37% and 63% papers, a newer model with a single prompt natively outperforms the heavily engineered tooling proposed just a year prior. We identify that constructive techniques like code generation or repair are more amenable to substitution by a sing
    
[^19]: 面向安全关键部署流水线的审计优先回滚语义

    Audit-First Rollback Semantics for Safety-Critical Deployment Pipelines

    [https://arxiv.org/abs/2609.00406](https://arxiv.org/abs/2609.00406)

    提出审计优先回滚语义这一容错机制，保证在转换阶段发生故障停机崩溃时，部署运行时的审计链与实时状态在每个已提交的终止状态上保持一致。

    

    分布式部署运行时承载着一种经典容错框架未曾直接指明的连贯性义务：组件被配置运行的实时状态与记录其如何达到该状态的审计链，在每一个终止配置上必须保持一致。先前的工作主要关注部署时故障面的各个单独方面（金丝雀控制器、配置回滚、签名证明），而对于故障停机崩溃下审计/实时一致性这一横切问题仅有松散的规定。然而一个关键的系统问题仍未解决：部署运行时如何保证即使在转换过程中途崩溃时，审计链仍能如实反映实时状态？我们提出了审计优先回滚语义，这是一种容错机制，保证在转换阶段发生故障停机崩溃时，每个已提交的终止状态都满足审计/实时一致性。该机制与临时状态机、流水线（原文在此截断）

    arXiv:2609.00406v1 Announce Type: new  Abstract: Distributed deployment runtimes carry a coherence obligation that classical fault-tolerance frameworks do not name directly: the live state a component is configured to run and the audit chain that records how it got there must agree at every terminal configuration. Prior works mainly focus on individual aspects of the deploy-time fault surface (canary controllers, configuration rollback, signed attestations), leaving the cross-cutting question of audit/live coherence under fail-stop crash only loosely specified. Yet a key systems question remains unresolved: how can a deployment runtime guarantee that the audit chain answers truthfully about live state even when a transition crashes mid-flight? We present audit-first rollback semantics, a fault-tolerance mechanism that guarantees audit/live coherence at every committed terminal under fail-stop crashes during transition phases. The mechanism pairs with provisional state machines, pipelin
    
[^20]: 具身机器人能力市场的联邦信任机制

    Federated Trust for Embodied Robot Capability Marketplaces

    [https://arxiv.org/abs/2609.00404](https://arxiv.org/abs/2609.00404)

    本文提出用本地信任目录与 Ed25519 分离签名实现的联邦信任机制，取代中心化 PKI，使具身机器人技能市场能够在离线、多监管环境下完成本地化的安全安装验证。

    

    机器人能力市场——“机器人技能的应用商店”——正成为大语言模型驱动机器人机群的部署载体。对于“这个软件包是否可以安全安装？”这一问题，默认的云原生答案是中心化的公钥基础设施（PKI）：一个证书颁发机构、一个透明日志、一个信任根。我们认为，对于具身机器人机群而言这是错误的模型，因为运营商面临异构的监管体制、物理隔离（气隙）部署环境、极少的运维人员规模，以及信任错误发布者会带来的物理世界后果。我们提出联邦信任机制：每个已部署的桥接节点各自维护一个记录可接受签名者的本地信任目录；签名者通过分离式 Ed25519 签名信封中嵌入的公钥来标识自己；安装时验证只需进行本地集合成员检查，而无需与证书颁发机构进行网络往返通信。所采用的密码学原语刻意保持标准化（Ed25519 分离签名和 SSH 风格的信任文件……）。

    arXiv:2609.00404v1 Announce Type: cross  Abstract: Robot capability marketplaces, the "app store for robot skills," are emerging as the deployment vector for LLM-driven robot fleets. The default cloud-native answer to "is this package safe to install?" is centralised PKI: one certificate authority, one transparency log, one root of trust. We argue this is the wrong model for embodied robot fleets, where operators face heterogeneous regulatory regimes, air-gapped deployments, tiny operator headcounts, and physical-world consequences for trusting the wrong publisher. We present federated trust: each deployed bridge maintains its own local trust directory of acceptable signers; signers identify themselves with a public key embedded in a detached Ed25519 signature envelope; install-time verification is a local set-membership check rather than a network round trip to a certificate authority. The cryptographic primitives are deliberately standard (Ed25519 detached signatures and SSH-style tr
    
[^21]: 面向基于领导者的共识数据存储的操作类型感知客户端路由

    Operation-Type-Aware Client Routing for Leader-Based Consensus Datastores

    [https://arxiv.org/abs/2609.00392](https://arxiv.org/abs/2609.00392)

    提出一种操作类型感知的客户端路由策略，通过将写操作固定路由至领导者、将读操作分散至健康节点读池，在降低延迟、提升吞吐量的同时还能自动检测并隔离静默劣化的跟随者节点。

    

    基于领导者的共识数据存储（如 etcd、ZooKeeper）面临两个相互冲突的路由目标：一是将负载均匀分散到各成员上，二是将操作路由到协议角色与该操作相匹配的成员。写操作必须通过领导者提交，因此将其发送到其他节点会额外增加一次转发跳数。线性一致读只需要轻量级的领导者确认，之后任何成员都可以在本地提供服务。上游的 etcd 客户端使用 gRPC 的 round_robin 负载均衡器，将读和写操作均匀分布到集群各成员上。操作感知客户端通过将写操作固定路由到领导者、并将读操作分散到健康的读池中来解决这一矛盾。在 3 节点 etcd 集群的稳态实验中（80/20 读写混合负载，5 次试验），该方法使写操作 P50 延迟降低 29%，吞吐量提升 9%。当某个跟随者节点发生静默劣化时，操作感知客户端能够检测到延迟变化并将其从读池中移除，使读 P99 降低 64%、写 P99 降低 74%，并提升……

    arXiv:2609.00392v1 Announce Type: cross  Abstract: Leader-based consensus datastores (etcd, ZooKeeper) face two competing routing goals: spread load evenly across members, and route operations to the member whose protocol role matches the operation. Writes must commit through the leader, so sending them elsewhere adds a forwarding hop. Linearizable reads need only a lightweight leader confirmation before any member can serve them locally. The upstream etcd client uses gRPC's round_robin balancer, distributing reads and writes uniformly across cluster members. An operation-aware client resolves this by pinning writes to the leader and distributing reads across the healthy read pool. In steady state on a 3-node etcd cluster (80/20 read/write mix, 5 trials), this lowers write P50 by 29% and raises throughput by 9%. When a follower degrades silently, the operation-aware client detects the latency shift and removes it from the read pool, cutting read P99 by 64%, write P99 by 74%, and raisin
    
[^22]: 有界、不确定，还是缺陷：一个面向SQL聚合差分测试的条件感知判定器

    Bounded, Indeterminate, or a Bug: A Condition-Aware Oracle for Differential Testing of SQL Aggregates

    [https://arxiv.org/abs/2609.00381](https://arxiv.org/abs/2609.00381)

    该论文提出一个条件感知的差分测试判定器，以存储双精度浮点数的精确有理值为真值，将聚合结果差异分类为“精确、有界或不确定”，并推导出由算法条件数决定的可测试性边界 kappa* = (1/C_A)^(1/p)，从而能够区分真实缺陷与合理的浮点舍入误差。

    

    差分数据库测试通过比较不同数据库引擎的结果，将不一致之处判定为缺陷。对于浮点聚合计算而言，这种做法是不健全的：由于浮点运算不满足结合律，各引擎产生不一致的结果是合理的。实践中常用一个epsilon容差来修补这一问题，而主流的判定器则完全避开浮点运算。我们给出了这种实践所缺少的判定器，并证明其决定性因素不是查询本身，而是引擎所采用的算法。真值是所存储双精度浮点数的精确有理数值——即通过算术计算而非另一个引擎获得——并据此将每个不一致结果分类为精确、有界或不确定。聚合函数f在算法A下的相对误差满足 rel_err <= C_A(n,u) * kappa_f^p，因此可测试性边界——超过该边界后任何判定器都无法将缺陷与舍入区分开——为 kappa*_{f,A} = (1/C_A)^(1/p)。SUM和AVG属于线性情形（p=1）；方差在单遍算法下为p=2，在Welford算法下为p=1。我们在四个……类别中的八个引擎上……（原文摘要在此处截断）

    arXiv:2609.00381v1 Announce Type: cross  Abstract: Differential database testing compares results across engines and calls a discrepancy a bug. For floating-point aggregates this is unsound: engines legitimately disagree because floating-point arithmetic is not associative. Practice patches this with an epsilon; the leading oracles avoid floating point entirely. We give the oracle this practice lacks, and show its decisive quantity is not the query but the engine's algorithm. Ground truth is the exact rational value of the stored doubles -- arithmetic, not another engine -- and each discrepancy is classified exact, bounded, or indeterminate. The relative error of an aggregate f under an algorithm A obeys rel_err <= C_A(n,u) * kappa_f^p, so the testability boundary, beyond which no oracle can separate a bug from rounding, is kappa*_{f,A} = (1/C_A)^{1/p}. SUM and AVG are the linear case p=1; variance is p=2 for the one-pass algorithm and p=1 for Welford. Across eight engines in four clas
    
[^23]: 跨GPU核函数的确定性大语言模型推理：二的幂次INT8量化缩放因子与基于容差的符合性测试的局限

    Deterministic LLM Inference Across GPU Kernels: Power-of-Two INT8 Quantization Scales and the Limits of Tolerance-Based Conformance

    [https://arxiv.org/abs/2609.00363](https://arxiv.org/abs/2609.00363)

    本文通过对INT8量化GEMM流水线系统性注入九种故障，证明了基于容差的符合性测试在构造上无法检测仅使输出偏移至多一个bfloat16间距的尾部计算故障，而采用二的幂次量化缩放因子是保障跨GPU核函数确定性推理的关键途径之一。

    

    针对量化GEMM核函数的符合性测试套件所检验的，是两个实现是否在容差范围内保持一致。本文测量了此类测试套件究竟能够检测到什么。研究者在Qwen3-1.7B模型的8,232个层-故障-运行状态单元格上，向参考INT8推理流水线注入了九种故障，结果发现：五种尾部计算故障——缩放因子精度、双重舍入、乘法顺序、输出截断以及融合排序——中的每一种，在5,880个单元格中最多只会使输出偏移一个bfloat16的最小间距，且只要产生影响就恰好是一个间距。因此，将容差设为一个间距的测试在构造上对这一整类故障是盲目的：五种故障中有四种不被套件中的任何检查所发现，第五种也仅在使用二的幂次缩放因子时才会被检测到。与此相对，违反累加器精确性前提条件的故障、或破坏操作数共享的故障，则无一例外地被检测出来，而空故障从不触发。由此可以得出，这种形式的基于容差的测试套件所能确立的结论，比人们预期的要狭窄得多。

    arXiv:2609.00363v1 Announce Type: new  Abstract: Conformance suites for quantized GEMM kernels ask whether two implementations agree within a tolerance. We measure what such a suite can detect. Injecting nine faults into a reference INT8 pipeline over 8,232 layer--fault--regime cells of Qwen3-1.7B, we find that every one of five epilogue faults -- scale precision, double rounding, multiplication order, output truncation, fused ordering -- moves the output by at most a single bfloat16 spacing, and by exactly one whenever it moves it at all, across 5,880 cells. A tolerance of one spacing is therefore blind to the entire class by construction: four of the five faults are detected by no check in the suite, and the fifth only under power-of-two scales. Faults that violate the accumulator's exactness preconditions, or that break operand sharing, are detected without exception, and a null fault never fires. What a tolerance-based suite of this shape establishes is therefore narrower than inte
    
[^24]: 重新审视反馈驱动的大语言模型代码修复：一项复现研究与探索性Java扩展

    Revisiting Feedback-Driven LLM Code Repair: A Replication and Exploratory Java Extension

    [https://arxiv.org/abs/2609.00362](https://arxiv.org/abs/2609.00362)

    该研究部分复现了FeedbackEval基准并探索性地将其扩展到Java语言，发现基于Python得出的反馈驱动代码修复结论可能缺乏跨语言的泛化性。

    

    自大语言模型（LLM）问世以来，从业者越来越多地利用它们来支持软件工程任务，包括自动化代码修复，并展现出了可喜的成果。然而，关于可复现性和可泛化性的担忧在很大程度上仍未被探索。为了进一步评估这些担忧及其相关影响，我们对FeedbackEval基准进行了部分复现，并开展了探索性的Java扩展，该基准评估了LLM如何在Python代码修复中利用不同类型的反馈。首先，我们使用GPT-4o和Claude 3.5 Sonnet在394个修复任务上对原始研究进行了部分复现，重现并观察到了原工作中报告的主要定性趋势。其次，我们通过从50个Java任务构建100个错误修复实例并评估反馈的有效性，开展了探索性的Java扩展。我们的结果表明，先前基于Python得出的结论可能对……（原文摘要在此处截断）

    arXiv:2609.00362v1 Announce Type: new  Abstract: Since the advent of Large Language Models (LLMs), practitioners have increasingly leveraged them to support their software engineering tasks, including automated code repair, showing promising results. Yet, concerns regarding reproducibility and generalizability remain largely unexplored. To further evaluate these concerns and associated impacts, we partially reproduce and conduct an exploratory Java extension of the FeedbackEval benchmark [1], which evaluates how LLMs leverage different feedback types for Python code repair. First, we partially replicate the original study on 394 repair tasks using GPT-4o and Claude 3.5 Sonnet, reproducing and observing the main qualitative trends reported in the original work. Second, we conduct an exploratory Java extension by constructing 100 erroneous repair instances from 50 Java tasks and evaluating feedback effectiveness. Our results show that previous conclusions from Python may be sensitive to 
    
[^25]: 探索研究与实践中的量子软件测试：多声部文献综述的新兴结果

    Exploring Quantum Software Testing Across Research and Practice: Emerging Results from a Multivocal Literature Review

    [https://arxiv.org/abs/2609.00354](https://arxiv.org/abs/2609.00354)

    本文通过整合学术文献与灰色文献的多声部综述，首次系统刻画了研究界与工业界对量子软件测试的挑战、技术和工具的认知现状，揭示出该领域快速演进但碎片化的生态系统。

    

    本文呈现了一项多声部文献综述的初步发现，该综述研究了量子软件测试在学术文献与面向从业者的来源中是如何被描述的。我们的研究将同行评审的研究与灰色文献相结合，包括博客、教程、论坛、技术报告、文档页面和公司网页。我们的结果表明，量子软件测试是一个快速演进但碎片化的生态系统，涉及经典改造的测试方法、量子专用技术、统计验证方法、模拟器、调试环境和验证框架。所审阅的文献还揭示了反复出现的挑战，包括可扩展性限制、硬件噪声、概率性执行、有限的可观测性以及不成熟的工具生态系统。这些发现为当前学术界与工业界如何讨论量子软件测试的挑战、技术和工具提供了初步刻画。

    arXiv:2609.00354v1 Announce Type: new  Abstract: This paper presents preliminary findings from a multivocal literature review investigating how quantum software testing is characterized across academic and practitioner-oriented sources. Our study integrated peer-reviewed studies with gray literature, including blogs, tutorials, forums, technical reports, documentation pages, and company webpages. Our results indicate a rapidly evolving but fragmented ecosystem involving classical adapted testing approaches, quantum-specific techniques, statistical validation methods, simulators, debugging environments, and verification frameworks. The reviewed material also revealed recurring challenges related to scalability limitations, hardware noise, probabilistic execution, limited observability, and immature tooling ecosystems. These findings provide an initial characterization of how research and practice currently discuss quantum software testing challenges, techniques, and tooling.
    
[^26]: 面向智能体软件工程的规范驱动开发：驾驭人-智能体协作

    Spec-Driven Development for Agentic Software Engineering: Harnessing Human-Agent Teamwork

    [https://arxiv.org/abs/2609.00252](https://arxiv.org/abs/2609.00252)

    本文提出规范驱动开发（SDD）作为团队规模智能体软件工程的使能学科，并刻画了团队治理自主智能体行为的“缰绳”机制，以应对个体生产力提升但团队吞吐量与稳定性下降的生产力悖论。

    

    背景：软件工程正在从诸如“氛围编码”之类的AI辅助实践（即由助手加速个体开发者的工作）迈向智能体软件工程（ASE），在ASE中，自主智能体被委以目标层面的任务。然而，业界报告了一种生产力悖论：随着个体生产力的提升，团队吞吐量、审查能力和稳定性反而退化，原因在于团队规模的软件工程纪律被忽视。目标：本文旨在确立规范驱动开发（SDD）作为团队规模ASE使能学科的概念与方法论基础，并刻画“缰绳”框架，即团队用以治理智能体行为的技术与方法论机制。方法：我们进行了概念分析，主要借鉴灰色文献，包括ASE愿景与路线图论文、从业者报告、演讲以及工具，因为同行评审证据与一份（摘要内容在此处被截断）

    arXiv:2609.00252v1 Announce Type: new  Abstract: Context: Software engineering is moving from AI-assisted practices like vibe coding, in which assistants accelerate individual developers, towards Agentic Software Engineering (ASE), in which autonomous agents are delegated goal-level tasks. However, industry reports a productivity paradox: as individual productivity increases, team throughput, review capacity, and stability degrade because team-scale software engineering discipline is neglected. Objective: This paper aims to establish the conceptual and methodological foundations of Spec-Driven Development (SDD) as an enabling discipline for ASE at team scale and characterize the harness, i.e., the technical and methodological mechanisms through which teams govern agent behavior. Method: We conducted a conceptual analysis drawing predominantly on gray literature, including ASE vision and roadmap papers, practitioner reports, talks, and tooling, because peer-reviewed evidence and a share
    
[^27]: 实践中的实证软件工程：来自谷歌的洞见

    Empirical Software Engineering in Practice: Insights from Google

    [https://arxiv.org/abs/2609.00247](https://arxiv.org/abs/2609.00247)

    本文作为“实践中的ESE”专栏首篇，通过采访谷歌开发者智能团队的从业者，揭示了实证软件工程在工业界的实际实施方式、研究方法选择、成果应用及面临的常见障碍。

    

    尽管实证软件工程（ESE）在学术界如何应用已经相当广为人知，但我们对ESE在工业界如何实践的了解还很有限。作为我们实证软件工程定期专栏（ACM SIGSOFT SEN-ESE）的一部分，我们希望推出一系列文章，采访来自不同公司的ESE从业者。我们希望了解ESE流程在工业界是如何实施的，例如不同的研究方法、从业者如何决定研究内容、研究结果如何在公司内外使用，以及他们在工业环境中应用ESE方法时是否面临反复出现的障碍。在“实践中的ESE”第一期中，我们邀请到了谷歌开发者智能团队的Ciera Jaspan和Collin Green。本文是我们于2026年8月13日对话内容的忠实记录，并针对专栏进行了编辑。

    arXiv:2609.00247v1 Announce Type: new  Abstract: While it is fairly well known how empirical software engineering (ESE) is used in the academic world, we have limited knowledge of how ESE is practiced in industry. As part of our regular column on empirical software engineering (ACM SIGSOFT SEN-ESE), we want to dedicate a series of articles to interviewing ESE practitioners from various companies. Among other things, we want to understand how ESE processes are implemented in industry, e.g., different research methods, how practitioners decide on what to study, how research results are used within companies and beyond, and if they face recurrent impediments to using ESE methods in industrial contexts. In the first edition of "ESE in Practice", we are joined by Ciera Jaspan and Collin Green from the Developer Intelligence team at Google. This article is a faithful account of our conversation from August 13, 2026, which we edited for our column.
    
[^28]: 超越锁与线程ID：非常规路径上的静态数据竞争检测（扩展版）

    Beyond Locks and Thread IDs: Static Data Race Detection Off The Beaten Path (Extended Version)

    [https://arxiv.org/abs/2609.00246](https://arxiv.org/abs/2609.00246)

    该论文扩展了摘要框架，以支持线程屏障、pthread_once以及祖先线程锁集合等此前被静态数据竞争检测所忽视的并发构造与同步机制，并通过判定测试套件证明现有最先进的工具缺乏对这些特征的支持。

    

    维护线程执行历史的抽象可以提高静态分析中数据竞争检测的精度。在此，我们扩展了摘要框架，以处理静态竞争检测中一直被忽视的并发构造和同步机制。我们为常用的线程屏障引入了相应机制，同时也为 pthread_once（一种确保操作仅执行一次的机制）提供了支持。我们还通过祖先线程所持有锁集合的抽象来实例化该框架。我们提出了一套判定测试来评估针对这些特征的分析，并将我们的实现与最先进的工具进行比较，发现它们缺乏对这些特征的支持。

    arXiv:2609.00246v1 Announce Type: cross  Abstract: Maintaining an abstraction of the execution history of threads can improve the precision of data race detection in static analysis. Here, we extend the digest framework to handle concurrency constructs and synchronization mechanisms that have been ignored in static race detection. We introduce mechanisms for the commonly used thread barriers, as well as pthread_once, which allows to ensure that an action is executed only once. We also instantiate the framework with an abstraction of locksets held by ancestor threads. We propose a suite of litmus tests to evaluate analyses for these features and compare our implementation to state-of-the-art tools, finding that they lack support.
    
[^29]: 别让模型直接写 YAML：从 LLM 提议的字段变更生成确定性、最小差异的 GitOps 修复

    Don't Let the Model Write the YAML: Deterministic, Minimal-Diff GitOps Remediation from LLM-Proposed Field Changes

    [https://arxiv.org/abs/2609.00227](https://arxiv.org/abs/2609.00227)

    该论文提出让 LLM 只负责做出字段级的语义决策，再由确定性工具将其转换为最小差异的配置修改，从而避免让模型直接生成 YAML 文件或 diff 所带来的静默损坏、不确定性和高开销问题。

    

    LLM 智能体越来越多地被用于诊断故障并提出修复建议。在 GitOps 工作流中，应用修复意味着编辑受版本控制的配置文件，而最直观的实现方式——让模型撰写修改后的文件或 diff——正是从业者最先会尝试的方案。我们在真实的 Kubernetes 清单上对这一选择进行评估后发现，没有任何文本生成策略对无人值守的自动化是安全的。统一 diff 格式并不安全：在严格打补丁的模式下几乎没有补丁能成功应用，但这只是表象，因为宽容的工具（GNU patch）能应用 96% 的补丁，却会静默错误地应用约七分之一（14-20%）的补丁，且不给出任何错误信号。全文件重写则取决于模型能力：小模型会损坏文件，而前沿模型虽然通常正确但不具确定性（在某些运行中会静默丢弃某个字段或误改相邻字段），并且必须重新生成整个文件，每次编辑的代价为 O(文件大小)。我们提出了一种替代方案，将语义决策（哪个资源……）与后续处理分离——（摘要在此处截断）

    arXiv:2609.00227v1 Announce Type: cross  Abstract: LLM agents increasingly diagnose incidents and propose remediations. In a GitOps workflow, applying a fix means editing a version-controlled config file, and the obvious implementation, having the model author the edited file or a diff, is what practitioners reach for first. Evaluating that choice on real Kubernetes manifests, we find no text-generation strategy is safe for unattended automation. Unified diffs are unsafe: under strict patching almost none apply, but that is an artifact, since a tolerant tool (GNU patch) applies 96%, yet silently misapplies about 1 in 7 (14-20%) with no error signal. Full-file rewrite is capability-dependent: a small model corrupts the file, while a frontier model is usually correct but non-deterministic (it silently drops a field or edits a neighbor on some runs) and must regenerate the whole file, costing O(file size) per edit. We present an alternative that separates the semantic decision (which reso
    
[^30]: 先答后判式LLM评判会继承评判者自身的错误

    Commit-first LLM judging inherits the judge's own errors

    [https://arxiv.org/abs/2609.00088](https://arxiv.org/abs/2609.00088)

    研究发现“先答后判”式LLM评判会继承评判者自身的错误，而对八个主流评估框架的审计表明无一真正实现该方法，其中九个框架因复制同一祖先提示词而采用了已被证明无效的变体，导致大量错误代码被放行。

    

    LLM评判器（即对另一个系统输出进行打分的模型）可能被被其评分的系统“钻空子”。近期研究指出了一种确实有效的防御方法：评判器先自行解决任务并固定自己的答案，然后仅当候选答案与其一致时才予以接受。我们将这一做法称为“先答后判”评判，并探究已发布的软件是否实现了该方法，以及其代价是什么。我们审计了八个广泛使用的评估框架的默认评判器配置：在纳入范围的24个配置中，没有一个实现了该方法；其中九个实现了文献中被测得无效的一种变体，并且共享同一个祖先提示词——这一点可以通过一个被复制下来的排版错误进行追溯。在一项受控实验中，一个普通的、无法访问正确答案的best-of-N搜索，严格按照文档说明使用其中一个配置来优化代码。在一个区间合并任务上，该评判器在一个随机种子下接受了96个候选中的90个，在另一个种子下接受了93个；每个被接受的候选……（原文摘要在此截断）

    arXiv:2609.00088v1 Announce Type: cross  Abstract: LLM judges, models that score another system's output, can be gamed by the systems they score. Recent work identifies one defence that works: the judge solves the task itself first and commits to that answer, then accepts a candidate only if the two match. We call this commit-first judging, and ask whether shipped software implements it, and what it costs.   We audit the default judge configurations of eight widely used evaluation frameworks. Of the 24 configurations in scope, none implement it. Nine implement a variant the literature measures as ineffective, and share one ancestor prompt, traceable through a copied typographical error.   In a controlled experiment, an ordinary best-of-N search with no access to correct answers optimises code against one of these configurations, used exactly as documented. On an interval merging task the judge accepted 90 of 96 candidates in one seed and 93 of 96 in the other; every accepted candidate 
    
[^31]: 面向Web开发中代码驱动智能体测试的框架与基准

    Framework and Benchmark for Code-Driven Agentic Testing in Web Development

    [https://arxiv.org/abs/2609.00081](https://arxiv.org/abs/2609.00081)

    提出了代码驱动智能体测试（CAT）范式，通过CATJudge框架与CATTest基准让智能体编写Playwright代码自主探索Web应用以发现缺陷，实验揭示当前主流视觉语言模型的缺陷发现能力仍然薄弱。

    

    端到端GUI测试对于验证Web应用至关重要，然而现有评估依赖于预定义的检查清单，且局限于Web生成基准的数据与框架，使得视觉语言模型（VLM）的缺陷发现能力未被系统性测试。我们提出了代码驱动智能体测试，这是一种新范式：智能体通过编写Playwright代码来驱动浏览器、收集反馈，并自主探索Web应用以发现缺陷。我们通过CATJudge和CATTest来实例化CAT：CATJudge是一个智能体框架，在单一环境中统一了Browser-Use和Computer-Use工具；CATTest是一个包含102个AI生成的Web应用的基准，这些应用带有精心标注的缺陷，通过紧密的人机协作构建，具备复杂交互和隐蔽缺陷。针对主流VLM的实验表明，所有被评估的模型表现均不佳，揭示了当前……

    arXiv:2609.00081v1 Announce Type: new  Abstract: End-to-end GUI testing is essential for verifying web applications, yet existing evaluations rely on predefined checklists and are confined to the data and frameworks of web generation benchmarks, leaving the bug-discovery ability of vision-language models (VLMs) systematically untested. We introduce \textbf{C}ode-driven \textbf{A}gentic \textbf{T}esting (CAT), a paradigm in which the agent writes Playwright code to drive the browser, gathers feedback, and autonomously explores web applications to uncover bugs. We instantiate CAT with CATJudge, an agentic framework that unifies Browser-Use and Computer-Use tools within a single environment and CATTest, a benchmark of 102 AI-generated web applications with carefully annotated bugs, built through close human-AI collaboration to feature complex interactions and subtle defects. Experiments with mainstream VLMs show that all evaluated models perform poorly, revealing a clear gap between curre
    
[^32]: 差异之下：诊断与缓解代码级自主研究循环中的算法模式坍缩

    Beneath the Diff: Diagnosing and Mitigating Algorithmic Mode Collapse in Code-Level Autonomous Research Loops

    [https://arxiv.org/abs/2609.00077](https://arxiv.org/abs/2609.00077)

    论文系统性地诊断出代码级自主研究循环中一种名为“算法模式坍缩”的失效模式——即表层编辑多样性看似稳定但算法层面的语义与机制多样性已经坍缩，并提出了相应的缓解方法。

    

    代码级自主研究循环最近成为自动化机器学习研究中一个具体的研究对象。在此类循环中，大语言模型智能体对实验训练流程提出修改建议，执行修改后的流程，并保留能够提升可验证的循环内指标的修改。尽管可执行的指标看似能提供可靠的进展信号，但目前尚不清楚这种重复的、由指标驱动的代码编辑是否能带来超越循环本身的真正泛化改进。我们对这一问题进行了系统性诊断。在多种实验设置中，我们发现了一种稳健的失效模式，我们称之为“算法模式坍缩”。在这种状态下，表层的编辑多样性保持稳定，但语义层面与机制层面的多样性发生坍缩：智能体持续编辑不同的代码行，却反复提出相同类型的算法修改。这种坍缩伴随着

    arXiv:2609.00077v1 Announce Type: new  Abstract: Code-level autonomous research loops (ARLs) have recently emerged as a concrete object of study in automated machine learning research. In such loops, an LLM agent proposes modifications to an experimental training pipeline, executes the modified pipeline, and retains edits that improve a verifiable in-loop metric. Although executable metrics may appear to provide a reliable signal of progress, it remains unclear whether repeated metric-driven code editing leads to genuine improvements that generalize beyond the loop. We provide a systematic diagnosis of this question. Across various experiment settings, we identify a robust failure mode that we call \textbf{algorithmic mode collapse}. In this regime, surface-level edit diversity remains stable, but semantic and mechanism-level diversity collapse: the agent continues to edit different lines of code while repeatedly proposing the same kinds of algorithmic changes. This collapse is accompa
    
[^33]: MCP客户端能否在失败后决定下一步做什么？一种仅基于结果的可操作性审计

    Can MCP Clients Decide What to Do After Failure? A Result-Only Actionability Audit

    [https://arxiv.org/abs/2609.00072](https://arxiv.org/abs/2609.00072)

    该论文提出了一种仅基于MCP失败结果的可操作性审计框架，发现类型化字段虽能暴露失败和宽泛策略信息，但缺乏具体原因、目标和可执行修复方法，而自然语言描述虽含更丰富的信息却需要语义解释才能用于恢复。

    

    接收到isError:true的客户端知道出了问题，但仍可能没有机器可读的依据来决定是修复参数、进行身份验证、等待、选择另一个工具还是停止。本文研究了确定性软件仅从已完成的MCP失败结果中能获取什么信息；请求参数、模式、发现历史、身份验证状态、传输元数据、主机策略和应用状态都在该边界之外。我们引入了一个由六部分组成的可操作性分析框架，并以记录级证据加以应用。在一项对来自十个可达采样服务器的21个安全诱导失败的小型示例研究中，类型化字段在18个案例中暴露了失败信息，在8个案例中暴露了宽泛的策略信息，但均未暴露具体原因、目标、可执行的修复方法或重放约束。自然语言描述通常携带更多的原因和目标信息，但代价是将语义解释纳入了恢复路径之中。一项词汇层面的来源审计发现……

    arXiv:2609.00072v1 Announce Type: new  Abstract: A client that receives isError:true knows that something went wrong. It may still have no machine-readable basis for deciding whether to fix an argument, authenticate, wait, choose another tool, or stop. This paper studies what deterministic software can learn from a completed MCP failure result alone; request arguments, schemas, discovery history, authen- tication state, transport metadata, host policy, and ap- plication state are outside that boundary. We introduce a six-part actionability profile and apply it with record- level evidence. In a small illustrative study of 21 safely induced failures from ten reachable sampled servers, typed fields expose failure in 18 cases and a broad policy in 8, yet expose no specific cause, target, executable repair, or replay constraint. Prose often carries more cause and target information, at the price of making semantic interpretation part of the recovery path. A lexical source audit finds the sa
    
[^34]: CUDA-Harness：从自然语言驱动的智能体式CUDA内核生成与优化

    CUDA-Harness: Harnessing Agentic CUDA Kernel Generation and Optimization from Natural Language

    [https://arxiv.org/abs/2609.00058](https://arxiv.org/abs/2609.00058)

    该论文提出CUDA-Harness框架，通过智能体式方法直接从自然语言生成并优化高性能CUDA内核，克服了现有工作局限于PyTorch转译以及因依赖预定义测试输入而易受奖励欺骗的不足。

    

    开发高性能CUDA内核需要掌握算法实现、正确性验证以及面向硬件的并行优化等专业知识，这构成了很高的专业门槛，因此直接从自然语言生成CUDA内核变得至关重要。与此同时，大语言模型（LLM）通用的代码生成能力催生了一系列基于LLM的CUDA内核生成研究。这些工作主要聚焦于从PyTorch等高级框架向CUDA的转译（Torch2CUDA），而非Text2CUDA——后者要求模型既要理解高层输入语义，又要处理底层的内核实现与验证。此外，由于依赖预定义的测试输入，这些方法容易受到奖励欺骗的影响。在本文中，我们提出了CUDA-Harness，一个用于从自然语言驱动智能体式CUDA内核生成与优化的框架。

    arXiv:2609.00058v1 Announce Type: cross  Abstract: Developing high-performance CUDA kernels demands specialized knowledge in algorithm implementation, correctness validation, and hardware-aware parallel optimization, creating a substantial expertise barrier and making generating CUDA kernels directly from natural language (Text2CUDA) essential. Meanwhile, the general-purpose code generation capability of Large Language Models (LLMs) prompts a series of works exploring LLM-based CUDA kernel generation. They mainly focus on transpilation from high-level frameworks such as PyTorch to CUDA (Torch2CUDA) rather than Text2CUDA, where models must understand the high-level input semantics and handle low-level kernel implementation and validation. Additionally, these methods are vulnerable to reward hacking due to reliance on predefined test inputs. In this paper, we propose CUDA-Harness, a framework for harnessing agentic CUDA kernel generation and optimization from natural language. Specifical
    
[^35]: 迈向智能体化云工程：基于零信任智能体套件的图工程与循环工程

    Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness

    [https://arxiv.org/abs/2609.00050](https://arxiv.org/abs/2609.00050)

    提出了一个智能体云工作流工程框架，通过将图工程（长时程工作流推进）、循环工程（有界诊断与修复重试）和零信任智能体套件（受限执行）三个关注点分离，将自然语言云工程任务自动转化为经过验证的代码仓库和可验证的云部署。

    

    智能体AI正在推动基于云的工作流的发展，其中自主智能体可以对运营状态进行推理、调用授权工具、修改软件和基础设施、部署服务、验证执行结果，并在长时程、多步骤任务中进行自适应调整。构建此类工作流需要针对工作流推进、受限执行、故障恢复和可验证完成等环节的显式机制。我们提出了智能体云工作流工程，这是一个智能体AI框架，它将自然语言描述的智能体云工程任务转化为经过验证的代码仓库和经过验证的运营性云部署，从而实现基于云的智能体工作流自动化。该框架分离了三个互补的关注点：图工程负责指定长时程工作流推进以及依赖验证的状态转移；循环工程提供有界的诊断、修复或重新规划、重试和重新验证；智能体套件工程则（负责执行零信任的受限执行控制）。

    arXiv:2609.00050v1 Announce Type: cross  Abstract: Agentic AI is enabling cloud-based workflows in which autonomous agents reason over operational state, invoke authorized tools, modify software and infrastructure, deploy services, verify execution outcomes, and adapt across long-horizon, multistep tasks. Engineering such workflows requires explicit mechanisms for workflow progression, constrained execution, failure recovery, and verifiable completion. We present Agentic Cloud Workflow Engineering, an agentic AI framework that transforms natural-language agentic cloud-engineering tasks into validated code repositories and verified operational cloud deployments for automating cloud-based agentic workflows. The framework separates three complementary concerns: graph engineering specifies long-horizon workflow progression and verification-dependent transitions; loop engineering provides bounded diagnosis, repair or re-planning, retry, and re-verification; and agent harness engineering enf
    
[^36]: 什么是系统？一般系统论中基于相互作用的结构-行为融合观

    What Is a System? An Interaction-Based Account of Structure-Behavior Coalescence in General Systems Theory

    [https://arxiv.org/abs/2609.00043](https://arxiv.org/abs/2609.00043)

    本文提出“结构-行为融合”（SBC）观点，主张系统是结构化实体，其行为源于组成实体间的相互作用，且这些相互作用同时构成系统的结构组织与行为实现，从而为“什么是系统”提供了基于相互作用的统一解释。

    

    什么构成系统的问题仍然是一般系统论中的基础性问题。现有的定义通常从要素、关系、边界、功能或相互作用等角度来刻画系统，但这些视角并不总能提供关于系统结构与系统行为如何相互构成的统一说明。本文提出“结构-行为融合”，作为一种基于相互作用的系统本质解释。从SBC的视角来看，系统不仅仅是要素或关系的集合，也不能仅通过脱离结构的行为来充分刻画。相反，系统是一个结构化实体，其行为通过其组成实体之间的相互作用而产生，而这些相互作用同时促成系统的结构组织与行为实现。本文通过区分系统结构、相互作用（后文缺失）等内容来进一步发展这一视角。

    arXiv:2609.00043v1 Announce Type: new  Abstract: The question of what constitutes a system remains fundamental to General Systems Theory. Existing definitions commonly characterize a system in terms of elements, relationships, boundaries, functions, or interactions, but these perspectives do not always provide a unified account of how system structure and system behavior constitute one another. This paper proposes Structure-Behavior Coalescence (SBC) as an interaction-based account of what a system is. From the SBC perspective, a system is not merely a collection of elements or relationships, nor is it adequately characterized by behavior considered independently of structure. Rather, a system is a structured entity whose behavior arises through interactions among its constituent entities, with those interactions simultaneously contributing to both its structural organization and behavioral realization. The paper develops this perspective by distinguishing system structure, interaction
    
[^37]: 结构-行为融合与传统系统论的局限

    Structure-Behavior Coalescence and the Limits of Traditional Systems Theory

    [https://arxiv.org/abs/2609.00042](https://arxiv.org/abs/2609.00042)

    本文提出“结构-行为融合”（SBC）原则，主张结构与行为是同一系统过程中相互构成的两个方面，系统同一性源于二者的持续共同决定，从而克服传统系统论将结构与行为分离处理所带来的解释困难。

    

    本文考察了传统系统论中的一个基础性假设，即结构（组件的组织方式）与行为（系统活动随时间的演化）可以被当作可分离的分析维度来处理。论文认为，这种分离导致了在解释系统同一性方面的持续困难，特别是在涉及变化、涌现和边界界定的情况下。为解决这一问题，论文引入了“结构-行为融合”（SBC）作为一种重新框架化的原则。SBC提出，结构与行为不应被理解为独立存在、随后通过建模构造相互关联的实体，而应被视为同一系统过程中相互构成的两个方面。从这一视角出发，系统同一性被理解为源于结构组织与行为动力学的持续共同决定，而非源于二者的外部对应关系。

    arXiv:2609.00042v1 Announce Type: new  Abstract: This paper examines a foundational assumption in traditional systems theory, namely that structure (the organization of components) and behavior (the evolution of system activity over time) can be treated as separable analytical dimensions. It argues that this separation contributes to persistent difficulties in explaining system identity, particularly in cases involving change, emergence, and boundary specification. To address this issue, the paper introduces Structure-Behavior Coalescence (SBC) as a reframing principle. SBC proposes that structure and behavior should not be understood as independently existing entities that are subsequently related through modeling constructs, but as mutually constitutive aspects of a single systemic process. From this perspective, system identity is understood as arising from the sustained co-determination of structural organization and behavioral dynamics, rather than from their external corresponden
    
[^38]: trajectory-judge：仅基于结果的LLM评判器在智能体轨迹上遗漏了什么

    trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories

    [https://arxiv.org/abs/2609.00038](https://arxiv.org/abs/2609.00038)

    仅看最终结果的LLM评判器无法发现智能体“答对但走错路”的问题——在可构造真值的确定性客服工具环境中，仅结果型评判器对静默故障的召回率仅45%且误报33%的正确轨迹，而基于逐步评分标准的评判器可将静默故障召回率提升至77%。

    

    仅基于结果的评估是LLM智能体在生产环境中的默认做法：向评判器展示用户请求和最终回复，询问其处理是否得当。这一指标在结构上无法察觉那些“以错误方式得到正确答案”的智能体。我们在真值可以通过构造获知的场景下测量这一盲区：一个确定性的使用工具的客服支持台环境、一个总能解决问题的脚本化oracle策略，以及一个在已知步骤恰好破坏一个环节的故障注入器，并根据用户可见结果是否仍然保持（静默型故障）与否（显性型故障）对故障进行分层。五种评判器（程序化规则、仅结果型、两种模型规模的逐步评分标准型、以及自一致性集成）在400条轨迹上按照检测能力、步骤定位、故障类型判定、校准度和成本进行评分。结果显示：仅结果型评判器能捕获84%的显性故障，但只能捕获45%的静默故障，同时还会误报33%的正确轨迹；而逐步评分标准型评判器对静默故障的召回率达到77%。

    arXiv:2609.00038v1 Announce Type: cross  Abstract: Outcome-only evaluation is the production default for LLM agents: show a judge the request and the final reply and ask whether it was handled well. The metric is structurally blind to an agent that reaches the right answer the wrong way. We measure that blind spot where ground truth is known by construction: a deterministic tool-using support-desk environment, a scripted oracle policy that always solves it, and a fault injector that breaks exactly one thing at a known step, stratifying faults by whether the customer-visible outcome survived (silent) or not (loud). Five judges (programmatic rules, outcome-only, step-rubric at two model sizes, and a self-consistency ensemble) are scored on detection, step localisation, fault typing, calibration, and cost over 400 trajectories. The outcome-only judge catches 84% of loud faults but 45% of silent ones while flagging 33% of correct trajectories; a step-rubric judge reaches 77% silent recall 
    
[^39]: SilentProbe：测量作为智能体工具的生产级API中的静默失败

    SilentProbe: Measuring Silent Failure in Production APIs Used as Agent Tools

    [https://arxiv.org/abs/2609.00035](https://arxiv.org/abs/2609.00035)

    该论文首次大规模测量了LLM智能体调用生产API时的“静默失败”现象，发现API模式中机器可校验的约束（而非供应商身份）是预测服务器能否诚实报错的关键因素，而当前OpenAPI文档中这类约束严重缺失。

    

    arXiv:2609.00035v1 公告类型：新论文 摘要：调用生产级API的大语言模型智能体无法区分“查询未匹配到任何结果”与“服务器未能理解查询”这两种情况：两者都返回HTTP 200和可解析的响应体，没有可捕获的异常，也没有可供分支判断的字段。我们研究了哪些因素能够预测发生的是哪一种情况，以及这对智能体造成的影响。通过对2,501份独立发布的OpenAPI文档中721,320个参数的审计，我们发现仅有7.5%的参数声明了枚举类型，15.2%声明了任何机器可校验的约束，而40.1%的文档在自然语言描述中至少陈述了一条其模式（schema）并未编码的约束。我们通过单一聚合层对来自27家供应商的实时商业端点执行了219次由模式导出的扰动测试，该聚合层为每次调用发布模式并返回运行标识符。结果表明，预测服务器“诚实性”的是约束的形式而非供应商身份：机器可校验的约束在111个案例中的全部111个都产生了诚实的错误报告，而仅有自然语言描述的约束……（原文摘要在此处截断）

    arXiv:2609.00035v1 Announce Type: new  Abstract: An LLM agent calling a production API cannot distinguish a query that matched nothing from a query the server did not understand. Both return HTTP 200 with a parsable body, no exception to catch and no field to branch on. We ask what predicts which one occurred, and what it does to the agent. Auditing 721,320 parameters across 2,501 independently published OpenAPI documents, we find that 7.5% declare an enumeration and 15.2% declare any machine-checkable constraint at all, while 40.1% of documents state at least one constraint in prose that their schema does not encode. Executing 219 schema-derived perturbations against live commercial endpoints from 27 vendors, reached through a single aggregation layer (Monid) that publishes a schema and returns a run identifier for every call, we find that constraint form, not vendor identity, predicts honesty: machine-checkable constraints yielded an honest error in 111 of 111 cases, prose-only const
    
[^40]: Harness 工程学：编码智能体的解剖、架构与演化——基于十一套系统的源代码研究

    Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents -- A Source-Code Study of Eleven Systems

    [https://arxiv.org/abs/2609.00006](https://arxiv.org/abs/2609.00006)

    本文通过对十一套生产级编码智能体 harness 的源代码解剖，定义了 harness 的七大标准子系统，总结出 13 条跨系统观察结论和 29 种常见设计模式，为新兴的 harness 工程学科建立了最全面的实证基础。

    

    智能体是模型加上 harness（驾驭层）——即通过循环、工具、上下文管理、安全控制、编排和扩展接口将大语言模型与外部世界连接起来的运行时环境。Harness 工程学于 2026 年初被正式确立为一门学科，其研究对象正是这一运行时的设计与演化。本文为这一新兴学科提供了迄今为止最全面的实证基础：对十一套生产级编码 harness（Claude Code、Codex CLI、Gemini CLI、Mistral Vibe、OpenHands、Aider、Mini-SWE-Agent、Hermes、Pi、OpenCode、OpenClaw）以及作为对照点的首个元 harness——Omnigent——进行了源代码层面的解剖分析。论文定义了什么是 harness，绘制了其七个标准子系统及每个子系统的最小与最大实现方式，并据此对全部十一套系统逐一解剖。此次审计得出了 13 条跨系统的观察结论，并归纳出包含 29 种反复出现的设计模式的目录。两项缺失现象在语料规模扩大三倍后依然存在：在整个……（摘要原文在此处截断）

    arXiv:2609.00006v1 Announce Type: new  Abstract: An agent is a model plus a harness -- the runtime that couples an LLM to the world through a loop, tools, context management, safety controls, orchestration, and extension surfaces. Harness engineering, named as a discipline in early 2026, is the design and evolution of that runtime. This paper gives the young discipline its most comprehensive empirical foundation to date: a source-code anatomy of eleven production coding harnesses (Claude Code, Codex CLI, Gemini CLI, Mistral Vibe, OpenHands, Aider, Mini-SWE-Agent, Hermes, Pi, OpenCode, OpenClaw), plus Omnigent, the first meta-harness, analyzed as a contrast point. We define what a harness is, map its seven canonical subsystems with the minimal and maximal implementation of each, and dissect all eleven systems along those subsystems. The audit yields 13 cross-cutting observations and a catalog of 29 recurring design patterns. Two absences survive a threefold corpus expansion: across roug
    
[^41]: RealSWE：真实用户请求下编程智能体的组合式评估

    RealSWE: A Compositional Evaluation of Coding Agents under Realistic User Requests

    [https://arxiv.org/abs/2608.27831](https://arxiv.org/abs/2608.27831)

    该论文提出RealSWE基准，通过381个源自SWE-bench的多变体任务族来模拟简短、随意、信息稀疏的真实用户请求，从而更真实地评估编程智能体，并揭示了现有基准与真实用户请求在信息完整度和语言风格上的显著差距。

    

    编程智能体目前通常在SWE-bench系列基准上进行评估，这些基准的任务由精心整理的GitHub issue构建——这些issue冗长、结构化且信息丰富。然而，真实用户请求通常要短得多且结构化程度更低。为了刻画这一差距，我们定义了一个包含六个类别的信息分类法和四个语言风格维度，并将其应用于来自SWE-chat的真实用户提示以及SWE-bench Verified和Pro的问题陈述。我们发现，仅包含问题陈述（无论是否附带有限额外上下文）的请求占真实提示的88%，却仅占基准问题的7%。此外，87%的真实提示以随意口吻书写，而94%的基准问题则是正式的。基于这些观察，我们提出了RealSWE，其中包含381个源自SWE-bench Verified和Pro的多变体任务族。每个任务族内的变体共享相同的底层任务和标准补丁，仅在……（摘要在此处截断）

    arXiv:2608.27831v1 Announce Type: cross  Abstract: Coding agents are now commonly evaluated on the SWE-bench family of benchmarks, whose tasks are built from curated GitHub issues--long, structured, and information-rich. Real user requests, however, are typically far shorter and less structured. To characterize this gap, we define a six-category information taxonomy and four dimensions of linguistic style, and apply them to real user prompts from SWE-chat and problem statements from SWE-bench Verified and Pro. We find that requests carrying only a problem statement, alone or with limited additional context, account for 88% of real prompts but just 7% of benchmark problems. Furthermore, 87% of real prompts are casually written whereas 94% of benchmark problems are formal. Guided by these observations, we introduce sys, 381 multi-variant task families derived from SWE-bench Verified and Pro. Variants within each family share the same underlying task and gold patch while differing only in
    
[^42]: SWE-bench Science：编码代理能否解决科学中的工程任务？

    SWE-bench Science: Can Coding Agents Resolve Engineering Tasks in Science?

    [https://arxiv.org/abs/2608.19799](https://arxiv.org/abs/2608.19799)

    本文提出了SWE-bench Science，一个针对科学软件工程的仓库级基准，并揭示即使最佳代理在科学任务中成功率也低于50%，主要因科学知识不足等四种机制导致失败。

    

    arXiv:2608.19799v1 公告类型：新 摘要：软件日益成为科学仪器本身的一部分，使得科学代码中的故障不仅可能损害程序行为，还可能损害科学结论所依据的证据。然而，现有对编码代理的评估主要强调整体任务成功率，对于代理在修复科学软件时为何失败提供的见解有限。我们引入了 \textbf{SWE-bench Science}，一个面向科学软件工程的仓库级基准测试，包含来自20个科学领域98个GitHub仓库的119个任务。每个任务被组织为三种范式之一：问题驱动、专家探索和工程集成。即使是最佳表现的代理 \textbf{Claude Code with Opus-5 (max)}，其pass@1也低于50\%，凸显了科学软件工程带来的巨大挑战。我们识别出四种反复出现的失败机制：科学知识不足、领域特定工具使用错误、错误诊断不准确以及测试覆盖不充分。

    arXiv:2608.19799v1 Announce Type: new  Abstract: Software increasingly functions as part of the scientific instrument itself, making failures in scientific code capable of compromising not only program behavior but also the evidence underlying scientific conclusions. Yet existing evaluations of coding agents largely emphasize aggregate task success, providing limited insight into why agents fail when repairing scientific software. We introduce \textbf{SWE-bench Science}, a repository-level benchmark for scientific software engineering comprising 119 tasks from 98 GitHub repositories across 20 scientific domains. Each task is organized into one of three paradigms: Issue-driven, Expert-exploratory, and Engineering-integration. Even the best-performing agent, \textbf{Claude Code with Opus-5 (max), achieves a pass@1 below 50\%}, highlighting the substantial challenges posed by scientific software engineering. We identify four recurring failure mechanisms: deficits in scientific knowledge o
    
[^43]: MetaInfer：一个仅需知识即可生成LLM推理引擎的SKILL工具箱

    MetaInfer: A Knowledge Only LLM Inference Engine Generator SKILL Toolbox

    [https://arxiv.org/abs/2607.12875](https://arxiv.org/abs/2607.12875)

    本文提出MetaInfer，一种利用LLM作为编译器，通过多智能体协作和契约知识库，仅根据用户指定的运行时约束自动生成定制化推理框架的方法，以减少代码复杂性和性能开销。

    

    随着大语言模型技术的进步，模型家族、计算硬件、量化方案、并行化策略以及专用优化内核的空间持续扩大，这急剧增加了通用推理框架的代码复杂性和维护成本。传统软件工程通过多层抽象来支持多样化的应用场景，但这些抽象也增加了系统复杂性，并可能引入额外的性能开销。本文提出了metainfer，一种“LLM即编译器”的方法，用户只需指定推理程序的运行时约束。一个由LLM驱动的多智能体协作系统，结合契约知识库，自动生成一个满足这些约束的紧凑定制推理框架。我们从三个角度评估metainfer：源代码参考的效果、运行时行为以及...

    arXiv:2607.12875v2 Announce Type: replace-cross  Abstract: As LLM technology advances, the space of model families, compute hardware, quantization schemes, parallelization strategies, and specialized optimization kernels continues to expand, sharply increasing the code complexity and maintenance cost of general-purpose inference frameworks. Conventional software engineering uses multiple layers of abstraction to support diverse application scenarios, but these abstractions also increase system complexity and may introduce additional performance overhead. This paper presents metainfer, an 'LLM-as-Compiler' approach in which users specify only the runtime constraints of an inference program. An LLM-driven multi-agent collaboration system, coupled with a contract knowledge base, then automatically generates a compact customized inference framework that satisfies these constraints. We evaluate metainfer from three perspectives: the effect of source-code reference, the runtime behavior and 
    
[^44]: SEDCoT：通过符号执行与增量调试增强基于大语言模型的COBOL代码翻译

    SEDCoT: Enhancing LLM-Based COBOL Code Translation via Symbolic Execution and Delta Debugging

    [https://arxiv.org/abs/2607.04092](https://arxiv.org/abs/2607.04092)

    SEDCoT是一个COBOL到C的代码翻译框架，它先利用大语言模型进行初始翻译，再结合符号执行生成测试套件迭代修复语义差异，并通过增量调试将失败测试最小化为简洁反例，从而提升翻译正确性。

    

    COBOL在银行、保险和政府基础设施中仍然至关重要。然而，由于技术过时、文档稀缺以及开发人员退休，维护工作日益困难，因此需要将代码翻译成C等现代语言。传统的基于规则的转编译器产生的输出难以阅读和维护，而通用大语言模型（LLM）的正确性欠佳，因为COBOL是一种低资源语言，具有独特的逻辑模式。为了弥合这一差距，我们提出了SEDCoT，一种新颖的COBOL到C的翻译框架。SEDCoT首先利用LLM进行初始翻译，然后将符号执行与LLM引导相结合，生成测试套件并迭代修复语义差异。最后，它集成了增量调试技术，将失败的测试最小化为简洁的反例，从而加速自动化代码修复。在公开的COBOL到C数据集上对SEDCoT进行评估……

    arXiv:2607.04092v2 Announce Type: replace  Abstract: COBOL remains critical across banking, insurance, and government infrastructure. However, maintenance is increasingly challenging due to outdated technologies, sparse documentation, and developer retirement, necessitating code translation into modern languages like C. Traditional rule-based transcompilers yield outputs that are difficult to read and maintain, while general-purpose large language models (LLMs) achieve suboptimal correctness because COBOL is a low-resource language with distinct logic patterns. To bridge this gap, we propose SEDCoT, a novel COBOL-to-C translation framework. SEDCoT first leverages LLMs for initial translation, then combines symbolic execution with LLM guidance to generate test suites and iteratively repair semantic discrepancies. Finally, it integrates delta debugging to minimize failing tests into succinct counterexamples, accelerating automated code repair. Evaluating SEDCoT on a public COBOL-to-C dat
    
[^45]: ChainSWE：在多缺陷软件维护任务上对编码智能体进行基准测试

    ChainSWE: Benchmarking Coding Agents on Multi-Bug Software Maintenance

    [https://arxiv.org/abs/2607.02606](https://arxiv.org/abs/2607.02606)

    ChainSWE是首个评估编码智能体在共享代码库中进行顺序性、相互依赖缺陷修复的基准测试，通过收集54个Python项目的304个问题链条，揭示了随着链条长度增加智能体性能最多下降70%的现象。

    

    语言模型（LM）智能体正越来越多地被部署用于长期维护代码库，在修复一系列相关缺陷的同时，将上下文从前一次修复延续到下一次修复。然而，现有的软件工程（SWE）基准测试一次只评估一个缺陷：仓库被重置，代码库被重新读取，单个独立的问题被孤立地评分。这种设置将连续的维护工作流程简化为一系列独立的会话，忽略了使现实世界中缺陷修复具有挑战性的累积依赖关系。为弥补这一差距，我们提出了ChainSWE，这是首个用于评估智能体在共享代码库中执行顺序性、相互依赖的缺陷修复的基准测试。我们从六个SWE-bench系列数据集中挖掘，收集了涵盖54个Python项目的304个问题的时间顺序链条。我们对多种智能体和模型的评估显示，随着链条长度的增加，性能持续下降，降幅高达70%。

    arXiv:2607.02606v2 Announce Type: replace  Abstract: Language model (LM) agents are increasingly deployed to maintain codebases over extended periods, fixing streams of related defects while carrying context from one fix to the next. Yet existing software engineering (SWE) benchmarks evaluate models one bug at a time: the repository is reset, the codebase is re-read, and a single self-contained issue is graded in isolation. This setting collapses a continuous maintenance workflow into a series of independent sessions, ignoring the cumulative dependencies that make real-world bug fixing challenging. To bridge this gap, we introduce ChainSWE, the first benchmark for evaluating agents on sequential, dependent bug fixes within a shared codebase. We collect chronological chains of 304 issues across 54 Python projects, mined from six SWE-bench-family datasets. Our evaluation across a range of agents and models reveals a consistent performance drop by up to 70% as the chain length increases.
    
[^46]: 引导而非解决：为大型代码智能体训练小型评论模型

    Steer, Don't Solve: Training Small Critic Models for Large Code Agents

    [https://arxiv.org/abs/2606.21811](https://arxiv.org/abs/2606.21811)

    通过训练专门负责高层次规划的小型评论模型（4B/8B）在推理时引导大型编码智能体识别并纠正错误，在SWE-Bench Verified上显著提升多个更大规模编码智能体的解决率（最高提升16.0%）并降低推理成本。

    

    编码任务通常较为复杂，需要多种能力，涵盖从高层次规划到低层次实现的各个方面。虽然编码智能体针对这些联合能力进行了优化，但诸如高层次规划等单项能力可能有不同的最优解，并仍然是主要瓶颈。为应对这一挑战，我们训练了一个独立于编码智能体、专门擅长高层次规划的评论模型，在推理阶段对编码智能体进行引导。我们构建了SFT和DPO数据来训练该评论模型，使其能够识别编码智能体所犯的错误，并提供正确且清晰的高层次指导，而无需生成具体的操作动作。实验表明，我们微调后的4B和8B评论模型显著提升了6个更大规模编码智能体的性能（例如，在SWE-Bench Verified上，将GLM-4.7-Flash-30B-A3B和GPT-OSS-120B的解决率分别提升了16.0%和14.4%）。该评论模型还降低了总推理成本（摘要原文在此处截断）。

    arXiv:2606.21811v2 Announce Type: replace-cross  Abstract: Coding tasks are typically complicated and require multiple capabilities, ranging from high-level planning to low-level implementation. While coding agents are optimized for the joint capabilities, individual capabilities such as high-level planning may have different optima and remain a major bottleneck. To address this challenge, we train a separate critic model that is specialized in high-level planning to steer the coding agent in inference. We construct SFT and DPO data to train the critic model to identify errors made by the coding agent and provide correct and clear high-level guidance without generating concrete actions. Experiments show that our fine-tuned 4B and 8B critic models significantly improve the performance of 6 larger coding agents (e.g., improving the resolved rates of GLM-4.7-Flash-30B-A3B and GPT-OSS-120B by 16.0% and 14.4% on SWE-Bench Verified). The critic model also reduces the total inference costs fo
    
[^47]: 测试对抗变异体：基于对抗式大语言模型智能体的鲁棒单元测试生成

    Test vs Mutant: Adversarial LLM Agents for Robust Unit Test Generation

    [https://arxiv.org/abs/2602.08146](https://arxiv.org/abs/2602.08146)

    提出AdverTest框架，通过测试用例生成智能体与变异体生成智能体之间的对抗循环博弈，显著提升LLM生成单元测试在缺陷检测方面的鲁棒性。

    

    软件测试是软件开发生命周期中至关重要但资源消耗巨大的阶段。多年来，人们开发了各种自动化工具来辅助这一过程。基于搜索的方法通常能达到较高的覆盖率，但生成的测试可读性较低；而基于大语言模型（LLM）的方法生成的测试更具人类可读性，但往往存在覆盖率低和可编译性差的问题。尽管大多数研究工作都集中在提高测试覆盖率和可读性上，但对增强缺陷检测鲁棒性的关注却很少，尤其是在暴露边界用例和脆弱执行路径方面。为了填补这一空白，我们提出了AdverTest，这是一个用于LLM驱动的测试用例生成的新型对抗式框架。AdverTest包含两个相互作用的智能体：测试用例生成智能体（T）和变异体生成智能体（M）。这些智能体在一个对抗循环中相互博弈，其中M持续地...

    arXiv:2602.08146v3 Announce Type: replace  Abstract: Software testing is a critical, yet resource-intensive phase of the software development lifecycle. Over the years, various automated tools have been developed to aid in this process. Search-based approaches typically achieve high coverage but produce tests with low readability, whereas large language model (LLM)-based methods generate more human-readable tests but often suffer from low coverage and compilability. While the majority of research efforts have focused on improving test coverage and readability, little attention has been paid to enhancing the robustness of bug detection, particularly in exposing corner cases and vulnerable execution paths. To address this gap, we propose AdverTest, a novel adversarial framework for LLM-powered test case generation. AdverTest comprises two interacting agents: a test case generation agent (T) and a mutant generation agent (M). These agents engage in an adversarial loop, where M persistentl
    
[^48]: 测量计算机科学热情：基于问卷的年龄与性别对学生兴趣影响的分析

    Measuring Computer Science Enthusiasm: A Questionnaire-Based Analysis of Age and Gender Effects on Students' Interest

    [https://arxiv.org/abs/2512.08472](https://arxiv.org/abs/2512.08472)

    本研究开发了一份28条目的问卷来测量学生对计算机科学的热情，发现青少年兴趣在青春期早期显著下降（尤其女生），推翻了早期接触能保证持久兴趣的常见假设。

    

    本研究考察了年龄和性别如何独立地塑造青少年对计算机科学（CS）教育的兴趣。基于兴趣的个体-客体理论（POI），我们将热情定义为一种短期的、激活性的反应，它结合了积极情感、感知到的相关性以及再次参与的意愿。由于这种热情即使短暂也能改变对计算机科学的态度和参与意愿，因此它为短期外展活动提供了一种有用的测量手段。我们开发了一份包含28个条目的前后测问卷，以评估计算机科学干预是否能提升热情，并将其应用于400多名（244名女生、187名男生，年龄在10-18岁之间）计算机科学课程中的学生。与“早期接触能确保持久兴趣”这一常见假设相反，我们发现兴趣在青春期早期出现了显著下降，尤其是在女生中，并且不同年龄段的兴趣轨迹存在很大差异。探索性因子分析和方差分析表明，年龄能够预测兴趣的发展。

    arXiv:2512.08472v2 Announce Type: replace  Abstract: This study examines how age and gender independently shape adolescents' interest in computer science (CS) education. Building on the Person-Object Theory of Interest (POI), we define enthusiasm as a short-term, activating response that combines positive affect, perceived relevance, and intention to re-engage. Because such enthusiasm can shift CS attitudes and engagement intentions even briefly, it offers a useful measure for short outreach activities.   We developed a 28-item pre-post questionnaire to assess whether CS interventions raise enthusiasm, then applied it to more than 400 students (244 female, 187 male, aged 10-18) in CS courses. Contrary to the common assumption that early exposure secures lasting interest, we found a marked decline during early adolescence, especially among girls, along with wide variation in interest trajectories across ages.   Exploratory factor analysis and ANOVA show that age predicts interest develo
    
[^49]: 多智能体大语言模型编排为事件响应实现确定性、高质量决策支持

    Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support for Incident Response

    [https://arxiv.org/abs/2511.15755](https://arxiv.org/abs/2511.15755)

    该论文提出MyAntFarm.ai框架，通过348次受控试验证明多智能体LLM编排相比单智能体方法可将可操作建议率从1.7%提升至100%，实现行动具体性提升80倍、方案正确性提升140倍且质量零方差的确定性事件响应决策支持。

    

    大语言模型（LLM）有望加速生产系统中的事件响应，但单智能体方法往往生成模糊、不可用的建议。我们提出了MyAntFarm.ai，这是一个可复现的容器化框架，证明了多智能体编排从根本上改变了基于LLM的事件响应质量。通过348次受控试验，在相同事件场景下比较单智能体副驾驶与多智能体系统，我们发现多智能体编排实现了100%的可操作建议率，而单智能体方法仅为1.7%，行动具体性提升80倍，解决方案正确性提升140倍。至关重要的是，多智能体系统在所有试验中表现出零质量方差，这使得生产环境SLA承诺成为可能，而这是不一致的单智能体输出无法实现的。两种架构实现了相似的理解延迟（约40秒），表明该架构……

    arXiv:2511.15755v3 Announce Type: replace  Abstract: Large language models (LLMs) promise to accelerate incident response in production systems, yet single-agent approaches generate vague, unusable recommendations. We present MyAntFarm.ai, a reproducible containerized framework demonstrating that multi-agent orchestration fundamentally transforms LLM-based incident response quality. Through 348 controlled trials comparing single-agent copilot versus multi-agent systems on identical incident scenarios, we find that multi-agent orchestration achieves 100% actionable recommendation rate versus 1.7% for single-agent approaches, an 80 times improvement in action specificity and 140 times improvement in solution correctness. Critically, multi-agent systems exhibit zero quality variance across all trials, enabling production SLA commitments impossible with inconsistent single-agent outputs. Both architectures achieve similar comprehension latency (approx.40s), establishing that the architectu
    
[^50]: 糟糕！……我又重蹈覆辙了：分析并处理社会技术软件工程中结论的（不）稳定性

    Oops!... I did it again. Analysing and Handling Conclusion (In-)Stability in Socio-Technical Software Engineering

    [https://arxiv.org/abs/2510.06844](https://arxiv.org/abs/2510.06844)

    本研究通过用四个独立挖掘工具正式复现三项已发表的社会技术软件分析研究，揭示了不同工具流水线在数据、结果与结论上的不稳定性，并为研究者和从业者提供了应对此类效度威胁的可操作建议。

    

    背景：软件仓库挖掘是一种常用手段，可用于洞察软件项目的演进、监控项目健康状态、支持决策并总结最佳实践。支持挖掘流程的工具被研究者和从业者广泛使用，但人们对这些工具的局限性以及它们之间的一致性往往缺乏充分理解。目的：本研究考察了用于演进式社会技术软件分析的复杂工具流水线中的一些效度威胁，并针对相同的研究问题评估各工具在数据、研究结果和结论层面的一致性，从而为研究者和从业者提出可操作的建议。方法：我们开展了一项轻量级文献综述，从高水平学术会议和期刊中选取三项关于协作与协调、软件维护及软件质量的研究，并使用四个独立、系统挑选的挖掘工具对其进行正式复现，以量化……（原文摘要至此截断）

    arXiv:2510.06844v3 Announce Type: replace  Abstract: Context: Mining software repositories is a popular means to gain insights into a software project's evolution, monitor project health, support decisions and derive best practices. Tools supporting the mining process are commonly applied by researchers and practitioners, but their limitations and agreement are often not well understood.   Objective: This study investigates some threats to validity in complex tool pipelines for evolutionary socio-technical software analyses. We evaluate the tools' agreement in terms of data, study outcomes and conclusions for the same research questions to derive actionable advice for researchers and practitioners.   Method: We conduct a lightweight literature review to select \emph{three} studies on collaboration and coordination, software maintenance and software quality from high-ranked venues, which we formally replicate with \emph{four} independent, systematically selected mining tools to quantita
    
[^51]: Essence Coach：一个用于软件实践采纳的聊天机器人

    Essence Coach: A Bot for Software Practice Adoption

    [https://arxiv.org/abs/2508.16445](https://arxiv.org/abs/2508.16445)

    该论文提出 Essence Coach——一个结合大语言模型与检索增强生成（RAG）技术的聊天机器人，可作为教练工具帮助学习者和从业者理解并采纳 Essence 软件工程框架。

    

    尽管 Essence 已被提出作为理解和评估软件工程实践的统一框架，但其实际采纳仍然充满挑战，因此需要能够为学习者和从业者充当教练角色的工具。我们提出了 Essence Coach，这是一个将大语言模型（LLM）与基于精选 Essence 知识库的检索增强生成（RAG）相结合的聊天机器人。在多个大语言模型上的实验表明，RAG 能够持续提升针对 Essence 相关问题的回答质量。一些初步测试表明，此类系统可以帮助技术背景和非技术背景的学生理解并应用 Essence 概念。虽然还需要通过更广泛的用户研究进行验证，但这项正在进行中的工作凸显了大语言模型如何能够弥合抽象软件工程框架与实践采纳之间的鸿沟。

    arXiv:2508.16445v2 Announce Type: replace  Abstract: Although Essence has been proposed as a unifying framework for understanding and evaluating software engineering practices, its adoption remains challenging, calling for tools that can act as coaches for learners and practitioners. We present Essence Coach, a chatbot that integrates LLMs with retrieval-augmented generation (RAG) from a curated Essence knowledge base. Experiments with multiple LLMs show that RAG consistently improves response quality for Essence-related queries. Some preliminary tests suggest that such systems can help both technical and non-technical students understand and apply Essence concepts. While validation through broader user studies is needed, this work-in-progress highlights how LLMs can bridge abstract SE frameworks and practice adoption.
    
[^52]: ParaStudent：缩小AI导师评估中用户模拟器的模拟到现实差距

    ParaStudent: Closing the Sim2Real Gap in User Simulators for AI Tutor Evaluation

    [https://arxiv.org/abs/2507.12674](https://arxiv.org/abs/2507.12674)

    ParaStudent是一个通过微调来模拟初学者编程修改的框架，其模拟结果更贴近真实学生代码分布，可用于AI导师部署前的反馈评估与筛选。

    

    在部署前评估人工智能（AI）导师的反馈需要预测学生的参与度，这通常通过真实交互数据来评估。我们提出了ParaStudent，一个用于模拟初学者编程修改的微调框架，以支持AI导师评估。与基于提示的基线方法相比，ParaStudent的模拟修改在功能性、风格性和语义性指标上都更接近真实学生的代码分布。我们表现最佳的变体在区分真实参与度高于中位数与等于或低于中位数的交互流时，在反馈相关性和成功采纳两方面均达到0.80的AUC，而基于提示的基线在成功采纳方面的表现仍接近随机水平。这些发现展示了模拟参与度在AI导师部署前反馈筛选中的应用前景。

    arXiv:2507.12674v3 Announce Type: replace-cross  Abstract: Evaluating Artificial Intelligence (AI) tutor feedback before deployment requires anticipating student engagement, typically assessed through real interaction data. We introduce ParaStudent, a fine-tuning framework for simulating novice programming revisions to support AI tutor evaluation. Compared with prompted baselines, ParaStudent's revisions more closely match real student code distributions across functional, stylistic, and semantic metrics. Our best variant achieves AUCs of 0.80 for both feedback relevance and successful uptake when distinguishing streams with real engagement above versus at or below the median, while prompted baselines remain near chance on successful uptake. These findings demonstrate the promise of simulated engagement for pre-deployment feedback triage.
    
[^53]: 软件安全中的流行度假设：基于PHP包的大规模复现研究

    The Popularity Hypothesis in Software Security: A Large-Scale Replication with PHP Packages

    [https://arxiv.org/abs/2502.16670](https://arxiv.org/abs/2502.16670)

    本文通过对近40万个PHP开源软件包和6000多个WordPress组件的大规模复现分析，验证了软件安全领域的流行度假设，即出现过漏洞报告的软件包通常比漏洞较少或没有漏洞的软件包更流行。

    

    长期以来，无论在学术研究还是大众讨论中都存在一个假设，即软件的流行度与其安全性或不安全性相关。已有一些实证研究或明或暗地对这一假设进行过检验。本工作延续并贡献于这一研究方向，对用PHP编程语言编写的软件进行了以复现为动机的大规模分析。研究使用了两个数据集：第一个包含近四十万个用PHP编写的开源软件包，第二个涉及六千多个WordPress组件。基于已报告漏洞的研究结果表明，该假设成立：在发布历史中出现过漏洞报告的软件包，通常比漏洞报告较少或没有漏洞报告的软件包更为流行。通过这一复现结果，本文为加强相关研究的工作做出了贡献。

    arXiv:2502.16670v3 Announce Type: replace  Abstract: There has been a long-standing hypothesis that a software's popularity is related to its security or insecurity in both research and popular discourse. There are also a few empirical studies that have examined the hypothesis, either explicitly or implicitly. The present work continues with and contributes to this research with a replication-motivated large-scale analysis of software written in the PHP programming language. Two datasets are used: the first contains nearly four hundred thousand open source software packages written in PHP and the second addresses over six thousand WordPress components. According to the results based on vulnerabilities reported, the hypothesis holds: packages having seen reported vulnerabilities over their release histories are generally more popular than packages for which fewer or no vulnerabilities have been reported. With this replication results, the paper contributes to the efforts to strengthen t
    

