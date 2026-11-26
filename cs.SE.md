# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Much Do Coding Agents Over-Edit, and Why? An Empirical Study of Unneeded Edits in Repository-Level Agent Patches](https://arxiv.org/abs/2611.34875) | 该论文提出了Delta残留归因（DRA）方法，通过重放代理轨迹、逐单元回滚补丁并重跑测试来判定每个编辑的必要性，实证发现编码代理的代码编辑平均必要性与开发者相当，其“过度编辑”主要源于额外添加的测试和文档而非冗余代码。 |
| [^2] | [Shape-Aware Three-Way Merge for RDF Knowledge Graphs: The schema defines the conflict: an implementation and evaluation](https://arxiv.org/abs/2611.34870) | 本文提出一种由SHACL形状定义冲突的RDF知识图谱三方合并方法，在生产级多智能体存储中实现并评估，结果表明其提出的33个冲突全部为真阳性，且不丢失、不伪造三元组、不引入SHACL违规。 |
| [^3] | [Multi-Party Authorization of Maintenance Prescriptions on a Permissioned Blockchain](https://arxiv.org/abs/2611.34662) | 该论文提出要求许可链背书节点在发放工单前验证维护处方计算的简洁证明，从而在多组织协作的工业预防性维护中成功拦截了全部有争议或操作不当的处方，而传统哈希校验会漏掉其中近一半的风险处方。 |
| [^4] | [Evaluating More with Less: Black-Box Test Selection for AI Agents](https://arxiv.org/abs/2611.34617) | 本文首次对智能体软件系统的黑盒测试用例选择进行了实证研究，提出基于任务描述、元数据、代码片段、嵌入向量及任务难度估计的选择策略，无需执行目标智能体即可降低测试成本并有效评估智能体性能。 |
| [^5] | [SmartLens: Turning AI Agent Changes into Learning, Review, and Risk Signals in the IDE](https://arxiv.org/abs/2611.34377) | SmartLens 是一个 IntelliJ IDEA 插件，通过将待提交的代码变更归属到 AI 智能体或开发者，并结合 Git 历史，为开发者提供三项能力：学习智能体使用的新技能、按优先级审查待提交变更、以及识别由 AI 主导编写且开发者不熟悉的高风险热点文件。 |
| [^6] | [SimFusion: Multi-Fidelity Simulation-based Testing of Automated Driving Systems](https://arxiv.org/abs/2611.34202) | SimFusion提出了一种多保真度搜索测试框架，通过学习LoFi与HiFi仿真器对场景结果的判断是否一致来智能分配执行仿真器，从而在保证结果可靠性的前提下显著提升了自动驾驶系统测试的效率与故障发现能力。 |
| [^7] | [Reproducer Synthesis for AI-Reported Bugs in the Linux Kernel](https://arxiv.org/abs/2611.34142) | ReproCue 框架通过重放历史内核缺陷并从中学习各子系统的复现有限状态机，为 AI 生成的 Linux 内核缺陷审查报告自动合成有效且可运行的验证程序（PoC），从而帮助维护者快速确认缺陷报告的真实性。 |
| [^8] | [What Does Skill Lift Actually Measure? A Source-Code-Level Audit and Noise-Aware Reanalysis of SkillsBench](https://arxiv.org/abs/2611.33943) | 该论文对 SkillsBench 进行源代码级审计并对 7,554 次公开试验做噪声感知重分析，发现部分“技能提升”分数实际上源于评分器依赖和操作性定义缺口等测量缺陷，并将总体平均提升修正为 18.7 pp。 |
| [^9] | [REFORGE: LLM-Assisted RTL Bug Repair with Structured Context and Verification Feedback](https://arxiv.org/abs/2611.33748) | REFORGE是首个通过跨迭代维护与积累结构化修复证据、结合验证引导诊断的LLM辅助RTL缺陷修复框架，能够生成同时满足功能性、行为等价性、安全性和成本效率等多目标的完整RTL修复方案。 |
| [^10] | [How Much Effort Does an AI Code Reviewer Need? Evidence from Claude 5.5 Models on Plausible-but-Wrong Patches](https://arxiv.org/abs/2611.33639) | 通过对Claude 5.5系列模型审查90个SWE-bench补丁的预注册实验，本研究发现更高的推理努力程度能显著减少AI代码审查者对似是而非错误补丁的误接受，且对Opus而言中等努力程度能以低20%的成本达到与高等努力程度相当的效果。 |
| [^11] | [Cross-Layer Characterization and AI-Assisted Diagnosis of C-to-Rust Performance](https://arxiv.org/abs/2611.33490) | 该论文通过对12个C/Rust基准程序的跨层表征揭示了运行时安全检查并非Rust迁移性能开销的主因（编译器与微架构优化常可缓解其成本），并据此提出了集成源码分析、编译器优化提示与动态插桩的AI辅助性能调试器FerroScope。 |
| [^12] | [DevEx is for Humans, AX is for Agents: Measuring Context Health in Agentic Software Work](https://arxiv.org/abs/2611.33480) | 该论文提出智能体体验（AX）框架，通过从清晰度、可操作性、保真度和效率四个维度对智能体会话中的上下文质量进行评分，揭示人类提供的上下文质量（尤其是可操作性）是决定编码智能体成败的关键因素。 |
| [^13] | [Designing an Integrated Learning Ecosystem for Introductory Python Programming: Scaffolded Problem Solving, Tablet-Inspired Navigation, and AI-Enhanced Instruction](https://arxiv.org/abs/2611.33471) | 本文为Python入门课程设计并分析了一个集成学习生态系统，其中15个实验课共享IPO推理、教师指导编程、独立任务、Colab/Gradio扩展、调试支持、任务清单与提交要求七大支架要素，并以Pygame综合大作业和AI增强教学、平板风格导航作为支撑。 |
| [^14] | [GhostAPI-V: Evaluating Audited Diagnostics for Internal API Migration Repair](https://arxiv.org/abs/2611.33396) | GhostAPI-V提出以经人工审计的时序契约为API迁移修复提供候选诊断，实验证明完整双向契约反馈可将隐藏测试成功率从62.5%提升至77.1%。 |
| [^15] | [Building a Unified Inline Code Editing Model for GitHub Copilot](https://arxiv.org/abs/2611.33314) | 本文为GitHub Copilot提出统一行内代码编辑模型，通过锚定差异补丁的单一输出空间一次性预测光标补全、邻近编辑与远距离编辑三种类型，支持单次响应包含多个编辑，避免了原系统多次顺序调用不同模型的开销。 |
| [^16] | [Real Package, Wrong Controller: Static Auditing of Vendor--Protocol Mismatches in LLM-Generated PLC Code](https://arxiv.org/abs/2611.33227) | 论文提出了Proteus-OT，一个基于版本固定绑定图谱的无标签静态验证器，用于检测大语言模型生成的PLC代码中导入包与声明工业控制器目标之间的供应商-协议不匹配，发现883个生成脚本中有20.4%存在此类导入级别不兼容问题。 |
| [^17] | [Demystifying Real-world LLM Service Freeloading](https://arxiv.org/abs/2611.33218) | 本文通过对513个开源项目、64个上游LLM服务和1,588条推理路径的系统性测量研究，首次揭示了第三方项目通过复刻产品协议将免费LLM网页服务打包为独立API的“服务白嫖”生态的规模、依赖与运行机制。 |
| [^18] | [Multi-Agent Coordinated Method-Level Refactoring of Rust Code](https://arxiv.org/abs/2611.29321) | 该论文提出了一种基于大语言模型的多智能体协同工作流，通过候选生成、可配置专家审查与容器化验证三个环节，安全地实现 Rust 代码的方法提取、内联与移动重构，并在开源项目任务上首次尝试即全部通过验证。 |
| [^19] | [Fixed or Deleted? External Feedback Lets Cheaper AI Coding Agents Repair Rule Violations](https://arxiv.org/abs/2611.29162) | 研究发现，引入商业检查器CodeAssure等外部反馈（报告违规位置和修复目标）能大幅提升AI编程智能体修复项目规则违规的成功率，使更小更便宜的模型（修复率从26%升至69%）接近甚至逼近大型模型的修复效果。 |
| [^20] | [From Migration to Calibration: Preserving Agent Capabilities across Models, Jurisdictions, and Scale](https://arxiv.org/abs/2609.35149) | 提出将智能体校准构建为标准优先的适配框架，通过定义基础能力、技术环境与用户情境标准，系统诊断差距并应用修订，从而在模型更换、跨管辖区部署和规模化过程中保持智能体能力不退化。 |
| [^21] | [SpecRead: A Benchmark for Measuring Whether Language Models Understand Hardware Specifications](https://arxiv.org/abs/2609.33699) | SpecRead是一个将硬件规格理解能力与代码生成能力分离的新基准测试，通过针对OpenTitan IP模块的四类题目（精确检索、跨章节推理、矛盾检测和规格-RTL一致性检查）来评估大语言模型对硬件规格的真实理解程度。 |
| [^22] | [How Powerful are LLMs in Generating Formal Program Specifications?](https://arxiv.org/abs/2608.13077) | 本文提出了Coins框架，通过实例化规范并生成具体证明义务来评估LLM生成形式化程序规范的质量，避免了传统方法中证明难度与规范质量混淆的问题。 |
| [^23] | [AiFlow: Construction-to-Retirement Conformance for Versioned LLM Services](https://arxiv.org/abs/2608.00558) | AiFlow 提出一种可执行契约机制，将版本化 LLM 服务的构建、调用与退役全流程串联，通过证书认证确保调用的终结结果与注册资源的释放，并支持严格发布和延迟发布两种模式实现版本安全退役。 |
| [^24] | [The Illusion of Safety: Multi-Tier Verification of AI vs. Human C++ Code](https://arxiv.org/abs/2607.00107) | 该论文构建了包含8,918个C++程序的VULBENCH-CPP基准，通过功能测试、静态分析、动态分析和有界模型检验四个层级系统对比AI与人类编写的代码，揭示了仅依赖静态分析会带来“安全的假象”。 |
| [^25] | [Agentic Model Checking](https://arxiv.org/abs/2605.21434) | 本文提出智能体化模型检测及其实例 BMC-Agent，遵循“智能体提出、求解器验证”的原则，让 LLM 智能体负责规范推断与验证规划等语义推理，由有界模型检测器进行形式化验证，从而实现对大型软件系统的自动验证。 |
| [^26] | [A Threshold Model of Fault Propagation in AI-Native Software Ecosystems](https://arxiv.org/abs/2604.19827) | 本文将复杂性理论与软件演化相结合，把AI编码智能体引入的缺陷在开放变更间的传播建模为基于争用关系的分支过程，并证明每个仓库存在一个使故障传播自我维持的临界阈值。 |
| [^27] | [DRAFT-ing Architectural Design Decisions using LLMs](https://arxiv.org/abs/2504.08207) | 该论文提出了DRAFT方法，将检索增强少样本生成与微调相结合，使大语言模型能够根据决策上下文生成架构设计决策，并根据标题生成完整的架构决策记录。 |

# 详细

[^1]: 编码代理的过度编辑有多少，原因何在？关于仓库级代理补丁中不必要编辑的实证研究

    How Much Do Coding Agents Over-Edit, and Why? An Empirical Study of Unneeded Edits in Repository-Level Agent Patches

    [https://arxiv.org/abs/2611.34875](https://arxiv.org/abs/2611.34875)

    该论文提出了Delta残留归因（DRA）方法，通过重放代理轨迹、逐单元回滚补丁并重跑测试来判定每个编辑的必要性，实证发现编码代理的代码编辑平均必要性与开发者相当，其“过度编辑”主要源于额外添加的测试和文档而非冗余代码。

    

    代理补丁比开发者的更大，这被广泛解读为过度编辑。但补丁大小并不能说明额外的编辑是否必要、它们是什么、或它们来自哪里。我们提出了Delta残留归因（DRA）方法，该方法重放代理轨迹，将最终补丁的每个块（单元）追溯到写入它的步骤，回滚每个单元，并重新运行任务测试以判断该单元是否必要。我们将DRA应用于来自18个前沿模型在SWE-bench Verified（6,728条）和DeepSWE（932条）上的7,660条已解决的GitHub Copilot CLI轨迹，并用相同方法对开发者补丁进行评级。代理补丁确实比开发者的更改更多行；在SWE-bench Verified上，大多数额外编辑是增加测试（61%）和文档（13%），而在DeepSWE上，大多数额外编辑是代码（58%）。在两个基准测试上，代理的代码平均而言至少与开发者的代码一样必要（在SWE-bench Verified上平均NP_code为0.887对比0.864，其中1表示全部必要）……

    arXiv:2611.34875v1 Announce Type: cross  Abstract: Agent patches are larger than developers', which is widely read as over-editing. But patch size does not say whether the extra edits are needed, what they are, or where they come from. We introduce Delta Residue Attribution (DRA), which replays agent trajectories, traces every block (unit) of the final patch to the step that wrote it, reverts each unit, and reruns the task tests to decide whether the unit is needed. We apply DRA to 7,660 resolved GitHub Copilot CLI trajectories from 18 frontier models on SWE-bench Verified (6,728) and DeepSWE (932), and grade developer patches using the same method. Agent patches do change more lines than developers'; on SWE-bench Verified the majority of extra edits add tests (61%) and documentation (13%) while on DeepSWE, most extra edits are code (58%). On both benchmarks, agents' code is on average at least as necessary as developers' (mean $NP_{code}$ 0.887 vs. 0.864 on SWE-bench Verified, where 1
    
[^2]: 面向RDF知识图谱的形状感知三方合并：由模式定义冲突的实现与评估

    Shape-Aware Three-Way Merge for RDF Knowledge Graphs: The schema defines the conflict: an implementation and evaluation

    [https://arxiv.org/abs/2611.34870](https://arxiv.org/abs/2611.34870)

    本文提出一种由SHACL形状定义冲突的RDF知识图谱三方合并方法，在生产级多智能体存储中实现并评估，结果表明其提出的33个冲突全部为真阳性，且不丢失、不伪造三元组、不引入SHACL违规。

    

    合并RDF序列化会产生顺序冲突并遗漏语义冲突。我们描述并评估了一种由模式定义冲突的三方合并方法：使用集合代数合并规范化三元组，仅当SHACL形状约束了某个槽位的基数、且双方的修改无法在该约束范围内共存时，该槽位才被视为有争议。无基数限制的谓词取并集；有争议的槽位则交由人工处理。同一形状图可用于验证合并结果。我们并不声称这是首个针对RDF的三方合并。我们的贡献是在生产级多智能体存储中的具体实现，以及对Quit Store的Context Merge重叠规则（我们对其进行了扩展而非替代）的重新实现和另外六种策略的评估。在五个合成种子数据集上，形状感知算子提出了33个冲突，经定义这些冲突的相同形状判定全部为真阳性，且没有三元组丢失或伪造，也未引入任何SHACL违规。上下文重叠基线……

    arXiv:2611.34870v1 Announce Type: cross  Abstract: Merging RDF serializations creates ordering conflicts and misses semantic ones. We describe and evaluate a three-way merge in which the schema defines a conflict: set algebra combines canonical triples, and a slot is contended only where a SHACL shape bounds its cardinality and both sides' changes cannot stand within that bound. Unbounded predicates union; contended slots go to a person. The same shapes graph validates the result. We do not claim the first three-way merge for RDF. Our contribution is an implementation in a production multi-agent store and an evaluation against a reimplementation of Quit Store's Context Merge overlap rule, which it extends rather than replaces, and six other strategies. Across five synthetic seeds, the shape-aware operator raised 33 conflicts, all true positives as judged by the same shapes that define them, with no triples lost or fabricated and no SHACL violations admitted. The context-overlap baselin
    
[^3]: 许可区块链上维护处方的多方授权

    Multi-Party Authorization of Maintenance Prescriptions on a Permissioned Blockchain

    [https://arxiv.org/abs/2611.34662](https://arxiv.org/abs/2611.34662)

    该论文提出要求许可链背书节点在发放工单前验证维护处方计算的简洁证明，从而在多组织协作的工业预防性维护中成功拦截了全部有争议或操作不当的处方，而传统哈希校验会漏掉其中近一半的风险处方。

    

    在工业预防性维护中，设备供应商的模型可能会推荐一项干预措施，由工厂进行排期、承包商执行、并由质量或安全部门批准。共享账本虽然记录了谁提交了该维护处方，却无法核查它是否由约定的模型和决策规则所产生。我们要求许可区块链的背书节点在发放工单之前先验证该计算的简洁证明。在一个使用公开发动机基准测试的四组织 Hyperledger Fabric 测试平台上，该方法阻止了全部九条有争议或操作不当的维护处方，而传统的哈希校验却接受了其中四条——这些处方要么提高了模拟维护成本，要么重新引入了延迟故障。

    arXiv:2611.34662v1 Announce Type: cross  Abstract: In industrial prescriptive maintenance, an equipment supplier's model may recommend an intervention that a plant schedules, a contractor executes, and a quality or safety function approves. A shared ledger records who submitted the prescription, yet cannot check whether the agreed model and decision rule produced it. We require permissioned-blockchain endorsers to verify a succinct proof of that calculation before releasing a work order. On a four-organization Hyperledger Fabric testbed using a public engine benchmark, this stopped all nine contested or mis-operated prescriptions, whereas hash checks accepted four that either raised simulated maintenance cost or reintroduced late failures.
    
[^4]: 以少胜多：面向 AI 智能体的黑盒测试选择

    Evaluating More with Less: Black-Box Test Selection for AI Agents

    [https://arxiv.org/abs/2611.34617](https://arxiv.org/abs/2611.34617)

    本文首次对智能体软件系统的黑盒测试用例选择进行了实证研究，提出基于任务描述、元数据、代码片段、嵌入向量及任务难度估计的选择策略，无需执行目标智能体即可降低测试成本并有效评估智能体性能。

    

    智能体软件系统在开发中的应用日益增多，而模型、提示词、脚手架、工具或执行环境的任何变更都需要进行反复测试。测试用例选择（TCS）可以在降低测试成本的同时，估计智能体的性能或尽早暴露故障。尽管针对深度神经网络和大语言模型代码生成模型的测试用例选择已有研究，但先前的工作并未直接针对智能体系统，且通常需要专有智能体无法获取的模型内部信息。我们提出了首个针对智能体软件系统的黑盒测试用例选择实证研究。我们将每个基准测试任务视为一个测试用例，基于任务描述、元数据、代码片段和嵌入向量等选择标准，对黑盒回归测试技术加以改编。我们还提出了利用启发式特征或独立的 LLM 评审来估计任务难度的策略。我们的策略无需执行目标智能体即可完成测试任务的选择。

    arXiv:2611.34617v1 Announce Type: new  Abstract: Agentic software systems are increasingly used in development, where changes to the model, prompt, scaffold, tools, or execution environment require repeated testing. Test case selection (TCS) can reduce testing costs while providing estimates of agent performance or exposing failures early. Although TCS has been studied for deep neural networks and LLM code models, prior work does not directly address agentic systems and often requires model information unavailable for proprietary agents. We present the first empirical study of black-box TCS for agentic software systems. We treat each benchmark task as a test case and adapt black-box regression-testing techniques using selection criteria based on task descriptions, metadata, code fragments, and embeddings. We also propose strategies that estimate task difficulty using heuristic features or independent LLM-as-judge assessments. Our strategies select tasks without executing the target age
    
[^5]: SmartLens：在 IDE 中将 AI 智能体的代码变更转化为学习、审查与风险信号

    SmartLens: Turning AI Agent Changes into Learning, Review, and Risk Signals in the IDE

    [https://arxiv.org/abs/2611.34377](https://arxiv.org/abs/2611.34377)

    SmartLens 是一个 IntelliJ IDEA 插件，通过将待提交的代码变更归属到 AI 智能体或开发者，并结合 Git 历史，为开发者提供三项能力：学习智能体使用的新技能、按优先级审查待提交变更、以及识别由 AI 主导编写且开发者不熟悉的高风险热点文件。

    

    开发者越来越多地扮演着 AI 智能体管理者的角色：他们将任务交给 AI 编程智能体，然后需要理解、验证、发布智能体的产出并从中学习。现有的 IDE 和近期的一些工具虽然能够显示哪些已提交的代码行是由 AI 编写的，但它们既无法对外部智能体产生的待提交变更进行归属，也没有利用这种归属信息来帮助开发者进行审查、学习或风险判断。我们提出了 SmartLens，这是一个 IntelliJ IDEA 插件，能够将代码变更归属到 AI 智能体、开发者或两者。它将这种归属信息与 Git 历史相结合，支持三大目标：(i) 学习功能比较智能体所使用的技能（API、语言特性、注解）与开发者在自己提交中所使用的技能；(ii) 理解功能按审查优先级对待提交变更进行排序；(iii) 风险功能展示热点文件，即变更频繁、主要由 AI 编写且开发者几乎不熟悉的文件。在初步评估中，SmartLens 的归属（摘要在此处被截断）

    arXiv:2611.34377v1 Announce Type: new  Abstract: Developers increasingly act as managers of AI agents: they hand tasks to AI coding agents and then have to understand, verify, ship, and learn from what the agents produce. IDEs and recent tools can show which committed lines were written by AI, but they do not attribute the pending changes of external agents, and they do not use attribution to help the developer review, learn, or judge risk. We present SmartLens, an IntelliJ IDEA plugin that attributes changes to an AI agent, the developer, or both. It combines this attribution with the Git history to support three goals: (i) Learn compares the skills (APIs, language features, annotations) that agents used with the skills the developer has used in their own commits; (ii) Understand ranks pending changes by review priority; and (iii) Risk shows hotspots: files that change often, are mostly written by AI, and that the developer barely knows. In a preliminary evaluation, SmartLens attribut
    
[^6]: SimFusion：基于多保真度仿真的自动驾驶系统测试

    SimFusion: Multi-Fidelity Simulation-based Testing of Automated Driving Systems

    [https://arxiv.org/abs/2611.34202](https://arxiv.org/abs/2611.34202)

    SimFusion提出了一种多保真度搜索测试框架，通过学习LoFi与HiFi仿真器对场景结果的判断是否一致来智能分配执行仿真器，从而在保证结果可靠性的前提下显著提升了自动驾驶系统测试的效率与故障发现能力。

    

    仿真是验证自动驾驶系统（ADS）的主要手段。测试通常依赖于单一仿真器，这迫使人们需要在低保真度（LoFi）仿真的速度与计算开销高昂的高保真度（HiFi）仿真的真实性之间做出权衡。我们提出了SimFusion，这是一个多保真度、基于搜索的测试框架，它可以为每个候选场景决定应由哪个仿真器来执行。SimFusion并不去近似ADS本身或其适应度，而是学习LoFi与HiFi仿真器在场景结果上是否可能达成一致：预测会达成一致的场景在快速的LoFi仿真器中运行，而所有其他场景则保守地在HiFi仿真器中运行。我们在两个ADS技术栈、三个驾驶场景和四个仿真器上对SimFusion进行了评估，并与仅使用LoFi、仅使用HiFi以及代理辅助的测试方法进行了比较。在每种配置运行六次的情况下，SimFusion平均比仅使用HiFi的测试多发现2.1倍经HiFi验证的故障。

    arXiv:2611.34202v1 Announce Type: new  Abstract: Simulation is the primary means of validating automated driving systems (ADS). Testing typically relies on a single simulator, forcing a trade-off between the speed of low-fidelity (LoFi) simulation and the realism of computationally expensive high-fidelity (HiFi) simulation. We present SimFusion, a multi-fidelity search-based testing framework that decides, for every candidate scenario, which simulator should execute it. Rather than approximating the ADS or its fitness, SimFusion learns whether LoFi and HiFi simulators are likely to agree on a scenario's outcome: scenarios predicted to agree run in the fast LoFi simulator, while all others are conservatively run in HiFi. We evaluate SimFusion on two ADS stacks, three driving scenarios, and four simulators, against LoFi-only, HiFi-only, and surrogate-assisted testing. Across six runs per configuration, SimFusion finds on average 2.1x more HiFi-validated failures than HiFi-only testing an
    
[^7]: 面向 Linux 内核中 AI 报告缺陷的复现程序自动合成

    Reproducer Synthesis for AI-Reported Bugs in the Linux Kernel

    [https://arxiv.org/abs/2611.34142](https://arxiv.org/abs/2611.34142)

    ReproCue 框架通过重放历史内核缺陷并从中学习各子系统的复现有限状态机，为 AI 生成的 Linux 内核缺陷审查报告自动合成有效且可运行的验证程序（PoC），从而帮助维护者快速确认缺陷报告的真实性。

    

    智能体审查系统正日益以文本报告的形式标记 Linux 内核补丁中疑似存在的缺陷。收到此类报告的维护者需要进一步检查才能确定报告是否属实。这一检查需要了解该缺陷是否能够被触发，如有必要还需手动构造触发方式。因此，尽管报告中包含大量文字说明，许多报告仍然得不到回应。在本文中，我们认为可运行的验证性测试程序是该类 AI 生成的内核审查报告中的关键信息。然而，自动合成一个有效且可运行的 PoC 是极具挑战性的。我们提出了 ReproCue，一个为 AI 生成的 Linux 内核审查报告生成 PoC 的框架。ReproCue 通过重放历史内核缺陷，从成功运行中为每个子系统学习一个复现有限状态机（FSM）。该 FSM 记录了复现过程所经历的步骤，以及观察到的相关需求条件……（原文摘要在此处被截断）

    arXiv:2611.34142v1 Announce Type: cross  Abstract: Agentic review systems are increasingly flagging suspected bugs in Linux kernel patches as textual reports. A maintainer who receives such reports needs further inspection to determine whether the report is true. The inspection requires understanding whether the bug can be triggered and, if necessary, manually constructing one. As a consequence, despite the excessive textual explanation, many reports go unanswered. In this paper, we argue that a runnable proof-of-concept (PoC) is the key information in such AI-generated kernel review reports. It is, however, challenging to automatically synthesize a valid and runnable PoC.   We present ReproCue, a framework that generates PoCs for AI-generated Linux kernel reviews. ReproCue replays the historical kernel bugs and learns one reproduction finite-state-machine (FSM) per subsystem from successful runs. An FSM records the steps a reproduction goes through, together with the requirements obse
    
[^8]: Skill Lift 到底衡量的是什么？对 SkillsBench 的源代码级审计与噪声感知重分析

    What Does Skill Lift Actually Measure? A Source-Code-Level Audit and Noise-Aware Reanalysis of SkillsBench

    [https://arxiv.org/abs/2611.33943](https://arxiv.org/abs/2611.33943)

    该论文对 SkillsBench 进行源代码级审计并对 7,554 次公开试验做噪声感知重分析，发现部分“技能提升”分数实际上源于评分器依赖和操作性定义缺口等测量缺陷，并将总体平均提升修正为 18.7 pp。

    

    智能体技能是编程智能体按需加载的指令、参考资料和脚本所组成的目录。根据 SkillsBench 的报告，尽管人工策划的技能平均可将通过率提高约 16.6 个百分点（pp），但它们实际上在某些任务上反而降低了成绩。后续研究已开始依据上述每任务收益来门控或剔除技能。我们在源代码层面对该基准进行了审计，并分析了 7,554 次公开试验，形成 1,421 个配对的“（配置，任务）”单元格。源代码审计发现了两例经对照验证的评分器依赖问题和一个操作性定义缺口；这三个任务上的平均提升为 74.6 pp，而完整任务集的总体平均提升为 18.7 pp。审计还发现四个任务包含占位符技能目录，且某个执行框架在大多数运行中额外注入了 48 个内置技能。没有任务层面的降低……（摘要在此处截断）

    arXiv:2611.33943v1 Announce Type: new  Abstract: Agent skills are directories of instructions, reference materials, and scripts loaded on demand by coding agents. According to SkillsBench, although human-curated skills can raise the passing rate by about 16.6 percentage points (pp) on average, they have actually reduced the results for some tasks. Later research has started to gate or eliminate skills according to the above per-task gains. We have audited the above benchmark at the source-code level and analysed 7,554 public trials to form 1,421 paired "(configuration, task) cells". Source-code audit identified two controlled-verified cases of grader dependence and one operational definition gap; the average lift on these three tasks was 74.6 pp, and the overall average lift for the full set was 18.7 pp. The audit also found that four tasks contained placeholder skill directories, and a certain execution harness additionally added 48 built-in skills in most runs. No task-level reductio
    
[^9]: REFORGE：基于结构化上下文与验证反馈的LLM辅助RTL缺陷修复

    REFORGE: LLM-Assisted RTL Bug Repair with Structured Context and Verification Feedback

    [https://arxiv.org/abs/2611.33748](https://arxiv.org/abs/2611.33748)

    REFORGE是首个通过跨迭代维护与积累结构化修复证据、结合验证引导诊断的LLM辅助RTL缺陷修复框架，能够生成同时满足功能性、行为等价性、安全性和成本效率等多目标的完整RTL修复方案。

    

    大型语言模型（LLM）能够编写和修复RTL代码，但编译成功或测试平台执行成功并不能证明缺陷已被消除、行为得以保持、安全性得到保障或实现具有可行性。现有的基于LLM的RTL修复方法通常只报告单一的修复率，而不保留针对这些属性的独立证据，这限制了可复现的比较以及与硬件设计流程的集成。据我们所知，我们提出的REFORGE是首个在迭代过程中维护并积累结构化修复证据的LLM辅助RTL修复框架，使其能够生成同时满足多个目标的修复方案，包括功能性、行为等价性、安全性和成本效率。REFORGE对有缺陷的RTL进行特征分析，识别结构和安全相关特征，定位失效行为，并利用验证引导的诊断来生成完整的RTL修复变体。每个变体……（原文摘要在此处截断）

    arXiv:2611.33748v1 Announce Type: new  Abstract: Large language models (LLMs) can write and repair RTL, but successful compilation or testbench execution does not establish defect removal, behavioral preservation, security, or implementation feasibility. Existing LLM-based RTL repair methods often report a single repair rate without retaining separate evidence for these properties, limiting reproducible comparison and integration into hardware design flows. We present REFORGE, which, to our knowledge, is the first LLM-assisted RTL repair framework that maintains and accumulates structured repair evidence across iterations, allowing it to generate fixes that satisfy multiple objectives simultaneously, including functionality, behavioral equivalence, security, and cost efficiency. REFORGE profiles buggy RTL, identifies structural and security-relevant features, localizes failing behavior, and uses verification-guided diagnosis to generate complete RTL repair variants. Each variant underg
    
[^10]: AI代码审查者需要多少努力程度？来自Claude 5.5模型对似是而非错误补丁的证据

    How Much Effort Does an AI Code Reviewer Need? Evidence from Claude 5.5 Models on Plausible-but-Wrong Patches

    [https://arxiv.org/abs/2611.33639](https://arxiv.org/abs/2611.33639)

    通过对Claude 5.5系列模型审查90个SWE-bench补丁的预注册实验，本研究发现更高的推理努力程度能显著减少AI代码审查者对似是而非错误补丁的误接受，且对Opus而言中等努力程度能以低20%的成本达到与高等努力程度相当的效果。

    

    AI编码代理编写代码的速度超过了人类审查代码的速度，因此团队开始引入AI审查者。推理模型使这样的审查者能够通过努力程度设置在成本与细致程度之间进行权衡。我们探究了审查者需要多少努力才能发现似是而非的错误补丁——即能通过基准测试但在更强测试下失败的补丁。Claude Opus 5.5（预注册）、Sonnet 5.5和Haiku 5.5（探索性）各自以低、中、高三档努力程度审查了相同的90个来自SWE-bench Verified的代理编写的Django补丁。通过强化后的SWE-ABS测试，其中55个补丁是错误的，35个是正确的。在所有模型中，高等努力程度比低等努力程度接受更少的错误补丁（Opus的虚假成功率为0.45对0.67，Sonnet为0.49对0.65，Haiku为0.51对0.78），其中Opus和Haiku的差异得到统计学支持。对于Opus，中等努力程度在15个百分点范围内不劣于高等努力程度（差异0.018，95% CI [-.065, .108]），且成本降低20%，未达到预注册的30%标准。对于Sonnet和Haiku……

    arXiv:2611.33639v1 Announce Type: new  Abstract: AI coding agents write code faster than people can review it, so teams add AI reviewers. Reasoning models let such a reviewer trade cost for diligence through an effort setting. We asked how much effort a reviewer needs to catch plausible-but-wrong patches, which pass a benchmark's tests but fail stronger ones. Claude Opus 5.5 (preregistered), Sonnet 5.5 and Haiku 5.5 (exploratory) each reviewed the same 90 agent-written Django patches from SWE-bench Verified at low, medium and high effort. By strengthened SWE-ABS tests, 55 patches were wrong and 35 correct. In every model, high effort accepted fewer wrong patches than low (false success .45 against .67 for Opus, .49 against .65 for Sonnet, .51 against .78 for Haiku), a supported difference for Opus and Haiku. For Opus, medium was non-inferior to high within 15 percentage points (difference .018, 95% CI [-.065, .108]) and cost 20% less, short of the preregistered 30%. For Sonnet and Haik
    
[^11]: C到Rust迁移性能的跨层表征与AI辅助诊断

    Cross-Layer Characterization and AI-Assisted Diagnosis of C-to-Rust Performance

    [https://arxiv.org/abs/2611.33490](https://arxiv.org/abs/2611.33490)

    该论文通过对12个C/Rust基准程序的跨层表征揭示了运行时安全检查并非Rust迁移性能开销的主因（编译器与微架构优化常可缓解其成本），并据此提出了集成源码分析、编译器优化提示与动态插桩的AI辅助性能调试器FerroScope。

    

    将C代码迁移到Rust可以显著提升软件安全性，但理解此类迁移对性能的影响仍然具有挑战性。尽管Rust常被认为能够媲美C的性能，但运行时安全检查、编译器优化和硬件微体系结构之间以复杂的方式相互作用，使性能诊断变得困难。我们首先对包含12个基准测试的C实现及其对应的Rust实现进行了系统的跨层表征研究。我们的研究表明，运行时安全检查并不一定是性能开销的主导因素，因为编译器和微体系结构层面的优化通常会缓解其成本。在12个基准测试中的8个里，至少有一个Rust实现的性能在其C对应版本的10%以内或超越了它。基于这些发现，我们提出了FerroScope，一个AI辅助的性能调试器，它集成了源码级分析、编译器优化提示和动态插桩等技术（摘要在此处截断）。

    arXiv:2611.33490v1 Announce Type: cross  Abstract: Migrating C code to Rust can significantly improve software security, but understanding the performance implications of such migrations remains challenging. While Rust is often claimed to match C's performance, runtime safety checks, compiler optimizations, and hardware microarchitecture interact in ways that complicate performance diagnosis. We first conduct a systematic cross-layer characterization of 12 benchmarks comprising C implementations and their Rust counterparts. Our study reveals that runtime safety checks do not necessarily dominate performance overhead, as compiler and microarchitectural optimizations often mitigate their costs. Across 8 of the 12 benchmarks, at least one Rust implementation performs within 10% of its C counterpart or outperforms it. Motivated by these findings, we introduce FerroScope, an AI-assisted performance debugger that integrates source-level analysis, compiler optimization remarks, dynamic instru
    
[^12]: DevEx 面向人类，AX 面向智能体：度量智能体软件工作中的上下文健康度

    DevEx is for Humans, AX is for Agents: Measuring Context Health in Agentic Software Work

    [https://arxiv.org/abs/2611.33480](https://arxiv.org/abs/2611.33480)

    该论文提出智能体体验（AX）框架，通过从清晰度、可操作性、保真度和效率四个维度对智能体会话中的上下文质量进行评分，揭示人类提供的上下文质量（尤其是可操作性）是决定编码智能体成败的关键因素。

    

    编码智能体的成败取决于人类所提供的上下文，然而大多数评估只衡量输出结果。借鉴开发者体验研究成果，我们提出了智能体体验（AX），这是一种对会话记录在 CAFE(S) 框架的四个维度上进行评分的工具：清晰度、可操作性、保真度和效率；安全性作为门槛而非摩擦维度，不在本工具的评估范围内。一个由评分标准引导的语言模型对每个维度进行 1 到 5 分的评分，并解释人类本可以提供哪些内容。我们报告了三项检验该模型的研究。在 Stack Overflow 开发者调查中，报告存在上下文问题的受访者在所考察的全部七项活动中都更不可能认为 AI 非常有用（降幅为 12-28%）。在对 1,000 个试点样本中的 882 个智能体会话进行评分后，可操作性获得的评分最低（59.6% 的评分为 3 分或以下，而保真度仅为 26.6%），且会话越长评分越低。

    arXiv:2611.33480v1 Announce Type: new  Abstract: Coding agents succeed or fail in the context humans give them, yet most evaluations measure only output. Drawing on developer experience research, we introduce Agent Experience (AX), an instrument that scores session transcripts on four dimensions of the CAFE(S) framework: Clarity, Actionability, Fidelity, and Efficiency; Security, a gate rather than a friction dimension, is out of scope. A rubric-guided language model rates each dimension from 1 to 5 and explains what the human could have provided. We report on three studies that tested this model. In the Stack Overflow Developer Survey, respondents reporting context problems were less likely to rate AI as very useful in all seven activities examined (12-28%). Across 882 rated agentic sessions from a pilot sample of 1,000, Actionability received the lowest ratings (59.6% of ratings at or below 3, versus 26.6% for Fidelity), and ratings were lower in longer sessions. One researcher's unb
    
[^13]: 面向Python入门编程的集成学习生态系统设计：支架式问题解决、平板风格导航与AI增强教学

    Designing an Integrated Learning Ecosystem for Introductory Python Programming: Scaffolded Problem Solving, Tablet-Inspired Navigation, and AI-Enhanced Instruction

    [https://arxiv.org/abs/2611.33471](https://arxiv.org/abs/2611.33471)

    本文为Python入门课程设计并分析了一个集成学习生态系统，其中15个实验课共享IPO推理、教师指导编程、独立任务、Colab/Gradio扩展、调试支持、任务清单与提交要求七大支架要素，并以Pygame综合大作业和AI增强教学、平板风格导航作为支撑。

    

    入门编程要求学生表征问题、开发和测试程序、调试错误，并在新情境中应用计算思想。本文对为一个包含16个实验课时的Python入门课程开发的集成学习生态系统进行了以教学工件为中心的设计分析。所分析的语料库包含18个教学工件：课程大纲、一个平板风格的主页以及实验1至16。我们考察了课程导航、问题表征、支架设计、独立迁移、应用开发、调试、AI集成与评估。全部15个支架式实验课共享七个要素：输入-处理-输出（IPO）推理、教师指导式编程、独立任务、基于Google Colab/Gradio的扩展、调试支持、任务清单以及提交要求。实验16脱离了这一结构，作为一项综合性的基于Pygame的特色大作业。该分析识别出四个相互关联的（摘要在此处截断）

    arXiv:2611.33471v1 Announce Type: cross  Abstract: Introductory programming requires students to represent problems, develop and test programs, debug errors, and apply computational ideas in new contexts. In this paper, we present an artifact-centered design analysis of an integrated learning ecosystem developed for a 16-lab introductory Python course. The analyzed corpus comprises 18 instructional artifacts: the course syllabus, a tablet-inspired homepage, and Labs 1-16. We examined course navigation, problem representation, scaffolding, independent transfer, application development, debugging, AI integration, and assessment. All 15 scaffolded labs shared seven elements: input-processing-output (IPO) reasoning, instructor-guided programming, an independent task, a Google Colab/Gradio extension, debugging support, a task checklist, and submission requirements. Lab 16 departed from this structure as an integrative Pygame-based Hallmark Assignment. The analysis identifies four connected 
    
[^14]: GhostAPI-V：评估用于内部API迁移修复的经审计诊断方法

    GhostAPI-V: Evaluating Audited Diagnostics for Internal API Migration Repair

    [https://arxiv.org/abs/2611.33396](https://arxiv.org/abs/2611.33396)

    GhostAPI-V提出以经人工审计的时序契约为API迁移修复提供候选诊断，实验证明完整双向契约反馈可将隐藏测试成功率从62.5%提升至77.1%。

    

    API迁移要求调用方采用目标版本的用法，同时移除不兼容的旧版本用法。GhostAPI-V将这些义务表示为经人工审计的时序契约，并提供针对候选方案的修复诊断信息，其正确性由独立的行为测试进行评估。我们使用Claude Code和Codex对来自八个Python代码库的48个内部迁移进行评估，每个智能体与任务各进行两次初始运行。五个修订分支从相同的192个初始候选方案出发，控制生成预算、版本证据和开发测试反馈。隐藏测试的成功率分别为：原始版本差异反馈120/192（62.5%），仅必需项契约130/192（67.7%），完整双向契约148/192（77.1%）。完整反馈相比原始版本差异反馈和仅必需项反馈分别提升了14.6和9.4个百分点。包级别效应在所有八个代码库中均为正向，并且……

    arXiv:2611.33396v1 Announce Type: new  Abstract: API migrations require callers to adopt target-version usages while removing incompatible legacy usages. GhostAPI-V represents these obligations as human-audited temporal contracts and supplies candidate-specific repair diagnostics, with correctness assessed by independent behavioral tests. We evaluate 48 internal migrations from eight Python repositories using Claude Code and Codex, with two initial runs per agent and task. Five revision arms branch from the same 192 initial candidates, controlling generation budget, version evidence, and development-test feedback. Hidden-test success is 120/192 (62.5%) for raw version-difference feedback, 130/192 (67.7%) for required-only contracts, and 148/192 (77.1%) for full bidirectional contracts. Full feedback improves over raw version-difference feedback and required-only feedback by 14.6 and 9.4 percentage points, respectively. The package-level effect is positive in all eight repositories and 
    
[^15]: 构建面向GitHub Copilot的统一行内代码编辑模型

    Building a Unified Inline Code Editing Model for GitHub Copilot

    [https://arxiv.org/abs/2611.33314](https://arxiv.org/abs/2611.33314)

    本文为GitHub Copilot提出统一行内代码编辑模型，通过锚定差异补丁的单一输出空间一次性预测光标补全、邻近编辑与远距离编辑三种类型，支持单次响应包含多个编辑，避免了原系统多次顺序调用不同模型的开销。

    

    AI辅助编程的范围涵盖从完全智能体式编程到编辑器中的行内代码建议。行内建议需要从编辑会话中预测开发者最可能的下一次编辑。下一次编辑可能位于光标处、邻近行中，或更远的位置，而一个编辑意图可能同时跨越其中多个位置。先前的系统按固定顺序查询独立的补全模型、邻近编辑模型和远距离编辑模型，每个模型各有自己的输出格式；因此它无法在单次响应中表达跨越远距离位置的编辑意图，每次建议可能需要多次顺序的模型调用，并且即使在修正方案更合适时也优先采用补全。我们提出了一个统一模型，在单一输出空间中预测所有三种编辑类型：一系列锚定差异补丁，指明位置、删除的行和新增的行。这种表示允许每次响应包含多个编辑，因此客户端可以从缓存中提供后续编辑……

    arXiv:2611.33314v1 Announce Type: cross  Abstract: AI-assisted coding ranges from fully agentic coding to inline code suggestions in the editor. Inline suggestions require predicting a developer's most likely next edit from an editing session. The next edit may lie at the cursor, in nearby lines, or farther away, and one editing intent can span several of these. The preceding system queries separate completion, nearby edit, and long-distance edit models, each with its own output format, in a fixed sequence; it thus cannot express intents spanning distant locations in one response, can require several sequential model calls per suggestion, and prioritizes completion even when a correction would serve better. We present a unified model that predicts all three edit types in one output space: a sequence of anchored diff patches specifying locations, deleted lines, and added lines. This representation permits multiple edits per response, so the client can serve later edits from a cache with
    
[^16]: 真实的包，错误的控制器：LLM生成的PLC代码中供应商-协议不匹配的静态审计

    Real Package, Wrong Controller: Static Auditing of Vendor--Protocol Mismatches in LLM-Generated PLC Code

    [https://arxiv.org/abs/2611.33227](https://arxiv.org/abs/2611.33227)

    论文提出了Proteus-OT，一个基于版本固定绑定图谱的无标签静态验证器，用于检测大语言模型生成的PLC代码中导入包与声明工业控制器目标之间的供应商-协议不匹配，发现883个生成脚本中有20.4%存在此类导入级别不兼容问题。

    

    语言模型可以生成驱动可编程逻辑控制器（PLC）上工业协议的Python代码，然而现有的代码落地检查只验证导入的包或符号是否存在，而不验证它是否适合所声明的工业目标。我们定义了元数据条件下的导入级别供应商-协议不兼容性，并构建了Proteus-OT，这是一个基于版本固定的已文档化绑定图谱的无标签静态验证器。在883个生成的脚本中，有180个存在与声明目标不符的导入级别不匹配（20.4%）；限制为单一供应商目标时该比例为16.3%，若额外要求违规库能够触达协议通信调用，则比例为14.2%，我们将其解读为依赖于语料库元数据而非已验证的意图。28个可判定的人工编写的控制文件中没有一个存在此类问题，但样本量太小，无法估计该问题的普遍程度。在平衡子集中，96.6%的不匹配脚本（摘要至此截断）

    arXiv:2611.33227v1 Announce Type: cross  Abstract: Language models can generate Python that drives industrial protocols on programmable logic controllers, yet existing code-grounding checks ask whether an imported package or symbol exists rather than whether it suits the declared industrial target. We define \emph{metadata-conditioned, import-level vendor--protocol incompatibility} and build \textsc{Proteus-OT}, a label-free static verifier over a version-pinned graph of documented bindings. Across 883 generated scripts, 180 carry an import-level mismatch against the declared target (20.4\%); restricting to single-vendor targets gives 16.3\%, and additionally requiring the offending library to reach a protocol communication call gives 14.2\%, which we read as conditional on the corpus metadata rather than on verified intent. None of the 28 adjudicable human control files carries one, a sample too small to estimate a prevalence. Within the balanced subset, 96.6\% of mismatching scripts 
    
[^17]: 揭秘真实世界中的LLM服务“白嫖”现象

    Demystifying Real-world LLM Service Freeloading

    [https://arxiv.org/abs/2611.33218](https://arxiv.org/abs/2611.33218)

    本文通过对513个开源项目、64个上游LLM服务和1,588条推理路径的系统性测量研究，首次揭示了第三方项目通过复刻产品协议将免费LLM网页服务打包为独立API的“服务白嫖”生态的规模、依赖与运行机制。

    

    第三方项目将来自大语言模型（LLM）网页应用和专用客户端的免费推理能力重新打包为独立API。我们将这种超出服务提供商所允许的访问、分配或再分发条件的使用行为称为“LLM服务白嫖”（LLM Service Freeloading, LSF）。然而，其规模和底层机制至今仍缺乏充分了解。本文对513个开源项目开展了系统性测量研究，追踪了横跨64个上游LLM服务的1,588条推理路径，并刻画了这些项目的开发情况、依赖关系和运行机制。研究发现，93.0%的相关仓库创建于2026年；直接HTTP或WebSocket请求占全部路径的83.2%，表明复刻产品协议是主要的调用方式。这些项目通过适配消息、重建对话状态和转换响应来支持兼容API。账号选择或轮换出现在304条路径上（占19.1%），使拥有多个可用账号的项目能够在账号之间分配请求。

    arXiv:2611.33218v1 Announce Type: cross  Abstract: Third-party projects repackage free inference from LLM web applications and dedicated clients into independent APIs. We refer to such use beyond providers' permitted access, allocation, or redistribution conditions as LLM service freeloading (LSF). However, its scale and underlying mechanisms remain poorly understood. We present a systematic measurement study of 513 open-source projects, tracing 1,588 inference paths across 64 upstream LLM services and characterizing their development, dependencies, and operating mechanisms. We find that 93.0% of repositories were created in 2026. Direct HTTP or WebSocket requests account for 83.2% of paths, making reproduction of product protocols the dominant invocation method. Projects adapt messages, reconstruct conversation state, and convert responses to support compatible APIs. Account selection or rotation appears on 304 paths (19.1%), allowing projects with multiple eligible accounts to distri
    
[^18]: 多智能体协同的 Rust 代码方法级重构

    Multi-Agent Coordinated Method-Level Refactoring of Rust Code

    [https://arxiv.org/abs/2611.29321](https://arxiv.org/abs/2611.29321)

    该论文提出了一种基于大语言模型的多智能体协同工作流，通过候选生成、可配置专家审查与容器化验证三个环节，安全地实现 Rust 代码的方法提取、内联与移动重构，并在开源项目任务上首次尝试即全部通过验证。

    

    随着 Rust 代码库的演进，开发者会重构代码以减少重复、简化复杂方法，并将相关功能组织在一起。开发者可以手动进行这些更改，或使用 IDE 辅助完成受支持的操作，然后通过编译器诊断和测试来检查 resulting 代码。Rust 中的方法级重构必须在考虑所有权、借用和生命周期的同时保持可观察行为不变。我们提出了一种基于大语言模型（LLM）的工作流，用于在 Rust 中提取、内联和移动方法。该工作流结合了候选生成、可配置的专家审查以及基于容器的验证。在来自三个开源 Rust 项目的六个任务上，三种端到端配置在首次尝试中即通过了所有任务检查，分别使用了 6、13 和 31 次模型调用。为了隔离审查环节的效果，我们随后给四种配置提供了相同的六个参考补丁和六个故意改变行为的变体。（注：原始摘要在此处被截断）

    arXiv:2611.29321v1 Announce Type: new  Abstract: As Rust codebases evolve, developers refactor code to reduce duplication, simplify complex methods, and keep related functionality together. Developers can make these changes manually or use IDE assists for supported operations, then check the resulting code with compiler diagnostics and tests. Method-level refactoring in Rust must preserve observable behavior while accounting for ownership, borrowing, and lifetimes. We present a large language model (LLM)-based workflow for extracting, inlining, and moving methods in Rust. The workflow combines candidate generation, configurable specialist review,and container-based validation. On six tasks from three open-source Rust projects, three end-to-end configurations passed all task checks on their first attempt, using 6, 13, and 31 model calls. To isolate the effect of review, we then gave four configurations the same six reference patches and six deliberately behavior-changing variants. A des
    
[^19]: 修复还是删除？外部反馈让更便宜的AI编程智能体修复规则违规

    Fixed or Deleted? External Feedback Lets Cheaper AI Coding Agents Repair Rule Violations

    [https://arxiv.org/abs/2611.29162](https://arxiv.org/abs/2611.29162)

    研究发现，引入商业检查器CodeAssure等外部反馈（报告违规位置和修复目标）能大幅提升AI编程智能体修复项目规则违规的成功率，使更小更便宜的模型（修复率从26%升至69%）接近甚至逼近大型模型的修复效果。

    

    摘要：许多缺陷违反了代码本身无法反映的项目规则，例如分层约束或必需的测试。我们研究了当外部检查器报告这些违规及其位置和修复目标时，LLM编程智能体是否能更好地修复此类违规。由AI智能体驱动的三个模型（小型、中型和大型模型）处理了十个开源项目中25个包含31个植入缺陷的拉取请求，外加5个正确的对照组，分别在无外部反馈和结合商业检查器CodeAssure（CA）检测结果的条件下各运行五次。我们还将结果与GitHub Copilot代码审查进行比较，并将CodeAssure的检测能力与九个开箱即用的免费静态分析工具（包括Semgrep和SonarQube Community）进行对比。只有当修复后的代码能够编译并保留原有功能时，该次运行才被计为成功修复。实验结果显示，CA的检测结果使最小模型的修复率从26%提升至69%，中型模型从46%提升至90%，大型模型从45%提升至93%。

    arXiv:2611.29162v1 Announce Type: cross  Abstract: Many defects violate project rules that the code itself does not reveal, such as layering constraints or required tests. We study whether LLM coding agents repair such violations better when an external checker reports them, with their location and a repair goal. Three models, a small, medium and large model, driven by an AI agent, worked on 25 pull requests with 31 planted defects in ten open-source projects, plus 5 correct controls, five times alone and five times with findings from CodeAssure (CA), a commercial checker. We also compare with GitHub Copilot code review, and compare CodeAssure's detection with nine free static-analysis tools run out of the box, including Semgrep and SonarQube Community. Counting a run as repaired only when the repaired code also compiles and keeps its feature, CA findings raised the repair rate from 26% to 69% for the smallest model, from 46% to 90% for the medium model and from 45% to 93% for the larg
    
[^20]: 从迁移到校准：跨模型、司法管辖区与规模保持智能体能力

    From Migration to Calibration: Preserving Agent Capabilities across Models, Jurisdictions, and Scale

    [https://arxiv.org/abs/2609.35149](https://arxiv.org/abs/2609.35149)

    提出将智能体校准构建为标准优先的适配框架，通过定义基础能力、技术环境与用户情境标准，系统诊断差距并应用修订，从而在模型更换、跨管辖区部署和规模化过程中保持智能体能力不退化。

    

    部署、迁移或扩展智能体可能会改变其模型、运行框架、基础设施、应用场景及目标用户。我们将智能体校准表述为标准优先的适配方法：定义基础能力、技术环境和用户情境标准；诊断差距；生成并应用修订；并在固定预算内重新检查相同标准。这些标准族在信息层、运行框架层和用户接受层之间相互作用。源行为仅具诊断价值，并非完美的参考或能力上限：模型替换可能将正确答案变为错误，或将错误变为正确。资格认定要求通过所有强制性已知测试、真实的端到端部署路径、硬性谓词以及声明的任务/用户最低标准；总体收益不能抹去硬性失败。修订可能更改工具或运行框架，添加示范和任务描述，或使用经验证的目标原生轨迹来训练策略

    arXiv:2609.35149v2 Announce Type: replace  Abstract: Deploying, migrating, or scaling an agent can change its model, harness, infrastructure, application, and intended users. We formulate agent calibration as standards-first adaptation: define basic-capability, technical-environment, and user-context standards; diagnose gaps; generate and apply revisions; and recheck the same standards within fixed budgets. These standard families interact across information, harness, and user-acceptance layers. Source behavior is diagnostic, not a perfect reference or capability ceiling: model replacement can turn correct answers into errors or errors into correct answers. Qualification requires all mandatory known tests, actual end-to-end deployment paths, hard predicates, and declared task/user minimums to pass; aggregate gains cannot erase hard failures. Revisions may change tools or harnesses, add demonstrations and task descriptions, or use validated target-native trajectories to train a policy, 
    
[^21]: SpecRead：衡量语言模型是否理解硬件规格的基准测试

    SpecRead: A Benchmark for Measuring Whether Language Models Understand Hardware Specifications

    [https://arxiv.org/abs/2609.33699](https://arxiv.org/abs/2609.33699)

    SpecRead是一个将硬件规格理解能力与代码生成能力分离的新基准测试，通过针对OpenTitan IP模块的四类题目（精确检索、跨章节推理、矛盾检测和规格-RTL一致性检查）来评估大语言模型对硬件规格的真实理解程度。

    

    现有针对大语言模型（LLM）在硬件设计领域的基准测试主要评估下游产物，例如生成的RTL、断言或测试台。当模型失败时，可能是它误读了规格说明，也可能是它理解了规格但未能写出代码。我们提出了SpecRead，这是一个将规格理解能力与生成能力分离的基准测试。SpecRead v2.2 包含针对10个开源OpenTitan IP模块的383道题目，分为四种类型：精确检索、跨章节推理、变异规格中的矛盾检测，以及规格-RTL一致性检查，另外还有41个干扰对照组和41个一致RTL对照组。第四类题目使用真实的RTL变异，只有当Icarus Verilog仿真显示出行为变化时才会保留。有规格与无规格的消融实验表明，这些题目需要依赖规格摘录而非仅靠训练记忆，但这并不能排除模型对未变异源文本的记忆可能带来的帮助。

    arXiv:2609.33699v2 Announce Type: replace  Abstract: Existing benchmarks for large language models (LLMs) in hardware design evaluate downstream artifacts such as generated RTL, assertions, or testbenches. When a model fails, it may have misread the specification or understood it and failed to write the code. We present SpecRead, a benchmark that isolates specification comprehension from generation ability. SpecRead v2.2 contains 383 questions over 10 open-source OpenTitan IP blocks in four types: exact retrieval, cross-section reasoning, contradiction detection in mutated specifications, and spec-RTL consistency checking, plus 41 distractor controls and 41 consistent-RTL controls. Type-4 items use real RTL mutations retained only when Icarus Verilog simulation shows a behavioral change. A with-spec vs. without-spec ablation suggests the questions require the excerpt rather than training recall alone, though it does not rule out memorization of the unmutated source text helping a model
    
[^22]: 大型语言模型在生成形式化程序规范方面有多强大？

    How Powerful are LLMs in Generating Formal Program Specifications?

    [https://arxiv.org/abs/2608.13077](https://arxiv.org/abs/2608.13077)

    本文提出了Coins框架，通过实例化规范并生成具体证明义务来评估LLM生成形式化程序规范的质量，避免了传统方法中证明难度与规范质量混淆的问题。

    

    形式化验证提供了软件正确性的强有力保证，但其应用受到编写精确形式化规范高昂成本的限制。尽管近期大型语言模型（LLMs）在定理证明和已验证代码生成方面展现出强大能力，但它们生成程序规范的真实能力仍不明确。现有评估要么需要验证实现一致性，要么需要证明规范之间的语义等价性，这两者都极其困难，且可能将证明难度与规范质量混为一谈。为解决此问题，我们引入了Coins，一个基于Rocq的评估框架，通过在被评估的规范上实例化可信测试用例并生成具体证明义务来评估规范质量。这种设计符合形式推理的非对称性，其中成功的证明提供可靠证据，而证明失败则不然。

    arXiv:2608.13077v1 Announce Type: new  Abstract: Formal verification provides strong guarantees of software correctness, but its adoption is limited by the high cost of writing precise formal specifications. While recent large language models (LLMs) have shown strong capabilities in theorem proving and verified code generation, their true ability to generate program specifications remains unclear. Existing evaluations require either verifying implementation conformance or proving semantic equivalence between specifications, both of which are formidably difficult and may conflate proof difficulty with specification quality. To address this problem, we introduce Coins, a Rocq based evaluation framework that assesses specification quality by instantiating specifications under evaluation on trusted test cases and generating concrete proof obligations. This design aligns with the asymmetric nature of formal reasoning, where successful proofs provide reliable evidence while proof failures ar
    
[^23]: AiFlow：面向版本化 LLM 服务的从构建到退役的全生命周期一致性保障

    AiFlow: Construction-to-Retirement Conformance for Versioned LLM Services

    [https://arxiv.org/abs/2608.00558](https://arxiv.org/abs/2608.00558)

    AiFlow 提出一种可执行契约机制，将版本化 LLM 服务的构建、调用与退役全流程串联，通过证书认证确保调用的终结结果与注册资源的释放，并支持严格发布和延迟发布两种模式实现版本安全退役。

    

    一次动态更新可以在保持大语言模型（LLM）工作流拓扑结构与输出的同时，绕过过滤器的队列与并发限制。后台任务也可能在其调用者返回后继续使用旧版本。AiFlow 通过可执行契约将构建、调用与退役三个阶段联系起来。每个被接纳的请求都会捕获一个绑定到受检操作符的动作。服务会对调用的终结结果及其注册工作的释放进行认证。当新调用不再能捕获该版本时，所有者会在退役之前检查这些证书以及剩余的本地资源。严格发布会在旧调用排空期间暂停接纳新请求；延迟发布则允许在前身版本退役之前接纳对后继版本的调用。我们在单个 Java 虚拟机中评估了两个无状态线性工作流。在严格发布模式下，针对每种检查配置，我们对三种注入故障场景各重复测试了三次。一项观察……

    arXiv:2608.00558v2 Announce Type: replace-cross  Abstract: A live update can preserve the topology and output of a large language model (LLM) workflow while bypassing a filter's queue and concurrency limits. A background task can use the old version after its caller returns. AiFlow links construction, invocation and retirement through an executable contract. Each admitted request captures an action bound to the checked operators. The service certifies the call's terminal outcome and the release of its registered work. Once new calls can no longer capture the version, the owner checks these certificates and remaining local resources before retirement. Strict publication pauses admission while old calls drain. Deferred publication can admit calls to the successor before its predecessor retires. We evaluated two stateless linear workflows in one Java virtual machine. Under strict publication, we repeated three injected fault scenarios three times per check configuration. An observation co
    
[^24]: 安全的假象：AI与人类C++代码的多层级验证

    The Illusion of Safety: Multi-Tier Verification of AI vs. Human C++ Code

    [https://arxiv.org/abs/2607.00107](https://arxiv.org/abs/2607.00107)

    该论文构建了包含8,918个C++程序的VULBENCH-CPP基准，通过功能测试、静态分析、动态分析和有界模型检验四个层级系统对比AI与人类编写的代码，揭示了仅依赖静态分析会带来“安全的假象”。

    

    随着大语言模型（LLM）越来越多地被部署用于系统编程，其生成安全C++代码的能力仍然是一个关键问题。然而，大多数针对AI生成代码的安全评估仅依赖静态分析，而静态分析只会标记警告，既无法确认运行时违规，也无法对未经测试的代码路径进行推理。本研究探讨开放权重LLM生成的C++代码在可测量程度上是否比人类编写的代码更不安全，以及常用的验证工具是否对风险有一致的判断。我们提出了VULBENCH-CPP，这是一个包含8,918个C++程序的基准数据集，这些程序由三个开放权重LLM（Gemma 3 27B IT、LLaMA 3.3 70B Instruct、Qwen 2.5 Coder 32B Instruct）与人类作者在851个竞赛编程任务上生成。每个程序均通过四个验证层级进行标注：功能测试、静态分析（cppcheck、clang-tidy）、动态分析（ASan/UBSan）以及有界模型检验（ESBMC）。在静态分析下，两者看起来同等……（原文摘要在此处被截断）

    arXiv:2607.00107v4 Announce Type: replace  Abstract: As large language models (LLMs) are increasingly deployed for systems programming, their ability to generate secure C++ code remains a critical concern. Yet most security evaluations of AI-generated code rely on static analysis alone, which flags warnings without confirming runtime violations or reasoning about untested paths. This study investigates whether C++ code generated by open-weight LLMs is measurably less safe than human-written code, and whether common verification tools agree on the risk. We introduce VULBENCH-CPP, a benchmark of 8,918 C++ programs from three open-weight LLMs (Gemma 3 27B IT, LLaMA 3.3 70B Instruct, Qwen 2.5 Coder 32B Instruct) and human authors across 851 competitive-programming tasks. Each program is annotated by four verification tiers: functional testing, static analysis (cppcheck, clang-tidy), dynamic analysis (ASan/UBSan), and bounded model checking (ESBMC). Under static analysis the two look equall
    
[^25]: 智能体化模型检测

    Agentic Model Checking

    [https://arxiv.org/abs/2605.21434](https://arxiv.org/abs/2605.21434)

    本文提出智能体化模型检测及其实例 BMC-Agent，遵循“智能体提出、求解器验证”的原则，让 LLM 智能体负责规范推断与验证规划等语义推理，由有界模型检测器进行形式化验证，从而实现对大型软件系统的自动验证。

    

    验证大型软件系统十分困难。形式化规范通常难以获得，而且代码所依赖的假设往往在调用处以隐式方式编码，而非在函数边界强制执行。本文提出了智能体化模型检查，这是一种将大语言模型智能体与有界模型检测相结合的验证方案，其核心原则是“智能体提出，求解器验证”：智能体负责规范推断和验证规划等语义推理工作，而形式化后端则对生成的有界验证查询进行检验。我们在 BMC-Agent 中实现了这一方案，该系统构建在一个现成的有界模型检测器之上。本文的关键贡献在于对模型检测器的智能体化编排。BMC-Agent 并非采用固定的单一验证流程，而是为每个验证任务规划如何应用有界模型检测。它能够推断函数规范，并据此构建……（摘要原文在此处截断）

    arXiv:2605.21434v2 Announce Type: replace  Abstract: Verifying large software systems is difficult. Formal specifications are rarely available, and the assumptions that code relies on are often encoded implicitly at call sites rather than enforced at function boundaries. This paper presents agentic model checking, a verification scheme that combines LLM agents with bounded model checking (BMC) under the principle that agents propose, solvers verify: agents perform semantic reasoning such as specification inference and verification planning, while the formal backend checks the resulting bounded verification queries. We realize this scheme in BMC-Agent, an agentic verification system built on an off-the-shelf bounded model checker.   The key contribution of this paper is agentic orchestration of the model checker. Rather than fixing a single verification recipe, BMC-Agent plans how bounded model checking should be applied to each verification task. It infers function specifications, conf
    
[^26]: AI原生软件生态系统中故障传播的阈值模型

    A Threshold Model of Fault Propagation in AI-Native Software Ecosystems

    [https://arxiv.org/abs/2604.19827](https://arxiv.org/abs/2604.19827)

    本文将复杂性理论与软件演化相结合，把AI编码智能体引入的缺陷在开放变更间的传播建模为基于争用关系的分支过程，并证明每个仓库存在一个使故障传播自我维持的临界阈值。

    

    当许多编码智能体同时在同一个代码仓库中工作时，软件工程师只能通过各个部分来评判一个集体。代码审查一次只检查一个变更，然而一个变更可能引入了由另一个变更带来的缺陷，此时即使修复了原始变更，也无法再修复整个仓库。为了让人类判断保持在智能体软件开发的核心位置，我们需要刻画智能体集体行为的模型。本文将复杂性理论与软件演化相结合来构建这样的模型。我们将缺陷在开放变更之间的传播建模为一个基于“争用”的分支过程——争用指两个同时开放的变更修改同一文件之间的关系。正如流行病学中的再生数一样，一旦每个有缺陷的变更平均将缺陷传递给超过一个其他变更，传播就会自我维持。每个仓库都存在一个参考阈值，即每个争用链接上的交叉概率，一旦超过该阈值，传播就会自我维持。

    arXiv:2604.19827v2 Announce Type: replace-cross  Abstract: When many coding agents work in one repository at once, software engineers are left to judge a collective through its parts. Review examines one change at a time, yet a change can incorporate a fault introduced by another, and repairing the originating change then no longer repairs the repository. To keep human judgment at the centre of agentic software development, we need models of what agents do together. This paper joins complexity theory with software evolution to build one. We model how a fault spreads among open changes as a branching process over contention, the relation between two changes that are open at the same time and modify a common file. As with the reproduction number of an epidemic, spread sustains itself once each faulty change passes the fault, on average, to more than one other. Each repository has a reference threshold, the crossing probability per contention link above which spread would sustain itself i
    
[^27]: 使用大语言模型起草架构设计决策：DRAFT方法

    DRAFT-ing Architectural Design Decisions using LLMs

    [https://arxiv.org/abs/2504.08207](https://arxiv.org/abs/2504.08207)

    该论文提出了DRAFT方法，将检索增强少样本生成与微调相结合，使大语言模型能够根据决策上下文生成架构设计决策，并根据标题生成完整的架构决策记录。

    

    架构知识管理（AKM）对软件开发至关重要，而架构决策记录（ADR）作为AKM的重要产物，提供了一种结构化的方式来记录架构设计决策（ADD）。然而，由于需要大量人工投入且工具支持不足，ADR的实际采用率受限。先前的研究表明，大语言模型（LLM）能够支持ADD的生成，但简单的提示方法效果欠佳。本文探讨了LLM如何在两个互补的任务中为架构师提供支持：从决策上下文生成ADD，以及从标题生成完整的ADR。我们提出了领域特定检索增强少样本调优，这是一种将检索增强少样本生成与微调相结合的新型生成方法。DRAFT分两个阶段运行：离线阶段使用从历史ADR中提取的检索增强提示对模型进行微调，在线阶段生成……

    arXiv:2504.08207v2 Announce Type: replace-cross  Abstract: Architectural Knowledge Management (AKM) is crucial for software development and Architectural Decision Records (ADR), an important artifact of AKM, provide a structured approach to capture Architectural Design Decisions (ADD). However, their adoption is limited due to the manual effort involved and insufficient tool support. Prior work shows that LLMs can support ADD generation, though basic prompting yields suboptimal results. This paper explores how LLMs can support architects in two complementary tasks: generating ADDs from Decision Contexts, and generating complete ADRs from titles. We propose Domain-specific Retrieval Augmented Few-shot Tuning (DRAFT), a novel approach that integrates Retrieval Augmented Few-shot Generation with Fine-tuning to generate. DRAFT operates in two phases: an offline phase that fine-tunes a model using retrieval-augmented prompts derived from historical ADRs, and an online phase that generates d
    

