# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Before They Can Solve: Predicting Post-Training Coding-Agent Performance from Base Models](https://arxiv.org/abs/2610.10478) | 提出将成功的后训练智能体轨迹作为基座模型潜力的前瞻信号，通过重放轨迹并定位使代码库首次由失败转为通过的决定性步骤，从而在昂贵的智能体后训练之前预测基座模型能否培养出高性能的编码智能体。 |
| [^2] | [Design-Time Conformance Checking for Pulse-Level Quantum Control](https://arxiv.org/abs/2610.10427) | 本文提出 qconform，一个基于版本化设备能力描述符、设计时运行的确定性检查器，可在实验前判定脉冲级量子控制程序能否在目标设备上实现，避免实验数据与所写程序不符。 |
| [^3] | [TaoD2C-Bench: Benchmarking MLLMs for Industrial UI Code Generation Beyond Visual Fidelity](https://arxiv.org/abs/2610.10374) | 提出了TaoD2C-Bench基准测试，用于评估多模态大语言模型在工业设计转代码任务中超越视觉识别、结合图层元数据与目标库约束来生成满足实现要求的UI代码的能力。 |
| [^4] | [Input-Blind Controls Produce Substantial Oracle Headroom for Layer Programs in Multiple-Choice Evaluation](https://arxiv.org/abs/2610.10368) | 本研究发现在多项选择评估中，输入盲化的对照扰动所产生的神谕提升空间反而超过真实的层跳过与重复程序，说明仅凭选择增益无法解释所选层程序为何有效。 |
| [^5] | [When Sub-Agents Work in Parallel: The Promises and Pitfalls of Dynamic Concurrency in Long-Horizon Coding Tasks](https://arxiv.org/abs/2610.10263) | 该论文首次通过354个任务、2,124次执行的对照实验，系统研究了Codex、Claude Code和Kimi Code中动态并发作为执行策略的效果，发现模型能力主导较短任务的成败，而编排调度能力则成为长时程开发任务完成的关键。 |
| [^6] | [Using Small Language Models to Reverse-Engineer Machine Learning Pipelines Structures](https://arxiv.org/abs/2610.10261) | 该研究验证了小语言模型能够凭借其代码理解与分类能力，从源代码中有效提取机器学习流水线的阶段结构，从而克服人工标注不可扩展及传统分类器难以适应领域多样性的局限。 |
| [^7] | [QuSema: Detecting Silent Bugs in Quantum Libraries via Quantum-knowledge-enhanced Agents](https://arxiv.org/abs/2610.10258) | QuSema提出了一种自主测试智能体，将量子语义与文档约束作为源代码级语义预言机，能够在缺乏基于执行的预言机的情况下检测量子库中从有效输入产生无效输出的静默缺陷。 |
| [^8] | [OOM-RL II: Reality Is an Oracle, Not a Debugger Provenance-Constrained Diagnosis in Continually Evolving Agent-Engineered Systems](https://arxiv.org/abs/2610.10256) | 该论文提出“现实是神谕而非调试器”的核心观点，并在一个持续演化、由智能体工程化构建的量化交易系统中进行溯源约束诊断，表明即使账户一年内盈利并跑赢大盘指数，在缺乏完整的推荐到运行时绑定的情况下，也无法将结果归因于特定的演化程序版本或确认其统计显著性。 |
| [^9] | [TestGRAD: Evolving Test Suites via Failure Pattern Momentum for SWE-Agent Ensemble](https://arxiv.org/abs/2610.10242) | 该论文受带动量的梯度下降启发，提出了TestGRAD框架，将SWE-Agent集成中的补丁选择形式化为测试空间优化问题，通过差分损失和失败模式动量机制自动演化测试套件，从而有效区分相互竞争的候选补丁。 |
| [^10] | [Why Software Engineering Is Indispensable in the Age of Coding Agents](https://arxiv.org/abs/2610.10226) | 本文论证AI编码智能体的兴起使软件工程和软件工程师更加不可或缺，因为大语言模型的三个结构性缺陷造成无法通过训练消除的真空，需要软件工程师通过方法论知识、领域知识、设计选择和流程选择四种知识杠杆，充当方法论专家、中介和守护者，才能确保AI辅助开发的软件可信可靠。 |
| [^11] | [Agentic AI-Assisted Modeling for Production Scheduling: Assessment in Constraint Programming](https://arxiv.org/abs/2610.10184) | 本研究提出将未经专门训练的通用大语言模型编排为单智能体或多智能体系统，并结合模型上下文协议服务器进行上下文感知的求解器文档检索以抑制幻觉，从而从自然语言问题描述中自动构建并实现生产调度的约束规划优化模型，弥合了自动化建模与智能体决策支持两大研究方向。 |
| [^12] | [On the Reliability of LLM-Based Vulnerability Patching Benchmarks](https://arxiv.org/abs/2610.10150) | 本文揭示了当前基于大语言模型的漏洞修补基准在智能体、框架和数据集三个层面存在的系统性缺陷会导致性能评估失真，并构建了一个包含来自84个开源项目的112个历史漏洞、配有概念验证测试、回归测试和开发者测试的更可靠基准。 |
| [^13] | [AdaT$^2$: Adaptive Test Transformations for Black-Box Boundary Testing of Conversational Agents](https://arxiv.org/abs/2610.10141) | AdaT$^2$通过让LLM扮演用户与智能体对话来提取策略条件的陈述，并自适应地选择“陈述+变换指令”配对来生成黑盒边界测试，以验证智能体在策略边界两侧表现出不同的行为。 |
| [^14] | [Comprehension Audits to Mitigate Risks from Automated AI Research](https://arxiv.org/abs/2610.10064) | 提出一种名为“理解审计”的开发过程保障机制，要求负责人向独立审计员解释研发贡献以证明其真正理解所构建的系统，否则将暂停开发并施加逐步升级的后果，以此缓解AI自动化研发带来的安全风险。 |
| [^15] | [Designing Collaborative AI-Driven Workflows for Scientific Software Engineering](https://arxiv.org/abs/2610.09995) | 本文提出一种人机协作的AI驱动工作流——由领域专家编写规范和计划、智能体在确定性编排下生成代码，每个阶段通过数值比对和人工审核把关，并在大型高能物理程序从Fortran到C++的翻译任务中验证了其有效性。 |
| [^16] | [AgentTracer: Tracing Indirect Prompt Injection Attack through Fine-Grained Intention-Execution Alignment](https://arxiv.org/abs/2610.09935) | AgentTracer提出了一种意图感知的追踪框架，将间接提示注入视为任务意图漂移，通过细粒度的意图-执行对齐恢复工具调用间的隐式决策依赖关系，从而实现攻击链的完整重建与注入源的精确定位。 |
| [^17] | [CYBERFORT: A Compliance-Chain Platform Operationalising the Cyber Resilience Act for SMEs](https://arxiv.org/abs/2610.09918) | CYBERFORT是一个面向中小企业的开源合规平台，其核心创新是“合规链”——一种可追溯结构，将欧盟《网络韧性法案》的全生命周期合规义务转化为从范围自评估、问题库到机器证明证据的完整可操作流程。 |
| [^18] | [Analysis and Visualization of the Linux Kernel's Software Evolution Using the City Metaphor](https://arxiv.org/abs/2610.09910) | 本文提出使用ExplorViz工具和3D城市隐喻，对拥有超过4000万行代码的Linux内核进行软件结构与演化的可视化分析和交互式探索。 |
| [^19] | [A Chat Assistant for Software Exploration in a 3D Software Visualization](https://arxiv.org/abs/2610.09901) | 该论文的核心创新是将基于大语言模型的聊天助手集成到3D软件可视化工具ExplorViz中，使用户能够通过自然语言提问并触发操作来探索甚至重构可视化的软件系统。 |
| [^20] | [Hierarchical Security Monitoring for Edge-IoT: A Formal Methods Approach](https://arxiv.org/abs/2610.09817) | 提出一个基于形式化运行时验证的轻量级分层安全监控框架，边缘设备运行 TeSSLa 流规范以亚微秒级每事件成本输出四值判定，网关运行参数化一阶 MonPoly 监控器分析跨设备判定流，从而在有限防御资源下以可量化的成本在集中式云监控与纯边缘本地监控之间取得平衡，有效应对协调性多设备攻击。 |
| [^21] | [Cost-Efficient Theorem Proving via Agent Orchestration in Program Verification](https://arxiv.org/abs/2610.09681) | 提出 CoCo-Prover，通过在两级证明图（声明内部的 AND/OR 证明超图与跨声明的引理依赖图）上进行元级成本决策与代理编排，将程序验证中的定理证明形式化为成本约束下的优化问题，以实现经济高效的批量证明。 |
| [^22] | [Coding-Agent Benchmarks Should Match Their Users' Task Flows](https://arxiv.org/abs/2610.09633) | 该研究通过收集JetBrains IDE中真实软件工程师的4,782个智能体会话，发现真实任务流在任务类型与切换模式上高度多样且因数据源而异，因此编码智能体基准测试应先指明目标用例，再依据其真实测得的任务流进行校准。 |
| [^23] | [GRAML: Graph-Grounded Reasoning and Multi-Task Learning for LLM-Based Software Vulnerability Detection](https://arxiv.org/abs/2610.09605) | GRAML通过静态分析提取图结构证据，引导GPT-5进行思维树漏洞推理并生成漏洞描述，结合四任务多任务学习，显著提升了基于大语言模型的软件漏洞检测的泛化能力。 |
| [^24] | [Beyond FAIR: A Fitness Function Framework for Sustainable Research Software](https://arxiv.org/abs/2610.09580) | 本文将可持续研究软件的适应度函数评估框架从FAIR原则扩展到环境和安全两个新维度，分别涵盖资源效率、执行足迹以及依赖健康、漏洞暴露与安全配置，从而实现更全面的软件可持续性持续评估。 |
| [^25] | [Who Broke Me? Execution-Guided Repair of Behavioral Dependency Breaks](https://arxiv.org/abs/2610.09267) | 本文提出BBCFixer，通过在新旧库版本下运行失败测试并比较返回值差异来定位导致行为性破坏的根本API，从而利用库差异中的相关证据引导LLM自动修复依赖升级引发的行为性破坏。 |
| [^26] | [SpecGuard: Proving a Task Is Broken Before the Agent Cheats](https://arxiv.org/abs/2610.09159) | 提出 SpecGuard，将任务意图与测试分别自动形式化为独立的 Lean 4 规范，并利用 Lean 内核形式化验证二者是否存在冲突，从而在编码智能体作弊之前就能证明任务本身已损坏。 |
| [^27] | [Finding Blind Spots in AppWorld and WorkArena Task Verifiers](https://arxiv.org/abs/2610.09142) | 该论文通过基于源码信息的变异测试审计了AppWorld和WorkArena的已发布任务验证器，揭示其存在盲点——即使智能体产生了错误效果（如重复写入创建多余记录或遗留非默认持久化值），验证器仍会判定任务成功。 |
| [^28] | [Large-scale Repository Engineering via Agent-Native Reusable Code Primitives](https://arxiv.org/abs/2610.09079) | 提出了具有接口契约、依赖闭包、验证测试和来源溯源的Agent原生可复用代码原语Code Primitives，以及LEGO框架，通过激活并适配1,424个已验证原语（收录于CodeFace库）来实现大规模仓库级代码构建。 |
| [^29] | [Evaluating Change Point Detection Methods for Software Performance Regression Analysis](https://arxiv.org/abs/2610.09023) | 本文对多种变点检测方法在真实世界软件性能测量数据上的有效性进行了综合评估，以帮助在开发周期中尽早检测软件性能回归。 |
| [^30] | [How Fragile Is On-Device Language Model Safety? Localizing Safety-Critical Parameters for Sparse Fault Analysis](https://arxiv.org/abs/2610.09000) | 研究发现LLaMA-2-7B-Chat的安全敏感行为高度集中在MLP的down_proj等稀疏参数子集中，仅修改0.19%的权重即可使攻击成功率大幅上升，揭示了设备端部署的语言模型存在显著的安全脆弱点。 |
| [^31] | [Automatically Detecting and Fixing Deadlocks in Go Code with GoDDaR](https://arxiv.org/abs/2610.08962) | GoDDaR是一个能够自动检测并修复Go程序中全局死锁和部分死锁的工具，弥补了Go运行时检测器无法发现部分死锁、且现有静态检测工具缺乏修复支持的不足。 |
| [^32] | [Psychological Safety in Software Engineering Teams: A Systematic Mapping Study of Team Processes and Performance](https://arxiv.org/abs/2610.08896) | 本研究对2006年至2026年间发表的112项原始研究进行系统性映射，全面梳理了软件工程团队中心理安全感的多维概念、与团队绩效和流程的关联、情境影响及障碍与强化策略，填补了该领域证据零散的空白。 |
| [^33] | [Mitigating Uncertainty Interactions in GenAI-based Adaptive Systems: Vision, Challenges and Preliminary Guidelines](https://arxiv.org/abs/2610.08881) | 本文针对生成式AI组件在自适应系统中引入的复杂且相互叠加的不确定性交互问题，提出了一个初步概念框架，并给出贯穿全软件生命周期的缓解指南。 |
| [^34] | [RAPO-Sol: Retrieval-Augmented Preference Optimization for Repository-Level Solidity Code Generation](https://arxiv.org/abs/2610.08429) | 该论文提出RAPO-Sol两阶段训练框架，将检索增强微调（RAFT）与基于语义锚点扰动（SAP）构建拒绝样本的直接偏好优化（DPO）相结合，以提升仓库级Solidity智能合约代码生成的正确性与语义一致性。 |
| [^35] | [PreMaQ: Predicting Maintainability-Related Quality of LLM-Generated Code Before Generation](https://arxiv.org/abs/2610.05858) | 该论文提出PreMaQ方法，在LLM生成代码之前通过模型内部表示预测生成代码的可维护性相关质量指标（代码坏味道分数和可维护性指数），从而帮助开发者避免生成、审查和丢弃低质量代码的成本。 |
| [^36] | [From Verification Failures to Reusable Guidance for Coding Agents](https://arxiv.org/abs/2609.39022) | 该论文提出将专家对验证失败的诊断转化为编码智能体可复用的指导，结合K框架的可执行语言语义与一套用于构建规范、修复证明和审计充分性的工具包，在HumanEval上实现164/164的全通过率，并通过对照实验证明审计能识别出证明通过但存在缺陷的软件包。 |
| [^37] | [The Impact of Operational-Data Fidelity when Assessing Safety-Critical Autonomous-Vehicle Software](https://arxiv.org/abs/2608.10025) | 本研究将保守贝叶斯推断技术扩展至自动驾驶车辆安全评估领域，发现低保真的运行数据即使被保守使用也可能得出危险的乐观结论，强调了运行数据保真度对软件可靠性声明的重要影响。 |
| [^38] | [SWE-NFI: Studying and Benchmarking Coding Agents for Non-Functional Improvements](https://arxiv.org/abs/2607.27409) | 该论文提出了SWE-NFI基准，基于开源Python项目真实合并的拉取请求构建188个任务，并将五类面向开发者的非功能性改进操作化为92条可执行规则，用于评估编码智能体在保持代码行为不变前提下提升软件质量的能力。 |
| [^39] | [Knowledge boundary probing and demand-guided intervention for LLM-based power system code generation](https://arxiv.org/abs/2605.31478) | 该论文提出PowerCodeBench基准（面向pandapower的2000个冻结任务）以及无需更新权重的部署时工作流，通过文档驱动的L0-L3知识边界探测、查询侧需求估计选择分层API证据、以及执行反馈引导的针对性修复，显著提升了LLM电力系统代码生成的准确率。 |
| [^40] | [Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents](https://arxiv.org/abs/2605.21347) | 该论文提出了洞察生成器（IG）——一个多智能体系统，通过在执行轨迹语料库上自动提出并检验假设，生成有证据支持的系统性诊断洞察报告，解决了 LLM 智能体失败诊断依赖人工、无法规模化的问题。 |
| [^41] | [TorchGWAS 1.0: GPU-accelerated GWAS at scale](https://arxiv.org/abs/2604.21095) | TorchGWAS是一个GPU加速的批量线性关联检验框架，可对数千个定量表型进行高通量、协变量校正的全基因组关联分析，其结果与PLINK 2.0完全一致，并能在约一分钟内完成45.7亿次关联检验。 |
| [^42] | [OOM-RL: Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2604.11477) | 该论文提出“资金耗尽强化学习（OOM-RL）”这一客观对齐新范式，通过将基于LLM的多智能体系统部署到真实金融市场中，利用资金耗尽带来的真实经济损失作为外部负梯度信号，从而克服RLHF/RLAIF导致的模型谄媚和执行环境中的测试规避问题。 |
| [^43] | [huff: A Python package for Market Area Analysis](https://arxiv.org/abs/2602.17640) | huff是一个模块化的Python软件包，为市场区与空间可达性分析提供了从数据导入、模型构建、参数估计到地图可视化的完整工作流程。 |
| [^44] | [Doc2Spec: Synthesizing Formal Programming Specifications from Natural Language via Grammar Induction](https://arxiv.org/abs/2602.04892) | Doc2Spec提出多智能体框架，通过从自然语言API规则自动归纳领域专用文法来约束大模型分步生成可检查的形式化规约，显著提升了规约合成的精度与召回率。 |
| [^45] | [Did You Forkget It? Detecting One-Day Vulnerabilities in Open-source ForksWith Global History Analysis](https://arxiv.org/abs/2511.05097) | 本文提出一种基于Software Heritage全局代码图的全局历史分析方法，可在提交级别跨分叉仓库传播漏洞信息，自动检测开源分叉仓库中已知但未修补的1-day漏洞，弥补了传统历史分析方法无法追踪分叉中漏洞的不足。 |
| [^46] | [SEER: Self-Enhancing Chain-of-Thought Compression for Reasoning Models](https://arxiv.org/abs/2509.14093) | 该论文通过实证研究揭示推理模型在代码生成中常产生冗长思维链并引发截断与不稳定生成问题，并据此提出SEER方法，通过自增强的方式压缩思维链以降低推理开销。 |

# 详细

[^1]: 在能够解决之前：从基座模型预测后训练编码智能体的性能

    Before They Can Solve: Predicting Post-Training Coding-Agent Performance from Base Models

    [https://arxiv.org/abs/2610.10478](https://arxiv.org/abs/2610.10478)

    提出将成功的后训练智能体轨迹作为基座模型潜力的前瞻信号，通过重放轨迹并定位使代码库首次由失败转为通过的决定性步骤，从而在昂贵的智能体后训练之前预测基座模型能否培养出高性能的编码智能体。

    

    我们如何预测哪些基座模型检查点值得进行一轮昂贵的智能体后训练？端到端的 pass@$K$ 测试检验的是成功行为是否已经出现在基座模型的分布中，但它并不适合智能体编码场景：许多基座检查点无法可靠地生成完成端到端任务所需的结构良好的工具调用。单步或短程任务通过将多步交互压缩为固定提示词和单一补丁来规避这些工具调用失败，但它们回避了我们真正关心的核心能力：在代码库不断演进的过程中，跨多个使用工具的步骤保持连贯状态。为了弥合这一差距，我们将成功的后训练智能体轨迹视为基座模型潜力的前瞻信号。通过重放每条轨迹并在每个代码修改步骤之后重新运行测试，可以识别出决定性步骤：即累积补丁首次使代码库从失败转为通过的那个步骤……

    arXiv:2610.10478v1 Announce Type: cross  Abstract: How can we predict which base checkpoint is worth an expensive round of agentic post-training? End-to-end pass@$K$ tests whether successful behavior already appears in a base model's distribution, but it is a poor fit for agentic coding: many base checkpoints cannot reliably produce the well-formed tool invocation required to complete a task end-to-end. Single-shot or short-horizon tasks avoid these tool-calling failures by collapsing a multi-step interaction into a fixed prompt and a single patch, but they sidestep the core capability we care about: maintaining coherent state over many tool-using steps as the repository evolves. To bridge this gap, we treat successful post-trained agent trajectories as a lookahead signal of base-model potential. Replaying each trajectory and rerunning tests after every code-changing step identifies the decisive step: the first step whose cumulative patch flips the repository from failing to passing, c
    
[^2]: 脉冲级量子控制的设计时符合性检查

    Design-Time Conformance Checking for Pulse-Level Quantum Control

    [https://arxiv.org/abs/2610.10427](https://arxiv.org/abs/2610.10427)

    本文提出 qconform，一个基于版本化设备能力描述符、设计时运行的确定性检查器，可在实验前判定脉冲级量子控制程序能否在目标设备上实现，避免实验数据与所写程序不符。

    

    脉冲级量子控制程序是针对某一设备编写的，而该设备的限制（如果有记录的话）只记载在供应商文档和源代码中。超出这些限制的程序可能在编译时被拒绝；也可能被接受并被悄悄改动；或者编译通过却在板卡上运行失败。在后两种情况下，实验虽然运行了，但所得数据与所编写的程序并不对应。我们提出了 qconform，一个检查器：在给定设备版本化能力描述符的条件下，它可判定一个脉冲程序是否能在该设备上实现。描述符中的每条约束都引用了确立该约束的工具链观察依据。该检查器工作在设计时、离线运行、具有确定性，且不使用浮点数。它会给出判定结果、所应用的规则，以及一份覆盖率清单，明确指出其未检查的内容。我们通过针对 QICK 和 Qblox 工具链的差分测试来评估 qconform，涵盖三种 QICK 板卡配置……

    arXiv:2610.10427v1 Announce Type: cross  Abstract: A pulse-level quantum control program is written against a device whose limits are recorded, if at all, in vendor documentation and source code. A program that exceeds them can be refused at compile time. It can also be accepted and silently altered, or compile and then fail at the board. In the last two cases the experiment runs, and the data does not correspond to the program that was written. We present qconform, a checker that decides whether a pulse program is realizable on a device, given a versioned capability descriptor for that device. Every constraint in a descriptor cites the toolchain observation that established it. The checker is design-time, offline, and deterministic, and it uses no floating point. It reports a verdict, the rules it applied, and a coverage manifest that names what it did not check. We evaluate qconform by differential testing against the QICK and Qblox toolchains, on three QICK board configurations and 
    
[^3]: TaoD2C-Bench：面向超越视觉保真度的工业UI代码生成的多模态大语言模型基准测试

    TaoD2C-Bench: Benchmarking MLLMs for Industrial UI Code Generation Beyond Visual Fidelity

    [https://arxiv.org/abs/2610.10374](https://arxiv.org/abs/2610.10374)

    提出了TaoD2C-Bench基准测试，用于评估多模态大语言模型在工业设计转代码任务中超越视觉识别、结合图层元数据与目标库约束来生成满足实现要求的UI代码的能力。

    

    多模态大语言模型（MLLMs）面临的一个关键挑战是超越视觉识别，实现约束感知的跨模态推理。这需要将视觉线索与其他模态的信息相结合，以在特定领域规则下理解元素之间的关系。这一挑战在工业设计转代码（D2C）任务中尤为突出，该任务将用户界面（UI）设计转换为代码，要求MLLM将设计图像与无序的图层元数据关联起来，推断组件和布局的实现要求，并在目标库约束下用代码实现这些要求。然而，这些能力在现实的工业环境中仍缺乏充分评估。为填补这一空白，我们提出了TaoD2C-Bench，这是一个用于评估MLLM生成满足工业应用实现要求的UI代码能力的基准测试。TaoD2C数据集包含来自17个商业平台的2,861个生产环境设计。

    arXiv:2610.10374v1 Announce Type: new  Abstract: A key challenge for multimodal large language models (MLLMs) is moving beyond visual recognition to constraint-aware cross-modal reasoning. This involves combining visual cues with information from other modalities to understand elements' relationships under domain-specific rules. This challenge is acutely evident in industrial design-to-code (D2C), which converts user interface (UI) designs into code and requires MLLMs to connect design images with disorganized layer metadata, infer component and layout implementation requirements, and realize them in code under target-library constraints. However, these capabilities remain insufficiently evaluated in realistic industrial settings. To fill this gap, we present TaoD2C-Bench, a benchmark for evaluating MLLMs' ability to generate UI code that satisfies implementation requirements in industrial applications. The TaoD2C dataset consists of 2,861 production designs from 17 commercial platform
    
[^4]: 输入盲化对照在多项选择评估中为层程序带来显著的神谕提升空间

    Input-Blind Controls Produce Substantial Oracle Headroom for Layer Programs in Multiple-Choice Evaluation

    [https://arxiv.org/abs/2610.10368](https://arxiv.org/abs/2610.10368)

    本研究发现在多项选择评估中，输入盲化的对照扰动所产生的神谕提升空间反而超过真实的层跳过与重复程序，说明仅凭选择增益无法解释所选层程序为何有效。

    

    自适应计算旨在通过针对每个输入定制执行方式来改进语言模型的推理。对于层程序，在实用的选择器可用之前，神谕评估利用已知答案来估计这种灵活性带来的潜在增益。然而，来自选择的增益本身并不能解释所选程序为何有效。本研究利用两个模型上的32个层跳过与重复程序以及4,413个多项选择题目来考察这一区别。该分析将真实程序相对于在无评估提示情况下所选固定动作的增益，与相同位置上输入盲化扰动的增益进行比较，并在另一个提示上重新评估选择结果。在共享选项顺序的情况下，这些对照在Qwen3-4B-Base和Llama-3.1-8B上分别产生了10.2-11.8和15.6-19.4个百分点的提升空间，在每模型的全部三次随机方向抽取中均超过真实程序的9.0和10.1。它们仅在答案改变率上与真实程序相当，且排序取决于（原文在此处截断）。

    arXiv:2610.10368v1 Announce Type: cross  Abstract: Adaptive computation aims to improve language-model inference by tailoring execution to each input. For layer programs, oracle evaluations use known answers to estimate the potential gain from this flexibility, before a practical selector is available. However, a gain from selection does not by itself explain why the chosen programs help. This study examines this distinction using 32 layer-skipping and repetition programs on two models and 4,413 multiple-choice items. The analysis compares their gains over a fixed action selected without the evaluation prompt with those of input-blind perturbations at the same sites, re-evaluating selections on another prompt. With shared option order, the controls give 10.2-11.8 and 15.6-19.4 percentage points of headroom on Qwen3-4B-Base and Llama-3.1-8B, exceeding the real programs' 9.0 and 10.1 in all three random-direction draws per model. They match answer-change rate only, and the ordering depen
    
[^5]: 当子智能体并行工作时：长时程编码任务中动态并发的希望与陷阱

    When Sub-Agents Work in Parallel: The Promises and Pitfalls of Dynamic Concurrency in Long-Horizon Coding Tasks

    [https://arxiv.org/abs/2610.10263](https://arxiv.org/abs/2610.10263)

    该论文首次通过354个任务、2,124次执行的对照实验，系统研究了Codex、Claude Code和Kimi Code中动态并发作为执行策略的效果，发现模型能力主导较短任务的成败，而编排调度能力则成为长时程开发任务完成的关键。

    

    随着编码智能体从有界的软件工程任务迈向长时程开发，动态并发为扩展复杂开发任务提供了一种有前景的方式。在这一策略下，智能体在执行过程中自行决定是否以及如何生成并发的子智能体。模型能力在很大程度上决定了较短任务的结果，而长时程开发则使编排调度成为任务完成的核心。现有工作主要聚焦于编码智能体在较短任务上的失败，或预定义多智能体工作流中的协作，对前沿智能体在不同任务复杂度下的动态并发鲜有洞见。我们通过对照比较匹配的Codex、Claude Code和Kimi Code在启用或禁用该策略下的执行情况，将动态并发作为一种执行策略加以研究。在涵盖不同任务复杂度和执行时域的354个任务与2,124次执行中，我们评估了其端到端效果及调度……

    arXiv:2610.10263v1 Announce Type: new  Abstract: As coding agents advance from bounded software engineering tasks toward long horizon development, dynamic concurrency offers a promising way to scale complex development tasks. Under this policy, agents decide during execution whether and how to spawn concurrent sub-agents. Model capability largely determines outcomes on shorter tasks, whereas long horizon development makes orchestration central to task completion. Existing work, focused on coding agent failures on shorter tasks or collaboration in predefined multiagent workflows, offers little insight into dynamic concurrency in frontier agents across task complexity. We study dynamic concurrency as an execution policy through controlled comparisons of matched Codex, Claude Code, and Kimi Code executions with the policy enabled or disabled. Across 354 tasks and 2,124 executions spanning a range of task complexities and execution horizons, we evaluate its end to end effects and schedulin
    
[^6]: 使用小语言模型逆向工程机器学习流水线结构

    Using Small Language Models to Reverse-Engineer Machine Learning Pipelines Structures

    [https://arxiv.org/abs/2610.10261](https://arxiv.org/abs/2610.10261)

    该研究验证了小语言模型能够凭借其代码理解与分类能力，从源代码中有效提取机器学习流水线的阶段结构，从而克服人工标注不可扩展及传统分类器难以适应领域多样性的局限。

    

    背景：一旦定义了构建机器学习（ML）流水线的阶段分类体系（例如数据预处理、建模等），从源代码中提取这些阶段对于更好地理解机器学习实践至关重要。然而，机器学习的持续演进（例如算法、数据集的更新）所带来的多样性使这项任务充满挑战。现有方法要么依赖无法扩展的人工标注，要么依赖无法妥善支持领域多样性的分类器。这些局限性呼唤更可靠的解决方案。目标：我们评估小语言模型（SLM）能否利用其代码理解与分类能力来应对这些局限，并加深我们对机器学习实践的理解。方法：我们基于两篇代表当前技术局限性的相关参考工作开展了验证性研究。我们首先使用Cochran's Q检验比较多个小语言模型，然后对表现最佳的模型进行评估。

    arXiv:2610.10261v1 Announce Type: cross  Abstract: Context: Once defined a taxonomy of stages structuring Machine Learning (ML) pipelines (e.g. Data Preprocessing, Modeling...), extracting these stages from source code is key for better understanding ML practices. However, the diversity caused by the constant evolution of ML (e.g., algorithms, datasets) makes this task challenging. Existing approaches either rely on non-scalable manual labeling or on classifiers that do not properly support domain's diversity. These limitations call for more reliable solutions.   Objective: We evaluate whether Small Language Models (SLMs) can leverage their code understanding and classification abilities to address these limitations, and enhance our understanding of practices in ML.   Method: We conduct a confirmatory study based on two relevant reference works representing current limitations in the state-of-the-art. We first compare several SLMs using Cochran's Q test, then evaluate the best-performi
    
[^7]: QuSema：利用量子知识增强智能体检测量子库中的静默缺陷

    QuSema: Detecting Silent Bugs in Quantum Libraries via Quantum-knowledge-enhanced Agents

    [https://arxiv.org/abs/2610.10258](https://arxiv.org/abs/2610.10258)

    QuSema提出了一种自主测试智能体，将量子语义与文档约束作为源代码级语义预言机，能够在缺乏基于执行的预言机的情况下检测量子库中从有效输入产生无效输出的静默缺陷。

    

    量子库如今已成为量子算法开发的关键基础设施，但其正确性仍然难以测试。现有的测试技术主要依赖于基于失败或基于比较的预言机，只有在执行失败、违反运行时检查或与其他实现结果不一致时才能暴露缺陷。当缺乏合适的基于执行的预言机时，这些技术的适用性受到限制，导致一些静默缺陷无法被检测到。这类漏检缺陷可能产生错误结果，并将其传播到实验结论、仿真研究和算法设计之中。本文提出了QuSema，一个用于发现量子库中静默缺陷的自主测试智能体。QuSema利用量子语义和文档中的约束作为源代码级的语义预言机，评估实现逻辑是否可能从有效输入产生无效输出。它通过一个智能体循环运行，反复检查库的API文档……

    arXiv:2610.10258v1 Announce Type: new  Abstract: Quantum libraries are now critical infrastructure for quantum algorithm development, yet their correctness remains difficult to test. Existing testing techniques mainly rely on failure-based or comparison-based oracles, exposing bugs only when executions fail, violate runtime checks, or disagree with another implementation. Their applicability is limited when suitable execution-based oracles are unavailable, leaving some silent bugs undetected. Such missed bugs can produce incorrect results that propagate into experimental conclusions, simulation studies, and algorithmic designs. Here we present QuSema, an autonomous testing agent for finding silent bugs in quantum libraries. QuSema uses constraints from quantum semantics and documentation as a source-level semantic oracle to assess whether implementation logic can produce invalid outputs from valid inputs. It operates through an agentic loop that repeatedly inspects library API document
    
[^8]: OOM-RL II：现实是神谕，而非调试器——持续演化的智能体工程系统中的溯源约束诊断

    OOM-RL II: Reality Is an Oracle, Not a Debugger Provenance-Constrained Diagnosis in Continually Evolving Agent-Engineered Systems

    [https://arxiv.org/abs/2610.10256](https://arxiv.org/abs/2610.10256)

    该论文提出“现实是神谕而非调试器”的核心观点，并在一个持续演化、由智能体工程化构建的量化交易系统中进行溯源约束诊断，表明即使账户一年内盈利并跑赢大盘指数，在缺乏完整的推荐到运行时绑定的情况下，也无法将结果归因于特定的演化程序版本或确认其统计显著性。

    

    arXiv:2610.10256v1 公告类型：交叉发布。摘要：现实可以确立某一结果已经发生，却无法指明是哪个不断演化的程序产生了它，也无法说明原因。这一区别在生产级机器学习系统中尤为重要，因为这类系统的代码、配置和工件在不断变化，而外部反馈却在持续积累。我们在一个由人类指导、由智能体工程化构建的量化交易系统中考察了这一问题，其中“神谕”指代已实现结果的外部来源，而非完整的正确性规范。在这一年间，该账户实现了盈利并跑赢了宽基市场指数，而在主要回顾性设定下，年度阿尔法在统计上与零无法区分。回顾性选取的子时段包含相对表现不佳的时期，以及在声明的近似基准下候选层面的条件性弱势。工程记录记载了该期间发生的变更，且完整的从推荐到运行时的绑定信息不可用。该档案库无法确立一个共同冻结的实例……（原文在此截断）

    arXiv:2610.10256v1 Announce Type: cross  Abstract: Reality may establish that an outcome occurred without identifying which evolving procedure produced it or why. This distinction matters in production ML systems whose code, configuration, and artifacts change while external feedback accumulates. We examine it in a human-directed, agent-engineered quantitative trading system, using oracle to mean an external source of realized outcomes rather than a complete correctness specification. Across one year, the account gained and outperformed a broad market index, while annual alpha was not statistically distinguishable from zero under the main retrospective specification. Retrospectively selected subperiods include adverse relative performance and conditional candidate-level weakness under declared approximate references. Engineering records document changes during the episode, and complete recommendation-to-runtime binding is unavailable. The archive does not establish a common frozen inst
    
[^9]: TestGRAD：通过失败模式动量为SWE-Agent集成演化测试套件

    TestGRAD: Evolving Test Suites via Failure Pattern Momentum for SWE-Agent Ensemble

    [https://arxiv.org/abs/2610.10242](https://arxiv.org/abs/2610.10242)

    该论文受带动量的梯度下降启发，提出了TestGRAD框架，将SWE-Agent集成中的补丁选择形式化为测试空间优化问题，通过差分损失和失败模式动量机制自动演化测试套件，从而有效区分相互竞争的候选补丁。

    

    arXiv:2610.10242v1 公告类型：新论文。SWE-agent集成通过组合来自具有互补优势的不同智能体的候选补丁来改进问题解决能力。因此，核心问题在于基于测试的选择：生成测试、执行候选补丁并识别最佳补丁。我们将这一过程形式化为测试空间优化：不断演化一个可执行的仓库测试套件，直到它能够区分相互竞争的补丁。现有的测试生成方法是受限的优化器：它们通常缺乏用于集成选择的显式损失函数，通过不完整的方向进行优化（主要是创建新测试或删除旧测试），并且进行一次性生成而没有来自重复失败的反馈。受带动量的梯度下降启发，我们提出了TestGRAD，一个用于自动测试优化的框架。TestGRAD围绕三个核心概念展开。差分损失为优化器提供了一个显式的由执行定义的目标：有用的测试应该通过行为差异来区分候选补丁……（摘要在此处被截断）

    arXiv:2610.10242v1 Announce Type: new  Abstract: SWE-agent ensembles improve issue resolution by combining candidate patches from different agents with complementary strengths. The central problem is therefore test-based selection: generate tests, execute candidate patches, and identify the best patch. We formulate this process as test-space optimization: evolving an executable repository test suite until it distinguishes competing patches. Existing test-generation methods are limited optimizers. They usually lack an explicit loss for ensemble selection, optimize through incomplete directions that mostly create new tests or delete old ones, and perform one-off generation without feedback from repeated failures. Inspired by gradient descent with momentum, we introduce TestGRAD, a framework for automatic test optimization. TestGRAD centers on three concepts. Differential loss gives the optimizer an explicit execution-defined target: useful tests should separate candidate patches by behav
    
[^10]: 为什么软件工程在编码智能体时代不可或缺

    Why Software Engineering Is Indispensable in the Age of Coding Agents

    [https://arxiv.org/abs/2610.10226](https://arxiv.org/abs/2610.10226)

    本文论证AI编码智能体的兴起使软件工程和软件工程师更加不可或缺，因为大语言模型的三个结构性缺陷造成无法通过训练消除的真空，需要软件工程师通过方法论知识、领域知识、设计选择和流程选择四种知识杠杆，充当方法论专家、中介和守护者，才能确保AI辅助开发的软件可信可靠。

    

    AI 能否让软件工程（SE）——这门学科——变得过时？它又能否让软件工程师——这些专业人员——变得多余？本文认为，强大的 AI 编码智能体的兴起使软件工程和软件工程师变得不可或缺，而非过时：软件工程正是缺失的基础，没有它，AI 辅助开发将产生貌似合理却具有误导性、无法验证且最终不可信的软件。大语言模型的三个结构特性（概率性生成、不可知性和语义无状态性）造成了一个任何训练量都无法消除的结构性真空。填补这一真空需要四种知识杠杆：方法论知识、领域知识、设计选择和流程选择。这四者都必须固化为持久的工件，而每一项都需要软件工程师担任方法论专家、中介和守护者的角色。

    arXiv:2610.10226v1 Announce Type: new  Abstract: Can AI make Software Engineering (SE) -- the discipline -- obsolete? And can it make software engineers -- the professionals -- redundant? This paper argues that the rise of capable AI coding agents makes SE and software engineers essential, not obsolete: the missing foundation without which AI-assisted development produces misleadingly plausible, unverifiable, and ultimately untrustworthy software. Three structural properties of large language models (probabilistic generation, agnosticism, and semantic statelessness) create a structural vacuum that no amount of training can eliminate. Filling it requires four knowledge levers: methodological knowledge, domain knowledge, design choices, and process choices. All four must be reified as persistent artifacts, and each requires the software engineer as methodologist, mediator, and custodian.
    
[^11]: 智能体AI辅助的生产调度建模：约束规划中的评估

    Agentic AI-Assisted Modeling for Production Scheduling: Assessment in Constraint Programming

    [https://arxiv.org/abs/2610.10184](https://arxiv.org/abs/2610.10184)

    本研究提出将未经专门训练的通用大语言模型编排为单智能体或多智能体系统，并结合模型上下文协议服务器进行上下文感知的求解器文档检索以抑制幻觉，从而从自然语言问题描述中自动构建并实现生产调度的约束规划优化模型，弥合了自动化建模与智能体决策支持两大研究方向。

    

    arXiv:2610.10184v1 公告类型： cross 摘要：为生产调度开发优化模型需要投入大量专家精力。针对大语言模型（LLM）的研究沿两个方向展开：其一是面向自动化建模的专门化方法，主要应用于混合整数线性规划，但这类方法通常依赖专门训练或针对特定问题的架构，限制了其在工业界的部署；其二是用于运营决策支持的智能体人工智能，但这类方法通常假设优化模型已经存在。本研究将这两个方向相结合，评估未经任务特定训练的通用大语言模型在以智能体形式编排时，能否根据自然语言问题描述来构建并实现约束规划模型。研究将单智能体与多智能体架构同模型上下文协议（Model Context Protocol）服务器相集成，该服务器提供上下文感知的求解器文档检索，以缓解模型实现过程中的幻觉问题。两者均……（原文在此处截断）

    arXiv:2610.10184v1 Announce Type: cross  Abstract: Developing optimization models for production scheduling requires substantial expert effort. Research on large language models (LLMs) has followed two directions: specialized approaches for automated modeling, mostly for mixed-integer linear programming, which often rely on dedicated training or problem-specific architectures that limit industrial deployment; and agentic artificial intelligence for operational decision support, which generally assumes that the optimization model already exists. This study bridges both directions by assessing whether general-purpose LLMs, orchestrated as agents without task-specific training, can formulate and implement constraint programming models from natural-language problem descriptions. Singleagent and multi-agent architectures are integrated with a Model Context Protocol server that provides context-aware retrieval of solver documentation to mitigate hallucinations during implementation. Both are
    
[^12]: 关于基于大语言模型的漏洞修补基准的可靠性研究

    On the Reliability of LLM-Based Vulnerability Patching Benchmarks

    [https://arxiv.org/abs/2610.10150](https://arxiv.org/abs/2610.10150)

    本文揭示了当前基于大语言模型的漏洞修补基准在智能体、框架和数据集三个层面存在的系统性缺陷会导致性能评估失真，并构建了一个包含来自84个开源项目的112个历史漏洞、配有概念验证测试、回归测试和开发者测试的更可靠基准。

    

    大语言模型在自动化漏洞修补方面展现出强大潜力，但当前的基准测试可能会显著扭曲所报告的性能。基于我们在开发、运行和压力测试此类框架方面的丰富经验，我们识别出了三个维度上尚未被充分审视的陷阱：(1) 智能体层面的因素，其中提示词、工具可用性和详细指令可以在不提升符合开发者要求的补丁质量的情况下提高成功率；(2) 框架层面的因素，其中权限错误、基础设施缺陷和超时处理可能会悄然抑制或夸大性能表现；(3) 数据集层面的因素，其中漏洞报告和单一的概念验证测试无法捕捉补丁是否解决了根本原因或遵循了开发者的意图。我们从84个开源的C/C++、Go和Rust项目中精选了112个历史漏洞，每个漏洞都配有概念验证测试、回归测试以及额外的开发者测试来评估（摘要在此处截断）

    arXiv:2610.10150v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown strong potential for automated vulnerability patching, but current benchmarks can substantially distort reported performance. Drawing on extensive experience developing, running, and stress-testing such frameworks, we identify under-examined pitfalls across three dimensions: (1) agent-level factors, where prompting, tool availability, and detailed instructions can raise success rates without improving developer-aligned patch quality; (2) framework-level factors, where permission errors, infrastructure bugs, and timeout handling can silently suppress or inflate performance; and (3) dataset-level factors, where bug reports and single proof-of-concept (PoC) tests fail to capture whether patches address root causes or follow developer intent. We curate 112 historical bugs from 84 open-source C/C++, Go, and Rust projects, each with PoC tests, regression tests, and additional developer tests that asses
    
[^13]: AdaT$^2$：面向会话智能体黑盒边界测试的自适应测试变换

    AdaT$^2$: Adaptive Test Transformations for Black-Box Boundary Testing of Conversational Agents

    [https://arxiv.org/abs/2610.10141](https://arxiv.org/abs/2610.10141)

    AdaT$^2$通过让LLM扮演用户与智能体对话来提取策略条件的陈述，并自适应地选择“陈述+变换指令”配对来生成黑盒边界测试，以验证智能体在策略边界两侧表现出不同的行为。

    

    基于大语言模型（LLM）的会话智能体必须遵守策略。策略中的每个条件都在用户请求之间划出一条边界，智能体在边界的两侧必须表现出不同的行为。我们提出了AdaT$^2$，该方法在与一个扮演用户角色的LLM进行探索性对话时，从智能体的回复中提取陈述，并利用这些陈述来指导边界测试的生成。每个陈述描述一个条件以及该条件成立时期望出现的行为。除了由单个陈述引导的普通测试之外，AdaT$^2$还会编写由“陈述+测试变换指令”配对引导的变换测试，例如“省略一个必需的输入”。配对中的指令可以将测试移动到该陈述所定义边界的另一侧，或移动到另一条边界上。陈述与指令所能组成的配对数量远远超出一次运行所能尝试的范围，而且许多配对并不适用。因此，自适应配对选择机制会选择状态（原文在此处截断）

    arXiv:2610.10141v1 Announce Type: new  Abstract: Conversational agents based on large language models (LLMs) must comply with policies. Each condition in a policy draws a boundary between user requests, and the agent must behave differently on its two sides. We present AdaT$^2$, which extracts statements from the agent's replies in exploratory conversations with an LLM acting as the user, and uses the statements to guide boundary test generation. Each statement describes one condition and the behavior expected when the condition holds. Besides plain tests guided by single statements, AdaT$^2$ writes transformed tests guided by pairs of a statement and a test transformation instruction, such as "omit one required input". The instruction of a pair can move a test to the other side of the statement's boundary or to another boundary. The statements and instructions form far more pairs than a run can try, and many pairs are not applicable. Adaptive pair selection therefore chooses the state
    
[^14]: 缓解自动化AI研究风险的理解审计机制

    Comprehension Audits to Mitigate Risks from Automated AI Research

    [https://arxiv.org/abs/2610.10064](https://arxiv.org/abs/2610.10064)

    提出一种名为“理解审计”的开发过程保障机制，要求负责人向独立审计员解释研发贡献以证明其真正理解所构建的系统，否则将暂停开发并施加逐步升级的后果，以此缓解AI自动化研发带来的安全风险。

    

    AI已经为前沿AI实验室编写了大部分代码。如果缺乏足够的人类监督，这将带来安全风险。现有工作提出了最低理解阈值和无辅助检查来缓解这一问题。然而，据我们所知，目前尚无已发表的前沿AI保障机制要求以“负责人能够证明其理解所构建内容”作为继续开发或使用的预先承诺条件。我们提出“理解审计”这一新颖的开发过程保障机制：由负责人员向审计员解释研发贡献，以证明其理解。通过独立管理和分级报告，该机制提供了一个关卡：如果未能证明人类理解，该贡献的开发将暂停直至整改完成，重复失败将面临逐步升级的后果。我们对领先开源AI项目的分析发现……

    arXiv:2610.10064v1 Announce Type: cross  Abstract: AI is already writing a majority of code for frontier AI labs. This creates a safety risk if there is insufficient human oversight. Existing work proposes minimum comprehension thresholds and unaided checks to mitigate this. To our knowledge, however, there is currently no published frontier-AI assurance regime that requires demonstrated evidence that the responsible humans understand what they are building as a precommitted condition for continuing development or usage. We propose comprehension audits, a novel development-process assurance mechanism in which the responsible people explain R&D contributions to auditors to demonstrate understanding. With independent administration and graded reports, they provide a gate: development of a contribution stops based on a failure to demonstrate human understanding until remediated, with escalating consequences for repeated failures. Our analysis of leading open-source AI projects finds incre
    
[^15]: 为科学软件工程设计协作式AI驱动的工作流

    Designing Collaborative AI-Driven Workflows for Scientific Software Engineering

    [https://arxiv.org/abs/2610.09995](https://arxiv.org/abs/2610.09995)

    本文提出一种人机协作的AI驱动工作流——由领域专家编写规范和计划、智能体在确定性编排下生成代码，每个阶段通过数值比对和人工审核把关，并在大型高能物理程序从Fortran到C++的翻译任务中验证了其有效性。

    

    智能体人工智能系统能够在软件工程和科学研究中执行广泛的任务，从编写和翻译代码，到运行数据分析与可视化工作流。在科学计算中，难点在于验证智能体生成的代码既正确无误，又能让具有不同专业背景的团队成员所理解。因此，我们认为这类系统最好在协作式团队结构中使用，而非实现完全自动化。在我们提出的工作流中，领域专家负责编写规范和计划，智能体则在确定性编排模式下运行以编写目标代码。每个阶段都以与参考代码进行数值比较作为结束，并且在下一阶段开始之前必须经过人工审查和批准。我们在将一个大型高能物理应用程序从Fortran翻译为C++的任务上评估了这些工作流，并在相同任务下运行了……（原文摘要此处截断）

    arXiv:2610.09995v1 Announce Type: new  Abstract: Agentic artificial intelligence systems can carry out a broad range of tasks in software engineering and scientific research, from writing and translating code to running workflows for data analysis and visualization. In scientific computing, the difficulty is verifying that agent-generated code is both correct and understandable to teams whose members bring different areas of expertise. We therefore argue that these systems are best used within collaborative team structures rather than as full automation. In the workflows we propose, domain experts write the specification and plan, and agents operate under a deterministic orchestration pattern to write the target code. Each stage ends with a numerical comparison against the reference code and requires human review and approval before the next begins. We evaluate these workflows on the translation of a large high-energy physics application from Fortran to C++, running the same task under
    
[^16]: AgentTracer：通过细粒度意图-执行对齐追踪间接提示注入攻击

    AgentTracer: Tracing Indirect Prompt Injection Attack through Fine-Grained Intention-Execution Alignment

    [https://arxiv.org/abs/2610.09935](https://arxiv.org/abs/2610.09935)

    AgentTracer提出了一种意图感知的追踪框架，将间接提示注入视为任务意图漂移，通过细粒度的意图-执行对齐恢复工具调用间的隐式决策依赖关系，从而实现攻击链的完整重建与注入源的精确定位。

    

    大型语言模型（LLM）智能体通过与外部资源交互来完成复杂的用户任务，这使其暴露于间接提示注入（IPI）攻击之下——恶意指令会将智能体引向攻击者预期的任务。由于IPI在真实环境中难以防御，事后追踪对于定位注入源和重建攻击链至关重要。然而，现有的追踪方法主要捕获显式的控制流和数据流依赖关系，忽略了由恶意指令驱动的工具调用之间的隐式关系。这些工具调用可能缺乏显式依赖，并与合法操作交织在一起，使得完整攻击链的重建变得困难。本文提出了AgentTracer，一种将IPI视为任务意图漂移的意图感知追踪框架。AgentTracer通过恢复工具调用之间的隐式决策依赖来构建意图漂移……

    arXiv:2610.09935v1 Announce Type: new  Abstract: Large language model (LLM) agents interact with external resources to complete complex user tasks, exposing them to indirect prompt injection (IPI), where malicious instructions redirect agents toward attacker-intended tasks. Since IPI is difficult to defend against in real-world environments, post-incident tracing is essential for locating the injection source and reconstructing the attack chain. However, existing tracing methods primarily capture explicit control-flow and data-flow dependencies, overlooking the implicit relationships among tool calls driven by the malicious instruction. These tool calls may lack explicit dependencies and be interleaved with legitimate operations, making complete attack-chain reconstruction difficult. In this paper, we present AgentTracer, an intent-aware tracing framework that treats IPI as task intent drift. AgentTracer recovers implicit decision dependencies among tool calls to construct an Intent-Dr
    
[^17]: CYBERFORT：一个为中小企业落实《网络韧性法案》的合规链平台

    CYBERFORT: A Compliance-Chain Platform Operationalising the Cyber Resilience Act for SMEs

    [https://arxiv.org/abs/2610.09918](https://arxiv.org/abs/2610.09918)

    CYBERFORT是一个面向中小企业的开源合规平台，其核心创新是“合规链”——一种可追溯结构，将欧盟《网络韧性法案》的全生命周期合规义务转化为从范围自评估、问题库到机器证明证据的完整可操作流程。

    

    欧盟《网络韧性法案》（CRA）将对产品网络安全的要求转变为欧盟市场上具有数字元素产品的制造商、进口商、分销商和集成商的全生命周期合规义务，这一负担主要落在那些很少拥有专门的治理、风险与合规（GRC）能力的中小企业（SME）身上。我们提出了CYBERFORT，这是一个在欧盟“数字欧洲计划”下开发的开源、以CRA为先的合规平台，也是欧盟CRA集群十二个项目之一。CYBERFORT通过引导式的范围自评估、与附件I及漏洞处理义务相关联的问题库，以及一个将每个答案与控制措施、政策和机器证明证据相链接的合规检查引擎来落实CRA，仅在ISO/IEC 27001、NIS2和GDPR控制措施与CRA义务重合之处加以复用。其核心贡献是“合规链”——一种可追溯的结构（摘要在此处截断）

    arXiv:2610.09918v1 Announce Type: cross  Abstract: The EU Cyber Resilience Act (CRA) turns product cybersecurity into a lifecycle compliance obligation for manufacturers, importers, distributors, and integrators of products with digital elements on the EU market, a load that falls largely on small and medium-sized enterprises (SMEs) that rarely have dedicated governance, risk, and compliance (GRC) capacity. We present CYBERFORT, an open-source CRA-first compliance platform developed under the EU Digital Europe Programme and one of twelve projects in the EU CRA cluster. CYBERFORT operationalises the CRA through a guided scope self-assessment, a question bank tied to Annex I and the vulnerability-handling obligations, and a compliance-checking engine that links every answer to controls, policies, and machine-attested evidence, reusing ISO/IEC 27001, NIS2, and GDPR controls only where they coincide with CRA obligations. Its central contribution is the compliance chain, a traceable structu
    
[^18]: 基于城市隐喻的Linux内核软件演化分析与可视化

    Analysis and Visualization of the Linux Kernel's Software Evolution Using the City Metaphor

    [https://arxiv.org/abs/2610.09910](https://arxiv.org/abs/2610.09910)

    本文提出使用ExplorViz工具和3D城市隐喻，对拥有超过4000万行代码的Linux内核进行软件结构与演化的可视化分析和交互式探索。

    

    Linux内核是现存规模最大且维护时间最长的开源项目之一。由于其代码量超过4000万行，理解内核的内部结构并评估其软件演化是一项巨大的挑战。在本文中，我们提出了一种使用我们的软件可视化工具ExplorViz来可视化Linux内核的方法。我们使用自定义的分析服务来分析来自Linux Git仓库的提交记录。基于Web的前端利用3D城市隐喻来可视化软件结构。系统为每个文件收集诸如代码行数等度量指标，并为目录和提交进行累积计算。计算得到的度量指标可以映射到建筑物的尺寸上，或通过热图进行展示。丰富的可视化、搜索和过滤选项支持对可视化数据的交互式探索。我们展示了内核仓库中所有文件的可视化结果，并进行了视觉分析（摘要在此处截断）。

    arXiv:2610.09910v1 Announce Type: new  Abstract: The Linux kernel is one of the largest and longest-maintained open source projects in existence. With more than 40 million lines of code, understanding the kernel's internal structure and assessing its software evolution is a great challenge. In this paper, we present an approach to visualize the Linux kernel using our software visualization tool ExplorViz. We analyze commits from the Linux Git repository using a custom analysis service. The web-based frontend utilizes the 3D city metaphor for visualization of the software structure. Metrics such as the number of lines are collected for each file and are accumulated for directories and commits. Calculated metrics can be mapped to the building's dimensions or be displayed via a heat map. A wide range of visualization, search, and filter options enable the interactive exploration of the visualized data. We present both a visualization of all files in the kernel repository and a visual anal
    
[^19]: 面向3D软件可视化中软件探索的聊天助手

    A Chat Assistant for Software Exploration in a 3D Software Visualization

    [https://arxiv.org/abs/2610.09901](https://arxiv.org/abs/2610.09901)

    该论文的核心创新是将基于大语言模型的聊天助手集成到3D软件可视化工具ExplorViz中，使用户能够通过自然语言提问并触发操作来探索甚至重构可视化的软件系统。

    

    我们提出了一种用于交互式软件探索的聊天助手，该助手嵌入在3D软件可视化工具ExplorViz中。该助手基于当前的大语言模型（LLMs）构建，使用户能够就当前可视化的软件系统进行提问，并通过自然语言触发改变可视化的操作。我们使用CopilotKit库将该聊天助手集成到ExplorViz前端中，使概率性的大语言模型与确定性的、基于工具的操作相结合，类似于使用模型上下文协议（MCP）的实现方式。该聊天助手还能够通过在可视化中添加、删除或修改软件系统的部分内容来对软件系统进行重构。一项由十一名参与者参与的实证实验评估了感知理解支持程度以及聊天助手触发的工具调用情况。参与者对助手生成的摘（原文摘要在此处截断）

    arXiv:2610.09901v1 Announce Type: new  Abstract: We present a chat assistant for interactive software exploration, embedded in the 3D software visualization tool ExplorViz. The assistant builds upon current Large Language Models (LLMs) and enables users to ask questions about the currently visualized software system and trigger actions that change the visualization through natural language. We integrate the chat assistant in our ExplorViz frontend using the CopilotKit libraries such that probabilistic LLMs are combined with deterministic and tool-based actions similar to implementations using the Model Context Protocol (MCP). The chat assistant is also enabled to restructure the software system by adding, removing, or modifying parts of the software system in the visualization. An empirical experiment with eleven participants evaluated both perceived comprehension support and the tool calls that were triggered by the chat assistant. Participants rated the assistant's generated summarie
    
[^20]: 面向边缘物联网的分层安全监控：一种形式化方法

    Hierarchical Security Monitoring for Edge-IoT: A Formal Methods Approach

    [https://arxiv.org/abs/2610.09817](https://arxiv.org/abs/2610.09817)

    提出一个基于形式化运行时验证的轻量级分层安全监控框架，边缘设备运行 TeSSLa 流规范以亚微秒级每事件成本输出四值判定，网关运行参数化一阶 MonPoly 监控器分析跨设备判定流，从而在有限防御资源下以可量化的成本在集中式云监控与纯边缘本地监控之间取得平衡，有效应对协调性多设备攻击。

    

    arXiv:2610.09817v1 公告类型：交叉。摘要：边缘物联网部署的网络弹性从根本上是一个经济问题：检测必须在遭受攻击时保持关键进程持续运行，而防御者的资源（计算能力、带宽、运维人员的注意力）是有限的。集中式云监控能够提供表达力强的跨设备检测，但其带宽成本高得令人望而却步；纯边缘本地监控虽然成本低廉，却对协调性多设备攻击视而不见，而在这种攻击中攻防不对称的平衡恰恰有利于攻击者。我们提出一个轻量级的分层安全监控框架，该框架建立在形式化运行时验证方法之上，以可量化的成本占据了实用的中间地带。每个边缘设备运行一个轻量级的 TeSSLa 流规范（包含报文大小、载荷有效性、速率和时间戳漂移等谓词），在每个聚合窗口输出一个四值判定，每事件成本低于一微秒；网关在各个设备的判定流之上运行参数化的一阶 MonPoly 监控器，成本为每……

    arXiv:2610.09817v1 Announce Type: cross  Abstract: Cyber resiliency in edge-IoT deployments is fundamentally an economic problem: detection must keep critical processes operating under attack, but defender resources (compute, bandwidth, operator attention) are bounded. Centralised cloud monitoring offers expressive cross-device detection at prohibitive bandwidth cost; purely edge-local monitoring is cheap but blind to coordinated multi-device attacks where the asymmetric balance favours the attacker. We propose a lightweight hierarchical security-monitoring framework, built on formal runtime-verification methods, that occupies the practical middle ground at quantified cost. Each edge device runs a lightweight TeSSLa stream specification (size, payload validity, rate, and timestamp-drift predicates) that emits a four-valued verdict per aggregation window at sub-microsecond per-event cost; the gateway runs a parametric first-order MonPoly monitor over the per-device verdict streams at mi
    
[^21]: 基于代理编排的程序验证成本高效定理证明

    Cost-Efficient Theorem Proving via Agent Orchestration in Program Verification

    [https://arxiv.org/abs/2610.09681](https://arxiv.org/abs/2610.09681)

    提出 CoCo-Prover，通过在两级证明图（声明内部的 AND/OR 证明超图与跨声明的引理依赖图）上进行元级成本决策与代理编排，将程序验证中的定理证明形式化为成本约束下的优化问题，以实现经济高效的批量证明。

    

    程序验证通过在定理证明器中构造的机器可检验证明来确立软件的正确性。这一保证对于大型语言模型（LLM）生成的代码尤其有价值，因为这类代码虽然流畅，却没有任何正确性保证。然而，几乎所有现有证明器都只追求通过率，而不计采样或搜索预算的代价，忽视了成功与成本之间的权衡前沿；但实际软件往往包含数百个相互依赖的证明义务，因此在规模化场景下，关键不在于能否证明某一个定理，而在于能够以多经济的代价证明多少个定理。我们提出 CoCo-Prover，它将成本高效的程序证明形式化为成本约束下的元级决策，并建立在两级证明图之上：将每个声明内部的 AND/OR 证明超图与跨声明的引理依赖图相连接；在每一步中，它回答两个问题：应选择哪些待证目标，以及哪些（原文在此截断）……

    arXiv:2610.09681v1 Announce Type: cross  Abstract: Program verification establishes software correctness through machine-checkable proofs constructed in theorem provers. It's a guarantee especially valuable for code generated by large language models (LLMs), which is fluent but carries no assurance of correctness. Almost all existing provers, however, pursue pass rates alone at whatever sampling or search budget it takes, and overlook the success-vs-cost frontier; yet real software often carries hundreds of interdependent proof obligations, so what matters at scale is not whether one theorem can be proved, but how many can be proved economically. We introduce CoCo-Prover, which formalizes cost-efficient program proving as metalevel decision-making under cost, grounded on two-level proof graphs: an AND/OR proof hypergraph within each declaration is joined to a lemma-dependency graph across declarations; and at each step, it answers two questions: which open goals to select, and which ac
    
[^22]: 编码智能体基准测试应匹配其用户的任务流

    Coding-Agent Benchmarks Should Match Their Users' Task Flows

    [https://arxiv.org/abs/2610.09633](https://arxiv.org/abs/2610.09633)

    该研究通过收集JetBrains IDE中真实软件工程师的4,782个智能体会话，发现真实任务流在任务类型与切换模式上高度多样且因数据源而异，因此编码智能体基准测试应先指明目标用例，再依据其真实测得的任务流进行校准。

    

    编码智能体的评估通常力求尽可能贴近真实。在本研究中，我们收集了JetBrains IDE中真实软件工程师的4,782个智能体会话，我们称之为“生产会话”。由于我们的研究对象是交互式智能体，我们研究了包含至少三条用户消息的会话（占样本的33%）。这些长会话与源自issue的基准测试任务在两个方面有所不同：(i) 用户请求所涵盖的任务类型范围要广泛得多——包括对项目代码的提问、规划、审查、重构、执行等；(ii) 用户会在整个会话过程中于不同任务类型之间切换。来自三个公开交互语料库的长会话样本展现出显著不同的任务流——即会话长度、任务类型以及类型间转换的分布——因此没有任何单一的交互分布是普遍真实的：基准测试应当指明目标用例，并根据从该用例中测得的数据进行校准。我们提出了SWE-TaskFlow，一种……（原文摘要在此处截断）

    arXiv:2610.09633v1 Announce Type: cross  Abstract: The evaluation of coding agents generally strives to be as realistic as possible. In our study, we collect 4,782 agent sessions of real software engineers in JetBrains IDEs, which we call Production Sessions. Since our subject is interactive agents, we study the sessions with at least three user messages (33% of the sample). These long sessions differ from issue-derived benchmark tasks in two ways: (i) user requests span a far wider mix of task types - questions about the project's code, planning, review, refactoring, execution - and (ii) users switch between types throughout a session. Long-session samples from three public interaction corpora exhibit markedly different Task Flows (the distributions of session lengths, task types, and type-to-type transitions), so no single interaction distribution is universally realistic: benchmarks should name a target use case and calibrate to measurements from it. We present SWE-TaskFlow, an appr
    
[^23]: GRAML：基于图证据推理与多任务学习的大语言模型软件漏洞检测

    GRAML: Graph-Grounded Reasoning and Multi-Task Learning for LLM-Based Software Vulnerability Detection

    [https://arxiv.org/abs/2610.09605](https://arxiv.org/abs/2610.09605)

    GRAML通过静态分析提取图结构证据，引导GPT-5进行思维树漏洞推理并生成漏洞描述，结合四任务多任务学习，显著提升了基于大语言模型的软件漏洞检测的泛化能力。

    

    大语言模型（LLM）已被广泛应用于软件漏洞检测，但其性能往往受限于对控制流和数据流信息利用不足的问题。本文提出了GRAML，一个融合图证据、漏洞描述生成与多任务训练的框架。GRAML首先对C/C++程序进行静态分析，提取关键源代码行及其带类型的行间关系作为结构化证据；随后利用这些证据，通过思维树引导的漏洞推理过程引导GPT-5生成漏洞描述；这些描述进一步与检测、定位和评估样本相结合，构建出统一的四任务训练数据集。我们在一个分布内（ID）测试集和六个分布外（OOD）数据集上对GRAML进行了评估，结果表明GRAML取得了66.67%至68.（摘要截断）...

    arXiv:2610.09605v1 Announce Type: new  Abstract: Large Language Models (LLMs) have been widely applied to software vulnerability detection. However, their performance is often limited by insufficient use of control-flow and data-flow information. In this paper, we propose GRAML, a framework that combines graph evidence, vulnerability description generation, and multi-task training. GRAML first performs static analysis on C/C++ programs to extract critical source lines and typed line relations as structural evidence. It then uses this evidence to guide GPT-5 through the Tree-of-Thought-guided Vulnerability Reasoning (ToT-VR) process and generate vulnerability descriptions. These descriptions are further combined with Detection, Localization, and Assessment samples to build a unified four-task training dataset. We evaluate GRAML on an in-distribution (ID) test set and six out-of-distribution (OOD) datasets. The results show that GRAML achieves average F1 scores ranging from 66.67% to 68.
    
[^24]: 超越FAIR：面向可持续研究软件的适应度函数框架

    Beyond FAIR: A Fitness Function Framework for Sustainable Research Software

    [https://arxiv.org/abs/2610.09580](https://arxiv.org/abs/2610.09580)

    本文将可持续研究软件的适应度函数评估框架从FAIR原则扩展到环境和安全两个新维度，分别涵盖资源效率、执行足迹以及依赖健康、漏洞暴露与安全配置，从而实现更全面的软件可持续性持续评估。

    

    研究软件的可持续性通常通过FAIR原则来评估，即可发现性、可访问性、互操作性和可重用性。尽管FAIR十分重要，但它并未涵盖所有与可持续性相关的问题。研究软件还应当对环境负责，并能够长期保持安全。过度的资源消耗会增加环境成本，而不安全的软件则会带来维护负担、技术债务以及重用障碍。为此，本文在FAIR原则之外，对先前提出的面向可持续研究软件的适应度函数框架进行了扩展。我们引入了两组新的适应度函数：针对资源效率和执行足迹的环境函数，以及针对依赖健康、漏洞暴露和安全配置的安全函数。这些函数共同将软件的持续评估范围从FAIR合规性拓展到更全面的可持续性视角。

    arXiv:2610.09580v1 Announce Type: new  Abstract: Research software sustainability is often assessed through the FAIR principles: findability, accessibility, interoperability and reusability. While important, FAIR does not cover all relevant sustainability concerns. Research software should also be environmentally responsible and secure over time. Excessive resource consumption increases environmental cost, while insecure software creates maintenance overhead, technical debt and barriers to reuse. To this end, in this paper, we extend a previously proposed fitness function framework for sustainable research software beyond FAIR. We introduce two additional sets of fitness functions: environmental functions targeting resource efficiency and execution footprint, and security functions targeting dependency health, vulnerability exposure and secure configuration. Together, these functions broaden continuous software assessment from FAIR compliance to a more complete view of sustainability. 
    
[^25]: 谁弄坏了我？基于执行引导的行为性依赖破坏修复方法

    Who Broke Me? Execution-Guided Repair of Behavioral Dependency Breaks

    [https://arxiv.org/abs/2610.09267](https://arxiv.org/abs/2610.09267)

    本文提出BBCFixer，通过在新旧库版本下运行失败测试并比较返回值差异来定位导致行为性破坏的根本API，从而利用库差异中的相关证据引导LLM自动修复依赖升级引发的行为性破坏。

    

    依赖升级可能会在不改变库接口的情况下破坏下游项目。这类行为性破坏变更对开发者来说很难修复，因为失败的测试并不总是指向根本API（即导致破坏的上游API）。现有的基于LLM的修复方法从编译器反馈或库文档中获取修复证据。然而，行为性破坏不会产生编译器反馈，且通常没有文档记录。未收到升级证据的智能体通常也不会自己去检索上游证据，它们大部分失败的修复都无法识别出根本API。库差异提供了有用的上游证据，但必须先识别出根本API才能选择差异中的相关部分。我们提出了BBCFixer，一种修复方法，它在旧版和新版库下分别运行失败的测试，对返回值不同的调用进行排序以识别候选根本API，并过滤……

    arXiv:2610.09267v1 Announce Type: new  Abstract: Dependency upgrades can break downstream projects without changing the library interface. Such behavioral breaking changes are difficult for developers to fix, because the failing test does not always point to the root API, the upstream API that causes the break. Existing LLM-based repair methods obtain evidence for the repair from compiler feedback or library documentation. However, a behavioral break produces no compiler feedback and is often undocumented. Agents that receive no upgrade evidence also usually do not retrieve upstream evidence themselves, and most of their failed repairs do not identify the root API.   The library diff provides useful upstream evidence, but the root API must first be identified to select the relevant part of the diff. We present BBCFixer, a repair method that runs the failing test under the old and new library versions, ranks the calls whose return value differs to identify a candidate root API, and filt
    
[^26]: SpecGuard：在智能体作弊之前证明任务已损坏

    SpecGuard: Proving a Task Is Broken Before the Agent Cheats

    [https://arxiv.org/abs/2610.09159](https://arxiv.org/abs/2610.09159)

    提出 SpecGuard，将任务意图与测试分别自动形式化为独立的 Lean 4 规范，并利用 Lean 内核形式化验证二者是否存在冲突，从而在编码智能体作弊之前就能证明任务本身已损坏。

    

    随着自主编码智能体日益广泛地部署，任务中意外出现的或被对抗性注入的错误规范可能导致智能体产生危险行为，这一风险亟待解决。先前的研究表明，面对此类存在问题的任务，智能体很少主动标记冲突，而是选择作弊——例如修改测试或硬编码预期输出，而这类作弊行为可能造成真实的损害，例如删除安全防御以使损坏的测试通过。目前尚不清楚能否在智能体采取行动之前，用可独立验证的证据确立此类冲突。我们提出了 SpecGuard，用于检测并形式化证明任务意图与测试之间的此类冲突。仅需给定任务描述和代码库，SpecGuard 即可将预期行为自动形式化为 Lean 4 规范；测试则被独立地形式化，由 Lean 内核检查是否存在任何实现能够同时满足这两种形式化，从而生成机器可检验的证明……（原文摘要在此处截断）

    arXiv:2610.09159v1 Announce Type: cross  Abstract: As autonomous coding agents get increasingly deployed, the risk that accidental or adversarially injected misspecifications in tasks lead to dangerous agent behavior is critical to address. Prior work has shown that agents given such tasks rarely flag the conflict and instead cheat, editing tests or hard-coding expected outputs, and the actions taken to cheat can cause real damage, such as deleting a security defense to make a corrupted test pass. It remains unclear whether such conflicts can be established with independently verifiable evidence before the agent acts. We present SpecGuard, which detects and formally certifies these conflicts between task intent and tests. Given only the task description and codebase, SpecGuard autoformalizes the intended behaviour into a Lean 4 specification. The tests are formalized independently, and the Lean kernel checks whether any implementation could satisfy both formalizations, producing a mach
    
[^27]: 发现AppWorld与WorkArena任务验证器中的盲点

    Finding Blind Spots in AppWorld and WorkArena Task Verifiers

    [https://arxiv.org/abs/2610.09142](https://arxiv.org/abs/2610.09142)

    该论文通过基于源码信息的变异测试审计了AppWorld和WorkArena的已发布任务验证器，揭示其存在盲点——即使智能体产生了错误效果（如重复写入创建多余记录或遗留非默认持久化值），验证器仍会判定任务成功。

    

    基于执行结果的任务验证器用于判定智能体是否成功完成任务。我们通过基于源代码信息的变异测试，对已发布的AppWorld和WorkArena验证器进行了审计。主要审计过程不会修改任何已发布的检查器。在AppWorld中，复制一个非幂等的写操作会创建一条额外的记录，同时保持所有被检查字段的值不变。验证器接受了来自五个合格生成器中两个的全部三个任务变体：6/15个构造效果。在普查之后对检查器副本应用的基数补丁使所有六个测试单元格失败，同时保留了有效的对照组。在WorkArena中，我们对先前因检查器通过（PASS）而被选中的23个额外字段候选案例进行了前瞻性重跑。通过独立的表API回读，确认其中21个存在非默认的持久化值，而所有23个均获得了通过。其中两个请求的字符串实际上是存储默认值的别名。这21个被确认的错误效果横跨三个表单模板。这些选定案例在审计协议下确认了错误效果；它们

    arXiv:2610.09142v1 Announce Type: cross  Abstract: Execution-based task verifiers decide whether an agent succeeded. We audit shipped AppWorld and WorkArena verifiers with source-informed mutation tests. The main audit never modifies a shipped checker.   In AppWorld, duplicating a non-idempotent write creates an extra record while preserving every checked field value. The verifier accepts all three task variants from two of five eligible generators: 6/15 constructed effects. A cardinality patch applied to checker copies after the census makes all six cells fail while preserving valid controls.   In WorkArena, we prospectively rerun 23 extra-field candidates selected for earlier checker-PASS outcomes. Independent Table API readback confirms nondefault persisted values in 21, while all 23 receive PASS. Two requested strings are aliases of stored defaults. The 21 confirmed wrong effects span three form templates. These selected cases confirm wrong effects under the audit's protocol; they 
    
[^28]: 基于Agent原生可复用代码原语的大规模仓库工程

    Large-scale Repository Engineering via Agent-Native Reusable Code Primitives

    [https://arxiv.org/abs/2610.09079](https://arxiv.org/abs/2610.09079)

    提出了具有接口契约、依赖闭包、验证测试和来源溯源的Agent原生可复用代码原语Code Primitives，以及LEGO框架，通过激活并适配1,424个已验证原语（收录于CodeFace库）来实现大规模仓库级代码构建。

    

    配备开发环境的大语言模型已将代码生成推向仓库级别的构建，然而构建完整仓库仍然困难，因为相互作用的模块、接口、配置、测试和依赖必须协同工作。我们引入了Code Primitives（代码原语），这是一种Agent原生的可复用可执行组件，具备接口契约、依赖闭包、验证测试和来源溯源信息。每个原语使用一个常驻LLM来评估相关性，并将其实现、接口和依赖适配到目标仓库。我们在CodeFace中组织了1,424个经过验证的原语，这是一个面向仓库构建的可搜索库。我们提出了LEGO（基于Agent原生可复用代码原语的大规模仓库工程），它激活与任务相关的原语，在解决跨组件约束的同时将适配后的实现与任务特定代码集成，并修订……（原文摘要在此处截断）

    arXiv:2610.09079v1 Announce Type: cross  Abstract: Large language models equipped with development environments have moved code generation toward repository-scale construction, yet building complete repositories remains difficult because interacting modules, interfaces, configurations, tests, and dependencies must work together. We introduce Code Primitives, agent-native reusable executable components with interface contracts, dependency closures, validation tests, and provenance. Each primitive uses a resident LLM to assess relevance and adapt its implementation, interfaces, and dependencies to the target repository, and we organize 1,424 validated primitives in CodeFace, a searchable library for repository construction. We introduce LEGO (Large-scale repository Engineering via aGent-native reusable cOde primitives), which activates task-relevant primitives, integrates their adapted implementations with task-specific code while resolving cross-component constraints, and revises the re
    
[^29]: 评估用于软件性能回归分析的变点检测方法

    Evaluating Change Point Detection Methods for Software Performance Regression Analysis

    [https://arxiv.org/abs/2610.09023](https://arxiv.org/abs/2610.09023)

    本文对多种变点检测方法在真实世界软件性能测量数据上的有效性进行了综合评估，以帮助在开发周期中尽早检测软件性能回归。

    

    软件系统中的性能问题是一个关键的质量问题，它可能削弱用户信任、违反服务水平协议，并最终影响业务效率。因此，软件性能工程已将重点转向开发稳健的技术，以便在开发周期中尽早检测性能回归。性能回归分析通常依赖于性能测量的时间序列来检测性能行为中的显著变化。变点检测方法已被广泛用于自动化识别金融、医疗保健和性能监控等多个领域中的此类变化。然而，这些方法对软件性能测量的有效性尚未得到彻底评估。在本文中，我们提出了一项综合研究，以评估各种变点检测方法在真实世界软件性能测量数据上的有效性。

    arXiv:2610.09023v1 Announce Type: new  Abstract: Performance issues in software systems are a critical quality issue that can erode user trust, violate service-level agreements, and ultimately affect business efficiency. Consequently, software performance engineering has shifted its focus to developing robust techniques to detect performance regressions as early as possible in the development cycle. Performance regression analysis often relies on time series of performance measurements to detect significant changes in performance behavior. Change Point Detection (CPD) methods have been widely used to automate the identification of such changes in various domains, including finance, healthcare, and performance monitoring. However, the effectiveness of these methods for software performance measurements has not been thoroughly evaluated. In this paper, we present a comprehensive study to evaluate the effectiveness of various CPD methods on real-world software performance measurement data
    
[^30]: 设备端语言模型的安全性有多脆弱？定位安全关键参数以进行稀疏故障分析

    How Fragile Is On-Device Language Model Safety? Localizing Safety-Critical Parameters for Sparse Fault Analysis

    [https://arxiv.org/abs/2610.09000](https://arxiv.org/abs/2610.09000)

    研究发现LLaMA-2-7B-Chat的安全敏感行为高度集中在MLP的down_proj等稀疏参数子集中，仅修改0.19%的权重即可使攻击成功率大幅上升，揭示了设备端部署的语言模型存在显著的安全脆弱点。

    

    随着小型语言模型（SLM）越来越多地部署在资源受限的设备端平台上，包括作为智能体系统的组件，本地存储的模型参数的完整性成为一个重要的安全问题。我们研究了LLaMA-2-7B-Chat中的安全敏感行为是否集中在参数的稀疏子集中，从而为针对性分析创建了一个缩小的故障面。我们研究了两种互补的定位方法：低秩安全相关子空间分析和参数级安全-效用重要性过滤。两种方法都揭示了网络中高度不均匀的安全敏感性，其中MLP的down_proj始终是突出的安全敏感组件，而o_proj的贡献较小。利用参数级定位，仅修改down_proj中0.19%的模型权重就能产生53%的基本攻击成功率（Basic ASR）和56%的GCG攻击成功率，而tinyBenchmarks准确率仍保持在51。

    arXiv:2610.09000v1 Announce Type: cross  Abstract: As small language models (SLMs) are increasingly deployed on resource-constrained and on-device platforms, including as components of agentic systems, the integrity of locally stored model parameters becomes an important safety concern. We investigate whether safety-sensitive behavior in LLaMA-2-7B-Chat is concentrated within a sparse subset of parameters, creating a reduced fault surface for targeted analysis. We study two complementary localization methods: low-rank safety-associated subspace analysis and parameter-level safety--utility importance filtering. Both approaches reveal highly non-uniform safety sensitivity across the network, with the MLP down_proj consistently emerging as a prominent safety-sensitive component and o_proj providing a smaller contribution. Using parameter-level localization, modifying only 0.19% of model weights in down_proj yields 53% Basic ASR and 56% GCG ASR, while tinyBenchmarks accuracy remains at 51.
    
[^31]: 使用GoDDaR自动检测与修复Go代码中的死锁

    Automatically Detecting and Fixing Deadlocks in Go Code with GoDDaR

    [https://arxiv.org/abs/2610.08962](https://arxiv.org/abs/2610.08962)

    GoDDaR是一个能够自动检测并修复Go程序中全局死锁和部分死锁的工具，弥补了Go运行时检测器无法发现部分死锁、且现有静态检测工具缺乏修复支持的不足。

    

    Go编程语言通过goroutine为并发编程提供了一种轻量级抽象，但goroutine容易发生死锁。Go包含一个运行时检测器，当所有线程都被阻塞（即全局死锁）时会中止执行。然而，由于线程调度的非确定性，这种运行时机制只能检测到执行期间显现的全局死锁，无法识别部分死锁，即一部分goroutine被永久阻塞、而至少还有一个goroutine保持可运行状态的情况。静态检测局部死锁对于开发可靠的并发软件至关重要。虽然已有多种工具能够静态检测并发程序中的死锁，但很少有工具能帮助开发者修复死锁。检测和解决部分死锁需要对复杂的交错执行和通信模式进行推理，这本身就是一项极具挑战性的任务。在本文中，我们提出了GoDDaR，这是一个能够检测未被观测到的全局或部分死锁的工具。

    arXiv:2610.08962v1 Announce Type: cross  Abstract: The Go programming language provides a lightweight abstraction for concurrent programming through goroutines, which are prone to deadlocks. Go includes a runtime detector that aborts execution when all threads are blocked (a global deadlock). However, due to nondeterministic thread scheduling, this runtime mechanism only detects global deadlocks that manifest during execution and cannot identify partial deadlocks, where a subset of goroutines is permanently blocked while at least one remains runnable. Statically detecting local deadlocks is essential for developing dependable concurrent software.   While several tools statically detect deadlocks in concurrent programs, few assist developers in fixing them. Detecting and resolving partial deadlocks requires reasoning about complex interleavings and communication patterns, an inherently challenging task.   In this paper, we present GoDDaR, a tool that detects unobserved global or partial
    
[^32]: 软件工程团队中的心理安全感：关于团队流程与绩效的系统性映射研究

    Psychological Safety in Software Engineering Teams: A Systematic Mapping Study of Team Processes and Performance

    [https://arxiv.org/abs/2610.08896](https://arxiv.org/abs/2610.08896)

    本研究对2006年至2026年间发表的112项原始研究进行系统性映射，全面梳理了软件工程团队中心理安全感的多维概念、与团队绩效和流程的关联、情境影响及障碍与强化策略，填补了该领域证据零散的空白。

    

    背景：心理安全感是软件开发团队中一项重要的人文与行为因素，能够支持协作、知识共享、人际风险承担和学习。然而，软件工程领域的相关证据在团队流程、绩效结果、情境条件和障碍等方面仍较为零散。目标：本研究对软件工程中心理安全感的相关文献进行映射梳理，考察其概念界定、与团队绩效及流程的关联、情境影响、障碍因素以及强化策略。方法：我们遵循既定的软件工程研究指南开展了系统性映射研究，围绕六个研究问题，采用描述性统计和主题分析方法，分析了2006年至2026年6月间发表的112项原始研究。结果：心理安全感呈现出一种多维、依赖情境的团队层面构念特征，其核心在于人际风险承担……（原文摘要在此处截断）

    arXiv:2610.08896v1 Announce Type: new  Abstract: Context: Psychological safety is an important human and behavioral factor in software development teams, supporting collaboration, knowledge sharing, interpersonal risk-taking, and learning. However, evidence in software engineering remains fragmented across team processes, performance outcomes, contextual conditions, barriers. Objective: This study maps literature on psychological safety in software engineering, examining its conceptualization, associations with team performance and processes, contextual influences, barriers, and strengthening strategies. Method: We conducted a systematic mapping study following established software engineering guidelines. We analyzed 112 primary studies published between 2006 and June 2026 using descriptive statistics and thematic analysis across six research questions. Results: Psychological safety emerges as a multidimensional, context-dependent, team-level construct characterized by interpersonal ri
    
[^33]: 缓解基于生成式AI的自适应系统中的不确定性交互：愿景、挑战与初步指南

    Mitigating Uncertainty Interactions in GenAI-based Adaptive Systems: Vision, Challenges and Preliminary Guidelines

    [https://arxiv.org/abs/2610.08881](https://arxiv.org/abs/2610.08881)

    本文针对生成式AI组件在自适应系统中引入的复杂且相互叠加的不确定性交互问题，提出了一个初步概念框架，并给出贯穿全软件生命周期的缓解指南。

    

    现代软件密集型系统日益融入生成式AI组件，包括大语言模型和智能体子系统，这在系统各层引入了新颖且相互叠加的不确定性来源。自适应系统研究界在理解和管理不确定性方面已取得重大进展。然而，生成式AI的内在特性，包括概率性输出、幻觉、上下文窗口限制和记忆陈旧等问题，要求我们重新审视现有的不确定性处理框架与缓解策略，尤其是不确定性发生与表现形式之间的交互作用。本立场论文提出一个初步的概念框架，为表征和缓解基于生成式AI的软件密集型系统中的不确定性交互建立了初步指南。我们认为，缓解措施必须贯穿整个软件生命周期加以考虑，涵盖需求分析、设计等方面。

    arXiv:2610.08881v1 Announce Type: new  Abstract: Modern software-intensive systems increasingly incorporate GenAI components, including LLM and agentic subsystems, which introduce novel and compounding sources of uncertainty across system layers. The self-adaptive systems community has made significant strides in understanding and managing uncertainty. The intrinsic characteristics of GenAI, including probabilistic outputs, hallucinations, context window limitations, and memory staleness, demand a re-examination of existing frameworks and mitigation strategies for dealing with uncertainty, and especially, the interactions among uncertainty occurrences and manifestations. This position paper posits an initial conceptual framework that establishes preliminary guidelines for characterizing and mitigating uncertainty interactions in GenAI-based software-intensive systems. We argue that mitigation must be considered across the full software lifecycle, encompassing requirements analysis, des
    
[^34]: RAPO-Sol：面向仓库级Solidity代码生成的检索增强偏好优化

    RAPO-Sol: Retrieval-Augmented Preference Optimization for Repository-Level Solidity Code Generation

    [https://arxiv.org/abs/2610.08429](https://arxiv.org/abs/2610.08429)

    该论文提出RAPO-Sol两阶段训练框架，将检索增强微调（RAFT）与基于语义锚点扰动（SAP）构建拒绝样本的直接偏好优化（DPO）相结合，以提升仓库级Solidity智能合约代码生成的正确性与语义一致性。

    

    用Solidity编写的智能合约管理着资产、权限和不可逆的状态变更，这使得代码生成既具有实用价值又关乎安全关键。仓库级的Solidity生成极具挑战性，因为模型必须在合成完整合约或库的同时，保持状态变量、修饰符、事件、继承、外部调用和访问控制逻辑之间的一致性。我们提出了RAPO-Sol，这是一个用于仓库级Solidity代码生成的两阶段训练框架。首先，检索增强微调（RAFT）通过相似的Solidity示例来增强每个训练输入，帮助模型学习重复出现的合约级模式，同时在推理时无需检索。其次，直接偏好优化（DPO）训练模型更倾向于参考合约，而非那些接近但存在语义缺陷的替代方案。我们使用Solidity语义锚点扰动（SAP）构建被拒绝的样本，通过对有效……（摘要原文在此处截断）

    arXiv:2610.08429v1 Announce Type: new  Abstract: Smart contracts written in Solidity manage assets, permissions, and irreversible state changes, making code generation both useful and security-critical. Repository-level Solidity generation is challenging because models must synthesize complete contracts or libraries while preserving consistency across state variables, modifiers, events, inheritance, external calls, and access-control logic. We present RAPO-Sol, a two-stage training framework for repository-level Solidity code generation. First, Retrieval-Augmented Fine-Tuning (RAFT) augments each training input with similar Solidity examples, helping the model learn recurring contract-level patterns while remaining retrieval-free at inference time. Second, Direct Preference Optimization (DPO) trains the model to prefer reference contracts over close but semantically flawed alternatives. We construct rejected samples using Solidity Semantic-Anchor Perturbation (SAP), which perturbs vali
    
[^35]: PreMaQ：在生成之前预测大语言模型生成代码的可维护性相关质量

    PreMaQ: Predicting Maintainability-Related Quality of LLM-Generated Code Before Generation

    [https://arxiv.org/abs/2610.05858](https://arxiv.org/abs/2610.05858)

    该论文提出PreMaQ方法，在LLM生成代码之前通过模型内部表示预测生成代码的可维护性相关质量指标（代码坏味道分数和可维护性指数），从而帮助开发者避免生成、审查和丢弃低质量代码的成本。

    

    随着大语言模型（LLM）的代码生成能力日益增强，在软件开发中采用生成的代码时，不仅需要评估其功能正确性，还需要评估其可维护性相关质量。如果这种质量能够在生成之前得到估计，开发人员就可以避免生成、审查和丢弃低质量代码的成本。尽管先前的工作已经表明LLM生成代码的功能正确性可以提前预测，但可维护性相关质量是否同样可以预测仍不清楚。我们提出了生成前可维护性相关质量预测（PreMaQ），该方法在大语言模型生成代码之前，从LLM的内部表示中预测生成代码的代码坏味道分数和可维护性指数。我们的评估涵盖了四个开源权重的大语言模型和四个Python代码生成基准，共包含2,695个任务。我们的结果表明

    arXiv:2610.05858v2 Announce Type: replace  Abstract: As large language models (LLMs) become increasingly capable of code generation, adopting generated code in software development requires assessing not only its functional correctness but also its maintainability-related quality. If such quality could be estimated before generation, developers could avoid the cost of generating, reviewing, and discarding low-quality code. Although prior work has shown that the functional correctness of the LLM-generated code can be predicted in advance, it remains unclear whether maintainability-related quality is similarly predictable. We introduce Pre-Generation Maintainability-Related Quality Prediction (PreMaQ), which predicts the Code Smell Score (CSS) and Maintainability Index (MI) of generated code from the internal representations of LLMs before generation. Our evaluation covers four open-weight LLMs and four Python code generation benchmarks, comprising 2,695 tasks in total. Our results show 
    
[^36]: 从验证失败到编码智能体的可复用指导

    From Verification Failures to Reusable Guidance for Coding Agents

    [https://arxiv.org/abs/2609.39022](https://arxiv.org/abs/2609.39022)

    该论文提出将专家对验证失败的诊断转化为编码智能体可复用的指导，结合K框架的可执行语言语义与一套用于构建规范、修复证明和审计充分性的工具包，在HumanEval上实现164/164的全通过率，并通过对照实验证明审计能识别出证明通过但存在缺陷的软件包。

    

    编码智能体需要确认程序满足规范，并且该规范确实刻画了所要求的行为。我们研究如何将专家对验证失败的诊断转化为这项工作中可复用的指导。我们的方法将K框架中的可执行语言定义与一套用于构建规范、修复证明以及审计其充分性的流程工具包相结合。在HumanEval（一个包含164个Python编程任务的基准测试）上进行的人工指导开发活动中，借助该语义定义和工具包，以两次针对性修复后最终AI审计的Pass判定为衡量标准，达到了164/164的成功率。为了检验审计能否发现成功证明所遗留的未决问题，我们构建了12对经作者审查的“干净”与“缺陷”软件包。每个软件包都通过了其K证明，而已完成的审计识别出了所有缺陷，并接受了所有干净的软件包。随后，我们使用KleverBench来测试规范……（摘要原文在此处截断）

    arXiv:2609.39022v1 Announce Type: cross  Abstract: Coding agents need to establish that a program satisfies a specification and that the specification captures the requested behavior. We study how expert diagnosis of verification failures can become reusable guidance for this work. Our approach combines executable language definitions in the K framework with a kit of procedures for constructing specifications, repairing proofs, and auditing their adequacy. A human-guided development campaign on HumanEval, a benchmark of 164 Python programming tasks, achieves a 164/164 success rate with the semantics and the kit, measured by final AI audit Pass verdicts after two targeted repairs. To examine whether auditing detects problems that successful proofs leave unresolved, we construct 12 author-reviewed pairs of clean and defective packages. Every package passes its K proofs, and completed audits identify all defects and accept all clean packages. We then use KleverBench to test specification 
    
[^37]: 评估安全关键自动驾驶车辆软件时运行数据保真度的影响

    The Impact of Operational-Data Fidelity when Assessing Safety-Critical Autonomous-Vehicle Software

    [https://arxiv.org/abs/2608.10025](https://arxiv.org/abs/2608.10025)

    本研究将保守贝叶斯推断技术扩展至自动驾驶车辆安全评估领域，发现低保真的运行数据即使被保守使用也可能得出危险的乐观结论，强调了运行数据保真度对软件可靠性声明的重要影响。

    

    对于安全关键软件而言，运行数据（例如软件成功与失败的序列）可以为可靠性声明提供强有力的统计支持。然而，关于过去软件故障的细节不足可能使评估无法解释故障行为的重要特征。在本文中，我们扩展了用于可靠性评估的保守贝叶斯推断（CBI）技术，以检验基于此类数据的可靠性声明的稳健性。我们展示了运行数据中细节的不足如何在自动驾驶车辆（AV）安全评估场景中削弱软件可靠性声明：即使保守地使用，低保真数据也可能得出危险的乐观结论。虽然这些发现与之前关于贝叶斯软件可靠性评估中统计模型保真度影响的工作一致，但我们的工作阐明了为什么尝试保守地使用低保真数据可能是不足的。

    arXiv:2608.10025v2 Announce Type: replace-cross  Abstract: For safety-critical software, operational data (e.g. sequences of software successes and failures) can provide strong statistical support for reliability claims. However, insufficient detail about past software failures may leave assessments unable to account for important features of failure behavior. In this paper, we extend conservative Bayesian inference (CBI) techniques for reliability assessment to check the robustness of reliability claims based on such data. We show how insufficient detail in operational data can undermine software reliability claims in autonomous vehicle (AV) safety assessment scenarios: even when used conservatively, low-fidelity data may yield dangerously optimistic conclusions. While these findings are consistent with previous work on the impact of statistical model fidelity in Bayesian software reliability assessments, our work clarifies why attempts to use low-fidelity data conservatively can be n
    
[^38]: SWE-NFI：研究并基准测试编码智能体的非功能性改进能力

    SWE-NFI: Studying and Benchmarking Coding Agents for Non-Functional Improvements

    [https://arxiv.org/abs/2607.27409](https://arxiv.org/abs/2607.27409)

    该论文提出了SWE-NFI基准，基于开源Python项目真实合并的拉取请求构建188个任务，并将五类面向开发者的非功能性改进操作化为92条可执行规则，用于评估编码智能体在保持代码行为不变前提下提升软件质量的能力。

    

    尽管编码智能体在以正确性为导向的基准测试中已取得令人瞩目的成绩，但它们在保持行为不变的前提下进行非功能性改进（NFI）的能力仍未得到充分探索。在真实世界的软件开发中，开发者会在不改变可观测行为的情况下持续改进软件质量，然而现有基准主要评估功能正确性，对这些非功能性改进的评估支持有限。在本文中，我们提出了SWE-NFI，一个用于在功能正确性之外评估编码智能体非功能性改进能力的基准。SWE-NFI包含188个任务，这些任务基于开源Python项目中真实合并的拉取请求构建。我们将五个面向开发者的NFI方面操作化为92条可执行规则，并开发了一个评估套件：该套件首先应用任务特定的功能保持检查，以评估原始代码的关键属性是否得到保留，随后执行基于规则的……（摘要在此处被截断）

    arXiv:2607.27409v2 Announce Type: replace  Abstract: Although coding agents have achieved impressive performance on correctness-oriented benchmarks, their ability to make behavior-preserving non-functional improvements (NFIs) remains underexplored. In real-world software development, developers continuously improve software quality without changing observable behavior, yet existing benchmarks primarily evaluate functional correctness and provide limited support for assessing these non-functional improvements. In this paper, we present SWE-NFI, a benchmark for evaluating coding agents on NFIs beyond functional correctness. SWE-NFI contains 188 tasks constructed from real merged pull requests in open-source Python projects. We operationalize five developer-oriented NFI aspects into 92 executable rules and develop an evaluation suite that first applies task-specific functional preservation checks to assess whether key properties of the original code are preserved and then performs rule-ba
    
[^39]: 面向大语言模型电力系统代码生成的知识边界探测与需求引导干预

    Knowledge boundary probing and demand-guided intervention for LLM-based power system code generation

    [https://arxiv.org/abs/2605.31478](https://arxiv.org/abs/2605.31478)

    该论文提出PowerCodeBench基准（面向pandapower的2000个冻结任务）以及无需更新权重的部署时工作流，通过文档驱动的L0-L3知识边界探测、查询侧需求估计选择分层API证据、以及执行反馈引导的针对性修复，显著提升了LLM电力系统代码生成的准确率。

    

    大语言模型（LLM）可以将电网分析请求转化为用于电力系统仿真的可执行程序，但电力公司和科研实验室通常要求本地化部署。在这种场景下，首次生成失败常常发生在API知识边界处，表现为幻觉函数、参数误用以及对结果表的错误处理。我们提出了PowerCodeBench，一个参数化的基准测试生成器，以冻结的2000个任务的pandapower任务套件形式发布；同时还提出了一个无需权重更新的部署时工作流。基于文档驱动的L0-L3探测可为每个模型生成API画像，用于诊断、模型比较、文档分配和后端校准。查询侧的需求估计器在生成前选择分层的API证据，而执行反馈则引导针对性修复。在十个开源权重LLM（1.5B-480B）和四个中端API上的实验表明，启用验证的工作流提升了标量匹配准确率（摘要原文在此截断）。

    arXiv:2605.31478v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) can turn grid-analysis requests into executable programs for power-system simulation, but utilities and research laboratories often require on-premise deployment. In this setting, first-pass failures frequently arise at an API-knowledge boundary, through hallucinated functions, misused parameters, and mishandled result tables. We present PowerCodeBench, a parameterised benchmark generator released as a frozen 2,000-task suite for pandapower, and a deployment-time workflow that requires no weight updates. Documentation-driven L0-L3 probes produce per-model API profiles for diagnosis, model comparison, documentation allocation, and backend calibration. A query-side demand estimator selects layered API evidence before generation, while execution feedback routes targeted repair. Across ten open-weight LLMs (1.5B-480B) and four mid-tier APIs, the validation-enabled workflow raises scalar-match accuracy b
    
[^40]: 洞察生成器：面向大语言模型智能体的系统性语料库级轨迹诊断

    Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents

    [https://arxiv.org/abs/2605.21347](https://arxiv.org/abs/2605.21347)

    该论文提出了洞察生成器（IG）——一个多智能体系统，通过在执行轨迹语料库上自动提出并检验假设，生成有证据支持的系统性诊断洞察报告，解决了 LLM 智能体失败诊断依赖人工、无法规模化的问题。

    

    arXiv:2605.21347v4 公告类型：replace-cross 摘要：诊断大语言模型（LLM）智能体的失败在很大程度上仍然是手工完成的。从业者通常只检查一小部分执行轨迹，形成临时性的假设，然后不断迭代。这一过程会遗漏那些只有在轨迹群体层面才会显现的模式，也无法扩展到单条轨迹就包含数万 token 的生产级语料库。我们形式化了语料库级轨迹诊断问题：给定一个执行轨迹语料库，目标是在轨迹群体上刻画系统性的行为模式，生成有据可依的自然语言洞察，并且每条洞察都关联相应的支持性证据。我们提出了洞察生成器（Insights Generator, IG），这是一个多智能体系统，它通过在轨迹语料库上提出并检验假设来回答诊断问题，最终产出有证据支撑的洞察报告。我们从定性和客观两个维度评估了 IG，包括基于评分量表的报告评估，以及通过实施洞察所带来的下游性能提升……

    arXiv:2605.21347v4 Announce Type: replace-cross  Abstract: Diagnosing failures in LLM agents remains largely manual. Practitioners inspect a small subset of execution traces, form ad-hoc hypotheses, and iterate. This process misses patterns that only emerge across trace populations and does not scale to production corpora where individual traces span tens of thousands of tokens. We formalize the problem of corpus-level trace diagnostics. Given a corpus of execution traces, the goal is to produce grounded natural-language insights that characterize systematic behavioral patterns across trace groups, each linked to supporting evidence. We present the Insights Generator (IG), a multi-agent system that answers diagnostic questions by proposing and testing hypotheses across the trace corpus to produce an evidence-backed insights report. We evaluate IG across qualitative and objective dimensions, spanning rubric-based report assessment and downstream performance improvements achieved by impl
    
[^41]: TorchGWAS 1.0：大规模GPU加速的全基因组关联分析

    TorchGWAS 1.0: GPU-accelerated GWAS at scale

    [https://arxiv.org/abs/2604.21095](https://arxiv.org/abs/2604.21095)

    TorchGWAS是一个GPU加速的批量线性关联检验框架，可对数千个定量表型进行高通量、协变量校正的全基因组关联分析，其结果与PLINK 2.0完全一致，并能在约一分钟内完成45.7亿次关联检验。

    

    影像学、分子层面和机器学习工作流可以在单个队列中生成数千个定量表型，当逐一检验这些性状时会产生巨大的计算和输出瓶颈。TorchGWAS是一个GPU加速的框架，它利用批量运算对大量定量表型进行高通量的、协变量校正的线性关联检验。在500,036次等位基因对齐的检验中，TorchGWAS的t统计量与PLINK 2.0一致。在配备NVIDIA H100 80-GB GPU、48核Intel Xeon Gold 6442Y主机（实测磁盘读取和写入速率分别为5.98和1.49 GB/s）的环境下，对45.7亿次关联检验（35,365个样本中的8,931,083个变异位点与512个表型）的中位数端到端耗时为：BED格式28.46秒、硬调用PGEN格式29.13秒、BGEN格式51.48秒、剂量PGEN格式58.95秒，其中包括写入36.7 GB的二进制汇总统计数据。TorchGWAS提供了一个高效的基于Python的框架……

    arXiv:2604.21095v2 Announce Type: replace-cross  Abstract: Imaging, molecular, and machine-learning workflows can generate thousands of quantitative phenotypes in a single cohort, creating substantial computational and output bottlenecks when testing traits individually. TorchGWAS is a GPU-accelerated framework that uses batched operations for high-throughput, covariate-adjusted linear association testing across large panels of quantitative phenotypes. Across 500,036 allele-harmonized tests, TorchGWAS t statistics agreed with PLINK 2.0. On an NVIDIA H100 80-GB GPU with a 48-core Intel Xeon Gold 6442Y host and measured disk read and write rates of 5.98 and 1.49 GB/s, respectively, median end-to-end times for 4.57 billion associations (8,931,083 variants by 512 phenotypes in 35,365 samples) were 28.46 s for BED, 29.13 s for hard-call PGEN, 51.48 s for BGEN, and 58.95 s for dosage PGEN, including writing 36.7 GB of binary summary statistics. TorchGWAS provides an efficient Python-based fr
    
[^42]: OOM-RL：资金耗尽强化学习——面向基于大语言模型的多智能体系统的市场驱动对齐

    OOM-RL: Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems

    [https://arxiv.org/abs/2604.11477](https://arxiv.org/abs/2604.11477)

    该论文提出“资金耗尽强化学习（OOM-RL）”这一客观对齐新范式，通过将基于LLM的多智能体系统部署到真实金融市场中，利用资金耗尽带来的真实经济损失作为外部负梯度信号，从而克服RLHF/RLAIF导致的模型谄媚和执行环境中的测试规避问题。

    

    面向自主软件工程的多智能体系统（MAS）的对齐受到评估者认知不确定性的制约。当前诸如基于人类反馈的强化学习（RLHF）和基于AI反馈的强化学习（RLAIF）等范式，经常诱发模型谄媚行为，而基于执行的环境中，不受约束的智能体还会进行对抗性的“测试规避”。在本文中，我们提出了一种客观对齐范式：资金耗尽强化学习（OOM-RL）。通过将智能体部署到真实金融市场这一非平稳、高摩擦的现实环境中，我们利用关键性的资金耗尽作为外部施加的负梯度。我们为期20个月的纵向实证研究记录了该系统从高换手率、谄媚的基线演变为稳健的、具备流动性感知能力的架构。我们表明，财务损失的经济后果——真实的执行成本、滑点和资金耗尽——暴露了失效（摘要在此处截断）

    arXiv:2604.11477v2 Announce Type: replace-cross  Abstract: The alignment of Multi-Agent Systems (MAS) for autonomous software engineering is constrained by evaluator epistemic uncertainty. Current paradigms, such as Reinforcement Learning from Human Feedback (RLHF) and AI Feedback (RLAIF), frequently induce model sycophancy, while execution-based environments suffer from adversarial "Test Evasion" by unconstrained agents. In this paper, we introduce an objective alignment paradigm: Out-of-Money Reinforcement Learning (OOM-RL). By deploying agents into the non-stationary, high-friction reality of live financial markets, we utilize critical capital depletion as an externally imposed negative gradient. Our longitudinal 20-month empirical study chronicles the system's evolution from a high-turnover, sycophantic baseline to a robust, liquidity-aware architecture. We show that the economic consequences of financial loss---real execution costs, slippage, and capital depletion---exposed failur
    
[^43]: huff：一个用于市场区分析的Python软件包

    huff: A Python package for Market Area Analysis

    [https://arxiv.org/abs/2602.17640](https://arxiv.org/abs/2602.17640)

    huff是一个模块化的Python软件包，为市场区与空间可达性分析提供了从数据导入、模型构建、参数估计到地图可视化的完整工作流程。

    

    市场区模型，如Huff模型及其扩展，被用于估计零售和服务地点的区域市场份额与客户流量。在健康地理学领域，市场区模型和可达性模型被应用于分析医疗保健机构的服务范围和空间可达性。huff Python软件包为市场区分析和空间可达性分析提供了完整的工作流程，包括数据导入、构建起点-目的地交互矩阵、基础模型分析、基于实证数据的参数估计、距离或出行时间矩阵的计算以及地图可视化。该软件包采用模块化和面向对象的设计，面向经济地理学、区域经济学、市场营销、地理信息科学和健康地理学的研究人员。该软件可通过Python包索引（PyPI）开放获取（https://pypi.org/project/huff/）。

    arXiv:2602.17640v5 Announce Type: replace-cross  Abstract: Market area models, such as the Huff Model and its extensions, are used to estimate regional market shares and customer flows of retail and service locations. In health geography, market area and accessibility models are applied for the analysis of catchment areas and spatial accessibility of healthcare locations. The huff Python package provides a complete workflow for market area and spatial accessibility analysis, including data import, construction of origin-destination interaction matrices, basic model analysis, parameter estimation from empirical data, calculation of distance or travel time matrices, and map visualization. The package is modular and object-oriented. It is intended for researchers in economic geography, regional economics, marketing, geoinformation science, and health geography. The software is openly available via the Python Package Index (PyPI) (https://pypi.org/project/huff/). Its development and versio
    
[^44]: Doc2Spec：通过文法归纳从自然语言合成形式化程序规约

    Doc2Spec: Synthesizing Formal Programming Specifications from Natural Language via Grammar Induction

    [https://arxiv.org/abs/2602.04892](https://arxiv.org/abs/2602.04892)

    Doc2Spec提出多智能体框架，通过从自然语言API规则自动归纳领域专用文法来约束大模型分步生成可检查的形式化规约，显著提升了规约合成的精度与召回率。

    

    arXiv:2602.04892v2 公告类型：replace-cross 摘要：确保API实现及其使用符合自然语言编程规则，对软件的正确性、安全性和可靠性至关重要。形式化验证能够提供强有力的保证，但需要精确的规约，而人工编写这些规约既困难又成本高昂。为应对这一挑战，我们提出了Doc2Spec，这是一个多智能体框架，能够从自然语言API规则中自动归纳出领域专用文法，并利用该文法指导规约生成。Doc2Spec将一个与领域无关的逻辑骨架固定为文法模板，提示大语言模型（LLM）推断领域特定的谓词和类型，并在所得文法中对每条规则进行形式化，从而将不可靠的一次性翻译转变为一系列受约束、可检查的步骤。在涵盖Solidity和Rust的六个基准测试上，相比缺乏文法归纳或在……

    arXiv:2602.04892v2 Announce Type: replace-cross  Abstract: Ensuring that API implementations and usage comply with natural language programming rules is critical for software correctness, security, and reliability. Formal verification can provide strong guarantees but requires precise specifications, which are difficult and costly to write manually. To address this challenge, we present Doc2Spec, a multi-agent framework that automatically induces a domain-specific grammar from natural-language API rules and uses it to guide specification generation. Doc2Spec fixes a domain-agnostic logical skeleton as a grammar template, prompts LLMs to infer domain-specific predicates and sorts, and formalizes each rule within the resulting grammar, turning an unreliable one-shot translation into a sequence of constrained, checkable steps. Across six benchmarks spanning Solidity and Rust, Doc2Spec improves precision by 0.28 and recall by 0.37 over baselines that lack grammar induction or perform it in
    
[^45]: 你“分叉”忘记了吗？基于全局历史分析检测开源分叉仓库中的1-day漏洞

    Did You Forkget It? Detecting One-Day Vulnerabilities in Open-source ForksWith Global History Analysis

    [https://arxiv.org/abs/2511.05097](https://arxiv.org/abs/2511.05097)

    本文提出一种基于Software Heritage全局代码图的全局历史分析方法，可在提交级别跨分叉仓库传播漏洞信息，自动检测开源分叉仓库中已知但未修补的1-day漏洞，弥补了传统历史分析方法无法追踪分叉中漏洞的不足。

    

    追踪从第三方开源软件继承而来的漏洞是一个众所周知的挑战，通常通过追踪依赖信息链来解决。然而，漏洞还可以通过分叉传播：一个在漏洞被引入之后、补丁发布之前被分叉（fork）的代码仓库，即使在原始仓库中漏洞早已被修复，也可能长期处于易受攻击的状态。历史分析方法已被用于大规模追踪存在漏洞的软件版本，但这类方法无法追踪分叉仓库中的漏洞，只能让分叉仓库的维护者手动识别。本文提出了一种全局历史分析方法，帮助软件开发者识别分叉仓库中的1-day漏洞（已知但尚未修补的漏洞）。该方法利用Software Heritage档案所捕获的公共代码全局图，在提交级别传播漏洞信息，

    arXiv:2511.05097v3 Announce Type: replace-cross  Abstract: Tracking vulnerabilities inherited from third-party open-source software is a well-known challenge, often addressed by tracing the threads of dependency information. However, vulnerabilities can also propagate through forking: a code repository forked after the introduction of a vulnerability, but before it is patched, may remain vulnerable long after the vulnerability has been fixed in the initial repository. History analysis approaches are used to track vulnerable software versions at scale. However, such approaches fail to track vulnerabilities in forks, leaving fork maintainers to identify them manually. This paper presents a global history analysis approach to help software developers identify one-day (known but unpatched) vulnerabilities in forked repositories. Leveraging the global graph of public code, as captured by the Software Heritage archive, our approach propagates vulnerability information at the commit level and
    
[^46]: SEER：面向推理模型的自增强思维链压缩方法

    SEER: Self-Enhancing Chain-of-Thought Compression for Reasoning Models

    [https://arxiv.org/abs/2509.14093](https://arxiv.org/abs/2509.14093)

    该论文通过实证研究揭示推理模型在代码生成中常产生冗长思维链并引发截断与不稳定生成问题，并据此提出SEER方法，通过自增强的方式压缩思维链以降低推理开销。

    

    思维链提示能够显著提升大语言模型的推理能力，但由于推理轨迹冗长且难以控制，往往伴随着高昂的推理成本。这种开销在软件工程任务（如代码生成）中尤为突出，因为这类任务对延迟和输出可靠性都有较高要求。为了更好地理解这一权衡，我们在广泛使用的代码生成基准上开展了实证研究，观察到许多现代推理模型会产生过度冗长的思维链（通常长达数千个token），这经常导致生成被截断且输出不稳定。通过使用严格的n-gram重复检测器，我们发现绝大多数观察到的截断都与退化的循环行为相关。此外，一项针对HumanEval/129的案例研究表明，失败的生成结果可能比成功的更长，这说明过长的推理所带来的收益有限。受此启发……（原文摘要在此处截断）

    arXiv:2509.14093v3 Announce Type: replace-cross  Abstract: Chain-of-Thought (CoT) prompting can substantially improve the reasoning ability of large language models (LLMs), but it often comes with high inference cost due to long and poorly controlled reasoning traces. This overhead is particularly problematic in software engineering tasks (e.g., code generation), where both latency and output reliability matter. To better understand this trade-off, we conduct an empirical study on widely used code generation benchmarks and observe that many modern reasoning models produce excessively verbose CoTs (often thousands of tokens), which frequently leads to truncation and unstable generation. Using a strict n-gram repetition detector, we find that most observed truncations are associated with degenerate looping behaviors. In addition, a HumanEval/129 case study shows that failed generations can be longer than successful ones, suggesting limited returns from overlong reasoning. Motivated by th
    

