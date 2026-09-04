# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SWE-Gate: Passing Functional Tests Is Not Enough for Software Engineering Agents](https://arxiv.org/abs/2609.04167) | 提出 SWE-Gate 基准，从真实 PR 评审评论中提取评审约束并构建带独立功能测试与约束测试的仓库级修复实例，首次将软件工程智能体的问题解决能力与评审约束遵守能力区分评估。 |
| [^2] | [ATIBA: Grounded Integrity and Quality Checking for Research Papers](https://arxiv.org/abs/2609.04123) | ATIBA 是一款对研究论文执行五项基于证据的完整性与质量自动检查的工具，涵盖参考文献核查、会议征稿要求合规与实证标准合规，所有判定结论均锚定于可逐字验证的原文引文，从而防止幻觉并保证检查结果可溯源。 |
| [^3] | [DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training](https://arxiv.org/abs/2609.04094) | DRACO通过在训练中动态生成评分准则，并以闭式解方式将轨迹级评判重新分配到具体步骤，解决了无真实成功信号时长程智能体训练的细粒度信用分配问题，在AppWorld上显著超越基础模型和稀疏奖励GRPO。 |
| [^4] | [PatchBench: Evaluating AI Agents for Vulnerability Patching](https://arxiv.org/abs/2609.04075) | 该研究提出补丁相似度度量方法，发现25%的AI智能体漏洞修复补丁存在记忆历史开发者补丁或仅通过修补崩溃堆栈来抑制崩溃而非修复根本原因的问题，揭示了现有漏洞修复评估方法面临的有效性威胁。 |
| [^5] | [When Models Edit Too Much: On the Fidelity of Minimal Code Edits](https://arxiv.org/abs/2609.04061) | 该研究揭示了前沿大语言模型在修复代码时普遍存在“过度编辑”问题（即使如GPT-5.5这样的强模型也不例外），并提出通过一条简单的保留指令即可显著减少不必要的代码改动、降低认知复杂度，同时提升修复准确率。 |
| [^6] | [LabelMate: An LLM-Driven Framework for Refined Issue Report Labeling](https://arxiv.org/abs/2609.04055) | 本文提出了LabelMate框架，利用大语言模型从历史问题报告中自动构建针对特定项目的标签体系，并自动为新问题报告分配相关标签，克服了现有方法需要大量人工干预、标签过于通用以及依赖已标注数据集的局限。 |
| [^7] | [CROCODIL: Cross-Model Code Editing with LLMs](https://arxiv.org/abs/2609.03894) | 论文发现大语言模型在编辑其他模型生成的陌生代码时会产生过多且过度的改动，为此提出了CROCODIL后训练框架，通过相似性奖励惩罚大幅改动并结合执行验证，在保证功能正确性的同时有效减少跨模型代码编辑中的过度修改。 |
| [^8] | [No One Left Behind: Cross-Level Analysis for Sustainable Software Engineering](https://arxiv.org/abs/2609.03861) | 本愿景论文指出软件可持续性挑战是系统性的，源于组织、过程与产品各社会技术层面之间的相互作用，因此需要对软件工程进行跨层次分析，以理解并解决可持续性问题相互关联、相互强化的根本原因。 |
| [^9] | [Quantisation of Abstract Data Types](https://arxiv.org/abs/2609.03778) | 该论文在泛代数框架下提出抽象量子数据类型的概念，形式化定义了经典数据类型的量子化，使等式规范可可靠地提升到量子设定，并将位预言机和相位预言机统一为该一般构造的特例。 |
| [^10] | [Virtual Testing of Automated Driving Systems through Credible Simulations](https://arxiv.org/abs/2609.03760) | 本文借鉴其他安全关键领域的成熟实践，提出了一个基于风险的框架，用于评估自动驾驶系统安全评估中仿真工具链的可信度，克服了传统仅依赖验证方法的扩展性不足问题。 |
| [^11] | [Can LLMs Extract Architectural Design Decisions from Source Code Commits? - A Preliminary Exploratory Study](https://arxiv.org/abs/2609.03721) | 该初步探索性研究表明，四种大语言模型在零样本和少样本提示下能够有效从源代码提交中提取架构设计决策，所有模型的BERT-F1均超过0.81，且少样本提示能进一步提升提取效果。 |
| [^12] | [Code Transformation Rule Synthesis using LLMs: Potential and Limits](https://arxiv.org/abs/2609.03592) | 本研究首次针对 Comby、GritQL 和 Ast-Grep 三种转换规则语言开展系统性实证研究，证明前沿大语言模型（如 GPT-5.4）的转换规则合成已超越概念验证阶段，在四类软件演化任务中表现优异，而小型开放权重模型仅能有效处理简单的局部变更。 |
| [^13] | [The Psychological Costs of Artificial Intelligence Adoption in Software Engineering](https://arxiv.org/abs/2609.03456) | 本研究首次关注软件工程领域组织采用AI过程中软件专业人员所承受的心理成本，挑战了“AI应用于软件工程是无成本的”这一常见假设。 |
| [^14] | [TIPCODER: Reinforcement Learning Boosted Test-time Instruction Proposer for Code Generation](https://arxiv.org/abs/2609.03309) | TipCoder提出了一种测试时指令提出器，通过强化学习与边际效用奖励在代码生成前自动生成针对特定问题的辅助提示，并结合奖励模型事后选择机制，有效提升代码生成的成功率。 |
| [^15] | [Refusing the Impossible: A Taxonomy and Benchmark for Code Hallucination in Large Language Models](https://arxiv.org/abs/2609.03267) | 该论文提出了一种将代码幻觉与普通代码错误区分开的三维分类体系（根据性、表现层次、行为），并构建了包含270个故意不可满足任务的对抗性基准，其中正确的模型行为应是拒绝生成。 |
| [^16] | [Two Truths and A Lie? Benchmarking Off-the-Shelf LLMs for Requirements Quality Assessment: Performance, False Alarms, and Misses](https://arxiv.org/abs/2609.03230) | 该论文首次对现成大语言模型在需求质量评估中的表现进行了系统性基准测试，基于INCOSE专家标准评估了两个系列的十个模型在多次运行、多组需求和多种采样温度下的性能、误报与漏报情况。 |
| [^17] | [Compound Prompt Constraints in LLM Code Generation: A Factorial Study of Format, Persona, and Urgency](https://arxiv.org/abs/2609.03156) | 本文通过3×3×3全因子实验（27种约束组合、5个OpenAI模型、164道HumanEval+问题、22,140次贪婪解码评估）系统考察格式、角色与紧迫性三类提示约束对LLM代码生成可靠性的联合影响，并将复合效应分解为加性预测与超加性交互退化项。 |
| [^18] | [Large Language Models and Language Server Protocol: a match made in context](https://arxiv.org/abs/2609.03086) | 该论文提出了Eiffel-tools，一个将大型语言模型与语言服务器协议结合的Eiffel语言工具，通过丰富的程序化提示和形式化验证器的自动重试机制，实现了76%至95%的bug修复率。 |
| [^19] | [Requirements After the First Edit: Mining Late Requirement Emergence and Rework in Real-World Coding-Agent Sessions](https://arxiv.org/abs/2609.03028) | 该论文通过对3,553个真实编码代理会话的挖掘，首次将实现后新需求的到来与其造成的代码失效（删除或替换已编写的代理代码行）直接关联起来，发现需求到来后引发的代码失效量约为匹配非需求事件的两倍。 |
| [^20] | [Boundary-Mutation Testing for Pattern-Based Secret Detection: A Rule-Level Method and Cross-Scanner Evaluation](https://arxiv.org/abs/2609.02983) | 提出边界变异测试方法，在规则级别评估基于模式的秘密扫描器，发现当凭证以连字符结尾时检测率从0.9976骤降至0.5233，并验证了可恢复完全鲁棒性的修复方案。 |
| [^21] | [The Illusion of Independent Quorums: Epistemic Fault Domains and Correlated Cognitive Failures in Agentic Quorums](https://arxiv.org/abs/2609.02925) | 论文提出认知故障域（EFD）与结构认知割 \kappa_E 来量化多智能体仲裁团因共享上游输入而产生的关联性认知失效风险，并证明单纯扩大仲裁团规模或增加投票者并不能带来真正的认知冗余。 |
| [^22] | [The Web-CLI: Verifiable Privacy for Tools, Models, and Inference Engines in the Browser](https://arxiv.org/abs/2608.28950) | 提出 Web-CLI 架构，将命令行工具、模型和推理引擎以零安装、可离线的浏览器应用形式完全在客户端运行，通过架构设计而非隐私策略实现可验证的隐私保护。 |
| [^23] | [LLM-Based Test Oracles: Source-of-Authority Taxonomy -- A Systematic Literature Review](https://arxiv.org/abs/2607.05031) | 本综述首次按权威来源对LLM测试预言机进行分类，发现超过半数预言机在无规范情况下仅依赖模型训练知识作出判决，揭示了该领域信任基础的隐患。 |
| [^24] | [LLM4Log: A Systematic Review of Large Language Model-based Log Analysis](https://arxiv.org/abs/2604.16359) | 本文对基于大语言模型的日志分析研究进行了系统性综述，覆盖从日志语句生成与维护、日志解析结构化，到异常检测、故障预测、根因分析和日志摘要的端到端流水线，并分析了LLM在此领域的优势与部署风险。 |
| [^25] | [A Longitudinal Study of Dependency Reclassifications in JavaScript Projects](https://arxiv.org/abs/2604.08747) | 本研究通过对33,087个JavaScript项目package.json文件的提交级分析，首次系统揭示了依赖项重新分类（包括移除和角色重分配）是一种普遍的维护活动，存在于79.1%的项目中，占所有依赖项维护提交的19.4%。 |
| [^26] | [A Physics-Informed Neuro-Fuzzy Framework for Quantum Error Attribution](https://arxiv.org/abs/2602.21253) | 该论文提出一种结合ANFIS与物理特征工程的神经模糊框架，通过Bhattacharyya否决这一硬物理约束在IBM 156量子比特处理器上以89.5%的准确率有效区分量子计算中的软件缺陷与随机硬件噪声。 |
| [^27] | [Detecting Multiple Semantic Concerns in Tangled Code Commits](https://arxiv.org/abs/2601.21298) | 本文首次将纠缠代码提交中的多关注点检测建模为多标签分类问题，并基于真实数据构建了人工纠缠提交的受控数据集，填补了使用语言模型检测提交中多个语义关注点这一研究空白。 |
| [^28] | [Security in the Age of AI Teammates: An Empirical Study of Agentic Pull Requests on GitHub](https://arxiv.org/abs/2601.00477) | 本研究基于包含33,000余个拉取请求的AIDev数据集，对GitHub上自主编码智能体提交的安全相关PR进行了大规模实证分析，系统刻画了AI智能体在软件安全方面的贡献模式、审查与接受情况以及导致PR被拒绝的关键信号。 |
| [^29] | [Evolving Excellence: Automated Optimization of LLM-based Agents](https://arxiv.org/abs/2512.09108) | 本文提出ARTEMIS，一个无代码的进化优化平台，通过语义感知的遗传算子自动联合优化LLM智能体的提示词、工具描述和参数等配置，无需架构修改即可显著提升智能体性能。 |
| [^30] | [Extending Fill-In-the-Middle with Instructions for Steerable Code Completion](https://arxiv.org/abs/2509.24637) | 提出指令感知中间填充（IFIM）微调方法，通过在FIM结构中引入结构分离的专门指令部分，使代码补全模型能够有效理解并优先处理开发者的自然语言指令，从而实现可控的代码补全。 |
| [^31] | [SWIRL: Interactive Sensemaking of Tool-Generated Warnings through Customized Summaries](https://arxiv.org/abs/2508.07169) | SWIRL通过交互式定制摘要和用户反馈动态推断分组规则，帮助程序员更高效地理解和归类静态分析工具生成的大量警告。 |

# 详细

[^1]: SWE-Gate：对软件工程智能体而言，通过功能测试还不够

    SWE-Gate: Passing Functional Tests Is Not Enough for Software Engineering Agents

    [https://arxiv.org/abs/2609.04167](https://arxiv.org/abs/2609.04167)

    提出 SWE-Gate 基准，从真实 PR 评审评论中提取评审约束并构建带独立功能测试与约束测试的仓库级修复实例，首次将软件工程智能体的问题解决能力与评审约束遵守能力区分评估。

    

    仓库级软件工程基准测试显著推动了编码智能体评估的发展，但现有基准主要衡量生成的补丁是否通过功能测试，而忽略了源自代码评审的验收约束（评审约束），而这些约束往往决定一个补丁在真实软件开发中是否可被接受。我们提出 SWE-Gate，一个面向软件工程智能体的仓库级基准，它在功能正确性之外，明确评估补丁对评审约束的遵守程度。SWE-Gate 从真实的拉取请求（PR）评审评论中提取评审约束，并围绕这些约束合成仓库级修复实例。每个实例都提供相互独立的功能测试和约束测试，并附带不合规补丁与标准（gold）补丁，从而能够将“问题解决能力”与“评审约束遵守能力”明确区分开来。我们构建了包含 303 个仓库级修复实例的 SWE-Gate……

    arXiv:2609.04167v1 Announce Type: cross  Abstract: Repository-level software engineering benchmarks have significantly advanced the evaluation of coding agents, but existing benchmarks primarily measure whether generated patches pass functional tests and overlook review-derived acceptance constraints (review constraints) that often influence whether a patch is acceptable in real-world software development. We introduce SWE-Gate, a repository-level benchmark for software engineering agents that explicitly evaluates review constraint compliance alongside functional correctness. SWE-Gate derives review constraints from real pull request review comments and synthesizes repository-level repair instances around these constraints. Each instance provides separate functional and constraint tests, together with non-compliant and gold patches, enabling explicit separation between issue resolution capability and review constraint compliance. We construct SWE-Gate with 303 repository-level repair i
    
[^2]: ATIBA：面向研究论文的可溯源完整性与质量检查

    ATIBA: Grounded Integrity and Quality Checking for Research Papers

    [https://arxiv.org/abs/2609.04123](https://arxiv.org/abs/2609.04123)

    ATIBA 是一款对研究论文执行五项基于证据的完整性与质量自动检查的工具，涵盖参考文献核查、会议征稿要求合规与实证标准合规，所有判定结论均锚定于可逐字验证的原文引文，从而防止幻觉并保证检查结果可溯源。

    

    检查稿件的参考文献完整性、其是否符合目标会议/期刊的特定投稿规则、以及其是否遵循社区报告标准，这些工作需要手动完成、重复繁琐，且每个会议/期刊的要求各不相同，因此在实践中往往执行得不一致或干脆被跳过。我们提出了 ATIBA，这是一个对稿件执行五项基于证据的完整性与质量检查的工具：参考文献完整性检查，将每条引文与文献信息源核对，标记已撤稿或无法找到的参考文献；会议/赛道合规性检查，直接从会议自身的征稿启事页面提取投稿标准并据此评估稿件，每项判定都锚定于该页面的逐字引文；针对 ACM SIGSOFT 实证标准的合规性检查，并配有防幻觉机制，会丢弃任何无法在稿件中逐字定位的证据引文；多模式 AI 评审（特定会议模式、正式模式以及……（原文摘要在此处截断）

    arXiv:2609.04123v1 Announce Type: new  Abstract: Checking a manuscript's reference integrity, its compliance with a target venue's specific submission rules, and its adherence to community reporting standards is manual, repetitive, and different for every venue so in practice it is done inconsistently or skipped. We present ATIBA, a tool that runs five grounded integrity and quality checks on a manuscript: a reference-integrity check that verifies each citation against bibliographic sources and flags retracted or unfindable references; a venue/track compliance check that derives submission criteria directly from a venue's own call-for-papers page and evaluates the manuscript against them, each verdict anchored to a verbatim quote from that page; an empirical-standards compliance check against the ACM SIGSOFT Empirical Standards, with a hallucination defence that discards any evidence quote it cannot locate verbatim in the manuscript; a multi-mode AI review (venue-specific, formal, and 
    
[^3]: DRACO：基于动态评分准则的长程智能体训练细粒度信用分配方法

    DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training

    [https://arxiv.org/abs/2609.04094](https://arxiv.org/abs/2609.04094)

    DRACO通过在训练中动态生成评分准则，并以闭式解方式将轨迹级评判重新分配到具体步骤，解决了无真实成功信号时长程智能体训练的细粒度信用分配问题，在AppWorld上显著超越基础模型和稀疏奖励GRPO。

    

    当任务具备程序化检查器时，基于可验证奖励的强化学习效果良好，但大多数长程智能体领域并不存在这样的检查器。我们在“结果盲设”下开展工作，即真实成功信号不可用的场景。多准则评分准则是提供此类奖励的常用方式；它们对每个轨迹仅评分一次，但单一标量在数十个步骤中是较弱的信号。我们提出DRACO：基于评分准则分布的优势分配信用优化方法。它在训练过程中动态生成评分准则以跟踪策略不断演进的能力，对每个完成的轨迹对这些准则评分一次，并将该评判重新分配到负责相关准则标注的步骤上，从而在GRPO中产生差异化的逐步优势。这种重新分配是闭式解形式，不引入任何需要训练的归因模块。在AppWorld上，DRACO比基础模型提升15.9分，比使用稀疏奖励训练的GRPO提升5.3分。

    arXiv:2609.04094v1 Announce Type: new  Abstract: Reinforcement Learning from Verifiable Rewards works well when a task has a programmatic checker, but most long-horizon agent domains have none. We work in the outcome-blind setting, where ground-truth success signals are not available. Multi-criteria rubrics are a popular way to supply such a reward; they are scored once per trajectory, but a single scalar is a poor signal across tens of steps. We propose DRACO: Distributing Rubric-based Advantage for Credit Optimization. It generates rubrics dynamically during training to track the policy's evolving capability, scores those rubrics once per completed trajectory, and redistributes that judgment over the steps responsible for annotated rubrics to produce differentiated per-step advantages in GRPO. The redistribution is closed-form and does not introduce any trained attribution module. On AppWorld, DRACO gains 15.9 points over the base model and 5.3 points over GRPO trained with a sparse 
    
[^4]: PatchBench：评估AI智能体的漏洞修复能力

    PatchBench: Evaluating AI Agents for Vulnerability Patching

    [https://arxiv.org/abs/2609.04075](https://arxiv.org/abs/2609.04075)

    该研究提出补丁相似度度量方法，发现25%的AI智能体漏洞修复补丁存在记忆历史开发者补丁或仅通过修补崩溃堆栈来抑制崩溃而非修复根本原因的问题，揭示了现有漏洞修复评估方法面临的有效性威胁。

    

    AI智能体最近在自动化漏洞修复方面展现出了强大的性能。然而，现有的评估通常仅通过测试所提供的概念验证（PoC）输入是否仍会触发崩溃来验证补丁的有效性。这给评估的有效性留下了两个关键威胁：智能体可能会复现已记忆的历史开发者补丁，或者它们可能生成仅能抑制所报告崩溃的表面级修复。我们针对C/C++漏洞修复问题研究了这些担忧。我们引入了一种补丁相似度度量方法来检测记忆化的补丁。平均而言，25%的智能体补丁与历史开发者补丁表现出高度相似性，这表明补丁记忆化是漏洞修复评估有效性面临的一个真实威胁。与此同时，智能体还经常利用基准测试的结构，通过在崩溃堆栈跟踪上进行修补来抑制崩溃，从而通过补丁验证，而不是定位并修复根本原因。

    arXiv:2609.04075v1 Announce Type: cross  Abstract: AI agents have recently demonstrated strong performance in automated vulnerability patching. However, existing evaluations often validate a patch only by testing whether the provided Proof-of-Concept (PoC) input still triggers a crash. This leaves two key threats to validity: agents may reproduce memorized historical developer patches, or they may generate surface-level fixes that only suppress the reported crash.   We study these concerns for C/C++ vulnerability patching. We introduce a patch similarity metric to detect memorized patches. On average, 25% of the agent patches exhibit substantial similarity to historical developer patches, indicating that patch memorization is a real threat to the validity of vulnerability patching evaluations. Meanwhile, agents also frequently exploit benchmark structures to pass patch validation by patching on the crash stack trace to suppress the crash, rather than localizing and fixing the root caus
    
[^5]: 当模型编辑过多：论最小代码编辑的保真度

    When Models Edit Too Much: On the Fidelity of Minimal Code Edits

    [https://arxiv.org/abs/2609.04061](https://arxiv.org/abs/2609.04061)

    该研究揭示了前沿大语言模型在修复代码时普遍存在“过度编辑”问题（即使如GPT-5.5这样的强模型也不例外），并提出通过一条简单的保留指令即可显著减少不必要的代码改动、降低认知复杂度，同时提升修复准确率。

    

    大语言模型越来越多地被用于编辑现有代码，但仅仅正确是不够的：有用的修复还应当是最小化的、可审查的，并且忠实于原始实现。我们研究了“过度编辑”现象，即模型重写代码的范围超出修复缺陷所需的趋势。我们基于400个BigCodeBench问题构建了一个评估框架，通过向参考解答中注入受控的AST级（抽象语法树级）破坏，为每个修复任务提供一个已知的最小补丁。研究发现，在各类前沿大语言模型中，过度编辑现象普遍存在，即使是在GPT-5.5这样的强大模型中也是如此：高Pass@1可能与不必要的巨大编辑和新增认知复杂度并存。一条保留指令能够显著减少这种行为，将平均超额Levenshtein距离从0.195降至0.131，减少26.6%的新增认知复杂度，并使Pass@1提高2.3个百分点。然而，这些收益并非简单地源于更大的推理（摘要在此处被截断）

    arXiv:2609.04061v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to edit existing code, but correctness alone is not enough: useful repairs should also be minimal, reviewable, and faithful to the original implementation. We study over-editing, the tendency of a model to rewrite code beyond what is required to fix a bug. We construct an evaluation framework from 400 BigCodeBench problems by injecting controlled AST-level corruptions into reference solutions, giving each repair task a known minimal patch. Across frontier LLMs, over-editing is widespread even among strong models like GPT-5.5: high Pass@1 can coexist with unnecessarily large edits and added cognitive complexity. A preservation instruction substantially reduces this behavior, lowering average excess Levenshtein distance from 0.195 to 0.131, reducing added cognitive complexity by 26.6%, and increasing Pass@1 by 2.3 points. However, these gains do not simply follow from a larger reasoning 
    
[^6]: LabelMate：一种由大语言模型驱动的精细化问题报告标注框架

    LabelMate: An LLM-Driven Framework for Refined Issue Report Labeling

    [https://arxiv.org/abs/2609.04055](https://arxiv.org/abs/2609.04055)

    本文提出了LabelMate框架，利用大语言模型从历史问题报告中自动构建针对特定项目的标签体系，并自动为新问题报告分配相关标签，克服了现有方法需要大量人工干预、标签过于通用以及依赖已标注数据集的局限。

    

    软件用户经常向产品的问题跟踪系统提交问题报告，以报告缺陷、提出改进建议或提出其他与产品相关的关注点。为这些问题报告添加标签有助于进行有效的规划并提高社区参与度。然而，由于设计合适的标签分类体系并将该体系中合适的标签分配给新的问题报告需要大量的人工工作，许多问题报告仍然没有被标注。现有的自动化标注方法试图缓解这些挑战，但它们存在一些关键局限性，例如需要大量的人工干预、只能分配通用标签，以及依赖现有的已标注数据集。为了解决这些局限性，我们提出了LabelMate，一种新颖的大语言模型（LLM）驱动的框架，该框架（1）从历史问题报告中推导出全面的、针对特定项目的标签集合，（2）自动为新的问题报告分配相关标签。

    arXiv:2609.04055v1 Announce Type: new  Abstract: Software users often submit issue reports to a product's issue tracking system to report defects, suggest enhancements, or raise other product-related concerns. Labeling these issue reports supports effective planning and improves community engagement. However, many issue reports remain unlabeled due to the substantial manual effort required to design an appropriate label taxonomy, then assign suitable labels from this taxonomy to new issue reports. Existing automated labeling approaches attempt to mitigate these challenges. However, they suffer from key limitations, such as extensive manual intervention, the assignment of generic labels, and a dependence on existing labeled datasets. To address these limitations, we propose LabelMate, a novel Large Language Model (LLM)-driven framework that (1) derives a comprehensive, project-specific label set from historical issue reports and (2) automatically assigns relevant labels to new issue rep
    
[^7]: CROCODIL：基于大语言模型的跨模型代码编辑

    CROCODIL: Cross-Model Code Editing with LLMs

    [https://arxiv.org/abs/2609.03894](https://arxiv.org/abs/2609.03894)

    论文发现大语言模型在编辑其他模型生成的陌生代码时会产生过多且过度的改动，为此提出了CROCODIL后训练框架，通过相似性奖励惩罚大幅改动并结合执行验证，在保证功能正确性的同时有效减少跨模型代码编辑中的过度修改。

    

    大型语言模型（LLMs）已成为代码生成和编辑中无处不在的工具。然而，开发团队通常会使用多个LLM助手。不同的开发者可能偏好不同的模型，而且单个开发者也可能在不同的编码会话中切换使用不同模型。因此，某个模型所做的编辑经常被应用到最初由另一个模型生成的陌生代码上。这些LLM通常在不同的数据集上训练，因此具有不同的风格偏好。那么，当LLM编辑最初由另一个具有不同编码风格的LLM编写的陌生代码时，它们的表现会有所不同吗？我们发现，模型倾向于对陌生代码进行更多、且往往是过度的编辑。我们提出了CROCODIL（基于大语言模型的跨模型代码编辑），这是一个后训练框架，用于在保持功能正确性的同时减少过度编辑。CROCODIL的相似性奖励会对大幅改动进行惩罚，而其执行……（注：原文摘要在此处截断）

    arXiv:2609.03894v1 Announce Type: new  Abstract: Large language models (LLMs) have become ubiquitous tools for code generation and editing. However, development teams often use multiple LLM assistants. Different developers may prefer different models, and individual developers may switch between models across different coding sessions. Because of this, the edits any one model makes are frequently applied to foreign code originally generated by another model. These LLMs are often trained on different datasets, and as a result have different stylistic preferences. Do LLMs behave differently when they edit foreign code originally written by a different LLM with a different coding style? We find that models tend to make more, and often excessive, edits on foreign code. We introduce CROCODIL (Cross-model Code Editing with LLMs), a post-training framework for reducing excessive edits while preserving functional correctness. CROCODIL's similarity reward penalizes large changes, while its exec
    
[^8]: 不落下任何人：面向可持续软件工程的跨层次分析

    No One Left Behind: Cross-Level Analysis for Sustainable Software Engineering

    [https://arxiv.org/abs/2609.03861](https://arxiv.org/abs/2609.03861)

    本愿景论文指出软件可持续性挑战是系统性的，源于组织、过程与产品各社会技术层面之间的相互作用，因此需要对软件工程进行跨层次分析，以理解并解决可持续性问题相互关联、相互强化的根本原因。

    

    软件工程师期望软件系统在整个生命周期中都高效、可维护、经济可行且具有社会责任感。然而不幸的是，在软件开发的组织、过程或产品层面做出的决策——即使改善了其中某个目标——往往会在其他层面产生意想不到的长期后果。为此，软件工程研究已探索了实现软件可持续性的多种途径，包括能源效率、资源优化和代码可维护性。然而，现有研究对于可持续性问题如何在软件工程的社会技术各层面之间产生并相互强化，所提供的解释仍然有限。此外，我们认为可持续性挑战并非孤立存在，而是系统性的：它们源于组织优先事项、开发实践和技术条件之间的相互作用。在这篇愿景论文中，我们介…（原文在此处截断）

    arXiv:2609.03861v1 Announce Type: new  Abstract: Software engineers expect a software system to be efficient, maintainable, economically viable, and socially responsible throughout its lifecycle. Unfortunately, decisions made at the organizational, process, or product levels of software development - even when they improve one of these goals - often have unintended long-term consequences at other levels. In response, software-engineering research has explored different ways to achieve software sustainability, including energy efficiency, resource optimization, and code maintainability. However, existing research offers limited explanations for how sustainability problems emerge and reinforce one another across the socio-technical levels of software engineering. Moreover, we argue that sustainability challenges are not isolated but rather systemic: They stem from interactions among organizational priorities, development practices, and technical conditions. In this vision paper, we intro
    
[^9]: 抽象数据类型的量子化

    Quantisation of Abstract Data Types

    [https://arxiv.org/abs/2609.03778](https://arxiv.org/abs/2609.03778)

    该论文在泛代数框架下提出抽象量子数据类型的概念，形式化定义了经典数据类型的量子化，使等式规范可可靠地提升到量子设定，并将位预言机和相位预言机统一为该一般构造的特例。

    

    在本文中，我们在泛代数的框架内引入了抽象量子数据类型的概念。这一概念为描述量子编程中的数据抽象提供了代数基础。我们形式化地定义了经典数据类型的量子化，并证明其等式规范可以被可靠地提升到量子设定中。经典函数的两种标准量子化方法——位预言机和相位预言机——作为这一一般构造的特例而自然出现。我们通过量子数组和量子纠错码的应用来展示该框架，说明如何通过数据类型量子化的视角来理解它们。我们进一步建立了量子化在何种条件下能够保持经典数据类型的结构关系与构造，包括嵌入、同构和乘积。

    arXiv:2609.03778v1 Announce Type: cross  Abstract: In this paper, we introduce a notion of abstract quantum data type within the framework of universal algebra. This notion provides an algebraic foundation for describing data abstraction in quantum programming. We formally define a quantisation of classical data types and show that their equational specifications can be soundly lifted to the quantum setting. Two standard quantisation methods for classical functions, namely the bit oracle and the phase oracle, arise as special cases of this general construction. We illustrate the framework with applications to quantum arrays and quantum error-correcting codes, showing how they can be understood through the lens of data-type quantisation. We further establish conditions under which quantisation preserves structural relationships and constructions of classical data types, including embeddings, isomorphisms, and products.
    
[^10]: 通过可信仿真实现自动驾驶系统的虚拟测试

    Virtual Testing of Automated Driving Systems through Credible Simulations

    [https://arxiv.org/abs/2609.03760](https://arxiv.org/abs/2609.03760)

    本文借鉴其他安全关键领域的成熟实践，提出了一个基于风险的框架，用于评估自动驾驶系统安全评估中仿真工具链的可信度，克服了传统仅依赖验证方法的扩展性不足问题。

    

    仿真在道路交通领域日益被用于支持安全相关的决策，特别是用于自动驾驶系统（ADS）的评估与审批。由于ADS行为的复杂性以及其运行设计域的庞大规模，完全依赖物理测试已不切实际，因此虚拟测试（VT）在审批阶段得到了广泛应用。这一转变引发了一个关键问题：用于支持道路安全决策的建模与仿真（M&S）结果是否可信。目前ADS领域的虚拟测试认可方法通常仅依赖验证实践，而这种做法在应用于复杂的多工具仿真环境时已被证明扩展性不佳。为解决这一局限，本文借鉴其他安全关键领域的成熟实践，提出了一个基于风险的框架，用于评估ADS安全评估中仿真工具链的可信度。

    arXiv:2609.03760v1 Announce Type: cross  Abstract: Simulation is increasingly used to support safety-related decision-making in road transport, particularly for the assessment and approval of automated driving systems (ADS). The complexity of ADS behavior and size of their operational design domains make exclusive reliance on physical testing impractical, leading to extensive use of virtual testing (VT) during the approval phase. This shift raises critical questions regarding the credibility of modelling and simulation (M&S) results used to support road safety decisions. Current VT accreditation approaches in the ADS domain typically rely on validation-only practices, which have been shown to scale poorly when applied to complex, multi-tool simulation environments. To address this limitation, this paper proposes a risk-based framework for assessing the credibility of simulation toolchains used in ADS safety evaluation, drawing inspiration from established practices in other safety-crit
    
[^11]: 大语言模型能否从源代码提交中提取架构设计决策？——一项初步探索性研究

    Can LLMs Extract Architectural Design Decisions from Source Code Commits? - A Preliminary Exploratory Study

    [https://arxiv.org/abs/2609.03721](https://arxiv.org/abs/2609.03721)

    该初步探索性研究表明，四种大语言模型在零样本和少样本提示下能够有效从源代码提交中提取架构设计决策，所有模型的BERT-F1均超过0.81，且少样本提示能进一步提升提取效果。

    

    背景：架构设计决策（ADD）捕捉了软件系统结构与演进背后的原理依据，但很少被明确记录，往往隐藏在源代码提交中。恢复这些决策对于架构知识管理（AKM）非常重要。问题：由于ADD具有隐式且非结构化的特性，从提交中提取ADD极具挑战性。大语言模型（LLM）在理解代码和文本方面已展现出强大能力，但其在该任务上的有效性仍未得到充分探索。研究：我们开展了一项初步研究，使用四个大语言模型（Gemini 3 Pro、DeepSeek R1、Kimi K2、Qwen3），采用零样本和少样本提示方法，对来自开源项目的30条开发者编写的ADD进行提取测试。我们使用ROUGE-L、BLEU、METEOR和BERTScore对输出进行评分，并由一位作者对Gemini的输出进行人工评审。结果：所有模型的BERT-F1得分均超过0.81，少样本提示提升了对齐效果（Gemini的BERT-F1：0.[摘要截断于此]）

    arXiv:2609.03721v1 Announce Type: cross  Abstract: Context: Architectural Design Decisions (ADDs) capture the rationale behind the structure and evolution of software systems but are rarely documented explicitly, and are often hidden inside source code commits. Recovering them is important for Architectural Knowledge Management (AKM). Problem: Extracting ADDs from commits is challenging due to their implicit and unstructured nature. Large Language Models (LLMs) have shown strong capabilities in understanding code and text, yet their effectiveness for this task remains underexplored. Study: We present a preliminary study using four LLMs (Gemini 3 Pro, DeepSeek R1, Kimi K2, Qwen3) with zeroshot and fewshot prompting on 30 developer-written ADDs from open-source projects. We score outputs with ROUGE-L, BLEU, METEOR, and BERTScore, and one author manually reviews the Gemini outputs. Results: All models reach a BERT-F1 above 0.81, and fewshot prompting improves alignment (Gemini BERT-F1: 0.
    
[^12]: 使用大语言模型进行代码转换规则合成：潜力与局限

    Code Transformation Rule Synthesis using LLMs: Potential and Limits

    [https://arxiv.org/abs/2609.03592](https://arxiv.org/abs/2609.03592)

    本研究首次针对 Comby、GritQL 和 Ast-Grep 三种转换规则语言开展系统性实证研究，证明前沿大语言模型（如 GPT-5.4）的转换规则合成已超越概念验证阶段，在四类软件演化任务中表现优异，而小型开放权重模型仅能有效处理简单的局部变更。

    

    由于黑盒特性，大语言模型（LLM）存在可解释性受限和缺乏确定性的问题，其使用成本也可能上升，尤其是在大型代码库上执行重复性任务时。为了缓解这些问题，我们针对三种用于转换规则的领域特定语言（即 Comby、GritQL 和 Ast-Grep）开展了一项新颖的实证研究。我们在六个多样化数据集上评估了三个大语言模型（GPT-5.4、GPT-oss-120B 和 Llama3.1-8B），这些数据集涵盖四类软件演化任务：API 误用修正、程序修复、API 迁移和语言版本迁移。我们的结果证明，借助强大的前沿模型，转换规则合成已超越了概念验证阶段。GPT-5.4 在大多数基准测试中实现了持续较高的规则适用率，并生成了最接近真实标准的转换结果。较小型的开放权重模型 GPT-oss-120B 和 Llama3.1-8B 在处理较简单的局部变更时仍然有效，但在处理（原文摘要在此处截断）。

    arXiv:2609.03592v1 Announce Type: new  Abstract: Due to their black-box nature, LLMs suffer from limited explain- ability and a lack of determinism. Their usage cost can also rise, particularly with repetitive tasks on large codebases. To mitigate this, we conduct a novel empirical study targeting three domain- specific languages for transformation rules, namely Comby, GritQL, and Ast-Grep. We evaluate three LLMs (GPT-5.4, GPT-oss-120B, and Llama3.1-8B) on six diverse datasets covering four software- evolution tasks: API misuse correction, program repair, API migra- tion, and language version migration. Our results provide evidence that transformation rule synthesis moves beyond proof-of-concept with strong frontier models. GPT-5.4 achieves consistently high rule applicability rates and produces transformations closest to the ground truth across most benchmarks. Smaller and open-weight GPT-oss-120B and Llama3.1-8B models remain effective for simpler, localized changes but struggle with
    
[^13]: 软件工程中人工智能应用的心理成本

    The Psychological Costs of Artificial Intelligence Adoption in Software Engineering

    [https://arxiv.org/abs/2609.03456](https://arxiv.org/abs/2609.03456)

    本研究首次关注软件工程领域组织采用AI过程中软件专业人员所承受的心理成本，挑战了“AI应用于软件工程是无成本的”这一常见假设。

    

    人工智能越来越多地被用于增强软件工程的工作流程。虽然代码生成仍然是主要应用场景，但各组织正在积极寻求将AI集成到其他实践中，例如测试用例生成和代码审查。组织层面的AI采用策略似乎主要关注生产力等有形成果。然而，AI是一种颠覆性力量，它被引入到那些在生成式AI取得近期进展之前，角色认同、团队规范和工作满意度来源早已确立的环境之中。从历史上看，技术颠覆曾在职场中造成心理和社会层面的压力，范围涵盖焦虑、意义感丧失，乃至技能退化和职业认同的破坏。因此，“AI应用于软件工程是没有成本的”这一假设可能并不准确。为此，本研究试图理解软件专业人员在组织采用AI的过程中所经历的心理成本。

    arXiv:2609.03456v1 Announce Type: cross  Abstract: Artificial intelligence (AI) is increasingly used to augment software engineering (SE) workflows. While code generation remains the main use case, organizations are actively seeking AI integration in other practices such as test cases generation and code reviews. Organizational AI adoption strategies seem to focus on tangible outcomes such as productivity. However, AI is a disruptive force, introduced into settings where role identity, team norms, and the sources of job satisfaction were well established before the recent advances in generative AI. Historically, technological disruptions have caused psychological and social strains in workplaces, ranging from anxiety and eroded meaning to deskilling and disrupted professional identities. The assumption that AI for SE is cost-free may not be accurate. Therefore, in this study we sought to understand the psychological costs software professionals experience during organizational AI adopt
    
[^14]: TIPCODER：强化学习增强的代码生成测试时指令提出器

    TIPCODER: Reinforcement Learning Boosted Test-time Instruction Proposer for Code Generation

    [https://arxiv.org/abs/2609.03309](https://arxiv.org/abs/2609.03309)

    TipCoder提出了一种测试时指令提出器，通过强化学习与边际效用奖励在代码生成前自动生成针对特定问题的辅助提示，并结合奖励模型事后选择机制，有效提升代码生成的成功率。

    

    面向代码生成的测试时扩展通常通过从固定指令采样多个程序来探索解空间。我们研究了一个互补的方向：实例级指令空间探索。我们观察到，许多编码失败源于原始提示所导致的约束缺失、被忽视的边界情况或误导性推理路径。为解决这一问题，我们提出了TipCoder，一个在代码合成之前生成针对特定问题辅助提示的测试时指令提出器。TipCoder将多轮调试轨迹提炼为主动式指导，并使用边际效用奖励通过强化学习进一步优化该提出器。在推理阶段，它会同时生成一个基础解决方案和一个提示引导的解决方案，并应用奖励模型进行事后选择。这种“探索-选择”设计使提示能够发掘候选方案的额外潜力，同时减少因不必要的指导而产生的性能回退。

    arXiv:2609.03309v1 Announce Type: new  Abstract: Test-time scaling for code generation typically explores the solution space by sampling multiple programs from a fixed instruction. We study a complementary direction: instance-level instruction-space exploration. Our observation is that many coding failures stem from missing constraints, overlooked edge cases, or misleading reasoning paths induced by the original prompt. To address this, we propose TipCoder, a test-time instruction proposer that generates problem-specific auxiliary tips before code synthesis. TipCoder distills multi-turn debugging trajectories into proactive guidance and further optimizes the Proposer with reinforcement learning using a marginal-utility reward. At inference time, it generates both a base solution and a tip-guided solution, and applies a Reward Model for post-hoc selection. This exploration-selection design allows tips to expose additional candidate potential while reducing regressions from unnecessary g
    
[^15]: 拒绝不可能之事：大语言模型代码幻觉的分类体系与基准测试

    Refusing the Impossible: A Taxonomy and Benchmark for Code Hallucination in Large Language Models

    [https://arxiv.org/abs/2609.03267](https://arxiv.org/abs/2609.03267)

    该论文提出了一种将代码幻觉与普通代码错误区分开的三维分类体系（根据性、表现层次、行为），并构建了包含270个故意不可满足任务的对抗性基准，其中正确的模型行为应是拒绝生成。

    

    大语言模型（LLMs）经常生成看似合理但缺乏现实依据的代码。这些代码可能导入并不存在的软件包，或声称实现了违反已被证明的定理的算法，同时却仍能编译和运行。我们将“代码幻觉”视为“无根据生成”进行研究，并将其与普通的“代码错误”（即有根据程序中的缺陷）区分开来。我们提出了一个包含三个维度的分类体系：**根据性**（对普遍真理的绝对违反 vs. 对偶然性事实或特定生态系统事实的相对虚构）、**表现层次**（语法、语义或事实层面），以及**行为**（从自信的虚构到退化的输出），并将其组织成一个严重性排序。我们构建了一套**对抗性**测试集，其中包含故意设计为不可满足的任务，正确的回应应当是拒绝生成，并根据我们的分类体系对模型的响应进行归类。该测试集包含270个提示……

    arXiv:2609.03267v1 Announce Type: new  Abstract: Large language models (LLMs) often produce code that looks plausible but is not grounded in reality. The code may import packages that do not exist or claim to implement algorithms that violate proven theorems, while still compiling and running. We study \emph{code hallucination} as \emph{ungrounded generation} and separate it from ordinary \emph{code error} (bugs in otherwise grounded programs). We propose a taxonomy with three dimensions: \textbf{groundedness} (absolute violations of universal truths vs.\ relative fabrications of contingent or ecosystem-specific facts), \textbf{manifestation level} (syntactic, semantic, or factual), and \textbf{behavior} (from confident fabrication to degenerate output), organized into a severity ordering. We build an \textbf{adversarial} suite of deliberately unsatisfiable tasks where the correct response is to refuse and categorize the responses under our taxonomy. The suite contains \textbf{270 prom
    
[^16]: 两真一谎？现成大语言模型在需求质量评估中的基准测试：性能、误报与漏报

    Two Truths and A Lie? Benchmarking Off-the-Shelf LLMs for Requirements Quality Assessment: Performance, False Alarms, and Misses

    [https://arxiv.org/abs/2609.03230](https://arxiv.org/abs/2609.03230)

    该论文首次对现成大语言模型在需求质量评估中的表现进行了系统性基准测试，基于INCOSE专家标准评估了两个系列的十个模型在多次运行、多组需求和多种采样温度下的性能、误报与漏报情况。

    

    需求工程（RE）决定着系统工程（SE）中所有后续工作的质量；在评审周期中未被发现的缺陷需求会传导为设计返工、进度延误和成本超支。由于需求通常以自然语言编写，生成式人工智能的最新进展使人们期待大语言模型（LLM）能够承担需求质量评估这一原本缓慢且高度依赖人类专业知识的任务。然而，关于LLM是否值得信赖来完成此任务的实证证据仍然稀缺。本研究首次对现成大语言模型在需求质量评估中的表现进行了基准测试分析。基于INCOSE质量标准构建的专家真值，我们评估了来自两个系列（OpenAI和Anthropic）的十个模型（每个系列五个世代），涵盖一百次独立运行、两组需求集和五种采样温度。随后提出四项贡献。

    arXiv:2609.03230v1 Announce Type: new  Abstract: Requirements engineering (RE) governs the quality of everything downstream in systems engineering (SE); defective requirements that survive review cycles propagate into design rework, schedule delays, and cost overruns. Because requirements are often written in natural language, recent advances in generative AI have raised expectations that large language models (LLMs) can absorb requirement quality assessment, a task otherwise slow and human expertise-intensive. Yet empirical evidence on whether LLMs can be trusted to do so remains scarce. This study presents the first benchmarking analysis of off-the-shelf LLM performance for requirement quality evaluation. Against an expert-derived ground truth built on INCOSE quality criteria, we evaluate ten models spanning two families (OpenAI and Anthropic) and five generations each, across one hundred independent runs, two requirement sets, and five sampling temperatures. Four contributions follo
    
[^17]: 大语言模型代码生成中的复合提示约束：关于格式、角色与紧迫性的全因子研究

    Compound Prompt Constraints in LLM Code Generation: A Factorial Study of Format, Persona, and Urgency

    [https://arxiv.org/abs/2609.03156](https://arxiv.org/abs/2609.03156)

    本文通过3×3×3全因子实验（27种约束组合、5个OpenAI模型、164道HumanEval+问题、22,140次贪婪解码评估）系统考察格式、角色与紧迫性三类提示约束对LLM代码生成可靠性的联合影响，并将复合效应分解为加性预测与超加性交互退化项。

    

    大语言模型（LLM）在软件工程流水线中的代码生成应用日益增多，而生产环境中的提示词往往同时组合多种约束。本文对输出格式、角色设定和紧迫性表述如何共同影响LLM代码生成可靠性进行了全因子实证研究。我们在受控的3×3×3设计中评估了全部27种组合，并将每个复合条件分解为加性预测项和一个捕捉超加性退化的残差交互项。该研究使用HumanEval+全部164个问题，覆盖来自GPT-4o系列、GPT-4.1系列和o3-mini的五个OpenAI模型，共产生22,140次贪婪解码评估。研究采用格式感知的提取流水线将格式失败与推理失败区分开，并通过McNemar检验、优势比和95%置信区间评估显著性。结果表明，复合约束可能产生架构……（原文摘要在此处截断）

    arXiv:2609.03156v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used in software engineering pipelines for code generation, where production prompts often combine multiple constraints. This paper presents a full-factorial empirical study of how output formatting, persona assignment, and urgency framing jointly affect LLM code-generation reliability. We evaluate all 27 combinations in a controlled 3x3x3 design and decompose each compound condition into an additive prediction and a residual interaction term that captures super-additive degradation. The study uses all 164 HumanEval+ problems across five OpenAI models from the GPT-4o family, GPT-4.1 family, and o3-mini, yielding 22,140 greedy-decoding evaluations. A format-aware extraction pipeline separates formatting failures from reasoning failures, and significance is assessed with McNemar's test, odds ratios, and 95% confidence intervals.   Results show that compound constraints can produce architecture-
    
[^18]: 大型语言模型与语言服务器协议：上下文中的天作之合

    Large Language Models and Language Server Protocol: a match made in context

    [https://arxiv.org/abs/2609.03086](https://arxiv.org/abs/2609.03086)

    该论文提出了Eiffel-tools，一个将大型语言模型与语言服务器协议结合的Eiffel语言工具，通过丰富的程序化提示和形式化验证器的自动重试机制，实现了76%至95%的bug修复率。

    

    本文介绍了Eiffel-tools，这是一个针对Eiffel编程语言的语言服务器协议（LSP）实现，它利用大型语言模型（LLM）来辅助开发经过静态验证的软件。该工具提供了多种交互式和非交互式命令来生成代码和规约。它利用语言和项目特定的知识来精确引导LLM，并使用静态验证器验证输出。它为输入构建丰富的程序化提示，并对输出进行修正或拒绝。此外，它会处理重试，直到程序通过验证。该工具的bug修复能力在2个公开数据集上使用3个模型进行了评估。通过结合LLM和形式化验证器，根据所使用的模型和提示，该工具可以修复76%至95%的bug。结果显示了修复尝试次数与成功率之间的权衡。

    arXiv:2609.03086v1 Announce Type: new  Abstract: This article introduces Eiffel-tools, a language server protocol (LSP) implementation for the Eiffel programming language that uses Large Language Models (LLMs) to aid the development of statically verified software. The tool provides various interactive and non-interactive commands to produce code and specifications. It uses language and project specific knowledge to precisely direct the LLM and verifies the output using a static verifier. It crafts rich programmatic prompts for the input and corrects or rejects the output. Furthermore, it handles the retries until the program passes verification. The tool's bug fixing capability is evaluated on 2 public datasets using 3 models. The tool can fix 76% to 95% of bugs by combining LLMs and a formal verifier depending on the model and prompts used. The results show the trade-off between the number of fixing attempts and the success rate.
    
[^19]: 首次编辑之后的需求：挖掘真实世界编码代理会话中的晚期需求涌现与返工

    Requirements After the First Edit: Mining Late Requirement Emergence and Rework in Real-World Coding-Agent Sessions

    [https://arxiv.org/abs/2609.03028](https://arxiv.org/abs/2609.03028)

    该论文通过对3,553个真实编码代理会话的挖掘，首次将实现后新需求的到来与其造成的代码失效（删除或替换已编写的代理代码行）直接关联起来，发现需求到来后引发的代码失效量约为匹配非需求事件的两倍。

    

    编码代理常常在用户尚未充分表达其需求之前就实施变更，这与需求工程中的一种模式相呼应：利益相关者在系统的部分内容已存在并可供其反应之前，无法表达某项约束。这种波动性在传统项目中与进度和预算超支相关，但仅在发布周期的粒度上得到体现。现有的编码代理研究仅部分弥补了这一空白：精心策划的基准测试在设计上就把需求固定在实现之前，而观察性研究仅报告了用户反悔的频率，却没有将新需求的到来与它们所导致的代码失效联系起来。我们使用3,553个符合条件的SWE-chat会话来解决这一问题，沿三个维度追踪实现后新需求的到来，并在仓库状态可以重放的情况下，将每次需求到来与一个代理指标关联起来：即删除或替换先前由代理编写的代码行。需求的到来之后所引发的代码失效量大约是匹配的非需求事件的两倍。

    arXiv:2609.03028v1 Announce Type: new  Abstract: Coding agents often implement changes before users have fully articulated their requirements, echoing a pattern from requirements engineering: stakeholders cannot express a constraint until part of the system exists to react to. This volatility is associated with schedule and budget overruns in traditional projects, but only at release-cycle granularity. Existing work on coding agents narrows this gap only partway: curated benchmarks fix requirements before implementation by design, and observational studies report pushback frequency without linking arrivals to the code invalidation they cause. We address this using 3,553 eligible SWE-chat sessions, coding post-implementation requirement arrivals along three dimensions and, where repository state can be replayed, linking each arrival to a proxy: deletion or replacement of prior agent-authored lines. A requirement's arrival is followed by roughly twice as much invalidation as matched non-
    
[^20]: 基于模式的秘密检测的边界变异测试：一种规则级方法与跨扫描器评估

    Boundary-Mutation Testing for Pattern-Based Secret Detection: A Rule-Level Method and Cross-Scanner Evaluation

    [https://arxiv.org/abs/2609.02983](https://arxiv.org/abs/2609.02983)

    提出边界变异测试方法，在规则级别评估基于模式的秘密扫描器，发现当凭证以连字符结尾时检测率从0.9976骤降至0.5233，并验证了可恢复完全鲁棒性的修复方案。

    

    基于模式的秘密扫描器通常使用基于示例的测试固件进行验证，这类固件固定了一个变量：凭证周围的文本。我们引入边界变异测试来改变这一上下文，从每条规则自身的正则表达式生成凭证，将其嵌入真实的源代码上下文中，并在规则级别而非工具级别对结果进行分类，从而得出三种检测指标。将该方法应用于三个扫描器——一个包含43条规则的开源扫描器、Gitleaks 8.21.2 和 TruffleHog 3.82.13——在十个上下文中，对主要主体的检测率保持在≥0.9976，但当凭证以连字符结尾时，检测率骤降至0.5233。共有五条规则受到影响：其中两条使用固定数量量词的规则完全且确定性地失效；三条使用可变数量量词的规则会发生回溯并匹配被截断的凭证；熵回退机制虽然挽救了部分失败案例，但降低了其严重性等级。我们验证了一种能够恢复完全鲁棒性的修复方法……

    arXiv:2609.02983v1 Announce Type: cross  Abstract: Pattern-based secret scanners are commonly validated with example-based fixtures that fix one variable: the text surrounding a credential. We introduce boundary-mutation testing to vary that context, generating credentials from each rule's own regular expression, embedding them in realistic source contexts, and classifying outcomes at the rule level rather than the tool level, yielding three detection metrics. Applied to three scanners - a 43-rule open-source scanner, Gitleaks 8.21.2, and TruffleHog 3.82.13 - detection in the primary subject holds at >=0.9976 across ten contexts but collapses to 0.5233 when a credential ends in a hyphen. Five rules are affected: two, with fixed-count quantifiers, fail totally and deterministically; three, with variable-count quantifiers, backtrack and match a truncated credential; an entropy fallback rescues some failures but downgrades their severity. We validate a repair restoring full robustness wit
    
[^21]: 独立仲裁团的幻觉：智能体仲裁团中的认知故障域与相关性认知失效

    The Illusion of Independent Quorums: Epistemic Fault Domains and Correlated Cognitive Failures in Agentic Quorums

    [https://arxiv.org/abs/2609.02925](https://arxiv.org/abs/2609.02925)

    论文提出认知故障域（EFD）与结构认知割 \kappa_E 来量化多智能体仲裁团因共享上游输入而产生的关联性认知失效风险，并证明单纯扩大仲裁团规模或增加投票者并不能带来真正的认知冗余。

    

    多智能体仲裁团被广泛用于授权高风险的基础设施与策略变更，然而不同的审查者往往共享上游遥测数据、文档或工具后端。当上游输入失效时，多张选票会坍缩到同一个被污染的成因上：复制并不等于认知冗余。我们引入了认知故障域和结构认知割 \kappa_E，它相对于一个显式的认知故障基准，量化了覆盖授权联盟所需的建模根故障的最小数量。在闭合因果核算、保守暴露和授权对齐的条件下，\kappa_E 为实现语义性妥协所需的根故障数量（\kappa_S）提供了下界。我们证明：任意大的仲裁团都可能保持 \kappa_E=1；识别共享祖先从不会增加计入的韧性；在固定阈值下增加投票者也无法在兼容暴露条件下提高割值。

    arXiv:2609.02925v1 Announce Type: cross  Abstract: Multi-agent quorums are widely used to authorize high-stakes infrastructure and policy mutations, yet distinct reviewers often share upstream telemetry, documents, or tool backends. When upstream inputs fail, multiple votes collapse onto a single corrupted cause: replication does not imply epistemic redundancy. We introduce Epistemic Fault Domains (EFDs) and the Structural Epistemic Cut \kappa_E, which quantifies the minimum number of modeled root faults whose exposure covers an authorizing coalition relative to an explicit Epistemic Fault Basis. Under closed causal accounting, conservative exposure, and authorization alignment, \kappa_E lower-bounds the number of roots required for semantic compromise (\kappa_S). We prove that arbitrarily large quorums can retain \kappa_E=1, that recognizing shared ancestry never increases credited resilience, and that adding voters at a fixed threshold cannot increase the cut under compatible exposur
    
[^22]: Web-CLI：浏览器中工具、模型与推理引擎的可验证隐私保护

    The Web-CLI: Verifiable Privacy for Tools, Models, and Inference Engines in the Browser

    [https://arxiv.org/abs/2608.28950](https://arxiv.org/abs/2608.28950)

    提出 Web-CLI 架构，将命令行工具、模型和推理引擎以零安装、可离线的浏览器应用形式完全在客户端运行，通过架构设计而非隐私策略实现可验证的隐私保护。

    

    我们提出了 Web-CLI，这是一种新颖的应用程序架构，它将强大的计算能力（编译为 WebAssembly 的命令行工具、通过客户端推理运行时运行的模型，以及 GPU 加速引擎）部署为零安装、可离线使用的浏览器应用程序，同时完整保留底层功能。与需要服务器端处理并将用户数据暴露给第三方的基于 Web 的替代方案不同，Web-CLI 应用程序完全在客户端执行，通过架构而非策略提供可验证的隐私保证。我们定义了该模式及其四个属性：保真性、渐进式披露、离线优先和零数据外流。我们展示了跨不同领域的四个参考实现：ffmpeg-webCLI，一个基于 FFmpeg 构建的浏览器视频编辑器；whisper-webCLI，通过 Transformers.js 实现的语音转录；chat-webCLI，基于 WebLLM 的语言模型推理；以及 3mf-webCLI，一个确定性工具。

    arXiv:2608.28950v1 Announce Type: cross  Abstract: We introduce the Web-CLI, a novel application architecture deploying powerful computational capabilities (command-line tools compiled to WebAssembly, models run through client-side inference runtimes, and GPU-accelerated engines) as zero-install, offline-capable browser applications that preserve full underlying capability. Unlike web-based alternatives that require server-side processing and expose user data to third parties, Web-CLI applications execute entirely on the client, providing a verifiable privacy guarantee by architecture rather than policy. We define the pattern and its four properties: fidelity, progressive disclosure, offline-first, and zero egress. We present four reference implementations across distinct domains: ffmpeg-webCLI, a browser-based video editor built on FFmpeg; whisper-webCLI, speech transcription via Transformers.js; chat-webCLI, WebLLM-based language model inference; and 3mf-webCLI, a deterministic tool 
    
[^23]: 基于大语言模型的测试预言机：权威来源分类法——一项系统性文献综述

    LLM-Based Test Oracles: Source-of-Authority Taxonomy -- A Systematic Literature Review

    [https://arxiv.org/abs/2607.05031](https://arxiv.org/abs/2607.05031)

    本综述首次按权威来源对LLM测试预言机进行分类，发现超过半数预言机在无规范情况下仅依赖模型训练知识作出判决，揭示了该领域信任基础的隐患。

    

    摘要：大语言模型（LLMs）越来越多地通过编写测试预言机或直接充当预言机来决定软件行为是否正确。然而，两个预言机可能看起来相同，却基于不同的依据：一个断言编码了书面规范，另一个仅依赖于模型在训练中学到的内容。先前的二次研究按形式或技术对预言机进行分类，很少依据决定判决可信度的属性——即其权威来源。本系统性文献综述按照2020年系统综述和元分析首选报告项目（PRISMA）指南进行，筛选了2,436条记录至54项纳入研究，并通过引文搜索（滚雪球法）扩展至总计83项。我们沿着三个维度阅读了文献集：预言机权威的来源、其采取的形式以及裁决其的机制。语料库中略多于一半的预言机在没有规范的情况下做出判决。这就是关键所在。

    arXiv:2607.05031v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly decide whether software behaves correctly, either by writing a test oracle or by acting as one. Yet two oracles can look identical and rest on different ground: one assertion encodes a written specification, another only what the model learned in training. Prior secondary studies sort oracles by form or by technique, rarely by the property that governs how far a verdict can be trusted: where its authority comes from. This systematic literature review, reported under the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) 2020 guidelines, screens 2,436 records to 54 included studies, extended by citation searching (snowballing) to 83 in total. We read the corpus along three axes: the source of an oracle's authority, the form it takes, and the mechanism that adjudicates it. Just over half of the corpus reaches a verdict with no specification at all. That is what le
    
[^24]: LLM4Log：基于大语言模型的日志分析系统性综述

    LLM4Log: A Systematic Review of Large Language Model-based Log Analysis

    [https://arxiv.org/abs/2604.16359](https://arxiv.org/abs/2604.16359)

    本文对基于大语言模型的日志分析研究进行了系统性综述，覆盖从日志语句生成与维护、日志解析结构化，到异常检测、故障预测、根因分析和日志摘要的端到端流水线，并分析了LLM在此领域的优势与部署风险。

    

    软件系统会产生大规模、不断演化的半结构化日志，这些日志是可靠性工程和智能运维的核心，但在存在数据漂移和标注有限的情况下难以进行规模化分析。预训练Transformer模型和指令微调大语言模型（LLM）的最新进展，通过实现语义泛化和跨源证据整合，重塑了日志分析领域，但同时也带来了部署风险，例如上下文长度限制、延迟和成本、隐私约束以及幻觉问题。本文提出了LLM4Log，这是一项针对基于LLM的日志分析的系统性综述，覆盖端到端流水线，包括上游的日志语句生成与维护、日志解析与结构化，以及下游任务，如异常检测、故障预测、根因分析和日志摘要。遵循结构化的文献搜索与人工筛选协议，我们于2025年11月完成了文献收集，并识别出14...

    arXiv:2604.16359v3 Announce Type: replace  Abstract: Software systems generate massive, evolving, semi-structured logs that are central to reliability engineering and AIOps, yet difficult to analyze at scale under drift and limited labels. Recent advances in pretrained Transformer models and instruction-tuned large language models (LLMs) have reshaped log analysis by enabling semantic generalization and cross-source evidence integration, but also introducing deployment risks such as context limits, latency and cost, privacy constraints, and hallucinations. This paper presents LLM4Log, a systematic review of LLM-based log analysis across the end-to-end pipeline, from upstream logging-statement generation and maintenance to log parsing/structuring and downstream tasks including anomaly detection, failure prediction, root cause analysis, and log summarization. Following a structured search and manual screening protocol, we completed literature collection in November 2025 and identified 14
    
[^25]: JavaScript项目中依赖项重新分类的纵向研究

    A Longitudinal Study of Dependency Reclassifications in JavaScript Projects

    [https://arxiv.org/abs/2604.08747](https://arxiv.org/abs/2604.08747)

    本研究通过对33,087个JavaScript项目package.json文件的提交级分析，首次系统揭示了依赖项重新分类（包括移除和角色重分配）是一种普遍的维护活动，存在于79.1%的项目中，占所有依赖项维护提交的19.4%。

    

    现代软件项目依赖第三方依赖项，随着项目的演进，这些依赖项的声明必须得到持续维护。以往的研究主要关注依赖项的版本更新，而对于开发者如何随时间推移将依赖项分配到不同角色的研究则相对匮乏。在本文中，我们研究了JavaScript项目的开发者如何对其依赖项进行重新分类，包括移除依赖项和重新分配其角色。通过分析package.json文件在提交级别的修改，我们重建了依赖项角色变更历史，并识别出反复出现的重新分类实践。我们对33,087个积极维护依赖项的JavaScript项目的分析表明，依赖项重新分类是一种普遍存在的维护活动，出现在79.1%的所研究项目中，占所有依赖项维护提交的19.4%。在这些项目中，几乎所有项目（97.2%）都会在某些时候移除依赖项，而38.0%的项目经历了依赖项角色的重新分配。

    arXiv:2604.08747v2 Announce Type: replace  Abstract: Modern software projects depend on third-party dependencies, whose declarations must be maintained as projects evolve. Prior work has focused on dependency version updates, while much less is known about how developers assign dependencies to different roles over time. In this paper, we investigate how developers of JavaScript projects reclassify their dependencies, including removal and role reassignment. By analyzing commit-level modifications to package.json files, we reconstruct dependency role histories and identify recurring reclassification practices. Our analysis of 33,087 JavaScript projects with active dependency maintenance reveals that dependency reclassification is a prevalent maintenance activity, occurring in 79.1% of the studied projects, and accounting for 19.4% of all dependency-maintenance commits. Of these projects, nearly all (97.2%) remove dependencies at some point, while 38.0% undergo role reassignments across 
    
[^26]: 一种用于量子误差归因的物理信息神经模糊框架

    A Physics-Informed Neuro-Fuzzy Framework for Quantum Error Attribution

    [https://arxiv.org/abs/2602.21253](https://arxiv.org/abs/2602.21253)

    该论文提出一种结合ANFIS与物理特征工程的神经模糊框架，通过Bhattacharyya否决这一硬物理约束在IBM 156量子比特处理器上以89.5%的准确率有效区分量子计算中的软件缺陷与随机硬件噪声。

    

    随着量子处理器的规模扩展到100量子比特以上，区分软件缺陷与随机硬件噪声成为一项关键的诊断挑战。我们提出了一种神经模糊框架，通过将自适应神经模糊推理系统（ANFIS）与基于物理的特征工程相结合来解决这一归因问题。我们引入了Bhattacharyya否决，这是一种基于设备经验表征的弱噪声底层的硬物理约束，并通过CPTP收缩保证单侧可靠性，防止分类器将拓扑上不可能出现的输出分布归因于噪声。该框架在IBM的156量子比特Heron r2处理器（ibm_fez）上进行了验证，涵盖17个算法系列的105个电路，实现了89.5%的有效准确率（±5.9%置信区间）。该系统实现了安全故障模式，将14.3%的模糊案例标记为需人工审查，而不是强制给出低置信度的预测。

    arXiv:2602.21253v3 Announce Type: replace-cross  Abstract: As quantum processors scale beyond 100 qubits, distinguishing software bugs from stochastic hardware noise becomes a critical diagnostic challenge. We present a neuro-fuzzy framework that addresses this attribution problem by combining Adaptive Neuro-Fuzzy Inference Systems (ANFIS) with physics-grounded feature engineering. We introduce the Bhattacharyya Veto, a hard physical constraint grounded in the empirically characterized weak-noise floor of the device, with CPTP contraction guaranteeing one-sided soundness that prevents the classifier from attributing topologically impossible output distributions to noise. Validated on IBM's 156-qubit Heron r2 processor (ibm_fez) across 105 circuits spanning 17 algorithm families, the framework achieves 89.5% effective accuracy (+/- 5.9% CI). The system implements a safe failure mode, flagging 14.3% of ambiguous cases for manual review rather than forcing low-confidence predictions. We r
    
[^27]: 检测纠缠代码提交中的多个语义关注点

    Detecting Multiple Semantic Concerns in Tangled Code Commits

    [https://arxiv.org/abs/2601.21298](https://arxiv.org/abs/2601.21298)

    本文首次将纠缠代码提交中的多关注点检测建模为多标签分类问题，并基于真实数据构建了人工纠缠提交的受控数据集，填补了使用语言模型检测提交中多个语义关注点这一研究空白。

    

    代码提交在版本控制系统（如 Git）中应当是原子性的，即专注于单一目标，例如添加功能或修复缺陷。然而在实践中，开发者常常将多个关注点捆绑在纠缠的提交中，从而掩盖了提交意图并使维护变得复杂。近期研究已使用约定式提交规范和语言模型来捕获提交意图，证明了小型语言模型（SLM）在本地基础设施中保持效率与隐私的同时，其性能可以接近大型语言模型（LLM）。然而，这些研究并未解决涉及多个关注点的纠缠提交问题，使得使用语言模型进行多关注点检测的可行性悬而未决。本文将纠缠提交中的多关注点检测构建为一个多标签分类问题，并基于真实世界数据构建了一个受控的人工纠缠提交数据集。随后我们提出了一种……（摘要原文在此处被截断）

    arXiv:2601.21298v2 Announce Type: replace  Abstract: Code commits in a version control system (e.g., Git) should be atomic, i.e., focused on a single goal, such as adding a feature or fixing a bug. In practice, however, developers often bundle multiple concerns into tangled commits, obscuring intent and complicating maintenance. Recent studies have used Conventional Commits Specification (CCS) and Language Models (LMs) to capture commit intent, demonstrating that Small Language Models (SLMs) can approach the performance of Large Language Models (LLMs) while maintaining efficiency and privacy within local infrastructure. However, they do not address tangled commits involving multiple concerns, leaving the feasibility of using LMs for multi-concern detection unresolved. In this paper, we frame multi-concern detection in tangled commits as a multi-label classification problem and construct a controlled dataset of artificially tangled commits based on real-world data. We then present an em
    
[^28]: AI队友时代的安全性：GitHub上智能体式拉取请求的实证研究

    Security in the Age of AI Teammates: An Empirical Study of Agentic Pull Requests on GitHub

    [https://arxiv.org/abs/2601.00477](https://arxiv.org/abs/2601.00477)

    本研究基于包含33,000余个拉取请求的AIDev数据集，对GitHub上自主编码智能体提交的安全相关PR进行了大规模实证分析，系统刻画了AI智能体在软件安全方面的贡献模式、审查与接受情况以及导致PR被拒绝的关键信号。

    

    自主编码智能体正日益作为AI队友被部署在现代软件工程中，独立编写大规模修改生产代码的拉取请求（PR）。本研究旨在系统性地刻画自主编码智能体在实践中如何为软件安全做出贡献、这些与安全相关的贡献如何被审查和接受，以及哪些可观察信号与PR被拒绝相关。我们使用AIDev数据集对智能体编写的PR进行了大规模实证分析，该数据集包含来自热门GitHub仓库的超过33,000个精心整理的PR。我们采用关键词过滤策略识别与安全相关的PR，随后进行人工验证，最终确认了1,293个与安全相关的智能体PR。随后，我们分析了不同自主智能体、编程生态系统和代码变更类型之间的安全相关PR的普遍性、接受结果和审查延迟。此外，我们还应用了定性开放编码方法（原文此处被截断）。

    arXiv:2601.00477v2 Announce Type: replace-cross  Abstract: Autonomous coding agents are increasingly deployed as AI teammates in modern software engineering, independently authoring pull requests (PRs) that modify production code at scale. This study aims to systematically characterize how autonomous coding agents contribute to software security in practice, how these security-related contributions are reviewed and accepted, and which observable signals are associated with PR rejection. We conduct a large-scale empirical analysis of agent-authored PRs using the AIDev dataset, comprising of over 33,000 curated PRs from popular GitHub repositories. Security-relevant PRs are identified using a keyword filtering strategy, followed by manual validation, resulting in 1,293 confirmed security-related agentic-PRs. We then analyze prevalence, acceptance outcomes, and review latency across autonomous agents, programming ecosystems, and types of code changes. Moreover, we apply qualitative open c
    
[^29]: 进化卓越：基于大语言模型的智能体自动化优化

    Evolving Excellence: Automated Optimization of LLM-based Agents

    [https://arxiv.org/abs/2512.09108](https://arxiv.org/abs/2512.09108)

    本文提出ARTEMIS，一个无代码的进化优化平台，通过语义感知的遗传算子自动联合优化LLM智能体的提示词、工具描述和参数等配置，无需架构修改即可显著提升智能体性能。

    

    基于大语言模型（LLM）构建的智能体AI系统在自动化复杂工作流程方面具有巨大潜力，涵盖从软件开发到客户支持等应用场景。然而，LLM智能体常常因配置不佳而表现欠佳——调优不当的提示词、工具描述和参数通常需要数周的手工打磨。现有的优化方法要么过于复杂而难以通用，要么孤立地处理各个组件，忽略了组件之间关键的相互依赖关系。我们提出了ARTEMIS，一个无代码的进化优化平台，通过语义感知的遗传算子对智能体配置进行联合优化。只需提供一个基准测试脚本和自然语言目标，ARTEMIS即可自动发现可配置组件、从执行日志中提取性能信号，并在无需修改架构的情况下演化配置。我们在四个代表性的智能体系统上对ARTEMIS进行了评估。

    arXiv:2512.09108v2 Announce Type: replace-cross  Abstract: Agentic AI systems built on large language models (LLMs) offer significant potential for automating complex workflows, from software development to customer support. However, LLM agents often underperform due to suboptimal configurations; poorly tuned prompts, tool descriptions, and parameters that typically require weeks of manual refinement. Existing optimization methods either are too complex for general use or treat components in isolation, missing critical interdependencies.   We present ARTEMIS, a no-code evolutionary optimization platform that jointly optimizes agent configurations through semantically-aware genetic operators. Given only a benchmark script and natural language goals, ARTEMIS automatically discovers configurable components, extracts performance signals from execution logs, and evolves configurations without requiring architectural modifications.   We evaluate ARTEMIS on four representative agent systems: 
    
[^30]: 通过指令扩展中间填充以实现可控的代码补全

    Extending Fill-In-the-Middle with Instructions for Steerable Code Completion

    [https://arxiv.org/abs/2509.24637](https://arxiv.org/abs/2509.24637)

    提出指令感知中间填充（IFIM）微调方法，通过在FIM结构中引入结构分离的专门指令部分，使代码补全模型能够有效理解并优先处理开发者的自然语言指令，从而实现可控的代码补全。

    

    当开发者的意图在代码上下文中未被充分说明时，代码补全模型往往会失败。为了缓解这一问题，开发者经常使用自然语言注释来阐明目标。然而，当前的代码补全模型无法有效地优先处理这些指令，因为它们仅仅通过中间填充目标进行预训练。一方面，与嘈杂的代码注释混杂在一起的自然语言指令，仅仅被视为前缀中背景上下文的一部分。另一方面，用于FIM目标的预训练数据集大多来源于开源代码库，这导致缺乏能够反映开发者在代码补全工作流程中的高意图指令到代码的配对数据。为了弥合这一差距，我们提出了指令感知的中间填充（IFIM），这是一种微调方法，它通过一个专门的、结构上分离的指令来扩展FIM结构。

    arXiv:2509.24637v3 Announce Type: replace  Abstract: Code completion models often fail when the developer's intent is under-specified in the code context. To mitigate this, developers frequently use natural language comments to clarify objectives. However, current code completion models fail to prioritize these directives effectively since they are merely pre-trained using the Fill-In-the-Middle (FIM) objective. On the one hand, the natural language instructions, mixed with the noisy code comments, are just treated as part of the background context within the prefix. On the other hand, the pre-training datasets for the FIM objective are mostly sourced from open-source repositories, which results in a scarcity of high-intent instruction-to-code pairings that reflect the developers' workflow in code completion. To bridge this gap, we propose Instruction-aware Fill-In-the-Middle (IFIM), a fine-tuning method that extends the FIM structure with a dedicated, structurally separated instructio
    
[^31]: SWIRL：通过定制化摘要实现工具生成警告的交互式意义建构

    SWIRL: Interactive Sensemaking of Tool-Generated Warnings through Customized Summaries

    [https://arxiv.org/abs/2508.07169](https://arxiv.org/abs/2508.07169)

    SWIRL通过交互式定制摘要和用户反馈动态推断分组规则，帮助程序员更高效地理解和归类静态分析工具生成的大量警告。

    

    使用错误查找工具的程序员通常需要逐个审查工具报告的警告。基于识别重复出现的主题和关系可以增强在给定问题空间中搜索表征的认知过程（即意义建构）这一洞察，我们提出了SWIRL，它通过交互式、定制化的摘要来支持对工具生成的警告进行解释。通过主动反馈，SWIRL能够动态推导出用于对相关警告进行分组的摘要规则。当用户将警告标记为有趣或不有趣时，SWIRL的规则推断算法会挖掘出共同特征，突出显示在包含关系、子类型、被调用方法、被访问字段和表达式等方面的结构相似性。我们在两个成熟的Java项目中，使用Infer和SpotBugs生成的真实世界警告对SWIRL进行了演示。在一项被试内用户研究中，参与者能够更好地阐明类似的不有趣警告的根本原因。

    arXiv:2508.07169v2 Announce Type: replace  Abstract: Programmers using bug-finding tools often review their reported warnings one by one. Based on the insight that identifying recurring themes and relationships can enhance the cognitive process of searching for representations of a given problem space (i.e., sensemaking), we propose SWIRL, which supports interpreting tool-generated warnings through interactive, customized summarization. With active feedback, SWIRL derives summary rules for grouping of related warnings on the fly. As users mark warnings as interesting or uninteresting, SWIRL's rule inference algorithm surfaces common characteristics, highlighting structural similarities in containment, subtyping, invoked methods, accessed fields, and expressions.   We demonstrate SWIRL on real-world warnings generated from Infer and SpotBugs on two mature Java projects. In a within-subject user study, our participants articulated root causes for similar uninteresting warnings with more 
    

