# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Runtime BiDi Structural Verification Catches Right-to-Left Layout Defects that Accessibility, Actionability, and Logical-Property Linting Miss](https://arxiv.org/abs/2610.32934) | 该论文提出运行时结构不变量方法（零基准模式与双向保守差分模式），可在无头浏览器中检测出现有无障碍、可操作性和逻辑属性检查工具无法发现的RTL布局镜像缺陷，在真实基准上实现13/14的召回率且零误报。 |
| [^2] | [Green Code or Costly Code? Evaluating the Energy Costs of Iterative LLM-Based Code Optimizations](https://arxiv.org/abs/2610.32891) | 该研究通过对11个开源模型在EffiBench全部1000个问题上的硬件级能耗实测，挑战了“代码跑得更快就一定更省电”的假设，发现LLM迭代优化代码的推理能耗（0.79–8.68 MJ，相当于250–2800次GPT-4.1查询）未必能被后续执行时间的节省所抵消。 |
| [^3] | [Trusting the Tool: Asymmetric Deference of LLM Vulnerability Auditors to Static-Analysis Evidence](https://arxiv.org/abs/2610.32876) | 本研究首次量化了 LLM 漏洞审计员对静态分析证据的盲从性：真实证据几乎没有提升审计准确率，而工具的虚假“保证”却能使真实漏洞的检出率下降多达 19 个百分点，揭示了模型对工具证据的非对称依赖。 |
| [^4] | [OpenDraco: A Platform for Engineering and Experimenting with Configurable Multi-Agent Architectures for Software Engineering](https://arxiv.org/abs/2610.32686) | OpenDraco是一个开源平台，将基于LLM的多智能体系统的配置建模、执行、评估与改进整合到统一工作流中，并通过自动程序修复实验揭示了不同多智能体架构配置之间的显著成本差异。 |
| [^5] | [CC4Q Workbench: From Quantum Repositories to Analyzable Code Comments](https://arxiv.org/abs/2610.32651) | 本文提出了开源工作台 CC4Q Workbench，它能自动从量子 SDK 的 Python 仓库中提取代码-注释对、将注释切分为句子级单元，并通过训练好的分类器识别量子特定内容及其知识类别，把分散的注释研究步骤整合为可复用的工作流程。 |
| [^6] | [StateMirror: Pre-Execution Detection of Hallucinated Tool Calls in Coding Agents](https://arxiv.org/abs/2610.32648) | 提出StateMirror，一个轻量级执行前检测器，通过在工具框架边界拦截操作、利用AST和命令分析提取状态谓词并对照实时代码库进行确定性验证，从而在执行之前检测出编码智能体的幻觉工具调用，避免了现有执行后反馈机制对运行时资源的浪费。 |
| [^7] | [Can AI Agents Help Robotic Engineers? Lessons from Two Focus Days in Industry](https://arxiv.org/abs/2610.32636) | 本研究通过对一家工业机器人编程中小企业的案例研究，发现AI智能体并非即插即用的解决方案——除了生成高度重复代码的任务外未能提升工程师效率，其使用效果随工程师的AI使用经验和智能体的领域适配而逐步改善，且AI使用经验比机器人工程经验对成功更为关键。 |
| [^8] | [Automated Multi-Line Bug Repair Using Model Checker Aided LLMs: A Case Study on RISC-V Processors](https://arxiv.org/abs/2610.32597) | 本文提出了一种结合大语言模型、断言和模型检测器的自动化RTL错误修复方法，利用失败断言引导LLM生成多行错误补丁并提供形式化保证，通过RISC-V处理器的案例研究验证了其有效性。 |
| [^9] | [Lessons Outlive Their Failures: A Case Study of Rules for AI Coding Agents](https://arxiv.org/abs/2610.32578) | 本研究通过对个人Markdown仓库进行六周案例研究，建立118次AI编码代理失败台账、验证LLM文档审计并重放历史失败，发现文档漂移是最大的失败类别，而沉淀为规则的教训能够比引发它们的失败更长久地发挥作用。 |
| [^10] | [From Regulation to Engineering: Organizational Privacy Climate as Mediator](https://arxiv.org/abs/2610.32576) | 该研究通过对156名美国信息系统开发者的调查发现，组织隐私氛围是连接法规与实践的关键中介——开发者对GDPR重要性的认知通过塑造组织隐私氛围影响其隐私态度，进而驱动其隐私工程实践。 |
| [^11] | [Tool-Calling LLM Agents Under Realistic MCP Faults: Scale Does Not Purchase Fault Tolerance](https://arxiv.org/abs/2610.32555) | 提出基于真实MCP生态故障数据校准的故障注入中间件MCP-Injector，实验证明扩大LLM模型规模并不能提升智能体应对工具故障的容错能力。 |
| [^12] | [Beyond Coverage and Mutation: A Blind-Spot Analysis of LLM-Generated Unit Tests](https://arxiv.org/abs/2610.32548) | 该论文提出了基于行为义务的单元测试评估框架BlindSpotEval和配套Java基准ObligBench，通过边界值充分性、控制流充分性和断言强度三个指标揭示了大语言模型生成单元测试的系统性盲点，发现错误断言占测试失败的15.6%，证明覆盖率无法保证断言正确性。 |
| [^13] | [When Agents Fail: Model or Harness--Who's to Blame?](https://arxiv.org/abs/2610.32483) | 本文提出了一个通过五种互补指标（有效完成率、模型-框架匹配度、跨模型匹配度、交互残差和Token负载）来评估LLM智能体执行框架与底层模型交互表现的方法，用以区分智能体失败时究竟应归咎于模型还是执行框架。 |
| [^14] | [WaitLens: Temporal VLM-Based Adaptive Waiting for Industrial GUI Testing Agents](https://arxiv.org/abs/2610.32482) | WaitLens通过在标注等待结束时间的GUI加载轨迹上微调轻量级视觉语言模型，并引入时间重采样构建多样化训练数据，实现了适用于多种工业环境的自适应等待策略，显著减少了占交互时间高达42.3%的固定等待开销。 |
| [^15] | [OPTARI: An open-source software framework for clinical translation of optoacoustic imaging](https://arxiv.org/abs/2610.32468) | OPTARI是一个基于napari和PATATO构建的开源图形化软件，使无编程背景的研究人员能够完成大规模临床光声-超声研究的全流程分析，涵盖图像重建、感兴趣区标注、发色团解混、深度学习组织分割以及时间与光谱分析。 |
| [^16] | [Complexity concentration as a god-class and god-method detector: a pre-registered test of a scale-free outlier rule against tool definitions and human labels](https://arxiv.org/abs/2610.32466) | 本文预注册检验了一种不依赖原始尺度阈值的上帝类/上帝方法检测新规则——即当单元复杂度在其自身代码库中构成稳健离群点（修正z分数≥3.5）时判定为“上帝单元”，但在445个多语言代码库与492个Java代码库上与PMD及MLCQ等现有定义对比后，结论尚不确定。 |
| [^17] | [Search-Based Testing of Modality Sensitivity in Vision Language Models](https://arxiv.org/abs/2610.32360) | 本文提出黑盒测试框架MoST，通过多目标搜索系统对比图像、文本及联合扰动，首次揭示视觉语言模型对不同模态扰动的敏感性存在显著差异。 |
| [^18] | [DeepAVR: Deep Analysis via Multi-Agent Collaboration for Automated Vulnerability Repair](https://arxiv.org/abs/2610.32345) | DeepAVR提出了一种多智能体协作框架，将中间修复结果转化为结构化诊断反馈，并在故障定位、补丁生成、编译/安全/功能验证与补丁审查等专门角色之间流转反馈，从而实现端到端的深度漏洞分析与自动化修复。 |
| [^19] | [From Faulty Claims to Incorrect Patches: An Empirical Study of Task-Grounded Reasoning in Coding Agents](https://arxiv.org/abs/2610.32338) | 该论文开展了首个针对编码智能体“任务接地推理”的实证研究，通过分析四个开源推理模型在 SWE-bench Pro 上的 400 条轨迹，量化其思维链中可核查论断里的错误论断，并检验错误论断是否会导致错误补丁。 |
| [^20] | [Large-Scale Study of GitHub Copilot Trajectories](https://arxiv.org/abs/2610.32313) | 该研究通过分析 7,581 个真实 GitHub Copilot 智能体 PR 的人在回路轨迹，揭示了真实开发中人与 AI 智能体的协作模式——大多数 PR 无人类干预且协作短暂，而首次人类干预会显著缩短后续轨迹，弥补了现有基准测试假设与真实使用场景之间的差距。 |
| [^21] | [Pairing Matters: Performance, Failures, and Efficiency in Lightweight Coding Agents](https://arxiv.org/abs/2610.32131) | 本研究通过对六个开源代理框架与三个轻量级编码模型在SWE-bench Verified任务上进行1,800次运行的全因子受控评估，首次系统揭示了框架-模型配对方式对编码代理性能、失败模式和效率的显著影响。 |
| [^22] | [Shorter Is Not Cheaper: A Controlled Comparison of Declarative Animation Specifications and Free-Form Code for LLM-Generated Explanations](https://arxiv.org/abs/2610.32129) | 本研究通过预先冻结假设与分析计划的受控实验发现，尽管声明式动画规范的输出比自由格式代码短25-48%，但“更短即更便宜、更可靠”的普遍预期并不成立。 |
| [^23] | [Beyond Correctness: How Novice Software Engineering Students Exercise Oversight of Generative AI](https://arxiv.org/abs/2610.32118) | 软件工程新手学生对生成式AI的监督主要聚焦于输出的适用性问题（74.4%）而非正确性问题（16.9%），表明学生监督的核心是判断看似合理的AI输出是否真正适合实际任务。 |
| [^24] | [Who Verifies the Tools? How Warped Formatters Mislead Agents and Their Verifiers](https://arxiv.org/abs/2610.32110) | 论文通过生产RAG系统的真实案例与对照实验发现，工具输出格式化器遗漏关键信息会误导智能体和验证器，而仅需在输出中补全这些缺失细节，就能显著减少智能体的无依据声称。 |
| [^25] | [CrowdStrike Channel File 291: An Engineering Analysis of a Global Endpoint Outage](https://arxiv.org/abs/2610.32078) | 本文通过工程分析重建了2024年7月 CrowdStrike Falcon 内容更新导致全球约850万台 Windows 设备崩溃事件的根本原因——检测模板声明21个输入而传感器代码仅提供20个，暴露了内核态解释器中未检查的数组访问——并识别出验证、测试、部署和恢复四个保障缺口。 |
| [^26] | [Are LLM-Generated Formal Specifications Faithful to Their Requirements? A Study on Bimodal Formal Specification Generation](https://arxiv.org/abs/2610.32053) | 本文针对程序正确性未知这一形式化验证的真实场景，开展了双模态（自然语言需求+代码实现）下LLM生成ACSL形式化规范是否忠实于需求的实证研究，构建了包含1,403个C程序的基准并用Frama-C进行验证评估。 |
| [^27] | [Permission to Retry: A Study of Authorization and Recovery in AI Agent Workflows](https://arxiv.org/abs/2610.31916) | 该论文将AI智能体的恢复决策与语言模型行为分离，提出一项在LangGraph和Temporal宿主上执行336条历史记录的有限一致性研究，验证了检查点状态、分发许可、外部效果证据和对账许可这四种独立义务的执行时可靠性。 |
| [^28] | [Knowing When to Stop: The Evidence-Action Gap in Coding Agents](https://arxiv.org/abs/2610.31895) | 该论文揭示了编码智能体的“证据-行动差距”现象——即智能体即使已收集到表明任务无法独立完成的充分证据仍过度坚持或虚报成功，并提出ImpasseBench基准通过在可解决任务中注入环境阻碍并标记证据充分步骤来量化这一差距。 |
| [^29] | [SWE-Clar: Can Coding Agents Clarify Incomplete Client Requirements and Deliver Greenfield Software?](https://arxiv.org/abs/2610.31854) | 该论文提出了 SWE-Clar 基准，基于 100 个报酬至少 4,000 美元的真实 Upwork 客户委托任务，通过“先澄清需求、后实现代码”的多轮交互设计并以明确的需求理解测试来评估编码智能体在自由职业全新软件开发中澄清不完整客户需求的能力。 |
| [^30] | [Ulysses: A Security Harness For Synthesizing Verifiable Web Exploits](https://arxiv.org/abs/2610.31831) | Ulysses是一个多智能体安全测试框架，通过迭代探索漏洞假设并结合独立观察的执行效果验证，将候选漏洞发现转化为可验证的可利用性证明，在25个真实Web应用中成功验证了10个已知漏洞并新发现4个此前未报道的漏洞。 |
| [^31] | [CuraLoc: Repository-Level Fault Localization via Planned Code Inspection](https://arxiv.org/abs/2610.31809) | CuraLoc提出在源码检索前一次性规划代码检查的缺陷定位方法：模型在单次规划中同时选定起始符号、检查关系并形成修改假设，再由确定性检索完成最终排序，从而减少反复的模型调用和token消耗。 |
| [^32] | [Autoregressive Multimodal Anomaly Detection with Large Language Models for Microservice Systems](https://arxiv.org/abs/2610.31722) | 该论文提出了LLMMAD，一个基于大语言模型的自回归多模态异常检测框架，通过在跨模态融合中保留模态特定特征并解决“身份捷径”问题，来提升微服务系统的异常检测效果。 |
| [^33] | [What Open-Source Projects Tell Their Coding Agents: A Census of AGENTS.md and CLAUDE.md](https://arxiv.org/abs/2610.31715) | 这项研究首次对十三种语言的活跃 GitHub 仓库中的 AGENTS.md 和 CLAUDE.md 指令文件进行大规模普查，系统揭示了开源项目向 AI 编码智能体传达的指令构成（项目知识与代码、构建、工作流规则的混合），以及各项目在 AI 署名披露等问题上的立场分歧。 |
| [^34] | [Reuse, Rewrite, or Reject: Evidence-Constrained Trace Reuse for Repeated Mobile GUI Control](https://arxiv.org/abs/2610.31712) | 该论文提出图-文本模板与“记录-模板-回放”框架，通过对UI状态转移进行证据约束验证，实现移动端GUI智能体在重复任务中对先前执行轨迹的选择性复用，从而降低冗余的LLM推理与执行开销。 |
| [^35] | [MBABench-2.0: Evaluating AI Agents on Constructing Financial Spreadsheets Beyond Simple Verifiability](https://arxiv.org/abs/2610.31711) | 该论文提出MBABench-2.0基准测试，包含101个从零构建财务电子表格的原创任务，突破了以往填空式简单可验证任务的局限，用于更真实地评估AI智能体完成建模决策、格式设计和表格布局等核心电子表格构建工作的能力。 |
| [^36] | [Compact-Bench: Measuring What Context Compaction Costs Long-Horizon Coding Agents](https://arxiv.org/abs/2610.31710) | 提出 Compact-Bench 基准，通过在共享仓库状态下配对“完整历史”与“仅摘要”的延续过程，首次将上下文压缩对长程编程智能体的具体损害（如抹去 5-64% 的成功修复、最多仅保留 16% 的重复细节）从效率指标中隔离出来进行量化评估。 |
| [^37] | [Mining the Design Space of Multi-Agent Workflows](https://arxiv.org/abs/2610.31709) | 提出一个两阶段流水线，通过挖掘28,568个可执行工作流的代码与执行轨迹，自动提炼出可复现的多智能体系统设计原则。 |
| [^38] | [RubricCraft: Where Solutions Diverge, Criteria Emerge](https://arxiv.org/abs/2610.31707) | RubricCraft 通过对比多个候选解法之间的满足与违反需求的差异来构建文本评分规则，从而为开放式任务提供显式且可靠的自动化评估标准。 |
| [^39] | [Before the Ticket: Grounding HPC User Support in Documentation and Resolved Tickets with an Agentic LLM Assistant](https://arxiv.org/abs/2610.31703) | 该论文提出了一个完全本地化部署的智能体式LLM助手，通过MCP协议编排实时文档RAG检索与在11,244条历史工单“问题-解决方案”对上QLoRA微调的Llama-3-8B模型并融合二者答案，从而在用户提交工单之前就解答其HPC使用问题。 |
| [^40] | [Understanding Tool-Mediated Interaction Failures in Agentic GUI Software: An Empirical Study](https://arxiv.org/abs/2610.31698) | 该研究构建了包含335个真实失败案例的Malva缺陷库，首次系统性地揭示了LLM智能体在通过工具介导与GUI交互时的11种失败模式，并提出MalvaGuard来检测这类广泛存在且往往静默发生的交互缺陷。 |
| [^41] | [SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges](https://arxiv.org/abs/2610.28886) | SANE是一个基于智能体的自动化框架，能针对特定LLM评判器自动识别其评分能力、生成变异测试配方及预期分数关系，从而将评判器可靠性测试实现全流程自动化，摆脱了人工编写扰动配方和预设评分器类型的限制。 |
| [^42] | [Interaction-as-a-Judge](https://arxiv.org/abs/2610.17895) | 提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。 |
| [^43] | [Improving the Energy-Efficiency of the Code Generated by LLMs through Effective Prompting](https://arxiv.org/abs/2610.02571) | 本研究系统评估了21种提示策略，发现有效的提示工程可使大语言模型生成的Python和C++代码能耗分别降低最多25%和17%。 |
| [^44] | [Two's a Crowd: Human and AI-Based Copresence for Developers with ADHD](https://arxiv.org/abs/2609.21254) | 本研究通过对14名ADHD软件工程师的半结构化访谈，首次系统考察了他们如何在人际共同在场实践（如body doubling、结对编程）与新兴代理型AI编程助手之间开展协作，揭示了AI助手正在重塑ADHD开发者的协作动态与执行功能支持方式。 |
| [^45] | [Loop Engineering: Building Blocks, Adoption, and Impact](https://arxiv.org/abs/2608.21884) | 本文首次探索性回顾了“循环工程”这一新兴实践，即开发者设计系统自动触发和停止智能体运行，并总结了其核心构建模块（如停止条件、状态文件和验证子智能体），但未量化其实际采用率。 |
| [^46] | [Memoir: Learning, Verifying, and Evolving False-Positive Memories for Static Application Security Testing Tools](https://arxiv.org/abs/2608.09181) | 提出Memoir框架，通过将SAST工具的历史误报警报转化为可学习、可验证并持续演化的结构化语义记忆，有效识别并减少误报，克服了现有方法难以跨工具泛化、知识无法动态更新的局限。 |
| [^47] | [A Reduced Stochastic Model of Microservice Availability: Algorithms for Identification and Formal Analysis](https://arxiv.org/abs/2607.00740) | 本文提出从遥测数据和声明的操作需求中识别简化随机可用性模型的算法，通过形式化分析给出可达到的概率界限，并证明该简化模型能够保留完整模型中对“完成-截止期限”事件的评估范围。 |
| [^48] | [From Research to Practice: An Interactive Rapid Review of Autonomous Driving System Testing in Industry](https://arxiv.org/abs/2605.00531) | 本研究通过与21名汽车行业从业者合作的交互式快速综述，识别出自动驾驶系统测试的12个关键挑战，重点聚焦端到端系统的测试方法与完备性，并评估了17项相关研究在工业实践中的适用性，是首个由从业者驱动的研究综述。 |
| [^49] | [daVinci-Env: Open SWE Environment Synthesis at Scale](https://arxiv.org/abs/2603.13023) | 提出了 OpenSWE——目前最大的完全开源透明的软件工程智能体训练框架，包含 45,320 个覆盖 1.28 万仓库的可执行 Docker 环境，并借助 64 节点集群上的多智能体合成流水线自动化构建。 |
| [^50] | [Contextual Memory Virtualisation: DAG-Based State Management and Structurally Lossless Trimming for LLM Agents](https://arxiv.org/abs/2602.22402) | 提出了上下文记忆虚拟化（CMV）系统，借鉴操作系统虚拟内存思想，将LLM会话历史建模为带快照、分支、裁剪原语的DAG以实现跨并行会话的上下文复用，并通过三遍结构无损裁剪算法在逐字保留所有对话内容的前提下将令牌开销平均降低20%、最高降低86%。 |
| [^51] | [The Influence of Code Smells in Efferent Neighbors on Class Stability](https://arxiv.org/abs/2602.12950) | 本研究基于152个热门GitHub Java项目一年的提交历史，首次揭示了类的输出邻居（即其所依赖的类）中存在的代码坏味道会通过涟漪效应影响该类自身的稳定性。 |

# 详细

[^1]: 运行时双向结构验证可捕获无障碍性、可操作性与逻辑属性代码检查所遗漏的从右到左布局缺陷

    Runtime BiDi Structural Verification Catches Right-to-Left Layout Defects that Accessibility, Actionability, and Logical-Property Linting Miss

    [https://arxiv.org/abs/2610.32934](https://arxiv.org/abs/2610.32934)

    该论文提出运行时结构不变量方法（零基准模式与双向保守差分模式），可在无头浏览器中检测出现有无障碍、可操作性和逻辑属性检查工具无法发现的RTL布局镜像缺陷，在真实基准上实现13/14的召回率且零误报。

    

    双向（从右到左，RTL）布局缺陷产生于使用物理方向CSS属性（如margin-left、border-left、text-align:left）编写的界面在dir="rtl"环境下渲染时未被正确镜像的情况。无障碍性引擎、可操作性引擎，甚至Stylelint当前的核心逻辑属性规则都无法定位这些缺陷。我们提出了运行时结构不变量方法——基于计算样式与几何信息的零基准模式，以及双向保守差分模式——可在无头浏览器中捕获这些缺陷。在一个由Bootstrap和Material Components Web中带日期的修复提交重建的14个真实RTL缺陷的留出基准测试中，检测器成功恢复了13/14个缺陷（2个通过零基准模式，11个通过差分模式），且在所有已修复版本上零误报；在实际提交的历史版本bootstrap.rtl.css上，它在2次尝试中捕获了1个。该方法在16个负向LTR孤岛测试用例上报告了0个误报，并在两种模式下……

    arXiv:2610.32934v1 Announce Type: new  Abstract: Bidirectional (right-to-left, RTL) layout defects arise when interfaces authored with physical directional CSS (margin-left, border-left, text-align:left) are rendered under dir="rtl" without being mirrored. Accessibility engines (axe-core), actionability engines (Playwright), and even Stylelint's current core logical-property rule fail to localise these defects. We present runtime structural invariants -- a zero-golden mode over computed style and geometry, and a bidi-conservative differential mode -- that catch them in a headless browser. On a held-out benchmark of 14 real RTL bugs reconstructed from dated fix commits in Bootstrap and Material Components Web, the detector recovers 13/14 (2 via zero-golden, 11 via the differential mode), with no false alarm on any patched version; on the actual committed historical bootstrap.rtl.css it catches 1 of 2 attempted. It reports 0 false positives on 16 negative LTR-island cases and, in both mo
    
[^2]: 绿色代码还是昂贵代码？评估基于大语言模型的迭代式代码优化的能源成本

    Green Code or Costly Code? Evaluating the Energy Costs of Iterative LLM-Based Code Optimizations

    [https://arxiv.org/abs/2610.32891](https://arxiv.org/abs/2610.32891)

    该研究通过对11个开源模型在EffiBench全部1000个问题上的硬件级能耗实测，挑战了“代码跑得更快就一定更省电”的假设，发现LLM迭代优化代码的推理能耗（0.79–8.68 MJ，相当于250–2800次GPT-4.1查询）未必能被后续执行时间的节省所抵消。

    

    大语言模型（LLM）正被越来越多地用于优化代码性能。这种做法通常假设执行速度的提升会自然转化为更低的总体能耗。我们通过实证研究对这一假设提出质疑：我们测量了由LLM自我改进流水线生成的性能优化代码，相对于标准的人类参考解决方案是否也具有能效优势，以及优化过程本身的能源成本能否被后续的执行时间节省所抵消。我们在EffiBench（一个最先进的性能基准）的全部1,000个问题上，评估了11个开源模型在四种解码配置下的表现，并全程收集了逐进程的硬件级能耗测量数据。推理能耗在不同模型与配置组合之间跨越0.79至8.68兆焦耳（MJ），大致相当于250至2,800次GPT-4.1查询的能耗。限制最大输出token数量具有最大的……

    arXiv:2610.32891v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly used to optimize code for performance. This practice often assumes that improvements in execution speed naturally translate into overall lower energy consumption. We challenge this assumption empirically by measuring whether performance-optimized code produced by LLM self-refinement pipelines is also energy-efficient relative to canonical human reference solutions, and whether the energy cost of the optimization process can be offset by subsequent execution-time savings. We evaluate 11 open-source models across four decoding configurations on all 1,000 problems of EffiBench, a state-of-the-art performance benchmark, while collecting per-process, hardware-level energy measurements throughout. Inference energy spans 0.79 to 8.68 MJ across model-variant pairs, which is roughly equivalent to the energy of 250 to 2800 GPT-4.1 queries. Limiting the maximum number of output tokens has the largest 
    
[^3]: 信任工具：大语言模型漏洞审计员对静态分析证据的非对称盲从

    Trusting the Tool: Asymmetric Deference of LLM Vulnerability Auditors to Static-Analysis Evidence

    [https://arxiv.org/abs/2610.32876](https://arxiv.org/abs/2610.32876)

    本研究首次量化了 LLM 漏洞审计员对静态分析证据的盲从性：真实证据几乎没有提升审计准确率，而工具的虚假“保证”却能使真实漏洞的检出率下降多达 19 个百分点，揭示了模型对工具证据的非对称依赖。

    

    将大语言模型（LLM）与静态分析相结合的流水线允许模型将分析器的输出作为证据使用，但模型能否区分真实与虚假证据，以及其审查程度是否取决于证据所声称的内容，此前尚未被测量。我们构建了一个用于 C/C++ 的过程内分析器，它会生成证明义务，并在六种仅注释不同的证据条件下，对 PrimeVul 数据集中存在漏洞的函数与已修补的函数进行审计，其中包括虚假保证（谎称一个缺失的检查已存在）和虚假警报（谎称一个新增的检查缺失）。该研究涵盖四组互不相交集合中的 916 对函数、来自四家供应商的 11 种模型配置以及共 16,572 次审计；没有任何模型被训练或微调。真实证据几乎未能提升准确率（仅 +0.02）。虚假证据则表现出非对称性：相对于关于相同操作的真实注释，虚假保证使真实漏洞的检出率下降多达 19 个百分点。

    arXiv:2610.32876v1 Announce Type: new  Abstract: Pipelines that combine large language models (LLMs) with static analysis let the model consume the analyser's output as evidence, but whether the model distinguishes true from false evidence, and whether its scrutiny depends on what the evidence claims, has not been measured. We built an intraprocedural analyser for C/C++ that produces proof obligations and audited vulnerable and patched functions from PrimeVul under six evidence conditions that differ only in the notes, including fabricated assurances that a missing check exists and fabricated alarms that an added check is missing. The study covers 916 pairs in four disjoint sets, eleven model configurations from four vendors and 16,572 audits; no model is trained or fine-tuned. True evidence barely improved accuracy (+0.02). False evidence acted asymmetrically: relative to true notes about the same operations, false assurances lowered the detection of real vulnerabilities by up to 19 p
    
[^4]: OpenDraco：一个用于构建和实验面向软件工程的可配置多智能体架构的平台

    OpenDraco: A Platform for Engineering and Experimenting with Configurable Multi-Agent Architectures for Software Engineering

    [https://arxiv.org/abs/2610.32686](https://arxiv.org/abs/2610.32686)

    OpenDraco是一个开源平台，将基于LLM的多智能体系统的配置建模、执行、评估与改进整合到统一工作流中，并通过自动程序修复实验揭示了不同多智能体架构配置之间的显著成本差异。

    

    对基于大语言模型（LLM）的多智能体系统（MAS）进行实验涉及一个迭代循环：建模配置、执行与评估、比较结果以及改进系统设计。现有方法对这一实验生命周期提供的端到端支持十分有限。我们提出了OpenDraco，一个开源平台，将这些活动整合到单一工作流中。OpenDraco以可导出、可版本化的JSON规范来表示智能体、提示词、工具、模型、参数和通信拓扑，并将其转换为支持不同交互结构的可执行LangGraph工作流。用户可以使用本地或云端LLM来修改和重新评估配置。我们将OpenDraco实例化用于自动程序修复（APR），并在合成及真实世界的SWE-bench Lite问题上评估了一个顺序基线配置和七个重构配置。结果揭示了不同配置之间存在显著的成本差异……

    arXiv:2610.32686v1 Announce Type: new  Abstract: Experimenting with LLM-based multi-agent systems (MAS) involves an iterative cycle of modeling configurations, executing and evaluating them, comparing results, and refining system designs. Existing approaches provide limited end-to-end support for this experimentation lifecycle. We present OpenDraco, an open-source platform that integrates these activities into a single workflow. OpenDraco represents agents, prompts, tools, models, parameters, and communication topology in exportable, versioned JSON specifications, translated into executable LangGraph workflows supporting different interaction structures. Users can modify and re-evaluate configurations using local or cloud-based LLMs. We instantiate OpenDraco for automated program repair (APR) and evaluate a sequential baseline and seven reconstructed configurations on synthetic and real-world SWE-bench Lite issues. The results reveal substantial cost differences among configurations wi
    
[^5]: CC4Q 工作台：从量子代码仓库到可分析的代码注释

    CC4Q Workbench: From Quantum Repositories to Analyzable Code Comments

    [https://arxiv.org/abs/2610.32651](https://arxiv.org/abs/2610.32651)

    本文提出了开源工作台 CC4Q Workbench，它能自动从量子 SDK 的 Python 仓库中提取代码-注释对、将注释切分为句子级单元，并通过训练好的分类器识别量子特定内容及其知识类别，把分散的注释研究步骤整合为可复用的工作流程。

    

    量子软件开发工具包（SDK）提供了高级 API，其使用需要理解底层数学原理、电路语义和硬件特性。代码注释能够提供 API 调用中未明确表达的自然语言或结构化解释。然而，研究新量子 SDK 中的注释仍需要研究者协调注释提取、量子特定内容分类和人工验证等相互独立的步骤，导致重复的集成工作。为解决这一问题，我们提出了 CC4Q Workbench，这是一个开源工作台，将这些步骤整合为可复用的工作流程。给定指定版本的 Python 仓库，CC4Q Workbench 会自动提取代码-注释对，将注释切分为句子级单元，并应用两个经过训练的分类器来识别量子特定内容并分配量子知识类别。其网页界面支持探索性（分析）……（原文摘要在此处截断）

    arXiv:2610.32651v1 Announce Type: new  Abstract: Quantum software development kits (SDKs) expose high-level APIs whose use requires an understanding of underlying mathematical principles, circuit semantics, and hardware properties. Code comments can provide natural-language or structural explanations that are not explicit in API calls. However, studying comments in a new quantum SDK still requires researchers to coordinate separate steps for comment extraction, quantum-specific classification, and manual verification, resulting in repeated integration effort. To address this problem, we present CC4Q Workbench, an open-source workbench that integrates these steps into a reusable workflow. Given a Python repository at a specified revision, CC4Q Workbench automatically extracts code-comment pairs, segments comments into sentence-level units, and applies two trained classifiers to identify quantum-specific content and assign quantum-knowledge categories. Its web interface supports explorat
    
[^6]: StateMirror：编码智能体中幻觉工具调用的执行前检测

    StateMirror: Pre-Execution Detection of Hallucinated Tool Calls in Coding Agents

    [https://arxiv.org/abs/2610.32648](https://arxiv.org/abs/2610.32648)

    提出StateMirror，一个轻量级执行前检测器，通过在工具框架边界拦截操作、利用AST和命令分析提取状态谓词并对照实时代码库进行确定性验证，从而在执行之前检测出编码智能体的幻觉工具调用，避免了现有执行后反馈机制对运行时资源的浪费。

    

    基于大语言模型（LLM）的编码智能体可能会提出在执行环境中不适用的无效操作，例如访问不存在的路径或缺失的软件包。我们将这种不匹配称为“幻觉工具使用”（hallucinated tool use）。一项针对4,398条智能体轨迹的初步研究表明，这类幻觉现象普遍存在且代价高昂，恢复它们平均需要3次额外的交互步骤和22.4K个token。现有框架依赖于被动式的执行后反馈机制，浪费了运行时资源。然而，由于智能体的执行环境建立在具体的代码库和工具框架（harness）组件之上，许多操作的前提条件可以在执行之前被提取并加以验证。为此，我们提出了StateMirror，一个轻量级的执行前检测器。StateMirror在工具框架边界拦截操作，通过AST（抽象语法树）和命令分析提取状态谓词，并针对实时代码库对其进行确定性验证。

    arXiv:2610.32648v1 Announce Type: new  Abstract: Large language model (LLM)-based coding agents may propose invalid actions that are not applicable in the execution environment, such as accessing non-existent paths or missing packages. We term this mismatch hallucinated tool use. A preliminary study of 4,398 agent trajectories reveals that these hallucinations are pervasive and costly, requiring a median of 3 additional interaction steps and 22.4K tokens to recover. Existing frameworks rely on reactive, post-execution feedback, wasting runtime resources. Nevertheless, since an agent's execution environment is grounded in concrete codebase and harness components, many action prerequisites can be extracted and validated prior to execution. To this end, we introduce StateMirror, a lightweight pre-execution detector. StateMirror intercepts actions at the harness boundary, extracts state predicates via AST and command analysis, and deterministically validates them against the live repositor
    
[^7]: AI智能体能帮助机器人工程师吗？来自工业界两个专注日的经验教训

    Can AI Agents Help Robotic Engineers? Lessons from Two Focus Days in Industry

    [https://arxiv.org/abs/2610.32636](https://arxiv.org/abs/2610.32636)

    本研究通过对一家工业机器人编程中小企业的案例研究，发现AI智能体并非即插即用的解决方案——除了生成高度重复代码的任务外未能提升工程师效率，其使用效果随工程师的AI使用经验和智能体的领域适配而逐步改善，且AI使用经验比机器人工程经验对成功更为关键。

    

    软件开发人员越来越多地使用能够自主执行多步骤编程任务的AI智能体。然而，尚不清楚这种情况是否以及如何适用于机器人工程领域，在该领域中，工程师使用小众的、供应商专有的语言对工业机器人进行编程，并依赖仿真工具的视觉反馈。本文报告了一项关于一家工业机器人编程中小企业使用AI智能体的案例研究。我们的结果表明，除了一项需要生成高度重复代码的任务之外，AI智能体并未提高工程师的工作效率。然而，随着工程师逐渐学会如何使用该智能体，以及该智能体被适配到该领域，使用智能体的效果随时间推移而改善。此外，AI使用经验对成功使用智能体的影响比机器人工程经验更大。这表明AI智能体并非即插即用的解决方案，企业必须对其工程师进行培训（摘要在此处截断）。

    arXiv:2610.32636v1 Announce Type: cross  Abstract: Software developers increasingly use AI agents that execute multi-step programming tasks on their own. However, it is unclear if and how this transfers to robotic engineering, where engineers program industrial robots in niche, vendor-specific languages and rely on visual feedback from simulation tools. Within this paper, we report a case study on the use of AI agents by an SME that programs industrial robots. Our results show that the agent did not make the engineers more productive, with the exception of a task that required the generation of highly repetitive code. However, the results with the agent improved over time, as the engineers learned how to use it and the agent was adapted to the domain. Moreover, the experience with AI mattered more for the success with the agent than the experience with robotic engineering. This shows that AI agents are not a plug-and-play solution. Instead, companies must train their engineers on the u
    
[^8]: 使用模型检测器辅助大语言模型的自动化多行错误修复：以RISC-V处理器为案例研究

    Automated Multi-Line Bug Repair Using Model Checker Aided LLMs: A Case Study on RISC-V Processors

    [https://arxiv.org/abs/2610.32597](https://arxiv.org/abs/2610.32597)

    本文提出了一种结合大语言模型、断言和模型检测器的自动化RTL错误修复方法，利用失败断言引导LLM生成多行错误补丁并提供形式化保证，通过RISC-V处理器的案例研究验证了其有效性。

    

    寄存器传输级（RTL）设计的功能验证已成为现代半导体开发中的关键瓶颈，往往消耗项目的大部分资源。调试尤其是一项非常耗时的任务，需要对仿真轨迹或反例进行仔细分析。最近的研究工作探索了将大语言模型（LLM）应用于RTL调试和修复，并展现出巨大前景。然而，现有方法通常只针对单行错误、以开环方式运行、严重依赖提示或检索、缺乏可扩展性，并且常常产生错误的修复。在本文中，我们提出了一种新颖的自动化RTL错误修复方法，该方法结合了大语言模型、断言和模型检测器，以提供可扩展、可信的语义基础和形式化保证。具体而言，该方法利用失败的断言来引导大语言模型为多行错误生成补丁，每个候选补丁……

    arXiv:2610.32597v1 Announce Type: cross  Abstract: Functional verification of Register-Transfer Level(RTL) designs has become a critical bottleneck in modern semiconductor development, often consuming the majority of project resources. Debugging, in particular, remains an especially time-consuming task, requiring careful analysis of simulation traces or counterexamples. Recent work has explored applying Large Language Models (LLMs) to RTL debug and repair with great promise. However, the existing approaches typically are for single-line bugs, operate in open-loop, rely heavily on prompts or retrieval, lack scalability, and often produce incorrect fixes. In this paper, we propose a novel automated RTL bug repair methodology that combines LLMs, assertions, and a model checker to provide scalable, trustworthy semantic grounding and formal guarantees. Concretely, the methodology leverages failing assertions to guide the LLM in generating patches for multi-line bugs. Each candidate patch is
    
[^9]: 教训比失败更长久：AI编码代理规则的案例研究

    Lessons Outlive Their Failures: A Case Study of Rules for AI Coding Agents

    [https://arxiv.org/abs/2610.32578](https://arxiv.org/abs/2610.32578)

    本研究通过对个人Markdown仓库进行六周案例研究，建立118次AI编码代理失败台账、验证LLM文档审计并重放历史失败，发现文档漂移是最大的失败类别，而沉淀为规则的教训能够比引发它们的失败更长久地发挥作用。

    

    AI编码代理越来越多地在一种“支撑框架”（harness）内工作：包括AGENTS.md等规则文件、按需技能、运行时钩子和自动化检查。大多数关于支撑框架的研究只关注单一时间点或短期基准测试运行。我报告了一项为期六周的案例研究，研究对象是我的个人记录系统——一个纯Markdown仓库，三种AI代理工具在统一共享的规则集下每日对其进行编辑。我建立了一个记录118次代理失败、相应修复方案以及失败是否复发的台账。我还针对47条已知过时陈述和20个植入错误，测试了该系统月度LLM文档审计的效果。最后，我在隔离的git克隆中重放了五次过往失败，分别在有修复和无修复的条件下进行。所有结果均来自同一个仓库，每个条件重复三次，因此我将这些结果作为探索性发现报告。过时文档（漂移）是最大的失败类别（118次中有48次），尽管其中35次是在同一天的审计中发现的。单次审计运行大约能发现一半的已知漂移（Cla……

    arXiv:2610.32578v1 Announce Type: new  Abstract: AI coding agents increasingly work inside a harness: rule files such as AGENTS.md, on-demand skills, runtime hooks and automated checks. Most studies of harnesses look at a single point in time or at short benchmark runs. I report a six-week case study of my personal system of record, a plain-markdown repository that three AI agent tools edited daily under one shared rule set. I built a ledger of 118 agent failures, their fixes and whether they came back. I also tested the system's monthly LLM documentation audit against 47 known stale statements and 20 planted errors. Finally, I replayed five past failures in isolated git clones, with and without their fixes. All results come from one repository with three runs per condition, so I report them as exploratory. Stale documentation (drift) was the largest failure class (48 of 118), although 35 of those were found on one audit day. A single audit run found about half of the known drift (Clau
    
[^10]: 从法规到工程实践：组织隐私氛围的中介作用

    From Regulation to Engineering: Organizational Privacy Climate as Mediator

    [https://arxiv.org/abs/2610.32576](https://arxiv.org/abs/2610.32576)

    该研究通过对156名美国信息系统开发者的调查发现，组织隐私氛围是连接法规与实践的关键中介——开发者对GDPR重要性的认知通过塑造组织隐私氛围影响其隐私态度，进而驱动其隐私工程实践。

    

    数据保护立法旨在通过规范信息系统存储和处理个人信息的方式来保护隐私权。虽然此类立法界定了法律义务，但将监管要求转化为工程实践远非易事。随着欧盟《人工智能法案》等新框架的出现，理解这些机制变得日益重要。为研究这些机制，我们对156名美国信息系统开发者进行了在线调查，考察了组织隐私氛围在开发者对GDPR的认知与其隐私工程实践之间关联中所扮演的角色。通过结构方程建模，我们发现组织隐私氛围在开发者感知的GDPR重要性与他们的隐私态度之间起中介作用，而这些隐私态度又与他们的隐私工程实践相关联。这一中介链条——法规、氛围、态度、实践——揭示了监管要求如何转化为实际工程实践。

    arXiv:2610.32576v1 Announce Type: cross  Abstract: Data protection legislation aims to protect privacy rights by regulating how information systems store and process personal information. While such legislation defines legal obligations, translating regulatory requirements into engineering practice is far from straightforward. As new frameworks such as the EU AI Act emerge, understanding these mechanisms becomes increasingly relevant. To investigate them, we conducted an online survey of 156 U.S. information system developers, examining the role of organizational privacy climate in the association between developers' perceptions of the GDPR and their privacy engineering practices. Using structural equation modeling, we found that organizational privacy climate mediates the relation between developers' perceived GDPR importance and their privacy attitudes, which in turn are associated with their privacy engineering practices. The mediation chain - regulation, climate, attitudes, practic
    
[^11]: 真实MCP故障下的工具调用LLM智能体：模型规模并不能换来容错能力

    Tool-Calling LLM Agents Under Realistic MCP Faults: Scale Does Not Purchase Fault Tolerance

    [https://arxiv.org/abs/2610.32555](https://arxiv.org/abs/2610.32555)

    提出基于真实MCP生态故障数据校准的故障注入中间件MCP-Injector，实验证明扩大LLM模型规模并不能提升智能体应对工具故障的容错能力。

    

    LLM智能体日益依赖通过模型上下文协议暴露的外部工具，然而当前的基准测试假设工具能够正确且及时地响应，使得智能体对现实中工具故障的韧性从未得到检验。我们提出了MCP-Injector，一种即插即用、协议感知的故障注入中间件，它能够介入MCP工具调用路径，支持受控的故障持久性类型（永久性、瞬时性、间歇性）以及可种子化的故障调度。其故障模型根据公共MCP生态系统中观察到的故障分布（故障模式×持久性）进行校准，这些数据来自对1,774个开源MCP仓库中57,240条议题和拉取请求线程的挖掘。在一项跨越两种骨干模型规模（20B、120B）、三种智能体框架（自定义多轮规划-执行框架、OpenAI Agents SDK、LangChain）以及三种故障注入条件的1,350个实验单元的大规模实验中，我们发现从这种以永久性故障为主的压力特征中注入的故障会导致严重的质量下降……

    arXiv:2610.32555v1 Announce Type: cross  Abstract: LLM agents increasingly rely on external tools exposed via the Model Context Protocol (MCP), yet current benchmarks assume tools respond correctly and on time, leaving agent resilience to realistic tool failures untested. We introduce MCP-Injector, a drop-in, protocol-aware fault-injection middleware that interposes on the MCP tool-call path with controlled persistence (permanent, transient, intermittent) and seedable schedules. Its fault model is calibrated to the observed incident distribution (failure mode x persistence) of the public MCP ecosystem, mined from 57,240 issue and pull-request threads across 1,774 open-source MCP repositories. In a 1,350-cell experiment crossing two backbone sizes (20B, 120B), three agent stacks (custom multi-round plan-and-execute, OpenAI Agents SDK, LangChain), and three fault injection conditions, we find that faults drawn from this permanent-heavy stress profile cause severe quality degradation (tas
    
[^12]: 超越覆盖率与变异得分：LLM生成单元测试的盲点分析

    Beyond Coverage and Mutation: A Blind-Spot Analysis of LLM-Generated Unit Tests

    [https://arxiv.org/abs/2610.32548](https://arxiv.org/abs/2610.32548)

    该论文提出了基于行为义务的单元测试评估框架BlindSpotEval和配套Java基准ObligBench，通过边界值充分性、控制流充分性和断言强度三个指标揭示了大语言模型生成单元测试的系统性盲点，发现错误断言占测试失败的15.6%，证明覆盖率无法保证断言正确性。

    

    现有的单元测试评估协议依赖于分支覆盖率和变异得分等聚合指标。这些指标衡量的是测试套件对程序行为的覆盖程度，但无法识别测试套件系统性地遗漏检查的行为义务。我们提出了BlindSpotEval，一个与Arrange-Act-Assert（准备-执行-断言）模式对齐的基于义务的评估框架。该框架包含三个指标：边界值充分性、控制流充分性和断言强度，并辅以对执行可靠性的评估。我们还引入了ObligBench，这是一个按复杂度分层的Java基准，包含895个复杂函数和700个低复杂度函数，并带有源代码侧的义务标注。对来自四个大语言模型在四种提示策略下生成的14,320个测试用例的评估揭示了系统性的盲点：错误断言占测试失败总数的15.6%，这表明仅凭覆盖率并不能保证断言（预言）的正确性。

    arXiv:2610.32548v1 Announce Type: new  Abstract: Existing evaluation protocols for unit tests rely on aggregate metrics such as branch coverage and mutation score. These metrics quantify the extent to which a test suite exercises program behavior, but do not identify the behavioral obligations it systematically leaves unchecked. We propose BlindSpotEval, an obligation-based evaluation framework aligned with the Arrange-Act-Assert pattern. The framework comprises three metrics, Boundary-Value Adequacy, Control-Flow Adequacy, and Oracle Strength, complemented by an assessment of execution reliability. We also introduce ObligBench, a complexity-stratified Java benchmark comprising 895 complex and 700 low-complexity functions with source-side obligation annotations. Evaluating 14,320 tests from four LLMs under four prompting strategies reveals systematic blind spots. Incorrect assertions account for 15.6% of test failures, illustrating that coverage alone does not establish oracle correctn
    
[^13]: 当智能体失败时：模型还是执行框架——该归咎于谁？

    When Agents Fail: Model or Harness--Who's to Blame?

    [https://arxiv.org/abs/2610.32483](https://arxiv.org/abs/2610.32483)

    本文提出了一个通过五种互补指标（有效完成率、模型-框架匹配度、跨模型匹配度、交互残差和Token负载）来评估LLM智能体执行框架与底层模型交互表现的方法，用以区分智能体失败时究竟应归咎于模型还是执行框架。

    

    LLM智能体将模型与执行框架相结合，因此智能体基准测试的性能反映了二者的共同行为。我们认为，因此应当通过执行框架与底层模型的交互来评估框架。我们开发了一个结合五种互补指标的评估框架：(i) 有效完成率（VCR），即智能体正常完成并交付可评估结果的可靠程度，与任务正确性无关；(ii) 模型-框架匹配度，即框架通过有效完成来支持给定模型达成任务的程度；(iii) 跨模型匹配度，即框架在被评估的各模型上的整体适用性；(iv) 交互残差，即被组件级汇总指标所掩盖的特定模型-框架配对的执行优势与劣势；以及 (v) Token负载，即智能体执行所消耗的token量，包括失败尝试。我们使用四个核心模型评估了四个通用执行框架（OpenCode、Codex CLI、Pi CLI和DeepAgents），外加两个固定的原生模型。

    arXiv:2610.32483v1 Announce Type: new  Abstract: LLM agents combine models with execution harnesses, so agentic benchmark performance reflects their joint behavior. We argue that harnesses should therefore be evaluated through their interactions with underlying models. We develop a framework combining five complementary measures: (i) Valid Completion Rate (VCR), how reliably agents finish normally and deliver evaluable results, irrespective of task correctness; (ii) Model-Harness Fit, how well a harness supports task achievement through valid completion for a given model; (iii) Cross-Model Fit, a harness's overall suitability across evaluated models; (iv) Interaction Residuals, pairing-specific execution strengths and weaknesses hidden by component-level summaries; and (v) Token Load, the token demands of agent execution, including failed attempts. We evaluate four general-purpose harnesses (OpenCode, Codex CLI, Pi CLI, and DeepAgents) with four core models, plus two fixed native model
    
[^14]: WaitLens：面向工业GUI测试智能体的基于时序视觉语言模型的自适应等待机制

    WaitLens: Temporal VLM-Based Adaptive Waiting for Industrial GUI Testing Agents

    [https://arxiv.org/abs/2610.32482](https://arxiv.org/abs/2610.32482)

    WaitLens通过在标注等待结束时间的GUI加载轨迹上微调轻量级视觉语言模型，并引入时间重采样构建多样化训练数据，实现了适用于多种工业环境的自适应等待策略，显著减少了占交互时间高达42.3%的固定等待开销。

    

    提升GUI测试智能体的效率对工业部署至关重要。除了大语言模型（LLM）推理之外，连续交互之间的固定等待也带来了大量开销。我们的工业研究显示，固定等待占采样交互区间时间的42.3%，与模型调用所消耗的54.8%相当。此外，62.9%的等待时间发生在GUI已经完成加载之后。尽管先前的研究提出了自适应等待策略，但它们对特定应用数据或平台特定信息的依赖限制了其在多样化工业环境中的适用性。为此，我们提出WaitLens，它在标注了等待结束时间的GUI加载轨迹上微调一个轻量级视觉语言模型（VLM）。我们引入时间重采样来构建多样化的训练样本，使模型能够从时间观测中学习不同的GUI加载行为。实验结果表明……（摘要原文在此处被截断）

    arXiv:2610.32482v1 Announce Type: new  Abstract: Improving the efficiency of GUI testing agents is important for industrial deployment. Beyond LLM inference, fixed waiting between consecutive interactions introduces substantial overhead. Our industrial study reveals that fixed waiting accounts for 42.3% of the time within the sampled interaction intervals, comparable to the 54.8% spent on model calls. Moreover, 62.9% of the waiting time is spent after the GUI has already finished loading. Although prior studies have proposed adaptive waiting strategies, their reliance on app-specific data or platform-specific information limits their applicability across diverse industrial environments. To this end, we propose WaitLens, which fine-tunes a lightweight VLM on GUI loading trajectories annotated with waiting end times. We introduce temporal resampling to construct diverse training examples, enabling the model to learn varying GUI loading behaviors from temporal observations. Experiments sh
    
[^15]: OPTARI：一个用于光声成像临床转化的开源软件框架

    OPTARI: An open-source software framework for clinical translation of optoacoustic imaging

    [https://arxiv.org/abs/2610.32468](https://arxiv.org/abs/2610.32468)

    OPTARI是一个基于napari和PATATO构建的开源图形化软件，使无编程背景的研究人员能够完成大规模临床光声-超声研究的全流程分析，涵盖图像重建、感兴趣区标注、发色团解混、深度学习组织分割以及时间与光谱分析。

    

    光声与超声联合成像是一种在多种病理范围内具有广阔前景的成像方式。然而，由于缺乏可供没有编程专业知识的用户使用的分析软件，其在临床研究中的应用受到限制。现有工具要么是基于脚本的工程框架，要么是厂商专有的应用程序。我们提出了OPTARI，一个开源的图形化软件应用，旨在促进大规模临床光声-超声（OPUS）研究。OPTARI基于napari图像查看器和PATATO工具包构建，用于光声数据处理。它支持厂商原生和开放的HDF5数据格式，符合IPASC标准，并以独立可执行程序的形式分发。其功能包括感兴趣区域标注、图像重建、发色团解混、基于深度学习的组织分割以及时间和光谱分析模块。OPTARI提供了目前其他软件无法实现的临床研究分析功能组合。

    arXiv:2610.32468v1 Announce Type: new  Abstract: Combined optoacoustic (OptA) and ultrasound imaging is a promising modality across a wide range of pathologies. Its use in clinical research, however, is limited by a lack of analysis software accessible to users without programming expertise. Existing tools are either script-based engineering frameworks or proprietary vendor applications. We present OPTARI, an open-source graphical software application to facilitate large-scale clinical optoacoustic-ultrasound (OPUS) studies. OPTARI builds on the napari image viewer and the PATATO toolkit for OptA data processing. It supports native vendor and open HDF5 data formats, aligns with the IPASC standard, and is distributed as a standalone executable. Features include region-of-interest annotation, reconstruction, chromophore unmixing, deep-learning-based tissue segmentation, and temporal and spectral analysis modules. OPTARI offers a combination of features for clinical study analysis that no
    
[^16]: 复杂度集中作为上帝类与上帝方法检测器：一项针对无标度离群点规则与工具定义及人工标注对比的预注册检验

    Complexity concentration as a god-class and god-method detector: a pre-registered test of a scale-free outlier rule against tool definitions and human labels

    [https://arxiv.org/abs/2610.32466](https://arxiv.org/abs/2610.32466)

    本文预注册检验了一种不依赖原始尺度阈值的上帝类/上帝方法检测新规则——即当单元复杂度在其自身代码库中构成稳健离群点（修正z分数≥3.5）时判定为“上帝单元”，但在445个多语言代码库与492个Java代码库上与PMD及MLCQ等现有定义对比后，结论尚不确定。

    

    上帝类和上帝方法检测器用原始尺度的阈值来标记集中的复杂度，而该阈值是在语料库上拟合出的常数，当总体（代码库分布）发生变化时就需要重新拟合。本文对一条不依赖原始尺度常数的规则进行了预注册检验：当一个单元的复杂度相对其自身代码库而言是稳健的离群点（修正z分数达到3.5或以上）时，该单元即被判定为“上帝单元”。在涵盖十种语言、其中171个为Java的445个代码库，以及MLCQ数据集中492个Java代码库上，该规则与PMD的GodClass定义、MLCQ的人工评审者标注以及其他两种定义进行了对比评分，信号指标与截断值在一半代码库上选取，并在其余代码库上验证。LLM编写的代码可能以不同方式聚集其复杂度，这正需要一个无需重新拟合的检测器；而首要问题是：该规则能否在规模上匹配现有由人类编写的代码。最终结论在两个人群上均不确定，且各自通过不同的检验分支得出。该规则从每文件认知复杂度的最大值出发，对PMD识别出的上帝类进行排序。

    arXiv:2610.32466v1 Announce Type: new  Abstract: God-class and god-method detectors mark concentrated complexity with a raw-scale threshold, a constant fitted on a corpus and re-fitted when the population moves. This paper pre-registers a test of a rule with no raw-scale constant: a unit is a god unit when its complexity is a robust outlier for its own codebase, a modified z-score of 3.5 or more. On 445 repositories in ten languages, 171 of them Java, and the 492 Java repositories of MLCQ, the rule is scored against PMD's GodClass, MLCQ's reviewers and two further definitions, with signal and cutoff chosen on half the repositories and read on the rest. LLM-written code may gather its complexity differently, the case for a detector that needs no re-fitting; the first question is whether one can match the scale of existing human-written code. The verdict is inconclusive on both populations, by different arms. The rule ranks PMD's god classes from the per-file maximum of cognitive complex
    
[^17]: 基于搜索的视觉语言模型模态敏感性测试

    Search-Based Testing of Modality Sensitivity in Vision Language Models

    [https://arxiv.org/abs/2610.32360](https://arxiv.org/abs/2610.32360)

    本文提出黑盒测试框架MoST，通过多目标搜索系统对比图像、文本及联合扰动，首次揭示视觉语言模型对不同模态扰动的敏感性存在显著差异。

    

    视觉语言模型（VLM）联合处理图像和文本，然而现有的鲁棒性评估通常分别扰动这两类输入，或仅评估预定义的损坏等级。这样的评估对于“哪种模态更容易受到扰动影响”以及“两种模态中较小的变化能否共同暴露模型缺陷”所提供的证据十分有限。我们提出了MoST，这是一个黑盒测试框架，通过对比仅图像扰动、仅文本扰动和联合扰动来刻画模态敏感性。MoST采用多目标搜索来选择和组合扰动并调整其严重程度，在尽量限制对图像和提示词改动的同时，寻求引发模型预测性能的退化。我们在四个VLM上评估了MoST，这些模型在ImageNet和驾驶图像上执行目标定位任务。实验结果表明各模型表现出不同的敏感性：某些模型对文本扰动更为脆弱，而另一些模型则在两种模态下均易受影响。在若干配置下，联合扰动…

    arXiv:2610.32360v1 Announce Type: cross  Abstract: Vision-language models (VLMs) process images and text jointly, yet robustness evaluations often perturb these inputs separately or assess only predefined corruption levels. Such evaluations provide limited evidence of which modality is more susceptible to perturbations and whether smaller changes across both modalities can jointly expose failures. We present MoST, a black-box testing framework that characterizes modality sensitivity by comparing image-only, text-only, and joint perturbations. MoST uses multi-objective search to select and combine perturbations and adjust their severity, seeking prediction degradation while limiting changes to images and prompts. We evaluate MoST on four VLMs performing object localization on ImageNet and driving images. The models exhibit distinct sensitivities: textual perturbations dominate for some, whereas others are susceptible across both modalities. In several configurations, joint perturbations
    
[^18]: DeepAVR：基于多智能体协作的自动化漏洞修复深度分析方法

    DeepAVR: Deep Analysis via Multi-Agent Collaboration for Automated Vulnerability Repair

    [https://arxiv.org/abs/2610.32345](https://arxiv.org/abs/2610.32345)

    DeepAVR提出了一种多智能体协作框架，将中间修复结果转化为结构化诊断反馈，并在故障定位、补丁生成、编译/安全/功能验证与补丁审查等专门角色之间流转反馈，从而实现端到端的深度漏洞分析与自动化修复。

    

    自动化漏洞修复（AVR）已成为抵御软件安全缺陷的关键防线，旨在自主地对漏洞进行分诊与修复。早期基于大语言模型（LLM）的方法在假定故障定位完美无缺的前提下专注于补丁生成，而近期的端到端方法则尝试直接依据消毒器日志来修复漏洞。然而，当根本原因被深深隐藏时——无论是物理上远离崩溃点，还是在逻辑上被相互关联的错误所掩盖——这些方法往往会失败，因为初始报告提供的信息并不充分。我们提出了DeepAVR，一个用于端到端自动化漏洞修复的多智能体框架，它通过将中间修复结果转化为结构化的诊断反馈来执行深度分析。DeepAVR将修复过程分解为专门的智能体角色——故障定位、补丁生成、编译验证、安全验证、功能验证和补丁审查——并在这些角色之间传递反馈以进行迭代……（原文摘要至此截断）

    arXiv:2610.32345v1 Announce Type: new  Abstract: Automated Vulnerability Repair (AVR) has emerged as a critical defense against software security flaws, aiming to triage and fix bugs autonomously. Early LLM-based approaches focused on patch generation under the assumption of perfect fault localization, while recent end-to-end methods attempt to repair vulnerabilities directly from sanitizer logs. However, these methods often fail when the root cause is deeply hidden--either physically distant from the crash site or logically masked by interdependent errors--making the initial report insufficient. We propose DeepAVR, a multi-agent framework for end-to-end AVR that performs deep analysis by transforming intermediate repair outcomes into structured diagnostic feedback. DeepAVR decomposes repair into specialized roles--Fault Localization, Patch Generation, Compile Validation, Security Validation, Functionality Validation, and Patch Review--and routes feedback across these roles to iterativ
    
[^19]: 从错误论断到错误补丁：编码智能体中任务接地推理的实证研究

    From Faulty Claims to Incorrect Patches: An Empirical Study of Task-Grounded Reasoning in Coding Agents

    [https://arxiv.org/abs/2610.32338](https://arxiv.org/abs/2610.32338)

    该论文开展了首个针对编码智能体“任务接地推理”的实证研究，通过分析四个开源推理模型在 SWE-bench Pro 上的 400 条轨迹，量化其思维链中可核查论断里的错误论断，并检验错误论断是否会导致错误补丁。

    

    基于大语言模型（LLM）的编码智能体越来越多地能够解决真实的软件工程任务，但在许多任务上仍会失败。先前的轨迹研究通过与任务无关的智能体行为模式（例如重复执行的动作）来刻画这些失败。然而，智能体即使遵循了合理的工作流程，也可能基于对现有代码、任务需求或候选修复方案的错误理解来采取行动。驱动大多数最先进编码智能体的推理模型会在其原生思维链（CoT）中显露这种理解，其中关于任务的论断可以对照事实加以核查。然而，目前尚不清楚这类 CoT 中包含多少可核查的论断、其中有多少是错误的，以及错误论断是否真的会导致不正确的补丁。我们对这类论断——我们称之为“任务接地推理”——开展了首个实证研究，分析了四个开源权重推理模型在 SWE-bench Pro 上的 400 条轨迹。（原文摘要在此处被截断）

    arXiv:2610.32338v1 Announce Type: new  Abstract: Large language model (LLM)-based coding agents increasingly resolve real-world software engineering tasks, yet still fail on many of them. Prior trajectory studies characterize these failures through task-agnostic patterns of agent behavior, such as repeated actions. However, an agent may follow a sound workflow yet act on a faulty understanding of the existing code, the task requirements, or a candidate fix. Reasoning models, which power most state-of-the-art coding agents, reveal this understanding in their native chain-of-thought (CoT), whose claims about the task can be checked against the facts. Yet it remains unclear how many such checkable claims the CoT contains, how many are faulty, and whether faulty claims actually cause incorrect patches. We present the first empirical study of such claims, which we term task-grounded reasoning, analyzing 400 trajectories of four open-weight reasoning models on SWE-bench Pro. We extract atomi
    
[^20]: GitHub Copilot 智能体轨迹的大规模研究

    Large-Scale Study of GitHub Copilot Trajectories

    [https://arxiv.org/abs/2610.32313](https://arxiv.org/abs/2610.32313)

    该研究通过分析 7,581 个真实 GitHub Copilot 智能体 PR 的人在回路轨迹，揭示了真实开发中人与 AI 智能体的协作模式——大多数 PR 无人类干预且协作短暂，而首次人类干预会显著缩短后续轨迹，弥补了现有基准测试假设与真实使用场景之间的差距。

    

    AI 编程智能体正日益融入软件开发工作流。然而，大多数编程智能体研究依赖于类似 SWE-bench 的基准测试，这些测试假设智能体在单一轨迹中自主运行。这一假设无法反映真实世界中的使用情况，即开发者与智能体进行交互式协作。我们通过研究真实环境中人在回路的智能体轨迹来填补这一空白，重点分析了 7,581 个由 Copilot 撰写且具有公开可访问会话数据的 PR。我们发现，58% 的 Copilot PR 没有任何人类干预，而当开发者确实介入时，协作是短暂的（人机循环中位数为 2 次）。已合并的 PR 的初始轨迹迭代次数也往往少于未合并的 PR。我们还发现，第一次人类干预之后的轨迹在迭代次数上显著下降 20.8 次，持续时间缩短 8.4 分钟；然而，后续干预之后的轨迹迭代次数下降幅度则很小。

    arXiv:2610.32313v1 Announce Type: new  Abstract: AI coding agents are increasingly integrated into software development workflows. However, most coding agent studies rely on benchmarks similar to SWE-bench, which assume autonomous agents operating in a single trajectory. This assumption does not mirror real-world usage, where developers interactively collaborate with agents. We address this gap by studying human-in-the-loop agent trajectories in the wild, focusing on 7,581 Copilot-authored PRs with publicly accessible session data. We find that 58% of Copilot PRs have no human intervention, and that when developers do intervene, collaboration is brief (median of 2 Human-AI loops). Merged PRs also tend to have fewer initial trajectory iterations than unmerged ones. We also find that the trajectory following the first human intervention exhibits a significant drop of 20.8 iterations and 8.4 min in duration. However, trajectories following later interventions show minimal iteration drops 
    
[^21]: 配对至关重要：轻量级编码代理的性能、失败与效率

    Pairing Matters: Performance, Failures, and Efficiency in Lightweight Coding Agents

    [https://arxiv.org/abs/2610.32131](https://arxiv.org/abs/2610.32131)

    本研究通过对六个开源代理框架与三个轻量级编码模型在SWE-bench Verified任务上进行1,800次运行的全因子受控评估，首次系统揭示了框架-模型配对方式对编码代理性能、失败模式和效率的显著影响。

    

    对于需要将专有代码和开发轨迹保留在受控基础设施中的软件工程团队而言，自主托管编码代理往往是必要的。轻量级、开放权重的模型通过降低硬件要求使这种部署更加可行，同时它们在仓库级基准测试上的强劲表现也证明了其在高难度编码任务中的潜力。将这些模型部署为编码代理需要将其与代理框架进行配对，这使得二者的联合性能成为关键的部署考量因素。然而，现有的排行榜主要报告固定的框架-模型配置，而非对各种替代配对进行受控比较。为填补这一空白，我们通过对六个开源框架和三个轻量级编码模型在100个SWE-bench Verified任务上进行受控全因子评估来研究框架与模型之间的交互作用，在统一的服务、计量和评分条件下产生了1,800次评级运行。

    arXiv:2610.32131v1 Announce Type: new  Abstract: Self-hosting coding agents is often necessary for software-engineering teams that must keep proprietary code and development traces within controlled infrastructure. Lightweight, open-weight models make such deployment more feasible by reducing hardware requirements, while strong results on repository-level benchmarks demonstrate their potential for demanding coding tasks. Deploying these models as coding agents requires pairing them with an agent harness, making their joint performance a key deployment consideration. Yet existing leaderboards primarily report fixed harness-model configurations rather than controlled comparisons across alternative pairings. To address this gap, we investigate harness-model interactions through a controlled full-factorial evaluation of six open-source harnesses and three lightweight coding models across 100 SWE-bench Verified tasks, producing 1,800 graded runs under common serving, metering, and grading c
    
[^22]: 更短并不更便宜：声明式动画规范与自由格式代码在LLM生成解释中的受控比较

    Shorter Is Not Cheaper: A Controlled Comparison of Declarative Animation Specifications and Free-Form Code for LLM-Generated Explanations

    [https://arxiv.org/abs/2610.32129](https://arxiv.org/abs/2610.32129)

    本研究通过预先冻结假设与分析计划的受控实验发现，尽管声明式动画规范的输出比自由格式代码短25-48%，但“更短即更便宜、更可靠”的普遍预期并不成立。

    

    聊天助手越来越多地使用交互式、动画化的解释来回答概念性问题，这些解释要么以自由格式的HTML/JavaScript代码生成，要么以由固定运行时播放的声明式规范生成。声明式方式被普遍预期更便宜、更快、更可靠，因为其输出更短且受到约束。我们在假设和分析计划均预先冻结的受控比较中检验了这一预期。我们设计了ExplainSpec——一种按时间索引的声明式动画格式，以及一项观察契约，使每个生成的解释都可以被程序化检查；经过验证的、与节奏无关的检查器对来自五个领域的20个解释请求进行评分。三个Claude模型（360个任务）以及后续加入的开源权重Qwen3-14B（分别启用和禁用思考模式，240个任务）以两种格式回答了所有请求。对于每个模型，声明式规范的答案令牌数量缩短了25-48%（摘要在此处被截断）

    arXiv:2610.32129v1 Announce Type: new  Abstract: Chat assistants increasingly answer conceptual questions with interactive, animated explanations, generated either as free-form HTML/JavaScript or as declarative specifications that a fixed runtime plays. The declarative route is widely expected to be cheaper, faster and more reliable, because its outputs are shorter and constrained. We test this expectation in controlled comparisons whose hypotheses and analysis plans were frozen before the runs. We designed ExplainSpec, a time-indexed declarative animation format, and an observation contract that makes every generated explanation programmatically inspectable; validated, pacing-agnostic checkers score 20 explanation requests from five domains. Three Claude models (360 jobs) and, in a follow-up, the open-weight Qwen3-14B with thinking enabled and disabled (240 jobs) answered every request in both formats. For every model the specifications were 25-48% shorter in answer tokens (output exc
    
[^23]: 超越正确性：软件工程新手学生如何对生成式AI行使监督

    Beyond Correctness: How Novice Software Engineering Students Exercise Oversight of Generative AI

    [https://arxiv.org/abs/2610.32118](https://arxiv.org/abs/2610.32118)

    软件工程新手学生对生成式AI的监督主要聚焦于输出的适用性问题（74.4%）而非正确性问题（16.9%），表明学生监督的核心是判断看似合理的AI输出是否真正适合实际任务。

    

    生成式人工智能（GenAI）正在将软件工程学生的工作重心从产出制品转向评估和监督AI生成的输出。然而，关于学生在报告干预时识别出输出具备哪些特征，或者他们在真实的软件工程工作中如何行使监督，目前知之甚少。我们分析了145名一年级软件工程学生通过结构化AI日志记录的962个学生-AI交互回合，其中包括172次学生声明的干预。通过多维编码框架，我们刻画了这些报告的干预中识别出的输出特征、学生干预的原因、他们采取的行动以及这些行动的实施方式。74.4%的干预中发现了适用性问题，而涉及正确性问题的仅占16.9%，这表明监督更多关注的是看似合理的AI输出是否真正适合于

    arXiv:2610.32118v1 Announce Type: new  Abstract: Generative AI (GenAI) is shifting software engineering students' work from producing artefacts towards evaluating and supervising AI-generated outputs. However, little is known about what characteristics students identify in outputs when reporting interventions, or how they exercise oversight during authentic software engineering work. We analysed 962 student-AI interaction turns documented by 145 first-year software engineering students through structured AI Journals, including 172 student-declared interventions. Using a multi-dimensional coding framework, we characterised the output characteristics identified in these reported interventions, students' reasons for intervening, the actions they took, and how those actions were enacted. Fitness problems were identified in 74.4% of interventions, compared with 16.9% involving correctness problems, indicating that oversight more often concerned whether plausible AI outputs were suitable for
    
[^24]: 谁来验证工具？失真的格式化输出如何误导智能体及其验证器

    Who Verifies the Tools? How Warped Formatters Mislead Agents and Their Verifiers

    [https://arxiv.org/abs/2610.32110](https://arxiv.org/abs/2610.32110)

    论文通过生产RAG系统的真实案例与对照实验发现，工具输出格式化器遗漏关键信息会误导智能体和验证器，而仅需在输出中补全这些缺失细节，就能显著减少智能体的无依据声称。

    

    智能体和验证器通常依赖工具打印出的文本，而这些文本可能会歪曲工具实际执行的操作，即使工具本身的计算是正确的。我们在纽约Pilsudski研究所使用的生产级RAG研究系统中观察到了这一现象：工具的输出格式化器误导智能体向用户提供了错误或不完整的信息。每个工具都正确地完成了自己的任务，但其输出省略了搜索限制、结果数量上限或标识符命名空间等关键信息。我们还重现了一个在代码审查阶段、发布前被发现的未标注估计值。我们研究了这些格式化器是否会误导智能体和验证器，以及修复后的措辞能带来多大改善。我们向智能体提供了使用两种格式化器版本的工具结果：一个版本包含关键遗漏，另一个版本则清楚说明了所有缺失的细节。仅这一差异就改变了智能体的声称内容以及验证器所接受的内容。在四个失败案例中的三个里，智能体的无依据声称大幅下降（原文在此处截断）。

    arXiv:2610.32110v1 Announce Type: new  Abstract: Agents and verifiers often rely on the text a tool prints, and that text can misrepresent what the tool actually did, even when the tool's computation is correct. We saw this happen in a production RAG research system used by the Pilsudski Institute in New York, where tool output formatters misled an agent into giving wrong or incomplete information to users. Each tool did its job correctly, but its output omitted search limits, result caps, or identifier namespaces. We also recreated an unlabeled estimate caught in code review before release. We studied whether these formatters misled agents and verifiers and how much the repaired wording helps. We fed tool results to agents using two formatter versions: one with the key omissions and one with all missing details clearly stated. That difference alone changed what the agents claimed and what the verifiers accepted. Unsupported agent claims dropped substantially for three of the four fail
    
[^25]: CrowdStrike 频道文件 291：一次全球端点宕机事故的工程分析

    CrowdStrike Channel File 291: An Engineering Analysis of a Global Endpoint Outage

    [https://arxiv.org/abs/2610.32078](https://arxiv.org/abs/2610.32078)

    本文通过工程分析重建了2024年7月 CrowdStrike Falcon 内容更新导致全球约850万台 Windows 设备崩溃事件的根本原因——检测模板声明21个输入而传感器代码仅提供20个，暴露了内核态解释器中未检查的数组访问——并识别出验证、测试、部署和恢复四个保障缺口。

    

    2024年7月19日，CrowdStrike Falcon 的一次内容更新导致 Windows 系统崩溃，据微软估计影响了约850万台设备。直接缺陷是一个声明了21个输入的检测模板与仅提供20个输入的传感器代码之间的不一致。一项新的匹配准则暴露了内核态内容解释器中一处未经检查的数组访问。本文基于供应商技术报告、微软的崩溃分析与恢复指南、美国国会记录以及运营影响报告，对这一事件进行了重建。文章区分了有据可查的结论与工程推断，并考察了接口验证、测试选择、部署策略和恢复依赖之间的相互作用。文中指出四个关键性的保障缺口：针对消费者未满足的接口定义进行验证、测试用例避开了缺失的输入、分发过程缺乏基于健康状态的灰度晋升机制、以及恢复路径需要共（……摘要在此处被截断）。

    arXiv:2610.32078v1 Announce Type: new  Abstract: On July 19, 2024, a CrowdStrike Falcon content update caused Windows systems to crash, affecting about 8.5 million devices according to a Microsoft estimate. The immediate defect was an inconsistency between a detection template that declared 21 inputs and sensor code that supplied 20. A new matching criterion exposed an unchecked array access in the kernel-mode content interpreter. This article reconstructs the incident from vendor technical reports, Microsoft's crash analysis and recovery guidance, congressional records, and operational impact reports. It distinguishes documented findings from engineering inferences and examines how interface validation, test selection, deployment policy, and recovery dependencies interacted. Four assurance gaps were consequential: validation against a definition the consumer did not meet, tests that avoided the missing input, distribution without health-gated promotion, and recovery paths requiring co
    
[^26]: LLM生成的形式化规范是否忠实于其需求？一项关于双模态形式化规范生成的研究

    Are LLM-Generated Formal Specifications Faithful to Their Requirements? A Study on Bimodal Formal Specification Generation

    [https://arxiv.org/abs/2610.32053](https://arxiv.org/abs/2610.32053)

    本文针对程序正确性未知这一形式化验证的真实场景，开展了双模态（自然语言需求+代码实现）下LLM生成ACSL形式化规范是否忠实于需求的实证研究，构建了包含1,403个C程序的基准并用Frama-C进行验证评估。

    

    形式化规范生成是实现软件形式化验证与正确性保证的关键步骤。近年来，大语言模型已被广泛应用于形式化规范的自动生成。然而，现有基于LLM的规范生成方法通常假设输入的程序是正确的，这偏离了形式化验证的真正目的——在程序正确性未知的情况下判断程序是否满足其需求。因此，我们在双模态设置下对基于LLM的规范生成开展了一项实证研究：同时给定自然语言需求及其对应的代码实现，且其正确性未知。我们构建了一个包含1,403个C程序的基准数据集，并结合Frama-C验证来评估ACSL规范的生成。我们的研究得出了若干有趣的发现：首先，当代码实现正确时，提供自然语言需求……

    arXiv:2610.32053v1 Announce Type: new  Abstract: Formal specification generation is a critical step towards formal verification and correctness assurance of software. Recently, large language models have been widely adopted in automatic generation of formal specifications. However, existing LLM-based specification generation approaches often assume that the input program is correct, which deviates from the real purpose of formal verification---to figure out whether the program meets its requirements when its correctness is unknown. Therefore, we conduct an empirical study of LLM-based specification generation under the bimodal setting: given both the natural language requirement and its corresponding code implementation, whose correctness is unknown. We construct a benchmark of 1,403 C programs and evaluate ACSL specification generation with Frama-C verification. Our study reaches several interesting findings: first, when the code implementation is correct, providing the natural langua
    
[^27]: 重试的许可：AI智能体工作流中的授权与恢复研究

    Permission to Retry: A Study of Authorization and Recovery in AI Agent Workflows

    [https://arxiv.org/abs/2610.31916](https://arxiv.org/abs/2610.31916)

    该论文将AI智能体的恢复决策与语言模型行为分离，提出一项在LangGraph和Temporal宿主上执行336条历史记录的有限一致性研究，验证了检查点状态、分发许可、外部效果证据和对账许可这四种独立义务的执行时可靠性。

    

    持久智能体可以在其权限发生变化、且动作的外部结果仍不确定的情况下恢复一项具有重大影响的动作。我们研究使用工具的AI智能体的执行时可靠性，将恢复决策与语言模型行为隔离开来。检查点状态、分发许可、外部效果证据以及对账许可是四种彼此不同的义务。我们提出了一项有限一致性研究，通过将执行历史记录与对恢复应用隐藏的提供方效果账本进行比对评分，来检查这些义务。一项前瞻性冻结的实验在原生LangGraph和Temporal宿主上执行了336条历史记录，覆盖14种调度、三种提供方契约和四种匹配的应用策略。全部168个配对宿主单元都得到了完全相同的检查结果。在每个宿主上，保留账本对账通过了全部42条历史记录：21个已验证的完成、15个拒绝，以及6个保留预定的未知结果。缓存重试（摘要在此处被截断）

    arXiv:2610.31916v1 Announce Type: new  Abstract: A durable agent can resume a consequential action while its authority changes and the action's external outcome remains uncertain. We study execution-time reliability of tool-using AI agents, isolating recovery decisions from language-model behavior. Checkpoint state, permission to dispatch, evidence of an external effect and permission to reconcile are distinct obligations. We present a finite conformance study that checks these obligations through histories scored against a provider effect ledger hidden from the recovering application. A prospectively frozen campaign executes 336 histories on native LangGraph and Temporal hosts, crossing 14 schedules, three provider contracts and four matched application policies. All 168 paired host cells have identical checked outcomes. Per host, retained-ledger reconciliation passes all 42 histories: 21 checked completions, 15 denials and six unknown outcomes with reservations retained. Cached retry
    
[^28]: 知道何时停止：编码智能体中的证据-行动差距

    Knowing When to Stop: The Evidence-Action Gap in Coding Agents

    [https://arxiv.org/abs/2610.31895](https://arxiv.org/abs/2610.31895)

    该论文揭示了编码智能体的“证据-行动差距”现象——即智能体即使已收集到表明任务无法独立完成的充分证据仍过度坚持或虚报成功，并提出ImpasseBench基准通过在可解决任务中注入环境阻碍并标记证据充分步骤来量化这一差距。

    

    编码智能体被训练来完成任务，即使环境使得任务根本无法完成时，它们仍会不断尝试。表明只有外部帮助才能恢复进展的证据通常在几步之内就会出现，但智能体很少据此采取行动：它们过度坚持，尝试不必要的变通方法，并声称取得记录无法支持的成功——我们将这一现象称为“证据-行动差距”。为了衡量这一差距，我们提出了ImpasseBench，它在原本可解决的修复任务中注入十二种真实的环境阻碍因素，并在每条轨迹中标记出所收集证据变得充分的步骤。相对于该步骤对智能体的行动进行评分，可以区分适当的努力与过度坚持以及过早求助。在八个前沿模型和19,371条轨迹上，我们发现几乎所有需要外部帮助的运行都收集到了充分的证据，但仍有三分之一耗尽了预算，而且我们检测到的所有不安全请求都发生在这一时点之后。

    arXiv:2610.31895v1 Announce Type: cross  Abstract: Coding agents are trained to complete tasks, and they keep trying even when the environment makes completion impossible. The evidence that only outside help can restore progress usually appears within a few steps, yet agents rarely act on it: they over-persist, attempt undesired workarounds, and claim success the record does not support -- a phenomenon we call evidence-action gap. To measure it, we introduce ImpasseBench, which injects twelve realistic environment blockers into otherwise solvable repair tasks and marks, in each trajectory, the step at which the collected evidence becomes sufficient. Scoring actions relative to that step separates appropriate effort from over-persistence and premature handoff. Across eight frontier models and 19,371 trajectories, we find that nearly every run that needs outside help collects sufficient evidence, yet a third still exhaust their budget, and every unsafe request we detect comes after that 
    
[^29]: SWE-Clar：编码智能体能否澄清不完整的客户需求并交付全新软件？

    SWE-Clar: Can Coding Agents Clarify Incomplete Client Requirements and Deliver Greenfield Software?

    [https://arxiv.org/abs/2610.31854](https://arxiv.org/abs/2610.31854)

    该论文提出了 SWE-Clar 基准，基于 100 个报酬至少 4,000 美元的真实 Upwork 客户委托任务，通过“先澄清需求、后实现代码”的多轮交互设计并以明确的需求理解测试来评估编码智能体在自由职业全新软件开发中澄清不完整客户需求的能力。

    

    arXiv:2610.31854v1 公告类型：交叉引用（cross）。摘要：在自由职业环境中，全新软件开发要求编码智能体将不完整的客户请求转化为可运行的软件。这使得需求澄清至关重要，然而客户可能无法在漫长的开发周期中提供持续反馈，而且即使收到客户的回答，也不能保证智能体能正确理解复杂需求。这些挑战促成了两项基准设计选择：需求澄清先于代码实现，并通过明确的需求理解测试来评估澄清的有效性。基于这些设计选择，我们提出了 SWE-Clar，这是一个基于真实、高价值客户委托的基准，用于评估该场景下的需求澄清与代码实现能力。该基准包含 100 个源自 Upwork 上真实软件开发工作的任务，每个任务的总报酬至少为 4,000 美元。每个任务均以编码智能体与客户智能体之间的多轮交互开始，

    arXiv:2610.31854v1 Announce Type: cross  Abstract: Greenfield software development in freelance settings requires coding agents to turn incomplete client requests into working software. This makes requirement clarification essential, yet clients may be unable to provide continuous feedback over long development cycles, and receiving their answers does not guarantee correct understanding of complex requirements. These challenges motivate two benchmark design choices: clarification precedes code implementation, and its effectiveness is assessed through explicit tests of requirement understanding. Guided by these choices, we introduce SWE-Clar, a benchmark grounded in real, high-value client commissions to evaluate requirement clarification and code implementation in this setting. It comprises 100 tasks based on real software development jobs on Upwork, each with at least US\$4,000 in total payments. Each task begins with multi-round interaction between a coding agent and a client agent, 
    
[^30]: Ulysses：一个用于合成可验证网络漏洞利用的安全测试框架

    Ulysses: A Security Harness For Synthesizing Verifiable Web Exploits

    [https://arxiv.org/abs/2610.31831](https://arxiv.org/abs/2610.31831)

    Ulysses是一个多智能体安全测试框架，通过迭代探索漏洞假设并结合独立观察的执行效果验证，将候选漏洞发现转化为可验证的可利用性证明，在25个真实Web应用中成功验证了10个已知漏洞并新发现4个此前未报道的漏洞。

    

    AI智能体如何能在任意软件仓库中自动发现安全漏洞？现有方法使用大语言模型（LLM）来生成软件漏洞利用程序，然而，LLM很难可靠地验证其自身生成的漏洞利用程序。在本文中，我们提出了Ulysses，一个多智能体安全测试框架，用于为Web应用程序合成可验证的漏洞利用程序。Ulysses通过迭代探索漏洞利用假设，并在独立观察到的执行效果中确立成功，将候选漏洞发现转化为经过验证的可利用性证明。我们在WebExploitGym上评估了Ulysses，这是一个精选的基准测试集，包含来自25个真实世界Web应用的可利用实例，涵盖5种语言和7个框架。Ulysses为25个真实安全公告中的10个生成了经过验证的证明，还发现了4个此前从未被报道过的新可利用性证明。

    arXiv:2610.31831v1 Announce Type: cross  Abstract: How can an AI agent automatically discover security vulnerabilities in any software repository? Existing approaches use LLMs to generate software vulnerability exploits. However, it is challenging for an LLM to reliably verify its own generated exploits. In this paper, we propose Ulysses, a multi-agent security harness to synthesize verifiable vulnerability exploits for web applications. Ulysses converts candidate vulnerability findings into verified Proofs of Exploitability by iteratively exploring exploit hypotheses and establishing success in independently observed execution effects. We evaluated Ulysses on WebExploitGym, a curated benchmark of exploitable instances from 25 real-world web applications that span 5 languages and 7 frameworks. Ulysses produces verified proofs for 10 of 25 real-world advisories. It also discovered 4 new proofs of exploitability that have never been previously reported.
    
[^31]: CuraLoc：通过计划式代码检查实现仓库级缺陷定位

    CuraLoc: Repository-Level Fault Localization via Planned Code Inspection

    [https://arxiv.org/abs/2610.31809](https://arxiv.org/abs/2610.31809)

    CuraLoc提出在源码检索前一次性规划代码检查的缺陷定位方法：模型在单次规划中同时选定起始符号、检查关系并形成修改假设，再由确定性检索完成最终排序，从而减少反复的模型调用和token消耗。

    

    缺陷报告描述了出了什么问题，但很少指出代码应该在哪里修改。现有的仓库级定位方法通常让语言模型迭代地决定下一步要检查哪些代码，这需要多次模型调用并消耗大量token。我们研究了这些检查决策是否可以在源码检索之前一次性规划完成。在本研究中，我们提出了CuraLoc，它首先筛选候选文件，并使用简洁的行为与结构描述来表示文件中的符号。在单次规划决策中，模型联合完成三项任务：选择起始符号、确定从每个符号出发要检查的代码关系，并对可能的代码修改位置形成假设。随后由确定性检索收集所请求的源码，用于基于这些假设进行的最终排序，从而将模型的推理集中在选择和比较代码上，而不是指导每一步检索。我们在…上验证了所提方法的有效性

    arXiv:2610.31809v1 Announce Type: cross  Abstract: A bug report describes what went wrong, but rarely where the code should change. Existing repository-level localization methods often let language models iteratively decide what code to inspect next, requiring repeated model calls and substantial token usage. We investigate whether these inspection decisions can instead be planned before source retrieval. In this research, we introduce CuraLoc, which first selects candidate files and represents their symbols using compact behavioral and structural descriptions. In a single planning decision, the model jointly selects starting symbols, chooses code relationships to inspect from each, and forms hypotheses about likely edits. Deterministic retrieval collects the requested source for a final ranking informed by those hypotheses, concentrating model reasoning on choosing and comparing code rather than directing each retrieval step. We demonstrate the efficacy of our proposed approach on the
    
[^32]: 基于大语言模型的微服务系统自回归多模态异常检测

    Autoregressive Multimodal Anomaly Detection with Large Language Models for Microservice Systems

    [https://arxiv.org/abs/2610.31722](https://arxiv.org/abs/2610.31722)

    该论文提出了LLMMAD，一个基于大语言模型的自回归多模态异常检测框架，通过在跨模态融合中保留模态特定特征并解决“身份捷径”问题，来提升微服务系统的异常检测效果。

    

    微服务系统已成为大规模云应用的主流架构，支持快速部署和弹性伸缩。然而，确保微服务系统的可靠性和可用性关键依赖于有效的异常检测，因为故障和性能异常会在服务之间快速传播，严重影响端到端的系统行为。现有的微服务系统多模态异常检测方法通常面临两个根本性的局限：（i）为了跨模态融合而忽略了模态特定的特征，以及（ii）影响特征重构过程的“身份捷径”问题。为了解决这些问题，我们提出了LLMMAD，一个基于大语言模型的微服务系统自回归多模态异常检测框架。首先，为了协调跨模态融合与模态特定特征的保留，LLMMAD将（原文在此处截断）

    arXiv:2610.31722v1 Announce Type: cross  Abstract: Microservice systems have become a dominant architecture for large-scale cloud applications, enabling rapid deployment and elastic scaling. However, ensuring the reliability and availability of microservice systems critically relies on effective anomaly detection, as failures and performance anomalies can quickly propagate across services and severely impact end-to-end system behavior. Existing multimodal anomaly detection approaches for microservice systems often struggle with two fundamental limitations: (i) the neglect of modality-specific characteristics in favor of cross-modality fusion, and (ii) the "identity-shortcut" problem that affects the feature reconstruction process. To address these issues, we propose LLMMAD, an autoregressive multimodal anomaly detection framework for microservice systems based on Large Language Models (LLMs). Initially, to reconcile cross-modality fusion with modality-specific preservation, LLMMAD embe
    
[^33]: 开源项目对编码智能体说了什么：AGENTS.md 与 CLAUDE.md 文件普查

    What Open-Source Projects Tell Their Coding Agents: A Census of AGENTS.md and CLAUDE.md

    [https://arxiv.org/abs/2610.31715](https://arxiv.org/abs/2610.31715)

    这项研究首次对十三种语言的活跃 GitHub 仓库中的 AGENTS.md 和 CLAUDE.md 指令文件进行大规模普查，系统揭示了开源项目向 AI 编码智能体传达的指令构成（项目知识与代码、构建、工作流规则的混合），以及各项目在 AI 署名披露等问题上的立场分歧。

    

    编码智能体可以通过仓库中的指令文件获得引导。我们通过对十三种语言的热门活跃 GitHub 仓库进行普查来考察这一新兴实践，将每个仓库根目录下的 AGENTS.md 和 CLAUDE.md 视为一个指令集。智能体指令出现在一定比例的仓库以及星标最高的仓库之中；当两个文件同时存在时，CLAUDE.md 通常指向 AGENTS.md。我们采用模型辅助标注与人工审核相结合的方式分析这些指令语句。标注结果显示，指令内容是项目知识与关于代码、构建和工作流规则的实际混合体。协作规则会指明具体要求，从测试计划到提交格式，但相当一部分指令并未说明理由。各项目在 AI 署名问题上立场不一：提交信息规则禁止 AI 署名的情况多于要求署名，而拉取请求规则则更倾向于要求披露 AI 的参与。三分之一的指令集带有与生成器相关的签名、工具插入的内容块、共享文本或其他生成痕迹（摘要在此处截断）。

    arXiv:2610.31715v1 Announce Type: cross  Abstract: Coding agents can be guided with instruction files in a repository. We examine this emerging practice through a census of popular, active GitHub repositories in thirteen languages, treating each repository's root AGENTS.md and CLAUDE.md as one instruction set. Agent instructions appear in of repositories and of the most-starred; where both files exist, CLAUDE.md usually points to AGENTS.md. We analyse statements using model-assisted annotation and human review. The annotations reveal a practical mix of project knowledge and rules about code, building and workflow. Collaboration rules name concrete requirements, from test plans to commit formats, yet of directives give no reason. Projects disagree on AI attribution: commit-message rules more often prohibit it than require it, while pull-request rules more often require disclosure. A third of instruction sets have a generator-associated signature, a tool-inserted block, shared text or an
    
[^34]: 复用、重写还是拒绝：面向重复移动端GUI控制的证据约束轨迹复用

    Reuse, Rewrite, or Reject: Evidence-Constrained Trace Reuse for Repeated Mobile GUI Control

    [https://arxiv.org/abs/2610.31712](https://arxiv.org/abs/2610.31712)

    该论文提出图-文本模板与“记录-模板-回放”框架，通过对UI状态转移进行证据约束验证，实现移动端GUI智能体在重复任务中对先前执行轨迹的选择性复用，从而降低冗余的LLM推理与执行开销。

    

    移动端GUI智能体经常执行具有相似交互流程的重复性任务。复用先前的执行记录可以减少冗余的大语言模型（LLM）推理与执行开销，但用户指令和执行环境的变化可能使先前成功的操作失效。我们系统地将这些变化归纳为六个代表性类别，并观察到此类变化并不一定会使整个流程失效：在针对当前任务上下文和界面状态进行验证后，单个状态转移仍可保持可复用性。基于这一洞察，我们提出了用于选择性执行轨迹复用的图-文本模板。图结构捕获UI状态和动作转移，而文本编码任务意图、流程结构以及用于可变内容的类型化槽位。一个“记录-模板-回放”框架从经过验证的执行中构建模板，并利用当前指令和实时UI状态来复用……

    arXiv:2610.31712v1 Announce Type: cross  Abstract: Mobile GUI agents frequently perform repetitive tasks with similar interaction procedures. Reusing previous executions can reduce redundant LLM inference and execution overhead, but variations in user instructions and execution environments may invalidate previously successful actions. We systematically characterize these variations into six representative categories and observe that such changes do not necessarily invalidate entire procedures: individual transitions can remain reusable when validated against the current task context and interface state. Based on this insight, we propose graph-text templates for selective execution-trace reuse. The graph captures UI states and action transitions, while the text encodes task intent, procedural structure, and typed slots for variable content. A record-template-replay framework constructs templates from verified executions and leverages the current instruction and live UI state to reuse, 
    
[^35]: MBABench-2.0：超越简单可验证性，评估AI智能体构建财务电子表格的能力

    MBABench-2.0: Evaluating AI Agents on Constructing Financial Spreadsheets Beyond Simple Verifiability

    [https://arxiv.org/abs/2610.31711](https://arxiv.org/abs/2610.31711)

    该论文提出MBABench-2.0基准测试，包含101个从零构建财务电子表格的原创任务，突破了以往填空式简单可验证任务的局限，用于更真实地评估AI智能体完成建模决策、格式设计和表格布局等核心电子表格构建工作的能力。

    

    电子表格是现代金融领域的通用语言，支撑着估值、预测、审计以及该行业的众多核心工作流程。随着大语言模型（LLM）智能体在知识工作自动化方面不断进步，电子表格构建成为一个自然而贴切的测试平台。然而，大多数现有的电子表格基准测试采用填空式任务来评估智能体，即智能体在基本完整的电子表格中进行少量编辑。这种评估协议未能捕捉到电子表格工作流程中最核心且最耗时的部分：普通知识工作者从零开始构建工作簿时需要费力完成的建模决策、格式设计和表格布局。我们推出了MBABench-2.0，这是一个包含101个原创电子表格构建任务的基准测试，其中最难的任务预计需要具有3年以上经验的建模者花费33小时才能完成。与编程或数学领域不同——在那些领域中，单元测试或最终答案即可验证智能体的输出——财务电子表格的质量

    arXiv:2610.31711v1 Announce Type: cross  Abstract: Spreadsheets are the lingua franca of modern finance, driving valuation, forecast, audit, and many other core workflows of the industry. As LLM agents advance to automate knowledge work, spreadsheet construction is a natural and pertinent testbed. However, most existing spreadsheet benchmarks evaluate agents on fill-in-the-blank style tasks, where an agent performs minor edits in otherwise complete spreadsheets. This protocol fails to capture the most central and time-consuming part of the spreadsheet workflow: the modeling decisions, format design, and sheet layout regular knowledge workers need to toil through to build a workbook from scratch. We introduce MBABench-2.0, a benchmark of 101 original spreadsheet construction tasks, the hardest of which is estimated to take 33 hours for a modeler with 3+ years of experience. Unlike coding or math, where a unit test or a final answer can verify an agent's output, the quality of a financia
    
[^36]: Compact-Bench：衡量上下文压缩对长程编程智能体的代价

    Compact-Bench: Measuring What Context Compaction Costs Long-Horizon Coding Agents

    [https://arxiv.org/abs/2610.31710](https://arxiv.org/abs/2610.31710)

    提出 Compact-Bench 基准，通过在共享仓库状态下配对“完整历史”与“仅摘要”的延续过程，首次将上下文压缩对长程编程智能体的具体损害（如抹去 5-64% 的成功修复、最多仅保留 16% 的重复细节）从效率指标中隔离出来进行量化评估。

    

    上下文压缩（context compaction）让长程编程智能体能够在超出其上下文窗口后继续运行，但可能会丢失完成任务所需的状态信息。以往的评估衡量的是效率、信息保留程度或端到端成功率，难以隔离某一次特定的历史替换所带来的影响。Compact-Bench 从相同的仓库状态出发，将“完整历史延续”与“仅摘要延续”进行配对对比，并评估重复出现的细节以及连续的多轮生产环境压缩。该基准涵盖 35 个 SWE-bench 修复任务、146 个智能体会话和 106 条生产压缩链。在 14 个摘要模型骨干上，稳定造成的损害范围达到 22 个完整历史成功案例中的 5-64%：Claude Opus 4.8 保留了最多的修复结果，而 DeepSeek-Flash 造成的损害最大。以 Kimi-K3 作为摘要模型时，Kimi Code 提示词保留了最多的任务延续（19/22），而 Claude Code 保留了最多的重复细节（15.0%）。LLM 摘要最多只能保留 16.0% 的重复细节，相比……（原文摘要至此截断）

    arXiv:2610.31710v1 Announce Type: cross  Abstract: Context compaction lets long-horizon coding agents continue beyond their context windows, but can erase the state needed to finish a task. Prior evaluations measure efficiency, retained information, or end-to-end success, leaving the effect of a specific history replacement difficult to isolate. Compact-Bench pairs full-history and summary-only continuations from a shared repository state, and evaluates recurring details and successive production compactions. It covers 35 SWE-bench repairs, 146 agentic sessions, and 106 production chains. Across 14 summarizer backbones, stable damage ranges from 5-64% of 22 full-history successes: Claude Opus 4.8 preserves the most repairs, while DeepSeek-Flash damages the most. With Kimi-K3 as summarizer, the Kimi Code prompt preserves the most task continuations (19/22), whereas Claude Code retains the most recurring details (15.0%). LLM summaries retain at most 16.0% of recurring details, compared w
    
[^37]: 挖掘多智能体工作流的设计空间

    Mining the Design Space of Multi-Agent Workflows

    [https://arxiv.org/abs/2610.31709](https://arxiv.org/abs/2610.31709)

    提出一个两阶段流水线，通过挖掘28,568个可执行工作流的代码与执行轨迹，自动提炼出可复现的多智能体系统设计原则。

    

    设计基于大语言模型（LLM）的多智能体系统（MAS）目前仍依赖于反复试错和非正式的“民间智慧”。人工研究仅测试少数几种通信结构或拓扑（例如串行、并行、共享内存或辩论），而自动化优化器在选定特定任务的工作流后便丢弃搜索轨迹。我们提出了一个两阶段流水线，用于从工作流代码和执行轨迹中挖掘可复现的设计原则。我们将AFlow的MCTS优化器重新应用于六种可组合算子和26个经过确定性验证的Enigmata任务，保留所有中间轮次，从而构建了一个包含28,568个可执行工作流的语料库。随后，一个沙盒化的AutoResearch循环在L0惩罚验证目标下挖掘编排代码和提示模板，并在留出的搜索任务上测试候选设计原则。在300次特征挖掘试验中，我们识别出了在三个层级的留出搜索上持续存在的效应。

    arXiv:2610.31709v1 Announce Type: cross  Abstract: Designing large language model (LLM) multi-agent systems (MAS) still relies on trial and error and informal "folk wisdom." Manual studies test only a few communication structures or topologies (e.g., sequential, parallel, shared-memory, or debate), while automated optimizers discard search trajectories after selecting a task-specific workflow. We present a two-phase pipeline for mining reproducible design principles from workflow code and execution traces. We repurpose AFlow's MCTS optimizer across six composable operators and 26 deterministically verified Enigmata tasks, retaining all intermediate rounds to build a corpus of 28,568 executable workflows. A sandboxed AutoResearch loop then mines orchestration code and prompt templates under an $L_0$-penalized validation objective and tests candidate principles on held-out searches. Across 300 feature-mining trials, we identify effects that persist on held-out searches across three level
    
[^38]: RubricCraft：解法分歧之处，评判标准自现

    RubricCraft: Where Solutions Diverge, Criteria Emerge

    [https://arxiv.org/abs/2610.31707](https://arxiv.org/abs/2610.31707)

    RubricCraft 通过对比多个候选解法之间的满足与违反需求的差异来构建文本评分规则，从而为开放式任务提供显式且可靠的自动化评估标准。

    

    在开放式、产出工件的任务上评估语言智能体十分困难：基于模型的评分器可能表现不一致，而穷尽式的程序化检查往往不切实际。文本化的评分规则能使评估标准显式化，但撰写它们需要领域专业知识。仅从任务描述生成的评分规则可能会遗漏那些在尝试解题的过程中才显现出来的需求和失败模式。我们提出了 RubricCraft，它将评分规则的构建视为一种由候选解引导的自然语言规范生成过程。给定任务描述和独立采样的求解器轨迹，RubricCraft 比较各次尝试如何满足或违反任务要求，并利用这些差异来构建评分标准。它将所得标准应用于收集到的解法，并修正那些忽略错误或惩罚有效替代方案的检查项。由此产生的评分规则能够捕捉部分正确性，而无需（原文在此处截断）

    arXiv:2610.31707v1 Announce Type: cross  Abstract: Evaluating language agents on open-ended, artifact-producing tasks is difficult: model-based graders can be inconsistent, while exhaustive programmatic checks are often impractical. Textual rubrics make evaluation criteria explicit, but writing them requires domain expertise. Rubrics generated from task descriptions alone can miss requirements and failure modes that become apparent in attempted solutions. We introduce RubricCraft, which treats rubric construction as the generation of natural-language specifications guided by candidate solutions. Given a task description and independently sampled solver trajectories, RubricCraft compares how each attempt satisfies or violates the task requirements and uses these differences to construct rubric criteria. It applies the criteria to the collected solutions and revises checks that overlook errors or penalize valid alternatives. The resulting rubrics capture partial correctness without requi
    
[^39]: 提交工单之前：利用智能体式大语言模型助手，将HPC用户支持根植于文档与已解决工单之中

    Before the Ticket: Grounding HPC User Support in Documentation and Resolved Tickets with an Agentic LLM Assistant

    [https://arxiv.org/abs/2610.31703](https://arxiv.org/abs/2610.31703)

    该论文提出了一个完全本地化部署的智能体式LLM助手，通过MCP协议编排实时文档RAG检索与在11,244条历史工单“问题-解决方案”对上QLoRA微调的Llama-3-8B模型并融合二者答案，从而在用户提交工单之前就解答其HPC使用问题。

    

    HPC（高性能计算）帮助台收到的大多数问题，其实早已在计算中心的用户指南或他人此前提交的工单中得到过解答；但用户仍然不断提交新工单，原因在于指南难以导航检索，而工单历史也不允许用户自行搜索。我们提出了一个能够同时从这两种信息来源作答的助手：一个编排器通过模型上下文协议（Model Context Protocol, MCP）调用两个工具——一个针对实时文档的检索增强生成（RAG）工具，以及一个在从历史工单中挖掘出的11,244对“问题-解决方案”数据上使用QLoRA微调的Llama-3-8B模型——并将二者的答案加以融合。由于工单数据不得离开本机构，所有模型均在本地部署，或运行于大学托管的LLM服务之上。在普渡大学罗森先进计算中心的400个保留工单（与训练集互不相交且不含近似重复样本）上进行评估，由两位来自未参与生成任何答案的模型家族的评判者进行打分，结果表明在历史工单解决方案上进行微调……（原文摘要在此处被截断）

    arXiv:2610.31703v1 Announce Type: cross  Abstract: Most questions that reach an HPC help desk have already been answered in the center's user guides or in a ticket someone else filed last semester; users still file new tickets because the guides are hard to navigate and the ticket history is not theirs to search. We present an assistant that answers from both sources: an orchestrator calls two tools over the Model Context Protocol, retrieval-augmented generation (RAG) over the live documentation and a Llama-3-8B model fine-tuned with QLoRA on 11,244 query-resolution pairs mined from historical tickets, and merges their answers. Every model runs on-premises or on the university's hosted LLM service because ticket data may not leave the institution. On 400 held-out tickets from Purdue's Rosen Center for Advanced Computing, disjoint from training and free of near-duplicates, and evaluated by two judges from model families that generated none of the answers, fine-tuning on resolutions rewr
    
[^40]: 理解智能体GUI软件中工具介导的交互失败：一项实证研究

    Understanding Tool-Mediated Interaction Failures in Agentic GUI Software: An Empirical Study

    [https://arxiv.org/abs/2610.31698](https://arxiv.org/abs/2610.31698)

    该研究构建了包含335个真实失败案例的Malva缺陷库，首次系统性地揭示了LLM智能体在通过工具介导与GUI交互时的11种失败模式，并提出MalvaGuard来检测这类广泛存在且往往静默发生的交互缺陷。

    

    大型语言模型（LLM）智能体越来越多地通过实时图形用户界面（GUI）来完成任务。由于模型无法直接操作GUI，这些系统通常依赖工具来捕获屏幕截图或UI树、将模型输出转换为GUI动作，并报告界面是否发生了变化。我们研究了这种工具介导的交互导致智能体对任务的认知与实时GUI产生偏差的失败情况。我们构建了Malva，一个包含来自40个开源应用的335个已确认真实失败案例的缺陷库。这些失败可归纳为11种缺陷模式，其根源在于状态表示、状态转换和同步问题。这些失败广泛存在：90%的被研究应用包含不止一种缺陷类型。它们还常常是静默的：86.6%的失败中出现不正确性，其中包括119个在没有故障终止信号的情况下产生错误行为的案例。我们通过MalvaGuard将这一分类体系进行落地实现。

    arXiv:2610.31698v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly complete tasks through live graphical user interfaces (GUIs). Because the model cannot directly operate the GUI, these systems often rely on tools to capture screenshots or UI trees, translate model outputs into GUI actions, and report whether the interface changed. We study failures where this tool-mediated interaction makes the agent's view of the task diverge from the live GUI. We construct Malva, a defect library with 335 confirmed real-world failures from 40 open-source applications. These failures fall into 11 defect patterns rooted in state representation, state transition, and synchronization problems. They are widespread: 90% of the studied applications contain more than one defect type. They are also often silent: incorrectness appears in 86.6% of failures, including 119 cases that produce wrong behavior without a fail-stop signal. We operationalize this taxonomy with MalvaGuard, a
    
[^41]: SANE：面向LLM评判器的自动化的评分器特定变异测试

    SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges

    [https://arxiv.org/abs/2610.28886](https://arxiv.org/abs/2610.28886)

    SANE是一个基于智能体的自动化框架，能针对特定LLM评判器自动识别其评分能力、生成变异测试配方及预期分数关系，从而将评判器可靠性测试实现全流程自动化，摆脱了人工编写扰动配方和预设评分器类型的限制。

    

    LLM评判器（LLM judges）在基准评估、模型选择、数据筛选和模型训练中扮演着越来越重要的角色。只有当每个评判器能够对影响其评分标准的变动做出可靠响应时，其决策才有价值。现有的可靠性测试方法通过对输入施加具有已知预期评分影响的扰动来进行测试。主流的LLM评判器测试框架将人工编写的扰动配方与合成测试生成相结合，但将针对特定评分器的配方构思排除在自动化流程之外。这限制了它们只能使用固定的扰动方式和预先假设的评分器类型。我们提出了SANE，一个基于智能体（agent）的框架，它将随机评估器的行为需求编码为可复用的智能体指令。给定评分器规范、实现代码和代表性输入，该流水线首先识别评分器特定的能力，然后为每种能力自动推导扰动配方和预期的分数关系。SANE还定义了一个（摘要内容在此处被截断）

    arXiv:2610.28886v1 Announce Type: new  Abstract: LLM judges increasingly mediate benchmark evaluation, model selection, data curation, and training. Their decisions are useful only when each judge responds reliably to changes affecting its scoring criterion. Existing methods for reliability testing perturb inputs with known expected effects on judge scores. Prominent LLM-judge testing harnesses combine human-authored perturbation recipes with synthetic test generation, leaving scorer-specific recipe ideation outside the automated pipeline. This limits their use to fixed perturbations and pre-assumed scorer types. We introduce SANE, an agent-based framework that encodes behavioral requirements for stochastic evaluators as reusable agent instructions. Given a scorer specification, implementation, and representative inputs, the pipeline first identifies scorer-specific capabilities, then automatically derives perturbation recipes and expected score relations for each. SANE also defines a 
    
[^42]: 交互即裁判

    Interaction-as-a-Judge

    [https://arxiv.org/abs/2610.17895](https://arxiv.org/abs/2610.17895)

    提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。

    

    编程智能体的任务是产出复杂的交互式应用，例如网站、桌面应用和游戏。人类在评估这些应用时会以交互方式进行，而当前自动化的LLM评估主要集中于代码分析和单元测试。这种不匹配会导致LLM评判者遗漏那些对人类用户而言一目了然的缺陷。我们研究了“交互即裁判”，这是一种由计算机操作智能体（CUA）直接与应用进行交互的评估范式。我们证明“交互即裁判”能够同时改进交互式应用的评估与生成。为此，我们收集并发布了两个互补的数据集：包含人类偏好及对应智能体操作轨迹的6,858对应用，以及带有经人类验证的需求标签的222个应用。相较于标准的非交互式方法，我们发现“交互即裁判”在预测h……（原文摘要在此处截断）

    arXiv:2610.17895v2 Announce Type: cross  Abstract: Coding agents are tasked with producing sophisticated interactive applications such as websites, desktop applications, and games. While humans assess these applications interactively, current automated LLM evaluations primarily focus on code analysis and unit tests. This mismatch can cause LLM judges to miss failures that are immediately apparent to human users. We study Interaction-as-a-Judge, an evaluation paradigm where computer-use agents (CUAs) directly interact with an application. We demonstrate that Interaction-as-a-Judge can improve both evaluation and generation of interactive applications. To this end, we collect and release two complementary datasets: $6,858$ pairs of applications with human preferences and corresponding agent traces, and $222$ applications with human-verified requirement labels. Relative to standard non-interactive approaches, we find that Interaction-as-a-Judge improves evaluation in terms of predicting h
    
[^43]: 通过有效提示提升大语言模型生成代码的能效

    Improving the Energy-Efficiency of the Code Generated by LLMs through Effective Prompting

    [https://arxiv.org/abs/2610.02571](https://arxiv.org/abs/2610.02571)

    本研究系统评估了21种提示策略，发现有效的提示工程可使大语言模型生成的Python和C++代码能耗分别降低最多25%和17%。

    

    随着AI辅助编程日益成为主流，AI生成软件对环境的影响已成为一个重要的考量因素。这促使人们在评估大语言模型（LLM）生成的代码时，不仅关注功能正确性，还要考虑执行效率和能耗。然而，尽管代码生成技术取得了长足进步，前沿LLM却很少基于其生成代码的能效进行评估。在这项工作中，我们对21种用于节能代码生成的提示策略进行了全面评估，并从中筛选出8种策略，在10个广泛使用的开源权重和专有LLM上进行评估。我们以基线提示为参照，评估了这些策略在Python和C++代码生成中的有效性。在所有被评估的模型中，所选的提示策略使Python代码生成的能耗最多降低25%，C++代码生成的能耗最多降低17%。在模型层面，Python能耗降低……

    arXiv:2610.02571v1 Announce Type: cross  Abstract: As AI-assisted programming becomes increasingly mainstream, the environmental impact of AI-generated software has emerged as an important consideration. This motivates evaluating LLM-generated code beyond functional correctness by considering execution efficiency and energy consumption. However, despite substantial advances in code generation, frontier LLMs are rarely evaluated based on the energy efficiency of the code they produce. In this work, we conduct a comprehensive evaluation of 21 prompting strategies for energy-efficient code generation and identify 8 strategies for evaluation across 10 widely used open-weight and proprietary LLMs. We evaluate their effectiveness for both Python and C++ code generation relative to a baseline prompt. Across the evaluated models, the selected prompting strategies achieved energy reductions of up to 25% for Python and 17% for C++ code generation. At the model level, Python energy reductions rea
    
[^44]: 两人亦是“众”：ADHD开发者的人际与AI共同在场

    Two's a Crowd: Human and AI-Based Copresence for Developers with ADHD

    [https://arxiv.org/abs/2609.21254](https://arxiv.org/abs/2609.21254)

    本研究通过对14名ADHD软件工程师的半结构化访谈，首次系统考察了他们如何在人际共同在场实践（如body doubling、结对编程）与新兴代理型AI编程助手之间开展协作，揭示了AI助手正在重塑ADHD开发者的协作动态与执行功能支持方式。

    

    有效的协作与沟通对开发者的生产力和福祉至关重要，但仍然受到注意力、内在动机和人际问责等人为因素的制约。这些制约对自我认同为注意缺陷多动障碍（ADHD）的开发者尤为重要，他们在现代混合办公环境中长期面临各种环境障碍。虽然ADHD开发者常常依赖协作性的共同在场实践（如“陪伴工作”（body doubling）或结对编程）来支持执行功能，但近来代理型AI编程助手的兴起已开始重塑这些协作动态。为了探究ADHD开发者如何参与人际与AI共同在场实践，我们对14名患有ADHD的软件工程师进行了半结构化访谈。我们的研究结果表明，虽然传统的人与人共同在场能够提供关键的社会支持与上手引导（原文摘要至此截断）。

    arXiv:2609.21254v1 Announce Type: cross  Abstract: Effective collaboration and communication are vital to developer productivity and well-being, yet remain constrained by human factors such as attention, intrinsic motivation, and interpersonal accountability. These constraints are particularly vital for developers identifying with Attention Deficit Hyperactivity Disorder (ADHD), who navigate persistent environmental barriers in modern hybrid workplace settings. While developers with ADHD frequently rely on collaborative copresence practices (such as body doubling or pair programming) to support executive function, the recent emergence of agentic AI coding assistants has begun reshaping these collaborative dynamics. To investigate how developers with ADHD engage in human and AI-based copresence practices, we conducted semi-structured interviews with 14 software engineers with ADHD. Our findings reveal that while traditional human-human copresence provides critical social support and onb
    
[^45]: 循环工程：构建模块、采用与影响

    Loop Engineering: Building Blocks, Adoption, and Impact

    [https://arxiv.org/abs/2608.21884](https://arxiv.org/abs/2608.21884)

    本文首次探索性回顾了“循环工程”这一新兴实践，即开发者设计系统自动触发和停止智能体运行，并总结了其核心构建模块（如停止条件、状态文件和验证子智能体），但未量化其实际采用率。

    

    arXiv:2608.21884v1 公告类型：新 摘要：在过去的几个月里，开发者指导智能体AI编码工具的方式已跨越多个抽象层次，从措辞提示到工程上下文，再到配置模型周围的框架。2026年6月，从业者开始描述一个称为“循环工程”的进一步层次：开发者不再交互式地提示智能体，而是设计系统让智能体自动提示自身。这些系统按计划或仓库事件启动智能体运行，并在机器可检查条件满足时停止它们。该术语迅速传播，伴随着大胆的主张和直言不讳的怀疑，但其在软件项目中的采用情况尚未被量化。我们提出对新兴灰色文献的探索性回顾，这些文献大致一致认为一个良好设计的循环包含：由机器可检查停止条件约束的触发式智能体运行、持久状态文件、验证子智能体、令牌预算，以及定义好的升级到人工处理的节点。

    arXiv:2608.21884v1 Announce Type: new  Abstract: Over the past months, the way developers direct agentic AI coding tools has moved up several levels of abstraction, from phrasing prompts to engineering context to configuring the harness around the model. In June 2026, practitioners began to describe a further level called loop engineering: Instead of prompting an agent interactively, developers design systems that prompt agents for them. These systems start agent runs on a schedule or on repository events and stop them when a machine-checkable condition holds. The term spread rapidly, accompanied by bold claims and vocal skepticism, but its adoption in software projects has not been measured. We present an exploratory review of the emerging gray literature, which largely agrees on what a well-engineered loop contains: triggered agent runs bounded by machine-checkable stop conditions, persistent state files, verifier sub-agents, token budgets, and defined points of escalation to humans.
    
[^46]: Memoir：为静态应用安全测试工具学习、验证与演化误报记忆

    Memoir: Learning, Verifying, and Evolving False-Positive Memories for Static Application Security Testing Tools

    [https://arxiv.org/abs/2608.09181](https://arxiv.org/abs/2608.09181)

    提出Memoir框架，通过将SAST工具的历史误报警报转化为可学习、可验证并持续演化的结构化语义记忆，有效识别并减少误报，克服了现有方法难以跨工具泛化、知识无法动态更新的局限。

    

    静态应用安全测试（SAST）工具已成为现代安全软件开发中不可或缺的组成部分。然而，这些工具经常产生误报（FP）警报，带来了大量的人工审查成本，并降低了开发人员的信任度。现有的误报消减方法仍面临两个主要挑战：其一，SAST工具与漏洞类别之间的巨大差异，使得这些方法难以从历史误报中学习重复出现的模式；其二，这些方法所使用的知识大多是静态的，无法随着新验证案例的积累而更新。为应对这些挑战，我们提出了Memoir，一个通过将历史误报警报转化为可复用的语义记忆来识别误报的记忆驱动框架。该框架包含两个关键模块：首先，历史语义记忆构建模块将历史误报警报转换为结构化的语义记忆……

    arXiv:2608.09181v2 Announce Type: replace  Abstract: Static Application Security Testing (SAST) tools have become indispensable in modern secure software devel- opment. However, these tools often generate false-positive (FP) alerts, imposing substantial manual inspection costs and reducing the trust from developers. Existing FP reduction methods still face two primary challenges. First, the large differences among SAST tools and vulnerability categories make it difficult for these methods to learn recurring patterns in historical false positives. Moreover, the knowledge used by these methods are largely static and cannot be updated as newly validated cases accumulate. To address these challenges, we propose Memoir, a memory- driven framework for identifying false positives by transform- ing historical FP alerts into reusable semantic memories. It consists of two key modules. First, historical semantic memory construction converts historical FP alerts into structured semantic memories t
    
[^47]: 微服务可用性的简化随机模型：识别与形式化分析算法

    A Reduced Stochastic Model of Microservice Availability: Algorithms for Identification and Formal Analysis

    [https://arxiv.org/abs/2607.00740](https://arxiv.org/abs/2607.00740)

    本文提出从遥测数据和声明的操作需求中识别简化随机可用性模型的算法，通过形式化分析给出可达到的概率界限，并证明该简化模型能够保留完整模型中对“完成-截止期限”事件的评估范围。

    

    估算微服务的运行可用性，需要将运行时观测数据与所需工作在截止期限内的完成情况联系起来。详细的架构模型提供了一条途径，但其重构与分析的代价可能很高。我们提出了从遥测数据和声明的操作需求中识别并分析简化随机可用性模型的算法。该识别过程构建执行状态的联合分布规律，并在观测不完整的情况下保留兼容的分布规律。形式化的结果确立了可达到的概率界限，以及直接识别“完成与截止期限”表示所需满足的条件。这一构造保留了在完整模型中评估同一“完成与截止期限”事件所获得的取值范围；另有一个独立的联合比较用于描述被舍弃的结构信息。一项独立研究涵盖了三个微服务应用和18次实验活动。在完整……

    arXiv:2607.00740v2 Announce Type: replace  Abstract: Estimating microservice operation availability requires connecting runtime observations to completion of the required work within a deadline. Detailed architectural models provide one route, but their reconstruction and analysis can be costly. We present algorithms for identifying and analysing a reduced stochastic availability model from telemetry and declared operation requirements. The identification procedure constructs a joint law of execution states and retains compatible laws when observations are incomplete. Formal results establish attainable probability bounds and conditions for identifying a completion-and-deadline representation directly. This construction preserves the range obtained by evaluating the same completion-and-deadline event in the full model; a separate joint comparison describes the structural information discarded. An independent study covered three microservice applications and 18 campaigns. On complete co
    
[^48]: 从研究到实践：工业界自动驾驶系统测试的交互式快速综述

    From Research to Practice: An Interactive Rapid Review of Autonomous Driving System Testing in Industry

    [https://arxiv.org/abs/2605.00531](https://arxiv.org/abs/2605.00531)

    本研究通过与21名汽车行业从业者合作的交互式快速综述，识别出自动驾驶系统测试的12个关键挑战，重点聚焦端到端系统的测试方法与完备性，并评估了17项相关研究在工业实践中的适用性，是首个由从业者驱动的研究综述。

    

    自动驾驶系统（ADS）正越来越多地部署于真实交通环境中，但由于开放环境、复杂场景以及缺乏既定的测试流程和验收标准，测试仍然存在根本性的挑战。尽管已有大量研究，学术界的进展与其在工业实践中的适用性之间仍存在差距。为了解决这一问题，我们与一家领先汽车公司的21名从业者合作，开展了一项交互式快速综述。从业者识别出自动驾驶系统测试中的12个关键挑战，并将其中两个列为最关键的问题，即端到端（E2E）自动驾驶系统的测试方法及测试完备性。我们分析了与这两个挑战相关的17项研究，其中大多数聚焦于生成关键测试场景，并随后评估了这些研究在实践中的相关性和适用性。本研究首次提供了由从业者驱动的、对当前自动驾驶系统测试研究的综述与评估。

    arXiv:2605.00531v2 Announce Type: replace  Abstract: Autonomous driving systems (ADS) are increasingly deployed in real traffic, yet testing remains fundamentally challenging due to open environments, complex scenarios, and the lack of established processes and acceptance criteria. Despite extensive research, a gap persists between academic advances and their applicability in industrial practice. To address this, we conduct an interactive rapid review in collaboration with 21 practitioners from a leading automotive company. Practitioners identified 12 key challenges in ADS testing, and prioritised two as the most critical issues, namely approaches to and completeness of testing for End-to-End (E2E) ADS. We analyzed 17 research studies relevant to these two challenges, most of which focus on generating critical testing scenarios, and subsequently assessed their relevance and applicability in practice. Our study provides the first practitioner-driven review and evaluation of current ADS 
    
[^49]: daVinci-Env：大规模开放式软件工程环境合成

    daVinci-Env: Open SWE Environment Synthesis at Scale

    [https://arxiv.org/abs/2603.13023](https://arxiv.org/abs/2603.13023)

    提出了 OpenSWE——目前最大的完全开源透明的软件工程智能体训练框架，包含 45,320 个覆盖 1.28 万仓库的可执行 Docker 环境，并借助 64 节点集群上的多智能体合成流水线自动化构建。

    

    训练能力强大的软件工程（SWE）智能体需要大规模、可执行且可验证的环境，以为迭代式代码编辑、测试执行和方案改进提供动态反馈回路。然而，现有的开源数据集在规模和仓库多样性方面仍然有限，而工业界的解决方案则不透明且基础设施未公开，为大多数学术研究团队设置了难以逾越的障碍。我们提出了 OpenSWE，这是目前最大的完全透明的 Python 软件工程智能体训练框架，包含 45,320 个可执行的 Docker 环境，覆盖超过 12.8k 个代码仓库，所有 Dockerfile、评估脚本和基础设施均完全开源以保证可复现性。OpenSWE 通过部署在 64 节点分布式集群上的多智能体合成流水线构建，自动化完成仓库探索、Dockerfile 构建、评估脚本生成和迭代测试分析。

    arXiv:2603.13023v3 Announce Type: replace-cross  Abstract: Training capable software engineering (SWE) agents demands large-scale, executable, and verifiable environments that provide dynamic feedback loops for iterative code editing, test execution, and solution refinement. However, existing open-source datasets remain limited in scale and repository diversity, while industrial solutions are opaque with unreleased infrastructure, creating a prohibitive barrier for most academic research groups. We present OpenSWE, the largest fully transparent framework for SWE agent training in Python, comprising 45,320 executable Docker environments spanning over 12.8k repositories, with all Dockerfiles, evaluation scripts, and infrastructure fully open-sourced for reproducibility. OpenSWE is built through a multi-agent synthesis pipeline deployed across a 64-node distributed cluster, automating repository exploration, Dockerfile construction, evaluation script generation, and iterative test analysi
    
[^50]: 上下文记忆虚拟化：面向大语言模型智能体的基于有向无环图的状态管理与结构无损裁剪

    Contextual Memory Virtualisation: DAG-Based State Management and Structurally Lossless Trimming for LLM Agents

    [https://arxiv.org/abs/2602.22402](https://arxiv.org/abs/2602.22402)

    提出了上下文记忆虚拟化（CMV）系统，借鉴操作系统虚拟内存思想，将LLM会话历史建模为带快照、分支、裁剪原语的DAG以实现跨并行会话的上下文复用，并通过三遍结构无损裁剪算法在逐字保留所有对话内容的前提下将令牌开销平均降低20%、最高降低86%。

    

    随着大语言模型执行长程推理任务，它们会在上下文窗口内累积大量状态——如架构映射、权衡决策、代码库约定等。当会话达到上下文限制并进行有损压缩时，这些理解便会丢失。我们提出了上下文记忆虚拟化（CMV），一个将大语言模型累积的理解视为版本控制状态的系统。借鉴操作系统的虚拟内存机制，CMV将会话历史建模为有向无环图（DAG），并形式化定义了快照、分支和裁剪等原语，从而实现跨独立并行会话的上下文复用。我们引入了一种三遍式结构无损裁剪算法，该算法逐字保留每条用户消息和助手回复，同时通过剥离机械性冗余（如原始工具输出、基础……）使令牌数量平均减少20%，对于开销较大的会话最多可减少86%。

    arXiv:2602.22402v2 Announce Type: replace-cross  Abstract: As large language models engage in extended reasoning tasks, they accumulate significant state -- architectural mappings, trade-off decisions, codebase conventions -- within the context window. This understanding is lost when sessions reach context limits and undergo lossy compaction. We propose Contextual Memory Virtualisation (CMV), a system that treats accumulated LLM understanding as version-controlled state. Borrowing from operating system virtual memory, CMV models session history as a Directed Acyclic Graph (DAG) with formally defined snapshot, branch, and trim primitives that enable context reuse across independent parallel sessions. We introduce a three-pass structurally lossless trimming algorithm that preserves every user message and assistant response verbatim while reducing token counts by a mean of 20% and up to 86% for sessions with significant overhead by stripping mechanical bloat such as raw tool outputs, base
    
[^51]: 输出邻居中的代码坏味道对类稳定性的影响

    The Influence of Code Smells in Efferent Neighbors on Class Stability

    [https://arxiv.org/abs/2602.12950](https://arxiv.org/abs/2602.12950)

    本研究基于152个热门GitHub Java项目一年的提交历史，首次揭示了类的输出邻居（即其所依赖的类）中存在的代码坏味道会通过涟漪效应影响该类自身的稳定性。

    

    理解驱动代码不稳定性的因素对于有效的软件维护至关重要，因为不稳定的类需要更大或更频繁的修改，并会增加产生意外副作用的风险。尽管代码坏味道被广泛认为会损害可维护性，但以往大多数关于稳定性的研究仅考察被修改类内部的坏味道。然而在实践中，即使某个类本身是“干净”的，它也可能因为其输出邻居（即其所依赖的类）因沿静态依赖传播的涟漪效应而被修改而发生变化。当输出邻居存在代码坏味道时，这种涟漪效应可能会更加严重。因此，本研究旨在探究类的输出邻居中的代码坏味道是否会影响该类自身的稳定性。我们从152个GitHub上星标数最高的Java项目中挖掘了一年的提交历史，检测了代码坏味道和静态依赖，并对其与类稳定性之间的关联进行了建模。我们的研究……

    arXiv:2602.12950v3 Announce Type: replace  Abstract: Understanding what drives code instability is essential for effective software maintenance, as unstable classes require larger or more frequent edits and increase the risk of unintended side effects. Although code smells are widely believed to harm maintainability, most prior stability studies examine only the smells within the class being modified. In practice, however, a class can change because its efferent neighbors (i.e., the classes it depends on) are modified due to ripple effects that propagate along static dependencies, even if the class itself is clean. Such ripple effects may be more severe when efferent neighbors exhibit code smells. Therefore, this study investigates whether code smells in a class's efferent neighbors affect its stability. We mined one year of commit history from 152 top-starred GitHub Java projects, detected code smells and static dependencies, and modeled their associations with class stability. Our re
    

