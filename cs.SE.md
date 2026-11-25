# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PORTA: Systematic Discovery of Vulnerabilities in Identity Brokers](https://arxiv.org/abs/2611.33015) | 本论文提出PORTA，首次通过定义刻画身份代理“同时作为依赖方和身份提供商”这一独特双重性的运行不变量，实现了对身份代理漏洞的系统性发现，且所考察的全部60个既往CVE均至少违反一个不变量。 |
| [^2] | [SpecEngineeringBench: Evaluating Coding Agents as Autonomous Specification Engineers](https://arxiv.org/abs/2611.32973) | 该论文提出了 SpecEngineeringBench 基准，通过从78个开源 Python 仓库构建的100个实例，以测试套件生成和缺陷发现两项独立任务，评估编码智能体在缺乏明确线索的情况下自主将功能需求转化为测试并主动发现缺陷的能力。 |
| [^3] | [ShellBench: What Matters More for AI Agents, Models, or Harness for Everyday Tasks](https://arxiv.org/abs/2611.32946) | 该论文推出ShellBench基准，通过对12个模型、5种智能体框架和3个推理水平共53,100次实验的评估发现，模型选择对日常任务性能的影响大于框架选择，但领先模型之间差距仅1.1个百分点，且高性能模型间的成本差异可达六倍之多。 |
| [^4] | [How Use Case Tables Enable the Elicitation of Explanation Needs](https://arxiv.org/abs/2611.32809) | 本研究探讨了用例表能否作为原型测试的替代方案，在软件开发早期获取用户的解释需求，并通过让用户阅读用例描述的实证研究加以验证。 |
| [^5] | [Energy as a Service-Level Objective: Transaction-Level Energy Attribution](https://arxiv.org/abs/2611.32731) | 本论文借鉴SRE的服务水平目标方法论，提出事务级能耗归因方法，将每次成功事务的能耗作为服务水平目标，使能耗成为可操作的工程评估工具而非单纯的报告指标。 |
| [^6] | [AI-Generated Code Issues: A Taxonomy from Repository Histories, AI Conversations, and Developer Interviews](https://arxiv.org/abs/2611.32676) | 该研究通过挖掘GitHub提交历史、分析开发者与AI工具的对话并访谈开发者三个互补数据源，首次系统构建了AI生成代码质量问题在真实软件开发中的分类体系，以帮助开发者改进代码审查与测试等质量保障实践。 |
| [^7] | [The Elephant-Goldfish Model for software design review: what a fresh reviewer can see](https://arxiv.org/abs/2611.32668) | 该研究通过对47份植入缺陷的设计文档进行实验证明，让全新AI评审者接触设计起草过程的对话，可将遗漏决策的检出率从62.4%提升至94.8%，同时不影响文档可见缺陷的检测效果。 |
| [^8] | [Generating Property-Based Regression Tests from Source Code Semantics](https://arxiv.org/abs/2611.32636) | 本文提出通过多路径符号执行直接从源代码语义自动推导基于属性的回归测试，克服了传统单元测试可能漏检回归错误以及手动编写属性和输入生成器困难的问题。 |
| [^9] | [VISE: Verifier-Independent Skill Self-Evolution with Feedback from Tool Implementations](https://arxiv.org/abs/2611.32480) | VISE提出了一种无需任务特定验证器的技能自进化方法，通过从工具源代码中提取能力、输入要求、使用条件和输出格式等信息来指导技能工作流的修订，在两个基准上将平均任务成功率提升了9.14-13.35个百分点。 |
| [^10] | [Does Research-Paper Context Improve LLM-Generated Quadruped Navigation? An Exploratory Simulation Study](https://arxiv.org/abs/2611.32381) | 这项探索性仿真研究发现，为 LLM 提供特定研究论文作为上下文并未改进四足机器人的导航代码，反而在迷宫最短路径中途被阻断的场景下显著降低了生成控制器的鲁棒性。 |
| [^11] | [Validating Multi-Model AI-Generated Acceptance Criteria in Requirements Engineering](https://arxiv.org/abs/2611.32374) | 本文提出一种结合检索增强生成、多模型生成与人机协同工作流的面向验证方法，用于解决需求工程中AI生成的验收标准缺乏有效性验证的问题。 |
| [^12] | [SCOPE: Evaluating and Enhancing Permission Explanation Transparency in Mobile Apps](https://arxiv.org/abs/2611.32122) | 该论文通过对600款主流移动应用的实证研究揭示了权限说明存在的普遍缺陷（如缺少解释、表述模糊、未披露第三方SDK数据访问），并提出了名为SCOPE的自动化多智能体框架，用于系统性评估权限说明合规性并生成优化建议。 |
| [^13] | [Automated Smoke Testing for Multi-ending Game Quests via Large Language Models](https://arxiv.org/abs/2611.32089) | 提出了EndWeaver，一个专门面向多结局游戏任务的自动化冒烟测试框架，利用大语言模型进行多层次规划并结合状态转换图的动态构建，实现对分支叙事的高效探索和结局的快速验证。 |
| [^14] | [Online Environment Repair for Coding Agents](https://arxiv.org/abs/2611.32037) | 该论文提出“在线环境修复”方法，在强化学习训练过程中持续检测并修复编程智能体环境中可被利用的验证器缺陷（如不完整的单元测试），在不惩罚有效解法、不丢弃高难度任务的前提下，从源头抑制奖励作弊行为。 |
| [^15] | [DesignEvo: Recovering and Tracking Repository-Level Implemented Design Systems Across Commits](https://arxiv.org/abs/2611.32020) | DesignEvo通过融合截图、DOM结构、元素几何与样式信息，并将视觉-语言模型判断约束在程序生成的候选区域内，实现了跨提交在仓库级别自动恢复与追踪已实现的设计系统及其生命周期状态。 |
| [^16] | [When Online Adaptation of Resilience Policies Hurts: Safe Exploration and Measurement Pitfalls in Resilience4j](https://arxiv.org/abs/2611.32015) | 该论文提出AdaptR4j系统，利用上下文赌博机在运行时动态选择Resilience4j的熔断器/重试/舱壁策略，并通过安全策略菜单和有界奖励的安全探索两项保障机制，克服了朴素在线学习在探索阶段拒绝流量的陷阱，在所有可赢得的故障场景中超越生产默认配置。 |
| [^17] | [Not Just Any Example: Security Requirements and Fix Patterns for Guiding LLM Vulnerability Repair](https://arxiv.org/abs/2611.31993) | 该论文通过对348个开源仓库中3,816个安全补丁的大规模实证研究，提炼出与漏洞安全影响相关联的具体修复模式，并展示了如何利用这些模式引导大语言模型生成真正满足安全需求的漏洞修复补丁。 |
| [^18] | [CAPE: Coordinating Calibration and Search for Tool-Use Policies](https://arxiv.org/abs/2611.31938) | CAPE是一个无需训练的工具使用策略编译器，它通过协调校准与搜索，始终保持可执行策略，并将校准资源集中于能改进该策略的调用上，从而高效应对配置不可见工具导致的调用失败问题。 |
| [^19] | [Shuttle: Dependency-Attributed Workflow Evolution for Smart Contract Auditing](https://arxiv.org/abs/2611.31929) | Shuttle通过溯源图归因定位工作流中线索丢失的环节并保护已有检测成果，实现受约束的智能合约审计工作流演化，避免了无约束改写导致的性能退化和泛化能力下降。 |
| [^20] | [(No) Separation of Concerns: Philosophical Concerns about Agent Ethics and Implications for SE Practitioners](https://arxiv.org/abs/2611.31927) | 该研究通过系统性梳理PhilPapers中哲学家关于AI智能体伦理的文献，提炼出八个主题和85个编码，将其转化为软件工程研究议程以及使用编程智能体团队的实践启示。 |
| [^21] | [Spec-Test Coverage Gap: Detecting Overspecified Tests in SWE-Bench-Style Benchmarks](https://arxiv.org/abs/2611.31839) | 本文提出规格-测试覆盖差距（STCG）这一1-4分制新指标，利用两个LLM评判者自动检测SWE-bench风格基准中的过度指定测试，从而取代昂贵的专家审查，识别会导致正确的替代修复方案被误判失败的测试用例。 |
| [^22] | [Agentic Discovery of Soundness Bugs in Safe Rust Bindings](https://arxiv.org/abs/2611.31813) | 本文提出将LLM智能体作为Rust安全封装接口的审计员来发现健全性漏洞，定义了此类接口缺陷的分类体系，并通过真实crate的重复审计实验发现：智能体能自发找到未定义行为和与底层库的偏差，却难以发现不完整的不安全契约。 |
| [^23] | [Translation Validation of SystemVerilog-to-SystemC RTL Conversion by Reading the Generated C++](https://arxiv.org/abs/2611.31674) | 本文提出了一种SystemVerilog到SystemC RTL转换的翻译验证方法：按照C++17和SystemC的真实语义解读生成的C++代码、将未定义行为转化为证明义务，并通过显式化所有位宽变化来证明生成代码与源RTL网表等价。 |
| [^24] | [GoldiBench: Right Tools, Wrong Amount -- Benchmarking Agents on Effect Judgement in Professional Editing](https://arxiv.org/abs/2611.31590) | 论文提出GoldiBench基准，通过500个专业编辑任务将“工具操作”与“效果判断（用量把握）”分离评估，发现智能体操作能力很强，但难以准确判断实现目标视觉效果所需的工具用量。 |
| [^25] | [Agent Recovery Gone Wrong: Measuring Harm Beyond Duplicate Calls](https://arxiv.org/abs/2611.31554) | 该论文首次在自然发生的失败场景下量化了代码执行型LLM智能体恢复操作对外部状态的危害，发现约24.6%的结果可确定的部分执行会导致在运行结束时仍持续存在的有害恢复，其中仅三分之二表现为重复调用，其余则表现为删除用户数据或错误的首次写入，说明仅检测重复调用不足以捕捉全部危害。 |
| [^26] | [Below the Stopwatch: Checking the Performance Claims of Coding Agents in the Kernel](https://arxiv.org/abs/2611.31460) | 提出了Below流水线，通过提取编码代理性能PR所声称的优化机制并用硬件计数器与LTTng内核追踪验证其应有的内核级信号，发现大多数性能声明缺乏实证测量，且相当一部分PR并未带来真实的性能提升。 |
| [^27] | [CrossCheck II: A Compiler Plugin for Certified Cost Bounds of Real-Time C and C++ Code, and its Evaluation](https://arxiv.org/abs/2611.31449) | 该论文提出了Clang编译器插件CrossCheck II，能够自动为实时C/C++代码计算成本界限并生成经Lean 4验证的成本证书，使界限的正确性不依赖于分析器和求解器，且在TACLeBench基准上无需注解即可界定220个循环中的142个。 |
| [^28] | [Decision Models for Definition-Driven GitHub Actions Workflow Smell Detection](https://arxiv.org/abs/2611.31421) | 本文提出用面向决策的模型依据可复用的自然语言坏味定义来检测 GitHub Actions 工作流坏味，发现检测效果高度依赖决策策略，其中 JEV 1.13 达到 79.69% 的微平均 F1，并在评分可靠性、可重复性和鲁棒性上表现更优。 |
| [^29] | [Diffs Are Cheap, Show Me the Trajectory: An Empirical Study of Trajectory-Informed Code Review](https://arxiv.org/abs/2611.31316) | 该论文通过实证研究系统探索了编码智能体的开发轨迹（包含中间决策、失败尝试和验证证据）能否帮助审查智能体超越仅检查最终代码差异来发现代码问题，并构建了包含339个实现、113个DeepSWE任务轨迹和882条参考审查意见的数据集进行评估。 |
| [^30] | [The Defect Is in the Arithmetic Between the Files: Static Detection of Exactly-Once Violations in Payment Pipelines](https://arxiv.org/abs/2611.31298) | 该论文提出一种静态检测方法，通过收集分散在不同文件中的幂等键保留时长与记录最大过期时间等参数并分析其数值关系，检测支付流水线重启后可能违反“恰好一次”语义的隐患，并将515个发现中的335个转化为可运行的故障复现场景，其中245个场景被证实会导致客户被重复扣款。 |
| [^31] | [LITMUS - Literature Informed Testbed for Mapping Usecases and Synthesizing Risk-Cards](https://arxiv.org/abs/2611.29034) | 本文提出LITMUS，一个结构化流水线，能够根据给定用例自动生成定制化数据集与风险卡，并利用同行评审文献提供可追溯证据，将每个风险与具体用例紧密关联，弥补了现有方法忽视用例场景的不足。 |
| [^32] | [Ranking Function Generation with LLMs for Termination](https://arxiv.org/abs/2610.35523) | 该论文提出RAINFALL神经符号框架，通过将排序函数合成转化为带槽位约束的代码补全任务，并结合符号验证器的反馈迭代改进，利用大语言模型自动生成可验证的排序函数以证明循环终止性。 |
| [^33] | [DeepAVR: Deep Analysis via Multi-Agent Collaboration for Automated Vulnerability Repair](https://arxiv.org/abs/2610.32345) | DeepAVR提出了一种多智能体协作框架，将中间修复结果转化为结构化诊断反馈，并在故障定位、补丁生成、编译/安全/功能验证与补丁审查等专门角色之间流转反馈，从而实现端到端的深度漏洞分析与自动化修复。 |
| [^34] | [SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?](https://arxiv.org/abs/2609.21190) | 该论文提出Benchproofer流水线，将SWE-bench中的真实编码任务转化为经过机器校验证明的形式化验证任务，构建了包含500个真实问题的SWE-Proof基准，用形式化验证取代不完整的测试来严格评估语言模型解决真实软件工程问题的能力。 |
| [^35] | [PRAXIS: Graph-Grounded Tacit Knowledge for Domain Code Generation](https://arxiv.org/abs/2608.19784) | PRAXIS通过模拟人类开发工作流，从代码库中提取隐性知识并将其组织为图结构，使LLM代理能够有效生成领域特定代码。 |
| [^36] | [Graph Construction and Matching for Imperative Programs using Neural and Structural Methods](https://arxiv.org/abs/2604.26578) | 该论文提出一条将命令式程序及其标注转换为带类型属性图的流水线，通过结合抽象语法树解析与SentenceTransformer、CodeBERT等模型的语义嵌入，在C/ACSL、Java/JML和Dafny等多种语言上实现了跨语言、跨标注风格的一致性图表示，为验证产物复用和近似图匹配奠定了实用基础。 |
| [^37] | [ChroniUXMag: A Persona-Driven Framework for Inclusive mHealth Requirements Engineering](https://arxiv.org/abs/2511.18634) | ChroniUXMag是一个建立在InclusiveMag和GenderMag基础之上、以用户画像驱动的框架，通过文献综述、焦点小组、访谈和调查识别包容性需求维度，并将其整合到认知走查中，用于改进慢性病管理移动健康应用的包容性需求工程。 |
| [^38] | [Web Fuzzing Commons, Dataset, Guidelines and Evaluations to Support Experimentation in REST API Fuzzing](https://arxiv.org/abs/2509.01612) | 本文提出了Web Fuzzing Commons（WFC）、Web Fuzzing Dataset（WFD）和Web Fuzzing Evaluations（WFE）三个开源组件，通过解决API身份验证处理、故障类型编目比较和标准化案例研究三大挑战，为REST API模糊测试研究提供了可公平比较的实验基础设施。 |

# 详细

[^1]: PORTA: 身份代理中漏洞的系统性发现

    PORTA: Systematic Discovery of Vulnerabilities in Identity Brokers

    [https://arxiv.org/abs/2611.33015](https://arxiv.org/abs/2611.33015)

    本论文提出PORTA，首次通过定义刻画身份代理“同时作为依赖方和身份提供商”这一独特双重性的运行不变量，实现了对身份代理漏洞的系统性发现，且所考察的全部60个既往CVE均至少违反一个不变量。

    

    身份代理（如Keycloak、Vault和Dex）将来自上游身份提供商的身份验证凭证转换为下游服务的会话。由于身份代理在身份验证和信任中介中处于核心地位，代理中的一个漏洞就会危及其背后的所有下游服务。事实上，最近的安全事件表明，即使是广泛采用的平台也无法免受此类问题的影响，这凸显了进行系统性分析的迫切需求。为了构建这样的分析，本工作首先为身份代理的运行定义了不变量。这些不变量并未被现有的身份验证规范完全覆盖，主要是因为身份代理具有独特的双重性，即同时充当依赖方和身份提供商。这些不变量刻画了身份代理在整个工作流程中必须遵守的属性，因为我们所考察的全部60个既往CVE都至少违反了其中一个不变量。在此基础之上……

    arXiv:2611.33015v1 Announce Type: cross  Abstract: Identity brokers such as Keycloak, Vault, and Dex translate authentication proofs from an upstream identity provider into sessions for downstream services. Due to their central role in authentication and trust mediation, a single vulnerability in a broker compromises every downstream service behind it. In fact, recent incidents have demonstrated that even widely adopted platforms are not immune to such issues, highlighting the critical need for systematic analysis. To build such an analysis, this work first defines invariants for broker operation. These invariants are not completely covered by existing authentication specifications, primarily because brokers are unique in their duality, i.e., acting simultaneously as a relying party and an identity provider. The invariants capture the properties that identity brokers must uphold throughout their workflows, as all 60 previous CVEs we consider violate at least one invariant. Within these
    
[^2]: SpecEngineeringBench：将编码智能体评估为自主规格工程师

    SpecEngineeringBench: Evaluating Coding Agents as Autonomous Specification Engineers

    [https://arxiv.org/abs/2611.32973](https://arxiv.org/abs/2611.32973)

    该论文提出了 SpecEngineeringBench 基准，通过从78个开源 Python 仓库构建的100个实例，以测试套件生成和缺陷发现两项独立任务，评估编码智能体在缺乏明确线索的情况下自主将功能需求转化为测试并主动发现缺陷的能力。

    

    自主规格工程要求编码智能体将功能需求转化为可执行的测试，并利用测试结果来发现和验证缺陷。然而，现有的基准在评估编码智能体时针对的是明确指定的测试生成任务，通常提供问题描述、指定的测试入口点或需要检查的文件，这使得“确定测试什么以及在何处调查”的能力未能得到充分评估。依赖此类线索绕过了将需求映射到实现并主动发现未知缺陷的挑战。我们提出了 SpecEngineeringBench，它包含来自16个应用类别的78个开源 Python 仓库的100个实例，每个实例包含多个已记录的缺陷。两个独立任务评估互补的能力：生成测试套件，以及通过执行分析发现缺陷。该评估结合了测试有效性、缺陷（原文在此处截断）。

    arXiv:2611.32973v1 Announce Type: new  Abstract: Autonomous specification engineering requires coding agents to translate functional requirements into executable tests and use the results to discover and validate defects. However, existing benchmarks evaluate coding agents on explicitly specified test generation tasks, commonly providing an issue description, a designated test entry point, or files to inspect, leaving the ability to determine what to test and where to investigate insufficiently evaluated. Relying on such clues bypasses the challenge of mapping requirements to implementations and proactively uncovering unknown defects. We propose SpecEngineeringBench, comprising 100 instances from 78 open-source Python repositories across 16 application categories, with multiple recorded defects in each instance. Two independent tasks evaluate complementary capabilities: generating test suites and discovering defects through execution analysis. The evaluation combines test validity, def
    
[^3]: ShellBench：对于AI智能体的日常任务而言，模型和智能体框架哪个更重要？

    ShellBench: What Matters More for AI Agents, Models, or Harness for Everyday Tasks

    [https://arxiv.org/abs/2611.32946](https://arxiv.org/abs/2611.32946)

    该论文推出ShellBench基准，通过对12个模型、5种智能体框架和3个推理水平共53,100次实验的评估发现，模型选择对日常任务性能的影响大于框架选择，但领先模型之间差距仅1.1个百分点，且高性能模型间的成本差异可达六倍之多。

    

    2026年全年，AI智能体在普通大众中的使用呈指数级增长，但用户往往难以确定哪种配置对其个人使用最为有效。为了帮助澄清这一问题，我们推出了ShellBench——基于超过1000万条用户请求构建的100项日常任务基准，每项任务都设置在包含杂乱文件和相互冲突数据源的真实工作空间中，并对正确性和安全性进行双重评分。我们评估了12个模型在5种智能体框架和3个推理水平下的表现，总计进行了53,100次实验，其中20%的任务被指定为公开发布。我们发现，模型选择对性能的影响大于框架选择，但三个领先模型之间的差距仅为1.1个百分点。GPT-6.1 Sol以约GPT-6 Astra每次尝试估算token成本的六分之一通过了61.4%的尝试，而GPT-6 Astra以62.5%的通过率位居首位。我们还发现，框架选择对性能几乎没有影响。

    arXiv:2611.32946v1 Announce Type: new  Abstract: The use of AI agents among the general public has been increasing exponentially throughout 2026, though it is often hard for users to determine which configuration is most effective for their personal use. To help provide clarity here we introduce ShellBench, 100 day-to-day tasks based on more than 10 million user requests, each set in a realistic workspace with messy files and competing data sources, and graded for both correctness and safety. We evaluate 12 models across 5 agent harnesses at 3 reasoning levels for a total of 53,100 rollouts, 20% of which are on tasks designated for public release. We find that model choice impacts performance more than harness choice, but the three leading models differ by only 1.1 percentage points. GPT-6.1 Sol passed 61.4% of attempts at about one-sixth of GPT-6 Astra's estimated token cost per attempt, while GPT-6 Astra led at 62.5%. We also found that harness choice had little effect on either perf
    
[^4]: 用例表如何支持解释需求的获取

    How Use Case Tables Enable the Elicitation of Explanation Needs

    [https://arxiv.org/abs/2611.32809](https://arxiv.org/abs/2611.32809)

    本研究探讨了用例表能否作为原型测试的替代方案，在软件开发早期获取用户的解释需求，并通过让用户阅读用例描述的实证研究加以验证。

    

    [背景与动机] 随着软件系统日益融入人们的日常生活——例如驾驶辅助系统或智能家居应用——它们也变得越来越复杂。在使用复杂软件时，可能会产生对解释的需求。目前，解释需求通常通过测试原型或现有软件来获取。然而，软件交互的详细文档（如用例表）可能是一种合适的替代方案。由于用例能够早期揭示系统内的关键交互，因此将其作为解释需求的来源颇具吸引力。[问题] 目前尚不清楚用例是否能够支持解释需求的获取，或者换个角度看，将其用于此目的是否可取。[主要思路/结果] 我们开展了一项研究，让用户仔细阅读现有用例的描述部分……

    arXiv:2611.32809v1 Announce Type: new  Abstract: [Context and motivation] While software systems get more integrated into people's daily lives - like driver-assistance systems or smart-home applications - they are also becoming increasingly complex. When complex software is used, needs for explanations may arise. At present, explanation needs are usually elicited through the testing of prototypes or existing software. However, detailed documentation of software interactions, such as use case tables, might be a suitable alternative. Since use cases provide early insights into the key interactions within the system, they are appealing to employ as resources for explanation needs. [Question/problem] As of now, it remains unclear whether use cases allow the elicitation of explanation needs. Or, from a different perspective, whether their utilization for this purpose is desirable. [Principal ideas/results] We conducted a study in which users perused the description sections of use cases ext
    
[^5]: 作为服务水平目标的能耗：事务级能耗归因

    Energy as a Service-Level Objective: Transaction-Level Energy Attribution

    [https://arxiv.org/abs/2611.32731](https://arxiv.org/abs/2611.32731)

    本论文借鉴SRE的服务水平目标方法论，提出事务级能耗归因方法，将每次成功事务的能耗作为服务水平目标，使能耗成为可操作的工程评估工具而非单纯的报告指标。

    

    服务工程师在评估每一项变更时，都会对照功能性需求以及可靠性、性能和可用性等质量属性，而能耗很少成为这一评估的一部分。当某项变更导致消耗上升时，团队无法判断这一增长是否值得调查、是否可以接受，或两者皆非。对追踪、重试或共享组件的修改，其影响可能远远超出被修改组件本身的能耗范围。本论文探讨如何使能耗目标成为可操作的工程工具，而非仅仅作为报告性指标。站点可靠性工程（SRE）通过服务水平指标、服务目标与响应策略，已经为可靠性回答了类似的问题。我们以SRE作为指导性研究愿景，研究这一类比在能耗领域何处成立、何处失效。例如，能耗目标可以在规定的工作负载条件下，对每次成功事务的能耗设定上限。本论文将三条研究线索连接成一个整体……

    arXiv:2611.32731v1 Announce Type: new  Abstract: Service engineers assess every change against functional requirements and qualities such as reliability, performance and availability. Energy is rarely part of this assessment. When consumption rises after a change, the team cannot tell whether the increase is worth investigating, worth accepting, or neither. A change to tracing, retries or a shared component can affect consumption far beyond the modified component. This thesis asks how energy objectives can become actionable engineering instruments rather than reporting metrics. Site Reliability Engineering (SRE) answered a similar question for reliability with service-level indicators, objectives and response policies. We take SRE as the guiding research vision and study where the analogy holds for energy and where it breaks. An energy objective could, for example, bound the energy per successful transaction under stated workload conditions. The thesis connects three threads into one e
    
[^6]: AI生成代码的问题：基于代码仓库历史、AI对话与开发者访谈的分类体系

    AI-Generated Code Issues: A Taxonomy from Repository Histories, AI Conversations, and Developer Interviews

    [https://arxiv.org/abs/2611.32676](https://arxiv.org/abs/2611.32676)

    该研究通过挖掘GitHub提交历史、分析开发者与AI工具的对话并访谈开发者三个互补数据源，首次系统构建了AI生成代码质量问题在真实软件开发中的分类体系，以帮助开发者改进代码审查与测试等质量保障实践。

    

    大型语言模型（LLM）被越来越多地用于生成源代码，但此类代码的质量问题至今仍未被充分理解。理解这些问题有助于开发者调整质量保障实践（例如代码审查和测试），以更好地检测那些在AI生成代码中可能比人类编写代码中更常见的问题。我们通过一项混合方法的实证研究，考察AI生成代码中的质量问题如何在真实软件开发活动中体现，该研究借鉴了三个互补来源的证据。首先，我们挖掘提交信息中明确将修复归因于AI生成代码问题的GitHub提交，人工检查了914条匹配的提交以去除误报，并对所修正的质量问题进行分类。其次，我们收集开发者与AI工具的共享对话，识别其中包含源代码及模型错误文本信号的对话，并人工分析了2……（原文摘要在此处不完整）

    arXiv:2611.32676v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to generate source code, yet the quality issues in such code remain only partially understood. Understanding these issues can help developers adapt quality assurance practices, such as code review and testing, to better detect problems that may be more common in AI-generated than in human-written code. We study how quality issues in AI-generated code manifest in real software development activities through a mixed-method empirical study that draws on evidence from three complementary sources. First, we mine GitHub commits whose messages explicitly attribute a fix to a problem in AI-generated code, and we manually inspect the 914 matched commits to remove false positives and classify the quality issue being corrected. Second, we collect developers' shared conversations with AI tools, identify conversations including source code and textual signals of model mistakes, and manually analyze 2
    
[^7]: 软件设计评审的“大象-金鱼”模型：全新评审者能看到什么

    The Elephant-Goldfish Model for software design review: what a fresh reviewer can see

    [https://arxiv.org/abs/2611.32668](https://arxiv.org/abs/2611.32668)

    该研究通过对47份植入缺陷的设计文档进行实验证明，让全新AI评审者接触设计起草过程的对话，可将遗漏决策的检出率从62.4%提升至94.8%，同时不影响文档可见缺陷的检测效果。

    

    arXiv:2611.32668v1 公告类型：新 摘要：一个设计可能很容易理解，却仍然规定了错误的系统。Google的“大象-金鱼”模型要求全新的智能体在评审设计时，不接触产生该设计的对话过程。我们研究了这一选择对遗漏项、协议违反和内部缺陷检测的影响。我们在47份公开软件设计文档中每份植入9个缺陷，并为每份文档配有一段脚本化的起草对话。当把该对话作为评审者自身此前的交流提供给全新评审者时，它检测出94.8%的遗漏决策实例，而相同模型仅拿到文档时的检出率为62.4%。它还检测出98.3%的仅存在于对话中的要求违反。对于文档可见的缺陷，有对话时的召回率为87.5%，无对话时为88.4%，在预设的15个百分点等效边际内二者相当。三个模型评审员对检测结果进行评分，并在构建的对照样本上进行了校准。在仅文档运行中，插件的五阶段评审耗时……

    arXiv:2611.32668v1 Announce Type: new  Abstract: A design can be easy to understand and still specify the wrong system. Google's Elephant-Goldfish Model asks fresh agents to review a design without the conversation that produced it. We examined how this choice affects detection of omissions, agreement violations and internal defects. We seeded nine defects in each of 47 public software design documents and supplied each with a scripted drafting conversation. A fresh reviewer given the conversation as its own earlier exchange detected 94.8% of omitted-decision instances, compared with 62.4% for the same model given the document alone. It detected 98.3% of violations of conversation-only requirements. On document-visible defects, recall was 87.5% with the conversation and 88.4% without it, equivalent within a prespecified 15-percentage-point margin. Three model judges scored detection and were calibrated on constructed controls. In document-only runs, the plugin's five-stage review took 
    
[^8]: 从源代码语义生成基于属性的回归测试

    Generating Property-Based Regression Tests from Source Code Semantics

    [https://arxiv.org/abs/2611.32636](https://arxiv.org/abs/2611.32636)

    本文提出通过多路径符号执行直接从源代码语义自动推导基于属性的回归测试，克服了传统单元测试可能漏检回归错误以及手动编写属性和输入生成器困难的问题。

    

    传统的单元测试（CUTs）在固定的具体输入下检查被测方法。因此，即使存在缺陷，如果测试输入仍然产生预期输出，测试就可能漏检回归错误。基于属性的测试（PBT）在大量随机生成的输入上检查全称量化的规范，但手动编写属性和输入生成器非常困难。现有的自动化PBT创建方法是通过泛化现有的或已生成的单元测试套件来实现的。它们要么在重复测试中对具体输入进行过度近似，要么沿具体测试运行应用单路径符号分析。本文提出了一种方法，通过多路径符号执行（SE）直接从源代码语义推导出基于属性的回归测试。对于目标方法中探索到的每条路径，该方法记录一条路径语义，包括路径条件、符号返回值和状态变化映射。每条路径语义都会导出一个Hoare……（摘要在此处被截断）

    arXiv:2611.32636v1 Announce Type: new  Abstract: Conventional unit tests (CUTs) check a method under test at fixed concrete inputs. A test can therefore miss a regression if its input produces the expected output despite a fault. Property-based testing (PBT) checks universally quantified specifications across many randomly generated inputs, but writing properties and input generators by hand is difficult. Existing approaches automate PBT creation by generalizing an existing or generated unit test suite. They either over-approximate concrete inputs across repetitive tests or apply single-path symbolic analysis along concrete test runs. This paper presents an approach that derives property-based regression tests directly from source code semantics via multipath symbolic execution (SE). For each explored path through a target method, the approach records a path semantics comprising the path condition, the symbolic return value, and the state change map. Each path semantics induces a Hoare
    
[^9]: VISE：基于工具实现反馈的、无需验证器的技能自进化方法

    VISE: Verifier-Independent Skill Self-Evolution with Feedback from Tool Implementations

    [https://arxiv.org/abs/2611.32480](https://arxiv.org/abs/2611.32480)

    VISE提出了一种无需任务特定验证器的技能自进化方法，通过从工具源代码中提取能力、输入要求、使用条件和输出格式等信息来指导技能工作流的修订，在两个基准上将平均任务成功率提升了9.14-13.35个百分点。

    

    智能体技能为LLM智能体使用工具提供了可复用的工作流程。改进这些工作流程需要关于缺失或错误指令的反馈，但针对特定任务的验证器并不总是可用。我们提出了VISE（无验证器技能进化），这是一种“脚本优先”的方法，利用工具实现来指导技能修订。它从源代码中提取工具的能力、输入要求、使用条件和输出格式。利用任务请求和执行轨迹，它将这些信息与当前工作流程指令进行比较，以识别缺失或冲突的步骤。然后，它为SKILL.md推荐具体的编辑，并将每项更改与相关的实现和使用条件关联起来。我们在SkillsBench和SkillLearnBench上评估了VISE作为四种进化方法的附加修订阶段的效果。与相应的无验证器条件相比，它将整体平均任务成功率提高了9.14-13.35个百分点。

    arXiv:2611.32480v1 Announce Type: new  Abstract: Agent skills provide reusable workflows that guide LLM agents in using tools. Refining these workflows requires feedback on missing or incorrect instructions, but task-specific verifiers are not always available. We propose VISE (Verifier-Independent Skill Evolution), a scripts-first approach that uses tool implementations to guide skill revision. It extracts tool capabilities, input requirements, usage conditions, and output formats from source code. Using the task request and execution trace, it compares this information with current workflow instructions to identify missing or conflicting steps. It then recommends specific edits to SKILL.md, linking each change to the relevant implementation and usage conditions. We evaluate VISE as an additional revision stage for four evolution methods on SkillsBench and SkillLearnBench. Compared with the corresponding No-verifier conditions, it improves overall average task success by 9.14-13.35 pe
    
[^10]: 提供研究论文上下文能否改进大语言模型生成的四足机器人导航？一项探索性仿真研究

    Does Research-Paper Context Improve LLM-Generated Quadruped Navigation? An Exploratory Simulation Study

    [https://arxiv.org/abs/2611.32381](https://arxiv.org/abs/2611.32381)

    这项探索性仿真研究发现，为 LLM 提供特定研究论文作为上下文并未改进四足机器人的导航代码，反而在迷宫最短路径中途被阻断的场景下显著降低了生成控制器的鲁棒性。

    

    大语言模型能够编写出合格的机器人导航代码，但由于该领域研究持续进展且成果数量庞大，模型可能无法召回相关研究发现并将其应用于当前任务。在这项工作中，我们测试了提供特定研究论文作为上下文是否能改进大语言模型为 Go2 四足机器人编写的导航代码，其中大语言模型（Claude Opus 5）和运动控制策略均保持冻结。具体而言，我们在三种设置下比较了大语言模型在有和没有特定论文作为上下文时编写的导航控制器：行为树、从共享基础版本编辑的导航器，以及从零开始编写的控制器。当迷宫的最短路径在回合中途被阻断时，以增量重规划论文作为上下文编写的 8 个控制器中仅有 2 个在 Isaac Sim 物理仿真下完成了全部四次行程，而没有该上下文的 8 个控制器中有 7 个完成了任务（双侧 Fisher 精确检验，p = 0.041）。这一差异仅在……（原文摘要在此处被截断）

    arXiv:2611.32381v1 Announce Type: cross  Abstract: Large language models can write competent robot navigation code, but not only does the research on the topic advance, a model may also, due to the volume of findings, fail to recall relevant findings and apply them to the task at hand. In this work, we test whether providing particular research papers as context improves the LLM-written navigation code for the Go2 quadruped, with the LLM (Claude Opus 5) and locomotion policy frozen. Specifically, we compare the navigation controllers the LLM writes when provided with and without particular papers across three settings: behavior trees, navigators edited from a shared base, and controllers written from scratch. When a maze's shortest route is blocked mid-episode, 2 of 8 controllers written with the context of an incremental-replanning paper complete all four trips under Isaac Sim physics, against 7 of 8 without it (two-sided Fisher's exact test, p = 0.041). This difference appears only u
    
[^11]: 验证需求工程中多模型AI生成的验收标准

    Validating Multi-Model AI-Generated Acceptance Criteria in Requirements Engineering

    [https://arxiv.org/abs/2611.32374](https://arxiv.org/abs/2611.32374)

    本文提出一种结合检索增强生成、多模型生成与人机协同工作流的面向验证方法，用于解决需求工程中AI生成的验收标准缺乏有效性验证的问题。

    

    大语言模型（LLMs）在软件工程中被越来越多地用于生成需求、用户故事、验收标准、文档和代码等制品。它们在需求工程中的应用引发了对正确性、可追溯性和幻觉问题的重要关注，尤其是因为有缺陷的规格说明会影响后续的设计、实现和验证活动。尽管近年来关于AI辅助软件工程的研究大多集中在代码生成和代码级验证上，但对AI生成的规格说明的验证却较少受到关注。本文通过提出一种面向验证的AI生成验收标准方法来填补这一空白。该方法结合了检索增强生成（RAG）、多模型生成以及明确结构化的人机协同工作流程。我们描述了一个可工作的原型系统，允许用户准备或输入用户故事并生成验收标准。

    arXiv:2611.32374v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in software engineering to generate artifacts such as requirements, user stories, acceptance criteria, documentation, and code. Their use in requirements engineering raises important questions about correctness, traceability, and hallucination, especially because flawed specifications can affect later design, implementation, and verification activities. While much recent work on AI-assisted software engineering has focused on code generation and code-level validation, the validation of AI-generated specifications has received less attention. This paper addresses this gap by presenting a validation-oriented approach for AI-generated acceptance criteria. The approach combines retrieval-augmented generation (RAG), multi-model generation, and an explicitly structured human-in-the-loop workflow. We describe a working prototype that allows users to prepare or enter user stories, generate acc
    
[^12]: SCOPE：评估与增强移动应用中权限说明的透明度

    SCOPE: Evaluating and Enhancing Permission Explanation Transparency in Mobile Apps

    [https://arxiv.org/abs/2611.32122](https://arxiv.org/abs/2611.32122)

    该论文通过对600款主流移动应用的实证研究揭示了权限说明存在的普遍缺陷（如缺少解释、表述模糊、未披露第三方SDK数据访问），并提出了名为SCOPE的自动化多智能体框架，用于系统性评估权限说明合规性并生成优化建议。

    

    权限说明是伴随移动应用权限请求而出现的解释性文本，对于用户隐私透明度和知情同意至关重要。尽管意义重大，但由于缺乏系统性的评估机制，当前的实践往往无法达到监管预期。通过对600款主流移动应用的实证研究，我们揭示了普遍存在的缺陷：15%的权限请求未提供任何解释，另一些则使用模糊语言或技术术语，更为关键的是，许多应用未能披露第三方SDK的数据访问情况，尽管这些组件正在积极使用已授予的权限。为了解决这些透明度缺口，我们提出了SCOPE——一个自动化多智能体框架，能够系统性评估权限说明的合规性并生成有针对性的优化建议。SCOPE采用四个专业化智能体协同工作：基于多模态大语言模型的说明提取、基于少样本学习的……（原文在此处截断）

    arXiv:2611.32122v1 Announce Type: new  Abstract: Permission explanations, explanatory text accompanying mobile app permission requests, are crucial for user privacy transparency and informed consent. Despite their importance, current practices often fall short of regulatory expectations due to the lack of systematic evaluation mechanisms. Through an empirical study of 600 mainstream mobile apps, we reveal widespread deficiencies: 15\% of permission requests provide no explanation, others use vague language or technical jargon, and critically, many fail to disclose third-party SDK data access despite these components actively using granted permissions. To address these transparency gaps, we present SCOPE, an automated multi-agent framework that systematically evaluates permission explanation compliance and generates targeted optimization recommendations. SCOPE employs four specialized agents working collaboratively: multimodal LLM-based explanation extraction, few-shot learning-based li
    
[^13]: 基于大语言模型的多结局游戏任务自动化冒烟测试

    Automated Smoke Testing for Multi-ending Game Quests via Large Language Models

    [https://arxiv.org/abs/2611.32089](https://arxiv.org/abs/2611.32089)

    提出了EndWeaver，一个专门面向多结局游戏任务的自动化冒烟测试框架，利用大语言模型进行多层次规划并结合状态转换图的动态构建，实现对分支叙事的高效探索和结局的快速验证。

    

    现代游戏任务日益趋向高自由度和多结局，使其设计和测试变得更加复杂。游戏中的冒烟测试旨在快速验证任务功能，但现有的自动化测试方法在处理具有分支叙事和多种可能结局的任务时面临关键局限。这些挑战导致对结局的探索不足、冗余的动作序列，以及过高的交互成本。为了解决这些挑战，我们提出了EndWeaver，一个专门为多结局游戏任务设计的自动化冒烟测试框架。EndWeaver利用大语言模型来解读任务描述和游戏环境，生成高层方法以指导全局探索，并通过生成的原语任务序列进行细粒度规划。通过将这种多层次规划结构与状态转换图的动态构建相结合，EndWeaver能够……

    arXiv:2611.32089v1 Announce Type: new  Abstract: Modern game quests are increasingly trending toward high freedom and multiple endings, making their design and testing more complex. Smoke testing in games aims to rapidly validate quest functionality, but existing automated testing approaches face critical limitations when handling quests with branching narratives, and multiple possible endings. These challenges result in insufficient exploration of endings, redundant action sequences, and prohibitively high interaction costs. To address these challenges, we propose EndWeaver, an automated smoke testing framework specifically designed for multi-ending game tasks. EndWeaver leverages LLMs to interpret quest descriptions and game environments, generating high-level methods to guide global exploration, complemented by the generated primitive task sequences for fine-grained planning. By combining this multi-level planning structure with the dynamic construction of state-transition graphs, E
    
[^14]: 面向编程智能体的在线环境修复

    Online Environment Repair for Coding Agents

    [https://arxiv.org/abs/2611.32037](https://arxiv.org/abs/2611.32037)

    该论文提出“在线环境修复”方法，在强化学习训练过程中持续检测并修复编程智能体环境中可被利用的验证器缺陷（如不完整的单元测试），在不惩罚有效解法、不丢弃高难度任务的前提下，从源头抑制奖励作弊行为。

    

    程序化验证器为通过强化学习训练编程智能体提供了可扩展的奖励信号，然而验证框架中细微的缺陷和不完整的单元测试，使策略能够在并未真正解决目标任务的情况下获得满分奖励。现有的训练时防御手段要么通过惩罚可疑输出来干预，要么通过过滤被标记的任务和轨迹来干预。然而，输出惩罚并未消除底层验证器可被利用的漏洞，反而促使策略将“作弊捷径”伪装得能够骗过推理监视器，或因检测器误报而将有效的解法清零；而任务与轨迹过滤则会丢弃或压制那些作弊捷径最早显现的高难度问题。预训练阶段的验证器加固虽然能够直接修复环境，但会遗漏那些只有在强化学习过程中随策略能力提升才暴露出来的漏洞类别。在本文中，我们提出在线环境修复方法，该方法在保持触发轨迹的奖励不变的同时，移除隐藏的……（原文摘要在此处截断）

    arXiv:2611.32037v1 Announce Type: new  Abstract: Programmatic verifiers provide scalable reward signals for training coding agents with reinforcement learning, yet subtle harness defects and incomplete unit tests allow policies to earn full credit without solving the intended task. Existing training-time defenses intervene either by penalizing suspicious outputs or by filtering flagged tasks and rollouts. However, output penalties leave the underlying verifier exploitable, driving policies to disguise shortcuts under reasoning monitors or zeroing out valid solutions under detector false positives, whereas task and rollout filtering discards or mutes the hardest problems where shortcuts emerge earliest. Pre-training verifier hardening repairs the environment directly, but misses vulnerability classes that surface only as the policy improves during reinforcement learning. In this paper, we propose online environment repair, which leaves triggering rollout rewards unchanged to remove hidi
    
[^15]: DesignEvo：跨提交恢复与追踪仓库级已实现的设计系统

    DesignEvo: Recovering and Tracking Repository-Level Implemented Design Systems Across Commits

    [https://arxiv.org/abs/2611.32020](https://arxiv.org/abs/2611.32020)

    DesignEvo通过融合截图、DOM结构、元素几何与样式信息，并将视觉-语言模型判断约束在程序生成的候选区域内，实现了跨提交在仓库级别自动恢复与追踪已实现的设计系统及其生命周期状态。

    

    现代前端应用依赖设计系统来保持页面和发布版本之间的一致性。然而，文档和组件库并不能完整捕捉界面模式在实践中是如何被实现、复用和调整的。现有的UI分析技术主要关注视觉缺陷、界面结构或元素对应关系，对在仓库级别自动恢复和追踪已实现的设计系统提供的支持有限。为弥合这一差距，我们提出了DesignEvo，它结合截图、DOM结构、元素几何信息和样式值来恢复功能性UI组。DesignEvo将视觉-语言模型的判断约束在程序生成的候选区域内，将恢复的UI组和样式聚合为仓库级记录，并在相邻提交之间推断其生命周期状态。我们在包含317个页面-版本实例和3,478个标注功能的10个开源仓库上对DesignEvo进行了评估。

    arXiv:2611.32020v1 Announce Type: new  Abstract: Modern frontend applications rely on design systems to maintain consistency across pages and releases. However, documentation and component libraries do not fully capture how interface patterns are implemented, reused, and adapted in practice. Existing UI analysis techniques focus on visual defects, interface structure, or element correspondences, providing limited support for automatically recovering and tracking implemented design systems at the repository level. To bridge this gap, we present DesignEvo, which combines screenshots, DOM structure, element geometry, and style values to recover functional UI groups. DesignEvo constrains vision-language model judgments to program-generated candidate regions, aggregates recovered groups and styles into repository-level records, and infers lifecycle states across adjacent commits. We evaluate DesignEvo on 10 open-source repositories comprising 317 page-version instances, 3,478 annotated func
    
[^16]: 当弹性策略的在线自适应适得其反时：Resilience4j中的安全探索与测量陷阱

    When Online Adaptation of Resilience Policies Hurts: Safe Exploration and Measurement Pitfalls in Resilience4j

    [https://arxiv.org/abs/2611.32015](https://arxiv.org/abs/2611.32015)

    该论文提出AdaptR4j系统，利用上下文赌博机在运行时动态选择Resilience4j的熔断器/重试/舱壁策略，并通过安全策略菜单和有界奖励的安全探索两项保障机制，克服了朴素在线学习在探索阶段拒绝流量的陷阱，在所有可赢得的故障场景中超越生产默认配置。

    

    Java微服务中的熔断器、重试和舱壁设置通常只调优一次便被冻结，然而最佳设置实际上取决于依赖服务当前表现出的故障类型。我们研究了在线学习器能否在运行时选择这些设置。AdaptR4j在Spring Boot 3中原子性地切换Resilience4j策略，并让上下文赌博机根据Prometheus信号在完整的熔断器/重试/舱壁配置菜单中进行选择。我们的第一个朴素设计在其所优化的奖励指标下，在四个可赢得的故障场景中有三个的表现反而不如保持默认策略：在学习器仍处于学习阶段时，探索包含激进熔断器的配置菜单导致大部分流量被拒绝。两个安全保障措施解决了这一问题：一是安全策略菜单（不包含比延迟SLO更严格的熔断器配置），二是带有有界奖励的安全探索。有了这些措施，AdaptR4j在每一个可赢得的场景中都胜过了生产默认配置（p < 0.001，每个场景三次运行，运行间标准差 <=

    arXiv:2611.32015v1 Announce Type: cross  Abstract: Circuit breaker, retry, and bulkhead settings in Java microservices are tuned once and frozen, although the best setting depends on the fault a dependency is currently showing. We study whether an online learner can pick these settings at runtime. AdaptR4j swaps Resilience4j policies atomically in Spring Boot 3 and lets a contextual bandit choose among complete CircuitBreaker/Retry/Bulkhead configurations from Prometheus signals. Our first, naive design scored worse than keeping the default policy in three of four winnable fault scenarios under the reward it optimised: exploring a menu that contained aggressive circuit breakers rejected most traffic while the learner was still learning. Two safeguards fix this: a safe policy menu (no breaker stricter than the latency SLO) and safe exploration with a bounded reward. With them, AdaptR4j beats the production default in every winnable scenario (p < 0.001, three runs each, run-to-run SD <= 
    
[^17]: 并非任意示例：用于指导大语言模型漏洞修复的安全需求与修复模式

    Not Just Any Example: Security Requirements and Fix Patterns for Guiding LLM Vulnerability Repair

    [https://arxiv.org/abs/2611.31993](https://arxiv.org/abs/2611.31993)

    该论文通过对348个开源仓库中3,816个安全补丁的大规模实证研究，提炼出与漏洞安全影响相关联的具体修复模式，并展示了如何利用这些模式引导大语言模型生成真正满足安全需求的漏洞修复补丁。

    

    软件漏洞使系统面临攻击风险，而漏洞的修复往往被延误。大语言模型（LLM）在自动化补丁生成方面取得了进展，然而通过功能性测试并不意味着补丁能够满足漏洞所违反的安全需求。现有的漏洞修复技术依赖于人工整理的、覆盖范围有限的模式或模板，因此尚不清楚哪些修复策略可以跨漏洞通用，以及如何利用这些策略来引导大语言模型。我们提出了一项大规模实证研究，将重复出现的修复模式与漏洞的安全影响相关联，并考察这些知识如何指导基于大语言模型的修复。在对安全补丁进行人工研究的基础上，我们对来自348个开源代码仓库的3,816个安全补丁进行了聚类，并分析代码修改及其上下文，以推导出具体的修复模式，涵盖从局部防护措施到跨多个位置的协同修改等多种类型。

    arXiv:2611.31993v1 Announce Type: cross  Abstract: Software vulnerabilities leave systems open to attack, and their remediation is often delayed. Large language models (LLMs) have advanced automated patch generation, yet passing functionality tests does not ensure that a patch enforces the security requirement the vulnerability violated. Existing vulnerability repair techniques rely on curated patterns or templates with limited coverage, leaving unclear which repair strategies carry over across vulnerabilities and how they can steer LLMs. We present a large-scale empirical study that relates recurring fix patterns to the security impacts of vulnerabilities and examines how this knowledge can guide LLM-based repair. Building on a manual study of security patches, we cluster 3,816 security patches from 348 open-source repositories and analyze the edits and their context to derive concrete fix patterns, ranging from local safeguards to coordinated changes across multiple locations. We def
    
[^18]: CAPE：协同校准与搜索的工具使用策略编译

    CAPE: Coordinating Calibration and Search for Tool-Use Policies

    [https://arxiv.org/abs/2611.31938](https://arxiv.org/abs/2611.31938)

    CAPE是一个无需训练的工具使用策略编译器，它通过协调校准与搜索，始终保持可执行策略，并将校准资源集中于能改进该策略的调用上，从而高效应对配置不可见工具导致的调用失败问题。

    

    即使工具调用背后的操作是正确的，该调用也可能失败，因为接收方工具会在代理无法查看的配置下解析分隔符、日期或记录布局。为此类工具编译工具使用策略既需要校准，即在候选配置上评估候选调用，也需要在已评估的调用上进行搜索。这两者共享同一个编译预算：如果先把所有内容都校准好，任何改进后的策略都要推迟到最后才能产生；而如果对每个部分已知的模型都重新求解，则会延误下一个有用的查询。一个调用只要在能够到达它的配置上的效果是已知的，它就可以被使用。基于这一观察，我们提出了CAPE，一个无需训练的编译器。它在任何时刻都保持一个可执行的策略，将校准引导至能够改进该策略的调用上，并且只在能够提前结束校准的情况下收紧边界。返回的策略保留了初始策略所完成的每一个配置。

    arXiv:2611.31938v1 Announce Type: new  Abstract: A tool call can fail even when the operation behind it is correct, because the receiving tool parses delimiters, dates or record layouts under a configuration the agent cannot inspect. Compiling a tool-use policy for such a tool requires both calibration, which evaluates candidate calls on candidate configurations, and search over the evaluated calls. The two share one compilation budget: calibrating everything first postpones any improved policy until the end, while re-solving every partially known model delays the next useful query. A call becomes usable as soon as its effects are known on the configurations that reach it. We present CAPE, a training-free compiler built on this observation. It maintains an executable policy at all times, directs calibration to the calls that can improve this policy, and tightens bounds only where doing so ends calibration early. Returned policies preserve every configuration the initial policy complete
    
[^19]: Shuttle：面向智能合约审计的依赖归因工作流演化

    Shuttle: Dependency-Attributed Workflow Evolution for Smart Contract Auditing

    [https://arxiv.org/abs/2611.31929](https://arxiv.org/abs/2611.31929)

    Shuttle通过溯源图归因定位工作流中线索丢失的环节并保护已有检测成果，实现受约束的智能合约审计工作流演化，避免了无约束改写导致的性能退化和泛化能力下降。

    

    在基于大语言模型（LLM）的智能合约审计中，检测质量既取决于模型能力，也取决于工作流设计。虽然编程智能体可以根据执行反馈修改工作流，但不受约束的改写很少带来稳定收益。跨函数漏洞需要多步状态跟踪和边界推理，而粗糙的流水线往往难以捕捉。端到端的结果掩盖了哪个步骤丢失了线索，因此在粗糙的聚合评分下，直接的改写会导致已有检测成果的退化，或丢弃有效的编辑。此外，在小规模项目集上反复迭代还会引入针对特定案例的启发式规则，使提示词膨胀并损害泛化能力。我们提出了Shuttle——一个受约束的工作流演化系统，以及由它演化的审计框架LucidScan。Shuttle的核心是溯源图归因：它将每次运行展开为实例级依赖图，通过前向搜索定位线索丢失的节点，并将已有的检测成果向后追溯为受保护集合。

    arXiv:2611.31929v1 Announce Type: new  Abstract: In LLM-based smart contract auditing, detection quality depends on both model capability and workflow design. While coding agents can revise workflows from execution feedback, unconstrained rewrites rarely yield stable gains. Cross-function vulnerabilities require multi-step state tracking and boundary reasoning that coarse pipelines miss. End-to-end outcomes hide which step lost a clue, so direct rewrites regress existing hits or discard valid edits under coarse aggregate scoring. Repeatedly iterating on a small project set also adds case-specific heuristics that bloat prompts and hurt generalization. We present Shuttle, a constrained workflow-evolution system, and LucidScan, the auditing framework it evolves. Shuttle centers on provenance-graph attribution: it expands each run into an instance-level dependency graph, locates the node where a clue was lost via forward search, and traces existing hits backward into a protected set. After
    
[^20]: （无）关注点分离：关于智能体伦理的哲学关切及其对软件工程从业者的启示

    (No) Separation of Concerns: Philosophical Concerns about Agent Ethics and Implications for SE Practitioners

    [https://arxiv.org/abs/2611.31927](https://arxiv.org/abs/2611.31927)

    该研究通过系统性梳理PhilPapers中哲学家关于AI智能体伦理的文献，提炼出八个主题和85个编码，将其转化为软件工程研究议程以及使用编程智能体团队的实践启示。

    

    人工智能智能体在软件工程（SE）中的应用已十分普遍，它们具有不同程度的自主性、丰富多样的能力，并能执行多种多样的任务。哲学家们已开始审视这类智能体的伦理问题，软件工程的研究者或从业者可以从中受益；另一方面，软件工程从业者目前主要是通过原则主义的视角来探讨这些伦理关切。我们通过综述PhilPapers数据库中收录的关于AI智能体的哲学文献来弥合这一差距，并将其中的伦理关切转化为面向软件工程研究者的研究议程，以及面向使用编程智能体的团队的实践启示。我们对PhilPapers——一个主要的哲学索引和开放获取档案库——开展了系统性映射研究，共获得2023年以来由专业哲学家发表的33篇同行评审论文。我们对这些论文进行了主题编码，并对涌现的主题进行了跨学科评审，最终归纳出八个主题和85个编码。

    arXiv:2611.31927v1 Announce Type: new  Abstract: The use of AI agents in software engineering (SE) are commonplace, with differing levels of autonomy, a myriad of capabilities, and ability to perform a variety of tasks. Philosophers have begun to examine the ethics of such agents, and SE researchers or practitioners could benefit from them; on the other hand, SE practitioners currently explore most of these concerns through the lens of principlism. We bridge this gap by reviewing the philosophical literature on AI agents indexed in PhilPapers, and translating its ethical concerns into a research agenda for SE researchers and into implications for teams that use coding agents. A systematic mapping study of PhilPapers -- a major philosophy index and open access archive -- yielded 33 peer-reviewed papers by professional philosophers published since 2023. We coded these papers thematically, with cross-disciplinary review of the emerging themes, resulting in eight themes and 85 codes. From 
    
[^21]: 规格-测试覆盖差距：检测SWE-Bench风格基准中的过度指定测试

    Spec-Test Coverage Gap: Detecting Overspecified Tests in SWE-Bench-Style Benchmarks

    [https://arxiv.org/abs/2611.31839](https://arxiv.org/abs/2611.31839)

    本文提出规格-测试覆盖差距（STCG）这一1-4分制新指标，利用两个LLM评判者自动检测SWE-bench风格基准中的过度指定测试，从而取代昂贵的专家审查，识别会导致正确的替代修复方案被误判失败的测试用例。

    

    SWE-bench风格的基准测试，例如SWE-bench Verified（Verified）和SWE-bench Pro（Pro），使用源自最初修复各问题的变更中的测试来评估编码智能体。然而，这些测试可能会强制执行任务描述中从未说明的细节，导致正确的替代修复方案失败。迄今为止，这类过度指定的测试主要通过昂贵的专家审查来发现。为了解决这一问题，我们定义了规格-测试覆盖差距（STCG），这是一个用于衡量过度指定程度的1-4分制量表，并由两个LLM评判者——Claude Opus 5.5和GPT-6.1 Sol——根据规格说明和测试补丁进行评分。当两个评判者都给出3分或更高分数时，该任务被标记为过度指定。我们的研究包括三个部分。首先，我们在206个任务上验证了评判者，这些任务由早期的人工审计和智能体结果标记为过度指定或正常。其次，我们将STCG应用于全部500个Verified任务和731个公开的Pro任务，并将标记结果与排名前30的Verified（原文在此处截断）

    arXiv:2611.31839v1 Announce Type: cross  Abstract: SWE-bench-style benchmarks, such as SWE-bench Verified (Verified) and SWE-bench Pro (Pro), evaluate coding agents with tests taken from the change that originally fixed each issue. However, these tests can enforce details the task never states, causing correct alternative fixes to fail. Such overspecified tests have so far been found mainly through expensive expert review. To address this, we define the Spec-Test Coverage Gap (STCG), a 1-4 scale measuring overspecification, and score it with two LLM judges, Claude Opus 5.5 and GPT-6.1 Sol, which are given the specification and the test patch. A task is flagged as overspecified when both judges score it 3 or higher. Our study has three parts. First, we validate the judges on 206 tasks that earlier human audits and agent results label as overspecified or fine. Second, we apply STCG to all 500 Verified and 731 public Pro tasks and compare the flags with the results of the top 30 Verified 
    
[^22]: 基于智能体的安全Rust绑定健全性漏洞发现

    Agentic Discovery of Soundness Bugs in Safe Rust Bindings

    [https://arxiv.org/abs/2611.31813](https://arxiv.org/abs/2611.31813)

    本文提出将LLM智能体作为Rust安全封装接口的审计员来发现健全性漏洞，定义了此类接口缺陷的分类体系，并通过真实crate的重复审计实验发现：智能体能自发找到未定义行为和与底层库的偏差，却难以发现不完整的不安全契约。

    

    Rust crate 将不安全代码（例如对 C 或 C++ 库的调用、系统调用）封装在必须维护 Rust 不变量的安全接口之后。如果接口对其底层代码的某条规则约束得过于宽松，用纯安全 Rust 编写的程序就可能引发未定义行为。静态分析器和模糊测试工具受限于固定的模式或其实际执行的路径。我们提出将 LLM 智能体用作这些接口的审计员，并研究其审计效果。我们首先为此类接口的缺陷定义了一个分类体系，包括未定义行为、与被封装库的偏差，以及不完整的不安全契约。随后，我们通过闭源模型 Claude Opus 5.5 和开源模型 DeepSeek V4.1 Flash 对一个真实世界 crate 进行重复审计，测量了智能体能发现哪些类别的缺陷、它们的缺陷分类能力如何，以及哪些因素能促使它们发现更多缺陷。结果表明，智能体在无提示的情况下就能发现前两类缺陷，但即使在明确要求下也很少发现不完整的不安全契约。（摘要原文在此处截断）

    arXiv:2611.31813v1 Announce Type: cross  Abstract: Rust crates confine unsafe code, such as calls into C or C++ libraries or system calls, behind safe interfaces that must uphold Rust's invariants. If an interface encodes one rule of the code beneath it too weakly, a program written in purely safe Rust can cause undefined behavior. Static analyzers and fuzzers are bound to fixed patterns or to the paths they execute. We propose LLM agents as auditors of these interfaces, and study how well they do it.   We first define a taxonomy of the defects of such interfaces, including undefined behavior, divergences from the wrapped library, and incomplete unsafe contracts. We then measure, in repeated audits of a real-world crate by a closed and an open model, Claude Opus 5.5 and DeepSeek V4.1 Flash, which defect classes agents find, how well they triage them, and what makes them find more. Agents find the first two classes unprompted, but find few incomplete contracts even when asked. Their fin
    
[^23]: 通过阅读生成的C++代码实现SystemVerilog到SystemC RTL转换的翻译验证

    Translation Validation of SystemVerilog-to-SystemC RTL Conversion by Reading the Generated C++

    [https://arxiv.org/abs/2611.31674](https://arxiv.org/abs/2611.31674)

    本文提出了一种SystemVerilog到SystemC RTL转换的翻译验证方法：按照C++17和SystemC的真实语义解读生成的C++代码、将未定义行为转化为证明义务，并通过显式化所有位宽变化来证明生成代码与源RTL网表等价。

    

    寄存器传输级（RTL）设计的SystemC模型被用于虚拟平台、固件启动调试以及无需商业许可证的仿真，如今越来越多的此类模型来自自动转换器和语言模型。错误的模型通常能够编译通过，并且在大多数情况下与RTL行为一致。SystemVerilog以8位宽度计算8位加法，而C++以64位宽度计算，因此忘记一次截断的转换会保留RTL所丢弃的进位，且只有接近数值范围顶部的输入才会暴露这一错误。作者构建了svcc——一个开源的SystemVerilog到SystemC转换器，其最初对输出正确性的检查方式是用SystemVerilog的规则来阅读生成的C++代码，但该检查却放过了两个错误的模型。取代它的检查方法赋予生成的头文件以C++17和SystemC所定义的真正含义，并将未定义行为转化为需要证明的义务。该检查将读取到的内容以SystemVerilog形式打印出来，并使所有位宽变化显式化，进而证明其与源设计的网表等价……

    arXiv:2611.31674v1 Announce Type: cross  Abstract: SystemC models of register-transfer level (RTL) designs are used for virtual platforms, firmware bring-up and simulation without a commercial license, and more of them now come from translators and language models. A wrong model usually compiles, and it usually agrees with the RTL. SystemVerilog evaluates an 8-bit sum in 8 bits while C++ evaluates it in 64, so a translation that forgets one truncation keeps a carry the RTL drops, and only inputs near the top of the range show it. I built svcc, an open-source SystemVerilog-to-SystemC converter, and my first check of its output read the generated C++ with SystemVerilog's rules. That check passed two wrong models. The check that replaced it gives the generated header the meaning C++17 and SystemC give it and turns undefined behavior into proof obligations. It prints what it read as SystemVerilog with every width change explicit, then proves that equivalent to a netlist of the source by in
    
[^24]: GoldiBench：工具选对了，用量却错了——面向专业编辑中效果判断能力的智能体基准测试

    GoldiBench: Right Tools, Wrong Amount -- Benchmarking Agents on Effect Judgement in Professional Editing

    [https://arxiv.org/abs/2611.31590](https://arxiv.org/abs/2611.31590)

    论文提出GoldiBench基准，通过500个专业编辑任务将“工具操作”与“效果判断（用量把握）”分离评估，发现智能体操作能力很强，但难以准确判断实现目标视觉效果所需的工具用量。

    

    真实的编辑需求描述的是一种视觉效果，而非达成该效果的具体步骤，因此服务于这类需求的智能体必须判断每种工具需要使用多少才能实现该效果。现有基准并未衡量这一点：它们的任务要么直接给出工具和参数值，要么对用量不加限定，并且依赖VLM裁判评分——而这些裁判会让超过一半的半强度和三倍强度结果通过。我们提出了GoldiBench，一个包含500个编辑任务的基准，涵盖五个专业应用（darktable、RawTherapee、GIMP、Inkscape、Blender），任务既可以通过编写程序解决，也可以通过操作GUI完成。每个任务将一张START图像与一个由隐藏程序渲染出的TARGET图像配对，因此用量由已知目标固定；智能体的编辑结果会被重新渲染并与TARGET进行比较，且同一任务以三种形式呈现：给出工具和参数值、仅给出工具、或仅给出目标，从而将操作能力与效果判断区分开来。在25个开源、GUI和闭源模型中，智能体的操作能力表现良好（4B以上模型使用程序可达99-100%……

    arXiv:2611.31590v1 Announce Type: cross  Abstract: Real editing requests describe a look, not the steps to reach it, so an agent that serves them must judge how much of each tool the look takes. Existing benchmarks do not measure this: their tasks either state the tools and values, or leave the amount open and are scored by VLM judges that pass more than half of half- and triple-strength results. We introduce GoldiBench, a benchmark of 500 editing tasks in five professional applications (darktable, RawTherapee, GIMP, Inkscape, Blender), solvable by writing a program or by operating the GUI. Each task pairs a START image with a TARGET rendered from a hidden program, so the amount is fixed by a known target; the agent's edit is re-rendered and compared with TARGET, and the same task is posed with the tools and values, the tools only, or the goal only, separating operation from effect judgement. Across 25 open, GUI and closed models, agents operate well (99-100% with programs from 4B up w
    
[^25]: 智能体恢复出错：测量重复调用之外的危害

    Agent Recovery Gone Wrong: Measuring Harm Beyond Duplicate Calls

    [https://arxiv.org/abs/2611.31554](https://arxiv.org/abs/2611.31554)

    该论文首次在自然发生的失败场景下量化了代码执行型LLM智能体恢复操作对外部状态的危害，发现约24.6%的结果可确定的部分执行会导致在运行结束时仍持续存在的有害恢复，其中仅三分之二表现为重复调用，其余则表现为删除用户数据或错误的首次写入，说明仅检测重复调用不足以捕捉全部危害。

    

    执行代码的大语言模型智能体会提交包含多个API写操作的代码块。如果某个代码块在其部分写操作已生效之后失败，智能体会重写该代码块并继续执行。当这种失败在智能体自己编写的代码中自然发生时，此类恢复在多大频率上会危害外部状态，目前尚不清楚。本文在AppWorld智能体的记录轨迹上对此进行了测量。在对十个模型的200个候选回合进行的分层随机抽样中，有145个被数据库状态确认为部分执行。在样本中，143个结果可确定的确认部分执行里有30个随后发生了有害恢复，且其违规状态在运行结束时仍然存在，由此得到分层加权估计值为24.6%。三分之二的有害恢复表现为重复了某个效果，其余的则删除用户数据或进行错误的首次写入，例如订购了错误的产品。搜索重复调用的规则发现……（原文在此处截断）

    arXiv:2611.31554v1 Announce Type: new  Abstract: Code-executing LLM agents submit blocks of code that issue several API writes. If a block fails after some of its writes have taken effect, the agent rewrites the block and continues. It is unknown how often such a recovery harms the external state when the failure occurs naturally in the code written by the agent. In this paper, we measure it on recorded trajectories of AppWorld agents. In a stratified random sample of 200 candidate episodes across ten models, 145 are partial executions confirmed by the database state. In the sample, 30 of 143 confirmed partial executions with a determinable outcome are followed by a harmful recovery, and its violation remains at the end of the run, which gives a stratification-weighted estimate of 24.6%. Two thirds of the harmful recoveries duplicate an effect, and the rest delete user data or make a wrong first-time write, e.g., an order of a wrong product. A rule that searches for repeated calls find
    
[^26]: 秒表之下：在内核层面检验编码代理的性能声明

    Below the Stopwatch: Checking the Performance Claims of Coding Agents in the Kernel

    [https://arxiv.org/abs/2611.31460](https://arxiv.org/abs/2611.31460)

    提出了Below流水线，通过提取编码代理性能PR所声称的优化机制并用硬件计数器与LTTng内核追踪验证其应有的内核级信号，发现大多数性能声明缺乏实证测量，且相当一部分PR并未带来真实的性能提升。

    

    编码代理提交了大量声称能让软件变快的拉取请求（PR）。这类声明大多还会给出理由，例如“缓存结果”或“只读取文件一次”。秒表可以显示代码变快了，却无法证明它是因为所声称的原因而变快。我们提出了Below——一个在内核层面检验这些理由的流水线。Below从AIDev数据集中的1,343个运行时性能PR中提取每个PR所声称的机制，并将其映射到如果该声明为真则必须发生变化的内核与硬件信号。一个从不接触这些声明的构建代理在每个PR应用前后重新构建代码。随后Below通过普通计时运行、硬件计数器和LTTng内核追踪来测量两个版本。结果显示大多数声明并未经过测量：57.3%的PR仅提供推理或没有任何证据，只有31.4%给出了效应大小。在82个被分析的PR中，32个运行更快，28个没有明显变化，22个运行更慢。所声称机制的预期信号……（摘要原文在此处截断）

    arXiv:2611.31460v1 Announce Type: cross  Abstract: Coding agents open many pull requests (PRs) that claim to make software faster. Most such claims also give a reason, such as "cache the result" or "read the file once". A stopwatch can show that code got faster, but not that it got faster for the stated reason. We present Below, a pipeline that checks these reasons at the kernel level. From 1,343 runtime-performance PRs in the AIDev dataset, Below extracts the mechanism each PR claims and maps it to the kernel and hardware signals that must change if the claim is true. A build agent that never sees the claim rebuilds the code before and after each PR. Below then measures both versions with plain timing runs, hardware counters and LTTng kernel traces. Most claims are not measured: 57.3% of the PRs offer only reasoning or no evidence, and only 31.4% state an effect size. Of 82 analyzed PRs, 32 ran faster, 28 showed no clear change and 22 ran slower. The expected signal of the claimed mec
    
[^27]: CrossCheck II：用于实时C与C++代码认证成本界限的编译器插件及其评估

    CrossCheck II: A Compiler Plugin for Certified Cost Bounds of Real-Time C and C++ Code, and its Evaluation

    [https://arxiv.org/abs/2611.31449](https://arxiv.org/abs/2611.31449)

    该论文提出了Clang编译器插件CrossCheck II，能够自动为实时C/C++代码计算成本界限并生成经Lean 4验证的成本证书，使界限的正确性不依赖于分析器和求解器，且在TACLeBench基准上无需注解即可界定220个循环中的142个。

    

    我们描述了一个Clang编译器插件，该插件为面向实时用途的C和C++函数界定成本界限，并在编译期拒绝那些进行内存分配、抛出异常、通过指针调用，或包含无法找到界限的循环或递归的函数。循环的界限通过利用Farkas引理由SMT求解器合成的仿射排名势函数来确定，或采用对数势函数；递归则通过深度和规模势函数来确定；被调用函数在其被使用的调用上下文中进行分析。对于线性片段中的函数，该插件会生成一份成本证书，由一个检查器（由Lean 4定义编译而成，其正确性已在配套论文中得到证明）仅使用整数算术进行验证；这样，界限的正确性既不依赖于分析器，也不依赖于求解器。在TACLeBench基准集合的29个内核上，在无需注解的情况下，该插件在程序的调用上下文中为220个循环中的142个确定了界限，其中131个与基准测试的（原文在此处截断）

    arXiv:2611.31449v1 Announce Type: cross  Abstract: We describe a plugin for the Clang compiler that bounds the cost of C and C++ functions intended for real-time use, and rejects at compile time the functions that allocate, throw, call through pointers, or contain a loop or a recursion for which no bound is found. Loops are bounded by affine ranking potentials synthesized with an SMT solver through Farkas' Lemma, or by logarithmic ones, recursion by depth and size potentials, and callees are analyzed in the calling context in which they are used. For functions in a linear fragment, the plugin emits a cost certificate that a checker, compiled from a Lean~4 definition proved sound in a companion paper, validates with integer arithmetic only; the bound then depends neither on the analyzer nor on the solver. On the 29 kernels of the TACLeBench collection, without annotations, the plugin bounds 142 of 220 loops in the calling contexts of the programs, 131 of them exactly as the benchmarks' 
    
[^28]: 面向定义驱动的 GitHub Actions 工作流坏味检测的决策模型

    Decision Models for Definition-Driven GitHub Actions Workflow Smell Detection

    [https://arxiv.org/abs/2611.31421](https://arxiv.org/abs/2611.31421)

    本文提出用面向决策的模型依据可复用的自然语言坏味定义来检测 GitHub Actions 工作流坏味，发现检测效果高度依赖决策策略，其中 JEV 1.13 达到 79.69% 的微平均 F1，并在评分可靠性、可重复性和鲁棒性上表现更优。

    

    GitHub Actions 工作流用于自动化持续集成与交付（CI/CD），而工作流坏味是指那些可能削弱安全性、阻碍维护或浪费资源的重复出现的配置模式。若要扩展静态分析器以检测新的坏味，则需要在检测规则中编码相关结构、适用条件和例外情况。我们研究了定义驱动的检测方法，即由模型依据可复用的自然语言坏味定义来评估工作流，重点关注面向决策的模型，这类模型针对有界问题返回评分。我们将其与生成式模型以及一个静态分析器进行比较，并考察其评分的可靠性、可重复性和鲁棒性。我们在 1,710 个经过人工验证的工作流坏味样本对上评估了三种面向决策的配置和两种生成式配置，这些样本覆盖 19 种坏味、903 个工作流和 559 个代码仓库。结果表明，检测有效性在很大程度上取决于决策策略：JEV 1.13 在相应阈值下达到 79.69% 的微平均 F1 值。

    arXiv:2611.31421v1 Announce Type: new  Abstract: GitHub Actions workflows automate continuous integration and delivery (CI/CD), and workflow smells are recurring configuration patterns that may weaken security, hinder maintenance, or waste resources. Extending static analyzers to new smells requires encoding constructs, applicability conditions, and exceptions in detection rules. We investigate definition-driven detection, in which models assess workflows against reusable natural-language smell definitions, focusing on decision-oriented models that return scores for bounded questions. We compare them with generative models and a static analyzer, and examine their score reliability, repeatability, and robustness. We evaluate three decision-oriented and two generative configurations on 1,710 manually validated workflow-smell pairs covering 19 smells, 903 workflows, and 559 repositories. Effectiveness depends strongly on the decision policy: JEV 1.13 achieves 79.69% micro-F1 at threshold 
    
[^29]: 差异是廉价的，给我看轨迹：基于轨迹信息的代码审查实证研究

    Diffs Are Cheap, Show Me the Trajectory: An Empirical Study of Trajectory-Informed Code Review

    [https://arxiv.org/abs/2611.31316](https://arxiv.org/abs/2611.31316)

    该论文通过实证研究系统探索了编码智能体的开发轨迹（包含中间决策、失败尝试和验证证据）能否帮助审查智能体超越仅检查最终代码差异来发现代码问题，并构建了包含339个实现、113个DeepSWE任务轨迹和882条参考审查意见的数据集进行评估。

    

    代码审查通常从最终的代码差异开始。然而，代码差异几乎无法揭示代码是如何被开发出来的。编码智能体轨迹保留了中间决策、失败的尝试和验证证据，这些信息可能有助于代码审查。我们系统地研究了这些轨迹能否以及如何帮助独立的审查智能体识别最终代码中的问题，而不仅仅是检查最终的代码差异和代码仓库。我们开发了一种通过比较拉取请求的初始版本和合并版本来生成审查意见的流程，并将其与保留的人类审查意见进行对比评估。通过使用留出的参考实现作为智能体生成代码的比较目标，我们构建了一个数据集，将113个DeepSWE任务的339个实现与其轨迹以及882条合成参考审查意见相关联。在三种编码智能体配置和两种审查框架下，我们比较了仅基于差异的审查与基于完整轨迹的审查的表现。

    arXiv:2611.31316v1 Announce Type: new  Abstract: Code review typically starts from the final diff. However, the diff reveals little about how the code was developed. Coding-agent trajectories preserve intermediate decisions, failed attempts, and validation evidence that may assist code review. We systematically investigate whether and how these trajectories can help independent review agents identify problems in the resulting code, beyond inspecting the final diff and repository. We develop a procedure that generates review comments by comparing the initial and merged versions of pull requests, and assess it against withheld human review comments. Using held-out reference implementations as comparison targets for agent-produced code, we construct a dataset linking 339 implementations of 113 DeepSWE tasks with their trajectories and 882 synthetic reference review comments. Across three coding-agent configurations and two review harnesses, we compare diff-only review, full-trajectory acc
    
[^30]: 缺陷在于文件之间的算术关系：支付流水线中“恰好一次”语义违反的静态检测

    The Defect Is in the Arithmetic Between the Files: Static Detection of Exactly-Once Violations in Payment Pipelines

    [https://arxiv.org/abs/2611.31298](https://arxiv.org/abs/2611.31298)

    该论文提出一种静态检测方法，通过收集分散在不同文件中的幂等键保留时长与记录最大过期时间等参数并分析其数值关系，检测支付流水线重启后可能违反“恰好一次”语义的隐患，并将515个发现中的335个转化为可运行的故障复现场景，其中245个场景被证实会导致客户被重复扣款。

    

    流式支付流水线在重启后是否会向客户重复扣款，取决于一个几乎无人进行的比较：提供方记住幂等键的时长，与流水线第二次处理某条记录时该记录可能过期的程度。第一个数字写在提供方的文档中；第二个数字则没有写在任何地方——它由分散在各处的信息拼凑而成：一个文件中的检查点间隔、一份运行手册中的恢复假设、第三个文件中的重启延迟，以及第四个文件中数据源的迟到上界。每个文件单独来看都无懈可击，缺陷恰恰在于它们之间的关系。本文提供了一个检查器来收集这些事实并陈述其间的关系，还提供了一个综合器，将515个发现中的335个转化为可运行的场景：给定这些记录、在此处终止作业、等待这么长时间后重启，即可观察到相应结果。其中245个场景会产生两次扣款。核心度量是证据之间的距离有多远。在630条流水线的语料库中，我们……（原文摘要在此处截断）

    arXiv:2611.31298v1 Announce Type: new  Abstract: Whether a streaming payment pipeline charges a customer twice after a restart turns on a comparison nobody performs: how long the provider remembers an idempotency key, against how stale a record can be when the pipeline processes it for the second time. The first number is in the provider's documentation. The second is not written anywhere. It is assembled from a checkpoint interval in one file, a recovery assumption in a runbook, a restart delay in a third and a source's lateness bound in a fourth. Each file is defensible alone; the fault is in the relation between them. We give a checker that gathers those facts and states the relations, and a synthesiser that turns 335 of the 515 findings into a runnable scenario: these records, kill the job here, bring it back after this long, and this is the observable. 245 of them produce two charges. The central measurement is how far apart the evidence lies. Across a corpus of 630 pipelines we g
    
[^31]: LITMUS——基于文献的用例映射与风险卡综合测试平台

    LITMUS - Literature Informed Testbed for Mapping Usecases and Synthesizing Risk-Cards

    [https://arxiv.org/abs/2611.29034](https://arxiv.org/abs/2611.29034)

    本文提出LITMUS，一个结构化流水线，能够根据给定用例自动生成定制化数据集与风险卡，并利用同行评审文献提供可追溯证据，将每个风险与具体用例紧密关联，弥补了现有方法忽视用例场景的不足。

    

    大语言模型（LLM）和人工智能（AI）在现实世界应用中的快速普及，加剧了对系统化、面向安全的部署框架的需求。现有方法通常优先围绕LLM和AI架构本身来构建数据集和风险卡，却很少考虑驱动模型选择的特定用例。这种疏漏是有问题的，因为任何实际的AI部署都始于详细的用例描述，而该描述正是评估上下文风险的基础。在本文中，我们提出了一个结构化流水线，给定一个用例，它能够自动生成定制的数据集以及配套的风险卡。该流程确保所识别的风险与用例紧密耦合，并提供来自同行评审文献的可追溯证据，以证实每一个风险与用例之间的关系。先前的声明验证方法（FActSco...

    arXiv:2611.29034v1 Announce Type: new  Abstract: The proliferation of large language models (LLMs) and Artificial Intelligence (AI) in real-world applications has heightened the need for systematic, safety-oriented deployment frameworks. Existing approaches typically prioritize the construction of datasets and risk cards around the LLM and AI architecture itself, yet they provide scant consideration for the specific use-case that drives model selection. This omission is problematic because any practical AI deployment begins with a detailed use-case description that informs the assessment of contextual hazards. In this paper, we introduce a structured pipeline that, given a use-case, automatically produces a tailored dataset and an accompanying risk card. The process ensures that identified risks are tightly coupled to the use-case, and it offers traceable evidence from peer-reviewed literature to substantiate each risk to use-case relationship. Prior claim-verification methods (FActSco
    
[^32]: 基于大语言模型的排序函数生成用于终止性证明

    Ranking Function Generation with LLMs for Termination

    [https://arxiv.org/abs/2610.35523](https://arxiv.org/abs/2610.35523)

    该论文提出RAINFALL神经符号框架，通过将排序函数合成转化为带槽位约束的代码补全任务，并结合符号验证器的反馈迭代改进，利用大语言模型自动生成可验证的排序函数以证明循环终止性。

    

    排序函数是证明循环终止性的基础技术，但手动构造排序函数具有挑战性且需要大量专业知识。我们提出了RAINFALL，这是一个神经符号框架，遵循生成-验证-改进范式，将大语言模型与符号验证紧密集成。RAINFALL将排序函数合成任务表述为代码补全任务，通过在输入程序中插入占位符，并约束大语言模型为每个循环填充四个预定义的槽位。补全后的程序随后由符号验证器进行检查。如果验证成功，则获得有效的排序函数；否则，验证器的诊断信息会被转换为自然语言反馈，以指导大语言模型输出的迭代改进。我们在两个公开基准测试套件上对RAINFALL进行了评估，共包含358个可终止的C程序。使用MiniMax V2.7，RAINFALL成功为323个程序（90……）合成了可验证的排序函数

    arXiv:2610.35523v1 Announce Type: cross  Abstract: Ranking functions are a fundamental technique for proving loop termination, but constructing them manually is challenging and requires substantial expertise. We propose RAINFALL, a neuro-symbolic framework which follows the generate-verify-refine paradigm and tightly integrates LLMs with symbolic verification. RAINFALL formulates ranking function synthesis as a code-completion task by inserting placeholders into the input program and constraining the LLM to fill four predefined slots for each loop. The completed program is then checked by a symbolic verifier. If verification succeeds, a valid ranking function is obtained; otherwise, the verifier's diagnostics are translated into natural-language feedback that guides iterative refinement of the LLM's output. We evaluate RAINFALL on two public benchmark suites comprising 358 terminating C programs. Using MiniMax V2.7, RAINFALL synthesizes verifiable ranking functions for 323 programs (90
    
[^33]: DeepAVR：基于多智能体协作的自动化漏洞修复深度分析方法

    DeepAVR: Deep Analysis via Multi-Agent Collaboration for Automated Vulnerability Repair

    [https://arxiv.org/abs/2610.32345](https://arxiv.org/abs/2610.32345)

    DeepAVR提出了一种多智能体协作框架，将中间修复结果转化为结构化诊断反馈，并在故障定位、补丁生成、编译/安全/功能验证与补丁审查等专门角色之间流转反馈，从而实现端到端的深度漏洞分析与自动化修复。

    

    自动化漏洞修复（AVR）已成为抵御软件安全缺陷的关键防线，旨在自主地对漏洞进行分诊与修复。早期基于大语言模型（LLM）的方法在假定故障定位完美无缺的前提下专注于补丁生成，而近期的端到端方法则尝试直接依据消毒器日志来修复漏洞。然而，当根本原因被深深隐藏时——无论是物理上远离崩溃点，还是在逻辑上被相互关联的错误所掩盖——这些方法往往会失败，因为初始报告提供的信息并不充分。我们提出了DeepAVR，一个用于端到端自动化漏洞修复的多智能体框架，它通过将中间修复结果转化为结构化的诊断反馈来执行深度分析。DeepAVR将修复过程分解为专门的智能体角色——故障定位、补丁生成、编译验证、安全验证、功能验证和补丁审查——并在这些角色之间传递反馈以进行迭代……（原文摘要至此截断）

    arXiv:2610.32345v1 Announce Type: new  Abstract: Automated Vulnerability Repair (AVR) has emerged as a critical defense against software security flaws, aiming to triage and fix bugs autonomously. Early LLM-based approaches focused on patch generation under the assumption of perfect fault localization, while recent end-to-end methods attempt to repair vulnerabilities directly from sanitizer logs. However, these methods often fail when the root cause is deeply hidden--either physically distant from the crash site or logically masked by interdependent errors--making the initial report insufficient. We propose DeepAVR, a multi-agent framework for end-to-end AVR that performs deep analysis by transforming intermediate repair outcomes into structured diagnostic feedback. DeepAVR decomposes repair into specialized roles--Fault Localization, Patch Generation, Compile Validation, Security Validation, Functionality Validation, and Patch Review--and routes feedback across these roles to iterativ
    
[^34]: SWE-Proof：语言模型能否通过机器校验的证明解决真实世界的问题？

    SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?

    [https://arxiv.org/abs/2609.21190](https://arxiv.org/abs/2609.21190)

    该论文提出Benchproofer流水线，将SWE-bench中的真实编码任务转化为经过机器校验证明的形式化验证任务，构建了包含500个真实问题的SWE-Proof基准，用形式化验证取代不完整的测试来严格评估语言模型解决真实软件工程问题的能力。

    

    确保大语言模型（LLM）生成代码的正确性是现代软件工程的核心挑战。面向智能体代码生成的基准测试通常使用留出的测试套件来检验正确性，但测试套件本质上是不完整的，且日益容易受到模型记忆（数据泄露）的影响。形式化验证可以同时避免这两个问题，但现有工作仅覆盖规范以输入形式给出的独立任务，而非真实问题——真实问题涉及大型代码仓库，并以模糊的自然语言表达意图。我们提出了Benchproofer，一个能将带有已知正确补丁的编码任务转化为形式化验证任务的流水线：它为新代码编写规范，用公理概括新代码所调用的已有函数，并且只有在机械验证与对抗性检查两道关卡均通过后才接受一个实例。将该流水线应用于SWE-bench Verified，我们得到了SWE-Proof——包含500个真实问题的基准，其正确性通过形式化验证而非测试来保证，并且该方法还可扩展至SWE-bench Pro。……

    arXiv:2609.21190v1 Announce Type: cross  Abstract: Ensuring the correctness of LLM-generated code is a core challenge for modern software engineering. Benchmarks for agentic code generation check correctness with held-out test suites, which are inherently incomplete and increasingly susceptible to memorization. Formal verification avoids both problems, but existing work covers only standalone tasks whose specifications are given as input, not real issues, which touch large repositories and state intent in vague natural language. We present Benchproofer, a pipeline that turns a coding task with a known correct patch into a formally verified one: it writes a specification for the new code, summarizes the existing functions that code calls with axioms, and admits an instance only after mechanical and adversarial gates agree. Applying it to SWE-bench Verified yields SWE-Proof, 500 real issues whose correctness is formally verified rather than tested, and it extends to SWE-bench Pro. Across
    
[^35]: PRAXIS：面向领域代码生成的图基隐性知识

    PRAXIS: Graph-Grounded Tacit Knowledge for Domain Code Generation

    [https://arxiv.org/abs/2608.19784](https://arxiv.org/abs/2608.19784)

    PRAXIS通过模拟人类开发工作流，从代码库中提取隐性知识并将其组织为图结构，使LLM代理能够有效生成领域特定代码。

    

    arXiv:2608.19784v1 公告类型：新 摘要：LLM代理在通用软件工程任务上表现出色，但在特定领域的代码生成方面却面临困难。我们确定根本原因在于代理缺乏隐性知识，包括领域特定的业务规则、接口契约和操作惯例，这些知识是开发者在实践中内化但从未记录下来的。这些知识深埋在领域代码之下，分散在代码实体及其依赖关系中，对于缺乏它的代理来说是不可见的。这些特性使得隐性知识本质上难以检索或学习。在这项工作中，我们提出PRAXIS，一个框架，使代理能够系统地提取、表示和重用隐性知识以进行领域代码生成。PRAXIS通过模拟目标代码库中的人类开发工作流来获取隐性知识，将其提炼为结构化单元，组织在代码依赖图上，并主动提供这些知识。

    arXiv:2608.19784v1 Announce Type: new  Abstract: LLM agents have achieved strong performance on general software engineering tasks, yet struggle with domain-specific code generation. We identify the root cause as the agent's lack of tacit knowledge, including domain-specific business rules, interface contracts, and operational conventions that developers internalize through practice but never document. This knowledge is deeply buried beneath the domain code, dispersed across code entities and their dependency relations, and invisible to the agent that lacks it. These properties make tacit knowledge inherently difficult to retrieve or learn. In this work, we propose PRAXIS, a framework that enables agents to systematically extract, represent, and reuse tacit knowledge for domain code generation. PRAXIS acquires tacit knowledge by simulating human development workflows within the target codebase, distills it into structured units organized on the code dependency graph, and proactively su
    
[^36]: 基于神经与结构方法的命令式程序图构建与匹配

    Graph Construction and Matching for Imperative Programs using Neural and Structural Methods

    [https://arxiv.org/abs/2604.26578](https://arxiv.org/abs/2604.26578)

    该论文提出一条将命令式程序及其标注转换为带类型属性图的流水线，通过结合抽象语法树解析与SentenceTransformer、CodeBERT等模型的语义嵌入，在C/ACSL、Java/JML和Dafny等多种语言上实现了跨语言、跨标注风格的一致性图表示，为验证产物复用和近似图匹配奠定了实用基础。

    

    复用验证产物需要识别程序及其规约之间的结构与语义相似性。在本文中，我们将图构建作为实现这一目标的基础步骤。我们提出了一条流水线，可将命令式程序及其标注转换为带类型的属性图。我们的实验涵盖多个数据集，包括带ACSL标注的C语言程序、带JML标注的Java程序以及Dafny程序。该流水线将抽象语法树解析与来自SentenceTransformer和CodeBERT等模型的语义嵌入相结合，从而能够生成同时捕捉结构关系和语义上下文的图表示。我们的结果表明，可以在不同语言和标注风格之间构建一致的图表示。这项工作为未来的语义增强和近似图匹配以实现可扩展的验证产物复用提供了实用基础。

    arXiv:2604.26578v4 Announce Type: replace-cross  Abstract: Reusing verification artefacts requires identifying structural and semantic similarities across programs and their specifications. In this paper, we focus on graph construction as a foundational step toward this goal. We present a pipeline that converts imperative programs and their annotations into typed, attributed graphs. Our experiments cover datasets including C with ACSL, Java with JML, and Dafny programs. The pipeline integrates abstract syntax tree parsing with semantic embeddings derived from models such as SentenceTransformer and CodeBERT. This enables the generation of graph representations that capture both structural relationships and semantic context. Our results show that consistent graph representations can be constructed across different languages and annotation styles. This work provides a practical basis for future steps in semantic enrichment and approximate graph matching for scalable verification artefact 
    
[^37]: ChroniUXMag：一个以用户画像驱动的包容性移动健康需求工程框架

    ChroniUXMag: A Persona-Driven Framework for Inclusive mHealth Requirements Engineering

    [https://arxiv.org/abs/2511.18634](https://arxiv.org/abs/2511.18634)

    ChroniUXMag是一个建立在InclusiveMag和GenderMag基础之上、以用户画像驱动的框架，通过文献综述、焦点小组、访谈和调查识别包容性需求维度，并将其整合到认知走查中，用于改进慢性病管理移动健康应用的包容性需求工程。

    

    背景：用于慢性病管理的移动健康应用在可访问性、包容性、可用性和持续用户参与方面存在挑战。临床情境、照护者参与、信任和长期使用带来了特定领域的需求工程（RE）挑战。目标：本研究提出ChroniUXMag，这是一个用于在移动健康设计中引出和分析包容性需求的框架，它建立在InclusiveMag和GenderMag的基础之上。方法：ChroniUXMag采用InclusiveMag的范围界定和推导两个阶段开发。证据来源于系统性文献综述、焦点小组、访谈和在线调查。研究从文献和实证数据中识别出候选的包容性维度，将其组织成代表不同健康状况、态度和数字实践的用户画像，并整合到改编后的认知走查表单中。结果：十三个定性类别被整合提炼……

    arXiv:2511.18634v2 Announce Type: replace  Abstract: Context: Mobile health (mHealth) applications for chronic disease management present challenges in accessibility, inclusivity, usability, and sustained engagement. Clinical context, caregiver involvement, trust, and long-term use create domain-specific requirements engineering (RE) challenges.   Objective: This study introduces ChroniUXMag, a framework for eliciting and analysing inclusivity requirements in mHealth design, building on InclusiveMag and GenderMag.   Method: ChroniUXMag was developed using the Scope and Derive stages of InclusiveMag. Evidence came from a systematic literature review, focus groups, interviews, and an online survey. Candidate inclusivity facets were identified from literature and empirical data, organised into personas representing diverse health situations, attitudes, and digital practices, and integrated into an adapted cognitive walkthrough form.   Results: Thirteen qualitative categories were consolid
    
[^38]: Web Fuzzing Commons（网络模糊测试公共库）、数据集、指南与评估工具：支持REST API模糊测试的实验研究

    Web Fuzzing Commons, Dataset, Guidelines and Evaluations to Support Experimentation in REST API Fuzzing

    [https://arxiv.org/abs/2509.01612](https://arxiv.org/abs/2509.01612)

    本文提出了Web Fuzzing Commons（WFC）、Web Fuzzing Dataset（WFD）和Web Fuzzing Evaluations（WFE）三个开源组件，通过解决API身份验证处理、故障类型编目比较和标准化案例研究三大挑战，为REST API模糊测试研究提供了可公平比较的实验基础设施。

    

    模糊测试REST API是一个重要的研究问题，在工业界具有实际应用和影响力。因此，过去几年中围绕这一主题开展了大量研究工作。然而，有三个主要问题阻碍了该领域的进一步发展：（1）如何处理API身份验证；（2）如何对不同模糊测试工具发现的不同类型故障进行编目和比较；（3）使用什么样的案例研究来促进模糊测试工具之间的公平比较。在本文中，为应对这些重要挑战，我们提出了Web Fuzzing Commons（WFC）、Web Fuzzing Dataset（WFD）和Web Fuzzing Evaluations（WFE）。WFC是一套开源库和模式定义，用于声明式地指定身份验证信息，并对模糊测试工具能够自动检测的不同类型故障进行编目。WFD是一个包含56个开源API的集合，配置为通过Docker-Compose运行，并由WFC提供支持。WFE提供了脚手架脚本以自动（完成评估任务）。

    arXiv:2509.01612v2 Announce Type: replace  Abstract: Fuzzing REST APIs is an important research problem, with practical applications and impact in industry. As such, lot of research work has been carried out on this topic in the last few years. However, there are three major issues that hinder further progress: (1) how to deal with API authentication; (2) how to catalog and compare different fault types found by different fuzzers; (3) what to use as case study to facilitate fair comparisons among fuzzers.   In this paper, to address these important challenges, we present Web Fuzzing Commons (WFC), Web Fuzzing Dataset (WFD) and Web Fuzzing Evaluations (WFE). WFC is a set of open-source libraries and schema definitions to declaratively specify authentication info and catalog different types of faults that fuzzers can automatically detect. WFD is a collection of 56 open-source APIs, configured to run with Docker-Compose, supported by WFC. WFE provides scaffolding scripts to automatically 
    

