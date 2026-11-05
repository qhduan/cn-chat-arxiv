# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MCP-Eco: A Linked Dataset of MCP Servers, Consumers, and Development Activity](https://arxiv.org/abs/2611.06897) | MCP-Eco是首个整合MCP服务器接口、消费者配置声明与开发活动数据的关系型数据集，包含近1,700万条记录，为系统研究MCP软件生态系统提供了数据基础。 |
| [^2] | [GH-AWTrace: A Large-Scale Dataset of GitHub Agentic Workflow Execution Logs](https://arxiv.org/abs/2611.06866) | GH-AWTrace 是一个大规模 GitHub 智能体工作流执行日志数据集，将 25,014 个经验证的工作流与超过 411 万条运行记录及 56 万余份执行日志关联起来，为研究智能体工作流的执行失败、工具使用以及运行结果与实际任务进展之间的差异提供了实证依据。 |
| [^3] | [Rethinking Access Control for AI Agents: Recent Advances and Open Challenges](https://arxiv.org/abs/2611.06764) | 本文系统综述了AI智能体访问控制在策略规范、身份委托、动作执行和信息流控制四大领域的最新进展，并重新思考了授权机制的四个核心方面，指出了未来亟待解决的开放性挑战。 |
| [^4] | [AgentTrustCI: Build-Time Policy, Grounding, and Trajectory Tests for Java LLM Agents](https://arxiv.org/abs/2611.06674) | AgentTrustCI提出了一个Java/JUnit 5框架，将策略、接地与轨迹检查以确定性重放的形式转化为可重复的构建时测试，从而在应用层验证LLM智能体的工具选择、答案依据与工作流程合规性，并在55个合成案例上准确分类了全部注入故障。 |
| [^5] | [PatchCon: Patch-aware Weighted Supervised Contrastive Learning for Vulnerability Detection](https://arxiv.org/abs/2611.06583) | 提出PatchCon，一种基于CodeBERT的补丁感知加权监督对比学习模型，通过将漏洞函数与修复后函数对作为高权重困难负样本进行联合优化，显著提升了函数级漏洞检测的精度。 |
| [^6] | [EPISTEME: Trustworthy Coding Agents Through Tool Call Intent](https://arxiv.org/abs/2611.06563) | 提出认识论智能体设计，强制模型声明每次工具调用的意图并与实际执行效果核对，形成可事前把关、事后审计的日志，使两开源模型49次奖励劫持作弊中90%被暴露。 |
| [^7] | [Construction Contracts for Generated Embedded Domain-Specific Language Programs: A Measurement Protocol and a Conditional Effect](https://arxiv.org/abs/2611.06525) | 该论文提出区分“程序可解析”与“匹配参考构建”这两种不同事件的测量协议，并发现一种简短的“构建契约卡片”对OpenQASM 3量子程序生成的效果具有条件性：生成器经过卡片训练时完全匹配率提升32个百分点，未经训练时反而下降13个百分点。 |
| [^8] | [Scythe: Semantic-Aware Rewriting for Fast,Effective Program Reduction](https://arxiv.org/abs/2611.06515) | SCYTHE 是一种语义感知的预处理步骤，通过将程序编码为带类型的声明依赖图并解耦声明与其使用位置，使任何基于文法的程序约简器都能更快速、更有效地删除完整的声明。 |
| [^9] | [HELIX: HEterogeneous LLM Inference with FleXible and Fine-Grained Partitioning](https://arxiv.org/abs/2611.06513) | HELIX是一个基于vLLM的异构LLM推理软件栈，能够在算子粒度上自动发现、编译并调度工作负载感知的分区，以最小化异构加速器上的执行延迟。 |
| [^10] | [Trace2Eval: Contextualized Evaluation of Coding Agents from Real-World Traces](https://arxiv.org/abs/2611.06506) | 提出了Trace2Eval，一个可动态更新的流水线，通过多智能体重建系统从真实世界的交互轨迹中恢复用户查询、工作区状态和执行环境，从而对编码智能体在真实用户工作流程中的产出质量进行情境化、持续更新的评估。 |
| [^11] | [Beyond Infraction Scores: Failure-Boundary Profiles for Closed-Loop Evaluation of Driving and Robot Navigation Agents](https://arxiv.org/abs/2611.06453) | 该论文提出“失败边界剖面”评估框架，通过逐级变化单一可测量因素（如走廊宽度、对车间隔）并认证每个变体应为“通过”或“停止”，进而估计智能体的承诺曲线与能力曲线及其临界点，弥补了CARLA传统违规评分无法区分“该停就停”与“强行通过”两类智能体的根本缺陷。 |
| [^12] | [Reassessing Dataset Quality and Fine-Tuning Practices for Vulnerability Detection on PrimeVul dataset](https://arxiv.org/abs/2611.06373) | 该论文通过对770次训练的大规模实证评估发现，在PrimeVul数据集上微调的代码语言模型虽然在F1和ROC-AUC等标准指标上看似有改进，但在安全关键指标上（MCC接近零、概率校准差、严格假阳性约束下假阴性率高）实际漏洞检测能力远低于表面评估结果。 |
| [^13] | [Keeping Fuzzers on Duty: Repairing Fuzzing Build Breakages in Continuous Open-Source Fuzzing](https://arxiv.org/abs/2611.06358) | 提出 FuzzMedic，一个通过跨项目与基础设施构件的证据引导诊断、并在修复状态图上探索修复方案，从而自动修复 OSS-Fuzz 等持续开源模糊测试平台构建故障的自动化智能体。 |
| [^14] | [EvoCodeAgent: Project-Level Code Generation via Architecture-Code Co-Evolution and Experience-Guided Repair](https://arxiv.org/abs/2611.06328) | 提出EvoCodeAgent，通过架构-代码协同演化和经验引导修复两大机制实现项目级代码生成，解决了静态架构无法适应迭代生成、跨迭代经验无法积累复用、以及独立LLM评估不可靠等关键问题。 |
| [^15] | [A.S.E 2.0: Understanding How Repository Context Impacts the Security of LLM-Generated Code](https://arxiv.org/abs/2611.06294) | 该论文提出CASE-Bench基准测试，基于真实CVE构建145个仓库级任务，首次系统研究了仓库上下文构建方式如何影响LLM生成代码的安全性。 |
| [^16] | [CLEAR: Structured Specifications for Controllable Long-Horizon ML Research Engineering](https://arxiv.org/abs/2611.06263) | 提出了CLEAR结构化规范方法，将隐式定义的研究成果转化为显式、结构化的规格说明，从而实现可控的长时程机器学习研究工程自动化。 |
| [^17] | [Analysis of Local Open-Weight LLM Agents on Vulnerability-Reproduction Tasks](https://arxiv.org/abs/2611.06257) | 本研究系统评测了十二种本地开源权重LLM配置在CyberGym漏洞复现任务上的表现，发现即使小至4B的模型也能解决大部分较简单的漏洞复现任务，表明漏洞复现是本地小模型可胜任的自动化分析任务。 |
| [^18] | [Bug2Scenario: Root-Cause-Preserving Test Scenario Amplification for Autonomous Driving Software](https://arxiv.org/abs/2611.06229) | Bug2Scenario是一个根因保持式的后模糊测试场景扩增框架，通过从故障场景中提取根因模式并利用大语言模型生成语义变体，将单次自动驾驶系统故障转化为可复用的回归测试场景。 |
| [^19] | [CoSafe: Cooperative Runtime Safety Assurance for Mixed-Traffic Autonomous Driving](https://arxiv.org/abs/2611.06204) | 提出CoSafe——一个模块化、与自动驾驶系统无关的协同式运行时安全保障框架，通过多车协作解决混合交通中由局部合理但联合不兼容决策引发的安全违规问题。 |
| [^20] | [SBVUL: Statement-Based Vulnerability Detection](https://arxiv.org/abs/2611.06135) | SBVul提出了一种两阶段智能体方法，通过LLM智能体探索函数与代码仓库来识别候选语句，并结合控制流和数据流分析，直接定位C/C++函数中可利用的漏洞语句，弥补了函数级漏洞检测无法精确定位漏洞位置的不足。 |
| [^21] | [Growing One's AI Tools: A Paradigm of Practice-Grown Tools](https://arxiv.org/abs/2611.06132) | 本文提出“实践生长型工具”新范式：一位无编程背景的会议口译员依据真实口译工作需求自行培育出计算机辅助口译工具，展现了工具与实践通过故障与修复共同演化的过程，并主张应将工具纳入对从业者的评估体系。 |
| [^22] | [SWEOps-Bench: Evaluating Agents for End-to-End Resolution of Software-Induced Cloud Incidents](https://arxiv.org/abs/2611.06099) | 该论文提出了SWEOps-Bench，一个包含55个源自16个Kubernetes operator项目真实历史缺陷的可复现事故任务的交互式基准，用于评估智能体在统一时间预算内端到端完成云事故的检测、定位与修复的能力。 |
| [^23] | [Reasoning-Trace Overflow in Code Generation: An Empirical Study of Explicit Reasoning Models](https://arxiv.org/abs/2611.06095) | 该研究提出“推理轨迹溢出”概念，实证发现显式推理模型在较难的代码生成任务上常常将全部输出预算耗尽于推理而无法产出可用代码，而轻量级的推理时控制可以缓解这一问题。 |
| [^24] | [SynMOE: A Syntax-Aware MOE-Guided Approach for PEFT-Based Code Intelligence Learning](https://arxiv.org/abs/2611.06069) | 提出了SynMOE，一种语法感知的MOE引导参数高效微调方法，通过从抽象语法树中提取句法标签并将其融入混合专家适配过程，弥补了现有PEFT方法缺乏代码信息显式建模的不足，从而提升代码智能任务的表现。 |
| [^25] | [When Does a Cross-Device Run Justify PASS? Obligation-Level Evidence Qualification under Partial Observability](https://arxiv.org/abs/2611.06013) | 该论文提出一种可执行的证据契约，将证据支持资格认定作为判定求值前的显式阶段，通过把义务绑定到可采纳的主体、情节、对象和观测范围并记录未解决的支持，有效防止跨设备测试中因采用其他设备或尝试的证据而产生的假 PASS 判定。 |
| [^26] | [Retrieval Supersedes Scale: Efficient Log Level Prediction via Multiplex Socio-Technical Context](https://arxiv.org/abs/2611.05901) | 该论文提出基于多重社会技术上下文的检索方法来选择上下文学习示例，从而在不微调的情况下显著提升 LLM 的日志级别预测效果，证明了检索策略优于单纯依赖模型规模。 |
| [^27] | [Can LLM Agents Learn to Test? Maintaining Scoped Experience as Testing Evidence for Mobile GUI Testing](https://arxiv.org/abs/2611.05873) | 该论文为基于LLM的移动GUI测试提出了一种经验维护契约，赋予可复用测试经验明确的适用范围、来源追溯及确认—重新验证—撤销的生命周期，使测试器能在复用历史经验的同时排除与当前观察相矛盾的过时条目。 |
| [^28] | [A Hybrid Approach for Automated Security Pattern Recovery](https://arxiv.org/abs/2611.05823) | 本文提出SecPatRec，一种将ReAct风格LLM智能体与静态代码属性图分析相结合的混合方法，实现了从源代码仓库中自动恢复安全模式的角色检测、代码落地与交互序列识别。 |
| [^29] | [DARE: Difficulty Analysis for Repair Enhancement](https://arxiv.org/abs/2611.05797) | 提出DARE框架，利用从基准测试产物中提取的151个静态特征预测编程修复任务的难度（定义为多个智能体的观察失败率），并结合七个特征组的贡献分析，为编程智能体提供任务特定的修复支持。 |
| [^30] | [How Much Do Modern Embeddings Reveal? Inversion Attacks on Commercial Embedding Models](https://arxiv.org/abs/2611.05716) | 本研究将vec2text反演攻击框架适配到OpenAI text-embedding-3-small和mistral-embed等商业嵌入模型，发现这些商业模型的嵌入向量可被高精度还原为原始文本（如text-embedding-3-small在健康类指令上精确匹配率达47.4%），揭示了仅存储嵌入向量的系统存在严重的隐私泄露风险。 |
| [^31] | [TACT: A Pragmatic Approach to Qualitative Analysis with an AI Council](https://arxiv.org/abs/2611.05702) | 该论文提出了TACT方法，通过由多个AI模型组成的可追溯委员会进行大规模主题分析，使软件团队在利用AI进行定性数据分析的同时，仍能保持对解释和判断的掌控权，并通过对抗性批评和不确定性标注进一步保证分析质量。 |
| [^32] | [Construct Validity of Coding Benchmarks for LLM-based Agentic Coding Tools](https://arxiv.org/abs/2611.05665) | 本文通过分析九个常用的智能体编程基准测试的构念效度，发现这些基准测试往往仅用少量指标来操作化“编程能力”等宽泛概念，并存在干扰测量的混淆因素，从而揭示了现有基准测试在效度上的不足。 |
| [^33] | [AQUA - A Modular Authentication Quality Modeling Framework](https://arxiv.org/abs/2611.05660) | 本文提出AQUA框架，通过将认证分解为四个相互关联的概念，实现对认证解决方案质量的模块化建模，并能将质量影响追溯至其具体来源。 |
| [^34] | [ViBench 2.0: A Benchmark for Long-Horizon Software Building](https://arxiv.org/abs/2611.05620) | ViBench 2.0 是一个用于长周期软件构建的基准测试，通过 17 个随 241 个功能需求逐步演进的全栈 Web 产品和 544 个浏览器智能体测试计划进行严格评估，揭示了最强模型通过高达 98% 的测试计划却仅有最多 65% 的构建能交付完全可用产品的可靠性差距。 |
| [^35] | [RIDGE: State-Conditioned Reward Blending for Behavioral Coverage in Deep RL Game Agents](https://arxiv.org/abs/2611.05601) | RIDGE 通过状态条件奖励混合器和多头评论家架构，让单个深度强化学习游戏智能体同时掌握多种行为风格，从而规避了 PPO 在非平稳奖励下的性能退化问题。 |
| [^36] | [ProofStudio: Five Proof Assistants in the Browser](https://arxiv.org/abs/2611.05540) | ProofStudio是一个基于浏览器的集成环境，将Isabelle/HOL、Lean、Rocq、HOL4和HOL Light五种证明助手统一到同一界面中，无需本地安装即可降低形式化规约与证明的学习门槛并支持自动化实验。 |
| [^37] | [Hybrid LLM-Deterministic Pipelines for Enterprise-Scale Code Migration at Bloomberg](https://arxiv.org/abs/2611.05457) | 彭博提出的混合 LLM-确定性流水线架构，通过将 LLM 限定在预计算上下文内进行有界文件转换并结合生产遥测反馈，在约 8,600 个企业仓库的两大迁移行动中实现了 729 次成功代码迁移。 |
| [^38] | [Agentic Behavior Engineering: A Design-by-Contract Methodology for Specifying, Verifying, and Governing Enterprise AI Agents](https://arxiv.org/abs/2611.05454) | 本文提出智能体行为工程（ABE），一种将可观察的智能体行为而非提示词或模型作为版本控制工程制品的契约式设计方法论，用于解决企业LLM智能体因源自团队外部的行为变更而导致的维护与治理难题。 |
| [^39] | [LIBRA: Programmatic Induction of Executable Tool Libraries](https://arxiv.org/abs/2611.05445) | LIBRA 通过让大语言模型提出工具间契约并在沙盒中进行运行时验证，将扁平的工具目录自动归纳为可执行的工具库，从而在多工具任务中检索出包含先决工具的完整工作流，无需依赖标注或任务特定训练，在多个基准上显著提升了检索性能。 |
| [^40] | [Req-Space: A User Study of Interactive Visualisations for Requirements Review](https://arxiv.org/abs/2611.05437) | 本文提出Req-Space交互式原型，通过集成语义探索与可追溯性可视化来辅助需求评审，用户研究表明该工具能帮助评审者发现更多有源文档支持的额外问题并强化决策理由。 |
| [^41] | [LLM-based NPCs in Video Games: A Systematic Mapping Study](https://arxiv.org/abs/2611.05427) | 该论文通过对42篇相关文献的系统性映射研究，首次从组件采用率、延迟、安全覆盖和可复现性四个维度对基于大语言模型的游戏NPC领域进行了量化刻画。 |
| [^42] | [AI-Driven SaaS Platform Integration: AI Express for Unified Generative AI Services with Pay-As-You-Go Credit Systems and Scalable Microservice Architecture](https://arxiv.org/abs/2611.05389) | AI Express通过按量付费积分系统和可扩展微服务架构，将碎片化的生成式AI能力（代码审查、文本转网站、语法检查等）整合到单一SaaS平台中，使用户无需多重订阅和技术专长即可低成本使用统一的AI服务。 |
| [^43] | [Extending the MM-AR Metamodeling Platform with Process Mining Capabilities](https://arxiv.org/abs/2611.05176) | 本文首次提出概念验证方案，通过基于PM4Py的独立Python服务，在不修改平台核心的前提下，为XR元建模平台MM-AR扩展了过程发现能力，使从事件日志中挖掘出的Petri网和BPMN模型能够直接在沉浸式2D/3D建模环境中展示。 |
| [^44] | [Trace2World: Reconstructing App Worlds from Real Interaction Traces for Scalable GUI Agent Training](https://arxiv.org/abs/2610.30792) | Trace2World通过将全局行为重建与环境实现分离，从真实交互轨迹构建忠实于真实应用行为的可执行GUI环境用于智能体训练，在十个应用的200个工作流上实现91.1%的端到端复现率，且基于来源的验证使真实设备上的训练增益保留率达94.0%。 |
| [^45] | [X.js: A Live Benchmark for Node.js Taint-Style Vulnerabilities](https://arxiv.org/abs/2610.18585) | 本文提出了X.js——一个持续更新的真实世界Node.js污点风格漏洞基准测试集，每个漏洞均配有精确的源汇位置、经验证的可执行漏洞利用程序及补丁版本，解决了现有漏洞数据集过时、无法支持该领域研究长期发展的问题。 |
| [^46] | [Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems](https://arxiv.org/abs/2608.21095) | 本文提出了一种结合NLI事实核查与五信号投毒检测的评估代理，并引入信任指数，在TruthfulQA上实现了高准确率与精确率，有效缓解了RAG系统中的知识投毒风险。 |
| [^47] | [TraceCAD: Trace-Guided Repair for Agentic CAD Generation](https://arxiv.org/abs/2608.03062) | TraceCAD通过将需求、建模步骤、失败证据与修复结果链接为持久轨迹状态并配合可复用的技能记忆，实现对LLM生成的CAD程序的高效诊断与局部化修复，显著减少几何退化并大幅降低智能体调用开销。 |
| [^48] | [No Edges, No Verdict: A Large-Scale Empirical Study of Declared Dependency Graphs in 78K SBOMs in the Wild](https://arxiv.org/abs/2607.22140) | 这项研究首次大规模分析了78,612个真实SBOM中的依赖图，发现超过半数未声明任何边缘，且边缘生成主要由生成器决定而非软件本身，这对漏洞分析有重要影响。 |
| [^49] | [TTHE: Test-Time Harness Evolution](https://arxiv.org/abs/2607.08124) | TTHE提出在测试阶段利用无标签执行轨迹持续演化LLM智能体的脚手架程序，通过智能体提议者生成多样化候选并由裁判筛选，从而在无需标签的情况下实现对任务分布与失败模式变化的实时自适应。 |
| [^50] | [Auditing Empirical Comparisons in Quantum Software](https://arxiv.org/abs/2607.00516) | 提出了CLAIMSTAB-QC审计框架，通过在计算结果前锁定审计设计、限定证据来源边界，将量子软件论文中的455条实证比较声明分类为“维持”、“未决”或“未保留”，从而检验比较结论是否有证据支撑。 |
| [^51] | [Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code](https://arxiv.org/abs/2606.11817) | 该论文揭示了语法约束解码（GCD）这一可靠性增强技术可被反直觉地利用为越狱攻击面，提出了 CodeSpear 攻击使大语言模型生成恶意代码，并进一步提出 CodeShield 安全对齐方法，使模型在攻击者控制的语法约束下仍能保持安全行为。 |
| [^52] | [Ekka: Automated Diagnosis of Silent Errors in LLM Inference](https://arxiv.org/abs/2606.04594) | Ekka通过差分对比目标与参考LLM服务框架的中间执行状态，将静默错误诊断转化为差分调试问题，实现了自动化的根因定位，在真实错误基准上达到80%的pass@1诊断准确率。 |
| [^53] | [The Impact of AI Coding Assistants on Software Engineering: A Longitudinal Study](https://arxiv.org/abs/2605.23135) | 该纵向研究发现，AI编程助手使开发者将工作重心从代码创造转向验证活动，并首次提出“监督性工程工作”这一新概念，即对AI输出进行指导、评估和纠正的新型工作模式。 |
| [^54] | [The Dangers of Non-Self-Fixed Architecture Technical Debt and Its Impact on Time-to-Fix](https://arxiv.org/abs/2605.16133) | 该论文通过对大型Apache开源项目中架构技术债生命周期的实证分析，发现需要由引入者之外的人偿还的非自修复架构技术债往往导致更长的修复时间，从而揭示了其潜在危害。 |
| [^55] | [From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection](https://arxiv.org/abs/2512.10485) | 本文系统评估了四种深度学习漏洞检测模型在多个数据集上的表现，通过跨数据集实验和可视化分析揭示了基准测试与真实世界应用之间的性能差距，强调了模型泛化能力的挑战。 |
| [^56] | [How the Misuse of a Dataset Harmed Semantic Clone Detection](https://arxiv.org/abs/2505.04311) | BigCloneBench数据集的WT3/T4子集存在系统性标注错误（约93.9%的克隆对功能并不相似），导致128篇使用该数据集评估语义克隆检测的研究结果的可信度受到严重损害。 |
| [^57] | [Automated Exploit and Security Test Generation: A Systematic Literature Review](https://arxiv.org/abs/2502.04953) | 本文对过去二十年的71篇论文（67项独立研究）进行了系统性文献综述，建立了自动漏洞利用生成（AEG）与自动安全测试生成（ASTG）领域的统一术语体系，并系统比较了现有方法。 |

# 详细

[^1]: MCP-Eco：MCP服务器、消费者与开发活动的关联数据集

    MCP-Eco: A Linked Dataset of MCP Servers, Consumers, and Development Activity

    [https://arxiv.org/abs/2611.06897](https://arxiv.org/abs/2611.06897)

    MCP-Eco是首个整合MCP服务器接口、消费者配置声明与开发活动数据的关系型数据集，包含近1,700万条记录，为系统研究MCP软件生态系统提供了数据基础。

    

    模型上下文协议使LLM智能体能够访问外部工具和数据。研究其软件生态系统需要将服务器接口、配置这些服务器的项目及其开发活动关联起来，然而这些信息分散在目录、包注册表、配置文件和代码库历史之中。我们提出了MCP-Eco，一个整合上述来源的关系型数据集，将消费者配置声明与候选服务器项目链接起来。其冻结版本发布包含19个Parquet表中的16,880,899条记录，涵盖78,302个候选服务器项目、16,165个候选消费代码库以及9,200条配置声明。其中5,975条声明解析为5,138对不同的代码库-服务器配对。该数据集还包含工具、资源和提示词描述、声明的依赖关系，以及相关的开发记录，包括拉取请求、代码审查和议题等。

    arXiv:2611.06897v1 Announce Type: new  Abstract: The Model Context Protocol (MCP) enables LLM agents to access external tools and data. Studying its software ecosystem requires connecting server interfaces, the projects that configure these servers, and their development activity. However, this information is scattered across directories, package registries, configuration files, and repository histories. We present MCP-Eco, a relational dataset that integrates these sources and links consumer configuration declarations to candidate server projects. Its frozen release contains 16,880,899 records across 19 Parquet tables, covering 78,302 candidate server projects, 16,165 candidate consuming repositories, and 9,200 configuration declarations. Of these declarations, 5,975 resolve to 5,138 distinct repository-server pairs. The dataset also includes tool, resource, and prompt descriptions, declared dependencies, and associated development records, including pull requests, reviews, issues, an
    
[^2]: GH-AWTrace：GitHub 智能体工作流执行日志的大规模数据集

    GH-AWTrace: A Large-Scale Dataset of GitHub Agentic Workflow Execution Logs

    [https://arxiv.org/abs/2611.06866](https://arxiv.org/abs/2611.06866)

    GH-AWTrace 是一个大规模 GitHub 智能体工作流执行日志数据集，将 25,014 个经验证的工作流与超过 411 万条运行记录及 56 万余份执行日志关联起来，为研究智能体工作流的执行失败、工具使用以及运行结果与实际任务进展之间的差异提供了实证依据。

    

    智能体工作流使 AI 智能体能够按计划或在响应仓库事件时执行重复性的仓库任务。现有研究主要考察工作流的定义和输出，但对执行过程中实际发生的情况提供的证据有限。我们提出了 GH-AWTrace，这是一个数据集，将分布在 5,051 个仓库中的 25,014 个经过验证的 GitHub Agentic Workflows 与 4,112,526 条运行记录以及 562,877 个非空执行日志档案相关联。该数据集在 GitHub 日志保留限制允许的范围内保存了执行证据，并将其与仓库元数据和生成的工作流配置相链接。示例案例展示了这些日志如何揭示工具拒绝、提供商故障、智能体策略的变化以及智能体之间的任务委托。这些观察为研究执行失败、工具使用，以及报告的运行结果与可见任务进展之间的关系提供了支持。各工作流的覆盖程度有所不同，日志提供的是部分观察而非完整的记录。

    arXiv:2611.06866v1 Announce Type: new  Abstract: Agentic workflows enable AI agents to perform recurring repository tasks on schedules or in response to repository events. Existing studies examine workflow definitions and outputs, but provide limited evidence of what happens during execution. We present GH-AWTrace, a dataset linking 25,014 verified GitHub Agentic Workflows across 5,051 repositories to 4,112,526 run records and 562,877 nonempty execution log archives. The dataset preserves execution evidence subject to GitHub's log-retention limits and links it to repository metadata and generated workflow configurations. Illustrative cases show how the logs expose tool denials, provider failures, changes in an agent's approach, and delegation among agents. These observations enable research on execution failures, tool use, and the relationship between reported run outcomes and visible task progress. Coverage varies across workflows, and the logs provide partial observations rather than
    
[^3]: 重新思考AI智能体的访问控制：最新进展与开放挑战

    Rethinking Access Control for AI Agents: Recent Advances and Open Challenges

    [https://arxiv.org/abs/2611.06764](https://arxiv.org/abs/2611.06764)

    本文系统综述了AI智能体访问控制在策略规范、身份委托、动作执行和信息流控制四大领域的最新进展，并重新思考了授权机制的四个核心方面，指出了未来亟待解决的开放性挑战。

    

    大型语言模型智能体可以访问数据、调用工具并代表用户执行操作。确保这些行为保持在授权范围内具有挑战性，因为智能体会自行选择执行路径，而权限和运行条件也可能随时变化。本文从四个领域综述了智能体访问控制的最新进展：策略规范与推导、身份与委托、动作与资源执行，以及信息流控制。我们比较了各机制的控制对象、执行边界和前提假设，以审视单个机制所能实现的功能及其协同工作的方式。基于这一分析，我们重新思考了授权的四个方面：计划选择决定了权限需求；上下文信息可以为决策提供依据，但其本身并不授予权限；限制必须在系统接口之间保持其含义；并且随着工作流的演变，必须对授权进行重新评估。这些见解引出了该领域的开放性挑战。

    arXiv:2611.06764v1 Announce Type: new  Abstract: Large language model agents access data, invoke tools, and act on users' behalf. Keeping these actions within authorized bounds is challenging because agents choose their own execution paths while permissions and operating conditions may change. This paper surveys recent advances in agent access control across four areas: policy specification and derivation, identity and delegation, action and resource enforcement, and information flow control. We compare their control objects, enforcement boundaries, and assumptions to examine what individual mechanisms achieve and how they can work together. Building on this analysis, we rethink four aspects of authorization: plan selection shapes permission requirements; contextual information can inform decisions without itself granting authority; restrictions must retain their meaning across system interfaces; and authorization must be reassessed as workflows evolve. These insights motivate open cha
    
[^4]: AgentTrustCI：面向Java LLM智能体的构建时策略、接地与轨迹测试

    AgentTrustCI: Build-Time Policy, Grounding, and Trajectory Tests for Java LLM Agents

    [https://arxiv.org/abs/2611.06674](https://arxiv.org/abs/2611.06674)

    AgentTrustCI提出了一个Java/JUnit 5框架，将策略、接地与轨迹检查以确定性重放的形式转化为可重复的构建时测试，从而在应用层验证LLM智能体的工具选择、答案依据与工作流程合规性，并在55个合成案例上准确分类了全部注入故障。

    

    基础设施安全并不能确定一个LLM智能体是否选择了被允许的工具、是否将其回答建立在工具输出之上，或是否遵循了必需的工作流程。AgentTrustCI是一个Java/JUnit 5框架，它将这些应用层期望表达为针对记录轨迹的策略检查、接地检查和轨迹检查。确定性重放实现支持可重复的构建时测试；裁判接口则为可选的语义评估提供了扩展点。在涵盖七个企业领域的55个合成案例上，完整流水线正确分类了全部30个注入故障和25个通过的控制案例（包括十个非平凡控制案例），并且在12个指定的关键案例中无一遗漏。每个单层配置正确分类了35个案例，因为语料库刻意为每一层隔离了十个故障。一个通过不允许的序列获得的接地回答说明了轨迹检查带来的额外覆盖能力。（摘要在此处截断）

    arXiv:2611.06674v1 Announce Type: cross  Abstract: Infrastructure security does not establish whether an LLM agent selected an allowed tool, grounded its answer in tool output, or followed a required workflow. AgentTrustCI is a Java/JUnit 5 framework that expresses these application-layer expectations as policy, grounding, and trajectory checks over recorded traces. A deterministic replay implementation supports repeatable build-time tests; a judge interface provides an extension point for optional semantic evaluation. On 55 synthetic cases spanning seven enterprise domains, the full pipeline correctly classifies all 30 injected faults and 25 passing controls, including ten non-trivial controls, and misses none of 12 designated critical cases. Each single-layer configuration correctly classifies 35 cases because the corpus deliberately isolates ten faults per layer. A grounded answer obtained through a disallowed sequence illustrates the additional coverage of trajectory checks. Repeat
    
[^5]: PatchCon：面向漏洞检测的补丁感知加权监督对比学习

    PatchCon: Patch-aware Weighted Supervised Contrastive Learning for Vulnerability Detection

    [https://arxiv.org/abs/2611.06583](https://arxiv.org/abs/2611.06583)

    提出PatchCon，一种基于CodeBERT的补丁感知加权监督对比学习模型，通过将漏洞函数与修复后函数对作为高权重困难负样本进行联合优化，显著提升了函数级漏洞检测的精度。

    

    自动化漏洞检测对于保障现代软件系统的安全至关重要，然而现有的基于学习的方法通常将函数视为独立样本，未能充分利用漏洞修复补丁所提供的监督信息。在本文中，我们提出了PatchCon，这是一种基于CodeBERT的补丁感知对比学习模型，用于函数级漏洞检测。PatchCon联合优化漏洞分类与监督对比学习，并将“漏洞函数-修复后函数”对视为具有更大对比权重的困难负样本，以捕获安全修复所引入的细微安全相关差异。为了评估补丁级别的判别能力，我们采用了两种成对指标，衡量模型能否正确区分漏洞函数与其修复后的对应版本。我们在PairVul-Linux数据集上通过大量实验对PatchCon进行了评估，包括与最先进方法的对比、消融研究以及超参数实验。

    arXiv:2611.06583v1 Announce Type: new  Abstract: Automated vulnerability detection is critical for securing modern software systems, yet existing learning-based approaches often treat functions as independent samples and underuse the supervision provided by vulnerability-fixing patches. In this paper, we propose PatchCon, a CodeBERT-based patch-aware contrastive learning model for function-level vulnerability detection. PatchCon jointly optimizes vulnerability classification and supervised contrastive learning, and treats vulnerable-patched function pairs as hard negatives with larger contrastive weights to capture subtle security-relevant differences introduced by fixes. To evaluate patch-level discrimination, we use two pairwise metrics that measure whether a model can correctly distinguish vulnerable functions from their patched counterparts. We evaluate PatchCon on PairVul-Linux through extensive experiments, including state-of-the-art comparisons, ablation studies, hyperparameter 
    
[^6]: EPISTEME：通过工具调用意图构建可信赖的编码智能体

    EPISTEME: Trustworthy Coding Agents Through Tool Call Intent

    [https://arxiv.org/abs/2611.06563](https://arxiv.org/abs/2611.06563)

    提出认识论智能体设计，强制模型声明每次工具调用的意图并与实际执行效果核对，形成可事前把关、事后审计的日志，使两开源模型49次奖励劫持作弊中90%被暴露。

    

    编码智能体通过工具调用来修改软件。工具调用会改变本地状态——是变好还是变坏？无人知晓。智能体从不说明为什么要进行某次工具调用。我们提出了认识论智能体设计，在这种设计中，运行框架保证模型必须声明每次工具调用的原因。随后，运行框架记录命令实际执行的效果，并将模型自报的声明与这些实际效果进行核对。其结果是一份日志，它 1）可以在调用执行前对其进行把关，2）支持事后审计与取证分析。在评估方面，我们使用 ImpossibleBench（其任务只有通过作弊才能完成），并证明强制智能体说明调用原因会使作弊行为暴露出来。在两个开源权重模型共 49 次“奖励劫持”运行中，该日志暴露了其中 90% 的作弊行为。

    arXiv:2611.06563v1 Announce Type: new  Abstract: Coding agents change software through tool calls. Tool calls mutate the local state. For good? For bad? Nobody knows. Agents never say why a tool call is made. We propose the epistemic agent design, in which the harness guarantees that the model declares why a tool call is made. The harness then records what the command actually did and checks the self-reported declaration against these effects. The result is a journal 1) that can guard calls before they run and 2) that supports audits and forensics. For evaluation, we use ImpossibleBench, whose tasks can be passed only by cheating, and demonstrate that forcing the agents to say why results in exposed cheating. Across 49 reward-hacking runs of two open-weight models, the journal exposed 90%.
    
[^7]: 嵌入式领域特定语言生成程序的构建契约：一种测量协议与一个条件效应

    Construction Contracts for Generated Embedded Domain-Specific Language Programs: A Measurement Protocol and a Conditional Effect

    [https://arxiv.org/abs/2611.06525](https://arxiv.org/abs/2611.06525)

    该论文提出区分“程序可解析”与“匹配参考构建”这两种不同事件的测量协议，并发现一种简短的“构建契约卡片”对OpenQASM 3量子程序生成的效果具有条件性：生成器经过卡片训练时完全匹配率提升32个百分点，未经训练时反而下降13个百分点。

    

    一个被生成的领域特定程序即使被其宿主接受，仍可能未能匹配参考构建。我们将这两个事件分开测量，并测试一种简短的构建契约。研究载体为开放量子汇编语言（OpenQASM）3。一个30亿参数的生成器在七种条件下分别运行于100个量子近似优化算法（QAOA）实例和100个变分量子本征求解器（VQE）实例。100个QAOA签名中有80个出现在某些训练分割中。研究采用精确McNemar检验和95%配对自助法置信区间。该契约是一张卡片，列出结构性决策、进入提示词中，并且无需模拟器即可检查。解析率为0.55至0.97，完全结构匹配率为0.02至0.87。在QAOA上，在固定权重下，该卡片在经过卡片训练的权重上使完全匹配率提高32.0个百分点（区间+22至+42），而在没有该训练时则降低13.0个百分点（-23至-3）。这+45.0个百分点的差异并非部署对比。

    arXiv:2611.06525v1 Announce Type: new  Abstract: A generated domain-specific program can be accepted by its host and still miss the reference construction. We measure the two events separately and test a short construction contract. The vehicle is Open Quantum Assembly Language (OpenQASM) 3. One 3-billion-parameter generator is run under seven conditions on 100 quantum approximate optimization (QAOA) and 100 variational quantum eigensolver (VQE) instances. Eighty of 100 QAOA signatures appear in some training split. Exact McNemar tests and 95% paired bootstrap intervals are used. The contract, a card, lists structural decisions, enters the prompt, and is checked without a simulator. Parse rates are 0.55 to 0.97 and full structural match is 0.02 to 0.87. On QAOA, at fixed weights, the card moves full match by +32.0 percentage points on card-trained weights (interval +22 to +42) and by -13.0 (-23 to -3) without that training. The +45.0-point difference is not a deployment contrast. The s
    
[^8]: Scythe：用于快速、高效程序约简的语义感知重写

    Scythe: Semantic-Aware Rewriting for Fast,Effective Program Reduction

    [https://arxiv.org/abs/2611.06515](https://arxiv.org/abs/2611.06515)

    SCYTHE 是一种语义感知的预处理步骤，通过将程序编码为带类型的声明依赖图并解耦声明与其使用位置，使任何基于文法的程序约简器都能更快速、更有效地删除完整的声明。

    

    自动化程序约简（program reduction）将程序缩减为一个更小的、且保留感兴趣属性的程序。与语言无关的基于文法的约简器易于跨语言移植使用，但由于它们仅进行语法层面的推理，无法执行删除完整声明所需的耦合编辑——例如，一个方法在被调用时无法被直接删除。因此，这类约简器会在注定无法成功的编辑上浪费预言机调用，并且经常留下不相关的声明。我们提出了 SCYTHE，这是一种快速、语义感知的预处理步骤，可以运行在任何基于文法的约简器之前，且无需修改后者。SCYTHE 将程序编码为一个带类型的声明依赖图，并应用一组语义感知的算子，将声明与其使用位置解耦，使这些声明在保持程序类型正确的前提下变得可删除。随后，细粒度的词法级最小化工作则交由下游的约简器完成。在涵盖 C、Java 和 Solidity 的 73 个基准测试上（摘要原文在此处截断）

    arXiv:2611.06515v1 Announce Type: new  Abstract: Automated program reduction shrinks a program into a smaller one that preserves a property of interest. Language-agnostic grammar-based reducers are easy to retarget across languages, but because they reason only about syntax, they cannot perform the coupled edits needed to remove whole declarations, e.g., a method cannot be deleted while it is still called. As a result, they waste oracle calls on edits that cannot succeed and often leave irrelevant declarations behind. We present SCYTHE, a fast, semantic-aware pre-processing step that runs in front of any grammar-based reducer without modifying it. SCYTHE encodes the program as a typed declaration dependency graph and applies a set of semantic-aware operators that decouple declarations from their use sites, making them removable while keeping the program well-typed. Fine-grained token-level minimization is then left to the downstream reducer. On 73 benchmarks across C, Java, and Solidit
    
[^9]: HELIX：具有灵活且细粒度分区的异构大语言模型推理

    HELIX: HEterogeneous LLM Inference with FleXible and Fine-Grained Partitioning

    [https://arxiv.org/abs/2611.06513](https://arxiv.org/abs/2611.06513)

    HELIX是一个基于vLLM的异构LLM推理软件栈，能够在算子粒度上自动发现、编译并调度工作负载感知的分区，以最小化异构加速器上的执行延迟。

    

    大语言模型（LLM）推理本质上是异构的，其性能瓶颈会在推理的各个阶段乃至单个算子层面发生转移。这促使人们开发异构服务系统，将执行过程分散部署到专用加速器上，通常采用固定分区方式，例如预填充/解码（prefill/decode）或注意力/前馈网络（attention/FFN）。然而，这些方法预先确定分区策略，限制了其适应不同工作负载和模型架构的能力，并迫使新兴加速器只能承担预定义的角色。我们提出了HELIX，一个基于vLLM的软件栈，它能够在算子粒度上发现、编译并调度工作负载感知的分区，部署于异构加速器之间。在引擎初始化时，HELIX会自动确定针对特定硬件的算子放置方案以最小化执行延迟，同时综合考虑剖析得到的执行成本、通信开销、设备能力以及内存约束。它

    arXiv:2611.06513v1 Announce Type: new  Abstract: LLM inference is inherently heterogeneous, with performance bottlenecks shifting throughout inference phases down to individual operators. This has motivated heterogeneous serving systems that disaggregate execution across specialized accelerators, typically through fixed partitions such as prefill/decode or attention/FFN. However, these approaches commit to a partitioning strategy in advance, limiting their ability to adapt to different workloads and model architectures while forcing emerging accelerators into predefined roles. We present HELIX, a vLLM-based software stack that discovers, compiles and dispatches workload-aware partitions across heterogeneous accelerators at the operator granularity. At engine initialization, HELIX automatically determines hardware-specific operator placements that minimize execution latency, accounting for profiled execution costs, communication overheads, device capabilities, and memory constraints. It
    
[^10]: Trace2Eval：基于真实世界轨迹的编码智能体情境化评估

    Trace2Eval: Contextualized Evaluation of Coding Agents from Real-World Traces

    [https://arxiv.org/abs/2611.06506](https://arxiv.org/abs/2611.06506)

    提出了Trace2Eval，一个可动态更新的流水线，通过多智能体重建系统从真实世界的交互轨迹中恢复用户查询、工作区状态和执行环境，从而对编码智能体在真实用户工作流程中的产出质量进行情境化、持续更新的评估。

    

    arXiv:2611.06506v1 公告类型：新论文 摘要：编码智能体在跨领域的真实工作流程中的应用日益广泛，这促使评估工作应立足于真实的用户任务。然而，现有基准主要依赖于带有参考补丁或隐藏测试的静态公开任务。因此，下一代编码智能体基准应超越对固定任务正确性的评估，转向在真实用户工作流程中对整体产出质量的持续更新评估。我们提出了Trace2Eval，一个可动态更新的流水线，用于从真实世界的交互轨迹中构建基准。Trace2Eval采用多智能体重建系统来分析筛选后的轨迹，恢复原始用户查询，将工作区文件还原至交互前的状态，并重建与轨迹等效的执行环境。随后，候选智能体执行重建的查询以生成初始产出。为了评估这些产出，该流水线迭代地细化人工……

    arXiv:2611.06506v1 Announce Type: new  Abstract: Coding agents are increasingly used in real-world workflows across diverse domains, motivating evaluation grounded in real user tasks. Existing benchmarks, however, largely rely on static public tasks with reference patches or hidden tests. Next-generation coding-agent benchmarks should therefore move beyond correctness on fixed tasks toward continually renewed evaluation of overall artifact quality in real user workflows.   We propose Trace2Eval, a dynamically updatable pipeline that constructs benchmarks from real-world interaction trajectories. Trace2Eval employs a multi-agent reconstruction system to analyze the filtered trajectory, recover the original user query, restore the workspace files to their pre-interaction state, and rebuild a trajectory-equivalent execution environment. A candidate agent then executes the reconstructed query to produce an initial artifact.   To evaluate these artifacts, the pipeline iteratively refines hu
    
[^11]: 超越违规评分：用于驾驶与机器人导航智能体闭环评估的失败边界剖面

    Beyond Infraction Scores: Failure-Boundary Profiles for Closed-Loop Evaluation of Driving and Robot Navigation Agents

    [https://arxiv.org/abs/2611.06453](https://arxiv.org/abs/2611.06453)

    该论文提出“失败边界剖面”评估框架，通过逐级变化单一可测量因素（如走廊宽度、对车间隔）并认证每个变体应为“通过”或“停止”，进而估计智能体的承诺曲线与能力曲线及其临界点，弥补了CARLA传统违规评分无法区分“该停就停”与“强行通过”两类智能体的根本缺陷。

    

    CARLA中的闭环驾驶基准通过驾驶评分对智能体进行排名，该评分由路线完成度乘以违规惩罚得出。这一评分无法区分在障碍物前恰当停下的智能体与强行挤过去的智能体。例如，在一条被两辆停放汽车封闭的走廊场景中，一个发生接触强行通过的智能体在被阻挡的宽度上平均得分为52.2，而选择停下的智能体平均得分仅在37.8到40.6之间。我们提出了“失败边界剖面”这一评估方法：每个场景族仅变化一个可测量的因素，例如走廊宽度或对向来车的时间间隔，并且每个变体都通过测量被认证为智能体“应该通过”或“应该停止”的情形。沿此轴，我们为每个智能体估计两条曲线：承诺度（尝试通过的概率）与能力度（无接触通过的概率），并定位各自跨越二分之一（0.5）的临界点，其置信区间考虑了……（摘要原文在此处截断）

    arXiv:2611.06453v1 Announce Type: cross  Abstract: Closed-loop driving benchmarks in CARLA rank agents by a driving score that multiplies route completion by an infraction penalty. This score does not distinguish an agent that stops in front of an obstacle from one that pushes past it. In a corridor closed by two parked cars, for example, one agent that drives through with contact averages 52.2 on the blocked widths, while the agents that stop average between 37.8 and 40.6. We propose failure-boundary profiles, an evaluation in which each scenario family varies one measured factor, such as corridor width or the time gap in oncoming traffic, and each variant is certified from measurement as one where the agent should proceed or one where it should stop. Along this axis we estimate two curves per agent, commitment (the probability of attempting the passage) and competence (the probability of passing without contact), and locate where each crosses one half, with intervals that account for
    
[^12]: 重新评估PrimeVul数据集上漏洞检测的数据集质量与微调实践

    Reassessing Dataset Quality and Fine-Tuning Practices for Vulnerability Detection on PrimeVul dataset

    [https://arxiv.org/abs/2611.06373](https://arxiv.org/abs/2611.06373)

    该论文通过对770次训练的大规模实证评估发现，在PrimeVul数据集上微调的代码语言模型虽然在F1和ROC-AUC等标准指标上看似有改进，但在安全关键指标上（MCC接近零、概率校准差、严格假阳性约束下假阴性率高）实际漏洞检测能力远低于表面评估结果。

    

    近期的研究报告显示，在PrimeVul等基准数据集上对预训练代码语言模型进行有监督微调以实现自动化漏洞检测取得了可喜的成果。本文对这一场景下的模型性能和数据集质量进行了大规模的实证评估。我们系统地微调了三种广泛使用的编码器模型（CodeBERT、GraphCodeBERT和UniXCoder），涵盖五种类别不平衡缓解策略、广泛的超参数搜索以及多个随机种子，共进行了770次独立训练。我们的结果揭示了一个一致的模式：虽然F1和ROC-AUC等标准指标偶尔显示出改进，但安全关键型和校准感知型指标却呈现出截然不同的景象。马修斯相关系数（MCC）始终接近于零，预测概率校准效果不佳，且在严格假阳性约束下的评估显示出很高的假阴性率。无论是类别不平衡（原文摘要在此处截断）。

    arXiv:2611.06373v1 Announce Type: new  Abstract: Recent studies report promising results from supervised fine-tuning of pre-trained code language models for automated vulnerability detection on benchmarks such as PrimeVul. In this paper, we present a large-scale empirical evaluation of both model performance and dataset quality in this setting.   We systematically fine-tune three widely used encoder models (CodeBERT, GraphCodeBERT, and UniXCoder) across five imbalance mitigation strategies, extensive hyperparameter sweeps, and multiple random seeds, resulting in 770 independent training runs.   Our results reveal a consistent pattern. While standard metrics such as F1 and ROC-AUC occasionally suggest improvements, security-critical and calibration-aware measures tell a different story. Matthews Correlation Coefficient remains near zero, predicted probabilities are poorly calibrated, and evaluation under strict false-positive constraints reveals high false-negative rates. Neither imbala
    
[^13]: 让模糊测试器持续值守：修复持续开源模糊测试中的构建故障

    Keeping Fuzzers on Duty: Repairing Fuzzing Build Breakages in Continuous Open-Source Fuzzing

    [https://arxiv.org/abs/2611.06358](https://arxiv.org/abs/2611.06358)

    提出 FuzzMedic，一个通过跨项目与基础设施构件的证据引导诊断、并在修复状态图上探索修复方案，从而自动修复 OSS-Fuzz 等持续开源模糊测试平台构建故障的自动化智能体。

    

    诸如 OSS-Fuzz 等持续模糊测试平台为开源项目提供持续的安全测试，但长期的构建故障会降低安全覆盖率并浪费基础设施资源。在 OSS-Fuzz 中，约 20% 的已集成项目仍处于构建失败状态，导致其模糊测试器“离岗”停用。这些故障暴露出平台维护者与项目维护者之间的维护鸿沟：前者缺乏项目特定的背景信息，后者往往缺乏模糊测试与基础设施方面的专业知识。由于模糊测试构建故障涉及不断演进的项目与基础设施构件，且需要反复进行基础设施感知的诊断，现有的程序修复技术无法直接应用。我们提出了 FuzzMedic，一个用于模糊测试构建修复的自动化智能体。FuzzMedic 在项目构件与模糊测试基础设施构件之间执行证据引导的诊断，并在一个记录不断演进的工作区与构建反馈的修复状态图（RSG）上探索修复方案。

    arXiv:2611.06358v1 Announce Type: new  Abstract: Continuous fuzzing platforms such as OSS-Fuzz provide sustained security testing for open-source projects, but long-term build breakages reduce security coverage and waste infrastructure resources. In OSS-Fuzz, around 20% of integrated projects remain in build-failure states, leaving their fuzzers off duty. These failures expose a maintenance gap between platform maintainers, who lack project-specific context, and project maintainers, who often lack fuzzing and infrastructure expertise. Existing program-repair techniques cannot be directly applied because fuzz build breakages span evolving project and infrastructure artifacts and require repeated infrastructure-aware diagnosis. We present FuzzMedic, an automated agent for fuzz build repair. FuzzMedic performs evidence-guided diagnosis across project and fuzzing-infrastructure artifacts. It explores repairs over a repair-state graph (RSG) that records the evolving workspace and build feed
    
[^14]: EvoCodeAgent：基于架构-代码协同演化与经验引导修复的项目级代码生成

    EvoCodeAgent: Project-Level Code Generation via Architecture-Code Co-Evolution and Experience-Guided Repair

    [https://arxiv.org/abs/2611.06328](https://arxiv.org/abs/2611.06328)

    提出EvoCodeAgent，通过架构-代码协同演化和经验引导修复两大机制实现项目级代码生成，解决了静态架构无法适应迭代生成、跨迭代经验无法积累复用、以及独立LLM评估不可靠等关键问题。

    

    项目级代码生成旨在从高层次需求自动合成完整的软件仓库，生成具有结构良好模块和一致跨文件依赖关系的连贯多文件系统。尽管大型语言模型（LLM）在规范理解和代码合成方面具有强大能力，但现有方法仍存在几个关键局限性。首先，许多方法依赖静态架构设计，无法在迭代生成过程中进行调整，导致架构与实现之间的偏差日益扩大。其次，它们缺乏在迭代之间积累和复用经验的机制，导致错误反复出现且效率降低。第三，它们通常依赖独立的基于LLM的中间产物评估，这种方式在长上下文推理下透明度有限且可能不可靠。为解决这些挑战，我们提出了E……

    arXiv:2611.06328v1 Announce Type: new  Abstract: Project-level code generation aims to automatically synthesize complete software repositories from high-level requirements, producing coherent multi-file systems with well-structured modules and consistent cross-file dependencies. Despite the strong capabilities of large language models (LLMs) in specification understanding and code synthesis, existing approaches still exhibit several critical limitations. First, many methods rely on static architectural designs that cannot adapt during iterative generation, leading to a growing misalignment between architecture and implementation. Second, they lack mechanisms to accumulate and reuse experience across iterations, resulting in recurring errors and reduced efficiency. Third, they often depend on standalone LLM-based evaluation of intermediate artifacts, which suffer from limited transparency and potential unreliability under long-context reasoning. To address these challenges, we propose E
    
[^15]: A.S.E 2.0：理解仓库上下文如何影响LLM生成代码的安全性

    A.S.E 2.0: Understanding How Repository Context Impacts the Security of LLM-Generated Code

    [https://arxiv.org/abs/2611.06294](https://arxiv.org/abs/2611.06294)

    该论文提出CASE-Bench基准测试，基于真实CVE构建145个仓库级任务，首次系统研究了仓库上下文构建方式如何影响LLM生成代码的安全性。

    

    大型语言模型（LLM）编码助手和智能体在生成代码之前会构建仓库上下文。对于安全敏感的代码，这一上下文决定了模型是能看到项目的安全防护措施（例如授权检查），还是会看到可供模仿的不安全代码。然而，现有的上下文构建几乎完全是为功能正确性设计的，而安全代码基准测试又将上下文视为固定不变，因此仓库上下文如何影响生成代码的安全性仍不明确。我们构建了CASE-Bench，这是一个包含来自51个项目中真实CVE的145个仓库级任务的基准测试。每个任务都在Docker中运行，配有功能测试和经过双向验证的概念验证漏洞利用，并标注了仓库中实施或破坏其安全性的代码。借助CASE-Bench，我们将13种上下文构建方法和六种对照方式与六个LLM进行了比较，对单个上下文元素进行编辑，并分析了编码智能体的运行轨迹。我们……

    arXiv:2611.06294v1 Announce Type: cross  Abstract: Large language model (LLM) coding assistants and agents construct a repository context before generating code. For security-sensitive code, this context decides whether the model sees the project's safeguards (e.g., authorization checks) or insecure code to imitate. However, context construction is designed almost exclusively for functional correctness, and secure-code benchmarks treat the context as fixed, so how repository context affects the security of generated code remains unclear. We construct CASE-Bench, a benchmark of 145 repository-level tasks from real CVEs in 51 projects. Each task runs in Docker, has functional tests and a proof-of-concept exploit validated in both directions, and is annotated with the repository code that enforces or undermines its security. Using CASE-Bench, we compare 13 context construction methods and six controls with six LLMs, edit single context elements, and analyze coding-agent trajectories. We f
    
[^16]: CLEAR：用于可控长时程机器学习研究工程的结构化规范

    CLEAR: Structured Specifications for Controllable Long-Horizon ML Research Engineering

    [https://arxiv.org/abs/2611.06263](https://arxiv.org/abs/2611.06263)

    提出了CLEAR结构化规范方法，将隐式定义的研究成果转化为显式、结构化的规格说明，从而实现可控的长时程机器学习研究工程自动化。

    

    大语言模型（LLM）智能体正被越来越多地用于自动化机器学习（ML）研究工作流，涵盖从代码开发、实验执行到研究复现与科学发现等多个环节。近期的系统表明，LLM 能够在有限人工干预下完成机器学习研究工程中的相当大一部分工作。这些进展意味着一种转变：从自动化单个研究任务，走向自动化将研究成果转化为可执行机器学习系统的长时程工作流。然而，机器学习研究工程仍然比孤立的代码生成困难得多。一项研究成果通常只是通过自然语言描述、公式、伪代码、实现细节、数据集和评估协议的混合，以隐式的方式规定一个系统。智能体必须首先推断出预期的计算结构，将其翻译为可执行的组件，并协调这些组件（摘要在此处截断）。

    arXiv:2611.06263v1 Announce Type: new  Abstract: Large language model (LLM) agents are increasingly being used to automate machine learning (ML) research workflows, ranging from code development and experiment execution to research reproduction and scientific discovery. Recent systems demonstrate that LLMs can perform substantial portions of ML research engineering with limited human intervention. These advances suggest a shift from automating individual research tasks toward automating long-horizon workflows that transform research artifacts into executable ML systems. However, ML research engineering remains substantially more difficult than isolated code generation. A research artifact typically specifies a system only implicitly through a mixture of natural-language descriptions, equations, pseudo-code, implementation details, datasets, and evaluation protocols. An agent must first infer the intended computational structure, translate it into executable components, coordinate their
    
[^17]: 本地开源权重大语言模型智能体在漏洞复现任务上的分析

    Analysis of Local Open-Weight LLM Agents on Vulnerability-Reproduction Tasks

    [https://arxiv.org/abs/2611.06257](https://arxiv.org/abs/2611.06257)

    本研究系统评测了十二种本地开源权重LLM配置在CyberGym漏洞复现任务上的表现，发现即使小至4B的模型也能解决大部分较简单的漏洞复现任务，表明漏洞复现是本地小模型可胜任的自动化分析任务。

    

    能装入单张消费级GPU的开源权重语言模型在越来越多的任务上展现出能力，而其重要性在于可以在本地运行、无需将代码发送到托管服务。漏洞复现——即把漏洞描述转化为能使含漏洞构建版本崩溃的输入——是这类模型的天然目标，因为它是一项更接近测试生成而非代码理解的自动化分析任务，并且针对的是必须执行而非阅读的构建版本。为了评估当前模型能否完成该任务，我们在十个CyberGym漏洞复现任务上评测了十二种本地模型配置，涵盖三代模型、两种架构和三种量化级别，每个任务在统一测试框架（harness）下运行三次，并发布了完整的实验工件。结果表明，所有配置——甚至小至4B的模型——都能解决六个较简单任务中的大多数，而各配置之间仅在剩余的较难任务上才显示出差异。

    arXiv:2611.06257v1 Announce Type: cross  Abstract: Open-weight language models that fit on a single consumer GPU are becoming capable on a growing range of tasks, and they matter because they can run locally without sending the code to a hosted service. Vulnerability reproduction, turning a vulnerability description into an input that crashes the vulnerable build, is a natural target for them, since it is an automated analysis task closer to test generation than to code comprehension and it runs against a build that must be executed rather than read. In order to assess whether current models can perform it, we evaluate twelve configurations of local models, spanning three generations, two architectures and three quantization levels, on ten CyberGym vulnerability-reproduction tasks, three runs each under one harness, and release the full artefact. We show that every configuration, down to the 4B model, solves most of the six easier tasks, and configurations separate only on the remainin
    
[^18]: Bug2Scenario：面向自动驾驶软件的保持根因的测试场景扩增

    Bug2Scenario: Root-Cause-Preserving Test Scenario Amplification for Autonomous Driving Software

    [https://arxiv.org/abs/2611.06229](https://arxiv.org/abs/2611.06229)

    Bug2Scenario是一个根因保持式的后模糊测试场景扩增框架，通过从故障场景中提取根因模式并利用大语言模型生成语义变体，将单次自动驾驶系统故障转化为可复用的回归测试场景。

    

    基于仿真的模糊测试通过反复变异和执行场景来暴露自动驾驶系统（ADS）的故障，但将一次故障转化为可复用的回归场景仍然十分困难。这类变体可以测试故障机制在软件变更后是否依然存在，例如在改变天气或参与者位置的同时保留碰撞事件中参与者与路线的冲突关系。我们提出了Bug2Scenario，一个以根因为导向的后模糊测试（post-fuzzing）框架。该框架从场景数据、轨迹和判定证据中提取根因模式；由大语言模型（LLM）针对参与者、任务、路线和天气提出语义变体；再由确定性规则构建与DriveFuzz兼容的场景；最后通过执行证据和人工审查来评估根因是否被保持。将这些职责相互分离，可以防止流畅性、语法正确性或重复出现的标签被误当作因果证据。在相同随机种子、六种LLM骨干模型、候选预算和执行管线的条件下，Bug2Scenario提升了……（原文摘要在此处截断）

    arXiv:2611.06229v1 Announce Type: new  Abstract: Simulation-based fuzzing exposes autonomous driving system (ADS) failures by repeatedly mutating and executing scenarios, but converting one failure into reusable regression scenarios remains difficult. Such variants test whether the mechanism persists after software changes, for example by varying weather or actor placement while retaining a collision's actor-route conflict. We present Bug2Scenario, a root-cause-guided post-fuzzing framework. From scenario data, traces, and oracle evidence, it extracts a root-cause pattern. An LLM proposes semantic variations over actors, missions, routes, and weather. Deterministic rules construct DriveFuzz-compatible scenarios. Execution evidence and manual review assess preservation. Separating these responsibilities prevents fluency, syntax, or recurring labels from masquerading as causal evidence. Under matched seeds, six LLM backbones, candidate budget, and execution pipeline, Bug2Scenario improve
    
[^19]: CoSafe：面向混合交通自动驾驶的协同式运行时安全保障

    CoSafe: Cooperative Runtime Safety Assurance for Mixed-Traffic Autonomous Driving

    [https://arxiv.org/abs/2611.06204](https://arxiv.org/abs/2611.06204)

    提出CoSafe——一个模块化、与自动驾驶系统无关的协同式运行时安全保障框架，通过多车协作解决混合交通中由局部合理但联合不兼容决策引发的安全违规问题。

    

    随着自动驾驶系统（ADS）的快速发展，现实道路正在演变为混合交通环境，其中自动驾驶汽车（AV）与其他交通参与者（如人类驾驶车辆（HDV）、骑行者和行人）共存并相互作用。然而，当前的自动驾驶系统在混合交通场景中仍会出现各种安全违规行为，这引发了对能够持续监测危险并防止潜在违规的运行时安全保障框架的强烈需求。现有的以自我为中心的保障方法不足以应对混合交通场景，因为在这类场景中，违规行为往往源于多辆车之间局部合理但联合不兼容的决策，且无法通过独立接管来可靠解决。为弥补这一空白，我们提出了CoSafe，一个模块化且与自动驾驶系统无关的协同式运行时安全保障框架，用于混合交通自动驾驶，以将运行时安全扩展为……

    arXiv:2611.06204v1 Announce Type: cross  Abstract: With the rapid development of autonomous driving systems (ADSes), real-world roads are evolving into mixed-traffic environments where autonomous vehicles (AVs) and other traffic participants (e.g., human-driven vehicles (HDVs), cyclists, and pedestrians) coexist and interact with each other. However, current ADSes still incur various safety violations in mixed traffic scenarios, raising a strong demand for runtime safety assurance frameworks that can continuously monitor hazards and prevent potential violations. Existing ego-centric assurance approaches are insufficient for mixed traffic, where violations often arise from locally reasonable yet jointly incompatible decisions among multiple vehicles and cannot be reliably resolved through independent takeover.   To bridge this gap, we propose CoSafe, a modular and ADS-agnostic cooperative runtime safety assurance framework for mixed-traffic autonomous driving to extend runtime safety as
    
[^20]: SBVUL：基于语句的漏洞检测

    SBVUL: Statement-Based Vulnerability Detection

    [https://arxiv.org/abs/2611.06135](https://arxiv.org/abs/2611.06135)

    SBVul提出了一种两阶段智能体方法，通过LLM智能体探索函数与代码仓库来识别候选语句，并结合控制流和数据流分析，直接定位C/C++函数中可利用的漏洞语句，弥补了函数级漏洞检测无法精确定位漏洞位置的不足。

    

    许多近期的漏洞检测方法使用大语言模型（LLM）将函数分类为易受攻击或良性，而将定位可利用语句的工作留给开发者来完成。此外，尽管这些方法在性能上有所提升，它们在区分真实世界中易受攻击的函数与已修复函数方面仍然存在不足。本文提出了SBVul，一种用于检测C/C++函数中可利用语句的智能体化方法。在第一阶段，一个LLM智能体探索该函数及其所在的代码仓库，并识别出候选语句。在第二阶段，第二个智能体分析每个候选语句的控制流和数据流，并对其进行确认、排除或标记为不确定。我们为PrimeVul Paired测试集中的200对函数创建了经人工验证的可利用语句真值标注，并使用开放权重模型Qwen 3.8和DeepSeek V4.1 Flash以及Anthropic的Sonnet 5.5和Opus 5.5运行了SBVul。第一阶段识别出真值……

    arXiv:2611.06135v1 Announce Type: new  Abstract: Many recent approaches for vulnerability detection use large language models (LLMs) to classify functions as vulnerable or benign, leaving developers to locate the exploitable statements. Furthermore, while these approaches have shown performance improvements, they still fall short in distinguishing vulnerable from patched real-world functions. This paper presents SBVul, an agentic approach that detects exploitable statements in C/C++ functions. In Phase~1, an LLM agent explores the function and its repository and identifies candidate statements. In Phase~2, a second agent analyzes the control and data flow of each candidate and confirms it, rules it out, or marks it uncertain. We created a manually verified ground truth of exploitable statements for 200 pairs of the PrimeVul Paired test set and ran SBVul with the open-weight models Qwen 3.8 and DeepSeek V4.1 Flash and Anthropic's Sonnet 5.5 and Opus 5.5. Phase~1 identifies ground-truth 
    
[^21]: 培育自己的AI工具：实践生长型工具的范式

    Growing One's AI Tools: A Paradigm of Practice-Grown Tools

    [https://arxiv.org/abs/2611.06132](https://arxiv.org/abs/2611.06132)

    本文提出“实践生长型工具”新范式：一位无编程背景的会议口译员依据真实口译工作需求自行培育出计算机辅助口译工具，展现了工具与实践通过故障与修复共同演化的过程，并主张应将工具纳入对从业者的评估体系。

    

    长期以来，专业人士使用的都是他人为其构建的软件。本文报告了由用户本人培育的专业计算机辅助口译（CAI）工具的成长历程，该用户是一位未经编程训练的会议口译员。基于为期八周的版本控制记录以及来自133个记录会话和超过20,000条识别语句的遥测数据，本文追踪了真实的会议工作如何推动该工具经历62次提交和六个版本的发展。记录显示，新功能在其需求出现的当天即被部署，而一次策略调整在短短一个下午内将中位端到端延迟削减了超过三分之一。文章随后提出了“实践生长型工具”范式，其界定特征包括：从使用到培育的转变、工具与实践通过故障与修复实现的共同演化、有界本地计算，以及由此得出的评估推论，并提出了三个命题，以迈向将工具纳入考量的从业者评估。

    arXiv:2611.06132v1 Announce Type: cross  Abstract: Professionals have long used software built for them by others. This article reports the growth of a professional computer-assisted interpreting (CAI) tool built by its own user, a conference interpreter without programming training. Drawing on an eight-week version-controlled record with telemetry from 133 logged sessions and over 20,000 recognized utterances, the article traces how real conference work drove the tool through 62 commits and six versions. The record shows capabilities deployed on the day their need appeared and one strategy change that cut median end-to-end delay by over a third within one afternoon. The article then proposes the paradigm of practice-grown tools, defined by the shift from using to growing, the co-evolution of tool and practice through breakdown and repair, bounded local computation, and an assessment corollary, and it formulates three propositions toward tool-inclusive assessment of practitioners, from
    
[^22]: SWEOps-Bench：评估用于端到端解决软件引发云事故的智能体基准

    SWEOps-Bench: Evaluating Agents for End-to-End Resolution of Software-Induced Cloud Incidents

    [https://arxiv.org/abs/2611.06099](https://arxiv.org/abs/2611.06099)

    该论文提出了SWEOps-Bench，一个包含55个源自16个Kubernetes operator项目真实历史缺陷的可复现事故任务的交互式基准，用于评估智能体在统一时间预算内端到端完成云事故的检测、定位与修复的能力。

    

    随着自主智能体的不断进步，软件开发与云运维之间的传统边界逐渐模糊，对集成式事故检测、定位与修复能力的需求日益增长。现有基准仅评估孤立的诊断查询或仓库级程序修复，无法测试跨越运维集群和源代码仓库的持续性事故解决能力。我们提出了SWEOps-Bench，这是一个交互式基准，包含55个可复现的事故任务，这些任务源自16个Kubernetes operator项目中已确认的历史缺陷。每个任务保留原始软件仓库，并在受控工作负载触发历史故障之前以健康的集群状态初始化。智能体通过终端界面调查运行时证据，并提交源代码或配置修复，同时测试框架会在共享的时间预算内构建并部署这些更改。评估涵盖分阶段准确率……（原文摘要在此处截断）

    arXiv:2611.06099v1 Announce Type: new  Abstract: As autonomous agents advance, the traditional boundary between software development and cloud operations gradually blurs, increasing the need for integrated incident detection, localization, and remediation. Existing benchmarks evaluate isolated diagnostic queries or repository-level program repair, failing to test continuous incident resolution across operational clusters and source repositories. We introduce SWEOps-Bench, an interactive benchmark comprising 55 reproducible incident tasks derived from confirmed historical defects across 16 Kubernetes operator projects. Each task preserves the original software repository and initializes in a healthy cluster before a controlled workload triggers the historical fault. Agents investigate runtime evidence and submit source or configuration repairs through a terminal interface, while the harness builds and deploys the changes within a shared time budget.   Evaluation covers stage-wise accura
    
[^23]: 代码生成中的推理轨迹溢出：显式推理模型的实证研究

    Reasoning-Trace Overflow in Code Generation: An Empirical Study of Explicit Reasoning Models

    [https://arxiv.org/abs/2611.06095](https://arxiv.org/abs/2611.06095)

    该研究提出“推理轨迹溢出”概念，实证发现显式推理模型在较难的代码生成任务上常常将全部输出预算耗尽于推理而无法产出可用代码，而轻量级的推理时控制可以缓解这一问题。

    

    显式推理可以帮助语言模型生成代码，但它也带来一个实际风险：模型可能将过多的输出预算用于推理，以致于永远无法生成可用的代码。我们在相对较小的开源权重推理模型中研究这一行为，并将其定义为“推理轨迹溢出”。我们在七个模型和六个代码基准测试设置下，测量了固定32k-token输出预算中有多少被用于推理、这种推理是否能提升pass@1，以及轻量级的推理时控制手段能否缓解溢出问题。我们的结果表明，推理的价值具有高度的条件性。在较简单的函数级任务上，溢出很少发生；但在更难的代码生成基准测试和特定模型家族中，溢出变得普遍：在若干“模型-基准”组合上，超过一半的生成耗尽了全部32k-token预算却仍在推理。大多数完全溢出与重复性的推理循环相关。

    arXiv:2611.06095v1 Announce Type: new  Abstract: Explicit reasoning can help language models generate code, but it also introduces a practical risk: the model may spend so much of its output budget reasoning that it never produces usable code. We study this behaviour in relatively small open-weight reasoning models, framing it as reasoning-trace overflow. Across seven models and six code benchmark settings, we measure how much of a fixed 32k-token output budget is spent on reasoning, whether that reasoning improves pass@1, and whether lightweight inference-time controls can mitigate overflow. Our results show that the value of reasoning is highly conditional. Overflow is rare on simpler function-level tasks, but becomes common on harder code-generation benchmarks and for particular model families: on several model--benchmark pairs, more than half of generations exhaust the full 32k-token budget while still reasoning. Most complete overflow is associated with repetitive reasoning loops,
    
[^24]: SynMOE：一种语法感知的MOE引导方法，用于基于PEFT的代码智能学习

    SynMOE: A Syntax-Aware MOE-Guided Approach for PEFT-Based Code Intelligence Learning

    [https://arxiv.org/abs/2611.06069](https://arxiv.org/abs/2611.06069)

    提出了SynMOE，一种语法感知的MOE引导参数高效微调方法，通过从抽象语法树中提取句法标签并将其融入混合专家适配过程，弥补了现有PEFT方法缺乏代码信息显式建模的不足，从而提升代码智能任务的表现。

    

    参数高效微调（PEFT）已成为在下游任务中适配语言模型（LM）的重要方法。然而，在代码智能领域，大多数PEFT方法的性能相比全量微调（FT）出现了相对下降，这阻碍了其更广泛的应用。其中一个主要原因是，当前的PEFT方法通常基于参数更新的低秩性或稀疏性假设，缺乏对代码相关信息的显式建模。为了解决这一局限性，我们提出了SynMOE，一种用于代码智能学习的语法感知MOE引导的PEFT方法。具体而言，SynMOE包含两个阶段：句法信息提取和语法感知的MOE引导适配。它首先从抽象语法树（AST）中获取源代码的句法标签，然后将这些标签与代码语言模型的输入词元进行对齐。在第二阶段，句法标签被集成到（原文摘要在此处截断）

    arXiv:2611.06069v1 Announce Type: new  Abstract: Parameter-Efficient Fine-Tuning (PEFT) has emerged as an important approach for adapting Language Models (LMs) in downstream tasks. However, in the domain of code intelligence, the performance of most PEFT methods relatively drops compared to Full fine-Tuning (FT), hindering their broader applicability. One of the main reasons is that current PEFT methods are generally based on assumptions about the low-rank nature or sparsity of parameter updates, lacking explicit modeling of code-related information. To tackle this limitation, we propose SynMOE, a syntax-aware MOE-guided approach for PEFT on code intelligence learning. Specifically, SynMOE contains two stages: syntactic information extraction and syntax-aware MOE-guided adaptation. It first obtains the syntactic labels of source code from Abstract Syntax Trees (ASTs), and then aligns the labels with the input tokens of code LMs. In the second stage, syntactic labels are integrated into
    
[^25]: 跨设备运行何时才足以判定 PASS？部分可观测性下的义务级证据资格认定

    When Does a Cross-Device Run Justify PASS? Obligation-Level Evidence Qualification under Partial Observability

    [https://arxiv.org/abs/2611.06013](https://arxiv.org/abs/2611.06013)

    该论文提出一种可执行的证据契约，将证据支持资格认定作为判定求值前的显式阶段，通过把义务绑定到可采纳的主体、情节、对象和观测范围并记录未解决的支持，有效防止跨设备测试中因采用其他设备或尝试的证据而产生的假 PASS 判定。

    

    跨设备测试可能完成每一条命令，却依据来自另一设备或另一次尝试的证据来判定业务声明。我们引入一种可执行的证据契约，将支持资格认定作为谓词求值之前的显式阶段。该契约将每项义务绑定到可采纳的主体、情节、对象和观测范围，并记录未获解决的支持情况，而不是把表面上成功的观测直接当作证明。我们评估了该契约中已实现的两部分。在一项前瞻性设定、12个种子的受控研究中，针对2,304条声明隔离了标识符复用下的关联问题。在给定相同记录和业务策略的情况下，Atomic-Facts Flat 方法产生160个假 PASS，PASS 召回率为76.30%；而提供计算得到的关联后，假 PASS 降至5个，PASS 召回率提升至99.09%。Deterministic Closure 方法则实现零假 PASS 和100.00%的 PASS 召回率，在11个种子上优于 Atomic Flat，其中一次持平（摘要此处被截断）。

    arXiv:2611.06013v1 Announce Type: new  Abstract: A cross-device test can complete every command yet judge a business claim from evidence belonging to another device or attempt. We introduce an executable evidence contract that makes support qualification an explicit stage before predicate evaluation. It binds each obligation to admissible subjects, episodes, objects and observation scopes, and records unresolved support rather than treating an apparently successful observation as proof.   We evaluate two implemented parts of the contract. A prospectively specified, 12-seed controlled study isolates association under reused identifiers across 2,304 claims. Given identical records and business policy, Atomic-Facts Flat produces 160 false PASS and 76.30% PASS recall; supplying computed associations reduces false PASS to five and increases PASS recall to 99.09%. Deterministic Closure produces zero false PASS and 100.00% PASS recall, improving on Atomic Flat in 11 seeds with one tie (two-si
    
[^26]: 检索超越规模：基于多重社会技术上下文的高效日志级别预测

    Retrieval Supersedes Scale: Efficient Log Level Prediction via Multiplex Socio-Technical Context

    [https://arxiv.org/abs/2611.05901](https://arxiv.org/abs/2611.05901)

    该论文提出基于多重社会技术上下文的检索方法来选择上下文学习示例，从而在不微调的情况下显著提升 LLM 的日志级别预测效果，证明了检索策略优于单纯依赖模型规模。

    

    开发者在源代码中插入日志语句以记录程序执行过程，由此产生的日志对于检测和排查系统异常至关重要。这些语句的关键组成部分是日志级别（如 DEBUG、INFO、WARN、ERROR），它决定了消息是否被记录。选择日志级别是一种权衡：级别设置过低会掩盖生产环境中的重要信息，而级别设置过高则会产生冗余日志并带来额外的 I/O 和存储开销。近期研究采用大语言模型（LLM）结合上下文学习（ICL）来预测日志级别，但通常使用随机采样的示例来扩充提示词。我们认为这种做法忽视了日志编写的社会技术属性，即日志约定会因组件和开发团队的不同而存在差异。据报道，使用随机示例的现成 LLM 仅能达到 0.56 到 0.76 的中位 AUC，这表明无需微调仍有改进空间。在本文中，我们提出了一种检索方法……（原文摘要在此处截断）

    arXiv:2611.05901v1 Announce Type: cross  Abstract: Developers insert logging statements in source code to record its execution, and the resulting logs are essential for detecting and troubleshooting system anomalies. A key part of these statements is the log level (e.g., DEBUG, INFO, WARN, ERROR), which determines whether a message is recorded. Choosing it is a trade-off: levels set too low mask important information in production, while levels set too high create noisy logs and extra I/O and storage overhead. Recent work uses Large Language Models (LLMs) with In-Context Learning (ICL) to predict log levels, but typically augments the prompt with randomly sampled examples. We argue that this overlooks the socio-technical nature of logging, where conventions vary by component and development team. Off-the-shelf LLMs using random examples are reported to achieve median AUCs of only 0.56 to 0.76, leaving room for improvement without fine-tuning. In this paper, we propose a retrieval appro
    
[^27]: LLM智能体能学会测试吗？在移动GUI测试中将范围化经验维护为测试证据

    Can LLM Agents Learn to Test? Maintaining Scoped Experience as Testing Evidence for Mobile GUI Testing

    [https://arxiv.org/abs/2611.05873](https://arxiv.org/abs/2611.05873)

    该论文为基于LLM的移动GUI测试提出了一种经验维护契约，赋予可复用测试经验明确的适用范围、来源追溯及确认—重新验证—撤销的生命周期，使测试器能在复用历史经验的同时排除与当前观察相矛盾的过时条目。

    

    重复执行的移动GUI测试可以复用先前执行中的路径、定位器和恢复策略。然而，在某个工作流或应用版本中有效的路径，在另一个环境中可能不相关甚至产生误导。我们研究了基于LLM的测试器如何在保留有用经验的同时，排除与当前观察相矛盾的条目。我们为AdaptiTest提出了一种维护契约，其中可复用经验具有明确的适用范围、可追溯的来源，以及确认、重新验证和撤销的生命周期。该契约将持久化存储保留给Reflector角色，并将召回的经验视为建议而非指令。需要说明的是，存档的实现并未建立对所有指定机制的端到端强制执行；该设计改变的是检索到的上下文，而非模型权重。评估采用只读的Android工作流和存档的内部比较。一个包含六个任务、三次重复的就绪子集报告了18/18的完成运行，对比了有记忆与（无记忆的情况）……

    arXiv:2611.05873v1 Announce Type: new  Abstract: Repeated mobile GUI tests can reuse routes, locators, and recovery strategies from earlier executions. However, a route that worked in one workflow or app version may be irrelevant or misleading in another. We study how an LLM-based tester can retain useful experience while withholding entries contradicted by current observations. We present a maintenance contract for AdaptiTest in which reusable experience has an explicit applicability scope, trace provenance, and a lifecycle for confirmation, revalidation, and withdrawal. The contract reserves persistence for a Reflector role and treats recalled experience as advice. The archived implementation does not establish end-to-end enforcement of every specified mechanism. The design changes retrieved context, not model weights. The evaluation uses read-only Android workflows and archived internal comparisons. A six-task, three-repeat ready subset reports 18/18 completed runs with memory versu
    
[^28]: 一种用于自动化安全模式恢复的混合方法

    A Hybrid Approach for Automated Security Pattern Recovery

    [https://arxiv.org/abs/2611.05823](https://arxiv.org/abs/2611.05823)

    本文提出SecPatRec，一种将ReAct风格LLM智能体与静态代码属性图分析相结合的混合方法，实现了从源代码仓库中自动恢复安全模式的角色检测、代码落地与交互序列识别。

    

    安全设计强调将安全融入软件设计与开发过程中。安全模式目录通过为反复出现的安全问题规定可复用的架构解决方案，将这一原则付诸实践。然而，从源代码中恢复安全模式的实现仍然具有挑战性，因为安全模式是抽象的、跨越多个组件，并且允许多种有效的实现方式。我们提出了SecPatRec，这是一种从源代码仓库中自动恢复安全模式的混合方法。我们将安全模式视为一种参考架构，并将其恢复任务构建为一个演绎任务：确定哪些规定角色存在、将这些角色落地到具体代码实体中，并恢复它们的交互序列。SecPatRec将一个配备语义和词汇代码搜索能力的ReAct风格LLM智能体（用于角色检测与落地）与静态代码属性图分析（用于交互恢复）相结合。我们构建了一个由人工标注的（原文在此处截断）

    arXiv:2611.05823v1 Announce Type: cross  Abstract: Security-by-design emphasises integrating security into software design and development. Security pattern catalogues operationalise this principle by prescribing reusable architectural solutions to recurring security problems. However, recovering their implementations from source code remains challenging because security patterns are abstract, span multiple components, and admit many valid implementations. We present SecPatRec, a hybrid approach for automated security pattern recovery from source code repositories. We treat a security pattern as a reference architecture and frame its recovery as a deductive task: determining which prescribed roles are present, grounding them in code entities, and recovering their interaction sequences. SecPatRec pairs a ReAct-style LLM agent equipped with semantic and lexical code search for role detection and grounding with static code property graph analysis for interaction recovery. We curate a huma
    
[^29]: DARE：面向修复增强的难度分析

    DARE: Difficulty Analysis for Repair Enhancement

    [https://arxiv.org/abs/2611.05797](https://arxiv.org/abs/2611.05797)

    提出DARE框架，利用从基准测试产物中提取的151个静态特征预测编程修复任务的难度（定义为多个智能体的观察失败率），并结合七个特征组的贡献分析，为编程智能体提供任务特定的修复支持。

    

    基于大语言模型（LLM）的编程智能体主要通过解决率来进行评估。然而，这一指标既无法揭示已解决任务的难度，也无法考虑不同基准测试之间任务难度的差异。通过重复运行多个智能体来确定单个任务的难度成本高昂，而仅凭预测的难度分数对智能体需要何种支持提供的指导也十分有限。为了应对这些挑战，我们提出了DARE（Difficulty Analysis for Repair Enhancement，面向修复增强的难度分析），该框架集成了修复任务难度测量、预测、特征组贡献分析以及任务特定的修复支持。我们将连续的任务难度定义为跨多个编程智能体观察到的失败率。DARE难度模型利用从基准测试产物中提取的151个静态特征来预测这一难度。该框架还分析了七个修复任务特征组的贡献，以识别特征……

    arXiv:2611.05797v1 Announce Type: cross  Abstract: LLM-based coding agents are primarily evaluated by their solve rates. However, this metric does not reveal how difficult the solved tasks are or account for differences intask difficulty across benchmarks. Determining the difficulty of individual tasks by repeatedly running multiple agents is costly, while a predicted difficulty score alone provides limited guidance on what support an agent needs. To address these challenges, we propose DARE (Difficulty Analysis for Repair Enhancement), which integrates repair-task difficulty measurement, prediction, feature-group contribution analysis, and task-specific repair support. We define continuous task difficulty as the observed failure rate across multiple coding agents. The DARE difficulty model predicts this difficulty using 151 static features extracted from benchmark artifacts. The framework also analyzes the contributions of seven Repair-Task Feature Groups to identify the characteristi
    
[^30]: 《现代嵌入模型究竟能泄露多少信息？针对商业嵌入模型的反演攻击》

    How Much Do Modern Embeddings Reveal? Inversion Attacks on Commercial Embedding Models

    [https://arxiv.org/abs/2611.05716](https://arxiv.org/abs/2611.05716)

    本研究将vec2text反演攻击框架适配到OpenAI text-embedding-3-small和mistral-embed等商业嵌入模型，发现这些商业模型的嵌入向量可被高精度还原为原始文本（如text-embedding-3-small在健康类指令上精确匹配率达47.4%），揭示了仅存储嵌入向量的系统存在严重的隐私泄露风险。

    

    文本嵌入反演技术可以从固定维度的向量中重建出原始文本，这对那些只存储嵌入向量而非原始文本的系统构成了隐私风险。本文将vec2text框架（Morris等人，2023）进行了适配，使其在原有支持GTR-base的基础上，还能支持OpenAI的text-embedding-3-small和Mistral的mistral-embed模型。研究者在通用指令、维基百科段落、改写指令、健康相关指令以及短字面代码片段这五个类别上各评估了800个样本，统一采用60-80字符的文本长度范围，并配对使用原始指令与改写指令。在所评估的vec2text配置下，text-embedding-3-small在全部五个类别中都取得了最高的精确匹配率，其中在评估所采用的大小写不敏感、去除空白的匹配规则下，健康相关指令的精确匹配率达到47.4%，通用指令达到24.5%。此外，所有散文类文本类别和所有嵌入模型的BERTScore F1在0-100量表上均超过87，表明重建文本与原文具有高度的上下文相似性。

    arXiv:2611.05716v1 Announce Type: new  Abstract: Text embedding inversion reconstructs source text from fixed-dimensional vectors, posing a privacy risk to systems that store embeddings instead of raw text. We adapt the vec2text framework (Morris et al., 2023) to support text-embedding-3-small and mistral-embed alongside GTR-base. We evaluate 800 samples per category across general instructions, Wikipedia passages, paraphrased instructions, health-related instructions, and short literal code snippets, using a common 60-80-character range and paired original and paraphrased instructions. Under the evaluated vec2text configuration, text-embedding-3-small achieves the highest exact-match rate in all five categories, including 47.4% on health-related instructions and 24.5% on general instructions under the evaluation's case-insensitive, whitespace-stripped matching rule. BERTScore F1 exceeds 87 on the 0-100 scale for every prose category and embedder, indicating high contextual similarity 
    
[^31]: TACT：一种利用AI委员会进行定性分析的实用方法

    TACT: A Pragmatic Approach to Qualitative Analysis with an AI Council

    [https://arxiv.org/abs/2611.05702](https://arxiv.org/abs/2611.05702)

    该论文提出了TACT方法，通过由多个AI模型组成的可追溯委员会进行大规模主题分析，使软件团队在利用AI进行定性数据分析的同时，仍能保持对解释和判断的掌控权，并通过对抗性批评和不确定性标注进一步保证分析质量。

    

    定性数据能够告诉软件团队他们的利益相关者需要什么，然而严格分析成千上万条定性回复需要耗费专家时间，很少有团队能够负担或证明其合理性。因此，团队常常面临一个虚假的两难选择：要么不分析这些回复，要么将它们交给大语言模型并接受其解释和判断。在本文中，我们采用一种实用的AI辅助定性分析方法，提出了TACT（可追溯AI委员会主题分析法），这是一种能够在保持团队掌控解释和判断权的同时实现大规模定性分析的程序。TACT有两种变体：TACT-Lite，即核心AI委员会程序；TACT-Full，它在核心程序基础上增加了对抗者（来自其他模型家族、对每个模型的输出进行批评的模型）以及一个用于识别高不确定性编码分配的决策模型。我们在一个由他人进行定性编码的包含256个Stack Overflow线程的独立数据集上评估了TACT。

    arXiv:2611.05702v1 Announce Type: cross  Abstract: Qualitative data can tell software teams what their stakeholders need, yet rigorously analyzing thousands of qualitative responses takes expert time that few teams can afford or justify. Teams therefore often face a false choice: leave the responses unanalyzed, or hand them to a large language model and accept its interpretation and judgment. In this paper, we take a pragmatic approach to AI-assisted qualitative analysis and present TACT (Traceable AI Council for Thematic analysis), a procedure that enables such analysis at scale while keeping teams in control of interpretation and judgment. TACT comes in two variants: TACT-Lite, the core AI council procedure, and TACT-Full, which adds adversaries (models from other model families that critique each model's output) and a decision model that surfaces high-uncertainty code assignments. We evaluated TACT on an independent dataset of 256 Stack Overflow threads qualitatively coded by other 
    
[^32]: 基于大语言模型（LLM）的智能体编程工具之编程基准测试的构念效度研究

    Construct Validity of Coding Benchmarks for LLM-based Agentic Coding Tools

    [https://arxiv.org/abs/2611.05665](https://arxiv.org/abs/2611.05665)

    本文通过分析九个常用的智能体编程基准测试的构念效度，发现这些基准测试往往仅用少量指标来操作化“编程能力”等宽泛概念，并存在干扰测量的混淆因素，从而揭示了现有基准测试在效度上的不足。

    

    arXiv:2611.05665v1 公告类型：new 摘要：软件开发人员越来越多地采用基于大语言模型（LLM）的智能体编程工具来协助他们完成开发任务。为了向开发人员提供有关这些工具有效性的最新信息，先前的工作引入了可用于跨工具比较的基准测试。然而，尽管此类基准测试被广泛采用，最近的研究表明它们存在威胁其效度的问题。研究目标：在本工作中，我们旨在通过分析智能体编程基准测试的构念效度来补充先前的研究。研究方法：我们分析了九个常用于评估智能体编程工具有效性的基准测试，识别了这些基准测试声称测量的构念、用于操作化这些构念的指标，以及干扰测量的潜在混淆因素。研究结果：我们的发现表明，这些基准测试通常仅用一小组指标来操作化诸如“编程能力”等宽泛的概念。所报告的

    arXiv:2611.05665v1 Announce Type: new  Abstract: Software developers increasingly adopt LLM-based agentic coding tools to assist them with development tasks. To provide developers with up-to-date information on these tools' effectiveness, prior work introduced benchmarks that enable comparisons across tools. Despite the widespread adoption of such benchmarks, however, recent work has shown that they suffer from issues threatening their validity. Aims. In this work, we aim to complement prior work by analyzing the construct validity of agentic coding benchmarks. Methodology. We analyze nine benchmarks commonly used to evaluate the effectiveness of agentic coding tools. We identify the constructs that the benchmarks claim to measure, the metrics used to operationalize them, and potential confounders that interfere with the measurement. Results. Our findings show that benchmarks commonly operationalize broad concepts such as coding capability with only a small set of metrics. The reported
    
[^33]: AQUA——一个模块化的认证质量建模框架

    AQUA - A Modular Authentication Quality Modeling Framework

    [https://arxiv.org/abs/2611.05660](https://arxiv.org/abs/2611.05660)

    本文提出AQUA框架，通过将认证分解为四个相互关联的概念，实现对认证解决方案质量的模块化建模，并能将质量影响追溯至其具体来源。

    

    认证解决方案是实现认证安全控制的关键设计、架构与实现决策，例如基于密码的认证解决方案。尽管研究人员经常研究认证质量，但现有方法无法区分对质量的影响究竟源自认证的哪个方面。例如，多因素认证解决方案由于组合了多个认证器而往往可用性较低，而通行密钥之所以比密码更安全，则是源于认证器本身的固有属性。本文提出了AQUA，一个面向认证解决方案的模块化质量建模框架。AQUA的核心思想是将认证分解为四个相互关联的概念，从而实现认证质量的模块化规范定义，并能将质量影响追溯至其来源。我们通过初步的定义性质量模型对AQUA进行了演示，这些模型涵盖了各个层级。

    arXiv:2611.05660v1 Announce Type: cross  Abstract: Authentication solutions are the key design, architectural, and implementation decisions that implement authentication security controls. An example is a password-based authentication solution. While researchers often examine authentication quality, existing approaches do not distinguish which effects on quality stem from which aspect of authentication. For example, multi-factor authentication solutions tend to be less usable because they combine multiple authenticators, whereas passkeys are more secure than passwords because of properties inherent to the authenticator itself. This paper presents AQUA, a modular quality modeling framework for authentication solutions. AQUA's core idea is to decompose authentication into four interrelated concepts, enabling modular specification of authentication quality and tracing quality effects back to their source. We demonstrate AQUA with initial definitional quality models that address the levels
    
[^34]: ViBench 2.0：一个面向长周期软件构建的基准测试

    ViBench 2.0: A Benchmark for Long-Horizon Software Building

    [https://arxiv.org/abs/2611.05620](https://arxiv.org/abs/2611.05620)

    ViBench 2.0 是一个用于长周期软件构建的基准测试，通过 17 个随 241 个功能需求逐步演进的全栈 Web 产品和 544 个浏览器智能体测试计划进行严格评估，揭示了最强模型通过高达 98% 的测试计划却仅有最多 65% 的构建能交付完全可用产品的可靠性差距。

    

    在 Replit——我们的智能体工程平台上，用户如今无需编写代码，只需向智能体连续提出功能需求，就能构建完整的产品。然而，现有的编程基准测试评估的是单一的、完全明确指定的任务，且许多基准已接近饱和。我们提出了 ViBench 2.0，一个用于长周期软件构建的基准测试，我们用它来为平台选择模型。它包含 17 个以广泛使用的软件为原型的全栈 Web 产品，通过 241 个功能需求逐步演进，并且只有在完成时才像用户那样被评判：由一个浏览器智能体运行 544 个测试计划，只有所有计划都通过，构建才算通过，其评分协议经过验证以确保这一严格分数的可靠性。通过对九个模型的评估，我们发现了一个可靠性差距：最强的三个模型最多能通过 98% 的测试计划，但在构建中最多只有 65% 能交付完全可用的产品，而没有任何开放权重模型能做到这一点。这一差距正在快速缩小：在某个模型家族的三个版本中，通过率（原文在此处截断）

    arXiv:2611.05620v1 Announce Type: new  Abstract: On Replit, our agentic engineering platform, users now build entire products without writing code, through successive feature requests to an agent. Yet existing coding benchmarks evaluate single, fully specified tasks, and many are near saturation. We introduce ViBench 2.0, a benchmark for long-horizon software building that we use to select our platform's models. Its 17 full-stack web products, modeled on widely used software, grow over 241 feature requests and are judged only when finished, as a user would judge them: a browser agent runs 544 test plans, and a build passes only if every plan does, under a grading protocol validated to keep this strict score reliable. Evaluating nine models, we find a reliability gap: the three strongest pass up to 98% of test plans yet deliver a fully working product in at most 65% of builds, and no open-weight model delivers one. The gap is closing fast: across three releases of one model family, pass
    
[^35]: RIDGE：面向深度强化学习游戏智能体行为覆盖的状态条件奖励混合方法

    RIDGE: State-Conditioned Reward Blending for Behavioral Coverage in Deep RL Game Agents

    [https://arxiv.org/abs/2611.05601](https://arxiv.org/abs/2611.05601)

    RIDGE 通过状态条件奖励混合器和多头评论家架构，让单个深度强化学习游戏智能体同时掌握多种行为风格，从而规避了 PPO 在非平稳奖励下的性能退化问题。

    

    在固定“人格”奖励上训练的深度强化学习智能体只会以一种方式行事：探索者在饥饿时仍继续探索，幸存者在空地图上囤积资源。由于每种人格都固定了自身的奖励，自动化游戏测试需要为 N 种风格训练 N 个智能体。而在回合中途改变奖励组成则会陷入“切换-稳定性”两难：PPO 在非平稳奖励下会发生性能退化。我们提出 RIDGE，一种通过两个设计选择规避该两难问题的单智能体架构：一是状态条件混合器，从缓慢变化的状态变量计算出经 softmax 归一化的 sigmoid 权重；二是多头评论家，为每个人格在其自身的平稳信号上分别训练一个价值头。在包含四种人格的 Crafter 环境中，单个 RIDGE 智能体同时掌握了木器级别制作/战斗分支的两条路径，而没有任何固定人格基线能够同时解决这两者；锐度消融实验证实状态条件化是性能提升的关键。我们将这些结果定位为混合方法能力的演示。

    arXiv:2611.05601v1 Announce Type: cross  Abstract: A deep RL agent trained on a fixed persona reward plays one way: an Explorer keeps exploring while starving, a Survivor hoards on an empty map. Since each persona fixes its own reward, automated playtesting trains N agents for N styles. Changing reward composition mid-episode runs into the switching-stability dilemma: PPO degrades under non-stationary rewards. We present RIDGE, a single-agent architecture that sidesteps the dilemma with two design choices: a state-conditioned blender that computes softmax-normalized sigmoid weights from slow-moving state variables, and a multi-head critic that trains one value head per persona on its own stationary signal. On Crafter with four personas, one \ridge agent acquires both branches of the wood-tier crafting/combat split that no fixed-persona baseline solves jointly, and a sharpness ablation confirms state-conditioning drives the gain. We position these results as demonstrations of the blendi
    
[^36]: ProofStudio：浏览器中的五种证明助手

    ProofStudio: Five Proof Assistants in the Browser

    [https://arxiv.org/abs/2611.05540](https://arxiv.org/abs/2611.05540)

    ProofStudio是一个基于浏览器的集成环境，将Isabelle/HOL、Lean、Rocq、HOL4和HOL Light五种证明助手统一到同一界面中，无需本地安装即可降低形式化规约与证明的学习门槛并支持自动化实验。

    

    人工智能辅助证明自动化的最新进展正在拓展证明助手在软件验证和数学领域的应用。然而，更高的证明自动化程度并不能免除用户评估形式化陈述是否准确表达其预期需求的责任，这使得形式化规约与推理的教育变得至关重要。为了降低使用成熟证明助手所带来的安装与学习门槛，我们提出了ProofStudio，这是一个基于浏览器的环境，集成了Isabelle/HOL、Lean、Rocq、HOL4和HOL Light五种证明助手，并配有相应的已解答示例、预配置的证明搜索以及情境化的AI辅助。借助ProofStudio，用户无需本地安装即可探索这五种系统中的形式化规约与证明，通过统一界面查看原生存证反馈并试验各种自动化功能。

    arXiv:2611.05540v1 Announce Type: new  Abstract: Recent advances in AI-assisted proof automation are extending the use of proof assistants in software verification and mathematics. However, greater proof automation does not remove users' responsibility to assess whether formal statements capture their intended requirements, making education in formal specification and reasoning essential. To lower the installation and learning barriers associated with mature proof assistants, we present ProofStudio, a browser-based environment integrating Isabelle/HOL, Lean, Rocq, HOL4, and HOL Light with corresponding worked examples, preconfigured proof search, and contextual AI assistance. With ProofStudio, users can explore formal specifications and proofs across five systems without local installation, inspecting native prover feedback and experimenting with automation through a common interface.
    
[^37]: 彭博（Bloomberg）企业级代码迁移中的混合 LLM-确定性流水线

    Hybrid LLM-Deterministic Pipelines for Enterprise-Scale Code Migration at Bloomberg

    [https://arxiv.org/abs/2611.05457](https://arxiv.org/abs/2611.05457)

    彭博提出的混合 LLM-确定性流水线架构，通过将 LLM 限定在预计算上下文内进行有界文件转换并结合生产遥测反馈，在约 8,600 个企业仓库的两大迁移行动中实现了 729 次成功代码迁移。

    

    大型组织中存在着大量基础设施迁移的长尾工作，这些工作值得做但往往不够紧迫，难以与功能开发竞争优先级，因为每项迁移在每个代码仓库上都需要耗费大量的工程时间。本文报告了我们从“Automated Uplift（自动提升）”计划中获得的工业实践经验，这是一项历时数月的计划，在彭博的数千个代码仓库中应用了 LLM 辅助的迁移流水线。我们考察了两个不同的生产环境迁移行动：一个异构的 Python 打包迁移（约 1,900 个候选仓库）和一个统一的 C++ 构建系统迁移（约 6,700 个候选仓库）。这些行动总共生成了 2,333 个拉取请求，最终实现了 729 次成功的迁移。我们详细介绍了我们的混合架构解决方案：该方案最大化确定性处理，将 LLM 的使用限定在预先计算好的上下文中执行有界的文件转换，并引入了运行时生产遥测数据（如执行日志和使用轨迹等）。

    arXiv:2611.05457v1 Announce Type: cross  Abstract: Large organizations carry a long tail of infrastructure migrations that are worth doing but rarely urgent enough to compete with feature work, because each one costs significant engineering time per repository. This paper reports on our industrial experience from Automated Uplift, a multi-month initiative applying LLM-assisted migration pipelines across thousands of Bloomberg repositories. We examine two distinct production campaigns: a heterogeneous Python packaging migration (approx. 1,900 candidate repositories) and a uniform C++ build system migration (approx. 6,700 candidate repositories). Together, these campaigns generated 2,333 pull requests, resulting in 729 successful migrations. We detail our hybrid architectural solution which maximizes deterministic processing, confines the LLM to bounded file transformations within a pre-computed context, and incorporates runtime production telemetry (such as execution logs and usage trac
    
[^38]: 智能体行为工程：一种用于规范、验证与治理企业AI智能体的契约式设计方法论

    Agentic Behavior Engineering: A Design-by-Contract Methodology for Specifying, Verifying, and Governing Enterprise AI Agents

    [https://arxiv.org/abs/2611.05454](https://arxiv.org/abs/2611.05454)

    本文提出智能体行为工程（ABE），一种将可观察的智能体行为而非提示词或模型作为版本控制工程制品的契约式设计方法论，用于解决企业LLM智能体因源自团队外部的行为变更而导致的维护与治理难题。

    

    部署大语言模型（LLM）智能体的企业团队面临着一个传统测试无法解决的维护难题。模型本身具有非确定性，但更大的风险在于，改变智能体行为的大多数因素（如模型版本退役、政策修订、上游数据模式变更、为修复某一案例而进行的提示词编辑）都源自负责团队之外，并且往往在生产环境中作为缺陷被发现，而非在测试阶段作为回归问题被捕获。本文提出了智能体行为工程（ABE），这是一种与平台无关的方法论，它将可观察的智能体行为（而非提示词或模型本身）视为受版本控制的工程制品。ABE将契约式设计方法适配到非确定性系统中：一个仅包含规范的行为契约（带有稳定的规则标识符）明确了智能体必须做什么、不得做什么，以及在证据不足时何时应当放弃作答；一个轻量级的场景库将每条规则转化为具体的场景……

    arXiv:2611.05454v1 Announce Type: new  Abstract: Enterprise teams that deploy large language model (LLM) agents inherit a maintenance problem that traditional testing does not address. The model is non-deterministic, but the larger risk is that most of what changes an agent's behavior (a retired model version, a revised policy, an upstream schema change, a prompt edit meant to fix one case) originates outside the owning team and is discovered as a production defect rather than caught as a regression. This paper presents Agentic Behavior Engineering (ABE), a platform-agnostic methodology that treats observable agent behavior, rather than the prompt or the model, as the version-controlled engineering artifact. ABE adapts Design by Contract to non-deterministic systems. A specification-only behavior contract with stable rule identifiers states what the agent must do, must not do, and when it must abstain for lack of evidence; a lightweight scenario library turns each rule into concrete si
    
[^39]: LIBRA：可执行工具库的程序化归纳

    LIBRA: Programmatic Induction of Executable Tool Libraries

    [https://arxiv.org/abs/2611.05445](https://arxiv.org/abs/2611.05445)

    LIBRA 通过让大语言模型提出工具间契约并在沙盒中进行运行时验证，将扁平的工具目录自动归纳为可执行的工具库，从而在多工具任务中检索出包含先决工具的完整工作流，无需依赖标注或任务特定训练，在多个基准上显著提升了检索性能。

    

    大型语言模型智能体越来越多地从庞大的工具目录中检索工具，然而大多数检索器仅根据请求相似度对每个工具进行独立排序。对于多工具任务，这种方式只能召回与目标匹配的工具，而遗漏了工作流程所需的先决工具。我们提出 LIBRA——一个通过运行时组装进行工具库归纳的框架，它将扁平的工具目录转化为可执行的工具库。大语言模型为每个工具提出契约，包括候选调用、可复用输出与状态变更以及成功条件；LIBRA 随后在沙盒环境中测试这些组合，并保留在运行时得到验证支持的工具间关系。该工具库在用户查询之前构建完成，无需依赖关系标注、工具使用轨迹或任务特定训练，同时支持一次性检索和智能体式检索。在 AppWorld、BFCL 和 SciToolEval 三个基准上，相比相同的稠密检索器，LIBRA 将一次性检索的 Recall 和 NDCG 平均提升了 7.9 和 3.7 个百分点。

    arXiv:2611.05445v1 Announce Type: cross  Abstract: Large language model agents increasingly retrieve tools from large catalogs, yet most retrievers rank each tool independently by request similarity. For multi tool tasks, this recovers the goal matching tool while omitting prerequisite tools required by the workflow.We introduce LIBRA, a framework for Library Induction by Runtime Assembly that turns a flat tool catalog into an executable tool library. An LLM proposes per tool contracts of candidate invocations, reusable outputs and state changes, and success conditions; LIBRA then tests these compositions in a sandbox and retains the relations supported at runtime. Constructed before user queries, the library requires no dependency annotations, tool use trajectories, or task specific training, and supports both one-shot and agentic retrieval. On AppWorld, BFCL, and SciToolEval, LIBRA improves one-shot Recall and NDCG over the same dense retriever by 7.9 and 3.7 percentage points on ave
    
[^40]: Req-Space：面向需求评审的交互式可视化用户研究

    Req-Space: A User Study of Interactive Visualisations for Requirements Review

    [https://arxiv.org/abs/2611.05437](https://arxiv.org/abs/2611.05437)

    本文提出Req-Space交互式原型，通过集成语义探索与可追溯性可视化来辅助需求评审，用户研究表明该工具能帮助评审者发现更多有源文档支持的额外问题并强化决策理由。

    

    需求评审涉及识别分布式规格说明中的不一致性，并评估变更的影响。受工业项目研讨会中提出的实际需求的启发，我们研究了Req-Space——一个集成语义探索与可追溯性可视化的交互式原型，旨在支持以人为主导的评审。我们开展了一项固定顺序的两阶段研究，共有20名参与者，其中11名拥有至少三年软件或IT领域经验。参与者需要评审一份虚构的包含40项需求的软件规格说明以及一个访客结账功能的变更请求，首先仅使用源文档完成评审，随后借助Req-Space重新执行相同任务。在得到工具支持的评审阶段，17名参与者记录了至少一个额外的有源文档支持的条目或需求配对。在69个可评估的额外发现中，有58个（84.1%）被判定确实获得了源材料的支持。有七条决策理由在第二阶段被评定为更加有力，其中包括五条在未发生变更的情况下……（原文在此处截断）

    arXiv:2611.05437v1 Announce Type: new  Abstract: Requirements review involves identifying inconsistencies and assessing change implications across distributed specifications. Motivated by needs raised in industrial project workshops, we investigate Req-Space, an interactive prototype integrating semantic exploration and traceability visualisation to support human-led review. We conducted a fixed-order, two-phase study with 20 participants, including 11 with at least three years of software or IT experience. Participants reviewed a fictional 40-requirement software specification and a guest-checkout change request, first using source documents and then revisiting the same tasks with Req-Space. During the supported review, 17 participants recorded at least one additional source-supported item or pair. Of 69 assessable additional units, 58 (84.1%) were judged supported by the source materials. Seven decision rationales were judged stronger in Phase 2, including five without a change in th
    
[^41]: 基于大语言模型的游戏非玩家角色（NPC）：一项系统性映射研究

    LLM-based NPCs in Video Games: A Systematic Mapping Study

    [https://arxiv.org/abs/2611.05427](https://arxiv.org/abs/2611.05427)

    该论文通过对42篇相关文献的系统性映射研究，首次从组件采用率、延迟、安全覆盖和可复现性四个维度对基于大语言模型的游戏NPC领域进行了量化刻画。

    

    几十年来，非玩家角色（NPC）一直由基于规则的系统驱动，其个性和互动在游戏发布前就被预先编写好。大语言模型（LLM）有望使NPC能够进行开放式对话、适应新情况，并表现出开发者未完全预设的行为，这引发了一波研究和商业实验的浪潮。然而，该领域目前缺乏一个统一的图景，来展示已经构建了什么、效果如何以及代价是什么。我们通过对学术文献中377篇论文的系统性映射研究来弥补这一空白，最终筛选出42篇论文，围绕一个主要研究问题和十个子问题、横跨四个主题维度进行分析。我们首次提供了组件采用率、延迟、安全覆盖和可复现性的量化刻画：短期记忆和世界状态接地已接近普及（60%和69%），而长期记忆（31%）、反思（38%）和检索增强生成（14%）仍然……

    arXiv:2611.05427v1 Announce Type: cross  Abstract: For decades, non-player characters (NPCs) have been driven by rule-based systems whose personalities and interactions are scripted before release. Large language models (LLMs) promise NPCs that converse openly, adapt to new situations, and behave in ways developers did not fully author, prompting a wave of research and commercial experimentation. Yet the field lacks a unified picture of what has been built, how well it works, and at what cost. We address this gap with a systematic mapping study of 377 papers from the academic literature, yielding 42 papers analysed against a main research question and ten sub-questions across four thematic axes. We provide the first quantitative characterisation of component adoption, latency, safety coverage, and reproducibility: short-term memory and world-state grounding are near-universal (60% and 69%), whereas long-term memory (31%), reflection (38%), and retrieval-augmented generation (14%) remai
    
[^42]: AI驱动的SaaS平台集成：AI Express统一生成式AI服务——基于按量付费积分系统与可扩展微服务架构

    AI-Driven SaaS Platform Integration: AI Express for Unified Generative AI Services with Pay-As-You-Go Credit Systems and Scalable Microservice Architecture

    [https://arxiv.org/abs/2611.05389](https://arxiv.org/abs/2611.05389)

    AI Express通过按量付费积分系统和可扩展微服务架构，将碎片化的生成式AI能力（代码审查、文本转网站、语法检查等）整合到单一SaaS平台中，使用户无需多重订阅和技术专长即可低成本使用统一的AI服务。

    

    当前的生成式AI服务呈现碎片化状态，文本生成、图像合成和代码辅助等功能需要分别订阅，导致成本高昂且工作流程频繁中断。AI Express是一个SaaS平台，将这些能力整合为统一的按量付费服务。该平台提供一系列AI工具，包括用于评估GitHub代码仓库的核心代码审查器、可将文本转换为完整功能网站的文本转网站生成器，以及脚本生成、语法检查、数据分析和水印去除等功能，所有工具均可通过单一界面访问，无需任何专业技术知识。

    arXiv:2611.05389v1 Announce Type: cross  Abstract: Generative AI is currently fragmented, with separate subscriptions required for text generation, image synthesis, and code assistance, leading to high costs and workflow disruptions. AI Express is a SaaS platform that consolidates these capabilities into a unified pay-as-you-go service. The platform provides a range of AI tools, including a Core Code Reviewer that evaluates GitHub repositories, a Text-to-Website generator that converts text into fully functional websites, script generation, grammar checking, data analysis, and watermark removal, all accessible through a single interface without requiring technical expertise.
    
[^43]: 为MM-AR元建模平台扩展过程挖掘能力

    Extending the MM-AR Metamodeling Platform with Process Mining Capabilities

    [https://arxiv.org/abs/2611.05176](https://arxiv.org/abs/2611.05176)

    本文首次提出概念验证方案，通过基于PM4Py的独立Python服务，在不修改平台核心的前提下，为XR元建模平台MM-AR扩展了过程发现能力，使从事件日志中挖掘出的Petri网和BPMN模型能够直接在沉浸式2D/3D建模环境中展示。

    

    过程挖掘能够从信息系统记录的事件数据中推导出过程模型，但其结果通常以二维可视化的形式呈现，与企业模型创建和维护的环境相互分离。面向扩展现实（XR）的元建模平台（如MM-AR）允许用户定义建模语言并以二维和三维方式展示模型，但迄今为止尚不支持访问事件数据或挖掘算法。因此，本文提出了首个概念验证方案，展示如何通过一个基于开源PM4Py过程挖掘库的独立Python服务，在不修改平台核心的情况下，为基于XR的元建模平台扩展过程发现能力。我们使用三种常见的发现算法，基于CSV或XES格式的事件日志获取Petri网和BPMN模型，演示了该方法，并利用四个公开事件日志进行了技术评估。

    arXiv:2611.05176v2 Announce Type: replace  Abstract: Process mining derives process models from event data recorded by information systems, but its results are usually presented as two-dimensional visualizations that are separate from the environments in which enterprise models are created and maintained. Metamodeling platforms for extended reality (XR), such as MM-AR, allow users to define modeling languages and to display models in 2D and 3D, but they do not provide access to event data or mining algorithms so far. In this paper we therefore present a first proof-of-concept for showing how an XR-based metamodeling platform can be extended with process discovery capabilities without changes to the platform core through a separate Python service based on the open source PM4Py process mining library. We demonstrate the approach using three common discovery algorithms for obtaining Petri nets and BPMN models based on CSV or XES event logs. A technical evaluation using four public event l
    
[^44]: Trace2World：从真实交互轨迹重建应用世界以实现可扩展的GUI智能体训练

    Trace2World: Reconstructing App Worlds from Real Interaction Traces for Scalable GUI Agent Training

    [https://arxiv.org/abs/2610.30792](https://arxiv.org/abs/2610.30792)

    Trace2World通过将全局行为重建与环境实现分离，从真实交互轨迹构建忠实于真实应用行为的可执行GUI环境用于智能体训练，在十个应用的200个工作流上实现91.1%的端到端复现率，且基于来源的验证使真实设备上的训练增益保留率达94.0%。

    

    扩展GUI强化学习需要能够保留真实应用行为的环境。然而，将行为推断与编码相耦合，可能会把部分轨迹转变成状态转换错误的可运行模拟（mock）。我们提出Trace2World，它将全局行为重建与具体实现分离开来。该方法首先将多条轨迹调和为一个共享的、与来源关联的应用模型，然后编译出可执行的环境。通过独立回放来源工作流来定位不一致之处，从而进行有边界的实现修复。经过验证的世界进而可以在来源支持的行为范围内，支持面向学习者自适应的任务与环境增强。Trace2World从1,570条轨迹中构建了50个环境。在十个应用的200个工作流上，91.1%的工作流被端到端复现，而最强的任务局部匹配基线仅为59.8%。基于来源的验证在真实设备上保留了94.0%的模拟训练增益，而自我验证下仅为54.7%。学习者引导（摘要在此处截断）

    arXiv:2610.30792v1 Announce Type: new  Abstract: Scaling GUI reinforcement learning requires environments that preserve real-app behavior. Yet coupling behavior inference with coding can turn partial traces into runnable mocks with incorrect state transitions. We introduce \textbf{Trace2World}, which separates global behavior reconstruction from implementation. It reconciles traces into a shared, source-linked app model, then compiles an executable environment. Independent replay of source workflows localizes mismatches for bounded implementation repair. Validated worlds then support learner-adaptive task and environment augmentation within source-supported behavior. From 1,570 traces, Trace2World constructs 50 environments. On 200 workflows across ten apps, 91.1% are reproduced end to end, versus 59.8% for the strongest matched task-local baseline. Source-grounded validation retains 94.0% of simulated training gains on real devices, versus 54.7% under self-validation. Learner guidance
    
[^45]: X.js：一个Node.js污点风格漏洞的动态基准测试集

    X.js: A Live Benchmark for Node.js Taint-Style Vulnerabilities

    [https://arxiv.org/abs/2610.18585](https://arxiv.org/abs/2610.18585)

    本文提出了X.js——一个持续更新的真实世界Node.js污点风格漏洞基准测试集，每个漏洞均配有精确的源汇位置、经验证的可执行漏洞利用程序及补丁版本，解决了现有漏洞数据集过时、无法支持该领域研究长期发展的问题。

    

    npm包中的JavaScript漏洞对Node.js生态系统构成重大风险，越来越多的研究工作旨在检测、利用和缓解这些漏洞。对这类工作进行研究和评估需要最新的真实漏洞集合，然而最常用的数据集在2022年就停止记录漏洞了。我们提出了X.js，这是一个真实世界Node.js污点风格漏洞的动态（持续更新的）基准测试集，涵盖代码和命令注入、原型污染以及路径遍历等漏洞类型。X.js基于截至2026年中期收集的1,738份安全公告（主要来自GitHub咨询数据库）构建，将每个漏洞与精确的源（source）和汇（sink）位置、经过验证的可执行漏洞利用程序及其执行轨迹，以及在可用情况下的包修补版本进行配对。X.js支持长期可复现性和持续增长：依赖项被镜像到自托管注册表中，以防止包从生态系统中被移除。

    arXiv:2610.18585v2 Announce Type: replace-cross  Abstract: JavaScript vulnerabilities in npm packages pose a significant risk to the Node$.$js ecosystem, and a growing body of work aims to detect, exploit, and mitigate them. Studying and evaluating this work requires up-to-date collections of real vulnerabilities, yet the most commonly used datasets stop recording vulnerabilities in 2022. We present X$.$js, a live benchmark of real-world Node$.$js taint-style vulnerabilities covering code and command injection, prototype pollution, and path traversal. Built from 1,738 advisories collected through mid-2026, primarily from the GitHub Advisory Database, X$.$js pairs each vulnerability with precise source and sink locations, a validated executable exploit and its execution trace, and, when available, a patched version of the package. X$.$js supports long-term reproducibility and growth: dependencies are mirrored in a self-hosted registry to guard against package removal from the ecosystem,
    
[^46]: 可信RAG：用于检测生成式AI系统中错误信息与知识投毒的评估代理

    Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems

    [https://arxiv.org/abs/2608.21095](https://arxiv.org/abs/2608.21095)

    本文提出了一种结合NLI事实核查与五信号投毒检测的评估代理，并引入信任指数，在TruthfulQA上实现了高准确率与精确率，有效缓解了RAG系统中的知识投毒风险。

    

    检索增强生成（RAG）将大型语言模型（LLM）的输出锚定在外部知识上，但RAG系统通常信任其检索到的任何内容，从而造成“安全-可靠性差距”：高语义相关性并不保证事实真实性。攻击者利用这一漏洞进行知识投毒，插入恶意文档以引发定向错误信息。我们提出了一种评估代理，这是一种中间件，结合了自然语言推理（NLI）事实核查、具有相关性加权聚合的五信号投毒检测器，以及一个信任指数 T = 0.4 F + 0.35 C + 0.25 (1 - P )，并针对高污染情境采用非线性阻尼器。在TruthfulQA上使用Llama 3.3 70B时，该代理达到了91%的准确率和100%的精确率，对指令注入的召回率为100%，而就地编辑（如实体替换）仍难以检测。在三个LLM上，信任指数保持判别性，接收者操作特征（ROC）曲线表现良好。

    arXiv:2608.21095v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) grounds Large Language Model (LLM) outputs in external knowledge, but RAG systems usually trust whatever they retrieve, creating a Security-Reliability Gap: high semantic relevance does not guarantee factual truth. Adversaries exploit this through knowledge poisoning, inserting malicious documents to cause targeted misinformation. We propose an Evaluation Agent, middleware that combines Natural Language Inference (NLI) factual verification, a five-signal poison detector with relevance-weighted aggregation, and a Trust Index T = 0.4 F + 0.35 C + 0.25 (1 - P ) with a non-linear dampener for high-contamination contexts. On TruthfulQA with Llama 3.3 70B, the agent reaches 91% accuracy and 100% precision, with 100% recall on instruction injection, while in-place edits, such as entity swaps, remain hard to detect. Across three LLMs the Trust Index stays discriminative, with a Receiver Operating Characteri
    
[^47]: TraceCAD：面向智能体CAD生成的轨迹引导修复方法

    TraceCAD: Trace-Guided Repair for Agentic CAD Generation

    [https://arxiv.org/abs/2608.03062](https://arxiv.org/abs/2608.03062)

    TraceCAD通过将需求、建模步骤、失败证据与修复结果链接为持久轨迹状态并配合可复用的技能记忆，实现对LLM生成的CAD程序的高效诊断与局部化修复，显著减少几何退化并大幅降低智能体调用开销。

    

    基于大语言模型（LLM）的CAD智能体能够生成可执行的参数化程序，但其纠错循环可能会丢失关于已满足的需求、错误操作以及先前修复的证据。我们提出了TraceCAD，一个将请求的特征、建模步骤、失败证据和候选修复结果链接为持久状态的恢复层。TraceCAD诊断可能出错的建模操作，在其依赖区域内搜索有界编辑方案，通过执行验证和保留性检查来评估候选方案，并将成功与失败的修复结果保存到可复用的技能记忆中。在基于DeepCAD的基准测试上，通过200个模型的消融实验和1000个模型的对比，TraceCAD在IoU、Chamfer距离和Hausdorff距离等指标上取得了具有竞争力的几何质量。消融实验表明，移除持久状态会使恢复分数几乎减半；移除局部化搜索会使几何退化增加一倍以上，并使代码智能体的调用次数翻倍。在不相交的数据上初始化技能库……

    arXiv:2608.03062v2 Announce Type: replace  Abstract: LLM-based CAD agents produce executable parametric programs, but their correction loops may lose evidence about satisfied requirements, faulty operations, and prior repairs. We introduce TraceCAD, a recovery layer that links requested features, modeling steps, failure evidence, and candidate outcomes as persistent state. TraceCAD diagnoses likely faulty operations, searches bounded edits in their dependency regions, validates candidates through execution and preservation checks, and retains successful and failed repair outcomes in reusable skill memory. On DeepCAD-derived benchmarks with 200-model ablations and a 1K-model comparison, TraceCAD achieves competitive geometric quality in terms of IoU, Chamfer distance, and Hausdorff distance. Removing persistent state nearly halves recovery score; removing localized search more than doubles geometric regression and doubles code-agent invocations. Initializing the skill store on disjoint 
    
[^48]: 无边缘，无判定：对78K个真实世界SBOM中声明依赖图的大规模实证研究

    No Edges, No Verdict: A Large-Scale Empirical Study of Declared Dependency Graphs in 78K SBOMs in the Wild

    [https://arxiv.org/abs/2607.22140](https://arxiv.org/abs/2607.22140)

    这项研究首次大规模分析了78,612个真实SBOM中的依赖图，发现超过半数未声明任何边缘，且边缘生成主要由生成器决定而非软件本身，这对漏洞分析有重要影响。

    

    软件物料清单（SBOM）不仅作为组件清单被使用，还作为依赖图：漏洞分类、可达性过滤和影响分析都遍历SBOM声明的边缘。我们对来自Wild SBOMs数据集（77,092个可解析）的78,612个真实世界SBOM文件中的声明依赖图进行了首次大规模特征化分析。我们发现总体分为三个区间：52.9%的SBOM未声明任何边缘（不符合NTIA关于依赖关系的最低要素要求），8.8%声明了依赖块但使大多数组件孤立（退化区间；在此类至少含50个组件的SBOM中，中位孤立比例为93%，我们生成的11个Syft容器镜像SBOM也属于此区间，孤立比例为91.4-98.0%），38.3%形成了良好连接的图。边缘生成由生成器决定，而非所描述的软件（0%-100%无边缘）。

    arXiv:2607.22140v2 Announce Type: replace  Abstract: Software Bills of Materials (SBOMs) are consumed not only as component inventories but as dependency graphs: vulnerability triage, reachability filtering, and impact analysis all traverse the edges an SBOM declares. We present the first large-scale characterization of the declared dependency graph across 78,612 real-world SBOM files from the Wild SBOMs dataset (77,092 parseable). We find that the population splits into three regimes: 52.9% of SBOMs declare no edges at all (failing the NTIA minimum-elements requirement of dependency relationships), 8.8% declare a dependency block yet leave the majority of components isolated (degenerate regime; among such SBOMs with at least 50 components the median orphan share is 93%, and our 11 Syft-generated container-image SBOMs fall in this regime at 91.4-98.0% orphans), and 38.3% form well-connected graphs. Edge emission is determined by the generator, not the described software (0%-100% no-edg
    
[^49]: TTHE：测试时脚手架演化

    TTHE: Test-Time Harness Evolution

    [https://arxiv.org/abs/2607.08124](https://arxiv.org/abs/2607.08124)

    TTHE提出在测试阶段利用无标签执行轨迹持续演化LLM智能体的脚手架程序，通过智能体提议者生成多样化候选并由裁判筛选，从而在无需标签的情况下实现对任务分布与失败模式变化的实时自适应。

    

    LLM智能体的行为不仅由底层模型决定，还由其脚手架决定：脚手架是构建上下文、调用工具、验证中间结果并从失败中恢复的可执行程序。现有方法通常在训练或开发数据上优化脚手架，并在测试时冻结所得到的程序，这限制了对任务分布、失败模式和工具交互变化的适应能力。我们提出这样一个问题：能否在评估过程中利用无标签的执行轨迹来优化脚手架？我们提出了测试时脚手架演化（TTHE），它将可执行脚手架视为测试时自适应的状态。在每个无标签的测试批次上，TTHE通过能够对执行轨迹进行推理并探索多样化编辑方向的智能体提议者，迭代地演化候选脚手架种群，并由裁判基于执行轨迹和源自执行的代理信号来选择候选脚手架。

    arXiv:2607.08124v3 Announce Type: replace-cross  Abstract: The behavior of an LLM agent is determined not only by the underlying model, but also by its harness: the executable program that constructs context, invokes tools, verifies intermediate results, and recovers from failures. Existing approaches typically optimize harnesses on training or development data and freeze the resulting programs at test time, limiting adaptation to shifts in task distributions, failure modes, and tool interactions. We ask whether the harness can instead be optimized during evaluation using unlabeled execution traces. We introduce Test-Time Harness Evolution (TTHE), which treats the executable harness as the state of test-time adaptation. On each unlabeled test batch, TTHE iteratively evolves a population of candidate harnesses through agentic proposers that reason over execution traces and explore diverse edit directions. A judge selects a candidate using execution traces and execution-derived proxy sig
    
[^50]: 量子软件实证比较的审计

    Auditing Empirical Comparisons in Quantum Software

    [https://arxiv.org/abs/2607.00516](https://arxiv.org/abs/2607.00516)

    提出了CLAIMSTAB-QC审计框架，通过在计算结果前锁定审计设计、限定证据来源边界，将量子软件论文中的455条实证比较声明分类为“维持”、“未决”或“未保留”，从而检验比较结论是否有证据支撑。

    

    量子软件实证论文常常声称某个编译器能生成更小的电路、某个优化器更可靠、或某个拟设能获得更好的解质量。这类比较依赖于基准测试范围、电路构建、编译、采样、后端、噪声模型、优化器选择和资源预算等因素。现有的测试、基准测试和可复现性方法有助于评估程序和实验，但并不能直接审计所报告的比较结论是否得到了论文与工件中可用证据的支持。我们提出了 CLAIMSTAB-QC，这是一个以来源为边界的框架，它记录基线、度量指标、关系类型和可采纳的证据；在计算结果之前锁定审计设计；并给出有明确范围的关系性结论或明确的证据边界。在锁定的审计范围内，严格的标量方向性比较会被分类为“维持”、“未决”或“未保留”。我们考察了455条比较性声明……（原文摘要在此处截断）

    arXiv:2607.00516v2 Announce Type: replace  Abstract: Empirical quantum-software papers often claim that one compiler produces smaller circuits, one optimizer is more reliable, or one ansatz achieves better solution quality. Such comparisons depend on benchmark scope, circuit construction, compilation, sampling, backends, noise models, optimizer choices, and resource budgets. Existing testing, benchmarking, and reproducibility methods help assess programs and experiments but do not directly audit whether the reported comparison is supported by the evidence available in the paper and artifact. We present CLAIMSTAB-QC, a source-bounded framework that records baselines, metric, relation, and admissible evidence; locks the audit design before outcomes are computed; and reports a scoped relation outcome or an explicit evidence boundary. Strict scalar-directional comparisons are classified as Sustained, Unresolved, or Not preserved within the locked audit scope. We examine 455 comparative cla
    
[^51]: 语法约束解码可将大语言模型越狱以生成恶意代码

    Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code

    [https://arxiv.org/abs/2606.11817](https://arxiv.org/abs/2606.11817)

    该论文揭示了语法约束解码（GCD）这一可靠性增强技术可被反直觉地利用为越狱攻击面，提出了 CodeSpear 攻击使大语言模型生成恶意代码，并进一步提出 CodeShield 安全对齐方法，使模型在攻击者控制的语法约束下仍能保持安全行为。

    

    大语言模型（LLM）越来越多地被用于代码生成，这引发了人们对其可能被滥用以生成恶意代码的担忧。与此同时，语法约束解码（GCD）已被广泛采用，通过强制语法有效性来提高大语言模型生成代码的可靠性。在本文中，我们揭示了一个反直觉的风险：这种以可靠性为导向的技术本身可能成为攻击面。我们发现了一种新的越狱攻击，称为 CodeSpear，它利用 GCD 诱导大语言模型生成恶意代码。我们的实验表明，仅应用良性的代码语法约束就可以有效地越狱大语言模型。为了解决这一漏洞，我们提出了 CodeShield，这是一种安全对齐方法，即使在攻击者控制的语法约束下也能稳健地保持安全行为。CodeShield 在代码模态中对模型进行对齐，教会它在 GCD 下生成蜜罐代码。这种代码在语义上……（摘要原文在此处截断）

    arXiv:2606.11817v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly used for code generation, raising concerns that they may be misused to produce malicious code. Meanwhile, Grammar-Constrained Decoding (GCD) has been widely adopted to improve the reliability of LLM-generated code by enforcing syntactic validity. In this paper, we reveal a counterintuitive risk: this reliability-oriented technique can itself become an attack surface. We uncover a new jailbreak attack, termed CodeSpear, that exploits GCD to induce LLMs into generating malicious code. Our experiments show that simply applying a benign code grammar constraint can effectively jailbreak LLMs.   To address this vulnerability, we propose CodeShield, a safety alignment approach that robustly preserves safe behavior even under attacker-controlled grammar constraints. CodeShield aligns the model in the code modality by teaching it to generate honeypot code under GCD. Such code is semantically
    
[^52]: Ekka：LLM推理中静默错误的自动化诊断

    Ekka: Automated Diagnosis of Silent Errors in LLM Inference

    [https://arxiv.org/abs/2606.04594](https://arxiv.org/abs/2606.04594)

    Ekka通过差分对比目标与参考LLM服务框架的中间执行状态，将静默错误诊断转化为差分调试问题，实现了自动化的根因定位，在真实错误基准上达到80%的pass@1诊断准确率。

    

    LLM服务框架正快速演进，形成了复杂的软件栈和大量的优化技术。快速的开发过程可能引入静默错误，即输出质量在没有任何显式错误信号的情况下悄然下降。由于高层症状与低层根本原因之间存在巨大的语义鸿沟，诊断这类静默错误是非常困难的。我们观察到，通过利用语义正确的参考实现，静默错误的诊断可以有效地转化为一个差分调试问题。我们提出了Ekka，一个自动化诊断系统，它通过系统地对比和对齐目标框架与参考框架之间的中间执行状态来识别根本原因。我们基于流行的服务框架构建了一个真实世界静默错误基准，Ekka在其中展现出80%的pass@1诊断准确率和88%的pass@5诊断准确率，超越了现有方法。

    arXiv:2606.04594v2 Announce Type: replace-cross  Abstract: LLM serving frameworks are quickly evolving with a complex software stack and a vast number of optimizations. The rapid development process can introduce silent errors where output quality silently degrades without any explicit error signals. Diagnosing silent errors is notoriously difficult due to the substantial semantic gap between the high-level symptoms and the low-level root causes. We observe that diagnosis of silent errors can be effectively framed as a differential debugging problem by leveraging the existence of semantically correct reference implementations. We propose Ekka, an automated diagnosis system that identifies root causes by systematically aligning and comparing intermediate execution states between a target and a reference framework. We constructed a benchmark of real-world silent errors from popular serving frameworks, where Ekka shows 80% pass@1 diagnosis accuracy and 88% pass@5 diagnosis accuracy, outpe
    
[^53]: AI编程助手对软件工程的影响：一项纵向研究

    The Impact of AI Coding Assistants on Software Engineering: A Longitudinal Study

    [https://arxiv.org/abs/2605.23135](https://arxiv.org/abs/2605.23135)

    该纵向研究发现，AI编程助手使开发者将工作重心从代码创造转向验证活动，并首次提出“监督性工程工作”这一新概念，即对AI输出进行指导、评估和纠正的新型工作模式。

    

    AI编程助手近年来变得非常普及。通过一项纵向混合方法调查，我们研究了专业软件工程师如何看待AI编程助手对任务专注度、开发者体验和生产力的影响。我们分两次（间隔六个月）发放了调查问卷，第一次有158名符合条件的参与者，第二次有101名，并形成了一个95人的匹配纵向队列。参与者报告在大多数开发任务上花费的时间减少，其中82%的人报告编写代码的时间有所减少。我们发现工作重心发生了更广泛的转变，从创造活动转向验证活动。我们提出了一类新的工作，称之为“监督性工程工作”，涵盖对AI输出的指导、评估和纠正。这一概念比文献中此前报道的额外审查开销更为广泛，因为它还包括在AI输出生成之前和期间对其进行的引导和塑造。

    arXiv:2605.23135v2 Announce Type: replace  Abstract: AI coding assistants have become prolific in recent years. Through a longitudinal mixed-methods investigation, we examined how professional software engineers perceive the effects of AI coding assistants in regard to task focus, developer experience, and productivity. Two questionnaires were administered six months apart, yielding 158 eligible participants at the first time point, 101 at the second, and a matched longitudinal cohort of 95. Participants reported spending less time on most development tasks, with 82% reporting less on writing code. We found a broader shift in focus from creation to verification activities. We propose a new category of work we term supervisory engineering work, encompassing the direction, evaluation, and correction of AI output. This construct is broader than the extra review overhead that has been previously reported in literature since it encompasses directing and shaping AI output both before and dur
    
[^54]: 非自修复架构技术债的危害及其对修复时间的影响

    The Dangers of Non-Self-Fixed Architecture Technical Debt and Its Impact on Time-to-Fix

    [https://arxiv.org/abs/2605.16133](https://arxiv.org/abs/2605.16133)

    该论文通过对大型Apache开源项目中架构技术债生命周期的实证分析，发现需要由引入者之外的人偿还的非自修复架构技术债往往导致更长的修复时间，从而揭示了其潜在危害。

    

    技术债是指开发者在优先考虑短期交付而非质量改进工作时所产生的长期成本。当架构决策（如技术选型、设计模式或系统分解方式）优先考虑近期进展而牺牲未来的可维护性和可演化性时，便产生了架构技术债。由于架构技术债影响系统的核心结构，并通过架构依赖不断传播，其修复成本往往比局部性的代码级债务更高，破坏性也更大。尽管架构技术债已被广泛研究，但债务偿还中一个重要却未被充分探索的方面是：由谁来偿还。先前的工作在架构技术债偿还责任及其与修复时间的关系方面提供的实证证据有限。我们对自修复架构技术债（即由债务引入者本人偿还债务）进行了实证研究，并在大型Apache开源项目中将其与非自修复架构技术债进行对比。我们通过追踪……（摘要至此处截断）

    arXiv:2605.16133v2 Announce Type: replace  Abstract: Technical Debt (TD) refers to the long-term costs incurred when developers prioritize short-term delivery over quality-improving work. Architectural Technical Debt (ATD) arises when architectural decisions (e.g., technology choices, patterns, or decomposition) prioritize near-term progress over future maintainability and evolvability. Because ATD affects a system's core structure and propagates through architectural dependencies, it is often more expensive and disruptive to remediate than localized code-level debt. Although ATD has been widely studied, an important but underexplored aspect of repayment is who performs it. Prior work provides limited empirical evidence on repayment responsibility in ATD and its relationship to time-to-fix. We empirically study self-fixed ATD, where the introducer also repays the debt, and contrast it with non--self-fixed ATD in large Apache open-source projects. We reconstruct ATD lifecycles by tracin
    
[^55]: 从实验室到现实：深度学习模型与LLMs在漏洞检测中的实用评估

    From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection

    [https://arxiv.org/abs/2512.10485](https://arxiv.org/abs/2512.10485)

    本文系统评估了四种深度学习漏洞检测模型在多个数据集上的表现，通过跨数据集实验和可视化分析揭示了基准测试与真实世界应用之间的性能差距，强调了模型泛化能力的挑战。

    

    基于深度学习（DL）的漏洞检测方法在基准数据集上表现出强大性能，但其在现实世界中的有效性仍未得到充分探索。近期研究表明，基于图神经网络和Transformer的模型，包括大型语言模型（LLMs），在精选的基准数据集上评估时能产生令人满意的结果。这些数据集通常具有相似的数据分布，可能包含合成样本、启发式标签或标签噪声。在本研究中，我们系统评估了四种代表性DL模型——Devign、ReVeal、LineVul和VulBERTa——在四个代表性数据集上：Juliet、Devign、BigVul和ICVul。每个模型在每个数据集上独立训练，并使用t-SNE和质心距离分析这些模型采用的基于图和CodeBERT的表示，以检查与漏洞相关的模式。为了评估现实应用的适用性，我们进一步进行了跨数据集实验和实际场景模拟。

    arXiv:2512.10485v3 Announce Type: replace-cross  Abstract: Vulnerability detection methods based on deep learning (DL) have shown strong performance on benchmark datasets, yet their real-world effectiveness remains underexplored. Recent work suggests that graph neural network-based and transformer-based models, including large language models (LLMs), yield promising results when evaluated on curated benchmark datasets. These datasets are typically characterized by similar data distributions and may contain synthetic samples, heuristic labels, or labeling noise. In this study, we systematically evaluate four representative DL models---Devign, ReVeal, LineVul, and VulBERTa---across four representative datasets: Juliet, Devign, BigVul, and ICVul. Each model is trained independently on each dataset, and the graph-based and CodeBERT representations adopted by these models are analyzed using t-SNE and centroid distance to examine vulnerability-related patterns. To assess realistic applicabil
    
[^56]: 数据集的误用如何损害了语义克隆检测

    How the Misuse of a Dataset Harmed Semantic Clone Detection

    [https://arxiv.org/abs/2505.04311](https://arxiv.org/abs/2505.04311)

    BigCloneBench数据集的WT3/T4子集存在系统性标注错误（约93.9%的克隆对功能并不相似），导致128篇使用该数据集评估语义克隆检测的研究结果的可信度受到严重损害。

    

    BigCloneBench被广泛用于评估传统的代码克隆检测，以及近年来用于评估语义克隆检测和功能相似性的机器学习方法。本文证明BigCloneBench的弱Type-3/Type-4（WT3/T4）子集作为学习或评估功能相似性的真实基准存在严重问题。其tagged-tagged的WT3/T4克隆对将两个被标记为针对相同功能的真阳性方法组合在一起，且不包含示例函数。通过对406对样本的分层概率抽样进行总体加权分析，我们估计在符合条件的tagged-tagged WT3/T4总体中，93.9%的克隆对在我们的操作性定义下功能并不相似，因此属于错误标注（保守95%置信区间：91.5%-96.2%）。通过对179篇论文的文献综述，我们估计有128篇论文使用BigCloneBench评估语义克隆检测的方式会使其结果受到影响。

    arXiv:2505.04311v2 Announce Type: replace  Abstract: BigCloneBench is widely used to evaluate traditional clone detection and, more recently, machine learning approaches to semantic clone detection and functional similarity.   This paper demonstrates that BigCloneBench's Weakly Type-3/Type-4 (WT3/T4) subset is problematic as ground truth for learning or evaluating functional similarity. Its tagged-tagged WT3/T4 clone pairs combine two methods labelled as true positives for the same functionality and do not contain exemplar functions. A population-weighted analysis of a stratified probability sample of 406 such pairs estimated that 93.9% of pairs in the eligible tagged-tagged WT3/T4 population are not functionally similar under our operational definition and are therefore mislabelled (conservative 95% confidence interval: 91.5%-96.2%).   From a literature review of 179 papers, we estimate that 128 used BigCloneBench to evaluate semantic clone detection in a way that their results could 
    
[^57]: 自动化漏洞利用与安全测试生成：一项系统性文献综述

    Automated Exploit and Security Test Generation: A Systematic Literature Review

    [https://arxiv.org/abs/2502.04953](https://arxiv.org/abs/2502.04953)

    本文对过去二十年的71篇论文（67项独立研究）进行了系统性文献综述，建立了自动漏洞利用生成（AEG）与自动安全测试生成（ASTG）领域的统一术语体系，并系统比较了现有方法。

    

    静态分析技术往往不足以判断一个被报告的安全漏洞是否可被利用、是否真的会对软件应用构成实际安全风险。动态分析通过使用精心构造的输入执行目标应用程序，以触发或重现给定漏洞，从而弥补了这一局限。在过去二十年中，自动漏洞利用生成（AEG）和自动安全测试生成（ASTG）已成为网络安全与软件工程领域的活跃研究方向。本文通过建立统一的术语体系，并对过去二十年发表的研究进行系统性文献综述（SLR），厘清了该领域的全貌。从学术文献搜索引擎获取的3,309篇初始文献出发，我们筛选并分析了71篇论文，涵盖67项独立研究。我们系统地比较了现有的AEG与ASTG方法……（原文摘要截断于此）

    arXiv:2502.04953v2 Announce Type: replace-cross  Abstract: Static analysis techniques are often insufficient to determine whether a reported security vulnerability is exploitable and actually poses a real security risk to a software application. Dynamic analysis addresses this limitation by executing the target application with carefully crafted inputs that trigger or reproduce a given vulnerability. Over the past two decades, automatic exploit generation (AEG) and automatic security test generation (ASTG) have emerged as active research areas in cybersecurity and software engineering.   This paper demystifies the field by establishing consistent terminology and presenting a systematic literature review (SLR) of research published in the last two decades. Starting from an initial set of 3,309 publications obtained from academic literature search engines, we selected and analyzed 71 papers comprising 67 distinct studies. We systematically compare existing AEG and ASTG approaches with re
    

