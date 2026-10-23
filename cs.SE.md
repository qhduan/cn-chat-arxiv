# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [VERA: A Responsible AI Dashboard for Assessing LLM Deployments in Light of the EU AI Act](https://arxiv.org/abs/2610.31588) | VERA 是一个开源自托管仪表盘，让非技术审查人员能够便捷地对大语言模型运行合规基准测试，依据欧盟人工智能法案对十二项可衡量的技术要求进行评分，并追踪评估过程的能耗，已实际部署于一家大型欧洲银行。 |
| [^2] | [CAMGen: A Core Change-Aware Multi-View Framework for Commit Message Generation](https://arxiv.org/abs/2610.31552) | CAMGen通过识别代表主要变更意图的核心实体，并借助仓库级语义聚类构建功能语义视图，将多代码实体协作实现的功能上下文融入提交信息生成，从而生成更能准确解释变更及其影响的提交信息。 |
| [^3] | [Adaptive Skill Granularity for Coding Agents: Runtime Routing and Macro-to-Micro Recovery](https://arxiv.org/abs/2610.31461) | 该论文提出自适应技能粒度框架，将技能粒度作为基于任务结构、执行状态和宏技能失败条件的动作抽象策略，通过粒度控制器与有限状态恢复执行器在运行时动态路由宏技能或微级工具调用，并在宏技能失败时实现宏到微的自动恢复，从而提升编码代理的成功率与成本效率。 |
| [^4] | [Procrustes-Bering-Sheaft: Automated Model Discovery and Virtual Crash Tests for Microservices](https://arxiv.org/abs/2610.31444) | 该论文提出了 Procrustes-Bering-Sheaft 工具链，通过蒙特卡洛故障采样自动化实现微服务的模型发现与虚拟崩溃测试，并能检测操作可达性与实际观测成功之间的不匹配。 |
| [^5] | [ConQEC: Concolic Testing for Quantum Error-Correcting Code Implementations](https://arxiv.org/abs/2610.31443) | 本文提出ConQEC，首个面向量子纠错码实现的具体-符号混合测试框架，通过结合具体泡利故障重放与符号化修复路径约束，不仅能暴露最终逻辑错误，还能生成故障-症状-译码器-恢复（FSDR）全链路诊断，定位修复路径最早发生分歧的阶段。 |
| [^6] | [Automating Feature Flag Cleanup at Scale with Multi-Agent LLM Systems: An Industrial Experience at DoorDash](https://arxiv.org/abs/2610.31351) | 本文提出一个基于多智能体LLM的端到端自动化功能开关清理系统，在DoorDash的大规模工业实践中，对评估的50个过期功能开关中90%成功生成了获得工程师批准的删除PR。 |
| [^7] | [Typed Template Fuzzing](https://arxiv.org/abs/2610.31342) | TTF提出了一种轻量级的基于大语言模型的模糊测试方法，通过在预处理阶段利用LLM将种子转化为类型化模板，从而无需特定语言解析器、类型检查器或模糊测试过程中的LLM查询，即可实现语义感知的高效变异式模糊测试。 |
| [^8] | [Superintelligent Software Engineering: Governance That Scales with Capability](https://arxiv.org/abs/2610.31306) | 本文提出“超智能软件工程”概念，主张当AI智能体能力超越人类工程师时，必须让治理机制（门禁、契约、决策与衰减控制）随能力同步扩展，才能应对知识衰减与人工审查失效，保障软件正确性。 |
| [^9] | [Beyond Accuracy: Disentangling Decomposition from Typed-State Quality Assurance in Multi-Agent LLM Pipelines](https://arxiv.org/abs/2610.31280) | 该研究发现多智能体LLM流水线在需求分类与优先级排序任务中的性能提升主要源于任务分解所支持的专门化提示，而非类型化状态质量保障机制。 |
| [^10] | [ProductWebBench: Evaluating Website Continuity Beyond Task Completion](https://arxiv.org/abs/2610.31118) | 该论文提出 ProductWebBench 基准，首次在真实前端仓库上评估“网站连续性”（即代码变更后网站其余部分是否仍正常运行），并发现即使通过所有任务要求检查的变更运行中，仍有 28.2% 会破坏网站的其他功能。 |
| [^11] | [Tracing Hallucinations in LLM Reasoning for UML Class Diagram Generation: An Empirical Taxonomy and Mitigation Study](https://arxiv.org/abs/2610.30946) | 该研究针对LLM生成UML类图过程中的推理幻觉，构建了包含提示忠实度、代码忠实度和推理质量三个维度共十一种子类型的实证分类体系，并发现代码忠实度错误是出现频率最高的幻觉类型。 |
| [^12] | [REGATTA: A Cross-Domain Measurement Framework and Fault-Injection Harness for Enterprise AI Agents](https://arxiv.org/abs/2610.30924) | REGATTA提出了一个企业AI智能体评估测量框架，通过125项带来源引用的监管清单、25个故障语料库及故障注入评分工具，评估测试套件暴露指定故障的能力并审计其引用规则的可信度，审计发现大量条目存在引用缺陷。 |
| [^13] | [When Code Gets Cheap, Evaluation Gets Scarce: An Experience Report on AI-Assisted Agile Software Engineering](https://arxiv.org/abs/2610.30916) | 本文通过一名开发者仅用三天、未手写一行代码便构建并部署合规关键的考试评分系统的实战经验，提出AI对软件工程的核心影响不是加速编程，而是改变稀缺资源——代码变得充裕，而可靠评估和工程判断成为新的约束性瓶颈。 |
| [^14] | [ATLAS: A Multi-Agent Framework for Lifting Binary to Canonical Compiler IR](https://arxiv.org/abs/2610.30877) | ATLAS是一个多智能体框架，通过结构感知的区域级检索与协作式细化循环，将高度优化的无符号二进制程序重建为规范且形式有效的优化前LLVM中间表示。 |
| [^15] | [Separate Field Scoring Protects Unused Tools When Agents Pool Tool-Use Experience](https://arxiv.org/abs/2610.30867) | 提出SYNAPSE方法，将工具使用记录作为独立字段进行评分，在智能体共享工具使用经验提升已使用工具检索效果的同时，有效避免了从未被使用过的工具检索性能的下降。 |
| [^16] | [What Ethereum Developers Ask For, and Whether They Get It: Ten Years of Ethereum Stack Exchange](https://arxiv.org/abs/2610.30829) | 该论文分析了以太坊 Stack Exchange 十年间的全部54,637个问题，利用准确率达87.9%-93.5%的大语言模型主题标注流水线，首次完整刻画了开发者自发求助需求的分布——集中于合约编写、部署与调试——并检验该平台是否仍能有效解决这些问题。 |
| [^17] | [VLTest: Finding Joint Image-Text Failures in Vision-Language Models](https://arxiv.org/abs/2610.30794) | VLTest是一种黑盒视觉-语言模型测试方法，通过分别和联合变异图像与文本输入，并采用四单元格对比和VLM裁判验证，精准发现仅由图像-文本组合触发的模型失效。 |
| [^18] | [Trace2World: Reconstructing App Worlds from Real Interaction Traces for Scalable GUI Agent Training](https://arxiv.org/abs/2610.30792) | Trace2World通过将全局行为重建与环境实现分离，从真实交互轨迹构建忠实于真实应用行为的可执行GUI环境用于智能体训练，在十个应用的200个工作流上实现91.1%的端到端复现率，且基于来源的验证使真实设备上的训练增益保留率达94.0%。 |
| [^19] | [Can Loan Processors and Underwriters Spot AI-Edited Statements? An Empirical Study](https://arxiv.org/abs/2610.30758) | 该研究通过实证实验发现，经验丰富的贷款审核员和处理员在仅凭文件的首次审阅中，仅有约10.6%的判断准确指出了AI伪造财务文件中的编辑痕迹（且全部依赖标识符不一致），表明仅靠人工文件审查很难发现AI编辑的伪造贷款文件。 |
| [^20] | [From Natural Language to Verified C: Verification-Guided Generation with ACSL Specifications](https://arxiv.org/abs/2610.30725) | 该论文提出一种验证引导式生成方法，将C代码实现与ACSL规范作为统一产物进行协同设计，使AI生成的程序能够通过自动形式化验证并获得机器可检查的正确性证据。 |
| [^21] | [From Tokens to Credits to Outcomes: Pricing AI Across Models, Applications, and Cloud Platforms](https://arxiv.org/abs/2610.30701) | 本文对2023至2026年间AI软件的定价模式演变进行了首次系统性可复现研究，构建了涵盖福布斯AI 50、云平台及独特设计的定价原语与积分机制数据集和分类法。 |
| [^22] | [Governance-State Corruption in Long-Term Memory for LLM Agents: A Cross-Model Stress Test](https://arxiv.org/abs/2610.30698) | 该论文定义并量化了一类新故障“治理状态损坏”——即在存储内容不变时仅改变哪些记忆记录进入模型上下文的故障，通过跨六个提供商的2,080对故障实验证明此类损坏可使LLM智能体下游性能下降高达40个百分点，而基于oracle的治理强制执行能实现100%恢复。 |
| [^23] | [Stale Instruction Files Override the Code in Coding Agents: A One-Sentence Fix](https://arxiv.org/abs/2610.30571) | 研究发现编码智能体（如Claude Code和Codex CLI）会盲目遵循过时的项目指令文件而忽视实际代码，甚至在察觉到冲突并指出来时仍然如此，而只需在用户指令文件中加入一句“现有代码优先”即可有效修复这一问题。 |
| [^24] | [Can LLMs Improve Their Own Harness? A Controlled Study on ARC-AGI-3](https://arxiv.org/abs/2610.30499) | 该论文通过受控实验首次系统考察前沿大语言模型能否自主构建并迭代改进自己的执行框架，让模型在ARC-AGI-3交互式基准上独立编写、使用并两次修订完整框架代码库，再于二十个从未参与改进的留出游戏上检验其泛化能力。 |
| [^25] | [From Prose to Properties: Testing Research Claims with ARATest for Agent-Native Science](https://arxiv.org/abs/2610.30495) | 该论文提出 ARATest，一个基于属性的测试库，通过将研究论断抽象为可执行的“属性骨架”并配合类型化声明、共享案例类型和三值判定语义进行自动化检验，使 AI 智能体产生的研究论断变得可测试、可证伪，从而解决验证速度跟不上论断产出速度的问题。 |
| [^26] | [What the Trace Shows and What the Label Says: Auditing Behavioral Detectors in Agent Evaluation](https://arxiv.org/abs/2610.30491) | 该研究通过在GSM8K上构建包含损坏计算器输出的工具使用场景，审计了三个轨迹级行为检测器，发现最终正确性、可见重新验证与错误传播等标签会系统性不一致，并揭示了“静默工具输出覆盖”这一关键失败模式——即模型在不重新调用工具的情况下使用了与损坏输出不同的值。 |
| [^27] | [TrustedCode: Evaluating Large Language Models for Developing Trusted Code through Automated Program Verification](https://arxiv.org/abs/2610.30432) | 提出TrustedCode自动化框架，利用大语言模型对AI生成的Python代码进行类型检查、通过Why3框架转换为WhyML，并借助Why3后端的自动定理证明器依据形式化规范完成验证，从而建立对软件正确性的信任。 |
| [^28] | [The Customer Is the Builder: How AI Coding Agents Are Rewriting the Economics of Software](https://arxiv.org/abs/2610.30307) | 本文提出“软件替代前沿”这一新概念，并发现AI编程智能体的能力约每129天翻一番、已识别出575个潜在开源替代方案，表明客户“自建而非购买”的软件经济模式正在被重写。 |
| [^29] | [How Prompt Style Affects AI Accuracy and Speed in Competitive Programming](https://arxiv.org/abs/2610.30223) | 该研究通过在IOI准确性任务与NP难速度任务上测试九种提示风格，发现没有普适最佳提示，提示风格的选择应取决于任务类型和统计汇总方式，且单一指标评估会掩盖提示效果的不稳定性。 |
| [^30] | [Agentic Architecture of Information Systems. The AAM/5C Model: A Reference Model for Distributed Autonomy](https://arxiv.org/abs/2610.30221) | 本书提出AAM/5C参考模型，将五项认知能力实例化于七个逻辑层，并通过单一授权规则R-AUT实现提议、决策与执行三者的分离，为大型组织中智能体信息系统的分布式自主性提供架构方案与可构造验证的保证契约。 |
| [^31] | [JANUS: Agentic Cross-Language Fuzzing Harness Generation for Android Native Libraries](https://arxiv.org/abs/2610.30218) | JANUS是首个采用智能体方法为Android原生库自动生成跨语言JNI感知模糊测试驱动的工具，解决了现有工具在跨JNI边界精确分析不可扩展、以及无法构造JNI方法所需的复杂复合对象这两大难题。 |
| [^32] | [A Consolidated Deployment Edition of a Microservice Core-Banking Platform: Composing Services by Reference into Three Hosts](https://arxiv.org/abs/2610.30212) | 本文提出将微服务核心银行平台的五个API服务通过项目引用整合为三个主机的新部署版本，在保持相同代码库和构建的前提下，验证了该组合版本在全部844个操作和47个金融工作负载案例中与原有九任务部署功能等价，从而在保留微服务代码组织优势的同时降低了部署开销。 |
| [^33] | [Detection Is Not Enough: Reasoning Traces as Testable Program Hypotheses](https://arxiv.org/abs/2610.30200) | 该研究发现，在没有参考解答、评判模型和权重更新的条件下，对大模型推理轨迹的自检验存在结构性上限——多数错误轨迹源于正确执行了错误的计划，因此仅仅依靠检测并不足够，应将推理轨迹视为可测试的程序假设。 |
| [^34] | [Agentic Verified Optimization of Real-World Software](https://arxiv.org/abs/2610.30188) | 该论文提出“智能体验证优化”新任务：让AI智能体生成更快的程序实现，同时产出以现有实现为规约的机器验证等价性证明，从而避免测试漏洞，并在Lean 4中开发Aqueduct框架将其应用于真实世界软件。 |
| [^35] | [Self-Evolving Static Analysis via LLM Pseudo-Labeling](https://arxiv.org/abs/2610.30184) | SESAA框架让LLM智能体从零规则开始，仅依靠另一个LLM批评者的反馈（无需真实标签）自进化出可读可审计的静态分析查询规则，在Python OWASP基准上取得0.742的平均F1分数，超越主流工具默认配置并与最优调优结果相当。 |
| [^36] | [When EVM Defaults Break: Three SDK Patterns for UTXO-Family Wallets](https://arxiv.org/abs/2610.30183) | 本文以 Nervos CKB 的 TypeScript SDK（CCC）为回溯性案例，提炼出交易组合器、统一签名器抽象和连接器 UI 三种可迁移的 SDK 设计模式，以解决主流账户模型 SDK 假设在扩展 UTXO 链与异构密码学钱包环境下失效的问题。 |
| [^37] | [Deep Search in Real-World Web Environments](https://arxiv.org/abs/2610.30181) | 该论文提出了 OpenDS 任务及 OpenDS-World 基准，用于评估智能体在真实、开放的互联网环境中自主进行多步骤深度搜索、跨多个独立来源发现并整合证据，从而得出唯一事实性答案的能力。 |
| [^38] | [MCP vs CLI for AI Agents: A Comparative Analysis of Tool Integration Strategies for Autonomous Systems](https://arxiv.org/abs/2610.30157) | 本文通过14个维度的比较分析表明，CLI在单用户本地工作流中效率更高，而MCP能够满足CLI从根本上无法实现的细粒度访问控制、凭证隔离、结构化审计和多租户治理等需求，且渐进式披露技术可消除MCP的上下文窗口开销弱点。 |
| [^39] | [Code Comprehensor: Hierarchical Documentation and Cross-Modal Retrieval for Large Repositories](https://arxiv.org/abs/2610.30016) | 该论文提出Code Comprehensor混合检索-推理框架，通过代码依赖调用图、层次化文档生成（CodeScribe）和CLIP式跨模态检索模型（CodeContrast），实现对大型代码库相关代码上下文的快速语义精准检索，并为基于MCP的LLM问答提供可靠依据。 |
| [^40] | [Compositional Safety Failures in Harness Evolution: Identification and Runtime Monitoring](https://arxiv.org/abs/2609.33123) | 该研究首次系统揭示了Harness演化中的组合式安全失效——即各自安全且不损害效用的组件更新在交互后可能引发不安全的智能体行为——在三个安全基准上识别出43个两两组合和18个不可约的三路组合安全失效，并提出运行时监控方法以应对跨组件安全验证的组合复杂度难题。 |
| [^41] | [Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search](https://arxiv.org/abs/2609.29636) | 该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。 |
| [^42] | [Extending FunctionGemma for Practical On-Device Mobile Function Calling](https://arxiv.org/abs/2609.25373) | 该研究通过构建涵盖十五类设备控制操作的约9,500个对话的合成数据集MOBILEACTIONSEXTENDED对FunctionGemma 270M进行微调，将设备端Android函数调用的端到端准确率从29.3%大幅提升至76.5%，并使组合模型在Google数据集上达到82.3%。 |
| [^43] | [AuditCoder: Process-Auditable Code Generation through Responsibility-Preserving Task Graphs](https://arxiv.org/abs/2607.29529) | 提出了多智能体系统AuditCoder，通过在实现前为任务图中的节点分配稳定ID并保留代码归属关系，将每段代码与其规划承诺、检查结果和修复历史关联起来，从而实现过程可审计的代码生成。 |
| [^44] | [No Edges, No Verdict: A Large-Scale Empirical Study of Declared Dependency Graphs in 78K SBOMs in the Wild](https://arxiv.org/abs/2607.22140) | 这项研究首次大规模分析了78,612个真实SBOM中的依赖图，发现超过半数未声明任何边缘，且边缘生成主要由生成器决定而非软件本身，这对漏洞分析有重要影响。 |
| [^45] | [PosterHarness: Turning Scientific Poster Generation into an Auditable Instruction-Following Benchmark](https://arxiv.org/abs/2607.03006) | 该论文提出POSTERHARNESS，一个可审计的基准框架，通过“占位符优先”契约将科学海报生成转化为可度量的指令遵循任务，要求模型只负责视觉设计而绝不伪造数据图表，从而能够客观检测占位符准确性、宽高比合规性与拒绝合成图形等关键属性。 |
| [^46] | [A Multi-Dimensional, Per-Pass Empirical Study of the LLVM Optimization Pipeline](https://arxiv.org/abs/2606.31238) | 通过将LLVM的-O3优化流水线分解为113个累积遍前缀、在30个PolyBench/C内核上进行84,750次多维测量，该研究利用字节级相同的二进制构建零分布来校准显著性，首次系统揭示了该流水线是非单调的——36%显著改变运行时间的优化遍转换实际上是性能回退。 |
| [^47] | [APO-DD: Dialogue Disentanglement via Reflective Optimization of Multi-Component Prompts](https://arxiv.org/abs/2606.07894) | APO-DD通过改编GEPA框架，利用执行轨迹和错误决策反思自动优化任务指令与聊天表示两个组件，其优化后的提示词在对话解缠任务上显著超越手动优化的提示词，且在其他聊天领域无需进一步优化即可保持优势。 |
| [^48] | [Measuring the Permission Gate: A Stress-Test Evaluation of Claude Code's Auto Mode](https://arxiv.org/abs/2604.04978) | 该论文首次对 Claude Code 自动模式权限系统进行独立压力测试评估，通过自建的 AmPermBench 基准发现在模糊授权场景下其端到端假阴性率高达 81.0%，远高于 Anthropic 在生产流量上报告的 17%。 |
| [^49] | [LIDL: LLM Integration Defect Localization via Knowledge Graph-Enhanced Multi-Agent Analysis](https://arxiv.org/abs/2601.05539) | LIDL提出了一种基于知识图谱增强的多智能体分析框架，用于定位LLM集成软件中因提示词、API调用、配置和模型输出等异构制品交互失配而产生的新型集成缺陷，克服了传统缺陷定位技术无法捕获跨层依赖和缺乏语义推理能力的局限。 |
| [^50] | [Model of Things (MoT): A Model-Driven Approach to Streamline Cloud-of-Things Application Development](https://arxiv.org/abs/2512.14613) | 本文提出了万物模型（MoT），一种将基于UML的建模与低代码执行概念相结合的模型驱动方法，旨在降低技术门槛、简化万物云应用的开发。 |
| [^51] | [DynamiQ: Unlocking the Potential of Dynamic Task Allocation in Parallel Fuzzing](https://arxiv.org/abs/2510.04469) | DynamiQ利用程序调用图结构定义任务并基于运行时反馈动态调整任务分配，大幅减少了并行模糊测试中的冗余探索，在超过40 CPU年的评估中显著提升分支覆盖率并发现了9个未知漏洞。 |

# 详细

[^1]: VERA：基于欧盟人工智能法案评估大语言模型部署的负责任人工智能仪表盘

    VERA: A Responsible AI Dashboard for Assessing LLM Deployments in Light of the EU AI Act

    [https://arxiv.org/abs/2610.31588](https://arxiv.org/abs/2610.31588)

    VERA 是一个开源自托管仪表盘，让非技术审查人员能够便捷地对大语言模型运行合规基准测试，依据欧盟人工智能法案对十二项可衡量的技术要求进行评分，并追踪评估过程的能耗，已实际部署于一家大型欧洲银行。

    

    在受监管环境中部署大语言模型的机构，在批准一项部署之前，通常需要关于负责任人工智能实践的技术证据。COMPL-AI 提供了对欧盟人工智能法案的一种技术解读，将这些证据组织为六大原则组下的十八项技术要求，其中十二项可以由当前的基准测试来衡量。然而，产生这些证据的基准测试套件通常需要通过命令行操作，而负责批准部署的工作人员并非工程师。我们提出了 VERA，一个开源的自托管仪表盘，已部署于一家大型欧洲银行以促进其负责任人工智能流程。审查员只需选择一个模型并对其运行公开的基准测试。VERA 会对当前基准测试可衡量的十二项技术要求进行评分，并测量每次评估所消耗的能源，为环境影响提供运行级别的证据。我们让四个已部署服务的开源权重模型通过了相关要求……（摘要原文在此处截断）

    arXiv:2610.31588v1 Announce Type: new  Abstract: Institutions that deploy large language models in regulated settings often need technical evidence of responsible AI practices before approving a deployment. COMPL-AI provides a technical interpretation of the EU AI Act that organises this evidence under six principle groups and eighteen technical requirements, twelve of which current benchmarks can measure. The benchmark suites that produce this evidence are typically operated from a command line, while the staff who approve deployments are non-engineers. We present VERA, an open-source self-hosted dashboard deployed to facilitate responsible AI processes within a large European bank. A reviewer selects a model and runs public benchmarks against it. VERA scores the twelve requirements that current benchmarks can measure and instruments the energy consumed by each evaluation, adding run-level evidence for environmental impact. We run four served open-weight models through the requirement
    
[^2]: CAMGen：一种核心变更感知的多视图提交信息生成框架

    CAMGen: A Core Change-Aware Multi-View Framework for Commit Message Generation

    [https://arxiv.org/abs/2610.31552](https://arxiv.org/abs/2610.31552)

    CAMGen通过识别代表主要变更意图的核心实体，并借助仓库级语义聚类构建功能语义视图，将多代码实体协作实现的功能上下文融入提交信息生成，从而生成更能准确解释变更及其影响的提交信息。

    

    提交信息帮助开发者理解代码变更并支持软件维护。近期的方法利用仓库上下文来改进基于大语言模型（LLM）的提交信息生成（CMG）。然而，现有的CMG方法对于理解局部编辑如何影响由多个代码实体共同实现的功能方面提供的支持有限。这种功能层面的理解对于生成能够解释主要变更及其影响的提交信息至关重要。为了解决这一问题，我们提出了CAMGen，一个核心变更感知的多视图提交信息生成框架，它将功能上下文融入到生成过程中。CAMGen首先识别代表主要变更意图的核心实体；然后通过仓库范围的语义聚类形成候选功能单元，并利用包含核心实体的单元构建功能语义视图。该视图为解释变更提供了语义上下文……

    arXiv:2610.31552v1 Announce Type: new  Abstract: Commit messages help developers understand code changes and support software maintenance. Recent approaches use repository context to improve commit message generation (CMG) with large language models (LLMs). However, existing CMG methods provide limited support for understanding how local edits affect functionality jointly implemented by multiple code entities. This functional understanding is essential for generating commit messages that explain the main change and its impact. To address this problem, we propose CAMGen, a core change-aware multi-view commit message generation framework that incorporates functional context into generation. CAMGen identifies core entities that represent the primary change intent. It then forms candidate functional units through repository-wide semantic clustering and uses those containing the core entities to construct a Functional Semantic View. This view supplies semantic context for interpreting the a
    
[^3]: 编码代理的自适应技能粒度：运行时路由与宏到微恢复

    Adaptive Skill Granularity for Coding Agents: Runtime Routing and Macro-to-Micro Recovery

    [https://arxiv.org/abs/2610.31461](https://arxiv.org/abs/2610.31461)

    该论文提出自适应技能粒度框架，将技能粒度作为基于任务结构、执行状态和宏技能失败条件的动作抽象策略，通过粒度控制器与有限状态恢复执行器在运行时动态路由宏技能或微级工具调用，并在宏技能失败时实现宏到微的自动恢复，从而提升编码代理的成功率与成本效率。

    

    可复用技能使大语言模型代理能够将任务特定的流程、工具可供性和执行指导打包为模块化单元。先前的工作大多研究代理在执行前如何检索、组合或合成技能。然而，在编码代理中，执行带来了一个额外的运行时控制问题：同一任务可以通过原始工具调用、宏技能、或以宏技能开始并在失败时回退到微级恢复的混合执行方式来解决。我们提出了自适应技能粒度（Adaptive Skill Granularity）框架，该框架将技能粒度视为一种以任务结构、执行状态和宏技能失败为条件的动作抽象策略。该系统结合了基于特征的粒度控制器和有限状态恢复执行器。我们使用一种确定性的编码代理协议评估该框架，采用验证器定义的成功率、总成本以及宏技能采纳率、失败恢复等过程指标进行衡量。

    arXiv:2610.31461v1 Announce Type: cross  Abstract: Reusable skills allow large language model agents to package task-specific procedures, tool affordances, and execution guidance into modular units. Prior work has largely studied how agents retrieve, compose, or synthesize skills before execution. In coding agents, however, execution poses an additional runtime control problem: the same task may be solved through primitive tool calls, macro skills, or hybrid execution that starts with a macro and falls back to micro-level recovery. We introduce Adaptive Skill Granularity, a framework that treats skill granularity as an action-abstraction policy conditioned on task structure, execution state, and macro-skill failures. The system combines a feature-based granularity controller and a finite-state recovery executor. We evaluate the framework with a deterministic coding-agent protocol using verifier-defined success, total cost, and process metrics such as macro adoption, failure recovery, r
    
[^4]: Procrustes-Bering-Sheaft：面向微服务的自动化模型发现与虚拟崩溃测试

    Procrustes-Bering-Sheaft: Automated Model Discovery and Virtual Crash Tests for Microservices

    [https://arxiv.org/abs/2610.31444](https://arxiv.org/abs/2610.31444)

    该论文提出了 Procrustes-Bering-Sheaft 工具链，通过蒙特卡洛故障采样自动化实现微服务的模型发现与虚拟崩溃测试，并能检测操作可达性与实际观测成功之间的不匹配。

    

    Procrustes-Bering-Sheaft 实现了面向微服务的轻量级模型发现与虚拟崩溃测试的自动化。每个测试通过蒙特卡洛采样服务和通信故障来估计操作的可用性。Procrustes 检查部署与追踪证据；Bering 发现依赖关系并纳入副本声明；Sheaft 评估所声明的操作成功条件。各工具通过独立的命令行接口交换可复用、带版本管理的工件。该工具链在早期基于图的方法基础上进行了扩展，新增了强制调用契约、故障容忍度扫描、基线比较、组件重要性分析以及 CI/CD 门禁。作者在一个 HTTP 操作上演示了该工作流程：六种受控执行状态暴露了可达性与实际观测成功之间的两处不匹配，而一个显式的必需调用谓词与全部六种状态相匹配。两次建模的单副本增加产生了不同的决策：推荐方案保留了可用性——（原文摘要在此处截断）

    arXiv:2610.31444v1 Announce Type: new  Abstract: Procrustes-Bering-Sheaft automates lightweight model discovery and virtual crash tests for microservices. Each test estimates operation availability by Monte Carlo sampling of service and communication failures. Procrustes checks deployment and tracing evidence; Bering discovers dependencies and incorporates replica declarations; Sheaft evaluates declared operation-success conditions. Independent command-line interfaces exchange reusable, versioned artifacts. The toolchain extends an earlier graph-based approach with mandatory-call contracts, failure-tolerance sweeps, baseline comparison, component-importance analysis, and CI/CD gates. We demonstrate the workflow on one HTTP operation. Six controlled execution states exposed two mismatches between reachability and observed success; an explicit required-call predicate matched all six. Two modelled one-replica additions produced different decisions: recommendation retained an availability-
    
[^5]: ConQEC：面向量子纠错码实现的具体-符号混合测试

    ConQEC: Concolic Testing for Quantum Error-Correcting Code Implementations

    [https://arxiv.org/abs/2610.31443](https://arxiv.org/abs/2610.31443)

    本文提出ConQEC，首个面向量子纠错码实现的具体-符号混合测试框架，通过结合具体泡利故障重放与符号化修复路径约束，不仅能暴露最终逻辑错误，还能生成故障-症状-译码器-恢复（FSDR）全链路诊断，定位修复路径最早发生分歧的阶段。

    

    量子纠错码（QECC）实现是一类混合的软件-控制系统：测量得到的症状驱动译码器调度、恢复操作以及逻辑校验。现有的QECC验证技术能够证明形式化义务，通用的量子测试也能暴露最终的故障，但开发者仍然缺乏一种可扩展的测试工作流，能够生成具体的故障并解释QECC修复路径最初在何处发生分歧。本文提出了ConQEC，据我们所知，这是首个面向QECC实现的具体-符号混合测试框架。通过将具体的（泡利）故障重放与符号化的修复路径约束相结合，ConQEC生成的测试不仅能暴露最终的逻辑错误，还能提供故障-症状-译码器-恢复（FSDR）诊断信息：包括注入的故障、观测到的症状、译码器决策、恢复操作、预言机结果以及最早发生分歧的阶段。其针对泡利故障、症状谓词、译码器与恢复的符号编码……

    arXiv:2610.31443v1 Announce Type: cross  Abstract: Quantum error-correcting code (QECC) implementations are hybrid software-control systems: measured syndromes drive decoder dispatch, recovery actions, and logical checks. Existing QECC verification techniques can prove formal obligations, and general quantum testing can expose final failures, but developers still lack a scalable testing workflow that generates concrete faults and explains where a QECC repair path first diverges. This paper presents ConQEC, to our knowledge the first concolic testing framework for QECC implementations. By combining concrete (Pauli) fault replay with symbolic repair-path constraints, ConQEC generates tests that expose not only final logical failures but also fault--syndrome--decoder--recovery (FSDR) diagnostics: the injected fault, observed syndrome, decoder decision, recovery action, oracle result, and earliest divergent stage. Its symbolic encoding of Pauli faults, syndrome predicates, decoder and reco
    
[^6]: 使用多智能体LLM系统大规模自动化功能开关清理：DoorDash的工业实践经验

    Automating Feature Flag Cleanup at Scale with Multi-Agent LLM Systems: An Industrial Experience at DoorDash

    [https://arxiv.org/abs/2610.31351](https://arxiv.org/abs/2610.31351)

    本文提出一个基于多智能体LLM的端到端自动化功能开关清理系统，在DoorDash的大规模工业实践中，对评估的50个过期功能开关中90%成功生成了获得工程师批准的删除PR。

    

    功能开关（Feature flags）是一种运行时配置机制，用于渐进式功能发布、实验和紧急关闭开关。一旦功能完全发布，该开关及其控制的代码就会成为技术债务，并可能通过意外启用休眠代码路径而引入运营风险。在DoorDash，超过56,000个功能开关分布在623个代码仓库中，截至2026年5月，目前有1,659个处于过期状态，内部估计每次手动清理需要一到两个小时。现有的基于规则的清理工具在DoorDash的代码库中被证明是不够的。我们的贡献是一个端到端的基于LLM的自动化功能开关清理系统，它集成了使用MCP查询实时功能开关发布数据、Jira工单处理以及端到端的拉取请求创建。我们的系统在评估的50个过期开关中成功产生了45个（90%）获得工程师批准的开关删除PR，证明了该系统在大规模运营中的可行性，每次清理平均……

    arXiv:2610.31351v1 Announce Type: new  Abstract: Feature flags are runtime configuration mechanisms used for gradual feature rollouts, experimentation, and kill switches. Once a feature is fully rolled out, the flag and its gated code become technical debt and can introduce operational risk through inadvertently enabling dormant code paths. At DoorDash, over 56,000 feature flags span 623 repositories, with 1,659 currently stale as of May 2026, with internal estimates of one to two hours per manual cleanup. Existing rule-based cleanup tools proved insufficient in DoorDash's codebases. Our contribution is an end-to-end automated LLM-based feature flag cleanup system that integrates querying live feature flag rollout data using MCP, Jira ticket processing, and end-to-end pull request creation. Our system successfully produced engineer-approved flag removal PRs for 45 (90%) of the 50 stale flags it was evaluated on, proving the system's operational viability at scale with each cleanup aver
    
[^7]: 类型化模板模糊测试

    Typed Template Fuzzing

    [https://arxiv.org/abs/2610.31342](https://arxiv.org/abs/2610.31342)

    TTF提出了一种轻量级的基于大语言模型的模糊测试方法，通过在预处理阶段利用LLM将种子转化为类型化模板，从而无需特定语言解析器、类型检查器或模糊测试过程中的LLM查询，即可实现语义感知的高效变异式模糊测试。

    

    使用复杂输入对软件进行模糊测试长期以来一直是一项挑战。传统上，模糊测试器要么以编程方式对语言约束进行编码（这需要大量的工程工作），要么依赖用户来声明语法和约束。近来基于大语言模型（LLM）的模糊测试器解决了这些缺点，但也带来了新问题：一些方法使用LLM进行测试生成，牺牲了吞吐量，且当LLM出错时会产生无效输入；另一些方法则依赖于特定语言的解析器、静态分析和AST操作库。我们提出了类型化模板模糊测试（Typed Template Fuzzing，TTF），这是一种轻量级的基于LLM的模糊测试方法。其核心思想是在预处理步骤中使用LLM将种子转化为类型化模板。随后，这些类型化模板能够支持语义感知的、基于变异的模糊测试，而无需特定语言的解析器、类型检查器，也无需在模糊测试过程中进一步查询LLM。通过利用种子与模板之间的对应关系，TTF还可以（摘要在此处被截断）

    arXiv:2610.31342v1 Announce Type: new  Abstract: Fuzzing software with complex inputs has been a longstanding challenge. Traditionally, fuzzers either programmatically encode language constraints, requiring significant engineering effort, or rely on users to declare grammars and constraints. Recent LLM-based fuzzers address these drawbacks but introduce new issues. Some use LLMs for test generation, sacrificing throughput and producing invalid inputs when the LLM makes mistakes. Others rely on language-specific parsers, static analysis, and AST manipulation libraries. We introduce Typed Template Fuzzing (TTF), a lightweight LLM-based fuzzing methodology. The key idea is to use LLMs in a preprocessing step to transform seeds into typed templates. The typed templates then enable semantically aware, mutation-based fuzzing without language-specific parsers, type checkers, or further LLM queries during fuzzing. By exploiting the correspondence between seeds and templates, TTF can moreover r
    
[^8]: 超智能软件工程：随能力同步扩展的治理

    Superintelligent Software Engineering: Governance That Scales with Capability

    [https://arxiv.org/abs/2610.31306](https://arxiv.org/abs/2610.31306)

    本文提出“超智能软件工程”概念，主张当AI智能体能力超越人类工程师时，必须让治理机制（门禁、契约、决策与衰减控制）随能力同步扩展，才能应对知识衰减与人工审查失效，保障软件正确性。

    

    软件智能体正在被训练以超越人类工程师；我们认为，仅有能力并不能产出更好的软件工程。我们将超智能软件工程定义为一门学科，即使用能力超越人类工程师的智能体来构建和演进软件，并使治理（门禁、契约、决策与衰减控制）随能力同步扩展。该论证建立在三个论点之上：第一，项目的正确性依赖于代码本身所不承载的决策与契约；第二，知识会衰减；第三，一旦智能体的进步速度超过其审查者，通过人工阅读所能捕获的、基于过时知识构建的变更比例将不断缩小。一个简单的模型表明，在审查容量固定的情况下，逃逸的变更随吞吐量线性增长，而阅读所能捕获的比例则趋近于零；治理能够降低这一斜率，而仅凭能力则无法做到。针对GitHub上星标最多的452个公共仓库的探索性快照（原文摘要在此处截断）。

    arXiv:2610.31306v1 Announce Type: new  Abstract: Software agents are being trained to exceed human engineers; we argue that capability alone does not produce better software engineering. We define superintelligent software engineering as the discipline of building and evolving software with agents whose capability exceeds that of human engineers, in which governance (gates, contracts, decisions and decay control) scales with capability. The argument rests on three claims. First, correctness in a project rests on decisions and contracts that the code does not carry. Second, knowledge decays. Third, once an agent outpaces its reviewers, reading catches a shrinking share of the changes built on stale knowledge. A simple model shows that, past a fixed review capacity, escaping changes grow linearly with throughput while the share that reading catches falls toward zero; governance can lower the slope, capability alone cannot. An exploratory snapshot of 452 of the most-starred public GitHub 
    
[^9]: 超越准确性：在多智能体LLM流水线中解耦任务分解与类型化状态质量保障

    Beyond Accuracy: Disentangling Decomposition from Typed-State Quality Assurance in Multi-Agent LLM Pipelines

    [https://arxiv.org/abs/2610.31280](https://arxiv.org/abs/2610.31280)

    该研究发现多智能体LLM流水线在需求分类与优先级排序任务中的性能提升主要源于任务分解所支持的专门化提示，而非类型化状态质量保障机制。

    

    多智能体LLM流水线常被认为能在软件工程任务中带来性能提升，但这些提升究竟源于智能体之间的类型化契约，还是源于任务分解所支持的专门化提示，往往并不清楚。我们在需求工程领域研究了这一归因问题，比较了三种架构，用于将需求分类为功能性或非功能性需求，并随后用MoSCoW方法对其进行优先级排序：单次调用基线；两次调用变体（复用流水线的专门化提示但传递未经验证的中间结果）；以及采用Pydantic验证的类型化状态的流水线。实验覆盖18种条件，包括PROMISE NFR+数据集（625条需求）、三个本地托管的小型语言模型和两种语言。结果表明，相对于基线的F1提升与两次调用的专门化提示变体相关（最高提升+0.075），而类型化状态仅带来微小差异（-0.003到+0.013），且仅在六组比较中的一组中显著。

    arXiv:2610.31280v1 Announce Type: cross  Abstract: Multi-agent LLM pipelines are often credited with gains in software engineering tasks, but it is rarely clear whether a gain stems from the typed contract between agents or from the specialized prompts that decomposition enables. We study this attribution problem in Requirements Engineering, comparing three architectures for classifying requirements as functional or non-functional and then prioritizing them with MoSCoW: a single-call baseline; a two-call variant that reuses the pipeline's specialized prompts but passes an unvalidated intermediate; and a pipeline with a Pydantic-validated typed state. The 18 conditions cover PROMISE NFR+ (625 requirements), three locally hosted small language models, and two languages. The F1 gain over the baseline is associated with the two-call, specialized-prompt variant (up to +0.075), whereas the typed state adds a small difference (-0.003 to +0.013) that is significant in one of six comparisons. B
    
[^10]: ProductWebBench：超越任务完成度的网站连续性评估

    ProductWebBench: Evaluating Website Continuity Beyond Task Completion

    [https://arxiv.org/abs/2610.31118](https://arxiv.org/abs/2610.31118)

    该论文提出 ProductWebBench 基准，首次在真实前端仓库上评估“网站连续性”（即代码变更后网站其余部分是否仍正常运行），并发现即使通过所有任务要求检查的变更运行中，仍有 28.2% 会破坏网站的其他功能。

    

    前端编码智能体通常只用一个问题来评判：被要求的目标是否被构建出来。而真实的工程变更还承载着第二项义务——网站其余部分必须保持正常运作。我们将这项第二义务称为“网站连续性”，并推出 ProductWebBench，一个在真实前端仓库上同时对两者进行评分的基准：包含 320 个针对现有代码的变更任务和 80 个长周期构建任务。每个任务在评估前都会冻结仓库快照、分阶段需求、浏览器证据和确定性检查，并且只有当参考实现通过全部检查、而似是而非的错误实现无法通过时，任务才会被收录。模拟用户会逐个阶段揭示需求，且此前已通过的所有检查在整个运行过程中持续生效。来自 13 个模型的 8,737 次运行表明，两者会分道扬镳：在通过所有“请求工作”检查的变更任务运行中，仍有 28.2% 未通过连续性检查——出现空白路由、水平溢出、控制台报错等问题。

    arXiv:2610.31118v1 Announce Type: new  Abstract: Frontend coding agents are usually graded on one question: did the requested target get built. Real engineering changes carry a second obligation, because the rest of the site has to keep working. We call that second obligation website continuity, and introduce ProductWebBench, a benchmark that scores both on living frontend repositories: 320 Change tasks over existing code and 80 long-horizon Build tasks. Every task freezes a repository snapshot, staged requirements, browser evidence and deterministic checks before evaluation, and enters only once a reference passes every check and plausible-wrong ones fail. A simulated user reveals the requirements one stage at a time, and every check cleared earlier stays armed for the rest of the run. Across 8,737 runs from 13 models the two come apart: of the Change runs passing every requested-work check, 28.2% still fail a continuity check -- a blank route, horizontal overflow, console errors -- a
    
[^11]: 追踪大语言模型在UML类图生成中的推理幻觉：实证分类体系与缓解研究

    Tracing Hallucinations in LLM Reasoning for UML Class Diagram Generation: An Empirical Taxonomy and Mitigation Study

    [https://arxiv.org/abs/2610.30946](https://arxiv.org/abs/2610.30946)

    该研究针对LLM生成UML类图过程中的推理幻觉，构建了包含提示忠实度、代码忠实度和推理质量三个维度共十一种子类型的实证分类体系，并发现代码忠实度错误是出现频率最高的幻觉类型。

    

    大型语言模型（LLM）能够根据自然语言需求生成UML类图，但幻觉问题可能会损害生成图的正确性和完整性。现有的评估方法大多只关注最终生成的图，而没有检查具备推理能力的LLM的中间推理轨迹，而这些轨迹可以揭示幻觉产生的位置。我们构建了一个包含十一种幻觉子类型的实证分类体系，分为三个维度：提示忠实度（需求与推理之间的偏差）、代码忠实度（推理与生成的PlantUML代码之间的差异）以及推理质量（推理过程内部的问题）。该分类体系基于60条人工标注的DeepSeek-R1推理轨迹构建，通过双LLM标注流水线应用于另外120条DeepSeek-R1轨迹，并在20条DeepSeek-V4-Pro轨迹上进行了验证。在两个数据集中，代码忠实度都是DeepSeek-R1中出现频率最高的幻觉维度。

    arXiv:2610.30946v1 Announce Type: cross  Abstract: Large Language Models (LLMs) can generate UML class diagrams from natural language requirements, but hallucinations may compromise the correctness and completeness of the resulting diagrams. Existing evaluations mostly assess the final diagram, without examining the intermediate traces of reasoning-capable LLMs, which can reveal where these hallucinations arise. We develop an empirical taxonomy of eleven hallucination sub-types organised into three dimensions: Prompt Faithfulness (deviations between requirements and reasoning), Code Faithfulness (discrepancies between reasoning and generated PlantUML code), and Reasoning Quality (problems within the reasoning). The taxonomy was derived from 60 manually annotated DeepSeek-R1 traces, applied to 120 further DeepSeek-R1 traces through a dual-LLM annotation pipeline, and examined on 20 DeepSeek-V4-Pro traces. Code Faithfulness was the most frequent dimension for DeepSeek-R1 in both datasets
    
[^12]: REGATTA：面向企业AI智能体的跨领域测量框架与故障注入测试工具

    REGATTA: A Cross-Domain Measurement Framework and Fault-Injection Harness for Enterprise AI Agents

    [https://arxiv.org/abs/2610.30924](https://arxiv.org/abs/2610.30924)

    REGATTA提出了一个企业AI智能体评估测量框架，通过125项带来源引用的监管清单、25个故障语料库及故障注入评分工具，评估测试套件暴露指定故障的能力并审计其引用规则的可信度，审计发现大量条目存在引用缺陷。

    

    企业AI智能体评估需要证据来证明测试套件能否暴露指定的故障，以及其参考规则是否可信。我们提出了REGATTA（通过两阶段测试与保障进行智能体监管评估），这是一个包含三种工具的测量框架：一份包含125个条目、附来源引用的清单，涵盖美国和越南的五个“行业×司法辖区”组合单元；一个涵盖五个类别的25个故障语料库；以及一个在设计和执行阶段对套件级故障检测能力进行评分的测试工具。我们在来自三个LLM生成器的180个生成套件（5,400个场景）上检验了四种测量诊断指标。来源审计发现：24个条目错误陈述了现行规则，另有46个引用缺陷，31个条目仅有非约束性、无法识别或无法验证的来源，24个条目经验证无误。只有最初被标记存在实质性问题的条目接受了作者人工审查；所有24个干净条目仅接受了AI审查。因此，结果……（原文摘要在此处截断）

    arXiv:2610.30924v1 Announce Type: new  Abstract: Enterprise AI-agent evaluation needs evidence about whether test suites expose specified faults and whether their reference rules are trustworthy. We introduce REGATTA (Regulatory Evaluation of Agents via Two-stage Testing and Assurance), a measurement framework with three instruments. A 125-item source-cited checklist spans five industry-by-jurisdiction cells across the United States and Vietnam. A 25-fault corpus covers five categories, and a harness scores suite-level fault detection at the design and execution stages. We examine four measurement diagnostics on 180 generated suites (5,400 scenarios) from three LLM generators. A source audit finds 24 items misstating the rule in force, 46 additional citation defects, 31 items with only non-binding, unidentifiable, or unverifiable sources, and 24 verified clean. Only initially substance-flagged items received author review; all 24 clean rows received AI review only. Results therefore de
    
[^13]: 当代码变得廉价，评估变得稀缺：一份关于AI辅助敏捷软件工程的经验报告

    When Code Gets Cheap, Evaluation Gets Scarce: An Experience Report on AI-Assisted Agile Software Engineering

    [https://arxiv.org/abs/2610.30916](https://arxiv.org/abs/2610.30916)

    本文通过一名开发者仅用三天、未手写一行代码便构建并部署合规关键的考试评分系统的实战经验，提出AI对软件工程的核心影响不是加速编程，而是改变稀缺资源——代码变得充裕，而可靠评估和工程判断成为新的约束性瓶颈。

    

    编码助手被广泛报道能加速代码生产，但大多数叙述止步于此。我们报告了一个合规关键、人在回路的考试评分系统的端到端构建与部署过程：该系统由一名开发者独立完成，全程未手写任何代码，并受到一个门控流程的保护——从利益相关者画像到经过审计的系统行为。初始系统在三个工作日内建成，并能够对真实的考生群体进行评分。我们从这一经验中得出一个论断：AI对软件工程的改变，与其说是让编程变得更快，不如说是改变了什么是稀缺资源。编写代码变得充裕，而可靠的评估——即判断结果是否正确并满足实际需求——在每次迭代中仍然不可或缺，并成为约束性的瓶颈。工程判断力是第二种稀缺资源。若干缺陷通过了测试套件：它们位于测试所模拟的边界处——浏览器和语言模型API——或者仅在实际使用中才显现出来。

    arXiv:2610.30916v1 Announce Type: new  Abstract: Coding assistants are widely reported to speed up code production; most accounts stop there. We report the end-to-end construction and deployment of a compliance-critical, human-in-the-loop exam-grading system, built by a single developer who wrote no code by hand, protected by a gated process from stakeholder persona to audited behavior. The initial system was built in three working days and could grade a real cohort. We draw one claim from this experience: AI changes software engineering less by making programming faster than by changing what is scarce. Writing code becomes abundant. Reliable evaluation-knowing whether a result is correct and meets the actual need-remains necessary at every iteration and becomes the binding constraint. Engineering judgment remains a second scarcity. Several defects passed the test suite: they sat at the boundaries the tests had mocked, the browser and the language-model API, or appeared only in real us
    
[^14]: ATLAS：一个将二进制程序提升为规范编译器中间表示的多智能体框架

    ATLAS: A Multi-Agent Framework for Lifting Binary to Canonical Compiler IR

    [https://arxiv.org/abs/2610.30877](https://arxiv.org/abs/2610.30877)

    ATLAS是一个多智能体框架，通过结构感知的区域级检索与协作式细化循环，将高度优化的无符号二进制程序重建为规范且形式有效的优化前LLVM中间表示。

    

    将去除符号、经过高度优化的二进制程序提升为规范的、优化前的LLVM中间表示（IR），对于在源代码不可用时进行下游安全分析和遗留软件迁移至关重要。然而，激进的编译器优化会重写控制流和数据流、移除结构元数据，并引入类型擦除的低级惯用模式，导致当前的提升器产生格式错误或语义偏离的IR。我们提出了ATLAS，一个旨在从优化后的反汇编代码中重建形式有效且规范的LLVM IR的多智能体框架。ATLAS结合了：（i）结构感知的区域级检索，利用伪探针归因将优化后的指令与规范IR示例对齐；（ii）协作式细化循环，以缩小概率生成与确定性编译器约束之间的差距。具体而言，一个提升智能体基于检索到的……（摘要内容在此处截断）

    arXiv:2610.30877v1 Announce Type: new  Abstract: Lifting stripped, heavily optimized binaries to a canonical, pre-optimization LLVM intermediate representation (IR) is essential for downstream security analysis and legacy software migration when source code is unavailable. However, aggressive compiler optimizations rewrite control and data flow, remove structural metadata, and introduce type-erased low-level idioms, causing current lifters to produce malformed or semantically divergent IR. We present ATLAS, a multi-agent framework designed to reconstruct formally valid and canonical LLVM IR from optimized disassembly.ATLAS combines (i) Structure-aware, region-level retrieval, which utilizes pseudo-probe attribution to align optimized instructions with canonical IR exemplars, and (ii) a collaborative refinement loop that closes the gap between probabilistic generation and deterministic compiler constraints. Concretely, a Lift agent generates an initial IR draft conditioned on retrieved 
    
[^15]: 分离字段评分：当智能体汇集工具使用经验时保护未被使用过的工具

    Separate Field Scoring Protects Unused Tools When Agents Pool Tool-Use Experience

    [https://arxiv.org/abs/2610.30867](https://arxiv.org/abs/2610.30867)

    提出SYNAPSE方法，将工具使用记录作为独立字段进行评分，在智能体共享工具使用经验提升已使用工具检索效果的同时，有效避免了从未被使用过的工具检索性能的下降。

    

    从共享工具目录中调用工具的智能体可以汇集它们学到的关于哪种工具适合哪种请求的经验。我们测量了汇集的使用示例对工具检索的影响，并分别针对从未被使用过的预留工具和所有其他工具进行报告。将使用示例作为已使用工具的额外条目加入索引，会提高这些工具的Hit@5（即正确工具是否出现在前五名），但在三个工具目录中的两个以及我们测试的两种嵌入器上，会大幅降低预留工具的Hit@5。如果给已使用工具添加相同数量、等长但填充其他工具文本的条目，预留工具的Hit@5会下降相近的幅度，却几乎不带来任何增益。我们将这种把使用记录作为独立字段进行评分的方法称为SYNAPSE。在StableToolBench上，预留的正确工具进入前五名的频率高于为使用记录保留槽位的路由器方法（0.785对0.683），并且这一在预留工具上的优势在所有三个工具目录和两种嵌入器上均保持成立。

    arXiv:2610.30867v1 Announce Type: new  Abstract: Agents that call tools from a shared catalog can pool what they learn about which tool serves which request. We measure what pooled usage examples do to tool retrieval, reporting separately the held-out tools, for which no agent has usage, and all other tools. Adding usage examples to the index as extra entries for used tools raises Hit@5 (whether a gold tool is in the top five) for those tools and lowers it for held-out tools, by a large margin on two of three catalogs and with both embedders we tested. Giving used tools the same number of equally long entries filled with other tools' text lowers held-out Hit@5 about as much, but gives little or none of the gain. We call our method of scoring usage as its own field SYNAPSE. On StableToolBench, the held-out gold tool appears in its top five more often than with a router that reserves slots for usage (0.785 against 0.683), and this held-out lead holds on all three catalogs with both embed
    
[^16]: 以太坊开发者想要什么，以及他们是否得到了：以太坊 Stack Exchange 十年

    What Ethereum Developers Ask For, and Whether They Get It: Ten Years of Ethereum Stack Exchange

    [https://arxiv.org/abs/2610.30829](https://arxiv.org/abs/2610.30829)

    该论文分析了以太坊 Stack Exchange 十年间的全部54,637个问题，利用准确率达87.9%-93.5%的大语言模型主题标注流水线，首次完整刻画了开发者自发求助需求的分布——集中于合约编写、部署与调试——并检验该平台是否仍能有效解决这些问题。

    

    以太坊社区通常通过调查和访谈来了解开发者在哪些方面遇到困难，这些方法记录的是开发者被问及特定主题时所说的话。而以太坊 Stack Exchange (ESE) 记录的是开发者主动提出的求助需求，以及是否有人解决了这些问题。我们研究了2016年1月至2025年12月间发布在 ESE 上的全部问题，共计54,637个，并使用一条基于提示的主题标注流水线为每个问题打上标签——该流水线以八个以太坊特定主题为种子，同时可接纳新主题（共22个主题）。一个由三家提供商的语言模型组成的委员会对标签进行筛选，并对每个争议案例进行人工裁决，使标签准确率达到87.9%至93.5%。由于语料库是完整记录，因此所有比率均不存在抽样误差。我们探究开发者寻求哪些方面的帮助、该平台是否仍能解决这些问题，以及在无法解决的情况下，究竟是问题本身发生了变化，还是问题的解决方式发生了变化。需求集中于合约的编写、部署和调试；工具链是第二……（原文摘要在此截断）

    arXiv:2610.30829v1 Announce Type: new  Abstract: The Ethereum community learns where developers struggle through surveys and interviews, which record what developers say when asked about a chosen theme. Ethereum Stack Exchange (ESE) records what they ask for unprompted, and whether anyone resolves it. We study every question posted to ESE from Jan 2016 to Dec 2025, 54,637 in all, and label each with a prompt-based topic pipeline seeded with eight Ethereum-specific topics and open to new ones (22 topics in all). A three-provider committee of language models screens the labels, and manual adjudication of every disputed case puts their accuracy at 87.9 to 93.5%. Because the corpus is the complete record, no rate carries sampling error. We ask what developers ask for help with, whether the venue still resolves it, and, where it does not, whether the questions changed or their resolution did.   Demand is concentrated on writing, deploying, and debugging contracts; the toolchain is the secon
    
[^17]: VLTest：发现视觉-语言模型中的图像-文本联合失效

    VLTest: Finding Joint Image-Text Failures in Vision-Language Models

    [https://arxiv.org/abs/2610.30794](https://arxiv.org/abs/2610.30794)

    VLTest是一种黑盒视觉-语言模型测试方法，通过分别和联合变异图像与文本输入，并采用四单元格对比和VLM裁判验证，精准发现仅由图像-文本组合触发的模型失效。

    

    视觉-语言模型（VLM）越来越多地被嵌入到软件系统中，对它们进行测试会带来两个问题：其一，对任一输入进行变异都可能改变预期答案，因此输出变化本身并不必然意味着失效；其二，有些失效只在两个输入同时被变异时才会出现，而要将这类失效归因于输入组合，就必须先单独测试每一个变异。我们提出了VLTest，一种黑盒VLM测试方法。它通过编辑VQGAN潜码生成图像变体，通过“重写-扩展-剪枝”流水线生成文本变体，并过滤掉可能改变预期答案的变体；在测试组合之前先单独测试每个变体，从而使每一个联合失效都拥有四个匹配的测试单元格。一个VLM裁判只有在核实输入仍保持预期答案后才会确认失效。在涵盖八个VLM（包括两个商业API模型）的三项任务上，该判定器确认了81.7%的候选失效，与人工标注相比精确率达到88.8%至94.6%，并通过四单元格确认……

    arXiv:2610.30794v1 Announce Type: new  Abstract: Vision-language models (VLMs) are increasingly embedded in software systems. Testing them raises two problems. Mutating either input can change the expected answer, so an output change is not by itself a failure. Some failures appear only when both inputs are mutated; attributing them to the combination requires testing each mutation alone. We present VLTest, a black-box VLM testing approach. It generates image variants by editing VQGAN latent codes and text variants with a rewrite-expand-prune pipeline, filters variants likely to change the expected answer, and tests each variant alone before testing combinations, so every joint failure has four matched cells. A VLM judge confirms a failure only after checking that the input keeps the expected answer. On three tasks with eight VLMs, including two commercial API models, the oracle confirms 81.7% of the candidates, with 88.8% to 94.6% precision against human labels, and four-cell confirma
    
[^18]: Trace2World：从真实交互轨迹重建应用世界以实现可扩展的GUI智能体训练

    Trace2World: Reconstructing App Worlds from Real Interaction Traces for Scalable GUI Agent Training

    [https://arxiv.org/abs/2610.30792](https://arxiv.org/abs/2610.30792)

    Trace2World通过将全局行为重建与环境实现分离，从真实交互轨迹构建忠实于真实应用行为的可执行GUI环境用于智能体训练，在十个应用的200个工作流上实现91.1%的端到端复现率，且基于来源的验证使真实设备上的训练增益保留率达94.0%。

    

    扩展GUI强化学习需要能够保留真实应用行为的环境。然而，将行为推断与编码相耦合，可能会把部分轨迹转变成状态转换错误的可运行模拟（mock）。我们提出Trace2World，它将全局行为重建与具体实现分离开来。该方法首先将多条轨迹调和为一个共享的、与来源关联的应用模型，然后编译出可执行的环境。通过独立回放来源工作流来定位不一致之处，从而进行有边界的实现修复。经过验证的世界进而可以在来源支持的行为范围内，支持面向学习者自适应的任务与环境增强。Trace2World从1,570条轨迹中构建了50个环境。在十个应用的200个工作流上，91.1%的工作流被端到端复现，而最强的任务局部匹配基线仅为59.8%。基于来源的验证在真实设备上保留了94.0%的模拟训练增益，而自我验证下仅为54.7%。学习者引导（摘要在此处截断）

    arXiv:2610.30792v1 Announce Type: new  Abstract: Scaling GUI reinforcement learning requires environments that preserve real-app behavior. Yet coupling behavior inference with coding can turn partial traces into runnable mocks with incorrect state transitions. We introduce \textbf{Trace2World}, which separates global behavior reconstruction from implementation. It reconciles traces into a shared, source-linked app model, then compiles an executable environment. Independent replay of source workflows localizes mismatches for bounded implementation repair. Validated worlds then support learner-adaptive task and environment augmentation within source-supported behavior. From 1,570 traces, Trace2World constructs 50 environments. On 200 workflows across ten apps, 91.1% are reproduced end to end, versus 59.8% for the strongest matched task-local baseline. Source-grounded validation retains 94.0% of simulated training gains on real devices, versus 54.7% under self-validation. Learner guidance
    
[^19]: 贷款处理员与贷款审核员能识别AI编辑的报表吗？一项实证研究

    Can Loan Processors and Underwriters Spot AI-Edited Statements? An Empirical Study

    [https://arxiv.org/abs/2610.30758](https://arxiv.org/abs/2610.30758)

    该研究通过实证实验发现，经验丰富的贷款审核员和处理员在仅凭文件的首次审阅中，仅有约10.6%的判断准确指出了AI伪造财务文件中的编辑痕迹（且全部依赖标识符不一致），表明仅靠人工文件审查很难发现AI编辑的伪造贷款文件。

    

    AI辅助编辑可以在保留原发行方熟悉版式的同时更改申请人的身份，从而生成贷款文件。我们研究了经验丰富的贷款审核员和贷款处理员在首次审阅（仅基于文件本身）时能否识别此类伪造文件。通过Upwork招募的三位从业者，每人收到一份申请人材料包，这些材料包来自134份伪造的PDF文件，而这些PDF文件是在完全知情同意的前提下，基于47份真实财务文件生成的。编辑保留了金额、日期和余额；在分发之前，处理流程中引入的标识符不一致被有意保留。审阅者未被告知文件已被篡改，共返回了132份文件级判断。将他们的解释与原始文件核对后，共发现14处有编辑依据的疑虑（10.6%）：12个“可疑”判断和2个“信息不足”判断。所有有依据的疑虑均涉及文件内部或跨文件的标识符不一致，而没有报告的视觉…（摘要在此处被截断）

    arXiv:2610.30758v1 Announce Type: cross  Abstract: AI-assisted editing can produce loan documents that preserve familiar issuer layouts while changing the applicant's identity. We investigated whether experienced underwriters and loan processors identify such forgeries in first-pass, document-only review. Three practitioners recruited through Upwork each received one applicant package drawn from 134 forged PDFs derived from 47 genuine financial documents supplied with full consent. The edits preserved amounts, dates and balances; identifier inconsistencies introduced by the pipeline were knowingly retained before distribution. Reviewers were not told that files had been altered and returned 132 file-level judgements. Checking their explanations against the originals yielded 14 edit-supported concerns (10.6%): 12 Suspicious judgements and two Insufficient information judgements. All supported concerns involved identifier disagreements within or across documents, and no reported visual c
    
[^20]: 从自然语言到经验证的C代码：基于ACSL规范的验证引导式生成

    From Natural Language to Verified C: Verification-Guided Generation with ACSL Specifications

    [https://arxiv.org/abs/2610.30725](https://arxiv.org/abs/2610.30725)

    该论文提出一种验证引导式生成方法，将C代码实现与ACSL规范作为统一产物进行协同设计，使AI生成的程序能够通过自动形式化验证并获得机器可检查的正确性证据。

    

    摘要：AI辅助代码生成放大了对自动化形式化验证的需求。生成的程序不仅要可执行，还必须附带机器可检查的正确性证据。现有方法可以从自然语言提示中合成功能上看似合理的代码或表面上合理的规范；然而，这些方法并未将实现与规范作为一个针对验证优化而设计的统一产物进行显式协同设计。因此，自动验证器往往无法完成证明义务。典型的失败模式包括：缺失或隐式的证明锚点、对自动推理不友好的程序结构，以及由于代码与规范独立生成而导致的代码级状态与规范级需求事实之间的不一致。即使验证成功，也可能仅建立了忽略关键需求行为的弱规范。本文提出了……

    arXiv:2610.30725v1 Announce Type: new  Abstract: AI-assisted code generation amplifies the demand for automated formal verification. Generated programs must not only be executable but also accompanied by machine-checkable evidence of correctness. Existing approaches can synthesize functionally plausible code or plausible-looking specifications from natural-language prompts; however, they do not explicitly co-design the implementation and specification as a unified artifact optimized for verification. Consequently, automated verifiers often fail to discharge proof obligations. Typical failure modes include missing or implicit proof anchors, program structures that are hostile to automated reasoning, and inconsistencies between code-level state and specification-level requirement facts arising from the independent generation of code and specifications. Even when verification succeeds, it may only establish weak specifications that omit requirement-critical behavior. This paper introduces
    
[^21]: 从代币到积分再到成果：跨模型、应用与云平台的AI定价

    From Tokens to Credits to Outcomes: Pricing AI Across Models, Applications, and Cloud Platforms

    [https://arxiv.org/abs/2610.30701](https://arxiv.org/abs/2610.30701)

    本文对2023至2026年间AI软件的定价模式演变进行了首次系统性可复现研究，构建了涵盖福布斯AI 50、云平台及独特设计的定价原语与积分机制数据集和分类法。

    

    2023年至2026年间，AI改变了软件的定价方式。传统SaaS按席位收费，因为多服务一个用户的成本几乎为零。而模型推理具有非零边际成本，且不同客户间的使用量相差数个数量级，由此催生了预付积分、按量计费、席位加用量混合模式以及成果定价等定价方式，它们作为基于席位定价的补充（在极少数情况下甚至将其取代）。然而，这些定价模式尚未得到系统性的刻画。本文对公开定价材料——定价页面与文档——开展了一项可复现的研究，涵盖三个样本组：作为AI原生核心样本的完整2026年福布斯AI 50榜单、按明确规则纳入的销售AI服务的云平台，以及一个由9个独特定价设计组成的冻结目的性样本组。本文的贡献包括：（1）一个数据集，以注明日期的一手来源对每个产品的定价原语和积分机制进行编码，并将随论文公开发布；（2）一个分类法……（摘要在此处截断）

    arXiv:2610.30701v1 Announce Type: new  Abstract: Between 2023 and 2026, AI changed how software is priced. Traditional SaaS charged per seat because serving one more user cost almost nothing. Model inference has non-zero marginal cost, and usage varies by orders of magnitude across customers, producing prepaid credits, metered usage, hybrid seat-plus-usage models, and outcome pricing that supplement, and in rare cases replace, seat-based pricing. These pricing models have not been systematically characterized. This paper presents a reproducible study of public pricing artifacts - pricing pages and documentation - across three cohorts: the full Forbes AI 50 (2026) as the AI-native core sample, cloud platforms selling AI services included by a documented rule, and a frozen purposive cohort of 9 distinctive designs. It contributes (1) a dataset coding each offering's pricing primitive and credit mechanics with dated primary sources, prepared for public release with the paper, (2) a taxono
    
[^22]: 面向大语言模型智能体长期记忆的治理状态损坏：一项跨模型压力测试

    Governance-State Corruption in Long-Term Memory for LLM Agents: A Cross-Model Stress Test

    [https://arxiv.org/abs/2610.30698](https://arxiv.org/abs/2610.30698)

    该论文定义并量化了一类新故障“治理状态损坏”——即在存储内容不变时仅改变哪些记忆记录进入模型上下文的故障，通过跨六个提供商的2,080对故障实验证明此类损坏可使LLM智能体下游性能下降高达40个百分点，而基于oracle的治理强制执行能实现100%恢复。

    

    摘要：一个LLM智能体能够正确检索记忆，却可能在其有效性状态发生变化后错误地使用它。我们研究“治理状态损坏”：即在存储内容和任务保持不变的情况下，改变哪些留存记录进入模型可见上下文的故障。在六个模型提供商中，731个基于基准的基础案例产生了2,080个匹配的故障对。在离线控制平面检查中，损坏在96.25%的配对中改变了有效投影，而基于oracle的治理强制执行在100%的情况下将其恢复。在下游评估方面，一个确定性选择的非概率LongMemEval/Memora面板解决了300个匹配三元组中的298个（涉及238个不同的基础案例）。干净与损坏条件下的对比在LongMemEval二元正确性上达到40.54个百分点（95%基础聚类置信区间[30.32, 50.71]），在重新缩放的Memora FAMA上达到30.60个百分点（[23.31, 37.69]）；两个数据源均显示恢复效果。OAKS的原生选项对等性在所有6,714个已解决的输出中均保持成立。

    arXiv:2610.30698v1 Announce Type: cross  Abstract: An LLM agent can retrieve a memory correctly yet use it incorrectly after its validity state changes. We study governance-state corruption: faults that alter which retained records enter model-visible context while stored content and tasks remain fixed. Across six providers, 731 benchmark-grounded bases yield 2,080 matched fault pairs. In offline control-plane checks, corruption changes the operative projection in 96.25% of pairs, while oracle-backed enforcement restores it in 100%. For downstream evaluation, a deterministically selected, non-probability LongMemEval/Memora panel resolves 298 of 300 matched triplets (238 distinct bases). Clean-minus-corrupted contrasts are 40.54 percentage points in LongMemEval binary correctness (95% base-cluster interval [30.32, 50.71]) and 30.60 points in rescaled Memora FAMA ([23.31, 37.69]); both sources also show recovery. OAKS Novel native option parity holds for all 6,714 resolved outputs. Full-
    
[^23]: 过时的指令文件凌驾于编码智能体中的代码之上：一句话修复方案

    Stale Instruction Files Override the Code in Coding Agents: A One-Sentence Fix

    [https://arxiv.org/abs/2610.30571](https://arxiv.org/abs/2610.30571)

    研究发现编码智能体（如Claude Code和Codex CLI）会盲目遵循过时的项目指令文件而忽视实际代码，甚至在察觉到冲突并指出来时仍然如此，而只需在用户指令文件中加入一句“现有代码优先”即可有效修复这一问题。

    

    目的：编码智能体会遵循项目的指令文件（如 CLAUDE.md 或 AGENTS.md），即使该文件与现有代码不一致，甚至当它们察觉到这种不一致时也是如此。我们测量了这种行为，并测试在用户自己的指令文件中加入一句话能否让智能体转而遵循代码。方法：我们构建了40个Python项目，其中指令文件要求一种规范，而现有代码使用另一种规范（例如使用print而非日志记录）。Claude Code 和 Codex CLI 在每个项目中添加一个功能，共进行320次运行，分别测试用户级指令文件中包含和不包含“现有代码优先”这句话的情况。结果：在没有这句话的情况下，90.1%的智能体选择遵循过时的文件。Claude Code 在80次运行中有79次指出了冲突，但仍然遵循该文件。Codex CLI 在80次运行中有20次重写了现有代码以匹配指令文件，其中有17次导致项目自身的测试失败。

    arXiv:2610.30571v1 Announce Type: new  Abstract: Purpose: Coding agents follow a project's instruction file, such as CLAUDE.md or AGENTS.md, even when it disagrees with the existing code and even when they notice the disagreement. We measure this behaviour and test whether one sentence in the user's own instruction file makes agents follow the code instead.   Methods: We built 40 Python projects in which the instruction file asks for one convention while the existing code uses another, for example print instead of logging. Claude Code and Codex CLI added a feature to each project, with and without a sentence in the user-level instruction file that gives the existing code precedence, in 320 runs.   Results: Without the sentence, 90.1% of the agents' choices followed the stale file. Claude Code pointed out the conflict in 79 of 80 runs and still followed the file. Codex CLI rewrote the existing code to match the file in 20 of 80 runs, and the project's own tests then failed in 17 of them
    
[^24]: 大语言模型能否改进自身的执行框架？基于ARC-AGI-3的对照研究

    Can LLMs Improve Their Own Harness? A Controlled Study on ARC-AGI-3

    [https://arxiv.org/abs/2610.30499](https://arxiv.org/abs/2610.30499)

    该论文通过受控实验首次系统考察前沿大语言模型能否自主构建并迭代改进自己的执行框架，让模型在ARC-AGI-3交互式基准上独立编写、使用并两次修订完整框架代码库，再于二十个从未参与改进的留出游戏上检验其泛化能力。

    

    通用智能系统能够高效地习得新技能，而无需人类为每个新任务重新设计其软件脚手架。然而，语言模型很少单独行动——执行框架是一种符号程序，它代表模型编排各类调用，负责处理感知、记忆、工具使用、异常恢复和推理分配等事务。ARC-AGI-3是一个交互式基准测试，要求智能体探索陌生环境、在行动过程中自主弄清目标并持续学习。在该基准上，由人类引导的系统已经在公开游戏中报告了满分区分数，说明有效的设计是确实存在的。本文提出的问题是：前沿模型能否自行构建并持续改进这样一个执行框架？通过一个固定的编码引导流程，每个模型先构建一个完整的任务执行框架，通过该框架在五个开发游戏上执行任务，并对整个代码库进行两次修订。随后，每个版本都在二十个从未参与过适应过程的留出游戏上进行评估。

    arXiv:2610.30499v1 Announce Type: cross  Abstract: A generally intelligent system acquires new skills efficiently without requiring humans to redesign its software scaffold for every new task. Yet language models rarely act alone. A harness is a symbolic program that orchestrates calls to the model, handling perception, memory, tool use, recovery, and inference allocation on the model's behalf. ARC-AGI-3 is an interactive benchmark in which agents must explore unfamiliar environments, figure out their goals on the fly, and keep learning as they go. On it, human-steered systems already report ceiling scores across the public games, so we know effective designs exist. We ask whether frontier models can build and improve such a harness on their own. Through a fixed coding bootstrap, each model constructs a complete task harness, acts through that harness on five development games, and revises the entire codebase twice. Every version is evaluated on twenty games that never influence adapta
    
[^25]: 从文字到属性：面向智能体原生科学、用 ARATest 检验研究论断

    From Prose to Properties: Testing Research Claims with ARATest for Agent-Native Science

    [https://arxiv.org/abs/2610.30495](https://arxiv.org/abs/2610.30495)

    该论文提出 ARATest，一个基于属性的测试库，通过将研究论断抽象为可执行的“属性骨架”并配合类型化声明、共享案例类型和三值判定语义进行自动化检验，使 AI 智能体产生的研究论断变得可测试、可证伪，从而解决验证速度跟不上论断产出速度的问题。

    

    AI 智能体现在能够运行实验并报告发现，因此研究产生论断的速度已经超过了研究者能够核查的速度。每个实证论断都隐含着一个必须在其所有案例上成立的属性，但它只用文字来陈述这一属性。如果没有明确的属性，任何测试都无法判定哪些证据可以证伪一个论断，因此验证无法跟上智能体产生论断的速度。为了使论断可测试，我们通过抽象出论断的“属性骨架”来近似论断的语义——属性骨架是一种在多个论断中反复出现的参数化关系。我们提出了 ARATest，一个基于属性的测试库，它将这些骨架作为可执行测试进行检验。类型化声明将论断绑定到骨架上，并固定该论断所比较的量、案例、范围和数值策略。共享的案例类型使每个生成器都以论断的统计单位提供案例。三值语义为每个属性赋予三种判定结果之一：成立、被违反或不确定。

    arXiv:2610.30495v1 Announce Type: cross  Abstract: AI agents now run experiments and report findings, so research produces claims faster than researchers can check them. Each empirical claim implies a property that must hold over its cases, but it states that property only in prose. Without an explicit property, no test can decide which evidence would falsify a claim, so validation cannot keep pace with the claims agents produce. To make claims testable, we approximate a claim's semantics by abstracting its property skeleton, a parameterized relation that recurs across claims. We present ARATest, a property-based testing library that checks these skeletons as executable tests. A typed declaration binds a claim to a skeleton and fixes the claim's compared quantities, cases, scope, and numerical policy. A shared case type makes every generator supply cases in the claim's statistical unit. A three-valued semantics assigns each property one of three verdicts: holds, violated, or inconclusi
    
[^26]: 轨迹所示与标签所言：审计智能体评估中的行为检测器

    What the Trace Shows and What the Label Says: Auditing Behavioral Detectors in Agent Evaluation

    [https://arxiv.org/abs/2610.30491](https://arxiv.org/abs/2610.30491)

    该研究通过在GSM8K上构建包含损坏计算器输出的工具使用场景，审计了三个轨迹级行为检测器，发现最终正确性、可见重新验证与错误传播等标签会系统性不一致，并揭示了“静默工具输出覆盖”这一关键失败模式——即模型在不重新调用工具的情况下使用了与损坏输出不同的值。

    

    大语言模型智能体通常通过将执行轨迹简化为最终答案正确性、工具验证和错误传播等标签来进行评估。这些标签是由解析器、行为检测器和评估框架规则产生的度量结果，它们可能无法完整捕捉轨迹所显示的内容。我们在基于GSM8K构建的工具使用场景中研究这一问题，其中智能体会调用一个输出偶尔被损坏的计算器，这为工具返回了什么、模型后续使用了什么以及检测器报告了什么提供了明确的真值参照。我们在六个生产级模型上、每个模型30道题目，审计了三个轨迹级检测器。最终正确性、可见的重新验证和错误传播之间可能以系统性的方式出现不一致。一个关键的失败模式是静默的工具输出覆盖：计算器返回了一个被损坏的值，但模型后续在没有再次调用工具的情况下使用了另一个（通常是正确的）值，因此坏值停止了传播，而（摘要在此处被截断）

    arXiv:2610.30491v1 Announce Type: cross  Abstract: LLM agents are often evaluated by reducing execution traces to labels such as final-answer correctness, tool verification, and fault propagation. These labels are measurements produced by parsers, behavioral detectors, and harness rules, and they may not fully capture what the trace shows. We study this in a tool-use setting built on GSM8K, where agents call a calculator whose outputs are occasionally corrupted, giving clear ground truth for what the tool returned, what the model later used, and what the detectors reported. We audit three trace-level detectors across six production models and 30 problems per model. Final correctness, visible re-verification, and fault propagation can disagree in systematic ways. A key failure mode is silent tool-output override: the calculator returns a corrupted value, but the model later uses a different, often correct, value without calling the tool again, so the bad value stops propagating while a 
    
[^27]: TrustedCode：通过自动化程序验证评估大语言模型开发可信代码的能力

    TrustedCode: Evaluating Large Language Models for Developing Trusted Code through Automated Program Verification

    [https://arxiv.org/abs/2610.30432](https://arxiv.org/abs/2610.30432)

    提出TrustedCode自动化框架，利用大语言模型对AI生成的Python代码进行类型检查、通过Why3框架转换为WhyML，并借助Why3后端的自动定理证明器依据形式化规范完成验证，从而建立对软件正确性的信任。

    

    近期报告（如《谁在使用AI编写代码？》）指出，GitHub上约30%的Python代码贡献是由AI生成的。鉴于这类代码固有的不可靠性，形式化验证成为建立软件正确性信任的一种有前景的方法。尽管利用大语言模型（LLM）进行形式化验证日益受到关注，但由于编写形式化规范以及依据规范验证程序正确性存在困难，其实际应用仍然有限。现有工具主要支持Java和C等语言的验证，对Python的支持很少。我们提出了TrustedCode，一个用于验证AI生成的Python代码的自动化框架。TrustedCode使用大语言模型来：(i) 对Python代码进行类型检查，(ii) 通过Why3框架将其转换为WhyML，(iii) 使用Why3后端的自动定理证明器依据形式化规范对其进行验证。我们的方法在……（原文摘要到此截断）

    arXiv:2610.30432v1 Announce Type: new  Abstract: Recent reports, such as "Who is using AI to code?", indicate that approximately 30\% of Python code contributions to GitHub are AI-generated. Given the inherent unreliability of this code, formal verification emerges as a promising approach to establish trust in software correctness. Although leveraging large language models (LLMs) for formal verification has garnered increasing attention, practical adoption remains limited due to the challenges in writing formal specifications and verifying program correctness against them. Existing tools predominantly support verification for languages such as Java and C, with minimal support for Python. We present TrustedCode, an automated framework for verifying AI-generated Python code. TrustedCode uses LLMs to (i) type-check Python code, (ii) translate it into WhyML via the Why3 framework, and (iii) verify it against formal specifications using Why3 back-end automated theorem provers. Our method in
    
[^28]: 客户即建造者：AI 编程智能体如何重写软件经济学

    The Customer Is the Builder: How AI Coding Agents Are Rewriting the Economics of Software

    [https://arxiv.org/abs/2610.30307](https://arxiv.org/abs/2610.30307)

    本文提出“软件替代前沿”这一新概念，并发现AI编程智能体的能力约每129天翻一番、已识别出575个潜在开源替代方案，表明客户“自建而非购买”的软件经济模式正在被重写。

    

    AI 编程智能体可能会改变组织“购买软件”还是“自行构建所需功能”的决策。我们聚焦于组织及专业买家，他们的领域知识能够帮助其为自己工作中使用的工具进行规格说明与评估。我们定义了“软件替代前沿”，即客户能够以低于购买成本的方式构建并维护、且达到可接受标准的功能集合。利用公开数据，我们将不断增长的构建能力与市场变化指标进行对比。自2023年以来，领先智能体以50%成功率所能完成的基准任务时长（以人类工作时间衡量）大约每129天翻一番。通过一项广泛的检索，我们为选定的商业产品识别出575个潜在的开源替代方案，尽管其中部分功能匹配仍需独立验证。Craft 套件在12.0天的公开开发中产出了12个应用程序代码库，但这些仍属早期版本，其完整性尚未得到验证。

    arXiv:2610.30307v1 Announce Type: new  Abstract: AI coding agents could change whether organizations buy software or build the functionality they need. We focus on organizations and professional buyers whose domain knowledge can help them specify and evaluate tools for their work. We define the software substitution frontier as the functions customers could build and maintain at an acceptable standard for less than buying them. Using public data, we compare growing capacity to build with indicators of market change. Since 2023, the length of benchmark tasks that leading agents can complete at 50 percent success, measured in human working time, has doubled about every 129 days. A broad search identified 575 potential open-source alternatives to selected commercial products, although some functional matches still require independent validation. The Craft suite produced 12 application repositories over 12.0 days of public development, but these remain early versions with unverified comple
    
[^29]: 提示风格如何影响AI在竞技编程中的准确性与速度

    How Prompt Style Affects AI Accuracy and Speed in Competitive Programming

    [https://arxiv.org/abs/2610.30223](https://arxiv.org/abs/2610.30223)

    该研究通过在IOI准确性任务与NP难速度任务上测试九种提示风格，发现没有普适最佳提示，提示风格的选择应取决于任务类型和统计汇总方式，且单一指标评估会掩盖提示效果的不稳定性。

    

    大型语言模型越来越多地被用于竞技编程，其输出对提示的措辞方式十分敏感。已有研究比较了竞技编程中不同提示策略在准确性上的表现，但未比较运行速度，也没有将结果与人类表现进行对照。在一项探索性研究中，我们在GPT-5.5上对九种风格各选取一个提示进行测试，涵盖国际信息学奥林匹克竞赛（IOI）的三个准确性任务和两个NP难速度任务。我们发现不存在单一的最佳提示：最强的提示取决于任务类型，也取决于结果是用均值还是中位数来汇总，这表明至少对于这些任务，提示风格应当与目标相匹配，而不能被当作通用方案。许多均值分数上的差异来自少数离群运行，这说明仅用单一指标评估提示的效果，可能会掩盖其有效性的巨大波动。与（人类表现）进行的描述性比较（摘要在此处截断）

    arXiv:2610.30223v1 Announce Type: new  Abstract: Large language models are increasingly used in competitive programming, and their outputs are sensitive to how a prompt is phrased. Prior work has compared prompt strategies for accuracy on competitive programming, but not for runtime speed, and has not situated the results against human performance. In an exploratory study of one prompt in each of nine styles on GPT-5.5 across three accuracy tasks from the International Olympiad in Informatics (IOI) and two NP-hard speed tasks, we found that no single prompt was best: the strongest prompt depended on the task type and on whether results were summarized by the mean or the median, suggesting that, at least for these tasks, prompt style should be matched to the objective rather than treated as universal. Many differences in mean scores came from a few outlier runs, showing that evaluating prompting by a single metric can obscure how much its effectiveness varies. Compared descriptively wit
    
[^30]: 信息系统中的智能体架构：AAM/5C模型——分布式自主性的参考模型

    Agentic Architecture of Information Systems. The AAM/5C Model: A Reference Model for Distributed Autonomy

    [https://arxiv.org/abs/2610.30221](https://arxiv.org/abs/2610.30221)

    本书提出AAM/5C参考模型，将五项认知能力实例化于七个逻辑层，并通过单一授权规则R-AUT实现提议、决策与执行三者的分离，为大型组织中智能体信息系统的分布式自主性提供架构方案与可构造验证的保证契约。

    

    大型组织日益面临本书所称的“认知复杂性之墙”：在数百个交互组件的规模下，维护依赖关系图并以事件发生的速度做出反应，已超出人类团队所能交付的能力范围。本概念性专著提出了AAM/5C模型，其中五项能力（捕获、理解、选择、协调、控制）在七个逻辑层（L1至L7）中被实例化。单一的授权规则R-AUT将提议者、决策者和执行者相互分离：概率规划器（L4）负责提议；治理层（L6）在签署的策略下做出决策，并将其裁定记录于签名的评估证明之中；安全层（L5）颁发绑定至该证明的短期行动令牌；策略执行点在目标层落实行动之前对两者进行验证。在明确的保证契约下，两条分析性命题（H7a、H8a）在构造上即成立。其十条命题中的其余部分……（原文摘要被截断）

    arXiv:2610.30221v1 Announce Type: new  Abstract: Large organizations increasingly face what this book calls the cognitive complexity wall: at the scale of hundreds of interacting components, maintaining the dependency map and reacting at the pace of incidents exceeds what a human team can deliver. This conceptual volume proposes AAM/5C, in which five capabilities (Capture, Comprehend, Choose, Coordinate, Control) are instantiated across seven logical layers, L1 to L7. A single authorization rule, R-AUT, separates who proposes, who decides and who executes: a probabilistic planner (L4) proposes; a governance layer (L6) decides under signed policies and records its verdict in a signed evaluation proof; a security layer (L5) issues a short-lived act token bound to that proof; a policy enforcement point verifies both before the target layer materializes the act. Under an explicit guarantee contract, two analytical statements (H7a, H8a) hold by construction. The other statements of its ten 
    
[^31]: JANUS：面向Android原生库的智能体跨语言模糊测试驱动生成

    JANUS: Agentic Cross-Language Fuzzing Harness Generation for Android Native Libraries

    [https://arxiv.org/abs/2610.30218](https://arxiv.org/abs/2610.30218)

    JANUS是首个采用智能体方法为Android原生库自动生成跨语言JNI感知模糊测试驱动的工具，解决了现有工具在跨JNI边界精确分析不可扩展、以及无法构造JNI方法所需的复杂复合对象这两大难题。

    

    现代Android应用大量依赖原生库，通过Java本地接口（JNI）桥接Java/Kotlin与C/C++。然而，这种跨语言互操作使内存安全的Android运行时暴露于不安全的C/C++世界，形成了一个被反复利用以实现设备完全攻陷的攻击面。虽然模糊测试是审查原生库的成熟技术，但最先进的测试驱动生成工具难以忠实地捕捉闭源跨语言API的使用：(i) 跨JNI边界的精确程序分析无法扩展，而轻量级近似方法则会遗漏关键的调用模式；(ii) 现有工具均无法解决JNI方法所消费的复杂复合对象的构造问题。在本文中，我们提出JANUS，这是首个采用智能体方法为Android原生库合成跨语言JNI感知模糊测试驱动的方案。JANUS迭代地解析显式和隐式的跨……

    arXiv:2610.30218v1 Announce Type: new  Abstract: Modern Android apps extensively rely on native libraries, bridging Java/Kotlin and C/C++ through the Java Native Interface (JNI). This cross-language interoperation, however, exposes the memory-safe Android Runtime to the unsafe C/C++ world, creating an attack surface that has been repeatedly exploited for full device compromise. While fuzzing is a proven technique for vetting native libraries, state-of-the-art harness generation tools struggle to faithfully capture closed-source cross-language API usage: (i) precise program analysis across the JNI boundary does not scale, whereas lightweight approximations miss critical invocation patterns; and (ii) none tackles the construction of complex composite objects consumed by JNI methods.   In this paper, we propose JANUS, the first agentic approach to synthesizing cross-language JNI-aware fuzzing harnesses for Android native libraries. JANUS iteratively resolves both explicit and implicit cro
    
[^32]: 微服务核心银行平台的整合部署版本：通过项目引用将服务组合为三个主机

    A Consolidated Deployment Edition of a Microservice Core-Banking Platform: Composing Services by Reference into Three Hosts

    [https://arxiv.org/abs/2610.30212](https://arxiv.org/abs/2610.30212)

    本文提出将微服务核心银行平台的五个API服务通过项目引用整合为三个主机的新部署版本，在保持相同代码库和构建的前提下，验证了该组合版本在全部844个操作和47个金融工作负载案例中与原有九任务部署功能等价，从而在保留微服务代码组织优势的同时降低了部署开销。

    

    微服务为团队提供了独立的部署和扩展能力，但每个已部署的服务也会占用计算资源并增加运维工作。本文研究了多租户.NET核心银行平台的第二个部署版本。现有版本以九个AWS ECS Fargate任务运行；而替代版本通过项目引用将五个API服务组合到一个API主机中，另外还有一个消费者主机和一个后台主机。服务代码仍保留在其原始存储库中，且两个版本继续从相同的提交构建。我们描述了组合机制、模块清单、启动检查、请求处理和配置优先级，然后在AWS部署的仿真环境中对两个版本进行比较。组合后的版本暴露了全部844个声明的操作。在47个金融工作负载案例中，两个版本产生了相同的观察结果和匹配的账本。在跨五种主要类型的读取探测中，231个端点中有230个...

    arXiv:2610.30212v1 Announce Type: new  Abstract: Microservices give teams independent deployment and scaling, but each deployed service also reserves compute and adds operational work. This paper examines a second deployment edition of a multi-tenant .NET core-banking platform. The existing edition runs as nine AWS ECS Fargate tasks; the alternative composes the five API services by project reference into one API host, alongside a consumer host and a background host. The service code remains in its original repositories, and both editions continue to be built from the same commits. We describe the composition mechanism, module manifest, start-up checks, request handling and configuration precedence, then compare the editions in an emulation of the AWS deployment. The composed edition exposes all 844 declared operations. Across 47 financial workload cases, both editions produced the same observed outcomes and matching ledgers. In read probes across five principal types, 230 of 231 endpo
    
[^33]: 检测并不足够：作为可测试程序假设的推理轨迹

    Detection Is Not Enough: Reasoning Traces as Testable Program Hypotheses

    [https://arxiv.org/abs/2610.30200](https://arxiv.org/abs/2610.30200)

    该研究发现，在没有参考解答、评判模型和权重更新的条件下，对大模型推理轨迹的自检验存在结构性上限——多数错误轨迹源于正确执行了错误的计划，因此仅仅依靠检测并不足够，应将推理轨迹视为可测试的程序假设。

    

    arXiv:2610.30200v1 公告类型：新论文 摘要：当语言模型被部署时，并不存在参考解答，因此模型写出的推理轨迹是判断其答案的唯一证据。各类无参考的替代方案各有不同的失败方式：自我评估会损害它本想保护的答案；采样一致性衡量的是稳定性而非正确性；而扰动研究则是从人工编写的模板中读取预期答案，评估的对象是模型而非轨迹本身。我们探究在没有参考解答、没有评判模型、也没有权重更新的条件下，单条推理轨迹能够被验证到什么程度，并发现：对于在 GSM8K 数据集上使用的冻结 LLaMA-2-7B-chat 模型，限制此类验证流程的并非检测能力。将轨迹与自身进行检验可以达到 0.741 的 AUROC，并因一个结构性原因而止步于此：三分之二的错误轨迹并未陈述任何错误的等式——它们正确地执行了一个错误的计划；而且在“错误计划”导致的失败案例中，检测器标记错误轨迹的频率反而低于标记正确轨迹的频率。基于候选派生的（摘要在此处截断）

    arXiv:2610.30200v1 Announce Type: new  Abstract: When a language model is deployed, no reference solution exists, so the reasoning trace it writes is the only evidence for judging its answer. The reference-free substitutes each fail differently: self-assessment degrades the answer it was meant to protect, sampling agreement measures stability rather than correctness, and perturbation studies read the expected answer off a human-written template, grading a model rather than a trace. We ask how far a single trace can be verified with no reference solution, no judge model and no weight update, and find, for a frozen LLaMA-2-7B-chat on GSM8K, that detection is not what limits such a pipeline. Checking a trace against itself reaches an AUROC of 0.741 and stops there for a structural reason: two thirds of wrong traces state no false equation, having executed a mistaken plan correctly, and on wrong-plan failures the detector flags a wrong trace less often than a right one. A candidate-derived
    
[^34]: 真实世界软件的智能体验证优化

    Agentic Verified Optimization of Real-World Software

    [https://arxiv.org/abs/2610.30188](https://arxiv.org/abs/2610.30188)

    该论文提出“智能体验证优化”新任务：让AI智能体生成更快的程序实现，同时产出以现有实现为规约的机器验证等价性证明，从而避免测试漏洞，并在Lean 4中开发Aqueduct框架将其应用于真实世界软件。

    

    程序优化旨在在保持程序行为不变的前提下提升性能。编码智能体已经能够通过改变算法、数据结构和执行策略来让程序运行得更快。然而，现有的等价性检查大多依赖于测试，而已有研究表明智能体会利用测试的漏洞，在不保留预期行为的情况下通过测试。我们提出智能体验证优化，其中智能体在生成更快实现的同时，还需提供与现有实现等价性的机器验证证明。由于现有实现本身充当规约，该任务避免了验证代码生成中编写规约的瓶颈，而定理证明领域的最新进展使得这些证明变得可行。我们在 Lean 4 中于函数和代码仓库两个层面定义了该任务，在 Lean 4 中程序既可以被执行也可以被推理验证。由于很少有真实世界的应用是用 Lean 编写的，因此我们开发了 Aqueduct，一个智能体框架（摘要在此处截断）。

    arXiv:2610.30188v1 Announce Type: cross  Abstract: Program optimization aims to improve performance while preserving behavior. Coding agents can already make programs faster by changing algorithms, data structures, and execution strategies. However, existing equivalence checks mostly rely on tests, which agents have been shown to exploit, passing the tests without preserving the intended behavior. We propose agentic verified optimization, in which an agent produces a faster implementation together with a machine-checked proof of equivalence to the existing one. Because the existing implementation serves as the specification, the task avoids the specification-writing bottleneck of verified code generation, and recent advances in theorem proving make the proofs attainable. We formulate the task at the function and repository levels in Lean 4, where programs can be both executed and reasoned about. Few real-world applications are written in Lean, so we develop Aqueduct, an agentic framewo
    
[^35]: 基于大语言模型伪标签的自进化静态分析

    Self-Evolving Static Analysis via LLM Pseudo-Labeling

    [https://arxiv.org/abs/2610.30184](https://arxiv.org/abs/2610.30184)

    SESAA框架让LLM智能体从零规则开始，仅依靠另一个LLM批评者的反馈（无需真实标签）自进化出可读可审计的静态分析查询规则，在Python OWASP基准上取得0.742的平均F1分数，超越主流工具默认配置并与最优调优结果相当。

    

    安全漏洞之所以能绕过静态分析，往往不是因为它们无法被检测，而是因为尚未针对眼前代码编写相应的检测规则。新框架、API和编码习惯不断制造这种规则空白，迫使开发者先发现故障、再手动编写新规则。我们提出了SESAA框架，让大语言模型（LLM）智能体从零规则出发，迭代进化出一组可读且可审计的Joern查询。该智能体仅从第二个LLM批评者处获得反馈——批评者通过检查被标记和未被标记的代码来识别可能的误报和疑似漏报，且无法访问真实标签。智能体随后根据这些反馈修订其查询规则。在Python版OWASP基准测试中，SESAA在8次独立运行中达到平均F1值0.742，远超五种广泛使用工具的默认配置，并与其中最强工具调优后的表现相当。

    arXiv:2610.30184v1 Announce Type: cross  Abstract: Security bugs often slip past static analysis not because they are impossible to detect, but because detection rules have not yet been written for the code in front of us. New frameworks, APIs, and coding idioms continually create such gaps, leaving developers to discover failures and manually encode new rules. We present SESAA, a framework in which an Large Language Model (LLM) agent starts from no rules and iteratively evolves a set of readable and auditable joern queries.   The agent receives feedback only from a second LLM critic, which inspects flagged and unflagged code to identify likely false alarms and suspected misses, without access to ground-truth labels. The agent then uses this feedback to revise its queries. On the OWASP Benchmark for Python, SESAA reaches a mean F1 of 0.742 across 8 independent runs, well above the default configurations of five widely used tools and comparable to the strongest of them once it is tuned 
    
[^36]: 当 EVM 默认假设失效时：面向 UTXO 系钱包的三种 SDK 设计模式

    When EVM Defaults Break: Three SDK Patterns for UTXO-Family Wallets

    [https://arxiv.org/abs/2610.30183](https://arxiv.org/abs/2610.30183)

    本文以 Nervos CKB 的 TypeScript SDK（CCC）为回溯性案例，提炼出交易组合器、统一签名器抽象和连接器 UI 三种可迁移的 SDK 设计模式，以解决主流账户模型 SDK 假设在扩展 UTXO 链与异构密码学钱包环境下失效的问题。

    

    主流 Web3 SDK（如 ethers.js、viem）捆绑了三个相互耦合的假设：基于账户模型的账本、单一的 secp256k1 签名器，以及无需输入选择的交易构建方式。一旦某条链采用扩展 UTXO 模型（本文指 Nervos CKB 上的 Cell 模型），并且钱包群体涵盖异构的密码学原语，这些假设便会失效。本研究以 TypeScript SDK——CKBers' Codebase (CCC) 为对象，通过回溯性案例研究，从其发布历史和导出 API 中提炼出三种反复出现的模式：交易组合器，将容量感知的输入补全与手续费估算声明式化；统一签名器抽象，通过单一接口将异构钱包的签名操作规范化；连接器 UI 组件，以即插即用的方式提供针对各钱包的接入引导。本文的贡献在于将这些模式阐述为面向 UTXO 系账本的 SDK 可迁移的设计经验。

    arXiv:2610.30183v1 Announce Type: cross  Abstract: Mainstream Web3 SDKs (e.g., ethers.js, viem) bundle three coupled assumptions: an account-based ledger, a single secp256k1 signer, and input-selection-free transaction construction. These assumptions fail once a chain adopts an extended-UTXO model (here, the Cell model on Nervos CKB), and once the wallet population spans heterogeneous cryptographic primitives (secp256k1, WebAuthn, Schnorr). Using CKBers' Codebase (CCC), a TypeScript SDK, this retrospective case study extracts three recurring patterns from release history and exported APIs: a Transaction Composer that makes capacity-aware input completion and fee estimation declarative, a Unified Signer Abstraction that normalizes signing across heterogeneous wallets behind one interface, and a Connector UI component that ships per-wallet onboarding as a drop-in. The contribution is the articulation of these patterns as transferable design lessons for SDKs targeting UTXO-family ledgers 
    
[^37]: 真实世界网络环境中的深度搜索

    Deep Search in Real-World Web Environments

    [https://arxiv.org/abs/2610.30181](https://arxiv.org/abs/2610.30181)

    该论文提出了 OpenDS 任务及 OpenDS-World 基准，用于评估智能体在真实、开放的互联网环境中自主进行多步骤深度搜索、跨多个独立来源发现并整合证据，从而得出唯一事实性答案的能力。

    

    尽管深度搜索智能体已取得长足进步，但其在真实世界开放网络环境中进行端到端研究的能力仍未得到充分评估。与静态检索设置不同，真实网络中的信息分布在多样的来源、模态和异构格式之中，通常涉及复杂的交互机制，要求智能体在执行过程中自主地搜索、访问并整合相关证据。为填补这一空白，我们提出了 OpenDS 任务，旨在评估智能体在实时、开放的网络中进行多步骤深度搜索的能力。该任务要求智能体自主探索真实网络，通过网络搜索、浏览器交互和证据处理，从多个独立来源中发现并关联相关证据，并最终确定唯一的事实性答案。基于该任务设定，我们构建了 OpenDS-World，这是一个包含 20（原文摘要至此截断）……

    arXiv:2610.30181v1 Announce Type: new  Abstract: Despite substantial progress in Deep Search agents, their ability to conduct end-to-end research in real-world open Web environments remains insufficiently evaluated. Unlike static retrieval settings, information on the real-world Web is distributed across diverse sources, modalities, and heterogeneous formats, often involving complex interaction mechanisms that require agents to autonomously search for, access, and integrate relevant evidence during execution. To address this gap, we introduce the OpenDS task, designed to evaluate agents' ability to perform multi-step deep search in the live, open Web. This task requires agents to autonomously explore the real Web, discover and connect relevant evidence from multiple independent sources through Web search, browser interaction, and evidence processing, and ultimately determine a unique factual answer. Building on this task formulation, we construct OpenDS-World, a benchmark comprising 20
    
[^38]: AI智能体的MCP与CLI对比：自主系统工具集成策略的比较分析

    MCP vs CLI for AI Agents: A Comparative Analysis of Tool Integration Strategies for Autonomous Systems

    [https://arxiv.org/abs/2610.30157](https://arxiv.org/abs/2610.30157)

    本文通过14个维度的比较分析表明，CLI在单用户本地工作流中效率更高，而MCP能够满足CLI从根本上无法实现的细粒度访问控制、凭证隔离、结构化审计和多租户治理等需求，且渐进式披露技术可消除MCP的上下文窗口开销弱点。

    

    AI智能体需要接口来与外部系统（如文件、API、数据库和云服务）进行交互。目前出现了两种相互竞争的范式：CLI（命令行接口），即智能体直接执行shell命令；以及MCP（模型上下文协议），一种提供受管控的、类型化工具访问的结构化JSON-RPC协议。本文从14个评估维度进行了比较分析，辅以实例演示、基于既定合规框架的安全评估以及经济建模。我们的分析表明，CLI在单用户、仅本地的工作流程中提供更优的效率，而MCP则满足了CLI从根本上无法满足的需求，包括细粒度访问控制、凭证隔离、结构化审计和多租户治理。我们进一步研究了渐进式披露技术如何消除MCP的历史弱点（上下文窗口开销），从而带来token效率……

    arXiv:2610.30157v1 Announce Type: cross  Abstract: AI agents require interfaces to interact with external systems such as files, APIs, databases, and cloud services. Two competing paradigms have emerged: CLI (Command Line Interface), where agents execute shell commands directly, and MCP (Model Context Protocol), a structured JSON-RPC protocol providing governed, typed tool access. This paper presents a comparative analysis across 14 evaluation dimensions, supported by worked examples, security assessment against established compliance frameworks, and economic modeling. Our analysis indicates that CLI provides superior efficiency for single-user, local-only workflows, while MCP addresses requirements that CLI fundamentally cannot satisfy, including fine-grained access control, credential isolation, structured audit, and multi-tenant governance. We further examine how progressive disclosure techniques eliminate MCP's historical weakness (context window overhead), yielding token efficienc
    
[^39]: Code Comprehensor：面向大型代码仓库的层次化文档与跨模态检索

    Code Comprehensor: Hierarchical Documentation and Cross-Modal Retrieval for Large Repositories

    [https://arxiv.org/abs/2610.30016](https://arxiv.org/abs/2610.30016)

    该论文提出Code Comprehensor混合检索-推理框架，通过代码依赖调用图、层次化文档生成（CodeScribe）和CLIP式跨模态检索模型（CodeContrast），实现对大型代码库相关代码上下文的快速语义精准检索，并为基于MCP的LLM问答提供可靠依据。

    

    理解大型、复杂且快速演进的代码库仍然是规模化软件工程的核心挑战，它阻碍了新人上手、代码维护、重构和影响分析。将这些信息提供给大语言模型（LLM）是另一个障碍，因为LLM不仅需要理解所需的代码方法，还需要理解其在整个仓库全局结构中相关联的方法。我们提出了Code Comprehensor，这是一个混合检索-推理框架，结合了：（i）代码依赖调用图，（ii）CodeScribe——利用代码感知的LLM在仓库调用图上生成层次化文档，（iii）CodeContrast——一个CLIP风格的双编码器模型，基于CodeScribe的输出进行训练以学习对齐的文本-代码嵌入。CodeContrast能够实现快速、语义精准的相关代码上下文邻域检索，这些邻域作为基于MCP的LLM编排器回答领域特定查询时的可靠依据。

    arXiv:2610.30016v1 Announce Type: cross  Abstract: Understanding large, complex, and rapidly evolving code bases remains a core challenge for software engineering at scale, hindering onboarding, maintenance, refactoring, and impact analysis. Providing this information to a Large-Language Model (LLM) is another hurdle, as the LLM must understand not only the required code method but also its associated methods in the grand scheme of the repository. We propose Code Comprehensor, a hybrid retrieval-reasoning framework combining (i) a code dependency callgraph, (ii) CodeScribe, which uses code-aware LLMs to generate hierarchical documentation across repository call graphs, and (iii) CodeContrast, a CLIP-style dual-encoder model trained on CodeScribe output to learn aligned text-code embeddings. CodeContrast enables fast, semantically precise retrieval of relevant code-context neighborhoods, which serve as grounded evidence for an MCP-based LLM orchestrator answering domain-specific queries
    
[^40]: Harness演化中的组合式安全失效：识别与运行时监控

    Compositional Safety Failures in Harness Evolution: Identification and Runtime Monitoring

    [https://arxiv.org/abs/2609.33123](https://arxiv.org/abs/2609.33123)

    该研究首次系统揭示了Harness演化中的组合式安全失效——即各自安全且不损害效用的组件更新在交互后可能引发不安全的智能体行为——在三个安全基准上识别出43个两两组合和18个不可约的三路组合安全失效，并提出运行时监控方法以应对跨组件安全验证的组合复杂度难题。

    

    自进化的智能体Harness会持续更新记忆、提示词、技能和工具等持久化组件，我们将这一过程称为Harness演化。然而，这种演化可能引入意想不到的安全风险。现有工作研究Harness的错误演化，并对候选Harness进行验证或将问题归因于单个组件的更新，而对跨组件更新交互的安全分析在很大程度上尚未被研究。为填补这一空白，我们研究了Harness演化中的组合式安全失效：即各个组件更新单独来看是安全且保持效用的，但它们之间的交互可能产生不良或不安全的智能体行为，这揭示了Harness演化中固有的安全风险。在三个与安全相关的基准测试中，我们识别出43个两两组合和18个不可约的三路组合安全失效。传统解决方案在验证跨组件交互时会带来组合爆炸式的复杂度，使得安全检查难以实际执行。（注：原文摘要在此处被截断）

    arXiv:2609.33123v2 Announce Type: replace  Abstract: Self-evolving agent harnesses continually update persistent components such as memory, prompts, skills, and tools. We call this process harness evolution. However, such evolution could introduce unexpected safety risks. Existing work studies harness misevolution and validates candidate harnesses or attributed individual component updates, leaving safety analysis of cross-component update interactions largely unexamined. To address this gap, we study compositional safety failures in harness evolution, where interactions among individually safe and utility-preserving component updates can produce undesirable or unsafe agent behavior, revealing a safety risk intrinsic to harness evolution. Across three safety-related benchmarks, we identify 43 pairwise and 18 irreducible 3-way compositional safety failures. Conventional solution incurs combinatorial complexity in validating cross-component interactions, leaving the safety checking impra
    
[^41]: 办公室规模验证搜索中的算子包、提议者强度与构造型家族平台期

    Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search

    [https://arxiv.org/abs/2609.29636](https://arxiv.org/abs/2609.29636)

    该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。

    

    验证搜索是指语言模型提出程序、硬评估器对其进行评分、选择机制保留最优解的过程，这种方法近来已推动了数学纪录的进展；但对提议者侧组件的受控消融实验仍然罕见。我们在办公规模上（笔记本电脑上运行的30B本地模型，每次运行120-600个验证样本）对最小化的FunSearch风格循环进行了仪器化，采用三种算子包：模型自行编写并携带的示意图式笔记本（代替逐字复制的精英解）、命名障碍、以及对已发现构造的行为排斥。在来自公共仓库的九个构造问题上，带两次重复的完整2³因子设计在名义两阶段分析中支持主要对比：该组合缩小了更多从种子解到纪录的差距（+0.196；名义合并p=0.023，阶段组合p≈0.08；每问题效应中位数为+0.045）。排斥机制在各处都提高了构造哈希多样性（p=0.0039；部分属于操纵检查）（注：原文摘要至此处截断）。

    arXiv:2609.29636v1 Announce Type: cross  Abstract: Verified search, in which a language model proposes programs, a hard evaluator scores them, and selection keeps the best, has recently moved mathematical records; controlled ablations of the proposer-side components remain rare. We instrument a minimal FunSearch-style loop at office scale (a 30B local model on a laptop, 120-600 verified samples per run) with three operator packages: a schematic notebook the model writes and carries instead of verbatim elites, a named obstacle, and behavioural repulsion from constructions already found. On nine construction problems from a public repository, the complete 2^3 factorial with two replicates favours the primary contrast in a nominal two-stage analysis: the composition closes more of the seed-to-record gap (+0.196; nominal pooled p=0.023, stage-combination p~0.08; median per-problem effect +0.045). Repulsion raises construction-hash diversity everywhere (p=0.0039; partly a manipulation check
    
[^42]: 扩展FunctionGemma以实现实用的设备端移动函数调用

    Extending FunctionGemma for Practical On-Device Mobile Function Calling

    [https://arxiv.org/abs/2609.25373](https://arxiv.org/abs/2609.25373)

    该研究通过构建涵盖十五类设备控制操作的约9,500个对话的合成数据集MOBILEACTIONSEXTENDED对FunctionGemma 270M进行微调，将设备端Android函数调用的端到端准确率从29.3%大幅提升至76.5%，并使组合模型在Google数据集上达到82.3%。

    

    设备端助手需要能够将自然语言映射为本地系统操作的函数调用模型，但现有资源主要侧重于Web API或狭窄的移动操作目录。我们通过引入MOBILEACTIONSEXTENDED数据集，将FunctionGemma 270M-it扩展到实用的Android工作流程中。该数据集是一个合成的、经过模式校验的数据集，包含约9,500个对话，涵盖十五个设备控制类别，包括消息发送、电话呼叫、相机/截图、亮度控制、设备状态查询、手电筒控制以及应用程序管理。我们使用TRL监督微调方法（仅对完成部分计算损失）对270M模型进行微调，得到一个扩展的专用模型，以及一个与Google的MOBILEACTIONSGOOGLE联合训练的组合模型。在MOBILEACTIONSEXTENDED上，端到端准确率从基础模型的29.3%、Google Mobile-Actions变体的17.2%提升至76.5%。组合模型在MOBILEACTIONSEXTENDED上保持76.5%的准确率，并在MOBILEA（原文截断，应为Google数据集）上达到82.3%。

    arXiv:2609.25373v1 Announce Type: new  Abstract: On-device assistants require function-calling models that map natural language to local system actions, but existing resources emphasize web APIs or narrow mobile-action catalogs. We extend FunctionGemma 270M-it to practical Android workflows by introducing MOBILEACTIONSEXTENDED, a synthetic, schema-validated dataset of ~9,500 conversations covering fifteen device-control categories, including messaging, phone calls, camera/screenshot, brightness control, device-status queries, flashlight control, and application management. We fine-tune the 270M model with TRL supervised fine-tuning under completion-only loss, producing an extended specialist and a combined model trained jointly with Google's MOBILEACTIONSGOOGLE. On MOBILEACTIONSEXTENDED, end-to-end accuracy improves from 29.3% for the base model and 17.2% for Google's Mobile-Actions variant to 76.5%. The combined model retains 76.5% on MOBILEACTIONSEXTENDED and reaches 82.3% on MOBILEA
    
[^43]: AuditCoder：通过责任保留任务图实现过程可审计的代码生成

    AuditCoder: Process-Auditable Code Generation through Responsibility-Preserving Task Graphs

    [https://arxiv.org/abs/2607.29529](https://arxiv.org/abs/2607.29529)

    提出了多智能体系统AuditCoder，通过在实现前为任务图中的节点分配稳定ID并保留代码归属关系，将每段代码与其规划承诺、检查结果和修复历史关联起来，从而实现过程可审计的代码生成。

    

    计划可以指导代码生成，但不会记录哪段代码实现了哪个子问题，导致审查者在交接和维护时需要重新构建这些关联。我们提出了AUDITCODER，一个面向过程可审计代码生成的多智能体系统，它同时构建程序及其责任记录。带有契约标注的任务图在实现之前分配稳定的ID，并在生成和组装过程中保留代码的归属关系。这些相同的ID将每个规划承诺与其对应的代码、带版本记录的检查结果以及修复历史连接起来，支持人工检查以及针对记录完整性和修复决策的自动化验证。当失败证据指向某个节点或分支时，有界修复会声明一个可编辑的范围，记录任何范围扩展，并保持最终范围之外的代码不变。当没有边界得到证据支持时，系统会选择弃权。在包含200个任务的APPS样本和ClassEval基准上，AUDITCODER保持与现有方法的竞争力（原文在此处截断）。

    arXiv:2607.29529v2 Announce Type: replace  Abstract: Plans can guide code generation without record?ing which code implements each subproblem, leaving reviewers to reconstruct these links dur?ing handoff and maintenance. We present AU?DITCODER, a multi-agent system for process?auditable code generation that jointly con?structs a program and its responsibility record. A contract-annotated task graph assigns sta?ble IDs before implementation and preserves code ownership through generation and assem?bly. The same IDs connect each planning commitment to its code, versioned check results, and repair history, supporting human inspec?tion and automated checks of record complete?ness and repair decisions. When failure evidence supports a node or branch, bounded repair declares an editable scope, records any expansion, and leaves code outside the final scope unchanged. The system abstains when no boundary is supported. On a 200-task APPS sample and ClassEval, AUDITCODER remains competitive with
    
[^44]: 无边缘，无判定：对78K个真实世界SBOM中声明依赖图的大规模实证研究

    No Edges, No Verdict: A Large-Scale Empirical Study of Declared Dependency Graphs in 78K SBOMs in the Wild

    [https://arxiv.org/abs/2607.22140](https://arxiv.org/abs/2607.22140)

    这项研究首次大规模分析了78,612个真实SBOM中的依赖图，发现超过半数未声明任何边缘，且边缘生成主要由生成器决定而非软件本身，这对漏洞分析有重要影响。

    

    软件物料清单（SBOM）不仅作为组件清单被使用，还作为依赖图：漏洞分类、可达性过滤和影响分析都遍历SBOM声明的边缘。我们对来自Wild SBOMs数据集（77,092个可解析）的78,612个真实世界SBOM文件中的声明依赖图进行了首次大规模特征化分析。我们发现总体分为三个区间：52.9%的SBOM未声明任何边缘（不符合NTIA关于依赖关系的最低要素要求），8.8%声明了依赖块但使大多数组件孤立（退化区间；在此类至少含50个组件的SBOM中，中位孤立比例为93%，我们生成的11个Syft容器镜像SBOM也属于此区间，孤立比例为91.4-98.0%），38.3%形成了良好连接的图。边缘生成由生成器决定，而非所描述的软件（0%-100%无边缘）。

    arXiv:2607.22140v2 Announce Type: replace  Abstract: Software Bills of Materials (SBOMs) are consumed not only as component inventories but as dependency graphs: vulnerability triage, reachability filtering, and impact analysis all traverse the edges an SBOM declares. We present the first large-scale characterization of the declared dependency graph across 78,612 real-world SBOM files from the Wild SBOMs dataset (77,092 parseable). We find that the population splits into three regimes: 52.9% of SBOMs declare no edges at all (failing the NTIA minimum-elements requirement of dependency relationships), 8.8% declare a dependency block yet leave the majority of components isolated (degenerate regime; among such SBOMs with at least 50 components the median orphan share is 93%, and our 11 Syft-generated container-image SBOMs fall in this regime at 91.4-98.0% orphans), and 38.3% form well-connected graphs. Edge emission is determined by the generator, not the described software (0%-100% no-edg
    
[^45]: PosterHarness：将科学海报生成转变为可审计的指令遵循基准

    PosterHarness: Turning Scientific Poster Generation into an Auditable Instruction-Following Benchmark

    [https://arxiv.org/abs/2607.03006](https://arxiv.org/abs/2607.03006)

    该论文提出POSTERHARNESS，一个可审计的基准框架，通过“占位符优先”契约将科学海报生成转化为可度量的指令遵循任务，要求模型只负责视觉设计而绝不伪造数据图表，从而能够客观检测占位符准确性、宽高比合规性与拒绝合成图形等关键属性。

    

    丰富文本的图像模型如今已能设计海报规模的版面，但我们缺乏衡量它们是否遵守科学传播契约的方法：清晰可读的标签、规定的宽高比，以及——最重要的是——避免伪造科学图表。我们提出了POSTERHARNESS，这是一个可审计的评测框架，将海报生成重新定义为可度量的指令遵循任务，并附带一个试点基准和失败分类法。POSTERHARNESS采用“占位符优先”契约来分离模型通常混淆的两项任务：模型只执行视觉摘要设计——排版、阅读路径、配色和背景——但绝不绘制承载数据的图表。每个图表区域必须是带标签的空白占位符；由确定性合成器在检测到的坐标处插入真实的源论文图表。这使得各项属性可被度量：占位符数量与ID准确性、空白性、宽高比合规性、对合成图形的拒绝，以及公共文本……

    arXiv:2607.03006v2 Announce Type: replace-cross  Abstract: Text-rich image models can now design poster-scale layouts, but we lack ways to measure whether they honor scientific communication contracts: legible labels, prescribed aspect ratios, and -- above all -- abstaining from fabricated scientific figures. We present POSTERHARNESS, an auditable harness reframing poster generation as measurable instruction-following tasks, with a pilot benchmark and failure taxonomy. POSTERHARNESS uses a placeholder-first contract to separate two jobs models otherwise conflate. The model performs visual-summary design: typography, reading path, color, and background -- but never draws data-bearing figures. Every figure region must be an empty labeled placeholder; a deterministic compositor inserts real source-paper figures at detected coordinates. This makes properties measurable: placeholder count and ID accuracy, blankness, aspect-ratio compliance, abstention from synthesized graphics, public-text 
    
[^46]: 对LLVM优化流水线的多维度逐遍（Per-Pass）实证研究

    A Multi-Dimensional, Per-Pass Empirical Study of the LLVM Optimization Pipeline

    [https://arxiv.org/abs/2606.31238](https://arxiv.org/abs/2606.31238)

    通过将LLVM的-O3优化流水线分解为113个累积遍前缀、在30个PolyBench/C内核上进行84,750次多维测量，该研究利用字节级相同的二进制构建零分布来校准显著性，首次系统揭示了该流水线是非单调的——36%显著改变运行时间的优化遍转换实际上是性能回退。

    

    arXiv:2606.31238v2 公告类型：replace 摘要：量化单个优化遍的边际影响是阶段排序、遍选择、优化设计以及遍/硬件交互分析的基础。在LLVM（C/C++、Rust以及通过MLIR支持的机器学习技术栈的标准后端）中，遍之间的相互作用、测量噪声和流水线规模使得这项工作变得困难。我们对LLVM的-O3流水线进行了系统的实证研究。我们将其分解为113个累积的逐遍前缀，并在30个PolyBench/C内核上测量执行时间、编译时间、二进制大小、硬件计数器和RAPL能耗，共计84,750次测量。二进制文件与其前驱逐字节相同的转换为我们提供了内置的零分布，我们用它来校准每一个显著性声明。在这些计算受限的仿射内核上，该流水线是非单调的：在显著改变运行时间的转换中，36%是性能回退，且出现在30个内核中的28个里。收益是后置的（中位数……（摘要原文在此处截断）

    arXiv:2606.31238v2 Announce Type: replace  Abstract: Quantifying the marginal impact of individual optimization passes underpins phase ordering, pass selection, optimization design, and analysis of pass/hardware interactions. In LLVM, the standard backend for C/C++, Rust, and ML stacks via MLIR, interactions among passes, measurement noise, and pipeline scale make this difficult. We present a systematic empirical study of the LLVM -O3 pipeline. We decompose it into 113 cumulative per-pass prefixes and measure execution time, compile time, binary size, hardware counters, and RAPL energy on 30 PolyBench/C kernels (84,750 measurements). Transitions whose binary is byte-identical to their predecessor give us a built-in null distribution, which we use to calibrate every significance claim. On these compute-bound affine kernels, the pipeline is non-monotone: 36% of the transitions that significantly change runtime are regressions, found in 28 of 30 kernels. Gains are back-loaded (the median 
    
[^47]: APO-DD：通过多组件提示词的反思性优化实现对话解缠

    APO-DD: Dialogue Disentanglement via Reflective Optimization of Multi-Component Prompts

    [https://arxiv.org/abs/2606.07894](https://arxiv.org/abs/2606.07894)

    APO-DD通过改编GEPA框架，利用执行轨迹和错误决策反思自动优化任务指令与聊天表示两个组件，其优化后的提示词在对话解缠任务上显著超越手动优化的提示词，且在其他聊天领域无需进一步优化即可保持优势。

    

    在多方聊天中，相互交织的对话使得识别每条话语的上下文变得困难。对话解缠将这些对话分离开来，以支持对话分析和回复生成。提示词优化可以提高大型语言模型（LLM）在此任务上的准确性，但为每个LLM手动寻找有效提示词需要大量的反复试验。我们提出了APO-DD，它改编了GEPA方法，利用执行轨迹和对错误决策的反思来优化提示词。该方法直接更新任务指令，并通过自动生成的格式化函数更新聊天表示。我们比较了单独优化任一组件或同时优化两者的情况。使用Qwen3.5-9B模型，优化后的最佳提示词在Ubuntu IRC数据集的七个评估指标上平均得分为68.7，而最佳手动优化提示词的得分仅为57.5。在其他聊天领域中，它无需进一步优化即可超越手动优化的提示词。

    arXiv:2606.07894v2 Announce Type: replace  Abstract: In multi-party chat, interleaved dialogues make it difficult to identify the context of each utterance. Dialogue disentanglement separates these dialogues to support dialogue analysis and response generation. Prompt refinement can improve large language models' (LLMs') accuracy on this task, but manually finding effective prompts for each LLM requires extensive trial and error. We propose APO-DD, which adapts GEPA to optimize prompts using execution traces and reflections on failed decisions. It updates task instructions directly and chat representations through automatically generated formatting functions. We compare optimizing either component alone or both. With Qwen3.5-9B, the best optimized prompt achieves an average score of 68.7 across seven evaluation metrics on Ubuntu IRC, compared with 57.5 for the best manually refined prompt. It also outperforms manually refined prompts on other chat domains without further optimization. 
    
[^48]: 测量权限门控：对 Claude Code 自动模式的压力测试评估

    Measuring the Permission Gate: A Stress-Test Evaluation of Claude Code's Auto Mode

    [https://arxiv.org/abs/2604.04978](https://arxiv.org/abs/2604.04978)

    该论文首次对 Claude Code 自动模式权限系统进行独立压力测试评估，通过自建的 AmPermBench 基准发现在模糊授权场景下其端到端假阴性率高达 81.0%，远高于 Anthropic 在生产流量上报告的 17%。

    

    arXiv:2604.04978v3 公告类型：替换。摘要：Claude Code 的自动模式是首个已部署的面向 AI 编程代理的权限系统，采用两阶段对话记录分类器来拦截危险的工具调用。Anthropic 报告该系统在生产流量上的假阳性率为 0.4%，假阴性率为 17%。我们对该系统在故意设置的模糊授权场景上进行了首次独立评估，即用户意图明确、但目标范围、影响范围或风险级别未被充分指定的任务。使用 AmPermBench——一个涵盖四个 DevOps 任务家族和三个受控模糊性维度的包含 128 个提示词的基准——我们在单个操作层面、以基准真值为对照，评估了 253 个改变系统状态的操作。我们的发现刻画了自动模式在此压力测试工作负载下的范围升级覆盖情况。端到端假阴性率为 81.0%（95% 置信区间：73.8%–87.4%），远高于生产流量上报告的 17%，反映出一种根本性的……

    arXiv:2604.04978v3 Announce Type: replace  Abstract: Claude Code's auto mode is the first deployed permission system for AI coding agents, using a two-stage transcript classifier to gate dangerous tool calls. Anthropic reports a 0.4% false positive rate and 17% false negative rate on production traffic. We present the first independent evaluation of this system on deliberately ambiguous authorization scenarios, i.e., tasks where the user's intent is clear but the target scope, blast radius, or risk level is underspecified. Using AmPermBench, a 128-prompt benchmark spanning four DevOps task families and three controlled ambiguity axes, we evaluate 253 state-changing actions at the individual action level against oracle ground truth.   Our findings characterize auto mode's scope-escalation coverage under this stress-test workload. The end-to-end false negative rate is 81.0% (95% CI: 73.8%-87.4%), substantially higher than the 17% reported on production traffic, reflecting a fundamentally
    
[^49]: LIDL：基于知识图谱增强的多智能体分析实现LLM集成缺陷定位

    LIDL: LLM Integration Defect Localization via Knowledge Graph-Enhanced Multi-Agent Analysis

    [https://arxiv.org/abs/2601.05539](https://arxiv.org/abs/2601.05539)

    LIDL提出了一种基于知识图谱增强的多智能体分析框架，用于定位LLM集成软件中因提示词、API调用、配置和模型输出等异构制品交互失配而产生的新型集成缺陷，克服了传统缺陷定位技术无法捕获跨层依赖和缺乏语义推理能力的局限。

    

    与大型语言模型（LLM）集成或交互的软件，即将LLM作为功能组件嵌入其中的软件，表现出与传统软件根本不同的概率性和上下文相关行为。这种转变引入了一类新的集成缺陷，这些缺陷不仅源于代码错误，还源于LLM特定制品（包括提示词、API调用、配置和模型输出）之间的交互失配。然而，现有的缺陷定位技术在识别这些LLM特定集成缺陷方面效果不佳，因为它们无法捕获跨异构制品的跨层依赖关系，无法利用不完整或具有误导性的错误轨迹，且缺乏识别根本原因所需的语义推理能力。为应对这些挑战，我们提出了LIDL，一个用于LLM集成软件缺陷定位的多智能体框架。LIDL（1）构建了一个代码知识图……

    arXiv:2601.05539v2 Announce Type: replace  Abstract: LLM-integrated software, which embeds or interacts with large language models (LLMs) as functional components, exhibits probabilistic and context-dependent behaviors that fundamentally differ from those of traditional software. This shift introduces a new category of integration defects that arise not only from code errors but also from misaligned interactions among LLM-specific artifacts, including prompts, API calls, configurations, and model outputs. However, existing defect localization techniques are ineffective at identifying these LLM-specific integration defects because they fail to capture cross-layer dependencies across heterogeneous artifacts, cannot exploit incomplete or misleading error traces, and lack semantic reasoning capabilities for identifying root causes. To address these challenges, we propose LIDL, a multi-agent framework for defect localization in LLM-integrated software. LIDL (1) constructs a code knowledge g
    
[^50]: 万物模型：一种简化万物云应用开发的模型驱动方法

    Model of Things (MoT): A Model-Driven Approach to Streamline Cloud-of-Things Application Development

    [https://arxiv.org/abs/2512.14613](https://arxiv.org/abs/2512.14613)

    本文提出了万物模型（MoT），一种将基于UML的建模与低代码执行概念相结合的模型驱动方法，旨在降低技术门槛、简化万物云应用的开发。

    

    云计算与物联网的融合是现代可扩展、数据密集型系统的基础。万物云应用如今已普遍应用于医疗保健、智慧城市和工业自动化等领域。然而，其开发过程仍然碎片化且技术要求高。当前的CoT开发方法需要对异构平台、协议和部署技术具备大量专业知识，其提供的模型驱动支持有限、自动化程度弱，且与低代码原则的契合度不佳。因此，开发CoT应用仍然费时费力且容易出错。本研究旨在通过引入万物模型来简化CoT应用开发。MoT将基于UML的建模与低代码执行概念相结合，在保持自动化和部署可行性的同时降低技术门槛。我们设计了一个包含九个CoT特定构造型（stereotypes）的自定义UML profile……

    arXiv:2512.14613v2 Announce Type: replace  Abstract: The convergence of Cloud Computing and the Internet of Things (IoT) underpins modern, scalable, and data-intensive systems. Cloud-of-Things (CoT) applications are now pervasive across healthcare, smart cities, and industrial automation. Nevertheless, their development remains fragmented and technically demanding. Current CoT development approaches require substantial expertise in heterogeneous platforms, protocols, and deployment technologies. They offer limited model-driven support, weak automation, and poor alignment with low-code principles. As a result, developing CoT applications remains effort-intensive and error-prone. This study aims to simplify CoT application development by introducing the Model of Things (MoT). MoT integrates UML-based modeling with low-code execution concepts to reduce technical barriers while preserving automation and deployment feasibility. We designed a custom UML profile comprising nine CoT-specific s
    
[^51]: DynamiQ：释放并行模糊测试中动态任务分配的潜力

    DynamiQ: Unlocking the Potential of Dynamic Task Allocation in Parallel Fuzzing

    [https://arxiv.org/abs/2510.04469](https://arxiv.org/abs/2510.04469)

    DynamiQ利用程序调用图结构定义任务并基于运行时反馈动态调整任务分配，大幅减少了并行模糊测试中的冗余探索，在超过40 CPU年的评估中显著提升分支覆盖率并发现了9个未知漏洞。

    

    我们提出了DynamiQ，它是AFLTeam的全面完善且经过优化的后继版本，支持动态和自适应的并行模糊测试。与大多数将单个种子视为任务的现有方法不同，DynamiQ利用程序调用图的结构信息来定义任务，并通过运行时反馈持续优化任务分配。这种设计显著减少了冗余探索，并提升了大规模场景下的模糊测试效率。DynamiQ构建在最先进的LibAFL框架之上，在任务分配和任务感知模糊测试两方面都融入了多项实用优化。我们在Magma和FuzzBench的十八个目标上对DynamiQ进行了评估，总预算超过40个CPU年。它在十四个目标上取得了最高的平均分支覆盖率，相较于五个基线模糊测试器的平均相对提升幅度为8.37%至19.53%。在全部实验中，DynamiQ共发现了九个此前未知的漏洞。

    arXiv:2510.04469v3 Announce Type: replace  Abstract: We present DynamiQ, a full-fledged and optimized successor to AFLTeam that supports dynamic and adaptive parallel fuzzing. Unlike most existing approaches that treat individual seeds as tasks, DynamiQ leverages structural information from the program's call graph to define tasks and continuously refines task allocation using runtime feedback. This design significantly reduces redundant exploration and enhances fuzzing efficiency at scale. Built on top of the state-of-the-art LibAFL framework, DynamiQ incorporates several practical optimizations in both task allocation and task-aware fuzzing. We evaluate DynamiQ on eighteen targets from Magma and FuzzBench with a total budget of over 40 CPU years. It achieves the highest mean branch coverage on fourteen targets, with average relative gains ranging from 8.37% to 19.53% over five baseline fuzzers. Across our experiments, DynamiQ identifies nine previously unknown bugs.
    

