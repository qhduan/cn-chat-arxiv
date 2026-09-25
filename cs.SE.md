# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Requirement-Bound Verified Commissioning: A Frozen Four-Billion-Parameter Local Model as a Candidate Generator under an External Acceptance Layer with Verification and Release Authority](https://arxiv.org/abs/2609.30219) | 本文提出一种将候选生成与发布权限分离的验收协议——冻结的四十亿参数本地模型仅负责生成候选，计划只有在外部闸门依据密封文法推导出事实后才发布，实验中21个虚构计划全部被拒，且83次发布可在无模型调用的情况下复现。 |
| [^2] | [Jev in the Wild: A Data-Driven Analysis of the Jev Model's Functionality, Applications and Ecosystem](https://arxiv.org/abs/2609.30216) | 本文通过对GitHub上2,170个公开Jev项目的大规模数据驱动分析，揭示了Jev作为可复用决策组件在不同领域中的多样化应用模式及其快速增长的生态系统。 |
| [^3] | [Jev-Mobile: Jev as an Executor for Mobile GUI Agents](https://arxiv.org/abs/2609.30186) | Jev-Mobile提出低频VLM规划与高频轻量级执行的新范式，由快速的类型化决策模型Jev在单个VLM决策下连续执行多个GUI动作，在AndroidWorld上达到79%任务成功率的同时大幅降低延迟与推理成本。 |
| [^4] | [NEUROTESTGEN: Neuro-Symbolic Guided Test Generation with Large Language Models](https://arxiv.org/abs/2609.30178) | NEUROTESTGEN提出了一种将符号执行与大型语言模型相结合的神经符号混合方法，能够针对指定的代码覆盖率目标生成既满足精确路径条件又现实可执行的测试用例。 |
| [^5] | [Evaluating Agent Skills for Version-Specific Plugin Migration: A Retrospective Study](https://arxiv.org/abs/2609.30120) | 该研究回顾性评估了一款插件升级智能体技能，发现尽管其使平均奖励提升4.92分，但深入审查暴露出评分标准中偏袒任一方的判分错误（如接受错误目录的谓词仍获满分），表明较高的诊断分数并不能保证迁移建议真正满足目标版本的契约。 |
| [^6] | [Era by Eon: Benchmarking Enterprise Agents on Hidden Knowledge](https://arxiv.org/abs/2609.30055) | 本文通过向Era by Eon基准引入八个依赖隐藏事实的问题模板——这些事实未被任何问题或文档直接陈述、只能从其他数据中推断——有效区分了在可运行代码场景下表现趋同的企业智能体，其中最强智能体在24次作答中答对18次，而六个模型中有四个最多仅答对6次。 |
| [^7] | [Style, Not Self: Surface Cues Explain Zero-Shot Code Attribution by Large Language Models](https://arxiv.org/abs/2609.30048) | 研究发现大语言模型在零样本识别自己代码时的表现并非源于真正的“自我认知”，而是可以由代码长度等表层风格特征所解释，因此对模型评审自我偏袒与合谋风险的担忧可能被夸大了。 |
| [^8] | [A Lightweight Ethereum Voting Prototype for Hospital Ethics Committees with Receipt-Based Inclusion Verification](https://arxiv.org/abs/2609.29981) | 本文提出了一个面向医院伦理委员会的轻量级以太坊区块链投票原型，利用角色控制、状态检查和回执哈希实现公开审计与交易上链验证，但系统仅提供假名化而非匿名投票，且存在多种残余安全风险。 |
| [^9] | [Formal Model Construction Guided by Model-Based Proof Sketches](https://arxiv.org/abs/2609.29870) | 提出了以基于模型的证明草图为中心的自动形式化方法ProGS，克服了现有“生成-修复”范式中修复效果过度依赖反馈粒度和LLM修复能力、且可能破坏其他层级验证属性的局限。 |
| [^10] | [SWE-PolyVision: Benchmarking Cross-Image Abductive Reasoning for Repository-Level Software Engineering](https://arxiv.org/abs/2609.29754) | 提出 SWE-PolyVision 基准，通过 92 个包含多图像输入的真实软件工程任务，首次系统评估智能体将跨图像视觉证据与代码证据整合并进行经验证的仓库级修复的能力，发现视觉访问的效果因模型和任务而异。 |
| [^11] | [Between the Commits: Process, Error, and Claim Reliability in a Wholly AI-Authored Codebase](https://arxiv.org/abs/2609.29744) | 本研究首次构建并分析了完全由Claude AI编写（无任何人工代码或测试）的21,000行Python工具的完整开发历史数据集，配合代码溯源工具与三种分类体系，发现14.3%的AI代码生成事件含有真实错误、约四分之一至五分之一的AI交互响应包含事实性错误。 |
| [^12] | [DSpec2Test: Specification-Driven Test Generation in Dafny](https://arxiv.org/abs/2609.29713) | DSpec2Test 是一个面向 Dafny 的规约驱动测试生成工具，它基于 DNF 等价类划分、可选的边界值分析以及 Z3 SMT 求解器，从形式化规约（而非实现细节）自动合成满足规约约束的测试输入与预期输出。 |
| [^13] | [On the Factors in Quantum Software Quality](https://arxiv.org/abs/2609.29705) | 本文以McCall经典“因素-准则-度量”框架为基础，提出了一个量子软件基础质量模型，在经典质量因素之上引入了量子特有属性（如概率性测量结果、后端依赖行为和量子-经典混合工作流）的扩展。 |
| [^14] | [Object-to-Source Mapping under Optimization: What a Control-Flow Trace Can Show, and Where Source-Level Coverage Breaks](https://arxiv.org/abs/2609.29656) | 本文界定了控制流轨迹重建机器级执行与源代码级结构覆盖之间的边界，指出编译器优化（如条件融合、分支替换和映射信息退化）会导致轨迹在二进制层面完整、但条件级源代码覆盖证据仍无法归因的问题，并将轨迹完整性、源代码归因与覆盖满足度进行了系统区分。 |
| [^15] | [Automated Abstraction Refinement for Information Flow Security in Embedded Systems](https://arxiv.org/abs/2609.29645) | 本文提出一种自动抽象细化方法，基于状态依赖关系和检测到的潜在信息泄露来启发式地自动选择抽象级别，从而解决嵌入式系统信息流分析中精确性与开销难以兼顾的问题，摆脱了以往需要手动定义抽象级别的局限。 |
| [^16] | [Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search](https://arxiv.org/abs/2609.29636) | 该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。 |
| [^17] | [iCoder-27B: Recursive AI-Led Development of Frontier Industrial Coding Model](https://arxiv.org/abs/2609.29626) | 专家仅通过高密度、低频次的接口将目标、流程与权限编码为可复用研究技能，智能体即可自主选择实验、诊断结果并迭代训练策略，最终递归式开发出具备前沿竞争力的工业编程模型iCoder-27B。 |
| [^18] | [CodeGraph: Open-Taxonomy Knowledge Graph for Source Code with Wikidata Grounding](https://arxiv.org/abs/2609.29474) | 该论文提出CodeGraph流水线，利用代码专用大语言模型对源代码进行开放分类法语义标注，并通过三阶段实体链接过程将其锚定到Wikidata，从而构建出首个面向源代码的开放分类法知识图谱。 |
| [^19] | [Judgment-Centred Software Engineering Education: A Post-Hype Review and Framework for AI-Augmented Learning](https://arxiv.org/abs/2609.29473) | 本文通过对2023至2026年研究与实践的整合性综述提出一个以判断力为中心的软件工程教育框架，指出生成式AI虽能提升解释、反馈与练习的可及性，但学习成效取决于先验知识、脚手架支持和验证等条件，教育目标应是保留并评估人类的理解能力。 |
| [^20] | [SWE-Prometheus: Measuring Engineering Governance Improvements in Real-World Repositories](https://arxiv.org/abs/2609.29465) | 该论文提出了SWE-Prometheus基准，首次系统评估大语言模型编程智能体在开放式仓库工程治理任务中的能力，要求智能体自主识别风险、排序干预优先级并验证变更，通过六个治理维度和多重验证机制对十个模型进行了评测。 |
| [^21] | [Demystifying Agent Skills for Smart Contract Auditing: Design, Effectiveness, Behavioral Impact](https://arxiv.org/abs/2609.29454) | 本文首次系统研究了智能体技能在智能合约安全审计中的应用，通过收集并评估83个真实世界的审计技能，揭示了其设计特征、提升漏洞检测的有效性以及对智能体执行行为的影响。 |
| [^22] | [Large Language Models for Programming: Actually Fixing or Reimplementing Incorrect Code?](https://arxiv.org/abs/2609.29410) | 本研究基于 Codeforces 竞赛编程的真实提交数据，通过对比人工修复补丁与模型生成结果的相似性，评估大语言模型在修复缺陷代码时究竟是真正修复原始代码还是倾向于重新实现全新方案。 |
| [^23] | [The Last Human Gate: Forward Deployed Engineering for Governance Automation](https://arxiv.org/abs/2609.29345) | 该论文提出将数字治理关卡视为可执行契约的任务替代框架，推导出剩余工作阈值以解释为何自动化多数案例反而可能增加人力，并通过 DGF-Bench 基准（300 个合成项目、899 次运行）实证了前沿大模型可达到最高 94.98% 的严格关卡成功率。 |
| [^24] | [Model-Based Retargeting to Many-Core CPS: Simulink-to-OpenCL Workflow](https://arxiv.org/abs/2609.29311) | 本文提出一种保持工作流的重定向方法，通过自动将Simulink模型转换为OpenCL代码实现CPS应用向多核处理器的移植，并在Kalray MPPA Coolidge2上以轨迹规划器成功验证了该方法的可行性。 |
| [^25] | [On the Impact of Requirement Smells in LLM-Based Code Generation](https://arxiv.org/abs/2609.29208) | 需求文本中的“需求异味”会降低大语言模型生成代码的功能正确性，且异味密度越高，正确性越低。 |
| [^26] | [HistoRAG: A Citation-Grounded Question Answering Assistant for Teaching with Scanned Local History and Heritage Archives](https://arxiv.org/abs/2609.29184) | HistoRAG 通过视觉语言模型转录扫描档案，并融合混合文本索引、关系数据库与知识图谱三种存储及轻量级路由，为地方历史与遗产教学提供每个事实都可溯源到具体卷册页码的引文支撑问答。 |
| [^27] | [Orchestrating AI-Assisted Code Remediation: Socio-Technical Bottlenecks in a Large Industrial Repository](https://arxiv.org/abs/2609.29172) | 本研究通过对大型工业C++代码库开展为期15天的实地案例研究，揭示了大规模AI辅助代码修复在持续集成、代码审查和团队协调等方面所面临的社会技术瓶颈。 |
| [^28] | [VidTutorAssistant: Automating Responses to Programming Tutorial Questions](https://arxiv.org/abs/2609.29129) | VidTutorAssistant是一个基于检索增强生成（RAG）技术的网络平台，通过视频转录文本的语义检索和GPT-4自动生成答案，解决了YouTube编程教程视频评论区观众提问得不到及时回复的问题。 |
| [^29] | [Where Does Exactly-Once Live? Model, Harness, and Tool-Contract Effects on Duplicate Side Effects in LLM Agents](https://arxiv.org/abs/2609.29095) | 该论文提出确定性沙盒基准 LIMBO，研究 LLM 智能体的“恰好一次”副作用语义应由模型、智能体框架还是工具契约来保障，并发现答案取决于故障类型：当即时回读能够揭示实际结果时，由模型来决定。 |
| [^30] | [Design, development, and preliminary validity and reliability evidence of the Software Engineering Self-Efficacy Scale (SESES)](https://arxiv.org/abs/2609.29068) | 本研究设计开发了涵盖需求工程、团队协作、软件质量管理、软件设计与架构及敏捷方法论五个维度的软件工程自我效能量表（SESES），并通过对527名计算机专业本科生的试测提供了初步效度与信度证据。 |
| [^31] | [Human-AI Collaboration for Multi-Line Task Adjustment Using Local Large Language Models and a Digital Twin](https://arxiv.org/abs/2609.29061) | 本研究提出了一种集成本地大语言模型、数字孪生与人工决策的多产线任务调整系统，通过“提议-验证-决策”工作流将操作员意图转化为经过仿真验证的候选策略，并实现了从需求到决策的全流程可追溯。 |
| [^32] | [Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers for LLM-Driven Pentest Agents](https://arxiv.org/abs/2609.28940) | 本文提出用JEV和Laya这类轻量级非生成式的“系统一”校准分类器作为LLM驱动的自主渗透测试智能体的专用决策层，以降低误报、纠正严重程度虚高并减少计算浪费。 |
| [^33] | [Automatic Harness Evolution for Hardware Design Verification: Can LLMs Consolidate Gains Across Discovered Harnesses?](https://arxiv.org/abs/2609.28908) | 该研究在硬件设计验证任务中对固定语言模型进行自动测试框架演化，发现虽然演化显著提升了完成尝试和任务覆盖率，但这些收益难以跨任务巩固和保持，表明LLM目前尚无法可靠地整合测试框架的改进收益。 |
| [^34] | [Specification-Driven Benchmarking for Automated Program Repair From Static Corpora to Executable Specifications](https://arxiv.org/abs/2609.28896) | 该论文提出“规范驱动基准测试”新范式，以可执行规范定义自动程序修复基准并通过生成、验证与语料库管理组件动态实现，从而克服静态数据集实验控制有限、易受污染、无法按需重新生成等局限。 |
| [^35] | [pytest-gpu-proof: Enabling Cloud-CPU Continuous Integration for GPU Code with Local GPU Attestation](https://arxiv.org/abs/2609.28862) | 该论文提出开源pytest插件pytest-gpu-proof，通过在本地运行GPU测试并生成签名认证收据，将其集成到低成本的云端CPU持续集成工作流中，解决了云端GPU CI成本高昂导致的GPU代码测试严重不足问题。 |
| [^36] | [RECLAIM: Can Agents Reproduce the Claims of Machine Learning Papers?](https://arxiv.org/abs/2609.28850) | RECLAIM是一个基于100篇NeurIPS 2025论文的可重建基准测试，通过预先定义复现目标、成功标准和GPU预算，并按作者发布资源分为运行、重训练、重新实现三个难度级别，用独立语言模型依据日志评分，结果显示最好的AI智能体也只能分别复现41%、27%和15%的论文结果。 |
| [^37] | [Pack Iteration in Swift: Ordinary Control Flow for Variadic Generics](https://arxiv.org/abs/2609.28822) | Swift 6.0引入的参数包迭代特性允许开发者使用普通的for-in循环遍历可变参数泛型的参数包，取代了传统上C++等语言中依赖递归“剥离”元素的复杂专家级编程模式。 |
| [^38] | [Category-Based MLM: Unifying Powertypes with Superclasses](https://arxiv.org/abs/2609.28810) | 该论文提出基于类别的多层次建模方法，将幂类型非传递性的成员关系语义与超类的深度特征化能力统一起来，从而解决了现有MLM方法中instance-of关系与深度特征化之间的语义矛盾。 |
| [^39] | [Soundness Checking of Taint Flow Models](https://arxiv.org/abs/2609.28750) | 提出一种“猜测-验证”方法，利用LLM智能体生成方法的精确污点流模型，并通过符号算法结合轻量级静态分析递归验证该模型的可靠性。 |
| [^40] | [Towards a Platform for Mastering Personal Sovereignty](https://arxiv.org/abs/2609.28736) | 提出名为"MyVirtualME"的虚拟可信平台，代表个人与企业和行政机构交互，将服务、数据和权限以人为中心进行组织，帮助个人重获数字时代的主权。 |
| [^41] | [CONCURDEP: Event-Guided Analysis of Dependency Invalidation in CPython Concurrency](https://arxiv.org/abs/2609.28608) | CONCURDEP 提出一种源码级静态分析方法，通过事件感知的原生并发依赖图将运行时语义依赖与并发或重入事件关联，从而检测 CPython 移除 GIL 后原生代码中依赖失效引发的安全问题。 |
| [^42] | [Developing a Unified Verification and Validation Activity Standard at JPL](https://arxiv.org/abs/2609.28600) | 该论文通过与29名不同学科从业者开展以人为本的设计研讨会，为JPL开发了统一的验证与确认活动标准模式，以平台无关的SysML模型形式化，在分离五种验证方法的同时维护共同属性集，并在Jama平台中实施以平衡严谨性与敏捷性。 |
| [^43] | [Agent Approval Laundering: Transitive Effects Beyond the Approved Invocation](https://arxiv.org/abs/2609.28586) | 该论文首次系统分析了智能体批准接口中“审批记录只覆盖入口调用、遗漏工作流传递效应”的批准洗白问题，形式化了六类效应上的闭包绑定批准并证明仅凭记录的策略存在信息极限，同时提出了将批准对象与执行后证据绑定的 Approval-to-Action 安全基准。 |
| [^44] | [Time-Series Foundation Models That Understand Data Revisions](https://arxiv.org/abs/2609.28576) | 该论文提出修订感知的时间序列基础模型 VINTAGE-TS，通过区分观测时间与信息可用时间、预测首次发布值并使用联合预测分布，解决了使用修订后数据评估历史预测时可能产生的前视偏差问题。 |
| [^45] | [Change-Provenant Supervision: Governing Learned Artifacts Under Policy Change](https://arxiv.org/abs/2609.28574) | 论文提出“变更溯源监督”框架，将记录的依赖谱系与独立的当前契约重验证相分离，借助字节级重建和密封契约账本，能够在权限变更后拒绝那些输出与合规产物完全相同、但来源已过期的学习产物（如 LoRA 适配器）。 |
| [^46] | [Where Cyber Agents Struggle: Bottleneck Analysis of Multi-Stage LLM Agents](https://arxiv.org/abs/2609.28572) | 本文通过端到端诊断研究揭示多阶段LLM网络智能体在自主攻击中的瓶颈，发现仅靠成功率会掩盖低效与证据误判问题，并提出成本感知评分与LLM-as-a-Judge分析来系统识别规划缺陷。 |
| [^47] | [Pretraining and adapting a language model on a dependency-free stack: GPT-2 124M from random weights, reproduced against llm.c, and a clinical adapter for Qwen3-0.6B](https://arxiv.org/abs/2609.28568) | 该论文用零第三方依赖的Zig机器学习栈numbat从随机权重独立复现了GPT-2 124M的完整预训练生命周期，各项指标与llm.c参考实现高度吻合，从而厘清了“语言模型本身的知识”与“特定训练软件的知识”，并进一步演示了Qwen3-0.6B模型的临床问答适配。 |
| [^48] | [Who Is Behind the Harness? Fingerprinting LLMs through Agentic Behavior](https://arxiv.org/abs/2609.28559) | 提出了一种名为LIDAR的主动式黑盒指纹识别方法，通过编码智能体在运行时的决策与行动（如编辑后验证、故障恢复和规范-测试冲突处理等行为）来识别框架背后的大语言模型身份。 |
| [^49] | [Synthesizing Proofs Using Proof Sharding and Exploration](https://arxiv.org/abs/2609.28535) | 提出了自动化工具ProofSaX，通过将大型验证任务分片并独立探索每个分片的证明搜索空间，减少人工干预，从而实现分布式系统正确性证明的规模化自动合成。 |
| [^50] | [IaC-Guard-V: A Verification Framework for LLM-Generated Infrastructure-as-Code Repairs](https://arxiv.org/abs/2609.28488) | IaC-Guard-V是一个以验证为中心的框架，从语法有效性、目标问题解决、回归安全性和补丁最小性四个维度系统评估LLM生成的IaC修复，并基于70个真实Terraform和Kubernetes配置错误工件构建基准进行实验。 |
| [^51] | [Generative AI May Reinforce Social Biases in Software Engineering Education](https://arxiv.org/abs/2609.28483) | 本研究揭示了软件工程教师在利用生成式AI进行团队组建和教育材料生成时会无意中强化性别与国籍等社会偏见，例如女性更容易被分配到前端开发角色。 |
| [^52] | [Drafter: A Python Library for Full-Stack Web Development in CS1](https://arxiv.org/abs/2609.28481) | 本文提出了面向CS1入门课程的开源Python库Drafter，让学生能够使用纯函数和最少的样板代码开发全栈Web应用，无需学习HTML或模板语言，从而将Web开发融入计算机科学入门教学。 |
| [^53] | [Who Finishes the Job? A Study of Follow-Up Fixes and Commit Authorship on AI Coding Agent Pull Requests](https://arxiv.org/abs/2609.26847) | 本研究通过跟踪五个主流AI编码智能体的6,774个已合并PR并与5,044个人类PR对比，首次系统揭示了AI智能体拉取请求合并后需要多少后续修复、由谁（智能体还是人类）来完成这些收尾工作。 |
| [^54] | [Grow the Harness, Not the Context: From Strategy-Free Scaffolds to Reusable Specialist Agents](https://arxiv.org/abs/2609.26760) | 该论文提出 Growing Harness 训练范式，通过失败定位、联合修复与成功优先门控，把任务反馈中反复出现的控制逻辑自动沉淀为可复用的可执行代码，让智能体框架本身从任务交互中“生长”出来，而 LLM 只需专注于任务特定的语义推理。 |
| [^55] | [An Empirical Analysis of Cross-OS Portability Issues in Python Projects](https://arxiv.org/abs/2609.25531) | 该论文开展了首个针对Python跨操作系统可移植性问题的大规模实证研究，分析了2,042个开源仓库，构建了包含7个主要故障类别、24个子类别、15个诊断特征和4种系统性修复模式的全面分类体系。 |
| [^56] | [The Vocabulary of Flaky Tests in Swift](https://arxiv.org/abs/2609.25516) | 该研究首次针对Swift语言评估了基于词汇的机器学习方法来预测不稳定测试，通过从15个开源项目收集数据并训练五个分类器，证明随机森林模型（F1=0.86，MCC=0.75）能够利用测试词汇特征有效识别不稳定测试，并显著优于简单基线方法。 |
| [^57] | [Scanning the Harness: A Repository Study of Configuration Exposures in AI Coding Agents](https://arxiv.org/abs/2609.07360) | 本研究对3,171个公开GitHub仓库开展了首次大规模系统性分析，识别出AI编码智能体配置中的六类安全暴露问题，发现15.4%-17.9%的设置存在未固定MCP包版本、宽泛执行权限授予等供应链风险。 |
| [^58] | [Metrics That Write Themselves: Evolving an Evaluator from Its Own Blind Spots](https://arxiv.org/abs/2608.18744) | 本文提出EvalCEGAR方法，通过反例引导抽象细化自动演化评估指标，利用碰撞对（正确与错误答案评分相同）作为作者请求，从自身盲点中生成可解释的缺陷检测操作符池，解决了报告生成等场景中自动评分指标缺失的问题。 |
| [^59] | [Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection](https://arxiv.org/abs/2608.17965) | 本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。 |
| [^60] | [Self-Evolving Coding Agents](https://arxiv.org/abs/2608.03392) | 本文系统综述了自我进化编码代理领域，定义其概念并区分于传统代理，强调通过持久更新组件从交互中改进行为以应对动态软件开发环境。 |
| [^61] | [Chart-Supported or Model-Supplied? Examining MLLM-Generated Claims for Accessible Visualization](https://arxiv.org/abs/2607.25021) | 本研究发现，提供数据表、标题、替代文本等无障碍图表上下文比提供图像本身更能有效促使MLLM生成有依据的直接声明并提升数值一致性。 |
| [^62] | [Detecting Data Poisoning in Code Generation LLMs via Black-Box, Vulnerability-Oriented Scanning](https://arxiv.org/abs/2603.17174) | CodeScan是首个用于审计代码生成大语言模型的黑盒、面向特定漏洞的扫描框架，通过分析多次生成结果的结构相似性和迭代发散分析来有效检测诱发不安全代码生成的数据投毒攻击。 |
| [^63] | [MOOSEnger: A Simulation-Aware AI Agent Framework for the MOOSE Ecosystem](https://arxiv.org/abs/2603.04756) | MOOSEnger 是一个面向 MOOSE 生态系统的仿真感知 AI 智能体框架，通过集成知识检索、HIT 语法解析、验证诊断与求解器反馈的“生成—检查—修复—运行”工作流，克服了大语言模型一次性生成仿真输入文件时易因微小错误而执行失败、且执行成功也不代表科学正确的问题。 |
| [^64] | [QMon: Monitoring the Execution of Quantum Circuits with Mid-Circuit Measurement and Reset](https://arxiv.org/abs/2512.13422) | QMon是一种实用的量子电路监控方法，通过结合电路中间测量、重置操作和因果锥重放技术，在保持量子电路原有运行行为的前提下，对电路中选定位置的中间量子态信息进行监测与比较。 |
| [^65] | [Search-Based Software Engineering and AI Foundation Models: Current Landscape and Future Roadmap](https://arxiv.org/abs/2505.19625) | 本文提出一份研究路线图，系统梳理了基于搜索的软件工程（SBSE）与AI基础模型（如大语言模型）的现状，并从基础模型增强SBSE、SBSE改进基础模型及两者融合三个核心方面指明了未来研究方向。 |
| [^66] | [Context-Aware Trust Verification for Identity-Based Software Signing](https://arxiv.org/abs/2406.15596) | 提出DiVerify框架，通过自动化验证软件签名发生的上下文条件（而不仅是签名者身份），解决了凭据泄露时签名仍能通过验证的安全缺陷。 |

# 详细

[^1]: 需求约束的验证式调试投运：在具有验证与发布权限的外部验收层下，以冻结的四十亿参数本地模型作为候选生成器

    Requirement-Bound Verified Commissioning: A Frozen Four-Billion-Parameter Local Model as a Candidate Generator under an External Acceptance Layer with Verification and Release Authority

    [https://arxiv.org/abs/2609.30219](https://arxiv.org/abs/2609.30219)

    本文提出一种将候选生成与发布权限分离的验收协议——冻结的四十亿参数本地模型仅负责生成候选，计划只有在外部闸门依据密封文法推导出事实后才发布，实验中21个虚构计划全部被拒，且83次发布可在无模型调用的情况下复现。

    

    本文针对机电调试中的传感器坐标与极性绑定问题，开发了一种验收协议。候选生成与发布权限被相互分离。确定性解析器无法支持的需求会被路由至一个冻结的、四十亿参数的本地语言模型。只有当两项事实能够由外部闸门在密封文法下推导得出时，计划才会被发布。在符合条件时，会向黄金标准用户请求唯一的规范答案。该协议在基准构建之前即已固定的评判标准下进行了一次评估，共144个任务，由相互隔离、无法访问闸门、文法或实验计划的智能体上下文编写。本文确立了三项贡献。第一，候选生成与发布决策被分别测量：在22个被路由的不可回答任务中，有21个提交了虚构的就绪计划，且全部被拒绝；相同的83次发布在没有模型调用的情况下被复现。第二，未发生任何虚假……（摘要原文在此截断）

    arXiv:2609.30219v1 Announce Type: cross  Abstract: An acceptance protocol is developed for sensor-coordinate and polarity binding in mechatronic commissioning. Candidate generation is separated from release authority. Requirements unsupported by a deterministic parser are routed to a frozen local language model with four billion parameters. Plans are released only when both facts can be derived by an external gate under a sealed grammar. One canonical answer is requested from a gold-standard user when eligible. The protocol was evaluated once under a criterion fixed before benchmark construction, on 144 tasks written by isolated agent contexts without access to the gate, grammar, or experimental plan. Three contributions are established. First, candidate generation and release decisions were measured separately. Fabricated ready plans were committed on 21 of 22 routed unanswerable tasks, and all were rejected. The same 83 releases were reproduced without model calls. Second, no false r
    
[^2]: Jev的真实世界应用：对Jev模型功能、应用与生态系统的数据驱动分析

    Jev in the Wild: A Data-Driven Analysis of the Jev Model's Functionality, Applications and Ecosystem

    [https://arxiv.org/abs/2609.30216](https://arxiv.org/abs/2609.30216)

    本文通过对GitHub上2,170个公开Jev项目的大规模数据驱动分析，揭示了Jev作为可复用决策组件在不同领域中的多样化应用模式及其快速增长的生态系统。

    

    Jev是一个快速、低成本的决策模型，能够通过选项、二元判断和评分来回答自然语言问题。随着其公开生态系统的快速增长，目前尚不清楚Jev在各类应用中是如何被使用的，以及公众关注度与项目分布之间的关系。为回答这些问题，我们对截至2026年9月22日从GitHub收集的2,170个公开可用的Jev项目进行了大规模的数据驱动分析。我们发现Jev的公开生态系统在早期呈现快速增长态势，既体现为新项目的涌现，也体现为对现有代码库的集成。在不同领域中，项目将Jev用于多种决策目的，并组合使用其多种接口。属性判断和评分被广泛使用，而动作选择、内容过滤、模型与工具选择的使用情况则因领域而异。这些模式表明，Jev作为一个可复用的决策组件，其功能会随周边工作流的不同而变化。同时……

    arXiv:2609.30216v1 Announce Type: new  Abstract: Jev is a fast, low-cost decision model that answers natural-language questions with choices, binary judgments, and scores. As its public ecosystem grows rapidly, it remains unclear how Jev is used across applications and how public attention relates to project distribution. To answer these questions, we conduct a large-scale, data-driven analysis of 2,170 publicly available Jev projects collected from GitHub as of September 22, 2026. We find rapid early growth in Jev's public ecosystem, with both new projects and integration into existing repositories. Across diverse domains, projects use Jev for multiple decision purposes and combine its interfaces. Attribute judgment and scoring are widely used, while the use of action selection, content filtering, and model and tool selection varies across domains. These patterns suggest that Jev serves as a reusable decision component whose functionality varies with the surrounding workflow. Meanwhil
    
[^3]: Jev-Mobile：Jev作为移动GUI智能体的执行器

    Jev-Mobile: Jev as an Executor for Mobile GUI Agents

    [https://arxiv.org/abs/2609.30186](https://arxiv.org/abs/2609.30186)

    Jev-Mobile提出低频VLM规划与高频轻量级执行的新范式，由快速的类型化决策模型Jev在单个VLM决策下连续执行多个GUI动作，在AndroidWorld上达到79%任务成功率的同时大幅降低延迟与推理成本。

    

    视觉-语言模型（VLM）已成为自主移动GUI智能体的常见基础，但大多数现有系统在几乎每个交互步骤都依赖VLM进行规划和动作定位，导致显著的延迟和模型服务成本。我们提出了Jev-Mobile，它将这一范式转变为低频VLM规划与高频轻量级执行相结合：VLM负责指定局部目标，无障碍树定义结构化的可执行动作空间，而Jev作为一个快速的类型化决策模型，在该空间内反复选择动作。这种设计允许在单个VLM决策下执行多个GUI动作，在保持自适应交互的同时减少了昂贵的VLM推理。在完整的AndroidWorld任务套件上，Jev-Mobile实现了79%的任务成功率，相比之下SeeAct-V为78%，逐步VLM基线为84%。在成功的轨迹中，它将平均端到端执行时间降低了约3倍。

    arXiv:2609.30186v1 Announce Type: new  Abstract: Vision-language models (VLMs) have become a common foundation for autonomous mobile GUI agents, but most existing systems rely on the VLM for both planning and action grounding at nearly every interaction step, leading to substantial latency and model-serving cost. We introduce Jev-Mobile, which shifts this paradigm to low-frequency VLM planning and high-frequency lightweight execution: the VLM specifies local goals, the accessibility tree defines a structured executable action space, and Jev, a fast typed decision model, repeatedly selects actions within this space. This design allows multiple GUI actions to be executed under a single VLM decision, reducing expensive VLM inference while preserving adaptive interaction. On the full AndroidWorld task suite, Jev-Mobile achieves 79% task success, compared with 78% for SeeAct-V and 84% for a Step-wise VLM baseline. Among successful trajectories, it reduces mean end-to-end execution time by 3
    
[^4]: NEUROTESTGEN：基于神经符号引导的大语言模型测试生成

    NEUROTESTGEN: Neuro-Symbolic Guided Test Generation with Large Language Models

    [https://arxiv.org/abs/2609.30178](https://arxiv.org/abs/2609.30178)

    NEUROTESTGEN提出了一种将符号执行与大型语言模型相结合的神经符号混合方法，能够针对指定的代码覆盖率目标生成既满足精确路径条件又现实可执行的测试用例。

    

    确保高结构覆盖率仍然是自动化测试生成中的一个根本性挑战，特别是对于复杂软件系统而言，到达特定的行或分支需要满足复杂的控制流和数据流约束。大型语言模型（LLMs）最近在生成类人测试用例方面展现出了强大的能力；然而，它们往往难以生成满足精确路径条件的输入。相反，符号执行可以系统地推导出这些约束，但它往往无法构建现实的、可执行的测试用例，并且受到可扩展性方面的限制。在本文中，我们提出了NEUROTESTGEN，这是一种混合方法，它将符号执行与LLM驱动的测试合成相结合，以生成针对按需代码覆盖目标的测试用例。给定一个方法中的一组目标语句，NEUROTESTGEN首先使用符号分析引擎（即Z3 SMT求解器）来……

    arXiv:2609.30178v1 Announce Type: new  Abstract: Ensuring high structural coverage remains a fundamental challenge in automated test generation, particularly for complex software systems where reaching specific lines or branches requires satisfying intricate control- and data-flow constraints. Large Language Models (LLMs) have recently demonstrated strong capabilities in producing human-like test cases; however, they often struggle to generate inputs that satisfy precise path conditions. Conversely, symbolic execution can systematically derive such constraints, but it often fails to construct realistic, executable test cases and is constrained by scalability limitations.   In this paper, we introduce NEUROTESTGEN, a hybrid approach that integrates symbolic execution with LLM-driven test synthesis to generate test cases targeting on-demand code coverage. Given a set of target statements within a method, NEUROTESTGEN first employs a symbolic analysis engine (i.e., the Z3 SMT solver) to e
    
[^5]: 评估智能体技能在特定版本插件迁移中的能力：一项回顾性研究

    Evaluating Agent Skills for Version-Specific Plugin Migration: A Retrospective Study

    [https://arxiv.org/abs/2609.30120](https://arxiv.org/abs/2609.30120)

    该研究回顾性评估了一款插件升级智能体技能，发现尽管其使平均奖励提升4.92分，但深入审查暴露出评分标准中偏袒任一方的判分错误（如接受错误目录的谓词仍获满分），表明较高的诊断分数并不能保证迁移建议真正满足目标版本的契约。

    

    智能体技能为编码智能体封装了针对特定版本的维护知识，但较高的诊断分数本身并不能表明由此产生的迁移建议满足目标版本的契约。我们通过一个包含64份报告的存档，对一款已发布的插件升级技能进行了回顾性研究，这些报告涉及16个静态迁移任务，每种条件下尝试两次，共328项判据决策。使用该技能后，平均记录奖励从93.83升至98.75，提升4.92分（95%任务自助法置信区间为[0.31, 10.86]）；该提升集中在一个任务上，且有八个任务对处于评分上限。将每项决策追溯到其契约领域并深入审查十份报告后，暴露了对任一实验组均有利的评分错误；其中一例中，一个接受父目录的包含判定谓词仍获得了满分。可执行探针证实了这一缺陷，并表明一个有效的teardown修复仅因更严格的生命周期评分标准而被排除。替换审查……（摘要原文在此处截断）

    arXiv:2609.30120v1 Announce Type: new  Abstract: Agent skills package version-specific maintenance knowledge for coding agents, but a higher diagnostic score does not by itself show that the resulting migration advice satisfies the target version's contract. We study a shipped plugin-upgrade skill through an archive of 64 reports on 16 static migration tasks, with two attempts per condition and 328 criterion decisions. With the skill, mean recorded reward rises from 93.83 to 98.75, a gain of 4.92 points (95% task-bootstrap interval [0.31, 10.86]); the gain is concentrated in one task, and eight task pairs are at the ceiling. Tracing every decision to its contract domain and reviewing ten reports in depth exposes grading errors that favor either arm; in one, a containment predicate that accepts the parent directory still receives full credit. Executable probes confirm this defect and show that a working teardown repair is excluded only by a narrower lifecycle rubric. Replacing the revie
    
[^6]: Era by Eon：基于隐藏知识的企业智能体基准测试

    Era by Eon: Benchmarking Enterprise Agents on Hidden Knowledge

    [https://arxiv.org/abs/2609.30055](https://arxiv.org/abs/2609.30055)

    本文通过向Era by Eon基准引入八个依赖隐藏事实的问题模板——这些事实未被任何问题或文档直接陈述、只能从其他数据中推断——有效区分了在可运行代码场景下表现趋同的企业智能体，其中最强智能体在24次作答中答对18次，而六个模型中有四个最多仅答对6次。

    

    在Era by Eon基准测试中，每个问题都说明了其答案的规则，代码根据一个生成的公司的数据计算出答案。当智能体可以运行代码时，四个最强的模型各自能答对27个此类问题中的22至25个，因此该基准几乎无法将它们区分开来。我们增加了八个依赖隐藏事实的问题模板。没有任何问题或文档直接陈述隐藏事实，看似包含该事实的记录显示的却是其他内容，隐藏事实由其他数据暗示得出。例如，销售系统记录某客户因时间原因放弃了一笔购买，但在一段通话录音中，客户却将原因归咎于服务中断。对于每个生成的公司，代码填充每个模板并在无需语言模型的情况下计算出精确答案。我们评估了12个智能体，每个智能体由一个模型与一个智能体程序配对，该程序将其连接到公司的各个系统。表现最好的智能体在其24次作答（每个问题3次）中正确回答了18次，六个模型中有四个在24次作答中最多仅答对6次（摘要在此处截断）。

    arXiv:2609.30055v1 Announce Type: cross  Abstract: In the Era by Eon benchmark, each question states the rules for its answer, and code computes the answer from a generated company's data. When agents can run code, the four strongest models each answer 22 to 25 of 27 such questions, so the benchmark barely separates them.   We add eight question templates that depend on hidden facts. No question or document states a hidden fact, and the records that seem to hold it show something else. Other data implies it. For example, the sales system says a customer dropped a purchase because of timing. On a recorded call, the customer blames an outage.   For each generated company, code fills each template and computes an exact answer without a language model. We evaluate 12 agents. Each pairs a model with an agent program, which connects it to the company's systems.   The best agent answers 18 of its 24 attempts, three per question, correctly. Four of the six models answer at most 6 of 24 with an
    
[^7]: 风格而非自我：表层线索解释大语言模型的零样本代码归因

    Style, Not Self: Surface Cues Explain Zero-Shot Code Attribution by Large Language Models

    [https://arxiv.org/abs/2609.30048](https://arxiv.org/abs/2609.30048)

    研究发现大语言模型在零样本识别自己代码时的表现并非源于真正的“自我认知”，而是可以由代码长度等表层风格特征所解释，因此对模型评审自我偏袒与合谋风险的担忧可能被夸大了。

    

    如果语言模型能够识别自己编写的代码，它可能会在充当评判者时偏袒该代码，而模型之间相互监控的情境可能导致合谋。我们在当前商业模型上对这种零样本能力进行了测试。五个大语言模型为MBPP、HumanEval和DS-1000生成解题方案，另有七个模型为MBPP生成方案，模型在四项任务中充当评估者：从一对方案中挑选出自己的方案、判断单个方案是否出自自己、识别两个方案中哪一个由指定模型编写，以及在盲测条件下评判代码质量。在单方案任务中，所有15个模型-基准组合的平衡准确率为49-58%，而原始准确率（38-67%）主要反映了模型宣称代码作者身份的难易程度。在成对任务中，14个评估者-对手组合的准确率与评估者自身方案更长这一因素的相关性高达r=0.93。对指定模型的归因在某些配对上取得成功，而在其他配对上则出现持续性的反转。一种基于规则的标准化方法……（摘要在此处被截断）

    arXiv:2609.30048v1 Announce Type: new  Abstract: If a language model can recognize code it wrote, it may favor that code as a judge, and instances of one model monitoring each other could collude. We test this zero-shot on current commercial models. Five LLMs generate solutions to MBPP, HumanEval, and DS-1000, seven more to MBPP, and models act as evaluators in four tasks: picking their own solution from a pair, judging whether a single solution is their own, identifying which of two solutions a named model wrote, and judging quality blind. In the single-solution task, balanced accuracy is 49-58% for all 15 model-benchmark combinations, while raw accuracy (38-67%) mostly reflects how readily a model claims authorship. In the pairwise task, accuracy across 14 evaluator-opponent combinations correlates at r=0.93 with how often the evaluator's solution is longer. Attribution to a named model succeeds on some pairs and is consistently inverted on others. A rule-based normalization that str
    
[^8]: 一种面向医院伦理委员会、支持基于回执的包含性验证的轻量级以太坊投票原型

    A Lightweight Ethereum Voting Prototype for Hospital Ethics Committees with Receipt-Based Inclusion Verification

    [https://arxiv.org/abs/2609.29981](https://arxiv.org/abs/2609.29981)

    本文提出了一个面向医院伦理委员会的轻量级以太坊区块链投票原型，利用角色控制、状态检查和回执哈希实现公开审计与交易上链验证，但系统仅提供假名化而非匿名投票，且存在多种残余安全风险。

    

    本文提出了一个基于 Solidity、Hardhat、React、MetaMask 和 ethers.js 的医院伦理委员会投票原型。该系统通过角色权限控制、病例状态检查、重复投票控制以及回执哈希，支持公开审计与交易上链包含性验证。由于投票事件会暴露钱包地址和投票内容，该设计提供的是假名化的可审计性，而非匿名或无记名投票；其回执既非无回执设计，也不具备抗胁迫性。评估报告显示 22 项功能测试全部通过，并给出了本地 Hardhat 环境的 Gas 消耗数据，其中每次投票消耗 284,137 Gas。一项 12 名参与者的模拟使用了假设概率，并非人类受试者实验证据。残余风险包括多钱包使用、管理员或前端被攻破、凭证重新分配、抢先交易、拒绝服务攻击以及未经测试的对抗路径。若要实现机密部署，则需要受治理的注册流程、加密选票、独立审计、对抗性测试以及重新（原文截断）……

    arXiv:2609.29981v1 Announce Type: cross  Abstract: This paper presents a Solidity, Hardhat, React, MetaMask, and ethers.js prototype for hospital ethics committee voting. Role controls, case-state checks, duplicate vote controls, and a receipt hash support public audit and transaction inclusion verification. Because vote events expose wallet addresses and vote values, the design provides pseudonymous auditability, not anonymous or secret-ballot voting; the receipt is neither receipt-free nor coercion-resistant. Evaluation reports 22 passing functional tests and local Hardhat gas use, including 284,137 gas per vote. A 12-participant simulation used assumed probabilities and is not human-subject evidence. Residual risks include multiple wallets, administrator or frontend compromise, credential reassignment, front-running, denial of service, and untested adversarial paths. Confidential deployment requires governed enrollment, encrypted ballots, independent audit, adversarial testing, repr
    
[^9]: 基于模型证明草图引导的形式化模型构建

    Formal Model Construction Guided by Model-Based Proof Sketches

    [https://arxiv.org/abs/2609.29870](https://arxiv.org/abs/2609.29870)

    提出了以基于模型的证明草图为中心的自动形式化方法ProGS，克服了现有“生成-修复”范式中修复效果过度依赖反馈粒度和LLM修复能力、且可能破坏其他层级验证属性的局限。

    

    形式化建模为系统正确性提供了强有力的保证，但开发和修复形式化模型仍然是一项劳动密集型的工作，需要大量的逻辑与形式化推理专业知识。近年来，基于大语言模型（LLM）的自动形式化智能体试图通过生成候选形式化模型并利用形式化工具的反馈对其进行修订来减轻这一负担。然而，现有方法遵循“生成-修复”范式，其中修复由所生成模型的验证失败驱动，因此严重依赖于反馈的粒度以及LLM自身的修复能力。其结果是，针对某一层级验证的修复可能会使另一层级的属性失效，这就需要对完整的事件防护条件集合进行推理。为了解决这些局限性，我们提出了证明草图引导的形式化模型合成方法（ProGS），这是一种以基于模型的证明草图为中心的自动形式化方法。

    arXiv:2609.29870v1 Announce Type: new  Abstract: Formal modeling provides strong guarantees about system correctness, but developing and repairing formal models remains labor-intensive and requires substantial expertise in logic and formal reasoning. Recent LLM-based autoformalization agents seek to reduce this burden by generating candidate formal models and revising them using feedback from formal tools. However, the existing approaches follow a generate-and-repair paradigm, in which repairs are driven by verification failures of the generated model and therefore depend heavily on both the granularity of the feedback and the LLM's repair capability. As a consequence, a repair targeting one level of verification may invalidate properties at another level, which requires reasoning over the complete set of event guards. To address these limitations, we propose Proof-Sketch-Guided Formal Model Synthesis (ProGS), an autoformalization method centered on model-based proof sketches. A model-
    
[^10]: SWE-PolyVision：面向仓库级软件工程的跨图像溯因推理基准测试

    SWE-PolyVision: Benchmarking Cross-Image Abductive Reasoning for Repository-Level Software Engineering

    [https://arxiv.org/abs/2609.29754](https://arxiv.org/abs/2609.29754)

    提出 SWE-PolyVision 基准，通过 92 个包含多图像输入的真实软件工程任务，首次系统评估智能体将跨图像视觉证据与代码证据整合并进行经验证的仓库级修复的能力，发现视觉访问的效果因模型和任务而异。

    

    当前多模态软件工程基准测试仅将图像作为额外的上下文呈现，但并未检验智能体能否将分布在多张图像中的证据整合到经过验证的仓库级修复中。我们提出 SWE-PolyVision，这是一个可执行的基准测试，包含来自 36 个开源组织的 92 个真实任务，其中 48 个为公开任务，44 个为私有保留任务。该发布版本包含 402 张静态图像和 6 个视频，每个任务至少有两个视觉输入。每个任务将固定的修复前仓库与隔离的验证器配对，并在三种访问模式（纯文本、原生视觉和工具介导视觉）中受支持的条件进行评估。在十一个代码模型上的实验表明，视觉访问会改变哪些任务能够被解决，但其效果同时取决于模型和任务。两个与推理轨迹关联的原生视觉案例展示了互补的视觉与文本线索如何引导实现源代码定位且经过验证的修复；受控干预实验表明，这种……（原文摘要在此处截断）

    arXiv:2609.29754v1 Announce Type: new  Abstract: Current multimodal software-engineering benchmarks expose images as additional context, but do not test whether an agent can integrate evidence distributed across images into a verified repository-level repair. We present SWE-PolyVision, an executable benchmark of 92 real tasks from 36 open-source organizations, with 48 public tasks and 44 private holdouts. The release contains 402 static images and 6 videos, with at least two visual inputs per task. Each task pairs a fixed pre-fix repository with an isolated verifier and is evaluated under the supported conditions among three access modes: Text-only, Native Vision, and Tool-mediated Vision. Across eleven coding models, visual access changes which tasks are solved, but effects depend on both model and task. Two trace-linked Native Vision cases illustrate how complementary visual and textual clues can lead to source-localized, verified repairs; controlled interventions show that this conv
    
[^11]: 提交之间：完全由AI编写的代码库中的开发过程、错误与声明可靠性

    Between the Commits: Process, Error, and Claim Reliability in a Wholly AI-Authored Codebase

    [https://arxiv.org/abs/2609.29744](https://arxiv.org/abs/2609.29744)

    本研究首次构建并分析了完全由Claude AI编写（无任何人工代码或测试）的21,000行Python工具的完整开发历史数据集，配合代码溯源工具与三种分类体系，发现14.3%的AI代码生成事件含有真实错误、约四分之一至五分之一的AI交互响应包含事实性错误。

    

    我们提出了：(i) 一个新数据集，包含一个完全由Claude AI构建的21,000行Python工具的完整开发历史，其中没有任何人工编写的代码或测试， 两个代码溯源追踪工具， 用于指令意图、提交溯源和响应可靠性的三种分类体系， 将这些工具应用于该数据集的分析。我们发现： 用户在编程代理CLI中的指令与IDE聊天中的指令在性质上不同，前者更侧重于理解、规划和咨询， 代码开发主要是主动进行的， 14.3%的AI代码生成事件包含真实错误，这些错误后来被AI编写的测试套件捕获， AI的交互式响应中大约每4到5个就有1个包含一个或多个事实性错误。

    arXiv:2609.29744v1 Announce Type: cross  Abstract: We present: (i) a new dataset consisting of the full development history of a 21,000-line Python tool built entirely by Claude AI, with no human-authored code or tests, (ii) two code-provenance tracing tools, (iii) three taxonomies for instruction intent, commit provenance, and response reliability, (iv) application of these to analyse the dataset. We find that: (i) user coding agent CLI instructions differ in kind from IDE-chat instructions, with a greater focus on comprehension, planning and consultation, (ii) code development is mainly proactive, (iii) 14.3% of AI code-generation events contain a real error later caught by the AI-authored test suite, (iv) roughly 1 in 4-5 of the AI's interactive responses contains one or more factual errors.
    
[^12]: DSpec2Test：Dafny 中基于规约驱动的测试生成

    DSpec2Test: Specification-Driven Test Generation in Dafny

    [https://arxiv.org/abs/2609.29713](https://arxiv.org/abs/2609.29713)

    DSpec2Test 是一个面向 Dafny 的规约驱动测试生成工具，它基于 DNF 等价类划分、可选的边界值分析以及 Z3 SMT 求解器，从形式化规约（而非实现细节）自动合成满足规约约束的测试输入与预期输出。

    

    Dafny 等验证感知语言将逻辑构造集成到代码中，能够自动验证程序的正确性。然而，在单独的验证无法覆盖的场景中（例如支持测试驱动开发），测试仍然很有用。现有的 Dafny 测试生成工具都是基于实现的，这限制了它们在此类场景中的适用性。我们提出了 DSpec2Test，一个面向 Dafny 的规约驱动测试生成工具，它能够从形式化规约自动推导测试，而无需考虑实现细节。我们的工具扩展了 Dafny 的 generate-tests 命令，新增了一种基于析取范式（DNF）等价类划分的黑盒模式，并支持可选的边界值分析（BVA）。DSpec2Test 依靠 Z3 SMT 求解器来合成满足规约推导约束的输入和预期输出。我们在使用 MutDafny 进行变异的 DafnyBench 程序上对 DSpec2Test 进行了评估。

    arXiv:2609.29713v1 Announce Type: new  Abstract: Verification-aware languages, such as Dafny, integrate logical constructs into code and enable automatic verification of program correctness. However, tests remain helpful in scenarios that verification alone does not address (e.g., to support test-driven development). Existing Dafny test generation tools are implementation-based, limiting their applicability in this context. We present DSpec2Test, a specification-driven test generation tool for Dafny that automatically derives tests from formal specifications, without considering implementation details. Our tool extends Dafny's generate-tests command with a new blackbox mode based on Disjunctive Normal Form (DNF) equivalence class partitioning and optional Boundary Value Analysis (BVA). DSpec2Test relies on the Z3 SMT solver to synthesize inputs and expected outputs that meet the specification-derived constraints. We evaluate DSpec2Test on programs from DafnyBench mutated using MutDafny
    
[^13]: 论量子软件质量的影响因素

    On the Factors in Quantum Software Quality

    [https://arxiv.org/abs/2609.29705](https://arxiv.org/abs/2609.29705)

    本文以McCall经典“因素-准则-度量”框架为基础，提出了一个量子软件基础质量模型，在经典质量因素之上引入了量子特有属性（如概率性测量结果、后端依赖行为和量子-经典混合工作流）的扩展。

    

    随着量子计算的发展，对高质量量子软件的需求日益增长。经典软件质量模型（如McCall模型）提供了有用的基础，但它们并未明确刻画量子软件的独特属性，包括概率性测量结果、依赖后端的行为以及量子-经典混合工作流。本文以McCall的“因素-准则-度量”结构作为组织骨架，提出了一个量子软件的基础质量模型。该模型借鉴代表性的经典软件质量模型和标准来扩展质量因素集合，并仅在必要之处引入量子特有的扩展。我们定义了由此得到的因素集合和质量准则，提供了核心因素与准则之间关系的多对多映射，并讨论了代表性的候选度量指标及其使用时所需的执行条件、参考条件和解释条件。

    arXiv:2609.29705v1 Announce Type: new  Abstract: As quantum computing evolves, the demand for high-quality quantum software is increasing. Classical software quality models, such as McCall's model, provide a useful foundation, but they do not explicitly capture distinctive properties of quantum software, including probabilistic measurement outcomes, backend-dependent behavior, and hybrid quantum-classical workflows. This paper proposes a base quality model for quantum software using McCall's factor-criterion-metric structure as the organizing backbone. It broadens the set of quality factors by drawing on representative classical software quality models and standards and introduces quantum-specific extensions only where needed. We define the resulting factor set and quality criteria, provide a many-to-many mapping of core factor-to-criterion relationships, discuss representative candidate metrics together with the execution, reference, and interpretation conditions needed to use them, a
    
[^14]: 优化下的目标代码到源代码映射：控制流轨迹能显示什么，以及源代码级覆盖在何处失效

    Object-to-Source Mapping under Optimization: What a Control-Flow Trace Can Show, and Where Source-Level Coverage Breaks

    [https://arxiv.org/abs/2609.29656](https://arxiv.org/abs/2609.29656)

    本文界定了控制流轨迹重建机器级执行与源代码级结构覆盖之间的边界，指出编译器优化（如条件融合、分支替换和映射信息退化）会导致轨迹在二进制层面完整、但条件级源代码覆盖证据仍无法归因的问题，并将轨迹完整性、源代码归因与覆盖满足度进行了系统区分。

    

    在经过验证的区间内，控制流轨迹可以重建机器级执行，但源代码级结构覆盖还需要一个经过验证的归因，将生成的指令和分支结果对应到源代码义务。本文定义了这一边界，并使用精确的编译器输出加以说明。编译器优化可以将多个源代码条件融合为一个分支，用条件选择或条件移动替换分支，并在调试和映射信息中降低溯源信息的质量。在这些情况下，轨迹对于生成的二进制文件可能是完整的，但条件级别的源代码证据仍然无法得到解决。因此，本文将轨迹完整性、源代码归因和覆盖满足度三者区分开来；将源代码到目标代码的转换与目标区域归因状态加以区分；并解释了当内部重新汇聚导致聚合分支观测不足以满足需求、而需要有序的逐实例条件求值历史时的情形。

    arXiv:2609.29656v1 Announce Type: new  Abstract: Within a validated interval, control-flow trace can reconstruct machine-level execution, but source-level structural coverage additionally requires a validated attribution from generated instructions and branch outcomes to source obligations. This paper defines that boundary and illustrates it with exact compiler output. Optimization can fuse multiple source conditions into one branch, replace a branch with a conditional select or move, and degrade provenance in debug and mapping information. In such cases, trace may be complete for the generated binary while condition-level source evidence remains unresolved. We therefore separate trace completeness, source attribution, and coverage satisfaction; distinguish source-to-object transformations from object-region attribution states; and explain when internal reconvergence makes aggregate branch observations insufficient and ordered per-instance condition-evaluation histories necessary. We c
    
[^15]: 嵌入式系统中信息流安全的自动抽象细化方法

    Automated Abstraction Refinement for Information Flow Security in Embedded Systems

    [https://arxiv.org/abs/2609.29645](https://arxiv.org/abs/2609.29645)

    本文提出一种自动抽象细化方法，基于状态依赖关系和检测到的潜在信息泄露来启发式地自动选择抽象级别，从而解决嵌入式系统信息流分析中精确性与开销难以兼顾的问题，摆脱了以往需要手动定义抽象级别的局限。

    

    信息流分析（IFA）是一种用于验证机密性和完整性的强大技术，因此对于安全敏感的嵌入式系统而言是非常理想的方法。然而，由于这些系统本质上具有并发性和时间依赖性，现有的针对嵌入式系统的信息流分析往往要么不够精确，要么代价高昂。在本文中，我们提出了一种利用自动抽象细化来解决这一问题的方法。其核心思想是：基于状态之间的依赖关系以及检测到的潜在信息泄露信息，启发式地选择抽象级别。我们的方法建立在我们之前工作的基础上，即利用符号执行在信息流分析中精确捕获进程之间的数据、控制、时序和事件依赖关系。为了以符号方式捕获变量的值，该分析采用了抽象解释。虽然现有方法需要手动定义抽象级别，但我们在本文中的新贡献在于实现了抽象级别的自动化选择（原文在此处截断）。

    arXiv:2609.29645v1 Announce Type: cross  Abstract: Information flow analysis (IFA) is a powerful technique for verifying confidentiality and integrity and is therefore highly desirable for security-sensitive embedded systems. However, as these systems are inherently concurrent and time-dependent, existing IFA for embedded systems tend to be either imprecise or expensive. In this paper, we propose an approach to tackle this problem using automatic abstraction refinement. The key idea is to heuristically choose abstraction levels based on information about dependencies between states and detected potential information leakage. Our approach builds on previous work, where we leverage symbolic execution to precisely capture data, control, timing, and event dependencies between processes within an IFA. To capture values symbolically, this analysis uses abstract interpretation. While the existing approach requires manual definition of abstraction levels, our novel contribution in this paper i
    
[^16]: 办公室规模验证搜索中的算子包、提议者强度与构造型家族平台期

    Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search

    [https://arxiv.org/abs/2609.29636](https://arxiv.org/abs/2609.29636)

    该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。

    

    验证搜索是指语言模型提出程序、硬评估器对其进行评分、选择机制保留最优解的过程，这种方法近来已推动了数学纪录的进展；但对提议者侧组件的受控消融实验仍然罕见。我们在办公规模上（笔记本电脑上运行的30B本地模型，每次运行120-600个验证样本）对最小化的FunSearch风格循环进行了仪器化，采用三种算子包：模型自行编写并携带的示意图式笔记本（代替逐字复制的精英解）、命名障碍、以及对已发现构造的行为排斥。在来自公共仓库的九个构造问题上，带两次重复的完整2³因子设计在名义两阶段分析中支持主要对比：该组合缩小了更多从种子解到纪录的差距（+0.196；名义合并p=0.023，阶段组合p≈0.08；每问题效应中位数为+0.045）。排斥机制在各处都提高了构造哈希多样性（p=0.0039；部分属于操纵检查）（注：原文摘要至此处截断）。

    arXiv:2609.29636v1 Announce Type: cross  Abstract: Verified search, in which a language model proposes programs, a hard evaluator scores them, and selection keeps the best, has recently moved mathematical records; controlled ablations of the proposer-side components remain rare. We instrument a minimal FunSearch-style loop at office scale (a 30B local model on a laptop, 120-600 verified samples per run) with three operator packages: a schematic notebook the model writes and carries instead of verbatim elites, a named obstacle, and behavioural repulsion from constructions already found. On nine construction problems from a public repository, the complete 2^3 factorial with two replicates favours the primary contrast in a nominal two-stage analysis: the composition closes more of the seed-to-record gap (+0.196; nominal pooled p=0.023, stage-combination p~0.08; median per-problem effect +0.045). Repulsion raises construction-hash diversity everywhere (p=0.0039; partly a manipulation check
    
[^17]: iCoder-27B：递归AI主导开发的前沿工业编程模型

    iCoder-27B: Recursive AI-Led Development of Frontier Industrial Coding Model

    [https://arxiv.org/abs/2609.29626](https://arxiv.org/abs/2609.29626)

    专家仅通过高密度、低频次的接口将目标、流程与权限编码为可复用研究技能，智能体即可自主选择实验、诊断结果并迭代训练策略，最终递归式开发出具备前沿竞争力的工业编程模型iCoder-27B。

    

    递归AI，即AI在构建和改进AI的过程中扮演日益完整的角色，是“以AI研发AI”这一愿景的皇冠明珠。尽管递归自我开发对于小模型、有界任务和固定时间预算已经变得可行，但这一雄心更具深远意义的实现——即开发出一个可发布、具备前沿竞争力的模型——仍然极具挑战性。在这项工作中，我们探讨了最低需要多少人类参与才能让智能体开发出前沿模型。我们将人类输入集中于一个高密度、低频率的接口：专家将目标、阶段脚手架、权限边界和操作流程编码为可复用的研究技能，而智能体则负责实例化这些先验知识、选择实验、诊断结果并修订训练策略。在具有挑战性的工业编程领域，智能体进化数据并协调监督微调（SFT）、在策略自蒸馏和强化学习（摘要在此处截断）

    arXiv:2609.29626v1 Announce Type: new  Abstract: Recursive AI, the prospect of AI taking an increasingly complete role in building and improving AI, is a crown jewel of AI for AI. Although recursive self-development has become practical for small models, bounded tasks, and fixed time budgets, a more consequential realization of this ambition, i.e., developing a release-ready, frontier-competitive model, remains far more challenging. In this work, we ask how little human involvement is sufficient for an agent to develop a frontier model. We concentrate human input into a high-density, low-frequency interface: experts encode objectives, stage scaffolds, permission boundaries, and operating procedures as reusable research skills, while the agent instantiates these priors, selects experiments, diagnoses outcomes, and revises the training strategy. In the challenging domain of industrial coding, the agent evolves data and coordinates SFT, on-policy self-distillation, and reinforcement learn
    
[^18]: CodeGraph：基于Wikidata实体锚定的开放分类法源代码知识图谱

    CodeGraph: Open-Taxonomy Knowledge Graph for Source Code with Wikidata Grounding

    [https://arxiv.org/abs/2609.29474](https://arxiv.org/abs/2609.29474)

    该论文提出CodeGraph流水线，利用代码专用大语言模型对源代码进行开放分类法语义标注，并通过三阶段实体链接过程将其锚定到Wikidata，从而构建出首个面向源代码的开放分类法知识图谱。

    

    GitHub和Software Heritage Archive等公共软件仓库存储了数十亿个文件，然而提取其中隐含的工程知识——即它们所实现的算法、所遵循的编程范式、所实例化的设计模式以及所服务的应用领域——仍然极具挑战性，因为现有的工具仅局限于句法和词法层面的分析。我们提出了一条利用代码专用大型语言模型构建源代码开放分类法语义标注的流水线。提取出的实体通过一个三阶段链接过程锚定到Wikidata：确定性的SPARQL阶段处理无歧义实体，深度研究智能体解析剩余的长尾实体，层级汇总阶段导入每个已解析Wikidata标识符的父级闭包。最终所得的标注被物化为一个面向源代码的开放分类法知识图谱。我们进一步引入了一个校准……

    arXiv:2609.29474v1 Announce Type: cross  Abstract: Public software repositories, like GitHub and Software Heritage Archive, store billions of files, yet extracting their implicit engineering knowledge ---i.e., the algorithms they implement, the paradigms they follow, the patterns they instantiate, and the application domains they serve--- remains challenging, as current tools are constrained to syntactic and token-level analysis. We present a pipeline for building an open-taxonomy semantic annotation of source code using a code-specialised Large Language Model. The extracted entities are grounded in Wikidata through a three-stage linking procedure: a deterministic SPARQL stage handles unambiguous entities, a Deep Research Agent resolves the residual long tail, and a hierarchy-rollup stage imports the parent-of closure of each resolved Wikidata identifier. The resulting annotations are materialised as a source-code-specific open-taxonomy knowledge graph. We further introduce a calibrate
    
[^19]: 以判断力为中心的软件工程教育：AI增强学习的后热潮时期回顾与框架

    Judgment-Centred Software Engineering Education: A Post-Hype Review and Framework for AI-Augmented Learning

    [https://arxiv.org/abs/2609.29473](https://arxiv.org/abs/2609.29473)

    本文通过对2023至2026年研究与实践的整合性综述提出一个以判断力为中心的软件工程教育框架，指出生成式AI虽能提升解释、反馈与练习的可及性，但学习成效取决于先验知识、脚手架支持和验证等条件，教育目标应是保留并评估人类的理解能力。

    

    生成式人工智能已经从一种颠覆性的新兴事物转变为软件开发和计算教育工作流程中的常态化组成部分，同时软件智能体开始在代码仓库、命令行、浏览器、测试和其他工具之间执行任务。教育问题不再应该是是否允许学生生成代码，而是软件工程（SE）专业能否在培养学生负责任地使用日益强大的AI系统的同时，保留并评估人类的理解能力。本文对2023年至2026年9月23日期间的研究与实践进行了结构化的整合性综述，并辅以关于AI素养、技术债务和人机协作方面的既有研究成果。证据支持一个有条件的结论：生成式AI可以改善解释、反馈、练习和短期任务完成方面的可及性，但学习成果取决于先验知识、脚手架支持、验证等条件。

    arXiv:2609.29473v1 Announce Type: new  Abstract: Generative artificial intelligence has moved from a disruptive novelty to a recurring part of software-development and computing-education workflows, while software agents are beginning to act across repositories, command lines, browsers, tests, and other tools. The educational problem is no longer whether students should be allowed to generate code, but whether software-engineering (SE) programs can preserve and assess human understanding while preparing students to work responsibly with increasingly capable AI systems. This paper presents a structured integrative review of research and practice from 2023 through 23 September 2026, supplemented by established work on AI literacy, technical debt, and human-AI collaboration. The evidence supports a conditional conclusion: GenAI can improve access to explanations, feedback, practice, and short-term task completion, but learning outcomes depend on prior knowledge, scaffolding, verification,
    
[^20]: SWE-Prometheus：衡量真实世界代码仓库中的工程治理改进

    SWE-Prometheus: Measuring Engineering Governance Improvements in Real-World Repositories

    [https://arxiv.org/abs/2609.29465](https://arxiv.org/abs/2609.29465)

    该论文提出了SWE-Prometheus基准，首次系统评估大语言模型编程智能体在开放式仓库工程治理任务中的能力，要求智能体自主识别风险、排序干预优先级并验证变更，通过六个治理维度和多重验证机制对十个模型进行了评测。

    

    基于大语言模型的编程智能体在仓库级软件工程任务上已取得显著进展。然而，现有的仓库基准测试通常从一个由人类确定的问题出发，评估补丁是否满足某个功能性信号。我们提出了SWE-Prometheus，一个针对更广泛任务——改进代码仓库工程治理——的基准测试。每个任务提供一个固定的代码快照和一个开放式的目标，要求智能体识别风险、确定干预措施的优先级并验证由此产生的变更。SWE-Prometheus通过配对证据、干净环境探测、行为门控以及对同一证据的两项独立教师评分，来评估六个治理维度。该基准包含60个仓库；十个模型在一个共享的22仓库公共子集上进行评估，其中平均归一化治理改进得分介于0.0568到0.5760之间，观察到的行为破坏率介于0%到23%之间。

    arXiv:2609.29465v1 Announce Type: new  Abstract: Large language model based coding agents have made substantial progress on repository-level software engineering tasks. Existing repository benchmarks, however, usually start from a human-identified issue and evaluate whether a patch satisfies a functional signal. We present SWE-Prometheus, a benchmark for the broader task of improving repository engineering governance. Each task provides a fixed snapshot and an open-ended objective, requiring the agent to identify risks, prioritize interventions, and verify the resulting changes. SWE-Prometheus evaluates six governance dimensions through paired evidence, clean-environment probes, behavior gates, and two independent teacher ratings of the same evidence. The benchmark contains 60 repositories; ten models are evaluated on a shared 22-repository public subset, where mean Normalized Governance Improvement ranges from 0.0568 to 0.5760 and observed behavior-breakage rates range from 0% to 23%.
    
[^21]: 揭秘智能合约审计的智能体技能：设计、有效性与行为影响

    Demystifying Agent Skills for Smart Contract Auditing: Design, Effectiveness, Behavioral Impact

    [https://arxiv.org/abs/2609.29454](https://arxiv.org/abs/2609.29454)

    本文首次系统研究了智能体技能在智能合约安全审计中的应用，通过收集并评估83个真实世界的审计技能，揭示了其设计特征、提升漏洞检测的有效性以及对智能体执行行为的影响。

    

    LLM智能体（尤其是Claude Code和OpenAI Codex）正逐渐成为超越纯编码工具的通用型工具。这些智能体可以通过技能（skills）得到增强——技能是打包了领域知识、工作流程和工具使用指令的可复用制品。然而，迄今为止，人们对这类技能是如何设计的、以及它们在实践中如何影响智能体的有效性和行为知之甚少。本文在智能合约安全审计这一智能体已展现出巨大潜力的领域中研究这些问题。我们从实际环境中系统性地收集了83个智能合约审计技能，并在七种“智能体—模型”配置组合下于EVMBench基准上进行评估。我们的研究考察了三个维度： 审计技能的设计特征，包括其结构、知识表示、工作流程和工具依赖关系； 它们在提升漏洞检测能力方面的有效性； 它们对智能体执行轨迹的影响。

    arXiv:2609.29454v1 Announce Type: new  Abstract: LLM agents, notably Claude Code and OpenAI Codex, are emerging as versatile tools beyond coding agents only. These agents can be enhanced with skills---reusable artifacts that package domain knowledge, workflows, and tool-use instructions. To date, however, little is known about how such skills are designed or how they affect agent effectiveness and behavior in practice. In this paper, we investigate these questions in smart contract security auditing, a domain in which agents have shown substantial promise. We systematically collect 83 smart contract audit skills from the wild and evaluate them on EVMBench across seven agent--model configurations. Our study examines three dimensions: (i) the design characteristics of audit skills, including their structure, knowledge representations, workflows, and tool dependencies; (ii) their effectiveness in improving vulnerability detection; and (iii) their influence on agent execution trajectories.
    
[^22]: 大语言模型用于编程：究竟是修复还是重新实现错误代码？

    Large Language Models for Programming: Actually Fixing or Reimplementing Incorrect Code?

    [https://arxiv.org/abs/2609.29410](https://arxiv.org/abs/2609.29410)

    本研究基于 Codeforces 竞赛编程的真实提交数据，通过对比人工修复补丁与模型生成结果的相似性，评估大语言模型在修复缺陷代码时究竟是真正修复原始代码还是倾向于重新实现全新方案。

    

    最近的研究表明，大语言模型能够在包括竞赛编程在内的多种编程环境中有效地解决问题和修复缺陷。现有方法主要独立评估大语言模型在解决问题或修复缺陷方面的性能，但并未探讨这两种能力之间的关系。本工作着重于确定大语言模型在修复缺陷时与原缺陷代码的偏离程度（与人工编写的补丁相比），以及是否存在倾向于生成全新解决方案的偏见。我们构建了一个数据集，包含来自 Codeforces 上几位用户的所有提交（约3000个），并将每个有缺陷的提交与其对应的人工修复进行匹配。通过将有缺陷的解决方案与人工修复之间的相似性作为基线，我们在3个 OpenAI GPT 模型（gpt-5-nano、gpt-5-mini、gpt-5.1）上评估了大语言模型生成的缺陷修复的质量。我们检查生成的解决方案是否解决了问题（摘要在此处截断）。

    arXiv:2609.29410v1 Announce Type: new  Abstract: Recent studies have shown that Large Language Models can effectively solve problems and fix bugs in diverse programming environments, including competitive programming. Existing approaches primarily evaluate LLM performance in problem solving or bug fixing independently, but do not explore the relationship between these two capabilities. This work focuses on determining how much the LLM deviates from a buggy solution to fix the bug compared to a human-written patch, and if there is a bias towards generating entirely new solutions. We construct a dataset with all the submissions ($\sim$ 3000) from a couple of users from Codeforces, and we match each buggy submission with its corresponding human fix. By using the similarity between the buggy solution and the human fix as a baseline, we evaluate the quality of LLM-generated bug fixes on 3 OpenAI GPT models (gpt-5-nano, gpt-5-mini, gpt-5.1). We check if the generated solutions solve the prob
    
[^23]: 最后一道人类关卡：面向治理自动化的前置部署工程

    The Last Human Gate: Forward Deployed Engineering for Governance Automation

    [https://arxiv.org/abs/2609.29345](https://arxiv.org/abs/2609.29345)

    该论文提出将数字治理关卡视为可执行契约的任务替代框架，推导出剩余工作阈值以解释为何自动化多数案例反而可能增加人力，并通过 DGF-Bench 基准（300 个合成项目、899 次运行）实证了前沿大模型可达到最高 94.98% 的严格关卡成功率。

    

    企业治理需要决策、证据以及可问责的权威，但它并不要求每个审查任务都保留其当前的人工实现方式。我们为数字治理框架提出了一种任务替代框架，将每个治理关卡视为可执行的契约。实现替代需要满足以下条件：充分且可获取的信息、有效的决策与权限校验，以及在计入异常处理、验证、纠正和维护工作后，总人力工作量的净减少。我们推导出了剩余工作阈值，并解释了为何对大多数案例进行自动化仍可能增加整体劳动量。前置部署工程将这些条件与由智能体、规则引擎、证据服务和升级机制构成的架构相衔接。DGF-Bench 基准提供了来自 300 个合成项目和 899 次可评估模型-项目运行的受控实验证据。Gemini 3.8 Flash、GPT-5.6 Luna 和 DeepSeek v4.1 Flash 分别取得了 94.98%、83.29% 和 74.18% 的严格关卡成功率；complet（原文摘要在此处截断）

    arXiv:2609.29345v1 Announce Type: new  Abstract: Enterprise governance requires decisions, evidence, and accountable authority; it does not require every review task to retain its current human implementation. We develop a task-substitution framework for Digital Governance Frameworks (DGF), treating each gate as an executable contract. Substitution requires sufficient accessible information, valid decision and authority checks, and a reduction in total human work after exceptions, verification, correction, and maintenance are counted. We derive a residual-work threshold and show why automating most cases can still increase labor. Forward deployed engineering connects these conditions to an architecture for agents, rule engines, evidence services, and escalation. DGF-Bench supplies controlled evidence from 300 synthetic projects and 899 evaluable model-project runs. Gemini 3.8 Flash, GPT-5.6 Luna, and DeepSeek v4.1 Flash achieve strict gate success of 94.98%, 83.29%, and 74.18%; complet
    
[^24]: 面向多核信息物理系统的基于模型重定向：Simulink到OpenCL的工作流

    Model-Based Retargeting to Many-Core CPS: Simulink-to-OpenCL Workflow

    [https://arxiv.org/abs/2609.29311](https://arxiv.org/abs/2609.29311)

    本文提出一种保持工作流的重定向方法，通过自动将Simulink模型转换为OpenCL代码实现CPS应用向多核处理器的移植，并在Kalray MPPA Coolidge2上以轨迹规划器成功验证了该方法的可行性。

    

    本文解决了信息物理系统（CPS）的基于模型开发（MBD）与先进多核执行之间的软件可移植性差距。我们提出了一种保持工作流的重定向方法，将具有候选级数据并行性的基于Simulink的CPS应用重定向到基于OpenCL的多核处理器上。我们的工具链无需为新平台手动重写模型，而是使用MathWorks GPU Coder提取数据并行的CUDA代码，然后通过自定义框架将其转换为OpenCL主机和设备代码。该转换过程处理了语法重写、API仿真以及平台特定的参数打包。我们在Kalray MPPA Coolidge2上为计算密集型的Frenet坐标系轨迹规划器部署了此工作流。结果表明，针对所评估的CPS工作负载和平台，这种保持工作流的重定向流水线是可行的。

    arXiv:2609.29311v1 Announce Type: new  Abstract: This paper addresses the software portability gap between Model-Based Development (MBD) and advanced many-core execution for Cyber-Physical Systems (CPS). We present a workflow-preserving retargeting approach for Simulink-based CPS applications with candidate-wise data parallelism to OpenCL-based many-core processors. Rather than manually rewriting models for new platforms, our toolchain uses MathWorks GPU Coder to extract data-parallel CUDA code, which is then translated into OpenCL host and device code via a custom framework. The conversion handles syntax rewriting, API emulation, and platform-specific argument packing. We deployed this workflow for a computationally intensive Frenet-frame trajectory planner on the Kalray MPPA Coolidge2. The results demonstrate the feasibility of a workflow-preserving retargeting pipeline for the evaluated CPS workload and platform.
    
[^25]: 论需求异味在基于大语言模型代码生成中的影响

    On the Impact of Requirement Smells in LLM-Based Code Generation

    [https://arxiv.org/abs/2609.29208](https://arxiv.org/abs/2609.29208)

    需求文本中的“需求异味”会降低大语言模型生成代码的功能正确性，且异味密度越高，正确性越低。

    

    软件需求通常被纳入大语言模型（LLM）辅助软件开发中所使用的提示词。近期研究表明，需求异味会影响需求与代码之间的自动化可追溯性，但关于其在代码生成中所产生影响的实证证据仍然有限。为弥补这一空白，我们在一项关于自动化可追溯性的前期研究基础上，复用其数据集和需求异味分类体系，并将其扩展用于评估大语言模型生成代码的功能正确性。我们使用一个由四个应用的需求及对应系统测试组成的基准测试集，逐步在原本清晰的需求中引入语义、语法和词汇层面的异味，并分析它们对生成实现的影响。我们的结果表明，异味密度的增加通常与基于测试套件的功能正确性下降相关，尽管无异味的需求仍可能生成

    arXiv:2609.29208v1 Announce Type: new  Abstract: Software requirements are typically incorporated into prompts used in LLM-assisted software development. Recent work has shown that requirement smells can affect automated traceability between requirements and code, but empirical evidence on their effects in code generation remains limited. To address this gap, we build upon a prior study on automated traceability by reusing its dataset and requirement smell taxonomy, while extending it to evaluate the functional correctness of LLM-generated code. Using a benchmark consisting of requirements and corresponding system tests for four applications, we progressively introduced semantic, syntactic, and lexical smells into otherwise clear requirements and analyzed their influence on generated implementations. Our results suggest that increasing \textit{smell density} was generally associated with lower test-suite-based functional correctness, although non-smelly requirements could still produce
    
[^26]: HistoRAG：一个基于引文溯源的问答助手，用于扫描版地方历史与遗产档案教学

    HistoRAG: A Citation-Grounded Question Answering Assistant for Teaching with Scanned Local History and Heritage Archives

    [https://arxiv.org/abs/2609.29184](https://arxiv.org/abs/2609.29184)

    HistoRAG 通过视觉语言模型转录扫描档案，并融合混合文本索引、关系数据库与知识图谱三种存储及轻量级路由，为地方历史与遗产教学提供每个事实都可溯源到具体卷册页码的引文支撑问答。

    

    准备地方历史与文化遗产课程的教师在备课时所依据的材料很难使用：原始资料是没有文本层的扫描书籍，辅助记录则是以电子表格形式发布的行政目录。通用聊天机器人能够流利地回答此类问题，但无法提供可验证的来源，而这恰恰是教师最需要的特性。本文提出了 HistoRAG，一个面向单一区域馆藏的问答助手，它为每一个事实都标注所引用的卷册和页码。HistoRAG 使用视觉语言模型对每一页进行转录，并根据 token 概率保留行级置信度。它从同一馆藏构建了三个存储：混合文本索引、关系型目录数据库，以及仅从实体密集段落中抽取的知识图谱。一个轻量级路由器将每个问题发送到所需的存储，使得计数类问题到达数据库，关系类问题……

    arXiv:2609.29184v1 Announce Type: new  Abstract: Teachers who prepare lessons on local history and cultural heritage work from material that is hard to use. The primary sources are scanned books without a text layer, and the supporting records are administrative catalogs released as spreadsheets. A general chatbot answers such questions fluently but without a verifiable source, which is the property a teacher needs most. This paper presents HistoRAG, a question answering assistant that answers from one regional collection and cites a volume and a page for every fact. HistoRAG transcribes each page with a vision language model and keeps a line level confidence from the token probabilities. It builds three stores from the same collection: a hybrid text index, a relational catalog database, and a knowledge graph extracted only from entity dense passages. A lightweight router sends each question to the stores it needs, so that counting questions reach the database and relational questions 
    
[^27]: 编排AI辅助的代码修复：大型工业代码库中的社会技术瓶颈

    Orchestrating AI-Assisted Code Remediation: Socio-Technical Bottlenecks in a Large Industrial Repository

    [https://arxiv.org/abs/2609.29172](https://arxiv.org/abs/2609.29172)

    本研究通过对大型工业C++代码库开展为期15天的实地案例研究，揭示了大规模AI辅助代码修复在持续集成、代码审查和团队协调等方面所面临的社会技术瓶颈。

    

    背景：在大型、长期存续的代码库中，代码退化问题若通过手动重构和机会性清理来修复，成本十分高昂。基于大语言模型（LLM）的编程助手可以大规模执行机械化的代码修复，但其对工业工作流的影响尚未得到充分探索。目标：我们研究了大规模AI辅助代码修复如何影响一个大型工业代码库中基于提交构建的持续集成（CI）、代码审查和团队协调，以及当AI辅助使源代码修改变得低廉时，哪些社会技术瓶颈会制约此类修复。方法：我们报告了一项为期15天的探索性单案例实地研究，研究过程中一名经验丰富的开发者使用命令行AI编程助手来修复一个闭源工业C++代码库中普遍存在的问题。我们将Gerrit元数据与开发者日记和团队聊天记录进行三角互证，并通过描述性统计和定性编码进行分析。结果：AI辅助

    arXiv:2609.29172v1 Announce Type: new  Abstract: Background: Code degradation in large, long-lived codebases is costly to remediate through manual refactoring and opportunistic clean-ups. LLM-based coding assistants can perform mechanical remediation at scale, but their impact on industrial workflows is underexplored. Objective: We investigate how massive AI-assisted code remediation affects build-on-commit continuous integration (CI), code review, and team coordination in a large industrial repository, and which socio-technical bottlenecks constrain such remediation when source editing becomes cheap through AI assistance. Method: We report on a 15-day exploratory single-case field study in which an experienced developer used a command-line AI coding buddy to remediate widespread issues in a closed-source industrial C++ repository. We triangulate Gerrit metadata with a developer diary and team chat, analyzed through descriptive statistics and qualitative coding. Results: AI-assisted re
    
[^28]: VidTutorAssistant：自动化回答编程教程问题

    VidTutorAssistant: Automating Responses to Programming Tutorial Questions

    [https://arxiv.org/abs/2609.29129](https://arxiv.org/abs/2609.29129)

    VidTutorAssistant是一个基于检索增强生成（RAG）技术的网络平台，通过视频转录文本的语义检索和GPT-4自动生成答案，解决了YouTube编程教程视频评论区观众提问得不到及时回复的问题。

    

    YouTube上的编程教程视频是软件开发者和学生的重要信息资源，其评论区已发展成为观众提出后续问题的活跃空间。然而，这些问题的数量往往超出内容创作者能够解答的范围，导致学习者无法获得所需的澄清说明。我们提出了VidTutorAssistant，这是一个自动化回答编程视频教程中观众问题的网络平台。VidTutorAssistant实现了一个检索增强生成（RAG）流水线：首先提取视频的文字记录，然后对其进行分段和向量化嵌入；接着将每条观众评论分类为问题或非问题，通过余弦相似度为每个已识别的问题检索最相关的文字记录片段，再利用大语言模型（GPT-4）生成答案，并以检索到的文字记录片段作为上下文来保证回答的准确性。我们验证了……（摘要原文在此处截断）

    arXiv:2609.29129v1 Announce Type: new  Abstract: Programming tutorial videos on YouTube are an important information resource for software developers and students, and their comment sections have evolved into active spaces where viewers ask follow-up questions. The volume of these questions, however, often exceeds what content creators can address, leaving learners without the clarifications they need. We present VidTutorAssistant, a web platform that automates responses to viewer questions on programming video tutorials. VidTutorAssistant implements a retrieval-augmented generation pipeline that extracts a video's transcript, then segments it and embeds it. It then classifies each viewer comment as being a question or non-question, retrieves the most relevant transcript segments to each identified question via cosine similarity, and then generates an answer to the question using an LLM (GPT-4), while grounding the response using the retrieved transcript segments as context. We validat
    
[^29]: 恰好一次语义由谁承担？模型、智能体框架与工具契约对 LLM 智能体重复副作用的影响

    Where Does Exactly-Once Live? Model, Harness, and Tool-Contract Effects on Duplicate Side Effects in LLM Agents

    [https://arxiv.org/abs/2609.29095](https://arxiv.org/abs/2609.29095)

    该论文提出确定性沙盒基准 LIMBO，研究 LLM 智能体的“恰好一次”副作用语义应由模型、智能体框架还是工具契约来保障，并发现答案取决于故障类型：当即时回读能够揭示实际结果时，由模型来决定。

    

    当使用工具的智能体的写操作超时或返回服务器错误时，该操作可能已经实际生效。盲目重试会导致重复执行——第二次扣款、第二次公告、第二次部署——而放弃重试则会跳过必要的工作。我们提出问题：恰好一次（exactly-once）行为应该在哪里强制执行：在模型中、在智能体框架（harness）中，还是在工具契约中？我们介绍了 LIMBO，一个由六个服务组成的确定性沙盒，这些服务具有真实的契约（可选的幂等键、最终一致性和缺失的读取路径），并在服务边界注入了十二种故障模式，包括延迟提交、重复投递和部分批次；每个回合都根据已提交效果的账本进行评分。在涵盖九个近期模型、三个生产级智能体框架、两种契约变体和十五种恢复条件共 25,930 个回合的实验中，答案取决于具体的故障类型。当即时回读能够揭示发生了什么时，由模型来决定……

    arXiv:2609.29095v1 Announce Type: cross  Abstract: When a tool-using agent's write times out or returns a server error, the action may already have taken effect. Retrying blindly duplicates it -- a second charge, a second announcement, a second deployment -- while giving up skips required work. We ask where exactly-once behaviour should be enforced: in the model, in the agent harness, or in the tool contract. We introduce LIMBO, a deterministic sandbox of six services with realistic contracts (optional idempotency keys, eventually consistent and missing read paths) and twelve fault modes injected at the service boundary, including late commits, redelivery and partial batches; every episode is graded against a ledger of committed effects. Across 25,930 episodes spanning nine recent models, three production agent harnesses, two contract variants and fifteen recovery conditions, the answer depends on the fault. When an immediate read-back can reveal what happened, the model decides: front
    
[^30]: 软件工程自我效能量表（SESES）的设计、开发及初步效度与信度证据

    Design, development, and preliminary validity and reliability evidence of the Software Engineering Self-Efficacy Scale (SESES)

    [https://arxiv.org/abs/2609.29068](https://arxiv.org/abs/2609.29068)

    本研究设计开发了涵盖需求工程、团队协作、软件质量管理、软件设计与架构及敏捷方法论五个维度的软件工程自我效能量表（SESES），并通过对527名计算机专业本科生的试测提供了初步效度与信度证据。

    

    本研究的目的是设计、开发并实施软件工程自我效能量表（SESES），并为其提供初步的效度与信度证据。我们以软件工程课程的指导内容与相关概念以及自我效能感理论为基础构建概念框架，生成了包含87个条目的初始题库，用以操作化并测量计算机专业本科生的软件工程自我效能感。该概念框架涵盖五个维度：1）需求工程；2）团队合作与协作；3）软件质量管理；4）软件设计与架构；5）软件敏捷方法论。我们对527名在本学期或前一学期修完软件工程课程的计算机专业本科生进行了SESES的试测，并采用主轴因子法进行探索性因子分析（EFA）……

    arXiv:2609.29068v1 Announce Type: new  Abstract: The purpose of this research is to design, develop, implement, and provide preliminary validity and reliability evidence of the Software Engineering Self-Efficacy Scale (SESES). Framed by a conceptual framework using guidance in software engineering curriculum and concepts along with the notion of self-efficacy, we generated an initial item pool of n = 87 items to operationalize and measure software engineering self-efficacy among undergraduate computing students. The conceptual framework traces five dimensions: 1) Requirements Engineering, 2) Teamwork and Collaboration, 3) Software Quality Management, 4) Software Design and Architecture, and 5) Software Agile Methodologies. We pilot tested the SESES with n = 527 undergraduate computing students who had completed a software engineering course in the current semester or a previous academic semester. We employed Exploratory Factor Analysis (EFA) with the Principal Axis Factoring method and
    
[^31]: 基于本地大语言模型与数字孪生的多产线任务调整人机协作

    Human-AI Collaboration for Multi-Line Task Adjustment Using Local Large Language Models and a Digital Twin

    [https://arxiv.org/abs/2609.29061](https://arxiv.org/abs/2609.29061)

    本研究提出了一种集成本地大语言模型、数字孪生与人工决策的多产线任务调整系统，通过“提议-验证-决策”工作流将操作员意图转化为经过仿真验证的候选策略，并实现了从需求到决策的全流程可追溯。

    

    自动化系统必须适应不断变化的任务、设备状态和人员配置情况，同时为人工审核提供证据。本研究提出了一种多产线任务调整系统，该系统集成了本地大语言模型、数字孪生和人工决策。“提议-验证-决策”工作流将操作员的意图转化为结构化需求，生成一组有界数量的候选策略，并对语义、仿真执行和运行约束进行检查。关联记录保留了从请求到验证证据和最终决策的全流程可追溯性。研究使用四条虚拟手术器械分拣线对三十条固定测试记录进行了评估：其中28条用于评估工作流，2条用于评估模型生成。18个工作流案例符合预期；自主策略-工作流的成功率为3/10，对无效输入的正确拒绝率为7/8。所有四个通过前序检查、产生完整证据并……（原文截断）

    arXiv:2609.29061v1 Announce Type: new  Abstract: Automation systems must adapt to changing tasks, equipment states, and staffing conditions while providing evidence for human review. This study presents a multi-line task-adjustment system integrating a local large language model, a digital twin, and human decision-making. A Propose-Verify-Decide workflow translates operator intent into structured requirements, generates a bounded set of candidate strategies, and checks semantics, simulation execution, and operational constraints. Linked records preserve traceability from requests to verification evidence and decisions. Thirty fixed test records were evaluated using four virtual surgical-instrument sorting lines: 28 assessed the workflow and two assessed model generation. Eighteen workflow cases met expectations; autonomous strategy-workflow success was 3/10, and correct rejection of invalid inputs was 7/8. All four cases that passed preceding checks, produced complete evidence, and rea
    
[^32]: 面向自主渗透测试框架的校准决策模型：JEV与Laya作为LLM驱动的渗透测试智能体的“系统一”决策层

    Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers for LLM-Driven Pentest Agents

    [https://arxiv.org/abs/2609.28940](https://arxiv.org/abs/2609.28940)

    本文提出用JEV和Laya这类轻量级非生成式的“系统一”校准分类器作为LLM驱动的自主渗透测试智能体的专用决策层，以降低误报、纠正严重程度虚高并减少计算浪费。

    

    自主渗透测试框架使用大型语言模型（LLM）进行侦察、漏洞利用和报告生成，但往往依赖同样的模型来确认发现结果、评定严重程度并选择智能体。这可能导致误报、严重程度虚高以及计算资源浪费。我们研究了“系统一”决策模型——一种轻量级非生成式分类器，可返回类型化且经过校准的判定结果——如何支持这些决策。我们做出了五项贡献。第一，我们定义了四个决策点：发现结果裁决、严重程度重新校准、智能体剪枝和确认循环。第二，我们展示了一项探索性的NeuroSploit案例研究，将一次使用TypeSafe系统一（Jev）的运行与一次不使用它的运行进行对比，测试目标为包含13个漏洞的Web应用。严重程度分布、运行时间以及按暴露数据类型评级的差异为该架构提供了动机，但并未确立统计显著性。第三，我们回顾了已发表的……（摘要原文在此处截断）

    arXiv:2609.28940v1 Announce Type: cross  Abstract: Autonomous penetration-testing harnesses use large language models (LLMs) for reconnaissance, exploitation, and reporting, but often rely on those same models to confirm findings, grade severity, and select agents. This can lead to false positives, inflated severity, and wasted compute. We examine how System One decision models, lightweight non-generative classifiers that return typed, calibrated verdicts, can support these decisions. We make five contributions. First, we define four decision points: finding adjudication, severity recalibration, agent pruning, and confirmation loops. Second, we present an exploratory NeuroSploit case study comparing one run with TypeSafe System One (Jev) and one without it against a web target containing 13 vulnerabilities. Differences in severity distribution, runtime, and grading by exposed data type motivate the architecture but do not establish statistical significance. Third, we review published s
    
[^33]: 硬件设计验证的自动测试框架演化：LLM 能否巩固已发现测试框架的收益？

    Automatic Harness Evolution for Hardware Design Verification: Can LLMs Consolidate Gains Across Discovered Harnesses?

    [https://arxiv.org/abs/2609.28908](https://arxiv.org/abs/2609.28908)

    该研究在硬件设计验证任务中对固定语言模型进行自动测试框架演化，发现虽然演化显著提升了完成尝试和任务覆盖率，但这些收益难以跨任务巩固和保持，表明LLM目前尚无法可靠地整合测试框架的改进收益。

    

    智能体的行为取决于围绕语言模型的测试框架，但语言模型能否可靠地改进此类硬件设计任务的测试框架仍不清楚。我们围绕一个固定的目标模型，在12个专有的设计验证根因定位任务上研究了测试框架的自动演化。在每项任务五次试验中，自动演化的测试框架将完成的尝试次数提高了71-76%，将任意命中任务覆盖率提高了80-100%，而总正确尝试仅提高了18-24%。最强的至少可复现两次的成功结果仅提高了一个任务，且后续候选者在任务之间交换收益而非保留收益。一个辅助候选者在排除于搜索之外的四任务验证集上有所改进，但在包含搜索和验证任务的后续12任务重放中与基线持平，因此所选收益未能在完整任务池中保持。在所测试的演化谱系中，有效的搜索……（摘要截断）

    arXiv:2609.28908v1 Announce Type: cross  Abstract: Agent behavior depends on the harness surrounding a language model, but it remains unclear whether language models can reliably improve such harnesses for hardware-design tasks. We study automatic harness evolution around a fixed subject model on 12 proprietary design-verification root-cause localization tasks. Across five trials per task, automatically evolved harnesses increased completed attempts by 71-76% and any-hit task coverage by 80-100%, while total correct attempts improved by only 18-24%. The strongest success reproducible at least twice result improved by one task, and later candidates exchanged gains across tasks rather than preserving them. An auxiliary candidate improved on a four-task validation set excluded from search but tied its baseline on a subsequent 12-task replay containing both search and validation tasks, so the selected gain did not persist across the full pool. Across the tested lineage, useful search, evid
    
[^34]: 面向自动程序修复的规范驱动基准测试：从静态语料库到可执行规范

    Specification-Driven Benchmarking for Automated Program Repair From Static Corpora to Executable Specifications

    [https://arxiv.org/abs/2609.28896](https://arxiv.org/abs/2609.28896)

    该论文提出“规范驱动基准测试”新范式，以可执行规范定义自动程序修复基准并通过生成、验证与语料库管理组件动态实现，从而克服静态数据集实验控制有限、易受污染、无法按需重新生成等局限。

    

    自动程序修复（APR）的基准测试传统上被构建为静态数据集，其特征继承自其中所包含的缺陷。尽管这一范式推动了数十年的进展，但有限的语料库所能提供的实验控制十分有限，随着其被反复使用而日益容易受到数据污染，并且无法随着评估需求的演进而被系统地重新生成或调整。我们提出规范驱动的基准测试，这是一种通过可执行规范来定义基准、并通过基准生成来实现的范式。该规范明确声明了基准的预期属性（包括程序上下文、缺陷分类体系、难度、验证策略以及语料库约束），而生成流水线则通过相互独立的生成、验证和语料库管理组件来实现这些要求。我们构建了这一方法的概念基础……

    arXiv:2609.28896v1 Announce Type: new  Abstract: Automated Program Repair (APR) benchmarks have traditionally been constructed as static datasets whose characteristics are inherited from the defects they contain. While this paradigm has enabled decades of progress, finite corpora provide limited experimental control, become increasingly susceptible to contamination as they are reused, and cannot be systematically regenerated or adapted as evaluation requirements evolve. We propose specification-driven benchmarking, a paradigm in which benchmarks are defined by executable specifications and realized through benchmark generation. The specification explicitly declares the intended properties of the benchmark (including program context, fault taxonomy, difficulty, validation strategy, and corpus constraints) while a generation pipeline realizes those requirements through independent generation, validation, and corpus management components. We develop the conceptual foundations of this appr
    
[^35]: pytest-gpu-proof：通过本地GPU认证为GPU代码实现云端CPU持续集成

    pytest-gpu-proof: Enabling Cloud-CPU Continuous Integration for GPU Code with Local GPU Attestation

    [https://arxiv.org/abs/2609.28862](https://arxiv.org/abs/2609.28862)

    该论文提出开源pytest插件pytest-gpu-proof，通过在本地运行GPU测试并生成签名认证收据，将其集成到低成本的云端CPU持续集成工作流中，解决了云端GPU CI成本高昂导致的GPU代码测试严重不足问题。

    

    GPU加速如今在机器人领域已十分普遍，但云端托管的GPU持续集成（CI）运行器成本高昂，导致GPU加速代码的测试严重不足。我们提出了pytest-gpu-proof，一个提供实用折中方案的开源pytest插件。测试可以在本地机器上运行，并附带一份签名收据，精确记录运行的内容及其产生的结果，随后集成到标准的CPU CI工作流中（例如GitHub Actions）。该工具已开源并发布在PyPI上，我们正在积极将其集成到实验室的整个软件栈中。

    arXiv:2609.28862v1 Announce Type: cross  Abstract: GPU acceleration is now routine across robotics, but cloud-hosted GPU continuous integration (CI) runners are expensive, resulting in severe under-testing of GPU-accelerated code. We present pytest-gpu-proof, an open-source pytest plugin offering a practical middle ground. Tests can be run on a local machine, signed with a receipt of exactly what ran and what it produced, and integrated into standard CPU CI workflows (e.g., GitHub Actions). The tool is open source and on PyPI, and we are actively integrating it across our lab's software stack.
    
[^36]: RECLAIM：智能体能复现机器学习论文的结论吗？

    RECLAIM: Can Agents Reproduce the Claims of Machine Learning Papers?

    [https://arxiv.org/abs/2609.28850](https://arxiv.org/abs/2609.28850)

    RECLAIM是一个基于100篇NeurIPS 2025论文的可重建基准测试，通过预先定义复现目标、成功标准和GPU预算，并按作者发布资源分为运行、重训练、重新实现三个难度级别，用独立语言模型依据日志评分，结果显示最好的AI智能体也只能分别复现41%、27%和15%的论文结果。

    

    复现一篇机器学习论文涉及大部分研究步骤，从安装软件、调试到运行实验，这些工作正日益由AI智能体来承担。我们提出了RECLAIM，一个基于100篇NeurIPS 2025论文的基准测试，可以每年从新会议中重建更新。对于每篇论文，我们预先固定要复现的结果、判定复现成功的标准以及GPU小时预算。智能体必须利用论文本身和作者发布的资源来复现该结果。作者发布的内容决定了难度级别：Run级（运行级）发布包含代码、数据和模型权重；Retrain级（重训练级）发布缺少权重，因此智能体需要自行训练模型；Reimplement级（重新实现级）发布缺少代码，因此智能体需要自己编写代码。我们使用一个独立的语言模型根据日志和输出（而非智能体自己的报告）来评分。我们在每篇论文上运行四个智能体各一次；结果发现每个级别中表现最好的智能体也仅能复现Run级论文的41%、Retrain级的27%和Reimplement级的15%。

    arXiv:2609.28850v1 Announce Type: new  Abstract: Reproducing a machine learning paper involves most research steps, from installing software and debugging to running experiments, work that AI agents increasingly do. We introduce RECLAIM, a benchmark of 100 NeurIPS 2025 papers that can be rebuilt yearly from new conferences. For each paper we fix in advance the result to reproduce, what counts as a successful reproduction, and a GPU-hour budget. An agent must reproduce that result using the paper and whatever its authors released. What the authors released decides the difficulty tier. Run-tier releases include code, data, and weights; Retrain-tier releases lack weights, so the agent trains the model; Reimplement-tier releases lack code, so the agent writes it. A separate language model grades runs from logs and outputs rather than agents' reports. We run four agents once per paper; the best agent in each tier reproduces only 41% of Run-tier papers, 27% at Retrain, and 15% at Reimplement
    
[^37]: Swift中的参数包迭代：面向可变参数泛型的普通控制流

    Pack Iteration in Swift: Ordinary Control Flow for Variadic Generics

    [https://arxiv.org/abs/2609.28822](https://arxiv.org/abs/2609.28822)

    Swift 6.0引入的参数包迭代特性允许开发者使用普通的for-in循环遍历可变参数泛型的参数包，取代了传统上C++等语言中依赖递归“剥离”元素的复杂专家级编程模式。

    

    可变参数泛型是类型安全元编程的强大工具。然而，在大多数广泛使用的编程语言中，它仍然是一个“专家专属”的特性，因为它依赖于复杂的模式，例如递归分解或无法与普通控制流自然组合的展开表达式。以C++为例，访问参数包中的元素传统上依赖于不直观的递归模式来“剥离”元素。本文介绍了参数包迭代，这是Swift 6.0中引入的一个特性，它允许开发者使用熟悉的命令式for-in循环来遍历参数包。通过将包展开作为迭代的一等来源，Swift弥合了高级表达能力与高级泛型编程之间的差距。我们详细介绍了该特性在Swift编译器中的设计与实现，重点关注在约束系统中的静态类型检查与动态执行之间架起桥梁所面临的挑战。

    arXiv:2609.28822v1 Announce Type: cross  Abstract: Variadic generics are a powerful tool for type-safe meta-programming. Yet in most widely used languages, they remain an "expert-only" feature due to their reliance on complex patterns such as recursive decomposition or expansion expressions that do not compose naturally with ordinary control flow. In C++, for example, accessing elements of parameter packs has traditionally relied on unintuitive recursive patterns that "peel off" elements.   This paper presents Pack Iteration, a feature introduced in Swift 6.0 that allows developers to iterate over parameter packs using a familiar, imperative for-in loop. By treating pack expansion as a first-class source for iteration, Swift bridges the gap between high-level expressiveness and advanced generic programming.   We detail the design and implementation of this feature within the Swift compiler, focusing on the challenges of bridging static type-checking in the constraint system with dynami
    
[^38]: 基于类别的多层次建模（MLM）：统一幂类型与超类

    Category-Based MLM: Unifying Powertypes with Superclasses

    [https://arxiv.org/abs/2609.28810](https://arxiv.org/abs/2609.28810)

    该论文提出基于类别的多层次建模方法，将幂类型非传递性的成员关系语义与超类的深度特征化能力统一起来，从而解决了现有MLM方法中instance-of关系与深度特征化之间的语义矛盾。

    

    多层次软件建模（MultiLevel Modeling，MLM）认为，在广泛的学科领域中进行概念建模可能需要对多个分类层次进行抽象。MLM方法依赖于哲学论证，主张对现实世界领域的忠实建模涉及如自然种类本体论中那样的重复类型分类。MLM的分层架构由位于较低层次的clabject类与位于较高层次、被称为类别类（category classes）的类之间的跨层次instance-of（实例）关系交织并定义而成。instance-of关系表示clabject作为类型对象在其（幂类型）类别类中的成员身份，且不具有传递性。所有MLM方法都支持某种形式的深度特征化（deep characterization），即类别类能够影响较低层次中的类。深度特征化是超类的本质特征，这与instance-of所表达的非传递性成员关系含义相矛盾。在本文中，我们提出了基于类别的MLM（Category-Based MLM）……

    arXiv:2609.28810v1 Announce Type: new  Abstract: MultiLevel software Modeling (MLM) suggests that conceptual modeling in broad subject domains might require abstraction of multiple classification levels. The MLM approach relies on philosophical arguments, claiming that faithful modeling of real-world domains involves repeated type classification as in ontologies of natural kinds. MLM leveled architecture is interwoven and defined by instance-of interlevel relationships between clabject classes in lower levels to classes termed category classes, in upper levels. The instance-of relation denotes membership of clabjects as type objects in their (powertypes) category classes, and is not transitive.   All MLM approaches support forms of deep characterization, i.e., category classes can influence classes in lower levels. Deep characterization is an essential feature of superclasses and contradicts the non-transitive membership meaning of instance-of.   In this paper, we introduce the Categor
    
[^39]: 污点流模型的可靠性验证

    Soundness Checking of Taint Flow Models

    [https://arxiv.org/abs/2609.28750](https://arxiv.org/abs/2609.28750)

    提出一种“猜测-验证”方法，利用LLM智能体生成方法的精确污点流模型，并通过符号算法结合轻量级静态分析递归验证该模型的可靠性。

    

    现有最先进的针对命令式编程语言的静态污点流分析，能够通过使用用户提供的高精度库方法污点流模型来扩展到大型应用程序。然而，手动精确地对方法的污点流进行建模既繁琐又可能不可靠。此外，通过过程间污点分析自动对方法建模可能效率低下。为了解决这个问题，我们提出了一种“猜测-验证”方法：（1）一个LLM智能体，用于生成方法的精确污点流模型；（2）一个符号算法，用于检查模型的可靠性。该算法推导出为了使LLM的污点流模型可靠，方法中哪些污点流必须不出现，并使用轻量级静态分析（例如类型系统和指针分析）来证明这些“必不流”。当这些分析不够充分时，该算法会推导出最大泛化的被调用者模型，并递归地验证其可靠性。

    arXiv:2609.28750v1 Announce Type: new  Abstract: Existing state-of-the-art static taint flow analyses for imperative programming languages can scale to large applications by using precise user-provided taint flow models of library methods. However, manually and precisely modeling a method's taint flows is tedious and potentially unsound. Furthermore, automatically modeling the method via an inter- procedural taint analysis can be inefficient. To solve this problem, we propose a guess-and-check approach: (1) an LLM agent that generates a precise taint flow model of a method and (2) a symbolic algorithm to check the soundness of the model. The algorithm deduces which taint flows must not occur in the method for the LLM's taint flow model to be sound, and uses lightweight static analyses (e.g., type system and pointer analysis) to prove these must-not-flows. When these analyses are insufficient, the algorithm deduces maximally-general callee models and recursively verifies their soundness
    
[^40]: 迈向掌握个人主权的平台

    Towards a Platform for Mastering Personal Sovereignty

    [https://arxiv.org/abs/2609.28736](https://arxiv.org/abs/2609.28736)

    提出名为"MyVirtualME"的虚拟可信平台，代表个人与企业和行政机构交互，将服务、数据和权限以人为中心进行组织，帮助个人重获数字时代的主权。

    

    正在进行的工作、行政、健康、出行和社交互动的数字化转型正在深刻重塑日常生活，持续将控制权从个人转移至大型平台提供商。尽管数据常被称为“21世纪的黄金”，但其真正的价值是通过访问、组合和利用数据的服务来实现的。如今，个人几乎没有主权：生活中的各类事件（例如更改地址、保险、工作或婚姻状况）需要在众多系统中进行碎片化、重复性的交互，使用户感到不堪重负而非获得赋能。我们主张对这一状况进行根本性的重新思考，并提出一个虚拟的、可信赖的平台，称为"MyVirtualME"，它代表个人与公司、行政机构及其他主体进行交互。该平台将服务、数据、权限和数据使用以人为中心（而非以外部主体为中心）进行组织，从而实现真正的控制与自主。

    arXiv:2609.28736v1 Announce Type: cross  Abstract: The ongoing digital transformation of work, administration, health, mobility, and social interaction is profoundly reshaping everyday life, steadily shifting control from individuals to large platform providers. Although data is often labeled the "gold of the 21st century", its real value is realized through services that access, combine, and exploit it. Today, individuals have little sovereignty: life events (e.g., changing an address, insurance, job, or marital status) require fragmented, repetitive interactions across numerous systems, leaving users overwhelmed rather than empowered. We argue for a fundamental rethinking of this situation and propose a virtual, trustworthy platform, called "MyVirtualME" that acts on behalf of the human individual towards companies, administrations, and other actors. This platform centers services, data, permissions, and data usage around humans, not foreign actors, allowing genuine control and trans
    
[^41]: CONCURDEP：CPython 并发中依赖失效的事件引导分析

    CONCURDEP: Event-Guided Analysis of Dependency Invalidation in CPython Concurrency

    [https://arxiv.org/abs/2609.28608](https://arxiv.org/abs/2609.28608)

    CONCURDEP 提出一种源码级静态分析方法，通过事件感知的原生并发依赖图将运行时语义依赖与并发或重入事件关联，从而检测 CPython 移除 GIL 后原生代码中依赖失效引发的安全问题。

    

    移除 CPython 的全局解释器锁（GIL）使原生代码暴露于普通 C 类型所不具备的并发环境。突变或重入操作可能在获取与使用之间撤销一个被借用的对象、存储指针、遍历状态或租约，而其所有者仍然存活，从而导致原生内存错误和运行时状态损坏。竞争分析追踪冲突访问，Python/C 生命周期分析追踪单个对象的状态。这些报告单元使得隐式的所有者-主体-存储关系在并行和重入事件下与后续使用相互脱节。我们提出了 CONCURDEP，一种针对依赖失效的源码级静态分析。其关键洞察是：表示原生使用所依赖的运行时属性，并询问哪个与目标匹配的事件可以在该依赖的活跃区域内将其撤销。CONCURDEP 恢复运行时语义依赖，通过一个事件感知的原生并发依赖图将它们与事件连接起来，并应用……（摘要在此处截断）

    arXiv:2609.28608v1 Announce Type: cross  Abstract: Removing CPython's Global Interpreter Lock (GIL) exposes native code to concurrency absent from ordinary C types. Mutation or re-entry can revoke a borrowed object, storage pointer, traversal state, or lease between acquisition and use while its owner remains alive, causing native memory errors and runtime-state corruption. Race analyses track conflicting accesses. Python/C lifecycle analyses track individual object states. These reporting units leave implicit owner-subject-storage relations disconnected from later uses under parallel and re-entrant events. We present CONCURDEP, a source-level static analysis of dependency invalidation. Its key insight is to represent the runtime property a native use requires and ask which target-matched event can revoke it within the dependency's live region. CONCURDEP recovers runtime-semantic dependencies, connects them to events through an event-aware native concurrency dependency graph, and appli
    
[^42]: 在喷气推进实验室（JPL）开发统一的验证与确认活动标准

    Developing a Unified Verification and Validation Activity Standard at JPL

    [https://arxiv.org/abs/2609.28600](https://arxiv.org/abs/2609.28600)

    该论文通过与29名不同学科从业者开展以人为本的设计研讨会，为JPL开发了统一的验证与确认活动标准模式，以平台无关的SysML模型形式化，在分离五种验证方法的同时维护共同属性集，并在Jama平台中实施以平衡严谨性与敏捷性。

    

    NASA喷气推进实验室（JPL）的验证与确认实践在过去十年中出现了分化，由此产生的碎片化问题增加了开销、降低了跨项目效率，并阻碍了机构知识的传递。我们提出了一个统一的V&V活动模式，该模式是通过以人为本的设计研讨会开发的，共有29名来自多种任务类型和学科的从业者参与。该模式建立在基于关系的架构之上，允许分离各种方法（测试、分析、检查、演示和设计评审），同时维护一个共同的属性集。该模式被形式化为一个与平台无关的SysML模型，定义了需求、V&V活动、场所以及证据之间的双向关系。在JPL的Jama平台中的实施展示了通过模板和模块化项目类型实现的受控定制，在支持自动化的同时平衡了严谨性与敏捷性……

    arXiv:2609.28600v1 Announce Type: new  Abstract: Verification and validation practices (V&V) at NASA's Jet Propulsion Laboratory (JPL) have diverged over the past decade, creating fragmentation that increases overhead, reduces cross-project efficiencies, and inhibits institutional knowledge transfer. We present a unified V&V activity schema developed through human-centered design workshops involving 29 practitioners across multiple mission types and disciplines. The schema builds on a relationship-based architecture that allows for separating methods (Test, Analysis, Inspection, Demonstration, and Review of Design) while maintaining a common attribute set. Formalized as a platform-agnostic SysML model, the schema defines bidirectional relationships between requirements, V&V activities, venues, and evidence. Implementation in JPL's Jama platform demonstrates controlled customization through templates and modular item types, balancing rigor with agility while enabling automations, patter
    
[^43]: 智能体批准洗白：超越被批准调用的传递性效应

    Agent Approval Laundering: Transitive Effects Beyond the Approved Invocation

    [https://arxiv.org/abs/2609.28586](https://arxiv.org/abs/2609.28586)

    该论文首次系统分析了智能体批准接口中“审批记录只覆盖入口调用、遗漏工作流传递效应”的批准洗白问题，形式化了六类效应上的闭包绑定批准并证明仅凭记录的策略存在信息极限，同时提出了将批准对象与执行后证据绑定的 Approval-to-Action 安全基准。

    

    编码智能体的批准接口将人类的决策绑定到某条命令或工具调用上，而开发者工具则会执行该调用所激活的传递性工作流。软件包安装可以运行生命周期钩子并写入文件；一次 MCP 调用可以行使网络权限。我们将由此产生的记录覆盖缺口称为“批准洗白”：持久的审批记录只载明了入口调用，却遗漏了其工作流所行使的效应。我们对智能体系统中这种“记录—闭包”关系进行了首次系统性安全分析，在六类效应上形式化了“闭包绑定批准”，并推导出一个信息极限：相同的策略可见字段可能需要不同的、依赖具体效应的决策，因此任何仅基于记录的策略都无法同时保证两者。我们提出“批准—行动安全基准”，将批准对象与决策时元数据绑定到执行后证据上。在 111 个固定的批准对象/追踪轨迹配对中，残留记录从 40……（原文摘要在此处被截断）

    arXiv:2609.28586v1 Announce Type: cross  Abstract: Coding-agent approval interfaces bind a human decision to a command or tool call, while developer tools execute the transitive workflow that invocation activates. Package installation can run lifecycle hooks and write files; an MCP call can exercise network authority. We call the resulting record-coverage failure approval laundering: the durable record names the entry invocation but omits effects exercised by its workflow.   We present the first systematic security analysis of this record-to-closure relation in agent systems. We formalize closure-bound approval over six effect classes and derive an information limit: identical policy-visible fields can require different effect-specific decisions, so no record-only policy can guarantee both. The Approval-to-Action Security Benchmark binds approval objects and decision-time metadata to post-execution evidence.   Across 111 fixed approval-object/trace pairs, residual records fall from 40 
    
[^44]: 理解数据修订的时间序列基础模型

    Time-Series Foundation Models That Understand Data Revisions

    [https://arxiv.org/abs/2609.28576](https://arxiv.org/abs/2609.28576)

    该论文提出修订感知的时间序列基础模型 VINTAGE-TS，通过区分观测时间与信息可用时间、预测首次发布值并使用联合预测分布，解决了使用修订后数据评估历史预测时可能产生的前视偏差问题。

    

    历史观测数据并非总是固定不变的：统计机构会随着新证据的到来而修订先前发布的数值。因此，基于当下下载数据进行的预测，可能会使模型接触到在其所谓做出预测的日期时尚不可用的信息。我们提出了 VINTAGE-TS，这是一种对时间序列基础模型的修订感知适配方法，它区分了观测时间与信息可用时间。其预测目标是下一期的首次发布值以及该发布之后固定天数内可获得的数值；两者均不被声明为最终真实值。联合预测分布保留了这些预测目标之间的依赖关系，并揭示了两者差异的不确定性。我们规定了基于 ALFRED 的滚动评估、匹配的 Chronos-2 对比、常规基线与修订感知基线，以及一项针对预训练数据重叠情况的独立审计。随附软件实现了有效性区间重建、删……

    arXiv:2609.28576v1 Announce Type: new  Abstract: Historical observations are not always fixed: statistical agencies revise previously published values as new evidence arrives. Forecasting from a contemporary download can therefore expose a model to information unavailable at the date it purportedly made a prediction. We propose VINTAGE-TS, a revision-aware adaptation of a time-series foundation model that distinguishes observation time from information-availability time. Its targets are the next period's first-published value and the value available a fixed number of days after that publication; neither is declared final truth. A joint predictive distribution preserves dependence between these targets and exposes uncertainty about their difference. We specify an ALFRED-based rolling evaluation, a matched Chronos-2 comparison, conventional and revision-aware baselines, and a separate audit of pretraining overlap. The accompanying software implements validity-interval reconstruction, del
    
[^45]: 变更溯源监督：策略变更下学习产物的治理

    Change-Provenant Supervision: Governing Learned Artifacts Under Policy Change

    [https://arxiv.org/abs/2609.28574](https://arxiv.org/abs/2609.28574)

    论文提出“变更溯源监督”框架，将记录的依赖谱系与独立的当前契约重验证相分离，借助字节级重建和密封契约账本，能够在权限变更后拒绝那些输出与合规产物完全相同、但来源已过期的学习产物（如 LoRA 适配器）。

    

    记录在案的依赖图无法证明自身不存在遗漏的边。因此，对于学习产物，仅凭图范围的失效机制无法在权限变更后为准入提供正当性依据，尤其当输出测试未能发现来源过期的派生时。我们将“记录的谱系”——用于提出影响范围与依赖解释——与“对每个保留目标进行独立的当前契约重验证”——提供相对于所声明契约的可靠性——区分开来。我们在两个证据层评估了这一设计：四个 Qwen3-14B LoRA 包将 1.028 GB 的适配器字节与训练及权限记录绑定；在互斥的版本化监督数据上训练的“过期”适配器与“全新鲜”适配器，在全部 80 个测试的权限中立输入上输出了完全相同的计划。尽管如此，基于字节的重建与密封的当前契约账本仍然拒绝了过期包与朴素追加包，并准入了全新鲜包与谱系选择性包。

    arXiv:2609.28574v1 Announce Type: new  Abstract: A recorded dependency graph cannot certify that it contains no omitted edge. For learned artifacts, graph-scoped invalidation therefore cannot by itself justify admission after authority changes, especially when output testing misses a provenance-stale derivation. We separate recorded lineage, which proposes impact scope and a dependency explanation, from independent current-contract revalidation of every retained target, which supplies soundness relative to the declared contract.   We evaluate this design in two evidence layers. Four Qwen3-14B LoRA packages bind 1.028 GB of adapter bytes to training and authority records. Stale and full-fresh adapters trained on mutually exclusive versioned supervision emitted identical plans on all 80 tested authority-neutral inputs. Byte-derived reconstruction and a sealed current-contract ledger nevertheless refused the stale and naive-append packages and admitted full-fresh and lineage-selective pac
    
[^46]: 网络智能体的困境之处：多阶段LLM智能体的瓶颈分析

    Where Cyber Agents Struggle: Bottleneck Analysis of Multi-Stage LLM Agents

    [https://arxiv.org/abs/2609.28572](https://arxiv.org/abs/2609.28572)

    本文通过端到端诊断研究揭示多阶段LLM网络智能体在自主攻击中的瓶颈，发现仅靠成功率会掩盖低效与证据误判问题，并提出成本感知评分与LLM-as-a-Judge分析来系统识别规划缺陷。

    

    基于LLM的多阶段网络智能体或许能够完成攻击工作流，但仍然存在脆弱、高成本或依赖对执行证据错误解读的问题。仅凭成功率会掩盖低效性、通过重试实现的适应能力以及对成败的识别水平。我们提出了一项针对自主攻击系统的端到端诊断研究，该系统包含编排器、执行器和验证器LLM，应用于类企业环境的横向移动场景。我们在两个场景和三种模式（专家定义、自搭建脚手架和完全自主）下评估了六个前沿模型。我们评估了验证器的一致性和证据锚定能力；引入了一种基于子任务条件、成本感知的评分机制，用于衡量异常的token消耗、重试次数和运行时间；并采用对比性LLM-as-a-Judge分析来识别规划缺陷，包括工具错位、计划相似、过度规格化、探测不足和恢复能力薄弱。验证器通常具有相关性并以证据为依据……

    arXiv:2609.28572v1 Announce Type: cross  Abstract: Multi-stage LLM-based cyber agents may complete attack workflows while remaining brittle, costly, or reliant on incorrect interpretations of execution evidence. Success rates alone obscure inefficiency, adaptation through retries, and recognition of success or failure. We present an end-to-end diagnostic study of an Autonomous Adversary system with orchestrator, executor, and validator LLMs in enterprise-like lateral-movement scenarios. Six frontier models are evaluated across two scenarios and three modes: expert-defined, self-scaffolded, and fully autonomous. We assess validator consistency and evidence grounding; introduce a subtask-conditioned, cost-aware score for abnormal token use, retries, and runtime; and use comparative LLM-as-a-Judge analysis to identify planning deficiencies, including tool misalignment, plan similarity, over-specification, inadequate probing, and weak recovery. Validators are generally relevant and evidenc
    
[^47]: 在无依赖软件栈上预训练与适配语言模型：从随机权重训练的GPT-2 124M及其与llm.c的对照复现，以及面向Qwen3-0.6B的临床适配器

    Pretraining and adapting a language model on a dependency-free stack: GPT-2 124M from random weights, reproduced against llm.c, and a clinical adapter for Qwen3-0.6B

    [https://arxiv.org/abs/2609.28568](https://arxiv.org/abs/2609.28568)

    该论文用零第三方依赖的Zig机器学习栈numbat从随机权重独立复现了GPT-2 124M的完整预训练生命周期，各项指标与llm.c参考实现高度吻合，从而厘清了“语言模型本身的知识”与“特定训练软件的知识”，并进一步演示了Qwen3-0.6B模型的临床问答适配。

    

    几乎所有在用的语言模型都是由同一个软件家族训练的。这种集中性使得一个难以回答的问题悬而未决：关于训练语言模型的已有知识中，有多少描述的是语言模型本身，又有多少描述的是那个软件？要回答这个问题，需要第二个能够支撑模型完整生命周期（而非仅复现某个算子）的独立实现。我们报告了这样一个完整生命周期。使用numbat——一个用Zig编写、无第三方运行时依赖的机器学习软件栈——我们从随机初始化开始，在99.1亿词元的网络文本上预训练一个1.244亿参数的GPT-2模型，随后将另一个小模型适配到临床问答任务。在两个阶段中，一个参考实现在相同硬件上并行运行，且有一个有权终止训练运行的伴随进程对每次运行进行监督。两者的结果高度一致：留出集交叉熵最终达到3.2588（对比已发表的3.29），HellaSwag得分0.3053（对比0.299）；在8组配对评估中……（摘要在此截断）

    arXiv:2609.28568v1 Announce Type: new  Abstract: Almost every language model in service was trained by one family of software. That concentration makes a question hard to settle: how much of what is known about training a language model describes language models, and how much describes that software? Settling it needs a second implementation able to carry a model through a whole lifecycle rather than reproduce one operator.   We report such a lifecycle. Using numbat, a machine-learning stack written in Zig with no third-party runtime dependencies, we pretrain a 124.4 M-parameter GPT-2 from random initialisation over 9.91 B tokens of web text, then adapt a separate small model to clinical question answering. A reference implementation runs on identical hardware at both stages, and a sidecar with authority to halt a run supervises each.   Agreement is close. Held-out cross-entropy finishes at 3.2588 against a published 3.29, and HellaSwag at 0.3053 against 0.299; across 8 paired evaluati
    
[^48]: 谁在框架背后？通过智能体行为对大语言模型进行指纹识别

    Who Is Behind the Harness? Fingerprinting LLMs through Agentic Behavior

    [https://arxiv.org/abs/2609.28559](https://arxiv.org/abs/2609.28559)

    提出了一种名为LIDAR的主动式黑盒指纹识别方法，通过编码智能体在运行时的决策与行动（如编辑后验证、故障恢复和规范-测试冲突处理等行为）来识别框架背后的大语言模型身份。

    

    大语言模型越来越多地通过编码智能体框架运行，这些框架会检查代码仓库、调用工具并修改文件。因此，替换此类智能体背后的模型可能会改变与安全相关的决策，包括它是否会验证更改或从故障中安全恢复。现有的LLM指纹识别方法主要是从直接文本或token分布推断模型身份。而在编码智能体中，这些信号会受到系统指令、控制器逻辑、工具和执行反馈的调节，限制了它们的迁移能力。我们提出了LIDAR（基于运行时决策与行动的LLM识别），这是一种针对编码智能体执行的主动式黑盒指纹识别方法。三个编码探测对在受控变更下揭示编辑后验证、瞬时故障恢复以及规范与测试冲突的解决行为。LIDAR使用互补的实例级和分布级特征来表示所产生的轨迹，并将其与（原文摘要在此处截断）

    arXiv:2609.28559v1 Announce Type: cross  Abstract: LLMs increasingly operate through coding-agent harnesses that inspect repositories, invoke tools, and modify files. Substituting the model behind such an agent can therefore change security-relevant decisions, including whether it verifies changes or recovers safely from failures. Existing LLM fingerprints largely infer identity from direct text or token distributions. In coding agents, these signals are mediated by system instructions, controller logic, tools, and execution feedback, limiting their transfer.   We present LIDAR (LLM Identification from Decisions and Actions at Runtime), an active black-box fingerprinting method for coding-agent execution. Three coding probe pairs expose post-edit verification, transient-failure recovery, and specification--test conflict resolution under controlled changes. LIDAR represents the resulting trajectories with complementary instance-level and distribution-level features and compares them wit
    
[^49]: 使用证明分片与探索技术合成证明

    Synthesizing Proofs Using Proof Sharding and Exploration

    [https://arxiv.org/abs/2609.28535](https://arxiv.org/abs/2609.28535)

    提出了自动化工具ProofSaX，通过将大型验证任务分片并独立探索每个分片的证明搜索空间，减少人工干预，从而实现分布式系统正确性证明的规模化自动合成。

    

    分布式系统难以正确实现，微妙的漏洞往往无法通过传统测试检测出来。形式化验证为证明复杂分布式系统的正确性提供了一种替代方案。尽管此前已有许多使形式化验证自动化和简便化的努力，但在软件开发中集成形式化验证仍然十分困难。程序员需要反复查询定理证明器才能找到其系统的正确证明。这种与定理证明器来回交互的过程涉及大量的人工干预，成为形式化验证在软件开发中推广采用的障碍。在本文中，我们通过减少寻找分布式系统正确证明所需的人工干预，来应对形式化验证在实践中规模化应用的挑战。我们提出了ProofSaX，这是一个自动化工具，它将大型验证任务进行分片，并对每个分片的证明搜索空间进行独立探索。

    arXiv:2609.28535v1 Announce Type: cross  Abstract: Distributed systems are hard to implement correctly, and subtle bugs can go undetected using traditional testing. Formal verification offers an alternative for proving the correctness of complex distributed systems. Despite previous efforts to automate and facilitate formal verification, it is still hard to integrate formal verification in software development. Programmers need to query the theorem prover repeatedly to find the correct proof of their system. This cycle of going back and forth with the theorem prover involves a lot of human intervention and is a barrier to adopting formal verification in software development.   In this paper, we address the challenges of scaling formal verification in practice by reducing the human intervention required to find the correctness proof of a distributed system. We propose ProofSaX, an automated tool that shards large verification tasks and explores the proof search space for each shard inde
    
[^50]: IaC-Guard-V：一个针对LLM生成的基础设施即代码修复的验证框架

    IaC-Guard-V: A Verification Framework for LLM-Generated Infrastructure-as-Code Repairs

    [https://arxiv.org/abs/2609.28488](https://arxiv.org/abs/2609.28488)

    IaC-Guard-V是一个以验证为中心的框架，从语法有效性、目标问题解决、回归安全性和补丁最小性四个维度系统评估LLM生成的IaC修复，并基于70个真实Terraform和Kubernetes配置错误工件构建基准进行实验。

    

    基础设施即代码配置错误是云安全事件的主要成因之一，而大型语言模型（LLM）正日益被提议作为自动化修复智能体。然而，LLM生成的IaC修复的可信度仍未得到充分研究。IaC带来了独特的验证挑战：安全扫描器在不同工具和配置下可能表现各异，基础设施语义无法通过普通的单元测试来验证，且云服务商特定的规则造成了碎片化的验证环境。我们提出了IaC-Guard-V，一个以验证为中心的框架，通过四个维度评估AI生成的IaC修复：语法有效性、目标问题解决、回归安全性和补丁最小性。我们构建了一个包含70个真实世界中配置错误的Terraform和Kubernetes工件的基准数据集，涵盖70条独特的扫描器规则和八类违规，并在三个LLM家族上评估了三种修复策略。

    arXiv:2609.28488v1 Announce Type: new  Abstract: Infrastructure-as-Code (IaC) misconfigurations are a leading cause of cloud security incidents, and Large Language Models (LLMs) are increasingly proposed as automated repair agents. Yet the trustworthiness of LLM-generated IaC repairs remains underexplored. IaC presents distinctive verification challenges: security scanners may behave differently across tools and configurations, infrastructure semantics cannot be validated through ordinary unit tests, and provider-specific rules create a fragmented verification landscape. We present IaC-Guard-V, a verification-centered framework that evaluates AI-generated IaC repairs through four dimensions: syntactic validity, target-issue resolution, regression safety, and patch minimality. We construct a benchmark of 70 real-world misconfigured Terraform and Kubernetes artifacts spanning 70 unique scanner rules and eight violation classes, and evaluate three repair strategies across three LLM famili
    
[^51]: 生成式人工智能可能会在软件工程教育中强化社会偏见

    Generative AI May Reinforce Social Biases in Software Engineering Education

    [https://arxiv.org/abs/2609.28483](https://arxiv.org/abs/2609.28483)

    本研究揭示了软件工程教师在利用生成式AI进行团队组建和教育材料生成时会无意中强化性别与国籍等社会偏见，例如女性更容易被分配到前端开发角色。

    

    生成式人工智能正日益广泛地应用于各类现实场景中。若不加谨慎评估，对这些系统的依赖可能产生意想不到的后果，例如刻板印象的强化和社会偏见的放大。此类风险在教育环境中尤为关键，因为早期的决策会塑造学生的兴趣、机会和职业发展轨迹。在本文中，我们研究了软件工程教师使用生成式AI可能会如何无意中强化软件领域特有的社会偏见。我们聚焦于两项代表性任务：基于学生档案的团队组建，以及教育材料视觉内容的生成。我们的结果揭示了这两项任务中存在显著偏见。在团队组建方面，性别和国籍等因素会影响角色分配（例如，与能力相当男性相比，女性更可能被分配到前端角色）。

    arXiv:2609.28483v1 Announce Type: cross  Abstract: Generative artificial intelligence (GenAI) is increasingly being deployed across a wide range of real-world applications. Without careful evaluation, reliance on these systems can have unintended consequences, such as reinforcement of stereotypes and amplification of social biases. Such risks are particularly important in educational settings, where early decisions can shape students' interests, opportunities, and career trajectories. In this paper, we investigate how GenAI use by software engineering instructors may inadvertently reinforce software-specific social biases. We focus on two representative tasks: team formation based on student profiles and the generation of visual content for educational materials. Our results reveal significant biases in both tasks. In team formation, factors such as gender and nationality affect role assignments (e.g., women are more likely to be assigned to front-end roles than equally qualified men).
    
[^52]: Drafter：一个用于CS1（计算机科学导论）课程全栈Web开发的Python库

    Drafter: A Python Library for Full-Stack Web Development in CS1

    [https://arxiv.org/abs/2609.28481](https://arxiv.org/abs/2609.28481)

    本文提出了面向CS1入门课程的开源Python库Drafter，让学生能够使用纯函数和最少的样板代码开发全栈Web应用，无需学习HTML或模板语言，从而将Web开发融入计算机科学入门教学。

    

    Web应用程序日益成为创建用户界面的主要方式，它们往往是初学者最常接触的软件。然而，Web开发依赖于HTML、CSS、JavaScript和后端技术等概念，这些挑战超出了入门课程的教学范围。此外，现代Web框架的许多特性与CS1的教学原则相冲突，例如避免全局可变状态和促进测试驱动开发。因此，尽管Web开发具有激发学习动机的潜力，但它很少被整合到CS1课程中。本文介绍了Drafter，一个面向CS1课程的新型开源Python库，使学生能够使用纯函数和最少的样板代码开发全栈Web应用程序。Drafter提供了生成HTML的简单函数，从而免去了学习HTML或模板语言的需要。其数据模型能够随学生对类型掌握程度的提高而扩展，从基本类型到嵌套列表、数据类（原文在此处截断）。

    arXiv:2609.28481v1 Announce Type: cross  Abstract: Web applications are increasingly the main way to create user interfaces, and they are often the most common software beginners have encountered. However, web development relies on concepts like HTML, CSS, JavaScript, and backend technologies, which are challenges beyond the scope of an introductory course. Additionally, many features of modern web frameworks conflict with CS1 principles, such as avoiding global mutable state and promoting test-driven development. Consequently, web development is rarely integrated into CS1 courses, despite its motivational potential. This paper introduces Drafter, a new open-source Python library for CS1, enabling students to develop full-stack web applications using pure functions with minimal boilerplate. Drafter has simple functions to generate HTML, eliminating the need to learn HTML or templating languages. The data model scales with students' mastery of types, from primitives to nested lists, Dat
    
[^53]: 谁来完成收尾工作？关于AI编码智能体拉取请求的后续修复与提交作者归属研究

    Who Finishes the Job? A Study of Follow-Up Fixes and Commit Authorship on AI Coding Agent Pull Requests

    [https://arxiv.org/abs/2609.26847](https://arxiv.org/abs/2609.26847)

    本研究通过跟踪五个主流AI编码智能体的6,774个已合并PR并与5,044个人类PR对比，首次系统揭示了AI智能体拉取请求合并后需要多少后续修复、由谁（智能体还是人类）来完成这些收尾工作。

    

    AI编码智能体现在已经贡献了合并到热门开源项目中的拉取请求（PR）的很大一部分。一个已合并的智能体PR通常被视为已完成的工作；然而，先前的研究已报告了智能体代码在合并后存在的问题（例如代码异味和静态分析问题）。但是，对于已合并的智能体PR随后被修复的频率，以及实际由谁来完成这些修复，人们知之甚少。在本文中，我们跟踪了来自AIDev-pop数据集（拥有至少500颗星的开源仓库）的6,774个已合并的智能体PR，涵盖五个AI编码智能体（OpenAI Codex、GitHub Copilot、Devin、Cursor和Claude Code），追踪其后续修复情况，并以来自相同仓库的5,044个同期人类PR作为基线进行对比。我们将每次合并与其候选修复进行关联，通过人工标注者和一个达到人类一致性水平的LLM裁判来验证每个候选修复（二元Cohen's Kappa为0.78，而人与人之间的一致性K为0.77，直接修复的精确率为90%）……

    arXiv:2609.26847v1 Announce Type: new  Abstract: AI coding agents now author a large share of pull requests (PRs) merged into popular open-source projects. A merged agent PR is usually considered finished work; yet, prior studies have reported issues in agent code after the merge (e.g., code smells and static-analysis issues). However, little is known about how often a merged agent PR is fixed afterward, and who actually authors the fixing. In this paper, we follow 6,774 merged agent PRs across five AI coding agents (OpenAI Codex, GitHub Copilot, Devin, Cursor, and Claude Code) from the AIDev-pop dataset (open-source repositories with at least 500 stars) into their follow-up fixes, against a baseline of 5,044 contemporaneous human PRs from the same repositories. We link each merge to its candidate fixes, verify every candidate with human annotators and an LLM judge that matches human-level agreement (binary Cohen's Kappa=0.78$ against a human-human K=0.77$, Direct-fix precision 90%), a
    
[^54]: 扩展框架而非上下文：从无策略脚手架到可复用的专家智能体

    Grow the Harness, Not the Context: From Strategy-Free Scaffolds to Reusable Specialist Agents

    [https://arxiv.org/abs/2609.26760](https://arxiv.org/abs/2609.26760)

    该论文提出 Growing Harness 训练范式，通过失败定位、联合修复与成功优先门控，把任务反馈中反复出现的控制逻辑自动沉淀为可复用的可执行代码，让智能体框架本身从任务交互中“生长”出来，而 LLM 只需专注于任务特定的语义推理。

    

    大语言模型（LLM）智能体通常需要处理一系列相关任务，然而标准的智能体框架反复要求模型在每个任务的上下文中重新构建相同的控制决策。我们研究能否转而利用任务反馈，将反复出现的控制逻辑转化为可复用的可执行代码，同时将 LLM 调用保留用于任务特定的语义推理。我们提出 Growing Harness（生长式框架），一种由失败引导的训练范式，它从一个无策略的脚手架中学习智能体框架本身；该脚手架仅暴露固定的模型与工具接口，而不编码任何任务求解控制器。函数级别的执行轨迹将每次失败定位到有界的代码范围内，优化器联合修复一个失败窗口，而以成功为先的保留集门控会回滚损害既有能力的修复序列。被接受的修改不断累积到同一个共享框架中，使其控制结构从任务反馈中自然涌现。在 BrowseComp-Plus 和 WebArena-Verified 上的实验（摘要此处截断）……

    arXiv:2609.26760v1 Announce Type: cross  Abstract: Large language model (LLM) agents often handle streams of related tasks, yet standard harnesses repeatedly ask the model to reconstruct the same control decisions inside each task's context. We study whether task feedback can instead turn recurring control into reusable executable code, while reserving LLM calls for task-specific semantic reasoning. We introduce Growing Harness, a failure-guided training paradigm that learns the agent harness itself from a strategy-free scaffold that exposes fixed model and tool interfaces but encodes no task-solving controller. Function-level execution traces localize each failure to a bounded code surface, an optimizer repairs a window of failures jointly, and a success-first held-out gate rolls back repair sequences that harm prior capability. Accepted edits accumulate in one shared harness, allowing its control structure to emerge from task feedback. Across BrowseComp-Plus and WebArena-Verified wit
    
[^55]: Python项目跨操作系统可移植性问题的实证分析

    An Empirical Analysis of Cross-OS Portability Issues in Python Projects

    [https://arxiv.org/abs/2609.25531](https://arxiv.org/abs/2609.25531)

    该论文开展了首个针对Python跨操作系统可移植性问题的大规模实证研究，分析了2,042个开源仓库，构建了包含7个主要故障类别、24个子类别、15个诊断特征和4种系统性修复模式的全面分类体系。

    

    尽管Python被设计为一种跨平台语言，但实际应用在部署到不同操作系统时会遇到可移植性故障。我们提出了首个针对Python跨操作系统可移植性问题的大规模实证研究，采用两种互补的方法分析了2,042个开源仓库：系统性的跨操作系统测试重执行以及对GitHub issue的人工分析。我们对500个项目的跨平台测试显示，11.2%的项目存在依赖于操作系统的测试失败。通过对240个GitHub issue的系统性分析，我们确认了102个真实的可移植性问题，涉及另外95个项目。我们构建了一个全面的分类体系，识别出7个主要故障类别——其中文件/目录操作、进程管理和库依赖最为普遍——以及24个不同的子类别、15个诊断特征和4种系统性修复模式。我们的评估表明，现有的静态分析工具（摘要在此处被截断）

    arXiv:2609.25531v1 Announce Type: new  Abstract: While Python is designed as a cross-platform language, real-world applications encounter portability failures when deployed across different operating systems. We present the first large-scale empirical study of cross-OS portability issues in Python, analyzing 2,042 open-source repositories using two complementary approaches: systematic cross-OS test reexecution and manual analysis of GitHub issues. Our cross-platform testing of 500 projects reveals that 11.2% exhibit OS-dependent test failures. Through systematic analysis of 240 GitHub issues, we confirm 102 genuine portability problems spanning 95 additional projects. We develop a comprehensive taxonomy identifying 7 primary failure categories - with file/directory operations, process management, and library dependencies being most prevalent - along with 24 distinct sub-categories, 15 diagnostic signatures, and 4 systematic repair patterns. Our evaluation reveals that existing static a
    
[^56]: Swift语言中不稳定测试的词汇特征

    The Vocabulary of Flaky Tests in Swift

    [https://arxiv.org/abs/2609.25516](https://arxiv.org/abs/2609.25516)

    该研究首次针对Swift语言评估了基于词汇的机器学习方法来预测不稳定测试，通过从15个开源项目收集数据并训练五个分类器，证明随机森林模型（F1=0.86，MCC=0.75）能够利用测试词汇特征有效识别不稳定测试，并显著优于简单基线方法。

    

    不稳定测试（Flaky tests）在代码未变更的情况下产生非确定性结果，削弱了持续集成（CI）的信心并延误交付。尽管基于词汇的机器学习预测方法已被证明对Java和JavaScript有效，但尚无研究针对Swift评估该方法，而Swift是一种测试风格以UI和异步代码为主的语言。我们通过重复执行和提交历史挖掘，从15个开源Swift项目中收集了91个不稳定测试和22,349个稳定测试，然后在TF-IDF一元加二元词组特征上，采用分层5折交叉验证训练了五个分类器（随机森林、决策树、朴素贝叶斯、支持向量机、K近邻）。随机森林取得了最佳性能（精确率=0.92，F1=0.86，AUC=0.95），并显著优于简单的基线方法，其中包括对信息量最大的词汇应用词汇阈值规则的方法，从而证实了真实的判别信号（MCC=0.75，而最佳基线仅为0.08）。信息增益分析揭示了两个互补的（摘要在此处被截断）

    arXiv:2609.25516v1 Announce Type: new  Abstract: Flaky tests produce non-deterministic outcomes without code change, eroding CI confidence and delaying deliveries. While vocabulary-based machine learning prediction has proven effective for Java and JavaScript, no study has evaluated it for Swift, a language whose testing style is dominated by UI and asynchronous code. We collect 91 flaky and 22,349 stable tests from 15 open-source Swift projects via re-execution and commit-history mining, then train five classifiers (Random Forest, Decision Tree, Naive Bayes, SVM, KNN) on TF-IDF unigram+bigram features under stratified 5-fold cross-validation. Random Forest achieves the best performance (Precision = 0.92, F1 = 0.86, AUC = 0.95) and substantially outperforms trivial baselines, among them a vocabulary-threshold rule applied to the most informative tokens, confirming a genuine discriminative signal (MCC = 0.75 vs. 0.08 for the best baseline). Information-gain analysis reveals two compleme
    
[^57]: 《扫描智能体框架：AI编码智能体配置暴露问题的仓库研究》

    Scanning the Harness: A Repository Study of Configuration Exposures in AI Coding Agents

    [https://arxiv.org/abs/2609.07360](https://arxiv.org/abs/2609.07360)

    本研究对3,171个公开GitHub仓库开展了首次大规模系统性分析，识别出AI编码智能体配置中的六类安全暴露问题，发现15.4%-17.9%的设置存在未固定MCP包版本、宽泛执行权限授予等供应链风险。

    

    AI编码智能体依赖于仓库指令、技能、钩子、工具服务器声明和子智能体定义。这些工件同时分发行为逻辑和对可执行依赖项的访问权限，使得配置审查成为智能体软件供应链的重要组成部分。我们研究了3,171个公开的GitHub仓库：包括2,660个组装配置的设置和511个技能集合。通过确定性分析、机械重推导、模型辅助审查和平台文档核查，我们识别出六类配置暴露和一致性问题。未固定的MCP包声明出现在9.8%的设置中，宽泛的执行权限授予占2.5%，宽泛的技能工具预批准占3.8%。这三者的并集覆盖了409个设置（15.4%）；在包含MCP配置的设置中，24.5%含有未固定的声明。若将必填字段缺失和技能格式问题纳入统计，设置的问题率升至17.9%，技能集合的问题率为6.8%。在保持六类问题固定不变的前提下，情境化的……（摘要原文在此处截断）

    arXiv:2609.07360v2 Announce Type: replace  Abstract: AI coding agents rely on repository instructions, skills, hooks, tool-server declarations, and subagent definitions. These artifacts distribute both behavior and access to executable dependencies, making configuration review part of the agent software supply chain. We study 3,171 public GitHub repositories: 2,660 assembled setups and 511 skill collections. Deterministic analysis, mechanical re-derivation, model-assisted review, and platform documentation checks identify six categories of configuration exposure and conformance issues. Unpinned MCP package declarations occur in 9.8% of setups, broad execution grants in 2.5%, and broad skill tool preapproval in 3.8%. Their union covers 409 setups (15.4%); among setups with MCP configuration, 24.5% contain an unpinned declaration. Including required-field and skill-format issues brings the setup rate to 17.9% and the collection rate to 6.8%. Holding the six categories fixed, contextual r
    
[^58]: 自我编写的指标：从自身盲点演化出评估器

    Metrics That Write Themselves: Evolving an Evaluator from Its Own Blind Spots

    [https://arxiv.org/abs/2608.18744](https://arxiv.org/abs/2608.18744)

    本文提出EvalCEGAR方法，通过反例引导抽象细化自动演化评估指标，利用碰撞对（正确与错误答案评分相同）作为作者请求，从自身盲点中生成可解释的缺陷检测操作符池，解决了报告生成等场景中自动评分指标缺失的问题。

    

    arXiv:2608.18744v1 公告类型：新 摘要：智能体在可靠自动指标的引导下能快速进步，而没有指标则会停滞不前；最需要这种指标的应用（如报告生成）恰恰是无人知道如何评分的领域。指标能自我编写吗？说清什么使答案优秀很难，但指出答案的问题则相对容易，因此我们演化的指标是一个小型Python操作符池，每个操作符为一个命名的缺陷标记候选答案，或弃权，并投票。直接让模型生成操作符是行不通的：183个候选仅实现96种不同行为，且来自一个巨大空间中的狭窄区域。EvalCEGAR转而借鉴程序验证中的反例引导抽象细化方法。它将操作符池视为一种抽象，并搜索碰撞——即两个答案在操作符评分下相同，但一个正确一个错误。该配对（而非提示）成为创作请求，当碰撞击败所有尝试时，循环会扩大操作符的定义范围。

    arXiv:2608.18744v1 Announce Type: new  Abstract: Agents improve quickly against a reliable automatic metric and stall without one, and the applications that need them most, report generation among them, are the ones nobody knows how to score. Can the metric write itself? Saying what makes an answer good is hard; pointing at something wrong with one is easier, so the metric we evolve is a pool of small Python operators that each flag a candidate for one named defect, or abstain, and vote. Asking a model for operators directly does not work: 183 candidates realise only 96 distinct behaviours, from one narrow region of an enormous space. EvalCEGAR instead borrows counterexample-guided abstraction refinement from program verification. It reads the pool as an abstraction and searches for a collision, two answers the operators score identically, one correct and one not. That pair, not a prompt, is the authoring request, and when a collision defeats every attempt the loop widens what an opera
    
[^59]: 过于自信而不安全：用于可靠日志异常检测的模型校准

    Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection

    [https://arxiv.org/abs/2608.17965](https://arxiv.org/abs/2608.17965)

    本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。

    

    在线日志异常检测对于维护大规模计算系统的可靠性至关重要。尽管基于语言模型的日志异常检测器取得了强大的检测性能，但其置信度估计仍校准不佳。我们表明，这些检测器经常对错误预测赋予过高的置信度，尤其是在严重类别不平衡下的异常日志中。此外，即使传统校准指标显示校准良好，错误预测的置信度仍持续偏高，这为运维监控系统造成了关键可靠性缺口。为解决此问题，我们提出了日志重建与距离（LoRD），一种轻量级的事后校准框架，用于可靠的日志异常检测。LoRD从正确分类的验证样本的潜在表示中学习预测路径特定的可靠性模型，并估计预测可靠性阈值。

    arXiv:2608.17965v1 Announce Type: cross  Abstract: Online log anomaly detection is critical for maintaining the reliability of large-scale computing systems. Although recent language model-based log anomaly detectors achieve strong detection performance, their confidence estimates remain poorly calibrated. We show that these detectors frequently assign excessive confidence to incorrect predictions, particularly for anomalous logs under severe class imbalance. Moreover, confidence on erroneous predictions remains persistently high even when conventional calibration metrics indicate good calibration, creating a critical reliability gap for operational monitoring systems. To address this issue, we propose Log Reconstruction and Distance (LoRD), a lightweight post-hoc calibration framework for reliable log anomaly detection. LoRD learns prediction-route-specific reliability models from latent representations of correctly classified validation samples and estimates prediction reliability th
    
[^60]: 自我进化的编码代理

    Self-Evolving Coding Agents

    [https://arxiv.org/abs/2608.03392](https://arxiv.org/abs/2608.03392)

    本文系统综述了自我进化编码代理领域，定义其概念并区分于传统代理，强调通过持久更新组件从交互中改进行为以应对动态软件开发环境。

    

    大型语言模型正越来越多地嵌入软件工程工作流程中，作为能够检查代码仓库、调用工具、执行测试、调试失败并生成补丁的编码代理。然而，尽管软件开发是一个动态且反馈丰富的过程，其中代码仓库会演化、依赖会变化、测试会失败，修复尝试会留下可复用的经验，但大多数现有代理在部署后仍然基本保持静态。这种矛盾促使了关于自我进化编码代理的研究日益增多，在这种代理中，代理通过持续更新其框架、记忆、技能和工具、模型侧组件、工作流程和拓扑，或环境和上下文，从先前的编码交互中改进其未来行为。在本综述中，我们对这一新兴领域进行了结构化综合。我们首先定义了自我进化的编码代理，并将其与传统编码代理和通用自我进化代理区分开来。

    arXiv:2608.03392v2 Announce Type: replace  Abstract: Large language models are increasingly embedded in software engineering workflows as coding agents that can inspect repositories, invoke tools, execute tests, debug failures, and generate patches. Yet most existing agents remain largely static after deployment, even though software development is a dynamic, feedback-rich process in which repositories evolve, dependencies change, tests fail, and repair attempts leave reusable experience. This tension has motivated a growing body of work on self-evolving coding agents, where the agent improves its future behavior by persistently updating its framework, memory, skills and tools, model-side components, workflow and topology, or environment and context from prior coding interactions. In this survey, we provide a structured synthesis of this emerging area. We first define self-evolving coding agents and distinguish them from conventional coding agents and general self-evolving agents. We t
    
[^61]: 图表支持还是模型自供？检验多模态大语言模型为无障碍可视化生成的声明

    Chart-Supported or Model-Supplied? Examining MLLM-Generated Claims for Accessible Visualization

    [https://arxiv.org/abs/2607.25021](https://arxiv.org/abs/2607.25021)

    本研究发现，提供数据表、标题、替代文本等无障碍图表上下文比提供图像本身更能有效促使MLLM生成有依据的直接声明并提升数值一致性。

    

    多模态大语言模型（MLLM）能够将可视化模式与外部原因、后果和领域知识联系起来，但这些解释的证据基础往往不明确。我们开展了一项探索性研究，涵盖来自四个来源的102个可视化图表、三个MLLM以及四种输入条件，这些条件改变了对图像的访问、无障碍图表上下文（如数据表、标题、替代文本和屏幕阅读器结构等非图像制品）以及保留上下文的提示框架。在1,224份描述中，我们分析了模型归因的DIRECT（直接）、DERIVED（推导）和SPECULATIVE（推测）标签，并对数值一致性进行了自动化审计。无障碍图表上下文使Gemini和GPT更倾向于生成DIRECT声明，并提高了某些模型的数值一致性。在完整上下文中添加图像并未带来一致的数值收益，而保留上下文的提示也未可靠地促使模型使用更谨慎的措辞。（注：原文摘要在此处截断）

    arXiv:2607.25021v2 Announce Type: replace  Abstract: Multimodal large language models (MLLMs) can connect visualization patterns to external causes, consequences, and domain knowledge, but the evidential basis of these interpretations is often unclear. We present an exploratory study of 102 visualizations from four sources, three MLLMs, and four input conditions that vary access to the image, accessible chart context (non-image artifacts such as data tables, captions, alt text, and screen-reader structures), and withheld-context framing. Across 1,224 descriptions, we analyze model-attributed DIRECT, DERIVED, and SPECULATIVE labels and conduct an automated audit of numeric agreement. Accessible chart context shifted Gemini and GPT toward DIRECT claims and improved numeric agreement for some models. Adding the image to the full context did not yield a consistent numeric benefit, and the withheld-context prompt did not reliably increase cautious language. The prompt-defined Real-World Sig
    
[^62]: 通过黑盒、面向漏洞的扫描检测代码生成大语言模型中的数据投毒

    Detecting Data Poisoning in Code Generation LLMs via Black-Box, Vulnerability-Oriented Scanning

    [https://arxiv.org/abs/2603.17174](https://arxiv.org/abs/2603.17174)

    CodeScan是首个用于审计代码生成大语言模型的黑盒、面向特定漏洞的扫描框架，通过分析多次生成结果的结构相似性和迭代发散分析来有效检测诱发不安全代码生成的数据投毒攻击。

    

    代码生成大语言模型（LLM）正日益被集成到现代软件开发工作流程中。最近的研究表明，这些模型容易受到后门和投毒攻击，从而生成不安全的代码，但有效的防御手段仍然有限。现有的扫描方法依赖于标记级别的生成一致性来反推攻击目标，这种方法对于源代码是无效的，因为相同的语义可以以不同的语法形式出现。我们提出了CodeScan，这是第一个用于审计代码生成大语言模型的黑盒、面向特定漏洞的扫描框架，其假设防御者指定目标漏洞类别并提供相应的与任务相关的提示。CodeScan通过分析在不同干净提示条件下多次生成结果之间的结构相似性来识别攻击目标。它将迭代发散分析与抽象语法树（AST）……

    arXiv:2603.17174v2 Announce Type: replace-cross  Abstract: Code generation large language models (LLMs) are increasingly integrated into modern software development workflows. Recent work has shown that these models are vulnerable to backdoor and poisoning attacks that induce the generation of insecure code, yet effective defenses remain limited. Existing scanning approaches rely on token-level generation consistency to invert attack targets, which is ineffective for source code where identical semantics can appear in diverse syntactic forms. We present CodeScan, the first black-box, vulnerability-specific scanning framework for auditing code generation LLMs, assuming that the defender specifies the target vulnerability classes and provides corresponding task-relevant prompts. CodeScan identifies attack targets by analyzing structural similarities across multiple generations conditioned on different clean prompts. It combines iterative divergence analysis with abstract syntax tree (AST
    
[^63]: MOOSEnger：面向 MOOSE 生态系统的仿真感知 AI 智能体框架

    MOOSEnger: A Simulation-Aware AI Agent Framework for the MOOSE Ecosystem

    [https://arxiv.org/abs/2603.04756](https://arxiv.org/abs/2603.04756)

    MOOSEnger 是一个面向 MOOSE 生态系统的仿真感知 AI 智能体框架，通过集成知识检索、HIT 语法解析、验证诊断与求解器反馈的“生成—检查—修复—运行”工作流，克服了大语言模型一次性生成仿真输入文件时易因微小错误而执行失败、且执行成功也不代表科学正确的问题。

    

    MOOSEnger 是一个面向多物理场面向对象仿真环境（MOOSE）生态系统的建模与仿真 AI 智能体框架，其核心是一个仿真感知的支撑框架，该框架将可替换的推理模型与扎根的领域知识、经修订的仿真工件、MOOSE 专属验证以及可执行的求解器反馈相结合。这一外围系统解决了一次性大语言模型生成的一个核心局限：微小的语法、模式、引用或求解器配置错误就可能阻止一个看似合理的输入文件成功执行，而仅仅成功执行并不能确立科学上的正确性。MOOSEnger 的仿真感知支撑框架集成了 MOOSE 知识检索、支持层次化输入文本（HIT）的解析、语法元数据、语言服务器诊断、修订控制的编写，以及本地或基于 MCP 的验证与执行，构成“生成—检查—修复—运行”的工作流，并将证据绑定到每（个环节）……

    arXiv:2603.04756v3 Announce Type: replace  Abstract: MOOSEnger is a modeling and simulation AI agent framework for the Multiphysics Object-Oriented Simulation Environment (MOOSE) ecosystem, built around a simulation-aware harness that combines an interchangeable reasoning model with grounded domain knowledge, revised simulation artifacts, MOOSE-specific validation, and executable solver feedback. This surrounding system addresses a central limitation of one-shot large language model generation: small syntax, schema, reference, or solver-configuration errors can prevent a plausible input from executing, while successful execution alone does not establish scientific correctness. MOOSEnger's simulation-aware harness integrates MOOSE knowledge retrieval, Hierarchical Input Text (HIT)-aware parsing, syntax metadata, language-server diagnostics, revision-controlled authoring, and local or MCP-backed validation and execution in a generate-check-repair-run workflow that binds evidence to each 
    
[^64]: QMon：利用电路中间测量与重置技术监控量子电路的执行

    QMon: Monitoring the Execution of Quantum Circuits with Mid-Circuit Measurement and Reset

    [https://arxiv.org/abs/2512.13422](https://arxiv.org/abs/2512.13422)

    QMon是一种实用的量子电路监控方法，通过结合电路中间测量、重置操作和因果锥重放技术，在保持量子电路原有运行行为的前提下，对电路中选定位置的中间量子态信息进行监测与比较。

    

    与经典软件不同（经典软件中的日志记录和运行时追踪可以有效揭示内部执行状态），量子电路具有独特的属性，例如不可克隆定理和测量诱导的坍缩，这些属性阻碍了对量子态的直接观察或复制。这些特性使得监控量子电路的执行尤为困难，也让调试和运行时监控等重要任务变得复杂。本文提出了QMon，这是一种实用的方法，它利用电路中间测量、重置操作和因果锥重放技术，在明确的条件下监控量子电路中选定的中间值，同时保持其原有的运行时行为。QMon支持在电路内的选定位置插入监控算子，从而可以对这些位置上预期的与实际观测到的单量子比特结果概率进行比较。在理想噪声环境下……（摘要原文在此处截断）

    arXiv:2512.13422v2 Announce Type: replace  Abstract: Unlike classical software, where logging and runtime tracing can effectively reveal internal execution status, quantum circuits possess unique properties, such as the no-cloning theorem and measurement-induced collapse, that prevent direct observation or duplication of their states. These characteristics make it especially challenging to monitor the execution of quantum circuits, complicating essential tasks such as debugging and runtime monitoring. This paper presents QMon, a practical methodology that leverages mid-circuit measurements, reset operations, and causal-cone replay to monitor selected intermediate values of quantum circuits while preserving their original runtime behavior under explicit conditions. QMon enables the instrumentation of monitoring operators at selected locations within the circuit, allowing comparisons between expected and observed one-qubit outcome probabilities at those locations. Under an ideal noise-fr
    
[^65]: 基于搜索的软件工程与AI基础模型：研究现状与未来路线图

    Search-Based Software Engineering and AI Foundation Models: Current Landscape and Future Roadmap

    [https://arxiv.org/abs/2505.19625](https://arxiv.org/abs/2505.19625)

    本文提出一份研究路线图，系统梳理了基于搜索的软件工程（SBSE）与AI基础模型（如大语言模型）的现状，并从基础模型增强SBSE、SBSE改进基础模型及两者融合三个核心方面指明了未来研究方向。

    

    基于搜索的软件工程（SBSE）将元启发式搜索技术与软件工程相结合，作为活跃的研究领域已有约25年历史。它已被应用于解决软件工程全生命周期中的众多问题，并在多个领域展现出广泛的适用性。随着人工智能（AI）的最新进展，特别是大语言模型（LLM）等基础模型（FM）的出现，SBSE与这些模型的协同演进方式仍不明确。在这一机遇窗口期，我们提出了一份研究路线图，阐明了SBSE与基础模型相关的当前研究格局，识别了开放性挑战，并概述了通过SBSE与基础模型的协同来推动SBSE发展的潜在研究方向。具体而言，我们分析了三个核心方面：利用基础模型增强SBSE、应用SBSE改进基础模型，以及探索SBSE与基础模型的融合。

    arXiv:2505.19625v4 Announce Type: replace-cross  Abstract: Search-based software engineering (SBSE), which integrates metaheuristic search techniques with software engineering, has been an active area of research for about 25 years. It has been applied to solve numerous problems across the entire software engineering lifecycle and has demonstrated its versatility in multiple domains. With recent advances in Artificial Intelligence (AI), particularly the emergence of foundation models (FMs) such as large language models (LLMs), the evolution of SBSE alongside these models remains undetermined. In this window of opportunity, we present a research roadmap that articulates the current landscape of SBSE in relation to FMs, identifies open challenges, and outlines potential research directions to advance SBSE through its synergy with FMs. Specifically, we analyze three core aspects: utilizing FMs to enhance SBSE, applying SBSE to advance FMs, and exploring the integration of SBSE and FMs. Fu
    
[^66]: 基于身份的软件签名的上下文感知信任验证

    Context-Aware Trust Verification for Identity-Based Software Signing

    [https://arxiv.org/abs/2406.15596](https://arxiv.org/abs/2406.15596)

    提出DiVerify框架，通过自动化验证软件签名发生的上下文条件（而不仅是签名者身份），解决了凭据泄露时签名仍能通过验证的安全缺陷。

    

    现代软件发布基础设施使用基于身份的软件签名将软件制品与已知身份相关联。例如，npm和Docker Hub等注册表在其制品来源工作流中依赖Sigstore的基于身份的软件签名。然而，现有的验证程序只能提供签名者身份的证据，却无法提供签名发生时的条件。因此，被泄露的凭据、身份提供商或签名工具仍然可以产生通过验证的签名。这一缺陷导致软件工程工具（例如包注册表、依赖分析工具、部署系统和CI/CD验证阶段）无法对软件签名实施可验证的上下文感知信任策略。我们提出了DiVerify，一个用于自动化验证软件签名条件的框架。DiVerify签名绑定来自多个独立范围提供者的身份声明……

    arXiv:2406.15596v4 Announce Type: replace-cross  Abstract: Modern software release infrastructure uses identity-based software signing to associate software artifacts with a known identity. For example, registries such as npm and Docker Hub rely on Sigstore's identity-based software signing in their artifact provenance workflows. However, existing verification procedures only provide evidence of a signer's identity, but not the conditions under which signing occurred. As a result, compromised credentials, identity providers, or signing tools can still produce signatures that pass verification. This gap prevents software engineering tools (e.g., package registries, dependency analysis tools, deployment systems, and CI/CD verification stages) from enforcing verifiable context-aware trust policies for software signing.   We present DiVerify, a framework for automated verification of software signing conditions. A DiVerify signature binds identity claims from multiple independent scope pro
    

