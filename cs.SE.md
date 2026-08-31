# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rethinking Vulnerability Remediation as a Capacity Allocation Problem](https://arxiv.org/abs/2608.28509) | 本文将漏洞修复重新建模为容量分配问题，基于Apache Jira、Mozilla Bugzilla等真实数据分析发现，在AI加速漏洞发现的时代，修复吞吐量（而非优先级排序准确性）才是主要瓶颈，并证明严重程度优先排序和容量预留等流控策略能显著缩短关键漏洞的修复延迟。 |
| [^2] | [A System-of-Systems Case Study for the Verification of Composed Digital Twins](https://arxiv.org/abs/2608.28498) | 本文通过一个基于VDM-RT形式化建模的温室系统之系统案例研究，识别了在系统之系统环境中验证组合式数字孪生所面临的基础性挑战，并探索了将相关性、可验证性、可替代性和保真度等数字孪生质量属性形式化以支持组合推理的途径。 |
| [^3] | [On the Maintenance and Co-evolution of Agent Plugins: An Empirical Study of Claude Code Plugin Marketplaces](https://arxiv.org/abs/2608.28497) | 该论文首次对Claude Code插件市场开展大规模实证研究，通过分析1,926个仓库、8,351个插件和77,773次提交，揭示了智能体插件是需要在自然语言指令、脚本和配置等组件间持续维护并协同演化的软件工件。 |
| [^4] | [Recovering Software Architecture Intent from Historical Work Items using Generative AI: A Mixed-Methods Industry Case Study](https://arxiv.org/abs/2608.28403) | 本研究提出一种基于大语言模型的半自动五步流水线，通过提示链、双向可追溯性和思维链推理从历史敏捷工作项中恢复C4架构图，专家访谈与稳定性分析表明生成的架构基线准确且对系统理解高度有用。 |
| [^5] | [When Verified Source Becomes Attack Input: Defending Smart Contracts Against LLM-Based Vulnerability Scanning](https://arxiv.org/abs/2608.28400) | 提出DeLLMGuard智能合约部署框架，通过多合约地址将公开披露的源代码与运行时执行分离，在保留源代码公开和授权审计能力的同时，有效防御基于大语言模型的恶意漏洞扫描攻击。 |
| [^6] | [Sustainability of Open-Source Machine Learning Robustness Assessment Tools: A Repository Mining Study](https://arxiv.org/abs/2608.28396) | 该研究通过对 GitHub 上 28 个开源机器学习鲁棒性评估工具仓库进行挖掘分析，实证考察了这些工具的社区参与、维护活动与项目长期可持续性。 |
| [^7] | [Where Does Balance Break? Boundary Discovery for Game Balance Testing under a Finite Simulation Budget](https://arxiv.org/abs/2608.28364) | 本文将游戏平衡回归测试形式化为有限仿真预算下的边界发现问题，旨在高效定位可接受平衡与平衡失效之间的分界输入。 |
| [^8] | [Adaptive Strategy Generation for Boundary Value Exploration Beyond Numeric Inputs](https://arxiv.org/abs/2608.28230) | ABEX是一个基于LLM智能体的边界值探索框架，通过自适应生成自然语言表达的探索策略取代手工设计的变异算子，将边界值探索从数值输入扩展到任意输入类型，并支持有效策略的存储与复用。 |
| [^9] | [From Architecture to Binary: Ensuring Cross-Domain Consistency in Model-Based Airborne Software Development](https://arxiv.org/abs/2608.28156) | 本文提出一种以仓库为中心的机载软件开发方法，通过关系接口数据库、跨仓库引用和自动化CI流水线，确保有人/无人飞行器开发中从系统架构、功能模型到嵌入式软件的跨领域一致性，且适用于资源受限的小型团队。 |
| [^10] | [Post-Edit Re-Verification in Simulator-Backed Engineering Agents: A Controlled Comparison of Verification-Cadence Guidance](https://arxiv.org/abs/2608.28147) | 该研究通过受控比较考察了在基于仿真器的工程智能体中，保留或省略“修改后请求新仿真”的显式验证节奏指导是否会影响编辑后重新验证行为，测量的是指令条件下的验证策略遵循度而非对证据过时的自发识别，并在DWSIM阀门-压力调节任务上用五个Qwen模型进行了评估。 |
| [^11] | [RESTCov: A Tool for Structural Coverage Analysis of REST APIs](https://arxiv.org/abs/2608.28114) | RESTCov是一个基于OpenAPI规范和HTTP请求/响应日志、无需访问源代码即可在路径、操作、参数、媒体类型和状态码等多个维度上对REST API进行结构化覆盖率分析的轻量级工具。 |
| [^12] | [Compared to What? A Human-Anchored Security Benchmark for LLM-Generated Infrastructure-as-Code](https://arxiv.org/abs/2608.28021) | 该论文提出GenIaC-SecBench基准，首次通过扫描634个人类编写的IaC模板建立规模匹配的人类安全基线，发现漏洞密度与代码规模强负相关，从而揭示此前未匹配规模的LLM安全评估实际衡量的是代码大小而非真实安全性。 |
| [^13] | [Moirae: A Multimodal Agent Collaborative Framework for Dynamic Android Malware Detection](https://arxiv.org/abs/2608.27994) | 提出Moirae框架，通过多模态智能体协作动态收集运行时证据（视觉欺骗线索、UI状态转换、运行时API行为），并融合多维度行为视图，解决了现有检测器面临的概念漂移和混淆攻击问题。 |
| [^14] | [CHISEL-ing Back Source Code with AI-enabled Iterative Recovery](https://arxiv.org/abs/2608.27981) | 提出了无需测试套件的CHISEL框架，仅依靠编译器静态分析与覆盖率引导模糊测试的迭代反馈，即可从Ghidra伪C代码中恢复可编译且语义等价的源代码。 |
| [^15] | [GraftyVul: Synthesising Insecure Programs Through Real-World Vulnerability Grafting](https://arxiv.org/abs/2608.27928) | GraftyVul通过将真实世界漏洞嫁接到开源项目中，生成了212个覆盖5种编程语言和23个CWE类别、经过验证且可利用的漏洞程序，同时兼顾了多样性、可复现性和真实性。 |
| [^16] | [Antipatterns in AI-assisted Qualitative Data Analysis: A Catalog of Temptations and Pitfalls for Software Engineering Researchers](https://arxiv.org/abs/2608.27927) | 本文基于作者数十年的定性研究经验，首次系统性地提出了AI辅助定性数据分析中的反模式目录，将其按影响程度递进划分为“危险驱动因素、操作失误和分析失败”三大类别，旨在帮助软件工程研究人员识别并规避AI辅助分析中损害研究严谨性与有效性的方法论风险。 |
| [^17] | [Decoupling is a Necessity: Transformation-Agnostic Decompiled Code Recovery under Optimization and Obfuscation](https://arxiv.org/abs/2608.27889) | 提出ReSource——首个面向变换无关源代码恢复的多阶段LLM框架，通过将二进制与源代码的差异解耦为词汇、语法、语义三个正交层次，在编译器优化与代码混淆的双重干扰下实现高质量的反编译代码恢复。 |
| [^18] | [DBRepro: Automated Database Synthesis via a Hybrid Constraint-Solving Approach for Reproducing Slow Queries](https://arxiv.org/abs/2608.27822) | DBRepro通过混合约束求解方法，从非侵入式元数据自动合成代理数据库，在保留全局统计分布的同时满足精确的局部基数约束，使查询优化器生成与慢查询相同的物理执行计划，从而实现安全的离线性能诊断。 |
| [^19] | [RiskBlend: A Multi-Signal Framework for Test Input Prioritization in Machine Learning Regression Testing](https://arxiv.org/abs/2608.27704) | RiskBlend提出了一种与分类器无关的多信号测试输入优先级排序框架，通过融合历史失败模式、预测偏移、决策边界偏移和邻域变化四种互补风险信号，在有限验证预算下更有效地发现机器学习模型重训练引发的回归缺陷。 |
| [^20] | [Operationalizing Regulations into Code: A Model to Enhance Governance and Compliance in LLM Selection for Software Engineering](https://arxiv.org/abs/2608.27703) | 本文通过设计科学研究方法提出一个三层模型，将欧盟《人工智能法案》、NIST RMF、GDPR等法规要求操作化为包含否决性和加权评分标准的多准则决策矩阵，从而增强软件工程项目中选择大语言模型时的治理与合规性。 |
| [^21] | [Predicting LLM Performance from Prompt Linguistic Features: An Empirical Study in Requirements Engineering](https://arxiv.org/abs/2608.27621) | 该研究证明提示词的可测量语言特征能够在推理之前预测大语言模型的性能，从而为需求工程中的提示词选择与优化提供了一种低成本、无需反复试错的方法。 |
| [^22] | [Image Augmentation as Test Generation for Deep Learning-Based Image Retrieval Systems](https://arxiv.org/abs/2608.27502) | 本文系统梳理了50种图像增强与生成技术并建立十类分类体系，同时通过大规模实证研究评估这些技术作为基于嵌入的图像检索系统测试生成器的有效性。 |
| [^23] | [Grounded Checklist Partial Credit for Agent Skill Trajectories](https://arxiv.org/abs/2608.27487) | 提出GCPC方法，由人类一次性定义可复用规则、大语言模型基于任务指令和官方验证器实例化任务专属清单，实现对智能体轨迹的可信部分给分评估，克服了二值任务成功率掩盖部分进展的局限。 |
| [^24] | [Report of the 2026 Workshop on Next-Generation Ecosystems for Scientific Computing: Harnessing Community, Software, and AI for Cross-Disciplinary Team Science](https://arxiv.org/abs/2608.26519) | 本报告基于2026年研讨会，提炼出科学计算生态系统未来发展的四大战略主题和八项社区行动优先事项，强调通过社会技术协同设计整合人工智能、软件和跨学科合作。 |
| [^25] | [SPECMINE: A Large-Scale Corpus of Spec-Driven Development Artifacts](https://arxiv.org/abs/2608.25202) | 我们提出了SPECMINE，这是首个大规模语料库，通过两次普查系统地捕捉了GitHub上规范驱动开发工件，为研究规范如何转化为代码提供了基础数据。 |
| [^26] | [VR-Themis: A Scalable Framework for Virtual Reality Application Clone Detection](https://arxiv.org/abs/2608.13290) | 本文提出了基于“层次-对象-行为”（HOB）模型的两阶段VR应用克隆检测框架VR-Themis，通过粗粒度统计特征聚类实现大规模数据集上的可扩展性，再对可疑应用进行细粒度深入分析，弥补了现有移动应用克隆检测方法无法有效检测VR应用克隆的不足。 |
| [^27] | [Set-shifting Behavioral Test for Harnessed Agents](https://arxiv.org/abs/2607.13396) | 该论文借鉴认知心理学中的“定势转换”概念，提出了一种通过在冗余工具库中隐藏地切换可靠工具组来测试LLM智能体适应能力的行为测试方法，并发现不同模型面对相同切换时表现出截然不同的行为模式。 |
| [^28] | [EvoRepair: Enhancing Vulnerability Repair Agents Through Experience-Based Self-Evolution](https://arxiv.org/abs/2605.30105) | EvoRepair是首个基于经验自进化的自动化漏洞修复智能体框架，通过循环的学习-修复过程实现修复经验的积累、精炼与跨漏洞复用，从而提升LLM的漏洞修复能力。 |
| [^29] | [Beyond Output Correctness: Benchmarking and Evaluating Large Language Model Reasoning in Coding Tasks](https://arxiv.org/abs/2604.12379) | 该论文提出了首个覆盖代码生成、摘要与分类三类编程任务的推理质量评估基准CodeRQ-Bench，并通过分析评估器失配案例得出设计启示，进而提出结合证据验证与歧义感知评分修正的两阶段评估器VERA，显著提升了编程任务中大语言模型推理质量的评估效果。 |
| [^30] | ["An Endless Stream of AI Slop": How Developers Discuss the Burden of AI-Assisted Software Development](https://arxiv.org/abs/2603.27249) | 该研究通过定性分析1,154条Reddit和Hacker News帖子，首次系统揭示了开发者对AI生成的低质量内容（“AI slop”）给软件开发带来负担的感知与应对，并将其框架化为一种公地悲剧——个人生产力收益以牺牲审查者、维护者和整个社区的利益为代价。 |
| [^31] | [Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond](https://arxiv.org/abs/2602.11671) | 提出Hydra框架，将代码视为结构化代码而非自然语言，通过结构感知索引策略解决现有RAG方法因分块和相似性检索导致的代码结构关系丢失问题，从而提升仓库级代码生成的效果。 |
| [^32] | [ASA: Backbone-Training-Free Representation Engineering for Tool-Calling Agents](https://arxiv.org/abs/2602.04935) | ASA提出了一种无需骨干训练的推理时激活引导方法，弥合了LLM智能体“知道该用工具却不敢用”的表示-行为鸿沟，在不微调模型的情况下显著提升了特定领域工具调用的可靠性。 |

# 详细

[^1]: 重新审视漏洞修复：一个容量分配问题

    Rethinking Vulnerability Remediation as a Capacity Allocation Problem

    [https://arxiv.org/abs/2608.28509](https://arxiv.org/abs/2608.28509)

    本文将漏洞修复重新建模为容量分配问题，基于Apache Jira、Mozilla Bugzilla等真实数据分析发现，在AI加速漏洞发现的时代，修复吞吐量（而非优先级排序准确性）才是主要瓶颈，并证明严重程度优先排序和容量预留等流控策略能显著缩短关键漏洞的修复延迟。

    

    随着AI加速漏洞发现，修复吞吐量可能成为比优先级排序准确性更大的约束。本研究将漏洞修复评估为一个流控问题，使用了Apache Jira、Mozilla Bugzilla、Red Hat安全勘误数据、五个公共Jira组织以及npm依赖图。研究发现，Apache的漏洞解决时间呈强重尾分布，而主要问题跟踪器中94-100%的新到达项进入估计达到或超过处理容量的队列。队列上下文模型仅提供中等程度的预测判别能力，且其表现很大程度上被简单的项目级基线所匹配。严重程度到修复速度的判别在不同系统之间差异显著。流控分析显示了更大的运营层面效应：队列从过载状态转为排空状态与更短的解决时间相关；在固定容量下采用严重程度优先的排序策略可减少关键项目的延迟；而容量预留机制可以减少关键项目的长期延迟。

    arXiv:2608.28509v1 Announce Type: new  Abstract: As AI accelerates vulnerability discovery, remediation throughput may become a greater constraint than prioritisation accuracy. This study evaluates vulnerability remediation as a flow-control problem using Apache Jira, Mozilla Bugzilla, Red Hat security errata, five public Jira organisations, and an npm dependency graph. Apache resolution times are strongly heavy-tailed, while 94-100% of arrivals in the primary issue trackers enter queues estimated to be at or above capacity. Queue-context models provide only moderate predictive discrimination and are largely matched by simple project-level baselines. Severity-to-speed discrimination varies substantially across systems. Flow-control analyses show larger operational effects: transitions from overloaded to draining queues are associated with shorter resolution times, severity-first sequencing reduces critical-item delay at fixed capacity, and capacity reservation can reduce prolonged crit
    
[^2]: 用于验证组合式数字孪生的系统之系统案例研究

    A System-of-Systems Case Study for the Verification of Composed Digital Twins

    [https://arxiv.org/abs/2608.28498](https://arxiv.org/abs/2608.28498)

    本文通过一个基于VDM-RT形式化建模的温室系统之系统案例研究，识别了在系统之系统环境中验证组合式数字孪生所面临的基础性挑战，并探索了将相关性、可验证性、可替代性和保真度等数字孪生质量属性形式化以支持组合推理的途径。

    

    当前构建可信的网络物理系统数字孪生的方法，在如何将相关性、可验证性、可替代性和保真度等质量属性形式化并加以验证方面缺乏实践指导。这一需求在系统之系统环境中被进一步放大，因为系统之系统依赖于多个数字孪生的组合。本研究的目标是识别数字孪生验证与确认框架在系统之系统环境中应解决的基础性挑战，并研究如何将单个数字孪生工件形式化，作为迈向组合推理的一步。我们提出了一个基于VDM-RT（维也纳开发方法-实时扩展）建模的数字孪生赋能温室系统之系统的案例研究，包括可执行的形式化模型、对数字孪生质量属性的基于属性的刻画，以及对在系统之系统层面尝试组合这些工件时所出现障碍的分析。我们探讨了如何将数字孪生质量属性操作化为一组可验证的性质。

    arXiv:2608.28498v1 Announce Type: new  Abstract: Current approaches to engineering dependable Digital Twins (DTs) of Cyber-Physical Systems lack practical guidance on how qualities such as relevance, verifiability, substitutability and fidelity may be formalised and verified. This need is amplified in Systems-of-Systems (SoS), where reliance is placed on the composition of DTs. The goal of this study is to identify foundational challenges that a framework for DT validation and verification should address in an SoS setting, and to investigate the formalisation of individual DT artefacts as a step towards compositional reasoning. We present a case study based on a DT-enabled greenhouse SoS modelled in VDM-RT (Vienna Development Method, Real-Time), including executable formal models, a property-based account of DT qualities, and an analysis of the obstacles arising when attempting to compose these artefacts at the SoS level. We consider how DT qualities may be operationalised as sets of v
    
[^3]: 关于智能体插件维护与协同演化的研究：对Claude Code插件市场的实证研究

    On the Maintenance and Co-evolution of Agent Plugins: An Empirical Study of Claude Code Plugin Marketplaces

    [https://arxiv.org/abs/2608.28497](https://arxiv.org/abs/2608.28497)

    该论文首次对Claude Code插件市场开展大规模实证研究，通过分析1,926个仓库、8,351个插件和77,773次提交，揭示了智能体插件是需要在自然语言指令、脚本和配置等组件间持续维护并协同演化的软件工件。

    

    AI编程智能体（即通过推理和工具使用来自动化开发任务的软件工具）正日益通过插件市场进行扩展，然而这些新兴仓库的结构、维护方式及协同演化动态在实证层面仍缺乏研究。与通过源代码交付功能的传统软件包不同，智能体插件通过自然语言指令文件、脚本和配置文件的组合来交付功能，这引出一个问题：这些插件究竟是需要跨组件协同演化的被维护工件，还是开发者只需编写一次、无需再次访问的一次性工件。为研究智能体插件的维护与协同演化，我们对1,926个托管Claude Code插件市场的仓库开展了实证研究，分析了2,018个市场中的8,351个插件和77,773次提交。我们发现该市场正在快速扩张，插件……

    arXiv:2608.28497v1 Announce Type: new  Abstract: AI coding agents, software tools that automate development tasks through reasoning and tool use, are increasingly extended through plugin marketplaces, yet the structure, maintenance, and co-evolution dynamics of these emerging repositories remain empirically unexplored. Unlike traditional software packages that deliver functionality through source code, agent plugins deliver functionality through a combination of natural-language instruction files, scripts, and configuration files, raising the question of whether these plugins are maintained artifacts that co-evolve across components, or one-off artifacts that developers write once and do not need to revisit. To study the maintenance and co-evolution of agent plugins, we conduct an empirical study of 1,926 repositories hosting Claude Code plugin marketplaces, analyzing 8,351 plugins and 77,773 commits across 2,018 marketplaces. We find that the marketplace is expanding rapidly, plugin-t
    
[^4]: 使用生成式AI从历史工作项中恢复软件架构意图：一项混合方法的工业案例研究

    Recovering Software Architecture Intent from Historical Work Items using Generative AI: A Mixed-Methods Industry Case Study

    [https://arxiv.org/abs/2608.28403](https://arxiv.org/abs/2608.28403)

    本研究提出一种基于大语言模型的半自动五步流水线，通过提示链、双向可追溯性和思维链推理从历史敏捷工作项中恢复C4架构图，专家访谈与稳定性分析表明生成的架构基线准确且对系统理解高度有用。

    

    软件架构往往只能部分地体现在代码中，而大部分设计意图存在于不断演进的项目制品中。在敏捷项目中，工作项、用户故事及相关跟踪文档保留了这些意图的宝贵痕迹，但它们很少能直接支持架构分析。本研究探讨了利用基于大语言模型（LLM）的流水线从历史敏捷工作项中恢复C4架构图。该半自动化的五步工作流采用提示链、双向可追溯性和思维链（Chain-of-Thought）推理，将非结构化的Azure DevOps工作项转化为可视化制品。我们在两个工业项目上进行评估，采用将定性专家访谈与定量稳定性分析相结合的混合方法设计。从业者认为生成的架构基线是准确的，并且对理解系统非常有用。由于严格受限于输入数据，这些制品反映了……

    arXiv:2608.28403v1 Announce Type: new  Abstract: Software architecture is often only partially captured in code, while much of the design intent lives in evolving project artifacts. In agile projects, work items, user stories, and related tracking documents preserve valuable traces of that intent, but they rarely support direct architectural analysis. This work investigates the recovery of C4 architecture diagrams from historical agile work items using an LLM-based pipeline. The semi-automatic five-step workflow employs a prompt chain, bidirectional traceability, and Chain-of-Thought reasoning to transform unstructured Azure DevOps work items into visual artifacts. Evaluated on two industry projects, we use a mixed-methods design combining qualitative expert interviews with a quantitative stability analysis. Practitioners perceive the generated architectural baselines as accurate and highly useful for system comprehension. Strictly bound by their input data, the artifacts mirror the do
    
[^5]: 当经过验证的源代码成为攻击输入：防御智能合约免受基于大语言模型的漏洞扫描

    When Verified Source Becomes Attack Input: Defending Smart Contracts Against LLM-Based Vulnerability Scanning

    [https://arxiv.org/abs/2608.28400](https://arxiv.org/abs/2608.28400)

    提出DeLLMGuard智能合约部署框架，通过多合约地址将公开披露的源代码与运行时执行分离，在保留源代码公开和授权审计能力的同时，有效防御基于大语言模型的恶意漏洞扫描攻击。

    

    智能合约是部署在区块链上用于管理数字资产的金融程序。为了与用户和投资者建立信任，智能合约项目通常会在区块链浏览器上公开其源代码，并将其与部署的字节码进行验证比对，使链上程序可以通过人类可读的实现方式被访问。然而，大语言模型（LLM）智能体正在改变这种披露机制的威胁模型。通过利用公开披露的源代码，近期的智能体工作流使得大规模扫描合约漏洞并实施攻击变得越来越可行。在本文中，我们提出了DeLLMGuard，一个智能合约部署框架，它能够在保留公开源代码披露和授权审计的同时，防御恶意的基于LLM的漏洞扫描。DeLLMGuard可以在真实世界的区块链环境中，通过多个合约地址将公开披露的源代码与运行时执行相互分离。LLM智能体必须（摘要在此处被截断）

    arXiv:2608.28400v1 Announce Type: cross  Abstract: Smart contracts are financial programs deployed on blockchains to manage digital assets. To build trust with users and investors, smart contract projects typically publish their source code on blockchain explorers and verify it against the deployed bytecode, making the on-chain program accessible through a human-readable implementation. However, LLM agents are changing the threat model of this disclosure mechanism. By leveraging publicly disclosed source code, recent agent workflows make it increasingly practical to scan contract vulnerabilities for exploits at large scale.   In this paper, we propose DeLLMGuard, a smart contract deployment framework that defends against malicious LLM-based vulnerability scanning while preserving public source disclosure and authorized auditing. DeLLMGuard can separate disclosed source code from runtime execution through multiple contract addresses in a real-world blockchain environment. LLM agents mus
    
[^6]: 开源机器学习鲁棒性评估工具的可持续性：一项代码仓库挖掘研究

    Sustainability of Open-Source Machine Learning Robustness Assessment Tools: A Repository Mining Study

    [https://arxiv.org/abs/2608.28396](https://arxiv.org/abs/2608.28396)

    该研究通过对 GitHub 上 28 个开源机器学习鲁棒性评估工具仓库进行挖掘分析，实证考察了这些工具的社区参与、维护活动与项目长期可持续性。

    

    鲁棒性评估对于在真实世界环境中部署机器学习（ML）系统至关重要，因为在这些环境中模型可能面临对抗性扰动、分布偏移以及其他运行压力。许多开源工具，包括 Adversarial Robustness Toolbox、Foolbox 和 Robustness Gym，都支持鲁棒性测试与评估。然而，关于这些工具如何被维护、如何获得公众参与以及如何长期延续，目前知之甚少，尽管从业者可能依赖它们来选择评估依赖项、复现鲁棒性评估并为人工智能保障提供证据。我们对开源鲁棒性工具生态系统进行了实证研究。从基于先前工作整理的种子集合出发，我们系统地搜索了 GitHub，并识别出 28 个鲁棒性工具仓库。我们分析了仓库工件，以刻画可观察到的社区参与、维护活动和项目寿命。

    arXiv:2608.28396v1 Announce Type: new  Abstract: Robustness evaluation is essential for deploying machine-learning (ML) systems in real-world settings, where models may face adversarial perturbations, distribution shifts, and other operational stressors. Many open-source tools, including Adversarial Robustness Toolbox, Foolbox, and Robustness Gym, support robustness testing and evaluation. However, little is known about how these tools are maintained, publicly engaged with, and sustained over time, even though practitioners may rely on them to select evaluation dependencies, reproduce robustness assessments, and provide evidence for AI assurance. We present an empirical study of the open-source robustness tooling ecosystem. Starting from a curated seed set derived from prior work, we systematically searched GitHub and identified 28 robustness-tool repositories. We analyzed repository artifacts to characterize observable community engagement, maintenance activity, and project longevity 
    
[^7]: 平衡在哪里被打破？有限仿真预算下游戏平衡测试的边界发现

    Where Does Balance Break? Boundary Discovery for Game Balance Testing under a Finite Simulation Budget

    [https://arxiv.org/abs/2608.28364](https://arxiv.org/abs/2608.28364)

    本文将游戏平衡回归测试形式化为有限仿真预算下的边界发现问题，旨在高效定位可接受平衡与平衡失效之间的分界输入。

    

    软件测试通常依赖于可复现执行和稳定的正确性判据等假设。然而，许多现代软件系统表现出非确定性的执行过程和庞大的行为空间，使得穷举式探索不切实际，单次运行的判断也不可靠。这些特性使得确定可接受行为与问题行为之间的分界线变得困难。竞技多人游戏是这类系统的一个具有挑战性的实例：必须维持游戏平衡，以确保没有任何单一策略占据支配地位。即使微小的参数变化也可能引发平衡的突然破坏，而检测此类失效需要在非确定性结果和高维参数空间下进行重复仿真。本文将游戏平衡回归测试形式化为有限仿真预算下的边界发现问题，其目标是高效识别靠近边界、即平衡开始被打破处的输入。

    arXiv:2608.28364v1 Announce Type: new  Abstract: Software testing often relies on assumptions such as reproducible executions and stable correctness criteria. However, many modern software systems exhibit non-deterministic executions and large behavior spaces, making exhaustive exploration impractical and single-run judgments unreliable. These characteristics make it difficult to identify where acceptable behavior ends and problematic behavior begins. Competitive multiplayer games represent a challenging instance of such systems, where balance must be maintained so that no single strategy dominates. Even small parameter changes can trigger abrupt balance disruption, yet detecting such failures requires repeated simulations under non-deterministic outcomes and high-dimensional parameter spaces. In this paper, we formulate game balance regression testing as a boundary-discovery problem under a finite simulation budget. The objective is to efficiently identify inputs near the boundary tha
    
[^8]: 面向超越数值输入的边界值探索的自适应策略生成

    Adaptive Strategy Generation for Boundary Value Exploration Beyond Numeric Inputs

    [https://arxiv.org/abs/2608.28230](https://arxiv.org/abs/2608.28230)

    ABEX是一个基于LLM智能体的边界值探索框架，通过自适应生成自然语言表达的探索策略取代手工设计的变异算子，将边界值探索从数值输入扩展到任意输入类型，并支持有效策略的存储与复用。

    

    软件行为在输入区域之间的边界处常常发生突变，而这些转变已被证明是容易引发故障的。边界值探索（BVE）通过搜索相似但会触发不同程序行为的输入对，来自动发现这些边界。现有的自动化BVE技术依赖于为每种输入类型、甚至为每个被测函数手工设计的变异算子，这使得它们的应用范围仅限于数值输入。我们提出了ABEX，一个基于大语言模型（LLM）的智能体框架，它用自适应策略生成取代了算子工程：专门的LLM智能体在执行反馈和质量多样性（QD）档案的引导下，提出、选择并执行边界探索策略。由于策略以自然语言表达，它们既能编码类型级别的知识，也能编码函数特定的知识，并且有效的策略甚至可以被存储和复用。我们在黑盒设置中评估了ABEX……

    arXiv:2608.28230v1 Announce Type: new  Abstract: Software behavior often changes abruptly at boundaries between input regions, and these transitions are known to be fault-prone. Boundary Value Exploration (BVE) automates boundary discovery by searching for pairs of similar inputs that nevertheless trigger different program behaviors. Existing automated BVE techniques rely on mutation operators hand-engineered for each input type, or even for each function under test, which has confined their use to numeric inputs. We present ABEX, an agentic LLM-based framework that replaces operator engineering with adaptive strategy generation: specialized LLM agents propose, select, and execute boundary-exploration strategies, guided by execution feedback and a quality-diversity (QD) archive. Because strategies are expressed in natural language, they can encode both type-level and function-specific knowledge, and effective strategies can even be stored and reused. We evaluate ABEX in a black-box set
    
[^9]: 从架构到二进制：确保基于模型的机载软件开发中的跨领域一致性

    From Architecture to Binary: Ensuring Cross-Domain Consistency in Model-Based Airborne Software Development

    [https://arxiv.org/abs/2608.28156](https://arxiv.org/abs/2608.28156)

    本文提出一种以仓库为中心的机载软件开发方法，通过关系接口数据库、跨仓库引用和自动化CI流水线，确保有人/无人飞行器开发中从系统架构、功能模型到嵌入式软件的跨领域一致性，且适用于资源受限的小型团队。

    

    本文提出了一种面向有人和无人驾驶飞行器的机载软件开发方法，旨在减少系统领域、基于模型的功能领域与嵌入式软件领域之间的不一致性。在受ARP-4754B和DO-178C等标准影响的环境中，这些不一致性通常源于跨领域边界的执行力度不足，而非流程定义的缺失。在先前提出的以关系接口数据库为中心的工具链基础上，我们识别出反复出现的失效模式，并提出一种以仓库为中心的实现方案来解决这些问题，该方案专为无重量级流程开销、资源受限的小型团队量身定制。每个领域都被分配一个主仓库，并通过跨仓库引用和专用的CI流水线来生成、更新和验证交换的工件。自动化接口更新、差异变更通知和一致性检查……

    arXiv:2608.28156v1 Announce Type: new  Abstract: This paper presents an airborne software development approach for manned and unmanned aerial vehicles aimed at reducing inconsistencies across system, model-based functional, and embedded software domains. In environments influenced by standards such as ARP-4754B and DO-178C, these inconsistencies typically stem from insufficient enforcement across domain boundaries rather than missing process definitions. Building on a previously proposed toolchain centered on a relational interface database, we identify recurring failure modes and propose a repository-centered implementation to address them, tailored to small, resource-constrained teams operating without heavyweight process overhead. Each domain is assigned a primary repository with cross-repository references and dedicated CI pipelines that generate, update, and validate the exchanged artifacts. Automated interface updates, differential change notifications, and consistency checks pro
    
[^10]: 基于仿真器的工程智能体中的编辑后重新验证：验证节奏指导的受控比较

    Post-Edit Re-Verification in Simulator-Backed Engineering Agents: A Controlled Comparison of Verification-Cadence Guidance

    [https://arxiv.org/abs/2608.28147](https://arxiv.org/abs/2608.28147)

    该研究通过受控比较考察了在基于仿真器的工程智能体中，保留或省略“修改后请求新仿真”的显式验证节奏指导是否会影响编辑后重新验证行为，测量的是指令条件下的验证策略遵循度而非对证据过时的自发识别，并在DWSIM阀门-压力调节任务上用五个Qwen模型进行了评估。

    

    与外部仿真器交互的工程智能体可能需要协调设计修改与针对修改后状态重新获取工程证据。我们探讨在保持与验证相关的状态/事实不变的前提下，保留或省略显式的验证节奏指导是否会改变编辑后的首次重新验证行为。节奏指导组（CG）保留了“在实质性修改后请求新仿真”的指令，而节奏省略组（CO）删除了该指令；两种条件均未使用硬性门控。因此，本研究测量的是指令条件下的编辑后验证策略遵循度，而非对先前证据已过时的自发识别。研究以DWSIM作为仿真器后端，采用连续阀门-压力调节任务，对五个阿里巴巴/Qwen模型在八个合成案例上进行评估；每个模型-案例-条件组合通过实时API调用执行三次……

    arXiv:2608.28147v1 Announce Type: new  Abstract: Engineering agents that interact with external simulators may need to coordinate design modification with reacquisition of engineering evidence for the modified state. We ask whether first post-edit re-verification changes when explicit verification-cadence guidance is retained versus omitted while verification-relevant state/facts are held constant. Cadence-Guided (CG) retained an instruction to request a new simulation after a substantive modification, whereas Cadence-Omitted (CO) removed that instruction; neither condition used a hard gate. The study therefore measures instruction-conditioned post-edit verification-policy adherence rather than spontaneous recognition that prior evidence has become stale. Using DWSIM as the simulator backend and continuous valve-pressure adjustment, five Alibaba/Qwen models were evaluated on eight synthetic cases; each model-case-condition combination was executed three times via live API calls, yieldi
    
[^11]: RESTCov：一个用于REST API结构化覆盖率分析的工具

    RESTCov: A Tool for Structural Coverage Analysis of REST APIs

    [https://arxiv.org/abs/2608.28114](https://arxiv.org/abs/2608.28114)

    RESTCov是一个基于OpenAPI规范和HTTP请求/响应日志、无需访问源代码即可在路径、操作、参数、媒体类型和状态码等多个维度上对REST API进行结构化覆盖率分析的轻量级工具。

    

    REST API在现代软件系统中被广泛使用，但开发人员和测试人员往往无法了解测试套件究竟执行了API规范中的哪些部分。传统的覆盖率分析通常依赖于源代码插桩，这对于分布式的、由外部维护的、只能通过黑盒执行方式访问的REST API而言并不现实。本文提出了RESTCov，一个轻量级工具，它基于OpenAPI规范和观测到的HTTP请求/响应日志来计算REST API的结构化覆盖率，并在路径、操作、参数、媒体类型、状态码以及状态类别等多个维度上报告覆盖情况。RESTCov同时生成机器可读的结果和人类可读的HTML报告，帮助用户检查覆盖缺口、诊断规范与日志之间的不匹配，并在无需访问实现代码的情况下评估REST API测试套件。

    arXiv:2608.28114v1 Announce Type: new  Abstract: REST APIs are widely used in modern software systems, but developers and testers often lack visibility into which parts of an API specification are exercised by a test suite. Traditional coverage analysis usually relies on source-code instrumentation, which is impractical for REST APIs that are distributed, externally maintained, and hence accessible only through black-box execution. This paper presents RESTCov, a lightweight tool that computes structural REST API coverage from an OpenAPI specification and observed HTTP request/response logs, reporting coverage across paths, operations, parameters, media types, status codes, and status classes. RESTCov produces both machine-readable results and a human-readable HTML report, helping users inspect coverage gaps, diagnose specification-log mismatches, and evaluate REST API test suites without requiring access to the implementation.   Screencast: https://youtu.be/mNz2P43OyUc   Repository: ht
    
[^12]: 与什么相比？一个以人类为基准的大语言模型生成基础设施即代码安全评测基准

    Compared to What? A Human-Anchored Security Benchmark for LLM-Generated Infrastructure-as-Code

    [https://arxiv.org/abs/2608.28021](https://arxiv.org/abs/2608.28021)

    该论文提出GenIaC-SecBench基准，首次通过扫描634个人类编写的IaC模板建立规模匹配的人类安全基线，发现漏洞密度与代码规模强负相关，从而揭示此前未匹配规模的LLM安全评估实际衡量的是代码大小而非真实安全性。

    

    大语言模型正被越来越多地用于编写基础设施即代码，而一个不安全的默认配置可能被直接部署到生产环境中。先前的评估仅报告模型生成的IaC的原始漏洞数量，但由于缺乏人类基线，无法判断模型是否真的比工程师更差。我们提出了GenIaC-SecBench，这是一个包含100个按架构复杂度分层的部署场景的基准测试，评估了来自四个厂商的12种模型配置，共产生1,196个IaC制品，并由三个独立的策略引擎进行扫描。关键的是，我们还使用相同的工具链扫描了634个人类编写的IaC模板，提供了首个规模匹配的人类安全基线。研究发现，漏洞密度与制品大小呈强负相关（Spearman ρ = -0.55，p < 10⁻⁷⁷），这意味着未匹配规模的比较实际衡量的是大小而非安全性。当在声明的资源数量上匹配后……

    arXiv:2608.28021v1 Announce Type: cross  Abstract: Large language models are increasingly used to author Infrastructure-as-Code (IaC), where a single insecure default can be deployed directly into production. Prior evaluations report raw vulnerability counts for model-generated IaC, but without a human baseline they cannot determine whether models are actually worse than engineers. We introduce GenIaC-SecBench, a benchmark of 100 deployment scenarios stratified by architectural complexity, evaluated across 12 model configurations from four vendors, producing 1,196 IaC artifacts scanned by three independent policy engines (Checkov, Trivy, KICS). Critically, we also scan 634 human-authored IaC templates with the same toolchain, providing the first size-matched human security baseline.   Vulnerability density is strongly inverse to artifact size (Spearman $\rho = -0.55$, $p < 10^{-77}$), meaning unmatched comparisons measure size rather than security. When matched on declared-resource cou
    
[^13]: Moirae：用于动态Android恶意软件检测的多模态智能体协作框架

    Moirae: A Multimodal Agent Collaborative Framework for Dynamic Android Malware Detection

    [https://arxiv.org/abs/2608.27994](https://arxiv.org/abs/2608.27994)

    提出Moirae框架，通过多模态智能体协作动态收集运行时证据（视觉欺骗线索、UI状态转换、运行时API行为），并融合多维度行为视图，解决了现有检测器面临的概念漂移和混淆攻击问题。

    

    Android生态系统面临着持续存在且快速演变的恶意软件威胁。现有的机器学习检测器容易受到概念漂移的影响，因为它们依赖于特定实现的特征，而这些特征的分布会随时间发生变化。大语言模型（LLM）提供了强大的语义理解和零样本推理能力，但当前基于LLM的检测器通常依赖于以代码为中心或单一维度的证据，使其容易受到混淆攻击的影响，并限制了全面的行为分析。我们提出了Moirae，一个用于动态Android恶意软件检测的多模态智能体协作框架。Moirae动态收集多模态运行时证据，并采用基于ReAct的专用智能体来分析互补的行为视图。检测过程首先识别视觉欺骗线索，对UI状态转换进行建模，并集成运行时API行为，以融合跨用户可见界面的多维度证据。

    arXiv:2608.27994v1 Announce Type: cross  Abstract: The Android ecosystem faces persistent and rapidly evolving malware threats. Existing machine learning detectors are vulnerable to concept drift because they rely on implementation-specific features whose distributions change over time. Large language models (LLMs) offer strong semantic understanding and zero-shot reasoning, but current LLM-based detectors typically depend on code-centric or single-dimensional evidence, making them susceptible to obfuscation and limiting comprehensive behavior analysis. We present {\sysname}, a multimodal agent collaborative framework for dynamic Android malware detection. {\sysname} dynamically collects multimodal runtime evidence and employs ReAct-based specialized agents to analyze complementary behavioral views. The detection process begins by identifying visual deception cues, modeling UI state transitions, and integrating runtime API behaviors to fuse multi-dimensional evidence across user-visibl
    
[^14]: CHISEL：利用AI赋能的迭代恢复技术还原源代码

    CHISEL-ing Back Source Code with AI-enabled Iterative Recovery

    [https://arxiv.org/abs/2608.27981](https://arxiv.org/abs/2608.27981)

    提出了无需测试套件的CHISEL框架，仅依靠编译器静态分析与覆盖率引导模糊测试的迭代反馈，即可从Ghidra伪C代码中恢复可编译且语义等价的源代码。

    

    反编译旨在从二进制文件中恢复高级、可编译且语义等价的代码。传统反编译器生成的伪C代码难以阅读且无法编译，而近期基于大语言模型（LLM）辅助的方法虽然能生成可读的代码，但在语义上往往不正确。LLM辅助的迭代恢复是一个新兴的研究方向，但先前的工作依赖于预先提供的测试套件来实现语义恢复。在本工作中，我们提出了CHISEL——一个无需测试套件的框架，用于从Ghidra生成的伪C代码中迭代地恢复源代码。CHISEL利用编译器（静态分析）和覆盖率引导的模糊测试器（差分分析）提供的简单而有效的反馈，并通过丰富的可观测信息实现有依据的差异检测与反馈、跨迭代的差异记忆以及最佳候选保留机制。我们在120个ExeBench样本上系统地评估了CHISEL在编译与语义恢复、反馈判定可靠性以及迭代开销方面的表现。（原文摘要在此处截断）

    arXiv:2608.27981v1 Announce Type: cross  Abstract: Decompilation aims to recover high-level, compilable, and semantically equivalent code from binaries. Traditional decompilers produce pseudo-C that is difficult to read and does not compile, while the recent LLM-assisted approaches generate readable, but semantically incorrect code. LLM-aided iterative recovery is an emerging branch of research, but prior works rely on supplied test suites for semantic recovery. In this work, we present CHISEL, a test suite-free framework to iteratively recover source code from Ghidra-derived pseudo-C. CHISEL uses simple yet effective feedback from a compiler (static analysis) and a coverage-guided fuzzer (differential analysis), augmented by rich observables for grounded divergence detection and feedback, cross-iteration divergence memory, and best candidate retention. We systematically evaluate CHISEL for compilation and semantic recovery, feedback oracle soundness, and iteration overhead on 120 ExeB
    
[^15]: GraftyVul：通过真实世界漏洞嫁接合成不安全程序

    GraftyVul: Synthesising Insecure Programs Through Real-World Vulnerability Grafting

    [https://arxiv.org/abs/2608.27928](https://arxiv.org/abs/2608.27928)

    GraftyVul通过将真实世界漏洞嫁接到开源项目中，生成了212个覆盖5种编程语言和23个CWE类别、经过验证且可利用的漏洞程序，同时兼顾了多样性、可复现性和真实性。

    

    漏洞数据集支撑着广泛的安全研究，包括漏洞检测、自动化修复和安全代码生成。然而，现有的数据集至少牺牲了三个理想特性中的一个：多样性（语言或漏洞类型）、可复现性/可执行性或真实性。因此，我们提出了GraftyVul，这是一个通过将真实世界的漏洞嫁接到开源项目中来构建漏洞程序的系统。这使得数据集立足于真实环境中观察到的漏洞，同时利用已知良好的构建和测试环境，使漏洞利用验证脚本能够保证引入的漏洞成功地改变程序的行为。使用GraftyVul，我们生成了212个经过验证且可利用的漏洞程序，涵盖五种编程语言（Python、TypeScript、Java、Go和C#）以及23个CWE类别。为了评估保真度，我们引入了一种语言……（摘要内容不完整）

    arXiv:2608.27928v1 Announce Type: cross  Abstract: Vulnerability datasets underpin a wide range of security research, including vulnerability detection, automated remediation, and secure code generation. However, existing datasets sacrifice at least one of three desirable properties: diversity (of language or vulnerability type), reproducibility/executability, or realism. We therefore present GraftyVul, a system that constructs vulnerable programs by grafting real-world vulnerabilities into open-source projects. This grounds the dataset in vulnerabilities observed in real-world contexts while harnessing known good build and test environments, enabling exploit-verification scripts to guarantee that an introduced vulnerability successfully alters a program's behaviour. Using GraftyVul, we generate 212 verified and exploitable vulnerable programs spanning five programming languages (Python, TypeScript, Java, Go, and C#) across 23 CWE categories. To evaluate fidelity, we introduce a langua
    
[^16]: AI辅助定性数据分析中的反模式：软件工程研究人员面临的诱惑与陷阱目录

    Antipatterns in AI-assisted Qualitative Data Analysis: A Catalog of Temptations and Pitfalls for Software Engineering Researchers

    [https://arxiv.org/abs/2608.27927](https://arxiv.org/abs/2608.27927)

    本文基于作者数十年的定性研究经验，首次系统性地提出了AI辅助定性数据分析中的反模式目录，将其按影响程度递进划分为“危险驱动因素、操作失误和分析失败”三大类别，旨在帮助软件工程研究人员识别并规避AI辅助分析中损害研究严谨性与有效性的方法论风险。

    

    AI辅助的定性数据分析（QDA）为简化软件工程（SE）研究流程提供了前所未有的机遇，然而不加批判地使用可能会损害分析的严谨性，并导致低质量研究加速涌入该领域。虽然战术层面的最佳实践会随时间自然演进，但目前软件工程研究人员在尝试AI辅助QDA时，缺乏识别和缓解方法论风险的战略性指导。基于我们数十年的定性软件工程研究专业知识与经验，以及对AI辅助QDA新兴格局的理解，本文提出了AI辅助QDA中的反模式目录——一组初看似乎有利、但最终会破坏分析严谨性与有效性的假设和实践。这些反模式被划分为三个类别，反映其影响的逐步升级：危险驱动因素、操作失误和分析失败。随着更多SE（原文在此处截断）

    arXiv:2608.27927v1 Announce Type: new  Abstract: AI-assisted qualitative data analysis (QDA) offers unprecedented opportunities to streamline software engineering (SE) research, yet uncritical use risks compromising analytical rigor and flooding the field with accelerated production of low-quality research. While tactical best practices will naturally evolve over time, SE researchers currently lack strategic guidance to identify and mitigate methodological risks when attempting AI-assisted QDA. Based on our decades of qualitative SE research expertise and experience combined with an understanding of the emerging landscape of AI-assisted QDA, this paper presents a catalog of antipatterns in AI-assisted QDA - a set of assumptions and practices that initially appear advantageous but ultimately undermine analytical rigor and validity. The antipatterns are grouped into three categories reflecting escalating impact: Dangerous Drivers, Operational Missteps, and Analytical Failures. As more SE
    
[^17]: 解耦是必需的：优化与混淆下变换无关的反编译代码恢复

    Decoupling is a Necessity: Transformation-Agnostic Decompiled Code Recovery under Optimization and Obfuscation

    [https://arxiv.org/abs/2608.27889](https://arxiv.org/abs/2608.27889)

    提出ReSource——首个面向变换无关源代码恢复的多阶段LLM框架，通过将二进制与源代码的差异解耦为词汇、语法、语义三个正交层次，在编译器优化与代码混淆的双重干扰下实现高质量的反编译代码恢复。

    

    逆向工程对于软件安全分析和漏洞检测至关重要。反编译，即将二进制代码提升为高级伪代码的过程，是这一任务的核心。然而，生产环境的二进制文件是充满敌意的环境：激进的编译器优化和对抗性混淆会共同破坏控制结构、模糊变量意图并伪装高级程序逻辑。因此，现有的基于大语言模型（LLM）的反编译工具经常遭受结构崩溃和语义幻觉的困扰。我们提出了ReSource，这是首个专为变换无关的源代码恢复设计的多阶段LLM框架。为了应对这些相互交织的失真，ReSource将二进制与源代码之间的差异概念化为三个正交层次，即词汇层、语法层和语义层，并相应地解耦恢复过程。首先，为了让LLM有据可依并防止逻辑漂移，它从cu（摘要在此处被截断）……

    arXiv:2608.27889v1 Announce Type: new  Abstract: Reverse engineering is essential for software security analysis and vulnerability detection. Decompilation, the process of lifting binaries to high-level pseudocode, is central to this task. However, production binaries are hostile environments: aggressive compiler optimizations and adversarial obfuscation jointly mangle control structures, obscure variable intents, and disguise high-level program logic. Consequently, existing LLM-based decompilation tools frequently suffer from structural collapse and semantic hallucinations. We present ReSource, the first multi-phase LLM framework designed for transformation-agnostic source recovery. To tackle these intertwined distortions, ReSource conceptualizes the binary-to-source discrepancies into three orthogonal tiers, namely lexical, syntactic, and semantic, and decouples the recovery process accordingly. First, to ground the LLM and prevent logic drift, it retrieves empirical priors from a cu
    
[^18]: DBRepro：基于混合约束求解方法的慢查询复现自动化数据库合成

    DBRepro: Automated Database Synthesis via a Hybrid Constraint-Solving Approach for Reproducing Slow Queries

    [https://arxiv.org/abs/2608.27822](https://arxiv.org/abs/2608.27822)

    DBRepro通过混合约束求解方法，从非侵入式元数据自动合成代理数据库，在保留全局统计分布的同时满足精确的局部基数约束，使查询优化器生成与慢查询相同的物理执行计划，从而实现安全的离线性能诊断。

    

    慢查询经常在数据库管理系统中造成严重的性能瓶颈。在线诊断其根本原因可能会加剧资源争用，而数据隐私法规通常禁止将生产数据复制到测试环境。因此，从非侵入式元数据合成一个代理数据库，使查询优化器生成相同的物理执行计划，对于离线诊断至关重要。高保真复现需要在保留全局统计分布的同时强制满足精确的局部基数。现有的数据驱动和工作负载感知方法无法同时满足这两个要求。我们提出了DBRepro，一个自动化的端到端框架，它将数据库生成形式化为一个受约束的分布合成问题。DBRepro从轻量级列统计信息初始化全局分布，从目标查询中提取执行约束，并逐步……（原文摘要在此处截断）

    arXiv:2608.27822v1 Announce Type: cross  Abstract: Slow queries frequently cause severe performance bottlenecks in database management systems. Diagnosing their root causes online risks exacerbating resource contention, while data privacy regulations often prohibit copying production data to test environments. Synthesizing a proxy database from non-intrusive metadata that induces the query optimizer to generate the same physical execution plans is therefore critical for offline diagnosis. High-fidelity reproduction requires preserving global statistical distributions while enforcing exact local cardinalities. Existing data-driven and workload-aware approaches cannot satisfy both requirements simultaneously.   We present DBRepro, an automated end-to-end framework that formulates database generation as a constrained distribution synthesis problem. DBRepro initializes a global distribution from lightweight column statistics, extracts execution constraints from target queries, and progress
    
[^19]: RiskBlend：一种用于机器学习回归测试的测试输入优先级排序多信号框架

    RiskBlend: A Multi-Signal Framework for Test Input Prioritization in Machine Learning Regression Testing

    [https://arxiv.org/abs/2608.27704](https://arxiv.org/abs/2608.27704)

    RiskBlend提出了一种与分类器无关的多信号测试输入优先级排序框架，通过融合历史失败模式、预测偏移、决策边界偏移和邻域变化四种互补风险信号，在有限验证预算下更有效地发现机器学习模型重训练引发的回归缺陷。

    

    当机器学习分类器被重新训练时，先前模型版本正确分类的输入可能会被更新后的版本错误分类，从而产生回归缺陷。由于验证预测结果与真实标签是否一致可能需要人工标注、专家审查或昂贵的仿真，而非廉价的模型推理，因此检测这些回归缺陷的代价很高。测试输入优先级排序通过将输入进行排序来应对这一问题，使有限的验证预算能够揭示尽可能多的回归缺陷。现有方法主要依赖单模型的置信度分数，未能利用预测结果、决策边界和局部邻域在模型版本之间发生的变化。我们提出了RiskBlend，一个与分类器无关的优先级排序框架，它结合了四种互补的风险信号：历史失败模式、预测偏移、决策边界偏移和邻域变化。这些信号通过……（原文摘要在此处截断）

    arXiv:2608.27704v1 Announce Type: new  Abstract: When machine learning classifiers are retrained, inputs correctly classified by the previous model version may be misclassified by the updated version, creating regression faults that are costly to detect because verifying predictions against ground truth may require human annotation, expert review, or expensive simulation rather than inexpensive model inference. Test input prioritization addresses this problem by ranking inputs so that a limited verification budget reveals as many regression faults as possible. Existing approaches rely predominantly on single-model confidence scores and do not exploit how predictions, decision boundaries, and local neighborhoods change between model versions. We propose RiskBlend, a classifier-agnostic prioritization framework that combines four complementary risk signals: historical failure patterns, prediction shift, decision-boundary shift, and neighborhood change. These signals are combined using va
    
[^20]: 将法规操作化为代码：一个增强软件工程中大语言模型选择治理与合规性的模型

    Operationalizing Regulations into Code: A Model to Enhance Governance and Compliance in LLM Selection for Software Engineering

    [https://arxiv.org/abs/2608.27703](https://arxiv.org/abs/2608.27703)

    本文通过设计科学研究方法提出一个三层模型，将欧盟《人工智能法案》、NIST RMF、GDPR等法规要求操作化为包含否决性和加权评分标准的多准则决策矩阵，从而增强软件工程项目中选择大语言模型时的治理与合规性。

    

    将大语言模型（LLM）集成到软件开发生命周期（SDLC）中可以提升开发人员的生产力，但在模型选择过程中也会引入安全、隐私和合规方面的风险。欧盟《人工智能法案》、NIST人工智能风险管理框架（RMF）、《通用数据保护条例》（GDPR）、巴西《通用个人数据保护法》（LGPD）以及ISO/IEC 42001等法规和框架确立了各项义务，但这些义务往往难以转化为技术决策中可操作的标准。本文提出了一个支持软件工程项目中LLM选择的治理与合规模型。该模型通过设计科学研究（DSR）方法开发，采用三层结构：(i) 监管要求，(ii) 组织治理能力，通过包含否决性标准和加权评分标准的多准则决策矩阵予以实例化，(iii) 生产力与可持续性（摘要内容在此处截断）。

    arXiv:2608.27703v1 Announce Type: new  Abstract: Integrating Large Language Models (LLMs) into the Software Development Life Cycle (SDLC) can improve developer productivity, but it also introduces security, privacy, and compliance risks during model selection. Regulations and frameworks such as the EU AI Act, the NIST AI Risk Management Framework (RMF), the General Data Protection Regulation (GDPR), the Lei Geral de Prote\c{c}\~ao de Dados (LGPD), and ISO/IEC 42001 establish obligations that are often difficult to translate into operational criteria for technical decision-making. This paper proposes a model to support governance and compliance in LLM selection for software engineering projects. The model is developed through Design Science Research (DSR) and is structured in three layers: (i) regulatory requirements, (ii) organizational governance capabilities, instantiated by a multi-criteria decision matrix with knock-out and weighted scoring criteria, and (iii) productivity and sust
    
[^21]: 从提示词语言特征预测大语言模型性能：需求工程中的实证研究

    Predicting LLM Performance from Prompt Linguistic Features: An Empirical Study in Requirements Engineering

    [https://arxiv.org/abs/2608.27621](https://arxiv.org/abs/2608.27621)

    该研究证明提示词的可测量语言特征能够在推理之前预测大语言模型的性能，从而为需求工程中的提示词选择与优化提供了一种低成本、无需反复试错的方法。

    

    背景。大语言模型（LLM）的输出对提示词的表述方式高度敏感：措辞上的微小变化就可能显著影响输出质量。这在软件工程中尤为重要，因为提示词引导着需求分析、代码生成和制品合成等任务。糟糕的表述会产生不可靠的制品，然而从业者缺乏在推理前评估提示词的原则性方法，使得提示词的选择依赖于昂贵的大模型调用和反复试错的优化。目标。我们研究提示词的可测量语言属性能否在推理前预测大语言模型的性能，从而实现低成本的提示词选择与优化，并在面向F1、F2、精确率和召回率的二元需求分类任务上进行了验证。方法。我们通过改变30个语言度量指标，从100个初始提示词生成了9,000个语言上受控的提示词变体，并使用五个开源大语言模型在625条标注需求上进行评估。回归预测器通过分层（摘要在此处截断）……

    arXiv:2608.27621v1 Announce Type: new  Abstract: Background. LLM outputs are highly sensitive to prompt formulation: small wording changes can substantially affect output quality. This matters in software engineering, where prompts guide requirements analysis, code generation, and artefact synthesis. Poor formulations yield unreliable artefacts, yet practitioners lack principled ways to assess a prompt before inference, making selection depend on costly LLM calls and trial-and-error refinement. Aims. We investigate whether measurable linguistic properties of prompts can predict LLM performance before inference, enabling low-cost prompt selection and refinement, validated on binary requirements classification targeting F1, F2, precision, and recall. Method. We generate 9,000 linguistically controlled prompt variants from 100 initial prompts by varying 30 linguistic metrics, evaluated with five open-source LLMs on 625 annotated requirements. Regression predictors are trained via stratifi
    
[^22]: 图像增强作为基于深度学习的图像检索系统的测试生成方法

    Image Augmentation as Test Generation for Deep Learning-Based Image Retrieval Systems

    [https://arxiv.org/abs/2608.27502](https://arxiv.org/abs/2608.27502)

    本文系统梳理了50种图像增强与生成技术并建立十类分类体系，同时通过大规模实证研究评估这些技术作为基于嵌入的图像检索系统测试生成器的有效性。

    

    确保基于深度学习的图像检索系统的可靠性是一项软件工程挑战。本文提出了双重贡献：(1) 对图像增强和生成技术的文献综述，识别出50种技术并将其组织为十个类别的分类体系；(2) 一项大规模实证研究，评估这些技术作为基于嵌入的图像检索系统的测试生成器的有效性。增强后的图像使用Amazon Titan和OpenCLIP进行嵌入，并从四个分析维度进行评估：(1) 嵌入空间相似性，(2) 通过四种估计器测量的嵌入不确定性，(3) 由LLaVA评分的语义真实性，(4) 检索失败率。实验在三个数据集上进行：CIFAR-10、ImageNet-1K以及来自工业合作伙伴的数据集。在所有评估的数据集和嵌入模型上，以及在所测试的单一严重性级别下……

    arXiv:2608.27502v1 Announce Type: new  Abstract: Ensuring the reliability of deep learning-based image retrieval systems is a software engineering challenge. This paper presents a dual contribution: (1) a literature review of augmentation and generation techniques which resulted in the identification of 50 techniques which we organized into a ten-category taxonomy, and (2) a large-scale empirical study that evaluates these techniques as test generators for embedding-based image retrieval systems. Augmented images are embedded using Amazon Titan and OpenCLIP, and evaluated across four analytical dimensions: (1) embedding-space similarity, (2) embedding uncertainty measured via four estimators, (3) semantic realism scored by LLaVA, and (4) retrieval failure rate. Experiments are performed on three datasets: CIFAR-10, ImageNet-1K, and a dataset from an industrial partner (March Networks). Across all evaluated datasets and embedding models, and under the single severity level tested for ea
    
[^23]: 面向智能体技能轨迹的有据可依清单式部分给分评估

    Grounded Checklist Partial Credit for Agent Skill Trajectories

    [https://arxiv.org/abs/2608.27487](https://arxiv.org/abs/2608.27487)

    提出GCPC方法，由人类一次性定义可复用规则、大语言模型基于任务指令和官方验证器实例化任务专属清单，实现对智能体轨迹的可信部分给分评估，克服了二值任务成功率掩盖部分进展的局限。

    

    语言模型智能体越来越多地在交互式环境中处理长时程任务，然而对它们的评估通常依赖于任务级别的成功率，即将整个执行轨迹简化为任务是否通过官方验证器。这种二值评分掩盖了部分进展，对于程序性智能体技能评估尤其受限，因为一项技能可以在不改变最终结果的情况下改变执行过程。虽然清单可以通过对各个任务要求分别评分来提供更细粒度的评估，但高昂的人工编写成本和不可靠的自动生成使得可信评估难以规模化。为应对这些挑战，我们提出了有据可依清单部分给分，这是一种由人类主导、由大语言模型实例化的智能体轨迹部分给分评估方法。人类只需一次性定义可复用的规则，大语言模型随后基于任务指令和官方验证器为具体任务实例化出一份清单。

    arXiv:2608.27487v1 Announce Type: new  Abstract: Language-model agents increasingly tackle long-horizon tasks in interactive environments, yet their evaluation commonly relies on task-level success rates by reducing an entire execution trajectory to whether the task passes an official verifier. This binary score hides partial progress and is particularly limited for procedural agent skill evaluations, since a skill can alter execution without changing the final outcome. While checklists provide finer-grained evaluation by scoring individual task requirements, costly manual authoring and unreliable automatic generation make trustworthy evaluation difficult to scale. To address these challenges, we introduce Grounded Checklist Partial Credit (GCPC), a human-governed and LLM-instantiated partial-credit evaluation of agent trajectories. Humans define reusable rules once, from which an LLM instantiates a task-specific checklist grounded in the task instruction and official verifier. To keep
    
[^24]: 2026年下一代科学计算生态系统研讨会报告：利用社区、软件和人工智能促进跨学科团队科学

    Report of the 2026 Workshop on Next-Generation Ecosystems for Scientific Computing: Harnessing Community, Software, and AI for Cross-Disciplinary Team Science

    [https://arxiv.org/abs/2608.26519](https://arxiv.org/abs/2608.26519)

    本报告基于2026年研讨会，提炼出科学计算生态系统未来发展的四大战略主题和八项社区行动优先事项，强调通过社会技术协同设计整合人工智能、软件和跨学科合作。

    

    科学计算正在经历快速转型，人工智能、异构计算、自动化和数据密集型研究的进步不仅重塑了计算工具，还重塑了支持科学发现的机构、劳动力模式和协作实践。本报告综合了2026年下一代科学计算生态系统研讨会的见解，该研讨会是聚焦于通过社会技术协同设计加强科学计算生态系统的三年系列会议中的第二届。研讨会讨论确定了四个相互依存的战略主题：面向人工智能驱动的科学发现的软件生态系统；信任、验证和可追溯性；人机协作与范式转变；以及劳动力、教学和治理。报告将这些主题转化为八项社区行动优先事项，涵盖共享研究基础设施、信任与可追溯性、用户体验、人机协作等方面。

    arXiv:2608.26519v1 Announce Type: cross  Abstract: Scientific computing is undergoing rapid transformation as advances in artificial intelligence, heterogeneous computing, automation, and data-intensive research reshape not only computational tools but also the institutions, workforce models, and collaborative practices that support scientific discovery. This report synthesizes insights from the 2026 Workshop on Next-Generation Ecosystems for Scientific Computing, the second in a three-year series focused on strengthening scientific computing ecosystems through socio-technical co-design. Workshop discussions identified four interdependent strategic themes: software ecosystems for AI-enabled scientific discovery; trust, validation, and traceability; human-AI teaming and paradigm shifts; and workforce, pedagogy, and governance. The report translates these themes into eight priorities for community action spanning shared research infrastructure, trust and traceability, user experience, hu
    
[^25]: SPECMINE：一个大规模的规范驱动开发工件语料库

    SPECMINE: A Large-Scale Corpus of Spec-Driven Development Artifacts

    [https://arxiv.org/abs/2608.25202](https://arxiv.org/abs/2608.25202)

    我们提出了SPECMINE，这是首个大规模语料库，通过两次普查系统地捕捉了GitHub上规范驱动开发工件，为研究规范如何转化为代码提供了基础数据。

    

    arXiv:2608.25202v1 公告类型：新 摘要：规范驱动开发（SDD）是一种快速兴起的新实践，其中由开发者编写、或（更常见地）由AI工具起草再由开发者整理的、结构化自然语言规范，驱动AI编码代理的实现。自2025年以来，一波工具（如GitHub Spec Kit [3]、OpenSpec [4]、AWS Kiro [5]以及数十种其他工具）已经出现，但这些工具产生的工件从未被大规模研究过。我们提出了SPECMINE，一个通过两次普查捕捉公共GitHub仓库中SDD的语料库：一次广泛普查覆盖了大多数工具的spec.md/specs.md文件（涵盖73,030个仓库中的470,795个文件，归属于17个命名工具），以及一次针对Kiro独特的需求/设计/任务布局的普查（涵盖12,910个仓库中的98,574个文件）。每个规范都附有完整的仓库元数据、完整的提交历史以及解析后的文档结构。规范如何转化为代码本身就是一个开放问题，因此对于...

    arXiv:2608.25202v1 Announce Type: new  Abstract: Spec-Driven Development (SDD) is a fast-emerging practice in which a structured natural-language specification, written by a developer, or (more often) drafted by an AI tool and then curated by the developer, drives an AI coding agent's implementation. A wave of tooling (GitHub Spec Kit [3], OpenSpec [4], AWS Kiro [5], and dozens of others) has appeared since 2025, yet the artifacts these tools produce have never been studied at scale. We present SPECMINE, a corpus that captures SDD in public GitHub repositories through two censuses: a broad census of spec.md/specs.md files covering most tools (470,795 files across 73,030 repositories, attributed to 17 named tools), and a Kiro census of its distinct requirements/design/tasks layout (98,574 files across 12,910 repositories). Each spec is enriched with full repository metadata, complete commit history, and parsed document structure. How a spec becomes code is itself an open question, so fo
    
[^26]: VR-Themis：一个可扩展的虚拟现实应用克隆检测框架

    VR-Themis: A Scalable Framework for Virtual Reality Application Clone Detection

    [https://arxiv.org/abs/2608.13290](https://arxiv.org/abs/2608.13290)

    本文提出了基于“层次-对象-行为”（HOB）模型的两阶段VR应用克隆检测框架VR-Themis，通过粗粒度统计特征聚类实现大规模数据集上的可扩展性，再对可疑应用进行细粒度深入分析，弥补了现有移动应用克隆检测方法无法有效检测VR应用克隆的不足。

    

    移动应用的重新打包（又称应用克隆）不仅威胁移动用户的安全与隐私，还侵犯了原始应用开发者的版权。然而，现有主要针对移动平台（如Android）的检测方法无法捕捉虚拟现实（VR）的本质特征，因此难以有效检测克隆的VR应用，而克隆VR应用在VR市场中经常成为非法用户攻击的目标。考虑到VR应用的独特特征，本文提出了一个基于“层次-对象-行为”（Hierarchy-Object-Behaviour, HOB）模型的两阶段应用克隆检测框架，即VR-Themis。首先，VR-Themis利用粗粒度阶段根据可检索的统计特征对应用进行聚类，使该工具能够扩展到大规模VR应用数据集。然后，在细粒度阶段，VR-Themis对可疑应用（在第一阶段中识别出的）进行深入分析……（原文摘要在此处截断）

    arXiv:2608.13290v1 Announce Type: cross  Abstract: Repackaging of mobile applications (aka app cloning) not only threatens the security and privacy of mobile users but also infringes upon the copyright of the original app developers. However, existing detection methods that primarily focus on mobile platforms (such as Android) fail to capture the essential features of virtual reality (VR). Consequently, they are inadequate for effectively detecting cloned VR apps, which have often been targeted by illegal users in the VR market. Considering the unique features of VR apps, this paper proposes a two-stage app clone detection framework, namely VR-Themis, based on \emph{Hierarchy-Object-Behaviour} (HOB). Firstly, VR-Themis exploits the coarse-grained stage to cluster apps based on their retrievable statistical features, making this tool scalable to large-scale VR app datasets. Then, in the fine-grained stage, VR-Themis performs in-depth analysis of the suspicious apps (identified in the fi
    
[^27]: 面向配备框架智能体的定势转换行为测试

    Set-shifting Behavioral Test for Harnessed Agents

    [https://arxiv.org/abs/2607.13396](https://arxiv.org/abs/2607.13396)

    该论文借鉴认知心理学中的“定势转换”概念，提出了一种通过在冗余工具库中隐藏地切换可靠工具组来测试LLM智能体适应能力的行为测试方法，并发现不同模型面对相同切换时表现出截然不同的行为模式。

    

    当可靠的工具在持续会话中悄然发生变化时，LLM智能体的工具选择会发生什么？我们从认知心理学中借鉴了“定势转换”的概念，研究智能体对隐藏可靠性变化的适应能力。我们为LLM智能体设计的认知测试挂载了冗余的工具与技能库，其中多个工具可以解决同一任务，但在隐藏的可靠性上存在差异。通过分支式的调度安排，我们在环境中切换可靠工具组，并与稳定的对照组进行比较，从而能够分离出每一次切换对智能体行为的独立影响。我们在一组配备框架的LLM上开展了研究，结果表明同一组切换在不同模型中引发了截然不同的行为：有些模型在几轮之内就固守某种固定模式，而另一些则持续变化。能力较弱的模型往往会忽略可靠工具组，而前沿模型则会在调用其他工具组的同时持续调用可靠工具组。（原文摘要在此处截断）

    arXiv:2607.13396v2 Announce Type: replace-cross  Abstract: What happens to an LLM agent's tool choice when the reliable tool silently changes within an ongoing session? We borrow the notion of set-shifting from cognitive psychology to study how well agents adapt to hidden reliability shifts. Our cognitive test for LLM agents mounts libraries of redundant tools and skills, in which many tools solve the same task but differ in hidden reliability. Using a branching schedule, we shift the reliable tool group in the environment and compare it with a stable control, allowing us to isolate the effect of each shift on the agent's behavior. We conduct our study on a panel of LLMs equipped with harnesses and show that the same set of shifts results in distinct behaviors across models: some latch onto a fixed routine within a few turns, whereas others continue to vary. Less capable models often omit the reliable tool group, while frontier models keep calling it alongside the other groups. We intr
    
[^28]: EvoRepair：通过基于经验的自进化增强漏洞修复智能体

    EvoRepair: Enhancing Vulnerability Repair Agents Through Experience-Based Self-Evolution

    [https://arxiv.org/abs/2605.30105](https://arxiv.org/abs/2605.30105)

    EvoRepair是首个基于经验自进化的自动化漏洞修复智能体框架，通过循环的学习-修复过程实现修复经验的积累、精炼与跨漏洞复用，从而提升LLM的漏洞修复能力。

    

    大型语言模型（LLMs）在自动化漏洞修复（AVR）方面展现出巨大潜力，但仍面临若干局限性，包括缺乏漏洞内部的经验积累以及缺乏跨漏洞的经验复用。因此，LLMs在迭代修复过程中可能反复犯类似的错误，且未能充分利用历史漏洞中宝贵的修复知识。为解决这些挑战，我们提出了EvoRepair，这是首个基于经验的自进化AVR智能体框架，使LLMs能够在长周期的漏洞修复任务中积累、精炼并利用领域特定知识。EvoRepair遵循循环的“学习-修复”过程：检索相关的历史经验以指导修复，从修复轨迹中提取新经验，并通过质量感知评分更新经验库。我们在PAT上对EvoRepair与12个代表性漏洞修复基线进行了评估。

    arXiv:2605.30105v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have shown promise for automated vulnerability repair (AVR), but they still face several limitations, including the lack of intra-vulnerability experience accumulation and the lack of cross-vulnerability experience reuse. As a result, LLMs may repeatedly make similar mistakes during iterative repair and underutilize valuable repair knowledge from historical vulnerabilities. To address these challenges, we propose EvoRepair, the first experience-based self-evolving AVR agent framework that enables LLMs to accumulate, refine, and leverage domain-specific knowledge across long-horizon vulnerability repairs. EvoRepair follows a cyclic learn-and-repair process that retrieves relevant past experiences to guide repair, extracts new experiences from repair trajectories, and updates an experience bank using quality-aware scoring. We evaluate EvoRepair against 12 representative vulnerability repair baselines on PAT
    
[^29]: 超越输出正确性：编程任务中大语言模型推理能力的基准测试与评估

    Beyond Output Correctness: Benchmarking and Evaluating Large Language Model Reasoning in Coding Tasks

    [https://arxiv.org/abs/2604.12379](https://arxiv.org/abs/2604.12379)

    该论文提出了首个覆盖代码生成、摘要与分类三类编程任务的推理质量评估基准CodeRQ-Bench，并通过分析评估器失配案例得出设计启示，进而提出结合证据验证与歧义感知评分修正的两阶段评估器VERA，显著提升了编程任务中大语言模型推理质量的评估效果。

    

    大语言模型（LLM）在解决编程任务时越来越依赖显式推理，然而评估这种推理的质量仍然具有挑战性。现有的推理评估器并非为编程任务设计，而当前的基准测试主要关注代码生成，其他编程任务在很大程度上尚未被探索。我们提出了CodeRQ-Bench，这是首个用于评估大语言模型在三类编程任务（生成、摘要和分类）中推理质量的基准。利用该基准，我们分析了来自现有评估器的1,069个失配案例，识别出五个反复出现的局限性，并由此得出四项针对编程任务推理评估的设计启示。基于这些启示，我们提出了VERA——一种结合基于证据的验证与歧义感知评分修正的两阶段评估器。在CodeRQ-Bench上的实验表明，VERA在四个数据集上持续优于强大的基线方法。

    arXiv:2604.12379v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly rely on explicit reasoning to solve coding tasks, yet evaluating the quality of this reasoning remains challenging. Existing reasoning evaluators are not designed for coding, and current benchmarks focus primarily on code generation, leaving other coding tasks largely unexplored. We introduce CodeRQ-Bench, the first benchmark for evaluating LLM reasoning quality across three coding task categories: generation, summarization, and classification. Using this benchmark, we analyze 1,069 mismatch cases from existing evaluators, identify five recurring limitations, and derive four design insights for reasoning evaluation in coding tasks. Guided by these insights, we propose VERA, a two-stage evaluator that combines evidence-grounded verification with ambiguity-aware score correction. Experiments on CodeRQ-Bench show that VERA consistently outperforms strong baselines across four datasets, imp
    
[^30]: “无穷无尽的AI垃圾”：开发者如何讨论AI辅助软件开发的负担

    "An Endless Stream of AI Slop": How Developers Discuss the Burden of AI-Assisted Software Development

    [https://arxiv.org/abs/2603.27249](https://arxiv.org/abs/2603.27249)

    该研究通过定性分析1,154条Reddit和Hacker News帖子，首次系统揭示了开发者对AI生成的低质量内容（“AI slop”）给软件开发带来负担的感知与应对，并将其框架化为一种公地悲剧——个人生产力收益以牺牲审查者、维护者和整个社区的利益为代价。

    

    “AI垃圾”（AI slop），即低质量的AI生成内容，正日益影响着软件开发，从生成的代码和拉取请求到文档和错误报告。然而，关于开发者如何感知和应对这一现象的实证研究仍然有限。我们对1,154条Reddit和Hacker News帖子中开发者讨论AI slop的内容进行了定性分析，构建了一个包含15个编码的编码本，并归纳为三个主题集群：审查摩擦（AI slop如何加重审查者的负担、侵蚀信任并促使人们采取应对措施）、质量退化（对代码库、知识资源和开发者能力的损害）以及驱动力与后果（系统性激励、强制采用、工艺侵蚀和劳动力动荡）。我们的发现将AI slop阐释为一种公地悲剧，即个人生产力的提升将成本外部化转嫁给审查者、维护者和更广泛的社区。我们报告了开发者提出的担忧。

    arXiv:2603.27249v4 Announce Type: replace  Abstract: "AI slop", that is, low-quality AI-generated content, is increasingly affecting software development, from generated code and pull requests to documentation and bug reports. However, there is limited empirical research on how developers perceive and respond to this phenomenon. We qualitatively analyzed how developers discuss AI slop in 1,154 Reddit and Hacker News posts, developing a codebook of 15 codes organized into three thematic clusters: Review Friction (how AI slop burdens reviewers, erodes trust, and prompts countermeasures), Quality Degradation (damage to codebases, knowledge resources, and developer competence), and Forces and Consequences (systemic incentives, mandated adoption, craft erosion, and workforce disruption). Our findings frame AI slop as a tragedy of the commons, where individual productivity gains externalize costs onto reviewers, maintainers, and the broader community. We report the concerns developers raise 
    
[^31]: 不要将代码视为自然语言：对仓库级代码生成及更广泛领域的影响

    Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond

    [https://arxiv.org/abs/2602.11671](https://arxiv.org/abs/2602.11671)

    提出Hydra框架，将代码视为结构化代码而非自然语言，通过结构感知索引策略解决现有RAG方法因分块和相似性检索导致的代码结构关系丢失问题，从而提升仓库级代码生成的效果。

    

    面向代码的大型语言模型在独立代码补全与生成方面已展现出卓越的成功，有时甚至超越人类的表现，然而在仓库级场景中，由于需要跨文件依赖和结构化上下文，其有效性会显著下降。现有的检索增强生成（RAG）方法通常借鉴自然语言处理（NLP）的策略，依赖于基于分块的索引和基于相似性的检索。分块会导致代码单元之间连贯性的丢失，并忽视结构关系，而基于相似性的检索方法则经常遗漏功能性相关的依赖，例如辅助函数、类或全局变量。为了解决这些局限性，我们提出了Hydra，一个将代码视为结构化代码而非自然语言的仓库级代码生成框架。我们的方法引入了（i）一种结构感知的索引策略，用于表示仓库（摘要在此处被截断）

    arXiv:2602.11671v2 Announce Type: replace  Abstract: Large language models for code (CodeLLMs) have demonstrated remarkable success in standalone code completion and generation, sometimes even surpassing human performance, yet their effectiveness diminishes in repository-level settings where cross-file dependencies and structural context are essential. Existing Retrieval-Augmented Generation (RAG) approaches often borrow strategies from NLP, relying on chunking-based indexing and similarity-based retrieval. Chunking results in the loss of coherence between code units and overlooks structural relationships, while similarity-driven methods frequently miss functionally relevant dependencies such as helper functions, classes, or global variables. To address these limitations, we present Hydra, a repository-level code generation framework that treats code as structured code rather than natural language. Our approach introduces (i) a structure-aware indexing strategy that represents reposito
    
[^32]: ASA：面向工具调用智能体的无需骨干训练的表示工程方法

    ASA: Backbone-Training-Free Representation Engineering for Tool-Calling Agents

    [https://arxiv.org/abs/2602.04935](https://arxiv.org/abs/2602.04935)

    ASA提出了一种无需骨干训练的推理时激活引导方法，弥合了LLM智能体“知道该用工具却不敢用”的表示-行为鸿沟，在不微调模型的情况下显著提升了特定领域工具调用的可靠性。

    

    将大语言模型（LLM）智能体适配到特定领域的工具调用，在接口不断演变的情况下仍然非常脆弱。提示词和模式（schema）工程易于部署，但在分布偏移和严格解析器下往往表现脆弱；而持续的参数高效微调虽然能提高可靠性，却以训练成本、维护成本和潜在的灾难性遗忘为代价。我们发现了一种关键的“懒惰智能体”（Lazy Agent）失效模式：工具使用的必要性几乎可以完美地从中间层激活中解码出来，但模型在进入工具模式时仍然表现得过于保守，这揭示了一种表示与行为之间的鸿沟。我们提出了激活引导适配器（Activation Steering Adapter, ASA），这是一种无需训练的推理时控制器，它执行单次中间层干预，并通过路由器条件化的引导向量混合以及探针引导的符号门控来针对特定工具领域，从而在放大真实意图的同时抑制虚假触发。在基于Qwen2.5-1.5B的MTU-Bench上，ASA显著提高了严格工具使用的F1分数。

    arXiv:2602.04935v4 Announce Type: replace  Abstract: Adapting LLM agents to domain-specific tool calling remains notably brittle under evolving interfaces. Prompt and schema engineering is easy to deploy but often fragile under distribution shift and strict parsers, while continual parameter-efficient fine-tuning improves reliability at the cost of training, maintenance, and potential forgetting. We identify a critical Lazy Agent failure mode where tool necessity is nearly perfectly decodable from mid-layer activations, yet the model remains conservative in entering tool mode, revealing a representation-behavior gap. We propose Activation Steering Adapter (ASA), a training-free, inference-time controller that performs a single-shot mid-layer intervention and targets tool domains via a router-conditioned mixture of steering vectors with a probe-guided signed gate to amplify true intent while suppressing spurious triggers. On MTU-Bench with Qwen2.5-1.5B, ASA improves strict tool-use F1 f
    

