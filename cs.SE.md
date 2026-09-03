# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Post-Training Language Models for Gold-Medal Performance in Coding Competitions](https://arxiv.org/abs/2609.02849) | 该研究通过结合大规模题目筛选、监督微调、强化学习以及反馈驱动的测试时计算策略 GenCorrect，使语言模型在 IOI 2025 编程竞赛中取得了超越金牌分数线（438.3 分）的成绩（Nano-CC 达 468 分，Ultra-CC 达 502 分）。 |
| [^2] | [ShikumiMiner: Mining Recurring Implementation Patterns in AI Codebases](https://arxiv.org/abs/2609.02789) | 本文提出ShikumiMiner静态分析框架，结合抽象语法树与控制流图特征，利用多标签随机森林模型检测、比较并分类C++本地大语言模型代码库中的重复实现模式，为LLM应用开发者提供设计洞察。 |
| [^3] | [Type Hints in Python Libraries and Frameworks: An Empirical Analysis of Adoption and Maintenance](https://arxiv.org/abs/2609.02782) | 该论文通过对1,000个热门GitHub仓库的大规模实证分析，首次系统揭示了Python库和框架中类型提示的采用率达91%但使用不一致的现象，并深入考察了注解的覆盖率、来源、历史演变以及开发者注解与Pyright推断类型之间的关系。 |
| [^4] | [The Import Tax: A Longitudinal Measurement of Startup Cost in the Python Ecosystem](https://arxiv.org/abs/2609.02753) | 该论文首次对Python生态系统开展了大规模纵向测量（500个热门PyPI包、五年数据、63,431次测量），系统量化了导入成本的高度偏斜性（中位数不足6毫秒 vs 第99百分位354毫秒），揭示了首次导入和子模块导入被基准测试严重低估的真实开销，并直接评估了Python 3.15全局延迟导入模式的效果。 |
| [^5] | [Automated Vulnerability Injection in Smart Contracts Using Large Language Models](https://arxiv.org/abs/2609.02624) | 该论文提出利用大语言模型自动向Solidity智能合约注入漏洞以构建评测数据集，通过多步骤验证流水线从近千个候选变体中筛选出32个涵盖25种漏洞类型的确认漏洞合约，并揭示了该方法在复杂结构和非局部漏洞类型上的局限性。 |
| [^6] | [AgOSS: A Dataset and Multi-Layer Characterization of Open-Source Agricultural Software](https://arxiv.org/abs/2609.02591) | 该论文构建了涵盖六种架构类别的66个农业开源软件仓库数据集AgOSS，并通过多层级供应链安全分析与非农业软件对照组对比，首次对农业开源软件生态系统的供应链安全状况进行了实证表征。 |
| [^7] | [PaperCompiler: Faithful Paper-to-Code Generation via Repository-Level Specification Compilation](https://arxiv.org/abs/2609.02272) | 论文提出PaperCompiler框架，将基于论文的证据编译为显式的仓库级实现规格，避免了现有论文到代码智能体中间输出被下游编码智能体忽略或曲解的问题，从而实现更忠实的论文到代码生成。 |
| [^8] | [From Prompting to Engineering: A Research Agenda for Prompt Engineering in Software Engineering](https://arxiv.org/abs/2609.02248) | 本文基于PROMPT-SE研讨会上组织的结构化社区讨论，提出了软件工程中提示工程的研究议程，将讨论成果归纳为提示工件标准化、评估与基准测试等五个关键领域，以推动非正式的提示实践向系统化的工程方法演进。 |
| [^9] | [ToolGate: An Executable Acceptance Pipeline for Tool-Dependent Scientific Benchmark Construction](https://arxiv.org/abs/2609.02067) | ToolGate提出了一条三关卡的可执行验收流水线，通过要求解题脚本复现答案并筛查无软件即可解答的题目，自动筛选出必须借助专业软件才能回答的高质量科学基准题目，从而替代昂贵的人工逐题审核。 |
| [^10] | [ExecRetrieval: Measuring the Functional-Correctness Gap in Code-Embedding Retrieval](https://arxiv.org/abs/2609.01865) | 提出 ExecRetrieval 基准（939 个 Python 任务），通过在搜索池中植入与规范实现几乎相同、但经执行验证的有缺陷变体，首次衡量了代码嵌入检索在区分功能正确代码与错误代码上的差距。 |
| [^11] | [Architecting Conversational Data Systems for Stateless LLM APIs: The Hydration Proxy Pattern](https://arxiv.org/abs/2609.01834) | 本文提出了水合代理模式，通过将会话持久化与推理引擎解耦的架构，解决无状态LLM API带来的对话状态管理负担，在确保平台对对话数据主权的同时实现安全的多阶段语义接地。 |
| [^12] | [Modelstamp: Pre-Deserialization Verification of Machine-Learning Artifacts and Runtime Environment State](https://arxiv.org/abs/2609.01781) | Modelstamp 是一个轻量级 Python 持久化库，它在模型反序列化之前通过 SHA-256 摘要、运行时元数据和包版本清单（可选 HMAC 认证）来验证机器学习工件的完整性与运行时环境状态，从而发现仅靠工件完整性检查无法察觉的环境漂移。 |
| [^13] | [From Silicon to Boot Code: Extending Automated Program Repair to Firmware-Layer Security Workarounds](https://arxiv.org/abs/2609.01769) | 该研究首次将自动程序修复从RTL设计阶段扩展到固件层，通过自动挖掘UEFI（EDK II）固件仓库提交历史中的修复模板，实现了流片后硬件漏洞（如Spectre v1）安全补丁的自动合成。 |
| [^14] | [Towards Behavior Tree-Guided Vulnerability Detection with Lightweight LLMs](https://arxiv.org/abs/2609.01758) | 本文提出将Java源代码解析为AST后再转换为行为树（BT）作为更紧凑的中间表示，从而在token数量受限的情况下提升轻量级大语言模型的软件漏洞检测性能。 |
| [^15] | [Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives](https://arxiv.org/abs/2609.01736) | 提出以自然语言取代API模式作为工具调用接口的“工具原语”设计，并构建包含25,519个函数的集中式仓库ToolFace供LLM在推理时动态检索工具，从而解决多步多轮推理脆弱及大规模工具目录下性能退化的问题。 |
| [^16] | [RosettaBitcoin: An Artifact-Backed Experience Report on Verification Infrastructure for Agent-Assisted Consensus Validators](https://arxiv.org/abs/2609.01702) | 本文通过对不可变软件快照的基于工件的可审计性分析，报告了RosettaBitcoin项目十二个独立实现的Bitcoin testnet4共识验证器的验证基础设施现状：所有移植版本均通过45/45脚本语料库证明和5,000区块基线，但均缺乏从空状态到链尖的完整证明且未满足二进制全节点门槛。 |
| [^17] | [Barriers to Using Static Application Security Testing (SAST) Tools: A Literature Review](https://arxiv.org/abs/2609.01669) | 本文通过文献综述揭示了开发人员不愿采用静态应用程序安全测试（SAST）工具的原因及其使用中面临的可用性障碍，指出部分问题需要开发人员投入解决，而另一些则需要SAST工具开发者加以改进。 |
| [^18] | [Investigating Software Aging in LLM-Generated Software Systems across Generation-and-Execution Environments](https://arxiv.org/abs/2608.26391) | 本研究首次通过实验揭示了LLM生成的软件系统在持续运行中表现出软件老化症状，且不同编程语言（JavaScript、Python、Rust）间的老化程度存在显著差异，为评估LLM生成代码的长期可靠性提供了实证依据。 |
| [^19] | [SPECMINE: A Large-Scale Corpus of Spec-Driven Development Artifacts](https://arxiv.org/abs/2608.25202) | 我们提出了SPECMINE，这是首个大规模语料库，通过两次普查系统地捕捉了GitHub上规范驱动开发工件，为研究规范如何转化为代码提供了基础数据。 |
| [^20] | [FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth](https://arxiv.org/abs/2608.20574) | 该论文提出了一个基于可执行烹饪真实数据的自动化基准测试FlavourBench，通过版本化系统和严格统计方法对27个前沿语言模型进行公平排名，消除了传统基准中的评判者偏差和缺失数据问题。 |
| [^21] | [Agentic Configuration Management (ACM): A Reference Configuration Model for Governed Agentic Systems](https://arxiv.org/abs/2608.11166) | 本文提出智能体配置管理（ACM），一个与框架无关的参考配置模型，通过类型化配置项、不可变版本基线、配置与运行时分离以及依赖感知的影响传播等机制，实现对异构智能体系统配置的统一治理。 |
| [^22] | [The Web4 Agent Economy: A Large-Scale Empirical Study of the Landscape, Challenges, and Opportunities](https://arxiv.org/abs/2606.25876) | 本文开展了首个针对Web4智能体生态系统的大规模实证研究，系统考察了智能体的实际部署使用方式、开发者面临的工程挑战以及项目社区的应对情况。 |
| [^23] | [Can Coding Agents Reproduce Findings in Computational Materials Science?](https://arxiv.org/abs/2605.00803) | 本文提出 AutoMat 基准，用于评估大语言模型编码智能体复现计算材料科学论文中科学论断的能力，涵盖恢复欠规范计算流程、驾驭专用工具链和验证证据是否支持论断三大挑战。 |
| [^24] | [MUCOCO: Automated Consistency Testing of Code LLMs](https://arxiv.org/abs/2604.19086) | 本文提出MUCOCO，一种利用保语义变异分析自动将程序转换为语义等价变异体、从而自动发现代码大语言模型不一致程序行为的自动化一致性测试方法。 |
| [^25] | [VulWeaver: Weaving Broken Semantics for Grounded Vulnerability Detection](https://arxiv.org/abs/2604.10767) | VulWeaver 是一种基于大语言模型的漏洞检测方法，其核心创新在于通过融合确定性规则与 LLM 语义推理构建增强的统一依赖图、结合显式与隐式上下文提取全面的漏洞信息，并借助漏洞类型专家指南的元提示引导 LLM 进行有依据的漏洞检测。 |
| [^26] | [A Longitudinal Study of Dependency Reclassifications in JavaScript Projects](https://arxiv.org/abs/2604.08747) | 本研究通过对33,087个JavaScript项目package.json文件的提交级分析，首次系统揭示了依赖项重新分类（包括移除和角色重分配）是一种普遍的维护活动，存在于79.1%的项目中，占所有依赖项维护提交的19.4%。 |
| [^27] | [PoC-Gym: Towards More Reliable LLM-Assisted Proof-of-Concept Exploit Generation](https://arxiv.org/abs/2602.04165) | 提出了PoC-Gym流水线，通过结合静态与动态信息（如CVE定制提示、静态追踪和覆盖率反馈）及多阶段验证机制，实现更可靠的基于LLM的Java安全漏洞PoC自动生成。 |

# 详细

[^1]: 面向编程竞赛金牌表现的语言模型后训练

    Post-Training Language Models for Gold-Medal Performance in Coding Competitions

    [https://arxiv.org/abs/2609.02849](https://arxiv.org/abs/2609.02849)

    该研究通过结合大规模题目筛选、监督微调、强化学习以及反馈驱动的测试时计算策略 GenCorrect，使语言模型在 IOI 2025 编程竞赛中取得了超越金牌分数线（438.3 分）的成绩（Nano-CC 达 468 分，Ultra-CC 达 502 分）。

    

    竞赛编程已成为检验大语言模型推理能力的关键测试，其中 IOI 和 ICPC 等国际赛事代表了最具挑战性的场景。我们提出了一条端到端的专门化流水线，结合了大规模题目筛选、合成推理轨迹、监督微调（SFT）和强化学习（RL）。利用 22,000 道精选题目，我们通过 SFT 和 RL 训练了 Nemotron-3-Nano-CC（30B-A3B），并仅通过 SFT 训练了 Nemotron-3-Ultra-CC（550B-A55B）。我们进一步提出了 GenCorrect，这是一种由反馈驱动的测试时计算策略，可迭代地生成、评估并改进多样化的解决方案。在 IOI 2025 上，Nano-CC 在后训练后从 130 分提升至 291 分，结合 GenCorrect 后达到 468 分，超过了 438.3 的金牌分数线，而 Ultra-CC 达到了 502 分。在这些结果的指导下，我们开发了一个面向竞赛的 Ultra-CC 系统，并在 IOI 2026 期间进行了前瞻性评估。

    arXiv:2609.02849v1 Announce Type: cross  Abstract: Competitive programming has become a key test of large language model reasoning, with international competitions such as IOI and ICPC representing its most challenging settings. We present an end-to-end specialization pipeline combining large-scale problem curation, synthetic reasoning traces, supervised fine-tuning (SFT), and reinforcement learning (RL). Using 22,000 curated problems, we train Nemotron-3-Nano-CC (30B-A3B) with SFT and RL and Nemotron-3-Ultra-CC (550B-A55B) with SFT alone. We further introduce GenCorrect, a feedback-driven test-time compute strategy that iteratively generates, evaluates, and refines diverse solutions. On IOI 2025, Nano-CC improves from 130 points to 291 after post-training and to 468 with GenCorrect, exceeding the gold threshold of 438.3 while Ultra-CC reaches 502. Guided by these results, we develop a competition-specific Ultra-CC system and evaluate it prospectively during IOI 2026. Under the same ti
    
[^2]: ShikumiMiner：挖掘AI代码库中的重复实现模式

    ShikumiMiner: Mining Recurring Implementation Patterns in AI Codebases

    [https://arxiv.org/abs/2609.02789](https://arxiv.org/abs/2609.02789)

    本文提出ShikumiMiner静态分析框架，结合抽象语法树与控制流图特征，利用多标签随机森林模型检测、比较并分类C++本地大语言模型代码库中的重复实现模式，为LLM应用开发者提供设计洞察。

    

    大语言模型通过理解、分析、总结和生成现代世界中的内容，正在为创新铺平道路。目前，工程师们已在开源仓库中开发了数千个大语言模型项目。然而，这些大语言模型项目是否存在潜在的实现模式仍是一个疑问。探索这些潜在模式将为旨在开发大语言模型项目的开发者提供新的视角。在本文中，我们提出了ShikumiMiner，这是一个结合抽象语法树（AST）和控制流图（CFG）特征的静态分析框架，用于检测和比较C++本地大语言模型代码库中的重复实现模式。我们分析了十个GitHub开源仓库，并使用多标签随机森林模型将函数分类为七个研究特定类别。研究这些模式可以为旨在设计大语言模型应用的开发者提供有价值的见解。

    arXiv:2609.02789v1 Announce Type: new  Abstract: Large language models are paving the way towards innovation by understanding, analyzing, summarizing and generating content in the modern world. Currently there are thousands of LLM projects developed by engineers in open-source repositories. However, whether these LLM projects have underlying patterns or not remains a question. Exploring these underlying patterns will give new dimensions to the developers who aim to develop these LLM projects. In this paper, we propose ShikumiMiner, a static-analysis framework that combines Abstract Syntax Tree (AST) and Control Flow Graph (CFG) features to detect and compare recurring implementation patterns in C++ local LLM codebases. We analyze ten GitHub open-source repositories and classify functions into seven study-specific categories using a multi-label Random Forest model. Studying these patterns can provide useful insights for developers aiming to design LLM applications.
    
[^3]: Python库与框架中的类型提示：采用与维护的实证分析

    Type Hints in Python Libraries and Frameworks: An Empirical Analysis of Adoption and Maintenance

    [https://arxiv.org/abs/2609.02782](https://arxiv.org/abs/2609.02782)

    该论文通过对1,000个热门GitHub仓库的大规模实证分析，首次系统揭示了Python库和框架中类型提示的采用率达91%但使用不一致的现象，并深入考察了注解的覆盖率、来源、历史演变以及开发者注解与Pyright推断类型之间的关系。

    

    背景：在Python中，类型提示允许开发者为变量和函数添加明确的类型信息标注，从而提升代码的清晰度和可靠性。尽管类型提示已被广泛支持，但人们对其在库和框架中的采用与维护情况知之甚少。目的：我们研究Python库和框架中类型提示的采用、使用、维护情况及其背后的原因。方法：我们分析了1,000个热门GitHub仓库，从中识别出库和框架并提取其类型注解。我们考察了注解的覆盖率、注解的位置和来源、注解在git历史中的演变，以及开发者注解与Pyright静态推断类型之间的关系。结果：在所分析的仓库中，91%的库至少使用过一次类型提示，但采用情况并不一致。在系统性使用类型提示的库中，维护者优先为函数参数和返回类型添加注解。

    arXiv:2609.02782v1 Announce Type: new  Abstract: Context: In Python, type hints allow developers to annotate variables and functions with explicit type information, improving code clarity and reliability. Although type hints are widely available, little is known about how they are adopted and maintained in libraries and frameworks. Objective: We investigate the adoption, usage, maintenance, and rationale of type hints in Python libraries and frameworks. Method: We analyzed 1,000 popular GitHub repositories, identifying libraries and frameworks and extracting their type annotations. We examined annotation coverage, the locations and origins of annotations, their evolution across git histories, and the relationship between developer annotations and types inferred by Pyright. Results: Of the analyzed repositories, 91% of libraries use type hints at least once, although adoption is inconsistent. Among libraries with systematic usage, maintainers prioritize function parameters and return ty
    
[^4]: 导入税：Python生态系统中启动成本的纵向测量

    The Import Tax: A Longitudinal Measurement of Startup Cost in the Python Ecosystem

    [https://arxiv.org/abs/2609.02753](https://arxiv.org/abs/2609.02753)

    该论文首次对Python生态系统开展了大规模纵向测量（500个热门PyPI包、五年数据、63,431次测量），系统量化了导入成本的高度偏斜性（中位数不足6毫秒 vs 第99百分位354毫秒），揭示了首次导入和子模块导入被基准测试严重低估的真实开销，并直接评估了Python 3.15全局延迟导入模式的效果。

    

    Python程序在每次进程启动时都要为导入付出代价，这一成本在稳态基准测试中不可见，但对于命令行工具、测试工作进程和无服务器冷启动而言却是主导性开销。Python 3.15主要基于轶事证据引入了显式延迟导入（PEP 810）；此前并不存在对生态系统导入成本的系统性测量。我们提出了这样一项测量：针对下载量最多的500个PyPI软件包，在五年的发布历史中按季度采样，在六个CPython版本（3.9-3.14）和两个平台（Apple M5/macOS 与 Intel Xeon/Linux）上进行测量，共计63,431次测量，并额外对3.15的全局延迟导入模式进行了直接测量。导入成本呈高度偏斜分布：一半的软件包导入时间不足6毫秒，但第99百分位数达到354毫秒；安装后的首次导入成本高出3至22倍（字节码编译所致）；而导入软件包子模块的成本比基准测试通常报告的顶层导入高出多达294倍。（原文摘要至此处截断）

    arXiv:2609.02753v1 Announce Type: new  Abstract: Python programs pay for their imports at every process start, a cost that is invisible in steady-state benchmarks but dominant for command-line tools, test workers, and serverless cold starts. Python 3.15 adds explicit lazy imports (PEP 810) largely on anecdotal evidence; no systematic measurement of the ecosystem's import cost exists. We present one: the 500 most-downloaded PyPI packages, sampled quarterly over five years of releases, measured under six CPython versions (3.9-3.14) on two platforms (Apple M5/macOS and Intel Xeon/Linux), for 63,431 measurements in total, plus direct measurement of 3.15's global lazy-import mode. Import cost is heavily skewed: half of packages import in under 6 ms, but the 99th percentile is 354 ms, the first import after installation costs 3-22x more (bytecode compilation), and importing a package's submodules costs up to 294x more than the top-level import that benchmarks report. The median package's cos
    
[^5]: 使用大语言模型的智能合约自动化漏洞注入

    Automated Vulnerability Injection in Smart Contracts Using Large Language Models

    [https://arxiv.org/abs/2609.02624](https://arxiv.org/abs/2609.02624)

    该论文提出利用大语言模型自动向Solidity智能合约注入漏洞以构建评测数据集，通过多步骤验证流水线从近千个候选变体中筛选出32个涵盖25种漏洞类型的确认漏洞合约，并揭示了该方法在复杂结构和非局部漏洞类型上的局限性。

    

    评估智能合约漏洞检测工具需要具有已知真实标注的数据集，然而这类数据集十分稀缺且难以手工构建。我们提出一种利用大语言模型自动向Solidity智能合约注入漏洞的方法，并通过针对OpenSCV中49种漏洞类型的案例研究加以验证。注入后的合约经过多步骤流水线验证，涵盖编译、执行、业务逻辑以及预期漏洞存在性的检查。将该 方法应用于来自SmartBugs的真实合约，大语言模型生成了近1000个候选变体；经过去重与验证后，最终保留了32个确认存在漏洞的合约，涵盖25种漏洞类型（存活率为16.58%）。存活的合约集中在结构较简单的目标以及具有局部语法模式的漏洞类型上。我们还报告了实际面临的挑战，包括大语言模型输出的不确定性等问题。

    arXiv:2609.02624v1 Announce Type: cross  Abstract: Assessing vulnerability detection tools for smart contracts requires datasets with known ground truth, yet such datasets are scarce and difficult to build by hand. We propose an approach that uses Large Language Models (LLMs) to automatically inject vulnerabilities into Solidity smart contracts, and demonstrate it in a case study targeting 49 vulnerability types from OpenSCV. Injected contracts are validated through a multi-step pipeline checking compilation, execution, business logic, and the presence of the intended vulnerability. Applied to real-world contracts from SmartBugs, LLMs generate nearly 1,000 candidate variants; after deduplication and validation, 32 confirmed vulnerable contracts spanning 25 vulnerability types survive (a 16.58% survival rate). Surviving contracts concentrate in structurally simpler targets and vulnerability types with localized syntactic patterns. We report practical challenges including LLMs' non-deter
    
[^6]: AgOSS：开源农业软件的数据集与多层次安全特征分析

    AgOSS: A Dataset and Multi-Layer Characterization of Open-Source Agricultural Software

    [https://arxiv.org/abs/2609.02591](https://arxiv.org/abs/2609.02591)

    该论文构建了涵盖六种架构类别的66个农业开源软件仓库数据集AgOSS，并通过多层级供应链安全分析与非农业软件对照组对比，首次对农业开源软件生态系统的供应链安全状况进行了实证表征。

    

    农业的许多方面都依赖于开源软件，这些软件涵盖农场管理平台、云服务、边缘网关、嵌入式系统以及部署在田间的传感器，形成了一个特定领域的软件供应链，但该供应链几乎未受到实证安全研究的关注。目前尚不清楚这一生态系统的供应链安全状况是否与同类非农业软件存在差异；如果存在差异，也不清楚这种差异究竟是源于农业领域本身，还是源于其中项目的规模和成熟度。作为保障农业开源软件安全的一步，我们提出了AgOSS，这是一个涵盖六种架构类别、共66个代码仓库的数据集。我们通过OpenSSF Scorecard、治理指标、基于SBOM的依赖分析以及KEV匹配来评估该数据集内的供应链安全状况，并与匹配的非农业对照组进行比较。我们报告了两项发现：第一，治理在很大程度上与所继承的依赖关系相互独立……（原文摘要在此处截断）

    arXiv:2609.02591v1 Announce Type: new  Abstract: Much of agriculture depends on open-source software spanning farm management platforms, cloud services, edge gateways, embedded systems, and field-deployed sensors, forming a domain-specific software supply chain that has drawn little empirical security attention. It is unknown whether this ecosystem's supply chain security posture differs from that of comparable non-agricultural software, and if it does, whether the difference reflects the agricultural domain or the size and maturity of the projects within it.   As a step towards securing agricultural open-source software, we present AgOSS, a dataset of 66 repositories across six architectural categories. We assess supply chain security within the dataset via OpenSSF Scorecard, governance metrics, SBOM-based dependency analysis, and KEV matching, and compare against matched non-agricultural controls. We report two findings. First, governance is largely independent of inherited dependenc
    
[^7]: PaperCompiler：通过仓库级规格编译实现忠实的论文到代码生成

    PaperCompiler: Faithful Paper-to-Code Generation via Repository-Level Specification Compilation

    [https://arxiv.org/abs/2609.02272](https://arxiv.org/abs/2609.02272)

    论文提出PaperCompiler框架，将基于论文的证据编译为显式的仓库级实现规格，避免了现有论文到代码智能体中间输出被下游编码智能体忽略或曲解的问题，从而实现更忠实的论文到代码生成。

    

    将研究论文忠实地转化为仓库级实现仍然具有挑战性，因为论文通常在高层次上描述方法，将实现假设隐含其中，并要求生成的代码仓库保持方法逻辑、评估协议和跨文件一致性。尽管论文到代码智能体最近取得了进展，但它们的中间输出通常以自由形式的计划或摘要呈现，下游编码智能体可能会忽略、重新解释或压缩这些内容，导致算法简化和仓库结构不一致。为了应对这些挑战，我们提出了PaperCompiler，一个将基于论文的证据编译为显式仓库级实现规格的论文到代码生成框架。PaperCompiler在获取实现相关证据的同时，保留来源出处，并区分论文支持的、推断的、外部委托的以及未解决的信息。（摘要原文截断）

    arXiv:2609.02272v1 Announce Type: cross  Abstract: Faithfully translating research papers into repository-level implementations remains challenging because papers often describe methods at a high level, leave implementation assumptions implicit, and require generated repositories to preserve method logic, evaluation protocols, and cross-file consistency. Despite recent advances in paper-to-code agents, their intermediate outputs are often presented as free-form plans or summaries that downstream coding agents may ignore, reinterpret, or compress, leading to algorithmic simplification and inconsistent repository structure. To address these challenges, we introduce PaperCompiler, a paper-to-code generation framework that compiles paper-grounded evidence into explicit repository-level implementation specifications. PaperCompiler grounds implementation-relevant evidence while preserving source provenance and distinguishing paper-supported, inferred, externally delegated, and unresolved inf
    
[^8]: 从提示到工程化：软件工程中提示工程的研究议程

    From Prompting to Engineering: A Research Agenda for Prompt Engineering in Software Engineering

    [https://arxiv.org/abs/2609.02248](https://arxiv.org/abs/2609.02248)

    本文基于PROMPT-SE研讨会上组织的结构化社区讨论，提出了软件工程中提示工程的研究议程，将讨论成果归纳为提示工件标准化、评估与基准测试等五个关键领域，以推动非正式的提示实践向系统化的工程方法演进。

    

    提示工程正日益广泛应用于软件工程（SE）的各项活动中，包括需求分析、编码、测试、文档编写、代码仓库分析和规划。然而，提示词及相关指令工件往往是通过针对特定任务的非正式实践来创建和演进的，在系统性评估、管理、可追溯性和治理方面支持有限。为了探讨软件工程如何推动这些实践的成熟，我们在与EASE 2026联合举办的首届软件工程实证提示工程国际研讨会（PROMPT-SE）上组织了一场结构化的社区讨论。与会者讨论了当前的提示实践、其采用与评估所面临的挑战，以及将提示工程融入软件开发的未来方向。我们将这些讨论总结为五个领域：提示工件与标准化；评估与基准测试；……（摘要原文在此处截断）

    arXiv:2609.02248v1 Announce Type: new  Abstract: Prompt engineering is increasingly used across Software Engineering (SE) activities, including requirements analysis, coding, testing, documentation, repository analysis, and planning. Yet prompts and related instruction artifacts are often created and evolved through task-specific and informal practices, with limited support for their systematic evaluation, management, traceability, and governance. To examine how SE can contribute to the maturation of these practices, we organized a structured community discussion at the First International Workshop on Empirical Prompt Engineering for Software Engineering (PROMPT-SE), co-located with EASE 2026. Participants discussed current prompting practices, challenges to their adoption and evaluation, and future directions for integrating prompt engineering into software development. We synthesized these discussions into five areas: prompt artifacts and standardization; evaluation and benchmarking;
    
[^9]: ToolGate：面向工具依赖型科学基准构建的可执行验收流水线

    ToolGate: An Executable Acceptance Pipeline for Tool-Dependent Scientific Benchmark Construction

    [https://arxiv.org/abs/2609.02067](https://arxiv.org/abs/2609.02067)

    ToolGate提出了一条三关卡的可执行验收流水线，通过要求解题脚本复现答案并筛查无软件即可解答的题目，自动筛选出必须借助专业软件才能回答的高质量科学基准题目，从而替代昂贵的人工逐题审核。

    

    arXiv:2609.02067v1 公告类型：新论文 摘要：科学基准通常由领域专家编写任务并交叉审核彼此的工作来构建，或者改编自教科书、已发表论文和在线资源中的现有材料。这些途径可以产生高质量的评估，但每个题目都需要投入大量人力。语言模型可以通过快速提出候选题目来减少这类重复性工作，剩下的问题在于如何验收。我们关注的是那些答案需要借助专业软件进行计算、而非仅凭推理就能得到的科学问题。如果候选题目的脚本运行失败或返回不同的答案，则该题目无效；如果模型无需软件就能回答该题目，则该题目过于简单。我们提出了ToolGate，它将每个生成的题目视为一个提案，只有通过三道关卡才会被保留。第一，可执行的解题脚本必须在使用科学软件运行时复现所提出的答案。第二，随机化的无工具筛查会拒绝那些模型无需软件即可解答的候选题目。

    arXiv:2609.02067v1 Announce Type: new  Abstract: Scientific benchmarks are commonly built by domain experts who write tasks and cross-check one another's work, or who adapt existing material from textbooks, published papers, and online resources. These routes can produce strong evaluations, but they require substantial per-item labor. Language models can reduce this repeated work by proposing candidates quickly. The remaining problem is acceptance. We target scientific questions whose answers require computations with specialist software rather than unaided reasoning alone. A candidate is invalid if its script fails or returns a different answer, or trivial if a model answers it without the software. We present ToolGate, which treats every generated item as a proposal and keeps it only if three gates pass. First, an executable solution script must reproduce the proposed answer when run with the scientific software. Second, randomized no-tool screening rejects candidates that models can
    
[^10]: ExecRetrieval：衡量代码嵌入检索中的功能正确性差距

    ExecRetrieval: Measuring the Functional-Correctness Gap in Code-Embedding Retrieval

    [https://arxiv.org/abs/2609.01865](https://arxiv.org/abs/2609.01865)

    提出 ExecRetrieval 基准（939 个 Python 任务），通过在搜索池中植入与规范实现几乎相同、但经执行验证的有缺陷变体，首次衡量了代码嵌入检索在区分功能正确代码与错误代码上的差距。

    

    基于嵌入的代码检索是编码智能体和检索增强代码生成的核心组件，在这些场景中，检索到功能正确的代码比检索到词汇上相似的代码更为重要。现有的代码检索基准并未在搜索池中植入受控的、经执行验证的、针对每个查询规范实现的单次编辑变体，因此“嵌入模型能否在检索场景中从功能上区分正确代码与近似克隆但不正确的代码”这一问题仍未得到解答。解决这一问题需要一个搜索池本身就包含相关反事实样本的基准——即与每个规范实现几乎完全相同、且经过执行验证的有缺陷变体——从而可以直接检验检索器的排序结果是否具备功能区分能力，而不仅仅是主题或身份上的重合。我们提出了 ExecRetrieval，包含 939 个 Python 任务，每个任务都配有一个经执行验证的规范实现，以及最多四个经执行验证的……

    arXiv:2609.01865v1 Announce Type: cross  Abstract: Embedding-based code retrieval is a core component of coding agents and retrieval-augmented code generation, where retrieving correct code matters more than retrieving lexically similar code. Existing code-retrieval benchmarks do not plant controlled, execution-verified single-edit variants of each query's canonical implementation in the search pool, leaving the question of whether embeddings can functionally discriminate correct from near-clone-but-incorrect code unanswered in a retrieval setting. Resolving this requires a benchmark whose search pool itself contains the relevant counterfactuals -- execution-verified buggy variants near-identical to each canonical -- so that a retriever's rank ordering can be directly tested for functional discrimination rather than topical or identity overlap. We introduce ExecRetrieval, 939 Python tasks each paired with one execution-verified canonical implementation and up to four execution-verified
    
[^11]: 为无状态大语言模型API构建对话式数据系统：水合代理模式

    Architecting Conversational Data Systems for Stateless LLM APIs: The Hydration Proxy Pattern

    [https://arxiv.org/abs/2609.01834](https://arxiv.org/abs/2609.01834)

    本文提出了水合代理模式，通过将会话持久化与推理引擎解耦的架构，解决无状态LLM API带来的对话状态管理负担，在确保平台对对话数据主权的同时实现安全的多阶段语义接地。

    

    随着企业平台向对话式推理界面转型，大语言模型API的无状态特性造成了架构上的鸿沟。虽然无状态性使AI提供商能够实现水平扩展，但它迫使客户端应用程序承担管理对话状态和语义记忆的全部负担。本工作提出了水合代理模式，这是一种将会话持久化与推理引擎解耦的架构。该框架在确保平台对对话数据主权的同时，支持安全的多阶段语义接地。我们进一步提出了上下文稳定化规范，以解决主权状态管理与KV缓存之间的权衡问题。

    arXiv:2609.01834v1 Announce Type: new  Abstract: As enterprise platforms transition to conversational reasoning interfaces, the stateless nature of LLM APIs creates an architectural gap. While statelessness enables horizontal scalability for AI providers, it forces client applications to manage the entire burden of conversational state and semantic memory. The work identifies the Hydration Proxy Pattern, an architecture that decouples session persistence from the reasoning engine. The framework ensures platform sovereignty over conversational data while enabling secure, multi-stage semantic grounding. We further propose the Context Stabilization Mandate to resolve the tradeoff between sovereign state management and KV caching.
    
[^12]: Modelstamp：反序列化前的机器学习工件与运行时环境状态验证

    Modelstamp: Pre-Deserialization Verification of Machine-Learning Artifacts and Runtime Environment State

    [https://arxiv.org/abs/2609.01781](https://arxiv.org/abs/2609.01781)

    Modelstamp 是一个轻量级 Python 持久化库，它在模型反序列化之前通过 SHA-256 摘要、运行时元数据和包版本清单（可选 HMAC 认证）来验证机器学习工件的完整性与运行时环境状态，从而发现仅靠工件完整性检查无法察觉的环境漂移。

    

    持久化的机器学习模型在字节层面可能保持完全一致，但其加载所在的软件环境却会不断演变，这产生了一个仅靠工件完整性检查无法暴露的验证问题。本文提出了 Modelstamp，一个轻量级的 Python 持久化库，用于在反序列化之前验证工件完整性及其所代表的运行时环境状态。在持久化时，Modelstamp 将序列化的工件与一个伴随的 JSON 清单文件相关联，该清单包含 SHA-256 摘要、运行时元数据以及来自一个有界的受跟踪包集合的已安装版本；一个单独记录的模型相关子集决定了哪些包版本参与漂移比较。可选的 HMAC 身份验证支持生产者与验证者共享密钥的工作流程。在验证时，工件及其所代表的当前环境会依据这些记录的证据进行检查，然后模型才会被反序列化。

    arXiv:2609.01781v1 Announce Type: new  Abstract: Persisted machine-learning models can remain byte-identical while the software environments in which they are loaded evolve, creating a verification problem that artifact integrity checks alone cannot expose. This paper presents Modelstamp, a lightweight Python persistence library for verifying artifact integrity and represented runtime-environment state before deserialization. At persistence time, Modelstamp associates a serialized artifact with a sidecar JSON manifest containing a SHA-256 digest, runtime metadata, and installed versions from a bounded tracked-package set; a separately recorded model-relevant subset determines which package versions participate in drift comparison. Optional HMAC authentication supports workflows in which the producer and verifier share a secret key. At verification time, the artifact and represented current environment are checked against this recorded evidence before the model is deserialized. Modelsta
    
[^13]: 从硅片到引导代码：将自动程序修复扩展至固件层安全规避方案

    From Silicon to Boot Code: Extending Automated Program Repair to Firmware-Layer Security Workarounds

    [https://arxiv.org/abs/2609.01769](https://arxiv.org/abs/2609.01769)

    该研究首次将自动程序修复从RTL设计阶段扩展到固件层，通过自动挖掘UEFI（EDK II）固件仓库提交历史中的修复模板，实现了流片后硬件漏洞（如Spectre v1）安全补丁的自动合成。

    

    自动程序修复（APR）研究一直局限于设计阶段：现有技术只能在芯片投产之前定位并修复RTL或HLS设计中的缺陷。一旦硬件漏洞在流片后显现，补丁就必须依靠人工生成——现有自动化方案仅解决补丁部署问题，而非补丁合成问题。我们研究了将最初为RTL修复开发的字典引导、“定位-合成-验证”式APR方法论扩展到固件层的可行性。一个自动化的提交聚类挖掘器无需依赖已知的CVE标识符，即可从EDK II（UEFI）固件仓库的完整提交历史中发现反复出现的修复模板，成功找回了全部三个已知的CVE修复活动，并额外发现了两个候选缺陷家族。基于真实的修复证据，我们构建了四个独立的定位器：C代码中缺失的推测执行屏障（CVE-2017-5753，Spectre v1）、数组写入前缺失的边界检查……（摘要原文在此处截断）

    arXiv:2609.01769v1 Announce Type: new  Abstract: Automated program repair (APR) research has been constrained to design time. Current techniques localize and fix bugs in RTL or HLS designs before a chip reaches production. Once a hardware vulnerability surfaces post-silicon, the patch must be manually generated: existing automation addresses patch deployment but not patch synthesis. We study the feasibility of extending a dictionary-guided, localize-synthesize-validate APR methodology originally developed for RTL repair to this firmware layer. An automated commit-clustering miner surfaces recurring fix templates across the EDK II (UEFI) firmware repository's full commit history without depending on known CVE identifiers, recovering all three known CVE-fix campaigns and surfacing two additional candidate bug families. Grounded in real fix evidence, we build four independent localizers: missing speculation barriers in C (CVE-2017-5753, Spectre v1), missing bounds checks before array writ
    
[^14]: 面向轻量级大语言模型的行为树引导漏洞检测

    Towards Behavior Tree-Guided Vulnerability Detection with Lightweight LLMs

    [https://arxiv.org/abs/2609.01758](https://arxiv.org/abs/2609.01758)

    本文提出将Java源代码解析为AST后再转换为行为树（BT）作为更紧凑的中间表示，从而在token数量受限的情况下提升轻量级大语言模型的软件漏洞检测性能。

    

    大型语言模型（LLMs）越来越多地被用于软件漏洞检测，但其性能取决于源代码在输入中的表示方式。大多数提示方法使用原始形式的源代码，而一些工作则提出使用结构化表示。抽象语法树（AST）是最流行的结构化表示方法之一，但AST的冗长性会增加相对于源代码的输入规模，使其难以适应某些LLM的上下文窗口限制。本文研究将行为树（BT）作为基于LLM的漏洞检测的替代中间表示。行为树比AST更紧凑地编码控制流、条件和可执行操作，使其在token数量受限时成为天然的选择。首先，我们提出了一个预处理阶段，将Java源代码解析为AST，然后将其转换为行为树表示。随后，我们对漏洞检测性能进行了比较……

    arXiv:2609.01758v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly used for software vulnerability detection, but their performance depends on how source code is represented in the input. Most prompting approaches use source code in its original form, while some works propose the use of structured representations. Abstract Syntax Trees (ASTs) are one of the most popular approaches, but AST verbosity increases input size relative to source code, making them hard to fit within some LLMs context windows. This paper investigates Behavior Trees (BTs) as an alternative intermediate representation for LLM-based vulnerability detection. BTs encode control flow, conditions, and executable actions more compactly than ASTs, making them a natural candidate when token count is a constraint. First, we propose a preprocessing stage that parses Java source code into ASTs and then converts them into BT representations. We then compare vulnerability detection performance ac
    
[^15]: 通过智能体原生可复用工具原语实现LLM工具使用中的Harness工程

    Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives

    [https://arxiv.org/abs/2609.01736](https://arxiv.org/abs/2609.01736)

    提出以自然语言取代API模式作为工具调用接口的“工具原语”设计，并构建包含25,519个函数的集中式仓库ToolFace供LLM在推理时动态检索工具，从而解决多步多轮推理脆弱及大规模工具目录下性能退化的问题。

    

    增强了外部工具的大型语言模型（LLM）在解决复杂现实任务方面已展现出卓越能力。然而，现有方法面临两个关键挑战：由工具输出类型和API模式不兼容导致的脆弱的多步与多轮推理，以及在大规模工具目录下的性能下降。为解决这些问题，我们提出了**工具原语**，这一设计以自然语言作为工具调用的接口，取代了僵化的基于API模式的调用方式，其中每个工具都被封装了一个LLM接口，在内部处理模式解析与执行，从而实现工具之间的自然通信，支持嵌套和多轮工具调用。基于工具原语，我们构建了**ToolFace**，一个包含25,519个函数的集中式仓库，LLM可以在推理时从中动态检索仅相关的工具，从而无需枚举原始API模式……（摘要原文在此处被截断）

    arXiv:2609.01736v1 Announce Type: cross  Abstract: Large language models (LLMs) augmented with external tools have demonstrated remarkable capability in solving complex real-world tasks. However, existing approaches suffer from two key challenges: brittle multi-step and multi-turn reasoning caused by incompatible tool output types and API schemas, and performance degradation under large tool catalogues. To address these, we introduce \textbf{Tool Primitives}, a design that replaces rigid API schema-based invocation with natural language as the interface for tool calling, where each tool is wrapped with an LLM interface that handles schema resolution and execution internally, enabling natural inter-tool communication for nested and multi-turn tool calling. Building on Tool Primitives, we host \textbf{ToolFace}, a centralized repository of 25,519 functions from which LLMs dynamically retrieve only the relevant tools at inference time, eliminating the need to enumerate raw API schemas in 
    
[^16]: RosettaBitcoin：关于代理辅助共识验证器验证基础设施的基于工件的经验报告

    RosettaBitcoin: An Artifact-Backed Experience Report on Verification Infrastructure for Agent-Assisted Consensus Validators

    [https://arxiv.org/abs/2609.01702](https://arxiv.org/abs/2609.01702)

    本文通过对不可变软件快照的基于工件的可审计性分析，报告了RosettaBitcoin项目十二个独立实现的Bitcoin testnet4共识验证器的验证基础设施现状：所有移植版本均通过45/45脚本语料库证明和5,000区块基线，但均缺乏从空状态到链尖的完整证明且未满足二进制全节点门槛。

    

    代理辅助的软件项目通常通过演示或汇总基准测试来报告，而这些方式掩盖了正确性声明是如何被采纳的。本经验报告研究了RosettaBitcoin——一个由单一开发者构建、包含十二个独立实现的Bitcoin testnet4共识验证器的项目——通过其2026年6月17日的不可变软件快照（DOI 10.5281/zenodo.20738249）进行分析。我们分析了该快照中被追踪的SQLite证据数据库、经过整理的工件索引、一致性测试夹具、验证脚本、阻塞记录和版本历史。在快照时点，全部十二个移植版本都拥有各自的45/45脚本语料库证明和严格的5,000区块基线。九个移植版本拥有规范的干净的50,000区块、100,000区块以及100,000区块之后的验证通道。Java拥有一个19.86秒的接近链尖的维护工件。没有任何移植版本拥有从空状态到链尖的证明，也没有任何移植版本满足该项目设定的二进制全节点门槛；Docker和实节点的能力差距依然存在。

    arXiv:2609.01702v1 Announce Type: new  Abstract: Agent-assisted software projects are often reported through demonstrations or aggregate benchmarks that conceal how correctness claims were admitted. This experience report studies RosettaBitcoin, a single-developer project that built twelve separately implemented Bitcoin testnet4 consensus validators, through its immutable 17 June 2026 software snapshot (DOI 10.5281/zenodo.20738249). We analyze the snapshot's tracked SQLite evidence database, curated artifact index, conformance fixtures, validation scripts, blocker records, and version history. At the snapshot, all twelve ports had port-owned 45/45 script-corpus proofs and strict 5,000-block baselines. Nine had canonical clean 50,000-block, 100,000-block, and post-100,000 validation lanes. Java had one 19.86-second near-tip maintenance artifact. No port had an empty-state-to-tip proof, and no port satisfied the project's binary full-node gate; Docker and live-node capability gaps remain
    
[^17]: 使用静态应用程序安全测试（SAST）工具的障碍：一项文献综述

    Barriers to Using Static Application Security Testing (SAST) Tools: A Literature Review

    [https://arxiv.org/abs/2609.01669](https://arxiv.org/abs/2609.01669)

    本文通过文献综述揭示了开发人员不愿采用静态应用程序安全测试（SAST）工具的原因及其使用中面临的可用性障碍，指出部分问题需要开发人员投入解决，而另一些则需要SAST工具开发者加以改进。

    

    开发人员面临着一个尚无明确解决方案的挑战性问题。现代软件漏洞入侵可能对企业和个人造成严重破坏。由于代码漏洞是主要成因，保障应用程序安全必须成为开发人员的首要任务。静态应用程序安全测试（SAST）有潜力通过协助识别和解决安全漏洞来加固应用程序。尽管如此，许多开发团队尚未在其环境中采用SAST工具。本文调研了近期文献，以揭示为什么一些开发人员对SAST持谨慎态度，并确定他们在使用SAST时遇到的具体问题。我们发现开发人员在使用SAST时面临各种可用性问题。其中一些是工具固有的问题，最终需要开发人员进行一定程度的投入，而另一些则是SAST工具开发者必须解决的工具缺陷。最后，我们认为，为了推动（SAST的广泛应用）……

    arXiv:2609.01669v1 Announce Type: new  Abstract: Developers face a challenging problem with no clear solution. Modern software breaches can wreak havoc on businesses and individuals alike. With code vulnerabilities being a leading cause, securing applications must be a priority for developers. Static Application Security Testing (SAST) has the potential to harden applications by assisting in the identification and resolution of security vulnerabilities. Despite this, many development teams have not adopted SAST tools into their environment. In this paper, we survey the recent literature to uncover why some developers are apprehensive towards SAST and identify what specific problems they encounter when using it. We found a variety of usability problems developers face when using SAST. Some are inherent of the tool and ultimately require some level of developer investment while others are tool shortcomings that SAST tool creators must address. Ultimately, we argue that in order to drive 
    
[^18]: 探究LLM生成的软件系统在不同生成与执行环境中的软件老化现象

    Investigating Software Aging in LLM-Generated Software Systems across Generation-and-Execution Environments

    [https://arxiv.org/abs/2608.26391](https://arxiv.org/abs/2608.26391)

    本研究首次通过实验揭示了LLM生成的软件系统在持续运行中表现出软件老化症状，且不同编程语言（JavaScript、Python、Rust）间的老化程度存在显著差异，为评估LLM生成代码的长期可靠性提供了实证依据。

    

    大型语言模型（LLM）越来越多地被用于从自然语言规范生成可执行的软件系统，从而加速开发并减少人工实现工作量。尽管近期研究已探讨了LLM生成代码的功能正确性、安全性、可维护性和鲁棒性，但关于此类系统在持续运行下的长期可靠性知之甚少。本文通过实验研究了LLM生成的服务型应用在不同编程语言中的软件老化症状。基于BaxBench衍生的后端场景，我们通过基于LLM的生成平台生成了面向JavaScript、Python和Rust的应用，使用BaxBench派生的测试进行验证，并对其施加48小时的工作负载执行。我们监测了内存使用、响应时间和吞吐量，并采用Mann-Kendall检验和Sen斜率估计进行分析。

    arXiv:2608.26391v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used to generate executable software systems from natural language specifications, accelerating development and reducing manual implementation effort. Although recent studies have investigated the functional correctness, security, maintainability, and robustness of LLM-generated code, little is known about the long-term reliability of such systems under sustained execution. In this paper, we experimentally investigate software aging symptoms in LLM-generated service-based applications across different programming languages. Using backend scenarios derived from BaxBench, we generated applications targeting JavaScript, Python, and Rust through LLM-based generation platforms, validated them with BaxBench-derived tests, and subjected them to 48-hour workload executions. We monitored memory usage, response time, and throughput and analyzed them using the Mann--Kendall test and Sen's slope estimato
    
[^19]: SPECMINE：一个大规模的规范驱动开发工件语料库

    SPECMINE: A Large-Scale Corpus of Spec-Driven Development Artifacts

    [https://arxiv.org/abs/2608.25202](https://arxiv.org/abs/2608.25202)

    我们提出了SPECMINE，这是首个大规模语料库，通过两次普查系统地捕捉了GitHub上规范驱动开发工件，为研究规范如何转化为代码提供了基础数据。

    

    arXiv:2608.25202v1 公告类型：新 摘要：规范驱动开发（SDD）是一种快速兴起的新实践，其中由开发者编写、或（更常见地）由AI工具起草再由开发者整理的、结构化自然语言规范，驱动AI编码代理的实现。自2025年以来，一波工具（如GitHub Spec Kit [3]、OpenSpec [4]、AWS Kiro [5]以及数十种其他工具）已经出现，但这些工具产生的工件从未被大规模研究过。我们提出了SPECMINE，一个通过两次普查捕捉公共GitHub仓库中SDD的语料库：一次广泛普查覆盖了大多数工具的spec.md/specs.md文件（涵盖73,030个仓库中的470,795个文件，归属于17个命名工具），以及一次针对Kiro独特的需求/设计/任务布局的普查（涵盖12,910个仓库中的98,574个文件）。每个规范都附有完整的仓库元数据、完整的提交历史以及解析后的文档结构。规范如何转化为代码本身就是一个开放问题，因此对于...

    arXiv:2608.25202v1 Announce Type: new  Abstract: Spec-Driven Development (SDD) is a fast-emerging practice in which a structured natural-language specification, written by a developer, or (more often) drafted by an AI tool and then curated by the developer, drives an AI coding agent's implementation. A wave of tooling (GitHub Spec Kit [3], OpenSpec [4], AWS Kiro [5], and dozens of others) has appeared since 2025, yet the artifacts these tools produce have never been studied at scale. We present SPECMINE, a corpus that captures SDD in public GitHub repositories through two censuses: a broad census of spec.md/specs.md files covering most tools (470,795 files across 73,030 repositories, attributed to 17 named tools), and a Kiro census of its distinct requirements/design/tasks layout (98,574 files across 12,910 repositories). Each spec is enriched with full repository metadata, complete commit history, and parsed document structure. How a spec becomes code is itself an open question, so fo
    
[^20]: FlavourBench：用可执行的烹饪真实数据对前沿语言模型进行排名

    FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth

    [https://arxiv.org/abs/2608.20574](https://arxiv.org/abs/2608.20574)

    该论文提出了一个基于可执行烹饪真实数据的自动化基准测试FlavourBench，通过版本化系统和严格统计方法对27个前沿语言模型进行公平排名，消除了传统基准中的评判者偏差和缺失数据问题。

    

    开放式语言模型基准测试通常继承一个评判者：人类偏好小组、另一个模型，或脆弱的精确匹配键。我们引入了FlavourBench，一个自动化基准测试，其中版本化的烹饪系统提供密集、可执行的真实数据。每个任务呈现八种食材，并要求选择三种食材的组合；在模型执行前，Epicure对所有56种可能的组合进行评分。我们在一个包含534个任务的相同核心集上评估了27个前沿端点，涵盖替代、配对和受限组合。每个排名的模型在每个面板和家族中恰好有89个有效响应（总共14,418个模型-任务单元），消除了排行榜上的差异性缺失。FlavourBench分数是冻结任务分数的等家族均值。我们使用50,000个锚点聚类自助重采样进行同时95%分数区间，以及100,000次符号翻转抽样进行所有351个配对模型对比，并采用Holm校正。两个独立的...

    arXiv:2608.20574v1 Announce Type: new  Abstract: Open-ended language-model benchmarks usually inherit a judge: a human preference panel, another model, or a brittle exact-match key. We introduce FlavourBench, an automated benchmark in which a versioned culinary system supplies dense, executable ground truth. Each task presents eight ingredients and asks for a three-ingredient portfolio; before model execution, Epicure scores all 56 possible portfolios. We evaluate 27 frontier endpoints on an identical 534-task core spanning substitution, pairing, and constrained composition. Every ranked model has exactly 89 valid responses per panel and family (14,418 model-task cells total), eliminating differential missingness from the leaderboard. The FlavourBench Score is the equal-family mean of the frozen task scores. We use 50,000 anchor-cluster bootstrap replicates for simultaneous 95% score bands and 100,000 sign-flip draws for all 351 paired model contrasts, with Holm control. The two indepe
    
[^21]: 智能体配置管理（ACM）：面向受治理智能体系统的参考配置模型

    Agentic Configuration Management (ACM): A Reference Configuration Model for Governed Agentic Systems

    [https://arxiv.org/abs/2608.11166](https://arxiv.org/abs/2608.11166)

    本文提出智能体配置管理（ACM），一个与框架无关的参考配置模型，通过类型化配置项、不可变版本基线、配置与运行时分离以及依赖感知的影响传播等机制，实现对异构智能体系统配置的统一治理。

    

    智能体系统日益由异构的智能体、提示词、工具、模型、技能、复合子系统、策略和执行工作流组成，其配置在不同框架和运行时环境中不断演化。现有的 LLMOps 和 AgentOps 平台虽然支持编排和可观测性，但缺乏一个通用的配置治理模型，无法将这些系统作为连贯的、可版本化的配置来进行表示与治理。本文提出智能体配置管理，这是一个面向异构智能体系统的、与框架无关的治理与配置参考模型。ACM 融合了类型化且可独立版本化的智能体配置项、不可变的修订版本与基线、明确的配置与运行时分离、生命周期与保障语义、依赖感知的影响传播以及运行时溯源等机制。异构的原生配置通过语义投影被规范化（原文在此处截断）……

    arXiv:2608.11166v2 Announce Type: replace  Abstract: Agentic systems are increasingly composed of heterogeneous agents, prompts, tools, models, skills, composite subsystems, policies, and execution workflows whose configurations evolve across frameworks and runtime environments. Existing LLMOps and AgentOps platforms support orchestration and observability but do not provide a common configuration-governance model for representing and governing these systems as coherent, versioned configurations.   This paper introduces Agentic Configuration Management (ACM), a framework-independent governance and configuration reference model for heterogeneous agentic systems. ACM combines typed and independently versioned Agentic Configuration Items, immutable revisions and baselines, explicit configuration-runtime separation, lifecycle and assurance semantics, dependency-aware impact propagation, and runtime provenance. Heterogeneous native configurations are normalized through semantic projection i
    
[^22]: Web4智能体经济：格局、挑战与机遇的大规模实证研究

    The Web4 Agent Economy: A Large-Scale Empirical Study of the Landscape, Challenges, and Opportunities

    [https://arxiv.org/abs/2606.25876](https://arxiv.org/abs/2606.25876)

    本文开展了首个针对Web4智能体生态系统的大规模实证研究，系统考察了智能体的实际部署使用方式、开发者面临的工程挑战以及项目社区的应对情况。

    

    互联网正在从Web3向Web4过渡，在这一阶段，自主智能体将作为独立的经济行为体运作。这些智能体现在可以持有加密货币钱包、执行链上交易，并为外部API调用付费。这一过渡需要一套新的基础设施技术栈，以支持关键的智能体操作，包括智能体与工具的交互、智能体之间的支付以及可验证的智能体身份，其代表是模型上下文协议（Model Context Protocol）、x402和EIP-8004等新兴协议。尽管业界对这些协议的兴趣日益增长，但真实世界的Web4智能体生态系统在很大程度上仍未得到充分探索。为弥合这一差距，我们开展了首个针对Web4生态系统的大规模实证研究。具体而言，我们的研究围绕三个相互关联的问题展开：Web4智能体在实践中如何被部署和使用；开发者在构建Web4智能体时面临哪些工程挑战；当前项目社区如何应对这些挑战……

    arXiv:2606.25876v2 Announce Type: replace  Abstract: The Internet is transitioning from Web3 toward Web4, where autonomous agents serve as independent economic actors. These agents can now hold crypto wallets, execute on-chain trades, and pay for external API calls. This transition calls for a new infrastructure stack capable of supporting key agent operations, including agent-to-tool interaction, agent-to-agent payments, and verifiable agent identity, represented by emerging protocols such as the Model Context Protocol, x402, and EIP-8004. Despite growing industrial interest in these protocols, the real-world Web4 agent ecosystem remains largely underexplored. To bridge this gap, we conduct the first large-scale empirical study of the Web4 ecosystem. Specifically, our study targets three interconnected questions: how Web4 agents are deployed and used in practice; what engineering challenges developers face when building Web4 agents; how current project communities respond to these cha
    
[^23]: 编码智能体能否复现计算材料科学中的研究发现？

    Can Coding Agents Reproduce Findings in Computational Materials Science?

    [https://arxiv.org/abs/2605.00803](https://arxiv.org/abs/2605.00803)

    本文提出 AutoMat 基准，用于评估大语言模型编码智能体复现计算材料科学论文中科学论断的能力，涵盖恢复欠规范计算流程、驾驭专用工具链和验证证据是否支持论断三大挑战。

    

    大语言模型正越来越多地被部署为自主编码智能体，并在软件工程基准测试中取得了极为出色的性能。然而，这种成功能否迁移到计算科学工作流程中尚不明确，因为这类任务不仅需要强大的编码能力，还需要能够驾驭复杂的、特定领域的操作流程，并在科学论断的语境下解释结果。为了解答这一问题，我们提出了 AutoMat，一个用于评估基于大语言模型的智能体复现计算材料科学论断能力的基准。AutoMat 包含三个相互关联的挑战：恢复欠规范的计算流程、驾驭专用工具链，以及判断所得到的结果能否支持某一论断。通过与领域专家紧密合作，我们从真实的材料科学论文中精选出一组论断，用以测试编码智能体能否恢复（此处摘要内容被截断）

    arXiv:2605.00803v2 Announce Type: replace-cross  Abstract: Large language models are increasingly deployed as autonomous coding agents and have achieved remarkably strong performance on software engineering benchmarks. However, it is unclear whether such success transfers to computational scientific workflows, where tasks require not only strong coding ability, but also the ability to navigate complex, domain-specific procedures and to interpret results in the context of scientific claims. To address this question, we present AutoMat, a benchmark for evaluating LLM-based agents' ability to reproduce claims from computational materials science. AutoMat poses three interrelated challenges: recovering underspecified computational procedures, navigating specialized toolchains, and determining whether the resulting evidence supports a claim. By working closely with subject matter experts, we curate a set of claims from real materials science papers to test whether coding agents can recover 
    
[^24]: MUCOCO：代码大语言模型的自动化一致性测试

    MUCOCO: Automated Consistency Testing of Code LLMs

    [https://arxiv.org/abs/2604.19086](https://arxiv.org/abs/2604.19086)

    本文提出MUCOCO，一种利用保语义变异分析自动将程序转换为语义等价变异体、从而自动发现代码大语言模型不一致程序行为的自动化一致性测试方法。

    

    代码大语言模型（Code LLMs）经常表现出不一致的程序行为。开发者通常使用基准测试来评估代码大语言模型，但大多数基准测试是手工构建的、静态的，且并未针对一致性这一属性。在本工作中，我们提出了这样一个科学问题：如何自动发现代码大语言模型中不一致的程序行为？为应对这一挑战，我们提出了一种名为MUCOCO的自动化一致性测试方法，该方法采用保语义变异分析来暴露代码大语言模型中的不一致行为。给定一个编码查询，MUCOCO会自动将其程序转换为语义等价的程序（即变异体），并检测变异体与原始程序之间的不一致性（例如输出不同或测试失败）。我们使用四种（4）编码任务和七个（7）大语言模型对MUCOCO进行了评估。结果表明，MUCOCO在暴露不一致性方面是有效的，并且优于最接近的基线方法（TURBULENCE）。大约每（注：摘要原文在此处截断）

    arXiv:2604.19086v2 Announce Type: replace  Abstract: Code LLMs often portray inconsistent program behaviors. Developers typically employ benchmarks to assess Code LLMs, but most benchmarks are hand-crafted, static and do not target consistency property. In this work, we pose the scientific question: how can we automatically discover inconsistent program behaviors in Code LLMs? To address this challenge, we propose an automated consistency testing method, called MUCOCO, which employs semantic-preserving mutation analysis to expose inconsistent behaviors in code LLMs. Given a coding query, MUCOCO automatically transforms its program into semantically equivalent programs (aka mutants) and detects inconsistencies between the mutants and the original program (e.g., different output or test failure). We evaluate MUCOCO using four (4) coding tasks and seven (7) LLMs. Results show that MUCOCO is effective in exposing inconsistency and outperforms the closest baseline (TURBULENCE). About one in
    
[^25]: VulWeaver：织补破碎语义以实现有依据的漏洞检测

    VulWeaver: Weaving Broken Semantics for Grounded Vulnerability Detection

    [https://arxiv.org/abs/2604.10767](https://arxiv.org/abs/2604.10767)

    VulWeaver 是一种基于大语言模型的漏洞检测方法，其核心创新在于通过融合确定性规则与 LLM 语义推理构建增强的统一依赖图、结合显式与隐式上下文提取全面的漏洞信息，并借助漏洞类型专家指南的元提示引导 LLM 进行有依据的漏洞检测。

    

    检测源代码中的漏洞仍然至关重要且充满挑战，因为传统静态分析工具构建的程序表示不够准确，而现有的基于大语言模型（LLM）的方法往往遗漏关键的漏洞上下文信息，且缺乏有依据的推理能力。在本文中，我们提出了 VulWeaver，这是一种新颖的基于 LLM 的方法，它将破碎的程序语义织补成准确的表示，并提取全面的漏洞上下文，从而实现有依据的漏洞检测。VulWeaver 首先通过将确定性规则与基于 LLM 的语义推理相结合，构建增强的统一依赖图（UDG），以解决静态分析不准确的问题。随后，它将程序切片得到的显式上下文与包含使用、定义和声明信息的隐式上下文相结合，提取全面的漏洞上下文。最后，VulWeaver 采用带有漏洞类型特定专家指南的元提示技术来引导 L（摘要至此截断）。

    arXiv:2604.10767v3 Announce Type: replace  Abstract: Detecting vulnerabilities in source code remains critical yet challenging, as conventional static analysis tools construct inaccurate program representations, while existing LLM-based approaches often miss essential vulnerability context and lack grounded reasoning. In this paper, we introduce VulWeaver, a novel LLM-based approach that weaves broken program semantics into accurate representations and extracts holistic vulnerability context for grounded vulnerability detection. VulWeaver first constructs an enhanced unified dependency graph (UDG) by integrating deterministic rules with LLM-based semantic inference to address static analysis inaccuracies. It then extracts holistic vulnerability context by combining explicit contexts from program slicing with implicit contexts, including usage, definition, and declaration information. Finally, VulWeaver employs meta-prompting with vulnerability type specific expert guidelines to steer L
    
[^26]: JavaScript项目中依赖项重新分类的纵向研究

    A Longitudinal Study of Dependency Reclassifications in JavaScript Projects

    [https://arxiv.org/abs/2604.08747](https://arxiv.org/abs/2604.08747)

    本研究通过对33,087个JavaScript项目package.json文件的提交级分析，首次系统揭示了依赖项重新分类（包括移除和角色重分配）是一种普遍的维护活动，存在于79.1%的项目中，占所有依赖项维护提交的19.4%。

    

    现代软件项目依赖第三方依赖项，随着项目的演进，这些依赖项的声明必须得到持续维护。以往的研究主要关注依赖项的版本更新，而对于开发者如何随时间推移将依赖项分配到不同角色的研究则相对匮乏。在本文中，我们研究了JavaScript项目的开发者如何对其依赖项进行重新分类，包括移除依赖项和重新分配其角色。通过分析package.json文件在提交级别的修改，我们重建了依赖项角色变更历史，并识别出反复出现的重新分类实践。我们对33,087个积极维护依赖项的JavaScript项目的分析表明，依赖项重新分类是一种普遍存在的维护活动，出现在79.1%的所研究项目中，占所有依赖项维护提交的19.4%。在这些项目中，几乎所有项目（97.2%）都会在某些时候移除依赖项，而38.0%的项目经历了依赖项角色的重新分配。

    arXiv:2604.08747v2 Announce Type: replace  Abstract: Modern software projects depend on third-party dependencies, whose declarations must be maintained as projects evolve. Prior work has focused on dependency version updates, while much less is known about how developers assign dependencies to different roles over time. In this paper, we investigate how developers of JavaScript projects reclassify their dependencies, including removal and role reassignment. By analyzing commit-level modifications to package.json files, we reconstruct dependency role histories and identify recurring reclassification practices. Our analysis of 33,087 JavaScript projects with active dependency maintenance reveals that dependency reclassification is a prevalent maintenance activity, occurring in 79.1% of the studied projects, and accounting for 19.4% of all dependency-maintenance commits. Of these projects, nearly all (97.2%) remove dependencies at some point, while 38.0% undergo role reassignments across 
    
[^27]: PoC-Gym：迈向更可靠的LLM辅助概念验证漏洞利用生成

    PoC-Gym: Towards More Reliable LLM-Assisted Proof-of-Concept Exploit Generation

    [https://arxiv.org/abs/2602.04165](https://arxiv.org/abs/2602.04165)

    提出了PoC-Gym流水线，通过结合静态与动态信息（如CVE定制提示、静态追踪和覆盖率反馈）及多阶段验证机制，实现更可靠的基于LLM的Java安全漏洞PoC自动生成。

    

    近年来，大型语言模型（LLM）已被用于安全相关任务，包括生成概念验证漏洞利用程序。已有多种LLM辅助方法被提出；这些方法通常从漏洞描述生成PoC并使用额外的引导。然而，此类方法往往效果不佳，因为它们用于验证的信号——如打印的标记、生成的文件或运行时副作用——可能并不代表漏洞确实被触发。对更可靠的PoC生成方法的研究亟待开展，但仍充满挑战。我们提出了PoC-Gym，这是一个基于LLM的Java安全漏洞PoC生成流水线。PoC-Gym同时利用静态和动态信息，例如针对CVE定制的提示、静态追踪和基于覆盖率的反馈，并迭代式地生成PoC候选。每个候选都经过一系列验证：执行是否完整、是否表现出成功信号，以及……（原文在此处截断）

    arXiv:2602.04165v3 Announce Type: replace  Abstract: Recently Large Language Models (LLMs) have been used in security-related tasks, including generating proof-of-concept (PoC) exploits. Several LLM-assisted approaches have been proposed; they typically generate PoCs from vulnerability descriptions and use additional guidance. But, such approaches are often ineffective because the signals-such as printed markers, generated files, or runtime side effects-that they use for validation may not imply that the vulnerability is triggered. Research for more reliable PoC generation is in need but yet remains challenging. We propose PoC-Gym, a pipeline for LLM-based PoC generation for Java security vulnerabilities. PoC-Gym uses both static and dynamic information, e.g., CVE-tailored prompts, static traces, and coverage-based feedback, and iteratively generates PoC candidates. Each candidate goes through a series of validations: whether the execution is complete, manifests a success signal, and r
    

