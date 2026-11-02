# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dittany: Contribution-Guided Repair Framework for MAS Applications](https://arxiv.org/abs/2610.42144) | Dittany 通过从任务需求中构建贡献图来追踪多智能体系统中已完成的有效结果与未满足的接受条件，从而在运行时实现动作级修正和图级恢复，在保留有效贡献的同时修复未满足任务需求的执行过程。 |
| [^2] | [No More Import Errors: Constrained Decoding for Guaranteed Valid Imports in LLM-Generated Tests](https://arxiv.org/abs/2610.41878) | 提出带查找的约束束解码（CBDL）方法，在解码阶段通过避免过早提交导致重复导入的token，既保证了LLM生成测试的导入100%有效，又解决了导入串流耗尽token预算的问题，且可无缝集成到现有测试生成流水线中。 |
| [^3] | [Stuck on Red? Understanding GitHub Actions Failing Streaks](https://arxiv.org/abs/2610.41863) | 该论文首次对 GitHub Actions 工作流的“失败连续期”开展了超大规模实证研究（覆盖 3000 多万次运行），揭示了失败运行往往接连发生、多用途工作流更易陷入长期失败等规律。 |
| [^4] | [Reparo: Synthesizing and Validating Access Control Policies](https://arxiv.org/abs/2610.41800) | Reparo 是一个基于大语言模型的访问控制策略合成、修复与验证流水线，它通过挖掘代码仓库中分散的授权意图证据生成初始策略，并利用官方验证工具自动修复错误并进行语义验证。 |
| [^5] | [A Failure Mode Taxonomy and Severity Model for Ethereum Smart Contracts](https://arxiv.org/abs/2610.41685) | 本文设计并验证了首个针对以太坊智能合约的失效模式分类法和严重性模型，填补了现有研究只关注漏洞、缺陷与攻击而忽视合约部署后可观察失效行为的空白。 |
| [^6] | [What Carries Architectural Signal? A Code-to-Architecture Mapping Study](https://arxiv.org/abs/2610.41628) | 该研究系统性地探究了代码到架构映射中究竟是哪些因素——文件表示类型、依赖邻居、文件夹路径或成员节点——真正承载了架构信号，指出依赖与目录信息可能已隐含在文件表示中，额外的图处理未必带来实质增益。 |
| [^7] | [Prompt Less, Infer More: Static Analysis-Guided LLM Type Annotation for Python](https://arxiv.org/abs/2610.41617) | SALT是一个由静态分析引导的LLM类型推断系统，通过静态解析类型、优先排序LLM查询并构建简洁的类型感知上下文，在12个真实Python项目上实现了最佳类型标注准确率，同时将LLM调用成本最多降低78.5%。 |
| [^8] | [SCRBench: Beyond Equivalence Checking for Semantic Change Reasoning in Code](https://arxiv.org/abs/2610.41562) | 该论文提出SCRBench基准，通过诊断语义变化和重构代码更改序列这两项新任务，超越了传统的语义等价性“是/否”判断，从而更深入地评估语言模型对代码语义变化的真实理解能力。 |
| [^9] | [Post-SQLFix: Property-Guided Repair of LLM-Generated SQL with Runtime Verification](https://arxiv.org/abs/2610.41472) | 该论文提出Post-SQLFix框架，将Text-to-SQL修复重新表述为属性引导的程序修复问题，通过方言感知的规范查询结构（CQS）和三层分层静态分析器对LLM生成的SQL进行验证，从而解决了现有执行反馈方法中“未经验证的模型既提出修复又做最终仲裁”的仲裁悖论。 |
| [^10] | [What Does a Repair Agent Gain from Remembering Its Failure? An Exploratory Study on Real Python Bugs](https://arxiv.org/abs/2610.41433) | 在75个真实Python bug的修复实验中发现，为失败后重试的智能体提供失败记忆（对话记录、自写经验教训或测试信号）虽在点估计上有提升但均不显著，且模型自写的经验教训相比完整对话记录并无优势。 |
| [^11] | [Architectural Debt Enforcement under Incomplete Observation: A Controlled Study of Ratchets and Evidence Qualification](https://arxiv.org/abs/2610.41276) | 该论文提出一种将观测资格认定、禁止关系标识、继承债务与授权基线更新相互分离的决策契约，用于在不完全观测条件下执行架构债务门控，并通过受控实验表明其候选棘轮机制在480个构造场景中匹配了全部预期结果，同时诚实保留了观测覆盖率限制与失败案例。 |
| [^12] | [Cognitive Inclusivity in AI Coding Assistants: Evidence from GitHub Copilot](https://arxiv.org/abs/2610.41192) | 该研究通过让20名学生使用GitHub Copilot完成代码生成、调试和重构任务的受控实验室实验，揭示了具有不同认知风格的开发者在使用AI编程助手时可能面临的认知包容性挑战。 |
| [^13] | [Lineage Is Memory: Parent Inheritance and Feedback Disclosure in Short-Budget LLM-Guided Optimization](https://arxiv.org/abs/2610.41157) | 该研究通过1,750次预注册实验首次将父代继承与反馈披露效应分离，发现贪婪的父代继承——即谱系本身充当记忆——才是短预算LLM引导优化中绝大部分进展的真正来源，而非每轮反馈所覆盖的失败模式广度。 |
| [^14] | [One Verdict Is Not Enough: Regression Testing Learned World Models and Their Test Harnesses](https://arxiv.org/abs/2610.41130) | 该论文提出了TetraState，一种针对学习型世界模型的回归测试协议，通过在四种固定的模型原生配置下运行成对的可能/不可能视频测试、单独报告无结论测试与失败执行、并将测试框架作为被测系统的一部分进行版本管理，从而解决了单一配置下测试结论不可靠以及测试框架故障导致错误判定的问题。 |
| [^15] | [RepGen: Reproducing Deep Learning Bugs through Context-Aware Code Generation and Verification](https://arxiv.org/abs/2610.41097) | RepGen 是一个开源工具，通过探索代码库构建上下文、生成候选复现脚本，并利用静态检查与大语言模型反馈进行迭代验证，从而在缺陷报告不完整的情况下自动复现深度学习缺陷。 |
| [^16] | [AIFaultBench: A Reproducible Benchmark of Real-World Faults in AI Software Systems](https://arxiv.org/abs/2610.41094) | 该论文提出了AIFaultBench，一个涵盖六类AI系统、包含来自105个软件仓库的770个真实故障的可复现基准测试，在文档化执行条件下成功复现了84.7%的故障，为AI软件调试研究提供了真实且可验证的故障资源。 |
| [^17] | [An Empirical Study of Gas Optimization Practices in Solidity Smart Contracts](https://arxiv.org/abs/2610.40885) | 该论文通过对2018至2025年间GitHub上开源Solidity仓库中400个与gas相关提交的多阶段实证分析，构建了智能合约gas优化实践的分类体系，并揭示了理论优化模式与开发者真实实践之间的显著差距。 |
| [^18] | [RiskyPy: Toward Model-Calibrated Prompt Generation for Joint Security and Functionality Evaluation of LLM-Generated Code](https://arxiv.org/abs/2610.40735) | RiskyPy提出了一种自动化流水线，利用功能性与漏洞金丝雀生成与模型能力相校准的安全导向代码生成数据集，实现了对LLM生成代码安全性与功能性的联合评估。 |
| [^19] | [MetaVul: Agentic Harness Discovery for Vulnerability Detection](https://arxiv.org/abs/2610.40721) | MetaVul提出了一个用于漏洞检测的自我改进框架发现机制，通过基于工作流的优化循环，利用成对的漏洞函数与修复函数的反馈迭代优化LLM智能体框架，克服了传统固定手工编写的框架难以自适应评估反馈、且改进难以跨数据集和模型迁移的局限。 |
| [^20] | [A small HER2 reader for routine H&E slides: real-world deployment](https://arxiv.org/abs/2610.40662) | 研究者构建了一个仅2200万参数的小型HER2判读器，可直接从常规H&E染色切片预测HER2状态，在外部验证中位列7个方法之首（AUC 0.812），性能与体积大4倍和50倍的编码器持平，并已实现真实世界部署，可作为病理医生评估HER2状态的基线工具。 |
| [^21] | [PR4Code: A Pull Requests Dataset for AI Code Generation](https://arxiv.org/abs/2610.39890) | 本文提出了PR4Code数据集，收录了从GitHub收集的4,508个Java和8,831个Python精选Pull Request（含元数据、提交历史和代码变更），用于在完整软件功能层面支持AI代码生成的研究与评估。 |
| [^22] | [From GPU to TPU with Coding Agents: What Works, What Fails, and What It Costs?](https://arxiv.org/abs/2610.34422) | 该研究构建了148个PyTorch研究项目的数据集，并使用Claude Code将其中10个项目从NVIDIA GPU迁移到Google Cloud TPU，通过独立评估发现迁移平均需2.4次人工干预、76%的执行尝试失败且近半数失败不报错，首次系统量化了智能体驱动代码迁移的可信度与成本。 |
| [^23] | [Can Agentic Code Review Catch Vulnerabilities Before They Are Merged?](https://arxiv.org/abs/2610.33509) | 提出了SecPRBench——首个基于经过验证的真实漏洞引入提交来评估智能体安全代码审查的综合性基准，涵盖Firefox、V8、OpenSSL等九个关键系统的265个任务，并发现八个前沿模型的召回率仅为32.5%至78.1%。 |
| [^24] | [Superintelligent Software Engineering: Governance That Scales with Capability](https://arxiv.org/abs/2610.31306) | 本文提出“超智能软件工程”概念，主张当AI智能体能力超越人类工程师时，必须让治理机制（门禁、契约、决策与衰减控制）随能力同步扩展，才能应对知识衰减与人工审查失效，保障软件正确性。 |
| [^25] | [CodeLift: Your Test Suite Is Secretly a Dense SWE Curriculum](https://arxiv.org/abs/2610.17224) | CodeLift通过在每次迭代中对测试与补丁进行分组、并给少数补丁能通过的测试组赋予更高权重，动态校准测试通过率奖励，把测试套件转化为自适应课程，为软件工程智能体的强化学习提供更可靠的密集奖励信号。 |
| [^26] | [Measuring Delivery Consistency in Practice: A DORA Extension from a Multi-Platform Release Setting](https://arxiv.org/abs/2606.00364) | 该论文提出“交付一致性”（DC）这一有界二阶矩度量指标，弥补了DORA指标无法反映部署节奏规律性的不足，并将其整合进包含八种原型的“交付健康矩阵”诊断框架，在多平台软件交付环境中进行了实践评估。 |
| [^27] | [Human-AI Synergy in Agentic Code Review](https://arxiv.org/abs/2603.15911) | 该论文通过对300个开源项目中278,790次代码审查对话的大规模实证分析，比较了人类审查者与AI智能体的反馈差异，揭示了人机协作模式及交互方式如何影响代码审查结果与建议采纳率。 |
| [^28] | [Instruction-Tuning Open-Weight Language Models for BPMN Model Generation](https://arxiv.org/abs/2512.12063) | 该论文提出InstruBPM方法，通过成对的文本-图表数据对开源大语言模型进行指令微调，并结合参数高效微调与量化技术，使其能够以低成本、保护隐私的本地化部署方式从自然语言描述直接生成高质量的BPMN流程模型。 |

# 详细

[^1]: Dittany：面向多智能体系统（MAS）应用的贡献引导式修复框架

    Dittany: Contribution-Guided Repair Framework for MAS Applications

    [https://arxiv.org/abs/2610.42144](https://arxiv.org/abs/2610.42144)

    Dittany 通过从任务需求中构建贡献图来追踪多智能体系统中已完成的有效结果与未满足的接受条件，从而在运行时实现动作级修正和图级恢复，在保留有效贡献的同时修复未满足任务需求的执行过程。

    

    基于大语言模型（LLM）的多智能体系统（MAS）应用在产生有用中间结果的同时，可能仍未能满足任务需求。修复此类执行过程需要识别出尚待完成的工作，以及能够支持这些工作的已完成结果。我们提出了 Dittany，一个面向 MAS 应用的贡献引导式运行时修复框架。Dittany 从任务需求和方法规范中推导出预期贡献与接受条件，并将它们的依赖关系连接成一张贡献图。运行时检查用于定位未满足的条件，并确定哪些结果仍然可用。当可用输入足够时，这些结果可引导动作级修正；当前置条件或未完成的工作需要修订时，则引导图级恢复。两种修复范围都会保留已验证的贡献，并在后续工作使用修复后的输出之前对其进行检查。我们在六个 MAS 框架上的 30 个应用和 9,000 个任务上对 Dittany 进行了评估，它能够检测运行时故障……

    arXiv:2610.42144v1 Announce Type: new  Abstract: LLM-based multi-agent system (MAS) applications can produce useful intermediate results while leaving task requirements unsatisfied. Repairing such executions requires identifying the work still needed and the completed results that can support it. We present \tool, a contribution-guided runtime repair framework for MAS applications. \tool derives expected contributions and acceptance conditions from task requirements and method specifications, and connects their dependencies in a contribution graph. Runtime checks locate unmet conditions and establish which results remain usable. These results guide action-level correction when available inputs suffice, or graph-level recovery when prerequisites or unfinished work need revision. Both repair scopes retain validated contributions and check repaired outputs before subsequent work uses them. We evaluate \tool on 30 applications across six MAS frameworks and 9,000 tasks. It detects runtime f
    
[^2]: 告别导入错误：约束解码确保LLM生成测试中的有效导入

    No More Import Errors: Constrained Decoding for Guaranteed Valid Imports in LLM-Generated Tests

    [https://arxiv.org/abs/2610.41878](https://arxiv.org/abs/2610.41878)

    提出带查找的约束束解码（CBDL）方法，在解码阶段通过避免过早提交导致重复导入的token，既保证了LLM生成测试的导入100%有效，又解决了导入串流耗尽token预算的问题，且可无缝集成到现有测试生成流水线中。

    

    基于LLM的测试生成流水线最常见的失败原因之一是导入错误。语法约束解码可以在token生成过程中修改LLM最后一层的概率，以保证导入的有效性。然而，我们发现这种方法可能导致退化解，即重复的有效导入（导入串流），在生成测试之前就耗尽了token预算。我们提出了带查找的约束束解码（Constrained Beam Decoding with Lookup，CBDL），这是一种新方法，在保证导入有效性的同时，通过避免过早提交会导致重复导入的token来解决导入串流问题。由于CBDL在解码阶段运行，因此可以集成到现有的基于LLM的测试生成流水线中。我们为Python和Java实现了CBDL，并在两个测试生成任务上进行了评估，结果表明生成的测试具有100%的有效导入，优于静态方法（RAG和LIBRO）以及动态方法（Agentic Repair）的导入有效性表现。

    arXiv:2610.41878v1 Announce Type: new  Abstract: One of the most common failure reasons of LLM-based test generation pipelines is import errors. Grammar constrained decoding can modify the probabilities of the LLM's final layer during token generation to guarantee valid imports. However, we find that it can lead to degenerate solutions, i.e., repetitive valid imports (import streaks) that exhaust the token budget before a test is generated. We propose Constrained Beam Decoding with Lookup (CBDL), a novel method that guarantees import validity while addressing import streaks by avoiding premature commitment to tokens that lead to repetitive imports. Because it operates at the decoding stage, CBDL can be integrated into existing LLM-based test generation pipelines.   We implement CBDL for Python and Java and evaluate it on two test generation tasks, showing that generated tests have 100% valid imports, outperforming both static (RAG and LIBRO) and dynamic (Agentic Repair) import-validity
    
[^3]: 困在红色状态？理解 GitHub Actions 的连续失败现象

    Stuck on Red? Understanding GitHub Actions Failing Streaks

    [https://arxiv.org/abs/2610.41863](https://arxiv.org/abs/2610.41863)

    该论文首次对 GitHub Actions 工作流的“失败连续期”开展了超大规模实证研究（覆盖 3000 多万次运行），揭示了失败运行往往接连发生、多用途工作流更易陷入长期失败等规律。

    

    GitHub Actions 是 GitHub 仓库事实上的 CI/CD 服务。工作流不应在非预期的情况下失败，也不应持续失败过久。我们对 43K+ 个 GitHub 仓库中 154K+ 个工作流的 3090 万余次运行进行了为期 238 天的实证研究，研究对象是“失败连续期”，即工作流连续不断失败的运行序列。我们基于每次运行的触发方式和提交内容，研究失败连续期如何开始、如何结束以及期间发生了什么。我们还按用途（如构建、测试或部署）对所有工作流进行分类，并检查了一小部分失败连续期的日志。研究发现，失败的运行大多以连续期的形式出现：其中四分之三紧跟在另一次失败的运行之后，而在至少有三次运行的工作流中，一半以上至少经历过一次失败连续期。服务于多种用途的工作流更容易出现失败连续期，即使规模相同也是如此。大多数失败连续期在数小时内结束，但仍有五分之一的失败连续期持续了更长的时间。

    arXiv:2610.41863v1 Announce Type: new  Abstract: GitHub Actions is the de facto CI/CD service for GitHub repositories. Its workflows should not fail when they are not expected to, nor keep on failing too long. We empirically study failing streaks, i.e., uninterrupted series of failing runs of a workflow, in 30.9M+ runs of 154K+ workflows of 43K+ GitHub repositories over 238 days. We study how failing streaks begin and end, and what happens during them, based on the trigger and the commit of each run. We also classify all workflows by their purposes, such as building, testing or deploying, and we inspect the logs of a small sample of failing streaks. Failing runs mostly come in streaks: three quarters of them follow another failing run, and half of the workflows with at least three runs have at least one failing streak. Workflows serving multiple purposes have failing streaks more often, even at the same size. Most failing streaks end within hours, but the one in five that last more tha
    
[^4]: Reparo：访问控制策略的合成与验证

    Reparo: Synthesizing and Validating Access Control Policies

    [https://arxiv.org/abs/2610.41800](https://arxiv.org/abs/2610.41800)

    Reparo 是一个基于大语言模型的访问控制策略合成、修复与验证流水线，它通过挖掘代码仓库中分散的授权意图证据生成初始策略，并利用官方验证工具自动修复错误并进行语义验证。

    

    为软件应用程序编写正确的访问控制策略十分困难，因为这既需要掌握授权语言方面的专业知识，又需要深入理解应用程序的授权需求。现有的合成与挖掘方法通常假设存在显式的授权输入以及可信的参考策略。然而，授权逻辑往往分散在代码、注释和文档中，而非集中于单一工件；且真实应用仓库中由开发者定义的策略常常不完整、过时或存在错误。我们提出了 Reparo，一个借助大语言模型（LLM）实现的访问控制策略合成、修复与验证流水线。Reparo 从代码仓库中收集有关授权意图的证据，并利用这些证据引导 LLM 生成初始策略。随后，它使用官方的策略语言验证工具来识别并自动修复策略中的错误。在语义验证方面，另一个不同的 LLM 会推导出……（摘要原文在此处截断）

    arXiv:2610.41800v1 Announce Type: new  Abstract: Writing correct access control policies for software applications is difficult because it requires expertise in the authorization language and a thorough understanding of the application's authorization requirements. Existing synthesis and mining approaches assume explicit authorization inputs and a trusted reference policy. However, authorization logic is often distributed across code, comments, and documentation rather than captured in a single artifact, and developer-defined policies in real-world application repositories are often incomplete, outdated, or incorrect. We present Reparo, an LLM-aided pipeline for access control policy synthesis, repair, and validation. Reparo collects repository evidence for authorization intent and uses it to guide an LLM in generating an initial policy. It then uses official policy-language validation tools to identify and automatically repair errors. For semantic validation, a different LLM derives a
    
[^5]: 以太坊智能合约的失效模式分类法与严重性模型

    A Failure Mode Taxonomy and Severity Model for Ethereum Smart Contracts

    [https://arxiv.org/abs/2610.41685](https://arxiv.org/abs/2610.41685)

    本文设计并验证了首个针对以太坊智能合约的失效模式分类法和严重性模型，填补了现有研究只关注漏洞、缺陷与攻击而忽视合约部署后可观察失效行为的空白。

    

    智能合约是区块链系统的核心组件，但其不可更改性使得已部署代码中的失效是灾难性且不可逆转的。现有工作对漏洞、缺陷和攻击进行了分类，然而智能合约的可观察失效——即合约部署后所观察到的与预期行为或预期账本状态的偏差——在很大程度上被忽视了。在本文中，我们设计并验证了一个针对以太坊智能合约的失效模式分类法，以及一个根据可观察影响对失效进行排序的严重性模型。我们以文献综述作为输入，遵循 Nickerson 等人的分类法开发方法来推导失效模式及其维度，并通过基于系统性文献综述、OpenSCV 漏洞目录以及搜索截止日期之后报告的智能合约事件的饱和度检验，对所得到的分类法进行验证。每个失效模式沿五个维度进行特征刻画（原文在此处截断）。

    arXiv:2610.41685v1 Announce Type: new  Abstract: Smart contracts are a core component of blockchain systems, but their immutability makes failures in deployed code catastrophic and irreversible. Existing work classifies vulnerabilities, defects, and attacks, yet the observable failures of smart contracts, the deviations from intended behavior or the expected ledger state seen once a contract is deployed, have been largely disregarded. In this paper, we design and validate a failure mode taxonomy for Ethereum smart contracts, together with a severity model that ranks failures by their observable impact. Using a literature review as input, we follow the taxonomy development method of Nickerson et al. to derive the failure modes and their dimensions, and we subject the resulting taxonomy to a saturation exercise over a systematic literature review, the OpenSCV vulnerability catalog, and smart contract incidents reported after the search cutoff. Each failure mode is characterized along fiv
    
[^6]: 什么承载了架构信号？一项代码到架构映射研究

    What Carries Architectural Signal? A Code-to-Architecture Mapping Study

    [https://arxiv.org/abs/2610.41628](https://arxiv.org/abs/2610.41628)

    该研究系统性地探究了代码到架构映射中究竟是哪些因素——文件表示类型、依赖邻居、文件夹路径或成员节点——真正承载了架构信号，指出依赖与目录信息可能已隐含在文件表示中，额外的图处理未必带来实质增益。

    

    代码到架构映射是将源文件分配到参考架构各模块的任务。实现制品的表示可以捕获代码文本、依赖关系和目录组织，但仅凭预测性能的提升并不能证明额外图处理的贡献——依赖关系可能已经编码在这些表示之中，而目录结构可能与参考分解高度吻合。我们在十个标注预算有限的 Java、C 和 C++ 系统上，研究了制品表示与图上下文如何影响映射效果。我们比较了词汇、结构和上下文三类文件表示，并在有、无依赖邻居的情况下分别评估每个预测器。基于异构图Transformer（HGT），我们比较了通过概括每个文件程序元素的成员节点、文件夹节点或二者同时扩展的文件级图，并对比了直接在文件表示中编码文件夹路径的变体。（原文摘要在此处截断）

    arXiv:2610.41628v1 Announce Type: cross  Abstract: Code-to-architecture mapping assigns source files to modules of a reference architecture. Representations of implementation artifacts can capture code text, dependencies, and directory organization, but predictive gains alone do not establish the contribution of additional graph processing. Dependencies may already be encoded in these representations, while directories may closely match the reference decomposition. We investigate how artifact representations and graph context affect mapping on ten Java, C, and C++ systems with a limited labeling budget. We compare lexical, structural, and contextual file representations and assess each predictor with and without dependency neighbors. Using a Heterogeneous Graph Transformer (HGT), we compare file-level graphs extended with member nodes summarizing each file's program elements, folder nodes, or both, alongside a variant encoding folder paths directly in file representations. We compare f
    
[^7]: 更少提示，更多推断：基于静态分析引导的LLM Python类型标注

    Prompt Less, Infer More: Static Analysis-Guided LLM Type Annotation for Python

    [https://arxiv.org/abs/2610.41617](https://arxiv.org/abs/2610.41617)

    SALT是一个由静态分析引导的LLM类型推断系统，通过静态解析类型、优先排序LLM查询并构建简洁的类型感知上下文，在12个真实Python项目上实现了最佳类型标注准确率，同时将LLM调用成本最多降低78.5%。

    

    类型注解能够提升Python程序的可维护性和可靠性，但由于手动标注耗时且容易出错，许多真实世界的项目仍然缺乏足够的类型注解。近年来基于神经网络和LLM的类型推断系统虽然可以恢复缺失的注解，但它们往往以牺牲准确性来换取效率。现有方法常常未能充分利用项目范围内的类型证据，或者通过不必要的模型调用和冗余的提示低效地利用这些证据。我们提出了SALT，一个由静态分析引导的基于LLM的类型推断系统。SALT将静态分析与LLM推断相结合，以静态方式解析类型、对LLM查询进行优先级排序，并构建简洁的类型感知上下文。为了证明SALT的有效性，我们在12个真实的Python项目上对其进行了评估。SALT在所评估的系统中取得了最佳准确率，与强大的编码助手基线相比，估计的LLM成本最多降低78.5%。它还……

    arXiv:2610.41617v1 Announce Type: new  Abstract: Type annotations improve the maintainability and reliability of Python programs, but many real-world projects remain under-annotated because manual annotation is time-consuming and error-prone. Recent neural and LLM-based type inference systems can recover missing annotations, yet they often trade accuracy for efficiency. Existing approaches often fail to exploit project-wide type evidence, or do so inefficiently through unnecessary model calls and noisy prompts. We present SALT, an LLM-based type inference system guided by static analysis. SALT integrates static analysis with LLM inference to resolve types statically, prioritize LLM queries, and construct concise type-aware context. To demonstrate the effectiveness of SALT, we evaluated it on 12 real-world Python projects. SALT achieves the best accuracy among the evaluated systems and reduces estimated LLM cost by up to 78.5% compared with a strong coding-assistant baseline. It also sh
    
[^8]: SCRBench：超越等价性检查的代码语义变化推理

    SCRBench: Beyond Equivalence Checking for Semantic Change Reasoning in Code

    [https://arxiv.org/abs/2610.41562](https://arxiv.org/abs/2610.41562)

    该论文提出SCRBench基准，通过诊断语义变化和重构代码更改序列这两项新任务，超越了传统的语义等价性“是/否”判断，从而更深入地评估语言模型对代码语义变化的真实理解能力。

    

    理解代码变化如何影响程序语义是语言模型代码推理的核心方面。这一能力通常通过询问两个程序是否语义等价来评估，但简单的“是/否”回答无法揭示模型是否知道是哪次代码更改导致了语义差异，也无法揭示是哪一序列的代码更改产生了修改后的程序。我们研究了两个直接提出这些问题的任务——诊断语义变化和重构代码更改序列——并结合语义等价性检测进行研究。我们构建了SCRBench，其中包含67,658对Python程序，这些程序对是通过应用保留或改变程序语义的代码更改序列生成的，并在4,500对的分层抽样子集上评估了五个语言模型。检测语义等价性暴露了较弱模型的不足，其准确率随着更改数量的累积而下降，但最强的模型仍接近性能上限。诊断语义变化则揭示了……

    arXiv:2610.41562v1 Announce Type: cross  Abstract: Understanding how code changes affect program semantics is a core aspect of code reasoning in language models. It is commonly evaluated by asking whether two programs are semantically equivalent, but a yes/no answer does not reveal whether a model knows which code change caused a semantic difference, or which sequence of code changes produced the modified program. We study two tasks that ask these questions directly, diagnosing semantic change and reconstructing sequences of code changes, alongside detecting semantic equivalence. We build SCRBench, 67,658 Python program pairs generated by applying sequences of code changes that preserve or alter program semantics, and evaluate five language models on a stratified subset of 4,500 pairs. Detecting semantic equivalence exposes weaker models, whose accuracy declines as changes accumulate, but the strongest models remain near the ceiling. Diagnosing semantic change reveals weaknesses that e
    
[^9]: Post-SQLFix：基于属性引导与运行时验证的LLM生成SQL修复方法

    Post-SQLFix: Property-Guided Repair of LLM-Generated SQL with Runtime Verification

    [https://arxiv.org/abs/2610.41472](https://arxiv.org/abs/2610.41472)

    该论文提出Post-SQLFix框架，将Text-to-SQL修复重新表述为属性引导的程序修复问题，通过方言感知的规范查询结构（CQS）和三层分层静态分析器对LLM生成的SQL进行验证，从而解决了现有执行反馈方法中“未经验证的模型既提出修复又做最终仲裁”的仲裁悖论。

    

    面向Text-to-SQL任务的大型语言模型（LLM）经常生成含有语法错误、无效标识符引用或违反SQL作用域与类型规则的查询。修复这些错误本质上是一个程序正确性问题，然而现有方法依赖于执行反馈（EF），其症状级别且方言相关的错误信息迫使LLM去猜测根本原因。我们在这一范式中识别出一个“仲裁悖论”：系统依赖未经验证的模型既提出修复方案，又对修复方案是否满足所需属性做出最终判断。我们通过将Text-to-SQL修复重新表述为属性引导的程序修复来解决这一悖论。我们的框架Post-SQLFix将SQL规范化为方言感知的规范查询结构（CQS），并通过具有三个分层属性层（语法层、上下文无关结构层、上下文相关语义层）的分层静态分析器对其进行验证。该分析器……

    arXiv:2610.41472v1 Announce Type: new  Abstract: Large Language Models (LLMs) for Text-to-SQL frequently generate queries with syntax errors, invalid identifier references, or violations of SQL scoping and typing rules. Repairing these errors is a program correctness problem, yet existing approaches rely on Execution Feedback (EF), whose symptom-level, dialect-specific error messages force the LLM to speculate on root causes. We identify an Arbitration Paradox in this paradigm: the system relies on unverified models both to propose repairs and to make the final judgment on whether those repairs satisfy the required properties. We resolve this paradox by reformulating Text-to-SQL repair as property-guided program repair. Our framework, Post-SQLFix, canonicalizes SQL into a dialect-aware Canonical Query Structure (CQS) and validates it through a hierarchical static analyzer with three stratified property layers (Syntactic, Context-Free Structural, Context-Sensitive Semantic). The analyze
    
[^10]: 修复智能体能从记住自己的失败中获得什么？基于真实Python Bug的探索性研究

    What Does a Repair Agent Gain from Remembering Its Failure? An Exploratory Study on Real Python Bugs

    [https://arxiv.org/abs/2610.41433](https://arxiv.org/abs/2610.41433)

    在75个真实Python bug的修复实验中发现，为失败后重试的智能体提供失败记忆（对话记录、自写经验教训或测试信号）虽在点估计上有提升但均不显著，且模型自写的经验教训相比完整对话记录并无优势。

    

    当一个编程智能体在修复任务中失败时，通常会在重试时附带失败记录：完整对话记录、自行撰写的经验教训，或提取的测试信号。我们探究这种失败记忆在重试中究竟带来了什么改变。在75个BugsInPy bug上，我们在五种条件下对gpt-5.4-mini的181次失败尝试分别进行了重试，采用预先声明的分析方案并报告其中的偏差。两项预先声明的后续实验使用更大的gpt-5.4模型重复了普通重试、对话记录和经验教训三种条件，并增加了仅提示（notice-only）的对照组。在暴露bug的测试上，每种记忆条件在点估计上均优于普通重试（相比29.8%的基础水平提升+5.5至+9.9个百分点），但均未达到统计显著。另有五项发现值得关注。第一，模型撰写的经验教训相比其总结的对话记录并无优势（-1.7个百分点，95%置信区间[-8.7, +4.9]；gpt-5.4为-3.5），尽管它还需要额外消耗一次模型调用。第二，对话记录的作用可能不止于告知智能体其失败：一个提示（摘要在此处截断）

    arXiv:2610.41433v1 Announce Type: new  Abstract: When a coding agent fails a repair task, it is often retried with a record of the failure: the transcript, a self-written lesson, or extracted test signals. We ask what such failure memory changes in the retry. On 75 BugsInPy bugs, we retried each of 181 failed attempts of gpt-5.4-mini under five conditions, with a predeclared analysis whose deviations we report. Two predeclared follow-ups repeated the plain-retry, transcript and lesson conditions with the larger gpt-5.4 and added a notice-only control. On the bug-revealing tests, every memory condition gained over a plain retry in point estimate (+5.5 to +9.9 points over 29.8%), but none significantly. Five further findings stand out. First, a model-written lesson showed no advantage over the transcript it summarises (-1.7 points, 95% CI [-8.7, +4.9]; -3.5 for gpt-5.4), although it costs an extra model call. Second, the transcript may do more than tell the agent that it failed: a notice
    
[^11]: 不完全观测下的架构债务执行：棘轮机制与证据资格认定的受控研究

    Architectural Debt Enforcement under Incomplete Observation: A Controlled Study of Ratchets and Evidence Qualification

    [https://arxiv.org/abs/2610.41276](https://arxiv.org/abs/2610.41276)

    该论文提出一种将观测资格认定、禁止关系标识、继承债务与授权基线更新相互分离的决策契约，用于在不完全观测条件下执行架构债务门控，并通过受控实验表明其候选棘轮机制在480个构造场景中匹配了全部预期结果，同时诚实保留了观测覆盖率限制与失败案例。

    

    架构债务门控必须能够区分允许的变更与观测不完全的变更。我们研究了一种决策契约，将观测资格认定、禁止关系标识、继承债务与授权基线更新相互分离。在来自四个代码库的240个固定文件中，当前观测器与程序化参考共同完成了166个文件（69.17%）的资格认定，且所有认定结果与参考相比的边完全一致；其余74个文件仍处于参考未知状态。各代码库的覆盖率从13/60到59/60不等。在共享60个源锚点的480个构造状态场景中，候选棘轮机制与成熟的依赖巡检基线均匹配了全部预期结果，而静态基线和仅计数基线分别仅匹配420个和360个。这些结果并未确立候选方案的优越性。保留的失败案例包括九个额外的假通过（false-PASS）探针和120次未获资格的全树计时尝试。随后的通用绑定层拒绝了54个指定的不匹配项……（原文摘要至此截断）

    arXiv:2610.41276v1 Announce Type: new  Abstract: Architectural debt gates must distinguish an allowed change from an incompletely observed one. We study a decision contract separating observation qualification, prohibited relation identity, inherited debt and authorized baseline updates. Across 240 fixed files from four repositories, the current observer and programmatic reference jointly qualify 166 files (69.17%), all with exact reference-relative edge agreement; 74 remain reference-UNKNOWN. Coverage varies from 13/60 to 59/60 by repository. On 480 constructed state scenarios sharing 60 source anchors, both the candidate ratchet and mature dependency-cruiser baseline match every expected outcome, whereas static and count-only baselines match 420 and 360. These results do not establish candidate superiority. Preserved failures include nine additional false-PASS probes and 120 unqualified full-tree timing attempts. A subsequent generic binding layer rejects 54 specified mismatches, acc
    
[^12]: AI编程助手中的认知包容性：来自GitHub Copilot的证据

    Cognitive Inclusivity in AI Coding Assistants: Evidence from GitHub Copilot

    [https://arxiv.org/abs/2610.41192](https://arxiv.org/abs/2610.41192)

    该研究通过让20名学生使用GitHub Copilot完成代码生成、调试和重构任务的受控实验室实验，揭示了具有不同认知风格的开发者在使用AI编程助手时可能面临的认知包容性挑战。

    

    AI编程助手被广泛用于日常软件开发任务。尽管已有研究考察了它们对生产力和代码质量的影响，但人们对这些工具如何支持多样化的思维方式仍知之甚少。来自非AI软件工程工具的证据表明，工具设计与用户认知风格之间的不匹配可能带来包容性障碍，但目前尚不清楚AI编程助手是否也存在类似问题。我们研究了具有不同认知风格的开发者如何体验和使用这些工具，以及他们是否表现出认知包容性方面的挑战。我们开展了一项受控实验室研究，邀请20名学生参与者使用GitHub Copilot完成代码生成、调试和重构任务。通过出声思维协议、半结构化访谈和归纳式主题分析，我们识别了用户交互中的模式。随后，我们应用GenderMag框架……（原文摘要在此处截断）

    arXiv:2610.41192v1 Announce Type: new  Abstract: AI coding assistants are commonly used for daily software development tasks. While prior research has examined their impact on productivity and code quality, less is understood about how these tools support diverse ways of thinking. Evidence from non-AI software engineering tools suggests that mismatches between tool design and users' cognitive styles can introduce inclusivity barriers, but it remains unclear whether similar issues arise in AI coding assistants. We investigate how developers with different cognitive styles experience and interact with these tools and whether they exhibit cognitive inclusivity challenges. We conducted a controlled laboratory study with 20 student participants using GitHub Copilot to complete code generation, debugging, and refactoring tasks. Using think-aloud protocols, semi-structured interviews, and inductive thematic analysis, we identify patterns in user interaction. We then apply the GenderMag framew
    
[^13]: 谱系即记忆：短预算LLM引导优化中的父代继承与反馈披露

    Lineage Is Memory: Parent Inheritance and Feedback Disclosure in Short-Budget LLM-Guided Optimization

    [https://arxiv.org/abs/2610.41157](https://arxiv.org/abs/2610.41157)

    该研究通过1,750次预注册实验首次将父代继承与反馈披露效应分离，发现贪婪的父代继承——即谱系本身充当记忆——才是短预算LLM引导优化中绝大部分进展的真正来源，而非每轮反馈所覆盖的失败模式广度。

    

    迭代式语言模型优化器通过评估候选方案、向提议模型展示失败案例样本、并保留能提升性能的修订来改进提示词和程序。近期研究将这一循环的局限归因于狭窄的单轮反思，为此引入了全集错误聚类、针对性小批量或更大的提议模型，以便优化器在每一步能看到更多失败模式。然而，现有的所有评估都仅在保持谱系的搜索中测试这些机制——即每次修订都编辑一个存活的父代——这使得父代继承的贡献与每轮反馈的广度混淆在一起。在涵盖九个提示词与程序合成基准的1,750次预注册实验中，我们在匹配的评估预算下将父代继承与反馈披露分离开来。研究发现贪婪的父代继承推动了绝大多数优化进展，其表现优于种子重置……（原文摘要在此处截断）

    arXiv:2610.41157v1 Announce Type: cross  Abstract: Iterative language-model optimizers refine prompts and programs by evaluating a candidate, showing a proposer model a sample of failing cases, and keeping revisions that improve performance. Recent work attributes the limits of this loop to narrow single-round reflection, introducing full-set error clustering, targeted minibatches, or larger proposer models so the optimizer sees more failure modes per step. Yet every existing evaluation tests these mechanisms exclusively inside lineage-preserving search, where each revision edits a surviving parent, leaving the contribution of parent inheritance confounded with the breadth of per-round feedback. Across 1,750 pre-registered runs spanning nine prompt and program synthesis benchmarks, we separate parent inheritance from feedback disclosure under matched evaluation budgets. We find that greedy parent inheritance drives the vast majority of optimization progress, outperforming seed-reset re
    
[^14]: 一个判定结果是不够的：对学习型世界模型及其测试框架进行回归测试

    One Verdict Is Not Enough: Regression Testing Learned World Models and Their Test Harnesses

    [https://arxiv.org/abs/2610.41130](https://arxiv.org/abs/2610.41130)

    该论文提出了TetraState，一种针对学习型世界模型的回归测试协议，通过在四种固定的模型原生配置下运行成对的可能/不可能视频测试、单独报告无结论测试与失败执行、并将测试框架作为被测系统的一部分进行版本管理，从而解决了单一配置下测试结论不可靠以及测试框架故障导致错误判定的问题。

    

    学习型世界模型以软件组件的形式发布，其模型版本、预处理和测试框架会在不同版本发布之间发生变化。关于模型更新的已有研究已经统计了被准确率检查所掩盖的逐样本回归（即负向翻转）。但仍存在两个问题：一个测试可能在某一种配置下通过而在另一种配置下失败，并且测试框架的故障可能在不发生崩溃的情况下产生错误的分数和错误的判定。我们提出了 TetraState，一种针对成对的可能/不可能视频测试的回归测试协议。每个测试在四种固定的模型原生配置下运行；其正确判定的数量将其归入四种结果类别之一。无结论的测试和失败的执行与判定率分开报告。修订版本通过逐测试的结果转换进行比较，并且测试框架作为被测系统的一部分进行版本管理。在全部 506 个 IntPhys 2 Main 测试以及两个异构系统 Cosmos 3 Nano 和 V-JEPA 2 上，两个 V-JEPA 2

    arXiv:2610.41130v1 Announce Type: new  Abstract: Learned world models ship as software components whose model versions, preprocessing, and test harnesses change between releases. Work on model updates already counts per-sample regressions, or negative flips, that an accuracy check hides. Two problems remain: a test can pass under one configuration and fail under another, and a harness fault can produce erroneous scores and wrong verdicts without a crash. We present TetraState, a regression-testing protocol for paired possible/impossible video tests. Each test runs under four fixed model-native configurations; its number of correct verdicts assigns one of four outcome classes. Inconclusive tests and failed executions are reported separately alongside the verdict rate. Revisions are compared by per-test outcome transitions, and the harness is versioned as part of the system under test. On all 506 IntPhys 2 Main tests and two heterogeneous systems, Cosmos 3 Nano and V-JEPA 2, two V-JEPA 2
    
[^15]: RepGen：通过上下文感知的代码生成与验证复现深度学习缺陷

    RepGen: Reproducing Deep Learning Bugs through Context-Aware Code Generation and Verification

    [https://arxiv.org/abs/2610.41097](https://arxiv.org/abs/2610.41097)

    RepGen 是一个开源工具，通过探索代码库构建上下文、生成候选复现脚本，并利用静态检查与大语言模型反馈进行迭代验证，从而在缺陷报告不完整的情况下自动复现深度学习缺陷。

    

    复现深度学习（DL）缺陷极具挑战性，因为故障可能涉及源代码、模型、数据、依赖项和执行环境，而不完整的缺陷报告因遗漏关键细节进一步增加了复现难度。我们提出了 RepGen，这是一个通过上下文感知代码生成与迭代验证来复现深度学习缺陷的开源工具。给定一份缺陷报告和代码仓库，RepGen 会探索代码库以构建学习增强的上下文，制定复现计划，并生成候选脚本。它利用静态检查和大语言模型（LLM）反馈来改进候选脚本，从而提升其正确性与相关性。在启用相应功能时，该工具会准备执行环境、运行脚本，并将输出与报告的故障进行对比评估。RepGen 提供交互式命令行界面，支持本地和托管的大语言模型、可配置的执行预算以及可查看的产物，并附带用于自动化议题处理的 GitHub Action。

    arXiv:2610.41097v1 Announce Type: new  Abstract: Reproducing deep learning (DL) bugs is challenging because failures can involve source code, models, data, dependencies, and execution environments. Incomplete bug reports further complicate reproduction by omitting essential details. We present RepGen, an open-source tool for reproducing DL bugs through context-aware code generation and iterative verification. Given a bug report and repository, RepGen explores the codebase to build a learning-enhanced context, develop a reproduction plan, and generate candidate scripts. It refines candidates using static checks and LLM feedback to improve their correctness and relevance. When enabled, the tool prepares the execution environment, runs the script, and assesses its output against the reported failure. RepGen provides an interactive command-line interface supporting local and hosted LLMs, configurable execution budgets, and inspectable artifacts, along with a GitHub Action for automated iss
    
[^16]: AIFaultBench：AI软件系统中真实故障的可复现基准测试

    AIFaultBench: A Reproducible Benchmark of Real-World Faults in AI Software Systems

    [https://arxiv.org/abs/2610.41094](https://arxiv.org/abs/2610.41094)

    该论文提出了AIFaultBench，一个涵盖六类AI系统、包含来自105个软件仓库的770个真实故障的可复现基准测试，在文档化执行条件下成功复现了84.7%的故障，为AI软件调试研究提供了真实且可验证的故障资源。

    

    基于AI的软件系统仍然难以调试，因为其故障可能涉及代码、模型、数据、依赖项、硬件和执行环境之间的复杂交互。现有基准测试仅覆盖特定类别的AI系统，而对故障真实性和可复现性的担忧凸显了对文档化执行条件和证据的需求。我们提出了AIFaultBench，这是一个包含来自76个组织的105个基于AI的软件仓库中770个真实漏洞的基准测试。它涵盖六类AI系统：机器学习、深度学习、强化学习、基于大语言模型的系统、智能体AI以及AI工具与基础设施。每个漏洞记录将原始报告和受影响的代码库与结构化元数据、环境设置脚本、复现触发器和调查证据相连接。我们在记录的执行条件下成功复现了652个漏洞（84.7%），并记录了其余漏洞的结果。

    arXiv:2610.41094v1 Announce Type: new  Abstract: AI-based software systems remain difficult to debug because their failures can involve interactions among code, models, data, dependencies, hardware, and execution environments. Existing benchmarks cover specific classes of AI systems, while concerns about fault realism and reproducibility highlight the need for documented execution conditions and evidence. We present AIFaultBench, a benchmark of 770 real-world bugs from 105 AI-based software repositories across 76 organizations. It covers six classes of AI systems: machine learning, deep learning, reinforcement learning, systems based on large language models, agentic AI, and AI tooling and infrastructure. Each bug record connects the original report and affected codebase to structured metadata, environment setup scripts, a reproduction trigger, and investigation evidence. We reproduced 652 bugs (84.7%) under the recorded execution conditions and documented the outcomes of the remaining
    
[^17]: Solidity智能合约中Gas优化实践的实证研究

    An Empirical Study of Gas Optimization Practices in Solidity Smart Contracts

    [https://arxiv.org/abs/2610.40885](https://arxiv.org/abs/2610.40885)

    该论文通过对2018至2025年间GitHub上开源Solidity仓库中400个与gas相关提交的多阶段实证分析，构建了智能合约gas优化实践的分类体系，并揭示了理论优化模式与开发者真实实践之间的显著差距。

    

    在以太坊生态系统中，智能合约的执行受gas模型约束，每一项计算操作都会产生直接的经济成本。因此，gas效率是去中心化应用实现经济可持续性和广泛普及的关键要求。尽管低成本开发至关重要，但理论上的优化模式与开发者在实际项目中所采用的真实实践之间仍存在显著差距。本文对2018年至2025年间在GitHub上创建的开源Solidity仓库中的gas优化实践进行了实证研究。采用软件仓库挖掘方法，我们通过一个包含基于NLP的筛选和人工定性检查的多阶段流程，识别出与gas相关的提交数据集并进行了分析。基于对400个代表性提交的分析，我们得出了gas优化实践的实证分类体系。

    arXiv:2610.40885v1 Announce Type: new  Abstract: In the Ethereum ecosystem, smart contract execution is governed by a gas model where every computational operation incurs a direct economic cost. Consequently, gas efficiency is a critical requirement for the economic sustainability and widespread adoption of decentralized applications. Despite the importance of cost-effective development, there remains a significant gap between theoretical optimization patterns and the actual practices applied by developers in real-world projects. This paper presents an empirical study investigating gas optimization practices in open-source Solidity repositories created on GitHub between 2018 and 2025. Adopting a Mining Software Repositories approach, we analyzed a dataset of gas-related commits identified through a multi-stage pipeline involving NLP-based filtering and manual qualitative inspection. Based on the analysis of 400 representative commits, we derive an empirical taxonomy of gas optimization
    
[^18]: RiskyPy：面向模型校准的提示生成方法，用于LLM生成代码的安全性与功能性联合评估

    RiskyPy: Toward Model-Calibrated Prompt Generation for Joint Security and Functionality Evaluation of LLM-Generated Code

    [https://arxiv.org/abs/2610.40735](https://arxiv.org/abs/2610.40735)

    RiskyPy提出了一种自动化流水线，利用功能性与漏洞金丝雀生成与模型能力相校准的安全导向代码生成数据集，实现了对LLM生成代码安全性与功能性的联合评估。

    

    评估提升LLM代码生成安全性的方法，需要能够同时评估安全性与功能性、并与所研究模型能力相适配的数据集。然而，现有基准的构建成本高昂，且通常无法校准到特定的模型能力范围。我们提出了RiskyPy，这是一个自动化流水线，利用功能性与漏洞金丝雀（canaries）生成与模型能力水平相校准的面向安全的代码生成数据集。我们将RiskyPy应用于Python，涵盖2025年CWE Top 25中的18个CWE领域，生成了101个提示。在Qwen3-4B和Seed-Coder-8B上的实验表明，所生成的任务在功能上仍然可达成，其中57-72%的提示能够生成功能正常的代码，而仅有5-10%的生成结果不存在Bandit安全告警，从而为安全改进方法的安全性与功能性联合评估留下了空间。

    arXiv:2610.40735v1 Announce Type: cross  Abstract: Evaluating approaches that improve the security of LLM code generation requires datasets that jointly assess security and functionality and remain adapted to the capabilities of the models being studied. However, existing benchmarks are costly to produce and are generally not calibrated to a chosen range of model capabilities. We introduce RiskyPy, an automated pipeline that generates security-oriented code-generation datasets calibrated to model proficiency using functionality and vulnerability canaries. We instantiate RiskyPy for Python using domains covering 18 CWEs from the 2025 CWE Top 25, producing 101 prompts. Experiments on Qwen3-4B and Seed-Coder-8B show that the generated tasks remain functionally accessible, with 57-72% of prompts yielding functional code, while only 5-10% of generations are free of Bandit findings, leaving room for joint evaluation of security and functionality of security improvement approaches.
    
[^19]: MetaVul：面向漏洞检测的智能体框架自动发现方法

    MetaVul: Agentic Harness Discovery for Vulnerability Detection

    [https://arxiv.org/abs/2610.40721](https://arxiv.org/abs/2610.40721)

    MetaVul提出了一个用于漏洞检测的自我改进框架发现机制，通过基于工作流的优化循环，利用成对的漏洞函数与修复函数的反馈迭代优化LLM智能体框架，克服了传统固定手工编写的框架难以自适应评估反馈、且改进难以跨数据集和模型迁移的局限。

    

    大语言模型（LLM）智能体正被越来越多地用于自动化漏洞检测，但可靠的检测仍然极具挑战性，因为智能体必须从良性的代码变化中辨别出细微的、与安全相关的修复。尽管底层LLM的能力已经显著提升，但智能体的准确性仍在很大程度上取决于框架——即围绕LLM定义脚手架结构的代码。这类框架通常是固定不变的，且大多由人工编写，限制了其根据评估器反馈进行自适应调整的能力。此外，当框架是静态的时候，在某一数据集或模型上取得的改进可能无法迁移到其他数据集或模型上。为了解决这些挑战，我们提出了MetaVul，一个用于漏洞检测中框架发现的自我改进框架。MetaVul从一个种子结构化推理脚手架出发，利用基于工作流的优化循环，借助成对的漏洞函数与修复函数所产生的反馈来迭代地优化框架，同时保持结构化推理。

    arXiv:2610.40721v1 Announce Type: cross  Abstract: LLM agents are increasingly used to automate vulnerability detection, but reliable detection is challenging because agents must distinguish subtle, security-relevant fixes from benign code variation. While underlying LLM capability has improved significantly, the accuracy of the agent still depends heavily on the harness: the code that defines the scaffolding around the LLM. Harnesses are typically fixed, and largely hand-written, limiting adaptation to evaluator feedback. Additionally, when harnesses are static, improvements on one dataset or model may not transfer to others. To address these challenges, we introduce MetaVul, a self-improving framework for harness discovery in vulnerability detection. Starting from a seed structured reasoning scaffold, MetaVul uses a workflow-based optimization loop to iteratively refine the harness using feedback from paired vulnerable and patched functions, while preserving the structured reasoning 
    
[^20]: 面向常规H&E切片的小型HER2判读器：真实世界部署

    A small HER2 reader for routine H&E slides: real-world deployment

    [https://arxiv.org/abs/2610.40662](https://arxiv.org/abs/2610.40662)

    研究者构建了一个仅2200万参数的小型HER2判读器，可直接从常规H&E染色切片预测HER2状态，在外部验证中位列7个方法之首（AUC 0.812），性能与体积大4倍和50倍的编码器持平，并已实现真实世界部署，可作为病理医生评估HER2状态的基线工具。

    

    HER2状态决定了哪些患者接受HER2靶向治疗。病理医生通过免疫组化（IHC）对HER2蛋白进行评分；而IHC 2+的临界病例——这是最难判断的情况——则需等待原位杂交（ISH）检测结果。我们构建了一个可直接读取常规H&E切片的HER2判读器，实现端到端自主可控，作为病理医生的基线工具。该模型采用2200万参数的ViT-S/16切片块编码器，在公开的TCGA切片上通过自监督学习、从H-optimus-0蒸馏以及两轮JEPA精炼进行训练；五个经交叉验证的基于注意力机制的切片分类器对其预测概率取平均；并在训练数据上固定了操作点。在一项配对的十种子协议下进行比较，基于我们编码器的判读器在从TCGA-BRCA到Yale-HER2的外部验证中位列7个方法之首（AUC 0.812），与基于4倍和50倍更大编码器的判读器性能持平。我们还在公开队列上重建了PathBench的编码器基准；我们的复现结果再现了其已发表的参考编码器排序，且在此基准下我们的编码器……

    arXiv:2610.40662v1 Announce Type: cross  Abstract: HER2 status decides who receives HER2-targeted therapy. Pathologists score HER2 protein by immunohistochemistry (IHC); equivocal IHC 2+ tumours, their hardest calls, wait for in situ hybridisation (ISH). We built a HER2 reader for routine H&E slides, owned end to end, as a baseline for pathologists. A 22M-parameter ViT-S/16 tile encoder was trained on public TCGA slides by self-supervision, distillation from H-optimus-0 and two rounds of JEPA refinement; five cross-validated attention-based slide classifiers average their probabilities; and an operating point is fixed on training data. Compared under one paired ten-seed protocol, readers on our encoder rank first of 7 on external validation from TCGA-BRCA to Yale-HER2 (AUC 0.812), level with readers on encoders 4 and 50 times larger. We also rebuilt PathBench's encoder benchmark on public cohorts; our replica reproduces its published order of reference encoders, and under it our encode
    
[^21]: PR4Code：一个面向AI代码生成的Pull Request数据集

    PR4Code: A Pull Requests Dataset for AI Code Generation

    [https://arxiv.org/abs/2610.39890](https://arxiv.org/abs/2610.39890)

    本文提出了PR4Code数据集，收录了从GitHub收集的4,508个Java和8,831个Python精选Pull Request（含元数据、提交历史和代码变更），用于在完整软件功能层面支持AI代码生成的研究与评估。

    

    AI驱动的代码生成正在从处理小型、局部的代码补全任务向支持完整软件功能实现的方向演进。为支持这一转变，本文提出了PR4Code，这是一个从GitHub收集的精选Pull Request（PR）数据集，包含4,508个Java和8,831个Python的PR，每个PR都附有元数据、提交历史和详细的代码变更。与专注于孤立代码片段的资源不同，PR4Code捕捉了功能实现的完整过程，提供了大量多样化的真实开发任务集合，可用于评估AI驱动的解决方案。我们的分析揭示了PR结构、提交粒度和文本内容方面的显著差异。此外，我们还发布了用于重新生成和更新数据集的脚本，以确保数据集的可重现性和可维护性。

    arXiv:2610.39890v2 Announce Type: replace  Abstract: AI-driven code generation is evolving from handling small, localized code-completion tasks toward supporting the implementation of full software features. To support this transition, this paper introduces PR4Code, a dataset of 4,508 Java and 8,831 Python curated Pull Requests (PRs) collected from GitHub, each enriched with metadata, commit histories, and detailed code changes. Unlike resources focused on isolated code fragments, PR4Code captures the feature implementation process, delivering a large and diverse collection of real-world development tasks that can be used to assess AI-driven solutions. Our analysis reveals substantial variability in PR structure, commit granularity, and textual content. In addition, we release scripts to regenerate and update the dataset, ensuring reproducibility and maintainability.
    
[^22]: 使用编程智能体从GPU迁移到TPU：什么有效、什么失败、代价几何？

    From GPU to TPU with Coding Agents: What Works, What Fails, and What It Costs?

    [https://arxiv.org/abs/2610.34422](https://arxiv.org/abs/2610.34422)

    该研究构建了148个PyTorch研究项目的数据集，并使用Claude Code将其中10个项目从NVIDIA GPU迁移到Google Cloud TPU，通过独立评估发现迁移平均需2.4次人工干预、76%的执行尝试失败且近半数失败不报错，首次系统量化了智能体驱动代码迁移的可信度与成本。

    

    一个编程智能体可以迁移一个机器学习工作负载并报告成功，但仍然会改变模型的训练方式。不断上涨的计算成本、有限的加速器容量以及对供应商的依赖，正促使各组织寻求图形处理器（GPU）的替代方案；一个关键问题是，由智能体生成的向张量处理单元（TPU）的迁移是否可信、其代价如何。我们构建了一个包含148个PyTorch机器学习研究项目的数据集，并使用Claude Code将其中十个项目从NVIDIA GPU迁移到Google Cloud TPU。我们独立评估每次迁移是否能够运行、是否保持数值行为、是否在预期硬件上执行、是否维持项目目标，以及是否提供可接受的性能。我们还考察了失败情况、人工投入与成本。八个项目满足我们的项目目标标准，迁移平均需要2.4次人工干预。在70次执行尝试中，53次失败（76%），其中26次失败（49%）没有引发任何错误信息。

    arXiv:2610.34422v1 Announce Type: new  Abstract: A coding agent can migrate a machine learning workload, report success, and still change how the model trains. Rising compute costs, limited accelerator capacity, and vendor dependence are pushing organizations toward alternatives to graphics processing units (GPUs); a key question is whether agent-generated migrations to tensor processing units (TPUs) can be trusted and at what cost. We build a dataset of 148 PyTorch ML research projects and use Claude Code to migrate ten of them from NVIDIA GPUs to Google Cloud TPUs. We independently assess whether each migration runs, preserves numerical behavior, executes on the intended hardware, maintains the project's goal, and delivers acceptable performance. We also examine failures, human effort, and costs. Eight projects meet our project-goal criterion, and migrations require 2.4 human interventions on average. Of 70 execution attempts, 53 fail (76%), and 26 of these failures (49%) raise no er
    
[^23]: 智能体代码审查能否在合并前捕获漏洞？

    Can Agentic Code Review Catch Vulnerabilities Before They Are Merged?

    [https://arxiv.org/abs/2610.33509](https://arxiv.org/abs/2610.33509)

    提出了SecPRBench——首个基于经过验证的真实漏洞引入提交来评估智能体安全代码审查的综合性基准，涵盖Firefox、V8、OpenSSL等九个关键系统的265个任务，并发现八个前沿模型的召回率仅为32.5%至78.1%。

    

    LLM智能体已被越来越多地用于以安全为核心的代码审查，但现有基准大多从后续的修复补丁中提取被审查的代码变更，而非真实的漏洞引入提交。在本文中，我们提出了SecPRBench，这是首个用于在经过验证的真实世界代码变更上评估智能体安全审查的综合性基准。更具体地说，我们构建了一个自动化智能体流水线，能够自主收集、构建并验证面向热门软件系统的大量真实漏洞引入提交。为确保评估的精确性，我们还采用LLM-as-a-judge（大语言模型作为裁判）对与真实漏洞匹配的发现给予评分。迄今为止，我们已从九个关键系统（包括拥有数百万行代码的Firefox、V8和OpenSSL）中收集了265个任务。在八个前沿模型上的评估显示，召回率介于32.5%至78.1%之间，其中Grok 4.6和GPT-6 sol位居前列。我们的研究还表明，大多数漏检情况并

    arXiv:2610.33509v1 Announce Type: cross  Abstract: LLM agents have been increasingly adopted for security-focused code review, but existing benchmarks mostly derive the reviewed changes from their later fixes instead of the real-world bug-introducing commits. In this paper, we present SecPRBench, the first comprehensive benchmark for evaluating agentic security review on validated real-world code changes. More specifically, we have built an automated agentic pipeline to autonomously harvest, build, and validate massive real-world bug-introducing commits for popular software systems. To ensure precise evaluation, we also adopt LLM-as-a-judge to credit the findings matching ground-truth vulnerabilities. To date, we have collected 265 tasks from nine critical systems, including Firefox, V8, and OpenSSL with millions of lines of code. Across eight frontier models, recall ranges from 32.5% to 78.1%, with Grok 4.6 and GPT-6 sol ranking at the top. Our study also shows that most misses are no
    
[^24]: 超智能软件工程：随能力同步扩展的治理

    Superintelligent Software Engineering: Governance That Scales with Capability

    [https://arxiv.org/abs/2610.31306](https://arxiv.org/abs/2610.31306)

    本文提出“超智能软件工程”概念，主张当AI智能体能力超越人类工程师时，必须让治理机制（门禁、契约、决策与衰减控制）随能力同步扩展，才能应对知识衰减与人工审查失效，保障软件正确性。

    

    软件智能体正在被训练以超越人类工程师；我们认为，仅有能力并不能产出更好的软件工程。我们将超智能软件工程定义为一门学科，即使用能力超越人类工程师的智能体来构建和演进软件，并使治理（门禁、契约、决策与衰减控制）随能力同步扩展。该论证建立在三个论点之上：第一，项目的正确性依赖于代码本身所不承载的决策与契约；第二，知识会衰减；第三，一旦智能体的进步速度超过其审查者，通过人工阅读所能捕获的、基于过时知识构建的变更比例将不断缩小。一个简单的模型表明，在审查容量固定的情况下，逃逸的变更随吞吐量线性增长，而阅读所能捕获的比例则趋近于零；治理能够降低这一斜率，而仅凭能力则无法做到。针对GitHub上星标最多的452个公共仓库的探索性快照（原文摘要在此处截断）。

    arXiv:2610.31306v1 Announce Type: new  Abstract: Software agents are being trained to exceed human engineers; we argue that capability alone does not produce better software engineering. We define superintelligent software engineering as the discipline of building and evolving software with agents whose capability exceeds that of human engineers, in which governance (gates, contracts, decisions and decay control) scales with capability. The argument rests on three claims. First, correctness in a project rests on decisions and contracts that the code does not carry. Second, knowledge decays. Third, once an agent outpaces its reviewers, reading catches a shrinking share of the changes built on stale knowledge. A simple model shows that, past a fixed review capacity, escaping changes grow linearly with throughput while the share that reading catches falls toward zero; governance can lower the slope, capability alone cannot. An exploratory snapshot of 452 of the most-starred public GitHub 
    
[^25]: CodeLift：你的测试套件其实是一套密集的软件工程训练课程

    CodeLift: Your Test Suite Is Secretly a Dense SWE Curriculum

    [https://arxiv.org/abs/2610.17224](https://arxiv.org/abs/2610.17224)

    CodeLift通过在每次迭代中对测试与补丁进行分组、并给少数补丁能通过的测试组赋予更高权重，动态校准测试通过率奖励，把测试套件转化为自适应课程，为软件工程智能体的强化学习提供更可靠的密集奖励信号。

    

    “全部测试通过”的二值奖励已成为训练软件工程智能体的标准做法，但其稀疏性抹去了部分正确补丁之间的差异。测试通过率是一种天然的替代方案，然而我们发现它并不能可靠地反映补丁质量：一个被测试的功能既可以由一个具有挑战性的测试覆盖，也可以由一百个简单的测试覆盖，而其中大多数可能早已是已被满足的“通过-通过”测试。我们提出了CodeLift，一种强化学习方法，它动态校准通过率奖励，将测试套件转变为自适应课程。在每次迭代中，CodeLift将“杀死”相同补丁的测试归为一组、将通过相同测试的补丁归为一组，然后为通过补丁组更少的测试组赋予更高的权重。这降低了大量简单测试的影响，使训练更聚焦于仍然具有挑战性的内容。在每个rollout内，CodeLift对补丁应用相同的评分规则（原文在此处截断）。

    arXiv:2610.17224v2 Announce Type: replace  Abstract: The pass-all-test binary reward has become a standard recipe for training software engineering agents, but its sparsity eliminates difference of partially correct patches. The test pass-rate is a natural alternative. However, we find that it is not a reliable indicator of patch quality. A functionality being tested can be covered by either one challenging test or one hundred trivial tests, where the majority might be already satisfied pass-to-pass tests. We introduce CodeLift, a reinforcement learning method that dynamically calibrates pass-rate rewards, turning the test suite into an adaptive curriculum. At each iteration, CodeLift groups tests that kill the same patches and patches that pass the same tests, then gives more weight to test groups that fewer patch groups pass. This reduces the impact of majority simple tests and focuses more on what is still challenging. Within each rollout, CodeLift applies the same scoring rule to r
    
[^26]: 实践中的交付一致性度量：来自多平台发布场景的DORA框架扩展

    Measuring Delivery Consistency in Practice: A DORA Extension from a Multi-Platform Release Setting

    [https://arxiv.org/abs/2606.00364](https://arxiv.org/abs/2606.00364)

    该论文提出“交付一致性”（DC）这一有界二阶矩度量指标，弥补了DORA指标无法反映部署节奏规律性的不足，并将其整合进包含八种原型的“交付健康矩阵”诊断框架，在多平台软件交付环境中进行了实践评估。

    

    DevOps研究与评估（DORA）框架是工程团队绩效测量中采用最广泛的度量体系。然而，每一项DORA指标都是一阶矩统计量或简单的比值，这限制了其对工程流程的潜在洞察。例如，部署频率等指标无法反映部署时间分布的形态，因此度量值相同的团队，既可能以节拍器般规律的节奏进行部署，也可能以不合意的爆发式节奏进行部署。我们一直在开发并试点“交付一致性”，这是一种有界的二阶矩度量，通过发布间隔的变异系数推导得出，用于衡量交付节奏的规律性。结合其他DORA概念，我们将DC整合进“交付健康矩阵”中——这是一个包含八种原型的诊断工具，可将联合读数映射为差异化的干预措施。我们报告了在一个四平台软件交付团队上进行的一项经验评估。

    arXiv:2606.00364v2 Announce Type: replace  Abstract: The DevOps Research and Assessment (DORA) framework is the most widely adopted measurement system for performance measurement across engineering teams. However, every DORA metric is a first-moment statistic or a simple ratio, which limits the potential insights into engineering process. For example, metrics like Deployment Frequency do not capture the distributional shape of deployment timing, so teams with identical measures can deploy on a metronomic cadence or in undesirably erratic bursts. We have been developing and piloting Delivery Consistency (DC), a bounded second-moment measure of cadence regularity derived from the coefficient of variation of inter-release intervals. In conjunction with other DORA concepts, we integrated DC into the Delivery Health Matrix, an eight-archetype diagnostic that maps joint readings to differentiated interventions. We report an experience evaluation on a four-platform software delivery group usi
    
[^27]: 智能体化代码审查中的人机协同

    Human-AI Synergy in Agentic Code Review

    [https://arxiv.org/abs/2603.15911](https://arxiv.org/abs/2603.15911)

    该论文通过对300个开源项目中278,790次代码审查对话的大规模实证分析，比较了人类审查者与AI智能体的反馈差异，揭示了人机协作模式及交互方式如何影响代码审查结果与建议采纳率。

    

    代码审查是一项关键的软件工程实践，开发人员在代码集成之前审查代码变更，以确保代码质量、检测缺陷并提升可维护性。近年来，能够理解代码上下文、规划审查行动并与开发环境交互的AI智能体被越来越多地集成到代码审查流程中。然而，目前缺乏实证证据来比较AI智能体与人类审查者在协作工作流中的有效性。为了填补这一空白，我们对300个开源GitHub项目中的278,790次代码审查对话进行了大规模实证分析。在本研究中，我们旨在比较人类审查者与AI智能体所提供的反馈差异。我们研究了审查对话中的人机协作模式，以理解交互如何塑造审查结果。此外，我们分析了人类审查者和AI智能体所提供的代码建议的采纳情况。

    arXiv:2603.15911v2 Announce Type: replace  Abstract: Code review is a critical software engineering practice where developers review code changes before integration to ensure code quality, detect defects, and improve maintainability. In recent years, AI agents that can understand code context, plan review actions, and interact with development environments have been increasingly integrated into the code review process. However, there is limited empirical evidence to compare the effectiveness of AI agents and human reviewers in collaborative workflows. To address this gap, we conduct a large-scale empirical analysis of 278,790 code review conversations across 300 open-source GitHub projects. In our study, we aim to compare the feedback differences provided by human reviewers and AI agents. We investigate human-AI collaboration patterns in review conversations to understand how interaction shapes review outcomes. Moreover, we analyze the adoption of code suggestions provided by human rev
    
[^28]: 面向BPMN模型生成的开源权重语言模型指令微调

    Instruction-Tuning Open-Weight Language Models for BPMN Model Generation

    [https://arxiv.org/abs/2512.12063](https://arxiv.org/abs/2512.12063)

    该论文提出InstruBPM方法，通过成对的文本-图表数据对开源大语言模型进行指令微调，并结合参数高效微调与量化技术，使其能够以低成本、保护隐私的本地化部署方式从自然语言描述直接生成高质量的BPMN流程模型。

    

    领域模型是软件工程的核心，它们能够促进共享理解、指导实现，并支持自动化分析和模型驱动开发。然而，尽管有这些好处，从业者往往因为建模耗时且需要稀缺的专业知识而跳过建模。为解决这一障碍，我们研究了经过指令微调适配的开源权重大语言模型，能否以经济高效且保护隐私的方式直接从自然语言描述生成高质量的BPMN流程模型。我们提出了InstruBPM，这是一种可复现的方法，它准备成对的文本-图表数据，并通过参数高效微调和量化技术对开源大语言模型进行指令微调，以实现本地化部署。我们从互补的角度评估微调后的模型：（i）使用BLEU、ROUGE-L和METEOR衡量文本/代码相似度，（ii）使用相对图编辑距离衡量结构保真度。

    arXiv:2512.12063v2 Announce Type: replace  Abstract: Domain models are central to software engineering, as they enable a shared understanding, guide implementation, and support automated analyses and model-driven development. Yet, despite these benefits, practitioners often skip modeling because it is time-consuming and demands scarce expertise. We address this barrier by investigating whether open-weight large language models, adapted via instruction tuning, can generate high-quality BPMN process models directly from natural language descriptions in a cost-effective and privacy-preserving way. We introduce InstruBPM, a reproducible approach that prepares paired text-diagram data and instruction tunes an open source large language model with parameter-efficient fine-tuning and quantization for on-prem deployment. We evaluate the tuned model through complementary perspectives: (i) text/code similarity using BLEU, ROUGE-L, and METEOR, (ii) structural fidelity using Relative Graph Edit Di
    

