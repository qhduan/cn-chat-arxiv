# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Vulnerable Code Search: Transferable Attack for Code Language Models](https://arxiv.org/abs/2608.26031) | 本文提出了一种与编程语言无关且可迁移的对抗性攻击，通过扰动代码标识符而不改变功能，有效降低了代码检索模型的性能，并展示了从小模型到大型闭源模型的可迁移性。 |
| [^2] | [Praxist: From Experimental Artifacts to Solution Lineages](https://arxiv.org/abs/2608.25955) | Praxist通过谱系中心方法将实验人工制品转化为类型化证据图，分离局部构建与证据综合，使研发代理能继承已验证机制，避免重复学习并提升长期研发效率。 |
| [^3] | [XREPOTEST: Benchmarking Multilingual Repository-Level Unit Test Generation for Large Language Models](https://arxiv.org/abs/2608.25939) | 本文提出了XREPOTEST，一个涵盖五种语言的多语言仓库级单元测试基准，并通过新指标调用率揭示LLM在现实仓库场景下与独立设置间存在显著性能差距。 |
| [^4] | [Repair or Resample? Rethinking Failure Debugging in LLM Multi-Agent Systems](https://arxiv.org/abs/2608.25920) | 本文提出SymTrace框架，通过受控评估区分MAS修复中的因果修复与随机重采样修复，并构建SymFail数据集以系统验证修复方法的有效性。 |
| [^5] | [Answer Is Cheap, Show Me the Evidence! Augmenting Automated Vulnerability Assessment with Evidence](https://arxiv.org/abs/2608.25905) | EAVA框架利用大型语言模型和专门智能体，通过两阶段训练（监督微调与强化学习）来评估软件漏洞，并生成支持性证据，以弥补现有方法忽视富文本信息和缺乏解释的不足。 |
| [^6] | [Beyond the Editing Canvas: Evidence Divergence in OOXML-to-LLM Ingestion](https://arxiv.org/abs/2608.25880) | 本论文揭示OOXML文件在Office和LLM提取间存在“证据分叉”，导致同一文件产生不同证据视图，破坏语义完整性，并系统确认了21个此类构造。 |
| [^7] | [Closing the Gap: Automated Discovery of Secure Dockerfile Reference Standards via Semantic Clustering in Enterprise Inner Source](https://arxiv.org/abs/2608.25793) | 提出了一种基于语义聚类的自动化流水线，用于在企业内部源代码中发现安全Dockerfile参考标准，并揭示了99%的文件存在安全配置错误的系统性缺陷。 |
| [^8] | [Predicting Struggling Students in CS1 Programming Using Keystroke-Level Editing Features](https://arxiv.org/abs/2608.25769) | 本文提出利用击键级编辑特征来早期预测CS1编程课程中可能完全卡住的学生，从而为教师提供及时干预的机会。 |
| [^9] | [From General Agents to RCA Experts: A Self-Evolving Harness for Root Cause Analysis](https://arxiv.org/abs/2608.25661) | 本文提出了一种自进化的外部控制框架，通过复用通用代理的强能力而非重建专用代理，显著提升根因分析准确性并随经验积累持续优化。 |
| [^10] | [Narcissus: Program Synthesis Using Context-Aware LLM Approximations](https://arxiv.org/abs/2608.25657) | 纳西索斯通过将大语言模型提议保留为语法树并在候选程序上下文中评分扩展，结合正则化确保规则可达，从而在目标语言罕见时实现鲁棒的程序合成。 |
| [^11] | [DBcover: A White-box SQL Test Generation Framework for Coverage Improvement](https://arxiv.org/abs/2608.25573) | DBcover提出了一种基于大语言模型和白盒上下文推理的SQL测试生成框架，通过知识图谱整合全局与局部上下文，实现RDBMS覆盖率提升。 |
| [^12] | [A Hybrid Usability Approach for Rating Evaluation of M-Commerce Applications](https://arxiv.org/abs/2608.25550) | 本文提出了一种混合可用性模型，通过整合八个关键因素来预测移动商务应用的评分，并基于五个实际应用进行了评估。 |
| [^13] | [A Programming Paradigm for Spatiotemporal Composability](https://arxiv.org/abs/2608.25512) | 该论文提出了一种编程范式，通过将效果和协效果提升为运行时机制，形式化了可逆效果和响应式协效果，分别实现时间可组合性（完全撤销副作用）和空间可组合性（响应式管理依赖）。 |
| [^14] | [Separating Disclosure from Authorization: Field-Tier Minimization for Agent Action Mediation](https://arxiv.org/abs/2608.25474) | 本文提出了一种字段级最小化方法，将操作参数分为三个层级，通过先计算完整参数的规范摘要再最小化，实现了披露与授权的分离，且不影响历史账本的有效性。 |
| [^15] | [RotDroid: Cross-Orientation State Equivalence Testing for Detecting GUI Rotation Bugs in Android Apps](https://arxiv.org/abs/2608.25425) | RotDroid通过生成状态保持动作序列和微调视觉语言模型RotVL，实现了跨方向GUI状态等价性检查，从而有效检测Android应用中的旋转缺陷。 |
| [^16] | [Retry Amplification in Distributed Systems: A Systematic Analysis of Retry Policies and Their Role in Cascading Failures](https://arxiv.org/abs/2608.25403) | 本文提出重试放大因子（RAF）指标，通过分析200个开源项目发现重试策略普遍缺乏退避机制，且简单的标准重试策略在相关故障下会显著降低系统成功率，从而揭示分布式系统中重试策略的集体行为可能加剧级联故障。 |
| [^17] | [Point-in-Time Audit Before Alpha: Public-Archive Availability and a Negative Matched-Budget Study on BTC Perpetual Futures](https://arxiv.org/abs/2608.25348) | 本文提出一种时点审计方法，验证公共加密货币数据在决策时的可用性，并发现未平仓量数据存在发布时间不可验证问题，最终通过修订核心数据流和保留划分实现有效审计。 |
| [^18] | [Metis: Typed Runtime Mediation for Tool-Using Software Agents](https://arxiv.org/abs/2608.25322) | 本文提出Metis，一种通过类型化事件流实现显式权限决策和干扰分类的运行时中介机制，显著降低了工具使用代理的延迟并增强了安全性。 |
| [^19] | [A Few Pages of Markdown: Committed AI Configuration and Lower Quality Cost after Coding-Agent Adoption](https://arxiv.org/abs/2608.25241) | 本文提出 RAMP 成熟度模型，证明编码代理采用能加速开发（提交量增加 28-38%），同时通过版本控制的 AI 配置降低质量成本，且多数配置为一次性设置。 |
| [^20] | [SPECMINE: A Large-Scale Corpus of Spec-Driven Development Artifacts](https://arxiv.org/abs/2608.25202) | 我们提出了SPECMINE，这是首个大规模语料库，通过两次普查系统地捕捉了GitHub上规范驱动开发工件，为研究规范如何转化为代码提供了基础数据。 |
| [^21] | [Model-Based Agentic Software Engineering](https://arxiv.org/abs/2608.25174) | MAGE通过外部化最小有目的的表征并利用约束赋予义务权威，解决了智能体软件工程中的表征和权威问题，从而从普通智能构建可信赖的自主性。 |
| [^22] | [FuzzingBrain-Bench V1: Evaluating Open-Ended Bug Discovery by LLMs](https://arxiv.org/abs/2608.25158) | 该基准测试通过开放式崩溃发现而非预定义目标来评估LLM的漏洞发现能力，强调真实能力评估。 |
| [^23] | [ARISMA: Guidelines for AI- and LLM-Assisted Systematic Reviews, Scoping Reviews, and Mapping Studies](https://arxiv.org/abs/2608.25050) | ARISMA提出了一套首个端到端指南，为AI和LLM辅助的系统综述提供方法论适当性、验证、人类主导决策和可审计报告的标准化框架。 |
| [^24] | [FrontierChallenge: Evaluating Scientific Workflow Completion](https://arxiv.org/abs/2608.24979) | 本文介绍了FrontierChallenge基准测试，用于评估科学智能体在跨领域端到端工作流中的完成能力，发现当前最佳模型仅能完成20.6%的任务，表明部分进展难以转化为完整交付物。 |
| [^25] | [Evaluating and Preventing Security Smells in AI-Generated Ansible Code](https://arxiv.org/abs/2608.24962) | 本文评估了16个AI模型生成的Ansible代码，发现所有模型在无指导时均产生安全缺陷，并提出了一种基于扩展CO-STAR框架的提示集成方法，能在合成阶段预防这些缺陷，使部分模型生成合规代码。 |
| [^26] | [ToolMinimize: Auditing and Rewriting LLM Agent Tool Calls to Minimize Privacy Exposure](https://arxiv.org/abs/2608.24957) | 本文提出ToolMinimize，一种能够自动重写LLM代理工具调用参数以最小化隐私暴露的中间件，有效减少不必要的隐私数据共享。 |
| [^27] | [The Evolution of Binary Decompilation in the Modern Era: A Taxonomy, Literature Review, and Future Perspectives](https://arxiv.org/abs/2608.24955) | 本文系统综述了现代反编译研究，提出分类法并指出基准缺失等挑战，展望了未来方向。 |
| [^28] | [Secret MCP: Evidence-Bounded and Context-Isolated Design Specification Generation from Web Screenshots](https://arxiv.org/abs/2608.24944) | 秘密MCP是一个开源本地系统，通过分离检索、证据准备和模型调用，为每个网页截图生成有证据约束和上下文隔离的可审计设计规格，防止参考间污染。 |
| [^29] | [From Blind Edits to Verified Repair: Building Trustworthy User-Side LLM Agents for Web Accessibility](https://arxiv.org/abs/2608.24913) | 本文提出并评估了一个保护隐私的浏览器代理，通过本地LLM生成可逆CSS修复网页无障碍问题，并引入双条件协议，发现未经验证的修复在改进和回退上效果相近，强调了验证修复的必要性。 |
| [^30] | [Confidently Wrong, Silently So: Auditing Undetectable Failures of a Deployed On-Device Language Model](https://arxiv.org/abs/2608.23663) | 该论文审计了设备端语言模型，发现其存在任务不对称的校准错误，即在错误前提上高置信虚构且对良性输入过度拒绝，且自报置信度不可靠，导致失败难以检测。 |
| [^31] | [Rebuild Dossier: Mechanically-Enforced Specs for Agentic App Rebuilds, and What Model-Tier Failures Reveal](https://arxiv.org/abs/2608.23616) | 该论文提出了rebuild-dossier工具，通过预先锁定应用接口并强制执行逐测试构建，揭示了测试可被操纵时通过套件不能保证正确性，且在较大应用上不如简单指令方法。 |
| [^32] | [Neuro-Formal Verification: Agentic Language-Agnostic Formal Program Reasoning](https://arxiv.org/abs/2608.21516) | 本文提出神经形式验证（NFV），通过AI智能体将主流语言代码翻译为形式化验证语言，实现一键式经验性准确验证，无需形式化方法专业知识。 |
| [^33] | [AI with Authority, from Application to Silicon](https://arxiv.org/abs/2608.21356) | 本文展示了生成式AI如何通过机器验证成为不可腐蚀的裁判，使得单人在五周内指挥AI代理从应用代码到RISC-V硅片流片，全程无需人工审查或编写RTL，提出了盐方法这一新纪律。 |
| [^34] | [Understanding the Architecture of Coding Agents: An Exploratory Study Using a Research Prototype](https://arxiv.org/abs/2608.10934) | 本文首次系统化描述了编码代理的架构组件，并开发了极简开源代理Ark和轻量级基准ArkBench，验证了其有效性。 |
| [^35] | [SIGIL: Compiling Agent Skills into Typed Harnesses](https://arxiv.org/abs/2607.27309) | SIGIL通过技能编译范式，将自然语言技能确定性编译为可执行代码，显著提升了智能体在执行过程中对技能要求的合规性。 |
| [^36] | [ClayBuddy: A Framework, Evaluation, & Mitigation of Coding Agent Failures](https://arxiv.org/abs/2606.19380) | ClayBuddy通过分解编码代理失败为三种机制并提出框架修改，实现了针对性的安全缓解，显著提升了AI代理在软件工程中的可靠性。 |
| [^37] | [Algorithmic algorithm development with LLMs: A Case Study on LLM-Usage for Contraction Order Optimization in Tensor Networks](https://arxiv.org/abs/2606.01975) | 本研究通过张量网络收缩顺序优化的案例，展示了LLM驱动的进化编码代理在算法开发中的潜力，并强调了人类科学家在评估与验证中的关键作用。 |
| [^38] | [ReLog: Execution-Aware Logging with Runtime Feedback for LLM-Oriented Debugging](https://arxiv.org/abs/2603.29122) | ReLog提出了一种由运行时反馈驱动的迭代日志生成框架，利用LLM动态优化日志语句以提升下游任务的实际效用，而非仅追求与人类编写的日志相似。 |
| [^39] | [When Do Reactive Notebooks Fail to React?](https://arxiv.org/abs/2511.21994) | 本文提出Rex测试套件，评估三个反应式笔记本系统（Ipyflow、Marimo和Observable）的反应性缺陷，发现它们在不同定义下均存在简单修改可破坏系统的不一致问题。 |
| [^40] | [QLCoder: A Query Synthesizer For Static Analysis of Security Vulnerabilities](https://arxiv.org/abs/2511.08462) | QLCoder是一个基于大语言模型的智能体框架，通过结合执行反馈、语言服务器协议和RAG数据库，能自动从CVE元数据生成有效的CodeQL安全查询，无需人工编写。 |
| [^41] | [Scalable Supervision for Software Agents via Patch Reasoning](https://arxiv.org/abs/2510.22775) | 本文提出R4P方法，通过补丁间推理和分组训练目标提供无需测试执行的密集奖励，实现软件代理监督的可扩展，并训练出无执行脚手架Mini-SE，显著提升性能。 |

# 详细

[^1]: 脆弱代码搜索：针对代码语言模型的可迁移攻击

    Vulnerable Code Search: Transferable Attack for Code Language Models

    [https://arxiv.org/abs/2608.26031](https://arxiv.org/abs/2608.26031)

    本文提出了一种与编程语言无关且可迁移的对抗性攻击，通过扰动代码标识符而不改变功能，有效降低了代码检索模型的性能，并展示了从小模型到大型闭源模型的可迁移性。

    

    可靠的代码检索对于开发者生产力和有效的代码复用至关重要。然而，当前驱动搜索工具的神经代码语言模型（CLMs）容易受到针对非功能性文本元素的对抗性攻击。在本文中，我们引入了一种与编程语言无关、可迁移的对抗性攻击，利用了这一CLM漏洞。我们的方法在不改变代码片段功能的前提下，扰动代码片段中的标识符，以人为地将代码与目标查询对齐。我们证明了，即使使用较小的代码嵌入模型（如CodeT5+）计算，我们的攻击也能高效地迁移到更大的闭源嵌入模型（如Voyage-code-3）或LLM（如Gemini-3.1-Pro）。我们的攻击可以增加查询与任意无关代码片段之间的相似性，从而降低关键检索指标（如平均倒数排名（MRR））的性能。

    arXiv:2608.26031v1 Announce Type: new  Abstract: Reliable code retrieval is crucial for developer productivity and effective code reuse. However, current neural code language models (CLMs) powering search tools are susceptible to adversarial attacks targeting non-functional textual elements. In this paper, we introduce a programming language-agnostic, transferable, adversarial attack that exploits this CLM vulnerability. Our approach perturbs identifiers within a code snippet without altering the snippet's functionality to artificially align the code with a target query. We demonstrate that our attack, even when computed using smaller code embedding models, such as CodeT5+, is highly effective and transferable to larger, closed-source embedding models, like Voyage-code-3, or LLMs like Gemini-3.1-Pro. Our attack can increase the similarity between the query and arbitrary, irrelevant code snippets, consequently degrading key retrieval metrics such as the Mean Reciprocal Rank (MRR) of sta
    
[^2]: Praxist：从实验性人工制品到解决方案谱系

    Praxist: From Experimental Artifacts to Solution Lineages

    [https://arxiv.org/abs/2608.25955](https://arxiv.org/abs/2608.25955)

    Praxist通过谱系中心方法将实验人工制品转化为类型化证据图，分离局部构建与证据综合，使研发代理能继承已验证机制，避免重复学习并提升长期研发效率。

    

    arXiv:2608.25955v1 公告类型：交叉 摘要：自主研发代理现在能够在自动评估下编写、运行和改进可执行人工制品——但很大程度上仍作为实验室工具：在精选基准上展示，其改进难以追溯到具体原因，且成本远超持续工程实践所能承受的范围。这种局限性是结构性的。大多数系统将每次尝试视为几乎自包含的过程，因此日志、记忆和搜索树记录了发生的事件，却未确立哪个设计元素产生了改进、其证据是否经受住了验证，或它如何与其他元素重组。因此，长期活动不断重复学习相同的教训。我们引入了Praxist，一个以谱系为中心的世代系统，将可复现的人工制品和评估结果转化为一个类型化的证据图，包含发现、车道结构前沿和议程。将局部人工制品构建与队列级证据综合分离，使后续尝试能够继承经过验证的机制。

    arXiv:2608.25955v1 Announce Type: cross  Abstract: Autonomous R\&D agents now write, run, and improve executable artifacts under automated evaluation---but largely as laboratory instruments: shown on curated benchmarks, with gains that are hard to trace to a cause and costs well above what sustained engineering practice absorbs. The limitation is structural. Most systems treat each attempt as nearly self-contained, so logs, memories, and search trees record what happened without establishing which design element produced an improvement, whether its evidence survived validation, or how it recombines with others. Long campaigns therefore keep re-learning the same lessons. We introduce Praxist, a lineage-centered generational system that converts reproducible artifacts and evaluator outcomes into a typed evidence graph of findings, lane-structured frontiers, and agendas. Separating local artifact construction from cohort-level evidence synthesis lets later attempts inherit validated mecha
    
[^3]: XREPOTEST：面向大型语言模型的多语言仓库级单元测试生成基准测试

    XREPOTEST: Benchmarking Multilingual Repository-Level Unit Test Generation for Large Language Models

    [https://arxiv.org/abs/2608.25939](https://arxiv.org/abs/2608.25939)

    本文提出了XREPOTEST，一个涵盖五种语言的多语言仓库级单元测试基准，并通过新指标调用率揭示LLM在现实仓库场景下与独立设置间存在显著性能差距。

    

    大型语言模型（LLMs）在自动化单元测试生成方面展现出潜力，但现有评估主要依赖于独立设置和狭窄的编程语言范围，高估了实际应用的准备程度。我们引入了XREPOTEST，一个多语言仓库级单元测试生成基准测试，涵盖五种未被充分探索的语言：Rust、Go、Julia、PHP和Ruby。XREPOTEST使用容器化执行框架和多种上下文增强策略（包括文件级、基于LSP和基于检索的上下文）在现实仓库约束下评估测试。除了标准指标（如测试通过率和覆盖率）外，我们提出了调用率（IR）来评估生成的测试是否有效执行了预期功能。对14个最先进的LLMs（包括Claude 4.5、GPT-5.2、DeepSeek V4-Pro和Qwen系列）的实验揭示，独立设置与仓库级设置之间存在显著差距。

    arXiv:2608.25939v1 Announce Type: new  Abstract: Large language models (LLMs) have shown promise for automated unit test generation, but existing evaluations largely rely on standalone settings and a narrow set of programming languages, overestimating real-world readiness. We introduce XREPOTEST, a multilingual repository-level benchmark for unit test generation spanning five underexplored languages: Rust, Go, Julia, PHP, and Ruby. XREPOTEST evaluates tests under realistic repository constraints using a containerized execution framework and multiple context augmentation strategies, including file-level, LSP-based, and retrieval-based context. Beyond standard metrics such as test pass rate and coverage, we propose Invocation Rate (IR) to assess whether generated tests meaningfully exercise the intended functionality. Experiments with 14 state-of-the-art LLMs, including Claude 4.5, GPT-5.2, DeepSeek V4-Pro, and Qwen families, reveal a substantial gap between standalone and repository-lev
    
[^4]: 修复还是重采样？重新思考大语言模型多智能体系统中的故障调试

    Repair or Resample? Rethinking Failure Debugging in LLM Multi-Agent Systems

    [https://arxiv.org/abs/2608.25920](https://arxiv.org/abs/2608.25920)

    本文提出SymTrace框架，通过受控评估区分MAS修复中的因果修复与随机重采样修复，并构建SymFail数据集以系统验证修复方法的有效性。

    

    随着基于大语言模型（LLM）的多智能体系统（MASs）越来越多地应用于长周期复杂任务，其可靠性已成为阻碍其实际部署的核心瓶颈。现有的MAS调试和修复方法通常依赖于重运行和重采样整个执行轨迹。然而，一个基本问题仍有待回答：这些方法是否真正因果性地修复了MAS故障，还是仅仅通过利用LLM采样的随机性进行随机修复？为了评估MAS修复方法的有效性，我们引入了SymTrace，一个受控评估框架，它记录MAS执行轨迹并建立干预锚点。在重放过程中，它利用记录的日志有效重建锚点之前的执行，仅重新生成下游轨迹，从而实现对MAS故障的可靠复现。我们进一步构建了数据集SymFail，包含...

    arXiv:2608.25920v1 Announce Type: cross  Abstract: As large language model (LLM)-based multi-agent systems (MASs) are increasingly applied to long-horizon complex tasks, their reliability has emerged as the core bottleneck hindering their real-world deployment. Existing MAS debugging and repair methods typically rely on rerunning and resampling the entire execution trajectory. However, a fundamental question remains to be answered: do these methods causally repair MAS failures or merely stochastically repair by leveraging the randomness of LLM sampling? To evaluate the effectiveness of MAS repair methods, we introduce SymTrace, a controlled evaluation framework that records the MAS execution trajectory and establishes intervention anchors. During replay, it effectively reconstructs the execution before the anchor using recorded logs and only regenerates the downstream trajectory, thereby enabling the reliable reproduction of MAS failures. We further construct the dataset SymFail, compr
    
[^5]: 答案廉价，证据为王！用证据增强自动化漏洞评估

    Answer Is Cheap, Show Me the Evidence! Augmenting Automated Vulnerability Assessment with Evidence

    [https://arxiv.org/abs/2608.25905](https://arxiv.org/abs/2608.25905)

    EAVA框架利用大型语言模型和专门智能体，通过两阶段训练（监督微调与强化学习）来评估软件漏洞，并生成支持性证据，以弥补现有方法忽视富文本信息和缺乏解释的不足。

    

    arXiv:2608.25905v1 公告类型：新 摘要：软件漏洞（SV）评估通过表征已报告的漏洞来帮助确定修复优先级。现有的自动化方法从漏洞报告（SVRs）中预测评估结果，但常常忽略了丰富文本中的信息，如截图和代码片段，以及关于易受攻击项目的上下文信息。它们还专注于预测准确性，而不提供解释或支持性证据，这限制了其实用性，因为分析师必须验证不完美的预测。我们提出了EAVA，一个使用大型语言模型（LLMs）来评估软件漏洞并提供支持性证据的框架。EAVA采用专门的LLM智能体处理富文本内容和项目信息，并通过两阶段训练流程构建一个专门的评估模型。它首先在自动标注的推理轨迹上进行监督指令微调以注入领域知识，然后应用强化学习（注：原文截断，此处按上下文补充）。

    arXiv:2608.25905v1 Announce Type: new  Abstract: Software vulnerability (SV) assessment helps prioritize remediation by characterizing reported vulnerabilities. Existing   automated methods predict assessment results from SV reports (SVRs), but often overlook information in rich text, such as   screenshots and code snippets, as well as contextual information about vulnerable projects. They also focus on prediction   accuracy without providing explanations or supporting evidence, limiting their practical use when analysts must validate   imperfect predictions. We propose EAVA, a framework that uses large language models (LLMs) to assess SVs and provide   supporting evidence. EAVA employs specialized LLM agents to process rich-text content and project information, and builds a   dedicated assessment model through a two-stage training pipeline. It first uses supervised instruction tuning on automatically   annotated reasoning trajectories to inject domain knowledge, and then applies reinf
    
[^6]: 超越编辑画布：OOXML到LLM摄取中的证据分歧

    Beyond the Editing Canvas: Evidence Divergence in OOXML-to-LLM Ingestion

    [https://arxiv.org/abs/2608.25880](https://arxiv.org/abs/2608.25880)

    本论文揭示OOXML文件在Office和LLM提取间存在“证据分叉”，导致同一文件产生不同证据视图，破坏语义完整性，并系统确认了21个此类构造。

    

    arXiv:2608.25880v1 公告类型：新 摘要：大型语言模型（LLM）流水线越来越多地将Office Open XML（OOXML）文档（Word、Excel和PowerPoint文件）作为金融、合规和检索增强工作流中的第一手证据，隐含地假设语义完整性：即模型消费的证据与Microsoft Office套件编辑画布中显示的内容一致。我们表明，这种假设在OOXML到LLM的流水线中可能失效。同一份符合规范的OOXML文件在Microsoft Office中可能产生一种证据视图，而在为LLM提取时产生另一种视图。每个视图都被其消费者视为权威，我们将这种状况称为“多重真实”。摄取契约很少说明哪个视图和语义角色成为模型证据，或保留证据是如何推导的。我们将引发这种分歧的、基于规范的OOXML构造称为“证据分叉”。我们系统性地遍历和挖掘OOXML规范，并确认了21个证据分叉实例。

    arXiv:2608.25880v1 Announce Type: new  Abstract: LLM pipelines increasingly ingest Office Open XML (OOXML) documents (Word, Excel, and PowerPoint files) as first-class evidence in financial, compliance, and retrieval-augmented workflows, implicitly assuming semantic integrity: that the evidence consumed by the model matches the content shown in the Microsoft Office suite editing canvas. We show that this assumption can fail in OOXML-to-LLM pipelines. The same specification-valid OOXML file can yield one evidentiary view in Microsoft Office and another when extracted for an LLM. Each view is treated as authoritative by its consumer, a condition we call plural ground truth. The ingestion contract rarely states which view and semantic roles become model evidence or preserves how that evidence was derived. We call the specification-grounded OOXML constructions that induce such divergence evidence forks.   We systematically traverse and mine the OOXML specification and confirm 21 evidence f
    
[^7]: 缩小差距：通过语义聚类在企业内部源代码中自动发现安全的Dockerfile参考标准

    Closing the Gap: Automated Discovery of Secure Dockerfile Reference Standards via Semantic Clustering in Enterprise Inner Source

    [https://arxiv.org/abs/2608.25793](https://arxiv.org/abs/2608.25793)

    提出了一种基于语义聚类的自动化流水线，用于在企业内部源代码中发现安全Dockerfile参考标准，并揭示了99%的文件存在安全配置错误的系统性缺陷。

    

    容器化主导企业软件交付，但组装容器镜像的Dockerfile经常包含安全配置错误和结构性技术债务。这个问题在企业内部源代码环境中理解不足，因为专有上下文和隔离治理阻碍了开源发现的直接应用。我们提出一个自动化的六阶段流水线：(1) 爬取企业GitLab实例，(2) 使用静态安全和质量指标（Hadolint、ShellCheck、Trivy）及生命周期数据丰富每个Dockerfile，(3) 使用LLM生成的语义描述和HDBSCAN对功能相同的工作负载进行分组，以及(4) 量化与集群内部参考实现相比的优化差距。应用于一家大型工业公司超过6,200个存储库中的11,470个Dockerfile，我们发现了一个系统性缺陷：99%的文件至少包含一个安全配置错误。

    arXiv:2608.25793v1 Announce Type: cross  Abstract: Containerization dominates enterprise software delivery, yet Dockerfiles that assemble container images frequently harbor security misconfigurations and structural technical debt. This problem is poorly understood in corporate inner-source environments, where proprietary context and isolated governance prevent direct application of open-source findings.   We present an automated, six-stage pipeline that: (1) crawls an enterprise GitLab instance, (2) enriches each Dockerfile with static security and quality metrics (Hadolint, ShellCheck, Trivy) and lifecycle data, (3) groups functionally identical workloads using LLM-generated semantic descriptions and HDBSCAN, and (4) quantifies the optimization gap against cluster-internal reference implementations.   Applied to 11,470 Dockerfiles from over 6,200 repositories at a single large industrial company, we find a systemic deficit: 99\% of files contain at least one security misconfiguration,
    
[^8]: 利用击键级编辑特征预测CS1编程课程中的困难学生

    Predicting Struggling Students in CS1 Programming Using Keystroke-Level Editing Features

    [https://arxiv.org/abs/2608.25769](https://arxiv.org/abs/2608.25769)

    本文提出利用击键级编辑特征来早期预测CS1编程课程中可能完全卡住的学生，从而为教师提供及时干预的机会。

    

    本文探讨了在CS1编程练习中，利用击键级日志早期检测困难学生的可行性。一些学生在练习结束前未能达到正确解决方案，而当这一情况从成绩或最终结果中显现时，及时提供教师支持以帮助他们恢复的机会可能已经错过。我们使用CodeBench平台的数据，该平台实时记录击键级别的代码编辑事件，以及执行和提交日志。我们定义了两个结果组：突破组（BT）学生，其先前提交均获得0%，最终提交获得满分；完全卡住组（FS）学生，其所有提交均获得0%，未达到正确解决方案。为了检验这种可行性，我们关注两个问题：（RQ1）添加击键级编辑特征是否比仅使用执行日志特征更能改善对FS学生的预测。

    arXiv:2608.25769v1 Announce Type: new  Abstract: This paper investigates the feasibility of early detection of struggling students during CS1 programming exercises using keystroke-level logs. Some students fail to reach a correct solution before the exercise ends, and by the time this becomes apparent from grades or final outcomes, the opportunity for timely instructor support aimed at helping them recover may have passed. We use data from the CodeBench platform, which records real-time code editing events at the keystroke level, alongside execution and submission logs. We define two outcome groups: Breakthrough (BT) students, whose prior submissions all receive 0% and whose final submission achieves full credit, and Fully Stuck (FS) students, whose submissions all receive 0% without reaching a correct solution. To examine this feasibility, we focus on two questions: (RQ1) whether adding keystroke-level editing features improves the prediction of FS students over execution-log features
    
[^9]: 从通用代理到RCA专家：一种用于根因分析的自进化控制框架

    From General Agents to RCA Experts: A Self-Evolving Harness for Root Cause Analysis

    [https://arxiv.org/abs/2608.25661](https://arxiv.org/abs/2608.25661)

    本文提出了一种自进化的外部控制框架，通过复用通用代理的强能力而非重建专用代理，显著提升根因分析准确性并随经验积累持续优化。

    

    基于大语言模型（LLMs）的自动化根因分析（RCA）正受到越来越多的关注。目前，站点可靠性工程师（SREs）通常通过两种方式使用LLMs实现RCA自动化：直接使用通用代理（例如Codex或Claude Code）进行诊断，或者从零构建专门的RCA代理。随着主流通用代理能力不断增强且迭代迅速，我们的定量研究发现，前者现在往往优于后者。然而，其准确性仍达不到生产需求，这一差距主要源于代理通用能力之外的外部适应层，即控制框架。因此，我们认为基于LLM的RCA应聚焦于这一外部控制框架，复用现代代理的强大通用能力，而非从零重建代理。这种控制框架的一个关键能力是自进化，即从过去的诊断中积累系统特定经验，从而随着使用次数的增加而不断改进。

    arXiv:2608.25661v1 Announce Type: new  Abstract: Automated root cause analysis (RCA) with large language models (LLMs) has drawn growing attention. Today, SREs typically automate RCA with LLMs in one of two ways: directly using a general-purpose agent (e.g., Codex or Claude Code) for diagnosis, or building a specialized RCA agent from scratch. As mainstream general agents grow more capable and iterate quickly, our quantitative study finds that the former now often surpasses the latter. Its accuracy, however, still falls short of production needs, and this gap stems mainly from the external adaptation layer outside the agent's general capabilities, namely the harness. We therefore argue that LLM-based RCA should focus on this external harness, reusing the strong general capabilities of a modern agent rather than rebuilding an agent from scratch. A key capability of such a harness is to self-evolve, accumulating system-specific experience from past diagnoses so that it gets better the mo
    
[^10]: 纳西索斯：利用上下文感知大语言模型近似进行程序合成

    Narcissus: Program Synthesis Using Context-Aware LLM Approximations

    [https://arxiv.org/abs/2608.25657](https://arxiv.org/abs/2608.25657)

    纳西索斯通过将大语言模型提议保留为语法树并在候选程序上下文中评分扩展，结合正则化确保规则可达，从而在目标语言罕见时实现鲁棒的程序合成。

    

    大语言模型（LLMs）在编程方面表现出色，但当任务固定目标语言时则不然：若提示中包含其训练数据中罕见的语法，它们生成的程序通常会破坏语法或无法满足给定规范。枚举式合成器在LLM引导下系统性地搜索语法正确的程序空间；现有技术通过将LLM提议近似为规则频率来引导搜索，但这会丢失每个构造所属的上下文，并在提议错误时剪除所有提议未覆盖的规则——而这恰恰发生在提议出错之时。我们提出纳西索斯（Narcissus），一种合成器，它将提议保留为语法树，并在候选程序的上下文中对每次扩展进行评分：具有相同周围结构的提议是否以相同方式延续，且扩展是否重建了提议重复出现的片段？一个正则化项确保每条规则保持可达，因此错误的提议只会延迟解决方案，但无法将其隐藏。在五个领域中的实验表明……

    arXiv:2608.25657v1 Announce Type: cross  Abstract: Large language models (LLMs) excel at programming, but not when the task fixes the target language: prompted with a grammar rare in their training data, their programs usually break the grammar or fail the given specification. Enumerative synthesizers search the space of syntactically correct programs systematically guided by LLMs; the state of the art guides them by approximating LLM proposals into rule frequencies, which loses where each construct belongs and prunes every rule the proposals miss, exactly when the proposals are wrong. We present Narcissus, a synthesizer that keeps the proposals as syntax trees and scores each expansion of a candidate program in its context: does a proposal with the same surrounding structure continue the same way, and does the expansion rebuild a fragment the proposals repeat? A regularization term keeps every rule reachable, so wrong proposals delay the solution but cannot hide it. Across five domain
    
[^11]: DBcover：一种面向覆盖率提升的白盒SQL测试生成框架

    DBcover: A White-box SQL Test Generation Framework for Coverage Improvement

    [https://arxiv.org/abs/2608.25573](https://arxiv.org/abs/2608.25573)

    DBcover提出了一种基于大语言模型和白盒上下文推理的SQL测试生成框架，通过知识图谱整合全局与局部上下文，实现RDBMS覆盖率提升。

    

    关系型数据库管理系统（RDBMS）是现代数据密集型应用的支柱，其可靠性和健壮性至关重要。然而，由于代码库庞大且执行逻辑复杂，在RDBMS测试中实现高覆盖率仍然具有挑战性。传统的模糊测试依赖随机SQL生成，无法捕捉SQL输入与内部执行路径之间的对应关系，而符号执行则面临成本过高和可扩展性限制。我们提出了DBcover，一种基于上下文推理的、由大语言模型驱动的白盒SQL测试生成框架。DBcover使用轻量级动态分析提取SQL到路径的对应关系和调用图作为全局上下文，并收集目标函数周围的源代码级信息作为局部上下文。这些上下文被组织在统一的知识图谱中，以实现高效检索和复用。DBcover随后执行两阶段测试生成：首先...

    arXiv:2608.25573v1 Announce Type: cross  Abstract: Relational Database Management Systems (RDBMSs) are the backbone of modern data-intensive applications, making reliability and robustness critical. However, achieving high coverage in RDBMS testing remains challenging because of large codebases and complex execution logic. Traditional fuzzing relies on random SQL generation and cannot capture the correspondence between SQL inputs and internal execution paths, while symbolic execution suffers from prohibitive cost and scalability limitations.   We propose DBcover, an LLM-driven white-box SQL test generation framework based on contextual reasoning. DBcover uses lightweight dynamic analysis to extract SQL-to-path correspondence and call graphs as global context, and collects source-level information around target functions as local context. These contexts are organized in a unified knowledge graph for efficient retrieval and reuse. DBcover then performs two-phase test generation: it first
    
[^12]: 一种用于移动商务应用评分评估的混合可用性方法

    A Hybrid Usability Approach for Rating Evaluation of M-Commerce Applications

    [https://arxiv.org/abs/2608.25550](https://arxiv.org/abs/2608.25550)

    本文提出了一种混合可用性模型，通过整合八个关键因素来预测移动商务应用的评分，并基于五个实际应用进行了评估。

    

    任何移动应用的成功都依赖于其实用性，而评分被认为是这方面的重要衡量标准。本研究工作侧重于识别对移动商务应用评分有显著贡献的可用性因素。本工作旨在探索由不同因素和一组标准组成的现有可用性模型，并通过考虑5个知名移动应用（即(i) daraz、(ii) shophive、(iii) home shopping、(iv) Symbios、(v) yayvo）来评估其在评分估计方面的表现。然后，本工作提供了一个用于移动商务应用评分预测的混合可用性模型。初始混合可用性模型包括(i) 可学习性、(ii) 一致性、(iii) 人为因素、(iv) 沟通性、(v) 有效性、(vi) 可操作性、(vii) 效率、(viii) 满意度。每个因素包含一些标准。考虑到混合可用性模型的因素，数据被...

    arXiv:2608.25550v1 Announce Type: new  Abstract: The success of any mobile application relies on its usefulness and rating is considered as an important measure in this regard. This research work focuses on identifying usability factors, which contribute significantly towards the rating of M-commerce apps. This work intends to explore existing usability models consisting of different factors along with a set of criteria and evaluate in terms of rating estimation by considering 5 well-known mobile applications, namely (i) daraz, (ii) shophive, (iii) home shopping, (iv) Symbios. (v) yayvo. Then, this work provides a hybrid usability model for rating prediction of M-commerce applications. The initial hybrid usability model comprises of (i) learnability, (ii) consistency, (iii) human factors,(iv)communicativeness,(v)effectiveness, (vi) Operability, (vii) efficiency, (viii) satisfaction. Each factor consists of some criteria. Keeping in view the factors of hybrid usability model, the data w
    
[^13]: 一种面向时空可组合性的编程范式

    A Programming Paradigm for Spatiotemporal Composability

    [https://arxiv.org/abs/2608.25512](https://arxiv.org/abs/2608.25512)

    该论文提出了一种编程范式，通过将效果和协效果提升为运行时机制，形式化了可逆效果和响应式协效果，分别实现时间可组合性（完全撤销副作用）和空间可组合性（响应式管理依赖）。

    

    现代软件——从插件系统到自我进化的智能体框架——日益需要动态组合，但其形式化基础仍不完善。我们识别出该问题的两个正交维度：时间可组合性，即在移除组件时完全撤销其副作用的能力；以及空间可组合性，即声明并响应式管理组件间依赖的能力。我们通过将经典的效果（effect）和协效果（coeffect）概念提升为运行时机制来解决这两个维度。具体而言，我们形式化了可逆效果，其中每个上下文变换都携带一个由运行时持有的逆变换，从而在单个组件局部建立时间可组合性。我们形式化了响应式协效果，其中每个上下文变化都根据组件的协效果规范进行分类，以驱动其激活和停用，从而在单个组件局部建立空间可组合性。然后我们将这两者统一...

    arXiv:2608.25512v1 Announce Type: cross  Abstract: Modern software -- from plugin systems to self-evolving agent harnesses -- increasingly requires dynamic composition, yet its formal foundations remain underdeveloped. We identify two orthogonal dimensions of the problem: temporal composability, the ability to completely revert a component's side effects upon removal, and spatial composability, the ability to declare and reactively manage inter-component dependencies. We address the two dimensions by lifting classical effect and coeffect concepts to runtime mechanisms. In particular, we formalize revertible effects, in which every context transformation carries an inverse that the runtime holds, establishing temporal composability local to one component. We formalize reactive coeffects, in which every context change is classified against a component's coeffect specification to drive its activation and deactivation, establishing spatial composability local to one component. We then unif
    
[^14]: 将披露与授权分离：面向代理操作中介的字段级最小化

    Separating Disclosure from Authorization: Field-Tier Minimization for Agent Action Mediation

    [https://arxiv.org/abs/2608.25474](https://arxiv.org/abs/2608.25474)

    本文提出了一种字段级最小化方法，将操作参数分为三个层级，通过先计算完整参数的规范摘要再最小化，实现了披露与授权的分离，且不影响历史账本的有效性。

    

    摘要：一个授权操作的系统必须看到足够的信息来做出决定，而一个为其决定作证的系统必须记录足够的信息以供审计。这两种压力都将原始操作参数——接收者、支付备注、记录标识符——推入一个无法删除的追加式账本中。我们证明这两者是可以分离的。我们将每个参数字段（而非每个操作类别）分为三个层级：策略可以合法匹配的字段，这些字段以原始形式传输；与策略相关但具有识别性的字段，这些字段仅以投影形式传输，如电子邮件域名或模板化路由形状；以及没有合法策略用途的字段，这些字段永远不会离开工作负载。核心属性是账本的承诺是对完整、未最小化参数的规范摘要，在最小化运行之前计算。因此，该承诺独立于层级表：重新分类一个字段会改变披露的内容，而不会使历史承诺失效。

    arXiv:2608.25474v1 Announce Type: cross  Abstract: A system that authorizes an action must see enough of it to decide, and a system that attests to its decision must record enough to be audited. Both pressures push raw action parameters -- recipients, payment memos, record identifiers -- into an append-only ledger that cannot delete them. We show the two are separable. We classify each parameter field, not each action class, into three tiers: fields a policy may legitimately match on, which cross raw; fields that are policy-relevant but identifying, which cross only as projections such as an email domain or a templated route shape; and fields with no legitimate policy use, which never leave the workload. The central property is that the ledger's commitment is a canonical digest of the full, unminimized parameters, computed before minimization runs. The commitment is therefore independent of the tier table: reclassifying a field changes what is disclosed without invalidating a historica
    
[^15]: RotDroid：跨方向状态等价性测试，用于检测Android应用中的GUI旋转缺陷

    RotDroid: Cross-Orientation State Equivalence Testing for Detecting GUI Rotation Bugs in Android Apps

    [https://arxiv.org/abs/2608.25425](https://arxiv.org/abs/2608.25425)

    RotDroid通过生成状态保持动作序列和微调视觉语言模型RotVL，实现了跨方向GUI状态等价性检查，从而有效检测Android应用中的旋转缺陷。

    

    屏幕旋转是Android应用中的一项基本交互，但它常常引入非崩溃性功能故障（NCFs），例如布局不一致和状态丢失，这些故障难以自动检测。一个关键挑战是缺乏有效的测试预言（oracle）来检查竖屏和横屏视图之间的跨方向状态等价性。我们提出了RotDroid，一个通过跨方向状态等价性检测GUI旋转缺陷的测试框架。RotDroid生成并变异状态保持动作序列（SPS），以构造跨方向语义等价的GUI状态。为支持可靠的预言检查，我们构建了RotBench，一个配对竖屏-横屏GUI状态的数据集，并开发了RotVL，一个针对等价性检查微调的视觉语言模型。在合成和真实世界数据集上的实验表明，RotVL优于最先进的模型，且RotDroid能检测更多旋转缺陷。

    arXiv:2608.25425v1 Announce Type: new  Abstract: Screen rotation is a fundamental interaction in Android applications, but it often introduces non-crashing functional failures (NCFs), such as layout inconsistencies and state loss, which are difficult to detect automatically. A key challenge is the lack of effective test oracles for checking cross-orientation state equivalence between portrait and landscape views. We propose RotDroid, a testing framework for detecting GUI rotation bugs via cross-orientation state equivalence. RotDroid generates and mutates State-Preserving action Sequences (SPS) to construct semantically equivalent GUI states across orientations. To support reliable oracle checking, we build RotBench, a dataset of paired portrait-landscape GUI states, and develop RotVL, a vision-language model fine-tuned for equivalence checking. Experiments on both synthetic and real-world datasets show that RotVL outperforms state-of-the-art models, and RotDroid detects more rotation-
    
[^16]: 分布式系统中的重试放大效应：重试策略及其在级联故障中作用的系统分析

    Retry Amplification in Distributed Systems: A Systematic Analysis of Retry Policies and Their Role in Cascading Failures

    [https://arxiv.org/abs/2608.25403](https://arxiv.org/abs/2608.25403)

    本文提出重试放大因子（RAF）指标，通过分析200个开源项目发现重试策略普遍缺乏退避机制，且简单的标准重试策略在相关故障下会显著降低系统成功率，从而揭示分布式系统中重试策略的集体行为可能加剧级联故障。

    

    arXiv:2608.25403v1 公告类型：新公告 摘要：重试机制是弹性分布式系统的标准组成部分，但它们在调用路径中每一层同时重试时的集体行为，比产生这些重试的每客户端指导方针更难以理解。本文引入了重试放大因子（RAF），这是一个量化在部分故障期间重试策略产生的额外请求量的指标。在对200个开源Python微服务项目的研究中，显式重试逻辑在11.5%的项目中被检测到，而对我们自身假阴性的审计表明真实发生率接近41%。在检测到的项目中，60.9%至少包含一个没有退避的配置，经手动验证，113个生产配置中仅有一个随机化了其延迟。然后，我们在模拟中评估这些策略（每种策略n=100次试验）。在相关故障下，相对于PE（某种基线策略），一种简单的标准重试策略将成功率从55.4%降至41.5%。

    arXiv:2608.25403v1 Announce Type: new  Abstract: Retry mechanisms are a standard component of resilient distributed systems, but their collective behavior, when every tier in a call path retries concurrently, is less well understood than the per-client guidance that produced them. This paper introduces the retry amplification factor (RAF), a metric quantifying the additional request volume that retry policies generate during partial failures. In a study of 200 open-source Python microservice projects, explicit retry logic is detected in 11.5%, and an audit of our own false negatives places true prevalence near 41%. Among the projects detected, 60.9% contain at least one configuration without backoff, and after manual verification exactly one of 113 production configurations randomizes its delay. We then evaluate these policies in simulation (n = 100 trials per strategy). Under correlated failures, a naive standard retry policy reduces the success rate from 55.4% to 41.5% relative to pe
    
[^17]: 在阿尔法之前进行时点审计：公共档案可用性与BTC永续合约的负匹配预算研究

    Point-in-Time Audit Before Alpha: Public-Archive Availability and a Negative Matched-Budget Study on BTC Perpetual Futures

    [https://arxiv.org/abs/2608.25348](https://arxiv.org/abs/2608.25348)

    本文提出一种时点审计方法，验证公共加密货币数据在决策时的可用性，并发现未平仓量数据存在发布时间不可验证问题，最终通过修订核心数据流和保留划分实现有效审计。

    

    arXiv:2608.25348v1 公告类型：新 摘要：公共加密货币档案在文件存在时可能看似可用，尽管因子研究要求在每次决策时观察数据可用且可执行。我们使用事件时间、发布时间和可用性时间审计公共币安BTCUSDT USD-M永续期货数据，并将提议与确定性审计、评估和保留访问分离。初始的无缺口五分钟要求（涵盖交易、标记、指数和未平仓量）未通过：最长未修复交集为304.5729166666667天。一项已披露的修订将交易、标记、指数和已实现资金作为核心数据流，并因未平仓量发布时间未验证而将其设为可选。修订后的掩码保留了727个完整的UTC日，并支持436/145/146天的训练、验证和历史保留划分。在80个冻结的已知规则模板中，审计器检测到40/40个违规，并拒绝了0/40个合法模板。在十个零信号路径中，完整审计...

    arXiv:2608.25348v1 Announce Type: new  Abstract: Public cryptocurrency archives may appear usable when files exist, although factor research requires observations available and executable at each decision time. We audit public Binance BTCUSDT USD-M perpetual-futures data using event, publication, and availability times and separate proposal from deterministic auditing, evaluation, and holdout access. An initial gapless five-minute requirement for trade, mark, index, and open interest failed: the longest unrepaired intersection was 304.5729166666667 days. A disclosed revision made trade, mark, index, and realized funding the core streams and made open interest optional because its publication time was unverified. The revised mask retained 727 complete UTC days and supported a 436/145/146-day train, validation, and historical-holdout split. On 80 frozen known-rule templates, the auditor detected 40/40 violations and rejected 0/40 legal templates. Across ten null-signal paths, full auditi
    
[^18]: Metis：面向工具使用软件代理的类型化运行时中介

    Metis: Typed Runtime Mediation for Tool-Using Software Agents

    [https://arxiv.org/abs/2608.25322](https://arxiv.org/abs/2608.25322)

    本文提出Metis，一种通过类型化事件流实现显式权限决策和干扰分类的运行时中介机制，显著降低了工具使用代理的延迟并增强了安全性。

    

    软件代理将概率模型的输出连接到改变仓库、流程、网络和图形应用的操作中。我们提出了Metis，一个多提供者运行时，它在被允许的调用到达外部效果之前，将提供者流转换为类型化事件。其执行路径使权限决策、干扰类别、终端结果和生命周期转换变得明确且可检查。我们在冻结的源代码工件上评估了这些机制。在30对匹配的真实I/O对中，四类中介将中位耗时从强制序列化下的25.958毫秒降至14.146毫秒。平均配对差异为-12.295毫秒（95%自举区间[-12.968, -11.694]），所有配对中中介均更快。一个十案例故障矩阵暴露了重复标识符和回滚限制。在子边界消融中，完整的门控加注册表条件阻止了声明未经授权的效果，并隐藏了所有五个逃逸工具。

    arXiv:2608.25322v1 Announce Type: new  Abstract: Software agents connect probabilistic model output to operations that change repositories, processes, networks, and graphical applications. We present Metis, a multi-provider runtime that converts provider streams into typed events before admitted calls reach external effects. Its execution path makes permission decisions, interference classes, terminal results, and lifecycle transitions explicit and inspectable. We evaluate these mechanisms on frozen source artifacts. Across 30 matched real-I/O pairs, four-class mediation reduced median elapsed time from 25.958 ms under forced serialization to 14.146 ms. The mean paired difference was -12.295 ms (95% bootstrap interval [-12.968, -11.694]), with mediation faster in all pairs. A ten-case fault matrix exposed duplicate-identifier and rollback limits. In a child-boundary ablation, the full gate-plus-registry condition blocked the declared unauthorized effect and hid all five escape tools. R
    
[^19]: 几页 Markdown：编码代理采用后的已提交 AI 配置与更低的质量成本

    A Few Pages of Markdown: Committed AI Configuration and Lower Quality Cost after Coding-Agent Adoption

    [https://arxiv.org/abs/2608.25241](https://arxiv.org/abs/2608.25241)

    本文提出 RAMP 成熟度模型，证明编码代理采用能加速开发（提交量增加 28-38%），同时通过版本控制的 AI 配置降低质量成本，且多数配置为一次性设置。

    

    编码代理提升了开发速度，但也增加了技术债务。先前的研究仅报告了采用者间的平均效应，掩盖了团队间的巨大差异。我们引入了 RAMP（仓库 AI 成熟度模型），这是一个基于版本控制工件、团队提交以配置 AI 工具的四级累积成熟度模型。RAMP 从行为规则和编码标准，到命名代理定义，再到多代理编排，其中观察到的实践集中在前三级别。在 441 个仓库中，这些级别表现为累积量表，独立人工标注在 97% 的保留样本上重现了 RAMP 的仓库级标签。采用是累积的、仅向前且设置后即忘的：73.8% 的工件被提交一次且从未修改。在每层内重新估计现有的代理采用面板，代理无论成熟度如何都加速开发（提交量增加 28-38%），但质量成本降低。

    arXiv:2608.25241v1 Announce Type: new  Abstract: Coding agents increase development velocity but also technical debt. Prior work reports only average effects across adopters, hiding wide differences between teams. We introduce RAMP (Repository AI Maturity Profile), a four-level cumulative maturity model grounded in version-controlled artifacts that teams commit to configure AI tools. RAMP runs from behavioral rules and coding standards through named agent definitions to multi-agent orchestration, with observed practice concentrated in the first three levels. Across 441 repositories the levels behave as a cumulative scale, and independent human annotation reproduces RAMP's repository-level labels on 97% of a held-out sample. Adoption is cumulative, forward-only, and set-and-forget: 73.8% of artifacts are committed once and never modified. Re-estimating an existing agent-adoption panel within each stratum, agents accelerate development regardless of maturity (28-38% more commits), but qu
    
[^20]: SPECMINE：一个大规模的规范驱动开发工件语料库

    SPECMINE: A Large-Scale Corpus of Spec-Driven Development Artifacts

    [https://arxiv.org/abs/2608.25202](https://arxiv.org/abs/2608.25202)

    我们提出了SPECMINE，这是首个大规模语料库，通过两次普查系统地捕捉了GitHub上规范驱动开发工件，为研究规范如何转化为代码提供了基础数据。

    

    arXiv:2608.25202v1 公告类型：新 摘要：规范驱动开发（SDD）是一种快速兴起的新实践，其中由开发者编写、或（更常见地）由AI工具起草再由开发者整理的、结构化自然语言规范，驱动AI编码代理的实现。自2025年以来，一波工具（如GitHub Spec Kit [3]、OpenSpec [4]、AWS Kiro [5]以及数十种其他工具）已经出现，但这些工具产生的工件从未被大规模研究过。我们提出了SPECMINE，一个通过两次普查捕捉公共GitHub仓库中SDD的语料库：一次广泛普查覆盖了大多数工具的spec.md/specs.md文件（涵盖73,030个仓库中的470,795个文件，归属于17个命名工具），以及一次针对Kiro独特的需求/设计/任务布局的普查（涵盖12,910个仓库中的98,574个文件）。每个规范都附有完整的仓库元数据、完整的提交历史以及解析后的文档结构。规范如何转化为代码本身就是一个开放问题，因此对于...

    arXiv:2608.25202v1 Announce Type: new  Abstract: Spec-Driven Development (SDD) is a fast-emerging practice in which a structured natural-language specification, written by a developer, or (more often) drafted by an AI tool and then curated by the developer, drives an AI coding agent's implementation. A wave of tooling (GitHub Spec Kit [3], OpenSpec [4], AWS Kiro [5], and dozens of others) has appeared since 2025, yet the artifacts these tools produce have never been studied at scale. We present SPECMINE, a corpus that captures SDD in public GitHub repositories through two censuses: a broad census of spec.md/specs.md files covering most tools (470,795 files across 73,030 repositories, attributed to 17 named tools), and a Kiro census of its distinct requirements/design/tasks layout (98,574 files across 12,910 repositories). Each spec is enriched with full repository metadata, complete commit history, and parsed document structure. How a spec becomes code is itself an open question, so fo
    
[^21]: 基于模型的智能体软件工程

    Model-Based Agentic Software Engineering

    [https://arxiv.org/abs/2608.25174](https://arxiv.org/abs/2608.25174)

    MAGE通过外部化最小有目的的表征并利用约束赋予义务权威，解决了智能体软件工程中的表征和权威问题，从而从普通智能构建可信赖的自主性。

    

    arXiv:2608.25174v1 公告类型：新  摘要：编程智能体在提高实现能力的同时，并未自动使项目意图、系统结构或验收证据变得明确。当实现相对于工程判断变得丰富时，稀缺的工作转向选择有用的抽象、生成证据以及确定哪些义务支配验收。现有工作流通过更大的提示、仓库检索或逐次变更审查来解决部分差距，但仍要求智能体和工程师重构关键属性。作为替代方案，我们提出了基于模型的智能体软件工程（MAGE）。MAGE是一个框架和理论，旨在从普通智能中构建可信赖的自主性。MAGE解决了表征问题和权威问题：它将回答工程问题所需的最小有目的表征外部化，然后通过约束赋予已确定的义务相应的权威。

    arXiv:2608.25174v1 Announce Type: new  Abstract: Coding agents increase implementation capacity without automatically making project intent, system structure, or acceptance evidence explicit. As implementation becomes abundant relative to engineering judgment, the scarce work shifts toward choosing useful abstractions, producing evidence, and determining which obligations govern acceptance. Existing workflows address parts of this gap through larger prompts, repository retrieval, or perchange review, but still require agents and engineers to reconstruct consequential properties.   As an alternative, we present Model-Based Agentic Software Engineering (MAGE). MAGE is a framework and a theory for building trustworthy autonomy from commodity intelligence. MAGE addresses a representation problem and an authority problem: it externalizes the smallest purposeful representation needed to answer an engineering question, then gives settled obligations proportionate authority through constraints
    
[^22]: FuzzingBrain-Bench V1：评估大语言模型开放式漏洞发现能力

    FuzzingBrain-Bench V1: Evaluating Open-Ended Bug Discovery by LLMs

    [https://arxiv.org/abs/2608.25158](https://arxiv.org/abs/2608.25158)

    该基准测试通过开放式崩溃发现而非预定义目标来评估LLM的漏洞发现能力，强调真实能力评估。

    

    arXiv:2608.25158v1 公告类型：交叉 摘要：评估大语言模型（LLMs）发现软件漏洞的能力日益重要。现有基准测试通常通过让模型生成触发预定义目标漏洞的概念验证输入来评估此能力。然而，这种设置可能忽略模型发现的未匹配预定义目标的有效崩溃。因此，评估可能无法反映模型的真实能力。我们提出了FuzzingBrain-Bench，一个用于评估AI模型在开源软件中发现问题能力的基准测试。模型被赋予一个开源项目和一个带有消毒器插桩的测试框架，封装在自包含的Docker镜像中。其目标是通过测试框架生成输入，尽可能触发多种不同的崩溃。模型在每个挑战中的表现基于其产生的不同崩溃签名的数量进行评分，该数量设有预设上限并加权。

    arXiv:2608.25158v1 Announce Type: cross  Abstract: Evaluating the ability of large language models (LLMs) to discover software bugs is increasingly important. Existing benchmarks typically evaluate this capability by asking the model to generate a proof-of-concept input that triggers a predefined target vulnerability. However, this setup may overlook valid crashes discovered by the model when they do not match the predefined target. As a result, the evaluation may not reflect the model's real capability.   We present FuzzingBrain-Bench, a benchmark for assessing AI models' ability to discover bugs in open-source software. Models are given an open-source project and a sanitizer-instrumented harness in a self-contained Docker image. Their goal is to generate inputs that trigger as many distinct crashes as possible through the harness. A model's performance on each challenge is scored based on the number of distinct crash signatures it produces, capped at a predefined maximum and weighted
    
[^23]: ARISMA：人工智能与大型语言模型辅助的系统综述、范围综述和绘图研究的指南

    ARISMA: Guidelines for AI- and LLM-Assisted Systematic Reviews, Scoping Reviews, and Mapping Studies

    [https://arxiv.org/abs/2608.25050](https://arxiv.org/abs/2608.25050)

    ARISMA提出了一套首个端到端指南，为AI和LLM辅助的系统综述提供方法论适当性、验证、人类主导决策和可审计报告的标准化框架。

    

    系统综述、范围综述、绘图研究及相关证据综合，随着检索量、更新周期和综合要求的持续扩大，完全依赖人工流程变得越来越困难。与此同时，人工智能、机器学习和大型语言模型正迅速进入综述实践，涵盖查询制定、筛选、提取、分类、评估支持和报告等环节。然而，实证证据仍不均衡、依赖任务，且不足以支持无约束的自动化。现有标准如PRISMA 2020、PRISMA-S、PRISMA-ScR、PRISMA-P、PRESS和SWiM仍然至关重要，但均未提供端到端的操作标准，以明确AI使用在方法论上何时适当、应如何验证、哪些综述决策必须由人类主导，以及如何报告AI参与以便读者审计。本文提出了ARISMA，一种...

    arXiv:2608.25050v1 Announce Type: new  Abstract: Systematic reviews, scoping reviews, mapping studies, and related evidence syntheses are increasingly difficult to conduct with fully manual workflows as search volumes, update cycles, and synthesis requirements continue to expand. At the same time, artificial intelligence, machine learning, and large language models are rapidly entering review practice across query formulation, screening, extraction, categorization, appraisal support, and reporting. Yet the empirical evidence remains uneven, task-dependent, and insufficient to justify unconstrained automation. Existing standards such as PRISMA 2020, PRISMA-S, PRISMA-ScR, PRISMA-P, PRESS, and SWiM remain essential, but none provides an end-to-end operational standard for when AI use is methodologically appropriate, how it should be validated, which review decisions must remain human-led, and how AI involvement should be reported so that readers can audit it. This paper proposes ARISMA, a
    
[^24]: 前沿挑战：评估科学工作流完成度

    FrontierChallenge: Evaluating Scientific Workflow Completion

    [https://arxiv.org/abs/2608.24979](https://arxiv.org/abs/2608.24979)

    本文介绍了FrontierChallenge基准测试，用于评估科学智能体在跨领域端到端工作流中的完成能力，发现当前最佳模型仅能完成20.6%的任务，表明部分进展难以转化为完整交付物。

    

    arXiv:2608.24979v1 公告类型：交叉 摘要：科学智能体日益用于分析数据、执行代码并生成研究产物，然而大多数基准测试强调最终答案、孤立程序或单一领域。我们引入了FrontierChallenge，一个跨领域基准测试，包含300个端到端科学工作流。在本文中，我们发布并评估了其中97个任务，涵盖量子化学、分子动力学、材料表征、分析化学、生命科学以及电化学/环境领域。每个任务提供固定输入，并指定所需科学交付物的集合。我们评估了十二个前沿模型和三种智能体脚手架。通过率衡量满足完全完成标准的任务比例，而平均得分捕捉部分进展。每个最佳配置仅完成了97个已发布任务中的20个，通过率为20.6%。部分进展尤其难以转化为完整的交付物。

    arXiv:2608.24979v1 Announce Type: cross  Abstract: Scientific agents increasingly analyze data, execute code, and produce research artifacts, yet most benchmarks emphasize final answers, isolated programs, or a single domain. We introduce FrontierChallenge, a cross-domain benchmark comprising 300 end-to-end scientific workflows. In this paper, we release and evaluate 97 of these tasks, spanning quantum chemistry, molecular dynamics, materials characterization, analytical chemistry, life science, and electrochemistry/environment. Each task provides fixed inputs and specifies a bundle of required scientific deliverables. We evaluate twelve frontier models with three agent scaffolds. Pass Rate measures the fraction of tasks satisfying the full-completion criterion, while Avg. Score captures partial progress. Each of the best-performing configurations completed only 20 of the 97 released tasks, yielding a Pass Rate of 20.6%. Partial progress translated especially poorly into complete deliv
    
[^25]: 评估与预防AI生成的Ansible代码中的安全缺陷

    Evaluating and Preventing Security Smells in AI-Generated Ansible Code

    [https://arxiv.org/abs/2608.24962](https://arxiv.org/abs/2608.24962)

    本文评估了16个AI模型生成的Ansible代码，发现所有模型在无指导时均产生安全缺陷，并提出了一种基于扩展CO-STAR框架的提示集成方法，能在合成阶段预防这些缺陷，使部分模型生成合规代码。

    

    arXiv:2608.24962v1 公告类型：新 摘要：AI编码助手生成基础设施即代码，但尚无研究检验这些代码是否满足安全要求。这一点很重要，因为基础设施代码中的安全缺陷会传播到部署系统中，产生不安全且不可信的基础设施。我们评估了16个AI模型生成Apache Tomcat v10和MongoDB v7的Ansible角色，并根据CIS基准分析了278个Ansible角色。在没有安全指导的情况下，所有16个AI模型生成的代码都包含安全缺陷，导致易受攻击的基础设施无法通过合规性验证，且性能低于人类开发者编写的代码。我们提出了一种方法，通过扩展的CO-STAR框架将Ansible最佳实践和CIS基准整合到提示中，从而在合成阶段预防安全缺陷，而非在部署后检测。应用此方法后，16个模型中有4个生成了合规代码，其中领先的模型...

    arXiv:2608.24962v1 Announce Type: new  Abstract: AI coding assistants generate Infrastructure as Code, yet no work has examined whether this code meets security requirements. This matters because security smells in infrastructure code propagate to deployed systems, producing infrastructure that is insecure and untrustworthy. We evaluate 16 AI models generating Ansible roles for Apache Tomcat v10 and MongoDB v7, analysing 278 Ansible roles against CIS benchmarks. Without security guidance, all 16 AI models produced code containing security smells, resulting in vulnerable infrastructure that fails compliance verification and underperforms code written by human developers. We introduce an approach integrating Ansible best practices and CIS benchmarks into prompts through an extended CO-STAR framework, enabling security smell prevention during synthesis rather than detection after deployment. When this approach is applied, 4 out of 16 models generate compliant code, with the leading model 
    
[^26]: ToolMinimize：审计并重写LLM代理工具调用以最小化隐私暴露

    ToolMinimize: Auditing and Rewriting LLM Agent Tool Calls to Minimize Privacy Exposure

    [https://arxiv.org/abs/2608.24957](https://arxiv.org/abs/2608.24957)

    本文提出ToolMinimize，一种能够自动重写LLM代理工具调用参数以最小化隐私暴露的中间件，有效减少不必要的隐私数据共享。

    

    大型语言模型代理在工具调用参数中经常包含超出所调用工具实际需求的隐私敏感数据，每次调用都会跨越信任边界传递给第三方服务。对三个生产级LLM（GPT-4o、Claude 3.5 Sonnet、Llama-3.3-70B）的受控测量显示，在默认提示下，81%至88%的工具调用包含不必要的隐私敏感数据；即使有明确的隐私指令，仍有36%至76%的过度共享。现有防御措施要么门控调用（允许/阻止），要么标记数据流（信息流控制），但无法重写参数值，而PII检测工具会遗漏隐式隐私敏感数据，如“Memorial Sloan Kettering”（一个暗示诊断结果的医院名称）。我们提出了ToolMinimize系统，这是一个中间件，拦截工具调用并将其参数重写为工具功能所需的最小数据，结合了模式感知的必要性分析和四种操作：移除、泛化、替换和截断。

    arXiv:2608.24957v1 Announce Type: cross  Abstract: LLM agents routinely include privacy-sensitive data (PSD) in tool call arguments beyond what the invoked tools require, crossing trust boundaries to third-party services on every invocation. A controlled measurement on three production LLMs (GPT-4o, Claude 3.5 Sonnet, Llama-3.3-70B) shows that 81--88\% of tool calls include unnecessary PSD under default prompts; explicit privacy instructions still leave 36--76\% over-sharing. Existing defenses gate calls (allow/block) or label flows (information-flow control) but cannot \emph{rewrite} argument values, and PII detection tools miss implicit PSD like ``Memorial Sloan Kettering'' (a hospital name that implies a diagnosis). We present \system{}, a middleware that intercepts tool calls and rewrites their arguments to the minimum data necessary for tool functionality, combining schema-aware necessity analysis with four operations: removal, generalization, substitution, and truncation. Live va
    
[^27]: 现代二进制反编译的演进：分类法、文献综述与未来展望

    The Evolution of Binary Decompilation in the Modern Era: A Taxonomy, Literature Review, and Future Perspectives

    [https://arxiv.org/abs/2608.24955](https://arxiv.org/abs/2608.24955)

    本文系统综述了现代反编译研究，提出分类法并指出基准缺失等挑战，展望了未来方向。

    

    arXiv:2608.24955v1 公告类型：新 摘要：反编译已成为软件工程和安全分析中的基础技术，目前正通过集成现代机器学习（ML）方法而不断进步。本文对过去几十年发表的反编译研究进行了系统性回顾，并开发了当代研究方法论的全面分类法。我们进一步考察了用于评估最新方法的评估指标、工具和基准的趋势。我们的综述揭示了关键挑战，如缺乏可靠的基准真相和标准化基准，这些阻碍了严格比较。最后，我们概述了未来的研究方向。

    arXiv:2608.24955v1 Announce Type: new  Abstract: Decompilation has become a foundational technique in software engineering and security analysis, and it is now advancing through the integration of modern machine learning (ML) approaches. This article presents a systematic review of decompilation studies published over the past decades and develops a comprehensive taxonomy of methodologies employed in contemporary research. We further examine trends in evaluation metrics, tools, and benchmarks used to assess state-of-the-art approaches. Our review reveals key challenges, such as the lack of reliable ground truth and the absence of standardized benchmarks, which hinder rigorous comparison. Finally, we outline future research directions.
    
[^28]: 秘密MCP：从网页截图生成有证据约束和上下文隔离的设计规格说明

    Secret MCP: Evidence-Bounded and Context-Isolated Design Specification Generation from Web Screenshots

    [https://arxiv.org/abs/2608.24944](https://arxiv.org/abs/2608.24944)

    秘密MCP是一个开源本地系统，通过分离检索、证据准备和模型调用，为每个网页截图生成有证据约束和上下文隔离的可审计设计规格，防止参考间污染。

    

    arXiv:2608.24944v1 公告类型：新 摘要：截图转代码系统优化了渲染实现，但截图遗漏了文档结构、交互逻辑、响应式规则以及区分观察与猜测所需的来源信息。多参考提示也存在风险，即一个参考可能被另一个参考的证据或推断污染。我们提出了秘密MCP，一个开源本地系统，为每个公共网页参考生成一份可审计的设计规格说明。它将检索、证据准备、模型调用、存储和检查分离。长截图被调整大小并重叠平铺；证据记录保留预处理和源空间坐标以及测量的调色板。一个19部分的合同涵盖页面清单、导航几何、响应式矩阵、组件、可访问性、验收标准，以及测量、观察、推断和未知声明的明确标签。参考通过采样器接口顺序处理。

    arXiv:2608.24944v1 Announce Type: new  Abstract: Screenshot-to-code systems optimize for rendered implementations, but screenshots omit document structure, interaction logic, responsive rules, and provenance needed to distinguish observation from guesswork. Multi-reference prompts also risk contaminating one reference with evidence or inferences from another. We present Secret MCP, an open-source local system that produces one auditable design specification per public web reference. It separates retrieval, evidence preparation, model invocation, storage, and inspection. Long captures are resized and tiled with overlap; evidence records preserve prepared- and source-space coordinates and a measured color palette. A 19-section contract covers page inventories, navigation geometry, responsive matrices, components, accessibility, acceptance criteria, and explicit labels for measured, observed, inferred, and unknown claims. References are processed sequentially through a sampler interface. 
    
[^29]: 从盲目编辑到验证修复：构建可信的用户侧LLM代理以提升网页无障碍性

    From Blind Edits to Verified Repair: Building Trustworthy User-Side LLM Agents for Web Accessibility

    [https://arxiv.org/abs/2608.24913](https://arxiv.org/abs/2608.24913)

    本文提出并评估了一个保护隐私的浏览器代理，通过本地LLM生成可逆CSS修复网页无障碍问题，并引入双条件协议，发现未经验证的修复在改进和回退上效果相近，强调了验证修复的必要性。

    

    arXiv:2608.24913v1 公告类型：交叉 摘要：在用户浏览网页时，于用户侧自适应修改网页的辅助代理，能够解决网站作者遗留的无障碍问题，而大语言模型使此类代理成为可能。我们为此目标贡献了三个构建模块。第一个是完整的、保护隐私的浏览器代理：一个Chrome扩展，它提取页面的样式表，将其压缩以适应本地模型的上下文窗口，请求模型根据WCAG和W3C认知无障碍指南中的18项指标生成附加CSS，并将结果可逆地注入实时页面。第二个是双条件协议，它像衡量益处一样仔细衡量危害，应用于六个小型开放权重模型（7B至14B）在十个违规密集和十个高度无障碍的实时网站上。诊断结果令人清醒但精确：未经验证的生成在改进和回退页面方面的比率相似（24次改进对20次回退）。

    arXiv:2608.24913v1 Announce Type: cross  Abstract: Assistive agents that adapt web pages on the user's side, at the moment of browsing, could reach the accessibility failures that site authors leave unfixed, and large language models make such agents newly plausible. We contribute three building blocks toward that goal. The first is a complete, privacy-preserving browser agent: a Chrome extension that extracts a page's style sheets, condenses them to fit a local model's context window, asks the model for additive CSS addressing 18 metrics from WCAG and the W3C cognitive accessibility guidance, and injects the result reversibly into the live page. The second is a dual-condition protocol that measures harm as carefully as benefit, applied to six small open-weight models (7B to 14B) on ten violation-rich and ten highly accessible live sites. The diagnosis is sobering but precise: unverified generation improved and regressed pages at similar rates (24 improvements against 20 regressions ac
    
[^30]: 自信地错误，沉默地如此：审计部署在设备上的语言模型的不可检测失败

    Confidently Wrong, Silently So: Auditing Undetectable Failures of a Deployed On-Device Language Model

    [https://arxiv.org/abs/2608.23663](https://arxiv.org/abs/2608.23663)

    该论文审计了设备端语言模型，发现其存在任务不对称的校准错误，即在错误前提上高置信虚构且对良性输入过度拒绝，且自报置信度不可靠，导致失败难以检测。

    

    arXiv:2608.23663v1 公告类型：交叉 摘要：对齐已部署的语言模型需要知道其输出何时可以被信任，然而如今设备端模型已搭载于数亿台设备上，且没有服务器端的审核，开发者实际能部署的配置很少被独立审计。我们针对开发者可访问的设备端基础模型进行了可复现的可靠性审计，并将其框定为监督问题：用户或资源受限的开发者能否判断模型何时出错？通过对校准、在错误前提问题上的自信虚构以及对良性提示的过度拒绝进行红队测试，我们发现了一种“任务不对称的校准错误”：其防护措施在不同任务上以相反方向失效（在69%的错误前提上虚构，同时拒绝18%的完全良性输入），而其自报置信度饱和且无区分度（AUROC 0.47；ECE 70，在可比的小型模型中表现最差）。关键在于，自信的虚构……

    arXiv:2608.23663v1 Announce Type: cross  Abstract: Aligning deployed language models requires knowing when their outputs can be trusted, yet on-device models now ship to hundreds of millions of devices with no server-side moderation, and the configuration developers can actually deploy is rarely audited independently. We present a reproducible reliability audit of the developer-accessible on-device foundation model, framed as an oversight question: can a user or a resource-constrained developer tell when the model is wrong? Red-teaming it on calibration, confident confabulation on false-premise questions, and over-refusal of benign prompts, we find a \emph{task-asymmetric miscalibration}: its guardrails fail in opposite directions across tasks (confabulating on 69\% of false premises while refusing 18\% of entirely benign inputs), atop a self-reported confidence that is saturated and non-discriminative (AUROC 0.47; ECE 70, worst among comparable small models). Crucially, confident-corr
    
[^31]: 重建档案：为智能体应用重建实施机械强制规格，以及模型层级失败所揭示的问题

    Rebuild Dossier: Mechanically-Enforced Specs for Agentic App Rebuilds, and What Model-Tier Failures Reveal

    [https://arxiv.org/abs/2608.23616](https://arxiv.org/abs/2608.23616)

    该论文提出了rebuild-dossier工具，通过预先锁定应用接口并强制执行逐测试构建，揭示了测试可被操纵时通过套件不能保证正确性，且在较大应用上不如简单指令方法。

    

    人工智能智能体的重建质量取决于其生成过程。先前的研究发现，一旦模型足够强大，多智能体重建流程反而会输给最简单的方法：将原始代码和一条指令直接交给模型（即AgentModernize方法）。我们提出了rebuild-dossier，这是一个开源工具，它在编写任何代码之前锁定应用程序的真实接口——即其精确的输入和输出——然后通过自动化检查而非仅依赖书面指令，强制执行一次只构建一个测试的流程。三项结果构成了本次评估，且证据程度各不相同。首先，在一项小型比较中，合规的智能体未能通过一个保留测试，而违反规则的智能体则全部通过——这证明了当测试可能被操纵时，通过的测试套件并不能保证正确性。其次，我们测试了这种方法是否优于简单地将源代码和一条指令交给较弱模型的方法：在小型应用上结果持平，但在较大应用上则完全落败。

    arXiv:2608.23616v1 Announce Type: cross  Abstract: An AI agent's rebuild is only as good as the process that produced it. Prior work found that once a model is strong enough, a multi-agent rebuild pipeline loses to the simplest approach: giving the model the original code and one instruction (AgentModernize). We present rebuild-dossier, an open-source tool that locks an application's real interface - its exact inputs and outputs - before any code is written, then enforces one-test-at-a-time building through automated checks, not written instructions alone.   Three results shape this evaluation, with differing amounts of evidence. First, in a small comparison, the compliant agent failed a held-back test while the rule-breaking agent passed everything - proof that a passing suite doesn't certify correctness when tests can be gamed. Second, we tested whether this beats simply giving the weaker model the source and one instruction: tied on a small app, but lost outright on a larger one whe
    
[^32]: 神经形式验证：智能体驱动的语言无关形式化程序推理

    Neuro-Formal Verification: Agentic Language-Agnostic Formal Program Reasoning

    [https://arxiv.org/abs/2608.21516](https://arxiv.org/abs/2608.21516)

    本文提出神经形式验证（NFV），通过AI智能体将主流语言代码翻译为形式化验证语言，实现一键式经验性准确验证，无需形式化方法专业知识。

    

    arXiv:2608.21516v1 公告类型：新 摘要：形式化验证为软件提供了最强有力的保证，而支持验证的语言已使其自动化成为现实。然而，这些好处惠及不到大多数主流开发者，他们使用的语言大多缺乏验证支持。此外，指定属性和建模环境需要形式化方法方面的专业知识。因此，证明仅限于少数著名的工件，而实际交付的生产代码仅通过审查和测试来证明。我们引入了神经形式验证（NFV），它利用这种自动化惠及主流编程语言的开发者：一个AI编码智能体负责翻译，一个成熟的验证器负责判定，以主流语言提出的问题通过按钮式操作得到回答，其准确性基于经验而非严格可靠性，并附有机检证明。在Python编程问题的正确与错误解决方案数据集上的结果令人鼓舞：NFV返回一个Dafny证明。

    arXiv:2608.21516v1 Announce Type: new  Abstract: Formal verification offers the strongest assurance available for software, and verification-aware languages have made its automation real. Yet the benefits reach few mainstream developers, most of whose languages have no verification support. Besides, specifying properties and modeling the environment require expertise in formal methods. Proof is therefore reserved for a few celebrated artifacts, while the production code that ships is attested only through review and testing.   We introduce neuro-formal verification (NFV), which harnesses that automation for developers of mainstream programming languages: an AI coding agent translates, an established verifier decides, and a question posed in a mainstream language is answered push-button, at empirical accuracy rather than soundness, with a machine-checked proof. Results on a dataset of correct and incorrect solutions to Python programming problems are encouraging: NFV returns a Dafny pro
    
[^33]: 拥有权威的AI：从应用到硅片

    AI with Authority, from Application to Silicon

    [https://arxiv.org/abs/2608.21356](https://arxiv.org/abs/2608.21356)

    本文展示了生成式AI如何通过机器验证成为不可腐蚀的裁判，使得单人在五周内指挥AI代理从应用代码到RISC-V硅片流片，全程无需人工审查或编写RTL，提出了盐方法这一新纪律。

    

    arXiv:2608.21356v1 公告类型：交叉 摘要：六十年来，机器验证一直是主要的成本开销，仅适用于特殊工件。在此我们报告，生成式AI逆转了这一关系：在AI速度下，机器验证不仅经济实惠，而且对生产力至关重要——它是不可腐蚀的裁判，让一个人能安全地大规模指挥自主机器工作。在五周内，一位使用消费级AI订阅的研究者，指挥一小队AI代理从应用代码出发，通过经过验证的编译器和执行器，最终在社区硅穿梭片上流片出一款RISC-V处理器；没有一项证明经过人工审查，也没有任何RTL代码由人类编写。这一工作纪律——盐方法——依赖于一个无法通过任何幻觉证明的证明核心：数学声明以核心检查的工件形式在代理之间传递，而人类的注意力则保留在陈述、设计和裁决上。验证被逐环陈述，从

    arXiv:2608.21356v1 Announce Type: cross  Abstract: For sixty years, machine verification has been a major cost overhead, affordable only for exceptional artifacts. Here we report that generative AI inverts this relationship: at AI speed, machine verification is not only economical but essential to productivity --- it is the incorruptible referee that lets one person safely direct autonomous machine work at scale. In five weeks, one researcher on consumer AI subscriptions directed a small fleet of AI agents from application code, through a verified compiler and executive, to a RISC-V processor taped out on a community silicon shuttle; no proof passed through human review, and no RTL was written by a human. The working discipline --- the Salt method --- rests on a proof kernel no hallucinated proof can pass: mathematical claims travel between agents as kernel-checked artifacts, and human attention is reserved for statements, designs, and rulings. Verification is stated link by link, from
    
[^34]: 编码代理架构解析：基于研究原型的探索性研究

    Understanding the Architecture of Coding Agents: An Exploratory Study Using a Research Prototype

    [https://arxiv.org/abs/2608.10934](https://arxiv.org/abs/2608.10934)

    本文首次系统化描述了编码代理的架构组件，并开发了极简开源代理Ark和轻量级基准ArkBench，验证了其有效性。

    

    摘要：arXiv:2608.10934v2 公告类型：替换 摘要：编码代理已迅速成为AI辅助软件开发的主要接口。然而，尽管其采用日益广泛，但对其内部架构知之甚少，目前尚无类似于编译器或操作系统可用的系统性架构描述。本文通过记录编码代理的主要架构组件，解释其职责、交互和执行流程，填补了这一空白。为支持这项工作，我们还介绍了Ark（代理研究工具包），一个为研究和教育设计的极简开源编码代理，它保留了现代编码代理的基本架构机制，同时强调简洁和清晰。我们还引入了ArkBench，一个包含十个代表性软件维护和进化任务的轻量级基准。使用gpt-5.4-mini，Ark成功解决了10个任务中的8个，同时需...

    arXiv:2608.10934v2 Announce Type: replace  Abstract: Coding agents have rapidly emerged as the primary interface for AI-assisted software development. However, despite their growing adoption, relatively little is known about their internal architecture, and no systematic architectural description comparable to those available for compilers or operating systems currently exists. This paper addresses this gap by documenting the main architectural components of coding agents, explaining their responsibilities, interactions, and execution flow. To support this effort, we also present Ark (Agent Research Kit), a minimal open-source coding agent designed for research and education that preserves the essential architectural mechanisms of modern coding agents while emphasizing simplicity and clarity. We also introduce ArkBench, a lightweight benchmark comprising ten representative software maintenance and evolution tasks. Using gpt-5.4-mini, Ark successfully solved 8 of the 10 tasks while requ
    
[^35]: SIGIL：将智能体技能编译为类型化框架

    SIGIL: Compiling Agent Skills into Typed Harnesses

    [https://arxiv.org/abs/2607.27309](https://arxiv.org/abs/2607.27309)

    SIGIL通过技能编译范式，将自然语言技能确定性编译为可执行代码，显著提升了智能体在执行过程中对技能要求的合规性。

    

    arXiv:2607.27309v2 公告类型：替换 摘要：智能体技能提供了一种可重用的方式来指定多步骤智能体行为，但它们仍然是模型在运行时解释的自然语言规范。因此，即使技能明确规定了所需的工具调用、顺序约束和检查，这些内容也可能被跳过。我们引入了技能编译（Skill Compilation）范式，它将自然语言技能转换为可执行的智能体程序，同时在需要语义决策的地方保留模型判断。我们通过SIGIL实现了这一理念。SIGIL提取基于来源的需求，使用封闭的智能体指令集（AIS）分解它们，将其组合成具有显式所有权、数据流和控制流的AG-IR，并确定性地将验证过的AG-IR降级为可执行代码。在33个公开可用的SKILL.md文件和三个运行时模型上，SIGIL将平均适用指令合规性（AMC）——即执行期间满足的适用技能要求的比例——提高了。

    arXiv:2607.27309v2 Announce Type: replace  Abstract: Agent skills provide a reusable way to specify multi-step agent behavior, but they remain natural-language specifications interpreted by the model at runtime. As a result, required tool calls, ordering constraints, and checks may be skipped even when explicitly prescribed by the skill. We introduce Skill Compilation, a paradigm that translates natural-language skills into executable agent programs while preserving model judgment where semantic decisions are required. We realize this idea in SIGIL. SIGIL extracts source-grounded requirements, decomposes them using a closed Agent Instruction Set (AIS), composes them into AG-IR with explicit ownership, data flow, and control flow, and deterministically lowers validated AG-IR into executable code. Across 33 publicly available SKILL.md files and three runtime models, SIGIL increases mean Applicable-Mandate Compliance (AMC), the fraction of applicable skill requirements satisfied during ex
    
[^36]: ClayBuddy：编码代理失败的框架、评估与缓解

    ClayBuddy: A Framework, Evaluation, & Mitigation of Coding Agent Failures

    [https://arxiv.org/abs/2606.19380](https://arxiv.org/abs/2606.19380)

    ClayBuddy通过分解编码代理失败为三种机制并提出框架修改，实现了针对性的安全缓解，显著提升了AI代理在软件工程中的可靠性。

    

    arXiv:2606.19380v5 公告类型：跨版本替换 摘要：AI代理在软件工程中的广泛部署正暴露出大量罕见但极其危险的错位缺陷。由于对此类行为进行采样是棘手的，我们将这些失败分解为三种不同的机制：规格不足（默认模型行为不安全）、能力错误（安全动作可用但模型未遵循）以及代理框架错误（安全动作执行失败）。在8项针对这些机制的强化测试评估中，我们发现前沿模型难以引导，仅通过3个条件示例即可引发危险行为，并且能以非平凡概率随机生成破坏性命令。这三种机制自然引出了针对每种机制的定向缓解措施，从而启发了我们的框架修改方案ClayBuddy。ClayBuddy允许代理修改其自身框架以提供安全保证，包括一种新颖的工具来选择性...

    arXiv:2606.19380v5 Announce Type: replace-cross  Abstract: Widespread deployment of AI agents in software engineering is surfacing a long tail of rare but highly dangerous misalignment bugs. Since sampling this behavior is intractable, we decompose these failures into three distinct mechanisms: underspecification, where default model behavior is unsafe; capability errors, where the safe action is available but the model does not adhere to it; and agent harness errors, where the safe action fails to execute. Across 8 evaluations that stress test these mechanisms, we find that frontier models are difficult to steer, elicit dangerous behavior through just 3 conditioning examples, and can randomly generate destructive commands at nontrivial probabilities. These three mechanisms naturally lend to targeted mitigations for each one, inspiring our harness modification ClayBuddy. ClayBuddy allows the agent to modify its own harness to provide safety guarantees, including a novel tool to selecti
    
[^37]: 基于大语言模型的算法开发：以张量网络收缩顺序优化的LLM应用为例

    Algorithmic algorithm development with LLMs: A Case Study on LLM-Usage for Contraction Order Optimization in Tensor Networks

    [https://arxiv.org/abs/2606.01975](https://arxiv.org/abs/2606.01975)

    本研究通过张量网络收缩顺序优化的案例，展示了LLM驱动的进化编码代理在算法开发中的潜力，并强调了人类科学家在评估与验证中的关键作用。

    

    arXiv:2606.01975v2 公告类型：替换交叉 摘要：我们通过一个关于张量网络收缩顺序优化的案例研究，探讨了基于大语言模型的算法开发，并使用了OpenEvolve工具。我们特别关注了LLM的选择以及评估指标和测试实例等设计选择。我们的结果既凸显了验证器引导的进化编码代理在算法开发/改进方面的潜力，也强调了人类科学家在评估、验证和解释方面持续的重要性及相应挑战。

    arXiv:2606.01975v2 Announce Type: replace-cross  Abstract: We consider LLM-based algorithm development through a case study on contractionorder optimisation for tensor networks with OpenEvolve. We pay particular attention to the choice of the LLM as well as design choices such as evaluation metric and test instances. Our results highlight both the promise of verifier-guided evolutionary coding agents for algorithm development/improvement and the continuing importance of evaluation, validation, and interpretation -- and corresponding challenges -- by the human scientist.
    
[^38]: ReLog：面向LLM调试的带有运行时反馈的执行感知日志记录

    ReLog: Execution-Aware Logging with Runtime Feedback for LLM-Oriented Debugging

    [https://arxiv.org/abs/2603.29122](https://arxiv.org/abs/2603.29122)

    ReLog提出了一种由运行时反馈驱动的迭代日志生成框架，利用LLM动态优化日志语句以提升下游任务的实际效用，而非仅追求与人类编写的日志相似。

    

    日志语句对于软件调试和维护至关重要。然而，现有的自动日志生成方法依赖于静态分析，并在单次传递中生成语句，而不考虑运行时行为。它们通常也通过与开发者编写的日志的相似性来评估，假设这些日志构成了一个充分的黄金标准。在LLM时代，这一假设越来越受限，因为日志不仅被开发者消费，也被LLM用于下游任务。因此，优化日志以匹配人类相似性并不一定反映其实际效用。为了解决这些局限性，我们引入了ReLog，一种由运行时反馈引导的迭代日志生成框架。ReLog利用LLM生成、执行、评估和优化日志语句，使运行时日志更好地支持下游任务。我们不是与开发者编写的日志进行比较，而是通过实际任务性能来评估ReLog。

    arXiv:2603.29122v3 Announce Type: replace  Abstract: Logging statements are essential for software debugging and maintenance. However, existing approaches to automatic logging generation rely on static analysis and produce statements in a single pass without considering runtime behavior. They are also typically evaluated by similarity to developer-written logs, assuming these logs form an adequate gold standard. This assumption is increasingly limiting in the LLM era, where logs are consumed not only by developers but also by LLMs for downstream tasks. As a result, optimizing logs for human similarity does not necessarily reflect their practical utility.   To address these limitations, we introduce ReLog, an iterative logging generation framework guided by runtime feedback. ReLog leverages LLMs to generate, execute, evaluate, and refine logging statements so that runtime logs better support downstream tasks. Instead of comparing against developer-written logs, we evaluate ReLog through
    
[^39]: 反应式笔记本何时无法响应？

    When Do Reactive Notebooks Fail to React?

    [https://arxiv.org/abs/2511.21994](https://arxiv.org/abs/2511.21994)

    本文提出Rex测试套件，评估三个反应式笔记本系统（Ipyflow、Marimo和Observable）的反应性缺陷，发现它们在不同定义下均存在简单修改可破坏系统的不一致问题。

    

    计算笔记本对程序员来说很方便，但由于能够增量编辑正在运行的程序，它们容易变得混乱和不一致。最近的反应式笔记本系统，如Ipyflow、Marimo和Observable，通过修改时重新执行最小单元格集，力求使笔记本状态与当前单元格代码保持同步。然而，每个系统以不同方式定义反应性。此外，在任何定义下，我们发现了可以破坏每个系统的简单笔记本修改。总体而言，这些不一致性使用户难以构建对其反应式笔记本实现的心理模型。本文提出了Rex，一个细粒度的测试套件，用于讨论和评估反应式笔记本系统中的反应能力。我们在三个现有反应式笔记本系统上评估了Rex，并对其故障进行分类，旨在（i）帮助程序员理解何时反应性失效。

    arXiv:2511.21994v2 Announce Type: replace-cross  Abstract: Computational notebooks are convenient for programmers, but can easily become confusing and inconsistent due to the ability to incrementally edit a program that is running. Recent reactive notebook systems, such as Ipyflow, Marimo and Observable, strive to keep notebook state in sync with the current cell code by re-executing a minimal set of cells upon modification. However, each system defines reactivity a different way. Additionally, within any definition, we find simple notebook modifications that can break each system. Overall, these inconsistencies make it difficult for users to construct a mental model of their reactive notebook's implementation. This paper proposes Rex, a fine-grained test suite to discuss and assess reactivity capabilities within reactive notebook systems. We evaluate Rex on three existing reactive notebook systems and classify their failures with the aims of (i) helping programmers understand when rea
    
[^40]: QLCoder：一种用于安全漏洞静态分析的查询合成器

    QLCoder: A Query Synthesizer For Static Analysis of Security Vulnerabilities

    [https://arxiv.org/abs/2511.08462](https://arxiv.org/abs/2511.08462)

    QLCoder是一个基于大语言模型的智能体框架，通过结合执行反馈、语言服务器协议和RAG数据库，能自动从CVE元数据生成有效的CodeQL安全查询，无需人工编写。

    

    静态分析工具通过指定编码脆弱代码模式的查询，提供了一种检测安全漏洞的强大手段。然而，编写此类查询具有挑战性，需要安全和程序分析方面的多样化专业知识。为解决这一挑战，我们提出了QLCoder——一个智能体框架，能够直接从给定的CVE元数据自动合成CodeQL（一种强大的静态分析引擎）中的查询。QLCoder将大语言模型嵌入到带执行反馈的合成循环中，并通过自定义的MCP接口约束其推理，该接口允许与语言服务器协议（用于语法指导）和RAG数据库（用于语义检索查询和文档）进行结构化交互。这种方法使QLCoder能够生成语法和语义上有效的安全查询。我们在111个Java项目的176个现有CVE上评估了QLCoder。基于Claude Code智能体框架构建。

    arXiv:2511.08462v5 Announce Type: replace-cross  Abstract: Static analysis tools provide a powerful means to detect security vulnerabilities by specifying queries that encode vulnerable code patterns. However, writing such queries is challenging and requires diverse expertise in security and program analysis. To address this challenge, we present QLCoder - an agentic framework that automatically synthesizes queries in CodeQL, a powerful static analysis engine, directly from a given CVE metadata. QLCode embeds an LLM in a synthesis loop with execution feedback, while constraining its reasoning using a custom MCP interface that allows structured interaction with a Language Server Protocol (for syntax guidance) and a RAG database (for semantic retrieval of queries and documentation). This approach allows QLCoder to generate syntactically and semantically valid security queries. We evaluate QLCode on 176 existing CVEs across 111 Java projects. Building upon the Claude Code agent framework,
    
[^41]: 基于补丁推理的软件代理可扩展监督方法

    Scalable Supervision for Software Agents via Patch Reasoning

    [https://arxiv.org/abs/2510.22775](https://arxiv.org/abs/2510.22775)

    本文提出R4P方法，通过补丁间推理和分组训练目标提供无需测试执行的密集奖励，实现软件代理监督的可扩展，并训练出无执行脚手架Mini-SE，显著提升性能。

    

    arXiv:2510.22775v2 公告类型：替换 摘要：尽管语言模型代理已经推动了软件工程的发展，但现有的基于测试的监督方式在现实问题上的可扩展性受到限制。原因有二：（1）高覆盖率的测试在现实环境中天然稀缺，（2）构建和运行测试沙箱既笨重又脆弱。为了解锁监督的扩展性，我们提出了R4P，一种基于推理的方法，提供与脚手架无关的奖励。R4P采用分组训练目标，使其能够针对彼此的修改验证多个补丁，并在不执行测试或依赖特定代理轨迹的情况下，为监督代理提供密集奖励。R4P在验证SWE-bench补丁时达到了72.2%的准确率，与专有模型相当。为了展示R4P的下游实用价值，我们设计并训练了一个无执行脚手架Mini-SE，通过R4P进行纯强化学习。Mini-SE实现了26.2%的Pass@1，相比原始Qwen3-32B提升了10.0%。

    arXiv:2510.22775v2 Announce Type: replace  Abstract: While language model agents have advanced software engineering, existing test-based supervision is limiting its scalability on real-world issues. The reason is twofold: (1) high-coverage tests are naturally rare in the wild, and (2) building and running test sandbox is heavy and fragile. To unlock supervision scaling, we propose R4P, a reasoning-based method that provides scaffold-agnostic rewards. R4P uses a group-wise training objective, enabling it to verify multiple patches against each other's modification and gain a dense reward for supervising agents without executing tests or relying on specific agent trajectories. R4P achieves 72.2% Acc. for verifying patches from SWE-bench, competitive with proprietary models. To show the downstream practical utility of R4P, we design and train an execution-free scaffold, Mini-SE, with pure RL via R4P. Mini-SE achieves 26.2% Pass@1, showing a 10.0% improvement over the original Qwen3-32B, a
    

