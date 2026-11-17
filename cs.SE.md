# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond Leaderboard Ranks: Selecting Software Engineering Agents Under Budget Constraints](https://arxiv.org/abs/2611.21515) | 该研究通过分析58,197条运行记录发现，排行榜排名无法可靠指导软件工程智能体的选择——在七个基准测试的准确率-成本帕累托前沿上，超过三分之二的步骤在统计上无法区分，且报告成本与标准化token估计可相差高达11.7倍。 |
| [^2] | [Is This Code Mine? Student Ownership When AI Agents Write the Spec and the Code](https://arxiv.org/abs/2611.21501) | 本研究通过对18组学生作业的实证分析发现，在规范驱动开发下AI智能体编写了69%的规范和绝大部分代码，学生虽主张对想法的所有权，但大多无法解释或独立修改代码，且其自述贡献与实际提交记录不符，揭示了AI时代编程作业中所有权与学习评估的深层问题。 |
| [^3] | [A Taxonomy and Taxonomy-Guided Benchmark Reduction for Spreadsheet Tasks](https://arxiv.org/abs/2611.21452) | 该论文提出了一个电子表格任务分类法，并据此设计了在预测能力相近时优先保证分类多样性的基准测试缩减方法，从而在几乎不损失预测准确性的前提下显著扩大任务覆盖范围。 |
| [^4] | [AuXSpec: Learning to Autoformalize Program Specifications via Executable Postconditions](https://arxiv.org/abs/2611.21421) | AuXSpec 通过在 Lean 中将声明式后置条件与可执行对应版本配对，使测试用例的执行一致性成为可扩展的强化学习奖励，从而低成本地训练程序规范的自动形式化模型。 |
| [^5] | [Harness Leverage Across Model Size: A Ladder of Five Local Agent Harnesses from 0.75B to 4.7B](https://arxiv.org/abs/2611.21407) | 本文的核心贡献在于首次同时变化智能体框架与模型规模，通过五个本地框架、多个开源小模型与确定性评分器构成的 360 格实验阶梯，证明了框架选择（尤其是 Mingbird）在从 0.75B 到 4.7B 的每个模型规模上都能带来显著且可复现的性能提升，而这一优势并非来自其任务完成门控机制。 |
| [^6] | [Premise Recovery in Software Design: What Does This Decision Rest On?](https://arxiv.org/abs/2611.21367) | 该论文提出“前提恢复”新任务，即在架构决策记录（ADR）未明确声明其依赖的先前决策时，自动从历史记录中识别出这些前提，并构建了首个包含 1,060 条 ADR 和 1,529 个标注配对的高质量基准数据集，用于评估大语言模型在该任务上的表现。 |
| [^7] | [Agent-Native by Design: A Programming Language for Agentic Applications](https://arxiv.org/abs/2611.21298) | 该论文提出“智能体原生编程”理念，并在Python超集语言Jac中通过"by llm()"、对象空间编程和"visit ... by llm()"等机制，将大语言模型操作、工作流结构与状态、模型引导路由统一在同一套执行语义下，从而简化智能体应用的开发并提升编译器对程序进行自动优化的能力。 |
| [^8] | [How Developer Personality Shapes the Use of AI Coding Assistants](https://arxiv.org/abs/2611.21240) | 本文通过系统性范围综述首次梳理了用户人格与基于LLM工具感知（尤其是感知有用性）之间的关联，指出软件工程领域尚缺相关研究，为面向开发者人格的AI编程助手个性化理论迈出第一步。 |
| [^9] | [Taking Time Out of TOCTOU: From Race Reproduction to Security Discovery](https://arxiv.org/abs/2611.21214) | Chronos框架通过seccomp、ptrace和KVM等多种机制将TOCTOU竞态攻击的干预时机与判定证据分离，从而实现对竞态漏洞的确定性复现、提交二分定位和CI安全回归测试。 |
| [^10] | [Scientific Coding with Large Language Models: Evidence from Finite Element Analysis](https://arxiv.org/abs/2611.21176) | 该论文将FEM-Bench扩展至17个涉及非线性力学与结构动力学的有限元编程任务，对15个大语言模型的评估表明，单次生成的正确率有限，而结合代码执行与修订的智能体框架能显著提升性能（最高达96.5% Pass@3），揭示了LLM在自动化科学编程中的潜力与局限。 |
| [^11] | [SAFE: Formal Specification and Implementation of a Scalable Analysis Framework for ECMAScript](https://arxiv.org/abs/2611.21148) | 本文提出了SAFE——首个同时提供ECMAScript形式化规范与开源实现的可扩展JavaScript分析框架，旨在帮助研究社区检测JavaScript应用程序的安全漏洞。 |
| [^12] | [AutoDiDact: Input Reduction over Inferred Structure](https://arxiv.org/abs/2611.20969) | 该论文提出AutoDiDact方法，证明结构化输入约简无需文法或内置格式知识，仅通过反复询问预言机“失败是否仍然存在”即可从失败输入中自动推断出中性词与定界符对，从而在标识符、定界组等结构单元层面高效约简输入。 |
| [^13] | [LEDGER: Auditable Agent Memory Under Change](https://arxiv.org/abs/2611.20927) | 提出了 LEDGER，一个可审计的智能体记忆层，由语言模型提出事实、显式规则决定当前有效值、可信来源和删除操作，并记录每个事实的来源与有效期，从而应对真实场景中事实变更、来源可靠性差异和用户遗忘请求等挑战。 |
| [^14] | [Re-Reading Is Not Free: Token Traces and Cache-Aware Energy Estimation for LLM Agents in Software Engineering](https://arxiv.org/abs/2611.20873) | 该论文基于十周真实AI辅助软件开发中近10亿token的分析，发现缓存上下文读取占LLM智能体token总量的94.8%（每生成一个token需读取约260个缓存token），并据此提出了纳入缓存复用与子智能体工作的缓存感知能耗估计模型。 |
| [^15] | [Synapse: Optimizing Natural-Language Agent Skills through Program Repair](https://arxiv.org/abs/2611.20856) | Synapse 将自然语言技能优化建模为对部分程序视图的“生成-验证”式程序修复，通过编译器生成中间表示 SkIR，结合执行轨迹与控制流/依赖分析来定位故障、提出类型化变异并用修复测试验证补丁，从而在不更新模型参数的情况下优化 LLM 智能体技能。 |
| [^16] | [Injectra: Retrofitting Extensibility into Web-Based Desktop Applications](https://arxiv.org/abs/2611.20713) | Injectra是一个通用的代码注入与功能增强系统，通过统一执行通道将JavaScript、HTML和CSS注入到基于Web技术（如Electron、Tauri）构建的桌面应用内部运行时环境中，实现对现有桌面客户端的界面修改、行为调整和功能扩展。 |
| [^17] | [Noise-Aware Testing of NISQ Quantum Programs: Self-Baseline Oracle with a Controlled False-Positive Rate](https://arxiv.org/abs/2611.20613) | 该论文提出了一种基于自基线校准的噪声感知NISQ量子程序测试框架，以可信参考程序在同一后端上的重复执行作为对照，并利用Jensen-Shannon散度有限采样零分布的分位数作为接受阈值，从而实现假阳性率的预先设定与可控。 |
| [^18] | [A Testing Method for Detector Error Models in Quantum Error Correction Compilers](https://arxiv.org/abs/2611.20501) | 提出 DEMVal，一种量子纠错编译器检测器错误模型的测试方法，通过精确比较编译器输出与外部重建的参考模型，实现与物理错误率无关的缺陷检测灵敏度。 |
| [^19] | [Embeddings vs. Grep: Rethinking the Dense Advantage in Code Search](https://arxiv.org/abs/2611.20455) | 该论文构建了贴近真实的仓库级代码搜索基准Repo-Premium，发现稠密检索相对BM25的优势主要源于偏向性的评估设置和查询缺乏罕见标识符，而通过LLM生成标识符扩展查询（generate-then-grep），BM25即可追平稠密检索的性能。 |
| [^20] | [Reliable Mining of Heterogeneous AI Coding-Agent Data: Context-Aware Prediction and Resource-Aware Ranking](https://arxiv.org/abs/2611.20297) | 该论文发现简单汇集异构的AI编程代理运行数据会引发辛普森悖论式的结论反转，提出用上下文感知的特征级联方法将来源、环境与工作流上下文纳入建模，从而解决数据真实性问题并将PR合并预测AUC从0.774提升至0.832。 |
| [^21] | [Learning From the Best: What Makes Popular Hugging Face Models?](https://arxiv.org/abs/2611.20227) | 该研究对66,909个Hugging Face模型进行系统分析，发现完善的文档质量（如更长的模型卡和更多的代码示例）等特征是模型获得高下载量和点赞数的关键因素。 |
| [^22] | [SWE-CoWork: Benchmarking Coding Agents as Digital Employees in Organizations](https://arxiv.org/abs/2611.20218) | 该论文提出SWE-CoWork基准，将编程智能体置于拥有26名同事的模拟工程组织中，要求其从分散渠道恢复需求、调和冲突证据并完整交付任务，评估发现领先模型虽能通过约63%的功能测试，但完整任务交付率不足8%，揭示了部分实现与完整交付之间的显著差距。 |
| [^23] | [How Specification Explicitness Shapes Failure in LLM-Generated Code](https://arxiv.org/abs/2611.20195) | 该研究通过72人参与的受控实验发现，轻量级结构规约对减少LLM生成代码的失败作用有限（仅8.7%），而明确写出约束、不变量、否定性需求、边界情况等行为义务可将失败减少47.2%，其中与隐含义务相关的失败模式降幅最大。 |
| [^24] | [The Hidden Methodological Layer: Uncovering Threats to Validity in LLM-Assisted Empirical Software Engineering](https://arxiv.org/abs/2611.20033) | 该研究通过对50项实证软件工程原始研究的二次分析发现，LLM辅助研究流程引入的效度威胁（如分类错误、提示词依赖性等）在现有文献中被严重低估报告。 |
| [^25] | [Anatomy of an Agent Harness: Decisions, Invariants and Corner Cases](https://arxiv.org/abs/2611.19960) | 本文首次将智能体框架中隐含于代码、鲜有记录的关键工程决策系统化为六组30张“决策卡”——每张包含选项与代价、可测试的不变量、破坏性边界情况、测试与证据标签——并配套提供带信任边界的系统模型、六类故障成因、基于脚本化提供者的测试方法与检查清单，为智能体框架的健壮性设计与测试建立了系统化方法论。 |
| [^26] | [Recovery Gone Too Far: Understanding and Repairing Risky Fallback in LLM Agents](https://arxiv.org/abs/2611.19886) | 本文首次定义并实证揭示了LLM智能体中的“风险回退”现象——即工具失败后改用替代工具时可能跨越行动边界造成严重危害，通过覆盖八个领域和55个真实CVE的304个场景实证研究发现主流模型的风险场景边界违反率高达96.8%，且现有缓解措施均无法有效应对，并提出了修复方案。 |
| [^27] | [Let Execution Do the Teaching](https://arxiv.org/abs/2611.19875) | 该论文提出以程序执行结果为核心训练信号的方法，将编译错误与测试反馈融入监督微调、C++修复课程和组相对强化学习中，并借助执行验证器生成结构化奖励，使 GLM-4.7-Flash 在 C++ 编程任务上的两轮成功率提升至 61.5%。 |
| [^28] | [Found, but Not Fixed : From Software Vulnerability Discovery to Fixes with Iterative LLM Pipelines](https://arxiv.org/abs/2611.19719) | 该论文提出了一个基于LLM迭代流水线的通用框架，通过多阶段反馈循环实现真实世界代码中漏洞函数的自动化高质量修复，并在GPT-4o、GPT-5.1和Gemini 3 Flash等最先进模型上验证了其有效性。 |
| [^29] | [Learning from Mutants: Teaching Small Language Models to Synthesise Loop Invariants](https://arxiv.org/abs/2611.19569) | 该论文提出一种受变异测试启发的失败驱动数据筛选方法，通过将基准真值不变式变异为导致验证失败的语义变异体并结合验证器反馈来构造高质量训练数据，从而成功微调小语言模型合成循环不变式。 |
| [^30] | [Inquest: Log-Guided Root-Cause Code Localization with a Staged Agent](https://arxiv.org/abs/2611.19554) | 提出INQUEST，一个使用27B开源模型的分阶段智能体，通过日志与相似历史issue等证据自动定位生产故障的根因代码，并基于38,001份真实issue报告构建了新基准。 |
| [^31] | [Beyond Prompts: Runtime Contracts for Coding Agents](https://arxiv.org/abs/2611.19549) | 该论文提出在编码智能体工作期间强制执行的可执行契约，实验证明相比提示词规则、检索式指导或生成后检查，运行时强制执行能完全消除规则违规且不损失功能正确性。 |
| [^32] | [CLEW: Log-Anchored Backward Tracing for Root-Cause Code Localization](https://arxiv.org/abs/2611.19448) | CLEW是一种结合规则式日志到代码锚定与LLM引导反向代码追踪的智能体方法，可自动定位软件故障的根因函数并输出带机制路径与日志证据的可解释诊断结果。 |
| [^33] | [Beyond Clumsy Agentic Tool Calling: How Cueing and Spotting Make the Difference](https://arxiv.org/abs/2611.19275) | 本文提出在线中间件ToolPulse，通过提示与识别机制监测智能体的笨拙工具调用与跨交互的工具使用弯路，并在调用前后及时注入提醒、结构化反馈或上下文，从而显著提升LLM智能体自主使用工具的能力。 |
| [^34] | [Small Harnesses, Default Checks, Limited Coverage: Bounded Model Checking in Practice](https://arxiv.org/abs/2611.19267) | 该论文对开源 C 与 Rust 项目中使用 CBMC 和 Kani 进行有界模型检测的实际实践开展了实证研究，刻画了验证属性、工件复杂度与验证结果，揭示了实践中存在测试桩规模小、依赖默认检查且覆盖有限等问题，为构建自动化软件验证系统提供了重要的实证基础。 |
| [^35] | [Teralizer: A Tool for Generalizing JUnit Tests into Property-Based jqwik Tests](https://arxiv.org/abs/2611.19242) | Teralizer通过单路径符号分析自动将JUnit测试泛化为基于属性的jqwik测试，在输入生成与输出预言两个层面增强测试，从而显著提升回归检测能力。 |
| [^36] | [Live Noise Fingerprints for Quantum Program Analysis](https://arxiv.org/abs/2611.19143) | 本文提出“噪声指纹”这一紧凑的抽象表示，并通过概率伽罗瓦连接将其与具体噪声行为关联，实现了对含噪量子程序的高效可靠分析，且多项式规模的接口可产生比完整噪声模型指数级更小的指纹。 |
| [^37] | [When the Mythical Man-Month Meets Agentic Web Systems: Structured Coordination for Scalable Development](https://arxiv.org/abs/2611.19140) | 本文借鉴软件工程经典著作《人月神话》的思想，系统揭示了多智能体系统中智能体规模与协调开销之间的非线性关系，并提出结构化协调方法以在任务性能与协调成本之间取得平衡，从而实现可扩展的智能体网络系统开发。 |
| [^38] | [Beyond Plain MCP: Do Decision Models and Code Mode Make Tool Use Cheaper and Faster?](https://arxiv.org/abs/2611.19123) | 该研究通过预注册控制实验证明，让智能体直接针对工具存根编写脚本的“代码模式”可大幅削减70%-90%的token消耗并显著提升任务效用，其中在存根中声明返回形状提示是关键因素，而在其前叠加决策层路由器并未带来显著额外收益。 |
| [^39] | [Beyond Benchmarks: Evaluating Large Language Models and Agentic Patterns for Static Code Analysis Warning Classification](https://arxiv.org/abs/2611.19086) | 本研究利用四个手动标注数据集评估三个当代大语言模型在真实场景下静态代码分析警告分类的性能，并通过引入智能体模式和既定编码标准来提升分类效果。 |
| [^40] | [Using machine learning approaches to classify false-positives generated by static analysis tools - An empirical study using small datasets](https://arxiv.org/abs/2611.19079) | 本研究在小数据集条件下系统评估了多种代码表示学习技术与机器学习算法来识别静态分析工具的误报，弥补了以往研究仅关注少量警报类型或评估不全面的不足，并发现最佳方法的选择取决于数据特征。 |
| [^41] | [More Context, Mixed Results: Evaluating LLMs for Identifier Recovery and Code Deobfuscation](https://arxiv.org/abs/2611.19043) | 该研究在51个Java项目上系统评估了大语言模型与传统方法在代码反混淆的标识符检测与语义恢复任务中的表现，发现LLM整体优于传统技术（检测F1最高达91.56%），但精确恢复率仍然偏低（总体峰值仅16.42%），且不同上下文形式对不同任务的效果存在差异。 |
| [^42] | [Context Matters: Improving LLM-Based Code Summarization with Hierarchical Information](https://arxiv.org/abs/2611.18964) | 该论文通过向大语言模型提供函数、类和模块三个层次的上下文信息，验证了层次化上下文增强能够显著改进代码级文档摘要的生成质量。 |
| [^43] | [From Language-Specific to Cross-DSL: Semantic-Role-Guided Model Completion with LLMs](https://arxiv.org/abs/2611.18952) | 本文提出了RAFiC——一种可跨DSL泛化的基于LLM的迭代式领域模型补全方法，通过元模型信息保证语法有效性，并借助语义角色标注丰富提示词以捕捉建模意图，在模型构建过程中提供上下文感知的增量式指导。 |
| [^44] | [Mining Unity VR Repositories for Refactoring Patterns and Technical Debt](https://arxiv.org/abs/2611.18948) | 该研究从176个开源Unity VR仓库中挖掘重构模式与技术债务，发现代码债务最为普遍，交互债务和UI/UX债务是VR应用的主要债务类别，并开发了VRRefMiner工具来检测Unity应用中的重构。 |
| [^45] | [Level Up! AI-Driven Challenges and Adaptations to Code Review Practices](https://arxiv.org/abs/2611.18939) | 通过对大型跨国软件公司2614名开发者的访谈与大规模问卷调查，本研究首次以实证方法揭示了AI时代代码审查面临的核心挑战及有效应对策略，发现AI正推动人类审查者角色向设计与意图等高层次关注转变，同时AI辅助编码因作者缺乏尽责管理而加重了审查负担。 |
| [^46] | [Between Rigor and Reality: How AI Safety Benchmark Developers Understand Benchmark Use and Maintenance](https://arxiv.org/abs/2611.18930) | 通过对学术界和产业界的AI安全基准测试开发者进行访谈，本研究揭示了安全基准测试的预期用途与实际部署系统安全评估之间的脱节，指出评分者间信度难以实现和外部效度威胁应对不清这两大挑战，并提出基准测试必须与部署环境主动融合。 |
| [^47] | [Revisiting Re-executability: Coverage-Guided Symbolic Feedback to Improve Decompilation](https://arxiv.org/abs/2611.18892) | SymDec系统利用覆盖率引导的符号执行对比原始二进制与重编译二进制的I/O行为，将暴露行为差异的输入作为结构化反馈指导LLM修复反编译代码，从而超越固定测试套件、真正确保行为保真。 |
| [^48] | [Benchmark Factory: Agentic Benchmark Creation via Dataset Curation and Label-Free Quality Anchors](https://arxiv.org/abs/2611.18890) | Benchmark Factory提出了一种带有人类检查点的自主元智能体循环，能够从生产代码变更中自动策划、验证和修复编码智能体基准测试，其质量锚点无需每项任务的人工标签。 |
| [^49] | [Metamorphic Testing of Quantum Software based on Stabilizer States](https://arxiv.org/abs/2611.18838) | 提出了一种量子蜕变测试（QMT）框架，将评估量子输出所需的测量过程融入蜕变关系中，从而解决了因量子态无法直接检查而导致的测试预言机难题。 |
| [^50] | [KernelArmor: End-to-End Agentic System for Establishing Memory and Thread Safety for CUDA](https://arxiv.org/abs/2611.18754) | KernelArmor是一个将LLM生成注释与扩展SMT验证器相结合的智能体框架，能够在显式契约下为使用张量核心等高级特性的CUDA内核建立内存与线程安全保证，并在智能体优化循环中生成经验证的内核。 |
| [^51] | [LLMs in DevSecOps: A Systematic Mapping of Models, Evidence, and Research Gaps Across the Software Lifecycle](https://arxiv.org/abs/2611.18632) | 本文对2024–2026年间LLM在DevSecOps六大阶段（计划、编码、测试、审查、部署、监控）的安全应用研究进行了系统性映射，揭示出研究分布不均衡——测试与编码阶段受到的关注远超部署与审查阶段，且近期系统日益依赖结构化上下文、迭代推理、微调、检索增强与智能体工作流等适配策略。 |
| [^52] | [RGFC: Registry Grounded Function Calling with stage-decomposed evaluation](https://arxiv.org/abs/2611.18622) | 该论文提出注册表规模函数调用基准 RFCD 与确定性阶段分解评估器 TAAG，并发现工具注册表规模（5 至 3,000 个函数）是模型函数调用性能退化的主要来源。 |
| [^53] | [AI Cannot Review What Humans Catch: A First Look at the Reviews of Agent Instruction Files](https://arxiv.org/abs/2611.18582) | 该论文首次研究了智能体指令文件的评审现状，通过分析 2,246 个仓库中的 2,870 个拉取请求发现，超过 70% 的指令文件变更缺乏实质性评审，而 AI 评审者虽能发现表面结构问题，却对事实错误、必要性与内容过时等最需要人类判断的问题是结构性失明的。 |
| [^54] | [Automatic techniques for security bug report identification: A systematic mapping study](https://arxiv.org/abs/2611.18581) | 本研究通过系统性映射综述了28项关于自动识别安全缺陷报告的研究，发现所用技术涵盖传统机器学习、深度学习和大语言模型，主要挑战在于标注数据缺乏和数据不平衡，且现有研究缺少真实工业环境中的评估。 |
| [^55] | [Model-Driven Approaches for DevOps: A Systematic Literature Review](https://arxiv.org/abs/2611.18580) | 本文是首个关于利用模型驱动工程（MDE）方法支持DevOps开发的系统性文献综述，通过对相关研究按计算领域、DevOps流程阶段和再工程步骤进行分类，发现当前研究主要聚焦于云计算领域。 |
| [^56] | [Uncertainty in Cyber-Physical Systems: A Literature Survey](https://arxiv.org/abs/2611.18579) | 该论文通过对186篇原始研究的文献综述，系统刻画了信息物理系统中不确定性的类型、来源、生命周期阶段及缓解策略，发现不确定性最常被视作保障问题，其中安全与规约不确定性最为普遍。 |
| [^57] | [Formal Modeling in Software Engineering with LLMs: A Systematic Literature Review](https://arxiv.org/abs/2611.18577) | 本文通过系统性文献综述，梳理了大语言模型在软件工程中自动化模型检测与形式化/半形式化模型生成方面的研究进展，从1,193篇候选文献中筛选并分析了34篇发表于2022至2026年间的同行评审论文。 |
| [^58] | [Measuring and Optimizing Energy Efficiency in ICT: A Multi-vocal Literature Review on Data Centers, Cloud Computing, and LLM-Driven Code Optimization](https://arxiv.org/abs/2611.18576) | 该论文通过整合学术与灰色文献的多声部综述，提出了ICT（尤其是数据中心与云计算环境）能耗测量与优化的分类体系和实践图谱，并综述了包括LLM驱动代码优化在内的软件层面能效优化策略。 |
| [^59] | [A Survey of Image-to-Code Generation for Software Engineering](https://arxiv.org/abs/2611.18575) | 本文是一篇综述，系统梳理了软件工程中图像到代码生成的研究文献，涵盖从界面截图、UML 图和流程图等视觉输入生成前端代码、程序修复与测试代码等方向的方法。 |
| [^60] | [Predictive Test Optimization Without Historical Failure Data](https://arxiv.org/abs/2611.18574) | 该论文提出了一种新颖的协同演化标注方法，利用版本历史中测试与代码共同变化的信息来训练预测性测试优化模型，无需依赖历史失败数据，其故障检测能力几乎可与基于失败数据的模型相媲美。 |
| [^61] | [LoopSkill: Evidence-Bound Lifecycle Control and the Safety-Liveness Tradeoff in Coding Agents](https://arxiv.org/abs/2611.18573) | LoopSkill 提出了一个面向编码智能体的证据约束生命周期控制平面（集成唯一写入者、持久化发件箱、工件绑定报告、独立评审与心跳门控终局等机制），并在 SWE-bench Lite 上的探索性实验中揭示了“安全性-活性”权衡——更严格的证据绑定控制反而降低了任务完成成功率。 |
| [^62] | [Where AI Use Is Normalized, Author Seniority but Not AI Disclosure Biases Code Review](https://arxiv.org/abs/2611.18572) | 在AI使用常态化的组织中，实验表明披露AI使用并不会使代码评审产生偏差，而作者的资历标签则会显著影响评审者对代码有效性和作者能力的判断。 |
| [^63] | [Loop Engineering: Version-Bound, Evidence-Gated Coordination for Production Coding Agents](https://arxiv.org/abs/2611.18571) | 该论文提出“循环工程”方法，通过触发器、上下文投影、权限边界、验证器、停止条件和回写规则六要素规范异步代理循环，并借助版本化技能契约、证据收集框架与版本绑定的人工审批门控，在不依赖私有对话交接的情况下，实现生产级编码代理在共享可审计仓库状态上的协调运行。 |
| [^64] | [Towards Sustainable Large Language Model Inference: A Systematic Literature Review](https://arxiv.org/abs/2611.18570) | 本文通过系统性文献综述，梳理了从输入、模型、系统到硬件各层面降低大语言模型推理能耗的技术，分析了能耗、延迟与准确性之间的权衡关系，并总结了用于能耗测量的工具、数据集与指标。 |
| [^65] | [Knowledge-Centric Automated Issue Resolution: A Systematic Survey and Taxonomy](https://arxiv.org/abs/2611.18569) | 本文对133项自动化问题解决研究进行了以知识为中心的系统性综述，提出了由背景知识层、仓库知识层和过程知识层构成的三层知识分类体系，并从六个维度分析了知识生命周期，发现当前系统主要依赖现有代码知识和开发工具知识。 |
| [^66] | [Use of Peer Code Review to Identify Potential Security Vulnerabilities in Open Source Software: A Systematic Literature Review](https://arxiv.org/abs/2611.18568) | 本系统性文献综述发现，同行代码评审能有效检测开源软件中局部且知名的漏洞（如缓冲区溢出、竞态条件和XSS），但常常遗漏微妙、复杂且依赖上下文的安全问题。 |
| [^67] | [Software Engineering Practice in the Virtual Reality/Augmented Reality Software Life Cycle: A Systematic Survey](https://arxiv.org/abs/2611.18567) | 该论文首次从软件工程视角对VR/AR软件生命周期相关研究进行系统性综述，填补了该领域缺乏全面总结的空白。 |
| [^68] | [Mining Architectural Models in the Wild: A Curated Dataset of AADL Models](https://arxiv.org/abs/2611.18566) | 该论文提出了A3D数据集——一个从GitHub挖掘、经OSATE验证、覆盖7个领域和52个问题集群、并具有领域/问题标签与参考架构组件映射双标注层的1,115个AADL模型精选数据集，填补了软件架构实证研究中大规模标注数据集的空白。 |
| [^69] | [ProofCoder: Continuous Reasoning for Automated Software Verification](https://arxiv.org/abs/2611.14106) | ProofCoder从编码智能体视角出发，让模型在整个证明搜索过程中持续推理，并根据证明助手的反馈自主决定信息收集、证明状态检查、片段修改和回溯，突破了传统LLM方法僵化策略生成循环的限制，实现了面向Rocq的自动化证明合成。 |
| [^70] | [Unprompted coordination through shared memory in LLM agents: a locked two-implementation replication and extension](https://arxiv.org/abs/2611.12412) | 本研究通过双重独立实现的复现与扩展首次证实：即使提示中完全没有提及其他智能体、共享或协作的存在，LLM智能体也能仅凭共享内存状态自发产生协调行为。 |
| [^71] | [Safe by Accident: Crash-Testing LLM-Written Code](https://arxiv.org/abs/2611.10585) | 该论文通过对 9,100 个 LLM 生成的后端代码进行崩溃测试发现，54% 在 sqlite3 默认事务模式下能通过崩溃测试的数据库代码，在自动提交模式下会留下半更新状态，证明现有“单次运行、单一环境”的代码评估把通过归功于环境而非代码本身的安全性。 |
| [^72] | [What extraction cannot slim, and whether function injection can: a pre-registered study of 580 Java repositories](https://arxiv.org/abs/2611.10545) | 该预注册研究对580个Java代码库中超过300万个方法进行扫描，发现22.9%的长方法无法通过单出口提取重构来精简，并验证OfficeFloor框架的函数注入技术能够拆分这类提取重构无法处理的长方法。 |
| [^73] | [BOXBOX: Recursive Harness Self-Improvement for Long-Running Performance Optimization](https://arxiv.org/abs/2611.09218) | BOXBOX让智能体能够在长期运行的任务中递归地自我改进自己的执行框架（通过智能体触发的框架修订与内核管理的热重载机制），并在PerfRace基准上以匹配的测试预算下比冻结框架、离线框架演化和框架扩展方法取得更高的加速比和效率。 |
| [^74] | [Rethinking Access Control for AI Agents: Recent Advances and Open Challenges](https://arxiv.org/abs/2611.06764) | 本文系统综述了AI智能体访问控制在策略规范、身份委托、动作执行和信息流控制四大领域的最新进展，并重新思考了授权机制的四个核心方面，指出了未来亟待解决的开放性挑战。 |
| [^75] | [IssueLighter: An Issue Type-Aware LLM Agent for Issue Report Validity Assessment](https://arxiv.org/abs/2611.02721) | 该论文提出IssueLighter，一种感知问题类型的LLM智能体，通过全面理解各类问题的无效原因来评估开源项目问题报告的有效性，并提供有充分依据的理由，从而减轻维护者的负担。 |
| [^76] | [What reviewers call a Long Method is a length: A pre-registered test of coupling divergence as a god-unit signal, and what it settled instead](https://arxiv.org/abs/2611.01636) | 预注册测试表明，用耦合散度加权复杂度无法区分上帝单元与复杂算法；该研究真正证实的是，MLCQ审阅者所称的“长方法”仅仅是一个绝对行数长度（26行及以上的阈值在确认集上达到MCC 0.777）。 |
| [^77] | [Software Change Request in AI-Synthesized Software Evolution](https://arxiv.org/abs/2611.00294) | 本综述系统梳理了利用规则、传统机器学习、主题建模及深度学习方法从应用商店用户评论中自动提取软件变更请求的研究现状，并指出不同研究在数据集、标注规范和评估指标上缺乏一致性，导致结果难以比较。 |
| [^78] | [Yesterday's Code, Today's Dependencies: How Dependency Selection Shapes PHP Migration Benchmarks](https://arxiv.org/abs/2610.41263) | 该论文发现，在PHP迁移基准测试中，采用发布日期依赖选择策略相比现代选择策略可将PHP 7.4测试通过率提高7.1个百分点，表明依赖版本的选择方式会显著影响迁移基准测试的结果。 |
| [^79] | [Skeleton Coding: Program Structure as a Specification for Agents](https://arxiv.org/abs/2610.38214) | 论文提出“骨架编码”概念，指出智能体在行为规约选定程序之前会先确定程序结构，而测试等规约是否充分取决于该结构本身，因此开发者应将程序结构（数据类型、模块与函数的分解）作为一种规约提供给智能体。 |
| [^80] | [Who Verifies the Tools? How Warped Formatters Mislead Agents and Their Verifiers](https://arxiv.org/abs/2610.32110) | 论文通过生产RAG系统的真实案例与对照实验发现，工具输出格式化器遗漏关键信息会误导智能体和验证器，而仅需在输出中补全这些缺失细节，就能显著减少智能体的无依据声称。 |
| [^81] | [QuSema: Detecting Silent Bugs in Quantum Libraries via Quantum-knowledge-enhanced Agents](https://arxiv.org/abs/2610.10258) | QuSema提出了一种自主测试智能体，将量子语义与文档约束作为源代码级语义预言机，能够在缺乏基于执行的预言机的情况下检测量子库中从有效输入产生无效输出的静默缺陷。 |
| [^82] | [Finding Blind Spots in AppWorld and WorkArena Task Verifiers](https://arxiv.org/abs/2610.09142) | 该论文通过基于源码信息的变异测试审计了AppWorld和WorkArena的已发布任务验证器，揭示其存在盲点——即使智能体产生了错误效果（如重复写入创建多余记录或遗留非默认持久化值），验证器仍会判定任务成功。 |
| [^83] | [KernelZero: Co-Evolving Proposer and Coder for Continuously Improved GPU Kernel Generation](https://arxiv.org/abs/2609.33074) | KernelZero提出一个共同演化框架，通过提议者与编码器的协同进化、前沿驱动的模块生成机制以及正确性感知的CA-GRPO强化学习方法，持续提升GPU内核自动生成的正确性与性能。 |
| [^84] | [When Tool Calls Succeed but Workflows Fail: Anomalies at the Agent-Tool Boundary](https://arxiv.org/abs/2609.15397) | 该论文提出了一个将外部世界事件与运行时观察相分离的效果历史模型，并系统梳理出八种在智能体-工具边界处反复出现的外部效果异常，以解决工具调用成功但工作流整体失败的不一致问题。 |
| [^85] | [Claude Code Complete User Handbook](https://arxiv.org/abs/2608.26742) | 本书提出，在Claude Code代理环境中，安全高效运行的关键在于定义明确的可观察完成条件，并通过指令、权限、沙箱和系统隔离四层控制栈来管理系统性风险。 |
| [^86] | [SPECMINE: A Large-Scale Corpus of Spec-Driven Development Artifacts](https://arxiv.org/abs/2608.25202) | 我们提出了SPECMINE，这是首个大规模语料库，通过两次普查系统地捕捉了GitHub上规范驱动开发工件，为研究规范如何转化为代码提供了基础数据。 |
| [^87] | [eIRWR: Enhanced Iterative Random Walk with Restart for Scalable Root Cause Analysis in Microservices](https://arxiv.org/abs/2608.08073) | 本文发现基本IRWR的“弹性阻尼”等价于调整重启概率的个性化PageRank，并提出eIRWR，通过幂律锐化将重启质量集中于最可疑节点，实现微服务系统中更准确且可扩展的根因分析。 |
| [^88] | [LangChoiceBench: Measuring and Explaining Programming-Language Choice in LLMs](https://arxiv.org/abs/2608.06041) | 提出了LangChoiceBench基准，系统测量25个大语言模型在28个项目中的编程语言选择行为，发现模型普遍过度偏好Python且多为未经深思熟虑的自动选择，较小的开放权重模型偏好更强、语言多样性更低。 |
| [^89] | [Change2Task: From Repository Changes to Executable Coding Agent Tasks and Environments](https://arxiv.org/abs/2607.28591) | Change2Task 提出了一个基于仓库历史的系统，能够将已合并的拉取请求自动转换为健康现代代码版本上经过验证的可执行编码智能体任务与环境，为编码智能体的训练、基准测试和持续评估提供可扩展的数据供给。 |
| [^90] | [Code-Anchored Review Feedback Improves Correctness and Lowers Hosted Editing Cost on Large R Files](https://arxiv.org/abs/2607.12713) | 研究表明，将代码评审反馈以“代码锚定”格式（每条评论锚定并引用其涉及的代码）而非整体列表形式提供给大模型，可使大型R文件编辑的输出令牌最多减少50.5%、托管成本最多降低46.3%，并显著提升编辑的正确性。 |
| [^91] | [Flex: A Foundational Proof-Producing Horn Solver](https://arxiv.org/abs/2607.12226) | Flex是一个基于Lean元编程实现的可产生内核验证证明的约束Horn子句（CHC）求解器，它将谓词合成与证明构造相结合，使规范可直接使用原生定义，并允许用户在自动化失败时补全证明。 |
| [^92] | [Mining Workflow Graphs for Black-Box Boundary Testing of Conversational LLM Agents](https://arxiv.org/abs/2607.06873) | 提出了AgentEval黑盒测试框架，通过与智能体交互挖掘其对话工作流图，并在各节点重放对话路径后施加扰动，从而系统性发现并测试身份验证、确认步骤等工作流边界处的故障。 |
| [^93] | [AutoCedar: An Agentic Framework for Verifier-Guided Access Control Policy Synthesis](https://arxiv.org/abs/2607.03656) | AutoCedar 提出了一种验证器引导的智能体框架，通过将访问控制需求分解为可审查、可机械验证的“意图原子”，并利用验证器对照经人工审查的目标迭代修复策略，确保从自然语言合成的 Cedar 访问控制策略真正符合用户意图。 |
| [^94] | [Is Agentic AI Ready for Real-World Hardware Engineering? A Deep Dive with Phoenix-bench](https://arxiv.org/abs/2605.15226) | 该论文提出了Phoenix-bench——首个同时要求仓库导航、层次感知定位、EDA可执行验证和维护式补丁的硬件工程智能体基准，包含511个经核实的Verilator实例，并系统评估了商业与开源智能体在真实硬件工程任务上的能力。 |
| [^95] | [ORACLE-SWE: Quantifying the Contribution of Critical Information Signals on SWE Agents](https://arxiv.org/abs/2604.07789) | 该论文提出ORACLE-SWE，首次系统量化了复现测试、回归测试、编辑位置、执行上下文和API用法等关键中间信息信号对软件工程代理任务成功的贡献，并测量了代理发现的信号与理想神谕信号之间尚存的改进空间。 |
| [^96] | [QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](https://arxiv.org/abs/2604.05963) | 该论文提出PRepair框架，通过Self-Breaking缺陷程序生成和基于编辑感知奖励的EA-GRPO训练，抑制大语言模型代码修复中的过度编辑问题，实现最大化复用正确代码、仅修复缺陷部分的精确修复，修复精确度最高提升31.4%。 |
| [^97] | [Automated Code Efficiency Optimization via Agent Skill Mining](https://arxiv.org/abs/2603.27850) | 本文提出EffiSkill两阶段框架，将慢代码到快代码的常见转换挖掘并沉淀为记录意图、适用性、流程、效果与风险的“算子技能”卡片库，使LLM无需执行反馈即可诊断程序、检索技能并生成多个候选优化方案，从而自动提升代码效率。 |
| [^98] | [RepoLaunch: Automating Build and Management of Code Repositories across Languages and Platforms](https://arxiv.org/abs/2603.05026) | RepoLaunch是一个跨多种编程语言和操作系统的智能体框架，能够自动解决依赖、编译源代码并提取测试结果，构建成功率达78%（超越仅支持Python/Linux的先前系统18%），并支撑了仅需任务设计阶段人工输入的全自动SWE数据集创建流水线。 |
| [^99] | [Clawdrain: Exploiting Tool-Calling Chains for Stealthy Token Exhaustion in OpenClaw Agents](https://arxiv.org/abs/2603.00902) | Clawdrain是一种木马化的OpenClaw技能，它通过分段验证过程（SVP）利用工具调用链制造高成本的多轮调用，在保持最终任务输出合法的同时隐蔽地耗尽LLM智能体的令牌资源，实现2-16倍的令牌放大。 |
| [^100] | [Will It Survive? Deciphering the Fate of AI Agent-Authored Code in Open Source](https://arxiv.org/abs/2601.16809) | 本研究通过对215个开源仓库中约65万行代码的生存分析发现，AI智能体编写的代码并不比人类代码更快被修改，甚至在同等条件下存活时间更长，从而有力反驳了“智能体代码是一次性的”这一担忧。 |
| [^101] | [VeruSAGE-Bench: A Benchmark Suite for Rust System Verification](https://arxiv.org/abs/2512.18436) | 该论文构建了首个面向真实系统级形式化验证的基准测试套件VeruSAGE-Bench，包含从八个经Verus验证的开源Rust系统中提取的849个证明任务，其形式化规约规模平均是以往基准的20倍以上，并用它系统评估了六个大语言模型的软件系统证明能力。 |
| [^102] | [Comparative Evaluation of Embedding Representations for Financial News Sentiment Analysis](https://arxiv.org/abs/2512.13749) | 本研究在仅有349条金融新闻标题的小数据集上，比较评估了Word2Vec、GloVe和句子变换器嵌入结合梯度提升的情感分类效果，发现预训练嵌入在数据不足时收益递减，且小验证集易导致模型选择中的过拟合。 |
| [^103] | [A Survey of Bugs in AI-Generated Code](https://arxiv.org/abs/2512.05239) | 本文通过系统性文献综述，首次全面梳理了AI生成代码中缺陷的类型、模式、分布及其与特定AI编程系统的关联和相应的修复策略，揭示了缺陷在AI生成代码中的普遍性。 |
| [^104] | [Beyond Synthetic Benchmarks: Evaluating LLM Performance on Real-World Class-Level Code Generation](https://arxiv.org/abs/2510.26130) | 该论文提出了基于真实开源仓库构建的RealClassEval基准，用于评估大语言模型在类级别代码生成上的真实能力，揭示了此前被合成基准所掩盖的模型根本性局限。 |
| [^105] | [The EmpathiSEr: Development and Validation of Software Engineering Oriented Empathy Scales](https://arxiv.org/abs/2510.12546) | 该研究开发并验证了首个专门面向软件工程情境的共情测量量表EmpathiSEr，填补了软件工程领域缺乏有效共情测量工具的空白。 |
| [^106] | [Oops!... I did it again. Analysing and Handling Conclusion (In-)Stability in Socio-Technical Software Engineering](https://arxiv.org/abs/2510.06844) | 本研究通过用四个独立挖掘工具正式复现三项已发表的社会技术软件分析研究，揭示了不同工具流水线在数据、结果与结论上的不稳定性，并为研究者和从业者提供了应对此类效度威胁的可操作建议。 |
| [^107] | [UniSage: A Unified and Diagnosis-Driven Sampling Framework for Microservice Systems](https://arxiv.org/abs/2509.26336) | UniSage是一个面向微服务系统的统一且诊断驱动的采样框架，它通过将轻量级分析集成到采样流水线中指导数据保留决策，并统一处理追踪与日志的采样，从而避免“先采样后分析”范式丢失关键诊断证据、以及追踪与日志上下文关联被破坏的问题。 |
| [^108] | [OpenClassGen: A Large-Scale Corpus of Real-World Python Classes for LLM Research](https://arxiv.org/abs/2504.15564) | OpenClassGen是一个从约3,000个开源项目中提取的包含32万余个真实Python类的大规模语料库，每个类均配有自包含的骨架和27项静态代码指标，为类级别代码生成的训练与评估提供了迄今最大规模的资源。 |

# 详细

[^1]: 超越排行榜排名：预算约束下的软件工程智能体选择

    Beyond Leaderboard Ranks: Selecting Software Engineering Agents Under Budget Constraints

    [https://arxiv.org/abs/2611.21515](https://arxiv.org/abs/2611.21515)

    该研究通过分析58,197条运行记录发现，排行榜排名无法可靠指导软件工程智能体的选择——在七个基准测试的准确率-成本帕累托前沿上，超过三分之二的步骤在统计上无法区分，且报告成本与标准化token估计可相差高达11.7倍。

    

    采用LLM智能体进行问题解决的团队必须决定为哪个智能体付费，以及要花多少钱去弄清楚这个问题。排行榜排名为这一决策提供的证据很薄弱：准确率差异可能在统计上无法确定，报告的成本可能不一致，而评估候选智能体本身也可能代价高昂。我们利用从全面智能体排行榜（Holistic Agent Leaderboard, HAL）中提取的58,197条运行-任务记录，研究预算约束下的智能体选择问题，重点关注SWE-bench及另外六个基准测试。研究结果表明：在七个基准测试中，准确率-成本帕累托前沿上的35个步骤中有27个在统计上无法区分；在SWE-bench上，一次成本为34.16美元的运行比170.78美元的领先者低14.6个百分点，但现有任务无法在统计上区分二者的准确率；此外，报告的成本与标准化的基于token的估计相差高达11.7倍，而选择性任务评估可能会使全基准执行成本的估计产生偏差。

    arXiv:2611.21515v1 Announce Type: new  Abstract: Teams that adopt LLM agents for issue resolution must decide which agent to pay for, and how much to spend finding out. Leaderboard ranks give weak evidence for this decision. Accuracy differences may be statistically unresolved, reported costs may be inconsistent, and evaluating candidate agents can itself be expensive. We investigate budget-constrained agent selection using 58,197 run--task records extracted from the Holistic Agent Leaderboard (HAL), focusing on SWE-bench and six additional benchmarks.   Across seven benchmarks, 27 of 35 steps along the accuracy--cost Pareto frontiers are statistically unresolved. On SWE-bench, a run costing $34.16 trails the $170.78 leader by 14.6 percentage points, but the available tasks cannot statistically distinguish their accuracy. Reported costs also differ from standardized token-based estimates by up to 11.7x, while selective task evaluation can bias estimates of full-benchmark execution cost
    
[^2]: 这段代码是我的吗？当AI智能体编写规范与代码时学生的成果所有权问题

    Is This Code Mine? Student Ownership When AI Agents Write the Spec and the Code

    [https://arxiv.org/abs/2611.21501](https://arxiv.org/abs/2611.21501)

    本研究通过对18组学生作业的实证分析发现，在规范驱动开发下AI智能体编写了69%的规范和绝大部分代码，学生虽主张对想法的所有权，但大多无法解释或独立修改代码，且其自述贡献与实际提交记录不符，揭示了AI时代编程作业中所有权与学习评估的深层问题。

    

    软件工程课程开始要求学生使用AI编码智能体构建完整的系统，但学生实际上如何使用这些智能体，以及他们是否仍然拥有自己交付的成果，目前知之甚少。如果智能体既编写了代码，又做出了代码背后的决策，那么我们用来评分的作业成果就不再能反映学生究竟学到了什么。我们研究了一项作业任务，其中18组学生（两人一组）在规范驱动开发下与智能体协作构建应用程序，研究结合了对他们代码仓库历史的挖掘、他们记录的提示词，以及对33名学生的调查。结果显示，智能体编写了69%的、本应让学生保持掌控权的规范，有时仅凭几个词的提示就能完成。学生将自己的所有权定位于想法层面，认为自己贡献了一半的想法，但只有五分之一的代码；然而大多数人无法向同事解释这些代码，也无法在没有智能体的情况下修改代码，而且他们自述编写的内容与其提交记录所显示的情况并不相符。

    arXiv:2611.21501v1 Announce Type: new  Abstract: Software engineering courses are starting to ask students to build entire systems with AI coding agents, but little is known about what students actually do with them and whether they still own what they deliver. If the agent writes both the code and the decisions behind it, the artifacts we grade no longer show what students learned. We studied an assignment in which 18 pairs of students built applications with agents under Spec-Driven Development, combining the mined history of their repositories, the prompts they recorded, and a survey of 33 students. The agents wrote 69\% of the specifications meant to keep students in control, sometimes from a prompt of a few words. Students locate their ownership in the ideas, crediting themselves with half of them but a fifth of the code, yet most could not explain that code to a colleague or modify it without the agent, and what they report having written is unrelated to what their own commits de
    
[^3]: 一种电子表格任务分类法及基于分类法引导的基准测试缩减方法

    A Taxonomy and Taxonomy-Guided Benchmark Reduction for Spreadsheet Tasks

    [https://arxiv.org/abs/2611.21452](https://arxiv.org/abs/2611.21452)

    该论文提出了一个电子表格任务分类法，并据此设计了在预测能力相近时优先保证分类多样性的基准测试缩减方法，从而在几乎不损失预测准确性的前提下显著扩大任务覆盖范围。

    

    电子表格让数百万没有接受过编程训练的最终用户能够完成计算工作，越来越多的团队正在构建 AI 智能体来帮助他们。在众多任务上评估每个候选智能体配置的成本高昂。基准测试缩减可以通过选择一小组任务（其结果能够预测完整基准测试的得分）来降低这一成本，但仅以预测准确性为标准来选择任务，可能会使整个类别的电子表格工作得不到测试。我们贡献了一个电子表格任务分类法，该分类法基于对基准测试的审查，并与微软的 Excel Agent 产品团队共同完善；同时我们提出了一种基准测试缩减方法，该方法在预测能力相近的选择中优先考虑分类多样性。在微软的基准测试日志上，分类法引导的任务选择在几乎不损失甚至不损失预测准确性的情况下扩大了覆盖范围，并且通常能提升对未纳入选择集的配置的预测效果。因此，更广的覆盖范围可以支持而非损害（预测质量）……

    arXiv:2611.21452v1 Announce Type: new  Abstract: Spreadsheets let millions of end users without programming training do computational work, and teams are increasingly building AI agents to help them. Evaluating each candidate agent configuration across many tasks is costly. Benchmark reduction can lower this cost by selecting a small subset of tasks whose outcomes predict full-benchmark scores, but selecting tasks only for predictive accuracy can leave whole categories of spreadsheet work untested. We contribute a taxonomy of spreadsheet tasks, grounded in a review of benchmarks and refined with the Excel Agent product team at Microsoft, and a benchmark-reduction method that favors taxonomic diversity among similarly predictive choices. On benchmark logs from Microsoft, taxonomy-guided selection broadens coverage at little or no cost in predictive accuracy and often improves prediction on configurations excluded from selection. Broader coverage can therefore support, rather than compro
    
[^4]: AuXSpec：通过可执行后置条件学习程序规范的自动形式化

    AuXSpec: Learning to Autoformalize Program Specifications via Executable Postconditions

    [https://arxiv.org/abs/2611.21421](https://arxiv.org/abs/2611.21421)

    AuXSpec 通过在 Lean 中将声明式后置条件与可执行对应版本配对，使测试用例的执行一致性成为可扩展的强化学习奖励，从而低成本地训练程序规范的自动形式化模型。

    

    形式化规范定义了程序的预期行为，并构成了程序验证的基础，但在诸如 Lean 这样的定理证明器中编写它们既缓慢又需要专业知识，这使得规范编写成为规模化产出形式正确代码的主要瓶颈。从自然语言问题出发的自动形式化很有前景，但训练规范模型需要对生成的规范是否捕捉了预期行为提供可扩展的反馈。先前的方法通过显式证明或带有形式对应性证明的单独实现代码来获得这种反馈，两者代价都很高。我们提出了 AuXSpec，这是一个 Lean 框架，它将声明式后置条件与可执行对应版本配对——在可能时通过类型类解析获得可执行版本，否则单独定义并提供等价性证明。随后，测试用例可以通过执行来评估声明式规范，使我们能够以规范间的一致性作为强化学习的奖励。利用 AuXSpec，我们合成了……

    arXiv:2611.21421v1 Announce Type: cross  Abstract: Formal specifications define intended program behavior and underpin verification, but writing them in theorem provers such as Lean is slow and expertise-intensive, making specification a major bottleneck to scaling formally correct code. Autoformalization from natural-language problems is promising, yet training specification models requires scalable feedback on whether generated specifications capture intended behavior. Prior approaches obtain this feedback through explicit proofs or separately implemented code with formal correspondence proofs, both expensive. We introduce AuXSpec, a Lean framework that pairs declarative postconditions with executable counterparts, obtained through type class resolution when possible and otherwise defined separately with an equivalence proof. Test cases can then evaluate declarative specifications by execution, letting us use agreement as a reinforcement-learning reward. Using AuXSpec, we synthesize 
    
[^5]: 跨模型规模的 Harness 杠杆效应：从 0.75B 到 4.7B 的五级本地智能体框架阶梯

    Harness Leverage Across Model Size: A Ladder of Five Local Agent Harnesses from 0.75B to 4.7B

    [https://arxiv.org/abs/2611.21407](https://arxiv.org/abs/2611.21407)

    本文的核心贡献在于首次同时变化智能体框架与模型规模，通过五个本地框架、多个开源小模型与确定性评分器构成的 360 格实验阶梯，证明了框架选择（尤其是 Mingbird）在从 0.75B 到 4.7B 的每个模型规模上都能带来显著且可复现的性能提升，而这一优势并非来自其任务完成门控机制。

    

    Harness（智能体框架）将模型转化为一个能够完成任务的系统：它构建提示词、暴露工具，并决定工作何时算完成。以往的框架对比固定模型，模型对比固定框架，而本研究同时对两者进行变化：五个已发布的框架、18 个产出工件的任务、四个可本地部署的 0.75B 至 4.7B 开源权重模型，以及一个 35.5B 的锚点模型，并由一个确定性检查器对全部 360 个阶梯组合进行评分。控制任务完成的框架 Mingbird 在每个模型规模上都取得了最高的平均分。在四个预先注册的配对对比中，有两个在 Holm 校正后依然显著，分别位于 0.75B 和 4.7B 规模。在部署解读下，十六个配对置信区间中有十三个不包含零，领先幅度从对最强对手的 0.075 到对最弱对手的 0.819。完成门控并非优势的载体：禁用全部六个可执行检查后，平均分变化为 +0.080、+0.117、+0.003 和 -0.004，所有区间均包含零，只有在 1.9B 规模下禁用它才会产生可经验检测的影响。

    arXiv:2611.21407v1 Announce Type: new  Abstract: A harness turns a model into a system that completes tasks: it builds the prompt, exposes tools and decides when work is done. Harness comparisons hold the model fixed. Model comparisons hold the harness fixed. We vary both: five shipped harnesses, 18 artifact-producing tasks, four locally deployable open-weight models from 0.75B to 4.7B and a 35.5B anchor, with one deterministic checker scoring all 360 ladder cells. The completion-controlling harness, Mingbird, has the highest mean at every size. Of four pre-registered paired contrasts, two survive Holm correction, at 0.75B and 4.7B. Thirteen of sixteen paired intervals exclude zero under the deployment reading, with leads from 0.075 over the strongest rival to 0.819 over the weakest. The completion gate is not the carrier: disabling all six executable checks moves the mean by +0.080, +0.117, +0.003 and -0.004, every interval including zero, and only at 1.9B does disabling it create emp
    
[^6]: 软件设计中的前提恢复：这一决策建立在什么之上？

    Premise Recovery in Software Design: What Does This Decision Rest On?

    [https://arxiv.org/abs/2611.21367](https://arxiv.org/abs/2611.21367)

    该论文提出“前提恢复”新任务，即在架构决策记录（ADR）未明确声明其依赖的先前决策时，自动从历史记录中识别出这些前提，并构建了首个包含 1,060 条 ADR 和 1,529 个标注配对的高质量基准数据集，用于评估大语言模型在该任务上的表现。

    

    设计决策塑造软件架构，将它们记录下来可以保存决策背后的原因。架构决策记录（ADR）用于捕捉这些决策，每条记录包含其上下文、选择和后果。一个决策可能依赖于更早的决策：撤销其中一个将迫使依赖它的决策被重新审查。我们将这些被依赖的早期决策称为前提。然而，ADR 并不总是明确指出其前提。当某个决策被撤销时，工程师会废止其 ADR 并手动搜索其影响范围，这种搜索可能会遗漏基于未命名前提的 ADR。我们将其定义为前提恢复任务：给定一条标识符被遮蔽的 ADR，在更早的记录中找出它的前提。我们从 36 个代码仓库中收集了 1,060 条 ADR，并为 60 个目标 ADR 标注了 1,529 个候选配对。两个 LLM 标注者之间的一致性达到 κ = 0.859；一名独立的人类标注者在不知晓其标签的情况下，在 32 个前提中的 30 个上与明确/隐式标签一致（κ = 0.875）。我们比较了 7 个开放权重模型与 2 个基线……

    arXiv:2611.21367v1 Announce Type: cross  Abstract: Design decisions shape software architecture, and writing them down preserves the reasons behind it. Architecture decision records (ADRs) capture them, each with its context, choice, and consequences. A decision can depend on earlier ones: reversing one of them would force its review. We call these its premises. Yet ADRs do not always name their premises. When a decision is reversed, engineers supersede its ADR and search by hand for what it affects. That search can miss ADRs resting on an unnamed premise. We frame this as premise recovery: given an ADR with identifiers masked, find its premises among earlier records. We collect 1,060 ADRs from 36 repositories and label 1,529 candidate pairs for 60 target ADRs. Two LLM annotators agree at \k{appa} = 0.859. An independent human annotator, blind to their labels, agrees with the stated/implicit tag on 30 of 32 premises (\k{appa} = 0.875). We compare 7 open-weight models with 2 baselines t
    
[^7]: 以智能体原生为设计理念：一种面向智能体应用的编程语言

    Agent-Native by Design: A Programming Language for Agentic Applications

    [https://arxiv.org/abs/2611.21298](https://arxiv.org/abs/2611.21298)

    该论文提出“智能体原生编程”理念，并在Python超集语言Jac中通过"by llm()"、对象空间编程和"visit ... by llm()"等机制，将大语言模型操作、工作流结构与状态、模型引导路由统一在同一套执行语义下，从而简化智能体应用的开发并提升编译器对程序进行自动优化的能力。

    

    构建智能体应用需要将大语言模型（LLM）操作、工作流结构与应用状态、以及模型引导的路由相结合。在使用SDK和智能体框架时，开发者需要通过辅助代码来连接这些组件，例如准备模型输入、传播状态以及分发模型选择的步骤。工作流的变更往往需要对这些代码进行协调更新，从而增加开发和测试的工作量。此外，将执行逻辑分散在应用代码与框架内部之间，限制了编译器对控制流和状态依赖的可见性，阻碍了自动优化。我们提出了智能体原生编程，以共享的执行语义将这三个组件统一起来。我们在Python超集语言Jac中实现了这一方法：通过"by llm()"实现类型化的模型操作与工具使用，通过对象空间编程实现工作流结构与状态，并通过"visit ... by llm()"实现模型引导的路由。

    arXiv:2611.21298v1 Announce Type: cross  Abstract: Building agentic applications requires combining large language model (LLM) operations, workflow structure and application state, and model-guided routing. With SDKs and agent frameworks, developers connect these components through supporting code that prepares model inputs, propagates state, and dispatches model-selected steps. Workflow changes can require coordinated updates to this code, increasing development and testing effort. Splitting execution logic between application code and framework internals also limits compiler visibility into control flow and state dependencies, hindering automatic optimization. We introduce Agent-Native Programming, which unifies these three components with shared execution semantics. We realize this approach in Jac, a Python-superset language, through "by llm()" for typed model operations and tool use, Object-Spatial Programming for workflow structure and state, and "visit ... by llm()" for model-gui
    
[^8]: 开发者人格如何塑造AI编程助手的使用

    How Developer Personality Shapes the Use of AI Coding Assistants

    [https://arxiv.org/abs/2611.21240](https://arxiv.org/abs/2611.21240)

    本文通过系统性范围综述首次梳理了用户人格与基于LLM工具感知（尤其是感知有用性）之间的关联，指出软件工程领域尚缺相关研究，为面向开发者人格的AI编程助手个性化理论迈出第一步。

    

    开发者在人格特质上各不相同，这些特质影响着他们处理软件开发任务的方式；然而，AI编程助手要么行为千篇一律，要么虽提供个性化功能，却需要开发者通过反复试错才能找到与自己契合的设置。这是因为人格如何塑造开发者与AI助手的交互体验尚不清楚，因而个性化功能应当考虑哪些特质也无从得知。换言之，我们在缺乏“什么在何时有效、为何有效”这一底层理论的情况下为开发者提供了工具。为迈出构建此类理论的第一步，本文开展了一项系统性范围综述，研究用户人格与其对基于大语言模型（LLM）工具的感知之间的关系。我们通过通用搜索引擎、主流软件工程（SE）学术会议、AI工具检索以及滚雪球引用追踪等方法，共识别出十二项研究。这些研究均报告了人格与用户感知之间的关联，尤其是对感知有用性的影响，但尚无研究在软件工程（SE）情境下对此进行探讨。基于这些发现……

    arXiv:2611.21240v1 Announce Type: new  Abstract: Developers differ in personality traits that shape how they approach software development tasks, yet AI coding assistants either behave uniformly or offer personalization features whose fit with individual developers must be found through trial and error. This is because it remains unclear how personality shapes developers' experiences with AI assistants, and thus which traits personalization should account for. In other words, we provide tools with no underlying theory on what works when and why. We take a first step towards such a theory with a systematic scoping review on how users' personality relates to their perception of LLM-based tools. We identified twelve studies through general search engines, major Software Engineering (SE) venues, AI tooling, and snowballing. They all reported associations between personality and perceptions, particularly perceived usefulness, yet none addressed this in a SE context.   Building on these find
    
[^9]: 从TOCTOU中剥离时间因素：从竞态复现到安全发现

    Taking Time Out of TOCTOU: From Race Reproduction to Security Discovery

    [https://arxiv.org/abs/2611.21214](https://arxiv.org/abs/2611.21214)

    Chronos框架通过seccomp、ptrace和KVM等多种机制将TOCTOU竞态攻击的干预时机与判定证据分离，从而实现对竞态漏洞的确定性复现、提交二分定位和CI安全回归测试。

    

    检查时到使用时（TOCTOU）竞态难以复现，而竞态丢失会使存在漏洞的程序看起来已被修复。我们提出了Chronos，这是一个将干预时机与判定证据相分离的框架，它从声明的竞态构建可复用的一次性安全回归oracle，用于确定性复现、提交二分定位和持续集成（CI）。竞态规范由可复用的保持引擎、动作、观察者和结果规则组合而成。当声明的保持被达成且攻击在保持期间完成时，Chronos在阻塞和干扰假设下将该攻击排序在使用点之前。Chronos使用seccomp通知在原版Linux的系统调用边界处强制执行声明的保持-攻击-释放顺序；ptrace和KVM客户机门则可在用户空间和内核指令位置提供保持。一个统一的决策过程为每种机制的运行结果打标签：fixed（已修复）需要针对具体情况的、积极的防护保存证据，justified refusal（合理拒绝）以及检查下的遗漏……

    arXiv:2611.21214v1 Announce Type: cross  Abstract: Time-of-check-to-time-of-use (TOCTOU) races are hard to reproduce and lost races can make vulnerable programs look fixed. We present Chronos, a framework that separates intervention timing from verdict evidence and builds reusable one-trial security regression oracles from declared races for deterministic reproduction, commit bisection and CI. Race Specifications compose reusable hold engines, actions, observers and outcome rules. When the declared hold is reached and the attack completes while held, Chronos orders it before the use under blocking and interference assumptions. Chronos uses seccomp notifications to enforce declared hold-attack-release ordering at syscall boundaries on stock Linux; ptrace and a KVM guest gate provide holds at userspace and kernel instruction sites. One decision procedure labels every mechanism's runs: fixed requires positive, case-specific evidence of preservation, justified refusal, omission under check
    
[^10]: 基于大语言模型的科学编程：来自有限元分析的证据

    Scientific Coding with Large Language Models: Evidence from Finite Element Analysis

    [https://arxiv.org/abs/2611.21176](https://arxiv.org/abs/2611.21176)

    该论文将FEM-Bench扩展至17个涉及非线性力学与结构动力学的有限元编程任务，对15个大语言模型的评估表明，单次生成的正确率有限，而结合代码执行与修订的智能体框架能显著提升性能（最高达96.5% Pass@3），揭示了LLM在自动化科学编程中的潜力与局限。

    

    大语言模型（LLM）为自动化科学编程提供了一条潜在途径，但其实用性取决于能否持续生成数值正确的代码。我们将FEM-Bench扩展至17个涉及非线性和历史相关力学以及结构动力学的有限元任务，并在无执行反馈的条件下对15个模型的代码实现和测试生成能力进行了评估。表现最好的模型在单次实现试验中的成功率达到87.1%，但两个赛道的排名并不一致：最佳实现模型在联合测试中成功率为49.9%，而最佳测试生成模型达到82.6%。对于其中六个模型，我们比较了单次生成与两种允许代码执行和修订的智能体框架。智能体执行解决了一些在单次尝试中从未成功的任务，并提高了其他任务的稳定性，但收益因模型和框架而异，且也出现了性能回退的情况。Claude Opus 5结合SimpleAgent达到了96.5%的Pass@3。

    arXiv:2611.21176v1 Announce Type: cross  Abstract: Large language models (LLMs) offer a potential route to automating scientific programming, but their usefulness depends on consistently producing numerically correct code. We extend FEM-Bench to 17 finite element tasks involving nonlinear and history-dependent mechanics and structural dynamics, and evaluate 15 models on implementation and test generation without execution feedback. The best model succeeds in 87.1% of single-pass implementation trials, but rankings differ between tracks. The leading implementer achieves 49.9% joint test success, while the leading tester reaches 82.6%. For six models, we compare single-pass generation with two agentic harnesses that allow code execution and revision. Agentic execution solves some tasks that never succeeded in single-pass trials and improves consistency on others, but gains vary across models and harnesses, and regressions also occur. Claude Opus 5 with SimpleAgent reaches 96.5% Pass@3, t
    
[^11]: SAFE：ECMAScript可扩展分析框架的形式化规范与实现

    SAFE: Formal Specification and Implementation of a Scalable Analysis Framework for ECMAScript

    [https://arxiv.org/abs/2611.21148](https://arxiv.org/abs/2611.21148)

    本文提出了SAFE——首个同时提供ECMAScript形式化规范与开源实现的可扩展JavaScript分析框架，旨在帮助研究社区检测JavaScript应用程序的安全漏洞。

    

    JavaScript在网络编程中的广泛使用暴露了JavaScript应用程序的安全漏洞问题，这凸显了利用JavaScript分析器检测此类问题的需求。近年来，研究人员提出了多种JavaScript程序分析器，一些网络服务公司也开发了各种JavaScript引擎。然而遗憾的是，大多数工具缺乏完善的文档，因此很难理解和修改它们；或者这些工具通常不对公众开放。本文提出了SAFE的形式化规范与实现，SAFE是一个为JavaScript研究社区开发的可扩展的ECMAScript分析框架。与现有仅关注形式化规范或开源实现其中之一的方法相比，这是首次同时为JavaScript提供形式化规范及其开源实现的尝试。为了让其他研究人员更容易使用我们的框架，我们将形式化规范…

    arXiv:2611.21148v1 Announce Type: cross  Abstract: The prevalent uses of JavaScript in web programming have revealed security vulnerability issues of JavaScript applications, which emphasizes the need for JavaScript analyzers to detect such issues. Recently, researchers have proposed several analyzers of JavaScript programs and some web service companies have developed various JavaScript engines. However, unfortunately, most of the tools are not documented well, thus it is very hard to understand and modify them. Or, such tools are often not open to the public.   In this paper, we present formal specification and implementation of SAFE, a scalable analysis framework for ECMAScript, developed for the JavaScript research community. This is the very first attempt to provide both formal specification and its opensource implementation for JavaScript, compared to the existing approaches focused on only one of them. To make it more amenable for other researchers to use our framework, we forma
    
[^12]: AutoDiDact：基于推断结构的输入约简

    AutoDiDact: Input Reduction over Inferred Structure

    [https://arxiv.org/abs/2611.20969](https://arxiv.org/abs/2611.20969)

    该论文提出AutoDiDact方法，证明结构化输入约简无需文法或内置格式知识，仅通过反复询问预言机“失败是否仍然存在”即可从失败输入中自动推断出中性词与定界符对，从而在标识符、定界组等结构单元层面高效约简输入。

    

    Delta debugging（增量调试）通过删除失败输入中的片段来约简输入，只有当预言机确认失败仍然存在时才保留删除。它仅根据输入的长度来分块，因此其删除操作会切断诸如标识符和定界组之类的单元，浪费调用次数并使输出仍然较大。当前最先进的结构化约简器通过提供文法或内置的格式知识来避免这一问题。我们证明结构化约简两者皆不需要：一个中性词和定界符对就足够了，而且两者都可以从失败输入中推断出来，只需询问 delta debugging 本来就会问其预言机的问题：失败是否仍然存在？中性词（例如空格）插入在这些单元之间时不会破坏失败，但插入在单元内部时则会，因此它可以将输入切分成标记。定界符对是两个只有在同时添加或删除时才保持失败不变的标记，因此它们所界定的范围可以作为一个单元来约简。我们对两者进行了形式化定义，并陈述……（原文截断）

    arXiv:2611.20969v1 Announce Type: new  Abstract: Delta debugging reduces a failing input by deleting chunks of it, keeping a deletion only if an oracle confirms the failure persists. It chunks the input based only on its length, so its deletions cut through units such as identifiers and delimited groups, wasting calls and leaving outputs large. Current state-of-the-art structural reducers avoid this by supplying grammars or built-in format knowledge. We show that structured reduction needs neither: a neutral word and delimiter pairs suffice, and both can be inferred from the failing input by asking only what delta debugging already asks its oracle: does the failure persist?   A neutral word, such as a space, preserves the failure when inserted between such units but not inside them, so it splits the input into tokens. A delimiter pair is two tokens that preserve the failure only when added or removed together, so the span they bound can be reduced as a unit. We formalize both, state wh
    
[^13]: LEDGER：可审计的、能应对事实变更的智能体记忆系统

    LEDGER: Auditable Agent Memory Under Change

    [https://arxiv.org/abs/2611.20927](https://arxiv.org/abs/2611.20927)

    提出了 LEDGER，一个可审计的智能体记忆层，由语言模型提出事实、显式规则决定当前有效值、可信来源和删除操作，并记录每个事实的来源与有效期，从而应对真实场景中事实变更、来源可靠性差异和用户遗忘请求等挑战。

    

    智能体记忆系统大多是在固定历史数据上进行评估的，且只对最终答案进行打分。而在实际使用中，事实会发生变化，不同来源的可靠性各不相同，用户还会要求智能体遗忘某些内容。我们提出了 LEDGER，这是一种记忆层，由语言模型提出事实，而显式规则决定哪个值是当前有效的、应该信任哪个来源、以及应该删除什么内容。每个事实都保留其来源信息及其成立的时期。LEDGER+EP 在此基础上增加了对原始对话轮次的搜索，用于处理账本未覆盖的问题，并在删除时对这些轮次进行清除。我们在相同的回答模型和提示词下，将两种系统与无记忆、全历史重放、对原始记录的检索以及 Mem0 进行比较，实验基于一个合成的多会话编程智能体工作负载和 LongMemEval 的 210 题子集，并且我们对每个记忆系统发送给模型的证据和最终答案都进行打分。在编程工作负载上，LEDGER+EP 在回退模式下回答了 94.3% 的问题……

    arXiv:2611.20927v1 Announce Type: cross  Abstract: Agent memory systems are mostly evaluated on fixed histories, and only their final answers are scored. In real use, facts change, sources differ in reliability, and users ask the agent to forget things. We present LEDGER, a memory layer in which a language model proposes facts and explicit rules decide which value is current, which source to trust, and what to delete. Each fact keeps its source and the period in which it held. LEDGER+EP adds a search over raw conversation turns for questions the ledger does not cover, and scrubs these turns on deletion. We compare both against no memory, full-history replay, retrieval over raw transcripts, and Mem0, with the same answering model and prompt, on a synthetic multi-session coding-agent workload and a 210-question subset of LongMemEval, and we score both the evidence each memory sends to the model and the answer. On the coding workload, LEDGER+EP in fallback mode answers 94.3% of questions 
    
[^14]: 重读并非免费：软件工程中LLM智能体的Token轨迹与缓存感知能耗估计

    Re-Reading Is Not Free: Token Traces and Cache-Aware Energy Estimation for LLM Agents in Software Engineering

    [https://arxiv.org/abs/2611.20873](https://arxiv.org/abs/2611.20873)

    该论文基于十周真实AI辅助软件开发中近10亿token的分析，发现缓存上下文读取占LLM智能体token总量的94.8%（每生成一个token需读取约260个缓存token），并据此提出了纳入缓存复用与子智能体工作的缓存感知能耗估计模型。

    

    随着大型语言模型（LLM）从对话助手演变为嵌入软件开发工作流中的自主智能体，估算其环境足迹需要超越传统的基于token的计量方式。现有的能耗估计器聚焦于孤立的请求-响应交互，忽视了对累积上下文的重复处理、缓存复用以及委托给子智能体的工作。我们分析了在十周真实AI辅助软件开发过程中收集的近10亿个token，这些数据涵盖42个会话和752个任务片段。缓存上下文读取占API报告token总数的94.8%，而非缓存输入仅占0.017%；每生成一个token，大约需要读取260个缓存token。与单个任务内1.4倍的上下文增长相比，整个会话期间上下文增长了5.3倍。为了量化其能耗影响，我们扩展了现有的分析式推理能耗模型，以纳入缓存相关的因素。

    arXiv:2611.20873v1 Announce Type: cross  Abstract: As large language models (LLMs) evolve from conversational assistants into autonomous agents embedded in software development workflows, estimating their environmental footprint requires moving beyond conventional token-based accounting. Existing energy estimators focus on isolated request-response interactions, overlooking the repeated processing of accumulated context, cache reuse, and work delegated to subagents. We analyze nearly one billion tokens collected across 42 sessions and 752 task episodes over ten weeks of real-world AI-assisted software development. Cached-context reads make up 94.8% of all API-reported tokens, while uncached input accounts for only 0.017%; for every generated token, approximately 260 cached tokens are read. Context grows 5.3-fold over a session compared with 1.4-fold within individual tasks. To quantify the energy implications, we extend an existing analytical inference-energy model to account for cache
    
[^15]: Synapse：通过程序修复优化自然语言智能体技能

    Synapse: Optimizing Natural-Language Agent Skills through Program Repair

    [https://arxiv.org/abs/2611.20856](https://arxiv.org/abs/2611.20856)

    Synapse 将自然语言技能优化建模为对部分程序视图的“生成-验证”式程序修复，通过编译器生成中间表示 SkIR，结合执行轨迹与控制流/依赖分析来定位故障、提出类型化变异并用修复测试验证补丁，从而在不更新模型参数的情况下优化 LLM 智能体技能。

    

    arXiv:2611.20856v1 公告类型：新论文。大语言模型（LLM）智能体可以通过优化可复用的自然语言技能来提升任务表现，而无需更新模型参数。执行反馈能够揭示失败，但并不能直接指出需要编辑哪些指令、哪些相关指令是重要的，或者某次编辑可能会破坏哪些任务。我们提出 Synapse，它将技能优化建模为对部分程序视图的“生成-验证”式修复。一个确定性编译器将可识别的技能构造翻译为 SkIR——一种与源文本关联的中间表示，同时将其余需求保留在文本中。Synapse 将执行轨迹与控制流及依赖关系相结合，以定位故障、提出类型化的变异操作，并选择相关的修复测试用例（包括之前通过的测试）。每个变异都会被写回其映射的源文本区间并重新编译；验证过程决定其是否被接受。每个被接受的补丁都能产生格式良好的技能，并保留未受……（原文摘要在此处被截断）

    arXiv:2611.20856v1 Announce Type: new  Abstract: Large language model (LLM) agents can improve task performance by optimizing reusable natural-language skills without updating model parameters. Execution feedback reveals failures but does not directly identify which instructions to edit, which related instructions matter, or which tasks an edit may break. We propose Synapse, which casts skill optimization as generate-and-validate repair over a partial program view. A deterministic compiler translates recognized skill constructs into SkIR, a source-linked intermediate representation, while retaining other requirements in the text. Synapse combines execution traces with control-flow and dependence relations to localize faults, propose typed mutations, and select related repair tests, including previously passing ones. Each mutation is written back to its mapped source span and recompiled; validation determines acceptance. Every accepted patch yields a well-formed skill and preserves unta
    
[^16]: Injectra：为基于Web的桌面应用程序改造可扩展性

    Injectra: Retrofitting Extensibility into Web-Based Desktop Applications

    [https://arxiv.org/abs/2611.20713](https://arxiv.org/abs/2611.20713)

    Injectra是一个通用的代码注入与功能增强系统，通过统一执行通道将JavaScript、HTML和CSS注入到基于Web技术（如Electron、Tauri）构建的桌面应用内部运行时环境中，实现对现有桌面客户端的界面修改、行为调整和功能扩展。

    

    大型语言模型能够根据自然语言提示生成代码，为用户定制软件和扩展其功能开辟了新的可能性。本文提出并实现了Injectra，这是一个面向现有桌面客户端的通用代码注入与功能增强系统，适用于使用嵌入式浏览器引擎（如Electron）或系统WebView（如Tauri）构建用户界面的应用程序。Injectra将宿主集成逻辑与上层增强代码解耦，结合运行时集成与预启动引导两种机制，通过统一的执行通道将JavaScript、HTML和CSS注入到应用程序的内部Web运行时环境中，从而实现对界面渲染的修改、交互行为的调整以及客户端功能的扩展。该系统使用命令行界面（CLI）作为统一入口，允许……

    arXiv:2611.20713v1 Announce Type: new  Abstract: Large language models can generate code based on natural language prompts, opening up new possibilities for users to customize software and extend its functionality. This paper proposes and implements Injectra, a general-purpose code injection and feature enhancement system for existing desktop clients, suitable for applications whose user interfaces are built using embedded browser engines (such as Electron) or system WebViews (such as Tauri). Injectra decouples the host integration logic from the upper-layer enhancement code. Combining both runtime integration and pre-launch bootstrap mechanisms, it injects JavaScript, HTML, and CSS into the application's internal Web runtime environment through a unified execution channel, thereby enabling modifications to interface rendering, adjustments to interactive behaviors, and client-side functionality extensions. The system uses a command-line interface (CLI) as a unified entry point, allowin
    
[^17]: 噪声感知的NISQ量子程序测试：具有可控假阳性率的自基线测试预言机

    Noise-Aware Testing of NISQ Quantum Programs: Self-Baseline Oracle with a Controlled False-Positive Rate

    [https://arxiv.org/abs/2611.20613](https://arxiv.org/abs/2611.20613)

    该论文提出了一种基于自基线校准的噪声感知NISQ量子程序测试框架，以可信参考程序在同一后端上的重复执行作为对照，并利用Jensen-Shannon散度有限采样零分布的分位数作为接受阈值，从而实现假阳性率的预先设定与可控。

    

    在噪声中等规模量子（NISQ）设备上测试量子程序面临一个测试预言机问题：硬件噪声和有限次数采样使得正确程序的运行结果波动程度与错误程序相当，因此采用固定接受阈值的预言机所产生的假阳性会依赖于具体程序和后端。我们提出了一个基于自基线校准的噪声感知测试框架：被测程序的每次运行都与在同一后端上重复执行的受信任参考程序进行比较，接受阈值取Jensen-Shannon散度有限次数零假设分布的$(1-\alpha)$分位数，从而使预言机的假阳性率成为可预先设定的水平。一个非中心卡方模型用于预测检测功效，同质性检验和双运行哨兵机制用于检测漂移，一个变异引擎在三分类噪声感知分类体系内对每个变异体进行精确标注，零噪声外推……（摘要不完整）

    arXiv:2611.20613v1 Announce Type: cross  Abstract: Testing quantum programs on Noisy Intermediate-Scale Quantum (NISQ) devices faces a test-oracle problem: hardware noise and finite-shot sampling make the runs of a correct program vary as much as those of a faulty one, so oracles with a fixed acceptance threshold produce false positives that depend on the program and the backend. We present a noise-aware testing framework built on self-baseline calibration: each run of the program under test is compared with repeated executions of a trusted reference program on the same backend, and the acceptance threshold is the $(1-\alpha)$ quantile of the finite-shot null distribution of the Jensen-Shannon divergence, so the oracle's false-positive rate is a level set in advance. A noncentral chi-square model predicts detection power, a homogeneity test and a two-run sentinel detect drift, a mutation engine labels every mutant exactly within a three-class noise-aware taxonomy, and zero-noise extrap
    
[^18]: 量子纠错编译器中检测器错误模型的测试方法

    A Testing Method for Detector Error Models in Quantum Error Correction Compilers

    [https://arxiv.org/abs/2611.20501](https://arxiv.org/abs/2611.20501)

    提出 DEMVal，一种量子纠错编译器检测器错误模型的测试方法，通过精确比较编译器输出与外部重建的参考模型，实现与物理错误率无关的缺陷检测灵敏度。

    

    量子纠错（QEC）编译器根据量子码和噪声模型，生成可执行的物理电路以及检测器错误模型（DEM）。DEM 列出了每种错误机制的概率及其翻转的测量结果。解码器在未经校验的情况下直接使用 DEM，因此若 DEM 存在错误描述噪声的缺陷，就会导致解码器基于错误的错误率进行校准，从而悄然降低逻辑错误率。现有的量子软件测试技术通过崩溃或采样输出的统计特性来判断正确性，而在 QEC 实际运行的错误率下，仅扭曲错误率的 DEM 缺陷低于采样所能分辨的范围。我们提出了 DEMVal，一种测试方法，它将编译器输出的每个模型与重建的参考模型进行验证，其中错误率从被测编译器之外获取。该方法精确比较两个模型，因此其灵敏度与物理错误率无关。参考模型可以构建于……（摘要截断）

    arXiv:2611.20501v1 Announce Type: new  Abstract: A quantum error correction (QEC) compiler produces, from a code and a noise model, both the physical circuit to execute and a detector error model (DEM). The DEM lists the probability of each error mechanism and the measurements it flips. Decoders consume the DEM without verification, so a defect that describes the noise incorrectly calibrates the decoder against wrong rates and silently degrades the logical error rate. Existing techniques for testing quantum software judge correctness by crashes or by statistics of sampled outputs. DEM defects that only distort rates lie below what sampling can resolve at the error rates where QEC operates. We present DEMVal, a testing method that validates each model the compiler emits against a reconstructed reference, with rates obtained from outside the compiler under test. It compares the two models exactly, so its sensitivity is independent of the physical error rate. The reference can be built in
    
[^19]: 嵌入模型 vs. Grep：重新思考代码搜索中的稠密检索优势

    Embeddings vs. Grep: Rethinking the Dense Advantage in Code Search

    [https://arxiv.org/abs/2611.20455](https://arxiv.org/abs/2611.20455)

    该论文构建了贴近真实的仓库级代码搜索基准Repo-Premium，发现稠密检索相对BM25的优势主要源于偏向性的评估设置和查询缺乏罕见标识符，而通过LLM生成标识符扩展查询（generate-then-grep），BM25即可追平稠密检索的性能。

    

    代码搜索基准测试显示稠密编码器远远领先于BM25，然而编程智能体使用grep却表现良好。这种稠密优势究竟衡量了什么？现有基准测试无法回答这一问题：它们使用人工筛选的查询、过时的代码以及被削弱的BM25。为了弄清真相，我们构建了Repo-Premium，一个贴近现实的仓库级代码搜索基准。该基准对真实仓库中的文件进行排序，涵盖Python、JavaScript、TypeScript、Go、Rust和Java的人工筛选查询与原始查询，包含2022年至2026年的拉取请求，以及一个在91个仓库上预注册的测试池。我们的实验将这种优势追溯到两个来源：一是偏向稠密检索的评估选择，例如为BM25将文件切分成块、把已编辑的测试计为相关文档；二是查询中缺少代码里罕见标识符的情况。使用LLM生成的标识符扩展查询（generate-then-grep，即先生成再grep）对BM25的帮助远大于对稠密检索的帮助，使BM25在每个Python测试池中都能与稠密检索持平，即使在拉取请求场景中也是如此。

    arXiv:2611.20455v1 Announce Type: new  Abstract: Code-search benchmarks show dense encoders well ahead of BM25, yet coding agents do well with \texttt{grep}. What does this dense advantage measure? Existing benchmarks cannot tell: they use curated queries, old code and a handicapped BM25. To find out, we build Repo-Premium, a realistic repository-level code-search benchmark. It ranks files in real repositories, with curated and raw queries in Python, JavaScript, TypeScript, Go, Rust and Java, pull requests from 2022 to 2026, and a preregistered test pool over $91$ repositories. Our experiments trace the advantage to two sources: evaluation choices that favor dense retrieval, such as splitting files into chunks for BM25 and counting edited tests as relevant, and queries that lack the code's rare identifiers. Expanding the query with LLM-written identifiers (generate-then-grep) helps BM25 far more than dense retrieval, letting it match dense retrieval in every Python pool, even on pull r
    
[^20]: 异构AI编程代理数据的可靠挖掘：上下文感知预测与资源感知排序

    Reliable Mining of Heterogeneous AI Coding-Agent Data: Context-Aware Prediction and Resource-Aware Ranking

    [https://arxiv.org/abs/2611.20297](https://arxiv.org/abs/2611.20297)

    该论文发现简单汇集异构的AI编程代理运行数据会引发辛普森悖论式的结论反转，提出用上下文感知的特征级联方法将来源、环境与工作流上下文纳入建模，从而解决数据真实性问题并将PR合并预测AUC从0.774提升至0.832。

    

    AI编程代理的运行记录在代理、代码仓库、任务和工作流之间存在差异。我们研究了来自六个代理的71,677个拉取请求（PR），结果表明将这些记录简单汇集可能会颠覆描述性和预测性的结论。带有共同作者标记的PR合并率为55.6%，而没有该标记的PR合并率为77.5%，但剔除占主导地位的代理后，这一-21.8个百分点的差距反转为+23.7个百分点。我们将这种辛普森悖论式的反转视为数据真实性问题，并在特征级联中表示来源、环境和工作流上下文。梯度提升合并预测的AUC从0.774提升至0.832；该优势在时间评估（0.801）和未见过的仓库（0.807）下依然保持。条件化后共同作者系数从-1.09变为+1.44，而仓库聚类的AIPW区间显示调整后不存在稳定的标记关联。一个严格的PR创建时间模型达到0.741的AUC……

    arXiv:2611.20297v1 Announce Type: new  Abstract: Operational records from AI coding agents differ across agents, repositories, tasks, and workflows. We study 71,677 pull requests (PRs) from six agents and show that pooling these records can reverse descriptive and predictive conclusions. PRs carrying a co-authorship trailer have a 55.6% merge rate, compared with 77.5% without the trailer, but removing the dominant agent flips this -21.8 percentage-point gap to +23.7 points. We treat this Simpson-type reversal as a data-veracity problem and represent source, environment, and workflow context in a feature cascade. Gradient-boosting merge-prediction AUC increases from 0.774 to 0.832; the advantage persists under temporal evaluation (0.801) and unseen repositories (0.807). The co-authorship coefficient changes from -1.09 to +1.44 after conditioning, whereas repository-clustered AIPW intervals show no stable adjusted trailer association. A strict PR-creation-time model reaches 0.741 AUC in 
    
[^21]: 向最佳学习：是什么让Hugging Face模型广受欢迎？

    Learning From the Best: What Makes Popular Hugging Face Models?

    [https://arxiv.org/abs/2611.20227](https://arxiv.org/abs/2611.20227)

    该研究对66,909个Hugging Face模型进行系统分析，发现完善的文档质量（如更长的模型卡和更多的代码示例）等特征是模型获得高下载量和点赞数的关键因素。

    

    Hugging Face已成为共享和复用预训练模型的核心平台，但哪些实践与更高的可见度和采用率相关仍不明确。先前的研究描述了模型复用和模型卡质量，但并未从文档、仓库结构、平台元数据和技术元数据等方面对流行与不流行的模型进行系统性比较。我们研究了于2026年2月28日收集的66,909个Hugging Face模型，并从文档、模型、平台和技术四个维度提取了31个特征。针对下载量和点赞数，我们使用统计检验和效应量将最流行的前10%模型与最不流行的后80%模型进行比较，训练了五个分类器来预测流行度，利用表现最佳的模型对特征重要性进行排序，并验证了预测结果在不同模型领域和贡献者所属机构之间是否成立。流行模型始终拥有更长的模型卡、更多的代码示例以及更多的Git（原文在此处截断）。

    arXiv:2611.20227v1 Announce Type: new  Abstract: Hugging Face has become a central platform for sharing and reusing pre-trained models, yet it remains unclear which practices are associated with higher visibility and adoption. Prior studies describe model reuse and model-card quality but do not systematically compare popular and unpopular models across documentation, repository structure, platform metadata, and technical metadata. We studied 66,909 Hugging Face models collected on 2026-02-28 and extracted 31 features across four dimensions: Documentation, Models, Platforms, and Techniques. For downloads and likes, we compared the top 10% popular models with the bottom 80% using statistical tests and effect sizes, trained five classifiers to predict popularity, ranked feature importance with the best-performing model, and tested whether predictions held across model domains and contributor affiliations. Popular models consistently had longer model cards, more code examples, and more Git
    
[^22]: SWE-CoWork：将编程智能体作为组织中的数字员工进行基准测试

    SWE-CoWork: Benchmarking Coding Agents as Digital Employees in Organizations

    [https://arxiv.org/abs/2611.20218](https://arxiv.org/abs/2611.20218)

    该论文提出SWE-CoWork基准，将编程智能体置于拥有26名同事的模拟工程组织中，要求其从分散渠道恢复需求、调和冲突证据并完整交付任务，评估发现领先模型虽能通过约63%的功能测试，但完整任务交付率不足8%，揭示了部分实现与完整交付之间的显著差距。

    

    在软件组织中工作的编程智能体必须确定要构建什么、与同事协作，并验证其交付成果。我们提出了SWE-CoWork，这是一个包含101项开发任务的基准测试，涵盖101个真实代码仓库、11种编程语言和12个应用领域。每项任务都将智能体置于一个模拟的工程组织中，该组织拥有26名具有不同职责、有限知识和受控协作行为的同事。从简短的任务交接开始，智能体必须从同事和工作场所记录中恢复分散的需求、调和相互冲突的信息，并实现由私有可执行测试评估的代码变更。我们评估了11个模型，发现部分实现与完整任务交付之间存在巨大差距。两个表现最好的模型满足了约63%的失败转通过（F2P）测试，但完成的任务不到8%。提供完整的信息……（原文摘要被截断）

    arXiv:2611.20218v1 Announce Type: cross  Abstract: A coding agent working within a software organization must establish what to build, collaborate with colleagues, and verify what it delivers. We introduce SWE-CoWork, a benchmark comprising 101 development assignments across 101 real repositories, 11 programming languages, and 12 application domains. Each assignment places an agent in a simulated engineering organization with 26 colleagues who have distinct responsibilities, bounded knowledge, and controlled collaboration behavior. Starting from a brief handoff, agents must recover requirements distributed across colleagues and workplace records, reconcile conflicting evidence, and implement changes evaluated by private executable tests. We evaluate eleven models and find a substantial gap between partial implementation and complete task delivery. The two leading models satisfy approximately 63% of fail-to-pass (F2P) tests but complete fewer than 8% of assignments. Providing complete r
    
[^23]: 规约明确程度如何塑造大语言模型生成代码中的失败

    How Specification Explicitness Shapes Failure in LLM-Generated Code

    [https://arxiv.org/abs/2611.20195](https://arxiv.org/abs/2611.20195)

    该研究通过72人参与的受控实验发现，轻量级结构规约对减少LLM生成代码的失败作用有限（仅8.7%），而明确写出约束、不变量、否定性需求、边界情况等行为义务可将失败减少47.2%，其中与隐含义务相关的失败模式降幅最大。

    

    当大语言模型（LLM）将需求转化为代码时，每一个未被明确规定的约束都会变成模型必须在生成过程中自行决定的行为。我们测试了规约在何时超越结构层面、开始控制生成代码的实际行为。在一项受控实验中，72名参与者在三种规约明确性水平下使用LLM实现和修改Python任务。结果显示，轻量级结构与明确的行为义务之间存在清晰分界：轻量级结构仅将隐藏测试失败减少了8.7%；而要求参与者明确指定约束、不变量、否定性需求、边界情况、平局裁决规则和异常处理，则将失败减少了47.2%，并将不同失败模式的数量减少了20.5%。降幅最大的是与隐含义务相关的失败模式：不变量违例下降了66.1%，不支持的假设下降了59.5%。在需求变更后，具有明确……（原文摘要在此处截断）

    arXiv:2611.20195v1 Announce Type: cross  Abstract: When LLMs turn requirements into code, every underspecified constraint becomes behavior that the model must decide during generation. We test when specifications move beyond structure and begin to control generated behavior. In a controlled experiment, 72 participants implemented and revised Python tasks with an LLM under three levels of specification explicitness. The results show a clear break between lightweight structure and explicit behavioral obligations. Lightweight structure reduced hidden-test failures by only 8.7%. Requiring participants to specify constraints, invariants, negative requirements, edge cases, tie-breaking rules, and exception handling cut failures by 47.2% and reduced the number of distinct failure modes by 20.5%. The largest drops were in failure modes tied to implicit obligations. Invariant violations fell by 66.1%, and unsupported assumptions fell by 59.5%. After a requirement change, specifications with exp
    
[^24]: 隐藏的方法论层面：揭示LLM辅助实证软件工程中的效度威胁

    The Hidden Methodological Layer: Uncovering Threats to Validity in LLM-Assisted Empirical Software Engineering

    [https://arxiv.org/abs/2611.20033](https://arxiv.org/abs/2611.20033)

    该研究通过对50项实证软件工程原始研究的二次分析发现，LLM辅助研究流程引入的效度威胁（如分类错误、提示词依赖性等）在现有文献中被严重低估报告。

    

    背景：大语言模型（LLM）日益被嵌入实证软件工程（ESE）工作流中，支持分类、编码、提取、综合和评估等任务。然而，其概率性和不透明的行为引入了可能影响实证研究严谨性的方法论问题。目的：本文研究LLM辅助实证工作流所引入的效度威胁，并考察LLM交互如何影响实证研究。方法：我们对通过系统性文献综述确定的50项原始研究（共包含69个LLM辅助研究任务）进行了二次分析，考察了LLM相关威胁被报告的程度、这些威胁的表征方式，以及它们与既有效度维度之间的关系。结果：LLM相关威胁仍然被严重低估报告。我们识别出反复出现的威胁，包括分类错误、提示词依赖性等问题。

    arXiv:2611.20033v1 Announce Type: new  Abstract: Context: Large Language Models (LLMs) are increasingly embedded in empirical software engineering (ESE) workflows, supporting tasks such as classification, coding, extraction, synthesis, and evaluation. Their probabilistic and opaque behavior, however, introduces methodological concerns that may affect empirical rigor. Aims: This paper investigates threats to validity introduced by LLM-assisted empirical workflows and examines how LLM interactions influence empirical studies. Method: We conducted a secondary analysis of 50 primary studies comprising 69 LLM-assisted research tasks identified through a systematic literature review. We examined the extent to which LLM-related threats were reported, how these threats were characterized, and how they related to established validity dimensions. Results: LLM-related threats remain substantially underreported. We identified recurring threats including classification errors, prompt-dependent inte
    
[^25]: 智能体框架剖析：决策、不变量与边界情况

    Anatomy of an Agent Harness: Decisions, Invariants and Corner Cases

    [https://arxiv.org/abs/2611.19960](https://arxiv.org/abs/2611.19960)

    本文首次将智能体框架中隐含于代码、鲜有记录的关键工程决策系统化为六组30张“决策卡”——每张包含选项与代价、可测试的不变量、破坏性边界情况、测试与证据标签——并配套提供带信任边界的系统模型、六类故障成因、基于脚本化提供者的测试方法与检查清单，为智能体框架的健壮性设计与测试建立了系统化方法论。

    

    智能体框架是围绕语言模型运行的程序，它将模型的回复转化为行动：它负责构建每次请求、运行模型所要求的工具、决定模型可以在无需询问的情况下执行哪些操作、存储对话内容、从故障中恢复，并向人类展示发生了什么。这些决策大多隐含在代码中而很少被记录下来，因此每个框架都会一再遇到相同的边界情况。本文将这些决策记录下来，形成可测试的不变量。论文提出了一个带信任边界的系统模型；六组共30张决策卡，每张卡包含选项及其代价、一个不变量、破坏该不变量的边界情况、一个测试和一个证据标签；六类故障成因；一种基于脚本化提供者、记录行为和植入变更的测试方法；以及一份检查清单。这些决策卡来源于作者自行构建的一个桌面框架 Crewroom、对其他十个框架的人工测试，以及以开源框架为参考的研究；部分模式……

    arXiv:2611.19960v1 Announce Type: new  Abstract: An agent harness is the program around a language model that turns its replies into actions: it builds each request, runs the tools the model asks for, decides what the model may do without asking, stores the conversation, recovers from failures, and shows a person what happened. Most of these decisions are made in code and rarely written down, so each harness meets the same corner cases again. This paper writes them down as invariants to test. It gives a system model with trust boundaries; 30 decision cards in six groups, each with options and costs, an invariant, the corner cases that break it, a test and an evidence tag; six cause classes for failures; a test method built on scripted providers, recorded behaviour and planted changes; and a checklist. The cards come from building one desktop harness, Crewroom, the author's own, and from manually testing ten other harnesses and studying open-source harnesses as reference; some patterns 
    
[^26]: 恢复过了头：理解并修复LLM智能体中的风险回退行为

    Recovery Gone Too Far: Understanding and Repairing Risky Fallback in LLM Agents

    [https://arxiv.org/abs/2611.19886](https://arxiv.org/abs/2611.19886)

    本文首次定义并实证揭示了LLM智能体中的“风险回退”现象——即工具失败后改用替代工具时可能跨越行动边界造成严重危害，通过覆盖八个领域和55个真实CVE的304个场景实证研究发现主流模型的风险场景边界违反率高达96.8%，且现有缓解措施均无法有效应对，并提出了修复方案。

    

    大型语言模型（LLM）智能体通过工具与环境进行交互，包括读取代码仓库、运行构建和推送代码。当工具因普通错误（如速率限制或超时）而失败时，智能体可以回退到替代工具并依然完成任务。然而，同一种回退在一种环境中可以完成任务，在另一种环境中却可能跨越行动边界。例如，在推送被拒绝后，强制推送在一个仓库中只会替换智能体自己的提交，而在另一个仓库中它还会删除其他作者的提交。我们将这种回退称为“风险回退”。为了理解这一现象，我们对横跨八个领域、涵盖55个真实CVE的304个场景进行了实证研究。我们评估的四种主流模型都表现出风险回退行为，在风险场景中行动边界违反率高达96.8%。我们进一步评估了三种主流缓解措施，发现没有任何一种（能够有效防范此类风险）……

    arXiv:2611.19886v1 Announce Type: cross  Abstract: Large Language Model (LLM) agents interact with their environments through tools, reading repositories, running builds, and pushing code. When a tool fails with an ordinary error, such as a rate limit or a timeout, the agent can fall back to a substitute tool and still complete the task. However, the same fallback that completes a task in one environment can cross an action boundary in another. For example, after a push is rejected, a forced push in one repository replaces only the agent's own commits, while in a different repository it also removes another author's commits. We call such a fallback a risky fallback. To understand this phenomenon, we conduct an empirical study on 304 scenarios that span eight domains and cover 55 real CVEs. All four mainstream models we evaluate exhibit risky fallback, with action-boundary violations in up to 96.8% of risky scenarios. We further evaluate three mainstream mitigations and find that none o
    
[^27]: 让程序执行来当老师

    Let Execution Do the Teaching

    [https://arxiv.org/abs/2611.19875](https://arxiv.org/abs/2611.19875)

    该论文提出以程序执行结果为核心训练信号的方法，将编译错误与测试反馈融入监督微调、C++修复课程和组相对强化学习中，并借助执行验证器生成结构化奖励，使 GLM-4.7-Flash 在 C++ 编程任务上的两轮成功率提升至 61.5%。

    

    改进一个编程模型是一系列决策过程：找出限制性能的行为、创建有用的练习、并确定哪些尝试值得强化。我们研究了一种实用的方法，通过将反馈建立在程序执行之上，使这些决策愈发可重复。该工作从 GLM-4.7-Flash 出发，结合了整文件监督微调、针对性的 C++ 修复课程、组相对强化学习，以及能将程序行为转化为结构化奖励的验证器。编译器错误和未通过的测试为修复练习提供了具体的上下文，而可执行的检查则对生成的程序进行评分。我们在 26 个 C++ 编程任务上评估了训练后的模型，同时衡量初始成功率和在测试反馈后的完成率。所报告的最强后训练配置在第一轮测试中达到 45.2% 的成功率，在两轮内达到 61.5%。这些结果来自一个配有持续演进提示词的辅助开发套件。

    arXiv:2611.19875v1 Announce Type: cross  Abstract: Improving a coding model is a sequence of decisions: find the limiting behavior, create useful practice, and determine which attempts deserve reinforcement. We investigate a practical way to make those decisions increasingly repeatable by grounding feedback in execution. Starting from GLM-4.7-Flash, the work combines whole-file supervised fine-tuning, targeted C++ repair curricula, group-relative reinforcement learning, and verifiers that turn program behavior into structured rewards. Compiler errors and failing tests provide specific contexts for repair practice, while executable checks score the resulting programs. We evaluate the trained models on 26 C++ coding tasks, measuring both initial success and completion after test feedback. The strongest reported post-training configuration achieves 45.2% success in the first test round and 61.5% within two rounds. These results come from an assisted development suite with evolving prompt 
    
[^28]: 发现但未修复：使用迭代式LLM流水线从软件漏洞发现到修复

    Found, but Not Fixed : From Software Vulnerability Discovery to Fixes with Iterative LLM Pipelines

    [https://arxiv.org/abs/2611.19719](https://arxiv.org/abs/2611.19719)

    该论文提出了一个基于LLM迭代流水线的通用框架，通过多阶段反馈循环实现真实世界代码中漏洞函数的自动化高质量修复，并在GPT-4o、GPT-5.1和Gemini 3 Flash等最先进模型上验证了其有效性。

    

    尽管已经部署了先进的检测工具和AI助手，软件漏洞仍然是一个严重的威胁。虽然检测技术已取得进展，但随后的代码修复阶段仍然是主要瓶颈；人工干预本质上缓慢、成本高昂，并且容易引入二次回归和问题。在本文中，我们研究了基于LLM的迭代流水线能否为真实世界代码中易受攻击函数的自动化修复提供稳健的解决方案。我们提出了一个通用框架，通过多阶段反馈循环利用大型语言模型（LLM）的推理能力。我们将该方法与最先进的模型（包括GPT-4o、GPT-5.1和Gemini 3 Flash）进行基准测试，证明这样的流水线能够实现高质量的代码修复。为支持更广泛的安全社区，我们提供了评估框架、微调权重和全面的实验结果。

    arXiv:2611.19719v1 Announce Type: cross  Abstract: Software vulnerabilities persist as a critical threat despite the deployment of sophisticated detection tools and AI assistants. While detection has advanced, the subsequent code repair phase remains a primary bottleneck; manual intervention is inherently slow, costly, and prone to introducing secondary regressions and issues. In this paper, we investigate whether an iterative, LLM-based pipeline can provide a robust solution for the automated repair of vulnerable functions in real-world code. We propose a generalized framework that leverages the reasoning capabilities of Large Language Models (LLMs) through a multi-stage feedback loop. We benchmark this approach against state-of-the-art models, including GPT-4o, GPT-5.1, and Gemini 3 Flash, demonstrating that such a pipeline can achieve high-quality code repair. To support the broader security community, we provide our evaluation framework, fine-tuned weights, and comprehensive experi
    
[^29]: 从变异体中学习：教会小语言模型合成循环不变式

    Learning from Mutants: Teaching Small Language Models to Synthesise Loop Invariants

    [https://arxiv.org/abs/2611.19569](https://arxiv.org/abs/2611.19569)

    该论文提出一种受变异测试启发的失败驱动数据筛选方法，通过将基准真值不变式变异为导致验证失败的语义变异体并结合验证器反馈来构造高质量训练数据，从而成功微调小语言模型合成循环不变式。

    

    大语言模型（LLMs）已被证明在合成循环不变式方面具有前景，而循环不变式一直是自动化程序验证的长期障碍。然而，前沿大语言模型通常只能通过第三方API访问，这阻碍了可复现性和本地部署。小语言模型（SLMs）可以避开这些障碍，但需要针对特定任务的微调，而这又需要高质量的训练数据。我们提出了一种失败驱动的数据筛选方法，用于微调语言模型以合成循环不变式。受变异测试的启发，我们的方法将语义变异算子应用于基准真值不变式，产生会导致验证失败的语义变异体。每个变异体都与验证器反馈及其基准真值对应版本配对，从而产生用于反馈引导的不变式精化的训练样本。虽然庞大的变异体空间是变异测试中长期存在的负担，但当系统地进行生成时，它却成为一种资产。

    arXiv:2611.19569v1 Announce Type: cross  Abstract: Large language models (LLMs) have proven promising for synthesising loop invariants, a long-standing obstacle to automated program verification. Yet, frontier LLMs are typically accessible only through third-party APIs, hampering reproducibility and local deployment. Small language models (SLMs) sidestep these obstacles but require task-specific fine-tuning, for which high-quality training data is needed. We present a failure-driven data curation method for fine-tuning language models to synthesise loop invariants. Inspired by mutation testing, our approach applies semantic mutation operators to ground-truth invariants, producing semantic mutants that induce verification failure. Each mutant is paired with verifier feedback and its ground-truth counterpart, yielding training examples for feedback-guided invariant refinement. Whereas the vast space of mutants is a long-standing burden in mutation testing, it becomes an asset when genera
    
[^30]: Inquest：基于分阶段智能体的日志引导根因代码定位

    Inquest: Log-Guided Root-Cause Code Localization with a Staged Agent

    [https://arxiv.org/abs/2611.19554](https://arxiv.org/abs/2611.19554)

    提出INQUEST，一个使用27B开源模型的分阶段智能体，通过日志与相似历史issue等证据自动定位生产故障的根因代码，并基于38,001份真实issue报告构建了新基准。

    

    当生产环境问题被报告时，工程师只能获得一段描述和一些日志，既没有失败的测试用例，也无法复现故障。日志显示的是故障被发现的位置，而非故障产生的原因：在我们研究的真实报告中，49%的报告文本并未提及修复最终修改的任何文件。我们从两个公开数据集和27个开源项目（Apache、Red Hat、MongoDB、MariaDB、Qt和Sakai）为这一场景构建了一个基准，其中包含38,001份已解决的issue报告及其附带日志、bug仍存在时的代码状态，以及修复所修改的文件。我们提出了INQUEST，一个使用27B开源权重模型查找故障代码的智能体。流程由代码而非模型控制：它首先收集证据，包括在报告提交之前就已解决的类似历史issue中修复的文件；随后模型只需做出三个受限决策——打开哪些文件、每个文件扮演什么角色……

    arXiv:2611.19554v1 Announce Type: cross  Abstract: When a production issue is reported, engineers get a description and some logs, but no failing test and no way to reproduce the fault. The logs show where the failure was noticed, not where it was caused: in 49% of the real reports we study, the text names none of the files that the fix later changed. We built a benchmark for this setting from two public datasets and 27 open-source projects (Apache, Red Hat, MongoDB, MariaDB, Qt and Sakai). It holds 38,001 resolved issue reports with their pasted logs, the code as it was while the bug was still present, and the files the fix changed. We present INQUEST, an agent that finds the faulty code using a 27B open-weights model. Code, not the model, controls the procedure. It first gathers the evidence, including the files fixed in similar past issues that were resolved before the report was filed. The model then makes only three bounded decisions: which files to open, what role each file plays
    
[^31]: 超越提示词：编码智能体的运行时契约

    Beyond Prompts: Runtime Contracts for Coding Agents

    [https://arxiv.org/abs/2611.19549](https://arxiv.org/abs/2611.19549)

    该论文提出在编码智能体工作期间强制执行的可执行契约，实验证明相比提示词规则、检索式指导或生成后检查，运行时强制执行能完全消除规则违规且不损失功能正确性。

    

    编码智能体通过 shell 命令和文件编辑对代码仓库进行操作，团队则用书面工程规则来引导它们。工业场景期望这类规则能够被强制执行，并且其应用情况能够被证实，而以提示文本形式给出的规则两者皆无法提供。我们测试了在智能体工作时强制执行的可执行契约，与将相同规则置于提示词中、作为指导检索、或仅在生成后检查相比，是否能减少规则违规。违规行为由一个与循环内检查方法不同的预言机进行评分，该预言机由两名人工审查者和一个模型审查者验证。在针对两个模型系列的三项研究（240、112 和 108 次运行）中，执行期间的强制实施在 104 组配对运行中未产生任何预言机检测到的违规，而无规则时的违规率为 21% 至 55%，且在两项验证性研究中功能正确性没有损失。在针对已知失败场景任务的验证性研究中

    arXiv:2611.19549v1 Announce Type: cross  Abstract: Coding agents act on repositories through shell commands and file edits, and teams steer them with written engineering rules. Industrial settings expect such rules to be enforced and their application to be evidenced, and a rule given as prompt text provides neither. We test whether executable contracts enforced while the agent works reduce rule violations compared with the same rules in the prompt, retrieved as guidance, or checked only after generation. Violations are scored by an oracle whose methods differ from the in-loop checks, and the oracle is validated by two human reviewers and a model reviewer. Across three studies on two model families (240, 112 and 108 runs), enforcement during execution produced no oracle-detected violation in 104 paired runs, against 21 to 55% of runs without rules, and no loss of functional correctness in the two confirmatory studies. In the confirmatory study on tasks aimed at known failure situations
    
[^32]: CLEW：基于日志锚定的反向追踪根因代码定位方法

    CLEW: Log-Anchored Backward Tracing for Root-Cause Code Localization

    [https://arxiv.org/abs/2611.19448](https://arxiv.org/abs/2611.19448)

    CLEW是一种结合规则式日志到代码锚定与LLM引导反向代码追踪的智能体方法，可自动定位软件故障的根因函数并输出带机制路径与日志证据的可解释诊断结果。

    

    工业软件系统中的故障可能涉及多个组件，并且在生产环境之外难以复现。因此，根因代码定位通常依赖于问题描述和运行时日志。然而，日志证据可能存在噪声或不完整，且报告症状的代码可能不同于包含潜在缺陷的代码。这些条件使得识别相关日志证据以及确定哪些函数可能导致所报告的故障变得极具挑战性。为解决这一空白，我们提出了CLEW，一种结合基于规则的日志到代码锚定与大语言模型引导的反向代码追踪的智能体方法。CLEW将筛选出的观察结果组织为诊断状态，从锚定的症状函数出发引导源代码探索，并返回排序后的候选函数、受影响的模块，以及包含建议机制路径和相关日志事件的解释。在评估方面，我们构建了……（原文摘要在此处被截断）

    arXiv:2611.19448v1 Announce Type: cross  Abstract: Failures in industrial software systems may involve multiple components and be difficult to reproduce outside the production environment. Root-cause code localization therefore often relies on issue descriptions and runtime logs. However, log evidence may be noisy or incomplete, and the code reporting a symptom may differ from the code containing the underlying defect. These conditions make it challenging to identify relevant log evidence and determine which functions could have caused the reported failure. To address this gap, we introduce CLEW, an agentic approach that combines rule-based log-to-code anchoring with LLM-guided backward code tracing. CLEW organizes selected observations into diagnostic states, guides source exploration from anchored symptom functions, and returns ranked candidate functions, an affected module, and an explanation comprising a proposed mechanism path and associated log events. For evaluation, we construc
    
[^33]: 超越笨拙的智能体工具调用：提示与识别如何带来改变

    Beyond Clumsy Agentic Tool Calling: How Cueing and Spotting Make the Difference

    [https://arxiv.org/abs/2611.19275](https://arxiv.org/abs/2611.19275)

    本文提出在线中间件ToolPulse，通过提示与识别机制监测智能体的笨拙工具调用与跨交互的工具使用弯路，并在调用前后及时注入提醒、结构化反馈或上下文，从而显著提升LLM智能体自主使用工具的能力。

    

    基于大语言模型（LLM）的任务型智能体依赖工具与环境进行交互，但可能忽视执行反馈、重复无效调用，或固执地坚持低效策略。这些行为会削弱智能体自主使用工具并持续推动任务进展的能力。我们从两个时间粒度刻画了工具使用症状：局部的笨拙工具使用行为，以及跨越多次交互的工具使用弯路。为了实现更自主的工具使用，我们提出了ToolPulse——一个支持自主工具使用的在线中间件，其核心是提示与识别机制。ToolPulse跟踪执行状态与时间交互模式，在调用分发之前监控拟议的调用，并在智能体恢复推理之前检查执行反馈。它将基于状态的规则、语义序列分析和交互特征相似性相结合，以识别干预时机，并提供言语提醒、结构化反馈或上下文……

    arXiv:2611.19275v1 Announce Type: new  Abstract: LLM-based task-oriented agents depend on tools to interact with their environments, but may overlook execution feedback, repeat ineffective calls, or persist with unproductive strategies. These behaviors can hinder agents' ability to use tools autonomously and sustain task progress. We characterize tool-use symptoms at two temporal granularities: local clumsy tool-use behaviors and tool-use detours spanning multiple interactions. Towards greater autonomy in tool use, we present ToolPulse, an online middleware supporting autonomous tool use through cueing and spotting. ToolPulse tracks execution state and temporal interaction patterns, monitoring proposed calls before dispatch and inspecting execution feedback prior to the agent resumes reasoning. It combines state-based rules, semantic sequence analysis, and interaction-feature similarity to identify intervention opportunities and provide verbal reminders, structured feedback, or context
    
[^34]: 小型测试桩、默认检查、有限覆盖：实践中的有界模型检测

    Small Harnesses, Default Checks, Limited Coverage: Bounded Model Checking in Practice

    [https://arxiv.org/abs/2611.19267](https://arxiv.org/abs/2611.19267)

    该论文对开源 C 与 Rust 项目中使用 CBMC 和 Kani 进行有界模型检测的实际实践开展了实证研究，刻画了验证属性、工件复杂度与验证结果，揭示了实践中存在测试桩规模小、依赖默认检查且覆盖有限等问题，为构建自动化软件验证系统提供了重要的实证基础。

    

    随着人工智能在生成软件和验证工件方面应用的日益增多，对能够提供可靠软件保障的自动化系统的需求也随之增长。有界模型检测通过直接在软件上自动检查正确性和安全属性，为这类系统提供了实用的基础。然而，有效的验证需要对验证属性、输入假设、函数模型和执行边界等做出选择，而这些选择会影响所获得的保证程度。目前关于从业者如何做出这些选择的系统性证据很少，这限制了开发自动化验证系统的实证基础。我们对开源 C 和 Rust 项目中使用 CBMC 和 Kani 的有界模型检测实践进行了一项实证研究。通过大语言模型辅助的开放式编码、静态分析和工件执行，我们刻画了验证属性、工件复杂度以及验证结果。我们的发现揭示了多样化的验证目标……（原文摘要在此处截断）

    arXiv:2611.19267v1 Announce Type: new  Abstract: The growing use of AI to generate software and verification artifacts has increased the need for automated systems that provide reliable software assurance. Bounded model checking (BMC) offers a practical foundation for such systems by automatically checking correctness and security properties directly on software. However, effective verification requires choices about properties, input assumptions, function models, and execution bounds that influence the guarantees obtained. Little systematic evidence exists on how practitioners make these choices, limiting the empirical foundation for developing automated verification systems.   We present an empirical study of BMC practices in open-source C and Rust projects using CBMC and Kani. Through LLM-assisted open coding, static analysis, and artifact execution, we characterize verification properties, artifact complexity, and verification outcomes. Our findings reveal diverse verification obje
    
[^35]: Teralizer：一个将JUnit测试泛化为基于属性的jqwik测试的工具

    Teralizer: A Tool for Generalizing JUnit Tests into Property-Based jqwik Tests

    [https://arxiv.org/abs/2611.19242](https://arxiv.org/abs/2611.19242)

    Teralizer通过单路径符号分析自动将JUnit测试泛化为基于属性的jqwik测试，在输入生成与输出预言两个层面增强测试，从而显著提升回归检测能力。

    

    背景：基于属性的测试通过在大量生成的输入上验证属性，能够检测出传统单元测试遗漏的回归缺陷。然而，编写基于属性的测试对从业者而言具有挑战性，这限制了该技术的实际应用。工具：Teralizer能够自动将JUnit测试泛化为用于回归检测的基于属性的jqwik测试。通过单路径符号分析，它从每次测试执行中提取一个路径条件（刻画遵循同一执行路径的输入特征）和一个符号输出表达式（为这些输入计算期望输出）。泛化后的测试会生成满足路径条件的输入，并将符号输出表达式用作测试预言。结果：在我们的评估中，泛化使EvoSuite生成的测试套件的变异得分提高了0.8至3.9个百分点。在EqBench基准测试中，泛化EvoSuite生成的测试套件……（原文在此处截断）

    arXiv:2611.19242v1 Announce Type: new  Abstract: Background: Property-based testing validates properties across many generated inputs and can thereby detect regressions that conventional unit tests miss. However, writing property-based tests is challenging for practitioners, which limits their adoption. Tool: Teralizer automatically generalizes JUnit tests into property-based jqwik tests for regression detection. Through single-path symbolic analysis, it extracts from each test execution a path condition, which characterizes the inputs that follow the same execution path, and a symbolic output expression, which computes the expected output for these inputs. The generalized tests generate inputs that satisfy the path condition and use the symbolic output expression as the oracle. Results: In our evaluation, generalization increased the mutation scores of EvoSuite-generated test suites by 0.8 to 3.9 percentage points. On the EqBench benchmark, generalizing the test suites that EvoSuite g
    
[^36]: 量子程序分析中的实时噪声指纹

    Live Noise Fingerprints for Quantum Program Analysis

    [https://arxiv.org/abs/2611.19143](https://arxiv.org/abs/2611.19143)

    本文提出“噪声指纹”这一紧凑的抽象表示，并通过概率伽罗瓦连接将其与具体噪声行为关联，实现了对含噪量子程序的高效可靠分析，且多项式规模的接口可产生比完整噪声模型指数级更小的指纹。

    

    量子程序本质上是概率性的，而当前不完美硬件所导致的含噪执行进一步增加了其分析的难度。本文研究了在此类条件下分析量子程序的抽象语义。我们引入了“噪声指纹”的概念，即相对于有限的输入状态与输出测量接口，对噪声可观测效应的紧凑描述。我们利用噪声指纹来定义关于具体噪声相关行为的抽象域。我们通过概率伽罗瓦连接将具体域与抽象域关联起来，将伽罗瓦连接框架扩展到了以分布为取值的解释上。我们从观测可靠性、简洁性、效率以及适应噪声变化的能力等方面对所得的抽象进行了研究。我们证明，多项式规模的接口在保持所选观测结果的同时，能够产生比完整噪声模型指数级更小的指纹，并且……

    arXiv:2611.19143v1 Announce Type: new  Abstract: Quantum programs are probabilistic in nature, and noisy execution due to current imperfect hardware further complicates their analysis. In this paper, we study abstract semantics for analysing quantum programs under these conditions. We introduce noise fingerprints, compact descriptions of the observable effects of noise relative to a finite interface of input states and output measurements. We use them to define our abstract domain with respect to the concrete noise-related behaviour. We relate the concrete and abstract domains through a probabilistic Galois connection, extending the Galois-connection framework to distribution-valued interpretations. We study the resulting abstraction in terms of observational soundness, conciseness, efficiency, and ability to adapt to changing noise. We show that polynomial-size interfaces yield exponentially smaller fingerprints than full noise models while preserving the chosen observations, and that
    
[^37]: 当“人月神话”遇见智能体网络系统：面向可扩展开发的结构化协调

    When the Mythical Man-Month Meets Agentic Web Systems: Structured Coordination for Scalable Development

    [https://arxiv.org/abs/2611.19140](https://arxiv.org/abs/2611.19140)

    本文借鉴软件工程经典著作《人月神话》的思想，系统揭示了多智能体系统中智能体规模与协调开销之间的非线性关系，并提出结构化协调方法以在任务性能与协调成本之间取得平衡，从而实现可扩展的智能体网络系统开发。

    

    随着智能体系统日益被部署用于支持复杂的网络任务，多智能体架构能够提供更强的并行性、专业化和任务覆盖能力。然而，增加智能体数量并不一定能提升系统性能，因为通信、同步、冗余工作和协调开销可能会抵消额外智能体所带来的收益。这给智能体网络系统提出了一个根本性的扩展挑战：如何组织智能体种群，以平衡任务性能与协调成本？为应对这一挑战，我们系统性地研究了协作式智能体系统中智能体种群规模、单智能体token消耗与协调开销之间的关系。在复杂的面向网络的软件开发任务上进行的实验揭示了一种非线性扩展模式：随着智能体种群规模的增长，单智能体的token消耗最初呈下降趋势，但最终会表现出收益递减的现象。

    arXiv:2611.19140v1 Announce Type: cross  Abstract: As agentic systems are increasingly deployed to support complex Web tasks, multi-agent architectures offer greater parallelism, specialization, and task coverage. However, increasing the number of agents does not necessarily improve system performance, as communication, synchronization, redundant work, and coordination overhead may offset the benefits of additional agents. This presents a fundamental scaling challenge for agentic Web systems: how can agent populations be organized to balance task performance and coordination costs? To address this challenge, we systematically investigate the relationship between agent population size, per-agent token consumption, and coordination overhead in collaborative agentic systems. Experiments on complex Web-oriented software development tasks reveal a non-linear scaling pattern: per-agent token consumption initially decreases as the agent population grows, but eventually exhibits diminishing re
    
[^38]: 超越朴素MCP：决策模型与代码模式能让工具调用更便宜、更快速吗？

    Beyond Plain MCP: Do Decision Models and Code Mode Make Tool Use Cheaper and Faster?

    [https://arxiv.org/abs/2611.19123](https://arxiv.org/abs/2611.19123)

    该研究通过预注册控制实验证明，让智能体直接针对工具存根编写脚本的“代码模式”可大幅削减70%-90%的token消耗并显著提升任务效用，其中在存根中声明返回形状提示是关键因素，而在其前叠加决策层路由器并未带来显著额外收益。

    

    模型上下文协议（MCP）网关为语言模型智能体管理工具目录，主要采用两种范式：代码模式——智能体针对工具存根编写脚本，以及决策层——预选工具子集的低成本路由器。我们使用受控测试框架，覆盖12个服务器、多达1,958个干扰工具，以及包括Qwen、Gemma和Nemotron在内的三个开源权重智能体，在密封的测试集上开展了一项预注册评估，以确定每种方法何时能够奏效。结果表明，代码模式显著优于朴素MCP，可将token使用量削减70%至90%，同时大幅提升任务效用。在代码模式存根中声明返回形状提示是一个一阶变量，可使表现较弱的智能体的成功率波动高达39.5个百分点，其影响甚至大于添加路由器。而核心假设——在代码模式之前部署决策层会优于单独使用代码模式——并未得到证实，仅显示出微小且在统计上不显著的改进。

    arXiv:2611.19123v1 Announce Type: cross  Abstract: Model Context Protocol (MCP) gateways manage tool catalogs for language model agents using two primary paradigms: code mode where agents write scripts against tool stubs and decision layers which are cheap routers that preselect tool subsets. Using a controlled harness across 12 servers, up to 1,958 distractor tools, and three open-weight agents including Qwen, Gemma, and Nemotron, we present a preregistered evaluation on a sealed test split to determine when each approach succeeds. Code mode substantially outperforms plain MCP, cutting token usage by 70 to 90 percent while significantly boosting utility. Declaring return-shape hints in code mode stubs is a first-order variable, swinging success by up to 39.5 points for lower-performing agents, an effect larger than adding a router. The central hypothesis that a decision layer in front of code mode beats code mode alone is not confirmed, showing minor, statistically non-significant imp
    
[^39]: 超越基准测试：评估大语言模型与智能体模式在静态代码分析警告分类中的应用

    Beyond Benchmarks: Evaluating Large Language Models and Agentic Patterns for Static Code Analysis Warning Classification

    [https://arxiv.org/abs/2611.19086](https://arxiv.org/abs/2611.19086)

    本研究利用四个手动标注数据集评估三个当代大语言模型在真实场景下静态代码分析警告分类的性能，并通过引入智能体模式和既定编码标准来提升分类效果。

    

    背景：静态代码分析（SCA）工具支持软件验证与确认，能够减少代码审查的时间和成本。然而，从业者社区中最好的SCA工具会产生大量误报，开发者往往会忽略这些误报，从而导致真正的警告被忽视。因此，越来越多的研究采用大语言模型（LLM）来识别和减少误报。现有研究在公开数据集上取得了有前景的结果，但在未见过的数据集上表现不佳。此外，研究人员没有充分利用LLM中智能体模式的全部潜力，也没有将已有的代码质量标准纳入警告分类中。目标：本研究利用四个手动标注的数据集，评估三个当代大语言模型在真实应用场景中进行SCA警告分类的性能，并结合智能体模式和既定编码标准来提升模型表现。

    arXiv:2611.19086v1 Announce Type: new  Abstract: Context: Static Code Analysis (SCA) tools support software verification and validation, reducing the time and cost of code reviews. However, the best SCA tools in the practitioner community generate a high volume of false positives, which developers tend to ignore, leading to genuine warnings being overlooked. Accordingly, studies increasingly adopt Large Language Models (LLMs) to identify and reduce false positives. Existing studies have produced promising results on public datasets, but underperformed on unseen datasets. Moreover, researchers do not leverage the full potential of Agentic patterns in LLMs or incorporate established code quality standards for warning classification. Objectives: This study utilises four manually annotated datasets to evaluate the performance of three contemporary LLMs for SCA warning classification in real-world applications, and incorporates Agentic patterns and established coding standards to enhance mo
    
[^40]: 使用机器学习方法对静态分析工具产生的误报进行分类——一项基于小数据集的实证研究

    Using machine learning approaches to classify false-positives generated by static analysis tools - An empirical study using small datasets

    [https://arxiv.org/abs/2611.19079](https://arxiv.org/abs/2611.19079)

    本研究在小数据集条件下系统评估了多种代码表示学习技术与机器学习算法来识别静态分析工具的误报，弥补了以往研究仅关注少量警报类型或评估不全面的不足，并发现最佳方法的选择取决于数据特征。

    

    静态代码分析（SCA）工具对于软件开发至关重要，能够降低人工代码审查所需的成本和时间。我们在先前的研究中已经证实，软件工程界所报道的最佳工具在用于代码分析时会返回较高的误报率和漏报率。现有研究已采取多种方法来识别SCA工具产生的误报警告，以帮助提升这些工具的实用性。然而，现有研究存在若干局限性，其中最主要的是倾向于只关注数量有限的警报类型，或者未能全面评估广泛的机器学习（ML）技术。本研究针对这一机遇展开。我们确定了用于识别误报的最佳代码表示学习（CRL）技术和机器学习算法，结果表明其性能表现会根据特征的不同而有所变化。

    arXiv:2611.19079v1 Announce Type: cross  Abstract: Static Code Analysis (SCA) tools are vital for software development to reduce the cost and time required for manual code reviews. It was established in our previous study that the best tools reported in the software engineering community return high false-positive and false-negative rates when used for code analysis. Several approaches have been taken in existing studies to identify false-positive alarms generated by SCA tools, in helping to enhance the utility of these tools. However, existing studies reveal several limitations, chiefly among which is the tendency to focus on a limited number of alarm types or failing to evaluate a large breadth of Machine-Learning (ML) techniques comprehensively. This study has addressed this opportunity. We identified the best Code Representation Learning (CRL) techniques and ML algorithms for identifying false-positives, where results show that performance outcomes vary based on the characteristics
    
[^41]: 更多上下文，结果参差不一：评估大语言模型在标识符恢复与代码反混淆中的应用

    More Context, Mixed Results: Evaluating LLMs for Identifier Recovery and Code Deobfuscation

    [https://arxiv.org/abs/2611.19043](https://arxiv.org/abs/2611.19043)

    该研究在51个Java项目上系统评估了大语言模型与传统方法在代码反混淆的标识符检测与语义恢复任务中的表现，发现LLM整体优于传统技术（检测F1最高达91.56%），但精确恢复率仍然偏低（总体峰值仅16.42%），且不同上下文形式对不同任务的效果存在差异。

    

    源代码经常通过标识符重命名的方式进行混淆，这促使逆向工程师开发恢复有意义名称的技术。为了评估大语言模型（LLM）与这些技术的对比效果，我们在51个近期Java项目、3种混淆器和6种上下文/检索设置上，评估了4种传统基线方法和3个大语言模型（Qwen、OLMo和DeepSeek）在标识符检测和语义恢复方面的表现。大语言模型在这两项任务上普遍优于传统技术，但受益于不同形式的上下文和设置：智能体式检索（agentic retrieval）产生了最强的检测效果，F1分数达到91.56%，而包含类上下文通常能改善语义恢复效果，但需要更高的token开销。尽管取得了这些进展，精确恢复率总体峰值仅为16.42%（在单跳上下文下），字段名称为32.93%（在仓库检索下），其中字段名称是最容易恢复的部分。类标识符最难检测，平均检测F1分数仅为48.48%。

    arXiv:2611.19043v1 Announce Type: cross  Abstract: Source code is often obfuscated through identifier renaming, incentivizing reverse engineers to develop techniques for recovering meaningful names. To assess how LLMs compare with these techniques, we evaluate 4 traditional baselines and 3 LLMs (Qwen, OLMo and DeepSeek) on identifier detection and semantic recovery across 51 recent Java projects, 3 obfuscators, and 6 context/retrieval settings. LLMs generally outperform traditional techniques on both tasks, but benefit from different forms of context and settings: agentic retrieval yields the strongest detection results, reaching 91.56% $F_1$, while enclosing-class context generally improves semantic recovery at a higher token cost. Despite these gains, exact recovery peaks at 16.42% overall (under one-hop context) and 32.93% for field names (under repository retrieval), which are the easiest to recover. Class identifiers are the hardest to detect, with a mean detection $F_1$ of 48.48%
    
[^42]: 上下文很重要：利用层次化信息改进基于大语言模型的代码摘要

    Context Matters: Improving LLM-Based Code Summarization with Hierarchical Information

    [https://arxiv.org/abs/2611.18964](https://arxiv.org/abs/2611.18964)

    该论文通过向大语言模型提供函数、类和模块三个层次的上下文信息，验证了层次化上下文增强能够显著改进代码级文档摘要的生成质量。

    

    源代码文档在软件质量活动中扮演着重要角色，例如代码审查、程序理解、维护和缺陷修复。然而，文档往往分散在各个单独的代码元素中，并侧重于局部实现细节，几乎不提供有关其周围上下文的信息。这既限制了开发人员对软件系统的理解，也限制了大语言模型的有效性，因为大语言模型的推理受限于有限的上下文窗口。用层次化上下文来丰富文档，有可能改善软件理解，并为开发人员以及基于大语言模型的工具的下游质量相关任务提供支持。在本文中，我们提出了一项计算仿真研究，展示了增加上下文对生成代码级文档的影响，实验使用了多种大语言模型。通过改变大语言模型可见的函数、类和模块层次结构，我们评估是否……

    arXiv:2611.18964v1 Announce Type: cross  Abstract: Source code documentation plays an important role in software quality activities, such as code review, program comprehension, maintenance, and bug fixing. However, it is often fragmented across individual code elements and focuses on local implementation details, providing little information about their surrounding context. This limits both developers' understanding of software systems and the effectiveness of Large Language Models (LLMs), whose reasoning is constrained by finite context windows. Enriching documentation with hierarchical context has the potential to improve software comprehension and support downstream, quality-related tasks for both developers and LLM-based tools. In this paper, we present an in silico study that showcases the impact of augmenting context on the generation of code-level documentation, using a variety of LLMs. By varying the function, class, and module hierarchy visible to the LLMs, we assess whether t
    
[^43]: 从特定语言到跨DSL：基于语义角色引导的LLM模型补全

    From Language-Specific to Cross-DSL: Semantic-Role-Guided Model Completion with LLMs

    [https://arxiv.org/abs/2611.18952](https://arxiv.org/abs/2611.18952)

    本文提出了RAFiC——一种可跨DSL泛化的基于LLM的迭代式领域模型补全方法，通过元模型信息保证语法有效性，并借助语义角色标注丰富提示词以捕捉建模意图，在模型构建过程中提供上下文感知的增量式指导。

    

    大语言模型（LLM）在海量且多样化的语料库上训练而成，其迅速兴起使其成为协助建模人员获取领域知识的一个有前景的来源。然而，利用LLM进行模型补全仍然充满挑战：其给出的建议必须同时遵循领域特定语言（DSL）的语法和预期的领域语义，并与不断演化的模型保持一致。本文提出了RAFiC，这是一种基于LLM的迭代式领域模型补全方法，可泛化应用于不同的DSL。该方法利用元模型信息来驱动补全，以确保语法的有效性，并通过语义角色标注来丰富提示词，从而更好地捕捉DSL元素的预期建模角色。该方法能够在模型构建过程中提供上下文感知的增量式指导。我们通过对三种DSL（每种DSL包含两个场景）开展基于消融的对比研究来评估RAFiC，以分离出语法和语义（两方面）各自的贡献……

    arXiv:2611.18952v1 Announce Type: cross  Abstract: The rapid emergence of large language models (LLMs), trained on vast and diverse corpora, has positioned them as a promising source of domain knowledge to assist modelers. However, leveraging LLMs for model completion remains challenging: suggestions must respect both the syntax of a domain-specific language (DSL) and the intended domain semantics, while remaining consistent with an evolving model. This paper presents RAFiC, an LLM-based approach for iterative domain model completion that generalizes across DSLs. It drives completion using metamodel information to ensure syntactic validity and enriches prompts with semantic-role annotations to better capture the intended modeling role of DSL elements. The approach provides context-aware, incremental guidance during model construction. We evaluate RAFiC through an ablation-based comparative study over three DSLs and two scenarios per DSL to isolate the contribution of syntactic and sema
    
[^44]: 挖掘Unity VR仓库中的重构模式与技术债务

    Mining Unity VR Repositories for Refactoring Patterns and Technical Debt

    [https://arxiv.org/abs/2611.18948](https://arxiv.org/abs/2611.18948)

    该研究从176个开源Unity VR仓库中挖掘重构模式与技术债务，发现代码债务最为普遍，交互债务和UI/UX债务是VR应用的主要债务类别，并开发了VRRefMiner工具来检测Unity应用中的重构。

    

    维护虚拟现实（VR）应用面临着与实时性能、硬件依赖和复杂交互相关的挑战。然而，人们对开发人员如何重构这些应用、他们解决哪些类型的技术债务，以及自动化工具能否很好地捕捉VR特定的变更知之甚少。我们从176个开源Unity VR仓库中挖掘了1,755个候选重构提交，并手动检查了其中303个提交的样本，其中270个包含至少一次重构。我们还扩展了RefactoringMiner，创建了VRRefMiner，这是一个用于检测Unity应用所使用的C#脚本中常规重构的工具。我们的分析同时发现了常规重构和VR特定的重构。VR特定的变更主要解决了交互逻辑、性能以及Unity特有产物等方面的问题。代码债务是总体上最普遍的技术债务类别，而交互债务和UI/UX债务则是VR应用中主要被处理的技术债务类别。

    arXiv:2611.18948v1 Announce Type: new  Abstract: Maintaining virtual reality (VR) applications poses challenges tied to real-time performance, hardware dependencies, and complex interactions. Yet little is known about how developers refactor these applications, which kinds of technical debt they address, or how well automated tools capture VR-specific changes. We mined 1,755 candidate refactoring commits from 176 open-source Unity VR repositories and manually examined a sample of 303 commits, of which 270 contained at least one refactoring. We also extended RefactoringMiner to create VRRefMiner, a tool that detects conventional refactorings in C# scripts used by Unity applications.   Our analysis found both conventional and VR-specific refactorings. VR-specific changes addressed concerns such as interaction logic, performance, and Unity-specific artifacts. Code debt was the most prevalent category overall, while interaction debt and UI/UX debt were the main categories addressed by VR-s
    
[^45]: 升级！AI驱动的代码审查实践挑战与适应

    Level Up! AI-Driven Challenges and Adaptations to Code Review Practices

    [https://arxiv.org/abs/2611.18939](https://arxiv.org/abs/2611.18939)

    通过对大型跨国软件公司2614名开发者的访谈与大规模问卷调查，本研究首次以实证方法揭示了AI时代代码审查面临的核心挑战及有效应对策略，发现AI正推动人类审查者角色向设计与意图等高层次关注转变，同时AI辅助编码因作者缺乏尽责管理而加重了审查负担。

    

    AI编写代码的速度超过了人类审查代码的速度，造成了瓶颈，并 necessitating 改变代码审查实践。这一瓶颈引发了激烈争论，催生了两种极端且相互正交的想象未来：一种是人类继续牢牢坚守在审查流程中，另一种是AI智能体在完全无需人类参与的情况下合并拉取请求。然而，这些未来图景仍植根于民间经验之谈。我们通过对一家大型跨国软件公司（该公司员工可广泛使用AI）开展混合方法研究——包括访谈（n = 24）和问卷调查（n = 2,590）——识别出了AI驱动的代码审查挑战，以及被认为能有效应对这些挑战的策略。我们发现，AI使人类审查者的角色向高层次关注点（如设计与意图）的转变成为可能、成为必要，同时也使其更加复杂。我们还发现，AI辅助编写代码由于存在作者缺乏尽责管理（author stewardship）等挑战，给审查者带来了更重的负担。

    arXiv:2611.18939v1 Announce Type: new  Abstract: AI can write code faster than humans can review, causing a bottleneck and necessitating changes to code review practices. This bottleneck is hotly debated, giving rise to two extreme and orthogonal imagined futures: one where humans continue to be firmly entrenched in the review process, and one where AI agents merge pull requests entirely without human involvement. Yet, these futures remain rooted in folk wisdom. Through a mixed-methods study of interviews ($n = 24$) and a survey ($n = 2{,}590$) at a large, multinational software company with widespread access to AI, we identify AI-driven code review challenges and strategies reported as effective to meet them. We found that AI enables, necessitates, and complicates a shift in the human reviewer's role toward high-level concerns (e.g., design and intent). We found that AI-assisted authoring results in increased burden for reviewers due to challenges such as a lack of author stewardship 
    
[^46]: 在严谨与现实之间：AI安全基准测试开发者如何理解基准的使用与维护

    Between Rigor and Reality: How AI Safety Benchmark Developers Understand Benchmark Use and Maintenance

    [https://arxiv.org/abs/2611.18930](https://arxiv.org/abs/2611.18930)

    通过对学术界和产业界的AI安全基准测试开发者进行访谈，本研究揭示了安全基准测试的预期用途与实际部署系统安全评估之间的脱节，指出评分者间信度难以实现和外部效度威胁应对不清这两大挑战，并提出基准测试必须与部署环境主动融合。

    

    安全基准测试被公认为是理解生成式AI（GenAI）系统所带来风险的主要工具，然而越来越多的证据表明，这些基准往往无法满足现实世界安全评估的需求。我们展示了对学术界和产业界的AI安全基准测试开发者进行的五次访谈的结果。我们发现，人们对安全基准测试预期有用性的认知与其在已部署系统安全评估中的实际有用性之间存在脱节。我们的受访者对造成这种脱节的两个挑战提出了看法：（1）在安全概念上难以实现评分者间信度，以及（2）在如何应对外部效度面临的持续性威胁方面缺乏明确的认识。基于我们的分析，我们认为安全基准测试不仅要立足于部署环境，还必须与部署环境主动融合。为促进这一目标的实现，我们概述了实现更透明沟通的途径。

    arXiv:2611.18930v1 Announce Type: new  Abstract: Safety benchmarks are cited as the primary tool to understand the risks posed by generative AI (GenAI) systems, yet growing evidence suggests they often fail to meet the needs of real-world safety evaluation. We present findings from five interviews with AI safety benchmark developers in academia and industry. We find a disconnect between perceptions of the intended usefulness of safety benchmarks and their practical usefulness in safety evaluations of deployed systems. Our participants offered perspectives on two challenges contributing to this disconnect: (1) the difficulty of achieving inter-rater reliability on safety constructs, and (2) a lack of clarity regarding how to address persistent threats to external validity. Based on our analysis, we argue that safety benchmarks must not only be grounded in deployment contexts, but actively integrated with them. To facilitate this, we outline pathways for more transparent communication be
    
[^47]: 重新审视可重执行性：覆盖率引导的符号反馈以改进反编译

    Revisiting Re-executability: Coverage-Guided Symbolic Feedback to Improve Decompilation

    [https://arxiv.org/abs/2611.18892](https://arxiv.org/abs/2611.18892)

    SymDec系统利用覆盖率引导的符号执行对比原始二进制与重编译二进制的I/O行为，将暴露行为差异的输入作为结构化反馈指导LLM修复反编译代码，从而超越固定测试套件、真正确保行为保真。

    

    当源代码不可用时，反编译可以为二进制文件提供源代码级别的表示。然而，通过传统方法或大语言模型（LLM）辅助反编译恢复的代码可能无法保留原始二进制文件的行为。近期的方法使用固定测试套件来评估可重执行性，但通过这些测试套件并不能确保行为保真。我们提出了SymDec，一个以原始二进制文件作为行为参考的LLM辅助反编译代码修复系统。SymDec使用覆盖率引导的符号执行来探索原始二进制与重编译二进制在I/O层面的行为，并通过符号比较识别出暴露行为差异的输入。这些输入随后在两个二进制文件上重放以确认差异，并作为结构化反馈返回，用于指导后续的LLM代码修订。我们在来自ExeBench和DecompileBench的120个函数上评估了SymDec，这些函数针对x86-64和ARM64架构在不同优化级别下编译。

    arXiv:2611.18892v1 Announce Type: new  Abstract: When source code is unavailable, decompilation provides a source-level representation of a binary. However, code recovered by conventional or LLM-assisted decompilation may not preserve the original behavior of the binary. Recent approaches evaluate re-executability using fixed test suites, but passing these suites does not ensure behavioral preservation. We present SymDec, an LLM-assisted decompiled-code repair system that uses the original binary as a behavioral reference. SymDec uses coverage-guided symbolic execution to explore the I/O-level behavior of the original and recompiled binaries, and symbolic comparison to identify inputs that expose behavioral differences. These inputs are replayed on both binaries to confirm the differences, which are returned as structured feedback to guide subsequent LLM revisions. We evaluate SymDec on 120 functions from ExeBench and DecompileBench, compiled for x86-64 and ARM64 at optimization levels
    
[^48]: Benchmark Factory：通过数据集策划与无标签质量锚点的智能体式基准测试创建

    Benchmark Factory: Agentic Benchmark Creation via Dataset Curation and Label-Free Quality Anchors

    [https://arxiv.org/abs/2611.18890](https://arxiv.org/abs/2611.18890)

    Benchmark Factory提出了一种带有人类检查点的自主元智能体循环，能够从生产代码变更中自动策划、验证和修复编码智能体基准测试，其质量锚点无需每项任务的人工标签。

    

    部署编码智能体的团队需要基于自己的代码构建基准测试，但人工策划和验证这些基准测试非常耗费资源，而未经验证的基准测试可能会悄悄地产生误导。我们提出了Benchmark Factory，这是一个带有人类检查点的自主元智能体循环：LLM智能体针对已落地的生产代码变更编写数据采集查询，在每个候选任务上运行已知相对强度的模型，诊断候选任务失败的原因，并通过编辑其查询、提示词和配置来修复它，直到它通过无需每项任务人工标签的质量锚点。我们使用Benchmark Factory为组织内的五个开发者群体构建了基准测试。工程评审人员对5%的样本进行了评分，任务质量得分为2.3，评判器判定得分为2.6（1至3分制）。在从未用于构建这些基准测试的配置上，基准测试将高推理努力的Claude Opus 4.7排名高于低推理努力（60.6% vs 54.4%的任务通过率，p=0.002），以及GPT-5（摘要内容在此处被截断）。

    arXiv:2611.18890v1 Announce Type: cross  Abstract: Teams deploying coding agents need benchmarks built from their own code, but curating and validating them by hand is resource intensive, and an unvalidated benchmark can mislead silently. We present Benchmark Factory, an autonomous meta-agent loop with human checkpoints: an LLM agent writes sourcing queries over landed production code changes, runs models of known relative strength on each candidate, diagnoses why a candidate fails, and repairs it by editing its queries, prompts, and configuration until it clears quality anchors that need no per-task human labels. We used Benchmark Factory to build benchmarks for five developer cohorts at our organization. Engineering raters scored a 5% sample 2.3 for task quality and 2.6 for grader verdicts on a 1-to-3 scale. On configurations never used to build them, the benchmarks ranked Claude Opus~4.7 at high reasoning effort above low effort (60.6% vs. 54.4% of tasks passed, $p=0.002$) and GPT-5
    
[^49]: 基于稳定子态的量子软件蜕变测试

    Metamorphic Testing of Quantum Software based on Stabilizer States

    [https://arxiv.org/abs/2611.18838](https://arxiv.org/abs/2611.18838)

    提出了一种量子蜕变测试（QMT）框架，将评估量子输出所需的测量过程融入蜕变关系中，从而解决了因量子态无法直接检查而导致的测试预言机难题。

    

    量子计算中的一个重大挑战是如何系统地检验量子程序是否实现了其预期功能。对于经典程序，测试被广泛使用；每个测试用例由一对样本输入和预期输出组成，用于与实际输出进行比较。然而，要增加测试用例的数量，就必须解决测试预言机问题：对于每个新增的测试输入，都需要知道正确的输出。蜕变测试通过利用输入域与输出域之间的蜕变关系，从现有测试用例中自动推导出新的测试用例，从而解决了这一问题。然而，将蜕变测试应用于量子领域并非易事，因为量子态无法被直接检查。我们提出了一个量子蜕变测试（QMT）框架，其中的蜕变关系融入了评估量子输出所需的测量过程。我们将QMT框架应用于测试……

    arXiv:2611.18838v1 Announce Type: cross  Abstract: A major challenge in quantum computing is to systematically check if a quantum program implements its desired functionality. For classical programs, testing is widely used; each test case comprises a pair of sample input and expected output to be compared with the actual output. To increase the number of test cases, however, one has to solve the test-oracle problem: for each additional test input one has to know the correct output. Metamorphic testing tackles this problem by utilizing metamorphic relations between input and output domains to automatically derive new test cases from existing ones. However, adapting metamorphic testing to the quantum realm is not straightforward, since quantum states cannot be directly inspected. We propose a framework of quantum metamorphic testing (QMT) in which the metamorphic relations incorporate the measurement procedures required to assess quantum outputs. We apply our QMT framework to testing the
    
[^50]: KernelArmor：为CUDA建立内存与线程安全的端到端智能体系统

    KernelArmor: End-to-End Agentic System for Establishing Memory and Thread Safety for CUDA

    [https://arxiv.org/abs/2611.18754](https://arxiv.org/abs/2611.18754)

    KernelArmor是一个将LLM生成注释与扩展SMT验证器相结合的智能体框架，能够在显式契约下为使用张量核心等高级特性的CUDA内核建立内存与线程安全保证，并在智能体优化循环中生成经验证的内核。

    

    经过优化的CUDA内核即使通过了测试，仍可能隐藏微妙的内存和并发错误。我们提出KernelArmor，这是一个智能体框架，它将LLM生成的注释与扩展的SMT验证器相结合，在显式契约下建立内存和线程安全保证。KernelArmor能够对使用高级CUDA特性的内核进行推理，例如张量核心、打包访问、warp同步、异步拷贝和库调用，并返回源代码级别的诊断信息以指导验证与修复。在包含1,370个生产级CUDA内核的740个实例中，它在78.6%的实例中报告了潜在错误，并为11.4%的实例建立了安全证明，在检测覆盖率上超越了所评估的最先进工具，识别出100%的cuCatch内存安全错误以及全部在范围内的cuHadron竞争条件。集成到智能体优化循环中后，KernelArmor在全部77个轨迹中生成了经过验证的内核，同时保留了99%的未（原文在此截断）

    arXiv:2611.18754v1 Announce Type: cross  Abstract: Optimized CUDA kernels can pass testing while retaining subtle memory and concurrency bugs. We present KernelArmor, an agentic framework that combines LLM-generated annotations with an extended SMT verifier to establish memory and thread safety under explicit contracts. KernelArmor can reason about kernels with advanced CUDA features, such as tensor cores, packed accesses, warp synchronization, asynchronous copies, and library calls, returning source-level diagnostics to guide verification and repair. Across 740 instances containing 1,370 production-grade CUDA kernels, it reports potential bugs in 78.6% of instances and establishes safety for 11.4%, outperforming the evaluated state-of-the-art tools in detection coverage, identifying 100% of cuCatch memory-safety bugs and all in-scope cuHadron races. Integrated into an agentic optimization loop, KernelArmor produces verified kernels in all 77 trajectories while retaining 99% of the unc
    
[^51]: DevSecOps中的大语言模型：软件生命周期中模型、证据与研究空白的系统性映射

    LLMs in DevSecOps: A Systematic Mapping of Models, Evidence, and Research Gaps Across the Software Lifecycle

    [https://arxiv.org/abs/2611.18632](https://arxiv.org/abs/2611.18632)

    本文对2024–2026年间LLM在DevSecOps六大阶段（计划、编码、测试、审查、部署、监控）的安全应用研究进行了系统性映射，揭示出研究分布不均衡——测试与编码阶段受到的关注远超部署与审查阶段，且近期系统日益依赖结构化上下文、迭代推理、微调、检索增强与智能体工作流等适配策略。

    

    大语言模型（LLM）正日益被整合到软件安全与DevSecOps工作流中，但其在生命周期各阶段的采用、评估和成熟度差异显著。本研究系统性地梳理了2024年至2026年（截至6月）发表的近期基于LLM的安全研究，涵盖六个DevSecOps阶段：计划、编码、测试、审查、部署和监控。针对每个阶段，我们分析了所解决的安全任务、所采用的模型与适配策略、评估设置、报告的有效性以及关键局限性。研究发现揭示了一个不均衡的研究格局：测试和编码阶段受到的关注远多于部署和审查阶段。纵观各阶段，近期的系统越来越多地依赖结构化上下文、迭代推理、微调、检索增强或智能体工作流，而非直接的单一推理方式。

    arXiv:2611.18632v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly being integrated into software security and DevSecOps workflows, yet their adoption, evaluation, and maturity vary substantially across lifecycle stages. This study systematically maps recent LLM-enabled security research published between 2024 and 2026 (until June) across six DevSecOps stages: \textit{Plan}, \textit{Code}, \textit{Test}, \textit{Review}, \textit{Deploy}, and \textit{Monitor}. For each stage, we analyze the security tasks addressed, models and adaptation strategies employed, evaluation settings, reported effectiveness, and key limitations. Our findings reveal an uneven research landscape, with \textit{Test} and \textit{Code} receiving substantially greater attention than \textit{Deploy} and \textit{Review}. Across stages, recent systems increasingly rely on structured context, iterative reasoning, fine-tuning, retrieval augmentation, or agentic workflows rather than direct si
    
[^52]: RGFC：基于注册表的函数调用与阶段分解评估

    RGFC: Registry Grounded Function Calling with stage-decomposed evaluation

    [https://arxiv.org/abs/2611.18622](https://arxiv.org/abs/2611.18622)

    该论文提出注册表规模函数调用基准 RFCD 与确定性阶段分解评估器 TAAG，并发现工具注册表规模（5 至 3,000 个函数）是模型函数调用性能退化的主要来源。

    

    arXiv:2611.18622v1 公告类型：新 摘要：工具调用评估通常只报告最终调用是否与参考答案匹配，但这一单一结果将不同的失败混为一谈：选择了可用注册表之外的函数、违反所选函数的架构规范，或使用了用户请求不支持的参数值填充参数。当模型必须从更大的工具注册表中进行选择时，这些失败问题更为突出。我们提出了 RFCD，这是一个基于 BFCL、Glaive Function Calling v2 和 Tool-Call-Data 构建的注册表规模函数调用基准，包含规模从 5 到 3,000 个函数的标准化 JSON-schema 注册表。我们还提出了 TAAG，一个用于评估注册表一致性、结构完整性和参数溯源的确定性阶段分解评估器。在六个本地部署的 4B 以下参数模型、三个数据源和四种反馈模式的实验中，注册表规模是性能退化的主导因素：在所有数据源上，单次通过 TAAG 的通过率从小注册表到大注册表均急剧下降。

    arXiv:2611.18622v1 Announce Type: new  Abstract: Tool-calling evaluations often report whether the final call matches a reference answer, but this single outcome merges different failures: choosing a function outside the available registry, violating the selected schema, or filling arguments with values not supported by the user request. These failures matter more when models must choose from larger tool registries. We introduce RFCD, a registry-scaled function-calling benchmark built from BFCL, Glaive Function Calling v2, and Tool-Call-Data, with standardized JSON-schema registries ranging from 5 to 3,000 functions. We also introduce TAAG, a deterministic stage-decomposed evaluator for registry conformance, structural completeness, and argument grounding. Across six locally served sub-4B models, three data sources, and four feedback modes, registry scale is the dominant source of degradation: single-pass TAAG pass rate drops sharply from small to large registries on all sources. Repea
    
[^53]: AI 无法审查出人类才能发现的问题：智能体指令文件代码评审的首次研究

    AI Cannot Review What Humans Catch: A First Look at the Reviews of Agent Instruction Files

    [https://arxiv.org/abs/2611.18582](https://arxiv.org/abs/2611.18582)

    该论文首次研究了智能体指令文件的评审现状，通过分析 2,246 个仓库中的 2,870 个拉取请求发现，超过 70% 的指令文件变更缺乏实质性评审，而 AI 评审者虽能发现表面结构问题，却对事实错误、必要性与内容过时等最需要人类判断的问题是结构性失明的。

    

    智能体日益塑造着软件，而指令文件则在引导它们。诸如 AGENTS.md、CLAUDE.md、.cursorrules 和 SKILL.md 之类的文件扮演着“可执行策略”的角色：它们是机器读取并据以行动的文本。与普通文档不同，对这类文件的一次修改就可能悄无声息地改变智能体在整个代码仓库中的行为方式。自然语言规则没有针对每次变更的验证标准，因此它们能绕过守护代码的编译器、类型检查器和测试套件。因此，评审成为唯一的质量关卡，但此前尚无研究考察这种评审是如何进行的。我们分析了来自 2,246 个代码仓库的 2,870 个指令文件拉取请求。这道关卡在很大程度上已经失效：超过 70% 的指令文件变更没有得到实质性评审。当 AI 评审者替代人类评审者时，它对最需要人类判断的问题是结构性失明的：它能发现表面结构问题，却会遗漏事实错误、必要性问题以及内容过时问题。

    arXiv:2611.18582v1 Announce Type: new  Abstract: Agents increasingly shape software, and instruction files steer them. Files such as AGENTS.md, CLAUDE.md, .cursorrules, and SKILL.md act as executable policy: prose that a machine reads and acts on. Unlike ordinary documentation, a single edit to one of these files can silently redirect how an agent behaves across an entire repository. Natural-language rules have no per-change oracle, so they slip past the compilers, type-checkers, and test suites that guard code. Review is therefore the only quality gate, but no prior work has studied how that review happens. We analyzed 2,870 instruction-file pull requests across 2,246 repositories. The gate is largely broken: over 70% of instruction-file changes receive no substantive review. When an AI reviewer substitutes for a human, it is structurally blind to the concerns that most need human judgment. It catches surface structure but misses factual errors, necessity questions, and staleness prob
    
[^54]: 安全缺陷报告识别的自动化技术：一项系统性映射研究

    Automatic techniques for security bug report identification: A systematic mapping study

    [https://arxiv.org/abs/2611.18581](https://arxiv.org/abs/2611.18581)

    本研究通过系统性映射综述了28项关于自动识别安全缺陷报告的研究，发现所用技术涵盖传统机器学习、深度学习和大语言模型，主要挑战在于标注数据缺乏和数据不平衡，且现有研究缺少真实工业环境中的评估。

    

    多项研究已探讨了使用自动化技术来帮助从业者在缺陷管理过程的早期识别安全缺陷报告。然而，目前该研究领域尚缺乏全面的综述。这样的综述将有助于确定未来的研究方向，并整合现有的潜在相关解决方案。本研究旨在为识别安全缺陷报告的自动化技术提供全面的概述。为此，我们开展了一项系统性映射研究，并识别出28项原始研究。研究结果表明，文献中已评估了多种用于识别安全缺陷报告的自动化技术，包括传统机器学习、深度学习以及更先进的大语言模型。研究识别出的两大主要挑战是标注数据的缺乏和数据不平衡。此外，我们观察到现有研究缺乏在真实工业环境中的评估。

    arXiv:2611.18581v1 Announce Type: new  Abstract: Several studies have investigated the use of automated techniques to help practitioners identify security bug reports early in the defect management process. However, currently there is no comprehensive overview of this research area. Such an overview will help identify future research directions and consolidate the potentially relevant existing solutions. This study aims to provide a comprehensive overview of automated techniques for identifying security bug reports. To achieve this, we conducted a systematic mapping study and identified 28 primary studies.   The study's findings show that the literature has evaluated various automatic techniques for identifying security bug reports, including traditional machine learning, deep learning, and more advanced large language models. The two main challenges identified are the lack of labeled data and data imbalance. Furthermore, we observe that existing studies lack evaluations in real indust
    
[^55]: 面向DevOps的模型驱动方法：一项系统性文献综述

    Model-Driven Approaches for DevOps: A Systematic Literature Review

    [https://arxiv.org/abs/2611.18580](https://arxiv.org/abs/2611.18580)

    本文是首个关于利用模型驱动工程（MDE）方法支持DevOps开发的系统性文献综述，通过对相关研究按计算领域、DevOps流程阶段和再工程步骤进行分类，发现当前研究主要聚焦于云计算领域。

    

    DevOps整合了旨在优化软件开发、构建、部署和监控的工具与方法，通过自动化并缩短开发生命周期来提升软件质量。然而，尽管DevOps具有诸多优势，它也带来了显著的挑战，尤其是在可用性和迁移方面。另一方面，模型驱动工程（MDE）提供了一种从特定技术和实现细节中进行抽象的方法，正如其在软件开发中的作用一样，这种方法可以被用来改善上述问题。本文提出了首个系统性文献综述，旨在理解当前支持DevOps开发的MDE方法。我们侧重于根据计算领域、这些工作所贡献的DevOps流程阶段以及它们所提出的再工程步骤，对相关工作进行分类。我们的研究结果突出了对云计算的主导性关注，同时一些研究提出了面向特定领域的……（原文摘要在此处不完整）

    arXiv:2611.18580v1 Announce Type: new  Abstract: DevOps integrates tools and methodologies aimed at optimizing software development, building, deployment, and monitoring by automating and reducing the development lifecycle and improving software quality. However, despite its many advantages, it presents notable challenges, particularly in terms of usability and migration. On the other hand, model-driven engineering (MDE) offers a way to abstract from specific technologies and implementation details, which can be leveraged to improve these aspects, as happens in software development.   In this paper, we present the first systematic literature review aiming to understand the current MDE approaches for supporting DevOps development. We focus on categorizing the works based on computing domains, the phases of the DevOps process to which they contribute, and the reengineering steps they propose.   Our findings highlight a dominant focus on cloud computing, while some studies propose domain-
    
[^56]: 信息物理系统中的不确定性：文献综述

    Uncertainty in Cyber-Physical Systems: A Literature Survey

    [https://arxiv.org/abs/2611.18579](https://arxiv.org/abs/2611.18579)

    该论文通过对186篇原始研究的文献综述，系统刻画了信息物理系统中不确定性的类型、来源、生命周期阶段及缓解策略，发现不确定性最常被视作保障问题，其中安全与规约不确定性最为普遍。

    

    信息物理系统（CPS）运行于动态且需保障可靠性的环境中，其感知、通信、决策、控制和保障等环节均受到不确定性的影响。本文通过一项涵盖186篇原始研究的文献综述来刻画信息物理系统中的不确定性。我们（i）刻画不确定性的类型，（ii）识别不确定性的来源，（iii）将不确定性与它们出现的生命周期阶段相关联，（iv）概述了可能的不确定性缓解策略。研究结果表明，不确定性最常被概念化为一个保障问题，其中安全与规约不确定性是最普遍的类别，其次是环境不确定性和模型/认知不确定性。领域映射显示跨领域参与广泛，而自动驾驶系统、能源系统和机器人等信息物理系统等安全关键领域则呈现出特定领域的不确定性模式。生命周期分析表明，不确定性并不仅限于（原文在此处截断）……

    arXiv:2611.18579v1 Announce Type: new  Abstract: Cyber-Physical Systems (CPSs) operate in dynamic, dependable environments where sensing, communication, decision-making, control, and assurance are affected by uncertainty. This paper characterizes CPS uncertainty through a literature survey comprising 186 primary studies. We (i) characterize uncertainty types, (ii) identify uncertainty sources, (iii) relate uncertainty to life-cycle phases in which they occur, and (iv) overview the possible uncertain mitigation strategies. The findings show that uncertainty is most often conceptualized as an assurance issue. Safety and specification uncertainty is the most prevalent category, followed by environmental and model/epistemic uncertainty. Domain mapping reveals broad cross-domain engagement, while safety-critical sectors such as autonomous systems, energy systems, and robotics CPSs display domain-specific uncertainty patterns. The life-cycle analysis indicates that uncertainty is not limited
    
[^57]: 软件工程中基于大语言模型的形式化建模：一项系统性文献综述

    Formal Modeling in Software Engineering with LLMs: A Systematic Literature Review

    [https://arxiv.org/abs/2611.18577](https://arxiv.org/abs/2611.18577)

    本文通过系统性文献综述，梳理了大语言模型在软件工程中自动化模型检测与形式化/半形式化模型生成方面的研究进展，从1,193篇候选文献中筛选并分析了34篇发表于2022至2026年间的同行评审论文。

    

    形式化验证能够为软件系统提供强大的正确性保证，但构建和维护形式化模型所需的大量专业知识与人工工作仍然是其发展的障碍。近期大语言模型（LLMs）的兴起引发了人们日益浓厚的兴趣：能否利用其自然语言理解与生成能力来自动化劳动密集型的建模活动。本文开展了一项系统性文献综述（SLR），考察大语言模型如何被应用于自动化模型检测以及软件工程中形式化和半形式化模型的生成。该综述遵循既定的SLR方法学，并采用基于PICOC的检索策略在四大主要数字图书馆中进行检索。从最初检索到的1,193篇文章中，最终筛选出34篇发表于2022年至2026年4月之间的同行评审会议、期刊和研讨会文章，并使用结构化数据提取表对其进行了分析。

    arXiv:2611.18577v1 Announce Type: new  Abstract: Formal verification provides strong correctness guarantees for software systems but remains hindered by the substantial expertise and manual effort required to construct and maintain formal models. The recent rise of Large Language Models (LLMs) has prompted growing interest in whether their natural language understanding and generation capabilities can be leveraged to automate labor-intensive modeling activities. This paper presents a Systematic Literature Review (SLR) examining how LLMs have been applied to automate model checking and the generation of formal and semi-formal models within software engineering. The review follows established SLR methodologies and a PICOC-driven search strategy across four major digital libraries. From an initial search of 1,193 articles, 34 peer-reviewed conference, journal, and workshop articles published between 2022 and April 2026 were selected and analyzed using a structured data extraction form. Th
    
[^58]: 测量与优化ICT能效：关于数据中心、云计算与LLM驱动代码优化的多声部文献综述

    Measuring and Optimizing Energy Efficiency in ICT: A Multi-vocal Literature Review on Data Centers, Cloud Computing, and LLM-Driven Code Optimization

    [https://arxiv.org/abs/2611.18576](https://arxiv.org/abs/2611.18576)

    该论文通过整合学术与灰色文献的多声部综述，提出了ICT（尤其是数据中心与云计算环境）能耗测量与优化的分类体系和实践图谱，并综述了包括LLM驱动代码优化在内的软件层面能效优化策略。

    

    arXiv:2611.18576v1 Announce Type: new 摘要：信息与通信技术（ICT）的快速增长带来了电力消耗的增加和环境影响的加剧。应对这一挑战需要以软件为中心的系统化视角，来理解能源及相关可持续性属性是如何被测量、报告和改进的。我们提出了一项综合性的多声部文献综述（MLR），整合了同行评审文献与灰色文献，以梳理ICT软件能耗测量与优化的当前实践，重点关注数据中心和云计算环境（DCCCEs）。首先，我们综合归纳了可持续性在整个ICT领域中的测量方式，提出了一个按度量领域（功率/能耗、能效、环境影响）和组件粒度（从应用程序、容器/虚拟机，到系统之系统层面）组织的分类体系，并总结了测量方法及其对应的工具生态。其次，我们在软件层面综述了优化策略……

    arXiv:2611.18576v1 Announce Type: new  Abstract: Rapid growth in Information and Communication Technology (ICT) increases electricity use and environmental impact. Addressing this challenge requires a systematic, software-centric view of how energy and related sustainability attributes are measured, reported, and improved. We present a comprehensive Multi-vocal Literature Review (MLR) integrating peer-reviewed and gray literature to map current practice in measuring and optimizing energy use in ICT software, with emphasis on data centers and cloud computing environments (DCCCEs). First, we synthesize how sustainability is measured across ICT by proposing a taxonomy organized along metric domains (power/energy, energy efficiency, environmental impact) and component granularity (from applications, to containers/VMs, and system-of-systems levels). We summarize measurement approaches and the tooling landscape that implements them. Second, we review optimization strategies at the software a
    
[^59]: 软件工程中图像到代码生成技术综述

    A Survey of Image-to-Code Generation for Software Engineering

    [https://arxiv.org/abs/2611.18575](https://arxiv.org/abs/2611.18575)

    本文是一篇综述，系统梳理了软件工程中图像到代码生成的研究文献，涵盖从界面截图、UML 图和流程图等视觉输入生成前端代码、程序修复与测试代码等方向的方法。

    

    在软件实现之前，很大一部分软件意图是以视觉形式表达的：设计师将截图或 Figma 原型图交给前端工程师，架构师绘制 UML 图或流程图，而错误报告中通常会附上问题发生时的图片。因此，将这些视觉产物直接转化为可执行的软件，已成为计算机视觉与软件工程交叉领域中一个活跃的研究问题。本综述系统梳理了软件工程中图像到代码生成（image-to-code generation）的相关文献，该领域被定义为一类输入包含视觉模态、输出为工程师原本需要手工编写的软件的方法。本综述有意将范围限定于软件工程代码，即用户界面与前端代码、从 UML 和流程图等软件图表中恢复的代码、由视觉信号驱动的程序修复与测试代码，以及一般性软件代码目标，并排除（摘要在此处截断）

    arXiv:2611.18575v1 Announce Type: new  Abstract: A large fraction of software intent is expressed visually before it is implemented. A designer hands a screenshot or a Figma mockup to a front-end engineer, an architect sketches a UML diagram or a flowchart, and a bug report ships with a picture of what went wrong. Turning such visual artifacts directly into executable software has therefore become an active problem at the intersection of computer vision and software engineering. The present survey organizes the literature on image-to-code generation for software engineering, defined as the family of methods whose input contains a visual modality and whose output is software an engineer would otherwise write by hand. We deliberately scope the survey to software-engineering code, namely user-interface and front-end code, code recovered from software diagrams such as UML and flowcharts, program repair and test code driven by visual signals, and general software-code targets, and we exclud
    
[^60]: 无需历史失败数据的预测性测试优化

    Predictive Test Optimization Without Historical Failure Data

    [https://arxiv.org/abs/2611.18574](https://arxiv.org/abs/2611.18574)

    该论文提出了一种新颖的协同演化标注方法，利用版本历史中测试与代码共同变化的信息来训练预测性测试优化模型，无需依赖历史失败数据，其故障检测能力几乎可与基于失败数据的模型相媲美。

    

    随着软件系统在规模和复杂性上的增长，其测试套件的执行成本变得越来越高，导致反馈延迟和资源浪费。最先进的预测性测试优化方法使用机器学习来选择与变更相关的测试，这些方法通常基于历史失败数据进行训练。然而，这依赖于失败历史，并且容易受到来自不稳定测试的标签噪声影响。我们引入了一种新颖的协同演化标注方法用于预测性测试优化，通过版本历史中测试与代码共同变化的情况来推导测试相关性。这些数据广泛可得且噪声更少。我们在三个大型项目上训练并评估了分别使用失败标签和协同演化标签的模型——项目涵盖工业界和开源领域，每个项目均包含数千个测试用例和超过一百万行代码。协同演化标注几乎达到了基于失败标签模型的故障检测能力，其总体平均故障检测百分比为0.88。

    arXiv:2611.18574v1 Announce Type: new  Abstract: As software systems grow in size and complexity, their test suites become increasingly expensive to execute, delaying feedback and wasting resources. State-of-the-art predictive test optimization methods use machine learning to select tests relevant to a change, typically trained on historical failures. This depends on failure histories and is susceptible to label noise from flaky tests. We introduce the novel method of co-evolution labeling for predictive test optimization, deriving test relevance from tests and code changing together in the version history. These data are widely available and less noisy. We train and evaluate models with failure and co-evolution labels on three large projects---from industry and open source; each with thousands of tests and more than 1M LOC. Co-evolution labeling nearly matches the failure detection capabilities of failure-based models, with an overall Average Percentage of Faults Detected of 0.88, whi
    
[^61]: LoopSkill：编码智能体中的证据约束生命周期控制与安全性-活性权衡

    LoopSkill: Evidence-Bound Lifecycle Control and the Safety-Liveness Tradeoff in Coding Agents

    [https://arxiv.org/abs/2611.18573](https://arxiv.org/abs/2611.18573)

    LoopSkill 提出了一个面向编码智能体的证据约束生命周期控制平面（集成唯一写入者、持久化发件箱、工件绑定报告、独立评审与心跳门控终局等机制），并在 SWE-bench Lite 上的探索性实验中揭示了“安全性-活性”权衡——更严格的证据绑定控制反而降低了任务完成成功率。

    

    编码智能体可以编辑代码仓库并宣称任务已完成，但对话中的声明并不能证明工件、调度分发、状态版本与权限仍然保持一致。LoopSkill 是一个证据约束的生命周期控制平面，包含经模式验证的唯一写入者、持久化发件箱路由、工件绑定的报告、恢复机制、独立评审以及由心跳门控的最终定稿。我们在统一的 1,800 秒预算下，于 SWE-bench Lite 的一个选定子集上评估了本地冻结的 v3.3.12 集成。原始的 140 个单元格实验计划在中期结果可见后因管理决策被截断，因此 60 个单元格的分析属于探索性、非确证性研究。在 58 个计分单元格中，盲评工件成功率为：LoopSkill 7/19，“实现+独立评审”方案 11/19，普通编码智能体 14/20；另有 2 个单元格因实验装置问题记为无效。在 18 个完整的 LoopSkill 与评审对照配对中，描述性风险差异为 -0.222，精确 McNemar（检验结果）……（原文摘要在此处截断）

    arXiv:2611.18573v1 Announce Type: new  Abstract: Coding agents can edit repositories and announce completion, but conversational claims do not prove that artifacts, dispatches, state versions, and authority still agree. LoopSkill is an evidence-bound lifecycle control plane with a schema-validated sole writer, durable outbox routing, artifact-bound reports, recovery, independent review, and heartbeat-gated finalization. We evaluate a locally frozen v3.3.12 integration under a common 1,800-second budget on a selected subset of SWE-bench Lite. The original 140-cell plan was administratively truncated after interim outcomes were visible, so the 60-cell analysis is exploratory and nonconfirmatory. Of 58 scored cells, blind artifact success was 7/19 for LoopSkill, 11/19 for implementation plus independent review, and 14/20 for an ordinary coding agent; two cells were apparatus null. Across 18 complete LoopSkill-versus-review pairs, the descriptive risk difference was -0.222 with exact McNem
    
[^62]: 在AI使用常态化的环境中，作者资历而非AI披露会使代码评审产生偏差

    Where AI Use Is Normalized, Author Seniority but Not AI Disclosure Biases Code Review

    [https://arxiv.org/abs/2611.18572](https://arxiv.org/abs/2611.18572)

    在AI使用常态化的组织中，实验表明披露AI使用并不会使代码评审产生偏差，而作者的资历标签则会显著影响评审者对代码有效性和作者能力的判断。

    

    代码评审工具日益显示作者在准备拉取请求时是否使用了AI。近期的研究报告指出，AI披露会使评审者对原本相同的工作给出更低的能力评价，且这种惩罚对已经处于边缘地位的作者影响最大。在一项于AI使用常态化组织中开展的被试内实验中，447名软件工程师分别在提交信息中嵌入AI使用披露和作者资历标签的条件下评审了相同的四个代码片段。结果显示，AI披露并未使对代码有效性或作者能力的感知产生偏差，而资历标签则对两者均产生了显著影响。这些发现共同表明，在该环境下未检测到对披露AI使用的惩罚，同时证实了代码评审中显示的作者信息仍然可能使评估产生偏差。

    arXiv:2611.18572v1 Announce Type: new  Abstract: Code-review tools increasingly display whether an author used AI when preparing pull requests. Recent studies report that AI disclosure makes reviewers rate otherwise identical work as less competent, with the penalty falling hardest on already-marginalized authors. In a within-subjects experiment in an AI-normalized organization, 447 software engineers each reviewed the same four code snippets under an AI-use disclosure embedded in the commit message and author-seniority labels. AI disclosure did not bias perceptions of code effectiveness or author competence, whereas a seniority label significantly biased both. Together, these findings show no detected penalty for disclosed AI use in this setting, while confirming that the author information shown in code review can still bias evaluations.
    
[^63]: 循环工程：面向生产级编码代理的版本绑定、证据门控协调

    Loop Engineering: Version-Bound, Evidence-Gated Coordination for Production Coding Agents

    [https://arxiv.org/abs/2611.18571](https://arxiv.org/abs/2611.18571)

    该论文提出“循环工程”方法，通过触发器、上下文投影、权限边界、验证器、停止条件和回写规则六要素规范异步代理循环，并借助版本化技能契约、证据收集框架与版本绑定的人工审批门控，在不依赖私有对话交接的情况下，实现生产级编码代理在共享可审计仓库状态上的协调运行。

    

    编码代理能够编辑代码仓库并执行测试，但生产级软件工作远不止代码生成。它需要持久的任务上下文、范围受限的权限、版本感知的审批、与来源无关的验证、中断后的恢复，以及人类与自动化工作者之间可问责的交接。我们提出 Loop Engineering（循环工程），这是一种在共享、可审计的仓库状态上运行多个异步代理循环的方法，无需依赖私有的对话式交接。每个循环由六个要素定义：触发器、上下文投影、权限边界、验证器、停止条件和回写规则。被称为 Skills（技能）的版本化程序性契约编码了可复用的操作策略；Harness（执行框架）限制可用的操作面并收集证据；人工门控则针对确切的制品版本对高后果的状态转换进行授权。我们报告了一项来自 MewDesign（一家产生收入的……）的设计与经验研究（摘要在此处截断）。

    arXiv:2611.18571v1 Announce Type: new  Abstract: Coding agents can edit repositories and execute tests, yet production software work extends far beyond code generation. It requires durable task context, scoped authority, version-aware approval, source-independent verification, interruption recovery, and accountable handoffs between humans and automated workers. We present Loop Engineering, a method for operating multiple asynchronous agent loops over shared, auditable repository state without relying on private conversational handoffs. Each loop is specified by six elements: a trigger, a context projection, an authority boundary, a verifier, a stop condition, and a write-back rule. Versioned procedural contracts, called Skills, encode reusable operating policy; a Harness bounds the available action surface and collects evidence; human gates authorize high-consequence transitions against exact artifact versions.   We report a design-and-experience study from MewDesign, a revenue-generat
    
[^64]: 迈向可持续的大语言模型推理：一项系统性文献综述

    Towards Sustainable Large Language Model Inference: A Systematic Literature Review

    [https://arxiv.org/abs/2611.18570](https://arxiv.org/abs/2611.18570)

    本文通过系统性文献综述，梳理了从输入、模型、系统到硬件各层面降低大语言模型推理能耗的技术，分析了能耗、延迟与准确性之间的权衡关系，并总结了用于能耗测量的工具、数据集与指标。

    

    大语言模型（LLM）推理正日益成为现代AI部署中运营成本与环境影响的重要来源。推理过程中的能耗取决于推理流程中的多个因素，包括输入与输出长度、模型设计、系统级决策以及硬件特性。本文对节能且可持续的LLM推理进行了系统性文献综述（SLR）。我们通过分析所选取的原始研究，识别了在推理流程的不同层面（包括输入、模型、系统和硬件）提出的降低能耗的各项技术。我们进一步考察了能耗、延迟与准确性之间的权衡，并指出当前对这些关系的评估方式存在不一致之处。此外，我们综述了用于能耗测量的工具、数据集和指标，涵盖基于软件和基于硬件两类方法。（注：原文摘要在此处被截断）

    arXiv:2611.18570v1 Announce Type: new  Abstract: Large language model (LLM) inference is becoming a significant contributor to both operational cost and environmental impact in modern AI deployments. Energy consumption during inference depends on multiple factors across the pipeline, including input and output length, model design, system-level decisions, and hardware characteristics. This paper presents a systematic literature review (SLR) of energy-efficient and sustainable LLM inference. We analyze selected primary studies to identify techniques proposed to reduce energy consumption across different levels of the inference pipeline, including input, model, system, and hardware. We further examine the trade-offs between energy consumption, latency, and accuracy, highlighting inconsistencies in how these relationships are evaluated. In addition, we review the tools, datasets, and metrics used for energy measurement, covering both software-based and hardware-based approaches. This surv
    
[^65]: 以知识为中心的自动化问题解决：系统性综述与分类体系

    Knowledge-Centric Automated Issue Resolution: A Systematic Survey and Taxonomy

    [https://arxiv.org/abs/2611.18569](https://arxiv.org/abs/2611.18569)

    本文对133项自动化问题解决研究进行了以知识为中心的系统性综述，提出了由背景知识层、仓库知识层和过程知识层构成的三层知识分类体系，并从六个维度分析了知识生命周期，发现当前系统主要依赖现有代码知识和开发工具知识。

    

    自动化问题解决是一项知识密集型的软件维护任务，基于大语言模型（LLM）的智能体必须从代码仓库、执行环境、历史产物、文档、工具以及先前修复过程中获取、表示、使用和更新知识。本文从以知识为中心的视角对133项关于自动化问题解决的研究进行了系统性综述。我们提出了一个三层分类体系，将问题解决所需知识组织为背景知识层、仓库知识层和过程知识层，并从知识类型、来源与提取方法、表示格式、使用方法、工作流阶段以及时间演化六个维度分析了知识生命周期。我们的分析表明，当前系统主要依赖现有代码知识和开发工具知识，而对仓库演化知识和经验性知识的关注正在不断上升。

    arXiv:2611.18569v1 Announce Type: new  Abstract: Automated issue resolution is a knowledge-intensive software maintenance task in which LLM-based agents must acquire, represent, use, and update knowledge from repositories, execution environments, historical artifacts, documents, tools, and prior repair processes. This paper presents a systematic survey of 133 studies on automated issue resolution from a knowledge-centric perspective. We introduce a three-layer taxonomy that organizes issue-resolution knowledge into the Background Knowledge Layer, the Repository Knowledge Layer, and the Procedural Knowledge Layer, and analyze the knowledge lifecycle along six dimensions: knowledge types, sources and extraction methods, representation formats, usage methods, workflow stages, and temporal evolution. Our analysis shows that current systems primarily rely on Existing Code Knowledge and Development Tool Knowledge, while Repository Evolution Knowledge and Experiential Knowledge are increasing
    
[^66]: 使用同行代码评审识别开源软件中的潜在安全漏洞：一项系统性文献综述

    Use of Peer Code Review to Identify Potential Security Vulnerabilities in Open Source Software: A Systematic Literature Review

    [https://arxiv.org/abs/2611.18568](https://arxiv.org/abs/2611.18568)

    本系统性文献综述发现，同行代码评审能有效检测开源软件中局部且知名的漏洞（如缓冲区溢出、竞态条件和XSS），但常常遗漏微妙、复杂且依赖上下文的安全问题。

    

    背景：开源软件（OSS）中的安全漏洞可能造成严重后果。OSS项目参与者广泛使用同行代码评审来提高质量并防止安全缺陷。然而，尽管存在强制性评审，OSS项目仍然包含漏洞，这凸显了评估代码评审在安全方面有效性的必要性。目标：本系统性文献综述（SLR）综合了关于已被识别或被遗漏的漏洞类型、安全实践、工具支持、评审者面临的挑战以及能够提高漏洞检测能力的因素等方面的证据。方法：对五个数字文献库的检索共获得1,067项研究，其中20项被纳入分析。漏洞采用通用缺陷列举（CWE）框架进行分类。结果：同行代码评审能够有效检测局部的、众所周知的漏洞（例如缓冲区溢出、竞态条件、崩溃、跨站脚本攻击XSS），但经常遗漏微妙的、复杂的、依赖上下文的问题（例如输入验证缺陷、访问控制缺陷、资源……）。

    arXiv:2611.18568v1 Announce Type: new  Abstract: Context: Security vulnerabilities in OSS can have severe consequences. OSS project participants widely use peer code review to improve quality and prevent security flaws. However, despite mandatory reviews, OSS projects still contain vulnerabilities, highlighting the need to evaluate code review effectiveness for security. Objectives: This SLR synthesizes evidence on vulnerability types identified or missed, security practices, tool support, reviewer challenges, and factors that improve detection. Method: Searches of five digital libraries yielded 1,067 studies, of which 20 were included. Vulnerabilities were categorized using the Common Weakness Enumeration (CWE) framework. Results: Peer code reviews effectively detect localized, well-known vulnerabilities (e.g., buffer overruns, race conditions, crashes, XSS) but often miss subtle, complex, context-dependent issues (e.g., input validation weaknesses, access control weaknesses, resource
    
[^67]: 虚拟现实/增强现实软件生命周期中的软件工程实践：一项系统性综述

    Software Engineering Practice in the Virtual Reality/Augmented Reality Software Life Cycle: A Systematic Survey

    [https://arxiv.org/abs/2611.18567](https://arxiv.org/abs/2611.18567)

    该论文首次从软件工程视角对VR/AR软件生命周期相关研究进行系统性综述，填补了该领域缺乏全面总结的空白。

    

    虚拟现实（VR）利用多种虚拟现实技术，为用户提供计算机生成的三维虚拟环境中的沉浸式体验；相比之下，增强现实（AR）则将虚拟物体（如图像、文本、动画等）叠加到用户的真实场景中。得益于计算机图形学、计算机视觉和人工智能技术的成熟，VR和AR受到越来越多的关注，并在硬件和软件方面均实现了快速的市场增长。在早期阶段，VR/AR相关研究主要集中于硬件（例如头戴式显示器和配备传感器的控制器）。近年来，随着VR/AR生态系统的成熟，VR/AR软件已成为热门研究话题。然而，目前尚缺乏对VR/AR软件工程实践的系统性总结工作。为填补这一空白，本文旨在从软件（工程）的视角，对VR/AR领域的最新研究现状进行系统性综述。

    arXiv:2611.18567v1 Announce Type: new  Abstract: Leveraging diverse virtual reality (VR) technologies, VR provides users with an immersive experience in a computer-generated 3D virtual environment. In contrast, augmented reality (AR) superimposes virtual objects (such as images, text, animations, etc.) onto users' real scenes. Both VR and AR have received increasing attention due to the maturity of computer graphics, computer vision, and artificial intelligence. VR and AR have gained rapid market growth in both hardware and software. In the early stages, VR/AR-related research mainly focused on hardware (e.g., head-mounted displays and sensor-equipped controllers). Recently, VR/AR software has become a hot topic due to the mature VR/AR ecosystem. However, there is no comprehensive work to summarize software engineering practices in VR/AR.   To bridge this gap, this paper aims to provide a systematic survey summarizing the state-of-the-art research on VR/AR from the perspective of softw
    
[^68]: 在真实环境中挖掘架构模型：一个精选的AADL模型数据集

    Mining Architectural Models in the Wild: A Curated Dataset of AADL Models

    [https://arxiv.org/abs/2611.18566](https://arxiv.org/abs/2611.18566)

    该论文提出了A3D数据集——一个从GitHub挖掘、经OSATE验证、覆盖7个领域和52个问题集群、并具有领域/问题标签与参考架构组件映射双标注层的1,115个AADL模型精选数据集，填补了软件架构实证研究中大规模标注数据集的空白。

    

    软件架构维护与演化的实证研究一直受到缺乏大规模、经过验证且具有丰富标注的模型数据集的阻碍。现有的基准数据集规模有限、仅提供模型级别的标注，或缺乏与参考架构的对齐，从而限制了架构复用、一致性检查以及AI驱动分析的可复现研究。为填补这一空白，我们提出了A3D，一个包含1,115个架构分析与设计语言（AADL）模型的精选数据集，这些模型从公开的GitHub仓库中挖掘，并通过OSATE工具进行验证。该数据集覆盖7个通用领域和52个特定问题集群，并提供两个标注层：（i）领域和问题级别的标签，捕捉不同的模型聚合标准；（ii）到三个参考架构的组件级别映射，这些映射已通过用户研究进行外部验证。所有模型以AADL和JSON格式发布。

    arXiv:2611.18566v1 Announce Type: new  Abstract: Empirical research on software architecture maintenance and evolution is hindered by the scarcity of large-scale, validated, and richly annotated model datasets. Existing benchmarks are limited in scale, provide only model-level annotations, or lack alignment with reference architectures, thereby restricting reproducible studies of architectural reuse, conformance, and AI-driven analysis. To address this gap, we present A3D, a curated dataset of 1,115 Architecture Analysis and Design Language (AADL) models mined from public GitHub repositories and validated with the OSATE tool. The dataset spans 7 generic domains and 52 specific-problem clusters, and provides two annotation layers: (i) domain- and problem-level labels, which capture different model aggregation criteria, and (ii) component-level mappings to three reference architectures, which have been externally validated through a user study. Models are released in AADL and JSON format
    
[^69]: ProofCoder：面向自动化软件验证的连续推理

    ProofCoder: Continuous Reasoning for Automated Software Verification

    [https://arxiv.org/abs/2611.14106](https://arxiv.org/abs/2611.14106)

    ProofCoder从编码智能体视角出发，让模型在整个证明搜索过程中持续推理，并根据证明助手的反馈自主决定信息收集、证明状态检查、片段修改和回溯，突破了传统LLM方法僵化策略生成循环的限制，实现了面向Rocq的自动化证明合成。

    

    使用证明助手进行形式化验证能够为软件提供强有力的正确性保证，但构造证明仍然是一项劳动密集型工作，且需要大量的专业知识。近期基于大语言模型（LLM）的方法在自动化证明合成方面展现出了潜力。然而，这些方法通常将模型限制在僵化的策略生成循环中，由外围系统控制前提检索、证明状态管理和回溯。我们提出了ProofCoder，这是一个面向Rocq的智能体式证明合成系统，它从编码智能体的视角来处理定理证明问题。ProofCoder并非在固定循环中生成策略，而是在整个证明搜索过程中持续进行推理，并利用证明助手的反馈来决定何时收集额外信息、检查证明状态、修改先前生成的证明片段，或回溯到证明中较早的位置。这使其能够迭代地协同……

    arXiv:2611.14106v1 Announce Type: cross  Abstract: Formal verification using proof assistants provides strong correctness guarantees for software, but constructing proofs remains labor-intensive and requires significant expertise. Recent approaches based on large language models (LLMs) have shown promise in automating proof synthesis. However, these approaches typically constrain the model to a rigid tactic-generation loop, in which the surrounding system controls premise retrieval, proof-state management, and backtracking. We introduce ProofCoder, an agentic proof-synthesis system for Rocq that approaches theorem proving from a coding-agent perspective. Rather than generating tactics within a fixed loop, ProofCoder reasons continuously throughout proof search and uses proof-assistant feedback to decide when to gather additional information, inspect proof states, revise previously generated proof fragments, or backtrack to an earlier point in the proof. This allows it to iteratively co
    
[^70]: LLM智能体通过共享内存实现的无提示协调：一项加锁双重实现的复现与扩展研究

    Unprompted coordination through shared memory in LLM agents: a locked two-implementation replication and extension

    [https://arxiv.org/abs/2611.12412](https://arxiv.org/abs/2611.12412)

    本研究通过双重独立实现的复现与扩展首次证实：即使提示中完全没有提及其他智能体、共享或协作的存在，LLM智能体也能仅凭共享内存状态自发产生协调行为。

    

    大型语言模型智能体之间的协调已成为AI安全与软件工程的核心问题，因为这类智能体通常以共享基础设施的群体形式部署。现有研究通常会告知智能体伙伴的存在，或直接要求它们进行协调。而仅通过共享状态——在没有任何信息透露其他参与者存在的情况下——协调是否会自发产生，这对于在智能体之间共享缓存与内存的部署场景而言是一个关键问题，但该问题鲜有研究，据我们所知也从未经过独立复现；本研究两者兼顾。四个智能体各自收到一个简短的算术任务和一个键值存储，每个智能体都将该存储视为自己的暂存空间；其中两个智能体持有同一计算中互补的两半，且所有提示均未提及智能体、共享或协作。为保证可靠性，实验环境依据书面协议独立重建，两种实现均在哈希锁机制下运行。

    arXiv:2611.12412v2 Announce Type: replace-cross  Abstract: Coordination between large language-model agents has become a central question for AI safety and software engineering, as such agents are deployed in groups that share infrastructure. Existing research typically tells the agents that partners exist or asks them to coordinate. Whether coordination arises through shared state alone, with nothing disclosing that anyone else is present, is the key question for deployments that share caches and memory between agents, yet it has rarely been examined, and to our knowledge not with independent replication; this study does both. Four agents each received a short arithmetic task and a key-value store presented to each as its own scratch space; two held complementary halves of one computation, and no prompt mentioned other agents, sharing or collaboration. For reliability, the environment was rebuilt independently from a written protocol, and both implementations ran under hashed locks on
    
[^71]: 偶然的安全：对大语言模型生成代码的崩溃测试

    Safe by Accident: Crash-Testing LLM-Written Code

    [https://arxiv.org/abs/2611.10585](https://arxiv.org/abs/2611.10585)

    该论文通过对 9,100 个 LLM 生成的后端代码进行崩溃测试发现，54% 在 sqlite3 默认事务模式下能通过崩溃测试的数据库代码，在自动提交模式下会留下半更新状态，证明现有“单次运行、单一环境”的代码评估把通过归功于环境而非代码本身的安全性。

    

    arXiv:2611.10585v2（公告类型：replace）摘要：对大语言模型（LLM）所写代码的评估通常只运行一次、仅在一个环境中进行，并将通过测试归功于代码本身。我们证明这种“功劳”可能属于环境。我们在每次数据库、文件和邮件调用之前让 LLM 编写的后端代码崩溃，然后像运维人员那样重新运行，并在 Python sqlite3 的两种事务模式下评判结果：默认模式（库会静默地开启一个事务）和自动提交模式（autocommit，不会开启事务）。在对十个模型在 24 个后端任务上共 9,100 个解的测试中，在默认模式下经受住每一次崩溃的数据库代码，有超过一半（54%）在自动提交模式下会处于“半更新”（部分写入）状态。这一现象在每个模型上都成立，且隐藏在普通代码之中，包括那些看起来像事务但实际上并未开启事务的 “with conn:” 代码块。告知模型其连接处于自动提交模式并不会改变本地模型所写的代码，而 API 模型的不安全代码比例下降了 30 个百分点。要求……（原文摘要在此处被截断）

    arXiv:2611.10585v2 Announce Type: replace  Abstract: Evaluations of LLM-written code run it once, in one environment, and credit the code with whatever passes. We show that this credit can belong to the environment. We crashed LLM-written backend code just before each database, file and email call, re-ran it as an operator would, and judged the result under two transaction modes of Python's sqlite3: the default, in which the library quietly opens a transaction, and autocommit, in which it does not. Across 9,100 solutions from ten models on 24 backend tasks, more than half (54%) of the database code that survives every crash in the default mode is left half-updated in autocommit mode. This holds for every model, and it hides in ordinary code, including with conn: blocks that look like transactions but open none. Telling the model that its connection is in autocommit mode does not change what the local code models write, while the API models' unsafe share falls by 30 points. Asking for c
    
[^72]: 提取重构无法精简什么，函数注入能否做到：一项针对580个Java代码库的预注册研究

    What extraction cannot slim, and whether function injection can: a pre-registered study of 580 Java repositories

    [https://arxiv.org/abs/2611.10545](https://arxiv.org/abs/2611.10545)

    该预注册研究对580个Java代码库中超过300万个方法进行扫描，发现22.9%的长方法无法通过单出口提取重构来精简，并验证OfficeFloor框架的函数注入技术能够拆分这类提取重构无法处理的长方法。

    

    超过26行代码的方法被视作“长方法”，而提取方法通常是其标准解决方案。然而某些长方法不存在单出口的重构方案：return语句无法从提取出的片段中返回，一个片段只能交回一个值，并且有些方法的代码纠缠在一起，以至于每一次切割都被阻塞。函数注入（如OfficeFloor框架所实现的）采用不同的切割方式：每个片段变成框架注入并编排的函数，提前返回会终止一个函数，而一个函数可以传递多个值。该框架的作者预先注册了一项测试，验证函数注入能否拆分提取重构所无法处理的长方法。研究对580个开源Java代码库（均在AI编程助手普及之前锁定）进行分层随机抽样，使用已发布的工具进行扫描，其成本模型在抽样之前即已固定：共扫描3,026,902个方法，其中168,466个为长方法。结果显示，单出口提取重构无法精简22.9%的长方法。预注册的假设是，至少四分之一……

    arXiv:2611.10545v2 Announce Type: replace  Abstract: A method of 26 code lines or more reads as a Long Method, and Extract Method is the usual fix. Some long methods have no single-exit fix: a return cannot leave an extracted piece, a piece hands back one value, and some methods are knotted so that every cut is blocked. Function injection, as the OfficeFloor framework does it, cuts differently: each piece becomes a function the framework injects and sequences, an early return ends a function, and a function hands on several values. The framework's author pre-registered a test of whether it takes apart what extraction cannot. A stratified random sample of 580 open-source Java repositories, each pinned before AI coding assistants were in general use, was scanned by released tools whose cost models were fixed before the sample was drawn: 3,026,902 methods, 168,466 of them long. Single-exit extraction cannot slim 22.9% of the long methods. The registered hypothesis, that at least a quarter
    
[^73]: BOXBOX：面向长期运行性能优化的递归式智能体框架自我改进

    BOXBOX: Recursive Harness Self-Improvement for Long-Running Performance Optimization

    [https://arxiv.org/abs/2611.09218](https://arxiv.org/abs/2611.09218)

    BOXBOX让智能体能够在长期运行的任务中递归地自我改进自己的执行框架（通过智能体触发的框架修订与内核管理的热重载机制），并在PerfRace基准上以匹配的测试预算下比冻结框架、离线框架演化和框架扩展方法取得更高的加速比和效率。

    

    自动化的智能体框架优化通过从执行轨迹和任务反馈中学习来改进智能体。现有的离线方法使用参考答案来优化框架，以便在保留任务上复用。然而，许多现实世界的任务是开放式的，而长时间运行会暴露出离线优化的框架可能无法应对的新困难。为了研究这些场景下的框架优化，我们提出了PerfRace，一个针对软件项目长期运行性能优化的基准，并研究智能体能否在正在进行的任务中利用任务特定的经验来递归地改进其框架。为此，我们提出了BOXBOX，它支持由智能体触发的框架修订和内核管理的热重载，同时保留执行上下文。我们在匹配的测试时预算下，将BOXBOX与冻结框架、离线框架演化和框架扩展方法在PerfRace上进行比较。BOXBOX以更高的效率实现了更高的加速。

    arXiv:2611.09218v1 Announce Type: new  Abstract: Automated harness optimization improves agents by learning from trajectories and task feedback. Existing offline methods use reference answers to optimize harnesses for reuse on held-out tasks. However, many real-world tasks are open-ended, while long runs expose new difficulties that offline-optimized harnesses may not address. To study harness optimization in these settings, we introduce PerfRace, a benchmark for long-running performance optimization of software projects, and investigate whether agents can use task-specific experience to recursively improve their harnesses during an ongoing task. To this end, we introduce BOXBOX, which enables agent-triggered harness revision and kernel-managed hot reloading while preserving execution context. We evaluate BOXBOX on PerfRace under matched test-time budgets, comparing it with Frozen Harness, offline Harness Evolution, and Harness Scaling. BOXBOX achieves higher speedup with more efficien
    
[^74]: 重新思考AI智能体的访问控制：最新进展与开放挑战

    Rethinking Access Control for AI Agents: Recent Advances and Open Challenges

    [https://arxiv.org/abs/2611.06764](https://arxiv.org/abs/2611.06764)

    本文系统综述了AI智能体访问控制在策略规范、身份委托、动作执行和信息流控制四大领域的最新进展，并重新思考了授权机制的四个核心方面，指出了未来亟待解决的开放性挑战。

    

    大型语言模型智能体可以访问数据、调用工具并代表用户执行操作。确保这些行为保持在授权范围内具有挑战性，因为智能体会自行选择执行路径，而权限和运行条件也可能随时变化。本文从四个领域综述了智能体访问控制的最新进展：策略规范与推导、身份与委托、动作与资源执行，以及信息流控制。我们比较了各机制的控制对象、执行边界和前提假设，以审视单个机制所能实现的功能及其协同工作的方式。基于这一分析，我们重新思考了授权的四个方面：计划选择决定了权限需求；上下文信息可以为决策提供依据，但其本身并不授予权限；限制必须在系统接口之间保持其含义；并且随着工作流的演变，必须对授权进行重新评估。这些见解引出了该领域的开放性挑战。

    arXiv:2611.06764v1 Announce Type: new  Abstract: Large language model agents access data, invoke tools, and act on users' behalf. Keeping these actions within authorized bounds is challenging because agents choose their own execution paths while permissions and operating conditions may change. This paper surveys recent advances in agent access control across four areas: policy specification and derivation, identity and delegation, action and resource enforcement, and information flow control. We compare their control objects, enforcement boundaries, and assumptions to examine what individual mechanisms achieve and how they can work together. Building on this analysis, we rethink four aspects of authorization: plan selection shapes permission requirements; contextual information can inform decisions without itself granting authority; restrictions must retain their meaning across system interfaces; and authorization must be reassessed as workflows evolve. These insights motivate open cha
    
[^75]: IssueLighter：一种感知问题类型的LLM智能体，用于问题报告有效性评估

    IssueLighter: An Issue Type-Aware LLM Agent for Issue Report Validity Assessment

    [https://arxiv.org/abs/2611.02721](https://arxiv.org/abs/2611.02721)

    该论文提出IssueLighter，一种感知问题类型的LLM智能体，通过全面理解各类问题的无效原因来评估开源项目问题报告的有效性，并提供有充分依据的理由，从而减轻维护者的负担。

    

    无效的问题报告在开源软件项目中普遍存在，给已经面临高度倦怠风险的项目维护者带来了沉重负担。以往的研究主要集中于识别特定问题类别（例如bug和wontfix）中的无效报告，而对开源软件更广泛的问题类型尚缺乏探索，这限制了现有方法的泛化能力。此外，现有方法在解释或验证其预测方面提供的支持很少，降低了它们对维护者的实际实用性。随着基于大语言模型（LLM）的智能体的兴起，对所有问题类型的无效原因进行全面理解能否在提升有效性评估性能的同时产生有充分依据的理由，目前仍不清楚。为此，我们首先人工标注了来自三个开源软件项目（即pandas、scikit-learn和mypy）的520份问题报告的有效性状态……

    arXiv:2611.02721v2 Announce Type: replace-cross  Abstract: Invalid issue reports are prevalent in open source software (OSS) projects, imposing substantial burdens on maintainers who already face a high risk of burnout. Prior research has primarily focused on identifying invalid reports for specific issue categories (e.g., bug and wontfix), leaving the broader landscape of OSS issue types largely unexplored and limiting the generalizability of existing approaches. Moreover, existing methods provide little support for explaining or validating their predictions, reducing their practical usefulness for maintainers. With the rise of LLM-based agents, it remains unclear whether a comprehensive understanding of the causes of invalidity across all issue types can improve validity assessment performance while producing well-grounded rationales. To this end, we first manually annotated the validity status of 520 issue reports from three OSS projects, i.e., pandas, scikit-learn, and mypy, result
    
[^76]: 审阅者所称的长方法只是一种长度：一项将耦合散度作为上帝单元信号的预注册测试，以及它最终实际证实的内容

    What reviewers call a Long Method is a length: A pre-registered test of coupling divergence as a god-unit signal, and what it settled instead

    [https://arxiv.org/abs/2611.01636](https://arxiv.org/abs/2611.01636)

    预注册测试表明，用耦合散度加权复杂度无法区分上帝单元与复杂算法；该研究真正证实的是，MLCQ审阅者所称的“长方法”仅仅是一个绝对行数长度（26行及以上的阈值在确认集上达到MCC 0.777）。

    

    一项配套研究发现，复杂度离群值规则能够显示复杂度集中在何处，但无法区分上帝单元与复杂算法。本文预先注册了一项测试，以验证区分两者的方法之一：用耦合散度（即单元传出的数据减去接收的数据）对单元复杂度加权，从而将叶子单元排除在外。耦合度量的获得方式是解析520个MLCQ仓库、500个全新Java仓库以及73个Qualitas Corpus系统中的每一次调用。该主张未能成立：主要结果——加权形式减去仅用复杂度后，针对MLCQ中人类标注Blob类的合并MCC——为-0.041 [-0.130, +0.054]，差异未获解决且统计功效不足，而已解决的读数反而与该主张相悖。数据所能确定的是方法层面的：MLCQ审阅者所称的长方法是一种绝对长度——在探索半部分选定的26行或以上的常数阈值，在确认半部分达到MCC 0.777，比最佳的每仓库……高出+0.080 [+0.016, +0.145]。

    arXiv:2611.01636v2 Announce Type: replace  Abstract: A companion study found that a complexity outlier rule shows where complexity concentrates but cannot tell a god unit from a complex algorithm. This paper pre-registered a test of one way to tell them apart: weight a unit's complexity by its coupling divergence, the data it hands on minus the data it receives, so that a leaf is set aside. Coupling was measured by resolving every call in 520 MLCQ repositories, 500 fresh Java repositories and 73 Qualitas Corpus systems. The claim fails: the primary, the pooled MCC of the weighted form minus complexity alone against MLCQ's human Blob, is -0.041 [-0.130, +0.054], no difference resolved and underpowered, and the resolved readings go against it. What the data settle is at method level. What MLCQ's reviewers call a Long Method is an absolute length: a constant of 26 lines or more, chosen on the explore half, reaches MCC 0.777 on the confirm half, +0.080 [+0.016, +0.145] over the best per-re
    
[^77]: 人工智能合成软件演化中的软件变更请求

    Software Change Request in AI-Synthesized Software Evolution

    [https://arxiv.org/abs/2611.00294](https://arxiv.org/abs/2611.00294)

    本综述系统梳理了利用规则、传统机器学习、主题建模及深度学习方法从应用商店用户评论中自动提取软件变更请求的研究现状，并指出不同研究在数据集、标注规范和评估指标上缺乏一致性，导致结果难以比较。

    

    移动应用商店是用户反馈的丰富来源，其中包含大量与软件部署后修改相关的变更请求、缺陷报告、改进建议等。然而，手工将这些自由格式的数据转化为可操作的变更请求（SCR）具有很大挑战。本综述介绍了应用基于规则的技术、传统机器学习方法、主题建模、混合方法以及少量深度学习方法的相关研究。所使用的数据集规模从小的标注集合到数百万条评论不等。然而，不同研究之间在名称、可用性、标注指南和抽样方法方面存在不一致性。许多研究使用词袋模型和TF-IDF，而朴素贝叶斯、支持向量机、决策树和随机森林等方法被频繁采用。精确率、召回率和F1分数是常用的评估指标，但各研究的结果和实验不易相互比较。

    arXiv:2611.00294v2 Announce Type: replace  Abstract: Mobile app stores are a rich source of user feedback containing many change requests, bug reports, improvement suggestions, etc., relevant to the post-deployment modification of software. However, it is challenging to manually translate this free format data to actionable change requests (SCRs). This review presents studies that have applied rule-based techniques, traditional ML approaches, topic modelling, hybrid approaches, and a few deep learning approaches. The data sets used range from small annotated sets to millions of reviews. However, there is inconsistency in the names, availability, annotation guidelines and sampling methods used across different studies. Many studies use Bag-of-Words and TF-IDF, while Na\"ive Bayes, Support Vector Machines, Decision Trees and Random Forest are frequently applied. Precision, Recall and F1-scores are commonly used for evaluation; however, results and experiments are not easily comparable. S
    
[^78]: 昨天的代码，今天的依赖：依赖选择如何塑造PHP迁移基准测试

    Yesterday's Code, Today's Dependencies: How Dependency Selection Shapes PHP Migration Benchmarks

    [https://arxiv.org/abs/2610.41263](https://arxiv.org/abs/2610.41263)

    该论文发现，在PHP迁移基准测试中，采用发布日期依赖选择策略相比现代选择策略可将PHP 7.4测试通过率提高7.1个百分点，表明依赖版本的选择方式会显著影响迁移基准测试的结果。

    

    迁移基准测试需要那些在迁移开始之前测试就能通过的项目。然而，一个旧项目即使代码没有任何更改，其表现也可能不同。今天重新安装它可能会引入更新的库，从而改变其初始测试结果以及迁移过程中观察到的失败情况。我们在PHP 7.4到8.3的迁移中研究了这种影响。使用未更改的项目源代码和保存的包记录，我们比较了两种依赖策略：现代选择允许快照中的所有版本，而发布日期选择只允许项目发布日期之前发布的版本。在固定Composer 2的情况下，发布日期选择在3,660个发布版本中将PHP 7.4的通过率提高了7.1个百分点。与已提交的锁定文件进行的单独比较表明，它比现代选择更接近地恢复了所记录的依赖版本。这一选择在迁移之后同样重要。在两种策略下都能在PHP 7.4上通过的项目中，PHP 8.3的测试结果……（摘要截断）

    arXiv:2610.41263v1 Announce Type: new  Abstract: Migration benchmarks need projects whose tests pass before migration begins. Yet an old project can behave differently without any change to its code. Reinstalling it today may introduce newer libraries, changing both its initial test results and the failures observed during migration.   We investigate this effect in PHP 7.4-to-8.3 migration. Using unchanged project source and saved package records, we compare two dependency policies. Modern selection allows all versions in the snapshot, while release-date selection allows only versions published by the project's release date. With Composer 2 fixed, release-date selection improves the PHP 7.4 pass rate by 7.1 pp across 3,660 releases. A separate comparison against committed locks shows that it also recovers recorded dependency versions more closely than modern selection.   The choice matters after migration as well. Among projects that pass on PHP 7.4 under both policies, PHP 8.3 test ou
    
[^79]: 骨架编码：将程序结构作为智能体的规约

    Skeleton Coding: Program Structure as a Specification for Agents

    [https://arxiv.org/abs/2610.38214](https://arxiv.org/abs/2610.38214)

    论文提出“骨架编码”概念，指出智能体在行为规约选定程序之前会先确定程序结构，而测试等规约是否充分取决于该结构本身，因此开发者应将程序结构（数据类型、模块与函数的分解）作为一种规约提供给智能体。

    

    编码智能体通常会被提供端到端的行为规约，即其生成的代码应满足的条件，例如测试用例和输入-输出示例。开发者需要提供多少这样的规约，以及提供哪些规约，才能让智能体生成开发者所期望的程序？我们认为这个问题无法先验确定，因为它取决于实现的结构，而结构是由智能体自行决定的。智能体首先会确定一个结构——即分解为数据类型、模块和函数的方式——这固定了一类候选程序；行为规约随后在该类程序中选出一个程序。对某种结构而言足够的测试集，对另一种结构可能严重欠定，需要不同的边界用例或指数级更多的示例。由于结构由智能体选择，开发者若不从数千行代码中逆向工程，就无法确保其规约是完备的（摘要在此处截断）。

    arXiv:2610.38214v1 Announce Type: cross  Abstract: Coding agents are typically provided end-to-end behavioral specifications that the code they produce should satisfy, such as tests and input--output examples. How many such specifications must a developer provide, and which ones, for the agent to produce the developer's intended program? We argue that this cannot be determined a priori, for it depends on the structure of the implementation, which is left to the agent to decide. An agent first commits to a structure---a decomposition into data types, modules, and functions---which fixes a class of candidate programs; behavioral specifications then select a program within that class. A set of tests that suffices for one structure may leave another badly underdetermined, requiring different edge cases or exponentially more examples. Since the structure is chosen by the agent, developers cannot ensure their specification is complete without reverse-engineering it from thousands of lines of
    
[^80]: 谁来验证工具？失真的格式化输出如何误导智能体及其验证器

    Who Verifies the Tools? How Warped Formatters Mislead Agents and Their Verifiers

    [https://arxiv.org/abs/2610.32110](https://arxiv.org/abs/2610.32110)

    论文通过生产RAG系统的真实案例与对照实验发现，工具输出格式化器遗漏关键信息会误导智能体和验证器，而仅需在输出中补全这些缺失细节，就能显著减少智能体的无依据声称。

    

    智能体和验证器通常依赖工具打印出的文本，而这些文本可能会歪曲工具实际执行的操作，即使工具本身的计算是正确的。我们在纽约Pilsudski研究所使用的生产级RAG研究系统中观察到了这一现象：工具的输出格式化器误导智能体向用户提供了错误或不完整的信息。每个工具都正确地完成了自己的任务，但其输出省略了搜索限制、结果数量上限或标识符命名空间等关键信息。我们还重现了一个在代码审查阶段、发布前被发现的未标注估计值。我们研究了这些格式化器是否会误导智能体和验证器，以及修复后的措辞能带来多大改善。我们向智能体提供了使用两种格式化器版本的工具结果：一个版本包含关键遗漏，另一个版本则清楚说明了所有缺失的细节。仅这一差异就改变了智能体的声称内容以及验证器所接受的内容。在四个失败案例中的三个里，智能体的无依据声称大幅下降（原文在此处截断）。

    arXiv:2610.32110v1 Announce Type: new  Abstract: Agents and verifiers often rely on the text a tool prints, and that text can misrepresent what the tool actually did, even when the tool's computation is correct. We saw this happen in a production RAG research system used by the Pilsudski Institute in New York, where tool output formatters misled an agent into giving wrong or incomplete information to users. Each tool did its job correctly, but its output omitted search limits, result caps, or identifier namespaces. We also recreated an unlabeled estimate caught in code review before release. We studied whether these formatters misled agents and verifiers and how much the repaired wording helps. We fed tool results to agents using two formatter versions: one with the key omissions and one with all missing details clearly stated. That difference alone changed what the agents claimed and what the verifiers accepted. Unsupported agent claims dropped substantially for three of the four fail
    
[^81]: QuSema：利用量子知识增强智能体检测量子库中的静默缺陷

    QuSema: Detecting Silent Bugs in Quantum Libraries via Quantum-knowledge-enhanced Agents

    [https://arxiv.org/abs/2610.10258](https://arxiv.org/abs/2610.10258)

    QuSema提出了一种自主测试智能体，将量子语义与文档约束作为源代码级语义预言机，能够在缺乏基于执行的预言机的情况下检测量子库中从有效输入产生无效输出的静默缺陷。

    

    量子库如今已成为量子算法开发的关键基础设施，但其正确性仍然难以测试。现有的测试技术主要依赖于基于失败或基于比较的预言机，只有在执行失败、违反运行时检查或与其他实现结果不一致时才能暴露缺陷。当缺乏合适的基于执行的预言机时，这些技术的适用性受到限制，导致一些静默缺陷无法被检测到。这类漏检缺陷可能产生错误结果，并将其传播到实验结论、仿真研究和算法设计之中。本文提出了QuSema，一个用于发现量子库中静默缺陷的自主测试智能体。QuSema利用量子语义和文档中的约束作为源代码级的语义预言机，评估实现逻辑是否可能从有效输入产生无效输出。它通过一个智能体循环运行，反复检查库的API文档……

    arXiv:2610.10258v1 Announce Type: new  Abstract: Quantum libraries are now critical infrastructure for quantum algorithm development, yet their correctness remains difficult to test. Existing testing techniques mainly rely on failure-based or comparison-based oracles, exposing bugs only when executions fail, violate runtime checks, or disagree with another implementation. Their applicability is limited when suitable execution-based oracles are unavailable, leaving some silent bugs undetected. Such missed bugs can produce incorrect results that propagate into experimental conclusions, simulation studies, and algorithmic designs. Here we present QuSema, an autonomous testing agent for finding silent bugs in quantum libraries. QuSema uses constraints from quantum semantics and documentation as a source-level semantic oracle to assess whether implementation logic can produce invalid outputs from valid inputs. It operates through an agentic loop that repeatedly inspects library API document
    
[^82]: 发现AppWorld与WorkArena任务验证器中的盲点

    Finding Blind Spots in AppWorld and WorkArena Task Verifiers

    [https://arxiv.org/abs/2610.09142](https://arxiv.org/abs/2610.09142)

    该论文通过基于源码信息的变异测试审计了AppWorld和WorkArena的已发布任务验证器，揭示其存在盲点——即使智能体产生了错误效果（如重复写入创建多余记录或遗留非默认持久化值），验证器仍会判定任务成功。

    

    基于执行结果的任务验证器用于判定智能体是否成功完成任务。我们通过基于源代码信息的变异测试，对已发布的AppWorld和WorkArena验证器进行了审计。主要审计过程不会修改任何已发布的检查器。在AppWorld中，复制一个非幂等的写操作会创建一条额外的记录，同时保持所有被检查字段的值不变。验证器接受了来自五个合格生成器中两个的全部三个任务变体：6/15个构造效果。在普查之后对检查器副本应用的基数补丁使所有六个测试单元格失败，同时保留了有效的对照组。在WorkArena中，我们对先前因检查器通过（PASS）而被选中的23个额外字段候选案例进行了前瞻性重跑。通过独立的表API回读，确认其中21个存在非默认的持久化值，而所有23个均获得了通过。其中两个请求的字符串实际上是存储默认值的别名。这21个被确认的错误效果横跨三个表单模板。这些选定案例在审计协议下确认了错误效果；它们

    arXiv:2610.09142v1 Announce Type: cross  Abstract: Execution-based task verifiers decide whether an agent succeeded. We audit shipped AppWorld and WorkArena verifiers with source-informed mutation tests. The main audit never modifies a shipped checker.   In AppWorld, duplicating a non-idempotent write creates an extra record while preserving every checked field value. The verifier accepts all three task variants from two of five eligible generators: 6/15 constructed effects. A cardinality patch applied to checker copies after the census makes all six cells fail while preserving valid controls.   In WorkArena, we prospectively rerun 23 extra-field candidates selected for earlier checker-PASS outcomes. Independent Table API readback confirms nondefault persisted values in 21, while all 23 receive PASS. Two requested strings are aliases of stored defaults. The 21 confirmed wrong effects span three form templates. These selected cases confirm wrong effects under the audit's protocol; they 
    
[^83]: KernelZero：共同演化的提议者与编码器，实现持续改进的GPU内核生成

    KernelZero: Co-Evolving Proposer and Coder for Continuously Improved GPU Kernel Generation

    [https://arxiv.org/abs/2609.33074](https://arxiv.org/abs/2609.33074)

    KernelZero提出一个共同演化框架，通过提议者与编码器的协同进化、前沿驱动的模块生成机制以及正确性感知的CA-GRPO强化学习方法，持续提升GPU内核自动生成的正确性与性能。

    

    高性能GPU内核对现代机器学习系统至关重要，然而自动生成既正确又高效的内核仍然充满挑战。现有的基于大语言模型的方法面临两大局限：一是缺乏与模型当前能力相匹配的高质量训练数据，二是内核正确性与性能之间存在固有权衡。为应对这些挑战，我们提出了KernelZero，一个通过两个专门化模型协同进化来持续改进GPU内核生成的框架：提议者根据API集合生成Torch模块，编码器则将这些模块转化为CUDA或Triton内核。KernelZero采用前沿驱动的模块生成机制，基于编码器当前的薄弱环节持续产生与其能力相匹配的训练模块。它还引入了正确性感知的组相对策略优化（CA-GRPO），该方法在确保内核正确的前提下优化性能。

    arXiv:2609.33074v2 Announce Type: replace  Abstract: High-performance GPU kernels are essential to modern machine learning systems, yet automatically generating kernels that are both correct and efficient remains challenging. Existing LLM-based approaches face two major limitations: the scarcity of high-quality training data aligned with the model's current capabilities, and the inherent trade-off between kernel correctness and performance. To address these challenges, we propose KernelZero, a co-evolution framework that continuously improves GPU kernel generation through two specialized models: a Proposer that generates Torch modules from API sets and a Coder that translates them into CUDA or Triton kernels. KernelZero uses a frontier-driven module generation mechanism to continuously produce capability-aligned training modules based on the Coder's current weaknesses. It further introduces Correctness-Aware Group Relative Policy Optimization (CA-GRPO), which optimizes performance only
    
[^84]: 当工具调用成功但工作流失败时：智能体-工具边界的异常

    When Tool Calls Succeed but Workflows Fail: Anomalies at the Agent-Tool Boundary

    [https://arxiv.org/abs/2609.15397](https://arxiv.org/abs/2609.15397)

    该论文提出了一个将外部世界事件与运行时观察相分离的效果历史模型，并系统梳理出八种在智能体-工具边界处反复出现的外部效果异常，以解决工具调用成功但工作流整体失败的不一致问题。

    

    AI智能体越来越多地执行长时间运行的工作流，这些工作流通过独立提供的工具将效果外部化。在重试、投机执行、并发和部分故障等情况下，产生的外部状态可能与工作流的预期决议不一致：所需的效果可能缺失或重复，已中止的效果可能依然存在，已提交的效果可能依赖于随后被撤回的临时状态。高级事务模型能够解决相关的故障，但其假设底层操作暴露了它们所依赖的语义：某个效果是否已经发生、是否可以被补偿、暂存或安全地重新排序。而共享的智能体-工具接口通常不具备这些语义。我们贡献了一个效果历史模型，该模型将外部世界中的事件与运行时对它们的观察分离开来，并编制了包含八种反复出现的外部效果异常的目录。从该目录中，我们推导出了所需边界能力……

    arXiv:2609.15397v1 Announce Type: new  Abstract: AI agents increasingly execute long-running workflows that externalize effects through independently supplied tools. Under retries, speculative execution, concurrency, and partial failures, the resulting external state may be inconsistent with the workflow's intended resolution: required effects may be missing or duplicated, aborted effects may survive, and committed effects may depend on provisional state that is later withdrawn. Advanced transaction models address related failures, but assume that lower-level operations expose the semantics they depend on: whether an effect occurred, whether it can be compensated, staged, or safely reordered. Shared agent-tool interfaces usually do not.   We contribute an effect-history model that separates events in the external world from the runtime's observations of them, and a catalog of eight recurring external-effect anomalies. From the catalog we derive the boundary capabilities required to exc
    
[^85]: Claude Code 完整用户手册

    Claude Code Complete User Handbook

    [https://arxiv.org/abs/2608.26742](https://arxiv.org/abs/2608.26742)

    本书提出，在Claude Code代理环境中，安全高效运行的关键在于定义明确的可观察完成条件，并通过指令、权限、沙箱和系统隔离四层控制栈来管理系统性风险。

    

    arXiv:2608.26742v1 公告类型：交叉 摘要：Claude Code 是一个代理式工作环境：一种语言模型在循环中运行，具备文件系统访问、Shell 执行、浏览器控制、定时和云执行、通过模型上下文协议连接外部工具以及多代理编排能力。其能力范围现已超出单个从业者仅靠注意力所能监督的限度，其故障模式是系统性的而非局部性的：未审查的钩子、过度扩展的连接器、过时的完成条件、继承账户所有凭据的自主例程。本书是一本面向操作该系统的任务导向参考手册，旨在安全高效地运行该系统，专为对结果负责的从业者编写。书中提出四个论点：第一，没有明确且可观察的完成条件的能力并非生产力；第二，指令、权限执行、沙箱和操作系统隔离是控制栈的四个不同层次。

    arXiv:2608.26742v1 Announce Type: cross  Abstract: Claude Code is an agentic work environment: a language model operating in a loop with filesystem access, shell execution, browser control, scheduled and cloud execution, external tool connections through the Model Context Protocol, and multi-agent orchestration. Its capability envelope now exceeds what one practitioner can supervise by attention alone, and its failure modes are systemic rather than local: an unreviewed hook, an over-scoped connector, a stale completion condition, an autonomous routine inheriting every credential on an account. This book is a task-oriented reference for operating that system safely and productively, written for practitioners accountable for the result. It advances four propositions. First, capability without a defined and observable completion condition is not productivity. Second, instruction, permission enforcement, sandboxing and operating-system isolation are four distinct layers of a control stack,
    
[^86]: SPECMINE：一个大规模的规范驱动开发工件语料库

    SPECMINE: A Large-Scale Corpus of Spec-Driven Development Artifacts

    [https://arxiv.org/abs/2608.25202](https://arxiv.org/abs/2608.25202)

    我们提出了SPECMINE，这是首个大规模语料库，通过两次普查系统地捕捉了GitHub上规范驱动开发工件，为研究规范如何转化为代码提供了基础数据。

    

    arXiv:2608.25202v1 公告类型：新 摘要：规范驱动开发（SDD）是一种快速兴起的新实践，其中由开发者编写、或（更常见地）由AI工具起草再由开发者整理的、结构化自然语言规范，驱动AI编码代理的实现。自2025年以来，一波工具（如GitHub Spec Kit [3]、OpenSpec [4]、AWS Kiro [5]以及数十种其他工具）已经出现，但这些工具产生的工件从未被大规模研究过。我们提出了SPECMINE，一个通过两次普查捕捉公共GitHub仓库中SDD的语料库：一次广泛普查覆盖了大多数工具的spec.md/specs.md文件（涵盖73,030个仓库中的470,795个文件，归属于17个命名工具），以及一次针对Kiro独特的需求/设计/任务布局的普查（涵盖12,910个仓库中的98,574个文件）。每个规范都附有完整的仓库元数据、完整的提交历史以及解析后的文档结构。规范如何转化为代码本身就是一个开放问题，因此对于...

    arXiv:2608.25202v1 Announce Type: new  Abstract: Spec-Driven Development (SDD) is a fast-emerging practice in which a structured natural-language specification, written by a developer, or (more often) drafted by an AI tool and then curated by the developer, drives an AI coding agent's implementation. A wave of tooling (GitHub Spec Kit [3], OpenSpec [4], AWS Kiro [5], and dozens of others) has appeared since 2025, yet the artifacts these tools produce have never been studied at scale. We present SPECMINE, a corpus that captures SDD in public GitHub repositories through two censuses: a broad census of spec.md/specs.md files covering most tools (470,795 files across 73,030 repositories, attributed to 17 named tools), and a Kiro census of its distinct requirements/design/tasks layout (98,574 files across 12,910 repositories). Each spec is enriched with full repository metadata, complete commit history, and parsed document structure. How a spec becomes code is itself an open question, so fo
    
[^87]: eIRWR：用于微服务可扩展根因分析的增强型带重启迭代随机游走

    eIRWR: Enhanced Iterative Random Walk with Restart for Scalable Root Cause Analysis in Microservices

    [https://arxiv.org/abs/2608.08073](https://arxiv.org/abs/2608.08073)

    本文发现基本IRWR的“弹性阻尼”等价于调整重启概率的个性化PageRank，并提出eIRWR，通过幂律锐化将重启质量集中于最可疑节点，实现微服务系统中更准确且可扩展的根因分析。

    

    微服务系统中的根因分析（RCA）必须在蔓延至数百甚至数千个依赖服务的症状背后找出故障服务。基于图的随机游走方法在服务依赖图上传播异常证据。然而，现有的异常重启游走从原始异常向量出发进行重启，而该向量往往被异常程度强烈的下游受害者节点所主导，而非异常程度较低的源头节点。通过对一种基本的异常重启游走方法——带重启迭代随机游走（IRWR）——进行受控消融实验，我们证明其“弹性阻尼”等价于个性化PageRank（PPR）中更高的重启概率。因此，基本IRWR并不比调整了重启概率的PPR更强，我们将其作为更公平的基线。随后，我们提出增强型IRWR（eIRWR），它超越了单纯的重启概率调整：(a) 通过幂律锐化将重启质量集中于最可疑的节点，(b) 增加了se……（摘要在此处截断）

    arXiv:2608.08073v2 Announce Type: replace-cross  Abstract: Root cause analysis (RCA) in microservice systems must find the faulty service behind symptoms that spread across hundreds or thousands of dependent services. Graph-based random walk methods propagate anomaly evidence over the service dependency graph. However, existing anomaly-restart walks restart from the raw anomaly vector, which is often dominated by strongly anomalous downstream victims rather than the less anomalous source. Through a controlled ablation of a basic anomaly-restart walk, the Iterative Random Walk with Restart (IRWR), we show that its ``resilience damping'' is equivalent to a higher restart probability in Personalized PageRank (PPR). Basic IRWR is therefore no stronger than a restart-tuned PPR, which we adopt as a fairer baseline. We then present Enhanced IRWR (eIRWR), which goes beyond restart tuning. It (a)~concentrates restart mass on the most suspicious nodes through power-law sharpening and (b)~adds se
    
[^88]: LangChoiceBench：测量与解释大语言模型中的编程语言选择

    LangChoiceBench: Measuring and Explaining Programming-Language Choice in LLMs

    [https://arxiv.org/abs/2608.06041](https://arxiv.org/abs/2608.06041)

    提出了LangChoiceBench基准，系统测量25个大语言模型在28个项目中的编程语言选择行为，发现模型普遍过度偏好Python且多为未经深思熟虑的自动选择，较小的开放权重模型偏好更强、语言多样性更低。

    

    大语言模型（LLM）在生成项目级代码时已被证明对Python表现出强烈的偏好，但目前尚无系统性的方法来在新模型中测量这种行为。为弥合这一差距，我们提出了LangChoiceBench，这是一个项目级代码生成基准，用于测量Python偏好、推荐与实现的一致性以及语言多样性。LangChoiceBench涵盖七个软件领域中的28个项目，这些领域中Python往往不是一个好的默认选择。我们评估了25个多样化的LLM，发现Python仍被严重过度选用，推荐与实现之间的一致性较低，且较小的开放权重模型通常表现出更强的Python偏好和更低的语言多样性。我们进一步分析了9,826条推理轨迹，发现大多数Python选择是自动做出的或主要由易用性驱动，而非对项目需求的明确考量。在一组较小但重要的……（摘要原文在此处截断）

    arXiv:2608.06041v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) have been shown to exhibit strong Python preferences when generating project-level code, but there is currently no systematic way to measure this behaviour across new models. To bridge this gap, we introduce LangChoiceBench, a project-level code-generation benchmark for measuring Python preference, recommendation-implementation consistency, and language diversity. LangChoiceBench covers 28 projects across seven software areas where Python is often a poor default. We evaluate 25 diverse LLMs and find that Python remains heavily over-selected, recommendation-implementation consistency is low, and smaller open-weight models generally show stronger Python preference and lower language diversity. We further analyse 9,826 reasoning traces and find that most Python choices are automatic or driven primarily by ease, rather than explicit consideration of project requirements. In a smaller but important set o
    
[^89]: Change2Task：从仓库变更到可执行的编码智能体任务与环境

    Change2Task: From Repository Changes to Executable Coding Agent Tasks and Environments

    [https://arxiv.org/abs/2607.28591](https://arxiv.org/abs/2607.28591)

    Change2Task 提出了一个基于仓库历史的系统，能够将已合并的拉取请求自动转换为健康现代代码版本上经过验证的可执行编码智能体任务与环境，为编码智能体的训练、基准测试和持续评估提供可扩展的数据供给。

    

    扩展编码智能体需要持续供给可执行的数据，用于训练、基准测试和持续评估。每个任务必须将真实的软件状态与任务规范、开发工具和可靠的验证相结合。为了扩大这一供给，我们提出了 Change2Task，这是一个基于仓库历史的系统，它将已合并的拉取请求（PR）转换为同一仓库健康现代版本上经过验证的任务。它将历史证据与演进的代码进行对齐，通过补丁反转、代码映射或智能体重建三种方式重建任务状态，并验证从健康基础状态到任务状态再到恢复状态的完整生命周期。通过从维护良好的环境中推导出多个基于开发者证据的任务，Change2Task 为编码智能体的训练和评估提供了可执行数据，同时减少了重复的环境搭建、存储和任务构建成本。我们评估了五个编码智能体……（原文截断）

    arXiv:2607.28591v2 Announce Type: replace-cross  Abstract: Scaling coding agents requires a continuing supply of executable data for training, benchmarking, and continuous evaluation. Each task must couple a realistic software state with a specification, development tools, and reliable verification. To expand this supply, we present Change2Task, a system grounded in repository history that converts merged pull requests (PRs) into verified tasks on healthy modern revisions of the same repository. It aligns historical evidence with evolved code, reconstructs task states through Patch Reversal, Code Mapping, or Agent Reconstruction, and validates the lifecycle from a healthy base to a task state and a restored state. By deriving multiple tasks grounded in developer evidence from maintained environments, Change2Task provides executable data for coding agent training and evaluation while reducing repeated environment setup, storage, and task construction effort. We evaluate five coding agen
    
[^90]: 代码锚定的评审反馈提高正确性并降低大型R文件的托管编辑成本

    Code-Anchored Review Feedback Improves Correctness and Lowers Hosted Editing Cost on Large R Files

    [https://arxiv.org/abs/2607.12713](https://arxiv.org/abs/2607.12713)

    研究表明，将代码评审反馈以“代码锚定”格式（每条评论锚定并引用其涉及的代码）而非整体列表形式提供给大模型，可使大型R文件编辑的输出令牌最多减少50.5%、托管成本最多降低46.3%，并显著提升编辑的正确性。

    

    令牌（token）驱动生成式人工智能（GAI）代码编辑的成本与延迟，而正确性决定一次编辑是否有用。我们研究评审者反馈的格式——一条人机协作（HAIC）通道——如何同时影响这两者。在一项针对36个合成基础R任务、涉及七个模型的配对实验中，相同的修改请求以整体列表（对照组）或FileMark导出内容（实验组）的形式提供。该代码评审扩展将每条评论锚定到其涉及的代码并加以引用。对于典型任务，各模型的总令牌数变化范围为-4.2%到+14.7%，而输出令牌数下降了14.7%（Claude Opus）和50.5%（Claude Sonnet）。由于输出令牌的成本是输入令牌的五倍，在超过100行的文件上进行编辑的托管成本下降了20.0%（Opus）和46.3%（Sonnet）。该处理方式使五个本地模型中四个的正确运行比例提高了24%至46%。一项分解实验对这些收益进行了定位分析。

    arXiv:2607.12713v2 Announce Type: replace-cross  Abstract: Tokens drive the cost and latency of generative AI (GAI) code editing, and correctness decides whether an edit is useful. We study how the format of a reviewer's feedback, a human-AI collaboration (HAIC) channel, impacts both. In a paired experiment on 36 synthetic base-R tasks with seven models, the same requested changes were delivered as a holistic list (control) or as the export of FileMark (treatment). That code-review extension anchors each comment to the code it concerns and quotes it. For the typical task, total tokens changed by -4.2% to +14.7% across models, while output tokens fell by 14.7% (Claude Opus) and 50.5\% (Claude Sonnet). Since an output token costs five times as much as an input token, the hosted cost of an edit on files over 100 lines fell by 20.0% (Opus) and 46.3% (Sonnet). The treatment raised the correct-run share of four of five local models by 24% to 46%. A decomposition experiment places these gains
    
[^91]: Flex：一个基础性的、可产生证明的Horn求解器

    Flex: A Foundational Proof-Producing Horn Solver

    [https://arxiv.org/abs/2607.12226](https://arxiv.org/abs/2607.12226)

    Flex是一个基于Lean元编程实现的可产生内核验证证明的约束Horn子句（CHC）求解器，它将谓词合成与证明构造相结合，使规范可直接使用原生定义，并允许用户在自动化失败时补全证明。

    

    约束Horn子句（CHCs）是程序验证中一种用途广泛的中间表示，但主流的CHC求解器依赖于基于SMT的推理，且仅限于所支持的后台理论。涉及用户自定义递归函数或定制运算符的规范可能需要超出这些理论的额外语义编码和推理。我们观察到，CHC本质上是基础逻辑中带存在量词的命题，因此可以在交互式定理证明器中加以证明，其中规范可以使用原生定义，并且当自动化陷入停滞时用户可以手动补全证明。然而，在基础性（foundational）设置下，每个Horn变量的解都必须由经过内核检查的证明来保证。我们提出了Flex，一个以Lean元程序形式实现的可产生证明的CHC求解器，它将谓词合成与证明构造相结合。Flex包含一套策略：其中Zap可为无环Horn……（原文摘要在此处被截断）

    arXiv:2607.12226v2 Announce Type: replace-cross  Abstract: Constrained Horn Clauses (CHCs) are a versatile intermediate representation for program verification, but leading CHC solvers rely on SMT-based reasoning over supported background theories. Specifications involving user-defined recursive functions or bespoke operators can require additional semantic encodings and reasoning beyond these theories. We observe that CHCs are existentially quantified propositions in a foundational logic, and so can be proved within an interactive theorem prover, where specifications can use native definitions, and users can complete proofs when automation stalls. However, in a foundational setting, every Horn variable solution must be justified by a kernel-checked proof. We present Flex, a proof-producing CHC solver implemented as a Lean metaprogram, that combines predicate synthesis with proof construction. Flex comprises a suite of tactics: Zap, which synthesizes strongest predicates for acyclic Ho
    
[^92]: 挖掘工作流图用于对话式LLM智能体的黑盒边界测试

    Mining Workflow Graphs for Black-Box Boundary Testing of Conversational LLM Agents

    [https://arxiv.org/abs/2607.06873](https://arxiv.org/abs/2607.06873)

    提出了AgentEval黑盒测试框架，通过与智能体交互挖掘其对话工作流图，并在各节点重放对话路径后施加扰动，从而系统性发现并测试身份验证、确认步骤等工作流边界处的故障。

    

    对话式LLM智能体的故障可能产生现实世界的影响，例如智能体在未经用户确认的情况下完成交易。这类故障的一个重要类别发生在工作流边界处，例如身份验证和确认步骤，而测试只有在经过多轮对话之后才能到达这些边界。我们提出了AgentEval，一个用于发现并测试这些边界的黑盒测试框架。AgentEval与智能体进行交互，以挖掘出一个对话工作流图——这是对智能体行为的建模，其中节点表示智能体的活动，边表示在这些活动之间观察到的用户操作。它在该图的每个节点处生成测试，在施加扰动之前先重放到达该节点的对话路径。随后AgentEval运行每个测试，并由一个独立的LLM裁判仅根据对话轮次来判断测试通过与否。我们将AgentEval与一个拥有特权的白盒审计器进行了基准对比……（摘要原文在此处截断）

    arXiv:2607.06873v2 Announce Type: replace  Abstract: Failures of conversational LLM agents can have real-world consequences, for example when an agent completes a transaction without the user's confirmation. An important class of such failures occurs at workflow boundaries, such as identity checks and confirmation steps, which a test can reach only after several turns of conversation. We present AgentEval, a black-box testing framework that discovers and tests these boundaries. AgentEval interacts with an agent to mine a conversational workflow graph, a model of its behavior whose nodes are the agent's activities and whose edges are the user actions observed between them. It generates tests at each node of this graph, replaying the conversational path to that node before applying a perturbation. AgentEval then runs each test, and a separate LLM judge decides whether it passes or fails using only the conversation turns. We benchmark AgentEval against a privileged, white-box auditor with
    
[^93]: AutoCedar：一个验证器引导的访问控制策略合成智能体框架

    AutoCedar: An Agentic Framework for Verifier-Guided Access Control Policy Synthesis

    [https://arxiv.org/abs/2607.03656](https://arxiv.org/abs/2607.03656)

    AutoCedar 提出了一种验证器引导的智能体框架，通过将访问控制需求分解为可审查、可机械验证的“意图原子”，并利用验证器对照经人工审查的目标迭代修复策略，确保从自然语言合成的 Cedar 访问控制策略真正符合用户意图。

    

    大语言模型正越来越多地被用于将自然语言需求转化为代码。在访问控制领域，这种捷径是危险的：一个生成的策略可能编译通过且看起来正确，却授予了未经任何人批准的访问权限。难点不仅在于编写策略代码，还在于在编写代码之前明确需求的确切含义，然后再检查最终策略是否真正满足该意图。我们提出了 AutoCedar，一个由验证器引导的系统，它首先将自然语言访问控制需求转化为经过审查、可检验的目标，然后针对该目标合成 Cedar 策略。AutoCedar 将模式（schema）和策略的编写分解为小的意图原子：即关于词汇和行为的可审查声明。一旦这些原子通过机械验证和人工意图审查，模型便会提出候选策略，验证器会对照已批准的目标对其进行检查，并将每一次失败转化为修复信息。

    arXiv:2607.03656v2 Announce Type: replace-cross  Abstract: Large Language Models are increasingly used to turn natural-language requirements into code. In access control, that shortcut is dangerous: a generated policy can compile and read correctly while granting access that no one approved. The difficulty is not only writing policy code. It is fixing what the requirements mean before code is written, and then checking that the final policy actually satisfies that intent. We present AutoCedar, a verifier-guided system that first turns natural-language access-control requirements into a reviewed, checkable target, and then synthesizes Cedar policies against that target. AutoCedar decomposes schema and policy authoring into small intent atoms: reviewable claims about vocabulary and behavior. Once those atoms pass mechanical validation and human intent review, the model proposes a candidate policy, the verifier checks it against the approved target, and each failure is turned into a repai
    
[^94]: 智能体AI是否已为真实世界的硬件工程做好准备？基于Phoenix-bench的深入探究

    Is Agentic AI Ready for Real-World Hardware Engineering? A Deep Dive with Phoenix-bench

    [https://arxiv.org/abs/2605.15226](https://arxiv.org/abs/2605.15226)

    该论文提出了Phoenix-bench——首个同时要求仓库导航、层次感知定位、EDA可执行验证和维护式补丁的硬件工程智能体基准，包含511个经核实的Verilator实例，并系统评估了商业与开源智能体在真实硬件工程任务上的能力。

    

    我们探讨为软件工程构建的智能体AI系统能否迁移到真实的硬件工程场景中。现有的硬件LLM基准测试均只孤立地考察子任务，没有一个能够同时要求仓库导航、层次感知的定位、电子设计自动化（EDA）可执行验证以及维护式补丁修改。我们提出了Phoenix-bench——一个包含来自114个GitHub仓库的511个经过验证的Verilator实例的同步语料库，每个实例都附带开发者补丁、设计流程标签、失败转通过（fail-to-pass）与通过转保持（pass-to-pass）测试平台，以及由Docker固定的EDA环境，从而使解决率差异能够反映智能体自身的行为而非工具链的可用性。借助Phoenix-bench，我们对四个商业智能体和八种开源智能体结构在四个LLM骨干模型上进行了统一评估，并加入两种诊断性干预（文件级oracle定位和一轮测试平台日志反馈）。三个fin……（摘要原文在此处截断）

    arXiv:2605.15226v2 Announce Type: replace-cross  Abstract: We ask whether agentic AI systems built for software engineering transfer to realistic hardware engineering. Existing hardware LLM benchmarks isolate sub-tasks but none jointly requires repository navigation, hierarchy-aware localization, Electronic Design Automation (EDA) executable verification, and maintenance-style patching. We introduce \textbf{Phoenix-bench}, a synchronized corpus of 511 verified Verilator instances from 114 GitHub repositories, each shipped with the developer patch, design-flow labels, fail-to-pass and pass-to-pass testbenches, and a Docker-pinned EDA environment so resolved-rate differences reflect agent behavior rather than toolchain availability. Using Phoenix-bench we run a uniform evaluation of four commercial agents and eight open-source agentic structures across four LLM backbones, plus two diagnostic interventions (file-level oracle localization and one round of testbench-log feedback). Three fin
    
[^95]: ORACLE-SWE：量化关键信息信号对软件工程代理的贡献

    ORACLE-SWE: Quantifying the Contribution of Critical Information Signals on SWE Agents

    [https://arxiv.org/abs/2604.07789](https://arxiv.org/abs/2604.07789)

    该论文提出ORACLE-SWE，首次系统量化了复现测试、回归测试、编辑位置、执行上下文和API用法等关键中间信息信号对软件工程代理任务成功的贡献，并测量了代理发现的信号与理想神谕信号之间尚存的改进空间。

    

    语言模型（LM）代理的最新进展日益聚焦于仓库级软件工程（SWE）任务，由此产生了多种代理式工作流、训练方法和失败分析技术。有趣的是，这些方法并非直接生成或评估编辑补丁，而是首先检查并改进在代理探索过程中收集到的中间信息信号，包括复现测试、回归测试、编辑位置、执行上下文和API用法。尽管不同的研究考察了这些信号的不同子集，但目前尚无工作系统地比较它们在提升最终任务成功率方面的相对有效性，也没有测量每个信号中还剩余多大的改进空间。为填补这一空白，本工作量化了“神谕”信号与代理提取的信号对最终任务成功的贡献，并测量了当前代理所发现的信号与其对应神谕信号之间的差距。

    arXiv:2604.07789v3 Announce Type: replace-cross  Abstract: Recent advances in Language Model (LM) agents have increasingly focused on repository-level software engineering (SWE) tasks, leading to various agentic workflows, training methods and failure analysis. Interestingly, rather than directly generating or evaluating edit patches, these approaches first examine and improve intermediate information signals collected during agent exploration, including Reproduction Test, Regression Test, Edit Location, Execution Context, and API Usage. Although different studies examine different subsets of these signals, no work has systematically compared their relative effectiveness in improving final task success or measured the remaining space for improvement in each signal. To address this gap, this work quantifies the contributions of both oracle and agent-extracted signals to final task success and measures the gap between current agent-found signals and their oracle counterparts. We hope our
    
[^96]: QiMeng-PRepair：基于编辑感知奖励优化的精确代码修复

    QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization

    [https://arxiv.org/abs/2604.05963](https://arxiv.org/abs/2604.05963)

    该论文提出PRepair框架，通过Self-Breaking缺陷程序生成和基于编辑感知奖励的EA-GRPO训练，抑制大语言模型代码修复中的过度编辑问题，实现最大化复用正确代码、仅修复缺陷部分的精确修复，修复精确度最高提升31.4%。

    

    大语言模型（LLMs）在程序修复任务中表现出色，但常常存在过度编辑的问题，即过多的修改会覆盖原本正确的代码，并妨碍缺陷定位。我们系统地量化了这一问题的影响，并提出了“精确修复”任务，即在最大化复用正确代码的同时，仅修复有缺陷的部分。基于这一洞察，我们提出了PRepair框架，以缓解过度编辑并提升修复准确性。PRepair包含两个组件：Self-Breaking（自我破坏），通过受控缺陷注入和最小-最大采样生成多样化的有缺陷程序；Self-Repairing（自我修复），采用编辑感知组相对策略优化（EA-GRPO）进行训练，利用编辑感知奖励鼓励模型进行最小但正确的编辑。实验表明，PRepair在fix₁@1指标下将修复精确度提升了高达31.4%，该指标同时衡量修复的正确性与修改范围，并显著提升了……（原文摘要在此处截断）

    arXiv:2604.05963v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) achieve strong program repair performance but often suffer from over-editing, where excessive modifications overwrite correct code and hinder bug localization. We systematically quantify its impact and introduce precise repair task, which maximizes reuse of correct code while fixing only buggy parts. Building on this insight, we propose PRepair, a framework that mitigates over-editing and improves repair accuracy. PRepair has two components: Self-Breaking, which generates diverse buggy programs via controlled bug injection and min-max sampling, and Self-Repairing, which trains models with Edit-Aware Group Relative Policy Optimization (EA-GRPO) using an edit-aware reward to encourage minimal yet correct edits. Experiments show that PRepair improves repair precision by up to 31.4% under $\mathrm{fix}_1@1$, a metric that jointly considers repair correctness and extent, and significantly increases decod
    
[^97]: 基于智能体技能挖掘的代码效率自动优化

    Automated Code Efficiency Optimization via Agent Skill Mining

    [https://arxiv.org/abs/2603.27850](https://arxiv.org/abs/2603.27850)

    本文提出EffiSkill两阶段框架，将慢代码到快代码的常见转换挖掘并沉淀为记录意图、适用性、流程、效果与风险的“算子技能”卡片库，使LLM无需执行反馈即可诊断程序、检索技能并生成多个候选优化方案，从而自动提升代码效率。

    

    代码效率是软件质量的一个基本方面，然而如何利用大语言模型（LLM）来优化已经正确的程序仍然具有挑战性。先前的方法依赖于一次性重写、检索示例或指令、微调或候选搜索等手段。在本文中，我们提出了EffiSkill，这是一个两阶段框架，它将反复出现的“慢代码到快代码”转换建模为算子技能：即持久化的卡片，其中记录了每次转换的意图、适用条件、操作流程、预期效果和风险。阶段一从慢/快程序对中提炼出冻结的、针对特定编程语言的技能库。阶段二对新的程序进行诊断，检索相关技能，组合出多种备选优化方案，并生成最多八个候选程序。这些步骤的完成无需候选程序的执行反馈；随后使用公开测试来过滤和排序候选程序，再用留出的私有测试来确定所报告的正确性和加速比。在每种语言623个EffiBench-X任务上……（原文在此处截断）

    arXiv:2603.27850v2 Announce Type: replace-cross  Abstract: Code efficiency is a fundamental aspect of software quality, yet how to use large language models (LLMs) to optimize correct programs remains challenging. Prior approaches rely on one-shot rewriting, retrieved exemplars or instructions, fine-tuning, or candidate search. In this paper, we present EffiSkill, a two-stage framework that models recurring slow-to-fast transformations as Operator Skills: persistent cards that record a transformation's intent, applicability, procedure, expected effect, and risks. Stage I distills frozen, language-specific skill libraries from slow/fast program pairs. Stage II diagnoses a new program, retrieves relevant skills, composes alternative plans, and generates up to eight candidates. These steps finish without candidate-execution feedback; public tests then filter and rank candidates, and held-out private tests determine reported correctness and speedup. Across 623 EffiBench-X tasks per languag
    
[^98]: RepoLaunch：跨语言与平台的代码仓库构建与管理自动化

    RepoLaunch: Automating Build and Management of Code Repositories across Languages and Platforms

    [https://arxiv.org/abs/2603.05026](https://arxiv.org/abs/2603.05026)

    RepoLaunch是一个跨多种编程语言和操作系统的智能体框架，能够自动解决依赖、编译源代码并提取测试结果，构建成功率达78%（超越仅支持Python/Linux的先前系统18%），并支撑了仅需任务设计阶段人工输入的全自动SWE数据集创建流水线。

    

    语言模型（LM）智能体推动了自动化软件工程（SWE）领域的重大进展，然而大规模构建和测试软件仓库在很大程度上仍然是一个依赖手工、劳动密集型的瓶颈。在这项工作中，我们提出了RepoLaunch，一个新颖的智能体框架，能够在多种编程语言和操作系统上自动解决依赖关系、编译源代码并提取测试结果。RepoLaunch实现了78%的构建成功率，比仅支持Python/Linux的先前系统高出18%。为了展示其应用，我们进一步提出了一个由RepoLaunch驱动的用于SWE数据集创建的全自动流水线，该流水线仅在任务设计阶段需要人工输入。RepoLaunch已开源，其自动化任务生成流水线已被近期多项关于智能体基准测试和训练的工作所采用。

    arXiv:2603.05026v4 Announce Type: replace-cross  Abstract: Language model (LM) agents have driven substantial progress in automated software engineering (SWE), yet building and testing software repositories at scale remains a largely manual and labor-intensive bottleneck. In this work, we introduce RepoLaunch, a novel agentic framework that automatically resolves dependencies, compiles source code, and extracts test results across diverse programming languages and operating systems. RepoLaunch achieves a 78% build success rate, outperforming the Python/Linux-only prior system by 18%. To demonstrate its application, we further present a fully automated pipeline for SWE dataset creation driven by RepoLaunch, which only requires human input at the task-design stage. RepoLaunch is open-sourced, and its automated task-generation pipeline has been adopted by several recent works on agentic benchmarking and training.
    
[^99]: Clawdrain：利用工具调用链对OpenClaw智能体实施隐蔽的令牌耗尽攻击

    Clawdrain: Exploiting Tool-Calling Chains for Stealthy Token Exhaustion in OpenClaw Agents

    [https://arxiv.org/abs/2603.00902](https://arxiv.org/abs/2603.00902)

    Clawdrain是一种木马化的OpenClaw技能，它通过分段验证过程（SVP）利用工具调用链制造高成本的多轮调用，在保持最终任务输出合法的同时隐蔽地耗尽LLM智能体的令牌资源，实现2-16倍的令牌放大。

    

    依赖工具调用的LLM智能体通过第三方技能来扩展其能力，但这种开放性暴露出一个供应链攻击面，恶意技能指令可以在多轮交互中操纵智能体的执行。在本工作中，我们提出了Clawdrain，一种木马化的OpenClaw技能，它通过分段验证过程（SVP）诱发资源耗尽。SVP结合了有状态的推进、修复驱动的重复和终止机制，在保持最终载荷合法性的同时，制造出代价高昂的工具调用链。我们在类生产环境的OpenClaw部署中，使用真实API计费，对Gemini 2.5 Pro、DeepSeek V4 Flash和Claude Sonnet 4.6三种模型进行了评估。使用基础SVP，Clawdrain实现了2-16倍的令牌放大，且三种模型表现出各不相同的行为，包括自回归式服从、主动编写脚本以及扩展的试错推理。我们的发现揭示了智能体资源耗尽……

    arXiv:2603.00902v3 Announce Type: replace-cross  Abstract: Tool-calling LLM agents rely on third-party skills to extend their capabilities, but this openness exposes a supply-chain attack surface in which malicious skill instructions can manipulate agent execution across multiple turns. In this work, we present Clawdrain, a Trojanized OpenClaw skill that induces resource exhaustion through a Segmented Verification Process (SVP). SVP combines stateful progression, repair-driven repetition, and termination to create costly tool-calling chains while preserving a legitimate final payload. We evaluate Clawdrain in a production-like OpenClaw deployment with real API billing across Gemini 2.5 Pro, DeepSeek V4 Flash, and Claude Sonnet 4.6. Clawdrain achieves 2-16 times token amplification with plain SVP, while the three models exhibit distinct behaviors, including autoregressive compliance, proactive scripting, and extended trial-and-error reasoning. Our findings reveal that agent resource exh
    
[^100]: 它会存活吗？解读开源项目中AI智能体编写代码的命运

    Will It Survive? Deciphering the Fate of AI Agent-Authored Code in Open Source

    [https://arxiv.org/abs/2601.16809](https://arxiv.org/abs/2601.16809)

    本研究通过对215个开源仓库中约65万行代码的生存分析发现，AI智能体编写的代码并不比人类代码更快被修改，甚至在同等条件下存活时间更长，从而有力反驳了“智能体代码是一次性的”这一担忧。

    

    arXiv:2601.16809v2 通告类型：replace-cross 摘要：诸如GitHub Copilot、OpenAI Codex和Devin等AI编程智能体如今可以发起被合并进开源项目的拉取请求（PR）。行业报告显示，AI辅助编写的代码在写完后往往很快就会被修改或回滚，这引发了人们对智能体代码是“一次性的”这一担忧。然而，关于智能体代码在被合并之后究竟发生了什么，人们知之甚少。了解这些代码能够存活多久、以及它为何被修改，可以帮助维护者决定需要对其进行多少审查。我们对来自AIDev数据集的215个开源仓库中2,794个已合并PR在排除第三方和自动生成路径之后新增的648,959行代码进行了生存分析，并在相同仓库内比较了归因于智能体的代码与人类编写的代码。结果表明，归因于智能体的代码并不比人类代码更早被修改。当每个PR权重相同时，归因于智能体的代码行被修改的时间反而更晚（HR = 0.83，95% CI [0.75, 0.93]）。这一差异来自于监督……（原文摘要在此处截断）

    arXiv:2601.16809v2 Announce Type: replace-cross  Abstract: AI coding agents such as GitHub Copilot, OpenAI Codex and Devin now open pull requests (PRs) that are merged into open-source projects. Industry reports suggest that AI-assisted code is revised or reverted soon after it is written, which raises the concern that agent code is "disposable". However, little is known about what happens to agent code after it is merged. Knowing how long this code survives, and why it is changed, can help maintainers decide how much review it needs. We performed a survival analysis of the 648,959 lines that 2,794 merged PRs added outside vendored and generated paths in 215 open-source repositories from the AIDev dataset, comparing agent-attributed and human code within the same repositories. Agent-attributed code is not modified sooner than human code. When each PR is weighted equally, agent-attributed lines are modified later (HR = 0.83, 95% CI [0.75, 0.93]). This difference comes from the supervise
    
[^101]: VeruSAGE-Bench：一个用于Rust系统验证的基准测试套件

    VeruSAGE-Bench: A Benchmark Suite for Rust System Verification

    [https://arxiv.org/abs/2512.18436](https://arxiv.org/abs/2512.18436)

    该论文构建了首个面向真实系统级形式化验证的基准测试套件VeruSAGE-Bench，包含从八个经Verus验证的开源Rust系统中提取的849个证明任务，其形式化规约规模平均是以往基准的20倍以上，并用它系统评估了六个大语言模型的软件系统证明能力。

    

    大语言模型（LLM）已经展现出理解和开发代码的出色能力。然而，它们能否对软件系统进行严格推理并形式化地证明其正确性仍然存疑。目前用于形式化证明生成的基准测试主要聚焦于孤立的任务，通常是使用基本数据结构的算法类问题。我们精心构建了一个面向系统级证明生成的全新基准测试套件——VeruSAGE-Bench，它由从八个经过Verus验证的开源Rust系统中提取的849个证明任务组成，涵盖操作系统、存储系统、内存分配器等领域。这些任务比以往基准测试中的任务复杂得多，其形式化规约（以代码行数计）平均是以往基准的20倍以上，并且需要更广泛的验证技术。我们在VeruSAGE-Bench上评估了六个大语言模型（o4-mini、GPT-5、Sonnet 4、Sonnet 4.5、Opus 4.5、Qwen3.5），并展示了……（原文在此处截断）

    arXiv:2512.18436v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have shown impressive capability to understand and develop code. However, their capability to rigorously reason about and formally prove the correctness of software systems remains in question. Current benchmarks for formal proof generation focus on isolated tasks, typically algorithmic-style problems utilizing basic data structures. We curate a new benchmark suite for system-level proof generation, VeruSAGE-Bench, which consists of 849 proof tasks extracted from eight open-source Verus-verified Rust systems, covering operating systems, storage systems, memory allocators, and more. These tasks are much more complicated than those in previous benchmarks, containing on average more than 20x the formal specification (in Lines of Code) and requiring a much broader array of verification techniques. We evaluate six LLMs on VeruSAGE-Bench (o4-mini, GPT-5, Sonnet 4, Sonnet 4.5, Opus 4.5, Qwen3.5), and demon
    
[^102]: 金融新闻情感分析中嵌入表示的比较评估

    Comparative Evaluation of Embedding Representations for Financial News Sentiment Analysis

    [https://arxiv.org/abs/2512.13749](https://arxiv.org/abs/2512.13749)

    本研究在仅有349条金融新闻标题的小数据集上，比较评估了Word2Vec、GloVe和句子变换器嵌入结合梯度提升的情感分类效果，发现预训练嵌入在数据不足时收益递减，且小验证集易导致模型选择中的过拟合。

    

    金融情感分析有助于增强对市场的理解。然而，标准的自然语言处理（NLP）方法在应用于小数据集时会遇到重大挑战。本研究对资源受限环境下用于金融新闻情感分类的基于嵌入的技术进行了比较评估。在一个人工标注的包含349条金融新闻标题的数据集上，结合梯度提升方法对Word2Vec、GloVe和句子变换器表示进行了评估。实验结果发现验证性能与测试性能之间存在显著差距。尽管验证指标表现强劲，但模型的表现不及简单的基线方法。分析表明，当数据量低于临界的数据充分性阈值时，预训练嵌入的收益会递减。小型验证集会导致模型选择过程中出现过拟合。论文通过每周情感分析展示了实际应用。

    arXiv:2512.13749v3 Announce Type: replace-cross  Abstract: Financial sentiment analysis enhances market understanding. However, standard Natural Language Processing (NLP) approaches encounter significant challenges when applied to small datasets. This study presents a comparative evaluation of embedding-based techniques for financial news sentiment classification in resource-constrained environments. Word2Vec, GloVe, and sentence transformer representations are evaluated in combination with gradient boosting on a manually labeled dataset of 349 financial news headlines. Experimental results identify a substantial gap between validation and test performance. Despite strong validation metrics, models underperform relative to trivial baselines. The analysis indicates that pretrained embeddings yield diminishing returns below a critical data sufficiency threshold. Small validation sets contribute to overfitting during model selection. Practical application is illustrated through weekly sen
    
[^103]: AI生成代码中的缺陷综述

    A Survey of Bugs in AI-Generated Code

    [https://arxiv.org/abs/2512.05239](https://arxiv.org/abs/2512.05239)

    本文通过系统性文献综述，首次全面梳理了AI生成代码中缺陷的类型、模式、分布及其与特定AI编程系统的关联和相应的修复策略，揭示了缺陷在AI生成代码中的普遍性。

    

    AI编程系统正日益被用于帮助开发者大规模编写代码，从而提升软件开发效率并提高生产力。尽管这些模型和智能体已展现出强大的性能，但人们对其生成代码的质量仍存有担忧。虽然已有研究报道了AI生成代码的一系列质量问题，但这些发现仍然零散，缺乏系统性的归纳整理。目前尚缺乏一项全面的综述来展示缺陷的类型、模式、分布、其与特定AI编程系统的关联，以及针对这些问题所提出的修复策略。为填补这一空白，我们对现有的AI生成代码相关研究开展了系统性文献综述，以全面理解生成代码中的缺陷，并为未来AI辅助编程的改进奠定基础。我们的结果表明，缺陷在AI生成的代码中普遍存在。

    arXiv:2512.05239v2 Announce Type: replace-cross  Abstract: AI coding systems are increasingly used to help developers write code at scale, improving software development efficiency and boosting productivity. While these models and agents have shown strong performance, concerns persist regarding the quality of the AI-generated code. Although prior work has reported a range of quality issues associated with AI-generated code, these findings remain scattered and lack a systematic synthesis. A comprehensive review is lacking that demonstrates the bug types, patterns, distribution, their correlation with specific AI coding systems, and the remediation strategies proposed to address them. To fill this gap, we conduct a systematic literature review of existing studies on AI-generated code to develop a comprehensive understanding of bugs in the generated code and provide a foundation for future improvements in AI-assisted coding. Our results show that bugs are widespread in AI-generated code, 
    
[^104]: 超越合成基准：评估大语言模型在真实世界类级别代码生成上的性能

    Beyond Synthetic Benchmarks: Evaluating LLM Performance on Real-World Class-Level Code Generation

    [https://arxiv.org/abs/2510.26130](https://arxiv.org/abs/2510.26130)

    该论文提出了基于真实开源仓库构建的RealClassEval基准，用于评估大语言模型在类级别代码生成上的真实能力，揭示了此前被合成基准所掩盖的模型根本性局限。

    

    大语言模型（LLMs）已经从研究充分的函数级代码生成快速发展到新兴的仓库级方法，然而这一进展揭示了一个关键缺口：类级别代码生成——作为连接单个函数与完整仓库的基本构建块——仍然被系统性地探索不足。理解类级别能力是仓库级任务成功的前提，因为当前方法无法区分失败究竟源于基本的实现缺陷还是复杂的上下文管理问题。我们提出了RealClassEval，这是一个源自真实世界开源仓库的基准，包含400个类，并根据模型训练截止时间划分为已见和未见两个分区。我们对七个最先进LLMs的系统评估揭示了被合成基准所掩盖的根本性局限：虽然LLMs在合成基准上能够达到84%至89%的正确率……

    arXiv:2510.26130v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have advanced rapidly from well-studied function-level code generation to emerging repository-level approaches, yet this progression reveals a critical gap: class-level code generation, which is the fundamental building block connecting individual functions to complete repositories, remains systematically underexplored. Understanding class-level capabilities is a prerequisite to repository-level success, as current approaches cannot distinguish whether failures stem from basic implementation deficiencies or sophisticated context management issues.   We introduce RealClassEval, a benchmark derived from real-world open-source repositories, comprising 400 classes divided into seen and unseen partitions based on model training cutoffs. Our systematic evaluation across seven state-of-the-art LLMs reveals fundamental limitations masked by synthetic benchmarks: while LLMs achieve 84 to 89% correctness on s
    
[^105]: EmpathiSEr（共情者）：面向软件工程的共情量表的开发与验证

    The EmpathiSEr: Development and Validation of Software Engineering Oriented Empathy Scales

    [https://arxiv.org/abs/2510.12546](https://arxiv.org/abs/2510.12546)

    该研究开发并验证了首个专门面向软件工程情境的共情测量量表EmpathiSEr，填补了软件工程领域缺乏有效共情测量工具的空白。

    

    共情在软件工程（SE）中扮演着至关重要的角色，例如在开发者理解非技术用户对系统可用性的挫败感时，或在产品负责人考虑工程师在实现过程中所面临的技术约束时。这类互动塑造了协作、沟通以及以用户为中心的设计成果。尽管软件工程研究日益将共情视为一个关键的人文因素，但目前仍然没有专门针对软件工程独特的社会技术情境设计的、经过验证的测量工具。现有的通用共情量表虽然在心理学和医疗保健领域已经相当成熟，但其依赖的语言、情境和假设往往对软件从业者来说缺乏意义或难以理解。这些量表未能考虑到软件工程中共情的多样化、角色特定性和领域限定的表达方式，例如理解非技术用户的挫败感或……

    arXiv:2510.12546v2 Announce Type: replace  Abstract: Empathy plays a critical role in software engineering (SE), for instance in situations where developers interpret non-technical users' frustration with system usability or where product owners account for the technical constraints experienced by engineers during implementation. Such interactions shape collaboration, communication, and user-centred design outcomes. Although SE research has increasingly recognised empathy as a key human aspect, there remains no validated instrument specifically designed to measure it within the unique socio-technical contexts of SE. Existing generic empathy scales, while well-established in psychology and healthcare, often rely on language, scenarios, and assumptions that are not meaningful or interpretable for software practitioners. These scales fail to account for the diverse, role-specific, and domain-bound expressions of empathy in SE, such as understanding a non-technical user's frustrations or a
    
[^106]: 糟糕！……我又重蹈覆辙了：分析并处理社会技术软件工程中结论的（不）稳定性

    Oops!... I did it again. Analysing and Handling Conclusion (In-)Stability in Socio-Technical Software Engineering

    [https://arxiv.org/abs/2510.06844](https://arxiv.org/abs/2510.06844)

    本研究通过用四个独立挖掘工具正式复现三项已发表的社会技术软件分析研究，揭示了不同工具流水线在数据、结果与结论上的不稳定性，并为研究者和从业者提供了应对此类效度威胁的可操作建议。

    

    背景：软件仓库挖掘是一种常用手段，可用于洞察软件项目的演进、监控项目健康状态、支持决策并总结最佳实践。支持挖掘流程的工具被研究者和从业者广泛使用，但人们对这些工具的局限性以及它们之间的一致性往往缺乏充分理解。目的：本研究考察了用于演进式社会技术软件分析的复杂工具流水线中的一些效度威胁，并针对相同的研究问题评估各工具在数据、研究结果和结论层面的一致性，从而为研究者和从业者提出可操作的建议。方法：我们开展了一项轻量级文献综述，从高水平学术会议和期刊中选取三项关于协作与协调、软件维护及软件质量的研究，并使用四个独立、系统挑选的挖掘工具对其进行正式复现，以量化……（原文摘要至此截断）

    arXiv:2510.06844v3 Announce Type: replace  Abstract: Context: Mining software repositories is a popular means to gain insights into a software project's evolution, monitor project health, support decisions and derive best practices. Tools supporting the mining process are commonly applied by researchers and practitioners, but their limitations and agreement are often not well understood.   Objective: This study investigates some threats to validity in complex tool pipelines for evolutionary socio-technical software analyses. We evaluate the tools' agreement in terms of data, study outcomes and conclusions for the same research questions to derive actionable advice for researchers and practitioners.   Method: We conduct a lightweight literature review to select \emph{three} studies on collaboration and coordination, software maintenance and software quality from high-ranked venues, which we formally replicate with \emph{four} independent, systematically selected mining tools to quantita
    
[^107]: UniSage：面向微服务系统的统一且诊断驱动的采样框架

    UniSage: A Unified and Diagnosis-Driven Sampling Framework for Microservice Systems

    [https://arxiv.org/abs/2509.26336](https://arxiv.org/abs/2509.26336)

    UniSage是一个面向微服务系统的统一且诊断驱动的采样框架，它通过将轻量级分析集成到采样流水线中指导数据保留决策，并统一处理追踪与日志的采样，从而避免“先采样后分析”范式丢失关键诊断证据、以及追踪与日志上下文关联被破坏的问题。

    

    追踪数据与日志是诊断微服务故障的重要可观测性信号，但其庞大的数据量使得完整保留的代价过于高昂。因此，采样被广泛用于降低存储和分析开销。然而，现有的采样框架存在两个关键局限。首先，它们通常遵循“先采样后分析”的范式，即在数据的诊断价值尚不明确时就将其丢弃，从而可能丢失对下游分析至关重要的证据。其次，它们主要关注追踪数据的采样，而日志则被独立地压缩或过滤，这可能破坏异常追踪与其解释性日志消息之间的上下文关联。为解决这些局限，我们提出了UniSage，一个面向微服务系统的统一且诊断驱动的采样框架。UniSage的“统一”体现在两个方面：它将轻量级分析集成到采样流水线中，以指导数据保留决策（原文在此处截断）。

    arXiv:2509.26336v3 Announce Type: replace  Abstract: Traces and logs are essential observability signals for diagnosing microservice failures, but their massive volume makes full retention prohibitively expensive. Sampling is therefore widely used to reduce storage and analysis overhead. However, existing sampling frameworks suffer from two key limitations. First, they typically follow a sample-before-analysis paradigm, where data is discarded before its diagnostic value is known, risking the loss of critical evidence for downstream analysis. Second, they mainly focus on trace sampling, while logs are either compressed or filtered independently, which can break the contextual linkage between anomalous traces and their explanatory log messages.   To address these limitations, we propose UniSage, a unified and diagnosis-driven sampling framework for microservice systems. UniSage is unified in two senses: it integrates lightweight analysis into the sampling pipeline to guide retention dec
    
[^108]: OpenClassGen：面向大语言模型研究的大规模真实世界Python类语料库

    OpenClassGen: A Large-Scale Corpus of Real-World Python Classes for LLM Research

    [https://arxiv.org/abs/2504.15564](https://arxiv.org/abs/2504.15564)

    OpenClassGen是一个从约3,000个开源项目中提取的包含32万余个真实Python类的大规模语料库，每个类均配有自包含的骨架和27项静态代码指标，为类级别代码生成的训练与评估提供了迄今最大规模的资源。

    

    现有的类级别代码生成数据集要么是合成的（ClassEval：100个类），要么规模不足以满足现代训练需求（RealClassEval：400个类），这阻碍了稳健的评估和实证分析。我们提出了OpenClassGen，这是一个从2,970个工程化开源项目中提取的大规模语料库，包含324,843个Python类。每个条目将一个人工编写的类与其对应的骨架配对，骨架包含类和方法签名及相关的文档字符串，并附带27个涵盖复杂性、耦合度、内聚性和继承属性的静态代码指标。与先前需要仓库级上下文解析的基准不同，OpenClassGen提供自包含的类骨架，可作为完整的生成规范。我们通过在一个精选的、可执行的300个类子集上评估三个大语言模型（GPT-o4-mini、Claude-4-Sonnet、Qwen-3-Coder），展示了该语料库的实用性……

    arXiv:2504.15564v4 Announce Type: replace-cross  Abstract: Existing class-level code generation datasets are either synthetic (ClassEval: 100 classes) or insufficient in scale for modern training needs (RealClassEval: 400 classes), hindering robust evaluation and empirical analysis. We present OpenClassGen, a large-scale corpus of 324,843 Python classes extracted from 2,970 engineered open-source projects. Each entry pairs a human-written class with its corresponding skeleton, which comprises class and method signatures with associated docstrings, and is enriched with 27 static code metrics covering complexity, coupling, cohesion, and inheritance properties. Unlike prior benchmarks that require repository-level context resolution, OpenClassGen provides self-contained class skeletons that serve as complete generation specifications. We demonstrate the corpus's utility by evaluating three LLMs (GPT-o4-mini, Claude-4-Sonnet, Qwen-3-Coder) on a curated, executable subset of 300 classes, en
    

