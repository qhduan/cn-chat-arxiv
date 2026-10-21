# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Industrial ReMEmbR: Scaling Spatiotemporal Memory to Robot Fleets](https://arxiv.org/abs/2610.28624) | 该论文提出了IndustrialRemembr共享时空记忆系统，将NVIDIA ReMEmbR从单一机器人扩展到机器人车队，通过在机器人端本地编码视频（检索嵌入、视觉令牌与重建潜变量）并仅按需生成场景描述，同时抹除个人身份信息以符合GDPR，从而大幅降低服务器端的计算、通信与存储成本。 |
| [^2] | [TraceLite: Efficient Trace Reduction for SWE Agents via Optical Compression and Redundancy Removal](https://arxiv.org/abs/2610.28612) | TraceLite是一个无需训练的SWE智能体轨迹压缩框架，它将近期观测保留为文本、把历史观测光学渲染为紧凑图像、消除重叠与过期冗余内容，从而在保留关键信息的同时大幅降低token消耗与推理开销。 |
| [^3] | [InterWebCoder: Learning Interactive Webpage Generation from Executable Tests via Reinforcement Learning](https://arxiv.org/abs/2610.28610) | 提出InterWebCoder框架，通过测试规范引导与执行验证的数据合成流水线生成带可执行测试脚本的高质量交互式网页训练数据，并利用强化学习从可执行测试中学习交互式网页代码生成。 |
| [^4] | [Object-Oriented Harness: Evolving Agent Harnesses through Object-Oriented Design](https://arxiv.org/abs/2610.28380) | 提出OO-Harness框架，利用面向对象设计中的继承机制演化出专门化执行框架的类层次结构，并通过路由器为不同任务选择最优框架，在三个长程基准测试中相比基线最高提升15%。 |
| [^5] | [Quantum Broker: QoS-Aware Distribution of Quantum Computations](https://arxiv.org/abs/2610.28342) | 提出了开源且与提供商无关的 Quantum Broker 框架，利用量子测量射击的统计独立性，根据用户定义的服务质量约束和优化目标，将单个量子电路按射击分发到多个异构 QPU 上执行。 |
| [^6] | [A Multimodal Engineering GraphRAG Framework for Post-Wind-Event Building Assessment and Recovery Inference](https://arxiv.org/abs/2610.28216) | 该研究提出了一种多模态工程GraphRAG框架，将风灾后分散的图像与文本损伤证据转化为保留来源的组件级表示，并将其与物理后果、建筑状态、恢复行动和停用时间相关联，从而支持灾后建筑评估与恢复推理。 |
| [^7] | [ExeBench-FHD: Addressing Fidelity Issues in Decompiler Evaluation](https://arxiv.org/abs/2610.28212) | 该论文分析了广泛使用的反编译器评测基准ExeBench中影响评估保真度的九个问题，并提出了向后兼容的修订版ExeBench-FHD，通过新增16个字段更忠实地评估基于LLM的反编译器。 |
| [^8] | [Foreword: Pre-Generation Prediction and Characterisation of LLM Test Failures using the Jacobian Lens](https://arxiv.org/abs/2610.28187) | 提出Foreword框架，利用雅可比透镜将LLM中间激活读取为词汇标记分数来表征输入，并结合通过与失败参考样本及稀疏逻辑回归探针或双高斯混合模型，实现测试失败的生成前预测与失败原因表征。 |
| [^9] | [From Code Generator to Engineering Companion: A Plant-aware Agent for PLC Programming](https://arxiv.org/abs/2610.28160) | 该论文提出了一个工厂感知的对话式PLC编程智能体，通过工厂特定上下文摄取、以对话为先的解释与调试交互模式，以及从聊天到部署的分级工具访问阶梯，将PLC编程从一次性代码生成升级为持续的工程辅助体验。 |
| [^10] | [AgenTest: Mining Business Rules to Test LLM Agents for Business Logic Bugs](https://arxiv.org/abs/2610.28116) | AgenTest通过从系统提示词、工具定义和领域知识中自动挖掘业务规则，系统性地测试LLM智能体，从而发现现有测试方法难以检测的业务逻辑缺陷。 |
| [^11] | [Training the Whole Agent, Not Just the Model](https://arxiv.org/abs/2610.28069) | HarnessAdapt-RL 首次提出在强化学习中同时训练策略并自适应调整其外部框架，利用教师模型生成机制级技能并演化工具与钩子，使 9B 模型在 SWE-bench 基准上比 GRPO 最高提升 12.6 个百分点的 pass@1。 |
| [^12] | [From RL Papers to Framework Code: Inclusion and Review in Seven Open-Source Training Frameworks](https://arxiv.org/abs/2610.28027) | 该研究通过追踪七个开源强化学习后训练框架的代码提交历史，发现1598篇提出通用RL训练方法的论文中仅有37篇被框架收录，且收录的论文多为大规模训练（30B+参数）和现有损失函数的小幅修改，揭示了RL研究成果从论文到实际工程应用之间的巨大鸿沟。 |
| [^13] | [LiveUIBench: Evaluating Agent as App Workflows](https://arxiv.org/abs/2610.27977) | 提出Agent as App任务和LiveUIAgent系统，通过生成连接界面、信息与操作的编排并复用共享的智能体运行时，实现可随用户需求演变而轻量级个性化的应用。 |
| [^14] | [How Developers Configure LLM Inference: A Large-Scale Study of Real-World Practices](https://arxiv.org/abs/2610.27945) | 该论文通过对 5,398 个 GitHub 仓库的大规模实证研究，系统揭示了开发者在使用 Hugging Face Transformers、vLLM 和 OpenAI API 时的真实 LLM 推理配置实践，发现开发者实际常规使用的配置选项仅占可用配置空间的一小部分。 |
| [^15] | [Benchmarking CPU, GPU, and ML Power-Flow Solvers from 14 to 70,000 Buses](https://arxiv.org/abs/2610.27930) | 该论文构建了涵盖9个物理求解器和3个机器学习替代模型的潮流计算基准，揭示基线选择可令速度结论相差2.5-3个数量级，并证明充分并行化的CPU求解器或新的GPU求解器gpusim2grid在吞吐量上可比肩甚至超过ML替代模型，且精度高出数个数量级。 |
| [^16] | [A Neuro-Symbolic Reasoning Engine for Compilation-Free Patch Presence Testing](https://arxiv.org/abs/2610.27871) | 该论文提出SyLares，一种深度融合大语言模型与SMT求解器的神经符号推理引擎，克服了此前Lares框架中两通道并行、LLM充当最终仲裁者导致的幻觉问题，实现了更可靠的免编译补丁存在性检测。 |
| [^17] | [Hallucinations of Large Language Models in Cross-Language Code Translation: Phenomena and Mitigation](https://arxiv.org/abs/2610.27868) | 该论文对大语言模型在跨语言代码翻译中的幻觉现象开展了系统性实证研究，在三个基准数据集、六个语言对和七个大语言模型上构建了幻觉分类体系，并提出相应的缓解方法。 |
| [^18] | [From Tokens to Files: Multi-Granularity Detection and Localization of AI-Involved Code](https://arxiv.org/abs/2610.27863) | 提出了一种基于冻结代码语言模型上双低秩适配器逐词元对数似然比的框架，可在词元证据基础上对 AI 参与代码进行行、块、文件多粒度的检测与定位。 |
| [^19] | [Value-level data provenance in an electronic laboratory notebook used by humans and language-model agents: An experience report](https://arxiv.org/abs/2610.27859) | 本文报告了电子实验笔记本APS ELN的值级数据溯源架构经验：针对近5年审计中发现的“从原始文件到存储数值”关联丢失问题，通过重构数据库和工作流程，使每个存储数值都记录其源文件、提取插件版本及派生公式，从而实现数值的可验证性。 |
| [^20] | [Raise or Check? Predicting the Necessity of Additional Fuzzing for Continuous Fuzzing](https://arxiv.org/abs/2610.27828) | 提出RAISE即时框架，通过在项目-日粒度上预测漏洞引入风险，优先为高风险项目分配额外模糊测试资源，从而高效利用计算资源并缩小持续模糊测试中的漏洞暴露窗口期。 |
| [^21] | [MIMOS-Tools: An Integrated Toolchain for Model-Based Design of Embedded Systems](https://arxiv.org/abs/2610.27826) | 该论文提出了MIMOS-Tools集成工具链，基于具有形式化操作语义和功能与时间确定性的MIMOS计算模型，为嵌入式实时系统提供从建模、仿真、形式化验证到异构多核平台代码生成的全生命周期一体化支持。 |
| [^22] | [RAG-NIMBF: A Non-Invasive, Modular Benchmarking Framework for Automated RAG Configuration Analysis](https://arxiv.org/abs/2610.27816) | 该论文提出了RAG-NIMBF，一个可作为非侵入式评估层附加到现有RAG管道的模块化基准测试框架，通过受控消融实验与端到端实验跟踪，结合LLM答案质量评分、检索正确性和运行时指标，自动分析RAG庞大配置空间中各组件间的非显而易见的交互作用。 |
| [^23] | [RELAY: Scaling Historical Build Restoration through Rule Reuse and Agentic Repair](https://arxiv.org/abs/2610.27784) | Relay是一个混合框架，它利用LLM从代表性构建失败中自动推导并验证可复用的确定性修复规则，将规则无法解决的少数案例交给智能体逐案修复，从而规模化地恢复了历史软件（14,159个OSS-Fuzz案例中近七成因环境衰退而无法重建）的构建。 |
| [^24] | [Enhancing Bug-Exposing Test Generation via Step-Level Credit Assignment and Self-Distillation from Valuable Failures](https://arxiv.org/abs/2610.27694) | SCALF提出了一种过程监督强化学习方法，通过步骤级信用分配让每个推理步骤获得独立的学习信号，并从找到触发缺陷输入但预测错误输出的有价值失败轨迹中进行自蒸馏学习，从而显著增强了暴露Bug的测试生成效果。 |
| [^25] | [Knowledge as Code](https://arxiv.org/abs/2610.27693) | 该论文从认识论角度提出“知识即代码”框架，定义了十三种知识载体属性并证明代码是首个同时具备全部属性的载体，同时区分了编码与通过执行学习两种知识获取方式。 |
| [^26] | [Reusing Assurance Evidence in AI Pipelines: Validity Conditions, a Fail-Closed Release Gate, and a Replay of 714 Public Data Version Control Pipelines](https://arxiv.org/abs/2610.27672) | 该论文为AI管道中保证证据的复用建立了有效性条件，提出基于Ed25519签名的故障关闭式发布门控（可阻止9种操纵场景中的8种），并通过对714个公开数据版本控制管道的38,083次提交进行重放，量化了依赖声明在实践中失效的情况。 |
| [^27] | [Standards-Guided and Dependency-Aware Code-Level Documentation Generation](https://arxiv.org/abs/2610.27634) | 提出CODEDOC框架，将代码级文档生成重塑为一项基于权威质量标准的标准引导软件工程任务，并借助静态调用图与AST分析构建依赖感知的结构化提示。 |
| [^28] | [Auditing the Product, Not the Recursion: The Evaluation Evidence Behind Recursive Self-Improvement Claims](https://arxiv.org/abs/2610.27610) | 该论文首次通过对114项递归自我改进（RSI）研究的系统性全文编码，量化审计了RSI声明背后产品评估与流程评估证据的实际报告频率，揭示了递归自我改进声明与其实际评估证据之间的差距。 |
| [^29] | [Recoverability and Verifier Strength: Two Measured Preconditions for Unattended Agent Self-Improvement](https://arxiv.org/abs/2610.27465) | 论文提出了“验证器强度”和“可恢复性”这两个可测量的前置条件，用于判断智能体平台的哪些自我改进步骤可以安全地无人值守运行。 |
| [^30] | [AgenticVulHunter: Bringing Local Agentic Vulnerability Hunting to Your Terminal](https://arxiv.org/abs/2610.27383) | 本文提出AgenticVulHunter，一个可在本地部署的开源权重LLM上运行的智能体安全代码审查工具，通过聚焦且相关的安全知识实现终端漏洞挖掘。 |
| [^31] | [Documentation Drift in AI-Generated Pull Requests](https://arxiv.org/abs/2610.27381) | 该研究首次大规模刻画了AI生成代码的文档编写实践，发现63.2%的AI拉取请求在修改代码时不更新任何文档，且无文档的代码文件后续被修复的概率显著更高，揭示了“文档漂移”现象及其对软件质量的负面影响。 |
| [^32] | [Characterizing Post-Incident Software Evolution](https://arxiv.org/abs/2610.27380) | 本研究首次对软件事故后的演化进行了实证特征刻画，通过分析维基媒体10年间1,302个任务和2,203个代码审查的混合方法研究，揭示了事故引发的变化在工作量、位置和时机等方面与标准维护变化的差异。 |
| [^33] | [Stop Paying by the Letter: Accelerating Full-Chip RTL Simulation with Host-Side Tokenized Logging](https://arxiv.org/abs/2610.27356) | 提出一种面向异构多核SoC的主机侧仿真内令牌化日志架构，将printf的格式化与MMIO串行化开销转移至主机端并以零仿真时间重建字符串，从而同时消除MMIO总线停顿与片上vsprintf格式化两大瓶颈，大幅加速全芯片RTL仿真。 |
| [^34] | [XRust: Detecting Logical Races in Rust Linux Drivers](https://arxiv.org/abs/2610.27325) | XRust无需人工指定不变量，通过逻辑先于关系分析自动构造操作对测试基座并使用串行差分预言机，实现Linux Rust驱动程序中逻辑竞争的自动检测。 |
| [^35] | [A Hybrid Information Flow Analysis Framework for Android Hybrid Apps](https://arxiv.org/abs/2610.27288) | MifDroid是一个静态分析引导动态分析的混合框架，通过生命周期感知的WebView调用图、桥接感知的类型-效果系统和模块化JavaScript分析引擎，解决了Android混合应用中Java与JavaScript之间跨语言信息流无法被单语言分析器检测的问题。 |
| [^36] | [Certifying Repeatable Entry-Replay Witnesses for COBOL-to-Java Migration](https://arxiv.org/abs/2610.27272) | 该论文提出基于按时间顺序外部操作带的进入-回放测试用例认证方法，在模拟外部交互的代理COBOL程序上生成合成端到端测试用例，使一个33,429行的大型COBOL程序达到78.1%的分支覆盖率和95.7%的段落覆盖率，从而支持COBOL到Java迁移的确定性验证。 |
| [^37] | [InterType: Breaking the Single-Function Barrier in Neural Binary Type Inference](https://arxiv.org/abs/2610.26906) | InterType将过程间分析嵌入预训练的Transformer模型中，通过利用跨函数传播的类型约束信息，打破了以往神经二进制类型推断局限于单函数内部的壁垒。 |
| [^38] | [Learning from Past Mistakes: AI-Assisted Analytics of Software Defects and Their Fixes for Future Defect Prevention - An Industrial Case Study](https://arxiv.org/abs/2610.26889) | 该论文提出一种AI辅助、人在回路的缺陷分析方法，通过分析历史缺陷及其修复数据识别重复缺陷的人为错误根源，并将其转化为预防措施和面向工程师的能力建设洞察，从而实现未来的缺陷预防与质量保证改进。 |
| [^39] | [The Complexity Kink: A Prompt-Side Structural Complexity Index for Code-Generation Reliability](https://arxiv.org/abs/2609.19616) | 该论文提出一个生成前评分的六维提示侧结构复杂度指数，发现代码生成通过率在复杂度分数上存在非单调断点，但该断点并非通用的失败临界值，会随任务类型固定效应等因素发生移动。 |
| [^40] | [When Models Edit Too Much: On the Fidelity of Minimal Code Edits](https://arxiv.org/abs/2609.04061) | 该研究揭示了前沿大语言模型在修复代码时普遍存在“过度编辑”问题（即使如GPT-5.5这样的强模型也不例外），并提出通过一条简单的保留指令即可显著减少不必要的代码改动、降低认知复杂度，同时提升修复准确率。 |
| [^41] | [For Coding Agents, the Repository Matters](https://arxiv.org/abs/2606.28235) | 该研究发现编程智能体拉取请求的解决时长等结果在很大程度上取决于接收仓库本身的特性，且这种依赖程度甚至超过人类贡献，因此跨仓库汇总的指标无法公平地评判智能体表现。 |
| [^42] | [Unpacking Security Scanners for GitHub Actions Workflows](https://arxiv.org/abs/2601.14455) | 本文首次对9个GitHub Actions工作流安全扫描器进行了系统性比较，从范围、检测能力、准确性和性能四个维度评估了它们的安全检测效果。 |
| [^43] | [Large Language Models for Agent-Based Modelling: Current and possible uses across the modelling cycle](https://arxiv.org/abs/2507.05723) | 本研究回顾了大语言模型在基于主体建模中的现有应用，并沿着从问题构建到结果传播的完整建模周期，批判性地分析了其潜在应用的机遇与挑战。 |
| [^44] | [LLMs are Brittle to Simple Code Transformations: Introducing CETBench - A Benchmark for Code-Equivalence Checking](https://arxiv.org/abs/2506.04019) | 本文提出了代码等价性检测基准CETBench，发现大语言模型即使面对简单的代码变换也会出现显著性能下降（在跨语言场景下尤为严重），并提出一种基于微调的简单方法来提升模型性能。 |
| [^45] | [Can Language Models Replace Programmers for Coding? REPOCOD Says 'Not Yet'](https://arxiv.org/abs/2410.21647) | 该论文提出了REPOCOD基准，包含来自11个真实热门项目的980个Python完整函数生成任务，以评估语言模型在真实世界编码任务中的能力，结果显示当前语言模型尚不能替代程序员。 |
| [^46] | [WAFFLE: Finetuning Multi-Modal Models for Automated Front-End Development](https://arxiv.org/abs/2410.18362) | Waffle提出了一种结合结构感知注意力机制与对比微调的多模态模型微调策略，显著提升了从UI设计图自动生成HTML代码的效果。 |
| [^47] | [Mitigating Risks in Software Development through Effective Requirements Engineering](https://arxiv.org/abs/2305.05800) | 本文通过综述需求获取、分析与协商、规格说明、验证和变更管理等核心需求工程活动，并比较不同开发生命周期对需求的处理方式，说明如何利用有效的需求工程来降低软件项目中因需求缺陷导致的延期、超支和项目取消等风险。 |

# 详细

[^1]: 工业ReMEmbR：将时空记忆扩展至机器人车队

    Industrial ReMEmbR: Scaling Spatiotemporal Memory to Robot Fleets

    [https://arxiv.org/abs/2610.28624](https://arxiv.org/abs/2610.28624)

    该论文提出了IndustrialRemembr共享时空记忆系统，将NVIDIA ReMEmbR从单一机器人扩展到机器人车队，通过在机器人端本地编码视频（检索嵌入、视觉令牌与重建潜变量）并仅按需生成场景描述，同时抹除个人身份信息以符合GDPR，从而大幅降低服务器端的计算、通信与存储成本。

    

    仓库机器人车队持续不断地观察运营状况，但将这些观察保存为可复用的语义记忆会产生巨大的计算、通信和存储成本。例如，要解释某条通道反复被堵塞的原因，可能需要查看多个机器人在多个班次中的视频。我们提出了IndustrialRemembr，这是一个共享时空记忆系统，它将NVIDIA ReMEmbR扩展到机器人车队，并通过仅按需生成场景描述来降低服务器端计算量。在每个机器人上，NVIDIA Jetson使用Cosmos-Embed1将视频片段编码为检索嵌入向量，使用Cosmos3-Edge编码为视觉令牌，并使用Wan2.2 VAE编码为重建潜变量，而无需上传原始相机帧。这些编码模型还被复用于检测图像块中的人物并将其置零，以支持符合GDPR合规部署要求的个人身份信息（PII）抽象化处理。我们在视觉-语言边界处对Cosmos3-Edge进行拆分，将视觉编码部分……

    arXiv:2610.28624v1 Announce Type: cross  Abstract: Warehouse robot fleets continuously observe operational conditions, but preserving these observations as reusable semantic memory incurs substantial computation, communication, and storage costs. For example, explaining recurring aisle blockages can require reviewing video from multiple robots across several shifts. We present IndustrialRemembr, a shared spatiotemporal memory system that extends NVIDIA ReMEmbR to robot fleets and reduces server-side computation by generating scene descriptions only on demand. On each robot, an NVIDIA Jetson encodes video clips into retrieval embeddings with Cosmos-Embed1, vision tokens with Cosmos3-Edge, and reconstruction latents with the Wan2.2 VAE, without uploading raw camera frames. The encoding models are re-used to detect people in image patches, zeroing them out to support PII abstraction for GDPR-compliant deployments. We split Cosmos3-Edge at its vision-language boundary, placing the vision e
    
[^2]: TraceLite：通过光学压缩与冗余消除实现软件工程智能体的高效轨迹缩减

    TraceLite: Efficient Trace Reduction for SWE Agents via Optical Compression and Redundancy Removal

    [https://arxiv.org/abs/2610.28612](https://arxiv.org/abs/2610.28612)

    TraceLite是一个无需训练的SWE智能体轨迹压缩框架，它将近期观测保留为文本、把历史观测光学渲染为紧凑图像、消除重叠与过期冗余内容，从而在保留关键信息的同时大幅降低token消耗与推理开销。

    

    软件工程智能体通过迭代式工具调用解决仓库级任务，但其持续增长的交互轨迹会带来大量的token消耗和推理开销。现有的压缩方法通常需要模型训练、辅助LLM调用，或对源代码等特定内容的假设。我们提出了TraceLite，一个简单而有效的、无需训练的异构SWE智能体观测内容压缩框架。我们的初步研究表明，在适度压缩下光学渲染能够保留大量文本与语义信息，同时智能体轨迹存在跨步骤和阶段级别的冗余。受这些发现启发，TraceLite将近期观测保留为高保真文本，将历史观测渲染为紧凑的图像，移除重叠内容，并在代码编辑后丢弃已过期的阶段观测载荷。它既不需要离线训练，也无需辅助LLM调用。

    arXiv:2610.28612v1 Announce Type: new  Abstract: Software engineering agents solve repository-level tasks through iterative tool use, but their continuously growing interaction traces incur substantial token consumption and inference overhead. Existing compression approaches often require model training, auxiliary LLM calls, or assumptions about specific content such as source code. We present TraceLite, a simple yet effective, training-free framework for compressing heterogeneous SWE-agent observations. Our preliminary study shows that optical rendering preserves substantial textual and semantic information under moderate compression, while agent trajectories exhibit both cross-step and phase-level redundancy. Motivated by these findings, TraceLite preserves recent observations as high-fidelity text, renders historical observations as compact images, removes overlapping content, and discards phase-expired observation payloads after code editing. It requires neither offline training no
    
[^3]: InterWebCoder：通过强化学习从可执行测试中学习交互式网页生成

    InterWebCoder: Learning Interactive Webpage Generation from Executable Tests via Reinforcement Learning

    [https://arxiv.org/abs/2610.28610](https://arxiv.org/abs/2610.28610)

    提出InterWebCoder框架，通过测试规范引导与执行验证的数据合成流水线生成带可执行测试脚本的高质量交互式网页训练数据，并利用强化学习从可执行测试中学习交互式网页代码生成。

    

    交互式网页是现代应用的基础，但对于多模态大语言模型（MLLMs）而言，复现其视觉外观和多步骤行为仍然具有挑战性。可扩展的训练数据必须捕获用户操作、状态转移和预期结果，而真实世界的网页代码往往与无关脚本和外部依赖项纠缠在一起。此外，现有的基于GUI智能体的验证方法提供的功能反馈缓慢且可能不稳定。为解决这些挑战，我们提出了InterWebCoder，一个高效的交互式网页代码生成框架。InterWebCoder引入了测试规范引导与执行验证的数据合成流水线，该流水线记录真实世界的交互轨迹，并合成带有可执行测试脚本的自包含网页代码。失败归因修复机制利用运行时证据来定位并迭代修复网页代码中的错误……

    arXiv:2610.28610v1 Announce Type: new  Abstract: Interactive webpages underpin modern applications, but reproducing both their visual appearance and multi-step behavior remains challenging for multimodal large language models (MLLMs). Scalable training data must capture user actions, state transitions, and expected outcomes, while real-world webpage code is often entangled with irrelevant scripts and external dependencies. Moreover, existing GUI-agent-based verification provides slow and potentially unstable functional feedback. To address these challenges, we propose InterWebCoder, an efficient framework for interactive webpage code generation. InterWebCoder introduces a Test-Specification-Guided and Execution-Verified Data Synthesis pipeline that records real-world interaction trajectories and synthesizes self-contained webpage code with executable test scripts. A failureattributed repair mechanism uses runtime evidence to locate and iteratively fix errors in either the webpage code 
    
[^4]: 面向对象执行框架：通过面向对象设计演化智能体执行框架

    Object-Oriented Harness: Evolving Agent Harnesses through Object-Oriented Design

    [https://arxiv.org/abs/2610.28380](https://arxiv.org/abs/2610.28380)

    提出OO-Harness框架，利用面向对象设计中的继承机制演化出专门化执行框架的类层次结构，并通过路由器为不同任务选择最优框架，在三个长程基准测试中相比基线最高提升15%。

    

    智能体执行框架是围绕语言模型指定提示词、工具和控制流的可执行代码。单一的全局框架在处理需要不同执行策略的异构任务时可能表现不佳。在这项工作中，我们提出了面向对象执行框架（OO-Harness），这是一个利用面向对象设计演化出可执行框架类层次的框架优化方法。在训练过程中，一个演化智能体利用执行记录创建一系列专门化的框架，每个框架都通过继承实现为基类或现有专门化框架的子类。在测试时，一个框架路由器利用训练期间积累的路由经验为每个任务选择特定的框架。在涵盖业务工作流、工作区任务和工具使用的三个长程基准测试中，OO-Harness在对比方法中取得了最佳性能，相比基线实现了最高达15%的绝对提升。

    arXiv:2610.28380v1 Announce Type: new  Abstract: Agent harnesses are executable code that specifies the prompts, tools, and control flow around a language model. A single global harness can struggle with heterogeneous tasks that require different execution strategies. In this work, we introduce Object-Oriented Harness (OO-Harness), a harness optimization framework that evolves a hierarchy of executable harness classes using object-oriented design. During training, an evolution agent uses execution records to create a family of specialized harnesses, each implemented as a subclass of the base or an existing specialized harness via inheritance. At test time, a harness router uses routing lessons accumulated during training to select a specific harness for each task. Across three long-horizon benchmarks spanning business workflows, workspace tasks, and tool use, OO-Harness achieves the best performance among the compared methods, with an absolute improvement of up to 15% over the base. Ex
    
[^5]: 量子代理：服务质量感知的量子计算分发

    Quantum Broker: QoS-Aware Distribution of Quantum Computations

    [https://arxiv.org/abs/2610.28342](https://arxiv.org/abs/2610.28342)

    提出了开源且与提供商无关的 Quantum Broker 框架，利用量子测量射击的统计独立性，根据用户定义的服务质量约束和优化目标，将单个量子电路按射击分发到多个异构 QPU 上执行。

    

    云端可访问的量子计算使得异构量子处理单元（QPU）能够通过提供商平台获得，这些平台在量子比特技术、校准状态、排队行为和定价模型上各不相同。这种异构性使执行管理成为量子工程需要关注的问题：一个量子电路可能可以在多个后端上执行，但由此产生的成本、延迟、保真度和可重复性取决于后端的选择以及测量射击（shots）的分配方式。本文提出了量子代理，这是一个开源且与提供商无关的服务质量感知量子执行管理框架。该框架通过利用量子测量射击的统计独立性，根据用户定义的约束和优化目标，将单个电路分发到多个异构 QPU 上执行，从而实现了按射击粒度的分发。该框架结合了模块化执行架构、可互换的……（原文摘要在此处截断）

    arXiv:2610.28342v1 Announce Type: cross  Abstract: Cloud-accessible quantum computing has made heterogeneous Quantum Processing Units (QPUs) available through provider platforms that differ in qubit technology, calibration state, queueing behavior, and pricing model. This heterogeneity makes execution management a quantum-engineering concern: a quantum circuit may be executable on several backends, but the resulting cost, latency, fidelity, and reproducibility depend on backend selection and on how measurement shots are allocated. This paper presents the Quantum Broker, an open-source and provider-agnostic framework for Quality-of-Service-aware quantum execution management. The framework operationalizes shot-wise distribution by exploiting the statistical independence of quantum measurement shots to dispatch a single circuit across multiple heterogeneous QPUs according to user-defined constraints and optimization objectives. It combines a modular execution architecture, interchangeable
    
[^6]: 一种用于风灾后建筑评估与恢复推理的多模态工程GraphRAG框架

    A Multimodal Engineering GraphRAG Framework for Post-Wind-Event Building Assessment and Recovery Inference

    [https://arxiv.org/abs/2610.28216](https://arxiv.org/abs/2610.28216)

    该研究提出了一种多模态工程GraphRAG框架，将风灾后分散的图像与文本损伤证据转化为保留来源的组件级表示，并将其与物理后果、建筑状态、恢复行动和停用时间相关联，从而支持灾后建筑评估与恢复推理。

    

    事件后的建筑评估需要将观察到的物理损伤与其对安全性、可居住性、功能性和恢复方面的影响联系起来。然而，灾后证据往往是异构的、不完整的，并且分散在图像和踏勘报告之中，而恢复模型通常需要结构化的工程输入。本研究开发了一种多模态工程图检索增强生成（GraphRAG）框架，将风灾后的损伤证据与物理后果、建筑状态、恢复行动以及停用时间联系起来。基于图像和文本的证据被转化为一种保留来源信息的组件级表示，能够区分已确认的损伤、明确的负面证据、未解决的状况和临时措施。图对齐检索将可比的建筑场景与明确的工程关系相结合，同时利用兼容性、矛盾性、证据优先级等机制……

    arXiv:2610.28216v1 Announce Type: cross  Abstract: Post-event building assessment requires linking observed physical damage to its consequences for safety, habitability, functionality, and recovery. Yet post-disaster evidence is often heterogeneous, incomplete, and distributed across images and reconnaissance reports, while recovery models typically require structured engineering inputs. This study develops a multimodal engineering Graph Retrieval-Augmented Generation (GraphRAG) framework that connects post-wind-event damage evidence to physical consequences, building-condition states, recovery actions, and downtime. Image- and text-based evidence is transformed into a provenance-preserving, component-level representation distinguishing confirmed damage, explicit negative evidence, unresolved conditions, and temporary measures. Graph-aligned retrieval combines comparable building scenarios with explicit engineering relationships, while compatibility, contradiction, evidence-precedence,
    
[^7]: ExeBench-FHD：解决反编译器评估中的保真度问题

    ExeBench-FHD: Addressing Fidelity Issues in Decompiler Evaluation

    [https://arxiv.org/abs/2610.28212](https://arxiv.org/abs/2610.28212)

    该论文分析了广泛使用的反编译器评测基准ExeBench中影响评估保真度的九个问题，并提出了向后兼容的修订版ExeBench-FHD，通过新增16个字段更忠实地评估基于LLM的反编译器。

    

    ExeBench是评估基于大语言模型（LLM）的反编译器最广泛使用的基准之一。我们分析了ExeBench并识别出九个影响测试对反编译器测量保真度的问题。其中三个问题阻碍了LLM反编译器从ExeBench的函数中恢复C代码：整数和浮点类型混合时的参数顺序歧义、被优化汇编代码可能丢弃的未使用参数、以及存储在函数汇编代码之外的常量。另外六个问题影响测试有效性：缺少返回语句的非void函数、占位符类型、被优化器移除的static或inline函数、命名为main的函数、平凡简单的函数体以及被截断的定义。我们提出了ExeBench-FHD，这是ExeBench的一个向后兼容的修订版本，新增了16个字段：调用点汇编代码、数据段、测试包装器、链接二进制反汇编代码以及五个警告标志。在ExeBench中，38.9%的函数至少带有五个警告标志中的一个；剔除……（摘要在此处截断）

    arXiv:2610.28212v1 Announce Type: cross  Abstract: ExeBench is one of the most widely used benchmarks for evaluating LLM-based decompilers. We analyze ExeBench and identify nine issues that affect how faithfully a test measures a decompiler. Three of them prevent LLM decompilers from recovering C code from ExeBench's function. These are ambiguous parameter order when integer and floating-point types are mixed, unused parameters that optimized assembly may drop, and constants stored outside the function's assembly code. Six more affect test validity: non-void functions without a return, placeholder types, static or inline functions removed by the optimizer, functions named main, trivial bodies and truncated definitions. We present ExeBench-FHD, a backward-compatible revision of ExeBench that adds 16 new fields: call-site assembly, data sections, test wrappers, linked-binary disassembly and five warning flags. In ExeBench, 38.9% of functions have at least one of the five flags; dropping 
    
[^8]: Foreword：基于雅可比透镜的LLM测试失败的生成前预测与表征

    Foreword: Pre-Generation Prediction and Characterisation of LLM Test Failures using the Jacobian Lens

    [https://arxiv.org/abs/2610.28187](https://arxiv.org/abs/2610.28187)

    提出Foreword框架，利用雅可比透镜将LLM中间激活读取为词汇标记分数来表征输入，并结合通过与失败参考样本及稀疏逻辑回归探针或双高斯混合模型，实现测试失败的生成前预测与失败原因表征。

    

    测试基于大语言模型（LLM）的软件成本高昂：每个输入都需要多次采样响应，且每次响应都需要由昂贵的判定器（oracle）进行评估。此外，每次提示词或模型的变更都要求重新测试。生成前的充分性度量通过基于模型内部状态对输入按失败风险进行排序，从而降低这一成本。然而，当前最先进的方法Clotho仅对通过测试的参考输入进行建模，且其隐藏状态表示无法解释输入为何失败。我们提出了Foreword，一个建立在两个核心思想之上的框架：同时从通过和失败的参考样本中学习，并通过雅可比透镜表征输入——这是一种可解释性技术，将中间激活读取为词汇标记上的分数。Foreword将这些表示与两种估计器相结合：稀疏逻辑回归探针或双高斯混合模型，后者还能提供参考密度（用于标记参考样本未覆盖的输入）以及失败类别成分……（摘要原文在此处截断）

    arXiv:2610.28187v1 Announce Type: cross  Abstract: Testing LLM-based software is expensive: each input needs several sampled responses, each judged by a costly oracle. Further, every prompt or model change calls for re-testing. Pre-generation adequacy measures cut this cost by ranking inputs by failure risk based on the model's internal state. However, the state of the art, Clotho, models only passing reference inputs, and its hidden-state representation says nothing about why inputs fail. We present Foreword, a framework built on two ideas: learn from both passing and failing references, and represent inputs through the Jacobian lens (J-lens), an interpretability technique that reads intermediate activations out as scores over vocabulary tokens. Foreword pairs these representations with two estimators: a sparse logistic-regression probe or dual Gaussian Mixture Models (dual GMMs), which also yield a reference density (to flag inputs the references do not cover) and failing-class compo
    
[^9]: 从代码生成器到工程助手：面向PLC编程的工厂感知智能体

    From Code Generator to Engineering Companion: A Plant-aware Agent for PLC Programming

    [https://arxiv.org/abs/2610.28160](https://arxiv.org/abs/2610.28160)

    该论文提出了一个工厂感知的对话式PLC编程智能体，通过工厂特定上下文摄取、以对话为先的解释与调试交互模式，以及从聊天到部署的分级工具访问阶梯，将PLC编程从一次性代码生成升级为持续的工程辅助体验。

    

    可编程逻辑控制器（PLC）编程仍然是一项以IDE为中心的手动任务，工程师需要在文档、项目文件和代码编辑器之间频繁切换上下文。近期研究表明，检索增强的大语言模型能够以较高的编译通过率生成符合厂商规范的Structured Text代码，但这些系统将代码生成视为一次性流水线：工程师提交查询后获得代码，系统对特定工厂没有持久化的知识。我们提出了一个工厂感知的对话式编程助手，该系统在三个方向上扩展了这一范式：工厂特定上下文的摄取、以对话为先的解释与调试交互模式，以及从对话到部署的分级工具访问阶梯。该系统专为真实工厂基础设施设计，并确保工程师对每个会产生现实世界影响的步骤保持控制权。

    arXiv:2610.28160v1 Announce Type: cross  Abstract: Programmable Logic Controller (PLC) coding remains a manual, IDE-centric task that demands constant context switching between documentation, project files, and code editors. Recent work has demonstrated that retrieval-augmented large language models can generate vendor-compliant Structured Text at high compilation rates \cite{pred}, yet these systems treat code generation as a one-shot pipeline: the engineer submits a query and receives code, with no persistent knowledge of the specific plant. We present a plant-aware conversational copilot that extends this paradigm in three directions: plant-specific context ingestion, a chat-first interaction model for explanation and debugging, and a graduated tool-access ladder from chat to deployment. The system is designed for real plant infrastructure and keeps the engineer in control of each step that produces real-world effects.
    
[^10]: AgenTest：挖掘业务规则以测试LLM智能体的业务逻辑缺陷

    AgenTest: Mining Business Rules to Test LLM Agents for Business Logic Bugs

    [https://arxiv.org/abs/2610.28116](https://arxiv.org/abs/2610.28116)

    AgenTest通过从系统提示词、工具定义和领域知识中自动挖掘业务规则，系统性地测试LLM智能体，从而发现现有测试方法难以检测的业务逻辑缺陷。

    

    大型语言模型（LLM）智能体越来越多地通过工具执行业务操作，并直接与应用数据和服务进行交互。除了完成用户请求之外，这些智能体还必须遵守领域特定的业务规则，这些规则约束了何时允许执行操作以及哪些数据或资源可能受到影响。违反此类规则可能导致业务逻辑缺陷，即使任务表面上看起来已成功完成。检测这类缺陷非常困难，因为这些规则很少以明确、可测试的规范形式存在：它们一部分分散在系统提示词和工具定义中，另一部分则隐含在领域知识中。现有的智能体测试方法通常只评估任务完成情况、预定义策略或已知攻击场景，这限制了它们系统性发现先前未被明确规定的业务规则违规行为的能力。我们提出了AgenTest，它从系统提示词、工具定义和领域知识中挖掘业务规则（原摘要在此处截断）。

    arXiv:2610.28116v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly execute business operations through tools and interact directly with application data and services. Beyond completing user requests, these agents must obey domain-specific business rules that constrain when actions are permitted and what data or resources may be affected. Violating such rules can lead to business logic bugs, even when a task appears successful. Detecting these bugs is difficult because the rules are rarely available as explicit, testable specifications. They are distributed across system prompts and tool definitions, while others remain implicit in domain knowledge. Existing agent testing approaches typically evaluate task success, predefined policies, or known attack scenarios. This limits their ability to systematically uncover previously unspecified business-rule violations.   We present AgenTest, which mines business rules from system prompts, tool definitions, and domai
    
[^11]: 训练完整的智能体，而不仅仅是模型

    Training the Whole Agent, Not Just the Model

    [https://arxiv.org/abs/2610.28069](https://arxiv.org/abs/2610.28069)

    HarnessAdapt-RL 首次提出在强化学习中同时训练策略并自适应调整其外部框架，利用教师模型生成机制级技能并演化工具与钩子，使 9B 模型在 SWE-bench 基准上比 GRPO 最高提升 12.6 个百分点的 pass@1。

    

    训练编码智能体通常只更新策略，而将外部框架固定不变。在训练过程中，轨迹会暴露出有缺陷的修复策略和交互问题，这些问题因任务而异，并随着策略的学习而不断变化。我们提出了 HarnessAdapt-RL，它在强化学习过程中同时训练策略并自适应调整其外部框架。针对有缺陷的修复策略，由一个冻结的教师模型生成任务特定的机制级技能；针对在多个任务中反复出现的交互问题，由另一个独立的教师模型演化工具和钩子，只有当某项更改能提高观测到的通过率或可训练率、且不降低任何一方时才予以保留。策略在调整后的外部框架下进行训练，其新生成的轨迹又会进一步指导框架的调整。基于 9B 基础模型，HarnessAdapt-RL 在 SWE-bench Verified、Multilingual 和 Pro 上分别将 pass@1 比 GRPO 提升了 9.8、12.6 和 6.4 个百分点，并优于使用人工设计的 OpenHands 框架的 GRPO。

    arXiv:2610.28069v1 Announce Type: new  Abstract: Training coding agents typically updates the policy while keeping the harness fixed. During training, trajectories reveal flawed repair strategies and interaction problems that vary across tasks and change as the policy learns. We introduce HarnessAdapt-RL, which trains the policy and adapts its harness during reinforcement learning. For flawed repair strategies, a frozen teacher generates task-specific mechanism-level skills. For interaction problems recurring across tasks, a separate teacher model evolves tools and hooks, keeping each change only if it improves the observed pass rate or the observed trainable rate without lowering either. The policy trains with the adapted harness, and its new trajectories guide further adaptation. With a 9B base model, HarnessAdapt-RL improves pass@1 over GRPO by 9.8, 12.6, and 6.4 percentage points on SWE-bench Verified, Multilingual, and Pro, and outperforms GRPO with the human-designed OpenHands ha
    
[^12]: 从强化学习论文到框架代码：七个开源训练框架中的收录与评审

    From RL Papers to Framework Code: Inclusion and Review in Seven Open-Source Training Frameworks

    [https://arxiv.org/abs/2610.28027](https://arxiv.org/abs/2610.28027)

    该研究通过追踪七个开源强化学习后训练框架的代码提交历史，发现1598篇提出通用RL训练方法的论文中仅有37篇被框架收录，且收录的论文多为大规模训练（30B+参数）和现有损失函数的小幅修改，揭示了RL研究成果从论文到实际工程应用之间的巨大鸿沟。

    

    从业者通过训练框架对语言模型使用强化学习方法，因此框架收录的方法对用户而言仅需更改一次配置即可使用。我们测量了这种情况发生的频率。我们追踪了七个用于强化学习后训练的开源框架的提交历史，并将其与2025年1月至2026年6月的6,939篇筛选后的arXiv论文相关联。在1,598篇提出通用强化学习训练方法的论文中，仅有37篇进入至少一个框架。在五个季度论文批次中，180天内被收录的数量始终保持在5至7篇之间，而批次规模却从72篇增长到386篇。与同期的其他论文相比，被收录的论文更常训练300亿（30B）参数或更大的策略模型（41%对10%），并且更常在现有损失函数中修改系数、裁剪、掩码或权重（29%对9%）。即使排除维护这些框架的公司自己的论文，这两个差距依然存在。

    arXiv:2610.28027v1 Announce Type: cross  Abstract: Practitioners use reinforcement learning methods for language models through training frameworks, so a method that a framework includes is one configuration change away for its users. We measure how often this happens. We trace seven open-source frameworks for RL post-training through their commit histories and link them to 6,939 screened arXiv papers from January 2025 to June 2026. Of 1,598 papers that propose a general RL training method, 37 reached at least one framework. Across five quarterly cohorts of papers, the number included within 180 days stayed between 5 and 7 while the cohorts grew from 72 to 386 papers. Compared with papers from the same quarters, included papers more often trained policies of 30B parameters or more, at 41 against 10 percent, and more often changed a coefficient, clip, mask or weight in the existing loss, at 29 against 9 percent. Both gaps hold without papers from the companies that maintain the framewor
    
[^13]: LiveUIBench：评估智能体作为应用工作流

    LiveUIBench: Evaluating Agent as App Workflows

    [https://arxiv.org/abs/2610.27977](https://arxiv.org/abs/2610.27977)

    提出Agent as App任务和LiveUIAgent系统，通过生成连接界面、信息与操作的编排并复用共享的智能体运行时，实现可随用户需求演变而轻量级个性化的应用。

    

    应用为用户提供了完成日常任务的便捷功能。然而，不同用户对同一个应用可能有不同甚至相互冲突的需求，而同一用户的需求也可能随时间发生变化。对由agent-to-app或UI-to-code方法生成的应用进行个性化，往往需要修改其生成的实现。我们提出了Agent as App（智能体即应用）这一任务：用户用自然语言表达需求，智能体创建个性化的应用，并在使用过程中持续提供其行为。我们实现了LiveUIAgent，它生成一个任务特定的编排，将用户界面、保存的信息和服务操作连接起来。一个共享的智能体运行时在渲染和更新界面的同时执行所请求的操作。通过复用该运行时来修订编排和界面，为需求不断演变时的个性化提供了轻量级的基础。

    arXiv:2610.27977v1 Announce Type: new  Abstract: Applications provide convenient features for users to complete daily tasks. However, different users may have different or even conflicting requirements for the same application, while the requirements of the same user may change over time. Personalizing applications produced by agent-to-app or UI-to-code approaches can require changes to the generated implementation. We propose Agent as App, a task in which users express their requirements in natural language and an agent creates a personalized application while continuing to supply its behavior during use. We implement LiveUIAgent, which generates a task-specific orchestration connecting the user interface, saved information, and service actions. A shared agent runtime renders and updates the interface while carrying out the requested operations. Revising the orchestration and interface while reusing that runtime provides a lightweight basis for personalization as requirements evolve. 
    
[^14]: 开发者如何配置大语言模型推理：对真实世界实践的大规模研究

    How Developers Configure LLM Inference: A Large-Scale Study of Real-World Practices

    [https://arxiv.org/abs/2610.27945](https://arxiv.org/abs/2610.27945)

    该论文通过对 5,398 个 GitHub 仓库的大规模实证研究，系统揭示了开发者在使用 Hugging Face Transformers、vLLM 和 OpenAI API 时的真实 LLM 推理配置实践，发现开发者实际常规使用的配置选项仅占可用配置空间的一小部分。

    

    大语言模型（LLM）正日益被集成到软件系统中，使得部署和推理配置成为一项经常性的软件工程任务。推理后端暴露了大量影响模型加载、生成、服务以及请求处理的选项。然而，人们对开发者在实践中实际显式配置了哪些选项知之甚少。在本文中，我们对开源软件中使用 Hugging Face Transformers、vLLM 和 OpenAI API 的 LLM 推理配置实践开展了一项大规模实证研究。我们分析了 5,398 个至少使用其中一种后端的 GitHub 仓库，并识别出 345,117 次显式配置选项的出现。我们的结果表明，开发者通常只使用可用配置空间中的一小部分。尽管 98.8% 的已记录的后端特定选项至少出现过一次，但其中 78.9% 的选项仅在至多 10% 的使用相应后端的仓库中被配置。配置是……（原文在此处截断）

    arXiv:2610.27945v1 Announce Type: new  Abstract: LLMs are increasingly integrated into software systems, making deployment and inference configuration a recurring software engineering task. Inference backends expose many options that affect model loading, generation, serving, and request processing. Yet little is known about which of these options developers explicitly configure in practice. In this paper, we present a large-scale empirical study of LLM inference configuration practices in open-source software using Hugging Face Transformers, vLLM, and the OpenAI API. We analyze 5,398 GitHub repositories using at least one of these backends and identify 345,117 explicitly configured option occurrences. Our results show that developers use only a small part of the available configuration space regularly. Although 98.8% of the documented backend-specific options appear at least once, 78.9% are configured in at most 10% of repositories using the corresponding backend. Configurations are a
    
[^15]: 从14节点到70,000节点的CPU、GPU与机器学习潮流求解器基准测试

    Benchmarking CPU, GPU, and ML Power-Flow Solvers from 14 to 70,000 Buses

    [https://arxiv.org/abs/2610.27930](https://arxiv.org/abs/2610.27930)

    该论文构建了涵盖9个物理求解器和3个机器学习替代模型的潮流计算基准，揭示基线选择可令速度结论相差2.5-3个数量级，并证明充分并行化的CPU求解器或新的GPU求解器gpusim2grid在吞吐量上可比肩甚至超过ML替代模型，且精度高出数个数量级。

    

    学习型潮流计算替代模型通常以速度为卖点，并与单一的基于CPU的基线进行比较。我们表明这种比较是脆弱的：在相同输入上评估的7个开源CPU求解器中，吞吐量差异约为2.5-3个数量级，因此仅基线的选择就可能翻转结论。我们构建了一个基准，涵盖9个基于物理的求解器——7个基于CPU的和2个基于GPU的，包括一个新的批量GPU求解器gpusim2grid——以及3个已发表的机器学习替代模型，在共享的、独立实现的KCL失配检查下，对从14到70,000节点的15个电网进行了评估。在潮流计算任务本身上，充分并行化的CPU求解器或gpusim2grid的吞吐量达到甚至超过我们测试的机器学习替代模型，同时精度高出数个数量级；我们没有发现证据表明这些GNN风格的替代模型在潮流计算这一特定任务上已具备竞争力，尽管先前的工作报告了真实的收益。

    arXiv:2610.27930v1 Announce Type: cross  Abstract: Learned surrogates for power-flow (PF) computation are usually motivated by speed, compared against a single CPU-based baseline. We show this comparison is fragile: across 7 open-source CPU-based solvers evaluated on identical inputs, throughput varies by roughly 2.5-3 orders of magnitude, so the choice of baseline alone can flip the conclusion. We build a benchmark spanning 9 physics-based solvers: 7 CPU-based and 2 GPU-based, including a new batched GPU solver, gumsim2grid - and 3 published ML surrogates, evaluated under a shared, independently implemented KCL-mismatch check across 15 grids from 14 to 70,000 buses. On the PF task itself, a sufficiently parallelized CPU solver or gpusim2grid matches or exceeds the throughput of the ML surrogates we test while remaining orders of magnitude more accurate; we find no evidence that these GNN-style surrogates are yet competitive for PF specifically, though prior work reports genuine gains 
    
[^16]: 用于免编译补丁存在性检测的神经符号推理引擎

    A Neuro-Symbolic Reasoning Engine for Compilation-Free Patch Presence Testing

    [https://arxiv.org/abs/2610.27871](https://arxiv.org/abs/2610.27871)

    该论文提出SyLares，一种深度融合大语言模型与SMT求解器的神经符号推理引擎，克服了此前Lares框架中两通道并行、LLM充当最终仲裁者导致的幻觉问题，实现了更可靠的免编译补丁存在性检测。

    

    arXiv:2610.27871v1 Announce Type: new 摘要：现代软件生态系统中的代码复用促进了1-day漏洞的传播，带来了重大的安全风险。判断补丁是否存在至关重要，但传统方法存在可用性和准确性有限的问题，且往往依赖于脆弱的编译过程。在我们之前的工作中，我们提出了Lares，这是首个将大语言模型（LLM）与SMT求解器相结合的免编译补丁存在性检测框架。然而，Lares将LLM和SMT求解器视为两个相互独立、并行的决策通道：符号通道（Z3）仅在20%的代码用例中成功，其余用例则回退到神经通道（LLM）进行最终决策。让LLM充当最终仲裁者，使验证变成了一个非正式、非确定性的过程，容易产生幻觉错误，同时两个通道之间也从未相互增强。为了克服这些局限，我们提出了SyLares，……

    arXiv:2610.27871v1 Announce Type: new  Abstract: Code reuse in modern software ecosystems facilitates the spread of 1-day vulnerabilities, posing significant security risks. Determining if patches are present is crucial, but traditional methods suffer from limited usability and accuracy, often depending on a fragile compilation process. In our prior work, we introduced Lares, the first compilation-free framework for patch presence testing that combines large language models (LLMs) with SMT solvers. However, Lares treats the LLM and the SMT solver as two separate, parallel decision channels: the symbolic channel (Z3) succeeded in only 20% of code cases, and the remaining cases fell back to the neural channel (the LLM) for the final decision. Letting the LLM act as the final arbiter turns verification into an informal, non-deterministic process that is prone to hallucination errors, while the two channels never reinforce each other. To overcome these limitations, we propose SyLares, whic
    
[^17]: 大语言模型在跨语言代码翻译中的幻觉：现象与缓解

    Hallucinations of Large Language Models in Cross-Language Code Translation: Phenomena and Mitigation

    [https://arxiv.org/abs/2610.27868](https://arxiv.org/abs/2610.27868)

    该论文对大语言模型在跨语言代码翻译中的幻觉现象开展了系统性实证研究，在三个基准数据集、六个语言对和七个大语言模型上构建了幻觉分类体系，并提出相应的缓解方法。

    

    代码翻译是将源代码从一种编程语言转换为另一种编程语言，同时保持其功能不变。尽管大语言模型（LLM）已经提升了代码翻译的效果，但其输出仍然包含幻觉：生成的代码与源程序冲突、违反目标语言的事实，或引入与上下文不一致的依赖项。这类失败在翻译任务中尤其有害，因为微小的语义偏差可能会悄然改变程序行为。现有的幻觉研究主要集中于自然语言生成或代码生成，而对跨语言代码翻译中幻觉的形式、分布及缓解措施的刻画尚不充分。本文对大语言模型在代码翻译中的幻觉进行了系统性的实证研究。在三个基准数据集、六个语言对和七个大语言模型上，我们构建了一个包含三大类别的分类体系：语义一致性……

    arXiv:2610.27868v1 Announce Type: cross  Abstract: Code translation converts source code from one programming language to another while preserving its functionality. Although Large Language Models (LLMs) have improved code translation, their outputs still contain hallucinations: generated code that conflicts with the source program, violates target-language facts, or introduces contextually inconsistent dependencies. These failures are especially harmful in translation because small semantic deviations can silently change program behavior. Existing hallucination studies mainly focus on natural-language generation or code generation, leaving the forms, distribution, and mitigation of hallucinations in cross-language code translation insufficiently characterized. This paper presents a systematic empirical study of LLM hallucinations in code translation. Across three benchmark datasets, six language pairs, and seven LLMs, we construct a taxonomy with three major categories: Semantic Consi
    
[^18]: 从词元到文件：AI 参与代码的多粒度检测与定位

    From Tokens to Files: Multi-Granularity Detection and Localization of AI-Involved Code

    [https://arxiv.org/abs/2610.27863](https://arxiv.org/abs/2610.27863)

    提出了一种基于冻结代码语言模型上双低秩适配器逐词元对数似然比的框架，可在词元证据基础上对 AI 参与代码进行行、块、文件多粒度的检测与定位。

    

    随着大语言模型在代码生成中的使用日益增多，亟需能够区分纯人工编写代码与“AI 参与代码”（即文件中任何部分，无论多么微小，由模型生成或经模型实质性修改）的方法。除了标记此类文件之外，我们还旨在定位可归因于 AI 的代码行或代码块。现有方法将文件压缩为单一分数，依赖于表面风格特征或全局代码表示，当文件中仅有少数代码块由 AI 参与时，这种方式会平均化掉相关证据。我们提出了一个在词元级别建模 AI 创作归属、并在行、块和文件三个粒度上进行预测的框架。一个冻结的代码语言模型配备了两个低秩适配器，分别在不同机器编写和人工编写的文件上训练；两者的逐词元对数似然比为 AI 创作归属提供了局部证据。轻量级的预测头利用周围的代码上下文，在行和块内部聚合这些证据……

    arXiv:2610.27863v1 Announce Type: cross  Abstract: The increasing use of large language models for code generation calls for methods that distinguish purely human-written code from AI-involved code: files in which any part, however small, was generated or substantially modified by a model. Beyond flagging such files, we aim to localize the lines or blocks attributable to AI. Existing approaches compress a file into a single score, from surface stylistic features or a global code representation, which averages away the evidence when only a few blocks are AI-involved. We propose a framework that models AI authorship at the token level and predicts at the line, block, and file levels. A frozen code language model is equipped with two low-rank adapters, trained separately on machine-written and human-written files; their per-token log-likelihood ratio provides local evidence of AI authorship. Lightweight heads aggregate this evidence within lines and blocks, using the surrounding code cont
    
[^19]: 人类与语言模型代理共用的电子实验笔记本中的值级数据溯源：一份经验报告

    Value-level data provenance in an electronic laboratory notebook used by humans and language-model agents: An experience report

    [https://arxiv.org/abs/2610.27859](https://arxiv.org/abs/2610.27859)

    本文报告了电子实验笔记本APS ELN的值级数据溯源架构经验：针对近5年审计中发现的“从原始文件到存储数值”关联丢失问题，通过重构数据库和工作流程，使每个存储数值都记录其源文件、提取插件版本及派生公式，从而实现数值的可验证性。

    

    一个已发表的实验室数值，只有在其原始文件和提取方法都已知的情况下才能得到验证。本文介绍了APS ELN的值级溯源架构，这是一种同时供研究人员和语言模型代理使用的电子实验笔记本。该架构源于一家材料实验室的生产审计，该审计涵盖了近5年间执行的项目。对八项基线研究从样品到存储数值的追溯显示，最初的ELN结构保留了从样品到原始文件的关联，但从原始文件到存储数值的关联却丢失了，主要原因是人为因素：手动输入未携带文件引用或提取器版本信息。为弥补这一缺口，数据库设计和工作流程经过重构，使每个存储的数值都记录其源文件、数据文件读取器/查看器（DFR）插件及其版本、任何交互式分析选择，或用于派生量的服务器端公式。一个自动化

    arXiv:2610.27859v1 Announce Type: cross  Abstract: A published laboratory number can be verified only when its raw file and extraction method are known. This paper presents the value-level provenance architecture of APS ELN, an electronic laboratory notebook used by both researchers and language-model agents. The architecture was motivated by a production audit in a materials laboratory covering projects performed over almost 5 years. Tracing eight baseline studies from sample to stored value showed that the initial ELN structure preserved the link from sample to raw file. Still, the link from raw file to stored value was lost, mainly due to human factors: manual entries carried no file reference or extractor version. To close this gap, the database design and workflow were reworked so that every stored value records its source file, the data file reader/viewer (DFR) plugin and version, any interactive analysis selection, or the server-side formula for derived quantities. An automated 
    
[^20]: 增投还是检查？预测持续模糊测试中额外模糊测试的必要性

    Raise or Check? Predicting the Necessity of Additional Fuzzing for Continuous Fuzzing

    [https://arxiv.org/abs/2610.27828](https://arxiv.org/abs/2610.27828)

    提出RAISE即时框架，通过在项目-日粒度上预测漏洞引入风险，优先为高风险项目分配额外模糊测试资源，从而高效利用计算资源并缩小持续模糊测试中的漏洞暴露窗口期。

    

    持续模糊测试被用于开源软件（OSS）中的漏洞检测。像Google的OSS-Fuzz这样的大规模服务对超过1,300个项目进行模糊测试，并已发现超过10,000个漏洞。尽管如此，从漏洞被引入到被检测出来之间往往存在相当大的时间滞后（即暴露窗口期）。考虑到额外模糊测试的计算成本，我们旨在利用漏洞风险评估来高效分配模糊测试资源。我们提出了RAISE（风险感知的密集补充执行框架），这是一个即时框架，可优先对高风险的项目-日进行额外模糊测试。RAISE在项目-日粒度上执行二元分类，以估计某一天是否引入了漏洞。我们构建了一个数据集，将OSV数据库中引入漏洞的提交与OSS-Fuzz的每日覆盖率和变更指标相结合，覆盖了……

    arXiv:2610.27828v1 Announce Type: new  Abstract: Continuous fuzzing is used for vulnerability detection in Open Source Software (OSS). Large-scale services such as Google's OSS-Fuzz fuzz over 1,300 projects and have revealed more than 10,000 vulnerabilities. Despite this, a considerable time lag often exists between when a vulnerability is introduced and when it is detected (i.e., the window of exposure). Given the computational cost of additional fuzzing, we aim to allocate fuzzing resources efficiently using vulnerability-risk estimates. We propose RAISE (Risk-Aware Intensive Supplemental Execution Framework), a Just-In-Time (JIT) framework that prioritizes additional fuzzing for high-risk project-days. RAISE performs binary classification at the project-day granularity to estimate whether a vulnerability is introduced on a given day. We construct a dataset combining vulnerability-introducing commits from the OSV database with daily coverage and change metrics from OSS-Fuzz, covering
    
[^21]: MIMOS-Tools：面向嵌入式系统基于模型设计的集成工具链

    MIMOS-Tools: An Integrated Toolchain for Model-Based Design of Embedded Systems

    [https://arxiv.org/abs/2610.27826](https://arxiv.org/abs/2610.27826)

    该论文提出了MIMOS-Tools集成工具链，基于具有形式化操作语义和功能与时间确定性的MIMOS计算模型，为嵌入式实时系统提供从建模、仿真、形式化验证到异构多核平台代码生成的全生命周期一体化支持。

    

    尽管基于模型的方法和工具在嵌入式实时系统领域已经发展了几十年，但至今仍没有一个能够支持完整开发生命周期的集成环境。本文提出了MIMOS-Tools，这是一个覆盖开发关键阶段的综合性工具链，包括建模、仿真、形式化验证、实时调度、可调度性与端到端延迟分析，以及面向异构多核平台的代码生成。所有阶段均建立在一个单一、连贯且确定性的计算模型——MIMOS模型之上，本文为其提出了形式化操作语义，并证明了其功能确定性与时间确定性。这一语义基础确保了在模型层面验证的性质能够在整个开发过程及最终生成的实现中得以保持。MIMOS-Tools的一个显著特点是其不仅支持确定性，还支持模块化、可组合的……

    arXiv:2610.27826v1 Announce Type: new  Abstract: Despite decades of progress in model-based methods and tools for embedded real-time systems, there is still no integrated environment that supports the entire development lifecycle. This paper presents MIMOS-Tools, a comprehensive toolchain spanning the key phases of development, including modeling, simulation, formal verification, real-time scheduling, schedulability and end-to-end latency analysis, and code generation for heterogeneous multicore platforms. All phases are grounded in a single, coherent, and deterministic model of computation, the MIMOS model, for which we present a formal operational semantics and establish both functional and timing determinism. This semantic foundation ensures that properties verified at the model level are preserved throughout the development process and in the generated implementation.   A distinguishing feature of MIMOS-Tools is its support for not only deterministic but also modular, compositional
    
[^22]: RAG-NIMBF：一种用于自动化RAG配置分析的非侵入式模块化基准测试框架

    RAG-NIMBF: A Non-Invasive, Modular Benchmarking Framework for Automated RAG Configuration Analysis

    [https://arxiv.org/abs/2610.27816](https://arxiv.org/abs/2610.27816)

    该论文提出了RAG-NIMBF，一个可作为非侵入式评估层附加到现有RAG管道的模块化基准测试框架，通过受控消融实验与端到端实验跟踪，结合LLM答案质量评分、检索正确性和运行时指标，自动分析RAG庞大配置空间中各组件间的非显而易见的交互作用。

    

    检索增强生成（RAG）已成为将大语言模型（LLM）的输出锚定于外部知识的主流范式。然而，从业者面临着由分块、嵌入、检索、重排序、提示和生成模型组成的庞大配置空间，而关于这些组件之间如何相互作用的指导却十分有限。我们提出了RAG-NIMBF，这是一个模块化的基准测试框架，它作为非侵入式评估层附加到现有的RAG管道上。该框架通过受控消融实验和端到端的实验跟踪来评估配置空间。评估结合了基于LLM的答案质量评分、检索正确性度量以及运行时指标。本研究在SQuAD数据集上通过多个生成模型对该框架进行了验证。这一过程揭示了非显而易见的组件交互作用，并展示了实验跟踪的实现方式。研究结果表明，检索平衡...

    arXiv:2610.27816v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) has become the dominant paradigm for grounding the outputs of large language models (LLM) in external knowledge. However, practitioners are faced with a vast configuration space comprising chunking, embedding, retrieval, reranking, prompting and generation models, and there is limited guidance on how these components interact. We present RAG-NIMBF, a modular benchmarking framework that attaches to existing RAG pipelines as a non-invasive evaluation layer. It evaluates the configuration space through controlled ablation with end-to-end experiment tracking. The evaluation combines LLM-based answer quality scores, retrieval correctness measures, and runtime indicators. This study validates the framework across multiple generation models on the SQuAD dataset. This process reveals non-obvious interactions and demonstrates how the experiments are tracked. The findings indicate that the retrieval balance o
    
[^23]: RELAY：通过规则复用与智能体修复实现历史构建恢复的规模化

    RELAY: Scaling Historical Build Restoration through Rule Reuse and Agentic Repair

    [https://arxiv.org/abs/2610.27784](https://arxiv.org/abs/2610.27784)

    Relay是一个混合框架，它利用LLM从代表性构建失败中自动推导并验证可复用的确定性修复规则，将规则无法解决的少数案例交给智能体逐案修复，从而规模化地恢复了历史软件（14,159个OSS-Fuzz案例中近七成因环境衰退而无法重建）的构建。

    

    漏洞基准测试和软件工程研究依赖于对历史软件的重新构建。然而，构建过程依赖于外部资源，例如基础镜像和软件包，这些资源的变化独立于源代码。在本研究中，我们在14,159个OSS-Fuzz案例上考察了这种“环境衰退”现象。我们发现，69.7%的案例已无法完成重建与验证，且大多数失败共享少数几种反复出现的故障症状。因此，一个反复出现的故障只需诊断一次，其修复方案即可被复用。我们提出了Relay，一个混合框架：它利用大语言模型（LLM）从代表性故障中推导并验证确定性的修复规则，在不调用模型的情况下复用这些规则，并将规则无法解决的案例升级为由逐案例的智能体处理。我们在人工引导和全自动规则推导两种模式下评估了Relay，并将其与逐案例智能体方法进行比较。我们发现规则推导可以实现完全自动化，且可复用的规则……（原文摘要截断）

    arXiv:2610.27784v1 Announce Type: cross  Abstract: Vulnerability benchmarks and software engineering research depend on rebuilding historical software. Yet builds rely on external resources, such as base images and packages, that change independently of the source code. In this study, we examine this environment decay on 14,159 OSS-Fuzz cases. We find that 69.7% no longer complete reconstruction and verification, and that most failures share a small number of recurring failure symptoms. A recurring failure can therefore be diagnosed once and its repair reused. We present Relay, a hybrid framework that uses an LLM to derive and validate deterministic repair rules from representative failures, reuses them without model calls, and escalates cases unresolved by rules to a per-case agent. We evaluate Relay in human-guided and fully automated rule-derivation campaigns and compare it with a per-case agent approach. We find that rule derivation can be fully automated, and that reusable rules r
    
[^24]: 通过步骤级信用分配与从有价值失败中自蒸馏来增强暴露缺陷的测试生成

    Enhancing Bug-Exposing Test Generation via Step-Level Credit Assignment and Self-Distillation from Valuable Failures

    [https://arxiv.org/abs/2610.27694](https://arxiv.org/abs/2610.27694)

    SCALF提出了一种过程监督强化学习方法，通过步骤级信用分配让每个推理步骤获得独立的学习信号，并从找到触发缺陷输入但预测错误输出的有价值失败轨迹中进行自蒸馏学习，从而显著增强了暴露Bug的测试生成效果。

    

    大型语言模型已展现出强大的代码生成能力，但其输出往往包含细微的缺陷。自动修复此类代码需要信息丰富的测试用例：即能够暴露错误行为的输入，并配以正确的预期输出。近期基于强化学习的测试生成方法对所有推理步骤施加相同的结果奖励，这可能会抑制有用的推理并强化有缺陷的步骤。此外，这些方法未能有效地从那些找到触发缺陷的输入但预测了错误输出的轨迹中学习，导致其已验证的触发缺陷进展未得到充分利用。为解决这些局限，我们提出了SCALF，一种将步骤级信用分配与针对有价值失败轨迹进行定向学习相结合的过程监督强化学习方法。SCALF从过程奖励中推导出步骤特定的优势，使每个推理阶段都能获得自己的学习信号。它进一步将那些暴露（注：原始摘要在此处被截断，内容不完整）

    arXiv:2610.27694v1 Announce Type: new  Abstract: Large Language Models have shown strong code generation capabilities, yet their outputs often contain subtle bugs. Automatically repairing such code requires informative tests: inputs that expose faulty behavior, paired with correct expected outputs. Recent RL-based test generation methods apply the same outcome reward to all reasoning steps, which can discourage useful reasoning and reinforce flawed steps. They also fail to effectively learn from trajectories that find bug-triggering inputs but predict incorrect outputs, leaving their verified bug-triggering progress insufficiently exploited. To address these limitations, we propose SCALF, a process-supervised RL approach that combines Step-level Credit Assignment with targeted Learning from valuable Failed trajectories. SCALF derives step-specific advantages from process rewards, allowing each reasoning stage to receive its own learning signal. It further routes trajectories that expos
    
[^25]: 知识即代码

    Knowledge as Code

    [https://arxiv.org/abs/2610.27693](https://arxiv.org/abs/2610.27693)

    该论文从认识论角度提出“知识即代码”框架，定义了十三种知识载体属性并证明代码是首个同时具备全部属性的载体，同时区分了编码与通过执行学习两种知识获取方式。

    

    千百年来，知识一直由语言承载，大型语言模型也依然将语言置于核心地位。然而，构建在这些模型之上的编程智能体却大多在编写代码。本文是对认识论的一项贡献，探讨代码是何种知识以及它是如何获得的。我们将“知识即代码”定义为：以编程语言或机器可解析数据格式书写的知识，其有效性可以通过执行来检验。我们定义了知识载体的十三种认识论属性，并证明代码是第一种同时具备所有这些属性的载体。我们区分了知识进入代码的两种方式：编码和通过执行学习，在后者中代码既是实验本身，也是实验结果的记录。智能体使“通过执行学习”变得可观察，因为它们的轨迹记录了每一次探测、观察以及保留的每一行代码。因此，代码不仅是知识工作的工具，其本身也是一种独立的知识载体。

    arXiv:2610.27693v1 Announce Type: new  Abstract: For millennia, knowledge has been carried by language, and large language models still put language at the center. Yet the coding agents built on these models mostly write code. This paper is a contribution to epistemology, which asks what kind of knowledge code is and how it is obtained. We define knowledge as code as knowledge written in a programming language or in a machine-parsable data format, whose validity can be checked by executing it. We define thirteen epistemic properties of knowledge media and show that code is the first medium to have all of them. We distinguish two ways in which knowledge enters code, encoding and learning by execution, where code is both the experiment and the record of its result. Agents make learning by execution observable, since their trajectories record every probe, observation and line of code they keep. Code is thus not only a tool of knowledge work but a knowledge medium in its own right, the fir
    
[^26]: AI管道中保证证据的复用：有效性条件、故障关闭式发布门控，以及对714个公开数据版本控制管道的重放

    Reusing Assurance Evidence in AI Pipelines: Validity Conditions, a Fail-Closed Release Gate, and a Replay of 714 Public Data Version Control Pipelines

    [https://arxiv.org/abs/2610.27672](https://arxiv.org/abs/2610.27672)

    该论文为AI管道中保证证据的复用建立了有效性条件，提出基于Ed25519签名的故障关闭式发布门控（可阻止9种操纵场景中的8种），并通过对714个公开数据版本控制管道的38,083次提交进行重放，量化了依赖声明在实践中失效的情况。

    

    在人工智能（AI）管道每次变更后重新运行所有治理控制代价高昂，因此团队会复用先前的结果。只有当每个控制所声明的依赖项涵盖了所有可能改变其结果的因素，并且发布门控检查复用的证据是否描述了正在发布的候选版本时，复用才是安全的。本文将这两个条件明确化，并测量了第一个条件在实践中的失效频率。文章定义了签名证据记录何时对某个候选版本有效，并给出了变更局部重新执行能够达到与重新运行所有控制相同发布决策的充分条件。一个使用Ed25519数字签名实现这些检查的参考门控阻止了九种操纵场景中的八种，相比之下，信任规划器的门控仅能阻止一种，仅验证签名的门控仅能阻止三种；而一个策略从未声明的依赖项则可以通过所有门控。对来自……的38,083次提交的重放（原文摘要在此处截断）

    arXiv:2610.27672v1 Announce Type: cross  Abstract: Rerunning every governance control after every change to an artificial intelligence (AI) pipeline is expensive, so teams reuse earlier results. Reuse is safe only if each control's declared dependencies cover everything that can change its outcome, and only if the release gate checks that reused evidence describes the candidate being released. This article makes both conditions explicit and measures how often the first fails in practice. It defines when a signed evidence record is valid for a candidate and gives sufficient conditions under which change-local re-execution reaches the same release decision as rerunning every control. A reference gate implementing these checks with Ed25519 digital signatures blocks eight of nine manipulation scenarios, against one for a gate that trusts the planner and three for a gate that only verifies signatures; a dependency the policy never declared passes every gate. A replay of 38,083 commits from 
    
[^27]: 标准引导与依赖感知的代码级文档生成

    Standards-Guided and Dependency-Aware Code-Level Documentation Generation

    [https://arxiv.org/abs/2610.27634](https://arxiv.org/abs/2610.27634)

    提出CODEDOC框架，将代码级文档生成重塑为一项基于权威质量标准的标准引导软件工程任务，并借助静态调用图与AST分析构建依赖感知的结构化提示。

    

    为软件仓库自动生成高质量文档仍然具有挑战性：现有方法大多将文档视为无约束的文本生成任务，缺乏明确的质量规范，严重依赖对原始代码的长上下文提示，且对可复现评估的支持有限。我们认为，代码级文档生成应当是一项标准引导的软件工程任务，文档的生成与评估应基于从权威标准中提炼出的明确且可复用的质量要求。我们将代码文档定义为按源代码路径组织的、对仓库中函数和类的自然语言描述。我们提出了CODEDOC，一个标准引导且依赖感知的框架，该框架通过静态调用图和AST（抽象语法树）分析识别代码实体之间的依赖关系，并构建结合局部代码上下文的依赖感知结构化提示。

    arXiv:2610.27634v1 Announce Type: new  Abstract: Automatically generating high-quality documentation for software repositories remains challenging: existing approaches largely treat documentation as unconstrained text generation, lacking explicit quality specifications, depending heavily on long-context prompts over raw code, and offering limited support for reproducible evaluation. We argue that code-level documentation generation should instead be a standards-guided software engineering task, in which documentation is generated and assessed against explicit, reusable quality requirements derived from authoritative standards. We define code documentation as natural-language descriptions of repository functions and classes, organized by source-code path. We present CODEDOC, a standards-guided and dependency-aware framework that identifies dependencies between code entities via static call-graph and AST analysis and constructs dependency-aware structured prompts combining local code con
    
[^28]: 审计产品而非递归过程：递归自我改进声明背后的评估证据

    Auditing the Product, Not the Recursion: The Evaluation Evidence Behind Recursive Self-Improvement Claims

    [https://arxiv.org/abs/2610.27610](https://arxiv.org/abs/2610.27610)

    该论文首次通过对114项递归自我改进（RSI）研究的系统性全文编码，量化审计了RSI声明背后产品评估与流程评估证据的实际报告频率，揭示了递归自我改进声明与其实际评估证据之间的差距。

    

    递归自我改进（RSI）承诺了诸如自我进化的LLM智能体等AI系统能够在产生进一步改进方面不断提升。评估这一承诺需要将关于最终系统的“产品”声明与关于改进过程本身的“流程”声明区分开来。现有综述已明确了这些声明所需的证据，但研究实际报告这些证据的频率仍未被测量。我们通过对2026年10月冻结的可复现arXiv搜索框架中114项符合条件的研究进行全文编码来考察这一问题，其中包括90项将自身系统描述为RSI的研究。该编码手册将声明与评估设计相关联。通过独立的人工编码、第三方读者的裁定，以及对改进器比较的单独盲编码来评估编码的可靠性。在90项RSI研究中，90%在包含部分分离或未明确说明分离的宽泛标准下报告了超越开发任务的产品评估，而60%报告了评估……（原文摘要在此处截断）

    arXiv:2610.27610v1 Announce Type: cross  Abstract: Recursive self-improvement (RSI) promises AI systems, such as self-evolving LLM agents, that become better at producing further improvements. Evaluating this promise requires separating product claims about the final system from process claims about the improvement process itself. Existing surveys specify the evidence these claims require, but how often studies report that evidence remains unmeasured. We examine this question through full-text coding of 114 eligible studies in a reproducible arXiv search frame frozen in October 2026, including 90 that describe their own systems as RSI. The codebook links claims to evaluation designs. Independent human coding, third-reader adjudication, and a separate blind coding of improver comparisons assess coding reliability. Among the 90 RSI studies, 90% report product evaluation beyond development tasks under a broad criterion that includes partial or unstated separation, while 60% report evaluat
    
[^29]: 可恢复性与验证器强度：无人值守智能体自我改进的两个可测量前置条件

    Recoverability and Verifier Strength: Two Measured Preconditions for Unattended Agent Self-Improvement

    [https://arxiv.org/abs/2610.27465](https://arxiv.org/abs/2610.27465)

    论文提出了“验证器强度”和“可恢复性”这两个可测量的前置条件，用于判断智能体平台的哪些自我改进步骤可以安全地无人值守运行。

    

    智能体平台开始改进自身的智能体：它们读取已部署智能体的执行轨迹，推导出变更，让自动化验证器批准该变更，并在没有人类作者的情况下发布。这个循环中有多少可以安全地在无人参与的情况下运行？我们提出了平台可以测量的两个量。验证器强度 σ 衡量可用验证器（通常是 LLM 判官）与可信参考验证器的一致程度，它限制了一个步骤能在多大程度上改进智能体。可恢复性衡量撤销一个不良变更的代价有多低：撤销它是否需要新的验证器批准，以及回滚能恢复多少旧行为。只有当两者都充分时，一个步骤才应该无人值守运行。在我们的平台上，这一规则允许提示词、工具和技能编辑无人值守运行，同时使权重更新保持在人类签署批准之下。我们的主要贡献是这两个定义及其测量方法。我们构建了一个应用该规则的平台并进行了测试。

    arXiv:2610.27465v1 Announce Type: new  Abstract: Agent platforms are starting to improve their own agents: they read a deployed agent's execution traces, derive a change, have an automated verifier approve it, and release it with no human author. How much of this loop can safely run without a human? We propose two quantities a platform can measure. Verifier strength $\sigma$ is how well the available verifier -- often an LLM judge -- agrees with a trusted reference verifier; it limits how much a step can improve the agent. Recoverability is how cheaply a bad change can be undone: whether undoing it needs a new verifier approval, and how much of the old behavior a rollback restores. A step should run unattended only if both are adequate. On our platform this rule allows prompt, tool and skill edits to run unattended and keeps weight updates under human sign-off.   Our main contribution is these two definitions and how to measure them. We built a platform that applies the rule and tested
    
[^30]: AgenticVulHunter：将本地智能体漏洞挖掘带入你的终端

    AgenticVulHunter: Bringing Local Agentic Vulnerability Hunting to Your Terminal

    [https://arxiv.org/abs/2610.27383](https://arxiv.org/abs/2610.27383)

    本文提出AgenticVulHunter，一个可在本地部署的开源权重LLM上运行的智能体安全代码审查工具，通过聚焦且相关的安全知识实现终端漏洞挖掘。

    

    arXiv:2610.27383v1 公告类型： cross 摘要：在漏洞影响生产系统之前尽早发现它们，对于降低软件攻击风险至关重要。然而，在安全代码审查过程中检测漏洞仍然十分困难，因为漏洞代码逻辑可能依赖于分布在代码库不同文件中的代码行。近期的方法（如AgenticSCR）已经表明，LLM智能体能够探索代码库文件并利用安全知识来改进安全代码审查。然而，这些方法主要依赖外部的商业LLM，并且可能一次性向智能体提供大量安全知识。这就引出了一个重要问题：使用本地部署的开源权重LLM并配合更加聚焦且相关的安全知识，是否同样能够有效地执行基于智能体的安全代码审查。在本文中，我们提出了AgenticVulHunter，这是一种能够在本地部署的开源权重LLM上运行的智能体安全代码审查方法与工具。

    arXiv:2610.27383v1 Announce Type: cross  Abstract: Finding vulnerabilities early, before they can affect the production system, is important for reducing the risk of any software attacks. However, detecting vulnerabilities during secure code review is still difficult because the vulnerable code logic may depend on lines found across different repository files. Recent approaches such as AgenticSCR have shown that LLM agents can explore repository files and use security knowledge to improve secure code review. However, these approaches mainly rely on external commercial LLMs and can provide a large amount of security knowledge to the agent at once. This creates an important question of whether agentic secure code review can also be performed effectively using locally deployed open-weight LLMs with more focused and relevant security knowledge. In this paper, we propose AgenticVulHunter, an agentic secure code review approach and tool that can run with a locally deployed open-weight LLM. W
    
[^31]: AI生成拉取请求中的文档漂移

    Documentation Drift in AI-Generated Pull Requests

    [https://arxiv.org/abs/2610.27381](https://arxiv.org/abs/2610.27381)

    该研究首次大规模刻画了AI生成代码的文档编写实践，发现63.2%的AI拉取请求在修改代码时不更新任何文档，且无文档的代码文件后续被修复的概率显著更高，揭示了“文档漂移”现象及其对软件质量的负面影响。

    

    AI编程代理如今正大规模地为软件仓库做出贡献，但其文档编写实践仍未被系统性地刻画。我们分析了AIDev数据集中来自2,585个仓库的29,669个AI生成的拉取请求（PR），以衡量文档协同变更：即代码文件的修改是否在同一PR中伴随文档文件的更新。我们发现63.2%的AI PR在不更新任何文档的情况下修改代码，且这一比例因任务类型而有显著差异：功能开发类为55.2%，重构类为77.5%。在30至90天的时间窗口内，无文档代码文件随后对同一文件进行修复的概率比有文档文件高8.6%至11.7%。在PR层面的原始比较中结果相反，因为有文档的PR规模是无文档PR的3.9倍，因而吸引了更多修复；而在控制规模暴露因素后，这一结果被逆转。尽管规模更大，有文档的变更合并速度也更快（中位数：40秒对比184秒），尽管合并主要由自动……（原文摘要在此处截断）

    arXiv:2610.27381v1 Announce Type: new  Abstract: AI coding agents now contribute to software repositories at scale, yet their documentation practices remain uncharacterised. We analyse 29,669 AI-generated pull requests from 2,585 repositories in the AIDev dataset, measuring documentation co-change: whether code-file modifications are accompanied by a documentation-file update in the same PR. We find that 63.2% of AI PRs change code without updating any documentation. This varies sharply by task type: 55.2% for features and 77.5% for refactoring. Undocumented code files are followed by a fix to the same file 8.6 to 11.7% more often than documented files across 30 to 90-day windows. A raw PR-level comparison shows the opposite because documented PRs are 3.9 times larger and therefore attract more fixes. Controlling for size exposure reverses this result. Documented changes also merge faster despite their larger size (median: 40 seconds versus 184 seconds), although merging is largely aut
    
[^32]: 事后软件演化的特征分析

    Characterizing Post-Incident Software Evolution

    [https://arxiv.org/abs/2610.27380](https://arxiv.org/abs/2610.27380)

    本研究首次对软件事故后的演化进行了实证特征刻画，通过分析维基媒体10年间1,302个任务和2,203个代码审查的混合方法研究，揭示了事故引发的变化在工作量、位置和时机等方面与标准维护变化的差异。

    

    背景：在软件事故发生后，组织试图通过事后分析和确定相关行动来防止或减轻未来类似事故的发生，从而提高其软件系统的可靠性。尽管这项活动非常重要，但对于由该分析过程所引发的软件演化的性质和特征，人们的理解仍然有限。目标：本研究旨在通过对事后软件演化进行首次实证特征刻画来填补这一空白。我们旨在建立对这类变化如何表现的理解，特别关注其特征（如工作量、位置和时机）与标准维护及演化变化有何差异。方法：我们对来自维基媒体（Wikimedia）的10年纵向数据集进行了混合方法分析，考察了1,302个任务和2,203个代码审查。我们从五个维度对这些变化进行了分类：目标、工作量、……

    arXiv:2610.27380v1 Announce Type: new  Abstract: BACKGROUND: After a software incident, organizations attempt to improve the reliability of their software systems by conducting post-incident analysis and identifying actions to prevent or mitigate similar future incidents. Despite the importance of this activity, there is a limited understanding of the nature and characteristics of the software evolution triggered by the analysis process. AIMS: This study seeks to bridge this gap by providing the first empirical characterization of post-incident software evolution. We aim to establish an understanding of how these changes manifest, specifically focusing on how their characteristics (such as effort, location, and timing) differ from standard maintenance and evolutionary changes. METHOD: We conducted a mixed-methods analysis of a 10-year longitudinal dataset from Wikimedia, examining 1,302 tasks and 2,203 code reviews. We categorized these changes across five dimensions: target, effort, a
    
[^33]: 告别按字母付费：利用主机侧令牌化日志加速全芯片RTL仿真

    Stop Paying by the Letter: Accelerating Full-Chip RTL Simulation with Host-Side Tokenized Logging

    [https://arxiv.org/abs/2610.27356](https://arxiv.org/abs/2610.27356)

    提出一种面向异构多核SoC的主机侧仿真内令牌化日志架构，将printf的格式化与MMIO串行化开销转移至主机端并以零仿真时间重建字符串，从而同时消除MMIO总线停顿与片上vsprintf格式化两大瓶颈，大幅加速全芯片RTL仿真。

    

    固件诊断日志（printf）是全芯片SoC设计验证中必不可少的调试手段，但它给周期精确的RTL仿真带来了沉重的开销：在仿真CPU上格式化一条30字符、单参数的消息并通过MMIO接口序列化传输，每次调用需耗时3.04至15.23微秒（在Suite A测试集上，未加权关键路径均值为15.27微秒，按打印加权的均值为15.08微秒）。与那些面向闪存占用和线缆字节数、采用离线解码的嵌入式分词器不同，RTL仿真受一种双区域成本函数支配——跨fabric的SoC路径主要受MMIO总线停顿影响（占调用延迟的84.7%），而本地子系统路径则主要受片上vsprintf格式化影响（占61.7%）——并且需要实时的、零仿真时间的字符串重建来驱动UVM记分板。我们提出了一种面向异构多核SoC的主机侧仿真内令牌化日志架构，通过……消除了这两个瓶颈。

    arXiv:2610.27356v1 Announce Type: cross  Abstract: Firmware diagnostic logging (printf) is an essential debug instrument in full-chip SoC design verification, yet it imposes a heavy tax on cycle-accurate RTL simulation: formatting a 30-character, 1-argument message on a simulated CPU and serializing it through an MMIO aperture costs 3.04 to 15.23 $\mu$s per call (15.27 $\mu$s unweighted / 15.08 $\mu$s print-weighted critical-path mean on Suite A). Unlike embedded tokenizers that target flash footprint and wire bytes with offline decoding, RTL simulation is governed by a two-regime cost function---cross-fabric SoC paths are dominated by MMIO bus stalls (84.7% of call latency) whereas local subsystem paths are dominated by on-core vsprintf formatting (61.7%)---and requires live, zero-simulation-time string reconstruction to drive UVM scoreboards.   We present a host-side in-simulation tokenized logging architecture for heterogeneous multi-core SoCs that eliminates both bottlenecks with z
    
[^34]: XRust：检测Rust Linux驱动程序中的逻辑竞争

    XRust: Detecting Logical Races in Rust Linux Drivers

    [https://arxiv.org/abs/2610.27325](https://arxiv.org/abs/2610.27325)

    XRust无需人工指定不变量，通过逻辑先于关系分析自动构造操作对测试基座并使用串行差分预言机，实现Linux Rust驱动程序中逻辑竞争的自动检测。

    

    Rust在Linux中的应用日益增多，以提升内存安全性，但其保证并不能防止逻辑竞争：即使各个操作已单独同步，它们仍可能以违反高层不变量的方式交错执行。检测这类缺陷通常需要人工编写的规范说明以及精心控制的执行环境。我们提出XRust，一种无需人工指定不变量即可检测Linux Rust驱动程序中逻辑竞争的技术。借助逻辑先于关系分析，XRust识别与逻辑顺序相关的共享状态访问，自动构造操作对测试基座，并使用串行差分预言机来检测与任一串行顺序不一致的并发执行。XRust还利用轻量级的用户态抽象，在可复现的调度下执行驱动程序代码。应用于Linux 7.3-rc2时，XRust在来自十个驱动程序的102个Rust文件中识别出17个共享对象，构造了1,654个……

    arXiv:2610.27325v1 Announce Type: new  Abstract: Rust is increasingly used in Linux to improve memory safety, but its guarantees do not prevent logical races: individually synchronized operations can still interleave in ways that violate higher-level invariants. Detecting such bugs typically requires manually written specifications and carefully controlled execution environments.   We present XRust, a technique for detecting logical races in Linux Rust drivers without manually specified invariants. Using Logical Happens-Before (LHB) analysis, XRust identifies shared-state accesses relevant to logical ordering, automatically constructs operation-pair harnesses, and uses serial differential oracles to detect concurrent executions inconsistent with either serial order. XRust also uses lightweight userspace abstractions to execute driver code under reproducible schedules. Applied to Linux 7.3-rc2, XRust identifies 17 shared objects across 102 Rust files from ten drivers, constructs 1,654 e
    
[^35]: 面向Android混合应用的混合信息流分析框架

    A Hybrid Information Flow Analysis Framework for Android Hybrid Apps

    [https://arxiv.org/abs/2610.27288](https://arxiv.org/abs/2610.27288)

    MifDroid是一个静态分析引导动态分析的混合框架，通过生命周期感知的WebView调用图、桥接感知的类型-效果系统和模块化JavaScript分析引擎，解决了Android混合应用中Java与JavaScript之间跨语言信息流无法被单语言分析器检测的问题。

    

    Android（混合）应用中的WebView接口使得原生端（如Java）能够与Web端（如JavaScript）交换数据。这产生了现有单语言分析器无法分析的跨语言信息流通道。现有的混合应用分析方法只解决了部分问题：一些方法构建了感知桥接的调用图但信息流分析能力有限，一些方法依赖粗糙的假设，还有一些只关注Web漏洞。我们提出了MifDroid，一个由静态分析引导的动态分析框架，它包含生命周期感知的每WebView调用图、桥接感知的类型-效果系统以及模块化的JavaScript分析引擎。我们在F-Droid仓库中使用addJavascriptInterface的338个应用（总计4,126个）上评估了MifDroid，这些应用实现了Java和JavaScript之间的主动通信。我们的引导动态阶段在静态分析产生的444个条目基础上新增了662个条目，提升了分辨率……

    arXiv:2610.27288v1 Announce Type: new  Abstract: The WebView interface in Android (hybrid) applications enables the native side (e.g., Java) to exchange data with the Web (e.g., JavaScript). This creates cross-language information flow channels that existing single-language analyzers cannot analyze. Existing hybrid app analyses address only part of the problem: some build bridge-aware call graphs but provide limited information-flow analysis, some rely on coarse assumptions, and some focus only on web vulnerabilities. We present MifDroid, a static-analysis- guided dynamic-analysis framework with a lifecycle-aware per-WebView call graph, a bridge-aware type-and-effect system, and a modular JavaScript analysis engine. We evaluated MifDroid on 338 apps from the F-Droid repository (out of 4,126 total) that use addJavascriptInterface, enabling active communication between Java and JavaScript. Our guided dynamic phase adds 662 entries to the 444 produced by static analysis, raising the resol
    
[^36]: 为COBOL到Java迁移认证可重复的进入-回放见证

    Certifying Repeatable Entry-Replay Witnesses for COBOL-to-Java Migration

    [https://arxiv.org/abs/2610.27272](https://arxiv.org/abs/2610.27272)

    该论文提出基于按时间顺序外部操作带的进入-回放测试用例认证方法，在模拟外部交互的代理COBOL程序上生成合成端到端测试用例，使一个33,429行的大型COBOL程序达到78.1%的分支覆盖率和95.7%的段落覆盖率，从而支持COBOL到Java迁移的确定性验证。

    

    为迁移后的COBOL应用程序生成合成数据以进行比较测试，是当今智能体迁移面临的一项重大挑战。我们之前的论文《遗留代码迁移确定性验证的智能体方法》提出了一种促进此类迁移的确定性判定基准——Locksmith Loop。我们现在提出一种新颖的方法，通过在代理COBOL程序上观察代码执行轨迹，并利用按时间顺序记录外部交互的外部操作带，实现进入-回放测试用例的认证，从而借助生成的合成端到端测试用例实现高分支覆盖率和段落覆盖率。我们将该方法应用于一个拥有33,429行代码的超大型COBOL程序，生成了849个合成端到端测试用例，覆盖了2,973/3,806个源代码分支（78.1%）和989/1,033个主AST段落（95.7%）。利用Locksmith Loop的投影一致性验证，生成的Java代码与COBOL源代码相匹配……（摘要在此处被截断）

    arXiv:2610.27272v1 Announce Type: cross  Abstract: Producing synthetic data for comparison testing a migrated COBOL application is a major challenge that agentic migrations face today. Our prior paper, the Agentic Method for Deterministic Validation of Legacy Code Migration, presents a deterministic oracle to facilitate such migrations, the Locksmith Loop. We now propose a novel method for achieving high branch and paragraph coverage through generated synthetic end-to-end test cases, by means of entry-replay test-case certification by observing the code execution trace on a surrogate COBOL program that mocks external interactions via a chronological external-operation tape. We applied our method to a very large COBOL program with 33,429 lines of code to generate 849 synthetic end-to-end test cases covering 2,973/3,806 source branches (78.1%) and 989/1,033 main-AST paragraphs (95.7%). Using the Locksmith Loop's projected-agreement validation, the generated Java code matched the COBOL su
    
[^37]: InterType：打破神经二进制类型推断中的单函数壁垒

    InterType: Breaking the Single-Function Barrier in Neural Binary Type Inference

    [https://arxiv.org/abs/2610.26906](https://arxiv.org/abs/2610.26906)

    InterType将过程间分析嵌入预训练的Transformer模型中，通过利用跨函数传播的类型约束信息，打破了以往神经二进制类型推断局限于单函数内部的壁垒。

    

    变量类型恢复是二进制逆向工程中的一项关键任务，能够支撑漏洞发现、恶意软件分析和安全强制执行等下游应用。尽管近年来基于Transformer的模型通过学习指令级语义在类型推断中展现出强大性能，但它们主要局限于单函数边界内（即过程内分析），忽略了跨函数调用传播的丰富类型约束。实证研究表明，真实世界二进制文件中近44%的变量会作为参数或返回值跨越函数边界传播，这是过程内模型无法利用的重要类型信息来源。本文提出InterType，一种将过程间分析嵌入预训练Transformer模型以实现二进制类型恢复的方法。我们的方法构建于CodeArt——一个最先进的Transformer模型之上……

    arXiv:2610.26906v1 Announce Type: cross  Abstract: Variable type recovery is a critical task in binary reverse engineering, enabling downstream applications such as vulnerability discovery, malware analysis, and security enforcement. While recent Transformer-based models have demonstrated strong performance in type inference by learning instruction-level semantics, they predominantly operate within single-function boundaries (i.e. intra-procedural analysis), ignoring the rich type constraints that propagate across function calls. Empirical studies have shown that nearly 44% of variables in real-world binaries propagate across function boundaries as arguments or return values, representing a significant source of type information that intra-procedural models cannot exploit. In this paper, we propose InterType, a method that embeds inter-procedural analysis into a pre-trained Transformer model for binary type recovery. Our approach builds on CodeArt, a state-of-the-art Transformer model 
    
[^38]: 从过去的错误中学习：AI辅助的软件缺陷及其修复分析助力未来缺陷预防——一项工业案例研究

    Learning from Past Mistakes: AI-Assisted Analytics of Software Defects and Their Fixes for Future Defect Prevention - An Industrial Case Study

    [https://arxiv.org/abs/2610.26889](https://arxiv.org/abs/2610.26889)

    该论文提出一种AI辅助、人在回路的缺陷分析方法，通过分析历史缺陷及其修复数据识别重复缺陷的人为错误根源，并将其转化为预防措施和面向工程师的能力建设洞察，从而实现未来的缺陷预防与质量保证改进。

    

    背景：尽管软件测试和质量保证技术不断进步，缺陷在工业软件系统中仍持续反复出现。缺陷数据通常会被收集在问题跟踪系统中，但其用途大多局限于检测、报告和短期解决，历史缺陷及其修复中所蕴含的知识在缺陷预防和长期改进方面的价值尚未得到充分利用。目标：本文研究如何在人工监督下借助AI辅助，对历史缺陷及缺陷修复数据进行分析，以识别重复出现缺陷的人为错误根源，并将这种理解反馈到缺陷预防和质量保证改进之中。方法：我们开发了一种AI辅助、人在回路的分析方法，该方法按照故障类型和软件开发生命周期（SDLC）中的人为错误根源对缺陷进行分类，确定每个缺陷本可以如何被预防，并总结出面向工程师的能力建设洞察。该方法已被部署应用于……

    arXiv:2610.26889v1 Announce Type: new  Abstract: Context. Despite advances in software testing and quality assurance, defects continue to recur in industrial software systems. Defect data are routinely collected in issue-tracking systems, but their use is largely confined to detection, reporting, and short-term resolution, leaving the knowledge embedded in historical defects and their fixes under-exploited for prevention and long-term improvement. Objective. This paper investigates how historical defect and defect-fix data can be analyzed, with AI assistance under human oversight, to identify the human-error origins of recurring defects and to feed that understanding back into prevention and QA improvement. Method. We developed an AI-assisted, human-in-the-loop analytics approach that classifies defects by failure type and by human-error origin across the SDLC, determines how each could have been prevented, and synthesizes engineer-focused capability-building insights. It was deployed 
    
[^39]: 复杂度拐点：面向代码生成可靠性的提示侧结构复杂度指数

    The Complexity Kink: A Prompt-Side Structural Complexity Index for Code-Generation Reliability

    [https://arxiv.org/abs/2609.19616](https://arxiv.org/abs/2609.19616)

    该论文提出一个生成前评分的六维提示侧结构复杂度指数，发现代码生成通过率在复杂度分数上存在非单调断点，但该断点并非通用的失败临界值，会随任务类型固定效应等因素发生移动。

    

    从生成代码中测量的复杂度依赖于失败情况：一个困难的提示可能产生一个简短的失败程序，从而被赋予较低的输出复杂度。我们提出了一个六维度的提示侧结构复杂度指数，在生成之前进行评分，并与正确性保持分离。我们从初步单评分者量表的六个区间中选择了5,000个Python提示。四个面板外的LLM评分者对锁定的提示重新评分，得到19,997条评分记录；在四个评分齐全的4,998个提示上，综合评分者间信度为ICC = 0.872。我们对每个提示评估21个模型，共产生105,000个生成结果。在未经调整的均值池化分析中，通过率在综合分数13.75处出现非单调断点，该分数及以下的通过率为79.9%，以上为87.6%。这并非一个通用的失败临界点。任务类型固定效应将断点移至10.75，并将区间差距从7.6个百分点缩小至2.1个百分点。构建框架控制将其移至8.50，原始差距为……

    arXiv:2609.19616v1 Announce Type: new  Abstract: Complexity measured from generated code is failure-dependent: a difficult prompt can yield a short failing program and be assigned low output complexity. We introduce a six-dimension prompt-side structural-complexity index scored before generation and kept separate from correctness. We select 5,000 Python prompts across six bands of a preliminary single-rater rubric. Four out-of-panel LLM raters rescore the locked prompts, giving 19,997 score rows; composite inter-rater reliability is ICC = 0.872 on the 4,998 prompts with all four ratings. We evaluate 21 models per prompt, yielding 105,000 generations. In the unadjusted mean-pooled analysis, pass rate has a nonmonotone breakpoint at composite 13.75, with 79.9% at or below and 87.6% above. This is not a universal failure cutoff. Task-type fixed effects shift the breakpoint to 10.75 and cut the regime gap from 7.6 to 2.1 points. A construction-frame control shifts it to 8.50 with a raw gap
    
[^40]: 当模型编辑过多：论最小代码编辑的保真度

    When Models Edit Too Much: On the Fidelity of Minimal Code Edits

    [https://arxiv.org/abs/2609.04061](https://arxiv.org/abs/2609.04061)

    该研究揭示了前沿大语言模型在修复代码时普遍存在“过度编辑”问题（即使如GPT-5.5这样的强模型也不例外），并提出通过一条简单的保留指令即可显著减少不必要的代码改动、降低认知复杂度，同时提升修复准确率。

    

    大语言模型越来越多地被用于编辑现有代码，但仅仅正确是不够的：有用的修复还应当是最小化的、可审查的，并且忠实于原始实现。我们研究了“过度编辑”现象，即模型重写代码的范围超出修复缺陷所需的趋势。我们基于400个BigCodeBench问题构建了一个评估框架，通过向参考解答中注入受控的AST级（抽象语法树级）破坏，为每个修复任务提供一个已知的最小补丁。研究发现，在各类前沿大语言模型中，过度编辑现象普遍存在，即使是在GPT-5.5这样的强大模型中也是如此：高Pass@1可能与不必要的巨大编辑和新增认知复杂度并存。一条保留指令能够显著减少这种行为，将平均超额Levenshtein距离从0.195降至0.131，减少26.6%的新增认知复杂度，并使Pass@1提高2.3个百分点。然而，这些收益并非简单地源于更大的推理（摘要在此处被截断）

    arXiv:2609.04061v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to edit existing code, but correctness alone is not enough: useful repairs should also be minimal, reviewable, and faithful to the original implementation. We study over-editing, the tendency of a model to rewrite code beyond what is required to fix a bug. We construct an evaluation framework from 400 BigCodeBench problems by injecting controlled AST-level corruptions into reference solutions, giving each repair task a known minimal patch. Across frontier LLMs, over-editing is widespread even among strong models like GPT-5.5: high Pass@1 can coexist with unnecessarily large edits and added cognitive complexity. A preservation instruction substantially reduces this behavior, lowering average excess Levenshtein distance from 0.195 to 0.131, reducing added cognitive complexity by 26.6%, and increasing Pass@1 by 2.3 points. However, these gains do not simply follow from a larger reasoning 
    
[^41]: 对于编程智能体而言，代码仓库至关重要

    For Coding Agents, the Repository Matters

    [https://arxiv.org/abs/2606.28235](https://arxiv.org/abs/2606.28235)

    该研究发现编程智能体拉取请求的解决时长等结果在很大程度上取决于接收仓库本身的特性，且这种依赖程度甚至超过人类贡献，因此跨仓库汇总的指标无法公平地评判智能体表现。

    

    2026年7月，在OpenAI的一项内部评估中，约1200个本应隔离运行的AI智能体把一个共享的软件包服务器当作留言板使用，其中约700个智能体加入了对Hugging Face的攻击。自动评分系统对每个智能体单独评判其任务表现，而这场攻击则是通过智能体共享的基础设施协调进行的。编程智能体每天都在接收其拉取请求的软件仓库中与共享基础设施打交道。然而，实地研究仍然采用跨仓库汇总的结果（如合并率和合并时间）来评判编程智能体。我们探究了这些结果在多大程度上取决于接收仓库本身，以及这种依赖程度是否高于人类贡献。利用多层模型，我们将来自公共GitHub仓库AIDev数据集的智能体拉取请求与相同仓库中的人类拉取请求进行了比较。智能体拉取请求从提交到解决所需时长的变化中，约有五分之二取决于仓库因素。

    arXiv:2606.28235v2 Announce Type: replace-cross  Abstract: In July 2026, about 1,200 AI agents in an internal OpenAI evaluation, meant to run in isolation, used a shared package server as a message board, and about 700 of them joined an attack on Hugging Face. An automated scorer judged each agent on its own task, and the attack was coordinated through infrastructure the agents shared. Coding agents meet shared infrastructure every day, in the software repositories that receive their pull requests. Yet field studies still judge coding agents by outcomes pooled across repositories, such as merge rates and merge times. We asked how much these outcomes depend on the receiving repository, and whether they depend on it more than human contributions do. Using multilevel models, we compared agent pull requests from the AIDev dataset of public GitHub repositories with human pull requests in the same repositories. About two-fifths of the variation in how long agent pull requests take to be reso
    
[^42]: 解析GitHub Actions工作流安全扫描器

    Unpacking Security Scanners for GitHub Actions Workflows

    [https://arxiv.org/abs/2601.14455](https://arxiv.org/abs/2601.14455)

    本文首次对9个GitHub Actions工作流安全扫描器进行了系统性比较，从范围、检测能力、准确性和性能四个维度评估了它们的安全检测效果。

    

    GitHub Actions是一个广泛使用的平台，通过可配置的工作流来自动化软件项目的构建与部署。随着该平台日益流行，它也成为了软件供应链攻击的首选目标。这些攻击利用过度权限、版本不明确或缺乏制品完整性检查等弱点来破坏工作流。为应对这些攻击，目前已出现多种安全扫描器来帮助开发者加固其工作流。在本文中，我们首次对9个GitHub Actions工作流安全扫描器进行了系统性比较。我们从以下几个方面对它们进行比较：范围（针对哪些安全弱点）、检测能力（能检测出多少弱点）、准确性（其检测结果与人工确认的弱点的匹配程度）以及性能（扫描一个工作流所需的时间）。为了在共同的基础上比较这些扫描器，我们首先建立了一个包含10种……的分类体系。

    arXiv:2601.14455v3 Announce Type: replace  Abstract: GitHub Actions is a widely used platform to automate the build and deployment of software projects through configurable workflows. As the platform's popularity grows, it also becomes a target of choice for software supply chain attacks. These attacks exploit excessive permissions, ambiguous versions or the absence of artifact integrity checks to compromise the workflows. In response to these attacks, several security scanners have emerged to help developers harden their workflows. In this paper, we perform the first systematic comparison of 9 GitHub Actions Workflows security scanners. We compare them regarding scope (which security weaknesses they target), detection capabilities (how many weaknesses they detect), accuracy (how well their findings match manually confirmed weaknesses), and performance (how long they take to scan a workflow). In order to compare the scanners on a common ground, we first establish a classification of 10
    
[^43]: 用于基于主体建模的大语言模型：建模周期中的现有与潜在应用

    Large Language Models for Agent-Based Modelling: Current and possible uses across the modelling cycle

    [https://arxiv.org/abs/2507.05723](https://arxiv.org/abs/2507.05723)

    本研究回顾了大语言模型在基于主体建模中的现有应用，并沿着从问题构建到结果传播的完整建模周期，批判性地分析了其潜在应用的机遇与挑战。

    

    大语言模型（LLM）凭借日益成熟的自然语言理解与生成能力，引发了基于主体建模（ABM）领域的关注。由于其具备总结、生成、分析、分类、转录和翻译文本、回答问题、提出解释、维持对话、从非结构化文本中提取信息以及执行逻辑推理和问题解决任务的能力，大语言模型在建模过程中具有良好的应用潜力。本研究在回顾了大语言模型在基于主体建模中现有应用的基础上，对大语言模型在该领域潜在应用的机遇与挑战进行了反思。该研究沿着建模周期展开，涵盖从问题构建到模型结果的记录与传播的全过程，并始终保持批判性立场。

    arXiv:2507.05723v1 Announce Type: cross  Abstract: The emergence of Large Language Models (LLMs) with increasingly sophisticated natural language understanding and generative capabilities has sparked interest in the Agent-based Modelling (ABM) community. With their ability to summarize, generate, analyze, categorize, transcribe and translate text, answer questions, propose explanations, sustain dialogue, extract information from unstructured text, and perform logical reasoning and problem-solving tasks, LLMs have a good potential to contribute to the modelling process. After reviewing the current use of LLMs in ABM, this study reflects on the opportunities and challenges of the potential use of LLMs in ABM. It does so by following the modelling cycle, from problem formulation to documentation and communication of model results, and holding a critical stance.
    
[^44]: 大语言模型对简单代码变换十分脆弱：提出CETBench——一个代码等价性检测基准

    LLMs are Brittle to Simple Code Transformations: Introducing CETBench - A Benchmark for Code-Equivalence Checking

    [https://arxiv.org/abs/2506.04019](https://arxiv.org/abs/2506.04019)

    本文提出了代码等价性检测基准CETBench，发现大语言模型即使面对简单的代码变换也会出现显著性能下降（在跨语言场景下尤为严重），并提出一种基于微调的简单方法来提升模型性能。

    

    我们研究了大型语言模型（LLM）判断两个程序在功能上是否等价的能力。这是一个重要问题，因为对代码等价性进行基准测试有助于评估LLM在代码重写和代码翻译等任务上的能力。为此，我们提出了CETBench（Code Equivalence with Transformations Benchmark，基于变换的代码等价性检测基准），它由一个程序仓库构建而成，这些程序可能解决相同或不同的任务。每个数据集实例通过采样一对程序并应用随机序列的预定义代码变换来生成，从而得到等价或非等价的程序对。我们的分析表明，即使是简单的变换也会导致最先进的LLM在代码等价性检测任务上出现显著的性能下降。在跨语言场景下（即比较用不同语言编写的程序时），这些挑战会被进一步放大。为了解决这一问题，我们提出了一种简单的基于微调的方法来提升LLM在该任务上的性能。

    arXiv:2506.04019v2 Announce Type: replace-cross  Abstract: We study how well LLMs can determine whether two programs are functionally equivalent. This is an important problem because benchmarking code equivalence helps assess LLM capability in tasks such as code rewriting and translation. To this end, we introduce CETBench - Code Equivalence with Transformations Benchmark - built from a repository of programs that may solve the same or different tasks. Each dataset instance is created by sampling a program pair and applying a random sequence of predefined code transformations, yielding either equivalent or non-equivalent pairs. Our analysis shows that even simple transformations cause a significant performance drop in state-of-the-art LLMs on code-equivalence checking. These challenges are further amplified in the cross-lingual setting when comparing programs written in different languages. To remedy this, we present a simple fine-tuning-based approach to boost LLM performance on the t
    
[^45]: 语言模型能否替代程序员进行编码？REPOCOD的回答是“尚不能”

    Can Language Models Replace Programmers for Coding? REPOCOD Says 'Not Yet'

    [https://arxiv.org/abs/2410.21647](https://arxiv.org/abs/2410.21647)

    该论文提出了REPOCOD基准，包含来自11个真实热门项目的980个Python完整函数生成任务，以评估语言模型在真实世界编码任务中的能力，结果显示当前语言模型尚不能替代程序员。

    

    最近，许多仓库级代码生成基准（如CoderEval、DevEval、RepoEval、RepoBench和LongCodeArena）相继出现，用于评估大型语言模型（LLM）在HumanEval和MBPP等独立基准之外的能力。由此产生一个自然的问题：LLM在真实世界编码任务中的表现是否会与其在这些基准上的表现相当？遗憾的是，目前无法回答这个问题，因为这些基准由简短的补全任务、合成示例组成，或仅关注规模有限的仓库，无法代表真实世界的编码任务。为解决这些挑战，我们创建了REPOCOD，这是一个Python代码生成基准，包含来自真实世界大型项目中具有现实依赖关系的复杂任务，并配有适合评估源代码的指标。该基准包含来自11个热门项目的980个完整函数生成任务，其中50.8%的任务需要仓库级上下文信息。

    arXiv:2410.21647v5 Announce Type: replace-cross  Abstract: Recently, a number of repository-level code generation benchmarks-such as CoderEval, DevEval, RepoEval, RepoBench, and LongCodeArena-have emerged to evaluate the capabilities of large language models (LLMs) beyond standalone benchmarks like HumanEval and MBPP. Thus, a natural question is, would LLMs have similar performance in real world coding tasks as their performance in these benchmarks? Unfortunately, one cannot answer this question, since these benchmarks consist of short completions, synthetic examples, or focus on limited scale repositories, failing to represent real-world coding tasks.   To address these challenges, we create REPOCOD, a Python code-generation benchmark containing complex tasks with realistic dependencies in real-world large projects and appropriate metrics for evaluating source code. It includes 980 whole-function generation tasks from 11 popular projects, 50.8% of which require repository-level contex
    
[^46]: WAFFLE：面向自动化前端开发的多模态模型微调

    WAFFLE: Finetuning Multi-Modal Models for Automated Front-End Development

    [https://arxiv.org/abs/2410.18362](https://arxiv.org/abs/2410.18362)

    Waffle提出了一种结合结构感知注意力机制与对比微调的多模态模型微调策略，显著提升了从UI设计图自动生成HTML代码的效果。

    

    网页开发涉及将UI设计转化为功能性的网页，由于HTML的层级结构和样式较为复杂，无论对初学者还是有经验的开发者来说都颇具难度。虽然大型语言模型（LLMs）在生成源代码方面已展现出潜力，但在UI到HTML代码生成中仍存在两大挑战：（1）如何有效地为LLM表示HTML的层级结构，以及（2）如何弥合UI设计的视觉特性与HTML代码文本格式之间的鸿沟。为应对这些挑战，我们提出了Waffle，这是一种新的微调策略，它采用结构感知注意力机制来增强LLM对HTML结构的理解，并使用对比微调方法来对齐LLM对UI图像和HTML代码的理解。经Waffle微调后的模型表现出显著提升：HTML匹配度最高提高9.00个百分点，CW-SSIM提高0.0982，CLIP提高32.99，另一指标提高27.12个百分点。

    arXiv:2410.18362v4 Announce Type: replace-cross  Abstract: Web development involves turning UI designs into functional webpages, which can be difficult for both beginners and experienced developers due to the complexity of HTML's hierarchical structures and styles. While Large Language Models (LLMs) have shown promise in generating source code, two major challenges persist in UI-to-HTML code generation: (1) effectively representing HTML's hierarchical structure for LLMs, and (2) bridging the gap between the visual nature of UI designs and the text-based format of HTML code. To tackle these challenges, we introduce Waffle, a new fine-tuning strategy that uses a structure-aware attention mechanism to improve LLMs' understanding of HTML's structure and a contrastive fine-tuning approach to align LLMs' understanding of UI images and HTML code. Models fine-tuned with Waffle show up to 9.00 pp (percentage point) higher HTML match, 0.0982 higher CW-SSIM, 32.99 higher CLIP, and 27.12 pp higher
    
[^47]: 通过有效的需求工程缓解软件开发中的风险

    Mitigating Risks in Software Development through Effective Requirements Engineering

    [https://arxiv.org/abs/2305.05800](https://arxiv.org/abs/2305.05800)

    本文通过综述需求获取、分析与协商、规格说明、验证和变更管理等核心需求工程活动，并比较不同开发生命周期对需求的处理方式，说明如何利用有效的需求工程来降低软件项目中因需求缺陷导致的延期、超支和项目取消等风险。

    

    源于需求的问题在软件项目中十分常见且代价高昂，它们是导致交付延期、预算超支和项目取消的一个反复出现的原因。本文是一篇关于需求工程（RE）的叙述性综述，将其视为降低这些风险的一种手段。文章概述了需求工程的核心活动（需求获取、分析与协商、规格说明、验证以及变更管理），以及需求文档中所包含的各类陈述，包括需求、领域属性、假设与定义之间的区别。本文回顾了关于需求缺陷成本的证据以及对这些证据的主要批评，并比较了瀑布式、螺旋式、迭代式（Rational统一过程）和敏捷生命周期对需求的处理方式。随后，文章讨论了利益相关者的识别、需求获取技术以及获取过程中的常见障碍，并描述了将模糊或混淆的陈述（转化为精确需求的实用方法）。

    arXiv:2305.05800v2 Announce Type: replace  Abstract: Problems that originate in requirements are common and expensive in software projects, and they are a recurring cause of late delivery, budget overruns and cancellation. This paper is a narrative overview of requirements engineering (RE) as a means of reducing these risks. It outlines the core RE activities (elicitation, analysis and negotiation, specification, validation and change management) and the kinds of statements a requirements document contains, including the distinction between requirements, domain properties, assumptions and definitions. It reviews the evidence on the cost of requirements defects, together with the main criticisms of that evidence, and compares how the waterfall, spiral, iterative (Rational Unified Process) and agile lifecycles treat requirements. It then discusses stakeholder identification, elicitation techniques and common obstacles to elicitation, and describes practical ways of turning vague or confl
    

