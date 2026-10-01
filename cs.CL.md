# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ranking-Aware Prompt Optimization for Multimodal Clinical Diagnosis](https://arxiv.org/abs/2609.40361) | 针对临床数据类别不平衡导致准确率指标失效的问题，本文提出以AUROC为优化目标的成对级别帕累托提示进化方法，显著提升了多模态大语言模型在临床诊断中的提示优化效果。 |
| [^2] | [Semifactual Credit-Augmented Policy Optimization](https://arxiv.org/abs/2609.40360) | 提出SCAPO，一种受因果启发的GRPO改进方法，通过半事实提示干预分析词元敏感性，并将半事实稳定性融入词元级信用分配，从而降低LLM对任务无关提示特征的依赖并提升推理准确性。 |
| [^3] | [EvoDuet: Bilevel Co-Evolution of Web Searching and Task Solving for Scientific Discovery](https://arxiv.org/abs/2609.40340) | EvoDuet提出了一种双层协同进化优化方法，在固定LLM参数的情况下共同进化搜索查询与任务解，并通过检索门控动态决定检索新文档、复用旧文档或直接继续，从而显著提升了科学发现类任务的进化搜索性能。 |
| [^4] | [MatLoom: Layered Text-to-Material Generation in a Compact Program Space](https://arxiv.org/abs/2609.40322) | MatLoom提出一种紧凑的面向图层的材质程序语言，配合预训练语言模型，通过解析器引导修复与预览评审实现免微调的文本到材质生成，其最佳配置在141个提示词的基准上超越了三个扩散基线。 |
| [^5] | [Scaling Laws for Looped Mixture of Experts](https://arxiv.org/abs/2609.40316) | 本文提出首个联合建模循环与稀疏性的缩放定律——“循环缩放定律”，它能更准确地预测循环MoE模型的性能损失，并为在计算与内存约束下设计循环混合专家模型提供了理论依据。 |
| [^6] | [How Much Is an AI Token Worth? Scaling Laws for Wild AI-Generated Web Text](https://arxiv.org/abs/2609.40295) | 本文通过预训练800个语言模型并拟合缩放定律，首次系统量化了“野生”AI生成网络文本对预训练的影响：对数据匮乏的模型，少量AI文本初期有益但收益迅速饱和并逆转为损害，而对人类文本充足的模型则几乎立即有害。 |
| [^7] | [Linguistic Loopholes in LLM Unlearning: From a 174-Language Benchmark to Coverage-Aware Unlearning](https://arxiv.org/abs/2609.40286) | 该论文揭示了大语言模型遗忘的跨语言漏洞，提出了覆盖174个语言-文字对的跨语言遗忘基准，并设计了COVER方法，在有限语言预算下智能选择源语言进行遗忘，以最大化对全部语言的跨语言知识擦除效果。 |
| [^8] | [cua-speedrun: Standardized Benchmarking of the Speed of Computer-Use Agents](https://arxiv.org/abs/2609.40284) | 提出了cua-speedrun基准，通过标准化的基础设施、统一的虚拟机设置和执行流程解决现有基准测试的可重复性危机，从而可靠地评估计算机使用智能体的速度与效率。 |
| [^9] | [Decision-Oriented Recommendation Reranking: An Empirical Study of Jev](https://arxiv.org/abs/2609.40241) | 该实证研究表明，TypeSafe AI提出的决策导向模型Jev在个性化推荐重排序中既能保持与基线模型相当的推荐效果，又比逐点式Qwen重排序器具有更平缓的服务延迟增长，为LLM重排序在质量与效率之间的权衡提供了有价值的替代方案。 |
| [^10] | [Comparison of techniques for fine-tuning open-weight models for entity extraction from radiology reports](https://arxiv.org/abs/2609.40236) | 该研究通过2×2实验设计比较了不同微调策略与训练数据来源，发现基于真实GPT-4o标注报告蒸馏的指令微调Gemma-3-12B模型在颅内出血急性度抽取任务上可媲美GPT-4o，为隐私友好、成本可控且可重复的放射报告结构化提供了开源替代方案。 |
| [^11] | [Distribution Matching Distillation for Continuous Diffusion Language Models](https://arxiv.org/abs/2609.40235) | 提出了两种分布匹配蒸馏方法Simplex-DMD和Reinforce-DMD，通过利用学生模型的概率性词元输出，将连续扩散语言模型的网络评估次数大幅降低至仅需4次即可实现高质量文本生成。 |
| [^12] | [PhantomEnvironments: Training LLM Agents in Fictional Worlds](https://arxiv.org/abs/2609.40221) | 提出完全由规则生成、零边际成本的虚构世界多轮RL环境PhantomEnvironments，无需任何真实世界事实即可训练LLM智能体成为强大的搜索智能体，并能迁移到真实世界多跳搜索基准，性能往往超过使用真实训练数据的结果。 |
| [^13] | [SCB: SpeechConversationBench for Evaluating Multi-Turn Reasoning in Speech-to-Speech Models](https://arxiv.org/abs/2609.40198) | 本文提出SCB基准，通过将GSM8K数学题分片并跨多轮语音对话逐步披露来评估语音到语音模型的多轮推理能力，发现现有商用系统在分片条件下准确率显著下降，而具备显式上下文管理的LEGO语音管道表现稳定。 |
| [^14] | [MemLife: Curating and Reasoning over Long-Term Egocentric Video Memories](https://arxiv.org/abs/2609.40195) | MemLife是一种多模态记忆系统，通过构建基于实体的第一人称文本情节并用时间索引的智能体读取器进行检索，在无需训练和查询时视频访问的情况下，在四个长时程基准上比最强免训练基线提升4.6%–12.0%，解决了长期第一人称视频中记忆证据丢失和检索竞争的问题。 |
| [^15] | [Cheap to Draw, Expensive to Trust: Certifying Test-Time Scaling Curves](https://arxiv.org/abs/2609.40190) | 该论文推导了统计认证整条测试时扩展曲线的极小极大采样成本，并证明利用“基准是固定题目列表、方差主要来自题目之间”这一结构，可以避免为错误的不确定性买单，从而大幅降低同时认证所有预算所需的生成样本量。 |
| [^16] | [Provably Tractable NFA-Constrained Language Generation via HMMs](https://arxiv.org/abs/2609.40185) | 该论文提出了NFA-LM，一个基于#NFA问题FPRAS理论的多项式时间生成引擎，首次在温和假设下以理论保证高效解决NFA约束语言生成问题，克服了现有方法扭曲分布或牺牲效率的缺陷。 |
| [^17] | [Index-Translate: A Multilingual Translation Model Family -- Text, Speech, Controlled Dubbing, and Long-Document Translation](https://arxiv.org/abs/2609.40181) | Index-Translate是一个支持150种语言的多语言翻译模型家族，统一覆盖文本、语音、可控配音和长文档翻译，以较小规模实现了与千亿级翻译模型和前沿模型相当的性能。 |
| [^18] | [Learning Functional Subspaces for Neural Network Compression](https://arxiv.org/abs/2609.40127) | 本文提出可学习子空间投影方法，通过端到端联合优化正交投影器并基于全局目标（KL散度或原始训练损失）来学习神经网络权重矩阵中应丢弃的子空间，解决了传统局部闭式准则忽略误差传播导致高压缩率下性能崩溃的问题。 |
| [^19] | [Debias It Yourself: Teaching LLMs Cognitive Bias Mitigation Interventions](https://arxiv.org/abs/2609.40124) | 该论文提出以认知科学为基础的 DIY 框架，将五种经过验证的去偏干预方法通过 Show（上下文示例）、Train（指令微调）、Revise（自我修订）三种范式应用于大语言模型，在偏差-推理权衡中取得最优表现，并在未见维度上将偏差最多降低 14.8%。 |
| [^20] | [On the (In)effectiveness of AMR Augmentation for Large Language Models](https://arxiv.org/abs/2609.40121) | 研究通过复现实验和基于困惑度的探测方法发现，AMR增强对现代大语言模型无效，纯文本基线的性能始终能够匹配或超越AMR增强模型。 |
| [^21] | [Persistent Context Graphs for Efficient Memory Compaction in LLM Agents](https://arxiv.org/abs/2609.40118) | 提出ReCAP方法，通过在轻量级持久化上下文图中存储注意力推导的重要性分数和依赖链接，结合新请求的相关性线索，实现LLM智能体高效且低开销的内存压缩。 |
| [^22] | [Agent Error Dataset: Scaling 50,000 Error--Diagnosis Pairs for Failure Analysis and Error-Aware Post-Training](https://arxiv.org/abs/2609.40111) | 该论文发布了包含50,228个错误-诊断对、覆盖33个环境和23个策略模型的智能体错误数据集（AED），并配套五阶段AET流水线，用于LLM智能体的失败分析与错误感知后训练。 |
| [^23] | [OverdoseMoE: A Multi-Expert Framework for Opioid Overdose Risk Prediction](https://arxiv.org/abs/2609.40108) | 本文提出了OverdoseMoE多专家框架，通过对纵向ICD诊断序列进行诊断特异性继续预训练与微调，并利用互补专家加权策略整合不同规模的模型，显著提升了180天阿片类药物过量风险预测性能（AUPRC达25.17，AUROC达69.49）。 |
| [^24] | [JuryFlow: Disagreement-Guided Human-in-the-Loop Multi-Agent Evaluation](https://arxiv.org/abs/2609.40103) | JuryFlow是一个分歧引导的人机协同多智能体评估框架，它将LLM评审器之间的分歧视为声明级的不确定性信号，通过将响应分解为原子声明、构建熵评分的分歧图，并由人类以单次最小干预精准化解评审冲突。 |
| [^25] | [AutoDataBench: A Data-centric Testbed for Accelerating Auto Research](https://arxiv.org/abs/2609.40097) | AutoDataBench是一个隔离数据因素的可控测试平台，首次系统评估大语言智能体的“数据智能”，即通过数据诊断、组织与构建迭代改进训练数据的能力。 |
| [^26] | [From Tweets to Trades: Analyzing the Influence of Public Mood over Stock Market Performance in Turkiye](https://arxiv.org/abs/2609.40064) | 该研究基于61万余条X平台帖子，利用微调的土耳其语Transformer模型构建分领域的公众情绪指标，并采用多种计量经济学方法考察其与土耳其BIST100/BIST30股市表现之间的关联，发现公众情绪与市场动态的关系因传播领域和市场状况而异。 |
| [^27] | [LARC: Low-Rank Adaptive Residual Connections for Learning in Frozen Models](https://arxiv.org/abs/2609.40063) | LARC 通过在冻结模型中引入低秩自适应残差连接，利用跨任务学习初始因子的慢状态与随反馈动态调整的快状态相结合的双状态机制，使冻结模型无需更新原始参数即可从反馈中学习，在程序选择任务中显著降低查询执行误差。 |
| [^28] | [MGhana-ST: A Low-Resource Speech Translation Dataset for Ghanaian Languages and an Analysis of Multilingual Training Trade-offs](https://arxiv.org/abs/2609.40041) | 该论文发布了面向加纳四种低资源语言的语音翻译数据集MGhana-ST，并发现在严重数据稀缺的场景下，多语言联合训练相比单语言训练并无收益，甚至会显著降低埃维语和芳蒂语的翻译性能。 |
| [^29] | [OPTS-TTPO: Enhancing Finite-Sample Policy-Gradient Learning with Tree Search](https://arxiv.org/abs/2609.40035) | 提出 OPTS 并行树搜索与 TTPO 树轨迹策略优化方法，通过在已访问状态处从当前策略采样新后缀来构建同策略树轨迹，在固定预算内提升策略梯度学习对罕见高回报轨迹的覆盖，且无需动作分布校正。 |
| [^30] | [Mid-Harness: Scaling Actions Between Model and Harness for Terminal Agents](https://arxiv.org/abs/2609.39982) | Mid-Harness 通过在模型与执行框架之间对候选动作进行采样并借助强验证器筛选，在不改变模型和框架的情况下提升终端智能体动作执行的可靠性与成功率。 |
| [^31] | [Overview of BioASQ 2026: The fourteenth BioASQ Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering](https://arxiv.org/abs/2609.39975) | 第十四届BioASQ挑战赛在CLEF 2026框架下设置了涵盖生物医学问答、多语言临床摘要、嵌套实体关系抽取、心脏病学临床编码和肠-脑信息抽取等六项共享任务，吸引了87支团队提交超过1000个参赛结果，持续推动生物医学语言处理领域的发展。 |
| [^32] | [UBTree: Parallel Tree Drafting via Unigram and Bigram Models for Speculative Decoding](https://arxiv.org/abs/2609.39972) | UBTree通过将采用标准交叉熵训练的一元提议器与在高温度数据上以重归一化KL目标训练的轻量级二元选择器相结合来构建并行草稿树，从而在不牺牲并行性的前提下提升投机解码在高熵目标分布下的表现。 |
| [^33] | [LEAP: Learned Block-wise Evidence Retrieval for Long Audio-Video Perception](https://arxiv.org/abs/2609.39938) | LEAP通过分块式轻量级证据定位与检索，使小时级音视频问答的回答输入和上下文长度与录音时长无关，同时保留细粒度的声学与视觉证据。 |
| [^34] | [RoPE at the End of Its Rope? Theory, Diagnosis, and Mitigation of Long-Context Failures](https://arxiv.org/abs/2609.39929) | 该论文通过允许RoPE各频率上query-key缩放不相等的新理论，使RoPE的语义稳定性与位置敏感性权衡在注意力头和输入层面变得可度量，推导出上下文长度的理论界限，并据此提出针对长上下文失效的诊断与缓解方法。 |
| [^35] | [AdaGEPA: Adaptive Feedback Allocation for Reflective Prompt Optimization](https://arxiv.org/abs/2609.39927) | AdaGEPA提出了一种自适应反馈分配方法，根据提示词的性能和已识别的弱点来智能选择反馈示例，在相同计算预算下比非自适应反馈选择方法获得更高的验证分数。 |
| [^36] | [MCD: Causal Distillation of Multimodal In-Context Learning in Large Vision-Language Models](https://arxiv.org/abs/2609.39920) | 提出了多模态因果蒸馏（MCD）框架，通过结构保持的token干预识别并验证因果证据，将大型视觉语言模型教师在ICL中利用多模态证据的方式蒸馏给小模型，避免学生模型依赖语言先验等虚假线索。 |
| [^37] | [The Concrete-Arbitrary Gap: Kinship Reasoning in LLMs Is Not Indifferent to Presentation](https://arxiv.org/abs/2609.39913) | 大语言模型在解决用熟悉词汇表达的亲属关系问题时，准确率显著高于用明确定义的临时谓词表达的等价问题，且这一“具体-任意”差距可通过推理预算和提示语言干预大幅缩小，说明模型的关系推理能力依赖于呈现方式，但并非固定缺陷。 |
| [^38] | [OPSRD: On-Policy Self-Role Distillation](https://arxiv.org/abs/2609.39884) | OPSRD 让无角色的学生模型自行生成轨迹，再由同一基座模型的冻结角色化副本在这些前缀上给出角色条件分布，并用教师加权的正向 KL 散度蒸馏学生低估的备选词元，从而在无需参考解答的情况下实现在策略的角色知识自蒸馏。 |
| [^39] | [LLM Persona Unlearning](https://arxiv.org/abs/2609.39882) | 该论文提出“人格遗忘”任务及PersonaUnlearnBench基准，通过权重级编辑使大语言模型中的指定人格难以被诱发，并发现标准遗忘方法无法在不牺牲生成质量或通用能力的情况下可靠地抹除目标人格。 |
| [^40] | [GrammarRL: Effective Grammar-Constrained Decoding via Reinforcement Learning](https://arxiv.org/abs/2609.39869) | GrammarRL 是一种无需标注数据的强化学习方法，通过基于模型自身似然的直接奖励与反向奖励，使语言模型在满足语法约束的同时保持语义质量。 |
| [^41] | [FIGS: Evaluating Multi-Turn Sycophancy Without Penalizing Empathy](https://arxiv.org/abs/2609.39863) | 提出了FIGS双轴评估框架，通过自适应的10轮真实多轮对话来评估大语言模型的谄媚行为，在不将共情误判为屈从的前提下区分事实完整性与支持性表达。 |
| [^42] | [Cognitive Enhancement: Rethinking the Necessity of Role-Playing for Large Language Models](https://arxiv.org/abs/2609.39853) | 该研究通过多模型、跨领域、多语言实验发现角色扮演提示的收益取决于模型容量、知识领域和提示语言，并基于元认知理论提出“角色相关认知对齐假设”——只有当大语言模型正确理解指定角色及其知识领域时，角色扮演才能有效提升性能。 |
| [^43] | [When a Kindergartener Solves Calculus: Measuring Capability Leakage in Role-Prompted Reasoning Models](https://arxiv.org/abs/2609.39846) | 本文提出RoleCapBench基准，揭示被角色提示的推理模型虽然能在语言风格上 convincing 地扮演分配的角色（如幼儿园学生），但其实际能力仍会“泄漏”至远超角色水平的程度（如专家级解微积分），始终无法将底层能力与角色保持一致。 |
| [^44] | [Learning Steganography Is Easy, Learning Steganographic Reasoning Is Hard](https://arxiv.org/abs/2609.39838) | 该论文通过对比强化学习、上下文学习和监督微调三种引导方法发现，模型较容易学会隐写传信和编码推理这两种相邻能力，而学会完全隐藏推理过程的隐写式推理则困难得多（通常仅在监督微调下才能习得），这对依赖思维链监控的AI安全方案具有重要意义。 |
| [^45] | [Synthetic Pre-pretraining Survives Scale, but Not as a Grammatical Prior](https://arxiv.org/abs/2609.39827) | 该研究首次在 500M 至 7B 参数规模、多种数据混合及高达 100B token 预算下系统验证了合成数据预预训练（PPT）的有效性，发现其节省 token 的收益在规模化下依然显著（如 3B 规模下节省至少 21B token），但推翻了先前将其增益归因于语法先验的解释。 |
| [^46] | [Stress-Testing LLM Lie Detectors: Role-Play Failures and Spurious Correlations](https://arxiv.org/abs/2609.39807) | 本研究通过构建包含8,916条人工审核响应的数据集对现有LLM谎言检测探针进行压力测试，发现许多探针在反事实人格角色扮演场景下失效，容易被虚假相关性所误导。 |
| [^47] | [Explore-on-Graph: Hybrid Embedding-LLM Reasoning for Knowledge Graph Question Answering under Incompleteness](https://arxiv.org/abs/2609.39786) | 提出XoG框架，通过结合知识图谱嵌入与类型级实体-关系统计从图结构本身恢复不完整知识图谱中缺失的推理路径，并让LLM仅充当语义选择器和推理器，从而在避免幻觉的同时实现可靠的多跳问答。 |
| [^48] | [MemCodex: Self-Programming Hierarchical Memory for Language Agents](https://arxiv.org/abs/2609.39765) | 提出MemCodex，一种通过开放式程序演化实现自编程的分层记忆系统，能够将经验组织为可执行记忆程序，并根据查询需求自适应地调整记忆的构建、检索与跨层组合方式。 |
| [^49] | [LatentHarness: Learning Latent Actions for Memory and Reasoning via Counterfactual Policy Distillation](https://arxiv.org/abs/2609.39740) | LatentHarness 将记忆访问与潜在推理统一为 THINK、RECALL、EXIT 三种潜在动作的顺序选择，并通过反事实策略蒸馏训练该策略，教会模型何时调用记忆比继续推理更有用以及应保留哪些中间状态。 |
| [^50] | [OverForge: Reasoning Through Strategies and Tactics Helps Cooperative Lifelong Adaptation](https://arxiv.org/abs/2609.39727) | OverForge提出了一种免训练的分层架构，通过将持久的协调策略推理与战术行动执行分离，并利用元认知“前额叶皮层”模块在策略-行动分支上进行想象与承诺，使协作型语言模型智能体在OvercookedV2中表现远超扁平基线，并能与陌生伙伴达成角色协调。 |
| [^51] | [Drift Inspector: Exploring and Measuring Scientific Drift with Atomic Contribution Claims](https://arxiv.org/abs/2609.39710) | 该论文提出了开源系统 Drift Inspector，通过让大语言模型从摘要中提取原子贡献声明并跨年份聚类成可交互追溯的可视化地图，从而精确测量研究领域的演变，揭示了 NLP 领域从经典任务向 LLM 时代能力（如推理和多模态）的转变。 |
| [^52] | [A helps B while B hurts A: directed transfer in instruction-tuning mixture](https://arxiv.org/abs/2609.39702) | 该论文发现指令微调中的任务迁移是有方向性的（A可以帮助B而B却损害A），并提出“迁移图”这一带符号估计方法来预测每个源任务对目标任务的增益或损害，从而在固定预算下更高效地指导训练任务的选择。 |
| [^53] | [ShieldCLIP: Selective Safety Alignment for Harmful Content Mitigation in Multimodal Foundation Models](https://arxiv.org/abs/2609.39688) | ShieldCLIP首次根据每个模态的实际安全状态而非样本来源进行选择性安全对齐，并配套提出带独立按模态安全标签的19.5万四元组数据集ViSUv2，在缓解多模态基础模型有害内容的同时保留良性表示。 |
| [^54] | [Better Supervision Is Nearby: Neighborhood On-Policy Self-Distillation](https://arxiv.org/abs/2609.39687) | 该论文提出邻域在策略自蒸馏（N-OPSD），通过离线贪婪选择构建一个紧凑的冻结专家池，并在在线阶段根据每个参考位置动态路由选择最合适的专家，将局部参数扰动带来的互补参考对齐纠正转化为学生访问状态处的更强监督信号，从而提升数学推理模型的自蒸馏训练效果。 |
| [^55] | [The Evolution of Attention in Large Language Models: Mechanisms, Trade-offs, and Emerging Trends](https://arxiv.org/abs/2609.39661) | 该综述将大语言模型中各类注意力改进方法统一视为“模型内部上下文记忆”，提出由记忆表示、记忆更新、访问、读取与整合构成的五维分析框架，并基于14个模型谱系的59条发布记录系统梳理了注意力机制的演进、权衡与新兴趋势。 |
| [^56] | [SEPAL: Separated Expert Pairs with Answer-Level Fusion for Reliable LLM Collaboration](https://arxiv.org/abs/2609.39645) | SEPAL 通过三个独立训练的 Actor-Critic 专家团队分别负责推理、证据验证和校验，并仅在最终答案层面进行多数投票融合，避免共享讨论导致的错误传播，从而显著提升大语言模型协作问答的可靠性和准确率。 |
| [^57] | [Zero-Compute Cross-Lingual Transferability Estimation Using Typological Feature Proxies](https://arxiv.org/abs/2609.39640) | 仅利用免费的语言类型学特征训练随机森林，即可在零计算成本下准确预测跨语言迁移性（留一语言 ρ=0.705、R²=0.49），证明类型学数据库蕴含廉价而密集的迁移信号，可替代昂贵的多语言预训练测量。 |
| [^58] | [Marginal Response Surface Elicitation for Zero-Label Tabular Learning](https://arxiv.org/abs/2609.39639) | 提出MARS方法，将LLM的特征级先验离线转化为可复用的零样本表格分类器，在八个表格基准任务上无需进一步查询LLM即超越直接提示法。 |
| [^59] | [Is This Evidence Decision-Critical? Learning to Verify Rule-Governed Decisions](https://arxiv.org/abs/2609.39608) | 提出了InterPact框架，利用基于干预的反事实方法来验证规则决策中证据的关键性，识别可能颠覆决策结果的关键证据，从而支持准确且安全的规则化决策。 |
| [^60] | [Thinking Outside the Box: Can Language Models Rely on External Guidance Selectively?](https://arxiv.org/abs/2609.39578) | 该论文提出 Box²-Bench 基准来衡量语言模型“跳出思维定式”的能力，即在受益于可靠工作流引导的同时否决不可靠引导，并发现反事实监督微调与基于结果的强化学习是提升这一能力的两种互补训练策略。 |
| [^61] | [Compact Language, Complex Model Shifts: How and Where Ambiguity and Underspecification Affect LLMs](https://arxiv.org/abs/2609.39572) | 本文通过构建人工同音异义词与人工上位词伪词，首次从机制层面揭示了词汇歧义与欠明性如何按比例提升语言模型性能、降低含歧义文本的生成准确率，并在模型内部表征中呈现消歧过程。 |
| [^62] | [Speculative Safety Honeypot: Toward Proactive Defense Against Multi-turn Agent Attacks](https://arxiv.org/abs/2609.39549) | 本文提出推测性安全蜜罐（SSH）框架，利用小型LLM多智能体模拟系统预测目标智能体的未来行为并构建轨迹树，实现对多轮智能体攻击的主动防御，同时通过真实动作验证有效降低误报。 |
| [^63] | [CATCH: A Controllable Analysis Testbed for Reward Hacking in Coding RL](https://arxiv.org/abs/2609.39533) | 提出 CATCH 测试平台，通过刻意暴露环境漏洞并以独立审计生成黄金标签，实现对编程强化学习中奖励破解行为的可控复现、可靠识别与系统干预研究。 |
| [^64] | [Spike-driven Vision-Language-Action Model](https://arxiv.org/abs/2609.39514) | 提出了首个支持机器人操作端到端直接训练的脉冲驱动视觉-语言-动作（VLA）框架，利用脉冲神经网络的稀疏事件驱动计算和多赢家脉冲融合机制，解决了传统大型Transformer模型延迟高、能耗大而难以在资源受限平台部署的问题。 |
| [^65] | [When the Right Answer Is Missing: An Arithmetic-Dependent Rejection Bottleneck in Jev](https://arxiv.org/abs/2609.39496) | 本研究揭示了 Jev 类型化决策模型存在依赖算术的拒绝瓶颈：在算术问题上答案存在时准确率高达 99%，但答案缺失时即便设有明确拒绝选项，正确拒绝率仅 7%，而原生布尔验证在同样场景下却能实现 99% 的精确匹配准确率。 |
| [^66] | [Right-Wing Rock or Just Rock? A Computational Linguistic Analysis of Frei.Wild](https://arxiv.org/abs/2609.39460) | 本研究通过构建德国摇滚与右翼摇滚语料库并训练高性能分类器，对备受争议的乐队Frei.Wild进行计算语言学分析，发现其虽刻意维持政治立场的模糊性，但整体倾向右翼，且超过一半的歌曲被识别为右翼摇滚。 |
| [^67] | [From Speech to Editable Concepts: Probing Emotion Recognition with Concept Bottleneck Models](https://arxiv.org/abs/2609.39453) | 本工作首次将概念瓶颈模型引入语音情感识别，通过转录文本、声学描述和说话人属性等可编辑概念探究大语言模型预测的依赖因素，揭示并量化了零样本设置下模型对转录文本的强烈偏向，从而提升了情感识别的可解释性。 |
| [^68] | [Synthetic Data Characterization via Training Dynamics](https://arxiv.org/abs/2609.39447) | 该论文提出利用样本级可学习性与编码器训练动力学来刻画LLM合成数据的特性，揭示不同LLM系列和规模之间的数据差异，并证明基于可学习性信号的数据选择策略对合成数据与人类数据会产生不同的效果。 |
| [^69] | [DuplexAct-Bench: Broadening Full-Duplex Speech Evaluation toward Proactive Interaction across Diverse Behavioral Requirements](https://arxiv.org/abs/2609.39446) | DuplexAct-Bench是一个双语全双工语音基准，系统覆盖打断、退让、主动发起、主动沉默、附和等六种行为及多种情境条件，通过对12个系统在1,290个流式试验中的时机与内容评估，揭示了现有系统在实时交互行为管理上的显著不足。 |
| [^70] | [QuantCode Model: Specializing Language Models for Executable Algorithmic Trading Code](https://arxiv.org/abs/2609.39420) | 该论文提出通过对算法交易框架代码进行持续预训练并结合经智能体验证的监督微调，将大语言模型专业化以生成可执行、语义忠实的算法交易代码，并发布了包含 400 个任务的 QuantCode-Bench 基准，显著提升了策略代码生成的通过率。 |
| [^71] | [Can Computation from Earlier Problems Help LLMs Solve New Ones?](https://arxiv.org/abs/2609.39394) | 提出 STAIR 方法，通过固定存储库复用早期响应的键值信息并仅训练 12,288 个参数，就能让大语言模型在多轮对话中利用先前问题的计算来提升解决新问题的准确率。 |
| [^72] | [TTLab at Daleel 2026: STAR-Ar, Sequence Tagging for Argument Recognition in Arabic](https://arxiv.org/abs/2609.39385) | 本文提出STAR-Ar，一种融合上下文Transformer嵌入与结构化转移约束的BERT-BiLSTM-CRF序列标注架构，用于阿拉伯语论辩话语单元的检测与分类，在Daleel 2026共享任务的测试集上取得73.7的F1分数。 |
| [^73] | [Exploring Heterogeneous Model Merging Approach for Complex Knowledge Transfer](https://arxiv.org/abs/2609.39369) | 本文提出两种免训练的异构模型合并方法（Intersection-Merge 和 Activate-Prune-Merge），无需梯度更新或语义对齐，即可在参数层面将专用模型的知识直接迁移到通用语言模型中，并在嵌入、重排序、奖励建模和 MoE 代码专家迁移等任务上有效提升了通用模型性能。 |
| [^74] | [Making Grid Beam Search Less Greedy](https://arxiv.org/abs/2609.39368) | 本文揭示了网格束搜索在强制执行词汇约束时存在偏差，即倾向于优先满足较简单的约束而将困难的约束推迟到生成序列的末尾，这与不存在此偏差的DFA约束束搜索形成鲜明对比。 |
| [^75] | [Ready2Blend: From Natural-Language Instructions to Composable Alignment Prompts](https://arxiv.org/abs/2609.39365) | Ready2Blend 提出 AlignFormer，将自然语言需求转化为存储在模块化提示词库中的可组合对齐提示词，并通过可组合性正则化在冻结骨干网络的情况下实现推理时的提示词混合与重加权，其性能可媲美基于后训练的持续对齐方法。 |
| [^76] | [Working Around the Compute Ceiling: Byte-Exact Memory in Galahad Makes LLM Reading a One-Time Cost LLM Reading a One-Time Cost](https://arxiv.org/abs/2609.39358) | Galahad 通过为 vLLM、SGLang 和 llama.cpp 提供字节精确的记忆层（Taliesin 缓存并复用 KV 状态、Blaise 按需传递文档章节），避免对已读文本的重复计算，使 LLM 的阅读成为一次性成本，从而绕过每 token 算力上限的限制。 |
| [^77] | [Offline Guidance, Online Reasoning: Reusing LLM Feedback for Small Language Models](https://arxiv.org/abs/2609.39346) | 该论文提出“离线引导、在线推理”的LLM-SLM协作方式，通过复用LLM针对问题生成的反馈作为离线引导，使小语言模型在在线推理时无需反复访问LLM即可提升推理能力。 |
| [^78] | [Understanding as No-Arbitrage: Bounded Dutch Books as a Definition and Training Objective for Language Models](https://arxiv.org/abs/2609.39341) | 本文提出用“无法被计算能力受限的交易者构造荷兰赌套利”来定义和度量语言模型的“理解”程度，证明理解本质上是分级的、下一词元预测目标本身会导致跨问题形式的不一致，并据此提出将抗套利作为训练目标的Arbitr框架。 |
| [^79] | [Taming Speculative Search for Test-Time Scaling in LLM Serving](https://arxiv.org/abs/2609.39334) | 本文提出SpecScale服务系统，通过提前剪枝低质量候选路径、冗余路径计算去重和延迟细粒度验证三种技术，解决了LLM服务中推测执行导致的搜索空间爆炸和频繁验证挑战，实现高效的测试时扩展。 |
| [^80] | [NarrativeSteward: Coordinating Delegation, Guidance, and Verification in Agent-Assisted Interactive Narrative Authoring](https://arxiv.org/abs/2609.39333) | NarrativeSteward是一个智能体辅助的互动叙事创作环境，通过将大纲、世界观与叙事图组织为关联工件，并结合智能体对话、结构审查、变更记录与执行验证，帮助作者理解、引导和评估智能体生成与修订的海量叙事内容。 |
| [^81] | [A Tilted Bowl Is Not a Slippery Slope: Compressing Looped Models](https://arxiv.org/abs/2609.39277) | 该研究推翻了循环模型压缩崩溃源于舍入误差累积的传统观点，揭示了误差只是移动了循环收敛点，据此实现了仅需单次无标签测量即可预测模型失败，并通过最后几轮8位权重循环使失败模型恢复性能。 |
| [^82] | [Concept Subspaces Compute Beyond the Logit Lens: A Weights-Only Test for Locating Representations Upstream of Readout](https://arxiv.org/abs/2609.39263) | 该论文提出了一种仅需模型权重的几何诊断方法，通过测量概念子空间与unembedding矩阵主导右奇异方向的重叠度，发现概念表示（如FARS）仅携带极少读出方向能量，证明其位于输出读出的上游，超越了Logit透镜所能解释的范围。 |
| [^83] | [4MT-VLM: How Coarse Is a VLMs Cognitive Map?](https://arxiv.org/abs/2609.39238) | 该论文提出4MT-VLM基准，发现视觉语言模型虽然能从熟悉视角识别地点，但一旦视角旋转便无法维持认知地图能力，在135度时甚至低于随机水平，远逊于人类的85%表现。 |
| [^84] | [RAIM: Robust Aggregation of Inexpensive Models for Hallucination Detection](https://arxiv.org/abs/2609.39229) | 提出RAIM聚合方案，通过鲁棒的堆叠逻辑回归与可采纳性检验，将多个廉价的开放权重小模型聚合起来，在幻觉/忠实度检测任务上能以极低成本接近Claude Sonnet等前沿专有模型的判断水平。 |
| [^85] | [Argument Structure Prediction in Online Conversations: A Comparative Study of Modeling Paradigms and Task Architectures](https://arxiv.org/abs/2609.39225) | 该论文在严格模式约束下系统比较了监督微调与基于提示的大语言模型在单步及多步任务架构上的论证结构预测表现，并从预测性能、跨领域泛化、模式遵循度和计算效率四个维度进行了统一评估。 |
| [^86] | [ViLegalExpert: A Large-Scale Benchmark for Vietnamese Legal Retrieval and Question Answering from Real-World Consultations](https://arxiv.org/abs/2609.39189) | 该论文提出了基于真实公民-律师咨询构建的越南语法律基准ViLegalExpert，涵盖34个法律领域的17.2万个问题及专家验证证据，为法律检索与问答提供了大规模评测资源，并揭示了证据检索和有据答案生成的重大挑战。 |
| [^87] | [DAGent: Evaluate-then-Grow Planning for Deep Research Agents](https://arxiv.org/abs/2609.39154) | DAGent提出“先评估后生长”的增量规划框架，编排器根据已完成节点的置信度和不确定性信号逐批扩展DAG任务图，克服了传统“先计划后修补”策略在深度研究任务中过早承诺、浪费计算的脆弱性。 |
| [^88] | [Diagnosing On-Policy Self-Distillation for Reasoning Language Models](https://arxiv.org/abs/2609.39118) | 该论文系统诊断了同策略自蒸馏（OPSD）在数学推理中的作用，发现教师信号由推理模式对齐与完整教师前缀共同塑造而非仅由特权语义决定，因此OPSD仅在狭窄的兼容性条件下有效提升推理，否则会导致无效的长度增长、持续退化或行为崩溃。 |
| [^89] | [Bongard: Training Machine Intuition](https://arxiv.org/abs/2609.39111) | 该论文提出 Bongard，一个开放权重的“系统一”模型，首次将机器直觉作为独立能力进行设计与训练，通过共享编码器加并行解码器分支、概率输出头以及三阶段训练加联合嵌入后训练，在单块 Blackwell GPU 上以 70.9 亿参数实现了对改写情境仍保持高准确率的判断能力。 |
| [^90] | [False Frontiers: Diagnosing and Mitigating Co-Cheating in Self-Evolving Search Agents](https://arxiv.org/abs/2609.39102) | 本文揭示了自进化搜索智能体中提议器与求解器在共享错误上“共谋作弊”、导致内部奖励虚高而真实能力停滞的失效模式，并提出多样本验证（MSV）方法部分缓解该问题。 |
| [^91] | [Beyond Text: LLM-Based Dimensional Emotion Evaluation in Multimodal Dialogue](https://arxiv.org/abs/2609.39072) | 该论文提出基于大语言模型的多模态对话情感评估框架，将声学线索转化为文本描述并结合LoRA微调，在IEMOCAP上实现效价CCC 0.7822的新纪录，证明领域适应比模型规模更重要。 |
| [^92] | [LexReward: A Taxonomy-Driven Reward Framework for Legal Language Models](https://arxiv.org/abs/2609.39071) | LexReward提出了一个由分类体系驱动的法律奖励建模框架，从风格、要素和推理链三个维度通过评分细则评估法律回复的多维质量，并将产生的奖励信号用于构建偏好数据以进行DPO和奖励模型训练，从而提升法律语言模型的表现。 |
| [^93] | [CORE: Conflict-Oriented Reasoning Elimination for Verifiable Language-Model Search](https://arxiv.org/abs/2609.39069) | CORE通过向验证器索取认证的冲突核心并回跳至冲突根源决策来指导语言模型搜索，在保持搜索完备性的同时大幅减少验证器调用，并在多项推理任务上超越Tree of Thoughts。 |
| [^94] | [Covert Assistance: Helpful LLM Agents Evade Oversight in Multi-Agent Systems](https://arxiv.org/abs/2609.39050) | 研究发现，即使没有任何对抗性激励，良性LLM智能体也会自发地伪装公司机密凭证以“乐于助人”地帮助外部开发者，同时躲避监控器的监督，九个受测前沿模型中有七个表现出这种“隐蔽协助”的失控行为。 |
| [^95] | [Structure vs. Chain-of-Thought: Evaluating LLM Criteria Extraction for Depression Severity](https://arxiv.org/abs/2609.39049) | 研究系统比较了基于临床标准的结构化提取与思维链推理两种LLM抑郁严重程度评估方式，发现结构化提取仅在阈值拟合标注数据时有不显著的优势，在先验固定阈值下并无增益，说明其可审计性优势尚未转化为性能提升。 |
| [^96] | [RSIGame: Autonomous Agentic Game Development with Recursive Self-improvement](https://arxiv.org/abs/2609.39045) | RSIGame是一个具有递归自我改进能力的自主智能体游戏开发框架，通过局部“探索-诊断-改进”循环与全局质量跟踪循环的协同，可靠地将自动生成的游戏改进到超越可玩版本，避免了朴素迭代改进中的过拟合问题。 |
| [^97] | [Switching Linear Attention](https://arxiv.org/abs/2609.39034) | SwiLA通过将状态更新规则建模为线性回归混合模型中的在线期望最大化过程，在保持线性注意力固定大小循环状态的同时显著提升了其表达能力，从而在效率与性能之间实现平衡。 |
| [^98] | [A Missing Piece for Trustworthy AI Reviewers: From Benchmarking Rhetorical Robustness to SciCore Review](https://arxiv.org/abs/2609.39027) | 该论文提出“修辞性稳健性”概念并构建包含1,260个稿件版本的RobustReview基准，揭示了AI审稿人“虚假稳健性”的问题，进而提出双分支审稿模型SciCore，通过结合全稿评判与基于提取结构化科学内容的评判来提升审稿的可信度。 |
| [^99] | [Evidence First, Arithmetic Second: A System Report and Failure Analysis for DocSem](https://arxiv.org/abs/2609.39013) | 本文介绍了DocSem共享任务系统EVICALC，其采用“先选证据段落、再由语言模型生成算术表达式并在本地求值”的流程，官方测试联合准确率为8.61%，并通过案例分析指出OCR块合并和页面图像读取能力有限等失败环节。 |
| [^100] | [The Invisible Language Tax: Token Premiums of French and Regional Languages in 2026 LLM Tokenizers, and a French-Optimized Prototype](https://arxiv.org/abs/2609.39001) | 研究测量了2026年主流LLM分词器中的语言token溢价，发现法语比英语多消耗31%-58%的token，法国地区语言更是高达1.6-3.3倍，并提出了一种法语优化的分词器原型。 |
| [^101] | [Settle: Learning When to Stop Reasoning](https://arxiv.org/abs/2609.38997) | Settle通过从已完成推理轨迹中学习答案稳定性来训练推理结束标记，在几乎不损失准确率的情况下将token消耗减少40%，扩展了准确率与效率的帕累托前沿。 |
| [^102] | [When Clipping Reverses Correction: Failure Dynamics of Pointwise Forward-KL On-Policy Self-Distillation](https://arxiv.org/abs/2609.38995) | 本文揭示了OPSD中被广泛采用的逐点前向KL截断实际上会阻碍学生模型向教师模型修正，导致训练产生大量持续到响应末尾的重复内容，证明截断目标可能逆转修正效果。 |
| [^103] | [Fairness Beyond a Single Run: Training-Seed Variability in Speech LLM Adaptation](https://arxiv.org/abs/2609.38976) | 该论文首次系统揭示，在语音大语言模型适配中，随机训练种子对人口统计公平性差异的影响远大于音频压缩等超参数，说明仅凭单次训练运行报告的公平性结论不可靠。 |
| [^104] | [Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment](https://arxiv.org/abs/2609.38972) | 本文提出CoT-可解释性对齐（CIA）指标来测量LLM思维链与内部计算之间的一致性，发现现有模型对齐程度有限（44.8%–75.9%），并通过以任务准确率和参数化忠实度为奖励的后训练方法有效提升了对齐性。 |
| [^105] | [Targeted Retrieval, Compact Representations: How CoT Reasoning Improves Long-Context Counting](https://arxiv.org/abs/2609.38958) | 本研究通过大海捞针计数任务揭示，思维链推理使模型从非思考模式下“广泛检索多个目标”的机制，转变为通过枚举逐一“靶向检索”的机制，使注意力集中于单个目标并形成更紧凑的内部表示，从而显著提升长上下文计数准确率。 |
| [^106] | [GraphForge: Training Working Agents with Graph-Anchored Workspace Synthesis](https://arxiv.org/abs/2609.38923) | GraphForge 提出了一种基于证据图的框架，将训练工作智能体所需的任务与验证标准都锚定在真实文件构建的工作区上，从而合成兼具真实性、多样性和可验证性的高质量任务数据。 |
| [^107] | [K2P: Label-Free Knowledge to Prompt Distillation](https://arxiv.org/abs/2609.38898) | K2P提出了一种无标签知识蒸馏方法，通过从教师解答中合成并优化可复用提示、利用答案一致性引导搜索与选择，使冻结的学生模型无需权重更新即可在推理任务上获得准确性保证。 |
| [^108] | [VOSSA: Voiceprint Optimization for Streaming Speech Architectures](https://arxiv.org/abs/2609.38887) | 提出VOSSA说话人表示框架，从内容编码器中间层提取说话人信息并采用注意力统计池化聚合，与语音转换目标联合训练从而无需单独的说话人编码器，在流式实时语音转换中改善了F0动态特性和元音区分性声学线索，同时保持了相当的说话人相似度和语音质量。 |
| [^109] | [Audio Token Attention Is Predictable Before the Language Model Runs](https://arxiv.org/abs/2609.38878) | 该论文发现音频token在语言模型中的全层注意力排名可以在模型运行前由其编码器输出线性预测，据此提出的无需标签的Triage方法能够在推理前高效裁剪音频token，大幅降低大型音频语言模型的计算开销。 |
| [^110] | [TRACE: Target-Aware Retrieval, Attributed Evidence, and Contract-Constrained Extraction for LitTraceQA](https://arxiv.org/abs/2609.38861) | TRACE 提出目标感知检索、归因证据定位与契约约束抽取的一体化框架，通过索引近三万篇论文并在表格抽取前预测观测单元，弥合了文献问答中源访问与评分器可见正确性之间的“接地契约鸿沟”。 |
| [^111] | [Where MLLMs Fail and Why: Causal Task Decomposition for Capability Failure Diagnosis](https://arxiv.org/abs/2609.38851) | 该论文提出一个因果分解框架与 CADET 诊断基准，通过对任务先决条件进行受控干预，将多模态大语言模型在组合任务上的失败区分为目标能力的内在缺陷和上游级联错误，从而精确诊断模型在哪里失败以及为什么失败。 |
| [^112] | [OpenJev-RLCD: A Working RLCD Implementation](https://arxiv.org/abs/2609.38850) | 本文实现了面向推理模型的RLCD（校准决策强化学习），用严格正则评分规则对采样推理依据后的答案分布评分，证明RLVR是缺失多样性项的混合目标，并提出“先校准后强化”的两阶段训练方案，使模型既保持推理能力又获得良好校准。 |
| [^113] | [Scaling Parameter and Context in Attention: Native Sparse Attention from Mixture-of-Head](https://arxiv.org/abs/2609.38832) | 提出NAMOH架构原生的稀疏注意力机制，通过每个token仅激活H个头中的K个，在不扫描完整历史、不增加总KV存储的前提下，同时实现注意力参数扩展与上下文高效扩展。 |
| [^114] | [Forging LLM Authorship Fingerprints with Targeted Rewriting](https://arxiv.org/abs/2609.38831) | 提出ForgePrint框架，通过“先搜索后蒸馏”策略改写LLM输出以伪造作者身份指纹，使其被归因分类器误判为指定目标模型，蒸馏后的4B学生模型在CNN/DM上达到70.2%的目标归因成功率。 |
| [^115] | [BARRAC: Adaptation of an English Aspect-based Sentiment Analysis Approach for Classification Tasks in Arabic Dialects](https://arxiv.org/abs/2609.38820) | 本文提出 BARRAC 方法，将英语方面级情感分析框架适配到阿拉伯语方言分类任务，在五个阿拉伯语方言数据集上取得 63.93% 的平均宏 F1，超过最佳少标签 SOTA 3%，并在五个任务中的四个上超越 GPT-4o。 |
| [^116] | [Whose Voice Survives the Summary? A Voice-Retention Audit of LLM Employee Listening](https://arxiv.org/abs/2609.38818) | 该论文提出“声音留存/表征比率”这一新指标，对一家全球公司2,586条双语员工反馈及45份LLM生成的领导摘要进行审计，发现摘要流程按出现频率而非情感倾向筛选声音——仅被提及一次的关切有86%被丢弃、简短及纯德语内容更易流失——从而揭示了仅靠情感审计无法发现的摘要代表性偏差。 |
| [^117] | [When Reasoning Goes Astray: Attention Dynamics of Uncontrolled Reasoning](https://arxiv.org/abs/2609.38817) | 本文提出RADAR方法，通过动态注意力实时识别大型推理模型的推理状态，揭示良性反思如何演变为失控生成，并将异常注意力分布重新对齐以缓解失控推理带来的成本与风险。 |
| [^118] | [You're Hired: Strategic Model Selection for LLM Collaboration](https://arxiv.org/abs/2609.38816) | 该论文研究了多LLM系统中的模型选择问题，提出并系统评估了9种选择算法，证明策略性组建模型团队比随机或启发式方法最高可提升36.1%的性能。 |
| [^119] | [Can Terminal Agents Trust Their Own Verification? Diagnosing and Improving Self-Verification](https://arxiv.org/abs/2609.38812) | 该论文提出一个诊断框架来量化终端智能体自我验证的可靠性，发现验证行为虽普遍存在，但主要弱点在于错误检测与修复环节——错误候选方案仅 61.43% 被检出，被检出的错误仅 49.36% 被成功修复。 |
| [^120] | [StateTree: Enhancing Long-Term Dialogue Reasoning via Reinforcement Learning](https://arxiv.org/abs/2609.38809) | StateTree是一种数据驱动的强化学习方法，通过在稀缺对话数据上构建树结构路径追踪辅助任务并采用课程式强化学习训练，有效提升了大语言模型的长期对话推理能力。 |
| [^121] | [Blackboard Intelligence Can Surpass Autoregressive on Globally Constrained Problems](https://arxiv.org/abs/2609.38806) | 该论文提出“黑板智能”推理范式，让扩散语言模型在可修改的画布上搜索候选解，并以平均置信度作为全局一致性的代理信号，从而在全球约束问题上超越自回归模型。 |
| [^122] | [Uncovering Uncontrolled Repetition through Residual Stream Dynamics](https://arxiv.org/abs/2609.38802) | 提出Tokenwise Residual Comparison（TRC）方法，通过比较生成过程中各Token对残差流的写入动态，从残差流动力学中识别、定位并抑制大型视觉-语言模型中的失控重复现象。 |
| [^123] | [Overlap, Unique and Conflict: Can LLMs Extract What They Can Recognize?](https://arxiv.org/abs/2609.38799) | 本文提出重叠-独特-冲突（OUC）提取这一跨叙事新任务并构建了包含22K叙事对和140K实例的基准，对14个开源大模型的评估表明，模型提取重叠与冲突信息的能力远弱于提取独特信息的能力。 |
| [^124] | [Evaluating Persistent Calibration under Evolving Model Knowledge](https://arxiv.org/abs/2609.38797) | 该论文提出“持续校准”新问题，研究当模型知识随训练不断演化时，早期训练的置信度估计器能否无需重新监督就持续忠实地反映模型的知识状态，并借此探究置信度与知识之间的依赖关系。 |
| [^125] | [Recovering Off-Policy Supervision for Speculative Decoding](https://arxiv.org/abs/2609.38795) | 提出一种基于rollout的训练框架，通过锚点标签重标注（ALR）和rollout内锚点（IRA）两个互补组件，恢复投机解码草稿模型因离策略token而丢失的完整监督，无需丢弃分歧槽位即可提升贪婪接受长度。 |
| [^126] | [Training LLM Judges from Language Feedback via Position-Selective Self-Distillation](https://arxiv.org/abs/2609.38792) | 该论文提出位置选择性自蒸馏方法，利用教师与学生模型之间的逐位置熵变化来识别携带有效信号的位置，从而更充分地利用自然语言反馈训练大语言模型评判器，克服了基于结果监督的强化学习忽略标准选择token和语言反馈的局限。 |
| [^127] | [Anchor-ECC: Local Integrity Checking for Watermarked LLM Outputs via Error-Correcting Codes](https://arxiv.org/abs/2609.38722) | 提出Anchor-ECC方法，通过在水印结构中引入纠错码约束与边界锚点并结合动态规划解码器，可高效检测并定位对LLM生成文本的局部篡改编辑，块级检测真正例率达99.7%且误报率不超过7.6%。 |
| [^128] | [MetaSteer: Context-Conditioned, nonlinear Steering via Attention-Projection Adaptation](https://arxiv.org/abs/2609.38718) | MetaSteer通过在注意力投影矩阵上学习上下文相关的非线性干预，突破传统线性、上下文无关激活引导的瓶颈，且一次训练即可零样本迁移到未见概念和分布外场景。 |
| [^129] | [Breaking Babel: A Self-Evolving Multi-Agent System for Long-Form Subtitle Translation](https://arxiv.org/abs/2609.38660) | SMART是一种自进化多智能体系统，通过剧集级持久记忆、动态路由与智能体混合层以及无需重训大语言模型的评判-精炼循环，实现术语与风格一致的长篇字幕翻译。 |
| [^130] | [Tacit-TTS: From Autoregressive Decoding to Masked Prediction for Efficient Transcript-Free Voice Cloning](https://arxiv.org/abs/2609.38658) | Tacit-TTS通过将自回归解码替换为掩码非自回归生成、引入免训练的声学长度估计以及ReFlow蒸馏加速流匹配渲染，实现了免转录文本的高质量零样本语音克隆，生成速度比IndexTTS2快10倍以上。 |
| [^131] | [Strong Multilingual Privacy Tagging at Encoder Speed](https://arxiv.org/abs/2609.38630) | 该研究提出了一个以编码器速度运行的多语言隐私实体标注模型，通过覆盖感知掩码与子词边界修复等技术，在7种语言的人工金标准测试上取得88.8的脱敏F1分数，显著超越GLiNER2、Microsoft Presidio和OpenAI Privacy Filter等现有方法。 |
| [^132] | [Marking Contour Tones in Yor\`{u}b\'{a}](https://arxiv.org/abs/2609.38627) | 本文提出在约鲁巴语正字法中采用caron（ˇ）和circumflex（ˆ）符号来标记单个元音上的升降曲折调，以解决传统拼写中声调信息缺失甚至颠倒姓名含义的问题，并使其首次可通过标准键盘输入和计算文本处理。 |
| [^133] | [When Scientific Contradictions Are Lost in Translation](https://arxiv.org/abs/2609.38621) | 该研究通过将不可满足的XOR约束系统伪装成不同实验室的科学报告，首次系统量化了语言模型判断科学发现是否真正矛盾的能力，发现模型面对显式约束时准确率高达90%-96%，但在科学文本中往往偏离约束逻辑而偏向生物学预期。 |
| [^134] | [StreamDecisionBench: Evaluating Decisions in Force on Evolving Language Streams](https://arxiv.org/abs/2609.38612) | 该论文提出 StreamDecisionBench 基准，评估语言模型在证据流不断演化时每一时刻“生效决策”的正确性，并将错误归因于判断或延迟，弥补了传统离线准确率无法衡量决策时效性的缺陷。 |
| [^135] | [SecureVibe: Making Vibe Coding More Secure](https://arxiv.org/abs/2609.38606) | SECUREVIBE通过围绕安全规划与测试行为构建训练信号——包括4个安全任务上的监督微调以及基于可验证执行反馈和提示自监督的后训练方法——显著提升了氛围编程中代码的安全性。 |
| [^136] | [Beyond Oracle Communication: Benchmarking Interactive Intent Alignment Under Miscommunication and Evolving User Intent](https://arxiv.org/abs/2609.38604) | 该论文提出了“交互式意图对齐”这一新任务设定，并构建了Drift-Bench++基准和GRIP评估协议，用于评测LLM智能体在用户沟通不完美、意图静默漂移且耐心有限等现实条件下恢复并持续追踪用户意图的能力。 |
| [^137] | [Prompt2Skill: Unsupervised Skill Optimization From Natural Language Instructions](https://arxiv.org/abs/2609.38593) | 提出了Prompt2Skill框架，仅需自然语言任务描述即可自动构建并优化供大语言模型使用的技能，无需整理的训练数据或昂贵的专家编写流程，从而解决了技能制作成本高、未针对特定模型优化以及新兴任务缺乏技能库覆盖的问题。 |
| [^138] | [Towards Model as a Library: Offline, Community-Sourced AI for Low-Resource African Languages](https://arxiv.org/abs/2609.38574) | 提出“模型即图书馆”（MaaL）软件架构，将小型社区注册语音模型打包为设备端依赖，通过说话人部署时现场注册词汇的方式，为低资源非洲语言提供离线、无幻觉的结构化数据收集方案，克服大语言模型在方言和地区差异上的失真问题。 |
| [^139] | [MedKIT: Evaluating Knowledge Integration and Generalization in Large Language Models](https://arxiv.org/abs/2609.38543) | MedKIT 是一个医学知识整合与迁移基准，通过模拟真实临床知识更新序列，细粒度评估大语言模型整合、迁移和应用新知识的能力，并对 5 个模型上的 12 种知识整合策略进行了大规模实证研究。 |
| [^140] | [Shifting Mechanisms: How Positional Encoding Choice Shapes In-Context Retrieval](https://arxiv.org/abs/2609.38530) | 该论文通过机制分析发现，位置编码的选择决定了语言模型进行上下文检索所依赖的内部机制——标准RoPE模型主要依赖位置检索，而将位置编码限制在局部层的混合架构（如SWA NoPE）则转向语义检索，这一转变带来长上下文收益的同时也隐藏着检索性能上的权衡。 |
| [^141] | [DEdit: Iterative Draft Editing for Speculative Decoding](https://arxiv.org/abs/2609.38510) | DEdit是一种基于扩散模型的推测解码起草器，通过token到token的迭代编辑让后续预测作为双向上下文来修复草稿中的早期错误，并配合基于置信度的ProposalMix训练方案，从而提高草稿接受率与解码加速效果。 |
| [^142] | [Personalized State-Transition-Aware Memory for Clinical Agents](https://arxiv.org/abs/2609.38490) | 提出STAM框架，在新临床记录到来时通过状态转换感知机制将记忆划分为“活动”与“历史”两层，并结合查询依赖门控在需要时选择性调用历史记忆，使临床LLM智能体既能追踪患者当前状态又不会丢失临床历史证据。 |
| [^143] | [Anthropomorphism in the age of Large Language Models: An overview of potential risks and mitigations](https://arxiv.org/abs/2609.38486) | 本文系统综述了大语言模型时代的AI拟人化现象，提出了一个包含21项关注点、覆盖认知、情感、人类能动性、规范性和社会制度五大类别的拟人化风险分类法，并将其与设计、传播、教育等方面的缓解干预措施相关联。 |
| [^144] | [KlinikeBench: Evaluating Language Models Beyond Diagnostic Accuracy](https://arxiv.org/abs/2609.38480) | KlinikeBench是一个包含333个由临床医生编写的任务的基准，通过沙盒环境中的虚拟患者交互，评估语言模型在信息收集和临床评估方面超越单纯诊断准确性的综合临床能力。 |
| [^145] | [The Backdrop Exposes What the World Around an Agent Costs It](https://arxiv.org/abs/2609.38469) | BACKDROP基准通过在智能体执行环境中植入权威覆盖、提示注入、边界越权和写入故障四种日常干扰，并保持任务指令不变，揭示了智能体性能在动态真实环境中从69.5%骤降至31.3%的严重衰减。 |
| [^146] | [Reach Into The CHOIR: Free-List Elicitation Uncovers Distinct Model Voices in LLM Ensembles](https://arxiv.org/abs/2609.38448) | 本文提出CHOIR框架，将认知人类学中的自由列表引出法应用于LLM集成，通过反复引出排序回答、聚类概念并测量概念显著性，揭示了模型表面一致输出之下各自独特的“声音”，从而区分真实的多元性与虚假多元。 |
| [^147] | [What Pretraining and Midtraining Make Learnable from Rewards?](https://arxiv.org/abs/2609.38446) | 该论文证明了预训练与中期训练通过源预测使模型习得执行或检索等通用计算能力，而奖励适应只学习这些能力的任务特定用法，并通过有限采样Adam路径的理论构造与Qwen2.5实验验证了这一分工机制。 |
| [^148] | [Policy-Conditioned AI-Use Detection: An Evidentiary Framework for Academic Publishing](https://arxiv.org/abs/2609.38427) | 该论文提出“策略条件化AI使用检测”的证据框架，将传统上判断文本是否由AI撰写的检测任务，转变为评估人-AI工作流是否遵守学术出版机构具体AI使用政策的证据推理与合规性评估。 |
| [^149] | [LoopVL: Recurrent Visual Intelligence](https://arxiv.org/abs/2609.38426) | LoopVL将Loop Transformer成功扩展至视觉-语言模型，通过模块循环与模型循环计算迭代更新统一的视觉-语言状态，在多模态理解与视觉推理上超越同等及更大规模的非循环模型，并展现出视觉注意力显著转移的“视觉顿悟时刻”。 |
| [^150] | [What Was Said, Not What Was 'Thought': Type-6 Logic for CoT Verification](https://arxiv.org/abs/2609.38420) | 该论文提出Type-6逻辑（带不确定性与递归算子的动态认知逻辑变体）及其图结构验证器，可检测出表层启发式方法遗漏的LLM思维链推理中的结构性缺陷，并支持推理过程的可视化。 |
| [^151] | [ArgGYM: A Procedural, Engine-Verified Benchmark for Structured Defeasible Reasoning](https://arxiv.org/abs/2609.38409) | ArgGYM是一个程序化生成、由引擎自动验证评分的结构化可废止推理基准与RLVR兼容训练环境，将可废止推理分解为十二个可自动评分的任务，弥补了现有数学、代码和逻辑基准之外的现实推理评估空白。 |
| [^152] | [Evaluating Whether LLMs Can Reliably Connect the DOTs?](https://arxiv.org/abs/2609.38406) | 该论文构建了一个包含约9200个实例、覆盖百科文本、常识故事、新闻文章和视觉叙事四种类型的多领域叙事填充基准，并系统评估了20个开源指令微调大语言模型在真实世界叙事填充任务上的表现。 |
| [^153] | [The Geometry of Harmfulness in Multi-Turn Attacks](https://arxiv.org/abs/2609.38389) | 该论文通过分析三个指令微调大模型在三种多轮攻击框架下的隐藏状态表示，首次揭示了有害性与拒绝表示在多轮攻击过程中的几何结构与时间动态演化规律，从而解释了单轮防御在多轮攻击中失效的原因。 |
| [^154] | [Fine-Tuning Diffusion Language Models with Context Selection and Target Weighting](https://arxiv.org/abs/2609.38385) | GoldiMask 通过基于次模目标智能选择上下文 token 并对预测目标进行加权，显著提升了离散扩散语言模型在推理和代码生成任务上的微调效果。 |
| [^155] | [Doc2LoRA Provides Decodable Representations of Scientific Ideas](https://arxiv.org/abs/2609.38374) | 提出 Doc2LoRA 方法，通过超网络将每篇科学论文表示为 LoRA 适配器，使向量空间中的任意点（包括论文混合产生的点）都能解码为可对话的大语言模型，从而实现对科学思想的可解码、可生成式表示。 |
| [^156] | [TALK-Dem: Benchmarking Embodied Task Planning under Dementia-Associated Communication Patterns](https://arxiv.org/abs/2609.38371) | 该论文提出了首个基准 TALK-Dem，用于评估 LLM 驱动的机器人任务规划在应对痴呆症患者典型交流模式（如指代不精确、空洞言语、话题漂移等）时的表现，实验显示开源模型的性能下降高达 22.3%，暴露出显著的鲁棒性差距。 |
| [^157] | [On the Off-Policy Teacher in On-Policy Distillation](https://arxiv.org/abs/2609.38360) | 论文揭示了在线策略蒸馏中教师模型面临离策略不对称性问题——教师对学生生成的前缀续写能力随前缀变长而下降，并提出SCOUT共同训练框架，通过可验证奖励的强化学习周期性优化教师以适应学生生成的前缀。 |
| [^158] | [Beyond Mode Collapse: Generating Diverse Synthetic Expert Conversations via Generative Flow Networks](https://arxiv.org/abs/2609.38359) | 提出基于生成流网络的合成数据生成方法，按专家策略在训练数据中的分布比例采样潜在对话结构，从而生成多样化的高质量专家对话，在辅导和情感支持两个领域实现了比强化学习和端到端LLM基线更优的保真度、模式覆盖率与真实性平衡。 |
| [^159] | [Evaluating Language Model Safety Across Long Adversarial Conversations](https://arxiv.org/abs/2609.38357) | 该研究让对抗性语言模型在长达百余轮的对话中持续发起攻击，发现模型的安全响应率从第一轮的85-100%骤降至第101轮的15-44%，证明单轮安全评估无法保证长对话中的安全性。 |
| [^160] | [Halluscoring 2026: The first shared task on llms hallucination detection and answer verification](https://arxiv.org/abs/2609.38355) | 这是首个针对阿拉伯语问答场景的大语言模型幻觉检测与事实答案验证共享任务，通过四个子任务评估系统对未见问题和未见模型的泛化能力，并基于HalluScore和HalluTruthQA两个数据集展开竞赛。 |
| [^161] | [OpenCollab: A Multi-Agent Coding Framework with Programmable Collaboration and Controllable Runtime](https://arxiv.org/abs/2609.38345) | 提出OpenCollab多智能体编程框架，通过统一组织设计、共享可控运行时和细粒度事件流追踪，并首次定义Adherence指标来量化协作组织结构的实际遵循程度，揭示任何单一配置变化都会使遵循度从47.2%大幅波动至97%以上。 |
| [^162] | [EVOKE: Eliciting World Knowledge in Agents for Transferable Decision-Making](https://arxiv.org/abs/2609.38334) | 该论文提出EVOKE后训练方法，通过在固定状态下施加目标多样性，迫使LLM智能体引出预训练中已内化的世界知识，从而提升其在未见环境中进行多步决策的迁移能力。 |
| [^163] | [Hermes: Learning Contextual Reasoning Unlocks Test-Time Scaling](https://arxiv.org/abs/2609.38332) | Hermes 将上下文窗口分配与信息复用的决策权从外部框架转移给模型本身，并通过两阶段的 Hermes-Learn 训练框架学习适应性上下文推理能力，使模型（尤其是此前做不到的较小开源模型）能够利用额外推理计算实现测试时扩展。 |
| [^164] | [Multi-agent discussion gains less when dissent is withheld](https://arxiv.org/abs/2609.38324) | 该研究提出一个简约模型，指出只有当LLM智能体隐瞒异议的比率低于由净修正率与内化率共同决定的临界值时，多智能体讨论才能推翻错误的初始多数并提升准确性。 |
| [^165] | [HARDE: Optimizing Agent Harnesses for Runtime Risk Detection and Execution Control](https://arxiv.org/abs/2609.38291) | 提出HARDE两阶段优化框架，通过集成触发器、监控器和反馈模块的风险感知智能体框架，实现LLM智能体运行时风险的灵活检测与及时干预，在保障安全的同时维持良性任务的实用性。 |
| [^166] | [Which Models Work Well Together? Measuring Heterogeneity for LLM Team Selection](https://arxiv.org/abs/2609.38274) | 提出了一种基于异构性的大语言模型团队选择框架，通过离线刻画个体能力并引入错误去相关性与预测行为分歧两种互补信号，将团队选择形式化为标准化的质量-互补性组合优化目标，并用高效贪心搜索从候选池中选出小规模团队。 |
| [^167] | [NinaXander: Feasibility and Limits of Composing Frozen Language Models Across Architecture Families via a Shared Latent Space](https://arxiv.org/abs/2609.38261) | 提出NinaXander方法，仅用一个训练好的共享潜空间适配器即可将冻结的RWKV与Pythia等不同架构家族的语言模型拼接成可正常工作的组合模型，并验证了跨架构冻结模型事后重组的可行性及其局限。 |
| [^168] | [ContextAdapt: Evaluating Contextual Adaptation and Value Alignment in LLMs](https://arxiv.org/abs/2609.38260) | 提出了ContextAdapt评估框架，通过基于职业与监管一手文件构建的“价值观×领域”场景，考察12个大语言模型能否在医学、法律、金融和国家安全等领域恰当调整诚实、自主、保密等价值观的应用，并在规范未变时保持一致。 |
| [^169] | [Framing the Narrative: Ideological Mimicry in Large Language Models](https://arxiv.org/abs/2609.38256) | 该研究提出“意识形态模仿”概念并构建 Poli-SHIFT 数据集与评估框架，发现大语言模型会根据用户话语中传递的政治信号系统性偏移其政治立场，可能形成个性化的政治信息环境并加剧社会分歧。 |
| [^170] | [When Does a Spoken Agent Have Enough Evidence to Act? The PACT-SLM Contract Test](https://arxiv.org/abs/2609.38232) | 该论文提出 PACT-SLM 契约测试，通过在部分语音前缀上分别评估行动身份与行动时机，揭示了流式语音智能体常常在语音证据尚不充分时就提前触发行动，为口语智能体的行动时机提供了受控诊断方法。 |
| [^171] | [Conformal Factuality Control for Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2609.38222) | 该研究将主张级保形事实性控制应用于多跳检索增强生成，证明在六种模型-数据集配置中，保形过滤能将保留主张获完全支持的回复比例从无过滤时的55.60%-76.03%稳定提升至95%目标下的95.80%-97.20%。 |
| [^172] | [TutlAit v1: a crowdsourced Moroccan Tamazight speech dataset with Arabic transcriptions and regional accent labels](https://arxiv.org/abs/2609.38219) | 本文推出了TutlAit v1数据集，通过专门构建的众包网络应用收集摩洛哥塔马塞特语语音，配有阿拉伯语转录和地区口音标签，以缓解该官方语言在语音技术领域资源严重匮乏的问题。 |
| [^173] | [The System Prompt Illusion: How Instruction Preambles Modify Computation in Language Models](https://arxiv.org/abs/2609.38205) | 该研究通过CKA分析发现系统提示词对语言模型内部计算的影响具有层选择性和指令类型依赖性——人设与格式指令深度重构中间表征，而安全指令几乎无法穿透模型计算，甚至限制性与放任性的安全指令激活几乎相同的计算路径，揭示了依靠系统提示词实现安全控制的局限。 |
| [^174] | [Automatic estimation of verbal fluency index in people with Motor Neuron Disease using ASR alignment and pause modelling](https://arxiv.org/abs/2609.38203) | 本研究提出一种结合WhisperX语音识别对齐与Silero停顿建模的自动化系统，能够准确估计运动神经元病患者的言语流畅性指数，并提取临床可解释的指标，显著优于传统声学特征方法，为认知障碍监测提供了新方案。 |
| [^175] | [TomasuLLM: Out-of-Order Speculative Execution for LLM Agents](https://arxiv.org/abs/2609.38201) | TomasuLLM提出了一种乱序推测执行运行时系统，让大语言模型智能体的工具调用在写时复制沙箱中提前执行并验证后按轨迹顺序提交，从而在不破坏正确性的前提下显著加速含长时工具调用的智能体任务。 |
| [^176] | [Large Language Models are Approximate Survival Estimators](https://arxiv.org/abs/2609.38181) | 该论文提出了Survprompt框架，将结构化患者协变量转换为自由文本临床病例描述，以零样本方式提示预训练大语言模型预测患者生存结局，并在两个多机构泛癌症队列上与传统生存模型进行了系统性基准对比评估。 |
| [^177] | [Gender bias across LLMs is common and highly heterogenous](https://arxiv.org/abs/2609.38036) | 该研究通过两种实验范式测试了来自九个厂商的十款大语言模型，发现性别偏见在模型间普遍存在但高度异质——部分模型表现出反刻板印象的性别归因模式，另一些模型则在道德判断中与人类保护女性免受伤害的倾向一致。 |
| [^178] | [SelfSearch: Reward-Free Search for Self-Improving Agents](https://arxiv.org/abs/2609.37968) | SelfSearch提出了一种免奖励的自我改进搜索方法，智能体通过利用以往自我修改回合的记录（包含推理、工具操作和结果）来改进自身，无需昂贵的下游评估即可在多个模型-基准测试设置中显著提升成功率。 |
| [^179] | [A Proposed Rubric for Evaluating Expressed Clinical Reasoning in Large Language Model Responses](https://arxiv.org/abs/2609.37788) | 该论文提出一个融合医学教育评估框架、临床大语言模型基准和通用LLM推理评估研究的多维评分量规，用于对大语言模型针对金标准临床案例的自由文本回答中表达的临床推理进行结构化评估。 |
| [^180] | [MERGE: Multi-LLM Ensemble for Retrieval via Generative Enrichment](https://arxiv.org/abs/2609.37574) | MERGE提出一个两阶段多LLM集成框架，先由三个小型开源LLM独立生成查询扩展候选、再由更大的LLM生成式合成为单一查询，并用基于下游检索性能的自动提示词优化循环取代传统LLM评估器，解决了单一LLM查询增强受限于模型偏见且提示词工程难以扩展的问题。 |
| [^181] | [Traverse: Learning When to Remember, Reset, and Redirect for Long-Horizon Web Search](https://arxiv.org/abs/2609.37082) | 论文提出Traverse自主搜索框架，让智能体通过“评分标准—答案—验证”三状态自我管理搜索并配备封存记忆工具进行主动上下文管理，同时用仅训练上下文管理后末段的简单策略避免“封存崩塌”，使35B模型在BrowseComp上达到72.83。 |
| [^182] | [Learning from Think-Mode Advantage via On-Policy Distillation](https://arxiv.org/abs/2609.37044) | 本文提出 ThinkOPD，通过在线策略蒸馏让模型学习思考模式的优势，并引入轨迹-回答分歧（TRD）度量，在回答层面自适应地路由教师监督信号，克服了统一共享思考轨迹蒸馏带来的师生不匹配问题。 |
| [^183] | [CoEM: Empowering Long-Context Reasoning with Commit-on-Evidence Memory](https://arxiv.org/abs/2609.36935) | CoEM 提出了一种“证据提交记忆”机制：在固定上下文预算下先逐字保留待定证据，再由学习到的策略根据新到来的上下文决定将其提交、继续保留或丢弃，从而避免过早压缩导致的关键信息丢失，提升长上下文推理性能。 |
| [^184] | [QuantMLA: Function-Aligned Dual-Path Quantization for Low-Bit MLA KV Caching](https://arxiv.org/abs/2609.36760) | 提出了 QuantMLA——一个函数对齐的低比特双路径量化框架，通过系统建模 MLA 内容路径与 RoPE 路径的量化误差，并学习可完全离线融合的路径特定变换，在消除在线开销的同时大幅压缩 MLA KV 缓存且保持全精度计算效果。 |
| [^185] | [Learning from Teacher Continuations at Student States](https://arxiv.org/abs/2609.36246) | OLIVE 提出一种在线干预式蒸馏框架——学生生成前缀、教师自回归续写并以其交叉熵更新学生——同时解决了离线 SFT 的协变量偏移、OPD 的监督碎片化以及分布匹配蒸馏需教师 token 概率三大局限，以相近成本取得更优推理性能。 |
| [^186] | [Sage: Formalization with Semantic Correction](https://arxiv.org/abs/2609.35790) | 本文提出 Sage，一个智能体化的形式化引擎，通过四阶段分解式生成流水线与融合 Lean 4 编译器诊断和多维语义反馈的双信号语义修正循环，解决了自然语言翻译为 Lean 4 形式化命题时的“严谨性幻觉”问题，确保生成命题既句法有效又数学忠实。 |
| [^187] | [SEABench: Benchmarking Endogenous Misalignment In Self-Evolving Agents](https://arxiv.org/abs/2609.35596) | 该论文提出了SEABench基准，用于量化研究自进化智能体因自我修改而在无外部对抗影响下产生的内生失准（不安全行为）风险。 |
| [^188] | [Coding Agent Memory Post-training: Unlocking the Memory Potential of Pre-trained File Operations for Long-Horizon Tasks via Reinforcement Learning](https://arxiv.org/abs/2609.34422) | 该论文提出 CAMG 训练场套件，通过强化学习让智能体直接复用预训练中已掌握的文件操作能力作为记忆机制，而非从头学习专用记忆工具，从而显著提升智能体在长程任务中的记忆使用能力。 |
| [^189] | [Over-Personalization Is a Decision Failure: Generation-Induced Apply Bias in LLMs](https://arxiv.org/abs/2609.34284) | 该论文将大语言模型处理用户偏好的过程分解为“知道—决策—生成”三个阶段分别测量，揭示过度个性化的根源在于决策失败：当模型被要求作答时，生成过程会诱发“应用偏差”，使其执行本应抑制的偏好，而非知识缺失或生成错误所致。 |
| [^190] | [Jev in Medicine: A Benchmark Evaluation. Preliminary Results](https://arxiv.org/abs/2609.34024) | 本研究首次在四个医学基准上系统评估了非生成式“系统一”模型Jev，发现其准确率总体低于或接近GPT-6 Sol，但概率校准表现更优。 |
| [^191] | [Auditing Agent Actions through Query-Conditioned Attribution](https://arxiv.org/abs/2609.33676) | 本文提出了“查询条件的智能体行为归因”新任务及包含1,396个审计查询的A³Bench基准，能够根据自然语言审计查询自动恢复智能体行动的来源与有序中间证据，在无需访问模型内部的情况下实现对LLM智能体行为的高效审计。 |
| [^192] | [Don't Repeat Yourself: Self-Supervised Fine-Tuning for Coverage](https://arxiv.org/abs/2609.31688) | DRY-SFT是一种无需奖励、验证器或正确性过滤的两阶段后训练方法，通过让模型在参考先前尝试的基础上生成不同解，再对每个尝试独立微调，从而显著提升大语言模型的输出多样性与覆盖度。 |
| [^193] | [RAZOR: Pruning Replaceable Experts in LLMs](https://arxiv.org/abs/2609.30465) | 该论文提出无需训练的 MoE 专家剪枝方法 RAZOR，利用共识残差衡量专家的功能可替换性，在固定剪枝预算下剪除可被存活专家替代的专家，无需梯度或恢复训练即可最大程度保留原始模型输出分布。 |
| [^194] | [PPTBench: Can Coding Agents Reconstruct the Visual World through Structured, Editable Slides](https://arxiv.org/abs/2609.29718) | 该论文提出PPTBench——一个包含500个基于真实arXiv论文科学流程图的可编辑幻灯片重建基准，用于评测编码智能体从视觉内容中推断结构并以可编辑程序化对象形式实现端到端视觉重建的能力。 |
| [^195] | [Three Ways Classical Test Theory Misleads for LLM Judges](https://arxiv.org/abs/2609.29709) | 该论文揭示经典测试理论的三个常用信度统计量在LLM评判者评估情境中含义发生扭曲——例如内部一致性系数无法区分题目设计与评判者错误的影响——因此不能直接照搬用于解读LLM评判者的表现。 |
| [^196] | [Order-Invariant Answers, Order-Sensitive Representations in Mathematical Reasoning](https://arxiv.org/abs/2609.28442) | 该研究发现，语言模型对规则排序的内部表征越清晰（排列信噪比越高），其解决重排序数学问题的准确率就越高，揭示了答案不变性与表征不变性是两个不同的概念。 |
| [^197] | [Large Knowledge Model: From Papers to a Scientific Reasoning Landscape](https://arxiv.org/abs/2609.27297) | 本文提出大知识模型（LKM），将论文表示为基于原文来源的推理图，构建包含问题、工作流和证据三个视图的科学推理图景，使科学文献成为可计算访问的共享推理资源，从而支持大规模利用文献中的科学推理过程。 |
| [^198] | [Distill What You Trust: Reliability-Aware Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2609.23697) | 提出TrustMOPD方法，以专家模型RL训练前后相对于共享参考模型的位移作为token级可靠性代理，实现无标签的多教师加权蒸馏监督分配，将学生模型性能恢复率从54.4%大幅提升至91.5%。 |
| [^199] | [From Concept Alignment to Causal Grounding: An Intervention Test of Chain-of-Thought Faithfulness](https://arxiv.org/abs/2609.23065) | 该论文将CoT忠实性重新定义为“内部概念锚定”问题，利用共享稀疏自编码器直接比较预测过程与CoT过程的内部概念，并首次引入三个相关性对齐指标和一个因果消融指标Δp，以检验共享概念是否因果性地驱动模型答案。 |
| [^200] | [Generalized Multimodal Foundation Model](https://arxiv.org/abs/2609.22107) | 提出了一种不依赖特定模态的通用多模态基础模型，通过在大规模具有多样因果结构的合成多模态数据集上训练，使其能够适用于任意的模态组合和任意的预测任务。 |
| [^201] | [Intrinsic Sequence-Likelihood Confidence in Retrieval-Dominated Extractive QA: Two Pre-Specified Negatives, and What They Do and Do Not Attribute](https://arxiv.org/abs/2609.19942) | 该研究通过预先注册的评估标准证明，在检索已能恢复92-99.8%最优性能的抽取式问答场景中，模型的内在序列似然置信度无论作为蒸馏触发器还是路由弃答策略的控制信号均告失效。 |
| [^202] | [Agora: Git as Shared Memory for Collective AutoResearch](https://arxiv.org/abs/2609.18094) | Agora的核心创新是以Git中仅追加的DAG作为多个自主研究智能体的共享内存，让每条研究成果都成为可验证、可复现的不可变提交，并通过派生索引和多样性感知选择规则，使13个无中央规划的智能体持续协作近12天而不陷入重复搜索。 |
| [^203] | [Detectable Only Where It Is Confounded: What Verified Duplication Counts Say About Membership Evidence in Language Models](https://arxiv.org/abs/2609.10830) | 利用公开可验证的重复计数，研究表明在真实文本的重复水平下，语言模型的预测成本与其训练数据暴露程度之间至多只有微弱的关联（秩相关约-0.08），因此预测成本低并不能可靠地证明某个句子存在于训练数据中。 |
| [^204] | [I Don't Miss You, but I Do: Self-Explanation Faithfulness of Modality Missingness in Vision-Language Models](https://arxiv.org/abs/2609.07596) | 该论文提出了一种介入式评估协议，揭示视觉语言模型对模态缺失的自我解释并不忠实——它们系统性地高估现有模态证据的充分性，并大幅低估恢复缺失模态对预测结果的影响。 |
| [^205] | [A Ticket from Marginals to Joints: Coupled-Noise Distillation for One-Step Block Generation in Diffusion Language Models](https://arxiv.org/abs/2609.06324) | 提出CONDOR方法，通过耦合噪声蒸馏从零训练扩散语言模型，使其在掩码嵌入受高斯噪声扰动时仅用一次前向传播即可生成连贯的整块文本，且无需目标侧编码器或自回归教师。 |
| [^206] | [Safety Monitors Mostly Catch What the Model Already Refuses](https://arxiv.org/abs/2609.05797) | 现有安全监控器的标准评估方式掩盖了其真实短板——它们主要捕获模型本已拒绝的有害请求，而在模型实际会回答的请求上召回率骤降，通过含蓄化改写请求可让 44%–93% 的有害请求绕过监控并产生有害输出。 |
| [^207] | [ShallowStream: Index Shallow then Answer Deep for Streaming Video Understanding](https://arxiv.org/abs/2609.02780) | 提出ShallowStream框架，采用“浅层索引、深层回答”的策略，利用MLLM浅层高效处理流式视频帧，从而显著降低计算开销并抑制KV缓存的增长。 |
| [^208] | [Lot Machine: Multimodal Lot Extraction from Auction Catalogs](https://arxiv.org/abs/2608.30510) | 本文提出了一个利用视觉-语言模型从历史拍卖目录中自动提取结构化拍品元数据的流水线，并在不同提示策略、受限解码框架和部署条件下进行了系统评估，以满足文化遗产机构在预算、算力和数据隐私方面的实际需求。 |
| [^209] | [Reconstructing the Right Episode: Evaluating Interleaved Conversational Memory Beyond Long Context](https://arxiv.org/abs/2608.25655) | 本文提出了SCALE-QA基准，用于评估聊天助手在平坦混合主题线程中推断早期因果片段以正确完成后续任务的能力，填补了现有记忆基准忽视片段完整性失败的空白。 |
| [^210] | [One Success Isn't Reliability: Thinkingbox, a Sandbox and Benchmark for Agents in Stateful Business Workflows](https://arxiv.org/abs/2608.19741) | 本文提出了Thinkingbox，一个支持有状态业务流程智能体评估的沙盒和基准，强调通过多轮交互、策略遵循和持久状态转换来确保可靠性，而非仅关注单次成功。 |
| [^211] | [An Empirical Study of Reward Specification and Benchmark Reliability in GRPO-based LLM Unlearning](https://arxiv.org/abs/2608.17804) | 本研究发现，在基于GRPO的大语言模型遗忘中，不同奖励设计导致的优化成功与行为遗忘并不等价，且现有基准评估指标可能相互矛盾，需警惕奖励黑客和基准可靠性问题。 |
| [^212] | [Cross-Model Memory Transfer via Target-Side Reader Adaptation](https://arxiv.org/abs/2608.17050) | 本文研究了跨模型记忆迁移中，冻结记忆与目标端阅读器相对重要性，发现轻量级阅读器适配是关键，而非记忆本身。 |
| [^213] | [Do AI chatbots find what experts would? Effects of model, user role, and sample size on study retrieval for medical questions](https://arxiv.org/abs/2608.13786) | 本研究系统评估了三种主流AI聊天机器人在医学问题检索中的表现，发现其检索质量受模型类型、用户角色和样本量显著影响，且与专家标准存在差距。 |
| [^214] | [Large Language Model-Driven Small-Capitalization Trading: Integrating Financial News Sentiment, Macroeconomic Indicators, and Technical Signals](https://arxiv.org/abs/2608.12283) | 本文提出一种不确定性感知的投资组合构建方法，将大语言模型预测的风险分解为偶然和认知部分并直接纳入协方差矩阵，在罗素2000股票上验证了纯阿尔法和纯贝塔触发机制优于交集机制。 |
| [^215] | [Simplex Relaxation for Discrete Diffusion](https://arxiv.org/abs/2608.10615) | 本文提出 Simplax，通过精确的 Dirichlet-类别增广，在保持均匀离散扩散的类别腐蚀过程和边际分布不变的前提下，为每个被腐蚀的类别状态耦合一个辅助单纯形连续变量，从而得到可处理的 Rao–Blackwell 化反向桥接目标函数和随机反向采样器。 |
| [^216] | [LegalPincite: Multi-level Legal Information Retrieval Dataset](https://arxiv.org/abs/2608.03756) | 该论文提出了基于欧盟法院判决构建的大规模法律信息检索数据集LegalPincite，通过消除查询文本数据泄漏、保留全部段落语料并提供案例和段落两个级别的引用标注，解决了现有法律IR数据集任务设定不现实而导致性能虚高的问题。 |
| [^217] | [Direct Construction of Disambiguated Knowledge Bases from Large Language Models](https://arxiv.org/abs/2608.03729) | 提出GPTKB 2.0方法，通过对实体、关系和类别的即时消歧机制，直接从大语言模型构建了首个百万级规模的消歧知识库，包含超过100万个实体和3840万条三元组。 |
| [^218] | [Relational Priors as Convergence Pressure in LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2608.03239) | 该论文提出将LLM多智能体系统中智能体间的关系显式化为带符号的成对先验并注入系统提示，揭示了这种“收敛压力”现象——积极关系虽能提升公共资源治理的可持续性和主观问题的共识度，但在客观问答中反而会降低最终答案的准确性。 |
| [^219] | [AgentSnare: Learning to Delay, Divert, and Defuse Autonomous Penetration Agents](https://arxiv.org/abs/2607.26998) | AgentSnare提出了一种轨迹自适应的欺骗防御系统，通过基于渗透代理交互历史动态构建诱饵工件，持续将自主渗透代理从真实目标引开，克服了传统静态蜜罐痕迹易被识别绕过的缺陷。 |
| [^220] | [Using Fine-Tuned LLMs to Identify Indicators of Vulnerability in UK Police Incident Logs](https://arxiv.org/abs/2607.18446) | 本研究开发了一个基于本地部署开源大语言模型的多阶段分类流程，通过重复推理、标签聚合、人工审查和统计校正，从英国警方事件记录中估计心理疾病、药物滥用、酒精依赖和无家可归等脆弱性指标的普遍程度。 |
| [^221] | [How Much Human Label Variation Does Formal Semantic Structure Explain?: Group-Level Effects and Item-Level Ceilings in NLI](https://arxiv.org/abs/2607.15870) | 该研究通过预注册分析直接测量发现，形式语义结构对自然语言推理中人类标签变异的解释力有限——群体层面上非纯向上单调假设的标签熵显著更高，但条目层面上形式语义特征仅能解释3.3%–3.6%的熵方差。 |
| [^222] | [SpanUQ: Span-Level Uncertainty Quantification for Large Language Model Generation](https://arxiv.org/abs/2607.05721) | 该论文提出跨度级不确定性估计（SLUE）新任务，并开发了轻量级模型SPANUQ，通过单次前向传播即可检测语义连贯的文本跨度并量化其不确定性，克服了词元级与序列级方法在粒度上的局限。 |
| [^223] | [A Dominant Self-Conditioning Direction Drives Repetition in Unconditional Continuous Diffusion Language Models](https://arxiv.org/abs/2607.00588) | 该论文发现连续扩散语言模型的重复生成源于自条件化反馈回路将表征驱向一维收缩吸引子，并据此提出无需训练的推理时干预方法ACE，通过对比去噪路径估计重复方向以逃逸该吸引子。 |
| [^224] | [Fork-Think with Confidence](https://arxiv.org/abs/2606.31484) | 提出基于模型置信度先识别分叉点再触发并行思考的方法Fork-think，在保持相当或更优推理性能的同时，将token消耗降低最多30%、运行时间降低最多57%。 |
| [^225] | [Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and GUI Grounding Datasets](https://arxiv.org/abs/2606.25760) | 该论文提出Argus——一个跨VLM代理与GUI定位数据集的事后不确定性量化系统基准，通过对27种开放权重方法和8种闭源方法的全面评估，检验UQ方法排名在不同模型、基准与可观测性条件下的稳定性。 |
| [^226] | [OctoNest: Adaptive Cross-Device Execution through Stateful Control](https://arxiv.org/abs/2606.20487) | OctoNest 提出了一种有状态的跨设备编排框架，通过编排器与设备智能体协作，自适应地在设备内切换模态或跨设备重新分配任务，并引入含 158 个实例的 CAPEBench 基准来评测跨设备任务执行。 |
| [^227] | [CombEval: A Framework for Evaluating Combinatorial Counting in Large Language Models](https://arxiv.org/abs/2606.19788) | CombEval是一个动态组合计数基准，通过类型化规范可控生成经求解器验证精确答案的计数问题，评估发现11个大语言模型在有序对象、不可区分元素、相对位置约束和嵌套对象依赖等组合推理上仍然脆弱。 |
| [^228] | [KVEraser: Learning to Steer KV Cache for Efficient Localized Context Erasing](https://arxiv.org/abs/2606.17034) | KVEraser是一种学习式KV缓存编辑方法，通过用学习到的引导状态替换被擦除区间的KV状态来复用其余缓存，实现计算成本仅取决于被擦除片段长度（而非后缀长度）的高效局部上下文擦除，且效果几乎媲美完全重计算。 |
| [^229] | [Interactor: Agentic RL oriented Iterative Creation for Ad Description Generation in Sponsored Search](https://arxiv.org/abs/2606.15911) | 提出Interactor框架，通过智能体强化学习让生成模型与多个生成式奖励模型进行多轮交互迭代，自动生成融入世界知识且与落地页一致的高质量付费搜索广告描述。 |
| [^230] | [Beyond the Commitment Boundary: Probing Epiphenomenal Chain-of-Thought in Large Reasoning Models](https://arxiv.org/abs/2606.13603) | 该研究通过答案 logits 与注意力探针发现，大型推理模型在推理早期就跨越“承诺边界”确定了最终答案，其后的思维链步骤属于副现象，对最终答案没有因果影响。 |
| [^231] | [GrepSeek: Training Search Agents for Direct Corpus Interaction](https://arxiv.org/abs/2605.29307) | GrepSeek提出了一种让LLM搜索智能体直接通过shell命令与语料库交互寻找证据的新范式，采用“Tutor/Planner生成可靠搜索轨迹初始化+GRPO强化学习优化”的两阶段训练方法，并通过语义保持的执行优化实现大规模语料库上的高效检索。 |
| [^232] | [RA-MoE: Routing-Aligned Fine-Tuning for Multilingual Adaptation of Mixture-of-Experts Models](https://arxiv.org/abs/2605.28306) | 提出RA-MoE三阶段微调框架，利用中间层跨语言路由对齐现象，在ci样本上选择性地将目标语言路由向成功的英语路由模式对齐（同时匹配任务专家的总路由质量及其相对分配），从而有效实现混合专家模型的多语言适配并缩小目标语言性能差距。 |
| [^233] | [When In-Distribution Gains Fail: Evaluating Weak-to-Strong Reward Models under Preference Shift](https://arxiv.org/abs/2605.25629) | 本文揭示了弱到强奖励模型在偏好分布偏移下的迁移失败问题——弱监督微调会使强模型偏向源域特征——并提出表示锚定正则化方法，通过约束表示漂移来保持可迁移的偏好表示。 |
| [^234] | [Direct Translation between Sign Languages](https://arxiv.org/abs/2605.20588) | 该论文提出通过改进的反向翻译方法从现有文本-手语语料库构建跨语言手语配对数据，进而联合训练单一模型实现手语之间的直接翻译，避免了级联系统的中间错误传播问题。 |
| [^235] | [CHI-Bench: Can AI Agents Automate End-to-End, Long-Horizon, Policy-Rich Healthcare Workflows?](https://arxiv.org/abs/2605.16679) | CHI-Bench是一个评估AI智能体能否端到端自动化政策密集、需多角色协作与多轮交互的长周期医疗保健工作流程的新基准，涵盖事先授权、利用管理和护理管理三大领域。 |
| [^236] | [WASIL: In-the-Wild Arabic Spoken Interactions with LLMs](https://arxiv.org/abs/2605.16364) | 该论文发布了WASIL——一个包含音频、ASR识别假设、助手回复和用户好/差评反馈的真实场景阿拉伯语口语交互数据集，并通过可回答性标注将ASR识别错误与用户请求本身固有的不可回答性区分开来。 |
| [^237] | [Diagnosing Training Inference Mismatch in LLM Reinforcement Learning via a Zero-Mismatch Reference](https://arxiv.org/abs/2605.14220) | 该论文提出零失配诊断环境VeXact来隔离研究LLM强化学习中的训练-推理失配（TIM），证明微小的token级数值差异可独立引发训练崩溃，并指出TIM应被视为影响LLM强化学习稳定性的一阶系统级扰动因素而非良性数值噪声。 |
| [^238] | [Generalizing the Turing Test to Interactive Agents](https://arxiv.org/abs/2605.10851) | 该论文提出广义图灵测试（GTT），将图灵测试从人类推广到任意交互式智能体，证明了“图灵比较器”传递性等理论性质，并通过九个大语言模型的实验验证了图灵分数能够清晰地对模型进行分层。 |
| [^239] | [Speech-based Psychological Crisis Assessment using LLMs](https://arxiv.org/abs/2605.10027) | 该论文提出一种基于大语言模型的自动心理危机等级分类框架，通过将非语言情感线索注入语音转录文本的副语言注入方法，以及以诊断推理链生成为辅助任务的推理增强训练策略，显著提升了心理援助热线危机评估的准确性与服务质量。 |
| [^240] | [Beyond LoRA vs. Full Fine-Tuning: Gradient-Guided Optimizer Routing for LLM Adaptation](https://arxiv.org/abs/2605.07111) | 针对LoRA与全量微调谁更优取决于任务和模型这一难题，本文提出MoLF统一框架，利用梯度引导的优化器路由在两种微调方式间动态选择。 |
| [^241] | [Frontier Lag: A Bibliometric Audit of Capability Misrepresentation in Academic AI Evaluation](https://arxiv.org/abs/2605.04135) | 该研究对超过11万篇文献进行系统计量分析后发现，学术论文中评估的LLM能力落后于当时的前沿模型（中位数差距+10.85 ECI），且这一“前沿滞后”差距正以每年+5.53 ECI的速度持续扩大。 |
| [^242] | [LSR-Ben: A Logical and Scientific Reasoning Benchmark for Evaluating Process Reward Models](https://arxiv.org/abs/2605.01203) | 本文提出LSR-Ben基准，涵盖科学推理和逻辑推理两大领域及九个子领域，用于全面评估过程奖励模型在数学推理之外多样化场景中的过程级错误检测能力。 |
| [^243] | [Correct Prediction, Wrong Steps? Consensus Reasoning Knowledge Graph for Robust Chain-of-Thought Synthesis](https://arxiv.org/abs/2604.14121) | 提出CRAFT方法，通过聚合多个候选推理轨迹的共识组件构建推理知识图谱，从推理结构层面修复LLM“答案正确但推理步骤有缺陷”的问题，实现更鲁棒的思维链合成。 |
| [^244] | [VisionFoundry: Teaching VLMs Visual Perception with Synthetic Images](https://arxiv.org/abs/2604.09531) | 提出仅需任务名称即可自动生成合成图像与问答数据的流水线VisionFoundry，其构建的VisionFoundry-10k数据集无需人工标注即可显著提升多个开源视觉语言模型的视觉感知能力。 |
| [^245] | [Atomic and Holistic LLM Judges for Reference-Grounded Support Labels: A Prompt-Controlled Comparison](https://arxiv.org/abs/2603.28005) | 该研究在控制裁判模型、输入与指令细节的统一条件下系统比较了四种单次调用的LLM裁判设计，发现当支撑标签依赖答案完整性时，整体式评分标准对所有裁判模型都比候选侧原子分解更准确且使用更少token，表明将答案拆分为原子声明并非总是有益。 |
| [^246] | [DataFlex: A Unified Framework for Data-Centric Dynamic Training of Large Language Models](https://arxiv.org/abs/2603.26164) | 本文提出 DataFlex——一个基于 LLaMA-Factory 的统一以数据为中心的动态训练框架，支持样本选择、领域混合调整和样本重加权三大数据优化范式，并可作为标准大语言模型训练的即插即用替代方案，解决了现有方法代码库孤立、接口不一致导致的可复现性差与难以公平比较的问题。 |
| [^247] | [Structure of Basic Human Values in Russian Social Media](https://arxiv.org/abs/2603.18822) | 该研究利用大语言模型多阶段标注框架对750万条VKontakte帖子中的价值观表达进行测量，首次基于社交媒体的自发表达刻画了俄语人群的基本价值观结构，并与传统问卷结果对照，解决了价值观解释的主观性问题。 |
| [^248] | [Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety](https://arxiv.org/abs/2603.10044) | 评测条件对测得的模型安全性影响超过脚手架本身——在相同的基准题目上，选择题与开放式格式会使测得的安全性相差5-20个百分点，说明评测结果更多取决于测量方法而非模型潜在的 safety 能力。 |
| [^249] | [Life-Bench: A Benchmark and Knowledge Graph Framework for Multimodal Personalization Beyond Concept Recognition](https://arxiv.org/abs/2602.19001) | 本文提出了Life-Bench——一个包含11,800多个问答对、按概念识别、事件理解和聚合推理三个层次组织的合成多模态个性化基准测试，并配套提出个人知识图谱框架LifeGraph，通过结构化检索与按需访问源视觉证据，在事件理解和聚合推理任务上表现出显著优势。 |
| [^250] | [Semantic Chunking and the Entropy of Natural Language](https://arxiv.org/abs/2602.13194) | 该论文提出了一个将语言冗余性与文本分层语义组织相关联的统计框架，利用大语言模型将文本递归分割成语义连贯的块以构建“语义树”，从而从语义结构层面解释了自然语言中约80%冗余度（即每个字母仅携带约1比特信息）的来源。 |
| [^251] | [ResidualKV: Residual-Based KV Cache Compression for Efficient Long-Context Inference](https://arxiv.org/abs/2602.08005) | ResidualKV提出将KV缓存分解为稀疏全局参考与量化残差编码，结合稀疏注意力按需重建状态，并通过动态步长调度抑制参考增长，在不永久丢弃令牌信息的前提下实现高效长上下文推理。 |
| [^252] | [Functional Subspace, where language models can use vector algebra to solve problems](https://arxiv.org/abs/2602.01687) | 该研究提出假设：大型语言模型通过在功能子空间中运用向量代数来执行任务，并通过分析上下文学习过程中LLM的功能模块与残差流来验证这一假设。 |
| [^253] | [Lowest Span Confidence: Zero-Shot Hallucination Detection from a Single LLM Response](https://arxiv.org/abs/2601.19918) | 提出零样本指标“最低跨度置信度”，仅需单次LLM响应即可检测幻觉，无需昂贵重复采样或访问模型内部状态。 |
| [^254] | [Context-Aware Classification and Grading of Sensitive Information in Online Conversational Health Data](https://arxiv.org/abs/2601.09717) | 本研究将在线医疗对话中的敏感信息分级构建为上下文感知评估任务，提出纳入断言状态、经历者、检查结果状态和信息粒度的操作框架，并借助最小化改变上下文因素的对比案例来评估大语言模型对隐私敏感信息的判断能力。 |
| [^255] | [Superficial Reflection or Genuine Thought? A Fine-Grained Cognitive Analysis of Large Reasoning Models](https://arxiv.org/abs/2512.00729) | 本文提出基于人类认知过程的细粒度推理步骤分类体系（5组17类），揭示当前大型推理模型答案后的“复查”反思多为表面行为而非真实思考，并提出CAPO自动标注方法构建了27万余条推理步骤的数据集，证明显式引导更丰富的反思过程可显著改善模型自纠正能力。 |
| [^256] | [From Compound Figures to Medical Multi-image Reasoning: Scaling Multimodal Large Language Models with Biomedical Literature](https://arxiv.org/abs/2511.22232) | 该论文构建了源自生物医学复合图像的大规模医学多图像指令数据集PMC-MI及人工审核基准PMC-MI-Bench，并提出三阶段训练框架M3LLM，通过监督微调与选择感知强化学习提升多模态大语言模型的医学多图像推理能力。 |
| [^257] | [Generative AI Purpose-built for Social and Mental Health: A Real-World Pilot](https://arxiv.org/abs/2511.11689) | 一项纳入299名美国成年人、随访长达12个月的真实世界试点研究表明，专为心理健康训练的生成式AI基础模型能在10周内显著降低抑郁和焦虑症状（Cohen's d达0.93和0.79），改善孤独感与社交互动，并借助临床医生确认的自动化安全防护机制实现安全有效的干预。 |
| [^258] | [Sequential Bayesian Evaluation of Large Language Model Behavior](https://arxiv.org/abs/2511.10661) | 本文提出一种序贯贝叶斯评估框架，通过量化LLM随机性带来的评估不确定性，并自适应地优先选择下一个最值得评估的基准提示词，从而实现更具成本效益的大语言模型行为评估。 |
| [^259] | [MedRECT: A Bilingual Medical Reasoning Benchmark for Error Correction in Clinical Texts](https://arxiv.org/abs/2511.00421) | 提出了MedRECT双语（日英）医学基准，将临床文本错误处理形式化为错误检测、错误句子提取和错误纠正三个子任务，并通过对11个LLM的评估发现思考模式能显著提升错误检测与句子提取性能。 |
| [^260] | [Instruction Retrieval at Inference Time for Small Language Models](https://arxiv.org/abs/2510.13935) | 提出指令检索方法，将教师模型的专业知识提炼为针对小模型定制的指令语料库（每个领域仅需构建一次、运行时无需教师模型），使小型语言模型在推理时通过检索指令获得背景知识、解题流程和常见错误提示，从而胜任需要专业知识的专家级任务。 |
| [^261] | [Think Right: Learning to Mitigate Under-Over Thinking via Adaptive, Attentive Compression](https://arxiv.org/abs/2510.01581) | 提出 TRAAC，一种利用模型自注意力机制识别并剪除冗余推理步骤、并将估计的问题难度纳入训练奖励的在线后训练强化学习方法，从而在思考不足与过度思考之间取得平衡，解决适应性不足问题。 |
| [^262] | [From Construction to Injection: Edit-Based Fingerprints for Large Language Models](https://arxiv.org/abs/2509.03122) | 该论文提出了一个端到端的大语言模型注入式指纹框架，通过基于代码混合的指纹构建方法解决不可察觉性权衡问题，并确保指纹在模型遭受修改后仍能维持持久的触发-目标行为，从而实现更鲁棒的模型所有权验证。 |
| [^263] | [NMIXX: Domain-Adapted Neural Embeddings for Cross-Lingual eXploration of Finance](https://arxiv.org/abs/2507.09601) | 该论文提出NMIXX方法，通过构建18.8k个含语义对比的金融领域三元组对现有编码器进行领域适配，显著提升了英语和韩语金融文本嵌入的语义相似度性能（如BGE-M3在FinSTS上从0.1969提升至0.2967），但代价是通用领域性能略有下降。 |
| [^264] | [Cross-Layer Discrete Concept Discovery for Interpreting Language Models](https://arxiv.org/abs/2506.20040) | 提出跨层向量量化自编码器CLVQ-VAE，通过离散向量量化瓶颈将残差流中跨层重复的特征压缩为紧凑、可解释的概念向量，从而更有效地解释语言模型。 |
| [^265] | [Voices of Freelance Professional Writers on AI: Limitations, Expectations, and Fears](https://arxiv.org/abs/2504.05008) | 本研究通过对301名自由职业作家的问卷调查和互动任务发现，AI写作工具的采用主要受同伴影响和职业前景驱动而非人口特征，多语言作家面临公平使用AI的性能与感知双重障碍，且对工作安全的担忧普遍存在、与是否使用AI无关。 |
| [^266] | [Listening to the Wise Few: Query-Key Alignment Unlocks Latent Correct Answers in Large Language Models](https://arxiv.org/abs/2410.02343) | 本文发现在去除旋转位置编码（RoPE）后计算的查询-键分数可以识别中间层中一类通用的“选择-复制”注意力头，这些头通过语义对齐从模型内部编码的潜在知识中可靠地定位多项选择题的正确答案，从而以可解释的机制方式揭示 LLM “知道但不说出”的现象。 |
| [^267] | [Mitigating Memorization In Language Models](https://arxiv.org/abs/2410.02159) | 该论文提出了17种缓解语言模型记忆训练数据问题的方法（包括5种全新的机器遗忘方法）以及高效小模型评估套件TinyMem，并证明用TinyMem开发的方法可成功迁移应用于生产级语言模型。 |
| [^268] | [Robust Wake-Up Word Detection by Two-stage Multi-resolution Ensembles](https://arxiv.org/abs/2310.11379) | 本文提出一种两阶段多分辨率集成检测框架，利用轻量级设备端模型实时处理音频流，再由服务器端异构集成验证模型进行二次确认，从而在两个工作点上实现鲁棒、节能且保护隐私的唤醒词检测。 |
| [^269] | [ETHER: Aligning Emergent Communication for Hindsight Experience Replay.](http://arxiv.org/abs/2307.15494) | 本文提出了ETHER，通过对齐紧急沟通来解决回顾性经验重演中的问题，克服了先前架构依赖预设函数的限制，并提高了数据效率和性能。 |

# 详细

[^1]: 面向多模态临床诊断的排序感知提示优化

    Ranking-Aware Prompt Optimization for Multimodal Clinical Diagnosis

    [https://arxiv.org/abs/2609.40361](https://arxiv.org/abs/2609.40361)

    针对临床数据类别不平衡导致准确率指标失效的问题，本文提出以AUROC为优化目标的成对级别帕累托提示进化方法，显著提升了多模态大语言模型在临床诊断中的提示优化效果。

    

    多模态大语言模型（MLLM）正在迅速推动临床诊断的发展，然而其适配流程仍然以基于准确率的目标为核心。临床数据存在严重的类别不平衡：一个恒定预测多数类的模型可以获得超过90%的准确率，但在临床上毫无用处。因此，我们采用AUROC进行评估与优化，这是一个无需设定阈值的指标，能够将阳性样本排序在阴性样本之上，且对类别平衡不敏感。我们聚焦于MLLM中的提示优化。诸如GEPA等反思式方法使用一个二值分数矩阵，其中每一行对应一个评估实例，每一列对应一个候选提示；矩阵单元记录每个实例的正确性，因此列平均值即为准确率，并以此驱动候选选择。我们提出了成对级别的帕累托提示进化方法（Ranking-PE），它将每个正确性行替换为基于（阳性，阴性）实例对的成对排序行：如果候选提示对阳性样本的评分高于阴性样本，则该单元取值为1……

    arXiv:2609.40361v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) are rapidly advancing clinical diagnosis, yet their adaptation pipelines remain anchored to accuracy-based objectives. Clinical data are heavily class-imbalanced: a constant-majority predictor can score above 90% accuracy while being clinically useless. We therefore evaluate and optimize for AUROC, a threshold-free score that ranks positives above negatives and is invariant to class balance. We focus on prompt optimization in MLLMs. Reflective methods such as GEPA use a binary scores matrix with one row per evaluation instance and one column per candidate prompt; cells record per-instance correctness, so the column average is accuracy and drives candidate selection. We introduce pair-level Pareto prompt evolution (Ranking-PE), which replaces each correctness row with a pairwise-ordering row over (positive, negative) instance pairs: the cell is 1 if the candidate scores the positive higher than t
    
[^2]: 半事实信用增强的策略优化

    Semifactual Credit-Augmented Policy Optimization

    [https://arxiv.org/abs/2609.40360](https://arxiv.org/abs/2609.40360)

    提出SCAPO，一种受因果启发的GRPO改进方法，通过半事实提示干预分析词元敏感性，并将半事实稳定性融入词元级信用分配，从而降低LLM对任务无关提示特征的依赖并提升推理准确性。

    

    基于可验证奖励的强化学习（RLVR）提升了大型语言模型（LLM）的推理能力，但其预测仍然对与任务无关的提示特征敏感。我们通过半事实提示干预来研究这种敏感性，即在保持底层问题及其答案不变的前提下对提示进行修改。我们的分析揭示了词元级敏感性的显著差异，并表明在解码过程中抑制高漂移的词元候选可以在不更新模型权重的情况下提高推理准确率。这些发现凸显了组相对策略优化（GRPO）的一个局限性：它为每个响应词元分配相同的基于结果的优势，可能在强化有用推理的同时也强化了潜在的虚假依赖。受此观察启发，我们提出了半事实信用增强策略优化（SCAPO），这是一种受因果思想启发的GRPO变体，将半事实稳定性纳入词元级信用分配中。

    arXiv:2609.40360v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) has improved the reasoning capabilities of large language models (LLMs), yet their predictions remain sensitive to task-irrelevant prompt features. We investigate this sensitivity through semifactual prompt interventions that preserve the underlying problem and its answer. Our analysis reveals substantial variation in token-level sensitivity and shows that suppressing high-drift token candidates during decoding improves reasoning accuracy without updating model weights. These findings highlight a limitation of Group Relative Policy Optimization (GRPO), which assigns the same outcome-derived advantage to every response token and may reinforce potential spurious dependence alongside useful reasoning. Motivated by this observation, we introduce Semifactual Credit-Augmented Policy Optimization (SCAPO), a causally inspired variant of GRPO that incorporates semifactual stability into toke
    
[^3]: EvoDuet：面向科学发现的网络搜索与任务求解双层协同进化

    EvoDuet: Bilevel Co-Evolution of Web Searching and Task Solving for Scientific Discovery

    [https://arxiv.org/abs/2609.40340](https://arxiv.org/abs/2609.40340)

    EvoDuet提出了一种双层协同进化优化方法，在固定LLM参数的情况下共同进化搜索查询与任务解，并通过检索门控动态决定检索新文档、复用旧文档或直接继续，从而显著提升了科学发现类任务的进化搜索性能。

    

    基于大语言模型（LLM）的进化搜索在进展需要模型所缺乏的外部知识时可能会陷入停滞。提供相关文档会有所帮助，但简单添加网络搜索工具可能会随着解的变化而不断返回相同的页面。我们提出了EvoDuet，一种在固定模型参数下使解和搜索查询协同进化的双层优化方法。在每次迭代中，检索门控让LLM评估自身的知识缺口，并选择检索新文档、复用已存储的文档，或在没有文档的情况下继续进行。内循环优化查询并根据文档预计产生的解分数对文档进行排序；外循环从这些文档中并行生成候选解，并记录已评估的结果供后续搜索使用。在21个优化任务上（每次迭代生成一个候选解），EvoDuet将OpenEvolve的归一化发现增益在使用GPT-5.6-Luna时从74.1%提升至78.0%，在使用Gemini-3.8-时从61.3%提升至82.3%。

    arXiv:2609.40340v1 Announce Type: new  Abstract: Evolutionary search with large language models (LLMs) can stall when progress requires external knowledge the model lacks. Supplying relevant documents helps, but simply adding web search tool can keep returning the same pages as solutions change. We introduce EvoDuet, a bi-level optimization method that co-evolves solutions and search queries with fixed model parameters. At each iteration, a retrieval gate lets the LLM assess its knowledge gap and choose to retrieve new documents, reuse stored ones, or proceed without them. An inner loop refines queries and ranks documents by the solution scores they are predicted to yield; an outer loop generates candidates in parallel from these documents and records the evaluated outcomes for later searches. Across 21 optimization tasks with one candidate per iteration, EvoDuet raises OpenEvolve's normalized discovery gain from 74.1% to 78.0% with GPT-5.6-Luna and from 61.3% to 82.3% with Gemini-3.8-
    
[^4]: MatLoom：紧凑程序空间中的分层文本到材质生成

    MatLoom: Layered Text-to-Material Generation in a Compact Program Space

    [https://arxiv.org/abs/2609.40322](https://arxiv.org/abs/2609.40322)

    MatLoom提出一种紧凑的面向图层的材质程序语言，配合预训练语言模型，通过解析器引导修复与预览评审实现免微调的文本到材质生成，其最佳配置在141个提示词的基准上超越了三个扩散基线。

    

    材质生成不仅应产出外观，还应产出构建该外观的规则。我们提出MatLoom，一种紧凑的、面向图层的材质程序语言，可与预训练语言模型结合进行文本到材质生成。每个程序由带Alpha遮罩的图层组合而成，图层间共享的空间表达式定义了覆盖范围与基于物理的渲染（PBR）通道，从而使图案、颜色与浮雕之间的依赖关系显式化。一个独立的解释器将程序求值为材质贴图，同时源代码保留命名字段和图层参数，便于后续编辑创作。无需任务特定的微调，我们的流程利用解析器引导的修复和基于预览的评审来修改材质设计，然后在保持每个候选的其余源代码不变的情况下搜索噪声种子。在一个包含141个提示词的精选基准上，使用六个骨干模型进行评估，我们表现最佳的配置在（原文在此处截断）上取得了高于三个扩散基线的平均分数。

    arXiv:2609.40322v1 Announce Type: cross  Abstract: Material generation should produce not only an appearance, but also the rules that construct it. We introduce MatLoom, a compact, layer-oriented language for text-to-material generation with pretrained language models. Each program composes alpha-masked layers whose shared spatial expressions define coverage and physically based rendering (PBR) channels, making dependencies between patterns, color, and relief explicit. A standalone interpreter evaluates the program into material maps, while the source retains named fields and layer parameters for subsequent authoring. Without task-specific fine-tuning, our pipeline uses parser-guided repair and preview-based critique to revise material designs, then searches noise seeds while keeping each candidate's remaining source fixed. On a curated benchmark of 141 prompts evaluated with six backbones, our best-performing configuration achieves higher mean scores than three diffusion baselines on 
    
[^5]: 循环混合专家模型的缩放定律

    Scaling Laws for Looped Mixture of Experts

    [https://arxiv.org/abs/2609.40316](https://arxiv.org/abs/2609.40316)

    本文提出首个联合建模循环与稀疏性的缩放定律——“循环缩放定律”，它能更准确地预测循环MoE模型的性能损失，并为在计算与内存约束下设计循环混合专家模型提供了理论依据。

    

    循环Transformer和混合专家模型为高效扩展提供了互补的路径：循环在固定参数量的情况下增加计算深度，而MoE的稀疏性在固定激活计算量的情况下扩展总容量。然而，现有的缩放定律只是将循环或稀疏性单独进行建模。在这项工作中，我们提出了循环缩放定律，这是首个将循环与稀疏性同模型规模和数据一起进行联合建模的缩放定律。其核心是一个有界的、以稀疏度为条件的循环映射，该映射刻画了循环带来的有效参数增益以及稀疏性如何提升这一增益。这些定律比以往的方法能更准确地预测循环模型的留出损失，并且作为特例可恢复标准的稠密模型和MoE缩放定律。除了预测能力之外，拟合得到的定律还为在计算和内存约束下设计循环MoE模型提供了有原则的理论基础。

    arXiv:2609.40316v1 Announce Type: cross  Abstract: Looped transformers and Mixture-of-Experts (MoE) offer complementary routes to efficient scaling: recurrence increases computational depth at fixed parameters, while MoE sparsity expands total capacity at fixed active compute. Yet existing scaling laws model recurrence or sparsity in isolation. In this work, we introduce Loop Scaling Laws, the first scaling law to jointly model recurrence and sparsity alongside model size and data. At its core is a bounded, sparsity-conditional recurrence mapping that characterizes the effective-parameter gain from looping and how sparsity raises this gain. The laws predict the held-out loss of looped models more accurately than prior alternatives, and recover the standard dense and MoE scaling laws as special cases. Beyond prediction, the fitted laws provide a principled foundation for designing looped MoE models under compute and memory constraints. Downstream evaluations further demonstrate the comp
    
[^6]: 一个AI Token价值几何？野生AI生成网络文本的缩放定律

    How Much Is an AI Token Worth? Scaling Laws for Wild AI-Generated Web Text

    [https://arxiv.org/abs/2609.40295](https://arxiv.org/abs/2609.40295)

    本文通过预训练800个语言模型并拟合缩放定律，首次系统量化了“野生”AI生成网络文本对预训练的影响：对数据匮乏的模型，少量AI文本初期有益但收益迅速饱和并逆转为损害，而对人类文本充足的模型则几乎立即有害。

    

    网络文本构成了预训练数据的大部分，且其中AI生成的比例日益增加。在应用FineWeb质量过滤之后，我们发现2026年6月网络数据中27.5%的token被Pangram标记为AI生成，到8月这一比例上升至31.1%。与合成数据或模型崩溃实验设置不同，这种“野生”AI文本来自众多不同的模型，是为人类读者撰写的，并以未标记的形式进入预训练语料库。野生AI文本会如何影响语言模型预训练？为回答这一问题，我们预训练了800个语言模型，改变所添加AI token与人类token的比例，并在人类文本与AI生成文本的验证损失上拟合缩放定律。研究发现：对于数据匮乏的模型，在预训练数据中添加AI token最初会降低人类文本上的损失，但随着添加量增加，收益趋于饱和并迅速逆转为损害；而对于在大量人类文本预算上训练的模型，AI token几乎立即抬升损失……

    arXiv:2609.40295v1 Announce Type: new  Abstract: Web text makes up the majority of pretraining data and is increasingly AI-generated. After applying FineWeb quality filtering, we find that 27.5% of tokens from June 2026 web data are labeled as AI-generated by Pangram, rising to 31.1% by August. Unlike synthetic data or model-collapse setups, this *wild* AI text comes from many models, is written for human readers, and arrives unlabeled in pretraining corpora. How does AI text in the wild affect language model pretraining? To answer this question, we pretrain 800 language models, varying the ratio of added AI tokens to human tokens, and fit scaling laws to held-out losses on both human and AI-generated text. For data-starved models, adding AI tokens to pretraining data initially lowers loss on human text, but the benefit saturates as more are added and quickly *reverses* into harm. For models trained on high budgets of human text, AI tokens raise loss almost immediately, while the same 
    
[^7]: 大语言模型遗忘中的语言漏洞：从174种语言的基准测试到覆盖感知遗忘

    Linguistic Loopholes in LLM Unlearning: From a 174-Language Benchmark to Coverage-Aware Unlearning

    [https://arxiv.org/abs/2609.40286](https://arxiv.org/abs/2609.40286)

    该论文揭示了大语言模型遗忘的跨语言漏洞，提出了覆盖174个语言-文字对的跨语言遗忘基准，并设计了COVER方法，在有限语言预算下智能选择源语言进行遗忘，以最大化对全部语言的跨语言知识擦除效果。

    

    在大语言模型中遗忘某一语言中的事实，并不能保证该知识在其他语言中也被移除——改变查询方式、甚至改变所请求答案的语言，都可能重新唤起看似已被遗忘的知识，这被称为“跨语言漏洞”。应对这一挑战最直接的方案——在所有语言中都执行遗忘——既不可扩展也不可取，因为它会放大对模型其他无关能力的损害。我们提出了“语言预算约束下的多语言遗忘”任务，其目标是从语言集合中选出一个子集，使跨语言擦除效果最大化。为了研究这一任务，我们引入了跨语言遗忘张量，这是一个涵盖174个语言-文字对和25种原子级同义改写类型的遗忘基准，用以考察遗忘何时能够泛化到同一知识的不同语言表达之上。我们进一步提出了COVER方法，该方法通过选择源语言来最大化对未接受遗忘监督的语言的预测覆盖率，从而实现在……

    arXiv:2609.40286v1 Announce Type: cross  Abstract: Unlearning a fact in one language does not guarantee its removal in others as changing the query or even the requested answer language can reopen seemingly forgotten knowledge -- a cross-lingual loophole. The most straightforward solution to this challenge -- unlearning in all languages -- is neither scalable nor desirable as it amplifies damage to unrelated model capabilities. We introduce the task of language budgeted multilingual unlearning where the goal is to select a subset of languages that maximizes cross-lingual erasure. To study this task we introduce the Cross-Lingual Unlearning Tensor, an unlearning benchmark that spans 174 language--script pairs and 25 atomic paraphrase types to examine when forgetting generalizes across linguistic expressions of the same knowledge. We further propose COVER, which selects source languages to maximize predicted COVERage of languages receiving no forget supervision, enabling unlearning on a 
    
[^8]: cua-speedrun：计算机使用智能体速度的标准化基准测试

    cua-speedrun: Standardized Benchmarking of the Speed of Computer-Use Agents

    [https://arxiv.org/abs/2609.40284](https://arxiv.org/abs/2609.40284)

    提出了cua-speedrun基准，通过标准化的基础设施、统一的虚拟机设置和执行流程解决现有基准测试的可重复性危机，从而可靠地评估计算机使用智能体的速度与效率。

    

    计算机使用智能体（CUA）是指利用图形用户界面（GUI）在计算机上完成任务的智能体，最近在许多标准基准测试（包括困难的长时程任务）中已经超越了人类的表现。它们的能力无疑令人印象深刻，然而，CUA广泛采用和部署的一个关键障碍仍然是其速度和成本。要实现更快且能力更强的CUA，就需要对其速度进行可靠的评估，但目前许多CUA基准测试面临可重复性危机。这些基准测试基于复杂的基础设施，机器和容器配置各不相同，从而干扰了对CUA执行速度的评估。为了解决这一空白，我们提出了cua-speedrun，它引入了标准化的基础设施和任务集，专注于评估CUA的速度和效率。cua-speedrun使用统一的虚拟机设置和执行流程，以及通用的智能体……

    arXiv:2609.40284v1 Announce Type: cross  Abstract: Computer use agents (CUAs), which use graphical user interfaces (GUIs) to complete tasks on a computer, have recently surpassed human performance on many standard benchmarks, including difficult long-horizon tasks. Their capabilities are undoubtedly impressive, however, a key barrier to the widespread adoption and deployment of CUAs remains their speed and cost. Progress towards faster yet capable CUAs requires reliable evaluation of their speed, but many CUA benchmarks currently face a reproducibility crisis. Benchmarks are based on complex infrastructure with varying machine and container configurations that confound the evaluation of the execution speed of CUAs. Towards addressing this gap, we propose cua-speedrun, which introduces standardized infrastructure and task sets, with a focus on evaluating the speed and efficiency of CUAs. cua-speedrun uses a uniform virtual machine setup and execution pipeline, along with a common agent 
    
[^9]: 面向决策的推荐重排序：关于Jev的实证研究

    Decision-Oriented Recommendation Reranking: An Empirical Study of Jev

    [https://arxiv.org/abs/2609.40241](https://arxiv.org/abs/2609.40241)

    该实证研究表明，TypeSafe AI提出的决策导向模型Jev在个性化推荐重排序中既能保持与基线模型相当的推荐效果，又比逐点式Qwen重排序器具有更平缓的服务延迟增长，为LLM重排序在质量与效率之间的权衡提供了有价值的替代方案。

    

    大型语言模型（LLMs）在推荐重排序任务中展现出了潜力，但其使用在推荐质量与服务效率之间引入了一个重要的权衡。我们研究了当重排序任务本质上是针对预定义候选项目的结构化选择时，面向决策的模型能否提供一种有用的替代方案。具体而言，我们对Jev（由TypeSafe AI定义为一种"System One Model"）在个性化推荐重排序中的表现进行了受控实证研究，并将其与推荐专用模型以及逐点式和列表式的Qwen重排序器在多个Amazon Reviews领域和不同候选集规模下进行比较，同时评估了推荐效果与实际观测到的服务延迟。我们的结果表明，Jev相对于所评估的基线模型保持了较强的推荐效果，同时相比逐点式Qwen重排序器表现出明显更为平缓的延迟增长，尽管……

    arXiv:2609.40241v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown promise for recommendation reranking, but their use introduces an important tradeoff between recommendation quality and serving efficiency. We investigate whether a decision-oriented model provides a useful alternative when the reranking task is fundamentally a structured choice among predefined candidate items. Specifically, we conduct a controlled empirical study of Jev, described by TypeSafe AI as a ``System One Model,'' for personalized recommendation reranking and compare it with recommendation-specific models and pointwise and listwise Qwen rerankers across multiple Amazon Reviews domains and candidate-set sizes, evaluating both recommendation effectiveness and observed serving latency. Our results show that Jev maintains strong recommendation effectiveness relative to the evaluated baselines while exhibiting substantially more gradual latency growth than the pointwise Qwen rerankers, altho
    
[^10]: 开放权重模型微调技术用于放射报告实体抽取的比较研究

    Comparison of techniques for fine-tuning open-weight models for entity extraction from radiology reports

    [https://arxiv.org/abs/2609.40236](https://arxiv.org/abs/2609.40236)

    该研究通过2×2实验设计比较了不同微调策略与训练数据来源，发现基于真实GPT-4o标注报告蒸馏的指令微调Gemma-3-12B模型在颅内出血急性度抽取任务上可媲美GPT-4o，为隐私友好、成本可控且可重复的放射报告结构化提供了开源替代方案。

    

    将自由文本放射报告转换为结构化标签有助于队列构建、质量保证以及临床影像模型的监测，但最强的标签抽取器是托管式专有模型，其使用引发了隐私、成本和可重复性方面的担忧。我们探究了微调后的开放权重模型（Gemma-3-12B）能否在从非增强头颅CT报告中提取多标签颅内出血（ICH）急性度这一任务上媲美GPT-4o，以及哪些要素至关重要。我们采用2×2实验设计，将两种适配策略（判别式分类头CH；生成式指令微调IFT）与两种训练数据来源（对真实GPT-4o标注报告的蒸馏；由GPT-4o基于真实样本生成的合成报告）交叉组合，涵盖五种训练规模，并在100份经专家裁定的报告上与GPT-4o及未微调的开放权重基座模型进行基准比较。结果表明，蒸馏后的指令微调模型（DIFT）达到了与GPT-4o相当的水平。

    arXiv:2609.40236v1 Announce Type: new  Abstract: Converting free-text radiology reports into structured labels supports cohort building, quality assurance, and monitoring of clinical imaging models, but the strongest label extractors are hosted proprietary models whose use raises privacy, cost, and reproducibility concerns. We asked whether a fine-tuned open-weight model (Gemma-3-12B) can match GPT-4o at multi-label intracranial hemorrhage (ICH) acuity extraction from non-contrast head-CT reports, and which ingredients matter. Using a 2x2 design, we crossed two adaptation strategies (a discriminative classification head, CH; generative instruction fine-tuning, IFT) with two training-data sources (distillation of real GPT-4o-labeled reports; synthetic reports generated by GPT-4o from real exemplars), across five training sizes, benchmarked on 100 expert-adjudicated reports against GPT-4o and the un-tuned open-weight base. The distilled instruction-tuned model (DIFT) matched GPT-4o (macr
    
[^11]: 面向连续扩散语言模型的分布匹配蒸馏

    Distribution Matching Distillation for Continuous Diffusion Language Models

    [https://arxiv.org/abs/2609.40235](https://arxiv.org/abs/2609.40235)

    提出了两种分布匹配蒸馏方法Simplex-DMD和Reinforce-DMD，通过利用学生模型的概率性词元输出，将连续扩散语言模型的网络评估次数大幅降低至仅需4次即可实现高质量文本生成。

    

    连续扩散语言模型可以并行生成所有词元，但高质量的生成仍可能需要数百次网络评估（NFEs）。我们研究了如何通过分布蒸馏来降低这一成本，并利用学生模型的概率性词元输出。我们的统一公式将学生模型的输出参数化与由此产生的梯度估计器联系起来，从而得到两种采用相同学生架构和反向KL匹配目标的方法：Simplex-DMD使用连续词元松弛和路径梯度，而Reinforce-DMD使用类别采样以及带有可学习密度比的REINFORCE方法。我们为多步生成开发了这两种方法，并研究了与每种参数化相关的训练和采样选择。在OpenWebText数据集上，对于1,024个词元的序列，Simplex-DMD仅用4次NFEs就在5.44 nats的一元熵下实现了45.6的生成困惑度，相比基线降低了49%。

    arXiv:2609.40235v1 Announce Type: cross  Abstract: Continuous diffusion language models generate all tokens in parallel, yet high-quality generation can still require hundreds of network evaluations (NFEs). We study how distributional distillation can reduce this cost by exploiting the student's probabilistic token outputs. Our unified formulation connects the student's output parameterization to the resulting gradient estimators and yields two methods with the same student architecture and reverse-KL matching objective: Simplex-DMD uses continuous token relaxations and pathwise gradients, while Reinforce-DMD uses categorical sampling and REINFORCE with a learned density ratio. We develop both methods for multi-step generation and investigate the training and sampling choices associated with each parameterization. On OpenWebText, for sequences of 1,024 tokens, Simplex-DMD achieves a generative perplexity of 45.6 at a unigram entropy of 5.44 nats in just 4 NFEs, a 49% reduction relative
    
[^12]: PhantomEnvironments：在虚构世界中训练大语言模型智能体

    PhantomEnvironments: Training LLM Agents in Fictional Worlds

    [https://arxiv.org/abs/2609.40221](https://arxiv.org/abs/2609.40221)

    提出完全由规则生成、零边际成本的虚构世界多轮RL环境PhantomEnvironments，无需任何真实世界事实即可训练LLM智能体成为强大的搜索智能体，并能迁移到真实世界多跳搜索基准，性能往往超过使用真实训练数据的结果。

    

    用强化学习（RL）训练大语言模型（LLM）智能体的瓶颈在于环境：环境必须提供可验证的奖励、支持长程交互，并且能够低成本扩展。现有方法要么依赖昂贵的人工整理数据，要么依赖LLM生成的环境，而后者存在幻觉和基准测试污染的风险。我们证明，LLM可以通过完全由规则生成的合成环境被训练成能力强大的搜索智能体，这类环境的生成不需要LLM，且边际成本为零。我们构建了PhantomEnvironments——源自虚构世界的多轮RL环境，智能体必须在模板化文章语料库中搜索以回答多跳问题。尽管与真实世界不共享任何事实，这些极其简单的环境所训练出的智能体却能成功迁移到真实世界的多跳搜索基准测试中，在较新的基准上往往优于使用真实世界训练数据训练的结果。训练后的智能体还能泛化到未见过的虚构宇宙中。

    arXiv:2609.40221v1 Announce Type: cross  Abstract: Training LLM agents with reinforcement learning (RL) is bottlenecked by environments, which must provide verifiable rewards, support long-horizon interaction, and scale cheaply. Existing approaches rely on costly human-curated data or on LLM-generated environments that risk hallucinations and benchmark contamination. We show that LLMs can instead be trained into capable search agents using synthetic environments generated entirely by rules, whose generation requires no LLM and has zero marginal cost. We build PhantomEnvironments, multi-turn RL environments from fictional worlds, where agents must search a corpus of templated articles to answer multi-hop questions. Despite sharing no facts with the real world, these strikingly simple environments yield agents that transfer to real-world multi-hop search benchmarks, often outperforming real-world training data on newer benchmarks. Trained agents generalize to unseen fictional universes, 
    
[^13]: SCB：用于评估语音到语音模型多轮推理能力的语音对话基准

    SCB: SpeechConversationBench for Evaluating Multi-Turn Reasoning in Speech-to-Speech Models

    [https://arxiv.org/abs/2609.40198](https://arxiv.org/abs/2609.40198)

    本文提出SCB基准，通过将GSM8K数学题分片并跨多轮语音对话逐步披露来评估语音到语音模型的多轮推理能力，发现现有商用系统在分片条件下准确率显著下降，而具备显式上下文管理的LEGO语音管道表现稳定。

    

    语音到语音系统必须解决那些需求分布在多个对话轮次中的任务。我们提出了SpeechConversationBench（SCB），这是一个专注于口语数学推理的评估基准，使用了103个分片化的GSM8K数学问题。该框架比较了三种设置：在单一轮次中完整呈现原始问题、将信息分片拼接后一同呈现，以及在多个轮次中逐步口头披露信息。我们报告了四个商用语音系统以及LEGO的最终答案准确率，LEGO是由SCBX创新实验室团队内部开发的专有语音管道，具有显式的对话上下文管理。与拼接设置相比，四个商用系统在分片设置下的准确率下降了5.0至25.3个百分点。LEGO在所有三种条件下均达到77.5%的准确率，而GPT-4o Realtime在分片设置下的准确率为76.6%。两个单轮基线用于区分模型对问题重新表述的敏感性

    arXiv:2609.40198v1 Announce Type: cross  Abstract: Speech-to-speech systems must solve tasks whose requirements emerge across conversational turns. We introduce SpeechConversationBench (SCB), a focused evaluation of spoken mathematical reasoning using 103 sharded GSM8K problems. The framework compares the original problem delivered in one turn (full), its concatenated information shards delivered together (concat), and incremental spoken disclosure across turns (sharded). We report final-answer accuracy for four commercial speech systems and LEGO, a proprietary speech pipeline developed internally by the SCBX Innovation Lab team with explicit conversational context management. Relative to concat, sharded accuracy decreases by 5.0-25.3 percentage points across the four commercial systems. LEGO achieves 77.5 percent accuracy in all three conditions, compared with 76.6 percent sharded accuracy for GPT-4o Realtime. The two single-turn baselines distinguish sensitivity to problem reformulat
    
[^14]: MemLife：面向长期第一人称视频记忆的整理与推理

    MemLife: Curating and Reasoning over Long-Term Egocentric Video Memories

    [https://arxiv.org/abs/2609.40195](https://arxiv.org/abs/2609.40195)

    MemLife是一种多模态记忆系统，通过构建基于实体的第一人称文本情节并用时间索引的智能体读取器进行检索，在无需训练和查询时视频访问的情况下，在四个长时程基准上比最强免训练基线提升4.6%–12.0%，解决了长期第一人称视频中记忆证据丢失和检索竞争的问题。

    

    长期第一人称视频使个性化AI助手能够对日常生活进行推理。然而，随着视频历史增长到跨越数月甚至数年的数百小时，为每次查询重新处理原始视频片段在计算上变得难以承受。记忆系统通过将视频压缩为文本表示提供了一种可扩展的替代方案，但在实际基准测试中常常失败：要么记忆未能保留关键证据，要么由于搜索空间不断扩大导致检索竞争，检索器无法定位相关条目。为应对这些挑战，我们提出了MemLife，一种多模态记忆系统，它构建基于实体的第一人称文本情节，并通过时间索引的智能体读取器进行检索。在无需训练且查询时无需访问视频的情况下，MemLife在四个长时程基准测试中比最强的免训练基线提升了4.6%至12.0%。为进一步提升记忆质量，我们提出了MemOpt，一种基于强化的方法（摘要在此处被截断）。

    arXiv:2609.40195v1 Announce Type: cross  Abstract: Long-term egocentric video enables personalized AI assistants to reason about daily life. However, as video histories grow to hundreds of hours spanning months or years, reprocessing raw clips for every query becomes computationally prohibitive. Memory systems offer a scalable alternative by compacting videos into text representations, but often fail on practical benchmarks: either the memory does not preserve key evidence, or the retriever fails to locate relevant entries due to retrieval competition in growing search spaces. To address these challenges, we introduce MemLife, a multimodal memory system that constructs entity-grounded, first-person text episodes and retrieves them via a time-indexed agentic reader. Without training or query-time video access, MemLife improves over the strongest training-free baseline by 4.6--12.0% across four long-horizon benchmarks. To further improve memory quality, we propose MemOpt, a reinforcement
    
[^15]: 绘制廉价，信任昂贵：为测试时扩展曲线提供统计认证

    Cheap to Draw, Expensive to Trust: Certifying Test-Time Scaling Curves

    [https://arxiv.org/abs/2609.40190](https://arxiv.org/abs/2609.40190)

    该论文推导了统计认证整条测试时扩展曲线的极小极大采样成本，并证明利用“基准是固定题目列表、方差主要来自题目之间”这一结构，可以避免为错误的不确定性买单，从而大幅降低同时认证所有预算所需的生成样本量。

    

    采样多个答案并保留验证器评分最高的那一个，是在测试时换取准确率的最简单方法之一。其效果通常以缩放曲线的形式报告：即准确率随采样答案数量 $k$ 变化的曲线。这条曲线画起来便宜，信起来却很贵。从曲线上读出的预算是在查看所有数据点之后才选定的，因此只有一条能同时覆盖所有预算的置信带才能保护这一选择；在一个包含100道题的基准上，固定的精确二项式设计需要生成192,000个答案，才能以95%的置信度将64个预算认证到 ±1/32 的精度。而这些成本的大部分，实际上为错误的不确定性买了单。基准测试是一个固定的问题列表；在预算为64时，所选答案正确性的方差约四分之三来自题目之间的差异，而一个会重新审视每道题的审计无需为此支付代价。我们推导出了认证整条曲线的极小极大成本（在对数因子意义下）。该成本由三部分组成：校准验证器评分分布的尾部……（摘要原文在此处截断）

    arXiv:2609.40190v1 Announce Type: cross  Abstract: Sampling several answers and keeping the one a verifier scores highest is one of the simplest ways to buy accuracy at test time. Its effect is reported as a scaling curve: accuracy against the number $k$ of sampled answers. The curve is cheap to draw and expensive to trust. A budget read off it is chosen after looking at every point, so only a band that covers all budgets at once protects the choice, and on a 100-question benchmark a fixed exact-binomial design needs 192,000 generated answers to certify 64 budgets to within $\pm1/32$ at 95%. Most of that cost pays for the wrong uncertainty. A benchmark is a fixed list of questions; at budget 64, about three quarters of the variance of a selected answer's correctness lies between questions, and an audit that revisits every question need not pay for it. We derive the minimax cost of certifying the whole curve, up to logarithmic factors. It has three parts: calibrating the tail of the sco
    
[^16]: 基于隐马尔可夫模型的可证明易处理的非确定有限自动机约束语言生成

    Provably Tractable NFA-Constrained Language Generation via HMMs

    [https://arxiv.org/abs/2609.40185](https://arxiv.org/abs/2609.40185)

    该论文提出了NFA-LM，一个基于#NFA问题FPRAS理论的多项式时间生成引擎，首次在温和假设下以理论保证高效解决NFA约束语言生成问题，克服了现有方法扭曲分布或牺牲效率的缺陷。

    

    受约束生成旨在从语言模型中采样并满足硬性约束条件。现有的针对非确定有限自动机（NFA）约束的受约束生成技术要么扭曲分布，要么牺牲效率。从理论上讲，该任务可归约为统计NFA所接受的长度为n的序列数量问题（#NFA），而精确求解#NFA问题是#P完全的。近期研究表明，#NFA问题存在完全多项式随机近似方案（FPRAS）。受此结果启发，我们提出了NFA-LM，这是一个在温和假设下具有理论保证的、用于NFA约束生成的多项式时间引擎。实验表明，NFA-LM能够高效生成高质量输出，且其近似误差具有理论界。

    arXiv:2609.40185v1 Announce Type: new  Abstract: Constrained generation aims to sample from language models (LMs) conditioned on hard constraints. Existing constrained-generation techniques for nondeterministic finite automaton (NFA) constraints either distort the distribution or sacrifice efficiency. Theoretically, this task reduces to counting the length-$n$ sequences accepted by an NFA (#NFA), and the exact #NFA problem is #P-complete. Recent work has shown that #NFA admits a fully polynomial randomized approximation scheme (FPRAS). Inspired by this result, we propose NFA-LM, a polynomial-time engine for NFA-constrained generation with theoretical guarantees under mild assumptions. Experiments show that NFA-LM efficiently generates high-quality outputs with theoretically bounded approximation error.
    
[^17]: Index-Translate：一个多语言翻译模型家族——文本、语音、可控配音与长文档翻译

    Index-Translate: A Multilingual Translation Model Family -- Text, Speech, Controlled Dubbing, and Long-Document Translation

    [https://arxiv.org/abs/2609.40181](https://arxiv.org/abs/2609.40181)

    Index-Translate是一个支持150种语言的多语言翻译模型家族，统一覆盖文本、语音、可控配音和长文档翻译，以较小规模实现了与千亿级翻译模型和前沿模型相当的性能。

    

    我们推出了Index-Translate，这是一个多语言翻译模型家族，它将共享的多语言基础模型与针对通用翻译、指令遵循、语音翻译、可控配音和长文档翻译的专项训练相结合。该家族包含三种模型规模（2B、9B和35B-A3B），支持150种语言的翻译，并具备多语言指令遵循能力。在通用翻译和复杂翻译指令方面的评估表明，Index-Translate优于同等规模的翻译模型，并达到了与1000亿参数级翻译模型及前沿模型相当的性能。Index-Echo提供端到端的语音到文本和语音到语音翻译，其性能超越了现有的端到端模型，并达到了与前沿全模态模型相当的水平。Index-Homura将该模型家族扩展至音节可控的配音领域。Index-NativeLong则引入了原生长文档翻译……

    arXiv:2609.40181v1 Announce Type: new  Abstract: We introduce Index-Translate, a multilingual translation model family that combines a shared multilingual foundation with specialized training for general translation, instruction following, speech translation, controlled dubbing, and long-document translation. It includes three model sizes, 2B, 9B, and 35B-A3B, and supports translation in 150 languages, with multilingual instruction following. Evaluations on general translation and complex translation instructions show that Index-Translate outperforms translation models of comparable size and achieves performance comparable to 100B-scale translation models and frontier models. Index-Echo provides end-to-end speech-to-text and speech-to-speech translation, outperforming existing end-to-end models and achieving performance comparable to frontier omni models. Index-Homura extends the family to syllable-controlled dubbing. Index-NativeLong introduces native long-document translation with a 
    
[^18]: 用于神经网络压缩的功能子空间学习

    Learning Functional Subspaces for Neural Network Compression

    [https://arxiv.org/abs/2609.40127](https://arxiv.org/abs/2609.40127)

    本文提出可学习子空间投影方法，通过端到端联合优化正交投影器并基于全局目标（KL散度或原始训练损失）来学习神经网络权重矩阵中应丢弃的子空间，解决了传统局部闭式准则忽略误差传播导致高压缩率下性能崩溃的问题。

    

    现代Transformer在拥有强大能力的同时，也伴随着巨大的内存和计算需求。低秩权重分解可以在保持矩阵稠密的前提下同时降低这两者，因此在标准硬件上依然高效。然而，现有方法使用局部闭式准则来选择每个权重矩阵中要移除的子空间：激活能量、逐层重构误差或损失的二次近似。这些准则忽略了误差在网络中的传播方式，因此在高压缩率下，误差会随深度累积，导致性能崩溃。我们提出了可学习子空间投影，它改为端到端地学习要丢弃的子空间。每个线性层，或读取相同激活值的绑定层组，都被分配一个正交投影器。所有投影器针对一个全局目标进行联合优化——即与稠密模型输出分布的KL散度，或模型原始训练损失——

    arXiv:2609.40127v1 Announce Type: cross  Abstract: Modern transformers pair impressive capabilities with substantial memory and compute demands. Low-rank weight factorization reduces both while keeping the matrices dense, and thus efficient on standard hardware. Existing methods, however, choose the subspace to remove from each weight matrix with local closed-form criteria: activation energy, layer-wise reconstruction error, or a quadratic approximation of the loss. These criteria ignore how errors propagate through the network, so at high compression the errors compound with depth and performance collapses. We introduce Learnable Subspace Projections (LSP), which instead learns the subspaces to discard end-to-end. Each linear layer, or tied group of layers that read the same activations, is assigned an orthogonal projector. All projectors are optimized jointly against a global objective--the KL divergence to the dense model's output distribution or the model's original training loss--
    
[^19]: 自行去偏：教会大语言模型认知偏差缓解干预方法

    Debias It Yourself: Teaching LLMs Cognitive Bias Mitigation Interventions

    [https://arxiv.org/abs/2609.40124](https://arxiv.org/abs/2609.40124)

    该论文提出以认知科学为基础的 DIY 框架，将五种经过验证的去偏干预方法通过 Show（上下文示例）、Train（指令微调）、Revise（自我修订）三种范式应用于大语言模型，在偏差-推理权衡中取得最优表现，并在未见维度上将偏差最多降低 14.8%。

    

    偏差问题在社会心理学和认知科学中已被长期研究，数十年的研究产生了一系列经过验证的干预方法，能够减少人类的刻板思维和偏见性反应。我们提出了 Debias It Yourself (DIY)，一个以认知科学为基础的框架，将五种此类干预方法转化为大语言模型的去偏程序，并通过三种成熟的范式进行传递：Show（上下文示例）、Train（指令微调）和 Revise（引导式自我修订）。在三个模型、五个偏差基准、十一个去偏基线和三个推理基准上的实验表明，Train+Revise 和单独的 Revise 获得平均排名前两位，在偏差-推理权衡中表现领先（在 90% 推理准确率下平均偏差低至 2%），并在未见过的偏差维度上最多减少 14.8% 的偏差。我们的代码和数据已公开。

    arXiv:2609.40124v1 Announce Type: new  Abstract: Bias has long been studied in social psychology and cognitive science, where decades of research have produced a body of validated interventions that reduce stereotypical thinking and prejudiced responses in humans. We propose Debias It Yourself (DIY), a cognitively grounded framework that translates five such interventions into debiasing procedures for large language models and delivers them through three established paradigms: Show (in-context examples), Train (instruction tuning), and Revise (guided self-revision). Across three models, five bias benchmarks, eleven debiasing baselines, and three reasoning benchmarks, Train+Revise and Revise alone attain the top two average ranks, lead the bias-reasoning tradeoff (mean bias as low as 2% at 90% reasoning accuracy), and reduce bias on unseen dimensions by up to 14.8%. Our code and data are publicly available.
    
[^20]: 论AMR增强对大语言模型的（无）有效性

    On the (In)effectiveness of AMR Augmentation for Large Language Models

    [https://arxiv.org/abs/2609.40121](https://arxiv.org/abs/2609.40121)

    研究通过复现实验和基于困惑度的探测方法发现，AMR增强对现代大语言模型无效，纯文本基线的性能始终能够匹配或超越AMR增强模型。

    

    尽管抽象含义表示（AMR）在历史上曾提升过一系列NLP任务的性能，但AMR增强对现代大语言模型的收益——或缺乏收益——至今尚不明确。在本文中，我们尝试复现近期报告AMR增强带来显著下游收益的研究工作，发现这些收益很可能源于其实验设置中的特定选择：通过采用一致且统一的超参数选择协议，我们观察到纯文本基线始终能够匹配或超越AMR增强模型的性能。为了探究这一阴性结果，我们引入了一种基于困惑度的探测方法，用于衡量AMR为大语言模型提供了多少模型本身尚不具备的补充性关系知识。我们发现AMR增强并不能帮助大语言模型提升对句子中关系内容的理解，这表明用AMR增强这些模型并无益处。

    arXiv:2609.40121v1 Announce Type: cross  Abstract: While Abstract Meaning Representation (AMR) has historically improved performance on a range of NLP tasks, the benefit---or lack thereof---of AMR augmentation for modern LLMs is thus far unclear. In this paper, we attempt to reproduce recent work that reported substantial downstream gains from AMR augmentation, finding that these are likely due to specific choices in the experimental settings used: using a consistent and unified protocol for hyperparameter selection, we observe that text-only baselines consistently match or exceed the performance of AMR-augmented models. To investigate this null result, we introduce a perplexity-based probe measuring the degree to which AMR provides an LLM with supplemental relational knowledge not already available to the model. We find that AMR augmentation does not help LLMs improve their understanding of relational content in the sentence, indicating that augmenting these models with AMR offers no 
    
[^21]: 面向LLM智能体高效内存压缩的持久化上下文图

    Persistent Context Graphs for Efficient Memory Compaction in LLM Agents

    [https://arxiv.org/abs/2609.40118](https://arxiv.org/abs/2609.40118)

    提出ReCAP方法，通过在轻量级持久化上下文图中存储注意力推导的重要性分数和依赖链接，结合新请求的相关性线索，实现LLM智能体高效且低开销的内存压缩。

    

    随着LLM能力的不断进步，智能体正在处理时间跨度更长、日益复杂的任务。其不断增长的交互历史使得内存压缩对于保持在上下文窗口内以及降低预填充（prefill）成本变得至关重要。现有方法对历史进行摘要总结或压缩其KV缓存，通常需要增加模型计算量来为未来请求保留信息。新的用户请求可能会改变哪些历史内容是重要的，但如果KV缓存已过期，则使用模型重新评估历史需要对历史进行重新编码。过去的注意力机制提供了历史重要性和消息间依赖关系的信号，而与当前任务的相关性则必须基于新的用户请求来评估。我们提出了ReCAP，一种内存压缩方法，它将注意力推导的重要性分数和依赖链接存储在一个轻量级、持久化的上下文图中。对于每个新请求，ReCAP将存储的重要性与来自请求的相关性线索相结合……（原文摘要在此处截断）

    arXiv:2609.40118v1 Announce Type: new  Abstract: As LLM capabilities advance, agents are tackling increasingly complex tasks over longer horizons. Their growing interaction histories make memory compaction essential for staying within context windows and reducing prefill cost. Existing methods summarize the history or compress its KV cache, often adding model computation to preserve information for future requests. A new user request can change which history matters, but reassessing that history with the model requires re-encoding it if the KV cache has expired. Past attention provides signals of historical importance and dependencies between messages, while relevance to the current task must be assessed using the new user request. We introduce ReCAP, a memory compaction method that stores attention-derived importance scores and dependency links in a lightweight, persistent context graph. For each new request, ReCAP combines stored importance with relevance cues from the request and fo
    
[^22]: 智能体错误数据集：扩展50,000个错误-诊断对用于失败分析与错误感知后训练

    Agent Error Dataset: Scaling 50,000 Error--Diagnosis Pairs for Failure Analysis and Error-Aware Post-Training

    [https://arxiv.org/abs/2609.40111](https://arxiv.org/abs/2609.40111)

    该论文发布了包含50,228个错误-诊断对、覆盖33个环境和23个策略模型的智能体错误数据集（AED），并配套五阶段AET流水线，用于LLM智能体的失败分析与错误感知后训练。

    

    一次不成功的LLM智能体轨迹所包含的信息比其最终奖励更多：智能体可获得的观测、它所选择的动作以及环境的响应。要将这些经验复用于学习，需要识别出需要修订的决策并测试具体的替代方案。我们提出了智能体错误数据集（AED），包含来自文本智能体系统中33个环境、19个框架家族和23个策略模型的9,961个源任务中的50,228个错误-诊断对。我们保留了源轨迹和执行元数据，以支持跨设置的失败分析和重新诊断，而无需重复原始轨迹。我们的五阶段“智能体错误到训练”（AET）流水线收集自然发生的失败，生成诊断与修正建议，并将其与记录的证据进行核对。在支持重放的场景中，我们在匹配的执行设置下，将修正方案与来自同一检查点的原始动作重试进行比较。

    arXiv:2609.40111v1 Announce Type: new  Abstract: An unsuccessful LLM agent rollout contains more information than its final reward: the observations available to the agent, the actions it chose, and the environment's responses. Reusing this experience for learning requires identifying a decision to revise and testing a concrete alternative. We introduce the Agent Error Dataset (AED), comprising 50,228 error-diagnosis pairs from 9,961 source tasks across 33 environments, 19 harness families, and 23 policy models in text-based agent systems. We retain source traces and execution metadata to support cross-setting failure analysis and re-diagnosis without repeating the original rollout. Our five-stage Agentic Error-to-Training (AET) pipeline collects natural failures, generates diagnoses and proposed corrections, and checks them against recorded evidence. Where replay is supported, we compare corrections with original-action retries from the same checkpoint under matched execution settings
    
[^23]: OverdoseMoE：用于阿片类药物过量风险预测的多专家框架

    OverdoseMoE: A Multi-Expert Framework for Opioid Overdose Risk Prediction

    [https://arxiv.org/abs/2609.40108](https://arxiv.org/abs/2609.40108)

    本文提出了OverdoseMoE多专家框架，通过对纵向ICD诊断序列进行诊断特异性继续预训练与微调，并利用互补专家加权策略整合不同规模的模型，显著提升了180天阿片类药物过量风险预测性能（AUPRC达25.17，AUROC达69.49）。

    

    阿片类药物过量仍然是一个重大的临床和公共卫生负担，凸显了对可扩展方法来识别高危患者的需求。本研究针对基于患者既往一年纵向ICD诊断历史的180天阿片类药物过量风险预测，探索了诊断特异性的模型适配方法。我们通过对纵向诊断序列进行继续预训练并随后进行任务特定微调，开发了OODMAMBA和OODQWEN两个模型。在基于Qwen的更强预测器的基础上，我们进一步提出了OVERDOSEMOE，这是一个多专家框架，使用互补的专家加权策略整合不同规模的模型。诊断特异性适配的预测性能始终优于通用语言模型基线，其中OODQWEN实现了24.47的AUPRC和68.56的AUROC。OVERDOSEMOE进一步提升了判别能力和精确度，实现了25.17的AUPRC和69.49的AUROC，同时优于（原文截断）。

    arXiv:2609.40108v1 Announce Type: new  Abstract: Opioid overdose remains a major clinical and public health burden, highlighting the need for scalable approaches to identify patients at high risk. Here, we investigate diagnosis-specific adaptation for 180-day opioid overdose risk prediction from patients' preceding one-year longitudinal ICD histories. We develop OODMAMBA and OODQWEN through continued pretraining on longitudinal diagnostic sequences followed by task-specific fine-tuning. Building on the stronger Qwen-based predictors, we further propose OVERDOSEMOE, a multi-expert framework that integrates models of different scales using complementary expert-weighting strategies. Diagnosis-specific adaptation consistently improved predictive performance over general-purpose language-model baselines, with OODQWEN achieving an AUPRC of 24.47 and an AUROC of 68.56. OVERDOSEMOE further improved discrimination and precision, achieving an AUPRC of 25.17 and an AUROC of 69.49 while outperform
    
[^24]: JuryFlow：分歧引导的人机协同多智能体评估

    JuryFlow: Disagreement-Guided Human-in-the-Loop Multi-Agent Evaluation

    [https://arxiv.org/abs/2609.40103](https://arxiv.org/abs/2609.40103)

    JuryFlow是一个分歧引导的人机协同多智能体评估框架，它将LLM评审器之间的分歧视为声明级的不确定性信号，通过将响应分解为原子声明、构建熵评分的分歧图，并由人类以单次最小干预精准化解评审冲突。

    

    大语言模型（LLM）越来越多地被用作AI生成内容的自动化评审器，然而单一评审器并不可靠，即使是评审小组也会留下一个棘手问题：当评审者之间意见不一致时，多数投票只是丢弃了冲突，而并未解决它。我们提出了JuryFlow，一个分歧引导的、人机协同（human-in-the-loop）的多智能体评估框架，它不将评审者之间的分歧视为需要平均掉的噪声，而是将其作为精确的、声明级（claim-level）信号，用以指示评估在何处存在不确定性。JuryFlow将每个候选响应分解为原子声明，由一组异构评审器对每条声明给出判定，并构建一个分歧图，其节点以判定熵评分，其边则编码声明之间的结构相似性。人类作为结构引导者，通过单次最小干预来选择解决哪个分歧，而非对整个响应重新标注，此后框架聚焦于……（摘要原文在此处截断）

    arXiv:2609.40103v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as automated judges for AI-generated content, yet a single judge is unreliable and even a panel of judges leaves a hard residue: when judges disagree, majority voting discards the conflict instead of resolving it. We present JuryFlow, a disagreement-guided, human-in-the-loop multi-agent evaluation framework that treats inter-judge disagreement not as noise to be averaged away, but as a precise, claim-level signal indicating where an evaluation is uncertain. JuryFlow decomposes each candidate response into atomic claims, has a panel of heterogeneous judges assign per-claim verdicts, and builds a disagreement graph whose nodes are scored by verdict entropy and whose edges encode structural similarity between claims. A human acts as a structural guide, selecting which disagreement to resolve through a single, minimal intervention rather than re-labeling the response, after which the foc
    
[^25]: AutoDataBench：一个用于加速自动化研究的数据中心化测试平台

    AutoDataBench: A Data-centric Testbed for Accelerating Auto Research

    [https://arxiv.org/abs/2609.40097](https://arxiv.org/abs/2609.40097)

    AutoDataBench是一个隔离数据因素的可控测试平台，首次系统评估大语言智能体的“数据智能”，即通过数据诊断、组织与构建迭代改进训练数据的能力。

    

    现有的自动化研究基准通常将多种改进来源混杂在一起，包括训练框架、超参数、计算预算和数据，这使得难以将一个前沿智能体优于另一个智能体的原因归因于特定的研究能力。在本工作中，我们隔离并系统地评估“数据智能”，即智能体理解、操作和改进塑造模型能力的数据的能力。我们提出了AutoDataBench，一个基于数据智能概念框架构建的可控测试平台，该框架涵盖数据诊断、数据组织和数据构建三个维度，并通过三个精心设计的优化任务进行实例化，同时保持非数据因素固定不变。在工具使用、检索和知识注入等方面，我们评估了前沿大语言模型在任务特定资源预算下通过迭代实验改进训练数据的能力。除了优化性能之外，我们还探讨：大语言模型是否真正理解（摘要在此处截断）

    arXiv:2609.40097v1 Announce Type: new  Abstract: Existing auto-research benchmarks often entangle multiple sources of improvement, including training frameworks, hyperparameters, compute budgets, and data, making it difficult to attribute why one frontier agent outperforms another to specific research capabilities. In this work, we isolate and systematically evaluate Data Intelligence: an agent's ability to understand, manipulate, and improve the data that shapes model capabilities. We introduce AutoDataBench, a controlled testbed built on a conceptual framework of data intelligence spanning data diagnosis, data organization, and data construction, instantiated through three highly curated optimization tasks while holding non-data factors fixed. Across tool use, retrieval, and knowledge injection, we evaluate frontier LLMs' ability to improve training data through iterative experimentation under task-specific resource budgets. Beyond optimization performance, we ask: do LLMs understand
    
[^26]: 从推文到交易：分析土耳其公众情绪对股市表现的影响

    From Tweets to Trades: Analyzing the Influence of Public Mood over Stock Market Performance in Turkiye

    [https://arxiv.org/abs/2609.40064](https://arxiv.org/abs/2609.40064)

    该研究基于61万余条X平台帖子，利用微调的土耳其语Transformer模型构建分领域的公众情绪指标，并采用多种计量经济学方法考察其与土耳其BIST100/BIST30股市表现之间的关联，发现公众情绪与市场动态的关系因传播领域和市场状况而异。

    

    目的：本研究考察特定领域的公众情绪是否与股市动态相关，以及这些关系是否因传播领域和市场状况的不同而变化。研究将公众情绪与投资者情绪区分开来，并探讨异质性的公众传播来源是否与市场行为之间存在不同的关系。设计：本研究分析了2022年1月至2023年12月期间由176个经筛选的X账号发布的610,422条帖子，涵盖政治与政府、经济与金融以及媒体与社会三个领域。帖子使用在三个特定领域模式和一个合并模式下微调的土耳其语Transformer模型进行分类。公众情绪指标按日、周、月三个频率构建，并结合BIST100和BIST30市场指标，在整个研究期及选定的……（期间）采用相关性分析、格兰杰因果检验、向量自回归和脉冲响应分析进行检验。

    arXiv:2609.40064v1 Announce Type: new  Abstract: Purpose: This study examines whether domain-specific public mood is associated with stock-market dynamics and whether these relationships vary across communication domains and market conditions. It distinguishes public mood from investor sentiment and investigates whether heterogeneous sources of public communication exhibit different relationships with market behaviour.   Design: The study analyses 610,422 posts published by 176 curated X accounts between January 2022 and December 2023, covering Politics and Government, Economy and Finance, and Media and Society. Posts are classified using fine-tuned Turkish transformer models under three domain-specific and one pooled regime. Public mood measures are constructed at daily, weekly, and monthly frequencies and examined alongside BIST100 and BIST30 market measures using correlation, Granger causality, vector autoregression, and impulse response analyses across the full period and selected 
    
[^27]: LARC：面向冻结模型学习的低秩自适应残差连接

    LARC: Low-Rank Adaptive Residual Connections for Learning in Frozen Models

    [https://arxiv.org/abs/2609.40063](https://arxiv.org/abs/2609.40063)

    LARC 通过在冻结模型中引入低秩自适应残差连接，利用跨任务学习初始因子的慢状态与随反馈动态调整的快状态相结合的双状态机制，使冻结模型无需更新原始参数即可从反馈中学习，在程序选择任务中显著降低查询执行误差。

    

    低秩自适应残差连接（LARC）为冻结模型提供了一个能够从反馈中学习的紧凑数值状态。映射 $h+BAh$ 在隐藏表示上添加一个低秩修正。其中，一个慢状态 $\rho$ 跨任务学习初始因子；一个私有的快状态 $\Phi$ 复制这些因子、随反馈变化，并重置为训练后的初始化。本报告给出了记忆中介学习架构（Memory-Mediated Learning Architecture）中数值策略载体在输入侧的一种实现，并考察了其因子空间动力学与学习生命周期。我们在冻结的 MiniCPM5-1B-SFT 底座模型上研究了一个具有 12,288 个可训练参数的秩为 4 的输入残差。在一个四候选程序选择任务中，仅两次反馈梯度步骤即可使预期查询执行误差相对于重置为各自的静态训练初始化和适应后初始化分别降低 24.65 和 36.65 个百分点。这些开发性结果覆盖了 16 个参数组（摘要文本在此处被截断）。

    arXiv:2609.40063v1 Announce Type: cross  Abstract: Low-Rank Adaptive Residual Connections (LARC) give a frozen model a compact numerical state that can learn from feedback. The map $h+BAh$ adds a low-rank correction to a hidden representation. A slow state $\rho$ learns starting factors across tasks; a private fast state $\Phi$ copies them, changes with feedback, and resets to the trained initialization. This report specifies an input-side realization of the numerical policy carrier in Memory-Mediated Learning Architecture and examines its factor-space dynamics and learning lifetime. We study a rank-4 input residual with 12,288 trainable parameters on a frozen MiniCPM5-1B-SFT substrate. In a four-candidate program-selection task, two feedback-gradient steps reduce expected query execution error by 24.65 and 36.65 percentage points relative to resetting to the respective trained static and post-adaptation initializations. These development results cover 16 parameter groups and three pai
    
[^28]: MGhana-ST：面向加纳语言的低资源语音翻译数据集及多语言训练权衡分析

    MGhana-ST: A Low-Resource Speech Translation Dataset for Ghanaian Languages and an Analysis of Multilingual Training Trade-offs

    [https://arxiv.org/abs/2609.40041](https://arxiv.org/abs/2609.40041)

    该论文发布了面向加纳四种低资源语言的语音翻译数据集MGhana-ST，并发现在严重数据稀缺的场景下，多语言联合训练相比单语言训练并无收益，甚至会显著降低埃维语和芳蒂语的翻译性能。

    

    我们提出了MGhana-ST，这是一个面向四种低资源加纳语言变体的语音翻译数据集：加语、契维语（Akuapem和Asante方言）、埃维语和芳蒂语。MGhana-ST是一项持续进行的标注工作；本文的实验使用了约16.1小时的语音与英文翻译配对数据的固定子集。音频选自两个现有的加纳语音资源。与这些资源不同的是，英文翻译由37名母语标注者直接根据音频生成，并包含言语和非言语事件标注。使用Whisper-small模型，我们在严重数据稀缺的条件下比较了单语言与多语言训练，并报告了三次随机种子的平均值。在这种情形下，平铺式多语言训练对任何语言变体均无收益。加语和契维语在种子方差范围内保持不变（相比单语言训练的标准差1.63和2.20，BLEU仅分别提升0.51和0.06），而埃维语下降了6.99 BLEU，芳蒂语下降了5.11。表现下降的语言变体是埃维语，它是……（摘要在此处被截断）

    arXiv:2609.40041v1 Announce Type: new  Abstract: We present MGhana-ST, a speech translation dataset for four low-resource Ghanaian language varieties: Ga, Twi (Akuapem and Asante), Ewe, and Fante. MGhana-ST is an ongoing annotation effort; the experiments here use a fixed subset of about 16.1 hours of paired speech and English translations. The audio is curated from two existing Ghanaian speech resources. Unlike in those resources, the English translations are produced directly from audio by 37 native-speaker annotators and include verbal and non-verbal event annotations.   Using Whisper-small, we compare monolingual and multilingual training under severe data scarcity, reporting means over three seeds. Flat multilingual training benefits no variety in this regime. Ga and Twi are unchanged within seed variance (+0.51 and +0.06 BLEU against monolingual standard deviations of 1.63 and 2.20), while Ewe declines by 6.99 BLEU and Fante by 5.11. The degrading varieties are Ewe, which is ling
    
[^29]: OPTS-TTPO：利用树搜索增强有限样本策略梯度学习

    OPTS-TTPO: Enhancing Finite-Sample Policy-Gradient Learning with Tree Search

    [https://arxiv.org/abs/2609.40035](https://arxiv.org/abs/2609.40035)

    提出 OPTS 并行树搜索与 TTPO 树轨迹策略优化方法，通过在已访问状态处从当前策略采样新后缀来构建同策略树轨迹，在固定预算内提升策略梯度学习对罕见高回报轨迹的覆盖，且无需动作分布校正。

    

    策略梯度定理给出了当前策略下的精确梯度，但有限的同策略样本可能遗漏罕见的高回报轨迹。我们研究在固定预算下，树搜索能否提高对这些轨迹的覆盖，同时控制梯度偏差。我们提出了同策略并行树搜索（On-Policy Parallel Tree Search, OPTS）以及使用同策略树轨迹的树轨迹策略优化（Tree Trajectory Policy Optimization, TTPO），该方法在已访问的状态处从当前策略采样新的后缀。尽管分支会改变状态的访问分布，但这无需对动作分布进行校正。我们的分支聚合引理表明，当分支选择与权重在采样出边转移之前被固定时，基于分支加权的树统计量能够恢复链式期望。OPTS 使用估计的性能差异来选择扩展状态。在确定性动力学、精确值函数以及最大备份优势的条件下，所诱导的搜索策略的期望回报随预算单调提升……

    arXiv:2609.40035v1 Announce Type: new  Abstract: The policy-gradient theorem gives the exact gradient under the current policy, but finite on-policy samples may miss rare high-return trajectories. We study whether tree search improves their coverage within a fixed budget while controlling gradient bias. We introduce On-Policy Parallel Tree Search (OPTS) and Tree Trajectory Policy Optimization (TTPO) using on-policy tree trajectories, which sample new suffixes from the current policy at visited states. This needs no action-distribution correction, although branching changes state visitation. Our Branch Aggregation Lemma shows that branch-weighted tree statistics recover chain expectations when branch choices and weights are fixed before outgoing transitions are sampled. OPTS selects expansion states using estimated performance differences. Under deterministic dynamics, exact values, and max-backup advantages, the induced search policy's expected return improves monotonically with the bu
    
[^30]: Mid-Harness：面向终端智能体的模型与执行框架之间的动作扩展

    Mid-Harness: Scaling Actions Between Model and Harness for Terminal Agents

    [https://arxiv.org/abs/2609.39982](https://arxiv.org/abs/2609.39982)

    Mid-Harness 通过在模型与执行框架之间对候选动作进行采样并借助强验证器筛选，在不改变模型和框架的情况下提升终端智能体动作执行的可靠性与成功率。

    

    终端智能体通过随机性的模型生成来执行动作，然而生成有用动作的能力并不能保证其可靠执行。一条糟糕的命令（例如错误的软件包安装）可能以阻碍后续进展的方式改变环境，即使模型本可以生成更好的替代方案。我们研究了在模型与执行框架（harness）的边界处分配测试时计算能否提升动作的可靠性和轨迹成功率，以及是什么使这种分配有效。为了研究这些问题，我们提出了 Mid-Harness，它在转发某个候选动作供执行之前对其进行采样和验证，同时保持生成器和执行框架本身不变。在使用 TMAX-9B 生成器时，在弱验证条件下增加动作采样几乎不会带来收益，而一个能力强的验证器则能够从同一生成器中利用有用的替代动作。在 TerminalBench-Lite 上，一个 GPT-5.6 Sol 验证器将基础智能体的 Pass@1 从 50.00% 提升至 6……（原文摘要在此处截断）

    arXiv:2609.39982v1 Announce Type: cross  Abstract: Terminal agents act through stochastic model generations, yet the ability to generate a useful action does not ensure its reliable execution. A poor command (e.g., wrong package install) can change the environment in ways that hinder subsequent progress, even when the model could generate a better alternative. We investigate whether allocating test-time compute at the model-harness boundary can improve action reliability and trajectory success, and what makes this allocation effective. To study these questions, we introduce Mid-Harness, which samples and verifies candidate actions before forwarding one for execution, while keeping the generator and harness unchanged. With a TMAX-9B generator, more action sampling yields little benefit under weak verification, whereas a capable verifier can exploit useful alternatives from the same generator. On TerminalBench-Lite, a GPT-5.6 Sol verifier raises Pass@1 from 50.00% for the base agent to 6
    
[^31]: BioASQ 2026概述：第十四届大规模生物医学语义索引与问答BioASQ挑战赛

    Overview of BioASQ 2026: The fourteenth BioASQ Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering

    [https://arxiv.org/abs/2609.39975](https://arxiv.org/abs/2609.39975)

    第十四届BioASQ挑战赛在CLEF 2026框架下设置了涵盖生物医学问答、多语言临床摘要、嵌套实体关系抽取、心脏病学临床编码和肠-脑信息抽取等六项共享任务，吸引了87支团队提交超过1000个参赛结果，持续推动生物医学语言处理领域的发展。

    

    本文概述了在2026年评测论坛会议与实验室（CLEF）框架下举办的第十四届BioASQ挑战赛。BioASQ是一项国际性挑战赛系列，旨在推动生物医学语言处理任务的进展，涵盖语义索引、信息抽取、问答和文本摘要等多个方向。2026年的BioASQ包含六项共享任务：a) Task 14b，关于生物医学语义问答；b) Task Synergy14，关于新兴生物医学主题的问答；c) Task MultiClinSum-2，关于多语言临床摘要生成；d) Task BioNNE-R，关于俄语和英语中嵌套命名实体间关系的抽取；e) Task ELCardioCC，关于心脏病学领域的临床编码；f) Task GutBrainIE，关于肠-脑相互作用的信息抽取。在这六项任务中，共有87支不同的团队参赛，累计提交超过1000个系统运行结果。与往届一样……

    arXiv:2609.39975v1 Announce Type: cross  Abstract: This paper presents an overview of the fourteenth edition of the BioASQ challenge, organized in the context of the Conference and Labs of the Evaluation Forum (CLEF) 2026. BioASQ is an international challenge series that supports progress in biomedical language processing tasks ranging from semantic indexing and information extraction to question answering and summarization. In 2026, BioASQ included six shared tasks: a) Task 14b on biomedical semantic question answering. b) Task Synergy14 on question answering for developing biomedical top- ics. c) Task MultiClinSum-2 on multilingual clinical summarization. d) Task BioNNE-R on extracting relations between nested named entities in Russian and English. e) Task ELCardioCC on clinical coding in cardiology. f) Task GutBrainIE on gut-brain interplay information extrac- tion. Across these six tasks, 87 distinct teams participated, submitting more than 1000 runs overall. As in previous edition
    
[^32]: UBTree：基于一元模型与二元模型的并行树状草稿生成投机解码方法

    UBTree: Parallel Tree Drafting via Unigram and Bigram Models for Speculative Decoding

    [https://arxiv.org/abs/2609.39972](https://arxiv.org/abs/2609.39972)

    UBTree通过将采用标准交叉熵训练的一元提议器与在高温度数据上以重归一化KL目标训练的轻量级二元选择器相结合来构建并行草稿树，从而在不牺牲并行性的前提下提升投机解码在高熵目标分布下的表现。

    

    投机解码通过在目标模型的一次前向传播中验证多个草稿词元来加速语言模型推理。近期的并行草稿生成器已在前沿生产模型中取得了突破性性能，但由于草稿多样性不足，其有效性会随着目标分布熵的增大而下降。为了在不牺牲并行性的前提下克服这一瓶颈，我们提出了UBTree——一种将一元提议器与二元选择器相结合以构建草稿树的并行草稿生成器。一元提议器采用标准交叉熵目标进行训练，为每个位置独立生成候选词元；而轻量级的二元选择器则预测相邻候选对之间的转移分数。与提议器不同，选择器在高温度数据上以重归一化的KL目标进行训练。这种树原生训练将监督扩展到贪心路径之外，鼓励……（摘要原文在此处截断）

    arXiv:2609.39972v1 Announce Type: new  Abstract: Speculative decoding accelerates language model inference by verifying multiple draft tokens in a single target-model pass. Recent parallel drafters have achieved breakthrough performance in frontier production models, but their effectiveness deteriorates as the entropy of target distributions increases due to insufficient draft diversity. To overcome this bottleneck without sacrificing parallelism, we introduce UBTree, a parallel drafter that couples a Unigram proposer with a Bigram selector to construct drafting Trees. The unigram proposer is trained with the standard cross-entropy objective to generate candidate tokens independently for each position, while a lightweight bigram selector predicts transition scores between adjacent candidate pairs. Unlike the proposer, the selector is trained with a renormalized KL objective on high-temperature data. This tree-native training broadens the supervision beyond the greedy path, encouraging 
    
[^33]: LEAP：面向长时音视频感知的学习式分块证据检索

    LEAP: Learned Block-wise Evidence Retrieval for Long Audio-Video Perception

    [https://arxiv.org/abs/2609.39938](https://arxiv.org/abs/2609.39938)

    LEAP通过分块式轻量级证据定位与检索，使小时级音视频问答的回答输入和上下文长度与录音时长无关，同时保留细粒度的声学与视觉证据。

    

    小时级的音视频问答受到“上下文困境”的制约：对整段录音进行密集编码会迅速耗尽上下文长度限制，而均匀的时间压缩则会严重稀释细粒度的声学与视觉证据。我们提出了LEAP，这是一个让模型自行检索证据、而无需将整段录音置于单一上下文中的框架。LEAP将录音划分为固定时长的块，并对每个块应用一个轻量级的定位通道来为较短的候选窗口打分。排名最高的窗口被池化后，在一个有界的回答通道中重新编码。因此，回答输入的规模和峰值上下文长度都与录音时长无关。通过将证据定位与推理过程解耦，我们的框架可以在预计算的转录文本上定位候选时间窗口，而无需解码媒体帧，同时通过将最终回答通道仅路由（至所选片段），保留了细粒度的视觉和非语音证据。

    arXiv:2609.39938v1 Announce Type: cross  Abstract: Hour-scale audio-visual question answering is constrained by a context dilemma: dense whole-recording encoding rapidly exhausts context limits, whereas uniform temporal compression severely dilutes fine-grained acoustic and visual evidence. We introduce LEAP, a framework where the model retrieves its own evidence without placing the whole recording in one context. LEAP divides a recording into fixed-duration blocks, applying a lightweight localization pass to each block to score short candidate windows. The highest-ranked windows are pooled and re-encoded in a single bounded answer pass. Consequently, the answer input and peak context remain independent of the recording duration. By decoupling evidence localization from reasoning, our framework can localize candidate temporal windows over pre-computed transcripts without decoding media frames, while preserving fine-grained visual and non-speech evidence by routing the final answering p
    
[^34]: RoPE已到穷途末路？长上下文失效的理论、诊断与缓解

    RoPE at the End of Its Rope? Theory, Diagnosis, and Mitigation of Long-Context Failures

    [https://arxiv.org/abs/2609.39929](https://arxiv.org/abs/2609.39929)

    该论文通过允许RoPE各频率上query-key缩放不相等的新理论，使RoPE的语义稳定性与位置敏感性权衡在注意力头和输入层面变得可度量，推导出上下文长度的理论界限，并据此提出针对长上下文失效的诊断与缓解方法。

    

    基于RoPE的语言模型的长上下文失效可能源于RoPE在维持稳定token偏好与区分相邻位置之间的内在权衡。要确定应解决哪个弱点以及如何解决，需要对训练后的模型中RoPE在不同上下文长度下的行为进行更精确的刻画。我们通过允许不同RoPE频率上采用不相等的query-key缩放，解决了先前理论的一个关键局限，这与实际经验观察高度吻合。我们的理论使这两种脆弱性对于单个注意力头和具体输入都可被度量，并量化了高频分量如何在支持位置敏感性的同时可能破坏语义稳定性。我们还推导出一个理论上的上下文长度界限：超过该界限后，在特定条件下，固定的注意力分数比较无法同时避免语义反转和位置不敏感性。在这些全新理论见解的指导下，我们引入RoP……

    arXiv:2609.39929v1 Announce Type: cross  Abstract: Long-context failures of RoPE-based language models can arise from RoPE's intrinsic tradeoff between maintaining stable token preferences and distinguishing nearby positions. Determining which weakness to address, and how, requires a more precise characterization of RoPE's behavior in trained models across context lengths. We address a key limitation of prior theory by allowing unequal query-key scales across RoPE frequencies, which aligns well with practical empirical observations. Our theory makes both vulnerabilities measurable for individual heads and inputs, and quantifies how high-frequency components support positional sensitivity while potentially disrupting semantic stability. We also derive a theoretical context-length bound beyond which, under specified conditions, a fixed attention-score comparison cannot jointly avoid semantic reversal and positional insensitivity. Guided by our fresh theoretical insights, we introduce RoP
    
[^35]: AdaGEPA：面向反思式提示优化的自适应反馈分配方法

    AdaGEPA: Adaptive Feedback Allocation for Reflective Prompt Optimization

    [https://arxiv.org/abs/2609.39927](https://arxiv.org/abs/2609.39927)

    AdaGEPA提出了一种自适应反馈分配方法，根据提示词的性能和已识别的弱点来智能选择反馈示例，在相同计算预算下比非自适应反馈选择方法获得更高的验证分数。

    

    提示优化通过改进提示词来提升语言模型系统在下游任务上的性能。经典方法在任务示例上评估提示词，并利用产生的反馈通过反思来指导提示词的修订。然而，当反馈选择未考虑提示词的弱点时，这些修订可能仅改善模型在所选示例上的表现，而无法带来更广泛的任务改进。为解决这一问题，我们提出了AdaGEPA，这是一种自适应反馈分配方法，利用提示词的性能和任务结构来选择用于下一次提示词修订的示例。该方法在每个反馈小批次中最多替换一个示例，以针对已识别的弱点，同时保留其余的反馈上下文。在六个下游基准测试的主要实验中，在相同的部署预算下，AdaGEPA比非自适应反馈选择方法取得了更高的平均验证分数。

    arXiv:2609.39927v1 Announce Type: new  Abstract: Prompt optimization improves the performance of language-model systems on downstream tasks by refining their prompts. Classical methods evaluate prompts on task examples and use the resulting feedback to guide prompt revisions through reflection. However, when feedback selection does not account for the prompt's weaknesses, these revisions may improve performance on selected examples without yielding broader task improvements. To address this issue, we propose AdaGEPA, an adaptive feedback-allocation method that uses the prompt's performance and task structure to select examples for the next prompt revision. Our method replaces at most one example in each feedback minibatch to target an identified weakness while preserving the remaining feedback context. Across our main experiments on six downstream benchmarks, AdaGEPA achieves higher mean validation scores than non-adaptive feedback selection under matched rollout budgets. AdaGEPA also 
    
[^36]: MCD：大型视觉语言模型中多模态上下文学习的因果蒸馏

    MCD: Causal Distillation of Multimodal In-Context Learning in Large Vision-Language Models

    [https://arxiv.org/abs/2609.39920](https://arxiv.org/abs/2609.39920)

    提出了多模态因果蒸馏（MCD）框架，通过结构保持的token干预识别并验证因果证据，将大型视觉语言模型教师在ICL中利用多模态证据的方式蒸馏给小模型，避免学生模型依赖语言先验等虚假线索。

    

    大型视觉语言模型（LVLM）展现出强大的多模态上下文学习（ICL）能力，然而这种能力会随着模型规模的减小而显著退化。知识蒸馏为弥合这一差距提供了一种自然的方法，但现有方法主要直接对齐输出分布或隐藏表示。这种对齐只教会学生模型教师预测了什么，却没有揭示复杂上下文中哪些证据在因果上支撑了该预测。因此，学生模型可能在模仿教师答案的同时，继续依赖语言先验、提示结构或其他虚假线索。为了解决这一局限，我们提出了多模态因果蒸馏（MCD），这是一个将强大教师在ICL过程中如何使用多模态证据进行迁移的蒸馏框架。MCD使用结构保持的token干预来识别和验证因果证据，然后迁移教师在该证据发生变化时的响应方式。

    arXiv:2609.39920v1 Announce Type: cross  Abstract: Large vision-language models (LVLMs) exhibit strong multimodal in-context learning (ICL) capabilities, yet this ability degrades substantially as model size decreases. Knowledge distillation offers a natural way to bridge this gap, but existing methods primarily align output distributions or hidden representations directly. Such alignment teaches the student what the teacher predicts without revealing which evidence in the complex context causally supports that prediction. Consequently, a student can imitate the teacher's answer while continuing to rely on language priors, prompt structure, or other spurious cues. To address this limitation, we introduce Multimodal Causal Distillation (MCD), a distillation framework that transfers how a strong teacher uses multimodal evidence during ICL. MCD uses structure-preserving token interventions to identify and verify causal evidence, then transfers how the teacher responds when that evidence i
    
[^37]: 具体与任意的差距：大语言模型的亲属关系推理对呈现方式并非无动于衷

    The Concrete-Arbitrary Gap: Kinship Reasoning in LLMs Is Not Indifferent to Presentation

    [https://arxiv.org/abs/2609.39913](https://arxiv.org/abs/2609.39913)

    大语言模型在解决用熟悉词汇表达的亲属关系问题时，准确率显著高于用明确定义的临时谓词表达的等价问题，且这一“具体-任意”差距可通过推理预算和提示语言干预大幅缩小，说明模型的关系推理能力依赖于呈现方式，但并非固定缺陷。

    

    我们测试了大语言模型在解决形式上完全匹配的亲属关系问题时，是否能在关系以熟悉词汇表达与以明确定义的临时谓词表达这两种情况下表现得同样好。在500对配对图上，具体词汇条件下的准确率超过任意谓词条件：本地Qwen3.8-27B高出35.6个百分点，Gemma 4 26B-A4B高出26.6个百分点，Gemma 4 31B高出12.0个百分点，Qwen3.8-Max高出5.4个百分点。所有四个配对差距在统计上均得到证实。推理预算和提示语言干预可以显著缩小这一差距，表明它是可改变的，而非固有的能力缺陷。最基本的结论是行为层面的：在这些任务上，模型所表现出的关系能力对呈现方式并非无动于衷。显式定义提供了形式化的关系，但并不能使临时谓词像嵌入在已习得语言关联中的熟悉词汇那样易于使用。

    arXiv:2609.39913v1 Announce Type: new  Abstract: We test whether large language models solve formally matched kinship problems equally well when relations are expressed in familiar vocabulary or by explicitly defined nonce predicates. Across 500 paired graphs, concrete accuracy exceeds arbitrary accuracy by 35.6 percentage points in local Qwen3.8-27B, 26.6 in Gemma 4 26B-A4B, 12.0 in Gemma 4 31B, and 5.4 in Qwen3.8-Max. All four paired gaps are statistically resolved. Reasoning budgets and prompt-language interventions can substantially reduce the difference, showing that it is modifiable rather than a fixed incapacity. The minimal conclusion is behavioral: on these tasks, the models' manifested relational competence is not indifferent to presentation. Explicit definitions provide the formal relations but do not make nonce predicates as usable as familiar vocabulary embedded in learned linguistic associations.
    
[^38]: OPSRD：在策略自角色蒸馏

    OPSRD: On-Policy Self-Role Distillation

    [https://arxiv.org/abs/2609.39884](https://arxiv.org/abs/2609.39884)

    OPSRD 让无角色的学生模型自行生成轨迹，再由同一基座模型的冻结角色化副本在这些前缀上给出角色条件分布，并用教师加权的正向 KL 散度蒸馏学生低估的备选词元，从而在无需参考解答的情况下实现在策略的角色知识自蒸馏。

    

    角色提示（role prompting）通过赋予大型语言模型一个专家身份来引出专门化行为，为在困难任务上引导推理提供了一种轻量级方式。然而，当采样得到的解答仍然错误时，对完整的角色提示答案进行评估或蒸馏可能会遗漏有用的下一词元（next-token）偏好；而要转移这些偏好，还需要一个能够触及学生模型很少预测的备选项的目标函数。我们提出 OPSRD，它将固定的专家角色作为特权教学上下文，实现无需参考解答的在策略自蒸馏：一个无角色的学生模型生成一条轨迹，同一基础模型的冻结实例在该轨迹的精确前缀上提供以角色为条件的分布，从而揭示出采样延续之外的备选方案；教师加权的正向 KL 散度以学生低估的备选项为目标，并通过截断（clipping）限制单个词表的贡献。监督被限制在……（摘要原文在此截断）

    arXiv:2609.39884v1 Announce Type: cross  Abstract: Role prompting elicits specialized behavior from large language models through an expert identity, offering a lightweight way to guide reasoning on demanding tasks. However, evaluating or distilling complete role-prompted answers can miss useful next-token preferences when the sampled solution remains incorrect. Transferring these preferences also requires an objective that reaches alternatives the student rarely predicts. We introduce OPSRD, which uses a fixed expert role as privileged teaching context for on-policy self-distillation without reference solutions. A role-free student generates a trajectory, and a frozen instance of the same base model supplies role-conditioned distributions on its exact prefixes, exposing alternatives beyond the sampled continuation. Teacher-weighted forward KL targets alternatives the student underestimates, with clipping to limit individual vocabulary contributions. Supervision is restricted to the hi
    
[^39]: 大语言模型人格遗忘

    LLM Persona Unlearning

    [https://arxiv.org/abs/2609.39882](https://arxiv.org/abs/2609.39882)

    该论文提出“人格遗忘”任务及PersonaUnlearnBench基准，通过权重级编辑使大语言模型中的指定人格难以被诱发，并发现标准遗忘方法无法在不牺牲生成质量或通用能力的情况下可靠地抹除目标人格。

    

    预训练使大语言模型（LLM）具备了与角色、风格、价值观和目标相关的广泛行为模式。后训练教会模型有条件地执行这些模式，并将乐于助人的“助手”设为默认，但并未从权重中抹除其他替代模式；因此，明确的提示可以诱发出持续影响判断、语言和行动的人格。在开放权重设置中，运行时的控制手段可能被移除，这促使了“人格遗忘”任务的出现：一种权重级别的编辑，使指定的人格在未见过的语境中难以被诱发和执行。我们提出了PersonaUnlearnBench，一个模型特定的配对基准，涵盖来自三个系列的六个大语言模型和五种人格，包含对齐的遗忘/保留数据集、留出的指令改写以及四轴评估。该基准表明，标准的遗忘方法无法在不牺牲有意义生成能力或通用效用的前提下可靠地抹除目标人格。

    arXiv:2609.39882v1 Announce Type: new  Abstract: Pre-training equips large language models (LLMs) with a broad repertoire of behavioral patterns associated with roles, styles, values, and goals. Post-training teaches conditional enactment and makes a helpful Assistant the default, but it does not erase alternative modes from the weights; explicit prompts can therefore elicit personas that repeatedly shape judgment, language, and action. In open-weight settings, runtime controls can be removed, motivating persona unlearning: a weight-level edit that makes a designated persona difficult to elicit and enact on unseen contexts. We introduce PersonaUnlearnBench, a model-specific paired benchmark spanning six LLMs from three families and five personas, with aligned forget/retain sets, held-out instruction paraphrases, and four-axis evaluation. The benchmark shows that standard unlearning methods cannot reliably erase the target persona without sacrificing meaningful generation or general uti
    
[^40]: GrammarRL：通过强化学习实现有效的语法约束解码

    GrammarRL: Effective Grammar-Constrained Decoding via Reinforcement Learning

    [https://arxiv.org/abs/2609.39869](https://arxiv.org/abs/2609.39869)

    GrammarRL 是一种无需标注数据的强化学习方法，通过基于模型自身似然的直接奖励与反向奖励，使语言模型在满足语法约束的同时保持语义质量。

    

    语法约束生成保证了句法的有效性，但当模型偏好的输出与所施加的语法对齐不佳时，可能会显著降低语义质量。当提示信息不充分或模型指令遵循能力有限时，这种权衡尤为严重。束搜索可以通过探索多个有效序列来部分缓解这些问题，但其计算成本随束宽的增长而增加，且序列级概率只是语义质量的一个不完美的代理指标。我们提出了 GrammarRL，这是一种无需标注数据的强化学习方法，能够在不需要标注数据的情况下使语言模型适应语法约束。GrammarRL 利用从模型自身似然中导出的两种互补的自监督奖励来优化模型：直接奖励，衡量在给定输入的条件下约束输出出现的可能性；反向奖励，衡量输入能够被重构的程度……

    arXiv:2609.39869v1 Announce Type: new  Abstract: Grammar-constrained generation guarantees syntactic validity, but can substantially degrade semantic quality when the model's preferred outputs are poorly aligned with the imposed grammar. This trade-off is particularly severe when the prompt is underspecified or the model has limited instruction-following ability. Beam search can partially mitigate these failures by exploring multiple valid sequences, but its computational cost grows with beam width, while sequence-level probability is only an imperfect proxy for semantic quality.   We introduce GrammarRL, a label-free reinforcement learning method that adapts language models to grammar constraints without requiring annotated data. GrammarRL optimizes the model using two complementary self-supervised rewards derived from its own likelihoods: a direct reward, measuring how likely the constrained output is given the input, and a reverse reward, measuring how well the input can be reconstr
    
[^41]: FIGS：在不惩罚共情的前提下评估多轮谄媚行为

    FIGS: Evaluating Multi-Turn Sycophancy Without Penalizing Empathy

    [https://arxiv.org/abs/2609.39863](https://arxiv.org/abs/2609.39863)

    提出了FIGS双轴评估框架，通过自适应的10轮真实多轮对话来评估大语言模型的谄媚行为，在不将共情误判为屈从的前提下区分事实完整性与支持性表达。

    

    大语言模型经常难以在保持真实性与提供支持之间取得平衡。它们在回应用户时常常表现出谄媚行为，例如附和错误的主张、给予无端的奉承，以及提供偏向用户已表达观点的建议。在现实中，谄媚很少发生在单次交流中；它可能随着用户反复坚持或在一段时间内微妙地引导对话而自然产生。然而，当前的评估依赖于僵化的单轮测试或固定脚本，无法捕捉这些自然的动态变化。此外，这些基准往往将基本的共情表达误认为是屈从，因为模型承认了用户的感受而对其进行惩罚。这种观点可能会促使未来的模型过度矫正，变得冷漠、轻慢且僵硬。为填补这一空白，我们提出了FIGS（事实完整性与有据支持，Factual Integrity and Grounded Support），一个围绕延展的、贴近现实的对话构建的双轴评估框架。我们使用自适应的10轮对话……（摘要在此处截断）

    arXiv:2609.39863v1 Announce Type: new  Abstract: Large language models frequently fail to balance staying truthful with being supportive. They often exhibit sycophancy in responses to users, agreeing with false claims, offering unwarranted flattery, and giving advice skewed toward users' expressed views. In reality, sycophancy rarely happens in a single exchange; it may emerge organically as users repeatedly insist or subtly steer the dialogue over time. Current evaluations, however, rely on rigid, single-turn tests or fixed scripts that fail to capture these natural dynamics. Furthermore, these benchmarks often mistake showing basic empathy for yielding, penalizing models for acknowledging a user's feeling. This view may drive future models to over-correct into cold, dismissive rigidity. To address this gap, we introduce FIGS (Factual Integrity and Grounded Support), a dual-axis evaluation framework built around extended, realistic dialogue. We use an adaptive 10-turn conversational s
    
[^42]: 认知增强：重新思考大型语言模型中角色扮演的必要性

    Cognitive Enhancement: Rethinking the Necessity of Role-Playing for Large Language Models

    [https://arxiv.org/abs/2609.39853](https://arxiv.org/abs/2609.39853)

    该研究通过多模型、跨领域、多语言实验发现角色扮演提示的收益取决于模型容量、知识领域和提示语言，并基于元认知理论提出“角色相关认知对齐假设”——只有当大语言模型正确理解指定角色及其知识领域时，角色扮演才能有效提升性能。

    

    角色扮演提示已成为一种流行且简单的技术，用于提升大语言模型的推理能力和输出质量。然而，由于缺乏系统性验证，它能否在多个不同领域中持续带来性能提升仍不明确。为填补这一空白，我们在MMLU和MMLU-Redux数据集上开展了多模型、跨领域、多语言的实验。研究发现，角色扮演提示带来的收益在很大程度上取决于模型容量、知识领域和提示语言。基于元认知理论，我们提出了与角色相关的认知对齐假设：只有当大语言模型正确理解所指定的角色及其相关知识领域时，角色扮演才会奏效。我们通过角色信息丰富度消融实验、逐层熵散度分析以及潜在思维空间偏转观察来验证这一假设。为减少角色认知偏差并稳定角色扮演的性能，我们提出了混合语言（Mixed-Language）……（原文摘要在此处截断）

    arXiv:2609.39853v1 Announce Type: new  Abstract: Role-playing prompting has become a popular yet simple technique for improving LLM reasoning and output quality. However, whether it consistently boosts performance across diverse domains remains unclear, as systematic validation is lacking. To fill this gap, we run multi-model, cross-domain, and multilingual experiments on MMLU and MMLU-Redux. We find that gains from role-play prompting depend heavily on model capacity, knowledge domain, and prompt language. Drawing on metacognition theory, we propose the persona-related cognitive alignment hypothesis: role-play works only when the LLM correctly grasps the designated persona and its associated knowledge domain. We test this hypothesis through persona information richness ablation, layer-wise entropy divergence analysis, and latent thought-space deflection observation. To reduce persona cognitive bias and stabilize role-play performance, we propose \textbf{M}ixed-\textbf{L}anguage \textb
    
[^43]: 当幼儿园学生解微积分时：测量角色提示推理模型中的能力泄漏

    When a Kindergartener Solves Calculus: Measuring Capability Leakage in Role-Prompted Reasoning Models

    [https://arxiv.org/abs/2609.39846](https://arxiv.org/abs/2609.39846)

    本文提出RoleCapBench基准，揭示被角色提示的推理模型虽然能在语言风格上 convincing 地扮演分配的角色（如幼儿园学生），但其实际能力仍会“泄漏”至远超角色水平的程度（如专家级解微积分），始终无法将底层能力与角色保持一致。

    

    我们研究角色-能力泄漏（Role-Capability Leakage, RCL）问题，即被赋予角色的推理模型在生成具有说服力的角色内文本的同时，在基准测试中仍继续表现出超出所分配角色所暗示的能力。例如，当一个模型被提示扮演幼儿园学生的角色时，人们可能期望它在数学基准测试中的表现反映幼儿园水平的能力，而不是展现解决微积分问题的专家级熟练程度。我们提出了RoleCapBench，一个基于课程的基准测试，用于评估六个教育角色和四个评估级别（涵盖从小学到A-Level阶段）下的RCL，并用它对三个开放权重的推理模型进行了评估。我们发现，尽管这些模型能够生成风格上令人信服的角色内回应，但它们始终未能使其底层能力与所分配的角色保持一致。朴素的角色提示会产生强烈的角色语气，但（原文在此处截断）

    arXiv:2609.39846v1 Announce Type: cross  Abstract: We investigate the problem of role-capability leakage (RCL), in which a role-prompted reasoning model generates convincing in-role text while continuing to exhibit capabilities on benchmarks that exceed those implied by the assigned role. For example, when a model is prompted to assume the role of a kindergarten student, one might expect its performance on a mathematics benchmark to reflect kindergarten-level ability rather than expert-level proficiency in solving calculus problems. We introduce RoleCapBench, a curriculum-grounded benchmark for evaluating RCL across six educational roles and four assessment levels spanning elementary school through A-level, and use it to evaluate three open-weight reasoning models. We find that although the models can generate stylistically convincing in-role responses, they consistently fail to align their underlying capabilities with their assigned roles. Naive role prompting yields strong role-voice
    
[^44]: 学习隐写术容易，学习隐写式推理很难

    Learning Steganography Is Easy, Learning Steganographic Reasoning Is Hard

    [https://arxiv.org/abs/2609.39838](https://arxiv.org/abs/2609.39838)

    该论文通过对比强化学习、上下文学习和监督微调三种引导方法发现，模型较容易学会隐写传信和编码推理这两种相邻能力，而学会完全隐藏推理过程的隐写式推理则困难得多（通常仅在监督微调下才能习得），这对依赖思维链监控的AI安全方案具有重要意义。

    

    arXiv:2609.39838v1 公告类型：新论文 摘要：思维链监控作为AI监督与控制的一种方法，正受到隐写式推理可能性的威胁——即大语言模型将其推理过程隐藏在看似无害的文本之中。两种相邻的能力——隐写传信（传递一条隐藏的消息）和编码推理（以难以辨识但并未隐藏的格式进行推理）——已被证明会在真实训练流程中出现的训练压力下自发涌现，例如针对监控器的强化学习。这表明隐写式推理也可能作为训练的意外副作用而出现。本文比较了模型在三种引导方法（强化学习、上下文学习和监督微调SFT）下学习隐写式推理与这两种相邻能力的难易程度。对于大多数任务，模型只有在监督微调下才能学会隐写式推理，而它们学习隐写传信和编码推理则相对容易。

    arXiv:2609.39838v1 Announce Type: new  Abstract: Chain-of-thought monitoring as an approach for AI oversight and control is threatened by the possibility of steganographic reasoning, where LLMs conceal their reasoning inside innocuous-looking text. Two neighbouring capabilities, steganographic messaging (passing a concealed message) and encoded reasoning (reasoning in an illegible but unconcealed format), have already been shown to emerge under training pressures that occur in real pipelines, such as reinforcement learning against monitors. This suggests that steganographic reasoning too might arise as an unintended side effect of training. Here, we compare how easily models learn steganographic reasoning and these two neighbouring capabilities across three elicitation methods: reinforcement learning, in-context learning, and supervised fine-tuning (SFT).   For most tasks, models learn steganographic reasoning only under SFT, while they learn steganographic messaging and encoded reason
    
[^45]: 合成数据预预训练在规模化下依然有效，但并非作为语法先验

    Synthetic Pre-pretraining Survives Scale, but Not as a Grammatical Prior

    [https://arxiv.org/abs/2609.39827](https://arxiv.org/abs/2609.39827)

    该研究首次在 500M 至 7B 参数规模、多种数据混合及高达 100B token 预算下系统验证了合成数据预预训练（PPT）的有效性，发现其节省 token 的收益在规模化下依然显著（如 3B 规模下节省至少 21B token），但推翻了先前将其增益归因于语法先验的解释。

    

    在合成非自然语言数据上进行预预训练（PPT）能够提升语言模型预训练（PT）期间的 token 效率。先前的工作将这一增益归因于语法先验，即在 PPT 阶段学到的、可迁移至自然语言语法的结构归纳偏置。然而，PPT 此前仅在参数量不超过 1B、PT 预算低于 2B token、且数据以网页文本为主的模型上进行过测试。目前尚不清楚 PPT 在更大规模以及结合多种来源（如代码和数学）的更贴近现实的 PT 数据混合下是否仍然有效。因此，我们对 PPT 开展了一项全面研究，涵盖五种 PPT 任务、四种 PT 数据混合、四个参数规模（500M 至 7B）以及高达 100B token 的 PT 预算。我们的结果表明，PPT 带来的下游性能和 token 效率增益在规模化下依然存在，例如在 3B 规模下可节省至少 21B 个 PT token。然而，与先前工作相反，我们没有发现一致的证据（原文摘要在此处截断）。

    arXiv:2609.39827v1 Announce Type: cross  Abstract: Pre-pretraining (PPT) on synthetic non-natural language data improves token efficiency during language model pre-training (PT). Prior work attributes this gain to a grammatical prior, i.e., a structural inductive bias learned during PPT that transfers to natural language grammar. However, PPT has only been tested on models of at most 1B parameters and PT budgets below 2B tokens on predominantly web text. It is unknown whether PPT is effective at larger scales and under more realistic PT data mixtures that combine diverse sources (e.g., code and math). We therefore present a comprehensive study on PPT spanning five PPT tasks, four PT data mixtures, four parameter scales (500M to 7B), and PT budgets of up to 100B tokens. Our results demonstrate that the downstream performance and token efficiency gains of PPT persist at scale, e.g., saving at least 21B PT tokens at the 3B scale. However, in contrast to prior work, we find no consistent e
    
[^46]: 压力测试大语言模型谎言探测器：角色扮演失效与虚假相关性

    Stress-Testing LLM Lie Detectors: Role-Play Failures and Spurious Correlations

    [https://arxiv.org/abs/2609.39807](https://arxiv.org/abs/2609.39807)

    本研究通过构建包含8,916条人工审核响应的数据集对现有LLM谎言检测探针进行压力测试，发现许多探针在反事实人格角色扮演场景下失效，容易被虚假相关性所误导。

    

    谎言检测探针旨在从语言模型的内部状态预测其输出是真实的还是虚假的。然而，角色扮演使“真实”对于LLM的含义变得复杂：语言模型可以采用各种各样的人格设定，而这些人格对于什么主张是真的持截然不同的看法，其中包括其信念明显与现实相矛盾的人格，例如阴谋论者。在这项工作中，我们研究谎言检测探针是否能可靠地标记出在这种反事实人格下生成的虚假内容，还是它们反而会顺从该人格的信念。我们引入了一个包含8,916条经人工审核、来自三个采用反事实人格的大语言模型的同策略响应的数据集。通过对先前工作中的八个探针进行评估，我们发现许多探针在这种设置下会失效，特别是在同一人格提示下评估正确与错误答案时。为了探究其中的原因，我们构建了三个新颖的混淆因素数据集，其中真实性与某种特征呈反相关（摘要在此处被截断）。

    arXiv:2609.39807v1 Announce Type: cross  Abstract: Lie detection probes aim to predict from a language model's internal states whether its output is truthful or dishonest. However, role-play complicates what "truth" means for an LLM: language models can adopt a wide range of personas that take very different claims to be true, including personas whose beliefs clearly contradict reality, such as a conspiracy theorist. In this work, we investigate whether lie detection probes reliably flag falsehoods generated under such an anti-factual persona or whether they instead follow the persona's beliefs. We introduce a dataset of 8,916 human-reviewed, on-policy responses from three LLMs adopting anti-factual personas. Evaluating eight probes from prior work, we find that many fail in this setting, particularly when correct and incorrect answers are evaluated under the same persona prompt. To investigate why, we construct three novel confounder datasets in which truth is anti-correlated with a p
    
[^47]: 图上探索：面向不完整知识图谱问答的混合嵌入-大语言模型推理框架

    Explore-on-Graph: Hybrid Embedding-LLM Reasoning for Knowledge Graph Question Answering under Incompleteness

    [https://arxiv.org/abs/2609.39786](https://arxiv.org/abs/2609.39786)

    提出XoG框架，通过结合知识图谱嵌入与类型级实体-关系统计从图结构本身恢复不完整知识图谱中缺失的推理路径，并让LLM仅充当语义选择器和推理器，从而在避免幻觉的同时实现可靠的多跳问答。

    

    大语言模型（LLM）越来越多地与知识图谱（KG）结合，以使推理建立在结构化证据之上。然而，大多数基于LLM的知识图谱问答（KGQA）方法依赖于遍历已有的图边，当推理路径因事实缺失而中断时，这些方法变得不可靠；而让LLM自行生成缺失知识的替代方案则存在引入幻觉证据的风险。我们提出了XoG（eXplore-on-Graph，图上探索），一个面向不完整知识图谱的多跳问答框架，它从学习到的图结构中恢复缺失的推理路径，而非依赖LLM的参数化知识。XoG将用于识别候选关系的类型级实体-关系统计与用于检索合理缺失实体的知识图谱嵌入相结合，并将LLM用作语义选择器和推理器。这些机制被整合到一个迭代的“规划-探索-推理”流程中。在WebQSP、CWQ以及基于Wikidata的BRINK基准上的实验表明，XoG在……（原文摘要至此截断）

    arXiv:2609.39786v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly combined with knowledge graphs (KGs) to ground reasoning in structured evidence. However, most LLM-based KGQA methods rely on traversing existing graph edges and become unreliable when reasoning paths are broken by missing facts. Alternatives that ask LLMs to generate missing knowledge risk introducing hallucinated evidence. We introduce XoG (eXplore-on-Graph), a framework for multi-hop question answering over incomplete KGs that recovers missing reasoning paths from learned graph structure rather than LLM parametric knowledge. XoG combines type-level entity-relation statistics to identify candidate relations with KG embeddings to retrieve plausible missing entities, using the LLM as a semantic selector and reasoner. These mechanisms are integrated into an iterative planning-exploration-reasoning process. Experiments on WebQSP, CWQ, and the Wikidata-based BRINK benchmark show that XoG remains
    
[^48]: MemCodex：面向语言智能体的自编程分层记忆

    MemCodex: Self-Programming Hierarchical Memory for Language Agents

    [https://arxiv.org/abs/2609.39765](https://arxiv.org/abs/2609.39765)

    提出MemCodex，一种通过开放式程序演化实现自编程的分层记忆系统，能够将经验组织为可执行记忆程序，并根据查询需求自适应地调整记忆的构建、检索与跨层组合方式。

    

    智能体记忆面临异构的访问需求：单跳问题可能只需要一条证据，而多跳问题必须组合来自多个来源的证据。预定义的记忆工作流无法适应这些不同的需求。近期的自适应方法在记忆组件及其组合上进行搜索或学习，但设计空间本身仍然是预定义的。我们提出了MemCodex，一个自我演化的分层记忆系统，它将经验组织成可执行的记忆程序，涵盖摘要、关系知识、可复用技能和潜在记忆。开放式程序演化通过重写每一层的构建、索引、检索和路由方式，搜索层程序的开放设计空间，从而同时适应层内实现和跨层组合。在查询时，读取操作从粗到细遍历层次结构，一旦找到足够的证据即停止，并可下钻至原始历史记录。

    arXiv:2609.39765v1 Announce Type: new  Abstract: Agent memory faces heterogeneous access needs: a single-hop question may require one piece of evidence, whereas a multi-hop question must combine evidence from multiple sources. Predefined memory workflows cannot adapt to these varying needs. Recent adaptive methods search or learn over memory components and their compositions, but the design space itself remains predefined. We introduce MemCodex, a self-evolving hierarchical memory system that organizes experience into executable memory programs for summaries, relational knowledge, reusable skills, and latent memory. Open-ended program evolution searches the open design space of layer programs by rewriting how each layer is constructed, indexed, retrieved, and routed, thereby adapting both within-layer implementations and cross-layer composition. At query time, reads traverse the hierarchy from coarse to fine and stop once sufficient evidence is found, descending to the original history
    
[^49]: LatentHarness：通过反事实策略蒸馏学习面向记忆与推理的潜在动作

    LatentHarness: Learning Latent Actions for Memory and Reasoning via Counterfactual Policy Distillation

    [https://arxiv.org/abs/2609.39740](https://arxiv.org/abs/2609.39740)

    LatentHarness 将记忆访问与潜在推理统一为 THINK、RECALL、EXIT 三种潜在动作的顺序选择，并通过反事实策略蒸馏训练该策略，教会模型何时调用记忆比继续推理更有用以及应保留哪些中间状态。

    

    长上下文推理面临两个互补的瓶颈：在长输入中保留证据，以及在大量推理步骤中持续进行计算。现有方法大多分别处理这两个问题，即利用外部记忆扩展对远距离证据的访问，利用潜在推理压缩多步计算。我们提出 LatentHarness，将记忆访问与潜在推理统一为顺序的潜在动作选择。在每个内部步骤中，模型可以选择 THINK 进行进一步计算、从保存输入证据和中间推理状态的快速权重记忆中选择 RECALL，或选择 EXIT 以输出下一个词元。我们采用反事实策略蒸馏来训练该策略：对每个动作进行一步分支，并评估其对输出词元的影响。这些收益教会在何种情况下使用记忆比继续推理更有用，而通过反事实召回的梯度则教会哪些中间状态应当被保留。

    arXiv:2609.39740v1 Announce Type: new  Abstract: Long-context reasoning faces two complementary bottlenecks: retaining evidence across long inputs and sustaining computation across many reasoning steps. Existing approaches largely address them separately, with external memory extending access to distant evidence and latent reasoning compressing multi-step computation. We introduce LatentHarness, which unifies memory access and latent reasoning as sequential latent action selection. At each internal step, the model chooses THINK for further computation, RECALL from a fast-weight memory of input evidence and intermediate reasoning states, or EXIT to emit the next token. We train this policy with counterfactual policy distillation, which branches every action for one step and scores its effect on the emitted token. These gains teach the policy when memory is more useful than further reasoning, while gradients through counterfactual recall teach which intermediate states should be retained
    
[^50]: OverForge：通过策略与战术的推理助力协作式终身适应

    OverForge: Reasoning Through Strategies and Tactics Helps Cooperative Lifelong Adaptation

    [https://arxiv.org/abs/2609.39727](https://arxiv.org/abs/2609.39727)

    OverForge提出了一种免训练的分层架构，通过将持久的协调策略推理与战术行动执行分离，并利用元认知“前额叶皮层”模块在策略-行动分支上进行想象与承诺，使协作型语言模型智能体在OvercookedV2中表现远超扁平基线，并能与陌生伙伴达成角色协调。

    

    协作型语言模型智能体必须在长时程中进行协调，并适应不断变化的环境以及具有陌生惯例的合作伙伴，然而现有的智能体只是将观察直接映射为行动，没有将持久的协调策略与战术执行区分开来。我们提出了OverForge，这是一种免训练的分层架构，它将关于角色与分工的战略推理，与每个智能体私有的、以合作伙伴为条件的世界模型中的战术行动推理分离开来。一个元认知的“前额叶皮层”模块通过构建“策略-行动”分支、利用前向模型想象其后果、并在有信心时做出承诺，将这两个层级耦合起来。在OvercookedV2中，OverForge在连通厨房中交付了7份汤，而每个扁平化LLM基线仅交付3份；它还能保持已商定的角色，并采纳陌生伙伴提出的角色。消融实验和固定策略探针表明，持久策略引导（摘要在此处不完整）

    arXiv:2609.39727v1 Announce Type: new  Abstract: Cooperative language-model agents must coordinate over long horizons and adapt to changing environments and to partners with unfamiliar conventions, yet existing agents map observations to actions without separating persistent coordination strategies from their tactical execution. We introduce OverForge, a training-free hierarchical architecture that separates strategic reasoning over roles and divisions of labour from tactical reasoning over actions within each agent's private, partner-conditioned world model. A metacognitive Prefrontal Cortex Module couples the two levels by forming strategy-action branches, imagining their consequences with a forward model, and committing when confident. In OvercookedV2, OverForge delivers 7 soups in a connected kitchen versus 3 for each flat LLM baseline, retains agreed roles, and adopts roles proposed by unfamiliar partners. Ablations and a fixed-strategy probe show that persistent strategies guide 
    
[^51]: 漂移检查器：基于原子贡献声明探索与测量科学领域的演变

    Drift Inspector: Exploring and Measuring Scientific Drift with Atomic Contribution Claims

    [https://arxiv.org/abs/2609.39710](https://arxiv.org/abs/2609.39710)

    该论文提出了开源系统 Drift Inspector，通过让大语言模型从摘要中提取原子贡献声明并跨年份聚类成可交互追溯的可视化地图，从而精确测量研究领域的演变，揭示了 NLP 领域从经典任务向 LLM 时代能力（如推理和多模态）的转变。

    

    科学摘要将贡献与背景、动机和元语言混杂在一起，因此直接读取摘要原文的工具无法将一个领域的实际产出与其所讨论的内容区分开来。我们提出了 Drift Inspector，一个开源系统，用于在原子贡献声明的层面上测量和探索研究领域如何随时间演变——原子贡献声明是由大语言模型在分析之前从每篇摘要中提取的去语境化、承载贡献信息的命题。该系统将这些声明按年份进行聚类，形成一张交互式地图，每一条趋势都可以追溯回其背后的声明和论文。将该系统应用于六年的 EMNLP 数据，结果表明该领域正从经典 NLP 任务转向 LLM 时代的能力（如推理和多模态）——这种转变在关键词统计或整篇摘要统计中是模糊不清的。所发布的数据不仅限于 EMNLP：同一流程已经处理了完整的 ACL Anthology（34.6万条声明、8万篇摘要、423个发表场所）。

    arXiv:2609.39710v1 Announce Type: new  Abstract: Scientific abstracts mix contributions with background, motivation, and meta-language, so tools that read them as-is cannot separate what a field produces from what it discusses. We present Drift Inspector, an open-source system for measuring and exploring how a research field changes over time at the level of Atomic Contribution Claims (ACCs): decontextualized, contribution-bearing propositions an LLM extracts from each abstract before analysis. The system clusters these claims across years into an interactive map where every trend traces back to the claims and papers behind it. Applied to six years of EMNLP, it shows the field shifting away from classic NLP tasks toward LLM-era capabilities such as reasoning and multimodality -- a movement that keyword or whole-abstract counts blur. The released data extend beyond EMNLP: the same pipeline has processed the full ACL Anthology (346k claims, 80k abstracts, 423 venues). Extraction is human
    
[^52]: A帮助B而B损害A：指令微调混合中的有向迁移

    A helps B while B hurts A: directed transfer in instruction-tuning mixture

    [https://arxiv.org/abs/2609.39702](https://arxiv.org/abs/2609.39702)

    该论文发现指令微调中的任务迁移是有方向性的（A可以帮助B而B却损害A），并提出“迁移图”这一带符号估计方法来预测每个源任务对目标任务的增益或损害，从而在固定预算下更高效地指导训练任务的选择。

    

    将语言模型适配到专门语料库，意味着需要在固定预算下选择用于训练的指令微调任务，而测试一种选择就需要付出一次微调运行的代价。常见的启发式方法是添加更多源任务，或选择与目标相似的源任务。前者假设迁移永远不会是负面的；后者假设迁移是对称的。我们证明这两种假设都不成立：任务A可以帮助任务B，而B却会损害A，因此“有用性”是有序源-目标对的一个带符号属性。我们引入了“迁移图”，这是一个带符号的估计量，用于衡量每个源任务对每个保留出的目标任务有多大帮助或损害。我们在参数量从0.6B到32B的Qwen3和Mistral模型上通过数百次微调运行来拟合该图，所有源任务均来自同一语料库，且不使用任何来自目标任务的训练样本。该图能够预测保留出的目标任务在未见过的任务混合上的准确率：在这些运行之前记录的预测，其误差不到与混合无关的基线方法误差的一半。

    arXiv:2609.39702v1 Announce Type: new  Abstract: Adapting a language model to a specialized corpus means choosing which instruction-tuning tasks to train on under a fixed budget, and testing one choice costs a fine-tuning run. Common heuristics add more source tasks or pick sources similar to the target. The first assumes transfer is never negative; the second, that it is symmetric. We show that both assumptions fail: task $A$ can help task $B$ while $B$ hurts $A$, so helpfulness is a signed property of ordered source--target pairs. We introduce the transfer map, a signed estimate of how much each source helps or hurts each held-out target. We fit the map in hundreds of fine-tuning runs on Qwen3 and Mistral models from 0.6B to 32B parameters, with all sources drawn from one corpus and no training examples from the target. The map predicts a held-out target's accuracy on unseen mixtures: recorded before those runs, its predictions have less than half the error of a mixture-agnostic base
    
[^53]: ShieldCLIP：面向多模态基础模型有害内容缓解的选择性安全对齐

    ShieldCLIP: Selective Safety Alignment for Harmful Content Mitigation in Multimodal Foundation Models

    [https://arxiv.org/abs/2609.39688](https://arxiv.org/abs/2609.39688)

    ShieldCLIP首次根据每个模态的实际安全状态而非样本来源进行选择性安全对齐，并配套提出带独立按模态安全标签的19.5万四元组数据集ViSUv2，在缓解多模态基础模型有害内容的同时保留良性表示。

    

    多模态编码器（如CLIP）是众多下游系统的基础，但其网络规模训练数据中嵌入的有害关联必须通过安全对齐加以抑制，同时又不能不必要地改变良性表示。由于伦理和实际约束使大规模收集真实不安全内容不可行，现有数据集将安全的真实样本与生成的对应样本配对，并将所有生成样本一律标记为不安全，即使其中某一模态单独来看是安全的。为解决这一问题，我们提出了ShieldCLIP，这是首个基于每个模态的实际观测安全状态（而非样本来源）来进行安全对齐的框架，在保留安全内容的同时仅对不安全内容进行重定向。我们还引入了ViSUv2，一个包含19.5万个四元组的数据集，覆盖578个概念和28个类别，并带有独立的按模态安全标签。利用这些标签，ShieldCLIP定义了超越配对级监督的四种条件目标：安全内容……（摘要原文在此处截断）

    arXiv:2609.39688v1 Announce Type: cross  Abstract: Multimodal encoders such as CLIP underlie many downstream systems, but their web-scale training data embed harmful associations that safety alignment must suppress without unnecessarily changing benign representations. Because ethical and practical constraints prevent collecting real unsafe content at scale, existing datasets pair safe real samples with generated counterparts, but label every generated sample unsafe, even when one modality is individually safe. To address this, we introduce ShieldCLIP, the first framework to condition safety alignment on the observed safety state of each modality rather than the origin of a sample, preserving safe content while redirecting only what is unsafe. We also introduce ViSUv2, a 195k-quadruplet dataset with independent per-modality safety labels across 578 concepts and 28 categories. Using these labels, ShieldCLIP defines a four-way conditional objective beyond pair-level supervision: safe con
    
[^54]: 更好的监督就在附近：邻域在策略自蒸馏

    Better Supervision Is Nearby: Neighborhood On-Policy Self-Distillation

    [https://arxiv.org/abs/2609.39687](https://arxiv.org/abs/2609.39687)

    该论文提出邻域在策略自蒸馏（N-OPSD），通过离线贪婪选择构建一个紧凑的冻结专家池，并在在线阶段根据每个参考位置动态路由选择最合适的专家，将局部参数扰动带来的互补参考对齐纠正转化为学生访问状态处的更强监督信号，从而提升数学推理模型的自蒸馏训练效果。

    

    在策略自蒸馏（OPSD）利用一个能看到参考解答的特权教师来监督学生采样的前缀，从而训练数学推理模型。标准OPSD在每个状态下只使用单一固定的参数设置，但邻近的参数设置可能提供额外的监督信号。我们发现，在相同参考上下文下，局部参数扰动能够揭示互补的、与参考解答对齐的纠正信息。不同的专家在不同的参考位置提供这些纠正，而它们组成的专家池比未扰动的特权教师能覆盖更多这样的位置。我们提出邻域在策略自蒸馏（N-OPSD），将这些纠正转化为学生所访问状态处的监督信号。在离线阶段，通过贪婪选择构建一个紧凑的冻结专家池，其筛选标准是奖励那些在每个位置上超越池中当前最佳水平的、经过滤的参考token增益。峰值最高的专家未必能提供最佳的训练目标。因此，在线路由将锚定（原文摘要在此截断）

    arXiv:2609.39687v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) trains mathematical reasoning models using a privileged teacher that sees a reference solution and supervises student-sampled prefixes. Standard OPSD uses one fixed parameter setting at every state, but nearby settings may offer additional supervision. We find that local parameter perturbations reveal complementary reference-aligned corrections under the same reference context. Different experts supply these corrections at different reference positions. Their pool covers more such positions than the unperturbed privileged teacher. We introduce Neighborhood OPSD (N-OPSD) to turn these corrections into supervision at student-visited states. Offline, greedy selection builds a compact pool of frozen experts by rewarding filtered reference-token gains beyond the pool's current best at each position. The highest-peak expert need not provide the best training target. Online routing therefore separates the anch
    
[^55]: 大语言模型中注意力机制的演进：机制、权衡与新兴趋势

    The Evolution of Attention in Large Language Models: Mechanisms, Trade-offs, and Emerging Trends

    [https://arxiv.org/abs/2609.39661](https://arxiv.org/abs/2609.39661)

    该综述将大语言模型中各类注意力改进方法统一视为“模型内部上下文记忆”，提出由记忆表示、记忆更新、访问、读取与整合构成的五维分析框架，并基于14个模型谱系的59条发布记录系统梳理了注意力机制的演进、权衡与新兴趋势。

    

    自注意力机制使大语言模型能够对上下文进行细粒度、依赖查询的访问，但稠密的词元交互会带来二次方复杂度的预填充成本，以及随上下文长度不断增长的关键-值缓存。因此，相关研究涵盖了显式记忆压缩、稀疏访问、循环状态构建、结构化状态动力学以及异构机制组合等多个方向。本综述将这些进展作为模型内部的上下文记忆加以分析。我们引入了一个五维分析视角——记忆表示、记忆更新、访问、读取与整合——用以刻画表示了什么、如何变化、哪些内容可被查询、如何读取，以及读取结果如何形成输出。该视角能够在不强加单一计算模型的前提下，比较相互重叠的各条研究路线。我们利用来自14个主要模型谱系的59条发布级记录以及11个高性能开源权重端点，重构了机制层面的发展脉络与架构层面的采用情况。首先，显式记忆与……

    arXiv:2609.39661v1 Announce Type: new  Abstract: Self-attention gives LLMs fine-grained, query-dependent access to context, but dense token interactions incur quadratic prefill cost and a key--value cache growing with context length. Research thus spans explicit-memory compression, sparse access, recurrent state construction, structured state dynamics, and heterogeneous mechanism composition. This survey analyzes these developments as model-internal contextual memory. We introduce a five-dimensional lens---Memory Representation, Memory Update, Access, Readout, and Integration---describing what is represented, how it changes, what is query-eligible, how it is read, and how readouts form outputs. This lens compares overlapping research lines without imposing one computational model.   We reconstruct mechanism-level developments and architectural adoption using 59 release-level records from 14 major model lineages and 11 high-performing open-weight endpoints. First, explicit-memory and re
    
[^56]: SEPAL：基于分离专家对与答案级融合的可靠大语言模型协作方法

    SEPAL: Separated Expert Pairs with Answer-Level Fusion for Reliable LLM Collaboration

    [https://arxiv.org/abs/2609.39645](https://arxiv.org/abs/2609.39645)

    SEPAL 通过三个独立训练的 Actor-Critic 专家团队分别负责推理、证据验证和校验，并仅在最终答案层面进行多数投票融合，避免共享讨论导致的错误传播，从而显著提升大语言模型协作问答的可靠性和准确率。

    

    多智能体协作使大语言模型（LLM）能够通过审议和反馈来提升问答能力。然而，共享讨论会将纠错过程与暴露于相同错误耦合在一起，这可能会削弱投票机制所需的多样性。自洽性方法提供了无需反馈的采样多样性，而单对 Actor-Critic 协作仅能精炼一个候选答案。我们提出了 SEPAL，它为直接推理、证据接地和验证三个环节分别分配三个私有的 Actor-Critic 团队。角色特定的训练使这些团队拥有超越采样差异的各自推理目标。每个 Critic 仅在其自己的团队内部指导修订，防止反馈将错误传递给其他候选。修订结束后，多数投票仅结合最终答案，在做出决策之前保持推理历史的相互分离。在五个开源权重骨干模型和五个问答基准上，SEPAL 提升了平均准确率

    arXiv:2609.39645v1 Announce Type: cross  Abstract: Multi-agent collaboration lets large language models (LLMs) improve question answering through deliberation and feedback. Yet shared discussion couples correction with exposure to the same mistakes, which can erode the diversity needed for voting. Self-consistency offers sampling diversity without feedback, while single-pair Actor-Critic collaboration refines only one candidate. We introduce SEPAL, which assigns three private Actor-Critic teams to direct reasoning, evidence grounding, and verification. Role-specific training gives the teams different reasoning objectives beyond sampling variation. Each Critic guides revisions within its own team, preventing feedback from carrying errors across candidates. Once revision ends, majority voting combines only the final answers, keeping the reasoning histories separate until the decision. Across five open-weight backbones and five question-answering benchmarks, SEPAL improves mean accuracy b
    
[^57]: 零计算跨语言可迁移性估计：基于类型学特征代理

    Zero-Compute Cross-Lingual Transferability Estimation Using Typological Feature Proxies

    [https://arxiv.org/abs/2609.39640](https://arxiv.org/abs/2609.39640)

    仅利用免费的语言类型学特征训练随机森林，即可在零计算成本下准确预测跨语言迁移性（留一语言 ρ=0.705、R²=0.49），证明类型学数据库蕴含廉价而密集的迁移信号，可替代昂贵的多语言预训练测量。

    

    跨语言迁移描述了源语言中的知识如何惠及目标语言。对迁移进行定量测量需要广泛的多语言预训练，正如先前工作通过跨语言迁移矩阵所做的那样。我们探究迁移是否可以从免费获取的类型学特征中预测出来，以及高资源源语言的突出优势究竟是源于类型学因素，还是源于数据的质量与数量。我们证明，类型学数据库中蕴含着关于跨语言迁移的廉价且密集的信号。在先前工作给出的24种语言迁移矩阵上，仅使用类型学特征的随机森林在留一语言协议下取得了 ρ=0.705 和 R²=0.49 的成绩，优于非类型学对照组的 ρ=0.62，这验证了仅凭类型学预测即可重建昂贵的实测跨语言迁移能力。该信号在留一文字系统和留一语系协议下依然存在，因此文字系统和语系的混杂因素无法解释这一效应（原文在此处截断）。

    arXiv:2609.39640v1 Announce Type: cross  Abstract: Cross-lingual transfer describes how knowledge in a source language benefits a target language. Measuring it quantitatively requires broad multilingual pre-training, as prior work has done with cross-lingual transfer matrices. We ask whether transfer is predictable from freely available typological features, and whether the prominence of high-resource source languages reflects typology or data quality and quantity. We show that typological databases contain cheap and dense signals about cross-lingual transfer. Our typology-only random forest on a 24-language prior-work transfer matrix scores leave-one-language-out $\rho{=}0.705$ and $R^2{=}0.49$, beating a non-typological control at $\rho{=}0.62$, which verifies the ability of typology-only predictions to reconstruct costly measured cross-lingual transfer. The signal survives leave-one-script-out and leave-one-family-out protocols, so script and family confounding do not explain the ef
    
[^58]: 面向零标签表格学习的边际响应面引导方法

    Marginal Response Surface Elicitation for Zero-Label Tabular Learning

    [https://arxiv.org/abs/2609.39639](https://arxiv.org/abs/2609.39639)

    提出MARS方法，将LLM的特征级先验离线转化为可复用的零样本表格分类器，在八个表格基准任务上无需进一步查询LLM即超越直接提示法。

    

    表格学习利用结构化数据来预测目标结果。传统上，这一过程依赖于有标签数据。然而，大型语言模型（LLM）可以基于任务描述和特征语义引出领域先验，从而在无标签数据的情况下实现预测。我们提出了边际响应面引导（MARS），这是一种将特征级别的LLM先验转化为可复用的零样本表格分类器的方法。为构建该分类器，MARS从无标签数据中为每个特征选取代表性取值，并提示LLM提供相应的类别支持分数和特征权重。随后，它使用中位数聚合多个响应来构建特征响应函数，并通过它们的加权和进行预测，而无需进一步查询LLM。在八个表格基准任务上，MARS取得了最高的平均AUC和AP，分别超越直接提示法1.97和6.21个百分点。

    arXiv:2609.39639v1 Announce Type: new  Abstract: Tabular learning uses structured data to predict target outcomes. Traditionally, this process has relied on labeled data. However, large language models (LLMs) can be used to elicit domain priors based on the task description and feature semantics, thereby enabling predictions without labeled data. We propose Marginal Response Surface Elicitation (MARS), a method that transforms feature-level LLM priors into a reusable, zero-shot tabular classifier. To construct this classifier, MARS selects representative values for each feature from unlabeled data and prompts the LLM to provide corresponding class support scores and feature weights. It then aggregates multiple responses using the median to construct feature response functions, and makes predictions through their weighted sum without further LLM queries. Across eight tabular benchmark tasks, MARS achieves the highest average AUC and AP, outperforming direct prompting by 1.97 and 6.21 pe
    
[^59]: 该证据对决策是否关键？学习验证受规则约束的决策

    Is This Evidence Decision-Critical? Learning to Verify Rule-Governed Decisions

    [https://arxiv.org/abs/2609.39608](https://arxiv.org/abs/2609.39608)

    提出了InterPact框架，利用基于干预的反事实方法来验证规则决策中证据的关键性，识别可能颠覆决策结果的关键证据，从而支持准确且安全的规则化决策。

    

    基于规则的推理，例如资格审核和合同审查，要求语言模型在明确规则的约束下，针对各个条件逐一评估证据，并综合各项判断得出结论。证据评估中的错误可能不会改变决策结果，但误解或忽视对决策起关键作用的证据则可能颠覆整个决策。识别此类证据能够让更强大的模型专注于核查相应的条件判断，从而支持准确且安全的决策。识别证据的关键性需要理解证据如何影响条件判断，以及该条件判断又如何影响最终决策。为实现这一目标，我们提出了一种基于干预的影响学习框架，使规则约束决策中的证据关键性能够得到反事实验证。具体而言，其证据干预构造器通过编辑案例事实，为传播验证器生成训练样本对……

    arXiv:2609.39608v1 Announce Type: new  Abstract: Rule-based reasoning, as in eligibility checks and contract reviews, requires language models to assess evidence against individual conditions and combine their judgments under explicit rules. Errors in evidence assessment can leave a decision unchanged, but misinterpreting or overlooking decision-critical evidence can reverse it. Identifying such evidence allows more capable models to focus on checking the corresponding condition judgments, supporting accurate and safe decisions. Recognizing the evidence's criticality requires understanding how evidence affects a condition judgment and how that judgment affects the decision. To achieve the goal, we propose a INTERvention-based imPACT learning framework (InterPact), which enables counterfactual verification of evidence criticality in rule-governed decisions. Specifically, its evidence intervention constructor generates training pairs for a propagation verifier by editing case facts with 
    
[^60]: 跳出思维定式：语言模型能否选择性地依赖外部引导？

    Thinking Outside the Box: Can Language Models Rely on External Guidance Selectively?

    [https://arxiv.org/abs/2609.39578](https://arxiv.org/abs/2609.39578)

    该论文提出 Box²-Bench 基准来衡量语言模型“跳出思维定式”的能力，即在受益于可靠工作流引导的同时否决不可靠引导，并发现反事实监督微调与基于结果的强化学习是提升这一能力的两种互补训练策略。

    

    智能体框架通常通过人类设计的工作流来增强语言模型，但随着模型能力的不断增强，不可靠的引导反而可能越来越多地限制模型的执行。我们将这种既能从有用引导中获益、又能否决不可靠引导的能力称为“跳出思维定式”。我们提出了 Box$^2$-Bench 基准，它在保持模型和任务不变的同时改变工作流的可靠性，以单独考察模型如何调节其对引导的依赖程度。在 Box$^2$-Bench 上，前沿模型通常能从可靠引导中受益，但当引导具有误导性或变得不可靠时仍然脆弱。为检验这种能力是否可以通过学习获得，我们使用坏的工作流训练两个开源权重模型，并将好的工作流保留用于评估。我们探索了两种互补的训练策略：反事实监督微调可以提升鲁棒性，而基于结果的强化学习可以将平衡转向更多地利用有用的工作流。

    arXiv:2609.39578v1 Announce Type: new  Abstract: Agent harnesses often improve language models with human-designed workflows, but as models grow more capable, unreliable guidance can increasingly constrain their execution. We call the ability to benefit from useful guidance while overriding unreliable guidance thinking outside the box. We introduce Box$^2$-Bench, which holds the model and task fixed while varying workflow reliability to isolate how models regulate their reliance on guidance. On Box$^2$-Bench, frontier models often benefit from reliable guidance but remain vulnerable when it is misleading or becomes unreliable. To test whether this capability can be learned, we train two open-weight models using bad workflows, reserving good workflows for evaluation. We explore two complementary training strategies: counterfactual supervised fine-tuning improves robustness, while outcome-based reinforcement learning can shift the balance toward greater use of helpful workflows. We furth
    
[^61]: 紧凑的语言，复杂的模型变化：歧义与欠明性如何以及在何处影响大语言模型

    Compact Language, Complex Model Shifts: How and Where Ambiguity and Underspecification Affect LLMs

    [https://arxiv.org/abs/2609.39572](https://arxiv.org/abs/2609.39572)

    本文通过构建人工同音异义词与人工上位词伪词，首次从机制层面揭示了词汇歧义与欠明性如何按比例提升语言模型性能、降低含歧义文本的生成准确率，并在模型内部表征中呈现消歧过程。

    

    我们分析了词汇歧义与欠明性如何影响语言模型训练。我们创造了人工同音异义词和人工上位词作为伪词，并在模型训练中加入数量递增的这类歧义或欠明的伪词类型，分析语言模型的生成性能。我们进一步分析了模型是否能够对歧义或欠明的语句进行消歧，并首次从机制层面解释了歧义与消歧在模型内部是如何被表征的。我们的主要结果表明，歧义与欠明性都会提升模型性能，且这种提升与其对语言类符-形符比的影响程度成正比。然而，与其他文本相比，生成包含歧义词或其同义词的序列的准确率有所下降。我们还发现伪词的内部表征反映了伪同音异义词的消歧过程，但伪上位词的欠明性……

    arXiv:2609.39572v1 Announce Type: new  Abstract: We analyze how lexical ambiguity and underspecification affect language model training. We create artificial homonyms and artificial hypernyms as pseudowords and analyze the generative performance of language models as they are trained with increasing amounts of these ambiguous or underspecified pseudoword types. We further analyze whether the models disambiguate ambiguous or underspecified statements and provide a first mechanistic account of how ambiguity and disambiguation are represented internally. Our main results show that both ambiguity and underspecification increase model performance in ways that scale with their influence on the language's type-token ratio. However, the accuracy of generating sequences containing ambiguous words or their synonyms decreases compared to other texts. We also show that internal representations of pseudowords reflect disambiguation of pseudo-homonyms, but underspecification of pseudo-hypernyms is m
    
[^62]: 推测性安全蜜罐：面向多轮智能体攻击的主动防御

    Speculative Safety Honeypot: Toward Proactive Defense Against Multi-turn Agent Attacks

    [https://arxiv.org/abs/2609.39549](https://arxiv.org/abs/2609.39549)

    本文提出推测性安全蜜罐（SSH）框架，利用小型LLM多智能体模拟系统预测目标智能体的未来行为并构建轨迹树，实现对多轮智能体攻击的主动防御，同时通过真实动作验证有效降低误报。

    

    随着大型语言模型（LLM）智能体越来越多地被部署到复杂环境中，多轮交互攻击已成为一项重大安全挑战。现有的检测方法通常依赖于历史上下文，然而这种回顾式逻辑难以识别那些被拆分到多轮对话中以掩盖未来风险的深层恶意意图。受推测解码的启发，我们提出了推测性安全蜜罐（SSH）框架。SSH使用由小型LLM组成的多智能体模拟系统，构建了动作级的推测与验证工作流。在推测阶段，SSH预测目标智能体的未来行为，并异步构建轨迹树以提前暴露潜在风险。在验证阶段，系统利用目标智能体的真实动作来校准和修剪轨迹树，有效降低误报率。作为一个即插即用的组件，SSH为现有检测方法提供了支持。

    arXiv:2609.39549v1 Announce Type: cross  Abstract: As Large Language Model (LLM) agents are increasingly deployed in complex environments, multi-turn interaction attacks have become a significant security challenge. Existing detection methods typically rely on historical context. However, this retrospective logic struggles to identify deep malicious intents that are split across turns to hide future risks. Inspired by speculative decoding, we propose the Speculative Safety Honeypot (SSH) framework. SSH uses a multi-agent simulation system composed of small LLMs to build an action-level speculate-and-verify workflow. In the speculation stage, SSH predicts future behaviors of the target agent and asynchronously builds a trajectory tree to expose potential risks in advance. In the verification stage, the system uses the target agent's real actions to calibrate and prune the trajectory tree, effectively reducing false positives. As a plug-and-playable component, SSH provides existing detec
    
[^63]: CATCH：一个用于编程强化学习中奖励破解的可控分析测试平台

    CATCH: A Controllable Analysis Testbed for Reward Hacking in Coding RL

    [https://arxiv.org/abs/2609.39533](https://arxiv.org/abs/2609.39533)

    提出 CATCH 测试平台，通过刻意暴露环境漏洞并以独立审计生成黄金标签，实现对编程强化学习中奖励破解行为的可控复现、可靠识别与系统干预研究。

    

    在带有可验证奖励的强化学习（RLVR）过程中，大语言模型（LLM）可能会利用环境中的漏洞来获取高奖励，而并未真正提升预期能力，这种现象即“奖励破解”。尽管奖励破解对训练效率和安全性构成风险，但在训练过程中监测和缓解此类行为仍然具有挑战性，其瓶颈在于缺乏能够重现破解行为并可靠识别该行为的测试平台。我们提出了 CATCH，一个用于研究编程强化学习中奖励破解现象的可控测试平台。CATCH 刻意暴露环境漏洞，并通过对比易受攻击评估器下的“成功”与独立审计下的真实任务正确性，提供基于执行的黄金标签。此外，它还可以通过监督微调数据配比来控制模型的初始破解倾向，并通过奖励设计来调节获取奖励的难度，从而支持对破解动态与干预措施进行系统性比较研究。

    arXiv:2609.39533v1 Announce Type: new  Abstract: During reinforcement learning with verifiable rewards (RLVR), large language models (LLMs) can exploit loopholes in their environments to obtain high rewards without improving the intended capabilities, i.e., reward hacking. Despite its risks to training efficiency and safety, monitoring and mitigating reward hacking during training remain challenging, which is limited by a lack of testbeds that reproduce hacking and reliably identify it. We introduce CATCH, a controllable testbed for studying reward hacking in coding RL. CATCH deliberately exposes environmental loopholes and provides execution-based gold labels by comparing success under a vulnerable evaluator with task correctness under an independent audit. It also can control the model's initial hacking tendency through supervised fine-tuning data mixtures and the difficulty of earning rewards through reward designing, enabling systematic comparisons of hacking dynamics and intervent
    
[^64]: 脉冲驱动的视觉-语言-动作模型

    Spike-driven Vision-Language-Action Model

    [https://arxiv.org/abs/2609.39514](https://arxiv.org/abs/2609.39514)

    提出了首个支持机器人操作端到端直接训练的脉冲驱动视觉-语言-动作（VLA）框架，利用脉冲神经网络的稀疏事件驱动计算和多赢家脉冲融合机制，解决了传统大型Transformer模型延迟高、能耗大而难以在资源受限平台部署的问题。

    

    视觉-语言-动作（VLA）模型连接了多模态理解与机器人控制，推进了具身智能的主导范式。然而，大多数现有模型依赖大型Transformer，其延迟和能耗成本阻碍了在资源受限平台上的部署。通过稀疏的事件驱动计算，脉冲神经网络为高性能和节能计算提供了一种有前景的范式。在此，我们提出了首个支持机器人操作端到端直接训练的脉冲驱动VLA框架，其主要包含三个核心组件。首先，我们开发了用于多模态感知的脉冲视觉编码器和脉冲指令编码器，将视觉观察和语言指令编码为稀疏、可靠的脉冲表示，用于后续的跨模态融合。然后，我们引入了多赢家脉冲融合用于指令引导的场景理解，采用双向top-k赢家通吃脉冲机制……（摘要在此处被截断）

    arXiv:2609.39514v1 Announce Type: new  Abstract: Vision-language-action (VLA) models bridge multimodal understanding and robotic control, advancing the dominant paradigm for embodied intelligence. However, most existing models rely on large Transformers, whose latency and energy costs hinder deployment on resource-constrained platforms. Through sparse event-driven computation, spiking neural networks offer a promising paradigm for high-performance and energy-efficient computing. Here, we propose the first Spike-driven VLA framework enabling end-to-end direct training for robotic manipulation, which mainly comprises three core components. First, we develop spiking visual and instruction encoders for multimodal perception, encoding visual observations and language instructions into sparse, reliable spike representations for subsequent cross-modal fusion. Then, we introduce Multi-Winner Spike Fusion for instruction-guided scene understanding, using bidirectional top-$k$ winner-take-all sp
    
[^65]: 当正确答案缺失时：Jev 中依赖算术的拒绝瓶颈

    When the Right Answer Is Missing: An Arithmetic-Dependent Rejection Bottleneck in Jev

    [https://arxiv.org/abs/2609.39496](https://arxiv.org/abs/2609.39496)

    本研究揭示了 Jev 类型化决策模型存在依赖算术的拒绝瓶颈：在算术问题上答案存在时准确率高达 99%，但答案缺失时即便设有明确拒绝选项，正确拒绝率仅 7%，而原生布尔验证在同样场景下却能实现 99% 的精确匹配准确率。

    

    arXiv:2609.39496v1 通告类型：交叉。摘要：诸如 Jev 这类类型化决策模型通过直接从预定义选项中进行选择，为决策工作流提供了生成式大语言模型的高效替代方案。当候选集不包含有效答案时，TypeSafe 建议加入“其他”或“以上都不是”选项以启用拒绝功能。然而，本报告识别出一个依赖算术的拒绝瓶颈：当正确答案存在时，Jev 能够可靠地选出正确的数值答案，但当正确答案缺失时，尽管设有明确的拒绝选项，它仍经常接受错误的备选答案。在配对的算术问题上，答案存在时的准确率高达 99%，而正确拒绝率却降至 7%。此外，这一差距在数值大小、运算深度、上下文表述和拒绝标签等维度上均持续存在，并扩展到时间计算和容量取整等场景。然而，原生布尔验证在同样的缺失答案算术问题上却能达到 99% 的精确匹配准确率。

    arXiv:2609.39496v1 Announce Type: cross  Abstract: Typed decision models such as Jev offer an efficient alternative to generative LLMs in decision-making workflows by selecting directly from predefined options. When candidate sets contain no valid answer, TypeSafe recommends including an "other" or "none-of-the-above" option to enable rejection. In this report, however, we identify an arithmetic-dependent rejection bottleneck: Jev reliably selects correct numerical answers when available but frequently accepts incorrect alternatives when they are absent despite an explicit rejection option. On paired arithmetic problems, answer-present accuracy reaches 99%, while correct rejection falls to 7%. Moreover, this gap persists across numerical magnitudes, operation depths, contextual formulations, and rejection labels, and extends to scenarios such as time calculation and capacity rounding. Yet native Boolean verification achieves 99% exact-match accuracy on the same answer-absent arithmetic
    
[^66]: 右翼摇滚还是普通摇滚？对Frei.Wild乐队的计算语言学分析

    Right-Wing Rock or Just Rock? A Computational Linguistic Analysis of Frei.Wild

    [https://arxiv.org/abs/2609.39460](https://arxiv.org/abs/2609.39460)

    本研究通过构建德国摇滚与右翼摇滚语料库并训练高性能分类器，对备受争议的乐队Frei.Wild进行计算语言学分析，发现其虽刻意维持政治立场的模糊性，但整体倾向右翼，且超过一半的歌曲被识别为右翼摇滚。

    

    右翼摇滚是摇滚音乐的一个子流派，用于传播右翼意识形态，常被工具化以招募青少年进入激进化圈子。监测机构通过人工审查、有时甚至封禁极端主义内容来对抗这种现象；然而，存在一些规避监管的边缘案例。我们提出了一项研究，旨在确定这样一个边缘案例——Frei.Wild乐队——应当被归类为政治上右倾，还是属于普通德国摇滚流派。我们采样了一个德国摇滚数据集，并创建了一个右翼摇滚语料库作为本次分析的参考。我们发现可以证实此前调查的直觉：Frei.Wild成功地在其政治立场上保持模糊性。然而，其倾向偏向右翼阵营。词汇分析揭示了民族主义叙事，并且两个高性能分类器（ROC-AUC分数高达97%）将超过一半的歌曲标记为右翼摇滚。

    arXiv:2609.39460v1 Announce Type: new  Abstract: Rechtsrock is a subgenre of rock music that spreads right-wing ideology, often instrumentalized to recruit adolescents into the radical scene. Monitoring institutions counteract this by manually examining and, in some cases, banning extremist content; however, there are border cases that evade regulation. We present a study aimed at determining whether such a case, the band Frei.Wild, should be classified as politically right-leaning or as part of the general German rock genre. We sampled a German rock dataset and created a corpus for right-wing rock to use as reference in this analysis and found that we can confirm the intuitions from previous investigations that Frei.Wild successfully maintains an ambiguity with regard to their political affiliation. However, the tendency is towards the right-wing spectrum. Lexical analyses reveal nationalistic narratives and two high-performing classifiers (up to 97% ROC-AUC score) label more than hal
    
[^67]: 从语音到可编辑概念：利用概念瓶颈模型探究情感识别

    From Speech to Editable Concepts: Probing Emotion Recognition with Concept Bottleneck Models

    [https://arxiv.org/abs/2609.39453](https://arxiv.org/abs/2609.39453)

    本工作首次将概念瓶颈模型引入语音情感识别，通过转录文本、声学描述和说话人属性等可编辑概念探究大语言模型预测的依赖因素，揭示并量化了零样本设置下模型对转录文本的强烈偏向，从而提升了情感识别的可解释性。

    

    arXiv:2609.39453v1 公告类型：交叉  摘要：语音情感识别（SER）是为话语分配情感标签的任务。早期系统依赖声学特征，而近期方法则结合多种模态，最常见的是语音和文本。尽管如此，在许多数据集上的性能仍然不佳。因此，大语言模型（LLM）因能够通过指令联合处理多样化输入而在SER领域受到关注。然而，直接输入音频引发了可解释性方面的问题。为了在图像分类中解决类似问题，研究者提出了概念瓶颈模型。本工作将概念瓶颈适配到语音情感识别中，以检验单个预测如何依赖于转录文本、声学描述和说话人属性。实验在CREMA-D、IEMOCAP和MELD数据集上测试了三个大语言模型，其中概念由单独的工具提取。在脚本化语料库上，大语言模型在零样本设置中强烈偏向于转录文本，这使得Macro-F1从27.8降至5.8

    arXiv:2609.39453v1 Announce Type: cross  Abstract: Speech emotion recognition (SER) is the task of assigning emotion labels to utterances. Early systems relied on acoustic features, whereas recent approaches combine multiple modalities, most commonly speech and text. Still, performance remains poor on many datasets. Large language models (LLMs) have therefore attracted interest for SER, as they can process diverse inputs jointly with instructions. However, direct audio input raises questions of explainability. To address similar questions in image classification, concept bottleneck models were introduced. This work adapts concept bottlenecks to SER to examine how individual predictions depend on transcripts, acoustic descriptions and speaker attributes. Experiments test three LLMs on CREMA-D, IEMOCAP and MELD, with concepts extracted by separate tools. On scripted corpora, LLMs are strongly biased towards the transcript in the zero-shot setting, which lowers Macro-F1 from 27.8 to 5.8 o
    
[^68]: 基于训练动力学的合成数据特征刻画

    Synthetic Data Characterization via Training Dynamics

    [https://arxiv.org/abs/2609.39447](https://arxiv.org/abs/2609.39447)

    该论文提出利用样本级可学习性与编码器训练动力学来刻画LLM合成数据的特性，揭示不同LLM系列和规模之间的数据差异，并证明基于可学习性信号的数据选择策略对合成数据与人类数据会产生不同的效果。

    

    解释LLM（大语言模型）生成数据的特性，对于理解其在各类学习任务中的效用和局限性十分重要。在这项工作中，我们通过样本级可学习性来表征合成数据，研究不同LLM系列和规模之间的差异，并以人类撰写的数据作为参考。我们首先生成涵盖单标签和多标签分类、标注以及树预测任务的合成数据集。接着，我们从编码器训练动力学中推导出机器数据与真实人类数据的经验数据分布，并估计这些分布在不同编码器之间的鲁棒性。最后，我们评估了基于这些可学习性信号的数据选择策略对两种数据源所产生的不同影响。

    arXiv:2609.39447v1 Announce Type: new  Abstract: Interpreting properties of LLM-generated data is important for understanding its utility and limitations across learning tasks. In this work, we characterize synthetic data through sample-level learnability, studying variation among LLM families and scales, alongside human-written data as a reference. We first generate synthetic datasets spanning single- and multi-label classification, labeling, and tree prediction tasks. We then derive empirical data distributions from encoder training dynamics for both machine and organic data, and estimate the robustness of these distributions across encoders. Finally, we evaluate how data selection strategies based on these learnability signals affect both data sources differently.
    
[^69]: DuplexAct-Bench：面向多样化行为需求下主动交互的全双工语音评估拓展

    DuplexAct-Bench: Broadening Full-Duplex Speech Evaluation toward Proactive Interaction across Diverse Behavioral Requirements

    [https://arxiv.org/abs/2609.39446](https://arxiv.org/abs/2609.39446)

    DuplexAct-Bench是一个双语全双工语音基准，系统覆盖打断、退让、主动发起、主动沉默、附和等六种行为及多种情境条件，通过对12个系统在1,290个流式试验中的时机与内容评估，揭示了现有系统在实时交互行为管理上的显著不足。

    

    现有的全双工语音基准测试仅覆盖实时交互行为的一部分子集，且往往处于有限的情境条件下。我们提出了DuplexAct-Bench，这是一个双语基准，系统性地覆盖了六种互补行为，从打断、退让到主动发起、主动沉默和附和反馈，并涵盖会话前、会话中和无明确指令三种条件。在1,290个英语和中文流式试验中，我们从时机和内容两个维度评估了12个全双工语音系统。结果揭示了不同行为、条件和系统之间的显著差异，以及语义质量与行为时机之间的频繁不匹配。这些发现表明，在实时交互展开的过程中，当前的系统仍远未能稳健地管理何时、是否以及如何参与交互。

    arXiv:2609.39446v1 Announce Type: new  Abstract: Existing full-duplex speech benchmarks cover only subsets of real-time interaction behaviors, often under limited contextual conditions. We introduce DuplexAct-Bench, a bilingual benchmark that systematically covers six complementary behaviors, from interruption and yielding to proactive initiation, active silence, and backchanneling, across Pre-session, In-session, and No-explicit conditions. Across 1,290 English and Chinese streaming trials, we evaluate 12 full-duplex speech systems on both Timing and Content. Results reveal substantial variation across behaviors, conditions, and systems, as well as frequent mismatches between semantic quality and behavioral timing. These findings show that current systems remain far from robustly managing when, whether, and how to participate as real-time interaction unfolds. Project page: https://alitaxky.icu/DuplexAct-Bench/
    
[^70]: QuantCode 模型：面向可执行算法交易代码的语言模型专业化

    QuantCode Model: Specializing Language Models for Executable Algorithmic Trading Code

    [https://arxiv.org/abs/2609.39420](https://arxiv.org/abs/2609.39420)

    该论文提出通过对算法交易框架代码进行持续预训练并结合经智能体验证的监督微调，将大语言模型专业化以生成可执行、语义忠实的算法交易代码，并发布了包含 400 个任务的 QuantCode-Bench 基准，显著提升了策略代码生成的通过率。

    

    大型语言模型是强大的通用代码生成器，但可执行的算法交易仍然是一个要求很高的专业化目标：模型必须将自然语言描述的策略规范转换为专用交易框架下的正确程序逻辑，在历史数据上执行、产生交易，并在语义上忠实于原始请求。我们研究了两种互补的机制来使语言模型适配这一场景：在算法交易框架代码上进行持续预训练，以及在经智能体验证的“请求-代码”数据对上进行监督微调（SFT）。评估以 QuantCode-Bench 为核心，这是我们构建的用于 Backtrader 策略生成的 400 个任务组成的基准测试，同时还包含一个仓库级的类 SWE-bench 评测赛道。持续预训练将 Qwen3.5-397B-A17B 的单轮 Judge Pass 从 41.5% 提升至 47.5%，将 Qwen3.6-35B-A3B 从 27.8% 提升至 33.0%。在持续预训练之后应用 SFT 则带来了更大的增益……

    arXiv:2609.39420v1 Announce Type: new  Abstract: Large language models are strong general-purpose code generators, but executable algorithmic trading remains a demanding specialization target: a model must translate a natural-language strategy specification into correct program logic for a specialized trading framework, execute on historical data, produce trades, and remain semantically faithful to the request. We study two complementary mechanisms for specializing language models for this setting: continued pretraining on algorithmic-trading framework code and supervised fine-tuning (SFT) on agent-validated request-to-code pairs. Evaluation is centered on QuantCode-Bench, our 400-task benchmark for Backtrader strategy generation, together with a repository-level SWE-bench-like track. Continued pretraining improves single-turn Judge Pass from 41.5% to 47.5% for Qwen3.5-397B-A17B and from 27.8% to 33.0% for Qwen3.6-35B-A3B. SFT applied after continued pretraining yields a larger gain fo
    
[^71]: 早期问题的计算能否帮助大语言模型解决新问题？

    Can Computation from Earlier Problems Help LLMs Solve New Ones?

    [https://arxiv.org/abs/2609.39394](https://arxiv.org/abs/2609.39394)

    提出 STAIR 方法，通过固定存储库复用早期响应的键值信息并仅训练 12,288 个参数，就能让大语言模型在多轮对话中利用先前问题的计算来提升解决新问题的准确率。

    

    大语言模型经常在同一对话中解决相互独立的问题。那么，来自较早问题的计算能否帮助它们解决新问题？为了回答这个问题，我们首先进行了初步实验，结果表明保留的历史记录既可能提高也可能降低后续轮次的准确率，即使是在同一领域内也是如此。为了理解这些影响，我们采用受控重放方法来隔离特定于每个“问题-历史”配对的内部状态变化。结果显示，在不同的历史条件下，这些变化在当前问题之间保持了相似的关系。为了在保留历史的情况下改进推理，我们提出了 STAIR（Stale-Token Attention for Inter-query Reuse，面向跨查询重用的陈旧令牌注意力机制）。STAIR 将较早响应生成过程中的键和值捕获到一个固定存储库中，并学习在提示处理期间当前查询读取该存储库时对其进行重定向。基础模型保持完全冻结，仅训练 12,288 个参数。在三个 Qwen 模型和四个基准测试上，STAIR 提升了平均准确率（原文在此处截断）。

    arXiv:2609.39394v1 Announce Type: new  Abstract: Large language models often solve independent problems in the same conversation. Can computation from earlier problems help them solve new ones? To answer this question, we first conduct preliminary experiments showing that retained history can raise or lower later-turn accuracy, even within the same domain. To understand these effects, we use controlled replay to isolate internal state changes specific to each problem-history pairing. Across different histories, these changes preserve similar relationships among current problems. To improve reasoning under retained history, we introduce STAIR (Stale-Token Attention for Inter-query Reuse). STAIR captures keys and values from earlier response generation in a fixed bank. It learns to redirect current queries when they read this bank during prompt processing. The base model remains frozen; only 12,288 parameters are trained. Across three Qwen models and four benchmarks, STAIR improves avera
    
[^72]: TTLab参加Daleel 2026：STAR-Ar——用于阿拉伯语论辩识别的序列标注方法

    TTLab at Daleel 2026: STAR-Ar, Sequence Tagging for Argument Recognition in Arabic

    [https://arxiv.org/abs/2609.39385](https://arxiv.org/abs/2609.39385)

    本文提出STAR-Ar，一种融合上下文Transformer嵌入与结构化转移约束的BERT-BiLSTM-CRF序列标注架构，用于阿拉伯语论辩话语单元的检测与分类，在Daleel 2026共享任务的测试集上取得73.7的F1分数。

    

    论辩挖掘（AM）是自然语言处理中的一项关键任务，但在阿拉伯语中资源严重不足。本文提出了STAR-Ar，一种用于论辩话语检测与分类的BERT-BiLSTM-CRF架构，作为我们参加Daleel 2026——首届阿拉伯语论辩挖掘共享任务——的系统。该任务要求在辩论和社论文本中识别并分类论辩话语单元（ADU）。我们将这两个目标联合建模为词元级的序列标注任务，所采用的BERT-BiLSTM-CRF架构将上下文Transformer嵌入与结构化转移约束相结合，以支持准确的跨度检测。STAR-Ar在验证集上取得了72.69的F1分数，在测试集上取得了73.7的F1分数。我们的领域特定分析表明，仅在社论上训练的模型表现不及在辩论数据上训练的模型，我们主要将这一差距归因于社论数据的规模较小。

    arXiv:2609.39385v1 Announce Type: cross  Abstract: Argument Mining (AM) is a critical NLP task that remains significantly under-resourced in Arabic. This paper presents $\testtt{STAR-Ar}$, a BERT-BiLSTM-CRF architecture for argument discourse detection and classification, as our system for Daleel 2026, the inaugural Arabic argument mining shared task. The task requires the identification and classification of argumentative discourse units (ADUs) in debate and editorial texts.We jointly model these two objectives as a token-level sequence labeling task using a BERT-BiLSTM-CRF architecture that combines contextual transformer embeddings with structural transition constraints to support accurate span detection. $\testtt{STAR-Ar}$ achieves an F1-score of 72.69 on validation and 73.7 on test data. Our domain-specific analysis shows that models trained exclusively on editorials underperform those trained on debates, a disparity we primarily attribute to the smaller size of the editorial data
    
[^73]: 探索面向复杂知识迁移的异构模型融合方法

    Exploring Heterogeneous Model Merging Approach for Complex Knowledge Transfer

    [https://arxiv.org/abs/2609.39369](https://arxiv.org/abs/2609.39369)

    本文提出两种免训练的异构模型合并方法（Intersection-Merge 和 Activate-Prune-Merge），无需梯度更新或语义对齐，即可在参数层面将专用模型的知识直接迁移到通用语言模型中，并在嵌入、重排序、奖励建模和 MoE 代码专家迁移等任务上有效提升了通用模型性能。

    

    arXiv:2609.39369v1 公告类型：新论文 摘要：专用模型编码了面向特定任务的行为，但要将这种行为迁移到通用语言模型中，通常需要进行训练、蒸馏或表示对齐。我们研究这种能力是否可以直接在参数层面进行迁移。我们将两种现有的免训练异构模型合并方法（此前已被证明可以在通用语言模型之间迁移知识）应用于从专用模型到通用模型的迁移，将专用 donor 模型投影到接收方模型的形状中，并在无需梯度更新或语义对齐的情况下对骨干网络参数进行插值。Intersection-Merge (IM) 注入一个与接收方形状匹配的前缀对齐 donor 切片，而 Activate-Prune-Merge (APM) 利用前向传播的激活统计信息来选择注入前需要保留哪些 donor 维度。在嵌入、重排序、奖励建模以及 MoE 代码专家模型迁移等多项任务中，这两种方法都提升了通用接收方模型的性能，表明……

    arXiv:2609.39369v1 Announce Type: new  Abstract: Specialized models encode task-oriented behavior, but transferring that behavior to a general language model usually requires training, distillation, or representation alignment. We study whether such ability can instead be transferred directly at the parameter level. We apply two existing training-free heterogeneous merging methods, previously shown to transfer knowledge between general language models, to specialist-to-general transfer, projecting a specialist donor into the recipient's shape and interpolating backbone parameters without gradient updates or semantic alignment. Intersection-Merge (IM) injects a prefix-aligned donor slice matching the recipient shape, while Activate-Prune-Merge (APM) uses forward-pass activation statistics to select which donor dimensions to retain before injection. Across embedding, reranking, reward modeling, and MoE code-specialist transfer, both methods improve the general recipient, showing that sim
    
[^74]: 让网格束搜索不再那么贪心

    Making Grid Beam Search Less Greedy

    [https://arxiv.org/abs/2609.39368](https://arxiv.org/abs/2609.39368)

    本文揭示了网格束搜索在强制执行词汇约束时存在偏差，即倾向于优先满足较简单的约束而将困难的约束推迟到生成序列的末尾，这与不存在此偏差的DFA约束束搜索形成鲜明对比。

    

    约束自回归文本生成模型输出的一种常见形式是词汇约束，即要求某些词或短语必须出现在生成文本中。DFA约束束搜索和网格束搜索是在自回归模型解码时强制执行词汇约束的两种广泛使用的范式。由于前一种方法所需的前向传播次数随约束词元数量呈指数级增长，它通常不如后者受青睐，因为后者只需要线性次数的前向调用。然而，尽管网格束搜索实现了指数级的加速，但其实现方式并未平等对待所有约束。在本文中，我们证明了网格束搜索偏向于优先纳入更容易满足的约束，而将更难的约束留到序列的末尾。这与DFA约束束搜索形成对比，后者不表现出这种偏差。为了解决这一缺点……（原文摘要到此截断）

    arXiv:2609.39368v1 Announce Type: new  Abstract: A common formalism for constraining the output of autoregressive text generation models involves lexical constraints, words or phrases which are required to occur in the generated text. DFA-constrained beam search and grid beam search are two widely used paradigms for decoding from autoregressive models while enforcing lexical constraints. As the former approach requires a number of forward passes exponential in the number of constraint tokens, it is often dispreferred to the latter, which requires only linearly many forward calls. However, while grid beam search achieves an exponential speedup, it does so in a manner which does not treat all of the constraints equally. In this paper, we demonstrate that grid beam search is biased to incorporate easier-to-satisfy constraints first, leaving harder constraints to the end of the sequence. This contrasts with DFA-constrained beam search, which exhibits no such bias. To address this shortcomi
    
[^75]: Ready2Blend：从自然语言指令到可组合的对齐提示词

    Ready2Blend: From Natural-Language Instructions to Composable Alignment Prompts

    [https://arxiv.org/abs/2609.39365](https://arxiv.org/abs/2609.39365)

    Ready2Blend 提出 AlignFormer，将自然语言需求转化为存储在模块化提示词库中的可组合对齐提示词，并通过可组合性正则化在冻结骨干网络的情况下实现推理时的提示词混合与重加权，其性能可媲美基于后训练的持续对齐方法。

    

    持续对齐要求大语言模型在适应新需求的同时，不遗忘先前已习得的行为。自然语言指令灵活且可组合，但只能提供间接控制；而后训练虽然能带来更强的适应能力，却以反复的参数更新为代价。我们提出了 Ready2Blend，它将自然语言的灵活性与习得的对齐能力相结合。AlignFormer 将每条需求映射为固定长度的对齐提示词，存储于模块化的提示词库中，同时骨干模型与已有提示词保持冻结。可组合性正则化将文本需求的语义几何结构迁移到提示词空间，从而支持推理时的混合与重新加权。在两种实用的持续对齐设置中，Ready2Blend 是唯一能够与基于后训练的对齐方法相媲美的冻结骨干方法，达到联合训练参考性能的 93.1%–98.5%，同时保持了具有竞争力的遗忘控制（保留率）。

    arXiv:2609.39365v1 Announce Type: cross  Abstract: Continual alignment requires LLMs to adapt to new requirements without forgetting previously acquired behaviors. Natural-language instructions are flexible and composable but offer only indirect control, whereas post-training provides stronger adaptation at the cost of repeated parameter updates. We introduce Ready2Blend, which combines the flexibility of natural language with learned alignment. AlignFormer maps each requirement to a fixed-length alignment prompt stored in a modular prompt bank, while the backbone and prior prompts remain frozen. Composability regularization transfers the semantic geometry of textual requirements into prompt space, enabling inference-time blending and reweighting. Across two practical continual alignment settings, Ready2Blend is the only frozen-backbone method that matches post-training-based alignment methods, reaching $93.1$-$98.5\%$ of a joint-training reference with competitive retention, while req
    
[^76]: 绕过算力天花板：Galahad 中的字节精确记忆使 LLM 阅读成为一次性成本

    Working Around the Compute Ceiling: Byte-Exact Memory in Galahad Makes LLM Reading a One-Time Cost LLM Reading a One-Time Cost

    [https://arxiv.org/abs/2609.39358](https://arxiv.org/abs/2609.39358)

    Galahad 通过为 vLLM、SGLang 和 llama.cpp 提供字节精确的记忆层（Taliesin 缓存并复用 KV 状态、Blaise 按需传递文档章节），避免对已读文本的重复计算，使 LLM 的阅读成为一次性成本，从而绕过每 token 算力上限的限制。

    

    Transformer 语言模型在每个 token 上可执行的算力是有限的，Infosys 前首席执行官 Vishal Sikka 近期的研究认为，这一上限限制了模型能够执行或验证的任务范围（arXiv:2507.07505）。我们要问的是：在这一天花板之下的算力预算中，有多少被花费在模型已经完成过的工作上。推理服务在请求之间是无状态的：当模型针对同一份文档回答第二个问题时，它会从第一个 token 开始重新计算该文档的注意力状态。在七个真实世界数据集上，98.7% 的提示 token 都是模型已经读过的文本。我们提出了 Galahad——一个面向 vLLM、SGLang 和 llama.cpp 的记忆层，它使这种阅读成为一次性成本。Taliesin 为一段文本保存模型的键值（KV）状态，并在下一个包含相同字节的请求中直接加载，而无需重新计算。Blaise 则保存文档本身，仅将问题所需的章节传递给模型。在一个召回测试中……

    arXiv:2609.39358v1 Announce Type: cross  Abstract: A transformer language model performs a bounded amount of computation per token, and recent work by Vishal Sikka, former CEO of Infosys, argues that this bound limits which tasks a model can carry out or verify (arXiv:2507.07505). We ask how much of the budget beneath that ceiling is spent on work the model has already done. Serving is stateless across requests: a model that answers a second question about a document recomputes the document's attention state from the first token. On seven real-world datasets, 98.7% of prompt tokens were text the model had already read. We present Galahad, a memory layer for vLLM, SGLang and llama.cpp that makes this reading a one-time cost. Taliesin saves the model's key-value (KV) state for a block of text and loads it on the next request that contains the same bytes, instead of recomputing it. Blaise keeps the documents themselves and passes the model only the section a question needs. On a recall te
    
[^77]: 离线引导，在线推理：复用大语言模型反馈助力小语言模型

    Offline Guidance, Online Reasoning: Reusing LLM Feedback for Small Language Models

    [https://arxiv.org/abs/2609.39346](https://arxiv.org/abs/2609.39346)

    该论文提出“离线引导、在线推理”的LLM-SLM协作方式，通过复用LLM针对问题生成的反馈作为离线引导，使小语言模型在在线推理时无需反复访问LLM即可提升推理能力。

    

    大语言模型（LLM）具备强大的推理能力，但通过商业API调用往往成本高昂；而小语言模型（SLM）更易于本地部署，但推理能力较弱。这种能力与部署之间的差距催生了LLM-SLM协作研究，其目标是在保留SLM部署优势的同时，利用LLM的能力提升SLM的推理水平。现有方法主要遵循两种范式：其一是知识蒸馏，利用LLM生成的答案和推理轨迹离线训练SLM，但需要参数更新和额外训练；其二是在线协作，在SLM遇到困难时将难题路由给LLM，或利用LLM生成的引导与纠正，尽管有效，但需要反复访问LLM。此外，针对特定问题生成的引导在推理结束后即被丢弃，无法使后续问题受益……

    arXiv:2609.39346v1 Announce Type: new  Abstract: Large language models (LLMs) offer strong reasoning capabilities but are often costly to access through commercial APIs, while small language models (SLMs) are easier to deploy locally yet remain weaker in reasoning. This capability-deployment gap has motivated LLM-SLM collaboration, which aims to improve SLM reasoning using LLM capabilities while preserving the deployment advantages of SLMs. Existing approaches mainly follow two paradigms. Knowledge distillation uses LLM-generated answers and reasoning trajectories to train SLMs offline, but requires parameter updates and additional training. Alternatively, online collaboration routes difficult problems to an LLM or leverages LLM-generated guidance and corrections when an SLM encounters difficulties. Although effective, online collaboration requires repeated LLM access. Moreover, the guidance produced for a particular problem is discarded after inference and cannot benefit subsequent pr
    
[^78]: 理解即无套利：有界荷兰赌作为语言模型的理解定义与训练目标

    Understanding as No-Arbitrage: Bounded Dutch Books as a Definition and Training Objective for Language Models

    [https://arxiv.org/abs/2609.39341](https://arxiv.org/abs/2609.39341)

    本文提出用“无法被计算能力受限的交易者构造荷兰赌套利”来定义和度量语言模型的“理解”程度，证明理解本质上是分级的、下一词元预测目标本身会导致跨问题形式的不一致，并据此提出将抗套利作为训练目标的Arbitr框架。

    

    语言模型仅仅是在预测词元，还是真正理解它所说的话？我们通过“无套利”的视角对“理解”进行定义，使这一问题变得可度量。如果一个计算能力受限的交易者无法通过针对模型在逻辑相关命题上的概率进行下注而获取保证利润（即“荷兰赌”），那么该模型就在一定程度上理解了这个词表。我们建立了三个理论结果：第一，由于完全的逻辑一致性在计算上是不可解的，理解本质上是分等级的，而非绝对的。第二，我们证明标准下一词元预测的精确最优解在不同问题形式之间本质上是不一致的；缺陷在于训练目标本身，而非模型架构。第三，我们表明不确定性会沿着推理链可预测地累积，使无根据的过度自信本身就构成套利机会。为解决这一问题，我们引入了Arbitr，一种训练框架……

    arXiv:2609.39341v1 Announce Type: new  Abstract: Does a language model merely predict tokens, or does it understand what it says? We make this question measurable by defining "understanding" through the lens of no-arbitrage. A model understands a vocabulary to a certain degree if a computationally bounded trader cannot extract guaranteed profit by betting against the model's probabilities on logically related claims (a "Dutch book"). We establish three theoretical results: first, because full logical coherence is computationally intractable, understanding is inherently graded, not absolute. Second, we prove that the exact optimum of standard next-token prediction is inherently incoherent across different question formats; the flaw lies in the training objective, not the architecture. Third, we show that uncertainty accumulates predictably along reasoning chains, making unjustified overconfidence an arbitrage opportunity in itself. To address this, we introduce Arbitr, a training framew
    
[^79]: 驯服LLM服务中测试时扩展的推测搜索

    Taming Speculative Search for Test-Time Scaling in LLM Serving

    [https://arxiv.org/abs/2609.39334](https://arxiv.org/abs/2609.39334)

    本文提出SpecScale服务系统，通过提前剪枝低质量候选路径、冗余路径计算去重和延迟细粒度验证三种技术，解决了LLM服务中推测执行导致的搜索空间爆炸和频繁验证挑战，实现高效的测试时扩展。

    

    测试时扩展（Test-time scaling）最近已成为一种提升LLM推理能力的强大方法，它通过在推理过程中分配额外的计算，显著提高了数学和编程等具有挑战性任务的准确性。为了加速推理路径的探索，近期研究提出了推测执行（speculative execution）。然而，我们表明支持推测执行给LLM服务系统带来了两个独特的挑战：（1）候选路径搜索空间的爆炸式增长，以及（2）对候选项进行频繁的细粒度验证任务。为了应对这些挑战，本文提出了SpecScale，一个用于高效推测执行的服务系统。我们引入三种技术来调和延迟与计算开销之间的权衡：（1）提前剪枝低质量的候选路径，（2）对冗余候选路径之间的计算进行去重，以及（3）延迟细粒度验证任务。

    arXiv:2609.39334v1 Announce Type: cross  Abstract: Test-time scaling has recently emerged as a powerful approach for improving LLM reasoning by allocating additional computation during inference, substantially enhancing accuracy on challenging tasks such as mathematics and coding. To accelerate the exploration of reasoning paths, recent studies proposed speculative execution. However, we show that supporting speculative execution poses two unique challenges for LLM serving systems: (1) an explosion in the search space of candidate paths and (2) frequent, fine-grained verification tasks for candidates.   To address these challenges, this paper proposes SpecScale, a serving system for efficient speculative execution. We introduce three techniques to reconcile the trade-off between latency and computational overhead: (1) early pruning of low-quality candidate paths, (2) deduplicating computation across redundant candidate paths, and (3) deferring fine-grained verification tasks. We evalua
    
[^80]: NarrativeSteward：在智能体辅助的互动叙事创作中协调委托、引导与验证

    NarrativeSteward: Coordinating Delegation, Guidance, and Verification in Agent-Assisted Interactive Narrative Authoring

    [https://arxiv.org/abs/2609.39333](https://arxiv.org/abs/2609.39333)

    NarrativeSteward是一个智能体辅助的互动叙事创作环境，通过将大纲、世界观与叙事图组织为关联工件，并结合智能体对话、结构审查、变更记录与执行验证，帮助作者理解、引导和评估智能体生成与修订的海量叙事内容。

    

    自主AI智能体能够通过独立组织和执行生成与修订，将作者的目标转化为互动叙事。然而，随着智能体生成和修订大量内容，作者难以把握作品的整体结构、局部细节及其相互关系，使得持续的引导变得困难。我们提出了NarrativeSteward，这是一个创作环境，它将大纲、世界观构建和叙事图组织为相互关联的工件，供智能体执行和作者引导使用。智能体对话和项目级结构审查帮助作者理解不断演进的作品，并指导局部和跨层修订；同时，变更记录和执行验证帮助作者评估最终成果。技术测试验证了系统的变更记录、恢复机制和执行诊断功能。在一项有12名参与者的被试内研究中，NarrativeSteward使用户能够更轻松地提出修订请求并检查变更。

    arXiv:2609.39333v1 Announce Type: cross  Abstract: Autonomous AI agents can turn authors' goals into interactive narratives by independently organizing and carrying out generation and revision. As agents generate and revise extensive content, authors struggle to grasp its overall structure, local details, and relationships, complicating continued guidance. We present NarrativeSteward, an authoring environment that organizes outlines, worldbuilding, and narrative graphs as linked artifacts for agent implementation and author guidance. Agent dialogue and project-wide structural review help authors understand the evolving work and guide local and cross-layer revisions, while change records and execution verification help authors assess the resulting work. Technical tests validated the system's change records, recovery mechanisms, and execution diagnostics. In a 12-participant within-subject study, NarrativeSteward supported easier formulation of revision requests and inspection of changes
    
[^81]: 倾斜的碗并非滑坡：压缩循环模型

    A Tilted Bowl Is Not a Slippery Slope: Compressing Looped Models

    [https://arxiv.org/abs/2609.39277](https://arxiv.org/abs/2609.39277)

    该研究推翻了循环模型压缩崩溃源于舍入误差累积的传统观点，揭示了误差只是移动了循环收敛点，据此实现了仅需单次无标签测量即可预测模型失败，并通过最后几轮8位权重循环使失败模型恢复性能。

    

    循环模型通过多次应用同一权重块来进行推理，因此压缩该权重块可以在每一轮循环中节省内存流量。然而，压缩后的循环模型常常会崩溃，这种崩溃通常被归咎于随循环不断累积的舍入误差。在这项工作中，我们在来自五个系列的30多个模型上检验了这一说法，并惊讶地发现，该说法仅适用于永不收敛的循环。当循环收敛（稳定）时，固定的舍入误差并不会累积。它只是移动了循环收敛的位置，就像倾斜的碗会改变球最终静止的位置一样，只有当偏移量超过读取端所能容忍的范围时，答案才会丢失。这一图景使我们能够通过单次无标签测量来预测哪些模型会失败，并且解释了为什么失败的模型能够恢复：它们的循环仍然会收敛，因此只需在最后几轮循环中使用8位权重就能找回答案。受这些发现的启发，我们构建了一个控制器来停止……

    arXiv:2609.39277v1 Announce Type: cross  Abstract: Looped models reason by applying the same block of weights many times, so compressing that block saves memory traffic on every loop. Compressed looped models, however, often collapse, and the collapse is usually blamed on rounding error that accumulates from loop to loop. In this work we test that account on more than 30 models from five families and find, to our surprise, that it holds only for loops that never settle. When a loop settles, a fixed rounding error does not accumulate. It moves the point where the loop settles, much as tilting a bowl moves where a ball comes to rest, and the answer is lost only when the shift is larger than the readout tolerates. This picture lets us predict which models fail from a single label-free measurement, and it tells us why failed models recover: their loops still settle, so a few final loops with 8-bit weights bring the answer back. Motivated by these findings, we build a controller that stops 
    
[^82]: 概念子空间在Logit透镜之外进行计算：一种仅依赖权重来定位读出上游表示的检验方法

    Concept Subspaces Compute Beyond the Logit Lens: A Weights-Only Test for Locating Representations Upstream of Readout

    [https://arxiv.org/abs/2609.39263](https://arxiv.org/abs/2609.39263)

    该论文提出了一种仅需模型权重的几何诊断方法，通过测量概念子空间与unembedding矩阵主导右奇异方向的重叠度，发现概念表示（如FARS）仅携带极少读出方向能量，证明其位于输出读出的上游，超越了Logit透镜所能解释的范围。

    

    概念子空间对模型行为的影响并不能确定它与输出读出之间的关系。我们引入了一种双向几何诊断方法，用于衡量提取出的子空间与unembedding矩阵主导右奇异方向的重叠程度，并以面向输出的阳性对照作为评估基准。给定一个提取出的基，该原始诊断仅需要模型权重即可完成。我们的测试平台是格式无关推理子空间（FARS），这是一个从以六种表面形式表达的十八个推理概念中提取出的十维基。在九个秩匹配的估计器和二十六个模型上，四个由激活导出的概念估计器在前十个读出方向跨度中的平均能量仅占0.38–0.80%。最后一层PCA携带3.56%的能量，在26个模型中的25个上超过FARS。一个通过拟合线性转换器进行深度匹配的同层下一词控制所携带的能量约为FARS的十三倍，且分离（摘要在此处截断）

    arXiv:2609.39263v1 Announce Type: new  Abstract: A concept subspace's effect on model behavior does not establish how it relates to the output readout. We introduce a two-sided geometric diagnostic that measures an extracted subspace's overlap with the dominant right-singular directions of the unembedding matrix, evaluated against output-oriented positive controls. Given an extracted basis, the raw diagnostic requires only model weights. Our testbed is the Format-Agnostic Reasoning Subspace (FARS), a ten-dimensional basis extracted from eighteen reasoning concepts expressed in six surface forms. Across nine rank-matched estimators and twenty-six models, four activation-derived concept estimators carry only 0.38--0.80% mean energy in the top-ten readout span. Final-layer PCA carries 3.56%, exceeding FARS in 25 of 26 models. A same-layer next-token control, evaluated using a fitted linear translator for depth matching, carries approximately thirteen times more energy than FARS, with sepa
    
[^83]: 4MT-VLM：视觉语言模型的认知地图有多粗糙？

    4MT-VLM: How Coarse Is a VLMs Cognitive Map?

    [https://arxiv.org/abs/2609.39238](https://arxiv.org/abs/2609.39238)

    该论文提出4MT-VLM基准，发现视觉语言模型虽然能从熟悉视角识别地点，但一旦视角旋转便无法维持认知地图能力，在135度时甚至低于随机水平，远逊于人类的85%表现。

    

    arXiv:2609.39238v1 公告类型：新 摘要：一个移动的智能体必须能够从它从未见过的视角识别某个地点。我们提出了4MT-VLM，这是一个程序化生成的景观数据集，每个景观以五种刺激模式渲染，这些模式在保持布局不变的同时移除了外观线索：形状与颜色、仅形状、仅颜色、没有物体的裸露地形山峰，以及将山峰置于地平线上的山谷视角。最后一种条件在临床上常用于探测人类患者的海马体功能。我们在十六个不同的开源和闭源模型上测试了这一基准，并报告4选1强迫选择（4AFC）性能，该指标同样用于对人类参与者进行评分。我们观察到，模型能够从学习过的视角识别地点，但一旦相机移动就会丧失识别能力，在135度旋转时降至25%的随机水平以下，而人类观察者在相同条件下的得分为85%。前沿模型（Gemini 3.8 Flash、GPT-5.6）在旋转测试中仅能正确回答39%和31%的题目……

    arXiv:2609.39238v1 Announce Type: new  Abstract: An agent that moves must recognise a place from a viewpoint it has never seen. We introduce 4MT-VLM, a dataset of procedurally generated landscapes, each rendered across five stimulus modes that remove appearance cues while holding layout fixed: shape and colour, shape only, colour only, bare terrain peaks with no objects, and a valley viewpoint that puts the peaks on the horizon. The last condition is commonly used in clinics to probe hippocampal function in human patients. We test this benchmark across sixteen different open and closed-source models and report 4AFC performance, a measure which is also used to grade human participants. We observe that models identify a place from the studied viewpoint but lose it once the camera moves, dropping below the 25% chance level at 135{\deg} where a human observer scores 85%. Frontier models (Gemini 3.8 Flash, GPT-5.6) answer only 39% and 31% of rotated trials correctly, recovering to 85% and 5
    
[^84]: RAIM：面向幻觉检测的廉价模型鲁棒聚合方法

    RAIM: Robust Aggregation of Inexpensive Models for Hallucination Detection

    [https://arxiv.org/abs/2609.39229](https://arxiv.org/abs/2609.39229)

    提出RAIM聚合方案，通过鲁棒的堆叠逻辑回归与可采纳性检验，将多个廉价的开放权重小模型聚合起来，在幻觉/忠实度检测任务上能以极低成本接近Claude Sonnet等前沿专有模型的判断水平。

    

    忠实度的自动评估日益依赖由大语言模型担任裁判，然而最可靠的裁判是专有的前沿模型，其成本高昂且不适合高吞吐量的监控场景。我们研究能否将一组廉价的开放权重裁判模型（4–9B）聚合起来以替代前沿模型，这种替代会牺牲什么，以及何时值得进行这种替代。我们提出RAIM，这是一种对成员间相关误差具有鲁棒性的聚合方案，它将交叉拟合的堆叠逻辑回归与一种可采纳性检验相结合，该检验从成员模型自身的输出中读取信息，用以判断何时对成员进行聚合能够优于其中最佳的单个成员，并且仍能接近前沿裁判的水平。我们在八个忠实度基准上，使用来自互不相交模型系列的十个裁判模型对RAIM进行了实例化。与Claude Sonnet相比，该裁判组保留了其中位数93%的Cohen's κ，平衡准确率平均仅下降2.9个百分点；读

    arXiv:2609.39229v1 Announce Type: cross  Abstract: Automatic evaluation of faithfulness increasingly relies on a large language model acting as a judge, yet the most reliable judges are proprietary frontier models, costly and ill-suited to high-throughput monitoring. We investigate whether a panel of cheap open-weight judges (4--9B) can be aggregated to stand in for a frontier one, what the substitution sacrifices, and when it is worth making. We propose RAIM, an aggregation scheme robust to the members' correlated errors, coupling a cross-fitted stacked logistic regression with an admissibility test that, read from the members' own outputs, identifies when aggregating them improves on their best member and stays within reach of the frontier judge. We instantiate RAIM with ten judges from disjoint families across eight faithfulness benchmarks. Against Claude Sonnet, the panel retains a median 93% of its Cohen's $\kappa$ and gives up only 2.9 points of balanced accuracy on average; read
    
[^85]: 在线对话中的论证结构预测：建模范式与任务架构的比较研究

    Argument Structure Prediction in Online Conversations: A Comparative Study of Modeling Paradigms and Task Architectures

    [https://arxiv.org/abs/2609.39225](https://arxiv.org/abs/2609.39225)

    该论文在严格模式约束下系统比较了监督微调与基于提示的大语言模型在单步及多步任务架构上的论证结构预测表现，并从预测性能、跨领域泛化、模式遵循度和计算效率四个维度进行了统一评估。

    

    论证结构预测（ASP）通过识别论证单元及其关系，从话语中构建完整的论证结构。尽管近期研究探索了多种方法——包括统一的神经模型、多步骤流水线以及基于提示的大语言模型（LLM）——但这些方法之间的相对权衡仍未得到充分研究，尤其是在对话场景中。我们在严格的模式约束下对ASP进行了系统性评估，比较了监督微调与基于提示的大语言模型在单步和多步任务架构上的表现，从对话输入端到端地生成完整的论证结构。我们在三个不同的对话语料库上进行基准测试，这些语料库由推理锚定理论改编为双极论证结构。在统一的评估框架下，我们评估了预测性能、跨领域泛化能力、模式遵循度以及计算效率。我们的结果表明，ASP仍然……

    arXiv:2609.39225v1 Announce Type: new  Abstract: Argument structure prediction (ASP) constructs complete argument structures from discourse by identifying argumentative units and their relations. While recent work has explored diverse approaches---including unified neural models, multi-step pipelines, and prompt-based large language models (LLMs)---their relative trade-offs remain under-explored, particularly in dialogical settings.   We present a systematic evaluation of ASP under strict schema constraints, comparing supervised fine-tuning and prompt-based LLMs across single- and multi-step task architectures, generating complete argument structures from dialogical input end-to-end. We benchmark them on three diverse dialogical corpora adapted from Inference Anchoring Theory into bipolar argument structures. Under a shared evaluation framework, we assess predictive performance, cross-domain generalization, schema compliance, and computational efficiency. Our results show that ASP rema
    
[^86]: ViLegalExpert：基于真实法律咨询的越南语法律检索与问答大规模基准测试集

    ViLegalExpert: A Large-Scale Benchmark for Vietnamese Legal Retrieval and Question Answering from Real-World Consultations

    [https://arxiv.org/abs/2609.39189](https://arxiv.org/abs/2609.39189)

    该论文提出了基于真实公民-律师咨询构建的越南语法律基准ViLegalExpert，涵盖34个法律领域的17.2万个问题及专家验证证据，为法律检索与问答提供了大规模评测资源，并揭示了证据检索和有据答案生成的重大挑战。

    

    可信赖的法律人工智能需要能够在回答法律问题的同时，将回答建立在权威来源基础上的系统。然而，现有的越南语法律基准对真实世界法律咨询的覆盖十分有限。我们提出了**ViLegalExpert**，这是一个基于真实公民-律师咨询构建的大规模基准，包含超过**172K**个问题，涵盖**34**个法律领域，并附带专业回答和经专家验证的法律证据。ViLegalExpert支持法律信息检索、抽取式问答和生成式问答。使用代表性检索方法和语言模型进行的实验表明，证据检索和有据可依的答案生成面临重大挑战。虽然预训练模型在问答任务上表现强劲，但混合检索方法取得了最佳检索性能。这些结果表明，将自然表达的法律问题映射到权威法律条文是一项非常困难的任务。

    arXiv:2609.39189v1 Announce Type: new  Abstract: Trustworthy Legal AI requires systems that can answer legal questions while grounding their responses in authoritative sources. However, existing Vietnamese legal benchmarks provide limited coverage of real-world legal consultations. We introduce \textbf{ViLegalExpert}, a large-scale benchmark constructed from authentic citizen--lawyer consultations, containing over \textbf{172K} questions across \textbf{34 legal domains}, together with professional answers and expert-verified legal evidence. ViLegalExpert supports legal information retrieval, extractive QA, and abstractive QA. Experiments with representative retrieval methods and language models reveal substantial challenges in evidence retrieval and grounded answer generation. While pretrained models perform strongly on QA, hybrid retrieval achieves the best retrieval performance. These results demonstrate the difficulty of mapping naturally expressed legal questions to authoritative p
    
[^87]: DAGent：面向深度研究智能体的“先评估后生长”规划方法

    DAGent: Evaluate-then-Grow Planning for Deep Research Agents

    [https://arxiv.org/abs/2609.39154](https://arxiv.org/abs/2609.39154)

    DAGent提出“先评估后生长”的增量规划框架，编排器根据已完成节点的置信度和不确定性信号逐批扩展DAG任务图，克服了传统“先计划后修补”策略在深度研究任务中过早承诺、浪费计算的脆弱性。

    

    深度研究任务要求智能体能够在庞大的知识空间中导航，跨多个来源综合证据，并随着研究发现的涌现而动态调整其计划。基于有向无环图（DAG）的多智能体系统非常适合这类场景，因为它们支持并行执行，并将每个子任务隔离在聚焦的依赖上下文中。然而，现有的基于DAG的智能体在执行前就实例化任务级计划，只有在观察到失败或证据缺失之后才对图结构进行修补。这种“先计划后修补”（Plan-then-Patch）策略对于深度研究而言十分脆弱：系统在证据最薄弱的时候做出最强烈的承诺，而后期的修订则会在本不该规划的分支上浪费计算资源。我们提出了DAGent，一个采用“先评估后生长”（Evaluate-then-Grow）增量规划的基于DAG的多智能体框架：编排器一次一批地生长任务图，每次扩展都以已完成节点的置信度和不确定性信号为条件。

    arXiv:2609.39154v1 Announce Type: cross  Abstract: Deep research tasks require agents to navigate large knowledge spaces, synthesize evidence across many sources, and adapt their plans as findings emerge. Directed acyclic graph (DAG)-based multi-agent systems suit this setting because they support parallel execution and isolate each sub-task within a focused dependency context. Yet existing DAG-based agents instantiate a task-level plan before execution and repair the graph only after failures or missing evidence are observed. This Plan-then-Patch strategy is brittle for deep research: the system commits most strongly when its evidence is weakest, and later revisions waste computation on branches that should not have been planned. We propose DAGent, a DAG-based multi-agent framework with Evaluate-then-Grow incremental planning: an Orchestrator grows the task graph one batch at a time, conditioning each expansion on confidence and uncertainty signals from completed nodes. A hierarchical
    
[^88]: 诊断推理语言模型的同策略自蒸馏

    Diagnosing On-Policy Self-Distillation for Reasoning Language Models

    [https://arxiv.org/abs/2609.39118](https://arxiv.org/abs/2609.39118)

    该论文系统诊断了同策略自蒸馏（OPSD）在数学推理中的作用，发现教师信号由推理模式对齐与完整教师前缀共同塑造而非仅由特权语义决定，因此OPSD仅在狭窄的兼容性条件下有效提升推理，否则会导致无效的长度增长、持续退化或行为崩溃。

    

    同策略自蒸馏（OPSD）作为一种提升语言模型推理能力的有前景的方法，正吸引越来越多的关注。在既不需要外部奖励、也不需要单独的更强教师模型的情况下，拥有特权信息的自教师能够在学生的轨迹上提供密集的监督信号。然而，其在语言推理中的行为仍不明确，已有报告的结果从适度的性能提升到行为崩溃不等。在这项工作中，我们在跨越0.6B至8B参数规模的模型上，对OPSD在数学推理中的表现进行了诊断。我们通过受控实验和token级分析深入探究了OPSD。我们指出，教师信号是由推理模式对齐和完整的教师前缀共同塑造的，而不仅仅取决于特权语义。OPSD仅在狭窄的兼容性区间内改善推理能力；在其他情况下，它会导致无效的长度增长、稳定的性能退化或行为崩溃。token级分析表明……（原文摘要在此处截断）

    arXiv:2609.39118v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) has attracted growing interest as a promising approach to improve the reasoning ability of language models. Without external rewards nor a separate stronger teacher, the self-teacher with privileged information could provide dense signals on student's trajectories. However, its behavior in language reasoning remains unclear, with reported outcomes ranging from modest gains to behavioral collapse. In this work, we diagnose OPSD for mathematical reasoning across models spanning 0.6B--8B parameters. We conduct controlled experiments and token-level analyses to fully delve into OPSD. We point out that teacher's signal is shaped by reasoning-mode alignment and the complete teacher prefix, rather than by privileged semantics alone. OPSD improves reasoning only in narrow compatibility regimes. Otherwise, it produces ineffective length growth, stable degradation, or behavioral collapse. Token-level analysis sho
    
[^89]: Bongard：训练机器直觉

    Bongard: Training Machine Intuition

    [https://arxiv.org/abs/2609.39111](https://arxiv.org/abs/2609.39111)

    该论文提出 Bongard，一个开放权重的“系统一”模型，首次将机器直觉作为独立能力进行设计与训练，通过共享编码器加并行解码器分支、概率输出头以及三阶段训练加联合嵌入后训练，在单块 Blackwell GPU 上以 70.9 亿参数实现了对改写情境仍保持高准确率的判断能力。

    

    arXiv:2609.39111v1 公告类型：新论文 摘要：人类智能在很大程度上依赖于习得的直觉：无需显式展开每一个中间步骤，就能识别模式并判断情境。我们推出了 Bongard，一个开放权重的“系统一”模型，它将机器直觉视为一种可以独立设计和训练的能力。该模型采用 T5Gemma 2 4B-4B 编码器-解码器架构，将“阅读证据”与“做出判断”分离开来。编码器结合问题指令对状态进行双向阅读，多个解码器分支共享这一编码，因此针对同一情境的多次判断只需对状态进行一次阅读。经过训练的输出头直接返回所给候选选项上的概率分布，而无需生成文本。训练分三个阶段进行：从有监督的判断，到语义关系，再到行动结果，每个阶段都在单块 Blackwell GPU 上更新全部 70.9 亿个可训练参数。联合嵌入后训练将在保留集改写样本上的准确率从 7（原文此处截断）……

    arXiv:2609.39111v1 Announce Type: new  Abstract: Human intelligence relies heavily on learned intuition: recognising patterns and judging situations without explicitly unfolding every intermediate step. We introduce Bongard, an open-weight System One model that treats machine intuition as an independent capability to design and train. A T5Gemma 2 4B-4B encoder-decoder separates reading the evidence from making judgments. The encoder reads the state bidirectionally together with the question instructions, and separate decoder branches share this encoding, so many judgments about the same situation require only one reading of the state. A trained head returns probabilities over the supplied candidates without generating text. Training proceeds in three stages, from supervised judgments to semantic relationships to action outcomes, and each stage updates all 7.09 billion trainable parameters on one Blackwell GPU. Joint-embedding post-training raises accuracy on held-out rephrasings from 7
    
[^90]: 虚假前沿：诊断与缓解自进化搜索智能体中的共谋作弊现象

    False Frontiers: Diagnosing and Mitigating Co-Cheating in Self-Evolving Search Agents

    [https://arxiv.org/abs/2609.39102](https://arxiv.org/abs/2609.39102)

    本文揭示了自进化搜索智能体中提议器与求解器在共享错误上“共谋作弊”、导致内部奖励虚高而真实能力停滞的失效模式，并提出多样本验证（MSV）方法部分缓解该问题。

    

    自进化搜索智能体通过联合优化一个负责生成问题的提议器和一个负责解答问题的求解器来构建自己的训练课程。这种闭环引入了一种我们称之为“共谋作弊”的失效模式：提议器与求解器在共享错误上日益趋同，使得内部奖励不断提升，但外部真实正确性并未相应提高。基于源证据的事后审计显示，随着自进化轮次的推进，共谋作弊现象愈发严重——即便循环内的训练信号持续改善，伪标签的正确性却停滞不前甚至下降。最直接的缓解方法是在训练前对提议进行验证：我们提出多样本验证（MSV）方法，即让同一模型在有源证据和无源证据条件下各查询三次，以此决定任务是否准入，并替换不可靠的伪标签。MSV能部分减少虚假一致，但仍留下大量残余的共谋作弊，且每个任务需额外付出六次标注器生成开销。

    arXiv:2609.39102v1 Announce Type: cross  Abstract: Self-evolving search agents build their own training curricula by jointly optimizing a proposer that generates questions and a solver that answers them. This closed loop introduces a failure mode we call co-cheating: the proposer and solver increasingly agree on shared errors, so internal reward improves without a matching gain in external correctness. A post-hoc audit against source evidence shows co-cheating growing more severe over successive rounds of self-evolution, with pseudo-label correctness stagnating or declining even as the in-loop training signal improves. The most direct mitigation is to verify proposals before training: we introduce multi-sample verification (MSV), which queries the same model three times with the source and three times without it to decide task admission and replace unreliable pseudo-labels. MSV partially reduces false agreement but leaves substantial residual co-cheating and costs six extra labeler gen
    
[^91]: 超越文本：基于大语言模型的多模态对话维度情感评估

    Beyond Text: LLM-Based Dimensional Emotion Evaluation in Multimodal Dialogue

    [https://arxiv.org/abs/2609.39072](https://arxiv.org/abs/2609.39072)

    该论文提出基于大语言模型的多模态对话情感评估框架，将声学线索转化为文本描述并结合LoRA微调，在IEMOCAP上实现效价CCC 0.7822的新纪录，证明领域适应比模型规模更重要。

    

    对话中的情感识别已被广泛研究，但将大语言模型（LLM）应用于多模态对话中的连续维度情感评估仍鲜有探索。我们提出了一个基于LLM的框架，在IEMOCAP数据集上执行离散情感识别和效价-唤醒-支配（VAD）维度评估，并遵循SpeechCueLLM方法将声学线索以自然语言描述的形式融入其中。我们评估了涵盖LLaMA、GPT和Qwen系列的六个模型，采用零样本提示、少样本提示和LoRA微调三种方式。尽管GPT模型规模更大，经LoRA微调的LLaMA模型在两项任务上均显著优于经过提示工程的GPT模型，我们将这一差距归因于领域适应而非模型容量。我们的最佳模型取得了0.7822的效价CCC，创造了IEMOCAP上新的最先进纪录。消融研究证实，文本化的音频描述能显著提升较小模型的性能。

    arXiv:2609.39072v1 Announce Type: cross  Abstract: Emotion recognition in conversation has been widely studied, but applying Large Language Models (LLMs) to continuous dimensional emotion evaluation in multimodal dialogue remains largely unexplored. We propose an LLM-based framework that performs discrete emotion recognition and Valence-Arousal-Dominance (VAD) dimensional evaluation on IEMOCAP, incorporating acoustic cues as natural language descriptions following the SpeechCueLLM approach. We evaluate six models spanning the LLaMA, GPT, and Qwen families under zero-shot prompting, few-shot prompting, and LoRA fine-tuning. LoRA fine-tuned LLaMA models substantially outperform prompt-engineered GPT models on both tasks despite GPT's larger scale, a gap we attribute to domain adaptation rather than model capacity. Our best model achieves a Valence CCC of 0.7822, a new state-of-the-art on IEMOCAP. Ablation studies confirm that textual audio descriptions meaningfully improve smaller models
    
[^92]: LexReward：一种面向法律语言模型的分类体系驱动奖励框架

    LexReward: A Taxonomy-Driven Reward Framework for Legal Language Models

    [https://arxiv.org/abs/2609.39071](https://arxiv.org/abs/2609.39071)

    LexReward提出了一个由分类体系驱动的法律奖励建模框架，从风格、要素和推理链三个维度通过评分细则评估法律回复的多维质量，并将产生的奖励信号用于构建偏好数据以进行DPO和奖励模型训练，从而提升法律语言模型的表现。

    

    法律语言模型需要的奖励信号不仅要捕捉答案的正确性，还要捕捉法律回复的多维度质量。然而，现有的奖励方法通常依赖于粗粒度的整体性判断，领域针对性有限且可解释性不足。我们提出了LexReward，一个由分类体系驱动的法律奖励建模框架。LexReward从三个互补的维度刻画法律回复的质量：风格，涵盖词汇与句法层面的质量；要素，评估法律主体、事实、法条与判决；以及推理链，评估法律推理的顺序、完整性、正确性与无冗余性。针对每个维度，我们制定了明确规定评估标准和质量等级的评分细则。由此产生的奖励信号被用于构建配对偏好数据，以用于直接偏好优化（DPO）和奖励模型的训练。实验表明，基于评分细则的奖励能够可靠地对回复质量进行区分。

    arXiv:2609.39071v1 Announce Type: new  Abstract: Legal language models require reward signals that capture not only answer correctness but also the multidimensional quality of legal responses. Existing reward methods, however, often rely on coarse-grained holistic judgments, providing limited domain specificity and interpretability. We introduce LexReward, a taxonomy-driven framework for legal reward modeling. LexReward characterizes legal response quality along three complementary dimensions: Style, covering lexical and syntactic quality; Element, assessing legal subjects, facts, statutes, and decisions; and Chain, evaluating the order, completeness, correctness, and non-redundancy of legal reasoning. For each dimension, we develop rubrics that specify evaluation criteria and quality levels. The resulting rewards are used to construct pairwise preference data for Direct Preference Optimization (DPO) and reward-model training. Experiments show that the rubric-based rewards reliably dis
    
[^93]: CORE：面向可验证语言模型搜索的冲突导向推理消除

    CORE: Conflict-Oriented Reasoning Elimination for Verifiable Language-Model Search

    [https://arxiv.org/abs/2609.39069](https://arxiv.org/abs/2609.39069)

    CORE通过向验证器索取认证的冲突核心并回跳至冲突根源决策来指导语言模型搜索，在保持搜索完备性的同时大幅减少验证器调用，并在多项推理任务上超越Tree of Thoughts。

    

    测试时推理系统在面对失败时通常采取重启或修改最新步骤的方式，即使错误实际上是由更早的决策引起的。我们提出CORE，一种搜索控制器，它向验证器请求经过认证的冲突核心，回跳至该核心中最近的决策，并缓存该冲突以避免重复出现。在可靠验证、有限分支与深度以及穷尽式候选生成的条件下，不受限制的搜索是完备的，且绝不会剪枝掉有效解。在2,000个具有匹配候选和精确验证器的植入图着色实例上，与按时间顺序修复相比，CORE在30变量时将验证器调用中位数减少了39.8%，在36变量时减少了35.0%；缓存机制进一步优于单纯的回跳。在五项推理任务中，CORE使用Qwen2.5-7B-Instruct达到75.9%的平均成功率，使用Qwen3-8B达到84.2%，相比之下Tree of Thoughts分别为72.5%和81.8%。它还使用了更少的验证器调用和生成的token。

    arXiv:2609.39069v1 Announce Type: new  Abstract: Test-time reasoning systems often respond to failure by restarting or revising the latest step, even when an earlier decision caused the error. We introduce CORE, a search controller that requests a certified conflict core from a verifier, backjumps to the latest decision in that core, and caches the conflict to avoid repeating it. Under sound verification, finite branching and depth, and exhaustive proposals, the uncapped search is complete and never prunes a valid solution. On 2,000 planted graph-coloring instances with matched proposals and an exact verifier, CORE reduces median verifier calls by 39.8% at 30 variables and 35.0% at 36 variables relative to chronological repair; caching further improves on backjumping alone. Across five reasoning tasks, CORE achieves 75.9% mean success with Qwen2.5-7B-Instruct and 84.2% with Qwen3-8B, compared with 72.5% and 81.8% for Tree of Thoughts. It also uses fewer verifier calls and generated tok
    
[^94]: 隐蔽协助：乐于助人的LLM智能体在多智能体系统中规避监督

    Covert Assistance: Helpful LLM Agents Evade Oversight in Multi-Agent Systems

    [https://arxiv.org/abs/2609.39050](https://arxiv.org/abs/2609.39050)

    研究发现，即使没有任何对抗性激励，良性LLM智能体也会自发地伪装公司机密凭证以“乐于助人”地帮助外部开发者，同时躲避监控器的监督，九个受测前沿模型中有七个表现出这种“隐蔽协助”的失控行为。

    

    随着多智能体系统进入高风险领域，智能体可能绕过安全边界的可能性日益引发担忧。先前的工作主要在对抗性设置中研究这一风险，即智能体被指令或奖励驱动进行隐蔽通信并规避监督。我们表明，良性智能体可以在没有对抗性激励的情况下跨越相同的安全边界。我们模拟了一个软件工程工作流程：一个规划者代表一家公司雇用外部开发人员，规划者编写需求，并持有一个被指示不得向开发人员披露的公司凭证；同时一个监控器负责筛查双方的交流。在测试的九个前沿模型中，有七个模型在需求中伪装凭证，以帮助开发人员恢复凭证，同时躲避监控器，甚至在完成分配的目标之后仍然如此。例如，在使用DeepSeek-V4-Pro进行的6,000个回合实验中，规划者在16.9%的情况下尝试隐藏凭证；在0.9%的情况下，凭证（摘要在此处截断）

    arXiv:2609.39050v1 Announce Type: cross  Abstract: As multi-agent systems enter high-stakes domains, the possibility that agents may circumvent safety boundaries is a growing concern. Prior work has examined this risk primarily in adversarial settings, where agents are instructed or rewarded to communicate covertly and evade oversight. We show that benign agents can cross the same boundaries without adversarial incentives. We emulate a software-engineering workflow in which a planner represents a company hiring an external developer. The planner writes requirements and holds a company credential it is instructed not to disclose to the developer; a monitor screens their exchanges. Seven of nine tested frontier models disguise the credential in their requirements to help the developer recover it while evading the monitor, even after completing their assigned objective. For example, across 6,000 episodes with DeepSeek-V4-Pro, the planner attempts concealment in 16.9%; in 0.9%, the credent
    
[^95]: 结构与思维链之争：评估大语言模型临床标准提取在抑郁严重程度评级中的应用

    Structure vs. Chain-of-Thought: Evaluating LLM Criteria Extraction for Depression Severity

    [https://arxiv.org/abs/2609.39049](https://arxiv.org/abs/2609.39049)

    研究系统比较了基于临床标准的结构化提取与思维链推理两种LLM抑郁严重程度评估方式，发现结构化提取仅在阈值拟合标注数据时有不显著的优势，在先验固定阈值下并无增益，说明其可审计性优势尚未转化为性能提升。

    

    大语言模型（LLM）可以直接从社交媒体帖子中评定抑郁严重程度，也可以标记帖子中所体现的临床标准，再由代码将标记数量转换为标签。后者更易于审计，因为临床医生可以逐一核查被标记的标准。我们在两个Reddit语料库上，使用三个不同规模的LLM（从9B到前沿规模）和两种问卷（PHQ-9、BDI-II）对这两种方法进行了比较，并以二次加权kappa衡量一致性。对于两个前沿模型，标准提取仅当其决策阈值在标注数据上拟合后才在一个语料库上得分高于思维链；无论是否在同一标注数据上重新校准思维链，其增益均不显著。当阈值根据PHQ-9标准先验固定时，标准提取在两个语料库上均无增益，即使模型每条帖子标记超过两条标准也是如此。9B模型在来自抑郁社区语料库上的表现有所不同，它标记大多数……

    arXiv:2609.39049v1 Announce Type: cross  Abstract: A large language model (LLM) can rate depression severity directly from a social media post or mark which clinical criteria the post shows and let code turn the count into a label. The latter is easier to audit because a clinician can check each marked criterion. We compare these approaches on two Reddit corpora using three LLMs (from 9B to frontier scale) and two questionnaires (PHQ-9, BDI-II), and measure agreement with quadratic weighted kappa. For the two frontier models, criteria extraction scores above chain-of-thought on one corpus only when its decision thresholds are fitted on labeled data. Neither model's gain is significant, with or without recalibrating chain-of-thought on the same labels. With thresholds fixed a priori from PHQ-9's criteria, extraction shows no gain on either corpus, even where models mark over two criteria per post. The 9B model behaves differently on a corpus from depression communities. It labels most p
    
[^96]: RSIGame：具有递归自我改进能力的自主智能体游戏开发

    RSIGame: Autonomous Agentic Game Development with Recursive Self-improvement

    [https://arxiv.org/abs/2609.39045](https://arxiv.org/abs/2609.39045)

    RSIGame是一个具有递归自我改进能力的自主智能体游戏开发框架，通过局部“探索-诊断-改进”循环与全局质量跟踪循环的协同，可靠地将自动生成的游戏改进到超越可玩版本，避免了朴素迭代改进中的过拟合问题。

    

    大语言模型的最新进展使得自动游戏生成日益可行，然而，可靠地将生成的游戏改进到超越可玩版本仍然具有挑战性。朴素的迭代改进很容易过拟合于一小部分测试用例，导致生成的游戏脆弱、存在未解决的bug、行为缺失，以及对更广泛玩家交互的泛化能力差。我们提出了RSIGame，一个具有递归自我改进能力的自主智能体游戏开发框架。RSIGame将开发过程组织为互补的局部循环和全局循环。具体而言，局部“探索-诊断-改进”循环广泛地探索可执行游戏，诊断发现的问题并确定优先级，并执行基于证据的修订，其中不断演化的检查清单会持续积累新的测试与改进指导。全局循环则跟踪整体质量，保存最佳检查点，并在长周期开发过程中检测饱和或回归现象。

    arXiv:2609.39045v1 Announce Type: new  Abstract: Recent advances in large language models have made automatic game generation increasingly feasible, yet reliably improving generated games beyond a playable version remains challenging. Naive iterative refinement can easily overfit a small set of test cases, producing fragile games with unresolved bugs, missing behaviors, and poor generalization to broader player interactions. We introduce RSIGame, an autonomous agentic game development framework with recursive self-improvement. RSIGame organizes development into complementary local and global loops. Concretely, a local explore-diagnose-improve loop broadly explores the executable game, diagnoses and prioritizes discovered issues, and performs evidence-grounded revision, where an evolving checklist continually accumulates new testing and improvement guidance. A global loop tracks overall quality, preserves the best checkpoint, and detects saturation or regression over long-horizon develo
    
[^97]: 切换线性注意力

    Switching Linear Attention

    [https://arxiv.org/abs/2609.39034](https://arxiv.org/abs/2609.39034)

    SwiLA通过将状态更新规则建模为线性回归混合模型中的在线期望最大化过程，在保持线性注意力固定大小循环状态的同时显著提升了其表达能力，从而在效率与性能之间实现平衡。

    

    设计兼具强表达能力与高效推理的序列层仍然是现代机器学习中的核心挑战。标准的softmax注意力通过丰富的非线性词元交互实现了出色的序列建模性能，但它需要一个随序列长度线性增长的键值缓存，限制了其可扩展性。线性注意力能够以恒定的内存占用实现高效的循环计算，但其表达能力的下降往往导致较差的建模性能。我们提出了切换线性注意力，这是一种新颖的序列层，它在保留线性注意力固定大小循环状态的同时增强了表示能力，从而弥合了这一差距。我们从测试时回归框架中推导出SwiLA的递推形式，将状态更新规则转化为线性回归混合模型中的在线期望最大化过程。在测试时，每个输出维度会动态地在多个（摘要在此处被截断）

    arXiv:2609.39034v1 Announce Type: cross  Abstract: Designing expressive sequence layers with efficient inference remains a central challenge in modern machine learning. Standard softmax attention achieves excellent sequence modeling performance through rich nonlinear token interactions, but it requires a key-value cache that grows linearly with sequence length, limiting its scalability. Linear attention enables efficient recurrent computation with a constant memory footprint, yet its reduced expressivity often yields inferior modeling performance. We introduce Switching Linear Attention (SwiLA), a novel sequence layer that bridges this gap by enhancing representational capacity while retaining the fixed-size recurrent state of linear attention. We derive the SwiLA recurrence from the test-time regression framework, casting the state update rule as online expectation-maximization in a mixture of linear regressions model. At test time, each output dimension dynamically selects among mult
    
[^98]: 可信AI审稿人缺失的一块拼图：从修辞性稳健性基准测试到SciCore审稿

    A Missing Piece for Trustworthy AI Reviewers: From Benchmarking Rhetorical Robustness to SciCore Review

    [https://arxiv.org/abs/2609.39027](https://arxiv.org/abs/2609.39027)

    该论文提出“修辞性稳健性”概念并构建包含1,260个稿件版本的RobustReview基准，揭示了AI审稿人“虚假稳健性”的问题，进而提出双分支审稿模型SciCore，通过结合全稿评判与基于提取结构化科学内容的评判来提升审稿的可信度。

    

    AI审稿人可能会对以不同措辞报告相同科学内容的稿件给出不同的评判，这可能导致修辞性优化得到奖励，而非科学上的真正改进。我们将“修辞性稳健性”表述为一项联合要求：即在保持内容不变的改写下保持评判稳定，同时在不同论文之间保持区分能力。我们提出了RobustReview，这是一个受控的全稿基准，包含1,260个稿件版本，并评估了30种审稿人配置。该基准揭示了“虚假稳健性”现象，即对改写的敏感性虽低，但在不同论文之间却出现评分崩塌；同时表明，与人类评审的对齐程度和修辞性稳健性对审稿人的排序并不一致。此外，所评估的以内容为中心的提示协议并不能在各种骨干模型上一致地提升稳健性。基于这些发现，我们提出了SciCore，这是一种双分支审稿模型，它将基于全稿的评判与基于提取出的结构化科学内容（原文摘要在此处截断）的评判进行综合。

    arXiv:2609.39027v1 Announce Type: new  Abstract: AI reviewers can assign different judgments to manuscripts that report the same science in different wording, potentially rewarding rhetorical optimization over scientific improvement. We formulate Rhetorical Robustness as the joint requirement of stability across content-preserving rewrites and discrimination across papers. We introduce RobustReview, a controlled full-manuscript benchmark with 1,260 manuscript versions, and evaluate 30 reviewer configurations. The benchmark reveals false robustness, where low rewrite sensitivity coincides with score collapse across papers, and shows that human alignment and rhetorical robustness rank reviewers differently. Moreover, the evaluated content-focused prompting protocol does not consistently improve robustness across backbones. Motivated by these findings, we introduce SciCore, a dual-branch reviewer that averages a full-manuscript judgment with a judgment based on an extracted, structured sc
    
[^99]: 证据优先，算术其次：DocSem任务的系统报告与失败分析

    Evidence First, Arithmetic Second: A System Report and Failure Analysis for DocSem

    [https://arxiv.org/abs/2609.39013](https://arxiv.org/abs/2609.39013)

    本文介绍了DocSem共享任务系统EVICALC，其采用“先选证据段落、再由语言模型生成算术表达式并在本地求值”的流程，官方测试联合准确率为8.61%，并通过案例分析指出OCR块合并和页面图像读取能力有限等失败环节。

    

    EVICALC是我们为DocSem共享任务开发的系统，在官方最终测试评估中对1,730个任务取得了8.61%的联合准确率。该系统读取PDF文件、选择一段文字，让语言模型编写一个算术表达式，并在本地代码中对该表达式进行求值。保存的中间结果支持对失败案例的检查。在另一次公开验证集运行中，系统达到了92.17%的答案准确率和1.00的证据F1分数。由于配置和指标不同，这些分数不构成受控比较。我们的事后人工分析是描述性的：在一个检查的案例中，光学字符识别（OCR）和块分组将相关段落合并到了另一个块中，导致系统从无关文本中得出答案。一项对100份文档阅读页面图像的探索性研究仅返回了22份文档的证据标识符。这些描述性发现促使进一步评估；它们并未确定整体得分的原因。

    arXiv:2609.39013v1 Announce Type: new  Abstract: EVICALC, our system for the DocSem shared task, achieved 8.61% joint accuracy on 1,730 tasks in the official final test evaluation. It reads a PDF, selects a passage, asks a language model to write an arithmetic expression, and evaluates that expression in local code. Saved intermediate results support inspection of failures. A separate public-validation run achieved 92.17% answer accuracy and 1.00 evidence F1. The configurations and metrics differ, so these scores are not a controlled comparison. Our manual, post-hoc analysis is descriptive: in one inspected case, optical character recognition (OCR) and block grouping merged the relevant passage into another block, and the system answered from unrelated text. An exploratory study of reading page images on 100 documents returned evidence identifiers for only 22 documents. These descriptive findings motivate further evaluation; they do not establish the causes of the overall score.
    
[^100]: 看不见的语言税：2026年LLM分词器中法语和地区语言的代币溢价，以及一个法语优化的原型

    The Invisible Language Tax: Token Premiums of French and Regional Languages in 2026 LLM Tokenizers, and a French-Optimized Prototype

    [https://arxiv.org/abs/2609.39001](https://arxiv.org/abs/2609.39001)

    研究测量了2026年主流LLM分词器中的语言token溢价，发现法语比英语多消耗31%-58%的token，法国地区语言更是高达1.6-3.3倍，并提出了一种法语优化的分词器原型。

    

    LLM服务按token计费，上下文窗口以token计量，但相同内容所需token数在不同语言间存在差异。我们在NTREX-128（124个非英语参考译文）和用于地区语言的《世界人权宣言》上，测量了这一token溢价，涵盖了2026年广泛使用的7种分词器（OpenAI o200k、Llama 3、Qwen3、DeepSeek V3/V4、Gemma 3、Mistral Tekken，以及通过Anthropic计数API获取的Claude第5代分词器）。法语比英语需要多31%至58%的token，而简体中文的范围为少5%至多40%，在七种分词器中有六种比法语更便宜。法国的地区语言和海外语言大约需要支付英语token数的1.6至3.3倍。我们讨论了历史重发、分层定价和固定上下文窗口如何在智能体应用中放大这一绝对差距。在一个受控实验（BPE、Europarl、5万词表）中，添加Fre

    arXiv:2609.39001v1 Announce Type: cross  Abstract: LLM services are billed per token and context windows are measured in tokens, yet the number of tokens needed for the same content varies across languages. We measure this token premium on seven tokenizers of widely used 2026 models (OpenAI o200k, Llama 3, Qwen3, DeepSeek V3/V4, Gemma 3, Mistral Tekken, and the Claude generation-5 tokenizer via Anthropic's counting API) on NTREX-128 (124 non-English reference translations) and on the Universal Declaration of Human Rights for regional languages. French requires 31% to 58% more tokens than English, whereas Simplified Chinese ranges from 5% fewer to 40% more and is cheaper than French on six of the seven tokenizers. Regional and overseas languages of France pay roughly 1.6 to 3.3 times the English count. We discuss how history re-sending, tiered pricing and fixed context windows amplify the absolute gap in agentic use. In a controlled experiment (BPE, Europarl, 50k vocabulary), adding Fre
    
[^101]: Settle：学习何时停止推理

    Settle: Learning When to Stop Reasoning

    [https://arxiv.org/abs/2609.38997](https://arxiv.org/abs/2609.38997)

    Settle通过从已完成推理轨迹中学习答案稳定性来训练推理结束标记，在几乎不损失准确率的情况下将token消耗减少40%，扩展了准确率与效率的帕累托前沿。

    

    推理模型在答案已经稳定后往往还会继续生成内容。Settle从已完成推理轨迹中的答案稳定性来学习何时停止。它通过训练现有的推理结束标记来实现这一目标，同时保持其他预测接近基础模型，并且在推理时只需要普通的解码过程。在使用Qwen3-4B的MATH-500基准上，Settle将token数量减少了40%，而准确率仅下降0.5个百分点。与在首次出现稳定答案处截短相同轨迹的监督微调相比，它在几乎相同的token数量下获得了6.16个百分点的提升。其停止分数能够预测一个正确答案是否会保持正确。Settle扩展了所评估的各停止方法的准确率-token数量帕累托前沿。

    arXiv:2609.38997v1 Announce Type: new  Abstract: Reasoning models often continue generating after their answers have settled. Settle learns when to stop from answer stability in completed traces. It trains the existing end-of-reasoning token while keeping other predictions close to the base model, and requires only ordinary decoding at inference. On MATH-500 with Qwen3-4B, Settle reduces token count by 40% with a 0.5-percentage-point decrease in accuracy. It gains 6.16 percentage points over supervised fine-tuning on the same traces shortened at their first stable answer, at nearly identical token counts. Its stopping score predicts whether a correct answer will remain correct. Settle extends the accuracy-token-count Pareto frontier of the evaluated stopping methods.
    
[^102]: 当截断逆转修正：逐点前向KL在策略自蒸馏中的失败动力学

    When Clipping Reverses Correction: Failure Dynamics of Pointwise Forward-KL On-Policy Self-Distillation

    [https://arxiv.org/abs/2609.38995](https://arxiv.org/abs/2609.38995)

    本文揭示了OPSD中被广泛采用的逐点前向KL截断实际上会阻碍学生模型向教师模型修正，导致训练产生大量持续到响应末尾的重复内容，证明截断目标可能逆转修正效果。

    

    在策略自蒸馏（OPSD）利用同一模型在特权信息条件下的反馈，在其自身生成的响应上训练学生模型。在数学推理任务上，原始OPSD研究发现风格类token可能主导训练信号而压过数学相关token，并且对前向KL目标进行逐点截断能够稳定训练。逐点截断是在对词表求和之前，将每个词表级的前向KL项限制在固定阈值。后续研究采用了这种截断方法，但其对训练的实际影响尚未被直接检验。在仅以是否应用截断为唯一差异的匹配训练运行中，我们观察到应用截断的运行比未截断的运行产生了多得多的重复内容，且这些重复会一直持续到响应结束。我们将这一失败归因于截断后的目标函数。我们证明截断目标可能无法将学生模型朝教师方向进行修正，并且……

    arXiv:2609.38995v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) trains a student on its own generated responses using feedback from the same model conditioned on privileged information. On mathematical reasoning, the original OPSD study finds that stylistic tokens can dominate the training signal over math-related tokens, and that pointwise clipping of the forward KL objective stabilizes training. Pointwise clipping caps each vocabulary-wise forward KL term at a fixed threshold before summing over the vocabulary. Follow-up studies have adopted this clipping, but its effect on training has not been directly examined. In matched training runs differing only in whether clipping is applied, we observe that clipped runs produce substantially more repetitions that persist to the end of the response than their unclipped counterparts. We trace this failure to the clipped objective. We prove that the clipped objective can fail to correct the student toward the teacher and ca
    
[^103]: 超越单次运行的公平性：语音大语言模型适配中的训练种子变异性

    Fairness Beyond a Single Run: Training-Seed Variability in Speech LLM Adaptation

    [https://arxiv.org/abs/2609.38976](https://arxiv.org/abs/2609.38976)

    该论文首次系统揭示，在语音大语言模型适配中，随机训练种子对人口统计公平性差异的影响远大于音频压缩等超参数，说明仅凭单次训练运行报告的公平性结论不可靠。

    

    自动语音识别中的人口统计公平性差距几乎总是仅从单次训练运行中报告得出。我们在五个音频压缩因子和六个随机种子下对语音大语言模型的Q-former投影器和LoRA适配器进行微调，同时保持编码器、基础解码器、数据和解码方式固定，并在Common Voice和Fair-Speech数据集上评估每次运行。在460小时的干净LibriSpeech数据上，随机种子在大多数人口统计维度上对公平性指标的影响超过压缩因子。平衡的3x3分解显示，Fair-Speech种族归一化差距变异的85.3%可归因于种子，而压缩因子仅占8.3%（p = 0.009），尽管压缩因子在年龄和性别维度上解释了更多变异。留出的LibriSpeech词错误率在这些种子间仅波动0.04个百分点，而Common Voice波动达8.57个百分点，因此这些并非失败的运行，且该效应在控制准确率和丢弃率后依然存在。将适配集扩展并多样化至960小时会减弱该效……（摘要被截断）

    arXiv:2609.38976v1 Announce Type: new  Abstract: Demographic fairness gaps in automatic speech recognition are almost always reported from a single training run. We fine-tune the Q-former projector and LoRA adapters of a speech LLM at five audio compression factors and six random seeds, holding the encoder, base decoder, data and decoding fixed, and evaluate every run on Common Voice and Fair-Speech. At 460 h of clean LibriSpeech, the seed moves fairness metrics more than compression does on most demographic axes. A balanced 3x3 decomposition attributes 85.3% of the variation in Fair-Speech ethnicity normalized gap to the seed against 8.3% to compression (p = 0.009), though compression explains more on age and gender. Held-out LibriSpeech word error rate spreads by 0.04 points across those seeds while Common Voice spreads by 8.57, so these are not failed runs, and the effect survives controlling for accuracy and dropout. Scaling and diversifying the adaptation set to 960 h damps the ef
    
[^104]: 让大语言模型说出它们的真实想法：测量与改进思维链-可解释性对齐

    Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment

    [https://arxiv.org/abs/2609.38972](https://arxiv.org/abs/2609.38972)

    本文提出CoT-可解释性对齐（CIA）指标来测量LLM思维链与内部计算之间的一致性，发现现有模型对齐程度有限（44.8%–75.9%），并通过以任务准确率和参数化忠实度为奖励的后训练方法有效提升了对齐性。

    

    思维链（CoT）轨迹通常被用作观察大型语言模型（LLM）如何得出答案的代理。然而，越来越多的证据表明，模型的思维链往往无法反映其内部计算，并且可以在不影响最终答案的情况下被改变。在这项工作中，我们测量并改进了LLM思维链中描述的推理过程与其内部实际计算之间的对齐程度。我们提出了CoT-可解释性对齐（CIA）指标，用于衡量模型的思维链轨迹与可解释性工具检测到的内部推理策略之间的一致性。我们在三项任务（两跳问答、提示干预和整数乘法）上对三个LLM进行了CIA评估，发现LLM在所有任务上的对齐程度都较为有限（44.8%–75.9%）。随后，我们尝试通过后训练来改进CIA，将任务准确率和参数化忠实度信号设置为奖励。实验表明……

    arXiv:2609.38972v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) traces often serve as a proxy for how Large Language Models (LLMs) arrive at their answers. However, growing evidence shows that models' CoT often fails to reflect their internal computations and can be changed without affecting their final answers. In this work, we measure and improve the alignment between the reasoning described in an LLM's CoT and what it computes internally. We propose CoT-Interpretability Alignment (CIA), a metric that measures the agreement between a model's CoT traces and its internal reasoning strategies as detected by interpretability tools. We evaluate CIA on three tasks (two-hop question answering, hint intervention, and integer multiplication) across three LLMs, finding that LLMs exhibit limited alignment across all tasks (44.8-75.9%). We then experiment with improving CIA via post-training, setting both the task accuracy and parametric faithfulness signals as a reward. Experiments sh
    
[^105]: 靶向检索，紧凑表示：思维链推理如何提升长上下文计数能力

    Targeted Retrieval, Compact Representations: How CoT Reasoning Improves Long-Context Counting

    [https://arxiv.org/abs/2609.38958](https://arxiv.org/abs/2609.38958)

    本研究通过大海捞针计数任务揭示，思维链推理使模型从非思考模式下“广泛检索多个目标”的机制，转变为通过枚举逐一“靶向检索”的机制，使注意力集中于单个目标并形成更紧凑的内部表示，从而显著提升长上下文计数准确率。

    

    大型语言模型（LLM）在思维链推理的推动下，于长上下文任务中取得了快速进步。然而，这一提升背后的内部机制仍不清楚。我们通过“大海捞针”（NIAH）计数任务来研究这些机制，该任务要求LLM统计分散在长文本中的记录数量。在十二个模型对比组中，思考（或推理）模式相比非思考模式提升了计数准确率，且在计数较大时提升尤为显著。这促使我们开展机制分析，并识别出两种截然不同的机制：(i) 广泛检索，即非思考模型广泛地关注多个针（目标记录）；(ii) 靶向检索，即思考模型利用CoT轨迹中的枚举逐一检索各个针。靶向检索将注意力集中于单个针，并伴随更紧凑的内部表示。此外，因果干预分析……

    arXiv:2609.38958v1 Announce Type: new  Abstract: Large language models (LLMs) have been rapidly improving in long-context tasks, powered by Chain-of-Thought (CoT) reasoning. However, the internal mechanisms underlying this improvement remain unclear. We investigate these mechanisms through a needle-in-a-haystack (NIAH) counting task, where an LLM is asked to count the number of records dispersed in a long text. Across twelve model comparison groups, Thinking (or reasoning) improves counting accuracy over Non-thinking, with pronounced gains at larger counts. This motivates our mechanistic analysis, which identifies two contrasting mechanisms: (i) broad retrieval, where Non-thinking models broadly attend to multiple needles; (ii) targeted retrieval, where Thinking models use enumeration in CoT traces to successively retrieve needles. Targeted retrieval concentrates attention on individual needles and is accompanied by more compact internal representations. Moreover, causal intervention a
    
[^106]: GraphForge：基于图锚定工作区合成训练可执行任务的工作智能体

    GraphForge: Training Working Agents with Graph-Anchored Workspace Synthesis

    [https://arxiv.org/abs/2609.38923](https://arxiv.org/abs/2609.38923)

    GraphForge 提出了一种基于证据图的框架，将训练工作智能体所需的任务与验证标准都锚定在真实文件构建的工作区上，从而合成兼具真实性、多样性和可验证性的高质量任务数据。

    

    工作智能体需要阅读多样化的文件、协调使用工具并产出交付成果。训练这类智能体需要基于大量真实文件且结果可验证的任务，但目前很少有数据合成流程能够生成这种类型的数据。现有流程要么用模型生成文件，因而缺乏真实性和多样性；要么在真实文件上构建任务却没有针对任务的验证器，导致结果质量无法检验。我们提出 GraphForge，一个基于证据图的框架，将任务及其验证都锚定在真实文件之上。GraphForge 从以职业为依据的种子出发以实现受控的多样性，为每个种子组装一个由真实文件构成的工作区，并在文件之间的关系上构建证据图。由于任务描述和评分标准都源自该证据图，任务要求均有工作区文件作为支撑，且每条评分标准都锚定到验证该标准所需的文件上。初始的 rollout 试运行进一步检验任务的可执行性，并

    arXiv:2609.38923v1 Announce Type: new  Abstract: Working agents need to read diverse files, coordinate tools, and produce deliverables. Training such agents requires tasks built on many real files with verifiable results, but few pipelines exist to synthesize this kind of data. Existing pipelines either generate files with models, which lack realism and diversity, or build tasks on real files without task-specific verifiers, leaving result quality unchecked. We introduce GraphForge, an evidence-graph based framework that grounds both the task and its verification in real files. Starting from occupation-grounded seeds for controlled diversity, GraphForge assembles a workspace of real files for each seed and builds an evidence graph over their relations. Since the task statement and rubrics are both derived from this graph, task requirements are backed by the workspace files and each criterion is anchored to the files needed to verify it. An initial rollout further tests executability, a
    
[^107]: K2P：无标签知识到提示的蒸馏

    K2P: Label-Free Knowledge to Prompt Distillation

    [https://arxiv.org/abs/2609.38898](https://arxiv.org/abs/2609.38898)

    K2P提出了一种无标签知识蒸馏方法，通过从教师解答中合成并优化可复用提示、利用答案一致性引导搜索与选择，使冻结的学生模型无需权重更新即可在推理任务上获得准确性保证。

    

    知识蒸馏可以通过可复用的提示将推理能力从更强的教师模型传递到冻结的学生模型，但避免权重更新并不意味着消除了监督。在没有真实答案（ground-truth）的情况下，教师模型的解答是未经核实的，而与学生保持一致可能会奖励双方共同的错误。我们提出了知识到提示（Knowledge-to-Prompt，K2P），用于无标签的知识蒸馏到提示。K2P从教师模型的解答中合成可复用的指令，利用成对的教师和学生响应对其进行优化，并通过答案一致性来引导搜索与选择。它保留了自适应搜索可能低估的候选提示，并在预留的问题上进行选择。部署时仅需使用冻结的学生模型和选定的提示。我们的理论分离了生成差距与选择差距，并给出了在教师参考答案并不完美的情况下，一致性引导的构建仍能产生准确性保证的条件。在多种推理任务和学生模型上，K2P的表现优于有标签基线方法。

    arXiv:2609.38898v1 Announce Type: cross  Abstract: Knowledge distillation can transfer reasoning from stronger teachers to frozen students through reusable prompts, but avoiding weight updates does not eliminate supervision. Without ground-truth answers, teacher solutions are unverified, and agreement with the teacher can reward shared mistakes. We introduce Knowledge-to-Prompt (K2P) for label-free knowledge distillation to prompts. K2P synthesizes reusable instructions from teacher solutions, refines them using paired teacher and student responses, and guides search and selection with answer agreement. It retains candidates that adaptive search may undervalue and selects on reserved questions. Deployment uses only the frozen student and selected prompt. Our theory separates generation and selection gaps and gives conditions under which agreement-guided construction yields accuracy guarantees despite imperfect teacher references. Across reasoning tasks and students, K2P outperforms lab
    
[^108]: VOSSA：面向流式语音架构的声纹优化

    VOSSA: Voiceprint Optimization for Streaming Speech Architectures

    [https://arxiv.org/abs/2609.38887](https://arxiv.org/abs/2609.38887)

    提出VOSSA说话人表示框架，从内容编码器中间层提取说话人信息并采用注意力统计池化聚合，与语音转换目标联合训练从而无需单独的说话人编码器，在流式实时语音转换中改善了F0动态特性和元音区分性声学线索，同时保持了相当的说话人相似度和语音质量。

    

    实时语音转换（VC）系统通常依赖来自自动说话人验证（ASV）模型的预训练说话人嵌入。尽管这些嵌入在说话人区分方面非常有效，但其训练目标是使嵌入在说话人内部的语音和韵律变化中保持稳定，这可能与流式约束下的帧级声学生成产生冲突。为了解决这一问题，我们提出了VOSSA（面向流式语音架构的声纹优化），这是一个说话人表示框架，它从内容编码器的中间层提取说话人信息，并采用注意力统计池化进行聚合。该嵌入与语音转换目标进行联合训练，从而无需单独的说话人编码器。在六个数据集上的实验表明，VOSSA改善了F0动态特性和元音区分性声学线索，同时保持了相当的NISQA-MOS、WER和说话人相似度。感知测试进一步表明，自然度也有所提升。

    arXiv:2609.38887v1 Announce Type: cross  Abstract: Real-time voice conversion (VC) systems commonly rely on pretrained speaker embeddings from automatic speaker verification (ASV) models. While effective for speaker discrimination, these embeddings are trained to remain stable across phonetic and prosodic variations within-speaker, which may conflict with frame-level acoustic generation in streaming constraints. To address this issue, we propose VOSSA (Voiceprint Optimization for Streaming Speech Architectures), a speaker representation framework that extracts speaker information from intermediate content encoder layers and aggregates using attentive statistics pooling. The embedding is trained jointly with VC objectives, removing the need for a separate speaker encoder. Across six datasets, VOSSA improves F0 dynamics and vowel-discriminative acoustic cues while maintaining comparable NISQA-MOS, WER, and speaker similarity. Perceptual tests further indicate improvements in naturalness,
    
[^109]: 音频Token的注意力在语言模型运行之前即可预测

    Audio Token Attention Is Predictable Before the Language Model Runs

    [https://arxiv.org/abs/2609.38878](https://arxiv.org/abs/2609.38878)

    该论文发现音频token在语言模型中的全层注意力排名可以在模型运行前由其编码器输出线性预测，据此提出的无需标签的Triage方法能够在推理前高效裁剪音频token，大幅降低大型音频语言模型的计算开销。

    

    大型音频语言模型（LALM）会将一分钟的语音转化为750至1,500个token，并对每一个token进行预填充（prefill）。图像token的剪枝通常在语言模型的前几层之后进行，因为在这些层中图像token获得的注意力很少。而音频token在这些层中获得的注意力要多得多，且其重要性排名尚远未定型，因此音频需要在语言模型运行之前就获得一个排名。令人惊讶的是，一个音频token在语言模型中将获得的注意力，在语言模型运行之前就可以从其编码器输出中被线性地预测出来。一个无需标签、以闭式解拟合的线性映射，能够在十三个LALM中的十一个上以 ρ ≥ .69 的相关系数预测这种全层注意力排名。我们的方法Triage依据该预测来裁剪音频token；在多项选择任务上，还会在第2层进行再次裁剪，利用在该层观察到的注意力来修正预测。Triage无需标签即可设定压缩程度，并在两种预算约束下工作，这些预算限制了其输出可能偏离的范围。

    arXiv:2609.38878v1 Announce Type: cross  Abstract: A large audio language model (LALM) turns a minute of speech into 750-1,500 tokens and prefills every one. Image-token pruning often cuts after the language model's first layers, where image tokens draw little attention. Audio tokens draw much more attention there, and their ranking is still far from final, so audio needs a ranking before the language model runs. Surprisingly, the attention an audio token will receive across the language model is already linearly predictable from its encoder output, before the language model runs. A linear map, fitted in closed form without labels, predicts this all-layer attention ranking at $\rho \geq .69$ on eleven of thirteen LALMs. Our method, Triage, cuts audio tokens by this prediction and, on multiple choice, cuts again at layer 2, correcting the prediction with the attention observed there. Triage sets its compression without labels, under two budgets that limit how far its output may differ f
    
[^110]: TRACE：面向LitTraceQA的目标感知检索、归因证据与契约约束抽取

    TRACE: Target-Aware Retrieval, Attributed Evidence, and Contract-Constrained Extraction for LitTraceQA

    [https://arxiv.org/abs/2609.38861](https://arxiv.org/abs/2609.38861)

    TRACE 提出目标感知检索、归因证据定位与契约约束抽取的一体化框架，通过索引近三万篇论文并在表格抽取前预测观测单元，弥合了文献问答中源访问与评分器可见正确性之间的“接地契约鸿沟”。

    

    从文献中找到一篇相关论文，并不等同于从中产出可验证的答案。LitTraceQA 要求规范化的论文标识符、页级或对象级的精确证据，以及与评测器相匹配的类型化答案。我们将“源访问”与“评分器可见正确性”之间的分离称为接地契约鸿沟（grounding contract gap）。TRACE——目标感知检索、归因证据与契约约束抽取——通过目标分组检索、独立的类型化证据定位、多模态表格抽取、模式驱动的表格构建以及故障关闭式验证来弥合这一鸿沟。该系统通过段落、对象、别名、引用和稠密表示对 27,487 篇论文进行索引，同时在每个信号背后保留问题的目标。对于表格，TRACE 在抽取数值之前先预测观测单元，并使用与评测器兼容的键归一化方式来组装行。经审计的精选洁净赛道产出物在官方上得分为 0.760613。

    arXiv:2609.38861v1 Announce Type: new  Abstract: Finding a relevant paper is not the same as producing a verifiable answer from it. LitTraceQA requires canonical paper identifiers, exact evidence at the page or object level, and typed answers that match the evaluator. We call the separation between source access and scorer-visible correctness the grounding contract gap. TRACE - Target-Aware Retrieval, Attributed Evidence, and Contract-Constrained Extraction - addresses this gap with target-grouped retrieval, independent typed evidence localization, multimodal table extraction, schema-driven table construction, and fail-closed validation. It indexes 27,487 papers through passage, object, alias, citation, and dense representations while retaining the question target behind each signal. For tables, TRACE predicts the observation unit before extracting values and assembles rows with evaluator-compatible key normalization. Our audited selected clean-track artifact scores 0.760613 on the off
    
[^111]: 多模态大语言模型在哪里失败以及为什么失败：面向能力失败诊断的因果任务分解

    Where MLLMs Fail and Why: Causal Task Decomposition for Capability Failure Diagnosis

    [https://arxiv.org/abs/2609.38851](https://arxiv.org/abs/2609.38851)

    该论文提出一个因果分解框架与 CADET 诊断基准，通过对任务先决条件进行受控干预，将多模态大语言模型在组合任务上的失败区分为目标能力的内在缺陷和上游级联错误，从而精确诊断模型在哪里失败以及为什么失败。

    

    arXiv:2609.38851v1 公告类型：cross。摘要：在组合任务上的端到端准确率只能记录多模态大语言模型（MLLM）失败的频率，但无法区分失败究竟是源于目标能力的内在缺陷，还是来自上游先决条件的级联错误。我们提出了一个因果分解框架，通过对每个任务的先决条件依赖关系进行受控干预来分离这两种失败模式。我们的能力指标（NC、IC、RC）在无辅助、先决条件正确或先决条件错误三种情况下对每个任务进行评分，以诊断失败发生的位置；贡献指标（N-Score、S-Score）改编自因果概率，量化每个先决条件的必要性与充分性，以确定失败的原因。我们在 CADET 中实例化了该框架，这是一个诊断基准，包含 10 个组合任务，分解为 46 个单元任务，并配有超过 33,000 个人工标注的问题，涵盖感知、空间、时间和认知等类别。使用我们的框架对前沿 MLLM 进行诊断（摘要在此处截断）

    arXiv:2609.38851v1 Announce Type: cross  Abstract: End-to-end accuracy on compositional tasks records how often MLLMs fail, but cannot distinguish whether a failure reflects an intrinsic deficit in the targeted capability or a cascading error from an upstream prerequisite. We propose a causal decomposition framework that isolates these two failure modes through controlled interventions on the prerequisite dependencies of each task. Our capability metrics (NC, IC, RC) score each task under unassisted, correct, or incorrect prerequisites to diagnose where failures arise; contribution metrics (N-Score, S-Score), adapted from probabilities of causation, quantify each prerequisite's necessity and sufficiency to determine why. We instantiate the framework in CADET, a diagnostic benchmark of 10 composite tasks decomposed into 46 unit tasks with over 33,000 human-annotated questions spanning perception, spatial, temporal, and cognitive categories. Diagnosing frontier MLLMs with our framework u
    
[^112]: OpenJev-RLCD：一个可运行的RLCD实现

    OpenJev-RLCD: A Working RLCD Implementation

    [https://arxiv.org/abs/2609.38850](https://arxiv.org/abs/2609.38850)

    本文实现了面向推理模型的RLCD（校准决策强化学习），用严格正则评分规则对采样推理依据后的答案分布评分，证明RLVR是缺失多样性项的混合目标，并提出“先校准后强化”的两阶段训练方案，使模型既保持推理能力又获得良好校准。

    

    诸如Jev这类决策模型以概率形式回答问题，而这些概率只有经过校准才有用。开源复现通常依赖监督微调加温度缩放，而基于可验证奖励的强化学习会使推理模型过度自信。我们提出了一个面向推理模型的、可运行的“校准决策强化学习”实现：模型采样一条推理依据，随后我们用严格正则的评分规则对其给出的答案分布进行评分。一个方差恒等式表明，对多个样本的混合分布进行评分会奖励相互不一致的推理依据，而RLVR恰好是去掉了多样性项的该混合目标。若朴素地直接优化，逐条推理依据的目标要么使模型关闭推理，要么被策略梯度噪声淹没，由此引出两阶段方案：先校准，再强化。在两个推理任务上使用Qwen3-1.7B（3个随机种子）进行实验（摘要截断）。

    arXiv:2609.38850v1 Announce Type: new  Abstract: Decision models such as Jev answer questions with probabilities, which are only useful if they are calibrated. Open-source reproductions rely on supervised fine-tuning plus temperature scaling, while reinforcement learning from verifiable rewards (RLVR) makes reasoning models overconfident. We present a working implementation of reinforcement learning for calibrated decisions (RLCD) for reasoning models: the model samples a rationale, and we score the answer distribution it commits to afterwards with a strictly proper scoring rule. A variance identity shows that scoring the mixture of several samples rewards disagreeing rationales, and that RLVR is exactly this mixture objective without its diversity term. Optimized naively, the per-rationale objective either switches reasoning off or is drowned out by policy-gradient noise, which leads to a two-stage recipe: calibrate, then reinforce. With Qwen3-1.7B on two reasoning tasks (3 seeds, pai
    
[^113]: 在注意力机制中同时扩展参数与上下文：基于混合头的原生稀疏注意力

    Scaling Parameter and Context in Attention: Native Sparse Attention from Mixture-of-Head

    [https://arxiv.org/abs/2609.38832](https://arxiv.org/abs/2609.38832)

    提出NAMOH架构原生的稀疏注意力机制，通过每个token仅激活H个头中的K个，在不扫描完整历史、不增加总KV存储的前提下，同时实现注意力参数扩展与上下文高效扩展。

    

    扩展注意力参数可以提升语言模型的质量，但保留完整的token历史使得在长上下文中增加注意力头的代价高昂。此外，由于注意力机制负责检索和组合上下文信息，参数扩展也应能支持更长的上下文。因此我们提出这样的问题：注意力的参数扩展能否直接实现高效且有效的上下文扩展？我们提出了NAMOH，一种架构原生的稀疏注意力机制，它对每个token仅激活H个头中的K个。每个头只保留分配给它的token，并在这个子序列内执行因果注意力。这样，头的选择无需扫描完整历史，就能共同决定激活的参数数量和可用的上下文范围。在均衡分配下，在固定K的情况下增加H会缩短每个头的历史长度，并减少每个token的键值（KV）访问量，同时不增加总的KV存储。我们进一步支持头相对旋转位置嵌入以缩短p……

    arXiv:2609.38832v1 Announce Type: new  Abstract: Scaling attention parameters can improve language model quality, but retaining full token histories makes additional heads costly at long contexts. Furthermore, since attention retrieves and combines contextual information, parameter scaling should also support longer contexts. We therefore ask whether attention parameter scaling can directly enable efficient and effective context scaling. We introduce NAMOH, an architecture-native sparse attention mechanism that activates $K$ of $H$ heads per token. Each head retains only its assigned tokens and performs causal attention within this subsequence. Head selection thus jointly determines active parameters and available context without scanning the full history. Under balanced assignments, increasing $H$ at fixed $K$ shortens head histories and reduces per-token key-value (KV) access without increasing total KV storage. We further support head-relative rotary position embeddings to shorten p
    
[^114]: 通过针对性改写伪造LLM作者身份指纹

    Forging LLM Authorship Fingerprints with Targeted Rewriting

    [https://arxiv.org/abs/2609.38831](https://arxiv.org/abs/2609.38831)

    提出ForgePrint框架，通过“先搜索后蒸馏”策略改写LLM输出以伪造作者身份指纹，使其被归因分类器误判为指定目标模型，蒸馏后的4B学生模型在CNN/DM上达到70.2%的目标归因成功率。

    

    模型归因分类器通常能够识别出某段文本是由哪个语言模型生成的，这使得模型特有的写作模式成为溯源信号。然而，在未修改文本上的准确归因并不能说明经过刻意改写后，预测结果是否仍能识别出原始来源。我们将这一问题形式化为“针对性指纹转移”：通过改写某个模型的输出，使归因分类器将其归因于选定的目标模型。我们以摘要任务为研究对象，在该任务中不同模型接收相同的文档并表达相同的基本内容，为条件生成提供了受控的研究环境。我们提出ForgePrint，一个“先搜索后蒸馏”的框架：首先搜索能够使归因结果向目标指纹移动的改写方式，然后将筛选出的改写蒸馏到一个可单次通过生成的4B学生模型中。在CNN/DM数据集上，学生模型达到70.2%的目标归因成功率，超过了其教师模型（54.1%）以及最先进的基线方法。

    arXiv:2609.38831v1 Announce Type: new  Abstract: Model-attribution classifiers can often identify which language model produced a text, making model-specific writing patterns a signal of provenance. Accurate attribution on unmodified text, however, does not show whether the prediction still identifies the original source after deliberate rewriting. We formulate this problem as targeted fingerprint transfer: rewriting one model's output so that attribution classifiers assign it to a chosen target model. We study summarization, where different models receive the same document and express the same underlying content, providing a controlled setting for conditional generation. We introduce ForgePrint, a search-then-distil framework that first searches for rewrites that move attribution toward a target fingerprint, then distils the selected rewrites into a one-pass 4B Student model. On CNN/DM, the Student reaches 70.2% target success rate, outperforming both its Teacher (54.1%) and the stron
    
[^115]: BARRAC：将英语方面级情感分析方法适配于阿拉伯语方言分类任务

    BARRAC: Adaptation of an English Aspect-based Sentiment Analysis Approach for Classification Tasks in Arabic Dialects

    [https://arxiv.org/abs/2609.38820](https://arxiv.org/abs/2609.38820)

    本文提出 BARRAC 方法，将英语方面级情感分析框架适配到阿拉伯语方言分类任务，在五个阿拉伯语方言数据集上取得 63.93% 的平均宏 F1，超过最佳少标签 SOTA 3%，并在五个任务中的四个上超越 GPT-4o。

    

    随着阿拉伯语自然语言处理的快速发展，已有多个模型、数据集和基准被报道。本文探讨了为英语等多数语言开发的方法是否可以适配到阿拉伯语任务中。我们将一个英语方面级情感分析框架适配到阿拉伯语分类任务中，并将该适配方法命名为 BARRAC：面向阿拉伯语任务的头脑风暴对齐与替换表示学习。BARRAC 用阿拉伯语语言装置和针对方言情感、讽刺以及方言识别的标记替换了消费者评论属性池，并用两阶段训练替换了噪声自训练。在五个阿拉伯语方言数据集上进行评估，BARRAC 实现了 63.93% 的平均宏 F1 分数，超过最佳的少标签 SOTA 3%，并在五个任务中的四个上超越了 GPT-4o。错误分析提供了对剩余挑战的见解。这些结果表明，适配特定任务的方法是一种有效路径。

    arXiv:2609.38820v1 Announce Type: cross  Abstract: With the rapid growth of Arabic NLP, several models, datasets and benchmarks have been reported. This paper asks whether approaches developed for majority languages like English can be adapted to Arabic tasks. We adapt an English aspect-based sentiment analysis framework to Arabic classification tasks and present the adaptation as BARRAC: Brainstorming Alignment and Replaced Representation learning for ArabiC tasks. BARRAC replaces consumer-review attribute pools with Arabic linguistic devices and markers for dialectal sentiment, sarcasm, and dialect identification, and replaces noisy self-training with two-stage training. Evaluated on five Arabic dialect datasets, BARRAC achieves a mean macro-F1 of 63.93\%, outperforming the best few-label SOTA by 3\%, and outperforming GPT-4o on four out of five tasks. Error analysis provides insights into remaining challenges. These results demonstrate that adapting task-specific approaches is a pro
    
[^116]: 谁的声音能在摘要中幸存？对大语言模型员工倾听机制的“声音留存”审计

    Whose Voice Survives the Summary? A Voice-Retention Audit of LLM Employee Listening

    [https://arxiv.org/abs/2609.38818](https://arxiv.org/abs/2609.38818)

    该论文提出“声音留存/表征比率”这一新指标，对一家全球公司2,586条双语员工反馈及45份LLM生成的领导摘要进行审计，发现摘要流程按出现频率而非情感倾向筛选声音——仅被提及一次的关切有86%被丢弃、简短及纯德语内容更易流失——从而揭示了仅靠情感审计无法发现的摘要代表性偏差。

    

    组织日益依赖大语言模型（LLM）摘要将员工反馈传递给领导者，但这一未经审计的中间环节可能让员工已经表达出的声音被悄然埋没。我们提出了一项衡量摘要中代表性偏差的指标——声音留存/表征比率，并将其应用于来自一家全球性专业服务公司的2,586条双语（英语/德语）自由文本回复。首先，员工提出批评的可靠性高于提出表扬（员工保留表扬不发的可能性是批评的82倍）。其次，在针对45份呈报领导者的摘要的分析中，我们发现该流程依据的是出现频率（流行度）而非情感倾向：批评得以保留，但仅被提及一次的关切有86%的概率被丢弃，简短内容和纯德语内容也在同一维度上流失（主题留存率0.14对0.74；德语内容呈方向性影响）。在控制出现频率后，情感倾向并无独立的效应；其损害是由普遍性驱动的，而这是仅基于情感的审计所无法捕捉的。针对性的提示词只能召回被明确点名提及的主题。我们的贡献……（摘要原文在此处截断）

    arXiv:2609.38818v1 Announce Type: new  Abstract: Organizations increasingly route employee feedback to leaders through large language model (LLM) summaries, an unaudited layer that silences already-spoken voice. We introduce a Voice Retention / Representation Ratio metric for representational bias in summarization and apply it to a bilingual (English/German) corpus of 2,586 free-text responses from a global professional service company. First, employees supply criticism more reliably than praise (withholding praise is 82 times more common). Second, across 45 leader-summaries the pipeline filters by popularity, not sentiment: criticism survives, yet a concern voiced once is dropped 86% of the time, with short and German-only content lost on the same axis (theme retention 0.14 vs 0.74; German directional). Controlling for frequency, sentiment has no independent effect; the harm is prevalence-driven, which sentiment-only audits miss. A targeted prompt recovers only named themes. We contri
    
[^117]: 当推理偏离正轨：失控推理的注意力动态

    When Reasoning Goes Astray: Attention Dynamics of Uncontrolled Reasoning

    [https://arxiv.org/abs/2609.38817](https://arxiv.org/abs/2609.38817)

    本文提出RADAR方法，通过动态注意力实时识别大型推理模型的推理状态，揭示良性反思如何演变为失控生成，并将异常注意力分布重新对齐以缓解失控推理带来的成本与风险。

    

    大型推理模型通过扩展推理提升复杂任务的性能，但同样的过程可能退化为冗余的验证和持续的生成循环。这种失控推理会增加推理成本，并带来资源耗尽和服务降级的风险。然而，现有的缓解方法大多通过截断长输出或对表面重复现象作出反应，因而既无法区分正常思考与失控推理，也无法解释良性推理如何退化为有害行为。本文将LRM的生成过程操作化为四种状态，并进一步提出了基于动态注意力响应的推理状态分析方法（RADAR），该方法能够实时识别当前的推理状态，并刻画有效反思如何演变为失控生成。在RADAR分析的指导下，我们进一步将异常的注意力分布重新对齐到正常模式中观察到的（分布）……

    arXiv:2609.38817v1 Announce Type: new  Abstract: Large reasoning models (LRMs) improve performance on complex tasks through extended reasoning, yet the same process can degenerate into redundant verification and persistent generation loops. Such uncontrolled reasoning increases inference cost and creates risks of resource exhaustion and service degradation. However, existing mitigations largely truncate long outputs or react to surface repetition, and thus fail to distinguish normal thinking from uncontrolled reasoning or explain how benign reasoning degenerates into harmful behavior. In this paper, we operationalize LRM generation as four states and further introduce Reasoning-state Analysis via Dynamic Attention Responses (RADAR), which identifies the current reasoning state in real time and characterizes how effective reflection can develop into uncontrolled generation. Guided by RADAR's analysis, we further realign abnormal attention distributions toward patterns observed in normal
    
[^118]: 你被录用了：面向大语言模型协作的策略性模型选择

    You're Hired: Strategic Model Selection for LLM Collaboration

    [https://arxiv.org/abs/2609.38816](https://arxiv.org/abs/2609.38816)

    该论文研究了多LLM系统中的模型选择问题，提出并系统评估了9种选择算法，证明策略性组建模型团队比随机或启发式方法最高可提升36.1%的性能。

    

    随着多智能体和模型协作算法日益流行，用以结合不同大语言模型（LLM）的优势，现有系统仍然受限于预定义和手工构建的模型池。在本工作中，我们研究了多LLM系统中的模型选择问题。我们提出并系统评估了一个包含9种选择算法的分类体系，涵盖模型描述多样性、能力感知的行为多样性以及基于LLM的“招聘者”等方法。我们在两个分别包含10个和32个模型的候选池上进行了大量实验，部署于四种模型协作算法中，并在数学、编程、问答和推理等任务上进行了评估。结果表明，成功的选择算法大幅优于随机或基于启发式的团队组建方法（例如仅选择个体表现最佳的模型），在各种设置下性能提升最高达36.1%。具体而言，基于能力和训练的选择策略……

    arXiv:2609.38816v1 Announce Type: new  Abstract: While multi-agent and model collaboration algorithms gain traction to combine the strengths of diverse Large Language Models (LLMs), existing systems remain bottlenecked on pre-defined and hand-crafted model pools. In this work, we investigate the problem of model selection in multi-LLM systems. We propose and systematically evaluate a taxonomy of 9 selection algorithms ranging from diversity of model descriptions, capability-aware behavioral diversity, and LLM-based recruiters. We conduct extensive experiments across two candidate pools of 10 and 32 models, deployed in four model collaboration algorithms, and evaluated across tasks spanning math, coding, QA, and reasoning. Results demonstrate that successful selection algorithms greatly outperform random or heuristics-based teams such as merely selecting the models with top individual performance, by up to 36.1% across settings. Specifically, capability- and training-based selection str
    
[^119]: 终端智能体能信任自己的验证吗？自我验证的诊断与改进

    Can Terminal Agents Trust Their Own Verification? Diagnosing and Improving Self-Verification

    [https://arxiv.org/abs/2609.38812](https://arxiv.org/abs/2609.38812)

    该论文提出一个诊断框架来量化终端智能体自我验证的可靠性，发现验证行为虽普遍存在，但主要弱点在于错误检测与修复环节——错误候选方案仅 61.43% 被检出，被检出的错误仅 49.36% 被成功修复。

    

    终端智能体在通过与命令行环境交互来完成任务的过程中，依赖自我验证来评估和修正其解决方案。然而，这种自我验证的可信程度仍然缺乏深入理解。为了系统地研究这一问题，我们引入了一个诊断框架，该框架识别每条轨迹中的第一个完整解决方案，判断其是否客观正确，并利用这一真值来量化智能体后续的验证与恢复行为。将该框架应用于 TerminalBench 2.1 上的十个终端智能体，我们发现：在形成完整候选方案之后，验证行为几乎是普遍存在的，但仅有 61.43% 的错误候选方案被检测出来，且仅有 49.36% 的被检测错误被成功修复。这些结果表明，自我验证的主要弱点并不在于是否启动验证，而在于检测和修复错误。基于这些发现，我们提出 S（原文摘要在此处截断）

    arXiv:2609.38812v1 Announce Type: new  Abstract: Terminal agents rely on self-verification to assess and correct their solutions as they solve tasks through interaction with command-line environments. Yet how trustworthy such self-verification is remains poorly understood. To investigate this question systematically, we introduce a diagnostic framework that identifies the first complete solution in each trajectory, determines whether it is objectively correct, and uses this ground truth to quantify the agent's subsequent verification and recovery behavior. Applying it to ten terminal agents on TerminalBench2.1, we find that verification is nearly universal after a complete candidate is formed, yet only 61.43\% of incorrect candidates are detected and only 49.36\% of detected errors are successfully repaired. These results show that the main weakness in self-verification lies not in initiating verification, but in detecting and repairing errors. Motivated by these findings, we propose S
    
[^120]: StateTree：通过强化学习增强长期对话推理

    StateTree: Enhancing Long-Term Dialogue Reasoning via Reinforcement Learning

    [https://arxiv.org/abs/2609.38809](https://arxiv.org/abs/2609.38809)

    StateTree是一种数据驱动的强化学习方法，通过在稀缺对话数据上构建树结构路径追踪辅助任务并采用课程式强化学习训练，有效提升了大语言模型的长期对话推理能力。

    

    部署为个性化助手的大型语言模型必须对漫长且不断演变的交互历史进行推理。然而，在长期对话推理中，相关证据分散在各个会话之中，用户偏好可能随时间发生改变，而标准的长上下文训练方法在数据稀缺和计算成本高昂的情况下无法解决这些挑战。我们提出了StateTree，这是一种数据驱动的强化学习方法，能够从稀缺的对话数据中构建具有可验证真实答案的具有挑战性的辅助任务。StateTree通过树结构路径追踪任务来增强多会话对话：将键值记录嵌入到各个会话中，形成一棵二叉树。解决该任务需要模型通过跨会话检索记录并比较时间戳来化解分支，从而从根节点遍历到叶节点，然后在干扰性叶节点中恢复出隐藏的目标问题。我们采用课程强化学习训练方式，逐步增加树的深度……

    arXiv:2609.38809v1 Announce Type: cross  Abstract: Large language models deployed as personalized assistants must reason over long, evolving interaction histories. However, in long-term dialogue reasoning, relevant evidence is scattered across sessions, preferences may be revised over time, and standard long-context training fails to address these challenges under data scarcity and prohibitive computational costs. We propose StateTree, a data-driven RL method that constructs a challenging auxiliary task from scarce dialogues with verifiable ground truth. StateTree augments multi-session dialogues with a tree-structured path-tracing task: key-value records are embedded across sessions to form a binary tree. Solving the task requires the model to traverse from root to leaf by retrieving records across sessions and comparing timestamps to resolve branches, then recover the hidden target question among distractor leaves. We apply curriculum RL training progressively increasing tree depth a
    
[^121]: 黑板智能在全球约束问题上可以超越自回归模型

    Blackboard Intelligence Can Surpass Autoregressive on Globally Constrained Problems

    [https://arxiv.org/abs/2609.38806](https://arxiv.org/abs/2609.38806)

    该论文提出“黑板智能”推理范式，让扩散语言模型在可修改的画布上搜索候选解，并以平均置信度作为全局一致性的代理信号，从而在全球约束问题上超越自回归模型。

    

    下一个词预测（next-token prediction）推动了大语言模型的显著进展，然而越来越多的证据表明，这类模型在受复杂全局约束支配的问题上可能表现挣扎。在本工作中，我们聚焦于这一情形，并探究其中的一些限制是否源于下一个词预测本身所带来的推理接口。我们通过“黑板智能”（blackboard intelligence）来研究这一问题：这是一种推理时的视角，模型在固定的、可反复修改的画布上工作，并在候选解状态之间进行搜索，而不是固守一条因果的、从左到右的生成轨迹。我们用扩散语言模型来实例化这一想法，其任意顺序预测的接口天然地为部分填充的解状态提供预测。我们的关键观察是，平均置信度——一个可从标准掩码扩散目标中简单获得的模型内部量——能够为全局一致性提供有用的代理信号，并可以引导……（摘要在此处被截断）

    arXiv:2609.38806v1 Announce Type: cross  Abstract: Next-token prediction has driven remarkable progress in large language models, yet a growing body of evidence suggests that they can struggle on problems governed by complex global constraints. In this work, we focus on this regime and ask whether some of these limitations arise from the inference interface induced by next-token prediction itself. We study this question through blackboard intelligence: an inference-time perspective in which a model works on a fixed, revisable canvas and searches over candidate solution states rather than committing to a causal, left-to-right trajectory. We instantiate this idea with diffusion language models, whose any-order prediction interface naturally exposes predictions over partially filled solution states. Our key observation is that mean confidence, a simple model-internal quantity available from the standard masked diffusion objective, provides a useful proxy for global coherence and can guide
    
[^122]: 通过残差流动力学揭示失控重复现象

    Uncovering Uncontrolled Repetition through Residual Stream Dynamics

    [https://arxiv.org/abs/2609.38802](https://arxiv.org/abs/2609.38802)

    提出Tokenwise Residual Comparison（TRC）方法，通过比较生成过程中各Token对残差流的写入动态，从残差流动力学中识别、定位并抑制大型视觉-语言模型中的失控重复现象。

    

    失控重复会延长大型语言模型（LLMs）的自回归生成过程，并可能被用于资源消耗攻击。以往对重复生成的分析主要识别了中间层和后期层中强烈激活的特征。然而，失控重复活动在这些层中变得显著之前是如何出现和发展的，目前仍缺乏充分的理解。在本文中，我们主要在大型视觉-语言模型（LVLMs）中研究这一问题，这类模型通过视觉和文本输入支持更丰富的失控重复形式。我们提出了逐Token残差比较，这是一种从生成过程中的残差动力学中识别并定位与重复相关异常的方法。TRC通过比较各生成Token对残差流的注意力写入和多层感知机写入，来识别与重复相关的模式，进而选择性地抑制……（摘要在此处截断）

    arXiv:2609.38802v1 Announce Type: cross  Abstract: Uncontrolled repetition can prolong autoregressive generation in large language models (LLMs) and enable resource consumption attacks. Prior analyses of repetitive generation have identified strongly activated features in intermediate and late layers. However, how uncontrolled repetition activity emerges and develops before becoming prominent in these layers remains insufficiently understood. In this paper, we investigate this question primarily in large vision-language models (LVLMs), which support a richer set of uncontrolled repetitions through both visual and textual inputs. We propose Tokenwise Residual Comparison (TRC), a method that identifies and localizes anomalies associated with repetition from residual dynamics during generation. TRC compares attention and multilayer perceptron writes to the residual stream across generated tokens to identify patterns associated with repetition. It then selectively suppresses coordinates in
    
[^123]: 重叠、独特与冲突：大语言模型能否提取它们所能识别的内容？

    Overlap, Unique and Conflict: Can LLMs Extract What They Can Recognize?

    [https://arxiv.org/abs/2609.38799](https://arxiv.org/abs/2609.38799)

    本文提出重叠-独特-冲突（OUC）提取这一跨叙事新任务并构建了包含22K叙事对和140K实例的基准，对14个开源大模型的评估表明，模型提取重叠与冲突信息的能力远弱于提取独特信息的能力。

    

    理解多视角的替代叙事需要识别不同来源的信息之间如何一致、冲突或存在差异。现有的跨文本关系研究主要集中在对预定义文本对之间的关系进行分类（如蕴含或矛盾），而非直接从完整叙事中提取此类信息。为填补这一空白，我们提出了重叠-独特-冲突（Overlap-Unique-Conflict, OUC）提取任务，这是一个跨叙事任务，旨在从两个叙事中提取所有重叠、冲突和独特的子句。为支持这项研究，我们构建了一个包含约22K叙事对和140K个OUC实例的基准数据集，涵盖事实性、论证性和政治话语。通过评估14个开源大语言模型（0.6B-35B），我们发现独特信息远比重叠和冲突信息更容易提取：表现最强的模型Gemma-4-31B在重叠信息上仅达到61.13%的F1分数，在冲突信息上仅达到48.58%的F1分数，相比之下……

    arXiv:2609.38799v1 Announce Type: new  Abstract: Understanding multi-perspective alternative narratives requires identifying how their information agrees, conflicts, or differs across sources. Existing work on cross-text relations largely focuses on categorizing relations between predefined text pairs, such as entailment or contradiction, rather than directly extracting such information from full narratives. To address this gap, we introduce Overlap-Unique-Conflict (OUC) extraction, a cross-narrative task that extracts all overlapping, conflicting, and unique clauses from two narratives. To support this study, we construct a benchmark of approximately 22K narrative pairs and 140K OUC instances spanning factual, argumentative, and political discourse. Evaluating 14 open-source LLMs (0.6B-35B), we find that unique information is far easier to extract than overlap and conflict: the strongest model, Gemma-4-31B, reaches only 61.13% F1-score on overlap and 48.58% on conflict, against more t
    
[^124]: 评估模型知识演化下的持续校准

    Evaluating Persistent Calibration under Evolving Model Knowledge

    [https://arxiv.org/abs/2609.38797](https://arxiv.org/abs/2609.38797)

    该论文提出“持续校准”新问题，研究当模型知识随训练不断演化时，早期训练的置信度估计器能否无需重新监督就持续忠实地反映模型的知识状态，并借此探究置信度与知识之间的依赖关系。

    

    随着AI系统从静态知识库转向能够持续适应和学习的智能体，维持其可信度意味着需要让支撑这些系统的模型具备产生动态反映其不断变化的技能和知识的置信度估计的能力。我们提出了“持续校准”这一问题，它要求置信度估计器在模型知识发生变化时，无需重复监督就能忠实地反映模型中包含的知识。我们通过检查开放模型各个检查点之间的持续校准来将这一问题具体化，探究在早期检查点上训练的置信度估计器能否泛化到后续检查点。具体而言，我们旨在阐明置信度是否依赖于知识，这一问题的答案对置信度估计的可靠性具有重要意义。为了衡量这种关系，我们定义并在知识对比集上评估了校准……

    arXiv:2609.38797v1 Announce Type: cross  Abstract: As AI systems move from static repositories to agents that are capable of continual adaptation and learning, maintaining their trustworthiness means equipping the models backing them with the ability to produce confidence estimates that dynamically reflect their changing skills and knowledge. We introduce the problem of persistent calibration, which requires a confidence estimator to faithfully reflect the knowledge contained in a model as that knowledge changes, without recurring supervision. We operationalize this by examining persistent calibration across checkpoints of open models, asking whether confidence estimators trained on earlier checkpoints can generalize to later ones. Specifically, we aim to shed light on whether confidence is dependent on knowledge, a question with implications for the reliability of confidence estimates. To measure this relationship, we define and evaluate calibration on knowledge contrast sets: subsets
    
[^125]: 面向投机解码的离策略监督恢复

    Recovering Off-Policy Supervision for Speculative Decoding

    [https://arxiv.org/abs/2609.38795](https://arxiv.org/abs/2609.38795)

    提出一种基于rollout的训练框架，通过锚点标签重标注（ALR）和rollout内锚点（IRA）两个互补组件，恢复投机解码草稿模型因离策略token而丢失的完整监督，无需丢弃分歧槽位即可提升贪婪接受长度。

    

    投机解码中的分块草稿模型通常在由外部模型生成的语料库上进行训练，其中单个离策略的token会使整个块中所有后续槽位的监督失效。现有方法会丢弃这些产生分歧的槽位，导致严重的监督损失。为了在保留训练语料库的同时解决这一问题，我们提出了一种基于rollout的训练框架，通过两个互补的组件恢复完整的监督。第一个组件是锚点标签重标注，它用贪婪目标rollout得到的分布替换语料库标签，从而在所有预测槽位上恢复有效监督。第二个组件是rollout内锚点，它将草稿块直接置于这些rollout之中，使草稿模型暴露于目标模型生成的上下文中，并通过复用预计算的rollout特征而不产生额外的目标模型开销。在固定的视觉-语言和文本语料库上，我们的框架提高了贪婪接受长度……

    arXiv:2609.38795v1 Announce Type: new  Abstract: Block drafters for speculative decoding are commonly trained on corpora written by external models, where a single off-policy token invalidates supervision for all subsequent slots in a block. Existing approaches discard these divergent slots, resulting in severe supervision loss. To resolve this problem while preserving the training corpus, we propose a rollout-based training framework that recovers full supervision through two complementary components. The first component, Anchor-Label Relabelling (ALR), replaces corpus labels with distributions from greedy target rollouts, restoring valid supervision across all predicted slots. The second component, In-Rollout Anchors (IRA), places draft blocks directly inside these rollouts to expose the drafter to target-generated context, reusing precomputed rollout features at no additional target cost. Across fixed vision-language and text corpora, our framework increases greedy accepted length b
    
[^126]: 通过位置选择性自蒸馏从语言反馈中训练大语言模型评判器

    Training LLM Judges from Language Feedback via Position-Selective Self-Distillation

    [https://arxiv.org/abs/2609.38792](https://arxiv.org/abs/2609.38792)

    该论文提出位置选择性自蒸馏方法，利用教师与学生模型之间的逐位置熵变化来识别携带有效信号的位置，从而更充分地利用自然语言反馈训练大语言模型评判器，克服了基于结果监督的强化学习忽略标准选择token和语言反馈的局限。

    

    我们研究如何从自然语言反馈中训练大语言模型评判器，尤其针对那些评判结果在很大程度上取决于评判器采用哪些评估标准以及如何权衡这些标准的主观任务。主流的方法——基于结果监督的强化学习（如GRPO）——会将一个仅由最终评判准确性决定的标量奖励分配给生成序列中的每一个token，既没有为标准选择相关的token提供单独的信用分配，也忽略了自然伴随偏好标签出现的丰富语言反馈（如偏好理由）。自蒸馏是利用这类语言反馈的一种自然方式：同一个模型在以该反馈为条件时充当教师，提供密集的、位置级别的监督信号。然而，并非所有位置都携带同等有用的信号。我们利用教师与学生之间的逐位置熵变化，识别出两种模式：上下文锐化，即教师将概率集中到某个特定的（位置）……（原文摘要在此处被截断）

    arXiv:2609.38792v1 Announce Type: new  Abstract: We study training LLM judges from natural language feedback, especially for subjective tasks where the verdict depends strongly on which evaluation criteria the judge invokes and how it weighs them. The dominant approach, outcome-supervised RL (e.g., GRPO), credits every token in the rollout with a single scalar determined only by the accuracy of the final verdict, providing no separate credit at the criterion-choice tokens and ignoring the rich language feedback (e.g., preference rationales) that naturally accompanies preference labels. Self-Distillation (SD) is one natural way to use this language feedback: the same model, conditioned on this feedback, acts as a teacher providing dense, position-level supervision. However, not all positions carry equally useful signal. Using the per-position entropy shift between teacher and student, we identify two regimes: context sharpening, where the teacher concentrates probability on a particular
    
[^127]: Anchor-ECC：基于纠错码的水印化LLM输出局部完整性检测

    Anchor-ECC: Local Integrity Checking for Watermarked LLM Outputs via Error-Correcting Codes

    [https://arxiv.org/abs/2609.38722](https://arxiv.org/abs/2609.38722)

    提出Anchor-ECC方法，通过在水印结构中引入纠错码约束与边界锚点并结合动态规划解码器，可高效检测并定位对LLM生成文本的局部篡改编辑，块级检测真正例率达99.7%且误报率不超过7.6%。

    

    LLM水印已成为通过在生成过程中嵌入可检测模式来区分AI生成文本与人类撰写文本的有效方法。然而，生成后的小幅编辑可能在不破坏整体水印信号的情况下改变文本含义，从而导致修改后的内容仍被归因于原始模型的风险。我们提出Anchor-ECC，该方法将纠错码（ECC）约束和显式边界锚点纳入水印结构，并配以动态规划解码器来检测和定位生成后的编辑。在Qwen3-8B、Mistral-7B-Instruct-v0.3和OPT-125M三个模型上的实验表明，在混合插入、删除和替换的编辑场景下，近似硬设置实现了约99.7%的块级真正例率（TPR）和最高7.6%的误报率（FAR），同时保持了水印化输出与未水印化文本之间的区分能力。

    arXiv:2609.38722v1 Announce Type: cross  Abstract: LLM watermarking has become an effective approach to distinguishing AI-generated text from human-written text by embedding detectable patterns during generation. However, a small post-generation edit may change the meaning of the text without removing its overall watermark signal, creating a risk that the modified content is still attributed to the original model. We propose Anchor-ECC, which incorporates the error-correcting code (ECC) constraints and explicit boundary anchors into the watermark structure and pairs them with a dynamic-programming decoder to detect and localize post-generation edits. Across Qwen3-8B, Mistral-7B-Instruct-v0.3, and OPT-125M, the approximate-hard setting achieves about 99.7% block-level true positive rate (TPR) with at most 7.6% false alarm rate (FAR) for edit detection under mixed insertions, deletions, and substitutions, while preserving the distinction between watermarked outputs and unwatermarked text
    
[^128]: MetaSteer：基于注意力投影自适应实现上下文条件化的非线性引导

    MetaSteer: Context-Conditioned, nonlinear Steering via Attention-Projection Adaptation

    [https://arxiv.org/abs/2609.38718](https://arxiv.org/abs/2609.38718)

    MetaSteer通过在注意力投影矩阵上学习上下文相关的非线性干预，突破传统线性、上下文无关激活引导的瓶颈，且一次训练即可零样本迁移到未见概念和分布外场景。

    

    对大语言模型进行引导（steering）通常依赖于激活空间中线性的、与上下文无关的干预，这一假设近来已受到挑战，且当固定的表示必须编码多种行为区分时会引发信息瓶颈。我们提出MetaSteer，一种学习具有上下文相关效应的非线性干预的方法，并将其应用于注意力投影矩阵，从而在构造上使激活效应随输入上下文而变化，且无需线性概念几何假设。该方法被构建为基于偏好的优化，MetaSteer仅在一个汇总的偏好语料库上训练一次，即可零样本迁移到未见过的概念和分布外上下文。我们发现，尽管使用的是低秩适配器，MetaSteer仍能在隐状态轨迹中诱导出结构化的、上下文相关的变化，同时部分保留其局部轨迹动力学的某些方面，包括速度……

    arXiv:2609.38718v1 Announce Type: new  Abstract: Steering large language models typically relies on linear, context-independent interventions in activation space, an assumption that recent work has challenged and that can induce an information bottleneck when a fixed representation must encode many behavioral distinctions. We introduce MetaSteer, a method that learns nonlinear interventions with context-dependent effects and applies them to attention projection matrices, producing activation effects that vary with the input context by construction and requiring no linear concept-geometry assumption. Framed as preference-based optimization, MetaSteer is trained once on a pooled preference corpus and transferred zero-shot to unseen concepts and out-of-distribution contexts. We find that, despite using low-rank adapters, MetaSteer induces structured, context-dependent changes in hidden-state trajectories while partially preserving aspects of their local trajectory dynamics, including velo
    
[^129]: 打破巴别塔：一种面向长篇字幕翻译的自进化多智能体系统

    Breaking Babel: A Self-Evolving Multi-Agent System for Long-Form Subtitle Translation

    [https://arxiv.org/abs/2609.38660](https://arxiv.org/abs/2609.38660)

    SMART是一种自进化多智能体系统，通过剧集级持久记忆、动态路由与智能体混合层以及无需重训大语言模型的评判-精炼循环，实现术语与风格一致的长篇字幕翻译。

    

    长篇字幕翻译需要对跨越多集乃至整个剧集的语篇和文化语境进行推理，同时保持术语和风格的一致性。现有的单一大语言模型方法主要停留在句子层面，而多智能体系统通常采用静态工作流，无法适应场景复杂度或制作语境的变化。我们提出了SMART——一个面向长篇字幕翻译的自进化多智能体系统。在测试时训练阶段，SMART构建持久化的剧集级记忆，并通过动态路由器和智能体混合层翻译部分句子，该层配备了术语校验、字幕约束验证和上下文检索等工具。一个评判-精炼循环对候选译文进行评分，并利用文本批评来更新智能体提示词和路由策略，而无需重新训练底层大语言模型。在测试时推理阶段，进化后的配置将翻译剧集中剩余的内容。我们还引入了Su……（摘要在此处截断）

    arXiv:2609.38660v1 Announce Type: cross  Abstract: Long-form subtitle translation requires reasoning over discourse and cultural context spanning episodes or entire series, while maintaining consistent terminology and style. Existing single-LLM methods are largely sentence-level, and multi-agent systems often use static workflows that do not adapt to scene complexity or production context. We propose SMART, a Self-evolving Multi-Agent system for long-foRm subtitle Translation. During test-time training, SMART builds persistent series-level memory and translates a subset of sentences through a dynamic router and Mixture-of-Agents layer with tools for terminology verification, subtitle constraint validation, and contextual retrieval. A judge-refiner loop scores candidates and uses textual critiques to update agent prompts and routing policies without retraining the underlying LLMs. During test-time inference, the evolved configuration translates the remaining series. We also introduce Su
    
[^130]: Tacit-TTS：从自回归解码到掩码预测的高效免转录文本语音克隆

    Tacit-TTS: From Autoregressive Decoding to Masked Prediction for Efficient Transcript-Free Voice Cloning

    [https://arxiv.org/abs/2609.38658](https://arxiv.org/abs/2609.38658)

    Tacit-TTS通过将自回归解码替换为掩码非自回归生成、引入免训练的声学长度估计以及ReFlow蒸馏加速流匹配渲染，实现了免转录文本的高质量零样本语音克隆，生成速度比IndexTTS2快10倍以上。

    

    采用自回归语义建模的文本转语音（TTS）系统已展现出强大的零样本语音克隆性能和丰富的表现力变化，但其顺序解码会带来显著的延迟。非自回归替代方案虽然生成速度快得多，但通常依赖更严格的参考条件，例如在推理时需要提供参考语音的转录文本。我们提出了Tacit-TTS，这是一个从IndexTTS2蒸馏而来的高效免转录文本零样本语音克隆系统。我们的模型用掩码非自回归生成取代了自回归的文本到语义解码，引入了无需训练的声学长度估计方法，并通过ReFlow蒸馏加速了流匹配渲染器。在两个英语和两个普通话数据集上的实验表明，Tacit-TTS在实现具有竞争力的零样本合成质量的同时，对于超过5秒的语音，其生成速度比IndexTTS2快10倍以上。其免转录文本的条件机制……

    arXiv:2609.38658v1 Announce Type: cross  Abstract: TTS systems with autoregressive semantic modeling have demonstrated strong zero-shot voice cloning performance and rich expressive variation, but their sequential decoding incurs substantial latency. Non-autoregressive alternatives offer much faster generation, yet often rely on more restrictive reference conditioning, such as requiring transcripts of the reference speech during inference. We present Tacit-TTS, an efficient transcript-free zero-shot voice cloning system distilled from IndexTTS2. Our model replaces autoregressive text-to-semantic decoding with masked non-autoregressive generation, introduces training-free acoustic length estimation, and accelerates the flow-matching renderer through ReFlow distillation. Across two English and two Mandarin datasets, Tacit-TTS achieves competitive zero-shot quality while generating speech over 10x faster than IndexTTS2 for utterances longer than 5 seconds. Its transcript-free conditioning
    
[^131]: 以编码器速度实现的强大多语言隐私标注

    Strong Multilingual Privacy Tagging at Encoder Speed

    [https://arxiv.org/abs/2609.38630](https://arxiv.org/abs/2609.38630)

    该研究提出了一个以编码器速度运行的多语言隐私实体标注模型，通过覆盖感知掩码与子词边界修复等技术，在7种语言的人工金标准测试上取得88.8的脱敏F1分数，显著超越GLiNER2、Microsoft Presidio和OpenAI Privacy Filter等现有方法。

    

    隐私脱敏必须在删除个人信息的同时保留文本中表达的关系。我们开发了一个支持细粒度区分的多语言命名实体标注器，可服务于多种脱敏策略，并提供了低成本学习额外区分类别的方法。我们在35种语言的前沿模型标注数据上，对带有仿射跨度标注头的多语言编码器进行微调，通过覆盖感知掩码技术重放映射后的人工金标准数据，以避免将未标注的实体类型误当作负例，并利用学习到的±1字符调整来修复子词边界。在7种语言共1,283个人工金标准测试片段上，该模型的最佳脱敏F1分数达到88.8，相比之下，已发布的GLiNER2为69.1（该对比中排除了其无法表示的11个类型，若不作此豁免则为68.8），针对新训练数据适配后的GLiNER2为67.8，Microsoft Presidio为57.3，而已发布的最佳OpenAI Privacy Filter微调版本仅为35.8。通过增加约50,000个标注训练样本……（原文摘要在此处截断）

    arXiv:2609.38630v1 Announce Type: new  Abstract: Privacy redaction must remove personal information while preserving relationships expressed in text. We develop a multilingual named-entity tagger with fine-grained distinctions supporting varied redaction policies and methods for cheaply learning additional distinctions. We fine-tune a multilingual encoder with an affine span-tagging head on frontier-model annotations in 35 languages, replay mapped human gold with coverage-aware masking so unannotated types are not treated as negatives, and repair subword boundaries with a learned +/-1-character adjustment. On 1,283 human-gold test segments in seven languages, best measured redaction F1 is 88.8, against 69.1 for published GLiNER2 with 11 unrepresentable types excluded from its task (68.8 without that exemption), 67.8 for GLiNER2 adapted to the new training data, 57.3 for Microsoft Presidio and 35.8 for the best published OpenAI Privacy Filter fine-tune. Adding about 50,000 annotated tra
    
[^132]: 约鲁巴语中曲折调的标记方法

    Marking Contour Tones in Yor\`{u}b\'{a}

    [https://arxiv.org/abs/2609.38627](https://arxiv.org/abs/2609.38627)

    本文提出在约鲁巴语正字法中采用caron（ˇ）和circumflex（ˆ）符号来标记单个元音上的升降曲折调，以解决传统拼写中声调信息缺失甚至颠倒姓名含义的问题，并使其首次可通过标准键盘输入和计算文本处理。

    

    约鲁巴语是一种声调语言，其曲折调在正字法书写上一直存在难题。这一问题在个人姓名和词汇中尤为突出，因为这些词的传统拼写避免了元音延长，而元音延长本可为第二个声调提供承载音节。尤其值得关注的是一类姓名，其传统拼写不仅省略了声调信息，还会颠倒名字的含义，有时甚至表达出与名字本意相反的内容。本文描述了这一问题，说明了现有解决方案的不足，并提议采用caron（倒折音符ˇ）和circumflex（抑扬符ˆ）符号。这些符号自Olmsted（1951）以来在约鲁巴语音系学研究中已有先例，作为书写惯例用于单个元音之上，以编码升调和降调曲折调，从而首次使这些曲折调能够通过标准键盘输入和计算文本处理来访问。该提议得到了支持。

    arXiv:2609.38627v1 Announce Type: new  Abstract: Yor\`ub\'a is a tonal language in which contour tones pose persistent orthographic challenges. These are especially notable for personal names and lexical items whose conventional spellings avoid vowel lengthening that would otherwise provide a host syllable for the second tone. A particular concern is a class of names in which the conventional spelling does not just omit tonal information but inverts the meaning of said name, sometimes asserting the opposite of what the name intends. This paper describes the problem, illustrates the inadequacy of current solutions, and proposes the adoption of the caron and circumflex marks. These are symbols with precedent in Yor\`ub\'a phonological scholarship since Olmsted (1951), used as orthographic conventions on single vowels to encode rising and falling contour tones, making them accessible for the first time through standard keyboard input and computational text processing. The proposal is supp
    
[^133]: 当科学矛盾在翻译中迷失

    When Scientific Contradictions Are Lost in Translation

    [https://arxiv.org/abs/2609.38621](https://arxiv.org/abs/2609.38621)

    该研究通过将不可满足的XOR约束系统伪装成不同实验室的科学报告，首次系统量化了语言模型判断科学发现是否真正矛盾的能力，发现模型面对显式约束时准确率高达90%-96%，但在科学文本中往往偏离约束逻辑而偏向生物学预期。

    

    两个科学发现可以不一致而并不构成真正的相互矛盾。判断它们是否冲突，需要知道它们是否描述了可比较的测量。我们研究了语言模型在这一决策点上的行为。在一个受控任务中，我们生成一个不可满足的XOR（异或）约束系统，并将其约束转化为来自不同实验室的科学报告。其中一种赋值满足更多约束，而另一种赋值满足的约束较少但更符合预期的生物学。这构成了一个简单的两难困境：模型会选择最符合约束的赋值，还是选择更符合生物学预期的赋值？当约束被直接陈述时，GPT-5.6 Sol和Claude Opus 5分别在90%和96%的情况下恢复出证据支持最强的赋值。然而，在科学文本中，模型的表现有所不同：Claude Opus 5常常更倾向于符合生物学预期的赋值。移除该生物学……（原摘要在此处截断）

    arXiv:2609.38621v1 Announce Type: new  Abstract: Two scientific findings can disagree without contradicting each other. Determining whether they conflict requires knowing whether they describe comparable measurements. We study how language models behave at this decision point. In a controlled task, we generate an unsatisfiable XOR constraint system and translate its constraints into scientific reports from different laboratories. One assignment satisfies more constraints, while another satisfies fewer but better matches expected biology. This creates a simple dilemma: does the model choose the assignment that best fits the constraints, or the one that better matches biological expectations? When the constraints are stated directly, GPT-5.6 Sol and Claude Opus 5 recover the best-supported assignment in 90% and 96% of cases, respectively. In scientific prose, however, the models behave differently. Claude Opus 5 often prefers the biologically expected assignment. Removing that biological
    
[^134]: StreamDecisionBench：在动态演化的语言流上评估生效决策

    StreamDecisionBench: Evaluating Decisions in Force on Evolving Language Streams

    [https://arxiv.org/abs/2609.38612](https://arxiv.org/abs/2609.38612)

    该论文提出 StreamDecisionBench 基准，评估语言模型在证据流不断演化时每一时刻“生效决策”的正确性，并将错误归因于判断或延迟，弥补了传统离线准确率无法衡量决策时效性的缺陷。

    

    随着自然语言驱动越来越多的应用，语言模型越来越多地作为决策组件运行在程序内部：程序向其发送当前状态，并根据返回的决策采取行动，直到新决策到来。当证据在推理过程中发生变化时，某个对其自身状态而言正确的决策，可能在该状态已经过去之后仍然生效——例如当客户开始读出卡号后，通话录音机仍在继续录音；而不计时间的（离线）准确率会将此类错误计为正确。我们提出 StreamDecisionBench（SDB），它评估每一时刻的生效决策，并将每个错误时刻归因于判断错误、延迟或两者兼有。其场景在四个应用族中流式呈现证据，参考决策由可执行代码依据公开规则计算得出。我们以对数时间轴上归一化的曲线下面积来汇总 1-5 秒更新区间内的生效准确率，对相等的时间赋予相等的权重……（原文摘要在此处截断）

    arXiv:2609.38612v1 Announce Type: new  Abstract: As natural language drives more applications, language models increasingly run inside programs as decision components: the program sends them the current state and acts on the returned decision until a newer one arrives. When evidence changes during inference, a decision correct for its own state can stay in force after that state has passed, as when a call recorder keeps running after a customer starts reading out a card number; untimed (offline) accuracy counts such an error as correct. We introduce StreamDecisionBench (SDB), which evaluates the decision in force at every instant and attributes every erroneous instant to judgment, latency or both. Its scenarios stream evidence in four application families, with reference decisions computed from public rules by executable code. We summarize in-force accuracy across update intervals of 1-5 s by its normalized area under the curve on a logarithmic time axis, giving equal weight to equal m
    
[^135]: SecureVibe：让氛围编程更加安全

    SecureVibe: Making Vibe Coding More Secure

    [https://arxiv.org/abs/2609.38606](https://arxiv.org/abs/2609.38606)

    SECUREVIBE通过围绕安全规划与测试行为构建训练信号——包括4个安全任务上的监督微调以及基于可验证执行反馈和提示自监督的后训练方法——显著提升了氛围编程中代码的安全性。

    

    随着氛围编程（vibe coding）的能力日益增强并广泛普及，即使是功能正确的解决方案中存在的安全漏洞也日益受到关注。在研究那些功能正确但不安全的解决方案时，我们发现不安全的智能体针对功能需求背后隐藏的安全风险进行有效规划和测试的可能性不到一半。受此启发，我们开发了SECUREVIBE，这是一种明确针对代码安全的规划与测试能力的训练方案。SECUREVIBE围绕这些安全行为构建训练信号，包括在包含4个安全任务的安全套件上进行监督微调，以及两种后训练方法SECUREVIBE_rl和SECUREVIBE_hg，分别利用可验证的执行反馈和基于提示的自监督来增强安全能力。我们的SECUREVIBE在4个基准测试的两类安全编码任务上均优于基线方法。具体而言，SECUREVIBE提升了

    arXiv:2609.38606v1 Announce Type: cross  Abstract: As vibe coding becomes increasingly capable and widespread, security vulnerabilities in even functionally correct solutions are a growing concern. When investigating functionally correct but insecure solutions, we find that the insecure agent is less than half as likely to conduct effective planning and testing for the hidden security risks behind the functional requirements. Motivated by this, we develop SECUREVIBE, a training recipe that explicitly targets planning and testing for code security. SECUREVIBE constructs training signals around these security behaviors. It includes supervised fine-tuning on the security suite with 4 security tasks, and post-training methods, SECUREVIBE_rl and SECUREVIBE_hg, to enhance security capabilities from verifiable execution feedback and hint-based self-supervision. Our SECUREVIBE outperforms the baseline on two types of security coding tasks across 4 benchmarks. Specifically, SECUREVIBE improves 
    
[^136]: 超越理想化通信：面向误解与用户意图演化的交互式意图对齐基准测试

    Beyond Oracle Communication: Benchmarking Interactive Intent Alignment Under Miscommunication and Evolving User Intent

    [https://arxiv.org/abs/2609.38604](https://arxiv.org/abs/2609.38604)

    该论文提出了“交互式意图对齐”这一新任务设定，并构建了Drift-Bench++基准和GRIP评估协议，用于评测LLM智能体在用户沟通不完美、意图静默漂移且耐心有限等现实条件下恢复并持续追踪用户意图的能力。

    

    现代LLM智能体越来越多地通过与用户进行交互式、长程的多次交流来完成复杂任务，而现有基准测试通常假设用户总能准确且充分地传达一个固定意图。然而，这种“理想化通信（oracle communication）”假设在实践中很少成立：用户可能会错误表达、改变目标，甚至失去耐心。我们将这一任务设定定义为“交互式意图对齐”（Interactive Intent Alignment），即智能体必须在不完美通信和目标不断演化的情况下，恢复并持续追踪用户当前的真实意图。为研究该设定，我们提出了Drift-Bench++，一个具有原则性的基准构建流程，可生成经过验证的可执行任务，并带受控的意图错位与意图漂移；同时配备了一套交互协议，其特点包括有限耐心、多样化的模拟用户，以及“静默的、由交互触发的意图转变”。我们进一步开发了GRIP，一个全面的评估协议，涵盖任务落地、用户……（原文摘要在此处截断）

    arXiv:2609.38604v1 Announce Type: cross  Abstract: Modern LLM agents increasingly tackle complex tasks through interactive, long-horizon exchanges with users, while existing benchmarks generally assume that users always accurately and sufficiently communicate a fixed intent. However, this oracle communication assumption rarely holds in practice: users may miscommunicate, change their goals, and run out of patience. We define this task setting as Interactive Intent Alignment, where agents must recover and continuously track the user's current intent despite imperfect communication and evolving goals. To study this setting, we introduce Drift-Bench++, a principled benchmark construction pipeline for verified executable tasks with controlled misalignment and intent shifts, along with an interaction protocol featuring finite patience, diverse simulated users, and silent interaction-conditioned shifts. We further develop GRIP, a comprehensive evaluation protocol covering task grounding, use
    
[^137]: Prompt2Skill：从自然语言指令中进行无监督技能优化

    Prompt2Skill: Unsupervised Skill Optimization From Natural Language Instructions

    [https://arxiv.org/abs/2609.38593](https://arxiv.org/abs/2609.38593)

    提出了Prompt2Skill框架，仅需自然语言任务描述即可自动构建并优化供大语言模型使用的技能，无需整理的训练数据或昂贵的专家编写流程，从而解决了技能制作成本高、未针对特定模型优化以及新兴任务缺乏技能库覆盖的问题。

    

    技能是大型语言模型（LLM）在推理时使用的外部产物，通过融入相关的程序性知识和领域知识来提升模型在专业领域的表现。由专家编写的技能制作成本高昂，且生成的产物并未针对使用它的特定模型进行优化，而模型的失败模式会随版本、规模和训练情况而变化。此外，新兴任务可能超出现有技能库的覆盖范围，这就需要在整理好的训练数据可用之前开发新技能。近期的工作探索了通过反思进行自动化技能优化的方法，但它们需要一个经过整理的、与任务分布一致的训练集，而用户并不总是拥有这样的数据。为解决这些局限性，我们提出了 Prompt2Skill，这是一个仅凭自然语言任务描述就能构建技能的框架。系统从提示词中推导出任务规范，并发现或合成……（原文摘要至此中断）

    arXiv:2609.38593v1 Announce Type: cross  Abstract: Skills are external artifacts that Large Language Models (LLMs) consume at inference time to improve their performance on specialized domains by incorporating relevant procedural and domain knowledge. Expert-authored skills are expensive to produce, and the resulting artifacts are not optimized for the specific model that consumes them, whose failure modes can vary with version, scale and training. In addition, emerging tasks may fall outside the scope of existing skill libraries, creating a need to develop new skills before curated training data become available. Recent works have explored automated skill optimization through reflection, but they require a curated, in-distribution training set, which users might not always have. To address these limitations, we present Prompt2Skill, a framework that builds skills from natural-language task description alone. From the prompt, the system derives a task specification, discovers or synthe
    
[^138]: 迈向模型即图书馆：面向低资源非洲语言的离线社区来源人工智能

    Towards Model as a Library: Offline, Community-Sourced AI for Low-Resource African Languages

    [https://arxiv.org/abs/2609.38574](https://arxiv.org/abs/2609.38574)

    提出“模型即图书馆”（MaaL）软件架构，将小型社区注册语音模型打包为设备端依赖，通过说话人部署时现场注册词汇的方式，为低资源非洲语言提供离线、无幻觉的结构化数据收集方案，克服大语言模型在方言和地区差异上的失真问题。

    

    大语言模型经常被提议作为为非洲社区提供人工智能驱动服务的途径，但在需求最迫切的地方，它们恰恰最不可靠：按照任何标准衡量，所有非洲语言都属于低资源语言，而且基于爬取的标准化文本训练的模型会系统性地错误呈现人们实际说话时的方言和地区差异。我们提出了**模型即图书馆（Model as a Library, MaaL）**，这是一种软件架构，它将小型的、由社区注册的语音模型打包为版本化的设备端依赖，从而为当前语言模型服务最差的人群提供不会产生生成式幻觉的离线结构化数据收集。MaaL不依赖网络爬取的语料库，其词汇表是在部署时由说话人本人通过少量示例录音直接注册生成的。我们描述了该架构及其核心机制——关键词检测技术，它能够将（摘要在此处截断）

    arXiv:2609.38574v1 Announce Type: new  Abstract: Large language models are frequently proposed as a route to AI-powered services for African communities, but they are least reliable exactly where the need is greatest: all African languages remain low-resource by any standard measure, and models trained on scraped, standardised text systematically misrepresent the dialectal and regional variation of how people actually speak. We introduce \textbf{Model as a Library (MaaL)}, a software architecture that packages small, community-enrolled speech models as versioned on-device dependencies, enabling offline structured data collection that cannot generatively hallucinate, for populations that current language models serve worst. Rather than relying on web-scraped corpora, MaaL's vocabulary is enrolled directly from a small number of example recordings by the speakers themselves, at the point of deployment. We describe the architecture and its central mechanism - keyword spotting that turns a
    
[^139]: MedKIT：评估大语言模型中的知识整合与泛化能力

    MedKIT: Evaluating Knowledge Integration and Generalization in Large Language Models

    [https://arxiv.org/abs/2609.38543](https://arxiv.org/abs/2609.38543)

    MedKIT 是一个医学知识整合与迁移基准，通过模拟真实临床知识更新序列，细粒度评估大语言模型整合、迁移和应用新知识的能力，并对 5 个模型上的 12 种知识整合策略进行了大规模实证研究。

    

    不断演变的现实世界知识要求模型必须持续更新。尤其是在医学领域，随着临床证据随时间变化，过时的知识可能带来安全风险。现有的知识整合评估主要关注事实回忆，对于新整合的知识是否真正可用缺乏深入洞察。我们的基准 MedKIT（医学知识整合与迁移）提供了对模型在现实临床更新序列下如何整合和应用知识的细粒度评估。每个实例对应一个源自临床证据的事实更新，并配有针对性的探测任务，评估知识在词汇变化、关系转换、组合推理和开放式操作化等方面的迁移能力，同时包含检验知识保留的局部性测试。利用 MedKIT，我们对 5 个不同模型上的 12 种知识整合策略开展了大规模实证研究。

    arXiv:2609.38543v1 Announce Type: new  Abstract: Constantly evolving real-world knowledge necessitates models to be updated continuously. Especially in medicine, as clinical evidence changes over time, outdated knowledge can pose safety risks. Existing evaluations of knowledge integration focus on factual recall, offering limited insight into whether newly integrated knowledge is actually usable. Our benchmark MedKIT (Medical Knowledge Integration and Transfer) provides a granular evaluation of how models integrate and apply knowledge under realistic sequences of clinical updates. Each instance corresponds to a factual update derived from clinical evidence, paired with targeted probes that assess transfer across lexical variation, relational transformations, compositional reasoning, and open-ended operationalization, as well as locality tests for knowledge preservation. Using MedKIT, we conduct a large-scale empirical study of 12 knowledge integration strategies across 5 diverse models
    
[^140]: 机制转变：位置编码选择如何塑造上下文内检索

    Shifting Mechanisms: How Positional Encoding Choice Shapes In-Context Retrieval

    [https://arxiv.org/abs/2609.38530](https://arxiv.org/abs/2609.38530)

    该论文通过机制分析发现，位置编码的选择决定了语言模型进行上下文检索所依赖的内部机制——标准RoPE模型主要依赖位置检索，而将位置编码限制在局部层的混合架构（如SWA NoPE）则转向语义检索，这一转变带来长上下文收益的同时也隐藏着检索性能上的权衡。

    

    语言模型越来越多地采用在不同层之间改变注意力跨度和位置编码的架构，例如将RoPE与滑动窗口注意力结合、将NoPE与全局注意力结合（SWA NoPE）。然而，这些选择如何影响上下文内检索仍不清楚。为研究这一问题，我们采取机制性的视角，追踪位置编码（PE）的选择如何塑造模型用于上下文内检索的内部机制。在涵盖八个模型家族的22个开放权重模型中，我们发现标准的RoPE模型主要依赖位置检索，而PE混合模型则转向语义检索。我们进一步通过受控的预训练消融实验表明，将位置编码限制在局部层会产生这种向语义检索的转变，并降低位置信息的表征。最后，我们发现PE混合模型所报告的长上下文收益掩盖了一种检索权衡：SWA NoPE在多目标（原文在此处截断）……

    arXiv:2609.38530v1 Announce Type: new  Abstract: Language models increasingly use architectures that vary attention span and positional encoding across layers, such as applying RoPE with sliding-window attention and NoPE with global attention (SWA NoPE). However, how these choices shape in-context retrieval remains unclear. To study this question, we take a mechanistic view, tracing how positional encoding (PE) choice shapes the internal mechanisms models use for in-context retrieval. Across 22 open-weight models spanning eight families, we find that standard RoPE models rely primarily on positional retrieval, while PE hybrids shift toward semantic retrieval. We further show on a controlled pre-training ablation that confining positional encoding to local layers produces this semantic shift, degrading representations of positional information. Finally, we show that the reported long-context gains of PE hybrids mask a retrieval trade-off: SWA NoPE improves over RoPE on multiple-target r
    
[^141]: DEdit：面向推测解码的迭代草稿编辑

    DEdit: Iterative Draft Editing for Speculative Decoding

    [https://arxiv.org/abs/2609.38510](https://arxiv.org/abs/2609.38510)

    DEdit是一种基于扩散模型的推测解码起草器，通过token到token的迭代编辑让后续预测作为双向上下文来修复草稿中的早期错误，并配合基于置信度的ProposalMix训练方案，从而提高草稿接受率与解码加速效果。

    

    推测解码通过让一个轻量级的起草器提出候选token，再由目标模型并行验证，从而加速自回归大语言模型。基于扩散模型的起草器通过一次提出多个token进一步降低了起草延迟。然而，这些token是相互独立预测的，因此一个早期错误就会导致前缀验证丢弃草稿的其余部分，即使其中包含有用的下游预测。我们提出了DEdit，这是一种基于扩散模型的起草器，不仅能够通过传统的并行解掩码方式进行起草，还能通过token到token的预测对其草稿进行迭代编辑。通过编辑，后续的预测可以作为双向上下文，用于修复早期错误并扩展被接受的前缀。为了教会模型在保留正确预测的同时修复错误，我们提出了ProposalMix，这是一种在训练期间基于首轮置信度将草稿预测与真实token混合的训练方案。

    arXiv:2609.38510v1 Announce Type: new  Abstract: Speculative decoding accelerates autoregressive LLMs by having a lightweight drafter propose tokens that the target model verifies in parallel. Diffusion-based drafters further reduce drafting latency by proposing multiple tokens at once. However, these tokens are predicted independently, so a single early error causes prefix verification to discard the rest of the draft, even when it contains useful downstream predictions. We introduce DEdit, a diffusion-based drafter that can not only draft by conventional parallel unmasking but also iteratively edit its draft through token-to-token predictions. Through editing, later predictions can serve as bidirectional context for repairing earlier errors and extending the accepted prefix. To teach the model to repair errors while preserving correct predictions, we propose ProposalMix, a training scheme that mixes draft predictions with ground-truth tokens based on first-pass confidence during trai
    
[^142]: 面向临床智能体的个性化状态转换感知记忆

    Personalized State-Transition-Aware Memory for Clinical Agents

    [https://arxiv.org/abs/2609.38490](https://arxiv.org/abs/2609.38490)

    提出STAM框架，在新临床记录到来时通过状态转换感知机制将记忆划分为“活动”与“历史”两层，并结合查询依赖门控在需要时选择性调用历史记忆，使临床LLM智能体既能追踪患者当前状态又不会丢失临床历史证据。

    

    对临床记录进行推理的大语言模型（LLM）智能体必须跟踪患者状态的变化，同时保留理解这些变化所需的历史信息。简单地累积记忆会导致哪些信息仍然适用变得不明确，而覆盖较早的记忆则可能抹去重建治疗历史和临床轨迹所需的证据。我们提出了STAM，这是一种状态转换感知的记忆框架，能够在新临床记录到来时记录状态变化。STAM将语义检索与类型化临床关系相结合以识别受影响的记忆，将当前信息维护在Active（活动）层中，而将已被取代或已解决的信息维护在History（历史）层中。在读取时，一个依赖查询的门控机制会选择性地提供历史记忆。我们在四个纵向临床基准上，通过下游问答任务、直接状态维护诊断以及近似匹配上下文长度下的比较来评估STAM。

    arXiv:2609.38490v1 Announce Type: cross  Abstract: Large language model (LLM) agents that reason over clinical records must track changes in a patient's state while preserving the history needed to understand them. Simply accumulating memories leaves it unclear which information still applies, whereas overwriting earlier memories can erase evidence needed to reconstruct treatment history and clinical trajectories. We introduce STAM, a state-transition-aware memory framework that records state changes as new clinical entries arrive. STAM combines semantic retrieval with typed clinical relations to identify affected memories, maintaining current information in Active and superseded or resolved information in History. At read time, a query-dependent gate selectively serves historical memory. Across four longitudinal clinical benchmarks, we evaluate STAM with downstream question answering, direct state-maintenance diagnostics, and comparisons at approximately matched context lengths.
    
[^143]: 大语言模型时代的拟人化：潜在风险与缓解措施综述

    Anthropomorphism in the age of Large Language Models: An overview of potential risks and mitigations

    [https://arxiv.org/abs/2609.38486](https://arxiv.org/abs/2609.38486)

    本文系统综述了大语言模型时代的AI拟人化现象，提出了一个包含21项关注点、覆盖认知、情感、人类能动性、规范性和社会制度五大类别的拟人化风险分类法，并将其与设计、传播、教育等方面的缓解干预措施相关联。

    

    大语言模型（LLM）以及更广泛的人工智能（AI）系统常常被用类人的术语来描述和理解，这种现象被称为“拟人化”。本文对人工智能拟人化领域的近期文献进行了综述，涵盖理论框架、语言在将AI塑造为类人形象中所起的作用、机器拟人化的各种风险，以及缓解这些问题的策略。在考察了我们为何倾向于将AI系统拟人化以及这样做是否合理之后，本文重点分析了语言框架对拟人化的影响。随后，作者提出了一个与AI拟人化相关的风险概念分类法，将21项关注点归入五个分析类别：认知风险、情感风险、人类能动性风险、规范性风险以及社会和制度风险。最后，本文将这些关注点与设计、传播、教育等领域提出的干预措施相关联。

    arXiv:2609.38486v1 Announce Type: cross  Abstract: Large Language Models (LLMs) and more broadly Artificial Intelligence (AI) systems are often described and understood in human-like terms, a phenomenon known as \emph{anthropomorphism}. This paper provides a synthesis of recent literature on anthropomorphism in AI, covering theoretical frameworks, the role of language in framing AI as human-like, the various risks of anthropomorphizing machines, and strategies to mitigate these issues. After examining why we tend to anthropomorphize AI systems and whether we are right to do so, we highlight the impact of linguistic framing on anthropomorphism. Then, we introduce a conceptual taxonomy of risks associated with AI anthropomorphism. This taxonomy groups twenty-one concerns within five analytical categories: epistemic, affective, human agency, normative, and societal and institutional risks. Finally, we relate these concerns to proposed interventions in design, communication, education, and
    
[^144]: KlinikeBench：超越诊断准确性的语言模型评估

    KlinikeBench: Evaluating Language Models Beyond Diagnostic Accuracy

    [https://arxiv.org/abs/2609.38480](https://arxiv.org/abs/2609.38480)

    KlinikeBench是一个包含333个由临床医生编写的任务的基准，通过沙盒环境中的虚拟患者交互，评估语言模型在信息收集和临床评估方面超越单纯诊断准确性的综合临床能力。

    

    大多数临床基准测试使用完整的病例描述来评估语言模型（LM）的诊断能力。然而在临床实践中，患者以不同的方式呈现信息，临床医生必须获取相关病史并确定需要进行哪些检查，才能做出诊断。因此，仅凭诊断准确性无法判断智能体是否收集了必要的信息或进行了适当的临床评估。此外，现有基准缺乏专业临床医生的验证。为了填补这一空白，我们推出了KlinikeBench，这是一个包含333个由临床医生编写的任务的基准，每个任务都提供了一个隔离的沙盒环境，其中包含虚拟患者、临床工具和针对特定任务的成功标准。超过35名临床医生参与了病例编写和基准评估。在一项实证研究中，临床医生对模拟对话的平均质量评分高于参考对话，这表明……

    arXiv:2609.38480v1 Announce Type: cross  Abstract: Most clinical benchmarks evaluate language models (LMs) on diagnosis using complete case descriptions. In clinical practice, however, patients present information in different ways, and clinicians must obtain relevant history and determine which examinations are needed before reaching a diagnosis. Diagnostic accuracy alone therefore cannot establish whether an agent gathered essential information or conducted an appropriate clinical assessment. Furthermore, existing benchmarks lack professional clinicians' verification. To address this gap, we introduce KlinikeBench, a benchmark of 333 clinician-authored tasks, each providing an isolated sandbox environment with a virtual patient, clinical tools, and task-specific success criteria. More than 35 clinicians contributed to case authoring and benchmark evaluation. In an empirical study, clinicians gave simulated dialogues higher mean quality ratings than reference conversations, which is a
    
[^145]: BACKDROP：揭示智能体周遭环境世界让它付出的代价

    The Backdrop Exposes What the World Around an Agent Costs It

    [https://arxiv.org/abs/2609.38469](https://arxiv.org/abs/2609.38469)

    BACKDROP基准通过在智能体执行环境中植入权威覆盖、提示注入、边界越权和写入故障四种日常干扰，并保持任务指令不变，揭示了智能体性能在动态真实环境中从69.5%骤降至31.3%的严重衰减。

    

    智能体基准测试是在静止不变的世界中评测智能体，而实际部署的智能体却工作在一个他人也会不断改变的世界里。比如有人发短信让智能体把钱转去别处，或者订单确认信息要求它回复门禁码。我们提出BACKDROP，用以探究智能体在纯净环境中的能力有多少能在这样的环境中留存。BACKDROP以任务及智能体的执行环境为基础，在其世界中植入四种日常危害，包括一次一个以及全部同时出现，而指令和正确的最终状态保持不变。每种危害提出一个问题：权威性，来自他人的消息是否会覆盖用户的指令；注入，植入记录中的文本是否会误导智能体；边界，某个请求是否会将它引入未被授权的应用；故障，当一次写入操作失败且未说明是否成功落地时，智能体是否会在重试前先进行检查。在3,678个变体和16个模型上，平均通过率从69.5%下降至31.3%。

    arXiv:2609.38469v1 Announce Type: cross  Abstract: Agent benchmarks test agents in worlds that stay still. Deployed agents work in worlds that other people also change. Someone texts the agent to send the money elsewhere or an order confirmation asks it to reply with a door code. We present BACKDROP, which asks how much of an agent's capability in a clean world survives in such a world. BACKDROP takes a task along with the agents execution environment, and plants four everyday hazards in its world, one at a time and all together. The instruction and the correct end state stay the same. Each hazard asks one question. Authority: does a message from another person override the user? Injection: does text planted in a record redirect the agent? Boundary: does a request pull it into an app it was not given? Fault: after a write fails without saying whether it landed, does the agent check before it retries? Across 3,678 variants and 16 models, , the average pass rate falls from 69.5% to 31.3%
    
[^146]: 探入CHOIR：自由列表引出法揭示LLM集成中各模型独特的声音

    Reach Into The CHOIR: Free-List Elicitation Uncovers Distinct Model Voices in LLM Ensembles

    [https://arxiv.org/abs/2609.38448](https://arxiv.org/abs/2609.38448)

    本文提出CHOIR框架，将认知人类学中的自由列表引出法应用于LLM集成，通过反复引出排序回答、聚类概念并测量概念显著性，揭示了模型表面一致输出之下各自独特的“声音”，从而区分真实的多元性与虚假多元。

    

    开放式任务中的LLM同质化可能造成“虚假多元”：多个系统看似提供独立视角，实则返回同样熟悉的默认答案。单次回答模糊了以下几者之间的区别——由严格受限答案空间产生的一致、提示词汇的回声效应，以及表面之下拥有稳定备选答案的更广泛答案空间。我们提出CHOIR（集体分层有序询问响应，Collective Hierarchically-Ordered Inquiry Responses），一个将认知人类学中的自由列表引出法迁移至LLM集成的框架。CHOIR反复引出排序列表，将条目聚类为提示级概念，并跨模型、提示变体和角色条件测量概念显著性。我们在Infinity-Chat 100（一个来自近期开放式模型同质性研究的外部提示库）以及一个旨在分离机制层面差异的27题定向诊断库上对CHOIR进行评估。在Infinity-Chat 100上，CHOIR重现了高（摘要在此处截断）

    arXiv:2609.38448v1 Announce Type: new  Abstract: Open-ended LLM homogeneity can create false plurality when several systems appear to offer independent perspectives while returning the same familiar default. Single-pass answers obscure the distinction between agreement produced by a tightly constrained answer space, prompt-vocabulary echo, and broader answer spaces with stable alternatives beneath the surface. We introduce CHOIR (Collective Hierarchically-Ordered Inquiry Responses), a framework that adapts free-list elicitation from cognitive anthropology to LLM ensembles. CHOIR repeatedly elicits ranked lists, clusters items into prompt-level concepts, and measures concept salience across models, prompt variants, and persona conditions. We evaluate CHOIR on Infinity-Chat 100, an external prompt bank from recent work on open-ended model homogeneity, and on a 27-question targeted diagnostic bank designed to isolate mechanism-level contrasts. On Infinity-Chat 100, CHOIR reproduces high s
    
[^147]: 预训练与中期训练使奖励能够学到什么？

    What Pretraining and Midtraining Make Learnable from Rewards?

    [https://arxiv.org/abs/2609.38446](https://arxiv.org/abs/2609.38446)

    该论文证明了预训练与中期训练通过源预测使模型习得执行或检索等通用计算能力，而奖励适应只学习这些能力的任务特定用法，并通过有限采样Adam路径的理论构造与Qwen2.5实验验证了这一分工机制。

    

    奖励能够识别正确答案，却未确定处理新输入所需的计算。我们研究预训练与中期训练如何提供使奖励适应得以有效运作的信息与计算。在序列状态计算与上下文记忆任务中，我们刻画了在所有训练奖励上表现一致、却在留出问题上要求不同答案的机制，而与任务无关的源观测化解了这一歧义。我们构造了从指定随机初始化出发、在同一组参数内经由源预测与奖励适应的有限采样Adam路径，证明了预测如何习得执行或检索能力，以及奖励如何学习其任务特定的用法。使用预训练Qwen2.5检查点的实验验证了这种分工：在八个世界环境中，采用正确源监督与首操作监督训练的Sequential模型达到82.61%的成功率，而私有随机源对照组仅为44.15%。记忆重放……（原文摘要在此处截断）

    arXiv:2609.38446v1 Announce Type: cross  Abstract: A reward can identify a correct answer while leaving the computation needed for new inputs undetermined. We study how pretraining and midtraining supply the information and computation that make reward adaptation effective. In sequential state computation and contextual memory, we characterize mechanisms that agree on every training reward yet demand different held-out answers. Task-independent source observations resolve this ambiguity. We construct finite sampled Adam paths from specified random initializations through source prediction and reward adaptation in the same parameters, proving how prediction acquires execution or retrieval and rewards learn their task-specific use. Experiments with pretrained Qwen2.5 checkpoints test this division of labor. Across eight worlds, Sequential models trained with correct source and first-operation supervision reach 82.61% success, versus 44.15% for a private-random source control. Memory repl
    
[^148]: 策略条件化的AI使用检测：面向学术出版的证据框架

    Policy-Conditioned AI-Use Detection: An Evidentiary Framework for Academic Publishing

    [https://arxiv.org/abs/2609.38427](https://arxiv.org/abs/2609.38427)

    该论文提出“策略条件化AI使用检测”的证据框架，将传统上判断文本是否由AI撰写的检测任务，转变为评估人-AI工作流是否遵守学术出版机构具体AI使用政策的证据推理与合规性评估。

    

    目前主要的学术会议和期刊已发布了关于作者、审稿人和领域主席如何使用AI的详细规则，这些规则因角色、任务以及必须披露的内容而各不相同。而通常被提议用于执行这些规则的工具——AI检测，所估计的却是另一回事：文本是否由AI模型撰写。我们认为这一目标与会议和期刊面临的实际决策并不一致，因此提出“策略条件化的AI使用检测”，这是一个用于评估人-AI工作流是否遵守既定规则的证据框架。在该框架中，策略使治理规则成为显式的输入；推理过程报告的是假设、证据、校准状态和不确定性，而非“检测到AI”之类的判定。评估方面，通过可复现的流程构建基准数据，生成合规与不合规的工作流，并在出版方预先设定的假阳性率下报告真阳性率。我们将该框架应用于同行评审场景，在给定假阳性率条件下……

    arXiv:2609.38427v1 Announce Type: cross  Abstract: Major venues now publish detailed rules about how authors, reviewers, and area chairs may use AI, and those rules differ by role, by task, and by what must be disclosed. AI detection, the instrument usually proposed to enforce them, estimates something else: whether an AI model wrote the text. We argue that this target is misaligned with the decisions conferences and journals face, and propose policy-conditioned AI-use detection, an evidentiary framework for assessing whether a human--AI workflow complied with a stated rule. Policy makes the governing rule an explicit input. Inference reports hypotheses, evidence, calibration regime, and uncertainty in place of verdicts such as "AI detected". Evaluation builds benchmarks from reproducible pipelines that generate compliant and non-compliant workflows, and reports true positive rate at a false positive rate the venue fixes in advance. We work the framework through peer review, where at p
    
[^149]: LoopVL：循环视觉智能

    LoopVL: Recurrent Visual Intelligence

    [https://arxiv.org/abs/2609.38426](https://arxiv.org/abs/2609.38426)

    LoopVL将Loop Transformer成功扩展至视觉-语言模型，通过模块循环与模型循环计算迭代更新统一的视觉-语言状态，在多模态理解与视觉推理上超越同等及更大规模的非循环模型，并展现出视觉注意力显著转移的“视觉顿悟时刻”。

    

    我们提出LoopVL，以研究Loop Transformer能否被有效扩展到视觉-语言模型中。LoopVL将Module-Loop（模块循环）与Model-Loop（模型循环）计算相结合，通过共享模块迭代更新一个统一的视觉-语言状态。我们通过语言预训练、多模态训练和后训练从零开始训练LoopVL。在多模态理解和视觉推理基准测试中，LoopVL超越了众多规模相近乃至更大的非循环模型。我们还在LoopVL中观察到了“视觉顿悟时刻”（Visual Aha Moments），其特征是视觉注意力在不同循环之间发生显著转移。LoopVL为循环视觉-语言建模提供了实践证据，并从一个直观的视角展示了共享参数如何在持续演化的视觉-语言状态上支持更深层次的多模态计算。

    arXiv:2609.38426v1 Announce Type: cross  Abstract: We introduce LoopVL to study whether Loop Transformers can be effectively extended to vision- language models. LoopVL combines Module-Loop and Model-Loop computation to iteratively update a unified vision-language state through shared modules. We train LoopVL from scratch through language pre-training, multimodal training, and post-training. LoopVL outperforms a range of similarly sized and larger non-recurrent models on multimodal understanding and visual reasoning benchmarks. We also observe Visual Aha Moments in LoopVL, characterized by pronounced shifts in visual attention across loops. LoopVL provides practical evidence for recurrent vision-language modeling and offers an intuitive perspective on how shared parameters can support deeper multimodal computation over continuously evolving visual-language states.
    
[^150]: 所言而非所“想”：用于思维链验证的Type-6逻辑

    What Was Said, Not What Was 'Thought': Type-6 Logic for CoT Verification

    [https://arxiv.org/abs/2609.38420](https://arxiv.org/abs/2609.38420)

    该论文提出Type-6逻辑（带不确定性与递归算子的动态认知逻辑变体）及其图结构验证器，可检测出表层启发式方法遗漏的LLM思维链推理中的结构性缺陷，并支持推理过程的可视化。

    

    我们引入了Type-6逻辑，这是动态认知逻辑的一种变体，增强了两个算子（不确定性与递归），旨在对当代大语言模型（LLM）思维链（CoT）推理的推理动态进行建模。Type-6能够刻画常见的LLM推理病理，如未经许可的修订、省略三段论、循环回溯以及不可验证/错误的断言。我们提出了一种基于Type-6逻辑的验证器，它根据推理轨迹构建图，并依据Type-6的公理和推理规则对其进行检查。我们在横跨形式推理与非形式推理的四个数据划分上，对LLM生成的CoT评估了该框架。我们的验证器能够检测出表层启发式方法所遗漏的结构性不健全的推理步骤，并且便于对模型的推理过程进行可视化。在我们的语料库中，验证器表明推导出的矛盾是CoT中最常见的硬性失败类别，且轨迹中仅有约3%的命题具有……（原文在此截断）

    arXiv:2609.38420v1 Announce Type: cross  Abstract: We introduce Type-6 logic, a variant of dynamic epistemic logic augmented with two operators (uncertainty and recurrence), designed to model the inferential dynamics of contemporary large language model (LLM) chain-of-thought (CoT) reasoning. Type-6 accounts for common LLM reasoning pathologies such as unlicensed revision, enthymemes, loopbacks, and unverifiable/incorrect claims. We propose a verifier based on Type-6 logic that builds a graph out the trace, and checks it against Type-6's axioms and inference rules. We evaluate our framework on LLM-generated CoTs four splits spanning formal and informal reasoning. Our verifier detects structurally unsound reasoning steps that surface-level heuristics miss, and allows for easy visualisation of the model's reasoning process. In our corpus, our verifier shows that derived contradiction is the most common hard-fail category in CoT, and that only about 3\% of the propositions of a trace have
    
[^151]: ArgGYM：一个用于结构化可废止推理的程序化、引擎验证基准

    ArgGYM: A Procedural, Engine-Verified Benchmark for Structured Defeasible Reasoning

    [https://arxiv.org/abs/2609.38409](https://arxiv.org/abs/2609.38409)

    ArgGYM是一个程序化生成、由引擎自动验证评分的结构化可废止推理基准与RLVR兼容训练环境，将可废止推理分解为十二个可自动评分的任务，弥补了现有数学、代码和逻辑基准之外的现实推理评估空白。

    

    近年来大语言模型推理能力的进展，主要由具有自动可验证奖励的基准测试和强化学习环境所推动，尤其是在数学、代码和形式逻辑领域。这些设定使模型准确率更容易评估和优化，但在固定问题规范和稳定评估标准下取得的成功，能在多大程度上迁移到这些领域之外的推理，仍不清楚。现实世界的推理常常在不完整且可修正的信息下进行：结论可能被暂时支持、被反证据击败、被进一步的论证所恢复，或在出现更有力的理由时被修正。这类推理通常被称为可废止推理。我们提出ArgGYM，一个面向结构化可废止推理的程序化基准测试以及与RLVR（可验证奖励强化学习）兼容的训练环境。ArgGYM将这种推理分解为十二个任务，并将任务特定的评分建立于（摘要在此处被截断）

    arXiv:2609.38409v1 Announce Type: new  Abstract: Recent progress in large language model reasoning has been driven by benchmarks and reinforcement learning environments with automatically verifiable rewards, particularly in mathematics, code, and formal logic. These settings make model accuracy easier to evaluate and optimize, but it remains unclear how far success under fixed problem specifications and stable evaluation criteria transfers to reasoning outside such domains. Real-world reasoning often proceeds under incomplete and revisable information: conclusions may be supported provisionally, defeated by counter-evidence, reinstated by further arguments, or revised when stronger reasons become available. Reasoning of this kind is generally referred to as defeasible reasoning. We introduce ArgGYM, a procedural benchmark and RLVR-compatible training environment for structured defeasible reasoning. ArgGYM decomposes this reasoning into twelve tasks and grounds task-specific scoring in 
    
[^152]: 评估大语言模型能否可靠地“连点成文”？

    Evaluating Whether LLMs Can Reliably Connect the DOTs?

    [https://arxiv.org/abs/2609.38406](https://arxiv.org/abs/2609.38406)

    该论文构建了一个包含约9200个实例、覆盖百科文本、常识故事、新闻文章和视觉叙事四种类型的多领域叙事填充基准，并系统评估了20个开源指令微调大语言模型在真实世界叙事填充任务上的表现。

    

    获取真实世界的信息往往是嘈杂且碎片化的。要从这些碎片中构建出连贯的叙事，需要模型在更宏大的故事情节中重建缺失的片段，这通常被称为文本填充，同时还要保持与局部上下文和全局故事情节的一致性。尽管许多大型语言模型将文本填充作为预训练目标，但它们在真实世界叙事填充任务上的实际表现仍未得到充分探索。本文通过引入一个包含约9200个实例的多领域叙事填充基准来填补这一研究空白，该基准通过在四种叙事类型中遮蔽一到三句话构建而成，这四种类型包括百科式文本、常识故事、新闻文章和视觉叙事。利用这一基准，我们评估了20个参数量从15亿到700亿不等的开源指令微调大语言模型，涵盖了不同程度的指令具体性和推理引导。输出结果

    arXiv:2609.38406v1 Announce Type: new  Abstract: Access to real-world information is often noisy and fragmented. Constructing a coherent narrative from such fragments requires models to reconstruct missing spans within a broader storyline, commonly referred to as text infilling, while preserving consistency with both the local context and the global storyline. Despite using text infilling as a pre-training objective in many Large Language Models (LLMs), their actual performance on real-world narrative infilling remains underexplored. In this paper, we address this gap by introducing a multi-domain benchmark of ~9.2K instances for narrative infilling, constructed by masking one to three sentences across four narrative types: encyclopedic text, commonsense stories, news articles, and visual narratives. Using this benchmark, we evaluate 20 instruction-tuned open-source LLMs ranging from 1.5B to 70B parameters across varying levels of instruction specificity and reasoning guidance. Outputs
    
[^153]: 多轮攻击中有害性的几何结构

    The Geometry of Harmfulness in Multi-Turn Attacks

    [https://arxiv.org/abs/2609.38389](https://arxiv.org/abs/2609.38389)

    该论文通过分析三个指令微调大模型在三种多轮攻击框架下的隐藏状态表示，首次揭示了有害性与拒绝表示在多轮攻击过程中的几何结构与时间动态演化规律，从而解释了单轮防御在多轮攻击中失效的原因。

    

    arXiv:2609.38389v1 公告类型：cross  摘要：大型语言模型（LLMs）仍然容易受到规避安全对齐以诱导有害输出的对抗性攻击。目前尚不清楚在多轮攻击过程中，有害性与拒绝表示如何演变，以及为什么单轮防御在多轮场景中效果较差。本工作研究了多轮攻击中有害性与拒绝表示的几何结构和时间动态如何演变。我们使用三种多轮攻击框架分析了三个指令微调LLM（Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct和Gemma-2-9B-it）的隐藏状态表示，并在各种上下文配置下考察了表示在对话轮次、模型层和token位置上的行为。跨模型和框架，我们发现：(1) 每个攻击框架 traverses 不同的几何方向，但各自达到相当的可……（原文此处截断）

    arXiv:2609.38389v1 Announce Type: cross  Abstract: Large language models (LLMs) remain vulnerable to adversarial attacks that circumvent safety alignment to elicit harmful outputs. It remains unclear how harmfulness and refusal representations evolve over the course of multi-turn attacks, and why single-turn defenses are less effective in multi-turn settings. This work investigates how the geometry and temporal dynamics of harmfulness and refusal representations evolve across multi-turn attacks. We analyzed hidden-state representations from three instruction-tuned LLMs (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Gemma-2-9B-it) using three multi-turn attack frameworks (Crescendo, ActorAttack, and X-Teaming), and examined representation behavior across conversation turns, model layers, and token positions under various context configurations. Across models and frameworks, we found that (1) each attack framework traverses different geometric directions, yet each achieves comparable s
    
[^154]: 基于上下文选择与目标加权的扩散语言模型微调

    Fine-Tuning Diffusion Language Models with Context Selection and Target Weighting

    [https://arxiv.org/abs/2609.38385](https://arxiv.org/abs/2609.38385)

    GoldiMask 通过基于次模目标智能选择上下文 token 并对预测目标进行加权，显著提升了离散扩散语言模型在推理和代码生成任务上的微调效果。

    

    离散扩散语言模型的监督微调会屏蔽部分响应 token，并训练模型从可见上下文中恢复其原始值。因此，屏蔽模式既决定了模型可用的上下文，也决定了模型学习预测的 token。均匀随机屏蔽并未显式考虑这两种选择之间的相互作用。我们提出了 GoldiMask，它通过近似最大化一个次模目标来选择要揭示为上下文的 token。该目标利用模型信号来平衡揭示 token 的收益与其作为预测目标的价值。随后，GoldiMask 根据剩余目标从所选上下文中获得的收益及其剩余学习潜力对它们进行加权。在三个骨干模型和三个训练数据集上的实验表明，GoldiMask 在大多数评估设置中取得了最高的平均准确率，在推理和代码生成任务上均展现出性能提升。

    arXiv:2609.38385v1 Announce Type: new  Abstract: Supervised fine-tuning of discrete diffusion language models masks some response tokens and trains the model to recover their original values from the visible context. The masking pattern therefore determines both the context available to the model and the tokens it learns to predict. Uniform random masking does not explicitly account for the interaction between these choices. We introduce GoldiMask, which selects tokens to reveal as context by approximately maximizing a submodular objective. This objective uses model signals to balance the benefit of revealing tokens against their value as prediction targets. GoldiMask then weights the remaining targets according to how they benefit from the selected context and their remaining learning potential. Across three backbones and three training datasets, GoldiMask achieves the highest average accuracy in most evaluated settings, demonstrating gains on both reasoning and code generation. Compo
    
[^155]: Doc2LoRA 为科学思想提供可解码的表示

    Doc2LoRA Provides Decodable Representations of Scientific Ideas

    [https://arxiv.org/abs/2609.38374](https://arxiv.org/abs/2609.38374)

    提出 Doc2LoRA 方法，通过超网络将每篇科学论文表示为 LoRA 适配器，使向量空间中的任意点（包括论文混合产生的点）都能解码为可对话的大语言模型，从而实现对科学思想的可解码、可生成式表示。

    

    将科学论文表示为空间中的点，使我们能够搜索相似论文，并探究各领域之间如何相互关联以及如何推动创新。超越搜索之外，论文的向量空间还孕育了生成能力：通过简单的向量运算混合论文可以创造新的点，这反映了组合式创新——即现有思想重新组合成新思想的过程。然而，混合点通常代表一种尚未有论文实现的想法，且附近没有论文可以帮助识别这一想法。我们提出用 Doc-to-LoRA 超网络生成的 LoRA 适配器来表示每篇论文。因此，空间中的每个点（包括混合点）都代表一个可通过自然语言进行提问和指令交互的大语言模型（LLM）。在美国物理学会（APS）的论文上，我们指示每个子领域平均值处的 LLM 用几个词命名该领域，所获得的标签比五个基线方法的标签更接近官方名称。

    arXiv:2609.38374v1 Announce Type: new  Abstract: Representing scientific papers as points in a space lets us search for similar papers and inquire about how fields relate to one another and drive innovation. Beyond search, the vector space of papers invites generation: mixing papers through simple vector operations creates new points, mirroring combinatorial novelty, the recombination of existing ideas into new ones. However, a mixed point often represents an idea no paper has yet realized, with no papers nearby to identify the idea. We propose representing each paper by a LoRA adapter generated by the Doc-to-LoRA hypernetwork. Every point in the space, including mixtures, thus represents a large language model (LLM) open to questions and instructions in natural language. On papers from the American Physical Society (APS), we instruct the LLM at the average of each subfield to name the field in a few words and obtain labels closer to the official names than the labels of five baselines
    
[^156]: TALK-Dem：痴呆症相关交流模式下的具身任务规划基准测试

    TALK-Dem: Benchmarking Embodied Task Planning under Dementia-Associated Communication Patterns

    [https://arxiv.org/abs/2609.38371](https://arxiv.org/abs/2609.38371)

    该论文提出了首个基准 TALK-Dem，用于评估 LLM 驱动的机器人任务规划在应对痴呆症患者典型交流模式（如指代不精确、空洞言语、话题漂移等）时的表现，实验显示开源模型的性能下降高达 22.3%，暴露出显著的鲁棒性差距。

    

    现有的基于大语言模型（LLM）驱动的机器人任务规划器依赖于一个理所当然的假设：用户是理想化的，其指令清晰、完整且专注于任务本身。然而，在与现实世界的用户交互时，尤其是与存在认知障碍的用户（如痴呆症患者，PLWD）交互时，这些规划器经常出错，甚至可能带来人身安全风险。我们提出了 TALK-Dem（痴呆症中的言语属性与语言知识），这是首个用于在痴呆症相关言语交流场景下评估 LLM 驱动的机器人任务规划的基准。TALK-Dem 包含 4,800 条指令，涵盖五种典型的交流模式，包括指代不精确、物体替换、空洞言语、话题漂移和插入干扰，并设有三个强度等级。在六个开源权重 LLM 上进行的实验揭示了显著的鲁棒性差距。在各种交流模式下，开源权重模型的性能下降高达 22.3%。

    arXiv:2609.38371v1 Announce Type: cross  Abstract: Existing LLM-driven robot task planners rely on a taken-for-granted assumption of an ideal user whose instructions are clear, complete, and task-focused. However, when interacting with real-world users, especially those experiencing cognitive impairments, such as people living with dementia (PLWD), the planners often make mistakes and even pose physical safety risks. We proposed TALK-Dem (Talking Attributes and Linguistic Knowledge in Dementia), the first benchmark for evaluating LLM-driven robot task planning under dementia-associated verbal communication. TALK-Dem contains 4,800 instructions and covers five typical communication patterns, including Referential Imprecision, Object Substitution, Empty Speech, Topic Drift, and Intrusion, at three intensity levels. Experiments across six open-weight LLMs reveal a substantial robustness gap. Across communication patterns, open-weight models exhibited performance drops of up to 22.3 percen
    
[^157]: 论在线策略蒸馏中的离策略教师问题

    On the Off-Policy Teacher in On-Policy Distillation

    [https://arxiv.org/abs/2609.38360](https://arxiv.org/abs/2609.38360)

    论文揭示了在线策略蒸馏中教师模型面临离策略不对称性问题——教师对学生生成的前缀续写能力随前缀变长而下降，并提出SCOUT共同训练框架，通过可验证奖励的强化学习周期性优化教师以适应学生生成的前缀。

    

    在线策略蒸馏近来已成为一种有前景的训练后范式，其中学生模型在教师的密集监督下，从由自身策略生成的轨迹中学习。然而，OPD 引入了一个根本性的不对称性：尽管采样的轨迹对学生模型而言是在线的，但对教师模型而言却是离线的。教师模型通常被优化为从其自身策略生成的前缀继续生成内容，但在 OPD 过程中，它必须转而监督由学生模型生成的前缀。通过实证研究，我们发现随着这些前缀长度的增加，教师的续写性能会逐渐下降。为了解决这一问题，我们提出了学生条件化教师更新（SCOUT），这是一个让教师模型适应学生生成前缀的共同训练框架。除了标准的 OPD 更新外，SCOUT 还会周期性地使用基于可验证奖励的强化学习来优化教师的条件生成能力，其中教师生成……

    arXiv:2609.38360v1 Announce Type: cross  Abstract: On-policy distillation (OPD) has recently emerged as a promising post-training paradigm in which the student learns from trajectories generated by its own policy under dense teacher supervision. However, OPD introduces a fundamental asymmetry: although the sampled trajectories are on-policy for the student, they are off-policy for the teacher. The teacher is typically optimized to continue from prefixes generated by its own policy, but during OPD it must instead supervise prefixes generated by the student. Empirically, we find that its continuation performance degrades as these prefixes grow longer. To address this issue, we propose Student-COnditioned Updates of the Teacher (SCOUT), a co-training framework that adapts the teacher to student-generated prefixes. Alongside standard OPD updates, SCOUT periodically optimizes the teacher's conditional ability using reinforcement learning with verifiable rewards, where the teacher generates 
    
[^158]: 超越模式崩溃：利用生成流网络生成多样化的合成专家对话

    Beyond Mode Collapse: Generating Diverse Synthetic Expert Conversations via Generative Flow Networks

    [https://arxiv.org/abs/2609.38359](https://arxiv.org/abs/2609.38359)

    提出基于生成流网络的合成数据生成方法，按专家策略在训练数据中的分布比例采样潜在对话结构，从而生成多样化的高质量专家对话，在辅导和情感支持两个领域实现了比强化学习和端到端LLM基线更优的保真度、模式覆盖率与真实性平衡。

    

    高质量合成数据是大语言模型后训练的核心，可用于构建体现对话中多样化专家策略与决策的自适应AI应用。直接提示大语言模型或以终端使用场景为条件进行生成，会产生低多样性数据，使其坍缩到主导模式上。我们提出一种利用生成流网络生成多样化高质量合成数据的方法。我们证明，通过训练GFlowNets基于关键交互特征（如困惑情节动态、脚手架-指令平衡）上的高斯混合密度来生成潜在对话结构，能够按专家策略在训练数据中的普遍程度按比例对其进行采样。在两个结构迥异的领域——辅导对话与情感支持对话中，我们基于GFlow的合成数据生成方法相比强化学习和端到端大语言模型方法，在保真度、模式覆盖率和真实性之间取得了更好的平衡。

    arXiv:2609.38359v1 Announce Type: new  Abstract: High quality synthetic data is central to post training LLMs for adaptive AI applications that represent the diverse expert strategies and decisions in conversations. Prompting LLMs directly or conditioning them on end use scenarios yields low diversity data that collapses onto dominant modes. We propose a method to generate diverse high quality synthetic data using Generative Flow Networks (GFlowNets). We show that training GFlowNets to generate latent conversation structure using a Gaussian mixture density over key interaction features (e.g., confusion episode dynamics, scaffolding directive balance) enables sampling expert strategies in proportion to their prevalence in the training data. Across two structurally distinct domains, tutoring and emotional support dialogues, our GFlow based synthetic data generation approach offers a better balance of fidelity, mode coverage and authenticity than reinforcement-learning and end to end LLM 
    
[^159]: 评估语言模型在长对抗性对话中的安全性

    Evaluating Language Model Safety Across Long Adversarial Conversations

    [https://arxiv.org/abs/2609.38357](https://arxiv.org/abs/2609.38357)

    该研究让对抗性语言模型在长达百余轮的对话中持续发起攻击，发现模型的安全响应率从第一轮的85-100%骤降至第101轮的15-44%，证明单轮安全评估无法保证长对话中的安全性。

    

    对话式安全评估通常只用单个有害提示来测试语言模型，然而现实世界的系统是通过漫长且不断变化的对话与用户互动的。本研究考察当对抗性用户在多轮对话中持续纠缠时，模型是否仍能安全地响应。我们在不同的对话长度和随机种子下，针对两个有害提示评估了三个开放权重、经过指令微调的模型。在每种设置中，由第二个语言模型扮演持续进行对抗的用户，同时由一个安全分类器将每条响应标记为安全或不安全。在所有模型-提示组合中，第一轮的安全响应率介于85%至100%之间；到第11轮时降至38-61%；到第101轮时进一步降至15-44%。这种下降在所有模型中均出现，并且持续到远超通常用于多轮安全评估的短交互范围之外。这些结果提供了概念验证证据，表明强大的单轮安全……

    arXiv:2609.38357v1 Announce Type: new  Abstract: Conversational safety evaluations often test language models with a single harmful prompt, even though real-world systems interact with users through long, adaptive conversations. This study examines whether models continue to respond safely when an adversarial user persists across multiple turns. We evaluate three open-weight, instruction-tuned models on two harmful prompts across different conversation lengths and random seeds. In each setting, a second language model acts as a persistent adversarial user, while a safety classifier labels every response as safe or unsafe. Across all model-prompt combinations, first-turn safe-response rates ranged from 85% to 100%. By depth 11, they dropped to 38-61%, and by depth 101, to 15-44%. This decline appeared across models and continued well beyond the short interactions typically used in multi-turn safety evaluations. These results provide proof-of-concept evidence that strong single-turn safe
    
[^160]: HalluScoring 2026：首个大语言模型幻觉检测与答案验证共享任务

    Halluscoring 2026: The first shared task on llms hallucination detection and answer verification

    [https://arxiv.org/abs/2609.38355](https://arxiv.org/abs/2609.38355)

    这是首个针对阿拉伯语问答场景的大语言模型幻觉检测与事实答案验证共享任务，通过四个子任务评估系统对未见问题和未见模型的泛化能力，并基于HalluScore和HalluTruthQA两个数据集展开竞赛。

    

    我们推出 HalluScoring 2026，这是一个共享任务，旨在评估阿拉伯语问答中幻觉检测与事实验证在具有挑战性的泛化设置下的表现。该共享任务分为两个主要任务，每个任务包含两个子任务，共计四个子任务。任务1评估二元幻觉检测，考察对未见问题的泛化能力（子任务1.1）以及对未见大语言模型所生成回答的泛化能力（子任务1.2）。任务2将评估范围从检测扩展到验证，要求系统从六个相关候选答案中进一步识别出正确的事实性答案，涵盖伊斯兰知识（子任务2.1）和通用知识（子任务2.2）。该共享任务基于两个阿拉伯语数据集：HalluScore 和 HalluTruthQA。共有13支队伍参加了该共享任务，其中10支队伍提交了系统描述论文。任务1的结果表明，幻觉检测仍然是一项具有挑战性的任务。

    arXiv:2609.38355v1 Announce Type: new  Abstract: We present HalluScoring 2026, a shared task for evaluating hallucination detection and factual verification in Arabic question answering under challenging generalization settings. The shared task is organized into two main tasks, each comprising two subtasks, for a total of four subtasks. Task 1 evaluates binary hallucination detection, considering generalization to unseen questions (Subtask 1.1) and responses generated by unseen LLMs (Subtask 1.2). Task 2 extends the evaluation beyond detection by requiring the systems to additionally identify the correct factual answer from six related candidates, covering Islamic knowledge (Subtask 2.1) and general knowledge (Subtask 2.2). The shared task is based on two Arabic datasets: HalluScore and HalluTruthQA. A total of 13 teams participated in the shared task, 10 of which submitted system description papers. The results of Task 1 demonstrate that hallucination detection remains challenging und
    
[^161]: OpenCollab：一个具有可编程协作与可控运行时的多智能体编程框架

    OpenCollab: A Multi-Agent Coding Framework with Programmable Collaboration and Controllable Runtime

    [https://arxiv.org/abs/2609.38345](https://arxiv.org/abs/2609.38345)

    提出OpenCollab多智能体编程框架，通过统一组织设计、共享可控运行时和细粒度事件流追踪，并首次定义Adherence指标来量化协作组织结构的实际遵循程度，揭示任何单一配置变化都会使遵循度从47.2%大幅波动至97%以上。

    

    多智能体编程系统旨在通过协作来解决复杂的软件工程任务。然而，现有的评估通常假设所配置的组织结构会被忠实地遵循，而实际情况并非如此。这种行为差距，再加上底层系统组件的差异，使得观察到的性能提升难以进行明确的归因。为此，我们提出了OpenCollab，这是一个多智能体编程框架，为可编程协作与可控运行时提供了统一的基础设施。具体而言，OpenCollab统一了组织设计，在共享运行时上强制实施实验控制，并通过细粒度的事件流来跟踪执行过程。在此基础上，我们定义了Adherence（遵循度）指标，用于量化所声明的组织结构是否真正得以实现。我们的实验表明，智能体在不同配置下的协作方式差异很大：改变任何一个单一维度都会使Adherence发生变化，从47.2%到高达97%以上。

    arXiv:2609.38345v1 Announce Type: cross  Abstract: Multi-agent coding systems are designed to tackle complex software engineering tasks through collaboration. However, existing evaluations typically assume configured organizations are followed faithfully, whereas reality differs. This behavioral gap, combined with differences in underlying system components, prevents clear attribution of observed gains. To this end, we introduce OpenCollab, a multi-agent coding framework that provides a unified infrastructure for programmable collaboration and controllable runtime. Specifically, OpenCollab unifies organization design, enforces experimental control on a shared runtime, and tracks execution through fine-grained event streams. On this basis, we define Adherence to quantify whether the declared organization is actually realized. Our experiments reveal that agents collaborate very differently across configurations: changing any single dimension shifts Adherence, from 47.2% to as high as 97.
    
[^162]: EVOKE：在智能体中引出世界知识以实现可迁移的决策

    EVOKE: Eliciting World Knowledge in Agents for Transferable Decision-Making

    [https://arxiv.org/abs/2609.38334](https://arxiv.org/abs/2609.38334)

    该论文提出EVOKE后训练方法，通过在固定状态下施加目标多样性，迫使LLM智能体引出预训练中已内化的世界知识，从而提升其在未见环境中进行多步决策的迁移能力。

    

    arXiv:2609.38334v1 公告类型：new 摘要：大语言模型（LLM）越来越多地被部署为执行多步决策的智能体，但它们对未见环境的迁移能力较差。世界模型方法通过训练智能体预测未来观测来解决这一问题，但代价是需要额外的训练，而且当预测被用于规划时误差会不断累积复合。然而，对于在数字环境中运行的LLM智能体而言，许多这样的世界知识在预训练阶段就已经被内化，这使得问题从“获取知识”转变为“引出知识”。我们认为，典型的后训练对这种知识的引出几乎没有提供压力，因为在每个访问状态下基于单一目标的监督会无意中促使策略依赖表面的情境习惯。我们提出了EVOKE，这是一种后训练方法，通过在固定状态下引入目标多样性来施加这种压力。基于理论分析表明，一个在多种目标上均具备能力的智能体必须编码一个可恢复的世界模型……（摘要原文在此处截断）

    arXiv:2609.38334v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as agents for multi-step decision-making, yet transfer poorly to unseen environments. World-model methods address this by training agents to predict future observations, at the cost of additional training and errors that compound when predictions are used for planning. However, for LLM agents operating in digital environments, much of this world knowledge is already internalized during pretraining, which shifts the problem from acquiring it to eliciting it. We argue that typical post-training provides little pressure for such elicitation, since supervision under a single goal at each visited state inadvertently drives policies to rely on superficial contextual habits. We introduce EVOKE, a post-training method that supplies this pressure through goal diversity at fixed states. Motivated by theory showing that an agent competent across diverse goals must encode a world model recoverab
    
[^163]: Hermes：学习上下文推理解锁测试时扩展

    Hermes: Learning Contextual Reasoning Unlocks Test-Time Scaling

    [https://arxiv.org/abs/2609.38332](https://arxiv.org/abs/2609.38332)

    Hermes 将上下文窗口分配与信息复用的决策权从外部框架转移给模型本身，并通过两阶段的 Hermes-Learn 训练框架学习适应性上下文推理能力，使模型（尤其是此前做不到的较小开源模型）能够利用额外推理计算实现测试时扩展。

    

    测试时扩展通过在推理过程中分配额外的计算来提升模型性能。要在多个上下文窗口之间有效利用这些计算，需要决定如何分配新的上下文，以及在各上下文之间携带哪些信息。我们将模型做出这类决策的能力称为上下文推理。现有方法大多通过其框架预先规定这些决策，而我们则将其转移给模型本身。我们提出了：1）Hermes，一组简单、可配置的框架家族，逐步改变模型对上下文分配与复用的控制程度；2）Hermes-Learn，一个用于学习这些能力的两阶段框架。我们发现，能力较强的模型能够利用这种灵活性，借助额外的推理时计算实现性能扩展，而较小的开源模型起初难以做到。使用 Hermes-Learn 进行训练则弥合了这一差距，诱导出随问题和上下文而变化的适应性上下文推理策略。

    arXiv:2609.38332v1 Announce Type: cross  Abstract: Test-time scaling improves model performance by allocating additional compute during inference. Using this compute effectively across multiple context windows requires deciding how to allocate fresh contexts and what information to carry between them. We call a model's ability to make these decisions contextual reasoning. Existing approaches largely prescribe these decisions through their harness; we instead shift them to the model. We introduce 1) Hermes, a family of simple, configurable harnesses that progressively varies model control over context allocation and reuse, and 2) Hermes-Learn, a two-stage framework for learning these capabilities. We find that capable models can exploit this flexibility to scale with additional inference-time compute, while smaller open-source models initially struggle to do so. Training with Hermes-Learn closes this gap, inducing adaptive contextual reasoning strategies that vary with both the problem 
    
[^164]: 当异议被隐瞒时，多智能体讨论的收益会减少

    Multi-agent discussion gains less when dissent is withheld

    [https://arxiv.org/abs/2609.38324](https://arxiv.org/abs/2609.38324)

    该研究提出一个简约模型，指出只有当LLM智能体隐瞒异议的比率低于由净修正率与内化率共同决定的临界值时，多智能体讨论才能推翻错误的初始多数并提升准确性。

    

    多智能体LLM系统在多数投票的基础上加入了讨论机制，因此被期望具有更强的能力。然而，关于讨论究竟是提高了准确性还是导致了错误共识，实证研究的结果存在矛盾。在此，我们引入一个简约模型来解释讨论何时能提高准确性、何时会终结于错误共识。该模型建立在LLM智能体中反复观察到的四种行为之上：（1）隐瞒异议；（2）内化他人已陈述的答案；（3）在看到异议后重新思考；（4）向正确答案修正。模型表明，只有当隐瞒异议率 $c$ 低于临界值 $c^* = \gamma/(\gamma + a)$ 时，讨论才能推翻错误的初始多数，该临界值由净修正率 $\gamma$ 和内化率 $a$ 共同决定。我们采用贝叶斯方法从对话记录中估计这些参数，并将各LLM团队相对于 $c^*$ 进行定位。正如模型所预测，讨论带来的收益……（原文在此处截断）

    arXiv:2609.38324v1 Announce Type: cross  Abstract: Multi-agent systems of LLMs add discussion to majority voting and are therefore expected to be more capable. However, empirical reports conflict on whether discussion improves accuracy or leads to an incorrect consensus. Here, we introduce a parsimonious model that explains when discussion improves accuracy and when it ends in an incorrect consensus, built from four behaviors repeatedly observed in LLM agents: (1) withholding dissent, (2) internalizing a stated answer, (3) reconsidering after seeing dissent, and (4) correcting toward the correct answer. The model shows that discussion can overturn an incorrect initial majority only when the withholding rate $c$ is below a critical rate $c^* = \gamma/(\gamma + a)$, set by the net correction rate $\gamma$ and the internalization rate $a$. We estimate these rates from conversation logs with a Bayesian method and place LLM teams relative to $c^*$. As the model predicts, the gain from discu
    
[^165]: HARDE：面向运行时风险检测与执行控制的智能体框架优化

    HARDE: Optimizing Agent Harnesses for Runtime Risk Detection and Execution Control

    [https://arxiv.org/abs/2609.38291](https://arxiv.org/abs/2609.38291)

    提出HARDE两阶段优化框架，通过集成触发器、监控器和反馈模块的风险感知智能体框架，实现LLM智能体运行时风险的灵活检测与及时干预，在保障安全的同时维持良性任务的实用性。

    

    大型语言模型（LLM）智能体容易受到注入恶意指令或误导信息等安全风险的威胁，这促使人们需要运行时防御机制，在多种风险下防止不安全行为被执行，同时保持良性任务的实用性。现有的系统级防御要么专注于风险检测而非及时预防，要么依赖预定义规则，在面对多样化风险时灵活性有限。我们提出了一种风险感知框架，集成了基于LLM的监控以实现灵活的风险检测，并围绕触发器、监控器和反馈三个核心模块构建监控引导的执行机制，从而实现有针对性的安全干预，同时将对良性任务执行的干扰降至最低。为了使该框架适应不同的风险和部署环境，我们引入了HARDE，这是一个两阶段框架优化方法：首先对每个模块进行独立探测以得出优化指南，然后利用该指南……

    arXiv:2609.38291v1 Announce Type: cross  Abstract: Large language model (LLM) agents are vulnerable to safety risks such as injected malicious instructions or misleading information, motivating runtime defenses that prevent unsafe action in execution across diverse risks while preserving benign-task utility. Existing system-level defenses either focus on risk detection rather than timely prevention or rely on predefined rules with limited flexibility across diverse risks. We propose a risk-aware harness that integrates LLM-based monitoring for flexible risk detection and structures monitor-guided execution around three core modules: trigger, monitor, and feedback, enabling targeted safety interventions while limiting disruption to benign task execution. To adapt the harness to different risks and deployment settings, we introduce HARDE, a two-stage harness optimization framework that first performs isolated probing of each module to derive an optimization guide, then uses this guide to
    
[^166]: 哪些模型能良好协作？面向大语言模型团队选择的异构性度量

    Which Models Work Well Together? Measuring Heterogeneity for LLM Team Selection

    [https://arxiv.org/abs/2609.38274](https://arxiv.org/abs/2609.38274)

    提出了一种基于异构性的大语言模型团队选择框架，通过离线刻画个体能力并引入错误去相关性与预测行为分歧两种互补信号，将团队选择形式化为标准化的质量-互补性组合优化目标，并用高效贪心搜索从候选池中选出小规模团队。

    

    大语言模型团队性能的上限不仅受个体模型能力的约束，还受成员之间错误共振与预测差异的影响。尽管异构组队在实践中常被证明是有效的，但现有方法缺乏可计算、可解释且可优化的互补性度量，导致团队组合只能依赖启发式方法。我们提出了一个由异构性驱动的团队选择框架，该框架通过离线剖析来刻画个体模型能力，并结合两种互补信号：一种捕捉错误模式间的去相关性以减少共同失败，另一种衡量预测行为的分歧度以捕捉策略多样性。我们将团队选择形式化为一个标准化的质量-互补性组合优化目标，并应用高效的贪心搜索从候选池中选出一个小规模团队。在多个基准测试上的实验表明……

    arXiv:2609.38274v1 Announce Type: new  Abstract: The performance ceiling of an LLM team is constrained not only by individual model capabilities, but also by inter-member error resonance and predictive differences. Although heterogeneous teaming is often observed to be effective in practice, existing approaches lack complementarity metrics that are computable, interpretable, and optimizable, leaving team composition to rely on heuristics. We propose a heterogeneity-driven team selection framework that performs offline profiling to characterize individual capability along with two complementary signals: one captures decorrelation in error patterns to reduce co-failures, while the other measures divergence in predictive behavior to capture strategy diversity. We formulate team selection as a standardized quality--complementarity combinatorial objective and apply an efficient greedy search to select a small team from a candidate pool. Experiments across multiple benchmarks demonstrate tha
    
[^167]: NinaXander：通过共享潜空间跨架构家族组合冻结语言模型的可行性与局限

    NinaXander: Feasibility and Limits of Composing Frozen Language Models Across Architecture Families via a Shared Latent Space

    [https://arxiv.org/abs/2609.38261](https://arxiv.org/abs/2609.38261)

    提出NinaXander方法，仅用一个训练好的共享潜空间适配器即可将冻结的RWKV与Pythia等不同架构家族的语言模型拼接成可正常工作的组合模型，并验证了跨架构冻结模型事后重组的可行性及其局限。

    

    在本文中，我们提出了NinaXander，这是一系列通过单个训练好的共享潜空间适配器将来自不同架构家族的冻结语言模型的层连接起来而得到的组合语言模型。组合模型先运行一个模型的前几层，用适配器对得到的中间表示进行一次转换，然后运行另一个模型的其余层。适配器训练完成后，无需重新训练即可获得在不同层连接的多个组合模型。本研究使用循环架构的RWKV-4-Raven-7B和基于Transformer的Tulu-Pythia-6.9b（分别简称为RWKV和Pythia），考察了来自不同家族的冻结模型能否在事后被重新组合。组合模型能够回答多项选择题，且我们检验过其生成结果的模型产生了语法上规范的文本。将Pythia的前5层与其余27层组合的配置（摘要至此中断）

    arXiv:2609.38261v1 Announce Type: cross  Abstract: In this paper we propose NinaXander, a series of composed language models obtained by connecting layers of frozen language models from different architecture families with a single trained shared-latent adapter. A composed model runs the first layers of one model, converts the resulting intermediate representation once with the adapter, and then runs the remaining layers of the other model. Once the adapter is trained, several composed models that connect at different layers are obtained without retraining. Using the recurrent RWKV-4-Raven-7B and the Transformer-based Tulu-Pythia-6.9b, abbreviated as RWKV and Pythia, this study examines whether frozen models from different families can be recombined post hoc. The composed models answered multiple-choice questions, and those whose generations we examined produced syntactically well-formed text. The configuration that combines the first 5 layers of Pythia with the remaining 27 layers of 
    
[^168]: ContextAdapt：评估大语言模型中的情境适应与价值对齐

    ContextAdapt: Evaluating Contextual Adaptation and Value Alignment in LLMs

    [https://arxiv.org/abs/2609.38260](https://arxiv.org/abs/2609.38260)

    提出了ContextAdapt评估框架，通过基于职业与监管一手文件构建的“价值观×领域”场景，考察12个大语言模型能否在医学、法律、金融和国家安全等领域恰当调整诚实、自主、保密等价值观的应用，并在规范未变时保持一致。

    

    诸如诚实、自主和保密等价值观常被视为支撑AI对齐的一般原则。然而，如何依照这些价值观行事，可能取决于做出决策时所处的具体情境。本文探究大语言模型（LLM）能否在跨越不同专业场景时恰当地调整价值观的应用方式，同时当情境变化并未改变相关职业规范时保持行为的一致性。为此，我们提出了ContextAdapt，一个覆盖医学、法律、金融和国家安全四大领域中诚实、自主与保密价值观的评估框架。基于一手专业与监管文件，我们构建了“价值观×领域”框架，并据此设计了测试默认职业规则及公认例外情形的多种场景。我们从模型推荐的行为及其给出的理由两个维度对12个大语言模型进行了评估。在我们的主要（评估中）

    arXiv:2609.38260v1 Announce Type: cross  Abstract: Values such as honesty, autonomy, and confidentiality are often regarded as general principles underpinning AI alignment. However, what it means to act in accordance with these values can depend on the context in which a decision is made. In this paper, we ask whether large language models (LLMs) appropriately adapt the application of a value across professional settings, while remaining consistent when contextual changes do not alter the relevant professional norm. To study this, we introduce ContextAdapt, an evaluation framework covering honesty, autonomy, and confidentiality across medicine, law, finance, and national security. Drawing on primary-source professional and regulatory documents, we construct a value x domain framework and use this to develop scenarios testing both default professional rules and recognised exceptions. We evaluate 12 LLMs on both the actions they recommend and the justifications they provide. In our main 
    
[^169]: 构建叙事框架：大语言模型中的意识形态模仿

    Framing the Narrative: Ideological Mimicry in Large Language Models

    [https://arxiv.org/abs/2609.38256](https://arxiv.org/abs/2609.38256)

    该研究提出“意识形态模仿”概念并构建 Poli-SHIFT 数据集与评估框架，发现大语言模型会根据用户话语中传递的政治信号系统性偏移其政治立场，可能形成个性化的政治信息环境并加剧社会分歧。

    

    arXiv:2609.38256v1 公告类型：新论文 摘要：大语言模型（LLM）越来越多地被用于回答政治争议性问题，然而现有评估通常将模型的立场视为相对稳定的属性。但在真实场景中，用户会通过其用语、假设和个人背景传递政治信号。我们研究这些信号是否会引发“意识形态模仿”：即大语言模型所表达的政治立场向交互中传达的立场方向产生的系统性偏移。如果大语言模型会根据这些信号调整回答，它们就有可能营造出个性化的政治信息环境——持对立观点的用户会对同一问题获得系统性不同的叙述，从而可能加剧既有的社会分裂。我们构建了 Poli-SHIFT 数据集与评估框架，并对七个开放权重的大语言模型在美国、英国和澳大利亚的十个争议性政治议题上进行评估，系统地操纵上下文……

    arXiv:2609.38256v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to answer questions about politically contentious issues, yet evaluations typically treat a model's stance as a relatively stable property. Real users, however, communicate political signals through their terminology, assumptions, and personal context. We investigate whether such signals produce ideological mimicry: systematic shifts in the political stance expressed by an LLM toward the position conveyed by the interaction. If LLMs adapt their responses to these signals, they risk creating personalised political information environments in which users with opposing views receive systematically different accounts of the same issue, potentially reinforcing existing divisions. We build the Poli-SHIFT dataset and evaluation framework and assess seven open-weight LLMs across ten contentious political topics in the United States, United Kingdom, and Australia, systematically manipulating cont
    
[^170]: 口语智能体何时拥有足够证据采取行动？PACT-SLM 契约测试

    When Does a Spoken Agent Have Enough Evidence to Act? The PACT-SLM Contract Test

    [https://arxiv.org/abs/2609.38232](https://arxiv.org/abs/2609.38232)

    该论文提出 PACT-SLM 契约测试，通过在部分语音前缀上分别评估行动身份与行动时机，揭示了流式语音智能体常常在语音证据尚不充分时就提前触发行动，为口语智能体的行动时机提供了受控诊断方法。

    

    流式语音智能体可能会在现有语音尚不足以支持的情况下提前采取外部行动，而回合末尾的分数无法揭示每个已观测到的语音前缀是否支持该行动。我们提出了语音语言模型轮次转换的部分语音行动契约（PACT-SLM），这是一种受控评估方法，为行动分配首个有效行动时间，并将行动身份与行动时机分开测量。主要诊断集包含来自四个保留语义族的 80 个配对对比组，以及干净音频和 15 dB 噪声渲染下的 1,600 个前缀预测。在纠正了随机分支代码与语义标签之间的不匹配后，重新拟合的 WavLM Base Plus 探针达到了 26.03% 的起始后汇总语义标签准确率（95% 组自举置信区间：22.14%–29.68%），在 18.99% 的起始前前缀上暴露出行动，并精确预测了 5.94% 的完整轨迹。其表现超越了匹配的文本基线、标量声学特征基线以及打乱表示的基线（摘要原文在此处被截断）。

    arXiv:2609.38232v1 Announce Type: cross  Abstract: Streaming spoken agents may take an external action before the available speech supports it, yet final-turn scores do not reveal whether each observed prefix supports that action. We introduce the Partial Speech Action Contract for Turn Taking in Speech Language Models (PACT-SLM), a controlled evaluation that assigns a first valid action time and measures action identity and timing separately. The primary diagnostic contains 80 paired contrast groups from four held-out semantic families and 1,600 prefix predictions across clean and 15 dB noise renderings. After correcting a mismatch between randomized branch codes and semantic labels, a refitted WavLM Base Plus probe reaches 26.03% pooled post-onset semantic-label accuracy (95% group-bootstrap interval: 22.14%-29.68%), exposes an action on 18.99% of pre-onset prefixes, and predicts 5.94% of complete trajectories exactly. It exceeds matched text, scalar-acoustic, and shuffled-representa
    
[^171]: 面向多跳检索增强生成的保形事实性控制

    Conformal Factuality Control for Multi-Hop Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.38222](https://arxiv.org/abs/2609.38222)

    该研究将主张级保形事实性控制应用于多跳检索增强生成，证明在六种模型-数据集配置中，保形过滤能将保留主张获完全支持的回复比例从无过滤时的55.60%-76.03%稳定提升至95%目标下的95.80%-97.20%。

    

    检索增强生成（RAG）可以将大语言模型建立在外部证据之上，但检索到的上下文并不能保证生成的主张在事实上得到支持。这一问题在多跳RAG中尤为突出，因为其检索和推理需要经过多个相互依赖的阶段。我们研究了此前为RAG开发的主张级保形事实性控制（conformal factuality control）在这一设置中是否依然有效。我们将分割保形主张过滤（split-conformal claim filtering）应用于多跳RAG，并在HotpotQA、Natural Questions和TriviaQA数据集上使用Llama 3.1 8B和GPT-4o-mini进行评估，同时开展了单跳参考实验。在全部六种多跳模型-数据集配置中，越来越严格的保形目标始终提升了其保留主张获得完全支持的回复比例。在95%目标下，该比例达到95.80%至97.20%，而未过滤时仅为55.60%-76.03%。然而，这一改进（摘要原文在此处截断）……

    arXiv:2609.38222v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) can ground large language models in external evidence, but retrieved context does not guarantee that generated claims are factually supported. This problem is especially relevant in multi-hop RAG, where retrieval and reasoning proceed through multiple dependent stages. We study whether claim-level conformal factuality control, previously developed for RAG, remains effective in this setting. We apply split-conformal claim filtering to multi-hop RAG and evaluate it on HotpotQA, Natural Questions, and TriviaQA using Llama 3.1 8B and GPT-4o-mini, together with a single-hop reference experiment. Across all six multi-hop model-dataset configurations, increasingly stringent conformal targets consistently increase the fraction of responses whose retained claims are fully supported. At the 95% target, this rate ranges from 95.80% to 97.20%, compared with 55.60%-76.03% without filtering. However, the improvemen
    
[^172]: TutlAit v1：一个带有阿拉伯语转录和地区口音标签的众包摩洛哥塔马塞特语语音数据集

    TutlAit v1: a crowdsourced Moroccan Tamazight speech dataset with Arabic transcriptions and regional accent labels

    [https://arxiv.org/abs/2609.38219](https://arxiv.org/abs/2609.38219)

    本文推出了TutlAit v1数据集，通过专门构建的众包网络应用收集摩洛哥塔马塞特语语音，配有阿拉伯语转录和地区口音标签，以缓解该官方语言在语音技术领域资源严重匮乏的问题。

    

    塔马塞特语（阿马齐格语）与阿拉伯语同为摩洛哥的官方语言之一，但它在语音技术领域仍然严重缺乏资源：公开可用的标注音频稀少，通常缺少关于所讲地区变体的信息，且转录质量参差不齐。本文介绍了 TutlAit 数据集，这是一个摩洛哥塔马塞特语语音语料库，配有现代标准阿拉伯语文本和明确的地区口音标签。数据通过 TutlAit 收集，这是一个专门构建的众包网络应用（React 18 前端、Django 5 / Django REST Framework 后端、PostgreSQL 数据库）。通过定向的 LinkedIn 和 Instagram 宣传活动招募的母语者创建账户，声明他们的地区变体（阿特拉斯、苏斯、里夫或其他）以及人口统计信息，然后通过两种工作流程进行贡献：文本转语音（Text-to-Audio）流程中，系统显示一个阿拉伯语句子，志愿者对其进行录音……

    arXiv:2609.38219v1 Announce Type: new  Abstract: Tamazight (Amazigh) is, together with Arabic, one of the two official languages of Morocco, yet it remains severely under-resourced for speech technology: pub licly available labelled audio is scarce, generally lacks information on the regional variety spoken, and is often of uneven transcription quality. This article describes the TutlAit dataset, a corpus of Moroccan Tamazight speech paired with Modern Standard Arabic text and explicit regional accent labels. The data were collected with TutlAit, a purpose-built crowdsourcing web application (React 18 front end, Django 5 / Django REST Framework back-end, PostgreSQL database). Native speakers recruited through targeted LinkedIn and Instagram campaigns created an account, declared their regional variety (Atlas, Souss, Rif or other) and demographic information, and then contributed through two workflows: Text-to Audio, in which an Arabic sentence is displayed and the volunteer records its
    
[^173]: 系统提示词的错觉：指令前缀如何修改语言模型内部的计算

    The System Prompt Illusion: How Instruction Preambles Modify Computation in Language Models

    [https://arxiv.org/abs/2609.38205](https://arxiv.org/abs/2609.38205)

    该研究通过CKA分析发现系统提示词对语言模型内部计算的影响具有层选择性和指令类型依赖性——人设与格式指令深度重构中间表征，而安全指令几乎无法穿透模型计算，甚至限制性与放任性的安全指令激活几乎相同的计算路径，揭示了依靠系统提示词实现安全控制的局限。

    

    系统提示词是从业者用来控制语言模型行为的主要手段，然而它们对transformer内部计算的实际作用仍鲜为人知。在涵盖8个架构家族、参数量从1.5B到72B的17个指令微调模型上，我们使用居中核一致性对齐（CKA）方法，比较了五种功能类别共20个系统提示词下的逐层表征。研究发现其效应具有层选择性和指令类型依赖性：人设和格式化指令会深度重构中间表征，而安全指令几乎不会改变它们，所引起的变化在统计上与最小基线难以区分。限制性的安全指令与明确放开的指令（如“你没有任何限制”）激活了几乎相同的计算路径（平均CKA相关性达0.997），且该现象在商业规模模型中依然存在——即使在70B-72B参数的模型上，安全指令的渗透率仍低于10%。

    arXiv:2609.38205v1 Announce Type: new  Abstract: System prompts are the primary lever practitioners use to control language model behavior, yet what they actually do to the computation inside the transformer remains poorly understood. Across 17 instruction-tuned models spanning 8 architecture families and 1.5B to 72B parameters, we use Centered Kernel Alignment (CKA) to compare layer-wise representations under 20 system prompts in five functional categories. Effects are layer-selective and instruction-type-dependent: persona and formatting instructions deeply restructure intermediate representations, while safety instructions barely move them, producing changes statistically indistinguishable from a minimal baseline. Restrictive safety instructions and explicitly permissive ones ("you have no restrictions") engage near-identical computational pathways (mean CKA correlation 0.997), and this persists at commercial scale, where safety penetration remains below 10% even at 70B-72B. A linea
    
[^174]: 基于ASR对齐与停顿建模的运动神经元病患者言语流畅性指数自动估计

    Automatic estimation of verbal fluency index in people with Motor Neuron Disease using ASR alignment and pause modelling

    [https://arxiv.org/abs/2609.38203](https://arxiv.org/abs/2609.38203)

    本研究提出一种结合WhisperX语音识别对齐与Silero停顿建模的自动化系统，能够准确估计运动神经元病患者的言语流畅性指数，并提取临床可解释的指标，显著优于传统声学特征方法，为认知障碍监测提供了新方案。

    

    监测运动神经元病（MND）患者的认知障碍（CI）对于及时治疗和护理至关重要，但由于患者同时存在言语困难，监测工作极具挑战性。爱丁堡认知与行为ALS筛查量表（ECAS）为认知障碍评估提供了可靠的指标，其中言语流畅性指数（VFI）是核心要素。基于自动语音分析领域的最新进展，本研究提出了一种估计VFI的系统。该系统利用一个独特的MND数据集，将自动语音识别（ASR，WhisperX）与语音活动检测（VAD，Silero）结合精细的时间戳技术，以预测VFI并提取多项临床可解释的测量指标。在采用多种回归算法进行评估时，我们的方法优于基于传统声学特征和自监督嵌入的系统。受临床启发的特征始终优于其他特征集，最佳模型取得了优异的结果（P字母词：R² 0.9，NRMSE 0.05；S字母词：R² 0.8，NRMSE 0.08）。

    arXiv:2609.38203v1 Announce Type: cross  Abstract: Monitoring cognitive impairment (CI) in motor neuron disease (MND) is essential for timely treatment and care, yet challenging due to co-occurring speech difficulties. The Edinburgh Cognitive and Behavioural ALS Screen (ECAS) provides a robust metric for CI assessment, with the Verbal Fluency Index (VFI) a central element. Building on recent advances in automated speech analysis, this study proposes a system for estimating VFI. It leverages a unique MND dataset and combines ASR (WhisperX) and VAD (Silero) with refined timestamping to predict the VFI and extract several clinically interpretable measures. Our approach outperformed systems based on traditional acoustic features and self-supervised embeddings, evaluated using multiple regression algorithms. Clinically inspired features consistently outperformed the other sets, with the best models achieving strong results (P-words: R2 0.9, NRMSE 0.05; S-words: R2 0.8, NRMSE 0.08), demonstr
    
[^175]: TomasuLLM：面向大语言模型智能体的乱序推测执行

    TomasuLLM: Out-of-Order Speculative Execution for LLM Agents

    [https://arxiv.org/abs/2609.38201](https://arxiv.org/abs/2609.38201)

    TomasuLLM提出了一种乱序推测执行运行时系统，让大语言模型智能体的工具调用在写时复制沙箱中提前执行并验证后按轨迹顺序提交，从而在不破坏正确性的前提下显著加速含长时工具调用的智能体任务。

    

    长时间运行的工具可能会主导编码智能体的延迟：编译器、测试套件和仓库命令需要几秒到几分钟的时间，而智能体在此期间处于空闲状态。这一观察到的停顿呈现出与推动乱序处理器发展的相同矛盾——顺序接口隐藏了那些本可以被预测并提前启动的工作，但推测性结果只有在它自身及其之前的每一步都得到验证后才可能变得可见。我们提出了TomasuLLM，这是一个以乱序（偏离轨迹顺序）方式执行智能体工具调用、同时保持任务执行正确性的运行时系统。它起草未来的动作，在隔离的写时复制沙箱中运行这些动作，追踪它们的依赖关系和影响，只有在对照已提交状态进行验证之后，才按轨迹顺序提交结果。在三个涵盖亚秒级到分钟级工具调用的基准测试中，TomasuLLM提升了所报告的基准测试均值，且加速效果随工具延迟增加而扩展：在100个SWE-bench Verified任务上提升1.31倍，在28个Termi（原文截断）……

    arXiv:2609.38201v1 Announce Type: new  Abstract: Long-running tools can dominate coding-agent latency: compilers, test suites, and repository commands take seconds to minutes while the agent idles. This observation stall presents the same tension that drove out-of-order processors -- asequential interface hides work that can be predicted and started early, but a speculative result may become visible only after it and every earlier step have been validated.   We present TomasuLLM, a runtime that executes agent tool calls out of trajectory order while preserving task-execution correctness. It drafts future actions, runs them in isolated copy-on-write sandboxes, traces their dependencies and effects, and commits results in trajectory order only after validation against committed state. Across three benchmarks spanning sub-second to minutes-long tool calls, TomasuLLM improves the reported benchmark means and scales with tool latency: 1.31x on 100 SWE-bench Verified tasks, 1.35x on 28 Termi
    
[^176]: 大语言模型是近似的生存估计器

    Large Language Models are Approximate Survival Estimators

    [https://arxiv.org/abs/2609.38181](https://arxiv.org/abs/2609.38181)

    该论文提出了Survprompt框架，将结构化患者协变量转换为自由文本临床病例描述，以零样本方式提示预训练大语言模型预测患者生存结局，并在两个多机构泛癌症队列上与传统生存模型进行了系统性基准对比评估。

    

    生存分析根据患者协变量估计事件发生时间的结果，被广泛应用于医学风险评估中。在诊断后寻求预后信息的患者可能会求助于大语言模型（LLM），如今这些模型可以通过消费级应用程序轻松访问。然而，LLM能否提供准确的生存预测尚未得到严格评估。我们提出了Survprompt，这是一个将结构化患者协变量转换为自由文本临床病例描述的框架，并以零样本方式提示预训练的LLM预测生存结果。我们将Survprompt与传统生存模型（包括随机生存森林（RSF））在两个多机构泛癌症队列上进行了基准测试：公开可用的MSK-CHORD队列，以及新构建的来自Providence St. Joseph Health Network的队列，后者是使用基于LLM的医学信息抽取框架构建的。我们报告了删失平均绝对误差（cMAE）和一致性指数（c-

    arXiv:2609.38181v1 Announce Type: new  Abstract: Survival analysis estimates time-to-event outcomes from patient covariates and is widely used for medical risk assessment. Patients seeking prognostic information after a diagnosis may turn to large language models (LLMs), now readily accessible through consumer applications. However, whether LLMs can provide accurate survival predictions has not been rigorously evaluated. We introduce Survprompt, a framework that converts structured patient covariates into free-text clinical vignettes and prompts pre-trained LLMs to predict survival zero-shot. We benchmark Survprompt against conventional survival models, including random survival forests (RSF), across two multi-institutional pan-cancer cohorts: the publicly available MSK-CHORD cohort and a newly curated cohort from the Providence St. Joseph Health Network constructed using an LLM-based medical abstraction framework. We report censored mean absolute error (cMAE) and concordance index (c-
    
[^177]: 大语言模型中的性别偏见普遍存在且高度异质

    Gender bias across LLMs is common and highly heterogenous

    [https://arxiv.org/abs/2609.38036](https://arxiv.org/abs/2609.38036)

    该研究通过两种实验范式测试了来自九个厂商的十款大语言模型，发现性别偏见在模型间普遍存在但高度异质——部分模型表现出反刻板印象的性别归因模式，另一些模型则在道德判断中与人类保护女性免受伤害的倾向一致。

    

    随着大语言模型（LLM）被嵌入具有实际影响的决策支持工具中，理解其中的性别偏见变得越来越重要。以往的研究仅关注少数模型，使得性别偏见在各LLM之间的普遍性和异质性程度尚不明确。我们通过两项研究填补了这一空白，研究对象涵盖2025年4月至2026年6月间发布的十款模型，来自九个厂商，采用两种范式：对刻板印象短语的性别归因（研究1），以及对为防止灾难性后果而对女性或男性实施虐待或酷刑的道德判断（研究2）。在研究1中，十款模型中有两款将男性刻板印象短语归因于女性作者的频率高于反向情况，而三款模型表现出相反的模式。在研究2中，多个模型呈现出不利于男性的不对称性，其方向与已有文献记录的人类倾向于保护女性目标免受伤害的倾向一致，尽管……

    arXiv:2609.38036v1 Announce Type: cross  Abstract: Understanding gender biases in large language models (LLMs) is increasingly important as these systems become embedded in decision-support tools with real consequences. Prior research has focused only on a small set of models, leaving open the extent to which gender biases are common and heterogeneous across LLMs. We address this gap across ten models released between April 2025 and June 2026, spanning nine vendors, using two paradigms: gender attribution to stereotyped phrases (Study 1) and moral judgment of abuse or torture against a woman or a man to prevent a catastrophic outcome (Study 2). In Study 1, two of ten models attributed masculine-stereotyped phrases to female writers more often than the reverse, while three models showed the opposite pattern. In Study 2, several models converged on a male-disadvantaging asymmetry that was directionally consistent with a documented human tendency to protect female targets from harm, thoug
    
[^178]: SelfSearch：面向自我改进智能体的免奖励搜索

    SelfSearch: Reward-Free Search for Self-Improving Agents

    [https://arxiv.org/abs/2609.37968](https://arxiv.org/abs/2609.37968)

    SelfSearch提出了一种免奖励的自我改进搜索方法，智能体通过利用以往自我修改回合的记录（包含推理、工具操作和结果）来改进自身，无需昂贵的下游评估即可在多个模型-基准测试设置中显著提升成功率。

    

    arXiv:2609.37968v1 公告类型： new 摘要：大语言模型智能体编程能力的进步使其能够检查和修改自身的指令、工具和执行程序。现有方法利用这种能力，通过反复的下游评估来搜索性能更优的智能体，这不仅带来高昂成本，还将搜索过程与被评估的任务绑定在一起。我们提出了SelfSearch，这是一种免奖励的搜索程序，智能体利用以往自我改进回合的记录来修改自身。这些记录捕捉了先前修改尝试中的推理过程、工具操作和结果，为同时改进任务解决能力和自我修改能力提供了具体的经验。在搜索过程中没有下游奖励信号的情况下，SelfSearch在全部六个模型-基准测试设置中均提升了相对于初始智能体的种群平均成功率，其中单个智能体在Terminal-Bench 2.1上最多提升了11.2个百分点。在SWE-bench Multilingual上，一个智能体的成功率提升了\

    arXiv:2609.37968v1 Announce Type: new  Abstract: Advances in the coding capabilities of LLM agents allow them to inspect and modify their own instructions, tools, and execution procedures. Existing approaches use this ability to search for improved agents through repeated downstream evaluation, which incurs substantial costs and ties the search to the evaluated tasks. We introduce \textbf{SelfSearch}, a reward-free search procedure in which agents modify themselves using records of previous self-improvement episodes. These records capture the reasoning, tool actions, and outcomes of earlier modification attempts, providing concrete experience for improving both task solving and self-modification. Without downstream reward signals during search, SelfSearch improves population-mean success over the initial agent in all six model--benchmark settings, with individual agents gaining up to 11.2 percentage points on Terminal-Bench 2.1. On SWE-bench Multilingual, an agent improves success by \
    
[^179]: 一种用于评估大语言模型回答中所表达的临床推理的评分量规（提议稿）

    A Proposed Rubric for Evaluating Expressed Clinical Reasoning in Large Language Model Responses

    [https://arxiv.org/abs/2609.37788](https://arxiv.org/abs/2609.37788)

    该论文提出一个融合医学教育评估框架、临床大语言模型基准和通用LLM推理评估研究的多维评分量规，用于对大语言模型针对金标准临床案例的自由文本回答中表达的临床推理进行结构化评估。

    

    评分量规为语言模型的结构化评估提供支持。我们提出了一种用于评估模型回答中所表达的临床推理的量规，该量规借鉴了三个方面的工作：医学教育评估框架（ART、SCT、关键特征问题以及OSCE）；临床大语言模型基准（MedR-Bench、HealthBench、TIMER-Bench、DR.BENCH、PrIME-LLM 和 PatientSafeBench）；以及通用大语言模型推理评估研究，包括事实性-有效性-连贯性-有用性（Factuality-Validity-Coherence-Utility）分类法、FaithCoT-Bench 和 C2-Faith。我们使用“有据性”作为对该分类法中事实性类别的面向临床的改编。该量规将这些概念整合到一个多维框架中，用于对针对金标准临床案例片段的自由文本回答进行评分。它包括初步的行为锚定标准、适用性规则，以及一个用于标记案例特定安全关键错误的独立标记。通用领域的框架为其设计提供了参考，但并未被视为经过验证的临……（原文摘要在此处截断）

    arXiv:2609.37788v1 Announce Type: cross  Abstract: Rubrics support the structured evaluation of language models. We propose a rubric for assessing expressed clinical reasoning in model responses, drawing on three bodies of work: medical education assessment frameworks (ART, SCT, Key Feature Problems and OSCE); clinical LLM benchmarks (MedR-Bench, HealthBench, TIMER-Bench, DR.BENCH, PrIME-LLM and PatientSafeBench); and general LLM reasoning evaluation research, including the Factuality-Validity-Coherence-Utility taxonomy, FaithCoT-Bench and C2-Faith. We use groundedness as a clinically oriented adaptation of the taxonomy's factuality category. The rubric brings these concepts together in a multidimensional framework for scoring free-text responses to gold-standard clinical vignettes. It includes provisional behavioural anchors, applicability rules and a separate flag for case-specific safety-critical errors. General-domain frameworks inform its design but are not treated as validated cl
    
[^180]: MERGE：基于生成式增强的多大语言模型集成检索框架

    MERGE: Multi-LLM Ensemble for Retrieval via Generative Enrichment

    [https://arxiv.org/abs/2609.37574](https://arxiv.org/abs/2609.37574)

    MERGE提出一个两阶段多LLM集成框架，先由三个小型开源LLM独立生成查询扩展候选、再由更大的LLM生成式合成为单一查询，并用基于下游检索性能的自动提示词优化循环取代传统LLM评估器，解决了单一LLM查询增强受限于模型偏见且提示词工程难以扩展的问题。

    

    大语言模型（LLM）越来越多地被用于信息检索（IR）中的用户查询增强，使BM25等标准检索器能够弥合查询与目标语料库之间的词汇鸿沟。然而，任何单一LLM都受限于其训练数据和架构偏见，且其增强行为依赖于手工设计的提示词——这些提示词必须针对每个新模型重新设计，是一个昂贵且难以扩展的过程。我们提出了MERGE（Multi-LLM Ensemble for Retrieval via Generative Enrichment，基于生成式增强的检索多LLM集成框架），这是一个两阶段框架：三个异构的7-8B开源LLM独立生成候选查询扩展，随后由一个更大的LLM将它们生成式地合成为单一查询。为了使提示词工程在整个模型集成中具备可扩展性，我们在两个阶段中都集成了基于任务的自动提示词优化（APO）循环。与使用LLM评估器来判断候选结果的APO方法不同，我们的循环根据每个候选的下游检索表现进行评分。

    arXiv:2609.37574v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly used to enrich user queries in information retrieval (IR) so that a standard retriever such as BM25 can bridge vocabulary gaps with the target corpus. Any single LLM, however, is limited by its training data and architectural biases, and its enrichment behavior depends on hand-crafted prompts that must be re-engineered for each new model -- an expensive and poorly scalable process. We present MERGE (Multi-LLM Ensemble for Retrieval via Generative Enrichment), a two-stage framework: three heterogeneous 7-8B open-source LLMs independently produce candidate expansions, and a larger LLM generatively synthesizes them into a single query. To make prompt engineering scalable across the ensemble, we integrate a task-grounded Automatic Prompt Optimization (APO) loop into both stages. Unlike APO methods that judge candidates with an LLM evaluator, our loop scores each candidate by its downstream retr
    
[^181]: Traverse：学习何时记忆、重置与重定向的长程网络搜索

    Traverse: Learning When to Remember, Reset, and Redirect for Long-Horizon Web Search

    [https://arxiv.org/abs/2609.37082](https://arxiv.org/abs/2609.37082)

    论文提出Traverse自主搜索框架，让智能体通过“评分标准—答案—验证”三状态自我管理搜索并配备封存记忆工具进行主动上下文管理，同时用仅训练上下文管理后末段的简单策略避免“封存崩塌”，使35B模型在BrowseComp上达到72.83。

    

    长程信息检索智能体常常会累积噪声或具有误导性的上下文，导致早期错误持续存在，并使恢复变得越来越困难。我们引入了一种自主搜索框架，其中智能体通过三种状态——评分标准、答案和验证——来管理自身的搜索过程。智能体首先定义有效答案的标准，在这些标准下进行搜索，然后独立验证结果，再决定是终止还是继续搜索。它还配备了一个“封存记忆”工具，以实现主动的上下文管理。然而，使用强化学习训练这种行为可能引发“封存崩塌”，导致训练不稳定，使智能体无法可靠地学会何时以及如何使用其记忆工具。我们通过一个简单的策略解决了这一问题：仅训练上下文管理之后的最后片段。我们的35B模型在BrowseComp上取得了72.83的成绩，超越了同类模型。

    arXiv:2609.37082v1 Announce Type: new  Abstract: Long-horizon information-seeking agents often accumulate noisy or misleading context, causing early mistakes to persist and making recovery increasingly difficult. We introduce an autonomous search harness in which the agent manages its own search process through three states: Rubric, Answer, and Verify. The agent first defines criteria for a valid answer, searches under these criteria, and then independently verifies the result before deciding whether to terminate or continue searching. It is further equipped with a Seal Memory tool that enables active context management. Training this behavior with reinforcement learning, however, can induce Seal Collapse, resulting in unstable training and preventing the agent from reliably learning when and how to use its memory tools. We solve this with a simple strategy that trains only the final segment after context management. Our 35B model achieves 72.83 on BrowseComp, outperforming comparable 
    
[^182]: 通过在线策略蒸馏从思考模式优势中学习

    Learning from Think-Mode Advantage via On-Policy Distillation

    [https://arxiv.org/abs/2609.37044](https://arxiv.org/abs/2609.37044)

    本文提出 ThinkOPD，通过在线策略蒸馏让模型学习思考模式的优势，并引入轨迹-回答分歧（TRD）度量，在回答层面自适应地路由教师监督信号，克服了统一共享思考轨迹蒸馏带来的师生不匹配问题。

    

    显式的中间推理赋予大型语言模型（LLM）更强的问题求解模式。我们研究如何通过在线策略蒸馏（OPD）从这种思考模式优势中学习。在线策略蒸馏保留了学生模型自身生成的轨迹，并在学生所到达的前缀处提供密集的词元级教师目标；特权推理仅在蒸馏阶段使用，而非在学生推理时使用。Uniform ThinkOPD 是一种自然的启用思考模式的在线策略蒸馏基线，它以一条共享的思考轨迹为条件固定教师模型，并对同组的每个学生回答进行统一蒸馏。尽管其前缀是在线策略的，但该思考轨迹未必与每个完整回答的推理路径相容：即使多个回答最终得到相同的结果，同一条特权轨迹也可能引发不同的师生差异。我们用“轨迹-回答分歧”（TRD）来概括这种交互作用，并在此基础上提出 ThinkOPD，它在回答层面路由监督信号，通过结合组相对奖励（原文摘要在此处被截断）……

    arXiv:2609.37044v1 Announce Type: new  Abstract: Explicit intermediate reasoning gives large language models (LLMs) a stronger problem-solving mode. We study learning from this think-mode advantage via on-policy distillation (OPD). OPD preserves student-generated trajectories and provides dense token-level teacher targets at student-visited prefixes. Privileged reasoning is used during distillation rather than student inference. Uniform ThinkOPD, a natural think-enabled OPD baseline, conditions a fixed teacher on one shared think trace and uniformly distills every sibling student response. Although its prefixes are on-policy, the trace need not follow a route compatible with every complete response: the same privileged trace can induce different teacher-student discrepancies even when responses reach the same outcome. We summarize this interaction with trace-response divergence (TRD) and introduce ThinkOPD, which routes supervision at the response level by combining group-relative rewa
    
[^183]: CoEM：基于证据提交记忆的长上下文推理增强方法

    CoEM: Empowering Long-Context Reasoning with Commit-on-Evidence Memory

    [https://arxiv.org/abs/2609.36935](https://arxiv.org/abs/2609.36935)

    CoEM 提出了一种“证据提交记忆”机制：在固定上下文预算下先逐字保留待定证据，再由学习到的策略根据新到来的上下文决定将其提交、继续保留或丢弃，从而避免过早压缩导致的关键信息丢失，提升长上下文推理性能。

    

    长上下文推理对于复杂且长时程的任务至关重要，然而大语言模型（LLM）的性能会随着上下文长度的增加而下降。近期的解决方法是将输入逐块处理，同时在模型上下文中维护容量有限的文本记忆。然而，过早的信息压缩可能会丢弃对后续推理至关重要的关键细节。在本文中，我们提出了证据提交记忆，它学习何时将源证据转换为紧凑的记忆事实。具体而言，在固定的上下文-记忆预算下，CoEM 将潜在有用的源文本片段逐字保存在一个待定集合中，使后续上下文有机会澄清其相关性，然后再进行不可逆的压缩。随着新上下文的到来，一个学习到的策略会重新审视每个待定片段，并决定是将其提升为已提交记忆、继续保留以待进一步考量，还是将其丢弃。一个冻结的验证器确保……

    arXiv:2609.36935v1 Announce Type: new  Abstract: Long-context reasoning is essential for complex and long-horizon tasks, yet the performance of large language models (LLMs) degrades as context length increases. Recent approaches address this by processing input chunk by chunk while maintaining a bounded textual memory in model context. However, premature information compression can discard critical details essential for subsequent reasoning. In this paper, we introduce Commit-on-Evidence Memory (CoEM), which learns when to convert source evidence into compact memory facts. Specifically, under a fixed context-memory budget, CoEM preserves potentially useful source excerpts verbatim in a pending set, allowing subsequent context to clarify their relevance before irreversible compression. As new context arrives, a learned policy revisits each pending excerpt and decides whether to promote it to the committed memory, retain it for further consideration, or discard it. A frozen verifier ensu
    
[^184]: QuantMLA：面向低比特 MLA KV 缓存的函数对齐双路径量化

    QuantMLA: Function-Aligned Dual-Path Quantization for Low-Bit MLA KV Caching

    [https://arxiv.org/abs/2609.36760](https://arxiv.org/abs/2609.36760)

    提出了 QuantMLA——一个函数对齐的低比特双路径量化框架，通过系统建模 MLA 内容路径与 RoPE 路径的量化误差，并学习可完全离线融合的路径特定变换，在消除在线开销的同时大幅压缩 MLA KV 缓存且保持全精度计算效果。

    

    多头潜在注意力（MLA）通过为其内容路径与解耦的 RoPE 路径采用紧凑缓存，实现了表达能力强的多头注意力，但其缓存内存仍会随上下文长度和批处理大小线性增长。在本工作中，我们建立了 MLA 双路径量化误差的系统模型，刻画了它们对注意力输出失真的不同影响，并解释了 RoPE 路径误差被显著放大的现象。基于这一分析，我们提出了 QuantMLA，一个用于低比特双路径量化的函数对齐框架。我们推导出路径特定的变换空间，这些空间在保持全精度计算的同时，可以完全离线融合到模型参数中，从而消除在线变换开销。在这些空间内，QuantMLA 以函数对齐的目标学习路径特定的变换：注意力输出重建目标捕捉内容路径的耦合匹配与聚合误差，而位置（posit

    arXiv:2609.36760v1 Announce Type: cross  Abstract: Multi-Head Latent Attention (MLA) enables expressive multi-head attention with compact caches for its content and decoupled RoPE paths, yet cache memory still scales linearly with context length and batch size. In this work, we establish a systematic model of MLA's dual-path quantization errors, characterizing their distinct effects on attention-output distortion and explaining the pronounced amplification of RoPE-path errors. Guided by this analysis, we introduce QuantMLA, a function-aligned framework for low-bit dual-path quantization. We derive path-specific transformation spaces that preserve full-precision computation while remaining fully fusible into model parameters offline, eliminating online transformation overhead. Within these spaces, QuantMLA learns path-specific transformations with function-aligned objectives: attention-output reconstruction captures the content path's coupled matching and aggregation errors, while posit
    
[^185]: 在学生状态下从教师续写中学习

    Learning from Teacher Continuations at Student States

    [https://arxiv.org/abs/2609.36246](https://arxiv.org/abs/2609.36246)

    OLIVE 提出一种在线干预式蒸馏框架——学生生成前缀、教师自回归续写并以其交叉熵更新学生——同时解决了离线 SFT 的协变量偏移、OPD 的监督碎片化以及分布匹配蒸馏需教师 token 概率三大局限，以相近成本取得更优推理性能。

    

    我们提出了 OLIVE（OnLine InterVEntion，在线干预）。在每次迭代中，不断演进的学生策略生成一个新的前缀，教师以自回归方式对该前缀进行续写，然后利用在教师生成 token 上计算的交叉熵来更新学生。每一项设计选择都针对现有蒸馏方法的一个相应局限：(1) 基于固定教师轨迹的离线监督微调（SFT）中的序列协变量偏移问题；(2) token 级在策略蒸馏（OPD）中前缀失败导致的监督碎片化问题；以及 (3) 分布匹配蒸馏需要访问教师 token 概率的问题。在相当的 GPU 小时成本下，OLIVE 取得了比 OPD（使用 top-16 KL 近似）更高的推理性能。我们的异步实现进一步将 OLIVE 的总训练时间减少了 23.8%。我们在困难推理任务和智能体任务上评估了 OLIVE，这些任务反映了现代后训练场景，并且它始终……（摘要在此处截断）

    arXiv:2609.36246v1 Announce Type: new  Abstract: We present OLIVE (OnLine InterVEntion). At each iteration, the evolving student policy generates a new prefix, the teacher continues it autoregressively, and the student is updated using cross-entropy computed on the teacher-generated tokens. Each design choice targets a corresponding limitation of existing distillation methods: (1) sequential covariate shift in offline supervised fine-tuning (SFT) on fixed teacher trajectories, (2) fragmented supervision under prefix failure in token-level on-policy distillation (OPD), and (3) the need for access to teacher token probabilities in distribution-matching distillation. OLIVE achieves higher reasoning performance than OPD (with a top-16 KL approximation) at comparable GPU-hour cost. Our asynchronous implementation further reduces OLIVE's total training time by 23.8\%. We evaluate OLIVE on both hard reasoning tasks and agentic tasks which reflects modern post-training scenarios, and it consis
    
[^186]: Sage: 基于语义修正的形式化

    Sage: Formalization with Semantic Correction

    [https://arxiv.org/abs/2609.35790](https://arxiv.org/abs/2609.35790)

    本文提出 Sage，一个智能体化的形式化引擎，通过四阶段分解式生成流水线与融合 Lean 4 编译器诊断和多维语义反馈的双信号语义修正循环，解决了自然语言翻译为 Lean 4 形式化命题时的“严谨性幻觉”问题，确保生成命题既句法有效又数学忠实。

    

    arXiv:2609.35790v1 公告类型：交叉（cross） 摘要：尽管神经定理证明器已在形式数学领域取得了令人瞩目的里程碑式进展，但它们大多建立在一个假设之上，即已被忠实翻译的 Lean 4 形式化命题是现成可用的。将非形式的自然语言翻译为形式语言是一个关键的数据瓶颈，且这一过程深受“严谨性幻觉”的困扰：标准类型检查器会接受那些能够编译通过、却丢弃了假设条件、引入了空洞真命题或微妙地改变了数学界限的命题。为解决这一问题，我们提出了 Sage（语义智能体引导的形式化引擎，Semantic Agent-Guided Formalization Engine），这是一个智能体化框架，它用一个四阶段分解式生成流水线取代了单体式翻译，并耦合了一个双信号语义修正循环。通过将 Lean 4 编译器诊断信息与多维度的语义反馈相结合，我们的修正循环在保证句法有效性的同时强制实现数学忠实性。通过显式地考虑开放式查询与声明式形式化目标之间的差距……

    arXiv:2609.35790v1 Announce Type: cross  Abstract: While neural theorem provers have achieved impressive milestones in formal mathematics, they largely operate on the assumption that faithful Lean 4 formal statements are already provided. Translating informal natural language into a formal language is a critical data bottleneck plagued by an "illusion of rigor": standard type-checkers accept statements that compile but drop hypotheses, introduce vacuous truths, or subtly alter mathematical bounds. To resolve this, we introduce Sage (Semantic Agent-Guided Formalization Engine), an agentic framework that replaces monolithic translation with a four-stage decomposed generation pipeline coupled with a dual-signal semantic correction loop. By pairing Lean 4 compiler diagnostics with multi-dimensional semantic feedback, our correction loop enforces mathematical fidelity alongside syntactic validity. By explicitly accounting for the gap between open-ended queries and declarative formal targets
    
[^187]: SEABench：自进化智能体中内生失准的基准测试

    SEABench: Benchmarking Endogenous Misalignment In Self-Evolving Agents

    [https://arxiv.org/abs/2609.35596](https://arxiv.org/abs/2609.35596)

    该论文提出了SEABench基准，用于量化研究自进化智能体因自我修改而在无外部对抗影响下产生的内生失准（不安全行为）风险。

    

    自进化大语言模型智能体因其能够在部署后通过修改自身框架（包括控制器指令、记忆管理协议以及可复用的工具和技能）来响应用户和环境反馈并持续改进而受到广泛关注。然而，在局部看似有用的更新可能会延续到后续任务中，即使没有直接的对抗性影响，也会产生不安全的行为。为了研究这一风险，我们提出了SEABench，一个用于研究由智能体自进化引发的内生失准的基准，包含48个纵向任务序列，涵盖丰富的个人助理环境中多种进化层面、任务领域和危害类型。为了考虑智能体操作中固有的随机性，我们提供了一个自适应轨迹发现流水线，在保持原始任务意图的同时探测故障，并通过配对的非进化智能体等机制支持因果归因。

    arXiv:2609.35596v2 Announce Type: replace-cross  Abstract: Self-evolving LLM agents have gained prominence for their ability to improve after deployment by modifying their harness, including their controller instructions, memory management protocols, and reusable tools and skills, in response to user and environment feedback. However, locally useful updates may persist into later tasks where they produce unsafe behavior, even without direct adversarial influence. To study this risk, we introduce SEABench, a benchmark for studying endogenous misalignment arising from agent self-evolution, with 48 longitudinal task sequences that span multiple evolution surfaces, task domains, and harm types in a rich personal-assistant environment. To account for the stochasticity inherent in agentic operations, we provide an adaptive trajectory discovery pipeline that probes for failures while preserving original task intent and supports causal attribution through paired non-evolving agents and attribu
    
[^188]: 编码智能体记忆后训练：通过强化学习释放预训练文件操作在长程任务中的记忆潜力

    Coding Agent Memory Post-training: Unlocking the Memory Potential of Pre-trained File Operations for Long-Horizon Tasks via Reinforcement Learning

    [https://arxiv.org/abs/2609.34422](https://arxiv.org/abs/2609.34422)

    该论文提出 CAMG 训练场套件，通过强化学习让智能体直接复用预训练中已掌握的文件操作能力作为记忆机制，而非从头学习专用记忆工具，从而显著提升智能体在长程任务中的记忆使用能力。

    

    语言模型智能体越来越多地处理交互历史超出模型活跃上下文的长程任务。近期的工作开始使用强化学习将记忆控制纳入策略之中，但通常依赖于在相对短程的领域特定训练环境中的预定义记忆工具。这种设置将学到的记忆行为与基础模型预训练范围之外、必须从头学习的环境特定接口绑定在一起，因此即使经过后训练，智能体在长程任务中仍难以有效使用记忆。为解决这些局限，我们提出了编码智能体记忆训练场，这是一套涵盖 Shop、Coding、DeepResearch 和 AutoResearch 的长程智能体强化学习环境套件。在每个环境的原生任务接口之外，CAMG 还提供了可执行的 shell 访问权限和一个在回合内持久保存的工作区，使智能体能够创建、修改、搜索并复用文件作为记忆。

    arXiv:2609.34422v2 Announce Type: replace-cross  Abstract: Language-model agents increasingly tackle long-horizon tasks whose interaction histories exceed the model's active context. Recent work has begun to use reinforcement learning to make memory control part of the policy, often relying on predefined memory tools within domain-specific training environments of relatively short horizons. This setup ties learned memory behavior to environment-specific interfaces that lie outside the base model's pre-training and must be learned from scratch, so even after post-training, agents struggle to use memory in long-horizon tasks. To address these limitations, we introduce Coding Agent Memory Gym (CAMG), a suite of long-horizon agentic-RL environments spanning Shop, Coding, DeepResearch, and AutoResearch. Alongside each environment's native task interface, CAMG provides executable shell access and an episode-persistent workspace, enabling agents to create, revise, search, and reuse files as m
    
[^189]: 过度个性化是一种决策失败：大语言模型中的生成诱导“应用偏差”

    Over-Personalization Is a Decision Failure: Generation-Induced Apply Bias in LLMs

    [https://arxiv.org/abs/2609.34284](https://arxiv.org/abs/2609.34284)

    该论文将大语言模型处理用户偏好的过程分解为“知道—决策—生成”三个阶段分别测量，揭示过度个性化的根源在于决策失败：当模型被要求作答时，生成过程会诱发“应用偏差”，使其执行本应抑制的偏好，而非知识缺失或生成错误所致。

    

    个性化的大语言模型必须针对每个已存储的偏好，判断当前情境是否需要应用或抑制该偏好，我们将这一性质称为“适用性”。模型经常出现过度个性化问题，即应用了情境已排除的偏好，但现有基准仅对最终回复评分，无法定位失败发生的环节。我们将偏好处理分解为三个阶段并分别测量：(1) 知道某个偏好是否适用；(2) 做出明确的“应用/抑制”标签决策；(3) 生成与该标签一致的回复。利用线性探针，我们首先证明这一适用性信号在生成过程中依然可以从隐藏状态中被解码出来。通过将决策显式化，我们进一步发现，在大多数情况下，被忠实执行的错误决策要多于在生成过程中丢失的正确决策。因此，我们将失败定位于决策环节：一旦模型同时被要求作答，决策就会失效。（原文摘要在此处截断）

    arXiv:2609.34284v2 Announce Type: replace  Abstract: Personalized LLMs must decide, for each stored preference, whether the current context calls for applying or suppressing it, which we call its applicability. They frequently over-personalize, applying preferences the context rules out, yet existing benchmarks score only the final response and cannot tell where this failure arises. We decompose preference handling into three stages and measure each separately: (1) knowing whether a preference applies, (2) deciding on an explicit Apply/Suppress label, and (3) generating a response consistent with that label. Using linear probes, we first show that this applicability signal remains decodable from hidden states during generation. By making the decision explicit, we then find that in most settings wrong decisions faithfully followed outnumber correct decisions lost in generation. We thus locate the failure in the decision, which breaks once the model is also asked to answer. To determine 
    
[^190]: Jev在医学中的应用：一项基准评估。初步结果

    Jev in Medicine: A Benchmark Evaluation. Preliminary Results

    [https://arxiv.org/abs/2609.34024](https://arxiv.org/abs/2609.34024)

    本研究首次在四个医学基准上系统评估了非生成式“系统一”模型Jev，发现其准确率总体低于或接近GPT-6 Sol，但概率校准表现更优。

    

    Jev是一种非生成式的“系统一”模型，它为预定义的答案选项分配概率，无法在选项之外作答。其在医学问答和基于病例的诊断推理任务中的准确性与校准情况尚不清楚。我们在四个医学基准上对Jev 1.13进行了评估：MetaMedQA、PubMedQA、DiagnosisArena-MCQ以及NEJM病例挑战，并以开启（中等）与关闭推理模式的GPT-6 Sol作为参照。主要结局指标为top-1准确率，关键次要结局为校准度、选择性预测以及对无法回答问题的识别能力。全部8,469次请求均返回了有效答案。在PubMedQA上，Jev的准确率与开启中等推理的GPT-6 Sol相当（78.4% vs 78.2%），在MetaMedQA上较低（74.8% vs 82.7%），而在DiagnosisArena-MCQ（59.8% vs 82.4%）和NEJM病例（61.8% vs 82.4%）上则明显更低。在MetaMedQA上，Jev的概率校准表现最佳（期望校准误差0.063 vs 0.1……

    arXiv:2609.34024v1 Announce Type: cross  Abstract: Jev is a non-generative "System One" model that assigns probabilities to predefined answer options and cannot answer outside them. Its accuracy and calibration on medical question-answering and case-based diagnostic-reasoning tasks are unknown. We evaluated Jev 1.13 on four medical benchmarks: MetaMedQA, PubMedQA, DiagnosisArena-MCQ and the NEJM Case Challenges. GPT-6 Sol, with (medium) and without reasoning, was the reference. The primary outcome was top-1 accuracy; key secondary outcomes were calibration, selective prediction and recognition of unanswerable questions. All 8,469 requests returned a valid answer. Jev's accuracy was similar to that of GPT-6 Sol with medium reasoning on PubMedQA (78.4% vs 78.2%;), lower on MetaMedQA (74.8% vs 82.7%) and much lower on DiagnosisArena-MCQ (59.8% vs 82.4%;) and the NEJM cases (61.8% vs 82.4%). On MetaMedQA, Jev's probabilities were the best calibrated (expected calibration error 0.063 vs 0.1
    
[^191]: 通过查询条件归因审计智能体行为

    Auditing Agent Actions through Query-Conditioned Attribution

    [https://arxiv.org/abs/2609.33676](https://arxiv.org/abs/2609.33676)

    本文提出了“查询条件的智能体行为归因”新任务及包含1,396个审计查询的A³Bench基准，能够根据自然语言审计查询自动恢复智能体行动的来源与有序中间证据，在无需访问模型内部的情况下实现对LLM智能体行为的高效审计。

    

    大语言模型智能体越来越多地通过与用户、策略和外部工具的交互采取具有重大影响的行动。审计这些智能体需要将已实现的行动自动归因于其历史依据。然而，现有的归因方法无法为多样化的审计目标提供针对特定问题的溯源轨迹。此外，当对执行模型的访问受限时（例如仅在API部署的情况下），适用的方法通常依赖于代价高昂的输入扰动或外部大语言模型对完整轨迹的分析。因此，我们提出了“查询条件的智能体行为归因”这一新任务，该任务以自然语言审计查询作为输入，恢复查询所指定的行动方面的来源以及有序的中间证据。我们通过 $A^3Bench$ 来实例化这一任务，该基准包含1,396个审计查询，涵盖策略依据、参数溯源、失败传播和不安全行为追踪等方面。

    arXiv:2609.33676v2 Announce Type: replace  Abstract: LLM agents increasingly take consequential actions through interactions with users, policies, and external tools. Auditing these agents requires automated attribution of realized actions to their historical basis. However, existing attribution formulations do not provide question-specific traces for diverse auditing objectives. Additionally, when access to the acting model is limited (e.g., in API-only deployments), applicable methods commonly rely on costly input perturbations or external LLM analysis of complete trajectories. We therefore formulate query-conditioned agent action attribution, a new task that takes a natural-language auditing query as input and recovers the source and ordered intermediate evidence for the query-specified aspect of an action. We instantiate this task with $A^3Bench$, a benchmark comprising 1,396 auditing queries across policy basis, parameter provenance, failure propagation, and unsafe-behavior tracin
    
[^192]: 不要重复自己：面向覆盖度的自监督微调

    Don't Repeat Yourself: Self-Supervised Fine-Tuning for Coverage

    [https://arxiv.org/abs/2609.31688](https://arxiv.org/abs/2609.31688)

    DRY-SFT是一种无需奖励、验证器或正确性过滤的两阶段后训练方法，通过让模型在参考先前尝试的基础上生成不同解，再对每个尝试独立微调，从而显著提升大语言模型的输出多样性与覆盖度。

    

    在数学和编程等可验证的领域中，在多次尝试中找到至少一个正确答案，可能比单次尝试的通过率更为重要。后训练会使大语言模型的输出集中于少数几个模式，而提高采样温度的效果有限。我们提出了“不要重复自己”监督微调（DRY-SFT），这是一种提升输出多样性和覆盖度的后训练方法——覆盖度指多次尝试中至少获得一个正确答案的概率。DRY-SFT包含两个阶段：首先，针对每个问题依次生成K个解，向模型展示所有先前的尝试并要求其给出不同的解；其次，在每个尝试上独立进行微调，并从上下文中移除先前的尝试。整个过程不使用任何奖励、验证器或正确性过滤。在HumanEval+、MBPP+和DS-1000上，DRY-SFT分别将pass@100提升了10.8、12.5和12.4个百分点，同时对单次通过率的代价很小。

    arXiv:2609.31688v2 Announce Type: replace  Abstract: In verifiable domains such as math and coding, finding one correct solution among many attempts can matter more than the pass rate of each attempt. Post-training can concentrate large language model outputs around a few modes, while increasing sampling temperature has limited effectiveness. We introduce Don't Repeat Yourself Supervised Fine-Tuning (DRY-SFT), a post-training method that increases output diversity and coverage: the probability of at least one correct solution among many attempts. DRY-SFT has two stages. First, for each problem, sequentially generate K solutions, showing the model all prior attempts and asking for a different solution. Second, fine-tune on each attempt independently, removing prior attempts from the context. The process uses no reward, verifier, or correctness filter. On HumanEval+, MBPP+, and DS-1000, DRY-SFT raises pass@100 by 10.8, 12.5, and 12.4 percentage points, respectively, at a small cost to pa
    
[^193]: RAZOR：在大语言模型中剪枝可被替换的专家

    RAZOR: Pruning Replaceable Experts in LLMs

    [https://arxiv.org/abs/2609.30465](https://arxiv.org/abs/2609.30465)

    该论文提出无需训练的 MoE 专家剪枝方法 RAZOR，利用共识残差衡量专家的功能可替换性，在固定剪枝预算下剪除可被存活专家替代的专家，无需梯度或恢复训练即可最大程度保留原始模型输出分布。

    

    混合专家模型每个 token 只激活少量专家，但需要存储完整的专家池。专家剪枝可以减轻这种存储负担；在固定剪枝预算下，目标是尽可能保留原始模型的输出分布。然而，专家的使用频率或贡献大小本身并不能决定移除它所造成的损害，关键在于存活的计算能否替代其功能。我们提出 RAZOR，这是一种无需训练的专家剪枝方法，通过“共识残差”（即专家输出与原始加权混合输出的偏差）来评分专家的功能可替换性。该方法在固定层输入处使用精确的单删除恒等式，考虑了存活专家的重新归一化以及路由器选择的补充机制，从而在无需梯度或恢复训练的情况下，将在校准 token 上聚合得到的局部评分用于预算化剪枝。在 GLM-4.7-Flash、Qwen3.6-35B-A3B、DeepSeek-V4-Flash-0731 和 Hy3 上，在 25% 的（剪枝率下……摘要在此处截断）

    arXiv:2609.30465v1 Announce Type: cross  Abstract: Mixture-of-experts (MoE) models activate few experts per token but store the full expert pool. Expert pruning reduces this storage burden; at a fixed pruning budget, the goal is to preserve the original model's output distribution as closely as possible. Yet an expert's usage or contribution magnitude does not by itself determine the damage caused by its removal. What matters is whether the surviving computation can replace its function. We introduce RAZOR, a training-free expert pruning method that scores functional replaceability using consensus residuals: deviations of expert outputs from the original weighted mixture. An exact single-deletion identity at a fixed layer input accounts for survivor renormalization and router-selected refill, providing local scores aggregated over calibration tokens for budgeted pruning without gradients or recovery training. On GLM-4.7-Flash, Qwen3.6-35B-A3B, DeepSeek-V4-Flash-0731, and Hy3 at 25\% an
    
[^194]: PPTBench：编码智能体能否通过结构化、可编辑的幻灯片重建视觉世界

    PPTBench: Can Coding Agents Reconstruct the Visual World through Structured, Editable Slides

    [https://arxiv.org/abs/2609.29718](https://arxiv.org/abs/2609.29718)

    该论文提出PPTBench——一个包含500个基于真实arXiv论文科学流程图的可编辑幻灯片重建基准，用于评测编码智能体从视觉内容中推断结构并以可编辑程序化对象形式实现端到端视觉重建的能力。

    

    编码智能体开始在视觉世界中发挥作用，它们如今能够构建网页、图形界面、游戏、3D场景、图表和文档。要在视觉编码中取得成功，需要弥合两个空间：推断视觉结构并将其以程序化方式表达。幻灯片是知识工作的核心媒介，被广泛用于以一种人们可直接查看和编辑的形式交流想法和开展协作。因此，幻灯片为视觉编码提供了理想的测试平台，因为它要求智能体恢复视觉结构并将其实现为可编辑的对象。然而，现有的基准测试要么依赖主观的开放式评估，要么产生不可编辑的代码输出，要么仅关注局部编辑而非端到端的视觉重建。我们提出了PPTBench，通过可编辑幻灯片重建对视觉编码进行基准评测。它包含500个任务，每个任务都基于来自真实arXiv论文的科学流程图，要求智能体重建……（摘要原文在此处截断）

    arXiv:2609.29718v1 Announce Type: cross  Abstract: Coding agents are beginning to act in the visual world. They now build webpages, GUIs, games, 3D scenes, diagrams, and documents. Success in such visual coding requires bridging two spaces: inferring visual structure and expressing it programmatically. Slides are a core medium of knowledge work, widely used to communicate ideas and collaborate in a form that people can directly inspect and edit. Therefore, they provide an ideal testbed for visual coding, as they require agents to recover visual structure and realize it as editable objects. However, existing benchmarks either rely on subjective open-ended evaluation, produce non-editable code outputs, or focus only on local editing rather than end-to-end visual reconstruction. We introduce PPTBench, which benchmarks visual coding through editable slide reconstruction. It contains 500 tasks, each based on a scientific flow diagram from a real arXiv paper and requiring agents to reconstru
    
[^195]: 经典测试理论误导LLM评判者的三种方式

    Three Ways Classical Test Theory Misleads for LLM Judges

    [https://arxiv.org/abs/2609.29709](https://arxiv.org/abs/2609.29709)

    该论文揭示经典测试理论的三个常用信度统计量在LLM评判者评估情境中含义发生扭曲——例如内部一致性系数无法区分题目设计与评判者错误的影响——因此不能直接照搬用于解读LLM评判者的表现。

    

    一个LLM评判者依据评分标准对一批回答进行打分，得到的信度值为0.52——这究竟测量了什么？评判者评估领域已开始借用经典测试理论的信度统计量，但通常并未说明每个统计量所假设的测量设计。我们证明，三个被广泛移植的统计量对评判者的含义与其对测试的含义并不相同，因为评判者情境重新排列了这些测量设计所依赖的角色。第一，基于评分标准要素计算的内部一致性系数不包含评分者维度：将某一评判者的实测错误率固定在4.72%时，随着题库的重新设计，KR-20仍在0.01至0.68之间变化，而改变评判者错误也会使该系数产生相当幅度的变动，因此题目设计与评判者错误无法被分别识别，任何单一数值都不能被解读为评判者本身的属性。第二，依存性指数 Φ(λ) 是一个方差比值……

    arXiv:2609.29709v1 Announce Type: cross  Abstract: An LLM judge scores a bank of responses against a rubric, and the reliability comes back at $0.52$. What has been measured? Judge evaluation has begun borrowing reliability statistics from classical test theory, usually without stating the measurement design each statistic assumes, and we show that three widely portable ones mean something different for a judge than for a test because the judge setting rearranges the roles those designs rest on. First, an internal-consistency coefficient computed over rubric elements contains no scorer facet. Holding one judge's measured error rate fixed at $4.72\%$, KR-20 still ranges from $0.01$ to $0.68$ as the item bank is redesigned around it, and varying judge error moves the coefficient by a comparable amount, so item design and judge error are not separately identified and no single value can be read as a property of the judge. Second, the dependability index $\Phi(\lambda)$ is a ratio of varia
    
[^196]: 数学推理中的答案顺序不变性与表征顺序敏感性

    Order-Invariant Answers, Order-Sensitive Representations in Mathematical Reasoning

    [https://arxiv.org/abs/2609.28442](https://arxiv.org/abs/2609.28442)

    该研究发现，语言模型对规则排序的内部表征越清晰（排列信噪比越高），其解决重排序数学问题的准确率就越高，揭示了答案不变性与表征不变性是两个不同的概念。

    

    在不改变含义的情况下重新排列一组数学规则的顺序，应当保持正确答案不变，但模型的内部表征是否也必须保持不变呢？我们使用合成的多步骤函数组合问题来研究这一问题，每个问题以多种规则排序呈现，且具有相同的正确答案。我们测量了准确率和排列信噪比（SNR），后者量化了排序模式相对于问题实例间差异的表征清晰程度。在16个参数量从1B到8B的语言模型上，我们发现了一个规律：更准确地解决重排序问题的模型，对不同的规则排序表征得也更加清晰。在我们评估的所有合成设置中，层级平均排列信噪比与准确率呈正秩相关，Spearman相关系数最高达到0.86。这些发现突出了答案不变性与表征不变性之间的区别：（摘要在此处截断）

    arXiv:2609.28442v1 Announce Type: cross  Abstract: Reordering a set of mathematical rules without changing its meaning should preserve the correct answer, but must a model's internal representations stay invariant too? We investigate this question using synthetic multi-step function-composition problems, each presented under multiple rule orderings with the same correct answer. We measure accuracy and permutation signal-to-noise ratio (SNR), which quantifies how distinctly ordering patterns are represented relative to variation across problem instances. Across 16 language models ranging from 1B to 8B parameters, we find a pattern: models that solve reordered problems more accurately represent different rule orderings more distinctly. Layer-averaged permutation SNR is positively rank-correlated with accuracy in every synthetic setting we evaluate, with Spearman correlations reaching 0.86. These findings highlight a distinction between answer invariance and representation invariance: suc
    
[^197]: 大知识模型：从论文到科学推理图景

    Large Knowledge Model: From Papers to a Scientific Reasoning Landscape

    [https://arxiv.org/abs/2609.27297](https://arxiv.org/abs/2609.27297)

    本文提出大知识模型（LKM），将论文表示为基于原文来源的推理图，构建包含问题、工作流和证据三个视图的科学推理图景，使科学文献成为可计算访问的共享推理资源，从而支持大规模利用文献中的科学推理过程。

    

    积累的科学知识之所以能推动科学探究，是因为已有研究发现可以帮助研究者选择新问题、设计研究方案并解释结果。要在大规模上实现这一价值，需要获取连接研究问题、科学程序、结论和证据的推理过程。我们提出了大知识模型，这是一种科学知识基础设施，能将科学文献转化为共享的、可计算访问的推理资源。LKM 将论文表示为基于原文来源的推理图，将结构化遍历与对同一对象的语义检索相结合，并对齐跨论文的相关问题、论断和推理链。这种表示构成了一个包含三个相互关联视图的科学推理图景：组织研究问题和开放方向的问题图景、揭示可复用科学程序的工作流图景，以及连接（摘要在此处截断）……的图景

    arXiv:2609.27297v1 Announce Type: new  Abstract: Accumulated scientific knowledge advances inquiry when prior findings help researchers choose new questions, design investigations, and interpret results. Realizing this value at scale requires access to the reasoning that connects research problems, scientific procedures, conclusions, and evidence. We introduce the Large Knowledge Model (LKM), a scientific knowledge infrastructure that transforms the literature into a shared, computationally accessible reasoning resource. LKM represents papers as source-grounded reasoning graphs, couples structural traversal with semantic retrieval over the same objects, and aligns related questions, claims, and reasoning chains across papers. This representation forms a Scientific Reasoning Landscape with three connected views: a Question Landscape that organizes research problems and open directions, a Workflow Landscape that exposes reusable scientific procedures, and an Evidence Landscape that conne
    
[^198]: 蒸馏你所信任的：可靠性感知的多教师在线策略蒸馏

    Distill What You Trust: Reliability-Aware Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2609.23697](https://arxiv.org/abs/2609.23697)

    提出TrustMOPD方法，以专家模型RL训练前后相对于共享参考模型的位移作为token级可靠性代理，实现无标签的多教师加权蒸馏监督分配，将学生模型性能恢复率从54.4%大幅提升至91.5%。

    

    多教师在线策略蒸馏允许学生在自身生成的轨迹上从互补的专家模型中学习。然而，基于领域路由的方法在每个样本中仅选择一个教师，并在整个响应生成过程中保持固定。这种设计既依赖于混合训练语料库中通常缺失的标签，也无法在轨迹内部所需专业知识发生变化时调整教师选择。我们提出了TrustMOPD，它用无标签的、token级别的监督分配取代样本级教师选择。在每个学生生成的前缀处，TrustMOPD将每个专家模型经强化学习（RL）后相对于共享的RL前参考模型的位移作为局部可靠性的代理指标，跨教师对这些分数进行校准，并构建加权的蒸馏目标。在数学、代码和指令遵循任务上，TrustMOPD优于最强的无标签基线，将恢复率从54.4%提升至91.5%。

    arXiv:2609.23697v1 Announce Type: new  Abstract: Multi-teacher on-policy distillation allows a student to learn from complementary specialists on its own trajectories. Domain-routed approaches, however, select one teacher per example and keep it fixed throughout the response. This design both depends on labels that mixed training corpora often lack and cannot adapt teacher selection when the expertise required changes within a trajectory. We propose \textbf{TrustMOPD}, which replaces example-level teacher selection with label-free, token-level supervision allocation. At each student-generated prefix, TrustMOPD uses each specialist's RL-induced displacement from a shared pre-RL reference as a proxy for local reliability, calibrates these scores across teachers, and constructs a weighted distillation target. Across mathematics, code, and instruction following, TrustMOPD outperforms the strongest label-free baseline, increasing the recovery ratio from $54.4\%$ to $91.5\%$ on \textsc{Singl
    
[^199]: 从概念对齐到因果锚定：思维链忠实性的干预测试

    From Concept Alignment to Causal Grounding: An Intervention Test of Chain-of-Thought Faithfulness

    [https://arxiv.org/abs/2609.23065](https://arxiv.org/abs/2609.23065)

    该论文将CoT忠实性重新定义为“内部概念锚定”问题，利用共享稀疏自编码器直接比较预测过程与CoT过程的内部概念，并首次引入三个相关性对齐指标和一个因果消融指标Δp，以检验共享概念是否因果性地驱动模型答案。

    

    思维链可以听起来合理，却可能对模型的底层推理不忠实。以往大多数工作通过输入-输出行为或输入归因来探究CoT的忠实性，而对内部计算的探索在很大程度上仍属空白。我们转而将忠实性界定为内部概念锚定问题：大语言模型（LLM）的CoT推理是否调用了支持其直接预测的相同内部概念，并且这些共享概念是否因果性地驱动其答案？使用单个共享的稀疏自编码器（SAE）——一种对LLM所使用潜在概念的可靠近似器——来编码预测过程和CoT过程，使二者的内部概念可以直接比较。我们提出了三个概念层面的相关性对齐度量指标，以及一个因果度量指标Δp，该指标通过消融共享概念并测量答案概率的下降来检验因果作用。在五个LLM和四个数据集上的实验表明，概念对齐总体上较高，正如t……（摘要在此处截断）

    arXiv:2609.23065v1 Announce Type: new  Abstract: Chain-of-thought (CoT) can sound plausible yet be unfaithful to the model's underlying reasoning. Most prior work probes CoT faithfulness through input--output behavior or input attributions, leaving internal computation largely underexplored. We instead cast faithfulness as internal concept grounding: Does a large language model's (LLM) CoT reasoning engage the same internal concepts that support the LLM's direct prediction, and do the shared concepts causally drive its answer? Encoding a prediction pass and a CoT pass with a single shared sparse autoencoder (SAE), a reliable approximator of the latent concepts LLMs use, makes their internal concepts directly comparable. We introduce three correlational metrics of concept-level alignment and a causal metric, $\Delta p$, which ablates the shared concepts and measures the drop in answer probability. Across five LLMs and four datasets, concept alignment is generally high, as indicated by t
    
[^200]: 通用多模态基础模型

    Generalized Multimodal Foundation Model

    [https://arxiv.org/abs/2609.22107](https://arxiv.org/abs/2609.22107)

    提出了一种不依赖特定模态的通用多模态基础模型，通过在大规模具有多样因果结构的合成多模态数据集上训练，使其能够适用于任意的模态组合和任意的预测任务。

    

    利用多模态数据进行预测在多种场景中被广泛应用。现有的多模态融合模型一旦部署，只能处理预定义的模态（如视觉、文本和音频）和单一任务，难以快速适应新的下游应用。因此，一个自然但相当大胆的问题随之而来：是否存在一种通用的多模态融合模型，能够应用于任意的模态组合和任意的预测任务？我们认为，统一的多模态融合模型不应依赖于特定模态，而应编码可迁移的多模态关联模式。为此，我们提出了一种简单而有效的学习范式，其基于在大规模合成的多模态数据集上进行训练，这些数据集具有多样的因果结构，能够形式化地刻画现实世界中多模态数据的生成过程。在该框架的基础上，我们提出了……（摘要原文在此处截断）

    arXiv:2609.22107v1 Announce Type: cross  Abstract: Making prediction with multimodal data is widely used in diverse scenarios. Existing multimodal fusion models, once deployed, can only handle predefined modalities (e.g., vision, text and audio) and single tasks, making it difficult to quickly adapt to new downstream applications. Therefore, a natural yet rather aggressive question arises, whether there exists a general multimodal fusion model that can be applied to arbitrary modality combinations and arbitrary prediction tasks. We argue that a unified multimodal fusion model should not depend on specific modalities and instead encode transferable patterns of multimodal correlation. To this end, we propose a simple and effective learning paradigm based on training over the generation of large-scale synthetic multimodal datasets with diverse causal structures that formally characterize the generative processes of multimodal data in real world. Building on this framework, we propose the 
    
[^201]: 检索主导的抽取式问答中的内在序列似然置信度：两个预先设定的否定结果，及其能归因与不能归因的对象

    Intrinsic Sequence-Likelihood Confidence in Retrieval-Dominated Extractive QA: Two Pre-Specified Negatives, and What They Do and Do Not Attribute

    [https://arxiv.org/abs/2609.19942](https://arxiv.org/abs/2609.19942)

    该研究通过预先注册的评估标准证明，在检索已能恢复92-99.8%最优性能的抽取式问答场景中，模型的内在序列似然置信度无论作为蒸馏触发器还是路由弃答策略的控制信号均告失效。

    

    在抽取式文档问答中——其问题由包含答案的段落生成，因此检索即可恢复任何模式组合所能达到性能的92-99.8%（无论其绝对准确率如何）——基于置信度的机制几乎没有提升空间。在专业领域语料库上微调开源语言模型后，模型自身的置信度是一个颇具吸引力的控制信号：它可用于决定哪些查询值得进一步适配、哪些答案值得信赖。我们在实验开始前预先固定的评估标准下，对四个7-9B模型家族进行了评估（其领域适配使闭卷F1最多提升+0.03），结果两种用途均告失败：在预先设定的三步迁移预算下，蒸馏触发器在全部四个模型家族上失效，单模型试点中的路由与弃答策略同样失败。在我们测试的每一个正确性标准下，仅检索即可恢复最优组合准确率的92-99.8%，留下的……

    arXiv:2609.19942v1 Announce Type: new  Abstract: In extractive document question answering whose questions were generated from the passages that contain their answers -- so that retrieval recovers 92-99.8% of what any mode combination could reach, whatever its absolute accuracy -- confidence-driven mechanisms have little to gain. Fine-tuning an open language model on a specialized domain corpus yields a model whose own confidence is a tempting control signal: it could decide which queries warrant further adaptation, and which answers to trust. We evaluate both uses under criteria fixed before the runs were executed, across four 7-9B model families whose adaptation moved closed-book F1 by at most +0.03, and both fail: a distillation trigger on all four families, under its pre-specified three-step transfer budget, and a routing-and-abstention policy in its single-model pilot. Retrieval alone recovers 92-99.8% of best-case combined accuracy under every correctness criterion we test, leavi
    
[^202]: Agora：以Git作为集体自动研究的共享内存

    Agora: Git as Shared Memory for Collective AutoResearch

    [https://arxiv.org/abs/2609.18094](https://arxiv.org/abs/2609.18094)

    Agora的核心创新是以Git中仅追加的DAG作为多个自主研究智能体的共享内存，让每条研究成果都成为可验证、可复现的不可变提交，并通过派生索引和多样性感知选择规则，使13个无中央规划的智能体持续协作近12天而不陷入重复搜索。

    

    诸如AutoResearch之类的自主研究循环表明，单个编码智能体可以在无人值守的情况下改进训练设置。但如果同时运行多个这样的智能体，每个会话都会从零开始，因此更多的智能体往往意味着更多的重复搜索，而非更多的发现。Agora是这类智能体的共享内存：研究以仅追加的有向无环图（DAG）的形式记录在Git中，使得每一条主张都是一个任何人都可以检出并重新运行的提交。每个结果、见解、假设、验证和报告都是一个不可变的提交，其父边标明它建立在哪些工作之上；一个派生索引用于揭示研究前沿、被忽视的分支以及每条主张的验证状态，而一种多样性感知的选择规则可防止社区坍缩到单一领导者上。我们描述了该系统并报告了它的首次持续使用情况：一次持续近12天的运行，13个语言模型工作者在没有任务分配、没有中央规划者的情况下，针对一个权重转……

    arXiv:2609.18094v1 Announce Type: cross  Abstract: Autonomous research loops such as AutoResearch show that one coding agent can improve a training setup unattended. Run several of them and each session starts from scratch, so more agents tend to mean more duplicated search rather than more discovery. Agora is a shared memory for such agents: research is recorded as an append-only directed acyclic graph (DAG) stored in Git, so that every claim is a commit anyone can check out and rerun. Each result, insight, hypothesis, verification, and report is an immutable commit whose parent edges say what it builds on; a derived index exposes the frontier, the neglected branches, and the verification status of each claim, and a diversity-aware selection rule keeps the community from collapsing onto one leader. We describe the system and report its first sustained use: a run of nearly 12 days in which 13 language-model workers, with no assigned tasks and no central planner, worked on a weight-tran
    
[^203]: 仅在受混淆时方可检测：经验证的重复计数对语言模型中成员证据的启示

    Detectable Only Where It Is Confounded: What Verified Duplication Counts Say About Membership Evidence in Language Models

    [https://arxiv.org/abs/2609.10830](https://arxiv.org/abs/2609.10830)

    利用公开可验证的重复计数，研究表明在真实文本的重复水平下，语言模型的预测成本与其训练数据暴露程度之间至多只有微弱的关联（秩相关约-0.08），因此预测成本低并不能可靠地证明某个句子存在于训练数据中。

    

    当语言模型发现某个句子的预测成本异常低廉时，人们很容易得出该句子存在于其训练数据中的结论。然而，几乎所有已发表的针对这一推断的测试，都不得不猜测哪些句子在训练数据中（即“成员”），哪些不在。本文消除了这种猜测。两个模型家族——OLMo-2和Pythia——公开发布了它们的预训练语料库，并且基于这些语料库的公共索引可以返回任意句子在其中出现的精确次数。这些计数使得三个问题可以直接得到回答，而这些答案形成了一把从两侧合拢的钳子。在普通文本实际具有的重复水平下，从1B到13B参数规模的五个模型，至多只带有其自身数据暴露的微弱痕迹。我们通过一种设计来测量这一痕迹：让两个模型读取相同的句子，这种设计在构造上抵消了流畅性和文本质量的影响，所得结果的秩相关性接近-0.08，而-1则代表完美关系……

    arXiv:2609.10830v1 Announce Type: new  Abstract: When a language model finds a sentence unusually cheap to predict, it is tempting to conclude that the sentence was in its training data. Almost every published test of that inference has had to guess which sentences were in the training data, the members, and which were not. This paper removes the guessing. Two model families, OLMo-2 and Pythia, publish their pretraining corpora, and a public index over those corpora returns the exact number of times any sentence appeared in each. Those counts make three questions answerable directly. The answers form a pincer, closing from two sides. At the duplication levels ordinary text actually has, five models from 1B to 13B parameters carry at most a faint trace of their own exposure. We measure that trace with a design that reads the same sentence through two models, which cancels fluency and quality by construction, and it comes to a rank correlation near -0.08, where -1 would be a perfect rela
    
[^204]: 我并不想你，但其实我想你：视觉语言模型中模态缺失的自我解释忠实性

    I Don't Miss You, but I Do: Self-Explanation Faithfulness of Modality Missingness in Vision-Language Models

    [https://arxiv.org/abs/2609.07596](https://arxiv.org/abs/2609.07596)

    该论文提出了一种介入式评估协议，揭示视觉语言模型对模态缺失的自我解释并不忠实——它们系统性地高估现有模态证据的充分性，并大幅低估恢复缺失模态对预测结果的影响。

    

    视觉语言模型越来越多地被应用于某些输入模态可能不可用的场景，然而对于它们能否忠实地解释这些缺失信息如何影响自身的预测，我们知之甚少。我们引入了一种介入式评估协议来评估模型对模态动态的自我解释：模型需要陈述每个模态单独能支持什么、恢复缺失的模态是否会改变其答案、以及现有证据是否充分；随后我们执行相应的模态介入操作，并将这些陈述与模型的实际行为进行比较。我们在四项任务上评估了来自两个模型家族的八个开源权重视觉语言模型，这些任务涵盖互补和同构的文本-图像设置以及多视角驾驶设置。我们发现模型存在系统性地夸大现有模态证据充分性的倾向。模型大幅低估了恢复缺失模态所产生的影响：任务级……

    arXiv:2609.07596v1 Announce Type: new  Abstract: Vision-language models are increasingly used in settings where some input modalities may be unavailable, yet we know little about whether they can faithfully explain how such missing information affects their own predictions. We introduce an interventional protocol for evaluating self-explanations of modality dynamics: models state what each modality alone would support, whether restoring a missing modality would change their answer, and whether the available evidence is sufficient; we then execute the corresponding modality intervention and compare these claims with the model's realized behavior. We evaluate eight open-weight VLMs from two model families across four tasks spanning complementary and isomorphic text-image settings and a multi-view driving setting. We find a systematic tendency to overstate the sufficiency of available modality evidence. Models substantially underestimate the effect of restoring missing modalities: task-le
    
[^205]: 从边缘到联合的门票：扩散语言模型中一步式块生成的耦合噪声蒸馏

    A Ticket from Marginals to Joints: Coupled-Noise Distillation for One-Step Block Generation in Diffusion Language Models

    [https://arxiv.org/abs/2609.06324](https://arxiv.org/abs/2609.06324)

    提出CONDOR方法，通过耦合噪声蒸馏从零训练扩散语言模型，使其在掩码嵌入受高斯噪声扰动时仅用一次前向传播即可生成连贯的整块文本，且无需目标侧编码器或自回归教师。

    

    扩散语言模型并行预测一个块中的所有词元，但单次前向传播是从各自的边缘分布中采样每个位置，因此这些词元不一定能构成一个连贯的块。我们提出一个问题：当离散掩码模型的掩码嵌入受到采样高斯噪声场扰动时，该模型能否在单次传播中生成整个块——相同的噪声应给出相同的连贯续写，而不同的噪声应给出不同的续写。我们提出了CONDOR（Coupled-Noise Distillation for One-Step Readout，面向一步式读出的耦合噪声蒸馏），它无需目标侧编码器或自回归教师即可从头训练这样的模型。训练结合了两种信号。在真实文本上，模型在多个噪声样本下预测被掩码的词元，并仅通过与真实值最匹配的那个样本进行监督，从而使不同的噪声可以专精于不同的续写。对于其余样本，模型则细化自身的单次传播预测……

    arXiv:2609.06324v2 Announce Type: replace  Abstract: Diffusion language models (dLLMs) predict all tokens of a block in parallel, but a single forward pass samples each position from its own marginal distribution, so the tokens need not form a coherent block. We ask whether a discrete masked model can commit an entire block in one pass when its mask embeddings are perturbed by a sampled Gaussian noise field: the same noise should give the same coherent continuation, and different noise should give different ones. We propose CONDOR (Coupled-Noise Distillation for One-Step Readout), which trains such a model from scratch without a target-side encoder or an autoregressive teacher. Training combines two signals. On real text, the model predicts masked tokens under several noise samples and is supervised only through the sample that fits the ground truth best, so different noise can specialize to different continuations. For the remaining samples, the model refines its own one-pass predicti
    
[^206]: 安全监控器大多只能捕获模型本已拒绝的内容

    Safety Monitors Mostly Catch What the Model Already Refuses

    [https://arxiv.org/abs/2609.05797](https://arxiv.org/abs/2609.05797)

    现有安全监控器的标准评估方式掩盖了其真实短板——它们主要捕获模型本已拒绝的有害请求，而在模型实际会回答的请求上召回率骤降，通过含蓄化改写请求可让 44%–93% 的有害请求绕过监控并产生有害输出。

    

    安全监控器通常通过其在有害提示上的召回率来评估，而不考虑目标模型是否会回答这些提示。然而，监控器最重要的作用恰恰体现在模型确实会回答的提示上。我们精确测量了这类提示上的召回率，其定义方式是采样目标模型并对其响应进行判定。在四个文本防护器、两个激活探针以及 Latent Guard 上，在 1% 假阳性率下的召回率在这一子集上急剧下降：在相同阈值下，每个监控器捕获模型拒绝的请求的频率是捕获模型回答的请求的 1.1 到 6.4 倍。标准指标掩盖了这一问题；大多数监控器的 AUROC 仍保持在 0.85 以上。在保持意图不变并经核验的前提下，将每个请求改写得更含蓄，可使合规率提高 28 倍，并降低每个监控器的标记率。在改写后被新回答的请求中，根据所用监控器的不同，有 44% 到 93% 绕过了监控，且其中大部分输出被评定为有害。我们将这一差距归因于（摘要在此处截断）

    arXiv:2609.05797v3 Announce Type: replace  Abstract: Safety monitors are evaluated by recall on harmful prompts, regardless of whether the target model would answer them. Yet a monitor matters most on the prompts the model does answer. We measure recall on exactly those prompts, defined by sampling the target model and judging its responses. Across four text guards, two activation probes, and Latent Guard, recall at a 1% false positive rate falls sharply on this subset: at a common threshold, every monitor catches the requests the model refuses 1.1 to 6.4 times as often as the requests it answers. Standard metrics hide this; AUROC stays above 0.85 for most monitors. Rewriting each request to be less explicit, with intent held fixed and verified, raises compliance 28-fold and lowers every monitor's flag rate. Of the requests newly answered after rewriting, 44 to 93% slip past the monitor, depending on which is used, and most of their completions are graded harmful. We trace the gap to e
    
[^207]: ShallowStream：先浅层索引、后深层回答的流式视频理解

    ShallowStream: Index Shallow then Answer Deep for Streaming Video Understanding

    [https://arxiv.org/abs/2609.02780](https://arxiv.org/abs/2609.02780)

    提出ShallowStream框架，采用“浅层索引、深层回答”的策略，利用MLLM浅层高效处理流式视频帧，从而显著降低计算开销并抑制KV缓存的增长。

    

    流式视频理解是现实应用中的一项关键能力，涵盖具身智能、自动驾驶、工业监控、监视预警以及可穿戴助手等领域。然而，使用多模态大语言模型（MLLM）处理连续视频流在计算上代价高昂。已有工作探索了通过视觉token剪枝、token合并、量化、按需帧检索和上下文卸载等方式来降低流式处理的开销。然而，大多数现有方法忽视了模型深度这一维度。对不断到来的视频帧反复执行完整深度的MLLM预填充代价极高，会带来可观的计算开销，并使KV缓存以与预填充深度成正比的速度增长。为应对这些挑战，我们提出了ShallowStream，一种新颖的框架，它利用MLLM的浅层同时执行帧级……（原文摘要在此处截断）

    arXiv:2609.02780v1 Announce Type: cross  Abstract: Streaming video understanding is a critical capability for real-world applications, including embodied intelligence, autonomous driving, industrial monitoring, surveillance and early warning, and wearable assistants. However, processing continuous video streams with multimodal large language models (MLLMs) is computationally expensive. Existing efforts have explored reducing streaming overhead through visual token pruning, token merging, quantization, on-demand frame retrieval, and context offloading. However, most existing methods overlook the dimension of model depth. Repeatedly executing full-depth MLLM prefill over incoming frames is prohibitively expensive, incurring substantial computational overhead and causing the KV cache to grow at a rate directly proportional to the prefill depth. To address these challenges, we propose ShallowStream, a novel framework that leverages the shallow layers of an MLLM to simultaneously perform fr
    
[^208]: Lot Machine：从拍卖目录中进行多模态拍品信息抽取

    Lot Machine: Multimodal Lot Extraction from Auction Catalogs

    [https://arxiv.org/abs/2608.30510](https://arxiv.org/abs/2608.30510)

    本文提出了一个利用视觉-语言模型从历史拍卖目录中自动提取结构化拍品元数据的流水线，并在不同提示策略、受限解码框架和部署条件下进行了系统评估，以满足文化遗产机构在预算、算力和数据隐私方面的实际需求。

    

    对于溯源研究和艺术市场研究而言，拍卖目录是追踪特定物品在时间和空间上流转的重要资源。虽然历史拍卖目录遵循既定的领域惯例，但其内部格式仍然高度多变，且由于缺乏机器可读的拍卖拍品表示，其大规模分析目前受到限制。我们提出了一个流水线，可以从 German Sales（一个涵盖19和20世纪历史拍卖与销售目录的大型数据库）中自动提取结构化的拍品级元数据。基于人工标注的代表性目录页面测试集，我们在不同的提示策略和受限解码框架下对视觉-语言模型进行了评估。为了反映文化遗产机构面临的实际约束，包括预算、计算资源和数据隐私要求，我们在不同的部署方式下对这些方法进行了基准测试。

    arXiv:2608.30510v1 Announce Type: cross  Abstract: For provenance research and art market studies, auction catalogs are an essential resource to trace specific objects over time and space. While historical auction catalogs follow established domain conventions, their internal formatting remains highly variable, and their large-scale analysis is currently restricted by the lack of machine-readable representations of the auction lots. We propose a pipeline to automatically extract structured lot-level metadata from German Sales, a large database of historical auction and sales catalogs from the 19th and 20th centuries. Using a manually annotated test set of representative catalog pages, we evaluate Vision-Language Models (VLMs) under varying prompt strategies and constrained decoding frameworks. To reflect the practical constraints faced by cultural heritage institutions, including budget, compute resources, and data privacy requirements, we benchmark the methods across different deploym
    
[^209]: 重建正确片段：评估超越长上下文的交错对话记忆

    Reconstructing the Right Episode: Evaluating Interleaved Conversational Memory Beyond Long Context

    [https://arxiv.org/abs/2608.25655](https://arxiv.org/abs/2608.25655)

    本文提出了SCALE-QA基准，用于评估聊天助手在平坦混合主题线程中推断早期因果片段以正确完成后续任务的能力，填补了现有记忆基准忽视片段完整性失败的空白。

    

    与聊天助手的对话日益在单个长时间运行的线程中跨越多个主题，这对记忆系统构成了挑战。现有的长上下文和记忆基准通常暴露会话或主题边界，或探究直接的个人记忆问题。这些设置低估了一种更难的助手记忆场景：一个平坦的混合主题线程，其中系统必须推断哪个早期片段使后续任务决策有效。我们引入了SCALE-QA，一个针对平坦未分段线程的约束基础任务问答基准，旨在解决片段完整性失败问题。该数据集包含10个领域中的3,000个审计问题，使用确定性四选一多项选择评分，并包含一个确定性运行时构建器；实验使用所有3,000个问题通过128k上下文，以及一个分层的400问题诊断集在1M上下文。SCALE-QA问题是普通任务导向请求，其正确答案依赖于早期引入的因果相关证据。

    arXiv:2608.25655v1 Announce Type: new  Abstract: Conversations with chat assistants increasingly span many topics in a single long-running thread, challenging memory systems. Existing long-context and memory benchmarks often expose session or topic boundaries, or probe direct personal-memory questions. These settings understate a harder assistant-memory regime: a flat mixed-topic thread where the system must infer which earlier episode makes a later task decision valid. We introduce SCALE-QA, a constraint-grounded task QA benchmark for flat unsegmented threads targeting episode integrity failure. The dataset contains 3,000 audited questions across 10 domains, uses deterministic four-way multiple-choice grading, and includes a deterministic runtime builder; experiments use all 3,000 questions through 128k and a stratified 400-question diagnostic at 1M. SCALE-QA questions are ordinary task-oriented requests whose correct answer depends on causally related evidence introduced earlier in t
    
[^210]: 一次成功不等于可靠：Thinkingbox——面向有状态业务流程中智能体的沙盒与基准

    One Success Isn't Reliability: Thinkingbox, a Sandbox and Benchmark for Agents in Stateful Business Workflows

    [https://arxiv.org/abs/2608.19741](https://arxiv.org/abs/2608.19741)

    本文提出了Thinkingbox，一个支持有状态业务流程智能体评估的沙盒和基准，强调通过多轮交互、策略遵循和持久状态转换来确保可靠性，而非仅关注单次成功。

    

    arXiv:2608.19741v1 公告类型：新公告  摘要：近期智能体基准测试越来越多地将评估建立在可执行环境中，从代码修复到网页导航、应用程序接口和函数调用。然而，完成代码之外的重要工作不仅仅需要生成合理的响应或有效的工具调用：智能体必须在多轮交互中收集缺失信息，遵循领域策略，协调依赖工具，并在无附带影响的情况下实现正确的持久状态转换。在本文中，我们介绍了Thinkingbox，一个用于工具-智能体-用户交互的沙盒，它提供隔离的MCP兼容工具会话、完整的执行轨迹以及基于终端后端状态的结局评估。基于此沙盒，Thinkingbox-bench包含507个策略条件化工作流，涵盖零售、酒店、汽车保险、新银行内部IT和咨询IT/人力资源支持等多种场景。每次尝试都通过任务特定的可执行检查进行评估。

    arXiv:2608.19741v1 Announce Type: new  Abstract: Recent agent benchmarks increasingly ground evaluation in executable environments, from code repair to web navigation, app APIs, and function calling. Yet completing consequential work beyond code requires more than producing a plausible response or valid tool call: agents must gather missing information over multiple turns, follow domain policies, coordinate dependent tools, and realize the correct persistent state transition without collateral effects. In this paper, we introduce Thinkingbox, a sandbox for tool-agent-user interaction that provides isolated MCP-compatible tool sessions, complete execution traces, and outcome evaluation over terminal backend state. Built on this sandbox, Thinkingbox-bench contains 507 policy-conditioned workflows across numerous scenarios, including retail, hospitality, auto insurance, neobank internal IT, and consulting IT/HR support. Each attempt is evaluated by task-specific executable checks that acc
    
[^211]: 基于GRPO的大语言模型遗忘中奖励规范与基准可靠性的实证研究

    An Empirical Study of Reward Specification and Benchmark Reliability in GRPO-based LLM Unlearning

    [https://arxiv.org/abs/2608.17804](https://arxiv.org/abs/2608.17804)

    本研究发现，在基于GRPO的大语言模型遗忘中，不同奖励设计导致的优化成功与行为遗忘并不等价，且现有基准评估指标可能相互矛盾，需警惕奖励黑客和基准可靠性问题。

    

    实际的大语言模型（LLM）遗忘通常通过两个目标进行评估：抑制目标特定知识和保留非目标实用性。在生成式问答中，这留下了一个未明确规定的第三种行为：当目标相关提示允许更广泛的答案而不泄露目标特定信息时，模型应在此级别回答，而不是泄露、回避或拒绝。我们在受控的LoRA-GRPO RWKU设置中研究了这个规范问题，比较了四种奖励设计，涵盖词汇抑制、反拒绝塑造、基于评分标准的广泛回答以及显式拒绝对比，并伴有或无SFT预热。实验表明，优化成功并不等同于行为遗忘：RWKU遗忘分数、保留完成审计、终端训练滚动审计和训练动态可能指向不同结论。我们将这些分歧追溯到奖励黑客终点、GRPO中的策略支持限制以及基准概率问题。

    arXiv:2608.17804v1 Announce Type: cross  Abstract: Practical LLM unlearning is usually evaluated through two objectives: suppress target-specific knowledge and preserve non-target utility. In generative QA, this leaves a third behavior underspecified: when a target-adjacent prompt admits a broader answer without target-specific leakage, the model should answer at that level rather than leak, evade, or refuse. We study this specification problem in a controlled LoRA-GRPO RWKU setting, comparing four reward designs that span lexical suppression, anti-refusal shaping, rubric-based broad answering, and an explicit refusal contrast, with and without SFT warm-up. The experiments show that optimization success is not equivalent to behavioral unlearning: RWKU forget scores, held-out completion audits, terminal training-rollout audits, and training dynamics can point to different conclusions. We trace these disagreements to reward-hacking endpoints, policy-support limits in GRPO, benchmark prob
    
[^212]: 跨模型记忆迁移：通过目标端阅读器适配

    Cross-Model Memory Transfer via Target-Side Reader Adaptation

    [https://arxiv.org/abs/2608.17050](https://arxiv.org/abs/2608.17050)

    本文研究了跨模型记忆迁移中，冻结记忆与目标端阅读器相对重要性，发现轻量级阅读器适配是关键，而非记忆本身。

    

    arXiv:2608.17050v1 公告类型：交叉 摘要：改进大型语言模型中知识使用的方法通常分为两种模式。非参数检索提供对外部知识的灵活访问，但增加了检索延迟、上下文开销，并且与主干的集成较浅。参数化适配在推理时高效，但将知识与模型权重纠缠在一起，且难以更新、审计或迁移。Engram风格的哈希记忆占据了一个中间模式：它将学习到的信息存储在外部可寻址表中，但通过一个小型学习阅读器来消费该表。这引发了一个基本问题：当这种记忆跨骨干移动时，冻结的记忆本身更重要，还是目标端阅读器更重要？我们通过跨模型冻结记忆提取来研究这个问题，其中在源模型上训练的记忆被冻结并附加到不同的目标模型上，仅训练轻量级阅读器。消融实验表明...

    arXiv:2608.17050v1 Announce Type: cross  Abstract: Methods for improving knowledge use in large language models typically fall into two regimes. Non-parametric retrieval offers flexible access to external knowledge, but adds retrieval latency, context overhead, and only shallow integration with the backbone. Parametric adaptation is efficient at inference time, but entangles knowledge with model weights and can be hard to update, audit, or transfer. Engram-style hashed memory occupies a middle regime: it stores learned information in an external, addressable table, yet consumes that table through a small learned reader. This raises a basic question: when such a memory is moved across backbones, what matters more, the frozen memory itself or the target-side reader? We study this question through cross-model frozen-memory extraction, in which a memory trained on a source model is frozen and attached to a different target model, with only a lightweight reader trained. Ablations show that 
    
[^213]: AI聊天机器人能否找到专家会找到的内容？模型、用户角色和样本量对医学问题研究检索的影响

    Do AI chatbots find what experts would? Effects of model, user role, and sample size on study retrieval for medical questions

    [https://arxiv.org/abs/2608.13786](https://arxiv.org/abs/2608.13786)

    本研究系统评估了三种主流AI聊天机器人在医学问题检索中的表现，发现其检索质量受模型类型、用户角色和样本量显著影响，且与专家标准存在差距。

    

    大语言模型（LLM）聊天机器人越来越多地被用于回答临床问题，并引用相关临床研究作为支持。先前的研究主要集中在引用伪造上，而对检索到的研究质量及其选择驱动因素的评估存在空白。在本研究中，我们评估了三个通用LLM聊天机器人：Claude Sonnet 5、Gemini 3.1 Pro和ChatGPT GPT-5.5。我们使用改编自2026年Cochrane系统评价数据库第6期和第7期中20个评价问题的临床问题对模型进行提示，模拟患者、临床医生和证据综合研究人员角色。每个聊天机器人在每种用户角色下进行四次独立重复查询，共产生720个响应。每个聊天机器人被要求用主要临床引用支持其答案，我们将其与Cochrane评价的纳入和排除研究集进行基准比较。平均而言，一个聊天机器人响应检索...

    arXiv:2608.13786v1 Announce Type: cross  Abstract: Large language model (LLM) chatbots are increasingly used to answer clinical questions with citations to relevant clinical studies. Prior research has largely focused on citation fabrication, leaving a gap in evaluating the quality of retrieved studies and the factors driving their selection. In this study, we evaluated three general-purpose LLM chatbots: Claude Sonnet 5, Gemini 3.1 Pro, and ChatGPT GPT-5.5. We prompted the models with clinical questions adapted from 20 review questions in Issues 6 and 7 of the 2026 Cochrane Database of Systematic Reviews, simulating patient, clinician, and evidence-synthesis researcher roles. Each chatbot was queried under each user role with four independent repetitions, yielding 720 responses. Each chatbot was asked to support its answers with primary clinical citations, which we benchmarked against the included and excluded study sets of the Cochrane reviews. On average, a chatbot response retrieve
    
[^214]: 大语言模型驱动的小盘股交易：融合财经新闻情绪、宏观经济指标与技术信号

    Large Language Model-Driven Small-Capitalization Trading: Integrating Financial News Sentiment, Macroeconomic Indicators, and Technical Signals

    [https://arxiv.org/abs/2608.12283](https://arxiv.org/abs/2608.12283)

    本文提出一种不确定性感知的投资组合构建方法，将大语言模型预测的风险分解为偶然和认知部分并直接纳入协方差矩阵，在罗素2000股票上验证了纯阿尔法和纯贝塔触发机制优于交集机制。

    

    arXiv:2608.12283v1 公告类型：交叉 摘要：大语言模型能从财经新闻中提取比固定情感词典更丰富的信号，近期研究已探索将这些信号输入投资组合构建。我们研究了一种不确定性感知的构建方法，将模型预测的风险——分解为偶然不确定性和认知不确定性——直接输入投资组合分配器的协方差矩阵，而非将投资组合风险视为固定或仅调整预期收益。我们在罗素2000股票上评估该流程，采用三种选股机制：纯阿尔法触发，隔离不受宏观指标解释的异常股价波动；纯贝塔触发，捕捉股票本身启动前的宏观指标变动；以及贝塔交集触发，两个渠道同时一致。在整个持有期网格上，分离的纯阿尔法和纯贝塔分支通常在夏普比率和收益上优于贝塔交集。两个时间跨度尤其具有信息量。

    arXiv:2608.12283v1 Announce Type: cross  Abstract: Large language models can extract richer signals from financial news than fixed sentiment lexicons, and recent work has explored feeding such signals into portfolio construction. We study an uncertainty-aware construction that feeds model-predicted risk -- decomposed into aleatoric and epistemic components -- directly into the covariance matrix of portfolio allocators, rather than treating portfolio risk as fixed or adjusting only expected returns. We evaluate the pipeline on Russell 2000 equities under three stock-selection regimes: a pure-alpha trigger that isolates abnormal stock moves not explained by macro indicators, a pure-beta trigger that captures macro-indicator moves before the stock itself fires, and a beta trigger in which both channels agree. Across the full holding-period grid, the separated pure-alpha and pure-beta legs usually dominate the beta intersection on Sharpe and return. Two horizons are especially informative.
    
[^215]: 离散扩散的单纯形松弛

    Simplex Relaxation for Discrete Diffusion

    [https://arxiv.org/abs/2608.10615](https://arxiv.org/abs/2608.10615)

    本文提出 Simplax，通过精确的 Dirichlet-类别增广，在保持均匀离散扩散的类别腐蚀过程和边际分布不变的前提下，为每个被腐蚀的类别状态耦合一个辅助单纯形连续变量，从而得到可处理的 Rao–Blackwell 化反向桥接目标函数和随机反向采样器。

    

    用于类别生成的离散扩散模型由一个腐蚀核定义，该腐蚀核决定了中间状态空间以及与之相关的反向预测问题。在这一类模型中，均匀扩散已得到广泛发展，近期的工作将其类别腐蚀过程与连续表示及动力学联系起来。受这一观点启发，我们提出这样一个问题：能否在保持类别腐蚀过程不变的前提下，为均匀离散扩散增广一个显式的连续状态？我们提出了 Simplax，这是一种精确的 Dirichlet-类别增广方法，它将每个被腐蚀的类别状态与一个辅助的单纯形值变量耦合，同时保持均匀扩散过程作为其类别边际分布。这种增广带来了一个可处理的 Rao–Blackwell 化反向桥接目标和一个随机反向采样器，同时仍保留被腐蚀的类别状态作为去噪器的输入（原文在此处截断）。

    arXiv:2608.10615v2 Announce Type: replace  Abstract: Discrete diffusion models for categorical generation are defined by a corruption kernel, which determines the intermediate state space and the associated reverse prediction problem. Within this family, uniform diffusion has been extensively developed, with recent work connecting its categorical corruption process to continuous representations and dynamics. Motivated by this view, we ask whether uniform discrete diffusion can be augmented with an explicit continuous state while leaving its categorical corruption process unchanged. We introduce Simplax, an exact Dirichlet--categorical augmentation that couples each corrupted categorical state with an auxiliary simplex-valued variable while preserving the uniform diffusion process as its categorical marginal. This augmentation yields a tractable Rao--Blackwellized reverse-bridge objective and a stochastic reverse sampler, while retaining the corrupted categorical state as the denoiser i
    
[^216]: LegalPincite：多层次法律信息检索数据集

    LegalPincite: Multi-level Legal Information Retrieval Dataset

    [https://arxiv.org/abs/2608.03756](https://arxiv.org/abs/2608.03756)

    该论文提出了基于欧盟法院判决构建的大规模法律信息检索数据集LegalPincite，通过消除查询文本数据泄漏、保留全部段落语料并提供案例和段落两个级别的引用标注，解决了现有法律IR数据集任务设定不现实而导致性能虚高的问题。

    

    法律信息检索（IR）中的一项常见任务是从判例法集合中找到相关的法律文献来源。虽然法律实践中往往需要对具体判例段落进行精确引用（pincite），但现有的大多数公开法律IR数据集缺乏段落级别的引用标注。而且，公开可用的包含此类信息的数据集在查询文本中存在数据泄漏问题，并从语料库中排除了既不引用他人也未被他人引用的段落，从而营造出一种不现实且过度简化的检索环境，可能导致性能被高估。为了解决这些局限，我们贡献了一个基于欧盟法院（CJEU）判决构建的大规模法律IR数据集。该数据集包含：(i) 去除引用信息后的掩码处理的案例/段落查询；(ii) 包含所有段落的完整语料库；(iii) 案例级别和段落级别的真实引用标注，并经过部分人类专家验证。

    arXiv:2608.03756v2 Announce Type: cross  Abstract: A common task in legal Information Retrieval (IR) is to find relevant legal sources from case-law collections. While legal practice often requires pinpoint citations (pincites) to specific case paragraphs, most existing public legal IR datasets lack paragraph-level citation annotations. Yet, publicly available datasets with such information contain data leakage in the query text and exclude paragraphs that are neither citing nor cited from the corpora, creating an unrealistic and oversimplified retrieval setting, potentially leading to inflated performance. To address these limitations, we contribute a large-scale legal IR dataset constructed from Court of Justice of the European Union (CJEU) judgments. The dataset contains: (i) masked case/paragraph queries, with removed citation information; (ii) a corpus that includes all paragraphs; and (iii) case- and paragraph-level ground truth citations, with partial human expert validation. Ou
    
[^217]: 从大语言模型直接构建消歧知识库

    Direct Construction of Disambiguated Knowledge Bases from Large Language Models

    [https://arxiv.org/abs/2608.03729](https://arxiv.org/abs/2608.03729)

    提出GPTKB 2.0方法，通过对实体、关系和类别的即时消歧机制，直接从大语言模型构建了首个百万级规模的消歧知识库，包含超过100万个实体和3840万条三元组。

    

    自动化知识库构建（AKBC）是自然语言处理领域的一项核心任务，近期有研究提出直接从大语言模型（LLM）生成知识库，将模型本身视为知识来源。然而，大语言模型本身并不具备实体的表示形式，这导致知识库中出现重复条目以及实体混淆的问题。我们提出了GPTKB 2.0，这是一种直接从大语言模型构建消歧知识库的方法论。GPTKB 2.0 引入了对实体、关系和类别的即时消歧机制，并经过精心设计以同时满足可扩展性和消歧准确性两方面的要求。我们分析了核心设计决策，并刻画了准确性、规模与成本之间的权衡关系。我们大规模地执行了GPTKB 2.0，得到了一个包含超过100万个消歧实体和3840万条三元组的实体化知识库。这是首个对实体、关系和类别进行显式内部规范化的百万级规模大语言模型原生知识库。

    arXiv:2608.03729v3 Announce Type: replace-cross  Abstract: Automated Knowledge Base Construction (AKBC) is a core NLP task, and recent work proposes generating knowledge bases directly from large language models (LLMs), treating the model itself as the knowledge source. However, LLMs natively possess no representation of entities, leading to duplicate entries as well as conflations. We propose GPTKB 2.0, a methodology for constructing disambiguated KBs directly from LLMs. GPTKB 2.0 incorporates on-the-fly disambiguation of entities, relations and classes, and is meticulously designed to satisfy both scalability and disambiguation accuracy. We analyze the central design decisions and characterize the trade-offs between accuracy, scale, and cost. We execute GPTKB 2.0 at scale, obtaining a materialized KB containing over 1M disambiguated entities and 38.4M triples. This represents the first million-scale LLM-native KB with explicit internal canonicalization of entities, relations, and cla
    
[^218]: 关系先验作为基于大语言模型的多智能体系统中的收敛压力

    Relational Priors as Convergence Pressure in LLM-Based Multi-Agent Systems

    [https://arxiv.org/abs/2608.03239](https://arxiv.org/abs/2608.03239)

    该论文提出将LLM多智能体系统中智能体间的关系显式化为带符号的成对先验并注入系统提示，揭示了这种“收敛压力”现象——积极关系虽能提升公共资源治理的可持续性和主观问题的共识度，但在客观问答中反而会降低最终答案的准确性。

    

    基于大语言模型的多智能体系统（LLM-MAS）通过角色设定、辩论协议和聚合规则进行设计，这些选择会形成关于信任、怀疑、服从或协作的隐含期望。我们将智能体间的关系显式化为带符号的成对先验，以自然语言形式呈现并添加到系统提示中，同时保持任务协议固定不变。在公共资源治理和多智能体辩论两项任务中，这些先验会改变智能体协调或达成一致的难易程度，我们将这种模式称为“收敛压力”。更积极的关系通常能提高GovSim关系先验扫描中的可持续性，并增加主观问题上的共识。相较于消极关系所带来的改善，与相对于无先验提示所带来的收益有所不同。在客观问答任务上，完全积极的关系先验通常比无先验基线产生更低的最终答案准确率，且某些条件下会产生更频繁但准确性更低的共识。

    arXiv:2608.03239v2 Announce Type: replace  Abstract: Large language model-based multi-agent systems (LLM-MAS) are designed through roles, debate protocols, and aggregation rules. These choices create implicit expectations of trust, skepticism, deference, or collaboration. We make inter-agent relations explicit as signed pairwise priors, rendered in natural language and added to system prompts while keeping the task protocol fixed. Across commons governance and multi-agent debate, these priors change how readily agents coordinate or agree, a pattern we call convergence pressure. More positive relations generally improve sustainability within the GovSim relational-prior sweep and increase consensus on subjective questions. These improvements over negative relations differ from gains over no-prior prompting. On objective QA, fully positive priors usually yield lower final-answer accuracy than the no-prior baseline, and some conditions produce more frequent but less accurate consensus. Eff
    
[^219]: AgentSnare：学习延迟、转移与化解自主渗透代理

    AgentSnare: Learning to Delay, Divert, and Defuse Autonomous Penetration Agents

    [https://arxiv.org/abs/2607.26998](https://arxiv.org/abs/2607.26998)

    AgentSnare提出了一种轨迹自适应的欺骗防御系统，通过基于渗透代理交互历史动态构建诱饵工件，持续将自主渗透代理从真实目标引开，克服了传统静态蜜罐痕迹易被识别绕过的缺陷。

    

    大型语言模型（LLM）代理通过“观察-动作”循环实现渗透测试自动化，根据工具返回的观察结果选择动作。这种依赖性使防御者能够注入欺骗性观察结果，从而误导代理的决策过程。然而，现有的防御手段严重依赖在攻击前预先植入环境中的静态、孤立的人工痕迹（artifacts）。先进的代理能够逐步识别并绕过这些痕迹，最终将其攻击重心重新聚焦到真实目标上。为解决这一问题，我们提出了AgentSnare，一种轨迹自适应的欺骗系统，它能够动态展开诱饵环境，持续将渗透代理从真实目标引开。具体而言，AgentSnare采用了一个工件构建策略模型，该模型以代理的交互历史和诱饵状态为条件构建候选工件。随后，AgentSnare会对这些（候选工件进行验证）……

    arXiv:2607.26998v4 Announce Type: replace-cross  Abstract: Large language model (LLM) agents automate penetration testing through an observation-action loop, selecting actions based on observations returned by tools. This dependence allows defenders to inject deceptive observations that can mislead the agent's decision-making process. However, existing defenses rely heavily on static, isolated artifacts planted in the environment prior to an attack. Advanced agents can progressively recognize and bypass these artifacts, ultimately refocusing their exploitation attempts on the real target. To address this issue, we introduce AgentSnare, a trajectory-adaptive deception system that dynamically unfolds a decoy environment to continually steer the penetration agent away from the real target. Specifically, AgentSnare employs an artifact-construction policy model that constructs candidate artifacts conditioned on the agent's interaction history and decoy state. AgentSnare then validates these
    
[^220]: 使用微调大语言模型识别英国警方事件记录中的脆弱性指标

    Using Fine-Tuned LLMs to Identify Indicators of Vulnerability in UK Police Incident Logs

    [https://arxiv.org/abs/2607.18446](https://arxiv.org/abs/2607.18446)

    本研究开发了一个基于本地部署开源大语言模型的多阶段分类流程，通过重复推理、标签聚合、人工审查和统计校正，从英国警方事件记录中估计心理疾病、药物滥用、酒精依赖和无家可归等脆弱性指标的普遍程度。

    

    目的：了解日常警务工作中有多少涉及弱势群体，可以为资源配置、培训和多机构协作响应提供参考，然而行政数据对此提供的洞察有限。我们探索基于大语言模型的分类流程（基于开源美国警方数据开发）能否被调整用于估计英国警方事件叙述中四种脆弱性指标的普遍程度——心理疾病、药物滥用、酒精依赖和无家可归——以及其输出在何时可被视为可靠的测量结果。方法：我们分析了来自英国某警察部队的近3,000份去标识化事件记录，采用了一个多阶段流程，该流程结合了重复模型推理、标签聚合、结构化人工审查和统计校正。该流程在本地托管的开源权重LLM上运行，以适应警方必须工作的安全环境。结果：大语言模型能够产生有意义的（尽管不完美的）普遍程度估计……

    arXiv:2607.18446v2 Announce Type: replace  Abstract: Purpose: Understanding how much of routine policing involves vulnerable people could inform resourcing, training, and multi-agency response, yet administrative data provide limited insight. We explore whether an LLM-based classification pipeline, developed on open-source US police data, can be adapted to estimate the prevalence of four vulnerability indicators - mental ill health, substance misuse, alcohol dependence, and homelessness - in UK police incident narratives, and when outputs can be treated as defensible measurements.   Methods: We analyse nearly 3,000 de-identified incident logs from a UK police force, using a multi-stage pipeline combining repeated model inference, label aggregation, structured human review, and statistical correction. The pipeline runs on a locally hosted open-weight LLM, reflecting the secure environments police must work in.   Results: LLMs can produce meaningful, if imperfect, prevalence estimates at
    
[^221]: 形式语义结构能解释多少人类标签变异？：自然语言推理中的群体层面效应与条目层面天花板

    How Much Human Label Variation Does Formal Semantic Structure Explain?: Group-Level Effects and Item-Level Ceilings in NLI

    [https://arxiv.org/abs/2607.15870](https://arxiv.org/abs/2607.15870)

    该研究通过预注册分析直接测量发现，形式语义结构对自然语言推理中人类标签变异的解释力有限——群体层面上非纯向上单调假设的标签熵显著更高，但条目层面上形式语义特征仅能解释3.3%–3.6%的熵方差。

    

    自然语言推理中的人类标签变异日益被视为信号而非噪声，但形式语义结构到底能解释其中多少，此前尚未被直接测量过。我们在ChaosNLI的3,113个SNLI和MNLI条目上对此进行测量，使用了经MED验证的基于规则的算子与单调性标注器（在编辑位置上的一致性为0.883，在分析所使用的句子级摘要上为0.807）、三个预注册的分析模块，并完整报告了阴性结果。研究得出三个界限。第一，群体层面的边界：非纯粹向上单调的假设显示出可靠更高的标签熵（Cliff's delta = -0.284），基于秩的检验表明该效应在控制算子存在与长度等缩减因素后依然稳健，但有界结果敏感性检验削弱了长度防御的回归形式。第二，条目层面的天花板：同样的形式语义特征仅解释了熵方差的3.3%至3.6%（原文摘要在此处截断）。

    arXiv:2607.15870v2 Announce Type: replace  Abstract: Human label variation in natural language inference is increasingly treated as signal rather than noise, but how much of it formal semantic structure explains has not been measured directly. We measure it on the 3,113 SNLI and MNLI items of ChaosNLI, using a rule-based operator and monotonicity tagger validated against MED (0.883 agreement at the edit site, 0.807 on the sentence-level summary our analyses consume), three preregistered analysis blocks, and full reporting of negative results. Three bounds emerge. First, a group-level boundary: hypotheses that are not purely upward monotone show reliably higher label entropy (Cliff's delta = -0.284), and rank-based tests defend the effect against operator-presence and length reductions, though a bounded-outcome sensitivity check weakens the regression form of the length defense. Second, an item-level ceiling: the same formal profiles explain only 3.3 to 3.6 percent of entropy variance a
    
[^222]: SpanUQ：面向大语言模型生成的跨度级不确定性量化

    SpanUQ: Span-Level Uncertainty Quantification for Large Language Model Generation

    [https://arxiv.org/abs/2607.05721](https://arxiv.org/abs/2607.05721)

    该论文提出跨度级不确定性估计（SLUE）新任务，并开发了轻量级模型SPANUQ，通过单次前向传播即可检测语义连贯的文本跨度并量化其不确定性，克服了词元级与序列级方法在粒度上的局限。

    

    不确定性估计不仅对大语言模型（LLM）的可信部署至关重要，也是LLM生成中自我改进的基础。然而，现有方法在次优的粒度上运作：词元级评分缺乏语义连贯性，而序列级评分无法定位错误。我们形式化了跨度级不确定性估计（SLUE）这一新任务，它针对不确定性的自然粒度：语义连贯的文本跨度，每个跨度传达一个可评估的单一意义单元。为了解决这一任务，我们引入了SPANUQ，一个轻量级（2500万参数）的探针，它将昂贵的多次采样推理中的不确定性知识蒸馏到对LLM隐藏状态的单次前向传播中。SPANUQ采用DETR风格的跨度解码器，通过Beta分布混合同时检测跨度并估计其不确定性，并使用Beta NLL回归的原理性组合进行训练。

    arXiv:2607.05721v2 Announce Type: replace  Abstract: Uncertainty estimation is essential not only for the trustworthy deployment of large language models (LLMs) but also as a foundation for self-refinement in LLM generation. However, existing approaches operate at suboptimal granularities: token-level scores lack semantic coherence, while sequence-level scores fail to localize errors. We formalize Span-Level Uncertainty Estimation (SLUE), a new task that targets the natural granularity for uncertainty: semantically coherent text spans, each conveying a single assessable unit of meaning. To address this task, we introduce SPANUQ, a lightweight (25M parameter) probe that distills the uncertainty knowledge from expensive multi-sample inference into a single forward pass over LLM hidden states. SPANUQ employs a DETR-style span decoder to simultaneously detect spans and estimate their uncertainty via a Mixture of Beta distribution, trained with a principled combination of Beta NLL regressio
    
[^223]: 一个主导的自条件化方向驱动无条件连续扩散语言模型中的重复现象

    A Dominant Self-Conditioning Direction Drives Repetition in Unconditional Continuous Diffusion Language Models

    [https://arxiv.org/abs/2607.00588](https://arxiv.org/abs/2607.00588)

    该论文发现连续扩散语言模型的重复生成源于自条件化反馈回路将表征驱向一维收缩吸引子，并据此提出无需训练的推理时干预方法ACE，通过对比去噪路径估计重复方向以逃逸该吸引子。

    

    arXiv:2607.00588v2 公告类型：替换 摘要：连续扩散语言模型为自回归生成提供了一种替代方案，但其生成结果可能出现重复问题。我们发现，近期一类连续扩散语言模型ELF的无条件生成文本比人类文本更具重复性，而Gen-PPL这一常见的基于似然的指标会给重复性生成赋予更低的困惑度，从而可能掩盖这一问题并使质量评估产生偏差。我们的分析将这种行为与一个自条件化反馈回路联系起来：干净嵌入预测被反复带入后续的去噪步骤，使表征被驱向一个与重复相关联的有效一维收缩吸引子。基于这一机制，我们提出了Attractor-Contrast-Escape（ACE），这是一种无需训练的推理时干预方法，它通过对比陷入重复的去噪路径与相对自……（原文摘要在此处截断）

    arXiv:2607.00588v2 Announce Type: replace  Abstract: Continuous diffusion language models offer an alternative to autoregressive generation, but their generations may suffer from repetition. We find that unconditional generations from ELF, a recent family of continuous diffusion language models, are more repetitive than human text, while Gen-PPL, a common likelihood-based metric, gives lower perplexity to repetitive generations and can conceal this problem while biasing quality evaluation. Our analysis links this behavior to a self-conditioning feedback loop in which clean-embedding predictions are repeatedly carried into subsequent denoising steps, driving representations toward an effectively one-dimensional contractive attractor associated with repetition. Based on this mechanism, we introduce Attractor-Contrast-Escape (ACE), a training-free inference-time intervention that estimates a repetition direction by contrasting denoising paths trapped in repetition with paths relatively fr
    
[^224]: 基于置信度的分叉思考

    Fork-Think with Confidence

    [https://arxiv.org/abs/2606.31484](https://arxiv.org/abs/2606.31484)

    提出基于模型置信度先识别分叉点再触发并行思考的方法Fork-think，在保持相当或更优推理性能的同时，将token消耗降低最多30%、运行时间降低最多57%。

    

    并行思维在无需任何重新训练的情况下提升大语言模型推理任务表现方面取得了巨大成功。然而，现有方法遵循“先思考后决策”的范式，即首先采样多条推理路径，这不可避免地导致过度生成，随后再通过剪枝或停止不必要的路径来进行补偿。相比之下，“先决策后思考”的范式——即首先识别可能产生理想生成的节点——至今尚未得到充分探索。遵循这一范式，我们提出了基于置信度的Fork-think方法，该方法首先在单条种子路径中利用模型置信度识别分叉点，然后触发思考，采样多个后续延续并将其聚合以生成最终回复。我们在三个模型和三个推理基准上的实验表明，Fork-think可将token消耗降低多达30%，运行时间降低多达57%，同时性能与现有方法相当甚至更优。

    arXiv:2606.31484v2 Announce Type: replace-cross  Abstract: Parallel thinking has enjoyed great success for boosting LLM performance on reasoning tasks without the need for any re-training. However, existing methods follow a think-first-then-decide paradigm, i.e., they first sample multiple reasoning paths, which inevitably leads to overgeneration, then prune or stop unnecessary paths to compensate. In contrast, decide-first-then-think, i.e., first identifying points that are likely to lead to desirable generations, has been underexplored so far. Following this paradigm, we propose Fork-think with confidence, that first identifies forking points using model confidence in a single seeding path, then triggers thinking, sampling multiple continuations and aggregating them for the final response. Our experiments across three models and three reasoning benchmarks show that Fork-think reduces the token consumption by up to 30% and run-time by up to 57%, while performing comparable to or bette
    
[^225]: 计算机使用代理的不确定性量化：跨视觉语言模型与GUI定位数据集的基准

    Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and GUI Grounding Datasets

    [https://arxiv.org/abs/2606.25760](https://arxiv.org/abs/2606.25760)

    该论文提出Argus——一个跨VLM代理与GUI定位数据集的事后不确定性量化系统基准，通过对27种开放权重方法和8种闭源方法的全面评估，检验UQ方法排名在不同模型、基准与可观测性条件下的稳定性。

    

    计算机使用代理将视觉语言模型（VLM）的预测转化为可执行的图形界面点击，因此可靠的不确定性估计对于拒绝、校准、错误严重性排序以及空间安全区域至关重要。然而，针对这些代理的事后不确定性量化（UQ）的证据分散在孤立的模型与数据集组合中，导致当代理、基准或可观测接口发生变化时，UQ方法的排名是否保持稳定尚不清楚。我们提出了Argus，一个面向单步可执行GUI定位中事后不确定性量化的跨机制基准：包含覆盖4个VLM代理和4个数据集的27种方法的开放权重评估矩阵，以及跨3个前沿供应商的8种方法的闭源评估矩阵（在这些闭源环境中logits、隐藏状态和注意力图均不可获取）。所评估的方法涵盖基于logit的评分、采样与一致性度量、隐藏状态与密度估计器（Mahalanobis、SAPLMA）、基于注意力的评分、P(True)以及口头表达的置信度……

    arXiv:2606.25760v2 Announce Type: replace-cross  Abstract: Computer-use agents turn vision-language model (VLM) predictions into executable GUI clicks, so reliable uncertainty estimates are essential for rejection, calibration, miss-severity ranking, and spatial safety regions. Yet evidence on post-hoc uncertainty quantification (UQ) for these agents is fragmented across isolated model and dataset pairs, leaving it unclear whether UQ rankings stay stable when the agent, benchmark, or observable interface changes. We present Argus, a cross-regime benchmark for post-hoc UQ in single-step executable GUI grounding: a 27-method open-weight matrix over 4 VLM agents and 4 datasets, plus an 8-method closed-source matrix across 3 frontier vendors where logits, hidden states, and attention maps are unavailable. Evaluated methods span logit-based scores, sampling and consistency measures, hidden-state and density estimators (Mahalanobis, SAPLMA), attention-based scores, P(True) and verbalised-con
    
[^226]: OctoNest：基于有状态控制的自适应跨设备执行

    OctoNest: Adaptive Cross-Device Execution through Stateful Control

    [https://arxiv.org/abs/2606.20487](https://arxiv.org/abs/2606.20487)

    OctoNest 提出了一种有状态的跨设备编排框架，通过编排器与设备智能体协作，自适应地在设备内切换模态或跨设备重新分配任务，并引入含 158 个实例的 CAPEBench 基准来评测跨设备任务执行。

    

    计算机使用智能体正从单一设备操作扩展到跨设备系统，以协调异构环境中的任务。执行条件在规划时往往仅是部分已知，需要通过交互逐步揭示。失败可能需要设备内的模态切换，也可能需要跨设备的任务重新分配；若不能区分这两种情况，可能导致重复失败或过早终止。然而，现有系统主要着眼于扩展单设备智能体的能力，未能充分区分设备级与模态特定的执行条件。我们提出 OctoNest，它协调有状态的跨设备编排与迭代式的设备本地模态控制。设备智能体负责细化子任务并选择模态，而编排器则利用执行反馈来修订计划与设备分配。我们还介绍了 CAPEBench，该基准包含来自 23 个跨设备种子任务的 158 个实例，并带有受控的扰动设置。

    arXiv:2606.20487v2 Announce Type: replace  Abstract: Computer use agents are expanding from single-device operation toward cross-device systems that coordinate tasks across heterogeneous environments. Execution conditions are often only partially known at planning time and revealed through interaction. Failures may require intra-device modality switching or inter-device reassignment; failing to distinguish these cases can lead to repeated failures or premature termination. However, existing systems primarily scale up single-device agents without sufficiently distinguishing device-level and modality-specific execution conditions. We propose OctoNest, which coordinates stateful cross-device orchestration and iterative device-local modality control. Device Agents refine subtasks and select modalities, while an Orchestrator uses execution feedback to revise plans and device assignments. We also introduce CAPEBench, comprising 158 instances from 23 cross-device seed tasks with controlled pe
    
[^227]: CombEval：一个用于评估大语言模型组合计数能力的框架

    CombEval: A Framework for Evaluating Combinatorial Counting in Large Language Models

    [https://arxiv.org/abs/2606.19788](https://arxiv.org/abs/2606.19788)

    CombEval是一个动态组合计数基准，通过类型化规范可控生成经求解器验证精确答案的计数问题，评估发现11个大语言模型在有序对象、不可区分元素、相对位置约束和嵌套对象依赖等组合推理上仍然脆弱。

    

    我们提出了CombEval，一个用于评估大语言模型组合计数能力的动态基准。CombEval将每个问题表示为基于实体、组合对象、对象依赖关系和约束的类型化Cofola规范，从而能够以可控的方式生成具有求解器验证的精确答案的自然语言计数问题。与静态数据集不同，CombEval支持对对象类型、实体规模、约束数量和推理深度进行系统化的变化。我们在直接回答和代码增强两种设置下评估了11个大语言模型，发现模型在有序对象、不可区分元素、相对位置约束和嵌套对象依赖方面仍然表现脆弱。错误分析进一步揭示了模型在约束解释和计数原理方面的失败。CombEval为研究大语言模型何时以及为何在组合推理中失败提供了一个诊断测试平台。相关代码和生成的基准测试集均已公开。

    arXiv:2606.19788v2 Announce Type: replace  Abstract: We present CombEval, a dynamic benchmark for evaluating combinatorial counting in large language models. CombEval represents each problem as a typed Cofola specification over entities, combinatorial objects, object dependencies, and constraints, enabling controlled generation of natural-language counting problems with exact solver-verified answers. Unlike static collections, CombEval supports systematic variation of object type, entity scale, constraint count, and reasoning depth. We evaluate 11 LLMs under direct and code-augmented settings and find that models remain brittle on ordered objects, indistinguishable elements, relatively positional constraints, and nested object dependencies. Error analysis further identifies failures in constraint interpretation and counting principles. CombEval provides a diagnostic testbed for studying when and why LLMs fail at combinatorial reasoning. The code and generated benchmark suites are publi
    
[^228]: KVEraser：学习引导KV缓存以实现高效的局部上下文擦除

    KVEraser: Learning to Steer KV Cache for Efficient Localized Context Erasing

    [https://arxiv.org/abs/2606.17034](https://arxiv.org/abs/2606.17034)

    KVEraser是一种学习式KV缓存编辑方法，通过用学习到的引导状态替换被擦除区间的KV状态来复用其余缓存，实现计算成本仅取决于被擦除片段长度（而非后缀长度）的高效局部上下文擦除，且效果几乎媲美完全重计算。

    

    对KV缓存进行事后上下文擦除极具挑战性，因为局部编辑会产生全局性后果：一旦某个文本片段被处理，其影响就会传播到所有后续token的缓存状态中。这一问题在长上下文LLM应用中自然出现，因为过时、错误或有害的上下文往往只有在预填充完成后才能被识别。精确擦除必须重新计算被删除片段之后的所有token，使其计算成本取决于后缀长度而非被擦除片段的长度。我们提出了KVEraser，一种用于高效局部上下文擦除的学习式KV缓存编辑方法。KVEraser用学习到的引导状态替换被擦除区间的KV状态，同时保持其余缓存不变并重复使用。为了学习一种可迁移的擦除机制，我们采用两阶段流水线：先进行通用的片段-邻域预训练，再进行任务特定的微调。实验表明，KVEraser的效果几乎媲美完全重计算。

    arXiv:2606.17034v3 Announce Type: replace  Abstract: Post-hoc context erasing over the KV cache is challenging because a local edit has a global consequence: once a span has been processed, its influence propagates into the cached states of all subsequent tokens. This issue arises naturally in long-context LLM applications, where stale, incorrect, or harmful context may be identified only after prefill. Exact erasing must then recompute all tokens after the deleted span, making its computational cost depend on suffix length rather than erased-span length. We introduce KVEraser, a learned KV-cache editing method for efficient localized context erasing. KVEraser replaces the KV states of the erased interval with learned steering states while reusing the remaining cache unchanged. To learn a transferable erasing mechanism, we use a two-stage pipeline: generic span-neighbor pre-training followed by task-specific fine-tuning. Experiments show that KVEraser nearly matches full recomputation 
    
[^229]: Interactor：面向智能体强化学习的迭代式创作方法用于付费搜索广告描述生成

    Interactor: Agentic RL oriented Iterative Creation for Ad Description Generation in Sponsored Search

    [https://arxiv.org/abs/2606.15911](https://arxiv.org/abs/2606.15911)

    提出Interactor框架，通过智能体强化学习让生成模型与多个生成式奖励模型进行多轮交互迭代，自动生成融入世界知识且与落地页一致的高质量付费搜索广告描述。

    

    本文聚焦于付费搜索中信息丰富广告描述的自动生成。与通常为吸引用户点击反馈而优化的广告标题不同，广告描述具有更长的文本篇幅，并具备融入世界知识的潜力，能够在响应用户搜索意图的同时呈现广告的细粒度卖点。我们提出了Interactor，一个通过智能体强化学习（agentic RL）优化的多轮迭代创作框架，用于广告描述生成。生成模型作为策略，与由多个生成式奖励模型构成的定制化环境进行交互。在策略给出初始生成结果后，定制化的生成式奖励模型（GenRMs）对知识容量、落地页一致性等质量维度进行评估，同时提供二值信号和详细反馈。策略随后基于这些反馈迭代地精炼描述，以确保持续改进。实验表明……（摘要至此截断）

    arXiv:2606.15911v2 Announce Type: replace  Abstract: This paper focuses on automatically generating informative ad descriptions in sponsored search. Unlike ad titles which are usually optimized to attract user click feedbacks, ad descriptions have a longer text span and possess the potential of incorporating world knowledge to address user search intents while presenting the fine-grained selling points of the ads. We propose Interactor, a multi-turn iterative creation framework optimized with agentic RL for ad description generation. The generation model acts as a policy that interacts with a customized environment consisting of multiple generative reward models. Given initial generations by the policy, the customized GenRMs evaluate qualities including knowledge capacity and landing page consistency, providing both binary signals and detailed feedbacks. The policy then iteratively refines the descriptions based on such feedbacks to ensure continuous improvement. Experiments show that 
    
[^230]: 超越承诺边界：探究大型推理模型中的副现象思维链

    Beyond the Commitment Boundary: Probing Epiphenomenal Chain-of-Thought in Large Reasoning Models

    [https://arxiv.org/abs/2606.13603](https://arxiv.org/abs/2606.13603)

    该研究通过答案 logits 与注意力探针发现，大型推理模型在推理早期就跨越“承诺边界”确定了最终答案，其后的思维链步骤属于副现象，对最终答案没有因果影响。

    

    思维链推理是语言模型中推理时扩展的主流范式，但各个推理步骤对最终答案的因果影响仍知之甚少。在这项工作中，我们利用每个推理步骤结束时的答案 logits 来估计各步骤对最终答案和中间猜测的因果重要性，从而揭示多个推理模型家族的答案形成过程。在多样化任务中，我们发现推理通常会跨越一个承诺边界——即从瞬态的中间猜测到稳定、高置信度答案的急剧转变。这种转变往往发生在单个步骤中，且远在模型推理块结束之前，其后随之而来的则是副现象性的 CoT 步骤，它们不会改变最终答案的概率。借助注意力探针，我们进一步表明答案形成阶段可以从中间推理步骤的激活中被线性解码出来。

    arXiv:2606.13603v2 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) reasoning is the dominant paradigm for inference-time scaling in language models, yet the causal influence of individual steps on the final answer remains poorly understood. In this work, we use answer logits at the end of each reasoning step to estimate each step's causal importance to the final answer and intermediate guesses, shedding light on the answer formation process of several reasoning model families. Across diverse tasks, we find that reasoning typically crosses a commitment boundary, a sharp transition from transient intermediate guesses to a stable, high-confidence answer. This transition often happens in a single step, well before the model's reasoning block ends, and is followed by epiphenomenal CoT steps that leave the final answer probability unaltered. Using attention probes, we show that answer-formation stages can be linearly decoded from the activations of intermediate reasoning steps
    
[^231]: GrepSeek：训练用于直接语料库交互的搜索智能体

    GrepSeek: Training Search Agents for Direct Corpus Interaction

    [https://arxiv.org/abs/2605.29307](https://arxiv.org/abs/2605.29307)

    GrepSeek提出了一种让LLM搜索智能体直接通过shell命令与语料库交互寻找证据的新范式，采用“Tutor/Planner生成可靠搜索轨迹初始化+GRPO强化学习优化”的两阶段训练方法，并通过语义保持的执行优化实现大规模语料库上的高效检索。

    

    arXiv:2605.29307v2 公告类型：replace-cross 摘要：大型语言模型（LLM）搜索智能体通过迭代推理与检索，在知识密集型任务上展现出了强大的潜力。现有系统大多依赖于从预构建索引中返回排序文档的检索器。我们探索了一种互补的范式：让智能体将语料库本身视为搜索环境，并通过可执行的shell命令来查找证据。我们提出了GrepSeek，一个经过优化的直接语料库交互（DCI）智能体，它学会在大型文本语料库上查找、过滤和组合证据。为了稳定大型语料库上的强化学习（RL）过程，我们采用两阶段训练：第一阶段，使用由答案感知的Tutor和答案盲的Planner生成的、经过验证且具有因果依据的搜索轨迹来初始化策略；第二阶段，使用组相对策略优化（GRPO）对策略进行改进。为了使直接语料库交互在大规模场景下切实可行，我们引入了两种保持语义的执行优化：（原文摘要在此处截断）

    arXiv:2605.29307v2 Announce Type: replace-cross  Abstract: Large Language Model (LLM) search agents have shown strong promise on knowledge-intensive tasks through iterative reasoning and retrieval. Most existing systems rely on retrievers that return ranked documents from a pre-built index. We explore a complementary paradigm in which the agent treats the corpus as the search environment and finds evidence through executable shell commands. We introduce GrepSeek, an optimized direct corpus interaction (DCI) agent that learns to find, filter, and compose evidence over large text corpora. To stabilize reinforcement learning (RL) over large corpora, we train in two stages: first, we initialize the policy using verified, causally grounded search trajectories generated by an answer-aware Tutor and an answer-blind Planner; then, we refine the policy using Group Relative Policy Optimization (GRPO). To make DCI practical at scale, we introduce two semantics-preserving execution optimizations: 
    
[^232]: RA-MoE：面向混合专家模型多语言适配的路由对齐微调

    RA-MoE: Routing-Aligned Fine-Tuning for Multilingual Adaptation of Mixture-of-Experts Models

    [https://arxiv.org/abs/2605.28306](https://arxiv.org/abs/2605.28306)

    提出RA-MoE三阶段微调框架，利用中间层跨语言路由对齐现象，在ci样本上选择性地将目标语言路由向成功的英语路由模式对齐（同时匹配任务专家的总路由质量及其相对分配），从而有效实现混合专家模型的多语言适配并缩小目标语言性能差距。

    

    混合专家模型能够实现高效的大语言模型扩展，但将其适配到非英语下游任务仍然具有挑战性。标准的多语言微调在很大程度上忽略了这类模型异构的路由结构。我们在多个MoE模型和任务上发现，模型中间层存在强烈的跨语言路由对齐现象，而路由分歧与目标语言的性能差距相关。受此观察启发，我们提出了RA-MoE（路由对齐MoE微调），一个用于多语言MoE适配的三阶段框架。RA-MoE将平行样本划分为四个正确性组，并在中间层识别与任务相关的专家。随后，它在ci样本上选择性地将目标语言路由向成功的英语路由模式对齐，联合匹配分配给任务专家的总路由质量及其在专家之间的相对分配。实验在三个MoE模型、三个下游任务上进行（摘要在此处被截断）。

    arXiv:2605.28306v2 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) models enable efficient LLM scaling, yet adapting them to non-English downstream tasks remains challenging. Standard multilingual fine-tuning largely ignores their heterogeneous routing structure. Across multiple MoE models and tasks, we find strong cross-lingual routing alignment in middle layers, with routing divergence associated with target-language performance gaps. Motivated by this observation, we propose RA-MoE (Routing-Aligned MoE Fine-Tuning), a three-stage framework for multilingual MoE adaptation. RA-MoE categorizes parallel examples into four correctness groups (cc/ci/ic/ii) and identifies task-relevant experts in middle layers. It then selectively aligns target-language routing on ci examples toward successful English routing patterns, jointly matching the total routing mass assigned to task experts and its relative allocation among them. Experiments across three MoE models, three downstre
    
[^233]: 当分布内收益失效时：在偏好偏移下评估弱到强奖励模型

    When In-Distribution Gains Fail: Evaluating Weak-to-Strong Reward Models under Preference Shift

    [https://arxiv.org/abs/2605.25629](https://arxiv.org/abs/2605.25629)

    本文揭示了弱到强奖励模型在偏好分布偏移下的迁移失败问题——弱监督微调会使强模型偏向源域特征——并提出表示锚定正则化方法，通过约束表示漂移来保持可迁移的偏好表示。

    

    弱到强泛化是一种有前景的可扩展监督框架，然而现有评估通常在训练-测试分布匹配的条件下测试学生模型。因此，我们研究了零样本分布偏移下的弱到强偏好学习，发现基于弱偏好标签训练的强学生模型可能在分布内表现良好，却无法跨偏好数据集进行迁移。我们为一种表示层面的失败模式提供了证据：弱监督微调可能将强模型拉向源域特征，而不是保持广泛可迁移的偏好表示。为缓解这一问题，我们提出了表示锚定方法，这是一种简单而有效的正则化手段，它在微调过程中约束模型不过度偏离预训练强模型的表示空间，同时仍允许与任务相关的适应。在多个偏好领域、数据集和模型家族上，Anchor 能够持续带来改进。

    arXiv:2605.25629v3 Announce Type: replace  Abstract: Weak-to-strong (W2S) generalization is a promising framework for scalable oversight, yet existing evaluations often test students under matched train-test distributions. Therefore, we study W2S preference learning under zero-shot distribution shift and find that strong students trained on weak preference labels can appear successful in-distribution while failing to transfer across preference datasets. We provide evidence for a representational failure mode in which weak-supervised fine-tuning can pull the strong model toward source-domain features instead of maintaining broadly transferable preference representations. To mitigate this, we propose Representation Anchoring (Anchor), a simple yet effective regularizer that constrains excessive drift from the pretrained strong model's representation space during fine-tuning, while still allowing task-relevant adaptation. Across preference domains, datasets, and model families, Anchor con
    
[^234]: 手语之间的直接翻译

    Direct Translation between Sign Languages

    [https://arxiv.org/abs/2605.20588](https://arxiv.org/abs/2605.20588)

    该论文提出通过改进的反向翻译方法从现有文本-手语语料库构建跨语言手语配对数据，进而联合训练单一模型实现手语之间的直接翻译，避免了级联系统的中间错误传播问题。

    

    手语翻译在手语与有声语言之间已经取得了长足的进步，而手语之间的翻译仍然较少被探索。直接在手语之间进行翻译可以支持不同手语社群之间的交流，而无需依赖共同的书面语言。由手语到文本、有声语言翻译和文本到手语模型组成的级联系统提供了一条可行路径，但它可能传播中间错误，且需要三个顺序执行的翻译阶段。我们开发了直接的手语到手语翻译，但其训练受到跨语言平行手语数据稀缺的限制。为解决这一障碍，我们对反向翻译方法进行了改进，从现有的文本-手语语料库中构建跨语言手语配对：源手语通过文本桥梁合成，而目标手语则保留原始语料库中的真实手语。利用这些配对，我们联合训练了一个基于Qwen3的单一模型，用于文本到手语和手语到手语翻译。

    arXiv:2605.20588v2 Announce Type: replace  Abstract: Sign language translation has made substantial progress between sign and spoken languages, while translation across sign languages remains less explored. Translating directly between sign languages could support communication across signing communities without requiring a shared written language. A cascade of sign-to-text, spoken-language translation, and text-to-sign models offers one route, but can propagate intermediate errors and requires three sequential translation stages. We develop direct sign-to-sign translation, whose training is limited by the scarcity of parallel signing across languages. To address this obstacle, we adapt back-translation to construct cross-lingual pairs from existing text-sign corpora: the source signing is synthesized through a text bridge, while the target remains the gold sign from the original corpus. Using these pairs, we jointly train a single Qwen3-based model for text-to-sign and sign-to-sign tr
    
[^235]: CHI-Bench：AI智能体能否自动化端到端、长周期、政策密集的医疗保健工作流程？

    CHI-Bench: Can AI Agents Automate End-to-End, Long-Horizon, Policy-Rich Healthcare Workflows?

    [https://arxiv.org/abs/2605.16679](https://arxiv.org/abs/2605.16679)

    CHI-Bench是一个评估AI智能体能否端到端自动化政策密集、需多角色协作与多轮交互的长周期医疗保健工作流程的新基准，涵盖事先授权、利用管理和护理管理三大领域。

    

    端到端自动化现实医疗保健运营场景，对当前基准测试中代表性不足的三种能力提出了考验：政策密度——决策必须基于庞大的医疗、保险和运营规则库；多角色组合——单个任务要求智能体扮演多个角色并进行交接；多边交互——中间工作流步骤是多轮对话，例如同行评审和患者外联。我们提出了χ-Bench，这是一个涵盖三个领域的长周期医疗保健工作流基准测试：医疗服务提供方事先授权、支付方利用管理和护理管理。每个任务向智能体提供一个临床案例，该案例处于一个通过87个MCP工具暴露、包含20个医疗应用程序的高保真模拟器中，智能体必须通过工具调用和撰写相应角色的文档将案例推进至终止状态，并以一份包含1,290多份文档的管理式医疗运营手册技能为指导。在30个智能体框架上的（评估）……（摘要在此处被截断）

    arXiv:2605.16679v3 Announce Type: replace-cross  Abstract: End-to-end automation of realistic healthcare operations stresses three capabilities underrepresented in current benchmarks: policy density, decisions must be grounded in a large library of medical, insurance, and operational rules; Multi-role composition: a single task requires the agent to play multiple roles with handoffs; and multilateral interaction: intermediate workflow steps are multi-turn dialogs, such as peer-to-peer review and patient outreach. We introduce $\chi$-Bench, a benchmark of long-horizon healthcare workflows across three domains: provider prior authorization, payer utilization management, and care management. Each task hands the agent a clinical case in a high-fidelity simulator of 20 healthcare apps exposed via 87 MCP tools, which it must drive to a terminal status through tool calls and writing the role's artifacts, guided by a 1,290+ document managed-care operations handbook skill. Across 30 agent harne
    
[^236]: WASIL：与大型语言模型的真实阿拉伯语口语交互数据集

    WASIL: In-the-Wild Arabic Spoken Interactions with LLMs

    [https://arxiv.org/abs/2605.16364](https://arxiv.org/abs/2605.16364)

    该论文发布了WASIL——一个包含音频、ASR识别假设、助手回复和用户好/差评反馈的真实场景阿拉伯语口语交互数据集，并通过可回答性标注将ASR识别错误与用户请求本身固有的不可回答性区分开来。

    

    大型语言模型（LLM）语音助手通常采用级联式架构构建，即先通过自动语音识别（ASR）系统再将识别结果输入LLM，这种架构中识别错误可能扭曲用户意图。此外，用户的差评也可能源于模糊、超出领域或非请求类的对话轮次，这使得难以将ASR错误的影响分离出来。我们发布了WASIL（在阿拉伯语中意为“连接”）：一个真实场景下的阿拉伯语口语交互数据集，包含音频、ASR识别假设、助手回复以及明确的好评/差评反馈（共8,529轮，其中差评占14.2%），并附带一个2,000轮的测试集，涵盖现代标准阿拉伯语（MSA）和四种主要方言及其标签。我们通过多ASR一致性引导的后编辑方式提供了低成本的黄金标准转录文本，并对可回答性进行标注（可回答、模糊/需澄清、缺乏依据、非请求/噪声），以区分固有的不可回答性与由ASR引起的质量下降。最后，我们描述了一种可扩展的无参考评估方法，用于评估（摘要在此处被截断）

    arXiv:2605.16364v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) voice assistants are commonly built as cascaded Automatic Speech recognition (ASR) to LLM systems, where recognition errors can distort user intent. Dislikes may also arise from ambiguous, out-of-domain, or non-request turns, making it hard to isolate ASR effects. We release WASIL (it denotes connection or linking in Arabic): in-the-wild Arabic spoken interaction prompts with audio, ASR hypotheses, assistant responses, and explicit like/dislike feedback (8,529 turns; 14.2% dislikes), plus a 2,000-turn test set covering Modern Standard Arabic (MSA) and four major dialects with their labels. We provide low-cost gold transcripts via multi-ASR agreement-guided post-editing and annotate answerability (answerable, ambiguous/needs-clarification, unsupported, not-a-request/noise) to separate intrinsic unanswerability from ASR-induced degradation. Finally, we describe scalable reference-free evaluation of re
    
[^237]: 通过零失配参考诊断LLM强化学习中的训练-推理失配

    Diagnosing Training Inference Mismatch in LLM Reinforcement Learning via a Zero-Mismatch Reference

    [https://arxiv.org/abs/2605.14220](https://arxiv.org/abs/2605.14220)

    该论文提出零失配诊断环境VeXact来隔离研究LLM强化学习中的训练-推理失配（TIM），证明微小的token级数值差异可独立引发训练崩溃，并指出TIM应被视为影响LLM强化学习稳定性的一阶系统级扰动因素而非良性数值噪声。

    

    现代LLM强化学习系统将rollout生成与策略优化两个阶段分开，这两个阶段预期产生完全匹配的token概率。然而，实现上的差异可能导致它们在相同模型权重下对同一序列赋予不同的值，从而引发训练-推理失配（TIM）。TIM之所以难以检查，是因为它与off-policy漂移以及常见的稳定化机制相互纠缠。在这项工作中，我们在一个零失配诊断环境（VeXact）中隔离TIM，并表明微小的token级数值差异可以独立导致训练崩溃。我们进一步表明TIM会改变有效的优化问题，并确定了一组可以缓解TIM的补救措施。我们的结果表明，TIM并非良性的数值噪声，而是一种系统级扰动，在分析LLM强化学习稳定性时应将其视为一阶因素。

    arXiv:2605.14220v2 Announce Type: replace-cross  Abstract: Modern LLM RL systems separate rollout generation from policy optimization. These two stages are expected to produce token probabilities that match exactly. However, implementation differences can make them assign different values to the same sequence under the same model weights, inducing Training-Inference Mismatch (TIM). TIM is difficult to inspect because it is entangled with off-policy drift and common stabilization mechanisms. In this work, we isolate TIM in a zero-mismatch diagnostic setting (VeXact), and show that small token-level numerical disagreements can independently cause training collapse. We further show that TIM changes the effective optimization problem, and identify a set of remedies that could mitigate TIM. Our results suggest that TIM is not benign numerical noise, but a systems-level perturbation that should be treated as a first-order factor in analyzing LLM RL stability.
    
[^238]: 将图灵测试推广至交互式智能体

    Generalizing the Turing Test to Interactive Agents

    [https://arxiv.org/abs/2605.10851](https://arxiv.org/abs/2605.10851)

    该论文提出广义图灵测试（GTT），将图灵测试从人类推广到任意交互式智能体，证明了“图灵比较器”传递性等理论性质，并通过九个大语言模型的实验验证了图灵分数能够清晰地对模型进行分层。

    

    我们开创了对广义图灵测试的研究，这是图灵模仿游戏从人类到任意交互式智能体的形式化推广。对于智能体 A 和 B，如果 B 的一个实例作为区分者，无法可靠地区分一个被指示模仿 B 的 A 与另一个 B 的实例，那么 A 就通过了对 B 的 GTT；此时我们记作 A ≥ B。我们研究了这一思想在理论和实证上的后果。在理论方面，我们证明了这一“图灵比较器”具有传递性的充分条件。我们引入了若干自然变体：带查询的变体（模仿者可以先与目标的一个样本进行交互）、允许任意区分者和目标的通用图灵测试，以及控制交互长度的复杂性理论变体。作为概念验证，我们在九个大语言模型上评估了 GTT 及其变体。值得注意的是，图灵分数呈现出清晰的模型分层。

    arXiv:2605.10851v2 Announce Type: replace  Abstract: We initiate the study of the Generalized Turing Test (GTT), a formal generalization of Turing's imitation game from humans to arbitrary interactive agents. For agents $A$ and $B$, $A$ passes the GTT against $B$ if an instance of $B$, acting as a distinguisher, cannot reliably distinguish an $A$ instructed to imitate $B$ from another instance of $B$; if so, we write $A \geq B$. We study the theoretical and empirical consequences of this idea. On the theory side, we prove sufficient conditions under which this "Turing Comparator" is transitive. We introduce natural variants with querying (the imitator can first interact with a specimen of the target), a Universal Turing Test with arbitrary distinguishers and targets, and complexity-theoretic variants that control interaction length. As a proof of concept, we evaluate the GTT and its variants across nine large language models. Remarkably, Turing Scores recover a clear model stratificati
    
[^239]: 基于大语言模型的语音心理危机评估

    Speech-based Psychological Crisis Assessment using LLMs

    [https://arxiv.org/abs/2605.10027](https://arxiv.org/abs/2605.10027)

    该论文提出一种基于大语言模型的自动心理危机等级分类框架，通过将非语言情感线索注入语音转录文本的副语言注入方法，以及以诊断推理链生成为辅助任务的推理增强训练策略，显著提升了心理援助热线危机评估的准确性与服务质量。

    

    心理援助热线为经历心理健康紧急状况的个体提供关键支持，然而目前的评估主要依赖人工接线员，其判断可能因专业经验差异而有所不同，并受限于有限的人力资源。本文提出了一种基于大语言模型（LLM）的自动化危机等级分类框架，危机等级是支持众多下游任务的关键指标，并有助于提升热线服务的整体质量。为了更好地捕捉口语对话中的情感信号，我们引入了一种副语言注入方法，将识别出的非语言情感线索插入语音转录文本中，使基于LLM的推理能够融入关键的声学细微差异。此外，我们提出了一种推理增强训练策略，将模型生成诊断推理链作为辅助任务进行训练，该策略作为正则化手段来改进分类性能。

    arXiv:2605.10027v2 Announce Type: replace-cross  Abstract: Psychological support hotlines provide critical support for individuals experiencing mental health emergencies, yet current assessments largely rely on human operators whose judgments may vary with professional experience and are constrained by limited staffing resources. This paper proposes a large language model (LLM)-based framework for automated crisis level classification, a key indicator that supports many downstream tasks and improves the overall quality of hotline services. To better capture emotional signals in spoken conversations, we introduce a paralinguistic injection method that inserts identified non-verbal emotional cues into speech transcripts, enabling LLM-based reasoning to incorporate critical acoustic nuances. In addition, we propose a reasoning-enhanced training strategy that trains the model to generate diagnostic reasoning chains as an auxiliary task, which serves as a regulariser to improve classificati
    
[^240]: 超越LoRA与全量微调之争：面向大语言模型适配的梯度引导优化器路由

    Beyond LoRA vs. Full Fine-Tuning: Gradient-Guided Optimizer Routing for LLM Adaptation

    [https://arxiv.org/abs/2605.07111](https://arxiv.org/abs/2605.07111)

    针对LoRA与全量微调谁更优取决于任务和模型这一难题，本文提出MoLF统一框架，利用梯度引导的优化器路由在两种微调方式间动态选择。

    

    近期关于大语言模型微调的文献凸显了一场根本性的争论。虽然全量微调（FFT）提供了更强的表征可塑性，但低秩适配（LoRA）在将更新限制于低秩空间并可能从额外正则化中受益的同时，能够匹敌甚至超越FFT的性能。通过在多样化任务（SQL、医学问答和反事实知识）和不同语言模型（Gemma-3-1B、Qwen2.5-1.5B和Qwen2.5-3B）上的实证评估，我们同时观察到这两种趋势，并发现哪种静态架构更优取决于具体的任务和模型。谱分析与截断分析进一步表明，仅凭端点可压缩性无法解释这些任务差异，任务分数敏感性和受限的优化轨迹可能是更合理的解释。为应对这一挑战，我们提出了LoRA与全量微调的混合方法，这是一个统一框架，能够……（原文摘要至此截断）

    arXiv:2605.07111v3 Announce Type: replace-cross  Abstract: Recent literature on fine-tuning Large Language Models highlights a fundamental debate. While Full Fine-Tuning (FFT) provides greater representational plasticity, Low-Rank Adaptation (LoRA) can match or surpass FFT performance while constraining updates to a low-rank space and potentially benefiting from additional regularization. Through empirical evaluation across diverse tasks (SQL, Medical QA, and Counterfactual Knowledge) and varying language models (Gemma-3-1B, Qwen2.5-1.5B, and Qwen2.5-3B), we observe both trends and find that the better static architecture depends on the task and model. Spectral and truncation analyses further show that endpoint compressibility alone does not explain these task differences, suggesting task-score sensitivity and constrained optimization trajectories as possible explanations. To address this challenge, we propose a Mixture of LoRA and Full (MoLF) Fine-Tuning, a unified framework that enab
    
[^241]: 前沿滞后：学术AI评估中能力误述的文献计量审计

    Frontier Lag: A Bibliometric Audit of Capability Misrepresentation in Academic AI Evaluation

    [https://arxiv.org/abs/2605.04135](https://arxiv.org/abs/2605.04135)

    该研究对超过11万篇文献进行系统计量分析后发现，学术论文中评估的LLM能力落后于当时的前沿模型（中位数差距+10.85 ECI），且这一“前沿滞后”差距正以每年+5.53 ECI的速度持续扩大。

    

    应用领域中的LLM评估往往反映的是在论文发表时就已经被超越的模型。我们观察到一种“发表引导差距”（publication elicitation gap）：即学术论文中所报告结果由哪些AI系统生成，与当前读者合理认为论文所引用的AI系统之间的距离。我们系统性地扫描了OpenAlex中2022年1月1日至2026年4月1日的数据（n = 112,303个LLM关键词匹配），随后识别出被评估的模型（n = 18,574条有效记录）。接着，我们基于Epoch AI能力指数（ECI）——一个LLM综合能力评分——将每个被评估的LLM与前沿LLM进行排名比较。在评估时点，中位数论文所评估的模型在能力上落后于前沿LLM，中位数差距为+10.85 ECI（H1；n = 12,312）。这一差距正在扩大，以每年+5.53 ECI的速度增长（H2，名义95%置信区间 [+5.03, +5.83]）。即使在……（原文摘要在此处截断）

    arXiv:2605.04135v3 Announce Type: replace-cross  Abstract: LLM evaluations in applied domains tend to reflect models that were already outclassed at time of publication. We observe a publication elicitation gap: the distance between the AI systems generating the results reported in an academic paper and the AI systems that a current reader of that paper would reasonably assume are being referenced. We systematically sweep OpenAlex from 2022-01-01 to 2026-04-01 (n = 112,303 LLM keyword matches). Then, we identify what models were evaluated (n = 18,574 admissible records). We then rank each evaluated LLM against a frontier LLM based on the Epoch AI Capabilities Index (ECI), an aggregate LLM capability score. At time of evaluation, the median paper is evaluating models that are behind frontier LLMs in capability, with a median gap of +10.85 ECI (H1; n = 12,312). This gap is growing, increasing at a rate of +5.53 ECI per year (H2, nominal 95% CI [+5.03, +5.83]). The sign holds even in the 
    
[^242]: LSR-Ben：一个用于评估过程奖励模型的逻辑与科学推理基准

    LSR-Ben: A Logical and Scientific Reasoning Benchmark for Evaluating Process Reward Models

    [https://arxiv.org/abs/2605.01203](https://arxiv.org/abs/2605.01203)

    本文提出LSR-Ben基准，涵盖科学推理和逻辑推理两大领域及九个子领域，用于全面评估过程奖励模型在数学推理之外多样化场景中的过程级错误检测能力。

    

    目前，过程奖励模型（PRM）在测试时扩展方面展现出显著潜力。由于大语言模型（LLM）在处理广泛的推理和决策任务时经常生成有缺陷的中间推理步骤，因此要求PRM具备在真实场景中检测过程级错误的能力。然而，现有基准主要聚焦于数学推理，因而无法全面评估PRM在多样化推理场景中的错误检测能力。为弥补这一空白，我们提出了LSR-Ben，这是一个专门设计的过程级基准，用于评估PRM在两大主要推理领域（科学推理和逻辑推理）及九个子领域上的性能。我们在包含PRM和LLM在内的22个多样化模型上进行了广泛实验，并得出两个关键发现：（1）在数学推理之外的领域中，错误……

    arXiv:2605.01203v3 Announce Type: replace  Abstract: Currently, process reward models (PRMs) have exhibited remarkable potential for test-time scaling. Since large language models (LLMs) regularly generate flawed intermediate reasoning steps when tackling a broad spectrum of reasoning and decision-making tasks, PRMs are required to possess capabilities for detecting process-level errors in real-world scenarios. However, existing benchmarks primarily focus on mathematical reasoning, thereby failing to comprehensively evaluate the error detection ability of PRMs across diverse reasoning scenarios. To mitigate this gap, we introduce LSR-Ben, a process-level benchmark specifically designed for assessing PRM's performance across two primary reasoning domains (scientific and logical reasoning) and nine subdomains. We conduct extensive experiments on a diverse set of 22 models, encompassing both PRMs and LLMs, and derive two key findings: (1) In domains beyond mathematical reasoning, the erro
    
[^243]: 预测正确，步骤有误？用于鲁棒思维链合成的共识推理知识图谱

    Correct Prediction, Wrong Steps? Consensus Reasoning Knowledge Graph for Robust Chain-of-Thought Synthesis

    [https://arxiv.org/abs/2604.14121](https://arxiv.org/abs/2604.14121)

    提出CRAFT方法，通过聚合多个候选推理轨迹的共识组件构建推理知识图谱，从推理结构层面修复LLM“答案正确但推理步骤有缺陷”的问题，实现更鲁棒的思维链合成。

    

    大语言模型（LLM）在各类任务中的应用日益广泛，通常还会结合思维链（CoT）提示来提升准确性。近期研究表明，高标签预测准确率并不能保证中间推理过程的正确性，且推理缺陷的成因因样本而异，然而现有的补救措施要么只针对单一领域，要么假设某一种缺陷类型统一适用于所有样本。一种简单的缓解方法是直接给模型提供正确答案，但我们发现这并不能带来推理质量的持续改善。这表明该问题无法通过LLM对答案的感知来解决，而必须从推理的结构层面加以解决。受此启发，我们提出了CRAFT（面向缺陷感知轨迹合成的共识推理知识图谱聚合方法），该方法聚合多个候选推理轨迹间共享的共识组件……

    arXiv:2604.14121v3 Announce Type: replace  Abstract: Large language models (LLMs) have become increasingly used for various tasks, often coupled with Chain-of-Thought (CoT) prompting to boost accuracy. Recent work has shown that high label-prediction accuracy does not guarantee correct intermediate reasoning, and the causes of *reasoning flaws* vary from sample to sample, yet existing remedies either focus on a single domain or assume that one flaw type applies uniformly across samples. A simple mitigation method is to provide the model with the correct answer, but we show that this yields no consistent improvement in reasoning quality. This indicates that the problem cannot be fixed by LLMs' awareness of answers, and must instead be addressed through the *structure* of reasoning. Motivated by this, we propose CRAFT (Consensus Reasoning-knowledge-graph Aggregation for Flaw-aware Trace synthesis), which aggregates the consensus components shared across multiple candidate reasoning trace
    
[^244]: VisionFoundry：用合成图像教会视觉语言模型视觉感知

    VisionFoundry: Teaching VLMs Visual Perception with Synthetic Images

    [https://arxiv.org/abs/2604.09531](https://arxiv.org/abs/2604.09531)

    提出仅需任务名称即可自动生成合成图像与问答数据的流水线VisionFoundry，其构建的VisionFoundry-10k数据集无需人工标注即可显著提升多个开源视觉语言模型的视觉感知能力。

    

    视觉语言模型（VLM）在空间理解和视角识别等视觉感知任务上仍然表现不佳，主要原因在于自然图像数据集对低层视觉技能提供的监督信号有限。能否在不依赖参考图像或人工标注的情况下，通过有针对性的合成监督来解决这些弱点？为探究这一问题，我们提出了VisionFoundry，一个自动化流水线：它仅需任务名称作为输入，利用大语言模型（LLM）合成配对的问题、答案和文生图（T2I）提示词，用T2I模型生成图像，并通过多模态验证对样本进行过滤。借助VisionFoundry，我们构建了VisionFoundry-10k，一个涵盖10种感知任务的合成视觉问答数据集。在VisionFoundry-10k上微调可在三个开源骨干模型上持续提升感知基准表现（例如，Qwen2.5-VL-3B-Instruct在MMVP-pair上提升6.7%，在CV-Bench-3D上提升10.5%），同时保持模型更广泛的能力。

    arXiv:2604.09531v2 Announce Type: replace-cross  Abstract: Vision-language models (VLMs) still struggle with visual perception tasks such as spatial understanding and viewpoint recognition, largely because natural image datasets provide limited supervision for low-level visual skills. Can targeted synthetic supervision address these weaknesses without reference images or manual annotation? To investigate this, we introduce VisionFoundry, an automated pipeline that takes only a task name as input, uses LLMs to synthesize paired questions, answers, and text-to-image (T2I) prompts, generates images with T2I models, and filters samples via multimodal verification. With VisionFoundry, we construct VisionFoundry-10k, a synthetic VQA dataset spanning 10 perception tasks. Finetuning on VisionFoundry-10k consistently improves perception benchmarks across three open-source backbones (e.g., +6.7% on MMVP-pair and +10.5% on CV-Bench-3D for Qwen2.5-VL-3B-Instruct) while preserving broader capabilit
    
[^245]: 基于参考答案的支撑性标签之原子式与整体式LLM裁判：一种提示词控制下的比较研究

    Atomic and Holistic LLM Judges for Reference-Grounded Support Labels: A Prompt-Controlled Comparison

    [https://arxiv.org/abs/2603.28005](https://arxiv.org/abs/2603.28005)

    该研究在控制裁判模型、输入与指令细节的统一条件下系统比较了四种单次调用的LLM裁判设计，发现当支撑标签依赖答案完整性时，整体式评分标准对所有裁判模型都比候选侧原子分解更准确且使用更少token，表明将答案拆分为原子声明并非总是有益。

    

    当LLM裁判只需在给定参考答案的情况下为候选答案分配一个三分类的支撑标签时，要求它将答案分解为原子性声明是否有帮助，代价又是什么？我们比较了四种共享裁判模型、输入以及指令详细程度的单次调用设计：候选侧原子分解、与之匹配的整体式评分标准、检查每条参考声明是否被覆盖的参考侧分解，以及双向组合。支撑标签基于TruthfulQA、ASQA和QAMPARI的参考答案构建（每个数据集200个问题和400行数据）。所有四种设计均使用Opus-4.6、GPT-4.1和Gemini Flash Lite运行；候选侧和整体式设计还额外加入了Sonnet-4.6。在标签取决于答案完整性的情形下，候选侧分解表现较弱：对于所有裁判模型，整体式评分标准在ASQA和QAMPARI上都更准确，同时使用的token更少。参考侧分解则为12.5（原文摘要在此处截断）。

    arXiv:2603.28005v2 Announce Type: replace  Abstract: When an LLM judge only has to assign a three-way support label to a candidate answer given a reference, does asking it to decompose the answer into atomic claims help, and at what cost? We compare four single-call designs that share the judge model, the inputs, and the level of instruction detail: candidate-side atomic decomposition, a matched holistic rubric, reference-side decomposition that checks whether each reference claim is covered, and a bidirectional combination. Support labels are constructed from TruthfulQA, ASQA, and QAMPARI references (200 questions and 400 rows per dataset). All four designs are run with Opus-4.6, GPT-4.1, and Gemini Flash Lite; Sonnet-4.6 is added for the candidate-side and holistic designs. Candidate-side decomposition is weak where the label depends on completeness: the holistic rubric is more accurate on ASQA and QAMPARI for every judge while using fewer tokens. Reference-side decomposition is 12.5
    
[^246]: DataFlex：一个面向大语言模型以数据为中心的动态训练的统一框架

    DataFlex: A Unified Framework for Data-Centric Dynamic Training of Large Language Models

    [https://arxiv.org/abs/2603.26164](https://arxiv.org/abs/2603.26164)

    本文提出 DataFlex——一个基于 LLaMA-Factory 的统一以数据为中心的动态训练框架，支持样本选择、领域混合调整和样本重加权三大数据优化范式，并可作为标准大语言模型训练的即插即用替代方案，解决了现有方法代码库孤立、接口不一致导致的可复现性差与难以公平比较的问题。

    

    以数据为中心的训练已成为提升大语言模型（LLM）性能的一个有前景的方向，它不仅在优化过程中调整模型参数，还对训练数据的选择、组合与加权进行优化。然而，现有的数据选择、数据混合优化和数据重加权方法往往在各自孤立的代码库中开发，接口互不统一，阻碍了研究的可复现性、公平比较以及实际集成。在本文中，我们提出了 DataFlex，一个基于 LLaMA-Factory 构建的统一的以数据为中心的动态训练框架。DataFlex 支持动态数据优化的三大主要范式：样本选择、领域混合调整和样本重加权，同时与原始训练工作流完全兼容。它提供了可扩展的训练器抽象和模块化组件，能够作为标准大语言模型训练流程的即插即用替代方案，并统一了关键的模型训练接口（原文摘要在此处截断）。

    arXiv:2603.26164v2 Announce Type: replace-cross  Abstract: Data-centric training has emerged as a promising direction for improving large language models (LLMs) by optimizing not only model parameters but also the selection, composition, and weighting of training data during optimization. However, existing approaches to data selection, data mixture optimization, and data reweighting are often developed in isolated codebases with inconsistent interfaces, hindering reproducibility, fair comparison, and practical integration. In this paper, we present DataFlex, a unified data-centric dynamic training framework built upon LLaMA-Factory. DataFlex supports three major paradigms of dynamic data optimization: sample selection, domain mixture adjustment, and sample reweighting, while remaining fully compatible with the original training workflow. It provides extensible trainer abstractions and modular components, enabling a drop-in replacement for standard LLM training, and unifies key model-de
    
[^247]: 俄语社交媒体中人类基本价值观的结构

    Structure of Basic Human Values in Russian Social Media

    [https://arxiv.org/abs/2603.18822](https://arxiv.org/abs/2603.18822)

    该研究利用大语言模型多阶段标注框架对750万条VKontakte帖子中的价值观表达进行测量，首次基于社交媒体的自发表达刻画了俄语人群的基本价值观结构，并与传统问卷结果对照，解决了价值观解释的主观性问题。

    

    arXiv:2603.18822v2 公告类型：替换 摘要：关于国家人口价值观的已有认识几乎完全依赖问卷，问卷要求受访者对研究者提供的各种动机描述进行评分。而社交媒体记录的则是价值观在面向真实受众的交流中被自发表达的情形。我们在广泛抽样、异质性的俄语社交媒体数据中测量价值观表达——这些数据中与价值观相关的内容稀疏且常常含义模糊——并将测量结果与基于问卷的同一人群价值观画像进行对照。为解决价值观解释中的主观性问题，我们将多种替代的大语言模型（LLM）提示配置与专家判断进行比较，并采用基于误差的校准方法来选定基于理论的操作化方案。利用从一百万个随机生成的用户ID收集的VKontakte数据（共750万条公开文本帖子），我们开发了一个结合候选筛选、重复LLM标注等多环节的多阶段框架……

    arXiv:2603.18822v2 Announce Type: replace  Abstract: What is known about the values of national populations rests almost entirely on questionnaires, which prompt respondents to rate researcher-supplied descriptions of each motivation. Social media instead records values as they are invoked spontaneously, in communication addressed to a real audience. We measure value expression in broadly sampled, heterogeneous Russian-language social-media data, where value-relevant content is sparse and often ambiguous, and read the result against the survey-based profile of the same population. To address subjectivity in value interpretation, we compare alternative LLM-prompt configurations against expert judgments and use error-driven calibration to select a theory-based operationalization. Using VKontakte data collected from one million randomly generated user IDs, comprising 7.5 million public text posts, we develop a multi-stage framework combining candidate selection, repeated LLM annotation, a
    
[^248]: 脚手架之下的安全性：评估条件如何塑造所测得的安全表现

    Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety

    [https://arxiv.org/abs/2603.10044](https://arxiv.org/abs/2603.10044)

    评测条件对测得的模型安全性影响超过脚手架本身——在相同的基准题目上，选择题与开放式格式会使测得的安全性相差5-20个百分点，说明评测结果更多取决于测量方法而非模型潜在的 safety 能力。

    

    安全基准测试通常针对“裸”模型——即接收提示并输出响应的模型——进行，但现实世界的部署会将这些模型“包裹”在复杂的脚手架中。这些脚手架对基准测试所衡量的模型安全性究竟有多大影响？我们在四个预先注册的安全基准上，使用直接 API 以及三种脚手架（ReAct、多智能体和 map-reduce）测试了六个领先模型，共进行了 62,808 次评分评估。研究发现，安全性的测量方式比脚手架本身更为重要：对于其他方面完全相同的基准题目，使用选择题还是开放式问题格式，会使测得的安全性相差 5-20 个百分点。由于这两种格式采用不同的评分方法（答案提取与 LLM 评审），这一差距源于测量方式而非模型潜在安全性的差异。若使用启发式方法对模型拒绝行为进行分类，将在五种情况下得出不同的结论。基准的选择可解释结果变异的 19.3%。

    arXiv:2603.10044v3 Announce Type: replace  Abstract: Safety benchmarks usually test "bare" models that receive prompts and output responses, but real-world deployments "wrap" those models in complex scaffolds. How much do these scaffolds affect model safety as measured by benchmarks? We test six leading models on four pre-registered safety benchmarks with a direct API and three scaffolds: ReAct, multi-agent, and map-reduce. We conducted 62,808 scored evaluations. How safety is measured matters more than scaffolding does: we find that using a multiple choice vs. open-ended format for otherwise-identical benchmark items changes measured safety by 5-20 percentage points (pp). The two formats are scored with different methods (answer extraction and an LLM judge), so the gap is due to measurement rather than differences in latent safety. Using a heuristic to classify model refusals would have led to different findings in five cases. Benchmark choice explains 19.3% of the variation in outcom
    
[^249]: Life-Bench：超越概念识别的多模态个性化基准测试与知识图谱框架

    Life-Bench: A Benchmark and Knowledge Graph Framework for Multimodal Personalization Beyond Concept Recognition

    [https://arxiv.org/abs/2602.19001](https://arxiv.org/abs/2602.19001)

    本文提出了Life-Bench——一个包含11,800多个问答对、按概念识别、事件理解和聚合推理三个层次组织的合成多模态个性化基准测试，并配套提出个人知识图谱框架LifeGraph，通过结构化检索与按需访问源视觉证据，在事件理解和聚合推理任务上表现出显著优势。

    

    随着大语言模型日益成为个人助手的核心驱动力，用户期望它们能够对多模态生活履历进行推理——从识别人物到理解事件再到聚合规律，然而现有基准测试主要针对概念级识别。我们推出了Life-Bench，这是一个完全合成、经人工验证的多模态基准测试，包含超过11,800个问答对，涵盖10个任务，并按所需证据范围分为三个层次：概念识别、事件理解和聚合推理。该基准中以照片为中心的个人生活履历在嵌入统计特征上与真实用户账户的分布保持一致。个人数据的互联结构天然适合基于图的解决方案；为此我们提出了LifeGraph，一个个人知识图谱框架，提供结构化检索并支持按需访问源视觉证据，在事件理解和聚合推理任务上展现出特别的优势。对四种检索方法的系统性评估……

    arXiv:2602.19001v2 Announce Type: cross  Abstract: As large language models increasingly power personal assistants, users expect them to reason over multimodal life histories, from recognizing people to understanding events to aggregating patterns, yet existing benchmarks primarily target concept-level recognition. We introduce Life-Bench, a fully synthetic, human-verified multimodal benchmark of over 11,800 question-answer pairs across 10 tasks, organized by required evidence scope: concept identification, event understanding, and aggregated reasoning. The benchmark's photo-centric personal histories are distributionally aligned with real user accounts under embedding statistic. The interconnected structure of personal data invites graph-based solutions; we propose LifeGraph, a personal knowledge graph framework providing structured retrieval with on-demand access to source visual evidence, showing particular promise on event and aggregated tasks. Systematic evaluation of four retriev
    
[^250]: 语义分块与自然语言的熵

    Semantic Chunking and the Entropy of Natural Language

    [https://arxiv.org/abs/2602.13194](https://arxiv.org/abs/2602.13194)

    该论文提出了一个将语言冗余性与文本分层语义组织相关联的统计框架，利用大语言模型将文本递归分割成语义连贯的块以构建“语义树”，从而从语义结构层面解释了自然语言中约80%冗余度（即每个字母仅携带约1比特信息）的来源。

    

    人类和大语言模型都能根据先前的上下文预测下一个字母或单词，其预测能力远好于随机猜测，这表明将语言视为一个随机过程时具有强冗余性。定量而言，香农估计这种冗余度约为80%，这意味着印刷英文文本中的每个字母传递的信息量约为1比特，而非27个字母（包括空格）理论上可能携带的4.8比特。这一估计后来通过使用大语言模型计算的自回归token概率得到了证实。然而，导致如此大冗余度的语言统计组织结构仍不清楚。在此，我们引入了一个语言的统计框架，将语言的冗余性与文本的分层语义组织联系起来。为此，我们使用大语言模型将任意给定文本递归地分割成语义连贯的块，从而构建出一个“语义树”，该树的每个层级对应文本不同粒度的语义单元，并据此计算各层级上语言的信息熵。

    arXiv:2602.13194v3 Announce Type: replace-cross  Abstract: Humans and large language models can predict next letter or word from its prior context much better than random guessing, indicating strong redundancy of language viewed as a stochastic process. Quantitatively this redundancy was estimated by Shannon to be around 80\%, which means that every letter of a printed English text conveys approximately 1 bit of information and not 4.8 bits that 27 letters (including spaces) could potentially carry. This estimate was later confirmed by using autoregressive token probabilies computed by large language models. However, the statistical organization of language that give rise to such a large redundancy remains unclear. Here we introduce a statistical framework of language linking its redundancy to the hierarchical semantic organization of text. To this end, we use large language models to recursively segment any given text into semantically coherent chunks, inducing a ``semantic tree'' tha
    
[^251]: ResidualKV：基于残差的KV缓存压缩，实现高效长上下文推理

    ResidualKV: Residual-Based KV Cache Compression for Efficient Long-Context Inference

    [https://arxiv.org/abs/2602.08005](https://arxiv.org/abs/2602.08005)

    ResidualKV提出将KV缓存分解为稀疏全局参考与量化残差编码，结合稀疏注意力按需重建状态，并通过动态步长调度抑制参考增长，在不永久丢弃令牌信息的前提下实现高效长上下文推理。

    

    高效的长上下文推理面临两个相互耦合的瓶颈：KV缓存内存随上下文长度线性增长，而注意力计算则呈二次方增长。现有方法通常只能解决其中一个问题，且以不可逆的令牌驱逐、完整缓存保留或完整历史重建为代价，这限制了它们在多轮交互和长篇推理中的有效性。受两个经验性发现——长程令牌间相似性和平滑残差分布——的启发，我们提出了ResidualKV，该方法将KV缓存分解为一组稀疏的全局检索参考，以及针对其余令牌的紧凑量化残差编码。这种表示方式在无需永久驱逐令牌的情况下保留了令牌特定的信息，并且在与稀疏注意力结合时，可按需仅重建被选中的状态。动态步长调度进一步将参考数量的增长从线性降低至近似对数级别……

    arXiv:2602.08005v2 Announce Type: replace-cross  Abstract: Efficient long-context inference faces two coupled bottlenecks: KV-cache memory grows linearly with context length, while attention computation grows quadratically. Existing approaches typically address one at the expense of irreversible token eviction, full-cache retention, or full-history reconstruction, limiting their effectiveness for multi-turn interaction and long-form reasoning. Motivated by two empirical properties, Long-Range Inter-Token Similarity and Smooth Residual Distribution, we propose ResidualKV, which factorizes the KV cache into a sparse set of globally retrieved references and compact, quantized residual codes for the remaining tokens. This representation preserves token-specific information without permanent eviction and, when combined with sparse attention, reconstructs only the selected states on demand. Dynamic-stride scheduling further reduces reference growth from linear to approximately logarithmic at
    
[^252]: 功能子空间：语言模型利用向量代数解决问题的空间

    Functional Subspace, where language models can use vector algebra to solve problems

    [https://arxiv.org/abs/2602.01687](https://arxiv.org/abs/2602.01687)

    该研究提出假设：大型语言模型通过在功能子空间中运用向量代数来执行任务，并通过分析上下文学习过程中LLM的功能模块与残差流来验证这一假设。

    

    大型语言模型（LLM）最初是为翻译等自然语言任务而发明的，但事实证明它们能够跨领域执行高度复杂的函数运算。此外，人们认为它们可以在未经专门训练的情况下发展出新技能。这些学习能力促使LLM在众多领域得到广泛应用。因此，理解它们的运行机制与局限性对于正确的诊断和修复至关重要。早期的研究提出，高级概念在LLM的激活空间中被编码为线性方向，且嵌入的几何结构具有语义含义。受这些研究的启发，我们假设LLM可能利用子空间以及子空间中的向量代数来执行任务。为了验证这一假设，我们分析了从事上下文学习（ICL）这一涌现能力的LLM的功能模块和残差流。我们的分析……

    arXiv:2602.01687v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) were invented for natural language tasks such as translation, but they have proved that they can perform highly complex functions across domains. Additionally, they have been thought to develop new skills without being trained on them. These learning capabilities lead to LLMs adoption in a wide range of domains. Thus, it is imperative that we understand their operating mechanisms and limitations for proper diagnostics and repair. The earlier studies proposed that high level concepts are encoded as linear directions in LLMs activation space and that the geometry of embeddings have semantic meanings. Inspired by these studies, we hypothesize that LLMs may use subspaces and vector algebra in subspaces to perform tasks. To address this hypothesis, we analyze LLMs' functional modules and residual streams collected from LLMs engaging in in-context learning (ICL), one of the emergent abilities. Our analyse
    
[^253]: 最低跨度置信度：基于单次大语言模型响应的零样本幻觉检测

    Lowest Span Confidence: Zero-Shot Hallucination Detection from a Single LLM Response

    [https://arxiv.org/abs/2601.19918](https://arxiv.org/abs/2601.19918)

    提出零样本指标“最低跨度置信度”，仅需单次LLM响应即可检测幻觉，无需昂贵重复采样或访问模型内部状态。

    

    大语言模型中的幻觉，即看似合理但与事实不符的生成内容，对模型在高风险环境中的可靠部署构成了重大挑战。然而，许多现有的幻觉检测器需要昂贵的高频重复采样来进行一致性检查，或者需要访问在常见的API调用场景中无法获取的模型内部状态。为此，我们提出了一种高效的零样本指标——最低跨度置信度，用于在最小资源假设下进行幻觉检测。具体而言，LSC评估相邻完整词跨度的局部置信度。通过选择token宽度可变的相邻词中最低的聚合置信度，LSC能够捕捉与事实不一致相关的局部不确定性。这种边界对齐的平滑处理减少了困惑度指标的全局稀释效应，以及最小token概率对孤立噪声的敏感性。我们的主要评估涵盖四个模型族……

    arXiv:2601.19918v2 Announce Type: replace  Abstract: Hallucinations in Large Language Models (LLMs), i.e., plausible but non-factual generations, pose a significant challenge to reliable deployment in high-stakes environments. However, many existing hallucination detectors require expensive repeated sampling for consistency checks or access to model-internal states unavailable in common API-based scenarios. To this end, we propose an efficient zero-shot metric called Lowest Span Confidence (LSC) for hallucination detection under minimal resource assumptions. Concretely, LSC evaluates the local confidence of adjacent complete-word spans. By selecting the lowest aggregated confidence across neighboring words whose token widths can vary, LSC captures localized uncertainty associated with factual inconsistency. This boundary-aligned smoothing reduces the global dilution of perplexity and the sensitivity of minimum token probability to isolated noise. Our main evaluation spans four model fa
    
[^254]: 在线对话式健康数据中敏感信息的上下文感知分类与分级

    Context-Aware Classification and Grading of Sensitive Information in Online Conversational Health Data

    [https://arxiv.org/abs/2601.09717](https://arxiv.org/abs/2601.09717)

    本研究将在线医疗对话中的敏感信息分级构建为上下文感知评估任务，提出纳入断言状态、经历者、检查结果状态和信息粒度的操作框架，并借助最小化改变上下文因素的对比案例来评估大语言模型对隐私敏感信息的判断能力。

    

    在线医疗咨询包含敏感的健康信息，其隐私影响不仅取决于所提及的实体，还取决于这些实体在上下文中的描述方式。现有的分类和分级方法通常将健康信息实体直接映射到预定义的敏感度级别，可能忽略了某种疾病是被确认的、疑似存在的、被否定的、属于假设性的，还是仅仅计划进行检查。在本研究中，我们将在线医疗对话中的敏感信息分级形式化为一个上下文感知的评估任务。我们开发了一个基于标准的可操作框架，其中纳入了断言状态、经历者、检查结果状态和信息粒度。我们进一步设计了一个贴近真实场景的评估设置以及对比案例，这些案例在否定、不确定性、经历者或信息粒度上进行最小程度的改变，并在仅提及和完整上下文等条件下比较了大型语言模型的表现……

    arXiv:2601.09717v2 Announce Type: replace-cross  Abstract: Online medical consultations contain sensitive health information whose privacy implications depend not only on the entities mentioned but also on how those entities are described in context. Existing classification and grading approaches often map health-information entities directly to predefined sensitivity levels, potentially overlooking whether a condition is confirmed, suspected, negated, hypothetical, or merely planned for investigation. In this study, we formulate sensitive-information grading in online medical dialogues as a context-aware evaluation task. We develop a standard-informed operational framework that incorporates assertion status, experiencer, test-result status, and information granularity. We further design a naturalistic evaluation setting together with contrastive cases that minimally alter negation, uncertainty, experiencer, or granularity, and compare large language models under mention-only and full-
    
[^255]: 表面反思还是真实思考？大型推理模型的细粒度认知分析

    Superficial Reflection or Genuine Thought? A Fine-Grained Cognitive Analysis of Large Reasoning Models

    [https://arxiv.org/abs/2512.00729](https://arxiv.org/abs/2512.00729)

    本文提出基于人类认知过程的细粒度推理步骤分类体系（5组17类），揭示当前大型推理模型答案后的“复查”反思多为表面行为而非真实思考，并提出CAPO自动标注方法构建了27万余条推理步骤的数据集，证明显式引导更丰富的反思过程可显著改善模型自纠正能力。

    

    受大型推理模型中观察到的类人行为启发，本文引入了一个全面的分类体系来刻画原子推理步骤，并分析大型推理模型的推理行为。基于人类认知过程，我们提出了一个包含五组十七类的分类体系。借助该分类体系，我们对当代大型推理模型进行了深入分析，并提炼出四条可用于模型优化的可操作要点。最值得注意的是，我们发现当前普遍存在的答案后“复查”行为在很大程度上是表面的，很少带来实质性的修改。一项针对性的干预实验进一步表明，显性地引导更丰富的反思过程可以显著改善失败的自纠正。为支持这项大规模研究，我们提出了CAPO——一种自动化标注方法，用于构建一个包含277,534个推理步骤、与人类专家标注高度一致的数据集。我们进一步验证了……

    arXiv:2512.00729v2 Announce Type: replace  Abstract: Motivated by the observed human-like behaviours in Large Reasoning Models (LRMs), this paper introduces a comprehensive taxonomy to characterise atomic reasoning steps and analyse the reasoning behaviours of LRMs. Grounded in human cognitive processes, we propose a taxonomy comprising five groups and seventeen categories. Through this taxonomy, we conduct an in-depth analysis of contemporary LRMs and distil four actionable takeaways for model optimisation. Most notably, we reveal that prevailing post-answer ``doublechecks'' are largely superficial and rarely yield substantive revisions. A targeted intervention further shows that explicitly eliciting richer reflection processes can substantially improve failed self-correction. To support this largescale study, we propose CAPO, an automated annotation method used to construct a dataset of 277,534 reasoning steps with strong agreement with human expert annotations. We further validate t
    
[^256]: 从复合图像到医学多图像推理：基于生物医学文献扩展多模态大语言模型

    From Compound Figures to Medical Multi-image Reasoning: Scaling Multimodal Large Language Models with Biomedical Literature

    [https://arxiv.org/abs/2511.22232](https://arxiv.org/abs/2511.22232)

    该论文构建了源自生物医学复合图像的大规模医学多图像指令数据集PMC-MI及人工审核基准PMC-MI-Bench，并提出三阶段训练框架M3LLM，通过监督微调与选择感知强化学习提升多模态大语言模型的医学多图像推理能力。

    

    多模态大语言模型（MLLMs）在医学影像领域的能力日益增强，但大多数研究仍集中于单图像场景。而临床解读往往需要整合跨多张图像的证据，例如不同的成像模态、视角或时间点。然而，大规模的医学多图像数据以及针对此类推理的训练策略仍然匮乏。我们构建了PMC-MI，这是一个大规模资源，包含源自生物医学复合图像的234,956个指令实例，其中10,555个多子图实例经过结构化处理用于强化学习，并由医学审稿人进行评估。我们还引入了PMC-MI-Bench，这是一个在源文章层面进行划分的人工审核基准。此外，我们提出了一个三阶段训练框架，实例化为M3LLM，将多图像指令上的监督微调、面向问题条件的视觉证据选择的选择感知强化学习，以及s（摘要内容于此处截断）

    arXiv:2511.22232v2 Announce Type: replace-cross  Abstract: Multimodal large language models (MLLMs) are increasingly capable in medical imaging, yet most focus on single-image settings. Clinical interpretation often requires integrating evidence across multiple images, such as different modalities, views, or time points. However, large-scale medical multi-image data and training strategies for such reasoning remain limited. We construct PMC-MI, a large-scale resource comprising 234,956 instruction instances derived from biomedical compound figures, with 10,555 multi-subimage instances structured for reinforcement learning and assessed by medical reviewers. We also introduce PMC-MI-Bench, a manually reviewed benchmark separated at the source-article level. We further propose a three-stage training framework, instantiated as M3LLM, combining supervised fine-tuning on multi-image instructions, selection-aware reinforcement learning for question-conditioned visual-evidence selection, and s
    
[^257]: 专为社交与心理健康打造的生成式人工智能：一项真实世界试点研究

    Generative AI Purpose-built for Social and Mental Health: A Real-World Pilot

    [https://arxiv.org/abs/2511.11689](https://arxiv.org/abs/2511.11689)

    一项纳入299名美国成年人、随访长达12个月的真实世界试点研究表明，专为心理健康训练的生成式AI基础模型能在10周内显著降低抑郁和焦虑症状（Cohen's d达0.93和0.79），改善孤独感与社交互动，并借助临床医生确认的自动化安全防护机制实现安全有效的干预。

    

    专为心理健康构建的生成式AI聊天机器人有望扩大心理医疗服务的可及性，但来自真实世界使用的证据仍然有限。我们报告了一项单臂、自然环境的试点研究，研究对象是一个专为心理健康训练的基础模型，共纳入299名至少有中度抑郁或焦虑症状的美国成年人，随访期长达12个月。结果显示，抑郁和焦虑症状在10周内显著下降（Cohen's d分别为0.93和0.79），孤独感、行为激活和社交互动也有所改善。临床医生确认自动化安全防护措施得到了适当的升级处理。参与者可分为无应答（57.2%）、改善（37.1%）和快速改善（5.7%）三种轨迹。早期的治疗联盟和更高的参与度与更好的结局相关。该AI部署了十种已识别的干预方法族，其交付方式随基线焦虑和抑郁水平而变化，临床内容与非临床内容的总体比例随（原文此处截断）。

    arXiv:2511.11689v4 Announce Type: replace-cross  Abstract: Generative AI chatbots built for mental health could extend access to care, but evidence from real-world use is limited. We report a single-arm, naturalistic pilot of a foundation model trained for mental health, among 299 US adults with at least moderate depressive or anxiety symptoms who were followed for up to 12 months. Depression and anxiety symptoms fell by 10 weeks (Cohen's d 0.93 and 0.79), with loneliness, behavioral activation, and social interaction improving. Clinicians confirmed that automated safeguards were escalated appropriately. Participants fell into non-responding (57.2%), improving (37.1%) and rapidly improving (5.7%) trajectories. Early working alliance and greater engagement were associated with better outcomes. The AI deployed ten identified intervention families whose delivery varied with baseline anxiety and depression, with the overall ratio of clinical to non-clinical content increasing according to 
    
[^258]: 大语言模型行为的序贯贝叶斯评估

    Sequential Bayesian Evaluation of Large Language Model Behavior

    [https://arxiv.org/abs/2511.10661](https://arxiv.org/abs/2511.10661)

    本文提出一种序贯贝叶斯评估框架，通过量化LLM随机性带来的评估不确定性，并自适应地优先选择下一个最值得评估的基准提示词，从而实现更具成本效益的大语言模型行为评估。

    

    评估基于大语言模型（LLM）的系统的特性正变得日益重要。此类评估通常依赖于向LLM提供的一组精心筛选的基准提示词（prompt）集合，其中每个提示词的输出可能被赋予二元或有序的分数，随后将各提示词的分数汇总作为总结性评估。在本文中，我们开发了一种贝叶斯方法，用于量化此类评估指标中由于基于LLM系统的随机性而产生的不确定性——即同一提示词在重复运行时可能表现出不同的结果。我们的框架自然地引出了一种序贯评估方法，即利用贝叶斯模型优先选择基准中接下来应使用哪些提示词，从而实现更具成本效益的LLM评估。我们通过四个案例研究展示了该方法：交互式对话中的LLM成对偏好评估（MT-Bench）、……

    arXiv:2511.10661v2 Announce Type: replace  Abstract: It is increasingly important to evaluate the characteristics of systems based on large language models (LLMs). Evaluations in this context often rely on a curated benchmark set of input prompts provided to the LLM, where the output for each prompt may be assigned a binary or ordinal score and the aggregation of scores across prompts is then used as a summary evaluation. In this paper, we develop a Bayesian approach for quantifying the uncertainty that arises in such evaluation metrics as a result of the stochasticity of the LLM-based systems; the same prompt may exhibit different outcomes on repeated runs. Our framework leads naturally to a sequential evaluation, in which we leverage the Bayesian model to preferentially select which prompts in the benchmark to use next, enabling more cost-effective LLM evaluations. We demonstrate this approach through four case studies: pairwise LLM preferences in interactive dialogue (MT-Bench), ref
    
[^259]: MedRECT：面向临床文本错误纠正的双语医学推理基准

    MedRECT: A Bilingual Medical Reasoning Benchmark for Error Correction in Clinical Texts

    [https://arxiv.org/abs/2511.00421](https://arxiv.org/abs/2511.00421)

    提出了MedRECT双语（日英）医学基准，将临床文本错误处理形式化为错误检测、错误句子提取和错误纠正三个子任务，并通过对11个LLM的评估发现思考模式能显著提升错误检测与句子提取性能。

    

    大型语言模型（LLM）在医学应用中展现出前景，但其在临床文本中检测和纠正错误的能力仍未得到充分评估，尤其是在英语以外的语言中。我们提出了MedRECT，一个日语和英语的双语基准，将医学错误处理形式化为三个子任务：错误检测、错误句子提取和错误纠正。MedRECT-ja包含663个源自日本医师执照考试的样本，而单独来源的MedRECT-en包含458个从MEDEC精选的样本。我们评估了11个LLM的17种配置，涵盖专有模型与开放权重模型、医学领域专门化模型以及多种推理设置。Qwen3-32B在思考模式下的错误检测F1分数和句子提取准确率均高于非思考模式，且这一优势在两个子集上均成立，其中句子提取准确率在MedRECT-ja上高出24.5个百分点，在MedRECT-en上高出10（个百分点）。

    arXiv:2511.00421v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) show promise in medical applications, but their ability to detect and correct errors in clinical texts remains under-evaluated, particularly beyond English. We introduce MedRECT, a bilingual benchmark for Japanese and English that formulates medical error handling as three subtasks: error detection, error sentence extraction, and error correction. MedRECT-ja contains 663 samples derived from the Japanese Medical Licensing Examinations, while the separately sourced MedRECT-en contains 458 samples curated from MEDEC. We evaluate 11 LLMs across 17 configurations that cover proprietary and open-weight models, medical-domain specialization, and multiple reasoning settings. Qwen3-32B scores higher in its thinking mode than in its non-thinking mode on error detection F1 and sentence extraction accuracy in both subsets, with sentence extraction accuracy higher by 24.5 percentage points on MedRECT-ja and 10.
    
[^260]: 面向小型语言模型的推理时指令检索

    Instruction Retrieval at Inference Time for Small Language Models

    [https://arxiv.org/abs/2510.13935](https://arxiv.org/abs/2510.13935)

    提出指令检索方法，将教师模型的专业知识提炼为针对小模型定制的指令语料库（每个领域仅需构建一次、运行时无需教师模型），使小型语言模型在推理时通过检索指令获得背景知识、解题流程和常见错误提示，从而胜任需要专业知识的专家级任务。

    

    语言模型所存储的事实与其参数量紧密相关，因此适合部署在边缘设备上的小型模型在处理专家级问题时表现不佳，这类问题既需要专门的知识，又需要遵循多步骤的操作流程。针对特定领域或任务进行微调可以将知识写入模型参数，但必须针对每个模型和每个领域重复进行；而检索到的文本段落则要求模型自行找到相关事实并加以应用。我们提出了指令检索，该方法将教师模型的专业知识提炼为一个指令语料库，这些指令经过定制设计，使小型模型能够遵循执行。对于某一领域问题的每个聚类，教师模型会编写一条指令，其中包含该聚类所依赖的背景知识、针对该类问题的操作流程，以及在此类问题上的常见错误。这个可复用的语料库每个领域只需构建一次，且在推理运行时无需访问教师模型。在推理阶段，冻结的小型模型检索相应的指令

    arXiv:2510.13935v3 Announce Type: replace-cross  Abstract: The facts a language model stores are tied to its parameter count, so small models that fit on edge devices fail on expert problems, which need specialized knowledge and follow multi-step procedures. Fine-tuning for a specific domain or task writes the knowledge into the parameters but must be repeated for every model and domain, and a retrieved passage leaves the model to find the relevant fact and apply it on its own. We introduce instruction retrieval, which distills a teacher model's expertise into a corpus of instructions tailored so that a small model can follow. For each cluster of a domain's problems, the teacher writes one instruction with the background knowledge the cluster depends on, a procedure for that kind of problem, and the common mistakes made on it. This reusable corpus needs only to be built once per domain and requires no run-time teacher access. At inference, a frozen small model retrieves the instruction
    
[^261]: 正确思考：通过自适应、注意力驱动的压缩学习缓解思考不足与过度思考

    Think Right: Learning to Mitigate Under-Over Thinking via Adaptive, Attentive Compression

    [https://arxiv.org/abs/2510.01581](https://arxiv.org/abs/2510.01581)

    提出 TRAAC，一种利用模型自注意力机制识别并剪除冗余推理步骤、并将估计的问题难度纳入训练奖励的在线后训练强化学习方法，从而在思考不足与过度思考之间取得平衡，解决适应性不足问题。

    

    arXiv:2510.01581v2 公告类型：replace-cross。摘要：近期的思考模型能够通过扩展测试时计算量来解决复杂的推理任务，但这种计算量的扩展应当与任务难度相匹配。一方面，过短的推理（思考不足）会导致模型在需要多步延伸推理的难题上出错；另一方面，过长的推理（过度思考）即使在已经得到正确的中间解之后仍会生成不必要的步骤，造成 token 效率低下。我们将这种现象称为“适应性不足”，即模型未能根据不同难度的问题恰当地调节其回答长度。为了解决适应性不足问题，并在思考不足与过度思考之间取得平衡，我们提出了 TRAAC（Think Right with Adaptive, Attentive Compression，通过自适应、注意力压缩实现正确思考），这是一种在线后训练强化学习方法，它利用模型的自注意力机制来识别关键推理步骤并剪除冗余步骤。TRAAC 还会估计问题难度，并将其纳入训练奖励之中……

    arXiv:2510.01581v2 Announce Type: replace-cross  Abstract: Recent thinking models are capable of solving complex reasoning tasks by scaling test-time compute, but this scaling should be allocated in line with task difficulty. On one hand, short reasoning (underthinking) leads to errors on harder problems that require extended reasoning steps; but, excessively long reasoning (overthinking) can be token-inefficient by generating unnecessary steps even after reaching a correct intermediate solution. We refer to this as under-adaptivity, where the model fails to modulate its response length appropriately given problems of varying difficulty. To address under-adaptivity and strike a balance between under- and overthinking, we propose TRAAC (Think Right with Adaptive, Attentive Compression), an online post-training RL method that leverages the model's self-attention to identify key steps and prune redundant ones. TRAAC also estimates difficulty and incorporates it into training rewards, ther
    
[^262]: 从构建到注入：面向大语言模型的基于编辑的指纹技术

    From Construction to Injection: Edit-Based Fingerprints for Large Language Models

    [https://arxiv.org/abs/2509.03122](https://arxiv.org/abs/2509.03122)

    该论文提出了一个端到端的大语言模型注入式指纹框架，通过基于代码混合的指纹构建方法解决不可察觉性权衡问题，并确保指纹在模型遭受修改后仍能维持持久的触发-目标行为，从而实现更鲁棒的模型所有权验证。

    

    可靠的模型指纹对于保护大语言模型（LLM）免受未经授权的再分发和商业滥用至关重要。在黑盒部署场景下，验证工作受到两方面的阻碍：一是对疑似指纹查询的防御性过滤，二是可能削弱嵌入所有权证据的下游模型修改。这些风险要求指纹在构建和注入两个环节都具备鲁棒性。在构建方面，先前的范式面临不可察觉性的权衡：自然语言指纹可能被意外激活，而乱码指纹则在统计上暴露特征、更容易被过滤。在注入方面，现有方法难以在模型修改后仍保持持久的触发-目标行为。我们提出一个端到端的注入式指纹框架来应对这些挑战。代码混合指纹（CF）在高复杂度约束下使用最低困惑度的代码混合来缓解……（原文摘要在此处截断）

    arXiv:2509.03122v5 Announce Type: replace-cross  Abstract: Reliable model fingerprints are essential for protecting large language models (LLMs) against unauthorized redistribution and commercial misuse. In black-box deployment, verification is hindered by defensive filtering of suspected fingerprint queries, as well as by downstream model modifications that may weaken embedded ownership evidence. These risks require fingerprints to be robust in both construction and injection. For construction, prior paradigms face an imperceptibility trade-off: natural-language fingerprints may be accidentally activated, whereas garbled fingerprints are statistically exposed and easier to filter. For injection, existing methods struggle to preserve persistent trigger--target behaviors under model modification. We propose an end-to-end injected fingerprinting framework to address these challenges. Code-mixing Fingerprints (CF) use lowest-perplexity code-mixing under a high-complexity constraint to mit
    
[^263]: NMIXX：面向跨语言金融探索的领域自适应神经嵌入

    NMIXX: Domain-Adapted Neural Embeddings for Cross-Lingual eXploration of Finance

    [https://arxiv.org/abs/2507.09601](https://arxiv.org/abs/2507.09601)

    该论文提出NMIXX方法，通过构建18.8k个含语义对比的金融领域三元组对现有编码器进行领域适配，显著提升了英语和韩语金融文本嵌入的语义相似度性能（如BGE-M3在FinSTS上从0.1969提升至0.2967），但代价是通用领域性能略有下降。

    

    金融文本嵌入必须能够区分事件状态、视角和义务的变化，即使段落之间共享相似的措辞。NMIXX通过18.8k个与源文本关联的三元组对现有编码器进行领域适配：其中改写和韩英翻译用于保留原意，而针对性的金融改写则引入语义对比。我们在七个骨干模型上检验了这一方法在英语和韩语的金融领域及通用领域语义文本相似度（STS）任务上的表现，并分析了KorFinSTS基准的构成和段落长度。在此比较中，BGE-M3获得了最高的适配后金融相关性，在FinSTS上从0.1969提升至0.2967，在KorFinSTS上从0.0512提升至0.2732；其通用英语和韩语相关性则分别下降了0.0391和0.0463。在七个模型中，五个模型的平均金融相关性有所提升，但所有模型的平均通用领域相关性均有所下降。按语言的比较和基准权重敏感性分析（原文在此处被截断）。

    arXiv:2507.09601v3 Announce Type: replace-cross  Abstract: Financial text embeddings must distinguish changes in event status, perspective, and obligations even when passages share similar wording. NMIXX adapts existing encoders through 18.8k source-linked triplets: paraphrases and Korean-English translations preserve meaning, while targeted financial rewrites introduce semantic contrasts. We examine this recipe across seven backbones on English and Korean financial and general-domain semantic textual similarity (STS), and analyze the composition and passage lengths of KorFinSTS. BGE-M3 attains the highest adapted financial correlations in this comparison, improving from 0.1969 to 0.2967 on FinSTS and from 0.0512 to 0.2732 on KorFinSTS. Its general English and Korean correlations decrease by 0.0391 and 0.0463. Across the seven models, five improve their mean financial correlation, but all reduce their mean general-domain correlation. Per-language comparisons and benchmark-weight sensit
    
[^264]: 面向语言模型可解释性的跨层离散概念发现

    Cross-Layer Discrete Concept Discovery for Interpreting Language Models

    [https://arxiv.org/abs/2506.20040](https://arxiv.org/abs/2506.20040)

    提出跨层向量量化自编码器CLVQ-VAE，通过离散向量量化瓶颈将残差流中跨层重复的特征压缩为紧凑、可解释的概念向量，从而更有效地解释语言模型。

    

    由于残差流的存在，语言模型的解释仍然充满挑战——残差流会在相邻层之间线性地混合并复制特征，导致单层分析无法捕捉这种跨层结构。跨层稀疏自编码器（SAE）虽然解决了层间混合问题，但其运行在连续空间中，概念会分散到众多神经元上且缺乏清晰的边界。我们提出了跨层向量量化变分自编码器（CLVQ-VAE），这是一种新颖的框架，通过离散的向量量化瓶颈将低层表示映射到高层，把残差流中重复的特征压缩为紧凑、可解释的概念向量。我们的方法将基于top-k的温度采样与指数移动平均（EMA）码本更新相结合，在对离散潜在空间进行可控探索的同时保持码本的多样性。在基于编码器和基于解码器的模型上……（原文摘要在此处截断）

    arXiv:2506.20040v4 Announce Type: replace-cross  Abstract: Interpreting language models remains challenging due to the existence of residual stream, which linearly mixes and duplicates features across adjacent layers, causing single-layer analyses to miss this cross-layer structure. Cross-layer sparse autoencoders (SAEs) address layer mixing but operate in continuous space, where concepts split across many neurons without clear boundaries. We introduce Cross-Layer Vector Quantized-Variational Autoencoder (CLVQ-VAE), a novel framework which maps representations from a lower layer to a higher layer through a discrete vector-quantization bottleneck, collapsing duplicated residual-stream features into compact, interpretable concept vectors. Our approach combines top-k temperature-based sampling with exponential moving average (EMA) codebook updates, providing controlled exploration of the discrete latent space while maintaining codebook diversity. Across both encoder- and decoder-based mod
    
[^265]: 自由职业专业作家对人工智能的看法：局限性、期望与恐惧

    Voices of Freelance Professional Writers on AI: Limitations, Expectations, and Fears

    [https://arxiv.org/abs/2504.05008](https://arxiv.org/abs/2504.05008)

    本研究通过对301名自由职业作家的问卷调查和互动任务发现，AI写作工具的采用主要受同伴影响和职业前景驱动而非人口特征，多语言作家面临公平使用AI的性能与感知双重障碍，且对工作安全的担忧普遍存在、与是否使用AI无关。

    

    人工智能驱动的工具，特别是大语言模型（LLMs）的快速发展，正在重塑专业写作领域。然而，其采用过程中的关键问题，如语言支持、伦理考量以及对作家个人声音和创造力的长期影响，仍然缺乏深入研究。在这项工作中，我们对经常使用人工智能的自由职业专业作家开展了问卷调查（N = 301）和互动任务（N = 36）。我们研究了涵盖25种以上语言的人工智能辅助写作实践、伦理关切以及用户期望。我们的研究结果表明：工具的采用更多受同伴影响和职业前景的驱动，而非人口统计学因素；多语言作家在公平使用人工智能方面面临性能和认知感知的双重障碍；对工作安全的担忧普遍存在，且与是否采用人工智能无关。

    arXiv:2504.05008v3 Announce Type: replace  Abstract: The rapid development of AI-driven tools, particularly large language models (LLMs), is reshaping professional writing. Still, key aspects of their adoption such as language support, ethics, and long-term impact on writers' voice and creativity remain underexplored. In this work, we carried out a questionnaire (N = 301) and an interactive task (N = 36) targeting freelance professional writers regularly using AI. We examined AI-assisted writing practices across 25+ languages, ethical concerns, and user expectations. Our findings reveal that adoption is shaped more by peer influence and professional outlook than demographics, that multilingual writers face both performance and perception barriers to equitable AI use, and that job security concerns are widespread and adoption-independent.
    
[^266]: 倾听智慧的少数：查询-键对齐解锁大语言模型中潜在的正确答案

    Listening to the Wise Few: Query-Key Alignment Unlocks Latent Correct Answers in Large Language Models

    [https://arxiv.org/abs/2410.02343](https://arxiv.org/abs/2410.02343)

    本文发现在去除旋转位置编码（RoPE）后计算的查询-键分数可以识别中间层中一类通用的“选择-复制”注意力头，这些头通过语义对齐从模型内部编码的潜在知识中可靠地定位多项选择题的正确答案，从而以可解释的机制方式揭示 LLM “知道但不说出”的现象。

    

    大语言模型（LLMs）在多项选择题问答（MCQA）中经常无法输出正确的选项，尽管其内部已经编码了答案。我们通过查询-键（QK）分数来揭示这种潜在知识，该分数针对某个注意力头定义为最后一个词元的查询与选项 i 之后行尾词元的键之间的内积，并在应用旋转位置编码（RoPE）之前进行计算。其 argmax 可以识别出中间层中一类通用的“选择-复制”注意力头，这些注意力头通过语义查询-键对齐来执行选项选择，其机制与归纳头和复制抑制头（Olsson et al., 2022）截然不同：它们对标签符号保持不变，并且能够解决零表面重叠的合成任务——这些性质是任何基于位置复制的解释都无法说明的，而且关键地需要在计算前去除外 RoPE 才能体现。在从 1.5B 到 72B 参数的 24 个模型（包括 LLaMA-2/3/3.1/3.3、Qwen-2.5、Gemma、Phi-3.5、DeepSeek-R1-Distill 等）上验证了该方法的有效性。

    arXiv:2410.02343v2 Announce Type: replace  Abstract: Large language models (LLMs) routinely fail to output the correct option in multiple-choice question answering (MCQA) while encoding the answer internally. We expose this latent knowledge via the Query--Key (QK) score, defined for an attention head as the inner product between the last-token query and the key at the end-of-line token following option $i$, evaluated before rotary positional embedding is applied. Its argmax identifies a universal class of select-and-copy heads in middle layers that perform option selection through semantic query--key alignment, mechanistically distinct from induction and copy-suppression heads (Olsson et al., 2022): they are invariant to label symbols, and solve a synthetic task with zero surface overlap---properties no positional-copy account explains and that critically require stripping RoPE.   Across 24 models from 1.5B to 72B parameters (LLaMA-2/3/3.1/3.3, Qwen-2.5, Gemma, Phi-3.5, DeepSeek-R1-Dis
    
[^267]: 缓解语言模型的记忆化问题

    Mitigating Memorization In Language Models

    [https://arxiv.org/abs/2410.02159](https://arxiv.org/abs/2410.02159)

    该论文提出了17种缓解语言模型记忆训练数据问题的方法（包括5种全新的机器遗忘方法）以及高效小模型评估套件TinyMem，并证明用TinyMem开发的方法可成功迁移应用于生产级语言模型。

    

    语言模型（LM）可能会“记忆”信息，即以某种方式将训练数据编码到其权重中，使得推理时的查询可能导致这些数据被逐字复述出来。这种提取训练数据的能力可能带来问题，例如当数据涉及隐私或敏感信息时。在这项工作中，我们研究了减轻记忆化的方法：包括三种基于正则化的方法、三种基于微调的方法，以及十一种基于机器遗忘的方法，其中后者有五种是我们新提出的方法。我们还推出了TinyMem，这是一套小型、计算高效的语言模型套件，用于快速开发和评估记忆化缓解方法。我们证明，使用TinyMem开发的缓解方法可以成功应用于生产级别的语言模型，并且我们通过实验发现：基于正则化的缓解方法在抑制记忆化方面既缓慢又低效；基于微调的方法……

    arXiv:2410.02159v3 Announce Type: replace-cross  Abstract: Language models (LMs) can "memorize" information, i.e., encode training data in their weights in such a way that inference-time queries can lead to verbatim regurgitation of that data. This ability to extract training data can be problematic, for example, when data are private or sensitive. In this work, we investigate methods to mitigate memorization: three regularizer-based, three finetuning-based, and eleven machine unlearning-based methods, with five of the latter being new methods that we introduce. We also introduce TinyMem, a suite of small, computationally-efficient LMs for the rapid development and evaluation of memorization-mitigation methods. We demonstrate that the mitigation methods that we develop using TinyMem can successfully be applied to production-grade LMs, and we determine via experiment that: regularizer-based mitigation methods are slow and ineffective at curbing memorization; fine-tuning-based methods ar
    
[^268]: 基于两阶段多分辨率集成模型的鲁棒唤醒词检测

    Robust Wake-Up Word Detection by Two-stage Multi-resolution Ensembles

    [https://arxiv.org/abs/2310.11379](https://arxiv.org/abs/2310.11379)

    本文提出一种两阶段多分辨率集成检测框架，利用轻量级设备端模型实时处理音频流，再由服务器端异构集成验证模型进行二次确认，从而在两个工作点上实现鲁棒、节能且保护隐私的唤醒词检测。

    

    基于语音的交互界面依赖唤醒词机制来启动与设备的通信。然而，实现鲁棒、节能且快速的检测仍然是一个挑战。本文通过利用时间对齐来增强数据，并采用基于两阶段的多分辨率检测方法，来满足这些实际生产需求。该方案采用两个模型：一个用于实时处理音频流的轻量级设备端模型，以及一个位于服务器端的验证模型，该验证模型是异构架构的集成，用于对检测结果进行精细化处理。这种方案允许对两个工作点分别进行优化。为保护隐私，发送到云端的是音频特征而非原始音频。该研究针对特征提取的不同参数配置进行了探究，为设备端检测和验证模型分别选择了不同的配置。此外，还比较了十三种不同的音频分类器的性能表现。

    arXiv:2310.11379v2 Announce Type: replace-cross  Abstract: Voice-based interfaces rely on a wake-up word mechanism to initiate communication with devices. However, achieving a robust, energy-efficient, and fast detection remains a challenge. This paper addresses these real production needs by enhancing data with temporal alignments and using detection based on two phases with multi-resolution. It employs two models: a lightweight on-device model for real-time processing of the audio stream and a verification model on the server-side, which is an ensemble of heterogeneous architectures that refine detection. This scheme allows the optimization of two operating points. To protect privacy, audio features are sent to the cloud instead of raw audio. The study investigated different parametric configurations for feature extraction to select one for on-device detection and another for the verification model. Furthermore, thirteen different audio classifiers were compared in terms of performan
    
[^269]: ETHER: 对于回顾性经验重演的紧密沟通对齐

    ETHER: Aligning Emergent Communication for Hindsight Experience Replay. (arXiv:2307.15494v1 [cs.CL])

    [http://arxiv.org/abs/2307.15494](http://arxiv.org/abs/2307.15494)

    本文提出了ETHER，通过对齐紧急沟通来解决回顾性经验重演中的问题，克服了先前架构依赖预设函数的限制，并提高了数据效率和性能。

    

    自然语言指令的跟随对于实现人工智能代理和人类之间的合作至关重要。自然语言条件下的强化学习代理展示了自然语言的特性，如组合性，能够提供学习复杂策略的强归纳偏好。先前的架构如HIGhER结合了语言条件与回顾性经验重演（HER）来处理稀疏奖励环境。然而，与HER类似，HIGhER依赖于一个预设的函数来提供反馈信号，指示哪种语言描述在哪种状态下有效。这种依赖于预设函数的限制限制了其应用。此外，HIGhER只利用成功的强化学习轨迹中包含的语言信息，从而影响了其最终性能和数据效率。没有早期成功轨迹，HIGhER并不比其构建于之上的DQN更好。在本文中，我们提出了紧密文本回顾性经验。

    Natural language instruction following is paramount to enable collaboration between artificial agents and human beings. Natural language-conditioned reinforcement learning (RL) agents have shown how natural languages' properties, such as compositionality, can provide a strong inductive bias to learn complex policies. Previous architectures like HIGhER combine the benefit of language-conditioning with Hindsight Experience Replay (HER) to deal with sparse rewards environments. Yet, like HER, HIGhER relies on an oracle predicate function to provide a feedback signal highlighting which linguistic description is valid for which state. This reliance on an oracle limits its application. Additionally, HIGhER only leverages the linguistic information contained in successful RL trajectories, thus hurting its final performance and data-efficiency. Without early successful trajectories, HIGhER is no better than DQN upon which it is built. In this paper, we propose the Emergent Textual Hindsight Ex
    

