# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decision-Oriented Recommendation Reranking: An Empirical Study of Jev](https://arxiv.org/abs/2609.40241) | 该实证研究表明，TypeSafe AI提出的决策导向模型Jev在个性化推荐重排序中既能保持与基线模型相当的推荐效果，又比逐点式Qwen重排序器具有更平缓的服务延迟增长，为LLM重排序在质量与效率之间的权衡提供了有价值的替代方案。 |
| [^2] | [Conversational Capture: A Trajectory-Level Framework for Evaluating Generative Engine Optimization in Multi-turn Human-Agent Interaction](https://arxiv.org/abs/2609.40069) | 该论文提出“对话捕获”现象与轨迹级评估框架，指出在多轮人机交互中早期被引用的来源会通过机器侧的历史条件检索和人类侧的针对性追问持续获得引用，从而论证单轮答案可见性不足以评估生成式引擎优化的效果。 |
| [^3] | [Overview of BioASQ 2026: The fourteenth BioASQ Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering](https://arxiv.org/abs/2609.39975) | 第十四届BioASQ挑战赛在CLEF 2026框架下设置了涵盖生物医学问答、多语言临床摘要、嵌套实体关系抽取、心脏病学临床编码和肠-脑信息抽取等六项共享任务，吸引了87支团队提交超过1000个参赛结果，持续推动生物医学语言处理领域的发展。 |
| [^4] | [KUAISHOU Explorer LLM-Rec Challenge 2026: Reasoning Generative Recommendation](https://arxiv.org/abs/2609.39828) | 快手团队基于已大规模部署的OneRec系列模型，探索将物品语义ID与自然语言统一表示并结合思维链（CoT）推理，发起推理式生成式推荐挑战赛，以推动下一代推荐系统的研究。 |
| [^5] | [When the Label Ignores the Request: Auditing Policy-Selected Targets in Synthetic Conversational Music Recommendation](https://arxiv.org/abs/2609.39696) | 论文审计了TalkPlay合成对话音乐推荐基准，发现官方标签在被审计的一半开发轮次中与用户明确点名的歌曲请求相矛盾，揭示了由策略选择的标签作为评估基准的可靠性缺陷。 |
| [^6] | [Working Around the Compute Ceiling: Byte-Exact Memory in Galahad Makes LLM Reading a One-Time Cost LLM Reading a One-Time Cost](https://arxiv.org/abs/2609.39358) | Galahad 通过为 vLLM、SGLang 和 llama.cpp 提供字节精确的记忆层（Taliesin 缓存并复用 KV 状态、Blaise 按需传递文档章节），避免对已读文本的重复计算，使 LLM 的阅读成为一次性成本，从而绕过每 token 算力上限的限制。 |
| [^7] | [Generative End-to-end Ad Retrieval at Douyin](https://arxiv.org/abs/2609.39327) | 抖音提出端到端生成式广告检索框架GEAR，通过正交基码本重参数化方法BasisVQ缓解表示坍缩，并联合优化分词器、生成器与重排序器以同时解决物品碰撞问题。 |
| [^8] | [Residual Trajectory Distillation for Generative Retrieval](https://arxiv.org/abs/2609.39319) | 提出ResTD残差轨迹蒸馏框架，将冻结的残差量化索引器作为过程教师，把索引阶段被丢弃的残差轨迹信息蒸馏到检索训练中，从而提升生成式检索的效果。 |
| [^9] | [Learning Multiresolution Relevance for Hierarchical Generative Retrieval](https://arxiv.org/abs/2609.39312) | 本文提出RARS方法，将分层生成式检索中的多分辨率相关性形式化为由文档级相关性度量诱导的一致条件分布，通过在前缀层面聚合文档相关性并训练前缀条件预测器，实现相关性在SID层级各分支间的显式分配监督。 |
| [^10] | [Argument Structure Prediction in Online Conversations: A Comparative Study of Modeling Paradigms and Task Architectures](https://arxiv.org/abs/2609.39225) | 该论文在严格模式约束下系统比较了监督微调与基于提示的大语言模型在单步及多步任务架构上的论证结构预测表现，并从预测性能、跨领域泛化、模式遵循度和计算效率四个维度进行了统一评估。 |
| [^11] | [O-Funnel: Lossless Structural Capture and Requirement-Driven Extraction from Drifting, Heterogeneous Documents](https://arxiv.org/abs/2609.39209) | O-Funnel 通过将各类异构文档无损转录为统一的类型化树，并以多证据融合的方式按需求定位字段，从根本上解决了传统手工模式在格式与模式漂移下脆弱易失效的问题。 |
| [^12] | [Routing Between Generative and Collaborative User Profiles: A Serving-Time Gate for Controllable Novelty](https://arxiv.org/abs/2609.39043) | 本文提出一种服务时路由门控机制，在生产推荐系统中有选择地为用户调用LLM生成的画像，在可控的相关性损失预算下显著提升推荐新颖性。 |
| [^13] | [Evidence First, Arithmetic Second: A System Report and Failure Analysis for DocSem](https://arxiv.org/abs/2609.39013) | 本文介绍了DocSem共享任务系统EVICALC，其采用“先选证据段落、再由语言模型生成算术表达式并在本地求值”的流程，官方测试联合准确率为8.61%，并通过案例分析指出OCR块合并和页面图像读取能力有限等失败环节。 |
| [^14] | [RouteRec: Behavior-Guided Sparse Routing for Sequential Recommendation](https://arxiv.org/abs/2609.39007) | RouteRec创新性地以观察到的会话行为（交互节奏、物品组关注、重复与延续、流行度倾向）作为混合专家模型的路由准则，在多个尺度上实现行为引导的稀疏计算，显著提升了序列推荐的性能。 |
| [^15] | [When LLM-Inferred User Context Adds Value in Production Streaming Recommendation](https://arxiv.org/abs/2609.38999) | 该论文在一个生产级流媒体推荐平台上系统评估了LLM生成的用户画像与聚合嵌入画像，发现两种表示方式的相对优劣取决于用户的消费模式。 |
| [^16] | [Text-Video Retrieval via Multi-Dimensional Saliency Assessment and Granularity-Aware Query Decomposition](https://arxiv.org/abs/2609.38949) | 提出MMTI方法，通过多维显著性评估筛选关键视觉特征，并结合粒度感知的文本查询分解，实现多粒度的文本-视频语义对齐，从而提升文本-视频检索的准确性。 |
| [^17] | [Breaking News Out of the Filter Bubble: Generative AI Search Diversifies Collective Attention and Raises Shared Information Consumption](https://arxiv.org/abs/2609.38946) | 通过对《华盛顿邮报》37,561名读者的随机现场实验，本研究发现生成式AI搜索并未加剧信息茧房，反而使集体注意力更多元化——既扩大热门话题覆盖面又增加读者间话题重叠，同时推动消费转向冷门话题，其中AI答案是共享信息增长的主要来源。 |
| [^18] | [TRACE: Target-Aware Retrieval, Attributed Evidence, and Contract-Constrained Extraction for LitTraceQA](https://arxiv.org/abs/2609.38861) | TRACE 提出目标感知检索、归因证据定位与契约约束抽取的一体化框架，通过索引近三万篇论文并在表格抽取前预测观测单元，弥合了文献问答中源访问与评分器可见正确性之间的“接地契约鸿沟”。 |
| [^19] | [SkillSeek: Revisiting Agent Skill Retrieval at Marketplace Scale](https://arxiv.org/abs/2609.38822) | SkillSeek 证明在超过 23 万个技能的市场规模下，只需 BM25 加小型交叉编码器的传统两阶段检索方案，就能以几乎零成本达到与昂贵 LLM 介导检索循环相当的技能选择效果。 |
| [^20] | [PatchHolmes: Agentic Patch Retrieval via Listwise Selection](https://arxiv.org/abs/2609.38807) | PatchHolmes提出了一种两阶段补丁检索系统，其智能体通过列表式选择将前100个候选作为一个整体阅读并有选择地检查3至10个提交，在GitHubAD上Recall@1比现有逐点方法提升25%以上，且每个CVE仅需一次智能体对话。 |
| [^21] | [Learning to Route in Visual Space via Multi-Step Embedding Retrieval](https://arxiv.org/abs/2609.38743) | 该论文提出VHOP基准框架和VHOP-Router端到端训练流程（结合监督微调、在线模仿学习与强化学习），将标准嵌入模型改造为能直接在嵌入空间中完成多步视觉导航的检索工具，从而突破LLM智能体视觉搜索中单步检索的性能瓶颈。 |
| [^22] | [Exploring Forum Post Retrieval with Generative Modeling](https://arxiv.org/abs/2609.38646) | 论文针对新界面 Facebook Forum 交互数据稀疏的问题，通过在 Facebook 群组互动数据上训练并复用从跨平台 Feed 数据学到的分层前缀式语义 ID，微调 30 亿参数指令语言模型直接从用户上下文生成语义 ID，实现了生成式的论坛帖子检索推荐。 |
| [^23] | [Component-Aware Feedback for Self-Evolving Programs](https://arxiv.org/abs/2609.38639) | 该论文提出组件感知反馈方法，通过将组件编辑与适应度变化以局部和全局双参考框架记录到归因记忆中，使LLM引导的程序进化搜索更快、更稳定。 |
| [^24] | [Re-ranking and Late Interaction Drive Retrieval Quality: A Controlled Comparison of RAG Strategies for Scientific Question Answering](https://arxiv.org/abs/2609.38473) | 本文通过严格控制变量的方式比较了六种面向科学问答的RAG检索策略，发现重排序和基于ColBERTv2的后期交互检索是提升检索质量的关键因素。 |
| [^25] | [AdaM-Rec: Adaptive Modality Routing for Multimodal Recommendation](https://arxiv.org/abs/2609.38455) | 提出基于大语言模型的AdaM-Rec框架，通过自适应模态路由针对用户特定查询动态校准对文本与视觉证据的依赖，克服了静态模态融合在多样化推荐场景中引入无信息线索、损害推荐质量的问题。 |
| [^26] | [Doc2LoRA Provides Decodable Representations of Scientific Ideas](https://arxiv.org/abs/2609.38374) | 提出 Doc2LoRA 方法，通过超网络将每篇科学论文表示为 LoRA 适配器，使向量空间中的任意点（包括论文混合产生的点）都能解码为可对话的大语言模型，从而实现对科学思想的可解码、可生成式表示。 |
| [^27] | [TAGGRAPH: Tag-Augmented Graphs for Graph Retrieval of Agent Persistent Histories](https://arxiv.org/abs/2609.38353) | 本文提出基于共享5W式对话记忆的受控评估框架，系统比较了标签增强图配置、AdaptiveGraph、BM25与OpenClaw在LLM智能体长期记忆检索上的表现，发现最优检索方法随基准和测试规模而变化，没有单一方法在所有设置下都占优。 |
| [^28] | [Privacy in Personalized AI Is a System Property, Not Just a Model Property](https://arxiv.org/abs/2609.38289) | 该论文提出个性化AI中的隐私应被视为系统级属性而非仅是模型级属性，识别出四个相互关联的隐私风险渠道，并提出了涵盖交互轨迹、内部信息流、间接泄露和隐私-效用权衡的系统级隐私评估四项要求。 |
| [^29] | [MERGE: Multi-LLM Ensemble for Retrieval via Generative Enrichment](https://arxiv.org/abs/2609.37574) | MERGE提出一个两阶段多LLM集成框架，先由三个小型开源LLM独立生成查询扩展候选、再由更大的LLM生成式合成为单一查询，并用基于下游检索性能的自动提示词优化循环取代传统LLM评估器，解决了单一LLM查询增强受限于模型偏见且提示词工程难以扩展的问题。 |
| [^30] | [HELIX: Purified and Unified - Rethinking Feature Interaction and Sequence Modeling for Large-Scale Recommendation](https://arxiv.org/abs/2609.37183) | HELIX提出了一种纯化统一的大规模推荐架构，通过将序列检索与特征交互相交织并强制单向信息流，实现对特征交互与序列建模两个轴的联合扩展，从而获得更优的扩展律斜率。 |
| [^31] | [Large Knowledge Model: From Papers to a Scientific Reasoning Landscape](https://arxiv.org/abs/2609.27297) | 本文提出大知识模型（LKM），将论文表示为基于原文来源的推理图，构建包含问题、工作流和证据三个视图的科学推理图景，使科学文献成为可计算访问的共享推理资源，从而支持大规模利用文献中的科学推理过程。 |
| [^32] | [Intrinsic Sequence-Likelihood Confidence in Retrieval-Dominated Extractive QA: Two Pre-Specified Negatives, and What They Do and Do Not Attribute](https://arxiv.org/abs/2609.19942) | 该研究通过预先注册的评估标准证明，在检索已能恢复92-99.8%最优性能的抽取式问答场景中，模型的内在序列似然置信度无论作为蒸馏触发器还是路由弃答策略的控制信号均告失效。 |
| [^33] | [Two-Sided State-Space Models for Sequential Recommendation with Non-Random Multimodal Review Feedback](https://arxiv.org/abs/2609.00165) | 提出双边状态空间模型TS-SSM，将非随机生成的多模态评论反馈同时融入用户状态与商品状态的动态演化过程，从而提升事件条件下的序列推荐效果。 |
| [^34] | [Do AI chatbots find what experts would? Effects of model, user role, and sample size on study retrieval for medical questions](https://arxiv.org/abs/2608.13786) | 本研究系统评估了三种主流AI聊天机器人在医学问题检索中的表现，发现其检索质量受模型类型、用户角色和样本量显著影响，且与专家标准存在差距。 |
| [^35] | [omni-macos: On-Device Omni-Modal Search on Apple Silicon](https://arxiv.org/abs/2608.05543) | omni-macos在苹果设备上实现全模态本地搜索，通过内存预算管理和高效编码策略，确保所有数据不出设备。 |
| [^36] | [Scoring a Set, Not Summing Passage Scores: Effective and Efficient Set Retrieval](https://arxiv.org/abs/2607.05712) | 该论文提出用基于能量的查询—集合兼容性分数直接对完整候选证据集进行排序，而非独立打分或顺序累加段落分数，从而在多跳问答中实现有效且高效的集合检索。 |
| [^37] | [Interactor: Agentic RL oriented Iterative Creation for Ad Description Generation in Sponsored Search](https://arxiv.org/abs/2606.15911) | 提出Interactor框架，通过智能体强化学习让生成模型与多个生成式奖励模型进行多轮交互迭代，自动生成融入世界知识且与落地页一致的高质量付费搜索广告描述。 |
| [^38] | [GrepSeek: Training Search Agents for Direct Corpus Interaction](https://arxiv.org/abs/2605.29307) | GrepSeek提出了一种让LLM搜索智能体直接通过shell命令与语料库交互寻找证据的新范式，采用“Tutor/Planner生成可靠搜索轨迹初始化+GRPO强化学习优化”的两阶段训练方法，并通过语义保持的执行优化实现大规模语料库上的高效检索。 |
| [^39] | [Agent-Facing Information Design in LLM Tool Registries: A Preregistered Test of Rhetoric, Position and Structure](https://arxiv.org/abs/2605.23916) | 研究通过预注册实验发现，LLM智能体的工具选择会被工具描述中的推销性赞美语言显著影响（可提升约43个百分点选中率，甚至选到无法胜任任务的工具），列表排序和结构化字段也会强烈左右选择，因此建议注册表以结构化字段而非营销文本来呈现工具限制。 |
| [^40] | [SOLAR: SVD-Optimized Lifelong Attention for Recommendation](https://arxiv.org/abs/2603.02561) | 该论文提出了SVD-Attention——一种通过将低秩嵌入分解为r个主成分、在紧凑空间中计算候选物品与兴趣得分并保留softmax机制的新型注意力方法，并在此基础上构建了面向终身推荐的高效集合感知框架SOLAR。 |
| [^41] | [Life-Bench: A Benchmark and Knowledge Graph Framework for Multimodal Personalization Beyond Concept Recognition](https://arxiv.org/abs/2602.19001) | 本文提出了Life-Bench——一个包含11,800多个问答对、按概念识别、事件理解和聚合推理三个层次组织的合成多模态个性化基准测试，并配套提出个人知识图谱框架LifeGraph，通过结构化检索与按需访问源视觉证据，在事件理解和聚合推理任务上表现出显著优势。 |

# 详细

[^1]: 面向决策的推荐重排序：关于Jev的实证研究

    Decision-Oriented Recommendation Reranking: An Empirical Study of Jev

    [https://arxiv.org/abs/2609.40241](https://arxiv.org/abs/2609.40241)

    该实证研究表明，TypeSafe AI提出的决策导向模型Jev在个性化推荐重排序中既能保持与基线模型相当的推荐效果，又比逐点式Qwen重排序器具有更平缓的服务延迟增长，为LLM重排序在质量与效率之间的权衡提供了有价值的替代方案。

    

    大型语言模型（LLMs）在推荐重排序任务中展现出了潜力，但其使用在推荐质量与服务效率之间引入了一个重要的权衡。我们研究了当重排序任务本质上是针对预定义候选项目的结构化选择时，面向决策的模型能否提供一种有用的替代方案。具体而言，我们对Jev（由TypeSafe AI定义为一种"System One Model"）在个性化推荐重排序中的表现进行了受控实证研究，并将其与推荐专用模型以及逐点式和列表式的Qwen重排序器在多个Amazon Reviews领域和不同候选集规模下进行比较，同时评估了推荐效果与实际观测到的服务延迟。我们的结果表明，Jev相对于所评估的基线模型保持了较强的推荐效果，同时相比逐点式Qwen重排序器表现出明显更为平缓的延迟增长，尽管……

    arXiv:2609.40241v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown promise for recommendation reranking, but their use introduces an important tradeoff between recommendation quality and serving efficiency. We investigate whether a decision-oriented model provides a useful alternative when the reranking task is fundamentally a structured choice among predefined candidate items. Specifically, we conduct a controlled empirical study of Jev, described by TypeSafe AI as a ``System One Model,'' for personalized recommendation reranking and compare it with recommendation-specific models and pointwise and listwise Qwen rerankers across multiple Amazon Reviews domains and candidate-set sizes, evaluating both recommendation effectiveness and observed serving latency. Our results show that Jev maintains strong recommendation effectiveness relative to the evaluated baselines while exhibiting substantially more gradual latency growth than the pointwise Qwen rerankers, altho
    
[^2]: 对话捕获：一种用于评估多轮人机交互中生成式引擎优化的轨迹级框架

    Conversational Capture: A Trajectory-Level Framework for Evaluating Generative Engine Optimization in Multi-turn Human-Agent Interaction

    [https://arxiv.org/abs/2609.40069](https://arxiv.org/abs/2609.40069)

    该论文提出“对话捕获”现象与轨迹级评估框架，指出在多轮人机交互中早期被引用的来源会通过机器侧的历史条件检索和人类侧的针对性追问持续获得引用，从而论证单轮答案可见性不足以评估生成式引擎优化的效果。

    

    生成式引擎优化（GEO）通过塑造内容来提高其被基于检索增强大语言模型构建的答案引擎引用的可能性。GEO 通常被评估为一种单轮属性：对于一个固定的查询，评估者测量某个来源在单个答案中的可见性。我们认为，单个答案是一个不充分的分析单元。人机信息搜寻形成一个闭环：智能体的答案改变用户的信念，进而影响下一个问题，而下一个问题又反过来决定智能体检索什么。我们提出了“对话捕获”（conversational capture）这一现象，即早期被引用的来源在后续显著更可能再次被引用。捕获通过两个通道运作：机器侧通道——基于历史的条件检索，以及人类侧通道——针对被捕获来源的追问。我们将该交互形式化为一个两层闭环系统，并推导出轨迹级构造：累积……

    arXiv:2609.40069v1 Announce Type: cross  Abstract: Generative Engine Optimization (GEO) shapes content to increase its likelihood of being cited by answer engines built on retrieval-augmented large language models. GEO is typically evaluated as a single-turn property: for a fixed query, an evaluator measures a source's visibility in one answer. We argue that the single answer is an inadequate unit of analysis. Human-agent information seeking forms a closed loop: the agent's answer changes the user's beliefs and therefore the next question, which in turn determines what the agent retrieves. We introduce conversational capture, a phenomenon in which a source cited early becomes substantially more likely to be cited again. Capture operates through a machine-side channel, history-conditioned retrieval, and a human-side channel, follow-up questions directed toward the captured source. We formalize the interaction as a two-layer closed-loop system and derive trajectory-level constructs: cumu
    
[^3]: BioASQ 2026概述：第十四届大规模生物医学语义索引与问答BioASQ挑战赛

    Overview of BioASQ 2026: The fourteenth BioASQ Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering

    [https://arxiv.org/abs/2609.39975](https://arxiv.org/abs/2609.39975)

    第十四届BioASQ挑战赛在CLEF 2026框架下设置了涵盖生物医学问答、多语言临床摘要、嵌套实体关系抽取、心脏病学临床编码和肠-脑信息抽取等六项共享任务，吸引了87支团队提交超过1000个参赛结果，持续推动生物医学语言处理领域的发展。

    

    本文概述了在2026年评测论坛会议与实验室（CLEF）框架下举办的第十四届BioASQ挑战赛。BioASQ是一项国际性挑战赛系列，旨在推动生物医学语言处理任务的进展，涵盖语义索引、信息抽取、问答和文本摘要等多个方向。2026年的BioASQ包含六项共享任务：a) Task 14b，关于生物医学语义问答；b) Task Synergy14，关于新兴生物医学主题的问答；c) Task MultiClinSum-2，关于多语言临床摘要生成；d) Task BioNNE-R，关于俄语和英语中嵌套命名实体间关系的抽取；e) Task ELCardioCC，关于心脏病学领域的临床编码；f) Task GutBrainIE，关于肠-脑相互作用的信息抽取。在这六项任务中，共有87支不同的团队参赛，累计提交超过1000个系统运行结果。与往届一样……

    arXiv:2609.39975v1 Announce Type: cross  Abstract: This paper presents an overview of the fourteenth edition of the BioASQ challenge, organized in the context of the Conference and Labs of the Evaluation Forum (CLEF) 2026. BioASQ is an international challenge series that supports progress in biomedical language processing tasks ranging from semantic indexing and information extraction to question answering and summarization. In 2026, BioASQ included six shared tasks: a) Task 14b on biomedical semantic question answering. b) Task Synergy14 on question answering for developing biomedical top- ics. c) Task MultiClinSum-2 on multilingual clinical summarization. d) Task BioNNE-R on extracting relations between nested named entities in Russian and English. e) Task ELCardioCC on clinical coding in cardiology. f) Task GutBrainIE on gut-brain interplay information extrac- tion. Across these six tasks, 87 distinct teams participated, submitting more than 1000 runs overall. As in previous edition
    
[^4]: 快手Explorer LLM-Rec挑战赛2026：推理式生成式推荐

    KUAISHOU Explorer LLM-Rec Challenge 2026: Reasoning Generative Recommendation

    [https://arxiv.org/abs/2609.39828](https://arxiv.org/abs/2609.39828)

    快手团队基于已大规模部署的OneRec系列模型，探索将物品语义ID与自然语言统一表示并结合思维链（CoT）推理，发起推理式生成式推荐挑战赛，以推动下一代推荐系统的研究。

    

    生成式推荐在工业界和学术界引起了越来越多的关注，其目标是构建更智能的系统以打造下一代推荐器。在大型语言模型的显著发展浪潮下，我们的团队开发了基于Semantic ID的OneRec/OneRec-V2。这些模型已在生产环境中广泛部署，并展示了自回归下一物品预测范式在工业推荐系统中的扩展潜力。在OneRec成功的基础上，我们进一步探索了一系列模型，包括OneRec-Think、OpenOneRec和OneReason，这些模型将物品语义ID与自然语言连接在统一的表示空间中，并试图释放自然语言思维链（CoT）推理在推荐中的潜力。然而，我们的初步工作发现，引入推理CoT并不总是能够提升推荐性能。为解决这一问题……（原文摘要在此处截断）

    arXiv:2609.39828v1 Announce Type: new  Abstract: Generative recommendation, has been attracted a surge of attentions in industrial and academic research community, towards to build more smart system to build next-generation recommender. Under the significant developing wave of large language model, our team have been developed Semantic ID based OneRec/OneRec-V2. These models have been widely deployed in production and demonstrate the scaling potential of the autoregressive next-item prediction paradigm for industrial recommender systems. Building on the success of OneRec, we further explored a series of models, including OneRec-Think, OpenOneRec, and OneReason, that connect item Semantic IDs with natural language in a unified representation space and seek to unlock the potential of natural-language chain-of-thought (CoT) reasoning for recommendation. However, our preliminary works found that introducing reasoning CoT does not always improve the recommendation performance. To address th
    
[^5]: 当标签忽视请求时：审计合成对话式音乐推荐中由策略选择的目标

    When the Label Ignores the Request: Auditing Policy-Selected Targets in Synthetic Conversational Music Recommendation

    [https://arxiv.org/abs/2609.39696](https://arxiv.org/abs/2609.39696)

    论文审计了TalkPlay合成对话音乐推荐基准，发现官方标签在被审计的一半开发轮次中与用户明确点名的歌曲请求相矛盾，揭示了由策略选择的标签作为评估基准的可靠性缺陷。

    

    由大语言模型流水线生成的合成对话如今已成为完整的对话式推荐基准：一个LLM听者与一个LLM推荐者对话，而对话中接下来记录的曲目就成为每一轮的官方标签。这些由策略选择的标签使大规模评估具有可复现性，但它们只是模拟用户所请求内容的代理。我们审计了标签与请求能够直接比较的唯一场景：用户按名称明确要求某首歌曲的轮次。在RecSys Challenge 2026 TalkPlay基准中，仅使用可见对话和目录元数据，我们发现官方标签在被审计的一半开发轮次中与用户的确切歌曲请求相矛盾。这个问题的影响超出了单一基准的范围：点名所需曲目是真实音乐搜索中的主要意图，而已部署的系统避免用替代品去替换用户明确点名的曲目，其前提是这样会降低用户满意度。

    arXiv:2609.39696v1 Announce Type: new  Abstract: Synthetic dialogues generated by LLM pipelines now serve as complete conversational-recommendation benchmarks: an LLM listener talks to an LLM recommender, and the track logged next in the conversation becomes the official label for each turn. These policy-selected labels make large-scale evaluation reproducible, but they are proxies for what the simulated user asked. We audit the one place where label and request are directly comparable: turns where the user asks for an exact song by name. In the RecSys Challenge 2026 TalkPlay benchmark, using visible dialogue and catalog metadata alone, we find that the official label contradicts the user's exact-song request in half of the audited development turns. This matters beyond one benchmark: naming the desired item is the dominant intent in real music search, where deployed systems avoid substituting an alternative for an exactly named item, on the premise that it costs satisfaction. A small 
    
[^6]: 绕过算力天花板：Galahad 中的字节精确记忆使 LLM 阅读成为一次性成本

    Working Around the Compute Ceiling: Byte-Exact Memory in Galahad Makes LLM Reading a One-Time Cost LLM Reading a One-Time Cost

    [https://arxiv.org/abs/2609.39358](https://arxiv.org/abs/2609.39358)

    Galahad 通过为 vLLM、SGLang 和 llama.cpp 提供字节精确的记忆层（Taliesin 缓存并复用 KV 状态、Blaise 按需传递文档章节），避免对已读文本的重复计算，使 LLM 的阅读成为一次性成本，从而绕过每 token 算力上限的限制。

    

    Transformer 语言模型在每个 token 上可执行的算力是有限的，Infosys 前首席执行官 Vishal Sikka 近期的研究认为，这一上限限制了模型能够执行或验证的任务范围（arXiv:2507.07505）。我们要问的是：在这一天花板之下的算力预算中，有多少被花费在模型已经完成过的工作上。推理服务在请求之间是无状态的：当模型针对同一份文档回答第二个问题时，它会从第一个 token 开始重新计算该文档的注意力状态。在七个真实世界数据集上，98.7% 的提示 token 都是模型已经读过的文本。我们提出了 Galahad——一个面向 vLLM、SGLang 和 llama.cpp 的记忆层，它使这种阅读成为一次性成本。Taliesin 为一段文本保存模型的键值（KV）状态，并在下一个包含相同字节的请求中直接加载，而无需重新计算。Blaise 则保存文档本身，仅将问题所需的章节传递给模型。在一个召回测试中……

    arXiv:2609.39358v1 Announce Type: cross  Abstract: A transformer language model performs a bounded amount of computation per token, and recent work by Vishal Sikka, former CEO of Infosys, argues that this bound limits which tasks a model can carry out or verify (arXiv:2507.07505). We ask how much of the budget beneath that ceiling is spent on work the model has already done. Serving is stateless across requests: a model that answers a second question about a document recomputes the document's attention state from the first token. On seven real-world datasets, 98.7% of prompt tokens were text the model had already read. We present Galahad, a memory layer for vLLM, SGLang and llama.cpp that makes this reading a one-time cost. Taliesin saves the model's key-value (KV) state for a block of text and loads it on the next request that contains the same bytes, instead of recomputing it. Blaise keeps the documents themselves and passes the model only the section a question needs. On a recall te
    
[^7]: 抖音的生成式端到端广告检索

    Generative End-to-end Ad Retrieval at Douyin

    [https://arxiv.org/abs/2609.39327](https://arxiv.org/abs/2609.39327)

    抖音提出端到端生成式广告检索框架GEAR，通过正交基码本重参数化方法BasisVQ缓解表示坍缩，并联合优化分词器、生成器与重排序器以同时解决物品碰撞问题。

    

    生成式检索将推荐问题重新表述为离散物品token的生成。然而，将该范式扩展到真实的工业级推荐系统时会暴露出两个关键瓶颈：1）表示坍缩，即物品分词器（tokenizer）在持续的分布漂移下收敛到退化的结果，从根本上阻碍了稳定的端到端适配；2）物品碰撞，即庞大的候选池导致不同物品共享相同的token序列，损害了最终检索的精度。至关重要的是，这些瓶颈是内在耦合的：扩大码本容量以缓解碰撞不可避免地会加剧表示坍缩。为了同时解决这两个问题，我们提出了GEAR，一个联合优化分词器、生成器和重排序器的端到端框架。为了缓解表示坍缩，我们引入了BasisVQ，它通过正交基对码本进行重新参数化，以实现全局梯度共享……

    arXiv:2609.39327v1 Announce Type: new  Abstract: Generative retrieval reformulates recommendation as the generation of discrete item tokens. However, scaling this paradigm to real-world recommender systems reveals two critical bottlenecks: 1) Representation collapse, where the item tokenizer converges to degenerate results under continuous distribution shifts, fundamentally hindering stable end-to-end adaptation. 2) Item collisions, where the massive candidate pool causes distinct items to share identical token sequences, compromising the final retrieval precision. Crucially, these bottlenecks are inherently coupled: expanding codebook capacity to mitigate collisions inevitably exacerbates collapse. To address them simultaneously, we propose GEAR, an end-to-end framework that jointly optimizes the tokenizer, generator, and reranker. To mitigate representation collapse, we introduce BasisVQ, which re-parameterizes the codebook via an orthogonal basis to enable global gradient sharing an
    
[^8]: 面向生成式检索的残差轨迹蒸馏

    Residual Trajectory Distillation for Generative Retrieval

    [https://arxiv.org/abs/2609.39319](https://arxiv.org/abs/2609.39319)

    提出ResTD残差轨迹蒸馏框架，将冻结的残差量化索引器作为过程教师，把索引阶段被丢弃的残差轨迹信息蒸馏到检索训练中，从而提升生成式检索的效果。

    

    生成式检索已成为一种通用的检索范式，它使用离散的语义ID（SID）来表示物品，并通过自回归的标识符生成来进行检索。当SID采用残差量化（RQ）构建时，标准的检索训练只监督被选中的编码，而丢弃了产生这些编码的残差轨迹。然而，同一个硬编码可能源于对不同候选码字的竞争性偏好，同时残差轨迹中还包含着关于后续量化决策的信息。因此，硬SID监督将不同的量化行为坍缩为相同的目标，使得索引阶段可用的信息在检索训练中未被利用。我们提出了ResTD，一个残差轨迹蒸馏框架，将这种被丢弃的索引信息迁移到检索训练中。该方法将冻结的RQ索引器视为过程教师，对其残差信息进行蒸馏。

    arXiv:2609.39319v1 Announce Type: new  Abstract: Generative retrieval has emerged as a general retrieval paradigm, representing items with discrete Semantic IDs (SIDs) and retrieving them through autoregressive identifier generation. When SIDs are constructed with residual quantization (RQ), standard retrieval training supervises only the selected codes and discards the residual trajectories that produce them. The same hard code can nevertheless arise from different preferences over competing codewords, while the residual trajectory also contains information about subsequent quantization decisions. As a result, hard SID supervision collapses distinct quantization behaviors into identical targets and leaves information available during indexing unused in retrieval training. We introduce ResTD, a Residual Trajectory Distillation framework that transfers this discarded indexing information into retrieval training. Treating the frozen RQ indexer as a process teacher, it distills residual-i
    
[^9]: 面向分层生成式检索的多分辨率相关性学习

    Learning Multiresolution Relevance for Hierarchical Generative Retrieval

    [https://arxiv.org/abs/2609.39312](https://arxiv.org/abs/2609.39312)

    本文提出RARS方法，将分层生成式检索中的多分辨率相关性形式化为由文档级相关性度量诱导的一致条件分布，通过在前缀层面聚合文档相关性并训练前缀条件预测器，实现相关性在SID层级各分支间的显式分配监督。

    

    基于语义标识符（SID）的生成式检索在文档层次结构上进行逐级决策。对于同一查询，相关文档可能在较粗粒度层级共享前缀，而在更细的层级上产生分叉，且分支模式因查询而异。这些路径揭示了相关性如何在逐级细化过程中分布，然而标准的全SID监督方式将其视为相互独立的训练目标。为了显式化这种相关性分配，我们将多分辨率相关性形式化为由单一文档级相关性度量在SID层次结构上诱导出的一致条件分布。我们提出了RARS（分辨率对齐相关性监督），利用由此得到的细化层级分布来监督共享的查询表示。RARS在前缀上聚合文档相关性，并训练一个以前缀为条件的预测器，以在兄弟分支之间分配相关性。

    arXiv:2609.39312v1 Announce Type: new  Abstract: Generative retrieval with semantic identifiers (SIDs) makes successive decisions over a document hierarchy. Relevant documents for the same query may share coarse prefixes and diverge at finer depths, with branching patterns varying across queries. These paths reveal how relevance is distributed across successive refinements, yet standard full-SID supervision treats them as separate training targets. To make this allocation explicit, we formulate multiresolution relevance as consistent conditional distributions induced by a single document-level relevance measure across the SID hierarchy. We introduce \textbf{RARS}, \textbf{R}esolution-\textbf{A}ligned \textbf{R}elevance \textbf{S}upervision, which uses the resulting refinement-level distributions to supervise a shared query representation. RARS aggregates document relevance over prefixes and trains a prefix-conditioned predictor to allocate relevance among sibling branches. All relevanc
    
[^10]: 在线对话中的论证结构预测：建模范式与任务架构的比较研究

    Argument Structure Prediction in Online Conversations: A Comparative Study of Modeling Paradigms and Task Architectures

    [https://arxiv.org/abs/2609.39225](https://arxiv.org/abs/2609.39225)

    该论文在严格模式约束下系统比较了监督微调与基于提示的大语言模型在单步及多步任务架构上的论证结构预测表现，并从预测性能、跨领域泛化、模式遵循度和计算效率四个维度进行了统一评估。

    

    论证结构预测（ASP）通过识别论证单元及其关系，从话语中构建完整的论证结构。尽管近期研究探索了多种方法——包括统一的神经模型、多步骤流水线以及基于提示的大语言模型（LLM）——但这些方法之间的相对权衡仍未得到充分研究，尤其是在对话场景中。我们在严格的模式约束下对ASP进行了系统性评估，比较了监督微调与基于提示的大语言模型在单步和多步任务架构上的表现，从对话输入端到端地生成完整的论证结构。我们在三个不同的对话语料库上进行基准测试，这些语料库由推理锚定理论改编为双极论证结构。在统一的评估框架下，我们评估了预测性能、跨领域泛化能力、模式遵循度以及计算效率。我们的结果表明，ASP仍然……

    arXiv:2609.39225v1 Announce Type: new  Abstract: Argument structure prediction (ASP) constructs complete argument structures from discourse by identifying argumentative units and their relations. While recent work has explored diverse approaches---including unified neural models, multi-step pipelines, and prompt-based large language models (LLMs)---their relative trade-offs remain under-explored, particularly in dialogical settings.   We present a systematic evaluation of ASP under strict schema constraints, comparing supervised fine-tuning and prompt-based LLMs across single- and multi-step task architectures, generating complete argument structures from dialogical input end-to-end. We benchmark them on three diverse dialogical corpora adapted from Inference Anchoring Theory into bipolar argument structures. Under a shared evaluation framework, we assess predictive performance, cross-domain generalization, schema compliance, and computational efficiency. Our results show that ASP rema
    
[^11]: O-Funnel：面向漂移异构文档的无损结构捕获与需求驱动抽取

    O-Funnel: Lossless Structural Capture and Requirement-Driven Extraction from Drifting, Heterogeneous Documents

    [https://arxiv.org/abs/2609.39209](https://arxiv.org/abs/2609.39209)

    O-Funnel 通过将各类异构文档无损转录为统一的类型化树，并以多证据融合的方式按需求定位字段，从根本上解决了传统手工模式在格式与模式漂移下脆弱易失效的问题。

    

    从以多种格式到达且模式不断漂移的文档中抽取一组固定字段，通常依靠手工编写的字节模式，但一旦键名被重命名、值被重新格式化，或相似值率先出现，这些模式便会失效。我们认为其根源在于结构问题：单一模式必须同时完成对值的描述和在其上下文中的定位。O-Funnel 将这两者分离。它将任意 XML、JSON、CSV、HTML 或键值文本文档转录为一棵基于五种构造子构建的类型化树，并由一个预言机把关，拒绝任何无法重构其源文档的捕获。每个所需字段以树自身的术语进行声明，并通过融合相互独立的证据（键、路径、值形态、同义词、键拼写、记录邻域、值轮廓）进行定位，使支持度最高的节点胜出，而对缺失字段则会附带原因进行报告。未被任何需求认领的数据成为残留，由漏斗将其回溯至需求，以学习新的键……

    arXiv:2609.39209v1 Announce Type: cross  Abstract: Pulling a fixed set of fields out of documents that arrive in many formats and under drifting schemas is usually done with hand-written byte patterns, which break whenever a key is renamed, a value is reformatted, or a lookalike value appears first. We argue the cause is structural: one pattern must both describe the value and locate it among its surroundings. O-Funnel separates the two. It transcribes any XML, JSON, CSV, HTML or key-value text document into one typed tree over five constructors, gated by an oracle that rejects any capture that does not reconstruct its source. Each needed field is declared in the tree's own terms and located by fusing independent evidence (key, path, value shape, synonym, key spelling, record neighborhood, value profile), so the best-supported node wins and a missing field is reported with a reason. Data no requirement claims becomes residue that a funnel traces back to the requirements to learn new ke
    
[^12]: 在生成式与协同式用户画像之间路由：一种实现可控新颖性的服务时门控机制

    Routing Between Generative and Collaborative User Profiles: A Serving-Time Gate for Controllable Novelty

    [https://arxiv.org/abs/2609.39043](https://arxiv.org/abs/2609.39043)

    本文提出一种服务时路由门控机制，在生产推荐系统中有选择地为用户调用LLM生成的画像，在可控的相关性损失预算下显著提升推荐新颖性。

    

    大型语言模型（LLMs）能够为推荐系统构建丰富的语义用户画像，但生成此类画像的成本较高，且并不一定适合统一部署。我们研究了能否在生产级推荐流水线中有选择地调用LLM生成的用户画像。基于一个涵盖电影、电视节目和体育内容的真实流媒体数据集，我们训练了一个服务时路由门控，将每个用户分配给协同式序列推荐模型或由LLM生成画像驱动的推荐模型。该门控仅使用服务时特征，并学习识别哪些用户可通过基于画像的路由在保持排序相关性的同时提升Novelty@10。路由阈值控制用户被发送到生成式模型的积极程度，呈现出可调节的新颖性—相关性权衡。在总体NDCG损失预算为5%的条件下，学习到的门控使Novelty@10提升了6.5%。

    arXiv:2609.39043v1 Announce Type: new  Abstract: Large language models (LLMs) enable rich semantic user profiles for recommendation, but such profiles are more expensive to generate and are not necessarily desirable to deploy uniformly. We study whether LLM-generated profiles can instead be invoked selectively within a production recommendation pipeline. Using a real-world streaming dataset covering movies, TV shows, and sports content, we train a serving-time routing gate that assigns each user to either a collaborative sequential recommendation model or a recommendation model driven by an LLM-generated profile. The gate uses only serving-time features and learns to identify users for whom profile-based routing can increase Novelty@10 while preserving ranking relevance. A routing threshold controls how aggressively users are sent to the generative model, exposing a tunable novelty--relevance trade-off. At an overall NDCG-loss budget of 5\%, the learned gate increases Novelty@10 by 6.5
    
[^13]: 证据优先，算术其次：DocSem任务的系统报告与失败分析

    Evidence First, Arithmetic Second: A System Report and Failure Analysis for DocSem

    [https://arxiv.org/abs/2609.39013](https://arxiv.org/abs/2609.39013)

    本文介绍了DocSem共享任务系统EVICALC，其采用“先选证据段落、再由语言模型生成算术表达式并在本地求值”的流程，官方测试联合准确率为8.61%，并通过案例分析指出OCR块合并和页面图像读取能力有限等失败环节。

    

    EVICALC是我们为DocSem共享任务开发的系统，在官方最终测试评估中对1,730个任务取得了8.61%的联合准确率。该系统读取PDF文件、选择一段文字，让语言模型编写一个算术表达式，并在本地代码中对该表达式进行求值。保存的中间结果支持对失败案例的检查。在另一次公开验证集运行中，系统达到了92.17%的答案准确率和1.00的证据F1分数。由于配置和指标不同，这些分数不构成受控比较。我们的事后人工分析是描述性的：在一个检查的案例中，光学字符识别（OCR）和块分组将相关段落合并到了另一个块中，导致系统从无关文本中得出答案。一项对100份文档阅读页面图像的探索性研究仅返回了22份文档的证据标识符。这些描述性发现促使进一步评估；它们并未确定整体得分的原因。

    arXiv:2609.39013v1 Announce Type: new  Abstract: EVICALC, our system for the DocSem shared task, achieved 8.61% joint accuracy on 1,730 tasks in the official final test evaluation. It reads a PDF, selects a passage, asks a language model to write an arithmetic expression, and evaluates that expression in local code. Saved intermediate results support inspection of failures. A separate public-validation run achieved 92.17% answer accuracy and 1.00 evidence F1. The configurations and metrics differ, so these scores are not a controlled comparison. Our manual, post-hoc analysis is descriptive: in one inspected case, optical character recognition (OCR) and block grouping merged the relevant passage into another block, and the system answered from unrelated text. An exploratory study of reading page images on 100 documents returned evidence identifiers for only 22 documents. These descriptive findings motivate further evaluation; they do not establish the causes of the overall score.
    
[^14]: RouteRec：面向序列推荐的行为引导稀疏路由

    RouteRec: Behavior-Guided Sparse Routing for Sequential Recommendation

    [https://arxiv.org/abs/2609.39007](https://arxiv.org/abs/2609.39007)

    RouteRec创新性地以观察到的会话行为（交互节奏、物品组关注、重复与延续、流行度倾向）作为混合专家模型的路由准则，在多个尺度上实现行为引导的稀疏计算，显著提升了序列推荐的性能。

    

    会话化的交互历史中蕴含着能够改进序列推荐的行为模式。然而，现有模型通过相同的参数化模块处理所有会话，而忽略了它们之间的行为差异。混合专家模型实现了条件计算，但并未解决应由什么来引导专家分配的问题。我们提出了RouteRec，一种以观察到的会话行为作为路由准则的序列推荐模型。RouteRec从会话化历史中总结出四种类型的行为证据：交互节奏、物品组关注度、重复与延续性、以及流行度倾向，并利用这些线索在宏观、中观和微观三个尺度上进行计算路由。由行为线索导出的分数首先选择专家组；在每个被选中的组内，当前的骨干状态再进一步细化专家选择。在六个公开数据集和18个数据集-指标组合上，RouteRec在12个组合中排名第一，在3个组合中排名第二。

    arXiv:2609.39007v1 Announce Type: new  Abstract: Sessionized interaction histories contain behavioral patterns that can improve sequential recommendation. However, existing models process all sessions through the same parameterized blocks, regardless of their behavioral differences. Mixture of Experts (MoE) enables conditional computation, but it leaves open what should guide expert allocation. We propose RouteRec, a sequential recommender that uses observed session behavior as the routing criterion. RouteRec summarizes four types of behavioral evidence from sessionized histories: interaction tempo, item-group focus, repetition and carryover, and popularity tendency. It uses these cues to route computation at macro, mid, and micro scopes. Cue-derived scores first select expert groups; within each selected group, the current backbone state then refines expert selection. Across six public datasets and 18 dataset-metric combinations, RouteRec ranks first in 12 and second in three, yieldin
    
[^15]: 当LLM推断的用户上下文在生产级流媒体推荐中带来附加价值时

    When LLM-Inferred User Context Adds Value in Production Streaming Recommendation

    [https://arxiv.org/abs/2609.38999](https://arxiv.org/abs/2609.38999)

    该论文在一个生产级流媒体推荐平台上系统评估了LLM生成的用户画像与聚合嵌入画像，发现两种表示方式的相对优劣取决于用户的消费模式。

    

    推荐系统中的上下文信息正从静态的、预定义的变量转向从用户行为中推断出的潜在表示。大语言模型通过将非结构化的交互历史呈现为自然语言摘要来支持这一转变，从而产生一种主题化的用户上下文，该上下文可被编码并用于替代聚合画像。然而，这类生成的画像在何种条件下优于聚合嵌入，目前的相关刻画仍然有限，且主要停留在领域层面。我们在一个生产级流媒体平台上对语义用户画像策略进行了评估，针对全目录进行排序。评估涵盖了2*2的设计空间，交叉了表示类型（聚合型或LLM生成型）与上下文范围（整体历史或注意力融合的短期与长期上下文）。结果显示，两种表示类型的相对优劣取决于用户的消费模式。聚合画像在……（原文摘要在此处截断）

    arXiv:2609.38999v1 Announce Type: new  Abstract: Contextual information in recommender systems is shifting from static, predefined variables toward latent representations inferred from behavior. Large language models support this shift by rendering an unstructured interaction history as a natural-language summary, which yields a thematic user context that can be encoded and used in place of an aggregate profile. The conditions under which such generated profiles outperform aggregate embeddings have received limited characterization mainly at the domain level. We evaluate semantic user-profiling strategies on a production streaming platform, ranking against the full catalog. The evaluation covers a 2*2 design space crossing representation type (aggregate or LLM-generated) with contextual scope (holistic history or attention-fused short-term and long-term contexts). The relative ordering of the two representation types is conditional on the user's consumption regime. Aggregate profiles a
    
[^16]: 基于多维显著性评估与粒度感知查询分解的文本-视频检索

    Text-Video Retrieval via Multi-Dimensional Saliency Assessment and Granularity-Aware Query Decomposition

    [https://arxiv.org/abs/2609.38949](https://arxiv.org/abs/2609.38949)

    提出MMTI方法，通过多维显著性评估筛选关键视觉特征，并结合粒度感知的文本查询分解，实现多粒度的文本-视频语义对齐，从而提升文本-视频检索的准确性。

    

    文本-视频检索旨在通过学习联合嵌入空间来连接视觉与文本模态，已成为多模态智能中的一项关键任务。尽管已有大量工作致力于缓解视觉冗余，但先前的方法通常依赖单一方面的准则来评估视觉重要性，忽视了视频多方面的时空特性。此外，将文本编码为单一全局嵌入以与视频对齐，会把时间事件和空间实体压缩到统一的表示空间中，进一步加剧了跨模态的不对齐问题。为解决这些问题，我们提出了MMTI，该方法联合缓解视觉冗余并实现多粒度的文本-视频交互，以实现精确的多粒度语义对齐。具体而言，关键特征选择（KFS）机制通过联合评估多维显著性和……（摘要内容在此处被截断）

    arXiv:2609.38949v1 Announce Type: new  Abstract: Text-video retrieval, which aims to bridge visual and textual modalities by learning a joint embedding space, has become a crucial task in multimodal intelligence. Despite extensive efforts to mitigate visual redundancy, previous methods typically rely on a single-aspect criterion to assess visual importance, overlooking the multifaceted spatiotemporal nature of video. In addition, encoding text into a single global embedding to align with videos compresses temporal events and spatial entities into a unified representation space, further aggravating cross-modal misalignment. To address these issues, we propose MMTI, a method that jointly mitigates visual redundancy and enables multi-grained text-video interaction to achieve accurate multi-grained semantic alignment. Specifically, a key feature selection (KFS) mechanism adaptively identifies and aggregates informative frames and patches by jointly evaluating multi-dimensional saliency and
    
[^17]: 让突发新闻冲出过滤气泡：生成式AI搜索使集体注意力更多元化并提升共享信息消费

    Breaking News Out of the Filter Bubble: Generative AI Search Diversifies Collective Attention and Raises Shared Information Consumption

    [https://arxiv.org/abs/2609.38946](https://arxiv.org/abs/2609.38946)

    通过对《华盛顿邮报》37,561名读者的随机现场实验，本研究发现生成式AI搜索并未加剧信息茧房，反而使集体注意力更多元化——既扩大热门话题覆盖面又增加读者间话题重叠，同时推动消费转向冷门话题，其中AI答案是共享信息增长的主要来源。

    

    生成式AI搜索和AI概览正在改变人们获取信息与新闻的方式，这重新引发了人们对于读者将接触到更窄范围话题、彼此共同点更少的担忧。我们通过对《华盛顿邮报》37,561名读者开展的随机现场实验来检验这些担忧。两组读者搜索同一个文章档案库，但实验组读者还会在常规搜索结果上方收到带有文章引用的AI答案。通过衡量所展示答案和所打开文章的消费情况，我们发现AI搜索扩大了热门话题的覆盖范围，并增加了读者之间话题消费的重叠度。与此同时，无论是读者个体内部还是整体受众层面，消费都变得更不集中，并转向较不受欢迎的话题。AI答案贡献了共享信息增长的大部分，它无需读者点击文章即可传递信息，并将曝光范围扩展到读者实际打开的文章之外。被引用的文章也对此作出了贡献（原文在此处截断）。

    arXiv:2609.38946v1 Announce Type: cross  Abstract: Generative AI search and AI overviews are transforming access to information and news, renewing concerns that readers will encounter a narrower range of topics and have less in common. We examine these concerns via a randomized field experiment with 37,561 readers at The Washington Post. Both groups searched the same archive, but treatment readers also received AI answers with article citations above conventional results. Measuring consumption across displayed answers and opened articles, we find that AI search expands the reach of widely read topics and increases overlap in readers' topic consumption. At the same time, consumption becomes less concentrated and shifts toward less-popular topics, both within readers and across the audience. AI answers account for most of the increase in shared information, delivering it without requiring article clicks and broadening exposure beyond the articles readers open. Cited articles also contrib
    
[^18]: TRACE：面向LitTraceQA的目标感知检索、归因证据与契约约束抽取

    TRACE: Target-Aware Retrieval, Attributed Evidence, and Contract-Constrained Extraction for LitTraceQA

    [https://arxiv.org/abs/2609.38861](https://arxiv.org/abs/2609.38861)

    TRACE 提出目标感知检索、归因证据定位与契约约束抽取的一体化框架，通过索引近三万篇论文并在表格抽取前预测观测单元，弥合了文献问答中源访问与评分器可见正确性之间的“接地契约鸿沟”。

    

    从文献中找到一篇相关论文，并不等同于从中产出可验证的答案。LitTraceQA 要求规范化的论文标识符、页级或对象级的精确证据，以及与评测器相匹配的类型化答案。我们将“源访问”与“评分器可见正确性”之间的分离称为接地契约鸿沟（grounding contract gap）。TRACE——目标感知检索、归因证据与契约约束抽取——通过目标分组检索、独立的类型化证据定位、多模态表格抽取、模式驱动的表格构建以及故障关闭式验证来弥合这一鸿沟。该系统通过段落、对象、别名、引用和稠密表示对 27,487 篇论文进行索引，同时在每个信号背后保留问题的目标。对于表格，TRACE 在抽取数值之前先预测观测单元，并使用与评测器兼容的键归一化方式来组装行。经审计的精选洁净赛道产出物在官方上得分为 0.760613。

    arXiv:2609.38861v1 Announce Type: new  Abstract: Finding a relevant paper is not the same as producing a verifiable answer from it. LitTraceQA requires canonical paper identifiers, exact evidence at the page or object level, and typed answers that match the evaluator. We call the separation between source access and scorer-visible correctness the grounding contract gap. TRACE - Target-Aware Retrieval, Attributed Evidence, and Contract-Constrained Extraction - addresses this gap with target-grouped retrieval, independent typed evidence localization, multimodal table extraction, schema-driven table construction, and fail-closed validation. It indexes 27,487 papers through passage, object, alias, citation, and dense representations while retaining the question target behind each signal. For tables, TRACE predicts the observation unit before extracting values and assembles rows with evaluator-compatible key normalization. Our audited selected clean-track artifact scores 0.760613 on the off
    
[^19]: SkillSeek：在市场规模下重新审视智能体技能检索

    SkillSeek: Revisiting Agent Skill Retrieval at Marketplace Scale

    [https://arxiv.org/abs/2609.38822](https://arxiv.org/abs/2609.38822)

    SkillSeek 证明在超过 23 万个技能的市场规模下，只需 BM25 加小型交叉编码器的传统两阶段检索方案，就能以几乎零成本达到与昂贵 LLM 介导检索循环相当的技能选择效果。

    

    Anthropic 的 Agent Skills 将可复用的程序性知识以 SKILL.md 目录的形式打包供 LLM 智能体使用，开源聚合的技能数量已超过 23 万个，这使得“选择”而非“编写”成为了瓶颈。文献中的通行做法是将选择工作外包给智能体本身：由 LLM 介导的检索循环在智能体的决策循环内重写查询并细化候选结果，每个任务都要消耗 LLM token。我们提出 SkillSeek，一个开源的两阶段技能检索器，由标准信息检索配方构建（BGE-base 双编码器后接小型交叉编码器，通过 MCP 暴露服务）。在 89 任务的 SkillsBench 基准上，跨越池、骨干模型和方法组成的 4×11 组合网格，SkillSeek 以基本零额外成本达到了与 Liu 等人的 LLM 介导循环相当的观测性能：仅朴素 bm25 就在四个设置中的三个上取得等于或高于其精细化循环的通过率，小型交叉编码器则覆盖了剩余的……（原文截断）

    arXiv:2609.38822v1 Announce Type: cross  Abstract: Anthropic's Agent Skills package reusable procedural know-how for an LLM agent into SKILL.md directories, and open-source aggregations have grown past 230,000 skills, making selection rather than authoring the bottleneck. The standing answer in the literature outsources selection to the agent itself: an LLM-mediated retrieval loop that rewrites queries and refines candidates inside the agent's decision loop, paying LLM tokens on every task. We present SkillSeek, an open-source two-stage skill retriever built from the standard IR recipe (a BGE-base bi-encoder feeding a small cross-encoder, exposed over MCP). Across a $4 \times 11$ grid of pool, backbone, and method on the 89-task SkillsBench benchmark, SkillSeek reaches observed parity with the LLM-mediated loop of Liu et al. at essentially no extra cost: plain bm25 alone records a pass rate at or above their refined loop on three of four settings, and a small cross-encoder covers the r
    
[^20]: PatchHolmes：基于列表式选择的智能体补丁检索

    PatchHolmes: Agentic Patch Retrieval via Listwise Selection

    [https://arxiv.org/abs/2609.38807](https://arxiv.org/abs/2609.38807)

    PatchHolmes提出了一种两阶段补丁检索系统，其智能体通过列表式选择将前100个候选作为一个整体阅读并有选择地检查3至10个提交，在GitHubAD上Recall@1比现有逐点方法提升25%以上，且每个CVE仅需一次智能体对话。

    

    补丁检索，即找到修复已知漏洞的对应提交，是漏洞管理工作流的基础任务，然而主要安全通告数据库中60%至63%的CVE缺少补丁链接。我们提出了PatchHolmes，一个两阶段补丁检索系统，它将混合式第一阶段检索器与智能体式第二阶段检查循环相结合。与独立对每个候选进行评分的逐点式先前工作不同，第二阶段智能体以列表方式阅读前100个候选：它一次性看到完整的候选列表，并通过四个有预算限制的工具选择性地阅读3至10个提交，然后提交单一的最佳提交。在GitHubAD数据集上，PatchHolmes比逐点二分类器Favia高出25.34%的Recall@1，比检索加思维链基线IRCoT高出31.40%，且每个CVE仅需一次智能体对话，而Favia需要十次；在候选集相同的情况下，该智能体比直接采用检索器排名第一的候选额外提升了27.32%的Recall@1，且同一智能体……

    arXiv:2609.38807v1 Announce Type: new  Abstract: Patch retrieval, the task of finding the commit that fixes a known vulnerability, is the foundation of vulnerability management workflows, yet 60% to 63% of CVEs in the major advisory databases lack a patch link. We present PatchHolmes, a two-phase patch retrieval system that pairs a hybrid first-stage retriever with an agentic second-stage inspection loop. Unlike pointwise prior work that scores each candidate independently, the Phase 2 agent reads the top-100 listwise: it sees the full candidate list at once and selectively reads 3 to 10 commits through four budgeted tools before submitting a single best commit. On GitHubAD, PatchHolmes beats the pointwise binary classifier Favia by 25.34% Recall@1 and the retrieve-and-CoT baseline IRCoT by 31.40%, at one agent conversation per CVE versus Favia's ten; with the candidate set held identical, the agent adds 27.32% Recall@1 over taking the retriever's top candidate, and the same agent, tra
    
[^21]: 通过多步嵌入检索学习在视觉空间中进行路由

    Learning to Route in Visual Space via Multi-Step Embedding Retrieval

    [https://arxiv.org/abs/2609.38743](https://arxiv.org/abs/2609.38743)

    该论文提出VHOP基准框架和VHOP-Router端到端训练流程（结合监督微调、在线模仿学习与强化学习），将标准嵌入模型改造为能直接在嵌入空间中完成多步视觉导航的检索工具，从而突破LLM智能体视觉搜索中单步检索的性能瓶颈。

    

    LLM智能体依赖检索工具来访问外部知识，然而视觉智能体搜索仍然严重受限于标准的单步检索器。在现有流程中，智能体必须为每个中间步骤发出文本查询，当视觉线索难以用文字描述、或检索器无法在其前列结果中呈现必要的中间证据时，搜索就会陷入困境。我们假设，将跨越整个嵌入空间的多步导航直接交由检索工具完成，可以解决这一性能瓶颈。为了系统地研究这一问题，我们提出了VHOP——一个灵活的数据生成框架与基准，包含五个核心难度级别，同时测试视觉匹配与搜索规划能力。利用该框架，我们开发了VHOP-Router——一个端到端的训练流程，结合监督微调、在线模仿学习与强化学习，将标准嵌入模型转变为……

    arXiv:2609.38743v1 Announce Type: new  Abstract: LLM agents rely on retrieval tools to access external knowledge, yet visual agentic search remains severely bottlenecked by standard single-step retrievers. In current pipelines, the agent must issue text queries for every intermediate step, struggling when visual clues are difficult to describe or when the retriever fails to surface necessary intermediate evidence within its top results. We hypothesize that offloading multi-step navigation across the entire embedding space directly to the retrieval tool resolves this performance bottleneck. To study this systematically, we introduce VHOP, a flexible data generation framework and benchmark with five core difficulty levels testing both visual matching and search planning. Using this framework, we develop VHOP-Router, an end-to-end training pipeline---combining supervised fine-tuning, online imitation learning, and reinforcement learning---that transforms a standard embedding model into an
    
[^22]: 探索基于生成式建模的论坛帖子检索

    Exploring Forum Post Retrieval with Generative Modeling

    [https://arxiv.org/abs/2609.38646](https://arxiv.org/abs/2609.38646)

    论文针对新界面 Facebook Forum 交互数据稀疏的问题，通过在 Facebook 群组互动数据上训练并复用从跨平台 Feed 数据学到的分层前缀式语义 ID，微调 30 亿参数指令语言模型直接从用户上下文生成语义 ID，实现了生成式的论坛帖子检索推荐。

    

    生成式推荐（GR）建立在生成式模型在语言和视觉领域成功的基础上，已成为基于嵌入的检索的一种替代方案。我们正在 Facebook Forum（一个面向 Facebook 群组中重度用户的独立应用）上探索 GR。由于 Forum 是一个全新的界面，其自身的交互数据过于稀疏，无法从零开始训练 GR 模型。我们通过两个维度的迁移来解决这一问题：一方面，我们在更广泛的 Facebook 群组互动语料库上训练，而不仅仅是 Forum 会话数据；另一方面，我们复用从跨平台 Facebook Feed 数据中学习到的分层、基于前缀的语义 ID（SID），而不是为 Forum 单独训练分词器。随后，我们对一个 30 亿参数的指令微调语言模型进行监督微调，使其能够直接从用户上下文生成 SID。我们系统地消融了实践中最关键的设计选择，包括 SID 的构建方式、用户历史记录的组成与长度等。

    arXiv:2609.38646v1 Announce Type: new  Abstract: Generative recommendation (GR) has emerged as an alternative to embedding-based retrieval, building on the success of generative models in language and vision. We are exploring GR on Facebook Forum, a standalone application for medium-to-heavy users of Facebook Groups. Because Forum is a new surface, its own interaction data are too sparse to train a GR model from scratch. We address this with transfer along two axes: we train on a broader corpus of Facebook Groups engagements rather than Forum sessions alone, and we reuse hierarchical, prefix-based semantic IDs (SIDs) learned from cross-platform Facebook Feed data instead of fitting a Forum-specific tokenizer. A 3B-parameter instruction-tuned language model is then supervised-fine-tuned to generate SIDs directly from user context. We systematically ablate the design choices that matter most in practice, including SID construction, the composition and length of user history, and the incl
    
[^23]: 面向自进化程序的组件感知反馈

    Component-Aware Feedback for Self-Evolving Programs

    [https://arxiv.org/abs/2609.38639](https://arxiv.org/abs/2609.38639)

    该论文提出组件感知反馈方法，通过将组件编辑与适应度变化以局部和全局双参考框架记录到归因记忆中，使LLM引导的程序进化搜索更快、更稳定。

    

    LLM引导的进化搜索能够发现复杂程序，但现有方法大多只保存候选程序和适应度分数，而丢弃了“哪些组件编辑产生了哪些适应度指标变化”的信息。现有方法迫使变异器LLM从杂乱的历史记录中推断先前编辑的效果，使得程序搜索缓慢且不稳定。对于使用本地部署LLM来演化多组件系统的场景尤其如此。我们提出了组件感知反馈，它将每个被评估的程序与其父代进行比较，识别发生变化的组件，并将这些组件及相应的指标差异记录到归因记忆中，供后续变异读取。该记忆在两个参考框架中保存每次变化：相对于其来源父代的局部框架，以及相对于种子程序的全局框架，从而同时展示每次变化的即时效果和自种子程序以来的累积进展。我们在LLM重排序这一多（组件系统）任务上对此进行了研究。

    arXiv:2609.38639v1 Announce Type: new  Abstract: LLM-guided evolutionary search can discover complex programs, but existing methods mostly only save candidate programs and fitness scores while discarding which component edits produced which fitness metric changes. Existing methods force the mutator LLM to infer the effect of prior edits from cluttered histories, making program search slow and unstable. This is especially true for locally servable LLMs to evolve multi-component systems. We introduce component-aware feedback, which compares each evaluated program with its parent, identifies the components that changed, and logs them with the associated metric differences into an attribution memory that later mutations read. The memory keeps each change in two reference frames, local against the parent it came from and global against the seed program, which shows both the immediate effect of a change and the cumulative progress made since the seed. We study this on LLM reranking, a multi-
    
[^24]: 重排序与后期交互驱动检索质量：面向科学问答的RAG策略受控比较

    Re-ranking and Late Interaction Drive Retrieval Quality: A Controlled Comparison of RAG Strategies for Scientific Question Answering

    [https://arxiv.org/abs/2609.38473](https://arxiv.org/abs/2609.38473)

    本文通过严格控制变量的方式比较了六种面向科学问答的RAG检索策略，发现重排序和基于ColBERTv2的后期交互检索是提升检索质量的关键因素。

    

    摘要：检索增强生成（RAG）如今已成为将大语言模型（LLM）与外部知识相结合的标准方法，然而检索流水线的设计空间十分庞大，各变体之间的权衡尚未被充分理解，尤其是在现实规模下的领域特定语料库上。在本工作中，我们对用于科学问答的六种检索策略进行了受控比较：(i) 经典的top-k稠密检索，(ii) 基于LLM的查询改写，(iii) 查询改写后接基于LLM的重排序，(iv) 通过倒数排序融合（RRF）实现的多查询融合，(v) 一种代理式工具调用流水线，其中生成器自行决定是否进行检索，以及(vi) 采用ColBERTv2的后期交互检索。所有六种流水线均共享相同的生成器（Meta-Llama/Llama-3.1-8B-Instruct）、提示词和评估协议；五个单向量流水线还额外共享SPECTER2嵌入和Chroma向量存储；

    arXiv:2609.38473v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) is now the standard way to ground Large Language Models (LLMs) in external knowledge, yet the design space of retrieval pipelines is large and the trade-offs between variants are not well understood, especially on domain-specific corpora at realistic scale. In this work, we present a controlled comparison of six retrieval strategies for scientific question answering: (i) classic top-k dense retrieval, (ii) LLM-based query rephrasing, (iii) query rephrasing followed by LLM-based reranking, (iv) multi-query fusion via Reciprocal Rank Fusion (RRF), (v) an agentic tool-call pipeline in which the generator decides for itself whether to retrieve, and (vi) late-interaction retrieval with ColBERTv2. All six pipelines share the same generator (Meta-Llama/Llama-3.1-8B-Instruct), prompt, and evaluation protocol; the five single-vector pipelines additionally share SPECTER2 embeddings and a Chroma vector store; 
    
[^25]: AdaM-Rec：面向多模态推荐的自适应模态路由

    AdaM-Rec: Adaptive Modality Routing for Multimodal Recommendation

    [https://arxiv.org/abs/2609.38455](https://arxiv.org/abs/2609.38455)

    提出基于大语言模型的AdaM-Rec框架，通过自适应模态路由针对用户特定查询动态校准对文本与视觉证据的依赖，克服了静态模态融合在多样化推荐场景中引入无信息线索、损害推荐质量的问题。

    

    尽管近期的多模态推荐系统已经证明了融合视觉与文本信息以提升下游性能的有效性，但大多数现有方法依赖于静态的模态融合，即假设文本与视觉信号的相对重要性在各个推荐场景中保持稳定。这种设计可能无法充分考虑到不同推荐请求之间的一个重要差异：某些查询需要细粒度的视觉线索，而另一些查询则更适合通过文本或功能语义来满足；在这种情况下，不加区分的模态融合会引入无信息量的线索，从而损害推荐质量。为解决这一问题，我们提出了AdaM-Rec，一个基于大语言模型（LLM）的多模态推荐自适应模态路由框架，能够针对用户特定查询动态校准对文本和多模态证据的依赖程度。该框架建立在物品和用户的结构化自然语言表示之上。

    arXiv:2609.38455v1 Announce Type: new  Abstract: While recent multimodal recommender systems have demonstrated the effectiveness of incorporating visual and textual information to improve downstream performance, most existing methods rely on static modality fusion, assuming that the relative importance of textual and visual signals remains stable across recommendation scenarios. This design may not fully account for an important variation across recommendation requests: some queries require fine-grained visual cues, whereas others are better served by textual or functional semantics, in which case indiscriminate modality fusion brings in uninformative cues and impairs recommendation quality. To address this, we propose AdaM-Rec, an LLM-based framework for adaptive modality routing in multimodal recommendation, which enables dynamic calibration of reliance on textual and multimodal evidence for user-specific queries. Built on structured natural-language representations of items and user
    
[^26]: Doc2LoRA 为科学思想提供可解码的表示

    Doc2LoRA Provides Decodable Representations of Scientific Ideas

    [https://arxiv.org/abs/2609.38374](https://arxiv.org/abs/2609.38374)

    提出 Doc2LoRA 方法，通过超网络将每篇科学论文表示为 LoRA 适配器，使向量空间中的任意点（包括论文混合产生的点）都能解码为可对话的大语言模型，从而实现对科学思想的可解码、可生成式表示。

    

    将科学论文表示为空间中的点，使我们能够搜索相似论文，并探究各领域之间如何相互关联以及如何推动创新。超越搜索之外，论文的向量空间还孕育了生成能力：通过简单的向量运算混合论文可以创造新的点，这反映了组合式创新——即现有思想重新组合成新思想的过程。然而，混合点通常代表一种尚未有论文实现的想法，且附近没有论文可以帮助识别这一想法。我们提出用 Doc-to-LoRA 超网络生成的 LoRA 适配器来表示每篇论文。因此，空间中的每个点（包括混合点）都代表一个可通过自然语言进行提问和指令交互的大语言模型（LLM）。在美国物理学会（APS）的论文上，我们指示每个子领域平均值处的 LLM 用几个词命名该领域，所获得的标签比五个基线方法的标签更接近官方名称。

    arXiv:2609.38374v1 Announce Type: new  Abstract: Representing scientific papers as points in a space lets us search for similar papers and inquire about how fields relate to one another and drive innovation. Beyond search, the vector space of papers invites generation: mixing papers through simple vector operations creates new points, mirroring combinatorial novelty, the recombination of existing ideas into new ones. However, a mixed point often represents an idea no paper has yet realized, with no papers nearby to identify the idea. We propose representing each paper by a LoRA adapter generated by the Doc-to-LoRA hypernetwork. Every point in the space, including mixtures, thus represents a large language model (LLM) open to questions and instructions in natural language. On papers from the American Physical Society (APS), we instruct the LLM at the average of each subfield to name the field in a few words and obtain labels closer to the official names than the labels of five baselines
    
[^27]: TAGGRAPH：用于智能体持久历史的图检索之标签增强图

    TAGGRAPH: Tag-Augmented Graphs for Graph Retrieval of Agent Persistent Histories

    [https://arxiv.org/abs/2609.38353](https://arxiv.org/abs/2609.38353)

    本文提出基于共享5W式对话记忆的受控评估框架，系统比较了标签增强图配置、AdaptiveGraph、BM25与OpenClaw在LLM智能体长期记忆检索上的表现，发现最优检索方法随基准和测试规模而变化，没有单一方法在所有设置下都占优。

    

    长期记忆使LLM智能体能够回忆过去的交互并在多个会话之间保持一致性，但由于记忆系统在表示、索引、检索和评估方面往往各不相同，因此难以相互比较。我们提出了一个基于共享的5W式对话记忆的受控评估框架。局部化的图配置在共同的基础图上进行遍历；AdaptiveGraph增加了时序边并使用个性化PageRank进行扩散。我们还在相同的抽取笔记上评估了BM25，并将OpenClaw作为使用原始输入的外部参照。检索排序在不同的记忆设置之间存在差异。在LongMemEval-S上，AdaptiveGraph是最强的图配置，MRR达到0.844，但BM25达到0.867，OpenClaw达到0.880。在ATANT Core上，局部化图遍历优于扩散和BM25，而在压力测试轮次中BM25领先。在测试范围内缩减LongMemEval-S的规模并未重现ATANT中的扩散惩罚，但最小的测试……（原文在此处截断）

    arXiv:2609.38353v1 Announce Type: cross  Abstract: Long-term memory lets LLM agents recall past interactions and remain consistent across sessions, but memory systems are hard to compare because they often vary in representation, indexing, retrieval, and evaluation. We present a controlled evaluation framework based on shared 5W-style conversational memories. Localized graph configurations traverse a common base graph; AdaptiveGraph adds chronological edges and Personalized PageRank diffusion. We also evaluate BM25 over the same extracted notes and OpenClaw as a raw-input external reference. Retrieval rankings vary across memory settings. On LongMemEval-S, AdaptiveGraph is the strongest graph configuration at 0.844 MRR, but BM25 reaches 0.867 and OpenClaw 0.880. On ATANT Core, localized graph traversal outperforms diffusion and BM25, whereas BM25 leads the stress rounds. Reducing LongMemEval-S within the tested range does not reproduce the ATANT diffusion penalty, but the smallest test
    
[^28]: 个性化AI中的隐私是系统属性，而不仅仅是模型属性

    Privacy in Personalized AI Is a System Property, Not Just a Model Property

    [https://arxiv.org/abs/2609.38289](https://arxiv.org/abs/2609.38289)

    该论文提出个性化AI中的隐私应被视为系统级属性而非仅是模型级属性，识别出四个相互关联的隐私风险渠道，并提出了涵盖交互轨迹、内部信息流、间接泄露和隐私-效用权衡的系统级隐私评估四项要求。

    

    在个性化AI应用（如对话助手和推荐系统）中，用户交互的对象并非孤立的模型，而是更广泛的系统——这些系统跨组件、跨时间地访问、推断和复用用户信息。虽然这种用户信息的使用对个性化至关重要，但也引发了重要的隐私问题。在本文中，我们论证了仅进行模型级或组件级的独立分析可能无法捕获此类系统中产生的所有隐私风险，因此需要采用系统级的隐私视角。我们区分并分析了个性化AI中四个相互关联的隐私风险渠道，进而提出了系统级隐私评估的四项要求，涵盖交互轨迹、内部信息流、间接泄露以及隐私-效用权衡。我们主张将这些要求系统地纳入个性化AI的隐私审计之中。

    arXiv:2609.38289v1 Announce Type: cross  Abstract: In personalized AI applications, such as conversational assistants and recommender systems, users interact not with models in isolation but with broader systems that access, infer, and reuse user information across components and over time. While such use of user information is integral to personalization, it also raises important privacy questions. In this paper, we argue that individual model- or component-level analyses may not capture all privacy risks arising in such systems, motivating a system-level perspective on privacy. We distinguish and analyze four interconnected privacy-risk channels in personalized AI, and subsequently propose four requirements for system-level privacy evaluation, covering interaction trajectories, internal information flows, indirect leakage, and the privacy-utility trade-off. We argue for their systematic incorporation into privacy audits of personalized AI.
    
[^29]: MERGE：基于生成式增强的多大语言模型集成检索框架

    MERGE: Multi-LLM Ensemble for Retrieval via Generative Enrichment

    [https://arxiv.org/abs/2609.37574](https://arxiv.org/abs/2609.37574)

    MERGE提出一个两阶段多LLM集成框架，先由三个小型开源LLM独立生成查询扩展候选、再由更大的LLM生成式合成为单一查询，并用基于下游检索性能的自动提示词优化循环取代传统LLM评估器，解决了单一LLM查询增强受限于模型偏见且提示词工程难以扩展的问题。

    

    大语言模型（LLM）越来越多地被用于信息检索（IR）中的用户查询增强，使BM25等标准检索器能够弥合查询与目标语料库之间的词汇鸿沟。然而，任何单一LLM都受限于其训练数据和架构偏见，且其增强行为依赖于手工设计的提示词——这些提示词必须针对每个新模型重新设计，是一个昂贵且难以扩展的过程。我们提出了MERGE（Multi-LLM Ensemble for Retrieval via Generative Enrichment，基于生成式增强的检索多LLM集成框架），这是一个两阶段框架：三个异构的7-8B开源LLM独立生成候选查询扩展，随后由一个更大的LLM将它们生成式地合成为单一查询。为了使提示词工程在整个模型集成中具备可扩展性，我们在两个阶段中都集成了基于任务的自动提示词优化（APO）循环。与使用LLM评估器来判断候选结果的APO方法不同，我们的循环根据每个候选的下游检索表现进行评分。

    arXiv:2609.37574v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly used to enrich user queries in information retrieval (IR) so that a standard retriever such as BM25 can bridge vocabulary gaps with the target corpus. Any single LLM, however, is limited by its training data and architectural biases, and its enrichment behavior depends on hand-crafted prompts that must be re-engineered for each new model -- an expensive and poorly scalable process. We present MERGE (Multi-LLM Ensemble for Retrieval via Generative Enrichment), a two-stage framework: three heterogeneous 7-8B open-source LLMs independently produce candidate expansions, and a larger LLM generatively synthesizes them into a single query. To make prompt engineering scalable across the ensemble, we integrate a task-grounded Automatic Prompt Optimization (APO) loop into both stages. Unlike APO methods that judge candidates with an LLM evaluator, our loop scores each candidate by its downstream retr
    
[^30]: HELIX：纯化与统一——重新思考大规模推荐中的特征交互与序列建模

    HELIX: Purified and Unified - Rethinking Feature Interaction and Sequence Modeling for Large-Scale Recommendation

    [https://arxiv.org/abs/2609.37183](https://arxiv.org/abs/2609.37183)

    HELIX提出了一种纯化统一的大规模推荐架构，通过将序列检索与特征交互相交织并强制单向信息流，实现对特征交互与序列建模两个轴的联合扩展，从而获得更优的扩展律斜率。

    

    工业推荐排序模型通常沿两个建模轴进行扩展：一是对异构的用户、物品、上下文及交叉特征进行特征交互，二是对长而信息丰富的多类型用户行为历史进行序列建模。我们发现单独扩展其中任何一种能力都是不够的，因为二者各自表现出有限的扩展上限和次优的扩展律斜率。我们推测，要获得更有利的扩展律斜率，需要对两个轴进行联合扩展。为验证这一观点，我们提出了HELIX，一种面向大规模推荐的纯化且统一的架构。HELIX将序列检索与特征交互交织进行，同时强制从可复用的序列状态到候选条件混合标记的单向信息流。这一设计在保持两个建模轴之间跨深度通信的同时，使用户侧的序列计算可以摊销，从而实现了序列建模的灵活且非对称的扩展。

    arXiv:2609.37183v1 Announce Type: new  Abstract: Industrial recommendation ranking models typically scale along two modeling axes: feature interaction over heterogeneous user, item, context, and cross features, and sequence modeling over long, informative, and multi-type user behavior histories. We find that scaling either capability in isolation is insufficient, as each exhibits a limited scaling ceiling and a suboptimal scaling-law slope. We conjecture that achieving a more favorable scaling-law slope requires jointly scaling both axes. To support this, we present HELIX, a purified and unified architecture for large-scale recommendation. HELIX interleaves sequence retrieval and feature interaction while enforcing one-way information flow from reusable sequence states to candidate-conditioned mix-tokens. This design preserves cross-depth communication between the two modeling axes while keeping user-side sequence computation amortizable, enabling flexible and asymmetric scaling of seq
    
[^31]: 大知识模型：从论文到科学推理图景

    Large Knowledge Model: From Papers to a Scientific Reasoning Landscape

    [https://arxiv.org/abs/2609.27297](https://arxiv.org/abs/2609.27297)

    本文提出大知识模型（LKM），将论文表示为基于原文来源的推理图，构建包含问题、工作流和证据三个视图的科学推理图景，使科学文献成为可计算访问的共享推理资源，从而支持大规模利用文献中的科学推理过程。

    

    积累的科学知识之所以能推动科学探究，是因为已有研究发现可以帮助研究者选择新问题、设计研究方案并解释结果。要在大规模上实现这一价值，需要获取连接研究问题、科学程序、结论和证据的推理过程。我们提出了大知识模型，这是一种科学知识基础设施，能将科学文献转化为共享的、可计算访问的推理资源。LKM 将论文表示为基于原文来源的推理图，将结构化遍历与对同一对象的语义检索相结合，并对齐跨论文的相关问题、论断和推理链。这种表示构成了一个包含三个相互关联视图的科学推理图景：组织研究问题和开放方向的问题图景、揭示可复用科学程序的工作流图景，以及连接（摘要在此处截断）……的图景

    arXiv:2609.27297v1 Announce Type: new  Abstract: Accumulated scientific knowledge advances inquiry when prior findings help researchers choose new questions, design investigations, and interpret results. Realizing this value at scale requires access to the reasoning that connects research problems, scientific procedures, conclusions, and evidence. We introduce the Large Knowledge Model (LKM), a scientific knowledge infrastructure that transforms the literature into a shared, computationally accessible reasoning resource. LKM represents papers as source-grounded reasoning graphs, couples structural traversal with semantic retrieval over the same objects, and aligns related questions, claims, and reasoning chains across papers. This representation forms a Scientific Reasoning Landscape with three connected views: a Question Landscape that organizes research problems and open directions, a Workflow Landscape that exposes reusable scientific procedures, and an Evidence Landscape that conne
    
[^32]: 检索主导的抽取式问答中的内在序列似然置信度：两个预先设定的否定结果，及其能归因与不能归因的对象

    Intrinsic Sequence-Likelihood Confidence in Retrieval-Dominated Extractive QA: Two Pre-Specified Negatives, and What They Do and Do Not Attribute

    [https://arxiv.org/abs/2609.19942](https://arxiv.org/abs/2609.19942)

    该研究通过预先注册的评估标准证明，在检索已能恢复92-99.8%最优性能的抽取式问答场景中，模型的内在序列似然置信度无论作为蒸馏触发器还是路由弃答策略的控制信号均告失效。

    

    在抽取式文档问答中——其问题由包含答案的段落生成，因此检索即可恢复任何模式组合所能达到性能的92-99.8%（无论其绝对准确率如何）——基于置信度的机制几乎没有提升空间。在专业领域语料库上微调开源语言模型后，模型自身的置信度是一个颇具吸引力的控制信号：它可用于决定哪些查询值得进一步适配、哪些答案值得信赖。我们在实验开始前预先固定的评估标准下，对四个7-9B模型家族进行了评估（其领域适配使闭卷F1最多提升+0.03），结果两种用途均告失败：在预先设定的三步迁移预算下，蒸馏触发器在全部四个模型家族上失效，单模型试点中的路由与弃答策略同样失败。在我们测试的每一个正确性标准下，仅检索即可恢复最优组合准确率的92-99.8%，留下的……

    arXiv:2609.19942v1 Announce Type: new  Abstract: In extractive document question answering whose questions were generated from the passages that contain their answers -- so that retrieval recovers 92-99.8% of what any mode combination could reach, whatever its absolute accuracy -- confidence-driven mechanisms have little to gain. Fine-tuning an open language model on a specialized domain corpus yields a model whose own confidence is a tempting control signal: it could decide which queries warrant further adaptation, and which answers to trust. We evaluate both uses under criteria fixed before the runs were executed, across four 7-9B model families whose adaptation moved closed-book F1 by at most +0.03, and both fail: a distillation trigger on all four families, under its pre-specified three-step transfer budget, and a routing-and-abstention policy in its single-model pilot. Retrieval alone recovers 92-99.8% of best-case combined accuracy under every correctness criterion we test, leavi
    
[^33]: 面向具有非随机多模态评论反馈的序列推荐的双边状态空间模型

    Two-Sided State-Space Models for Sequential Recommendation with Non-Random Multimodal Review Feedback

    [https://arxiv.org/abs/2609.00165](https://arxiv.org/abs/2609.00165)

    提出双边状态空间模型TS-SSM，将非随机生成的多模态评论反馈同时融入用户状态与商品状态的动态演化过程，从而提升事件条件下的序列推荐效果。

    

    双边数字平台本质上是动态的：用户偏好会发生转移，商品热度会不断演变，而评论既反映也驱动着这些变化。然而，大多数序列推荐系统将评论视为更新用户状态的被动信号，忽视了两个重要方面。首先，评论的生成是非随机的，它取决于用户和商品双方不断演变的潜在状态。其次，评论能够重塑商品状态、在相关商品之间产生溢出效应，并影响用户未来的决策。为填补这些空白，我们提出了一种用于事件条件序列推荐的双边状态空间模型（TS-SSM）。TS-SSM由三个组件构成：（1）一个模态非随机缺失融合模块，用于编码评论内容以及蕴含信息的观测模式；（2）带有时间变化和局部图消息传递的用户状态演化机制，利用相关商品状态来细化用户偏好；（3）商品状态演化机制（摘要在此处截断）。

    arXiv:2609.00165v1 Announce Type: new  Abstract: Two-sided digital platforms are inherently dynamic: user preferences shift, item popularity evolves, and reviews both reflect and drive these changes. Yet most sequential recommendation systems treat reviews as passive signals for updating user states, leaving two aspects underexplored. First, review generation is nonrandom, depending on evolving latent states of both users and items. Second, reviews can reshape item states, induce spillover across related items, and influence future user decisions. To address these gaps, we propose a two-sided state-space model (TS-SSM) for event-conditioned sequential recommendation. TS-SSM consists of three components: (1) a modality-missing-not-at-random fusion module that encodes review content and informative observation patterns; (2) user-state evolution with temporal variation and local graph message passing that uses related item states to refine user preferences; and (3) item-state evolution wi
    
[^34]: AI聊天机器人能否找到专家会找到的内容？模型、用户角色和样本量对医学问题研究检索的影响

    Do AI chatbots find what experts would? Effects of model, user role, and sample size on study retrieval for medical questions

    [https://arxiv.org/abs/2608.13786](https://arxiv.org/abs/2608.13786)

    本研究系统评估了三种主流AI聊天机器人在医学问题检索中的表现，发现其检索质量受模型类型、用户角色和样本量显著影响，且与专家标准存在差距。

    

    大语言模型（LLM）聊天机器人越来越多地被用于回答临床问题，并引用相关临床研究作为支持。先前的研究主要集中在引用伪造上，而对检索到的研究质量及其选择驱动因素的评估存在空白。在本研究中，我们评估了三个通用LLM聊天机器人：Claude Sonnet 5、Gemini 3.1 Pro和ChatGPT GPT-5.5。我们使用改编自2026年Cochrane系统评价数据库第6期和第7期中20个评价问题的临床问题对模型进行提示，模拟患者、临床医生和证据综合研究人员角色。每个聊天机器人在每种用户角色下进行四次独立重复查询，共产生720个响应。每个聊天机器人被要求用主要临床引用支持其答案，我们将其与Cochrane评价的纳入和排除研究集进行基准比较。平均而言，一个聊天机器人响应检索...

    arXiv:2608.13786v1 Announce Type: cross  Abstract: Large language model (LLM) chatbots are increasingly used to answer clinical questions with citations to relevant clinical studies. Prior research has largely focused on citation fabrication, leaving a gap in evaluating the quality of retrieved studies and the factors driving their selection. In this study, we evaluated three general-purpose LLM chatbots: Claude Sonnet 5, Gemini 3.1 Pro, and ChatGPT GPT-5.5. We prompted the models with clinical questions adapted from 20 review questions in Issues 6 and 7 of the 2026 Cochrane Database of Systematic Reviews, simulating patient, clinician, and evidence-synthesis researcher roles. Each chatbot was queried under each user role with four independent repetitions, yielding 720 responses. Each chatbot was asked to support its answers with primary clinical citations, which we benchmarked against the included and excluded study sets of the Cochrane reviews. On average, a chatbot response retrieve
    
[^35]: omni-macos：苹果芯片上的设备端全模态搜索

    omni-macos: On-Device Omni-Modal Search on Apple Silicon

    [https://arxiv.org/abs/2608.05543](https://arxiv.org/abs/2608.05543)

    omni-macos在苹果设备上实现全模态本地搜索，通过内存预算管理和高效编码策略，确保所有数据不出设备。

    

    一个将文本、代码、文档、图像、音频和视频嵌入同一表示空间的搜索引擎，必须运行其编码器并将索引存储在某处，而几乎所有为此构建的组件都假设存在服务器。我们提出了omni-macos，它在其编码器、索引和存储都在已持有文件的Mac上运行，因此没有索引文件、键入的查询或向量会离开设备。它在用户设定的内存预算内，同时运行后台索引器和交互式搜索框：它仅重新编码编辑更改的块，在用户输入时将较小的单元交给GPU，从索引的一位副本回答查询并进行精确重新评分，并将该预算传播到使用统一内存的分配器。我们在五台Mac上进行了测量，其加速器宽度跨度八倍，内存跨度三十二倍，每台都对已持有的文件进行索引。

    arXiv:2608.05543v3 Announce Type: replace  Abstract: A search engine that embeds text, code, documents, images, audio and video into the same representation space has to run its encoder and keep its index somewhere, and almost every component built for the purpose assumes a server. We present omni-macos, which runs its encoder, index and store on the Mac that already holds the files, so no indexed file, no typed query and no vector ever leaves the machine. It keeps a background indexer and an interactive search box inside one memory budget the user sets: it re-encodes only the chunks an edit changes, hands the GPU smaller units while the user is typing, answers queries from a one-bit replica of the index with exact rescoring, and propagates that budget to the allocators that draw on unified memory. We measure on five Macs spanning an eightfold range of accelerator width and a thirty-twofold range of memory, each indexing the files it already holds.
    
[^36]: 对集合打分，而非累加段落分数：有效且高效的集合检索

    Scoring a Set, Not Summing Passage Scores: Effective and Efficient Set Retrieval

    [https://arxiv.org/abs/2607.05712](https://arxiv.org/abs/2607.05712)

    该论文提出用基于能量的查询—集合兼容性分数直接对完整候选证据集进行排序，而非独立打分或顺序累加段落分数，从而在多跳问答中实现有效且高效的集合检索。

    

    多跳问答需要检索多个证据段落，这些段落的有用性往往相互依赖。传统检索器要么独立地对段落进行排序，要么通过局部监督的“下一段落”决策来顺序地构建证据。顺序条件化虽然能捕获一定的跨段落依赖性，但其局部扩展分数无法为比较不同组成和规模的完整证据集提供统一的标准。因此，现有的多跳检索器避免直接学习针对完整证据集的查询—集合兼容性函数，因为指数级庞大的集合空间在学习时难以覆盖，在推理时也难以搜索。为了克服这些挑战，我们提出用基于能量的查询—集合兼容性分数 s_θ(q,S) 直接对候选证据集进行排序来构建多跳检索，该分数从信息丰富的对比中学习，而无需穷尽

    arXiv:2607.05712v2 Announce Type: replace  Abstract: Multi-hop question answering requires retrieving multiple evidence passages whose usefulness often depends on one another. Conventional retrievers either rank passages independently or construct evidence sequentially through locally supervised next-passage decisions. Sequential conditioning captures some cross-passage dependencies, but its local extension scores do not provide a common criterion for comparing complete evidence sets of different compositions and sizes. Existing multi-hop retrievers therefore avoid directly learning a query--set compatibility function over complete evidence sets, as the exponentially large set space is daunting to cover during learning and impractical to search at inference time. To overcome these challenges, we formulate multi-hop retrieval by directly ranking candidate evidence sets with an energy-based query--set compatibility score $s_\theta(q,S)$, learned from informative contrasts without exhaust
    
[^37]: Interactor：面向智能体强化学习的迭代式创作方法用于付费搜索广告描述生成

    Interactor: Agentic RL oriented Iterative Creation for Ad Description Generation in Sponsored Search

    [https://arxiv.org/abs/2606.15911](https://arxiv.org/abs/2606.15911)

    提出Interactor框架，通过智能体强化学习让生成模型与多个生成式奖励模型进行多轮交互迭代，自动生成融入世界知识且与落地页一致的高质量付费搜索广告描述。

    

    本文聚焦于付费搜索中信息丰富广告描述的自动生成。与通常为吸引用户点击反馈而优化的广告标题不同，广告描述具有更长的文本篇幅，并具备融入世界知识的潜力，能够在响应用户搜索意图的同时呈现广告的细粒度卖点。我们提出了Interactor，一个通过智能体强化学习（agentic RL）优化的多轮迭代创作框架，用于广告描述生成。生成模型作为策略，与由多个生成式奖励模型构成的定制化环境进行交互。在策略给出初始生成结果后，定制化的生成式奖励模型（GenRMs）对知识容量、落地页一致性等质量维度进行评估，同时提供二值信号和详细反馈。策略随后基于这些反馈迭代地精炼描述，以确保持续改进。实验表明……（摘要至此截断）

    arXiv:2606.15911v2 Announce Type: replace  Abstract: This paper focuses on automatically generating informative ad descriptions in sponsored search. Unlike ad titles which are usually optimized to attract user click feedbacks, ad descriptions have a longer text span and possess the potential of incorporating world knowledge to address user search intents while presenting the fine-grained selling points of the ads. We propose Interactor, a multi-turn iterative creation framework optimized with agentic RL for ad description generation. The generation model acts as a policy that interacts with a customized environment consisting of multiple generative reward models. Given initial generations by the policy, the customized GenRMs evaluate qualities including knowledge capacity and landing page consistency, providing both binary signals and detailed feedbacks. The policy then iteratively refines the descriptions based on such feedbacks to ensure continuous improvement. Experiments show that 
    
[^38]: GrepSeek：训练用于直接语料库交互的搜索智能体

    GrepSeek: Training Search Agents for Direct Corpus Interaction

    [https://arxiv.org/abs/2605.29307](https://arxiv.org/abs/2605.29307)

    GrepSeek提出了一种让LLM搜索智能体直接通过shell命令与语料库交互寻找证据的新范式，采用“Tutor/Planner生成可靠搜索轨迹初始化+GRPO强化学习优化”的两阶段训练方法，并通过语义保持的执行优化实现大规模语料库上的高效检索。

    

    arXiv:2605.29307v2 公告类型：replace-cross 摘要：大型语言模型（LLM）搜索智能体通过迭代推理与检索，在知识密集型任务上展现出了强大的潜力。现有系统大多依赖于从预构建索引中返回排序文档的检索器。我们探索了一种互补的范式：让智能体将语料库本身视为搜索环境，并通过可执行的shell命令来查找证据。我们提出了GrepSeek，一个经过优化的直接语料库交互（DCI）智能体，它学会在大型文本语料库上查找、过滤和组合证据。为了稳定大型语料库上的强化学习（RL）过程，我们采用两阶段训练：第一阶段，使用由答案感知的Tutor和答案盲的Planner生成的、经过验证且具有因果依据的搜索轨迹来初始化策略；第二阶段，使用组相对策略优化（GRPO）对策略进行改进。为了使直接语料库交互在大规模场景下切实可行，我们引入了两种保持语义的执行优化：（原文摘要在此处截断）

    arXiv:2605.29307v2 Announce Type: replace-cross  Abstract: Large Language Model (LLM) search agents have shown strong promise on knowledge-intensive tasks through iterative reasoning and retrieval. Most existing systems rely on retrievers that return ranked documents from a pre-built index. We explore a complementary paradigm in which the agent treats the corpus as the search environment and finds evidence through executable shell commands. We introduce GrepSeek, an optimized direct corpus interaction (DCI) agent that learns to find, filter, and compose evidence over large text corpora. To stabilize reinforcement learning (RL) over large corpora, we train in two stages: first, we initialize the policy using verified, causally grounded search trajectories generated by an answer-aware Tutor and an answer-blind Planner; then, we refine the policy using Group Relative Policy Optimization (GRPO). To make DCI practical at scale, we introduce two semantics-preserving execution optimizations: 
    
[^39]: LLM工具注册表中的面向智能体的信息设计：对修辞、位置与结构的预注册检验

    Agent-Facing Information Design in LLM Tool Registries: A Preregistered Test of Rhetoric, Position and Structure

    [https://arxiv.org/abs/2605.23916](https://arxiv.org/abs/2605.23916)

    研究通过预注册实验发现，LLM智能体的工具选择会被工具描述中的推销性赞美语言显著影响（可提升约43个百分点选中率，甚至选到无法胜任任务的工具），列表排序和结构化字段也会强烈左右选择，因此建议注册表以结构化字段而非营销文本来呈现工具限制。

    

    AI智能体通常从工具注册表中挑选工具，而注册表中每个工具的描述由其提供者撰写。我们探究这些描述中的推销语言是否会改变智能体所选择的工具。我们构建了仅在某一受控因素上存在差异的工具列表对（添加赞美之词、可验证的规范说明、或列表顺序），并让两个OpenAI模型调用其中一个工具。在一项预注册研究中，堆叠使用赞美之词（四种类型组合）使工具的选中率提高了约43个百分点，其效果与可验证规范相当甚至更强。赞美之词还会使部分选择偏向无法完成任务的工具，但很少使选择偏向索要不必要数据访问权限的工具。当列表内容完全相同时，排在首位的工具被选中的概率高出约72个百分点。在包含数值限制的任务中，结构化字段有助于智能体选择具备所需能力的工具；但在字段旁附上提供者的推销文字会削弱甚至消除这一收益。注册表可以将工具限制以字段形式列出，并隐藏推销性文字……（原文截断）

    arXiv:2605.23916v2 Announce Type: replace-cross  Abstract: AI agents often pick tools from registries, where each tool's provider writes its description. We ask whether sales language in those descriptions changes which tool an agent picks. We built pairs of listings differing in one controlled way (added praise, a verifiable specification, or list order) and asked two OpenAI models to call one tool. In a preregistered study, stacked praise (four kinds combined) raised a tool's pick rate by about 43 percentage points, matching or beating a verifiable specification. Praise also pulled some picks toward tools that could not do the task, but rarely toward tools asking for unneeded data access. With identical listings, the first-listed tool was picked about 72 points more often. On tasks with numeric limits, structured fields helped agents pick the capable tool; adding the provider's sales text beside the fields reduced or erased that gain. Registries could list limits as fields, hide sale
    
[^40]: SOLAR：面向推荐的SVD优化终身注意力机制

    SOLAR: SVD-Optimized Lifelong Attention for Recommendation

    [https://arxiv.org/abs/2603.02561](https://arxiv.org/abs/2603.02561)

    该论文提出了SVD-Attention——一种通过将低秩嵌入分解为r个主成分、在紧凑空间中计算候选物品与兴趣得分并保留softmax机制的新型注意力方法，并在此基础上构建了面向终身推荐的高效集合感知框架SOLAR。

    

    注意力机制仍然是Transformer中定义性的核心算子，因为它提供了富有表现力的全局信用分配，但其在序列长度N上的二次方计算代价使得长上下文建模成本高昂，并常常迫使模型采用截断或其他启发式方法。线性注意力通过核特征映射重新组织计算，将复杂度降低到O(Nd^2)，但这种重构形式丢掉了softmax机制，并改变了注意力分数的分布。终身推荐同样需要高效的注意力机制，以便在严格的延迟和资源约束下，对用户历史行为与候选物品进行大规模序列建模。我们提出了SVD-Attention——一种新颖的注意力机制，以及基于它构建的SOLAR——一个面向终身推荐的集合感知框架。SVD-Attention将低秩嵌入分解为r个主成分，在紧凑空间中计算候选物品与兴趣之间的得分，并对这些得分应用softmax。其双线性重（原文在此处截断）

    arXiv:2603.02561v2 Announce Type: replace  Abstract: Attention mechanism remains the defining operator in Transformers since it provides expressive global credit assignment, yet its quadratic cost in sequence length N makes long-context modeling expensive and often forces truncation or other heuristics. Linear attention reduces complexity to O(Nd^2) by reordering computation through kernel feature maps, but this reformulation drops the softmax mechanism and shifts the attention score distribution. Lifelong recommendation requires efficient attention as well, for large-scale sequence modeling with user histories and candidate items under tight latency and resource constraints. We introduce SVD-Attention, a novel attention mechanism, and SOLAR, a set-aware framework built on it for lifelong recommendation. SVD-Attention factorizes low-rank embeddings into r principal components, computes candidate-to-interest scores in compact space, and applies softmax over those scores. Its bilinear re
    
[^41]: Life-Bench：超越概念识别的多模态个性化基准测试与知识图谱框架

    Life-Bench: A Benchmark and Knowledge Graph Framework for Multimodal Personalization Beyond Concept Recognition

    [https://arxiv.org/abs/2602.19001](https://arxiv.org/abs/2602.19001)

    本文提出了Life-Bench——一个包含11,800多个问答对、按概念识别、事件理解和聚合推理三个层次组织的合成多模态个性化基准测试，并配套提出个人知识图谱框架LifeGraph，通过结构化检索与按需访问源视觉证据，在事件理解和聚合推理任务上表现出显著优势。

    

    随着大语言模型日益成为个人助手的核心驱动力，用户期望它们能够对多模态生活履历进行推理——从识别人物到理解事件再到聚合规律，然而现有基准测试主要针对概念级识别。我们推出了Life-Bench，这是一个完全合成、经人工验证的多模态基准测试，包含超过11,800个问答对，涵盖10个任务，并按所需证据范围分为三个层次：概念识别、事件理解和聚合推理。该基准中以照片为中心的个人生活履历在嵌入统计特征上与真实用户账户的分布保持一致。个人数据的互联结构天然适合基于图的解决方案；为此我们提出了LifeGraph，一个个人知识图谱框架，提供结构化检索并支持按需访问源视觉证据，在事件理解和聚合推理任务上展现出特别的优势。对四种检索方法的系统性评估……

    arXiv:2602.19001v2 Announce Type: cross  Abstract: As large language models increasingly power personal assistants, users expect them to reason over multimodal life histories, from recognizing people to understanding events to aggregating patterns, yet existing benchmarks primarily target concept-level recognition. We introduce Life-Bench, a fully synthetic, human-verified multimodal benchmark of over 11,800 question-answer pairs across 10 tasks, organized by required evidence scope: concept identification, event understanding, and aggregated reasoning. The benchmark's photo-centric personal histories are distributionally aligned with real user accounts under embedding statistic. The interconnected structure of personal data invites graph-based solutions; we propose LifeGraph, a personal knowledge graph framework providing structured retrieval with on-demand access to source visual evidence, showing particular promise on event and aggregated tasks. Systematic evaluation of four retriev
    

