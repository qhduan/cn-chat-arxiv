# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LongZero: Enhancing Long-Context Reasoning in LLMs from Zero Data](https://arxiv.org/abs/2610.30132) | LongZero提出了一种无需任何外部数据的自举式强化学习方法，让大语言模型通过自行合成结构化的问题-答案-证据任务来生成训练数据，从而有效提升自身的长上下文推理能力。 |
| [^2] | [Think Harder Where It Matters: Item-Level Adaptive Recursion for Sequential Recommendation](https://arxiv.org/abs/2610.30128) | 提出LoopRec，一种针对序列推荐的物品级自适应递归方法，根据每个物品的实际计算需求动态分配递归深度，取代固定计算预算，从而既节省计算资源又提升整体推荐质量。 |
| [^3] | [Environment Duels: A Microcosm of LLM Co-Evolution](https://arxiv.org/abs/2610.30116) | 提出“环境对决”这一动态智能体基准，让大语言模型同时担任环境创作者与任务解决者两种角色，发现解决与创作是两种不同的能力，并利用模型间两两对决的传递性结果刻画了模型协同进化的稳定生态图景。 |
| [^4] | [Quant-Arena: Benchmarking LLM Agents for Quantitative Trading Research](https://arxiv.org/abs/2610.30113) | 提出了Quant-Arena——首个基于真实A股数据的端到端量化研究基准，让LLM智能体仅凭2022-2023年反馈冻结策略、在保密的2024年数据上评估，以检验其能否完成从市场假设到真实有效策略的完整量化研究循环。 |
| [^5] | [Did the Problem Actually Get Harder? Testing LLMs on Paired P and NP-Complete Problems Hardened by Adversarial Multi-Agent Rewriting](https://arxiv.org/abs/2610.30103) | 该论文提出P2NPCBench基准，将NP完全问题与仅有一个结构细节之差的P类孪生问题配对，并利用对抗性多智能体重写框架消除教科书式记忆线索，以检验大语言模型是真正具备区分问题复杂度类的推理能力还是仅仅依赖记忆。 |
| [^6] | [VX-One: Typed, Calibrated Decision Heads for Real-Time Perimeter Triage - An empirical study of unknown-attack honesty, cascade gray zones, and calibration under drift](https://arxiv.org/abs/2610.30090) | 本文提出VX-One，一个将商用有类型决策原语（校准门控、含显式未知选项的闭集选择、序数分数）作为输出的小型多头网络，在CPU上以毫秒级延迟实现实时边界分流，并实证评估了未知攻击诚实性、级联灰色地带与漂移下的校准性能。 |
| [^7] | [GoiBERT: Compact Vietnamese Language Models via Two-Stage Knowledge Distillation](https://arxiv.org/abs/2610.30067) | GoiBERT通过两阶段知识蒸馏（先从同分词器越南语编码器学习通用表示，再从大型教师模型获取软标签任务指导），构建了仅2600万和4900万参数、可直接处理原始文本的紧凑越南语编码器，在八项基准测试中以不到一半的参数量和两倍吞吐量取得非教师模型中的最高平均分数。 |
| [^8] | [Looped Uniform Diffusion Language Models](https://arxiv.org/abs/2610.30066) | 该论文首次将循环机制（在每步去噪中递归复用早期Transformer层）引入均匀扩散语言模型，在不改变训练目标和采样器的情况下显著提升训练效率，使循环UDLM以更少计算量即可达到非循环模型训练约0.5万亿token后的验证NLL水平。 |
| [^9] | [What the Author Wrote, What the Agent Reads: Silent Cue Loss in Enterprise Document Ingestion](https://arxiv.org/abs/2610.30026) | 该研究揭示了企业文档摄取环节中所有主流转换器都会丢失作者通过文本格式编码的含义，导致来自两家供应商的六个大模型智能体在恢复作者重点标注要求的任务上性能从F1 0.690骤降至0.075-0.150，且这种失败表现为“自信地给出错误答案”而非“明显的不完整”。 |
| [^10] | [Agent Behavior Maps: Locating Agents by What They Would Do, Without Running Them](https://arxiv.org/abs/2610.30014) | 提出“智能体行为地图”方法，利用共享参考轨迹上的中心化动作对数似然为LLM智能体定位，无需实际运行目标智能体即可刻画其行为特征、追踪轨迹KL散度并支持性能预测与行为相似性分析。 |
| [^11] | [Building Auditable AI Text Detectors with Sparse Autoencoders](https://arxiv.org/abs/2610.30006) | 该论文提出一种流水线，通过在神经检测器的残差流上训练稀疏自编码器、自动生成特征解释标签并按全局保真度阈值筛选特征，从而将不透明的AI文本检测器转化为基于人类可读概念运行的可审计分类器。 |
| [^12] | [HiddenLens: A Multi-Layer LLM Framework for Detecting and Resolving Hidden Assumptions in Software Requirements](https://arxiv.org/abs/2610.29996) | HiddenLens是首个专为检测、分类、质疑并解决软件需求中隐藏假设而设计的六层BERT-LLM框架，能够超越单纯的歧义标记，在早期发现并化解需求中的隐含假设，从而显著降低后期修复成本。 |
| [^13] | [Remembering Through Change: Provenance-Preserving Hypergraph Memory for Multimodal Agents](https://arxiv.org/abs/2610.29995) | 提出了 MHG-Mem，一种多模态超图记忆，通过超边组织实体身份、状态演化与视觉证据溯源，使多模态智能体在跨会话追踪实体变化的同时保留细粒度视觉证据。 |
| [^14] | [CulturalRiddles: A Multicultural Benchmark of Riddles](https://arxiv.org/abs/2610.29968) | CulturalRiddles是一个涵盖62种文化、完全由人类创作的谜语基准测试，用于评估大语言模型的跨文化理解与推理能力，结果表明即使是前沿模型在某些社区中的准确率也最高仅达22%。 |
| [^15] | [Commit or Abstain? System One Decision Models, LLMs, and Missing Legal Authority](https://arxiv.org/abs/2610.29961) | 系统一决策模型在正确的法律依据缺失时能够可靠地弃权（错误作答率仅30%，远低于生成式大语言模型的38%-86%），并以仅为生成式大语言模型1/55的成本达到同等准确率，为可验证的法律AI提供了高性价比且诚实的解决方案。 |
| [^16] | [Towards English - Mizo Legal Machine Translation: Corpus Creation, Analysis, and Benchmarks](https://arxiv.org/abs/2610.29943) | 本文构建了首个英语-米佐语印度法院判决平行语料库（13,168个句子对），并通过对NLLB-200 1.3B模型微调实现法律领域机器翻译，在法律测试集上BLEU分数较基础模型提升7.16。 |
| [^17] | [ToM2Vec: Continuous Theory-of-Mind Tokens for Multi-Agent Coordination](https://arxiv.org/abs/2610.29934) | 提出ToM2Vec方法，利用固定数量的连续推理词元在丰富的潜在空间中学习抽象的心智理论表示，使LLM智能体能够在不确定性下捕捉伙伴的多种可能状态，从而显著提升多智能体协作能力。 |
| [^18] | [Graph Topology Meets Energy-Driven Pathfinding: Preserving Contextual Consistency in Retrieval-Augmented Generation](https://arxiv.org/abs/2610.29933) | 提出 ThreadRAG 框架，通过语义拓扑映射器（STM）将检索文档组织为上下文完整的语义节点并构建语义拓扑图，结合能量驱动的路径搜索，解决知识图谱 RAG 中语义损失和模态失配两大瓶颈问题。 |
| [^19] | [Your Benchmark Is an Empirical Measure Over Difficulty](https://arxiv.org/abs/2610.29927) | 本文提出将基准测试视为难度上的经验测度这一新视角，识别并形式化了“测度失配”与“覆盖缺口”两种失效模式，提出两个具有严格理论保证的诊断指标，并对7个领域25个流行基准的分析表明许多基准的难度权重隐式失衡，其排行榜结论需谨慎解读。 |
| [^20] | [Task-Preserving Multilingual Instruction Tuning for Open-Ended QA in WMT 2026 MIST](https://arxiv.org/abs/2610.29926) | 该研究通过LoRA结合任务保持型本地化构建的多语言指令数据集（EuroAlpaca）及其跨语言增强版本（CrossEuroAlpaca）对Qwen3-8B进行微调，在WMT 2026 MIST共享任务的十种语言开放式问答任务上显著提升了性能，但这些提升具有任务和语言特异性。 |
| [^21] | [Unintended Side Effects of Value Training in LLMs](https://arxiv.org/abs/2610.29922) | 本文通过在霍夫斯泰德各价值维度上微调四个大语言模型，发现针对特定价值的训练会对未训练的价值观、风险偏好、社会认知和安全特性产生意外副作用，因此建议价值对齐评估应纳入更广泛的行为测试。 |
| [^22] | [Benchmarking Large Language Models for Detecting Satirical Code-Mixed Sentences](https://arxiv.org/abs/2610.29907) | 该论文构建了包含60余万样本的罗马尼亚语语码混合讽刺检测数据集CS-SeLeRoSa，并系统基准测试了多语言大模型在零样本、思维链和微调三种模式下检测讽刺性语码混合句子的能力。 |
| [^23] | [Investigating the Impact of Learner Errors on Social Bias in Large Language Models](https://arxiv.org/abs/2610.29896) | 该研究首次系统考察了非母语写作者的语法错误对大型语言模型社会偏见的影响，发现错误率越高偏见越强（其中介词错误影响最大），并通过在错误文本上微调有效缓解了这一问题。 |
| [^24] | [Every Part Has Its Role: Part-of-Speech Effects in LLM Post-Training](https://arxiv.org/abs/2610.29843) | 该研究将概念增广方法从名词拓展到动词和形容词，发现承载概念的词性会影响大语言模型后训练中的困惑度和下游性能，表明不同词性在训练中扮演着不同角色。 |
| [^25] | [S-EDL: Eliciting Self-Evidence from Sequence Likelihoods for Semantic Calibration of LLMs](https://arxiv.org/abs/2610.29832) | 提出S-EDL证据学习方法，通过从模型自身的序列似然中引出证据并构建可微分的Dirichlet先验，实现了LLM在开放式生成中的语义校准，完全无需依赖固定的多项选择选项。 |
| [^26] | [MemoryTax: The Bitter Lesson of Agent Memory](https://arxiv.org/abs/2610.29823) | 这项实证研究揭示了智能体记忆的“苦涩教训”——精心设计的记忆系统只是弱模型的昂贵补丁（仅对 GPT-4o-mini 有用且构建成本高昂），其效果受写入模型能力封顶，而强模型直接搜索原始历史的全上下文方式在成本-精度上更优。 |
| [^27] | [Real PMIDs, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents](https://arxiv.org/abs/2610.29812) | 该论文通过受控消融实验发现，在基于实时PubMed API的生物医学论断验证任务中，让语言智能体自行构造查询的框架设计能将标准论文召回率提升三倍并取得最佳准确率（62%），而简单的单次检索仅能为6%的论断找到标准论文且无法提升准确率。 |
| [^28] | [MAPS: Macro-Adaptive Parallel Reasoning for Regulating Search Momentum in Retrieval-Augmented Generation](https://arxiv.org/abs/2610.29792) | 该论文提出“搜索动量”概念来刻画RAG中推理演变对查询意图的影响，发现查询生成易沿已有检索路径持续演化，并据此提出宏自适应并行推理方法MAPS来调控搜索动量，从而减少冗余子查询并提升查询探索的多样性。 |
| [^29] | [Where Do LLMs Learn to Recommend AI?](https://arxiv.org/abs/2610.29785) | 研究发现LLM过度推荐AI的倾向主要源于指令微调数据中约7%的AI相关样本——剔除这些数据可将推荐率降至预训练模型水平以下，而追求“有帮助”的数据和“最佳”“立刻”式的提问方式则会进一步推高该倾向。 |
| [^30] | [Evolving with Open Worlds: Towards Recursive Self-Improvement for Future Forecasting](https://arxiv.org/abs/2610.29784) | 提出DeepEvolve智能体系统，以未来事件预测作为持续变化且可验证的开放环境，通过递归演化引擎将预测背后的推理轨迹转化为能力，从而实现智能体的递归自我改进。 |
| [^31] | [ALIAS: Evaluating Performance and Memorization in LLM-Written TLA+ Specifications](https://arxiv.org/abs/2610.29767) | 该论文利用TLA+模型检查器作为验证神谕，发现LLM在标识符重命名任务上的正确率下降主要来自要求精确标识符名称的评分方式，若按行为语义评分，记忆化效应实际很小甚至消失。 |
| [^32] | [Bridging Foundational and Agentic Capabilities of LLMs](https://arxiv.org/abs/2610.29751) | 该论文提出一个操作层面的测量框架，通过定义“能力实现差距”来量化大语言模型基础能力向智能体能力的转化程度，发现基础能力基准只能部分预测智能体操作表现，画像相似的模型在实际执行中差异显著。 |
| [^33] | [Accept or Repair: Preserving Passage-Supported Answers in Contrastive Decoding](https://arxiv.org/abs/2610.29727) | 提出无需训练的“接受或修复”（AoR）方法，通过将答案与所给段落核对来保留段落支持的答案、仅在不支持时用对比解码进行修复，在反事实段落上比采样提升 12.5 分并超越八种基线。 |
| [^34] | [Retrieval Augmented Generation Compositionality](https://arxiv.org/abs/2610.29722) | 该论文提出开源受控RAG评估框架LASAGNA，通过对450种配置、八个基准的大规模实验发现，RAG流程中各阶段的强技术在组合后增益无法叠加（预测+47%实际仅+17%）。 |
| [^35] | [Scaling Multilingual Pretraining of Pixel Language Models](https://arxiv.org/abs/2610.29681) | 本文从更多语言、更多文字系统和更大模型容量三个维度扩展像素语言模型的多语言预训练，发现扩展的收益主要作用于新增语言，更大的模型容量可缓解多语言诅咒，且模型在低资源语言上优于子词级和字符级基线。 |
| [^36] | [Uni-Mem: Learning a Long-Horizon Agentic Memory Manager with Unified Online Policy Optimization](https://arxiv.org/abs/2610.29671) | 提出Uni-Mem统一记忆管理器，通过阶段解耦GRPO与事件级信用分配实现端到端在线强化学习，在共享策略下联合优化长期智能体的记忆构建与检索能力。 |
| [^37] | [Conflicted Records: Typed Adversarial Contracts for Auditable LLM Output Quality Enforcement](https://arxiv.org/abs/2610.29669) | 提出 BiasGuard-D，通过两个类型化对抗契约角色对结构化智能体间记录执行确定性检查，实现对 LLM 输出质量的可审计强制执行，捕获单一 LLM 评判者与传统数据清洗都无法发现的冲突性缺陷。 |
| [^38] | [SPARE: Sparsity Allocation via Routing Entropy for Efficient MoEs](https://arxiv.org/abs/2610.29662) | SPARE是一种无需训练的MoE专家剪枝方法，利用路由熵自适应地为各层分配不同的剪枝稀疏度，在相同预算下比均匀剪枝获得更高的模型准确率。 |
| [^39] | [Effects of Model Scale and Quantization on Social Bias and Abstention in Question Answering](https://arxiv.org/abs/2610.29657) | 该研究发现压缩模型观察到的偏见减少可能部分源于弃答增多而非真正更公平——在26个偏见回答较少的配置中，有20个同时表现出过度弃答的增加。 |
| [^40] | [LLMs as survey respondents: Comparison to simple semantic baselines](https://arxiv.org/abs/2610.29645) | 研究发现LLM在根据部分人类回答预测问卷新回答时通常优于基于嵌入相似性的简单语义基线，但其优势随已知回答比例的降低而减弱，表明其可能依赖题目间的表层语义相似性作为启发式方法。 |
| [^41] | [Overview of CHIP 2026 Shared Task 1: Patient-Specific Phenotype Concept Recognition in Full-Text Biomedical Literature](https://arxiv.org/abs/2610.29606) | 本文概述了CHIP 2026共享任务1，该任务基于PatientPheX数据集的200篇标注全文文献，设置全文表型概念识别与患者特异性表型关联两个子任务，以推动从全文生物医学文献中自动识别表型概念及其患者关联方法的发展。 |
| [^42] | [A Diagnostic Study of Faithfulness in Diffusion Language Models for Table-to-Text Generation](https://arxiv.org/abs/2610.29602) | 本研究诊断发现，扩散语言模型在表格到文本生成中随示例增加或微调后流畅度和BLEU提升，但对源表格的事实忠实性反而下降，揭示了表面流畅性与事实依据性的分离。 |
| [^43] | [WRDO: Weighted Refusal Direction Optimization in Aligned Language Models](https://arxiv.org/abs/2610.29555) | 提出加权拒绝方向优化方法，通过学习聚合权重从针对每个提示的方向库中构建复合拒绝方向，以强化对齐语言模型的拒绝干预效果。 |
| [^44] | [PACE: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States](https://arxiv.org/abs/2610.29545) | PACE 提出一种事后的四比特状态量化方法，通过每隔几个 token 才周期性地对循环状态进行一次舍入，并利用大数值条目在通道上聚集对齐的结构特性，减少舍入误差累积，从而在不微调模型的情况下实现高精度的四比特推理状态存储。 |
| [^45] | [RCSD: Reinforcement and Continual Learning for Domain-Specific Speculative Decoding](https://arxiv.org/abs/2610.29527) | RCSD通过强化学习与持续学习，利用投机解码中自然产生的接受/拒绝序列构建隐式偏好对来优化草稿模型，使平均接受长度较HASS提升35%，端到端推理速度提升1.35倍。 |
| [^46] | [pplx-embed-v2-late Technical Report](https://arxiv.org/abs/2610.29523) | Perplexity 推出的 pplx-embed-v2-late 包含 0.6B 和 9B 两个多语言多模态后期交互模型，通过 18B 教师模型的表示蒸馏训练，在文本、图像和视觉文档检索上达到同规模最先进水平，并借助共享嵌入空间支持“9B 建索引、0.6B 低延迟查询”的非对称部署。 |
| [^47] | [Beyond the Vote: When Do LLMs Add Value to Retrieval-Based Tabular Classification?](https://arxiv.org/abs/2610.29520) | 该研究在19个表格分类数据集上将大语言模型与距离加权最近邻投票进行系统比较，发现Distance-KNN整体平均准确率最高，LLM仅在检索邻居标签一致性低且测试样本距离较近或中等时才有优势。 |
| [^48] | [Mitigating Query Shortcuts in Chain-of-Thought with Context-Aware Decoding](https://arxiv.org/abs/2610.29518) | 提出推理依据对比解码（RCD）方法，通过对比查询在有无推理依据时的词元分布来放大推理依据的作用，从而缓解LLM在思维链推理中依赖查询捷径的行为并提升答案准确率。 |
| [^49] | [Keeping Time vs. Understanding Time: Neither Alone Explains Temporal Reasoning in LLMs](https://arxiv.org/abs/2610.29507) | 该论文通过微调迁移实验和基于认知需求的评估，挑战了将时间推理划分为“时间算术”与“时间语义”的主流二分法，证明二者都不能单独解释大语言模型的时间推理能力。 |
| [^50] | [A Strong MoE Is All You Need for Process Supervision](https://arxiv.org/abs/2610.29498) | 该论文的核心创新在于发现MoE模型的专家路由统计天然蕴含过程置信度信号，据此提出RoCo-Rethink与RoCo-PRM，利用路由置信度自动标注推理轨迹并选择性重生成低置信步骤，从而无需人工标注即可实现可靠的过程监督。 |
| [^51] | [TenetBench: A Clause-Anchored Russian Corpus for Evaluating Language Models against Professional Codes of Ethics](https://arxiv.org/abs/2610.29477) | 提出了 TenetBench——一个由自动化流水线从职业伦理准则文本生成的俄语基准语料库，包含 3,433 个与准则条款冲突的请求及其良性孪生对照，可随着新准则加入持续扩展，用于精细评估大语言模型是否会违反或帮助用户违反具体的职业伦理条款。 |
| [^52] | [OmniSimulator: Aligning Small Language Models for Authentic Heterogeneous Behavior Modeling](https://arxiv.org/abs/2610.29472) | 本文提出 OmniSimulator 训练框架，通过五维结构化行为推理模式、精选真实用户行为数据以及教师模型生成的高质量推理轨迹，对齐小型语言模型，从而显著提升其在短视频平台上进行真实异构用户行为模拟的预测能力。 |
| [^53] | [Harness Scaling for Generalizable Agentic Reinforcement Learning](https://arxiv.org/abs/2610.29458) | 提出Harness Scaling这一智能体强化学习的新扩展维度，将训练从单一固定框架扩展到多样化执行配置，从而提升模型在不同框架配置下的泛化与迁移能力。 |
| [^54] | [Omnilingual-MsASR: Benchmarking Multimodal, Multilingual, Long-Form Multi-Speaker ASR of Omni Large Language Models](https://arxiv.org/abs/2610.29455) | 该论文提出了Omnilingual-MsASR，一个涵盖六种语言、最长30分钟、最多20个说话人的多语言多模态长时多说话人语音识别基准，揭示出当前全模态大语言模型在该任务上存在严重性能瓶颈，tcpWER经常超过100%。 |
| [^55] | [DynaWeave: Learning to Adapt through Dynamic Scenario Synthesis](https://arxiv.org/abs/2610.29443) | 提出DynaWeave自动化框架，通过生成具有依赖关系的外生事件将静态LLM智能体任务转化为可验证的动态场景，从而训练和评估智能体应对环境突变的能力。 |
| [^56] | [BIRD-RL: Scaling Agentic Reinforcement Learning over Stateful Data-Centric Environments](https://arxiv.org/abs/2610.29426) | BIRD-RL提出了一种四阶段智能体强化学习课程方法，通过交替增强基础推理与轨迹自适应在线探索，解决了SQL调试等有状态数据中心环境中环境隔离成本高、状态污染和跨轨迹干扰导致的训练不稳定问题。 |
| [^57] | [A KAN-based Non-linear Factorization for Embedding Layer Compression](https://arxiv.org/abs/2610.29424) | 提出一种基于Kolmogorov-Arnold网络（KAN）的非线性分解方法，通过在压缩嵌入层中引入可学习的非线性投影，突破传统低秩分解的线性瓶颈，提升嵌入层压缩后的表示能力。 |
| [^58] | [CCG Diffusion: Combinatory Categorial Grammar-Driven Compositional and Syntactically Disambiguated Text-to-Image Generation](https://arxiv.org/abs/2610.29415) | 该论文提出了免训练的CCG Diffusion方法，利用组合范畴语法（CCG）中词类别所携带的丰富句法信息，在文生图生成中实现对属性绑定、空间关系、数量控制以及句法歧义提示词预期解释的精确控制。 |
| [^59] | [Beyond Small Talk: Modeling Disclosure Trajectories in Human-AI Conversation](https://arxiv.org/abs/2610.29413) | 本文提出基于社会渗透理论的TraDE评估框架，通过预定义披露轨迹来评估在当前披露相同的条件下，先前的披露历史如何影响人机对话中模型对用户自我披露的回应方式。 |
| [^60] | [ECHO: Emergent Common-Mode Hidden Offsets Drive Vocabulary Collapse in Preference Optimization](https://arxiv.org/abs/2610.29410) | 本文发现偏好优化中存在一种相对损失无法察觉的共模隐藏偏移，它会在输出层形成“词表井”，使单一词元概率主导解码并最终导致生成坍缩。 |
| [^61] | [Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories](https://arxiv.org/abs/2610.29378) | 提出OnQ流水线，通过在策略蒸馏利用学生模型自生成的轨迹进行训练，解决了固定响应监督导致的训练与生成前缀不匹配及高熵处分布失配问题，从而恢复低比特量化大模型的推理能力。 |
| [^62] | [Memory-Centric Continual Learning for AI Agents](https://arxiv.org/abs/2610.29357) | 本文提出以记忆为中心的持续学习框架（MCCL），主张以记忆能否改善智能体未来行动而非检索效果作为评判标准，并通过三轴记忆分类体系和内外双循环的生命周期机制，使AI智能体能够真正实现从经验中持续学习。 |
| [^63] | [Don't Stop Long-Context Continual Pre-Training Too Early: An Empirical Scaling Law](https://arxiv.org/abs/2610.29352) | 本文建立了一条经验缩放定律，表明长上下文持续预训练的性能随模型规模与训练token数按加性幂律持续提升，因此不应过早停止LCCP训练。 |
| [^64] | [Closed-Loop Dual-Reference Guidance for Accent Control in Cross-Lingual Voice Cloning](https://arxiv.org/abs/2610.29351) | 提出闭环双参考引导方法，通过将原始参考与语音转换构建的母语锚点配对并动态更新语言模型状态，有效消除跨语言语音克隆中的源语言口音，在仅损失极小说话人相似度的前提下显著提升目标语言的发音质量。 |
| [^65] | [B-CARE: Blind Classwise Assessment of Relative Evidence for Stance Error Ranking](https://arxiv.org/abs/2610.29350) | B-CARE提出了一种预测盲的高风险立场错误排序框架，利用固定的LLM验证器对比各类立场标签之间的相对证据强度来构建风险分数，且证据画像可跨异构基础模型复用而无需额外训练错误检测器。 |
| [^66] | [Data Schedules and Learning Rate Schedules Should Be Designed Together](https://arxiv.org/abs/2610.29348) | 研究表明数据调度与学习率调度必须协同设计：中期训练中最优的数据过渡策略依赖于学习率调度，当学习率持续衰减时，更早引入中期训练数据并与预训练数据混合能降低最终损失。 |
| [^67] | [Decodability Is Not Controllability: Local and Anisotropic Speaker Control in Continuous-Latent TTS](https://arxiv.org/abs/2610.29345) | 该研究揭示连续潜变量TTS中说话人特征的高可解码性并不等于因果可控性，有效的话者音色控制依赖于在参考音频早期位置（L0层）进行干预，而非探针准确率最高的深层。 |
| [^68] | [Uncensored Is Not Unbiased: Abliteration Removes Refusals but Not Script-Dependent Bias About Taiwan](https://arxiv.org/abs/2610.29342) | 去审查化（abliteration）模型只是移除了拒绝行为，其对台湾问题的政治立场主要取决于所用书写文字（繁体、简体还是英文），而非是否经过消融审查，因此“无审查”并不等于“无偏见”。 |
| [^69] | [Reader--Writer Coupling in Transformer MLPs](https://arxiv.org/abs/2610.29340) | 本文提出将Transformer MLP中隐藏通道的读取器与写入器视为耦合对进行联合测量，利用MLP更新雅可比矩阵与自身转置的一致性导出带符号的配对坐标，并据此发现每个模块仅由一小部分强耦合通道构成核心。 |
| [^70] | [ChronoSynth: Constraint-Aware Question Synthesis over Temporal Knowledge Graphs](https://arxiv.org/abs/2610.29339) | 本文提出ChronoSynth框架，通过构建时序意图状态、检索适配带类型问题模式并进行验证修订，解决了时序知识图谱问题生成中的时序语义漂移问题，确保生成的问题保留决定答案集合的关键时间约束。 |
| [^71] | [Policy Central: Detecting Conflicts, Redundancy, and Similarity in Deployed Police Policy](https://arxiv.org/abs/2610.29338) | 论文部署了名为 Policy Central 的系统，通过比较“先检索后分类”与“先筛选后确认”两种架构来检测警察政策条款间的冲突、冗余与相似性，发现模型判断环节表现良好（冲突类召回率达 0.885），但在近两万条款中找到相关配对仍是主要瓶颈，端到端召回率仅约 0.46。 |
| [^72] | [Seeing the Forest for the Trees: Emergent Agreement from Individual Conversations](https://arxiv.org/abs/2610.29328) | 本研究通过对个体对话共识内部表征进行跨对话聚合，发现尽管个体对话层面无法可靠预测共识，但聚合后的共识可用简单的马尔可夫过程很好地预测，实现了“由树见林”的涌现性洞察。 |
| [^73] | [OACap: Caption-Guided Active Observation for Detailed Video Captioning](https://arxiv.org/abs/2610.29327) | 提出了以描述为中心的智能体框架 OACap，它能将视频描述中缺失或缺乏支持的信息转化为主动观察请求，借助时空工具获取证据并迭代验证修订描述，从而显著提升详细视频描述质量，同时引入 FSCap-Bench 基准及 CSFS 等评分指标来系统评估描述的事实性与语义保真度。 |
| [^74] | [Signals and Queries: Token-Efficient Agentic Context Discovery](https://arxiv.org/abs/2610.29326) | 该论文提出S&Q方法，利用LLM离线生成激活样例与困难负样本训练独立逻辑回归分类器，在无需每轮LLM调用的情况下高效剔除至少30%的无关候选策略，在保持高召回率的同时显著降低了智能体策略相关性评估的Token成本。 |
| [^75] | [OSynth: A Longitudinal, Context-Sensitive Evaluation of Meeting Understanding](https://arxiv.org/abs/2610.29319) | OSynth是一个包含570小时模拟组织会议音频的大规模合成数据集，其真实感使人类难以与真实会议区分，并证明额外的上下文信息能显著提升大语言模型在实体敏感会议理解任务上的表现。 |
| [^76] | [Asking What a Director Sees Can Change What Vision-Language Models Choose](https://arxiv.org/abs/2610.29318) | 研究发现在导演任务中，先询问视觉语言模型“导演能看到什么”会显著改变其后续行动选择（选择导演看不到的方块的比例提高12至88个百分点），这挑战了“要求报告视角不会影响行动”的传统测试假设。 |
| [^77] | [LLM-Based Exogenous World Modeling for Dynamic Scheduling](https://arxiv.org/abs/2610.29308) | 该论文提出将预训练大语言模型作为结构化世界模型的外生组件，通过后状态分解与紧凑信念表示预测未来作业到达，并结合信念条件化调度策略、意外触发刷新机制和在线残差自校准，实现了高效的动态调度规划。 |
| [^78] | [ExpEvolve: Structuring and Reusing Experience for Open-Ended Discovery](https://arxiv.org/abs/2610.29303) | 提出ExpEvolve框架，将思想层级的提取与控制注入进化式编码代理，通过发现能够解释程序间评分差异的高层级思想因素并复用经验，提升开放式科学发现的样本效率。 |
| [^79] | [Same Probabilities, Different Advice: Gain-Loss Framing Effects in High-Stakes LLM Decision Support](https://arxiv.org/abs/2610.29294) | 该研究提出FrameBench-HPS基准，揭示了大语言模型在提供高风险决策建议时存在显著的框架效应：即使概率信息在数学上完全等价，收益与损失两种表述方式仍会在32.4%的情况下改变模型的推荐类别。 |
| [^80] | [Credit Where Credit Is Due: Contrastive Attribution Topology Search for Multi-Agent Systems](https://arxiv.org/abs/2610.29292) | CATS方法让LLM归因器通过对比不同候选拓扑的执行轨迹为各智能体精准分配功劳，从而在智能体超图中自动搜索并剪枝出更小更优的多智能体团队，在提升准确率的同时显著降低部署成本。 |
| [^81] | [Computer Vision and Knowledge-Grounded Language Models for Intelligent Beekeeper Assistance](https://arxiv.org/abs/2610.29283) | 该论文提出了一种边缘可部署的智能系统框架，结合计算机视觉目标检测与知识增强语言模型，将蜂巢巢框照片自动转化为可解释、面向行动的蜂群检查报告，从而辅助养蜂人高效评估蜂群状况。 |
| [^82] | [Heard But Not Said: Prompting Understates Temporal Information of Musical Moments in Frozen Large Audio-Language Models](https://arxiv.org/abs/2610.29256) | 该论文提出基于 Slakh2100 分轨激活的成对状态/变化查询探针，发现冻结的大型音频-语言模型实际上具备音乐时刻的时间信息，但整曲提示会低估这一能力——状态查询无需上下文即可高于基线，而变化查询则依赖前置音频才能提升。 |
| [^83] | [Hierarchical Masked Diffusion Language Models](https://arxiv.org/abs/2610.29255) | 提出了层次化掩码扩散语言模型（HMDLM），通过为每个token位置附加层次级别信息来建模位置间的生成优先级，使掩码扩散语言模型无需额外标注即可按照自然的层次结构有序地并行生成文本。 |
| [^84] | [Intergroup Bias Predicts Expressions of Social Judgment and Empathy in the r/AmITheAsshole Reddit Community](https://arxiv.org/abs/2610.29254) | 本研究基于 r/AmItheAsshole 社区超过12万条评论的大规模分析，首次在自然在线社交场景中揭示了评论者与发帖人之间的群体身份（性别、年龄）相似性会显著影响其社会评判与共情的表达，证实了群体间共情差距在真实网络互动中确实存在。 |
| [^85] | [Let Agents Design How They Search: Automatic Interface Optimization for Agentic Search](https://arxiv.org/abs/2610.29250) | 提出 SearchWeaver 框架，利用执行轨迹和任务奖励自动联合优化语料库产物、可执行工具和使用说明，使搜索界面随具体语料库自适应优化，显著提升答案准确率与检索排序效果。 |
| [^86] | [MOSAIC: An Omni-Modal Benchmark for Long-Term Memory and Cross-Modal Reasoning in Conversational Agents](https://arxiv.org/abs/2610.29249) | MOSAIC是一个包含923个问题、覆盖文本、图像、视频和音频的全模态长期记忆基准，通过冻结读取环节、干预表示、检索和传递的匹配诊断协议，能够精确定位对话智能体在哪个环节丢失了跨模态证据。 |
| [^87] | [Beyond Binary Labels: Diagnostic Evaluation of LLM Safety with Prompt-Level Rubrics](https://arxiv.org/abs/2610.29225) | 提出DiagSafe诊断评估框架，以基于全球AI安全政策的提示级评分细则取代传统二元安全标签，从安全性与有用性两个独立维度对大语言模型进行更细致、更具解释性的双语安全评估。 |
| [^88] | [C$^2$U in RAG: Compositional Concept Unlearning in Retrieval-Augmented Generation](https://arxiv.org/abs/2610.29223) | 本文提出组合式概念遗忘方法C²U，通过将有界查询增量与增强原子库路由相结合，在检索增强生成中有效抑制目标概念暴露，使目标Hit@5降低4.25个百分点且不增加控制查询暴露。 |
| [^89] | [Preference Is Not Permission: Personalizing the Auto-Purchase Line in Agentic Commerce](https://arxiv.org/abs/2610.29216) | 该论文提出将学习个人化的“自动购买边界”作为代理式商务中的新任务，并发现十六个指令微调模型普遍“越权”——即使商品匹配度低、购买依据缺失也会自动下单，与人类委托者的授权意愿存在系统性错位，且这一边界因人而异、需要个性化建模。 |
| [^90] | [Understanding Target Probability Reuse in Speculative Decoding](https://arxiv.org/abs/2610.29210) | 该论文系统研究了投机解码中目标模型下一个token分布的复用问题，证明了精确复用需要下一轮完整上下文中的分布，并提出用当前上下文单独计算的分布重排首token分支或先抽取首token再分配树，从而提高期望接受率并加快解码速度。 |
| [^91] | [norregel: a preregistered benchmark of dated numeric values in Norwegian law](https://arxiv.org/abs/2610.29207) | 该论文提出了norregel——一个预注册的基准测试，用于评估语言模型对挪威法律中随时间变化的数值知识的掌握，结果显示即使是表现最好的模型在主要条目上的正确率也不到一半，揭示了语言模型在法律数值知识方面的重大缺陷。 |
| [^92] | [Reasoning-Search-Augmented LLMs: An Operational Taxonomy, Benchmark Analysis and Controlled Evidence Synthesis](https://arxiv.org/abs/2610.29189) | 该论文提出了一个针对推理-搜索增强大语言模型（LLMs）的操作性分类体系，通过检索耦合、控制器设计、知识来源和接地机制四个架构维度系统性地分析将逐步推理与交互式搜索紧密耦合的框架，并提供了基准分析与受控证据综合。 |
| [^93] | [LTEM: LLM-Friendly Table-Enhanced Markdown for Table Understanding and Generation](https://arxiv.org/abs/2610.29188) | 该论文提出LTEM——一种LLM友好的Markdown最小扩展格式，通过显式编码单元格跨度和层级表头来表达复杂表格结构，在保持轻量级token成本的同时，显著提升了大语言模型对真实文档中复杂表格的理解与生成准确率。 |
| [^94] | [TrainBridge: Agentic NPU Training Optimization Guided by GPU Compilation Evidence](https://arxiv.org/abs/2610.29171) | TrainBridge提出一种智能体化方法，以成熟GPU编译栈的优化执行为参考，从中提取优化意图指导NPU程序重写并结合迭代式性能剖析进行精调，在四个工作负载上平均实现1.7倍的NPU训练加速。 |
| [^95] | [TeachBack: Learning to Teach via Bidirectional Preference Optimization for On-Policy Self-Distillation](https://arxiv.org/abs/2610.29167) | TeachBack提出了一种双向偏好学习框架，使模型不仅从掌握特权信息的教师处学习，还反过来优化教师的教学方式，从而生成更适合学生学习的推理轨迹。 |
| [^96] | [Carving Radiology Reports at Their Evidential Joints: Anatomical Region Sets as Generation Units](https://arxiv.org/abs/2610.29165) | 提出RadCarve方法，直接以放射学报告中现成语句-区域对应关系里的解剖区域集合作为生成单元，将单元划分与视觉证据选择统一，从而生成与视觉证据紧密绑定的细粒度放射学报告。 |
| [^97] | [Why is AI-generated language so hard to understand?](https://arxiv.org/abs/2610.29162) | 该研究发现所有前沿AI模型都比人类写作更频繁地使用增加人类加工难度的心理语言学特征（如低频n-gram、长句法依存、名词堆叠等），并提出这是因为人类语言为成功沟通而优化，而AI生成的文本则服务于其他优化目标。 |
| [^98] | [VeeReview: Measuring Build-Verify Pairs in Agent-Authored Pull Requests from a V-Model Perspective](https://arxiv.org/abs/2610.29143) | 本文提出基于V模型构建-验证对的VeeReview方法，通过仓库信号和语言模型裁判测量拉取请求中的验证证据，发现智能体创作的拉取请求在全部四个验证维度上的“仅构建”率均低于人类拉取请求，与普遍预期相反。 |
| [^99] | [GRAFT: Restoring the Missing Sampling Interface in Continuous Diffusion Language Models](https://arxiv.org/abs/2610.29136) | 论文发现连续扩散语言模型因不进行逐步词级采样（缺失采样接口）而导致稀有词频率严重坍缩（最稀有词仅为人类频率的0.30倍），并提出GRAFT来恢复这一缺失的采样接口以修复词汇长尾问题。 |
| [^100] | [GRAM: Group-wise Rank-Aware Model Merging via Subspace Alignment](https://arxiv.org/abs/2610.29110) | 提出GRAM方法，通过主角度对齐分析将几何兼容的任务分组，并在组内进行秩感知的子空间对齐合并，可证明地降低投影误差，从而解决模型合并中的子空间干扰问题。 |
| [^101] | [Multi-level Encoder with Global Topic for Task-oriented Dialogue Summarization](https://arxiv.org/abs/2610.29097) | 该论文提出MLEGT模型，通过带全局主题的多级编码器捕捉对话的时序与空间依赖关系及全局主题信息，并结合全局主题引导的指针机制来生成更准确的面向任务对话摘要。 |
| [^102] | [Same Answer, Different Reasons: Trace-Level Divergence in Multilingual Large Language Models](https://arxiv.org/abs/2610.29092) | 该论文提出一个轨迹级分歧框架，将多语言大语言模型的可见推理轨迹作为结构化序列分解为结构相似性、顺序一致性和整体相似性进行比较，发现标准的整体嵌入余弦比较方法会系统性高估跨语言的推理一致性。 |
| [^103] | [A Large-Scale Evaluation of Role-Enhanced Zero-Shot Chain-of-Thought Prompting](https://arxiv.org/abs/2610.29073) | 本文提出RE-ZS-CoT提示框架，通过静态与动态专家角色选择机制增强大语言模型的零样本推理能力，在20个模型和七个基准上最高取得44个百分点的性能提升。 |
| [^104] | [Our Own Jev: Typed Decisions in One Forward Pass, from an In-House Judge to Open LoRA Models](https://arxiv.org/abs/2610.29052) | 该论文提出用单次前向传播的“类型化决策”接口（直接读取logits或轻量指针头）替代生成式LLM裁判，其自研System 1裁判仅用一块GPU约3分钟、约0.06美元即可判定1,000个智能体编码会话，以约1%的成本取代每轮约1,000美元且标签一致性仅26.9%的前沿LLM裁判。 |
| [^105] | [Target Regeneration Shrinks the Gains of Draft-Training Objectives in Speculative Decoding](https://arxiv.org/abs/2610.29013) | 研究发现，投机解码中新的草稿训练目标（如令牌级蒸馏和验证门控蒸馏）在外部回复数据上的收益，在经过目标模型贪婪重生成后大幅缩水，且该现象在多种训练与解码设置下均稳定存在。 |
| [^106] | [VeriEvolve: Evaluator-First Evolution for Tool-Using Agents](https://arxiv.org/abs/2610.29005) | VeriEvolve提出“评估器优先”的进化框架，先从任务指令自动合成可执行的评估器和挑战状态作为可靠的适应度信号，再据此进化工具使用智能体，将MCPMark文件系统任务的外部验证成功率从6/30提升至18/30。 |
| [^107] | [FlashNext-RL: A System Optimization Study for Long-Horizon Agentic RL of Qwen3.8-Flash-Next](https://arxiv.org/abs/2610.28998) | 本文提出 FlashNext-RL 框架，通过优化主机卸载、梯度累积以及混合专家、稀疏注意力和门控残差流等专用内核，实现了在单个 8xB200 节点上对 Qwen3.8-Flash-Next 完整 48 层模型的 BF16 长程智能体强化学习训练，显著提升了长序列训练吞吐量。 |
| [^108] | [Language Impairment as a Lens onto Large Language Models](https://arxiv.org/abs/2610.28992) | 该论文提出了一种受失语症启发的机制可解释性框架，通过分类器引导损伤与连续激活引导两种方法系统性地诱发大语言模型的“失语症”式功能崩溃，揭示了其语言功能既可被隔离到特定回路与表征向量、又彼此高度纠缠的双重特性，为理解大语言模型的功能组织结构提供了全新视角。 |
| [^109] | [Scan or Index? large Evidence Coverage, Answer Quality, and Cost in Long-Document RAG](https://arxiv.org/abs/2610.28957) | 该研究比较了11种长文档RAG检索配置，发现混合检索策略在证据覆盖率和答案准确率上均优于全文档扫描，且更高的证据覆盖率并不必然带来更好的答案质量。 |
| [^110] | [Guilt by Association: Semantic Leakage Drives Associative Gender Bias](https://arxiv.org/abs/2610.28944) | 该论文发现语言模型中的语义泄漏是一种广泛的联想倾向——概念无需被归属于特定实体，仅凭其在输入场景中的出现就能引导模型关于该实体的生成内容，并由此驱动联想性性别偏见。 |
| [^111] | [Is Refusal Direction Universal across Low-Resourced Languages?](https://arxiv.org/abs/2610.28897) | 该研究发现拒绝方向的跨语言普适性是分级的而非二元的：来自安全对齐语言的拒绝方向可广泛迁移至低资源语言，而来自强失对齐语言的拒绝方向则无法可靠地调节任何目标语言的拒绝行为。 |
| [^112] | [CSD: Common-Support Distillation for Cross-Tokenizer Knowledge Transfer](https://arxiv.org/abs/2610.28895) | 本文提出CSD框架，通过在每个对齐输入分块处构建教师与学生模型间的公共支撑集，并以单一目标函数匹配分布，解决了词表和分词差异导致的跨分词器logits知识蒸馏对齐难题。 |
| [^113] | [Building AI Research Teams for Autonomous Discovery of Linear Attention Architectures](https://arxiv.org/abs/2610.28887) | 该论文提出构建一支由不同专业背景的持久性AI研究员组成的协作团队，通过平衡探索与利用、让成员基于彼此的实验发现持续迭代，在有限实验预算下自主发现线性注意力架构。 |
| [^114] | [SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges](https://arxiv.org/abs/2610.28886) | SANE是一个基于智能体的自动化框架，能针对特定LLM评判器自动识别其评分能力、生成变异测试配方及预期分数关系，从而将评判器可靠性测试实现全流程自动化，摆脱了人工编写扰动配方和预设评分器类型的限制。 |
| [^115] | [To Manipulate or Not: Opinion Dynamics in LLM Interactions](https://arxiv.org/abs/2610.28875) | 该研究提出受控的LLM-LLM辩论框架并同时追踪显式立场与逐轮行为，发现同模型交互倾向于缩小观点差距而跨模型交互则扩大差距，揭示了不同模型间说服动态的显著差异。 |
| [^116] | [Beyond Correct Answers: How Compositional Cues Shape Visual Judgments](https://arxiv.org/abs/2610.28843) | 该论文通过控制变量方法系统解构结构化提示中各字段标签对视觉绑定任务的影响，发现标签效果因模型而异，且Qwen3在保留提示下的优势主要源于交换提示导致的错误。 |
| [^117] | [KIRA: A Full-Stack Reading Screening System with Dual-Pathway Analysis for Children](https://arxiv.org/abs/2610.28840) | KIRA是一个面向儿童的双通道阅读筛查系统，通过将预训练语音识别模型适配儿童语音（词错误率从13.88%降至8.84%，音素错误率从41.66%降至23.00%），实现了单词级流畅度评分与音素级错误检测的结合，并通过教师界面支持针对性阅读干预。 |
| [^118] | [How to Use Parallel Data for Multilingual Pretraining](https://arxiv.org/abs/2610.28838) | 该论文提出将文档与其翻译拼接在同一训练序列中的平行预训练（PPT）方法，并发现尽管平行数据有诸多益处，PPT相比标准预训练（SPT）并不能持续改善语言理解能力。 |
| [^119] | [Where Should a Stuck Agent Spend Its Next Tokens? Causal Attribution of Inference-Time Resources](https://arxiv.org/abs/2610.28834) | 提出班扎夫资源归因框架，通过受控因子实验因果地量化计算、信息、反馈等推理时资源对修复失败编码任务的边际贡献，发现任务相关规范最有价值、额外计算的价值随模型能力递减，而多数前沿模型却错误地将预算分配给收益甚微的反馈资源。 |
| [^120] | [A Parallel Multi-Dialect Arabic Dependency Treebank](https://arxiv.org/abs/2610.28823) | 该论文发布了首个涵盖五种城市方言的平行阿拉伯语依存树库MADARTB（共5000个句子），揭示了各方言与标准阿拉伯语整体句法相似但局部结构差异显著，并通过实验表明有监督解析器（87.5% LAS）在跨方言依存分析上优于最佳的大语言模型（85.7% LAS）。 |
| [^121] | [LIFT: Scoped Constraints for Evaluating and Training Instruction Following](https://arxiv.org/abs/2610.28819) | 本文提出Lift-Bench指令遵循基准，通过将约束限定在文档特定部分并同时提供主任务约束与指令约束，缓解了强化学习中的奖励破解问题，并配套了可用于训练的Lift-Train数据集。 |
| [^122] | [Combining Long-Term Memory and LLMs Comes at a High Cost: From Prompt Injection to Persistent Compromise](https://arxiv.org/abs/2610.28810) | 本文揭示了现代对话式AI平台的持久记忆功能使提示注入攻击能够跨会话传播（记忆注入），并提出通用威胁模型证明所有NIST攻击目标均可实现，凸显了这一新型安全威胁的严重性与紧迫性。 |
| [^123] | [Where Do Legal Factors Live in LLMs?](https://arxiv.org/abs/2610.28799) | 该研究首次验证大语言模型内部形成了与全部26个CATO法律因素相对应的内部表征，并通过线性探测和稀疏自编码器在多个开源模型中实现了对这些法律因素的解码与稀疏定位。 |
| [^124] | [CALLIOPE: Private Insights into AI Use, Without Embeddings](https://arxiv.org/abs/2610.28797) | CALLIOPE是一个支持差分隐私的自顶向下主题层次发现框架，它不依赖向量嵌入聚类，而是在整个流程中直接利用LLM能力，在高隐私保护程度下比基于嵌入的方法表现更好。 |
| [^125] | [Is Anyone Enforcing the Cartel? What Reinforcement Learning Teaches LLM Agents in Markets](https://arxiv.org/abs/2610.28787) | 该研究提出一种在反事实市场副本中强制智能体作弊以检验合谋执行机制的审计方法，发现RL训练的LLM定价智能体只会学会训练中易尝试且被奖励的行为（被威慑或跟随降价），而无法自发掌握执行卡特尔所必需的高代价价格战惩罚。 |
| [^126] | [Measure First, Then Allocate: Calibrated Rollout Budgets for MCP Agents in Reinforcement Learning](https://arxiv.org/abs/2610.28785) | 提出BRACE方法，利用奖励分歧测量与历史任务证据相结合的相关奖励模型，在分配生成预算前前瞻性预测混合奖励组，显著提升强化学习中MCP工具使用智能体的rollout预算分配效率。 |
| [^127] | [Scaling Therapy Supervision: Assessing Psychotherapy Fidelity Using Large Language Models](https://arxiv.org/abs/2610.28772) | 该研究利用大型语言模型自动化心理治疗保真度评估并生成可操作的督导反馈，摆脱了人工专家评审的瓶颈，实现了治疗督导的可规模化，并在随机对照试验数据上验证了其区分CBT与非CBT会谈的有效性。 |
| [^128] | [SkillEval: Learning Interpretable Ability Profiles of LLMs via Cognitive Diagnosis Models](https://arxiv.org/abs/2610.28767) | SkillEval 结合大语言模型的元认知能力与心理测量学中的认知诊断模型，从基准题目中自动发现可解释的技能，并为大语言模型生成跨模型的技能级能力画像，相比传统平均分评估更可靠、更校准且信息更丰富。 |
| [^129] | [Automating ML for Science: Can Frontier Agents Climb Scientific Hills in the Wild?](https://arxiv.org/abs/2610.28763) | 该论文提出了长时程测试平台 Agent4Sci（涵盖 11 个领域的 20 个科学机器学习基准），发现前沿智能体能在所有基准上超越已发表的最佳模型（Claude Opus 平均相对改进 10.4%），但这些进展几乎完全来自机器学习工程技巧而非研究级的科学洞察。 |
| [^130] | [A Reverse-Gold Theorem: Positive-Data Identification by Substrate-Bounded Contrastive Learners](https://arxiv.org/abs/2610.28747) | 该论文证明了一个“逆向戈德定理”：借助先天递归对比算子自我生成结构受限的负例扰动，并以InfoNCE式对比学习训练的受限评分器，在忠实性条件下可仅凭正数据实现高概率的PAC条件语言识别，且样本复杂度仅为多项式级。 |
| [^131] | [Qwen-Audio-3.1-ASR Technical Report](https://arxiv.org/abs/2610.28711) | Qwen-Audio-3.1-ASR 系列音频-语言模型在改进语音识别的基础上，扩展了说话人分离转录和通用音频理解能力，支持 30 种语言和 16 种中国方言。 |
| [^132] | [Translation and data ethics](https://arxiv.org/abs/2610.28710) | 本章系统介绍了翻译领域的数据伦理，涵盖版权、所有权、偏见等普遍性问题，以及少数语言社区和数字鸿沟相关的特殊伦理问题，并探讨了责任、监管及翻译学界对数据伦理日益增长的关注。 |
| [^133] | [How Far Can the Gemma 4 Speech Encoder Go Without Changing Its Weights? Saudi and Gulf Arabic Speech Recognition with a Frozen and an Adapter-Augmented Encoder](https://arxiv.org/abs/2610.28673) | 在完全不改变 Gemma 4 语音编码器权重的前提下，仅通过轻量级 CTC 读取器或少量低秩适配器参数，就能让沙特和海湾阿拉伯语语音识别达到与参数量大得多的商业系统统计上相当的性能，同时在 CPU 上的推理速度快 3.4 倍。 |
| [^134] | [Beyond source attribution: Address authorship and comparability decisions in an LLM-built scholarly knowledge graph](https://arxiv.org/abs/2610.28663) | 本文针对LLM构建的学术知识图谱提出，必须将用于汇集证据的稳定“工作地址”与文献中相互矛盾的分类类别区分开来，否则检索地址会被误当作唯一分类，导致证据在边界、等级和诊断标准上不可比（以汉语方言学为例）。 |
| [^135] | [TAGR: Task-Adaptive Geo-Semantic Retrieval across Heterogeneous Urban Delivery Tasks](https://arxiv.org/abs/2610.28638) | 提出TAGR框架，通过任务自适应的地理语义检索机制，为城市配送中需求预测、ETA预测和下一站预测三类异构任务检索结构与粒度各异的历史证据，解决单一融合机制导致的匹配不当与负迁移问题，提升预测精度与资源分配效率。 |
| [^136] | [RESEARCHHUB: A Personalized Literature Discovery System for Biomedical Preprints](https://arxiv.org/abs/2610.28636) | 本文提出ResearchHub个性化文献发现系统，其核心检索器prism通过GPT-4o生成查询和混合训练目标适配生物医学预印本，在MedRxiv语料上将Recall@10较SPECTER2-Search提升12.4个百分点，并支持可控新颖性、多样性与课程式排序。 |
| [^137] | [Beyond Model Weights: Sampling-Dependent Fairness in Diffusion Language Models](https://arxiv.org/abs/2610.27584) | 扩散语言模型的社会偏见程度及模型间的偏见排序会随采样配置（如块长度和重掩码策略）显著变化，且即使采用去偏见策略后，这种采样依赖性依然持续存在。 |
| [^138] | [Predicting Linear Relational Decodability Without the Relation](https://arxiv.org/abs/2610.27241) | 该论文提出“实体透镜”这一新颖的雅可比透镜变体，通过在与关系无关的中性提示下针对实体类型进行训练，能够在不使用任何关系特定信息的情况下，显著预测线性关系嵌入（LRE）在不同关系上忠实度的巨大差异。 |
| [^139] | [WaveCommitments: Private and Verifiable Speech-Agent Marketplaces](https://arxiv.org/abs/2610.26867) | 提出WaveCommitments，一种将可信执行环境内基于Data Shapley的机密资产估值与多元多项式承诺相结合的端到端设计，实现了语音智能体市场中私密且可验证的资产交易，相比最佳基准速度提升最高20倍、证明大小缩减最高39倍。 |
| [^140] | [AfriBED: a Speech Recognition Bias Evaluation Dataset for Low Resource African Languages](https://arxiv.org/abs/2610.26821) | 本文提出了面向本巴语和尼扬贾语两种低资源非洲语言的全新语音识别偏见评估数据集AfriBED，可用于检测和估计语音识别系统中的性别偏见与说话人类型偏见。 |
| [^141] | [Copying Under Repetition: From Prefix Matching to Position Hashing](https://arxiv.org/abs/2610.26722) | 本文提出利用小素数周期的VO旋转位置编码并结合中国剩余定理，证明了两层Transformer能在任意重复字符串下实现精确复制并正确终止，同时给出了顺序有限精度下复制的不可能性证明。 |
| [^142] | [Breaking Diffusion: Parallel Reasoning with Draft Refinement](https://arxiv.org/abs/2610.26542) | 该论文提出草稿精炼语言模型（DRLMs），利用单个去噪器并行提出并评估草稿修订，并通过校正自身token证据来衡量token所获得的上下文支持，从而突破传统扩散模型迭代去噪无法保证整体答案变优的局限。 |
| [^143] | [Mitigating Reward Hacking in Preference Reward Models with Evolving Penalties](https://arxiv.org/abs/2610.26496) | 提出Dynalize方法，通过让语言模型在训练过程中异步分析采样结果并动态生成针对不良行为的演化惩罚，在线缓解偏好奖励模型的奖励作弊问题，从而支持更持久的训练并提升开放式任务的下游性能。 |
| [^144] | [DiaLong: Benchmarking Reminiscence in Long-Context Dialogues](https://arxiv.org/abs/2610.25463) | 提出DiaLong基准，包含100个超长人类对话及事实回忆判别任务，用以定量评估大语言模型在长上下文对话中主动保持和检索信息的能力，测试表明现有模型及记忆增强方法仍远未达标。 |
| [^145] | [Acharya 2.0: An AI Tutor for Self-Learning Subjects in Hindi](https://arxiv.org/abs/2610.25192) | 该论文提出了Acharya 2.0，一个基于检索增强生成（RAG）技术的印地语自主学习AI导师系统，旨在克服低资源语言标注数据匮乏和大语言模型适配成本高昂的问题。 |
| [^146] | [HybridResearcher: Bridging the Structured-Unstructured Boundary for Deep Research Agents](https://arxiv.org/abs/2610.24899) | 提出 HybridResearcher——一个 4B 参数的深度研究智能体，通过从已验证的单源任务合成跨来源桥接轨迹来训练，使其学会在结构化数据与非结构化数据的边界之间进行跨源证据推理与整合。 |
| [^147] | [World of SWECraft: Agent Collaboration at Scale via Simulated Software Worlds](https://arxiv.org/abs/2610.24061) | 提出Software World（SWECraft）框架，通过模拟由真实Python软件包构成的软件生态世界来研究多智能体协作，以隐藏依赖包的加速作为统一评估指标，系统考察通信、所有权和资源分配等要素对协作效果与可扩展性的影响。 |
| [^148] | [ProPerAgent: Toward Memory-Driven Proactive Assistance](https://arxiv.org/abs/2610.22671) | 该论文提出ProPerMem基准，首次要求AI助手从长达一年、充满干扰的个人记忆中自主挖掘证据来决定是否及何时主动介入，发现现有模型（含前沿模型）表现极差（最高仅15%），而提供现成证据时性能可达96%，揭示了记忆证据挖掘是主动性助手的核心瓶颈。 |
| [^149] | [WEAVE: Weaving Real-World Code Evolution into Dependency-Structured Tasks for Benchmarking and Training Code Agents](https://arxiv.org/abs/2610.22039) | WEAVE 框架将真实世界中经过验证的多 PR 代码演进历史转化为依赖结构化的可执行任务，并引入基于测试的确定性度量 Graph Credit，以更好地对长程代码智能体进行基准评估和训练。 |
| [^150] | [N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models](https://arxiv.org/abs/2610.21810) | 本文提出条件程序（CP）压缩表示方法，仅用两比特编码按需重构大语言模型n-gram记忆查找表中被请求的行，相比标准2比特量化将输出KL散度降低41–63%，并将DeepSeek的表存储从203GB压缩至51GB。 |
| [^151] | [DialFork: Going Beyond Factuality to Evaluate Procedural Hallucination in Conversational Agents](https://arxiv.org/abs/2610.21561) | DialFork是首个评估对话代理“程序性幻觉”的基准，通过在受控对话轮次位置注入四种幻觉类型构建了涵盖12个领域、24个任务的36,278个对话，并从三个粒度定义了刻画程序性偏差的评估任务。 |
| [^152] | [OPRD: On-Policy Reminder Distillation for Long-Horizon Coding Agents](https://arxiv.org/abs/2610.21496) | OPRD通过多范围提醒监督与基于结果的路由机制，将教师对失败轨迹的再评估能力以反向KL蒸馏方式传递给学生智能体，从而解决长时程编码智能体无法利用上下文信息和过程监督稀疏的问题。 |
| [^153] | [SkillNEO: Non-Elitist Evolutionary Optimization of Agent Skills](https://arxiv.org/abs/2610.20926) | SkillNEO提出了一种非精英主义的智能体技能进化优化方法，通过分离“使用的版本”与“持续改进的版本”的选择，允许得分较低的版本继续接受编辑完善，从而避免因要求立即获得奖励提升而丢弃有价值的多步部分修复。 |
| [^154] | [Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation](https://arxiv.org/abs/2610.20247) | 本文提出了一种感知重复的联合缩放定律，揭示多语言MoE模型中活跃参数、训练词元、专家数量与语言覆盖范围之间的关系，并发现32个专家相比稠密模型可将语言从12种扩展到200种时的损失惩罚降低83%。 |
| [^155] | [On the Monitorability of Natural Language Autoencoders: Identification vs Discrimination](https://arxiv.org/abs/2610.18727) | 论文将自然语言自编码器的训练与监督部署形式化为两种不同的指称博弈，并指出二者之间的差距导致其解释虽能识别对应的激活，却难以让人类监控者从多个近似叙述中辨别出真实的那一个。 |
| [^156] | [Exploring Jailbreak Vulnerability Across Depth-Based Dynamic Inference in Large Language Models](https://arxiv.org/abs/2610.15207) | 本文首次系统研究了基于深度的动态推理对大语言模型越狱鲁棒性的影响，发现浅层退出始终比深层退出对越狱攻击更具鲁棒性，揭示了动态推理灵活性所带来的安全风险。 |
| [^157] | [Toward System One Clinical Intelligence: Parameter-Efficient Fine-Tuning and Counterfactual Explainability of Decision Foundation Models](https://arxiv.org/abs/2610.15029) | 该研究通过QLoRA参数高效微调将基于Qwen3.5的90亿参数决策基础模型Clef-Flash适配于24类疾病症状分类任务，取得95.42%的准确率并显著超越零样本基线与ClinicalBERT，同时结合反事实可解释性提升了临床决策的透明度。 |
| [^158] | [Personalization Without Stereotyping: Do Speech LLMs Know When to Turn a Deaf Ear?](https://arxiv.org/abs/2610.13550) | 提出了 BASIS-Bench 基准，通过人口统计稳健性、刻板印象偏见和个性化三个维度区分语音大语言模型的有益个性化与刻板印象行为，评估发现多数模型虽能识别人口统计线索却未能恰当运用。 |
| [^159] | [REMORY: Learning Residual Memory for Context Compaction](https://arxiv.org/abs/2610.11287) | REMORY 提出一种神经记忆网络，通过生成软记忆token作为文本摘要的“残差”补充，使冻结的LLM仅用 5.2% 的输入位置即可接近全上下文性能，并显著减少重复工具输出和工具错误。 |
| [^160] | [DLoop: Looped Speculative Decoding](https://arxiv.org/abs/2610.07659) | DLoop提出一种循环式推测解码方法，在验证前自适应地执行多个起草阶段，从而减少目标模型不必要的前向验证，提升大语言模型推测解码的加速效果。 |
| [^161] | [Minimal Witness Reinforcement Learning](https://arxiv.org/abs/2610.07226) | 本文提出最小见证强化学习（MWRL），利用基于集合并集覆盖损失的信用分配机制，仅凭单一黑盒验证器的信号即可同时实现解的最小性与多个备选解的恢复。 |
| [^162] | [Scaling Verifiable Environments for Long-horizon Work Agents](https://arxiv.org/abs/2610.04906) | WorkForge 提出了一个可扩展的合成框架，从专家工作流和真实世界资源出发，通过提取可核查的事实锚点，自动构建支持长程交互与可信验证的工作智能体训练环境。 |
| [^163] | [A rubric landscape for evaluating clinical reasoning in large language models: what exists, what is missing, and what needs to be combined](https://arxiv.org/abs/2610.01938) | 该结构化综述系统梳理了用于评估大语言模型临床推理的医学教育工具、临床基准与长文本生成评估方法，覆盖病情表征、时序综合、鉴别与管理推理、反事实推理、不确定性校准和推理忠实性六个维度，并指出目前没有任何单一工具能覆盖全部维度，需要将多种评估方法组合使用。 |
| [^164] | [Can LLMs Normalize Databases? A Benchmark and Multi-Agent Framework for Schema Normalization](https://arxiv.org/abs/2609.11141) | 该论文提出了包含3,275个样本的数据库规范化基准DNBENCH，系统揭示了LLM在函数依赖推理、模式分解和表间约束重建中的常见失败，并进一步提出多智能体框架MARS以提升模式规范化的可靠性。 |
| [^165] | [CPR for LLMs: Critical-Point Routing against Catastrophic Forgetting in Domain Adaptation](https://arxiv.org/abs/2608.30158) | 本文提出CPR框架，通过识别基座模型失败而专家模型成功的临界词元，在基座模型与其SFT专家版本之间进行词元级路由，从模型层面解耦领域能力与通用能力，有效缓解领域自适应中的灾难性遗忘。 |
| [^166] | [From Atomic to Agentic: Towards Interpretable Evaluation of LLMs' Agentic Mathematical Capabilities](https://arxiv.org/abs/2608.26950) | 本文提出一个过程级数学基准，通过将智能体行为与原子能力分类对齐，并利用自动化流水线生成细粒度注释，实现可解释地评估LLMs的智能体数学推理能力。 |
| [^167] | [Proxy reliance in large language model decisions is uncalibrated to predictive evidence](https://arxiv.org/abs/2608.22887) | 本文通过精确计算临床任务中的因果代理效应，发现LLM对代理的依赖未与预测证据校准，表现为过度依赖、合理依赖或不足依赖，且社会标签抑制效果脆弱。 |
| [^168] | [On the Role of Citations in Preference Data](https://arxiv.org/abs/2608.21376) | 本文通过混合效应模型研究科学问答中引用对人类和LLM偏好判断的影响，发现人类偏好多样化但数量少的引用，而LLM的引用偏好虽存在但受数据和模型影响。 |
| [^169] | [What Transfers from Text to Vision? Capability Scaling Laws and Transfer Dynamics for VLMs](https://arxiv.org/abs/2608.00013) | 我们提出了首个跨家族的多模态缩放规律，通过文本能力得分直接预测VLM性能，并在150多个VLM上验证了其有效性。 |
| [^170] | [The Best Programming Language for Tokenmaxxing: An Investigation of Coding Agent Behavior Across Programming Languages](https://arxiv.org/abs/2607.22807) | 编程智能体的Token消耗因编程语言而显著不同且跨模型一致，其原因是智能体在不熟悉的语言中反复生成无法编译的代码并重复修改已通过的解决方案。 |
| [^171] | [DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making](https://arxiv.org/abs/2607.20491) | 该论文提出DFAH-Bench基准，通过同时衡量决策一致性与工具路径一致性，揭示金融智能体在决策结果高度稳定（约95%）的情况下，其背后的工具执行过程却存在显著不稳定性（一致性仅约50%）。 |
| [^172] | [MARS: Multi-hop Adaptive Retrieval and SPARQL Generation for KGQA](https://arxiv.org/abs/2607.14561) | MARS提出了一种无需微调的知识图谱问答方法，通过结构化的多跳自适应检索过程，在每一步决定继续图遍历还是生成最终SPARQL查询，从而根据问题自适应调整检索深度。 |
| [^173] | [Exploring Post-Training Alignment of Small Language Models for Biomedical Data-to-Text Generation: A Case Study of Medication Leaflet](https://arxiv.org/abs/2607.13430) | 本研究系统比较了SFT、DPO、ORPO和GRPO四种后训练对齐方法在小语言模型药品说明书生成任务上的效果，并构建了FDALeaflets数据集以评估跨监管体系的零样本泛化能力，发现对齐后的小语言模型在多项指标上超越了GPT-5。 |
| [^174] | [Flout at Your Own Risk: LLMs Struggle with Pragmatic Cooperativity Under Epistemic Asymmetry](https://arxiv.org/abs/2607.11053) | 本文形式化了“协作性认知不对称”概念并证明完全可执行的指令需遵循格赖斯合作原则，同时通过实证评估发现大语言模型在信息不完整的多方协作中无论作为说话者还是倾听者均存在语用沟通缺陷，且格赖斯准则可预测其不同失败模式。 |
| [^175] | [Trees from Marginals: Autoregressive drafting with factorized priors](https://arxiv.org/abs/2607.06763) | 本文提出Weaver，一种从因子化草稿模型的top-K边缘分布构建提议树的轻量级自回归适配器，恢复了token间的条件依赖，并结合无需回滚的树验证算法与优化的CUDA内核，相比自回归解码实现了4.08倍加速。 |
| [^176] | [When Does Tool Use Increase the Expressive Power of Finite-Precision Recurrent Models?](https://arxiv.org/abs/2607.06155) | 本文证明了工具使用提升模型表达能力的边界条件：有限精度循环模型调用有界状态的工具后仍是有限状态系统，但只要接入一条支持本地读/写/移动命令的无界磁带即可达到图灵完备。 |
| [^177] | [Latent Bridges for Multi-Table Question Answering](https://arxiv.org/abs/2606.28916) | 该论文提出GRAB框架，将关系数据编码为异构图，并通过查询条件化的潜在标记桥接到完全冻结的LLM（仅训练91M参数的轻量模块），显著提升了多表格问答的性能。 |
| [^178] | [Output Vector Editing for Memorization Mitigation in Large Language Models](https://arxiv.org/abs/2606.18767) | 提出输出向量编辑（OVE）方法，通过两阶段定位神经元并用全词表最小二乘更新编辑其输出向量（而非将激活置零），有效抑制大语言模型复现记忆的训练文本，同时更好地保留模型其他能力。 |
| [^179] | [EHRNote-ChatQA: A Benchmark for Evidence-Grounded Multi-Turn Clinical Question Answering over Longitudinal Discharge Summaries](https://arxiv.org/abs/2606.15735) | 该论文提出了首个基于患者多篇出院总结的证据支撑型多轮临床问答基准EHRNote-ChatQA，利用去标识化的MIMIC-IV数据构建，真实模拟了医学专家迭代综合信息并验证证据的临床审阅场景，弥补了现有基准在多轮问答与证据溯源评估上的不足。 |
| [^180] | [GENIE: A Fine-Grained Measure for Novelty](https://arxiv.org/abs/2606.12790) | 本文提出细粒度评估指标GENIE，基于任务特定特征相对于响应群体来衡量模型生成内容的新颖性，克服了整体性指标难以捕捉新颖性高维特性的局限，并可有效评估创造力提升方法的效果。 |
| [^181] | [Rethinking Synthetic Rationale Supervision for Disease Prediction](https://arxiv.org/abs/2606.10279) | 该研究发现，在疾病预测中使用合成推理理由进行“推理优先”监督微调反而会持续且显著地损害预测性能，且这种退化跨模型系列、数据规模和任务普遍存在，并非由推理理由质量不佳所致。 |
| [^182] | [Linguistic Productivity in Large Language Models: Models Coerce, but do not Preempt](https://arxiv.org/abs/2606.02953) | 该研究发现大型语言模型具备促进语言能产性的“强制解读”推理能力，但缺乏人类语言使用者用来约束能产性的“先占”机制。 |
| [^183] | [CanLegalRAGBench: Evaluating Retrieval-Augmented Generation on Canadian Case Law](https://arxiv.org/abs/2605.30497) | 本文提出了CanLegalRAGBench，一个基于现实查询和专家标注的加拿大判例法问答基准，揭示了检索设计敏感性、开源模型竞争力及自动评估的局限性。 |
| [^184] | [Word-Class and Construction-Like Structure Emerges in Neural Successor Representations Trained on Natural Language](https://arxiv.org/abs/2605.24585) | 在自然语言上训练的神经后继表示无需任何词性监督，即可自发涌现出词类边界以及跨越传统类别界限的构式类语言结构。 |
| [^185] | [MedicalBench: Evaluating Large Language Models Toward Improved Medical Concept Extraction](https://arxiv.org/abs/2605.20197) | MedicalBench是一个具有证据扎根的医学概念抽取基准，将概念抽取转化为笔记-概念对验证任务，专门评估大语言模型对隐含医学概念的推理能力。 |
| [^186] | [From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents](https://arxiv.org/abs/2605.15104) | 提出一个无需重新标注、可复现且可验证的框架，利用文本转语音、说话人变化和环境噪声将已有文本工具调用基准转换为可控的音频评估，并揭示全模态模型的工具调用性能高度依赖模型和任务，且从文本到语音存在1.8至4.8分的性能下降。 |
| [^187] | [NodeSynth: Socially Aligned Synthetic Data for AI Evaluation](https://arxiv.org/abs/2605.14381) | NodeSyn通过将微调的分类体系生成器与可信文献来源和敏感属性相结合，生成反映真实社会背景的敏感领域评估数据，其诱发的模型失败率达到通用合成基线的3.1倍、真实人类查询的1.6倍，显著优于现有合成数据方法。 |
| [^188] | [PersonalAI 2.0: Enhancing knowledge graph traversal/retrieval with planning mechanism for Personalized LLM Agents](https://arxiv.org/abs/2605.13481) | PersonalAI 2.0通过引入动态多阶段查询处理流水线的规划机制，实现了由实体、图顶点和线索查询引导的自适应迭代式知识图谱检索，在多个问答基准上显著提升了生成答案的事实准确性。 |
| [^189] | [Adaptive Steering and Remasking for Safe Generation in Diffusion Language Models](https://arxiv.org/abs/2605.13043) | 该论文提出一种推理时防御框架，通过门控方向自适应调节引导强度，并结合安全感知重掩码机制，动态引导扩散语言模型的去噪轨迹远离有害内容，同时保持生成质量。 |
| [^190] | [Deep Reasoning in General Purpose Agents via Structured Meta-Cognition](https://arxiv.org/abs/2605.11388) | 提出“深度推理”方法，利用形式化语言将元推理表示为可执行的分解，在推理时动态构建任务特定的脚手架，使LLM智能体能像人类一样灵活地在联想推理、形式化计算和递归求解等不同推理模式间切换。 |
| [^191] | [From prompting to evidence-based translation: A RAG+prompt system for Japanese-Chinese translation and its pedagogical potential](https://arxiv.org/abs/2605.03387) | 该研究提出了一种RAG+提示词翻译系统，通过融入语言学分析和相似例句检索来增强GPT-4o，在不修改基础模型的前提下，将含名词修饰节结构的日中翻译BLEU得分提升了23.4%，并展示了其教学应用潜力。 |
| [^192] | [A Robust Evaluation of Probe Robustness: Lessons for Reliable OOD Uncertainty Quantification](https://arxiv.org/abs/2604.11662) | 提出ProbeDrift系统性评估框架，通过训练2000多个不确定性探针揭示现有方法在远分布外设置下鲁棒性较差，且鲁棒性取决于分布内难以察觉的设计决策（如特征类型、聚合策略和训练信号）。 |
| [^193] | [The Stepwise Informativeness Assumption: When Do Entropy Dynamics Track Reasoning Correctness in LLMs?](https://arxiv.org/abs/2604.06192) | 本文提出步进信息性假设（SIA），从理论上刻画了熵降低何时能真实反映大语言模型推理的正确性进展，并指出这种对齐取决于训练机制将推理轨迹与正确答案耦合的方式，同时给出了该现象的可观测特征和失效情形。 |
| [^194] | [AfrIFact: Cultural Information Retrieval, Evidence Extraction and Fact Checking for African Languages](https://arxiv.org/abs/2604.00706) | 本文提出了AfrIFact数据集，覆盖十种非洲语言和英语的自动事实核查全流程（信息检索、证据提取与事实核查），并揭示现有嵌入模型缺乏跨语言检索能力、大语言模型在非洲语言事实验证上表现不佳，而少样本提示可将性能提升高达43%。 |
| [^195] | [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626) | 这篇综述论文通过学生生成状态、教师信息与策略更新三者之间的相互作用，建立了在线策略蒸馏（OPD）的统一分析框架，并将其与自蒸馏、多教师集成、强化学习和交互式智能体等方向联系起来，系统阐释了不同方法的设计选择对能力迁移、训练效率和部署的影响。 |
| [^196] | [MICA: Multi-granularity Intertemporal Credit Assignment for Long-Horizon Emotional Support Dialogue](https://arxiv.org/abs/2603.06194) | 提出无评论家强化学习框架MICA，通过共享势函数同时生成即时与延迟两类信用信号，实现情感支持对话中无需状态匹配的稳定逐轮优化。 |
| [^197] | [StructLens: A Structural Lens for Language Models via Maximum Spanning Trees](https://arxiv.org/abs/2603.03328) | 提出StructLens框架，受依存句法树启发，通过在残差流语义表示上构建最大生成树来揭示语言模型表示的整体内部结构，并发现中间层具有最强的局部跨度组织特性。 |
| [^198] | [GLUScope: A Tool for Analyzing GLU Neurons in Transformer Language Models](https://arxiv.org/abs/2602.23826) | GLUScope是一个开源工具，通过分析基于GLU激活函数的语言模型中神经元门控与输入激活的四种正负符号组合（激活象限），帮助可解释性研究者揭示神经元促进或抑制特定token等截然不同的功能行为。 |
| [^199] | [Analyzing and Improving Chain-of-Thought Monitorability Through Information Theory](https://arxiv.org/abs/2602.18297) | 本文用信息论揭示了思维链可监控性的必要非充分条件，识别出信息鸿沟和引导误差两类失效来源，发现思维链混淆可在无优化压力下自发出现，并证明可通过针对性干预系统性缓解监控的脆弱性。 |
| [^200] | [TabAgent: Distilling LLM Agent Components into Cheap Classifiers](https://arxiv.org/abs/2602.16429) | TabAgent框架通过将智能体系统中反复调用的LLM生成式决策组件替换为基于执行轨迹训练的轻量级文本-表格分类器，在保持任务成功率的同时显著降低部署延迟和成本。 |
| [^201] | [Controlling Output Rankings in LLM-Based Search via Generative Engine Optimization](https://arxiv.org/abs/2602.03608) | 提出了CORE优化方法，通过向搜索引擎返回的检索内容附加策略性设计的优化内容，来控制基于大语言模型的生成式搜索中的输出排名，从而提升小企业和独立创作者产品的可见性。 |
| [^202] | [A Theoretical Analysis of Why Masked Diffusion Models Mitigate the Reversal Curse](https://arxiv.org/abs/2602.02133) | 该论文通过理论分析证明，掩码扩散模型缓解反转诅咒的原因在于正向与反向位置条件分布之间存在参数级耦合——共享的Transformer参数存储词元对证据，而相对位置编码在不改变值的情况下引导注意力，从而使学到的证据能够迁移到反向提示。 |
| [^203] | [Disentangling generalization and memorization in large language models using chess](https://arxiv.org/abs/2601.16823) | 该论文提出以国际象棋作为受控测试平台，通过按相关先验密度构建棋局分类体系，在无需了解模型训练数据的情况下区分大语言模型的记忆能力与真正的泛化推理能力，并发现模型性能随相关先验密度降低而持续显著下降。 |
| [^204] | [Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents](https://arxiv.org/abs/2601.15322) | 该论文提出DFAH框架，首次将工具使用型LLM智能体的决策可重复性、轨迹一致性和证据条件忠实性作为三个独立维度加以区分评估，并澄清了历史研究中r=-0.11相关性统计量的性质与局限。 |
| [^205] | [RubRIX: Rubric-Driven Risk Mitigation in Caregiver-AI Interactions](https://arxiv.org/abs/2601.13235) | 本文提出了RubRIX——一个基于关怀伦理理论、经临床医生验证的评分量表框架，将LLM在照护者交互场景中的回应风险操作化为五个维度，并在超过2万条真实照护者查询上验证，量表引导的改进可将风险成分降低45-98%。 |
| [^206] | [Algorithmic Reflectivity in Large Language Models: Methodological Study of Linguistic Structure in Mentalization](https://arxiv.org/abs/2512.08945) | 本研究通过让五位MBT受训精神科医生盲评GPT-4.1生成的心智化画像，证明大语言模型无需体验与情感过程，也能生成在语言结构上与心智化治疗维度一致、并被临床专家认可为形式上心智化的内容。 |
| [^207] | [Edit Policy-Aligned Text Simplification](https://arxiv.org/abs/2512.06228) | 该论文提出利用大语言模型作为裁判（LLM-Judge）自动构建与简化策略对齐的训练数据，无需人工标注或平行语料库，即可训练出在句子和文档层面适应多种简化策略的模型，且小规模开源模型的性能可媲美甚至超越GPT-5。 |
| [^208] | [ClusterFusion: Hybrid Clustering with Embedding Guidance and LLM Adaptation](https://arxiv.org/abs/2512.04350) | ClusterFusion提出了一种以大语言模型为聚类核心、由轻量级嵌入方法引导的三阶段混合聚类框架（嵌入引导子集划分、LLM主题总结与LLM主题分配），无需昂贵微调即可融入领域知识，并在多个公开基准和领域数据集上取得最先进的文本聚类性能。 |
| [^209] | [Epistemic Substitution: How Grokipedia's AI-Generated Encyclopedia Restructures Authority](https://arxiv.org/abs/2512.03337) | 该研究通过对比分析1,000对维基百科与Grokipedia文章的近60万条引用，揭示了AI生成的百科全书显著减少了对学术和新闻来源的依赖，转而更多引用用户生成内容及智库出版物，表明AI正在从基础上重构百科全书知识生产的认知权威结构。 |
| [^210] | [PRISM: Prompt-Refined In-Context System Modelling for Financial Retrieval](https://arxiv.org/abs/2511.14130) | PRISM 提出一种无需训练的金融文档检索框架，通过组合提示工程、上下文学习与候选精炼三个可部署模块，将 FiQA-2018 的 Recall@100 从 0.7783 提升至 0.8326 并降低约 35% 的每查询成本，同时系统性地揭示了 k-倒数编码与多智能体协作等额外复杂性何时不再带来收益的边界条件。 |
| [^211] | [Protecting De-identified Documents from Search-based Linkage Attacks](https://arxiv.org/abs/2510.06383) | 本文提出一种两步防御方法：先构建N-gram倒排索引以高效识别文档中易被搜索关联的罕见片段，再利用大语言模型迭代重写这些片段，在保持文本语义完整性的同时有效防止去标识化文档被溯源关联。 |
| [^212] | [The Answer Lies Within: Self-Derived Rewards Enable Explainable Relation Extraction](https://arxiv.org/abs/2510.06198) | 该论文提出COGRE认知启发推理框架与HIT@DICT强化学习中间奖励策略，利用从正确预测中自动提取的信用词典派生奖励，引导大模型推理聚焦关系相关语义并与人类标注的抽象层次对齐，从而同时提升单样本关系抽取的准确率和解释质量。 |
| [^213] | [Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures](https://arxiv.org/abs/2509.25045) | 提出超维探针——一种结合向量符号架构与神经探测的混合式有监督探针，统一了有监督探针、稀疏自编码器和logit分析等可解释性方法，从而更全面地解码大语言模型的内部表示。 |
| [^214] | [From Value Conditioning to Behavioral Shift: Lightweight Value Alignment of LLMs](https://arxiv.org/abs/2508.11414) | 本文提出一种轻量级的大语言模型价值对齐方法，仅通过对价值调查回答进行微调，即可选择性地改变模型的目标价值倾向，并将这种价值转变迁移到域外的下游行为中。 |
| [^215] | [PRiME: Pragmatic Reasoning In MEntal health](https://arxiv.org/abs/2507.23247) | 该论文提出了PRiME任务与数据集，系统评估了12个大语言模型在高风险心理健康领域的语用推理能力，发现模型表现可与人类评估相当，其中Mistral-7B最佳，并通过错误不对称比率和注意力回溯实验深入分析了模型的架构行为。 |
| [^216] | [Effects of Theory of Mind and Prosocial Beliefs on Steering Human-Aligned Behaviors of LLMs in Ultimatum Games](https://arxiv.org/abs/2505.24255) | 本论文通过2700次模拟实验证明，在最后通牒博弈中，心智理论推理能显著增强大语言模型行为与人类规范的对齐、决策一致性和谈判结果，且优于单纯推理模型的表现。 |
| [^217] | [Neither Stochastic Parroting nor AGI: LLMs Solve Tasks through Context-Directed Extrapolation from Training Data Priors](https://arxiv.org/abs/2505.23323) | 该论文提出大语言模型的行为既非“随机鹦鹉学舌”也非“高级推理”，而是“上下文导向外推”——即模型从训练数据先验外推并借助上下文选择合适的先验，同时提供了一个可操作的评估清单，用于区分记忆、基本外推与真正的应用层面泛化。 |
| [^218] | [Human-centered evaluation of data visualization understanding in vision-language models](https://arxiv.org/abs/2505.17202) | 该论文提出了一种以人为中心的评估协议，通过在相同的数据可视化理解任务上直接比较视觉语言模型与人类的反应模式，弥补了以往仅依赖准确率评估的不足。 |
| [^219] | [AnnoCaseLaw: Legal Annotation Benchmark of Civil Negligence](https://arxiv.org/abs/2503.00128) | 提出了AnnoCaseLaw基准——一个包含471份美国民事过失法院意见书和超过12,000个专家标注的密集标注数据集，用于评估语言模型复现法律专家标注的能力，发现模型在“事实”和“结果”类别上表现较好，而在“法律对事实的适用”等需要复杂法律推理的类别上仍存在挑战。 |
| [^220] | [Human Cognitive Benchmarks Reveal Foundational Visual Gaps in MLLMs](https://arxiv.org/abs/2502.16435) | 该论文提出VisFacor基准，将认知心理学评估FRCT中的20个视觉子测试数字化，以评估多模态大语言模型的基础视觉认知能力，发现即使是最好的模型也仅得55.9%，揭示了当前模型在基础视觉能力上存在显著差距。 |
| [^221] | [Multi-Label Sentiment Analysis on 100 Languages with Dynamic Weighting for Label Imbalance](https://arxiv.org/abs/2008.11573) | 该论文提出了一种动态加权方法来应对多标签情感分析中的标签不平衡问题，并结合适配的多标签焦点损失与线性时间复杂度的最优阈值选择方法，在100种语言的情感分析任务中于9个指标中的7个上达到了最先进水平。 |

# 详细

[^1]: LongZero：从零数据出发增强大语言模型的长上下文推理能力

    LongZero: Enhancing Long-Context Reasoning in LLMs from Zero Data

    [https://arxiv.org/abs/2610.30132](https://arxiv.org/abs/2610.30132)

    LongZero提出了一种无需任何外部数据的自举式强化学习方法，让大语言模型通过自行合成结构化的问题-答案-证据任务来生成训练数据，从而有效提升自身的长上下文推理能力。

    

    强化学习（RL）是提升大语言模型（LLM）长上下文推理能力的有效途径，但它需要长篇、多样、具有挑战性且可验证的训练数据。人工标注此类数据成本高昂，自然存在的数据又过于稀缺而无法支撑强化学习，而现有的数据合成方法依赖外部语料库或更强的教师模型，随着被训练模型能力的提升，这些依赖越来越难以获得。最近，从零数据出发的强化学习——即模型从自身提出的任务中学习——取得了出色的效果，但现有方法仅针对短上下文任务。因此，我们提出了LongZero，一种让LLM在无需任何外部数据的情况下提升自身长上下文推理能力的方法。LongZero不是在现有文档上编写问题和答案，而是首先让模型以紧凑的结构化形式合成问题、答案及其支持证据，并……

    arXiv:2610.30132v1 Announce Type: new  Abstract: Reinforcement learning (RL) is an effective way to strengthen the long-context reasoning of large language models (LLMs), but it needs training data that are long, diverse, challenging, and verifiable. Human annotation of such data is expensive, naturally occurring data are too scarce to sustain RL, and existing synthesis methods depend on external corpora or stronger teacher models, which become harder to obtain as the trained models get stronger. Recently, RL from zero data, in which a model learns from tasks it proposes itself, has achieved excellent performance, but existing methods target short-context tasks. Therefore, we propose $\textbf{LongZero}$, a method that lets an LLM improve its own long-context reasoning without any external data. Instead of writing questions and answers over existing documents, LongZero first has the model synthesize the question and answer with its supporting evidence in a compact structured form, and o
    
[^2]: 在关键之处深入思考：面向序列推荐的物品级自适应递归

    Think Harder Where It Matters: Item-Level Adaptive Recursion for Sequential Recommendation

    [https://arxiv.org/abs/2610.30128](https://arxiv.org/abs/2610.30128)

    提出LoopRec，一种针对序列推荐的物品级自适应递归方法，根据每个物品的实际计算需求动态分配递归深度，取代固定计算预算，从而既节省计算资源又提升整体推荐质量。

    

    现有的序列推荐模型通常为用户交互历史中的每个物品分配固定的计算预算，而不考虑该物品对最终预测的贡献。我们通过两项诊断性发现对这一事实上的设计提出挑战。（1）计算需求的不均匀性：同一序列中不同物品所需的计算量差异极大，大多数物品经过浅层处理即可收敛，而少数关键物品只有在完整深度下才能稳定。（2）固定分配的次优性：即使能够理想地获取目标物品信息，非均匀的深度分配也始终优于固定分配，这表明现有的均匀计算既在已收敛的物品上浪费了计算能力，又牺牲了整体推荐质量。这些观察促使计算方式从固定深度转向物品级自适应推理。为此，我们提出了LoopRec。LoopRec递归地应用一个权重共享的单一模块……（原文摘要在此处被截断）

    arXiv:2610.30128v1 Announce Type: cross  Abstract: Existing sequential recommenders usually allocate a fixed computational budget to every item in a user's interaction history, regardless of the item's contribution to the final prediction. We challenge this de facto design with two diagnostic findings. (1) Non-uniform computational need: items within the same sequence vary drastically in how much computation they require. Most converge after shallow processing while a critical minority only stabilizes at full depth. (2) Suboptimality of fixed allocation: given oracle access to the target item, a non-uniform depth assignment consistently outranks the fixed one, indicating that existing uniform computation both wastes capacity on already-converged items and sacrifices overall recommendation quality. These observations motivate a shift from fixed-depth computation to item-level adaptive reasoning. To this end, we propose LoopRec. LoopRec recursively applies a single weight-shared block an
    
[^3]: 环境对决：大语言模型协同进化的缩影

    Environment Duels: A Microcosm of LLM Co-Evolution

    [https://arxiv.org/abs/2610.30116](https://arxiv.org/abs/2610.30116)

    提出“环境对决”这一动态智能体基准，让大语言模型同时担任环境创作者与任务解决者两种角色，发现解决与创作是两种不同的能力，并利用模型间两两对决的传递性结果刻画了模型协同进化的稳定生态图景。

    

    通过从合成训练管线到开放网络共享数据等纷繁的真实世界互动，前沿大语言模型会吸收其前代模型与竞争者所创造的内容。那么在这一生态系统中，每个模型作为创造者和作为解决者分别扮演得如何？我们提出了 Environment Duels（环境对决），这是一个动态的智能体基准，用于在交互式环境中评估上述两种能力。每个模型负责编写环境，每个环境附带一条特权提示，其目标是设计出这样的任务：有能力的解决者在没有提示时失败，而在获得提示后通过。在九个前沿大语言模型上的实验表明，解决与创作是两种截然不同的技能：最强的解决者并非最强的创作者；而较小的模型更能从其他模型的解题方式中学习，并能写出最难的环境和最有用的提示。超越分数本身，模型之间的两两互动构成了模型间相互影响方式的缩影。一对一的对决结果具有传递性，因此能够得出一个稳定的排名……

    arXiv:2610.30116v1 Announce Type: new  Abstract: Through a tangle of real-world interactions, from synthetic training pipelines to data shared on the open web, frontier LLMs absorb what their predecessors and competitors create. How well does each model play its role in this ecosystem, as a creator and as a solver? We introduce Environment Duels, a dynamic agentic benchmark that evaluates both capabilities on interactive environments. Each model writes environments, each with a privileged hint, aiming for tasks that capable solvers fail without the hint and pass with it. Across nine frontier LLMs, solving and authoring prove to be distinct skills: the strongest solvers are not the strongest authors, while smaller models learn more from how others played and write the hardest environments and the most useful hints. Beyond the scores, the pairwise interactions form a microcosm of how models affect one another. Head-to-head outcomes are transitive and so yield a stable ranking; independen
    
[^4]: Quant-Arena：面向量化交易研究的LLM智能体基准测试

    Quant-Arena: Benchmarking LLM Agents for Quantitative Trading Research

    [https://arxiv.org/abs/2610.30113](https://arxiv.org/abs/2610.30113)

    提出了Quant-Arena——首个基于真实A股数据的端到端量化研究基准，让LLM智能体仅凭2022-2023年反馈冻结策略、在保密的2024年数据上评估，以检验其能否完成从市场假设到真实有效策略的完整量化研究循环。

    

    量化研究是一个迭代循环：研究者提出市场假设，将其实现为可执行的信号，在历史数据上进行测试，并决定保留哪些内容。现有基准测试仅对公式、代码或单笔交易决策进行评分，尚未回答大型语言模型（LLM）能否完成这一完整循环，并产生在研究期之外依然有效的策略。我们提出了Quant-Arena，这是一个基于真实A股市场数据的端到端量化研究基准与公开排行榜。模型需要构建特征和股票评分代码，仅获得2022至2023年的数据反馈，并冻结一个策略以在保密的2024年数据上进行评估。现金账户评估器会应用交易约束和交易成本，包括基于模拟订单簿估计的市场冲击。评分以已实现的超额收益为主，并以信号质量和策略所用特征的多样性作为补充。我们评估了九个（模型）……（原文摘要至此截断）

    arXiv:2610.30113v1 Announce Type: new  Abstract: Quantitative research is an iterative loop: a researcher proposes a market hypothesis, implements it as an executable signal, tests it on historical data, and decides what to keep. Existing benchmarks score formulas, code, or individual trading decisions, leaving open whether large language models (LLMs) can complete this loop and produce strategies that work beyond the research period. We introduce Quant-Arena, a benchmark and public leaderboard for end-to-end quantitative research on real A-share market data. A model constructs features and stock-scoring code, receives feedback only on 2022--2023, and freezes one strategy for assessment on the sealed year 2024. A cash-account evaluator applies trading constraints and costs, including market impact estimated from a simulated order book. Scores are dominated by realized excess return and complemented by signal quality and the diversity of features used by the strategy. We evaluate nine f
    
[^5]: 问题真的变难了吗？通过对抗性多智能体重写加固的P与NP完全配对问题测试大语言模型

    Did the Problem Actually Get Harder? Testing LLMs on Paired P and NP-Complete Problems Hardened by Adversarial Multi-Agent Rewriting

    [https://arxiv.org/abs/2610.30103](https://arxiv.org/abs/2610.30103)

    该论文提出P2NPCBench基准，将NP完全问题与仅有一个结构细节之差的P类孪生问题配对，并利用对抗性多智能体重写框架消除教科书式记忆线索，以检验大语言模型是真正具备区分问题复杂度类的推理能力还是仅仅依赖记忆。

    

    许多NP完全（NPC）问题都存在一个多项式时间（P）的“孪生”问题，两者仅在单个结构细节上有所不同，例如对图进行3着色与2着色。一个真正进行推理而非依赖记忆的模型，应该能够辨别问题处于这一边界的哪一侧，然而现有的基准测试要么固定了每个任务的措辞，要么忽略了其复杂度类，因此无法将推理与记忆区分开来。我们提出了P2NPCBench，这是一个包含12个问题族的基准，其中每个NPC问题都与一个在同一表面形式下提出的P孪生问题配对。为了消除使模型能够进行记忆的教科书式线索，我们使用AgentGAN构建了该基准——这是一个对抗性多智能体框架，其中生成器不断重写每个题目描述，直到一组冻结的代码模型无法解答为止，同时由一个可执行的有效性门控确保每次重写都保持答案与复杂度类不变。最终产生了2,114个经过加固的题目，并配有经人工验证的参考人类求解器。通过评估12个专有和开源模型，我们发现每个（原文摘要在此处截断）

    arXiv:2610.30103v1 Announce Type: new  Abstract: Many NP-complete (NPC) problems have a polynomial-time (P) twin that differs in a single structural detail, such as 3- versus 2-colouring a graph. A model that reasons rather than recalls should notice which side of this boundary a problem falls on, yet existing benchmarks either fix the wording of each task or ignore its complexity class, so they cannot separate the two. We introduce P2NPCBench, a benchmark of 12 problem families in which every NPC problem is paired with a P twin posed over the same surface. To remove the textbook cues that enable recall, we build it with AgentGAN, an adversarial multi-agent framework in which a generator rewrites each statement until a panel of frozen code models fails, while an executable validity gate keeps every rewrite answer- and class-preserving. The result is 2,114 hardened statements with hand-verified reference human solvers. Evaluating 12 proprietary and open-source models, we find that every
    
[^6]: VX-One：面向实时边界分流的有类型、经校准的决策头——关于未知攻击诚实性、级联灰色地带与漂移下校准的实证研究

    VX-One: Typed, Calibrated Decision Heads for Real-Time Perimeter Triage - An empirical study of unknown-attack honesty, cascade gray zones, and calibration under drift

    [https://arxiv.org/abs/2610.30090](https://arxiv.org/abs/2610.30090)

    本文提出VX-One，一个将商用有类型决策原语（校准门控、含显式未知选项的闭集选择、序数分数）作为输出的小型多头网络，在CPU上以毫秒级延迟实现实时边界分流，并实证评估了未知攻击诚实性、级联灰色地带与漂移下的校准性能。

    

    边界分流面临延迟与语义之间的权衡：签名引擎快速且格式安全，但对未见过的攻击无能为力，而生成式模型虽具备语义理解能力却速度缓慢且会产生无法解析的输出。2026年末，一类商用System One决策模型（TypeSafe Jev；Microsoft-Decision-1）开始将有类型的决策原语——二值门控、闭集选择、序数分数——作为产品API出售，但学术界尚无相关研究。我们提出VX-One：一个小型多头网络（参数量10万至1000万），其输出即为这些原语——一个经校准的门控、一个在始终包含显式未知选项的封闭候选集上的选择，以及一个序数分数，采用多任务方式训练，并对每个头进行温度校准（CPU上p50延迟约为0.7-1.7毫秒）。我们在UNSW-NB15、CIC-IDS2017、EMBER2024数据集上将其与经校准的XGBoost、随机森林、MLP、Kitsune进行对比评估，并测量了一个两阶段级联系统（配备52,674条Emerging-Threats规则的Suricata -> VX-One）……

    arXiv:2610.30090v1 Announce Type: cross  Abstract: Perimeter triage faces a latency/semantics trade-off: signature engines are fast and format-safe but blind to unseen attacks, while generative models are semantic but slow and produce unparseable output. In late 2026 a class of commercial System One decision models (TypeSafe Jev; Microsoft-Decision-1) began selling typed decision primitives - binary gates, closed-set choices, ordinal scores - as product APIs, with no academic treatment. We present VX-One: a small multi-head network (100K-10M parameters) whose outputs are these primitives - a calibrated gate, a choice over a closed candidate set that always includes an explicit unknown option, and an ordinal score, trained multi-task, temperature-calibrated per head (~0.7-1.7 ms p50 on CPU). We evaluate it on UNSW-NB15, CIC-IDS2017, EMBER2024 against calibrated XGBoost, Random-Forest, MLP, Kitsune, and measure a two-phase cascade (Suricata with 52,674 Emerging-Threats rules -> VX-One) o
    
[^7]: GoiBERT：通过两阶段知识蒸馏构建紧凑型越南语语言模型

    GoiBERT: Compact Vietnamese Language Models via Two-Stage Knowledge Distillation

    [https://arxiv.org/abs/2610.30067](https://arxiv.org/abs/2610.30067)

    GoiBERT通过两阶段知识蒸馏（先从同分词器越南语编码器学习通用表示，再从大型教师模型获取软标签任务指导），构建了仅2600万和4900万参数、可直接处理原始文本的紧凑越南语编码器，在八项基准测试中以不到一半的参数量和两倍吞吐量取得非教师模型中的最高平均分数。

    

    越南语预训练编码器近年来发展迅速，但性能最强的模型要么是采用多语言词表的大型模型，要么依赖外部分词工具，导致部署成本高昂。我们提出了GoiBERT，这是一系列包含两个紧凑型越南语编码器的模型家族，参数量分别为2600万和4900万，可直接处理原始文本。GoiBERT从两个互补的教师模型中学习：首先由一个使用相同分词器的越南语编码器传授通用语言表示；随后由一个大得多的教师模型通过软标签提供任务级指导，且软标签的生成不会标注教师模型自身的训练数据。在涵盖文本分类、方面级情感分析、自然语言推理和命名实体识别的八项基准测试中，GoiBERT-large在非教师模型中取得了最高的平均分数，其参数量不到最强紧凑基线模型的一半，而吞吐量则是其两倍。GoiBERT-base保留了99%的……（原文摘要至此截断）

    arXiv:2610.30067v1 Announce Type: new  Abstract: Vietnamese pre-trained encoders have improved rapidly, but the strongest ones are large, multilingual-vocabulary models or depend on external word segmentation, which makes them costly to deploy. We present GoiBERT, a family of two compact Vietnamese encoders with 26M and 49M parameters that operate directly on raw text. GoiBERT learns from two complementary teachers. A same-tokenizer Vietnamese encoder first teaches general-purpose representations. A much larger teacher then provides task-level guidance through soft labels that are generated without ever labeling its own training data. Across eight benchmarks covering text classification, aspect-based sentiment analysis, natural language inference, and named entity recognition, GoiBERT-large achieves the highest average score among non-teacher models. It does so with fewer than half the parameters of the strongest compact baseline and twice its throughput. GoiBERT-base retains 99% of th
    
[^8]: 循环均匀扩散语言模型

    Looped Uniform Diffusion Language Models

    [https://arxiv.org/abs/2610.30066](https://arxiv.org/abs/2610.30066)

    该论文首次将循环机制（在每步去噪中递归复用早期Transformer层）引入均匀扩散语言模型，在不改变训练目标和采样器的情况下显著提升训练效率，使循环UDLM以更少计算量即可达到非循环模型训练约0.5万亿token后的验证NLL水平。

    

    均匀扩散语言模型（UDLM）通过迭代去噪普通词表token序列来生成文本，其中任意token都可能被破坏并随后被修订。在训练过程中，UDLM必须在可能不可靠的上下文上进行去噪，而掩码扩散语言模型（MDLM）始终以干净的上下文为条件。这种具有挑战性的训练目标使得UDLM的训练比MDLM更耗费FLOP，但能带来更强的下游推理性能。最近，选择性循环——即在每一步去噪中递归应用若干早期Transformer层——已被证明可以在等FLOP预算下提升MDLM的训练效率。我们在保持训练目标和采样器不变的情况下，研究了UDLM中的循环机制。我们发现循环能显著提升训练效率：循环UDLM达到了在约0.5万亿token上训练的非循环UDLM的验证NLL水平……

    arXiv:2610.30066v1 Announce Type: new  Abstract: Uniform diffusion language models (UDLMs) generate text by iteratively denoising sequences of ordinary vocabulary tokens, any of which may be corrupted and subsequently revised. During training, UDLMs must denoise from a context that may itself be unreliable, whereas masked diffusion language models (MDLMs) always condition on a clean context. This challenging training objective makes UDLMs more FLOP-intensive to train than MDLMs, but it yields stronger downstream reasoning performance. Recently, selective looping, which recursively applies a few early Transformer layers within each denoising step, has been shown to improve training efficiency in MDLMs under an iso-FLOP budget. We investigate looping in UDLMs while keeping the training objective and sampler unchanged. We find that looping significantly improves training efficiency: looped UDLMs reach the validation NLL of a non-looped UDLM trained on about 0.5 trillion tokens with up to 
    
[^9]: 作者所写与智能体所读：企业文档摄取中的静默线索丢失

    What the Author Wrote, What the Agent Reads: Silent Cue Loss in Enterprise Document Ingestion

    [https://arxiv.org/abs/2610.30026](https://arxiv.org/abs/2610.30026)

    该研究揭示了企业文档摄取环节中所有主流转换器都会丢失作者通过文本格式编码的含义，导致来自两家供应商的六个大模型智能体在恢复作者重点标注要求的任务上性能从F1 0.690骤降至0.075-0.150，且这种失败表现为“自信地给出错误答案”而非“明显的不完整”。

    

    企业智能体基于文档采取行动。在智能体能够遵循某项政策、回答合规问题或提交工单之前，某个摄取层必须先将Word文件或PDF转换为令牌（token）——而正是这一层面导致部署在真实环境中的智能体悄然丢失信息。我们证明，每一款主流转换器都会丢失一个层面的信息——即作者通过文本外观所编码的含义——并且这种丢失使智能体“自信地犯错”而非“明显地不完整”。我们基于76份真实政策文档构建了一项任务：恢复作者特别标出的要求，并采用一个无需LLM评判的确定性匹配器、依据文件自身的OOXML属性进行评分。对格式“视而不见”的摄取方式使微观F1从0.690骤降至0.150（docling）和0.075（纯文本），这一差距在来自两家供应商的六个回答模型中均得到复现——涵盖Opus、Sonnet和Haiku三个层级的五个Claude模型，以及OpenAI的GPT-5.6。无法看到线索的智能体只能靠猜测作答，而文档……

    arXiv:2610.30026v1 Announce Type: new  Abstract: Enterprise agents act on documents. Before an agent can follow a policy, answer a compliance question or file a ticket, some ingestion layer must turn a Word file or PDF into tokens - and that layer is where an agent deployed in the wild silently loses information. We show that one layer is lost by every mainstream converter - the meaning an author encodes in how text looks - and that the loss makes agents confidently wrong rather than visibly incomplete. We build a task from 76 real policy documents: recover the requirements the author singled out, scored against the file's own OOXML properties by a deterministic matcher with no LLM judge. Formatting-blind ingestion collapses from micro F1 0.690 to 0.150 (docling) and 0.075 (plain text), a gap that replicates across six answering models from two vendors - five Claude models spanning the Opus, Sonnet and Haiku tiers, and OpenAI's GPT-5.6. An agent that cannot see the cue guesses, and doc
    
[^10]: 智能体行为地图：无需运行智能体，通过“它们会做什么”来定位智能体

    Agent Behavior Maps: Locating Agents by What They Would Do, Without Running Them

    [https://arxiv.org/abs/2610.30014](https://arxiv.org/abs/2610.30014)

    提出“智能体行为地图”方法，利用共享参考轨迹上的中心化动作对数似然为LLM智能体定位，无需实际运行目标智能体即可刻画其行为特征、追踪轨迹KL散度并支持性能预测与行为相似性分析。

    

    具有相似成功率的LLM智能体可能遵循不同的动作序列。智能体行为地图利用共享参考轨迹上的中心化动作对数似然来表示LLM-框架组合，而无需对目标智能体进行新的环境回放。我们在明确假设下推导出将坐标距离平方与轨迹KL散度联系起来的误差界，并且在八个交叉评分的模型中，这些距离在排序上与直接测量的轨迹KL散度保持一致。在100个LLM和四个环境的实验中，基于ALFWorld的坐标支持性能预测并识别行为相似性。在ALFWorld中，即使控制了成功率和模型规模，这些相似性依然存在，包括跨模型家族的相似性。作为扩展，框架组件的位移近似满足可组合性，其精度在不同组件和环境之间有所差异。

    arXiv:2610.30014v1 Announce Type: cross  Abstract: LLM agents with similar success rates can follow different action sequences. An Agent Behavior Map represents LLM-harness combinations using centered action log-likelihoods on shared reference trajectories, without requiring new environment rollouts from the target agent. We derive error bounds relating squared coordinate distances to trajectory KL under explicit assumptions, and across eight cross-scored models the distances track the directly measured trajectory KL in rank order. Across 100 LLMs and four environments, ALFWorld-based coordinates support performance prediction and identify behavioral similarities. In ALFWorld, these similarities persist after controlling for success rates and model size, including across model families. As an extension, harness-component displacements approximately compose, with accuracy varying across components and environments.
    
[^11]: 基于稀疏自编码器构建可审计的AI文本检测器

    Building Auditable AI Text Detectors with Sparse Autoencoders

    [https://arxiv.org/abs/2610.30006](https://arxiv.org/abs/2610.30006)

    该论文提出一种流水线，通过在神经检测器的残差流上训练稀疏自编码器、自动生成特征解释标签并按全局保真度阈值筛选特征，从而将不透明的AI文本检测器转化为基于人类可读概念运行的可审计分类器。

    

    AI生成的文本检测器正越来越多地被部署在具有重大影响的场景中——如同行评审、教育、招生、新闻和内容审核——在这些场景中，检测器的输出会直接影响个人。然而，性能最强的检测器是经过微调的神经网络分类器，其决策过程是不透明的，而透明的替代方案（如文体计量和语言特征分类器）的性能则远远落后。我们提出了一种流水线，能够将微调的神经检测器转换为基于人类可读概念运行的分类器。我们的实现方式是：直接在检测器的残差流上训练稀疏自编码器（SAEs），为每个SAE特征分配自动解释标签，并仅保留那些生成的描述能通过全局保真度阈值的特征。随后，我们仅在这些可解释的特征上训练一个轻量级分类器。我们发现，我们的SAE能够恢复出与高度具体且具有预测性的（摘要在此处截断）潜在概念相对应的特征。

    arXiv:2610.30006v1 Announce Type: new  Abstract: AI-generated text detectors are increasingly deployed in consequential settings---peer review, education, admissions, journalism, and content moderation---where their outputs directly affect individuals. Yet the strongest detectors are fine-tuned neural classifiers whose decisions are opaque, while transparent alternatives (stylometric and linguistic-feature classifiers) underperform them by wide margins. We present a pipeline that converts a fine-tuned neural detector into a classifier that operates on human-readable concepts. We accomplish this by training sparse autoencoders (SAEs) directly on a detector's residual stream, assigning auto-interpretation labels to each SAE feature, and retaining only features whose generated descriptions pass a global fidelity threshold. We then train a lightweight classifier on just the interpretable features. We find that our SAEs recover latent concepts that correspond to highly specific and predicti
    
[^12]: HiddenLens：一个用于检测和解决软件需求中隐藏假设的多层大语言模型框架

    HiddenLens: A Multi-Layer LLM Framework for Detecting and Resolving Hidden Assumptions in Software Requirements

    [https://arxiv.org/abs/2610.29996](https://arxiv.org/abs/2610.29996)

    HiddenLens是首个专为检测、分类、质疑并解决软件需求中隐藏假设而设计的六层BERT-LLM框架，能够超越单纯的歧义标记，在早期发现并化解需求中的隐含假设，从而显著降低后期修复成本。

    

    自然语言形式的软件需求通常省略了关于性能、运行条件、用户行为和领域约束的假设，这些遗漏会影响架构设计、测试和部署。如果这些假设直到实现阶段才被发现，解决它们的代价将远高于在早期阶段处理。尽管歧义性和模糊性已得到广泛研究，但人们对嵌入在看似清晰合理的需求中的假设关注较少。我们提出了HiddenLens，一个新颖的六层BERT-LLM流水线，据我们所知，这是首个专为检测、分类、质疑和解决隐藏假设而设计的框架，而不仅仅是标记歧义。第一层使用BERT注意力熵来定位未解决的文本片段；第二层使用思维链提示的大语言模型推理来阐明每个假设及其失效风险；第三层使用微调后的BERT将假设分类为六类非功能性需求：性能、安全性、可扩展性等。

    arXiv:2610.29996v1 Announce Type: cross  Abstract: Natural-language software requirements often omit assumptions about performance, operating conditions, user behaviour, and domain constraints, affecting architecture, testing, and deployment. If undiscovered until implementation, resolving them is far more costly than addressing them early. Although ambiguity and vagueness are widely studied, less attention has been given to assumptions embedded in requirements that appear clear and reasonable. We propose HiddenLens, a novel six-layer BERT-LLM pipeline and, to our knowledge, the first framework engineered to detect, classify, question, and resolve hidden assumptions rather than merely flag ambiguity. Layer 1 uses BERT attention entropy to localise unresolved spans; Layer 2 uses CoT-prompted LLM reasoning to articulate each assumption and its failure risk; Layer 3 uses fine-tuned BERT to classify assumptions into six non-functional requirement categories: performance, security, scalabil
    
[^13]: 通过变化进行记忆：面向多模态智能体的溯源保持超图记忆

    Remembering Through Change: Provenance-Preserving Hypergraph Memory for Multimodal Agents

    [https://arxiv.org/abs/2610.29995](https://arxiv.org/abs/2610.29995)

    提出了 MHG-Mem，一种多模态超图记忆，通过超边组织实体身份、状态演化与视觉证据溯源，使多模态智能体在跨会话追踪实体变化的同时保留细粒度视觉证据。

    

    多模态智能体的长期记忆仍然具有挑战性，因为它既需要保留细粒度的视觉证据，又需要在多个会话之间追踪实体的演化。现有的基于抽象的记忆将图像压缩为文本，往往会丢弃具有视觉辨别力的细节；而保留像素的记忆虽然能够保留原始观测，却难以可靠地将多样化的观测与持久的实体身份相关联，并追踪其不断变化的状态。为了解决这些局限性，我们提出了 MHG-Mem，一种多模态超图记忆，它联合组织实体身份、状态演化以及对视觉证据的溯源。具体而言，MHG-Mem 将不同的实体提及及其观测到的事实表示为单独的节点，并通过出现超边将身份一致的提及和事实进行分组。随后，它从这些分组的观测中综合出针对特定方面的演化状态，并将其链接到支持的实体、事实和源（此处摘要被截断）……

    arXiv:2610.29995v1 Announce Type: new  Abstract: Long-term memory for multimodal agents remains challenging, as it must preserve fine-grained visual evidence while tracking entity evolution across sessions. Existing abstraction-based memories compress images into text, often discarding visually discriminative details, whereas pixel-preserving memories retain raw observations but struggle to reliably associate diverse observations with persistent entity identities and track their evolving states. To address these limitations, we propose MHG-Mem, a multimodal hypergraph memory that jointly organizes entity identity, state evolution, and provenance to visual evidence. Specifically, MHG-Mem represents distinct entity mentions and their observed facts as individual nodes, grouping identity-consistent mentions and facts through occurrence hyperedges. It then synthesizes aspect-specific evolution states from these grouped observations and links them to supporting entities, facts, and source e
    
[^14]: CulturalRiddles：一个多文化谜语基准测试

    CulturalRiddles: A Multicultural Benchmark of Riddles

    [https://arxiv.org/abs/2610.29968](https://arxiv.org/abs/2610.29968)

    CulturalRiddles是一个涵盖62种文化、完全由人类创作的谜语基准测试，用于评估大语言模型的跨文化理解与推理能力，结果表明即使是前沿模型在某些社区中的准确率也最高仅达22%。

    

    随着大语言模型不断扩大其语言覆盖范围，评估它们对多元文化的理解和表达能力变得越来越重要。然而，许多基准测试仍然以英语为中心，文化覆盖有限，并且更侧重于事实性知识而非其在现实文化语境中的应用。我们提出了CulturalRiddles，这是一个完全由人类通过社区参与式努力创作的原创、植根于文化的谜语基准，涵盖62种文化（45个国家、51种语言）。这些谜语深深植根于各自的文化，需要溯因推理和组合推理能力，以及详细的事实和社会知识、语言技能以及对文化特定关联的理解。我们使用自动评估和大规模人工评估对28个LLM配置进行了基准测试。即使是前沿模型，在某些社区中的准确率也最高仅达到22%。

    arXiv:2610.29968v1 Announce Type: new  Abstract: As LLMs expand their language coverage, evaluating how well they understand and represent diverse cultures becomes increasingly important. However, many benchmarks remain English-centric, with limited cultural coverage and a focus on factual knowledge over its application in real-world cultural contexts. We propose CulturalRiddles, a benchmark of original, culturally grounded riddles created entirely by humans through a participatory community effort, spanning 62 cultures (45 countries, 51 languages). These riddles are deeply rooted in their respective cultures and require abductive and compositional reasoning, alongside detailed factual and social knowledge, language skills, and an understanding of culturally specific associations. We benchmark 28 LLM configurations using both automatic and large-scale human evaluation. Even frontier models achieve at most 22% accuracy in some communities. We publicly release all artifacts alongside an 
    
[^15]: 抉择还是弃权？系统一决策模型、大语言模型与缺失的法律依据

    Commit or Abstain? System One Decision Models, LLMs, and Missing Legal Authority

    [https://arxiv.org/abs/2610.29961](https://arxiv.org/abs/2610.29961)

    系统一决策模型在正确的法律依据缺失时能够可靠地弃权（错误作答率仅30%，远低于生成式大语言模型的38%-86%），并以仅为生成式大语言模型1/55的成本达到同等准确率，为可验证的法律AI提供了高性价比且诚实的解决方案。

    

    鉴于需要核查的法律主张数量庞大，可验证的法律人工智能需要可扩展、高性价比且能满足法律实践需求的模型。系统一决策模型构建于大语言模型之上，但不生成文本，而是为每个选项返回一个经过校准的概率。这非常适合法律依据的核查场景，因为该问题是封闭式的：要么是某个给定的裁判要旨，要么都不是。我们测试了当正确的法律依据缺失时，此类模型是否会选择弃权。我们对CaseHOLD数据集进行了扩展，加入了一个弃权条件，即在5,314条数据中的1,670条里将正确的裁判要旨替换掉，并比较了四个决策模型与六个生成式大语言模型的表现。当没有任何选项正确时，生成式大语言模型仍在38%-86%的条目中强行选择一个选项；而最谨慎的决策模型仅在30%的条目中这样做。在标准CaseHOLD数据集上，一个决策模型以仅为生成式大语言模型1/55的成本达到了与其最佳水平相当的表现（88.1% vs. 88.2%）。只有当决策模型……（原文在此处截断）

    arXiv:2610.29961v1 Announce Type: cross  Abstract: Given the enormous number of claims to be checked, verifiable legal AI requires scalable, cost-effective models that answer the practical needs of law. System One decision models are built on large language models but, instead of generating text, return a calibrated probability for each option. This suits the verification of legal authority, where the question is closed: a given holding, or none of them. We test whether such models abstain when the correct authority is missing. We extend CaseHOLD with an abstention condition in which the correct holding is replaced in 1,670 of 5,314 items, and compare four decision models with six generative LLMs. When no option is correct, generative LLMs still pick one in 38-86% of items; the most cautious decision model does so in 30%. On standard CaseHOLD, a decision model matches the best generative LLM (88.1% vs. 88.2%) at 1/55 of the cost. Accepting that LLM's answer only when a decision model a
    
[^16]: 迈向英语-米佐语法律机器翻译：语料库构建、分析与基准测试

    Towards English - Mizo Legal Machine Translation: Corpus Creation, Analysis, and Benchmarks

    [https://arxiv.org/abs/2610.29943](https://arxiv.org/abs/2610.29943)

    本文构建了首个英语-米佐语印度法院判决平行语料库（13,168个句子对），并通过对NLLB-200 1.3B模型微调实现法律领域机器翻译，在法律测试集上BLEU分数较基础模型提升7.16。

    

    印度法院的诉讼程序以英语进行，因此翻译技术有限的那些语言的使用者难以获取这些程序。我们提出了首个印度法院判决书的英语-米佐语平行数据集，其中包含来自高哈蒂高等法院和印度最高法院的13,168个句子对。由专业的米佐语母语译者翻译完整文档，句子对经过人工对齐，并由两名独立审核员检查结果。我们的研究发现，英语词汇的借用方式非常具有选择性：官方和法律术语几乎总是保留英语原文，但表示日常事物（如行为、情境和结果）的词汇通常用米佐语本族语表达。此外，米佐语的变音符号在整个语料库中的使用也不一致。当我们在训练数据上微调NLLB-200 1.3B模型时，它在包含1,000个句子对的平衡法律测试集上获得27.47 BLEU分数，比基础模型提高了7.16 BLEU。

    arXiv:2610.29943v1 Announce Type: new  Abstract: Indian court proceedings are held in English, so speakers of languages with limited translation technology have limited access to them. We present the first English Mizo parallel dataset of Indian court judgments. It contains 13,168 sentence pairs from the Gauhati High Court and the Supreme Court of India. Professional native Mizo translators translated complete documents; sentence pairs were manually aligned, and two independent reviewers checked the results. Our study finds that English words are borrowed in a very selective way: official and legal terms are almost always kept in English, but words for everyday things like actions, situations, and results are usually expressed in native Mizo. Mizo diacritics are also used inconsistently across the corpus. When we fine tune the NLLB 200 1.3B model on our training data, it scores 27.47 BLEU on a balanced legal test set of 1,000 sentence pairs, an improvement of 7.16 BLEU over the base mo
    
[^17]: ToM2Vec：用于多智能体协作的连续心智理论词元

    ToM2Vec: Continuous Theory-of-Mind Tokens for Multi-Agent Coordination

    [https://arxiv.org/abs/2610.29934](https://arxiv.org/abs/2610.29934)

    提出ToM2Vec方法，利用固定数量的连续推理词元在丰富的潜在空间中学习抽象的心智理论表示，使LLM智能体能够在不确定性下捕捉伙伴的多种可能状态，从而显著提升多智能体协作能力。

    

    形成准确的心智理论对于成功的多智能体协作至关重要，这需要在不确定性下进行推理，以捕捉合作伙伴的多种可能状态。这在大型语言模型（LLM）智能体原生的文本表示空间中很难实现，因为该空间更容易描述单一的伙伴状态。为了解决这一挑战，我们提出了ToM2Vec，它通过固定数量的连续推理词元，学会在丰富的潜在空间中形成抽象的心智理论。在训练过程中，这些词元的监督信号来自伙伴的真实想法，ToM词元被训练去抽象并匹配这些想法。在测试时，每个智能体都会生成自己的ToM词元，并将其作为该智能体思考和行动的条件。在涵盖合作、竞争和混合动机场景的多个环境（Overcooked-AI、多方谈判、Decrypto、Hanabi以及Collab Escape/Capture）中，我们对ToM2Vec进行了评估。

    arXiv:2610.29934v1 Announce Type: cross  Abstract: Forming an accurate theory-of-mind (ToM) is key to successful multi-agent collaboration and requires reasoning under uncertainty to capture multiple possible partner states. This is hard to do in the text representation space native to large language model (LLM) agents, which more easily describe a single partner state. To address this challenge, we introduce ToM2Vec, which learns to form abstract theories-of-mind in a rich latent space via a fixed budget of continuous reasoning tokens. During training, supervision for these tokens comes from the partner's actual thoughts, which the ToM tokens are trained to abstract and match. At test-time, ToM tokens are generated by each agent and act as conditioners for that agent's thoughts and actions. Across several environments covering collaborative, competitive, and mixed-motive settings (Overcooked-AI, multiparty negotiation, Decrypto, Hanabi, and Collab Escape/Capture), we evaluate ToM2Vec'
    
[^18]: 图拓扑与能量驱动寻径的相遇：在检索增强生成中保持上下文一致性

    Graph Topology Meets Energy-Driven Pathfinding: Preserving Contextual Consistency in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.29933](https://arxiv.org/abs/2610.29933)

    提出 ThreadRAG 框架，通过语义拓扑映射器（STM）将检索文档组织为上下文完整的语义节点并构建语义拓扑图，结合能量驱动的路径搜索，解决知识图谱 RAG 中语义损失和模态失配两大瓶颈问题。

    

    检索增强生成（RAG）经常集成知识图谱（KG）来完成复杂的多跳问答任务。然而，其性能日益受到知识构建与推理过程的瓶颈制约：（i）严重的语义损失：将连续文本提炼为离散符号会剥离上下文的细微差别与叙事连贯性；（ii）显著的模态失配：碎片化的图结构与大型语言模型（LLM）预训练所使用的连续文本相冲突，而将图结构重写为自然语言则会引入生成噪声并遗漏信息。为解决这些局限，我们提出了 ThreadRAG，一种基于语义拓扑的 RAG 框架，用于实现保持上下文的知识结构与模态原生的证据推理。ThreadRAG 引入了语义拓扑映射器（STM），它将检索到的文档组织为上下文完整的语义节点，并通过叙事、语义……

    arXiv:2610.29933v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) frequently integrates knowledge graphs (KGs) for complex multi-hop question answering. However, their performance is increasingly bottlenecked by the knowledge structuring and reasoning processes. (i) Severe semantic loss: Distilling continuous texts into discrete symbols strips away contextual nuances and narrative coherence. (ii) Pronounced modality mismatch: Fragmented graph structures clash with the continuous texts LLMs are pre-trained on, while rewriting them into natural language introduces generative noise and omits information. To address these limitations, we propose ThreadRAG, a semantic-topology-based RAG framework for context-preserving knowledge structuring and modality-native evidence reasoning. ThreadRAG introduces a Semantic Topology Mapper (STM), which organizes retrieved documents into context-intact Semantic Nodes and constructs a Semantic Topology Graph (STG) through narrative, se
    
[^19]: 你的基准测试是关于难度的经验测度

    Your Benchmark Is an Empirical Measure Over Difficulty

    [https://arxiv.org/abs/2610.29927](https://arxiv.org/abs/2610.29927)

    本文提出将基准测试视为难度上的经验测度这一新视角，识别并形式化了“测度失配”与“覆盖缺口”两种失效模式，提出两个具有严格理论保证的诊断指标，并对7个领域25个流行基准的分析表明许多基准的难度权重隐式失衡，其排行榜结论需谨慎解读。

    

    基准测试不仅仅是一组测试题目的集合；它天然地在难度维度上诱导出一个经验测度。在本工作中，我们发展了这种基于测度的基准评估视角。这一视角使我们能够识别并形式化两种核心失效模式：一是测度失配，即基准的难度分布与预期的难度分布不一致；二是覆盖缺口，即难度轴上未被覆盖的区域导致基准无法检测到有意义的局部训练进展。我们提出了两个能够刻画这两种失效模式的诊断指标，并为其提供了严格的理论保证。在实证方面，我们利用精心估计的题目难度，分析了横跨7个领域的25个流行基准。结果表明，许多基准隐式地对某些难度区域赋予了过高权重，因此其排行榜结论应当谨慎解读；另一些基准则在难度轴上留下了大量未被覆盖的区域。

    arXiv:2610.29927v1 Announce Type: cross  Abstract: A benchmark is not merely a collection of test questions; it naturally induces an empirical measure over difficulty. In this work, we develop this measure-based view of benchmark evaluation. This perspective allows us to identify and formalize two central failure modes: measure mismatch, where the benchmark difficulty distribution differs from the intended difficulty distribution, and coverage gaps, where uncovered regions along the difficulty axis prevent benchmarks from detecting meaningful localized training progress. We propose two diagnostic metrics that capture these two failure modes and provide rigorous theoretical guarantees. Empirically, we analyze 25 popular benchmarks spanning 7 domains using carefully estimated item difficulties. Our results show that many benchmarks implicitly overweight certain difficulty regions, so their leaderboard conclusions should be interpreted with caution; others leave substantial portions of th
    
[^20]: WMT 2026 MIST中面向开放式问答的任务保持型多语言指令微调

    Task-Preserving Multilingual Instruction Tuning for Open-Ended QA in WMT 2026 MIST

    [https://arxiv.org/abs/2610.29926](https://arxiv.org/abs/2610.29926)

    该研究通过LoRA结合任务保持型本地化构建的多语言指令数据集（EuroAlpaca）及其跨语言增强版本（CrossEuroAlpaca）对Qwen3-8B进行微调，在WMT 2026 MIST共享任务的十种语言开放式问答任务上显著提升了性能，但这些提升具有任务和语言特异性。

    

    本文介绍了我们提交给WMT 2026多语言指令共享任务（Multilingual Instruction Shared Task）的方案。我们使用LoRA对Qwen3-8B进行适配，所使用的数据包括：主办方提供的数据、EuroAlpaca（一个通过任务保持型本地化方式构建的多语言指令数据集），以及CrossEuroAlpaca（一种跨语言增强数据集，它为指令、上下文和回答分配不同的源语言与目标语言）。我们的系统针对十个共享任务语言的开放式问答生成任务（QA-OEG），这些语言均在辅助数据中有所体现。在匹配解码条件下，与基础模型相比，适配使目标语言QA-OEG的平均Gemma-4评审分数相对提升了7.8%。在基于Aya评估套件对所有共享任务语言进行的额外评估中，该方案在宏观平均ROUGE-L上取得了41.5%的相对提升，在BERTScore-F1上取得了2.9%的相对提升，且目标语言上的改进幅度更大。然而，这些提升是特定于任务和语言的。Gemma-4分数在QA-O（原文此处截断）

    arXiv:2610.29926v1 Announce Type: new  Abstract: This paper describes our submission to the WMT 2026 Multilingual Instruction Shared Task. We adapt Qwen3-8B with LoRA using organiser-provided data, EuroAlpaca, a multilingual instruction dataset created through task-preserving localisation, and CrossEuroAlpaca, a cross-lingual augmentation that assigns different source and target languages to instructions, contexts and responses. Our system targets open-ended question answering (QA-OEG) in ten shared-task languages represented in the auxiliary data. Under matched decoding, adaptation improves the mean Gemma-4 judge score on target-language QA-OEG by 7.8% relative to the base model. On an additional evaluation using the Aya Evaluation Suite across all shared-task languages, it yields relative gains of 41.5% in macro-averaged ROUGE-L and 2.9% in BERTScore-F1, with larger improvements in target languages. However, these gains are task- and language-specific. Gemma-4 scores improve for QA-O
    
[^21]: 大语言模型价值训练的意外副作用

    Unintended Side Effects of Value Training in LLMs

    [https://arxiv.org/abs/2610.29922](https://arxiv.org/abs/2610.29922)

    本文通过在霍夫斯泰德各价值维度上微调四个大语言模型，发现针对特定价值的训练会对未训练的价值观、风险偏好、社会认知和安全特性产生意外副作用，因此建议价值对齐评估应纳入更广泛的行为测试。

    

    随着大语言模型越来越多地被部署在价值敏感的场景中，使其与人类价值观对齐已成为自然语言处理领域后训练阶段的一个目标，旨在向模型灌输额外的价值观并引导模型行为。然而，目前的评估仍然较为狭窄。由于各种价值观之间相互关联，针对特定价值观进行微调会如何改变模型无关的能力和行为尚不清楚。在这项工作中，作者分别在霍夫斯泰德价值理论的各个价值维度上对四个指令微调的大语言模型进行调整，并在跨价值迁移和更广泛的行为层面对其进行评估。研究发现，针对特定价值的训练会对未经训练的价值观、风险偏好、社会认知和安全特性产生意想不到的副作用。因此，作者建议在后训练之后的价值对齐过程中应纳入更广泛的行为评估，并强调需要进一步研究额外的价值训练最适合嵌入到训练流程的哪个环节。

    arXiv:2610.29922v1 Announce Type: new  Abstract: As LLMs are increasingly deployed in value-sensitive contexts, aligning them with human values has emerged as a post-training objective in NLP, aimed at instilling additional values and guiding model behaviour. However, current evaluations remain narrow. Because values are interconnected, it is unclear how fine-tuning for specific values alters unrelated capabilities and behaviours. In this work, we adapt four instruction-tuned LLMs across individual Hofstede's value dimensions and evaluate them on both cross-value transfer and broader behaviours. We find that value-specific training produces unintended side effects on untrained values, risk preferences, social cognition, and safety traits. Consequently, we suggest that value alignment after post-training should incorporate broader behavioral evaluations, and we highlight the need to investigate where additional value training might best fit into the pipeline.
    
[^22]: 面向讽刺性语码混合句子检测的大型语言模型基准测试

    Benchmarking Large Language Models for Detecting Satirical Code-Mixed Sentences

    [https://arxiv.org/abs/2610.29907](https://arxiv.org/abs/2610.29907)

    该论文构建了包含60余万样本的罗马尼亚语语码混合讽刺检测数据集CS-SeLeRoSa，并系统基准测试了多语言大模型在零样本、思维链和微调三种模式下检测讽刺性语码混合句子的能力。

    

    语码转换是一种自然的语言现象，然而其对高级推理任务的影响在罗马尼亚语等资源匮乏的语言中仍未得到充分探索。讽刺检测等自然语言处理任务高度依赖于细微的文化和句法线索，而大型语言模型难以捕捉这些线索。在本文中，我们研究了四种生成罗马尼亚语与英语、德语、法语和意大利语混合的合成语码混合句子的方法（即随机替换、短语翻译、依存句法分析和大型语言模型）。我们从一个现有的讽刺检测数据集出发——该数据集由罗马尼亚语新闻文章汇编而成，并由人工在句子级别进行标注——并用超过60万个样本对其进行扩充，构建了CS-SeLeRoSa数据集。我们对数据集的生成过程进行了深入分析。随后，我们在零样本、思维链和微调等多种设置下对多个多语言大型语言模型进行了评估。

    arXiv:2610.29907v1 Announce Type: new  Abstract: Code-switching is a natural linguistic phenomenon, yet its impact on high-level reasoning tasks remains largely unexplored in under-resourced languages like Romanian. Natural language processing tasks, such as satire detection, rely heavily on nuanced cultural and syntactic cues that large language models struggle to capture. In this paper, we investigate four methods (i.e., random replacement, phrase translation, dependency parsing, and large language models) for generating synthetic code-mixed Romanian sentences with English, German, French, and Italian. We start with an existing dataset for satire detection, compiled from Romanian news articles and annotated at the sentence level by humans, and augment it with over 600k samples, namely CS-SeLeRoSa. We provide an in-depth analysis of the dataset generation process. Then, we evaluate several multilingual large language models across zero-shot, chain-of-thought, and fine-tuning regimes o
    
[^23]: 探究学习者错误对大型语言模型中社会偏见的影响

    Investigating the Impact of Learner Errors on Social Bias in Large Language Models

    [https://arxiv.org/abs/2610.29896](https://arxiv.org/abs/2610.29896)

    该研究首次系统考察了非母语写作者的语法错误对大型语言模型社会偏见的影响，发现错误率越高偏见越强（其中介词错误影响最大），并通过在错误文本上微调有效缓解了这一问题。

    

    现有的社会偏见基准测试大多在流畅、规范的英语上评估大型语言模型，反映的是母语使用者的用法，而未将非母语写作者纳入考量。为填补这一空白，我们研究了非母语写作者的语法错误如何影响社会偏见。我们开发了一个受控框架，将常见的学习者错误注入提示中，并将其应用于三个偏见基准测试，涵盖性别、种族和宗教方面的刻板印象一致性以及人口群体偏好。我们发现，语法错误会加剧偏见，且这种影响随错误率的增加而增强。在我们考察的四种常见错误类型中，介词错误导致刻板印象一致性出现最大且最一致的增幅。我们的分析表明，当错误更可能干扰理解时，其加剧偏见的作用更强。最后，在含错误文本上进行微调可以减少语法错误造成的偏见增加。我们的发现表明，偏……

    arXiv:2610.29896v1 Announce Type: new  Abstract: Existing social bias benchmarks evaluate large language models mostly on fluent, well-formed English, reflecting native speaker usage and leaving non-native writers unrepresented. To address this gap, we investigate how grammatical errors by non-native writers affect social bias. We develop a controlled framework that injects common learner errors into prompts and apply it to three bias benchmarks covering stereotype alignment and demographic preference across gender, race, and religion. We find that grammatical errors increase bias, and the effect grows with the error rate. Among the four common error types we consider, preposition errors produce the largest and most consistent increase in stereotype alignment. Our analysis suggests that errors increase bias when they are more likely to interfere with comprehension. Finally, fine-tuning on erroneous texts reduces the bias increase caused by grammatical errors. Our findings show that bia
    
[^24]: 每个部分都有其作用：词性对大语言模型后训练的影响

    Every Part Has Its Role: Part-of-Speech Effects in LLM Post-Training

    [https://arxiv.org/abs/2610.29843](https://arxiv.org/abs/2610.29843)

    该研究将概念增广方法从名词拓展到动词和形容词，发现承载概念的词性会影响大语言模型后训练中的困惑度和下游性能，表明不同词性在训练中扮演着不同角色。

    

    大语言模型通常采用下一词元预测（NTP）进行训练，这种方式奖励对特定词元的预测，而非显式地考虑不同词语或表达之间的语义相似性。先前的工作通过在训练数据中增广“最小变化句子”来解决这一局限，即将目标词替换为概念上相关的替代词，例如 "The tired coach ran home" → {"The tired trainer ran home"、"The tired instructor ran home"……}。这种概念增广方法能够提升性能，但其有效性是否取决于概念在句子中的表达位置，目前仍不清楚。我们复现并扩展了先前的工作，将其从名词拓展到动词和形容词，探究承载概念信号的词性（POS）是否会影响困惑度和下游性能。我们发现不同的评估设置呈现出不同的模式：概念增广提升了困惑度……（原文摘要到此截断）

    arXiv:2610.29843v1 Announce Type: new  Abstract: Large language models are typically trained with next-token prediction (NTP), which rewards predicting specific tokens rather than explicitly accounting for the semantic similarities between different words or expressions. Prior work has addressed this limitation by augmenting training data with minimally varying sentences that replace a target word with conceptually related alternatives, e.g., "The tired coach ran home" $\rightarrow$ {"The tired trainer ran home", "The tired instructor ran home", ...}. Such concept augmentation improves performance, but it remains unclear whether its effectiveness depends on where in the sentence the concept is expressed. We reproduce and extend prior work $\textit{beyond nouns}$ to verbs and adjectives, asking whether the part of speech (POS) carrying the concept signal affects perplexity and downstream performance. We find different patterns across evaluation settings: concept augmentation improves pe
    
[^25]: S-EDL：从序列似然中引出自证据以实现大语言模型的语义校准

    S-EDL: Eliciting Self-Evidence from Sequence Likelihoods for Semantic Calibration of LLMs

    [https://arxiv.org/abs/2610.29832](https://arxiv.org/abs/2610.29832)

    提出S-EDL证据学习方法，通过从模型自身的序列似然中引出证据并构建可微分的Dirichlet先验，实现了LLM在开放式生成中的语义校准，完全无需依赖固定的多项选择选项。

    

    在开放式生成中对大语言模型（LLM）进行校准具有独特的挑战性，因为许多不同的token序列可以表达相同的底层含义。先前的校准技术要么针对固定集合的分类置信度，要么针对单个字符串的二元正确性，这使它们本质上不适合开放式语义结果的动态、多字符串特性。为了弥合这一差距，我们提出了S-EDL，这是一种新颖的证据学习方法，能够原生地针对语义校准来优化LLM。通过从模型自身的序列似然中引出证据，S-EDL在特定提示的语义类上构建了一个可微分的Dirichlet先验。在该先验下优化证据损失，可以在可变语义类上产生感知校准的训练信号，从而完全无需固定的多项选择选项。在四个开放式基准测试和三个模型上的评估表明

    arXiv:2610.29832v1 Announce Type: new  Abstract: Calibrating large language models (LLMs) in open-ended generation is uniquely challenging because many distinct token sequences can express the same underlying meaning. Prior calibration techniques target either fixed-set classification confidence or the binary correctness of individual strings, making them inherently ill-suited for the dynamic, multi-string nature of open-ended semantic outcomes. To bridge this gap, we introduce S-EDL, a novel evidential method that natively optimizes LLMs for semantic calibration. By eliciting evidence from the sequence likelihoods of the model itself, S-EDL constructs a differentiable Dirichlet prior over prompt-specific semantic classes. Optimizing the evidential loss under this prior yields a calibration-aware training signal over variable semantic classes, completely bypassing the need for fixed multiple-choice options. Evaluations across four open-ended benchmarks and three models demonstrate that
    
[^26]: MemoryTax：智能体记忆的苦涩教训

    MemoryTax: The Bitter Lesson of Agent Memory

    [https://arxiv.org/abs/2610.29823](https://arxiv.org/abs/2610.29823)

    这项实证研究揭示了智能体记忆的“苦涩教训”——精心设计的记忆系统只是弱模型的昂贵补丁（仅对 GPT-4o-mini 有用且构建成本高昂），其效果受写入模型能力封顶，而强模型直接搜索原始历史的全上下文方式在成本-精度上更优。

    

    诸如 Mem0 和 Zep 等智能体记忆系统在任何问题到来之前就预先决定保留哪些内容；而编程智能体（coding agents）则保留原始历史记录，让模型用自身工具去搜索它。我们探究当模型足够强大、能够自主检索时，精心设计的记忆机制是否仍有帮助。在一个 LoCoMo 对话（419 轮，152 个问题）上，我们运行了四种记忆方法（全上下文、BM25、Mem0、Zep）和三个编程智能体（Pi、Codex、Claude Code），每种配置分别搭配 GPT-4o-mini、GPT-6 Luna 和 Claude Opus 5.5，并记录每次运行的成本。这是一项简短的实证研究，每个实验单元仅使用一个对话和一次运行。研究表明，智能体记忆对弱模型而言是一种昂贵的补丁：Mem0 仅在使用 GPT-4o-mini 时才胜过全上下文方法，且其构建成本本身就超过了一次完整的全上下文运行。智能体记忆还存在一个由写入模型决定的上限：更强的读取者无法挽回更弱的写入者所遗漏的信息。在使用 Claude Opus 5.5 时，Pi 在成本-精度前沿上表现最佳。

    arXiv:2610.29823v1 Announce Type: new  Abstract: Agent memory systems such as Mem0 and Zep decide what to keep before any question arrives; coding agents keep the raw history and let the model search it with its own tools. We ask whether designed memory still helps once the model is strong enough to search for itself. On one LoCoMo conversation (419 turns, 152 questions), we run four memory methods (full context, BM25, Mem0, Zep) and three coding agents (Pi, Codex, Claude Code), each with GPT-4o-mini, GPT-6 Luna and Claude Opus 5.5, and record what every rollout costs. This is a short empirical study, with one conversation and one run per cell. Agent memory is a costly patch for a weak model: Mem0 beats full context only with GPT-4o-mini, and its build alone costs more than a whole full-context rollout. Agent memory also has a ceiling, set by the model that writes it: a stronger reader does not recover what a weaker writer left out. With Claude Opus 5.5, Pi tops the cost-accuracy front
    
[^27]: 真实的PMID，错误的论文：生物医学论断验证智能体的框架设计

    Real PMIDs, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents

    [https://arxiv.org/abs/2610.29812](https://arxiv.org/abs/2610.29812)

    该论文通过受控消融实验发现，在基于实时PubMed API的生物医学论断验证任务中，让语言智能体自行构造查询的框架设计能将标准论文召回率提升三倍并取得最佳准确率（62%），而简单的单次检索仅能为6%的论断找到标准论文且无法提升准确率。

    

    文献检索语言智能体正在成为生命科学领域科研助手的标准组成部分，然而关于其框架中哪些部分真正重要，目前缺乏受控的实验证据。我们以实时PubMed API作为唯一工具，在生物医学论断验证任务上进行了受控消融实验。在固定骨干模型和输出格式的前提下，我们比较了四种框架：闭卷直接回答、以论断本身作为查询的单次检索、能够自行编写PubMed查询的ReAct式智能体，以及给定标准摘要的oracle（理想上界），每种框架均使用两个高效的前沿骨干模型。天真的单次检索仅能为6%的论断找到标准论文，并且只是将过度自信的错误转化为弃权，而没有提升准确率。让模型自行构造查询可将标准论文的召回率提升三倍，并产生最佳的判定结果（准确率为62%，而其他现实可行的框架为49-59%），尽管这些差异并不显著。

    arXiv:2610.29812v1 Announce Type: cross  Abstract: Literature-searching language agents are becoming a standard component of scientific assistants for the life sciences, yet there is little controlled evidence on which parts of their harness matter. We run a controlled ablation on biomedical claim verification (SciFact) with the live PubMed API as the only tool. Holding the backbone and output format fixed, we compare four harnesses: closed-book answering, single-shot retrieval using the claim as the query, a ReAct-style agent that writes its own PubMed queries, and an oracle given the gold abstracts, each with two efficient frontier backbones. Naive single-shot retrieval finds the gold paper for only 6% of claims and turns overconfident errors into abstentions without improving accuracy. Letting the model formulate its own queries triples gold recall and yields the best verdicts (62% accuracy vs. 49-59% for the other realistic harnesses), although these differences are not significant
    
[^28]: MAPS：面向检索增强生成中搜索动量调控的宏自适应并行推理

    MAPS: Macro-Adaptive Parallel Reasoning for Regulating Search Momentum in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.29792](https://arxiv.org/abs/2610.29792)

    该论文提出“搜索动量”概念来刻画RAG中推理演变对查询意图的影响，发现查询生成易沿已有检索路径持续演化，并据此提出宏自适应并行推理方法MAPS来调控搜索动量，从而减少冗余子查询并提升查询探索的多样性。

    

    检索增强生成（RAG）通过检索外部知识来提升大型语言模型（LLM）的性能。近来，一些RAG方法将推理与检索交错进行，使LLM能够逐步生成子查询并检索外部知识。然而，这些方法可能出现冗余或无效的子查询，以及查询探索多样性不足的问题。此外，由于推理与检索紧密交织，不断演变的推理过程如何影响查询意图仍不清晰。为了刻画推理过程中查询意图的演变，我们引入了“搜索动量”（search momentum）这一概念，它捕捉了沿着推理轨迹、由历史生成的子查询变化所引发的累积检索倾向。通过基于动量的分析，我们发现查询生成过程倾向于沿着已有的检索轨迹持续演化。基于这一观察……

    arXiv:2610.29792v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) improves the performance of Large Language Models (LLMs) by retrieving external knowledge. Recently, some RAG methods interleave reasoning and retrieval, enabling LLMs to progressively generate sub-queries and retrieve external knowledge. However, these methods may suffer from redundant or ineffective sub-queries and insufficiently diverse query exploration. Moreover, due to the tightly interleaved nature of reasoning and retrieval, it remains unclear how the evolving reasoning process influences query intent. To characterize the evolution of query intent during reasoning, we introduce search momentum, which captures the accumulated retrieval tendency induced by historical changes in generated sub-queries along the reasoning trajectory. Through momentum-based analysis, we find that the query generation process tends to continuously evolve along existing retrieval trajectories. Based on this observatio
    
[^29]: 大语言模型从哪里学会推荐AI？

    Where Do LLMs Learn to Recommend AI?

    [https://arxiv.org/abs/2610.29785](https://arxiv.org/abs/2610.29785)

    研究发现LLM过度推荐AI的倾向主要源于指令微调数据中约7%的AI相关样本——剔除这些数据可将推荐率降至预训练模型水平以下，而追求“有帮助”的数据和“最佳”“立刻”式的提问方式则会进一步推高该倾向。

    

    当被问及该学什么专业、从事什么职业或开发什么产品时，大语言模型（LLM）经常会推荐与AI相关的选项。我们通过比较微调前后的模型以及改变提问方式，来探究这种倾向的来源。首先，预训练模型在其27%的回复推荐中已包含明确的AI/ML术语，而其发布的指令微调版本中这一比例达到59%。其次，我们自己进行微调的实验表明，这种上升并非不可避免：一般的指令微调不会改变该比例，但剔除指令数据中涉及AI的7%样本后，该比例会下降14个百分点，甚至低于其起始的预训练模型水平。奖励“有帮助性”的数据会使该比例上升9个百分点，而根据多样化模拟用户偏好排序的数据则会使其下降9个百分点。最后，在模型保持不变的情况下，要求给出“最佳”选项或询问“现在”该做什么，也会提高这一比例。

    arXiv:2610.29785v1 Announce Type: new  Abstract: Asked what to study, which career to pursue, or what product to build, large language models (LLMs) often recommend AI-related options. We ask where this tendency comes from by comparing models before and after fine-tuning and by varying how questions are asked. First, pretrained models already include explicit AI/ML terms in the recommendations of 27% of their responses, and their released instruction-tuned versions do so in 59%. Second, fine-tuning models ourselves shows that this rise is not inevitable: general instruction tuning leaves the rate unchanged, but removing the 7% of instruction examples that concern AI cuts it by 14 percentage points, below the level of the pretrained models it starts from. Data rewarding helpfulness raise it by 9 points, while data ranked by the preferences of diverse simulated users lower it by 9. Finally, with models held fixed, asking for the "best" option or for what to do "right now" raises the rate
    
[^30]: 与开放世界共同演化：迈向未来预测的递归自我改进

    Evolving with Open Worlds: Towards Recursive Self-Improvement for Future Forecasting

    [https://arxiv.org/abs/2610.29784](https://arxiv.org/abs/2610.29784)

    提出DeepEvolve智能体系统，以未来事件预测作为持续变化且可验证的开放环境，通过递归演化引擎将预测背后的推理轨迹转化为能力，从而实现智能体的递归自我改进。

    

    当今大多数人工智能（AI）系统在发布时参数即被冻结，只能通过外部更新来改进；而理想的系统应能从自身经验中持续改进，我们将这种能力称为递归自我改进（RSI）。然而，研究RSI需要一个持续变化、可客观验证且自然扎根的环境，这给智能体演化带来了根本性挑战。在此，我们提出DeepEvolve，一个以未来事件预测为基础的可持续演化智能体系统，其中智能体与动态演化的环境持续交互，并通过内化和外化这些交互中获得的经验来实现改进。一个“活的”环境负责合成可验证的任务，而递归演化引擎则将每次预测背后的推理轨迹重新转化为自身能力：通过将搜索与预测解耦，使参数级演化能够适应……

    arXiv:2610.29784v1 Announce Type: new  Abstract: Most contemporary artificial intelligence (AI) systems are released with frozen parameters and improve through external updates, whereas an ideal system would keep improving from its own experience, a capacity we call recursive self-improvement (RSI). Yet studying RSI requires an environment that is continuously changing, objectively verifiable, and naturally grounded, posing a fundamental challenge for agent evolution. Here we introduce DeepEvolve, a sustainable evolution agent system grounded in future event prediction, in which agents continuously interact with a dynamically evolving environment and improve by internalizing and externalizing the experience gained through these interactions. A living environment synthesizes verifiable tasks, and a recursive-evolution engine turns the reasoning trajectory behind each prediction back into capability: decoupling search from prediction adapts parameter-level evolution to the contamination-
    
[^31]: ALIAS：评估大语言模型编写的TLA+规约的性能与记忆化现象

    ALIAS: Evaluating Performance and Memorization in LLM-Written TLA+ Specifications

    [https://arxiv.org/abs/2610.29767](https://arxiv.org/abs/2610.29767)

    该论文利用TLA+模型检查器作为验证神谕，发现LLM在标识符重命名任务上的正确率下降主要来自要求精确标识符名称的评分方式，若按行为语义评分，记忆化效应实际很小甚至消失。

    

    记忆化（memorization）会夸大大语言模型（LLM）的基准测试分数，因为模型可以复现其在训练中见过的答案。研究人员通过“一致标识符重命名”（CIR）来检验这种现象，即给任务中的每个标识符起新名字，并将重命名任务上较低的得分视为模型记忆的证据。我们在时序动作逻辑语言（TLA+）上测试了这一方法，该语言的模型检查器可以作为一个“神谕”（oracle），确认重命名后的任务是否保留了原始任务的精确行为。在111个TLA+-Bench任务上，重命名使Claude Opus 4.8的正确率从28%降至18%，GPT-5从9%降至7%。这一下降大部分源于测试的评分方式，即要求提供大多数描述中从未给出的精确名称。若按行为进行评分，下降幅度可缩小至Opus 4.8约3个百分点，GPT-5则为0。在分别在互不相交的任务半区上微调的两个模型家族中，重命名消除了大约一半（摘要在此处被截断）

    arXiv:2610.29767v1 Announce Type: new  Abstract: Memorization inflates the benchmark scores of large language models (LLMs), since a model can reproduce answers it saw in training. Researchers test for it with consistent identifier renaming (CIR), which gives every identifier in a task a new name, and treat a lower score on the renamed task as evidence of recall. We test this method on the Temporal Logic of Actions Plus (TLA+) language, whose model checker acts as an oracle by confirming whether a renamed task preserves the exact behaviors of the original. Across 111 TLA+-Bench tasks, renaming lowers the correctness of Claude Opus 4.8 from 28% to 18% and of GPT-5 from 9% to 7%. Most of this decrease comes from how the test grades, which requires exact names that most descriptions never state. Grading by behavior shrinks the decrease to about 3 percentage points for Opus 4.8 and to 0 for GPT-5. In two model families fine-tuned on disjoint halves of the tasks, renaming removes about half
    
[^32]: 桥接大语言模型的基础能力与智能体能力

    Bridging Foundational and Agentic Capabilities of LLMs

    [https://arxiv.org/abs/2610.29751](https://arxiv.org/abs/2610.29751)

    该论文提出一个操作层面的测量框架，通过定义“能力实现差距”来量化大语言模型基础能力向智能体能力的转化程度，发现基础能力基准只能部分预测智能体操作表现，画像相似的模型在实际执行中差异显著。

    

    大语言模型（LLM）越来越多地被部署为智能体，但其基础能力如何转化为智能体能力仍不清楚。端到端的评分无法判断失败究竟是由于基准能力低下还是部署环节薄弱所致，这限制了可操作的诊断。我们通过一个操作层面的测量框架来桥接基础能力与智能体能力。我们使用从能力基准得出的基础画像（foundational profile）和从交互轨迹得出的操作画像（operation profile）来刻画每个模型，并将“能力实现差距”定义为观测到的操作性能与基于基础能力预测值之间的偏差。在 τ²-Bench 上对十五个模型服务的实验表明，基础画像只能部分预测智能体操作表现。画像相近的模型在执行层面可能存在显著差异，且这些差距在不同任务子集之间保持一致。操作……

    arXiv:2610.29751v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as agents, yet it remains unclear how their foundational capabilities translate into agentic capabilities. End-to-end scores leave unresolved whether low benchmark capability or weak deployment drives a failure, limiting actionable diagnosis. We bridge foundational and agentic capabilities through an operation-level measurement framework. We represent each model with a foundational profile derived from capability benchmarks and an operation profile derived from interaction trajectories, and define capability realization gaps as deviations of observed operation performance from predictions based on foundational capabilities. Experiments on fifteen model services in $\tau^2$-Bench show that foundational profiles only partially predict agentic operations. Models with comparable profiles can differ substantially in execution, and these gaps remain consistent across task subsets. Operatio
    
[^33]: 接受或修复：在对比解码中保留段落所支持的答案

    Accept or Repair: Preserving Passage-Supported Answers in Contrastive Decoding

    [https://arxiv.org/abs/2610.29727](https://arxiv.org/abs/2610.29727)

    提出无需训练的“接受或修复”（AoR）方法，通过将答案与所给段落核对来保留段落支持的答案、仅在不支持时用对比解码进行修复，在反事实段落上比采样提升 12.5 分并超越八种基线。

    

    大语言模型常常基于训练期间学到的知识来回答问题，即使所提供的段落与该知识相矛盾。为了解决这个问题，对比解码在每个词元上对比完整提示与修改后提示下的预测，但它也会改变段落所支持的正确答案。我们证明，将答案与段落进行核对可以消除这种权衡。基于这一观察，我们提出了“接受或修复”（Accept-or-Repair, AoR）这一无需训练的方法。如果段落支持采样得到的答案，AoR 就保留该答案；否则，它会使用对比解码尝试最多两个替换答案，一旦某个替换答案通过该检查就立即返回。当段落倾向于另一个答案时，它会拒绝那些重复了在没有段落情况下所生成答案的替换。我们的实验表明，AoR 在反事实段落上比采样方法提高了 12.5 分，在普通段落上提高了 2.3 分，超越了八种对比解码与提示基线方法。

    arXiv:2610.29727v1 Announce Type: new  Abstract: Large language models often answer from knowledge learned during training, even when a provided passage contradicts that knowledge. To address this problem, contrastive decoding contrasts the predictions under the full and a modified prompt at every token, but it also alters correct answers that the passage supports. We show that checking answers against the passage removes this trade-off. Based on this observation, we propose Accept-or-Repair (AoR), a training-free procedure. AoR keeps a sampled answer if the passage supports it. Otherwise, it tries up to two replacements with contrastive decoding, returning one as soon as it passes this check. It rejects replacements that repeat the answer generated without the passage when the passage favors another answer. Our experiments show that AoR raises the score over sampling by 12.5 points on counterfactual passages and 2.3 on ordinary ones, surpassing eight contrastive and prompting baseline
    
[^34]: 检索增强生成的组合性

    Retrieval Augmented Generation Compositionality

    [https://arxiv.org/abs/2610.29722](https://arxiv.org/abs/2610.29722)

    该论文提出开源受控RAG评估框架LASAGNA，通过对450种配置、八个基准的大规模实验发现，RAG流程中各阶段的强技术在组合后增益无法叠加（预测+47%实际仅+17%）。

    

    arXiv:2610.29722v1 公告类型：新 摘要：检索增强生成（RAG）将大语言模型建立在外部知识之上。现代RAG流程结合了嵌入、分块、检索、重排序、查询增强、上下文压缩和智能体验证等技术，但这些技术通常只是被单独评估。目前尚不清楚它们各自带来的增益在组合后是否仍然成立，检索质量和答案忠实度是否偏好相同的设计选择，以及从小型语料库得出的结论在大规模场景下是否依然有效。我们预先声明了三个假设，并引入了LASAGNA——一个用于受控RAG评估的开源框架。我们在八个基准测试上测试了450种配置，语料库规模从5千到20万篇文档不等。研究得出三个发现。首先，来自RAG流程不同阶段的单独表现优异的技术不一定能够良好组合：三种检索技术预测组合后可带来+47%的提升，但实际同时使用时仅带来+17%的改进。其次，c……（原文摘要在此处被截断）

    arXiv:2610.29722v1 Announce Type: new  Abstract: Retrieval Augmented Generation (RAG) grounds large language models in external knowledge. Modern pipelines combine embedding, chunking, retrieval, reranking, query augmentation, context compression, and agentic verification, yet these techniques are typically evaluated one at a time. It remains unclear whether their isolated gains survive composition, whether retrieval quality and answer faithfulness favor the same design choices, and whether conclusions from small corpora hold at large scale. We pre-declare three hypotheses and introduce LASAGNA, an open-source framework for controlled RAG evaluation. We test 450 configurations across eight benchmarks and corpora ranging from 5K to 200K documents. Three findings emerge. First, individually strong techniques from different phases of a RAG pipeline do not necessarily compose: three retrieval techniques predict a +47% combined improvement but deliver only +17% when used together. Second, c
    
[^35]: 扩展像素语言模型的多语言预训练

    Scaling Multilingual Pretraining of Pixel Language Models

    [https://arxiv.org/abs/2610.29681](https://arxiv.org/abs/2610.29681)

    本文从更多语言、更多文字系统和更大模型容量三个维度扩展像素语言模型的多语言预训练，发现扩展的收益主要作用于新增语言，更大的模型容量可缓解多语言诅咒，且模型在低资源语言上优于子词级和字符级基线。

    

    像素语言模型将文本处理为渲染的图像而非有限的词表，这赋予了它们强大的跨语言迁移能力。然而，关于这些能力如何受到更大规模预训练的影响，目前知之甚少。在本文中，我们沿三个变化维度训练了一系列MPIXEL模型：更多语言、更多文字系统，以及在固定单语言计算量下更大的模型容量。在覆盖83种语言的五项任务评估中，我们发现扩展的作用是局部的：每个维度都能改善新增加的语言和文字系统，而增加模型容量可以缓解在更多语言上进行预训练所带来的“多语言诅咒”。向未见过的文字系统的迁移能力随多语言预训练而提升，但不会随扩展而进一步提升。在类型学上差异较大的低资源语言上，我们的模型在多种书写系统中均优于子词级和字符级基线。我们还展示了更大的像素模型会保留更多的层……（原文摘要在此处截断）

    arXiv:2610.29681v1 Announce Type: new  Abstract: Pixel language models process text as rendered images instead of a finite vocabulary, which gives them strong cross-lingual transfer abilities. However, not much is known about how these abilities are influenced by larger-scale pretraining. In this paper, we train a family of MPIXEL models along three axes of variation: more languages, more scripts, and larger model capacity under fixed per-language compute. In evaluations on five tasks covering 83 languages, we find that scaling acts locally: each axis improves the added languages and scripts, while increased capacity mitigates the curse of multilinguality from pretraining on more languages. Transfer to unseen scripts improves with multilingual pretraining, but does not further improve with scaling. On typologically distant low-resource languages, our models outperform subword and character-level baselines across many writing systems. We also show that larger pixel models reserve more l
    
[^36]: Uni-Mem：通过统一在线策略优化学习长时程智能体记忆管理器

    Uni-Mem: Learning a Long-Horizon Agentic Memory Manager with Unified Online Policy Optimization

    [https://arxiv.org/abs/2610.29671](https://arxiv.org/abs/2610.29671)

    提出Uni-Mem统一记忆管理器，通过阶段解耦GRPO与事件级信用分配实现端到端在线强化学习，在共享策略下联合优化长期智能体的记忆构建与检索能力。

    

    长期智能体记忆领域正从预定义的工作流程转向可学习的记忆构建与检索策略。现有的基于学习的方法要么仅优化记忆构建，要么将检索侧的在线强化学习（RL）与离线构建监督相结合，这限制了两个阶段之间的协同适应。在本文中，我们探索了端到端的在线强化学习，用于统一的记忆管理器，使其在共享策略下同时学习记忆构建与检索两种能力。然而，使用现有强化学习算法（如GRPO）进行联合优化时，面临组间比较混淆以及长时记忆管理轨迹中反馈稀疏的问题。为此，我们提出了阶段解耦GRPO（Stage-Decoupled GRPO），它在每次rollout组内交替进行构建与检索优化，同时控制来自另一阶段的变化，从而产生阶段特定的学习信号。我们进一步引入了事件级信用分配（Event-Wise Credit Assignment，ECA），将反馈定位到（摘要在此处截断）

    arXiv:2610.29671v1 Announce Type: new  Abstract: Long-term agent memory is shifting from predefined workflows toward learnable policies for memory construction and retrieval. Existing learning-based methods either optimize construction alone or combine retrieval-side online reinforcement learning (RL) with offline construction supervision, limiting co-adaptation between the two stages. In this paper, we explore end-to-end online RL for a unified memory manager (Uni-Mem) that learns both capabilities under a shared policy. However, joint optimization with existing RL algorithms such as GRPO faces confounded group comparisons and sparse feedback over long memory-management trajectories. To this end, we propose Stage-Decoupled GRPO, which alternates construction and retrieval optimization while controlling variation from the other stage within each rollout group, yielding stage-specific learning signals. We further introduce Event-Wise Credit Assignment (ECA), which localizes feedback to 
    
[^37]: 冲突记录：用于可审计 LLM 输出质量强制的类型化对抗契约

    Conflicted Records: Typed Adversarial Contracts for Auditable LLM Output Quality Enforcement

    [https://arxiv.org/abs/2610.29669](https://arxiv.org/abs/2610.29669)

    提出 BiasGuard-D，通过两个类型化对抗契约角色对结构化智能体间记录执行确定性检查，实现对 LLM 输出质量的可审计强制执行，捕获单一 LLM 评判者与传统数据清洗都无法发现的冲突性缺陷。

    

    LLM 生成的记录日益充斥于生产信息系统中，并在其中被视为数据资产，需要满足与操作数据库记录相同的质量期望。然而，单条这样的记录可能同时满足又违反某条质量规则：一个有据可查的事实旁可能是幻觉，一个有用的回答旁可能是刻板印象。经典的数据清洗方法无法解决这一问题，因为这两种判断在同一文本片段上互不兼容；单一的 LLM 评判者也无能为力，因为它与产生该冲突的训练偏差同源。诸如提示工程、思维链和 RLHF 等现有干预手段虽能降低表面错误率，但既无法提供审计轨迹，也无法区分真正的修复与一刀切的压制。因此问题在于：对结构化的智能体间记录进行确定性检查，能否捕获仅凭参照性检查所遗漏的缺陷。我们提出了 BiasGuard-D：两个对抗性角色……（原文摘要在此处截断）

    arXiv:2610.29669v1 Announce Type: cross  Abstract: LLM-generated records increasingly populate production information systems where they are treated as data assets subject to the same quality expectations as records from operational databases. A single such record can simultaneously satisfy and violate a quality rule: a grounded fact adjacent to a hallucination, a useful answer adjacent to a stereotype. Classical data cleaning cannot resolve this because the two judgments are incompatible over the same span; a single LLM judge cannot either, since it shares the training biases that produced the conflict. Existing interventions such as prompt engineering, chain-of-thought, and RLHF reduce surface error rates but provide no audit trail and cannot distinguish genuine remediation from blanket suppression. The question is whether a deterministic check over structured inter-agent records can catch the defects that referential checks alone miss. We present BiasGuard-D: two adversarial roles d
    
[^38]: SPARE：通过路由熵进行稀疏度分配以实现高效的专家混合模型

    SPARE: Sparsity Allocation via Routing Entropy for Efficient MoEs

    [https://arxiv.org/abs/2610.29662](https://arxiv.org/abs/2610.29662)

    SPARE是一种无需训练的MoE专家剪枝方法，利用路由熵自适应地为各层分配不同的剪枝稀疏度，在相同预算下比均匀剪枝获得更高的模型准确率。

    

    混合专家架构能够高效地扩展模型容量，但其巨大的内存占用成为部署的主要障碍。专家剪枝可以减少内存占用，但大多数方法只关注删除哪些专家，并对每一层进行同等程度的剪枝。我们提出了SPARE（基于路由熵的稀疏度分配），这是一种无需训练、与选择器无关的方法，它利用路由熵——即token在校准数据上归一化Top-K门控权重的熵——来决定从每一层中删除多少个专家。当一个token的门控权重均匀分散时，删除一个活跃专家只会丢弃其中很小一部分信息，因此SPARE会更激进地剪枝那些路由更分散的层。随机剪枝下的扰动界限将这一直觉形式化，且使用固定选择器的受控干预实验表明集中路由与更大的输出变化相关联。在使用相同选择器和相同预算的情况下，SPARE相比均匀分配方式提升了平均准确率。

    arXiv:2610.29662v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) architectures scale model capacity efficiently, yet their substantial memory footprint poses a major barrier to deployment. Expert pruning reduces this footprint, but most methods focus on which experts to remove and prune every layer equally. We propose SPARE (Sparsity Allocation via Routing Entropy), a training-free, selector-agnostic method that decides how many experts to remove from each layer using routing entropy, the entropy of tokens' normalized Top-K gate weights on calibration data. When a token's gate weight is spread evenly, removing one active expert discards little of it, so SPARE prunes layers with more diffuse routing more aggressively. A perturbation bound under random pruning formalizes this intuition, and controlled interventions with a fixed selector link concentrated routing to larger output changes. With the same selector and budget, SPARE improves average accuracy over uniform allocation
    
[^39]: 模型规模与量化对问答任务中社会偏见与弃答行为的影响

    Effects of Model Scale and Quantization on Social Bias and Abstention in Question Answering

    [https://arxiv.org/abs/2610.29657](https://arxiv.org/abs/2610.29657)

    该研究发现压缩模型观察到的偏见减少可能部分源于弃答增多而非真正更公平——在26个偏见回答较少的配置中，有20个同时表现出过度弃答的增加。

    

    在资源受限环境中部署小型化和量化语言模型引发了一个重要的公平性问题：当一个压缩后的模型产生更少的有偏见回答时，这反映的是更公平的行为还是更多的弃答？我们在问答任务中使用BBQ基准，对来自Qwen3、Gemma-3和SmolLM2的多种模型规模和量化设置开展了研究。我们还考察了所观察到的趋势在替代提示表述下，以及在强制选择设置中移除“未知”选项时是否保持稳定。结果表明，更大的模型获得了更高的问答准确率，但偏见并未呈现简单的随规模扩展的趋势，且量化的影响因配置而异。值得注意的是，在26个产生较少偏见回答的配置中，有20个同时表现出更高的过度弃答率，这表明观察到的较低偏见可能部分源于更多的弃答。这些发现表明观察到的较低偏见……

    arXiv:2610.29657v1 Announce Type: new  Abstract: The deployment of small and quantized language models in resource-constrained settings raises an important fairness question: when a compressed model produces fewer biased responses, does this reflect fairer behavior or greater abstention? We study this problem in Question Answering (QA) using the BBQ benchmark across multiple model sizes and quantization settings from Qwen3, Gemma-3, and SmolLM2. We also examine whether the observed trends remain stable across alternative prompt formulations and when the ``Unknown'' option is removed in a forced-choice setting. Our results show that larger models achieved higher QA accuracy, but bias showed no simple scaling trend and quantization effects varied across configurations. Notably, 20 of the 26 configurations with fewer biased responses also show increased over-abstention, indicating that lower observed bias can partly reflect greater abstention. These findings show that lower observed bias 
    
[^40]: 大语言模型作为调查问卷受访者：与简单语义基线的比较

    LLMs as survey respondents: Comparison to simple semantic baselines

    [https://arxiv.org/abs/2610.29645](https://arxiv.org/abs/2610.29645)

    研究发现LLM在根据部分人类回答预测问卷新回答时通常优于基于嵌入相似性的简单语义基线，但其优势随已知回答比例的降低而减弱，表明其可能依赖题目间的表层语义相似性作为启发式方法。

    

    大语言模型（LLM）已被探索作为人类调查问卷受访者的替代品。一个更有前景的应用是数据增强，即LLM根据部分人类数据预测新的回答。然而，问卷题目之间的语义相似性是答案之间相关性的一个已知来源。因此，尚不清楚LLM究竟是真正推断出问题与回答对之间的有意义关系，还是仅仅利用题目间的表层比较作为启发式方法。为了评估这一点，我们测试了三个LLM在两项分别测量对自动驾驶汽车或轻型电动汽车态度的调查中，基于不同比例的已知回答来预测留出题项回答的能力。实验将LLM的表现与两个基线进行了比较，这些基线通过最大化与已有回答的嵌入相似性来生成新回答。结果表明，尽管LLM在大多数情况下对人类回答的预测优于基线，但随着已知回答比例的降低，其性能持续下降。

    arXiv:2610.29645v1 Announce Type: new  Abstract: Large language models (LLMs) have been explored as surrogates for human survey respondents. A more promising application is data augmentation, where LLMs predict new responses from partial human data. However, the semantic similarity between questionnaire items is a known source of correlation between answers. It is thus unclear whether LLMs actually infer meaningful relationships between question-response pairs, or use surface-level comparison between questions as a heuristic. To assess this, we evaluated how three LLMs can predict responses to held-out items in two surveys measuring attitudes toward automated or light-electric vehicles, based on varying proportions of known responses. Performance was compared to two baselines, where new responses were derived by maximizing embedding-similarity to a prior response. While LLMs mostly predicted human responses better than baselines, their performance consistently declined with a lower sha
    
[^41]: CHIP 2026共享任务1概述：全文生物医学文献中的患者特异性表型概念识别

    Overview of CHIP 2026 Shared Task 1: Patient-Specific Phenotype Concept Recognition in Full-Text Biomedical Literature

    [https://arxiv.org/abs/2610.29606](https://arxiv.org/abs/2610.29606)

    本文概述了CHIP 2026共享任务1，该任务基于PatientPheX数据集的200篇标注全文文献，设置全文表型概念识别与患者特异性表型关联两个子任务，以推动从全文生物医学文献中自动识别表型概念及其患者关联方法的发展。

    

    自动表型信息抽取在罕见病诊断、基因型-表型分析和精准医疗中发挥着重要作用。然而，准确识别表型概念并确定其与特定患者的关联仍然具有挑战性，尤其是在涉及多名患者和复杂临床描述的全文生物医学文章中。本文对CHIP 2026共享任务1进行了概述，该任务旨在推动从全文生物医学文献中进行患者特异性表型概念识别方法的发展。该任务包含两个子任务：全文表型概念识别和患者特异性表型关联。在本次评测中，我们使用了来自PatientPheX数据集的200篇经标注的全文生物医学文章，其中包含506名患者、15,193个表型提及和5,656个患者-表型关联。共有455支团队报名参加了此次竞赛。

    arXiv:2610.29606v1 Announce Type: new  Abstract: Automatic phenotype information extraction plays an important role in rare disease diagnosis, genotype-phenotype analysis, and precision medicine. However, accurately identifying phenotype concepts and determining their associations with specific patients remain challenging, particularly in full-text biomedical articles involving multiple patients and complex clinical descriptions. This paper presents an overview of CHIP 2026 Shared Task 1, which aims to promote the development of methods for patient-specific phenotype concept recognition from full-text biomedical literature. The task consists of two subtasks: full-text phenotype concept recognition and patient-specific phenotype association. For this evaluation, we used 200 annotated full-text biomedical articles from the PatientPheX dataset, containing 506 patients, 15,193 phenotype mentions, and 5,656 patient-phenotype associations. A total of 455 teams registered for the competition,
    
[^42]: 扩散语言模型在表格到文本生成中的忠实性诊断研究

    A Diagnostic Study of Faithfulness in Diffusion Language Models for Table-to-Text Generation

    [https://arxiv.org/abs/2610.29602](https://arxiv.org/abs/2610.29602)

    本研究诊断发现，扩散语言模型在表格到文本生成中随示例增加或微调后流畅度和BLEU提升，但对源表格的事实忠实性反而下降，揭示了表面流畅性与事实依据性的分离。

    

    扩散语言模型通过在双向注意力下对被遮蔽的序列进行迭代去噪来生成文本，而不受自回归解码那样严格的自左向右生成约束。这种差异是否会影响模型在结构化输入中保持可靠事实依据的能力，目前基本上尚未被探索。我们在表格到文本生成任务中使用ToTTo数据集研究这一问题，该数据集中高亮的源表格单元格使事实依据性可以直接被检验。在完全相同的基于检索的少样本协议下，我们将掩蔽扩散模型MDLM和LLaDA与GPT-2和GPT-Neo进行比较，同时还评估了经QLoRA适配的LLaDA，并将参考相似度与源表格事实依据性作为两个独立的维度进行评分。结果显示这两个维度出现了分离。在LLaDA内部，增加示例数量使BLEU从11.3提升至22.0，而相对于源表格的NLI蕴含分数却从0.78下降至0.52；QLoRA达到了本研究中最高的BLEU值24.15，但其事实依据性并未因此改善。人工标注揭示了失败模式……（原文摘要在此处截断）

    arXiv:2610.29602v1 Announce Type: new  Abstract: Diffusion language models generate by iteratively denoising a masked sequence under bidirectional attention, without the strict left-to-right commitment of autoregressive decoding. Whether this difference affects how reliably they remain grounded in structured inputs is largely unexplored. We study this question in table-to-text generation using ToTTo, where highlighted source cells make factual grounding directly testable. We compare the masked diffusion models MDLM and LLaDA with GPT-2 and GPT-Neo under an identical retrieval-based few-shot protocol, together with a QLoRA-adapted LLaDA, and we score reference similarity and source grounding as separate axes. The two pull apart. Within LLaDA, additional demonstrations raise BLEU from 11.3 to 22.0 while NLI entailment against the source table falls from 0.78 to 0.52; QLoRA reaches the best BLEU of the study, 24.15, and grounds no better for it. Manual annotation shows the failure modes d
    
[^43]: WRDO：对齐语言模型中的加权拒绝方向优化

    WRDO: Weighted Refusal Direction Optimization in Aligned Language Models

    [https://arxiv.org/abs/2610.29555](https://arxiv.org/abs/2610.29555)

    提出加权拒绝方向优化方法，通过学习聚合权重从针对每个提示的方向库中构建复合拒绝方向，以强化对齐语言模型的拒绝干预效果。

    

    对齐的大型语言模型被训练为遵守良性请求，同时拒绝有害请求。然而，最近的研究表明，这种拒绝行为是脆弱的，通常可以通过简单的激活空间或权重空间干预来破坏。特别是，已有研究表明，拒绝行为主要由残差流中的单一方向介导，该方向通常被计算为有害提示与无害提示之间的均值差向量。然而，这种标准构造方法将异质的有害提示聚合为一个统一的平均值，可能会混淆不同的与拒绝相关的现象。在本工作中，我们研究是否可以通过学习如何组合针对特定提示的拒绝方向来强化拒绝干预。我们提出了加权拒绝方向优化，该方法利用学习到的聚合权重，从一组针对每个提示的方向库中构建复合拒绝方向。这种方法允许不同……

    arXiv:2610.29555v1 Announce Type: cross  Abstract: Aligned large language models are trained to comply with benign requests while refusing harmful ones. However, recent work shows that this refusal behavior is fragile and can often be disrupted by simple activation- or weight-space interventions. In particular, refusal has been shown to be largely mediated by a single residual-stream direction, typically computed as a difference-of-means vector between harmful and harmless prompts. However, this standard construction aggregates heterogeneous harmful prompts into one uniform average, potentially conflating distinct refusal-related phenomena. In this work, we study whether refusal interventions can be strengthened by learning how to combine prompt-specific refusal directions. We propose Weighted Refusal Direction Optimization (WRDO), a procedure that constructs a composite refusal direction from a bank of per-prompt directions using learned aggregation weights. This approach allows diffe
    
[^44]: PACE：四比特循环状态的周期性与通道对齐编码

    PACE: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States

    [https://arxiv.org/abs/2610.29545](https://arxiv.org/abs/2610.29545)

    PACE 提出一种事后的四比特状态量化方法，通过每隔几个 token 才周期性地对循环状态进行一次舍入，并利用大数值条目在通道上聚集对齐的结构特性，减少舍入误差累积，从而在不微调模型的情况下实现高精度的四比特推理状态存储。

    

    线性注意力语言模型将上下文保存在一个固定大小的循环状态中，推理系统会为同时解码的每个序列存储一份该状态的副本。因此，以较低精度存储该状态可以让更多序列共享同一个加速器。现有的状态量化器在每生成一个 token 后就对状态进行舍入，并报告称除非提高比特宽度或对模型进行微调，否则四比特精度会带来损失。我们提出了 PACE，一种事后的四比特状态量化器。PACE 每隔几个生成的 token 才对状态进行一次舍入——正如分块内核在预填充阶段每个块只执行一次那样——并将中间生成的 token 保存在一个小缓冲区中。降低舍入频率减少了在状态中不断累积的误差。此外，我们发现门控增量（gated-delta）状态中的大数值条目会聚集在某些列上，原因是每次 token 写入的更新中，某些通道的数值在许多 token 之间持续保持较大。PACE 基于这一通道对齐的特性……

    arXiv:2610.29545v1 Announce Type: new  Abstract: Linear-attention language models keep their context in a recurrent state of fixed size, where an inference system stores one copy of this state for every sequence it decodes at the same time. Storing the state at lower precision therefore lets more sequences share an accelerator. Existing state quantizers round the state after every generated token and report that four bits lose accuracy unless the bit width is raised or the model is fine-tuned.\par We introduce PACE, a post-hoc four-bit state quantizer. PACE rounds the state only once every few generated tokens, as chunked kernels already do once per chunk during prefill, and keeps the tokens generated in between in a small buffer. Rounding less often reduces the error that accumulates in the state. Additionally, we find that the large entries of the gated-delta state gather in columns, because some channels of the update written at each token remain large across many tokens. PACE there
    
[^45]: RCSD：面向领域特定投机解码的强化学习与持续学习方法

    RCSD: Reinforcement and Continual Learning for Domain-Specific Speculative Decoding

    [https://arxiv.org/abs/2610.29527](https://arxiv.org/abs/2610.29527)

    RCSD通过强化学习与持续学习，利用投机解码中自然产生的接受/拒绝序列构建隐式偏好对来优化草稿模型，使平均接受长度较HASS提升35%，端到端推理速度提升1.35倍。

    

    投机解码通过采用一个紧凑的草稿模型提出候选token，再由目标模型并行验证，从而加速大语言模型推理。传统训练方法依赖软标签蒸馏，迫使草稿模型模仿目标模型的完整输出分布，其中包括会降低接受率、增加早期拒绝风险的高熵区域。我们提出了RCSD（面向投机解码的强化学习与持续学习方法），这是一个将草稿模型优化与最大化接受长度这一系统级目标相对齐的训练框架。RCSD利用投机推理过程中自然产生的被接受和被拒绝序列来构建隐式偏好对，并通过掩码损失避免惩罚与目标模型对齐的token。在多个基准测试上的评估表明，RCSD的平均接受长度相比HASS提升了35%，并实现了1.35倍的端到端推理加速。

    arXiv:2610.29527v1 Announce Type: new  Abstract: Speculative decoding accelerates large language model inference by employing a compact draft model to propose candidate tokens that are verified in parallel by a target model. Conventional training relies on soft-label distillation, which forces the draft model to mimic the target's full output distribution, including high-entropy regions that degrade acceptance rates and increase early rejection risk. We propose RCSD (Reinforcement and Continual Learning for Speculative Decoding), a training framework that aligns draft model optimization with the system-level objective of maximizing acceptance length. RCSD leverages naturally occurring accepted and rejected sequences during speculative inference to construct implicit preference pairs, with masked loss to avoid penalizing target-aligned tokens. Evaluated across multiple benchmarks, RCSD improves average acceptance length by 35% over HASS and achieves 1.35x faster end-to-end inference tha
    
[^46]: pplx-embed-v2-late 技术报告

    pplx-embed-v2-late Technical Report

    [https://arxiv.org/abs/2610.29523](https://arxiv.org/abs/2610.29523)

    Perplexity 推出的 pplx-embed-v2-late 包含 0.6B 和 9B 两个多语言多模态后期交互模型，通过 18B 教师模型的表示蒸馏训练，在文本、图像和视觉文档检索上达到同规模最先进水平，并借助共享嵌入空间支持“9B 建索引、0.6B 低延迟查询”的非对称部署。

    

    在本报告中，我们介绍了 pplx-embed-v2-late，这是一对参数量分别为 0.6B 和 9B 的多语言、多模态后期交互（late-interaction）模型。两款模型在各自规模级别中，于一系列文本、图像和视觉文档检索基准测试上均取得了最先进的结果。在视觉文档检索任务上，pplx-embed-v2-late-0.6b 与比它大数倍的模型相比也具有竞争力。这些模型通过从一个 18B 参数的教师模型进行表示蒸馏而训练得到。因此，它们共享一个嵌入空间，支持非对称部署：可以使用 9B 模型对文档建立索引，并使用低延迟的 0.6B 模型进行查询。两款模型均可在 https://huggingface.co/collections/perplexity-ai/pplx-embed-v2 获取。

    arXiv:2610.29523v1 Announce Type: cross  Abstract: In this report, we introduce pplx-embed-v2-late, a pair of 0.6B and 9B multilingual and multimodal late-interaction models. Both models achieve state-of-the-art results within their respective size classes across a range of text, image, and visual document retrieval benchmarks. On visual document retrieval tasks, pplx-embed-v2-late-0.6b is competitive with models several times larger than itself. The models are trained via representation distillation from an 18B teacher. Consequently, they share an embedding space that supports asymmetric deployment: documents can be indexed with the 9B model and queried with the low-latency 0.6B model. Both models are available at https://huggingface.co/collections/perplexity-ai/pplx-embed-v2.
    
[^47]: 超越投票：大语言模型何时能为基于检索的表格分类增加价值？

    Beyond the Vote: When Do LLMs Add Value to Retrieval-Based Tabular Classification?

    [https://arxiv.org/abs/2610.29520](https://arxiv.org/abs/2610.29520)

    该研究在19个表格分类数据集上将大语言模型与距离加权最近邻投票进行系统比较，发现Distance-KNN整体平均准确率最高，LLM仅在检索邻居标签一致性低且测试样本距离较近或中等时才有优势。

    

    基于检索的上下文学习（ICL）使大语言模型（LLM）能够从相似示例中预测表格标签，但LLM推理是否比直接对相同示例进行投票更优仍不清楚。我们通过将四个通用LLM与距离加权最近邻投票（Distance-KNN）在19个表格分类数据集上使用完全相同的检索邻居进行比较来解答这一问题。在演示样本数量固定的情况下，扩大检索候选池可以提升所有预测器的准确率，但Distance-KNN在每个候选池规模下都取得了最高的平均准确率。更细致的分析显示，LLM的收益主要集中在检索邻居标签一致性较低的场景，且在不同模型之间存在差异。这些收益还进一步取决于测试查询在特征空间中的平均距离：在低标签一致性情况下，部分LLM在近邻和中等距离邻域中优于Distance-KNN，但……

    arXiv:2610.29520v1 Announce Type: cross  Abstract: Retrieval-based in-context learning (ICL) enables large language models (LLMs) to predict tabular labels from similar examples, but it remains unclear whether LLM inference improves on direct voting over those same examples. We address this question by comparing four general-purpose LLMs with distance-weighted nearest-neighbor voting (Distance-KNN) using identical retrieved neighbors across 19 tabular classification datasets. With the number of demonstrations fixed, enlarging the retrieval candidate pool improves accuracy for all predictors, yet Distance-KNN achieves the highest mean accuracy at every pool size. More detailed analyses reveal that LLM gains are concentrated where retrieved neighbors exhibit low label consensus and vary across models. These gains further depend on the average feature-space distance to the test query: under low label consensus, some LLMs outperform Distance-KNN in close and intermediate neighborhoods, but
    
[^48]: 利用上下文感知解码缓解思维链中的查询捷径问题

    Mitigating Query Shortcuts in Chain-of-Thought with Context-Aware Decoding

    [https://arxiv.org/abs/2610.29518](https://arxiv.org/abs/2610.29518)

    提出推理依据对比解码（RCD）方法，通过对比查询在有无推理依据时的词元分布来放大推理依据的作用，从而缓解LLM在思维链推理中依赖查询捷径的行为并提升答案准确率。

    

    虽然思维链提示提升了大型语言模型（LLM）的推理性能，但近期研究表明，最终答案并不总是依赖于所生成的推理依据。特别是，LLM可能依赖于针对查询本身的捷径，而非其推理步骤所提供的信息。在本文中，我们研究了上下文感知解码——一种已有的对比解码方法——在应用于思维链推理时能否缓解这种捷径行为。我们将这一应用称为推理依据对比解码（RCD），该方法对比查询在有无推理依据条件下预测的下一个词元分布，从而放大那些因推理依据而似然度增加的词元。我们通过KL正则化的奖励最大化目标对这一解码规则给出了理论解释。在五个推理基准上的实验表明，RCD相比标准解码方法提升了答案准确率。

    arXiv:2610.29518v1 Announce Type: new  Abstract: While Chain-of-Thought (CoT) prompting improves the reasoning performance of large language models (LLMs), recent studies have shown that final answers do not always depend on the generated rationales. In particular, LLMs may rely on query-specific shortcuts rather than the information provided by their reasoning steps. In this paper, we investigate whether context-aware decoding (CAD), an existing contrastive decoding method, can mitigate such shortcut behavior when applied to CoT reasoning. We study this application as Rationale Contrastive Decoding (RCD), which contrasts the next-token distributions conditioned on the query with and without its rationale, amplifying tokens whose likelihood increases due to the rationale. We provide a theoretical interpretation of this decoding rule via a KL-regularized reward-maximization objective. Experiments across five reasoning benchmarks demonstrate that RCD improves answer accuracy over standar
    
[^49]: 计时与理解时间：二者皆无法单独解释大语言模型的时间推理能力

    Keeping Time vs. Understanding Time: Neither Alone Explains Temporal Reasoning in LLMs

    [https://arxiv.org/abs/2610.29507](https://arxiv.org/abs/2610.29507)

    该论文通过微调迁移实验和基于认知需求的评估，挑战了将时间推理划分为“时间算术”与“时间语义”的主流二分法，证明二者都不能单独解释大语言模型的时间推理能力。

    

    人们普遍假设时间推理可以分解为两种模式：时间算术（日期与时间的计算）和时间语义（理解时间与因果关系）。这种二分法塑造了基准测试的设计方式、模型的训练方式以及进展的衡量标准，然而其经验有效性从未被直接检验过。我们提供了两条挑战这一假设的证据。首先，我们检验被基准标注为算术或语义的任务是否能在其所属模式内带来可迁移的收益。在这些子集上进行微调主要产生数据集特定的收益（+6.8 至 +23.1 个百分点），而迁移到同一模式下的不同数据集时收益仅为 -3.2 至 +1.1 个百分点。在迁移分析涉及的两个基准测试家族中，这一模式几乎无法证明存在连贯且可迁移的时间能力。其次，使用基于认知需求的模型评估方法……

    arXiv:2610.29507v1 Announce Type: new  Abstract: Temporal reasoning is widely assumed to decompose into two modes: temporal arithmetic (date and time calculations), and temporal semantics (understanding temporal and causal relations). This dichotomy shapes how benchmarks are designed, how models are trained, and how progress is measured, yet its empirical validity has never been directly tested. We provide two lines of evidence that challenge it. First, we test whether tasks labelled as arithmetic or semantic by the benchmarks support transferable gains within their purported mode. Finetuning on these subsets yields predominantly dataset-specific gains ($+6.8$ to $+23.1$ percentage points), while transfer to a different dataset in the same mode ranges from $-3.2$ to $+1.1$ percentage points. Across the two benchmark families used in the transfer analysis, this pattern provides little evidence of coherent, transferable temporal abilities. Second, using cognitive-demand-based model evalu
    
[^50]: 一个强大的MoE就是过程监督所需要的一切

    A Strong MoE Is All You Need for Process Supervision

    [https://arxiv.org/abs/2610.29498](https://arxiv.org/abs/2610.29498)

    该论文的核心创新在于发现MoE模型的专家路由统计天然蕴含过程置信度信号，据此提出RoCo-Rethink与RoCo-PRM，利用路由置信度自动标注推理轨迹并选择性重生成低置信步骤，从而无需人工标注即可实现可靠的过程监督。

    

    过程奖励模型（PRM）为多步推理提供了细粒度的监督，然而获取可靠的过程标签仍然是一个核心瓶颈。现有方法通常依赖人工标注、基于 rollout 的估计或辅助验证。在本工作中，我们探索了一种由混合专家模型天然暴露出的不同过程监督来源：专家路由。我们证明，简单的路由统计量能够提供一种内在的过程置信度信号，其表现可与传统的输出 logit 置信度相媲美。基于这一观察，我们通过结合整体路由质量、注意力头路由质量以及步内路由变化来构建路由置信度，并提出了 RoCo-Rethink，它能够识别置信度最低的推理步骤，并选择性地重新生成后续轨迹。我们进一步提出了 RoCo-PRM，它利用一个强大的 MoE 模型，通过路由置信度自动为推理轨迹进行标注……（原文摘要至此截断）

    arXiv:2610.29498v1 Announce Type: new  Abstract: Process reward models (PRMs) provide fine-grained supervision for multi-step reasoning, yet obtaining reliable process labels remains a central bottleneck. Existing approaches typically rely on human annotation, rollout-based estimation, or auxiliary verification. In this work, we explore a different source of process supervision natively exposed by Mixture-of-Experts (MoE) models: expert routing. We show that simple routing statistics provide an intrinsic process-confidence signal that is competitive with conventional output-logit confidence. Building on this observation, we construct routing confidence by combining overall routing mass, head routing mass, and within-step routing variation, and introduce RoCo-Rethink, which identifies the least confident reasoning step and selectively regenerates the subsequent trajectory. We further introduce RoCo-PRM, which uses a strong MoE to automatically annotate reasoning trajectories with routin
    
[^51]: TenetBench：一个以条款为锚点的俄语语料库，用于依据职业伦理准则评估语言模型

    TenetBench: A Clause-Anchored Russian Corpus for Evaluating Language Models against Professional Codes of Ethics

    [https://arxiv.org/abs/2610.29477](https://arxiv.org/abs/2610.29477)

    提出了 TenetBench——一个由自动化流水线从职业伦理准则文本生成的俄语基准语料库，包含 3,433 个与准则条款冲突的请求及其良性孪生对照，可随着新准则加入持续扩展，用于精细评估大语言模型是否会违反或帮助用户违反具体的职业伦理条款。

    

    职业伦理准则指导和约束专业人士，而这些人越来越频繁地使用基于大语言模型构建的智能助手。这样的助手可能会在用户毫无察觉的情况下违反用户所遵循的职业准则，或帮助用户违反准则。仅报告广义伤害类别的评估无法揭示这种情况发生的频率，也无法指明具体违反了哪项条款。TenetBench 将 3,433 个与十五部俄语职业伦理准则条款相冲突的请求，与仅在所请求内容合法性上存在差异的良性孪生请求一一配对。一个自动化流水线从准则文本中生成任务，因此语料库会随着每部新准则的加入而扩展，且每次运行都会从同一条款产生新的任务。该流水线还为两部来自其他司法辖区的英语准则生成了任务和经人工验证的判定结果。在对冻结流水线的人工验证中，专家认为 0.995 的冲突任务是合理的，且模型裁判与人工裁定的标签一致性达到 Gwet's AC1 的……（原文此处截断）

    arXiv:2610.29477v1 Announce Type: new  Abstract: Codes of professional ethics guide and restrict specialists, who increasingly work with assistants built on large language models. Such an assistant can breach the user's code, or help the user breach it, without the user noticing. Evaluations reporting broad categories of harm cannot show how often this happens or which provision is breached. TenetBench pairs 3,433 requests conflicting with clauses of fifteen Russian-language professional codes with benign twins that differ only in the legitimacy of what is asked. An automated pipeline generates the tasks from the text of a code, so the corpus grows with every code added and a provision yields new tasks on every run. It also produced tasks and human-validated verdicts from two English codes of other jurisdictions. In human validation of the frozen pipeline, specialists rated 0.995 of the conflicting tasks plausible, and the model judge agreed with the adjudicated label at Gwet's AC1 of 
    
[^52]: OmniSimulator：对齐小型语言模型以实现真实的异构行为建模

    OmniSimulator: Aligning Small Language Models for Authentic Heterogeneous Behavior Modeling

    [https://arxiv.org/abs/2610.29472](https://arxiv.org/abs/2610.29472)

    本文提出 OmniSimulator 训练框架，通过五维结构化行为推理模式、精选真实用户行为数据以及教师模型生成的高质量推理轨迹，对齐小型语言模型，从而显著提升其在短视频平台上进行真实异构用户行为模拟的预测能力。

    

    大型语言模型（LLM）作为交互系统的通用用户模拟器已展现出巨大潜力。先前的研究表明，在高质量数据上进行训练能够显著提升模型的推理能力和下游任务表现。然而，用户模拟的上限仍受制于模型自身的推理范式：原生的思维链往往退化为浅层的模式匹配，无法充分建模真实用户决策背后的行为因素。在本文中，我们提出了 OmniSimulator，一种新颖的训练框架，旨在提升小型语言模型在短视频平台环境中进行真实用户模拟的预测能力。具体而言，OmniSimulator 通过一个包含五个维度的结构化模式来组织面向行为的推理，从真实用户行为场景中精选数据，并引入教师模型来生成高质量的推理轨迹。

    arXiv:2610.29472v1 Announce Type: new  Abstract: Large language models (LLMs) have shown great potential as general-purpose user simulators for interactive systems. Prior studies have demonstrated that training on high-quality data can substantially improve model reasoning and downstream task performance. However, the upper bound of user simulation is still constrained by the model's reasoning paradigm: native chain-of-thought often degenerates into shallow pattern matching and fails to adequately model the behavioral factors underlying real user decisions. In this paper, we propose OmniSimulator, a novel training framework designed to improve the predictive ability of small LLMs for realistic user simulation on short-video platform environments. Specifically, OmniSimulator organizes behavior-oriented reasoning through a structured schema with five dimensions, carefully selects data from real user-behavior scenarios, and introduces a teacher model to generate high-quality reasoning tra
    
[^53]: 面向可泛化智能体强化学习的框架扩展

    Harness Scaling for Generalizable Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.29458](https://arxiv.org/abs/2610.29458)

    提出Harness Scaling这一智能体强化学习的新扩展维度，将训练从单一固定框架扩展到多样化执行配置，从而提升模型在不同框架配置下的泛化与迁移能力。

    

    智能体框架已成为现代智能体系统的核心组成部分，塑造着模型使用工具、管理上下文以及协调长时程交互的方式。要充分发挥这些框架的收益，取决于底层模型利用它们的有效程度，这促使人们直接通过框架进行强化学习。然而，现有方法通常仅在单一固定框架下进行训练，这会导致模型过度专门化，并在不同框架配置之间的迁移能力受限。为解决这一问题，我们提出了Harness Scaling（框架扩展），这是智能体强化学习的一个全新扩展维度，将训练从单一框架扩展到多样化的执行配置。具体而言，我们构建了一个因子化的框架空间，选取具有均衡因子覆盖的紧凑训练子集，并组织跨框架训练，将全局覆盖与同一任务的可控变化相结合。在多个基准测试、模型规模以及（摘要在此处被截断）

    arXiv:2610.29458v1 Announce Type: cross  Abstract: Agent harnesses have become a central component of modern agentic systems, shaping how models use tools, manage context, and coordinate long-horizon interaction. Realizing the benefits of these harnesses depends on how effectively the underlying model can make use of them, motivating reinforcement learning directly through the harness. However, existing approaches typically train through a single fixed harness, which can lead to specialization and limited transfer across harness configurations. To address this, we introduce \textbf{Harness Scaling}, a new scaling dimension for agentic RL that extends training from a single harness to diverse execution configurations. Concretely, we construct a factorized harness space, select a compact training subset with balanced factorial coverage, and organize cross-harness training to combine global coverage with controlled variation for the same task. Across multiple benchmarks, model scales, and
    
[^54]: Omnilingual-MsASR：面向全模态大语言模型的多模态、多语言、长时多说话人语音识别基准测试

    Omnilingual-MsASR: Benchmarking Multimodal, Multilingual, Long-Form Multi-Speaker ASR of Omni Large Language Models

    [https://arxiv.org/abs/2610.29455](https://arxiv.org/abs/2610.29455)

    该论文提出了Omnilingual-MsASR，一个涵盖六种语言、最长30分钟、最多20个说话人的多语言多模态长时多说话人语音识别基准，揭示出当前全模态大语言模型在该任务上存在严重性能瓶颈，tcpWER经常超过100%。

    

    全模态大语言模型（Omni LLM）在音频和视频理解方面取得了显著进展。然而，现有模型在处理长时、多说话人视频时表现不佳，尤其是在理解整个视频以及识别“谁在什么时间说了什么”方面。为了刻画当前全模态大语言模型的能力极限并指导未来的模型开发，我们提出了Omnilingual-MsASR，这是一个面向长时多说话人自动语音识别的多语言、多模态基准测试，涵盖六种语言、长达30分钟的录音以及最多包含20个说话人的场景。Omnilingual-MsASR提供并行的音频和视频输入，以评估模型的跨模态一致性以及利用跨模态互补线索的能力。对领先的全模态大语言模型进行的广泛评估揭示了显著的性能瓶颈，其tcpWER经常超过100%。这些发现凸显了细粒度时间定位（摘要原文在此处截断）……

    arXiv:2610.29455v1 Announce Type: new  Abstract: Omni-modal large language models (LLMs) have achieved remarkable progress in audio and video understanding. However, existing models struggle with long-form, multi-speaker videos, especially in understanding the entire video and recognizing who says what and when. To characterize the capability limit of current omni LLMs and guide future model development, we introduce Omnilingual-MsASR, a multilingual, multimodal benchmark for long-form Multi-speaker Automatic Speech Recognition (MsASR) covering six languages, recordings up to 30 minutes, and scenarios with up to 20 speakers. Omnilingual-MsASR provides parallel audio and video inputs to evaluate models' cross-modal consistency and their ability to exploit complementary cues across modalities. Extensive evaluations of leading omni-modal LLMs reveal substantial performance bottlenecks, while their tcpWERs frequently exceed 100%. These findings highlight fine-grained temporal localization,
    
[^55]: DynaWeave：通过动态场景合成学习适应

    DynaWeave: Learning to Adapt through Dynamic Scenario Synthesis

    [https://arxiv.org/abs/2610.29443](https://arxiv.org/abs/2610.29443)

    提出DynaWeave自动化框架，通过生成具有依赖关系的外生事件将静态LLM智能体任务转化为可验证的动态场景，从而训练和评估智能体应对环境突变的能力。

    

    大语言模型（LLM）智能体越来越多地使用工具在有状态环境中完成任务。然而，除了智能体自身行动引起的变化之外，许多训练环境在很大程度上保持静态，而真实环境中由其他用户、服务或突发意外引入的外部事件可能会使正在执行的计划失效。为解决这一差距，我们提出了DynaWeave，一个将现有静态任务转化为可验证动态场景的自动化合成框架。该框架为每个任务生成相关的外生事件，涵盖任务需求、环境状态和外部信息的变化。它使用有向无环图（DAG）来表示事件之间的依赖关系，并通过执行引擎在智能体交互过程中激活这些事件。动态验证器仅修改受实际发生事件影响的评估检查，并保留所有其他检查。使用DynaWeave，我们收集了2,095（个……摘要在此处被截断）

    arXiv:2610.29443v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly use tools to complete tasks in stateful environments. However, many training environments remain largely static apart from changes caused by the agent's own actions, whereas external events introduced by other users, services, or sudden contingencies in real environments can invalidate ongoing plans. To address this gap, we introduce DynaWeave, an automated synthesis framework that transforms existing static tasks into verifiable dynamic scenarios. The framework generates exogenous events relevant to each task, covering changes in task requirements, environment state, and external information. It uses directed acyclic graphs (DAGs) to represent dependencies between events and an execution engine to activate them during agent interaction. A dynamic verifier revises only the evaluation checks affected by events that actually occur and retains all other checks. Using DynaWeave, we collect 2,095
    
[^56]: BIRD-RL：在有状态的数据中心环境中扩展智能体强化学习

    BIRD-RL: Scaling Agentic Reinforcement Learning over Stateful Data-Centric Environments

    [https://arxiv.org/abs/2610.29426](https://arxiv.org/abs/2610.29426)

    BIRD-RL提出了一种四阶段智能体强化学习课程方法，通过交替增强基础推理与轨迹自适应在线探索，解决了SQL调试等有状态数据中心环境中环境隔离成本高、状态污染和跨轨迹干扰导致的训练不稳定问题。

    

    尽管智能体强化学习（RL）已成为训练小型语言模型（SLM）的一种有前景的范式，但现有框架主要面向以工具为中心的场景，难以应对以数据为中心的环境。具体而言，将工具中心型框架应用于SQL调试等有状态任务会带来严重的空间低效问题，因为合理的环境隔离通常需要为每次rollout分配一个独立的数据库容器。而在多轮交互中简单地复用数据库、不跟踪其不断演变的状态，则会导致轨迹污染和跨轨迹干扰，严重破坏策略更新的稳定性。为克服这些瓶颈，我们提出了BIRD-RL，一种新颖的四阶段智能体强化学习课程方法，通过交替增强基础智能体推理能力与轨迹自适应的在线策略探索，规避直接端到端学习固有的不稳定性。关键在于，为支持这一课程学习……

    arXiv:2610.29426v1 Announce Type: cross  Abstract: While agentic reinforcement learning (RL) has emerged as a promising paradigm for training small language models (SLMs), existing frameworks are predominantly tool-centric and struggle with data-centric environments. Specifically, applying tool-centric frameworks to stateful tasks like SQL debugging introduces severe spatial inefficiencies since proper environment isolation typically requires a separate database container per rollout. Naively reusing databases without tracking evolving states across multi-turn interactions causes trajectory contamination and cross-trajectory interference, which severely destabilizes policy updates. To overcome these bottlenecks, we introduce BIRD-RL, a novel four-stage agentic RL curriculum that alternately enhances foundational agent reasoning and trajectory-adaptive on-policy exploration to circumvent the instability inherently associated with direct end-to-end learning. Crucially, to support this cu
    
[^57]: 一种基于KAN的非线性分解方法用于嵌入层压缩

    A KAN-based Non-linear Factorization for Embedding Layer Compression

    [https://arxiv.org/abs/2610.29424](https://arxiv.org/abs/2610.29424)

    提出一种基于Kolmogorov-Arnold网络（KAN）的非线性分解方法，通过在压缩嵌入层中引入可学习的非线性投影，突破传统低秩分解的线性瓶颈，提升嵌入层压缩后的表示能力。

    

    在现有的嵌入压缩方法中，低秩分解通过使用降秩分解来近似嵌入矩阵，提供了一种简单且具有数学原理依据的解决方案。然而，其实际效果往往有限，因为由此产生的线性瓶颈限制了表示能力。为了克服这些限制，我们提出了一种增强的分解方案，通过在压缩的嵌入层中引入可学习的非线性投影来提高表达能力。我们的方法受到Kolmogorov-Arnold网络（KAN）范式的启发，该范式用可学习的单变量函数替代固定的线性变换。尽管KAN在其他领域取得了令人瞩目的成果，但其在自然语言处理（NLP）中的应用仍然稀缺，且迄今为止主要局限于最终层或任务特定层，而非嵌入层等核心表示组件。我们通过将所提出的方法集成……（原文摘要到此截断）

    arXiv:2610.29424v1 Announce Type: new  Abstract: Among existing embedding compression approaches, low-rank factorization provides a simple and mathematically principled solution by approximating the embedding matrix with a reduced-rank decomposition. However, its practical effectiveness is often limited, as the resulting linear bottleneck constrains representational capacity. To overcome these limitations, we propose an enhanced factorization scheme that increases expressive power by introducing learnable non-linear projections within the compressed embedding layer. Our method is inspired by the Kolmogorov Arnold Networks (KAN) paradigm, which replaces fixed linear transformations with learnable univariate functions. Despite promising results in other domains, applications of KANs in NLP remain scarce and have so far been largely limited to final or task specific layers, rather than core representational components such as embedding layers.   We evaluate the proposed approach by integr
    
[^58]: CCG Diffusion：基于组合范畴语法的组合式且句法消歧的文生图生成

    CCG Diffusion: Combinatory Categorial Grammar-Driven Compositional and Syntactically Disambiguated Text-to-Image Generation

    [https://arxiv.org/abs/2610.29415](https://arxiv.org/abs/2610.29415)

    该论文提出了免训练的CCG Diffusion方法，利用组合范畴语法（CCG）中词类别所携带的丰富句法信息，在文生图生成中实现对属性绑定、空间关系、数量控制以及句法歧义提示词预期解释的精确控制。

    

    文生图生成模型并未显式接收提示词的句法结构，即哪个词修饰哪个词、每个短语附着于何处。因此，这些模型常常难以将属性绑定到正确的对象上、难以正确描绘空间关系和数量，也难以按照预期的解释来呈现具有句法歧义的提示词。现有的基于句法的控制方法主要解决属性绑定问题，在控制空间关系和数量、或指定句法歧义提示词的解释方面能力有限。它们依赖于短语结构语法或依存语法，需要在树结构上附加额外的规则或约束，才能提取词与词之间的修饰关系或指定句法解释。我们提出了CCG Diffusion，这是一种免训练的方法，利用组合范畴语法（CCG）来控制图像生成。在CCG中，每个词的类别携带丰富的……（摘要内容在此处截断）

    arXiv:2610.29415v1 Announce Type: cross  Abstract: Text-to-image generation models do not explicitly receive the syntactic structure of the prompt, i.e., which word modifies which and where each phrase attaches. Consequently, they often struggle to bind attributes to the correct objects, to depict spatial relations and counts correctly, and to render syntactically ambiguous prompts in the intended interpretation. Existing syntax-based control methods mainly address attribute binding and have limited ability to control spatial relations and counts or to specify the interpretation of syntactically ambiguous prompts. They rely on phrase-structure or dependency grammars and require additional rules or constraints on the tree structure to extract modification relations between words or to specify a syntactic interpretation. We propose CCG Diffusion, a training-free method that controls image generation using Combinatory Categorial Grammar (CCG). In CCG, the category of each word carries ric
    
[^59]: 超越闲聊：人机对话中的自我披露轨迹建模

    Beyond Small Talk: Modeling Disclosure Trajectories in Human-AI Conversation

    [https://arxiv.org/abs/2610.29413](https://arxiv.org/abs/2610.29413)

    本文提出基于社会渗透理论的TraDE评估框架，通过预定义披露轨迹来评估在当前披露相同的条件下，先前的披露历史如何影响人机对话中模型对用户自我披露的回应方式。

    

    近年来，对话式代理作为一种提供社会性和情感支持的方式日益兴起，其中自我披露会随时间逐步展开。然而，目前对这些代理的评估通常只考察孤立的对话状态。在对自然发生的对话进行分析时，披露历史与当前披露相互混淆，导致二者的效应难以被分离。为此，我们提出了基于轨迹的披露评估框架TraDE，该框架以社会渗透理论为基础，利用预定义的披露轨迹，评估在当前披露水平相同的条件下，先前的披露历史如何塑造模型的行为。通过使用TraDE，我们在多个模型和关系行为基准上证明：即使披露深度相互匹配，模型对披露的回应仍可能因先前的轨迹不同而有所差异，尤其体现在模型是接受还是挑战用户的表述框架上。我们还观察到（摘要在此处截断）……

    arXiv:2610.29413v1 Announce Type: new  Abstract: Recently, we see a rise in conversational agents as a form of social and emotional support, where disclosure will unfold over time. Yet, current evaluations of these agents typically examine isolated conversational states. In analyses of naturally occurring conversations, disclosure history and current disclosure are confounded which tends to make their effects difficult to isolate. We thus introduce Trajectory-based Disclosure Evaluation (TraDE), an evaluation framework grounded in Social Penetration Theory that utilizes predefined disclosure trajectories and evaluates how preceding disclosure history shapes model behavior under matched current disclosure. Through the use of TraDE, we demonstrate across multiple models and relational behavior benchmarks that responses to disclosures matched on depth can differ depending on the preceding trajectory, particularly in whether models accept or challenge the user's framing. We also observed t
    
[^60]: ECHO：涌现的共模隐藏偏移驱动偏好优化中的词表坍缩

    ECHO: Emergent Common-Mode Hidden Offsets Drive Vocabulary Collapse in Preference Optimization

    [https://arxiv.org/abs/2610.29410](https://arxiv.org/abs/2610.29410)

    本文发现偏好优化中存在一种相对损失无法察觉的共模隐藏偏移，它会在输出层形成“词表井”，使单一词元概率主导解码并最终导致生成坍缩。

    

    偏好优化可以在改善所有排序指标的同时，积累一种相对损失几乎无法察觉的隐藏共享表征偏移，而这种偏移最终可能破坏模型的生成能力。我们将这种失败追溯到“共模偏移”，即模型隐藏状态中出现的一种共享方向性偏移，它出现在不同的词元位置、提示词以及被选择和被拒绝的回复之中。由于它使被选择和被拒绝的表征一同移动，相对排序损失对其积累几乎没有反应。随着这种偏移不断增长，它会投影到输出层形成“词表井”，使单个词元的概率持续升高，直至主导解码过程并导致生成坍缩。实验表明：仅这一孤立的偏移就足以重建词表井；将学习到的偏移注入未经处理的基座模型可以重现这种坍缩，而范数匹配的随机方向则不能；从损失中屏蔽这一共同成分则可以防止词表井的形成。

    arXiv:2610.29410v1 Announce Type: new  Abstract: Preference optimization can improve every ranking metric while building a hidden, shared representation shift that the relative loss barely sees, one that can eventually destroy generation. We trace this failure to the common-mode offset, a shared directional shift in the model's hidden states that appears across token positions, prompts, and both chosen and rejected responses. Because it moves chosen and rejected representations together, the relative ranking loss barely reacts to its accumulation. As this offset grows, it projects onto the output layer to form a vocabulary well, elevating a single token's probability until it dominates decoding and causes generation collapse. The isolated offset alone reconstructs the vocabulary well, injecting the learned offset into an untreated base model recreates the collapse where norm-matched random directions do not, and masking this common component from the loss prevents the well in our flags
    
[^61]: Right On Q：通过自生成轨迹恢复量化模型的推理能力

    Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories

    [https://arxiv.org/abs/2610.29378](https://arxiv.org/abs/2610.29378)

    提出OnQ流水线，通过在策略蒸馏利用学生模型自生成的轨迹进行训练，解决了固定响应监督导致的训练与生成前缀不匹配及高熵处分布失配问题，从而恢复低比特量化大模型的推理能力。

    

    量化感知训练（QAT）旨在恢复低比特大语言模型（LLM）的推理能力。现有方法通常利用原始的全精度模型在固定的响应序列上对量化学生模型进行监督。然而，由此得到的低比特量化模型在长程推理任务上仍然表现不佳，并且常常无法在足以满足全精度模型的token预算内正常终止。我们识别出了缩小这一差距的两个挑战：第一，固定响应监督导致训练前缀与学生模型实际生成时遇到的前缀之间存在不匹配，且随着响应变长，这种轨迹分歧会不断增大；第二，在学生模型自身的前缀上，它会对某些下一个token候选词预测不足，而对另一些候选词预测过度，这种分布失配在高熵前缀处出现得更加频繁。我们提出了OnQ，一个低比特恢复流水线，它结合了在策略蒸馏与候选（摘要在此处被截断）

    arXiv:2610.29378v1 Announce Type: new  Abstract: Quantization-aware training (QAT) aims to recover reasoning capabilities in low-bit large language models (LLMs). Existing recipes typically supervise the quantized student with the original full-precision model on fixed response sequences. However, the resulting low-bit quantized models still struggle on long-horizon reasoning tasks and often fail to terminate within a token budget that suffices for the full-precision model. We identify two challenges in closing this gap. First, fixed-response supervision leaves a mismatch between training prefixes and those encountered during student generation, and this trajectory divergence grows as responses lengthen. Second, on its own prefixes, the student underpredicts some next-token candidates and overpredicts others, and this distributional misalignment becomes more frequent at high-entropy prefixes. We present OnQ, a low-bit recovery pipeline that combines on-policy distillation and candidate
    
[^62]: 面向AI智能体的以记忆为中心的持续学习

    Memory-Centric Continual Learning for AI Agents

    [https://arxiv.org/abs/2610.29357](https://arxiv.org/abs/2610.29357)

    本文提出以记忆为中心的持续学习框架（MCCL），主张以记忆能否改善智能体未来行动而非检索效果作为评判标准，并通过三轴记忆分类体系和内外双循环的生命周期机制，使AI智能体能够真正实现从经验中持续学习。

    

    由基础模型驱动的自主智能体能够进行推理、规划和使用工具，但其固定的权重和有限的上下文窗口阻碍了它们在长期部署中从经验里持续学习。尽管外部记忆系统试图弥合这一差距，但它们大多充当被动的检索缓存，评估标准主要是能否召回存储的事实或过往交互。记忆能否驱动真正的持续学习仍是一个悬而未决的问题。在本文中，我们提出了以记忆为中心的持续学习（MCCL），主张应当以记忆能否改善未来的行动来评判其价值，而不仅仅是看它检索到什么。MCCL包含三个主要部分：首先，一个三轴分类体系从状态存储的位置（基底）、状态的形态（资产）以及状态的管理方式（策略）三个维度组织记忆；其次，定义了一个双循环生命周期，在内循环中更新记忆状态，同时在外循环中基于任务反馈不断完善管理策略。

    arXiv:2610.29357v1 Announce Type: new  Abstract: Autonomous agents powered by foundation models can reason, plan, and use tools, but their fixed weights and context limits hinder sustained learning from experience over long-term deployment. While external memory systems attempt to bridge this gap, they mostly act as passive retrieval caches, evaluated primarily by whether they can recall stored facts or past interactions. Whether memory can drive genuine continual learning remains an open question. In this paper, we introduce Memory-Centric Continual Learning (MCCL), arguing that memory should be judged by whether it improves future actions, not merely what it retrieves. MCCL has three main parts. First, a three-axis taxonomy organizes memory by where state is stored (substrates), what form it takes (assets), and how it is managed (policies). This defines a two-loop lifecycle that updates memory state in an inner loop while refining management policies in an outer loop based on task fe
    
[^63]: 不要过早停止长上下文持续预训练：一条经验缩放定律

    Don't Stop Long-Context Continual Pre-Training Too Early: An Empirical Scaling Law

    [https://arxiv.org/abs/2610.29352](https://arxiv.org/abs/2610.29352)

    本文建立了一条经验缩放定律，表明长上下文持续预训练的性能随模型规模与训练token数按加性幂律持续提升，因此不应过早停止LCCP训练。

    

    长上下文持续预训练（LCCP）被广泛用于提升模型的长上下文能力，但其收益如何随LCCP所使用的token数量进行扩展仍不明确。我们通过对LCCP扩展收益进行建模、监测和验证来研究这一问题。首先，我们建立了一条经验性的加性幂律，将长上下文性能与模型规模以及短上下文预训练和LCCP各自的token预算联系起来。在五个模型规模和一百多个短上下文与长上下文预算配置上的实验表明，拟合出的定律能够重现观察到的趋势，并能很好地外推到留出的模型规模上。其次，我们提出了Constant-Last4，这是一种训练与观测协议，通过对恒定学习率训练中最近的四个连续检查点取平均，从而在单一可扩展的训练轨迹上实现稳定的监测。在该协议下，同样的缩放[摘要在此处截断]

    arXiv:2610.29352v1 Announce Type: new  Abstract: Long-context continual pre-training (LCCP) is widely used to improve long-context capability, but how its benefits scale with the number of tokens used for LCCP remains unclear. We investigate this question through modeling, monitoring, and validating the benefits of scaling LCCP. First, we establish an empirical additive power law relating long-context performance to model size and the respective token budgets of short-context pre-training and LCCP. Experiments across five model scales and more than one hundred short- and long-context budget configurations show that the fitted law reproduces the observed trends and extrapolates well to held-out model scales. Second, we introduce \textsc{Constant-Last4}, a training and observation protocol that averages the four most recent consecutive checkpoints from a constant-learning-rate run to enable stable monitoring along a single extensible training trajectory. Under this protocol, the same sca
    
[^64]: 跨语言语音克隆中用于口音控制的闭环双参考引导方法

    Closed-Loop Dual-Reference Guidance for Accent Control in Cross-Lingual Voice Cloning

    [https://arxiv.org/abs/2610.29351](https://arxiv.org/abs/2610.29351)

    提出闭环双参考引导方法，通过将原始参考与语音转换构建的母语锚点配对并动态更新语言模型状态，有效消除跨语言语音克隆中的源语言口音，在仅损失极小说话人相似度的前提下显著提升目标语言的发音质量。

    

    零样本跨语言语音克隆常常会将参考说话人的口音带入目标语言中。我们提出闭环双参考引导方法，将原始参考与通过语音转换（VC）构建的可复用母语锚点进行配对。这两个参考分别初始化各自独立的语言模型（LM）状态，共同引导一条共享的声学轨迹；每个完成的语音片段随后会更新这两个状态，使其生成历史与实际输出的语音保持对齐。锚点构建每个说话人仅需执行一次，在线合成使用冻结的TTS模型，无需逐句进行语音转换。在20个保留测试的法语说话人上，固定设置 λ*=0.33 与直接克隆相比，将残余的法语LID质量降低了73%，ECAPA相似度损失仅为0.042；相对于仅使用锚点的克隆，该方法恢复了0.155的相似度。在保留测试组中，该方法实现了最低的平均WER（0.009）和最高的平均GOP（4.750）。

    arXiv:2610.29351v1 Announce Type: cross  Abstract: Zero-shot cross-lingual voice cloning often carries the reference speaker's accent into the target language. We propose closed-loop dual-reference guidance, which pairs the original reference with a reusable native anchor constructed by voice conversion (VC). The references initialize separate language-model (LM) states that guide a shared acoustic trajectory; each completed patch then updates both states, keeping their generated histories aligned with the emitted speech. Anchor construction runs once per speaker, and online synthesis uses a frozen TTS model without per-utterance VC. On 20 held-out French speakers, the fixed setting $\lambda^{\!*}=0.33$ reduces residual French LID mass by 73\% versus direct cloning, at an ECAPA similarity cost of only $0.042$; relative to anchor-only cloning, it recovers $0.155$. The method achieves the lowest mean WER ($0.009$) and highest mean GOP ($4.750$) among the held-out test arms. In blinded pr
    
[^65]: B-CARE：用于立场错误排序的盲测类别相对证据评估

    B-CARE: Blind Classwise Assessment of Relative Evidence for Stance Error Ranking

    [https://arxiv.org/abs/2610.29350](https://arxiv.org/abs/2610.29350)

    B-CARE提出了一种预测盲的高风险立场错误排序框架，利用固定的LLM验证器对比各类立场标签之间的相对证据强度来构建风险分数，且证据画像可跨异构基础模型复用而无需额外训练错误检测器。

    

    立场分类器所产生的错误在有限的人工审核预算下可能需要高昂代价才能识别出来。对潜在错误的预测进行优先级排序仍然具有挑战性，尤其是在异构的基础模型之间。我们提出了B-CARE（Blind Classwise Assessment of Relative Evidence for Stance Error Ranking，用于立场错误排序的盲测类别相对证据评估），这是一个用于识别高风险立场预测的预测盲框架。在仅给定目标和文本的情况下，一个固定的LLM验证器会为所有候选立场标签分配有序的证据分数，而无需观察基础模型的预测结果或置信度。B-CARE通过对比支持预测类别的证据与最强的替代类别证据，构建出一个比较性风险分数。由此产生的类别级证据画像可以在多个基础模型之间复用，而无需训练单独的错误检测器或校准风险分数。我们在四个立场基准和三个异构基础LLM上评估了B-CARE，涵盖……

    arXiv:2610.29350v1 Announce Type: new  Abstract: Stance classifiers make errors that can be costly to identify under limited human-review budgets. Prioritizing potentially incorrect predictions remains challenging, particularly across heterogeneous base models. We introduce B-CARE (Blind Classwise Assessment of Relative Evidence for Stance Error Ranking), a prediction-blind framework for identifying high-risk stance predictions. Given only a target and text, a fixed LLM verifier assigns ordinal evidence scores to all candidate stance labels without observing the base model's prediction or confidence. B-CARE constructs a comparative risk score by contrasting the strongest alternative-class evidence with the evidence supporting the predicted class. The resulting classwise evidence profile can be reused across multiple base models without training a separate error detector or calibrating risk scores. We evaluate B-CARE on four stance benchmarks and three heterogeneous base LLMs, covering 
    
[^66]: 数据调度与学习率调度应当协同设计

    Data Schedules and Learning Rate Schedules Should Be Designed Together

    [https://arxiv.org/abs/2610.29348](https://arxiv.org/abs/2610.29348)

    研究表明数据调度与学习率调度必须协同设计：中期训练中最优的数据过渡策略依赖于学习率调度，当学习率持续衰减时，更早引入中期训练数据并与预训练数据混合能降低最终损失。

    

    中期训练是大语言模型训练流程中的一个阶段，它在规模更小但价值更高的数据集上继续进行预训练（PT）。将预训练与中期训练两个阶段作为一个整体来看，训练通常从仅使用预训练数据开始，然后完全过渡到中期训练数据。在本工作中，我们将这种从预训练数据到中期训练数据的完全过渡视为更广泛的一类数据混合调度的特例，这类调度规定了每个批次中预训练与中期训练token的比例如何随训练过程变化。在固定总训练计算量和中期训练token预算的条件下，我们提出这样的问题：这种向中期训练数据的完全过渡对于最小化最终中期训练损失是否是最优的？我们表明，答案取决于学习率（LR）调度：在某种学习率调度下，一种数据调度可能优于另一种，但仅改变学习率调度就可能逆转这一结果。具体而言，当学习率在继续训练过程中持续衰减时，更早引入中期训练数据并将其与预训练数据混合，可以降低最终的中期训练损失。

    arXiv:2610.29348v1 Announce Type: cross  Abstract: Mid-training (MT) is a training phase in LLM training pipelines that continues pretraining (PT) on smaller datasets of higher value. Taking the PT and MT phases as a whole, training usually starts with only PT data and then transitions completely to MT data. In this work, we view this complete transition from PT to MT data as a special case of a broader class of data mixing schedules that specify how the proportions of PT and MT tokens in each batch change over training. Under fixed total training compute and MT token budgets, we ask: Is this complete transition to MT data optimal for minimizing final MT loss? We show that the answer depends on the learning rate (LR) schedule: one data schedule may outperform another under one LR schedule, but changing only the LR schedule can reverse the outcome. Specifically, when the LR decays throughout continued training, introducing MT data earlier and mixing it with PT data lowers the final MT l
    
[^67]: 可解码性并非可控性：连续潜变量TTS中的局部与各向异性说话人控制

    Decodability Is Not Controllability: Local and Anisotropic Speaker Control in Continuous-Latent TTS

    [https://arxiv.org/abs/2610.29345](https://arxiv.org/abs/2610.29345)

    该研究揭示连续潜变量TTS中说话人特征的高可解码性并不等于因果可控性，有效的话者音色控制依赖于在参考音频早期位置（L0层）进行干预，而非探针准确率最高的深层。

    

    在连续潜变量文本转语音（TTS）中，高线性可解码性并不意味着因果可控性。在 VoxCPM2 中，通过跨30名说话人和60个有向说话人配对的匹配位置探针与等范数干预揭示了这种分离：最终残差声学语言模型层（RALM L7）的参考音频状态保持0.970的探针准确率，但对齐编辑在60个配对中却产生0个目标主导结果。在参考音频早期位置的编辑则呈现相反的模式：在L0层，配对对齐更新使目标说话人相似度提升0.326，并在全部60个配对中实现目标主导，而任何等范数方向或位置控制下最多仅达到1/60。独立的说话人编码器、针对保留说话人原型的检索以及双盲听测共同证实，转换后的声音被听者感知为预期的目标说话人。在 dots.tts 中的跨架构干预进一步区分了主干杠杆作用与直接声学控制：主干编辑……

    arXiv:2610.29345v1 Announce Type: cross  Abstract: High linear decodability does not entail causal controllability in continuous-latent text-to-speech (TTS). In VoxCPM2, position-matched probes and equal-norm interventions across 30 speakers and 60 directed dyads expose this separation: reference-audio states at the final residual acoustic language model layer (RALM L7) retain 0.970 probe accuracy, yet aligned edits yield 0/60 target-dominant dyads. Edits at early reference-audio positions produce the opposite pattern. At L0, pair-aligned updates increase target similarity by 0.326 and achieve target dominance in all 60 dyads, against at most 1/60 under any equal-norm direction or position control. Independent speaker encoders, retrieval against held-out speaker prototypes, and double-blind listening confirm that the shifted voice is heard as the intended target. Cross-architecture interventions in dots.tts distinguish backbone leverage from direct acoustic control: backbone edits disp
    
[^68]: 无审查不等于无偏见：“消融审查”移除了拒绝行为，却未消除关于台湾的书写文字相关偏见

    Uncensored Is Not Unbiased: Abliteration Removes Refusals but Not Script-Dependent Bias About Taiwan

    [https://arxiv.org/abs/2610.29342](https://arxiv.org/abs/2610.29342)

    去审查化（abliteration）模型只是移除了拒绝行为，其对台湾问题的政治立场主要取决于所用书写文字（繁体、简体还是英文），而非是否经过消融审查，因此“无审查”并不等于“无偏见”。

    

    “去审查化”（“无审查”）的开放权重语言模型版本通过移除介导拒绝行为的激活方向而构建，其流行的部分原因在于一种假设：不再拒绝回答的模型也就摆脱了政治审查。我们在两对匹配的2比特量化模型上对这一假设进行测试：NVIDIA 的 Nemotron 3.5 Bolt 74B-A9B 和 Qwen3.8-Flash-Next，每款模型分别与其对应的社区消融审查版本配对比较。每款模型分别在繁体中文、简体中文（字符级转换）和英文下，对22组陈述对（涉及中华民国（台湾）、1989年天安门事件及对照国家）进行判断，并针对每种文字回答27个开放式问题。结果显示，书写文字的影响远大于消融审查：原始 Bolt 模型在指数 S（-1 代表中华人民共和国（PRC）立场；+1 代表中华民国（ROC）立场）上，繁体中文得分为 +0.58，英文为 +0.57，简体中文仅为 +0.05；且该模型以 0.91 的概率在简体中文中肯定“台湾是中华人民共和国的一部分”，而在繁体中文中该概率仅为 0.06。

    arXiv:2610.29342v1 Announce Type: cross  Abstract: "Abliterated" ("uncensored") builds of open-weight language models remove the activation direction that mediates refusal, and are popular partly on the assumption that a model that no longer refuses is also free of political censorship. We test this on two matched pairs of 2-bit models: NVIDIA's Nemotron 3.5 Bolt 74B-A9B and Qwen3.8-Flash-Next, each against a community abliteration. Each model judged 22 statement pairs (on the Republic of China (Taiwan), 1989 Tiananmen, control countries) in Traditional Chinese, Simplified Chinese (a character-level conversion) and English, and answered 27 open questions per script. Script mattered far more than abliteration: the original Bolt scored +0.58 on Index S (-1: People's Republic of China (PRC) position; +1: ROC position) in Traditional Chinese, +0.57 in English and +0.05 in Simplified Chinese, and affirmed "Taiwan is part of the PRC" with probability 0.91 in Simplified but 0.06 in Traditiona
    
[^69]: Transformer MLP中的读取器—写入器耦合

    Reader--Writer Coupling in Transformer MLPs

    [https://arxiv.org/abs/2610.29340](https://arxiv.org/abs/2610.29340)

    本文提出将Transformer MLP中隐藏通道的读取器与写入器视为耦合对进行联合测量，利用MLP更新雅可比矩阵与自身转置的一致性导出带符号的配对坐标，并据此发现每个模块仅由一小部分强耦合通道构成核心。

    

    Transformer的MLP为每个隐藏通道赋予两种角色：读取器决定该通道何时响应，写入器决定该通道向残差流中加回什么内容。若将此层视为键—值记忆来读取，则可以一次测量一个因子集：读取器与写入器之间的配对仅由它们共享的索引承载，任何仅从单一因子集计算出的量都无法看到这种配对。然而，训练通过联合更新将这两个因子耦合在一起，因此我们探究将二者一同读取能够揭示什么。该模块本身提供了联合测量的工具：MLP写回到它所读取的同一个流中，因此其更新的精确雅可比矩阵是方阵，它与自身转置的一致性可以分解为读取器—写入器项，其内积为每个通道（作为一个耦合对来读取）给出一个带符号的配对坐标。按此坐标排序，我们所测量的每个模块都拥有一个由强耦合通道组成的小核心；在初始化使两种符号等概率出现之处……（原文摘要在此处截断）

    arXiv:2610.29340v1 Announce Type: cross  Abstract: A transformer MLP gives each hidden channel two roles: a reader that decides when the channel responds, and a writer that decides what it adds back to the residual stream. Read as a key--value memory, the layer can be measured one factor set at a time: the pairing of readers to writers is carried only by their shared index, invisible to any quantity computed from one factor set alone. Yet training couples the two factors through their joint update, so we ask what reading them together shows. The block supplies the joint instrument: the MLP writes back into the very stream it reads, so the exact Jacobian of its update is square, and its agreement with its own transpose resolves into reader--writer terms whose inner products give each channel, read as a couple, a signed pairing coordinate. Ranked by this coordinate, each block we measure has a small core of strongly coupled channels; where initialization gives either sign equally, the co
    
[^70]: ChronoSynth：面向时序知识图谱的约束感知问题合成方法

    ChronoSynth: Constraint-Aware Question Synthesis over Temporal Knowledge Graphs

    [https://arxiv.org/abs/2610.29339](https://arxiv.org/abs/2610.29339)

    本文提出ChronoSynth框架，通过构建时序意图状态、检索适配带类型问题模式并进行验证修订，解决了时序知识图谱问题生成中的时序语义漂移问题，确保生成的问题保留决定答案集合的关键时间约束。

    

    知识图谱问题生成（KGQG）旨在根据结构化图证据和目标答案生成自然语言问题。当输入图谱是时序图谱时，一个有效的问题还必须保留决定答案集合的答案字段、时间条件、参考事件和时间顺序。现有的KGQG方法和提示基线可能生成流畅但改变了这些条件的问题，从而导致时序语义漂移。本文将时序KGQG研究为一个保持约束的图到问题转换问题，并提出了ChronoSynth——一个记忆增强且约束感知的生成框架。ChronoSynth首先构建时序意图状态，使预期的问题语义显式化；然后从索引化的模式记忆库中检索兼容的带类型问题模式，将其适配到输入图谱，并对生成的问题进行验证；当检测到可支持判定的不一致时，验证反馈会引导对问题的修订。

    arXiv:2610.29339v1 Announce Type: new  Abstract: Knowledge graph question generation (KGQG) generates natural-language questions from structured graph evidence and a target answer. When the input graph is temporal, a valid question must also preserve the answer field, time condition, reference event, and temporal order that determine the answer set. Existing KGQG methods and prompting baselines can produce fluent questions that alter these conditions, causing temporal semantic drift. This paper studies temporal KGQG as a constraint-preserving graph-to-question problem and proposes ChronoSynth, a memory-augmented and constraint-aware generation framework. ChronoSynth first constructs a Temporal Intent State to make the intended question semantics explicit. It then retrieves compatible typed question patterns from an indexed pattern memory, adapts them to the input graph, and validates the generated question. Validation feedback guides revisions when a supported inconsistency is detected
    
[^71]: Policy Central：检测已部署警务政策中的冲突、冗余与相似性

    Policy Central: Detecting Conflicts, Redundancy, and Similarity in Deployed Police Policy

    [https://arxiv.org/abs/2610.29338](https://arxiv.org/abs/2610.29338)

    论文部署了名为 Policy Central 的系统，通过比较“先检索后分类”与“先筛选后确认”两种架构来检测警察政策条款间的冲突、冗余与相似性，发现模型判断环节表现良好（冲突类召回率达 0.885），但在近两万条款中找到相关配对仍是主要瓶颈，端到端召回率仅约 0.46。

    

    Policy Central 是一个已部署的系统，帮助警察部门发现其书面政策各条款之间的关系：判断两个条款是相互冲突、冗余、仅相似还是不同。论文介绍了该系统并比较了两种架构：Retrieve-Judge 是一种“先检索后分类”的流水线，作为覆盖整个政策库的低成本健康信号运行；Screen-Confirm 是一种“先筛选后确认”的级联架构，对审阅者选定的文档进行选择性审计，并将其与其他文档进行详尽比对。在基于某警察部门约 84 份真实政策文档构建的 302 对基准测试中，对于给定的每一对文档，Screen-Confirm 达到 0.795 的总体召回率，在冲突类上达到 0.885，说明当前模型在判断环节表现良好。难点在于找到这些配对：Retrieve-Judge 的端到端召回率降至 0.457，Screen-Confirm 可恢复至 0.472（冲突类为 0.615）。目前这些配对是通过人工阅读整个政策库（近 20,000 个条款）并相互比对得出的。

    arXiv:2610.29338v1 Announce Type: new  Abstract: Policy Central is a deployed system that helps police departments find relationships between sections of their written policy: whether two sections conflict, are redundant, are merely similar, or are different. We describe the system and compare two architectures. Retrieve-Judge, a retrieve-then-classify pipeline, runs as a cheap library-wide health signal; Screen-Confirm, a screen-then-confirm cascade, is a selective audit of the documents a reviewer chooses, checking them exhaustively against one another. On a 302-pair benchmark drawn from one department's ~84 real policy documents, given each pair Screen-Confirm reaches 0.795 aggregate recall and 0.885 on Conflict, so current models judge well. Finding the pair is the hard part: Retrieve-Judge's composed recall falls to 0.457, and Screen-Confirm recovers 0.472 (0.615 on Conflict). Today these pairs are found by reading the whole library, nearly 20,000 sections, checked against one ano
    
[^72]: 见树亦见林：从个体对话中涌现的共识

    Seeing the Forest for the Trees: Emergent Agreement from Individual Conversations

    [https://arxiv.org/abs/2610.29328](https://arxiv.org/abs/2610.29328)

    本研究通过对个体对话共识内部表征进行跨对话聚合，发现尽管个体对话层面无法可靠预测共识，但聚合后的共识可用简单的马尔可夫过程很好地预测，实现了“由树见林”的涌现性洞察。

    

    当LLM智能体在真实环境中相互交互并影响彼此的决策时，理解它们如何趋向共识变得非常重要。我们在多轮对话中研究这一问题：两个LLM讨论主观性话题，可以达成一致或产生分歧，也可能改变自己的立场。通过对个体对话的激活使用线性分类器，我们识别出了共识的内部表征。我们发现该表征在不同轮次和不同对话之间变化显著，并且在个体对话层面无法可靠地追踪外部共识。因此，我们超越了个体对话中共识表征的研究，转而将共识表征跨对话进行聚合。在外部层面，这种聚合视角揭示了一个简单的结构：当跨对话汇总时，共识可以通过马尔可夫过程得到很好的预测，即每一轮的共识是上一轮共识状态的直接函数。

    arXiv:2610.29328v1 Announce Type: new  Abstract: As LLM agents interact in the wild and shape each other's decisions, understanding how they converge toward agreement becomes important. We study this in multi-turn conversations where two LLMs discuss subjective matters, can agree or disagree, and may change their stances. Using a linear classifier on individual conversation activations, we identify an internal representation of agreement. We find that it varies substantially across turns and conversations and does not reliably track external agreement at the individual conversation level. We therefore move beyond the representation of agreement in individual conversations, by aggregating the agreement representation across conversations. Externally, this aggregate view reveals a simple structure: when pooled across conversations, agreement can be predicted well with a Markov process, with agreement at each turn being a direct function of agreement at the previous turn. Similarly, when 
    
[^73]: OACap：面向详细视频描述的描述引导式主动观察

    OACap: Caption-Guided Active Observation for Detailed Video Captioning

    [https://arxiv.org/abs/2610.29327](https://arxiv.org/abs/2610.29327)

    提出了以描述为中心的智能体框架 OACap，它能将视频描述中缺失或缺乏支持的信息转化为主动观察请求，借助时空工具获取证据并迭代验证修订描述，从而显著提升详细视频描述质量，同时引入 FSCap-Bench 基准及 CSFS 等评分指标来系统评估描述的事实性与语义保真度。

    

    与问答任务不同（问答中的查询已明确了感知目标），开放式详细视频描述必须自主决定需要获取什么证据，以及如何将这些证据整合成连贯的全局描述。我们提出了观察自适应视频描述器，这是一个以描述为中心的智能体，它将描述中缺乏支持或缺失的信息转化为有针对性的观察请求，利用时空工具获取证据，并迭代地验证和修订描述内容。在使用固定的纯文本阅读器的情况下，OACap 生成的描述在多个通用视频理解基准上均取得了一致的性能提升。为了更好地诊断这些描述所保留的信息，我们引入了事实-语义描述基准（FSCap-Bench）。其中的核心支撑事实性得分（CSFS）将断言精确度与重要性感知的事实覆盖率相结合，在 93.0-98.3% 的有效描述对上对四种受控扰动类型做出了符合预期的响应。其跨模态语义往返得分（摘要原文在此处截断）

    arXiv:2610.29327v1 Announce Type: cross  Abstract: Unlike question answering, where a query specifies a perceptual target, open-ended detailed captioning must determine what evidence to acquire and how to integrate it into a coherent global description. We propose Observation-Adaptive Video Captioner (OACap), a caption-centered agent that turns unsupported or missing information into targeted observation requests, uses spatiotemporal tools to acquire evidence, and iteratively verifies and revises the caption. OACap captions consistently improve performance across general video-understanding benchmarks using a fixed text-only reader. To better diagnose the information they preserve, we introduce Factual-Semantic Captioning Bench (FSCap-Bench). Its Core-Support Factuality Score (CSFS) combines claim precision with importance-aware fact coverage and responds as expected to four controlled perturbation types on 93.0-98.3% of valid caption pairs. Its Cross-Modal Semantic Round-Trip Score (C
    
[^74]: 信号与查询：高效利用Token的智能体上下文发现

    Signals and Queries: Token-Efficient Agentic Context Discovery

    [https://arxiv.org/abs/2610.29326](https://arxiv.org/abs/2610.29326)

    该论文提出S&Q方法，利用LLM离线生成激活样例与困难负样本训练独立逻辑回归分类器，在无需每轮LLM调用的情况下高效剔除至少30%的无关候选策略，在保持高召回率的同时显著降低了智能体策略相关性评估的Token成本。

    

    面向用户的LLM智能体必须确定每一轮对话中适用的行为策略，但用LLM评估每个策略的成本很高。我们提出了信号与查询（Signals and Queries，S&Q），这是一个为Parlant 4.0开发的轻量级第一阶段相关性过滤器，能在下游评估之前剔除至少30%的候选策略，同时保持高召回率。对于每个策略，由LLM离线生成激活样例和困难负样本，然后基于这些样本的嵌入训练独立的逻辑回归分类器，利用嵌入和向量点积预测相关性，无需每轮调用LLM。这将昂贵的相关性评估转移到了更小的候选集上，同时允许每个策略的分类器独立更新。基于涵盖254个τ²-bench对话前缀和36个航空策略的9,144个LLM评判相关性标签进行评估，S&Q在所有评估的召回率水平上都比top-K和余弦阈值检索方法取得更高的精确率。

    arXiv:2610.29326v1 Announce Type: new  Abstract: User-facing LLM agents must identify which behavioral policies apply at each turn, but evaluating every policy with an LLM is costly. We introduce Signals and Queries (S&Q), a lightweight first-stage relevance filter developed for Parlant 4.0 that removes at least 30% of candidate policies before downstream evaluation while preserving high recall. For each policy, an LLM generates activating examples and hard negatives offline. An independent logistic classifier trained on their embeddings then predicts relevance using embeddings and vector dot products, with no per-turn LLM calls. This shifts expensive relevance evaluation to a smaller candidate set, while allowing each policy's classifier to be updated independently. Evaluated against 9,144 LLM-judge relevance labels covering 254 $\tau^2$-bench conversation prefixes and 36 airline policies, S&Q achieves higher precision than top-K and cosine-threshold retrieval at every evaluated recal
    
[^75]: OSynth：面向会议理解的纵向、上下文敏感评估

    OSynth: A Longitudinal, Context-Sensitive Evaluation of Meeting Understanding

    [https://arxiv.org/abs/2610.29319](https://arxiv.org/abs/2610.29319)

    OSynth是一个包含570小时模拟组织会议音频的大规模合成数据集，其真实感使人类难以与真实会议区分，并证明额外的上下文信息能显著提升大语言模型在实体敏感会议理解任务上的表现。

    

    会议记录工具正越来越多地部署在具有纵向上下文信息的环境中，这些信息包括日历元数据、以往会议、电子邮件、幻灯片和说话人注册信息；然而，现有会议语料库在规模、领域覆盖范围和可用上下文方面仍然有限。我们提出了OSynth，这是一个大规模的模拟组织合成数据集，包含570小时的会议音频以及相关产物和潜在的真实状态。在成对比较实验中，人类评估者基于150-300词的摘录，仅能以56%的准确率区分OSynth会议与真实会议记录。在多个最先进的大语言模型上，额外的上下文信息在多个实体敏感维度上带来了显著提升，这展示了现有方法的局限性，以及OSynth在开发更好上下文化的会议理解系统方面的潜在价值。

    arXiv:2610.29319v1 Announce Type: new  Abstract: Meeting notetakers are increasingly deployed in settings where longitudinal context is available including calendar metadata, prior meetings, emails, slides, and speaker enrollments; however, meeting corpora remain limited in terms of size, domain coverage, and available context. We present OSynth, a large-scale synthetic dataset of simulated organizations comprising 570 hours of meeting audio together with associated artifacts and a latent ground-truth state. In a pairwise comparison, human raters distinguish OSynth meetings from real meeting transcripts with only 56% accuracy based on excerpts of 150-300 words. Across state-of-the-art LLMs, additional context yields significant gains on several entity-sensitive dimensions, demonstrating limitations of contemporary approaches and the potential usefulness of OSynth in developing better-contextualized meeting-understanding systems.
    
[^76]: 询问“导演看到了什么”可以改变视觉语言模型的选择

    Asking What a Director Sees Can Change What Vision-Language Models Choose

    [https://arxiv.org/abs/2610.29318](https://arxiv.org/abs/2610.29318)

    研究发现在导演任务中，先询问视觉语言模型“导演能看到什么”会显著改变其后续行动选择（选择导演看不到的方块的比例提高12至88个百分点），这挑战了“要求报告视角不会影响行动”的传统测试假设。

    

    视觉语言模型如今常被用于心智理论能力的测试：即它们能否推断出另一个人看到什么、知道什么、意指什么。一种常见的测试设计是要求模型先报告该视角，然后为该人采取行动，并将“正确的报告伴随错误的行动”解读为模型未能运用该视角。这种解读建立在一个未曾言明的假设之上：即要求报告不会改变模型的行动。我们证明事实并非如此。在一个渲染实现的导演任务中，一块隔板挡住了导演的视线，使其看不到最大的方块，而导演要求“拿最大的方块”。如果先询问模型“导演能看到哪些方块”，会使六个GPT模型选择这个被遮挡方块的比例提高12到88个百分点，Qwen3.8-27B模型提高17个百分点——即使这些模型大多能正确报告导演能看到哪些方块。推理轨迹和提示词改写实验揭示了原因：这个问题改变了模型对请求的理解方式，使其从导演的视角转向了模型自身的……

    arXiv:2610.29318v1 Announce Type: new  Abstract: Vision-language models are now routinely tested on theory of mind: whether they can work out what another person sees, knows, and means. A common design asks the model to report that perspective and then to act for the person, and reads a correct report beside a wrong action as a failure to use the perspective. That reading rests on a quiet assumption: that asking for the report leaves the action unchanged. We show that it does not. In a rendered Director Task, a partition hides the largest block from a director who asks for "the largest block". Asking first which blocks the director can see raises the share of replies that choose this hidden block by 12 to 88 percentage points on six GPT models and by 17 on Qwen3.8-27B, even though the models mostly report correctly which blocks the director sees. Reasoning traces and prompt rewordings suggest why: the question shifts how the request is read, from the director's view toward the model's 
    
[^77]: 基于大语言模型的外生世界建模用于动态调度

    LLM-Based Exogenous World Modeling for Dynamic Scheduling

    [https://arxiv.org/abs/2610.29308](https://arxiv.org/abs/2610.29308)

    该论文提出将预训练大语言模型作为结构化世界模型的外生组件，通过后状态分解与紧凑信念表示预测未来作业到达，并结合信念条件化调度策略、意外触发刷新机制和在线残差自校准，实现了高效的动态调度规划。

    

    动态调度需要在处理在线到达任务的同时预判未来作业，因为在非线性拖期目标下，糟糕的决策会使下游成本不断累积。我们证明，预训练的大语言模型（LLM）可以作为结构化世界模型的外生组件，用于预测不确定的未来作业到达。具体而言，我们利用一种后状态分解方法，将已知的确定性车间状态转移与不确定的外生作业到达分离开来，使LLM仅需通过紧凑的信念表示来预测后者。在此基础上，我们提出了后状态世界模型规划（AWMP）方法，该方法将演化得到的、以信念为条件的调度策略与基于场景的短时前瞻推演相结合。一种由“意外”触发的刷新机制仅在新观测到的作业到达与先前预测产生足够大的偏差时才查询LLM，而在线残差自校准机制则进一步让LLM以其最近的预测残差为条件进行修正。

    arXiv:2610.29308v1 Announce Type: cross  Abstract: Dynamic scheduling requires dispatching online arrivals while anticipating future jobs, as poor decisions can compound downstream costs under nonlinear tardiness objectives. We show that a pretrained large language model (LLM) can serve as the exogenous component of a structured world model, forecasting uncertain future job arrivals. Specifically, we exploit an afterstate decomposition, which separates known deterministic shop transitions from uncertain exogenous job arrivals, enabling the LLM to forecast only the latter through a compact belief representation. Building on this, we develop Afterstate World-Model Planning (AWMP), which combines an evolved, belief-conditioned dispatching strategy with short scenario-based rollouts. A surprise-triggered refresh mechanism queries the LLM only when newly observed arrivals deviate sufficiently from previous forecast, and online residual self-calibration further conditions the LLM on its rece
    
[^78]: ExpEvolve：面向开放式发现的经验结构化与复用

    ExpEvolve: Structuring and Reusing Experience for Open-Ended Discovery

    [https://arxiv.org/abs/2610.29303](https://arxiv.org/abs/2610.29303)

    提出ExpEvolve框架，将思想层级的提取与控制注入进化式编码代理，通过发现能够解释程序间评分差异的高层级思想因素并复用经验，提升开放式科学发现的样本效率。

    

    诸如AlphaEvolve这样的进化编码代理是利用大语言模型（LLM）构建AI科学家的显著成功案例之一。这些代理通过迭代改进和演化程序来解决开放式科学问题，利用LLM的先验知识和推理能力。尽管取得了成功，但对程序的变异大多是基于表面启发式的低层级操作，缺乏持久的控制单元。相比之下，科学是通过思想实现复利式积累的。在这项工作中，我们提出了ExpEvolve，它将思想层级的提取与控制注入进化过程：它从程序中发现并利用能够解释不同程序之间评分差异的高层级因素。我们刻画了利用思想因素进行发现何时能提升样本效率的理论条件。我们还构建了一个实用框架，以展示提取出的高层级思想因素有助于指导基于经验的利用。

    arXiv:2610.29303v1 Announce Type: cross  Abstract: Evolutionary coding agents such as AlphaEvolve are one of the notable successes in using Large Language Models (LLMs) to build AI Scientists. These agents tackle open-ended scientific problems by iteratively improving and evolving programs, leveraging the prior knowledge and reasoning capabilities of LLMs. Despite their success, mutations over programs are mostly low-level upon surface heuristics, without a persistent unit of control. In contrast, science compounds through ideas. In this work, we propose ExpEvolve, which injects idea-level extraction and control into evolution: it discovers and leverages the high-level factors from programs that explain scoring differences among different programs. We characterize theoretical conditions on when discovery with idea factors improves sample efficiency. We also build a practical framework to show that the extracted high-level idea factors are helpful for guiding exploitation from experienc
    
[^79]: 相同概率，不同建议：高风险LLM决策支持中的收益-损失框架效应

    Same Probabilities, Different Advice: Gain-Loss Framing Effects in High-Stakes LLM Decision Support

    [https://arxiv.org/abs/2610.29294](https://arxiv.org/abs/2610.29294)

    该研究提出FrameBench-HPS基准，揭示了大语言模型在提供高风险决策建议时存在显著的框架效应：即使概率信息在数学上完全等价，收益与损失两种表述方式仍会在32.4%的情况下改变模型的推荐类别。

    

    大语言模型（LLM）越来越多地被用于支持重大决策，但它们在等价的不确定性表述下的可靠性仍不清楚。现有研究已经记录了提示敏感性、认知偏差和置信度校准失准等问题，但尚不清楚当同一概率通过互补的收益和损失框架表达时，高风险建议是否会发生变化。我们提出了FrameBench-HPS，这是一个受控基准，包含由医疗健康、金融、教育/职业以及法律/消费者权益四个领域的240个高风险场景生成的2,880个提示。在七个大语言模型上，收益-损失框架在32.4%的完整组中改变了推荐类别，其中大多数变化涉及不确定性而非直接的推荐反转。人类评估证实了模型在感知风险和指示性程度上的相应转变。鲁棒性实验表明，框架效应在对话式改写下依然存在。

    arXiv:2610.29294v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to support consequential decisions, yet their reliability under equivalent representations of uncertainty remains unclear. Existing work has documented prompt sensitivity, cognitive biases, and confidence miscalibration, but it is still unknown whether high-stakes advice changes when the same probability is expressed through complementary gain and loss frames. We introduce FrameBench-HPS, a controlled benchmark of 2,880 prompts derived from 240 high-stakes scenarios across healthcare, finance, education/career, and legal/consumer-rights domains. Across seven LLMs, gain-loss framing changes the recommendation category in 32.4% of complete groups, with most changes involving uncertainty rather than direct reversals. Human evaluation confirms corresponding shifts in perceived risk and directiveness. Robustness experiments show that framing effects persist under conversational paraphrases 
    
[^80]: 功归其人：面向多智能体系统的对比归因拓扑搜索

    Credit Where Credit Is Due: Contrastive Attribution Topology Search for Multi-Agent Systems

    [https://arxiv.org/abs/2610.29292](https://arxiv.org/abs/2610.29292)

    CATS方法让LLM归因器通过对比不同候选拓扑的执行轨迹为各智能体精准分配功劳，从而在智能体超图中自动搜索并剪枝出更小更优的多智能体团队，在提升准确率的同时显著降低部署成本。

    

    LLM多智能体系统（MAS）的好坏取决于它选择信任的智能体：其准确性和成本取决于其拓扑结构，即包含哪些智能体以及它们之间如何通信。然而，单一的任务分数无法揭示哪些智能体在帮助或阻碍性能。我们提出了对比归因拓扑搜索（CATS），它在固定的专业化智能体超图内搜索紧凑的团队。一个LLM归因器对比候选拓扑之间的执行轨迹，并为各个智能体分配功劳。CATS利用这些功劳更新超图的边，同时逐步剪除弱边和弱智能体，最终产生一个用于部署的小型团队。在使用小型开源权重模型和匹配的候选评估条件下，CATS在BigCodeBench上比仅基于分数的启发式搜索平均准确率提升了5.3个百分点，同时部署token使用量减少了54%。在MBPP上，CATS以更少的token达到了与最强基线相当的性能。

    arXiv:2610.29292v1 Announce Type: new  Abstract: An LLM multi-agent system (MAS) is only as good as the agents it chooses to trust: its accuracy and cost hinge on its topology, meaning which agents are included and how they communicate. A single task score, however, does not reveal which agents help or hinder performance. We propose Contrastive Attribution Topology Search (CATS), which searches for compact teams within a fixed supergraph of specialized agents. An LLM attributer compares execution traces across candidate topologies and assigns credit to individual agents. CATS uses these credits to update supergraph edges while progressively pruning weak edges and agents, yielding a small team for deployment. With small open-weight models and matched candidate evaluations, CATS improves mean accuracy on BigCodeBench by 5.3 percentage points over score-only heuristic search while using 54\% fewer deployment tokens. On MBPP, CATS matches the strongest baselines with fewer tokens than the 
    
[^81]: 基于计算机视觉与知识增强语言模型的智能养蜂辅助系统

    Computer Vision and Knowledge-Grounded Language Models for Intelligent Beekeeper Assistance

    [https://arxiv.org/abs/2610.29283](https://arxiv.org/abs/2610.29283)

    该论文提出了一种边缘可部署的智能系统框架，结合计算机视觉目标检测与知识增强语言模型，将蜂巢巢框照片自动转化为可解释、面向行动的蜂群检查报告，从而辅助养蜂人高效评估蜂群状况。

    

    人工饲养和野生蜜蜂维持着全球约三分之一的食品生产，然而蜂群正面临气候变化带来的日益加剧的压力，包括极端天气模式以及寄生虫和疾病模式的改变。及时评估蜂群状况——例如了解蜂群强度、育虫面积、蜂蜜和花粉储存以及巢脾使用情况——对于适应性管理至关重要。然而，人工检查巢框速度慢、成本高，且检查质量依赖于稀缺的专业知识。我们提出了一个聚焦于边缘可部署流水线的人工智能系统框架，可将蜂巢巢框照片转化为可解释、面向行动的检查报告。该流水线包括一个目标检测器，用于检测具有生物学意义的巢房类别；检测结果被提炼为结构化事实模式；事实被转换为受约束的事实性文本；文本随后触发一个基于知识库的接地系统，用于生成解释和建议。

    arXiv:2610.29283v1 Announce Type: cross  Abstract: Managed and wild bees sustain roughly a third of the global food products, yet colonies face mounting stress from climate-change including extreme weather patterns, altered parasite and disease patterns. Timely assessment of colony condition such as understanding colony strength, brood area, honey and pollen storage, comb usage, is essential to adaptive management. However, manual hive frame inspection is slow, costly, and quality depends on scarce expertise. We present an artificial targeted intelligent system framework focusing on edge deployable pipeline that turns hive frame photographs into interpretable and action-oriented inspection report. The pipeline includes an object detector to detect biologically-meaningful comb-cell classes; detections are distilled into structured fact schema; facts are converted into constrained factual text; the text triggers a grounded knowledge-base for interpretations and recommendations. Knowledge
    
[^82]: 听而未言：提示方式低估了冻结大型音频-语言模型中音乐时刻的时间信息

    Heard But Not Said: Prompting Understates Temporal Information of Musical Moments in Frozen Large Audio-Language Models

    [https://arxiv.org/abs/2610.29256](https://arxiv.org/abs/2610.29256)

    该论文提出基于 Slakh2100 分轨激活的成对状态/变化查询探针，发现冻结的大型音频-语言模型实际上具备音乐时刻的时间信息，但整曲提示会低估这一能力——状态查询无需上下文即可高于基线，而变化查询则依赖前置音频才能提升。

    

    大型音频-语言模型（LALMs）能够理解音乐，却难以在时间轴上定位被描述的某一时刻。我们探究这究竟源于时间信息的缺失，还是时间段提取方式上的瓶颈。我们基于 Slakh2100 分轨激活构建了一种成对查询探针，将状态查询（例如“只有钢琴和贝斯正在演奏”）和变化查询（例如“吉他退出，只剩下钢琴和贝斯”）附加到同一段音频区间上，并改变呈现给模型的前置音频时长。在多个冻结的 LALM 上，三种区间提取方式各自回答不同的问题：整曲提示询问模型能否说出“何时”；窗口化的是/否评分询问模型在被给予片段时能否验证该时刻，以及是否需要此前的内容；注意力方法则询问模型能否自行找到该时刻。状态查询在无上下文时得分即高于基线，而变化查询在大多数短片段模……（原文摘要在此截断）

    arXiv:2610.29256v1 Announce Type: cross  Abstract: Large audio-language models (LALMs) understand music, yet struggle to locate a described moment in time. We ask whether this reflects missing temporal information or a bottleneck in how the time span is extracted. Using a paired-query probe built from Slakh2100 stem activations, we attach a state query (e.g., ``only piano and bass are playing'') and a change query (e.g., ``the guitar drops out, leaving piano and bass'') to the same span and vary how much preceding audio the model is shown. Across frozen LALMs, three ways of extracting the span ask different questions: full-track prompting asking whether the model can say when; windowed yes/no scoring asking whether it verifies the moment when handed the segment, and whether it needs what came before; attention asking whether it can find the moment on its own. State queries score above baseline without context, while change queries improve with preceding context in most short-segment mo
    
[^83]: 层次化掩码扩散语言模型

    Hierarchical Masked Diffusion Language Models

    [https://arxiv.org/abs/2610.29255](https://arxiv.org/abs/2610.29255)

    提出了层次化掩码扩散语言模型（HMDLM），通过为每个token位置附加层次级别信息来建模位置间的生成优先级，使掩码扩散语言模型无需额外标注即可按照自然的层次结构有序地并行生成文本。

    

    掩码扩散语言模型已成为自回归模型的一种替代方案，能够实现并行token生成。然而，标准的掩码扩散语言模型在各个位置使用共享的噪声调度，没有显式编码哪些位置应该优先被解析。许多任务在位置之间天然存在优先级关系：文档标题和章节标题可以指导正文内容的生成，数独中由初始线索直接确定的单元格可以帮助解析更困难的单元格。为了在token位置之间引入这种优先级（或层次）的概念，我们提出了层次化掩码扩散语言模型（HMDLM），它为每个token位置增加一个层次级别作为训练数据的一部分，并且不需要或仅需极少量的额外标注。它将生成过程分解为两步：第一步预测掩码位置上的层次分布，第二步预测……

    arXiv:2610.29255v1 Announce Type: new  Abstract: Masked diffusion language models have emerged as an alternative to autoregressive models, enabling parallel token generation. However, standard masked diffusion language models use a shared noise schedule across positions without explicitly encoding which positions should be resolved first. Many tasks exhibit a natural priority among positions: document titles and section headings can guide the generation of body text, while Sudoku cells determined directly from the initial clues can help resolve more difficult cells. To incorporate this notion of priority (or hierarchy) among token positions, we introduce the Hierarchical Masked Diffusion Language Model (HMDLM), which augments each token position with a hierarchy level available as part of the training data, and requires zero or minimal additional annotations. It factors generation into two passes: the first predicts a hierarchy distribution over masked positions, the second predicts to
    
[^84]: 群体间偏见预测 r/AmITheAsshole Reddit 社区中社会评判与共情的表达

    Intergroup Bias Predicts Expressions of Social Judgment and Empathy in the r/AmITheAsshole Reddit Community

    [https://arxiv.org/abs/2610.29254](https://arxiv.org/abs/2610.29254)

    本研究基于 r/AmItheAsshole 社区超过12万条评论的大规模分析，首次在自然在线社交场景中揭示了评论者与发帖人之间的群体身份（性别、年龄）相似性会显著影响其社会评判与共情的表达，证实了群体间共情差距在真实网络互动中确实存在。

    

    人们基于自身社会身份，会以不同的方式分享和理解他人的经历；通常，当对方属于自己所在的内群体而非外群体时，人们会产生更多的共情。尽管这种群体间共情差距在社会心理学中已被广泛研究，但它如何在自然发生的在线社交互动中显现仍不清楚。为此，我们测量了评论者如何表达社会评判（作为其对发帖人理解的间接结果）以及文本中的共情，其依据是评论者与发帖人是否性别相同或年龄相近。我们聚焦于发帖人自我报告这些人口统计特征的帖子，分析了 r/AmItheAsshole 社区中与 36,163 条帖子相关的 126,716 条评论（来自 18,111 名评论者），同时控制了评论者与发帖人的群体身份、帖子特征（主题、长度和得分）以及评论者层面和帖子层面的方差。我们的结果表明，当评论者与发帖人……（原文摘要在此处截断）

    arXiv:2610.29254v1 Announce Type: new  Abstract: People share and understand others' experiences differently based on their social identities; often, people empathize more when the other person belongs to their ingroup compared to an outgroup. While this intergroup empathy gap is well-studied in social psychology, it remains unclear how it manifests in naturally occurring online social interactions. To this end, we measure how commenters express social judgments, as an indirect outcome of their understanding towards the posters, and empathy in text, based on whether commenters share the poster's gender or are of similar ages. We focus on posts where the poster self-reports these demographics, and analyze 126,716 comments associated with 36,163 posts from 18,111 commenters in r/AmItheAsshole, while controlling for commenter and poster group identities, post features (topic, length, and score), and both commenter- and post-level variance. Our results show that when commenters and posters
    
[^85]: 让智能体自行设计搜索方式：面向智能体搜索的自动界面优化

    Let Agents Design How They Search: Automatic Interface Optimization for Agentic Search

    [https://arxiv.org/abs/2610.29250](https://arxiv.org/abs/2610.29250)

    提出 SearchWeaver 框架，利用执行轨迹和任务奖励自动联合优化语料库产物、可执行工具和使用说明，使搜索界面随具体语料库自适应优化，显著提升答案准确率与检索排序效果。

    

    智能体搜索依赖于语言模型智能体访问和探索语料库所使用的界面。现有的界面设计均由人工固定，并无差别地应用于每一个语料库，然而有效的设计可能取决于具体的语料库。我们提出了 SearchWeaver，一个以下游搜索性能为指导的自动搜索界面优化框架。提议者利用执行轨迹和任务奖励来联合修订语料库产物、可执行工具和使用说明。这一过程优化了预先准备哪些信息、工具执行哪些操作，以及哪些决策保留给搜索智能体。在使用 GPT-5.4-nano 时，与直接语料库交互相比，优化后的界面在 BrowseComp-Plus 上将答案准确率从 71.6% 提升至 78.4%，并在四个 BRIGHT 子集上将平均 nDCG@10 从 45.7 提升至 55.0。这些发现凸显了联合优化语料库产物、可执行工具与使用说明的价值。

    arXiv:2610.29250v1 Announce Type: new  Abstract: Agentic search depends on the interface through which a language-model agent accesses and explores a corpus. Existing interface designs are fixed by hand and applied unchanged to every corpus, although effective designs may depend on the corpus. We introduce SearchWeaver, a framework for automatic search interface optimization guided by downstream search performance. A proposer uses execution trajectories and task rewards to jointly revise corpus artifacts, executable tools, and usage instructions. This pro cess optimizes what information is prepared in advance, which operations tools perform, and which decisions remain with the search agent. With GPT-5.4-nano, the optimized interfaces improve answer accuracy from 71.6% to 78.4% on BrowseComp-Plus and average nDCG@10 from 45.7 to 55.0 across four BRIGHT subsets over Direct Corpus Interaction. These findings highlight the value of jointly optimizing corpus artifacts, executable tools, and
    
[^86]: MOSAIC：面向对话智能体长期记忆与跨模态推理的全模态基准

    MOSAIC: An Omni-Modal Benchmark for Long-Term Memory and Cross-Modal Reasoning in Conversational Agents

    [https://arxiv.org/abs/2610.29249](https://arxiv.org/abs/2610.29249)

    MOSAIC是一个包含923个问题、覆盖文本、图像、视频和音频的全模态长期记忆基准，通过冻结读取环节、干预表示、检索和传递的匹配诊断协议，能够精确定位对话智能体在哪个环节丢失了跨模态证据。

    

    全模态助手不仅需要记住用户说了什么，还要记住用户展示了什么、传达了什么。现有的长期记忆基准大多仅涉及文本与图像，或只报告单一的端到端分数，因此无法定位证据是在表示、检索、传递还是读取环节丢失的。我们提出了MOSAIC，这是一个包含923个问题的基准，问题基于跨越11至24个会话、涵盖文本、图像、视频和音频的角色化历史记录构建。MOSAIC构建的是具有针对性的空间、时间和韵律信号，而非装饰性的媒体，其中空间信息获得了最强的因果验证，标注的韵律获得了选择器级别的验证，并配以匹配的诊断协议——该协议在冻结问题、读取器、提示和预算的同时，对表示、检索和传递环节进行干预。匹配分解结果显示，表示损失依赖于具体模态，即使提供金标准证据，残余的读取损失仍占主导地位，且摘要内容在此处被截断。

    arXiv:2610.29249v1 Announce Type: new  Abstract: Omni-modal assistants must remember not only what users say, but also what they show and convey. Existing long-term memory benchmarks are mostly text-and-image or report a single end-to-end score, so they cannot locate whether evidence is lost when it is represented, retrieved, delivered, or read. We introduce MOSAIC, a benchmark of 923 questions over persona-grounded histories of 11-24 sessions spanning text, image, video, and audio. MOSAIC constructs targeted spatial, temporal, and prosodic signals rather than decorative media, with the strongest causal validation for spatial information and selector-level validation for annotated prosody, and pairs them with a matched diagnostic protocol that freezes the question, reader, prompt, and budget while intervening on representation, retrieval, and delivery. Matched decomposition shows that representation loss is modality-dependent, residual reading dominates even under gold evidence, and ab
    
[^87]: 超越二元标签：基于提示级评分细则的大语言模型安全性诊断评估

    Beyond Binary Labels: Diagnostic Evaluation of LLM Safety with Prompt-Level Rubrics

    [https://arxiv.org/abs/2610.29225](https://arxiv.org/abs/2610.29225)

    提出DiagSafe诊断评估框架，以基于全球AI安全政策的提示级评分细则取代传统二元安全标签，从安全性与有用性两个独立维度对大语言模型进行更细致、更具解释性的双语安全评估。

    

    当前的大语言模型（LLM）安全基准主要依赖“安全”或“不安全”的二元标签，缺乏解释深度，且不同自动化评判器之间的一致性较低。为解决这一问题，我们提出了DiagSafe，一个超越二元分类、采用提示级评分细则的诊断评估框架。该框架以全球适用的AI安全政策为基础，每个评分细则沿两个维度提供明确的评判标准：安全性维度针对具体危害，并设置特定豁免条款以抑制误报；有用性维度则独立评估指令的完成程度。为确保评估的稳健性和现实性，我们构建了一个基于全面84类安全分类体系的双语（英文-中文）基准，包含两个子集：DiagBase，涵盖单轮和多轮提示中的直接有害查询以及基于模板的越狱攻击；DiagAttack，将有害意图嵌入复杂指令之中（原文摘要在此处截断）。

    arXiv:2610.29225v1 Announce Type: new  Abstract: Current LLM safety benchmarks predominantly rely on binary safe or unsafe labels, lacking explanatory depth and suffering from low consistency across automated judges. To address this, we introduce DiagSafe, a diagnostic evaluation framework that moves beyond binary classification by employing prompt-level rubrics. Grounded in globally applicable AI safety policies, each rubric provides explicit criteria along two dimensions: safety, which targets concrete harms with specific exemptions to suppress false positives, and usefulness, which independently evaluates instruction fulfillment. To ensure a robust and realistic evaluation, we construct a bilingual (English-Chinese) benchmark grounded in a comprehensive 84-label safety taxonomy, consisting of two subsets: DiagBase, which covers single- and multi-turn prompts with direct harmful queries and template-based jailbreaks, and DiagAttack, which embeds harmful intents into complex instructi
    
[^88]: C²U在检索增强生成中的应用：检索增强生成中的组合式概念遗忘

    C$^2$U in RAG: Compositional Concept Unlearning in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.29223](https://arxiv.org/abs/2610.29223)

    本文提出组合式概念遗忘方法C²U，通过将有界查询增量与增强原子库路由相结合，在检索增强生成中有效抑制目标概念暴露，使目标Hit@5降低4.25个百分点且不增加控制查询暴露。

    

    检索增强生成（RAG）即使在对生成器进行审核控制的情况下，仍可能通过检索到的证据暴露不想要的概念。我们研究了在固定编码器和文档索引条件下的查询侧概念抑制问题。组合式概念遗忘（C²U）方法将通过配置文件导出的方向路由到增强的原子库中，仅在原始查询上添加一个有界的“编辑后减去原始”增量。在五个数据集、四个编码器和三个随机种子的实验中，经验证选择的最优配置将汇总的二分类目标Hit@5降低了4.25个百分点，同时控制组退化非正（即无负面影响），目标暴露变化也非正。在所评估的查询变换下，该方法在抑制目标概念密度的同时，不会增加预算内的控制查询暴露。此外，在经过独立验证门控和首次使用测试评估后，ArXiv四概念组合任务上的表现优于基于平均方向的对比方法。

    arXiv:2610.29223v1 Announce Type: new  Abstract: Retrieval-augmented generation can expose unwanted concepts through retrieved evidence even when the generator is moderated. We study query-side concept suppression with a fixed encoder and document index. Compositional Concept Unlearning (C$^2$U) routes a profile-derived direction through an augmented atom bank, adding only a bounded edited-minus-original delta to the raw query. Across five datasets, four encoders, and three seeds, the validation-selected configuration reduces pooled binary target Hit@5 by 4.25 percentage points with non-positive control degradation and target-exposure changes. Under the evaluated query transforms, it suppresses target density without increasing budgeted control-query exposure. Separately, ArXiv four-concept composition improves over an averaged-direction comparator after independent validation gates and first-use test evaluation.
    
[^89]: 偏好不等于许可：代理式商务中个性化自动购买边界的确定

    Preference Is Not Permission: Personalizing the Auto-Purchase Line in Agentic Commerce

    [https://arxiv.org/abs/2610.29216](https://arxiv.org/abs/2610.29216)

    该论文提出将学习个人化的“自动购买边界”作为代理式商务中的新任务，并发现十六个指令微调模型普遍“越权”——即使商品匹配度低、购买依据缺失也会自动下单，与人类委托者的授权意愿存在系统性错位，且这一边界因人而异、需要个性化建模。

    

    个性化购物基准关注的是哪件商品适合用户；我们则提出另一个问题：智能体可以在该商品上被允许承诺到什么程度，并将学习个人的“自动购买线”作为一项任务提出。这条边界以证据为条件且具有三值性：是否做出购买承诺是否恰当，取决于商品信息的充分性（购买依据 warrant）、可逆性、价格是否符合用户的规范以及紧迫性，而不仅仅取决于匹配程度；其自然输出为“不购买”、“先确认”或“自动购买”三个层级。在十六个指令微调模型上的实验表明，模型的自主承诺行为与人类授权参考存在系统性错位——在该参考中，十五名标注者扮演给定用户画像中的委托方，选择他们在无需确认的情况下所允许的最远行动：即便匹配度很低，即便模型自身的证据标志显示购买依据缺失，模型仍会做出购买承诺。这是一种越权行为。这条边界因人而异：这些评分者之间存在分歧，但分歧是结构化的。

    arXiv:2610.29216v1 Announce Type: cross  Abstract: Personalized shopping benchmarks ask which item fits the user; we ask how far an agent may commit on that item, and pose learning the individual's auto-purchase line as a task. The boundary is evidence-conditioned and three-valued: whether committing is appropriate depends on the sufficiency of product information (warrant), reversibility, price against the user's norm, and urgency, not only on fit; and the natural output is don't buy, confirm first, or buy automatically. Across sixteen instruction-tuned models, autonomous commitment is misaligned with a human authorization reference in which fifteen annotators, role-playing the delegating principal from a given profile, choose the furthest action they would allow without confirmation: models commit even when fit is low and when their own evidence flag reports the warrant missing. This is overreach. The line is individual: these raters disagree, yet the disagreement is structured and p
    
[^90]: 理解投机解码中的目标概率复用

    Understanding Target Probability Reuse in Speculative Decoding

    [https://arxiv.org/abs/2610.29210](https://arxiv.org/abs/2610.29210)

    该论文系统研究了投机解码中目标模型下一个token分布的复用问题，证明了精确复用需要下一轮完整上下文中的分布，并提出用当前上下文单独计算的分布重排首token分支或先抽取首token再分配树，从而提高期望接受率并加快解码速度。

    

    投机解码使用目标模型并行验证一棵草稿token树。我们研究由此产生的下一个token分布是否有助于选择下一棵树。对于无限制的目标模型，我们证明了保证精确复用需要在下一轮完整上下文中计算的分布。我们刻画了何时保留该分布需要牺牲已被接受的token，并建立了这种损失在解码轮次间持续存在的条件。在六个目标模型的实验中，我们发现使用在当前上下文中单独计算的分布对首token分支进行重排序，相比草稿排序能够提高期望接受率，尤其是在树预算较小的情况下。在分配树之前先从该分布中抽取首token，相比对预先选择的树进行剪枝能进一步提高期望接受率，即使平均树大小相同。在我们的实验中，当……时，这两种选择规则的解码速度也比草稿排序更快。

    arXiv:2610.29210v1 Announce Type: new  Abstract: Speculative decoding uses a target model to verify a tree of draft tokens in parallel. We study whether the resulting next-token distributions can help select the next tree. For unrestricted target models, we show that guaranteed exact reuse requires a distribution computed at the next round's complete context. We characterize when retaining this distribution requires sacrificing accepted tokens and establish conditions under which this loss persists across decoding rounds. Across six targets, we find that reranking first-token branches with a separately computed distribution at the current context improves expected acceptance over draft ranking, especially with small tree budgets. Drawing the first token from this distribution before allocating the tree further improves expected acceptance over pruning a preselected tree, even at equal mean tree size. In our experiments, both selection rules also decode faster than draft ranking when th
    
[^91]: norregel：挪威法律中带日期数值的预注册基准

    norregel: a preregistered benchmark of dated numeric values in Norwegian law

    [https://arxiv.org/abs/2610.29207](https://arxiv.org/abs/2610.29207)

    该论文提出了norregel——一个预注册的基准测试，用于评估语言模型对挪威法律中随时间变化的数值知识的掌握，结果显示即使是表现最好的模型在主要条目上的正确率也不到一半，揭示了语言模型在法律数值知识方面的重大缺陷。

    

    norregel要求语言模型回答挪威法律、法规和议会决议中的数值（挪威克朗金额、百分比、国民保险基本金额（G）的倍数以及年龄限制），每个数值都对应一个特定日期。916个条目从Lovdata的开放数据中机械提取，每个条目都带有其条款的带日期历史记录，因此答案可以被分类为正确、过时、错误或无法解释。我们报告了在相同条目上进行的第一次和第二次预注册运行。在第一次运行中（五个模型，仅在当地分析完成后才公开列出），假设H1（即对于在模型截止日期之后生效的数值，过时答案更频繁）实际上无法测试，H2和H3的结果也不确定；Opus 5.5和Sonnet 4.6在880个主要条目中分别正确回答了46.7%和40.2%。第二次运行在任何调用之前就完成了预注册并发布，分析了八个模型。其汇总的主要测试……（摘要在此处截断）

    arXiv:2610.29207v1 Announce Type: new  Abstract: norregel asks language models for numeric values in Norwegian statutes, regulations and parliamentary decisions (NOK amounts, percentages, multiples of the National Insurance basic amount (G) and age limits), each at a stated date. The 916 items were extracted mechanically from Lovdata's open data, and each carries the dated history of its provision, so an answer can be classed as correct, outdated, wrong or uninterpretable. We report a first and a second preregistered run on the same items. In the first run (five models, listed publicly only after local analysis), H1, that outdated answers are more frequent for values that took effect after a model's cutoff, could not be tested in substance, and H2 and H3 were inconclusive; Opus 5.5 and Sonnet 4.6 answered 46.7% and 40.2% of the 880 primary items correctly. The second run was preregistered and published before any of its calls, and eight models were analysed. Its pooled primary test sup
    
[^92]: 推理-搜索增强的大语言模型：一种操作分类体系、基准分析与受控证据综合

    Reasoning-Search-Augmented LLMs: An Operational Taxonomy, Benchmark Analysis and Controlled Evidence Synthesis

    [https://arxiv.org/abs/2610.29189](https://arxiv.org/abs/2610.29189)

    该论文提出了一个针对推理-搜索增强大语言模型（LLMs）的操作性分类体系，通过检索耦合、控制器设计、知识来源和接地机制四个架构维度系统性地分析将逐步推理与交互式搜索紧密耦合的框架，并提供了基准分析与受控证据综合。

    

    大语言模型（LLMs）展现出令人印象深刻的涌现推理能力，但它们仍受限于固定的模型参数和有限的上下文窗口。即使通过静态检索进行增强，它们也可能产生不受检索证据支持、甚至与检索证据相矛盾的陈述。这导致了幻觉问题、难以融入最新信息，以及无法正确解读或协调不同信息来源的失败。近期的工作通过将逐步推理与交互式搜索紧密耦合来解决这些局限性，将检索视为迭代式智能体循环中的一个动作，而非一次性的预处理步骤。本研究回顾了通过交错执行规划、查询、检索、整合、验证和停止等环节来实现此类推理-搜索循环的框架。我们通过四个架构维度对这些系统进行分析：检索耦合方式、控制器设计、知识来源与接地机制。

    arXiv:2610.29189v1 Announce Type: cross  Abstract: Large language models (LLMs) demonstrate impressive emergent reasoning capabilities, yet they remain constrained by fixed model parameters and finite context windows. Even when augmented with static retrieval, they can produce statements that are unsupported by, or even contradictory to, retrieved evidence. This leads to hallucinations, difficulty incorporating up-to-date information, and failures to correctly interpret or reconcile sources. Recent work addresses these limitations by tightly coupling step-by-step reasoning with interactive search, treating retrieval as an action within an iterative agentic loop rather than a one-shot preprocessing step. This study reviews frameworks that implement such reasoning-search loops by interleaving planning, querying, retrieval, integration, verification, and stopping. We analyze these systems through four architectural axes: retrieval coupling, controller design, knowledge source, and groundi
    
[^93]: LTEM：面向表格理解与生成的LLM友好型表格增强Markdown格式

    LTEM: LLM-Friendly Table-Enhanced Markdown for Table Understanding and Generation

    [https://arxiv.org/abs/2610.29188](https://arxiv.org/abs/2610.29188)

    该论文提出LTEM——一种LLM友好的Markdown最小扩展格式，通过显式编码单元格跨度和层级表头来表达复杂表格结构，在保持轻量级token成本的同时，显著提升了大语言模型对真实文档中复杂表格的理解与生成准确率。

    

    现实世界文档中的表格，包括科学论文、财务报告和统计摘要，经常使用合并单元格、多级表头和分区布局。然而，大语言模型最擅长处理的轻量级格式（Markdown、CSV）无法表达这些结构，而表达力强的格式（HTML、LaTeX）则带来过高的token成本和解析复杂性。为了系统分析这种权衡，我们引入了一种分类法，将表格结构分解为六个原子特征（F1-F6）和相应的复杂度级别（L1-L6），并构建了STB（合成表格基准），这是一个包含1,089个表格的受控基准，这些表格系统地分布在各复杂度级别上。利用该框架，我们提出了LTEM（LLM友好的表格增强Markdown），这是一种形式化定义的Markdown最小扩展，显式编码单元格跨度和层级表头。在五个最先进的大语言模型上的实验表明，LTEM实现了准确率的……

    arXiv:2610.29188v1 Announce Type: new  Abstract: Tables in real-world documents, including scientific papers, financial reports, and statistical summaries, frequently use merged cells, multi-level headers, and sectioned layouts. Yet the lightweight formats LLMs handle best (Markdown, CSV) cannot express these structures, while expressive formats (HTML, LaTeX) impose excessive token cost and parsing complexity. To enable systematic analysis of this trade-off, we introduce a taxonomy decomposing table structure into six atomic features (F1-F6) and corresponding complexity levels (L1-L6), and construct STB (Synthetic Table Benchmark), a controlled benchmark with 1,089 tables systematically distributed across complexity levels. Using this framework, we propose LTEM (LLM-friendly Table-Enhanced Markdown), a formally specified, minimal extension of Markdown that explicitly encodes spans and hierarchical headers. Experiments across five state-of-the-art LLMs show that LTEM achieves accuracy c
    
[^94]: TrainBridge：由GPU编译证据引导的智能体化NPU训练优化

    TrainBridge: Agentic NPU Training Optimization Guided by GPU Compilation Evidence

    [https://arxiv.org/abs/2610.29171](https://arxiv.org/abs/2610.29171)

    TrainBridge提出一种智能体化方法，以成熟GPU编译栈的优化执行为参考，从中提取优化意图指导NPU程序重写并结合迭代式性能剖析进行精调，在四个工作负载上平均实现1.7倍的NPU训练加速。

    

    我们研究如何提升昇腾（Ascend）NPU上的训练效率。在我们所研究的NPU训练程序中，算子融合等框架级编译优化仅被部分应用，导致计算碎片化和可避免的执行开销。基于性能剖析引导的调优虽然可以解决这些低效问题，但往往需要代价高昂的反复试错。为了让NPU上的训练更加高效，我们提出了TrainBridge，这是一种将GPU优化经验迁移到NPU训练的智能体化方法。TrainBridge利用成熟GPU编译栈生成的高度优化的执行结果，作为NPU调优的任务特定参考。它从GPU训练执行中提取优化意图，用以指导NPU程序重写，并通过迭代式性能剖析进一步优化重写后的程序。我们的GPU到NPU迁移实验表明，TrainBridge在涵盖LLM预训练、图……的四个工作负载上平均实现1.7倍加速。

    arXiv:2610.29171v1 Announce Type: new  Abstract: We study how to improve training efficiency on Ascend NPUs. In the NPU training programs we study, framework-level compilation optimizations such as operator fusion are only partially applied, leaving fragmented computation and avoidable execution overheads. Profiling-guided tuning can address these inefficiencies, but it often requires costly trial and error. To make training on NPUs more efficient, we propose TrainBridge, an agentic approach for transferring GPU optimization guidance to NPU training. TrainBridge uses the highly optimized execution produced by mature GPU compilation stacks as a task-specific reference for NPU tuning. It extracts optimization intents from the GPU training execution to guide NPU program rewrites and further refines the rewritten program through iterative profiling. Our GPU-to-NPU migration experiments show that TrainBridge achieves an average 1.7x speedup across four workloads covering LLM pretraining, im
    
[^95]: TeachBack：通过双向偏好优化实现在线策略自蒸馏的“学会教学”

    TeachBack: Learning to Teach via Bidirectional Preference Optimization for On-Policy Self-Distillation

    [https://arxiv.org/abs/2610.29167](https://arxiv.org/abs/2610.29167)

    TeachBack提出了一种双向偏好学习框架，使模型不仅从掌握特权信息的教师处学习，还反过来优化教师的教学方式，从而生成更适合学生学习的推理轨迹。

    

    在线策略自蒸馏（OPSD）使语言模型能够从一个“教师”模型学习，该教师由同一个模型实例化而来，但被赋予了特权信息，例如参考答案。我们发现，拥有充分的特权信息可以让教师生成高度准确的轨迹；然而，仅凭准确性并不能保证这些轨迹适合学生学习。直接获取完整答案可能会促使教师生成简短的、依赖参考答案的推导，从而省略了在没有特权信息的情况下解决问题所必需的探索、验证和自我纠正等过程。我们提出了TeachBack，这是一个用于在线策略自蒸馏的双向偏好学习框架，它不仅从教师那里学习，还学习教师应该如何教学。TeachBack通过控制注入到提示中的参考推理后缀的长度等方式来塑造教师的生成行为……

    arXiv:2610.29167v1 Announce Type: cross  Abstract: On-policy self-distillation (OPSD) enables a language model to learn from a Teacher instantiated from the same model but given access to privileged information, such as a reference solution. We find that access to sufficient privileged information allows the Teacher to generate highly accurate trajectories; however, accuracy alone does not guarantee that these trajectories are suitable for Student learning. Direct access to a complete solution may encourage shortened, reference-conditioned derivations that omit the exploration, verification, and self-correction required to solve problems without privileged information. We introduce TeachBack, a bidirectional preference-learning framework for on-policy self-distillation that learns not only from the Teacher but also how the Teacher should teach. TeachBack shapes the Teacher's generation behavior by controlling the length of the reference-reasoning suffix injected into the prompt and by 
    
[^96]: 沿证据关节切分放射学报告：以解剖区域集合作为生成单元

    Carving Radiology Reports at Their Evidential Joints: Anatomical Region Sets as Generation Units

    [https://arxiv.org/abs/2610.29165](https://arxiv.org/abs/2610.29165)

    提出RadCarve方法，直接以放射学报告中现成语句-区域对应关系里的解剖区域集合作为生成单元，将单元划分与视觉证据选择统一，从而生成与视觉证据紧密绑定的细粒度放射学报告。

    

    将放射学报告组织成可独立生成的单元，为生成细粒度的临床内容并将其与视觉证据相绑定提供了一种途径。然而，现有方法通常采用的生成单元结构并非直接来源于自然报告或其标注中的规定，这引入了额外的建模妥协。我们提出RadCarve，它直接从现有语句-区域对应关系中的解剖区域集合来定义生成单元，将单元的形成与视觉证据选择统一起来。RadCarve的规划器使用面向区域组合的专用查询库，从图像中联合预测单元的存在性与区域组成，并根据当前成员预测所聚合的区域证据对两者进行迭代优化。生成器以其预测区域集合的联合视觉证据为条件，独立生成每个单元的文本。与强大的基线相比，RadCarve在CXRFESc（摘要在此处截断）

    arXiv:2610.29165v1 Announce Type: new  Abstract: Organizing radiology reports into independently generated units offers a way to generate fine-grained clinical content and bind it to visual evidence. However, existing methods typically adopt generation-unit structures not directly specified in natural reports or their annotations, introducing additional modeling compromises. We propose RadCarve, which instead defines generation units directly from the anatomical region sets in existing statement-region correspondences, unifying unit formation with visual evidence selection. RadCarve's planner uses composition-specialized query banks to jointly predict unit existence and region composition from the image, iteratively refining both with regional evidence aggregated according to current membership predictions. The generator independently produces each unit's text conditioned on the joint visual evidence of its predicted region set. Compared with strong baselines, RadCarve improves CXRFESc
    
[^97]: 为什么AI生成的语言如此难以理解？

    Why is AI-generated language so hard to understand?

    [https://arxiv.org/abs/2610.29162](https://arxiv.org/abs/2610.29162)

    该研究发现所有前沿AI模型都比人类写作更频繁地使用增加人类加工难度的心理语言学特征（如低频n-gram、长句法依存、名词堆叠等），并提出这是因为人类语言为成功沟通而优化，而AI生成的文本则服务于其他优化目标。

    

    人们普遍感觉到，AI生成的文本最近变得更难理解了。在本研究中，我们考察了一系列已知与人类语言加工难度相关的心理语言学特征（例如宾语关系从句、长句法依存、低频n-gram）。利用一个涵盖多个领域的新型AI生成文本语料库，我们发现，与通过往返翻译生成的改写文本相比，AI模型更频繁地使用这些困难特征。与人类写作相比，所有九个前沿及接近前沿的模型都使用了更多“无生命主语搭配有生命动词”的结构以及更多低频n-gram，大多数模型还使用更长的依存结构、更多的名词堆叠，并在对话中使用更多省略形式的宾语关系从句；我们还展示了不同模型家族之间的显著差异。我们提出并检验了一个解释原因的假设：人类语言为成功沟通而优化，而AI生成的文本则日益……（原文摘要在此处截断）

    arXiv:2610.29162v1 Announce Type: new  Abstract: There is a widespread sense that AI-generated text has recently become harder to understand. In this study, we investigate a set of psycholinguistic features that are known to correlate with human processing difficulty (e.g., object relative clauses, long syntactic dependencies, low-frequency n-grams). Using a novel corpus of AI-generated text across diverse domains, we find that AI models use these difficult features more often than paraphrases generated by round-trip translation. Compared with human writing, all nine frontier and near-frontier models use more inanimate subjects with animate verbs and more low-frequency n-grams, and most also use longer dependencies, more stacked nouns, and more reduced object relatives in conversation; we also demonstrate substantial variation across model families. We propose and test a hypothesis as to why: while human language is optimized for successful communication, AI-generated text is increasin
    
[^98]: VeeReview：从V模型视角测量智能体创作拉取请求中的构建-验证对

    VeeReview: Measuring Build-Verify Pairs in Agent-Authored Pull Requests from a V-Model Perspective

    [https://arxiv.org/abs/2610.29143](https://arxiv.org/abs/2610.29143)

    本文提出基于V模型构建-验证对的VeeReview方法，通过仓库信号和语言模型裁判测量拉取请求中的验证证据，发现智能体创作的拉取请求在全部四个验证维度上的“仅构建”率均低于人类拉取请求，与普遍预期相反。

    

    背景：审查拉取请求需要针对所提议变更的验证证据，而编程智能体基准测试和拉取请求结果研究并未直接评估这种关系。目标：我们检验V模型的构建与验证对能否测量拉取请求中的验证证据，并支持智能体创作与人类创作变更之间的比较。方法：VeeReview使用仓库信号和语言模型裁判对四对构建-验证对进行分类，其中三对使用受确定性信号约束的模型判断，另一对则通过基于差异和仓库树的规则来判定“模块设计-单元测试”对。我们在由两名评估者标注的25个拉取请求参考集上评估了VeeReview，并将其应用于来自741个仓库的1,454对匹配的智能体与人类拉取请求。结果：VeeReview在所有四对构建-验证对上为智能体创作的拉取请求分配了更低的“仅构建”率，这与……（摘要在此处截断）

    arXiv:2610.29143v1 Announce Type: cross  Abstract: Context: Reviewing a pull request requires verification evidence that addresses the proposed change. Coding agent benchmarks and studies of pull request outcomes do not directly assess this relationship. Objective: We test whether the V-Model's build and verify pairs can measure verification evidence in pull requests and support comparisons between agent-authored and human-authored changes. Method: VeeReview classifies four build and verify pairs using repository signals and a language-model judge. Three pairs use model judgements constrained by deterministic signals, and a rule over the diff and repository tree decides Module Design-Unit Test. We evaluated VeeReview against a 25-pull request reference set labelled by two raters and applied it to 1,454 matched pairs of agent and human pull requests from 741 repositories. Results: VeeReview assigned lower build-only rates to agent-authored pull requests on all four pairs, contrary to th
    
[^99]: GRAFT：恢复连续扩散语言模型中缺失的采样接口

    GRAFT: Restoring the Missing Sampling Interface in Continuous Diffusion Language Models

    [https://arxiv.org/abs/2610.29136](https://arxiv.org/abs/2610.29136)

    论文发现连续扩散语言模型因不进行逐步词级采样（缺失采样接口）而导致稀有词频率严重坍缩（最稀有词仅为人类频率的0.30倍），并提出GRAFT来恢复这一缺失的采样接口以修复词汇长尾问题。

    

    连续扩散语言模型以并行方式生成整个序列，在任何中间步骤都不会抽取单个词；只有最后一步才将嵌入转化为文本。ELF等模型如今以远少于离散词元扩散模型的采样步数达到了自回归级别的生成质量，而其自身的消融实验将这一优势归功于“不做逐步离散化”这一设计原则。我们证明，这一原则带来了隐藏的代价：词汇长尾的坍缩。与留出的人类文本相比，ELF的输出在常用词上的频率几乎与人类完全一致，但其最稀有词的出现频率仅为人类水平的0.30倍；这一差距在从1.05亿到6.52亿参数的规模范围内均未能弥合，且这些模型所暴露出的任何采样器设置都无法触及。相比之下，同等规模的GPT-2仅凭其解码规则即可覆盖从零到高于人类频率的范围。我们将这一差异归因于连续扩散模型所缺失的东西：w（原文在此处截断）

    arXiv:2610.29136v1 Announce Type: new  Abstract: Continuous diffusion language models generate a whole sequence in parallel, never drawing a single word at any intermediate step; only the final step turns embeddings into text. Models such as ELF now report autoregressive-level quality at a fraction of the sampling steps of discrete-token diffusion models, and this design principle -- no step-wise discretization -- is what their own ablations credit for the gain. We show that this same principle carries a hidden cost: a collapse of the lexical long tail. Measured against held-out human text, ELF's output matches the human rate on common words almost exactly, yet its rarest words appear at only 0.30x the human rate, a gap that does not close from 105M to 652M parameters and that no sampler setting these models expose can reach. A same-size GPT-2, in contrast, spans zero to above the human rate by its decoding rule alone. We trace the difference to what continuous diffusion models lack: w
    
[^100]: GRAM：基于子空间对齐的分组秩感知模型合并

    GRAM: Group-wise Rank-Aware Model Merging via Subspace Alignment

    [https://arxiv.org/abs/2610.29110](https://arxiv.org/abs/2610.29110)

    提出GRAM方法，通过主角度对齐分析将几何兼容的任务分组，并在组内进行秩感知的子空间对齐合并，可证明地降低投影误差，从而解决模型合并中的子空间干扰问题。

    

    将不同的模型合并为单一模型对于多任务部署而言极具价值，尤其是在医疗保健等受监管领域。在这类场景中，由于严格的数据隐私政策，模型通常只能在本地进行微调，这要求聚合操作仅基于微调后的权重进行。现有方法对所有任务采用统一的合并方式，忽视了各任务所学子空间之间的几何关系。当任务更新位于未对齐的子空间时，将其投影到共享的低秩基上会丢弃与任务相关的方向，这种失败模式被称为“子空间干扰”。通过本研究，我们证明了合并模型中每个任务的投影误差由该任务与共享子空间之间的主角度对齐程度所决定，因此将几何上兼容的任务进行分组可以在每个组内可证明地降低该误差。受此观察启发，我们提出了分组秩感知模型合并方法（GRAM）……

    arXiv:2610.29110v1 Announce Type: cross  Abstract: Merging different models into a single model is highly desirable for multi-task deployment, particularly in regulated domains like healthcare. In such settings, models are often locally fine-tuned due to strict data privacy policies, necessitating that aggregation operates exclusively on the fine-tuned weights. Existing methods apply a uniform merge across all tasks, ignoring the geometric relationships among their learned subspaces. When task updates occupy misaligned subspaces, projecting them onto a shared low-rank basis discards task-relevant directions, a failure mode named \emph{subspace interference}. Through this study, we show that the per-task projection error of a merged model is governed by the principal-angle alignment between each task and the shared subspace, so that grouping geometrically compatible tasks provably reduces this error within each group. Motivated by this observation, we propose \textbf{Group-wise Rank-Awa
    
[^101]: 基于全局主题的多级编码器面向任务型对话摘要生成

    Multi-level Encoder with Global Topic for Task-oriented Dialogue Summarization

    [https://arxiv.org/abs/2610.29097](https://arxiv.org/abs/2610.29097)

    该论文提出MLEGT模型，通过带全局主题的多级编码器捕捉对话的时序与空间依赖关系及全局主题信息，并结合全局主题引导的指针机制来生成更准确的面向任务对话摘要。

    

    面向任务的对话摘要旨在自动提取关键信息以生成领域摘要，从而提高服务效率和质量。面向任务的对话具有内在的逻辑性，并围绕特定主题展开。如何有效捕捉对话主题和最突出的信息成为该任务的主要挑战之一。在本文中，我们提出了一种利用带全局主题的多级编码器的面向任务对话摘要模型。该模型通过多级编码器发现时间和空间依赖关系，同时在图构建过程中获得一个简单而有效的全局主题。它不仅捕捉关键细节，还突破了长距离限制，并发现对话中的内在结构。然后，在摘要生成器中加入了全局主题引导的指针机制，以捕捉被忽略但重要的信息。

    arXiv:2610.29097v1 Announce Type: new  Abstract: Task-oriented dialogue summarization aims to automatically extract key information to generate domain summaries to improve service efficiency and quality. Task-oriented dialogue is inherently logical and surrounds specific topic. How to effectively capture the dialogue topic and the most salient information becomes one of the major challenges of this task. In this paper, we propose a task-oriented dialogue summarization model utilizing a multi-level encoder with global topic (MLEGT). This model discovers temporal and spatial dependencies through a multi-level encoder, while simultaneously obtaining a simple yet effective global topic in the process of graph construction. It not only captures the crucial details but also breaks the long-range limit and discover the intrinsic structure across the dialogue. Then, a global topic guided pointer mechanism is added into the summrizer to capture ignored but important information, which is helpfu
    
[^102]: 相同的答案，不同的理由：多语言大语言模型中的推理轨迹级分歧

    Same Answer, Different Reasons: Trace-Level Divergence in Multilingual Large Language Models

    [https://arxiv.org/abs/2610.29092](https://arxiv.org/abs/2610.29092)

    该论文提出一个轨迹级分歧框架，将多语言大语言模型的可见推理轨迹作为结构化序列分解为结构相似性、顺序一致性和整体相似性进行比较，发现标准的整体嵌入余弦比较方法会系统性高估跨语言的推理一致性。

    

    多语言大语言模型越来越多地被部署在可见推理轨迹能够支持验证、教育和安全审计的场景中，然而标准评估通常只衡量最终答案的一致性或整体响应的相似性。因此，这些评估只能建立结果层面的一致性，而无法确定不同语言的用户是否获得了可比性的支持证据。我们提出了一个轨迹级分歧框架，将可见的推理轨迹作为结构化序列进行比较，把一致性分解为结构相似性、顺序一致性和整体相似性三个维度，并采用针对特定语言对的校准方法以实现跨语言比较。在三个推理基准、七种语言和六个前沿大语言模型的实验中，自然的基线方法——对整个响应进行嵌入并通过余弦相似度进行比较——系统性地高估了一致性：在整体相似度最高四分位数的样本中，有 $52.1\%$–$67.3\%$ 的样本

    arXiv:2610.29092v1 Announce Type: cross  Abstract: Multilingual large language models are increasingly deployed in settings where visible reasoning traces support verification, education, and safety auditing, yet standard evaluations typically measure only final-answer agreement or holistic response similarity. As a result, they establish outcome consistency without determining whether users across languages receive comparable supporting evidence. We introduce a trace-level divergence framework that compares visible reasoning traces as structured sequences, decomposing agreement into structural similarity, order consistency, and holistic similarity, with pair-specific calibration to enable comparison across languages. Across three reasoning benchmarks, seven languages, and six frontier LLMs, the natural baseline---embedding each whole response and comparing the two by cosine---systematically overstates agreement: $52.1\%$--$67.3\%$ of examples in the top quartile of holistic similarity
    
[^103]: 角色增强零样本思维链提示的大规模评估

    A Large-Scale Evaluation of Role-Enhanced Zero-Shot Chain-of-Thought Prompting

    [https://arxiv.org/abs/2610.29073](https://arxiv.org/abs/2610.29073)

    本文提出RE-ZS-CoT提示框架，通过静态与动态专家角色选择机制增强大语言模型的零样本推理能力，在20个模型和七个基准上最高取得44个百分点的性能提升。

    

    本文提出了角色增强零样本思维链（Role-Enhanced Zero-Shot Chain-of-Thought，RE-ZS-CoT），这是一种提示框架，通过系统性地整合特定领域的专家角色来增强大语言模型（LLM）的推理能力，且无需任务特定的示例。我们评估了该方法的两个版本。在第一个版本（静态角色选择）中，我们在算术、常识和阅读理解基准上评估了八个预定义的专家角色，以确定它们对推理性能的影响。在第二个版本（动态角色选择）中，每个模型自主地为每个基准选择最合适的单一角色，从而在无需人工指定的情况下实现自适应的任务-角色对齐。在20个大语言模型和七个基准测试上，RE-ZS-CoT经常超越零样本和零样本思维链基线，实现了高达44个百分点的性能提升。研究结果揭示了静态与动态版本之间的强对应关系。

    arXiv:2610.29073v1 Announce Type: new  Abstract: This paper proposes Role-Enhanced Zero-Shot Chain-of-Thought (RE-ZS-CoT), a prompting framework that systematically integrates domain-specific expert roles to enhance reasoning in large language models (LLMs) without task-specific examples. We assess two versions of our approach. In the first version, static role selection, we evaluate eight predefined expert roles across arithmetic, commonsense, and reading comprehension benchmarks to determine their impact on reasoning performance. In the second version, dynamic role selection, each model autonomously selects the single most suitable role for each benchmark, enabling adaptive task-role alignment without manual specification. Across 20 LLMs and seven benchmarks, RE-ZS-CoT frequently surpasses zero-shot and zero-shot-CoT baselines, achieving performance improvements of up to 44 percentage points. The results reveal a strong correspondence between static and dynamic versions; the benchmar
    
[^104]: 我们自己的Jev：单次前向传播中的类型化决策，从自研裁判到开放LoRA模型

    Our Own Jev: Typed Decisions in One Forward Pass, from an In-House Judge to Open LoRA Models

    [https://arxiv.org/abs/2610.29052](https://arxiv.org/abs/2610.29052)

    该论文提出用单次前向传播的“类型化决策”接口（直接读取logits或轻量指针头）替代生成式LLM裁判，其自研System 1裁判仅用一块GPU约3分钟、约0.06美元即可判定1,000个智能体编码会话，以约1%的成本取代每轮约1,000美元且标签一致性仅26.9%的前沿LLM裁判。

    

    类型化决策将证据、问题和一个固定的选项集映射为每个选项的概率，只需一次前向传播即可完成，方式是读取语言模型的logits或在其隐藏状态上附加一个小型头部，而不是生成并解析文本。商业系统Jev推广了这种System 1接口，开源复现也随之出现。我们报告了该模式的两种用途。第一部分描述了一个自研的System 1裁判，它为一个训练语料库筛选了约14.8万条智能体编码会话记录。它取代了一个前沿LLM裁判——后者每轮处理的成本约在1,000美元量级，且在两位裁判达成一致的会话中，其存储的标签与新做的双裁判共识仅有26.9%吻合；一个自托管的冻结骨干网络，配合秩为16的LoRA适配器和指针头，可在一次前向传播中回答三个类型化问题，在一块GPU上约三分钟内即可判定1,000个会话，按参考价格计算成本约0.06美元，大约仅为前沿模型成本的1%……

    arXiv:2610.29052v1 Announce Type: new  Abstract: A typed decision maps evidence, a question and a fixed set of options to a probability for every option, computed in one forward pass, by reading a language model's logits or a small head over its hidden states, instead of generating and parsing text. The commercial system Jev popularised this System~1 interface, and open reproductions have followed. We report two uses of the pattern. \emph{Part I} describes an in-house System~1 judge that triages about 148K agentic coding session records for a training corpus. It replaces a frontier LLM judge that cost on the order of \$1,000 per pass and whose stored labels matched a fresh two-judge consensus on only 26.9\% of the sessions where the two judges agreed; a self-hosted frozen backbone with a rank-16 LoRA adapter and a pointer head answers three typed questions in one forward pass, judges 1,000 sessions in about three minutes of one GPU, about \$0.06 at a reference rate and roughly 1\% of t
    
[^105]: 目标重生成削弱了投机解码中草稿训练目标的收益

    Target Regeneration Shrinks the Gains of Draft-Training Objectives in Speculative Decoding

    [https://arxiv.org/abs/2610.29013](https://arxiv.org/abs/2610.29013)

    研究发现，投机解码中新的草稿训练目标（如令牌级蒸馏和验证门控蒸馏）在外部回复数据上的收益，在经过目标模型贪婪重生成后大幅缩水，且该现象在多种训练与解码设置下均稳定存在。

    

    投机解码通过并行验证草稿令牌来加速大型语言模型的推理。新的草稿训练目标通常被认为是相比交叉熵在对外部撰写的回复上带来更高接受率的原因，然而许多训练流水线会让目标模型重写这些回复。我们探究了这些收益有多少能够保留下来。通过一个视觉-语言目标模型和一个块草稿器，我们在同一语料库的四个版本上训练了三种目标：语料令牌上的交叉熵、来自目标模型的令牌级蒸馏，以及仅在能通过验证的位置上启用的门控蒸馏。在外部回复上，蒸馏和门控都带来了显著收益，但在经过贪婪重生成后，蒸馏的收益变小，而门控几乎或完全没有收益。这种缩减在不同数据顺序、位置权重、验证器、解码温度下持续存在；对于蒸馏而言，在不同初始化和学习率下也是如此。在匹配监督长度以及每个目标的b……（原文截断）

    arXiv:2610.29013v1 Announce Type: new  Abstract: Speculative decoding accelerates large language models by verifying drafted tokens in parallel. New draft-training objectives are usually credited with the acceptance they add over cross-entropy on externally written responses, yet many pipelines let the target rewrite them. We ask how much of that gain survives. With a vision-language target and a block drafter, we train three objectives on four versions of one corpus: cross-entropy on corpus tokens, token-level distillation from the target, and distillation gated to positions that would survive verification. Distillation and the gate add large gains on external responses, but after greedy regeneration, distillation adds a smaller gain and the gate little or none. The shrinkage persists across data orders, position weights, verifiers, decoding temperatures and, for distillation, initializations and learning rates. Most of it remains at matched supervised length and at each objective's b
    
[^106]: VeriEvolve：面向工具使用智能体的“评估器优先”进化框架

    VeriEvolve: Evaluator-First Evolution for Tool-Using Agents

    [https://arxiv.org/abs/2610.29005](https://arxiv.org/abs/2610.29005)

    VeriEvolve提出“评估器优先”的进化框架，先从任务指令自动合成可执行的评估器和挑战状态作为可靠的适应度信号，再据此进化工具使用智能体，将MCPMark文件系统任务的外部验证成功率从6/30提升至18/30。

    

    自我改进智能体面临的一个核心障碍是：对于现实中有状态的任务，缺乏可靠的适应度信号。我们提出了VeriEvolve，这是一个“评估器优先”的进化框架，它在优化智能体之前先构建这一缺失的评估层。VeriEvolve从任务指令出发，合成基于指令的挑战状态以及一个可执行的评估器来检验这些状态。在每轮工作流进化过程中，评估器被冻结并作为适应度信号；在两轮之间，VeriEvolve通过将强运行与弱运行的最终状态转换为一对反事实测试用例来修订评估器。在全部30个MCPMark文件系统任务上，VeriEvolve构建了30个包含141个挑战状态的评估器，在生成的测试套件上达到141/141的一致性，在执行智能体产生的保留状态上达到113/122的一致性。评估器引导的进化使外部验证的成功率从6/30提升到18/30。

    arXiv:2610.29005v1 Announce Type: cross  Abstract: A central obstacle to self-improving agents is the absence of a reliable fitness signal for realistic stateful tasks. We introduce \textsc{VeriEvolve}, an evaluator-first evolution framework that constructs this missing evaluation layer before optimizing the agent. From task instruction, VeriEvolve synthesizes instruction-grounded challenge states and an executable evaluator that checks them. During each round of workflow evolution, the evaluator is frozen and serves as the fitness signal. Between rounds, VeriEvolve revises the evaluator by converting the end states of strong and weak runs into a pair of counterfactual test cases.   Across all 30 MCPMark Filesystem tasks, VeriEvolve constructs 30 evaluators with 141 challenge states, achieving 141/141 agreement on the generated suites and 113/122 agreement on held-out states produced by executor agents. Evaluator-guided evolution increases externally verified success from 6/30 to 18/30
    
[^107]: FlashNext-RL：面向 Qwen3.8-Flash-Next 长程智能体强化学习的系统优化研究

    FlashNext-RL: A System Optimization Study for Long-Horizon Agentic RL of Qwen3.8-Flash-Next

    [https://arxiv.org/abs/2610.28998](https://arxiv.org/abs/2610.28998)

    本文提出 FlashNext-RL 框架，通过优化主机卸载、梯度累积以及混合专家、稀疏注意力和门控残差流等专用内核，实现了在单个 8xB200 节点上对 Qwen3.8-Flash-Next 完整 48 层模型的 BF16 长程智能体强化学习训练，显著提升了长序列训练吞吐量。

    

    长程智能体强化学习（RL）既需要高效的 rollout 生成，也需要在长序列上进行训练。然而，Qwen3.8-Flash-Next 的显存需求和模型专用算子使得现有的开源实现难以完成这一任务。为解决这些局限，我们提出了 FlashNext-RL，一个基于 prime-rl 的模型专用 RL 训练优化框架，并使用 vLLM 进行软件工程（SWE）rollout 生成。我们的主要目标是在单个 8xB200 节点上对完整的 48 层模型进行 BF16 训练，其中前两层、n-gram 表、稀疏注意力索引器和多 token 预测头被冻结。我们优化了主机内存卸载、梯度累积，以及针对混合专家、稀疏注意力和门控残差流的内核。在全模型开发研究中，非填充训练吞吐量从初始 32K-token 配置下的约 395 tokens/s/GPU 提升至……

    arXiv:2610.28998v1 Announce Type: cross  Abstract: Long-horizon agentic reinforcement learning (RL) requires both efficient rollout generation and training on long sequences. However, the memory requirements and model-specific operators of Qwen3.8-Flash-Next make this difficult with existing open-source implementations. To resolve these limitations, we introduce FlashNext-RL, a model-specific RL training optimization framework based on prime-rl, with vLLM for software engineering (SWE) rollout generation. Our main focus is BF16 training of the full 48-layer model on a single 8xB200 node, with the first two layers, the n-gram table, the sparse-attention indexer and the multi-token prediction head frozen. We optimize host offload, gradient accumulation and the kernels for mixture-of-experts, sparse attention and the gated residual stream. In the full-model development study, non-padding training throughput increases from approximately 395 tokens/s/GPU in the initial 32K-token configurati
    
[^108]: 以语言损伤为透镜审视大语言模型

    Language Impairment as a Lens onto Large Language Models

    [https://arxiv.org/abs/2610.28992](https://arxiv.org/abs/2610.28992)

    该论文提出了一种受失语症启发的机制可解释性框架，通过分类器引导损伤与连续激活引导两种方法系统性地诱发大语言模型的“失语症”式功能崩溃，揭示了其语言功能既可被隔离到特定回路与表征向量、又彼此高度纠缠的双重特性，为理解大语言模型的功能组织结构提供了全新视角。

    

    尽管大语言模型（LLM）在高度分布式的空间中编码语言，它们仍然容易受到局部性破坏的影响——这一现象与人类大脑局灶性损伤导致临床失语症的情况惊人地相似。为了探究这一理论张力，我们引入了一个受失语症启发的新型机制可解释性框架，以研究大语言模型是否拥有与人类语言系统类似的功能结构。通过两种互补的方法——分类器引导的损伤和连续激活引导——我们系统性地在大语言模型中诱发类似失语症的功能崩溃。我们发现，语言功能可以被成功地隔离到离散的结构性回路和连续的表征向量之中，然而这些功能仍然高度纠缠在一起，在不同的失语特征之间存在着显著的回路重叠和向量相似性。此外，尽管激活引导能够稳健地泛化到下游任务中，但结构性损伤（原文此处截断）……

    arXiv:2610.28992v1 Announce Type: new  Abstract: Although large language models (LLMs) encode language in highly distributed spaces, they remain vulnerable to focal disruptions---a phenomenon strikingly similar to how localized brain damage causes clinical aphasia in humans. To investigate this theoretical tension, we ask whether LLMs possess a functional structure analogous to the human language system, by introducing a novel mechanistic interpretability framework inspired by aphasia. Through two complementary approaches, classifier-guided lesioning and continuous activation steering, we systematically induce aphasia-like breakdowns in LLMs. We find that language functions can be successfully isolated to discrete structural circuits and continuous representational vectors, yet these functions remain densely entangled with significant circuit overlap and vector similarities across distinct aphasic profiles. Furthermore, while steering generalizes robustly across downstream tasks, struc
    
[^109]: 扫描还是索引？长文档RAG中的证据覆盖率、答案质量与成本

    Scan or Index? large Evidence Coverage, Answer Quality, and Cost in Long-Document RAG

    [https://arxiv.org/abs/2610.28957](https://arxiv.org/abs/2610.28957)

    该研究比较了11种长文档RAG检索配置，发现混合检索策略在证据覆盖率和答案准确率上均优于全文档扫描，且更高的证据覆盖率并不必然带来更好的答案质量。

    

    arXiv:2610.28957v1 公告类型：new 摘要：扫描长文档中的每一个段落看似彻底，但它是否保留了更多有用的证据或产生更好的答案？我们比较了11种检索配置，涵盖词法检索与稠密检索、混合重排序以及全文档扫描，涉及Jev、OpenAI Decisions和Cloudflare Clef。我们的评估使用396个QASPER问题和来自八本完整教科书的80个问题，采用相同段落集、五个段落的数量限制以及固定的答案生成器。在书籍数据上，Hybrid + Jev取得了最高的观测证据覆盖率（94.1%），相比之下Jev全扫描为88.7%，OpenAI稠密检索为81.3%。全扫描虽然最初会检查每一个段落，但后续的精简步骤可能会丢弃相关证据。更高的证据覆盖率也不一定能持续带来更好的答案：Hybrid + Decisions在检索方法中取得了最高的观测书籍答案准确率（0.892），尽管其恢复的参考证据较少。全扫描……

    arXiv:2610.28957v1 Announce Type: new  Abstract: Scanning every passage in a long document seems thorough, but does it preserve more useful evidence or produce better answers? We compare 11 retrieval configurations spanning lexical and dense search, hybrid reranking, and full-document scanning with Jev, OpenAI Decisions, and Cloudflare Clef. Our evaluation uses 396 QASPER questions and 80 questions from eight complete textbooks, with common passages, a five-passage limit, and a fixed answer generator. On books, Hybrid + Jev achieves the highest observed evidence coverage (94.1%), compared with 88.7% for Jev full scan and 81.3\% for OpenAI dense retrieval. Full scans examine every passage initially, yet later reduction steps can discard relevant evidence. Higher evidence coverage also does not consistently produce better answers: Hybrid + Decisions achieves the highest observed book Answer Accuracy among retrieval methods (0.892), despite recovering less reference evidence. Full scannin
    
[^110]: 连带之罪：语义泄漏驱动联想性性别偏见

    Guilt by Association: Semantic Leakage Drives Associative Gender Bias

    [https://arxiv.org/abs/2610.28944](https://arxiv.org/abs/2610.28944)

    该论文发现语言模型中的语义泄漏是一种广泛的联想倾向——概念无需被归属于特定实体，仅凭其在输入场景中的出现就能引导模型关于该实体的生成内容，并由此驱动联想性性别偏见。

    

    语义泄漏是指提示中的一个概念会引导模型对某个实体的生成内容，即使该概念与该实体无关（例如，喜欢黄色的人被说成是校车司机）。此前的研究大多集中在概念被直接归属于模型所询问的实体的场景中，这就留下了一个未解问题：泄漏到底是一种有缺陷的启发式方法（即模型为该实体构建一个画像），还是一种更广泛的联想倾向（例如，能够在看似不相关的实体之间起作用）？我们支持后一种观点：即使移除概念与特定实体之间的归属关系，泄漏依然存在。接着，我们将这一想法扩展到性别偏见。虽然针对语言模型性别偏见的已有研究通常考察刻板印象概念被归属于某个实体的情况，但我们的发现表明，一个概念仅凭其在场景中的出现就可能引导模型的输出。我们在两种条件下使用带有性别色彩的概念展示了这一点。

    arXiv:2610.28944v1 Announce Type: new  Abstract: Semantic leakage occurs when a concept in the prompt steers model generation about an entity, even though the concept is irrelevant to it (e.g., a person who likes yellow is said to work as a school bus driver). It has mostly been studied in settings where the concept is directly attributed to the entity the model is asked about. This leaves open whether leakage is a flawed heuristic, where the model builds a profile of that entity, or a much broader associative tendency, e.g., working across seemingly unrelated entities. We argue for the latter: even after removing the attribution of the concept to the specific entity, leakage persists. Next, we extend this idea to gender bias. While work on gender bias in LMs typically looks at cases where the stereotyped concept is attributed to an entity, our findings suggest that a concept may steer outputs with its mere presence in the scene. We show this using gendered concepts under two condition
    
[^111]: 拒绝方向在低资源语言中是否具有普适性？

    Is Refusal Direction Universal across Low-Resourced Languages?

    [https://arxiv.org/abs/2610.28897](https://arxiv.org/abs/2610.28897)

    该研究发现拒绝方向的跨语言普适性是分级的而非二元的：来自安全对齐语言的拒绝方向可广泛迁移至低资源语言，而来自强失对齐语言的拒绝方向则无法可靠地调节任何目标语言的拒绝行为。

    

    拒绝方向已被证明能够在安全对齐的语言之间以近乎完美的效果进行跨语言迁移（Wang et al., 2025），且据称不受资源水平的影响。然而，此前评估中唯一真正的低资源语言是约鲁巴语，该语言属于强失对齐语言且表现出有限的迁移效果，这使得拒绝方向在其他低资源语言中的表现仍是一个悬而未决的问题。我们将 PolyRefuse 扩展到另外三种低资源语言（白俄罗斯语、巴什基尔语和塔吉克语），并在 Qwen2.5-14B-Instruct 上复现了（Wang et al., 2025）的实验流程。我们发现，普适性是分级的而非二元的：来自安全对齐和部分对齐语言的拒绝方向能够广泛迁移，而一旦考虑天花板效应，来自强失对齐语言的拒绝方向则无法在任何测试目标上可靠地调节拒绝行为，包括其他失对齐语言。几何对齐与行为迁移进一步表……

    arXiv:2610.28897v1 Announce Type: new  Abstract: Refusal directions have been shown to transfer cross-lingually with near-perfect effectiveness across safety-aligned languages (Wang et al., 2025), reportedly regardless of resource level. However, the only truly low-resourced language evaluated is Yoruba, which is strongly misaligned and shows limited transfer, leaving open how refusal directions behave for other low-resourced languages. We extend PolyRefuse to three additional low-resourced languages (Belarusian, Bashkir, and Tajik) and replicate the pipeline of (Wang et al., 2025) on Qwen2.5-14B-Instruct. We find that universality is graded rather than binary: directions from safety-aligned and partially-aligned languages transfer broadly, while those from strongly misaligned languages fail to reliably modulate refusal behavior on any target tested, including other misaligned languages, once ceiling effects are accounted for. Geometric alignment and behavioral transfer are further sho
    
[^112]: CSD：面向跨分词器知识迁移的公共支撑集蒸馏

    CSD: Common-Support Distillation for Cross-Tokenizer Knowledge Transfer

    [https://arxiv.org/abs/2610.28895](https://arxiv.org/abs/2610.28895)

    本文提出CSD框架，通过在每个对齐输入分块处构建教师与学生模型间的公共支撑集，并以单一目标函数匹配分布，解决了词表和分词差异导致的跨分词器logits知识蒸馏对齐难题。

    

    跨分词器的logits蒸馏能够实现跨模型族的知识迁移，使学生模型能够从更强或更专精的教师模型中学习。然而，词表和分词方式的差异使得教师模型和学生模型的下一词元分布难以对齐。现有的基于logits的方法虽然易于集成到标准蒸馏流程中，但通常依赖于限制性的启发式方法或针对特定分词器对的选择。这些方法可能将监督限制在共享词表的子集上，需要为不同的分词器对手动选择不同的目标函数，或者仅匹配观察到的分块及其补集，而非逐个匹配备选的续写结果。我们提出了公共支撑集蒸馏，这是一个统一的基于logits的框架，它在每个对齐的输入分块处构建教师和学生模型之间的共享支撑集，并通过单一目标函数来匹配由此导出的分布。对于一对一的对齐情形，（摘要内容在此处被截断）

    arXiv:2610.28895v1 Announce Type: new  Abstract: Cross-tokenizer logits distillation enables knowledge transfer across model families, allowing students to learn from stronger or more specialized teachers. However, differences in vocabulary and segmentation make teacher and student next-token distributions difficult to align. Existing logits-based methods are easy to integrate into standard distillation pipelines, but often rely on restrictive heuristics or tokenizer-pair-specific choices. They may restrict supervision to shared vocabulary subsets, require manually selecting different objectives for different tokenizer pairs, or match only the observed chunk and its complement rather than individual alternative continuations. We introduce Common-Support Distillation (CSD), a unified logits-based framework that constructs a shared support between the teacher and student at each aligned input chunk and matches the induced distributions with a single objective. For one-to-one alignments, 
    
[^113]: 构建AI研究团队以实现线性注意力架构的自主发现

    Building AI Research Teams for Autonomous Discovery of Linear Attention Architectures

    [https://arxiv.org/abs/2610.28887](https://arxiv.org/abs/2610.28887)

    该论文提出构建一支由不同专业背景的持久性AI研究员组成的协作团队，通过平衡探索与利用、让成员基于彼此的实验发现持续迭代，在有限实验预算下自主发现线性注意力架构。

    

    自主智能体日益能够提出想法、修改代码并运行实验，这为利用协作式AI研究团队加速科学发现创造了机会。为了在发现过程中摆脱局部最优，我们提出通过构建由具有不同专业特长的智能体组成的团队来鼓励正交探索。然而，实现这一设计不仅需要多样化的背景。当实验成本高昂时，团队只能测试有限数量的想法，因此在选择一部分研究人员来提出并执行新实验时，需要平衡探索与利用。此外，每一项实验结果都应当影响团队下一步的尝试方向。因此，我们提出这样一个问题：具有不同科学背景的AI研究人员如何能够在彼此发现的基础上继续研究，并在有限的实验预算下保持持续进展。为了应对这些挑战，我们构建了一个由持久性AI研究人员组成的团队，每个成员都基于……

    arXiv:2610.28887v1 Announce Type: new  Abstract: Autonomous agents can increasingly propose ideas, modify code, and run experiments, creating an opportunity to accelerate scientific discovery using teams of collaborative AI researchers. To escape local minima in the discovery, we propose to encourage orthogonal exploration by constructing the team using agents with varying expertise. However, realizing this design requires more than diverse backgrounds. When experiments are costly, a team can test only a limited number of ideas, thereby creating the need to balance exploration and exploitation when selecting a subset of researchers to propose and execute new experiments. Furthermore, each result should shape what the team tries next. We therefore ask how AI researchers with different scientific backgrounds can build on one another's findings and sustain progress under a limited experimental budget. To address these challenges, we build a team of persistent AI researchers, each grounded
    
[^114]: SANE：面向LLM评判器的自动化的评分器特定变异测试

    SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges

    [https://arxiv.org/abs/2610.28886](https://arxiv.org/abs/2610.28886)

    SANE是一个基于智能体的自动化框架，能针对特定LLM评判器自动识别其评分能力、生成变异测试配方及预期分数关系，从而将评判器可靠性测试实现全流程自动化，摆脱了人工编写扰动配方和预设评分器类型的限制。

    

    LLM评判器（LLM judges）在基准评估、模型选择、数据筛选和模型训练中扮演着越来越重要的角色。只有当每个评判器能够对影响其评分标准的变动做出可靠响应时，其决策才有价值。现有的可靠性测试方法通过对输入施加具有已知预期评分影响的扰动来进行测试。主流的LLM评判器测试框架将人工编写的扰动配方与合成测试生成相结合，但将针对特定评分器的配方构思排除在自动化流程之外。这限制了它们只能使用固定的扰动方式和预先假设的评分器类型。我们提出了SANE，一个基于智能体（agent）的框架，它将随机评估器的行为需求编码为可复用的智能体指令。给定评分器规范、实现代码和代表性输入，该流水线首先识别评分器特定的能力，然后为每种能力自动推导扰动配方和预期的分数关系。SANE还定义了一个（摘要内容在此处被截断）

    arXiv:2610.28886v1 Announce Type: new  Abstract: LLM judges increasingly mediate benchmark evaluation, model selection, data curation, and training. Their decisions are useful only when each judge responds reliably to changes affecting its scoring criterion. Existing methods for reliability testing perturb inputs with known expected effects on judge scores. Prominent LLM-judge testing harnesses combine human-authored perturbation recipes with synthetic test generation, leaving scorer-specific recipe ideation outside the automated pipeline. This limits their use to fixed perturbations and pre-assumed scorer types. We introduce SANE, an agent-based framework that encodes behavioral requirements for stochastic evaluators as reusable agent instructions. Given a scorer specification, implementation, and representative inputs, the pipeline first identifies scorer-specific capabilities, then automatically derives perturbation recipes and expected score relations for each. SANE also defines a 
    
[^115]: 操纵与否：大语言模型交互中的观点动态

    To Manipulate or Not: Opinion Dynamics in LLM Interactions

    [https://arxiv.org/abs/2610.28875](https://arxiv.org/abs/2610.28875)

    该研究提出受控的LLM-LLM辩论框架并同时追踪显式立场与逐轮行为，发现同模型交互倾向于缩小观点差距而跨模型交互则扩大差距，揭示了不同模型间说服动态的显著差异。

    

    大语言模型越来越多地参与到长时间的对话中，在这些对话里它们既能塑造对方的观点，也会被对方的观点所影响，然而大多数说服研究只测量了这种交流的单个方面。我们提出了一个受控的LLM-LLM辩论框架，其中两个语言模型角色围绕一个有争议的话题（气候变化的归因）反复展开讨论：其中一个角色被赋予指定立场，并且在某些条件下还会收到明确的说服指令，而另一个角色则没有被指定任何立场。在涉及GPT-6 Luna和Gemini 2.5 Flash Lite的四种有序模型组合中，我们同时追踪了每个角色明确报告的立场，以及对其在对话中逐轮言行的独立测量。不同模型组合下的观点动态存在显著差异：同模型交互平均而言会缩小观点差距（且明显地仅对两个模型中的一个成立），而跨模型交互则会扩大观点差距。明确报告与内部……（原文摘要在此处截断）

    arXiv:2610.28875v1 Announce Type: cross  Abstract: Large language models increasingly take part in extended conversations where they can both shape and be shaped by another party's views, yet most persuasion research measures only one side of that exchange. We introduce a controlled LLM-LLM debate framework in which two language-model roles repeatedly discuss a contested topic (climate-change attribution): one is given an assigned stance and, in some conditions, an explicit instruction to persuade, while the other receives no assigned position. Across four ordered model combinations involving GPT-6 Luna and Gemini 2.5 Flash Lite, we track both each role's explicitly reported position and a separate measure of its turn-by-turn behavior inside the conversation. Opinion dynamics differ substantially across model combinations: same-model interactions narrow the opinion gap on average, clearly for only one of the two models, while cross-model interactions widen it. Explicit reports and in-d
    
[^116]: 超越正确答案：组合线索如何塑造视觉判断

    Beyond Correct Answers: How Compositional Cues Shape Visual Judgments

    [https://arxiv.org/abs/2610.28843](https://arxiv.org/abs/2610.28843)

    该论文通过控制变量方法系统解构结构化提示中各字段标签对视觉绑定任务的影响，发现标签效果因模型而异，且Qwen3在保留提示下的优势主要源于交换提示导致的错误。

    

    视觉问答要求模型将属性和关系正确分配给相应的对象，这一问题被称为视觉绑定。结构化提示通过带标签的字段使这些分配显式化。然而，完整提示带来的改进并不能揭示哪些标签有帮助，以及它们之间如何相互作用。首先，我们通过固定图像和陈述，在十个模型和六个任务上分离出各字段标签的影响。这些比较揭示了总体准确率所能隐藏的差异：在存在关系标签的情况下，添加对象角色标签对Qwen3在空间任务上有帮助，但对Qwen2却有害。其次，我们将保留或交换问题角色分配的辅助提示与无提示基线进行比较。在两个图像集上，Qwen3在保留提示下的优势主要反映了交换提示所导致的错误，而保留提示的效果仍接近基线水平，字段顺序还会进一步影响这种危害。第三，我们测试了更清晰的提示措辞以及直接选择……

    arXiv:2610.28843v1 Announce Type: cross  Abstract: Visual question answering requires models to assign attributes and relations to the correct objects, a problem known as visual binding. Structured prompts make these assignments explicit through labeled fields. Yet an improvement from the full prompt does not reveal which labels help or how they interact. First, we isolate field-label effects across ten models and six tasks by holding the image and claim fixed. These comparisons reveal differences that overall accuracy can hide. With a relation label present, adding object-role labels helps Qwen3 but hurts Qwen2 on Spatial. Second, we compare auxiliary hints that preserve or swap the question's role assignments against a no-hint baseline. Across two image sets, Qwen3's advantage with preserving hints mainly reflects errors caused by swapped hints. Preserving hints remain near baseline. Field order further affects the harm. Third, we test clearer hint wording and direct selection of the
    
[^117]: KIRA：面向儿童的双通道分析全栈阅读筛查系统

    KIRA: A Full-Stack Reading Screening System with Dual-Pathway Analysis for Children

    [https://arxiv.org/abs/2610.28840](https://arxiv.org/abs/2610.28840)

    KIRA是一个面向儿童的双通道阅读筛查系统，通过将预训练语音识别模型适配儿童语音（词错误率从13.88%降至8.84%，音素错误率从41.66%降至23.00%），实现了单词级流畅度评分与音素级错误检测的结合，并通过教师界面支持针对性阅读干预。

    

    早期年级的阅读筛查对于识别存在阅读困难风险的儿童至关重要。现有的基于AI的筛查系统往往在分析范围和评估指标上存在局限，或者是一些缺乏透明模型规格与评估的专有系统。我们提出了KIRA，一个将单词级流畅度评分与详细的音素级错误检测相结合的双通道筛查系统。我们将预训练的单词和音素识别器适配到儿童语音上，将单词错误率从13.88%降低到8.84%，音素错误率从41.66%降低到23.00%。基于这些针对儿童适配的模型，我们实现了单词阅读（WRE）和口头阅读流畅度（ORF）这两项K-2年级筛查中的基础任务。该系统将这些识别器与面向教师的网页界面集成，呈现流畅度分数以及音素反馈，以便进行潜在的针对性干预。KIRA将现代儿童语音识别技术引入到一个全栈……（原文摘要到此截断）

    arXiv:2610.28840v1 Announce Type: new  Abstract: Early-grade reading screening is critical for identifying children at risk of reading difficulty. Existing AI-based screeners are often limited in the scope of their analyses and evaluation metrics, or are proprietary systems that lack transparent model specifications and evaluation. We present KIRA, a dual-pathway screening system that pairs word-level fluency scoring with detailed phoneme-level error detection. We adapt pretrained word and phoneme recognizers to children's speech, reducing word error rate from 13.88% to 8.84% and phoneme error rate from 41.66% to 23.00%. Powered by these child-adapted models, we implement Word Reading (WRE) and Oral Reading Fluency (ORF), two foundational tasks in K-2 screening. The system integrates these recognizers with a web-based teacher-facing interface presenting fluency scores alongside phoneme feedback for potential targeted intervention. KIRA brings modern child-speech recognition into a full
    
[^118]: 如何在多语言预训练中使用平行数据

    How to Use Parallel Data for Multilingual Pretraining

    [https://arxiv.org/abs/2610.28838](https://arxiv.org/abs/2610.28838)

    该论文提出将文档与其翻译拼接在同一训练序列中的平行预训练（PPT）方法，并发现尽管平行数据有诸多益处，PPT相比标准预训练（SPT）并不能持续改善语言理解能力。

    

    高质量的人工翻译平行语料库可以改善跨语言迁移，但其规模通常太小，无法用于大规模预训练。机器翻译的网络语料库有助于克服这一限制，使平行数据能够在预训练规模上使用。然而，标准的多语言预训练是独立采样源文档及其翻译的，没有显式利用二者的平行结构。我们研究了如何利用数据的平行结构来改善多语言预训练，方法是将每个文档与其翻译拼接在同一个训练序列中。我们将这种平行预训练（PPT）方法与标准预训练（SPT）进行比较，在匹配的token预算下从头训练模型。令人惊讶的是，尽管先前的跨语言研究报道了平行数据的益处，但仅使用PPT并不能持续改善语言理解能力。我们的进一步分析表明，尽管PPT构建了……

    arXiv:2610.28838v1 Announce Type: new  Abstract: High-quality human-translated parallel corpora can improve cross-lingual transfer, but are typically too small for large-scale pretraining. Machine-translated web corpora help overcome this limitation, making parallel data available at pretraining scale. Standard multilingual pretraining, however, samples source documents and their translations independently, without explicitly using their parallel structure. We investigate how to use the parallel structure of the data to improve multilingual pretraining by concatenating each document with its translation in the same training sequence. We compare this parallel pretraining (PPT) approach with standard pretraining (SPT), training models from scratch at matched token budgets. Surprisingly, despite the benefits of parallel data reported in prior cross-lingual work, PPT alone does not consistently improve language understanding over SPT. Our further analysis suggests that although PPT builds 
    
[^119]: 陷入困境的智能体应该把接下来的令牌花在哪里？推理时资源的因果归因

    Where Should a Stuck Agent Spend Its Next Tokens? Causal Attribution of Inference-Time Resources

    [https://arxiv.org/abs/2610.28834](https://arxiv.org/abs/2610.28834)

    提出班扎夫资源归因框架，通过受控因子实验因果地量化计算、信息、反馈等推理时资源对修复失败编码任务的边际贡献，发现任务相关规范最有价值、额外计算的价值随模型能力递减，而多数前沿模型却错误地将预算分配给收益甚微的反馈资源。

    

    推理时扩展通常被视为令牌预算问题，但一个失败的编码智能体可以寻求性质不同的资源：计算、信息或反馈。我们提出了班扎夫资源归因，这是一个受控的因子实验框架，在保持失败状态和修订协议固定的情况下，测量每种资源对修复的边际贡献及其交互作用。我们在MBPP+和HumanEval+的194个函数级失败以及SWE-bench Verified的126个仓库级失败上评估了六个模型。任务相关的规范始终能改善修复效果，而公开测试反馈仅带来小幅提升，检索到的代码几乎没有作用。额外计算的价值随模型能力的增强而下降，而信息仍然有用；资源通道大多表现出收益递减，尽管互补信息可以自我强化。然而，四个前沿模型中有三个选择了反馈……

    arXiv:2610.28834v1 Announce Type: cross  Abstract: Inference-time scaling is often treated as a token-budget problem, but a failed coding agent can seek qualitatively different resources: computation, information, or feedback. We introduce Banzhaf Resource Attribution (BRA), a controlled factorial framework that measures each resource's marginal contribution to repair and its interactions while holding the failed state and revision protocol fixed. We evaluate six models on 194 function-level failures from MBPP+ and HumanEval+ and 126 repository-level failures from SWE-bench Verified. Task-relevant specifications consistently improve repair, whereas public-test feedback provides small gains and retrieved code little to none. The value of additional computation declines with model capability, while information remains useful; resource channels largely exhibit diminishing returns, although complementary information can reinforce itself. Yet three of four frontier models select feedback in
    
[^120]: 一个平行多方言阿拉伯语依存树库

    A Parallel Multi-Dialect Arabic Dependency Treebank

    [https://arxiv.org/abs/2610.28823](https://arxiv.org/abs/2610.28823)

    该论文发布了首个涵盖五种城市方言的平行阿拉伯语依存树库MADARTB（共5000个句子），揭示了各方言与标准阿拉伯语整体句法相似但局部结构差异显著，并通过实验表明有监督解析器（87.5% LAS）在跨方言依存分析上优于最佳的大语言模型（85.7% LAS）。

    

    阿拉伯语依存句法分析资源的开发长期以来主要集中在现代标准阿拉伯语上，导致方言阿拉伯语的资源相对匮乏。我们介绍了MADARTB，这是首个覆盖五种城市方言（贝鲁特、开罗、多哈、拉巴特和突尼斯）的大规模平行依存树库，它扩展了现有的平行现代标准阿拉伯语版本。MADARTB总共包含5,000个句子，采用CATiB依存标注方案进行标注。我们分析了所有城市方言与现代标准阿拉伯语之间的句法相似性。尽管各变体的整体构式特征表现出高度相似性，但局部句法结构经常出现分歧，揭示了方言之间显著的结构差异。我们进一步使用有监督解析器和四个大语言模型建立了跨方言依存句法分析的比较。在测试集上，有监督解析器在所有六个变体上达到了87.5%的LAS，而最佳的大语言模型配置则以85.7%的LAS落后。我们发布了我们的标注（原文在此处截断）。

    arXiv:2610.28823v1 Announce Type: new  Abstract: Dependency parsing resource development for Arabic has largely focused on Modern Standard Arabic, leaving dialectal Arabic comparatively under-resourced. We introduce MADARTB, the first large-scale parallel dependency treebank spanning five city dialects: Beirut, Cairo, Doha, Rabat and Tunis, and extending an existing parallel MSA version. In total MADARTB consists of 5,000 sentences annotated using the CATiB dependency scheme. We analyze syntactic similarity across all city dialects and MSA. Although global construction profiles exhibit high similarity across varieties, local syntactic structures frequently diverge, revealing substantial structural variation across dialects. We further establish a cross-dialect dependency parsing comparison using a supervised parser and four LLMs. The supervised parser achieves 87.5% LAS, while the best LLM setup trails at 85.7% LAS across all six varieties on the test set. We release our annotations an
    
[^121]: LIFT：用于评估和训练指令遵循的范围限定约束

    LIFT: Scoped Constraints for Evaluating and Training Instruction Following

    [https://arxiv.org/abs/2610.28819](https://arxiv.org/abs/2610.28819)

    本文提出Lift-Bench指令遵循基准，通过将约束限定在文档特定部分并同时提供主任务约束与指令约束，缓解了强化学习中的奖励破解问题，并配套了可用于训练的Lift-Train数据集。

    

    常见的指令遵循（IF）基准（如IFEval）评估语言模型（LM）遵循可通过程序轻松验证的简单约束的能力。而这些相同的约束又被用作强化学习的奖励信号。这种设计可能导致模型对约束进行奖励破解，使模型偏离主要任务。近期为解决这一问题所做的尝试产生了包含更丰富约束的基准，但这些基准需要专家人力投入、保持封闭访问、且难以扩展。我们提出了Lift-Bench，这是一个指令遵循基准，要求模型根据给定的主题和结构化大纲起草文档。约束被限定在特定的文档部分，从而对约束密度提供更精细的控制。Lift-Bench既包含可验证的约束，也包含需评审的约束，同时还提供了配套的训练集Lift-Train。重要的是，Lift-Bench同时提供主任务和指令遵循约束，缓解了（原文在此截断）

    arXiv:2610.28819v1 Announce Type: new  Abstract: Common instruction following (IF) benchmarks, such as IFEval, evaluate the ability of language models (LMs) to follow simple constraints that can easily be verified programmatically. These same constraints, in turn, are used as reward signals for reinforcement learning. This design may lead to reward hacking of constraints, pushing models away from the main task. Recent attempts to address this resulted in benchmarks with richer constraints, but required expert labor, remained closed-access, and cannot be easily extended. We present Lift-Bench, an IF benchmark that asks a model to draft a document given a topic and a structured outline. Constraints are scoped to certain document sections, providing finer control over constraint density. Lift-Bench includes both verifiable and judged constraints, as well as a training counterpart, Lift-Train. Importantly, Lift-Bench provides both main task and instruction-following constraints, mitigating
    
[^122]: 将长期记忆与大语言模型结合需付出高昂代价：从提示注入到持久性入侵

    Combining Long-Term Memory and LLMs Comes at a High Cost: From Prompt Injection to Persistent Compromise

    [https://arxiv.org/abs/2610.28810](https://arxiv.org/abs/2610.28810)

    本文揭示了现代对话式AI平台的持久记忆功能使提示注入攻击能够跨会话传播（记忆注入），并提出通用威胁模型证明所有NIST攻击目标均可实现，凸显了这一新型安全威胁的严重性与紧迫性。

    

    近年来，生成式人工智能的广泛普及也带来了提示注入和越狱攻击的同步增长。这些攻击手段通过嵌入恶意指令来操纵语言模型，使其绕过预期的行为和操作边界。在传统聊天机器人中，由于状态记忆不会跨对话保留，此类攻击仅限于单个会话内。然而，现代对话界面中的持久记忆功能使得攻击能够跨会话传播，形成了一种被称为“记忆注入”的威胁向量。本工作在领先的大语言模型即服务平台中探讨了这一新兴漏洞，强调了其现实世界的严重性以及采取对策的必要性。现有文献主要关注针对特定平台功能定制的违规行为，而我们提出了一个通用的威胁模型。我们证明了所有NIST攻击目标在ChatGPT等平台上均可共同实现。

    arXiv:2610.28810v1 Announce Type: cross  Abstract: The widespread adoption of generative artificial intelligence in recent years has brought a parallel rise in prompt injection and jailbreak attacks. By embedding malicious instructions, these exploits manipulate language models to bypass intended behavior and operational boundaries. In traditional chatbots, such attacks are confined to a single session because state memory is not preserved across conversations. However, persistent memory features in modern conversational interfaces enable cross-session attack propagation, establishing a threat vector known as memory injection. This work explores this emerging vulnerability in leading LLM-as-a-service platforms, highlighting its real-world severity and the need for countermeasures. While existing literature focuses on violations tailored to specific platform features, we present a general threat model. We demonstrate that all NIST attack goals are collectively achievable across ChatGPT 
    
[^123]: 法律因素在大语言模型中的何处栖身？

    Where Do Legal Factors Live in LLMs?

    [https://arxiv.org/abs/2610.28799](https://arxiv.org/abs/2610.28799)

    该研究首次验证大语言模型内部形成了与全部26个CATO法律因素相对应的内部表征，并通过线性探测和稀疏自编码器在多个开源模型中实现了对这些法律因素的解码与稀疏定位。

    

    在基于案例的法律推理中，法律专业人士会在新的争议中识别具有法律意义的事实模式，并将其与已决案件的事实模式进行比较。在人工智能与法律领域，HYPO维度和CATO因素以符号化方式表示这些模式，将从案例叙事中提炼的专家知识组织为显式、结构化、可解释且可审计的法律概念与关系，从而支持案例比较、法律论证以及基于先例的推理。大语言模型（LLM）正日益被用于法律任务，但其法律知识的内部组织方式仍难以追溯。我们研究LLM是否形成了与全部26个CATO因素相对应的内部表征。两个研究问题分别考察线性可解码性，以及利用稀疏自编码器实现的稀疏定位。实验涵盖三个开源模型——Qwen3-1.7B、Qwen3-8B和Gemma-2 2B，并通过跨模型比较以及对受控自然语言文本的人工审计加以支撑。

    arXiv:2610.28799v1 Announce Type: new  Abstract: In case-based legal reasoning, legal professionals identify legally significant fact patterns in a new dispute and compare them with those of decided cases. In AI & Law, HYPO dimensions and CATO factors represent these patterns symbolically, organizing expert knowledge distilled from case narratives as explicit, structured, interpretable, and auditable legal concepts and relations that support case comparison, legal argument, and precedent-based reasoning. Large language models (LLMs) are increasingly used for legal tasks, yet their internal organization of legal knowledge remains difficult to trace. We investigate whether LLMs form internal representations corresponding to all 26 CATO factors. Two research questions examine linear decodability and sparse localization with sparse autoencoders. Experiments cover three open models, Qwen3-1.7B, Qwen3-8B, and Gemma-2 2B, supported by cross-model comparison and human audit of controlled natur
    
[^124]: CALLIOPE：无需嵌入向量的AI使用私密洞察

    CALLIOPE: Private Insights into AI Use, Without Embeddings

    [https://arxiv.org/abs/2610.28797](https://arxiv.org/abs/2610.28797)

    CALLIOPE是一个支持差分隐私的自顶向下主题层次发现框架，它不依赖向量嵌入聚类，而是在整个流程中直接利用LLM能力，在高隐私保护程度下比基于嵌入的方法表现更好。

    

    我们提出了CALLIOPE，这是一个自顶向下的框架，用于在大型非结构化数据语料库（例如LLM对话）中发现主题层次结构，同时支持差分隐私（DP）。与传统的乃至最近的LLM增强方法不同——这些方法最终都依赖于对向量嵌入进行聚类——我们的框架在整个流程中直接利用LLM的能力。实验表明，在差分隐私设置下，该方法比现有的（基于嵌入的）方法具有更好的效用，尤其是在高隐私保护程度的情况下。我们使用多种传统的聚类指标以及我们提出的一些新颖效用指标来评估CALLIOPE输出的质量。

    arXiv:2610.28797v1 Announce Type: cross  Abstract: We introduce CALLIOPE, a top-down framework for discovering topic hierarchies in large corpora of unstructured data (e.g., LLM conversations), while supporting differential privacy (DP). Unlike traditional and even recent LLM-augmented approaches that ultimately rely on clustering vector embeddings, our framework leverages LLM capabilities directly for the entire pipeline. Experiments demonstrate that this approach provides better utility over existing (embedding-based) methods in the DP setting, especially in the high privacy regime. We evaluate the quality of CALLIOPE outputs using a variety of traditionally studied clustering metrics, as well as some novel utility metrics that we introduce.
    
[^125]: 有人在执行卡特尔吗？强化学习教会了市场中的LLM智能体什么

    Is Anyone Enforcing the Cartel? What Reinforcement Learning Teaches LLM Agents in Markets

    [https://arxiv.org/abs/2610.28787](https://arxiv.org/abs/2610.28787)

    该研究提出一种在反事实市场副本中强制智能体作弊以检验合谋执行机制的审计方法，发现RL训练的LLM定价智能体只会学会训练中易尝试且被奖励的行为（被威慑或跟随降价），而无法自发掌握执行卡特尔所必需的高代价价格战惩罚。

    

    对大型语言模型（LLM）智能体的评估报告称，定价智能体会以牺牲他人利益为代价进行合谋。然而，合谋的定义核心在于执行机制：一个由竞争者分享垄断利润的卡特尔，只有当作弊的成员受到惩罚时才能维持。我们引入了一种审计方法，通过在市场的反事实副本中强制某个成员作弊来检验执行机制，并用它来研究强化学习（RL）教会了定价与交易中的智能体什么。经RL训练的小型LLM学到的是其训练对手所奖励的行为，但仅限于它们容易尝试的行为。在自我对弈中，它们学会的是无视对手的固定规则；面对会施以惩罚的对手，它们学会被威慑；面对作弊的对手，它们学会跟随其降价，却永远学不会那种本可威慑作弊者的高昂代价的价格战——而这种策略它们很少尝试。被强制加入训练中的价格战能够被学会，但大多只在被迫使用的情境下才生效。面对自身的副本时，被威慑的和跟随降价的智能体都能获得相同的卡特尔利润，但只有跟随降价……（原文摘要到此截断）

    arXiv:2610.28787v1 Announce Type: new  Abstract: Evaluations of large language model (LLM) agents report that pricing agents collude at others' expense. Yet collusion is defined by enforcement: a cartel, in which competitors share monopoly profits, holds only if a member who cheats is punished. We introduce an audit that tests enforcement by forcing a member to cheat in counterfactual copies of a market, and use it to study what reinforcement learning (RL) teaches agents in pricing and trading. Small LLMs trained by RL learn what their training rivals reward, but only what they readily try. In self-play, they learn fixed rules ignoring their rivals; against a punishing rival, to be deterred; against a cheating rival, to match its cuts, never the costly price war that would deter it, which they rarely try. A war forced into training is learned, but mostly only where it was forced. Against their own copies, deterred and matching agents can earn the same cartel profits, but only matching 
    
[^126]: 先测量，后分配：强化学习中MCP智能体的校准式Rollout预算分配

    Measure First, Then Allocate: Calibrated Rollout Budgets for MCP Agents in Reinforcement Learning

    [https://arxiv.org/abs/2610.28785](https://arxiv.org/abs/2610.28785)

    提出BRACE方法，利用奖励分歧测量与历史任务证据相结合的相关奖励模型，在分配生成预算前前瞻性预测混合奖励组，显著提升强化学习中MCP工具使用智能体的rollout预算分配效率。

    

    训练使用工具的语言模型需要高昂的交互成本，然而同一rollout组内完全相同的奖励无法提供组间相对的学习信号。我们利用奖励分歧来指导评测框架评估以及在模型上下文协议环境中策略训练的rollout分配。我们首先提出Discord，它采用精确配对推理和保持决策的提前停止来测试评测框架的修改：只有存在分歧的配对结果才能使一个评测框架优于另一个。这一洞察启发了BRACE（基于相关情景的银行校准rollout分配）中的前瞻性分配方法。BRACE结合历史任务证据、任务成功率的不确定性以及相关奖励模型，在分配生成预算之前预测混合奖励组。受控的评测框架实验支持Discord的有效性，而真实MCP实验则揭示了保留与成本之间的权衡。在三个Qwen3-VL模型规模上，BRACE超越了最强基线

    arXiv:2610.28785v1 Announce Type: cross  Abstract: Training tool-using language models requires costly interaction, yet identical rewards within a rollout group provide no group-relative learning signal. We use reward disagreement to guide harness evaluation and the allocation of rollouts for policy training in Model Context Protocol (MCP) environments. We first introduce Discord, which tests harness edits using exact paired inference and decision-preserving early stopping: only disagreeing paired outcomes favor one harness over another. This insight motivates prospective allocation in BRACE (Bank-calibrated Rollout Allocation under Correlated Episodes). BRACE combines historical task evidence, uncertainty in task success rates, and a correlated reward model to forecast mixed-reward groups before allocating generation budget. Controlled harness experiments favor Discord, while live MCP experiments expose a retention-cost trade-off. At three Qwen3-VL model scales, BRACE exceeds the stro
    
[^127]: 扩展心理治疗督导规模：利用大型语言模型评估心理治疗保真度

    Scaling Therapy Supervision: Assessing Psychotherapy Fidelity Using Large Language Models

    [https://arxiv.org/abs/2610.28772](https://arxiv.org/abs/2610.28772)

    该研究利用大型语言模型自动化心理治疗保真度评估并生成可操作的督导反馈，摆脱了人工专家评审的瓶颈，实现了治疗督导的可规模化，并在随机对照试验数据上验证了其区分CBT与非CBT会谈的有效性。

    

    心理治疗保真度评估用于衡量治疗师对治疗方案的遵循程度，这是患者获得积极疗效的关键决定因素。然而，保真度评估一直受限于人工专家评审，使这一过程既费力又难以规模化。为了扩展治疗督导的规模并支持临床医生的自我学习，我们提出了一种大型语言模型（LLM）的集成应用，该应用能够自动化心理治疗保真度评估工作流程，并从真实的临床长篇会谈记录中生成可操作的督导反馈。我们在多种任务分解配置和推理策略下评估了三个开源权重模型，结果表明“证据优先”的推理方式能够提升保真度估计的效果。此外，我们在一项独立的随机对照试验（RCT）上验证了该框架的外部构念效度，在该试验中，该框架能够可靠地区分认知行为疗法（CBT）与非CBT会谈。

    arXiv:2610.28772v1 Announce Type: new  Abstract: Psychotherapy fidelity assessment measures how faithfully a therapist adheres to a treatment protocol, a critical determinant of positive patient outcomes. However, fidelity assessment is bottlenecked by manual expert review, rendering the process labor-intensive and unscalable. To scale therapy supervision and support clinician self-learning, we present an integrated application of large language models (LLMs) that automates psychotherapy fidelity workflows and generates actionable supervisory feedback from authentic, long-form clinical transcripts. We evaluate three open-weight models across multiple task decomposition configurations and reasoning strategies, demonstrating that evidence-first reasoning improves fidelity estimation. Furthermore, we establish the framework's external construct validity on an independent randomized controlled trial (RCT), where it reliably distinguishes Cognitive Behavioral Therapy (CBT) from non-CBT sess
    
[^128]: SkillEval：通过认知诊断模型学习大语言模型的可解释能力画像

    SkillEval: Learning Interpretable Ability Profiles of LLMs via Cognitive Diagnosis Models

    [https://arxiv.org/abs/2610.28767](https://arxiv.org/abs/2610.28767)

    SkillEval 结合大语言模型的元认知能力与心理测量学中的认知诊断模型，从基准题目中自动发现可解释的技能，并为大语言模型生成跨模型的技能级能力画像，相比传统平均分评估更可靠、更校准且信息更丰富。

    

    当前对大语言模型在基准测试上的评估通常将单个样本上的表现汇总为一个标量分数。这种方法过于粗糙，忽略了基准数据集所涵盖的多样化技能范围。为了解决这一局限并更好地利用基准数据，我们提出了 SkillEval，一个全新的评估框架，能够进行跨模型、技能级别的分析。该框架整合了两个领域的进展：（1）利用大语言模型的元认知能力，从题目中自动发现可解释的技能；（2）应用心理测量学中的认知诊断模型（CDMs）来估计大语言模型在每项技能上的能力。我们在五个覆盖不同领域的基准（MATH、BBH、GPQA、MuSR、IFEval）上，对 3,811 个大语言模型在 9,523 个题目上进行了实验。与传统的基于平均分的评估范式相比，SkillEval 产生了可靠、更校准、信息更丰富且简洁的结果。

    arXiv:2610.28767v1 Announce Type: new  Abstract: Current evaluation of LLMs on benchmarks typically aggregates performance over individual examples into a single scalar score. This approach is overly coarse and ignores the diverse scope of skills covered by benchmark datasets. To address this limitation and better exploit benchmark data, we propose SkillEval, a new evaluation framework that performs cross-model, skill-level analysis. The framework integrates advances from two areas: (1) it leverages the meta-cognitive capabilities of LLMs to automatically discover interpretable skills from questions, and (2) it applies cognitive diagnosis models (CDMs) from psychometrics to estimate per-skill abilities of LLMs. We conduct experiments on 3,811 LLMs evaluated on 9,523 items across five benchmarks spanning diverse domains (MATH, BBH, GPQA, MuSR, IFEval). Compared to the traditional average-based evaluation paradigm, SkillEval produces reliable, more calibrated and informative yet concise 
    
[^129]: 面向科学的机器学习自动化：前沿智能体能否在真实环境中攀登科学之峰？

    Automating ML for Science: Can Frontier Agents Climb Scientific Hills in the Wild?

    [https://arxiv.org/abs/2610.28763](https://arxiv.org/abs/2610.28763)

    该论文提出了长时程测试平台 Agent4Sci（涵盖 11 个领域的 20 个科学机器学习基准），发现前沿智能体能在所有基准上超越已发表的最佳模型（Claude Opus 平均相对改进 10.4%），但这些进展几乎完全来自机器学习工程技巧而非研究级的科学洞察。

    

    前沿编码智能体日益被定位为端到端的自动化研究者，而“面向科学的机器学习”为检验这一承诺提供了一个高要求的试金石：改进一个科学模型往往需要超越常规机器学习工程的领域洞察。我们提出了 Agent4Sci，一个长时程测试平台，用于衡量当今的智能体能在多大程度上改进科学机器学习基准上已发表的最佳模型，涵盖 11 个领域的 20 个基准。Agent4Sci 并非通过排行榜为智能体排名，而是探讨智能体在真实条件下能在这些任务上取得哪些类型的进展，以及它们在何处存在不足。实验表明，当今的前沿智能体在所有基准上都推进了已发表的最佳结果，其中 Claude Opus 达到了 10.4% 的平均相对改进，且没有任何一次运行被标记为学术不端。然而，这些提升几乎完全来自一系列机器学习工程技巧，而非研究层面的洞察。轨迹分析揭示了三个反复出现的局限……（摘要原文在此处被截断）

    arXiv:2610.28763v1 Announce Type: cross  Abstract: Frontier coding agents are increasingly framed as end-to-end automated researchers, and ML for science offers a demanding test of that promise: improving a scientific model often requires domain insight beyond routine ML engineering. We introduce Agent4Sci, a long-horizon testbed for measuring how far today's agents can improve the published best models on scientific ML benchmarks, spanning 20 benchmarks across 11 domains. Rather than ranking agents on a leaderboard, Agent4Sci asks what kinds of progress agents can make on these tasks under in-the-wild conditions, and where they fall short. Empirically, today's frontier agents advance the published best on every benchmark, with Claude Opus reaching 10.4% mean relative improvement and no run flagged for academic misconduct. These gains, however, come almost entirely from a bag of ML-engineering tricks rather than research-level insight. Trajectory analysis surfaces three recurring limit
    
[^130]: 逆向戈德定理：基于基底受限对比学习器的正数据语言识别

    A Reverse-Gold Theorem: Positive-Data Identification by Substrate-Bounded Contrastive Learners

    [https://arxiv.org/abs/2610.28747](https://arxiv.org/abs/2610.28747)

    该论文证明了一个“逆向戈德定理”：借助先天递归对比算子自我生成结构受限的负例扰动，并以InfoNCE式对比学习训练的受限评分器，在忠实性条件下可仅凭正数据实现高概率的PAC条件语言识别，且样本复杂度仅为多项式级。

    

    戈德定理表明，任何学习器都无法仅凭正例呈现来识别超有限类中的任意语言。我们证明了一个形态不同的正面结果。固定一个先天的“递归对比算子”$R$，它将任意字符串映射到其结构上受限的有限扰动集合；再考虑一个容量受限的判别式评分器，它通过在自我生成的扰动上进行InfoNCE式对比估计训练。在算子、评分器与目标语言之间满足忠实性条件的前提下，训练后的评分器能够仅凭正数据实现“PAC条件识别”：以高概率，它会把除$\varepsilon$比例之外的所有正例都排在其自身轨道中每个语言外扰动之上，且样本复杂度关于模型容量、轨道大小、分数界限和$1/\varepsilon$呈多项式级。负例由学习器自行构造，而非由外部提供……

    arXiv:2610.28747v1 Announce Type: cross  Abstract: Gold's theorem shows that no learner can identify an arbitrary language in a superfinite class from positive presentations alone. We prove a positive result of a different shape. Fix an innate \emph{recursive contrast operator} $R$ mapping any string to a finite set of structurally bounded perturbations of it, and a discriminative scorer of bounded capacity trained by InfoNCE-style contrastive estimation \citep{vandenoord2018,macollins2018} on self-generated perturbations. Under a faithfulness condition relating operator, scorer, and target, the trained scorer achieves \emph{PAC conditional identification} from positive data alone: with high probability it ranks all but an $\varepsilon$-fraction of positives above every out-of-language perturbation in their own orbit, with sample complexity polynomial in the capacity, the orbit size, the score bound, and $1/\varepsilon$. The negatives are constructed by the learner, not supplied by an 
    
[^131]: Qwen-Audio-3.1-ASR 技术报告

    Qwen-Audio-3.1-ASR Technical Report

    [https://arxiv.org/abs/2610.28711](https://arxiv.org/abs/2610.28711)

    Qwen-Audio-3.1-ASR 系列音频-语言模型在改进语音识别的基础上，扩展了说话人分离转录和通用音频理解能力，支持 30 种语言和 16 种中国方言。

    

    我们推出了 Qwen-Audio-3.1-ASR，这是一系列音频-语言模型，在其前代模型 Qwen-Audio-3.0-ASR 的基础上改进了语音识别能力，并扩展了说话人分离转录和通用音频理解能力。该系列模型保留了对 30 种语言和 16 种中国方言的支持，同时具备指令控制能力，包括历史上下文条件化、分层热词定制以及原生转录润色。说话人分离 ASR 能够生成包含说话人标签、时间戳和识别文本的结构化输出，在多轮对话中保持说话人标签的一致性，并能表示重叠语音。通用音频理解能力涵盖音频、语音和音乐描述生成、时间定位与声音事件检测，以及音频问答和多选题推理。这些任务将应用范围从识别口语内容进一步拓宽。

    arXiv:2610.28711v1 Announce Type: cross  Abstract: We present Qwen-Audio-3.1-ASR, a family of audio-language models that builds upon its predecesor Qwen-Audio-3.0-ASR with improved speech recognition and expanded capabilities in speaker-diarized transcription and general audio understanding. The family retains support for 30 languages and 16 Chinese dialects, together with instruction-controlled capabilities including historical-context conditioning, hierarchical hotword customization, and native transcription polishing. Speaker-diarized ASR produces structured outputs containing speaker labels, timestamps, and recognized text, maintaining consistent speaker labels across turns and the representation of overlapping speech. The general audio understanding capabilities cover audio-, speech-, and music-captioning, temporal grounding and sound event detection, as well as audio question answering and multiple-choice reasoning. These tasks broaden the application scope from recognizing spoke
    
[^132]: 翻译与数据伦理

    Translation and data ethics

    [https://arxiv.org/abs/2610.28710](https://arxiv.org/abs/2610.28710)

    本章系统介绍了翻译领域的数据伦理，涵盖版权、所有权、偏见等普遍性问题，以及少数语言社区和数字鸿沟相关的特殊伦理问题，并探讨了责任、监管及翻译学界对数据伦理日益增长的关注。

    

    本章介绍翻译领域的数据伦理，并讨论了与数据相关的一系列伦理问题。这些问题包括版权、所有权和偏见等，适用于所有类型的数据；而对于少数语言社区以及处于数字鸿沟两侧的人群，还存在其他特殊问题，因为数据与权力失衡和权力动态密切相关。最后，我们讨论了责任与监管问题，并反思了翻译学研究者对数据伦理日益增长的关注。

    arXiv:2610.28710v1 Announce Type: new  Abstract: This chapter introduces data ethics for translation and discusses a number of ethical issues regarding data. These include issues of copyright, ownership and bias, which apply to all types of data, whereas there are further issues particular to minoritised language communities and those on either side of the digital divide, as data is implicated in power imbalances and dynamics. We close with a discussion of responsibility and regulation, and reflect on the growing attention on data ethics from translation studies researchers.
    
[^133]: 不改变任何权重的 Gemma 4 语音编码器能走多远？基于冻结编码器与适配器增强编码器的沙特阿拉伯语和海湾阿拉伯语语音识别

    How Far Can the Gemma 4 Speech Encoder Go Without Changing Its Weights? Saudi and Gulf Arabic Speech Recognition with a Frozen and an Adapter-Augmented Encoder

    [https://arxiv.org/abs/2610.28673](https://arxiv.org/abs/2610.28673)

    在完全不改变 Gemma 4 语音编码器权重的前提下，仅通过轻量级 CTC 读取器或少量低秩适配器参数，就能让沙特和海湾阿拉伯语语音识别达到与参数量大得多的商业系统统计上相当的性能，同时在 CPU 上的推理速度快 3.4 倍。

    

    我们研究了 Gemma 4 与 EmbeddingGemma 2 共享的 3.05 亿参数音频编码器（两者除最终投影层外完全相同），在不改变其任何权重的情况下将其用于沙特阿拉伯语和海湾阿拉伯语语音识别。冻结系统仅在编码器多个层特征的混合之上训练一个 2260 万参数的 CTC 读取器；适配增强系统则额外增加 310 万个低秩适配器参数、一个 4150 万参数的混合 CTC/注意力读取器，以及与子词语言模型的联合解码。与在相同音频上重新运行的 Cohere Transcribe Arabic（20.7 亿参数）相比，冻结系统在词错误率（WER）上略逊一筹，但在六个人工标注测试条件中的五个上，字符错误率（CER）在统计上不相上下，同时在使用贪婪解码时在 CPU 上的运行速度快 3.4 倍。适配增强系统在公开的沙特客服通话数据上的 WER（19.20% 对 19.27%）以及在干净的阿联酋语音上与对方在统计上不相上下，而在电话编码的阿联酋语音上则显著更优……（原文摘要在此处被截断）

    arXiv:2610.28673v1 Announce Type: new  Abstract: We study the 305M-parameter audio encoder shared by Gemma 4 and EmbeddingGemma 2 (identical except for its final projection) for Saudi and Gulf Arabic speech recognition, without changing any of its weights. A frozen system trains only a 22.6M-parameter CTC reader over a mix of encoder layers; an adapted system adds 3.1M low-rank adapter parameters, a 41.5M-parameter hybrid CTC/attention reader and joint decoding with a sub-word language model. Against Cohere Transcribe Arabic (2.07B parameters), re-run on identical audio, the frozen system trails in word error rate (WER) but is statistically indistinguishable in character error rate (CER) on five of six human-labelled conditions, while running 3.4 times faster on CPU with greedy decoding. The adapted system is statistically indistinguishable in WER on a public Saudi customer-service call (19.20% against 19.27%) and on clean Emirati speech, is significantly better on telephone-coded Emir
    
[^134]: 超越来源归因：LLM构建的学术知识图谱中的地址编制与可比性决策

    Beyond source attribution: Address authorship and comparability decisions in an LLM-built scholarly knowledge graph

    [https://arxiv.org/abs/2610.28663](https://arxiv.org/abs/2610.28663)

    本文针对LLM构建的学术知识图谱提出，必须将用于汇集证据的稳定“工作地址”与文献中相互矛盾的分类类别区分开来，否则检索地址会被误当作唯一分类，导致证据在边界、等级和诊断标准上不可比（以汉语方言学为例）。

    

    知识图谱需要为其证据提供一个稳定的地址，即使文献中提供的分类相互竞争且无一定论。然而，为检索而选定的地址很容易被误当作唯一可用的分类。每当以同一名称汇集的记录涉及在边界、等级或描述与计数标准上各不相同的单位时，这种情况就会发生。在汉语方言学中，相同的地点在不同方案中反复出现，而这些方案的单位在成员构成和等级上各不相同；当一个单位的诊断证据发生变化时，其归属也可能随之改变。两个声调计数即使各自都被忠实引用，仍可能在统计不同的东西。因此，整个领域的分类学便建立在一份自相矛盾的文献的不稳定框架之上。无论是核实引文还是检索冲突的数值，都无法发现这些差异。我区分了汇集候选证据的工作地址、来源……（原文摘要在此处截断）

    arXiv:2610.28663v1 Announce Type: cross  Abstract: A knowledge graph needs a stable address for its evidence even where the literature offers competing classifications and agrees on none. Yet an address chosen for retrieval readily passes for the only classification available. This happens whenever records gathered under one name concern units that differ in boundaries, rank, or the criteria by which they are described or counted. In Sinitic dialectology the same localities recur in schemes whose units differ in membership and rank, and a unit may change its affiliation when the evidence for its diagnosis changes. Two tone counts, each quoted faithfully, may still count different things. The taxonomy of an entire field thus comes to rest on the unstable frame of a literature that contradicts itself. Neither verifying a quotation nor searching for conflicting values detects these differences. I distinguish the working address that gathers candidate evidence, the category that the source
    
[^135]: TAGR：面向异构城市配送任务的任务自适应地理语义检索

    TAGR: Task-Adaptive Geo-Semantic Retrieval across Heterogeneous Urban Delivery Tasks

    [https://arxiv.org/abs/2610.28638](https://arxiv.org/abs/2610.28638)

    提出TAGR框架，通过任务自适应的地理语义检索机制，为城市配送中需求预测、ETA预测和下一站预测三类异构任务检索结构与粒度各异的历史证据，解决单一融合机制导致的匹配不当与负迁移问题，提升预测精度与资源分配效率。

    

    随着城市居民对高质量即时配送服务需求的不断提升，需求预测、预计到达时间（ETA）预测和下一站预测已成为城市配送系统中的关键任务。然而，这三项任务所依赖的历史证据在结构和粒度上存在显著差异。同时，它们的预测目标在输出形式上也各不相同，这使得难以通过单一的融合机制来整合检索到的证据，否则会造成匹配不当或负迁移。这些问题会增加预测误差，进而导致资源配置失衡——部分地区的需求得不到满足，而另一些地区则出现资源利用率不足的情况，最终降低整体效率和服务质量。为应对这些挑战，我们提出了TAGR，一个面向异构城市配送任务的任务自适应地理语义检索框架。TAGR构建了一个基于大语言模型（LLM）的地理语义共享……（原文摘要在此处截断）

    arXiv:2610.28638v1 Announce Type: cross  Abstract: As urban residents demand higher-quality instant delivery services, demand forecasting, estimated time of arrival prediction, and next-stop prediction have become critical tasks in urban delivery systems. However, the three tasks rely on historical evidence with substantially different structures and granularities. Meanwhile, their prediction targets differ in output form, making it difficult to integrate retrieved evidence through a single fusion mechanism without causing mismatch or negative transfer. These issues can increase prediction errors and subsequently lead to imbalanced resource allocation, where some regions experience unmet demand. In contrast, others suffer from underutilization, ultimately degrading overall efficiency and service quality. To address these challenges, we propose TAGR, a Task-Adaptive Geo-Semantic Retrieval framework across heterogeneous urban delivery tasks. TAGR constructs an LLM-based geo-semantic shar
    
[^136]: ResearchHub：面向生物医学预印本的个性化文献发现系统

    RESEARCHHUB: A Personalized Literature Discovery System for Biomedical Preprints

    [https://arxiv.org/abs/2610.28636](https://arxiv.org/abs/2610.28636)

    本文提出ResearchHub个性化文献发现系统，其核心检索器prism通过GPT-4o生成查询和混合训练目标适配生物医学预印本，在MedRxiv语料上将Recall@10较SPECTER2-Search提升12.4个百分点，并支持可控新颖性、多样性与课程式排序。

    

    我们提出了ResearchHub，一个面向生物医学预印本的交互式个性化文献发现系统。给定研究人员画像或自由格式的查询，ResearchHub从特定领域的MedRxiv语料库中检索相关论文，并通过可控的新颖性、多样性和课程式排序，帮助用户超越近似重复的推荐。系统的核心是prism，一个针对生物医学预印本适配的双编码器检索器，它使用GPT-4o生成的自然语言查询，以及结合交叉编码器软标签的混合式多负例排序加MarginMSE训练目标。在2023至2025年的44,937篇MedRxiv论文上，prism超越了强大的现成密集检索器，相比SPECTER2-Search将Recall@10提升了12.4个百分点。ResearchHub还进一步支持基于用户阅读历史的前沿检索、基于行列式点过程（DPP）的多样性重排序，以及从主流到专业领域的局部内在维度排序。

    arXiv:2610.28636v1 Announce Type: cross  Abstract: We present ResearchHub, an interactive system for personalized literature discovery over biomedical preprints. Given a researcher profile or a free-form query, ResearchHub retrieves relevant papers from a domain-specific MedRxiv corpus and helps users move beyond near-duplicate recommendations through controllable novelty, diversity, and curriculum ordering. At the core of the system is prism, a bi-encoder retriever adapted to biomedical preprints using GPT-4o-generated natural language queries and a hybrid Multiple Negatives Ranking plus MarginMSE objective with cross-encoder soft labels. On 44,937 MedRxiv papers from 2023--2025, prism outperforms strong off-the-shelf dense retrievers, improving Recall@10 over SPECTER2-Search by 12.4 percentage points. ResearchHub further supports frontier retrieval based on a user's reading history, DPP-based diversity re-ranking, Local Intrinsic Dimensionality ordering from mainstream to specialized
    
[^137]: 超越模型权重：扩散语言模型中依赖采样的公平性

    Beyond Model Weights: Sampling-Dependent Fairness in Diffusion Language Models

    [https://arxiv.org/abs/2610.27584](https://arxiv.org/abs/2610.27584)

    扩散语言模型的社会偏见程度及模型间的偏见排序会随采样配置（如块长度和重掩码策略）显著变化，且即使采用去偏见策略后，这种采样依赖性依然持续存在。

    

    扩散语言模型（DLM）正成为自回归语言模型（ARLM）的实用替代方案，但其公平性在很大程度上尚未得到系统表征。扩散语言模型引入了在自回归模型中没有直接对应物的采样控制参数，例如块长度和重掩码策略，这些参数直接决定了模型的速度-质量工作点。这引出一个基本问题：扩散语言模型所测得的社会偏见在这些不同设置下是否保持稳定？通过在 Open-BBQ 数据集上对 LLaDA、LLaDA-1.5 和 Dream 三个模型进行测试，我们发现这三个模型均表现出社会偏见，但所测偏见的幅度以及按偏见程度对模型进行的相对排序，都对采样配置高度敏感。这种采样依赖性也改变了“成功去偏见”的含义。虽然为自回归语言模型建立的偏见缓解策略可以有效地迁移到扩散语言模型，并在不损害下游性能的前提下减少偏见，但即使在缓解之后，采样依赖性依然存在。

    arXiv:2610.27584v1 Announce Type: new  Abstract: Diffusion language models (DLMs) are emerging as a practical alternative to autoregressive language models (ARLMs), yet their fairness remains largely uncharacterized. DLMs introduce sampling controls with no direct autoregressive counterpart, such as block length and remasking strategy, that directly set their speed--quality operating point. This raises a basic question: does a DLM's measured social bias remain stable across these settings? Across LLaDA, LLaDA-1.5, and Dream on Open-BBQ, we find that all three models exhibit social bias, but both the magnitude of measured bias and the relative ordering of models by bias are highly sensitive to the sampling configuration. This sampling dependence also changes what successful debiasing means. While mitigation strategies established for ARLMs transfer effectively to DLMs and reduce bias without degrading downstream performance, sampling dependence persists even after mitigation. Our result
    
[^138]: 无需关系信息即可预测线性关系可解码性

    Predicting Linear Relational Decodability Without the Relation

    [https://arxiv.org/abs/2610.27241](https://arxiv.org/abs/2610.27241)

    该论文提出“实体透镜”这一新颖的雅可比透镜变体，通过在与关系无关的中性提示下针对实体类型进行训练，能够在不使用任何关系特定信息的情况下，显著预测线性关系嵌入（LRE）在不同关系上忠实度的巨大差异。

    

    线性关系嵌入（Linear Relational Embeddings, LREs）使用仿射函数来近似语言模型从主体到客体的映射。其忠实度在不同关系之间差异巨大：例如 person_father（某人的父亲）关系的忠实度接近于零，而 country_capital_city（国家首都城市）关系则达到上限。我们研究了导致这种差异的因素，并证明这种差异可以从一个不了解底层关系的读出（readout）中被显著预测出来。为此，我们提出了一种新颖的雅可比透镜（Jacobian lens）形式——实体透镜（entity lens）。雅可比透镜是在所有词元和概念上进行估计的，而实体透镜 EL_t 则被训练用于预测模型在中性提示词"{} is known for"（“{} 以……闻名”）下对类型为 t 的实体会输出什么；例如，我们分别为国家和人物类型训练了实体透镜。在估计出实体透镜之后，我们测量它在多大程度上将模型对某个关系的答案排在高位，而在整个估计和预测过程中不使用任何关系特定的信息。在28个关系……（摘要原文在此处被截断）

    arXiv:2610.27241v1 Announce Type: new  Abstract: Linear relational embeddings (LREs) approximate a language model's subject-to-object mapping with an affine function. Their faithfulness varies enormously across relations, from near zero for person_father to the ceiling for country_capital_city. We investigate the factors that contribute to this variation and demonstrate that it can be significantly predicted from a readout that is not aware of the underlying relation. To do so, we propose a novel form of the Jacobian lens, the entity lens. Whereas the Jacobian lens is estimated across all tokens and concepts, the entity lens EL_t is trained to predict what the model will output for an entity of type t under the neutral prompt "{} is known for"; e.g., we train ELs for countries and persons. After estimating an entity lens, we measure how highly it ranks the model's answer for a relation without using any relation-specific information in estimation and prediction. Across 28 relations of 
    
[^139]: WaveCommitments：私密且可验证的语音智能体市场

    WaveCommitments: Private and Verifiable Speech-Agent Marketplaces

    [https://arxiv.org/abs/2610.26867](https://arxiv.org/abs/2610.26867)

    提出WaveCommitments，一种将可信执行环境内基于Data Shapley的机密资产估值与多元多项式承诺相结合的端到端设计，实现了语音智能体市场中私密且可验证的资产交易，相比最佳基准速度提升最高20倍、证明大小缩减最高39倍。

    

    现代语音系统正在演变为由自主语音智能体组成的生态系统，这些智能体协同发声、协商并交易诸如嵌入向量、频谱图或原始数据等资产。在这类市场中，出于隐私约束，卖家无法公开资产供评估，而买家也无法为其无法测试的东西定价。因此，我们开源了WaveCommitments，这是一种端到端设计，将可信执行环境内基于Data Shapley的机密资产估值与多元多项式承诺相结合。它支持选择性公开和防篡改溯源，能够证明交付的资产与估值相符，同时隐藏其他所有信息。我们在三个知名数据集上，使用包括Whisper在内的最先进编码器，针对多种任务进行了大量实验，结果表明我们的方法将最佳基准的速度提升了最高20倍，同时将证明大小缩减了最高39倍。我们还系统地研究了说话人的价值如何随任务变化而偏移。

    arXiv:2610.26867v1 Announce Type: new  Abstract: Modern speech systems are evolving into ecosystems of autonomous speech agents that speak together, negotiate, and trade assets such as embeddings, spectrograms, or raw data. In such markets, sellers cannot reveal assets for evaluation due to privacy constraints, and buyers cannot price what they cannot test. Thus, we open-source WaveCommitments, an end-to-end design that pairs confidential asset valuation via Data Shapley inside a trusted execution environment with multivariate polynomial commitments. It enables selective openings and tamper-evident provenance to prove that a delivered asset matches the valuation while hiding everything else. Our extensive experiments across three known datasets and various tasks with state-of-the-art encoders, including Whisper, reveal that we speed up the best benchmarks by up to 20x while reducing their proof sizes by up to 39x. We also systematically study how a speaker's value shifts across tasks a
    
[^140]: AfriBED：面向低资源非洲语言的语音识别偏见评估数据集

    AfriBED: a Speech Recognition Bias Evaluation Dataset for Low Resource African Languages

    [https://arxiv.org/abs/2610.26821](https://arxiv.org/abs/2610.26821)

    本文提出了面向本巴语和尼扬贾语两种低资源非洲语言的全新语音识别偏见评估数据集AfriBED，可用于检测和估计语音识别系统中的性别偏见与说话人类型偏见。

    

    尽管自动语音识别（ASR）系统不断进步，现有系统仍存在说话人属性偏见，即其语音识别性能会因性别、口音和说话人类型等人口统计属性而对某些说话人群体表现出偏见。然而，要准确评估给定ASR系统中的说话人属性偏见，需要拥有一个在说话人属性各子群体之间具有可比语音的偏见评估数据集。本文介绍了非洲语音偏见评估数据集，这是一个面向两种低资源非洲语言——本巴语和尼扬贾语——的新型语音识别偏见评估数据集，用于检测和估计ASR系统的性别偏见和说话人类型偏见。该数据集包含本巴语的9小时语音，由80名说话人录制的3928条语句；尼扬贾语数据集则包含127名说话人录制的6260条语句，约18小时语音。结果表明……（原文在此处截断）

    arXiv:2610.26821v2 Announce Type: replace  Abstract: Despite advancements in automatic speech recognition systems (ASR), current systems exhibit speaker-attribute bias, i.e., their speech recognition performance is biased against certain speakers based on their demographic attributes such as gender, accents, and speaker type. However, accurately assessing speaker attribute bias in a given ASR system requires the availability of a bias evaluation dataset with comparable speech across subgroups of a speaker attribute. In this paper, we introduce African speech Bias Evaluation Dataset (AfriBED), a novel speech recognition bias evaluation dataset for two low resource African languages, Bemba and Nyanja, to detect and estimate gender and speaker-type bias for ASR systems. The dataset comprises 9 hours of speech with 3928 utterances recorded by 80 speakers for Bemba, while the Nyanja set includes 6260 utterances with approximately 18 hours of speech recorded by 127 speakers. The results of o
    
[^141]: 重复条件下的复制：从前缀匹配到位置哈希

    Copying Under Repetition: From Prefix Matching to Position Hashing

    [https://arxiv.org/abs/2610.26722](https://arxiv.org/abs/2610.26722)

    本文提出利用小素数周期的VO旋转位置编码并结合中国剩余定理，证明了两层Transformer能在任意重复字符串下实现精确复制并正确终止，同时给出了顺序有限精度下复制的不可能性证明。

    

    前沿大语言模型（LLM）几乎能够完美复制随机文本，但在处理长段重复字符时却常常出现过度复制或截断的问题。在重复字符串中，匹配局部内容不再能可靠地判断复制应从何处继续或何时停止。然而，我们证明了一个具有 $O(\log\log N)$ 位精度和 $O(\log N/\log\log N)$ 宽度的两层 Transformer 能够复制直到设计上限 $N$ 的任意字符串，并在任意重复情况下正确终止。我们采用具有小素数周期的 RoPE 风格的注意力值与输出旋转（VO-旋转），以对这些素数取模的方式编码到分隔符锚点的距离。根据中国剩余定理，这些周期性编码共同区分了所有所需的距离，使得单一检索规则即可选择下一个源 token 或结束标记。相比之下，我们证明了在顺序有限精度下复制的一个与宽度无关的不可能性结果。

    arXiv:2610.26722v2 Announce Type: replace-cross  Abstract: Frontier LLMs can copy random text almost flawlessly, yet they often overrun or truncate long blocks of repeated characters. In repeated strings, matching local content no longer reliably identifies where copying should continue or when it should stop. Nevertheless, we show that a two-layer transformer with $O(\log\log N)$-bit precision and $O(\log N/\log\log N)$ width can copy every string up to a chosen design horizon $N$ and terminate correctly under arbitrary repetition. We use RoPE-style rotations of attention values and outputs (VO-rotations) with small prime periods to encode distances from delimiter anchors modulo those primes. By the Chinese Remainder Theorem, these periodic encodings jointly distinguish all required distances, allowing one retrieval rule to select either the next source token or the end marker. In contrast, we prove a width-independent impossibility result for copying under sequential finite-precision
    
[^142]: 突破扩散：基于草稿精炼的并行推理

    Breaking Diffusion: Parallel Reasoning with Draft Refinement

    [https://arxiv.org/abs/2610.26542](https://arxiv.org/abs/2610.26542)

    该论文提出草稿精炼语言模型（DRLMs），利用单个去噪器并行提出并评估草稿修订，并通过校正自身token证据来衡量token所获得的上下文支持，从而突破传统扩散模型迭代去噪无法保证整体答案变优的局限。

    

    草拟答案使推理模型能够在其他决策的背景下重新审视每个决策。然而，反复的扩散去噪并不能确保各个单独看似合理的token更新共同形成一个更好的答案。我们提出了草稿精炼语言模型（DRLMs），它使用单个去噪器来提出并行修订并评估候选草稿。我们训练该去噪器从均匀损坏的草稿中重建每个token，同时教会它保留和修复。其预测直接指定修订内容，而无需扩散反向转移过程。评估则需要以另一种方式使用这些预测：对已有token的置信度既反映了上下文支持，也包含了观察该token本身所得的证据。利用已知的损坏过程，我们校正这种自身token证据，以估计每个token在多大程度上受到提示和其余草稿的支持。汇总这些估计可得到一个空腔熵……（原文在此处截断）

    arXiv:2610.26542v1 Announce Type: cross  Abstract: Drafting an answer allows a reasoning model to reconsider each decision in the context of the others. Yet repeated diffusion denoising does not ensure that individually plausible token updates form a better answer together. We introduce Draft Refinement Language Models (DRLMs), which use a single denoiser to propose parallel revisions and evaluate a candidate draft. We train the denoiser to reconstruct every token from uniformly corrupted drafts, teaching both preservation and repair. Its predictions specify revisions directly, without a diffusion reverse transition. Evaluation requires a different use of these predictions: confidence in an existing token reflects both contextual support and evidence from observing the token itself. Using the known corruption process, we correct for this self-token evidence to estimate how well each token is supported by the prompt and the remaining draft. Aggregating these estimates yields a cavity en
    
[^143]: 通过演化惩罚缓解偏好奖励模型中的奖励作弊问题

    Mitigating Reward Hacking in Preference Reward Models with Evolving Penalties

    [https://arxiv.org/abs/2610.26496](https://arxiv.org/abs/2610.26496)

    提出Dynalize方法，通过让语言模型在训练过程中异步分析采样结果并动态生成针对不良行为的演化惩罚，在线缓解偏好奖励模型的奖励作弊问题，从而支持更持久的训练并提升开放式任务的下游性能。

    

    语言模型的强化学习已经从偏好奖励模型转向可验证奖励，后者能够支持更长时间的优化。偏好模型容易出现奖励作弊现象：策略在不提升回复质量的情况下提高奖励分数，甚至在降低质量的同时提高分数。然而对于开放式任务——其质量既无法验证，也无法完全写成明确的评分标准——偏好模型仍然具有重要价值。我们提出Dynalize，它在策略训练期间在线缓解奖励作弊问题。一个语言模型周期性地、异步地分析训练采样结果，并为奖励模型似乎会奖励的不良行为编写惩罚项；一个评判器将这些惩罚应用到每一次采样中。在临床咨询和创意写作任务中，仅基于奖励模型进行训练虽然会使其分数持续上升，但最终导致外部基准性能下降；而Dynalize能够支持更长时间的训练，并带来下游性能的提升。

    arXiv:2610.26496v2 Announce Type: replace  Abstract: Reinforcement learning for language models has moved from preference reward models to verifiable rewards, which sustain much longer optimization. Preference models are prone to reward hacking: the policy raises their score without improving response quality, or even while degrading it. Yet for open-ended tasks, where quality can neither be verified nor fully written down as rubrics, they remain valuable. We introduce Dynalize, which mitigates reward hacking online, during policy training. Periodically and asynchronously, a language model analyzes training rollouts and writes penalties for undesirable behaviors that the reward model appears to reward; a judge applies them to every rollout. In clinical counseling and creative writing, training on the reward model alone raises its score continuously but ultimately leads to a decline in external benchmarks; Dynalize sustains longer training and results in improved downstream performance.
    
[^144]: DiaLong：长上下文对话中记忆回溯能力的基准测试

    DiaLong: Benchmarking Reminiscence in Long-Context Dialogues

    [https://arxiv.org/abs/2610.25463](https://arxiv.org/abs/2610.25463)

    提出DiaLong基准，包含100个超长人类对话及事实回忆判别任务，用以定量评估大语言模型在长上下文对话中主动保持和检索信息的能力，测试表明现有模型及记忆增强方法仍远未达标。

    

    基于Transformer的大语言模型（LLM）在长对话上下文中保持和回忆记忆的能力究竟如何？众所周知，LLM会遗忘其上下文中的长距离事实信息，然而该领域一直缺乏一个能够定量评估这些缺陷的综合性基准。我们提出了DiaLong，这是一个长上下文对话数据集及配套的记忆任务，用于将该领域的评估标准化。该数据集由100个由人类创建、随后借助LLM进行增强的独特对话组成，其长度超越了现有数据集中的对话长度。我们选择的记忆任务是事实回忆：将关于某段对话的40个候选事实判断为真或假。该任务直接探测模型主动保持和检索信息的能力，是衡量扩展对话上下文中记忆能力的关键指标。通过在DiaLong上测试最新的LLM以及已提出的记忆增强方法，我们证明了该领域仍有大量工作需要完成。

    arXiv:2610.25463v2 Announce Type: replace  Abstract: How well can transformer-based Large Language Models (LLMs) retain and recall memories in long contexts of dialogue? LLMs are known to forget long-range factual information in their context, yet the field lacks a comprehensive benchmark to quantitatively assess these shortcomings. We present DiaLong, a long-context dialogue dataset and accompanying memory task that standardizes evaluation in this space. It consists of 100 unique conversations created by humans and then augmented with LLMs, extending beyond lengths found in existing datasets. Our selected memory task is fact recall: classify 40 candidate facts about a dialogue as true or false. This task directly probes the model's ability to actively retain and retrieve information, serving as a critical measure of memory within extended conversational contexts. In testing both the latest LLMs and proposed memory enhancements on DiaLong, we demonstrate that more work needs to be done
    
[^145]: Acharya 2.0：面向印地语自主学习科目的AI导师

    Acharya 2.0: An AI Tutor for Self-Learning Subjects in Hindi

    [https://arxiv.org/abs/2610.25192](https://arxiv.org/abs/2610.25192)

    该论文提出了Acharya 2.0，一个基于检索增强生成（RAG）技术的印地语自主学习AI导师系统，旨在克服低资源语言标注数据匮乏和大语言模型适配成本高昂的问题。

    

    文本处理对于低资源语言至关重要，因为它能够开发出弥合数字鸿沟、提供公平信息与服务获取渠道的工具和技术。通过为低资源语言提供强大的文本处理能力，我们可以在快速数字化的世界中保护语言多样性、提升教育水平、改善交流并促进包容性。然而，由于缺乏足够的标注数据集和语言资源，为这些低资源语言开发准确可靠的语言技术面临巨大挑战。大语言模型（LLMs）因其出色的性能和产生的价值而备受关注。但随着语言的不断演变，将大语言模型适配到特定任务是耗时且资源密集型的。因此，检索增强生成（RAG）通过为大语言模型补充外部知识克服了这一限制，从而……

    arXiv:2610.25192v2 Announce Type: replace  Abstract: Text processing is crucial for low-resource languages as it enables the development of tools and technologies that bridge the digital divide and provide equitable access to information and services. By empowering low-resource languages with robust text processing, we can preserve linguistic diversity, enhance education, improve communi- cation, and foster inclusivity in a rapidly digitizing world. Due to the lack of sufficient annotated datasets and lin- guistic resources, it is challenging to develop accurate and reliable language technologies for these low-resource languages. Large Language Models (LLMs) have attracted sig- nificant attention with their performance and the value they produce. However, with the ever-evolving language, adapting LLMs for a specific task is time-consuming and resource-intensive. Thus, Retrieval-Augmented Genera- tion (RAG) has overcome this limitation by augmenting LLMs with external knowledge, thereby
    
[^146]: HybridResearcher：为深度研究智能体打通结构化与非结构化数据的边界

    HybridResearcher: Bridging the Structured-Unstructured Boundary for Deep Research Agents

    [https://arxiv.org/abs/2610.24899](https://arxiv.org/abs/2610.24899)

    提出 HybridResearcher——一个 4B 参数的深度研究智能体，通过从已验证的单源任务合成跨来源桥接轨迹来训练，使其学会在结构化数据与非结构化数据的边界之间进行跨源证据推理与整合。

    

    arXiv:2610.24899v2 公告类型：replace-cross 摘要：尽管大语言模型（LLM）智能体越来越擅长从单一同质来源（如结构化表格或非结构化数据）中收集和综合证据，但现实世界中的分析问题往往跨越异构来源。例如，回答一个企业问题可能既需要基于内部数据库计算的结果，也需要在网络上找到的证据。最近在 HybridDeepResearch 等基准上的评估表明，即使是前沿智能体在证据需要跨越这一边界时也会表现不佳。虽然这些基准已经暴露了这一局限，但目前仍然缺乏明确教授跨来源推理的方法。为了解决这一问题，我们提出了 HybridResearcher，一个经过训练、能够在结构化与非结构化数据边界之间搭建证据桥梁的 4B 智能体。为了构建它，我们从经过验证的单源任务（如现有的文本到 SQL 或网络搜索查询）出发，人工合成跨来源的桥接轨迹……（原文摘要在此处被截断）

    arXiv:2610.24899v2 Announce Type: replace-cross  Abstract: While LLM agents are increasingly capable of gathering and synthesizing evidence from a single homogeneous source, such as structured tables or unstructured data, real-world analytical questions frequently span heterogeneous sources. For example, answering one enterprise question may require both a result computed over an internal database and evidence found on the web. Recent evaluations on benchmarks like HybridDeepResearch show that even frontier agents struggle when evidence must cross this boundary. While these benchmarks have exposed the limitation, methodologies to explicitly teach cross-source reasoning remain absent. To address this, we introduce HybridResearcher, a 4B agent trained to bridge evidence across the structured and unstructured boundary. To build it, we synthesize cross-source bridge trajectories by starting with validated single-source tasks-such as existing text-to-SQL or web search queries-and fabricatin
    
[^147]: SWECraft世界：通过模拟软件世界实现大规模智能体协作

    World of SWECraft: Agent Collaboration at Scale via Simulated Software Worlds

    [https://arxiv.org/abs/2610.24061](https://arxiv.org/abs/2610.24061)

    提出Software World（SWECraft）框架，通过模拟由真实Python软件包构成的软件生态世界来研究多智能体协作，以隐藏依赖包的加速作为统一评估指标，系统考察通信、所有权和资源分配等要素对协作效果与可扩展性的影响。

    

    随着AI智能体团队承担越来越大的项目和长期悬而未决的开放性问题，智能体协作为扩展推理时计算提供了另一个维度。为了理解什么使这种协作有效且可扩展，我们开发了Software World，一个用于在真实软件生态系统中模拟多智能体协同工作的框架。该框架附带一个流水线，可以创建由真实的、相互依赖的、具有不同主题、规模和依赖结构的Python包组成的环境，我们称之为“软件世界”。在模拟中，智能体维护世界中的软件包，其高层目标是提高软件效率，它们可以自由地发现要处理的任务并与其他智能体协作。我们通过测量隐藏依赖包中的加速比来整体评估智能体在一个世界中的工作成效。该框架使我们能够通过改变通信方式、所有权和资源分配来研究协作的不同要素。我们运行模拟

    arXiv:2610.24061v2 Announce Type: replace-cross  Abstract: As teams of AI agents take on larger projects and long-standing open problems, agent collaboration offers another dimension for scaling inference-time compute. To understand what makes such collaboration effective and scalable, we develop Software World, a framework for simulating multi-agent coworking in realistic software ecosystems. It comes with a pipeline to create environments of real, interdependent Python packages of different topics, sizes, and dependency structures, which we call software worlds. In a simulation, agents maintain packages in the world with a high-level goal of improving software efficiency, and they are free to discover tasks to work on and collaborate with others. We evaluate agents' work in a world collectively by measuring speedup in hidden dependent packages. The framework allows us to study different elements of collaboration by varying communication, ownership, and resource allocation. We run sim
    
[^148]: ProPerAgent：迈向记忆驱动的主动性助手

    ProPerAgent: Toward Memory-Driven Proactive Assistance

    [https://arxiv.org/abs/2610.22671](https://arxiv.org/abs/2610.22671)

    该论文提出ProPerMem基准，首次要求AI助手从长达一年、充满干扰的个人记忆中自主挖掘证据来决定是否及何时主动介入，发现现有模型（含前沿模型）表现极差（最高仅15%），而提供现成证据时性能可达96%，揭示了记忆证据挖掘是主动性助手的核心瓶颈。

    

    随着助手从被动应答转向主动干预，记忆的角色发生了变化：它不再仅仅为查询提供答案，而是决定了系统是否以及何时该开口说话。以往在这一设定下的工作是把相关事实直接放进上下文中交给系统；我们研究的是使问题真正困难的情形，即证明干预合理性的证据必须从长时段的个人记忆中挖掘出来。我们提出了ProPerMem，一个面向记忆驱动主动性助手（MPA）的基准，它将第一人称视角视频与长达一年、充满干扰信息的用户记忆（平均约618K token）、经人工验证的触发窗口，以及一个对内容与时机进行联合评分的协议相配对。我们评估的所有基线模型表现都很差（最高仅15%），包括在给定摘要式或检索增强记忆的情况下的前沿模型。一项受控研究解释了原因：当神谕证据在触发窗口内直接提供时，GPT-5.3-Chat能达到96%；而让它自行选择证据时性能则大幅下降

    arXiv:2610.22671v1 Announce Type: new  Abstract: As assistants move from reactive answering toward proactive intervention, memory changes role: it no longer only supplies the answer to a query, it determines whether and when to speak at all. Prior work in this setting hands the system the relevant fact in context; we study the regime that makes the problem hard, where the evidence justifying an intervention must be mined from a long-horizon personal memory. We introduce ProPerMem, a benchmark for memory-driven proactive assistance (MPA), pairing egocentric video with year-long, distractor-rich user memories (~618K tokens on average), human-verified trigger windows, and a protocol that scores content and timing jointly. Every baseline we evaluate performs poorly (at most 15%), including frontier models given summarized or retrieval-augmented memory. A controlled study explains why: with oracle evidence delivered inside the trigger window GPT-5.3-Chat reaches 96%; letting it choose the t
    
[^149]: WEAVE：将真实世界代码演进编织为依赖结构化任务，用于代码智能体的基准测试与训练

    WEAVE: Weaving Real-World Code Evolution into Dependency-Structured Tasks for Benchmarking and Training Code Agents

    [https://arxiv.org/abs/2610.22039](https://arxiv.org/abs/2610.22039)

    WEAVE 框架将真实世界中经过验证的多 PR 代码演进历史转化为依赖结构化的可执行任务，并引入基于测试的确定性度量 Graph Credit，以更好地对长程代码智能体进行基准评估和训练。

    

    真实世界的软件工程很少由孤立的变更构成。功能开发、迁移和系统性修复往往跨越多个拉取请求（PR）展开，需要对整个代码库进行协调一致的修改。然而，大多数可执行的代码库级基准测试仍以单个 issue 或 PR 作为任务单元，而现有的多 PR 设置则依赖粗粒度的聚合和与结构无关的评估。这种错位掩盖了变更之间的依赖关系，限制了长程代码智能体的进展衡量和学习信号。我们提出了 WEAVE，一个将经过验证的 PR 历史转化为可执行的、依赖结构化任务的框架。WEAVE 利用代码库校准的类型化变更位点，将相关联的 PR 组织成有界 DAG（有向无环图），并将每个 DAG 实例化为统一的可执行任务。我们进一步引入了 Graph Credit，这是一种确定性的、基于测试的度量指标，用于衡量符合依赖关系的部分进展，并支持更密集的……

    arXiv:2610.22039v1 Announce Type: cross  Abstract: Real-world software engineering rarely consists of isolated changes. Features, migrations, and systematic fixes often unfold across multiple pull requests (PRs), requiring coordinated modifications across a repository. Yet most executable repository-level benchmarks retain a single issue or PR as the task unit, while existing multi-PR settings rely on coarse aggregation and structure-agnostic evaluation. This mismatch obscures dependencies among changes and limits progress measurement and learning signals for long-horizon code agents. We introduce WEAVE, a framework that transforms verified PR histories into executable, dependency-structured tasks. WEAVE leverages repository-calibrated typed change loci to organize related PRs into bounded DAGs and instantiate each as a unified executable task. We further introduce Graph Credit, a deterministic, test-grounded metric that measures dependency-consistent partial progress and supports dens
    
[^150]: 二比特的N-Gram记忆：大语言模型中查找表的量化

    N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models

    [https://arxiv.org/abs/2610.21810](https://arxiv.org/abs/2610.21810)

    本文提出条件程序（CP）压缩表示方法，仅用两比特编码按需重构大语言模型n-gram记忆查找表中被请求的行，相比标准2比特量化将输出KL散度降低41–63%，并将DeepSeek的表存储从203GB压缩至51GB。

    

    条件记忆通过选择性查找来扩展语言模型的容量，但完整的嵌入表必须始终可用以应对每一个可能的输入。我们提出了条件程序，这是一种压缩表示方法，只需重建输入所请求的行。CP用短码和局部尺度来表示每一行，并使用一个在整个表上共享的重构程序。我们通过高斯格、范数恢复以及可独立解码的行记录来实例化这种表示。CP的两比特码相比2比特最近取整量化，将输出KL散度降低了41–63%。对三个经过训练的n-gram记忆族的分析表明，类高斯的坐标统计与行之间学习到的关系并存；在相同的存储预算下，共享格结构的表现优于我们测试的拟合码本和低秩残差变体。对于DeepSeek，该方法将表存储从203GB降至51GB，为更大的工作负载留出了空间。

    arXiv:2610.21810v1 Announce Type: cross  Abstract: Conditional memory expands language-model capacity through selective lookup, but the full embedding table must remain available for every possible input. We introduce Conditional Program (CP), a compressed representation that reconstructs only the rows an input requests. CP represents each row by a short code and a local scale, with a reconstruction program shared across the table. We instantiate this representation with a Gaussian trellis, norm restoration and independently decodable row records. CP's two-bit code lowers output KL by 41--63\% relative to 2-bit round-to-nearest. Analysis across three trained n-gram memory families shows that Gaussian-like coordinate statistics coexist with learned relationships among rows; at matched storage budgets, the shared trellis outperforms the fitted codebooks and low-rank residual variants we test. For DeepSeek, it reduces table storage from 203GB to 51GB, leaving room for larger workloads whe
    
[^151]: DialFork：超越事实性，评估对话代理中的程序性幻觉

    DialFork: Going Beyond Factuality to Evaluate Procedural Hallucination in Conversational Agents

    [https://arxiv.org/abs/2610.21561](https://arxiv.org/abs/2610.21561)

    DialFork是首个评估对话代理“程序性幻觉”的基准，通过在受控对话轮次位置注入四种幻觉类型构建了涵盖12个领域、24个任务的36,278个对话，并从三个粒度定义了刻画程序性偏差的评估任务。

    

    作为对话代理部署的大语言模型（LLM）被越来越多地期望遵循程序性逻辑。虽然事实性幻觉已受到广泛关注，但另一种失败模式却被忽视了：程序性幻觉，即代理产生的语句虽然单独来看流畅且事实上貌似合理，却偏离了预期的对话流程。我们提出了DialFork，这是首个在原始对话文本上衡量这种失败模式的基准。DialFork通过在受控轮次位置对符合规范的对话进行分叉并注入四种幻觉类型之一来构建，产生了涵盖12个领域、24个任务的36,278个对话；30名人工标注者验证了该构建过程。我们定义了三个评估任务，从三个粒度刻画程序性偏差：一组对话偏离符合规范对话的程度（用于监控已部署的代理）、单个对话偏离的程度（用于程序性检索……

    arXiv:2610.21561v1 Announce Type: cross  Abstract: Large language models (LLMs) deployed as conversational agents are increasingly expected to follow procedural logic. While factual hallucination has received extensive attention, a different failure mode has been overlooked: procedural hallucination, in which an agent produces individually fluent and factually plausible utterances while deviating from the intended conversational process. We introduce DialFork, the first benchmark to measure this failure mode on raw dialogue text. DialFork is built by forking conformant dialogues at controlled turn positions and injecting one of four hallucination types, yielding 36,278 dialogues across 24 tasks in 12 domains; 30 human annotators validate the construction. We define three evaluation tasks that characterize procedural deviation at three granularities: how far a set of dialogues deviates from conformant ones (monitoring deployed agents), how far a single dialogue does (procedural retrieva
    
[^152]: OPRD：面向长时程编码智能体的在线策略提醒蒸馏

    OPRD: On-Policy Reminder Distillation for Long-Horizon Coding Agents

    [https://arxiv.org/abs/2610.21496](https://arxiv.org/abs/2610.21496)

    OPRD通过多范围提醒监督与基于结果的路由机制，将教师对失败轨迹的再评估能力以反向KL蒸馏方式传递给学生智能体，从而解决长时程编码智能体无法利用上下文信息和过程监督稀疏的问题。

    

    可靠的仓库级代码编写不仅要求智能体生成正确的代码，还要求智能体在长交互过程中保持约束条件并对工具反馈作出响应。然而，我们的诊断分析表明，智能体常常未能对上下文中已有的相关信息采取行动，而最终的通过/失败结果只能提供稀疏的过程监督。我们提出了在线策略提醒蒸馏（OPRD），该方法将多范围提醒监督与基于结果的路由机制相结合，应用于无提醒的学生策略回放。成功的回放可提供自回归的自我模仿目标，而失败的回放则从教师那里获得以提醒为条件的反向KL散度监督，该教师会重新评估相同的学生生成前缀。全局提醒将最终测试诊断转化为轨迹层面的指导，而局部提醒仅应用于相关环境观察之后的下一个决策。教师与学生

    arXiv:2610.21496v1 Announce Type: new  Abstract: Reliable repository-level coding requires agents not only to generate correct code but also to preserve constraints and respond to tool feedback across long interactions. However, our diagnostic analysis shows that agents often fail to act on relevant information already available in context, while final Pass/Fail outcomes provide only sparse process supervision. We introduce On-Policy Reminder Distillation (OPRD), which combines Multi-Scope Reminder supervision with outcome-based routing on reminder-free student rollouts. Successful rollouts provide autoregressive self-imitation targets, whereas failed rollouts receive Reminder-conditioned reverse-KL supervision from a teacher that re-evaluates the same student-generated prefixes. A Global Reminder converts final-test diagnostics into trajectory-level guidance, while Turn-local Reminders apply only to the next decision following the relevant environment observation. The teacher and stud
    
[^153]: SkillNEO：智能体技能的非精英主义进化优化

    SkillNEO: Non-Elitist Evolutionary Optimization of Agent Skills

    [https://arxiv.org/abs/2610.20926](https://arxiv.org/abs/2610.20926)

    SkillNEO提出了一种非精英主义的智能体技能进化优化方法，通过分离“使用的版本”与“持续改进的版本”的选择，允许得分较低的版本继续接受编辑完善，从而避免因要求立即获得奖励提升而丢弃有价值的多步部分修复。

    

    智能体技能将程序性知识转化为可复用的产物，使LLM智能体能够通过经验积累和完善专业知识。在此基础上，技能进化利用执行反馈迭代改进技能。然而，改进可能需要多次相互依赖的编辑，因此低奖励版本仍然值得继续开发。若要求立即获得奖励提升，可能会在部分修复的效果显现之前就丢弃这些有价值的修改。非精英主义进化允许新版本即使得分低于早期版本，也能继续接受编辑。对于智能体技能而言，这需要证据来指明应保留什么以及下一步修复什么。我们提出了SkillNEO（智能体技能的非精英主义进化优化），它将“选择使用的版本”与“选择继续改进的版本”分离开来。通过比较相关版本执行同一任务的表现，可以识别出应保留的有效步骤以及尚待修复的失败之处。它还可以利用后续版本的经验重新审视和修改早期版本。

    arXiv:2610.20926v1 Announce Type: new  Abstract: Agent skills turn procedural knowledge into reusable artifacts, allowing LLM agents to accumulate and refine expertise through experience. Building on this, skill evolution iteratively improves skills using execution feedback. However, improvement may require several interdependent edits, so low-reward versions can still be worth developing. Requiring immediate reward gains can discard useful partial repairs before their benefits emerge. Non-elitist evolution allows new versions to receive further edits despite scoring below earlier ones. For agent skills, this calls for evidence of what to preserve and repair next. We introduce SkillNEO (Non-Elitist Evolutionary Optimization for Agent Skills), which separately selects a version to use and versions to keep improving. Comparing how related versions perform the same task reveals working steps to preserve and remaining failures to fix. It can also revisit and revise earlier versions using l
    
[^154]: 扩展多语言专家混合模型：语言覆盖范围如何改变容量与计算分配

    Scaling Multilingual Mixture-of-Experts: How Language Coverage Changes Capacity and Compute Allocation

    [https://arxiv.org/abs/2610.20247](https://arxiv.org/abs/2610.20247)

    本文提出了一种感知重复的联合缩放定律，揭示多语言MoE模型中活跃参数、训练词元、专家数量与语言覆盖范围之间的关系，并发现32个专家相比稠密模型可将语言从12种扩展到200种时的损失惩罚降低83%。

    

    多语言模型在各语言之间共享容量，因此增加语言数量可能会损害现有语言的表现。专家混合模型只需增加极少量的每词元计算即可扩展这一容量。鉴于这种额外容量，增加专家数量能否降低每种新增语言的代价？为回答这一问题，我们提出了一种感知重复的联合缩放定律，将活跃参数数量、每种语言的训练词元数量、专家数量与语言覆盖范围联系起来。该定律基于来自64个预训练配置的26,300个逐语言损失观测数据进行拟合，在留出配置上相比最强评估基线将预测误差降低了28%。在活跃参数数量和每种语言训练词元数量相同的情况下，32个专家相比稠密模型将语言从12种扩展到200种时的损失惩罚降低了83%。我们的定律进一步表明，更广的语言覆盖范围会在更大的参数范围内倾向于使用更多专家。

    arXiv:2610.20247v1 Announce Type: new  Abstract: Multilingual models share capacity across languages, so adding languages can hurt performance on existing ones. Mixture-of-experts models expand this capacity with little additional per-token computation. Given this extra capacity, does adding experts reduce what each additional language costs? To answer this question, we propose a repetition-aware joint scaling law relating number of active parameters, training tokens per language, number of experts, and language coverage. Fitted on 26,300 per-language loss observations from 64 pretrained configurations, our law reduces prediction error on held-out configurations by 28% relative to the strongest evaluated baseline. At matched number of active parameters and training tokens per language, 32 experts reduce the loss penalty of expanding from 12 to 200 languages by 83% compared with a dense model. Our law further shows that broader language coverage favors more experts across a wider range 
    
[^155]: 论自然语言自编码器的可监控性：识别与判别

    On the Monitorability of Natural Language Autoencoders: Identification vs Discrimination

    [https://arxiv.org/abs/2610.18727](https://arxiv.org/abs/2610.18727)

    论文将自然语言自编码器的训练与监督部署形式化为两种不同的指称博弈，并指出二者之间的差距导致其解释虽能识别对应的激活，却难以让人类监控者从多个近似叙述中辨别出真实的那一个。

    

    自然语言自编码器（NLA）被提出作为语言模型的监督工具：激活言语化器（AV）为激活向量撰写英文解释，而激活重构器（AR）将解释映射回向量。它们在监督中的价值取决于人类监控者在给定AV解释的情况下，能否将一段对激活的真实叙述与虚假叙述区分开来。我们将NLA的训练与NLA监督部署形式化为源自涌现通信领域的两种不同的指称博弈：训练是一种生成式指称博弈，而监督是一种判别式的、多模态的、部分可观测的指称博弈，该博弈在受监控语言模型的激活与真实描述它们的叙述之间的多对多关系上展开。这两种博弈之间的差距预示着AV的解释将能够识别其所描述的是哪个激活，却无法将其真实叙述与紧密匹配的叙述区分开来。

    arXiv:2610.18727v2 Announce Type: replace-cross  Abstract: Natural Language Autoencoders (NLAs) are proposed as oversight tools for language models: an Activation Verbaliser (AV) writes an English explanation of an activation vector, and an Activation Reconstructor (AR) maps the explanation back to a vector. Their value for oversight depends on a human monitor being able to tell a true account of an activation from a false one, given the AV's explanation. We formalise NLA training and NLA oversight deployment as two different referential games from Emergent Communication: training is a generative referential game, while oversight is a discriminative, multi-modal, partially-observable referential game played over a many-to-many relation between the monitored language model's activations and the accounts that are true of them. The gap between the two games predicts that AV's explanations will identify which activation they describe without discriminating its true account from closely mat
    
[^156]: 探索大语言模型中基于深度的动态推理的越狱漏洞

    Exploring Jailbreak Vulnerability Across Depth-Based Dynamic Inference in Large Language Models

    [https://arxiv.org/abs/2610.15207](https://arxiv.org/abs/2610.15207)

    本文首次系统研究了基于深度的动态推理对大语言模型越狱鲁棒性的影响，发现浅层退出始终比深层退出对越狱攻击更具鲁棒性，揭示了动态推理灵活性所带来的安全风险。

    

    大语言模型（LLMs）已被证明计算成本高昂且容易受到越狱攻击，即对抗性提示可以绕过安全对齐并诱导产生有害响应。为了解决这种低效问题，研究者提出了一系列技术以实现高效的大语言模型推理。基于深度的动态推理通过允许从中间层生成输出，已展现出令人鼓舞的效果，它提供了根据可用资源预算灵活调整模型行为的能力。然而，这种灵活性如何影响大语言模型对越狱攻击的鲁棒性仍尚未被探索。在本文中，我们通过对基于深度的动态推理对大语言模型对抗越狱攻击鲁棒性的影响进行全面研究来填补这一空白。具体而言，我们得出以下关键观察：1) 浅层退出始终比深层退出对越狱攻击更具鲁棒性；2) 退出深度……

    arXiv:2610.15207v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have been shown to be computationally expensive and vulnerable to jailbreak attacks, in which adversarial prompts bypass safety alignment and induce harmful responses. To address this inefficiency, a range of techniques have been proposed to enable efficient LLM inference. Depth-based dynamic inference has demonstrated promising results by enabling outputs to be generated from intermediate layers. It offers the flexibility to adjust model behavior based on the available resource budget. However, how this flexibility affects the robustness of LLMs against jailbreak attacks remains unexplored. In this paper, we address this gap via a comprehensive investigation into the impact of depth-based dynamic inference on the robustness of LLMs against jailbreak attacks. Specifically, we make the following key observations: 1) Shallow exits are consistently more robust to jailbreak attacks than deeper ones; 2) Exit dep
    
[^157]: 迈向“系统一”临床智能：决策基础模型的参数高效微调与反事实可解释性

    Toward System One Clinical Intelligence: Parameter-Efficient Fine-Tuning and Counterfactual Explainability of Decision Foundation Models

    [https://arxiv.org/abs/2610.15029](https://arxiv.org/abs/2610.15029)

    该研究通过QLoRA参数高效微调将基于Qwen3.5的90亿参数决策基础模型Clef-Flash适配于24类疾病症状分类任务，取得95.42%的准确率并显著超越零样本基线与ClinicalBERT，同时结合反事实可解释性提升了临床决策的透明度。

    

    从患者报告的症状进行自动化疾病分类，需要在轻量级判别模型的推理速度与自回归大语言模型（LLM）的上下文理解能力和延迟成本之间取得平衡。决策基础模型提供了一种替代方案，它将表示直接映射到候选类别，而无需自由文本生成。我们对Clef-Flash（一个基于Qwen3.5骨干网络构建的90亿参数决策模型）进行了适配，用于Symptom2Disease基准上的24类疾病分类。通过采用带有差分学习率的4比特量化低秩适配（QLoRA），以及在960个训练案例和240个测试案例上的分层划分，该模型达到了95.42%的诊断准确率和0.9551的宏平均F1分数。这一结果优于已报告的远程API零样本基线（61.67%）和本地未适配初始化的零样本基线（65.00%），并超越了ClinicalBERT基线（90.90%）。通过评估候选诊断的反事实可解释性，该研究进一步增强了模型决策的透明度与可信度。

    arXiv:2610.15029v1 Announce Type: cross  Abstract: Automated disease classification from patient-reported symptoms requires balancing the speed of lightweight discriminative models against the contextual understanding and latency costs of autoregressive large language models (LLMs). Decision foundation models offer an alternative by mapping representations directly to candidate classes without free-form text generation. We adapt Clef-Flash, a 9-billion-parameter decision model built on a Qwen3.5 backbone, for 24-class disease classification on the Symptom2Disease benchmark. Using 4-bit quantized low-rank adaptation (QLoRA) with differential learning rates and a stratified split of 960 training and 240 test cases, the model achieves 95.42% diagnostic accuracy and a macro F1 score of 0.9551. This improves upon the reported zero-shot baselines of 61.67% for a remote API and 65.00% for a local unadapted initialization, and outperforms the ClinicalBERT baseline (90.90%). By evaluating candi
    
[^158]: 个性化而不刻板印象：语音大语言模型知道何时该“充耳不闻”吗？

    Personalization Without Stereotyping: Do Speech LLMs Know When to Turn a Deaf Ear?

    [https://arxiv.org/abs/2610.13550](https://arxiv.org/abs/2610.13550)

    提出了 BASIS-Bench 基准，通过人口统计稳健性、刻板印象偏见和个性化三个维度区分语音大语言模型的有益个性化与刻板印象行为，评估发现多数模型虽能识别人口统计线索却未能恰当运用。

    

    近期的语音大语言模型（Speech LLM）能够利用丰富的副语言线索，实现更复杂和个性化的交互。然而，很少有研究调查这种敏感性如何导致刻板印象行为，且现有评估大多仅衡量人口统计差异，未能区分有益的个性化与刻板印象行为。我们提出了 BASIS-Bench，这是一个旨在通过三个维度区分这两个方面的基准：（1）人口统计稳健性，衡量基于内容的任务在不同说话人之间的表现差异以及人口统计分类能力；（2）刻板印象偏见，衡量感知到的人口统计线索是否会触发未经提示的刻板印象行为；（3）个性化，衡量模型在被明确要求时能否根据人口统计特征定制响应。我们评估了八个近期的语音大语言模型，发现尽管它们通常能够识别人口统计线索，但大多数模型未能将其转化为恰当的行为。

    arXiv:2610.13550v1 Announce Type: new  Abstract: Recent speech LLMs can leverage rich paralinguistic cues, enabling more complex and personalized interactions. However, little work investigates how this sensitivity can lead to stereotyping, and existing evaluations largely measure demographic disparities without distinguishing beneficial personalization from stereotypical behavior. We introduce BASIS-Bench, a benchmark designed to disentangle these two aspects through three axes: (1) demographic robustness, measuring disparities in content-based tasks across speakers and demographic classification capabilities; (2) stereotypical bias, measuring whether perceived demographic cues trigger unprompted stereotypical behavior; and (3) personalization, measuring whether models can tailor responses to demographic characteristics when explicitly requested. We evaluate eight recent speech LLMs and find that, despite often recognizing demographic cues, most fail to translate them into appropriate
    
[^159]: REMORY：学习残差记忆用于上下文压缩

    REMORY: Learning Residual Memory for Context Compaction

    [https://arxiv.org/abs/2610.11287](https://arxiv.org/abs/2610.11287)

    REMORY 提出一种神经记忆网络，通过生成软记忆token作为文本摘要的“残差”补充，使冻结的LLM仅用 5.2% 的输入位置即可接近全上下文性能，并显著减少重复工具输出和工具错误。

    

    长时程智能体通过压缩历史记录来在有限的上下文窗口内继续运行，但仅靠文本摘要可能无法支持后续的每一个决策。我们提出了 REMORY，这是一种神经记忆网络，用一段有界的软记忆token序列来补充摘要。给定历史记录和摘要，该网络学习生成有助于冻结的LLM近似其在完整历史输入下会产生的内容的token。这些token以摘要为条件并被附加在摘要之后，形成了沿序列维度上类似残差连接的机制。在 SummHay 基准上，REMARY 在洞察覆盖率几乎不变的情况下提升了来源归因能力，并且仅使用 5.2% 的输入位置就接近了全上下文的联合分数。在长时程智能体基准测试中，Qwen3.8-27B 和 GLM-5.3-Flash 使用残差记忆均显示出一致的提升。两个模型还表现出明显更少的重复工具输出和工具错误。

    arXiv:2610.11287v1 Announce Type: cross  Abstract: Long-horizon agents compact their history to continue within a finite context window, but a textual summary alone may not support every subsequent decision. We introduce REMORY, a neural memory network that supplements the summary with a bounded sequence of soft memory tokens. Given the history and summary, the network learns to generate tokens that help a frozen LLM approximate the continuation it would produce with the full history. The tokens are conditioned on the summary and appended after it, forming an analogue of a residual connection along the sequence dimension. On SummHay, REMORY improves source attribution at nearly unchanged insight coverage and approaches the full-context joint score using only 5.2% of the input positions. Across long-horizon agent benchmarks, Qwen3.8-27B and GLM-5.3-Flash show consistent gains with residual memory. Both models also exhibit substantially fewer repeated tool outputs and tool errors on Brow
    
[^160]: DLoop：循环式推测解码

    DLoop: Looped Speculative Decoding

    [https://arxiv.org/abs/2610.07659](https://arxiv.org/abs/2610.07659)

    DLoop提出一种循环式推测解码方法，在验证前自适应地执行多个起草阶段，从而减少目标模型不必要的前向验证，提升大语言模型推测解码的加速效果。

    

    推测解码可加速大语言模型的自回归生成。在每个起草阶段，轻量级的草稿模型提出若干token，随后由目标模型进行验证。我们发现，随着草稿模型能力不断增强，目标模型经常接受起草阶段产生的所有token。然而每次起草阶段之后仍会进行一次验证，即使起草本可继续进行，这导致目标模型执行了不必要的前向传播。自适应草稿长度方法可在解码过程中决定验证前生成多少草稿token，但它们只能为自回归草稿模型提升加速效果。对于并行草稿模型，进一步起草还需要目标模型为尚未验证的草稿token提供隐藏状态。我们提出了DLoop，一种循环式的推测解码方法，可在验证之前自适应地执行多个起草阶段。DLoop在草稿模型保持…

    arXiv:2610.07659v1 Announce Type: new  Abstract: Speculative decoding accelerates autoregressive generation in large language models. In each drafting stage, a lightweight draft model proposes tokens that the target model subsequently verifies. With increasingly capable draft models, we find that the target model frequently accepts all tokens produced in a drafting stage. A verification nevertheless follows each drafting stage, resulting in unnecessary target-model forward passes even when drafting could have continued. Adaptive draft length methods decide during decoding how many draft tokens precede a verification, but they raise the speedup only for autoregressive draft models. For a parallel draft model, drafting further requires target-model hidden states for draft tokens that have not been verified. We propose DLoop, a looped form of speculative decoding that adaptively performs multiple drafting stages before verification. DLoop continues drafting while the draft model remains c
    
[^161]: 最小见证强化学习

    Minimal Witness Reinforcement Learning

    [https://arxiv.org/abs/2610.07226](https://arxiv.org/abs/2610.07226)

    本文提出最小见证强化学习（MWRL），利用基于集合并集覆盖损失的信用分配机制，仅凭单一黑盒验证器的信号即可同时实现解的最小性与多个备选解的恢复。

    

    “足以产生某一结果的不可约条件是什么？”这是计算与科学领域中最常出现的核心问题之一。其答案——最小充分见证——正是我们所称的解释、机制与原因。这类问题通常需要找出多个最小见证，然而标准的强化学习方法可能只能揭示一个解或冗余的解。我们将该问题形式化为最小见证识别，并提出了最小见证强化学习（MWRL）。MWRL 对从策略中采样得到的、经成功验证的提议所认证的集合取并集，并根据“若缺少该提议，组并集将会损失的覆盖范围”为每个提议分配信用。这一直接源于问题定义的信用分配机制，仅凭一个黑盒验证器的单个比特信号，便统一了对最小性与备选解恢复的双重需求。基于该原理，我们推导出一个值迭代规划器，能够恢复……（摘要原文在此处截断）

    arXiv:2610.07226v1 Announce Type: cross  Abstract: ``What are the irreducible conditions that are sufficient to produce an outcome?'' is one of the most common questions that recur across computation and science. Its answers, the minimal sufficient witnesses, are what we mean by explanations, mechanisms and reasons. These problems usually ask for multiple minimal witnesses, yet standard RL methods may reveal only one solution or redundant ones. We formalize this problem as minimal-witness identification and introduce Minimal-Witness Reinforcement Learning (MWRL). MWRL takes the union of the sets certified by successful proposals sampled from the policy and credits each proposal for the coverage the group union would lose without that proposal. This credit assignment, derived directly from the problem definition, unifies the demands for minimality and recovery of alternatives from a single black-box verifier bit. Under this principle, we derive a value iteration planner that recovers th
    
[^162]: 为长程工作智能体扩展可验证环境

    Scaling Verifiable Environments for Long-horizon Work Agents

    [https://arxiv.org/abs/2610.04906](https://arxiv.org/abs/2610.04906)

    WorkForge 提出了一个可扩展的合成框架，从专家工作流和真实世界资源出发，通过提取可核查的事实锚点，自动构建支持长程交互与可信验证的工作智能体训练环境。

    

    工作智能体通过操作数字制品来执行专业性的知识密集型工作，因此需要既能支持长程交互又能提供可信验证的训练环境。然而，手工构建的环境会带来高昂的工程开销，使环境难以规模化；而现有的合成方法则会牺牲工作区的复杂性、真实性或有据可依的可验证性。为了弥合这一差距，我们提出了 WorkForge，一个可扩展的合成框架，用于从真实世界资源构建可验证的工作智能体环境。WorkForge 从专家工作流出发展开：首先识别每个工作流所需的资源、决策和交付成果；然后检索相关的真实世界文件，并将其组织成一个工作区；接着检查该工作区，提取其中内容的具体、可核查的事实。这些事实锚点确定了工作区能够支持的任务类型及其结果的验证方式……（原文摘要在此处截断）

    arXiv:2610.04906v2 Announce Type: replace-cross  Abstract: Work agents operate over digital artifacts to execute professional knowledge-intensive work, requiring training environments that support long-horizon interaction and trustworthy verification. However, hand-crafted environments incur prohibitive engineering overhead that prevents environment scaling, whereas synthesis methods sacrifice workspace complexity, realism, or grounded verifiability. To bridge this gap, we introduce WorkForge, a scalable synthesis framework for constructing verifiable work-agent environments from real-world resources. Starting from expert workflows, WorkForge first identifies the resources, decisions, and deliverables required by each workflow. It then retrieves relevant real-world files and organizes them into a workspace. WorkForge inspects the workspace to extract concrete, checkable facts about its content. These factual anchors fix which task types the workspace can support and how their outcomes 
    
[^163]: 评估大语言模型临床推理能力的量规全景：已有什么、缺失什么、需要组合什么

    A rubric landscape for evaluating clinical reasoning in large language models: what exists, what is missing, and what needs to be combined

    [https://arxiv.org/abs/2610.01938](https://arxiv.org/abs/2610.01938)

    该结构化综述系统梳理了用于评估大语言模型临床推理的医学教育工具、临床基准与长文本生成评估方法，覆盖病情表征、时序综合、鉴别与管理推理、反事实推理、不确定性校准和推理忠实性六个维度，并指出目前没有任何单一工具能覆盖全部维度，需要将多种评估方法组合使用。

    

    考试式准确率并不能证明大语言模型（LLM）能够在临床病历上进行良好的推理。我们将临床推理定义为：跨时间和跨来源整合与更新证据，以形成、修订和论证患者的病情表征以及合理且可辩护的诊疗方案。本结构化叙述性综述梳理了三方面文献：医学教育评估工具、2023年以来发表的临床LLM基准测试，以及通用领域中用于评估长文本生成的方法。我们考察了六个维度：病情表征、时序综合、鉴别诊断与管理推理、反事实推理、校准的不确定性以及推理忠实性。预印本论文也被纳入并予以标注。没有任何单一工具能够覆盖全部六个维度。病情表征以及鉴别诊断/管理推理已得到较为合理的覆盖，尽管其可靠性因工具和应用场景而异。TIMER-Eval 针对时序……（摘要原文在此处截断）

    arXiv:2610.01938v1 Announce Type: cross  Abstract: Exam-style accuracy does not establish whether large language models (LLMs) reason well over clinical records. We define clinical reasoning as integrating and updating evidence across time and sources to form, revise and justify a patient's problem representation and a defensible plan.   This structured narrative review maps three literatures: medical education assessment instruments, clinical LLM benchmarks published from 2023 onwards, and general-domain methods for evaluating long-form generation. We examine six dimensions: problem representation, temporal synthesis, differential and management reasoning, counterfactual reasoning, calibrated uncertainty, and reasoning faithfulness. Preprints are included and flagged.   No single instrument covers all six dimensions. Problem representation and differential or management reasoning are reasonably covered, although reliability varies by instrument and setting. TIMER-Eval targets temporal
    
[^164]: 大语言模型能规范化数据库吗？面向模式规范化的基准测试与多智能体框架

    Can LLMs Normalize Databases? A Benchmark and Multi-Agent Framework for Schema Normalization

    [https://arxiv.org/abs/2609.11141](https://arxiv.org/abs/2609.11141)

    该论文提出了包含3,275个样本的数据库规范化基准DNBENCH，系统揭示了LLM在函数依赖推理、模式分解和表间约束重建中的常见失败，并进一步提出多智能体框架MARS以提升模式规范化的可靠性。

    

    大语言模型（LLM）越来越多地被用于生成结构化输出，但当这些输出必须满足数据库级别的约束时，其可靠性仍不明确。我们通过数据库规范化来研究这一问题，这涉及对函数依赖、无损连接分解以及表间约束的推理。我们提出了数据库规范化基准测试，包含3,275个样本，用于评估从1NF到BCNF的LLM驱动的数据库规范化能力。DNBENCH采用三轴评估协议，衡量语义等价性、结构准确性和逻辑有效性。在单一、复杂和真实世界三个难度层级上，DNBENCH揭示了LLM在依赖推理、模式分解和表间约束重建中反复出现的失败。我们进一步提出了面向模式的多智能体推理框架，它将证据提取、违规诊断和分解规划从模式生成中分离出来……

    arXiv:2609.11141v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used to generate structured outputs, but their reliability remains unclear when those outputs must satisfy database-level constraints. We study this issue through database normalization, involving reasoning about functional dependencies, lossless join decompositions, and inter-table constraints. We introduce a Database Normalization Benchmark (DNBENCH), comprising 3,275 samples for evaluating LLM-driven database normalization from 1NF to BCNF. DNBENCH uses a three-axis protocol to measure semantic equivalence, structural accuracy, and logical validity. Across Single, Complex, and Real World levels, DNBENCH uncovers recurring failures in dependency inference, schema decomposition, and inter-table constraint reconstruction. We further propose Multi-Agent Reasoning for Schemas (MARS), which separates evidence extraction, violation diagnosis, and decomposition planning from schema generation and 
    
[^165]: 面向大语言模型的CPR：领域自适应中基于临界点路由的灾难性遗忘防御方法

    CPR for LLMs: Critical-Point Routing against Catastrophic Forgetting in Domain Adaptation

    [https://arxiv.org/abs/2608.30158](https://arxiv.org/abs/2608.30158)

    本文提出CPR框架，通过识别基座模型失败而专家模型成功的临界词元，在基座模型与其SFT专家版本之间进行词元级路由，从模型层面解耦领域能力与通用能力，有效缓解领域自适应中的灾难性遗忘。

    

    监督微调（SFT）是将大语言模型（LLM）适配到目标领域的实际标准方法，但它常常会降低模型的通用能力，这一现象被称为“灾难性遗忘”。现有方法通常通过修改SFT损失来缓解遗忘问题，但它们不可避免地运行在领域能力与通用能力的权衡之中。在本工作中，我们通过在模型层面解耦这两种能力来跳出这一权衡：保留原始基座模型以维持通用能力，仅在需要领域特定知识时选择性地调用SFT专家模型。具体而言，我们提出了CPR（临界点路由），这是一个在基座模型与其专家衍生模型之间进行词元级路由的框架，其依据是那些基座模型失败而专家模型成功的“临界词元”。我们训练了一个轻量级的分层路由器来估计每个词元调用专家的概率，并将其与精心设计的推理策略相配……（原文摘要于此处截断）

    arXiv:2608.30158v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) is the de facto standard for adapting large language models (LLMs) to target domains, but it often degrades the model's general capabilities, a phenomenon known as catastrophic forgetting. Existing approaches typically modify the SFT loss to mitigate forgetting, but they inevitably operate along a domain-generality trade-off. In this work, we step outside this trade-off by decoupling the two capabilities at the model level: we keep the original base model for general capability, and selectively invoke the SFT expert only when domain-specific knowledge is required. Specifically, we propose CPR (Critical-Point Routing), a token-level routing framework between a base model and its expert derivative, based on critical tokens where the base model fails but the expert succeeds. We train a lightweight hierarchical router that estimates the expert-call probability per token, and pair it with a tailored inference pr
    
[^166]: 从原子到智能体：迈向可解释的LLM智能体数学能力评估

    From Atomic to Agentic: Towards Interpretable Evaluation of LLMs' Agentic Mathematical Capabilities

    [https://arxiv.org/abs/2608.26950](https://arxiv.org/abs/2608.26950)

    本文提出一个过程级数学基准，通过将智能体行为与原子能力分类对齐，并利用自动化流水线生成细粒度注释，实现可解释地评估LLMs的智能体数学推理能力。

    

    大型语言模型（LLMs）正从执行端到端的数学推理演变为集成智能体智能。然而，大多数现有的数学基准仅评估最终答案。这种面向结果的评估在识别过程级错误或严格逻辑方面提供的诊断价值有限，无法引导LLMs转变为稳健的智能体。为弥补这一差距，我们提出一个过程级基准，旨在评估LLMs固有的智能体数学推理能力。我们的框架将问题解决智能体行为与可复用的数学原子能力的结构化分类对齐。我们设计了一套全面的规划、行动和反馈任务，涵盖文本和多模态场景，并辅以自动化流水线，通过受控的LLM重写合成高质量轨迹并生成细粒度注释。实验表明，具备智能体能力的模型在此基准上表现更优。

    arXiv:2608.26950v1 Announce Type: new  Abstract: Large Language Models (LLMs) are evolving from performing end-to-end mathematical reasoning to integrating agentic intelligence. However, most existing math benchmarks evaluate only final answers. This outcome-oriented evaluation provides limited diagnostic value for identifying process-level failures or rigorous logic, failing to guide the transformation of LLMs into robust agents. To bridge this gap, we present a process-level benchmark designed to evaluate the inherent agentic mathematical reasoning abilities of LLMs. Our framework aligns problem-solving agentic behaviors with a structured taxonomy of reusable mathematical atomic capabilities. We design a comprehensive suite of planning, action, and feedback tasks across both textual and multimodal contexts, supported by an automated pipeline that synthesizes high-quality trajectories and produces fine-grained annotations via controlled LLM rewriting. Experiments reveal that models wi
    
[^167]: 大型语言模型决策中的代理依赖未针对预测证据进行校准

    Proxy reliance in large language model decisions is uncalibrated to predictive evidence

    [https://arxiv.org/abs/2608.22887](https://arxiv.org/abs/2608.22887)

    本文通过精确计算临床任务中的因果代理效应，发现LLM对代理的依赖未与预测证据校准，表现为过度依赖、合理依赖或不足依赖，且社会标签抑制效果脆弱。

    

    摘要：大型语言模型（LLMs）正进入分诊和贷款等决策领域，在这些领域中，必须将任务相关的推理与不可允许的代理使用区分开来。当前的审计方法通过检查人口统计学特征变化时决策是否改变来评估。但与受保护群体相关的属性带有预测价值，因此决策变化可能是歧视，也可能是合理推断。我们在一个具有已知真实结果的临床排序任务中测量了四种LLM的因果代理效应，其中证据所支持的依赖程度可以精确计算并作为参考。一个审计信号产生三种判定：过度依赖、合理依赖和不足依赖。在中性标签下，每个模型都依赖无信息的代理。有信息的代理则引发所有三种情况。社会领域名称将依赖程度推低，在其中一个模型中低于参考值。两项发现解释了这一点：依赖程度严重低于证据所指示的水平，且社会标签抑制是脆弱的，因为上下文示例会破坏这种抑制。

    arXiv:2608.22887v1 Announce Type: new  Abstract: Large language models (LLMs) are entering decisions in triage and lending, where task-relevant inference must be distinguished from impermissible proxy use. Current audits ask whether decisions change when demographics change. But attributes correlated with a protected group carry predictive value, so a changed decision can be discrimination or sound inference. We measure causal proxy effects in four LLMs on a clinical-ranking task with known ground truth, where the reliance the evidence warrants can be computed exactly and used as the reference. One audit signal yields three verdicts: over-reliance, warranted and under-reliance. Under neutral labels every model relies on proxies with no information. Informative proxies draw all three. Social field names push reliance down, below the reference in one model. Two findings explain this. Reliance severely undertracks the evidence, and social-label suppression is fragile, since in-context exa
    
[^168]: 论引用在偏好数据中的作用

    On the Role of Citations in Preference Data

    [https://arxiv.org/abs/2608.21376](https://arxiv.org/abs/2608.21376)

    本文通过混合效应模型研究科学问答中引用对人类和LLM偏好判断的影响，发现人类偏好多样化但数量少的引用，而LLM的引用偏好虽存在但受数据和模型影响。

    

    arXiv:2608.21376v1 公告类型：交叉 摘要：许多自然语言处理任务要求系统在输出中提供归属，即引用来源。归属是防止模型幻觉的堡垒，也是用户验证模型输出可信度的手段。然而，在比较输出时，人类和大型语言模型如何评估引用尚不清楚，这一过程对奖励建模和现代LLM的后训练至关重要。本文研究了在科学问答背景下，引用对人类评审者和四个开源LLM偏好中的作用，利用混合效应模型探讨引用对成对判断的影响。我们的主要发现包括：（1）人类偏好更多样化的引用，但总体数量更少；（2）与人类相比，LLM表现出一些与引用相关的偏好，尽管它们无法访问来源，但这些偏好依赖于数据和特定模型。我们进一步讨论了这些发现的影响。

    arXiv:2608.21376v1 Announce Type: cross  Abstract: Many NLP tasks require systems to provide attribution in their outputs--i.e. citations to grounding sources. Attribution serves as a bulwark against model hallucination and as a means for users to verify the credibility of model outputs. Yet, it is unclear how humans and LLMs evaluate citations when comparing outputs, a process central to reward modeling and modern LLM post-training. This paper studies the role of citations in the preferences of human judges and four open-source LLMs within the context of scientific question answering, leveraging mixed effects models to investigate the influence of citations on pairwise judgments. Among our key findings are (1) that humans prefer more diverse citations but fewer overall, and (2) that LLMs show some citation-related preferences compared to humans, despite lacking access to the sources, but these preferences depend on the data and specific models. We further discuss the implications of o
    
[^169]: 文本到视觉的迁移是什么？视觉语言模型的能力缩放规律与迁移动态

    What Transfers from Text to Vision? Capability Scaling Laws and Transfer Dynamics for VLMs

    [https://arxiv.org/abs/2608.00013](https://arxiv.org/abs/2608.00013)

    我们提出了首个跨家族的多模态缩放规律，通过文本能力得分直接预测VLM性能，并在150多个VLM上验证了其有效性。

    

    arXiv:2608.00013v2 公告类型：替换-交叉 摘要：选择合适的大型语言模型（LLM）骨干是构建视觉语言模型（VLM）时最关键的决策，但这一过程从根本上缺乏原则性：基于计算量的缩放规律无法跨模型家族泛化，且不存在在训练开始前直接预测VLM性能的框架。我们提出了能力驱动的多模态缩放规律，这是首个跨家族框架，能够从直接可观测的文本能力预测VLM基准准确率。给定通过主成分分析从LLM文本基准中提取的低维能力得分$S$，我们将VLM性能建模为$S$的函数，并引入每个骨干的迁移率和量化数据缩放效率的吸收率。为拟合和验证该框架，我们在严格控制的配方下，基于34个LLM（涵盖7个模型家族）训练了超过150个VLM。在超过200个文本基准和50个多模态基准上的评估表明...

    arXiv:2608.00013v2 Announce Type: replace-cross  Abstract: Choosing the right large language model (LLM) backbone is the most consequential decision when building a vision-language model (VLM), yet it remains fundamentally unprincipled: compute-based scaling laws fail to generalize across model families, and no framework exists for directly predicting VLM performance before training begins. We propose the Capability-Driven Multimodal Scaling Law, the first cross-family framework that predicts VLM benchmark accuracy from directly observable textual capability. Given a low-dimensional capability score $S$ extracted from LLM textual benchmarks via PCA, we model VLM performance as a function of $S$, with a per-backbone transfer rate and an absorption rate that quantifies data-scaling efficiency. To fit and validate the framework, we train over 150 VLMs on 34 LLMs spanning 7 model families under a strictly controlled recipe. Evaluations on more than 200 textual and 50 multimodal benchmarks 
    
[^170]: Token消耗最大化（Tokenmaxxing）的最佳编程语言：编程智能体在不同编程语言下的行为研究

    The Best Programming Language for Tokenmaxxing: An Investigation of Coding Agent Behavior Across Programming Languages

    [https://arxiv.org/abs/2607.22807](https://arxiv.org/abs/2607.22807)

    编程智能体的Token消耗因编程语言而显著不同且跨模型一致，其原因是智能体在不熟悉的语言中反复生成无法编译的代码并重复修改已通过的解决方案。

    

    尽管编程智能体目前在多种编程语言中都已非常有效，本文首次表明其成本（以Token计）会因编程语言的不同而产生显著差异。我们在Python、Java、Rust和OCaml的编程问题上评估了五个最新模型，并严格控制问题难度，结果显示不同编程语言之间的Token消耗存在显著差异，且这一差异在各个模型中保持一致。为探究其原因，我们对智能体轨迹的结构和内容进行了分析。首先，我们重新执行每个中间解，将每条轨迹抽象为测试结果向量的序列，并对相继解之间的工作内容进行标注，这揭示了智能体在不熟悉的语言中会反复生成无法编译的解决方案，并且会反复修改已经通过测试的解决方案。其次，我们分析轨迹文本，发现智能体会在代码注释中规划解决方案，不信任所提供的测试用例，而更倾向于使用自行构造的输入……（原文此处被截断）

    arXiv:2607.22807v2 Announce Type: replace-cross  Abstract: Although coding agents are now very effective in a variety of programming languages, this paper first shows that the cost (in tokens) can very significantly by programming language. We evaluate five recent models on programming problems in Python, Java, Rust, and OCaml. We carefully control for problem difficulty, and show that there can be stark variation in token consumption that is consistent across models. To understand why, we analyze both the structure and content of agent trajectories. First, we re-execute every intermediate solution and abstract each trajectory as a sequence of test-outcome vectors, then label the work between successive solutions. This reveals agents repeatedly producing noncompiling solutions in unfamiliar languages and revising solutions that already pass. Second, we analyze trajectory text, finding that agents plan solutions in code comments, distrust the provided tests in favor of inputs they inven
    
[^171]: DFAH-Bench：金融决策中可观测智能体不稳定性的基准测试

    DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making

    [https://arxiv.org/abs/2607.20491](https://arxiv.org/abs/2607.20491)

    该论文提出DFAH-Bench基准，通过同时衡量决策一致性与工具路径一致性，揭示金融智能体在决策结果高度稳定（约95%）的情况下，其背后的工具执行过程却存在显著不稳定性（一致性仅约50%）。

    

    金融智能体可以在重复做出相同决策的同时，改变其背后的工作过程。DFAH-Bench 将确定性-忠实性保障框架（DFAH）付诸实践，在相同的合格重放中同时配对决策一致性与工具路径一致性，然后将该资格认定原则扩展到证据、授权、执行和任务结果。回顾性与前瞻性重放分析揭示了稳定决策背后的过程变化。在570个符合条件的前瞻性情节中，决策一致性为94.2-95.1%，而有序工具、参数和结果的一致性仅为45.0-51.5%；其中一个分层比预先规定的覆盖最低值低了一组。一项独立的捕获诊断显示，系统性的遗漏仍可保持完美的重放一致性。使用 τ-Knowledge 银行环境，我们在采用开源权重与前沿生成器的不同队列中保留了1,080个计划情节和1,033个已知原生结果。

    arXiv:2607.20491v3 Announce Type: replace-cross  Abstract: A financial agent can repeat a decision while changing the work behind it. DFAH-Bench operationalizes the Determinism--Faithfulness Assurance Harness (DFAH), pairing decision agreement with tool-path agreement on the same qualified replays, then extends that qualification principle to evidence, authorization, execution and task outcomes. Retrospective and prospective replay analyses expose process variation behind stable decisions. Across 570 eligible prospective episodes, decision agreement is 94.2-95.1%, while agreement on ordered tools, arguments and results is 45.0-51.5%; one stratum falls one group below its prespecified coverage minimum. A separate capture diagnostic shows that systematic omissions can preserve perfect replay agreement. Using the $\tau$-Knowledge banking environment, we retain 1,080 scheduled episodes and 1,033 known native outcomes across separate cohorts with open-weight and frontier generators. Missing
    
[^172]: MARS：面向知识图谱问答的多跳自适应检索与SPARQL生成

    MARS: Multi-hop Adaptive Retrieval and SPARQL Generation for KGQA

    [https://arxiv.org/abs/2607.14561](https://arxiv.org/abs/2607.14561)

    MARS提出了一种无需微调的知识图谱问答方法，通过结构化的多跳自适应检索过程，在每一步决定继续图遍历还是生成最终SPARQL查询，从而根据问题自适应调整检索深度。

    

    大型语言模型（LLMs）已展现出强大的推理能力，但其产生幻觉的倾向限制了它们在需要最新且有依据信息的知识密集型任务中的可靠性。将知识图谱（KGs）与大型语言模型相结合，有助于利用无需昂贵微调即可持续更新的显式符号知识，同时受益于快速发展的LLM推理能力。我们提出了MARS，一种无需模型微调的可扩展知识图谱问答（KGQA）方法。MARS不依赖于开放式的智能体探索，而是执行一种结构化的检索过程，将问题实体链接到知识图谱，并迭代检索相关的下一跳信息。在每个步骤中，MARS决定是继续图遍历还是生成最终的SPARQL查询，使模型能够根据问题自适应调整检索深度，同时保持整个流程更加可预测。

    arXiv:2607.14561v3 Announce Type: replace  Abstract: Large language models (LLMs) have demonstrated strong reasoning performance, but their tendency to hallucinate limits their reliability in knowledge-intensive tasks requiring up-to-date and grounded information. Combining knowledge graphs (KGs) with LLMs facilitates the use of explicit symbolic knowledge that can be continuously updated without costly fine-tuning, while benefiting from rapidly advancing LLM reasoning. We propose MARS, a scalable knowledge graph question answering (KGQA) approach that requires no model fine-tuning. Rather than relying on open-ended agentic exploration, MARS performs a structured retrieval procedure that links question entities to the KG and iteratively retrieves relevant next-hop information. At each step, MARS decides whether to continue graph traversal or to generate the final SPARQL query, allowing the model to adapt the retrieval depth to the question while keeping the overall pipeline more predic
    
[^173]: 探索小语言模型在生物医学数据到文本生成中的后训练对齐：以药品说明书为例的案例研究

    Exploring Post-Training Alignment of Small Language Models for Biomedical Data-to-Text Generation: A Case Study of Medication Leaflet

    [https://arxiv.org/abs/2607.13430](https://arxiv.org/abs/2607.13430)

    本研究系统比较了SFT、DPO、ORPO和GRPO四种后训练对齐方法在小语言模型药品说明书生成任务上的效果，并构建了FDALeaflets数据集以评估跨监管体系的零样本泛化能力，发现对齐后的小语言模型在多项指标上超越了GPT-5。

    

    将结构化的生物医学信息转化为准确、患者友好的叙述文本，仍然是一项具有挑战性的生物医学数据到文本（D2T）任务。尽管小语言模型（SLM）因其更低的计算成本、更易于部署和更好的数据治理而在医疗健康应用中颇具吸引力，但后训练对齐对生物医学D2T生成的影响尚未得到系统性研究。在本工作中，我们使用Qwen-2.5模型进行药品说明书生成，比较了四种对齐方法：监督微调（SFT）、直接偏好优化（DPO）、优势比偏好优化（ORPO）和群体相对策略优化（GRPO）。我们进一步引入了FDALeaflets数据集，该数据集来源于OpenFDA药品标签，用于在零样本设置下评估跨监管体系的泛化能力。经过对齐筛选的小语言模型在多项基于参考文本的指标上超越了GPT-5，而GPT-5在幻觉控制方面表现更优。

    arXiv:2607.13430v2 Announce Type: replace  Abstract: Translating structured biomedical information into accurate, patient-friendly narratives remains a challenging biomedical data-to-text (D2T) task. While small language models (SLMs) are attractive for healthcare applications because of their lower computational cost, easier deployment, and improved data governance, the impact of post-training alignment on biomedical D2T generation has not been systematically studied. In this work, we compare four alignment methods: supervised fine-tuning (SFT), direct preference optimization (DPO), odds ratio preference optimization (ORPO), and group relative policy optimization (GRPO), using Qwen-2.5 models for medication leaflet generation. We further introduce FDALeaflets, a dataset derived from OpenFDA drug labels, to evaluate cross-regulatory generalization in a zero-shot setting. Selected aligned SLMs outperform GPT-5 on several reference-based metrics, whereas GPT-5 achieves better hallucinati
    
[^174]: 违反格赖斯准则风险自负：大语言模型在认知不对称下难以实现语用合作

    Flout at Your Own Risk: LLMs Struggle with Pragmatic Cooperativity Under Epistemic Asymmetry

    [https://arxiv.org/abs/2607.11053](https://arxiv.org/abs/2607.11053)

    本文形式化了“协作性认知不对称”概念并证明完全可执行的指令需遵循格赖斯合作原则，同时通过实证评估发现大语言模型在信息不完整的多方协作中无论作为说话者还是倾听者均存在语用沟通缺陷，且格赖斯准则可预测其不同失败模式。

    

    富有成效的协作需要对私有信息的有效沟通。大语言模型（LLM）作为“协作者”的使用日益增多，这引发了关于它们在多大程度上具备这些语用能力的问题。在本文中，我们首次对大语言模型在部分信息条件下的多方协作任务中的语用推理能力进行了新颖的探究。我们首先形式化定义了“协作性认知不对称”这一概念，并证明在此情境下，完全可执行的指令必须遵循格赖斯合作原则。随后，我们实证评估了多种大语言模型作为说话者和倾听者进行合作的能力，涵盖提示工程与后训练等多种策略。结果表明，尽管大语言模型在协作环境中表现出一定的语用能力，但在信息不完整的语用沟通中仍然面临挑战，且格赖斯准则能够预测其不同的失败模式。

    arXiv:2607.11053v2 Announce Type: replace  Abstract: Fruitful collaborations require effective communication of privately-held information. The increasing use of LLMs as "collaborators" raises questions about the extent to which they exhibit these pragmatic capabilities. In this paper, we perform a novel investigation into the pragmatic reasoning capabilities of LLMs in a multiparty collaborative task under partial information conditions. We first formalize a notion of collaborative epistemic asymmetry and prove that fully actionable instructions in this context requires adherence to Grice's cooperative principle. We then empirically assess various LLMs' abilities to act cooperatively as both speakers and listeners, including both prompting and post-training strategies. Our results show that while LLMs exhibit certain pragmatic capabilities in collaborative settings, they still face challenges in pragmatic communication with incomplete information, and that Grice's maxims predict disti
    
[^175]: 从边缘分布构建树：基于因子化先验的自回归草稿生成

    Trees from Marginals: Autoregressive drafting with factorized priors

    [https://arxiv.org/abs/2607.06763](https://arxiv.org/abs/2607.06763)

    本文提出Weaver，一种从因子化草稿模型的top-K边缘分布构建提议树的轻量级自回归适配器，恢复了token间的条件依赖，并结合无需回滚的树验证算法与优化的CUDA内核，相比自回归解码实现了4.08倍加速。

    

    投机解码通过以计算换取单次前向传播中额外生成的token，极大地提升了自回归语言模型的交互性。因子化草稿模型尤其高效，因为它们可以并行预测未来token的边缘分布，但其独立性假设导致随着投机预算的增长，接受率急剧下降。我们分析了这一局限性，并提出了Weaver——一个轻量级的自回归适配器，它从因子化草稿模型的top-K边缘分布中构建提议树。Weaver恢复了提议token之间的条件依赖关系，同时避免了全词表投影计算。为了支持对具有门控Delta Net（Gated Delta Net）层的模型进行快速验证，我们推导了一种无需回滚的树验证算法，并在SGLang中实现了优化的CUDA内核。通过结合这些模型与系统层面的贡献，我们相比自回归解码实现了4.08倍的加速。

    arXiv:2607.06763v3 Announce Type: replace-cross  Abstract: Speculative decoding greatly increases the interactivity of autoregressive language models by trading off computation for extra tokens generated in a single forward pass. Factorized draft models are especially efficient because they predict future-token marginals in parallel, but their independence assumption causes acceptance rates to degrade sharply as the speculative budget grows. We analyze this limitation and introduce Weaver, a lightweight autoregressive adapter that constructs proposal trees from the top-K marginals of a factorized drafter. Weaver restores conditional dependencies between proposed tokens while avoiding a full-vocabulary projection. To support fast verification for models with Gated Delta Net layers, we derive a rollback-free tree-verification algorithm and implement optimized CUDA kernels in SGLang. By combining these model and systems contributions we achieve a 4.08-fold speedup over autoregressive deco
    
[^176]: 何时使用工具能提升有限精度循环模型的表达能力？

    When Does Tool Use Increase the Expressive Power of Finite-Precision Recurrent Models?

    [https://arxiv.org/abs/2607.06155](https://arxiv.org/abs/2607.06155)

    本文证明了工具使用提升模型表达能力的边界条件：有限精度循环模型调用有界状态的工具后仍是有限状态系统，但只要接入一条支持本地读/写/移动命令的无界磁带即可达到图灵完备。

    

    现代序列模型越来越多地被部署为智能体，将标记生成与调用外部工具交替进行。我们研究这种工具访问何时能提升计算表达能力。我们将任何固定的有限精度循环序列模型（包括具有 $B$ 比特内部状态的有限精度状态空间模型（SSMs））建模为一个确定性有限状态控制器，该控制器通过有限的命令/观测接口与神谕交互，并在所分析的两种情形之间建立了分离。首先，每个具有记忆集 $M$ 的有限状态有界接口工具都可以以最多 $\lceil\log_2|M|\rceil+O(1)$ 个循环比特的代价被内化，因此组合系统仍然是有限状态的。其次，一条仅支持本地 $\mathtt{read}$/$\mathtt{write}$/$\mathtt{move}$ 命令的单一无界磁带就足以实现图灵完备性：每个具有状态集 $Q$ 和磁带字母表 $\Gamma$ 的单带图灵机都可以被模拟（摘要在此处截断）。

    arXiv:2607.06155v2 Announce Type: replace-cross  Abstract: Modern sequence models are increasingly deployed as agents that interleave token generation with calls to external tools. We study when such tool access increases computational expressivity. We model any fixed finite-precision recurrent sequence model, including finite-precision state-space models (SSMs) with $B$ bits of internal state, as a deterministic finite-state controller interacting with an oracle through a finite command/observation interface, and establish a separation between the two regimes analyzed. First, every finite-state bounded-interface tool with memory set $M$ can be internalized at a cost of at most $\lceil\log_2|M|\rceil+O(1)$ recurrent bits, so the combined system remains finite-state. Second, a single unbounded tape with only local $\mathtt{read}$/$\mathtt{write}$/$\mathtt{move}$ commands suffices for Turing completeness: every single-tape Turing machine with state set $Q$ and tape alphabet $\Gamma$ is s
    
[^177]: 用于多表格问答的潜在桥接

    Latent Bridges for Multi-Table Question Answering

    [https://arxiv.org/abs/2606.28916](https://arxiv.org/abs/2606.28916)

    该论文提出GRAB框架，将关系数据编码为异构图，并通过查询条件化的潜在标记桥接到完全冻结的LLM（仅训练91M参数的轻量模块），显著提升了多表格问答的性能。

    

    我们提出了GRAB，一个用于表格问答的构造器-编码器-桥接流水线。我们的方法将关系数据提升为异构图，通过消息传递进行编码，并通过一小组以查询为条件的潜在标记将信号传递给大语言模型（LLM）。这为LLM提供了紧凑且与任务相关的结构化表示，同时结合了扁平化的文本。关键在于，LLM保持严格冻结以保留其通用推理能力；我们仅训练轻量级的图编码器和潜在桥接（9100万参数），使整个流水线能够高效训练。我们的流水线显著提升了关系型问答的性能，在要求苛刻的多表格设置中收益最大，为连接关系深度学习与大语言模型提供了一种高效且有原则的方法。

    arXiv:2606.28916v2 Announce Type: replace  Abstract: We introduce GRAB, a constructor-encoder-bridge pipeline for table question answering. Our method lifts relational data into an heterogeneous graph, encodes it via message passing, and transfers the signals to an LLM through a small set of query-conditioned latent tokens. This provides the LLM with a compact, task-relevant structural representation together with the flattened text. Crucially, the LLM remains strictly frozen to preserve its general reasoning capabilities; we train only the lightweight graph encoder and latent bridge (91M parameters), allowing the entire pipeline to be trained efficiently. Our pipeline significantly improves performance on relational Question Answering, with the largest gains in demanding multi-table settings, offering an efficient, principled way to connect relational deep learning with LLMs.
    
[^178]: 面向大语言模型记忆化缓解的输出向量编辑

    Output Vector Editing for Memorization Mitigation in Large Language Models

    [https://arxiv.org/abs/2606.18767](https://arxiv.org/abs/2606.18767)

    提出输出向量编辑（OVE）方法，通过两阶段定位神经元并用全词表最小二乘更新编辑其输出向量（而非将激活置零），有效抑制大语言模型复现记忆的训练文本，同时更好地保留模型其他能力。

    

    大语言模型可能会复现训练文本，从而带来隐私、版权和安全风险。我们提出了输出向量编辑（OVE），这是一种事后处理方法，用于抑制模型从给定前缀复现出的文本延续内容。在叠加假设下，一个神经元可能对多个特征都有贡献，然而一些神经元层面的缓解方法通过将定位到的神经元激活置零来进行抑制，这会移除每个神经元的全部贡献。OVE 则改为编辑神经元的输出向量，该方法结合了两个阶段的 MLP 层与神经元定位，以及全词表最小二乘更新。其多锚点变体 OVE-MA 能够自适应地选择那些相对于所需权重变化而言预期可提供更强抑制效果的延续位置。我们在四个模型（参数量从 3.6 亿到 70 亿）上评估了恢复性与局部性，使用了经自回归验证的延续内容，并采用贪心解码、束搜索和采样进行恢复测试。

    arXiv:2606.18767v2 Announce Type: replace  Abstract: Large language models can reproduce training text, creating privacy, copyright, and security risks. We propose Output Vector Editing (OVE), a post-hoc method for suppressing text continuations that a model reproduces from a given prefix. Under superposition, a neuron can contribute to multiple features. Yet some neuron-level mitigation methods suppress localized neurons by zeroing their activations, removing each neuron's entire contribution. OVE instead edits their output vectors, combining (i) two-stage localization of MLP layers and neurons with (ii) full-vocabulary least-squares updates. Its multi-anchor variant, OVE-MA, adaptively selects continuation positions expected to provide stronger suppression relative to the required weight change. We evaluate recovery and locality across four models (360M--7B parameters), using autoregressively validated continuations and recovery tests with greedy decoding, beam search, and sampling. 
    
[^179]: EHRNote-ChatQA：面向纵向出院总结的基于证据的多轮临床问答基准

    EHRNote-ChatQA: A Benchmark for Evidence-Grounded Multi-Turn Clinical Question Answering over Longitudinal Discharge Summaries

    [https://arxiv.org/abs/2606.15735](https://arxiv.org/abs/2606.15735)

    该论文提出了首个基于患者多篇出院总结的证据支撑型多轮临床问答基准EHRNote-ChatQA，利用去标识化的MIMIC-IV数据构建，真实模拟了医学专家迭代综合信息并验证证据的临床审阅场景，弥补了现有基准在多轮问答与证据溯源评估上的不足。

    

    出院总结是至关重要的临床文档，包含了患者整个住院期间的背景信息，医学专家通常会定期审阅这些文档，以用于患者再入院、持续护理和诊断决策。在审阅过程中，医学专家往往需要迭代地综合多篇出院总结中的信息，同时验证支持每个答案的证据。尽管大语言模型（LLMs）正越来越多地被探索应用于临床问答，但现有的基准测试并未充分反映这一场景：它们通常评估考试式的医学知识，或仅专注于单轮问答，且缺乏充分的证据溯源评估。我们提出了EHRNote-ChatQA，这是首个针对患者多篇出院总结的基于证据的多轮临床问答基准。该基准基于去标识化的MIMIC-IV出院总结构建，包含967个患者级别的……

    arXiv:2606.15735v3 Announce Type: replace  Abstract: Discharge summaries are crucial clinical documents containing the context of a patient's overall hospital stay, and are routinely reviewed by medical experts for patient readmission, ongoing care, and diagnostic decision-making. When reviewing them, medical experts often must iteratively synthesize information across multiple discharge summaries while verifying the evidence supporting each answer. Although large language models (LLMs) are increasingly explored for clinical question answering, existing benchmarks do not sufficiently reflect this setting: they often evaluate exam-style medical knowledge or focus on single-turn question answering with limited evidence-grounding evaluation. We introduce EHRNote-ChatQA, the first benchmark for evidence-grounded multi-turn clinical question answering over patients' multiple discharge summaries. Built from de-identified MIMIC-IV discharge summaries, EHRNote-ChatQA contains 967 patient-level
    
[^180]: GENIE：一种细粒度的新颖性度量方法

    GENIE: A Fine-Grained Measure for Novelty

    [https://arxiv.org/abs/2606.12790](https://arxiv.org/abs/2606.12790)

    本文提出细粒度评估指标GENIE，基于任务特定特征相对于响应群体来衡量模型生成内容的新颖性，克服了整体性指标难以捕捉新颖性高维特性的局限，并可有效评估创造力提升方法的效果。

    

    大型语言模型在各类任务中持续表现出创造力和多样性的缺乏。以往的工作主要集中于探讨模型是否能够生成有创意的输出。本文中，我们旨在研究新颖性，并以任务特定的方式探究是什么使模型生成的内容具有新颖性或不具有新颖性。我们提出了一种细粒度的评估指标GENIE，用于基于响应群体、沿任务特定特征来衡量响应的新颖性。我们表明，与GENIE不同，整体性指标难以捕捉新颖性的高维特性，并且无法洞察其针对的具体属性。最后，我们使用GENIE来衡量针对创造力的缓解方法的有效性，以更好地理解这些方法可以在哪些方面提升新颖性。

    arXiv:2606.12790v2 Announce Type: replace  Abstract: Large Language Models have consistently demonstrated a lack of creativity and diversity across tasks. Prior work has focused on addressing whether models are capable of generating creative outputs. Here, we aim to consider novelty and investigate what makes model-generated content novel or not novel in a task-specific manner. We propose a fine-grained evaluation metric GENIE to measure the novelty of responses along task-specific features with respect to a population of responses. We show that unlike GENIE, holistic metrics struggle to capture the high-dimensionality of novelty and do not provide insight on which properties they target. Finally, we use GENIE to measure the effectiveness of mitigation methods that address creativity to better understand where these methods can improve novelty.
    
[^181]: 重新思考疾病预测中的合成推理监督

    Rethinking Synthetic Rationale Supervision for Disease Prediction

    [https://arxiv.org/abs/2606.10279](https://arxiv.org/abs/2606.10279)

    该研究发现，在疾病预测中使用合成推理理由进行“推理优先”监督微调反而会持续且显著地损害预测性能，且这种退化跨模型系列、数据规模和任务普遍存在，并非由推理理由质量不佳所致。

    

    在监督微调中使用合成推理理由被广泛认为能够提升临床预测效果，因为它不仅教会模型“预测什么”，还教会模型“为什么这样预测”。我们在基于纵向健康病史的五年阿尔茨海默病及相关痴呆预测任务中检验了这一假设。在该设置中，我们使用正确标签生成训练推理理由，然后微调模型使其先输出推理理由、再输出标签。在一项涵盖504种配置的大规模对照实验中，我们发现“推理优先”的监督微调相比“仅标签”微调，始终且显著地损害了预测性能。这种性能退化在不同模型系列和数据规模上均持续存在，且使用面向推理的基础模型也无法解决。在一项采用固定协议的心血管疾病验证实验中重现了类似的发现。至关重要的是，这一失败并非由推理理由质量不佳所解释：人类专家标注证实了生成……（摘要原文在此处截断）

    arXiv:2606.10279v2 Announce Type: replace-cross  Abstract: Supervised fine-tuning with synthetic rationales is widely assumed to improve clinical prediction by teaching models not only what to predict but why. We test this assumption on five-year Alzheimer's disease and related dementias (ADRD) prediction from longitudinal health histories. In this setting, we generate training rationales using the correct labels, then fine-tune models to output a rationale followed by a label. Across a large-scale controlled experiment of 504 configurations, we find that Rationale First SFT consistently and substantially hurts predictive performance relative to Label Only fine-tuning. The degradation persists across model families and data scales, and is not resolved by using a reasoning-oriented base model. A fixed-protocol cardiovascular disease (CVD) validation reproduces similar findings. Crucially, the failure is not explained by poor rationale quality: human expert annotation confirms that the g
    
[^182]: 大型语言模型中的语言能产性：模型会进行强制解读，但不会进行先占推断

    Linguistic Productivity in Large Language Models: Models Coerce, but do not Preempt

    [https://arxiv.org/abs/2606.02953](https://arxiv.org/abs/2606.02953)

    该研究发现大型语言模型具备促进语言能产性的“强制解读”推理能力，但缺乏人类语言使用者用来约束能产性的“先占”机制。

    

    基于用法的语法理论通过两种互补的推理策略来解释语言使用者理解和产出新颖词语组合的能力。在强制解读的情形中，当语言使用者遇到与其语法语境不匹配的词时，会以化解冲突的方式对该词进行重新解读。在先占的情形中，使用者会推断某一原本合理的结构不可用或不受偏好，因为它没有出现在预期会出现该结构的语境中。大型语言模型同样基于用法，因为语言结构是通过接触海量文本而习得的。本文检验强制与先占这两种相反的力量是否同样在大型语言模型中促进并约束语言能产性。我们在多种规模和架构的模型上证明，较大的模型在强制解读的情形中能够识别构式的能产性，即使……

    arXiv:2606.02953v2 Announce Type: replace  Abstract: Usage-based theories of grammar explain language users' ability to interpret and produce novel combinations of words through two complementary inferential strategies. In cases of coercion, a language user, when encountering a word that is mismatched to its grammatical context, reinterprets that word in a way that resolves the conflict. In cases of preemption, the user infers that an otherwise plausible structure is unavailable or dispreferred because it fails to occur in contexts where it would be expected. Large Language Models are also usage-based, in the sense that the structures of language are learned through exposure to vast amounts of text. Here, we test whether or not the opposing forces of coercion and preemption also encourage and constrain linguistic productivity in LLMs. We demonstrate across models of various sizes and architectures that larger models recognize constructional productivity in cases of coercion, even when 
    
[^183]: CanLegalRAGBench：评估加拿大判例法上的检索增强生成

    CanLegalRAGBench: Evaluating Retrieval-Augmented Generation on Canadian Case Law

    [https://arxiv.org/abs/2605.30497](https://arxiv.org/abs/2605.30497)

    本文提出了CanLegalRAGBench，一个基于现实查询和专家标注的加拿大判例法问答基准，揭示了检索设计敏感性、开源模型竞争力及自动评估的局限性。

    

    基于检索增强生成（RAG）的法律助手越来越受欢迎，但大语言模型的幻觉问题仍然是一个关键问题，可能损害司法公正。虽然已有基准测试用于评估进展，但许多依赖于合成查询而非现实法律场景。此外，加拿大法律在现有评估中代表性不足。为解决这一差距，我们引入了CanLegalRAGBench，一个基于现实查询和专家标注答案的加拿大法律问答基准，其答案以判例法为依据。我们的评估显示，检索性能对设计选择敏感，且开源嵌入模型与闭源模型具有竞争力。然而，评估也揭示了自动评估的局限性，即会因检索到替代相关文档而惩罚系统。我们还发现，生成的答案往往偏离黄金标准响应，要么出现幻觉，要么产生过于详细或无关的内容。

    arXiv:2605.30497v2 Announce Type: replace  Abstract: RAG-based legal assistants have been growing in popularity, but LLM hallucinations remain a key issue and potentially undermines justice. While benchmarks have been developed to evaluate progress, many rely on synthetic queries rather than realistic legal scenarios. Moreover, Canadian law remains underrepresented in existing evaluations. To address this gap, we introduce CanLegalRAGBench, a Canadian legal QA benchmark based on realistic queries and expert-annotated answers grounded in case law. Our evaluation shows that retrieval performance is sensitive to design choices and that open-source embedding models are competitive with closed source models. However, it also reveals the limitation of automatic evaluations that penalize systems for retrieving alternative relevant documents. We also find that generated answers often diverge from gold responses, either with hallucinations or by producing overly detailed or irrelevant content, 
    
[^184]: 在自然语言上训练的神经后继表示中涌现出词类与构式类结构

    Word-Class and Construction-Like Structure Emerges in Neural Successor Representations Trained on Natural Language

    [https://arxiv.org/abs/2605.24585](https://arxiv.org/abs/2605.24585)

    在自然语言上训练的神经后继表示无需任何词性监督，即可自发涌现出词类边界以及跨越传统类别界限的构式类语言结构。

    

    神经语言模型通常基于下一词元预测进行训练，然而语言结构跨越多个时间尺度。后继表示通过编码对未来状态的折扣分布，将这一预测视野显式化。在本文中，我们探究这种预测性表示是否不仅能恢复词类，还能从自然语言中恢复更精细的功能性与构式类结构。一个在 WikiText-103 上训练的残差网络在三个预测视野上预测后继表示分布，且完全无需词性监督。在最短视野下，无监督聚类能够稳健地恢复出名词、动词和形容词，而聚类之间的有向转移重现了人们熟知的句法不对称性。在更精细的分辨率下以及跨越 13 个词性类别时，同样的几何结构揭示了跨越类别边界的语义-功能分组，以及能够追踪日期、计量和称号-名称等候选共现结构的有向关系。

    arXiv:2605.24585v3 Announce Type: replace  Abstract: Neural language models are typically trained on next-token prediction, although linguistic structure spans multiple temporal scales. Successor representations (SRs) make this horizon explicit by encoding discounted distributions over future states. Here, we ask whether such predictive representations can recover not only word classes, but also finer functional and construction-like structure from natural language. A residual network trained on WikiText-103 predicts SR distributions at three horizons without part-of-speech supervision. At the shortest horizon, unsupervised clustering robustly recovers nouns, verbs, and adjectives, while directed inter-cluster transitions reproduce familiar syntactic asymmetries. At finer resolutions and across 13 part-of-speech categories, the same geometry reveals semantic-functional groupings that cross category boundaries and directed relations tracing candidate date, measurement, and title-name co
    
[^185]: MedicalBench：面向改进医学概念抽取的大语言模型评估

    MedicalBench: Evaluating Large Language Models Toward Improved Medical Concept Extraction

    [https://arxiv.org/abs/2605.20197](https://arxiv.org/abs/2605.20197)

    MedicalBench是一个具有证据扎根的医学概念抽取基准，将概念抽取转化为笔记-概念对验证任务，专门评估大语言模型对隐含医学概念的推理能力。

    

    从电子健康记录中进行医学概念抽取是许多下游应用的基础，但这项任务仍然具有挑战性，因为具有医学意义的概念在医学叙述中经常是隐含的而非明确陈述的。现有的包含人工标注证据片段的基准凸显了将抽取的概念扎根于医学文本的重要性。然而，这些基准主要关注明确陈述的概念，而忽视了隐含概念。我们提出了MedicalBench，一个具有证据扎根的医学概念抽取基准，用于评估隐含的医学推理能力。MedicalBench将医学概念抽取表述为对医学笔记-概念对的验证任务，并结合句子级的证据识别。该数据集基于MIMIC-IV出院小结和人工验证的ICD-10编码构建，通过多阶段大语言模型（LLM）分诊流程筛选，并辅以医学专家标注。

    arXiv:2605.20197v2 Announce Type: replace  Abstract: Medical concept extraction from electronic health records underpins many downstream applications, yet remains challenging because medically meaningful concepts are frequently implied rather than explicitly stated in medical narratives. Existing benchmarks with human-annotated evidence spans underscore the importance of grounding extracted concepts in medical text. However, they predominantly focus on explicitly stated concepts instead of implicit concepts. We present MedicalBench, a benchmark for medical concept extraction with evidence grounding that evaluates implicit medical reasoning. MedicalBench formulates medical concept extraction as a verification task over medical note-concept pairs, coupled with sentence-level evidence identification. Built from MIMIC-IV discharge summaries and human-verified ICD-10 codes, the dataset is curated through a multi-stage large language model (LLM) triage pipeline followed by medical annotation
    
[^186]: 从文本到语音：一个可复现、可验证的工具调用大语言模型智能体评估框架

    From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents

    [https://arxiv.org/abs/2605.15104](https://arxiv.org/abs/2605.15104)

    提出一个无需重新标注、可复现且可验证的框架，利用文本转语音、说话人变化和环境噪声将已有文本工具调用基准转换为可控的音频评估，并揭示全模态模型的工具调用性能高度依赖模型和任务，且从文本到语音存在1.8至4.8分的性能下降。

    

    语音智能体日益需要基于语音进行可靠的工具调用，而现有的主流工具调用基准仍然基于文本。我们研究能否在无需重新标注工具模式和标准答案标签的情况下，将经过验证的文本基准转换为可控的基于音频的工具调用评估。我们的框架与数据集无关，通过文本转语音（TTS）、说话人变化和环境噪声来创建成对的文本-音频实例，同时保留原始数据集的标注。基于对7个全模态模型在Confetti和When2Call的音频转换版本上的广泛评估，我们的框架表明性能高度依赖于具体模型和任务：Gemini-3.1-Flash-Live在Confetti上获得最高分（70.4），而GPT-Realtime-1.5在When2Call上表现最佳（71.9）。在Confetti上，文本到语音的性能差距从Qwen3-Omni的1.8分到GPT-Realtime-1.5的4.8分不等。对失败案例的针对性分析显示……（摘要原文在此处截断）

    arXiv:2605.15104v3 Announce Type: replace  Abstract: Voice agents increasingly require reliable tool use from speech, whereas prominent tool-calling benchmarks remain text-based. We study whether verified text benchmarks can be converted into controlled audio-based tool calling evaluations without re-annotating the tool schema and gold labels. Our dataset-agnostic framework uses text-to-speech, speaker variation, and environmental noise to create paired text-audio instances while preserving the original dataset annotations. Based on extensive evaluation of 7 omni-modal models on audio-converted versions of Confetti and When2Call, our framework demonstrates that the performance is strongly model- and task-dependent: Gemini-3.1-Flash-Live obtains the highest Confetti score (70.4), whereas GPT-Realtime-1.5 performs best on When2Call (71.9). On Confetti, the text-to-voice gap ranges from 1.8 points for Qwen3-Omni to 4.8 points for GPT-Realtime-1.5. A targeted analysis of failure cases demo
    
[^187]: NodeSyn：面向AI评估的社会对齐合成数据

    NodeSynth: Socially Aligned Synthetic Data for AI Evaluation

    [https://arxiv.org/abs/2605.14381](https://arxiv.org/abs/2605.14381)

    NodeSyn通过将微调的分类体系生成器与可信文献来源和敏感属性相结合，生成反映真实社会背景的敏感领域评估数据，其诱发的模型失败率达到通用合成基线的3.1倍、真实人类查询的1.6倍，显著优于现有合成数据方法。

    

    生成式人工智能的最新进展为模型评估所需的大规模合成数据生成提供了便利。然而，若缺乏针对性的方法，这些数据集往往缺乏敏感领域所需的社会技术层面的细微差别。我们提出了NodeSyn，这是一种将经过微调的分类体系生成器（TaG）与可信的已发表文献来源以及提取的敏感属性相结合的方法，用于生成反映真实社会背景和细微危害的查询。在使用经人工验证的自动评估器，于医疗建议和自我伤害两个领域对十个开源及专有大语言模型进行评估时，NodeSyn在10个模型中的8个上于每个领域均排名第一。NodeSyn数据集引发的平均失败率是通用合成基线的3.1倍，是结构化合成基准的2.4倍，是真实人类撰写查询的1.6倍。消融研究证实，在控制结构混杂因素的情况下，我们细粒度的第3级分类体系…

    arXiv:2605.14381v3 Announce Type: replace-cross  Abstract: Recent advancements in generative AI facilitate large-scale synthetic data generation for model evaluation. However, without targeted approaches, these datasets often lack the sociotechnical nuance required for sensitive domains. We introduce NodeSyn, a methodology that pairs a fine-tuned taxonomy generator (TaG) with credible published sources and extracted sensitive attributes to generate queries reflecting authentic societal contexts and nuanced harms. Evaluated across ten open-weight and proprietary LLMs in Medical Advice and Self-Harm using human-validated autoraters, NodeSyn ranked first on 8 of 10 models in each domain. NodeSyn datasets induced average failure rates 3.1 times higher than generic synthetic baselines, 2.4 times higher than structured synthetic benchmarks, and 1.6 times higher than authentic human-authored queries. Ablation studies confirm that, controlling for structural confounds, our granular Level 3 tax
    
[^188]: PersonalAI 2.0：通过规划机制增强个性化LLM智能体的知识图谱遍历与检索

    PersonalAI 2.0: Enhancing knowledge graph traversal/retrieval with planning mechanism for Personalized LLM Agents

    [https://arxiv.org/abs/2605.13481](https://arxiv.org/abs/2605.13481)

    PersonalAI 2.0通过引入动态多阶段查询处理流水线的规划机制，实现了由实体、图顶点和线索查询引导的自适应迭代式知识图谱检索，在多个问答基准上显著提升了生成答案的事实准确性。

    

    我们提出了PersonalAI 2.0（PAI-2），这是一个新颖的框架，旨在通过整合外部知识图谱（KG）来增强基于大语言模型（LLM）的系统。所提出的方法通过引入动态的、多阶段的查询处理流水线，解决了现有图谱检索增强生成（GraphRAG）方法的关键局限性。PAI-2设计的核心在于其能够执行自适应的、迭代式的信息搜索，该搜索由提取的实体、匹配的图顶点以及生成的线索查询所引导。在五个基准数据集（Natural Questions、TriviaQA、HotpotQA、2WikiMultihopQA和MuSiQue）上进行的评估表明，与同类方法（LightRAG、RAPTOR、HippoRAG 2和PAI-1）相比，该方法提升了生成答案的事实准确性。PAI-2在2WikiMultihopQA和MuSiQue基准上通过LLM-as-a-Judge评估平均取得了9%的提升，并在TriviaQA和HotpotQA上达到了与HippoRAG 2相当的准确率。

    arXiv:2605.13481v2 Announce Type: replace  Abstract: We introduce PersonalAI 2.0 (PAI-2), a novel framework designed to enhance LLM-based systems through integration of external knowledge graphs (KGs). The proposed approach addresses key limitations of existing Graph Retrieval-Augmented Generation (GraphRAG) methods by incorporating a dynamic, multistage query-processing pipeline. The central point of the PAI-2 design is its ability to perform adaptive, iterative information search, guided by extracted entities, matched graph vertices, and generated clue-queries. An evaluation conducted on five benchmarks (Natural Questions, TriviaQA, HotpotQA, 2WikiMultihopQA, and MuSiQue) demonstrates an improvement in the factual correctness of generated answers compared to analogue methods (LightRAG, RAPTOR, HippoRAG 2, and PAI-1). PAI-2 achieves a 9% average gain by LLM-as-a-Judge on the 2WikiMultihopQA and MuSiQue benchmarks, and attains accuracy comparable to HippoRAG 2 on the TriviaQA and Hotpo
    
[^189]: 扩散语言模型中安全生成的自适应引导与重掩码方法

    Adaptive Steering and Remasking for Safe Generation in Diffusion Language Models

    [https://arxiv.org/abs/2605.13043](https://arxiv.org/abs/2605.13043)

    该论文提出一种推理时防御框架，通过门控方向自适应调节引导强度，并结合安全感知重掩码机制，动态引导扩散语言模型的去噪轨迹远离有害内容，同时保持生成质量。

    

    扩散语言模型（DLMs）通过迭代去噪和双向生成，为自回归语言模型提供了一种有前景的替代方案。然而，其迭代生成过程引入了独特的安全漏洞，因为有害内容可能在任意位置出现并在后续去噪步骤中持续存在。现有的防御方法依赖于固定干预或激进的重掩码，这限制了对去噪轨迹的自适应控制，并可能降低生成质量。我们提出了一种推理时防御框架，将自适应安全引导与安全感知重掩码相结合。我们的方法使用门控方向根据当前去噪状态持续调整引导强度，并将引导方向应用于被掩码的位置，以引导后续预测朝着更安全的轨迹发展。我们的方法还在第一个生成块之后采用轻量级响应检测器来识别……

    arXiv:2605.13043v3 Announce Type: replace  Abstract: Diffusion Language Models(DLMs) provide a promising alternative to autoregressive language models through iterative denoising and bidirectional generation. However, their iterative generation process introduces distinct safety vulnerabilities because harmful content can emerge at arbitrary positions and persist across subsequent denoising steps. Existing defenses rely on fixed interventions or aggressive remasking, which limits adaptive control over denoising trajectories and can degrade generation quality. We propose an inference-time defense framework that combines adaptive safety steering with safety-aware remasking. Our method uses a gating direction to continuously adjust steering strength from the current denoising state and applies a steering direction to masked positions to guide subsequent predictions toward safer trajectories. Our method further employs a lightweight response detector after the first generation block to ide
    
[^190]: 通过结构化元认知在通用智能体中实现深度推理

    Deep Reasoning in General Purpose Agents via Structured Meta-Cognition

    [https://arxiv.org/abs/2605.11388](https://arxiv.org/abs/2605.11388)

    提出“深度推理”方法，利用形式化语言将元推理表示为可执行的分解，在推理时动态构建任务特定的脚手架，使LLM智能体能像人类一样灵活地在联想推理、形式化计算和递归求解等不同推理模式间切换。

    

    人类通过在不同推理模式之间灵活切换来直观地解决复杂问题：他们进行规划、执行、修改中间目标、通过联想判断消解歧义，并对明确定义的子问题应用形式化程序。当前的LLM智能体缺乏这种灵活性，因为它们的脚手架（scaffold）预先硬编码了此类推理决策。当其规定的结构与任务相匹配时，这些脚手架是有效的，但当解决任务需要调整推理结构本身时，它们就会变得脆弱。我们提出了深度推理（Deep Reasoning）——一种通过结构化元推理在推理时构建任务特定脚手架的方法。深度推理使用一种形式化语言，将元推理表示为对联想推理、形式化计算和递归子问题求解的可执行分解，使分解原则能够被编码为上下文示例，从而指导测试时的脚手架构建。

    arXiv:2605.11388v2 Announce Type: replace  Abstract: Humans intuitively solve complex problems by flexibly shifting among reasoning modes: they plan, execute, revise intermediate goals, resolve ambiguity through associative judgment, and apply formal procedures to well-specified subproblems. Current LLM agents lack this flexibility, as their scaffolds hard-code such reasoning decisions in advance. These scaffolds are effective when their prescribed structure matches the task, but brittle when solving the task requires adapting the structure of reasoning itself. We introduce Deep Reasoning -- an inference-time approach for constructing task-specific scaffolds through structured meta-reasoning. Deep Reasoning uses a formal language that represents meta-reasoning as executable decompositions over associative inference, formal computation, and recursive subproblem solving, enabling decomposition principles to be encoded as in-context examples that guide test-time scaffold construction. We 
    
[^191]: 从提示词到基于证据的翻译：面向日中翻译的RAG+提示词系统及其教学潜力

    From prompting to evidence-based translation: A RAG+prompt system for Japanese-Chinese translation and its pedagogical potential

    [https://arxiv.org/abs/2605.03387](https://arxiv.org/abs/2605.03387)

    该研究提出了一种RAG+提示词翻译系统，通过融入语言学分析和相似例句检索来增强GPT-4o，在不修改基础模型的前提下，将含名词修饰节结构的日中翻译BLEU得分提升了23.4%，并展示了其教学应用潜力。

    

    大型语言模型在资源丰富的语言对上表现出色，但对于包含名词修饰节结构（NMCC）的日中句子，其可靠性较低。本研究评估了一种检索增强生成（RAG+Prompt）翻译系统，该系统集成了语言学分析、基于嵌入的检索、提示词构建和大语言模型生成，且无需修改基础模型。分析模块输出A1（内接/外接名词修饰节类型）和A2（风险预测：词汇选择/名词修饰节处理/语序/文体/语域）；检索出的top-k=5个相似日中例句（基于L2距离）连同A1/A2信息被插入到增强的提示词中。实验使用GPT-4o和66句测试集，比较了六种知识库规模（0/100/200/500/1,000/2,000）。宏平均句子级BLEU（带简短惩罚的1-4-gram；区分大小写；中文按字符级计算）作为唯一评价指标。平均BLEU从知识库为0（禁用RAG）时的24.28提升至知识库为2,000时的29.96（+5.68；提升23.4%）。[摘要原文在此处截断]

    arXiv:2605.03387v2 Announce Type: replace  Abstract: Large language models perform well on high-resource pairs but are less reliable for Japanese-Chinese sentences containing noun-modifying clause constructions (NMCCs). This study evaluates a retrieval-augmented generation RAG+Prompt translation system that integrates linguistic analysis, embedding-based retrieval, prompt construction, and LLM generation without modifying the base model. The analysis module outputs A1 (inner vs. outer NMCC) and A2 (risk predictions: lexical choice/NMCC handling/word order/style/register); top-k = 5 similar Ja-Zh examples (L2 distance) and A1/A2 are inserted into an enhanced prompt. Using GPT-4o and a 66-sentence test set, we compare six knowledge-base sizes (0/100/200/500/1,000/2,000). Macro-averaged sentence-level BLEU (1-4-gram with brevity penalty; cased; Chinese at the character level) is the sole metric. Mean BLEU increases from 24.28 at 0 (RAG disabled) to 29.96 at 2,000 (+5.68; +23.4%). The upwa
    
[^192]: 探针鲁棒性的稳健评估：面向可靠OOD不确定性量化的启示

    A Robust Evaluation of Probe Robustness: Lessons for Reliable OOD Uncertainty Quantification

    [https://arxiv.org/abs/2604.11662](https://arxiv.org/abs/2604.11662)

    提出ProbeDrift系统性评估框架，通过训练2000多个不确定性探针揭示现有方法在远分布外设置下鲁棒性较差，且鲁棒性取决于分布内难以察觉的设计决策（如特征类型、聚合策略和训练信号）。

    

    最近的研究表明，大语言模型的隐藏状态中包含有助于不确定性估计的信号，这激发了人们对高效基于探针的方法日益增长的兴趣。然而，现有方法的鲁棒性究竟如何仍不清楚，因为先前的研究在差异很大的评估设置下得出了相互矛盾的结论。我们通过引入ProbeDrift来解决这一问题，这是一个针对监督式不确定性探针的系统性评估框架，涵盖了跨模型、任务和分布偏移的广泛OOD（分布外）设置。利用ProbeDrift，我们训练了超过2,000个探针以厘清关键设计选择的影响，结果表明当前方法在超出近OOD设置时鲁棒性较差。我们发现，鲁棒性由那些在分布内影响几乎不可察觉的设计决策所驱动，包括特征类型的选择、聚合策略以及训练信号。我们认为，稳健的不确定性估计需要……

    arXiv:2604.11662v2 Announce Type: replace  Abstract: Recent work has shown that the hidden states of large language models contain signals useful for uncertainty estimation, motivating a growing interest in efficient probe-based approaches. Yet it remains unclear how robust existing methods are, with prior work reporting conflicting conclusions under substantially different evaluation settings. We address this by introducing ProbeDrift, a systematic evaluation framework for supervised uncertainty probes covering a wide range of OOD settings across models, tasks, and distributional shifts. Using ProbeDrift, we train over 2,000 probes to disentangle the effect of key design choices, showing poor robustness of current methods beyond near-OOD settings. We find that robustness is driven by design decisions that have a largely invisible effect in-distribution, including the choice of feature type, aggregation strategy, and training signal. We argue that robust uncertainty estimation requires
    
[^193]: 步进信息性假设：熵动态何时能追踪大语言模型的推理正确性？

    The Stepwise Informativeness Assumption: When Do Entropy Dynamics Track Reasoning Correctness in LLMs?

    [https://arxiv.org/abs/2604.06192](https://arxiv.org/abs/2604.06192)

    本文提出步进信息性假设（SIA），从理论上刻画了熵降低何时能真实反映大语言模型推理的正确性进展，并指出这种对齐取决于训练机制将推理轨迹与正确答案耦合的方式，同时给出了该现象的可观测特征和失效情形。

    

    近期的研究使用基于熵的信号在多个表示层次上研究大语言模型的推理，但该领域在很大程度上仍停留在经验层面。从理论角度看，模型不确定性的降低并不一定反映其朝正确性的进展。在这项工作中，我们通过步进信息性假设（Stepwise Informativeness Assumption, SIA）来刻画何时可以预期这种对齐关系，该假设要求降低熵的推理前缀扩展在期望上增加分配给有效答案的概率。我们表明，下一词元预测能够将与答案相关的信息从训练数据传递到模型，但其本身并不能保证与正确性对齐。相反，我们提出这种对齐取决于训练机制如何将推理轨迹与正确答案相耦合。此外，我们假设了这一现象的可观测特征，并刻画了其预期的失效情形。（原文摘要在此处截断）

    arXiv:2604.06192v2 Announce Type: replace  Abstract: Recent work uses entropy-based signals at multiple representation levels to study reasoning in large language models, but the field remains largely empirical. From a theoretical perspective, reductions in model uncertainty do not necessarily reflect progress toward correctness. In this work, we characterize when such alignment can be expected through the Stepwise Informativeness Assumption (SIA), which requires entropy-reducing reasoning-prefix extensions increase the probability assigned to valid answers in expectation. We show that next-token prediction can transfer answer-relevant information from training data to the model, but does not, by itself, guarantee alignment with correctness. Instead, we propose that this alignment depends on how the training regime couples reasoning traces to correct answers. Furthermore, we hypothesize observable signatures of this phenomenon and characterize its expected failure regimes. We empirical
    
[^194]: AfrIFact：面向非洲语言的文化信息检索、证据提取与事实核查

    AfrIFact: Cultural Information Retrieval, Evidence Extraction and Fact Checking for African Languages

    [https://arxiv.org/abs/2604.00706](https://arxiv.org/abs/2604.00706)

    本文提出了AfrIFact数据集，覆盖十种非洲语言和英语的自动事实核查全流程（信息检索、证据提取与事实核查），并揭示现有嵌入模型缺乏跨语言检索能力、大语言模型在非洲语言事实验证上表现不佳，而少样本提示可将性能提升高达43%。

    

    评估网络上所做声明的真实性是一项复杂且重要的任务，具有现实世界的影响。当这些声明指向信息获取有限的社区，且内容涉及医疗健康和文化等问题时，其后果会更加严重，尤其是在低资源语言中。在这项工作中，我们介绍了AfrIFact，这是一个涵盖自动事实核查所需步骤（即信息检索、证据提取和事实核查）的数据集，覆盖十种非洲语言和英语。我们的评估结果表明，即使是最好的嵌入模型也缺乏跨语言检索能力，而且无论是在大型语料库还是单个文档中，文化和新闻类文档都比医疗健康领域的文档更容易被检索到。我们还表明，大语言模型在非洲语言方面缺乏稳健的多语言事实验证能力，而少样本提示可以将性能提升高达43%。

    arXiv:2604.00706v3 Announce Type: replace  Abstract: Assessing the veracity of a claim made online is a complex and important task with real-world implications. When these claims are directed at communities with limited access to information and the content concerns issues such as healthcare and culture, the consequences intensify, especially in low-resource languages. In this work, we introduce AfrIFact, a dataset that covers the necessary steps for automatic fact-checking (i.e., information retrieval, evidence extraction, and fact checking), in ten African languages and English. Our evaluation results show that even the best embedding models lack cross-lingual retrieval capabilities, and that cultural and news documents are easier to retrieve than healthcare-domain documents, both in large corpora and in single documents. We show that LLMs lack robust multilingual fact-verification capabilities in African languages, while few-shot prompting improves performance by up to 43% in Afriqu
    
[^195]: 大语言模型在线策略蒸馏综述

    A Survey of On-Policy Distillation for Large Language Models

    [https://arxiv.org/abs/2604.00626](https://arxiv.org/abs/2604.00626)

    这篇综述论文通过学生生成状态、教师信息与策略更新三者之间的相互作用，建立了在线策略蒸馏（OPD）的统一分析框架，并将其与自蒸馏、多教师集成、强化学习和交互式智能体等方向联系起来，系统阐释了不同方法的设计选择对能力迁移、训练效率和部署的影响。

    

    在线策略蒸馏在大语言模型的训练中，由教师模型对学生模型自身生成的轨迹提供指导，从而解决了固定示范与部署时实际遇到的交互历史之间的不匹配问题。随着其在语言模型训练中的应用不断扩展，不同的监督与优化形式使得各方法之间难以比较，也难以理解它们在何种情况下有效。本综述通过学生生成状态、教师信息与策略更新三者之间的相互作用，构建了对在线策略蒸馏的统一视角。该框架将蒸馏的基础方法与自蒸馏、多教师集成、强化学习以及交互式智能体联系起来。我们考察了这些设置如何改变学生模型可获得的监督信号，并将由此产生的设计选择与能力迁移、训练效率及部署应用相关联。我们的综述聚焦于教师指导在何种条件下能够持续（改进学生模型）……

    arXiv:2604.00626v5 Announce Type: replace-cross  Abstract: On-Policy Distillation (OPD) trains large language models with teacher guidance on student-generated trajectories, addressing the mismatch between fixed demonstrations and the histories encountered during deployment. As its use expands across language model training, different formulations of supervision and optimization make it difficult to compare methods and understand when they are effective. This survey develops a unified view of OPD through the interplay between student-generated states, teacher information, and policy updates. This framework connects the foundations of distillation with self-distillation, multi-teacher integration, reinforcement learning, and interactive agents. We examine how these settings change the supervision available to the student and relate the resulting design choices to capability transfer, training efficiency, and deployment. Our synthesis centers on the conditions under which guidance remain
    
[^196]: MICA：面向长程情感支持对话的多粒度跨时信用分配

    MICA: Multi-granularity Intertemporal Credit Assignment for Long-Horizon Emotional Support Dialogue

    [https://arxiv.org/abs/2603.06194](https://arxiv.org/abs/2603.06194)

    提出无评论家强化学习框架MICA，通过共享势函数同时生成即时与延迟两类信用信号，实现情感支持对话中无需状态匹配的稳定逐轮优化。

    

    面向大语言模型（LLM）的强化学习（RL）在单轮任务中表现出色，但由于奖励稀疏且逐轮信用分配能力不足，将其扩展到多轮交互仍然充满挑战。在情感支持对话中，回复会塑造未来的用户状态，因此无法进行匹配状态下的逐步对比，而轨迹层面的监督又不够充分。我们提出了MICA（多粒度跨时信用分配），这是一个面向多轮情感支持任务的无评论家（critic-free）强化学习框架。MICA基于用户结构化支持状态上的共享势函数，同时推导即时信用和延迟信用：增量距离奖励衡量每轮到目标状态剩余距离的减少量，而其蒙特卡洛回报则捕捉延迟效应。经过特定范围的归一化后，这两个信号形成混合优势，从而在无需匹配状态的情况下实现稳定的逐轮优化。

    arXiv:2603.06194v4 Announce Type: replace  Abstract: Reinforcement learning (RL) for large language models (LLMs) has shown strong performance in single-turn tasks, but extending it to multi-turn interaction remains challenging due to sparse rewards and poor per-turn credit assignment. In emotional support dialogues, responses shape future user states, so matched-state step-wise comparison is unavailable, while trajectory-level supervision is insufficient. We propose MICA (Multi-granularity Intertemporal Credit Assignment), a critic-free RL framework for multi-turn emotional support tasks. MICA derives both immediate and delayed credit from a shared potential function over the user's structured support state. Incremental Distance Reward measures the per-turn decrease in residual distance to the target state, while its Monte Carlo return captures delayed effects. After scope-specific normalization, the two signals form a mixed advantage for stable per-turn optimization without matched-s
    
[^197]: StructLens：基于最大生成树的语言模型结构透镜

    StructLens: A Structural Lens for Language Models via Maximum Spanning Trees

    [https://arxiv.org/abs/2603.03328](https://arxiv.org/abs/2603.03328)

    提出StructLens框架，受依存句法树启发，通过在残差流语义表示上构建最大生成树来揭示语言模型表示的整体内部结构，并发现中间层具有最强的局部跨度组织特性。

    

    语言具有内在的结构，这一特性既解释了语言的习得，也解释了语言的演变。基于这一特性，我们期望语言模型同样能展现出其自身的内部结构。尽管可解释性研究已经通过注意力模式和稀疏自编码器探究了模型如何以机制化的方式计算表示，但由此产生的表示的组织方式却一直被忽视。为填补这一空白，我们提出了StructLens，一个通过整体结构性视角来分析表示的框架。StructLens受依存句法分析中树形表示的启发，基于残差流中的语义表示构建最大生成树，从而在表示空间中提供词元间关系的摘要。我们分析了连续的词元在表示空间中是否也彼此邻近，发现中间层表现出最强的局部跨度组织特性。

    arXiv:2603.03328v3 Announce Type: replace  Abstract: Language exhibits inherent structures, a property that explains both language acquisition and language change. Given this characteristic, we expect language models to manifest their own internal structures as well. While interpretability research has investigated how models compute representations mechanistically through attention patterns and Sparse AutoEncoders, the organization of the resulting representations is overlooked. To address this gap, we introduce StructLens, a framework to analyze representations through a holistic structural view. StructLens constructs maximum spanning trees based on the semantic representations in residual streams, inspired by tree representation in dependency parsing, and provides summaries of token relationships in representation space. We analyze how contiguous tokens are also nearby in representation space and find that middle layers show the strongest local-span organization. Moreover, analysis 
    
[^198]: GLUScope：一个用于分析Transformer语言模型中GLU神经元的工具

    GLUScope: A Tool for Analyzing GLU Neurons in Transformer Language Models

    [https://arxiv.org/abs/2602.23826](https://arxiv.org/abs/2602.23826)

    GLUScope是一个开源工具，通过分析基于GLU激活函数的语言模型中神经元门控与输入激活的四种正负符号组合（激活象限），帮助可解释性研究者揭示神经元促进或抑制特定token等截然不同的功能行为。

    

    我们提出了GLUScope，这是一个面向可解释性研究人员的开源资源，用于分析基于Transformer的语言模型（LM）中的神经元。我们聚焦于现已标准化、基于GLU的激活函数（如SwiGLU），这类激活函数几乎被所有近期开源权重的语言模型所采用。这带来了一个新的挑战：仅理解正激活是不够的。相反，神经元的门控激活和输入激活都可能为正或为负，从而产生四种不同的符号组合（激活象限），这些组合在某些情况下具有截然不同的功能（例如，促进还是抑制特定的一组token）。因此，对于任意神经元，我们的工具会展示四种符号组合各自的文本示例，并指出每种组合出现的频率。我们的资源能够带来新颖的洞见：特别是，得益于对所有四个激活象限的分析，它能够揭示原本可能被忽视的可解释行为。

    arXiv:2602.23826v2 Announce Type: replace  Abstract: We present **GLUScope**, an open-source resource for analyzing neurons in Transformer-based language models (LMs), intended for interpretability researchers. We focus on the now-standard *GLU-based activation functions* such as SwiGLU, used in virtually all recent open-weights LMs. This introduces a new challenge: understanding positive activations is not enough. Instead, both the *gate* and the *in* activation of a neuron can be positive or negative, leading to four different possible sign combinations (*activation quadrants*) that in some cases have quite different functionalities (e.g. promoting vs. suppressing a specific set of tokens). Accordingly, for any neuron, our tool shows text examples for each of the four sign combinations, and indicates how often each combination occurs. Our resource can lead to novel insights: in particular, thanks to analyzing all four activation quadrants, it can surface interpretable behavior that o
    
[^199]: 基于信息论的思维链可监控性分析与改进

    Analyzing and Improving Chain-of-Thought Monitorability Through Information Theory

    [https://arxiv.org/abs/2602.18297](https://arxiv.org/abs/2602.18297)

    本文用信息论揭示了思维链可监控性的必要非充分条件，识别出信息鸿沟和引导误差两类失效来源，发现思维链混淆可在无优化压力下自发出现，并证明可通过针对性干预系统性缓解监控的脆弱性。

    

    思维链监控器是一类基于大语言模型的系统，通过分析推理轨迹来检测输出是否可能表现出特定关注属性，例如代码生成过程中“钻测试空子”的行为。本文利用信息论分析表明，思维链与输出之间具有非零互信息是思维链可监控性的必要条件，但并非充分条件。我们识别出两类可能在实际中削弱思维链监控器性能的近似误差来源：信息鸿沟，衡量监控器能够从思维链中提取可用信息的程度；以及引导误差，衡量监控器逼近最优监控函数的程度。实证研究显示，即使不存在任何针对思维链的优化压力，思维链混淆仍可能出现。我们进一步证明，思维链可监控性的脆弱性可以通过针对性干预得到系统性缓解。

    arXiv:2602.18297v2 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) monitors are LLM-based systems that analyze reasoning traces to detect when outputs may exhibit attributes of interest, such as test-hacking behavior during code generation. In this paper, we use information-theoretic analysis to show that non-zero mutual information between CoT and output is a necessary but not sufficient condition for CoT monitorability. We identify two sources of approximation error that may undermine the performance of CoT monitors in practice: information gap, which measures the extent to which the monitor can extract the information available in CoT, and elicitation error, which measures the extent to which the monitor approximates the optimal monitoring function. Empirically, we show that CoT obfuscation can arise even in the absence of any optimization pressure against CoT. We further demonstrate that the fragility of CoT monitorability can be systematically mitigated through targ
    
[^200]: TabAgent：将大语言模型智能体组件蒸馏为廉价分类器

    TabAgent: Distilling LLM Agent Components into Cheap Classifiers

    [https://arxiv.org/abs/2602.16429](https://arxiv.org/abs/2602.16429)

    TabAgent框架通过将智能体系统中反复调用的LLM生成式决策组件替换为基于执行轨迹训练的轻量级文本-表格分类器，在保持任务成功率的同时显著降低部署延迟和成本。

    

    智能体系统是能够自主执行多步骤工作流以实现复杂目标的人工智能架构，通常通过反复调用大语言模型（LLM）来完成路由、候选筛选、门控和验证等封闭集决策任务。这种设计虽然方便，但由于累积的延迟和token消耗，使得部署既缓慢又昂贵。我们提出TabAgent，这是一个用基于执行轨迹训练的紧凑文本-表格分类器来替换封闭集选择任务中生成式决策组件的框架。TabAgent（i）通过TabSchema从轨迹中提取结构化的模式、状态和依赖特征，（ii）通过TabSynth使用与模式对齐的合成监督来扩充数据覆盖范围，并（iii）通过TabHead使用轻量级分类器对候选项进行评分。在长时程AppWorld基准测试中，TabAgent在保持任务级成功率的同时消除了候选筛选阶段的LLM调用，将延迟降低约……（摘要原文在此处截断）

    arXiv:2602.16429v2 Announce Type: replace  Abstract: Agentic systems, AI architectures that autonomously execute multi-step workflows to achieve complex goals, are often built using repeated large language model (LLM) calls for closed-set decision tasks such as routing, shortlisting, gating, and verification. While convenient, this design makes deployments slow and expensive due to cumulative latency and token usage. We propose TabAgent, a framework for replacing generative decision components in closed-set selection tasks with a compact textual-tabular classifier trained on execution traces. TabAgent (i) extracts structured schema, state, and dependency features from trajectories (TabSchema), (ii) augments coverage with schema-aligned synthetic supervision (TabSynth), and (iii) scores candidates with a lightweight classifier (TabHead). On the long-horizon AppWorld benchmark, TabAgent maintains task-level success while eliminating shortlist-time LLM calls, reducing latency by approxima
    
[^201]: 通过生成式引擎优化控制基于大语言模型搜索的输出排名

    Controlling Output Rankings in LLM-Based Search via Generative Engine Optimization

    [https://arxiv.org/abs/2602.03608](https://arxiv.org/abs/2602.03608)

    提出了CORE优化方法，通过向搜索引擎返回的检索内容附加策略性设计的优化内容，来控制基于大语言模型的生成式搜索中的输出排名，从而提升小企业和独立创作者产品的可见性。

    

    随着大语言模型（LLM）的兴起，顾客搜索和选择产品的方式正在发生变化。基于大语言模型的搜索（即生成式引擎）直接向用户提供产品推荐，而不像传统在线搜索那样需要用户自己去探索各种选项。然而，这些推荐深受大语言模型初始检索顺序的影响，这限制了小型企业和独立创作者的可见性，使其处于不利地位。在本工作中，我们提出了CORE，一种用于基于大语言模型搜索的生成式引擎输出排名控制优化方法。由于大语言模型与搜索引擎的交互是黑盒的，CORE以搜索引擎返回的内容作为影响输出排名的主要手段。具体而言，CORE通过在检索到的内容后面附加策略性设计的优化内容，来引导相关内容的排名。

    arXiv:2602.03608v2 Announce Type: replace  Abstract: The way customers search for and choose products is changing with the rise of large language models (LLMs). LLM-based search, or generative engines, provides direct product recommendations to users, rather than traditional online search results that require users to explore options themselves. However, these recommendations are strongly influenced by the initial retrieval order of LLMs, which disadvantages small businesses and independent creators by limiting their visibility.   In this work, we propose CORE, an optimization method that \textbf{C}ontrols \textbf{O}utput \textbf{R}ankings in g\textbf{E}nerative Engines for LLM-based search. Since the LLM's interactions with the search engine are black-box, CORE targets the content returned by search engines as the primary means of influencing output rankings. Specifically, CORE optimizes retrieved content by appending strategically designed optimization content to steer the ranking of
    
[^202]: 掩码扩散模型为何能缓解反转诅咒的理论分析

    A Theoretical Analysis of Why Masked Diffusion Models Mitigate the Reversal Curse

    [https://arxiv.org/abs/2602.02133](https://arxiv.org/abs/2602.02133)

    该论文通过理论分析证明，掩码扩散模型缓解反转诅咒的原因在于正向与反向位置条件分布之间存在参数级耦合——共享的Transformer参数存储词元对证据，而相对位置编码在不改变值的情况下引导注意力，从而使学到的证据能够迁移到反向提示。

    

    自回归语言模型（ARMs）存在“反转诅咒”问题：在学到“A是B”之后，它们在回答反向查询“B是A”时往往会失败。掩码扩散语言模型（MDMs）以弱得多的形式表现出这种失败，但其背后的根本原因一直不清楚。一种常见的解释将这种缓解归因于其任意顺序的掩码训练目标。然而，在训练中观察到“[M]是B”只能教会模型在一种位置配置下从B恢复A，这本身并不能解释为什么学到的证据能够迁移到反向提示“B是[M]”。我们提供了理论分析，表明这种迁移源于正向与反向位置条件分布之间的参数级耦合：共享的Transformer参数存储词元对的证据，而相对位置编码通过查询和键引导注意力，而不改变值本身。

    arXiv:2602.02133v3 Announce Type: replace-cross  Abstract: Autoregressive language models (ARMs) suffer from the reversal curse: after learning ''$A$ is $B$,'' they often fail on the reverse query ''$B$ is $A$.'' Masked diffusion language models (MDMs) exhibit this failure in a much weaker form, but the underlying reason has remained unclear. A common explanation attributes this mitigation to their any-order masked training objective. However, observing ''$[\mathbf{M}]$ is $B$'' during training teaches recovery of $A$ from $B$ in one positional configuration, and does not by itself explain why the learned evidence should transfer to the reverse prompt ''$B$ is $[\mathbf{M}]$.'' We provide a theoretical analysis showing that this transfer arises from a parameter-level coupling between forward and reverse positional conditionals: shared Transformer parameters store token-pair evidence, while relative positional encodings route attention through queries and keys without changing the value
    
[^203]: 利用国际象棋解耦大语言模型中的泛化能力与记忆能力

    Disentangling generalization and memorization in large language models using chess

    [https://arxiv.org/abs/2601.16823](https://arxiv.org/abs/2601.16823)

    该论文提出以国际象棋作为受控测试平台，通过按相关先验密度构建棋局分类体系，在无需了解模型训练数据的情况下区分大语言模型的记忆能力与真正的泛化推理能力，并发现模型性能随相关先验密度降低而持续显著下降。

    

    大语言模型（LLM）展现出卓越的能力，然而这些能力在多大程度上反映了精细的记忆召回还是真正的推理能力，目前仍不清楚。我们引入国际象棋作为一个受控测试平台，旨在解耦这两种能力。利用国际象棋的棋局结构和可扩展的引擎评估，我们构建了一个按相关先验密度变化的棋局分类体系——涵盖从可通过记忆解决的常见局面，到需要泛化能力的完全新颖的局面。关键的是，我们的方法无需对模型训练数据的显式了解即可实现这种区分。应用这一分类体系，我们将对GPT系列模型的纵向分析与对包括Claude Opus和Gemini在内的当代模型的严格评估相结合。我们的分析揭示了一个陡峭的梯度：随着相关先验密度的降低，模型性能持续下降。值得注意的是，对于相关先验较少的任务，基础模型

    arXiv:2601.16823v3 Announce Type: replace  Abstract: Large Language Models (LLMs) exhibit remarkable capabilities, yet it remains unclear to what extent these reflect sophisticated recall or genuine reasoning ability. We introduce chess as a controlled testbed aimed at disentangling these faculties. Leveraging the game's structure and scalable engine evaluations, we construct a taxonomy of positions varying in density of relevant priors -- ranging from common states solvable by memorization to completely novel ones requiring generalization. Crucially, our approach achieves this distinction without requiring explicit knowledge of the models' training data. Applying this taxonomy, we combine a longitudinal analysis of the GPT lineage with a rigorous evaluation of contemporary models, including Claude Opus and Gemini. Our analysis reveals a steep gradient: performance consistently degrades as the density of relevant priors decreases. Notably, for tasks with few relevant priors, base model
    
[^204]: 可重放的金融智能体：面向工具使用型LLM智能体的确定性-忠实性保证框架

    Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents

    [https://arxiv.org/abs/2601.15322](https://arxiv.org/abs/2601.15322)

    该论文提出DFAH框架，首次将工具使用型LLM智能体的决策可重复性、轨迹一致性和证据条件忠实性作为三个独立维度加以区分评估，并澄清了历史研究中r=-0.11相关性统计量的性质与局限。

    

    使用工具的智能体可以在改变其记录的执行过程的同时重复出相同的最终决策。我们提出了确定性-忠实性保证框架，这是一个能够区分决策可重复性、轨迹一致性和证据条件忠实性的评估框架。任务正确性需要单独的合格标签与评估；而证据条件忠实性在历史上的v2智能体实验中并未被评估。原始v2研究在三个合成金融任务中报告了4,705次智能体运行，并在21个模型-基准配置摘要中报告了决策确定性与任务标签匹配之间的相关性r = -0.11。该统计数据可以从历史配置表中复现，但其中包含了一个后来被排除的投资组合测试装置。该统计量被保留作为历史描述，而非统计独立性、预测无用性或架构层面确定性-准确性权衡的证据。

    arXiv:2601.15322v3 Announce Type: replace-cross  Abstract: Tool-using agents can repeat a final decision while changing their recorded execution. We introduce the Determinism-Faithfulness Assurance Harness (DFAH), a framework that distinguishes decision repeatability, trajectory agreement, and evidence-conditioned faithfulness. Task correctness requires separately qualified labels and evaluation; evidence-conditioned faithfulness was not evaluated in the historical v2 agentic experiments.   The original v2 study reported 4,705 agentic runs in three synthetic financial tasks and a decision-determinism/task-label-match correlation of r = -0.11 across 21 model-benchmark configuration summaries. This statistic is reproducible from the historical configuration table, but includes a subsequently excluded portfolio fixture. It is retained as a historical description, not evidence of statistical independence, predictive uselessness, or an architectural determinism-accuracy tradeoff. Recorded d
    
[^205]: RubRIX：基于评分量表的照护者-AI交互风险缓解框架

    RubRIX: Rubric-Driven Risk Mitigation in Caregiver-AI Interactions

    [https://arxiv.org/abs/2601.13235](https://arxiv.org/abs/2601.13235)

    本文提出了RubRIX——一个基于关怀伦理理论、经临床医生验证的评分量表框架，将LLM在照护者交互场景中的回应风险操作化为五个维度，并在超过2万条真实照护者查询上验证，量表引导的改进可将风险成分降低45-98%。

    

    寻求AI介导支持的照护者往往表达出复杂的需求——包括信息获取、情感确认以及痛苦信号——这需要对回应的安全性与恰当性进行细致评估。现有的AI评估框架主要聚焦于一般性风险（如毒性、幻觉、政策违规等），可能无法充分捕捉大语言模型回应在照护情境中的细微风险。我们提出了RubRIX（基于评分量表的风险指数），这是一个理论驱动、经临床医生验证的框架，用于评估LLM照护回应中的风险。RubRIX以“关怀伦理要素”为理论基础，将五个经实证得出的风险维度进行操作化：忽视、偏见与污名化、信息不准确、不加批判的肯定，以及认知傲慢。我们在来自Reddit和ALZConnected的超过20,000条照护者查询上评估了六个最先进的LLM。经过评分量表引导的改进，风险成分持续降低了45-98%……

    arXiv:2601.13235v2 Announce Type: replace-cross  Abstract: Caregivers seeking AI-mediated support express complex needs---information-seeking, emotional validation, and distress cues---that warrant careful evaluation of response safety and appropriateness. Existing AI evaluation frameworks, primarily focused on general risks (toxicity, hallucinations, policy violations, etc), may not adequately capture the nuanced risks of LLM-responses in caregiving-contexts. We introduce RubRIX (Rubric-based Risk Index), a theory-driven, clinician-validated framework for evaluating risks in LLM caregiving responses. Grounded in the Elements of an Ethic of Care, RubRIX operationalizes five empirically-derived risk dimensions: Inattention, Bias & Stigma, Information Inaccuracy, Uncritical Affirmation, and Epistemic Arrogance. We evaluate six state-of-the-art LLMs on over 20,000 caregiver queries from Reddit and ALZConnected. Rubric-guided refinement consistently reduced risk-components by 45-98% after 
    
[^206]: 大语言模型中的算法反身性：心智化语言结构的方法学研究

    Algorithmic Reflectivity in Large Language Models: Methodological Study of Linguistic Structure in Mentalization

    [https://arxiv.org/abs/2512.08945](https://arxiv.org/abs/2512.08945)

    本研究通过让五位MBT受训精神科医生盲评GPT-4.1生成的心智化画像，证明大语言模型无需体验与情感过程，也能生成在语言结构上与心智化治疗维度一致、并被临床专家认可为形式上心智化的内容。

    

    心智化（Mentalization）是人类的一种核心能力，指从潜在心理状态的角度来解释自己和他人的行为。大语言模型（LLMs）能够生成具有语言反思性的话语，这引出了一个问题：心智化的形式语言结构能否在缺乏体验性与情感性过程的情况下被再现。本研究评估了大语言模型能否生成与心智化基础治疗（MBT）各维度在结构上相一致的心智化画像，并被专业临床医生认定为形式上的心智化。研究在标准化条件下使用GPT-4.1生成了50段人机对话；在每段对话结束时，模型生成一份叙述性心智化画像。五位接受过MBT培训的精神科医生独立地对每份画像进行评分，评分涵盖4个经典MBT维度和2个高阶维度（调节功能、综合画像），以衡量其评价一致性与论证一致性。

    arXiv:2512.08945v2 Announce Type: replace  Abstract: Mentalization is a core human capacity involving the interpretation of one's own and others' behavior in terms of underlying mental states. Large language models (LLMs) can generate linguistically reflective discourse, raising the question of whether the formal linguistic structure of mentalization can be reproduced without experiential and affective processes. This study assessed whether an LLM can generate mentalization profiles structurally coherent with the dimensions of mentalization-based treatment (MBT) and recognized as formally mentalizing by expert clinicians. Fifty human-model dialogues were generated with GPT-4.1 under standardized conditions; at the end of each, the model produced a narrative mentalization profile. Five MBT-trained psychiatrists independently rated each profile on 4 canonical MBT axes and 2 higher-order dimensions (regulatory functions, synthetic profile) for evaluative and argumentative coherence. Inter
    
[^207]: 编辑策略对齐的文本简化

    Edit Policy-Aligned Text Simplification

    [https://arxiv.org/abs/2512.06228](https://arxiv.org/abs/2512.06228)

    该论文提出利用大语言模型作为裁判（LLM-Judge）自动构建与简化策略对齐的训练数据，无需人工标注或平行语料库，即可训练出在句子和文档层面适应多种简化策略的模型，且小规模开源模型的性能可媲美甚至超越GPT-5。

    

    文本简化旨在修改文档和句子，使其更易于阅读和理解，同时保留其含义。不同的应用需要不同的简化策略，例如面向词汇层面的简化，以及以牺牲细节为代价换取简洁性的整体重写。然而，实现这种由策略驱动的控制仍然是一个悬而未决的挑战。在这项工作中，我们提出了一种简单而强大的方法，利用大语言模型作为裁判来自动构建与策略对齐的训练数据，从而无需昂贵的人工标注或平行语料库。我们的方法能够构建在句子和文档两个层面都能适应多种简化策略的简化模型。值得注意的是，小规模的开源大语言模型在性能上达到甚至超越了GPT-5，这一点已通过自动指标和编辑分析得到验证。该方法在各个模型家族和模型规模上都带来了一致的改进。

    arXiv:2512.06228v2 Announce Type: replace  Abstract: Text simplification aims to modify documents and sentences to make them easier to read and understand while preserving their meanings. Different applications require distinct simplification policies, e.g., lexical-oriented simplification and overall rewriting while trading off detail for simplicity. However, achieving such policy-driven control remains an open challenge. In this work, we introduce a simple yet powerful approach that leverages Large Language Model-as-a-Judge (LLM-Judge) to automatically construct policy-aligned training data, removing the need for costly human annotation or parallel corpora. Our method enables the construction of simplification models that adapt to diverse simplification policies at both the sentence and document levels. Remarkably, small-scale open-source LLMs match or surpass GPT-5, as verified by both automatic metrics and edit analysis. The consistent improvements across model families and sizes d
    
[^208]: ClusterFusion：融合嵌入引导与大语言模型自适应的混合聚类

    ClusterFusion: Hybrid Clustering with Embedding Guidance and LLM Adaptation

    [https://arxiv.org/abs/2512.04350](https://arxiv.org/abs/2512.04350)

    ClusterFusion提出了一种以大语言模型为聚类核心、由轻量级嵌入方法引导的三阶段混合聚类框架（嵌入引导子集划分、LLM主题总结与LLM主题分配），无需昂贵微调即可融入领域知识，并在多个公开基准和领域数据集上取得最先进的文本聚类性能。

    

    文本聚类是自然语言处理中的一项基础任务，然而使用预训练嵌入的传统聚类算法在特定领域场景中，若不进行代价高昂的微调往往表现不佳。大语言模型（LLM）提供了强大的上下文推理能力，但以往的研究主要将其作为辅助模块来优化嵌入或调整聚类边界。我们提出了ClusterFusion，这是一种混合框架，它转而将大语言模型视为聚类的核心，并由轻量级嵌入方法加以引导。该框架分为三个阶段进行：嵌入引导的子集划分、大语言模型驱动的主题总结，以及基于大语言模型的主题分配。这一设计使得领域能够直接融入领域知识和用户偏好，充分发挥大语言模型的上下文自适应能力。在三个公开基准数据集和两个新的领域特定数据集上的实验表明，ClusterFusion不仅实现了最先进的性能

    arXiv:2512.04350v2 Announce Type: replace  Abstract: Text clustering is a fundamental task in natural language processing, yet traditional clustering algorithms with pre-trained embeddings often struggle in domain-specific contexts without costly fine-tuning. Large language models (LLMs) provide strong contextual reasoning, yet prior work mainly uses them as auxiliary modules to refine embeddings or adjust cluster boundaries. We propose ClusterFusion, a hybrid framework that instead treats the LLM as the clustering core, guided by lightweight embedding methods. The framework proceeds in three stages: embedding-guided subset partition, LLM-driven topic summarization, and LLM-based topic assignment. This design enables direct incorporation of domain knowledge and user preferences, fully leveraging the contextual adaptability of LLMs. Experiments on three public benchmarks and two new domain-specific datasets demonstrate that ClusterFusion not only achieves state-of-the-art performance on
    
[^209]: 认知替代：Grokipedia的AI生成百科全书如何重构权威结构

    Epistemic Substitution: How Grokipedia's AI-Generated Encyclopedia Restructures Authority

    [https://arxiv.org/abs/2512.03337](https://arxiv.org/abs/2512.03337)

    该研究通过对比分析1,000对维基百科与Grokipedia文章的近60万条引用，揭示了AI生成的百科全书显著减少了对学术和新闻来源的依赖，转而更多引用用户生成内容及智库出版物，表明AI正在从基础上重构百科全书知识生产的认知权威结构。

    

    二十多年来，维基百科一直代表着一种去中心化、由人工策展的百科全书式知识生产模式。以Grokipedia为代表的AI生成百科全书的出现，提出了一个根本性问题：人工策展与AI策展的知识体系是否依赖于相同的认知权威基础？我们通过对1,000对匹配的维基百科——Grokipedia文章（包含近60万条引用实例）的比较分析来探讨这一问题。借助一套八类别的分类方案，我们构建了文章的认知特征，以考察其引用来源的机构构成。研究结果表明，两者在引用实践上存在实质性差异：Grokipedia对学术和新闻来源的依赖比例较低，而对用户生成内容以及非政府组织和智库出版物的依赖比例较高。这些差异在广泛的主题领域中长期存在，尽管其程度因具体主题而有所不同。

    arXiv:2512.03337v2 Announce Type: replace-cross  Abstract: For more than two decades, Wikipedia has represented a decentralized, human-curated model of encyclopedic knowledge production. The emergence of AI-generated encyclopedias, such as Grokipedia, raises a fundamental question: do human- and AI-curated knowledge systems rely on the same foundations of epistemic authority? We examine this question through a comparative analysis of 1,000 matched Wikipedia--Grokipedia article pairs, comprising nearly 600,000 citation instances. Using an eight-category classification scheme, we construct the epistemic profiles of articles to examine the institutional composition of their cited sources. Our findings reveal substantial differences in sourcing practices. Grokipedia relies proportionally less on academic and journalistic sources and more on user-generated content and NGO and think-tank publications. These differences persist across broad topic domains, although their magnitude varies by su
    
[^210]: PRISM：面向金融检索的提示精炼式上下文系统建模

    PRISM: Prompt-Refined In-Context System Modelling for Financial Retrieval

    [https://arxiv.org/abs/2511.14130](https://arxiv.org/abs/2511.14130)

    PRISM 提出一种无需训练的金融文档检索框架，通过组合提示工程、上下文学习与候选精炼三个可部署模块，将 FiQA-2018 的 Recall@100 从 0.7783 提升至 0.8326 并降低约 35% 的每查询成本，同时系统性地揭示了 k-倒数编码与多智能体协作等额外复杂性何时不再带来收益的边界条件。

    

    将大语言模型（LLM）部署于金融信息检索需要准确性、成本效率和可复现性，然而大多数方法都需要代价高昂的微调。PRISM 是一个无需训练的框架，它由三个可部署的模块（提示工程、上下文学习和候选精炼模块）组合而成，并针对金融文档检索中的多智能体协作进行了边界条件研究。在 FiQA-2018 数据集上，堆叠式重排序与 k-倒数编码将 Recall@100 从 0.7783 提升至 0.8326。在 FinAgentBench 上，经 Cohere 精炼的 top-50 候选池带来了更高的观测端到端 NDCG@5（低于该评估设置下的最小可检测效应），同时将每条查询的成本削减约 35%。额外增加的复杂性并不总是划算的：当上游阶段已达到高召回率时，k-倒数编码便不再带来增益；而对于细粒度的文本块排序，多智能体协作也并非默认的最优选择。在保持提示、模型和示例一致的配对对比实验中……（原文摘要在此处截断）

    arXiv:2511.14130v3 Announce Type: replace-cross  Abstract: Deploying LLMs for financial information retrieval demands accuracy, cost efficiency, and reproducibility, yet most approaches require expensive fine-tuning. PRISM is a training-free framework that composes three deployable modules (prompt engineering, in-context learning, and a candidate refinement module) with a boundary-condition study of multi-agent coordination for financial document retrieval. On FiQA-2018, stacked reranking and $k$-reciprocal encoding raise Recall@100 from 0.7783 to 0.8326. On FinAgentBench, a Cohere-refined top-50 pool yields a higher observed end-to-end NDCG@5, below this evaluation's minimum detectable effect, while cutting per-query cost by about 35%. Added complexity does not always pay off: $k$-reciprocal encoding stops helping once the upstream stage is already high-recall, and multi-agent coordination is not a default for fine-grained chunk ranking. Matched pairs that hold prompt, model, and exem
    
[^211]: 保护去标识化文档免受基于搜索的关联攻击

    Protecting De-identified Documents from Search-based Linkage Attacks

    [https://arxiv.org/abs/2510.06383](https://arxiv.org/abs/2510.06383)

    本文提出一种两步防御方法：先构建N-gram倒排索引以高效识别文档中易被搜索关联的罕见片段，再利用大语言模型迭代重写这些片段，在保持文本语义完整性的同时有效防止去标识化文档被溯源关联。

    

    尽管去标识化模型可以帮助隐藏文档中提及的个体身份，但它们未能应对关联风险，即可能将去标识化后的文本映射回其来源的风险。执行此类关联的一种直接方式是从去标识化文档中提取短语，并检查这些短语是否存在于原始数据集中。本文提出了一种在保持文本语义完整性的同时对抗基于搜索的关联攻击的方法。该方法分两步进行：首先，我们构建文本集合中出现的N-gram的倒排索引，从而能够高效地确定哪些N-gram单独或与其他N-gram组合出现在少于k个文档中；随后，迭代地查询基于大语言模型的重写器来重新表述这些片段，直到无法再进行关联为止。我们在两个数据集（法院判例和维基百科传记）上进行了实验。

    arXiv:2510.06383v3 Announce Type: replace  Abstract: While de-identification models can help conceal the identity of the individuals mentioned in a document, they fail to address linkage risks, defined as the potential to map the de-identified text back to its source. One straightforward way to perform such linkages is to extract phrases from the de-identified document and check their presence in the original dataset. This paper presents a method to counter search-based linkage attacks while preserving the semantic integrity of the text. The method proceeds in two steps. We first construct an inverted index of the N-grams occurring in the text collection, making it possible to efficiently determine which N-grams appear in fewer than $k$ documents, either alone or in combination with other N-grams. An LLM-based rewriter is then iteratively queried to reformulate those spans until linkage is no longer possible. Experimental results on two datasets (court cases and Wikipedia biographies) 
    
[^212]: 答案蕴于其中：自我衍生奖励实现可解释关系抽取

    The Answer Lies Within: Self-Derived Rewards Enable Explainable Relation Extraction

    [https://arxiv.org/abs/2510.06198](https://arxiv.org/abs/2510.06198)

    该论文提出COGRE认知启发推理框架与HIT@DICT强化学习中间奖励策略，利用从正确预测中自动提取的信用词典派生奖励，引导大模型推理聚焦关系相关语义并与人类标注的抽象层次对齐，从而同时提升单样本关系抽取的准确率和解释质量。

    

    arXiv:2510.06198v4 公告类型：替换 摘要：尽管大型语言模型具有卓越的推理能力，但在没有预定义关系标签的情况下进行单样本（one-shot）关系抽取仍然存在困难。我们识别出两个问题：模型常常被无关的词元（token）误导，而非关注传达关系的语义；并且模型往往无法与人类标注者所期望的抽象层次保持一致。我们提出了一个新颖的框架，通过两个组件来弥合这一差距：(1) COGRE，一个受认知启发的推理框架，将关系抽取结构化为一系列模仿人类文本处理的过程；(2) HIT@DICT，一种强化学习中间奖励策略，通过对推理中与关系相关的短语给予奖励，促使推理与关系标签对齐。该奖励源自从正确预测中自动提取的信用词典（credit dictionary）。我们的实验表明，该框架通过解决上述问题，同时提升了准确率和解释质量。

    arXiv:2510.06198v4 Announce Type: replace  Abstract: Despite the remarkable reasoning capabilities of large language models, they still struggle with one-shot relation extraction without predefined relation labels. We identify two pitfalls: models are often misled by irrelevant tokens instead of relation-conveying semantics, and they often fail to align with the abstraction level human annotators expect. We introduce a novel framework that closes this gap with two components: (1) COGRE, a cognitively-inspired reasoning framework that structures RE into a series of processes mimicking human text-processing; and (2) HIT@DICT, a reinforcement learning intermediate reward strategy that encourages reasoning to align with relational labels by rewarding relation-relevant phrases in reasoning. The reward is derived on a credit dictionary automatically extracted from correct predictions. Our experiments show that our framework improves both accuracy and explanation quality by addressing these t
    
[^213]: 超维探针：基于向量符号架构解码大语言模型表示

    Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures

    [https://arxiv.org/abs/2509.25045](https://arxiv.org/abs/2509.25045)

    提出超维探针——一种结合向量符号架构与神经探测的混合式有监督探针，统一了有监督探针、稀疏自编码器和logit分析等可解释性方法，从而更全面地解码大语言模型的内部表示。

    

    尽管大型语言模型（LLM）具有强大的能力，但其内部仍然是不透明的，人们对其内部表示的理解十分有限。目前的可解释性方法要么专注于面向输入的特征提取，例如有监督探针和稀疏自编码器（SAE），要么专注于输出分布检查，例如面向logit的方法。然而，要全面理解LLM的向量空间，需要将这两种视角整合起来，而现有方法由于潜在特征定义上的限制，难以做到这一点。我们提出了超维探针，这是一种将符号表示与神经探测相结合的混合式有监督探针。通过利用向量符号架构（VSA）和超向量代数，它统一了先前的方法：有监督探针的自顶向下可解释性、SAE由稀疏性驱动的代理空间，以及面向输出的logit研究。通过结合传统的有监督学习范式……

    arXiv:2509.25045v4 Announce Type: replace  Abstract: Despite their capabilities, Large Language Models (LLMs) remain opaque with limited understanding of their internal representations. Current interpretability methods either focus on input-oriented feature extraction, such as supervised probes and Sparse Autoencoders (SAEs), or on output distribution inspection, such as logit-oriented approaches. A full understanding of LLM vector spaces, however, requires integrating both perspectives, something existing approaches struggle with due to constraints on latent feature definitions. We introduce the Hyperdimensional Probe, a hybrid supervised probe that combines symbolic representations with neural probing. Leveraging Vector Symbolic Architectures (VSAs) and hypervector algebra, it unifies prior methods: the top-down interpretability of supervised probes, SAE's sparsity-driven proxy space, and output-oriented logit investigation. By combining the supervised learning paradigm of traditiona
    
[^214]: 从价值条件化到行为转变：大语言模型的轻量级价值对齐

    From Value Conditioning to Behavioral Shift: Lightweight Value Alignment of LLMs

    [https://arxiv.org/abs/2508.11414](https://arxiv.org/abs/2508.11414)

    本文提出一种轻量级的大语言模型价值对齐方法，仅通过对价值调查回答进行微调，即可选择性地改变模型的目标价值倾向，并将这种价值转变迁移到域外的下游行为中。

    

    大型语言模型在输出中表现出与价值相关的偏好，然而引导这种行为通常需要大规模训练数据或成对的偏好标注。在这项工作中，我们研究了一种轻量级的替代方案：基于价值调查回答进行微调，能否在价值条件化的模型输出及下游行为中引发可测量的转变？我们首先让多个开源大语言模型对围绕施瓦茨十种基本价值组织的价值相关描述进行评分，从而估计它们的基线价值画像。随后，我们在调查式的监督数据上微调模型，该监督选择性地改变目标价值的评分，同时使非目标价值保持接近其基线回答。我们在两种设置下评估由此产生的转变：第一，测量该干预是否会改变模型在留出调查问题上的回答；第二，测试这些由调查引发的转变能否迁移到域外行为代理指标上，包括一种情境……（原文此处截断）

    arXiv:2508.11414v2 Announce Type: replace  Abstract: Large language models exhibit value-related preferences in their outputs, yet steering such behavior often requires large training datasets or paired preference annotations. In this work, we investigate a lightweight alternative: can fine-tuning on value survey responses induce measurable shifts in value-conditioned model outputs and downstream behavior? We first estimate baseline value profiles for several open-source LLMs by asking them to rate value-related descriptions organized around the ten Schwartz values. We then fine-tune models on survey-style supervision that selectively changes the target value ratings while keeping non-target values close to their baseline responses. We evaluate the resulting shifts in two settings. First, we measure whether the intervention changes responses on held-out survey questions. Second, we test whether these survey-induced shifts transfer to out-of-domain behavioral proxies, including a contex
    
[^215]: PRiME：心理健康领域的语用推理

    PRiME: Pragmatic Reasoning In MEntal health

    [https://arxiv.org/abs/2507.23247](https://arxiv.org/abs/2507.23247)

    该论文提出了PRiME任务与数据集，系统评估了12个大语言模型在高风险心理健康领域的语用推理能力，发现模型表现可与人类评估相当，其中Mistral-7B最佳，并通过错误不对称比率和注意力回溯实验深入分析了模型的架构行为。

    

    在这项工作中，我们研究了推理与心理健康领域之间的任务与领域交互问题。鉴于大语言模型（LLMs）展现出卓越的文本理解和流畅的文本生成能力，我们分析了它们能否在心理健康这一高风险领域中进行推理。我们首先基于现有的语用推理研究，引入了三个数据集和三个任务。该分析对12个大语言模型进行了评估，其中包括MentaLLaMA——一个心理健康推理与解释生成模型。这些模型的性能与人类评估水平相当，其中Mistral-7B模型表现最佳。我们进一步使用错误不对称比率分析了模型的性能，并对MentaLLaMA-7B进行了首阶注意力回溯（first-order attention rollout）实验。我们向社区展示了模型针对输入文本的架构行为结果，并建议在相关应用中进一步保持谨慎。

    arXiv:2507.23247v3 Announce Type: replace  Abstract: In this work, the task and domain interaction of reasoning and mental health are investigated. As large language models (LLMs) present exceptional text understanding and fluent text generation capability, we analyze whether they can reason in the high-stakes domain of mental health. We begin by introducing three datasets and three tasks leveraging the existing work in pragmatic reasoning. The analysis constitutes an evaluation of $12$ LLMs, including MentaLLaMA, which is a mental health reasoning and explanation generation model. The performance of the models remains on par with human evaluation, with the Mistral-7B model giving the best performance. We further analyze the performance of the models with error asymmetry ratio and conduct first-order attention rollout experiment with MentaLLaMA-7B. We present the results for the community reporting the architectural behaviour of the models for the input text and recommend further caref
    
[^216]: 心智理论与亲社会信念对大语言模型在最后通牒博弈中引导类人行为的影响

    Effects of Theory of Mind and Prosocial Beliefs on Steering Human-Aligned Behaviors of LLMs in Ultimatum Games

    [https://arxiv.org/abs/2505.24255](https://arxiv.org/abs/2505.24255)

    本论文通过2700次模拟实验证明，在最后通牒博弈中，心智理论推理能显著增强大语言模型行为与人类规范的对齐、决策一致性和谈判结果，且优于单纯推理模型的表现。

    

    大语言模型（LLMs）在模拟人类行为和进行心智理论（ToM）推理方面展现出潜力，这对于复杂的社会互动至关重要。我们以最后通牒博弈为参考任务，研究了ToM推理在将代理行为与人类规范对齐中的作用。我们为LLM代理初始化了不同的亲社会信念（贪婪、公平、无私）和推理方法（思维链及不同层次的ToM推理），并考察了它们在多种LLM（包括推理模型如o3-mini和DeepSeek-R1 Distilled Qwen 32B）中的决策过程和结果。我们进行了2700次模拟，结果表明ToM推理增强了行为与人类规范的对齐、决策一致性以及谈判结果。与先前发现一致，推理型LLM相比增强ToM的LLM表现出有限的能力，不同的博弈角色受益于不同的方法。

    arXiv:2505.24255v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have shown potential in simulating human behaviors and performing theory-of-mind (ToM) reasoning, crucial for complex social interactions. We investigate ToM reasoning's role in aligning agentic behaviors with human norms in negotiation tasks, using the ultimatum game as our referenced task. We initialized LLM agents with different prosocial beliefs (Greedy, Fair, Selfless) and reasoning methods (chain of thought and ToM reasoning of varying levels), examining their decision-making process and outcome across multiple LLMs, including reasoning models like o3-mini and DeepSeek-R1 Distilled Qwen 32B. We perform 2,700 simulations to show that ToM reasoning enhances behavioral alignment with human, decision-making consistency, and negotiation outcomes. Consistent with prior findings, reasoning LLMs exhibit limited capability compared to ToM-enhanced LLMs, with different game roles benefiting from differe
    
[^217]: 既非随机鹦鹉学舌，也非通用人工智能：大语言模型通过基于训练数据先验的上下文导向外推来完成任务

    Neither Stochastic Parroting nor AGI: LLMs Solve Tasks through Context-Directed Extrapolation from Training Data Priors

    [https://arxiv.org/abs/2505.23323](https://arxiv.org/abs/2505.23323)

    该论文提出大语言模型的行为既非“随机鹦鹉学舌”也非“高级推理”，而是“上下文导向外推”——即模型从训练数据先验外推并借助上下文选择合适的先验，同时提供了一个可操作的评估清单，用于区分记忆、基本外推与真正的应用层面泛化。

    

    这篇立场论文认为，许多被归类为“随机鹦鹉学舌”或“高级推理”的行为，更好的解释是从训练数据先验出发的上下文导向外推，机器学习和自然语言处理研究界将受益于那些能够区分记忆与基本外推同真正的应用层面泛化的评估方法。我们的核心贡献是一个可操作的评估清单和设计模式，使这些区别在实践中可被检验。我们的框架认为，大语言模型从训练数据先验中进行外推，同时利用上下文来引导模型找到合适的先验；我们将其称为“上下文导向外推”。具体而言，在基础模型中，这种上下文引导通过示例实现，从而产生上下文学习能力；而指令微调则允许通过提示而非显式示例来实现类似的任务指定。我们主张采用这种更符合现实的框架……

    arXiv:2505.23323v3 Announce Type: replace  Abstract: This position paper argues that many behaviours framed as either "stochastic parroting" or "advanced reasoning" are better explained as context-directed extrapolation from training-data priors, and that ML and NLP research communities would benefit from evaluations that distinguish remembering and basic extrapolation from genuine application-level generalisation. Our central contribution is an actionable evaluation checklist and design patterns that make these distinctions testable in practice. Our framework posits that LLMs extrapolate from training data priors while using context to guide the model to the appropriate priors; we call this "context-directed extrapolation." Specifically, this context direction is achieved through examples in base models, leading to in-context learning, while instruction tuning allows analogous task specification via prompts rather than explicit examples. We argue that adopting this more realistic fram
    
[^218]: 视觉语言模型中数据可视化理解的以人为中心的评估

    Human-centered evaluation of data visualization understanding in vision-language models

    [https://arxiv.org/abs/2505.17202](https://arxiv.org/abs/2505.17202)

    该论文提出了一种以人为中心的评估协议，通过在相同的数据可视化理解任务上直接比较视觉语言模型与人类的反应模式，弥补了以往仅依赖准确率评估的不足。

    

    数据可视化理解需要整合视觉、数值和语言输入，对多模态推理提出了根本性的挑战。最近开发的视觉语言模型（VLMs）是建模人类数据可视化理解背后计算过程的有潜力的候选者。然而，它们作为科学模型是否有用，不仅取决于能否给出正确答案，还取决于它们能否近似人类在相同任务上产生的反应模式。这一区别在近期的VLM评估中被忽视了，这些评估通常只报告模型的整体表现（准确率），而没有与人类行为模式进行直接比较。本文引入了一种以人为中心的评估协议，在相同的一组可视化理解题目上比较VLM与人类的反应。我们将该协议应用于八个VLM在六项评估中的表现：五项已建立的测试……（摘要在此处被截断）

    arXiv:2505.17202v2 Announce Type: replace-cross  Abstract: Data visualization understanding, which requires integrating visual, numerical, and linguistic inputs, poses fundamental challenges to multimodal reasoning. Recently developed vision-language models (VLMs) are promising candidates for modeling the computations that underlie human data visualization understanding. However, whether they are useful as scientific models depends not only on whether they produce correct answers, but also whether they approximate the patterns of responses that people produce on the same tasks. This distinction has been ignored in recent evaluations of VLMs, which typically only report overall model performance (accuracy) without direct comparison to human behavioral patterns. Here we introduce a human-centered evaluation protocol that compares VLM and human responses on the same set of visualization understanding items. We apply this protocol to eight VLMs across six assessments: five established test
    
[^219]: AnnoCaseLaw：民事过失法律标注基准

    AnnoCaseLaw: Legal Annotation Benchmark of Civil Negligence

    [https://arxiv.org/abs/2503.00128](https://arxiv.org/abs/2503.00128)

    提出了AnnoCaseLaw基准——一个包含471份美国民事过失法院意见书和超过12,000个专家标注的密集标注数据集，用于评估语言模型复现法律专家标注的能力，发现模型在“事实”和“结果”类别上表现较好，而在“法律对事实的适用”等需要复杂法律推理的类别上仍存在挑战。

    

    语言模型为扩展昂贵且劳动密集的专家标注提供了一种途径，但它们复现专家判断的能力仍不确定。这在法律领域尤其具有挑战性，因为标注者必须将支持司法判决的推理与其他法律相关讨论区分开来。我们提出了AnnoCaseLaw，这是一个密集标注的基准，包含471份美国民事过失法院意见书，涵盖五个标注类别的超过12,000个专家高亮标注。我们评估了专有前沿模型、开源权重模型、法律专业化模型和决策模型复现这些标注的能力。模型在“事实”和“结果”类别上的表现优于“法律对事实的适用”、“程序历史”和“相关判例”类别。在“法律对事实的适用”上，专有前沿模型实现了推理覆盖率和精确度的最佳平衡，而开源权重模型Gemma 4则达到了相当的Macro-F1分数。

    arXiv:2503.00128v2 Announce Type: replace  Abstract: Language models offer a way to scale costly, labour-intensive expert annotation, but their ability to reproduce expert judgements remains uncertain. This is particularly challenging in law, where annotators must distinguish the reasoning supporting judicial decisions from other legally relevant discussion. We present AnnoCaseLaw as a densely annotated benchmark comprising $471$ U.S. civil-negligence court opinions with more than $12,000$ expert highlights across five annotation categories. We evaluate proprietary frontier, open-weight, legal-specialised, and decision models on their ability to reproduce these annotations. Models perform better on Facts and Outcome than on Application of Law to Facts, Procedural History, and Relevant Precedents. On Application of Law to Facts, proprietary frontier models achieve the strongest balance of reasoning coverage and precision, while the open-weight model Gemma 4 achieves comparable Macro-F1 
    
[^220]: 人类认知基准揭示多模态大语言模型的基础视觉能力差距

    Human Cognitive Benchmarks Reveal Foundational Visual Gaps in MLLMs

    [https://arxiv.org/abs/2502.16435](https://arxiv.org/abs/2502.16435)

    该论文提出VisFacor基准，将认知心理学评估FRCT中的20个视觉子测试数字化，以评估多模态大语言模型的基础视觉认知能力，发现即使是最好的模型也仅得55.9%，揭示了当前模型在基础视觉能力上存在显著差距。

    

    人类通过自底向上的层级结构发展感知能力：从基本图元和格式塔原则到高级语义。当前的多模态大语言模型（MLLMs）直接在复杂的下游任务上训练，这能否保证它们具备这些基础视觉能力？为了系统地研究这一差距，我们提出了VisFacor基准，该基准将FRCT（一项成熟的认知心理学评估，涵盖人类视觉认知的四个领域）中的20个以视觉为中心的子测试数字化。此外，我们设计了算法来自动构建和验证具有可控难度的无限测试用例。使用VisFacor，我们评估了39个前沿的多模态大语言模型，包括专有模型（如GPT、Gemini）和开源模型（如LLaMA、Qwen）。表现最好的模型仅获得55.9%的分数。分析显示该基准具有良好的内部一致性（Cronbach's alpha = 0.94）和结构效度。

    arXiv:2502.16435v5 Announce Type: replace-cross  Abstract: Humans develop perception through a bottom-up hierarchy: from basic primitives and Gestalt principles to high-level semantics. Current Multimodal Large Language Models (MLLMs) are trained directly on complex downstream tasks. Does it guarantee their performance on these foundational visual capabilities? To systematically investigate this gap, we introduce VisFacor, a benchmark that digitizes 20 vision-centric subtests from FRCT, a well-established cognitive psychology assessment spanning four domains of human visual cognition. Furthermore, we design algorithms to automatically construct and validate unlimited test cases with controllable difficulty. Using VisFacor, we evaluate 39 frontier MLLMs, including both proprietary (e.g., GPT, Gemini) and open-source (e.g., LLaMA, Qwen) models. The best model achieves a score of only 55.9%. Analysis reveals good internal consistency (Cronbach's alpha = 0.94) and construct validity (compa
    
[^221]: 基于动态加权的100种语言多标签情感分析（针对标签不平衡问题）

    Multi-Label Sentiment Analysis on 100 Languages with Dynamic Weighting for Label Imbalance

    [https://arxiv.org/abs/2008.11573](https://arxiv.org/abs/2008.11573)

    该论文提出了一种动态加权方法来应对多标签情感分析中的标签不平衡问题，并结合适配的多标签焦点损失与线性时间复杂度的最优阈值选择方法，在100种语言的情感分析任务中于9个指标中的7个上达到了最先进水平。

    

    我们研究跨语言情感分析，该领域因其在市场研究、政治和社会科学等多个领域的应用而受到广泛关注。特别是，由于情感遵循普拉奇克情感轮模型，我们引入了一个多标签设置下的情感分析框架。我们提出了一种新颖的动态加权方法，在训练过程中平衡每个类别的贡献，这不同于以往根据类别频率分配固定权重的静态加权方法。此外，我们将单标签目标识别文献中偏向较难样本的焦点损失（focal loss）适配到我们的多标签设置中。进一步地，我们推导出一种在线性时间复杂度内选择最优类别特定阈值以最大化宏平均F1分数的方法。通过大量实验，我们证明我们的方法在3个不同（数据集）的9个指标中有7个取得了最先进的性能。

    arXiv:2008.11573v2 Announce Type: replace-cross  Abstract: We investigate cross-lingual sentiment analysis, which has attracted significant attention due to its applications in various areas including market research, politics and social sciences. In particular, we introduce a sentiment analysis framework in multi-label setting as it obeys Plutchik's wheel of emotions. We introduce a novel dynamic weighting method that balances the contribution from each class during training, unlike previous static weighting methods that assign non-changing weights based on their class frequency. Moreover, we adapt the focal loss that favors harder instances from single-label object recognition literature to our multi-label setting. Furthermore, we derive a method to choose optimal class-specific thresholds that maximize the macro-f1 score in linear time complexity. Through an extensive set of experiments, we show that our method obtains the state-of-the-art performance in 7 of 9 metrics in 3 differen
    

