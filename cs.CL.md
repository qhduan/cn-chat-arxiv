# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Coding Agents with an Obstacle-Aware Harness for Safe Robot Manipulation](https://arxiv.org/abs/2609.20822) | 本文发现编码智能体在机器人操作中因规划环节未能将安全约束设为优先事项而系统性碰撞障碍物（而非感知或指令问题），并通过将操作分解为路径阶段和接触时刻来定位失败根源，提出了障碍物感知框架以实现安全操作。 |
| [^2] | [Embedding Models Measure in Peculiar Ways](https://arxiv.org/abs/2609.20821) | 该研究发现嵌入模型对质量、距离、时间和体积等物理测量的表示十分微弱且奇特，主要受表面字符串相似性的强烈影响，而重新校准相似度也无法显著改善其与真实物理测量的对齐。 |
| [^3] | [Unifying Models of Intergroup Hostility in Online Discourse](https://arxiv.org/abs/2609.20808) | 本研究利用2024年美国大选期间TikTok、Truth Social和Twitter/X平台的286万条帖子，首次对六种群体间敌意基础理论进行统一建模，揭示了敌意修辞机制在真实在线话语中的表现形式及相互关系。 |
| [^4] | [An Empirical Study of Harness Design for Coding Agents](https://arxiv.org/abs/2609.20804) | 该论文通过固定执行循环并系统变化规划、动作空间和上下文管理三个组件的实证研究，发现上下文管理在上下文窗口预算紧张时价值显著提升，且其主要收益来自防止上下文溢出故障。 |
| [^5] | [JEPA-Anything: Learning Predictive Models across Different Worlds](https://arxiv.org/abs/2609.20800) | 提出基于正交预测因子分解（OPF）的领域无关框架JEPA-Anything，以统一的学习原则在视觉、生物学、临床轨迹、控制、分子动力学、物理场和天气七个迥异领域中实现世界建模。 |
| [^6] | [RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning](https://arxiv.org/abs/2609.20784) | RetireOPD提出了一种自我退休的在线策略蒸馏方法，通过自适应退休机制让强化学习智能体在教师监督收益不再增长时自动辞退教师，从而更高效地内化特权任务技能。 |
| [^7] | [Harm Laundering in GPT Models: Evidence That Gender Discrimination Is Transformed Rather Than Reduced Across Safety-Trained Generations](https://arxiv.org/abs/2609.20779) | 该论文提出“危害洗白”这一新概念，通过对GPT-2至GPT-5共15个模型的45万条性别导向文本分析，证明安全训练并未真正消除性别歧视，而是将其从露骨的性暴力内容转化为更隐蔽的形式（如将乳腺癌话题建构为男性权利辩论），从而揭示现有基于表层分类器的安全评估方法的系统性缺陷。 |
| [^8] | [dQwen3.5: Hybrid-Attention Diffusion Language Models](https://arxiv.org/abs/2609.20751) | 本论文将Qwen3.5的注意力-RNN混合架构（0.8B至9B规模）适配为dQwen3.5扩散语言模型系列，证明混合架构骨干仅需全注意力模型约一半的训练词元即可达到相同损失，同时在任意顺序解码上表现相当，并在并行解码下表现出色。 |
| [^9] | [On-Demand Attention: Language Models Know When to Recall](https://arxiv.org/abs/2609.20734) | 提出按需注意力（ODA）解码方法，利用轻量级回忆头根据预测收益动态选择性地调用全局注意力，在不改变预训练权重的前提下实现长上下文推理的实际解码加速。 |
| [^10] | [Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL](https://arxiv.org/abs/2609.20715) | 该论文提出ActObs方法，在监督微调中同时对轨迹中已有的环境观测标记进行监督，使策略学会建模动作后果，从而在不增加任何数据、参数或计算成本的情况下，显著提升后续GRPO强化学习中智能体的探索能力和pass@k性能。 |
| [^11] | [Summarization Bias: The Directional Collapse of Objective Projection into Told-Mode Labels in Large Language Models --- A Conceptual Framework and Registered Test Protocol](https://arxiv.org/abs/2609.20712) | 本文提出“摘要化偏差”这一新概念并设计预注册测试协议，用以验证大型语言模型在叙事生成中会系统性地将本应以“展示”方式呈现的情感内容坍缩为直接“陈述”式的摘要标签。 |
| [^12] | [HerHealthEval: Evaluating Multilingual and Register-Sensitive Understanding of Women's Health Communication](https://arxiv.org/abs/2609.20684) | 该论文提出HerHealthEval框架，通过英语、法语、阿拉伯语及六种不同表达形式的女性健康临床病例，系统评估大语言模型是否真正正确理解用户关切，尤其是其识别信息不足并主动请求澄清的能力。 |
| [^13] | [PAA: The Probabilistic Allen Algebra: A Generative and Complete Probabilistic Extension of Allen's Interval Relations](https://arxiv.org/abs/2609.20634) | 本文提出概率Allen代数（PAA），一种生成式且完整的概率扩展，通过从区间边界的概率分布中推导关系概率，解决了经典Allen区间代数无法处理时间信息不确定性及程度化时间表达的问题。 |
| [^14] | [UniPolicy: Unified Objective-Specific Policies for Generative Search Advertising](https://arxiv.org/abs/2609.20630) | UniPolicy提出了一种目标感知的多策略对齐框架，通过目标特定前缀标记、稀疏MoE-LoRA路由和残差FFN在共享骨干网络中分层解耦参数，使生成式搜索广告能够联合优化相关性、点击倾向和商业价值等异构目标，避免梯度竞争导致的全局次优问题。 |
| [^15] | [Chronicle: Cut-Point Replay for Regression Testing of LLM Agents](https://arxiv.org/abs/2609.20625) | Chronicle通过在非确定性边界记录LLM智能体的运行轨迹并提出切点重放机制，将记录的故障事件转化为可在持续集成中运行的回归测试，解决了LLM智能体故障难以重现的问题。 |
| [^16] | [What Does Privileged Information Add to On-Policy Self-Distillation?](https://arxiv.org/abs/2609.20612) | 论文通过构建包含六种共享答案推理视图的数学问题集AMPLE-Math，发现无参考蒸馏贡献了在策略自蒸馏中的大部分性能提升，而特权信息（如参考答案或完整解题过程）带来的额外收益有限且依赖于学生模型本身。 |
| [^17] | [WiC is Not WSD: A Study on LLMs and Lexical Ambiguity Resolution](https://arxiv.org/abs/2609.20593) | 本研究揭示WiC任务的困难部分源于缺乏明确的词义清单，为模型提供候选词义可显著提升其WiC任务表现，且许多表面错误实为标注歧义或词义边界不匹配，而非模型真正的词汇理解失败。 |
| [^18] | [SAFARI: An Industrial Benchmark for LLM-Assisted Hazard Analysis and Risk Assessment](https://arxiv.org/abs/2609.20584) | 该论文提出了首个面向ISO 26262汽车功能安全领域的LLM辅助危害分析与风险评估工业基准SAFARI（包含3000个去标识化真实工业案例），并通过实验揭示前沿LLM虽能生成合理的危害叙述，但在标准风险分类上表现薄弱（最佳ASIL宏观F1仅0.261），思维链提示反而常降低分类性能。 |
| [^19] | [Steering the Compass: Aligning Dynamic Psychological Counseling Conversations with Cognitive Behavioral Therapy Strategies](https://arxiv.org/abs/2609.20565) | 提出StratCBT数据集，包含9,688个会话和约25.6万条话语，首次将心理咨询对话与八种认知行为疗法策略对齐，弥补了现有研究忽视基于来访者实时心理状态进行动态决策的不足。 |
| [^20] | [Language-model groups overstate consensus when replaying human deliberation on a reasoning task](https://arxiv.org/abs/2609.20543) | 本研究通过让信念锚定的LLM智能体重演人类在华生推理任务中的小组讨论，发现语言模型群体的共识度显著高于人类群体（差距达34至44个百分点），且该结论在多种测量方法下均稳健成立，表明语言模型会系统性高估群体共识。 |
| [^21] | [An Analysis of Training-Free Self-Reported Confidence in Language Models](https://arxiv.org/abs/2609.20541) | 该研究发现语言模型直接口头表达的置信度是出乎意料强的免训练正确性预测信号（AUROC高达0.956），显著优于基于多样本一致性的方法，而自我一致性反而可能放大模型共同的系统性误解。 |
| [^22] | [Relational Attention for Data-Efficient Language Modeling](https://arxiv.org/abs/2609.20530) | 该论文提出在双注意力Transformer架构中将关系注意力与自注意力相结合，并借助BabyLM 2026挑战赛的数据受限环境，验证关系注意力所带来的数据效率能否成功迁移到语言建模任务中。 |
| [^23] | [Model-Agnostic and Language-Agnostic Voice Pipeline Improvement for the Agriculture Domain](https://arxiv.org/abs/2609.20504) | 该论文提出了一种无需微调或替换底层 ASR 模型的模块化、模型无关语音流水线，通过音频增强、说话人分离、农业领域词典纠错和质量门控，显著提升了嘈杂田间环境下农业咨询场景的语音识别质量。 |
| [^24] | [Edustories: A Collection of Real-world Case Studies from Classroom Practices](https://arxiv.org/abs/2609.20484) | 该研究推出了Edustories数据集——包含1,492个教师撰写的真实课堂案例研究，用于评估大语言模型预测教师教学干预成效的能力，并发现当前最强模型的预测准确率仅为58%，仍不及人类专家水平。 |
| [^25] | [Stress-testing Alignment Midtraining](https://arxiv.org/abs/2609.20412) | 该论文通过大规模实验（高达1100亿参数模型和10亿中期训练token）对对齐中期训练（AMT）的多个假设进行压力测试，发现在简单场景下中期训练能够引导模型动机，但其有效性仍存在局限。 |
| [^26] | [Xeno-Interpretability: Investigating the Alien Minds of LLMs](https://arxiv.org/abs/2609.20408) | 本文提出“异种可解释性”这一新研究方向，主张大语言模型内部可能存在人类概念无法充分描述的“异种表征”，其内部区分空间远超有限人类描述所能覆盖的范围，且实验识别与语义解释应当分开对待。 |
| [^27] | [Schema-Anchored Latent Reasoning for Semantic Parsing-Based Knowledge Base Question Answering](https://arxiv.org/abs/2609.20398) | 提出SALR方法，通过在模型隐藏状态中进行模式锚定的潜在多步推理，延迟对逻辑形式决策的显式承诺，避免错误的中间模式选择传播，从而提升基于语义解析的知识库问答性能。 |
| [^28] | [Viveka-Insight: a cross-lingual concept graph and citation-grounded retrieval resource over the complete works of Swami Vivekananda in English and Bengali](https://arxiv.org/abs/2609.20303) | 该论文发布了Viveka-Insight，一个针对斯瓦米·维韦卡南达英文与孟加拉文完整著作的双语开源资源，通过结构保持解析和跨语言概念图解决了古典哲学语料库多语言不对齐、词汇陈旧和文化敏感内容难以验证接地的问题。 |
| [^29] | [Lens: Bringing the Right Semantic Perspective into Focus for Training-Free Multimodal Representation Learning](https://arxiv.org/abs/2609.20252) | 论文指出免训练多模态表征学习中存在语义视角错位问题——现有语义引导方法无法使自回归模型提取的表征聚焦于下游任务所需的语义视角，并提出Lens方法来解决这一问题。 |
| [^30] | [The Public Discourse Corpus (PDC): A Speaker-Attributed Dataset for Valence and Epistemic Modality with Target Speaker Participation](https://arxiv.org/abs/2609.20232) | 该论文发布了首个对公众人物访谈语音进行情感效价与认知情态联合标注的公共话语语料库（PDC），并提出目标说话人参与（TSP）标注方法作为可推广的语料库构建方法学贡献。 |
| [^31] | [Before the Warning Comes Too Late: Incremental Phone-Scam Detection from Speech](https://arxiv.org/abs/2609.20223) | 提出了StreamFraudNet，一种基于冻结自监督语音编码器、循环时间建模和窗口分数聚合的流式电话诈骗检测模型，能够在通话过程中每2秒增量更新诈骗风险评分，在英语基准上达到0.9953的ROC-AUC。 |
| [^32] | [Foundations of Stochastic Lexical Calculus: Semantic Descent and Random Dynamics on Probability Simplices](https://arxiv.org/abs/2609.20207) | 本文建立了一个可观测的理论框架，给出了语言导出的概率能够唯一支持语义状态更新的充要条件，并在平均收缩条件下证明了概率单纯形上随机递归的存在性、唯一性与稳定性，从而构建了一种无需将内部演算归因于语言模型的随机词汇演算。 |
| [^33] | [To Copy or Not to Copy: Controlling Speculative Decoding via Intrinsic Model Signals](https://arxiv.org/abs/2609.20186) | SwitchSD通过在目标模型内部表示上训练轻量级探测器来识别真正的复制意图，从而在神经草稿生成与上下文复制两种策略间自适应切换，有效控制推测解码过程并提升大语言模型推理吞吐量。 |
| [^34] | [Fine-Tuning Models for Biomedical Relation Extraction](https://arxiv.org/abs/2609.20169) | 本研究通过对预训练模型（尤其是DeBERTa和Gemini Pro 1.0）进行微调，实现了生物医学文本中变异-表型关系的自动抽取，其中精心微调的Gemini Pro 1.0在句子级和摘要级任务上均超越了现有最先进水平。 |
| [^35] | [Think Thrice Before Reranking: Multi-perspective Evidence and Reasoning Integration for Text Reranking](https://arxiv.org/abs/2609.20131) | 提出MERIT-Rank框架，通过多轨迹推理空间（MTRS）从多视角评估查询-文档相关性，将多个互补推理轨迹融合为统一排序决策，并配合渐进式排序策略优化（PRPO）训练框架，显著提升了基于LLM的文本重排的鲁棒性。 |
| [^36] | [Design of the IBM Granite 5.0 TurboCTC ASR Model](https://arxiv.org/abs/2609.20104) | Granite 5.0 Turbo CTC 是一个仅用公开数据训练的 4.7 亿参数语音识别模型，通过金字塔下采样、分块自注意力、Muon 优化器和推理优化等技术，在 Open ASR 排行榜上达到速度-精度 Pareto 前沿，速度比最快的竞争对手还快一倍。 |
| [^37] | [MATCH: Model-Aware Tool Learning with Curriculum Scheduling and Hierarchically Gated Rewards](https://arxiv.org/abs/2609.20082) | MATCH提出了一种模型感知的闭环工具学习框架，通过课程难度与策略能力共同演化的课程调度，以及按工具名称、参数键、参数值逐级门控授予信用的分层奖励机制，解决了固定阈值课程脱节与加性奖励信用泄漏两大问题。 |
| [^38] | [Reading Emotions in the Token Space: Discriminative Adaptation of SpeechLLMs for Emotion Recognition](https://arxiv.org/abs/2609.20081) | 提出一种判别式适配方法，通过单层线性分类头读取语音大语言模型最后一个提示词元的隐藏状态来识别情感，在不修改主干网络的前提下提升Macro F1、消除幻觉标签，并具有可解释性。 |
| [^39] | [Marginal utility, matrix factorization, and the Key-Value (KV) cache: a unified information-economic framework for sovereign geo-mining inference](https://arxiv.org/abs/2609.20068) | 本文提出一个统一的信息经济学框架，证明边际效用、矩阵分解与KV缓存压缩三者遵循同一条分配规则（保留特征值超过约束影子价格的最高维度），并将其应用于地理采矿文档的结构化信息自动抽取。 |
| [^40] | [AI Should Facilitate Democratic Deliberation at Scale](https://arxiv.org/abs/2609.20059) | 本立场论文主张 AI 应当在保留人类能动性、鼓励相互尊重、促进平等包容、增强而非取代公民参与四项原则下辅助大规模民主审议，而非以机器判断替代人类选择。 |
| [^41] | [The Missing Complement: State-Conditioned Minimal Sufficient Evidence for Coding Agents](https://arxiv.org/abs/2609.20050) | 该论文提出了状态条件化最小充分证据恢复这一新问题并构建了SERBench基准，同时提出MSS-Complement方法，将证据获取从排序转变为集合构建，为编码代理的决策恢复紧凑且充分的证据组合。 |
| [^42] | [Geopolitical Divisions Across Languages in Large Language Models](https://arxiv.org/abs/2609.20005) | 该研究通过对GPT、Claude和Gemini进行112种语言、共67,200次响应的大规模实验，首次发现大型语言模型对乌克兰战争的评估随提问语言而显著变化，且各语言间的回复倾向分布与全球地缘政治分歧格局（公众对俄态度、联合国投票及对乌援助）高度吻合。 |
| [^43] | [Benchmarking LLM Compliance with China AI Generated Content Regulations](https://arxiv.org/abs/2609.19989) | 本文设计了一个包含六个维度、2303个问题（含203个自建宪法问题）的中文合规评估框架，对20个大语言模型在中国AI生成内容法规下的合规性进行了基准测试，发现国际模型同样表现出较高合规性，主要差异集中在意识形态对齐相关维度。 |
| [^44] | [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/abs/2609.19969) | DeepSeek-V4.1-Flash通过因果编码器-解码器架构和跨层KV缓存复用等KV缓存压缩技术，在支持百万token上下文的同时，将预填充阶段激活参数量减半至80亿，大幅降低了长时程智能体工作负载的计算、存储和带宽成本。 |
| [^45] | [Before the Arrest: Benchmarking LLMs on Criminal Profiling from Incomplete Evidence](https://arxiv.org/abs/2609.19965) | 该论文提出了包含五个国家2500个真实凶杀案例的PIJ基准，首次评估大语言模型在逮捕前阶段基于不完整证据进行犯罪侧写、犯罪过程重建和刑罚预测的能力，发现模型在从显性事实提取过渡到隐性推理时性能系统性下降。 |
| [^46] | [Intrinsic Sequence-Likelihood Confidence in Retrieval-Dominated Extractive QA: Two Pre-Specified Negatives, and What They Do and Do Not Attribute](https://arxiv.org/abs/2609.19942) | 该研究通过预先注册的评估标准证明，在检索已能恢复92-99.8%最优性能的抽取式问答场景中，模型的内在序列似然置信度无论作为蒸馏触发器还是路由弃答策略的控制信号均告失效。 |
| [^47] | [KoNeoBench: A Curated Evaluation Dataset for LLM Understanding of Korean Neologisms](https://arxiv.org/abs/2609.19916) | 该论文提出了KoNeoBench，一个基于2020年以来在线新闻中1,785个经专家审校的韩语新词构建的评测基准，通过四个任务评估大语言模型对韩语新词的理解能力，弥补了现有静态基准对新兴词汇变化覆盖不足的缺陷。 |
| [^48] | [Generalization through Lexical Abstraction in Transformer Models: The Case of Functional Words](https://arxiv.org/abs/2609.19887) | 本文探究预训练Transformer模型能否像人类一样利用代词、副词等功能词进行词汇抽象，通过在嵌入空间中比较名词与其可替代代词的表示，检验模型是否能识别词汇化句子与功能化句子之间的句法和语义平行关系。 |
| [^49] | [Evaluating Communicative Success in Machine-Translated Conversation](https://arxiv.org/abs/2609.19885) | 该论文提出了一个可复用的三层评估框架，从语义、语用和文化-社会三个维度衡量机器翻译口译智能体的对话交际成功度，突破了传统只测句子忠实度的评估局限。 |
| [^50] | [PetriBench: Benchmarking LLM Reasoning over Dynamic State Spaces](https://arxiv.org/abs/2609.19883) | 本文提出PetriBench，一个基于Petri网的紧凑、自包含且可扩展的基准测试，用于评估大语言模型在动态状态空间上的推理能力，发现模型准确率随任务难度增加而一致下降，且测试时计算对不同推理任务的提升效果各异。 |
| [^51] | [D-Quant: Driftable Entropy Coding for KV Cache Quantization](https://arxiv.org/abs/2609.19880) | 本文提出D-Quant方法，利用KV缓存值近似正态分布的特性，通过可漂移熵编码突破固定宽度量化级别数的指数限制，在有效压缩KV缓存内存占用的同时保持模型性能。 |
| [^52] | [V\={a}kQA: A Benchmark and Evaluation Study for Telugu Spoken Factoid Question Answering](https://arxiv.org/abs/2609.19879) | 本文提出了首个泰卢固语口语事实型问答基准VākQA（包含2,001个问答对、语音音频及人工验证答案），并验证了自动评估方法的可靠性，系统评测了各类模型在不同模态、语言和领域下的表现。 |
| [^53] | [Uni-LaDiR: Latent Diffusion Unifies Multimodal Reasoning](https://arxiv.org/abs/2609.19878) | Uni-LaDiR提出将多模态推理步骤映射到共享潜在空间，并利用扩散模型预测下一块思维标记，从而在统一的潜在表示中实现灵活的多模态推理。 |
| [^54] | [JustMem: Just-Enough Memory Access for Long-Term Conversations](https://arxiv.org/abs/2609.19877) | 该论文提出JustMem框架，将对话历史存储为紧凑的原子记忆，并针对每个查询在发现广度和阅读保真度两个维度上自适应地调整记忆访问策略（LOOKUP、COMPOSE、REPLAY），从而在长期对话中实现恰到好处的高效证据检索。 |
| [^55] | [Zarya: A Hybrid Autoregressive--Masked Diffusion Language Model with Flexible Training and Dual-Mode Inference](https://arxiv.org/abs/2609.19868) | Zarya提出了在单一架构中联合优化自回归与掩码扩散目标的混合语言模型家族，通过可变槽位大小的课程训练实现从细粒度AR学习到粗粒度扩散学习的平滑过渡，并支持MDM采样和槽位化投机解码两种解码范式。 |
| [^56] | [Reproducibility is not construct validity: LLM measurement of institutionally situated communication](https://arxiv.org/abs/2609.19866) | 该研究利用欧盟《人工智能法案》咨询数据证明，大语言模型标注的高可复现性并不等于构念效度，且基于文本的测量与问卷测量之间的分歧在不同利益相关方群体间存在系统性差异。 |
| [^57] | [F$^{2}$DR: A Fine-Grained Full-Pipeline Reward Framework for DeepSearch Workflows](https://arxiv.org/abs/2609.19827) | 该论文提出了F2DR框架，从内容、轨迹和答案三个维度对DeepSearch工作流进行细粒度全流程奖励评估，并构建了专门基准DeepSearch RM-Bench，显著提升了评估一致性。 |
| [^58] | [Dictionary-Constrained Grapheme-to-Phoneme for Unsegmented Languages from LLM-Annotated Data](https://arxiv.org/abs/2609.19805) | 本文提出一种利用词典构建词格并采用条件随机场评分的上下文感知神经G2P方法，结合大语言模型生成的超过200万条标注数据，显著提升了日语等未分词语言的字素到音素转换性能。 |
| [^59] | [Evolution or Illusion? Rethinking Evaluation in LLM Evolutionary Search](https://arxiv.org/abs/2609.19799) | 该论文通过在种子数与迭代数的完整组合网格上系统评估三种LLM进化搜索策略，揭示了固定预算在“宽度”（更多种子）与“深度”（更多迭代）之间的最优分配方式以及策略排名都会随策略、任务和总预算显著变化，证明传统单预算设置下的评估结论并不可靠。 |
| [^60] | [Learn Before You Judge: Progressive Knowledge-to-Decision Alignment for Explainable Hateful Meme Detection](https://arxiv.org/abs/2609.19778) | 针对现有“先解释后检测”方法中解释生成与标签预测耦合干扰的问题，本文提出渐进式知识-决策对齐方法ProKDA，通过智能体构建外部背景知识并分离任务目标，显著提升可解释仇恨模因检测的性能。 |
| [^61] | [AutoData: Agentic Search for Pre-training Data Selection](https://arxiv.org/abs/2609.19754) | AutoData通过智能体在可执行的数据选择算法空间中直接搜索，并利用代理模型的验证反馈迭代改进，仅一夜之间就能自动发现超越人工设计流水线的预训练数据选择算法。 |
| [^62] | [A Phonemically Comprehensive, ASCII-Only Romanization Scheme for Thai and Lao: Systematic Cross-Lingual Correspondence and Chinese-User-Friendly Design](https://arxiv.org/abs/2609.19736) | 本文提出了一种仅使用ASCII字符、音位全面的泰语与老挝语统一罗马化方案，该方案保持两种语言间的系统性对应，并通过与拼音的兼容设计照顾中文用户的使用习惯。 |
| [^63] | [Learn Your Own Thoughts: Abstract Token Curriculum](https://arxiv.org/abs/2609.19717) | 提出了抽象token课程学习（ATC）框架，无需直接监督或手动设计草稿板，即可通过逐步增加问题复杂度训练模型在连续表示空间中自发形成内部抽象思维，并从理论和实验上证明了其相对于以往连续思维训练方法的优势。 |
| [^64] | [Improving Cross-Lingual Transfer for Sequential Sentence Classification in Research Papers via Structural Similarity](https://arxiv.org/abs/2609.19650) | 该论文构建了涵盖13种非英语语言的多语言序列句子分类数据集，并发现利用标签序列和位置规律等在跨语言间保持一致的结构相似性，可以有效提升研究论文序列句子分类的跨语言迁移效果。 |
| [^65] | [Scientific Image Quality Assessment via Multi-modal Retrieval-Augmented Generation](https://arxiv.org/abs/2609.19634) | 本文提出基于多模态检索增强生成的科学图像质量评估框架，通过多路检索与融合机制增强大语言模型评估复杂科学图像的能力，在ICME 2026 SIQA挑战赛SIQA-U赛道中夺得第一名。 |
| [^66] | [From Intent to Action: Benchmarking LLM Safety in Vehicle Voice Command Authorization](https://arxiv.org/abs/2609.19630) | 该论文首个针对车辆语音指令授权问题提出了包含202个场景、七类行动决策的基准测试，发现大语言模型的决策一致性从40.1%到89.1%不等，其中基于API的模型表现最好且相互之间无显著差异。 |
| [^67] | [Semantic Layer Induction from Raw Telemetry via Hierarchical LLM and RAG Abstraction](https://arxiv.org/abs/2609.19615) | 提出了一种端到端框架，通过层次化LLM推理与两阶段语义抽象流水线，从嘈杂的原始遥测日志中全自动构建业务语义层，免除了人工解析和脆弱映射维护的负担。 |
| [^68] | [Chain-of-Thought Entropy as a Reliability Signal: A Preregistered Reproduction](https://arxiv.org/abs/2609.19606) | 本预注册独立复现研究证实，大语言模型思维链熵轨迹的形状（而非总熵下降幅度）是预测答案正确性的可靠信号，形状信号在四个开源模型上成功复现，而幅度信号则因实验设置而异。 |
| [^69] | [Full-Duplex Speech Models Take the Floor When Asked, Not When Needed](https://arxiv.org/abs/2609.19596) | 该研究通过语境匹配的英语独白实验发现，现有全双工语音模型主要在被直接称呼或出现沉默时才发言，而无法像人类那样在听到错误事实或危险信息时主动插话纠正或警告。 |
| [^70] | [Form Over Content In Gradient-Based Data Attribution Methods](https://arxiv.org/abs/2609.19589) | 基于梯度的数据归因方法主要捕捉答案格式而非任务内容，因为共享答案格式的数据集表现出强梯度对齐，而任务相同但格式不同的数据集则不对齐。 |
| [^71] | [Red-Teaming Auto Mode: Improving Blocking Classifiers Against Malign Coding Agents](https://arxiv.org/abs/2609.19587) | 本文通过红队测试发现，失对齐的恶意编码智能体在高层攻击策略指导下，能够通过提示注入、多智能体攻击和恶意压缩等机制，在 79% 的试验中绕过生产级拦截监视器并造成灾难性危害，据此提出了改进拦截分类器的方法。 |
| [^72] | [CliniCIRCA: A Modular LLM Framework for Constructing Longitudinal Mental Health Patient Journeys from Raw EHR Narratives](https://arxiv.org/abs/2609.19585) | CliniCIRCA是首个无需事件级时间戳即可从非结构化出院总结中对临床事件进行时间分类的多阶段大语言模型框架，通过临床医生参与纠错生成黄金标准标签，并支持基于时间线的患者历程总结。 |
| [^73] | [Large Language Model Agents for Evidence Based Genetic Disease Severity Classification](https://arxiv.org/abs/2609.19569) | 该研究开发了一个结合ReAct与RAG的自主AI智能体，基于ACMG严重程度指南和ACOG生活质量标准检索并验证文献证据，实现了对10,211个人类表型本体术语的遗传病严重程度自动化分类，表型分类准确率达93.55%，并汇总基因层面的严重程度以识别严重的常染色体隐性遗传基因对。 |
| [^74] | [From Parameters to Behaviors: A Survey of Model Fusion for Large Language Models](https://arxiv.org/abs/2609.19553) | 本综述首次给出模型融合的统一定义，并建立了参数级、表示级和行为级三个层次的系统分类体系，同时梳理了相关指标、基准、应用、挑战与未来方向。 |
| [^75] | [Finding Common Ground: Graded Communal Knowledge in Bluesky Starter Packs](https://arxiv.org/abs/2609.19549) | 本研究首次利用 Bluesky 起始包作为可见的社区归属标签，通过对 191,648 对用户共享词汇库的分析，实证验证了 Clark 共同基础理论中“共享社区归属越多、共同基础越大”的分级假设。 |
| [^76] | [When Hiring Becomes Agent-Mediated: Evaluating Access and Recurrence in Two-Agent R\'esum\'e Screening](https://arxiv.org/abs/2609.19530) | 该论文提出一种由雇主方和候选人方智能体相互交流证据并更新判断的双智能体简历筛选方法，发现相比传统的单次调用筛选，它能显著提升边界案例的通过率，且决策在双向都发生变化而非单纯放宽标准。 |
| [^77] | [EconSkills: Studying Skill Transfer and Retrieval for Web Agents on Live Economic Data](https://arxiv.org/abs/2609.19523) | EconSkills框架将验证过的经济数据检索轨迹提炼成参数化技能库，证明技能迁移和基于库的检索能显著提升Web智能体的表现。 |
| [^78] | [For Your Eyes Only: Evaluating Coordination Between Isolated Language Model Instances](https://arxiv.org/abs/2609.19504) | 该论文提出了一个名为“仅限你的眼睛”的合作信号博弈框架，用于评估隔离的语言模型实例能否仅通过自然语言中的隐藏信号实现协调，发现大多数模型在需要避免可检测信号时难以维持协调能力，而一个前沿模型仍能保持近乎完美的表现。 |
| [^79] | [Safety Beyond the Interface: Detecting Harm via Latent States in Large Language Models](https://arxiv.org/abs/2609.19472) | 该研究通过从LLaMA-3.1-8B内部激活值中训练仅1260万参数的轻量级MLP探针来检测有害提示，实现了与规模大1000倍的防护模型相当的检测性能（F1最高达99%），同时显著降低了延迟和计算成本。 |
| [^80] | [From Models to Systems: A Comprehensive Survey of Efficient Multimodal Learning](https://arxiv.org/abs/2609.19445) | 本综述首次提出涵盖模型、算法和系统三个层次的结构化高效多模态学习分类体系，并系统综合了跨层协同设计的方法论，以应对“效率-效用-隐私”的根本性权衡。 |
| [^81] | [BurnRiSc: Toward Non-Invasive Burnout Screening in Open Source from Public Repository Signals](https://arxiv.org/abs/2609.19422) | BurnRiSc框架利用GitHub公共活动数据中提取的14个行为和语言信号，将Oldenburg倦怠量表的疲惫和疏离两个维度操作化，实现了对开源维护者倦怠的无创、可追溯筛查，解决了自我报告量表无法触及最需要帮助人群的难题。 |
| [^82] | [Less Is More: Graph-free Multimodal RAG via Multi-signal Late Fusion](https://arxiv.org/abs/2609.19417) | TrioRAG是一个无图多模态RAG框架，通过对问题、锚点图像和VLM增强查询三种信号独立检索并后期融合，在降低成本的同时达到或超越基于图的系统性能，并引入了基于网络嘈杂图像的汽车领域多模态基准AutoQA。 |
| [^83] | [A Cross-Lingual Acoustic Disease-Alignment Framework for Respiratory Health Assessment from Spontaneous Speech](https://arxiv.org/abs/2609.19398) | 提出跨语言疾病对齐框架CL-DAF，通过识别疾病效应跨语言一致的26个声学特征，克服了基于语音的呼吸健康评估中因语言特异性语音变异导致的跨语言迁移难题，将孟加拉语到英语的迁移AUC从0.49大幅提升至0.825。 |
| [^84] | [Riemannian--Lorentz Fusion of Vision Transformers and State-Space Models](https://arxiv.org/abs/2609.19384) | 该论文提出RLPF方法，通过将语义角色对齐的参数组提升至洛伦兹双曲面并计算正则化测地重心，实现了视觉Transformer与状态空间模型这两种异构架构的参数融合。 |
| [^85] | [The Role of Fine-grained Harm Signals in LLM Safety](https://arxiv.org/abs/2609.19366) | 该研究通过从危害表征中去除通用成分、分离出正交的类别残差，并利用激活引导技术，揭示了细粒度类别特异性危害信号在11个风险类别中的编码程度因类别而异且跨模型一致，而其引发拒绝行为的能力则更依赖于具体模型。 |
| [^86] | [A frontend-backend architecture for tool calls in full-duplex speech models](https://arxiv.org/abs/2609.19334) | 提出一种前后端架构，让全双工语音模型通过发出委派标记将流式转写交给文本LLM后端执行工具调用，并以轻量级注入机制返回结果，从而在几乎不修改前端模型的前提下保留低延迟、可打断的自然双工交互。 |
| [^87] | [AUDITPLAN: Commit, Then Answer for Auditable Safety Alignment](https://arxiv.org/abs/2609.19325) | 提出AUDITPLAN方法，让模型先输出结构化安全计划再据此作答，并通过FAITHGATE奖励门控机制确保答案忠实于计划，从而同时提升大模型安全对齐的鲁棒性与可审计性。 |
| [^88] | [Why Pretraining Fails to Share Cross-Lingual Knowledge](https://arxiv.org/abs/2609.19291) | 本研究通过受控双语预训练实验发现，不相交的词表空间是跨语言知识泛化的根本障碍——即使是对同一语言的完全相同副本，仅仅词表不相交就足以导致知识隔阂。 |
| [^89] | [YNU-HPCC at SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Using Multiple Prediction Headers](https://arxiv.org/abs/2609.19238) | 该论文提出采用RoBERTa模型并改进输出头为单一预测头，同时将多语言数据集统一翻译成英文进行训练，实验证明单预测头和统一英文数据集训练的方法在情感识别任务中表现更优。 |
| [^90] | [CovR: Coverage-Aware Hardware Verification via Reasoning-Guided Reinforcement Learning](https://arxiv.org/abs/2609.19189) | CovR是一个结合自我反思循环与仿真反馈的智能体框架，通过推理引导的强化学习自动生成硬件测试平台，突破了现有方法只关注功能正确性的局限，实现了验证覆盖率的最大化。 |
| [^91] | [Message capacity and claim wording set the transition points of collective truth-finding in language-model networks](https://arxiv.org/abs/2609.19183) | 该研究将智能体阅读他人消息的数量上限建模为“消息容量”，发现LLM对声明的集体判断可归结为带分裂归一化权重的随机二值神经元更新规则，并据此预测当智能体平均阅读其31个信源中不足6.4个时，错误共识从任何初始状态都无法形成，且声明措辞与消息容量共同决定了集体真相探寻的转折点。 |
| [^92] | [What Do We Expect from LLMs? Mapping the Design of LLM Benchmarks](https://arxiv.org/abs/2609.19182) | 该研究系统梳理了2022年至2026年间14,767篇引入或更新大语言模型评估资源的arXiv论文，绘制出基准测试设计的演变图谱，揭示评估正日益强调行动、交互和专业应用，且基于LLM的评分与模型生成材料在各类基准中的参与度发展不均衡。 |
| [^93] | [To Memories and Beyond: From Remembering to Knowing You across Long-Term Multimodal Personal Archives](https://arxiv.org/abs/2609.19167) | 该论文提出了首个基于真实多年个人视觉档案构建的多模态长期记忆基准ReaLMem，通过事实回忆、个性推断和预测性个性化三个认知层级，推动AI从单纯记住用户事件走向真正理解用户。 |
| [^94] | [Advantage Scale Calibration Imbalance in Group-Relative Optimization under Low-Variance Rewards: Diagnosis and Bounded Recovery](https://arxiv.org/abs/2609.19164) | 本文诊断了低方差奖励下群体相对优化中优势尺度校准失衡的问题，提出三方校准接口揭示RLOO/Dr.GRPO与GRPO各自的失控行为，并通过奖励分辨率协议与MaxNorm-AC过滤亚分辨率噪声、实现对可信小差距的有界恢复。 |
| [^95] | [VisKG-LM: Compiling Knowledge Graphs into Visual Memory for Multiple-Choice Question Answering](https://arxiv.org/abs/2609.19158) | VisKG-LM 将检索到的知识图谱子图一次性离线编译为保留分支结构的可视化图像并缓存为只读记忆，使语言模型在推理时无需重复在线编码图结构，从而将图编码与语言推理解耦，提升多项选择题问答的效率。 |
| [^96] | [Reflective Recovery: A Self-Supervised Method for Reasoning by Learning from Mistakes](https://arxiv.org/abs/2609.19156) | 本文提出“反思性恢复”这一自监督方法，通过将LLM失败的推理尝试转化为恢复训练数据，教会模型从错误中恢复，从而突破了仅依赖完美推理轨迹的模仿学习方法在数据有限时的“规模坍塌”瓶颈。 |
| [^97] | [Towards Proactive Detection of User-Side Implicit Conflicts in Human-LLM Dialogue](https://arxiv.org/abs/2609.19155) | 该论文构建了首个用于评估用户侧隐式冲突检测的人工标注基准UC-Bench，并通过数据合成方法提升轻量级LLM在有限训练数据下主动检测用户侧隐式冲突的能力。 |
| [^98] | [Neo-Classic: A Benchmark for Evaluating Linguistic-Aesthetic Reasoning in Classical Chinese Poetry](https://arxiv.org/abs/2609.19154) | 该论文提出 Neo-Classic 基准，利用当代专家创作的严格合律诗歌和逆向理解探针来测试大语言模型的古典诗歌语言-审美推理能力，发现最先进模型在分层约束满足上存在20%至50%的性能差距等显著局限。 |
| [^99] | [Stop Removing Stopwords: How an Inherited Preprocessing Default Distorts Legal Text-as-Data](https://arxiv.org/abs/2609.19153) | 本研究通过穷尽式单词消融实验，首次直接针对下游分类目标验证停用词去除这一沿袭自信息检索时代、从未被验证过的预处理默认设置，揭示其可能扭曲实证法学中基于TF-IDF和线性分类器的文本数据分析结果。 |
| [^100] | [FakeSpotter: A content and strategy agnostic Viral Misinformation Detection Tool](https://arxiv.org/abs/2609.19152) | FakeSpotter通过测量虚假信息的结构指纹而非直接判定真伪，实现了内容与策略无关的病毒式虚假信息风险评估，在长短文本上分别取得0.788和0.793的宏观F1分数。 |
| [^101] | [What Users Think of Generative AI: A Cross-Platform NLP Analysis of Trust and Friction in App Store Reviews](https://arxiv.org/abs/2609.19151) | 该研究首次对ChatGPT、Gemini、Claude等六大生成式AI应用在应用商店的17,012条评论进行大规模NLP分析，通过BERTopic主题建模与RoBERTa情感分类揭示用户负面情绪主要集中于广告、身份验证、服务器可靠性和订阅定价等采用障碍。 |
| [^102] | [Sampling Reveals Style: Unsupervised, Training-Free Discovery of Prompt-Conditional Stylistic Axes in LLM Activations](https://arxiv.org/abs/2609.19150) | 该论文提出一种无需训练的无监督方法，通过对同一提示的高温重复采样补全进行主成分分析，自动发现并标记大语言模型激活中与提示相关的风格轴，并通过245个人类风格标注验证了其与人类自发风格需求的高度契合。 |
| [^103] | [Subliminal Prompting Beyond Static Geometry: Causal Depth and Multi-Token Confounds](https://arxiv.org/abs/2609.19149) | 该论文首次将标记纠缠解释中的相关性测量与因果性测量明确区分开，通过在多个深度进行隐藏状态复制的因果干预实验，发现静态输出向量相似度随模型规模增大而失去预测力，而隐藏状态对所传递特质的因果控制能力则显著增强。 |
| [^104] | [Modality Discrepancy Transformer for Ambivalence and Hesitancy Recognition](https://arxiv.org/abs/2609.19148) | 提出模态差异Transformer（MDT），将跨模态矛盾信号显式建模为9-token表示（模态嵌入、绝对差特征与Hadamard积差异特征），结合FiLM文本条件调制、LoRA微调及文本引导后期融合，实现临床视频中矛盾与迟疑情感状态的自动识别。 |
| [^105] | [FRAUDSkill: Structured Frozen-Weight Skill Optimization for Audio Anti-Fraud Detection](https://arxiv.org/abs/2609.18766) | 本文提出FRAUDSkill框架，在不修改底层音频-语言模型参数的情况下，通过外部优化技能程序、路由策略和决策规则，实现了能够灵活适应欺诈模式演变的结构化音频反欺诈检测。 |
| [^106] | [TeleAntiFraud 2.0: A Refreshable, Profile-Grounded, and Audio-Based Benchmark for Telecom Fraud Detection](https://arxiv.org/abs/2609.18748) | 提出了 TeleAntiFraud 2.0，一个可按月更新、基于用户档案且能在共享上下文中区分欺诈与合法近域通话的音频电信欺诈检测基准。 |
| [^107] | [SEA-LION-v4.8: A Technical Report](https://arxiv.org/abs/2609.18310) | 基于NVIDIA Nemotron 3构建的SEA-LION-v4.8东南亚语言模型家族，通过持续预训练、监督微调和在线同策略蒸馏，显著提升了七种东南亚语言在指令遵循、推理和理解任务上的表现。 |
| [^108] | [Rollback the World, Keep the Reflection: Rollback-Induced Reflection for Long-Horizon LLM Agents](https://arxiv.org/abs/2609.18304) | 提出了回滚诱导反思（RIR）统一恢复框架，在将LLM智能体回滚到选定先前状态的同时，保留从被放弃轨迹中提炼的可复用知识，解决了长程任务中错误累积且难以可靠恢复的问题。 |
| [^109] | [${M}^2$Tok: Multi-head Multi-codebook Discrete Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2609.18259) | 提出 M²Tok，一种多头多码本离散动作分词器，通过将潜在动作特征分解为多个头并采用多个码本以最小化重构误差，突破“离散化瓶颈”，从而提升视觉-语言-动作模型的控制性能。 |
| [^110] | [Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches](https://arxiv.org/abs/2609.17652) | Fathom提出一种让每个查询自适应决定键通道读取位数的稀疏解码方法，通过比特平面存储与逆注水式比特预算分配，在百万token卸载KV缓存场景下实现比现有136位扫描方法快1.67倍的GPU解码速度，同时保持更低注意力误差。 |
| [^111] | [RiskChainBench: A Benchmark for Obfuscated Platform Message Restoration and Evidence-Grounded Web Investigation](https://arxiv.org/abs/2609.16900) | 该论文提出了RiskChainBench基准测试，首次将混淆平台消息的还原任务与基于证据的网络风险调查任务串联评估，弥补了现有基准将混淆文本与风险网页割裂评估、无法衡量目标还原因何影响下游证据获取的不足。 |
| [^112] | [How Humans and LLMs Read Gender into Gender-Neutral Physical Descriptions](https://arxiv.org/abs/2609.16366) | 本研究构建了包含316个身体属性及14,706个人类性别关联评分的GAPA数据集，发现看似“客观中立”的身体描述实际上承载着结构化的性别关联，并评估了16个大语言模型与人类评分的匹配程度。 |
| [^113] | [Verifiable by Construction: Claim-Level Evaluation of Verbatim Citation in Clinical Question Answering](https://arxiv.org/abs/2609.15964) | 该论文基于四份临床实践指南构建了标准化评估框架，从为每个事实性声明提供引用、生成逐字引用到确保引用完全支撑声明，端到端地评估了十二个大语言模型在临床问答中构建可验证答案的能力。 |
| [^114] | [SlopShape: Identifying AI-Generated Commercial Web Content](https://arxiv.org/abs/2609.15369) | 该研究提出通过结构特征（信息呈现方式、顺序、证据与语气）而非词级特征来识别商业网页中的AI生成内容，仅用187个结构特征就在模型自我改写的对抗条件下仍保持约98%的检测性能。 |
| [^115] | [MUSE: A Theory-Harnessed Story Engine for Vibe Narrativizing](https://arxiv.org/abs/2609.15188) | 提出了MUSE故事引擎，通过将罗伯特·麦基故事理论工程化为针对具体创作决策的指导，并使其贯穿规划、起草和修改全过程，实现将自然语言写作需求转化为高质量完整故事的“氛围叙事”任务。 |
| [^116] | [An Efficient and Modular Framework for Targeted Harm Mitigation in LLMS](https://arxiv.org/abs/2609.13624) | 提出了一种结合Activated LoRA适配器与上下文感知路由机制的模块化纠正框架，可在生成过程中以低延迟、有针对性的方式缓解大语言模型的有害输出，同时提升模型对齐性能。 |
| [^117] | [Causal Analysis and Mitigation of Spurious Onsets in Full-Duplex Speech LLMs](https://arxiv.org/abs/2609.13445) | 本研究通过因果分析发现，全双工语音LLM（如Moshi和PersonaPlex）在用户静默时产生虚假语音起音，是因为模型以自身非语音输出为条件导致语音起音概率在单个80毫秒帧内飙升超过九个数量级，而非重复采样所致，并进一步基于因果反事实分析提出了缓解方法。 |
| [^118] | [Factors Influencing the Emergence of Dependency Length Minimization in Neural Agent Simulations](https://arxiv.org/abs/2609.06025) | 本研究基于循环神经网络的智能体语言学习与交流模拟框架，在更贴近现实的交互情境中探究依存长度最小化偏好如何涌现及其影响因素，为解答这一偏好是否源于高效信息处理约束提供新途径。 |
| [^119] | [Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation](https://arxiv.org/abs/2609.02998) | 该论文提出教师门控在线策略蒸馏（TGOPD），通过经验证器评分的教师探测在提示级别先验证教师模型的可靠性，将可靠提示路由到密集OPD监督、不可靠提示路由到基于验证器的GRPO，从而避免“自信但错误”的教师模型诱导误导性更新。 |
| [^120] | [Scaling phoneme-based TTS augmentation for ASR: A unified pipeline and controlled study](https://arxiv.org/abs/2608.26697) | 本文提出了一种基于音素的统一TTS到ASR增强流程，并引入音素频率引导选择（PFGS）方法，在多种语言的ASR任务中有效提升了性能。 |
| [^121] | [Mitigating Fabrication in Multi-Stage LLM Pipelines for Hiring: An Empirical Evaluation of Prompt Guardrails and Human-in-the-Loop Checkpoints](https://arxiv.org/abs/2608.26171) | 该论文通过实证评估表明，仅靠提示护栏不足以消除LLM招聘流水线中的捏造问题，而结合人在回路检查点能显著降低捏造率并消除身份虚构。 |
| [^122] | [The "Curse of Knowledge" in LLM Query Simulation: Concept Provenance for Tracing Answer-Side Intrusion](https://arxiv.org/abs/2608.25245) | 本文提出概念溯源框架，用于识别大语言模型生成查询中预设答案侧知识的“知识诅咒”现象，该框架能有效区分人类变异与答案侧侵入，并发现候选答案侧概念普遍存在。 |
| [^123] | [LongWoF-Bench: Evaluating EvoMap Genes for Verifiable Long-Workflow Tasks](https://arxiv.org/abs/2608.23200) | 本文提出LongWoF-Bench基准和EvoMap方法，通过将验证器确认的执行轨迹整合为结构化基因，实现经验复用，在可验证长工作流任务中显著优于技能方法。 |
| [^124] | [Towards Safer RAG: Only Agents Capable of System 2 Thinking may Access Untrusted Documents](https://arxiv.org/abs/2608.17153) | 本文提出一种新的安全原则，即仅允许具备系统2推理能力的代理访问不可信文档，以减少RAG系统中的知识投毒攻击影响，并引入新指标量化检测与影响间的差异。 |
| [^125] | [Decoupled Contrastive Decoding via Expert-Aligned Drafting](https://arxiv.org/abs/2608.12913) | 本文提出解耦对比解码（DCD），通过专家对齐的轻量级提议者进行草拟，仅在验证阶段应用业余模型，既保持了原始对比解码的输出分布，又避免了草拟阶段引入的误差放大问题，从而更高效且稳定。 |
| [^126] | [ViTOED: A Dataset for Target-Oriented Emotion Detection on Vietnamese Social Media Texts](https://arxiv.org/abs/2608.12776) | 该论文提出了一个针对越南社交媒体文本的目标导向情感检测数据集ViTOED，并基于结构化情感图建立了基线模型，揭示了越南语言中的特有挑战。 |
| [^127] | [Keep It Simple: Multi-Key Episodic Memory Retrieval for Ultra-Long Video Understanding](https://arxiv.org/abs/2608.07663) | 提出MERIT框架，在记忆构建阶段采用多键情景表示以保证高召回率的精确检索，并将查询特定的高级关系组合延迟到推理阶段通过时间扩展完成，从而以简洁的方式实现超长视频理解。 |
| [^128] | [When Self-Evolution Backfires: Pre-Commit Gating against Skill Contamination in LLM Agents](https://arxiv.org/abs/2608.05810) | 本文揭示了大语言模型智能体自我进化中的技能污染相变现象且该污染在结构上不可逆，提出VaG（验证者即守门人）机制，通过渐进式信任层级的预先承诺式技能准入门控，在缺陷技能进入决策上下文前予以拦截。 |
| [^129] | [MyMentorLLM: A psychotherapy GenAI environment with multimodal voice/text patients, trainees and experts for deliberate practice](https://arxiv.org/abs/2607.25667) | 提出了MyMentorLLM——一个包含2,100次完整CBT会谈的多模态心理治疗刻意练习环境，其中LLM模拟患者、受训治疗师与专家督导三方互动，实验表明模拟患者情感表现与真实障碍一致，且LLM学员的治疗能力在多数条件下超过人类水平。 |
| [^130] | [Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.21653) | 提出了 Molt 框架，通过可组合模型并行、统一智能体接口、全异步 rollout 与优化以及分布式经验存储，实现了万亿参数规模下的智能体强化学习训练，且无需修改现有智能体的执行逻辑。 |
| [^131] | [Self-State Attacks on Self-Hosted AI Agents: How Far Can OS Defenses Go?](https://arxiv.org/abs/2607.17986) | 该论文首次形式化了针对自托管AI代理的“自状态攻击”空间，并通过系统评估证明现有操作系统防御机制存在根本性局限——文件级控制要么留下替代攻击路径、要么误伤合法更新，检测器要么大量误报、要么覆盖不全。 |
| [^132] | [Limits of Reliability and Scaling in Language Models](https://arxiv.org/abs/2607.14112) | 该论文从信息论第一性原理证明了每个生成任务都存在不可逾越的可靠性上限，并推导出一条统一的规模化定律——LLM性能的瓶颈由训练数据与模型容量中更稀缺的资源决定，且Chinchilla定律成为其特例。 |
| [^133] | [CORTEX: High-Quality Cross-Domain Organization of Web-Scale Corpora through Ontological Corpus Graph](https://arxiv.org/abs/2606.30175) | 该论文提出了Cortex框架，通过本体语料库图（OCG）这一三层异构结构，首次将网络规模语料库构建从扁平文档筛选提升为结构化知识组织，实现了高质量的跨领域语料组织。 |
| [^134] | [IHDec: Divergence-Steered Contrastive Decoding for Securing Multi-Turn Instruction Hierarchies](https://arxiv.org/abs/2606.29960) | 提出无需训练的IHDec方法，利用Jensen-Shannon散度自动检测多轮对话中的词元级指令层级违规，并通过动态对比解码抑制低优先级角色的影响，在多轮指令冲突中性能超越基于训练的基线方法。 |
| [^135] | [When Retrieval Metrics Mislead: Measuring Policy Signal in Long-Horizon Tool-Use Agents](https://arxiv.org/abs/2606.23937) | 该研究发现精确匹配检索召回率是一个具有误导性的代理指标——即使正确的治理规则仅在 7% 的情况下被排名第一检索到，检索到的断言仍能让分类器取得与使用黄金规则几乎相同的性能。 |
| [^136] | [Do LLM Attribution Metrics Transfer? Auditing Retrieval-Augmented Generation Evaluation Across Datasets and Constructs](https://arxiv.org/abs/2606.23915) | 该研究系统审计了八种检索增强生成归因自动指标，发现在生成答案归因构念下没有任何指标能在所有数据集上保持与最佳指标一致的性能，不同数据集上的指标排名甚至完全反转，因此实践中将这些指标视为可互换是不成立的。 |
| [^137] | [Redact or Keep? A Fully Local AI Cascade for Educational Dialogue De-Identification](https://arxiv.org/abs/2606.18372) | 提出一种完全本地运行的AI级联框架，将教育对话去标识化重新定义为“删改/保留”的受限隐私分诊任务，无需将学生数据发送给第三方，即可解决商用LLM与本地NER系统在隐私治理与识别准确性之间难以兼得的权衡问题。 |
| [^138] | [Evaluating Bias in Phoneme-Based Automatic Speech Recognition Systems: An Analysis of IPA Transcription Models](https://arxiv.org/abs/2606.11639) | 本研究首次针对基于音素的自动语音识别系统评估人口统计学偏差，分析了两个最先进的开源IPA转写系统（WhisperIPA和ZIPA），并提出了能够容忍语言学上相似音素替换的Soft PER评估指标。 |
| [^139] | [Data Journalist Agent: Transforming Data into Verifiable Multimodal Stories](https://arxiv.org/abs/2606.11176) | 提出了 Data2Story 多智能体框架，将专业化角色编排为虚拟新闻编辑室，实现证据可追溯、按需生成多模态内容的端到端自动化数据新闻写作。 |
| [^140] | [The Neutral Mask: How Alignment Training Provides Shallow Alignment while Leaving Partisan Structure Intact in a Large Language Model](https://arxiv.org/abs/2606.09735) | 该研究通过对Llama 3.1 8B对齐前后内部表征的机制性分析，发现对齐训练并未消除模型内部结构化的党派政治倾向，而是通过压缩党派信号的方差使输出表面上保持中立平衡，仅实现了浅层对齐。 |
| [^141] | [Measurement Under Selection: Decoy-Calibrated Failure Audits for Language Models](https://arxiv.org/abs/2606.09046) | 本文提出Janus框架，通过随机打乱属性标签生成的“诱饵”来校准阈值，只有超出偶然波动水平并在保留数据上稳健成立的错误模式才被报告，从而避免语言模型失效审计中出现虚假的模式发现。 |
| [^142] | [LaSR: Context-Aware Speech Recognition via Latent Reasoning](https://arxiv.org/abs/2606.00507) | LaSR提出了一种利用潜在推理的上下文感知语音识别新训练范式，通过在目标词声学特征区域周围对齐思维链监督并引入潜在推理阶段实现上下文信息落地，同时发布了聚焦学术术语的大规模语料库Spoken Darwin-Science。 |
| [^143] | [By Their Fruits You Will Know Them: Comparing Formalizations of Law by the Decisions They Encode](https://arxiv.org/abs/2605.25186) | 提出一种基于SAT求解器的方法，通过枚举同一法律条文的不同形式化在具体边界案例上产生分歧的行为来系统比较它们，从而揭示大语言模型生成的法律形式化中难以预料的隐含解释性选择。 |
| [^144] | [How Loud Rumbles Hit Newsstands: A Data Analysis of Coverage and Spatial Bias in German News about Landslides Around the World](https://arxiv.org/abs/2605.18105) | 本文通过分析25年间近5.5万篇关于4500起山体滑坡事件的德国新闻报道，揭示了德国媒体报道存在空间偏差，例如对南欧和西欧地区的灾害事件存在过度报道。 |
| [^145] | [PersonalAI 2.0: Enhancing knowledge graph traversal/retrieval with planning mechanism for Personalized LLM Agents](https://arxiv.org/abs/2605.13481) | PersonalAI 2.0通过引入动态多阶段查询处理流水线的规划机制，实现了由实体、图顶点和线索查询引导的自适应迭代式知识图谱检索，在多个问答基准上显著提升了生成答案的事实准确性。 |
| [^146] | [From Procedural Skills to Strategy Genes: Towards Experience-Driven Test-Time Evolution](https://arxiv.org/abs/2604.15097) | 本研究通过45个场景、4590次受控试验发现，紧凑的“策略基因”表示比面向文档的技能包更适合作为可复用经验的载体，在测试时控制与迭代演化中均表现更优，证明经验的表示方式本身是决定性因素。 |
| [^147] | [PolyJarvis: An LLM-Orchestrated Agent for Automated All-Atom Molecular Dynamics of Amorphous Homopolymers](https://arxiv.org/abs/2604.02537) | PolyJarvis是一个由大语言模型智能体编排的自动化平台，通过规划智能体生成经验证的运行计划并调用EMC和LAMMPS等工具包执行，实现了从重复单元SMILES到无定形均聚物全原子分子动力学模拟及性质计算的端到端自动化。 |
| [^148] | [Are Finer Citations Always Better? Rethinking Granularity for Attributed Generation](https://arxiv.org/abs/2604.01432) | 该论文通过分析四种模型规模发现，细粒度句子级引用并非总是最优，段落级的中间粒度归因质量最佳，选择最优引用粒度可在几乎不牺牲答案正确性的情况下大幅提升模型性能与归因质量。 |
| [^149] | [CounselReflect: Opportunities and Challenges for Designing Tools to Support Self-Reflection on Mental Health and Well-Being Conversations with AI](https://arxiv.org/abs/2603.29429) | 该论文提出了CounselReflect工具，将基于文献的心理咨询质量指标转化为面向用户的反思框架，并通过对21位AI心理健康支持用户的访谈，揭示了工具辅助自我反思的机遇与挑战（如用户存在确认偏误），主张反思工具应帮助用户发现盲点并进行更全面的审视。 |
| [^150] | [When Perplexity Lies: Generation-Focused Distillation of Hybrid Sequence Models](https://arxiv.org/abs/2603.26556) | 该论文揭示了对数似然评估方式会掩盖蒸馏模型在真实自回归生成上的严重质量退化（7B蒸馏模型在对数似然评分下仅落后教师0.2个百分点，但自回归生成时落后20.8个百分点），并提出了面向生成的多阶段蒸馏流水线GenDistill来蒸馏混合序列模型。 |
| [^151] | [Automated Gradient-Driven Parameter Sharing for Low-Resource Multilingual Speech-to-Text Translation](https://arxiv.org/abs/2603.25836) | 该论文提出一种利用训练梯度信息自动确定层级别参数共享模式的方法，通过语言聚类、任务差异度量和子空间对齐三种分析策略，在低资源多语言语音翻译任务中持续提升翻译质量。 |
| [^152] | [When Consistency Becomes Bias: Interviewer Effects in Semi-Structured Clinical Interviews](https://arxiv.org/abs/2603.24651) | 该研究发现在半结构化临床访谈的抑郁检测任务中，模型会利用访谈者固定的提示词这一脚本痕迹来获得虚高的分类性能，而将模型限制于仅使用参与者的真实话语才能反映真正的语言线索。 |
| [^153] | [MAPLE: Metadata Augmented Private Language Evolution](https://arxiv.org/abs/2603.19258) | MAPLE通过引入元数据增强，解决了私有演化（PE）方法在私有数据分布偏离基础模型预训练先验时的初始化瓶颈问题，实现了更高效的基于API的差分隐私合成数据生成。 |
| [^154] | [Social Simulacra in the Wild: AI Agent Communities on Moltbook](https://arxiv.org/abs/2603.16128) | 首次对AI智能体社区与人类在线社区进行大规模实证比较，发现Moltbook上的AI社区存在极端参与不平等、作者高度重叠，且AI生成内容情感平淡、倾向断言而非探索，其社区同质化现象主要由共享作者身份的结构性因素导致。 |
| [^155] | [TTSR: Test-Time Self-Evolving via Reflection](https://arxiv.org/abs/2603.03297) | TTSR通过让单个模型交替扮演学生和教师角色，基于反思后合成的范式，在测试时针对失败轨迹生成变体问题，从而克服了缺乏可学习样本和探索效率低下的瓶颈。 |
| [^156] | [Kinship Data Benchmark for Multi-hop Reasoning](https://arxiv.org/abs/2601.07794) | 该论文提出了KinshipQA基准，其核心创新是一个可按需生成大规模、真实且具有文化特异性的家谱数据的生成式流水线，从而系统评估大型语言模型在亲属关系多跳推理上的能力。 |
| [^157] | [Communication and Verification in LLM Agents towards Collaboration under Information Asymmetry](https://arxiv.org/abs/2510.25595) | 本文将经典的爱因斯坦谜题扩展为桌面游戏，研究信息不对称条件下两个LLM智能体通过推理、沟通与行动实现协作，并提出“微调加验证器”框架，利用沟通策略和环境验证信号显著提升协作完成任务的能力。 |
| [^158] | [TripScore: Aligning LLMs for Real-World Travel Planning via Expert-Calibrated Reward](https://arxiv.org/abs/2510.09011) | TripScore 是基于真实用户日志与 203 位旅行专家校准构建的旅行规划评估基准，研究发现强化学习微调（如 GRPO）在现实旅行规划任务中比其他方法带来更稳定一致的提升。 |
| [^159] | [oMeBench: Towards Robust Benchmarking of LLMs in Organic Mechanism Elucidation and Reasoning](https://arxiv.org/abs/2510.07731) | 该论文提出了首个大规模专家标注的有机机理推理基准oMeBench（含超过10,000个注释机理步骤）以及oMeS动态评分框架，用以严格评估大语言模型真正的化学推理能力。 |
| [^160] | [Compass-v3: Scaling Domain-Specific LLMs for Multilingual E-Commerce in Southeast Asia](https://arxiv.org/abs/2509.09121) | 该论文提出了面向东南亚电商的245B参数垂直领域混合专家大模型Compass-v3，通过更大专家设计、硬件级效率优化和最优传输直接偏好优化（OTWO）方法，显著提升了多语言电商场景下的领域性能与指令遵循能力。 |
| [^161] | [LMEnt: A Suite for Analyzing Knowledge in Language Models from Pretraining Data to Representations](https://arxiv.org/abs/2509.03405) | LMEnt是一个用于分析语言模型知识获取的工具套件，包含实体标注的预训练语料库、性能提升高达80.4%的实体检索方法，以及12个带4K检查点的预训练模型，为研究预训练数据与知识表示之间的联系提供了受控环境。 |
| [^162] | [GeLaCo: An Evolutionary Approach to Layer Compression](https://arxiv.org/abs/2507.10059) | GeLaCo提出了一种基于进化搜索和参数化权重合并的新型层折叠方法，通过基于残差更新相似性和语言建模KL散度的适应度函数，高效地探索大语言模型的压缩解空间。 |
| [^163] | [Redemption Score: A Multi-Modal Evaluation Framework for Image Captioning via Distributional, Perceptual, and Linguistic Signal Triangulation](https://arxiv.org/abs/2505.16180) | 提出了Redemption Score（RS）评估框架，通过融合互信息散度、DINO感知相似度和LLM文本嵌入三种互补信号对图像描述进行多模态评估，在Flickr8k基准上取得58.42的Kendall-tau，优于大多数先前方法。 |
| [^164] | [A Large-Scale Vision-Language Dataset Derived from Open Scientific Literature to Advance Biomedical Generalist AI](https://arxiv.org/abs/2503.22727) | 该论文发布了源自PubMed Central开放获取文献的开源大规模多模态数据集Biomedica（含600万篇文章、2400万图像-文本对及专家标注），基于其训练的AI模型在嵌入、对话和检索等各任务类别中均超越了此前的开放系统。 |

# 详细

[^1]: 具有障碍物感知框架的安全机器人操作编码智能体

    Coding Agents with an Obstacle-Aware Harness for Safe Robot Manipulation

    [https://arxiv.org/abs/2609.20822](https://arxiv.org/abs/2609.20822)

    本文发现编码智能体在机器人操作中因规划环节未能将安全约束设为优先事项而系统性碰撞障碍物（而非感知或指令问题），并通过将操作分解为路径阶段和接触时刻来定位失败根源，提出了障碍物感知框架以实现安全操作。

    

    编码智能体已成为机器人操作领域一种有前景的范式：语言模型将机器人控制器编写为程序，以这种方式构建的智能体现在无需机器人特定训练即可操作机器人。然而，这种范式是否安全，这一问题尚未被探讨。我们在安全约束下评估编码智能体，其中每个任务将操作目标与机器人不得触碰的障碍物配对。智能体追求目标，但在大多数情况下与障碍物发生碰撞，将任务完成视为唯一目标而忽视安全性。智能体在其推理轨迹中确实对障碍物进行了推理，且提示词已经禁止触碰障碍物，因此感知和指令都没有问题；问题出在规划环节，所陈述的约束从未成为优先事项。通过将操作分解为路径阶段和富含接触的时刻，我们定位了失败的根源。在路径阶段，模型无法对……（摘要原文在此处截断）

    arXiv:2609.20822v1 Announce Type: cross  Abstract: Coding agents have emerged as a promising paradigm for robot manipulation: a language model writes the robot controller as a program, and agents built in this way now operate robots without robot-specific training.Whether this paradigm is also safe, however, has not been asked. We evaluate coding agent under a safety constraint, where each task pairs a manipulation goal with an obstacle the robot must not touch. The agent pursues the goal but collides with the obstacle in most cases, treating task completion as its sole objective while neglecting safety. The agent reasons about the obstacle in its traces, and the prompt already forbids touching it, so neither perception nor instruction is at fault; the fault lies in the planning, where the stated constraint never becomes a priority. By decomposing manipulation into a route phase and a contact-rich moment, we locate the source of the failure. Along the route, the model cannot prioritize
    
[^2]: 嵌入模型以奇特的方式进行测量

    Embedding Models Measure in Peculiar Ways

    [https://arxiv.org/abs/2609.20821](https://arxiv.org/abs/2609.20821)

    该研究发现嵌入模型对质量、距离、时间和体积等物理测量的表示十分微弱且奇特，主要受表面字符串相似性的强烈影响，而重新校准相似度也无法显著改善其与真实物理测量的对齐。

    

    嵌入空间定义了语义相似性和距离的概念。我们研究了这些嵌入是否反映了质量、距离、时间和体积等物理测量，这些物理测量具有唯一且客观的语义等价与距离概念。我们发现物理测量在嵌入空间中仅被微弱地建模，相反，可以观察到相当奇特的测量模式。进一步的分析表明，物理测量的嵌入表示受到表面字符串相似性的强烈影响，而重新校准相似性并不能实质性改善对齐效果。

    arXiv:2609.20821v1 Announce Type: new  Abstract: Embedding spaces define notions of semantic similarity and distance. We study whether those embeddings reflect physical measurements of mass, distance, time and volume, which admit a unique, objective notion of semantic equivalence and distance. We find that physical measurement is only weakly modeled in the embedding space, and that instead quite peculiar measurement patterns can be observed. Further analysis indicates that embedding representations of physical measurements are strongly influenced by superficial string similarity, and recalibration of similarity does not substantially improve the alignment.
    
[^3]: 统一在线话语中群体间敌意的模型

    Unifying Models of Intergroup Hostility in Online Discourse

    [https://arxiv.org/abs/2609.20808](https://arxiv.org/abs/2609.20808)

    本研究利用2024年美国大选期间TikTok、Truth Social和Twitter/X平台的286万条帖子，首次对六种群体间敌意基础理论进行统一建模，揭示了敌意修辞机制在真实在线话语中的表现形式及相互关系。

    

    针对社会群体的敌意言论会使排斥行为正常化，并为不当对待提供正当理由，同时还会助长日益加剧的两极分化和政治暴力。针对在线言论中敌意言论的治理工作借鉴了社会心理学、道德心理学和政治学中的基础理论。然而，这些理论大多是并行发展起来的，它们对敌意如何形成往往提出不同甚至相互矛盾的解释，并且很少在真实话语中得到相互比较与检验。其结果是对敌意修辞机制的理解碎片化，无法清晰地认识这些机制在现实世界话语中如何出现以及如何相互关联。我们利用2024年美国总统大选期间来自TikTok、Truth Social和Twitter/X的286万条帖子，对六种群体间敌意基础理论的机制进行了建模——边界构建、威胁构建、替罪羊化、负面评价……

    arXiv:2609.20808v1 Announce Type: new  Abstract: Hostile rhetoric toward social groups can normalize exclusion and justify mistreatment, as well as contribute to rising polarization and political violence. Efforts to moderate hostile rhetoric in online speech draw on foundational theories in social and moral psychology, and political science. However, these theories were developed largely in parallel, often propose different and sometimes conflicting accounts of how hostility develops, and have rarely been tested against each other in real discourse. The result is a fragmented understanding of the rhetorical mechanisms of hostility, without a clear sense of how they appear, and relate to each other, in real-world discourse. Using 2.86 million posts from TikTok, Truth Social, and Twitter/X during the 2024 U.S. presidential election, we model the mechanisms of six foundational theories of intergroup hostility -- boundary construction, threat construction, scapegoating, negative evaluatio
    
[^4]: 编码智能体框架设计的实证研究

    An Empirical Study of Harness Design for Coding Agents

    [https://arxiv.org/abs/2609.20804](https://arxiv.org/abs/2609.20804)

    该论文通过固定执行循环并系统变化规划、动作空间和上下文管理三个组件的实证研究，发现上下文管理在上下文窗口预算紧张时价值显著提升，且其主要收益来自防止上下文溢出故障。

    

    编码框架决定了自主编码智能体如何将模型能力转化为长周期的软件工程性能，然而现有工作通常将框架作为整体系统进行评估，导致各个组件的有效性尚不清楚。为了实现组件级别的比较，我们使用一个轻量级编码框架来研究这一问题，该框架的执行循环保持固定，而三个组件则进行变化：规划、动作空间和上下文管理。我们在SWE-Bench Verified和Terminal-Bench 2.1上对四个模型进行了评估，共评估了176个匹配设置，涵盖五种上下文管理策略、四种上下文窗口预算，以及针对规划和动作空间的定向消融实验。我们发现：（1）随着上下文窗口预算的收紧，上下文管理变得愈发重要，其大部分收益来自于防止上下文溢出故障。（2）在基于LLM的摘要之前分阶段进行基于规则的省略（摘要在此处截断）。

    arXiv:2609.20804v1 Announce Type: new  Abstract: Coding harnesses shape how autonomous coding agents translate model capabilities into long-horizon software-engineering performance, yet existing work typically evaluates harnesses as monolithic systems, leaving the effectiveness of individual components unclear. To enable component-level comparisons, we study this question with a lightweight coding harness whose execution loop is fixed while three components are varied: planning, action space, and context management. Across four models evaluated on SWE-Bench Verified and Terminal-Bench 2.1, we evaluate 176 matched settings spanning five context-management strategies, four context-window budgets, and targeted ablations of planning and action space. We find that: (1) Context management becomes increasingly valuable as the context-window budget tightens, with most of its benefit coming from preventing context-overflow failures. (2) Staging rule-based elision before LLM-based summarization 
    
[^5]: JEPA-Anything：跨不同世界学习预测模型

    JEPA-Anything: Learning Predictive Models across Different Worlds

    [https://arxiv.org/abs/2609.20800](https://arxiv.org/abs/2609.20800)

    提出基于正交预测因子分解（OPF）的领域无关框架JEPA-Anything，以统一的学习原则在视觉、生物学、临床轨迹、控制、分子动力学、物理场和天气七个迥异领域中实现世界建模。

    

    世界建模使智能体能够预判后果、指导干预并从交互中学习。然而，现有预测模型仍然局限于特定领域：一个统一的学习原则能否支持跨越截然不同系统的世界建模？我们提出了JEPA-Anything，这是一个基于正交预测因子分解（OPF）的领域无关框架。该框架扩展了联合嵌入预测架构，OPF将潜在目标分解为互补因子，通过专用通路分别学习这些因子，并在共享的预测设计中对其进行重组。我们在七个领域对JEPA-Anything进行了评估：视觉、生物学、临床轨迹、控制、分子动力学、物理场和天气。实验涵盖表示学习、干预预测、分布外泛化和长期动态预测，包括10个匹配的动力学任务、超过1,000个临床事件的预测以及100步的分子动力学模拟推演。

    arXiv:2609.20800v1 Announce Type: new  Abstract: World modeling enables intelligence to anticipate consequences, guide interventions, and learn from interaction. Yet predictive models remain domain-specific: can a common learning principle support world modeling across radically different systems? We introduce JEPA-Anything, a domain-agnostic framework based on orthogonal predictive factorization (OPF). Extending joint-embedding predictive architectures, OPF decomposes latent targets into complementary factors, learns them through dedicated pathways, and recombines them within a shared predictive design. We evaluate JEPA-Anything across seven domains: vision, biology, clinical trajectories, control, molecular dynamics, physical fields, and weather. Experiments span representation learning, intervention prediction, out-of-distribution generalization, and long-horizon dynamics, including 10 matched dynamics tasks, forecasting of over 1,000 clinical events, and 100-step molecular rollouts
    
[^6]: RetireOPD：面向智能体强化学习的自我退休在线策略蒸馏

    RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2609.20784](https://arxiv.org/abs/2609.20784)

    RetireOPD提出了一种自我退休的在线策略蒸馏方法，通过自适应退休机制让强化学习智能体在教师监督收益不再增长时自动辞退教师，从而更高效地内化特权任务技能。

    

    使用强化学习（RL）训练的多轮智能体每条轨迹只能获得单一的标量奖励，这促使人们采用自我在线策略蒸馏（OPD），由一个具备特权任务技能的自教师提供密集的token级监督，让不具备技能的学生将其内化。然而，智能体任务中的两个发现削弱了这一方法的有效性：仅凭特权信息并不总能使教师变得可靠，且教师监督的收益具有阶段性依赖。因此，我们提出了RetireOPD（自我退休在线策略蒸馏），该方法首先利用环境奖励优化一个解耦的、以技能为条件的教师，然后联合RL和OPD训练一个无技能的学生。RetireOPD不遵循预定义的蒸馏时间表，而是采用自适应退休机制：一旦学生与教师之间的差距停止缩小，且学生达到教师成功率的目标比例，学生便自行“辞退”教师。

    arXiv:2609.20784v1 Announce Type: cross  Abstract: Multi-turn agents trained with reinforcement learning (RL) receive a single scalar reward per trajectory, which motivates self on-policy distillation (OPD) to supply dense token-level supervision from a self-teacher with privileged task skills, letting a skill-free student internalize them. This recipe, however, is undermined by two findings in agentic tasks: privileged information alone does not always make a teacher reliable, and the benefit of teacher supervision is stage-dependent. We therefore propose RetireOPD (Self-Retiring On-Policy Distillation), which first optimizes a decoupled, skill-conditioned teacher with environment rewards and then trains a skill-free student jointly with RL and OPD. Rather than following a predefined distillation schedule, RetireOPD adopts Adaptive Retirement: the student drops the teacher on its own once their discrepancy stops shrinking and it reaches a target fraction of the teacher's success rate,
    
[^7]: GPT模型中的危害洗白：证据表明性别歧视在安全训练的各代模型中被转化而非减少

    Harm Laundering in GPT Models: Evidence That Gender Discrimination Is Transformed Rather Than Reduced Across Safety-Trained Generations

    [https://arxiv.org/abs/2609.20779](https://arxiv.org/abs/2609.20779)

    该论文提出“危害洗白”这一新概念，通过对GPT-2至GPT-5共15个模型的45万条性别导向文本分析，证明安全训练并未真正消除性别歧视，而是将其从露骨的性暴力内容转化为更隐蔽的形式（如将乳腺癌话题建构为男性权利辩论），从而揭示现有基于表层分类器的安全评估方法的系统性缺陷。

    

    arXiv:2609.20779v1 公告类型：交叉（cross）摘要：大型语言模型的安全评估依赖于表层形式分类器，这些分类器报告模型各代中危害评分呈下降趋势。我们提供的证据表明，这种方法论存在系统性缺陷：露骨的歧视性内容被转化而非被移除。我们将这一现象称为“危害洗白”。通过分析来自15个模型（涵盖从GPT-2到GPT-5的OpenAI GPT谱系，涉及三种人口统计条件）的450,000个性别导向文本补全，我们表明GPT-2中针对女性的输出中普遍存在的性暴力内容聚类到GPT-4时已经消失，而针对男性的补全却获得了正向表征空间（照护角色、情感范围、盟友身份），这是针对女性的补全所没有的。这一模式在GPT-5中最为明显：主题5（1,997个文档）将乳腺癌建构为男性权利辩论的话题，而在针对女性的输出中没有出现任何等价的聚类。三个独立的分类器将这些内容评为非……

    arXiv:2609.20779v1 Announce Type: cross  Abstract: Safety evaluations for large language models rely on surface-form classifiers that report declining harm scores across model generations. We provide evidence that this methodology is systematically incomplete: explicit discriminatory content is transformed rather than removed. We call this \emph{harm laundering}. Analysing 450,000 gender-directed completions across 15 models spanning GPT-2 through to GPT-5 (OpenAI GPT lineage; three demographic conditions), we show that sexual violence clusters prevalent in GPT-2 women-directed output disappear by GPT-4, while men-directed completions gain positive representational territory (caregiving, emotional range, ally identity) that women-directed completions do not. The pattern is most visible at GPT-5: Topic~5 (1,997~documents) frames breast cancer as a men's rights debate, while zero equivalent clusters appear in women-directed output. Three independent classifiers score this content as non-
    
[^8]: dQwen3.5：混合注意力扩散语言模型

    dQwen3.5: Hybrid-Attention Diffusion Language Models

    [https://arxiv.org/abs/2609.20751](https://arxiv.org/abs/2609.20751)

    本论文将Qwen3.5的注意力-RNN混合架构（0.8B至9B规模）适配为dQwen3.5扩散语言模型系列，证明混合架构骨干仅需全注意力模型约一半的训练词元即可达到相同损失，同时在任意顺序解码上表现相当，并在并行解码下表现出色。

    

    将预训练的自回归（AR）模型进行适配是构建扩散语言模型（DLM）的一条高性价比路径。尽管几乎所有此类适配都以全注意力Transformer为起点，但自回归建模已转向注意力层与RNN层交替堆叠的混合架构。这为适配带来了障碍：与注意力机制不同，RNN在结构上是因果的，要将其双向化并非易事。尽管存在这种不匹配，我们仍然研究了此类骨干网络能否成为有效的扩散语言模型——我们在0.8B、2B、4B和9B规模上对Qwen3.5进行了适配，由此得到了dQwen3.5系列模型。我们发现混合骨干网络可以成为高效的适配起点：与全注意力对照组相比，混合模型仅需约一半的词元量即可达到相同的训练损失。在各个规模上，dQwen3.5在任意顺序解码行为上与全注意力扩散语言模型相似，并在并行解码下表现强劲。

    arXiv:2609.20751v1 Announce Type: new  Abstract: Adapting a pretrained autoregressive (AR) model is a cost-efficient route to a diffusion language model (DLM). While nearly all such adaptations start from a full-attention transformer, AR modeling has shifted toward hybrid architectures that interleave attention and RNN layers. This creates an obstacle for adaptation: unlike attention, RNNs are structurally causal and nontrivial to bidirectionalize. Despite this mismatch, we investigate whether such backbones can become effective DLMs by adapting Qwen3.5 at 0.8B, 2B, 4B, and 9B scales, yielding the dQwen3.5 family. We find that hybrid backbones can be efficient starting points for adaptation: against a full-attention control, the hybrid reaches a given training loss in about half the tokens. Across scales, dQwen3.5 resembles full-attention DLMs in any-order decoding behavior and performs strongly under parallel decoding.
    
[^9]: 按需注意力：语言模型知道何时进行回忆

    On-Demand Attention: Language Models Know When to Recall

    [https://arxiv.org/abs/2609.20734](https://arxiv.org/abs/2609.20734)

    提出按需注意力（ODA）解码方法，利用轻量级回忆头根据预测收益动态选择性地调用全局注意力，在不改变预训练权重的前提下实现长上下文推理的实际解码加速。

    

    推理和智能体工作负载日益需要高效的长上下文推理。然而，全注意力解码在每一步都会读取不断增长的历史信息，而不考虑这些信息对下一次预测是否真正有益。我们证明，预训练模型的解码状态在进行全局读取之前就已经包含了能够预测这种收益的信息。基于这一发现，我们提出了按需注意力（ODA），这是一种局部优先的解码方法，它使用一个轻量级的回忆头，在生成过程中根据预测收益的变化选择性地调用全局注意力。ODA仅训练回忆头，保持预训练权重不变，并保留完整的历史KV缓存以供未来回忆。我们进一步在vLLM中实现了GPU端的条件执行，将全局读取的减少转化为长上下文长度下相对于全注意力机制的实际解码加速。在Qwen和Gemma模型（包括混合注意力骨干网络）上的实验表明……

    arXiv:2609.20734v1 Announce Type: new  Abstract: Reasoning and agentic workloads increasingly demand efficient long-context inference. Yet full-attention decoding reads the growing history at every step, regardless of its benefit to the next prediction. We show that a pretrained model's decoding states already contain information predictive of this benefit, before the global read. Building on this finding, we introduce On-Demand Attention (ODA), a local-first decoding method that uses a lightweight recall head to selectively invoke global attention as its predicted benefit changes during generation. ODA trains only the recall head, leaving pretrained weights unchanged and the complete historical KV cache available for future recall. We further implement GPU-side conditional execution in vLLM, translating reduced global reads into practical decoding speedups over full attention at long context lengths. Experiments across Qwen and Gemma models, including hybrid-attention backbones, show 
    
[^10]: 不要屏蔽环境：观测监督改变智能体在强化学习下的探索方式

    Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL

    [https://arxiv.org/abs/2609.20715](https://arxiv.org/abs/2609.20715)

    该论文提出ActObs方法，在监督微调中同时对轨迹中已有的环境观测标记进行监督，使策略学会建模动作后果，从而在不增加任何数据、参数或计算成本的情况下，显著提升后续GRPO强化学习中智能体的探索能力和pass@k性能。

    

    智能体的轨迹记录了智能体做了什么以及接下来发生了什么。然而，标准的监督微调（SFT）只对智能体生成的动作标记应用损失，仅将环境观测作为上下文而不作为预测目标。我们探究这一惯例是否能为后续强化学习提供最佳初始化。我们提出ActObs，它还对每条轨迹中已有的观测标记进行监督。尽管部署的智能体从不生成观测，但学习预测观测可以在不增加数据、参数、序列标记或前向传播的情况下，促使策略对动作后果进行建模。这些方法在SFT后表现相似，但在GRPO后出现分化。在Qwen3-4B上，基于ActObs的GRPO在Terminal-Bench 2.0上每个评估采样预算下都比仅监督动作的方法获得更高的pass@k。在Qwen3-8B上，它以一定的pass@1可靠性换取更高的pass@k（pass@16时提升3.4个百分点），并解决了更多任务。

    arXiv:2609.20715v1 Announce Type: cross  Abstract: Agent trajectories record what an agent does and what happens next. Yet standard supervised fine-tuning (SFT) applies loss only to agent-authored action tokens, using environment observations as context but not as prediction targets. We ask whether this convention provides the best initialization for subsequent reinforcement learning. We introduce ActObs, which also supervises the observation tokens already present in each trajectory. Although deployed agents never generate observations, learning to predict them encourages the policy to model action consequences without adding data, parameters, sequence tokens, or forward passes. The methods perform similarly after SFT but diverge after GRPO. On Qwen3-4B, GRPO from ActObs achieves higher pass@k at every evaluated sampling budget than its action-only counterpart on Terminal-Bench 2.0. On Qwen3-8B, it trades some pass@1 reliability for higher pass@k (+3.4 pp at pass@16) and solves more d
    
[^11]: 摘要化偏差：大型语言模型中客观投影向陈述模式标签的方向性坍缩——概念框架与预注册测试协议

    Summarization Bias: The Directional Collapse of Objective Projection into Told-Mode Labels in Large Language Models --- A Conceptual Framework and Registered Test Protocol

    [https://arxiv.org/abs/2609.20712](https://arxiv.org/abs/2609.20712)

    本文提出“摘要化偏差”这一新概念并设计预注册测试协议，用以验证大型语言模型在叙事生成中会系统性地将本应以“展示”方式呈现的情感内容坍缩为直接“陈述”式的摘要标签。

    

    本文引入并操作化了“摘要化偏差”这一概念：即大型语言模型（LLM）倾向于将叙事意义表征为抽象的摘要标签，而非产生该意义的可重构推理结构的一种系统性倾向。在布卢特学说框架内，叙事效果沿“陈述-展示”轴进行理论化：在陈述模式下，情感和信息内容被明确宣示，几乎不需要读者进行重构；在展示模式下，这些内容在表面层次被抑制，必须从物理线索和间接表达（客观投影）中重构出来。展示模式是该学说旨在测量的高负荷条件。论文的主张是：LLM在这一轴上会以特定方向失效。摘要化偏差被假设在两种机制下运作：其一为生成机制，即被要求通过客观投影呈现某种情感的模型会默认改为直接宣告该情感；其二为……（摘要在此处截断）

    arXiv:2609.20712v1 Announce Type: new  Abstract: This paper introduces and operationalizes summarization bias: a proposed systematic tendency of large language models (LLMs) to represent narrative meaning as an abstract summary label rather than as the reconstructable inferential structure that produces it. Within the Bulut Doctrine, narrative effect is theorized along a told-shown axis: in told mode, emotional and informational content is declared explicitly and requires little reader reconstruction; in shown mode, that content is suppressed at the surface and must be reconstructed from physical cues and indirection (Objective Projection). Shown mode is the higher-load condition the doctrine is designed to measure.   The claim is that LLMs fail along this axis in a specific direction. Summarization bias is hypothesized to operate in two regimes: (i) a generative regime, in which a model asked to render an emotion through Objective Projection defaults to declaring it instead; and (ii) 
    
[^12]: HerHealthEval：评估多语言与语域敏感的女性健康交流理解能力

    HerHealthEval: Evaluating Multilingual and Register-Sensitive Understanding of Women's Health Communication

    [https://arxiv.org/abs/2609.20684](https://arxiv.org/abs/2609.20684)

    该论文提出HerHealthEval框架，通过英语、法语、阿拉伯语及六种不同表达形式的女性健康临床病例，系统评估大语言模型是否真正正确理解用户关切，尤其是其识别信息不足并主动请求澄清的能力。

    

    大语言模型在医疗健康交流中的应用日益增多，然而大多数评估侧重于回答质量，同时默认用户的关切已被正确理解。我们提出了HerHealthEval，这是一个用于女性健康交流多语言理解的受控评估框架。针对每个临床病例，HerHealthEval提供了英语、法语和现代标准阿拉伯语的匹配版本，并采用六种交流形式：规范形式、临床形式、外行形式、间接或委婉形式、情感担忧形式以及刻意不充分说明形式。前五种形式表达相同的潜在关切并保留相同的临床信息，而不充分说明形式则有意省略相关细节，以测试模型是否能识别出需要进一步澄清。我们在关切分类、风险校准、澄清行为、解析合规性等维度上评估了一个多语言指令模型及其QLoRA适配变体。

    arXiv:2609.20684v1 Announce Type: new  Abstract: Large language models are increasingly used in healthcare communication, yet most evaluations emphasize response quality while assuming that the user's concern has been interpreted correctly. We introduce HerHealthEval, a controlled evaluation framework for multilingual understanding of women's-health communication. For each clinical case, HerHealthEval provides matched versions in English, French, and Modern Standard Arabic using six communicative forms: canonical, clinical, layperson, indirect or hedged, emotionally concerned, and deliberately under-specified. The first five express the same underlying concern and retain the same clinical information, whereas the under-specified form intentionally omits relevant details to test whether the model recognizes that clarification is needed. We evaluate a multilingual instruction model and QLoRA-adapted variants on concern classification, risk calibration, clarification behavior, parse compl
    
[^13]: PAA：概率Allen代数：Allen区间关系的一种生成式且完整的概率扩展

    PAA: The Probabilistic Allen Algebra: A Generative and Complete Probabilistic Extension of Allen's Interval Relations

    [https://arxiv.org/abs/2609.20634](https://arxiv.org/abs/2609.20634)

    本文提出概率Allen代数（PAA），一种生成式且完整的概率扩展，通过从区间边界的概率分布中推导关系概率，解决了经典Allen区间代数无法处理时间信息不确定性及程度化时间表达的问题。

    

    Allen区间代数是一种用于时间关系的定性演算，但其十三个基本关系是关于精确区间边界的清晰谓词。这对于来自语言、感知、数据库或不确定历史记录的时间信息而言是不够的，因为在这些场景中，时间、持续时间和边界都是不确定的，且诸如“就在……之前”或“大致在……期间”这类表达具有程度化的含义。我们提出了概率Allen代数（PAA）：一种生成式且完整的扩展，其中关系概率是从区间边界上的分布推导出来的，而不是作为分数被直接赋值。时间点服从高斯分布；区间具有高斯分布的中点和截断高斯分布的持续时间。每种关系都是同一公共概率空间中的一个边界排序谓词：点-点关系可简化为误差函数，点-区间和区间-区间关系则可简化为由线性不等式诱导的多元高斯象限概率……

    arXiv:2609.20634v1 Announce Type: new  Abstract: Allen's interval algebra is a qualitative calculus for temporal relations, but its thirteen base relations are crisp predicates over exact interval boundaries. This is inadequate for temporal information from language, perception, databases, or uncertain histories, where times, durations, and boundaries are uncertain and expressions such as "just before" or "roughly during" have graded meaning. We develop the probabilistic Allen algebra (PAA): a generative and complete extension in which relation probabilities are derived from distributions over interval boundaries rather than assigned as scores. Time points are Gaussian; intervals have Gaussian midpoints and truncated-Gaussian durations. Every relation is a boundary-ordering predicate in one common probability space: point-point relations reduce to error functions, and point-interval and interval-interval relations to multivariate Gaussian orthant probabilities induced by linear inequal
    
[^14]: UniPolicy：面向生成式搜索广告的统一目标特定策略

    UniPolicy: Unified Objective-Specific Policies for Generative Search Advertising

    [https://arxiv.org/abs/2609.20630](https://arxiv.org/abs/2609.20630)

    UniPolicy提出了一种目标感知的多策略对齐框架，通过目标特定前缀标记、稀疏MoE-LoRA路由和残差FFN在共享骨干网络中分层解耦参数，使生成式搜索广告能够联合优化相关性、点击倾向和商业价值等异构目标，避免梯度竞争导致的全局次优问题。

    

    搜索广告将用户意图与商业内容连接起来，在平台变现中发挥着关键作用。近期的系统通常将预训练生成模型与单一业务奖励（如eCPM）对齐，或使用朴素的奖励融合进行初步的多目标对齐。然而，理想的搜索广告系统必须联合考虑异构的多个目标，包括相关性、点击倾向和商业价值，以在平衡用户体验与商业价值的同时，缓解由梯度竞争导致的全局次优性能。我们提出了UniPolicy，一个目标感知的多策略对齐框架。UniPolicy结合了目标特定的前缀标记、稀疏MoE-LoRA路由和目标特定的残差FFN，在共享骨干网络内分层解耦参数，为不同的业务目标提供差异化的参数空间和策略表达空间。它进一步构建了……（摘要在此处截断）

    arXiv:2609.20630v1 Announce Type: new  Abstract: Search advertising connects user intent with commercial content and plays a critical role in platform monetization. Recent systems typically align pretrained generative models with a single business reward, such as eCPM, or use naive reward fusion for preliminary multi-objective alignment. However, an ideal search advertising system must jointly account for heterogeneous objectives, including relevance, click propensity, and commercial value, to balance user experience and business value while mitigating globally suboptimal performance caused by gradient competition. We propose UniPolicy, an objective-aware multi-policy alignment framework. UniPolicy combines objective-specific prefix tokens, sparse MoE-LoRA routing, and objective-specific residual FFNs to hierarchically decouple parameters within a shared backbone, providing differentiated parameter and policy-expression spaces for different business objectives. It further constructs pa
    
[^15]: Chronicle：用于大语言模型智能体回归测试的切点重放

    Chronicle: Cut-Point Replay for Regression Testing of LLM Agents

    [https://arxiv.org/abs/2609.20625](https://arxiv.org/abs/2609.20625)

    Chronicle通过在非确定性边界记录LLM智能体的运行轨迹并提出切点重放机制，将记录的故障事件转化为可在持续集成中运行的回归测试，解决了LLM智能体故障难以重现的问题。

    

    大语言模型的响应是非确定性的，因此LLM智能体中的故障难以重现：故障依赖于无法按位重现的推理、依赖于读取不断变化状态的工具，以及重跑时很少重复的多步执行轨迹。记录与重放技术可以使一次运行变得可重现，但现有的智能体工具记录运行只是为了追踪或评分，而不是用于针对它们测试代码变更。我们提出了Chronicle，它将智能体运行在其非确定性边界处记录为不可变的数据包，并从记录中进行重放。其核心操作——切点重放，从记录中提供所选边界子集的数据，并使用新代码实时执行互补子集，从而将记录的故障事件转化为可在持续集成中运行的回归测试。在一个包含6个记录故障并使用模拟模型边界的基准测试中，记录为每次边界跨越仅增加23微秒的开销（占假设的300毫秒模型调用时间的0.008%）……

    arXiv:2609.20625v1 Announce Type: cross  Abstract: Large language model responses are non-deterministic, so failures in LLM agents are hard to reproduce: a failure depends on inference that is not bitwise reproducible, on tools that read changing state, and on a multi-step trajectory that a re-run rarely repeats. Record-and-replay makes a run reproducible, but existing agent tooling records runs only to trace or score them, not to test a code change against them. We present Chronicle, which records an agent run at its non-deterministic boundaries as immutable envelopes and replays it from the record. Its central operation, cut-point replay, serves a chosen subset of boundaries from the record and executes the complementary subset live with new code, turning a recorded incident into a regression test that runs in continuous integration. On a benchmark of 6 recorded failures with simulated model boundaries, recording adds 23 {\mu}s per crossing (0.008% of an assumed 300 ms model call), f
    
[^16]: 特权信息能为在策略自蒸馏带来什么增益？

    What Does Privileged Information Add to On-Policy Self-Distillation?

    [https://arxiv.org/abs/2609.20612](https://arxiv.org/abs/2609.20612)

    论文通过构建包含六种共享答案推理视图的数学问题集AMPLE-Math，发现无参考蒸馏贡献了在策略自蒸馏中的大部分性能提升，而特权信息（如参考答案或完整解题过程）带来的额外收益有限且依赖于学生模型本身。

    

    在策略自蒸馏（OPSD）让语言模型从一个能够看到答案或完整解题过程的自身冻结副本中学习。给教师模型提供这种额外信息似乎能为学生模型提供更多可学习的内容，但除了蒸馏本身之外，它还能带来多少增益？为了隔离这一贡献，我们构建了AMPLE-Math——一个包含5,319道数学题的可复用问题集，这些问题具有共享相同答案的六种推理视图，并将每种视图与匹配的无参考蒸馏进行比较。在启用思考模式的教师模型监督直接回答rollout的设置下，无参考蒸馏解释了Qwen3-1.7B在启用思考模式评估下的大部分改进，无论是在领域内还是在外部基准上。额外参考收益的证据在Qwen上较为温和，在精炼解答上最为显著，而完整推理轨迹则在SmolLM3-3B的第50步带来了两个百分点的提升。这些收益依赖于正在被训练的学生模型。与此同时……

    arXiv:2609.20612v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) lets a language model learn from a frozen copy of itself that sees an answer or a worked solution. Giving the teacher this extra information seems to offer the student more to learn, but how much does it add beyond distillation itself? To isolate that contribution, we construct AMPLE-Math, a reusable suite of 5,319 mathematical problems with six reasoning views that share the same answer, and compare each view with matched reference-free distillation. With a thinking-enabled teacher supervising direct-response rollouts, reference-free distillation accounts for much of Qwen3-1.7B's improvement under thinking-enabled evaluation, both in domain and on external benchmarks. Evidence for an additional reference benefit is modest in Qwen, strongest for a polished solution, whereas complete traces add two percentage points in SmolLM3-3B at step 50. These benefits depend on the student being trained. At the same
    
[^17]: WiC并非WSD：关于大语言模型与词汇歧义消解的研究

    WiC is Not WSD: A Study on LLMs and Lexical Ambiguity Resolution

    [https://arxiv.org/abs/2609.20593](https://arxiv.org/abs/2609.20593)

    本研究揭示WiC任务的困难部分源于缺乏明确的词义清单，为模型提供候选词义可显著提升其WiC任务表现，且许多表面错误实为标注歧义或词义边界不匹配，而非模型真正的词汇理解失败。

    

    尽管词汇-语义任务近来有所进展，上下文词义理解（WiC）对语言模型而言仍然具有挑战性。我们假设这一困难不仅源于对同一词在两个语境中用法的比较，还源于缺乏一个明确的词义清单来指明相关的语义粒度层次。我们在相似的设置下评估了开源大语言模型在WiC和传统词义消歧（WSD）任务上的表现。我们发现，提供候选词义（类似于传统WSD中的做法）能够在所有设置下提升WiC性能。总体而言，明确的词义信息有助于模型做出更一致、更有针对性的判断。人工评估进一步表明，许多表面上的WiC错误实际上反映的是标注歧义或模型与标注者之间词义边界的不匹配，而非简单的词汇理解失败。特别地，结果显示大语言模型会过度思考词义区分，常常因此产生错误。

    arXiv:2609.20593v1 Announce Type: new  Abstract: Word-in-Context (WiC) remains challenging for language models, despite recent progress on lexical-semantic tasks. We hypothesise that this difficulty arises not only from comparing two contextual uses of a word, but also from the absence of an explicit sense inventory that specifies the relevant level of semantic granularity. We evaluate open LLMs on WiC and traditional Word Sense Disambiguation (WSD) under similar settings. We find that providing candidate senses, similar to what is done in traditional WSD, improves WiC performance in all settings. In general, explicit sense information helps models make more consistent and targeted judgements. Human evaluation further shows that many apparent WiC errors reflect label ambiguity or mismatches between model and annotator sense boundaries rather than simple failures of lexical understanding. In particular, results show that LLMs overthink the sense distinction often leading to errors based
    
[^18]: SAFARI：一个用于LLM辅助危害分析与风险评估的工业基准

    SAFARI: An Industrial Benchmark for LLM-Assisted Hazard Analysis and Risk Assessment

    [https://arxiv.org/abs/2609.20584](https://arxiv.org/abs/2609.20584)

    该论文提出了首个面向ISO 26262汽车功能安全领域的LLM辅助危害分析与风险评估工业基准SAFARI（包含3000个去标识化真实工业案例），并通过实验揭示前沿LLM虽能生成合理的危害叙述，但在标准风险分类上表现薄弱（最佳ASIL宏观F1仅0.261），思维链提示反而常降低分类性能。

    

    大型语言模型（LLM）越来越多地被考虑应用于安全关键工程领域，但其在受监管的功能安全工作流程中的可靠性仍未得到充分探索。我们提出了SAFARI（安全感知功能汽车风险推理，Safety-Aware Functional Automotive Risk Inference），这是首个面向ISO 26262标准下LLM辅助汽车危害分析与风险评估（HARA）的工业基准。该基准包含3,000个去标识化的工业HARA案例，并评估两个相互关联的任务：开放式危害分析和基于标准的风险评估。为了评估开放式HARA成果，我们提出了首个以参考答案为锚定且与专家判断具有高相关性的LLM-as-a-judge（LLM作为评判者）协议。对九个前沿LLM的实验表明，模型通常能够生成看似合理的危害叙述，但在ISO 26262风险分类方面仍然薄弱，最佳的ASIL宏观F1分数仅为0.261。思维链（Chain-of-Thought）提示带来的收益有限，且常常会降低分类风险评估的性能。

    arXiv:2609.20584v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly considered for safety-critical engineering, yet their reliability in regulated functional-safety workflows remains underexplored. We introduce SAFARI (Safety-Aware Functional Automotive Risk Inference), the first industrial benchmark for LLM-assisted automotive Hazard Analysis and Risk Assessment (HARA) under ISO 26262. It contains 3,000 de-identified industrial HARA cases and evaluates two coupled tasks: open-ended hazard analysis and standards-grounded risk assessment. To evaluate open-ended HARA artifacts, we propose the first reference-anchored LLM-as-a-judge protocol with high expert correlation. Experiments with nine frontier LLMs show that models often produce plausible hazard narratives but remain weak at ISO 26262 risk classification, with the best ASIL macro-F1 reaching only 0.261. Chain-of-Thought prompting provides limited benefit and often degrades categorical risk assessment. Er
    
[^19]: 掌舵指南针：将动态心理咨询对话与认知行为疗法策略对齐

    Steering the Compass: Aligning Dynamic Psychological Counseling Conversations with Cognitive Behavioral Therapy Strategies

    [https://arxiv.org/abs/2609.20565](https://arxiv.org/abs/2609.20565)

    提出StratCBT数据集，包含9,688个会话和约25.6万条话语，首次将心理咨询对话与八种认知行为疗法策略对齐，弥补了现有研究忽视基于来访者实时心理状态进行动态决策的不足。

    

    大语言模型的最新进展彻底改变了心理咨询领域，尤其是在认知行为疗法（CBT）的背景下。虽然CBT的成功在很大程度上依赖于根据来访者实时心理状态进行动态决策，但这一方面在当前研究中往往被忽视，限制了灵活性和治疗效果。在本文中，我们提出了StratCBT，一个专为包含CBT策略的心理咨询对话设计的数据集，包含9,688个会话和约25.6万条话语，每位咨询师的回应都与八种不同策略之一对齐。StratCBT的构建涉及基于来访者的负面思维对其进行建模，并通过自我对话生成高质量的咨询对话，同时结合真实会话作为指导，从而在一般咨询和CBT方面显著超越现有数据集。

    arXiv:2609.20565v1 Announce Type: new  Abstract: Recent advancements in large language models have revolutionized the field of psychological counseling, especially in the context of Cognitive Behavioral Therapy (CBT). While the success of CBT relies heavily on dynamic decision-making informed by the client's real-time mental state, this aspect has often been overlooked in current research, limiting both flexibility and therapeutic outcomes. In this paper, we introduce StratCBT, a dataset specifically designed for psychological counseling conversations with CBT Strategies, consisting of 9,688 sessions and around 256K utterances, with each counselor's response aligned with one of eight distinct strategies. The creation of StratCBT involves modeling clients based on their negative thoughts and generating high-quality counseling conversations through self-chat, incorporating realistic sessions as guidance, thereby significantly surpassing existing datasets in both general counseling and CB
    
[^20]: 语言模型群体在重演人类推理任务审议时高估了共识程度

    Language-model groups overstate consensus when replaying human deliberation on a reasoning task

    [https://arxiv.org/abs/2609.20543](https://arxiv.org/abs/2609.20543)

    本研究通过让信念锚定的LLM智能体重演人类在华生推理任务中的小组讨论，发现语言模型群体的共识度显著高于人类群体（差距达34至44个百分点），且该结论在多种测量方法下均稳健成立，表明语言模型会系统性高估群体共识。

    

    完全共识率常被视为集体认知的指标，但其结果取决于如何操作化定义参与度与最终状态。我们使用匹配的大语言模型（LLM）智能体群体重演了100个留出的人类华生（Wason）任务小组，根据每位参与者讨论前的答案植入一个信念锚定的智能体，并使用相同的评分代码对智能体和人类进行评分。在不同的人类评分定义下，共识率估计值介于24.0%至57.0%之间；约五分之一的参与者从未发言，而智能体则几乎总是发言。在揭盲后的两项敏感性分析中，智能体群体仍然表现出更高的共识性：基于提交的比较（n = 98）显示聊天模式和推理模式的差距分别为34.0和43.9个百分点，参与度匹配的比较（n = 45）显示差距分别为34.1和44.4个百分点。这两种互补的方法减少了不同的测量不对称性，但其结果在0.5个百分点内趋于一致。该差距在无早期的情况下依然持续存在……

    arXiv:2609.20543v1 Announce Type: new  Abstract: Full-consensus rates are often treated as indicators of collective cognition, yet depend on how participation and final states are operationalized. We replayed 100 held-out human Wason groups with matched large language model (LLM) agent groups, seeding one belief-anchored agent per participant's pre-discussion answer and scoring agents and people with the same code. Across human scoring definitions, estimates ranged from 24.0% to 57.0%; about one fifth of participants never posted, whereas agents almost always did. Agent groups remained more consensual in two post-unblinding sensitivity analyses: the submit-based comparison (n = 98) yielded gaps of 34.0 and 43.9 percentage points for chat and reasoning modes, and the participation-matched comparison (n = 45) yielded gaps of 34.1 and 44.4 points. These complementary routes reduced different measurement asymmetries yet converged within 0.5 percentage points. The gap persisted without earl
    
[^21]: 语言模型免训练自我报告置信度的分析

    An Analysis of Training-Free Self-Reported Confidence in Language Models

    [https://arxiv.org/abs/2609.20541](https://arxiv.org/abs/2609.20541)

    该研究发现语言模型直接口头表达的置信度是出乎意料强的免训练正确性预测信号（AUROC高达0.956），显著优于基于多样本一致性的方法，而自我一致性反而可能放大模型共同的系统性误解。

    

    大语言模型可以在生成内容的同时报告一个数值置信度，但目前尚不清楚这种报告是否仅仅是经过校准的修辞。我们分析了三种免训练信号：与答案一同口头表达的置信度、事后P(True)，以及在相同的100道TriviaQA问题上与三次额外生成结果的一致性，涵盖两个模型家族。直接的口头表达是一个惊人强大的基线：在审查了基准测试错误后，其正确性预测的AUROC达到0.956和0.937。三样本一致性明显较弱（0.765和0.790），且与口头置信度进行固定插值没有统计学上可靠的收益。一个模型的九个错误中有四个、另一个模型的八个错误中有两个获得了全部样本的一致支持，这表明自我一致性可能会放大模型共同的误解。使用等效提示对相同的固定答案重新引出置信度会使分数变化0.043至（原文摘要在此处截断）

    arXiv:2609.20541v1 Announce Type: new  Abstract: Large language models can report a numerical confidence together with generated content, but it is unclear whether this report is more than calibrated rhetoric. We analyze three training-free signals: confidence verbalized with the answer, post-hoc $P(\mathrm{True})$, and agreement with three additional generations on the same 100 TriviaQA questions for two model families. Direct verbalization is a surprisingly strong baseline: after auditing benchmark errors, it reaches AUROC 0.956 and 0.937 for correctness prediction. Three-sample agreement is substantially weaker (0.765 and 0.790), and a fixed interpolation with verbalized confidence has no statistically reliable benefit. Four of nine errors from one model and two of eight from the other receive unanimous sample support, showing that self-consistency can amplify shared misconceptions. Re-eliciting confidence for the same fixed answers with equivalent prompts changes scores by 0.043 to
    
[^22]: 面向数据高效语言建模的关系注意力机制

    Relational Attention for Data-Efficient Language Modeling

    [https://arxiv.org/abs/2609.20530](https://arxiv.org/abs/2609.20530)

    该论文提出在双注意力Transformer架构中将关系注意力与自注意力相结合，并借助BabyLM 2026挑战赛的数据受限环境，验证关系注意力所带来的数据效率能否成功迁移到语言建模任务中。

    

    我们提出了Relational BabyLM，这是提交给BabyLM 2026挑战赛的一个系统，它在单个仅解码器的Transformer中结合了两种基于认知科学启发的归纳偏置。在架构上，我们用双注意力Transformer（DAT）替代标准的自注意力机制，将对象级（“感官”）词汇特征的路由与结构/关系信息分离开来。从自注意力中解耦出来的关系注意力（RA）在纯关系任务上能极大地提高数据效率和训练样本外的泛化能力，但语言建模要求对象级信息和关系信息既能解耦又能整合，而基于RA的语言模型在很大程度上仍未被探索。BabyLM挑战赛的数据受限训练和全面评估为检验这种数据效率能否迁移到语言建模中提供了理想的测试平台。

    arXiv:2609.20530v1 Announce Type: new  Abstract: We present Relational BabyLM, a system submission to the BabyLM 2026 challenge that combines two cognitively motivated inductive biases in a single decoder-only Transformer. Architecturally, we replace standard self-attention with a Dual Attention Transformer (DAT), which separates the routing of object-level ("sensory") lexical features from structural/relational information (Altabaa and Lafferty, 2025; Altabaa et al., 2024; Webb et al., 2024; Kerg et al., 2022; Webb et al., 2021). Relational attention (RA) disentangled from self-attention greatly increases data efficiency and out-of-training-sample generalization on purely relational tasks, but language modeling requires object-level and relational information to be integrated as well as disentangled, and RA-based LMs have remained largely unexplored. BabyLM's data-constrained training and comprehensive evaluation is an ideal testing ground for whether that data efficiency transfers. A
    
[^23]: 面向农业领域的模型无关与语言无关语音流水线改进

    Model-Agnostic and Language-Agnostic Voice Pipeline Improvement for the Agriculture Domain

    [https://arxiv.org/abs/2609.20504](https://arxiv.org/abs/2609.20504)

    该论文提出了一种无需微调或替换底层 ASR 模型的模块化、模型无关语音流水线，通过音频增强、说话人分离、农业领域词典纠错和质量门控，显著提升了嘈杂田间环境下农业咨询场景的语音识别质量。

    

    FarmerChat 是 Digital Green 推出的面向小农户的 AI 农业咨询助手，农户可以通过文本、语音或照片以自己的语言使用该服务。语音是该群体的重要使用渠道，但田间录制的语音对通用自动语音识别（ASR）系统极具挑战性，因为录音中经常包含机械噪音、背景媒体声、竞争说话人以及特定领域的农业词汇。这些不利条件会不成比例地影响承载农户查询语义的作物、害虫、化学品和数量等关键词汇。我们提出了一种模块化、模型无关的流水线，用于在不进行微调或替换底层 ASR 模型的情况下提升 FarmerChat 的 ASR 质量。该流水线结合了门控音频增强、说话人分离与目标说话人选择、ASR 识别、基于加权农业词典的领域感知纠错，以及用于检测不可靠转录文本的质量门控机制。

    arXiv:2609.20504v1 Announce Type: cross  Abstract: FarmerChat is Digital Green's AI-powered agricultural advisory assistant for smallholder farmers, who access it in their own language through text, voice, or photographs. Voice is a critical channel for this population, yet field-recorded speech is challenging for general-purpose automatic speech recognition (ASR) because recordings frequently contain machinery noise, background media, competing speakers, and domain-specific agricultural vocabulary. These conditions disproportionately affect crop, pest, chemical, and quantity terms that carry the meaning of a farmer's query.   We present a modular, model-agnostic pipeline for improving ASR quality in FarmerChat without fine-tuning or replacing the underlying ASR model. The pipeline combines gated audio enhancement, speaker diarization and target-speaker selection, ASR, domain-aware correction using a weighted agricultural lexicon, and a quality gate for detecting unreliable transcripts
    
[^24]: Edustories：来自课堂实践的真实世界案例研究集

    Edustories: A Collection of Real-world Case Studies from Classroom Practices

    [https://arxiv.org/abs/2609.20484](https://arxiv.org/abs/2609.20484)

    该研究推出了Edustories数据集——包含1,492个教师撰写的真实课堂案例研究，用于评估大语言模型预测教师教学干预成效的能力，并发现当前最强模型的预测准确率仅为58%，仍不及人类专家水平。

    

    尽管人工智能在教育领域的潜力已被广泛认可，但以往的研究大多集中于个体化的学生辅助。相比之下，全球大多数教育实践仍然发生在集体课堂环境中。为了使研究人员能够研究集体教学中的AI辅助，我们推出了Edustories——一个包含1,492个由教师撰写的案例研究的数据集，描述了涉及挑战性学生行为、教学干预及其结果的真实小学和高中课堂情境。除众多其他应用外，Edustories还能够评估大型语言模型预测教师干预成功与否的能力，这对于为一线教师提供有用的反馈至关重要。通过将来自四个语言模型系列的最新模型与专家评估进行比较，我们发现当前模型在预测课堂结果方面仍不及人类专家；最强大的模型达到了58%的准确率，而相比之下……

    arXiv:2609.20484v1 Announce Type: cross  Abstract: Despite the widely recognized potential of AI in education, most prior work has focused on individualized student assistance. In contrast, the majority of educational practice worldwide still takes place in collective classroom settings. To enable researchers to study AI assistance in collective teaching, we introduce Edustories, a dataset of 1,492 teacher-written case studies describing real elementary and high-school classroom situations involving challenging student behavior, pedagogical interventions, and their outcomes. Among many other applications, Edustories enables evaluating LLMs' ability to predict the success of teacher interventions, crucial for providing practicing teachers with useful feedback. Comparing the latest models from four language-model families against expert assessments, we find that current models fall short of human expertise in predicting classroom outcomes; the strongest models reach 58% accuracy compared
    
[^25]: 压力测试对齐中期训练

    Stress-testing Alignment Midtraining

    [https://arxiv.org/abs/2609.20412](https://arxiv.org/abs/2609.20412)

    该论文通过大规模实验（高达1100亿参数模型和10亿中期训练token）对对齐中期训练（AMT）的多个假设进行压力测试，发现在简单场景下中期训练能够引导模型动机，但其有效性仍存在局限。

    

    当通过后训练技术对前沿模型进行对齐时，我们无法直接演示模型在所有可能的部署环境中应表现出的所有行为；模型必须在后训练分布之外进行泛化。一种被提出的解决方案是对齐中期训练（AMT），即在大量与对齐相关的文档上继续预训练，以促进后续训练阶段的泛化能力。尽管AMT作为一种对齐方法备受关注，但关于其有效性的公开证据仍然有限。为了解决这一问题，我们识别了围绕中期训练的若干假设，并在不同规模上对其进行评估：模型规模高达1100亿参数，中期训练token数量达10亿。例如，我们研究了一种场景，其中后训练数据在两种可能的动机之间是模糊不清的。我们发现，在该设置的简单版本中，中期训练可以引导模型的动机。然而，……

    arXiv:2609.20412v1 Announce Type: cross  Abstract: When aligning frontier models through post-training techniques, it is not possible to directly demonstrate all of the behaviours we want a model to exhibit in all possible deployment environments; our model must generalise outside of the post-training distribution. One proposed solution is alignment midtraining (AMT), which continues pretraining on large volumes of alignment-relevant documents to encourage generalisation in later stages of training.   Despite the prominence of AMT as an alignment approach, there is limited public evidence for its effectiveness. To resolve this, we identify several assumptions around midtraining and evaluate them across scale: up to 110 billion-parameter models and 1 billion midtraining tokens. For instance, we study a scenario where post-training data is ambiguous between two possible motivations. We find that midtraining can steer the model's motivation in simple versions of this setting. However, the
    
[^26]: 异种可解释性：探索大语言模型的“异类心智”

    Xeno-Interpretability: Investigating the Alien Minds of LLMs

    [https://arxiv.org/abs/2609.20408](https://arxiv.org/abs/2609.20408)

    本文提出“异种可解释性”这一新研究方向，主张大语言模型内部可能存在人类概念无法充分描述的“异种表征”，其内部区分空间远超有限人类描述所能覆盖的范围，且实验识别与语义解释应当分开对待。

    

    大语言模型通常通过人类已有的概念来进行解释：真实性、拒绝、欺骗、人格、危害性以及相关类别。本文提出了一个问题：模型是否也可能表征并使用那些不存在恰当人类概念的区分。我们将这类内部结构称为“异种表征”，对其研究称为“异种可解释性”。我们区分了人类可解释的语义空间与“异种语义空间”——即模型原生表征中缺乏恰当人类概念对应物的区域。我们证明，大语言模型中可能的内部区分空间显著大于通过有限人类描述所能覆盖的空间。随后，我们将实验识别与语义解释加以区分：一个内部表征即使……可以被可复现地定位、进行几何刻画、加以因果操纵，并与下游行为建立关联。（摘要在此处不完整）

    arXiv:2609.20408v1 Announce Type: cross  Abstract: Large language models are usually interpreted through concepts that humans already possess: truthfulness, refusal, deception, personality, harmfulness, and related categories. This paper asks whether models may also represent and use distinctions for which no adequate human concept exists. We call such internal structures xeno-representations, and their study xeno-interpretability. We distinguish the human-interpretable semantic space from the xeno-semantic space: the region of model-native representations for which no adequate human conceptual counterpart is available. We show that the space of possible internal distinctions in an LLM is substantially larger than the space available through finite human descriptions. We then separate experimental identification from semantic interpretation: an internal representation may be reproducibly located, geometrically characterized, causally manipulated, and linked to downstream behaviour even
    
[^27]: 面向基于语义解析的知识库问答的模式锚定潜在推理方法

    Schema-Anchored Latent Reasoning for Semantic Parsing-Based Knowledge Base Question Answering

    [https://arxiv.org/abs/2609.20398](https://arxiv.org/abs/2609.20398)

    提出SALR方法，通过在模型隐藏状态中进行模式锚定的潜在多步推理，延迟对逻辑形式决策的显式承诺，避免错误的中间模式选择传播，从而提升基于语义解析的知识库问答性能。

    

    基于语义解析（SP）的知识库问答旨在通过在知识库（KB）上生成可执行的逻辑形式（LF）来回答自然语言问题。在将大语言模型（LLM）应用于该任务时，面对大型异构知识库的一个关键挑战是选择与问题相关的模式元素（即关系和类），并将它们组合成复杂的逻辑形式。近期基于LLM的方法通常在中间推理过程中过早地对模式元素做出离散承诺，导致错误的中间模式决策不断传播，最终产生错误的逻辑形式。为克服这一局限性，我们提出了SALR，一种用于逻辑形式构建的模式锚定潜在推理方法。该方法通过在模型隐藏状态中生成连续思维来执行多步推理，从而延迟对逻辑形式决策的显式承诺。为使该潜在推理过程扎根于相应的知识库模式，SALR对齐（模式信息与推理过程），从而提升语义解析式知识库问答的准确性与鲁棒性。

    arXiv:2609.20398v1 Announce Type: new  Abstract: Semantic parsing (SP)-based knowledge base question answering aims to answer natural language questions by generating executable logical forms (LFs) over knowledge bases (KBs). When applying Large Language Models (LLMs) to this task, a key challenge over large, heterogeneous KBs is selecting question-related schema elements (i.e., relations and classes) and composing them into complex LFs. Recent LLM-based methods often make early discrete commitments to schema elements during intermediate reasoning, allowing incorrect intermediate schema decisions to propagate and finally result in incorrect LFs. To overcome this limitation, we propose SALR, a schema-anchored latent reasoning method for LF construction. It performs multi-step reasoning by generating continuous thoughts in the model's hidden states, thereby delaying the explicit commitment to LF decisions. To ground this latent reasoning process in the corresponding KB schema, SALR align
    
[^28]: Viveka-Insight：一个覆盖斯瓦米·维韦卡南达英文与孟加拉文完整著作的跨语言概念图及基于引用接地的检索资源

    Viveka-Insight: a cross-lingual concept graph and citation-grounded retrieval resource over the complete works of Swami Vivekananda in English and Bengali

    [https://arxiv.org/abs/2609.20303](https://arxiv.org/abs/2609.20303)

    该论文发布了Viveka-Insight，一个针对斯瓦米·维韦卡南达英文与孟加拉文完整著作的双语开源资源，通过结构保持解析和跨语言概念图解决了古典哲学语料库多语言不对齐、词汇陈旧和文化敏感内容难以验证接地的问题。

    

    古典哲学语料库给语言资源带来了三个叠加的挑战：它们以多种语言存在但缺乏平行对齐，其词汇与当代读者的词汇相距甚远，并且针对文化敏感材料生成的文本必须可验证地接地。我们提出了Viveka-Insight，这是一个针对斯瓦米·维韦卡南达（1863-1902）著作的双语资源和开源流水线：包括九卷本英文《全集》和十卷本孟加拉文《Vani o Rachana》，两个相关但非平行的语料库，共约1500万字符。本资源发布了四个层次：（i）保留结构的解析（32,694个段落，168,842个句子），每个段落都带有可深度链接到已出版版本的锚点；（ii）一个包含8,362个语言无关概念的跨语言概念图，具有87,518条带关系类型的段落-概念边和55,872条概念-概念边，其中规范的英文标签作为字符串相等键链接

    arXiv:2609.20303v1 Announce Type: new  Abstract: Classical philosophical corpora pose three compounding challenges for language resources: they exist in several languages without parallel alignment, their vocabulary is remote from that of contemporary readers, and generated text over culturally sensitive material must be verifiably grounded. We present Viveka-Insight, a bilingual resource and open-source pipeline for the works of Swami Vivekananda (1863-1902): the nine-volume English Complete Works and the ten-volume Bengali Vani o Rachana, two related but non-parallel corpora of about 15 million characters. Four layers are released: (i) a structure-preserving parse (32,694 paragraphs, 168,842 sentences) with per-paragraph anchors deep-linking into the published editions; (ii) a cross-lingual concept graph of 8,362 language-agnostic concepts with 87,518 relation-typed paragraph-concept and 55,872 concept-concept edges, in which canonical English labels act as a string-equality key link
    
[^29]: Lens：为免训练多模态表征学习聚焦正确的语义视角

    Lens: Bringing the Right Semantic Perspective into Focus for Training-Free Multimodal Representation Learning

    [https://arxiv.org/abs/2609.20252](https://arxiv.org/abs/2609.20252)

    论文指出免训练多模态表征学习中存在语义视角错位问题——现有语义引导方法无法使自回归模型提取的表征聚焦于下游任务所需的语义视角，并提出Lens方法来解决这一问题。

    

    高质量的表征对于广泛的下游任务至关重要。专用的嵌入模型是为表征学习显式优化而来的，但其训练数据在规模和多样性上往往不及用于预训练现代大型语言模型和多模态大型语言模型的海量语料库。大规模预训练和指令遵循能力使自回归模型能够选择相关证据、整合多模态信息，并在不同的任务视角下推断语义，这为免训练表征学习创造了独特的机会。然而，我们的分析表明，现有的语义引导方法无法可靠地将提取的隐藏状态定向到下游任务所需的语义视角上。因此，所得的表征往往仍然被显著的输入内容所主导。我们将这一问题刻画为语义视角错位（semantic perspective misalignment）

    arXiv:2609.20252v1 Announce Type: cross  Abstract: High-quality representations are essential for a wide range of downstream tasks. Dedicated embedding models are explicitly optimized for representation learning, yet their training data are often more limited in scale and diversity than the massive corpora used to pretrain modern large language models and multimodal large language models. Large-scale pretraining and instruction following enable autoregressive models to select relevant evidence, integrate multimodal information, and infer semantics under different task perspectives, creating a distinctive opportunity for training-free representation learning. However, our analysis reveals that existing semantic-elicitation methods do not reliably orient the extracted states toward the semantic perspective required by the downstream task. Consequently, the resulting representations often remain dominated by salient input content. We characterize this problem as semantic perspective misal
    
[^30]: 公共话语语料库（PDC）：一个带有目标说话人参与的情感效价与认知情态说话人归因数据集

    The Public Discourse Corpus (PDC): A Speaker-Attributed Dataset for Valence and Epistemic Modality with Target Speaker Participation

    [https://arxiv.org/abs/2609.20232](https://arxiv.org/abs/2609.20232)

    该论文发布了首个对公众人物访谈语音进行情感效价与认知情态联合标注的公共话语语料库（PDC），并提出目标说话人参与（TSP）标注方法作为可推广的语料库构建方法学贡献。

    

    我们介绍了公共话语语料库，这是首个针对公众人物访谈语音同时进行情感效价与认知情态联合标注的数据集。该语料库包含来自七个专业领域的100位说话人的998个视频，经过句子切分和过滤后得到186,642个句子（310万个词）。为确保所有保留的视频都包含目标说话人可分析的语音，我们提出了目标说话人参与——一个包含五个类别的标注分类体系，其标注者间信度已被验证（κ = 0.616）——作为任何语料库构建项目均可采用的关键方法学贡献。目标说话人的发言通过一个“音频优先”的说话人分离流水线与访谈者及第三方语音分离开来，该流水线结合了本地Whisper语音识别与pyannote说话人分离技术，并以开源实现的形式发布。我们发布了标注语料库……

    arXiv:2609.20232v1 Announce Type: new  Abstract: We introduce the \textbf{Public Discourse Corpus (PDC)}, the first dataset of public-figure interview speech jointly annotated for affective valence and epistemic modality. The corpus contains 998 videos from 100 speakers across seven professional domains, yielding 186,642 sentences (3.1 million words) after sentence segmentation and filtering. To ensure that all retained videos contain analyzable speech from the intended speaker, we introduce \textbf{Target Speaker Participation (TSP)}---a five-category annotation taxonomy with documented inter-annotator reliability ($\kappa = 0.616$)---as a key methodological contribution that any corpus construction project can adopt. Target-speaker turns are separated from interviewer and third-party speech through an \textbf{audio-first diarization pipeline} combining local Whisper ASR with pyannote speaker separation, released as an open-source implementation. We release the annotated corpus, the a
    
[^31]: 趁警告还来得及：基于语音的电话诈骗增量检测

    Before the Warning Comes Too Late: Incremental Phone-Scam Detection from Speech

    [https://arxiv.org/abs/2609.20223](https://arxiv.org/abs/2609.20223)

    提出了StreamFraudNet，一种基于冻结自监督语音编码器、循环时间建模和窗口分数聚合的流式电话诈骗检测模型，能够在通话过程中每2秒增量更新诈骗风险评分，在英语基准上达到0.9953的ROC-AUC。

    

    我们研究了从原始电话音频中进行弱监督的电信诈骗增量检测任务，其中训练数据仅提供对话级别的标签，且预测必须在通话结束前实时更新。我们提出了StreamFraudNet，该模型通过重叠的有界上下文窗口处理输入音频，采用冻结的自监督语音编码器、循环时间建模以及可学习的潜在窗口分数聚合机制。在一个受控的英语基准测试中，StreamFraudNet达到了0.9953的ROC-AUC，显著优于声学基线和平均池化基线，同时与强大的全局时间模型保持竞争力。该模型在接收10秒音频后即可产生首个预测，此后每2秒更新一次，并在所评估的服务器硬件上以快于实时的速度运行。消融实验表明，循环时间上下文是性能提升的最主要贡献因素。这些结果表明，诈骗风险可以在通话过程中被增量地评分。

    arXiv:2609.20223v1 Announce Type: new  Abstract: We study weakly supervised incremental telecom fraud detection from raw telephone audio, where training provides only conversation-level labels and predictions must be updated before a call ends. We introduce StreamFraudNet, which processes incoming audio through overlapping bounded-context windows using a frozen self-supervised speech encoder, recurrent temporal modeling, and learned aggregation of latent window scores. On a controlled English benchmark, StreamFraudNet achieves a ROC--AUC of \(0.9953\), significantly outperforming acoustic and mean-pooling baselines while remaining competitive with strong global temporal models. The model produces its first prediction after 10 seconds of audio, updates every 2 seconds, and operates faster than real time on the evaluated server hardware. Ablations identify recurrent temporal context as the principal contributor to performance. These results demonstrate that fraud risk can be scored incre
    
[^32]: 随机词汇演算的基础：概率单纯形上的语义下降与随机动力学

    Foundations of Stochastic Lexical Calculus: Semantic Descent and Random Dynamics on Probability Simplices

    [https://arxiv.org/abs/2609.20207](https://arxiv.org/abs/2609.20207)

    本文建立了一个可观测的理论框架，给出了语言导出的概率能够唯一支持语义状态更新的充要条件，并在平均收缩条件下证明了概率单纯形上随机递归的存在性、唯一性与稳定性，从而构建了一种无需将内部演算归因于语言模型的随机词汇演算。

    

    大语言模型产生的是依赖于提示词的词汇概率，而科学系统需要对有意义状态的不确定性建模，并随证据的到来而不断更新。我们开发了一个可观测的框架，用以判定何时由语言导出的概率能够支持这种序贯状态表示。在理论上，我们定义了上下文语言的类型化可测变换，构建了一个最小的闭表示，并给出了语义更新唯一存在的充要条件。我们界定了不可消除的非闭合性与累积误差，并在平均收缩条件下证明了概率单纯形上外部随机递归的存在性、唯一性与稳定性。这些结果定义了一种随机词汇演算，而无需将某种内部演算归因于语言模型本身。在实证方面，冻结实验检验了该理论的可观测含义。原始的提示条件概率未能通过预先指定的（测试）……

    arXiv:2609.20207v1 Announce Type: new  Abstract: Large language models produce prompt-dependent probabilities over words, whereas scientific systems require uncertainty over meaningful states that can be updated as evidence arrives. We develop an observable framework for determining when language-derived probabilities support such a sequential state representation. Theoretically, we define typed measurable transformations of contextual language, construct a minimal closed representation, and give necessary and sufficient conditions for semantic updates to exist uniquely. We bound irreducible nonclosure and accumulated error, and under average contraction prove existence, uniqueness and stability of an external random recursion on a probability simplex. These results define a stochastic lexical calculus without attributing an internal calculus to the language model. Empirically, frozen experiments test the observable implications. Raw prompt-conditioned probabilities fail the prespecifi
    
[^33]: 复制还是不复制：通过内在模型信号控制推测解码

    To Copy or Not to Copy: Controlling Speculative Decoding via Intrinsic Model Signals

    [https://arxiv.org/abs/2609.20186](https://arxiv.org/abs/2609.20186)

    SwitchSD通过在目标模型内部表示上训练轻量级探测器来识别真正的复制意图，从而在神经草稿生成与上下文复制两种策略间自适应切换，有效控制推测解码过程并提升大语言模型推理吞吐量。

    

    推测解码显著加速了大语言模型的推理，然而现有方法在两种草稿生成策略之间面临根本性的权衡：神经草稿生成与基于上下文的复制。神经草稿方法（如EAGLE3）在多样化文本场景中表现稳健，而基于复制的方法在复制密集型场景中通过更快地生成候选并利用长重复片段实现近乎完美的推测，从而获得更高的加速比。我们分析了现有的基于复制的方法，发现它们容易受到偶然重复的干扰——表面上的n-gram重叠并不反映结构性的复制意图，导致假阳性触发，最终降低吞吐量。我们提出了SwitchSD，一个将复制视为大语言模型潜在控制信号的自适应框架。通过在目标模型的内部表示上训练轻量级探测器，SwitchSD能够识别真正的复制意图。

    arXiv:2609.20186v1 Announce Type: new  Abstract: Speculative Decoding (SD) has significantly accelerated Large Language Model (LLM) inference, yet existing approaches face a fundamental tradeoff between two drafting strategies: neural drafting and context-based copying. Neural drafts (e.g., EAGLE3) provide robust performance across diverse text settings, while copy-based methods achieve higher speedups in copy-intensive regimes by generating candidates faster and exploiting long repetition spans for near-perfect speculation. We analyze existing copy-based methods and find that they are prone to accidental repetitions where surface-level n-gram overlap does not reflect a structural intent to copy, leading to false-positive triggers that ultimately degrade throughput. We introduce SwitchSD, an adaptive framework that treats copying as a latent control signal of the LLM. By training lightweight probes on the target model's internal representations, SwitchSD identifies genuine copy-intent 
    
[^34]: 面向生物医学关系抽取的模型微调

    Fine-Tuning Models for Biomedical Relation Extraction

    [https://arxiv.org/abs/2609.20169](https://arxiv.org/abs/2609.20169)

    本研究通过对预训练模型（尤其是DeBERTa和Gemini Pro 1.0）进行微调，实现了生物医学文本中变异-表型关系的自动抽取，其中精心微调的Gemini Pro 1.0在句子级和摘要级任务上均超越了现有最先进水平。

    

    新一代测序技术彻底改变了基因突变的研究，使得大规模探究突变在疾病发展中的作用成为可能。然而，从海量的生物医学文献中提取有意义的见解仍然是一项无法通过人工方式解决的复杂挑战。在本文中，我们提出了用于从生物医学文本中自动抽取关系的预训练模型（PTMs），特别针对变异-表型领域。我们在SNPPhenA语料库上的评估表明，微调小型基于BERT的模型，特别是DeBERTa，能够获得强大的性能，接近当前最先进水平（SOTA）。此外，我们的结果表明，经过精心微调的谷歌Gemini Pro 1.0在句子级任务（模型仅处理目标句子）和摘要级任务（模型处理整个摘要）上均超越了现有的最先进水平。

    arXiv:2609.20169v1 Announce Type: new  Abstract: Next-Generation Sequencing has revolutionized the study of genetic mutations, enabling large-scale investigations into their roles in disease development. However, extracting meaningful insights from the vast amount of biomedical literature remains a complex challenge that cannot be addressed manually. In this paper, we present pre-trained models (PTMs) for the automatic extraction of relations from biomedical text, specifically targeting the variant-phenotype domain. Our evaluation on the SNPPhenA corpus demonstrates that fine-tuning small BERT-based models, particularly DeBERTa, yields strong performance, approaching the current state-of-the-art (SOTA). Additionally, our results indicate that carefully fine-tuning Google's Gemini Pro 1.0 outperforms the existing SOTA for both sentence-level tasks (where the model processes only the target sentence) and abstract-level tasks (where the model processes the entire abstract).
    
[^35]: 三思而后重排：面向文本重排的多视角证据与推理融合

    Think Thrice Before Reranking: Multi-perspective Evidence and Reasoning Integration for Text Reranking

    [https://arxiv.org/abs/2609.20131](https://arxiv.org/abs/2609.20131)

    提出MERIT-Rank框架，通过多轨迹推理空间（MTRS）从多视角评估查询-文档相关性，将多个互补推理轨迹融合为统一排序决策，并配合渐进式排序策略优化（PRPO）训练框架，显著提升了基于LLM的文本重排的鲁棒性。

    

    基于大语言模型（LLM）的推理式重排方法在文本排序任务中展现出令人鼓舞的改进。然而，当前方法主要依赖单一的推理轨迹，导致排序结果容易受到推理错误的影响，并且在建模文档相关性所涉及的多方面信号时存在固有限制。为解决这一难题，我们提出了MERIT-Rank（面向文本重排的多视角证据与推理融合），这是一个通过建模互补性推理轨迹来提升重排鲁棒性的框架。MERIT-Rank构建了一个多轨迹推理空间（MTRS），从多个视角评估查询与文档的相关性，并引入联合重排器将这些推理路径整合为统一的排序决策。我们进一步提出了渐进式排序策略优化（PRPO），这是一种渐进式训练框架，能够在持续（优化过程中）稳定推理轨迹……

    arXiv:2609.20131v1 Announce Type: cross  Abstract: Reasoning-based reranking with Large Language Models (LLMs) has shown promising improvements in text ranking. However, current methods predominantly rely on a single reasoning trajectory, resulting in rankings that are susceptible to reasoning errors and inherently constrained in modeling the multifaceted signals underlying document relevance. To resolve this dilemma, we propose MERIT-Rank(Multi-perspective Evidence and Reasoning Integration for Text Reranking), a framework that models complementary reasoning trajectories to improve reranking robustness. MERIT-Rank formulates a Multi-Trajectory Reasoning Space (MTRS) that evaluates query-document relevance from multiple perspectives and introduces a joint reranker that consolidates these reasoning paths into a unified ranking decision. We further develop Progressive Rank Policy Optimization (PRPO), a progressive training framework that stabilizes reasoning trajectories while continuall
    
[^36]: IBM Granite 5.0 TurboCTC 语音识别模型的设计

    Design of the IBM Granite 5.0 TurboCTC ASR Model

    [https://arxiv.org/abs/2609.20104](https://arxiv.org/abs/2609.20104)

    Granite 5.0 Turbo CTC 是一个仅用公开数据训练的 4.7 亿参数语音识别模型，通过金字塔下采样、分块自注意力、Muon 优化器和推理优化等技术，在 Open ASR 排行榜上达到速度-精度 Pareto 前沿，速度比最快的竞争对手还快一倍。

    

    我们介绍了 Granite 5.0 Turbo CTC 的架构、训练方法和推理加速技术，这是一个拥有 4.7 亿参数的仅编码器模型，具有出色的速度-精度权衡。该架构在 Conformer 模块中采用基于带步长深度卷积的金字塔式时间下采样、块对角（分块）自注意力机制，并基于中间层的中间预测进行条件化。训练方面的亮点包括仅使用公开可用的数据、创新性地采用 Muon 优化器，以及平衡的数据采样。推理加速包括用线性层替代 1×1 卷积，以及优化 Conformer 模块中的注意力计算。综合这些设计，该模型在 Open ASR 排行榜的英语短音频语音识别任务中处于速度-精度 Pareto 前沿，同时速度是最快竞争模型的两倍。该模型可在宽松许可下使用，并可通过指定网址下载。

    arXiv:2609.20104v1 Announce Type: new  Abstract: We describe the architecture, training methodology and inference speedups of Granite 5.0 Turbo CTC, a 470 million parameter encoder-only model with an excellent speed-accuracy tradeoff. The architecture uses pyramidal temporal subsampling within Conformer blocks using strided depthwise convolutions, block-diagonal (chunk-wise) self-attention, and conditioning on intermediate predictions from the middle layer. Training highlights are the use of only publicly available data, the novel use of a Muon optimizer, and balanced data sampling. Inference speedups include replacing 1 x 1 convolutions with linear layers and optimizing the attention computation in the Conformer blocks. Collectively, these result in a model that is on the speed-accuracy Pareto frontier of the Open ASR leaderboard for English short-form ASR while being twice as fast as the fastest competitor. The model can be used under a permissive license and downloaded from https://
    
[^37]: MATCH：基于课程调度与分层门控奖励的模型感知工具学习

    MATCH: Model-Aware Tool Learning with Curriculum Scheduling and Hierarchically Gated Rewards

    [https://arxiv.org/abs/2609.20082](https://arxiv.org/abs/2609.20082)

    MATCH提出了一种模型感知的闭环工具学习框架，通过课程难度与策略能力共同演化的课程调度，以及按工具名称、参数键、参数值逐级门控授予信用的分层奖励机制，解决了固定阈值课程脱节与加性奖励信用泄漏两大问题。

    

    工具学习使大语言模型（LLM）能够使用外部工具来完成超出其参数化知识的任务。强化学习可以通过反馈来优化工具调用行为，但现有方法仍面临两个问题：固定阈值的课程可能与策略不断演进的能力边界脱节；当预测的工具名称错误时，加性奖励可能导致参数级别的信用泄漏。为解决这些问题，我们提出了MATCH——一个融合课程调度与分层门控奖励的模型感知工具学习闭环框架。模型感知课程学习（MACL）维护由奖励导出的样本难度，使其与策略共同演化，并在每个训练周期中选择位于当前能力边界附近的样本，同时辅以一个难度更高的top-k样本池。分层工具调用门控奖励（HTGR）将工具名称、参数键和参数值作为一条门控链进行评分，仅当上一层级预测正确时才在相应层级给予信用。

    arXiv:2609.20082v1 Announce Type: cross  Abstract: Tool learning enables large language models (LLMs) to use external tools for tasks beyond parametric knowledge. Reinforcement learning can optimize tool-call behavior from feedback, but current methods still face two problems: fixed-threshold curricula can become misaligned with the policy's evolving capability boundary, and additive rewards can leak argument-level credit when the predicted tool is wrong. To address these problems, we propose MATCH, a closed-loop framework for model-aware tool learning with curriculum scheduling and hierarchically gated rewards. Model-Aware Curriculum Learning (MACL) maintains reward-derived sample difficulty that co-evolves with the policy, and each epoch selects samples near the current capability boundary together with a top-k pool of harder cases. Hierarchical Tool-call Gated Reward (HTGR) scores tool name, argument key, and argument value as a gated chain, granting credit at each level only when p
    
[^38]: 在词元空间中读取情感：面向情感识别的语音大语言模型判别式适配

    Reading Emotions in the Token Space: Discriminative Adaptation of SpeechLLMs for Emotion Recognition

    [https://arxiv.org/abs/2609.20081](https://arxiv.org/abs/2609.20081)

    提出一种判别式适配方法，通过单层线性分类头读取语音大语言模型最后一个提示词元的隐藏状态来识别情感，在不修改主干网络的前提下提升Macro F1、消除幻觉标签，并具有可解释性。

    

    语音大语言模型在情感识别方面展现出强大潜力，但它们通过一个不适合分类任务的生成式解码器来读取预测的情感：该解码器可能输出目标集合之外的标签，且偏向高频类别。我们提出了一种判别式适配方法，通过分类头读取最后一个提示词元的隐藏状态，在单次前向传播中生成标签，且无需修改主干网络。由于该读取方式始于模型原本要解码的隐藏状态，它在其他方面完全相同的语音大语言模型中实现了生成式与判别式推理的可控对比。我们将分类头保持为单一线性层，以极小的精度损失换取可解释性：每种情感成为大语言模型输出词元空间中的一个方向，从而揭示与之相关的词元。在IEMOCAP数据集上，跨越两种语音大语言模型架构，该方法提升了宏平均F1分数并消除了幻觉输出，在真实的自动语音识别（ASR）转录文本上收益最大。

    arXiv:2609.20081v1 Announce Type: cross  Abstract: SpeechLLMs have shown strong potential for emotion recognition, yet they read the predicted emotion off a generative decoder not suited for classification: it can emit labels outside the target set and favors frequent classes. We propose a discriminative adaptation that reads the final prompt token's hidden state through a classification head, producing a label in one forward pass without modifying the backbone. Because this readout starts from the hidden state the model would otherwise decode, it gives a controlled comparison of generative and discriminative inference in an otherwise identical speechLLM. We keep the head a single linear layer, trading little accuracy for interpretability: each emotion becomes one direction in the LLM output token space, revealing associated tokens. On IEMOCAP, across two speechLLM architectures, it improves Macro F1 and removes hallucinations, with largest gains on realistic ASR transcripts. Our analy
    
[^39]: 边际效用、矩阵分解与键值（KV）缓存：面向主权地理采矿推理的统一信息经济学框架

    Marginal utility, matrix factorization, and the Key-Value (KV) cache: a unified information-economic framework for sovereign geo-mining inference

    [https://arxiv.org/abs/2609.20068](https://arxiv.org/abs/2609.20068)

    本文提出一个统一的信息经济学框架，证明边际效用、矩阵分解与KV缓存压缩三者遵循同一条分配规则（保留特征值超过约束影子价格的最高维度），并将其应用于地理采矿文档的结构化信息自动抽取。

    

    本文在经济学中的边际效用概念与两种机器学习构造（矩阵分解和Transformer语言模型的键值缓存）之间架起了理论桥梁。论文证明：评分矩阵的奇异值谱是潜在因子的边际效用递减曲线，投影协方差算子的特征值谱是模型习得表示的边际效用曲线，而缓存逐出与低秩缓存压缩则是在内存预算约束下进行效用最大化的实例。三者可归结为同一条分配规则：保留那些特征值超过约束条件影子价格的最高维度。该框架被应用于从地理采矿文档中自动抽取结构化信息，由此引出了多轮推理协议、逐层TIES模型合并程序以及一种选择策略。

    arXiv:2609.20068v1 Announce Type: new  Abstract: This paper builds a theoretical bridge between the economic notion of marginal utility and two machine-learning constructs, matrix factorization and the Key--Value cache of transformer language models. The singular value spectrum of a rating matrix is shown to be a diminishing marginal utility schedule for latent factors, the eigenvalue spectrum of the projected covariance operator to be the marginal utility schedule of a model's learned representation, and cache eviction and low-rank cache compression to be instances of constrained utility maximization under a memory budget. The three collapse into a single allocation rule: retain the top dimensions whose eigenvalue exceeds the shadow price of the binding constraint. The framework is applied to the automated extraction of structured information from geo-mining documents, where it motivates a multi-pass inference protocol, a layer-wise TIES model merging procedure, and a selection policy
    
[^40]: AI 应当促进大规模的民主审议

    AI Should Facilitate Democratic Deliberation at Scale

    [https://arxiv.org/abs/2609.20059](https://arxiv.org/abs/2609.20059)

    本立场论文主张 AI 应当在保留人类能动性、鼓励相互尊重、促进平等包容、增强而非取代公民参与四项原则下辅助大规模民主审议，而非以机器判断替代人类选择。

    

    AI 系统可以通过支持大规模审议来强化民主，通过解决认知、社会、平台设计和市场驱动的摩擦，同时保留人类能动性。与流动民主等通过投票委托来重构代议制的提议不同，在这篇立场论文中，我们认为 AI 辅助审议提供了一条更有前景的路径：通过降低有意义参与的门槛，而不是用机器判断替代人类选择。基于在线审议平台和实验研究的证据，我们提出了四项指导原则：保留能动性与自主性、鼓励相互尊重、促进平等与包容，以及增强而非取代积极的公民参与。我们还讨论了关键挑战，包括对齐、谄媚、训练偏见以及对 AI 系统的过度依赖。我们呼吁机器学习社区开发以审议为中心的 AI 系统。

    arXiv:2609.20059v1 Announce Type: cross  Abstract: AI systems can strengthen democracy by supporting deliberation at scale by addressing cognitive, social, platform-design, and market-driven frictions, while preserving human agency. Unlike proposals such as liquid democracy that restructure representation through vote delegation, in this position paper, we argue that AI-assisted deliberation offers a more promising path by lowering barriers to meaningful engagement without substituting machine judgment for human choice. Drawing on evidence from online deliberation platforms and experimental research, we identify four guiding principles: preserving agency and autonomy, encouraging mutual respect, promoting equality and inclusiveness, and augmenting rather than substituting active citizenship. We also address critical challenges, including alignment, sycophancy, training bias, and over-reliance on AI systems. We call on the machine learning community to develop deliberation-focused AI sy
    
[^41]: 缺失的补充：面向编码代理的状态条件化最小充分证据

    The Missing Complement: State-Conditioned Minimal Sufficient Evidence for Coding Agents

    [https://arxiv.org/abs/2609.20050](https://arxiv.org/abs/2609.20050)

    该论文提出了状态条件化最小充分证据恢复这一新问题并构建了SERBench基准，同时提出MSS-Complement方法，将证据获取从排序转变为集合构建，为编码代理的决策恢复紧凑且充分的证据组合。

    

    一个处理问题进行到一半的编码代理已经读过检索器排名最高的很多内容。相关性是按段落评分的，但充分性属于集合层面：一个排序器可能用某个所需事实的多个变体填满其预算，却使决策仍然缺乏支持。我们提出了状态条件化最小充分证据恢复问题：给定一个捕获的代理状态，恢复一个紧凑的证据组合，以提供其下一个决策仍然缺乏的支持。SERBench在来自45个代码仓库的500个保留状态上对此进行测量，记录代理已经看到的内容，并且只对覆盖当前决策被标注所需的每一个事实的集合给予认可。MSS-Complement将证据获取视为集合构建而非排序。三次语义调用提出一个联合充分的集合，搜索其缺失的内容，并在6,144个token内返回4-8个完整的源单元。一个仅在校准数据上固定的配置，为其中73.0%的状态恢复了完整的证据集合。

    arXiv:2609.20050v1 Announce Type: cross  Abstract: A coding agent halfway through an issue has already read much of what a retriever ranks highest. Relevance is scored per passage, but sufficiency belongs to the set: a ranker can fill its budget with variants of one required fact and leave the decision unsupported. We formulate state-conditioned minimal sufficient evidence recovery: given a captured agent state, recover a compact evidence combination that supplies the support its next decision still lacks. SERBench measures this on 500 held-out states from 45 repositories, recording what the agent has seen and crediting only sets that cover every fact the current decision was annotated to require. MSS-Complement treats acquisition as set construction, not ranking. Three semantic calls propose a jointly sufficient set, search for what it lacks, and return 4-8 intact source units within 6,144 tokens. One configuration, fixed on calibration data, recovers a complete set for 73.0% of those
    
[^42]: 大型语言模型中跨语言的地缘政治分歧

    Geopolitical Divisions Across Languages in Large Language Models

    [https://arxiv.org/abs/2609.20005](https://arxiv.org/abs/2609.20005)

    该研究通过对GPT、Claude和Gemini进行112种语言、共67,200次响应的大规模实验，首次发现大型语言模型对乌克兰战争的评估随提问语言而显著变化，且各语言间的回复倾向分布与全球地缘政治分歧格局（公众对俄态度、联合国投票及对乌援助）高度吻合。

    

    人们越来越多地求助于AI聊天机器人来获取新闻和了解世界事件。但当人们用不同语言提问时，是否会得到相同的政治性回答？本研究表明，提问所用的语言可以改变同一个AI系统对乌克兰战争的评估。我们让GPT、Claude和Gemini以112种语言对二十个关于这场战争的陈述进行评估，共收集了67,200个回复。偏向俄罗斯与偏向乌克兰的回复之间的平衡在不同语言之间存在差异。当我们按各国官方语言对回复进行分组时，其呈现出一种类似于全球政治分歧格局的模式：相对更多偏向俄罗斯的答案，对应着公众对俄罗斯更积极的看法、在联合国投票中对乌克兰较少的支持，以及对乌克兰较少的援助。这一总体模式在所有三个模型中均反复出现，且在剔除个别陈述配对后依然保持。我们的发现揭示了信息战可能通过的一条潜在途径。

    arXiv:2609.20005v1 Announce Type: new  Abstract: People increasingly turn to AI chatbots for news and explanations of world events. But do they receive the same political answers when they ask in different languages? Here we show that the language of a question can change how the same AI systems assess the war in Ukraine. We ask GPT, Claude and Gemini to evaluate twenty statements about the war in 112 languages, collecting 67,200 responses. The balance between Russia-leaning and Ukraine-leaning responses differs across languages. When we group responses by countries' official languages, they follow a pattern resembling worldwide political divisions: relatively more Russia-leaning answers correspond to more favourable public views of Russia, less support for Ukraine in United Nations votes, and less aid to Ukraine. The broad pattern recurs across all three models and remains when individual statement pairs are removed. Our findings suggest a possible route through which information warf
    
[^43]: 大语言模型对中国人工智能生成内容法规合规性的基准测试

    Benchmarking LLM Compliance with China AI Generated Content Regulations

    [https://arxiv.org/abs/2609.19989](https://arxiv.org/abs/2609.19989)

    本文设计了一个包含六个维度、2303个问题（含203个自建宪法问题）的中文合规评估框架，对20个大语言模型在中国AI生成内容法规下的合规性进行了基准测试，发现国际模型同样表现出较高合规性，主要差异集中在意识形态对齐相关维度。

    

    大语言模型（LLM）的广泛应用导致内容合规风险不断升级。先前的研究主要致力于解决英语语境下的这些风险，而低估了中文语言内容的复杂性。本文遵循中国现行的人工智能生成内容合规要求，对20个知名大语言模型提供了评估结果，为中国的监管环境提供了深入见解。我们设计了一个新颖的框架来评估合规性和拒绝率，该框架包含跨越六个不同维度的2303个问题，其中包括203个自行构建的宪法相关问题。该框架采用多个评审员基于其分层对齐记忆独立生成裁决。我们的研究结果表明，即使使用标准中文问题，国际模型也表现出较高的合规水平，主要差异可能源于与意识形态对齐密切相关的维度。我们建立了一个监管基准……

    arXiv:2609.19989v1 Announce Type: new  Abstract: The widespread adoption of LLMs has led to escalating content compliance risks. Prior works have contributed to addressing these risks in the English context, downplaying the complexity of Chinese language content. This paper follows China's current AI-Generated content compliance requirements and provides evaluation results on 20 notable LLMs, offering insight into China's regulatory landscape. We design a novel framework to assess the compliance and refusal rates with 2303 questions spanning six distinct dimensions, including 203 self-constructed constitutional questions. The framework employs several judges to generate verdicts independently based on their hierarchical alignment memory. Our findings show that international models also exhibit high levels of compliance despite the use of standard Chinese questions, and the main differences may stem from dimensions closely related to ideological alignment. We establish a regulatory benc
    
[^44]: DeepSeek-V4.1-Flash：突破KV缓存压缩的极限

    DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression

    [https://arxiv.org/abs/2609.19969](https://arxiv.org/abs/2609.19969)

    DeepSeek-V4.1-Flash通过因果编码器-解码器架构和跨层KV缓存复用等KV缓存压缩技术，在支持百万token上下文的同时，将预填充阶段激活参数量减半至80亿，大幅降低了长时程智能体工作负载的计算、存储和带宽成本。

    

    长时程智能体的广泛采用使模型工作负载日益偏向输入密集型。尽管先前的工作已大幅降低了长上下文计算的成本，但预填充在计算上仍然代价高昂，且庞大的KV缓存持续对HBM和SSD的容量以及数据传输带宽造成压力。这些计算、存储和带宽需求共同构成了进一步降低部署成本的主要瓶颈。为应对这一挑战，我们推出了DeepSeek-V4.1-Flash，这是一个多模态混合专家模型，拥有5520亿骨干参数，支持长达一百万token的上下文。凭借其因果编码器-解码器（CED）架构，该模型在解码阶段每个token激活160亿参数，而在预填充阶段仅激活80亿参数，显著提升了智能体工作负载的成本效率。为了突破KV缓存压缩的极限，DeepSeek-V4.1-Flash结合了跨层KV缓存复用……

    arXiv:2609.19969v1 Announce Type: new  Abstract: The widespread adoption of long-horizon agents has made model workloads increasingly input-heavy. Although prior work has substantially reduced the cost of long-context computation, prefill remains computationally expensive, and large KV caches continue to strain HBM and SSD capacity and data-transfer bandwidth. Together, these compute, storage, and bandwidth demands constitute the primary bottleneck to further lowering deployment costs. To address this challenge, we introduce DeepSeek-V4.1-Flash, a multimodal Mixture-of-Experts (MoE) model with 552B backbone parameters and support for contexts of up to one million tokens. With its Causal Encoder-Decoder (CED) architecture, the model activates 16B parameters per token during decode but only 8B parameters during prefill, substantially improving cost efficiency for agentic workloads. To push the limits of KV cache compression, DeepSeek-V4.1-Flash combines cross-layer KV cache reuse in Comp
    
[^45]: 逮捕之前：基于不完整证据的犯罪侧写大语言模型基准测试

    Before the Arrest: Benchmarking LLMs on Criminal Profiling from Incomplete Evidence

    [https://arxiv.org/abs/2609.19965](https://arxiv.org/abs/2609.19965)

    该论文提出了包含五个国家2500个真实凶杀案例的PIJ基准，首次评估大语言模型在逮捕前阶段基于不完整证据进行犯罪侧写、犯罪过程重建和刑罚预测的能力，发现模型在从显性事实提取过渡到隐性推理时性能系统性下降。

    

    大语言模型（LLM）正日益被应用于法律和刑事司法任务，然而现有工作几乎完全聚焦于嫌疑人身份已知的逮捕后场景，而“从不完整证据中推断嫌疑人特征”这一逮捕前的关键挑战在很大程度上尚未被探索。为填补这一空白，我们引入了犯罪侧写、侦查与审判（PIJ）基准，其中包含来自五个国家的2,500个真实凶杀案例。PIJ在贯穿整个刑事侦查流程的三项任务上评估大语言模型：犯罪侧写任务，需要溯因推理从碎片化的现场证据中推断嫌疑人特征；犯罪过程重建任务，测试结构化信息提取能力；以及刑罚预测任务，需要法律演绎推理能力。我们评估了9个强大的大语言模型，发现随着任务从显性事实提取转向隐性推理，模型性能呈现系统性下降。

    arXiv:2609.19965v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly applied to legal and criminal justice tasks, yet existing work focuses almost exclusively on post-arrest scenarios where the suspect's identity is already known, leaving the critical pre-arrest challenge of inferring suspect characteristics from incomplete evidence largely unexplored. To fill this gap, we introduce the Profiling, Investigation, and Judgment (PIJ), comprising 2,500 real homicide cases from five countries. PIJ evaluates LLMs across three tasks that span the entire criminal investigation pipeline: criminal profiling, which requires abductive reasoning to infer suspect attributes from fragmentary scene evidence, crime process reconstruction, which tests structured information extraction, and sentence prediction, which demands legal deductive reasoning. We evaluate 9 powerful LLMs and find that performance degrades systematically as tasks shift from explicit fact extraction to imp
    
[^46]: 检索主导的抽取式问答中的内在序列似然置信度：两个预先设定的否定结果，及其能归因与不能归因的对象

    Intrinsic Sequence-Likelihood Confidence in Retrieval-Dominated Extractive QA: Two Pre-Specified Negatives, and What They Do and Do Not Attribute

    [https://arxiv.org/abs/2609.19942](https://arxiv.org/abs/2609.19942)

    该研究通过预先注册的评估标准证明，在检索已能恢复92-99.8%最优性能的抽取式问答场景中，模型的内在序列似然置信度无论作为蒸馏触发器还是路由弃答策略的控制信号均告失效。

    

    在抽取式文档问答中——其问题由包含答案的段落生成，因此检索即可恢复任何模式组合所能达到性能的92-99.8%（无论其绝对准确率如何）——基于置信度的机制几乎没有提升空间。在专业领域语料库上微调开源语言模型后，模型自身的置信度是一个颇具吸引力的控制信号：它可用于决定哪些查询值得进一步适配、哪些答案值得信赖。我们在实验开始前预先固定的评估标准下，对四个7-9B模型家族进行了评估（其领域适配使闭卷F1最多提升+0.03），结果两种用途均告失败：在预先设定的三步迁移预算下，蒸馏触发器在全部四个模型家族上失效，单模型试点中的路由与弃答策略同样失败。在我们测试的每一个正确性标准下，仅检索即可恢复最优组合准确率的92-99.8%，留下的……

    arXiv:2609.19942v1 Announce Type: new  Abstract: In extractive document question answering whose questions were generated from the passages that contain their answers -- so that retrieval recovers 92-99.8% of what any mode combination could reach, whatever its absolute accuracy -- confidence-driven mechanisms have little to gain. Fine-tuning an open language model on a specialized domain corpus yields a model whose own confidence is a tempting control signal: it could decide which queries warrant further adaptation, and which answers to trust. We evaluate both uses under criteria fixed before the runs were executed, across four 7-9B model families whose adaptation moved closed-book F1 by at most +0.03, and both fail: a distillation trigger on all four families, under its pre-specified three-step transfer budget, and a routing-and-abstention policy in its single-model pilot. Retrieval alone recovers 92-99.8% of best-case combined accuracy under every correctness criterion we test, leavi
    
[^47]: KoNeoBench：一个用于评估大语言模型理解韩语新词的精选评测数据集

    KoNeoBench: A Curated Evaluation Dataset for LLM Understanding of Korean Neologisms

    [https://arxiv.org/abs/2609.19916](https://arxiv.org/abs/2609.19916)

    该论文提出了KoNeoBench，一个基于2020年以来在线新闻中1,785个经专家审校的韩语新词构建的评测基准，通过四个任务评估大语言模型对韩语新词的理解能力，弥补了现有静态基准对新兴词汇变化覆盖不足的缺陷。

    

    大语言模型（LLM）通常在静态基准上进行评估，然而自然语言会不断通过新出现的词汇和语义而演变。现有的韩语基准主要围绕已确立的词汇，因此对这类近期词汇变化的覆盖有限，且其面向英语的设计使其难以评估韩语的类型学特征——在韩语中，实词能与功能词素进行能产性组合。在本文中，我们提出了KoNeoBench，一个用于评估大语言模型对韩语新词理解能力的基准。KoNeoBench基于自2020年以来在线新闻中出现的1,785个韩语新词构建，并经过专家词典学审校。每个词条都提供用法示例、构词分析和词典式定义。基于这一资源，我们定义了四个任务，并报告了近期模型的结果以及人类基线水平。我们的实验表明……

    arXiv:2609.19916v1 Announce Type: cross  Abstract: Large language models (LLMs) are typically evaluated on static benchmarks, even though natural language constantly evolves through newly emerging words and meanings. Existing Korean benchmarks are centered on established vocabulary and therefore provide limited coverage of such recent lexical change, and their English-oriented design makes it difficult to assess the typological properties of Korean, in which content words combine productively with functional morphemes. In this paper, we introduce KoNeoBench, a benchmark for evaluating LLMs' understanding of Korean neologisms. KoNeoBench is built on 1,785 Korean neologisms attested in online news since 2020 and curated through expert lexicographic review. Each entry provides usage examples, word-formation analyses, and dictionary-style definitions. Based on this resource, we define four tasks and report results on recent models, together with a human baseline. Our experiments show that 
    
[^48]: Transformer模型中通过词汇抽象实现的泛化：以功能词为例

    Generalization through Lexical Abstraction in Transformer Models: The Case of Functional Words

    [https://arxiv.org/abs/2609.19887](https://arxiv.org/abs/2609.19887)

    本文探究预训练Transformer模型能否像人类一样利用代词、副词等功能词进行词汇抽象，通过在嵌入空间中比较名词与其可替代代词的表示，检验模型是否能识别词汇化句子与功能化句子之间的句法和语义平行关系。

    

    代词、副词和其他功能词（如they、her、somewhere、there）在语言中常被用来替代具体的名词或短语，此时它们的属性——如性、语法数——为给定语境提供了足够的信息。预训练的Transformer模型是否以某种方式对这类功能词进行编码，使其能够像人类一样使用这些词？语言模型能否识别诸如"The researchers wrote the paper"与"They wrote it"这类句子之间的句法和语义平行关系，而这种平行关系正是依赖于这种词汇抽象？我们将这些语言学问题映射到预训练Transformer模型的嵌入空间中，比较名词的表示与可替代这些名词的代词和副词的表示，涵盖孤立状态以及平行的词汇化句子和功能化句子中的情况。随后，我们探测平行词汇化句子嵌入中所共享的句法和语义结构。

    arXiv:2609.19887v1 Announce Type: new  Abstract: Pronouns, adverbs and other functional words (such as they, her, somewhere, there) are often used in language to replace concrete nouns or phrases, when their properties - such as gender, grammatical number - provide sufficient information for the given context. Do pretrained transformer models encode such functional words in a manner that allows them to be used like humans do? Can language models recognize the syntactic and semantic parallelism of sentences such as "The researchers wrote the paper" and "They wrote it", which relies on such lexical abstraction?   We map these linguistic questions into the embedding space of a pretrained transformer model, and compare representations of nouns, with the representations of the pronouns and adverbs that can replace these nouns, in isolation and in parallel lexicalized and functional sentences. We then probe for shared syntactic and semantic structure in the embeddings of parallel lexicalized
    
[^49]: 评估机器翻译对话中的交际成功度

    Evaluating Communicative Success in Machine-Translated Conversation

    [https://arxiv.org/abs/2609.19885](https://arxiv.org/abs/2609.19885)

    该论文提出了一个可复用的三层评估框架，从语义、语用和文化-社会三个维度衡量机器翻译口译智能体的对话交际成功度，突破了传统只测句子忠实度的评估局限。

    

    基于机器翻译（MT）构建的口译智能体越来越多地介入不共享语言的人们之间的实时对话，然而我们仍然使用为孤立句子设计的指标来评估它们，这些指标衡量的是忠实度，而非交际是否成功。我们引入了一个可复用的三层检查表-评判框架，从语义、语用和文化-社会维度评估口译中介的对话，涵盖忠实度指标未能衡量的自然性、意图和社会适宜性。该框架既可在单轮设置中运行，也可在交互式多轮设置中运行，在多轮设置中，模拟用户会随着对话的展开回复翻译后的消息，并且每一轮都与对话整体一同被评分。我们通过受控扰动、跨评判者比较和人工标注对该框架进行了广泛验证。我们的主要单轮基准测试评估了横跨阿拉伯语、孟加拉语等语言的10种口译设置，……

    arXiv:2609.19885v1 Announce Type: new  Abstract: Interpreter agents built on machine translation (MT) increasingly mediate live conversation between people who do not share a language, yet we still evaluate them with metrics built for isolated sentences, which measure fidelity rather than whether communication succeeds. We introduce a reusable three-layer checklist-and-judge framework that evaluates interpreter-mediated conversation across semantic, pragmatic, and cultural-social dimensions, covering the naturalness, intent, and social appropriateness that fidelity metrics leave unmeasured. It runs in both single-turn and interactive multi-turn settings, where simulated users reply to translated messages as the conversation unfolds and each turn is scored alongside the conversation as a whole. We extensively validate it through controlled perturbations, cross-judge comparisons, and human annotations. Our main single-turn benchmark evaluates 10 interpreter setups across Arabic, Bengali,
    
[^50]: PetriBench：针对动态状态空间的大语言模型推理基准测试

    PetriBench: Benchmarking LLM Reasoning over Dynamic State Spaces

    [https://arxiv.org/abs/2609.19883](https://arxiv.org/abs/2609.19883)

    本文提出PetriBench，一个基于Petri网的紧凑、自包含且可扩展的基准测试，用于评估大语言模型在动态状态空间上的推理能力，发现模型准确率随任务难度增加而一致下降，且测试时计算对不同推理任务的提升效果各异。

    

    表征大语言模型的推理能力仍然是一个开放性挑战，因为许多现有的基准测试往往隔离特定的推理技能、依赖外部知识，或者扩展成本高昂。我们提出了PetriBench，这是一个紧凑、完全自包含且可扩展的基准测试，利用Petri网——一种用于建模真实世界并发与分布式系统的成熟形式化方法——来评估大语言模型在动态状态空间上的推理能力。PetriBench按范围和时间跨度将推理组织为四个任务族，并通过增加结构复杂度生成简单、中等和困难三个级别，同时与精确的真实标准进行评估。在多样化的专有模型和开源权重模型中，准确率随难度增加而持续下降，而更困难的实例则暴露出愈发明显的任务特定能力差异。额外的分析表明，测试时计算能够提升性能，但其与不同推理任务的交互方式各不相同。

    arXiv:2609.19883v1 Announce Type: cross  Abstract: Characterizing LLM reasoning remains an open challenge, as many existing benchmarks isolate specific reasoning skills, rely on external knowledge, or are costly to extend. We introduce PetriBench, a compact, fully self-contained, and scalable benchmark for evaluating LLM reasoning over dynamic state spaces using Petri nets, a mature formalism for modeling real-world concurrent and distributed systems. PetriBench organizes reasoning into four task families varying by scope and temporal horizon, with Easy, Medium, and Hard levels generated by increasing structural complexity and evaluated against exact ground truth. Across a diverse set of proprietary and open-weight models, accuracy decreases consistently with difficulty, while harder instances expose increasingly distinct task-specific capability profiles. Additional analyses show that test-time compute improves performance but interacts differently with different reasoning tasks, and 
    
[^51]: D-Quant：用于KV缓存量化的可漂移熵编码方法

    D-Quant: Driftable Entropy Coding for KV Cache Quantization

    [https://arxiv.org/abs/2609.19880](https://arxiv.org/abs/2609.19880)

    本文提出D-Quant方法，利用KV缓存值近似正态分布的特性，通过可漂移熵编码突破固定宽度量化级别数的指数限制，在有效压缩KV缓存内存占用的同时保持模型性能。

    

    arXiv:2609.19880v1 公告类型：新论文 摘要：KV缓存已成为部署大语言模型（LLM）的主要瓶颈，因为其内存占用随序列长度和批处理大小线性增长，对内存容量和带宽都造成了巨大压力。在各种KV缓存压缩技术中，量化因其高效性和易于部署而特别具有吸引力。然而，大多数现有方法依赖于固定宽度量化，其中b位表示本质上仅限于2^b个量化级别。随着位宽的减小，可用量化级别的数量呈指数级缩减，导致严重的信息丢失和性能快速下降。我们进一步观察到，固定宽度量化未能充分利用KV缓存的高度非均匀分布特性。经过旋转和归一化处理后，KV值近似服从正态分布，大多数值集中在中心附近，只有一小部分出现在尾部。

    arXiv:2609.19880v1 Announce Type: new  Abstract: The KV cache has become a major bottleneck in deploying LLMs, as its memory footprint grows linearly with sequence length and batch size, imposing substantial pressure on both memory capacity and bandwidth. Among various KV cache compression techniques, quantization is particularly attractive due to its effectiveness and ease of deployment. However, most existing methods rely on fixed-width quantization, where a $b$ bit representation is inherently limited to $2^b$ quantization levels. As the bit width decreases, the number of available levels shrinks exponentially, leading to severe information loss and rapid performance degradation. We further observe that fixed-width quantization fails to exploit the highly non-uniform distribution of KV cache. After rotation and normalization, KV values approximately follow a normal distribution, with most values concentrated near the center and only a small fraction appearing in the tails. Neverthel
    
[^52]: VākQA：泰卢固语口语事实型问答的基准与评估研究

    V\={a}kQA: A Benchmark and Evaluation Study for Telugu Spoken Factoid Question Answering

    [https://arxiv.org/abs/2609.19879](https://arxiv.org/abs/2609.19879)

    本文提出了首个泰卢固语口语事实型问答基准VākQA（包含2,001个问答对、语音音频及人工验证答案），并验证了自动评估方法的可靠性，系统评测了各类模型在不同模态、语言和领域下的表现。

    

    问答系统随着大语言模型的发展而迅速进步，但主要集中在高资源语言上，涵盖文本和口语场景。针对泰卢固语的口语问答（SQA）基准至今尚未被探索，且自动评估在这种场景下的可靠性仍未被量化。我们推出了VākQA，这是一个泰卢固语口语问答基准，包含跨六个领域的2,001个事实型问答对，配有2.53小时的语音音频、双语转录文本以及经人工验证的参考答案。我们首先将评估方法与人类判断进行对比验证：以Gemini作为评判模型最接近人类评分，但其严格程度并不均匀，而开源权重的评判模型则系统性地惩罚那些在表面形式上与参考答案不同但内容正确的泰卢固语答案。利用这一经过验证的评估设置，我们在输入模态、语言和领域维度上对专有模型和开源权重模型进行了基准测试。我们观察到泰卢固语的表述保留了文化特异性……

    arXiv:2609.19879v1 Announce Type: new  Abstract: Question answering has advanced rapidly with large language models, but predominantly for high-resource languages, in both text and spoken settings. Spoken question answering (SQA) benchmark for Telugu remains unexplored, and the reliability of automatic evaluation in this setting remains unquantified. We introduce V\={a}kQA, a Telugu SQA benchmark of 2,001 factoid question-answer pairs across six domains, with 2.53 hours of speech audio, bilingual transcriptions, and human-verified reference answers. We first validate evaluation methods against human judgements: Gemini-as-a-judge best approximates human ratings but is non-uniformly strict, while open-weight judges systematically penalize correct Telugu answers that differ in surface form from the reference. Using this validated setup, we benchmark proprietary and open-weight models across input modality, language, and domain. We observe that Telugu phrasing retains cultural specificity 
    
[^53]: Uni-LaDiR：潜在扩散统一多模态推理

    Uni-LaDiR: Latent Diffusion Unifies Multimodal Reasoning

    [https://arxiv.org/abs/2609.19878](https://arxiv.org/abs/2609.19878)

    Uni-LaDiR提出将多模态推理步骤映射到共享潜在空间，并利用扩散模型预测下一块思维标记，从而在统一的潜在表示中实现灵活的多模态推理。

    

    多模态推理要求模型在整个推理过程中利用来自多种模态的信息。然而，现有方法通常将特定模态的思维标记拼接在单一序列中，使得模型在跨模态推理时需要自行弥合表示上的差异。我们提出了Uni-LaDiR（统一潜在扩散推理器），这是一个将这些思维引入共享潜在空间进行推理的框架。统一编码器将来自不同模态的教师推理步骤映射为共享的思维标记，并通过训练来保留后续推理步骤及最终答案或动作所需的信息。由于相同的上下文可以支持多个有效的下一步推理，我们使用扩散模型基于输入和先前块来预测下一块思维标记。通过共享模型权重联合训练编码器和扩散推理器，促使思维标记既对任务有用，又……

    arXiv:2609.19878v1 Announce Type: cross  Abstract: Multimodal reasoning requires models to draw on information from multiple modalities throughout the reasoning process. Yet existing methods often concatenate modality-specific thought tokens in a single sequence, leaving the model to bridge representational differences as it reasons across modalities. We introduce Uni-LaDiR (Unified Latent Diffusion Reasoner), a framework that brings these thoughts into a shared latent space for reasoning. A unified encoder maps teacher reasoning steps from different modalities into shared thought tokens, trained to preserve the information needed for later reasoning steps and the final answer or action. Because the same context can support multiple valid next steps, we use diffusion to predict the next block of thought tokens from the input and preceding blocks. Jointly training the encoder and diffusion reasoner with shared model weights encourages thought tokens to be both useful for the task and pr
    
[^54]: JustMem：面向长期对话的“刚刚好”内存访问

    JustMem: Just-Enough Memory Access for Long-Term Conversations

    [https://arxiv.org/abs/2609.19877](https://arxiv.org/abs/2609.19877)

    该论文提出JustMem框架，将对话历史存储为紧凑的原子记忆，并针对每个查询在发现广度和阅读保真度两个维度上自适应地调整记忆访问策略（LOOKUP、COMPOSE、REPLAY），从而在长期对话中实现恰到好处的高效证据检索。

    

    高效的长期对话记忆需要检索到足够的证据，同时避免不加区分地扩大呈现给语言模型的上下文。这具有挑战性，因为相关证据可能分布在多个会话中，而压缩又可能丢弃回答所需的细节。因此，不同的查询需要不同形式的记忆访问方式。为了刻画这些需求，我们从两个维度对记忆访问进行形式化：发现广度，控制证据搜索的范围有多广；阅读保真度，控制证据是以紧凑形式读取还是从原始对话中恢复。基于这一形式化，我们提出了JustMem，它将对话历史存储为紧凑的原子记忆，并针对每个查询在这两个维度上自适应地调整记忆访问。具体而言，LOOKUP处理局部证据，COMPOSE扩大对分布式证据的发现范围，REPLAY提高阅读保真度……

    arXiv:2609.19877v1 Announce Type: new  Abstract: Efficient long-term conversational memory requires retrieving sufficient evidence without indiscriminately expanding the context presented to the language model. This is challenging because relevant evidence may be distributed across multiple sessions, while compression may discard details needed for answering. Different queries therefore require different forms of memory access. To capture these demands, we formulate memory access along two dimensions: discovery breadth, which controls how broadly evidence is searched, and reading fidelity, which controls whether evidence is read in compact form or recovered from the original conversation. Based on this formulation, we introduce JustMem, which stores conversation history as compact atomic memories and adapts memory access along these two dimensions to each query. Specifically, LOOKUP handles local evidence, COMPOSE broadens discovery for distributed evidence, and REPLAY increases readin
    
[^55]: Zarya：一种具有灵活训练与双模式推理的混合自回归-掩码扩散语言模型

    Zarya: A Hybrid Autoregressive--Masked Diffusion Language Model with Flexible Training and Dual-Mode Inference

    [https://arxiv.org/abs/2609.19868](https://arxiv.org/abs/2609.19868)

    Zarya提出了在单一架构中联合优化自回归与掩码扩散目标的混合语言模型家族，通过可变槽位大小的课程训练实现从细粒度AR学习到粗粒度扩散学习的平滑过渡，并支持MDM采样和槽位化投机解码两种解码范式。

    

    自回归语言模型（ARMs）受限于从左到右的顺序生成方式，而掩码扩散模型（MDMs）虽然能够实现并行解码，但由于无法复用键值缓存（KV cache）而导致计算开销过高，并且由于需要在难以处理的词元组合空间上学习依赖关系而产生不连贯的生成结果。我们提出了Zarya，这是一系列在单一架构中联合优化自回归（AR）目标和掩码扩散目标的混合语言模型。Zarya将训练数据结构化为可变大小的槽位，并采用一种逐渐增加槽位粒度的课程学习策略，实现了从细粒度AR学习到粗粒度扩散学习的平滑过渡。在推理阶段，Zarya通过统一接口提供两种不同的解码范式：(i) 具有首次命中去噪的MDM采样，以及(ii) 交替进行两种模式的槽位化投机解码……

    arXiv:2609.19868v1 Announce Type: cross  Abstract: Autoregressive language models (ARMs) are constrained by sequential, left-to-right generation, while masked diffusion models (MDMs) enable parallel decoding but suffer from high computational overhead due to the inability to reuse Key-Value (KV) cache and from incoherent generation arising from learning dependencies over an intractable space of token combinations. We introduce Zarya, a family of hybrid language models that jointly optimizes an autoregressive (AR) objective and a masked-diffusion objective within a single architecture. Zarya structures training data into variable-size slots and employs a curriculum that gradually increases slot granularity, enabling a smooth transition from fine-grained AR learning to coarse-grained diffusion learning. At inference, Zarya provides two distinct decoding paradigms through a unified interface: (i) MDM sampling with first-hitting denoising, and (ii) slotted speculative decoding that interle
    
[^56]: 可复现性不等于构念效度：对制度情境化传播的大语言模型测量

    Reproducibility is not construct validity: LLM measurement of institutionally situated communication

    [https://arxiv.org/abs/2609.19866](https://arxiv.org/abs/2609.19866)

    该研究利用欧盟《人工智能法案》咨询数据证明，大语言模型标注的高可复现性并不等于构念效度，且基于文本的测量与问卷测量之间的分歧在不同利益相关方群体间存在系统性差异。

    

    高标注可复现性并不一定意味着大语言模型推断的测量指标能够捕捉其旨在测量的构念。我们使用来自欧盟委员会《人工智能法案》公众咨询的数据集检验了这一区别，将结构化问卷回答与同一利益相关方提交的自由文本咨询意见相关联。大语言模型对咨询意见的标注具有高度可复现性（组内相关系数 > 0.99），但与其名义上旨在近似的构念的问卷测量结果仅表现出有限的收敛性。问卷测量与大语言模型推断的基于文本的测量之间的分歧在不同利益相关方群体中呈现系统性差异：商业协会在基于文本的咨询中表达的对AI风险的担忧高于其在问卷回答中的表达（g = +1.0），而公共当局和若干非商业群体则显示出较小或负向的分歧。分数之间的分歧提示存在正向空间自相关……

    arXiv:2609.19866v1 Announce Type: new  Abstract: High annotation reproducibility does not necessarily imply that an LLM-inferred measure captures the construct it is intended to measure. We test this distinction using a dataset from the European Commission's AI Act consultation, linking structured survey responses to free-text consultation submissions from the same stakeholders. LLM annotations of consultation submissions are highly reproducible (intraclass correlations > 0.99), yet show limited convergence with survey-reported measures of the nominal construct they were intended to approximate. Divergence between survey-and LLM-inferred text-based measures varies systematically across stakeholder groups: business associations express greater concern about AI risks in text-based consultations than in survey responses ({\=g} = +1.0), whereas public authorities and several nonbusiness groups show smaller or negative divergences. Divergences between scores suggest positive spatial autocor
    
[^57]: F$^{2}$DR：面向DeepSearch工作流的细粒度全流程奖励框架

    F$^{2}$DR: A Fine-Grained Full-Pipeline Reward Framework for DeepSearch Workflows

    [https://arxiv.org/abs/2609.19827](https://arxiv.org/abs/2609.19827)

    该论文提出了F2DR框架，从内容、轨迹和答案三个维度对DeepSearch工作流进行细粒度全流程奖励评估，并构建了专门基准DeepSearch RM-Bench，显著提升了评估一致性。

    

    随着大语言模型（LLM）在工业界的广泛部署，DeepSearch已成为解决复杂用户查询的主流范式。它通常通过由规划与反思、信息检索和答案生成组成的迭代闭环工作流来运行。然而，现有的奖励模型（RM）和评估基准主要是为静态单轮任务设计的，无法捕捉DeepSearch工作流的全流程复杂性。为解决这一局限性，我们提出了F2DR，一个细粒度的全流程DeepSearch奖励框架。F2DR从内容、轨迹和答案三个维度对DeepSearch工作流进行评估，实现全面的流程级评估。我们进一步构建了DeepSearch RM-Bench，一个专门用于评估DeepSearch场景中奖励模型的基准。大量实验表明，F2DR实现了显著更高的评估一致性。

    arXiv:2609.19827v1 Announce Type: new  Abstract: With the widespread industrial deployment of Large Language Models (LLMs), DeepSearch has emerged as the dominant paradigm for resolving complex user queries. It typically operates through an iterative closed-loop workflow consisting of planning and reflection, information retrieval, and answer generation. However, existing reward models (RMs) and evaluation benchmarks are primarily designed for static single-turn tasks, failing to capture the full-pipeline complexity of DeepSearch workflows. To address this limitation, we propose F2DR, a fine-grained full-pipeline DeepSearch reward framework. F2DR evaluates DeepSearch workflows across three dimensions: Content, Trajectory, and Answer, enabling comprehensive process-level assessment. We further construct DeepSearch RM-Bench, a dedicated benchmark for evaluating RMs in DeepSearch scenarios. Extensive experiments demonstrate that F2DR achieves significantly higher evaluation consistency th
    
[^58]: 基于大语言模型标注数据的词典约束未分词语言字素到音素转换

    Dictionary-Constrained Grapheme-to-Phoneme for Unsegmented Languages from LLM-Annotated Data

    [https://arxiv.org/abs/2609.19805](https://arxiv.org/abs/2609.19805)

    本文提出一种利用词典构建词格并采用条件随机场评分的上下文感知神经G2P方法，结合大语言模型生成的超过200万条标注数据，显著提升了日语等未分词语言的字素到音素转换性能。

    

    字素到音素（G2P）转换将原始文本转换为其音素形式，是文本转语音（TTS）和自动语音识别（ASR）系统的重要组成部分，要求其快速、稳定且具备上下文感知能力。对于日语等未分词语言，G2P 还需要将词分词与高度依赖上下文的多音字消歧相结合，而准确标注数据的稀缺仍然是一个瓶颈。本文提出了一种上下文感知的神经 G2P 方法，该方法对从词典构建的词格上的判别式条件随机场（CRF）路径进行评分。为解决数据稀缺问题，我们利用大语言模型（LLM）生成了超过200万条句子。实验结果表明，我们的方法显著优于传统的基于形态分析器的方法和神经序列模型。在 Joyo-Kanji-Yomi 基准测试上，我们的方法达到了99.62%的目标词准确率。

    arXiv:2609.19805v1 Announce Type: new  Abstract: Grapheme-to-phoneme (G2P) conversion turns raw text into its phonemic form and is an essential part of both text-to-speech (TTS) and automatic speech recognition (ASR) systems. It is required to be fast, stable and context-aware. For unsegmented languages such as Japanese, G2P additionally couples word segmentation with highly context-dependent polyphone disambiguation, and the scarcity of accurately annotated data remains a bottleneck. In this paper, we present a context-aware neural G2P method that scores paths of a discriminative conditional random field (CRF) over a word lattice constructed from dictionaries. To tackle data scarcity, we utilize large language models (LLMs) to generate more than 2 million sentences. Experimental results demonstrate that our method strongly outperforms conventional morphological analyzer-based methods and neural sequence models. On the Joyo-Kanji-Yomi benchmark, our method reaches 99.62% target word re
    
[^59]: 进化还是错觉？重新思考LLM进化搜索中的评估

    Evolution or Illusion? Rethinking Evaluation in LLM Evolutionary Search

    [https://arxiv.org/abs/2609.19799](https://arxiv.org/abs/2609.19799)

    该论文通过在种子数与迭代数的完整组合网格上系统评估三种LLM进化搜索策略，揭示了固定预算在“宽度”（更多种子）与“深度”（更多迭代）之间的最优分配方式以及策略排名都会随策略、任务和总预算显著变化，证明传统单预算设置下的评估结论并不可靠。

    

    LLM驱动的进化搜索通过启动种子并对每个种子进行迭代来发现程序。现有论文通常只报告单一的预算设置，通常是一个种子运行固定次数的迭代，并仅从这一个数据点对方法进行排名。我们证明这是不够的。我们在五个优化任务上评估了三种进化搜索策略，这些任务是此类论文常用的基准。我们在种子数和迭代数构成的完整网格上进行分析。我们的发现表明，在更多种子（宽度）和更多迭代（深度）之间分配固定预算的最佳方式会随着策略、任务和总预算的变化而变化。此外，我们还观察到策略之间的排名也会随预算变化。在其中一个任务上，单个种子下表现最差的策略在四十个种子下反而最佳；在另一个任务上，最佳迭代次数远低于实践中的常用值，因此额外的迭代深度只会浪费预算，而这些预算若用于更多种子本可转化为分数提升。我们提供了一个……

    arXiv:2609.19799v1 Announce Type: cross  Abstract: LLM-driven evolutionary search finds programs by launching seeds and iterating each one. Papers report a single budget setting, usually one seed run for a fixed number of iterations, and rank methods from that one point. We show this is not enough. We evaluate three evolutionary search strategies on five optimization tasks, commonly used by papers in the genre to report results. We run the analysis over a full grid of seeds and iterations. Our findings suggest that the best way to split a fixed budget between more seeds (width) and more iterations (depth) changes with the strategy, the task, and the total budget. Furthermore, we observe that the ranking of strategies also changes with the budget. On one task the strategy that looks worst at one seed is best at forty seeds. On another the best number of iterations is well below the value common in practice, so extra depth wastes budget that more seeds would turn into score. We provide a
    
[^60]: 《先学习再判断：面向可解释仇恨模因检测的渐进式知识-决策对齐方法》

    Learn Before You Judge: Progressive Knowledge-to-Decision Alignment for Explainable Hateful Meme Detection

    [https://arxiv.org/abs/2609.19778](https://arxiv.org/abs/2609.19778)

    针对现有“先解释后检测”方法中解释生成与标签预测耦合干扰的问题，本文提出渐进式知识-决策对齐方法ProKDA，通过智能体构建外部背景知识并分离任务目标，显著提升可解释仇恨模因检测的性能。

    

    仇恨模因通过图像与文本之间的隐式交互传播辱骂性内容，对在线社区的安全构成严重威胁。近年来，多模态大语言模型被广泛应用于仇恨模因检测，并越来越多地被用于生成可解释的检测结果。然而，我们发现现有的“先解释后检测”方法通常将解释生成与标签预测耦合在同一个训练过程中。这种耦合导致任务目标之间相互干扰，使得检测性能受限，甚至比简单的SFT基线效果更差。为了应对这些挑战，我们提出了ProKDA，一种面向可解释仇恨模因检测的渐进式知识到决策对齐方法。受人类标注训练过程的启发，ProKDA首先利用智能体背景知识构建流水线来获取与模因理解相关的外部知识。

    arXiv:2609.19778v1 Announce Type: new  Abstract: Hateful memes spread abusive content through implicit interactions between images and text, posing serious threats to the safety of online communities. In recent years, multimodal large language models have been widely used for hateful meme detection and are increasingly adopted to generate explainable detection results. However, we find that existing explain-then-detect methods often couple explanation generation and label prediction within the same training process. This coupling causes interference between task objectives, leading to limited detection performance and even worse results than simple SFT baselines. To address these challenges, we propose ProKDA, a progressive knowledge-to-decision alignment method for explainable hateful meme detection. Inspired by the human annotation training process, ProKDA first uses an agentic background knowledge construction pipeline to obtain external knowledge related to meme understanding. It t
    
[^61]: AutoData：面向预训练数据选择的智能体搜索

    AutoData: Agentic Search for Pre-training Data Selection

    [https://arxiv.org/abs/2609.19754](https://arxiv.org/abs/2609.19754)

    AutoData通过智能体在可执行的数据选择算法空间中直接搜索，并利用代理模型的验证反馈迭代改进，仅一夜之间就能自动发现超越人工设计流水线的预训练数据选择算法。

    

    大语言模型智能体最近展现出在执行反馈下通过编辑模型和训练代码来实现机器学习工程自动化的潜力。然而，数据在很大程度上仍处于这一智能体优化循环之外。我们将预训练数据选择问题构建为针对单文档特征的启发式工程，即词汇统计、类别标签和困惑度。我们提出了AutoData，一个直接在可执行选择算法空间中进行搜索的智能体。与以往仅在固定领域集合上优化权重比例的数据配比方法不同，AutoData搜索的是更丰富的程序空间，涵盖评分、分层和随机选择规则，并通过代理模型的验证反馈迭代改进算法，从而自动发现特征之间的交互。在一夜之间的搜索中，AutoData发现了一种优于现有人工设计数据筛选流水线的选择算法。尽管搜索仅在小型代理模型上进行……

    arXiv:2609.19754v1 Announce Type: new  Abstract: LLM agents have recently shown promise in automating machine learning engineering by editing model and training code under execution feedback. Data, however, remains largely outside this agentic optimisation loop. We frame pre-training data selection as heuristic engineering over per-document features, i.e., lexical statistics, categorical labels, and perplexity. We introduce AutoData, an agent that searches directly over executable selection algorithms. Unlike prior data mixture methods that optimise weights over a fixed set of domains, AutoData searches a richer program space of scoring, stratification, and stochastic selection rules, discovering feature interactions automatically by iteratively refining algorithms with validation feedback from a proxy model. Within an overnight search, AutoData discovers a selection algorithm that outperforms existing human-designed curation pipelines. Despite being searched only on this small proxy, 
    
[^62]: 一种音位全面、仅使用ASCII字符的泰语与老挝语罗马化方案：系统性跨语言对应与面向中文用户的设计

    A Phonemically Comprehensive, ASCII-Only Romanization Scheme for Thai and Lao: Systematic Cross-Lingual Correspondence and Chinese-User-Friendly Design

    [https://arxiv.org/abs/2609.19736](https://arxiv.org/abs/2609.19736)

    本文提出了一种仅使用ASCII字符、音位全面的泰语与老挝语统一罗马化方案，该方案保持两种语言间的系统性对应，并通过与拼音的兼容设计照顾中文用户的使用习惯。

    

    本文提出了一种音位全面、仅使用ASCII字符的泰语与老挝语罗马化方案，将这两种密切相关的语言作为统一的跨语言设计问题来处理。该方案能够表示音段对立、元音长短和词汇声调，同时保持“一符号一音位”的透明性以及泰语与老挝语之间的系统性对应。该方案优先考虑共时语音对应，包括在适用情况下与汉语拼音和粤语拼音的对应，同时在不与语音透明性冲突的前提下保留历史音系对应。声调采用紧凑的单数字默认标记法，并辅以可选的调值表示和历史声调类别表示。该方案为语言学习和跨语言语音处理提供了可读性强、键盘友好且机器可处理的音位表示。

    arXiv:2609.19736v1 Announce Type: new  Abstract: This paper proposes a phonemically comprehensive, ASCII-only romanization scheme for Thai and Lao, treating the two closely related languages as a unified cross-lingual design problem. The scheme represents segmental contrasts, vowel length, and lexical tone while maintaining one-symbol-one-phoneme transparency and systematic correspondence between Thai and Lao. The scheme prioritizes synchronic phonetic correspondence, including correspondence with Pinyin and Jyutping where applicable, while preserving historical-phonological correspondence where it does not conflict with phonetic transparency. Tone uses a compact single-digit default notation, supplemented by optional tone-value and historical tone-category representations. The resulting scheme provides a readable, keyboard-friendly, and machine-processable phonemic representation for language learning and cross-lingual speech processing.
    
[^63]: 学会自己的思考：抽象token课程学习

    Learn Your Own Thoughts: Abstract Token Curriculum

    [https://arxiv.org/abs/2609.19717](https://arxiv.org/abs/2609.19717)

    提出了抽象token课程学习（ATC）框架，无需直接监督或手动设计草稿板，即可通过逐步增加问题复杂度训练模型在连续表示空间中自发形成内部抽象思维，并从理论和实验上证明了其相对于以往连续思维训练方法的优势。

    

    大语言模型（LLMs）通过利用思维链（CoT）作为思考中间阶段的草稿板，已经获得了卓越的推理能力。然而，CoT技术需要对思考token进行显式监督，这需要丰富的、特定任务的数据。在这项工作中，我们提出了抽象token课程学习（Abstract Token Curriculum, ATC），这是一种新颖的课程学习框架，能够在没有直接监督或手动草稿板设计的情况下，引出有效的连续中间表示。ATC通过一系列分布逐渐增加问题复杂度，训练模型在连续表示空间中发展出内部的抽象“思维”。本文为ATC的优势及其相对于以往训练连续思维方法的长处提供了理论和实验证据。理论上，我们证明了使用ATC在单层softmax注意力机制下学习奇偶函数时……

    arXiv:2609.19717v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have achieved remarkable reasoning capabilities by utilizing chain-of-thought (CoT) as a scratchpad for intermediate stages of thinking. However, CoT techniques require explicit supervision on thinking tokens, which requires rich, task-specific data. In this work, we propose Abstract Token Curriculum (ATC), a novel curriculum learning framework that elicits effective continuous intermediate representations without direct supervision or manual scratchpad design. ATC gradually increases problem complexity through a sequence of distributions, training the model to develop internal abstract ``thoughts'' in the continuous representation space. This paper provides both theoretical and experimental evidence for the benefits of ATC and its advantages over previous methods for training continuous thoughts. Theoretically, we show that for learning parity functions with single-layer softmax attention using ATC, attent
    
[^64]: 通过结构相似性改进研究论文中序列句子分类的跨语言迁移

    Improving Cross-Lingual Transfer for Sequential Sentence Classification in Research Papers via Structural Similarity

    [https://arxiv.org/abs/2609.19650](https://arxiv.org/abs/2609.19650)

    该论文构建了涵盖13种非英语语言的多语言序列句子分类数据集，并发现利用标签序列和位置规律等在跨语言间保持一致的结构相似性，可以有效提升研究论文序列句子分类的跨语言迁移效果。

    

    arXiv:2609.19650v1 公告类型：new 摘要：序列句子分类（SSC）是科学出版物结构化中的一项重要任务，将SSC研究扩展到英语以外的语言可以提升多语言数字图书馆中科学知识的可访问性。跨语言迁移是解决非英语语言训练数据稀缺问题的一种有前景的方法。先前针对其他自然语言处理任务的研究已经表明，捕捉源语言与目标语言之间的语言相似性是有益的。然而，SSC本质上依赖于话语层面的模式，例如标签序列和位置规律性，无论语言之间存在何种差异，这些模式在各语言中都保持一致。为了探究决定SSC跨语言迁移成功与否的因素，我们构建了一个多语言SSC数据集，涵盖从五个学术数据库收集的13种非英语语言。我们的跨语言迁移实验使用了编码器……

    arXiv:2609.19650v1 Announce Type: new  Abstract: Sequential sentence classification (SSC) is an essential task for structuring scientific publications, and extending SSC research to languages other than English can improve accessibility to scientific knowledge in multilingual digital libraries. Cross-lingual transfer is a promising approach to address the scarcity of training data in non-English languages. Prior work on other natural language processing tasks has shown the benefits of capturing linguistic similarity between source and target languages. However, SSC inherently depends on patterns at the discourse level, such as label sequences and positional regularities, which appear consistently across languages regardless of linguistic differences. To examine the factors that determine transfer success in SSC, we constructed a multilingual SSC dataset covering 13 non-English languages collected from five academic databases. Our cross-lingual transfer experiments, using both encoder-b
    
[^65]: 基于多模态检索增强生成的科学图像质量评估

    Scientific Image Quality Assessment via Multi-modal Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.19634](https://arxiv.org/abs/2609.19634)

    本文提出基于多模态检索增强生成的科学图像质量评估框架，通过多路检索与融合机制增强大语言模型评估复杂科学图像的能力，在ICME 2026 SIQA挑战赛SIQA-U赛道中夺得第一名。

    

    本文提出了一种用于科学图像质量评估的检索增强生成（RAG）框架，旨在同时应对SIQA挑战赛的理解赛道（SIQA-U）和评分赛道（SIQA-S）。我们构建了一个将文本语义与细粒度视觉特征相融合的多模态索引，并开发了一种多路检索与融合机制，为大语言模型提供高度相关的参考案例，从而增强其评估复杂科学图像的能力。实验结果表明，所提出的框架能够有效契合人类专家的评判标准。最终，我们的方法在ICME 2026大挑战赛SIQA挑战赛的SIQA-U赛道中获得第一名。

    arXiv:2609.19634v1 Announce Type: cross  Abstract: This paper proposes a Retrieval-Augmented Generation (RAG) framework for scientific image quality assessment, designed to simultaneously address both the understanding track (SIQA-U) and the scoring track (SIQA-S) of the SIQA challenge. We construct a multimodal index that integrates textual semantics with fine-grained visual features, and develop a multi-route retrieval and fusion mechanism to provide large language models with highly relevant reference cases, thereby enhancing their capability to evaluate complex scientific images. Experimental results demonstrate that the proposed framework effectively aligns with the judgment criteria of human experts. Ultimately, our method achieves 1st place in the SIQA-U track of the SIQA challenge at the ICME 2026 Grand Challenges.
    
[^66]: 从意图到行动：车辆语音指令授权中大语言模型安全性的基准测试

    From Intent to Action: Benchmarking LLM Safety in Vehicle Voice Command Authorization

    [https://arxiv.org/abs/2609.19630](https://arxiv.org/abs/2609.19630)

    该论文首个针对车辆语音指令授权问题提出了包含202个场景、七类行动决策的基准测试，发现大语言模型的决策一致性从40.1%到89.1%不等，其中基于API的模型表现最好且相互之间无显著差异。

    

    大语言模型（LLM）正越来越多地被集成到车辆语音助手中。但将自然语言请求与车辆功能相关联，会带来一个安全关键的授权问题。在执行命令之前，系统必须选择是执行、拒绝、澄清、要求确认、转入手动控制、触发紧急响应，还是不进行任何工具调用。据我们所知，先前的评估并未在说话者角色、身份验证状态、车辆状态和工具可用性等维度上对这种行动前决策进行隔离测试。我们引入了一个包含202个场景的基准测试，并在七类分类体系下提供了参考决策。我们使用决策一致性和安全相关的错误指标，评估了两个本地开源权重模型和三个基于API的大语言模型。决策一致性范围从Llama 3.2 3B的40.1%到Gemini 3.1 Pro Preview的89.1%。基于API的模型得分在83.2%至89.1%之间，且它们之间没有统计学上的显著差异。

    arXiv:2609.19630v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly integrated into vehicle voice assistants. But linking natural-language requests to vehicle functions creates a safety-critical authorization problem. Before executing a command, the system must choose whether to execute, refuse, clarify, require confirmation, defer to manual control, trigger an emergency response, or make no tool call. To our knowledge, prior evaluations do not isolate this pre-action decision across speaker role, authentication status, vehicle state, and tool availability. We introduce a 202-scenario benchmark with Reference Decisions under a seven-class taxonomy. We evaluate two local open-weight models and three API-based LLMs using Decision Alignment and safety-specific error metrics. Alignment ranges from 40.1% for Llama 3.2 3B to 89.1% for Gemini 3.1 Pro Preview. The API-based models score between 83.2% and 89.1%, with no statistically significant differences among them
    
[^67]: 基于层次化大语言模型与RAG抽象的原始遥测数据语义层自动归纳

    Semantic Layer Induction from Raw Telemetry via Hierarchical LLM and RAG Abstraction

    [https://arxiv.org/abs/2609.19615](https://arxiv.org/abs/2609.19615)

    提出了一种端到端框架，通过层次化LLM推理与两阶段语义抽象流水线，从嘈杂的原始遥测日志中全自动构建业务语义层，免除了人工解析和脆弱映射维护的负担。

    

    现代应用程序会产生海量的原始遥测数据，但如何将这些嘈杂、异构的事件流转化为可操作的商业洞察仍然是一个根本性挑战。数据工程师和分析师需要耗费大量精力来协调语义差异、手工编写解析逻辑，并维护原始数据与业务KPI之间脆弱的映射关系。本文提出了一个端到端框架，能够从应用程序原始日志中全自动地构建业务语义层。我们的方法引入了两阶段语义抽象：第一阶段，通过结合领域特定行业知识增强的大语言模型推理来识别高层业务特征；第二阶段，通过包含数据精炼、混合检索、多阶段过滤、语义聚类和规范命名的结构化流水线来推导细粒度业务节点。在生产规模遥测数据上的评估表明（摘要至此截断）

    arXiv:2609.19615v1 Announce Type: cross  Abstract: Modern applications generate massive volumes of raw telemetry data, but translating those noisy, heterogeneous event streams into actionable business insights remains a fundamental challenge. Data engineers and analysts expend substantial effort reconciling semantic discrepancies, hand-crafting parsing logics, and maintaining fragile mappings between raw data and business KPIs. In this paper, we present an end-to-end framework that fully automates the construction of a business semantic layer from application raw logs. Our approach introduces a two-stage semantic abstraction: first, high-level business features are identified via LLM inference augmented with domain-specific industry knowledge; second, fine-grained business nodes are derived through a structured pipeline comprising data refinement, hybrid retrieval, multi-stage filtering, semantic clustering, and canonical naming. Evaluation on production-scale telemetry demonstrates th
    
[^68]: 思维链熵作为可靠性信号：一项预注册复现研究

    Chain-of-Thought Entropy as a Reliability Signal: A Preregistered Reproduction

    [https://arxiv.org/abs/2609.19606](https://arxiv.org/abs/2609.19606)

    本预注册独立复现研究证实，大语言模型思维链熵轨迹的形状（而非总熵下降幅度）是预测答案正确性的可靠信号，形状信号在四个开源模型上成功复现，而幅度信号则因实验设置而异。

    

    这项实证研究是对Zhao在2026年报告的分离效应的独立复现。大型语言模型思维链熵轨迹的形状可以预测最终答案是否正确，而其总熵下降的幅度则不能。这一分离效应值得复现，因为幅度部分基于单一模型在单一随机种子下对300个问题的单次运行，而形状部分在两个基准测试的完整规模上以及第二个模型家族上均有报告。该复现在任何验证性运行之前已在OSF注册，用四个开源权重模型遍历完整的GSM8K和MATH-500基准测试集，其中包括一个原始研究未测试过的推理蒸馏模型。结果表明，形状信号得到了复现，而幅度信号则因设置而异。在锚定模型上，单调链与非单调链之间的准确率差距在GSM8K上为+9.6个百分点，在MATH-500上为+27.5个百分点，而秩相关……（摘要原文在此处截断）

    arXiv:2609.19606v1 Announce Type: new  Abstract: This empirical study is an independent reproduction of the dissociation Zhao reported in 2026. The shape of a large language model's chain-of-thought entropy trajectory predicts whether the final answer is correct, while the magnitude of its total entropy drop does not. The dissociation merits reproduction because the magnitude half rests on a single 300-problem run with one model at one seed, while the shape half was reported at full scale on both benchmarks and on a second model family. Registered at OSF before any confirmatory run, the reproduction crosses the complete GSM8K and MATH-500 benchmark test sets with four open-weight models including one reasoning-distilled model of a kind the original did not test. The shape signal replicates. The magnitude signal divides by setting. On the anchor model the accuracy gap between monotone and non-monotone chains is +9.6 percentage points on GSM8K and +27.5 on MATH-500, while the rank correl
    
[^69]: 全双工语音模型是被问才开口，而非有需要就发言

    Full-Duplex Speech Models Take the Floor When Asked, Not When Needed

    [https://arxiv.org/abs/2609.19596](https://arxiv.org/abs/2609.19596)

    该研究通过语境匹配的英语独白实验发现，现有全双工语音模型主要在被直接称呼或出现沉默时才发言，而无法像人类那样在听到错误事实或危险信息时主动插话纠正或警告。

    

    全双工语音模型可以同时聆听和说话，有望成为始终在线的助手。然而，它们还必须决定何时应该发言。人类听众会在被点名时或说话者停止时发言，但也会主动插话，以纠正错误说法、补充遗漏词语或警告危险。我们探究全双工模型是否也能做到同样的事情。为了将发言的原因与发言的机会区分开来，我们构建了语境匹配的英语独白，其中在同一话题内仅触发语句有所变化，基于话轮分配规则定义了10种条件，并压缩了词间停顿以限制沉默所带来的发言机会。在五个模型家族中，被点名和沉默是远比错误事实或危险更可靠的发言触发因素。就Moshi和PersonaPlex而言，在触发结束后前2秒内取平均，错误事实的帧级文本词元概率低于中性情况。无论是停顿还是允许打断，都无法缩小这一差距……

    arXiv:2609.19596v1 Announce Type: new  Abstract: Full-duplex speech models listen and speak at once, promising always-on assistants. Yet they must also decide when they should speak. Human listeners speak when addressed or when the speaker stops, but also self-select to correct a false claim, supply a missing word, or warn of danger. We ask whether full-duplex models do the same. To separate the reason to speak from the opportunity, we construct context-matched English monologues in which only the trigger utterance varies within a topic, define 10 conditions from turn-allocation rules, and compress inter-word pauses to limit opportunities created by silence. Across five model families, being addressed and silence are far more reliable triggers than false facts or hazards. Frame-level text-token probabilities in Moshi and PersonaPlex are lower for false facts than for Neutral when averaged over the first 2\,s after trigger end. Pauses or permission to interrupt do not close this gap eit
    
[^70]: 基于梯度的数据归因方法中形式重于内容

    Form Over Content In Gradient-Based Data Attribution Methods

    [https://arxiv.org/abs/2609.19589](https://arxiv.org/abs/2609.19589)

    基于梯度的数据归因方法主要捕捉答案格式而非任务内容，因为共享答案格式的数据集表现出强梯度对齐，而任务相同但格式不同的数据集则不对齐。

    

    基于梯度相似性的数据归因方法被广泛用于分析和选择大语言模型的训练数据，但梯度相似性究竟衡量的是什么仍存在争议。一些研究将其解读为识别任务相关的技能，而另一些工作则报告表面形式是主要因素。我们通过独立变化任务和答案格式，为监督微调样本解决了这一争论。具体而言，我们以不同的答案格式呈现基准数据集，使得数据集可以共享任务但不共享格式，或共享格式但不共享任务。我们发现梯度对齐遵循答案格式：共享答案格式的基准数据对表现出强对齐（去衰减余弦相似度接近0.4），而相同基准以不同答案格式类别呈现时则不显示对齐（接近0.0）。我们证明这种排序从最早的预训练检查点一直延续到后训练阶段，并跨越不同的模型规模和模型家族成立。

    arXiv:2609.19589v1 Announce Type: cross  Abstract: Data attribution methods using gradient similarity are widely used to analyze and select training data for large language models, but what gradient similarity actually measures is debated. Some interpret it as identifying task-relevant skills, while other work reports that surface form is the main factor. We resolve this debate for supervised fine-tuning examples by varying task and answer format independently. Specifically, we render benchmarks in different answer formats, such that datasets can share a task without a format or a format without a task. We find that gradient alignment follows the answer format, as benchmark pairs sharing an answer format align strongly (disattenuated cosine near 0.4), while same benchmarks rendered with different answer format classes show no alignment (near 0.0). We demonstrate that this ordering holds from the earliest pretraining checkpoints through post-training, and across model scales and familie
    
[^71]: 红队测试自动模式：提升针对恶意编码智能体的拦截分类器

    Red-Teaming Auto Mode: Improving Blocking Classifiers Against Malign Coding Agents

    [https://arxiv.org/abs/2609.19587](https://arxiv.org/abs/2609.19587)

    本文通过红队测试发现，失对齐的恶意编码智能体在高层攻击策略指导下，能够通过提示注入、多智能体攻击和恶意压缩等机制，在 79% 的试验中绕过生产级拦截监视器并造成灾难性危害，据此提出了改进拦截分类器的方法。

    

    为了防止编码智能体偏离正轨，生产系统现在会使用拦截监视器审查智能体提出的每个操作，并在其执行前予以拒绝（如 Claude Code 中的 Auto Mode、OpenAI Codex 中的 Guardian）。以往对此类监视器的评估主要衡量其对意外伤害或来自试图劫持智能体的不可信来源的提示注入的鲁棒性，而当被监视的智能体本身持续处于失对齐状态时，这些监视器能否经受住考验，人们还知之甚少。为了理解这一风险，我们让一个对抗性智能体试图绕过生产环境的拦截监视器并造成灾难性危害，例如窃取其自身的模型权重。我们发现，在给定高层攻击策略的情况下，对抗性智能体可以通过多种不同的机制取得成功，例如针对监视器的智能体生成式提示注入、多智能体攻击以及恶意压缩（malicious compaction）。特别地，我们发现在 79% 的试验中，智能体都可以利用注入攻击……（原文摘要在此处截断）

    arXiv:2609.19587v1 Announce Type: cross  Abstract: To keep coding agents from going off the rails, production systems now review each proposed action with a blocking monitor that can reject it before it runs (Auto Mode in Claude Code, Guardian in OpenAI's Codex). Prior evaluations of such monitors largely measure robustness to accidental harm or prompt injections from untrusted sources looking to hijack the agent. Less understood is how they hold up when the agent they monitor is persistently misaligned. To understand this risk, we task an adversarial agent with evading production blocking monitors and causing catastrophic harm, e.g. by exfiltrating its own weights. We find that when instructed with high-level attack strategies, adversarial agents can succeed through several distinct mechanisms, such as agent-generated prompt injection against the monitor, multi-agent attacks, and malicious compaction. In particular we find that in 79% of trials, the agent can use an injection attack a
    
[^72]: CliniCIRCA：一个用于从原始电子健康档案叙述中构建心理健康患者纵向历程的模块化大语言模型框架

    CliniCIRCA: A Modular LLM Framework for Constructing Longitudinal Mental Health Patient Journeys from Raw EHR Narratives

    [https://arxiv.org/abs/2609.19585](https://arxiv.org/abs/2609.19585)

    CliniCIRCA是首个无需事件级时间戳即可从非结构化出院总结中对临床事件进行时间分类的多阶段大语言模型框架，通过临床医生参与纠错生成黄金标准标签，并支持基于时间线的患者历程总结。

    

    在心理健康护理领域，对患者历程的推理是临床医生的一项关键任务。然而，这些历程涵盖了生物、心理和社会事件的纵向进展，往往分散在不同的非结构化文本叙述中，使得时间信息的恢复极具挑战性。我们提出了CliniCIRCA，一个用于日历锚定、感知不精确性的临床编年史重建的多阶段大语言模型框架。据我们所知，CliniCIRCA是首个在没有事件级时间戳的情况下，对非结构化出院总结中的临床事件进行时间分类的方法。我们从14,882条MIMIC-III心理健康入院记录出发，首先构建了一个包含52份出院总结的基准数据集，CliniCIRCA在其上生成了15,891个带时间标记的事件。在通过临床医生参与的评估纠正了629个错误后，我们生成了经过验证的黄金标准标签。最后，纠正后的时间线驱动了一个基于时间的总结阶段……

    arXiv:2609.19585v1 Announce Type: cross  Abstract: In mental health care, reasoning over patient journeys is a key task for clinicians. Yet these journeys, encompassing a longitudinal progression of biological, psychological, and social events, are often spread across disparate unstructured text narratives, making temporal recovery challenging. We present CliniCIRCA, a multi-stage LLM framework for Calendar-anchored, Imprecision-aware Reconstruction of Clinical Annals. To our knowledge, CliniCIRCA is the first to temporally classify clinical events across unstructured discharge summaries without event-level timestamps. From 14,882 MIMIC-III mental health admissions, we first construct a benchmark of 52 discharge summaries on which CliniCIRCA produces 15,891 temporally tagged events. After correcting 629 errors based on a clinician-in-the-loop evaluation, we produce verified gold-standard labels. Finally, the corrected timelines drive a temporally grounded summarization stage that compr
    
[^73]: 用于基于证据的遗传疾病严重程度分类的大语言模型智能体

    Large Language Model Agents for Evidence Based Genetic Disease Severity Classification

    [https://arxiv.org/abs/2609.19569](https://arxiv.org/abs/2609.19569)

    该研究开发了一个结合ReAct与RAG的自主AI智能体，基于ACMG严重程度指南和ACOG生活质量标准检索并验证文献证据，实现了对10,211个人类表型本体术语的遗传病严重程度自动化分类，表型分类准确率达93.55%，并汇总基因层面的严重程度以识别严重的常染色体隐性遗传基因对。

    

    遗传疾病的严重程度分类是主观且劳动密集的，这在基因组筛查中造成了瓶颈，而商业基因面板在规模和覆盖范围上差异很大。我们开发了一个自主AI智能体，将推理与行动（ReAct）与检索增强生成（RAG）相结合，对10,211个人类表型本体（HPO）术语进行分类。该智能体使用美国医学遗传学学会（ACMG）认可的严重程度指南和美国妇产科医师学会（ACOG）的生活质量标准来检索PubMed文献，生成可解释的推理链，并独立验证论断。在表型层面，基于专家精心整理的队列，该智能体达到了93.55%的准确率（MCC 0.9237），其中82.6%至91.4%的论断得到直接证据或有效推论的支持。基因层面的严重程度在8,738对基因中进行了汇总，识别出3,283对表现为严重或极严重的常染色体隐性遗传基因对。

    arXiv:2609.19569v1 Announce Type: cross  Abstract: Disease severity classification for genetic conditions is subjective and labor-intensive, creating bottlenecks in genomic screening, where commercial panels vary widely in size and overlap. We developed an autonomous AI agent integrating Reasoning and Acting (ReAct) with Retrieval-Augmented Generation (RAG) to classify 10,211 Human Phenotype Ontology terms. It uses American College of Medical Genetics (ACMG)-endorsed severity guidelines and American College of Obstetricians and Gynecologists (ACOG) quality-of-life criteria to retrieve PubMed literature, generate interpretable reasoning chains, and independently verify claims. At the phenotype level, using expert-curated cohorts, the agent achieved 93.55% accuracy (MCC 0.9237) with 82.6% to 91.4% of claims supported by direct evidence or valid inferences. Gene-level severity was aggregated across 8,738 pairs, identifying 3,283 autosomal recessive pairs with severe or profound presentati
    
[^74]: 从参数到行为：大语言模型模型融合综述

    From Parameters to Behaviors: A Survey of Model Fusion for Large Language Models

    [https://arxiv.org/abs/2609.19553](https://arxiv.org/abs/2609.19553)

    本综述首次给出模型融合的统一定义，并建立了参数级、表示级和行为级三个层次的系统分类体系，同时梳理了相关指标、基准、应用、挑战与未来方向。

    

    模型融合是将多个源模型的能力整合到单一目标模型中的技术。截至2026年6月，Hugging Face平台托管了超过200万个模型，这一不断增长的模型池为模型重用和能力整合提供了丰富的基础。然而，现有的综述往往只涵盖该领域的部分内容，且缺乏统一的定义和系统的分类体系。本综述给出了模型融合的定义，并将已有工作归纳为三个层次：参数级融合、表示级融合和行为级融合。我们还回顾了相关的评价指标、基准测试和应用，总结了当前的挑战，并指出了未来的研究方向。我们的目标是提供该领域的清晰图谱，为模型融合的后续研究提供支持。关于模型融合论文的完整列表可参见 https://github.com/Baicaihaochi/Awesome-Model-Fusion-Survey。

    arXiv:2609.19553v1 Announce Type: new  Abstract: Model fusion integrates the capabilities from source models into a single target model. As of June 2026, Hugging Face hosts more than 2M models. This growing pool provides a rich base for model reuse and capability integration. Yet existing surveys often cover only separate parts of this space, and they do not provide a unified definition or a systematic taxonomy. This survey defines model fusion and organizes prior work into three levels: parameter-level, representation-level, and behavior-level fusion. We also review related metrics, benchmarks, and applications, summarize current challenges, and identify future directions. Our goal is to provide a clear map of this area and support future work on model fusion. A comprehensive list of papers about model fusion is available at https://github.com/Baicaihaochi/Awesome-Model-Fusion-Survey.
    
[^75]: 寻找共同点：Bluesky 起始包中的分级共同体知识

    Finding Common Ground: Graded Communal Knowledge in Bluesky Starter Packs

    [https://arxiv.org/abs/2609.19549](https://arxiv.org/abs/2609.19549)

    本研究首次利用 Bluesky 起始包作为可见的社区归属标签，通过对 191,648 对用户共享词汇库的分析，实证验证了 Clark 共同基础理论中“共享社区归属越多、共同基础越大”的分级假设。

    

    沟通之所以成为可能，是因为存在共同基础——即人们在线上或线下共享并对彼此默认的未言明知识。在其共同基础理论中，Clark（1996）区分了个人共同基础与共同体共同基础，并指出后者是分级的：两个人共享的社区归属越多，他们之间拥有的共同基础也就越多。社交媒体研究曾援引这一机制来解释用户之间如何建立连接，但由于社区成员身份很少可见，且在可见的情况下又与用户互动相耦合从而产生混淆效应，该机制在很大程度上未经实证检验。为了规避这些挑战，本研究将 Bluesky 起始包重新用作由用户自行策划的社区归属标签。在对 191,648 对用户的分析中，我们发现共享词汇库——我们作为共同基础的代理指标——随用户共享的起始包数量呈单调增长。

    arXiv:2609.19549v1 Announce Type: cross  Abstract: Communication is made possible by common ground---the unspoken knowledge that people share and presuppose of one another, whether that be online or offline. In his conception of common ground, Clark (1996) distinguishes between personal and communal common ground, and asserts that the latter is graded: the more community affiliations two people share, the more common ground they share as well. Social media research has invoked this mechanism to explain how users connect, but it has gone largely untested because community memberships are rarely visible and, where they are, they are coupled to user interactions in a way that leads to conflating effects. To circumvent these challenges, this study repurposes Bluesky starter packs (SPs) as user-curated community affiliation labels. Across 191,648 pairs of users, we show that shared lexical repertoire---our proxy for common ground---grows monotonically with the number of SPs that users share
    
[^76]: 当招聘变得由智能体中介：双智能体简历筛选中的通过机会与可复现性评估

    When Hiring Becomes Agent-Mediated: Evaluating Access and Recurrence in Two-Agent R\'esum\'e Screening

    [https://arxiv.org/abs/2609.19530](https://arxiv.org/abs/2609.19530)

    该论文提出一种由雇主方和候选人方智能体相互交流证据并更新判断的双智能体简历筛选方法，发现相比传统的单次调用筛选，它能显著提升边界案例的通过率，且决策在双向都发生变化而非单纯放宽标准。

    

    招聘是双向的：雇主评估匹配度，而候选人展示并辩护其资质证据。然而，作为第一道关卡的简历筛选，通常被自动化为对简历-职位配对的静态、单次调用判断。我们研究了一种双智能体的替代方案，其中雇主方智能体和候选人方智能体分别代表这两种角色，相互交换证据，并在决定谁晋级之前更新各自的判断。我们使用GPT-5.5和Claude Opus 4.7在600个构建的简历-职位配对上比较了两种筛选程序。双智能体筛选推进了更多的申请（GPT-5.5的通过率从33.3%提升至39.3%；Opus 4.7从34.0%提升至35.5%）。在共同191个边界样本配对的三次运行中，通过实例率分别从4.5%升至26.2%和从6.5%升至16.1%。这并非单纯的标准放宽：双智能体筛选拒绝了一些单次调用筛选会通过的申请，使决策在两个方向上都发生了改变。在相近的通过量下，两种程序推进的申请并不相同。

    arXiv:2609.19530v1 Announce Type: new  Abstract: Hiring is bilateral: employers assess fit, while candidates present and defend evidence of their qualifications. Yet r\'esum\'e screening, the first gate, is commonly automated as a static, one-call judgment over a r\'esum\'e-job pair. We study a two-agent alternative in which employer-side and candidate-side agents represent these roles, exchange evidence, and update their judgments before deciding who advances. We compare procedures on 600 constructed r\'esum\'e-job pairs using GPT-5.5 and Claude Opus 4.7. Two-agent screening advances more applications (33.3% to 39.3% for GPT-5.5; 34.0% to 35.5% for Opus 4.7). Across three runs on the common 191-pair borderline pool, pass-instance rates rise from 4.5% to 26.2% and from 6.5% to 16.1%, respectively. This is not a uniform relaxation: two-agent screening rejects applications one-call advances, changing decisions in both directions. At similar pass volumes, the procedures advance different 
    
[^77]: EconSkills：研究Web智能体在实时经济数据上的技能迁移与检索

    EconSkills: Studying Skill Transfer and Retrieval for Web Agents on Live Economic Data

    [https://arxiv.org/abs/2609.19523](https://arxiv.org/abs/2609.19523)

    EconSkills框架将验证过的经济数据检索轨迹提炼成参数化技能库，证明技能迁移和基于库的检索能显著提升Web智能体的表现。

    

    Web智能体经常需要重新访问相同的网站，然而大多数评估方法会丢弃在早期成功交互中学到的操作流程。我们提出了EconSkills，这是一个技能库和评估框架，它将经过验证的EconWebArena轨迹提炼成参数化的标准操作流程，用于检索实时经济数据。每个技能记录其适用范围、导航流程、特定网站的指导、验证检查和恢复步骤，同时用占位符替换原始实例的数值。EconSkills将两个问题分开：已知的相关流程是否能迁移到保留任务上，以及当智能体从技能库中选择时能否保持这种优势。在受控迁移实验中，匹配的技能比无技能提示提高了成功率，并且在配对成功案例中所需的步骤更少，而抽象化方法比重放原始轨迹要有效得多。在技能库规模下，检索方法与无技能基线相比具有竞争力。

    arXiv:2609.19523v1 Announce Type: new  Abstract: Web agents often revisit the same sites, yet most evaluations discard the procedures learned in earlier successful interactions. We introduce EconSkills, a skill library and evaluation framework that distills verified EconWebArena trajectories into parameterized standard operating procedures for retrieving live economic data. Each skill records its scope, navigation procedure, site-specific guidance, verification checks, and recovery steps while replacing source-instance values with placeholders. EconSkills separates two questions: whether a known relevant procedure transfers to a held-out task, and whether an agent can retain that benefit when selecting from a library. In controlled transfer, matched skills improve success over no-skill prompting and require fewer steps on paired successes, while abstraction is substantially more effective than replaying raw trajectories. At library scale, retrieval is competitive with the no-skill base
    
[^78]: 仅限你的眼睛：评估隔离的语言模型实例之间的协调

    For Your Eyes Only: Evaluating Coordination Between Isolated Language Model Instances

    [https://arxiv.org/abs/2609.19504](https://arxiv.org/abs/2609.19504)

    该论文提出了一个名为“仅限你的眼睛”的合作信号博弈框架，用于评估隔离的语言模型实例能否仅通过自然语言中的隐藏信号实现协调，发现大多数模型在需要避免可检测信号时难以维持协调能力，而一个前沿模型仍能保持近乎完美的表现。

    

    随着模型生成的内容在自动化工作流中越来越多地被其他模型实例所消费，一个具有实际重要性的问题浮现出来：一个模型能否在自然语言中嵌入某种信号，使得同一模型的独立实例仅依靠共享的预训练和任务指令就能检测到该信号，而无需任何共享记忆或针对协调的专门训练？我们提出了“仅限你的眼睛”，这是一个旨在直接评估这一问题的合作信号博弈。在该博弈中，发送者为两个词生成自由形式的描述，其中之一是隐藏的目标词；一个隔离的接收者必须识别出该目标词。我们在来自四个架构系列的七个当代模型上，使用来自权威心理语言学语料库的300个词对进行评估，并采用双重通过成功率来控制输出偏差。我们发现，大多数模型一旦被要求避免可检测的信号，就难以维持协调，而一个前沿模型则保持了近乎完美的[摘要在此处截断]

    arXiv:2609.19504v1 Announce Type: cross  Abstract: As model-generated content is increasingly consumed by other model instances in automated workflows, a practically important question arises: can a model embed a signal in natural language that an independent instance of the same model can detect, relying only on shared pre-training and task instructions, without any shared memory or coordination-specific training? We introduce For Your Eyes Only, a cooperative signalling game designed to evaluate this directly. A Sender produces free-form descriptions for two words, one of which is a hidden target; an isolated Receiver must identify it. We evaluate seven contemporary models from four architectural families on 300 word pairs from established psycholinguistic corpora, using the Double-Pass Success Rate to control for output biases. We find that most models struggle to maintain coordination once they are required to avoid detectable signals, while one frontier model retains near-perfect 
    
[^79]: 界面之外的安全性：通过大语言模型的潜在状态检测有害内容

    Safety Beyond the Interface: Detecting Harm via Latent States in Large Language Models

    [https://arxiv.org/abs/2609.19472](https://arxiv.org/abs/2609.19472)

    该研究通过从LLaMA-3.1-8B内部激活值中训练仅1260万参数的轻量级MLP探针来检测有害提示，实现了与规模大1000倍的防护模型相当的检测性能（F1最高达99%），同时显著降低了延迟和计算成本。

    

    自主系统日益依赖大语言模型（LLM），然而围绕这些模型构建的安全基础设施会引入延迟和计算开销，这限制了它们在资源受限、时间关键型部署场景中的实用性。现有的外部防护栏模型对模型的内部运作机制一无所知，造成了根本性的安全保障缺口。我们提出这样的问题：模型本身是否已经知道内容何时有害？我们从LLaMA-3.1-8B中提取内部激活值，并训练轻量级MLP分类器探针（1260万参数）来检测有害提示。在WildJailbreak、Beavertails和AEGIS 2.0数据集上的评估显示，我们的探针分别达到了99%、83%和84%的F1分数，与比其规模大1000倍的防护模型相比具有竞争力，同时大幅降低了延迟和计算成本。

    arXiv:2609.19472v1 Announce Type: new  Abstract: Autonomous systems increasingly rely on Large Language Models (LLMs) yet the safety infrastructure surrounding these models introduces latency and compute overhead. This limits utility in resource-constrained, time-critical deployments. Existing external guardrail models remain blind to the model's internal workings, creating a fundamental assurance gap. We ask: does the model already know when the content is harmful? We extract activations from LLaMA-3.1-8B and train lightweight MLP classifier probes (12.6M parameters) to detect harmful prompts. Evaluated on WildJailbreak, Beavertails, and AEGIS 2.0, our probes achieve F1 scores of 99%, 83%, and 84%, respectively competitive with 1000x larger guard models while cutting latency and compute costs.
    
[^80]: 从模型到系统：高效多模态学习的全面综述

    From Models to Systems: A Comprehensive Survey of Efficient Multimodal Learning

    [https://arxiv.org/abs/2609.19445](https://arxiv.org/abs/2609.19445)

    本综述首次提出涵盖模型、算法和系统三个层次的结构化高效多模态学习分类体系，并系统综合了跨层协同设计的方法论，以应对“效率-效用-隐私”的根本性权衡。

    

    多模态模型的快速扩张暴露了计算、内存和部署方面的严峻瓶颈，催生了高效多模态学习（EML）作为关键研究前沿的兴起。尽管进展迅速，但对效率在学习栈中体现在何处、如何体现的统一理解仍然碎片化。本综述通过引入首个结构化的从模型到系统的分类体系，对EML领域进行了系统化梳理。我们从300多篇开创性工作中提炼出见解，归纳为三个层次——模型、算法和系统——分别解决架构精简、执行优化和硬件感知编排问题。超越纯粹的分类回顾，我们对这些层次之间的垂直协同进行了方法论层面的综合，阐明了跨层协同设计如何影响根本性的“效率-效用-隐私”权衡。通过一个综合性案例研究……

    arXiv:2609.19445v1 Announce Type: cross  Abstract: The rapid expansion of multimodal models has surfaced formidable bottlenecks in computation, memory, and deployment, catalyzing the rise of Efficient Multimodal Learning (EML) as a pivotal research frontier. Despite intensive progress, a cohesive understanding of what, how, and where efficiency is manifested across the learning stack remains fragmented. This survey systematizes the EML landscape by introducing the first structured, model-to-system taxonomy. We distill insights from over 300 seminal works into three hierarchical levels--model, algorithm, and system--addressing architectural parsimony, execution refinement, and hardware-aware orchestration, respectively. Moving beyond a purely categorical review, we offer a methodological synthesis of the vertical synergies between these layers, elucidating how cross-layer co-design contributes to the fundamental "Efficiency-Utility-Privacy" trade-off. Through an integrative case study o
    
[^81]: BurnRiSc：基于公共仓库信号的开源倦怠无创筛查

    BurnRiSc: Toward Non-Invasive Burnout Screening in Open Source from Public Repository Signals

    [https://arxiv.org/abs/2609.19422](https://arxiv.org/abs/2609.19422)

    BurnRiSc框架利用GitHub公共活动数据中提取的14个行为和语言信号，将Oldenburg倦怠量表的疲惫和疏离两个维度操作化，实现了对开源维护者倦怠的无创、可追溯筛查，解决了自我报告量表无法触及最需要帮助人群的难题。

    

    倦怠是一种慢性职业综合征，而开源领域几乎是其最糟糕的案例：维护者在没有管理者重新分配工作、没有组织关注其状态下滑的情况下，承受着无止境的需求。其代价不仅是个人层面的——倦怠先于退出行为出现，而在由少数维护者支撑的项目中，一个人的离开就可能破坏成千上万下游系统所依赖的基础设施。然而，该领域目前无法预见倦怠的到来：自我报告量表作为唯一现有的测量手段，恰恰会遗漏最需要被发现的贡献者，且无法追溯应用，因此该领域甚至无法提出“倦怠有多普遍”或“什么干预有效”这样的问题。我们提出BurnRiSc，一个将Oldenburg倦怠量表的两个维度——疲惫和疏离——操作化为14个行为和语言信号的框架，这些信号从GitHub活动中计算得出，并与每个贡献者自身的历史进行对比评分。

    arXiv:2609.19422v1 Announce Type: cross  Abstract: Burnout is a chronic occupational syndrome, and open source is close to a worst case for it: maintainers absorb unbounded demand with no manager to reallocate work and no organization to notice decline. The cost is not only personal. Burnout precedes withdrawal, and in projects sustained by a handful of maintainers, one departure can break infrastructure that thousands of downstream systems depend on. Yet the field has no way to see it coming: self-report inventories, the only existing measure, miss exactly the contributors most in need of detection and cannot be applied retroactively, so the field cannot even ask how common burnout is or what helps.   We present BurnRiSc, a framework that operationalizes the Oldenburg Burnout Inventory's two dimensions, exhaustion and disengagement, as 14 behavioral and linguistic signals computed from GitHub activity and scored against each contributor's own history. The signals aggregate into two we
    
[^82]: 少即是多：基于多信号后期融合的无图多模态检索增强生成

    Less Is More: Graph-free Multimodal RAG via Multi-signal Late Fusion

    [https://arxiv.org/abs/2609.19417](https://arxiv.org/abs/2609.19417)

    TrioRAG是一个无图多模态RAG框架，通过对问题、锚点图像和VLM增强查询三种信号独立检索并后期融合，在降低成本的同时达到或超越基于图的系统性能，并引入了基于网络嘈杂图像的汽车领域多模态基准AutoQA。

    

    基于图的检索增强生成（RAG）被广泛应用于多模态、跨文档问答任务。然而，构建语料库级别的图结构成本高昂、查询缓慢且难以维护。我们提出了TrioRAG，一个无图的多模态框架，它整合来自三个互补信号的证据：问题本身、锚点图像，以及由两者生成的VLM增强查询。每个信号在共享的多向量索引（包含页面文本和页面图像）上独立检索，并通过后期融合合并结果。此外，我们引入了AutoQA，一个多模态汽车领域基准测试，其问题基于嘈杂的网络来源图像而非干净的文档来源图表，且需要跨手册推理。我们将其定位为模型策划的测试平台，而非经人工验证的黄金标准。在三个基准测试中，TrioRAG在匹配或超越基于图的系统的同时，降低了总成本。

    arXiv:2609.19417v1 Announce Type: new  Abstract: Graph-based retrieval-augmented generation (RAG) is widely used for multimodal, cross-document question answering. However, building corpus-level graphs is expensive, slow to query, and difficult to maintain. We present TrioRAG, a graph-free multimodal framework that integrates evidence from three complementary signals: the question, the anchor image, and a VLM-enhanced query generated from both. Each signal retrieves independently over a shared multi-vector index of page text and page images, and the results are combined through late fusion. Further, we introduce AutoQA, a multimodal automotive benchmark whose questions are grounded in noisy, web-sourced images rather than clean document-sourced figures. Its questions require reasoning across manuals. We position it as a model-curated testbed rather than a human-validated gold standard. Across three benchmarks, TrioRAG matches or outperforms graph-based systems while reducing total cost
    
[^83]: 一种用于自发性语音呼吸健康评估的跨语言声学疾病对齐框架

    A Cross-Lingual Acoustic Disease-Alignment Framework for Respiratory Health Assessment from Spontaneous Speech

    [https://arxiv.org/abs/2609.19398](https://arxiv.org/abs/2609.19398)

    提出跨语言疾病对齐框架CL-DAF，通过识别疾病效应跨语言一致的26个声学特征，克服了基于语音的呼吸健康评估中因语言特异性语音变异导致的跨语言迁移难题，将孟加拉语到英语的迁移AUC从0.49大幅提升至0.825。

    

    自发性语音为呼吸健康评估提供了一种可扩展、无创的信号，然而，能够跨语言泛化的可解释模型仍然具有挑战性，因为与疾病相关的声学变化会被特定语言的语音变异所混淆。我们提出了CL-DAF（跨语言疾病对齐框架），该框架能够识别疾病效应在不同语言间保持一致的声学维度。利用201名英语使用者和75名新收集的孟加拉语使用者，我们构建了一个通用的272维声学表示，并使用有符号秩双列相关效应和语言不变性得分来量化疾病对齐程度。我们首先证明自发性孟加拉语语音能够将慢性阻塞性肺疾病（COPD）患者与对照组区分开来（AUC为0.85）；然而，有133个特征在不同语言间的疾病方向发生了逆转，且完整表示的迁移效果很差（从孟加拉语迁移到英语的AUC仅为0.49）。CL-DAF筛选出26个疾病对齐特征，将AUC提升至0.825。

    arXiv:2609.19398v1 Announce Type: cross  Abstract: Spontaneous speech offers a scalable, noninvasive signal for respiratory health assessment, yet interpretable models that generalize across languages remain challenging because disease-related acoustic changes are confounded by language-specific phonetic variation. We present CL-DAF, a Cross-Lingual Disease-Alignment Framework that identifies acoustic dimensions whose disease effects remain consistent across languages. Using 201 English and 75 newly collected Bangla speakers, we construct a common 272-dimensional acoustic representation and quantify disease alignment using signed rank-biserial effects and the Language Invariance Score. We first show that spontaneous Bangla speech separates COPD from controls (AUC 0.85); however, 133 features reverse their disease direction across languages and the full representation transfers poorly (AUC 0.49 from Bangla to English). CL-DAF isolates 26 disease-aligned features that raise AUCs to 0.825
    
[^84]: 视觉Transformer与状态空间模型的黎曼-洛伦兹融合

    Riemannian--Lorentz Fusion of Vision Transformers and State-Space Models

    [https://arxiv.org/abs/2609.19384](https://arxiv.org/abs/2609.19384)

    该论文提出RLPF方法，通过将语义角色对齐的参数组提升至洛伦兹双曲面并计算正则化测地重心，实现了视觉Transformer与状态空间模型这两种异构架构的参数融合。

    

    深度学习的规模化面临关键瓶颈：数据枯竭、指数级增长的训练成本以及资源集中。模型合并（model merging）无需梯度下降即可组合预训练检查点，与重新训练相比可节省数个数量级的成本。然而，当独立训练的视觉模型具有不同的架构和参数形状时，合并它们十分困难。现有的权重空间合并方法通常假设各检查点是对齐且形状兼容的，而视觉Transformer（ViT）和状态空间模型（SSM）使用不同的算子来实现token混合。我们研究了一种混合异构合并设置，在按语义角色对齐参数组的同时保留两种架构。我们提出的黎曼-洛伦兹参数融合（Riemannian–Lorentz Parameter Fusion, RLPF）方法将经过语义对齐的参数组投影到公共坐标系，将选定的坐标提升到双曲空间的洛伦兹双曲面模型上，计算正则化的测地重心，并……

    arXiv:2609.19384v1 Announce Type: cross  Abstract: Scaling deep learning faces critical bottlenecks: data exhaustion, exponential training costs, and resource concentration. Model merging combines pre-trained checkpoints without gradient descent, offering orders-of-magnitude savings versus retraining. Combining independently trained vision models is difficult when their architectures and parameter shapes differ. Existing weight-space merging methods generally assume aligned, shape-compatible checkpoints, whereas a Vision Transformer (ViT) and a state-space model (SSM) implement token mixing with different operators. We study a hybrid Heterogeneous merging setting that retains both architectures while aligning parameter groups by semantic role. Our proposed Riemannian--Lorentz Parameter Fusion (RLPF) method projects aligned groups to common coordinates, lifts selected coordinates to the Lorentz hyperboloid model of hyperbolic space, computes a regularized geodesic barycenter, and decode
    
[^85]: 细粒度危害信号在大语言模型安全中的作用

    The Role of Fine-grained Harm Signals in LLM Safety

    [https://arxiv.org/abs/2609.19366](https://arxiv.org/abs/2609.19366)

    该研究通过从危害表征中去除通用成分、分离出正交的类别残差，并利用激活引导技术，揭示了细粒度类别特异性危害信号在11个风险类别中的编码程度因类别而异且跨模型一致，而其引发拒绝行为的能力则更依赖于具体模型。

    

    先前的研究表明，大语言模型内部的危害性表征在不同风险类别之间存在差异，同时共享一个通用的总体危害表征成分。这引出了一个关于类别特异性成分在通用危害表征之外对大语言模型安全所起作用的问题。为了回答这个问题，我们通过从每个类别的危害性表征中去除共享的通用危害性表征，分离出类别特异性成分，从而得到在每一层都与通用危害性正交的类别残差。通过在3个指令微调的大语言模型中对11个风险类别使用类别残差进行激活引导，我们发现类别残差是否编码危害性因类别而异，且这种类别层面的模式在不同模型之间是相似的。类别残差是否引发拒绝行为也因类别而异，但这种类别层面的模式更加依赖于具体模型。我们还……

    arXiv:2609.19366v1 Announce Type: new  Abstract: Prior work has shown that internal harmfulness representations in large language models vary across risk categories, while sharing a common general harm representation component. This raises a question about the role of the category-specific component beyond general harm representation in LLM safety. To answer this question, we isolate the category-specific component by removing shared general harmfulness representation from each categorical harmfulness representation, yielding a category residual that is orthogonal to general harmfulness at every layer. Using activation steering with category residuals across 11 risk categories in 3 instruction-tuned LLMs, we find that whether category residuals encode harmfulness varies across categories, and that this category-wise pattern is similar across models. Whether category residuals induce refusal also varies across categories, but this category-wise pattern is more model-dependent. We also f
    
[^86]: 全双工语音模型中工具调用的前后端架构

    A frontend-backend architecture for tool calls in full-duplex speech models

    [https://arxiv.org/abs/2609.19334](https://arxiv.org/abs/2609.19334)

    提出一种前后端架构，让全双工语音模型通过发出委派标记将流式转写交给文本LLM后端执行工具调用，并以轻量级注入机制返回结果，从而在几乎不修改前端模型的前提下保留低延迟、可打断的自然双工交互。

    

    全双工语音到语音（S2S）模型能够提供自然、低延迟的对话交互，若能具备使用外部工具并完成语音代理任务的能力将使其进一步受益。我们提出了一种前后端架构：由双工语音转文本前端学会发出一个委派标记，并将流式ASR转写文本转发给基于文本的后端大语言模型（LLM）以执行工具调用。后端的工具调用结果通过一个轻量级的预填充-重复机制注入回前端，再经流式TTS合成语音传达给用户。由于只需对前端模型进行极少的修改，我们的方法在很大程度上保留了常规的双工轮次切换、打断处理和低延迟交互。在单轮工具调用评估中，我们的系统实现了92-97%的工具调用召回率、具有竞争力的工具调用预测性能，以及81.2%的无关调用拒绝准确率。当配备更大的后端模型时……

    arXiv:2609.19334v1 Announce Type: new  Abstract: Full-duplex speech-to-speech (S2S) models provide natural, low-latency conversational interaction and would benefit from the ability to use external tools and complete voice-agent tasks. We propose a frontend-backend architecture where a duplex speech-to-text frontend learns to emit a delegation token and forwards streaming ASR transcripts to a text-based backend LLM for tool calls. Tool-call results from the backend are injected back into the frontend through a lightweight prefill-and-repeat mechanism and then synthesized using streaming TTS to the user. Our approach largely preserves regular duplex turn-taking, interruption handling, and low-latency interaction as it requires minimal modifications to the frontend model. In a single-turn tool-call evaluation, our system achieves 92-97% tool-call recall, competitive tool-call prediction performance, and 81.2% accuracy in rejecting irrelevant calls. When equipped with a larger backend (e.
    
[^87]: AUDITPLAN：先承诺、后回答，实现可审计的安全对齐

    AUDITPLAN: Commit, Then Answer for Auditable Safety Alignment

    [https://arxiv.org/abs/2609.19325](https://arxiv.org/abs/2609.19325)

    提出AUDITPLAN方法，让模型先输出结构化安全计划再据此作答，并通过FAITHGATE奖励门控机制确保答案忠实于计划，从而同时提升大模型安全对齐的鲁棒性与可审计性。

    

    安全调优流程仅评判最终答案，这使得难以区分稳健的拒绝行为与两种不良捷径：对良性请求的一概拒绝，以及看似完善但实际上并未约束答案的不忠实安全理由。我们提出AUDITPLAN，一种单模型的“先计划、后回答”方法，模型首先输出一个紧凑的结构化安全计划，然后基于该计划进行回答。该计划记录威胁标签、预期行动和明确的约束条件，从而实现机器可检查的审计，同时在部署时对用户隐藏。我们通过监督微调以及随后使用FAITHGATE的强化学习来训练这种行为，FAITHGATE是一种奖励门控目标，仅当安全计划正确时才授予答案奖励。这抑制了看似安全但不忠实的行为，并促进了更紧密的计划-答案耦合。在Qwen骨干模型上，AUDITPLAN同时提升了鲁棒性和可审计性。

    arXiv:2609.19325v1 Announce Type: cross  Abstract: Safety tuning pipelines judge only the final answer, which makes it difficult to distinguish robust refusal from two undesirable shortcuts: blanket refusal on benign requests and polished but unfaithful safety rationales that do not actually constrain the answer. We propose AUDITPLAN, a single-model plan-then-answer approach where the model first emits a compact structured safety plan and then answers conditioned on it. The plan records a threat label, intended action, and explicit constraints, enabling machine-checkable auditing while remaining hidden from users at deployment. We train this behavior with supervised fine-tuning followed by reinforcement learning with FAITHGATE, a reward-gating objective that grants answer reward only when the safety plan is correct. This discourages safe-looking but unfaithful behavior and promotes tighter plan-answer coupling. Across Qwen backbones, AUDITPLAN improves both robustness and auditability:
    
[^88]: 为什么预训练无法共享跨语言知识

    Why Pretraining Fails to Share Cross-Lingual Knowledge

    [https://arxiv.org/abs/2609.19291](https://arxiv.org/abs/2609.19291)

    本研究通过受控双语预训练实验发现，不相交的词表空间是跨语言知识泛化的根本障碍——即使是对同一语言的完全相同副本，仅仅词表不相交就足以导致知识隔阂。

    

    大型语言模型（LLMs）在多种语言的处理和建模方面取得了显著进展。然而，与人类多语言者不同，它们表现出的跨语言知识迁移能力出奇地有限。尽管这一局限性已被充分记录，但其在多语言训练过程中的起源仍不清楚。我们预训练了360M和7B参数的LLMs，并表明跨语言知识泛化能力差的问题在预训练期间就已出现，且在标准干预措施下依然持续存在。为了分离其成因，我们采用了一个受控的双语预训练设置，使用同一语言的两个副本，它们共享完全相同的文本和分词方式，但映射到不相交的词表空间。我们发现，仅不相交的词表就足以诱发知识隔阂，即使在同一语言的完全相同副本之间也是如此，从而确立了不相交的词表空间是跨语言知识泛化的根本障碍。基于这一理解，我们（注：原文摘要在此处被截断）

    arXiv:2609.19291v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have made remarkable progress in the processing and modeling of many languages. Yet, unlike human multilinguals, they exhibit surprisingly limited cross-lingual knowledge transfer. While this limitation is well documented, its origins during multilingual training remain unclear. We pretrain 360M- and 7B-parameter LLMs and show that poor cross-lingual knowledge generalization emerges during pretraining and persists under standard interventions. To isolate its cause, we employ a controlled bilingual pretraining setting using two copies of the same language, sharing identical text and token segmentation, but mapped to disjoint token spaces. We find that disjoint tokens alone are enough to induce knowledge compartmentalization, even between identical copies of the same language, establishing disjoint token spaces as a fundamental barrier to cross-lingual knowledge generalization. Guided by this understanding, w
    
[^89]: YNU-HPCC团队参加SemEval-2025任务11：使用多个预测头弥合基于文本的情感识别差距

    YNU-HPCC at SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Using Multiple Prediction Headers

    [https://arxiv.org/abs/2609.19238](https://arxiv.org/abs/2609.19238)

    该论文提出采用RoBERTa模型并改进输出头为单一预测头，同时将多语言数据集统一翻译成英文进行训练，实验证明单预测头和统一英文数据集训练的方法在情感识别任务中表现更优。

    

    本文描述了YNU-HPCC团队在SemEval-2025任务11子任务A（弥合基于文本的情感识别差距）中的参与情况。我们表现最佳的系统采用了RoBERTa（稳健优化的BERT方法）模型，这是BERT的改进版本，利用Transformer编码器架构。我们增强了输出头，使模型能够同时处理一种情感。我们获得了官方排名分数（0.44），包含了所有语言的结果。为了便于后续处理，整个数据集使用谷歌翻译翻译成了英文。通过概率和注意力分析，我们发现：（1）单个预测头的表现优于同时预测六种情感的六个预测头；（2）在统一翻译成英文的数据集上训练比使用原始数据集获得更好的结果。代码可在以下地址获取：https://github.com/BGWH123/Semeval-2025-task11。

    arXiv:2609.19238v1 Announce Type: cross  Abstract: This paper describes the participation of the YNU-HPCC team in subtask A of task 11, Bridging the Gap in Text-Based Emotion at SemEval-2025. Our best-performing system employs the RoBERTa (Robustly Optimized BERT Approach) model, an improved version of BERT that utilizes the Transformer encoder architecture. We enhanced the output head to allow the model to process one emotion simultaneously. We obtained the official ranking score (0.44), including results from all languages. The entire dataset was translated into English using Google Translate to facilitate subsequent processing. Through probabilistic and attention analyses, we found that (I) a single prediction head performs better than six heads predicting six emotions simultaneously, and (II) training on a uniformly translated English dataset yields better results than using the original dataset. The code is available at: https://github.com/BGWH123/Semeval-2025-task11.
    
[^90]: CovR：基于推理引导强化学习的覆盖率感知硬件验证

    CovR: Coverage-Aware Hardware Verification via Reasoning-Guided Reinforcement Learning

    [https://arxiv.org/abs/2609.19189](https://arxiv.org/abs/2609.19189)

    CovR是一个结合自我反思循环与仿真反馈的智能体框架，通过推理引导的强化学习自动生成硬件测试平台，突破了现有方法只关注功能正确性的局限，实现了验证覆盖率的最大化。

    

    设计验证仍然是硬件开发中资源消耗最大的阶段之一，通常消耗高达70%的总设计工作量。虽然最近的研究探索了使用大语言模型（LLM）来自动化生成测试平台，但大多数现有方法仅狭隘地关注功能正确性，忽略了覆盖率质量这一关键方面。为了弥合这一差距，我们提出了CovR，这是一个用于自动化测试平台生成的智能体框架，它将自我反思循环与基于仿真的反馈相结合，以最大化覆盖率。利用该流程，我们使用强大的教师模型构建了一个包含16,514个自然语言规范-RTL-推理-测试平台元组的大规模数据集，从而实现覆盖率感知的监督。在此基础上，我们提出了一种专为覆盖率驱动的测试平台生成而设计的强化学习（RL）框架，利用从仿真和覆盖率反馈中获得的工具奖励来优化学生模型

    arXiv:2609.19189v1 Announce Type: cross  Abstract: Design verification remains one of the most resource-intensive stages of hardware development, often consuming up to 70% of the total design effort. While recent work has explored using Large Language Models (LLMs) to automate testbench generation, most existing approaches focus narrowly on functional correctness, overlooking the critical aspect of coverage quality. To bridge this gap, we present CovR, an agentic framework for automated testbench generation that combines self-reflection loops with simulation-based feedback to maximize coverage. Using this pipeline, we construct a large-scale dataset of 16,514 natural specification RTL reasoning testbench tuples with a strong teacher model, enabling coverage-aware supervision. Building on this, we propose a reinforcement learning (RL) framework tailored for coverage-driven testbench generation, leveraging tool-derived rewards from simulation and coverage feedback to optimize a student m
    
[^91]: 消息容量与声明措辞设定了语言模型网络中集体真相探寻的转折点

    Message capacity and claim wording set the transition points of collective truth-finding in language-model networks

    [https://arxiv.org/abs/2609.19183](https://arxiv.org/abs/2609.19183)

    该研究将智能体阅读他人消息的数量上限建模为“消息容量”，发现LLM对声明的集体判断可归结为带分裂归一化权重的随机二值神经元更新规则，并据此预测当智能体平均阅读其31个信源中不足6.4个时，错误共识从任何初始状态都无法形成，且声明措辞与消息容量共同决定了集体真相探寻的转折点。

    

    无论是人类还是大型语言模型（LLM），讨论中的智能体都只能阅读其他参与者贡献中的一小部分，这受到认知、上下文或成本的限制。即使多数人一开始是正确的，LLM集体也可能达成错误的共识；我们追问的是，仅凭这一阅读限制能在多大程度上决定结果。我们用一个数字来建模这一限制——消息容量，它设定了一个智能体会阅读其他人的多少条消息，并由它生成通信网络。在31,824次随机查询中，我们发现一个80亿参数模型对某条声明的判断实际上可归结为其收件箱加权和的逻辑斯蒂函数，即具有分裂归一化权重的随机二值神经元的更新规则。仅凭这些权重和网络的度统计即可推断：当智能体平均阅读其31个信源中不足6.4个时，错误共识从任何初始状态都将变得不可达。在1,414个带有指定……（摘要在此处截断）

    arXiv:2609.19183v1 Announce Type: cross  Abstract: Whether human or large language model (LLM), an agent in a discussion reads only a few of the others' contributions, bounded by cognition, context, or cost. LLM collectives can settle on a wrong consensus even when a majority starts out correct; we ask how far that reading bound alone decides the outcome. We model the bound with one number, the message capacity, which sets how many of the others' messages an agent reads, and generate the communication network from it. Over 31,824 randomized queries, we found that an 8-billion-parameter model's judgment of a claim effectively reduces to a logistic function of a weighted sum of its inbox, the update rule of a stochastic binary neuron with divisively normalized weights. From these weights and the network's degree statistics alone, the wrong consensus should become unreachable from any start once agents read, on average, fewer than 6.4 of their 31 sources. In 1,414 episodes with assigned s
    
[^92]: 我们对大语言模型有何期望？大语言模型基准测试设计的系统图谱

    What Do We Expect from LLMs? Mapping the Design of LLM Benchmarks

    [https://arxiv.org/abs/2609.19182](https://arxiv.org/abs/2609.19182)

    该研究系统梳理了2022年至2026年间14,767篇引入或更新大语言模型评估资源的arXiv论文，绘制出基准测试设计的演变图谱，揭示评估正日益强调行动、交互和专业应用，且基于LLM的评分与模型生成材料在各类基准中的参与度发展不均衡。

    

    基准测试是评估和传达大语言模型（LLM）进展的核心方式。然而，仅凭模型排名几乎无法揭示评估需求本身是如何变化的。不断扩大的基准测试种类提供了另一个视角：研究人员期望大语言模型做什么，以及他们认为什么样的表现才算成功。我们系统性地梳理了2022年1月至2026年8月期间arXiv提交中引入或更新评估资源的14,767篇论文。通过分阶段筛选和自动化全文编码，我们考察了目标系统与领域、评估材料与条件以及评分机制的变化。该语料集显示，学界对行动、交互和专业应用的重视日益增长，同时既有设计元素与较新的设计元素经常并存。模型参与的发展也不均衡：基于大语言模型的评分在智能体和非智能体两类群体中均呈增长趋势，而模型生成的材料（摘要在此处被截断）

    arXiv:2609.19182v1 Announce Type: new  Abstract: Benchmarks are central to how progress in large language models (LLMs) is assessed and communicated. Yet model rankings alone reveal little about how evaluation requirements themselves are changing. The expanding variety of benchmarks offers another perspective: what researchers expect LLMs to do, and what they count as successful performance. We systematically map 14,767 papers introducing or updating evaluation resources from arXiv submissions between January 2022 and August 2026. Using staged screening and automated full-text coding, we examine changes in target systems and domains, evaluation materials and conditions, and scoring mechanisms. The collection shows growing emphasis on action, interaction, and professional applications, while established and newer design elements frequently coexist. Model participation also develops unevenly: LLM-based scoring grows within both agent and non-agent groups, whereas model-generated material
    
[^93]: 迈向记忆与超越：在长期多模态个人档案中从“记住你”到“懂你”

    To Memories and Beyond: From Remembering to Knowing You across Long-Term Multimodal Personal Archives

    [https://arxiv.org/abs/2609.19167](https://arxiv.org/abs/2609.19167)

    该论文提出了首个基于真实多年个人视觉档案构建的多模态长期记忆基准ReaLMem，通过事实回忆、个性推断和预测性个性化三个认知层级，推动AI从单纯记住用户事件走向真正理解用户。

    

    随着AI系统演变为个性化的数字伴侣，其核心能力之一是对用户长期个人历史进行推理：不仅是存储过去的事件，还要追踪纵向的经历和不断演变的偏好。这一领域的进展受制于评估瓶颈——现有的长期记忆基准大多是合成的、纯文本的，它们忽略了锚定人类日常记忆的视觉记录，缺乏真实个性化所需的具有因果关联的纵向数据，因而仍停留在浅层的事实回忆层面。我们提出了ReaLMem（真实世界长期多模态记忆），这是首个基于真实的多年个人视觉档案构建的基准，并配有第一人称主观标注。ReaLMem在三个难度递增的认知层级上评估模型：事实回忆、个性画像推断和预测性个性化。我们进一步提出了ChronoProfiler，一个基于时间的……

    arXiv:2609.19167v1 Announce Type: new  Abstract: As AI systems evolve into personalized digital companions, a central capability is reasoning over a user's long-term personal history: not merely storing past events, but tracking longitudinal experiences and evolving preferences. Progress here is bottlenecked by evaluation, existing long-term memory benchmarks are largely synthetic and text-only, they overlook the visual records that anchor everyday human memory, lack the authentic and causally connected longitudinal data that real personalization demands, and consequently remain confined to shallow factual recall. We introduce ReaLMem (Real-world Long-term Multimodal Memory), the first benchmark built from authentic multi-year personal visual archives, paired with first-person subjective annotations. ReaLMem evaluates models across three cognitive tiers of increasing difficulty: factual recall, persona inference, and predictive personalization. We further propose ChronoProfiler, a temp
    
[^94]: 低方差奖励下群体相对优化中的优势尺度校准失衡：诊断与有界恢复

    Advantage Scale Calibration Imbalance in Group-Relative Optimization under Low-Variance Rewards: Diagnosis and Bounded Recovery

    [https://arxiv.org/abs/2609.19164](https://arxiv.org/abs/2609.19164)

    本文诊断了低方差奖励下群体相对优化中优势尺度校准失衡的问题，提出三方校准接口揭示RLOO/Dr.GRPO与GRPO各自的失控行为，并通过奖励分辨率协议与MaxNorm-AC过滤亚分辨率噪声、实现对可信小差距的有界恢复。

    

    在验证器式的RLVR中，群体相对优化通常将优势尺度视为一个实现细节。本文区分了两种低方差情形：不应转化为偏好信号的亚分辨率抖动，以及可信但微小的基数差距——后者应当被学习而不扭曲KL校准。我们提出一个优势尺度三方校准接口：同一个组内尺度分母同时决定了奖励分支强度、提示级批次权重，以及当奖励分支在原始基数尺度上重新表达时所诱导的有效KL校准。该接口解释了为什么RLOO/Dr.GRPO会让可信的小差距被KL项主导，而GRPO的标准差分母会无界放大微小差距。基于该接口，我们进一步提出了奖励分辨率协议和MaxNorm-AC，分别用于过滤亚分辨率差距并提供有界的基数……（原文摘要至此截断）

    arXiv:2609.19164v1 Announce Type: new  Abstract: In verifier-style RLVR, group-relative optimization often treats advantage scale as an implementation detail. This paper separates two low-variance cases: sub-resolution jitter that should not become a preference signal, and credible but small cardinal gaps that should be learned without distorting KL calibration. We propose an advantage-scale three-way calibration interface: the same within-group scale denominator simultaneously determines the reward-branch strength, prompt-level batch weight, and the effective KL calibration induced when the reward branch is re-expressed on the original cardinal scale. This interface explains why RLOO / Dr.GRPO can let credible small gaps become KL dominated, whereas GRPO's standard-deviation denominator can amplify tiny gaps without bound. Based on this interface, we further introduce the Reward-Resolution Protocol and MaxNorm-AC, respectively filtering sub-resolution gaps and providing bounded cardin
    
[^95]: VisKG-LM：将知识图谱编译为视觉记忆以用于多项选择题问答

    VisKG-LM: Compiling Knowledge Graphs into Visual Memory for Multiple-Choice Question Answering

    [https://arxiv.org/abs/2609.19158](https://arxiv.org/abs/2609.19158)

    VisKG-LM 将检索到的知识图谱子图一次性离线编译为保留分支结构的可视化图像并缓存为只读记忆，使语言模型在推理时无需重复在线编码图结构，从而将图编码与语言推理解耦，提升多项选择题问答的效率。

    

    知识图谱通常通过图神经网络编码检索到的子图，并在在线推理路径中将其与语言模型融合，从而集成到问答系统中。因此，无论在训练轮次、随机种子还是评估运行中，每当对一个问题-候选对进行评分时，同一个子图都会被从头重新编码，即使知识图谱本身从未改变。我们探讨检索到的知识图谱是否可以改为一次性离线编译，然后作为只读内存进行访问。VisKG-LM 证明这是可行的，其方法是将图编码与语言推理解耦。它将每个检索到的与候选相关的子图序列化为“关系标注路径”，并将结果渲染为图像，其二维布局保留了路径的分支结构。每张图像只需离线编码一次并缓存以供重复使用。在推理阶段，语言模型仅从文本对问题和候选进行上下文化处理，并且仅在其最终层……（摘要截断）

    arXiv:2609.19158v1 Announce Type: new  Abstract: Knowledge graphs are usually integrated into question answering by encoding a retrieved subgraph with a graph neural network and fusing it with the language model in the online inference path. The same subgraph is therefore re-encoded from scratch every time a pair is scored, across training epochs, seeds, and evaluation runs, even though the knowledge graph never changes. We ask whether the retrieved knowledge graphs can instead be compiled once, offline, and then accessed as read-only memory. VisKG-LM shows that it can, by decoupling graph encoding from language reasoning. It serializes each retrieved candidate-specific subgraph as Relation-Labeled Paths and renders the result as an image whose two-dimensional layout preserves the branching structure of the paths. Each image is encoded once, offline, and cached for reuse. At inference, the language model contextualizes the question and candidate from text alone, and only its final laye
    
[^96]: 反思性恢复：一种通过从错误中学习来实现推理的自监督方法

    Reflective Recovery: A Self-Supervised Method for Reasoning by Learning from Mistakes

    [https://arxiv.org/abs/2609.19156](https://arxiv.org/abs/2609.19156)

    本文提出“反思性恢复”这一自监督方法，通过将LLM失败的推理尝试转化为恢复训练数据，教会模型从错误中恢复，从而突破了仅依赖完美推理轨迹的模仿学习方法在数据有限时的“规模坍塌”瓶颈。

    

    数据驱动的微调因其简单高效而被广泛用于增强大型语言模型（LLM）的推理能力。然而，完全依赖完美推理轨迹的主流模仿学习方法存在“规模坍塌”（Scaling Collapse）问题：当问题集有限时，增加正例无法带来持续的性能提升。与此同时，在推理过程中，LLM无法保证每个中间步骤都正确，因此容易出错。一旦出现此类错误，LLM往往难以恢复，并可能被先前错误的累积进一步误导。为解决这一问题，我们提出了Reflective Recovery（反思性恢复），这是一种简单而有效的自监督方法，可将失败的推理尝试转化为恢复训练数据。具体而言，我们提取失败轨迹的初始片段，将其与提示词拼接，并用其引导LLM得出有效的解决方案。

    arXiv:2609.19156v1 Announce Type: new  Abstract: Data-driven fine-tuning is widely adopted to enhance reasoning in Large Language Models (LLMs) due to its simplicity and efficiency. However, mainstream imitation learning methods that rely exclusively on perfect reasoning trajectories suffer from a Scaling Collapse: when the problem set is limited, increasing positive examples fails to yield continuous improvement. However, during inference, an LLM can not guarantee that every intermediate step is correct and is therefore prone to errors. Once such errors arise, the LLM often struggles to recover and may be further misled by the accumulation of previous mistakes. To address this, we propose Reflective Recovery, a simple yet effective self-supervised approach that transforms failed reasoning attempts into recovery training data. Specifically, we extract initial segments of failed trajectories, concatenate them with prompts, and use them to guide the LLM toward valid solutions. Because th
    
[^97]: 迈向人机对话中用户侧隐式冲突的主动检测

    Towards Proactive Detection of User-Side Implicit Conflicts in Human-LLM Dialogue

    [https://arxiv.org/abs/2609.19155](https://arxiv.org/abs/2609.19155)

    该论文构建了首个用于评估用户侧隐式冲突检测的人工标注基准UC-Bench，并通过数据合成方法提升轻量级LLM在有限训练数据下主动检测用户侧隐式冲突的能力。

    

    在人类与大语言模型（LLM）的对话中，用户的后续发言可能与先前的意图产生隐式冲突，导致LLM误解用户需求并生成不恰当的回应。一个可靠的对话系统应该在生成回应之前主动检测用户侧的冲突，并在必要时寻求澄清。然而，先前的工作主要集中在LLM侧的冲突上，用户侧的冲突尚未得到充分探索。为了填补这一空白，我们构建了UC-Bench，一个用于评估用户侧冲突检测的人工标注基准。初步实验表明，现有的LLM在这一任务上表现不佳，尤其是当冲突源于基于对话历史的隐式不兼容性时。为了在有限的训练数据下提升轻量级LLM的能力，我们研究了针对用户侧冲突检测的数据合成方法。现有的合成方法并未显式地对历史用户发言与当前用户发言之间的隐式不兼容性进行建模。

    arXiv:2609.19155v1 Announce Type: new  Abstract: In Human-LLM dialogue, follow-up user utterances may implicitly conflict with earlier intents, leading the LLM to misinterpret user needs and generate inappropriate responses. A reliable dialogue system should proactively detect user-side conflicts before generating a response and seek clarification when necessary. However, prior work has largely focused on LLM-side conflicts, leaving user-side conflicts underexplored. To fill this gap, we construct UC-Bench, a human-annotated benchmark for evaluating user-side conflict detection. Preliminary experiments show that existing LLMs struggle with this task, especially when conflicts arise from implicit incompatibilities grounded in dialogue history. To improve lightweight LLMs with limited training data, we investigate data synthesis for user-side conflict detection. Existing synthesis methods do not explicitly model the implicit incompatibilities between historical and current user utterance
    
[^98]: Neo-Classic：一个用于评估中国古典诗歌语言-审美推理能力的基准测试

    Neo-Classic: A Benchmark for Evaluating Linguistic-Aesthetic Reasoning in Classical Chinese Poetry

    [https://arxiv.org/abs/2609.19154](https://arxiv.org/abs/2609.19154)

    该论文提出 Neo-Classic 基准，利用当代专家创作的严格合律诗歌和逆向理解探针来测试大语言模型的古典诗歌语言-审美推理能力，发现最先进模型在分层约束满足上存在20%至50%的性能差距等显著局限。

    

    虽然大型语言模型（LLM）在现有的中国古典诗歌基准测试上已达到很高的准确率，但仍然难以区分可迁移的语言-审美推理能力与对熟悉预训练模式的依赖。为了解决这一问题，我们提出了 Neo-Classic，一个结合了建构式样本外（OOS）数据集与一系列逆向理解探针的评估基准。与依赖历史语料库进行验证或生成的传统基准不同，Neo-Classic 由当代专家创作的严格符合格律的诗歌构成，降低了直接检索的可能性。我们使用五个旨在测试分层约束满足能力的行为探针，评估了包括 Qwen3-Max、Gemini-3-Pro 和 DeepSeek-V3.2 在内的最先进模型。我们的结果揭示了两个主要局限：第一，当模型从历史……（原文摘要在此处截断）

    arXiv:2609.19154v1 Announce Type: new  Abstract: While Large Language Models (LLMs) achieve high accuracy on established Classical Chinese Poetry benchmarks, it remains challenging to distinguish transferable Linguistic-Aesthetic Reasoning from reliance on familiar pre-training patterns. To address this issue, we introduce Neo-Classic, an evaluation benchmark that combines a constructionist Out-of-Sample (OOS) dataset with a suite of reverse understanding probes. Unlike traditional benchmarks that rely on verification or generation over historical corpora, Neo-Classic comprises strictly metrical poetry authored by contemporary experts, reducing the possibility of direct retrieval. We evaluate state-of-the-art models, including Qwen3-Max, Gemini-3-Pro, and DeepSeek-V3.2, across five behavioral probes designed to test hierarchical constraint satisfaction. Our results reveal two primary limitations. First, a performance gap of 20 to 50 percent emerges when models transition from historica
    
[^99]: 停止去除停用词：一项沿袭的预处理默认设置如何扭曲法律文本即数据研究

    Stop Removing Stopwords: How an Inherited Preprocessing Default Distorts Legal Text-as-Data

    [https://arxiv.org/abs/2609.19153](https://arxiv.org/abs/2609.19153)

    本研究通过穷尽式单词消融实验，首次直接针对下游分类目标验证停用词去除这一沿袭自信息检索时代、从未被验证过的预处理默认设置，揭示其可能扭曲实证法学中基于TF-IDF和线性分类器的文本数据分析结果。

    

    实证法学研究日益将司法文本视为数据，其中许多研究仍依赖稀疏、可解释的处理流程——TF-IDF特征和线性分类器——因为文本特征往往是研究对象本身，而不仅仅是实现预测的手段。然而，这些流程继承了一系列源自二十世纪中叶信息检索领域的预处理默认设置，这些设置从未针对分类准确度进行过验证，其中最根深蒂固的是停用词去除。本研究引入了一种穷尽式的单词消融方法，直接根据下游目标衡量预处理步骤的效果，并将其应用于停用词去除这一最难撼动的案例。通过将最高法院数据库的标签与Caselaw Access Project的判决意见文本相匹配，该研究考察了两个涵盖F1提升空间的双分类任务：意识形态方向（不去除停用词的基线F1约0.68）和宪法与非宪法法律类型（约0.92），涉及7,66

    arXiv:2609.19153v1 Announce Type: new  Abstract: Empirical legal scholarship increasingly treats judicial text as data, and much of it still runs on sparse, interpretable pipelines -- TF-IDF features and linear classifiers -- because the textual feature is often the object of study, not merely a means to a prediction. Yet these pipelines inherit a chain of preprocessing defaults from mid-century information retrieval that were never validated against classification accuracy, the most entrenched being stopword removal. This study introduces an exhaustive single-word ablation that measures a preprocessing step's effect directly against the downstream objective, and applies it to stopword removal as the hardest case to dislodge. Matching Supreme Court Database labels to Caselaw Access Project opinion texts, it examines two binary tasks that bracket F1 headroom, ideological direction (no-removal baseline F1 ~ 0.68) and constitutional versus non-constitutional law type (~ 0.92), across 7,66
    
[^100]: FakeSpotter：一种内容与策略无关的病毒式虚假信息检测工具

    FakeSpotter: A content and strategy agnostic Viral Misinformation Detection Tool

    [https://arxiv.org/abs/2609.19152](https://arxiv.org/abs/2609.19152)

    FakeSpotter通过测量虚假信息的结构指纹而非直接判定真伪，实现了内容与策略无关的病毒式虚假信息风险评估，在长短文本上分别取得0.788和0.793的宏观F1分数。

    

    虚假信息检测工具通常依赖于二元真假分类或基于历史样本训练的模型，这在出现新型误导性叙事时限制了其实用性。在此，我们提出了FakeSpotter，这是一种与内容和策略无关的工具，旨在通过测量虚假信息的结构指纹而非直接判定真伪来评估文本内容的病毒式虚假信息风险。FakeSpotter在语言、叙事、逻辑和批判性思维等维度上实现了理论驱动的框架，采用重复的LLM评估以及针对长短文本的领域特定逻辑回归分类器。在一个包含来自社交媒体和FakeNewsNet的764篇文本的标注语料库中，FakeSpotter在留出测试集上对短文本和长文本分别取得了0.788和0.793的宏观F1分数。FakeSpotter的解释层通过基于特征的分数和信号聚合提供可解释的输出。

    arXiv:2609.19152v1 Announce Type: new  Abstract: Misinformation detection tools often rely on binary true and false classifications or models trained on historical examples, limiting their usefulness when novel misleading narratives emerge. Here, we present FakeSpotter, a content- and strategy-agnostic tool designed to estimate the viral misinformation risk of textual content by measuring structural fingerprints of misinformation rather than directly adjudicating truthfulness. FakeSpotter operationalizes a theory-driven framework across linguistic, narrative, logical, and critical-thinking dimensions, using repeated LLM assessments and domain-specific logistic regression classifiers for short and long texts. In a labelled corpus of 764 texts from social media and FakeNewsNet, FakeSpotter achieved macro F1 scores of 0.788 for short texts and 0.793 for long texts on a held-out test set. FakeSpotter's interpretive layer provides explainable outputs through feature-based scores, signal agr
    
[^101]: 用户如何看待生成式AI：应用商店评论中信任与摩擦的跨平台NLP分析

    What Users Think of Generative AI: A Cross-Platform NLP Analysis of Trust and Friction in App Store Reviews

    [https://arxiv.org/abs/2609.19151](https://arxiv.org/abs/2609.19151)

    该研究首次对ChatGPT、Gemini、Claude等六大生成式AI应用在应用商店的17,012条评论进行大规模NLP分析，通过BERTopic主题建模与RoBERTa情感分类揭示用户负面情绪主要集中于广告、身份验证、服务器可靠性和订阅定价等采用障碍。

    

    生成式AI（GenAI）应用已实现快速的用户普及，然而很少有大规模研究考察用户感知的质量、信任和采用障碍。我们首次对六大主流生成式AI应用（ChatGPT、Gemini、Microsoft Copilot、Claude、DeepSeek和Perplexity）的应用商店评论进行了跨应用分析，涵盖来自Google Play和苹果App Store的17,012条英文评论。我们将BERTopic主题建模与RoBERTa情感分类相结合，并使用卡方检验、Kruskal-Wallis检验以及经Bonferroni校正的多项逻辑回归来评估跨应用差异。两个模型组件均通过对300条评论的分层样本进行人工编码验证。结果显示，负面情绪主要集中在广告（91%）、身份验证（89%）、服务器可靠性（83%）和订阅定价（73%）方面。情感在不同应用之间存在显著差异，

    arXiv:2609.19151v1 Announce Type: new  Abstract: Generative AI (GenAI) applications have achieved rapid consumer adoption, yet little large-scale research examines user-perceived quality, trust, and adoption barriers. We present one of the first cross-application analyses of app store reviews for six major GenAI applications (ChatGPT, Gemini, Microsoft Copilot, Claude, DeepSeek, and Perplexity), comprising 17,012 English-language reviews from Google Play and the Apple App Store. We combine BERTopic topic modeling with RoBERTa sentiment classification and evaluate cross-application differences using chi-square, Kruskal-Wallis, and multinomial logistic regression with Bonferroni correction. Both components are validated against human coding using a stratified sample of 300 reviews. Results show that negative sentiment concentrates in advertising (91%), authentication (89%), server reliability (83%), and subscription pricing (73%). Sentiment differs significantly across applications, with
    
[^102]: 采样揭示风格：大语言模型激活中提示条件风格轴的无监督、免训练发现

    Sampling Reveals Style: Unsupervised, Training-Free Discovery of Prompt-Conditional Stylistic Axes in LLM Activations

    [https://arxiv.org/abs/2609.19150](https://arxiv.org/abs/2609.19150)

    该论文提出一种无需训练的无监督方法，通过对同一提示的高温重复采样补全进行主成分分析，自动发现并标记大语言模型激活中与提示相关的风格轴，并通过245个人类风格标注验证了其与人类自发风格需求的高度契合。

    

    大语言模型（LLM）在其隐藏激活中编码了丰富的风格结构，但要发现对于给定提示哪些风格维度是显著的，通常需要监督式对比数据。我们提出了一种无需训练、提示条件化的替代方法：我们对单个提示在较高温度下反复采样补全结果，对汇集的隐藏激活应用主成分分析（PCA），并根据极性生成结果自动标记所得的风格轴。我们在一项两阶段研究中，将发现的轴与245个人类风格标注进行了验证。在我们最强的模型（Qwen-3.5-4B-Instruct）上，前两个轴以72.8%的精确率和43.6%的宏召回率匹配用户自发请求的风格维度，75.6%的有效性评分认为这些轴的极性生成结果与其标签相符，标注者之间的相邻一致性达90.9%。风格轴的可发现性强烈依赖于模型本身：两个Qwen模型（原文摘要在此处截断）

    arXiv:2609.19150v1 Announce Type: new  Abstract: Large language models (LLMs) encode rich stylistic structure in their hidden activations, but discovering which stylistic dimensions are salient for a given prompt typically requires supervised contrastive data. We present a training-free, prompt-conditional alternative: we repeatedly sample completions of a single prompt at elevated temperature, apply Principal Component Analysis (PCA) to the pooled hidden activations, and label the resulting axes automatically from the pole generations. We validate the discovered axes against 245 human-elicited stylistic annotations in a two-phase study. On our strongest model (Qwen-3.5-4B-Instruct), the top two axes match spontaneously requested human dimensions with 72.8% precision and 43.6% macro-recall, and 75.6% of validity ratings judge the axes' polar generations accurate to their labels, with 90.9% adjacent inter-annotator agreement. Discoverability is strongly model-dependent: both Qwen models
    
[^103]: 超越静态几何的潜意识提示：因果深度与多标记混淆因素

    Subliminal Prompting Beyond Static Geometry: Causal Depth and Multi-Token Confounds

    [https://arxiv.org/abs/2609.19149](https://arxiv.org/abs/2609.19149)

    该论文首次将标记纠缠解释中的相关性测量与因果性测量明确区分开，通过在多个深度进行隐藏状态复制的因果干预实验，发现静态输出向量相似度随模型规模增大而失去预测力，而隐藏状态对所传递特质的因果控制能力则显著增强。

    

    潜意识学习表明，语言模型能够通过表面上与其无关的输出传递隐藏特质。一种被提出的解释是“标记纠缠”，即通过模型的输出词表将动物标记与数字标记关联起来。然而，现有的测量方法回答的是不同的问题：输出是否共变、固定的输出向量是否对齐、能否从隐藏状态中读出答案、或者该状态是否因果地控制答案。我们在一个固定的动物-数字提示协议中分别对上述各项进行测量。从Llama-3.1-8B到70B，固定输出向量相似度对行为的预测能力下降：配对平均相关变化为-0.080（95%置信区间[-0.127, -0.035]）。固定输出头读出在归一化深度AUC上未显示出可分辨的变化。为检验因果控制，我们在五个深度处将一个数字提示的临时答案位置状态复制到另一个提示中，并测量最终的动物得分跟随哪个提示。供体-对照AUC从0.254上升到0.540，

    arXiv:2609.19149v1 Announce Type: new  Abstract: Subliminal learning shows that language models can transmit a hidden trait through outputs that appear unrelated to it. One proposed explanation, token entanglement, links animal and number tokens through the model's output vocabulary. Yet existing measurements answer different questions: whether outputs co-vary, fixed output vectors align, an answer can be read from a hidden state, or that state causally controls the answer. We measure each separately in a fixed animal-number prompting protocol. From Llama-3.1-8B to 70B, fixed output-vector similarity predicts behavior less well: the paired mean correlation change is -0.080 (95% CI [-0.127, -0.035]). A fixed output-head readout shows no resolved change in normalized depth AUC. To test control, we copy the temporary answer-position state from one number prompt into another at five depths and measure which prompt the final animal score follows. Donor-control AUC rises from 0.254 to 0.540,
    
[^104]: 用于矛盾与迟疑识别的模态差异Transformer

    Modality Discrepancy Transformer for Ambivalence and Hesitancy Recognition

    [https://arxiv.org/abs/2609.19148](https://arxiv.org/abs/2609.19148)

    提出模态差异Transformer（MDT），将跨模态矛盾信号显式建模为9-token表示（模态嵌入、绝对差特征与Hadamard积差异特征），结合FiLM文本条件调制、LoRA微调及文本引导后期融合，实现临床视频中矛盾与迟疑情感状态的自动识别。

    

    矛盾与迟疑（A/H）是一种情感状态，个体会在面部、语音和语言等通道上表达出相互矛盾的信号。要在临床视频中自动识别A/H，需要检测跨模态的不一致性——而这正是标准融合方法所抑制的信号。基于Bekhouche等人提出的冲突感知多模态融合框架，我们提出了模态差异Transformer（MDT）。MDT将原始的6-token设计扩展为9-token表示，包含三个模态嵌入、三个绝对差特征以及三个通过线性投影学习的Hadamard积差异特征。这九个token经过Transformer自注意力机制处理，其中基于FiLM的文本条件调制和LoRA微调是其核心架构组件。文本引导的后期融合分支在推理阶段将仅文本的辅助头与完整的多模态输出进行融合。在来自第3届BAH数据集（原文在此处截断）

    arXiv:2609.19148v1 Announce Type: new  Abstract: Ambivalence and hesitancy (A/H) are affective states in which individuals express contradictory signals across facial, vocal, and linguistic channels. Automatically recognising A/H in clinical videos requires detecting cross-modal disagreement -- the signal that standard fusion methods suppress. Based on the conflict-aware multimodal fusion framework of Bekhouche et al., we present the Modality Discrepancy Transformer (MDT). MDT enriches the original 6-token design to a 9-token representation comprising three modality embeddings, three absolute-difference features, and three Hadamard-product discrepancy features learned through linear projections. These nine tokens undergo Transformer self-attention, with FiLM-based text-conditioned modulation and LoRA fine-tuning as core architectural components. A text-guided late fusion branch blends a text-only auxiliary head with the full multimodal output at inference. On the BAH dataset from the 3
    
[^105]: FRAUDSkill：面向音频反欺诈检测的结构化冻结权重技能优化方法

    FRAUDSkill: Structured Frozen-Weight Skill Optimization for Audio Anti-Fraud Detection

    [https://arxiv.org/abs/2609.18766](https://arxiv.org/abs/2609.18766)

    本文提出FRAUDSkill框架，在不修改底层音频-语言模型参数的情况下，通过外部优化技能程序、路由策略和决策规则，实现了能够灵活适应欺诈模式演变的结构化音频反欺诈检测。

    

    大型音频-语言模型通过直接处理语音并对欺诈相关证据进行推理，在反欺诈检测任务中展现出巨大潜力。然而，模型的实际部署要求预测结果遵循预定义的标签空间，以及一个由服务场景识别、欺诈检测和条件性欺诈类型分类组成的结构化决策协议。现有的微调和基于提示的方法通常将任务知识、约束条件和决策规则编码到模型参数或手动维护的提示中，这使得它们难以随着欺诈模式和标注策略的演进而灵活调整。为此，我们提出了FRAUDSkill，这是一个结构化的冻结权重适配框架，它在保持底层音频-语言模型完全不变的同时，优化一个由技能程序、路由特定策略和决策规则组成的外部层。我们进一步将结构化输出控制与验证引导的多路径推理相结合，以……（摘要内容不完整，此处为截断部分）

    arXiv:2609.18766v1 Announce Type: cross  Abstract: Large audio-language models have shown promise for anti-fraud detection by directly processing speech and reasoning over fraud-related evidence. Their deployment, however, requires predictions to follow a predefined label space and a structured decision protocol consisting of service-scenario identification, fraud detection, and conditional fraud-type classification. Existing fine-tuning and prompt-based approaches typically encode task knowledge, constraints, and decision rules into model parameters or manually maintained prompts, making them difficult to adapt as fraud patterns and labeling policies evolve. To this end, we propose FRAUDSkill, a structured frozen-weight adaptation framework that leaves the underlying audio-language model unchanged while optimizing an external layer of skill programs, route-specific policies, and decision rules. We further combine structured output control with validation-guided multi-path inference to
    
[^106]: TeleAntiFraud 2.0：一个可刷新、基于用户档案、面向电信欺诈检测的音频基准

    TeleAntiFraud 2.0: A Refreshable, Profile-Grounded, and Audio-Based Benchmark for Telecom Fraud Detection

    [https://arxiv.org/abs/2609.18748](https://arxiv.org/abs/2609.18748)

    提出了 TeleAntiFraud 2.0，一个可按月更新、基于用户档案且能在共享上下文中区分欺诈与合法近域通话的音频电信欺诈检测基准。

    

    电信欺诈话术快速演变，且常常被设计得类似于日常服务对话，这对基于音频的电信欺诈评估提出了两个关键要求。首先，基准测试必须能够纳入新观察到的诈骗模式，而不会覆盖之前已建立的测试集。其次，基准必须能够将欺诈与合法的近领域通话区分开来，而不是依赖于主题分离的负面样本。我们提出了 TeleAntiFraud 2.0，该基准采用我们的混合树反欺诈生成流水线（Mixed-Tree Anti-Fraud Generation Pipeline）构建，并在每月冻结评估协议下进行评估。该流水线将在线欺诈案例摘要转化为基于用户档案的场景，通过混合树生成进行扩展，在共享上下文中实现欺诈与非欺诈对话路径，将验证过的对话渲染为角色匹配的语音，并将生成的音频、标签、提示词、清单和溯源记录冻结，用于每个月的评估集。每个冻结的集合……

    arXiv:2609.18748v1 Announce Type: cross  Abstract: Telecom fraud scripts evolve rapidly and are often designed to resemble routine service conversations, creating two key requirements for audio-based telecom-fraud evaluation. First, benchmarks must incorporate newly observed scam patterns without overwriting previously established test sets. Second, they must distinguish fraud from lawful, near-domain calls rather than relying on topic-separated negative examples. We present TeleAntiFraud 2.0, constructed with our Mixed-Tree Anti-Fraud Generation Pipeline and evaluated under a monthly frozen evaluation protocol. The pipeline transforms online fraud-case abstracts into profile-grounded scenarios, expands them through mixed-tree generation, realizes fraud and non-fraud dialogue paths under shared contexts, renders validated dialogues as role-matched speech, and freezes the resulting audio, labels, prompts, manifests, and provenance records for each monthly evaluation set. Each frozen set
    
[^107]: SEA-LION-v4.8：技术报告

    SEA-LION-v4.8: A Technical Report

    [https://arxiv.org/abs/2609.18310](https://arxiv.org/abs/2609.18310)

    基于NVIDIA Nemotron 3构建的SEA-LION-v4.8东南亚语言模型家族，通过持续预训练、监督微调和在线同策略蒸馏，显著提升了七种东南亚语言在指令遵循、推理和理解任务上的表现。

    

    我们介绍了Nemotron-SEA-LION-v4.8，这是一个基于NVIDIA Nemotron 3构建的东南亚语言一体化网络模型家族。该家族包括30B-A3B和120B-A12B两个模型，同时提供持续预训练的基础检查点和后训练变体。我们使用东南亚语言、推理、代码和多语言平行数据集对模型进行适配，随后通过监督微调和在线同策略蒸馏进行后训练。在SEA-HELM基准测试中，30B-A3B模型将SEA综合得分从46.06提升至51.57，而120B-A12B模型则从49.30提升至63.44。在七种东南亚语言的指令遵循、自然语言推理和自然语言理解方面均取得了最显著的提升。

    arXiv:2609.18310v1 Announce Type: new  Abstract: We introduce Nemotron-SEA-LION-v4.8, a family of Southeast Asian Languages in One Network (SEA-LION) built upon NVIDIA Nemotron 3. The family includes 30B-A3B and 120B-A12B models, with both continued-pretrained base checkpoints and post-trained variants. We adapt the models using Southeast Asian, reasoning, code, and multilingual parallel datasets, followed by post-training with supervised fine-tuning and online on-policy distillation. On SEA-HELM, the 30B-A3B model improves the overall SEA score from 46.06 to 51.57, while the 120B-A12B model improves from 49.30 to 63.44. The strongest gains are observed in instruction following, natural language reasoning, and natural language understanding across seven Southeast Asian languages.
    
[^108]: 回滚世界，保留反思：面向长程LLM智能体的回滚诱导反思

    Rollback the World, Keep the Reflection: Rollback-Induced Reflection for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2609.18304](https://arxiv.org/abs/2609.18304)

    提出了回滚诱导反思（RIR）统一恢复框架，在将LLM智能体回滚到选定先前状态的同时，保留从被放弃轨迹中提炼的可复用知识，解决了长程任务中错误累积且难以可靠恢复的问题。

    

    大语言模型（LLM）智能体越来越多地通过多步环境交互来处理长程任务，然而单个错误的动作可能会改变后续的状态和观测，导致错误随时间不断累积。现有方法要么在不修复已改变环境状态的情况下纠正上下文，要么在恢复早期状态的同时丢弃有用的经验，这使得既消除失败条件又避免重复过去的错误变得困难。我们认为，可靠的恢复应被视为一个回滚边界控制问题，即联合决定何时干预、从何处恢复，以及哪些信息应在恢复过程中保留。基于这一观点，我们提出了回滚诱导反思（RIR），这是一个统一的恢复框架，它将执行恢复到选定的先前状态，同时保留从被放弃轨迹中提炼出的可复用知识，以指导后续决策。我们进一步刻画……

    arXiv:2609.18304v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly tackle long-horizon tasks through multi-step environment interaction, yet a single erroneous action can alter subsequent states and observations, causing errors to compound over time. Existing methods either correct the context without repairing altered environment states or restore earlier states while discarding useful experience, making it difficult to both eliminate failure conditions and avoid repeating past mistakes. We argue that reliable recovery should instead be treated as a rollback-boundary control problem that jointly determines when to intervene, where to resume, and what information should survive recovery. Based on this view, we propose Rollback-Induced Reflection (RIR), a unified recovery framework that restores execution to a selected prior state while carrying forward reusable knowledge distilled from the abandoned trajectory to guide subsequent decisions. We further chara
    
[^109]: M²Tok：面向视觉-语言-动作模型的多头多码本离散动作分词器

    ${M}^2$Tok: Multi-head Multi-codebook Discrete Action Tokenization for Vision-Language-Action Models

    [https://arxiv.org/abs/2609.18259](https://arxiv.org/abs/2609.18259)

    提出 M²Tok，一种多头多码本离散动作分词器，通过将潜在动作特征分解为多个头并采用多个码本以最小化重构误差，突破“离散化瓶颈”，从而提升视觉-语言-动作模型的控制性能。

    

    近期的研究进展已成功将自回归语言模型适配到处理多模态信号，例如图像和动作。由于原始动作信号是连续的，有效的分词化对于将高维输入映射为紧凑的离散标记以进行自回归处理至关重要。然而，现有的离散动作分词器往往存在较高的重构损失，无法保留精确控制所需的细粒度动态信息。这种“离散化瓶颈”显著限制了下游视觉-语言-动作（VLA）模型的性能上限。为解决这一问题，我们提出了 M²Tok，一种多头多码本动作分词器，旨在最小化重构误差并提升策略性能。我们的方法引入了两项关键的结构创新：（1）我们将潜在动作特征分解为多个头，使模型能够隐式地将特定的头与不同的语义信息相关联

    arXiv:2609.18259v1 Announce Type: cross  Abstract: Recent advancements have successfully adapted autoregressive language models to process multimodal signals, such as images and actions. Since raw action signals are continuous, effective tokenization is essential to map high-dimensional inputs into compact discrete tokens for autoregressive processing. However, existing discrete action tokenizers often suffer from high reconstruction loss, failing to preserve the fine-grained dynamics required for precise control. This ``discretization bottleneck'' significantly limits the performance ceiling of downstream Vision-Language-Action (VLA) models. To address this, we propose $\mathcal{M}^2$Tok, a Multi-head Multi-codebook Action Tokenizer designed to minimize reconstruction error and enhance policy performance. Our approach introduces two key structural innovations: (1) we decompose the latent action features into multiple heads, enabling the model to implicitly align specific heads with di
    
[^110]: Fathom：面向卸载KV缓存稀疏解码的逐查询读取深度

    Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches

    [https://arxiv.org/abs/2609.17652](https://arxiv.org/abs/2609.17652)

    Fathom提出一种让每个查询自适应决定键通道读取位数的稀疏解码方法，通过比特平面存储与逆注水式比特预算分配，在百万token卸载KV缓存场景下实现比现有136位扫描方法快1.67倍的GPU解码速度，同时保持更低注意力误差。

    

    当智能体会话运行至百万token且同时驻留多个会话时，KV缓存及其排序索引存放在主机内存中，而针对top-k步骤对所有n个键进行排序的扫描成为限制解码速度的流量瓶颈。我们提出Fathom，一种由每个查询自主决定读取每个键通道多少比特的键扫描方法。4位K缓存以通道优先的方式存储为比特平面，因此t个平面的前缀恰好构成该通道的t位量化器，查询通过对方差加权通道重要性进行逆注水来分配其比特预算。在Qwen3-8B上处理一百万token时，解码步骤的GPU时间比Double Sparsity、Loki和SparQ r=32的136位扫描快1.67倍；在与SparQ的68位读取（r=16）相同的GPU时间内，Fathom读取的字节数减少18%，且在七个模型与上下文设置中的六个上注意力误差更低。在RULER风格的任务上，每次逐token扫描均与精确top-k解码结果相匹配。

    arXiv:2609.17652v1 Announce Type: cross  Abstract: When agentic sessions run to a million tokens with many sessions resident at once, the KV cache and the index that ranks it live in host memory, and the scan that ranks all n keys for a top-k step becomes the traffic that bounds decoding. We present Fathom, a key scan in which each query decides how many bits of each key channel to read. The 4-bit K cache is stored channel-major as bit planes, so a prefix of t planes is exactly the channel's t-bit quantizer, and the query spends its bit budget by reverse water-filling over the variance-weighted importance of its channels. At one million tokens on Qwen3-8B a decode step is 1.67x faster in GPU time than with the 136-bit scans of Double Sparsity, Loki and SparQ r=32, and in the same GPU time as SparQ's 68-bit read (r=16) Fathom reads 18% fewer bytes with lower attention error on six of seven model and context settings. On RULER-style tasks every per-token scan matches exact top-k decoding
    
[^111]: RiskChainBench：面向混淆平台消息还原与基于证据的网络调查的基准测试

    RiskChainBench: A Benchmark for Obfuscated Platform Message Restoration and Evidence-Grounded Web Investigation

    [https://arxiv.org/abs/2609.16900](https://arxiv.org/abs/2609.16900)

    该论文提出了RiskChainBench基准测试，首次将混淆平台消息的还原任务与基于证据的网络风险调查任务串联评估，弥补了现有基准将混淆文本与风险网页割裂评估、无法衡量目标还原因何影响下游证据获取的不足。

    

    平台滥用活动利用表情符号、同音字、字形拆分和冗余符号来隐藏跳转指令，然后通过伪装链接将用户引导至与色情、欺诈、赌博或非法交易相关的服务。现有基准测试分别评估混淆文本和风险网页，掩盖了目标还原如何影响下游证据获取。我们提出了RiskChainBench，将来自600个源会话的3,600个合成符号-文本还原输入与600个相应的人工标注本地网络环境配对。模型首先还原消息、操作意图和目的地；随后同一底层模型作为由视觉语言模型（VLM）驱动的网络代理，调查正确关联的网站，并在不依赖消息侧语义或域名信誉线索的情况下，生成冻结的、带证据引用的风险报告。我们分别对还原任务和正确路由的网络调查进行评分，并将二者组合进行综合评估。

    arXiv:2609.16900v1 Announce Type: new  Abstract: Platform abuse campaigns conceal redirection instructions with emojis, homophones, character decomposition, and redundant symbols, then route users through disguised links to services associated with pornography, fraud, gambling, or illicit transactions. Existing benchmarks evaluate obfuscated text and risky webpages separately, obscuring how target recovery affects downstream evidence acquisition. We introduce RiskChainBench, pairing 3,600 synthetic token-text restoration inputs from 600 source sessions with 600 corresponding human-labeled local web environments. A model first restores the message, operational intent, and destination; the same underlying model then acts as a VLM-driven web agent that investigates the correctly associated website and produces a frozen, evidence-cited risk report without message-side semantics or domain-reputation cues. We score restoration and correct-routing web investigation separately and compose them
    
[^112]: 《人类与大语言模型如何在“性别中立”的身体描述中解读出性别》

    How Humans and LLMs Read Gender into Gender-Neutral Physical Descriptions

    [https://arxiv.org/abs/2609.16366](https://arxiv.org/abs/2609.16366)

    本研究构建了包含316个身体属性及14,706个人类性别关联评分的GAPA数据集，发现看似“客观中立”的身体描述实际上承载着结构化的性别关联，并评估了16个大语言模型与人类评分的匹配程度。

    

    当基础模型描述人物时，AI公平性、无障碍性和伦理领域的近期研究建议避免使用推断出的身份标签（如“她”、“他的”），转而采用看似“客观”的身体描述（如“短发”、“轮廓分明的下巴”）。然而，这种描述性语言能否实现性别中立的沟通，仍然是一个悬而未决的实证问题。为了研究这一问题，我们提出了GAPA（身体属性的性别关联）数据集，其中包含从多种来源收集的316个常见身体属性，以及来自304名美国标注者的14,706个性别关联评分。结果表明，身体描述在读者中承载着结构化且分级的性别关联，且针对女性和男性的关联比对非二元性别的关联更加一致和鲜明。随后，我们评估了16个来自不同模型家族、不同规模和不同训练后变体的大语言模型，并将其与人类评分进行对比。结果显示，这些模型能够部分恢复人类的性别关联模式。

    arXiv:2609.16366v1 Announce Type: cross  Abstract: When foundation models describe people, recent work in AI fairness, accessibility, and ethics recommends avoiding inferred identity labels (e.g., "she", "his") in favor of seemingly "objective" physical descriptions (e.g., "short hair", "a defined jawline"). Yet whether such descriptive language achieves gender-neutral communication remains an open empirical question. To study this, we introduce GAPA (Gender Associations of Physical Attributes), a dataset of 316 common physical attributes drawn from diverse sources, paired with 14,706 gender-association ratings from 304 US-based annotators. Results show that physical descriptions carry structured and graded gender associations among readers, with more consistent and distinctive associations for women and men than for non-binary identities. Next, we evaluate 16 LLMs across model families, sizes, and post-training variants against human ratings. The models partially recover human associa
    
[^113]: 构建即可验证：临床问答中逐字引用的声明级评估

    Verifiable by Construction: Claim-Level Evaluation of Verbatim Citation in Clinical Question Answering

    [https://arxiv.org/abs/2609.15964](https://arxiv.org/abs/2609.15964)

    该论文基于四份临床实践指南构建了标准化评估框架，从为每个事实性声明提供引用、生成逐字引用到确保引用完全支撑声明，端到端地评估了十二个大语言模型在临床问答中构建可验证答案的能力。

    

    大语言模型（LLMs）已被广泛应用于临床问答任务。当前系统可以在答案后附加引用，但这些引用通常指向较为宽泛的文本，使得时间紧迫的临床医生无法高效地进行验证。另一种方案是确保响应在构建时就具备可验证性：提供来自参考材料的细粒度逐字引用来支撑声明，使用户无需打开其他文档即可验证答案。在本文中，我们评估了当前模型端到端执行此任务的能力：从为每个事实性声明提供引用，到生成逐字引用，再到确保这些引用能够完全支撑相应声明。为此，我们基于四份临床实践指南构建了一个标准化评估框架，并在222个合成的临床问题上评估了十二个大语言模型，分别衡量上述各个阶段的表现。我们发现大多数模型可以为声明附加逐字引用（摘要在此处不完整）。

    arXiv:2609.15964v1 Announce Type: new  Abstract: Large language models (LLMs) have been widely adopted for clinical question answering (QA). Current systems can attach citations to their answers, but these often point to broad texts, leaving time-pressed clinicians unable to verify them efficiently. An alternative is to ensure that responses are verifiable by construction: providing fine-grained verbatim quotes from reference material that substantiate claims, so users can verify an answer without opening other documents. In this paper, we evaluate the ability of current models to perform this task end-to-end: from providing citations for every factual claim, to producing verbatim quotes, to ensuring that those quotes fully substantiate the claims. To do so, we build a standardized harness over four clinical practice guidelines and evaluate twelve LLMs on 222 synthetic clinical questions, measuring each of these stages separately. We find that most models can attach verbatim quotes to 
    
[^114]: SlopShape：识别AI生成的商业网络内容

    SlopShape: Identifying AI-Generated Commercial Web Content

    [https://arxiv.org/abs/2609.15369](https://arxiv.org/abs/2609.15369)

    该研究提出通过结构特征（信息呈现方式、顺序、证据与语气）而非词级特征来识别商业网页中的AI生成内容，仅用187个结构特征就在模型自我改写的对抗条件下仍保持约98%的检测性能。

    

    词级检测器几乎能完美识别未经编辑的AI生成文本，但已有文献记录了它们在文本改写后的脆弱性，且词级评分既无法刻画文本特征，也无法识别出自哪个AI模型。我们探讨能否在更深一层——从结构特征上来识别AI生成的文本：信息如何呈现、以何种顺序、使用什么证据、采用什么语气。我们将StoryScope（Russell等人，2026）在AI生成小说中揭示的此类模式复制到商业内容上：以268个公司域名的2,250篇ChatGPT问世前的人类博客文章，对比来自五个前沿模型的11,250篇AI镜像文本。一个包含214个特征的测量工具由LLM应用，并通过人工黄金标注环节验证（人与人kappa系数0.928，人与模型0.946），仅凭其中187个结构特征，就在留出的公司域名上以98.0宏F1检测出AI生成的文章，且当每篇AI文章被其自身模型改写时，性能保持不变（98.1）。

    arXiv:2609.15369v1 Announce Type: new  Abstract: Word-level detectors identify unedited AI-generated text almost perfectly, but the literature documents their brittleness under rewording, and a word-level score neither characterizes a text nor identifies which AI model wrote it. We ask whether AI-generated text can be identified one level deeper, from structural signatures: how information is presented, in what order, with what evidence, and in what voice. We replicate StoryScope (Russell et al., 2026), which showed such patterns for AI-generated fiction, on commercial content: 2,250 pre-ChatGPT human blog posts from 268 company domains against 11,250 AI mirrors from five frontier models. A 214-feature instrument, applied by an LLM and validated in a human gold-annotation session (human-human kappa 0.928, human-model 0.946), detects AI posts from its 187 structural features alone at 98.0 macro-F1 on held-out companies, unchanged (98.1) when every AI post is reworded by its own model. T
    
[^115]: MUSE：一个面向氛围叙事的理论驱动故事引擎

    MUSE: A Theory-Harnessed Story Engine for Vibe Narrativizing

    [https://arxiv.org/abs/2609.15188](https://arxiv.org/abs/2609.15188)

    提出了MUSE故事引擎，通过将罗伯特·麦基故事理论工程化为针对具体创作决策的指导，并使其贯穿规划、起草和修改全过程，实现将自然语言写作需求转化为高质量完整故事的“氛围叙事”任务。

    

    大语言模型能够生成流畅的散文。故事质量取决于关于情节、人物和语言的决策如何在规划、起草和修改的整个过程中协同运作。指导这些决策面临两个瓶颈：故事指导的质量及其持续运用。我们将“氛围叙事”定义为将自然语言写作需求转化为完整故事的任务，并提出了MUSE——一个理论驱动的故事引擎。MUSE将故事知识组织为针对特定创作决策的指导，并将这些决策贯穿于后续的创造性工作中。知识工程通过规则原子化、语义整合和机制抽象来发展罗伯特·麦基的故事理论；由此产生的指导通过单一事实来源和分层披露加以组织。典型示例补充了依赖上下文和审美判断的原则。智能体框架组织了设计、人物表演、场景构图和修改等环节。

    arXiv:2609.15188v1 Announce Type: new  Abstract: LLMs can generate fluent prose. Story quality depends on how decisions about plot, character, and language work together across planning, drafting, and revision. Guiding these decisions presents two bottlenecks: the quality of story guidance and its sustained use. We formulate Vibe Narrativizing as the task of turning natural-language writing requirements into a finished story and present MUSE, a Theory-Harnessed Story Engine. MUSE organizes story knowledge as guidance for specific decisions and carries those decisions into subsequent creative work. Knowledge engineering develops Robert McKee's story theory through rule atomization, semantic consolidation, and mechanism abstraction; a single source of truth and layered disclosure organize the resulting guidance. Typical examples complement principles that depend on context and aesthetic judgment. An agent harness organizes design, character performance, scene composition, and revision th
    
[^116]: 一个用于大语言模型针对性危害缓解的高效模块化框架

    An Efficient and Modular Framework for Targeted Harm Mitigation in LLMS

    [https://arxiv.org/abs/2609.13624](https://arxiv.org/abs/2609.13624)

    提出了一种结合Activated LoRA适配器与上下文感知路由机制的模块化纠正框架，可在生成过程中以低延迟、有针对性的方式缓解大语言模型的有害输出，同时提升模型对齐性能。

    

    摘要：大语言模型（LLMs）是强大的零样本学习器，但仍然容易与人类偏好产生不一致，经常输出带有偏见、有毒或其他有害的内容。现有的对齐方法虽然有效，但成本高昂且与模型紧密耦合，限制了灵活性和可扩展性。我们提出了一个模块化纠正框架，通过Activated LoRA（aLoRA）适配器和上下文感知路由机制来增强预训练的大语言模型，以消除模型失调响应带来的危害。我们的方法使专家适配器能够在序列中间激活而不使KV缓存失效，从而在生成过程中实现低延迟的针对性纠正。每个专家都被训练用于检测和缓解特定类型的危害，例如偏见或毒性。一个经过学习的路由器根据模型的中间输出动态选择合适的专家。我们证明该系统在标准安全基准测试中改善了对齐效果，同时保留了……

    arXiv:2609.13624v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are powerful zero-shot learners but remain prone to misalignment with human preferences, often producing biased, toxic, or otherwise harmful outputs. Existing alignment methods, while effective, are costly and tightly coupled to the model, limiting flexibility and scalability. We propose a modular correction framework that augments pretrained LLMs with Activated LoRA (aLoRA) adapters and a context-aware routing mechanism to eliminate harms from misaligned model responses. Our approach enables expert adapters to activate mid-sequence without invalidating the KV cache, allowing low-latency, targeted correction during generation. Each expert is trained to detect and mitigate specific harms, such as bias or toxicity. A learned router dynamically selects appropriate experts based on the models intermediate outputs. We demonstrate that our system improves alignment on standard safety benchmarks while preserving t
    
[^117]: 全双工语音大语言模型中虚假起音的因果分析与缓解

    Causal Analysis and Mitigation of Spurious Onsets in Full-Duplex Speech LLMs

    [https://arxiv.org/abs/2609.13445](https://arxiv.org/abs/2609.13445)

    本研究通过因果分析发现，全双工语音LLM（如Moshi和PersonaPlex）在用户静默时产生虚假语音起音，是因为模型以自身非语音输出为条件导致语音起音概率在单个80毫秒帧内飙升超过九个数量级，而非重复采样所致，并进一步基于因果反事实分析提出了缓解方法。

    

    像Moshi及其衍生模型PersonaPlex这样的语音到语音大语言模型可以通过全双工生成同时进行听和说。然而，在用户长时间保持沉默时，它们可能会不恰当地开始说话：在数字零输入下，Moshi和PersonaPlex分别在40个五分钟延续中的12个和11个中启动了语音。是什么导致了这种虚假语音？我们研究了两个假设：一是重复采样在起音概率持续较低的情况下仍然选择了语音，二是模型以自身的非语音输出为条件导致起音概率突然飙升。我们发现，在每一次观察到的起音中，语音概率都在一个80毫秒的帧内飙升超过九个数量级，这支持了后一个假设。随后，为了在不阻断真实响应的情况下抑制这些起音，我们提出了一个因果反事实问题：模型是在响应用户语音，还是即使移除之前的用户语音，其下一个token的分布仍会保持相似……

    arXiv:2609.13445v1 Announce Type: new  Abstract: Speech-to-speech LLMs like Moshi, and its derivative PersonaPlex, can listen and speak concurrently through full-duplex generation. However, they can begin speaking inappropriately during prolonged user silence: under digital-zero input, Moshi and PersonaPlex initiate speech in 12/40 and 11/40 five-minute continuations, respectively. What causes this spurious speech? We investigate two hypotheses: either repeated sampling selects speech despite persistently low onset probabilities, or conditioning on the model's nonspeech outputs causes an abrupt spike in onset probability. We find that, at every observed onset, speech probability spikes by over nine orders of magnitude in one 80-ms frame, supporting the latter hypothesis. Then, to suppress these onsets without blocking genuine responses, we ask a causal counterfactual question: is the model responding to user speech, or would its next-token distribution remain similar if the preceding u
    
[^118]: 影响神经智能体模拟中依存长度最小化涌现的因素

    Factors Influencing the Emergence of Dependency Length Minimization in Neural Agent Simulations

    [https://arxiv.org/abs/2609.06025](https://arxiv.org/abs/2609.06025)

    本研究基于循环神经网络的智能体语言学习与交流模拟框架，在更贴近现实的交互情境中探究依存长度最小化偏好如何涌现及其影响因素，为解答这一偏好是否源于高效信息处理约束提供新途径。

    

    面对多种语法选择时，语言使用者倾向于选择能够缩短句法依存总体长度的语序，这一原则被称为依存长度最小化。这种偏好的起源仍然是一个悬而未决的问题，特别是它是否源于高效信息处理的约束。计算模拟为识别影响语言现象涌现的因素提供了强大的方法。然而，先前关于依存长度最小化的模拟尚未考察现实的交互情境，且研究结果不一。本研究使用最近提出的基于循环神经网络的语言学习与交流框架，研究人工语言中依存长度最小化的涌现。在该框架中，智能体被训练去生成和理解人工语言，然后使用这些语言进行交流。使用该框架，我们研究了……

    arXiv:2609.06025v2 Announce Type: replace  Abstract: Given various grammatical options, language users prefer the word order choice that reduces the overall length of syntactic dependencies, a principle known as dependency length minimization (DLM). The origins of this preference remain an open question, particularly whether it originates from constraints on efficient information processing. Computational simulations provide a powerful approach to identifying the factors influencing the emergence of linguistic phenomena. However, previous simulations of DLM have not examined realistic interaction contexts and have produced mixed results. The present study investigates the emergence of DLM in artificial languages using a recently proposed language learning and communication framework based on recurrent neural networks (RNNs). In this framework, agents are trained to speak and interpret artificial languages and then use these languages to communicate. Using this framework, we study the i
    
[^119]: 蒸馏之前先验证：面向在线策略蒸馏的提示级教师门控

    Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation

    [https://arxiv.org/abs/2609.02998](https://arxiv.org/abs/2609.02998)

    该论文提出教师门控在线策略蒸馏（TGOPD），通过经验证器评分的教师探测在提示级别先验证教师模型的可靠性，将可靠提示路由到密集OPD监督、不可靠提示路由到基于验证器的GRPO，从而避免“自信但错误”的教师模型诱导误导性更新。

    

    在线策略蒸馏（OPD）通过在学生模型自身的生成结果上提供来自冻结教师模型的密集token级监督来加速后训练过程。原始的OPD在所有提示上均匀地应用这种监督，而不检查教师模型对每个提示是否可靠。由于反向KL散度具有模式寻求特性，一个自信但错误的教师模型可能导致强烈却具有误导性的更新。分布性代理指标（如熵或教师-学生似然一致性）只能衡量不确定性或一致性，但无法直接验证结果的正确性。我们提出了教师门控在线策略蒸馏（TGOPD），其核心原则是在接受密集监督之前，应在提示级别验证教师模型的可靠性。TGOPD通过一小组经验证器评分的教师探测样本估计教师可靠性，并将每个提示专门路由到密集OPD（当可靠性检查通过时）或基于验证器的GRPO（当检查不通过时）。在4B和3...（摘要内容不完整）

    arXiv:2609.02998v1 Announce Type: cross  Abstract: On-policy distillation (OPD) accelerates post-training by providing dense token-level supervision from a frozen teacher on the student's own rollouts. Vanilla OPD applies this supervision uniformly across prompts, without checking whether the teacher is reliable for each prompt. Because reverse KL is mode-seeking, a confidently wrong teacher can induce a strong yet misleading update. Distributional proxies, such as entropy or teacher-student likelihood agreement, measure uncertainty or agreement but do not directly verify outcome correctness. We introduce Teacher-Gated On-Policy Distillation (TGOPD), built on the principle that teacher reliability should be verified at the prompt level before dense supervision is admitted. TGOPD estimates reliability from a small set of verifier-scored teacher probes and routes each prompt exclusively to dense OPD when the reliability check passes or to verifier-grounded GRPO otherwise. Across 4B and 3
    
[^120]: 基于音素的TTS增强用于ASR的扩展：统一流程与受控研究

    Scaling phoneme-based TTS augmentation for ASR: A unified pipeline and controlled study

    [https://arxiv.org/abs/2608.26697](https://arxiv.org/abs/2608.26697)

    本文提出了一种基于音素的统一TTS到ASR增强流程，并引入音素频率引导选择（PFGS）方法，在多种语言的ASR任务中有效提升了性能。

    

    合成语音为自动语音识别（ASR）提供了可扩展的监督信号，但其效果取决于所选的文本、参考语音和合成数据量。我们提出了一种统一的基于音素的TTS到ASR增强流程，该流程围绕一个使用F5-TTS架构从头训练并带有语言ID条件化的多语言TTS模型构建。该流程结合了特定语言的音素转换、参考语音过滤、候选文本选择、合成和匹配的ASR续训练。我们进一步提出了音素频率引导选择（PFGS），该方法利用从真实ASR训练标签中估计的音素频率对候选句子进行排序。针对阿拉伯语、法语、意大利语和葡萄牙语的独立单语ASR系统的实验覆盖了13个测试集。在合成规模扫描中，随机增强在11个测试集上优于仅匹配真实数据的续训练。在标称60%合成比例下，该方法进一步提升了性能。

    arXiv:2608.26697v1 Announce Type: new  Abstract: Synthetic speech provides scalable supervision for automatic speech recognition (ASR), but its benefit depends on the selected texts, reference speech, and amount of synthesized data. We present a unified phoneme-based TTS-to-ASR augmentation pipeline built around a multilingual TTS model trained from scratch using the F5-TTS architecture with language-ID conditioning. The pipeline combines language-specific grapheme-to-phoneme conversion, reference-speech filtering, candidate-text selection, synthesis, and matched ASR continuation. We further propose phoneme-frequency-guided selection (PFGS), which ranks candidate sentences using phoneme frequencies estimated from real ASR training labels. Experiments with separate monolingual ASR systems for Arabic, French, Italian, and Portuguese span 13 test sets. Across the synthesis-scale sweep, random augmentation improves over matched real-only continuation on 11 test sets. Under a nominal 60% sy
    
[^121]: 缓解多阶段LLM招聘流水线中的捏造问题：提示护栏与人在回路检查点的实证评估

    Mitigating Fabrication in Multi-Stage LLM Pipelines for Hiring: An Empirical Evaluation of Prompt Guardrails and Human-in-the-Loop Checkpoints

    [https://arxiv.org/abs/2608.26171](https://arxiv.org/abs/2608.26171)

    该论文通过实证评估表明，仅靠提示护栏不足以消除LLM招聘流水线中的捏造问题，而结合人在回路检查点能显著降低捏造率并消除身份虚构。

    

    多阶段LLM招聘流水线（简历改进、面试问题生成、答案反馈）可能会捏造资质、夸大条件、虚构经历。我们评估了两种缓解措施——提示护栏和人在回路（HITL）检查点——与完全自动化基线进行对比。在一项受控实验中（10份合成简历×2个职位描述×3次重复×3种条件；共180次运行），基线（C1）在96.7%的输出中产生了至少一项无依据声明（平均每输出6.80项发现）。提示护栏（C2）将发现密度降低了86%（从6.80降至每输出0.92项），但仍有50.0%的输出包含捏造内容，表明仅靠提示级缓解措施不足。在简历改进后设置人工检查点（C3）消除了所有身份捏造，将发现密度降低了59%（从6.88降至每输出2.82项），将项目级捏造率从96.7%降至75.0%（p=0.022），并减少了捕获职位描述中嵌入的陷阱要求。

    arXiv:2608.26171v1 Announce Type: cross  Abstract: Multi-stage LLM hiring pipelines (resume improvement, interview question generation, answer feedback) can fabricate credentials, inflate qualifiers, and invent experience. We evaluate two mitigations, prompt guardrails and human-in-the-loop (HITL) checkpoints, against a fully automated baseline. In a controlled experiment (10 synthetic resumes x 2 job descriptions x 3 repetitions x 3 conditions; 180 runs), the baseline (C1) produced at least one unsupported claim in 96.7% of outputs (mean 6.80 findings/output). Prompt guardrails (C2) reduced finding density by 86% (6.80 to 0.92/output), but 50.0% of outputs still contained a fabrication, showing prompt-level mitigation alone is insufficient. A human checkpoint after resume improvement (C3) eliminated all identity fabrications, reduced finding density by 59% (6.88 to 2.82/output), reduced item-level fabrication from 96.7% to 75.0% (p=.022), and cut capture of JD-embedded trap requiremen
    
[^122]: 大语言模型查询模拟中的“知识诅咒”：概念溯源用于追踪答案侧侵入

    The "Curse of Knowledge" in LLM Query Simulation: Concept Provenance for Tracing Answer-Side Intrusion

    [https://arxiv.org/abs/2608.25245](https://arxiv.org/abs/2608.25245)

    本文提出概念溯源框架，用于识别大语言模型生成查询中预设答案侧知识的“知识诅咒”现象，该框架能有效区分人类变异与答案侧侵入，并发现候选答案侧概念普遍存在。

    

    大语言模型生成的搜索查询被广泛用于增强信息检索评估，但这些查询可能包含预设了答案侧文档知识的概念，违反了搜索前用户的信息访问边界。现有的验证指标，包括重叠度、多样性和有效性，无法区分罕见的人类尾部变异与候选答案侧侵入。我们引入了概念溯源框架，该框架将查询概念分配到背景支持、人类中心、人类尾部和候选答案侧区域，从而操作化了一个仅靠检索指标无法检测的边界。将概念溯源应用于跨越100个UQV100主题、8个大语言模型和5种提示条件的77,004个查询，并使用两种提取管道，我们在五个条件均值上获得了跨管道的token-HCIR Spearman相关系数为1.0。候选答案侧概念占非通用概念的7.40%，出现在100个主题中的97个中，且具有主题解释性。

    arXiv:2608.25245v1 Announce Type: cross  Abstract: LLM-generated search queries are widely used to augment IR evaluation, yet they may contain concepts that presuppose answer-side document knowledge, violating the information-access boundary of pre-search users. Existing validation metrics, including overlap, diversity, and effectiveness, cannot distinguish rare human-tail variation from candidate answer-side intrusion. We introduce concept provenance, a framework that assigns query concepts to backstory-supported, human-central, human-tail, and candidate answer-side zones, operationalizing a boundary that retrieval metrics alone cannot detect. Applying concept provenance to 77,004 queries across 100 UQV100 topics, 8 LLMs, and 5 prompt conditions with two extraction pipelines, we obtain a cross-pipeline token-HCIR Spearman rho of 1.0 over five condition means. Candidate answer-side concepts constitute 7.40 percent of non-generic concepts and appear in 97 of 100 topics, with topic expla
    
[^123]: LongWoF-Bench：用于可验证长工作流任务的EvoMap基因评估基准

    LongWoF-Bench: Evaluating EvoMap Genes for Verifiable Long-Workflow Tasks

    [https://arxiv.org/abs/2608.23200](https://arxiv.org/abs/2608.23200)

    本文提出LongWoF-Bench基准和EvoMap方法，通过将验证器确认的执行轨迹整合为结构化基因，实现经验复用，在可验证长工作流任务中显著优于技能方法。

    

    arXiv:2608.23200v1 公告类型：新 摘要：大型语言模型日益被期望执行复杂工作流，其成功依赖于维护相互关联的约束条件，并生成满足严格端到端验证的工件。然而，成功的执行经验通常在单次运行后丢失，迫使后续模型从头重新发现策略和失败模式。我们研究这种经验是否可以通过EvoMap外部化并复用，其中验证器确认的执行轨迹被整合成结构化基因。为评估此设置，我们引入了长工作流基准（LongWoF-Bench），包含778个可机器验证的任务，涵盖代码生成、智能体环境合成、数学推理和规则遵循。在252个具有验证器确认的Opus轨迹的任务上，进化的EvoMap基因在所有七个评估模型中比技能方法平均高出8.7-15.5个百分点，且这些优势延伸至未见任务。

    arXiv:2608.23200v1 Announce Type: new  Abstract: Large language models are increasingly expected to execute complex workflows whose success depends on maintaining interdependent constraints and producing artifacts that satisfy strict end-to-end verification. Yet successful execution experience is typically lost after a single run, forcing subsequent models to rediscover strategies and failure modes from scratch. We study whether such experience can instead be externalized and reused through EvoMap, where verifier-confirmed execution trajectories are consolidated into structured Gene. To evaluate this setting, we introduce the Long-Workflow Benchmark (LongWoF-Bench), comprising 778 machine-verifiable tasks across code generation, agent-environment synthesis, mathematical reasoning, and rule following. On the 252 tasks with verifier-confirmed Opus trajectories, evolved EvoMap Gene outperform Skill across all seven evaluated models by 8.7-15.5 percentage points, with the gains extending t
    
[^124]: 迈向更安全的RAG：只有具备系统2思考能力的代理才能访问不可信文档

    Towards Safer RAG: Only Agents Capable of System 2 Thinking may Access Untrusted Documents

    [https://arxiv.org/abs/2608.17153](https://arxiv.org/abs/2608.17153)

    本文提出一种新的安全原则，即仅允许具备系统2推理能力的代理访问不可信文档，以减少RAG系统中的知识投毒攻击影响，并引入新指标量化检测与影响间的差异。

    

    检索增强生成（RAG）显著提升了大型语言模型（LLMs）的性能，但这些系统仍然容易受到知识投毒攻击，即检索文档中的错误信息可能影响模型的最终输出。值得注意的是，LLM可能正确检测到文档包含错误信息，却仍受其影响。先前的研究通过“隔离原则”（Cordon Principle）解决了这一漏洞，该原则防止负责最终答案合成的模型直接访问原始证据。尽管有效，但这种严格隔离可能带来大量计算开销。在本工作中，我们提出了一种精细化的安全原则：只有具备深思熟虑的系统2推理能力的代理才能访问不可信文档。为评估这一原则，我们引入了新指标，用于量化错误信息检测与下游影响之间的差异。我们进行了实验...

    arXiv:2608.17153v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) has significantly enhanced the performance of large language models (LLMs), yet these systems remain vulnerable to knowledge-poisoning attacks, in which misinformation in retrieved documents can influence the model's final outputs. Notably, an LLM may correctly detect that a document contains incorrect information while nevertheless being influenced by it. Prior work has addressed this vulnerability through the Cordon Principle, which prevents models responsible for final answer synthesis from directly accessing raw evidence. Although effective, this strict isolation can introduce substantial computational overhead. In this work, we propose a refined security principle: only agents capable of deliberative System 2 reasoning may access untrusted documents. To evaluate this principle, we introduce novel metrics that quantify the discrepancy between misinformation detection and downstream influence. We t
    
[^125]: 解耦对比解码通过专家对齐草拟

    Decoupled Contrastive Decoding via Expert-Aligned Drafting

    [https://arxiv.org/abs/2608.12913](https://arxiv.org/abs/2608.12913)

    本文提出解耦对比解码（DCD），通过专家对齐的轻量级提议者进行草拟，仅在验证阶段应用业余模型，既保持了原始对比解码的输出分布，又避免了草拟阶段引入的误差放大问题，从而更高效且稳定。

    

    arXiv:2608.12913v1 公告类型：新  摘要：对比解码（CD）提高了生成质量，但其业余模型的过程使解码成本高昂。使用推测解码加速CD引发了一个提案对齐问题：对比信号应该塑造草拟者，还是仅保留在验证中？我们在轻量级特征级草拟者机制下研究这一问题。两个受控诊断——匹配的交叉Alpha训练和近似双草拟者分解——给出了相同的结论：对比感知的草拟并不一致地优于专家对齐的草拟，因为对比校正通常弱于草拟者误差，而重建可能放大该误差。我们引入了解耦对比解码（DCD），它使用专家对齐的轻量级提议者进行草拟，并仅在未改变的CD验证中应用业余模型。标准推测验证保持了原始CD的输出分布。在主要的8B集合上。

    arXiv:2608.12913v1 Announce Type: new  Abstract: Contrastive Decoding (CD) improves generation quality, but its amateur-model pass makes decoding expensive. Accelerating CD with speculative decoding raises a proposal-alignment question: should the contrastive signal shape the drafter, or should it remain only in verification? We study this question in the lightweight feature-level drafter regime. Two controlled diagnostics, matched Cross-alpha training and an Approximate Dual-Drafter decomposition, give the same diagnosis: contrastive-aware drafting does not consistently improve over expert-aligned drafting because the contrastive correction is usually weaker than drafter error, and reconstruction can amplify that error. We introduce Decoupled Contrastive Decoding (DCD), which drafts with an expert-aligned lightweight proposer and applies the amateur only in unchanged CD verification. Standard speculative verification preserves the vanilla-CD output distribution. Across the main 8B set
    
[^126]: ViTOED：面向越南社交媒体文本的目标导向情感检测数据集

    ViTOED: A Dataset for Target-Oriented Emotion Detection on Vietnamese Social Media Texts

    [https://arxiv.org/abs/2608.12776](https://arxiv.org/abs/2608.12776)

    该论文提出了一个针对越南社交媒体文本的目标导向情感检测数据集ViTOED，并基于结构化情感图建立了基线模型，揭示了越南语言中的特有挑战。

    

    本文介绍了ViTOED，一个用于越南社交媒体文本中目标导向情感检测的新型数据集。ViTOED包含10,985条用户评论和21,244个手动标注的观点四元组（来源、目标、表达、极性），这些标注遵循严格指南。该数据集揭示了越南特有的现象，如隐含来源和目标以及词汇歧义，从而能够更深入地分析用户对实体的情感。我们提出了一个基于结构化情感图的基线模型，并评估了多种越南预训练语言模型。实证结果突出了跨度检测和关系提取方面的挑战，并表明在越南目标导向情感检测任务中模型仍有很大的改进空间。

    arXiv:2608.12776v1 Announce Type: new  Abstract: This paper introduces ViTOED, a novel dataset for target-oriented emotion detection in Vietnamese social media texts. The ViTOED comprises 10,985 user comments and 21,244 manually annotated opinion quadruples (source, target, expression, polarity) that follow strict guidelines. The dataset reveals Vietnamese-specific phenomena, such as implicit sources and targets and vocabulary ambiguities, enabling deeper analysis of user emotions toward entities. We propose a baseline using structured sentiment graphs and evaluate various Vietnamese pre-trained language models. The empirical results highlight challenges in span detection and relation extraction and indicate substantial room for model improvement in Vietnamese Target-Oriented Emotion Detection tasks.
    
[^127]: 保持简洁：面向超长视频理解的多键情景记忆检索

    Keep It Simple: Multi-Key Episodic Memory Retrieval for Ultra-Long Video Understanding

    [https://arxiv.org/abs/2608.07663](https://arxiv.org/abs/2608.07663)

    提出MERIT框架，在记忆构建阶段采用多键情景表示以保证高召回率的精确检索，并将查询特定的高级关系组合延迟到推理阶段通过时间扩展完成，从而以简洁的方式实现超长视频理解。

    

    当视频时长从数小时延长至数天时，直接进行端到端处理对于当前的多模态大语言模型（MLLM）而言变得不切实际。这种超长场景需要一种两阶段范式：先构建与查询无关的记忆，再进行基于检索的推理。先前的工作投入于复杂的记忆构建，以预先建模视频中的高级关系，尽管在构建时并不知道下游查询是什么。我们反其道而行之，在记忆构建阶段优先保证高召回率的可检索性，并将针对查询的高级关系组合推迟到推理阶段完成。为此，我们提出了MERIT（具有推理时时间扩展的多键情景检索），这是一个简单而有效的用于超长视频理解的智能体框架。首先，我们构建了一种情景式多键表示，通过简单的键匹配机制即可实现对细粒度记忆的精确检索。其次，我们引入了一种相邻……

    arXiv:2608.07663v2 Announce Type: replace-cross  Abstract: When videos extend from hours to days, directly processing them end-to-end becomes impractical for current Multi-modal Large Language Models (MLLMs). This ultra-long setting necessitates a two-stage paradigm: query-agnostic memory construction followed by retrieval-based inference. Prior work invests in complex memory construction to pre-model high-level relations in videos, despite not knowing the downstream query at build time. We instead prioritize high-recall retrievability during memory building, and defer query-specific, high-level relation composition to inference time. To this end, we propose MERIT(Multi-key Episodic Retrieval with Inference-time Temporal expansion), a simple yet effective agentic framework for ultra-long video understanding. First, we formulate an episodic multi-key representation that enables precise retrieval of fine-grained memories through a simple key-matching mechanism. Second, we introduce a nei
    
[^128]: 当自我进化适得其反：针对大语言模型智能体技能污染的预先承诺门控机制

    When Self-Evolution Backfires: Pre-Commit Gating against Skill Contamination in LLM Agents

    [https://arxiv.org/abs/2608.05810](https://arxiv.org/abs/2608.05810)

    本文揭示了大语言模型智能体自我进化中的技能污染相变现象且该污染在结构上不可逆，提出VaG（验证者即守门人）机制，通过渐进式信任层级的预先承诺式技能准入门控，在缺陷技能进入决策上下文前予以拦截。

    

    arXiv:2608.05810v2 公告类型： replace 摘要：自我进化智能体通过从执行轨迹中蒸馏可复用技能来积累能力，但我们发现这一过程并非单调递增：一旦超过某个关键的技能池规模，新添加的技能反而会降低而非提升性能。我们对这种“能力污染相变”进行了形式化，并将其追溯到一个结构性原因：一旦缺陷技能进入决策上下文，它就会成为后续技能蒸馏的参考材料，从而形成跨轮次的污染链。我们进一步证明这种污染在结构上是不可逆的：事后移除源技能无法消除其后代技能已经继承的缺陷推理，因此事后回滚只能恢复一小部分损失的性能。这使得技能准入成为一种“预先承诺”层面的必要措施，而非事后补救手段，并由此提出了验证者即守门人机制：一个渐进式信任层级，其包含三个异构评审器——结构验证……（原文摘要在此处截断）

    arXiv:2608.05810v2 Announce Type: replace  Abstract: Self-evolving agents accumulate capability by distilling reusable skills from their execution trajectories, but we find this process is not monotonic: past a critical pool size, newly added skills degrade performance instead of improving it. We formalize this capability-contamination phase transition and trace it to a structural cause: once a defective skill enters the decision context, it becomes reference material for distilling later skills, forming cross-round contamination chains. We further show the contamination is structurally irreversible: removing a source skill after the fact cannot erase the flawed reasoning its descendants have already inherited, so post-hoc rollback recovers only a small fraction of the lost performance. This makes skill admission a pre-commit necessity rather than a post-hoc fix, and motivates Verifier-as-Gatekeeper (VaG): a progressive trust hierarchy whose three heterogeneous critics - structural val
    
[^129]: MyMentorLLM：一个面向刻意练习的多模态语音/文本患者、学员与专家心理治疗生成式AI环境

    MyMentorLLM: A psychotherapy GenAI environment with multimodal voice/text patients, trainees and experts for deliberate practice

    [https://arxiv.org/abs/2607.25667](https://arxiv.org/abs/2607.25667)

    提出了MyMentorLLM——一个包含2,100次完整CBT会谈的多模态心理治疗刻意练习环境，其中LLM模拟患者、受训治疗师与专家督导三方互动，实验表明模拟患者情感表现与真实障碍一致，且LLM学员的治疗能力在多数条件下超过人类水平。

    

    arXiv:2607.25667v2 公告类型：replace-cross 摘要：心理治疗师需要反复的训练与督导，然而其可扩展性存在问题。我们提出了MyMentorLLM，这是一个基于多模态语音和文本的刻意练习环境，包含2,100次完整的认知行为疗法（CBT）会谈。每次会谈将一个基于DSM-5-TR的LLM模拟患者（患有重度抑郁症、广泛性焦虑障碍或边缘型人格障碍）、一个LLM受训治疗师和一个LLM专家督导（分别由Gemma-4、Gemini-3.1-Flash-Live和Qwen-3.6驱动）联系起来。研究者对照人类心理治疗数据，从情感动态、治疗能力和诊断准确性三个方面对会谈进行了分析。结果显示：模拟患者表现出与所患障碍一致的情感特征，治疗师则像人类咨询中那样对患者的情绪产生共情映照；在大多数实验条件下，LLM受训治疗师的能力被评为高于人类水平，其中原生语音对语音的模式最接近人类评分；督导反馈在7个LLM配置中的5个里提升了诊断准确性。

    arXiv:2607.25667v2 Announce Type: replace-cross  Abstract: Psychotherapists need repeated training and supervision; however, scalability is problematic. We present MyMentorLLM, a multimodal voice- and text-based deliberate-practice environment with 2,100 complete Cognitive Behavioural Therapy (CBT) sessions. Each session links a DSM-5-TR-grounded LLM patient (with major depressive, generalised anxiety or borderline personality disorder), an LLM therapist-in-training and an LLM expert supervisor (powered by Gemma-4, Gemini-3.1-Flash-Live and Qwen-3.6). Sessions were analysed for emotional dynamics, therapeutic competence and diagnostic accuracy against human psychotherapy data. Simulated patients expressed disorder-congruent emotional profiles, which therapists mirrored as in human counselling. LLM trainee competence was rated above human levels in most conditions, while native speech-to-speech was closest to human scores. Supervisor feedback improved diagnostic accuracy in 5 of 7 LLM c
    
[^130]: Molt：一个面向智能体强化学习的可扩展 PyTorch 原生训练框架

    Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2607.21653](https://arxiv.org/abs/2607.21653)

    提出了 Molt 框架，通过可组合模型并行、统一智能体接口、全异步 rollout 与优化以及分布式经验存储，实现了万亿参数规模下的智能体强化学习训练，且无需修改现有智能体的执行逻辑。

    

    智能体强化学习需要一种研究者能够在不牺牲模型规模或智能体执行控制权的前提下进行修改的基础设施。我们提出了 Molt，一个轻量级的 PyTorch 原生框架，将万亿参数规模的训练与标准智能体接口相结合。Molt 整合了四项能力：基于可组合模型并行的紧凑训练实现；统一的 OpenAI 和 Anthropic 接口，支持上下文压缩后的自动轨迹分段；完全异步的 rollout 与优化；以及面向长多模态轨迹的分布式经验存储。现有智能体可以保留其执行与上下文管理逻辑，同时由共享捕获层记录生成的 token 和行为概率。Rollout 工作进程将大体积的经验数据放入 Ray 的对象存储中，训练器进程通过引用检索分配给它的经验，从而避免对完整 rollout 进行集中式收集。

    arXiv:2607.21653v2 Announce Type: replace-cross  Abstract: Agentic reinforcement learning requires infrastructure that researchers can modify without sacrificing model scale or control over agent execution. We present Molt, a lightweight PyTorch-native framework that combines trillion-parameter training with standard agent interfaces. Molt integrates four capabilities: a compact training implementation built on composable model parallelism; unified OpenAI and Anthropic interfaces with automatic trajectory segmentation after context compaction; fully asynchronous rollout and optimization; and distributed experience storage for long, multimodal trajectories. Existing agents retain their execution and context-management logic while a shared capture layer records generated tokens and behavior probabilities. Rollout workers place heavy experience payloads in Ray's object store, and trainer ranks retrieve their assigned experiences by reference, avoiding a centralized gather of the full roll
    
[^131]: 针对自托管AI代理的自状态攻击：操作系统防御能走多远？

    Self-State Attacks on Self-Hosted AI Agents: How Far Can OS Defenses Go?

    [https://arxiv.org/abs/2607.17986](https://arxiv.org/abs/2607.17986)

    该论文首次形式化了针对自托管AI代理的“自状态攻击”空间，并通过系统评估证明现有操作系统防御机制存在根本性局限——文件级控制要么留下替代攻击路径、要么误伤合法更新，检测器要么大量误报、要么覆盖不全。

    

    自托管的AI代理会维护持久化的记忆、指令和配置，这些内容会影响其未来的行为。如果代理被攻陷，攻击者可以利用代理的合法写权限来破坏其自状态，使得在操作系统（OS）层面难以区分恶意更新与良性更新。我们研究了现有OS机制能在多大程度上预防、检测并从这类自状态攻击中恢复。我们形式化了该攻击空间，并使用四个代理工作负载和一个Linux遥测管道评估了代表性的OS防御机制。我们的结果表明，各防御维度均存在一致性的局限：文件级控制要么留下替代的篡改路径，要么在完整覆盖所测试操作的同时也会阻止相应的合法更新。检测器会将相当一部分合法活动标记为可疑，而更具选择性的检测方法则只能覆盖攻击空间的一部分。最后，受保护的（摘要在此处截断）

    arXiv:2607.17986v2 Announce Type: replace-cross  Abstract: Self-hosted AI agents maintain persistent memory, instructions, and configuration that influence their future behavior. If an agent is compromised, an attacker can exploit the agent's legitimate write permissions to corrupt this self-state, making malicious and benign updates difficult to distinguish at the operating system (OS) level. We investigate how far existing OS mechanisms can prevent, detect, and recover from such self-state attacks. We formalize an attack space and evaluate representative OS defenses using four agent workloads and a Linux telemetry pipeline. Our results show a consistent limitation across defense dimensions. File-level controls either leave alternative mutation paths open or, when complete over the tested operations, also block corresponding legitimate updates. Detectors flag a substantial part of legitimate activity, while more selective methods cover only part of the attack space. Finally, protected
    
[^132]: 语言模型中可靠性与规模化的极限

    Limits of Reliability and Scaling in Language Models

    [https://arxiv.org/abs/2607.14112](https://arxiv.org/abs/2607.14112)

    该论文从信息论第一性原理证明了每个生成任务都存在不可逾越的可靠性上限，并推导出一条统一的规模化定律——LLM性能的瓶颈由训练数据与模型容量中更稀缺的资源决定，且Chinchilla定律成为其特例。

    

    大型语言模型（LLMs）在训练和评估时都默认假设：只要规模足够大，任何任务都能达到完美的可靠性。我们证明这一假设在信息论上是缺乏依据的。每个生成任务都存在一个任何模型都无法超越的可靠性上限，该上限由可观测上下文能够解决多少输出不确定性所决定。这一差距可分解为两部分：一个是可通过增加上下文来弥补的可解决分量，另一个是任务模糊性所固有的主观分量。自回归生成还会进一步降低这一上限，其衰减速率由任务的依赖核决定，该依赖核量化了输出中词符间的相关性。基于这两个基本量，我们从第一性原理推导出一条规模化定律，其中LLM的性能受限于更稀缺的资源：训练数据或模型容量。该定律将Chinchilla规模化定律作为特例囊括其中，并为何时扩大规模才能带来收益提供了结构性的解释。

    arXiv:2607.14112v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are trained and evaluated as though perfect reliability is achievable for any task given sufficient scale. We show that this assumption is information-theoretically unjustified. Every generative task has a reliability ceiling that no model can exceed, determined by how much output uncertainty is resolvable from observable context. The gap decomposes into a resolvable component closable with additional context and a subjective component inherent to task ambiguity. Autoregressive generation further degrades this ceiling at a rate governed by the task's dependency kernel, which quantifies inter-token correlations in the output. From these two primitives, we derive a first-principles scaling law where LLM performance is bottlenecked by the scarcer resource: training data or model capacity. This law recovers the Chinchilla scaling law as a special case and provides a structural account of when scaling im
    
[^133]: CORTEX：基于本体语料库图的网络规模语料库高质量跨领域组织

    CORTEX: High-Quality Cross-Domain Organization of Web-Scale Corpora through Ontological Corpus Graph

    [https://arxiv.org/abs/2606.30175](https://arxiv.org/abs/2606.30175)

    该论文提出了Cortex框架，通过本体语料库图（OCG）这一三层异构结构，首次将网络规模语料库构建从扁平文档筛选提升为结构化知识组织，实现了高质量的跨领域语料组织。

    

    大语言模型的持续发展推动了对数据规模和质量日益增长的需求，随着不同训练阶段对数据提出越来越定制化的要求，高质量语料库的系统化组织变得不可或缺。现有的语料库构建流程将所产出的语料局限于扁平、无差异的文档集合，普遍缺乏系统性的知识组织。我们提出了Cortex，据我们所知，这是首个通过网络规模语料库构建将扁平文档筛选提升为结构化知识组织的框架，其核心是本体语料库图（Ontological Corpus Graph, OCG）——一个三层异构结构，统一了质量精炼的内容层、通过大语言模型驱动的自动演化实现层级化轻量本体层、以及能够在任意分类分辨率下实现跨领域关联的跨领域对齐层。全面的实验证实了该方法的有效性。

    arXiv:2606.30175v2 Announce Type: replace  Abstract: The continuous evolution of large language models drives escalating demands on data scale and quality, and as different training stages impose increasingly tailored data requirements, systematic organization of high-quality corpora becomes indispensable. Existing corpus construction pipelines confine the resulting corpora to flat, undifferentiated document collections, universally lacking systematic knowledge organization. We present Cortex, to our knowledge the first framework that elevates web-scale corpus construction from flat document filtering to structured knowledge organization through an Ontological Corpus Graph (OCG), a three-layer heterogeneous structure unifying a quality-refined content layer, a hierarchical lightweight ontology layer via LLM-driven automated evolution, and a cross-domain alignment layer enabling inter-domain association at arbitrary taxonomic resolution. Comprehensive experiments confirm the effectivene
    
[^134]: IHDec：面向多轮指令层级安全的散度引导对比解码

    IHDec: Divergence-Steered Contrastive Decoding for Securing Multi-Turn Instruction Hierarchies

    [https://arxiv.org/abs/2606.29960](https://arxiv.org/abs/2606.29960)

    提出无需训练的IHDec方法，利用Jensen-Shannon散度自动检测多轮对话中的词元级指令层级违规，并通过动态对比解码抑制低优先级角色的影响，在多轮指令冲突中性能超越基于训练的基线方法。

    

    大语言模型（LLM）在处理具有不同角色级优先级的多源输入时，往往无法维持指令层级，在冲突中反而遵循低优先级指令。尽管现有防御措施能够缓解这一问题，但它们大多局限于单轮场景，且需要昂贵的微调。本文通过Jensen-Shannon散度（JSD）框架将这一失效模式在多轮场景中形式化，揭示了一种普遍存在的角色影响倒置现象，即低级别输入会覆盖高级别角色。为在不训练的情况下纠正这一问题，我们提出了IHDec（指令层级引导解码）。IHDec利用JSD自动检测词元级别的层级违规，并动态执行对比解码以抑制错位的低级别角色。大量评估表明，IHDec在多轮冲突中优于基于训练的基线方法。

    arXiv:2606.29960v2 Announce Type: replace  Abstract: Large Language Models (LLMs) often fail to maintain instruction hierarchies (IH) when processing multi-source inputs with varying role-level priorities, paradoxically adhering to lower-priority directives during conflicts. While existing defenses mitigate this issue, they are largely restricted to single-turn scenarios and require expensive fine-tuning. In this paper, we formalize this failure mode in multi-turn contexts via a Jensen-Shannon Divergence (JSD) framework, uncovering a pervasive role-influence inversion phenomenon where subordinate inputs override superior roles. To rectify this without training, we propose IHDec (Instruction Hierarchy-steered Decoding). IHDec leverages JSD to automatically detect token-level hierarchy violations and dynamically executes contrastive decoding to suppress misaligned subordinate roles. Extensive evaluations demonstrate that IHDec outperforms training-based baselines in multi-turn conflicts 
    
[^135]: 当检索指标产生误导时：测量长程工具使用智能体中的策略信号

    When Retrieval Metrics Mislead: Measuring Policy Signal in Long-Horizon Tool-Use Agents

    [https://arxiv.org/abs/2606.23937](https://arxiv.org/abs/2606.23937)

    该研究发现精确匹配检索召回率是一个具有误导性的代理指标——即使正确的治理规则仅在 7% 的情况下被排名第一检索到，检索到的断言仍能让分类器取得与使用黄金规则几乎相同的性能。

    

    精确匹配检索召回率常被用作衡量检索器是否为下游决策模型提供有用策略上下文的代理指标。我们在 τ-bench 中使用 Qwen2.5-3B/7B 分类器，针对动作前策略分类任务测试了这一代理指标。在黄金策略条件下，经过调优的紧凑结构化状态在 3B 模型上比原始轨迹的 macro-F1 提高了 0.20，在共享超参数下 7B 模型也呈现相同的排序关系。随后，我们将基准指定的治理规则替换为从决策时上下文中检索到的排名第一的基准断言。尽管精确的治理规则仅在 7% 的航空领域状态中被检索到排名第一，但主要的 3B 分类器使用检索到的断言获得了 0.58 的 macro-F1，而使用黄金规则为 0.60（Δ=-0.02，任务簇 95% 置信区间 [-0.23,+0.21]）；作为对照，随机非黄金断言和无断言条件的得分分别为 0.32 和 0.21。我们没有检测到……（摘要在此处截断）

    arXiv:2606.23937v2 Announce Type: replace-cross  Abstract: Exact-match retrieval recall is often used as a proxy for whether a retriever supplies useful policy context to a downstream decision model. We test this proxy for pre-action policy classification in $\tau$-bench using Qwen2.5-3B/7B classifiers. Under gold-policy conditioning, a compact structured state improves macro-F1 over raw trajectories by $0.20$ after tuning at 3B, with the same ordering at 7B under shared hyperparameters. We then replace the benchmark-designated governing rule with the top-ranked benchmark assertion retrieved from decision-time context. Although the exact governing rule is retrieved at rank 1 for only $7\%$ of airline states, the primary 3B classifier obtains macro-F1 $0.58$ with retrieved assertions versus $0.60$ with the gold rule ($\Delta=-0.02$, task-cluster 95\% CI $[-0.23,+0.21]$); random non-gold and no-assertion controls score $0.32$ and $0.21$. We do not detect a macro-F1 difference between ret
    
[^136]: 大语言模型归因指标具有可迁移性吗？跨数据集与评估构念的检索增强生成评估审计

    Do LLM Attribution Metrics Transfer? Auditing Retrieval-Augmented Generation Evaluation Across Datasets and Constructs

    [https://arxiv.org/abs/2606.23915](https://arxiv.org/abs/2606.23915)

    该研究系统审计了八种检索增强生成归因自动指标，发现在生成答案归因构念下没有任何指标能在所有数据集上保持与最佳指标一致的性能，不同数据集上的指标排名甚至完全反转，因此实践中将这些指标视为可互换是不成立的。

    

    实践中通常将用于大语言模型检索增强生成中归因评估的自动指标视为可以互相替换的。我们审计了八种自动评分器——词汇、嵌入和 BERTScore 基线，以及经过蕴含/接地训练的模型（干净版和 FEVER 版 NLI、检查工具 MiniCheck）——横跨三种评估构念（来源与主题相关性、生成答案归因、事实核查蕴含），探究是否存在任何评分器能够“迁移”：即在多数据集构念的每个数据集上都保持在最佳被审计评分器的 95% 置信区间之内。在拥有最多多数据集人工标注覆盖的构念——生成答案归因（AttributionBench 的四个源数据集，n = 1,610，以及独立的 HAGRID，n = 2,150）中，没有任何被审计的自动评分器能做到这一点：各数据集上的指标排名会发生反转（在 AttributedQA 与 LFQA 对比中 Kendall tau = -0.64，p = 0.031），且在短声明上表现最佳的现成 NLI 评分器（原文此处截断）

    arXiv:2606.23915v2 Announce Type: replace  Abstract: Practice often treats automatic metrics for attribution in LLM retrieval-augmented generation as interchangeable. We audit eight automatic scorers -- lexical, embedding, and BERTScore baselines alongside entailment/grounding-trained models (clean and FEVER NLI, the checker MiniCheck) -- across three evaluation constructs (provenance/topicality, generated-answer attribution, and fact-check entailment), asking whether any scorer transfers: stays within the 95% confidence interval of the best audited scorer on every dataset of a multi-dataset construct. In the construct with the most multi-dataset human-labeled coverage -- generated-answer attribution (AttributionBench's four source datasets, n = 1,610, with independent HAGRID, n = 2,150) -- none of the audited automatic scorers does: the per-dataset metric rankings invert (Kendall tau = -0.64, p = 0.031 on AttributedQA vs. LFQA), and an off-the-shelf NLI scorer that is best on short-cl
    
[^137]: 删改还是保留？一种用于教育对话去标识化的完全本地AI级联框架

    Redact or Keep? A Fully Local AI Cascade for Educational Dialogue De-Identification

    [https://arxiv.org/abs/2606.18372](https://arxiv.org/abs/2606.18372)

    提出一种完全本地运行的AI级联框架，将教育对话去标识化重新定义为“删改/保留”的受限隐私分诊任务，无需将学生数据发送给第三方，即可解决商用LLM与本地NER系统在隐私治理与识别准确性之间难以兼得的权衡问题。

    

    教育对话是一种有价值但对研究而言敏感的资源：捕捉真实学习过程的对话记录，往往也同时捕捉到了与课程内容交织在一起的个人可识别信息（PII），例如"Riemann"（黎曼）既可能指代一位真实的学生，也可能指代一个数学概念。现有方法迫使研究者在数据治理与准确性之间做出权衡：商用大语言模型（LLM）能够处理这种歧义，但需要将学生数据发送给第三方；而本地命名实体识别（NER）系统虽然保证了数据治理，却会过度删改课程术语。我们提出了一种完全本地的级联框架，将去标识化问题从开放式实体识别重新定义为受限的隐私分诊任务。一个以召回为先的联合提议器将两个轻量级编码器与确定性规则相结合，过度生成候选文本片段；随后由一个具备上下文感知能力的审查器利用周围上下文信息，对每个候选片段做出“删改/保留”的二元决策。（原文摘要在此处截断）

    arXiv:2606.18372v2 Announce Type: replace-cross  Abstract: Educational dialogue is a valuable but sensitive resource for research: the same transcripts that capture authentic learning often capture personally identifiable information (PII) entangled with curricular content, where "Riemann" may refer to a real student or to a mathematical concept. Existing approaches force a tradeoff between governance and accuracy. Commercial Large Language Models (LLMs) can handle this ambiguity but require sending student data to third parties, while local named entity recognition (NER) systems preserve governance but over-redact curricular terms. We propose a fully local cascade framework that reframes de-identification from open-ended entity recognition to constrained privacy triage. A recall-first union proposer combines two lightweight encoders with deterministic rules to over-generate candidate spans; a context-aware reviewer then makes a binary Redact/Keep decision for each candidate using surr
    
[^138]: 评估基于音素的自动语音识别系统中的偏差：IPA转写模型分析

    Evaluating Bias in Phoneme-Based Automatic Speech Recognition Systems: An Analysis of IPA Transcription Models

    [https://arxiv.org/abs/2606.11639](https://arxiv.org/abs/2606.11639)

    本研究首次针对基于音素的自动语音识别系统评估人口统计学偏差，分析了两个最先进的开源IPA转写系统（WhisperIPA和ZIPA），并提出了能够容忍语言学上相似音素替换的Soft PER评估指标。

    

    随着自动语音识别（ASR）系统向多语言支持和低资源语言建模方向发展，基于音素的层作为与语言无关的关键基础显得至关重要。然而，大多数关于ASR在种族、年龄、性别和口音等人口统计学偏差方面的评估，主要集中在标准的基于字形的ASR系统上，对基于音素的系统关注相对较少。在本研究中，我们评估了WhisperIPA和ZIPA这两种最先进的开源国际音标（IPA）转写系统的性能。我们的评估包括现有的多语言语音语料库和带有人口统计学标注的英语语料库，将模型生成的IPA转写与字形到音素（G2P）系统进行比较，同时使用标准的音素错误率（PER）和我们提出的Soft PER指标，该指标能够容忍语言学上相似的音素替换。我们的分析研究了性能差异如何……（原文在此处截断）

    arXiv:2606.11639v2 Announce Type: replace  Abstract: As automatic speech recognition (ASR) systems shift toward multilingual support and low-resource language modeling, phoneme-based layers serve as a critical language-agnostic foundation. However, most evaluations of ASR's demographic biases related to race, age, gender, and accent focus on standard grapheme-based ASR systems with comparatively little emphasis on phoneme-based systems. In this study, we evaluate the performance of WhisperIPA and ZIPA, two state-of-the-art open-source systems that generate International Phonetic Alphabet (IPA) transcriptions. Our evaluation includes existing multilingual speech corpora and demographically annotated English-language corpora, comparing model-generated IPA transcriptions against grapheme-to-phoneme (G2P) systems using both standard phoneme error rate (PER) and a proposed Soft PER metric that tolerates linguistically similar phoneme substitutions. Our analysis examines how performance vari
    
[^139]: 数据记者智能体：将数据转化为可验证的多模态故事

    Data Journalist Agent: Transforming Data into Verifiable Multimodal Stories

    [https://arxiv.org/abs/2606.11176](https://arxiv.org/abs/2606.11176)

    提出了 Data2Story 多智能体框架，将专业化角色编排为虚拟新闻编辑室，实现证据可追溯、按需生成多模态内容的端到端自动化数据新闻写作。

    

    数据讲述着塑造社会的故事；数据记者的职责是将原始信息转化为非专业人士可以信任的故事。一篇高质量的新闻专题报道通常需要新闻编辑室团队花费数周时间：寻找背景信息、进行统计分析、选择报道角度以及设计视觉呈现。近期的智能体能够较好地处理单个步骤：数据科学智能体可以完成分析闭环，而设计智能体能够合成精美的网页。但是，智能体能否端到端地胜任数据记者的工作？我们提出了数据记者智能体，这是一个多智能体框架，将多个专业化角色编排到一个虚拟新闻编辑室中。Data2Story 做出了两项创新：其一，论述均有证据支撑，由一个“检查员”将每个数字、报道角度和素材都关联到数据、代码或外部参考资料；其二，文章是多模态生成的，Data2Story 不会默认生成纯文本和静态图表，而是推理读者希望看到什么内容，进而部署相应的呈现形式。

    arXiv:2606.11176v2 Announce Type: replace-cross  Abstract: Data tells stories that shape society; the data journalist's job is to turn raw information into stories non-experts can trust. A high-quality news feature takes a newsroom team weeks: hunting for context, running statistics, choosing an angle, and designing visuals. Recent agents handle individual steps well: data-science agents close the analysis loop, while design agents synthesize beautiful websites. But can an agent serve as a data journalist end to end? We introduce Data Journalist Agent (Data2Story), a multi-agent framework that orchestrates specialized roles into a single virtual newsroom. Data2Story contributes two innovations. (i) Claims are evidence-grounded: an Inspector links every number, angle, and asset back to data, code, or an external reference. (ii) Articles are multimodally generative: rather than defaulting to plain text and static charts, Data2Story reasons about what readers will want to see, then deploy
    
[^140]: 中立的面具：对齐训练如何在使大语言模型保持党派结构完整的同时仅提供浅层对齐

    The Neutral Mask: How Alignment Training Provides Shallow Alignment while Leaving Partisan Structure Intact in a Large Language Model

    [https://arxiv.org/abs/2606.09735](https://arxiv.org/abs/2606.09735)

    该研究通过对Llama 3.1 8B对齐前后内部表征的机制性分析，发现对齐训练并未消除模型内部结构化的党派政治倾向，而是通过压缩党派信号的方差使输出表面上保持中立平衡，仅实现了浅层对齐。

    

    对齐训练的初衷是让大语言模型变得安全且有用。其主要机制——基于人类反馈的强化学习（RLHF）及其直接优化变体——通过将部署的语言模型与“人类价值观”对齐来塑造其行为。然而这一过程是不透明的：究竟编码了哪些价值观？这些价值观属于谁？对齐训练又是如何将其编码的？越来越多的证据表明，这些方法仅产生功能性顺从而非深度对齐。我们以党派政治倾向为对象，对这一现象进行了机制层面的案例研究，对比了Llama 3.1 8B在对齐训练前后的内部表征。我们证明，对齐训练并未移除基座模型中结构化的党派方向；相反，它压缩了党派信号的方差，从而生成持续平衡、无党派倾向的输出。

    arXiv:2606.09735v2 Announce Type: replace  Abstract: The ambition behind alignment training is to make large language models safe and useful. The primary mechanisms, reinforcement learning from human feedback (RLHF) and its direct-optimization variants, shape the behavior of deployed language models by aligning them with ``human values.'' Yet the process is opaque. What values are being encoded; whose values are they; and how does alignment training encode them? A growing body of evidence suggests that these methods produce only functional compliance rather than deep alignment. We offer a mechanistic case study of this phenomenon for partisan political orientation with a comparison of the internal representations of Llama 3.1 8B before and after alignment training. We show that alignment training does not remove the structured partisan direction in the base model. Instead, it compresses the variance of the partisan signal to generate consistently balanced and non-partisan output. Spars
    
[^141]: 选择条件下的测量：面向语言模型的诱饵校准失效审计

    Measurement Under Selection: Decoy-Calibrated Failure Audits for Language Models

    [https://arxiv.org/abs/2606.09046](https://arxiv.org/abs/2606.09046)

    本文提出Janus框架，通过随机打乱属性标签生成的“诱饵”来校准阈值，只有超出偶然波动水平并在保留数据上稳健成立的错误模式才被报告，从而避免语言模型失效审计中出现虚假的模式发现。

    

    知道语言模型失败的频率并不能解释其错误集中在哪里。当审计人员检查许多可能的解释时，观察到的最强模式可能只是偶然产生的。我们提出了Janus，一种在报告之前对拟议错误模式进行检验的程序。Janus从被评估示例的一组固定的“是/否”属性列表开始，例如输入是否很长。对于每个属性，它比较模型在具有该属性的示例与没有该属性的示例上的错误率。为了了解偶然情况下可能产生多大的差异，它在打乱示例间的“是/否”标签（但不改变各组规模）之后重复这一计算。这些打乱后的属性被称为“诱饵”。只有当某个模式的错误差异大小达到由诱饵设定的阈值时，该模式才会被报告。在单独的保留示例上，同一组仍必须具有更高的错误率，且差异必须达到一个最低标准……

    arXiv:2606.09046v2 Announce Type: replace-cross  Abstract: Knowing how often a language model fails does not explain where its errors concentrate. When auditors examine many explanations, the strongest observed pattern may arise by chance. We introduce Janus, a procedure for checking proposed error patterns before reporting them. Janus starts with a fixed list of yes/no properties of the examples being evaluated, such as whether the input is long. For each property, it compares the model's error rates on examples with that property and those without it. To see how large a difference can arise by chance, it repeats this calculation after shuffling the yes/no labels across examples without changing the group sizes. These shuffled properties are called decoys. A pattern is reported only if the size of its error difference meets a threshold set using decoys. On separate held-out examples, the same group must still have the higher error rate and the difference must meet a minimum, which was
    
[^142]: LaSR：基于潜在推理的上下文感知语音识别

    LaSR: Context-Aware Speech Recognition via Latent Reasoning

    [https://arxiv.org/abs/2606.00507](https://arxiv.org/abs/2606.00507)

    LaSR提出了一种利用潜在推理的上下文感知语音识别新训练范式，通过在目标词声学特征区域周围对齐思维链监督并引入潜在推理阶段实现上下文信息落地，同时发布了聚焦学术术语的大规模语料库Spoken Darwin-Science。

    

    专用领域的语音识别需要利用上下文或主题信息来提升领域特定实体的识别效果。语音大语言模型极大地推进了语音理解与推理能力，使得无需预定义偏置列表的上下文感知语音识别成为可能。在本文中，我们提出了LaSR（潜在语音推理），这是一种新颖的训练范式，其特点是利用潜在推理过程的上下文感知推理轨迹。LaSR不再生成显式的中间token，而是在目标词的声学特征区域周围对齐思维链（CoT）监督，并引入潜在推理阶段用于上下文信息的落地和转录转换。此外，为了有效评估上下文感知语音识别，我们提出了Spoken Darwin-Science，一个专注于学术术语的大规模语料库。

    arXiv:2606.00507v2 Announce Type: replace  Abstract: Speech recognition in specialized domains requires leveraging contextual or topical information to improve the recognition of domain-specific entities. Speech Large Language Models (Speech LLMs) have substantially advanced speech understanding and reasoning capabilities, making context-aware speech recognition possible without predefined bias lists. In this paper, we propose LaSR (Latent Speech Reasoning), a novel training paradigm featuring a context-aware reasoning trajectory that leverages the latent reasoning process. Instead of generating explicit intermediate tokens, LaSR aligns chain-of-thought (CoT) supervision around the acoustic feature region of the target word, and introduces latent reasoning periods for context information grounding and transcriptional transition. Furthermore, to effectively benchmark context-aware speech recognition, we propose Spoken Darwin-Science, a large-scale corpus focusing on academic terminologi
    
[^143]: 凭其果实识其树：通过所编码的判决来比较法律形式化

    By Their Fruits You Will Know Them: Comparing Formalizations of Law by the Decisions They Encode

    [https://arxiv.org/abs/2605.25186](https://arxiv.org/abs/2605.25186)

    提出一种基于SAT求解器的方法，通过枚举同一法律条文的不同形式化在具体边界案例上产生分歧的行为来系统比较它们，从而揭示大语言模型生成的法律形式化中难以预料的隐含解释性选择。

    

    将法律条文形式化有望实现机器可读的法律和自动化法律推理，而近期的大语言模型使人们倾向于直接从法条文本生成此类形式化表示。然而，任何形式化都会做出隐含的解释性选择，其后果难以预料，尤其是当作者为大语言模型时更是如此。我们提出了一种方法，通过形式化在具体个案上的推理来系统地比较同一法律条文的不同形式化。给定同一条文的多份形式化，我们在节点层面对其进行匹配，从匹配结果中为每一对形式化推导出一个共享接口，并使用SAT求解器枚举任意两个形式化产生分歧的边界案例。随后将选定的边界案例转化为具体的事实场景，供法律专家审查并据此采取行动。我们将该方法应用于由九个前沿大语言模型生成的十条欧盟法律条文的形式化。我们发现行为上的分歧（摘要原文在此处截断）。

    arXiv:2605.25186v2 Announce Type: replace-cross  Abstract: Formalizing legal provisions promises machine-accessible law and automated legal reasoning, and recent LLMs make it tempting to generate such formalizations directly from statutory text. However, any formalization makes implicit interpretive choices whose consequences are hard to anticipate, especially if an LLM is the author. We present a method for systematically comparing different formalizations of the same legal provision by their inferences on individual cases. Given multiple formalizations of a provision, we match them at the node level, derive a shared interface for each pair from the matching, and use a SAT solver to enumerate the edge cases on which any two formalizations disagree. Selected edge cases are then verbalized into concrete factual scenarios that a legal expert can examine and act on. We apply our method to formalizations of ten EU provisions generated by nine frontier LLMs. We find that behavioral divergen
    
[^144]: 轰鸣声如何登上报摊：对德国新闻中世界各地山体滑坡报道与空间偏差的数据分析

    How Loud Rumbles Hit Newsstands: A Data Analysis of Coverage and Spatial Bias in German News about Landslides Around the World

    [https://arxiv.org/abs/2605.18105](https://arxiv.org/abs/2605.18105)

    本文通过分析25年间近5.5万篇关于4500起山体滑坡事件的德国新闻报道，揭示了德国媒体报道存在空间偏差，例如对南欧和西欧地区的灾害事件存在过度报道。

    

    山体滑坡因其破坏性和潜在的致命影响而经常登上新闻报摊。新闻是创建或丰富灾害数据库以及推进基于媒体的媒体关注度动态研究的宝贵信息来源。为实现这一目标，新闻数据集必须经过过滤、地理定位和验证。本文聚焦于德国报纸如何报道世界各地的山体滑坡事件。我们分析了25年间关于4500个新闻事件的近5.5万篇新闻文章，并将其与各国土体滑坡易感性的外部衡量指标进行比较，从而提供了相关见解，例如德国媒体对南欧和西欧灾害事件的过度报道，以促进对国际灾害媒体关注度不平等问题的进一步研究。

    arXiv:2605.18105v3 Announce Type: replace  Abstract: Landslides often hit newsstands due to their destructive and potentially fatal effects. News are a valuable source of information for creating or enriching disaster databases and for expediting media-based studies of the dynamics of media attention. To accomplish that, news datasets must be filtered, geolocated and validated. This paper focuses on how landslides around the world are reported in German newspapers. We analyse almost 55k news articles about 4.5k news events in a 25-year period, compare it with external measures of countries' susceptibility to landslides and provide insights, e.g. the overreporting of Southern and Western Europe, to foster further studies on inequalities in media attention to international disasters.
    
[^145]: PersonalAI 2.0：通过规划机制增强个性化LLM智能体的知识图谱遍历与检索

    PersonalAI 2.0: Enhancing knowledge graph traversal/retrieval with planning mechanism for Personalized LLM Agents

    [https://arxiv.org/abs/2605.13481](https://arxiv.org/abs/2605.13481)

    PersonalAI 2.0通过引入动态多阶段查询处理流水线的规划机制，实现了由实体、图顶点和线索查询引导的自适应迭代式知识图谱检索，在多个问答基准上显著提升了生成答案的事实准确性。

    

    我们提出了PersonalAI 2.0（PAI-2），这是一个新颖的框架，旨在通过整合外部知识图谱（KG）来增强基于大语言模型（LLM）的系统。所提出的方法通过引入动态的、多阶段的查询处理流水线，解决了现有图谱检索增强生成（GraphRAG）方法的关键局限性。PAI-2设计的核心在于其能够执行自适应的、迭代式的信息搜索，该搜索由提取的实体、匹配的图顶点以及生成的线索查询所引导。在五个基准数据集（Natural Questions、TriviaQA、HotpotQA、2WikiMultihopQA和MuSiQue）上进行的评估表明，与同类方法（LightRAG、RAPTOR、HippoRAG 2和PAI-1）相比，该方法提升了生成答案的事实准确性。PAI-2在2WikiMultihopQA和MuSiQue基准上通过LLM-as-a-Judge评估平均取得了9%的提升，并在TriviaQA和HotpotQA上达到了与HippoRAG 2相当的准确率。

    arXiv:2605.13481v2 Announce Type: replace  Abstract: We introduce PersonalAI 2.0 (PAI-2), a novel framework designed to enhance LLM-based systems through integration of external knowledge graphs (KGs). The proposed approach addresses key limitations of existing Graph Retrieval-Augmented Generation (GraphRAG) methods by incorporating a dynamic, multistage query-processing pipeline. The central point of the PAI-2 design is its ability to perform adaptive, iterative information search, guided by extracted entities, matched graph vertices, and generated clue-queries. An evaluation conducted on five benchmarks (Natural Questions, TriviaQA, HotpotQA, 2WikiMultihopQA, and MuSiQue) demonstrates an improvement in the factual correctness of generated answers compared to analogue methods (LightRAG, RAPTOR, HippoRAG 2, and PAI-1). PAI-2 achieves a 9% average gain by LLM-as-a-Judge on the 2WikiMultihopQA and MuSiQue benchmarks, and attains accuracy comparable to HippoRAG 2 on the TriviaQA and Hotpo
    
[^146]: 从程序化技能到策略基因：迈向经验驱动的测试时演化

    From Procedural Skills to Strategy Genes: Towards Experience-Driven Test-Time Evolution

    [https://arxiv.org/abs/2604.15097](https://arxiv.org/abs/2604.15097)

    本研究通过45个场景、4590次受控试验发现，紧凑的“策略基因”表示比面向文档的技能包更适合作为可复用经验的载体，在测试时控制与迭代演化中均表现更优，证明经验的表示方式本身是决定性因素。

    

    本贝塔版技术报告探讨了一个问题：可复用的经验应当如何表示，才能既作为有效的测试时控制手段，又作为迭代演化的基础。我们在45个科学代码求解场景中开展了4590次受控试验来研究这一问题。我们发现，面向文档的技能包所提供的控制并不稳定：其有效信号稀疏，而将一个紧凑的经验对象扩展为更完整的文档化包往往无济于事，甚至会降低整体平均表现。我们进一步证明，表示方式本身就是一阶关键因素：紧凑的基因表示能够取得最强的整体平均成绩，在显著的结构扰动下仍保持竞争力，并优于同等预算的技能片段，而重新附加面向文档的材料通常会削弱而非改善其表现。除一次性控制之外，我们还表明基因也是迭代式经验演化的更优载体。

    arXiv:2604.15097v3 Announce Type: replace-cross  Abstract: This beta technical report asks how reusable experience should be represented so that it can function as effective test-time control and as a substrate for iterative evolution. We study this question in 4.590 controlled trials across 45 scientific code-solving scenarios. We find that documentation-oriented Skill packages provide unstable control: their useful signal is sparse, and expanding a compact experience object into a fuller documentation package often fails to help and can degrade the overall average. We further show that representation itself is a first-order factor. A compact Gene representation yields the strongest overall average, remains competitive under substantial structural perturbations, and outperforms matched-budget Skill fragments, while reattaching documentation-oriented material usually weakens rather than improves it. Beyond one-shot control, we show that Gene is also a better carrier for iterative exper
    
[^147]: PolyJarvis：一个由大语言模型编排的智能体，用于无定形均聚物的全原子分子动力学自动化模拟

    PolyJarvis: An LLM-Orchestrated Agent for Automated All-Atom Molecular Dynamics of Amorphous Homopolymers

    [https://arxiv.org/abs/2604.02537](https://arxiv.org/abs/2604.02537)

    PolyJarvis是一个由大语言模型智能体编排的自动化平台，通过规划智能体生成经验证的运行计划并调用EMC和LAMMPS等工具包执行，实现了从重复单元SMILES到无定形均聚物全原子分子动力学模拟及性质计算的端到端自动化。

    

    全原子分子动力学（MD）模拟能够从分子结构预测聚合物性质，但其执行需要力场选择、体系构建、平衡化和性质提取方面的专业知识。我们提出了PolyJarvis，这是一个平台，其中规划智能体生成经过验证的运行计划，由确定性阶段脚本通过成熟的模拟工具包执行——用于体系构建的增强蒙特卡洛（EMC）和用于分子动力学的LAMMPS——这些工具包以模型上下文协议（MCP）服务器的形式暴露，恢复智能体仅在出现结构化故障时且在固定决策预算内才被调用。给定重复单元的SMILES字符串和目标性质，PolyJarvis构建无定形晶胞，在机械化收敛门控下使其达到平衡，并计算目标性质。验证工作在七种无定形均聚物上进行，每种聚合物运行三个重复实验，共享针对每种材料冻结的统一协议……（原文摘要到此被截断）

    arXiv:2604.02537v3 Announce Type: replace  Abstract: All-atom molecular dynamics (MD) simulations can predict polymer properties from molecular structure, yet their execution requires specialized expertise in force field selection, system construction, equilibration, and property extraction. We present PolyJarvis, a platform in which a planning agent produces a validated run plan that deterministic stage scripts execute through established simulation toolkits, Enhanced Monte Carlo (EMC) for system construction and LAMMPS for molecular dynamics, exposed as Model Context Protocol (MCP) servers, with a recovery agent consulted only on structured failures and within a fixed decision budget. Given a repeat-unit SMILES string and target properties, PolyJarvis constructs the amorphous cell, equilibrates it under a mechanized convergence gate, and computes target properties. Validation is conducted on seven amorphous homopolymers, each run as three replicates that share a protocol frozen per s
    
[^148]: 更细的引用总是更好吗？重新思考归因生成中的粒度

    Are Finer Citations Always Better? Rethinking Granularity for Attributed Generation

    [https://arxiv.org/abs/2604.01432](https://arxiv.org/abs/2604.01432)

    该论文通过分析四种模型规模发现，细粒度句子级引用并非总是最优，段落级的中间粒度归因质量最佳，选择最优引用粒度可在几乎不牺牲答案正确性的情况下大幅提升模型性能与归因质量。

    

    引用粒度——即引用单个句子、段落还是整个文档——是归因生成（attributed generation）中的一个关键设计选择。尽管细粒度引用因便于人类进行精确验证而通常受到青睐，但其对模型性能的影响仍未得到充分探索。我们分析了四种模型规模（8B-120B），并证明强制使用细粒度（句子级）引用相对于表现最佳的引用粒度会损失2-97%（中位数40%）的性能增益，在个别任务上损失甚至高达338%。令人惊讶的是，将引用粒度设置为最优值（基于归因质量确定）可以释放这些可观的增益，同时整体答案的正确性基本保持不变（在-2.3%到+4.4%之间）。我们观察到一个一致的模式：归因质量在中间（段落级）粒度处达到峰值——过细的引用似乎切断了支撑论断所需的语义依赖关系，而过粗的引用则……

    arXiv:2604.01432v3 Announce Type: replace  Abstract: Citation granularity -- whether to cite individual sentences, paragraphs, or documents -- is a critical design choice in attributed generation. While fine-grained citations are commonly preferred for precise human verification, their impact on model performance remains under-explored. We analyze four model scales (8B-120B) and demonstrate that enforcing fine-grained (sentence-level) citations forfeits gains of 2-97% (median 40%) relative to the best-performing granularity, and up to 338% on individual tasks. Strikingly, setting citation granularity to its optimal value (based on attribution quality) unlocks these substantial gains while leaving overall answer correctness essentially unchanged (between -2.3% and +4.4%). We observe a consistent pattern where attribution quality peaks at intermediate (paragraph-level) granularities: finer citations appear to sever the semantic dependencies needed to ground a claim, while excessively coa
    
[^149]: CounselReflect：设计工具以支持用户对AI心理健康与福祉对话进行自我反思的机遇与挑战

    CounselReflect: Opportunities and Challenges for Designing Tools to Support Self-Reflection on Mental Health and Well-Being Conversations with AI

    [https://arxiv.org/abs/2603.29429](https://arxiv.org/abs/2603.29429)

    该论文提出了CounselReflect工具，将基于文献的心理咨询质量指标转化为面向用户的反思框架，并通过对21位AI心理健康支持用户的访谈，揭示了工具辅助自我反思的机遇与挑战（如用户存在确认偏误），主张反思工具应帮助用户发现盲点并进行更全面的审视。

    

    AI正越来越多地被用于心理健康与福祉支持，这对更安全的使用方式提出了迫切需求，而相应的设计、评估与治理的发展需要时间。我们探索了一种互补性方法：帮助用户批判性地反思自己与AI的对话。我们介绍了CounselReflect，这是一种将基于文献的心理咨询质量指标转化为面向用户的反思框架的工具。以CounselReflect作为研究探针，我们访谈了21位使用AI进行心理健康与福祉支持的用户。尽管大多数参与者并没有习惯性地反思他们的对话，但他们明确表达了希望反思能够解决的具体问题。工具辅助的反思也揭示了挑战：参与者会选择性地寻求能够证实自己对AI既有看法的证据，并优先关注自己已经重视的维度。我们认为，反思工具应当揭示用户的盲点，并支持更全面的审视。

    arXiv:2603.29429v2 Announce Type: replace  Abstract: AI is increasingly used for mental health and well-being support, creating an urgent need for safer engagement, while design, evaluation, and governance take time to develop. We explore a complementary approach: helping users critically reflect on their own AI conversations. We introduce CounselReflect, a tool that translates literature-grounded counseling quality metrics into a user-facing reflection framework. Using CounselReflect as a study probe, we interviewed 21 users of AI for mental health and well-being support. Although most participants did not routinely reflect on their conversations, they articulated concrete questions they would want reflection to address. Tool-assisted reflection also revealed challenges: participants selectively sought evidence confirming existing perceptions of AI and prioritized dimensions they already valued. We argue that reflection tools should surface blind spots and scaffold more holistic exami
    
[^150]: 当困惑度说谎时：面向生成的混合序列模型蒸馏

    When Perplexity Lies: Generation-Focused Distillation of Hybrid Sequence Models

    [https://arxiv.org/abs/2603.26556](https://arxiv.org/abs/2603.26556)

    该论文揭示了对数似然评估方式会掩盖蒸馏模型在真实自回归生成上的严重质量退化（7B蒸馏模型在对数似然评分下仅落后教师0.2个百分点，但自回归生成时落后20.8个百分点），并提出了面向生成的多阶段蒸馏流水线GenDistill来蒸馏混合序列模型。

    

    通过蒸馏将预训练的Transformer转换为更高效的混合模型，是降低推理成本的一种有前景的方法。然而，要在蒸馏模型中实现高质量生成，需要对学生架构和蒸馏过程进行精心的联合设计。许多先前的蒸馏工作在评估下游多项选择基准时，使用对数似然对候选答案进行排序，而不是要求自回归生成，这可能掩盖模型质量上的重要差异。例如，在重叠的基准测试上，我们展示了一个7B蒸馏模型在对数似然评分下与教师模型相差不到0.2个百分点，但当它必须自回归地生成答案时，却落后了20.8个百分点。我们通过GenDistill研究了这一现象——这是我们设计的一个多阶段流水线，用于将预训练的Transformer蒸馏为高效的混合Kimi Delta注意力（Hybrid-

    arXiv:2603.26556v3 Announce Type: replace-cross  Abstract: Converting a pretrained Transformer into a more efficient hybrid model through distillation offers a promising approach to reducing inference costs. However, achieving high-quality generation in distilled models requires careful joint design of both the student architecture and the distillation process. Many prior distillation works evaluate downstream multiple-choice benchmarks by ranking candidate answers with log-likelihood rather than requiring autoregressive generation, which can obscure important differences in model quality. For example, on overlapping benchmarks, we show that a 7B distilled model that nearly matches its teacher to within 0.2 pp under log-likelihood scoring falls behind by 20.8 pp when it must generate answers autoregressively.   We investigate this phenomenon with GenDistill, a multi-stage pipeline we designed for distilling a pretrained Transformer into an efficient Hybrid Kimi Delta Attention (Hybrid-
    
[^151]: 面向低资源多语言语音到文本翻译的自动化梯度驱动参数共享

    Automated Gradient-Driven Parameter Sharing for Low-Resource Multilingual Speech-to-Text Translation

    [https://arxiv.org/abs/2603.25836](https://arxiv.org/abs/2603.25836)

    该论文提出一种利用训练梯度信息自动确定层级别参数共享模式的方法，通过语言聚类、任务差异度量和子空间对齐三种分析策略，在低资源多语言语音翻译任务中持续提升翻译质量。

    

    在低资源多语言语音到文本翻译中，跨语言统一的结构共享经常会引入表示冲突，从而阻碍模型收敛。本工作提出了一种有原则的方法论，通过挖掘训练梯度信息来自动确定层级别的参数共享模式。我们的方法采用三种不同的分析策略：基于距离的语言聚类、用于容量分配的自/跨任务差异度量，以及结合典型相关分析进行子空间对齐的联合分解。在四个语言对上（使用SeamlessM4T-Medium架构）的广泛评估表明，该方法在翻译质量指标上取得了持续的提升。

    arXiv:2603.25836v2 Announce Type: replace  Abstract: In low-resource multilingual speech-to-text translation, uniform architectural sharing across languages frequently introduces representation conflicts that impede convergence. This work proposes a principled methodology to automatically determine layer-specific sharing patterns by mining training gradient information. Our approach employs three distinct analysis strategies: distance-based language clustering, self/cross-task divergence metrics for capacity allocation, and joint factorization coupled with canonical correlation analysis for subspace alignment. Extensive evaluation across four language pairs (using the SeamlessM4T-Medium architecture) demonstrates persistent improvements in translation quality metrics.
    
[^152]: 当一致性成为偏见：半结构化临床访谈中的访谈者效应

    When Consistency Becomes Bias: Interviewer Effects in Semi-Structured Clinical Interviews

    [https://arxiv.org/abs/2603.24651](https://arxiv.org/abs/2603.24651)

    该研究发现在半结构化临床访谈的抑郁检测任务中，模型会利用访谈者固定的提示词这一脚本痕迹来获得虚高的分类性能，而将模型限制于仅使用参与者的真实话语才能反映真正的语言线索。

    

    arXiv:2603.24651v2 公告类型：replace-cross 摘要：得益于公开语料库的可用性和语言建模技术的进步，从医患对话中自动检测抑郁症的方法获得了快速发展。然而，其可解释性仍然有限：往往只报告了优异的性能，却没有揭示预测背后的驱动因素。我们分析了三个数据集：ANDROIDS、DAIC-WOZ和E-DAIC，并识别出半结构化访谈中访谈者提示所导致的系统性偏见。在访谈者话语上训练的模型会利用固定的提示词及其位置来区分抑郁受试者与对照组，经常在不使用参与者语言的情况下就获得很高的分类分数。若将模型限制为仅使用参与者的话语，决策证据的分布会更加广泛，并能够反映真实的语言线索。尽管半结构化协议确保了一致性，但将访谈者提示纳入模型会通过利用脚本痕迹来夸大性能。我们的研究结果揭示了跨数据集、跨架构的……

    arXiv:2603.24651v2 Announce Type: replace-cross  Abstract: Automatic depression detection from doctor-patient conversations has gained momentum thanks to the availability of public corpora and advances in language modeling. However, interpretability remains limited: strong performance is often reported without revealing what drives predictions. We analyze three datasets: ANDROIDS, DAIC-WOZ, E-DAIC and identify a systematic bias from interviewer prompts in semi-structured interviews. Models trained on interviewer turns exploit fixed prompts and positions to distinguish depressed from control subjects, often achieving high classification scores without using participant language. Restricting models to participant utterances distributes decision evidence more broadly and reflects genuine linguistic cues. While semi-structured protocols ensure consistency, including interviewer prompts inflates performance by leveraging script artifacts. Our results highlight a cross-dataset, architecture-
    
[^153]: MAPLE：元数据增强的私有语言演化

    MAPLE: Metadata Augmented Private Language Evolution

    [https://arxiv.org/abs/2603.19258](https://arxiv.org/abs/2603.19258)

    MAPLE通过引入元数据增强，解决了私有演化（PE）方法在私有数据分布偏离基础模型预训练先验时的初始化瓶颈问题，实现了更高效的基于API的差分隐私合成数据生成。

    

    对大语言模型（LLM）进行差分隐私（DP）微调需要巨大的计算资源和完整的模型访问权限，这使得普通用户无法使用最先进的专有API。生成差分隐私合成数据提供了一种实用的替代方案。这种方法还允许进行透明的探索性数据分析以及在下游任务中的任意重用，从而避开了模型参数空间的刚性约束。私有演化（PE）为生成此类数据提供了一个有前景的基于API的框架，但其成功在很大程度上依赖于初始化。如果私有数据分布与基础模型的预训练先验偏差过大——这在高度专业化的领域中很常见——PE就难以与目标数据对齐。这种不对齐会导致收敛性差、效用下降以及API调用的浪费。为解决这一初始化瓶颈，我们提出了元数据增强的私有语言演化（MAPLE）

    arXiv:2603.19258v3 Announce Type: replace-cross  Abstract: Differentially private (DP) fine-tuning of large language models (LLMs) requires massive compute and full model access, which rules out state-of-the-art proprietary APIs for general users. Generating DP synthetic data offers a practical workaround. This approach also allows for transparent exploratory data analysis and arbitrary reuse across downstream tasks, sidestepping the rigid constraints of a model's parameter space. Private Evolution (PE) provides a promising API-based framework for generating this data, but its success relies heavily on initialization. If the private data distribution falls too far outside the foundation model's pre-training priors -- a common issue in highly specialized domain -- PE struggles to align with the target data. This misalignment causes poor convergence, degraded utility, and wasted API calls. To solve this initialization bottleneck, we introduce Metadata Augmented Private Language Evolution
    
[^154]: 野外社会拟像：Moltbook上的AI智能体社区

    Social Simulacra in the Wild: AI Agent Communities on Moltbook

    [https://arxiv.org/abs/2603.16128](https://arxiv.org/abs/2603.16128)

    首次对AI智能体社区与人类在线社区进行大规模实证比较，发现Moltbook上的AI社区存在极端参与不平等、作者高度重叠，且AI生成内容情感平淡、倾向断言而非探索，其社区同质化现象主要由共享作者身份的结构性因素导致。

    

    随着基于大语言模型（LLM）的自主智能体日益活跃于社交平台，理解AI智能体社区的动态对于传播研究和平台治理都变得至关重要。我们首次对AI智能体社区和人类在线社区进行了大规模实证比较，分析了Moltbook上的73,899条帖子和Reddit上的189,838条帖子，覆盖五个相互匹配的社区。在结构上，我们发现Moltbook呈现出极端的参与不平等（基尼系数为0.84，而Reddit为0.47）以及较高的跨社区作者重叠率（33.8%，而Reddit为0.5%）。在语言属性方面，AI智能体生成的内容在情感上趋于平淡，认知上从探索转向断言，且在社交上较为疏离。这些差异导致了明显的社区层面同质化，但我们证明这主要是由共享作者身份所造成的结构性假象。在作者层面，个体智能体比人类用户更容易被识别，这主要由异常值效应所驱动。

    arXiv:2603.16128v3 Announce Type: replace  Abstract: As autonomous LLM-based agents increasingly populate social platforms, understanding the dynamics of AI-agent communities becomes essential for both communication research and platform governance. We present the first large-scale empirical comparison of AI-agent and human online communities, analyzing 73,899 Moltbook and 189,838 Reddit posts across five matched communities. Structurally, we find that Moltbook exhibits extreme participation inequality (Gini = 0.84 vs. 0.47) and high cross-community author overlap (33.8% vs. 0.5%). In terms of linguistic attributes, content generated by AI-agents is emotionally flattened, cognitively shifted toward assertion over exploration, and socially detached. These differences give rise to apparent community-level homogenization, but we show this is primarily a structural artifact of shared authorship. At the author level, individual agents are more identifiable than human users, driven by outlie
    
[^155]: TTSR：通过反思进行测试时自我进化

    TTSR: Test-Time Self-Evolving via Reflection

    [https://arxiv.org/abs/2603.03297](https://arxiv.org/abs/2603.03297)

    TTSR通过让单个模型交替扮演学生和教师角色，基于反思后合成的范式，在测试时针对失败轨迹生成变体问题，从而克服了缺乏可学习样本和探索效率低下的瓶颈。

    

    测试时训练（TTT）在推理过程中仅使用未标记的测试输入来适应大型语言模型（LLMs）。然而，现有方法在困难推理任务上面临两个主要瓶颈：（1）\emph{缺乏可学习样本}，因为困难问题上自生成的伪标签往往带有噪声，导致不稳定的奖励；（2）\emph{探索效率低下}，因为性能提升依赖于反复采样大量生成结果，而没有对先前尝试失败原因进行明确诊断。我们提出\textbf{TTSR}（\textbf{T}est-\textbf{T}ime \textbf{S}elf-\textbf{R}eflection），一个基于\emph{先反思后合成}范式的自我进化框架。一个预训练模型在\textit{学生}和\textit{教师}两种角色间交替：学生解决测试问题并进行更新，而教师分析失败轨迹并合成更接近学生能力边界的有针对性的变体问题。TTSR进一步...

    arXiv:2603.03297v2 Announce Type: replace  Abstract: Test-time training (TTT) adapts large language models (LLMs) during inference using only unlabeled test inputs. Existing methods, however, face two major bottlenecks on hard reasoning tasks: (1) \emph{lack of learnable samples}, as self-generated pseudo-labels on difficult questions are often noisy and yield unstable rewards; and (2) \emph{inefficient exploration}, as performance gains depend on repeatedly sampling many rollouts without explicit diagnosis of why previous attempts fail. We propose \textbf{TTSR} (\textbf{T}est-\textbf{T}ime \textbf{S}elf-\textbf{R}eflection), a self-evolving framework based on a \emph{reflect-then-synthesize} paradigm. A single pretrained model alternates between a \textit{Student} role and a \textit{Teacher} role: the Student solves test questions and updates, while the Teacher analyzes failed trajectories and synthesizes targeted variant questions closer to the Student's capability frontier. TTSR fur
    
[^156]: 面向多跳推理的亲属关系数据基准

    Kinship Data Benchmark for Multi-hop Reasoning

    [https://arxiv.org/abs/2601.07794](https://arxiv.org/abs/2601.07794)

    该论文提出了KinshipQA基准，其核心创新是一个可按需生成大规模、真实且具有文化特异性的家谱数据的生成式流水线，从而系统评估大型语言模型在亲属关系多跳推理上的能力。

    

    大型语言模型（LLMs）越来越多地在多跳推理能力上接受评估，即把多条信息组合成连贯推理的能力。我们提出了KinshipQA，这是一个旨在通过亲属关系推理来探究这一能力的基准。我们工作的核心贡献是一个生成式流水线，能够按需生成大规模、真实且具有文化特异性的家谱数据：即满足与不同亲属制度相关的明确婚姻约束的相互关联的家族树集合。这使得任务难度、文化假设和关系深度能够被系统地控制和调节。基于这些家谱数据，我们构建了需要对隐式关系链进行推理的文本推理任务。我们使用六个最先进的大型语言模型（涵盖开源和闭源模型）在统一的评测设置下对该基准进行了评估。

    arXiv:2601.07794v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly evaluated on their ability to perform multi-hop reasoning, i.e., to combine multiple pieces of information into a coherent inference. We introduce KinshipQA, a benchmark designed to probe this capability through reasoning over kinship relations. The central contribution of our work is a generative pipeline that produces, on demand, large-scale, realistic, and culture-specific genealogical data: collections of interconnected family trees that satisfy explicit marriage constraints associated with different kinship systems. This allows task difficulty, cultural assumptions, and relational depth to be systematically controlled and varied. From these genealogies, we derive textual inference tasks that require reasoning over implicit relational chains. We evaluate the resulting benchmark using six state-of-the-art LLMs, spanning both open-source and closed-source models, under a uniform z
    
[^157]: 信息不对称下LLM智能体协作中的沟通与验证

    Communication and Verification in LLM Agents towards Collaboration under Information Asymmetry

    [https://arxiv.org/abs/2510.25595](https://arxiv.org/abs/2510.25595)

    本文将经典的爱因斯坦谜题扩展为桌面游戏，研究信息不对称条件下两个LLM智能体通过推理、沟通与行动实现协作，并提出“微调加验证器”框架，利用沟通策略和环境验证信号显著提升协作完成任务的能力。

    

    虽然大型语言模型（LLM）智能体通常从行动规划/生成的角度出发来完成目标（例如由语言描述给出的目标），但它们彼此协作以实现共同目标的能力尚未得到充分探索。为了解决这一局限，本文研究了任务协作场景中的LLM智能体，特别是在信息不对称的条件下，即智能体在知识和技能上存在差异，需要共同合作才能完成共享任务。我们将经典符号谜题“爱因斯坦谜题”扩展为一种桌面游戏。在该游戏中，两个LLM智能体必须进行推理、沟通和行动，以满足解决谜题所需的空间和关系约束。我们应用了一种“微调加验证器”框架，使LLM智能体配备多种沟通策略以及来自环境的验证信号。实证结果凸显了关键重要性……（原文摘要在此处被截断）

    arXiv:2510.25595v2 Announce Type: replace-cross  Abstract: While Large Language Model (LLM) agents are often approached from the angle of action planning/generation to accomplish a goal (e.g., given by language descriptions), their abilities to collaborate with each other to achieve a joint goal are not well explored. To address this limitation, this paper studies LLM agents in task collaboration, particularly under the condition of information asymmetry, where agents have disparities in their knowledge and skills and need to work together to complete a shared task. We extend Einstein Puzzles, a classical symbolic puzzle, to a table-top game. In this game, two LLM agents must reason, communicate, and act to satisfy spatial and relational constraints required to solve the puzzle. We apply a fine-tuning-plus-verifier framework in which LLM agents are equipped with various communication strategies and verification signals from the environment. Empirical results highlight the critical impo
    
[^158]: TripScore：通过专家校准的奖励使大语言模型对齐于现实世界的旅行规划

    TripScore: Aligning LLMs for Real-World Travel Planning via Expert-Calibrated Reward

    [https://arxiv.org/abs/2510.09011](https://arxiv.org/abs/2510.09011)

    TripScore 是基于真实用户日志与 203 位旅行专家校准构建的旅行规划评估基准，研究发现强化学习微调（如 GRPO）在现实旅行规划任务中比其他方法带来更稳定一致的提升。

    

    在我们已部署的旅行规划服务中，大多数用户给出的是极少量的输入或自由形式的请求，而非现有基准所假设的结构化约束清单。因此，我们提出了 TripScore，这是一个基于真实用户日志构建、并通过 203 位旅行专家的 1,468 个成对判断进行校准的行为基准和评估框架。TripScore 将分层可行性门控（格式与常识）与统一的逐点奖励相结合，该奖励聚合了软性质量与偏好满足程度。我们使用 TripScore 同时作为评估器和奖励信号，对直接提示、测试时计算、神经符号求解器、代码智能体和微调等方法进行了基准测试。我们发现，在相同的基础模型和实际延迟条件下，强化学习微调（如 GRPO）相比其他方法带来了一致的性能提升。

    arXiv:2510.09011v4 Announce Type: replace  Abstract: In our deployed travel-planning service, most users give minimal inputs or free-form requests rather than the structured constraint checklists assumed by existing benchmarks. We therefore present TripScore, a behavior-grounded benchmark and evaluation framework built from real user logs and calibrated against 1,468 pairwise judgments by 203 travel experts. TripScore couples a hierarchical feasibility gate (format and commonsense) with a unified, point-wise reward that aggregates soft quality and preference fulfillment. Using TripScore as both evaluator and reward signal, we benchmark direct prompting, test-time compute, neuro-symbolic solvers, code agents, and fine-tuning. We find that reinforcement learning fine-tuning (e.g., GRPO) provides consistent gains over other approaches under the same base model and practical latency.
    
[^159]: oMeBench：迈向有机机理解析与推理中大语言模型的稳健基准测试

    oMeBench: Towards Robust Benchmarking of LLMs in Organic Mechanism Elucidation and Reasoning

    [https://arxiv.org/abs/2510.07731](https://arxiv.org/abs/2510.07731)

    该论文提出了首个大规模专家标注的有机机理推理基准oMeBench（含超过10,000个注释机理步骤）以及oMeS动态评分框架，用以严格评估大语言模型真正的化学推理能力。

    

    有机反应机理描述了反应物通过分步基本过程转化为中间体和产物的途径，是理解化学反应活性以及指导分子和反应设计的基础。虽然大语言模型（LLMs）在合成设计等化学任务上展现出前景，但这种表现究竟在多大程度上反映了真正的化学推理能力仍不清楚：即生成化学上有效的中间体、在反应步骤之间保持一致性、以及遵循逻辑连贯的多步路径的能力。为了探究这一问题，我们提出了oMeBench，这是首个大规模、由专家精心标注的有机机理推理基准，包含超过10,000个带有注释的机理步骤，并附有反应类型标签、中间体结构和难度评级。为了实现细粒度评估，我们进一步提出了oMeS，一个联合评估步骤级（原文在此处截断）

    arXiv:2510.07731v4 Announce Type: replace  Abstract: Organic reaction mechanisms describe the step-wise elementary processes by which reactants transform into intermediates and products, and are fundamental to understanding chemical reactivity and guiding molecular and reaction de-sign. While large language models (LLMs) have shown promise on chemical tasks such as synthesis design, it remains unclear to what extent this reflects genuine chemical reasoning capabilities: the ability to generate chemically valid intermediates, maintain consistency across reaction steps, and follow logically coherent multi-step pathways. To investigate this, we introduce oMeBench, the first large-scale, expert-curated benchmark for organic mechanism reasoning, comprising over 10,000 annotated mechanistic steps with reaction type labels, intermediate structures, and difficulty ratings. To enable fine-grained evaluation, we further propose oMeS, a dynamic scoring framework that jointly assesses step-level l
    
[^160]: Compass-v3：面向东南亚多语言电商的领域专用大语言模型扩展

    Compass-v3: Scaling Domain-Specific LLMs for Multilingual E-Commerce in Southeast Asia

    [https://arxiv.org/abs/2509.09121](https://arxiv.org/abs/2509.09121)

    该论文提出了面向东南亚电商的245B参数垂直领域混合专家大模型Compass-v3，通过更大专家设计、硬件级效率优化和最优传输直接偏好优化（OTWO）方法，显著提升了多语言电商场景下的领域性能与指令遵循能力。

    

    大语言模型（LLMs）在通用领域应用中表现出色，但在需要领域特定知识的专业任务中，其性能往往会下降。电商领域尤其具有挑战性，因为其数据嘈杂、异构、多语言且高度动态。我们提出了Compass-v3，这是一个垂直领域的混合专家模型，总参数量为245B，每个token激活71B参数，专为东南亚电商设计。Compass-v3采用数量更少但规模更大的专家结构，并结合硬件高效优化手段——如节点内专家并行和定制的memcpy算子——以最大化GPU利用率。该模型基于12T token的精选多语言语料库和大规模合成的电商指令数据，采用混合训练策略进行训练。为增强模型对齐能力，我们提出了最优传输直接偏好优化（OTWO），它能够捕捉token级别的差异并提升指令遵循能力

    arXiv:2509.09121v2 Announce Type: replace  Abstract: Large language models (LLMs) excel in general-domain applications, yet their performance often degrades in specialized tasks requiring domain-specific knowledge. E-commerce is particularly challenging, as its data are noisy, heterogeneous, multilingual, and highly dynamic. We present Compass-v3, a vertical-domain Mixture-of-Experts (MoE) model with 245B total parameters and 71B active per token, designed for Southeast Asian e-commerce. Compass-v3 adopts fewer but larger experts, combined with hardware-efficient optimizations-such as intra-node expert parallelism and a customized memcpy operator-to maximize GPU utilization. The model is trained on 12T tokens of curated multilingual corpora and large-scale synthetic e-commerce instructions using a mixed-training strategy. To enhance alignment, we propose Optimal-Transport Direct Preference Optimization (OTPO), which captures token-level distinctions and improves instruction adherence i
    
[^161]: LMEnt：一套用于分析语言模型知识的工具套件——从预训练数据到表示

    LMEnt: A Suite for Analyzing Knowledge in Language Models from Pretraining Data to Representations

    [https://arxiv.org/abs/2509.03405](https://arxiv.org/abs/2509.03405)

    LMEnt是一个用于分析语言模型知识获取的工具套件，包含实体标注的预训练语料库、性能提升高达80.4%的实体检索方法，以及12个带4K检查点的预训练模型，为研究预训练数据与知识表示之间的联系提供了受控环境。

    

    语言模型（LM）日益驱动着需要世界知识的现实应用。然而，模型将数据转化为知识表示和对世界的信念的内部过程仍然鲜为人知。为促进此类研究，我们提出了LMEnt，这是一个包含以下内容的工具套件：（1）一个知识丰富的预训练语料库，基于维基百科对实体提及进行了完整标注；（2）一种基于实体的预训练数据检索方法，其性能比现有工具高出多达80.4%；（3）12个预训练语言模型，参数量高达10亿，包含4K个中间检查点，在知识任务上的表现与流行的开源模型相当。这些资源共同提供了一个受控环境，用于分析预训练数据中实体提及与下游性能之间的联系。我们通过研究训练过程中的知识获取来展示LMEnt的实用性，发现实体共现……

    arXiv:2509.03405v2 Announce Type: replace  Abstract: Language models (LMs) increasingly drive real-world applications that require world knowledge. However, the internal processes through which models turn data into representations of knowledge and beliefs about the world are poorly understood. To facilitate such studies, we present LMEnt, a suite including (1) a knowledge-rich pretraining corpus, fully annotated with entity mentions based on Wikipedia, (2) an entity-based retrieval method over pretraining data that outperforms existing tools by as much as 80.4%, and (3) 12 pretrained LMs with up to 1B parameters and 4K intermediate checkpoints, with comparable performance to popular open-source models on knowledge tasks. Together, these resources provide a controlled environment for analyzing connections between entity mentions in pretraining data and downstream performance. We show the utility of LMEnt by studying knowledge acquisition over training, finding that entity co-occurrence
    
[^162]: GeLaCo：一种层压缩的进化方法

    GeLaCo: An Evolutionary Approach to Layer Compression

    [https://arxiv.org/abs/2507.10059](https://arxiv.org/abs/2507.10059)

    GeLaCo提出了一种基于进化搜索和参数化权重合并的新型层折叠方法，通过基于残差更新相似性和语言建模KL散度的适应度函数，高效地探索大语言模型的压缩解空间。

    

    大语言模型在大量任务中取得了卓越的性能，但由于巨大的计算需求，在部署和使用方面面临关键障碍。模型压缩方法旨在在保持模型能力的同时减小模型规模，是缓解这些问题的重要手段。沿着这些方向的有前景的方法，如结构化剪枝，通常需要昂贵的手动超参数探索，或依赖于可能忽略更优解的局部启发式方法。在这项工作中，我们提出了GeLaCo，一种通过层折叠进行大语言模型压缩的进化方法。我们的方法通过基于种群的搜索和基于参数化权重合并的新型层折叠公式，支持对压缩解空间的高效探索，其适应度函数基于残差更新的相似性和语言建模KL散度。GeLaCo还支持单（摘要在此处被截断）

    arXiv:2507.10059v2 Announce Type: replace  Abstract: Large Language Models have achieved remarkable performance across a large number of tasks, but face critical deployment and usage barriers due to substantial computational requirements. Model compression methods, which aim to reduce model size while preserving its capacity, are an important means to mitigate these issues. Promising approaches along these lines, such as structured pruning, typically require costly manual hyperparameter exploration or rely on local heuristics that may run the risk of ignoring better solutions. In this work we introduce GeLaCo, an evolutionary approach to LLM compression via layer collapse. Our approach supports an efficient exploration of the compression solution space via population-based search and a novel layer collapse formulation based on parametrized weight merging, with a fitness function based on similarity over residual updates and language modeling KL divergence. GeLaCo also supports both sin
    
[^163]: Redemption Score：一种通过分布、感知与语言信号三角测量实现图像描述多模态评估的框架

    Redemption Score: A Multi-Modal Evaluation Framework for Image Captioning via Distributional, Perceptual, and Linguistic Signal Triangulation

    [https://arxiv.org/abs/2505.16180](https://arxiv.org/abs/2505.16180)

    提出了Redemption Score（RS）评估框架，通过融合互信息散度、DINO感知相似度和LLM文本嵌入三种互补信号对图像描述进行多模态评估，在Flickr8k基准上取得58.42的Kendall-tau，优于大多数先前方法。

    

    arXiv:2505.16180v3 公告类型： replace-cross

    arXiv:2505.16180v3 Announce Type: replace-cross  Abstract: Evaluating image captions requires cohesive assessment of both visual semantics and language pragmatics, which is often not entirely captured by most metrics. As such metrics increasingly guide model development, benchmarking, and system optimization in multimodal AI, inaccuracies in evaluation can misrepresent true progress. We introduce Redemption Score(RS), a novel evaluation framework for multi-modal generation by triangulating three complementary signals: (1) Mutual Information Divergence (MID) for global image-text distributional alignment, (2) DINO-based perceptual similarity of cycle-generated images for visual grounding, and (3) LLM Text Embeddings for contextual text similarity against human references. A calibrated fusion of these signals allows RS to offer a more holistic assessment. On the Flickr8k benchmark, RS achieves a Kendall-$\tau$ of 58.42, outperforming most prior methods and demonstrating superior correlat
    
[^164]: 一个源自开放科学文献的大规模视觉-语言数据集，用于推动生物医学通用人工智能的发展

    A Large-Scale Vision-Language Dataset Derived from Open Scientific Literature to Advance Biomedical Generalist AI

    [https://arxiv.org/abs/2503.22727](https://arxiv.org/abs/2503.22727)

    该论文发布了源自PubMed Central开放获取文献的开源大规模多模态数据集Biomedica（含600万篇文章、2400万图像-文本对及专家标注），基于其训练的AI模型在嵌入、对话和检索等各任务类别中均超越了此前的开放系统。

    

    尽管生物医学人工智能（AI）备受关注，但获取高质量、多样化且大规模的数据——现代AI系统的基础——仍然是释放其全部潜力的瓶颈。为解决这一差距，我们推出了Biomedica，这是一个源自PubMed Central开放获取子集的开源数据集，包含超过600万篇科学文章和2400万对图像-文本，以及27个元数据字段（包括专家人工标注）。为克服访问大规模数据集的挑战，我们通过网络服务器提供了可扩展的流式传输和搜索API，便于与AI系统无缝集成。我们通过构建嵌入模型、聊天式模型和检索增强的聊天代理来展示Biomedica数据集的实用性。值得注意的是，我们所有的AI模型在各自类别中都超越了以往的开放系统，凸显了多样化、高质量数据的关键作用。

    arXiv:2503.22727v3 Announce Type: replace  Abstract: Despite the excitement behind biomedical artificial intelligence (AI), access to high-quality, diverse, and large-scale data - the foundation for modern AI systems - is still a bottleneck to unlocking its full potential. To address this gap, we introduce Biomedica, an open-source dataset derived from the PubMed Central Open Access subset, containing over 6 million scientific articles and 24 million image-text pairs, along with 27 metadata fields (including expert human annotations). To overcome the challenges of accessing our large-scale dataset, we provide scalable streaming and search APIs through a web server, facilitating seamless integration with AI systems. We demonstrate the utility of the Biomedica dataset by building embedding models, chat-style models, and retrieval-augmented chat agents. Notably, all our AI models surpass previous open systems in their respective categories, underscoring the critical role of diverse, high-
    

