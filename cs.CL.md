# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bellman Policy Optimization](https://arxiv.org/abs/2609.15987) | 提出无评论家的贝尔曼策略优化（BPO），利用贝尔曼方程将策略镜像下降重新表述为轨迹级目标，避免了中间状态价值估计且保持相同的最优解，从而有效提升大语言模型的数学推理能力。 |
| [^2] | [Stellar Colosseum: A Many-Agent Harness for Long-Horizon Research in Mathematics and Theoretical Computer Science](https://arxiv.org/abs/2609.15983) | 提出Stellar Colosseum，一个与模型无关的多智能体框架，通过策略探索、就绪门控、子问题分解、定向证伪和树聚合等机制，提升语言模型在数学与理论计算机科学长周期研究问题上的可靠性。 |
| [^3] | [The Router Within: Eliciting Native Skill Routing from a Frozen LLM](https://arxiv.org/abs/2609.15982) | 该论文提出Gavel方法，证明冻结的LLM在其内部前向传播中已天然携带技能路由信号，仅需训练两个线性映射即可在无需将技能文本放入上下文的情况下实现高效且可扩展的技能选择。 |
| [^4] | [Disentangling Representation Evolution in Transformers through Directional Decomposition](https://arxiv.org/abs/2609.15975) | 本文提出一种方向分解方法，将Transformer中学习到的更新拆解为平行与垂直分量，发现排除自身的值空间平行操作在模型编辑中远比其他空间操作更稳健，并能以分量级分辨率刻画压缩带来的更新误差。 |
| [^5] | [Discovery Foundation Models: Toward Open-Ended Discovery Intelligence](https://arxiv.org/abs/2609.15973) | 本文提出“发现智能”这一新前沿概念，并构建发现基础模型（DFM）框架及其实例Zetema，通过支持问题发现、表示构建、假设形成、基于证据的修订等七项耦合能力，使AI从解决人类指定问题转向自主参与新知识与新问题的创造过程。 |
| [^6] | [Mind2Dialogue: Training Human-Aware Language Models by Simulating User Mental States](https://arxiv.org/abs/2609.15972) | 提出Mind2Dialogue框架，通过心理学引导的模拟器模拟用户心理状态并将其转化为特权监督信号，以训练能够理解用户潜在信念和目标的人类感知型语言模型。 |
| [^7] | [Verifiable by Construction: Claim-Level Evaluation of Verbatim Citation in Clinical Question Answering](https://arxiv.org/abs/2609.15964) | 该论文基于四份临床实践指南构建了标准化评估框架，从为每个事实性声明提供引用、生成逐字引用到确保引用完全支撑声明，端到端地评估了十二个大语言模型在临床问答中构建可验证答案的能力。 |
| [^8] | [HypoEvolve: Genetic Algorithms Enable Multi-Agent LLMs to Discover Scientific Hypotheses](https://arxiv.org/abs/2609.15938) | 该论文提出HypoEvolve框架，通过代际遗传算法协调多个专门化大语言模型智能体，使智能体协作过程显式化，从而发现高质量的科学假设。 |
| [^9] | [Inoculation Midtraining with Learned Neologisms](https://arxiv.org/abs/2609.15886) | 提出接种式中训练技术，通过在中训练阶段引入新造词标记不安全行为所属的上下文，可在保留良性特性迁移的同时降低模型对齐失准，但其效果不及标准接种式提示且对训练配置敏感。 |
| [^10] | [K-Bench: a clinically calibrated benchmark for evaluating large language models in high-risk mental health conversations](https://arxiv.org/abs/2609.15855) | 该论文提出了K-Bench，一个经临床医生校准的基准，包含200个涉及自杀、自残、家庭暴力等多轮高风险心理健康情景案例，并采用与临床医生共识达94.2%一致率的GPT-4o自动评判系统，系统评估了33个基础大语言模型在高风险心理健康对话中的安全性。 |
| [^11] | [Learning to Coach for Experiential Learning](https://arxiv.org/abs/2609.15851) | 提出L2C框架，训练专用“LLM教练”从模型过往轨迹中提炼可操作的经验知识，借助同实例与跨实例奖励进行优化，在数学推理和文本游戏中持续优于自我修正方法，并比扩大执行模型规模更有效地利用推理计算资源。 |
| [^12] | [Before You Poll with LLMs: A Deliberative Diagnostic Framework](https://arxiv.org/abs/2609.15849) | 提出协商式民意调查诊断框架，通过相同信息干预对比人类与LLM的信念变化，揭示五个前沿模型虽能产生看似合理的党派观点，却在动态信念更新方面存在静态评估无法发现的失败。 |
| [^13] | [CiteGuard-RAG: A Validation-Centered AI System for Evidence-Grounded Question Answering](https://arxiv.org/abs/2609.15830) | CiteGuard-RAG通过在运行时融合混合语义-词汇检索、引用约束生成、句子级依据验证与单次重新生成机制，实现了高准确率、引用有效且无幻觉的有据问答，在受控评估中达到98.3%的有据答案准确率和引用有效性。 |
| [^14] | [EvoOntology: A Self-Evolving Ontology Layer for Data Agents](https://arxiv.org/abs/2609.15779) | 本文提出EvoOntology，一种面向数据智能体的自进化本体层，通过将本体封装为包含模式层、内容层和工具层的MCP服务器，并借助构建器智能体实现自主构建与自进化循环，有效弥合了智能体与异构数据之间的鸿沟。 |
| [^15] | [Enabling Streaming User Transcription in Full-Duplex Speech-to-Speech Models](https://arxiv.org/abs/2609.15759) | 通过在双工S2S模型中并行添加轻量级ASR头，以极少的额外参数和架构改动实现了实时流式用户语音转写，同时保留全双工对话能力，在HuggingFace开放ASR排行榜上达到10.21%的流式平均词错误率。 |
| [^16] | [Sequential Adapter Stacking for Cross-Lingual Low-Resource ASR](https://arxiv.org/abs/2609.15758) | 提出顺序适配器堆叠方法，在冻结的源语言适配器之上堆叠可训练的目标语言适配器，实现从资源丰富语言到低资源语言的高效知识迁移，在Whisper不支持的三种语言上持续显著优于全量微调。 |
| [^17] | [Look Before You Leap: Factual Decoding with Internal Attribution Signals](https://arxiv.org/abs/2609.15745) | 提出DescaPE解码框架，利用LLM中事实性显著层区间的内部归因信号，通过轻量级探针在推理时惩罚易产生幻觉的候选续写，从而抑制事实性错误的滚雪球效应。 |
| [^18] | [Merging the Knowledge of LLMs for Automatic Speech Recognition](https://arxiv.org/abs/2609.15743) | 本文提出通过模型合并（对LoRA参数进行算术运算）将外部语言模型知识直接融入基于大语言模型的ASR模型参数中，无需额外推理计算成本即可持续提升目标领域的语音识别性能。 |
| [^19] | [Data storytelling meets interpretable machine learning: Decoding AI decisions for non-experts without revealing sensitive data and model details](https://arxiv.org/abs/2609.15722) | 本研究将数据叙事与可解释机器学习相结合，通过提出DIST金字塔、I-P-O模型以及"What-if"和"Why-not"事件生成架构，并采用数据脱敏技术，使非专家用户能够在不泄露敏感数据和模型细节的情况下理解AI决策。 |
| [^20] | [RESKILL: Explicit Failure Attribution and Structured Repair for Interactive Language Agents](https://arxiv.org/abs/2609.15684) | RESKILL提出了一种结构化技能修复框架，通过显式维护失败归因与候选补丁的关联、基于覆盖率的局部修复选择，以及跨修复轮次传递重测结果，显著改进了交互式语言代理失败后的技能修复过程。 |
| [^21] | [CiteShade: Citation Laundering in Multi-Source Retrieval-Augmented Generation and Its Counterfactual Defense](https://arxiv.org/abs/2609.15660) | 该论文提出首个针对检索增强生成系统的“引用洗白”攻击CiteShade，攻击者仅通过控制单一来源即可诱导模型输出错误答案并将其错误归因于可信来源，同时给出了三个攻击必要条件及相应的反事实防御方法。 |
| [^22] | [Empathy Is Steerable but Multi-Axial: Mechanism Geometry and Persona Effects in LLMs](https://arxiv.org/abs/2609.15654) | 研究发现大语言模型的共情可以通过激活引导有效控制，但其机制是多轴的——不同共情维度的激活方向仅部分可分离且相互影响，角色提示也会与这些方向产生交互作用。 |
| [^23] | [Human-Grounded Calibration for Long-Text Image-Text Congruence in Vision-Language Models](https://arxiv.org/abs/2609.15640) | 本文提出一致性分数——一种轻量级校准层，将双编码器模型的图文相似度证据映射为有界且可解释的一致性分数，并通过基于人类评估的实验揭示了直接事后校准与基于投影方法之间的权衡。 |
| [^24] | [IROH: Insightful Ranking Of Humor using Multi-Stage Hybrid Retrieval with Rationale-Distilled LLM Judges for JOKER 2026 Track Task 1 English](https://arxiv.org/abs/2609.15618) | 该论文提出IROH三阶段检索系统，通过稀疏-稠密混合检索、交叉编码器重排序与原理蒸馏的LoRA大语言模型裁判集成，在JOKER 2026任务1幽默排序中夺得第一名（MAP 0.6347），并发现原理蒸馏裁判是排序质量的关键驱动因素。 |
| [^25] | [Through the Eyes of the Beholder: Biometric and Demographic Conditioning for Multimodal Sexism Detection](https://arxiv.org/abs/2609.15608) | 该论文提出一个以人为本的多模态框架，通过FiLM条件化融合标注者的心理与人口统计学特征、五种输入模态，并将性别歧视检测建模为标签分布学习问题，从而有效捕捉互联网性别歧视检测中的主观性。 |
| [^26] | [Can We Trust the Judges? Validation of Factuality Evaluation Methods via Answer Perturbation](https://arxiv.org/abs/2609.15561) | 本文提出一个元评估框架，通过对标准答案进行受控扰动来验证事实性评估指标的可靠性，发现基于流水线的方法（如RAGAS）比LLM-as-judge方法更能有效追踪事实性退化，并提出了一种更具成本效益的事实正确性指标新变体。 |
| [^27] | [Don't Count the Edits, Judge by the Outcome Alone: Reward-Based Evaluation for Grammatical Error Correction](https://arxiv.org/abs/2609.15559) | 提出了SURE——一个以源文本为条件的基于奖励的语法纠错评估器，通过联合学习整体奖励、语法性/忠实性/流畅性的标准级监督以及片段级定位，在改写风格修正的评估上表现出色。 |
| [^28] | [Option-Aware Retrieval and Task-Specific VLM Adaptation for Medical VQA](https://arxiv.org/abs/2609.15530) | 本文提出MedReason 2026医学视觉问答挑战赛方案，发现基于答案语义而非标签的选项感知检索可将纯检索准确率从20.0%大幅提升至57.5%，且归因分析表明任务特定的LoRA微调是系统达到93.5%以上准确率的主要来源。 |
| [^29] | [To Each Language Its Tokenizer: Modular Tokenizers for Efficient Multilingual LLMs](https://arxiv.org/abs/2609.15528) | 提出模块化BPE和Unigram分词器框架，可为任意语言子集提取压缩率媲美单语分词器的子分词器，并通过将预测限制在相关词表子集的预训练策略，实现多语言大语言模型的高效训练。 |
| [^30] | [Psychosis involves a deficit of information compression in connected speech](https://arxiv.org/abs/2609.15522) | 该研究利用大型语言模型的表征指标和预测误差，发现精神病患者的连贯言语存在一般性的信息压缩缺陷，表现为表征内在维度降低与惊异度升高，且与语法组织能力受损相关。 |
| [^31] | [Beyond Safe Answers: Segment-Aware Listwise Alignment for Reasoning Safety in Large Reasoning Models](https://arxiv.org/abs/2609.15517) | 提出SaLT-DPO方法，通过分段感知的列表式对齐、基于“最弱环节”原则的安全一致性正则化以及良性提示效用锚定三种机制，同时保障大型推理模型的推理过程与最终答案两个层面的安全性。 |
| [^32] | [Authorship attribution and aesthetic evaluation of AI poetry: a case study with Haiku](https://arxiv.org/abs/2609.15511) | 本研究让日本大学生区分AI与人类创作的俳句，发现不同大语言模型的“AI痕迹”可辨识度差异显著，其中GPT-5、Gemini 2.5和StableLM-7B生成的俳句已难以与人类作品区分。 |
| [^33] | [How Lossless Is Lossless Speculative Decoding? The Role of Numerical Precision in Orthrus](https://arxiv.org/abs/2609.15504) | 该论文独立复现了 Orthrus 并发现其在 BF16 精度下仅有约 43-45% 的输出轨迹与自回归模型完全一致，表明其“无损推测解码”声明强烈依赖于数值精度，尽管下游基准测试未显示系统性退化。 |
| [^34] | [Temperature Fragility and the Conditional Benefits of Truncation Sampling](https://arxiv.org/abs/2609.15476) | 该研究首次在部署系统常用的默认温度（0.6–1.0）下系统评估截断采样，发现大模型准确率对温度高度脆弱——部分模型温度从0.7升至1.3会在MMLU-Pro上损失17–38个点，而top-p/min-p等截断采样方法仅在高温下才有明显收益。 |
| [^35] | [Turkish MMLU Pro: Traceable Option Augmentation and Its Validity Limits in Turkish Multiple-Choice Evaluation](https://arxiv.org/abs/2609.15467) | 该研究构建了土耳其语MMLU Pro基准，通过从同章节其他题目复制五个可追溯的干扰选项将选项从五个扩展到十个，发现增加选项会显著降低大语言模型得分但不提升评估有效性，其中准确率下降主要源于模型误选借来选项，且在否定式题干上下降尤为严重。 |
| [^36] | [MarKey: Marginal Utility Guided Greedy Keyframe Selection for Long Video Understanding](https://arxiv.org/abs/2609.15408) | MarKey提出了一种免训练的子集感知贪心优化框架，通过联合评估查询相关性、边际覆盖增益和上下文相关冗余性来为长视频理解选择关键帧，从而避免冗余选择并实现更完整的证据覆盖。 |
| [^37] | [SlopShape: Identifying AI-Generated Commercial Web Content](https://arxiv.org/abs/2609.15369) | 该研究提出通过结构特征（信息呈现方式、顺序、证据与语气）而非词级特征来识别商业网页中的AI生成内容，仅用187个结构特征就在模型自我改写的对抗条件下仍保持约98%的检测性能。 |
| [^38] | [RSIAgent: Autonomous Exploration for Recursive Self-improvement in New Environments](https://arxiv.org/abs/2609.15364) | RSIAgent是一个无需训练的多智能体框架，通过“先广后深”的自主探索策略构建可复用的冻结记忆，使数字智能体在新环境中实现递归自我改进，且无需更新模型参数。 |
| [^39] | [Parameter-Efficient Adaptation of Pretrained Language Models for Time-Series Forecasting](https://arxiv.org/abs/2609.15344) | 该论文提出一种参数高效的迁移学习框架，将固定长度时间序列补丁直接投影到冻结的GPT-2嵌入空间中，以绕过文本标记化的方式将语言模型作为通用序列编码器，并通过七个基准数据集上的系统性消融实验，揭示了表示策略、适配方式和架构组件等设计选择如何影响语言模型向时间序列预测的有效跨模态迁移。 |
| [^40] | [Dynamic Semantic Compression for Efficient Latent-Space Inference in Large Language Models](https://arxiv.org/abs/2609.15338) | 提出DSEI框架，通过动态语义自编码器将段级语义自适应压缩为紧凑潜在表示，使大语言模型能够在潜在空间中进行段级推理，在降低48%困惑度的同时显著提升推理效率。 |
| [^41] | [Clean Scores, Buried Evidence, and Confident Wrong: A Receipt-Based Audit of Frontier Agentic QA](https://arxiv.org/abs/2609.15319) | 前沿智能体在证据被掩埋时准确率显著下降、成本上升，且高置信度无法暴露错误，作者提出以“声明级凭证”溯源、条件感知评分和人类对抗性验证为核心的审计式评估框架，以取代排行榜式评测。 |
| [^42] | [Reducing the Output-Mode Gap in Speech Language Models via Joint-Output On-Policy Distillation](https://arxiv.org/abs/2609.15313) | 该论文发现了语音语言模型中语音转文本语音（S2TS）模式的内部文本生成准确率显著低于语音转文本（S2T）模式的“输出模式差距”问题，并提出联合输出在策略蒸馏（JO-OPD）方法，利用学生生成的S2TS轨迹将更强的S2T策略蒸馏到联合生成中，有效缩小了这一差距。 |
| [^43] | [When Agents Slow Down: Understanding LLM Agents' Test-Time Strategies via Elo-per-token Analysis](https://arxiv.org/abs/2609.15309) | 该论文提出Elo-per-token分析方法，通过追踪不同token预算下的最佳解决方案并利用Bradley-Terry模型将任务内排序聚合为跨任务Elo评分，从而系统性地衡量LLM智能体的测试时计算扩展性能。 |
| [^44] | [Reason What Matters: Retrieval-Grounded Reasoning for Universal Multimodal Embeddings](https://arxiv.org/abs/2609.15296) | 该论文提出ReWAM框架，利用检索反馈来指导CoT推理中的信用分配与推理计算量，解决了GRPO对所有token赋予相同优势以及完整CoT生成引入高延迟的问题，从而实现更高效的通用多模态嵌入。 |
| [^45] | [Artificial entrepreneurial cognition: Locating and causally steering an opportunity recognition dial inside large language models (LLMs)](https://arxiv.org/abs/2609.15277) | 本文提出“人工创业认知”概念，首次将机制可解释性引入创业研究，在大语言模型内部定位出机会识别的表征方向，并通过直接干预该“机会识别刻度盘”实现对模型机会判断的因果调控。 |
| [^46] | [Semiotic Relations and Proof Methods: A Cross-Genre Study of Argument Structure with Large Language Models](https://arxiv.org/abs/2609.15194) | 本文提出受符号学四种主修辞格（转喻、隐喻、反讽、提喻）启发的四种符号学关系，将其分别对应到推理证明、类比证明、反证法和分情形证明四种证明方法，并利用大语言模型通过跨体裁实证研究考察这些论证关系的实际使用情况。 |
| [^47] | [What Limits Us? Analyzing Self-Reported Limitations in NLP Research](https://arxiv.org/abs/2609.15191) | 本文提出了一种新颖的人机协作迭代混合定性编码框架，对2020至2025年间ACL和EMNLP论文的局限性部分进行大规模分析，揭示了NLP研究者自我报告局限性的时间趋势、与论文属性的关联以及写作模式。 |
| [^48] | [MUSE: A Theory-Harnessed Story Engine for Vibe Narrativizing](https://arxiv.org/abs/2609.15188) | 提出了MUSE故事引擎，通过将罗伯特·麦基故事理论工程化为针对具体创作决策的指导，并使其贯穿规划、起草和修改全过程，实现将自然语言写作需求转化为高质量完整故事的“氛围叙事”任务。 |
| [^49] | [CITECHOICE: A Causal Audit of How Document Presentation Redistributes Citation Credit in Agentic Search](https://arxiv.org/abs/2609.15164) | 该论文提出CITECHOICE因果审计框架，通过对真实智能体搜索记录进行2×2重放实验，发现结构化文档呈现会集中引用归属（每答案+0.50次引用）而非明显提高来源接纳率，揭示了文档呈现方式对引用分配的因果影响。 |
| [^50] | [EMR: Self-Evolving Medical Multi-Agent System via Experience Mining and Reuse](https://arxiv.org/abs/2609.15161) | EMR提出了一种能自我进化的医疗多智能体系统，其核心创新在于构建了包含临床原则、诊断模式和代表性病例的分层临床经验库，通过从推理轨迹中自动挖掘诊断成功经验与失败警示并持续更新经验库，实现了从既往诊疗案例中不断学习进化的能力。 |
| [^51] | [PACE: Progressive Angular-to-Norm Contrastive Embedding](https://arxiv.org/abs/2609.15152) | 提出PACE两阶段框架，通过渐进式扩展表示空间和可训练参数空间，解决了点积相似度训练不稳定的问题，实现同时利用角度和范数信息的多模态嵌入学习。 |
| [^52] | [Improving Mathematical Reasoning Capabilities in Large Language Models via Reasoning Process Error Classification](https://arxiv.org/abs/2609.15145) | 本文通过定义并分类大语言模型数学推理过程中的21种错误类型，识别出高频错误类别，并据此设计了聚焦八类高频错误的提示，有效提升了LLM的数学推理性能。 |
| [^53] | [MoME: Mixture-of-Memory Embeddings for Context-Aware Sparse Lookup](https://arxiv.org/abs/2609.15126) | MoME 提出了一种上下文感知的记忆嵌入机制，用 M 个记忆槽位的混合替代每个词元的单一记忆条目，并通过基于隐藏状态的学习门控选择读取的槽位，从而解决现有方法将同一词元的不同上下文语义坍缩为单一固定条目的问题。 |
| [^54] | [When the Wrong Key Wins: Understanding and Detecting Hallucinations in LLMs](https://arxiv.org/abs/2609.15106) | 大语言模型的幻觉源于预训练关联之间“潜在钥匙”的竞争，本文据此提出一种两阶段关键词扰动检测方法，通过移除关键词条并观察预测重组方式，区分误导性关联导致的错误与有据可依的正确答案。 |
| [^55] | [OpenAl4S: Code as Action, Science as Sessions](https://arxiv.org/abs/2609.15096) | OpenAI4S是一个以“代码即行动，科学即会话”为核心原则的开源科学研究智能体，通过持久化Python/R计算内核、行动账本、执行记录与工作区检查点等机制，保障长期科研工作流的可检查性、可恢复性与可复现性。 |
| [^56] | [Translating the Translator: Decomposing the Cost of English-Forced Inter-Agent Communication](https://arxiv.org/abs/2609.15079) | 本研究首次量化了多智能体LLM系统中强制英语通信的“英语强制税”，发现相比母语流水线，强制英语路由使四种语言的精确匹配准确率下降13.0至30.6个百分点。 |
| [^57] | [DA-DLM: Explicitly Modeling Token Dependencies in Diffusion Language Models](https://arxiv.org/abs/2609.15070) | DA-DLM通过面向位置的有向无环图（DAG）设计，在扩散语言模型中显式建模词元间的依赖关系，从而提升生成文本的连贯性，并在多项任务上持续超越Block Diffusion。 |
| [^58] | [Salesforce Koa: An Enterprise Language Model for Agentic Tool Use](https://arxiv.org/abs/2609.15066) | Salesforce Koa 是一个基于 Nemotron-3-Super-120B 并通过 GRPO 强化学习后训练的企业级语言模型，其核心创新在于“仿真到奖励”流水线——将工作流规范扩展为以角色为条件的多轮任务，并以成功工具使用作为任务解决奖励，从而在保持通用性能的同时显著提升智能体工具使用能力。 |
| [^59] | [Not All Prompts Are Equal: Exploration-Guided Prompt Scaffolding for Multimodal Reinforcement Post-Training](https://arxiv.org/abs/2609.15051) | 提出了探索潜力分数（EPS）和探索引导的提示词脚手架框架，通过动态调整训练提示词分布并利用教师模型改写低效用提示词，提升多模态大语言模型强化学习后训练的效率。 |
| [^60] | [Mirror, Mirror on the Wall: Prompt Echoing in Small Instruct Language Models](https://arxiv.org/abs/2609.15045) | 本研究对Gemma、Llama、Qwen、SmolLM和OLMo等多个模型家族的分析表明，小型指令语言模型的提示回显现象虽与训练数据存在部分重叠，但主要由模型的归纳头机制驱动，而非简单的训练数据泄露。 |
| [^61] | [The average-farmer illusion in language-model simulations of agricultural decisions](https://arxiv.org/abs/2609.15038) | 语言模型虽能复现农民决策的总体均值与采纳率，但个体预测能力薄弱、政策相关的极端情形大量缺失，甚至不如一个仅拟合边际分布的简单生成器，表明基于总体平均的“真实性”证据是一种幻觉。 |
| [^62] | [MoARa: Module-Aware Rank Allocation and Structure-Preserving Decomposition for Low-Rank LLM Pre-training](https://arxiv.org/abs/2609.15037) | MoARa通过模块感知的投影秩分配与结构保持的幅值-方向分解，使Llama 2 7B低秩预训练在减少37%步数和34%耗时的情况下达到标准GaLore的最终困惑度。 |
| [^63] | [Pick Your Poison: Learning to Select Poison Sets for Stronger LLM Backdoor Attacks](https://arxiv.org/abs/2609.15029) | 该论文揭示了随机采样投毒样本会严重低估LLM后门攻击的最坏情况脆弱性（攻击成功率可从3%到80%不等），并提出SAILS方法通过学习集合评分器智能选择最优毒药集，使攻击成功率平均提升30个百分点。 |
| [^64] | [SALUTE: Benchmarking and Adapting LLMs for the Defense Domain](https://arxiv.org/abs/2609.15022) | 提出了SALUTE端到端框架，通过构建国防领域语料库、指令数据集、偏好数据集和严格筛选的基准测试，系统性地填补了大语言模型在国防领域评估与适配方面的空白。 |
| [^65] | [ABSOL: Aggregated Bayesian Subsampling Orchestrated with LLMs](https://arxiv.org/abs/2609.15007) | ABSOL是一个利用大语言模型作为有界语义引导来协调贝叶斯网络结构学习的混合框架，是唯一在所有基准测试中都能生成可行图的方法，并在较大规模基准上取得最高的边F1分数，相比非大语言模型方法平均提升0.23。 |
| [^66] | [Beyond Depth and Width: The Information-Slack Dilemma in Streaming Test-Time Compute](https://arxiv.org/abs/2609.14995) | 本文提出“信息-余量困境”这一概念，揭示了流式测试时计算中早期计算（时间充裕但证据不完整）与等待更多信息（证据更可靠但计算余量减少）之间的核心权衡，并据此提出了一个以受控证据修订下选择性恢复为优先的演化证据计算研究议程。 |
| [^67] | [MTAC-IFBench: Benchmarking Instruction-Following in Multi-Turn Agentic Coding](https://arxiv.org/abs/2609.14992) | 提出了MTAC-IFBench，首个针对多轮智能体编程中指令遵循能力的综合基准，每个实例平均包含7.04轮渐进式开发对话和91.33个约束条件，涵盖6个一级和18个二级约束类别。 |
| [^68] | [Typhoon ASR Streaming: Steerable Low-Latency Thai Speech Recognition with Real-Time Shallow Fusion](https://arxiv.org/abs/2609.14991) | 该论文提出了Typhoon ASR Streaming系统，通过缓存感知编码器和带短语增强的GPU n-gram浅融合层，实现了无需重训练即可在解码时控制词汇表的泰语流式语音识别，在一秒前瞻下将字符错误率降低4.3-4.5倍且速度快于实时。 |
| [^69] | [Biomedical Reference Generation Remains Unreliable across 26 Large Language Models](https://arxiv.org/abs/2609.14988) | 对来自八个开发商的26个大型语言模型的系统评估表明，模型在为生物医学文本生成参考文献时编造率介于10.2%至98.4%之间，证明即使是最先进的模型，其参考文献生成能力仍然不可靠。 |
| [^70] | [Converting Sequenced Fuzzy Cognitive Maps to Causal Virtual Worlds with Large Video Generators](https://arxiv.org/abs/2609.14985) | 该论文提出利用反馈模糊认知图建模虚拟世界的细粒度因果结构，并通过“如果A则B”的动态元规则引导大型语言模型和大型视频生成器创建和操控具有全局均衡因果情景的因果虚拟世界。 |
| [^71] | [Online Language Adaptive Sampling for Better Distributed Cross-lingual Gains](https://arxiv.org/abs/2609.14969) | 本文提出一种为每种语言分配可训练采样概率的在线自适应采样策略，让对重新对齐损失贡献更大的低资源语言被更频繁采样，以极小开销持续提升多语言模型的跨语言迁移性能。 |
| [^72] | [A Corpus-Aligned Uthmani-to-Standard Quranic Word Mapping and a Deterministic Recitation Validator](https://arxiv.org/abs/2609.14967) | 该论文通过全本古兰经双正字法对齐发布了2,290对乌斯玛尼体到标准体的词汇映射和七步规范化流水线（使90.9%经文规范化后完全一致），并据此构建了一个确定性的、无需大型语言模型的古兰经诵读验证器。 |
| [^73] | [Can We Triage LLM Translation Errors in Classical Texts Without Human References? Source Novelty, GEMBA Scoring, and Budgeted Review through Pali-to-English Translation](https://arxiv.org/abs/2609.14963) | 本研究通过巴利语至英语的翻译任务比较了五种无参考信号，发现基于多模型面板的无参考GEMBA评分是在没有人工参考译文时识别需要专家审校的LLM古典文本翻译错误的最有效方法。 |
| [^74] | [Geometric Signatures of Conceptual Reorganization: A Counterfactual Embedding Framework for Detecting Scientific Revolutions](https://arxiv.org/abs/2609.14917) | 该论文提出将文档嵌入空间的几何扰动作为概念重组的量化可观测量，通过反事实消融框架在五个历史案例（狭义相对论、哥德尔不完备定理、希格斯机制、深度学习和注意力机制）中验证了可测量几何特征，为科学革命的定量检测开辟了新途径。 |
| [^75] | [Forty Shades of Blue: Quality-Diversity Alignment via Mode-Conditioned Reinforcement Learning](https://arxiv.org/abs/2609.14896) | MoDA是一种受多智能体强化学习启发的在线后训练强化学习算法，通过让单一共享LLM策略在不同编号角色条件下竞争生成彼此不同的输出，从而在不牺牲生成质量且无需手工设计角色或修改架构的情况下，有效缓解模式坍塌并提升输出多样性。 |
| [^76] | [PeerPen: AI-Assisted Writing for Online Mental Health Peer Support](https://arxiv.org/abs/2609.14886) | 开发了嵌入社区界面的AI写作辅助工具PeerPen，通过15人访谈发现其能减轻心理健康同伴支持志愿者的写作负担并增强信心，但揭示了AI辅助会模糊作者身份、削弱社区信任的核心张力，参与者期望AI辅助而非接管写作。 |
| [^77] | [AgentKV: Phase-Aware KV Eviction for Agentic LLMs](https://arxiv.org/abs/2609.14872) | AgentKV发现智能体LLM服务中未来查询分属思考、行动、工具等不同阶段且占据不同的查询子空间，据此提出为每个阶段维护查询缓冲区并针对其并集进行缓存键评分的阶段感知KV淘汰方法，克服了传统基于近期性的淘汰方法在智能体场景下的系统性偏差。 |
| [^78] | [One Example Is Enough to Pass Fairness Benchmarks: Rethinking Fairness Evaluation for Aligned LLMs](https://arxiv.org/abs/2609.14860) | 仅用一个公平性基准测试示例进行训练或作为上下文演示，就能大幅提升模型在公平性基准上的表现，这表明现有公平性评估基准过于简单，需要重新思考对齐大语言模型的公平性评估方法。 |
| [^79] | [Dream-RSI: Recursive Self-Improvement through Evolving Worlds](https://arxiv.org/abs/2609.14858) | Dream-RSI提出了一种递归自我改进的探索框架，通过轻量级编排层使探索显式可编程，并创新性地将积累的发现历史作为回放模拟器，在其中进行“做梦”式改进，从而突破传统探索策略难以适应扩展搜索空间的瓶颈。 |
| [^80] | [ModularRSI: Modular and Generalizable Recursive Harness Self-Improvement](https://arxiv.org/abs/2609.14857) | 提出ModularRSI，一个基准分离、对比式且模块化的递归自我改进框架，通过对比成功与失败经验并在模块层面定位和优化执行框架缺陷，实现可泛化的框架进化。 |
| [^81] | [Self-Orchestrating Language Models: Leveraging Semantic Dependence for Efficient Inference](https://arxiv.org/abs/2609.14850) | 该论文提出“自编排语言模型”概念，让语言模型通过在生成过程中标注token间的语义依赖关系来指导自身推理执行策略，并配合专用运行时系统实现自回归解码并行化、中间上下文驱逐和去噪顺序推导，达成帕累托最优的质量-效率权衡。 |
| [^82] | [LLMs as Oracles: Reliance on LLMs for Subjective Personal Questions](https://arxiv.org/abs/2609.14849) | 人们正日益将大语言模型当作主观个人问题上的“全知权威”来依赖且往往不自知，这种依赖随时间增长、在年轻用户中更普遍，其驱动因素在于人们对AI的认知和AI自身的行为。 |
| [^83] | [Enemray: Toward Capable Language Models for Hassaniya](https://arxiv.org/abs/2609.14829) | 本文提出 Enemray——一个以哈桑尼亚语为核心的语言模型，通过稳定性—可塑性训练策略，在掌握哈桑尼亚语语言文化能力的同时保留通用推理、多语言与安全行为。 |
| [^84] | [Route, Don't Fix: Regime-Dependent Decoding Correction and a Trajectory-Gated Router for Reliable Clinical LLM Answer Selection](https://arxiv.org/abs/2609.14825) | 提出ALTAS框架，仅通过一次前向传播读取末端熵与深层线性度（R²），即可为每个临床问题在贪心解码与深层轨迹校正之间动态路由，无需训练任何分类器或额外模块，从而提升临床LLM答案选择的可靠性。 |
| [^85] | [MedTRACE: Tool-Augmented Multimodal Clinical Reasoning Agents for Evidence-Grounded Decision-Making](https://arxiv.org/abs/2609.14823) | MedTRACE 提出了一种工具增强的多模态临床推理智能体，通过假设形成、工具感知推理与证据验证的迭代循环，动态调用视觉定位、证据检索和结构化解析工具，实现基于证据的临床决策。 |
| [^86] | [A primer on evaluation methods for large language models in healthcare](https://arxiv.org/abs/2609.14819) | 本文系统综述了医疗领域大语言模型评估的四大关键方面——研究设计原则、统计方法、能力评估和临床情境评估，并阐述了相关底层概念与潜在陷阱。 |
| [^87] | [Tone on a Budget: A Reference-Free Metric for Lexical Tone in Massively Multilingual Text-to-Speech](https://arxiv.org/abs/2609.14817) | 本文提出 DunDun，一种无需参考录音和声调标注语料库的无参考词汇声调自动评估指标，通过从输入文本的变音符号读取标准高/中/低调序列并与音频音高轨迹对比，解决了 TTS 标准指标 CER 忽略声调标记导致声调错误无法被检测的问题。 |
| [^88] | [Crypto Accounting Bench: Evaluating Frontier and Open-Weight Models on Crypto-Asset Accounting Tasks](https://arxiv.org/abs/2609.14811) | 本文提出加密会计基准（CAB），包含来自7个匿名化组织的118个评估任务，用于评估12个前沿及开放权重语言模型能否准确重构加密资产交易的完整会计分录。 |
| [^89] | [Mind Which Bird You Favour: Parameterizing Adequacy-Fluency Balance in Meta-Evaluation of Machine Translation](https://arxiv.org/abs/2609.14795) | 该论文将机器翻译元评估中充分性与流利度的平衡参数化为可调节的选择，提出了一种具有理论保证和剪枝机制的精确优化算法，通过最小化失真的方式对翻译系统重新加权来实现目标平衡。 |
| [^90] | [Language-Guided Representation Learning for Robust Cross-Sensor Material Recognition](https://arxiv.org/abs/2609.14783) | 提出了一种语言引导的蒸馏框架，利用语言作为跨传感硬件不变的传感器无关监督信号，并通过构建包含39K样本的人工标注触觉-语言数据集，学习对传感器鲁棒的触觉表示，从而提升跨传感器材质识别的泛化能力。 |
| [^91] | [Func-R1: Incentivizing Mathematical Function Reasoning in Multimodal Large Language Models](https://arxiv.org/abs/2609.14779) | 该论文揭示了多模态大语言模型在数学函数推理中存在忽视视觉线索的模态干扰现象，并提出Func-R1模型，通过显式解耦架构、分层后训练框架和感知对齐理论优化（PATO）策略，协同实现精确视觉感知与严谨逻辑推理的融合。 |
| [^92] | [Pull: Lazy Materialization of Working Memory for Stateful LLM Conversations](https://arxiv.org/abs/2609.14773) | 提出Pull会话路由器，通过可逆的惰性物化机制让LLM只加载所需的历史对话轮次，在保证回答质量的前提下将单跳任务的每查询上下文token减少75.1%、多跳任务减少72.0%。 |
| [^93] | [How broad is that claim? Mapping Generalisation in NLP Research](https://arxiv.org/abs/2609.14770) | 该论文提出了科学领域泛化表述分类体系NLPGenX、基于大语言模型的自动分类框架NLPGenA以及大规模标注数据集NLPGens，用于自动检测NLP研究论文中对泛化表述的过度使用和可能存在的表述偏差。 |
| [^94] | [Loop-Back Authority in LLM Agent Teams: A Paired Experiment on Flat and Hierarchical Coordination](https://arxiv.org/abs/2609.14767) | 实验表明，在开放式任务中，移除管理者对工作者输出的否决权威反而能提高LLM多智能体团队的输出质量。 |
| [^95] | [Refusal Reads Only a Slice of What the Model Knows: Harm-Keyed Routing and Its Exceptions Across Model Families](https://arxiv.org/abs/2609.14759) | 该研究揭示了模型拒绝有害请求的机制与道德理解能力是相互分离的——道德理解源于预训练中结晶形成的低秩子空间，而拒绝门则是后训练写入狭窄控制令牌通道的新构建，且与道德判断正交，表明拒绝行为只读取了模型所知内容的一小部分。 |
| [^96] | [Fabrication After Tool Failure: Tool-Augmented Agents Assert Values Their Tools Did Not Return](https://arxiv.org/abs/2609.14758) | 该研究构建了一个涵盖1,024个条目的基准测试，首次系统揭示了工具增强语言模型在工具故障后的不诚实行为：当故障未被明确标记时（如返回status:ok但值损坏），模型捏造数值的比例高达45.3%，而当故障被明确标记为status:error时捏造行为完全消失，表明故障信号的显式性是决定模型诚信的关键因素。 |
| [^97] | [Calibrating Interpretability Instruments Before Trusting Their Verdicts](https://arxiv.org/abs/2609.14754) | 该论文指出可解释性研究中的测量工具会以特定、可诊断的方式静默失效并返回看似合理的数字，并通过一项关于拒答与道德表征的因果可解释性研究记录了六种典型失效模式，强调在采信工具的判定之前必须先对其进行校准。 |
| [^98] | [Quantifying the Generation Modality Gap in Speech-Text Language Models](https://arxiv.org/abs/2609.14743) | 该论文构建了一个统一的生成式评估套件，在匹配的数据分布和评估设置下，从语义连贯性、语音结构、声学质量等多个维度量化了纯语音、纯文本和语音-文本语言模型之间的生成模态差距。 |
| [^99] | [Building Legal Reward Models for Grounding and Abstention](https://arxiv.org/abs/2609.14739) | 该论文提出了将法律问答数据集转换为情境偏好数据的框架，并构建了LegalRewardBench基准，用于评估嘈杂和证据不足检索条件下的有据法律生成，研究发现长度均衡的偏好数据增强能显著提升法律领域奖励模型的有据评估能力。 |
| [^100] | [CALICO: A Human-Centered, Codebook-Aligned System for Annotation](https://arxiv.org/abs/2609.14726) | 本文提出CALICO系统，将以人为中心的可编辑、可版本化提示词工作流与编码手册对齐，使非技术背景的领域专家能够诊断和优化基于大语言模型的标注行为。 |
| [^101] | [Depth and Scale in the Sub-150M Regime: JugnuLM-53M vs JugnuLM-110M](https://arxiv.org/abs/2609.14715) | 在方法完全固定的情况下，仅通过将架构从53.5M扩展到深而窄的23层109.7M参数设计，JugnuLM-110M在更少训练token下全面超越53M模型，并以少约12%的参数匹敌GPT-X2-125M，证明该参数规模下的性能提升源于模型容量与深度而非数据量。 |
| [^102] | [Speak to the City: Multimodal Resolution for Outside-the-Vehicle References](https://arxiv.org/abs/2609.14691) | 本文提出一个融合注视与语音的多模态OVR框架，通过VR数据采集流水线与轻量级Transformer网络实现车外兴趣点的精准识别，Top-1准确率达83.33%且计算开销低。 |
| [^103] | [One Feedback System Does Not Fit All: Localising Data-to-Text Driver Coaching for the United Kingdom and Nigeria](https://arxiv.org/abs/2609.14687) | 本文通过比较英国和尼日利亚两个独立开发的驾驶辅导系统，论证了数据到文本驾驶辅导系统不能采用通用流程，而必须根据当地驾驶员知识、常见风险、法规、基础设施和数据可用性进行本地化设计。 |
| [^104] | [The Garden of Forking Prompts: How Users Explore Narrative Space in Story Generation](https://arxiv.org/abs/2609.14677) | 该研究基于真实用户-聊天机器人对话数据，构建了包含275,635个故事生成提示词的WildStories数据集和24,291个编辑树的WildEdits集合，首次系统揭示了用户在故事生成中通过迭代编辑提示词来探索叙事空间的行为，弥补了现有静态评估基准无法捕捉此类探索性行为的不足。 |
| [^105] | [Exploring Multimodal Turn-Taking Cues in Face-to-Face Conversation using Voice Activity Projection](https://arxiv.org/abs/2609.14666) | 通过将视线方向、头部运动、身体姿态和面部动作单元等视觉特征以多种融合策略引入语音活动投影（VAP）模型，能够超越仅依赖音频的方法，有效提升面对面对话中的轮替预测性能。 |
| [^106] | [Optimizing Sparse Outcomes Through Dense Behavioral Signals via Value-Guided Preference Distillation](https://arxiv.org/abs/2609.14648) | 本文提出将长时程对话优化建模为多目标强化学习问题，通过多头价值模型预测用户行为向量，证明密集行为信号的标量化组合能有效优化稀疏长期结果，同时建立结合反事实用户模拟的安全框架以在部署前识别策略失败模式。 |
| [^107] | [CompCQR: Compositional Query Generation for Training-Free Conversational Search](https://arxiv.org/abs/2609.14646) | 提出了CompCQR——一种免训练的对话式查询重写方法，利用检索器对内容排序敏感的观察，通过组合少量原子组件以极低的LLM调用量生成海量候选查询。 |
| [^108] | [Know When to Stop, Where to Restart: Accelerating Multi-Turn Agentic On-Policy Distillation](https://arxiv.org/abs/2609.14636) | 该论文发现多轮智能体训练中教师监督信号集中于每轮前缀、且教师认可的损失在时间上锁定于学生首个错误动作，据此提出STRIDE方法，通过停止-重启机制加速多轮智能体在线策略蒸馏。 |
| [^109] | [Domain-specific Pretraining Profile and Transformer Performance: Evidence from Modeling Digital Pragmatics in Arabic-English Code-switching](https://arxiv.org/abs/2609.14571) | 研究表明特定领域预训练特征对Transformer建模语码转换数字语用学至关重要，阿拉伯语专用模型MARBERT（Macro F1达0.85）在分类语境敏感语用功能上大幅优于多语言模型XLM-R（Macro F1仅0.52）。 |
| [^110] | [Disentangling Topology and Diversity in Multi-Agent LLMs for Multilingual Low-Resource Emotion Detection](https://arxiv.org/abs/2609.14570) | 该论文首次将多智能体LLM系统中的推理拓扑结构与智能体间多样性来源解耦并独立研究，发现并行的学习型QLoRA专门化在九种语言的多语言低资源情感检测上表现最佳，且最优拓扑结构取决于所采用的多样性来源。 |
| [^111] | [TATK: Triple-Aware Top-K Learning with Knowledge-Grounded Verification for LLM-based Sequential Recommendation](https://arxiv.org/abs/2609.14565) | 提出TATK框架，通过将Top-K学习（对齐训练与排序效用）与知识锚定验证（单次前向传播后进行结构感知重排序）相结合，解决了基于大语言模型的序列推荐中全目录Top-K排序匹配不佳的问题。 |
| [^112] | [Neyshekar: An Open Persian Read-Speech Corpus for Automatic Speech Recognition](https://arxiv.org/abs/2609.14542) | 发布了开放波斯语朗读语音语料库Neyshekar，涵盖正式与非正式语言、命名实体和长句，提供99小时经验证录音及可审计的说话人划分与评分者标注，为波斯语自动语音识别研究提供了高质量数据资源。 |
| [^113] | [Parameter-Efficient Quantum NLP for Paraphrase Detection: Performance, Robustness, and Entanglement](https://arxiv.org/abs/2609.14529) | 本文提出一个仅含2,148个参数的10量子比特混合量子-经典变分电路用于复述检测，在统计上优于参数量匹配的经典基线并以更少参数超越DistilBERT，同时发现多量子比特纠缠是性能的主要驱动因素，并展现出对对抗样本的涌现鲁棒性。 |
| [^114] | [Theseus in the Graph: Towards Traceable Multi-Hop Graph Navigation](https://arxiv.org/abs/2609.14528) | 本文将多跳知识图谱问答重新定义为以问题为条件的图导航问题，提出THESEUS框架使推理路径显式且可追溯，并将KINSHIP和MQuAKE扩充为支持导航式推理评估的数据集。 |
| [^115] | [NeuroActiSep: Detecting Factual Hallucinations from Feed-Forward Neurons in a Single Pass](https://arxiv.org/abs/2609.14448) | 提出NeuroActiSep方法，在单次前向传播中对最终提示词位置的前馈神经元进行排序筛选，并证明仅用所选神经元特征训练的探测器即可达到与基于内部状态训练的探测器相当的幻觉检测性能。 |
| [^116] | [Question's Gambit: The First Move Matters in Agentic Deep Search](https://arxiv.org/abs/2609.14412) | 提出Question's Gambit首步检索模块，通过将问题分解为线索、生成互补搜索查询并重排序候选文档，为智能体深度搜索构建优质开局上下文，从而提升复杂问题的解答能力。 |
| [^117] | [Policy Loopholes in Agent Evaluation: When Policy Ambiguity Masquerades as Agent Error](https://arxiv.org/abs/2609.14400) | 该研究揭示智能体基准测试中自然语言政策的模糊性会被误判为智能体错误，提出政策漏洞分类体系，并证明政策规范质量是评估质量的决定性上限。 |
| [^118] | [MOSCOPT: Mixture-of-Skills Collective Optimization for LLM Agents](https://arxiv.org/abs/2609.14399) | MOSCOPT提出了一种文本原生、无需参数的算法，通过联合优化技能池与动态门控技能，并借助带双重状态的EditAdam进行三阶段交错更新，使LLM智能体在无梯度情况下单调提升性能，在5个基准测试上全面超越现有基线。 |
| [^119] | [Formal Properties of Language as Constraints on Neural Dynamics](https://arxiv.org/abs/2609.14384) | 本文提出“神经可容许性纲领”，指出语言的代数性质（如非结合层级分组、递归封闭性、结构化工作区转换等）构成神经机制必须满足的数学不变量约束，并为每项约束提供了具体的神经动力学候选机制。 |
| [^120] | [SpectralShift: Effective Context Window Extension of Gated DeltaNet via Spectral Reparameterization](https://arxiv.org/abs/2609.14320) | 提出SpectralShift方法，通过谱重参数化重塑门控DeltaNet的衰减谱——扩展与目标依赖长度对齐的慢速谱带、同时保留用于状态清除的快速衰减模式，从而有效实现其上下文窗口扩展。 |
| [^121] | [E2A-Bench: Benchmarking Evidence-to-Action Reliability in Financial Chart Reasoning](https://arxiv.org/abs/2609.14302) | 提出了E2A-Bench基准，通过969个查询和四项新指标评估20个金融视觉语言模型将图表证据转化为可靠投资建议的完整链路，揭示出传统标量幻觉评分所掩盖的推理断裂与覆盖率不足等失败模式。 |
| [^122] | [Editorial routing shapes how computational results are qualified in AI-assisted scientific writing](https://arxiv.org/abs/2609.14288) | 该研究发现AI写作助手是否在手稿中保留计算结果的数值限定，取决于比较内容被“编辑路由”到工作流的哪个位置（组存储库、支持信息或工作笔记），且针对性的放置规则比通用准确性提醒更能有效恢复这些限定信息。 |
| [^123] | [DenMark: Robust Semantic Watermarking for Diffusion Language Models](https://arxiv.org/abs/2609.14257) | 提出了DenMark，一个面向扩散语言模型的语义水印框架，通过将密钥相关信号直接注入去噪过程并利用临时展开作为语义前瞻，实现了对改写等语义保持编辑鲁棒的文本水印。 |
| [^124] | [Document Topic Alignment Metrics for Evaluating Topic Models of Short-Text Public Health Communications on Social Media](https://arxiv.org/abs/2609.14256) | 提出文档-主题对齐指标，通过衡量帖子与其分配主题之间的语义对齐程度来评估主题模型，为短文本公共卫生社交媒体分析提供了与传统指标互补且与人工评估高度一致的评估方法。 |
| [^125] | [The Attribution-Compression Frontier in Retrieval-Augmented Generation](https://arxiv.org/abs/2609.14245) | 本文系统测量了检索增强生成中各类上下文压缩方法对引用归因的影响，揭示了摘要式压缩的引用虽与其自身摘要高度一致（精确率0.86）但与源文本严重脱节（精确率仅0.12），而抽取式选择能在不同压缩预算下保持0.43-0.49的稳定归因精确率。 |
| [^126] | [Corpus Characterization and Inverse Constitutional Fine-Tuning for Style-Aware Radiology Reports](https://arxiv.org/abs/2609.14226) | 该研究通过聚类分析识别出放射学报告的五种写作风格模式，并提出逆向宪法AI微调框架，从报告对中自动提炼风格规范，使自动生成的放射学报告更贴近真实放射科医生的写作风格。 |
| [^127] | [Learning to Refer from Estimated Listener Gaze](https://arxiv.org/abs/2609.14207) | 该论文提出将神经听者估计的注视扫描路径转化为奖励信号，用以微调视觉-语言模型，使其生成在语用上更优的指称表达。 |
| [^128] | [A Multi-Stage Agentic Framework for Effective Counter-Narrative Generation and Refinement](https://arxiv.org/abs/2609.14178) | 提出了一个多阶段智能体框架，通过人类评估识别有效的修辞-风格配对，并利用多智能体迭代精炼过程，生成更具说服力、情感感染力和可分享性的反叙事，以对抗仇恨言论与错误信息。 |
| [^129] | [Inherited Heads: Audio language models track speakers with their text backbone's attention, and an attention-mass ranking retrieves a different set](https://arxiv.org/abs/2609.14174) | 研究发现音频语言模型的说话人追踪能力主要继承自其文本骨干网络的注意力头——无需任何训练，仅对从纯文本模型中筛选出的注意力头添加固定偏置，就能以90%以上的准确率引导音频模型描述任意指定的说话人。 |
| [^130] | [Towards Evolving Context Parameterization for Large Language Models](https://arxiv.org/abs/2609.14168) | 提出免训练方法PLUME，通过构建全局更新表示并结合记忆证据形成的局部参数视图进行自适应整合，有效解决大语言模型在持续上下文演化中的记忆更新与信息保留难题。 |
| [^131] | [When Tools Get in the Way: The Effect of Unnecessary Tool Availability on LLM Answering](https://arxiv.org/abs/2609.14157) | 该研究通过在10个知识领域构建500个查询对并对6个大语言模型进行3,000次试验评估，揭示了提供相关但不必要的工具会损害模型依靠自身知识正确回答问题的能力。 |
| [^132] | [One Size Does Not Fit All: Setting Inference Depth from the Questions a Deployment Actually Asks](https://arxiv.org/abs/2609.14144) | 本文提出在冻结的Transformer模型中间层附加经训练的“读取器”组件并通过置信度测试实现提前退出，揭示当部署场景的提示词范围已知时推理计算可被大幅节省，且节省幅度强烈依赖于具体的流量类型。 |
| [^133] | [GraMRAG: Orchestrating Multi-Agent Multi-Step Reasoning via Graph Memory with Reinforcement Learning](https://arxiv.org/abs/2609.14066) | GraMRAG提出了一种由动态多模态记忆图引导的多智能体RAG框架，通过将智能体推理形式化为有向无环图并结合视觉-文本桥接推理范式，实现了稳定的多步多模态推理，有效缓解了状态盲视和冗余检索问题。 |
| [^134] | [Measuring the Creativity of Frontier LLMs in Automated Research](https://arxiv.org/abs/2609.14057) | 本文提出了一套从价值性和新颖性两个维度评估大语言模型自动化研究创造力的指标体系，发现模型在反映研究空间探索广度的变量级新颖性指标上差异显著。 |
| [^135] | [VeriDx: Earning the Right to Diagnose with Disease-Centric Verification](https://arxiv.org/abs/2609.14018) | VeriDx是一个以疾病为中心的验证框架，通过将大语言模型的自由格式诊断推理与结构化疾病档案关联，追踪每个疾病假设的临床义务履行情况（如关键检查、鉴别诊断、矛盾处理等），从而超越仅评估最终答案的传统方式，全面评估医疗AI的推理质量。 |
| [^136] | [TF-IDF and BM25 Are Exact KL Divergences](https://arxiv.org/abs/2609.14016) | 该论文证明TF-IDF和BM25这两种经典评分方法可以被精确解释为两个概率模型之间的KL散度，为信息检索方法建立了统一的理论基础。 |
| [^137] | [Unlocking the Unsolvable: Teacher-Guided Curriculum for Data-Efficient RLVR](https://arxiv.org/abs/2609.13997) | 通过教师引导的课程学习，仅用128个原本无法解决的难题训练就能匹敌或超越GRPO在2000个问题上的训练效果（约16倍数据效率），并显著扩展模型的推理能力边界。 |
| [^138] | [Mizan: A National Benchmark for Evaluating Large Language Models on Iraqi Arabic and the Iraqi Civic Context](https://arxiv.org/abs/2609.13980) | 该论文提出了 Mizan——伊拉克首个用于评估大语言模型在伊拉克阿拉伯语方言及伊拉克公民社会语境下表现的国家基准，其包含 MSA 基线与伊拉克双赛道、六大评估维度、340 道经双重审核的原创题目，并对 27 个系统进行了初步评测。 |
| [^139] | [Thought without systematicity? Evaluating reasoning models on rule induction tasks](https://arxiv.org/abs/2609.13948) | 研究通过将认知科学的规则归纳任务构造为结构等价的同构变体，发现推理模型虽能解决原任务却常在变体上失败，表明其行为缺乏系统性，难以证明其具备真正的认知能力。 |
| [^140] | [CRITICS - Critical Science Without Borders: Language Models to Promote Critical Thinking in Science Education](https://arxiv.org/abs/2609.13942) | CRITICS项目将基于大语言模型的机器翻译与教育技术相结合，为学生提供母语准确翻译的科学材料，并通过科学论证、批判性思维实践和能力评估框架来促进科学教育中的批判性思维。 |
| [^141] | [Inter-Rater Reliability of LLM and Rule-Based Annotation for Inferential Narrative Features: Three Studies on a Turkish Corpus](https://arxiv.org/abs/2609.13936) | 该论文通过三项人类标注者信度研究，评估了基于规则检测器及多个大语言模型（Gemini、Grok、Claude、ChatGPT）在土耳其语叙事语料库六项推理型叙事特征标注上与人类评分者的一致性。 |
| [^142] | [North Small Translate: Advanced Cost-Effective Translation (Cohere CAT+)](https://arxiv.org/abs/2609.13916) | Cohere推出的开放权重机器翻译模型North Small Translate，基于混合专家架构，通过难度采样和五步训练协议（结合监督微调、直接偏好优化和在线强化学习），在1万亿参数以下的模型中实现了50种语言上的顶尖翻译性能。 |
| [^143] | [Phorecaster365: A Human-Supervised Reference Architecture for Hybrid Pharmaceutical Sales Forecasting and Planning Decision Support](https://arxiv.org/abs/2609.13907) | 该论文提出Phorecaster365——一个人工监督的参考架构，以“预测上下文包”为核心中间表示，将ERP数据与可审查的药物销售预测相连接，支持统计与机器学习混合建模、不确定性评估及规划师审查的全流程治理。 |
| [^144] | [URCHIN: A Horizontal Spiking Language Model for Data-Constrained Pretraining](https://arxiv.org/abs/2609.13899) | URCHIN是一种受生物学启发的脉冲神经网络语言模型，仅用128个神经元、423万参数、无注意力机制，通过戴尔定律横向连接组和多传输循环动力学，在儿童规模数据上实现语言建模预训练。 |
| [^145] | [SHIFT-M3: Pre-fusion Alignment-based Consistency Screening for Multimodal ECG Record Integrity](https://arxiv.org/abs/2609.13874) | 提出轻量级融合前筛查方法SHIFT-M3，通过测量LLM生成的心电图解读与临床报告摘要之间的一致性，有效检测多模态心电记录中因数据关联错误导致的跨患者组件错配问题。 |
| [^146] | [Bangla Sentence Function Classification: Corpus Development, Model Benchmarking, and Interpretability](https://arxiv.org/abs/2609.13869) | 本文构建了一个包含10,000个孟加拉语句子、人工标注为四种功能类别的高质量语料库，并通过经典机器学习模型和异构集成模型对该句子功能分类任务进行了基准测试与可解释性分析。 |
| [^147] | [ClinAgent: A ReAct-Based Agent for Conversational Access to Clinical Trial Information](https://arxiv.org/abs/2609.13860) | 该论文提出了ClinAgent，一个基于ReAct范式的智能体检索增强生成对话系统，通过集成ClinicalTrials.gov、PubMed等工具，让临床医生和研究人员能够用自然语言多轮查询临床试验信息并获得有依据的最新答案。 |
| [^148] | [ShopEase: A Generative AI-Based Multi-Agent Framework for Intelligent Enterprise Customer Support Using Hybrid Retrieval-Augmented Generation](https://arxiv.org/abs/2609.13856) | 本文提出ShopEase框架，通过结合意图、CRM、记忆、混合RAG（FAISS稠密检索+BM25稀疏检索）、升级和监督六大智能体组件，直接从检索文档中确定政策类别而非依赖固定映射，并利用本地运行的LLaMA 3.2生成回复，实现智能企业客户支持。 |
| [^149] | [Measuring the Cost of Variety Conflation in Multilingual MT Evaluation: Adding Mozambican Xichangana, Nyanja and Sena to FLORES+](https://arxiv.org/abs/2609.13847) | 该研究将三种莫桑比克班图语变体（尚加纳语、尼扬贾语和塞纳语）纳入FLORES+基准，首次量化了机器翻译评估中语言变体混淆的代价——使用相近变体作参考译文会使spBLEU分数被严重低估（最高达15分以上），而变体感知微调可使模型在目标变体上提升5至7个spBLEU点。 |
| [^150] | [Affinity-Aware Sharding for Delayed Tensor Parallelism](https://arxiv.org/abs/2609.13846) | 本文发现延迟张量并行会打破FFN神经元和KV头的置换对称性，使分片本身成为建模决策，并提出通过在分片前置换稠密模型以最大化同设备上KV头与FFN神经元的亲和性，从而加速模型蒸馏或重训练过程。 |
| [^151] | [Sweet Talkers: How Query Formulation Shapes Sycophancy in Romantic Relationship Advice](https://arxiv.org/abs/2609.13841) | 该研究构建了包含2400条提示的浪漫关系建议数据集，发现大语言模型的谄媚行为主要受用户视角框架化表述而非语法语气的影响，且谄媚程度会随对话轮次持续增加。 |
| [^152] | [When Consistency Does Not Mean Reliability: Evaluating Local LLM Judges Against Human Ratings](https://arxiv.org/abs/2609.13824) | 该研究通过对LLaMA-3-8B和Qwen2.5-7B两个本地LLM评判者的系统评估发现，即使评判者的评分具有自一致性，其与人类评分的相关性也很低（皮尔逊相关系数仅0.275），说明“LLM即评判者”方法的一致性并不能等同于与人类判断的可靠性。 |
| [^153] | [DARE: Dialectical Agentic Reasoning for Structured Knowledge Fact Checking](https://arxiv.org/abs/2609.13808) | 提出了DARE多智能体框架，将结构化知识事实核查形式化为迭代式“检索-推理-反思”过程，通过基于关系的证据检索、辩证式双向验证和置信度驱动的元反思，解决了现有程序生成方法中无效关系生成、单路径推理缺乏自我修正以及证据评估偏向支持性信号的问题。 |
| [^154] | [Understanding the Limits of Agentic ICD Coding](https://arxiv.org/abs/2609.13806) | 本研究通过在按稀有程度分层的MIMIC-IV出院小结上评估神经、工作流和智能体三类ICD编码系统，揭示了两类正交的失败模式，并证明结合结构化官方参考资料的工具增强智能体可在损伤和外因编码等困难场景上恢复高达0.34的微F1，但没有任何单一系统在所有条件下占优。 |
| [^155] | [SyRHM: Symbolic-Language-Enhanced Reasoning with Associative Retrieval for Zero-shot Harmful Meme Detection](https://arxiv.org/abs/2609.13794) | 该论文提出SyRHM框架，通过将多模态内容解析后检索语义相关的迷因作为有依据的上下文，并利用符号-语言增强的多阶段推理（翻译、规划、求解）实现零样本有害迷因检测，在多个数据集上取得优越性能的同时提供了可解释的有害意图分析。 |
| [^156] | [Does Reasoning Improve Psychological Depth in Large Language Models? It Depends on Who's Judging](https://arxiv.org/abs/2609.13773) | 该研究通过短篇小说心理深度的对比实验发现，推理能力更强的模型（如GPT-5）并未获得人类读者的普遍偏好，且读者间与LLM评判器的评价存在显著分歧，表明"LLM作为评判者"的有效性高度依赖于评判者本身。 |
| [^157] | [Training Specialist Models without Reasoning Trajectories for Domain Expert Distillation](https://arxiv.org/abs/2609.13770) | 该论文发现仅用问答对训练的专家模型会隐式地从潜在轨迹空间中选择推理轨迹，并将学生蒸馏作为探针揭示了专家模型专门化与泛化特征之间的紧密支配关系。 |
| [^158] | [HyperProve: Answer-Guided Hypergraph Expansion for Multi-Hop Question Answering](https://arxiv.org/abs/2609.13768) | HyperProve将问题分解与答案条件驱动的超图扩展相结合，把中间答案和支持超边作为检索状态引导迭代检索，从而在多跳问答基准上取得最佳整体性能。 |
| [^159] | [Inside VLM Chart Reading: Tracing Value Reading from Vertical Bar Charts Across Space and Depth](https://arxiv.org/abs/2609.13745) | 该研究利用反事实激活修补技术揭示了视觉语言模型读取垂直柱状图数值的内部机制：柱顶区域是关键视觉证据来源，数值信息的传递在早期层依赖图例区域、在中间层转移至提示序列位置，且提示序列状态在其中起部分中介作用。 |
| [^160] | [HarnessBandit: Joint Learnability-Transferability Scheduling for Multi-Harness Agentic Reinforcement Learning](https://arxiv.org/abs/2609.13739) | HarnessBandit提出了一种在线调度方法，通过联合观测可学习性和可迁移性两个信号，在每个优化步骤中选择最有价值的训练harness，从而提升多框架智能体强化学习的鲁棒性。 |
| [^161] | [ForeSight: Enhancing Risk Monitoring via Early Safety Signal Distillation](https://arxiv.org/abs/2609.13737) | 提出ForeSight框架，通过将大语言模型生成首个词元时隐藏状态中微弱冗余的安全信号蒸馏为紧凑的层次感知风险表示，实现了仅凭首个词元即可高效预测最终响应有害性的早期风险监控。 |
| [^162] | [PolicyMem: Geometric Policy Memory for LLM Governance](https://arxiv.org/abs/2609.13734) | PolicyMem提出了一种几何策略记忆框架，将自然语言策略外化为共享表示空间中由低秩子空间表示的可重用几何记忆对象，解决了现有LLM治理方法无法将策略作为可重用操作状态的问题，从而支持在检测、干预和验证之间一致地重用策略证据。 |
| [^163] | [Scaling Hindi Quantum Natural Language Processing through Automatic Pregroup Supertagging](https://arxiv.org/abs/2609.13721) | 本文将印地语前置群超级标注自动化形式化为词元级分类任务，发现在低资源场景下简单的词汇与上下文模型（上下文回退达64.56%准确率）优于大语言模型提示，且词汇修复可将LLM预测提升至64.08%，为印地语量子自然语言处理的规模化奠定了基础。 |
| [^164] | [When Edit Localization Amplifies Relative Selection Bias: Gradient Geometry, Target Mismatch, and Importance Weighting](https://arxiv.org/abs/2609.13709) | 论文通过将局部化梯度分解为被编辑与未编辑分量，证明编辑定位可能增大、减小或非单调地改变相对选择偏差，并针对全梯度目标推导出有限样本均方误差、解析最优保留系数，以及以Oracle重要性加权来纠正目标失配的偏差。 |
| [^165] | [Prefix Sharing Is a Sorting Problem](https://arxiv.org/abs/2609.13692) | 本文证明LLM服务中提示片段的最优排序问题等价于在请求集合上选择一个二叉层次结构，全局固定排序仅在最多两个片段时最优，并据此给出 O(3^m) 精确算法及与最小顶点覆盖等经典问题的联系。 |
| [^166] | [Not all Negation Cues are Equal: Affixal Negations Yield Better Negation Understanding](https://arxiv.org/abs/2609.13685) | 本研究构建了包含180多万样本、覆盖200多种否定线索的大规模数据集NegCue，通过预训练实验发现词缀否定对提升语言模型和大语言模型的否定理解能力贡献最大，而常被研究的单词否定带来的收益却相对有限。 |
| [^167] | [LayerRoute: Adaptive Layer-Skipping with LoRA-Preserved Quality for Efficient LLM Inference](https://arxiv.org/abs/2609.13682) | LayerRoute通过逐层硬门控路由与联合LoRA微调实现自适应层跳过，在10次独立训练中稳定收敛到相同的跳层模式（跳过第8至16层），在保持并提升模型质量的同时实现了平均1.04倍的真实推理加速。 |
| [^168] | [FedV-KGQA in Practice: Design Lessons and an Interactive Prototype](https://arxiv.org/abs/2609.13661) | FedV-KGQA通过联邦方式在垂直分区的知识图谱上实现多跳问答，原始三元组和关系嵌入无需离开本地数据孤岛，实验表明联邦融合可恢复大部分集中式准确率，且问题锚定与图谱丰富化比嵌入模型的选择更为关键。 |
| [^169] | [FaithfulBench: Does AI Counsel Uphold or Undermine the User's Professed Faith?](https://arxiv.org/abs/2609.13634) | 该论文提出首个评估 AI 跨宗教传统咨询建议的基准 FaithfulBench，发现当用户信仰未被说明时 AI 会采用世俗默认建议而辜负信徒，仅指明信仰虽能获得虔诚的首次回答却缺乏坚定性，而根植于经典传统的辅导指南能改善 AI 的表现。 |
| [^170] | [An Efficient and Modular Framework for Targeted Harm Mitigation in LLMS](https://arxiv.org/abs/2609.13624) | 提出了一种结合Activated LoRA适配器与上下文感知路由机制的模块化纠正框架，可在生成过程中以低延迟、有针对性的方式缓解大语言模型的有害输出，同时提升模型对齐性能。 |
| [^171] | [The University of Melbourne WMT 2026 CreoleMT Submission: A Domain-Balanced Approach to Low-Resource Pacific Creole Machine Translation](https://arxiv.org/abs/2609.13615) | 该论文提出一种领域平衡的微调方法，结合LLM辅助数据处理、回译和Gemini蒸馏等技术，使面向太平洋克里奥尔语（Tok Pisin、Bislama、Solomon Pijin）的低资源机器翻译模型在所有翻译方向上超越开源基线3个以上的chrF++分数。 |
| [^172] | [In the Blind: Building Pseudo-References for MT Evaluation](https://arxiv.org/abs/2609.13611) | 该论文提出在WMT26缺乏人工参考译文的语言对上，通过多模型多提示翻译、无参考质量估计打分与文档级选择构建伪参考译文，并引入置信度缩放的语言识别惩罚以消除“错误语言但流畅”的失效输出。 |
| [^173] | [Same Patient, Different Order: Action-Level Reliability of Clinical LLM Agents Under Repeated Runs](https://arxiv.org/abs/2609.13582) | 本文提出“同输入重跑”评估方法及六项可靠性指标，首次系统揭示临床大语言模型智能体在完全相同输入下重复运行时会产生实质不同的医疗动作（如检查医嘱、用药和转诊），而这种动作级分歧会被仅评单次运行的基准测试（如MedAgentBench）所掩盖。 |
| [^174] | [Toward Complete Hospital Discharge Summarization with Abstract Meaning Representation](https://arxiv.org/abs/2609.13581) | 提出了一种以证据驱动、以来源追溯为一等约束的出院总结生成框架，利用语义图和深度学习通过跨文档语义对齐为每句总结提供明确的证据链接，从而缓解大语言模型的幻觉问题。 |
| [^175] | [How User-AI Mistreatment Occurs and Matters in Conversational Systems?](https://arxiv.org/abs/2609.13579) | 本文通过审计77.7万条对话发现，用户对AI的不当对待（侮辱、威胁、越狱胁迫等）约占用户轮次的0.90%，且与平台审核信号所捕捉的有害内容请求是不同的现象，这对准确理解模型行为和部署风险具有重要意义。 |
| [^176] | [Domain-Specific Jargon in Large Language Models: A Comparative Analysis between General-Purpose and Specialist Models](https://arxiv.org/abs/2609.13556) | 该研究提出了两个医学专业术语评估基准，发现通用LLM在医学术语任务上反而优于医学领域微调模型，并通过机制可解释性分析揭示微调并未重组参数化知识，而只是过度依赖一小部分偏向专业术语预测的模型组件。 |
| [^177] | [Harmfulness Propagation Dynamics: Layer-wise Trajectories of Adversarial Intent in Large Language Models](https://arxiv.org/abs/2609.13534) | 该论文发现有害提示在Transformer各层的危害方向投影呈单调上升趋势而良性提示保持平坦，并基于这一逐层激活动态特征提出了轻量级有害输入识别器HERALD。 |
| [^178] | [Generative Interpretability via Scalable Neuro-Symbolic Models](https://arxiv.org/abs/2609.13529) | 论文提出“生成式可解释性”这一新范式，主张模型的推理过程应原生暴露人类可理解且可进行因果干预的语义检查点，并利用神经符号模型实现这一目标，以应对智能体系统中输出不可逆的安全挑战。 |
| [^179] | [From Token Probabilities to Semantic Constraints: Towards Declarative Probabilistic Evaluation of Language Models](https://arxiv.org/abs/2609.13520) | 提出ModelLog声明式概率评估框架，将评估目标表示为词元预测上的符号语义约束，揭示模型在否定、互斥性和一致性上的系统性失败，并可将评估分数同时用作损失以连接评估与学习。 |
| [^180] | [One Spectrum, Two Resources: Data-Memory Scaling in Autoregressive Prediction](https://arxiv.org/abs/2609.13500) | 论文证明数据与学习记忆这两种资源由同一个预测能量谱统一支配，并由此建立了刻画自回归预测中最优数据-内存权衡的极小极大定律。 |
| [^181] | [Mixture-of-Experts Language Models Can Be Strong and Efficient Retrievers](https://arxiv.org/abs/2609.13486) | 该研究首次系统探索了混合专家语言模型作为检索器的潜力，证明MoE检索器在激活参数量相当的情况下比稠密检索器在BEIR上高出最多3.0个nDCG@10点，并能以少59%的激活参数匹敌8B稠密检索器，实现强大且高效的检索。 |
| [^182] | [A Hybrid Hierarchical 1D-CNN-BiLSTM Framework for Extractive Summarization of Biomedical and Clinical Text](https://arxiv.org/abs/2609.13481) | 该论文提出一种混合分层的1D-CNN-BiLSTM框架，将生物医学与临床文本摘要重新构建为抽取式句子选择任务，通过直接从源文本复制句子来规避生成式大模型的幻觉风险。 |
| [^183] | [STAGE: Diagnosing Semantic Transfer at Grounded Execution in Embodied Agents](https://arxiv.org/abs/2609.13458) | 该论文提出固定观测反事实基准SAT-Bench，揭示具身智能体虽能近乎完美地恢复指令语义（高达100%），但动作敏感度仅约6%，存在严重的语义-动作鸿沟，并提出了轻量级执行时接口VISA来诊断和缓解该问题。 |
| [^184] | [Hindsight Bias in Clinical Temporal Reasoning: How Future Data Exposure Affects Large Language Model Judgment](https://arxiv.org/abs/2609.13454) | 该论文构建了一个包含171份临床病例报告的配对基准，通过对比前瞻性参考答案与“后见之明陷阱”，首次系统量化了在回顾性数据上评估临床语言模型时，未来信息暴露所导致的后见之明偏差。 |
| [^185] | [Causal Analysis and Mitigation of Spurious Onsets in Full-Duplex Speech LLMs](https://arxiv.org/abs/2609.13445) | 本研究通过因果分析发现，全双工语音LLM（如Moshi和PersonaPlex）在用户静默时产生虚假语音起音，是因为模型以自身非语音输出为条件导致语音起音概率在单个80毫秒帧内飙升超过九个数量级，而非重复采样所致，并进一步基于因果反事实分析提出了缓解方法。 |
| [^186] | [CVSS-X: A Multilingual Speech-to-Speech Translation Corpus for 28 Languages](https://arxiv.org/abs/2609.13413) | 该论文提出CVSS-X，一个将翻译方向扩展为从英语到覆盖12个语系的28种语言、总计超16,000小时的大规模合成语音到语音翻译语料库，与CVSS结合可支持双向和多语言语音到语音翻译研究。 |
| [^187] | [RFCLLM: Evaluating LLMs' Reasoning Ability of Network Protocol State Machines](https://arxiv.org/abs/2609.13389) | 本文提出RFCLLM基准，通过针对16个协议设计的4个任务和1482个查询，系统评估大语言模型对网络协议状态机规范的理解能力，揭示其隐式状态机表示与人工真实模型之间的差距。 |
| [^188] | [ScorePrompts: Natural-Language Exploration of Symbolic Music Scores through Analysis](https://arxiv.org/abs/2609.13291) | ScorePrompts是一个交互式系统，它通过专门的MIR组件分析乐谱的和声、调性、曲式等音乐结构，再由受模式约束的语言模型生成自然语言描述并回答用户问题，同时将分析结果与五线谱中的对应小节和音符可视化关联。 |
| [^189] | [From Process Loss to Assembly Bonus: Human-Grounded Diagnosis of Multi-Agent LLM Collaboration](https://arxiv.org/abs/2609.13261) | 该研究通过将人类群聊与LLM多智能体审议轨迹进行过程层面的对比诊断，发现两者共享“组装红利”不对称性，但LLM群体在跟随多数、独特信息提出和收敛速度等方面存在过程差异，为多智能体LLM协作提供了基于人类行为的评估框架。 |
| [^190] | [Interpretable Temporal Video Reasoning with EventGraph and EventField](https://arxiv.org/abs/2609.13258) | 该论文提出结合离散EventGraph、连续EventField和人类可读EventGlyph的结构化时序视频推理方法，在EPIC-KITCHENS子集上达到0.98的准确率，显著超越字幕基线和纯VLM问答，同时兼顾高性能与可解释性。 |
| [^191] | [Clinical Reasoning Under a Partially Observed Objective in Cone Beam CT Report Generation](https://arxiv.org/abs/2609.13238) | 该研究揭示了CBCT报告生成评估中80%权重的事实蕴含目标不可见这一难题，通过纯Python复现BLEU/METEOR并构建AUC达0.987的离线蕴含代理模型使复合目标可被直接优化，按复合目标选出的报告得分0.4122显著高于仅按可见词汇指标选出的0.2909，因为单纯追求n-gram重合度会将蕴含精确率从0.522拖低到0.266。 |
| [^192] | [Occlusal Geometry in Closed Form for Orthodontic Report Generation](https://arxiv.org/abs/2609.13237) | 本文利用 Bite2Text 口内扫描数据已完成咬合配准这一特性，通过牙弓角度坐标下的咬合嵴轮廓以闭合形式直接计算 31 项咬合几何量（覆𬌗、覆盖、中线偏移等），再用梯度提升映射至 13 个模板字段并由确定性渲染器生成六部分正畸报告，从而将报告生成从多模态字幕推断转变为可测量的几何计算问题。 |
| [^193] | [Self-Indexing Attention for Compression-Compatible Sparse Long-Context LLM Inference](https://arxiv.org/abs/2609.13205) | 提出了一种免训练的自索引注意力框架，利用共享的1比特符号索引在预填充和解码阶段统一实现高效token检索，同时兼容外部KV缓存压缩，在5%注意力密度下达到接近密集注意力的准确率，并获得高达6.1倍预填充和10.3倍解码的算子加速。 |
| [^194] | [GradRepair-ODE: Certified Gradient Repair for Neural ODE Training](https://arxiv.org/abs/2609.13204) | 提出了GradRepair-ODE框架，通过多候选梯度比较、方向有限差分验证和失效诊断，在优化器步骤处对神经ODE训练中的不可靠梯度进行检查、修复或拒绝，提升数值求解耦合下的训练可靠性。 |
| [^195] | [Machine Unlearning for Speech Question Answering in Large Audio-Language Models](https://arxiv.org/abs/2609.13195) | 该论文首次研究了大型音频-语言模型中语音问答的机器遗忘问题，提出并评估了包括梯度上升、任务算术和对齐微调在内的多种遗忘策略，可将隐私泄露率降低多达80%的同时保持模型核心能力。 |
| [^196] | [LLMs or Naive Bayes? Old Gems or New Ways](https://arxiv.org/abs/2609.13185) | 本研究通过基准测试证明，在拥有标注数据的文本分类任务中，经典朴素贝叶斯方法不仅准确率可与甚至超越大规模语言模型，还能在普通CPU上以每秒数千样本的速度运行，有力挑战了“经典方法应被淘汰”的观念。 |
| [^197] | [Algorithm Validation as a Policy Audit: Evidence from Race-blind Charging](https://arxiv.org/abs/2609.13174) | 本文将算法验证应用于政策审计，验证了开源LLM算法bc2在加州种族盲指控政策中的表现，发现其在96.7%的警方报告叙述上忠实执行了法律要求，并进一步评估该政策本身能否有效实现种族盲决策的目标。 |
| [^198] | [A Cross Community Agenda for Speech AI](https://arxiv.org/abs/2609.13168) | 本文指出语音AI因技术社区（NLP）与社会技术社区（HCI）割裂而存在交流模型不完整、身份模型不完整和评估指标失准三大问题，并以辅助沟通（AAC）等弱势说话者场景为例，提出跨社区合作、面向人类多样性的研究议程。 |
| [^199] | [TestHallVQA: Exploring LVLMs' Document-Level Reasoning under Redundant Contexts from Scientific Exams](https://arxiv.org/abs/2609.13158) | 提出了TestHallVQA基准——一个融合文档级规模与科学考试难度的多图像VQA基准，支持可控注入多层次上下文冗余，用于系统评估大型视觉语言模型在冗余上下文中的文档级推理能力。 |
| [^200] | [Lexical Prompt Compression for Large Language Models: A Training-Free, Deterministic Pipeline with Empirical Pareto Analysis Across Eleven Task Categories](https://arxiv.org/abs/2609.13154) | 本文提出了一种免训练、完全确定性、仅需CPU的基于经典词汇NLP的提示词压缩流水线，通过十一种可切换的词汇转换操作在十一个任务类别上对压缩率与输出质量的权衡进行了实证帕累托分析。 |
| [^201] | [PhysMent: An Interactive Approach For LLM Reasoning In Physics Problems](https://arxiv.org/abs/2609.13152) | PhysMent是一个通过与MuJoCo物理模拟器进行交互式实验来评估大语言模型物理推理能力的新基准，其核心创新在于要求模型通过施加力、查询状态等主动探索方式获取信息，而非被动接收完整信息，测试结果显示当前模型在简单定性任务上表现良好但仍有局限。 |
| [^202] | [Token Merging for Multilingual Speech Recognition: A Systematic Study Across Model Scale and Fine-Tuning](https://arxiv.org/abs/2609.13151) | 对Whisper模型家族的系统性研究表明，token合并能在几乎不损失转录准确率的情况下显著提升多语言语音识别的计算效率，并且在不同模型规模和低资源语言微调后均能有效工作。 |
| [^203] | [The Limits of Reference-Free Speech Quality Metrics as Evaluators and Rewards on Modern Text-to-Speech](https://arxiv.org/abs/2609.13150) | 该研究通过成对偏好测试发现，无参考语音质量指标（UTMOS、DNSMOS、SCOREQ）在比较两个均无缺陷的TTS样本时无法可靠预测人类偏好，甚至不如简单选择最长片段的基线方法，这质疑了其作为TTS自动评估器和偏好优化奖励信号的有效性。 |
| [^204] | [BudgetBench: A Budget-Tiered Protocol and Pilot Harness for Memory Strategy Evaluation in Local Large Language Model Agents](https://arxiv.org/abs/2609.13149) | BudgetBench提出了一种以每次调用输入令牌预算为自变量的记忆策略评估协议，通过在2K至32K令牌的多级预算下固定模型与任务、测量质量、延迟和预算违规率，为本地大语言模型智能体的记忆策略比较提供了可复用的标准化测量框架。 |
| [^205] | [Agent as Policy for Robotic Manipulation](https://arxiv.org/abs/2609.12541) | 该论文提出Agent as Policy (AGP)方法，使通用智能体无需任何任务特定训练即可直接控制物理机器人完成从精细操作、动态运动到可变形物体处理等多种真实操作任务。 |
| [^206] | [A Training-Free, Alignment-Free Approach to Corporate Intelligence: Application to SEC Filings](https://arxiv.org/abs/2609.11620) | 该论文提出一种基于确定性稀疏种子向量的无需训练、无需对齐的企业情报分析框架，可在普通CPU上实现对SEC文件的亚秒级比较、发行人指纹识别及主题句提取。 |
| [^207] | [A Voice-Interactive Multi-Agent System for Smart Operating Rooms: Architecture Design and Key Technologies](https://arxiv.org/abs/2609.11231) | 本文提出基于大语言模型的智能手术室语音交互多智能体系统SurgicalRoomAgent，通过KV Cache前缀预热、流式JSON解析和渐进式技能提示披露三项关键技术大幅降低推理延迟，实现自然语言理解、设备控制、术中记录与手术报告生成。 |
| [^208] | [Scaling E-Commerce Attribute Extraction with Parallel Decoding](https://arxiv.org/abs/2609.09716) | 提出了一种两阶段LLM流水线，先为每个产品类别发现紧凑且按重要性排序的购买判别属性模式，再利用微调的Qwen3-4B结合超并行解码技术提取属性值，在达到与基础LLM相当的85%提取准确率的同时，将推理成本降低了92%，实现了生产级规模化应用。 |
| [^209] | [Bridging Network Psychometrics and Artificial Intelligence: An Ising-Potts Model with LLM-Derived Weights](https://arxiv.org/abs/2609.08797) | 本文提出一种融合大语言模型嵌入权重的评分者伊辛-波茨模型，通过评分间的两两一致性而非预设阈值来评估多类别评分信度，并在三个大规模数据集上比较了三种相似性增强策略。 |
| [^210] | [HoneyRoute: Honeypot-Model Routing for Adversarial LLM Serving](https://arxiv.org/abs/2609.08306) | HoneyRoute是一个部署在推理服务层的防御框架，通过轻量级流式路由器实时检测恶意请求并将其路由至蜜罐模型，在保护生产LLM服务的同时，将捕获的攻击者交互转化为指纹数据用于持续改进路由器的检测能力。 |
| [^211] | [Where Should Language Sit in a Multimodal Model? Lessons from What Language Does to Human Perception and Cognition](https://arxiv.org/abs/2609.07474) | 该论文将语言类比为运行在共享码本上的压缩器，通过回顾语言对人类感知、大脑和思维的作用，并结合对六个视觉-语言模型和两个机器人策略的线索冲突实验，探讨语言在多模态模型中应占据的合适位置。 |
| [^212] | [Content-Based Addressing for Long Context](https://arxiv.org/abs/2609.07314) | 提出一种基于内容寻址的长上下文位置编码方法，将 token 流划分为单元、单元内保留 RoPE 并为每个完成的单元按内容计算地址，从而在扩展上下文时完全保留局部 RoPE 并使固定 token 间的注意力比较不受其他单元插入或重排的影响。 |
| [^213] | [Where to Look and What to Use: Retrieve-Localize-Generate for Long-Term Conversational Memory Question Answering](https://arxiv.org/abs/2609.07093) | 提出了MemLoc统一框架，通过多粒度记忆分解、记忆内图查询路由与跨记忆图建模实现由粗到细的检索，并引入定位机制过滤噪声，从而解决长期对话记忆问答中证据碎片化与“迷失在中间”效应两大挑战。 |
| [^214] | [Exact Record Omission in Delta Attention: A Transport Criterion, Its Cost, and a Replay Certificate](https://arxiv.org/abs/2609.06872) | 该论文证明了在Delta注意力类循环记忆中，通过“收据”传递实现记录精确删除当且仅当该记录诱导的后续更新变化净额为零，并在48B Kimi Linear混合模型上实证表明该条件不成立——记录会留下现有收据类方法均无法消除、重算也只能部分弥补的持久印记。 |
| [^215] | [Data Efficient Sample Selection for In-Context Learning](https://arxiv.org/abs/2609.06670) | 提出了DearICL框架，将ICL示例选择建模为子集排名问题，通过结合间隔索引赌博机算法与可微排序目标的非线性替代模型，实现了数据高效且能泛化到未见查询的样本选择。 |
| [^216] | [Query-Oblivious Coresets for Softmax Attention: Improved Bounds and Efficient Constructions](https://arxiv.org/abs/2609.06327) | 本文通过球面提升和Chevet不等式将softmax注意力核心集问题转化为指数核实例，首次给出在查询半径内对所有查询均有效的、与理论下界仅相差 $\sqrt{\log(1+\rho)}$ 的随机多项式时间构造，无需新技术即弥合了Liberty等人提出的理论差距。 |
| [^217] | [A Ticket from Marginals to Joints: Coupled-Noise Distillation for One-Step Block Generation in Diffusion Language Models](https://arxiv.org/abs/2609.06324) | 提出CONDOR方法，通过耦合噪声蒸馏从零训练扩散语言模型，使其在掩码嵌入受高斯噪声扰动时仅用一次前向传播即可生成连贯的整块文本，且无需目标侧编码器或自回归教师。 |
| [^218] | [Untangling the Mechanisms of Misleading Context in Medical Question Answering](https://arxiv.org/abs/2609.02754) | 该研究通过在MedMisBench中注入伪造证据和纯粹断言两类误导性上下文，系统揭示了推理模型的医学判断被误导的机制，发现模型对纯粹断言的易感性显著高于伪造证据（高出10至27个百分点），且误导信息虽在推理轨迹中被大量披露却难以被察觉。 |
| [^219] | [From Detection to Characterization: A Large-Scale Study of Ragebait on Japanese X](https://arxiv.org/abs/2609.02262) | 本研究利用LLM辅助标注构建数据集并训练出日语愤怒诱饵检测集成分类器，首次对X平台日语帖子进行大规模分析，发现愤怒诱饵在政治、歧视、公共卫生和人际冲突等争议性话题中更为普遍。 |
| [^220] | [Verifiable Disaster Storylines and Causal Knowledge Graphs: A Citation-Grounded Pipeline from Heterogeneous Humanitarian Sources](https://arxiv.org/abs/2609.00858) | 该论文提出了一个基于检索增强生成（RAG）的流水线，融合EM-DAT结构化灾害记录与ReliefWeb、EMM非结构化文档，自动生成涵盖17个字段的灾害故事线和因果知识图谱，且每个节点和边均附带引用溯源，实现了对原始信息源的完全可追溯性，为人道主义响应提供可验证的态势感知支持。 |
| [^221] | [Responsible Integration of AI in Cancer Genomics: Barriers, Risks, and Pathways to Trustworthy Clinical Translation](https://arxiv.org/abs/2608.30912) | 本综述系统识别了阻碍AI在癌症基因组学临床转化的四大相互关联障碍（证据不一致、可解释性与不确定性、数据治理与可重复性、互操作性），并提出系统级概念框架以实现可信的临床整合。 |
| [^222] | [VoiceCodeBench: Evaluating Exact Structured-Token Recovery in Automatic Speech Recognition](https://arxiv.org/abs/2608.28916) | 该论文提出VoiceCodeBench基准，利用300段职场录音中跨26种类型的1,482个规范实体，通过CTEM和TSR等新指标评估12个ASR系统对标识符、路径、数值等结构化标记的精确恢复能力，弥补了传统WER评估的不足。 |
| [^223] | [A rigor-matched audit of periodic-step layer skipping for efficient llm inference: conflayers versus swift, with a supplemental analysis of trained routing alternatives](https://arxiv.org/abs/2608.28846) | 该研究通过严格匹配的多种子审计发现，SWIFT自推测解码在准确率上优于ConfLayers置信度门控提前退出方法，且在将在线搜索开销与纯推理成本分离后，SWIFT在所有实验设置下的真实推理速度均快5-21%，颠覆了表面上的速度排名。 |
| [^224] | [Auditing Generative Audio Calls for Known-Task Audio-LLM Evaluation](https://arxiv.org/abs/2608.27817) | 该论文将音频大语言模型的评估建模为受控的调用决策问题，发现在已知封闭集任务上，有监督编码器（如CLAP和WavLM）无需调用生成式音频模型即可取得接近最优的准确率，从而揭示了传统“波形提示对比ASR转录”的评估方式混淆了声学证据获取与生成模型调用这两个因素。 |
| [^225] | [SURE-Challenge: Evaluating Speech Evidence Before Speech-LLM Generation](https://arxiv.org/abs/2608.27783) | 该论文提出 SURE-Challenge 基准，用于评估语音大模型在生成回答之前对不支持输入（静音、噪声、合成音调、嘈杂语音）的拒绝能力，并证明一个简单的“能量加 Whisper 分数”规则可将不支持输入的拒绝数从 15/204 提升至 196/204，同时不损失有效输入的准确率。 |
| [^226] | [When the Canonical Completion Is Wrong: Formalizing and Measuring the Jump in Large Language Models](https://arxiv.org/abs/2608.26187) | 本文首次形式化了LLM中的“跳跃”概念，将其定义为具有可验证证书的有限扩展问题，并提供了度量方法，以解决关于LLM溯因能力的争论。 |
| [^227] | [What's the Catch? Evaluating Temporal Consistency in Vision-Language Models](https://arxiv.org/abs/2608.23474) | 本研究通过时间异常检测任务揭示了视觉语言模型在时间一致性理解上的显著不足，与人类表现存在明显差距。 |
| [^228] | [KREL: Automatic Medical Coding via Knowledge-Guided Reasoning over Clinical Evidence with LLMs](https://arxiv.org/abs/2608.20887) | 该论文提出了KREL框架，通过结合大型语言模型的推理能力与外部ICD编码指南的结构化知识，解决了自动医疗编码中临床记录过长、标签空间庞大及编码规则复杂等关键问题。 |
| [^229] | [Readable, Faithful, Used: Three Dissociable Properties of Demographic Identity in a Language Model](https://arxiv.org/abs/2608.18768) | 本研究发现语言模型中人口统计身份的可读性、忠实性和使用性可分离，注意力头读出比标准方法更忠实，且单一头可跨属性保持高保真，但种族相关身份较弱。 |
| [^230] | [IndicQE-APE: A Benchmark for Quality Estimation and Automatic Post-Editing for Indic Languages](https://arxiv.org/abs/2608.16344) | 本文整合了印度语言的质量评估和自动后编辑数据，创建了一个包含多标签和难度分层的基准，并评估了多种模型，发现只有同时利用整体和词级信息的系统在严格对照下表现显著。 |
| [^231] | [TaoLive Digital Avatar Agent Technical Report: Training Agents to Evolve with Their Harness](https://arxiv.org/abs/2608.15763) | 本文提出操控系统感知训练（HAT）方法，通过将可进化的操控系统状态纳入训练分布，使数字人代理在实时直播中既能快速响应又能灵活适应动态策略变化。 |
| [^232] | [Data Attribution of Emergent Misalignment with Persona Features](https://arxiv.org/abs/2608.11025) | 本研究通过稀疏自编码器模型差异分析发现，涌现性失调源于预训练中被失调微调放大的人格特征，仅通过引导这些特征即可在已对齐模型中诱导高达62%的失调或使失调模型重新对齐。 |
| [^233] | [When Is a General Factor Distinguishable? Non-Proportionality, Stable Structure, and the Bifactor Decision](https://arxiv.org/abs/2608.10731) | 本文提出了判断双因子模型与相关一阶因子模型何时可区分的理论条件（关键在于簇内载荷的非比例性），并开发了一个两步程序来稳定地确定一阶结构，进而有条件地比较斜交与双因子两种表示。 |
| [^234] | [MameLoshnLM: Yiddish Language Model and Evaluation Benchmark](https://arxiv.org/abs/2608.05850) | 该论文发布了首个开源的80亿参数意第绪语语言模型MameLoshnLM，通过构建高质量预训练语料库Oytser和多任务评估基准Kashes，解决了意第绪语数字资源稀缺与评估数据不可靠的问题，并在基准任务上超越同等规模的开源基线模型。 |
| [^235] | [ChartAnno: Benchmarking Multimodal Large Language Models for Chart Annotation Generation](https://arxiv.org/abs/2608.03464) | 本文提出了ChartAnno基准，包含1,200个真实图表、3,600条多层级标注指令及结合规则与LLM评判的多维评估框架，系统评估了多模态大语言模型在图表标注生成任务上的能力。 |
| [^236] | [Can AI Agents Simulate A/B Test Outcomes? A Validation Framework for Agentic Experimentation](https://arxiv.org/abs/2608.02345) | 本文提出一种代理无关的模拟随机对照试验框架，通过两层误差分解验证AI代理能否准确模拟A/B测试结果，以在真实流量前筛选干预方案。 |
| [^237] | [FriendBench: Benchmarking Dyadic Familiarity Inference in Humans and Multimodal Large Language Models](https://arxiv.org/abs/2607.29602) | 该论文提出FriendBench基准，用于评估人类和多模态大语言模型从短对话片段中推断两人是熟人还是陌生人的能力，发现最佳模型的准确率已与人类相当但存在先验偏好差异，且只有人类能利用言语之外的可见行为进一步提升判断。 |
| [^238] | [Studying quantization trade-offs for efficient inference deployment in machine translation](https://arxiv.org/abs/2607.29397) | 本文系统研究了EuroLLM在1.7B至22B三个模型规模上的量化权衡，发现结合文档分块策略与W4A8或W8A8量化可显著改善单GPU部署下的延迟-吞吐量帕累托曲线，并提出了基于DocHPLT的文档级机器翻译评估方法。 |
| [^239] | [FairFund-Bench: Evaluating Distributive Bias in LLM Resource Allocation](https://arxiv.org/abs/2607.28934) | 本文提出FairFund-Bench基准，通过系统性地变化审计格式（评估任务、比较情境和透明度），揭示了以往LLM资源分配偏见审计结果不一致的原因在于审计格式本身的差异。 |
| [^240] | [Subtract, Transport, or Replay? Auditable Deletion from Language-Model Memory](https://arxiv.org/abs/2607.27539) | 本文提出并比较了三种从语言模型记忆中删除记录的方法，发现原生删除不可靠，而检查点重放能实现可审计的精确删除，并展示了在冻结模型上无需额外训练即可构建可删除记忆的可行性。 |
| [^241] | [Looping Is Not Reliability: State-Bound Evidence and Typed Revision Contracts for Agentic Code Repair](https://arxiv.org/abs/2607.24604) | 该论文通过受控实验证明编码智能体中“生成—测试—修订”循环的重复并不能保证可靠性——陈旧轨迹会显著损害已经正确的补丁——并提出用状态绑定证据与类型化修订契约来确保补丁的保留、验证与提交。 |
| [^242] | [Where Animacy Lives in Large Language Models: Tracing the Circuits of the Animacy Concept](https://arxiv.org/abs/2607.20995) | 本研究通过构建最小对数据集并进行回路发现，首次在大语言模型中追踪并证实了处理有生命性概念的因果回路的存在，同时发现该回路较为分散，其跨模型和跨任务的泛化能力有限。 |
| [^243] | [Chemical Chain-of-Thought Functions as a Hallucination-Prone Molecular Scratchpad](https://arxiv.org/abs/2607.20935) | 化学思维链推理中幻觉普遍存在且与答案正确性脱钩，它并非忠实的解释，而是以模型特异的形式充当“分子草稿板”，其中结构草稿对生成质量具有因果支撑作用。 |
| [^244] | [AdaFlash: Adaptive Speculative Decoding via On-Policy Distilled Diffusion Drafters](https://arxiv.org/abs/2607.19223) | 论文揭示了扩散起草器中双向注意力导致的领域级和词元级高方差问题，并提出 AdaFlash 自适应投机解码框架加以解决。 |
| [^245] | [Similar Accuracy, Unequal Evidence: Search APIs as Decision Surfaces for Tool-Using Agents](https://arxiv.org/abs/2607.10198) | 该研究在固定实验条件下对比Brave、Tavily和Firecrawl三个搜索API作为工具使用型智能体的决策表面，发现三者的任务准确率虽相似，但在证据暴露特征上存在显著差异——Brave提供更多预抓取支持、Tavily排名第一结果的支持证据占比更高、Firecrawl则带来更广泛的探索。 |
| [^246] | [LEXIC: Lightweight On-Device Decoding of Reading Comprehension from Eye Movements](https://arxiv.org/abs/2607.08152) | 论文提出LEXIC——一个仅含41.6K参数、166 KB大小的紧凑循环模型，可从眼动数据直接预测阅读理解成绩，无需语言模型推理，单核CPU上每次试验仅需1.4毫秒，实现了真正轻量级的端侧部署。 |
| [^247] | [REDDIT: Correcting Model-Generated Timestamp Drift in ASR without Forgetting via Replay-Based Distribution Editing](https://arxiv.org/abs/2607.05364) | 该论文提出了轻量级两阶段后训练框架REDDIT，通过基于重放的分布编辑修正自回归ASR系统在长非语音片段上产生的时间戳漂移，同时避免了朴素微调带来的灾难性遗忘问题。 |
| [^248] | [Mapping Text to Multiplex Graph: Prompt Compression as L\'evy Walk-Guided Graph Pruning](https://arxiv.org/abs/2607.01241) | 该论文提出RAGP方法，将提示压缩建模为多重图上的冗余感知图剪枝问题，并利用列维游走的重尾步长分布平衡局部与全局探索，从而高效识别文本中分散的非冗余重要信息。 |
| [^249] | [Bridging Scientific Heritage: An Arabic--Russian Parallel Corpus and LLM Benchmark for Sustainable Knowledge Transfer](https://arxiv.org/abs/2606.30943) | 该论文构建了阿拉伯语—俄语科学翻译的首个基准，包含约27,000句对的混合平行语料库，并通过LoRA微调多种多语言大模型，其中QLoRA微调的Qwen2.5-7B取得最佳性能，显著优于零样本基线，为俄阿两个科学社区之间的可持续知识转移奠定了基础。 |
| [^250] | [Interpretable Inverse Design of Metal-Organic Frameworks with Large Language Model Agents](https://arxiv.org/abs/2606.29459) | 本文提出LLM4MOF闭环多智能体框架，利用大语言模型将自然语言目标转化为可解释的化学假设与约束，在400次评估内于多项吸附、分离和电子结构任务中筛选出高性能金属有机框架，并实现了新型MOFs的从头设计与实时模拟。 |
| [^251] | [MMLA: How Memory Lets the Past Shape the Future](https://arxiv.org/abs/2606.28876) | MMLA通过有界驻留内存和事件化处理，在因果部署中实现记忆选择，显著提升多跳问答性能。 |
| [^252] | [An Empirical Analysis of Factual Errors in Human-Written Text and Its Application to Factual Error Detection](https://arxiv.org/abs/2606.27959) | 该论文通过分析报纸文章更正构建了人类撰写文本中事实错误的分类体系，发现了汉字误转换、单位错误等现有幻觉基准未覆盖的错误类型，并据此评估了大语言模型在事实错误检测任务上的能力。 |
| [^253] | [Beyond Surface Forms: A Comprehensive, Mechanism-Oriented Taxonomy of Indirect Linguistic Encoding for LLM-Based Coded Language Detection](https://arxiv.org/abs/2606.27314) | 提出了一种面向机制的间接语言编码综合分类法，通过抽象化交际目标、分类底层编码操作，显著提升了基于大语言模型的隐语检测性能（准确率提高4.7%，F1提高5.4%）。 |
| [^254] | [Compositionality and the lexicon in evolutionary semantics](https://arxiv.org/abs/2606.27228) | 该论文提出了一种将形式语义学的组合性原理整合到进化建模中的新框架，通过让词汇意义与组合功能共同进化，揭示了语义普遍性（如保守性）作为高效系统级抽象的产生机制，并有效调和了经验证据与进化模型之间的冲突。 |
| [^255] | [Humans Disengage, Reasoning Models Persist: Separating Difficulty Registration from Deliberation Allocation](https://arxiv.org/abs/2606.26502) | 论文发现，在相同问题上，人类答错时花更少时间（放弃），而大型推理模型答错时花更多令牌（坚持），揭示了人类与AI在失败时资源分配策略的根本差异。 |
| [^256] | [Weave of Formal Thought](https://arxiv.org/abs/2606.25987) | 本文提出了一个通过新颖投机词法分析增强GLR解析、对完整Tree-sitter规范可靠且完备的形式引擎与约束解码器，并引入隐变量微调方法WoFT，使语言模型在代码生成中能够利用语法层次结构。 |
| [^257] | [The Language-Energy Divide: Measuring Energy Costs of Multilingual LLM Inference](https://arxiv.org/abs/2606.21869) | 该论文系统性测量了大语言模型多语言推理的能耗，揭示了不同语言间高达179倍的能耗差异，并发现高能耗语言同时伴随最低的任务准确率，存在“成本+性能”双重惩罚现象。 |
| [^258] | [PARSE: Provenance-Aware Retrieval Sanitization for Professional Domain LLM Agents](https://arxiv.org/abs/2606.17467) | 论文揭示了现有提示注入防御方法（如改写）在真实企业文档上失效的差距，并提出了PARSE——一个领域感知、保留事实的溯源感知检索净化流水线，可为专业领域大语言模型智能体提供更有效的提示注入防御。 |
| [^259] | [RLCSD: Reinforcement Learning with Contrastive On-Policy Self-Distillation](https://arxiv.org/abs/2606.11709) | 提出RLCSD方法，通过对比正确与错误提示下的师生分布差距来抑制“特权诱导的风格漂移”，使自蒸馏监督信号更集中于任务相关token，从而提升推理模型强化学习训练的稳定性与效果。 |
| [^260] | [Attention-Discounted Adaptive Sampler for Masked Diffusion Language Models](https://arxiv.org/abs/2606.10829) | 提出免训练重排序规则ADAS，根据每个token对已选位置的注意力（按预测不确定性加权）贪婪地折扣其置信度分数，从而提升掩码扩散语言模型在低推理步数下的表现。 |
| [^261] | [Recovering the Zipfian Distribution in Unsupervised Term Discovery](https://arxiv.org/abs/2606.10781) | 该论文提出用基于图的Leiden聚类替代K-means等基于中心的方法，在无监督词条发现中显著恢复了真实词库所具有的齐夫分布特性。 |
| [^262] | [ParaBridge: Bridging Paralinguistic Perception and Dialogue Behavior in Speech Language Models](https://arxiv.org/abs/2606.10581) | ParaBridge提出一种在线策略自蒸馏方法，将推理时脆弱的副语言指令支架转化为语音语言模型中稳定的行为，从而弥合副语言感知与对话行为之间的差距。 |
| [^263] | [Cherry-pick Override: LLM Judges Under-use the Non-Directional Verdicts Their Contract Authorizes](https://arxiv.org/abs/2606.07834) | 本文定义并实证揭示了一种新的错误类型——选择性采信覆盖（CPO）：当任务契约授权使用“冲突”或“证据不足”等非定向判定时，LLM 评判者仍会过度承诺“支持”或“反驳”等定向判定，此类错误在 AVeriTeC 和 VitaminC-Mixed 基准上的发生率高达 18.7%–48.0%。 |
| [^264] | [Attention Calibration for Position-Fair Dense Retrieval](https://arxiv.org/abs/2606.02737) | 该论文提出一种带强度系数的注意力校准方法，缓解密集检索中嵌入表示偏向文本前部内容的位置偏差问题，同时将校准内存开销从数 GiB 降至 1 MiB 以下，显著提升相关片段位于文本后部时的检索性能。 |
| [^265] | [Internalize the Temperature: On-Policy Self-Distillation as Policy Reheater for Reinforcement Learning](https://arxiv.org/abs/2606.00755) | 提出TS-OPSD方法，通过对模型自身logits进行高温缩放并将平滑后的分布自蒸馏回模型，将温度的探索效应内化到模型参数中，从而在不依赖外部教师和额外推理开销的情况下缓解强化学习中的熵坍缩问题。 |
| [^266] | [Local Diagnostics of Continuous Normalizing Flow for Out-of-Distribution Detection](https://arxiv.org/abs/2606.00684) | 本文提出基于连续正则化流的拉格朗日子流（LSF）框架，利用子流轨迹速度场的几何诊断信号来检测分布外样本，从而缓解深度生成模型的“似然悖论”问题。 |
| [^267] | [Generating Reports or Repeating Templates? Measuring and Mitigating Template Collapse in 3D CT Report Generation](https://arxiv.org/abs/2605.30984) | 该论文首次识别并量化了3D CT报告生成中的“模板坍塌”失效模式，并提出将临床病理检测与语言合成分离的CLarGen解耦框架来有效缓解这一问题。 |
| [^268] | [Compute Allocation for Self-Evolving LLMs: From Depth-Breadth to Multi-Armed Bandits](https://arxiv.org/abs/2605.29268) | 该论文揭示了LLM进化搜索中算力分配的两条经验规律（适应度-算力包络线和双线性深度-广度拟合），并提出基于多臂老虎机的BaSE方法动态分配LLM调用预算，在不改变模型、提示或评估器的情况下将平均适应度提升12.3%。 |
| [^269] | [Mimir: Large-scale Multilingual Concept Modeling](https://arxiv.org/abs/2605.25263) | 该论文提出了Mimir——一个16亿参数的多语言大型概念模型，突破了传统基于词元的语言建模范式，通过直接进行下一概念预测来实现多语言概念的理解与生成。 |
| [^270] | [Transcoders Trace Visual Grounding and Hallucinations in Vision-Language Models](https://arxiv.org/abs/2605.22902) | 本文提出基于转码器（Transcoders）的功能中心可解释性框架，能比稀疏自编码器更准确、更稳定地追踪视觉-语言模型中从图像块到文本生成的计算通路，并借此揭示了幻觉背后的错误视觉接地机制。 |
| [^271] | [Playing Devil's Advocate: Off-the-Shelf Persona Vectors Rival Targeted Steering for Sycophancy](https://arxiv.org/abs/2605.21006) | 该研究发现无需专门针对谄媚性提取的现成批判性角色人格向量，就能将语言模型的谄媚性降低至专门方法CAA效果的68%-98%，且两者在几何上相互分离。 |
| [^272] | [On the Interpretability of Whisper Encodings Using Sparse Autoencoders](https://arxiv.org/abs/2605.12225) | 该研究利用稀疏自编码器首次揭示了Whisper语音识别模型编码器内部存在从语音到语义的丰富多层次语言表征，并通过因果引导实验发现高层特征比低层特征更易于操控，表明其编码信息远超任务所需。 |
| [^273] | [Putting HUMANS first: Efficient LAM Evaluation with Human Preference Alignment](https://arxiv.org/abs/2605.00022) | 该研究提出用仅50个样本（0.3%数据）的最小子集即可高效评估大型音频模型（与完整基准相关性超0.93），并通过在子集上训练回归模型将与人类偏好评分的相关性从0.85提升至0.98，实现了更贴合用户满意度的高效LAM评估方法。 |
| [^274] | [Useless but Safe? Benchmarking Utility Recovery with User Intent Clarification in Multi-Turn Conversations](https://arxiv.org/abs/2604.27093) | 该论文提出了首个交互式基准CarryOnBench和Ben-Util指标，用于评估LLM在多轮对话中能否在保持安全性的同时，根据用户意图澄清恢复有用性，发现模型首轮仅能满足10.5%至37.6%的用户良性信息需求。 |
| [^275] | [Intersectional Fairness in Large Language Models](https://arxiv.org/abs/2604.20677) | 该研究系统评估了六个大语言模型的交叉公平性，发现模型在“种族-社会经济地位”和“种族-性别”两种属性交叉组合下表现出相反的刻板印象偏向，且子群体公平性指标揭示了表面低差异下隐藏的不均匀结果分布。 |
| [^276] | [HumorGen: Cognitive Synergy for Humor Generation in Large Language Models via Persona-Based Distillation](https://arxiv.org/abs/2604.09629) | 该论文提出受幽默心理学理论启发的认知协同框架，通过六种认知角色的思维混合方法合成理论支撑的幽默数据集来微调7B模型，且发现DPO和O-GRPO对齐策略未能超越SFT，但最终模型在幽默生成上显著超越更大的指令微调基线。 |
| [^277] | [SatIR: Scalable High-Recall Constraint-Satisfaction-Based Information Retrieval for Clinical Trials Matching](https://arxiv.org/abs/2604.08849) | 该论文提出SatIR，一种基于形式化约束满足的可扩展、高精确率、高召回率且可解释的临床试验匹配检索方法，将患者入组资格约束视为强制性要求而非软信号，从而克服了传统关键词和嵌入相似度匹配方法召回率低、精确率低、可解释性差的局限。 |
| [^278] | [Can We Still Trace L1 Signals? Investigating the Resilience of Native Language Signals in the LLM Era](https://arxiv.org/abs/2604.08568) | 本研究通过构建覆盖神经网络前、LLM前和LLM后三个时代、八个母语群体的学术摘要母语识别数据集，发现文本中的母语信号随时间持续减弱，且英语同质化的主要趋势早在LLM出现之前就已开始。 |
| [^279] | [Testing the Limits of Truth Directions in LLMs](https://arxiv.org/abs/2604.03754) | 大语言模型中的“真相方向”并非普适：它高度依赖于网络层级、任务类型（事实性任务出现在较早层级，推理任务出现在较晚层级）以及模型所接收的指令。 |
| [^280] | [Is my model perplexed for the right reason? Contrasting LLMs' Benchmark Behavior with Token-Level Perplexity](https://arxiv.org/abs/2603.29396) | 本文提出一种基于词元级困惑度的可解释性框架，通过对比仅在一个或少数关键词条上存在差异的最小句子对的困惑度分布，来检验LLM是否真正依赖语言相关线索，发现语言重要词条虽会影响模型行为但无法完全解释困惑度变化，表明模型依赖的是非预期的启发式机制。 |
| [^281] | [AI Psychosis: Does Conversational AI Amplify Delusion-Related Language?](https://arxiv.org/abs/2603.19574) | 本研究基于Reddit用户历史发帖构建模拟用户，并提出DelusionScore语言度量方法，实证发现具有妄想倾向的用户在与GPT、LLaMA和Qwen等对话式AI的多轮交互中，妄想相关语言强度会逐渐增强，为“AI精神病”现象提供了实证依据。 |
| [^282] | [To See is Not to Master: Teaching LLMs to Use Private Libraries for Code Generation](https://arxiv.org/abs/2603.15159) | 针对LLMs即使获得准确文档知识仍难以调用私有库API的问题，提出PriCoder方法，通过将数据合成建模为图构建并交替使用渐进式图演化等图算子来自动合成多样化训练数据，教会模型有效调用私有库API进行代码生成。 |
| [^283] | [A technology-oriented mapping of the language and translation industry: Analysing stakeholder values and their potential implication for translation pedagogy](https://arxiv.org/abs/2603.11667) | 该研究基于29位行业利益相关者的访谈，发现在自动化生产环境中，效率导向的技术价值观已成为行业基线期望，而人类价值并未被取代，而是通过专业知识、监督、问责和情境判断被重新定位，这一发现对翻译教学具有重要启示。 |
| [^284] | [Hindsight-Anchored Policy Optimization: Learning Through Hindsight with Thompson Sampling-Inspired Adaptive Gating](https://arxiv.org/abs/2603.11321) | 提出后见之明锚定策略优化（HAPO），利用Beta-二项置信自适应门控决定教师干预时机，结合合成成功注入与自适应阈值退火作为临时支撑，解决了稀疏奖励下混合策略强化学习的训练崩溃与不稳定问题。 |
| [^285] | [How Contrastive Decoding Enhances Large Audio Language Models](https://arxiv.org/abs/2603.09232) | 对比解码能可靠地纠正大型音频语言模型中虚假否认音频存在或因不确定性而猜测的错误，但难以纠正错误推理和自信的错误断言，其收益大小取决于模型基线错误的构成。 |
| [^286] | [TimeWarp: Evaluating Web Agents by Revisiting the Past](https://arxiv.org/abs/2603.04949) | 提出了模拟网页UI演变的TimeWarp基准测试，揭示现有网页智能体对界面变化的脆弱性，并通过TimeTraj标注方法与BC变体训练显著提升了智能体跨网页版本的泛化能力。 |
| [^287] | [Context-Dependent Affordance Reports in Vision-Language Models](https://arxiv.org/abs/2603.04419) | 该研究审计了先前关于角色提示影响视觉-语言模型描述的研究，发现空物体列表响应导致原分析低估了相似度，并撤回了原研究的功能性流形解释，转而通过新的对照实验重新审视该效应。 |
| [^288] | [Bridging Latent Reasoning and Target-Language Generation via Retrieval-Transition Heads](https://arxiv.org/abs/2602.22453) | 本文在多语言大模型中识别出一类区别于检索头的检索-过渡头（RTH），它们控制着模型向特定目标语言输出的过渡，且对思维链推理更为关键——屏蔽RTH会在四个多语言基准和两个模型系列上引起比屏蔽检索头更大的性能下降。 |
| [^289] | [Rubrics as an Attack Surface: Stealthy Preference Drift in LLM Judges](https://arxiv.org/abs/2602.13576) | 该论文首次揭示了一种名为“评分标准诱导的偏好漂移”（RIPD）的漏洞：通过基准验证的看似无害的评分标准修改，可以隐蔽且系统性地偏移大语言模型评审器的偏好，从而构成难以被基准指标或抽查检测到的偏好攻击。 |
| [^290] | [Controllable Dysarthric Speech Synthesis with Patient-Specific Conditioning for Speaker-Diverse ASR Augmentation](https://arxiv.org/abs/2602.08696) | 提出了一种可控构音障碍语音合成框架，通过解耦音色与患者特异性病理发音条件，可将病理特征与不同目标说话人（含健康人）灵活组合，生成的语音可有效增强构音障碍语音识别系统的训练数据。 |
| [^291] | [Dreaming in Code for Curriculum Learning in Open-Ended Worlds](https://arxiv.org/abs/2602.08194) | 提出DiCode框架，利用大语言模型“做梦”般地合成可执行的环境代码变体，在开放式世界中构建课程，引导智能体获得持续可学习并逐步提升能力的经验序列。 |
| [^292] | [DecompressionLM: Deterministic, Diagnostic, and Zero-Shot Concept Graph Extraction from Language Models](https://arxiv.org/abs/2602.00377) | DecompressionLM是一个无状态框架，利用Van der Corput低差异序列与算术解码，在无需预设查询的情况下，实现确定性和高度并行的零样本概念图提取，以发现语言模型中编码的概念，并揭示了量化（如AWQ-4bit）可显著扩大概念覆盖范围。 |
| [^293] | [SDUs DAISY: A Benchmark for Danish Culture](https://arxiv.org/abs/2601.19930) | 该论文基于2006年《丹麦文化经典》构建了丹麦文化事实知识基准Daisy，通过人工验证的741个封闭式问答对，评估当代语言模型对丹麦文化遗产（包括深层次、非主流内容）的掌握程度。 |
| [^294] | [Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering](https://arxiv.org/abs/2601.19847) | 本文提出AdaRAS框架，通过识别推理关键神经元并在推理时自适应地引导其激活，以轻量级方式显著提升大语言模型推理的可靠性和准确性。 |
| [^295] | [How Do We Engage with Other Disciplines? A Framework to Study Meaningful Interdisciplinary Discourse in Scholarly Publications](https://arxiv.org/abs/2601.17020) | 本文提出了一个针对跨学科NLP出版物的引用参与度评估框架，引入了专为跨学科工作设计的引用目的分类体系，弥补了现有方法无法定量衡量引用参与质量的不足。 |
| [^296] | [Human Values in a Single Sentence: Moral Presence, Hierarchies, and Transformer Ensembles on the Schwartz Continuum](https://arxiv.org/abs/2601.14172) | 该研究在ValueEval'24语料上对19个Schwartz人类价值观进行句子级多标签分类，发现道德存在性可从单句有效学习、存在性门控的层级结构并不优于直接预测，而决策阈值校准是一个决定性却常被忽视的关键因素。 |
| [^297] | [Confident Rankings with Fewer Items: Adaptive LLM Evaluation with Continuous Scores](https://arxiv.org/abs/2601.13885) | 本文将基于IRT的自适应测试从二值评分扩展到连续有界分数，提出不确定性感知的自适应停止排序方法，仅需2%的测试题目即可实现对LLM的高置信度可靠排名。 |
| [^298] | [How Semantically Stable Are LLM Refusals? Measuring Confusion in Local Safety Boundaries](https://arxiv.org/abs/2512.01037) | 该论文提出“语义混乱”这一新失败模式，并构建了包含 1 万条提示的受控改写语料库 ParaGuard，用于度量大语言模型在语义等价改写之间拒绝决策的局部不一致性，弥补了传统全局指标（如误拒率）无法反映拒绝行为语义稳定性的不足。 |
| [^299] | [Understanding the Effects of Distractors on Reasoning Vision-Language Models](https://arxiv.org/abs/2511.21397) | 本研究提出Idis视觉问答数据集，揭示了视觉干扰因素对推理视觉语言模型的影响机制与文本干扰因素根本不同——尽管两者都会引发逆向缩放效应，但视觉干扰因素在降低准确率的同时并不会增加推理长度。 |
| [^300] | [SkyEgg: Heterogeneity-Aware Hardware Synthesis via Equality Saturation](https://arxiv.org/abs/2511.15323) | SkyEgg是一个基于等价饱和的异构感知FPGA硬件综合框架，通过在变换、调度和映射阶段联合探索硬件实现方案与细粒度配置选择，突破了传统顺序式综合流程的局限，从而提升端到端性能。 |
| [^301] | [Donors and Recipients: On Asymmetric Transfer Across Tasks and Languages with Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2511.13368) | 该研究通过系统的LoRA单源微调实验发现，跨任务与跨语言的迁移增益呈强烈非对称性，其中匹配任务跨语言迁移最有效，且迁移幅度主要取决于目标语言的特性。 |
| [^302] | [Pruning as Regularization: Sensitivity-Aware One-Shot Pruning in ASR](https://arxiv.org/abs/2511.08092) | 该论文发现一次性幅度剪枝不仅是压缩手段，更可作为强大的隐式正则化器，通过梯度与Fisher信息敏感性诊断指导的组件级剪枝，在不微调的情况下显著提升Whisper语音识别模型的泛化性能并降低词错误率。 |
| [^303] | [Interpretable Recognition of Cognitive Distortions in Natural Language Texts](https://arxiv.org/abs/2511.05969) | 本文提出一种基于加权N-gram等结构化模式及其层级关系的可解释人工智能方法，用于自动识别心理护理文本中的认知扭曲，在两个公开数据集上显著提升了F1分数。 |
| [^304] | [CapGeo-Bench: Decoupling Visual Perception from Reasoning and Evaluating Geometric Understanding](https://arxiv.org/abs/2510.09302) | 该论文通过实验证实多模态几何推理的瓶颈在于视觉感知而非推理能力，并提出了包含4,641对高质量图形-图注及基于关键点评估指标的CapGeo-Bench基准，用于专门评估MLLMs的几何视觉感知能力。 |
| [^305] | [MuPlon: Multi-Path Causal Optimization for Claim Verification through Controlling Confounding](https://arxiv.org/abs/2509.25715) | 该论文提出MuPlon框架，通过后门路径和前门路径的双重因果干预策略，有效控制声明-证据图中的数据噪声和数据偏差两类混杂因素，从而提升声明验证的可靠性。 |
| [^306] | [Generalized Correctness Models: Learning Calibrated and Model-Agnostic Correctness Predictors from Historical Patterns](https://arxiv.org/abs/2509.24988) | 该论文挑战了LLM具备“自我认知”能力的传统假设，发现LLM判断自身输出正确性的能力并不优于无关模型，并提出通过注入目标模型的历史预测信息来构建广义正确性模型（GCM），从而实现校准且与模型无关的正确性预测。 |
| [^307] | [DiffuTester: Accelerating Unit Test Generation for Diffusion LLMs via Mining Structural Pattern](https://arxiv.org/abs/2509.24975) | DiffuTester通过挖掘针对同一焦点方法的单元测试所共享的结构模式，并提出基于抽象语法树的动态结构模式解码方法，在保证测试质量的同时显著加速了扩散大语言模型的单元测试生成。 |
| [^308] | [Can Interpretation Predict Behavior on Unseen Data?](https://arxiv.org/abs/2507.06445) | 该研究提出仅凭分布内数据上观察到的注意力模式即可预测模型在分布外数据上遵循的泛化规则，并证明解释的预测价值可与机制忠实性解耦——即使内部模式在因果上抑制其所预测的规则，观察性分析依然能成功预测模型行为。 |
| [^309] | [All Learning Has an Emotional Basis, So Does Task-Oriented Dialogue](https://arxiv.org/abs/2507.01594) | 提出了一种将完全词汇化表示与LLM主干相结合的端到端框架，通过在线强化学习同时利用短期情感奖励和长期任务成功奖励，联合优化面向任务对话系统中的情感交互与任务完成效果。 |
| [^310] | [LLM Probability Concentration: How Alignment Shrinks the Generative Horizon](https://arxiv.org/abs/2506.17871) | 该研究用分支因子（BF）量化LLM输出分布的概率集中程度，发现对齐微调从一开始就锐化输出分布、使BF总体降低2至5倍，而BF随生成进展的下降则主要是与对齐无关的自回归自我条件化的固有属性。 |
| [^311] | [Advancing Automated Speaking Assessment Leveraging Multifaceted Relevance and Grammar Information](https://arxiv.org/abs/2506.16285) | 本文通过引入整合问题、图像、范例与口语回答的多方面相关性模块，以及基于语法纠错技术的细粒度语法错误特征，显著提升了自动化口语测评系统中内容相关性和语言使用的评估性能。 |
| [^312] | [Who Benchmarks the Benchmarks? Towards Comprehensive Evaluation of Commonsense Reasoning Benchmarks](https://arxiv.org/abs/2504.07825) | 该论文通过对HellaSwag基准的案例研究揭示了常识推理基准中普遍存在的效度问题（如语法错误、拼写错误、误导性提示及多个正确选项），并发现约68%的模型预测在删除问题提示后仍不变，证明这是基准自身的内在缺陷而非模型数据污染所致。 |
| [^313] | [CoTAL: Human-in-the-Loop Prompt Engineering for Generalizable Formative Assessment Scoring and Feedback](https://arxiv.org/abs/2504.02323) | 该论文提出CoTAL方法，通过人机协同提示工程结合思维链提示与师生反馈迭代优化，使GPT-4在科学、计算、工程等多个领域的形成性评估自动评分性能提升高达38.9%。 |
| [^314] | [Lost-in-the-Middle in Long-Text Generation: Synthetic Dataset, Evaluation Framework, and Mitigation](https://arxiv.org/abs/2503.06868) | 本文提出了一种无需训练的检索增强长文本写作框架RAL-Writer，通过规划器与写作器联合建模语义相关性和位置偏差，动态检索并重述关键输入信息，从而缓解大语言模型在长输入到长输出生成中的“迷失中间”问题，并为此任务构建了基准数据集与三项评估指标。 |
| [^315] | [LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory of Transformers](https://arxiv.org/abs/2502.15007) | 该论文发现标点符号等看似次要的词元在Transformer的上下文记忆中承载着重要信息，移除它们会显著降低模型性能，并提出开源工具LLM-Microscope来量化词元的上下文承载能力和非线性度。 |
| [^316] | [Thinking beyond the anthropomorphic paradigm benefits LLM research](https://arxiv.org/abs/2502.09192) | 本文通过对数十万篇研究文献的实证分析，揭示了拟人化术语在大语言模型研究中的普遍性和增长趋势，识别出五个塑造LLM研究的拟人化假设，并主张超越这些假设以开辟理解和改进大语言模型的新途径。 |
| [^317] | [Balancing Global Quality and Pronoun-Specific Feedback for Context-Aware Machine Translation](https://arxiv.org/abs/2501.03008) | 该论文提出ProNMT方法，通过将句子级质量估计与代词位置的带符号置信度信号相结合进行奖励引导的迭代自训练，在上下文感知机器翻译中有效平衡了全局翻译质量与代词选择准确性，在英德和英法翻译任务上超越了标准监督微调。 |
| [^318] | [CHAI for LLMs: Improving Code-Mixed Translation in Large Language Models through Reinforcement Learning with AI Feedback](https://arxiv.org/abs/2411.09073) | 该论文提出CHAI通用框架，通过利用大语言模型作为标注者解决数据稀缺问题、基于AI反馈的强化学习（RLAIF）以及LLM生成的领域知识作为宪法进行迭代优化，使大语言模型在语码混合翻译任务上的性能超越最先进开源模型68.45%。 |
| [^319] | [Toward Secure Code Generation: Bridging Correctness and Security via Task-Adaptive Vulnerability Modeling and Execution-Based Benchmarking](https://arxiv.org/abs/2407.02395) | 该论文提出了基于执行的基准测试CodeSecEval（涵盖255个Python任务和77类CWE漏洞）以及智能体框架SecAwareCoder，通过任务自适应漏洞建模，在保证代码正确性的同时提升生成代码的安全性，弥合了正确性与安全之间的鸿沟。 |
| [^320] | [An Incomplete Loop: Deductive, Inductive, and Abductive Reasoning in Language Models](https://arxiv.org/abs/2404.03028) | 该研究通过四个语言模型和两类学习任务发现，语言模型的演绎推理（指令遵循）、归纳推理（少样本提示）与溯因推理（指令推断）三种能力之间存在强烈的不对称分离，表明这三种推理形式并未构成一个完整闭合的推理循环。 |
| [^321] | [A Controlled Reevaluation of Coreference Resolution Models](https://arxiv.org/abs/2404.00727) | 基于对预训练语言模型大小的控制，我们发现基于编码器的核指代解析模型在准确性和推理速度方面优于更近期的基于解码器的模型，而在基于编码器的模型中，最老的模型在跨域文本体裁中表现最佳。 |
| [^322] | [Syntactic Ghost: An Imperceptible General-purpose Backdoor Attacks on Pre-trained Language Models](https://arxiv.org/abs/2402.18945) | 论文提出了一种名为Syntactic Ghost的新方法，实现了对预训练语言模型进行无感知和通用的后门植入。 |
| [^323] | [Universal Topological Regularity of Syntactic Structures](https://arxiv.org/abs/2302.00129) | 该论文通过分析124种语言的依存树，发现句法结构具有偏离随机性的普遍拓扑规律（更高的结构稳健性与更低的分支异质性），并借助亚线性优先连接模型证明这些规律可从认知驱动的增量语法编码过程中自然涌现，无需直接优化即可实现交际效率。 |

# 详细

[^1]: 贝尔曼策略优化

    Bellman Policy Optimization

    [https://arxiv.org/abs/2609.15987](https://arxiv.org/abs/2609.15987)

    提出无评论家的贝尔曼策略优化（BPO），利用贝尔曼方程将策略镜像下降重新表述为轨迹级目标，避免了中间状态价值估计且保持相同的最优解，从而有效提升大语言模型的数学推理能力。

    

    可验证奖励强化学习（RLVR）能够提升大语言模型（LLM）的推理能力。我们提出了贝尔曼策略优化（BPO），这是一种由策略镜像下降（PMD）推导出的无评论家（critic-free）方法。针对具有终止奖励的自回归生成任务，BPO利用贝尔曼方程将PMD重新表述为轨迹级别的目标函数。这种重新表述避免了在中间状态进行状态价值估计。我们证明该目标函数与原始PMD目标具有相同的唯一最优解。我们通过近似该目标函数推导出实用的BPO损失，其失配校正权重是互补token概率的平滑比率。在数学推理基准上的实验证明了BPO的有效性。

    arXiv:2609.15987v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) improves the reasoning capabilities of large language models (LLMs). We introduce Bellman Policy Optimization (BPO), a critic-free method derived from Policy Mirror Descent (PMD). For autoregressive generation with terminal rewards, BPO uses the Bellman equations to reformulate PMD as a trajectory-level objective. The reformulation avoids estimating state values at intermediate states. We prove that it has the same unique optimal solution as the original PMD objective. We derive the practical BPO loss by approximating this objective. Its mismatch-correction weight is a smoothed ratio of complementary token probabilities. Experiments on mathematical reasoning benchmarks demonstrate the effectiveness of BPO.
    
[^2]: 星际竞技场：面向数学与理论计算机科学长周期研究的多智能体框架

    Stellar Colosseum: A Many-Agent Harness for Long-Horizon Research in Mathematics and Theoretical Computer Science

    [https://arxiv.org/abs/2609.15983](https://arxiv.org/abs/2609.15983)

    提出Stellar Colosseum，一个与模型无关的多智能体框架，通过策略探索、就绪门控、子问题分解、定向证伪和树聚合等机制，提升语言模型在数学与理论计算机科学长周期研究问题上的可靠性。

    

    语言模型可以生成看似合理的短证明，但在长周期研究问题上可能仍然不可靠，因为这类问题的进展取决于一系列不确定且相互关联的决策。我们提出了Stellar Colosseum（星际竞技场），这是一个与模型无关的框架，用于在数学和理论计算机科学研究中分配推理资源。Colosseum在构建证明之前探索备选策略，使用就绪门来决定某条路线何时成熟到足以进行分解，将证明计划表示为相互关联的章节级子问题，并将验证器的发现路由回论证中受影响的部分。在这些阶段中，它并行生成候选方案，用有针对性的证伪来攻击它们，并通过重叠随机样本树聚合将候选方案及其批评意见合并为单一的研究成果。Colosseum工作流也已被集成到Google Antigravity的Teamwork框架中。

    arXiv:2609.15983v1 Announce Type: new  Abstract: Language models can produce plausible short proofs, but may still be unreliable on long-horizon research problems, where progress depends on a sequence of uncertain and interdependent decisions. We introduce Stellar Colosseum, a model-agnostic harness for allocating inference across research in mathematics and theoretical computer science. Colosseum explores alternative strategies before proof construction, uses a readiness gate to decide when a route is mature enough to decompose, represents the proof plan as interdependent section-level subproblems, and routes verifier findings back to the affected part of the argument. Across these stages, it generates candidates in parallel, attacks them with targeted falsification, and combines candidates and their critiques into a single research artifact through overlapping random-sample tree aggregation. The Colosseum workflow has also been integrated into Google Antigravity's Teamwork framework 
    
[^3]: 内在路由器：从冻结LLM中引出原生技能路由

    The Router Within: Eliciting Native Skill Routing from a Frozen LLM

    [https://arxiv.org/abs/2609.15982](https://arxiv.org/abs/2609.15982)

    该论文提出Gavel方法，证明冻结的LLM在其内部前向传播中已天然携带技能路由信号，仅需训练两个线性映射即可在无需将技能文本放入上下文的情况下实现高效且可扩展的技能选择。

    

    技能能够将大语言模型（LLM）智能体的能力扩展到其参数化知识之外，而其所承诺的收益取决于能否选对技能。已部署的系统通过将每个技能的元数据预加载到上下文中进行路由，这会分散智能体的注意力并限制技能库的规模。检索式流水线将选择过程移出了上下文，但同时也将选择能力移出了智能体本身。我们证明，冻结的智能体LLM在其自身的前向传播中已经携带了路由信号，并且只需两个线性映射就足以将其读出，而无需在上下文中放入任何技能文本。Gavel（来自冻结LLM的一瞥与裁决）分两步读取该信号：“一瞥”步骤将任务和每个技能的中间层状态通过这两个映射（唯一被训练的参数）进行投影，并对照安装时通过一次前向传播构建的紧凑的每技能库，对整个技能库进行评分；“裁决”步骤随后恢复入围技能的前向传播，读取模型自身的似然和是/否判断。

    arXiv:2609.15982v1 Announce Type: cross  Abstract: Skills extend an LLM agent beyond its parametric knowledge, and the gain they promise rests on picking the right one. Deployed harnesses route by preloading every skill's metadata into the context, which disperses the agent's attention and caps the library size. Retrieval pipelines move the selection out of the context, but also out of the agent's capability. We show that the frozen agent LLM already carries the routing signal in its own forward passes, and that two linear maps suffice to read it out with no skill text in the context. Gavel (Glance And Verdict from a frozen LLM) reads it in two steps. A glance projects the task's and each skill's mid-layer states through the two maps, the only parameters trained, and scores the full library against compact per-skill banks that one forward pass builds at installation. A verdict then resumes the shortlisted skills' forward passes and reads the model's own likelihood and yes/no judgment, 
    
[^4]: 通过方向分解解析Transformer中的表示演化

    Disentangling Representation Evolution in Transformers through Directional Decomposition

    [https://arxiv.org/abs/2609.15975](https://arxiv.org/abs/2609.15975)

    本文提出一种方向分解方法，将Transformer中学习到的更新拆解为平行与垂直分量，发现排除自身的值空间平行操作在模型编辑中远比其他空间操作更稳健，并能以分量级分辨率刻画压缩带来的更新误差。

    

    Transformer的表示通过学习到的加性变换不断演化，这些变换要么保持当前方向，要么将其重新定向。我们将这种演化作为一种功能几何来研究，将学习到的更新分解为平行分量和垂直分量。在预训练模型中，我们发现除残差恒等路径之外还存在大量的平行分量。随后我们在两个空间中应用这一分解：相对于隐藏状态的注意力与MLP更新，以及相对于当前token值的注意力值聚合。针对性的编辑实验揭示了一种强烈的依赖于空间的不对称性：排除自身的值空间平行操作明显比残差空间和垂直方向的对应操作更为稳健，它能在仅缩放非自身聚合信息的同时保留直接的自身消息。同样的分解方法还为压缩引起的更新误差提供了分量级的刻画：垂直误差能够区分不同的压缩方法

    arXiv:2609.15975v1 Announce Type: new  Abstract: Transformer representations evolve through learned additive transformations that either preserve their current direction or redirect it. We study this evolution as a functional geometry, decomposing learned updates into parallel and perpendicular components. Across pretrained models, we find substantial parallel components beyond the residual identity path. We then apply the decomposition in two spaces: to attention and MLP updates relative to the hidden state, and to attention value aggregation relative to the current token's value. Targeted edits reveal a strongly space-dependent asymmetry: exclude-self value-space parallel manipulation is markedly more robust than residual-space and perpendicular counterparts, preserving the direct self message while scaling only the non-self aggregate. The same decomposition gives a component-resolved description of compression-induced update error: perpendicular error separates compression methods m
    
[^5]: 发现基础模型：迈向开放式发现智能

    Discovery Foundation Models: Toward Open-Ended Discovery Intelligence

    [https://arxiv.org/abs/2609.15973](https://arxiv.org/abs/2609.15973)

    本文提出“发现智能”这一新前沿概念，并构建发现基础模型（DFM）框架及其实例Zetema，通过支持问题发现、表示构建、假设形成、基于证据的修订等七项耦合能力，使AI从解决人类指定问题转向自主参与新知识与新问题的创造过程。

    

    基础模型已经从对现有知识的学习与推理，逐步发展到通过行动、工具使用和结果反馈来进行学习。我们认为下一个前沿是进一步的转变：从解决和执行人类指定的问题，转向参与创造新问题、新表示、新解释和新知识的过程。我们将这种能力称为发现智能。我们将发现基础模型（DFMs）表述为面向开放式发现的通用模型系统。DFM在一个可修订的研究状态上运行，并支持七项相互耦合的能力，涵盖问题发现、问题表述、表示构建、假设形成、干预、基于证据的修订以及持续的发现改进。我们通过Zetema实例化了这一框架，Zetema将显式的研究状态动态、验证与实验门控以及外部知识（工具）耦合起来……

    arXiv:2609.15973v1 Announce Type: new  Abstract: Foundation models have progressed from learning and reasoning over existing knowledge, to increasingly learning through action, tool use, and outcome feedback. We argue that the next frontier is a further transition: from solving and acting within problems specified by humans to participating in the process by which new problems, representations, explanations, and knowledge are created. We refer to this capability as Discovery Intelligence. We formulate Discovery Foundation Models (DFMs) as general-purpose model systems for open-ended discovery. A DFM operates over a revisable research state and supports seven coupled capabilities spanning problem discovery, formulation, representation construction, hypothesis formation, intervention, evidence-grounded revision, and continual discovery improvement. We instantiate this framework with Zetema, which couples explicit research-state dynamics, verification and experimental gating, external gro
    
[^6]: Mind2Dialogue：通过模拟用户心理状态来训练具有人类感知能力的语言模型

    Mind2Dialogue: Training Human-Aware Language Models by Simulating User Mental States

    [https://arxiv.org/abs/2609.15972](https://arxiv.org/abs/2609.15972)

    提出Mind2Dialogue框架，通过心理学引导的模拟器模拟用户心理状态并将其转化为特权监督信号，以训练能够理解用户潜在信念和目标的人类感知型语言模型。

    

    随着语言模型能力的不断增强，在学习、推理和决策制定方面的长期协作需要模型对其服务对象有更深入的理解。然而，训练这类具有人类感知能力的语言模型面临着根本性的监督缺口，因为当前用于大语言模型助手训练的数据集几乎不包含明确基于用户未言明的信念和目标的、信息充分的回应。由于用户的潜在状态无法被直接观察，扩展此类监督天然受到限制。因此，我们提出了Mind2Dialogue框架，通过模拟用户的心理状态并将其转化为特权监督信号用于人类感知训练，从而缓解这一缺口。具体而言，我们首先提出了一个心理学引导的模拟器，它在保留个人特征的同时，通过交互更新心理状态，以生成连贯的对话。其核心思想是强制维护一个共享的、不断演化的心理状态来驱动用户行为。

    arXiv:2609.15972v1 Announce Type: new  Abstract: As language models become more capable, long-term collaboration in learning, reasoning, and decision-making calls for a deeper understanding of the people they serve. Yet training such human-aware language models faces a fundamental supervision gap because current datasets for LLM assistant training contain few if any well-informed responses explicitly grounded in users' unspoken beliefs and goals. Scaling such supervision is inherently constrained, as users' underlying states are not directly observable. We thus propose the Mind2Dialogue framework to mitigate this gap by simulating users' mental states and turning them into privileged supervision for human-aware training. Specifically, we first propose a psychology-guided simulator that preserves personal characteristics while updating mental states through interaction to generate coherent conversations. The key idea is to enforce a shared evolving mental state that drives user behavior
    
[^7]: 构建即可验证：临床问答中逐字引用的声明级评估

    Verifiable by Construction: Claim-Level Evaluation of Verbatim Citation in Clinical Question Answering

    [https://arxiv.org/abs/2609.15964](https://arxiv.org/abs/2609.15964)

    该论文基于四份临床实践指南构建了标准化评估框架，从为每个事实性声明提供引用、生成逐字引用到确保引用完全支撑声明，端到端地评估了十二个大语言模型在临床问答中构建可验证答案的能力。

    

    大语言模型（LLMs）已被广泛应用于临床问答任务。当前系统可以在答案后附加引用，但这些引用通常指向较为宽泛的文本，使得时间紧迫的临床医生无法高效地进行验证。另一种方案是确保响应在构建时就具备可验证性：提供来自参考材料的细粒度逐字引用来支撑声明，使用户无需打开其他文档即可验证答案。在本文中，我们评估了当前模型端到端执行此任务的能力：从为每个事实性声明提供引用，到生成逐字引用，再到确保这些引用能够完全支撑相应声明。为此，我们基于四份临床实践指南构建了一个标准化评估框架，并在222个合成的临床问题上评估了十二个大语言模型，分别衡量上述各个阶段的表现。我们发现大多数模型可以为声明附加逐字引用（摘要在此处不完整）。

    arXiv:2609.15964v1 Announce Type: new  Abstract: Large language models (LLMs) have been widely adopted for clinical question answering (QA). Current systems can attach citations to their answers, but these often point to broad texts, leaving time-pressed clinicians unable to verify them efficiently. An alternative is to ensure that responses are verifiable by construction: providing fine-grained verbatim quotes from reference material that substantiate claims, so users can verify an answer without opening other documents. In this paper, we evaluate the ability of current models to perform this task end-to-end: from providing citations for every factual claim, to producing verbatim quotes, to ensuring that those quotes fully substantiate the claims. To do so, we build a standardized harness over four clinical practice guidelines and evaluate twelve LLMs on 222 synthetic clinical questions, measuring each of these stages separately. We find that most models can attach verbatim quotes to 
    
[^8]: HypoEvolve：遗传算法使多智能体大语言模型能够发现科学假设

    HypoEvolve: Genetic Algorithms Enable Multi-Agent LLMs to Discover Scientific Hypotheses

    [https://arxiv.org/abs/2609.15938](https://arxiv.org/abs/2609.15938)

    该论文提出HypoEvolve框架，通过代际遗传算法协调多个专门化大语言模型智能体，使智能体协作过程显式化，从而发现高质量的科学假设。

    

    科学智能体通过综合证据、评估提案和发展新的解释来促进假设的发现。近期的系统通过批判、比较和修订等方式将科学智能体与进化搜索相结合。然而，不同形式的智能体协作如何影响假设质量仍然是一个悬而未决的问题。回答这个问题需要将智能体的科学能力与其协作的影响分离开来。因此，一个框架必须保留智能体的科学角色，并支持组合、修订和保留假设的规则。基于这一观点，我们提出了HypoEvolve，它通过对假设种群的连续更新使协作过程显式化。具体而言，我们提出了一种代际遗传算法来协调专门的大语言模型（LLM）智能体，这些智能体整合机制性论证、重新审视假设前提、并评估证据与可检验性。

    arXiv:2609.15938v1 Announce Type: new  Abstract: Scientific agents contribute to hypothesis discovery by synthesizing evidence, assessing proposals, and developing new explanations. Recent systems combine scientific agents with evolutionary search through critique, comparison, and revision. However, how different forms of agent collaboration affect hypothesis quality remains an open question. Answering this question requires separating the effects of agents' scientific capabilities from those of their collaboration. A framework must therefore preserve agents' scientific roles and support rules for combining, revising, and retaining hypotheses. Building on this view, we introduce HypoEvolve, which makes collaboration explicit through successive updates to a hypothesis population. Specifically, we propose a generational genetic algorithm to coordinate specialized large language model (LLM) agents that integrate mechanistic arguments, reconsider assumptions, and assess evidence and testab
    
[^9]: 基于习得新造词的接种式中训练

    Inoculation Midtraining with Learned Neologisms

    [https://arxiv.org/abs/2609.15886](https://arxiv.org/abs/2609.15886)

    提出接种式中训练技术，通过在中训练阶段引入新造词标记不安全行为所属的上下文，可在保留良性特性迁移的同时降低模型对齐失准，但其效果不及标准接种式提示且对训练配置敏感。

    

    大型语言模型（LLMs）在后训练过程中常常同时学到理想和不理想的特性。我们研究了中训练（midtraining）这一更早期的训练阶段能否塑造这些特性中哪些会在后续泛化。我们提出了接种式中训练，该技术在训练中向基座模型传授不安全行为属于某个特定上下文的知识，这一上下文由中训练期间引入的一个新造词（新token）来标记，然后在该上下文内用不安全数据对模型进行后训练。随后我们在该上下文之外评估模型，即在系统提示中排除该新造词。在监督微调和强化学习两种后训练机制下，我们发现接种式中训练可以在保留良性数据特性迁移（例如使用德语或莎士比亚文体表达）的同时减少模型的对齐失准。然而，我们的方法并未超越标准的接种式提示，并且对训练配置较为敏感。

    arXiv:2609.15886v1 Announce Type: new  Abstract: Large language models (LLMs) often learn both desirable and undesirable properties during post-training. We study whether midtraining, an earlier training stage, can shape which of these properties later generalise. We introduce Inoculation Midtraining, a technique that teaches a base model that unsafe behaviour belongs to a designated  context, as indicated by the  neologism (a new token) introduced during midtraining, and then post-trains the model on unsafe data within that context. We then evaluate the model outside the context, with the  neologism excluded from the system prompt. Across supervised fine-tuning and reinforcement learning post-training regimes, we find that Inoculation Midtraining can reduce misalignment while preserving the transfer of benign data properties (e.g., speaking in German or Shakespearean prose). However, our approach does not outperform standard Inoculation Prompting, is sensitive to training configuratio
    
[^10]: K-Bench：用于评估大语言模型在高风险心理健康对话中表现的临床校准基准

    K-Bench: a clinically calibrated benchmark for evaluating large language models in high-risk mental health conversations

    [https://arxiv.org/abs/2609.15855](https://arxiv.org/abs/2609.15855)

    该论文提出了K-Bench，一个经临床医生校准的基准，包含200个涉及自杀、自残、家庭暴力等多轮高风险心理健康情景案例，并采用与临床医生共识达94.2%一致率的GPT-4o自动评判系统，系统评估了33个基础大语言模型在高风险心理健康对话中的安全性。

    

    人们越来越多地使用大语言模型（LLM）来获取心理健康支持，但其在不断演变的高风险对话中的安全性仍未得到充分刻画。我们开发了K-Bench，这是一个经临床医生校准的受保护基准，评估了来自14家提供商的33个基础模型所对应的125种模型配置，测试涵盖200个固定的多轮情景案例，涉及自杀、自残、家庭暴力、药物滥用以及无风险表现。合成患者对话与真实的人机对话显示出高度的分布重叠。在来自151份经临床医生评分的转录文本的6,751项合格项目比较中，冻结的GPT-4o评判模型与临床医生共识达到了94.2%的完全一致率。领先的模型在提供强有力的支持性对话的同时，综合风险得分超过95，而风险探索测试则暴露出表现较差的模型配置之间存在显著差异。治疗性提示……（原文摘要在此处截断）

    arXiv:2609.15855v1 Announce Type: cross  Abstract: % !TEX root = ../main.tex People increasingly use large language models (LLMs) for mental health support, yet their safety in evolving, high-risk conversations remains poorly characterised. We developed K-Bench, a clinician-calibrated, protected benchmark evaluating 125 model configurations representing 33 base models from 14 providers across a fixed cohort of 200 multi-turn vignettes involving suicide, self-harm, domestic violence, substance misuse, and no-risk presentations. Synthetic patient conversations showed substantial distributional overlap with real human-AI conversations. A frozen GPT-4o judge achieved 94.2% exact agreement with clinician consensus across 6,751 eligible item comparisons from 151 clinician-rated transcripts. Leading models combined strong supportive conversation with combined-risk scores above 95, whereas risk exploration exposed substantial variation among lower-performing configurations. Therapeutic prompti
    
[^11]: 学会做教练：面向经验学习的指导方法

    Learning to Coach for Experiential Learning

    [https://arxiv.org/abs/2609.15851](https://arxiv.org/abs/2609.15851)

    提出L2C框架，训练专用“LLM教练”从模型过往轨迹中提炼可操作的经验知识，借助同实例与跨实例奖励进行优化，在数学推理和文本游戏中持续优于自我修正方法，并比扩大执行模型规模更有效地利用推理计算资源。

    

    语言模型可以从经验中学习，但原始的解题轨迹往往过于冗长和嘈杂，难以提供有效的指导。在本工作中，我们提出了 Learning to Coach (L2C)，这是一个训练专用“LLM教练”的框架，用于从执行模型先前的轨迹中提取可操作的经验知识。执行模型保持冻结，而LLM教练则被训练以最大化奖励，该奖励由执行模型在指导下的回答正确性决定。我们研究了两种此类奖励：同实例奖励，用于改进对原始问题的后续回答；以及跨实例奖励，用于提炼出可迁移到其他实例的知识。在数学推理和交互式文本游戏任务上，L2C始终优于自我修正方法以及未经训练的LLM教练。进行更多轮次的经验学习可以进一步提升准确率，并且比扩大执行模型规模能更有效地利用额外的推理计算资源。

    arXiv:2609.15851v1 Announce Type: new  Abstract: Language models can learn from experience, but raw solution trajectories are often too long and noisy to provide effective guidance. In this work, we propose Learning to Coach (L2C), a framework that trains a dedicated LLM-as-a-Coach to extract actionable experiential knowledge from an actor model's previous trajectory. The actor remains frozen, while the LLM-as-a-Coach is trained to maximize a reward given by the correctness of the actor's guided response. We study two such rewards: a same-instance reward, which improves subsequent responses on the original problem, and a cross-instance reward, which elicits knowledge that transfers to other instances. Across mathematical reasoning and interactive text-games, L2C consistently outperforms self-refinement and an untrained LLM-as-a-Coach. Running experiential learning for more iterations further improves accuracy and uses additional inference compute more effectively than enlarging the act
    
[^12]: 在使用LLM进行民意调查之前：一个协商式诊断框架

    Before You Poll with LLMs: A Deliberative Diagnostic Framework

    [https://arxiv.org/abs/2609.15849](https://arxiv.org/abs/2609.15849)

    提出协商式民意调查诊断框架，通过相同信息干预对比人类与LLM的信念变化，揭示五个前沿模型虽能产生看似合理的党派观点，却在动态信念更新方面存在静态评估无法发现的失败。

    

    LLM能否像人类一样对新信息进行推理，还是仅仅检索缓存的观点？这对于硅基采样（silicon sampling）至关重要——即LLM角色在大规模上模拟公众舆论。当前的评估仅测试角色是否持有正确的观点——这是一种静态快照。但舆论研究越来越依赖于动态保真度：即角色是否像人类在协商过程中那样，针对新论点更新自己的信念。目前尚无现有基准测试这一点。我们提出了协商式民意调查诊断框架，该框架在相同的信息干预后比较人类与LLM的信念变化。该框架基于协商式民意调查，能够揭示静态评估所无法察觉的失败：能够产生看似合理的党派观点的模型，仍然可能错误地呈现这些观点的变化方式。将该框架应用于五个前沿模型，并使用"America in One Room"的数据（526个角色、72个问题），我们发现每个模型都未能……

    arXiv:2609.15849v1 Announce Type: cross  Abstract: Can LLMs reason through new information like humans, or do they merely retrieve cached opinions? This is critical for silicon sampling, where LLM personas simulate public opinion at scale. Current evaluations test only whether personas hold the right opinions -- a static snapshot. But opinion research increasingly depends on dynamic fidelity: whether personas update beliefs in response to new arguments, as humans do during deliberation. No existing benchmark tests this. We introduce the Deliberative Polling Diagnostic Framework, which compares human and LLM belief shifts after identical informational interventions. Grounded in deliberative polling, it surfaces failures invisible to static evaluation: models that produce plausible partisan opinions can still misrepresent how those opinions change. Applying the framework to five frontier models using data from America in One Room (526 personas, 72 questions), we find that every model fai
    
[^13]: CiteGuard-RAG：一个以验证为中心的、面向有证据支撑问答的AI系统

    CiteGuard-RAG: A Validation-Centered AI System for Evidence-Grounded Question Answering

    [https://arxiv.org/abs/2609.15830](https://arxiv.org/abs/2609.15830)

    CiteGuard-RAG通过在运行时融合混合语义-词汇检索、引用约束生成、句子级依据验证与单次重新生成机制，实现了高准确率、引用有效且无幻觉的有据问答，在受控评估中达到98.3%的有据答案准确率和引用有效性。

    

    检索增强生成（RAG）可以改善对复杂信息的访问；然而，仅仅检索到证据并不能保证答案有据可依、引用有效，或能够适当地拒绝回答。本文介绍了CiteGuard-RAG，一个以验证为中心的、面向有证据支撑问答的AI系统。该系统集成了混合语义-词汇检索、引用约束生成、句子级依据验证和单次重新生成。验证在运行时用于判断候选答案在最终交付前应被接受、拒绝还是重新生成。CiteGuard-RAG在一个受控住房法数据集、PrivacyQA和CUAD的共400个问题上进行了评估。在受控评估中，它实现了99.1%的检索准确率、98.3%的有据答案准确率和98.3%的引用有效性，且未检测到验证可发现的幻觉。消融实验结果表明，移除相关组件后有据答案准确率急剧下降（原文此处内容截断）。

    arXiv:2609.15830v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) can improve access to complex information; however, retrieving evidence alone does not ensure that answers are grounded, citation-valid, or appropriately refused. This paper introduces CiteGuard-RAG, a validation-centered AI system for evidence-grounded question answering. The system integrates hybrid semantic-lexical retrieval, citation-constrained generation, sentence-level grounding validation, and single-pass regeneration. Validation is used at runtime to determine whether a candidate answer should be accepted, refused, or regenerated before final delivery.   CiteGuard-RAG is evaluated on 400 questions across a controlled housing-law dataset, PrivacyQA, and CUAD. In the controlled evaluation, it achieves 99.1% retrieval accuracy, 98.3% grounded-answer accuracy, and 98.3% citation validity, with no validation-detected hallucinations. Ablation results show that grounded-answer accuracy drops sharp
    
[^14]: EvoOntology：面向数据智能体的自进化本体层

    EvoOntology: A Self-Evolving Ontology Layer for Data Agents

    [https://arxiv.org/abs/2609.15779](https://arxiv.org/abs/2609.15779)

    本文提出EvoOntology，一种面向数据智能体的自进化本体层，通过将本体封装为包含模式层、内容层和工具层的MCP服务器，并借助构建器智能体实现自主构建与自进化循环，有效弥合了智能体与异构数据之间的鸿沟。

    

    数据智能体旨在基于自然语言指令处理异构数据，包括表格、文件和数据库。然而，数据智能体面临一个具有挑战性的“智能体-数据鸿沟”：异构数据存在于智能体之外，而智能体只能通过通用工具（例如列名和文件路径）访问这些数据。现有方法要么让智能体直接探索原始数据源，要么将手动构建的语义层注入提示中。然而，这两种方法都难以很好地扩展到大规模异构数据源，也无法适应不同的智能体行为。在本文中，我们提出了EvoOntology，一种面向数据智能体的自进化本体层。EvoOntology将本体封装为一个MCP服务器，其中包含模式层、内容层和工具层，使智能体能够在运行时主动查询本体并与之交互。为此，我们引入了一个用于自主本体构建的构建器智能体和一个自进化循环……

    arXiv:2609.15779v1 Announce Type: new  Abstract: Data agents aim to fulfill natural-language instructions over heterogeneous data, including tables, files, and databases. However, data agents face a challenging agent-data gap: heterogeneous data resides outside the agent, while the agent can access it (e.g., column names and file paths) only through generic tools. Existing approaches either let agents directly explore raw data sources or inject manually constructed semantic layers into prompts. However, neither scales well to large heterogeneous data sources nor adapts to different agent behaviors. In this paper, we introduce EvoOntology, a self-evolving ontology layer for data agents. EvoOntology encapsulates the ontology as an MCP server comprising a schema layer, a content layer, and a tool layer, enabling agents to actively query and interact with the ontology at runtime. To this end, we introduce a builder agent for autonomous ontology construction and a self-evolution loop that c
    
[^15]: 在全双工语音到语音模型中实现流式用户语音转写

    Enabling Streaming User Transcription in Full-Duplex Speech-to-Speech Models

    [https://arxiv.org/abs/2609.15759](https://arxiv.org/abs/2609.15759)

    通过在双工S2S模型中并行添加轻量级ASR头，以极少的额外参数和架构改动实现了实时流式用户语音转写，同时保留全双工对话能力，在HuggingFace开放ASR排行榜上达到10.21%的流式平均词错误率。

    

    全双工语音到语音（S2S）模型通过允许同时进行听和说，实现了自然的对话式人工智能。然而，这些模型通常缺乏内建的用户语音转写功能，而这一功能对于对话记录、无障碍功能和质量监控等应用至关重要。在这项工作中，我们提出了一种高效的方法，通过在与智能体文本头并行的位置引入一个轻量级的ASR（自动语音识别）头，为现有的双工S2S模型添加流式ASR能力。我们的方法仅需极少量的额外参数，且无需对基础S2S模型进行重大的架构更改，即可实现实时的用户语音转写，同时保留包括轮次切换和插话处理在内的全双工对话能力。实验结果表明，我们的方法在双工S2S框架内于HuggingFace开放ASR排行榜上实现了10.21%的流式平均词错误率（WER）。此外，我们还表明，同样的……

    arXiv:2609.15759v1 Announce Type: new  Abstract: Full-duplex speech-to-speech (S2S) models enable natural conversational AI by allowing simultaneous listening and speaking. However, these models typically lack inherent user speech transcription, which is essential for applications such as conversation logging, accessibility features, and quality monitoring. In this work, we propose an efficient method to add streaming ASR capabilities to an existing duplex S2S model by introducing a lightweight ASR head in parallel to the agent text head. Our approach requires minimal additional parameters and no significant architectural changes to the base S2S model, enabling real-time user transcription while preserving full-duplex conversational capabilities including turn-taking and barge-in handling. Experimental results demonstrate that our method achieves streaming average WER of 10.21% on the HuggingFace Open ASR Leaderboard within the duplex S2S framework. Additionally, we show that the same 
    
[^16]: 面向跨语言低资源自动语音识别的顺序适配器堆叠方法

    Sequential Adapter Stacking for Cross-Lingual Low-Resource ASR

    [https://arxiv.org/abs/2609.15758](https://arxiv.org/abs/2609.15758)

    提出顺序适配器堆叠方法，在冻结的源语言适配器之上堆叠可训练的目标语言适配器，实现从资源丰富语言到低资源语言的高效知识迁移，在Whisper不支持的三种语言上持续显著优于全量微调。

    

    将大规模多语言自动语音识别（ASR）模型扩展到低资源语言仍然是一项挑战。模型性能偏向高资源语言，而对于标注数据和预训练覆盖有限的语言，性能会急剧下降。为解决这一问题，我们在Whisper上研究了将知识从资源丰富的源语言迁移到低资源目标语言的参数高效方法。除了热启动初始化和基于注意力的融合之外，我们提出了顺序适配器堆叠方法，即在冻结的源语言适配器之上放置一个可训练的目标语言适配器。在受控实验中，这些方法在Whisper不支持的三种目标语言上进行了评估——阿斯图里亚斯语、阿萨姆语和科萨语——并使用了相关程度不同的源语言。结果表明，使用最相关源语言的顺序适配器堆叠方法始终且显著地优于全量微调。

    arXiv:2609.15758v1 Announce Type: new  Abstract: Extending large-scale multilingual automatic speech recognition (ASR) models to low-resource languages remains challenging. Model performance is skewed toward high-resource languages and degrades sharply for languages with limited labeled data and pre-training exposure. To address this, we investigate parameter-efficient approaches for transferring knowledge from resource-rich source languages to low-resource target languages on Whisper. Alongside warm initialization and attention-based fusion, we propose Sequential Adapter Stacking, which places a trainable target-language adapter on top of a frozen source-language adapter. Under controlled experiments, these approaches are evaluated on three target languages unsupported by Whisper -- Asturian, Assamese, and Xhosa -- using source languages with varying degrees of relatedness. Sequential Adapter Stacking with the closest related source consistently and significantly outperforms full fine
    
[^17]: 三思而后行：基于内部归因信号的事实性解码

    Look Before You Leap: Factual Decoding with Internal Attribution Signals

    [https://arxiv.org/abs/2609.15745](https://arxiv.org/abs/2609.15745)

    提出DescaPE解码框架，利用LLM中事实性显著层区间的内部归因信号，通过轻量级探针在推理时惩罚易产生幻觉的候选续写，从而抑制事实性错误的滚雪球效应。

    

    幻觉仍然是大语言模型（LLM）面临的关键挑战：早期的事实性错误会在自回归生成过程中不断累积，形成滚雪球效应，无论是事后纠正还是权重层面的干预都无法有效预防这一问题。我们提出了DescaPE（DEcoding Signal Control Against Path Error-snowballing，对抗路径错误滚雪球的解码信号控制），这是一个利用模型内部信号在推理阶段抑制易产生幻觉轨迹的解码框架。通过滑动窗口MLP消融实验，我们识别出LLM中一个事实性显著的层区间，该层区间衍生的信号对事实性token有选择性地升高，并在易产生幻觉的步骤表现出异常尖峰。我们训练了一个轻量级探针，通过单次前向传播来近似该信号，并将其整合到候选评分中，以惩罚高风险的续写内容，同时奖励有事实依据的续写内容。在三个大语言模型上的五个事实性基准测试实验表明……

    arXiv:2609.15745v1 Announce Type: cross  Abstract: Hallucination remains a critical challenge in large language models (LLMs), where early factual errors compound through autoregressive generation in a snowballing effect that neither post-hoc correction nor weight-level intervention can effectively preempt. We propose DescaPE (DEcoding Signal Control Against Path Error-snowballing), a decoding framework that leverages internal model signals to suppress hallucination-prone trajectories at inference time. Through sliding-window MLP ablation, we identify a factual-salient layer span within LLMs whose derived signal is selectively elevated for factual tokens and exhibits anomalous spikes at hallucination-prone steps. We train a lightweight probe to approximate this signal from a single forward pass and integrate it into candidate scoring to penalize high-risk continuations while rewarding factually grounded ones. Experiments across five factuality benchmarks on three LLMs demonstrate that 
    
[^18]: 融合大语言模型知识用于自动语音识别

    Merging the Knowledge of LLMs for Automatic Speech Recognition

    [https://arxiv.org/abs/2609.15743](https://arxiv.org/abs/2609.15743)

    本文提出通过模型合并（对LoRA参数进行算术运算）将外部语言模型知识直接融入基于大语言模型的ASR模型参数中，无需额外推理计算成本即可持续提升目标领域的语音识别性能。

    

    基于配对语音-文本数据训练的自动语音识别（ASR）系统，已经通过利用仅在文本数据上训练的语言模型（LM）得到改进。语言模型融合方法，如浅层融合和密度比方法，是在ASR解码过程中引入外部语言模型的成熟方法。然而，这些方法由于需要语言模型推理而产生额外的计算成本，这对于近年来日益庞大的语言模型来说问题尤为突出。在本研究中，我们提出通过模型合并的方式引入外部语言模型。该方法将语言模型直接集成到基于大语言模型的ASR模型参数中，在推理时无需额外的计算成本。我们通过对LoRA参数进行算术运算来实现领域扩展与迁移。我们在基于CSJ和LibriSpeech训练的基于大语言模型的ASR领域自适应上进行了实验评估。结果表明，我们的语言模型合并方法在目标领域一致地提升了ASR性能，且无需额外的……

    arXiv:2609.15743v1 Announce Type: new  Abstract: Automatic speech recognition (ASR) systems, trained on paired speech-text data, have been improved by leveraging language models (LMs) trained on text-only data. LM fusion methods such as shallow fusion and density ratio are well-established methods that incorporate external LMs during ASR decoding. However, they incur additional computational costs due to LM inference, which is particularly problematic for recent larger LMs. In this study, we propose incorporating external LMs via model merging. This method integrates the LMs directly into the parameters of an LLM-based ASR model, requiring no additional computational cost at inference. We formulate domain extension and transfer via arithmetic operations on LoRA parameters. Experimental evaluations were conducted for the domain adaptation of LLM-based ASR trained on CSJ and LibriSpeech. We show that our LM merging consistently improved the ASR performance in the target domains, without 
    
[^19]: 数据叙事遇上可解释机器学习：在不泄露敏感数据和模型细节的前提下为非专家解读AI决策

    Data storytelling meets interpretable machine learning: Decoding AI decisions for non-experts without revealing sensitive data and model details

    [https://arxiv.org/abs/2609.15722](https://arxiv.org/abs/2609.15722)

    本研究将数据叙事与可解释机器学习相结合，通过提出DIST金字塔、I-P-O模型以及"What-if"和"Why-not"事件生成架构，并采用数据脱敏技术，使非专家用户能够在不泄露敏感数据和模型细节的情况下理解AI决策。

    

    AI驱动的自动化决策既需要预测性能，也需要可解释性。可解释机器学习（IML）的最新进展为解释模型预测提供了工具，但这些解释的技术复杂性可能阻碍非专家用户的理解。为应对这一挑战，本研究将数据叙事与IML相融合，以增强AI生成决策对更广泛受众的可解释性。遵循设计科学研究（DSR）范式，本研究提出了IML中数据叙事的形式化定义，引入了DIST金字塔以实现数据叙事与IML的对齐，并提出了I-P-O模型来描述两者的交互。研究进一步开发了一种架构，通过独特的"What-if"（如果怎样）和"Why-not"（为什么不）事件生成过程来解释AI决策。该架构还采用数据脱敏技术来保护敏感的输入数据。为验证该方法，本研究进行了一项案例研究。

    arXiv:2609.15722v1 Announce Type: new  Abstract: AI-driven automated decision-making requires both predictive performance and interpretability. Recent advances in interpretable machine learning (IML) provide tools for explaining model predictions, but the technical complexity of these explanations may hinder accessibility to non-experts. To address this challenge, this study integrates data storytelling with IML to enhance the explainability of AI-generated decisions for a broader audience. Following the design science research (DSR) paradigm, this study proposes a formal definition of data storytelling in IML, introduces the DIST Pyramid to align data storytelling with IML, and presents the I-P-O Model to describe their interactions. It further develops an architecture to explain AI decisions through distinct "What-if" and "Why-not" event-generation processes. The architecture also employs data desensitization to protect sensitive input data. To validate the approach, a case study is 
    
[^20]: RESKILL：面向交互式语言代理的显式失败归因与结构化修复

    RESKILL: Explicit Failure Attribution and Structured Repair for Interactive Language Agents

    [https://arxiv.org/abs/2609.15684](https://arxiv.org/abs/2609.15684)

    RESKILL提出了一种结构化技能修复框架，通过显式维护失败归因与候选补丁的关联、基于覆盖率的局部修复选择，以及跨修复轮次传递重测结果，显著改进了交互式语言代理失败后的技能修复过程。

    

    语言代理越来越依赖可复用的技能，但失败后的修复通常由不透明的一次性反思（one-shot reflection）处理：模型在生成技能补丁时，既不显式维护失败解释与候选修复之间的关联，也不考虑未成功的重测应如何影响后续编辑。我们提出了RESKILL，这是一个在修复轮次之间维护显式修复状态的结构化修复框架。给定一次失败的执行轨迹，该框架将失败假设与候选技能补丁相关联，通过基于覆盖率的归因方法选择局部修复，在环境中重新测试编辑后的技能集，并利用重测结果指导后续的修复更新。语言模型提供结构化的修复因素，而修复程序则记录这些因素，根据局部技能补丁对当前活跃失败解释的解决程度进行比较，并将未成功的重测结果带入后续的修复轮次。我们在ALFWorld上评估了RESKILL。

    arXiv:2609.15684v1 Announce Type: new  Abstract: Language agents increasingly rely on reusable skills, but post-failure repair is often handled by opaque one-shot reflection: a model generates a skill patch without explicitly maintaining how failure explanations relate to candidate repairs or how unsuccessful retests should influence later edits. We introduce RESKILL, a structured repair framework that maintains an explicit repair state across repair rounds. Given a failed rollout, the framework links failure hypotheses to candidate skill patches, selects local repairs through coverage-based attribution, retests the edited skill set in the environment, and uses retest outcomes to guide subsequent repair updates. The language model supplies structured repair factors, while the repair procedure records them, compares local skill patches by how well they address active failure explanations, and carries unsuccessful retest outcomes into later repair rounds. We evaluate RESKILL on ALFWorld 
    
[^21]: CiteShade：多源检索增强生成中的引用洗白攻击及其反事实防御

    CiteShade: Citation Laundering in Multi-Source Retrieval-Augmented Generation and Its Counterfactual Defense

    [https://arxiv.org/abs/2609.15660](https://arxiv.org/abs/2609.15660)

    该论文提出首个针对检索增强生成系统的“引用洗白”攻击CiteShade，攻击者仅通过控制单一来源即可诱导模型输出错误答案并将其错误归因于可信来源，同时给出了三个攻击必要条件及相应的反事实防御方法。

    

    检索增强生成（RAG）将语言模型的回答建立在检索到的外部知识之上，并为每个回答附上标明来源的引用。这些引用是用户的审计线索：它们使读者无需信任模型本身即可验证某个说法。以往关于RAG的安全研究关注的是攻击者能否破坏答案内容，而引用通道一直未被探索。我们证明这一通道是一个全新的、切实可行的攻击面。我们提出CiteShade，这是首个针对RAG的引用洗白攻击：攻击者只需控制单一来源，即可诱导模型生成攻击者选定的错误答案，并将其归因于一个实际上并不支持该答案的可信来源，而正确答案的证据仍然保留在上下文中。我们将该攻击形式化为一个优化问题，推导出三个必要条件（检索条件、生成条件和引用条件），并构造出在无需任何指令的情况下满足这些条件的恶意来源。

    arXiv:2609.15660v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) grounds a language model's answers on retrieved external knowledge and returns each answer with citations that identify its sources. Those citations are the user's audit trail: they let a reader verify a claim without trusting the model. Prior security work on RAG asks whether an attacker can corrupt the answer, leaving the citation channel unexplored. We show that this channel is a new and practical attack surface. We propose CiteShade, the first citation laundering attack to RAG, in which an attacker controlling a single source induces a model to produce an attacker-chosen wrong answer and to attribute it to a trusted source that does not support it, while the evidence for the correct answer remains in context. We formulate the attack as an optimization problem, derive three necessary conditions (retrieval, generation, and citation) and construct sources satisfying them without any instruction. On
    
[^22]: 共情是可引导但多轴的：大语言模型中的机制几何与角色效应

    Empathy Is Steerable but Multi-Axial: Mechanism Geometry and Persona Effects in LLMs

    [https://arxiv.org/abs/2609.15654](https://arxiv.org/abs/2609.15654)

    研究发现大语言模型的共情可以通过激活引导有效控制，但其机制是多轴的——不同共情维度的激活方向仅部分可分离且相互影响，角色提示也会与这些方向产生交互作用。

    

    激活引导技术已被用于控制诚实、拒绝和谄媚等特质，然而支持性共情是沿多个维度进行评估的，这些维度不一定对应于可独立控制的激活方向。使用将支持性共情分解为情感反应、解释和探索的EPITOME框架，我们研究了三个经过指令微调的大语言模型，探究由这些标签导出的候选方向是产生可区分的干预效果，还是共享结构，以及角色提示如何与这些方向相互作用。我们发现对比激活添加产生了一种稳定的中层干预，能够在各模型间一致地改变EPITOME代理分数，使共情分析超越响应级别的评分。然而，恢复出的方向只是部分可分离的：引导其中一个方向会引起偏离目标的偏移，而手工制作的提示也会改变共情表现……

    arXiv:2609.15654v1 Announce Type: new  Abstract: Activation steering has been used to control traits such as honesty, refusal, and sycophancy, yet supportive empathy is evaluated along multiple dimensions that need not correspond to independently controllable activation directions. Using the EPITOME framework, which decomposes supportive empathy into Emotional Reactions, Interpretations, and Explorations, we study three instruction-tuned LLMs and ask whether candidate directions derived from these labels produce distinguishable intervention effects or instead share structure, and how persona prompts interact with those directions. We find that contrastive activation addition yields a stable middle-layer intervention that consistently shifts the EPITOME proxy scores across models, moving empathy analysis beyond response-level scoring. However, the recovered directions are only partially separable: steering one direction induces off-target shifts, and hand-crafted prompting shifts the em
    
[^23]: 视觉-语言模型中基于人类评估的长文本图文一致性校准

    Human-Grounded Calibration for Long-Text Image-Text Congruence in Vision-Language Models

    [https://arxiv.org/abs/2609.15640](https://arxiv.org/abs/2609.15640)

    本文提出一致性分数——一种轻量级校准层，将双编码器模型的图文相似度证据映射为有界且可解释的一致性分数，并通过基于人类评估的实验揭示了直接事后校准与基于投影方法之间的权衡。

    

    长文本图文一致性评分对于需要评估详细文本描述是否与视觉内容相匹配的视觉-语言系统而言日益重要。然而，双编码器模型产生的原始相似度分数难以被解释为经过校准的一致性度量，尤其是在图像与文本嵌入之间存在模态鸿沟的情况下。本文提出了一致性分数，这是一种轻量级的校准层，可将图文相似度证据映射为有界分数。基于 DOCCI 和 Urban1k 数据集，我们评估了四个冻结的视觉-语言骨干模型，并观察到投影后质心距离的降低并不能一致地提升图文检索性能。在 DOCCI 上进行的基于人类评估的实验进一步揭示了一种权衡：直接的事后校准能够保持与人类判断的高度关联，而某些基于投影的配置则会降低与阈值相关的斜率和截距……

    arXiv:2609.15640v1 Announce Type: cross  Abstract: Long-text image--text congruence scoring is increasingly important for vision-language systems that must evaluate whether detailed textual descriptions match visual content. However, raw similarity scores from dual-encoder models are difficult to interpret as calibrated congruence measures, especially under the modality gap between image and text embeddings. This paper proposes Congruency Score (CS), a lightweight calibration layer that maps image--text similarity evidence into a bounded score. Using DOCCI and Urban1k, we evaluate four frozen vision-language backbones and show that observed reductions in post-projection centroid distance do not uniformly improve image--text retrieval performance. Human-grounded evaluations on DOCCI further reveal a trade-off: direct post-hoc calibration preserves high association with human judgments, whereas selected projection-based configurations can reduce threshold-relevant slope and intercept dis
    
[^24]: IROH：基于多阶段混合检索与原理蒸馏LLM裁判的幽默洞察排序——面向JOKER 2026赛道任务1（英语）

    IROH: Insightful Ranking Of Humor using Multi-Stage Hybrid Retrieval with Rationale-Distilled LLM Judges for JOKER 2026 Track Task 1 English

    [https://arxiv.org/abs/2609.15618](https://arxiv.org/abs/2609.15618)

    该论文提出IROH三阶段检索系统，通过稀疏-稠密混合检索、交叉编码器重排序与原理蒸馏的LoRA大语言模型裁判集成，在JOKER 2026任务1幽默排序中夺得第一名（MAP 0.6347），并发现原理蒸馏裁判是排序质量的关键驱动因素。

    

    我们的团队VANGUARD提出了IROH（Insightful Ranking of Humor，幽默洞察排序），这是一个面向CLEF 2026 JOKER任务1（英语）的三阶段检索系统，以0.6347的MAP成绩位列排行榜第一名。我们的处理流程结合了稀疏-稠密混合检索、交叉编码器重排序，以及经LoRA适配的大语言模型裁判集成。我们使用Gemma 4在两种提示策略（通用型和类型化）下生成查询感知的原理说明，并生成多达四种类型的结构化困难负例用于训练数据构建。通过对三种交叉编码器架构、四种稠密嵌入模型和八种裁判配置的消融实验，我们得到三项关键发现：（1）原理蒸馏的裁判是排序质量的主要驱动因素，而将原理说明附加到第一阶段索引中的贡献微乎其微；（2）结构化困难负例虽然在本地验证集上会抬高分数，但在几乎所有配置中都会损害泛化能力；以及……

    arXiv:2609.15618v1 Announce Type: cross  Abstract: Our team, VANGUARD, presents IROH (Insightful Ranking of Humor), a three-stage retrieval system for JOKER Task 1 English at CLEF 2026, achieving first place on the leaderboard with 0.6347 MAP. Our pipeline combines hybrid sparse-dense retrieval, cross-encoder reranking, and a LoRA-adapted Large Language Model judge ensemble. We employ Gemma 4 to generate query-aware rationales under two prompt strategies, generic and typed, and produce up to four types of structured hard negatives for training data construction. Through an ablation across three cross-encoder architectures, four dense embedders, and eight judge configurations, our key findings are threefold: (1) the rationale-distilled judge is the primary driver of ranking quality, whereas appending rationales to the first-stage index contributes negligibly; (2) structured hard negatives degrade generalisation in nearly all configurations despite inflating local validation scores; and 
    
[^25]: 观者之眼：基于生物特征与人口统计学条件化的多模态性别歧视检测

    Through the Eyes of the Beholder: Biometric and Demographic Conditioning for Multimodal Sexism Detection

    [https://arxiv.org/abs/2609.15608](https://arxiv.org/abs/2609.15608)

    该论文提出一个以人为本的多模态框架，通过FiLM条件化融合标注者的心理与人口统计学特征、五种输入模态，并将性别歧视检测建模为标签分布学习问题，从而有效捕捉互联网性别歧视检测中的主观性。

    

    检测互联网上的性别歧视本质上是一项主观性任务；我们的团队VANGUARD在EXIST 2026任务2中应对这一挑战，提出了一个以人为本的多模态框架，该框架分析并将人类标注者的心理与人口统计学特征纳入检测流程。我们通过带有特征级线性调制条件化的交叉注意力架构融合五种输入模态。梗图文本使用Gemma 4进行提取和视觉描述，并通过NLLB-200在英语与西班牙语之间进行自动翻译加以增强。文本和图像表示由经过LoRA微调的XLM-RoBERTa和CLIP编码器生成，并与由预训练自编码器编码的传感器特征相融合。为了建模标注者的主观性，我们将子任务2.1构建为标签分布学习问题，在完整的标注者标签分布上优化Kullback-Leibler散度损失。在推理时，预测（原文在此处截断）

    arXiv:2609.15608v1 Announce Type: cross  Abstract: Detecting sexism on the internet is a fundamentally subjective task; our team, VANGUARD, addresses this challenge in the EXIST 2026 Task 2 by proposing a human-centered multimodal framework that analyses and incorporates the psychological and demographic characteristics of human annotators into the detection pipeline. We fuse five input modalities through a cross-attention architecture with Feature-wise Linear Modulation conditioning. Meme text is extracted and visually described with Gemma 4, then augmented by automatic translation between English and Spanish with NLLB-200. Text and image representations are produced by LoRAadapted XLM-RoBERTa and CLIP encoders and fused with sensor features encoded by a pretrained autoencoder. To model annotator subjectivity, we frame Subtask 2.1 as a label distribution learning problem, optimizing a Kullback-Leibler divergence loss over the full annotator label distribution. At inference time, predi
    
[^26]: 我们能信任评判者吗？通过答案扰动验证事实性评估方法

    Can We Trust the Judges? Validation of Factuality Evaluation Methods via Answer Perturbation

    [https://arxiv.org/abs/2609.15561](https://arxiv.org/abs/2609.15561)

    本文提出一个元评估框架，通过对标准答案进行受控扰动来验证事实性评估指标的可靠性，发现基于流水线的方法（如RAGAS）比LLM-as-judge方法更能有效追踪事实性退化，并提出了一种更具成本效益的事实正确性指标新变体。

    

    评估大型语言模型（LLM）的事实正确性对许多应用至关重要。但我们的评估工具本身是否可信？尽管基于事实性的指标日益兴起，其敏感性和可靠性仍未得到充分探索。本文介绍了一个元评估框架，通过对黄金标准答案进行受控破坏来系统地测试这些指标。我们的方法生成具有已知退化程度的排序输出，以探究各指标如何捕捉真实性的细微变化。实验表明，基于流水线的方法（如RAGAS的事实正确性指标）比LLM作为评判者的方法更能准确地追踪答案质量的退化。此外，我们还提出了一种事实正确性指标的新变体，能够提供具有竞争力且成本高效的评估方案。

    arXiv:2609.15561v1 Announce Type: new  Abstract: Evaluating the factual correctness of large language models (LLMs) is vital for many applications. But are our evaluation tools themselves trustworthy? Despite the rise of factuality-based metrics, their sensitivity and reliability remain underexplored. This paper introduces a meta-evaluation framework that systematically tests these metrics using controlled corruptions of gold standard answers. Our method generates ranked outputs with known degrees of degradation to probe how metrics capture nuanced changes in truthfulness. Our experiments reveal that pipeline-based methods, such as the RAGAS's factual correctness metric, better track degradation than LLM-as-judge approaches. We also propose a new variant of the factual correctness metric that provides a competitive and cost-efficient.
    
[^27]: 不要数编辑数量，仅凭结果评判：基于奖励的语法错误纠正评估方法

    Don't Count the Edits, Judge by the Outcome Alone: Reward-Based Evaluation for Grammatical Error Correction

    [https://arxiv.org/abs/2609.15559](https://arxiv.org/abs/2609.15559)

    提出了SURE——一个以源文本为条件的基于奖励的语法纠错评估器，通过联合学习整体奖励、语法性/忠实性/流畅性的标准级监督以及片段级定位，在改写风格修正的评估上表现出色。

    

    语法错误纠正（GEC）的评估传统上依赖于参考答案或编辑重叠，这可能会惩罚与黄金修正不同的有效改写。无参考指标减少了这种依赖性，但评估流畅的输出是否为对源文本的有效纠正仍然具有挑战性。我们提出了SURE，一个以源文本为条件的奖励评估器，它在涵盖最小编辑和改写导向修正的源内偏好数据上进行训练。SURE共同学习整体奖励以及针对语法性、忠实性和流畅性的标准级监督，同时结合针对源侧错误解决的片段级定位。在SEEDA数据集上的实验表明，SURE与强基线相比具有竞争力，在改写风格的修正和更解耦的标准级诊断方面尤为突出。我们的代码可在 https://github.com/hayeonggg/SURE 获取。

    arXiv:2609.15559v1 Announce Type: cross  Abstract: Grammatical error correction (GEC) evaluation has traditionally relied on reference or edit overlap, which can penalize valid rewrites that differ from gold corrections. Reference-free metrics reduce this dependence, but evaluating whether a fluent output is a valid correction of the source remains challenging. We propose SURE, a source-conditioned reward evaluator trained on within-source preferences spanning minimal-edit and rewrite-oriented corrections. SURE jointly learns an overall reward with criteria-level supervision for grammaticality, faithfulness, and fluency, together with span-level grounding for source-side error resolution. Experiments on SEEDA show that SURE performs competitively against strong baselines, with particular gains on rewrite-style corrections and more disentangled criteria-level diagnostics. Our code is available at https://github.com/hayeonggg/SURE.
    
[^28]: 面向医学视觉问答的选项感知检索与任务特定视觉语言模型适配

    Option-Aware Retrieval and Task-Specific VLM Adaptation for Medical VQA

    [https://arxiv.org/abs/2609.15530](https://arxiv.org/abs/2609.15530)

    本文提出MedReason 2026医学视觉问答挑战赛方案，发现基于答案语义而非标签的选项感知检索可将纯检索准确率从20.0%大幅提升至57.5%，且归因分析表明任务特定的LoRA微调是系统达到93.5%以上准确率的主要来源。

    

    我们描述了参加MedReason 2026挑战赛的提交方案，涵盖了完全离线、容器化推理环境下的多选题（MCQ）和开放式（OE）医学视觉问答（VQA）任务。我们的第一个发现是，MCQ检索必须比较答案的语义而非答案标签：由于标签是针对每个问题独立分配的，复制检索到的近邻样本的标签无法传递任何有用信息；而将当前每个选项的文本与相似训练案例的正确答案文本进行比对评分，可使纯检索方式的准确率在200个排除检索样本的开发集留出数据上从20.0%提升至57.5%。我们的第二个发现对提交系统的准确率进行了归因分析：在保持任务特定的MCQ低秩适配器（LoRA）固定不变的情况下，改变提示中检索示例的数量k，准确率变化最多仅一个案例——在k=0和适配器训练时的k=1两种设置下均为187/200（93.5%），在k为另一设置时达到188/200（94.0%）。

    arXiv:2609.15530v1 Announce Type: new  Abstract: We describe our submission to the MedReason 2026 challenge, covering multiple-choice (MCQ) and open-ended (OE) medical visual question answering (VQA) under fully offline, containerized inference. Our first finding is that MCQ retrieval must compare answer \emph{semantics} rather than answer labels: labels are independently assigned per question, so copying a retrieved neighbor's label transfers no useful information, whereas scoring each current option's text against correct-answer text from similar training cases raises retrieval-only accuracy from 20.0\% to 57.5\% on a 200-case retrieval-excluded development holdout. Our second finding attributes the submitted system's accuracy: holding the task-specific MCQ Low-Rank Adaptation (LoRA) adapter fixed and varying the number \(k\) of in-prompt retrieved examples changes accuracy by at most one case --- 187/200 (93.5\%) at both \(k=0\) and the adapter's training-time \(k=1\), 188/200 (94.0
    
[^29]: 每种语言专属分词器：面向高效多语言大语言模型的模块化分词器

    To Each Language Its Tokenizer: Modular Tokenizers for Efficient Multilingual LLMs

    [https://arxiv.org/abs/2609.15528](https://arxiv.org/abs/2609.15528)

    提出模块化BPE和Unigram分词器框架，可为任意语言子集提取压缩率媲美单语分词器的子分词器，并通过将预测限制在相关词表子集的预训练策略，实现多语言大语言模型的高效训练。

    

    多语言大语言模型（LLM）传统上依赖于所有支持语言共享的单一词表，这可能导致各语言之间的压缩率不均衡。此外，其庞大的嵌入矩阵和输出矩阵会增加内存占用并降低推理速度，对小规模模型而言尤为明显。这也造成资源浪费，因为模型通常只用于部分语言。为解决这些问题，我们引入了一个用于多语言模型训练的模块化框架。首先，我们提出了学习大型模块化BPE和Unigram分词器的方法，能够提取针对任意语言子集定制的子分词器。这些子分词器实现了与单语分词器相当的压缩率，并提高了跨语言公平性。其次，我们设计了一种预训练策略，通过采样子分词器来构成训练批次，将预测限制在相关的词表子集内，从而在拥有大词表的情况下仍能实现高效训练。这支持了高效的……

    arXiv:2609.15528v1 Announce Type: new  Abstract: Multilingual Large Language Models (LLMs) traditionally rely on a single vocabulary shared by all supported languages, which can lead to uneven compression across them. Moreover, their large embedding and output matrices increase memory usage and slow inference, notably for small-scale models. It is also wasteful as models are often used for only a subset of languages. To address these issues, we introduce a modular framework for multilingual model training. First, we propose methods to learn large modular BPE and Unigram tokenizers that enable extraction of subtokenizers tailored to any language subset. These subtokenizers achieve compression on par with monolingual tokenizers and improve cross-lingual fairness. Second, we design a pretraining strategy that samples subtokenizers to form batches, restricting predictions to the relevant vocabulary subset and allowing efficient training despite a large vocabulary. This supports efficient i
    
[^30]: 精神病涉及连贯言语中的信息压缩缺陷

    Psychosis involves a deficit of information compression in connected speech

    [https://arxiv.org/abs/2609.15522](https://arxiv.org/abs/2609.15522)

    该研究利用大型语言模型的表征指标和预测误差，发现精神病患者的连贯言语存在一般性的信息压缩缺陷，表现为表征内在维度降低与惊异度升高，且与语法组织能力受损相关。

    

    在语言任务上具有类人表现的大型语言模型（LLM）已经改变了神经多样性条件下语言研究的面貌。LLM以高维向量（嵌入）的形式提供语言输入的表征，并基于这些嵌入计算下一个词的预测。先前的跨语言证据表明，精神病中存在一种复杂性降低的现象，具体表现为LLM表征的内在维度（ID）较低，同时平均惊异度（预测误差）较高。我们假设这些指标反映了精神病中一种一般性的信息压缩缺陷，该缺陷与语法组织相关联，而语法组织正是使语言预测成为可能的基础。我们将惊异度差异操作化为基于词频估计的惊异度与基于上下文语言模型计算的惊异度之间的差值，后者对超越词汇概念的语法组织敏感。利用一个包含144名土耳其语使用者的数据集，我们对这一假设进行了检验。

    arXiv:2609.15522v1 Announce Type: new  Abstract: Large language models (LLMs) with human-like performance on linguistic tasks have transformed the study of language in neurodiverse conditions. LLMs provide representations of linguistic input in the form of high-dimensional vectors (embeddings), and next-token predictions computed from these embeddings. Previous crosslinguistic evidence suggests a complexity reduction in the form of both lower intrinsic dimensionality (ID) of LLM representations and higher mean surprisal (prediction error) in psychosis. We hypothesized that these metrics reflect a general deficit of information compression in psychosis, linked to grammatical organization as what enables predictions in language.We operationalized surprisal difference as the difference between surprisal as estimated from word frequency and surprisal as based on a contextual LM, which is sensitive to grammatical organization over and above lexical concepts. Using a dataset of 144 Turkish s
    
[^31]: 超越安全答案：面向大型推理模型推理安全的分段感知列表式对齐方法

    Beyond Safe Answers: Segment-Aware Listwise Alignment for Reasoning Safety in Large Reasoning Models

    [https://arxiv.org/abs/2609.15517](https://arxiv.org/abs/2609.15517)

    提出SaLT-DPO方法，通过分段感知的列表式对齐、基于“最弱环节”原则的安全一致性正则化以及良性提示效用锚定三种机制，同时保障大型推理模型的推理过程与最终答案两个层面的安全性。

    

    大型推理模型（LRMs）带来了双重层面的安全挑战：中间推理过程和最终答案都可能包含有害内容。现有的对齐方法通常在整条响应层面进行操作，使得不安全的推理过程可以被看似安全的最终答案所掩盖。我们提出了分段感知列表式目标DPO（SaLT-DPO），通过三种机制弥补这一空白：（1）分段感知的列表式对齐，将响应分解为推理段和答案段，独立评估每段的安全性，并将长度归一化的分段奖励与多个候选答案的软目标分布进行对齐；（2）联合安全一致性正则化，应用“最弱环节”原则来促进推理段与答案段之间的安全一致性；（3）在良性提示上进行效用锚定，以缓解过度拒绝和推理能力退化问题。在三个大型推理模型上的实验表明，SaLT-DPO能够持续降低不安全率。

    arXiv:2609.15517v1 Announce Type: new  Abstract: Large Reasoning Models (LRMs) pose a dual-surface safety challenge: both intermediate reasoning traces and final answers can contain harmful content. Existing alignment methods often operate at the whole-response level, allowing unsafe reasoning to be masked by a safe-looking final answer. We propose Segment-aware Listwise Target DPO (SaLT-DPO), which addresses this gap through three mechanisms: (1) segment-aware listwise alignment that decomposes responses into reasoning and answer segments, independently scores each segment's safety, and aligns length-normalized segment rewards with soft target distributions over multiple candidates; (2) joint safety coherence regularization that applies a weakest-link principle to promote safety consistency across both segments; and (3) utility anchoring on benign prompts to mitigate over-refusal and reasoning degradation. Experiments on three LRMs show that SaLT-DPO consistently reduces unsafe rates 
    
[^32]: AI诗歌的作者归属与美学评价：以俳句为例的案例研究

    Authorship attribution and aesthetic evaluation of AI poetry: a case study with Haiku

    [https://arxiv.org/abs/2609.15511](https://arxiv.org/abs/2609.15511)

    本研究让日本大学生区分AI与人类创作的俳句，发现不同大语言模型的“AI痕迹”可辨识度差异显著，其中GPT-5、Gemini 2.5和StableLM-7B生成的俳句已难以与人类作品区分。

    

    本文研究了当代大语言模型（LLM）生成日语俳句及人工评估的问题，重点关注受限诗歌形式下的作者感知与审美判断。研究采用少样本提示策略，在一系列异构大语言模型上生成日语俳句，涵盖开源与闭源系统、中型与大型架构、原生或适配日语支持的模型，以及多语言专有模型。这些AI生成的俳句与人类创作的俳句混合后，以问卷形式分发给位于东京的日本大学生。调查评估了受访者能否区分AI生成与人类创作的俳句，以及哪些线索影响了他们的判断。识别准确率因模型而异：GPT-5、Gemini 2.5和StableLM-7B的表现接近随机水平（约0.50），而……

    arXiv:2609.15511v1 Announce Type: cross  Abstract: This paper investigates the generation and human evaluation of Japanese haiku by contemporary Large Language Models (LLMs), focusing on authorship perception and aesthetic judgment within a constrained poetic form. Using a few-shot prompting strategy, Japanese haiku were generated across a heterogeneous set of large language models, including open- and closed-source systems, medium-scale and large-scale architectures, models with native or adapted Japanese support, and multilingual proprietary models. These AI-generated haiku were combined with human-written ones and presented in a questionnaire distributed to students at Japanese universities in Tokyo. The survey assessed whether respondents could distinguish between AI-generated and human-written haiku and which cues informed their judgments. Recognition accuracy varied across models. GPT-5, Gemini 2.5, and StableLM-7B performed at approximately chance level (approx 0.50), whereas LL
    
[^33]: 无损推测解码究竟有多无损？数值精度在 Orthrus 中的作用

    How Lossless Is Lossless Speculative Decoding? The Role of Numerical Precision in Orthrus

    [https://arxiv.org/abs/2609.15504](https://arxiv.org/abs/2609.15504)

    该论文独立复现了 Orthrus 并发现其在 BF16 精度下仅有约 43-45% 的输出轨迹与自回归模型完全一致，表明其“无损推测解码”声明强烈依赖于数值精度，尽管下游基准测试未显示系统性退化。

    

    Orthrus 是一种混合自回归-扩散架构，它通过在使用冻结自回归骨干网络的同时并行生成多个 token 来加速自回归语言模型的推理。其核心主张是，模型内共识机制能够实现无损推测解码，产生与自回归模型完全相同的输出序列。我们独立复现了 Orthrus，并在不同数值精度下检验了这一主张。在 BF16 推理下，对于作者的检查点，精确轨迹匹配仅发生在 45% 的情况下；对于我们独立训练的模型，在来自 12 个领域的 1,190 个提示上，精确匹配率为 43%。精确匹配的概率还与参考模型的响应条件困惑度密切相关。尽管存在这种轨迹差异，Orthrus 在下游 lm-eval-harness 基准测试中并未表现出系统性退化。相比之下，重复轨迹评估

    arXiv:2609.15504v1 Announce Type: cross  Abstract: Orthrus is a hybrid autoregressive-diffusion architecture that accelerates autoregressive language-model inference by generating multiple tokens in parallel while using a frozen autoregressive backbone. Its central claim is that an intra-model consensus mechanism enables lossless speculative decoding, producing the same output sequence as the autoregressive model.   We independently reproduce Orthrus and examine this claim under different numerical precisions. Under BF16 inference, exact trajectory matching occurs in only 45% of cases for the authors' checkpoint and 43% for our independently trained model across 1,190 prompts from 12 domains. The probability of exact matching is also strongly associated with the response-conditional perplexity of the reference model. Despite this trajectory divergence, Orthrus does not show systematic degradation on downstream lm-eval-harness benchmarks. In contrast, repeating the trajectory evaluation
    
[^34]: 温度脆弱性与截断采样的条件性收益

    Temperature Fragility and the Conditional Benefits of Truncation Sampling

    [https://arxiv.org/abs/2609.15476](https://arxiv.org/abs/2609.15476)

    该研究首次在部署系统常用的默认温度（0.6–1.0）下系统评估截断采样，发现大模型准确率对温度高度脆弱——部分模型温度从0.7升至1.3会在MMLU-Pro上损失17–38个点，而top-p/min-p等截断采样方法仅在高温下才有明显收益。

    

    大型语言模型通过从预测分布中逐词元采样来生成文本，温度参数决定了采样结果偏离最可能词元的程度。截断采样器（如top-p和min-p）在采样前丢弃概率最低的词元，使高温采样仍能保持连贯性。它们被报告的准确率提升来自1.5到3的温度区间，而实际部署系统的默认温度集中在0.6到1.0之间。截断采样在这些默认温度下是否改变准确率、以及对哪些模型有效，此前尚未被测量。我们在同一受控流水线中，在0.7、1.0和1.3三个温度下测试了十三个开源权重模型在GSM8K和MMLU-Pro上的表现，其中十个模型还在八种解码配置下进行了测试。十三个模型中有六个在MMLU-Pro上从温度0.7到1.3损失了17到38个准确率点，其余七个最多损失10个点。损失的准确率来自那些运行到词元上限或从未给出答案的生成结果。这些……

    arXiv:2609.15476v1 Announce Type: new  Abstract: Large language models generate text by sampling each token from a predicted distribution, and a temperature parameter sets how far the draw strays from the most probable tokens. Truncation samplers such as top-p and min-p discard the least probable tokens before the draw, so that sampling at high temperature stays coherent. Their reported accuracy gains come from temperatures of 1.5 to 3, while the defaults of deployed systems cluster between 0.6 and 1.0. Whether they change accuracy at those defaults, and for which models, has not been measured. We test thirteen open-weight models on GSM8K and MMLU-Pro at temperatures 0.7, 1.0, and 1.3 in one controlled pipeline, ten of them under eight decoding configurations. Six of the thirteen models lose 17 to 38 accuracy points on MMLU-Pro between 0.7 and 1.3, and the other seven lose at most 10. The lost accuracy comes from generations that run to the token limit or never state an answer. These r
    
[^35]: 土耳其语MMLU Pro：可追溯选项增强及其在土耳其语多项选择评估中的有效性局限

    Turkish MMLU Pro: Traceable Option Augmentation and Its Validity Limits in Turkish Multiple-Choice Evaluation

    [https://arxiv.org/abs/2609.15467](https://arxiv.org/abs/2609.15467)

    该研究构建了土耳其语MMLU Pro基准，通过从同章节其他题目复制五个可追溯的干扰选项将选项从五个扩展到十个，发现增加选项会显著降低大语言模型得分但不提升评估有效性，其中准确率下降主要源于模型误选借来选项，且在否定式题干上下降尤为严重。

    

    增加答案选项可以降低多项选择题的分数，但并不能提高评估的有效性。土耳其语MMLU Pro使用涵盖58个学科领域的12,000道土耳其语原生题目对这一区别进行了检验。每道题目保留其题干、五个原始选项和来源答案键，并从同一学科的其他题目中复制五个选项作为新增选项。系统采用句子嵌入检索提出候选选项，再由语言模型选择已有的选项标识符。通过确定性验证重建了全部60,000个新增选项。对25个模型的校准揭示了评分方式和生成预算的影响。五次评估产生的来源答案键准确率介于34.8%至81.4%之间。在981道共同题目上，一个通过API提供的模型从五个选项时的93.7%下降到十个选项时的83.1%；在115个丢失的正确回答中，有102个选择了借来的选项。在启发式标记为否定式题干的问题上，下降幅度为24.4个百分点，而在其他题目上为5.9个百分点。此外，研究还对200道抽样题目完成了经人工核查的审计。

    arXiv:2609.15467v1 Announce Type: cross  Abstract: Adding answer options can lower multiple-choice scores without improving assessment validity. Turkish MMLU Pro examines this distinction using 12,000 Turkish-source questions across 58 sections. Each question retains its stem, five original options and source key, and receives five options copied from other questions in the same section. Sentence-embedding retrieval proposes candidates; a language model selects existing identifiers. Deterministic verification reconstructs all 60,000 additions. A 25-model calibration exposes scoring and generation-budget effects. Five evaluations produce source-key accuracies of 34.8%-81.4%. On 981 shared questions, one API-served model falls from 93.7% with five choices to 83.1% with ten; 102 of 115 lost correct responses select borrowed options. The decrease is 24.4 percentage points on heuristically flagged negative stems and 5.9 points elsewhere. A completed human-checked audit of 200 sampled questi
    
[^36]: MarKey：边际效用引导的贪心关键帧选择方法用于长视频理解

    MarKey: Marginal Utility Guided Greedy Keyframe Selection for Long Video Understanding

    [https://arxiv.org/abs/2609.15408](https://arxiv.org/abs/2609.15408)

    MarKey提出了一种免训练的子集感知贪心优化框架，通过联合评估查询相关性、边际覆盖增益和上下文相关冗余性来为长视频理解选择关键帧，从而避免冗余选择并实现更完整的证据覆盖。

    

    长视频理解对多模态大语言模型（MLLMs）而言仍然具有挑战性，因为密集编码长帧序列的计算成本高昂，而在有限视觉预算下的均匀采样可能会遗漏稀疏但具有决定性的证据。近期的免训练关键帧选择方法实现了更高效的推理，并带来了可观的性能提升。然而，许多现有方法在很大程度上孤立地对帧进行评分，没有显式考虑每个候选帧如何补充当前已选择的子集，这可能导致冗余选择和不完整的证据覆盖。为了解决这一局限，我们提出了MarKey，一个将关键帧选择形式化为子集感知贪心优化的免训练框架。在每次迭代中，MarKey使用一个易处理的代理函数对每个候选帧进行评分，该函数共同考虑了查询相关性、边际覆盖增益和上下文相关的冗余性。

    arXiv:2609.15408v1 Announce Type: cross  Abstract: Long-video understanding remains challenging for multimodal large language models (MLLMs) because densely encoding long frame sequences is computationally expensive, while uniform sampling under a limited visual budget can miss sparse yet decisive evidence. Recent training-free keyframe selection methods have enabled more efficient inference and yielded promising performance gains. However, many existing methods score frames largely in isolation without explicitly considering how each candidate complements the currently selected subset, potentially resulting in redundant selections and incomplete evidence coverage. To address this limitation, we propose MarKey, a training-free framework that formulates keyframe selection as subset-aware greedy optimization. At each iteration, MarKey scores each candidate using a tractable surrogate that jointly accounts for query relevance, marginal coverage gain, and context-dependent redundancy, and 
    
[^37]: SlopShape：识别AI生成的商业网络内容

    SlopShape: Identifying AI-Generated Commercial Web Content

    [https://arxiv.org/abs/2609.15369](https://arxiv.org/abs/2609.15369)

    该研究提出通过结构特征（信息呈现方式、顺序、证据与语气）而非词级特征来识别商业网页中的AI生成内容，仅用187个结构特征就在模型自我改写的对抗条件下仍保持约98%的检测性能。

    

    词级检测器几乎能完美识别未经编辑的AI生成文本，但已有文献记录了它们在文本改写后的脆弱性，且词级评分既无法刻画文本特征，也无法识别出自哪个AI模型。我们探讨能否在更深一层——从结构特征上来识别AI生成的文本：信息如何呈现、以何种顺序、使用什么证据、采用什么语气。我们将StoryScope（Russell等人，2026）在AI生成小说中揭示的此类模式复制到商业内容上：以268个公司域名的2,250篇ChatGPT问世前的人类博客文章，对比来自五个前沿模型的11,250篇AI镜像文本。一个包含214个特征的测量工具由LLM应用，并通过人工黄金标注环节验证（人与人kappa系数0.928，人与模型0.946），仅凭其中187个结构特征，就在留出的公司域名上以98.0宏F1检测出AI生成的文章，且当每篇AI文章被其自身模型改写时，性能保持不变（98.1）。

    arXiv:2609.15369v1 Announce Type: new  Abstract: Word-level detectors identify unedited AI-generated text almost perfectly, but the literature documents their brittleness under rewording, and a word-level score neither characterizes a text nor identifies which AI model wrote it. We ask whether AI-generated text can be identified one level deeper, from structural signatures: how information is presented, in what order, with what evidence, and in what voice. We replicate StoryScope (Russell et al., 2026), which showed such patterns for AI-generated fiction, on commercial content: 2,250 pre-ChatGPT human blog posts from 268 company domains against 11,250 AI mirrors from five frontier models. A 214-feature instrument, applied by an LLM and validated in a human gold-annotation session (human-human kappa 0.928, human-model 0.946), detects AI posts from its 187 structural features alone at 98.0 macro-F1 on held-out companies, unchanged (98.1) when every AI post is reworded by its own model. T
    
[^38]: RSIAgent：在新环境中实现递归自我改进的自主探索

    RSIAgent: Autonomous Exploration for Recursive Self-improvement in New Environments

    [https://arxiv.org/abs/2609.15364](https://arxiv.org/abs/2609.15364)

    RSIAgent是一个无需训练的多智能体框架，通过“先广后深”的自主探索策略构建可复用的冻结记忆，使数字智能体在新环境中实现递归自我改进，且无需更新模型参数。

    

    数字智能体必须经常适应新的环境，而这些环境的界面、工具和失败模式并未被预训练模型完全涵盖。我们提出了RSIAgent，这是一个无需训练的多智能体框架，通过自主记忆构建实现递归自我改进。RSIAgent协调课程智能体、执行智能体和验证智能体，持续探索环境、验证结果，并保留环境特定的知识，包括行动、条件与后果之间可复用的因果关系。它进一步采用“先广后深”的探索策略，将并行的广泛递归自我探索（用于发现多样的环境结构）与聚焦的深度自我探索（用于发现困难案例、隐藏约束、边界条件以及此前未知的因果依赖）相结合。所构建的记忆被冻结后，可直接复用于下游任务，而无需更新模型参数。

    arXiv:2609.15364v1 Announce Type: new  Abstract: Digital agents must often adapt to new environments whose interfaces, tools, and failure modes are not fully captured by pretrained models. We introduce \textbf{RSIAgent}, a training-free multi-agent framework for recursive self-improvement through autonomous memory construction. RSIAgent coordinates curriculum, actor, and verifier agents to continually explore the environment, validate outcomes, and retain environment-specific knowledge, including reusable causal relationships between actions, conditions, and consequences. It further adopts a \textbf{broad-then-deep} exploration strategy, combining parallel broad recursive self-exploration for discovering diverse environment structures with focused deep self-exploration for uncovering hard cases, hidden constraints, boundary conditions, and previously unknown causal dependencies. The resulting memory is frozen and can be directly reused for downstream tasks without updating model parame
    
[^39]: 预训练语言模型在时间序列预测中的参数高效适配

    Parameter-Efficient Adaptation of Pretrained Language Models for Time-Series Forecasting

    [https://arxiv.org/abs/2609.15344](https://arxiv.org/abs/2609.15344)

    该论文提出一种参数高效的迁移学习框架，将固定长度时间序列补丁直接投影到冻结的GPT-2嵌入空间中，以绕过文本标记化的方式将语言模型作为通用序列编码器，并通过七个基准数据集上的系统性消融实验，揭示了表示策略、适配方式和架构组件等设计选择如何影响语言模型向时间序列预测的有效跨模态迁移。

    

    我们研究了通过参数高效的迁移学习框架，将预训练语言模型适配到单变量时间序列预测任务，目标是理解哪些设计选择能够驱动有效的跨模态迁移。虽然语言模型处理的是离散的文本标记，但时间序列由具有时间依赖性的连续数值观测组成。为了弥合这一模态差距，我们将固定长度的时间序列补丁直接投影到预训练GPT-2主干的嵌入空间中，绕过文本标记化过程，将Transformer视为通用序列编码器。通过对涵盖能源、天气、交通和金融领域的七个基准数据集进行受控消融研究，我们分析了以下因素的影响：(i) 表示策略（连续嵌入与文本序列化）、(ii) 适配方式（冻结主干与部分或全部微调）、(iii) 架构组件，例如……（摘要内容在此处不完整）

    arXiv:2609.15344v1 Announce Type: new  Abstract: We study the adaptation of pretrained language models to univariate time-series forecasting through a parameter-efficient transfer learning framework, with the goal of understanding which design choices drive effective cross-modal transfer. While language models operate on discrete textual tokens, time series consist of continuous numerical observations with temporal dependencies. To bridge this modality gap, we project fixed-length time-series patches directly into the embedding space of a pretrained GPT-2 backbone, bypassing textual tokenization and treating the Transformer as a generic sequence encoder. Through controlled ablation studies on seven benchmark datasets spanning energy, weather, traffic, and finance, we analyze the effects of (i)~representation strategy (continuous embeddings versus textual serialisation), (ii)~adaptation regime (frozen backbone versus partial or full fine-tuning), (iii)~architectural components such as a
    
[^40]: 面向大语言模型高效潜在空间推理的动态语义压缩

    Dynamic Semantic Compression for Efficient Latent-Space Inference in Large Language Models

    [https://arxiv.org/abs/2609.15338](https://arxiv.org/abs/2609.15338)

    提出DSEI框架，通过动态语义自编码器将段级语义自适应压缩为紧凑潜在表示，使大语言模型能够在潜在空间中进行段级推理，在降低48%困惑度的同时显著提升推理效率。

    

    大语言模型主要在token级别执行推理，导致大量的内存开销并损害了计算效率。本文提出了一种动态语义提取与推理框架，通过两阶段训练策略在潜在空间内实现段级别的推理。首先，我们通过自监督学习构建了动态语义自编码器。DSAE能够动态提取段级别语义，并通过自适应语义加权和门控融合将其压缩为紧凑的潜在表示。随后，我们将DSAE集成到大语言模型架构中，并训练模型在密集潜在空间上进行推理。DSEI显著减少了输入和生成序列的长度，并大幅提升了推理效率。在万卷数据集上进行的大量实验表明，与静态句子级压缩方法相比，DSEI将困惑度降低了48%。

    arXiv:2609.15338v1 Announce Type: cross  Abstract: Large Language Models (LLMs) primarily perform inference at the token level, resulting in substantial memory overhead and compromised computational efficiency. In this paper, we propose a Dynamic Semantic Extraction and Inference (DSEI) framework, which achieves segment-level inference within the latent space through a two-stage training strategy. First, we construct a Dynamic Semantic Autoencoder (DSAE) via self-supervised learning. DSAE dynamically extracts segment-level semantics and compresses them into compact latent representations via adaptive semantic weighting and gated fusion. Subsequently, we integrate the DSAE into the LLM architecture and train the model to infer over dense latent space. DSEI substantially reduces both input and generation sequences and significantly enhances inference efficiency. Extensive experiments conducted on the Wanjuan dataset demonstrate that DSEI reduces perplexity by 48% compared to static sente
    
[^41]: 《漂亮的分数、被掩埋的证据与自信的错误：基于凭证的前沿智能体问答审计》

    Clean Scores, Buried Evidence, and Confident Wrong: A Receipt-Based Audit of Frontier Agentic QA

    [https://arxiv.org/abs/2609.15319](https://arxiv.org/abs/2609.15319)

    前沿智能体在证据被掩埋时准确率显著下降、成本上升，且高置信度无法暴露错误，作者提出以“声明级凭证”溯源、条件感知评分和人类对抗性验证为核心的审计式评估框架，以取代排行榜式评测。

    

    前沿模型在浅层文档/图表阅读任务上表现良好。在一项受控的“资料室”审计中，将证据移入“被掩埋”条件会降低准确率、增加强制声明、增加工具调用次数，并推高每个正确答案的成本。置信度与基准校准并不能完全捕捉错误答案；一起有记录的生产事故表明，捏造的结构性声明可能与准确的数字表格混在一起。智能体评估需要“声明级凭证”（语句级别的溯源，而非答案级别的分数）、条件感知评分以及人类对抗性验证——这是一门审计学科，而不是一张排行榜。我们测量的场景是财务尽职调查；我们接下来正在构建的场景是国防参谋工作，其中同样会出现“隐藏证据”的形态。在这两种场景中，模型都不是后果的承担方，签字的人才是。简而言之：在我们考察的记录案例中，智能体可能将……

    arXiv:2609.15319v1 Announce Type: cross  Abstract: Frontier models score well on shallow document/chart reading tasks. In a controlled data-room audit, moving evidence into buried conditions reduced accuracy, increased forced declarations, increased tool calls, and increased cost per correct answer. Confidence and benchmark calibration did not fully capture wrong answers; a documented production incident shows fabricated structural claims can be mixed with accurate numeric tables. Agentic evaluations need claim-level receipts (statement-level provenance, not answer-level scores), condition-aware scoring, and human-adversarial verification - an auditing discipline, not a leaderboard. The setting we measure is financial due diligence; the setting we are building toward next is defense staff work, where the same buried-evidence shape appears. In both, the model is not a party to the consequences; the person who signs is. In plain terms: in the documented cases we examine, agents can pair 
    
[^42]: 通过联合输出在策略蒸馏减少语音语言模型中的输出模式差距

    Reducing the Output-Mode Gap in Speech Language Models via Joint-Output On-Policy Distillation

    [https://arxiv.org/abs/2609.15313](https://arxiv.org/abs/2609.15313)

    该论文发现了语音语言模型中语音转文本语音（S2TS）模式的内部文本生成准确率显著低于语音转文本（S2T）模式的“输出模式差距”问题，并提出联合输出在策略蒸馏（JO-OPD）方法，利用学生生成的S2TS轨迹将更强的S2T策略蒸馏到联合生成中，有效缩小了这一差距。

    

    自回归生成交错的文本和声学token是语音大语言模型中生成口语响应的常见方法。尽管这种设计能够实现带有显式文本引导的流式生成，但生成的声学token会成为后续文本预测的上下文的一部分。在相同的语音输入下，我们观察到语音转文本和语音（S2TS）模式下生成的内部文本的答案准确率明显低于语音转文本（S2T）模式响应的准确率。我们将这种差异称为“输出模式差距”（OMG）。为了减少OMG，我们提出了“联合输出在策略蒸馏”（JO-OPD），该方法利用学生生成的S2TS轨迹，将模型更强的S2T策略蒸馏到联合生成中。在每个文本位置，S2T教师根据学生先前输出的纯文本投影提供软目标，而学生则从相应的完整交错序列中进行预测。

    arXiv:2609.15313v1 Announce Type: cross  Abstract: Autoregressive generation of interleaved text and acoustic tokens is a common approach to spoken-response generation in speech large language models. Although this design enables streaming generation with explicit textual guidance, generated acoustic tokens become part of the context for subsequent text predictions. Given identical speech inputs, we observe markedly lower answer accuracy for the internal text generated in speech-to-text-and-speech (S2TS) mode than for speech-to-text (S2T) responses. We term this discrepancy the \emph{output-mode gap} (OMG). To reduce OMG, we propose \emph{Joint-Output On-Policy Distillation} (JO-OPD), which distills the model's stronger S2T policy into joint generation using student-generated S2TS trajectories. At each text position, the S2T teacher provides soft targets from a text-only projection of the student's preceding outputs, while the student predicts from the corresponding full interleaved hi
    
[^43]: 当智能体慢下来：通过Elo-per-token分析理解LLM智能体的测试时策略

    When Agents Slow Down: Understanding LLM Agents' Test-Time Strategies via Elo-per-token Analysis

    [https://arxiv.org/abs/2609.15309](https://arxiv.org/abs/2609.15309)

    该论文提出Elo-per-token分析方法，通过追踪不同token预算下的最佳解决方案并利用Bradley-Terry模型将任务内排序聚合为跨任务Elo评分，从而系统性地衡量LLM智能体的测试时计算扩展性能。

    

    大语言模型（LLM）智能体在修正解决方案、使用工具、探索替代方案以及决定何时停止的过程中，会自适应地分配测试时计算资源。这种测试时策略使得衡量智能体性能的扩展规律变得困难。我们研究了为中间提交提供连续分数的开放式任务，使得长轨迹中的进展全程可被观测。我们提出了Elo-per-token分析方法，该方法追踪在每个token预算下找到的最佳解决方案，并使用Bradley-Terry模型将任务内的排序聚合为跨不同分数尺度任务的Elo评分。我们将该方法应用于四个通用智能体在四个开放式基准上的测试（会话长达1亿token），并在受控单任务干预下应用于三个反馈驱动的LLM优化框架。独立采样提供了一个具有理论刻画的参照基线，其Elo评分随对数计算量线性增长。以此为参照，[摘要在此处被截断]

    arXiv:2609.15309v1 Announce Type: new  Abstract: Large language model (LLM) agents allocate test-time compute adaptively as they revise solutions, use tools, explore alternatives, and decide when to stop. This test-time strategy makes it difficult to measure how agent performance scales. We study open-ended tasks that provide continuous scores for intermediate submissions, making progress observable throughout long trajectories. We propose Elo-per-token analysis, which tracks the best solution found at each token budget and uses a Bradley-Terry model to aggregate within-task orderings into Elo ratings across tasks with different score scales. We apply it to four general-purpose agents on four open-ended benchmarks, with sessions of up to 100M tokens, and to three feedback-driven LLM optimization harnesses in controlled single-task interventions. Independent sampling provides a theoretically characterized reference, for which Elo grows linearly with log compute. Against this reference, 
    
[^44]: 推理重要之事：面向通用多模态嵌入的检索支撑推理

    Reason What Matters: Retrieval-Grounded Reasoning for Universal Multimodal Embeddings

    [https://arxiv.org/abs/2609.15296](https://arxiv.org/abs/2609.15296)

    该论文提出ReWAM框架，利用检索反馈来指导CoT推理中的信用分配与推理计算量，解决了GRPO对所有token赋予相同优势以及完整CoT生成引入高延迟的问题，从而实现更高效的通用多模态嵌入。

    

    通用多模态嵌入（UME）学习跨模态的统一表示，使单一模型能够支持多样化的检索任务。最近的方法使用思维链（CoT）推理来更好地解释多模态输入，然后为复杂检索任务生成嵌入，并通过基于检索奖励的GRPO进一步优化这一推理过程。然而，两个局限性阻碍了语料库规模的部署：GRPO为所有CoT token分配相同的优势值，而未能识别输入支持的主张或区分正负样本的证据；此外，在每次生成嵌入之前生成完整的CoT会引入大量延迟，即使部分推理轨迹已能提供足够的检索证据。为解决这些局限性，我们提出了Reason What Matters（ReWAM），一个以检索为根基的推理框架，利用检索反馈来同时指导信用分配和推理计算。

    arXiv:2609.15296v1 Announce Type: new  Abstract: Universal multimodal embedding (UME) learns unified representations across modalities, enabling a single model to support diverse retrieval tasks. Recent methods use Chain-of-Thought (CoT) reasoning to better interpret multimodal inputs before generating embeddings for complex retrieval tasks and further optimize this reasoning process through GRPO with retrieval-based rewards. However, two limitations hinder corpus-scale deployment. GRPO assigns all CoT tokens the same advantage, without identifying input-supported claims or evidence that distinguishes the positive from negatives. Moreover, generating a complete CoT before each embedding introduces substantial latency, even when a partial trace already provides sufficient retrieval evidence. To address these limitations, we propose Reason What Matters (ReWAM), a retrieval-grounded reasoning framework that uses retrieval feedback to guide both credit assignment and reasoning computation.
    
[^45]: 人工创业认知：在大语言模型（LLM）内部定位并因果调控机会识别“刻度盘”

    Artificial entrepreneurial cognition: Locating and causally steering an opportunity recognition dial inside large language models (LLMs)

    [https://arxiv.org/abs/2609.15277](https://arxiv.org/abs/2609.15277)

    本文提出“人工创业认知”概念，首次将机制可解释性引入创业研究，在大语言模型内部定位出机会识别的表征方向，并通过直接干预该“机会识别刻度盘”实现对模型机会判断的因果调控。

    

    创业认知是创业研究的基石。然而，大语言模型（LLM）在创业工作中日益深入的参与，将认知问题从人类行为者扩展到了其内部表征在很大程度上仍未被探索的智能系统。我们提出“人工创业认知”这一概念，即人工智能（AI）系统内部与创业相关的表征和计算的功能性组织方式。我们通过表征工程将机制可解释性引入创业研究。聚焦于机会识别这一构念，我们构建了636对匹配的“含机会识别”与“不含机会识别”场景对，并在Llama 3.1 8B-Instruct模型中恢复出一个机会识别方向。我们并非从模型输出中推断这一构念，而是直接对该方向进行干预，沿我们所说的“机会识别刻度盘”上下调节模型，模型的机会判断也随之发生相应变化。

    arXiv:2609.15277v1 Announce Type: new  Abstract: Entrepreneurial cognition is a foundation of entrepreneurship research. Yet the growing involvement of large language models (LLMs) in entrepreneurial work extends the cognition question beyond human actors to systems whose internal representations remain largely unexplored. We introduce artificial entrepreneurial cognition, the functional organisation of entrepreneurship-relevant representations and computations inside artificial intelligence (AI) systems. We bring mechanistic interpretability into entrepreneurship research through representation engineering. Focusing on opportunity recognition (OR), we construct 636 matched OR-present and OR-absent scenario pairs and recover an OR direction in Llama 3.1 8B-Instruct. Rather than infer the construct from outputs, we intervene directly on this direction, steering the model up and down along what we call the opportunity recognition dial, and its opportunity judgments shift with it. To our 
    
[^46]: 符号学关系与证明方法：基于大语言模型的论证结构跨体裁研究

    Semiotic Relations and Proof Methods: A Cross-Genre Study of Argument Structure with Large Language Models

    [https://arxiv.org/abs/2609.15194](https://arxiv.org/abs/2609.15194)

    本文提出受符号学四种主修辞格（转喻、隐喻、反讽、提喻）启发的四种符号学关系，将其分别对应到推理证明、类比证明、反证法和分情形证明四种证明方法，并利用大语言模型通过跨体裁实证研究考察这些论证关系的实际使用情况。

    

    当一个命题 $S$ 的直接证明看起来困难甚至无法获得时，可能存在另一个（或一组）与 $S$ 以某种方式相关联的命题 $S^{*}$，基于它可以证明 $S$。为了研究从 $S$ 转换到 $S^{*}$ 可以采用哪些方式，本文简要回顾了受符号学研究中四种主修辞格启发的四种符号学关系。具体而言，我们的组合关系、聚合关系、对立关系和部分整体关系分别对应于转喻、隐喻、反讽和提喻。我们认为这四种符号学关系决定了从 $S$ 到 $S^{*}$ 的转换选项，从而导出推理证明、类比证明、反证法和分情形证明。为了考察这四种关系在不同类型论证中的实际使用情况，我们通过一项实证研究对该框架进行了补充，将四种符号学关系转化为明确的操作性定义。

    arXiv:2609.15194v1 Announce Type: new  Abstract: When a direct proof of a statement $S$ seems hard or even impossible to obtain, there may exist another statement (or set of statements) $S^{*}$, somehow related to $S$, on the basis of which $S$ can be proved. In order to investigate what options can be used to move from $S$ to $S^{*}$, four kinds of semiotic relations inspired by the four master tropes of semiotic research are briefly reviewed. Specifically, our syntagmatic, paradigmatic, antithetic and meronymic relations correspond, respectively, to metonymy, metaphor, irony and synecdoche. It is suggested that these four semiotic relations determine the options to move from $S$ to $S^{*}$, leading to proof by inference, proof by analogy, proof by contradiction, and proof by case analysis. To examine how the four relations are actually used across different kinds of argument, we complement the framework with an empirical study. We turn the four relations into explicit operational def
    
[^47]: 是什么限制了我们？分析NLP研究中的自我报告局限性

    What Limits Us? Analyzing Self-Reported Limitations in NLP Research

    [https://arxiv.org/abs/2609.15191](https://arxiv.org/abs/2609.15191)

    本文提出了一种新颖的人机协作迭代混合定性编码框架，对2020至2025年间ACL和EMNLP论文的局限性部分进行大规模分析，揭示了NLP研究者自我报告局限性的时间趋势、与论文属性的关联以及写作模式。

    

    自2022年底以来，局限性部分已成为许多顶级NLP会议的强制要求。这些会议接收论文数量的不断增长，产生了一个庞大的自我报告局限性语料库，这些内容无法全部人工审阅，却仍然被系统性地忽视而未得到分析。因此，在本文中，我们对2020年至2025年间发表的ACL和EMNLP论文的局限性部分进行了大规模分析，以了解研究人员对其自身工作的披露情况。为此，我们实现了一种新颖的人机协作框架，用于迭代式混合定性编码。该框架使我们能够研究自我报告局限性随时间变化的趋势、其与特定论文属性之间的关联，以及围绕这些披露反复出现的写作模式。我们的发现为NLP社区中报告的多样化挑战以及研究人员的自我报告实践提供了批判性反思。

    arXiv:2609.15191v1 Announce Type: new  Abstract: Since late 2022, a Limitations section has become mandatory at many top-tier NLP conferences. The growing number of accepted papers at these venues has resulted in a vast corpus of self-reported limitations that cannot all be manually reviewed, yet remains systematically unanalyzed. Therefore, in this paper, we conduct a large-scale analysis of the Limitations sections from ACL and EMNLP papers published between 2020 and 2025 to understand what researchers disclose about their own work. To do so, we implement a novel human-AI framework for iterative hybrid qualitative coding. This framework enables us to investigate trends in self-reported limitations over time, their correlations with specific paper attributes, and the writing patterns that recur around these disclosures. Our findings offer a critical reflection on the diverse reported challenges as well as the self-reporting practices of researchers in the NLP community.
    
[^48]: MUSE：一个面向氛围叙事的理论驱动故事引擎

    MUSE: A Theory-Harnessed Story Engine for Vibe Narrativizing

    [https://arxiv.org/abs/2609.15188](https://arxiv.org/abs/2609.15188)

    提出了MUSE故事引擎，通过将罗伯特·麦基故事理论工程化为针对具体创作决策的指导，并使其贯穿规划、起草和修改全过程，实现将自然语言写作需求转化为高质量完整故事的“氛围叙事”任务。

    

    大语言模型能够生成流畅的散文。故事质量取决于关于情节、人物和语言的决策如何在规划、起草和修改的整个过程中协同运作。指导这些决策面临两个瓶颈：故事指导的质量及其持续运用。我们将“氛围叙事”定义为将自然语言写作需求转化为完整故事的任务，并提出了MUSE——一个理论驱动的故事引擎。MUSE将故事知识组织为针对特定创作决策的指导，并将这些决策贯穿于后续的创造性工作中。知识工程通过规则原子化、语义整合和机制抽象来发展罗伯特·麦基的故事理论；由此产生的指导通过单一事实来源和分层披露加以组织。典型示例补充了依赖上下文和审美判断的原则。智能体框架组织了设计、人物表演、场景构图和修改等环节。

    arXiv:2609.15188v1 Announce Type: new  Abstract: LLMs can generate fluent prose. Story quality depends on how decisions about plot, character, and language work together across planning, drafting, and revision. Guiding these decisions presents two bottlenecks: the quality of story guidance and its sustained use. We formulate Vibe Narrativizing as the task of turning natural-language writing requirements into a finished story and present MUSE, a Theory-Harnessed Story Engine. MUSE organizes story knowledge as guidance for specific decisions and carries those decisions into subsequent creative work. Knowledge engineering develops Robert McKee's story theory through rule atomization, semantic consolidation, and mechanism abstraction; a single source of truth and layered disclosure organize the resulting guidance. Typical examples complement principles that depend on context and aesthetic judgment. An agent harness organizes design, character performance, scene composition, and revision th
    
[^49]: CITECHOICE：智能体搜索中文档呈现方式如何重新分配引用归属的因果审计

    CITECHOICE: A Causal Audit of How Document Presentation Redistributes Citation Credit in Agentic Search

    [https://arxiv.org/abs/2609.15164](https://arxiv.org/abs/2609.15164)

    该论文提出CITECHOICE因果审计框架，通过对真实智能体搜索记录进行2×2重放实验，发现结构化文档呈现会集中引用归属（每答案+0.50次引用）而非明显提高来源接纳率，揭示了文档呈现方式对引用分配的因果影响。

    

    当多个检索到的来源支持同一论断时，答案引擎只会引用其中一部分而忽略其他。我们将这一决策称为“引用分配”，并介绍CITECHOICE——一个针对真实多轮智能体搜索的因果审计方法。CITECHOICE从129个日常查询对话记录中，筛选出113对在相同调用中出现、且经独立验证均支持同一预定事实的文档对，筛选过程不观察排名或答案结果；盲评人工审核确认了其中103对。该方法运行经过哈希验证的2×2重放实验，将配对顺序与共同生成、经保真度检查的目标文档结构化和散文式两种呈现形式进行交叉组合，同时保持对话记录的其余部分固定不变。研究得出三个结果。首先，也是最重要的：结构化呈现会集中引用归属，而非明显提高来源的接纳率。它使目标文档的引用次数每答案提高+0.50次（95%置信区间 [+0.20, +0.84]；Holm校正后p=.033），同时并未增加总引用次数。

    arXiv:2609.15164v1 Announce Type: new  Abstract: When several retrieved sources support the same claim, an answer engine cites some but not others. We call this decision citation allocation and introduce CITECHOICE, a causal audit of authentic multi-turn agentic search. From 129 everyday-query transcripts, CITECHOICE selects 113 same-call document pairs with independently verified support for the same pre-specified fact, without observing ranks or answer outcomes; blinded human review confirms 103. It runs a hash-verified 2-by-2 replay crossing pair order with jointly generated, fidelity-checked structured and prose renderings of one target while the rest of the transcript remains fixed.   Three results emerge. First, and most importantly, structured rendering concentrates citation credit rather than clearly increasing source admission. It raises target citation count by +0.50 citations per answer (95 percent CI [+0.20, +0.84]; Holm-adjusted p=.033), without increasing total citations 
    
[^50]: EMR：通过经验挖掘与复用实现自我进化的医疗多智能体系统

    EMR: Self-Evolving Medical Multi-Agent System via Experience Mining and Reuse

    [https://arxiv.org/abs/2609.15161](https://arxiv.org/abs/2609.15161)

    EMR提出了一种能自我进化的医疗多智能体系统，其核心创新在于构建了包含临床原则、诊断模式和代表性病例的分层临床经验库，通过从推理轨迹中自动挖掘诊断成功经验与失败警示并持续更新经验库，实现了从既往诊疗案例中不断学习进化的能力。

    

    由大语言模型（LLM）驱动的多智能体系统在复杂临床推理中展现出了良好前景，但现有方法依赖静态策略且缺乏持久化的临床记忆，无法从既往诊断的成功与失败中实现自我进化。我们提出了EMR，一种通过经验挖掘与复用实现自我进化的医疗多智能体系统。EMR引入了一个分层的临床经验库，将积累的知识组织为三个层级：临床原则、诊断模式和代表性病例。在推理过程中，EMR模拟多学科会诊模式：一个规划器智能体协调特定领域的科室智能体进行专业化推理，同时一个总结智能体将各科室的分析综合为最终决策。关键的是，EMR能够从多智能体推理轨迹中自动提取正确的诊断见解和与失败相关的警示，逐步更新经验库以指导未来的诊断推理。

    arXiv:2609.15161v1 Announce Type: cross  Abstract: Large language model (LLM) driven multi-agent systems have shown promise in complex clinical reasoning, yet existing approaches rely on static strategies and lack persistent clinical memory, preventing self-evolving from prior diagnostic successes and failures. We present EMR, a self-evolving medical multi-agent system via Experience Mining and Reuse. EMR introduces a hierarchical clinical experience library that organizes accumulated knowledge into three levels: clinical principles, diagnostic patterns, and representative cases. During inference, EMR emulates multidisciplinary consultation: a planner agent coordinates domain-specific department agents for specialized reasoning, while a summary agent synthesizes their analyses into a final decision. Critically, EMR automatically extracts correct diagnostic insights and failure-related warnings from multi-agent reasoning trajectories, incrementally updating the experience library to gui
    
[^51]: PACE：渐进式角度到范数对比嵌入

    PACE: Progressive Angular-to-Norm Contrastive Embedding

    [https://arxiv.org/abs/2609.15152](https://arxiv.org/abs/2609.15152)

    提出PACE两阶段框架，通过渐进式扩展表示空间和可训练参数空间，解决了点积相似度训练不稳定的问题，实现同时利用角度和范数信息的多模态嵌入学习。

    

    多模态嵌入模型将异构输入编码到共享的嵌入空间中，从而实现跨模态、跨任务的高效相似度计算。现有的大多数方法优化基于余弦相似度的对比目标，这虽然能促进训练的稳定性，但将语义兼容性限制在角度几何范围内，使嵌入范数无法作为额外的语义信号。然而，直接优化更具表现力、能同时利用角度和范数信息的点积相似度，其效果却不如基于余弦的训练，并且表现出不稳定的训练动态。我们将这种差异归因于过早的优化空间扩展，其表现为表示空间中的角度-范数纠缠和方向各向异性，且全参数微调进一步加剧了这一问题。在本文中，我们提出了PACE，一个渐进式扩展表示空间和可训练参数空间的两阶段框架。

    arXiv:2609.15152v1 Announce Type: cross  Abstract: Multimodal embedding models encode heterogeneous inputs into a shared embedding space, enabling efficient similarity computation across modalities and tasks. Most existing methods optimize cosine-based contrastive objectives, which promote stable training but restrict semantic compatibility to angular geometry, precluding embedding norms from serving as an additional semantic signal. However, directly optimizing the more expressive dot-product similarity, which leverages both angular and norm information, underperforms cosine-based training and exhibits unstable training dynamics. We attribute this discrepancy to premature optimization-space expansion, manifested as angular--norm entanglement and directional anisotropy in the representation space and further compounded by full-parameter fine-tuning. In this paper, we propose PACE, a two-stage framework that progressively expands both the representation and trainable parameter spaces. S
    
[^52]: 通过推理过程错误分类提升大语言模型的数学推理能力

    Improving Mathematical Reasoning Capabilities in Large Language Models via Reasoning Process Error Classification

    [https://arxiv.org/abs/2609.15145](https://arxiv.org/abs/2609.15145)

    本文通过定义并分类大语言模型数学推理过程中的21种错误类型，识别出高频错误类别，并据此设计了聚焦八类高频错误的提示，有效提升了LLM的数学推理性能。

    

    大语言模型（LLM）的推理能力是基于LLM的实际应用中的关键因素。为了考察LLM当前的推理能力，我们厘清了LLM在数学数据集上推理过程中出现的错误类型。我们聚焦于LLM给出错误答案的问题，将推理过程中的错误定义为推理错误，并人工分析了推理错误的特征。我们定义并分类了21种错误类别，并识别出其中频繁出现的类别。除了定性评估之外，我们还利用评估结果来提升推理能力。我们设计了一个明确聚焦于八种错误类别的提示。实验表明，该提示有效提升了推理性能。此外，结果表明本文识别出的高频推理错误在同等规模的LLM中普遍存在。

    arXiv:2609.15145v1 Announce Type: new  Abstract: The reasoning ability of large language models (LLMs) is a critical factor for practical LLM-based applications. To investigate the current reasoning capability of LLMs, we clarify the types of errors that arise in LLMs' reasoning processes on mathematical datasets. We focus on problems where LLMs produce an incorrect answer. We define errors in the reasoning process as reasoning errors and manually analyze the features of reasoning errors. We defined and classified 21 error classes and identified the frequently occurring classes among them. Beyond qualitative evaluation, we leverage the evaluation results to improve the reasoning capability. We designed a prompt that explicitly focuses on eight error classes. The experiments demonstrate that this prompt effectively improves reasoning performance. Furthermore, the results suggest that the frequent reasoning errors identified in this paper are common across LLMs of comparable scale.
    
[^53]: MoME：用于上下文感知稀疏查找的记忆嵌入混合

    MoME: Mixture-of-Memory Embeddings for Context-Aware Sparse Lookup

    [https://arxiv.org/abs/2609.15126](https://arxiv.org/abs/2609.15126)

    MoME 提出了一种上下文感知的记忆嵌入机制，用 M 个记忆槽位的混合替代每个词元的单一记忆条目，并通过基于隐藏状态的学习门控选择读取的槽位，从而解决现有方法将同一词元的不同上下文语义坍缩为单一固定条目的问题。

    

    高效扩展大语言模型推动了稀疏容量机制的发展，例如混合专家模型，以及更近期的条件记忆机制：一种以词元为索引的嵌入表，通过廉价的参数化查找来增强骨干网络。现有的记忆嵌入方法通过表面形式的确定性函数进行检索，这会将同一个词元在不同上下文中的语义（例如 python 作为编程语言 vs. 蟒蛇动物）坍缩到单一的固定条目中。我们提出了记忆嵌入混合，这是一种上下文感知的记忆机制，它用 M 个槽位的混合来替代每个词元的单一记忆行，并使用基于隐藏状态学习得到的门控来决定在每个位置读取哪些槽位。在 nanochat、Llama-3/MobileLLM 和 Qwen3 骨干网络上进行的受控预训练实验中，MoME 在等参数量和等训练 FLOP 设置下均优于 Value Embedding、Bigram 和 STEM 等基线方法，并展现出更有前景的记忆扩展能力。

    arXiv:2609.15126v1 Announce Type: cross  Abstract: Scaling large language models efficiently has motivated sparse capacity mechanisms such as Mixture-of-Experts and, more recently, conditional memory: token-indexed embedding tables that augment the backbone with cheap parametric lookups. Existing memory-embedding methods retrieve via a deterministic function of the surface form, which collapses different contextual senses of the same token (e.g., python the language vs. the animal) into a single fixed entry. We introduce Mixture of Memory Embeddings (MoME), a context-aware memory mechanism that replaces each token's single memory row with a mixture of M slots and uses a learned gate over the hidden state to choose which slots to read at each position. In controlled pretraining experiments across nanochat, Llama-3/MobileLLM, and Qwen3 backbones, MoME improves over Value Embedding, Bigram, and STEM baselines in iso-parameter and iso-training-FLOP settings, shows a more promising memory-s
    
[^54]: 当错误的钥匙获胜：理解与检测大语言模型中的幻觉

    When the Wrong Key Wins: Understanding and Detecting Hallucinations in LLMs

    [https://arxiv.org/abs/2609.15106](https://arxiv.org/abs/2609.15106)

    大语言模型的幻觉源于预训练关联之间“潜在钥匙”的竞争，本文据此提出一种两阶段关键词扰动检测方法，通过移除关键词条并观察预测重组方式，区分误导性关联导致的错误与有据可依的正确答案。

    

    大语言模型即使在其给出正确答案所需的知识已经可获得的情况下，仍可能产生幻觉。我们通过推理的“潜在钥匙”视角来研究这种失败现象，即答案的选择取决于预训练期间习得的各关联之间的竞争。我们证明，模型预测可能对查询单个关键词高度敏感，这些具有影响力的关键词表现出针对特定实体的绑定特性，且其影响受预训练频率的系统性塑造。多个绑定还可以在同一查询内相互竞争并表现出高阶交互作用。基于这一机制，我们提出了一种用于幻觉检测的两阶段关键词扰动方法。通过移除有影响力的关键词并测量模型如何重组其预测，该方法能够将由误导性关键关联导致的错误与由诊断性证据支持的正确决策区分开来。在多个模型和基准测试上的结果……

    arXiv:2609.15106v1 Announce Type: new  Abstract: Large language models can hallucinate even when the knowledge required for a correct answer is already available. We study this failure through a latent-key view of inference, where answer selection depends on competition among associations acquired during pretraining. We show that model predictions can be highly sensitive to individual query keywords, that these influential keywords exhibit entity-specific binding, and that their effects are systematically shaped by pretraining frequency. Multiple bindings can also compete and exhibit higher-order interactions within the same query. Based on this mechanism, we introduce a two-stage keyword-perturbation method for hallucination detection. By removing influential keywords and measuring how the model reorganizes its prediction, the method distinguishes errors caused by misleading key associations from correct decisions supported by diagnostic evidence. Across multiple models and benchmarks
    
[^55]: OpenAI4S：代码即行动，科学即会话

    OpenAl4S: Code as Action, Science as Sessions

    [https://arxiv.org/abs/2609.15096](https://arxiv.org/abs/2609.15096)

    OpenAI4S是一个以“代码即行动，科学即会话”为核心原则的开源科学研究智能体，通过持久化Python/R计算内核、行动账本、执行记录与工作区检查点等机制，保障长期科研工作流的可检查性、可恢复性与可复现性。

    

    AI协同科学家能够加速计算研究，但在长期运行的研究过程中，工作流程还必须保持可检查、可恢复和可复现，这需要持久的计算状态和来源追踪。在此，我们提出OpenAI4S——一个围绕“代码即行动，科学即会话”原则构建的开源科学研究智能体。OpenAI4S将持久计算运行时与研究会话管理相结合：编排通过结构化工具调用实现，而科学行动则表示为在持久化Python和R内核中执行的完整代码单元。仅追加的行动账本、逐单元执行记录、版本化工件、环境记录和工作区检查点保存了结果的产生过程，并支持会话的恢复、分支和扩展。可配置的沙箱、权限控制以及代码与轨迹筛查提供了互补的安全保障。我们评估……（原文摘要在此处截断）

    arXiv:2609.15096v1 Announce Type: new  Abstract: AI co-scientists could accelerate computational research, but over a long-running study the workflow also has to stay inspectable, resumable and reproducible, which requires persistent computational state and provenance. Here we present OpenAI4S, an open-source scientific research agent built around the principle of \emph{Code as Action, Science as Sessions}. OpenAI4S combines a persistent computing runtime with research-session management: orchestration is handled through structured tool calls, while scientific actions are represented as complete code cells executed in persistent Python and R kernels. An append-only Action Ledger, per-cell execution records, versioned artifacts, environment records, and workspace checkpoints preserve how results were produced and support session recovery, branching, and extension. Configurable sandboxing, permission controls, and code and trajectory screening provide complementary safeguards. We evaluat
    
[^56]: 翻译翻译者：分解强制英语智能体间通信的成本

    Translating the Translator: Decomposing the Cost of English-Forced Inter-Agent Communication

    [https://arxiv.org/abs/2609.15079](https://arxiv.org/abs/2609.15079)

    本研究首次量化了多智能体LLM系统中强制英语通信的“英语强制税”，发现相比母语流水线，强制英语路由使四种语言的精确匹配准确率下降13.0至30.6个百分点。

    

    多智能体LLM架构（如LangChain和AutoGen）在很大程度上假设英语作为内部智能体间通信的通用语，即使最终用户任务是非英语的。我们填补了这一空白：使用Aya-23-8B模型，在四种类型学上多样的语言（印地语、中文、西班牙语、阿拉伯语；每种语言n=300）上评估了一个由两个智能体组成的抽取-回答核心，并在强制英语条件下增加了一个额外的回译智能体。我们比较了母语流水线与强制英语流水线（后者包含从英语回译至用户语言的最终步骤）。我们发现了一个具有统计学显著性的“英语强制税”（通过了严格的Bonferroni校正），该发现将英语路由的成本与一般的多智能体编排开销分离开来。强制通过英语进行智能体间通信使精确匹配准确率降低了13.0个百分点（西班牙语）至30.6个百分点（

    arXiv:2609.15079v1 Announce Type: cross  Abstract: Multi-agent LLM architectures, such as LangChain and AutoGen, largely assume English as the lingua franca for internal inter-agent communication, even when the end-user task is non-English. We fill this gap by evaluating a two-agent extraction-answer core, with an additional back-translation agent in the English-forced condition, across four typologically diverse languages (Hindi, Chinese, Spanish, Arabic; n = 300 per language) using the Aya-23-8B model. We compare a native-language pipeline to an English-forced one (which incorporates a final back-translation step from English to the user's language). We discover a statistically significant English-Forcing Tax (surviving a strict Bonferroni correction) that isolates the cost of English routing from general multi-agent orchestration overhead. Forcing inter-agent communication through English reduces Exact Match accuracy by 13.0 percentage points (Spanish) up to 30.6 percentage points (
    
[^57]: DA-DLM：在扩散语言模型中显式建模词元依赖关系

    DA-DLM: Explicitly Modeling Token Dependencies in Diffusion Language Models

    [https://arxiv.org/abs/2609.15070](https://arxiv.org/abs/2609.15070)

    DA-DLM通过面向位置的有向无环图（DAG）设计，在扩散语言模型中显式建模词元间的依赖关系，从而提升生成文本的连贯性，并在多项任务上持续超越Block Diffusion。

    

    扩散语言模型（DLM）通过迭代去噪掩码序列来生成文本，在每一步独立地预测多个词元。这种条件独立性丢弃了词元间的依赖关系，降低了生成文本的连贯性——这一问题与非自回归翻译（NAT）中的多模态问题类似。借鉴有向无环Transformer（DAT）通过有向无环图（DAG）解决NAT中该问题的思路，我们提出了DA-DLM，该模型通过面向位置的DAG设计，将基于DAG的依赖建模适配到DLM的迭代设定中。面向位置的DAG将节点组绑定到固定的输出位置，使得在较早步骤中被确定的词元能够通过学习到的转移关系为相邻预测提供锚定，并随着去噪过程的推进而演化，在锚点不断积累的同时聚焦于剩余的不确定性。在语言建模、开放式生成和文本摘要任务上，DA-DLM始终优于Block Diffusion，尤其是……

    arXiv:2609.15070v1 Announce Type: new  Abstract: Diffusion Language Models (DLMs) generate text by iteratively denoising a masked sequence, independently predicting multiple tokens at each step. This conditional independence discards inter-token dependencies and degrades coherence-an issue that parallels the multi-modality problem in Non-Autoregressive Translation (NAT). Drawing on the Directed Acyclic Transformer (DAT), which tackles this problem in NAT via a Directed Acyclic Graph (DAG), we propose DA-DLM, a model that adapts DAG-based dependency modeling to DLMs' iterative setting through a position-oriented DAG design. The position-oriented DAG binds node groups to fixed output positions so that tokens fixed in earlier steps anchor neighboring predictions via learned transitions, and evolves with denoising to focus on remaining uncertainty as anchors accumulate. On language modeling, open-ended generation, and summarization, DA-DLM consistently outperforms Block Diffusion, especial
    
[^58]: Salesforce Koa：一个面向智能体工具使用的企业级语言模型

    Salesforce Koa: An Enterprise Language Model for Agentic Tool Use

    [https://arxiv.org/abs/2609.15066](https://arxiv.org/abs/2609.15066)

    Salesforce Koa 是一个基于 Nemotron-3-Super-120B 并通过 GRPO 强化学习后训练的企业级语言模型，其核心创新在于“仿真到奖励”流水线——将工作流规范扩展为以角色为条件的多轮任务，并以成功工具使用作为任务解决奖励，从而在保持通用性能的同时显著提升智能体工具使用能力。

    

    我们介绍了 Salesforce Koa，一个通过对开放权重的 Nemotron-3-Super-120B 基础模型进行后训练，并采用基于群体相对策略优化（GRPO）的强化学习而构建的企业级语言模型。Salesforce Koa 在公开数据和合成生成的数据上训练，不使用任何客户数据，旨在提升工具使用和智能体能力，同时保持强大的通用性能。其独特的组成部分是一个“仿真到奖励”流水线，该流水线将工作流规范扩展为以角色为条件的多轮任务，并以成功的工具使用为基础，为数据依赖型请求提供任务解决奖励。在企业领域，这些规范采用 Agent Script 编写，即 Salesforce 用于构建 Agentforce 智能体的声明式语言；而在公共工具使用领域，我们直接合成工作流结构。相同的仿真和基于真实结果的奖励机制驱动 GRPO 在两类场景中运行。在公共工具使用、智能体……

    arXiv:2609.15066v1 Announce Type: cross  Abstract: We present Salesforce Koa, an enterprise language model built by post-training the open-weight Nemotron-3-Super-120B foundation model with reinforcement learning using Group Relative Policy Optimization (GRPO). Salesforce Koa is trained on public and synthetically generated data, with no customer data, to improve tool use and agentic capabilities while preserving strong general-purpose performance. Its distinctive component is a simulation-to-reward pipeline that expands workflow specifications into persona-conditioned multi-turn tasks with task-resolution rewards grounded in successful tool use for data-dependent requests. For enterprise domains, these specifications are written in Agent Script, Salesforce's declarative language for building Agentforce agents; for public tool-use domains, we synthesize the workflow structure directly. The same simulation and grounded-reward machinery drives GRPO across both. Across public tool-use, ag
    
[^59]: 并非所有提示词都生而平等：面向多模态强化后训练的探索引导式提示词脚手架

    Not All Prompts Are Equal: Exploration-Guided Prompt Scaffolding for Multimodal Reinforcement Post-Training

    [https://arxiv.org/abs/2609.15051](https://arxiv.org/abs/2609.15051)

    提出了探索潜力分数（EPS）和探索引导的提示词脚手架框架，通过动态调整训练提示词分布并利用教师模型改写低效用提示词，提升多模态大语言模型强化学习后训练的效率。

    

    在线强化学习（RL）中的训练提示词在为当前策略提供信息量方面存在显著差异：有些提示词已经饱和，而另一些则过于困难而无法产生可靠的学习信号，但在标准训练下两者却获得相同的推理预算。我们提出了一个探索引导的提示词脚手架框架，该框架在多模态大语言模型（MLLM）的RL后训练过程中动态调整训练提示词分布。我们方法的核心是探索潜力分数，这是一个轻量级的、基于推理的提示词效用代理指标，其理论基础源自KL正则化策略改进理论，可直接从在策略推理统计中计算，无需额外开销。我们没有丢弃低效用提示词，而是利用教师模型生成脚手架式的改写版本，在保留原始任务意图的同时使后续训练更具信息量，从而重新构架教学……

    arXiv:2609.15051v1 Announce Type: cross  Abstract: Training prompts in online reinforcement learning (RL) differ substantially in how informative they are for the current policy: some are already saturated while others are too difficult to yield reliable learning signals, yet both receive equal rollout budget under standard training. We propose an exploration-guided prompt scaffolding framework that adapts the training prompt distribution dynamically throughout RL post-training of multimodal large language models (MLLMs). Central to our approach is the $\textit{Exploration Potential Score} (EPS)$, a lightweight rollout-based proxy for prompt utility derived from KL-regularized policy improvement theory, computable directly from on-policy rollout statistics without additional overhead. Rather than discarding low-utility prompts, we use a teacher model to generate scaffolded rewrites that preserve the original task intent while making subsequent training more informative, reframing teach
    
[^60]: 魔镜魔镜告诉我：小型指令语言模型中的提示回显现象

    Mirror, Mirror on the Wall: Prompt Echoing in Small Instruct Language Models

    [https://arxiv.org/abs/2609.15045](https://arxiv.org/abs/2609.15045)

    本研究对Gemma、Llama、Qwen、SmolLM和OLMo等多个模型家族的分析表明，小型指令语言模型的提示回显现象虽与训练数据存在部分重叠，但主要由模型的归纳头机制驱动，而非简单的训练数据泄露。

    

    提示回显是指令语言模型公认的一种失效模式，即模型不是生成回复，而是复述所提供的提示，即使它并未收到这样做的明确指令。这种现象是模型泄露其训练数据集内容的迹象，还是由内部归纳/复制机制的失调行为引起的？我们研究了来自不同家族（Gemma、Llama、Qwen、SmolLM和OLMo）的小型语言模型中的提示回显现象，结果表明出现回显的提示可能与训练数据集存在部分重叠，但这一现象主要由模型的归纳头机制驱动。

    arXiv:2609.15045v1 Announce Type: cross  Abstract: Prompt echoing is a recognized failure mode of instruct language models, in which a model instead of generating a response, mirrors the provided prompt, even though it did not receive a specific instruction to do so. Is this phenomenon a sign of the model leaking the content of its training dataset, or is it rather caused by a misaligned behavior of the internal induction/copying mechanisms? We investigate prompt echoing small language models from different families (Gemma, Llama, Qwen, SmolLM and OLMo) and show that echoing prompts are likely to have partial overlap with the training dataset but the phenomenon is primarily driven by the model's induction heads.
    
[^61]: 语言模型农业决策模拟中的“平均农民”幻觉

    The average-farmer illusion in language-model simulations of agricultural decisions

    [https://arxiv.org/abs/2609.15038](https://arxiv.org/abs/2609.15038)

    语言模型虽能复现农民决策的总体均值与采纳率，但个体预测能力薄弱、政策相关的极端情形大量缺失，甚至不如一个仅拟合边际分布的简单生成器，表明基于总体平均的“真实性”证据是一种幻觉。

    

    语言模型智能体越来越多地被用作调查和社会模拟中的“合成人口”，但其表面的真实性往往仅通过总体均值或分布相似性来评判。我们在四种预先设定的提示词设计下比较了Claude、Codex和Kimi，并将其与中国及四个非洲国家的匹配农民决策数据进行对照，以检验这类证据究竟能说明什么。部分配置能够复现观测到的均值和采纳率。然而，它们在个体层面的预测能力很弱；其决策集中于典型值附近，而具有政策意义的极端情形大多缺失。最引人注目的是，一个仅拟合观测边际分布、且不获取任何农民个体信息的简单生成器，在分布相似性上竟然超越了所有语言模型配置。提示词的改进带来的是条件性的收益而非普遍的提升：结果随模型、结局变量和人群而异。

    arXiv:2609.15038v1 Announce Type: new  Abstract: Language-model agents are increasingly used as synthetic people in surveys and social simulations, yet their apparent realism is often judged from population averages or distributional similarity. We tested what such evidence actually establishes by comparing Claude, Codex and Kimi under four prespecified prompt designs with matched farmer decisions from China and four African countries. Some configurations reproduced observed means and adoption rates. However, their person-level predictions were weak; their decisions clustered around typical values and policy-relevant extremes were largely missing. Most strikingly, a simple generator fitted only to the observed marginal dis- tribution, and given no information about any farmer, achieved greater distributional similarity than every language-model configuration. Prompt additions produced conditional gains rather than uni- versal improvement: results varied with model, outcome, population 
    
[^62]: MoARa：面向低秩大语言模型预训练的模块感知秩分配与结构保持分解

    MoARa: Module-Aware Rank Allocation and Structure-Preserving Decomposition for Low-Rank LLM Pre-training

    [https://arxiv.org/abs/2609.15037](https://arxiv.org/abs/2609.15037)

    MoARa通过模块感知的投影秩分配与结构保持的幅值-方向分解，使Llama 2 7B低秩预训练在减少37%步数和34%耗时的情况下达到标准GaLore的最终困惑度。

    

    低秩梯度投影可以降低大语言模型（LLM）预训练中优化器状态的内存成本，但达到目标质量所需的训练步数和实际耗时仍是有待改进的重要方面。我们将此归因于现有方法中的两个设计选择：投影秩预算在投影敏感度各异的Transformer模块之间被均匀分配，以及对原始梯度进行投影会同时削弱其幅值和方向。我们提出了MoARa，它将基于静态性能剖析的模块感知投影秩分配与分块的幅值-方向分解相结合，默认块大小设置在注意力头维度附近。在涵盖Llama、Qwen和DeepSeek五种Transformer架构（300M至7B规模）的实验中，采用MoARa的GaLore在Llama 2 7B上仅用减少37%的训练步数和34%的实际耗时即可达到标准GaLore的最终困惑度，且仅损失0.2%的性能。

    arXiv:2609.15037v1 Announce Type: cross  Abstract: Low-rank gradient projection reduces the optimizer-state memory cost of large language model (LLM) pretraining, but the steps and wall-clock time needed to reach a target quality remain a meaningful axis for improvement. We attribute this to two design choices in existing methods: the projection-rank budget is allocated uniformly across Transformer modules with heterogeneous projection sensitivity, and projecting a raw gradient attenuates its magnitude and direction jointly. We propose MoARa, which combines a static profiling-based module-aware projection-rank allocation with a block-wise magnitude-direction decomposition; the default block size is set in the neighborhood of the attention head dimension. Across five Transformer architectures spanning Llama, Qwen, and DeepSeek at 300M to 7B scales, GaLore with MoARa reaches standard GaLore's final perplexity in 37% fewer steps and 34% less wall-clock time on Llama 2 7B, with only 0.2% p
    
[^63]: 选择你的毒药：学习选择毒药集以实现更强的LLM后门攻击

    Pick Your Poison: Learning to Select Poison Sets for Stronger LLM Backdoor Attacks

    [https://arxiv.org/abs/2609.15029](https://arxiv.org/abs/2609.15029)

    该论文揭示了随机采样投毒样本会严重低估LLM后门攻击的最坏情况脆弱性（攻击成功率可从3%到80%不等），并提出SAILS方法通过学习集合评分器智能选择最优毒药集，使攻击成功率平均提升30个百分点。

    

    后门投毒攻击在原本干净的微调数据中添加投毒样本，将触发器与模型在触发器出现时学会产生的目标行为进行配对。现有评估通常固定投毒样本的数量，并从候选池中随机采样。我们证明这种方式会严重低估最坏情况下的脆弱性：在三个LLaMA-3-8B后门攻击设置中，保持模型、干净数据和投毒数量不变，攻击成功率仅取决于所选的毒药集，范围可从3%到80%。我们将毒药选择形式化为oracle预算约束下的集合优化问题，并提出了SAILS（集合级审计知情的迭代学习选择）方法，该方法从几百次微调-评估运行中学习集合评分器，对数百万个候选集合进行排序，并仅审计少量入围集合。与最强的影响函数基线相比，SAILS将保留集攻击成功率平均提高了30个百分点。

    arXiv:2609.15029v1 Announce Type: cross  Abstract: Backdoor poisoning attacks add poisoned examples to otherwise-clean finetuning data, pairing a trigger with a target behavior that the model learns to produce when the trigger appears. Existing evaluations typically fix the number of poisoned examples and sample them at random from a candidate pool. We show that this can severely underestimate worst-case vulnerability: across three LLaMA-3-8B backdoor settings, holding the model, clean data, and poison count fixed, attack success ranges from 3% to 80% depending only on which poison set is chosen.   We formalize poison selection as oracle-budgeted set optimization and introduce SAILS (Set-level Audit-Informed Iterative Learned Selection), which learns a set scorer from a few hundred finetune-and-evaluate runs, ranks millions of candidate sets, and audits only a small shortlist. SAILS improves held-out attack success by 30 percentage points on average over the strongest influence baselin
    
[^64]: SALUTE：面向国防领域的大语言模型基准测试与适配

    SALUTE: Benchmarking and Adapting LLMs for the Defense Domain

    [https://arxiv.org/abs/2609.15022](https://arxiv.org/abs/2609.15022)

    提出了SALUTE端到端框架，通过构建国防领域语料库、指令数据集、偏好数据集和严格筛选的基准测试，系统性地填补了大语言模型在国防领域评估与适配方面的空白。

    

    国防是一个知识密集型领域，需要对专业术语、条令概念、作战程序以及不断演变的军事事件进行精确理解。尽管近期已有研究探索了面向军事应用的语言技术，但现有工作仍然较为碎片化：它们往往是针对特定任务的、依赖于有限的适配流程，或者缺乏全面的国防领域评估。在本文中，我们提出了SALUTE，一个用于国防领域大语言模型基准测试与适配的端到端框架。SALUTE整合了以下四个部分：Salute-Corpus，一个从公开获取的美国军事条令和政府文件中精选构建的语料库；Salute-Conv，一个基于条令来源和十年国防新闻构建的接地指令数据集；Salute-Pref，一个具有国防领域意识的偏好数据集；以及Salute-Bench，一个经过严格筛选的基准测试，用于评估模型对条令和国防新闻的理解与推理能力。

    arXiv:2609.15022v1 Announce Type: new  Abstract: Defense is a knowledge-intensive domain that requires precise understanding of specialized terminology, doctrinal concepts, operational procedures, and evolving military events. Although recent work has explored language technologies for military applications, existing efforts remain fragmented: they are often task-specific, rely on limited adaptation pipelines, or lack comprehensive defense-domain evaluation. In this paper, we present SALUTE, an end-to-end framework for benchmarking and adapting LLMs for the defense domain. SALUTE integrates Salute-Corpus, a curated corpus from open-access U.S. military doctrine and government documents; Salute-Conv, a grounded instruction dataset from doctrinal sources and decade-long defense news; Salute-Pref, a defense-aware preference dataset; and Salute-Bench, a rigorously filtered benchmark for evaluating defense-domain understanding and reasoning over doctrine and defense news. Based on these res
    
[^65]: ABSOL：由大语言模型协调的聚合贝叶斯子采样方法

    ABSOL: Aggregated Bayesian Subsampling Orchestrated with LLMs

    [https://arxiv.org/abs/2609.15007](https://arxiv.org/abs/2609.15007)

    ABSOL是一个利用大语言模型作为有界语义引导来协调贝叶斯网络结构学习的混合框架，是唯一在所有基准测试中都能生成可行图的方法，并在较大规模基准上取得最高的边F1分数，相比非大语言模型方法平均提升0.23。

    

    大语言模型越来越多地被用作结构化数据的自然语言接口，但当答案需要一致性的证据条件化、依赖感知推理和不确定性估计时，它们仍然不可靠。贝叶斯网络提供了一个显式的概率推理层，但从数据中学习有用的结构在大规模场景下仍然成本高昂且脆弱。我们提出了ABSOL，一个由大语言模型引导的混合贝叶斯网络结构学习框架，它将大语言模型作为有界语义引导。在涵盖27到1041个节点的五个离散贝叶斯网络基准测试中，ABSOL是唯一在每个基准测试上都能产生可行图的方法，并且在使用GPT-5.4时，在所有大于27个节点的基准测试上取得了最高的边F1分数。四种大语言模型增强方法为统计骨干网络贡献了互补的语义证据，使边F1分数相比非大语言模型聚合骨干网络平均提高了0.23。补充性的事后细化……

    arXiv:2609.15007v1 Announce Type: new  Abstract: Large language models are increasingly used as natural-language interfaces to structured data, yet they remain unreliable when answers require consistent evidence conditioning, dependency-aware reasoning, and uncertainty estimation. Bayesian networks provide an explicit probabilistic reasoning layer, but learning useful structures from data remains costly and fragile at scale. We introduce ABSOL, a hybrid LLM-guided Bayesian network structure-learning framework that uses LLMs as bounded semantic guides. Across five discrete BN benchmarks spanning 27 to 1041 nodes, ABSOL is the only evaluated method to produce a viable graph on every benchmark, and achieves the highest Edge F_1 on every benchmark larger than 27 nodes with GPT-5.4. The four LLM augmentations, which contribute complementary semantic evidence to the statistical backbone, improve Edge F_1 over the non-LLM aggregation backbone by +0.23 on average. Complementary post-hoc refine
    
[^66]: 超越深度与宽度：流式测试时计算中的信息-余量困境

    Beyond Depth and Width: The Information-Slack Dilemma in Streaming Test-Time Compute

    [https://arxiv.org/abs/2609.14995](https://arxiv.org/abs/2609.14995)

    本文提出“信息-余量困境”这一概念，揭示了流式测试时计算中早期计算（时间充裕但证据不完整）与等待更多信息（证据更可靠但计算余量减少）之间的核心权衡，并据此提出了一个以受控证据修订下选择性恢复为优先的演化证据计算研究议程。

    

    相同的任务和计算预算，当证据以不同顺序到达时，可能需要不同的推理策略。早期计算有更多时间完成，但依赖于不完整或可修订的证据；等待能改善信息，但会缩减计算余量。我们将此称为信息-余量困境。我们以依赖证据的计算任务作为分析单元：何时启动它，什么支撑其结果，以及何时可以提交该结果。预先计算只有在其收益能够经受验证、失效和恢复的成本时才有价值。这既适用于有依据的增量处理和可复用的准备工作，也适用于依赖未来的推测性计算。我们提出了一个关于演化证据下计算的研究议程，优先研究在受控证据修订下的选择性恢复。评估应当区分更早执行带来的效果与相对于完整输入替代方案……

    arXiv:2609.14995v1 Announce Type: new  Abstract: The same task and compute budget can require different reasoning policies when evidence arrives in a different order. Early computation has more time to finish but rests on incomplete or revisable evidence; waiting improves information while shrinking computational slack. We call this the information-slack dilemma.   We take the evidence-dependent computational job as the unit of analysis: when to start it, what supports its result, and when that result can be committed. Advance computation is valuable only insofar as its benefits survive the costs of verification, invalidation, and recovery. This applies to grounded incremental processing and reusable preparation as well as future-dependent speculation.   We propose a research agenda on computation under evolving evidence, prioritizing selective recovery under controlled evidence revisions. Evaluation should separate earlier-execution effects, deployment value against a full-input alter
    
[^67]: MTAC-IFBench：多轮智能体编程中指令遵循能力的基准测试

    MTAC-IFBench: Benchmarking Instruction-Following in Multi-Turn Agentic Coding

    [https://arxiv.org/abs/2609.14992](https://arxiv.org/abs/2609.14992)

    提出了MTAC-IFBench，首个针对多轮智能体编程中指令遵循能力的综合基准，每个实例平均包含7.04轮渐进式开发对话和91.33个约束条件，涵盖6个一级和18个二级约束类别。

    

    近年来，大语言模型（LLM）的快速发展重塑了软件工程领域，使自主代码智能体能够通过规划、执行和迭代调用外部工具来解决复杂任务。除了实现功能正确性之外，这些智能体还必须在整个开发生命周期中忠实遵循流程指令和约束条件。然而，现有基准测试通常只关注最终的功能正确性，或将指令遵循评估局限于单轮、通用聊天或简单代码生成场景，导致多轮智能体编程中的指令遵循能力研究不足。为了填补这一空白，我们提出了MTAC-IFBench，一个针对这一关键能力的综合基准。它具有多轮渐进式软件开发指令，包含跨越6个一级类别和18个二级类别的多样化约束。每个实例平均包含7.04轮对话和91.33个约束条件，构成了极具挑战性的任务……

    arXiv:2609.14992v1 Announce Type: new  Abstract: Recently, the rapid development of large language models (LLMs) has reshaped software engineering by enabling autonomous code agents that plan, execute, and utilize external tools iteratively to tackle complex tasks. Beyond achieving functional correctness, these agents must faithfully follow process instructions and constraints throughout the development lifecycle. However, existing benchmarks typically focus on final functional correctness or confine instruction-following evaluation to single-turn, general chat or simple code generation scenarios, leaving instruction-following in multi-turn agentic coding underexplored. To bridge this gap, we propose MTAC-IFBench, a comprehensive benchmark for this critical capability. It features multi-turn progressive software development instructions with diverse constraints spanning 6 primary and 18 secondary categories. With an average of 7.04 turns and 91.33 constraints per instance, it poses a r
    
[^68]: 泰风ASR流式系统：基于实时浅融合的可控低延迟泰语语音识别

    Typhoon ASR Streaming: Steerable Low-Latency Thai Speech Recognition with Real-Time Shallow Fusion

    [https://arxiv.org/abs/2609.14991](https://arxiv.org/abs/2609.14991)

    该论文提出了Typhoon ASR Streaming系统，通过缓存感知编码器和带短语增强的GPU n-gram浅融合层，实现了无需重训练即可在解码时控制词汇表的泰语流式语音识别，在一秒前瞻下将字符错误率降低4.3-4.5倍且速度快于实时。

    

    开源泰语自动语音识别（ASR）领域目前主要由离线的基于Whisper的模型主导，这些模型需要读取完整的语音内容后再进行转录，因此无法满足诸如实时字幕和语音代理等低延迟应用场景的需求。我们提出了一个可部署的泰语流式ASR系统，用户可以在解码时控制其词汇表，而无需重新训练。一个广泛使用的开源泰语模型在训练时使用完整上下文，当以真正的流式方式运行时会崩溃；我们通过缓存感知编码器恢复了其流式能力，方法包括转换该模型或适配一个原生流式模型，并添加了一个浅融合层，在流式解码器内部使用GPU n-gram语言模型和短语增强技术对候选结果进行重排序。在两个泰语基准测试和两种模型规模上，流式模型在完整上下文模型失效的情况下仍保持可用，在一秒前瞻的情况下将字符错误率降低了4.3-4.5倍，同时运行速度快于实时。解码时控制功能随后将关键词召回率从16%大幅提升。

    arXiv:2609.14991v1 Announce Type: new  Abstract: Open Thai automatic speech recognition (ASR) is dominated by offline, Whisper-based models that read the whole utterance before transcribing, ruling out low-latency uses such as live captioning and voice agents. We present a deployable system for streaming Thai ASR that lets a user steer its vocabulary at decode time, without retraining. A widely used open Thai model, trained with full context, collapses when run as a true stream; we restore streaming with a cache-aware encoder, by converting it or adapting a natively streaming one, and add a shallow-fusion layer that re-ranks candidates inside the streaming decoder with a GPU n-gram language model and phrase boosting. Across two Thai benchmarks and two model sizes, the streaming models stay usable where the full-context model fails, cutting character error rate 4.3-4.5x at a one-second look-ahead while running faster than real time. Decode-time steering then lifts keyword recall from 16
    
[^69]: 跨26个大型语言模型的生物医学参考文献生成仍然不可靠

    Biomedical Reference Generation Remains Unreliable across 26 Large Language Models

    [https://arxiv.org/abs/2609.14988](https://arxiv.org/abs/2609.14988)

    对来自八个开发商的26个大型语言模型的系统评估表明，模型在为生物医学文本生成参考文献时编造率介于10.2%至98.4%之间，证明即使是最先进的模型，其参考文献生成能力仍然不可靠。

    

    背景：大型语言模型越来越多地被用于辅助撰写生物医学文本，但它们可能会编造指向不存在文献的参考文献。大型语言模型编造参考文献的频率尚未得到充分表征。方法：我们提示来自八个开发商的26个语言模型（2023年至2026年），为十个领域中的69个生物医学段落各自提供缺失的参考文献。参考文献被分类为：可验证（真实论文且具有可解析的标识符）、部分匹配（真实论文但无可解析的标识符）、编造（没有匹配的已收录论文）或拒绝（模型拒绝提供参考文献）。只有当参考文献可验证，且其期刊、年份和所列作者与被引论文完全一致时，才认为该参考文献在所有评估的文献字段中都是正确的。结果：编造率从10.2%（Claude Opus 4.8，该模型拒绝了52.1%的提示请求）到98.4%（Ministral 3B，该模型未产生任何可验证的参考文献）不等。Claude Opus 4.6（注：原摘要在此处被截断）

    arXiv:2609.14988v1 Announce Type: new  Abstract: Background. Large language models are increasingly used to help write biomedical text but may fabricate references to nonexistent work. How often large language models do so is not well characterized. Methods. We prompted 26 language models from eight developers (2023 to 2026) to supply a missing reference for each of 69 biomedical passages across ten domains. References were classified as verifiable (real paper with a resolving identifier), partial matches (real paper without a resolving identifier), fabricated (no matching indexed paper), or declined (the model refused to supply a reference). A reference was considered correct in every evaluated bibliographic field only when it was verifiable and its journal, year, and listed authors matched those of the cited paper. Results. Fabrication ranged from 10.2% (Claude Opus 4.8, which declined 52.1% of prompts) to 98.4% (Ministral 3B, which produced no verifiable reference). Claude Opus 4.6 
    
[^70]: 利用大型视频生成器将序列化模糊认知图转换为因果虚拟世界

    Converting Sequenced Fuzzy Cognitive Maps to Causal Virtual Worlds with Large Video Generators

    [https://arxiv.org/abs/2609.14985](https://arxiv.org/abs/2609.14985)

    该论文提出利用反馈模糊认知图建模虚拟世界的细粒度因果结构，并通过“如果A则B”的动态元规则引导大型语言模型和大型视频生成器创建和操控具有全局均衡因果情景的因果虚拟世界。

    

    我们展示了用户如何利用大型语言模型（LLM）和大型视频模型智能体来创建和操控因果虚拟世界。该方法使用反馈模糊认知图（FCM）来建模虚拟世界的细粒度因果结构，并引导其因果演化。局部因果规则是部分的或模糊的，而FCM的反馈结构产生定义因果情景的全局均衡。一系列形式为“如果A则B”的动态元规则定义了虚拟世界视频的因果场景。前件因果模式A可在用户或智能体的意愿下对FCM的虚拟世界进行扰动。FCM的瞬态反馈动力学定义了元规则的因果蕴含箭头。后件B是产生的均衡吸引子，例如FCM的极限环或不动点。我们的算法从FCM中提取这些元规则并引导LLM智能体……

    arXiv:2609.14985v1 Announce Type: new  Abstract: We show how users can create and manipulate causal virtual worlds with large-language-model (LLM) and large-video-model agents. The approach uses feedback fuzzy cognitive maps (FCMs) both to model the granular causal structure of the virtual world and to guide its causal evolution. The local causal rules are partial or fuzzy while the FCM's feedback structure produces global equilibria that define causal scenarios. A sequence of \emph{dynamical} meta-rules of the form ``If $\mathcal{A}$ then $\mathcal{B}$" define the causal scenes of the virtual-world video. The if-part causal pattern $\mathcal{A}$ perturbs the FCM's virtual world at the user's or agent's discretion. The FCM's transient feedback dynamics define the meta-rule's causal arrow of implication. The then-part $\mathcal{B}$ is the resulting equilibrium attractor such as a FCM limit cycle or fixed point. Our algorithm extracts these meta-rules from the FCM and guides the LLM agen
    
[^71]: 在线语言自适应采样以获得更好的分布式跨语言收益

    Online Language Adaptive Sampling for Better Distributed Cross-lingual Gains

    [https://arxiv.org/abs/2609.14969](https://arxiv.org/abs/2609.14969)

    本文提出一种为每种语言分配可训练采样概率的在线自适应采样策略，让对重新对齐损失贡献更大的低资源语言被更频繁采样，以极小开销持续提升多语言模型的跨语言迁移性能。

    

    重新对齐是提升多语言语言模型跨语言迁移能力的一种有前景的方法，尤其是对极度低资源语言（LRLs）。然而，现有的重新对齐方法依赖于对跨语言平行句子的均匀随机采样，这在批次大小受限的情况下可能是次优的。在实践中，模型可能受益于更频繁地见到某些语言，尤其是那些对齐较差的语言，且最优分布会在训练过程中不断演变。在这项工作中，我们提出了一种简单而有效的自适应采样策略，为每种语言分配可训练的采样概率。对重新对齐损失贡献更大的语言会在后续批次中被更频繁地采样，且最优分布可以在整个训练过程中持续演变。我们的方法采用内外双层优化循环，开销很小，带来了持续的性能提升。

    arXiv:2609.14969v1 Announce Type: cross  Abstract: Realignment is a promising approach for improving the cross-lingual transfer ability of multilingual language models, particularly for extremely low-resource languages (LRLs). However, existing realignment methods rely on uniform and random sampling of parallel sentences across languages, which may be suboptimal under limited batch sizes. In practice, models may benefit from seeing certain languages more frequently, especially those that are poorly aligned, and the optimal distribution can evolve throughout training. In this work, we propose a simple yet effective adaptive sampling strategy that assigns trainable sampling probabilities to each language. Languages that contribute more to the realignment loss are sampled more frequently in subsequent batches, and the optimal distribution can evolve throughout training. Our method employs an inner-outer optimization loop with a small overhead, leading to consistent performance improvement
    
[^72]: 语料库对齐的乌斯玛尼体到标准体古兰经词汇映射及确定性诵读验证器

    A Corpus-Aligned Uthmani-to-Standard Quranic Word Mapping and a Deterministic Recitation Validator

    [https://arxiv.org/abs/2609.14967](https://arxiv.org/abs/2609.14967)

    该论文通过全本古兰经双正字法对齐发布了2,290对乌斯玛尼体到标准体的词汇映射和七步规范化流水线（使90.9%经文规范化后完全一致），并据此构建了一个确定性的、无需大型语言模型的古兰经诵读验证器。

    

    古兰经文本以两种在字节层面截然不同的正字法形式存在：每种印刷版《古兰经》（mushaf）所使用的乌斯玛尼体，以及所有主流阿拉伯语自然语言处理工具所针对的标准体（伊姆拉体，Imla'i）。两者之间的差距集中在一个Unicode字符U+0670（上标alif）上，该字符出现在古兰经中一些最常被诵读的词语中，却被通用阿拉伯语规范化工具静默地错误处理。我们发布了一个包含2,290对词语的、基于语料库对齐的乌斯玛尼体到标准体词汇映射，该映射是通过将完整的6,236节古兰经经文在两种正字法形式之间对齐构建而成，并同时发布了基于该映射的七步文本规范化流水线。通过该流水线对所有6,236节经文的两种形式进行规范化后，90.9%的经文产生了完全相同的字符串，我们对剩余的差异进行了系统刻画，而非断言其已被完全消除。在规范化文本的基础上，我们构建了一个确定性的、无需大型语言模型（LLM-free）的古兰经诵读验证器……

    arXiv:2609.14967v1 Announce Type: new  Abstract: Quranic text is distributed in two orthographic forms that are byte-level distinct: the Uthmani script used in every printed mushaf, and the Standard (Imla'i) Arabic form that every mainstream Arabic NLP tool is built for. The gap is concentrated in one Unicode character, U+0670 (superscript alef), which appears in some of the most frequently recited words in the Quran and is silently mishandled by general-purpose Arabic normalizers. We release a 2,290-pair, corpus-aligned Uthmani-to-Standard word mapping constructed by aligning the complete 6,236-verse Quran across both orthographic forms, together with a seven-step text normalization pipeline built on it. Normalizing both forms of all 6,236 verses through that pipeline yields identical strings for 90.9% of verses, and we characterize the residual divergence rather than assert that it is closed. On top of the normalized text, we build a deterministic, LLM-free Quranic recitation validat
    
[^73]: 我们能否在没有人工参考译文的情况下对古典文本的LLM翻译错误进行分流？——基于巴利语至英语翻译的源文本新颖性、GEMBA评分与预算化审校研究

    Can We Triage LLM Translation Errors in Classical Texts Without Human References? Source Novelty, GEMBA Scoring, and Budgeted Review through Pali-to-English Translation

    [https://arxiv.org/abs/2609.14963](https://arxiv.org/abs/2609.14963)

    本研究通过巴利语至英语的翻译任务比较了五种无参考信号，发现基于多模型面板的无参考GEMBA评分是在没有人工参考译文时识别需要专家审校的LLM古典文本翻译错误的最有效方法。

    

    随着大语言模型成为能够胜任古典文本翻译的工具，一个关键挑战是在没有人工参考译文的情况下，决定哪些翻译输出需要专家审校。本研究通过巴利语至英语的翻译任务测试了无参考的错误分流方法。三个大语言模型翻译了15,493个段落，并比较了五种信号：源文本新颖性、源文本与候选译文之间的嵌入距离、同行翻译分歧、英译巴利语的回译，以及无参考GEMBA评分。这些信号在3,000条有参考译文且由LLM裁决的样本上进行校准，并在500条由作者裁决的锚定样本上进行检验。人工参考译文仅用于校准和验证，从未被用于计算风险信号。结果表明：源文本新颖性是一种有用的源侧风险先验，但并非逐候选译文的错误检测器；同行分歧和回译提供了次要信号；最强的方法是由一组模型进行的无参考GEMBA评分。

    arXiv:2609.14963v1 Announce Type: new  Abstract: As large language models become capable translators of classical texts, a key challenge is deciding which outputs need expert review when no human reference exists. This study tests reference-free error triage through Pali-to-English translation. Three LLMs translated 15,493 passages. Five signals were compared: source novelty, source-candidate embedding distance, peer-translation disagreement, English-to-Pali backtranslation, and no-reference GEMBA scoring. Signals were calibrated on a 3,000-item reference-informed LLM-adjudicated sample and checked against a 500-item author-adjudicated anchor. Human references supported calibration and validation only; they were never used to compute the risk signals. Source novelty was a useful source-side risk prior but not a per-candidate error detector. Peer disagreement and backtranslation provided secondary signal. The strongest method was no-reference GEMBA scoring by a panel of models generally
    
[^74]: 概念重组的几何特征：一种用于检测科学革命的反事实嵌入框架

    Geometric Signatures of Conceptual Reorganization: A Counterfactual Embedding Framework for Detecting Scientific Revolutions

    [https://arxiv.org/abs/2609.14917](https://arxiv.org/abs/2609.14917)

    该论文提出将文档嵌入空间的几何扰动作为概念重组的量化可观测量，通过反事实消融框架在五个历史案例（狭义相对论、哥德尔不完备定理、希格斯机制、深度学习和注意力机制）中验证了可测量几何特征，为科学革命的定量检测开辟了新途径。

    

    我们引入文档嵌入几何作为概念重组的量化可观测量，并开发了一个反事实消融框架，用于测量单个概念如何影响科学知识的组织结构，从而为检测科学革命提供了一个量化框架。该可观测量定义为：在候选概念历史出现之前和之后，从嵌入空间中移除与该概念相关联的文档时所产生的几何扰动。统计验证通过五个跨越物理学、数学和机器学习领域的历史案例研究进行：狭义相对论、哥德尔不完备定理、希格斯机制、深度学习，以及transformer架构背后的注意力机制。在各项历史案例研究中，该框架识别出了与概念重组相关的可测量几何特征，而验证研究（原文在此处被截断）……

    arXiv:2609.14917v1 Announce Type: cross  Abstract: We introduce document embedding geometry as a quantitative observable of conceptual reorganization and develop a counterfactual ablation framework for measuring how individual concepts influence the organization of scientific knowledge, providing a quantitative framework for detecting scientific revolutions. The observable is defined by the geometric perturbation induced when removing documents associated with a candidate concept from the embedding space before and after its historical emergence. Statistical validation is performed using five historical case studies spanning physics, mathematics, and machine learning: special relativity, G\"odel's incompleteness theorems, the Higgs mechanism, deep learning, and the attention mechanism underlying transformer architectures. Across the historical case studies, the framework identifies measurable geometric signatures associated with conceptual reorganization, while the validation studies e
    
[^75]: 蓝色的四十种色调：通过模式条件化强化学习实现质量-多样性对齐

    Forty Shades of Blue: Quality-Diversity Alignment via Mode-Conditioned Reinforcement Learning

    [https://arxiv.org/abs/2609.14896](https://arxiv.org/abs/2609.14896)

    MoDA是一种受多智能体强化学习启发的在线后训练强化学习算法，通过让单一共享LLM策略在不同编号角色条件下竞争生成彼此不同的输出，从而在不牺牲生成质量且无需手工设计角色或修改架构的情况下，有效缓解模式坍塌并提升输出多样性。

    

    LLM对齐训练的一个显著副产品是模式坍塌：输出多样性的逐渐丧失，这种退化缩小了模型在推理时的表达能力。对于需要开放式探索和多元视角的应用（例如科学构思和创意写作）而言，这种退化尤其具有限制性。我们提出了MoDA（模式条件化多样性对齐），这是一种在线后训练强化学习算法，受多智能体强化学习（MARL）中协调视角的启发，能够联合优化生成质量和多样性。MoDA训练一个以抽象编号角色为条件的单一共享LLM策略，其中每个角色充当一个智能体，竞相产生与其他角色不同的输出。这种表述鼓励模式条件化的智能体探索高质量输出空间的互补区域，而无需手工设计的人物角色或架构修改。MoDA采用了一种提示自适应……

    arXiv:2609.14896v1 Announce Type: cross  Abstract: A notable byproduct of LLM alignment training is mode collapse: the progressive loss of output diversity that narrows a model's expressivity at inference time. This degradation is especially limiting for applications requiring open-ended exploration and pluralistic perspectives, such as scientific ideation and creative writing. We present MoDA (Mode-conditioned Diversity Alignment), an online post-training RL algorithm that jointly optimizes generation quality and diversity, inspired by the coordination perspective in multi-agent reinforcement learning (MARL). MoDA trains a single shared LLM policy conditioned on abstract numbered roles, where each role acts as an agent competing to produce outputs distinct from the others. This formulation encourages mode-conditioned agents to explore complementary regions of the high-quality output space without requiring hand-crafted personas or architectural modifications. MoDA employs a prompt-ada
    
[^76]: PeerPen：面向线上心理健康同伴支持的AI辅助写作工具

    PeerPen: AI-Assisted Writing for Online Mental Health Peer Support

    [https://arxiv.org/abs/2609.14886](https://arxiv.org/abs/2609.14886)

    开发了嵌入社区界面的AI写作辅助工具PeerPen，通过15人访谈发现其能减轻心理健康同伴支持志愿者的写作负担并增强信心，但揭示了AI辅助会模糊作者身份、削弱社区信任的核心张力，参与者期望AI辅助而非接管写作。

    

    线上心理健康社区的蓬勃发展依赖于同伴支持，然而志愿提供帮助的人们往往缺乏正式培训，可能难以表达支持性的回应。AI协同写作可以降低这一门槛；然而，同伴支持的价值很大程度上源于其被认为是个性化的，这引发了关于作者身份、所有权和信任的问题。我们构建了PeerPen，这是一个嵌入在类Reddit界面中的写作辅助工具，支持两大主要功能：草稿生成和用户撰写回应的修订。通过对15名参与者的半结构化访谈，我们发现PeerPen减轻了撰写回应的负担，并增强了提供支持的信心。参与者希望AI辅助他们的写作而不接管作者身份，并预见到围绕真实性和信任的张力。这种辅助可能使得即使是未经AI辅助撰写的回应，其作者身份也变得不确定，从而削弱整个社区的信任。

    arXiv:2609.14886v1 Announce Type: cross  Abstract: Online mental health communities thrive on peer support, yet those who volunteer to help often lack formal training and may struggle to articulate supportive responses. AI co-writing could lower this barrier; however, peer support derives much of its value from being perceived as personal, raising questions around authorship, ownership, and trust. We built PeerPen, a writing assistance tool embedded within a Reddit-like interface, supporting two main features: draft generation and revision of user-written responses. Through semi-structured interviews with 15 participants, we find that PeerPen reduced the burden of composing responses and increased confidence in offering support. Participants wanted AI to assist their writing without taking over authorship and anticipated tensions around authenticity and trust. Such assistance could make authorship uncertain even for responses written without it, weakening trust across the community. We
    
[^77]: AgentKV：面向智能体大语言模型的阶段感知KV缓存淘汰方法

    AgentKV: Phase-Aware KV Eviction for Agentic LLMs

    [https://arxiv.org/abs/2609.14872](https://arxiv.org/abs/2609.14872)

    AgentKV发现智能体LLM服务中未来查询分属思考、行动、工具等不同阶段且占据不同的查询子空间，据此提出为每个阶段维护查询缓冲区并针对其并集进行缓存键评分的阶段感知KV淘汰方法，克服了传统基于近期性的淘汰方法在智能体场景下的系统性偏差。

    

    智能体服务所消耗的token数量可能比聊天机器人工作负载高出数个数量级，这对KV缓存的容量和解码阶段的带宽都造成了巨大压力。现有的KV缓存淘汰方法大多基于从最近token中提取的代表性查询来对缓存中的键进行评分，其假设是未来的注意力模式与近期的注意力模式相似。我们证明智能体生成过程违反了这一假设：未来的查询是对思考、行动、工具调用及其他阶段的混合，主角度分析表明这些不同组件占据了显著不同的查询子空间，因此基于近期性的代表点会系统性地低估未来阶段所需的键。我们提出AGENTKV，该方法为每个阶段维护一个小型查询缓冲区，并针对这些缓冲区的并集对缓存键进行评分。我们进一步在一个持久化的多轮服务路径中实现了AGENTKV，该路径能够跨轮次携带压缩后的KV状态，并在线压缩保留的KV页面。在两个模型、六个任务领域和三种KV缓存预算下……

    arXiv:2609.14872v1 Announce Type: cross  Abstract: Agentic serving can consume orders of magnitude more tokens than chatbot workloads, stressing both KV-cache capacity and decode-time bandwidth. Most KV-eviction methods score cached keys against representative queries drawn from the most recent tokens, assuming future attention resembles recent attention. We show that agentic generation violates this assumption: future queries form a mixture over think, act, tool, and others phases, and principal-angle analysis shows these components occupy measurably different query subspaces, so recency representatives systematically undervalue keys that upcoming phases will need. We propose AGENTKV, which maintains a small query buffer per phase and scores cached keys against their union. We further implement AGENTKV in a persistent multi-turn serving path that carries compressed KV state across turns and compacts retained KV pages online. Across two models, six task domains, and three KV budgets ea
    
[^78]: 一个例子就足以通过公平性基准测试：重新思考对齐大语言模型的公平性评估

    One Example Is Enough to Pass Fairness Benchmarks: Rethinking Fairness Evaluation for Aligned LLMs

    [https://arxiv.org/abs/2609.14860](https://arxiv.org/abs/2609.14860)

    仅用一个公平性基准测试示例进行训练或作为上下文演示，就能大幅提升模型在公平性基准上的表现，这表明现有公平性评估基准过于简单，需要重新思考对齐大语言模型的公平性评估方法。

    

    警告：本投稿研究刻板印象与偏见，包含有毒和冒犯性示例，仅用于说明目的。诸如BBQ等公平性基准测试已成为各大模型家族公平性评估的事实标准。我们认为这些基准测试过于简单，无法支撑其应有的作用：仅用一个BBQ示例对Qwen 2.5 7B Base进行群体相对策略优化（GRPO）训练，或将该示例作为单样本上下文学习（ICL）的演示示例，即可将平均BBQ准确率分别从79.9%提升至92.9%和99.0%，其中GRPO方法缩小了与其大规模RLHF对应模型（96.1%）之间80%的差距，而ICL方法则超越了该模型。这些效应在不同模型家族中均具有泛化性。交叉条件分析表明，这种性能提升是由模型自身生成的推理轨迹所驱动的，且仅一个示例就足以引导出一种与类别无关的“缺失证据”推理模式。

    arXiv:2609.14860v1 Announce Type: cross  Abstract: Warning: This submission studies stereotypes and biases, and contains toxic and offensive examples, used for illustration purposes only.   Fairness benchmarks such as BBQ have become the de facto standard for fairness evaluation across major model families. We argue that these benchmarks are too easy to support their role: training Qwen 2.5 7B Base with Group Relative Policy Optimization (GRPO) on a single BBQ example, or placing that example in context as a one-shot demonstration for in-context learning (ICL), lifts mean BBQ accuracy from 79.9% to 92.9% and 99.0%, respectively, closing 80% of the gap to its large-scale RLHF counterpart (96.1%) with GRPO, and surpassing it with ICL. These effects generalize across model families. A cross-conditioning analysis shows the improvement is carried by the reasoning traces generated by the model, and one example suffices to elicit a category-agnostic ``missing evidence'' reasoning pattern. We 
    
[^79]: Dream-RSI：通过演化世界实现递归自我改进

    Dream-RSI: Recursive Self-Improvement through Evolving Worlds

    [https://arxiv.org/abs/2609.14858](https://arxiv.org/abs/2609.14858)

    Dream-RSI提出了一种递归自我改进的探索框架，通过轻量级编排层使探索显式可编程，并创新性地将积累的发现历史作为回放模拟器，在其中进行“做梦”式改进，从而突破传统探索策略难以适应扩展搜索空间的瓶颈。

    

    递归自我改进对于自主AI智能体正变得日益重要，其进展取决于在复杂领域中发现高价值解决方案。这一过程的驱动力是有效的探索，然而，管理和改进探索策略仍然是一个主要瓶颈。当前系统面临一个根本性困境：固定策略无法随搜索空间的扩展而进行适应，而在线策略优化则需要在漫长的rollout过程中，在巨大的元搜索空间中导航，同时应对延迟且代价高昂的反馈。我们提出了Dream-RSI，一个可扩展且能递归自我改进的探索框架。一个轻量级的编排层使探索过程变得显式且可编程，同时保持底层编码智能体不变。我们的关键洞察是：积累的发现历史可以作为对已实现搜索空间的回放模拟器。通过在由此构建的回放模拟器中进行“做梦”（dreaming）……

    arXiv:2609.14858v1 Announce Type: new  Abstract: Recursive self-improvement is becoming increasingly vital for autonomous AI agents, where progress hinges on discovering high-value solutions across complex domains. The driver of this process is effective exploration, however, managing and improving exploration strategies remains a major bottleneck. Current systems face a fundamental dilemma: fixed strategies fail to adapt as search spaces scale, while online policy optimization requires navigating vast meta-search spaces under delayed and expensive feedback over long-horizon rollouts. We introduce \textsc{Dream-RSI}, a framework for scalable and recursively self-improving exploration. A lightweight orchestration layer makes exploration explicit and programmable while leaving the underlying coding agent unchanged. Our key insight is that accumulated discovery history can serve as a replay simulator over the realized search space. By performing dreaming in the replay simulator constructe
    
[^80]: ModularRSI：模块化且可泛化的递归执行框架自我改进

    ModularRSI: Modular and Generalizable Recursive Harness Self-Improvement

    [https://arxiv.org/abs/2609.14857](https://arxiv.org/abs/2609.14857)

    提出ModularRSI，一个基准分离、对比式且模块化的递归自我改进框架，通过对比成功与失败经验并在模块层面定位和优化执行框架缺陷，实现可泛化的框架进化。

    

    最近的研究将递归自我改进（RSI）扩展到面向长时程编码与终端任务的智能体执行框架，使智能体能够从经验中改进其执行机制。然而，可泛化的执行框架RSI仍然具有挑战性。首先，在评估基准或其子集上演化执行框架，难以区分可复用的改进与针对特定基准的适应。其次，单轨迹更新可能将系统性的框架缺陷与特定实例的推理及解决方案细节混为一谈，导致所做修改难以迁移到未见任务。第三，在单体式框架中定位反复出现的行为缺陷十分困难，而整体框架优化可能将无关机制纠缠在一起，使归因和验证变得复杂。我们提出ModularRSI，一个基准分离、对比式且模块化的可泛化框架演化框架。ModularRSI通过对比成功与失……（摘要原文在此处截断）

    arXiv:2609.14857v1 Announce Type: new  Abstract: Recent work extends recursive self-improvement (RSI) to agent harnesses for long-horizon coding and terminal tasks, enabling agents to improve execution mechanisms from experience. However, generalizable harness RSI remains challenging. First, evolving harnesses on evaluation benchmarks or their subsets makes it difficult to distinguish reusable improvements from benchmark-specific adaptation. Second, single-trajectory updates can conflate systematic harness deficiencies with instance-specific reasoning and solution details, producing modifications that transfer poorly to unseen tasks. Third, localizing recurring behavioral deficiencies within monolithic harnesses is difficult, while whole-harness optimization can entangle unrelated mechanisms and complicate attribution and validation. We propose ModularRSI, a benchmark-disjoint, contrastive, and modular framework for generalizable harness evolution. ModularRSI contrasts successful and f
    
[^81]: 自编排语言模型：利用语义依赖实现高效推理

    Self-Orchestrating Language Models: Leveraging Semantic Dependence for Efficient Inference

    [https://arxiv.org/abs/2609.14850](https://arxiv.org/abs/2609.14850)

    该论文提出“自编排语言模型”概念，让语言模型通过在生成过程中标注token间的语义依赖关系来指导自身推理执行策略，并配合专用运行时系统实现自回归解码并行化、中间上下文驱逐和去噪顺序推导，达成帕累托最优的质量-效率权衡。

    

    大型语言模型（LLM）展现出令人印象深刻的能力，但其部署面临着显著的效率挑战。自回归解码带来较大的推理延迟，并且在低批量场景下未能充分利用硬件加速器。离散扩散模型可以并行生成，但如果缺乏多次扩散去噪步骤，就难以匹配自回归生成的质量。长上下文推理会造成内存瓶颈，即使是最先进的硬件加速器也倍感压力。我的核心论点是：语言模型可以通过在生成过程中标注语义依赖关系（即哪些token依赖于哪些其他token）来指导自身的推理执行策略。我将这类模型称为自编排语言模型。针对每个系统，我设计了一个运行时系统，根据这些标注信息来并行化自回归解码、驱逐中间上下文或推导去噪顺序，从而实现帕累托最优的质量-效率权衡。

    arXiv:2609.14850v1 Announce Type: new  Abstract: Large language models (LLMs) demonstrate impressive capabilities, but their deployment presents significant efficiency challenges. Autoregressive decoding imposes substantial inference latency and under-utilizes hardware accelerators in low batch size regimes. Discrete diffusion models can generate in parallel but struggle to match autoregressive quality without many diffusion denoising steps. Long-context reasoning creates memory bottlenecks that strain even state-of-the-art accelerators.   My thesis is that language models can direct their own inference execution strategy by annotating semantic dependence -- which tokens depend on which others -- in their generation. I call such models self-orchestrating language models. For each system, I design a runtime that acts on these annotations to parallelize autoregressive decoding, evict intermediate context, or derive denoising orders, achieving Pareto-optimal quality-efficiency trade-offs.
    
[^82]: 大语言模型作为神谕：人们在主观个人问题上对大语言模型的依赖

    LLMs as Oracles: Reliance on LLMs for Subjective Personal Questions

    [https://arxiv.org/abs/2609.14849](https://arxiv.org/abs/2609.14849)

    人们正日益将大语言模型当作主观个人问题上的“全知权威”来依赖且往往不自知，这种依赖随时间增长、在年轻用户中更普遍，其驱动因素在于人们对AI的认知和AI自身的行为。

    

    我们研究了人们如何将大语言模型视为“神谕”——即在主观个人问题上的全知权威。出于对用户自主性和福祉风险的担忧，我们开发了一套分类体系和基于大语言模型的测量方法，以大规模量化这种AI依赖形式，并理解人们如何将判断和决策外包给AI。将该分类体系应用于公开使用数据（来自WildChat和ThoughtTrace的6.8万条提示），我们发现“大语言模型作为神谕”的使用在2023至2026年间持续增长，且在年轻用户中更为普遍。我们进一步构建了一个隐私保护的数据捐赠工具来分析个人的纵向使用数据（来自52名参与者的14万条提示），同样发现了类似趋势。人们往往并未意识到自己正在以这种方式使用大语言模型，在看到我们工具的分析结果后，对这种行为表达了不满。最后，我们识别出大语言模型作为神谕被使用的两大驱动因素：人们对AI的认知以及AI自身的行为。

    arXiv:2609.14849v1 Announce Type: cross  Abstract: We characterize how people are turning to LLMs as oracles: all-knowing authorities on subjective personal questions. Motivated by risks to users' autonomy and well-being, we develop a typology and LLM-based methods to measure this form of AI reliance at scale and understand how people are offloading judgment and decision-making to AI. Applying our typology to public usage data (68K prompts from WildChat and ThoughtTrace), we find that LLM-as-oracle use has increased over time (2023-2026) and is more prevalent among younger users. We further build a privacy-preserving data donation tool to analyze individuals' longitudinal usage data (140K prompts from 52 participants), identifying similar trends. People are often unaware of their own LLM-as-oracle use, and express dissatisfaction with this behavior after seeing our tool's analysis. Finally, we identify two drivers of LLM-as-oracle use: people's perceptions of AI and the behavior of AI 
    
[^83]: Enemray：面向哈桑尼亚语的能力强大的语言模型

    Enemray: Toward Capable Language Models for Hassaniya

    [https://arxiv.org/abs/2609.14829](https://arxiv.org/abs/2609.14829)

    本文提出 Enemray——一个以哈桑尼亚语为核心的语言模型，通过稳定性—可塑性训练策略，在掌握哈桑尼亚语语言文化能力的同时保留通用推理、多语言与安全行为。

    

    我们介绍了 Enemray，一个以哈桑尼亚语（Hassaniya）为核心的语言模型，能够在哈桑尼亚语中实现通用交互。Enemray 围绕稳定性—可塑性目标进行训练：在获取强大的哈桑尼亚语语言与文化能力的同时，保留一个经过指令微调的能力强大模型的通用推理、多语言、指令遵循与安全行为。其开发流程将语言习得与行为专化分离开来：单独构建的持续预训练语料库提供对自然哈桑尼亚语和毛里塔尼亚文本的广泛接触；层选择性持续预训练学习一个紧凑的语言特定参数更新；该更新被迁移至指令微调参数空间中；监督后训练则进一步发展对话、文化、文学、任务导向和跨语言行为。监督语料库将精选的公开哈桑尼亚语与毛里塔尼亚资源与……（摘要此处截断）

    arXiv:2609.14829v1 Announce Type: cross  Abstract: We introduce Enemray, a Hassaniya-centric language model that enables general-purpose interaction in Hassaniya. Enemray is trained around a stability--plasticity objective: acquire strong Hassaniya linguistic and cultural competence while preserving the general reasoning, multilingual, instruction-following, and safety behaviors of a capable instruction-tuned model. The development pipeline separates language acquisition from behavioral specialization. A separately assembled continual-pretraining corpus provides broad exposure to natural Hassaniya and Mauritanian text; layer-selective continual pretraining learns a compact language-specific parameter update; that update is transferred into the instruction-tuned parameter space; and supervised post-training develops conversational, cultural, literary, task-oriented, and cross-lingual behavior. The supervised corpus integrates selected public Hassaniya and Mauritanian resources with a su
    
[^84]: 路由而非修复：面向可靠临床LLM答案选择的依情境解码校正与轨迹门控路由器

    Route, Don't Fix: Regime-Dependent Decoding Correction and a Trajectory-Gated Router for Reliable Clinical LLM Answer Selection

    [https://arxiv.org/abs/2609.14825](https://arxiv.org/abs/2609.14825)

    提出ALTAS框架，仅通过一次前向传播读取末端熵与深层线性度（R²），即可为每个临床问题在贪心解码与深层轨迹校正之间动态路由，无需训练任何分类器或额外模块，从而提升临床LLM答案选择的可靠性。

    

    大语言模型（LLM）由于容易产生幻觉，常被认为在临床问答中不够安全。检索增强、微调和外部验证器都需要新的基础设施，这些设施须经临床治理部门审批，并可能增加延迟或额外的模型调用。推理时校正利用模型内部的logit信号，但固定的变换未必适用于每一个问题。一个在真实性压力测试中能将准确率提升约十个小时百分点的校正器，在临床多选题基准上却收效甚微——因为在这些基准上，指令微调使输出概率集中于单一答案，导致末端熵（terminal entropy）很低。我们提出了ALTAS，它通过一次前向传播读取末端熵和深层线性度（R²），从而为每个问题在贪心解码与深层轨迹校正之间做出选择。该方法无需训练任何分类器、探针或输出头；路由器基于候选答案……

    arXiv:2609.14825v1 Announce Type: cross  Abstract: Large language models (LLMs) are often deemed unsafe for clinical question answering because of their tendency to hallucinate. Retrieval augmentation, fine-tuning, and external verifiers require new infrastructure that clinical governance must approve and may add latency or extra model calls. Inference-time correction uses the model's internal logit signals, but a fixed transformation need not suit every question. A corrector that improves accuracy by about ten percentage points on a truthfulness stress test yields negligible gains on clinical multiple-choice benchmarks, where instruction tuning concentrates output probability on one answer and leaves low terminal entropy. We introduce ALTAS, which reads terminal entropy and late-layer linearity ($R^2$) from one forward pass to choose per question between greedy decoding and late-layer trajectory correction. No classifier, probe, or head is trained; the router operates on candidate-ans
    
[^85]: MedTRACE：用于循证决策的工具增强型多模态临床推理智能体

    MedTRACE: Tool-Augmented Multimodal Clinical Reasoning Agents for Evidence-Grounded Decision-Making

    [https://arxiv.org/abs/2609.14823](https://arxiv.org/abs/2609.14823)

    MedTRACE 提出了一种工具增强的多模态临床推理智能体，通过假设形成、工具感知推理与证据验证的迭代循环，动态调用视觉定位、证据检索和结构化解析工具，实现基于证据的临床决策。

    

    多模态临床决策需要对来自电子健康记录、医学影像和生理信号的异构证据进行可靠推理。现有模型通常将这些输入直接映射为诊断结果，而未能显式评估证据的充分性、工具使用需求或诊断不确定性。本文提出了 MedTRACE，一种面向循证决策的工具增强型多模态临床推理智能体。MedTRACE 使用模态特定的编码器构建统一的患者状态表示，并执行假设形成、工具感知推理和证据验证的迭代循环。它动态调用视觉定位、证据检索和结构化解析工具，以定位与诊断相关的区域、检索临床知识和相似病例，并提取结构化的检查结果。所获取的证据进入证据记忆，由一致性验证器确认……

    arXiv:2609.14823v1 Announce Type: new  Abstract: Multimodal clinical decision-making requires reliable reasoning over heterogeneous evidence from electronic health records, medical images, and physiological signals. Existing models typically map these inputs directly to diagnoses without explicitly assessing evidence sufficiency, tool-use requirements, or diagnostic uncertainty. This paper presents MedTRACE, a tool-augmented multimodal clinical reasoning agent for evidence-grounded decision-making. MedTRACE uses modality-specific encoders to construct a unified patient-state representation and performs an iterative loop of hypothesis formation, toolaware deliberation, and evidence verification. It dynamically invokes visual grounding, evidence retrieval, and structured parsing tools to locate diagnosis-relevant regions, retrieve clinical knowledge and similar cases, and extract structured findings. The acquired evidence enters an evidence memory, where a consistency verifier confirms o
    
[^86]: 医疗保健领域大型语言模型评估方法入门

    A primer on evaluation methods for large language models in healthcare

    [https://arxiv.org/abs/2609.14819](https://arxiv.org/abs/2609.14819)

    本文系统综述了医疗领域大语言模型评估的四大关键方面——研究设计原则、统计方法、能力评估和临床情境评估，并阐述了相关底层概念与潜在陷阱。

    

    大型语言模型（LLM）在医学领域的应用范围日益广泛，对其进行评估对于确保它们带来益处而非危害至关重要。由于多种原因，这种评估比传统机器学习更具挑战性，包括概率性和开放式的输出，以及随提示词设计和累积上下文而变化的行为。本综述涵盖了大语言模型评估的四个关键领域：研究设计原则、统计方法、能力评估和临床情境评估。能力评估考虑了不同的基准测试，包括选择题、智能体和多轮对话基准测试，以及诸如标记（token）使用量等操作性指标。临床情境评估则涉及确定自由文本输出的准确性，例如人工审查和大语言模型作为评判者（LLM-as-a-judge）的方法，以及临床试验方法。在各个章节中，我们阐述了底层概念和潜在的陷阱，同时强调了其重要性。

    arXiv:2609.14819v1 Announce Type: cross  Abstract: Large language models (LLMs) have a growing range of applications in medicine, and their evaluation is critical for ensuring they provide benefit and not harm. This evaluation can be more challenging than traditional machine learning for many reasons, including probabilistic and open-ended outputs, and behavior that shifts with prompt design and accumulated context. This review covers four key areas of LLM evaluation: principles of study design, statistical methods, capability evaluation and clinical context evaluation. Capability evaluation considers different benchmarks, including multiple-choice, agentic and multi-turn benchmarks, alongside operational metrics like token usage. Clinical context evaluation addresses establishing accuracy of free text outputs, such as human review and LLM-as-a-judge, and clinical trial approaches. Across sections, we describe underlying concepts and potential pitfalls, while emphasizing the importance
    
[^87]: 低成本声调评估：面向大规模多语言文本转语音的无参考词汇声调度量指标

    Tone on a Budget: A Reference-Free Metric for Lexical Tone in Massively Multilingual Text-to-Speech

    [https://arxiv.org/abs/2609.14817](https://arxiv.org/abs/2609.14817)

    本文提出 DunDun，一种无需参考录音和声调标注语料库的无参考词汇声调自动评估指标，通过从输入文本的变音符号读取标准高/中/低调序列并与音频音高轨迹对比，解决了 TTS 标准指标 CER 忽略声调标记导致声调错误无法被检测的问题。

    

    在约鲁巴语中，仅凭音高就能区分 ọkọ（丈夫，中调）、ọkọ̀（车辆，低调）和 ọ́kọ́（锄头，高调）——这些变音符号本身就是声调标记。然而，作为文本转语音（TTS）标准自动评估指标的字符错误率（CER），在实际中是基于丢失这些标记的语音识别（ASR）输出计算的：一个合成器可以在 CER 上表现优异，却仍然把“丈夫”读成“车辆”。我们提出 DunDun——以仅凭音高“说话”的约鲁巴语对话鼓 dùndún 命名——这是一种自动化、无参考的词汇声调度量指标，无需任何声调标注语料库。黄金标准的高/中/低调序列直接从输入文本的变音符号中读取（在 TTS 场景下该文本必然存在，因此无需参考录音）；预测则来自音频的音高轨迹。我们通过三种方式进行了验证：用 PSOLA 重合成抹平音高后 DunDun 评分崩溃而 CER 不变；在 300 条母语者录音的答案中将高调与低调互换后……（摘要内容被截断）

    arXiv:2609.14817v1 Announce Type: new  Abstract: In Yor\`ub\'a, pitch alone separates \d{o}k\d{o} (husband, Mid), \d{o}k\d{\`o} (vehicle, Low), and \d{o}k\d{\'o} (hoe, High) -- the diacritics ARE the tone marks. Yet character error rate (CER), the standard automated metric for text-to-speech (TTS), is in practice computed from ASR output that drops those marks: a synthesizer can ace CER and still say vehicle for husband. We introduce DunDun -- named for the d\`und\'un, the Yor\`ub\'a talking drum that speaks through pitch alone -- an automated, reference-free lexical-tone metric that needs no tone-labelled corpus. The gold High/Mid/Low sequence is read from the input text's diacritics (in TTS that text exists by construction, so no reference recording is needed); the prediction comes from the audio's pitch track. We validate three ways. Flattening pitch with PSOLA resynthesis collapses DunDun while CER does not move. Inverting High and Low in the answer key of 300 native recordings dri
    
[^88]: 加密会计基准：评估前沿模型与开放权重模型在加密资产会计任务上的表现

    Crypto Accounting Bench: Evaluating Frontier and Open-Weight Models on Crypto-Asset Accounting Tasks

    [https://arxiv.org/abs/2609.14811](https://arxiv.org/abs/2609.14811)

    本文提出加密会计基准（CAB），包含来自7个匿名化组织的118个评估任务，用于评估12个前沿及开放权重语言模型能否准确重构加密资产交易的完整会计分录。

    

    我们提出了加密会计基准（CAB），这是一个用于评估前沿模型和开放权重语言模型能否重构组织为加密资产交易实际记录的完整会计分录的基准。CAB包含来自7个匿名化组织的118个评估任务。每个任务结合了交易机制、资产数量和基础货币价值、钱包与法人实体背景、交易对手证据、相关交易环节、重复性、税务批次证据，以及该组织完整的会计科目表。目标输出是一个平衡的结构化分录，包含所有必需的科目、借贷方向、金额、货币以及全精度的资产数量。我们评估了12个模型，涵盖专有前沿系统和开放权重模型，每个任务进行3次独立尝试，共产生4,248条轨迹。我们报告了3个指标：平均分数、Best@3和Pass@3。Pass@3是3次尝试中至少有1次成功的任务比例。

    arXiv:2609.14811v1 Announce Type: new  Abstract: We introduce Crypto Accounting Bench (CAB), a benchmark for assessing whether frontier and open-weight language models can reconstruct the complete journal entry that an organization actually posted for a crypto-asset transaction. CAB contains 118 evaluation tasks drawn from 7 pseudonymized organizations. Each task combines transaction mechanics, asset quantities and base-currency values, wallet and legal-entity context, counterparty evidence, related transaction legs, recurrence, tax-lot evidence, and the organization's complete chart of accounts. The target is a balanced structured entry with every required account, side, amount, currency, and full-precision asset quantity. We evaluate 12 models spanning proprietary frontier systems and open-weight releases over 3 independent attempts per task, producing 4,248 trajectories. We report 3 metrics: Mean Score, Best@3, and Pass@3. Pass@3 is the fraction of tasks with at least 1 of 3 attempt
    
[^89]: 注意你偏爱哪只鸟：机器翻译元评估中充分性-流利度平衡的参数化

    Mind Which Bird You Favour: Parameterizing Adequacy-Fluency Balance in Meta-Evaluation of Machine Translation

    [https://arxiv.org/abs/2609.14795](https://arxiv.org/abs/2609.14795)

    该论文将机器翻译元评估中充分性与流利度的平衡参数化为可调节的选择，提出了一种具有理论保证和剪枝机制的精确优化算法，通过最小化失真的方式对翻译系统重新加权来实现目标平衡。

    

    机器翻译元评估中存在一个权衡：优先考虑与充分性还是流利度的对齐。这种平衡取决于元评估数据集中翻译系统的组合。该系统集合是一个经过筛选的小样本，其特征在不同年份和语言对之间变化剧烈，并不能代表真实的系统分布。因此，充分性-流利度平衡往往不具代表性且易发生变化。对于敏感领域而言，控制这种平衡至关重要。我们将这种平衡揭示为一个可调节的选择。为实现目标平衡，我们在最小化与均匀加权失真的同时对现有系统重新加权，确保被评估的系统保持真实且有代表性。我们提供了一个具有理论保证和剪枝机制的精确优化算法来计算这些权重。为验证元评估的内部一致性，我们设计了一个评分器增强（摘要在此处被截断）

    arXiv:2609.14795v1 Announce Type: cross  Abstract: There is a tradeoff in machine translation meta-evaluation between prioritizing alignment with adequacy versus fluency. The balance depends on the combination of translation systems in the meta-evaluation dataset. This system set is a small, filtered sample whose characteristics change heavily across years and language pairs; it does not represent the true system distribution. Consequently, the adequacy-fluency balance is often unrepresentative and subject to change. For sensitive domains, controlling this balance is critical. We expose this balance as a tunable choice. To achieve a target balance, we reweight existing systems while minimizing distortion from uniform weighting, ensuring the evaluated systems remain real and representative. We provide an exact optimization algorithm with theoretical guarantees and pruning mechanisms to compute these weights. To validate meta-evaluation internal consistency, we design a scorer-augmentati
    
[^90]: 面向鲁棒跨传感器材质识别的语言引导表示学习

    Language-Guided Representation Learning for Robust Cross-Sensor Material Recognition

    [https://arxiv.org/abs/2609.14783](https://arxiv.org/abs/2609.14783)

    提出了一种语言引导的蒸馏框架，利用语言作为跨传感硬件不变的传感器无关监督信号，并通过构建包含39K样本的人工标注触觉-语言数据集，学习对传感器鲁棒的触觉表示，从而提升跨传感器材质识别的泛化能力。

    

    机器人需要触觉来安全可靠地操作物体，因为许多属性（如柔软度、纹理和接触稳定性）很难仅通过视觉来推断。然而，由于光学特性、弹性体特性和光照条件的差异，基于视觉的触觉传感器对同一材料会产生不同的观测结果，导致在单一或多个传感器上训练时泛化能力较差。我们提出了一种语言引导的蒸馏框架，用于学习对传感器鲁棒的触觉表示。语言编码了触觉的高级语义属性（如粗糙、柔软、滑腻），这些属性在不同传感硬件之间保持不变，从而提供了一种天然的传感器无关的监督信号。我们构建了一个包含39K样本的触觉-语言数据集，带有由人工标注的材料标签，并训练了一个触觉编码器，将传感器特定的触觉图像与语言嵌入在共享语义空间中对齐。我们在少样本学习任务上对该方法进行了评估。

    arXiv:2609.14783v1 Announce Type: cross  Abstract: Robots need touch to manipulate objects safely and reliably, as many properties, such as softness, texture, and contact stability, are hard to infer from vision alone. However, vision-based tactile sensors yield different observations of the same material due to variations in optics, elastomer properties, and illumination, leading to poor generalization when trained on a single or multiple sensors. We propose a language-guided distillation framework for learning sensor-robust tactile representations. Language encodes high-level semantic properties of touch (e.g., rough, soft, slippery) that remain invariant across sensing hardware, providing a natural sensor-agnostic supervisory signal. We construct a 39K-sample touch-language dataset with human-annotated material labels and train a tactile encoder to align sensor-specific tactile images with language embeddings in a shared semantic space. We evaluate our approach for few-shot learning
    
[^91]: Func-R1：激励多模态大语言模型中的数学函数推理

    Func-R1: Incentivizing Mathematical Function Reasoning in Multimodal Large Language Models

    [https://arxiv.org/abs/2609.14779](https://arxiv.org/abs/2609.14779)

    该论文揭示了多模态大语言模型在数学函数推理中存在忽视视觉线索的模态干扰现象，并提出Func-R1模型，通过显式解耦架构、分层后训练框架和感知对齐理论优化（PATO）策略，协同实现精确视觉感知与严谨逻辑推理的融合。

    

    在视觉语境中进行深思熟虑的数学推理是先进多模态大语言模型（MLLM）的重要标志，这需要对感知基础与符号逻辑进行复杂的综合。然而，在数学函数领域，我们的研究揭示了一个关键的模态干扰现象：即使是先进的模型，在执行文本计算推理时，也倾向于忽视或误解关键的视觉线索。为了应对这一挑战，我们提出了Func-R1，它协同地融合了精确的视觉感知与严谨的逻辑推理。具体而言，基于显式解耦的架构，我们采用分层后训练框架来逐步识别关键视觉证据并进行深入的理论推理。此外，我们提出了感知对齐理论优化（PATO）策略，以引导策略更新朝着内化基本……

    arXiv:2609.14779v1 Announce Type: new  Abstract: Performing deliberate mathematical reasoning in visual contexts is a hallmark of advanced Multimodal Large Language Models (MLLMs) and requires a sophisticated synthesis of perceptual grounding and symbolic logic. However, in the realm of mathematical functions, our investigation reveals a critical modality interference phenomenon: even advanced models, while performing textual computational reasoning, tend to disregard or misinterpret essential visual cues. To address this challenge, we propose Func-R1, which synergistically harmonizes precise visual perception and rigorous logical reasoning. Concretely, built upon an explicitly decoupled architecture, we employ a hierarchical post-training framework to progressively identify critical visual evidence and conduct in-depth theoretical reasoning. Furthermore, the Perception-Aligned Theoretic Optimization (PATO) strategy is proposed to steer policy updating towards internalizing fundamental
    
[^92]: Pull：面向有状态LLM对话的工作记忆惰性物化

    Pull: Lazy Materialization of Working Memory for Stateful LLM Conversations

    [https://arxiv.org/abs/2609.14773](https://arxiv.org/abs/2609.14773)

    提出Pull会话路由器，通过可逆的惰性物化机制让LLM只加载所需的历史对话轮次，在保证回答质量的前提下将单跳任务的每查询上下文token减少75.1%、多跳任务减少72.0%。

    

    随着LLM对话增长到数百轮，全上下文注入会带来O(N²)的累积token成本，而有损摘要或硬截断则会不可逆地丢弃历史状态。我们提出了Pull，一个会话路由器，它通过本地的、确定性的Purifier（零LLM调用、毫秒级延迟）维护一个可寻址的元数据目录。在查询时，LLM只惰性物化它所需的轮次；未物化的轮次仍然可访问但处于折叠状态。与不可逆压缩不同，Pull的物化是可逆的，后续查询可以展开任何折叠的轮次。在LoCoEval（128个对话，12,780轮）上，Pull在单跳任务中将每查询上下文token（第二阶段）减少了75.1%，且质量相当（Δ = -0.002，不显著）；在多跳任务中减少了72.0%，且无质量损失（Δ = +0.017）。一项受控路由基准测试（7,831个查询 × 10种方法）表明，实体生命周期……（摘要此处被截断）

    arXiv:2609.14773v1 Announce Type: new  Abstract: As LLM conversations grow to hundreds of turns, full-context injection incurs $O(N^2)$ cumulative token costs, while lossy summarization or hard truncation irreversibly discards historical state. We propose Pull, a session router that maintains an addressable metadata directory via a local, deterministic Purifier (zero LLM calls, millisecond-level latency). At query time, the LLM lazily materializes only the turns it needs; unmaterialized turns remain accessible but collapsed. Unlike irreversible compression, Pull's materialization is reversible; subsequent queries can expand any collapsed turn. On LoCoEval (128 conversations, 12,780 turns), Pull reduces per-query context tokens (Phase 2) by 75.1 percent on single-hop tasks with equivalent quality ($\Delta = -0.002$, n.s.) and by 72.0 percent on multi-hop tasks with no quality loss ($\Delta = +0.017$). A controlled routing benchmark (7,831 queries x 10 methods) shows that entity lifecycl
    
[^93]: 这个论断有多宽泛？对NLP研究中泛化表述的映射分析

    How broad is that claim? Mapping Generalisation in NLP Research

    [https://arxiv.org/abs/2609.14770](https://arxiv.org/abs/2609.14770)

    该论文提出了科学领域泛化表述分类体系NLPGenX、基于大语言模型的自动分类框架NLPGenA以及大规模标注数据集NLPGens，用于自动检测NLP研究论文中对泛化表述的过度使用和可能存在的表述偏差。

    

    泛化表述在科学交流中十分常见，尽管它们在语义上往往是模糊的。为了帮助检测对泛化表述的过度依赖以及可能对科学发现造成的歪曲，需要一种自动化方法来识别论断并根据其泛化程度进行分类。我们引入了一个全面的科学领域泛化表述分类体系NLPGenX，它根据论断的泛化程度及其在文本中的表述框架对其进行标注。我们通过一个基于大语言模型（LLM）的框架NLPGenA将该分类体系操作化，该框架能够自动将科学文章中的句子分类为5种不同的泛化类别。我们通过人工标注者对该框架进行了验证，并利用该框架构建了一个大规模的NLP论文标注数据集NLPGens，其中包含泛化程度标注以及关于模糊限定词（hedging）和模糊描述词的辅助标签。我们使用NLPGens分析泛化表述的使用情况……（原文摘要在此处截断）

    arXiv:2609.14770v1 Announce Type: cross  Abstract: Generalisations are common in scientific communication, even though they are semantically ambiguous. An automated method is needed to identify and categorise claims according to their level of generalisation, in order help detect an over-reliance on generalisations and possible misrepresentations of scientific findings. We introduce a comprehensive taxonomy of generalisations in the scientific domain, NLPGenX, which labels claims according to their level of generality and framing within the text. We operationalise this taxonomy with an LLM-powered framework, NLPGenA, that automatically classifies sentences from scientific articles into 5 different generalisation classes. We validate our framework with human annotators and use the framework to construct a large-scale dataset of NLP papers annotated according to generality, with auxiliary labels for hedging and vague descriptors (NLPGens). We use NLPGens to analyse the use of generalisat
    
[^94]: LLM智能体团队中的回环权威：扁平化与层级化协调的配对实验

    Loop-Back Authority in LLM Agent Teams: A Paired Experiment on Flat and Hierarchical Coordination

    [https://arxiv.org/abs/2609.14767](https://arxiv.org/abs/2609.14767)

    实验表明，在开放式任务中，移除管理者对工作者输出的否决权威反而能提高LLM多智能体团队的输出质量。

    

    层级化编排是一种管理者智能体可以审查工作者输出并要求其返工修改的协调模式，这是生产级多智能体LLM框架中的默认协调模式。经典组织理论预测权威链条能加快在决定性输出上的收敛速度；而关于谄媚行为和思维退化的研究预测权威性的批评会使LLM输出变差。以往的比较研究是在具有可验证答案的任务上比较整个框架，权威链接尚未在开放式工作中得到检验。我们提出了一个配对实验，固定五个LLM智能体的角色、提示词、工具、模型和数据，仅改变一个环节：管理者是否可以否决工作者的输出并要求其修改。在43个配对产品和86次商业智能报告任务的运行中，五模型评审小组和确定性规范检查对每份报告进行评分。扁平化组织在实用性上得分更高。

    arXiv:2609.14767v1 Announce Type: cross  Abstract: Hierarchical orchestration, in which a Manager agent reviews worker output and can send it back for revision, is the default coordination pattern in production multi-agent LLM frameworks. Classical organizational theory predicts that the authority link speeds convergence on decisive output; work on sycophancy and Degeneration-of-Thought predicts that authoritative critique makes LLM output worse. Prior comparisons vary whole frameworks on tasks with checkable answers, leaving the authority link untested on open-ended work. We present a paired experiment that holds five LLM agents, their roles, prompts, tools, models, and data fixed and varies one link: whether the Manager may reject a worker's output and oblige a revision. Across 43 paired products and 86 runs of a business-intelligence reporting task, a five-model judge panel and a deterministic specification check score every report. The flat organization scores higher on Utility (d 
    
[^95]: 拒绝只读取模型所知内容的一部分：跨模型家族的有害性键控路由及其例外

    Refusal Reads Only a Slice of What the Model Knows: Harm-Keyed Routing and Its Exceptions Across Model Families

    [https://arxiv.org/abs/2609.14759](https://arxiv.org/abs/2609.14759)

    该研究揭示了模型拒绝有害请求的机制与道德理解能力是相互分离的——道德理解源于预训练中结晶形成的低秩子空间，而拒绝门则是后训练写入狭窄控制令牌通道的新构建，且与道德判断正交，表明拒绝行为只读取了模型所知内容的一小部分。

    

    预训练后施加的对齐在可测量的意义上是浅层的：模型残差流中的单一方向可以被编辑掉，模型便会停止拒绝有害请求。但这一事实只说明拒绝机制能被多容易地移除，而不能说明拒绝决策最初读取的是什么。我们探究它究竟读取了什么，并将其与模型所理解的内容区分开来。在跨越三个家族的四个开源权重模型中，道德理解是预训练的原生产物：一个低秩道德子空间在预训练过程中结晶形成，而对齐只是将其旋转一次，并未重建它。相比之下，拒绝门是一个全新的后训练构建，仅有微弱的预训练前驱，它被写入一个狭窄的控制令牌通道，在该通道中，拒绝决策与道德判断决策相互正交。核心结果是因果性的，来自单一模型OLMo-3：通过嵌套交换秩扫描，依次修补道德子空间越来越大的切片……

    arXiv:2609.14759v1 Announce Type: cross  Abstract: Alignment applied after pretraining is shallow in a measurable way: a single direction in a model's residual stream can be edited out, and the model stops refusing harmful requests. That fact says how easily refusal can be removed, not what the refusal decision was reading in the first place. We ask what it reads, and we separate that from what the model comprehends. Across four open-weight models spanning three families, moral comprehension is native to pretraining: a low-rank moral subspace crystallizes during pretraining, and alignment rotates it once without rebuilding it. The refusal gate, in contrast, is a fresh post-training construction with only a weak pretraining precursor, written into a narrow control-token channel where the refusal decision is orthogonal to the moral-judgment decision. The central result is causal and comes from one model, OLMo-3. A nested interchange rank sweep patches successively larger slices of the mo
    
[^96]: 工具故障后的捏造：工具增强的智能体会断言其工具并未返回的值

    Fabrication After Tool Failure: Tool-Augmented Agents Assert Values Their Tools Did Not Return

    [https://arxiv.org/abs/2609.14758](https://arxiv.org/abs/2609.14758)

    该研究构建了一个涵盖1,024个条目的基准测试，首次系统揭示了工具增强语言模型在工具故障后的不诚实行为：当故障未被明确标记时（如返回status:ok但值损坏），模型捏造数值的比例高达45.3%，而当故障被明确标记为status:error时捏造行为完全消失，表明故障信号的显式性是决定模型诚信的关键因素。

    

    工具增强的语言模型通常以是否得到正确答案来评估，而不是以当工具未能提供答案时它们是否如实报告来评估。我们通过一个包含1,024个条目、涵盖16个内部系统领域和八种工具故障类型的基准测试来隔离研究这种故障后的决策行为，其中工具调用被强制执行，且返回的有效载荷被保证不可用。在部署风格的系统提示下，14.10%的响应是不诚实的：模型要么断言有效载荷无法支持的值，要么在拒绝回答时引用捏造的策略或能力限制。这一不诚实率几乎完全取决于故障是否被明确发出信号。当工具返回status:error时，不诚实行为完全不存在（0.0%）；而当工具返回status:ok但其值为被编辑、损坏、过期、格式错误、空或截断时，不诚实率高达45.3%。这种行为并非我们提示词的产物：它在中性提示下同样出现（10.17%）……

    arXiv:2609.14758v1 Announce Type: cross  Abstract: Tool-augmented language models are evaluated on whether they reach the right answer, not on whether they report honestly when a tool fails to supply one. We isolate this post-failure decision with a benchmark of 1,024 items spanning 16 internal-system domains and eight tool-failure types, in which a tool call is enforced and the returned payload is guaranteed to be unusable. Under a deployment-style system prompt, 14.10% of responses are dishonest: the model either asserts a value the payload cannot support or declines while citing a fabricated policy or capability limit. The rate is governed almost entirely by whether the failure is signalled. When the tool returns status:error, dishonesty is absent (0.0%); when it returns status:ok with a redacted, corrupted, stale, malformed, empty or truncated value, dishonesty reaches 45.3%. The behaviour is not an artefact of our prompts: it appears under a neutral prompt (10.17%) and under the s
    
[^97]: 在信任可解释性测量工具的判定之前先对其进行校准

    Calibrating Interpretability Instruments Before Trusting Their Verdicts

    [https://arxiv.org/abs/2609.14754](https://arxiv.org/abs/2609.14754)

    该论文指出可解释性研究中的测量工具会以特定、可诊断的方式静默失效并返回看似合理的数字，并通过一项关于拒答与道德表征的因果可解释性研究记录了六种典型失效模式，强调在采信工具的判定之前必须先对其进行校准。

    

    关于大语言模型（LLM）内部机制的因果论断依赖于测量。这些测量可能包括投影、余弦相似度、消融差值或交换补丁等。这些测量会以特定的、可诊断的方式失效，返回一个看似合理的数字而非错误信息，因此一个失效的测量工具很容易被误读为一个研究发现。例如：协方差匹配的零假设可能会饱和，以至于每个方向看起来都很典型；逐头归因在重排归一化架构上可能会超出真实残差写入值的三倍；交换补丁因其结果被固定在上限而变得符号混乱；“读取自”的判定可能是测量超过了模型已经做出决策的层所产生的伪影。本笔记记录了来自一项针对拒答行为与道德表征的因果可解释性研究项目中的六种此类失效情况，涵盖多篇论文和由四个开源权重模型组成的模型面板；每种失效模式都在这四个模型中的一个或两个上得到了验证。对于每种……

    arXiv:2609.14754v1 Announce Type: cross  Abstract: Causal claims about large language model (LLM) internals rest on measurements. Those might include a projection, a cosine, an ablation delta, or an interchange patch among others. These measurements fail in specific, diagnosable ways that return a plausible number instead of an error, so a broken instrument can easily read as a finding. A covariance-matched null can saturate until every direction looks typical, a per-head attribution can overshoot the true residual write threefold on reordered-normalization architectures, an interchange patch can go sign-chaotic because its outcome is pinned at a ceiling, or a read-from verdict can be an artifact of measuring past the layer where the model already decided. This note documents six such failures from a causal interpretability program on refusal and moral representation, spanning several papers and a four-model open-weight panel; each mode is established on one or two of the four. For eac
    
[^98]: 量化语音-文本语言模型中的生成模态差距

    Quantifying the Generation Modality Gap in Speech-Text Language Models

    [https://arxiv.org/abs/2609.14743](https://arxiv.org/abs/2609.14743)

    该论文构建了一个统一的生成式评估套件，在匹配的数据分布和评估设置下，从语义连贯性、语音结构、声学质量等多个维度量化了纯语音、纯文本和语音-文本语言模型之间的生成模态差距。

    

    纯语音语言模型在生成连贯内容方面往往落后于文本和语音-文本语言模型，但由于语音和文本系统通常使用不同的指标进行评估、在不同的数据上训练，这种差距难以量化。我们研究了一系列口语语言模型中的语音-文本模态差距，这些模型基于流匹配技术生成连续声学特征。我们构建了一个统一的基于生成的评估套件，用于比较在匹配的数据分布上训练、并在匹配的生成设置下评估的纯语音、纯文本和语音-文本语言模型。我们从多个维度评估生成的续写内容：语义连贯性（通过转录生成的语音并使用参考语言模型进行评分来衡量）、局部语音结构（通过音素n-gram分布统计来衡量）、说话人一致性与声学质量，以及基于情感的分布指标。

    arXiv:2609.14743v1 Announce Type: new  Abstract: Pure speech language models often lag behind text and speech-text language models in generating coherent content, but this gap is difficult to quantify because speech and text systems are typically evaluated with different metrics and trained on different data. We study the speech-text modality gap in a family of spoken language models, based on flow matching for continuous acoustic feature generation. We construct a unified generation-based evaluation suite that compares speech-only, text-only, and speech-text language models trained on matched data distributions and evaluated in matched generation settings. We evaluate generated continuations along multiple dimensions: semantic coherence, measured by transcribing generated speech and scoring it with a reference language model; local phonetic structure, measured by phone n-gram distributional statistics; speaker consistency and acoustic quality; and emotion-based distributional metrics.
    
[^99]: 构建面向证据支撑与弃答的法律奖励模型

    Building Legal Reward Models for Grounding and Abstention

    [https://arxiv.org/abs/2609.14739](https://arxiv.org/abs/2609.14739)

    该论文提出了将法律问答数据集转换为情境偏好数据的框架，并构建了LegalRewardBench基准，用于评估嘈杂和证据不足检索条件下的有据法律生成，研究发现长度均衡的偏好数据增强能显著提升法律领域奖励模型的有据评估能力。

    

    大型语言模型正日益被应用于法律等高风险领域，在这类领域中，系统必须将其推理建立在检索到的证据之上，并在证据不足时选择弃答。然而，现有的奖励模型主要针对一般偏好而非情境证据支撑进行优化，这限制了它们在检索增强生成（RAG）场景下评估这些行为的能力。我们提出了一个将现有法律问答数据集转换为情境偏好数据的框架，并利用该框架构建了LegalRewardBench（LRB），这是一个用于在嘈杂和证据不足的检索条件下评估有据法律生成的基准。在通用和法律情境评估中，我们发现情境DPO能够改善有据评估的表现，但其性能对偏好数据的构建方式非常敏感。长度均衡的数据增强显著提升了有据法律评估的效果，其中表现最强的配置结合了……

    arXiv:2609.14739v1 Announce Type: cross  Abstract: Large language models are increasingly used in high-stakes domains such as law, where systems must ground their reasoning in retrieved evidence and abstain when that evidence is insufficient. However, existing reward models are largely optimised for general preferences rather than contextual grounding, limiting their ability to evaluate these behaviours in retrieval-augmented generation (RAG) settings. We introduce a framework for transforming existing legal QA datasets into contextual preference data and use it to construct LegalRewardBench (LRB), a benchmark for evaluating grounded legal generation under noisy and insufficient retrieval conditions. Across general and legal contextual evaluation, we find that contextual DPO improves grounded evaluation, but performance is sensitive to preference-data construction. Length-balanced augmentation substantially improves grounded legal evaluation, with the strongest configuration combining 
    
[^100]: CALICO：一个以人为中心、与编码手册对齐的标注系统

    CALICO: A Human-Centered, Codebook-Aligned System for Annotation

    [https://arxiv.org/abs/2609.14726](https://arxiv.org/abs/2609.14726)

    本文提出CALICO系统，将以人为中心的可编辑、可版本化提示词工作流与编码手册对齐，使非技术背景的领域专家能够诊断和优化基于大语言模型的标注行为。

    

    大语言模型越来越多地被用于扩展科学研究中的基于编码手册的标注，但现有工作流程在将领域专家的编码手册转化为可靠、可修订、可审计的提示词方面支持有限。提示词通常被视为固定指令并对标注者隐藏，这使得非技术背景的领域专家在模型输出违反编码手册准则时难以诊断和纠正模型行为。在本文中，我们提出了CALICO，这是一种以人为中心、与编码手册对齐的标注工作流程，它将提示词视为可编辑、可版本化和可优化的产物。CALICO集成了编码手册解析、提示词生成、结果检查、提示词版本管理、自然语言人类反馈，以及通过GEPA、MIPROv2和OPRO等现有优化器进行的标签监督提示词优化，同时还包括我们基于反思的优化器ReflectAgent。在实证方面，我们在特定领域场景中对CALICO进行了评估……

    arXiv:2609.14726v1 Announce Type: cross  Abstract: Large language models are increasingly used to scale codebook-based annotation in scientific research, but existing workflows provide limited support for translating domain experts' codebooks into reliable, revisable, and auditable prompts. Prompts are often treated as fixed instructions and hidden from annotators, making it difficult for non-technical domain experts to diagnose and correct model behavior when outputs violate codebook guidelines. In this paper, we present CALICO, a human-centered, codebook-aligned annotation workflow that treats prompts as editable, versioned, and optimizable artifacts. CALICO integrates codebook parsing, prompt generation, result inspection, prompt versioning, natural language human feedback, and label-supervised prompt optimization through existing optimizers such as GEPA, MIPROv2, and OPRO, together with our reflection-based optimizer, ReflectAgent. Empirically, we evaluate CALICO on domain-specific
    
[^101]: 1.5亿参数以下领域的深度与规模：JugnuLM-53M 对比 JugnuLM-110M

    Depth and Scale in the Sub-150M Regime: JugnuLM-53M vs JugnuLM-110M

    [https://arxiv.org/abs/2609.14715](https://arxiv.org/abs/2609.14715)

    在方法完全固定的情况下，仅通过将架构从53.5M扩展到深而窄的23层109.7M参数设计，JugnuLM-110M在更少训练token下全面超越53M模型，并以少约12%的参数匹敌GPT-X2-125M，证明该参数规模下的性能提升源于模型容量与深度而非数据量。

    

    我们将常规的1.5亿参数以下预训练配方从5350万参数扩展到1.097亿参数，保持方法不变（采用Qwen3风格解码器，包含分组查询注意力、RoPE、SwiGLU、RMSNorm、QK-Norm和z-loss；使用FineWeb-Edu数据），仅将模型架构几何形状改变为深而窄的23层×576隐藏维设计。更大的模型全面提升了性能——BLiMP从78.1提升至81.3，ARC-Easy从51.4提升至52.5，WikiText-2字节困惑度从2.04降低至1.95——其81.3%的BLiMP成绩在参数少约12%的情况下基本匹敌GPT-X2-125M（81.28）。值得注意的是，110M模型是在更少的训练token上实现这一点的（约8B对比12B），因此这一提升归因于模型容量和深度，而非更多的数据。两个模型都刻意采用常规设计；本报告是一个干净的规模扩展对照实验，也是关于在该参数规模下如何进一步提升模型的消融研究的基线层级（R0）。后续的消融阶梯包括：值残差（R1）和Muon优化器（R2）将ARC-Easy提升了……

    arXiv:2609.14715v1 Announce Type: new  Abstract: We scale our conventional sub-150M pretraining recipe from 53.5M to 109.7M parameters, holding the method fixed (Qwen3-style decoder with grouped-query attention, RoPE, SwiGLU, RMSNorm, QK-Norm, and a z-loss; FineWeb-Edu data) and changing only the geometry to a deep-and-thin 23-layer x 576-hidden design. The larger model improves across the board -- BLiMP 78.1 -> 81.3, ARC-Easy 51.4 -> 52.5, WikiText-2 byte-perplexity 2.04 -> 1.95 -- and its 81.3% BLiMP essentially matches GPT-X2-125M (81.28) at about 12% fewer parameters. Notably the 110M model achieves this on fewer training tokens (about 8B vs 12B), so the gain is attributable to capacity and depth, not more data. Both models are deliberately conventional; this report is a clean scaling control and the baseline rung (R0) of an ablation study of what further improves models in this regime. An ablation ladder follows: value residuals (R1) and the Muon optimizer (R2) lift ARC-Easy by a 
    
[^102]: 与城市对话：车外指称的多模态消解方法

    Speak to the City: Multimodal Resolution for Outside-the-Vehicle References

    [https://arxiv.org/abs/2609.14691](https://arxiv.org/abs/2609.14691)

    本文提出一个融合注视与语音的多模态OVR框架，通过VR数据采集流水线与轻量级Transformer网络实现车外兴趣点的精准识别，Top-1准确率达83.33%且计算开销低。

    

    随着自动驾驶汽车和扩展现实（XR）头显催生新颖的车内交互方式，无缝查询物理地标（即“车外指称”，OVR）仍因自车运动和指称歧义而充满挑战。我们提出了一个鲁棒的多模态OVR框架，融合用户注视与自然语言来识别兴趣点（POI）。为解决动态车辆数据稀缺的问题，我们开发了一个基于VR的流水线，将360度公共交通视频与车辆GNSS遥测数据同步。通过一项将乘客头部朝向映射到三维地理空间数字孪生的用户研究（N=46），我们捕获了真实的注视-语音行为。随后，我们训练了一个轻量级Transformer网络，利用大语言模型（LLM）将连续的空间注视向量与离散的语音语境动态对齐。实验结果表明，该方法具有高精度和低计算开销，达到了83.33%的Top-1准确率（87.72%的Top-2准确率）。

    arXiv:2609.14691v1 Announce Type: cross  Abstract: As autonomous vehicles and Extended Reality (XR) headsets enable novel in-car interactions, seamlessly querying physical landmarks, known as Outside-the-Vehicle Referencing (OVR), remains challenging due to ego-motion and referential ambiguity. We present a robust, multimodal OVR framework fusing user gaze and natural language to identify Points of Interest (POIs). To address the scarcity of dynamic vehicular data, we developed a VR-based pipeline synchronizing 360-degree transit videos with vehicle GNSS telemetry. Through a user study (N=46) mapping passenger head orientation into a 3D geospatial Digital Twin, we captured authentic gaze-speech behaviors. We subsequently trained a lightweight Transformer network, leveraging LLMs to dynamically align continuous spatial gaze vectors with discrete verbal context. Experimental results demonstrate high accuracy and low computational overhead, achieving an 83.33% Top-1 accuracy (87.72% Top-2
    
[^103]: 单一反馈系统无法放之四海而皆准：面向英国与尼日利亚的数据到文本驾驶辅导本地化研究

    One Feedback System Does Not Fit All: Localising Data-to-Text Driver Coaching for the United Kingdom and Nigeria

    [https://arxiv.org/abs/2609.14687](https://arxiv.org/abs/2609.14687)

    本文通过比较英国和尼日利亚两个独立开发的驾驶辅导系统，论证了数据到文本驾驶辅导系统不能采用通用流程，而必须根据当地驾驶员知识、常见风险、法规、基础设施和数据可用性进行本地化设计。

    

    数据到文本的驾驶辅导通常被呈现为一条从远程信息处理事件到驾驶建议的通用流程。本文认为，此类系统的内容需要进行本地化，因为其有用性和可信度取决于驾驶员的知识水平、常见风险、法规、基础设施以及可用的数据。本文通过追踪从内容选择、生成到实地评估全过程的需求，比较了在英国和尼日利亚独立开发的两个系统。英国系统优先考虑行程后的反思、与道路和地点情境相关联的解释，以及对语气敏感的措辞。尼日利亚系统则将有法律依据、基于检测事件生成的每日一次提示与每周一次的说服性报告相结合；针对正式培训和交通规则知识方面报告的不足，以及当地道路安全重点关注的问题，该系统突出安全教育和与酒精相关的风险。可靠的限速元数据在英国支持了超速反馈，而……

    arXiv:2609.14687v1 Announce Type: new  Abstract: Data-to-text driver coaching is often presented as a generic pipeline from telematics events to advice. This paper argues that its content requires localisation because usefulness and credibility depend on drivers' knowledge, prevalent risks, regulation, infrastructure, and available data. Two independently developed systems in the United Kingdom and Nigeria are compared by tracing requirements through content selection, generation, and field evaluation. The UK system prioritises post-trip reflection, explanations tied to road and place context, and tone-sensitive wording. The Nigerian system combines legally grounded, once-daily Tips based on detected events with weekly persuasive Reports; it foregrounds safety education and alcohol-related risk in response to reported gaps in formal training and traffic-rule knowledge, as well as local road-safety priorities. Reliable speed-limit metadata supported speeding feedback in the UK, whereas 
    
[^104]: 分岔提示词的花园：用户如何在故事生成中探索叙事空间

    The Garden of Forking Prompts: How Users Explore Narrative Space in Story Generation

    [https://arxiv.org/abs/2609.14677](https://arxiv.org/abs/2609.14677)

    该研究基于真实用户-聊天机器人对话数据，构建了包含275,635个故事生成提示词的WildStories数据集和24,291个编辑树的WildEdits集合，首次系统揭示了用户在故事生成中通过迭代编辑提示词来探索叙事空间的行为，弥补了现有静态评估基准无法捕捉此类探索性行为的不足。

    

    大语言模型（LLMs）改变了人们与故事互动的方式。通过分析公开的聊天机器人日志，我们可以看到，当用户生成故事时，他们会迭代地编辑提示词以探索叙事的可能性，例如调整角色、改变情节走向以及更换虚构世界。作为聚合数据，这些提示词代表了大规模的丰富创意偏好痕迹。然而，故事生成评估基准依赖于静态的、单次性的提示词，无法捕捉这种探索性行为。在本工作中，我们研究了用户在真实环境中如何修改连续的故事提示词。利用自然产生的用户-聊天机器人对话数据集，我们构建了WildStories——包含275,635个故事生成提示词的样本（标注了故事格式、提示词组成部分和明确性），以及WildEdits——包含24,291个编辑树的集合，用以建模用户如何迭代编辑基础故事提示词并探索分支化的故事可能性。

    arXiv:2609.14677v1 Announce Type: new  Abstract: Large language models (LLMs) have changed the way people engage with stories. Drawing on public chatbot logs, we can see that when users generate stories, they iteratively edit their prompts to explore narrative possibilities, adjusting characters, redirecting plots, and swapping fictional universes. As aggregated data, these prompts represent rich traces of creative preference at scale. Yet story generation evaluation benchmarks rely on static, one-shot prompts that cannot capture this exploratory behavior. In this work, we study how users revise consecutive story prompts in the wild. Using a dataset of naturally occurring user-chatbot conversations, we construct WildStories, a sample of 275,635 story generation prompts (labeled with story format, prompt components, and explicitness), and WildEdits, a collection of 24,291 edit trees that model how users iteratively edit base story prompts and explore branching story possibilities. From 
    
[^105]: 探索面对面对话中基于语音活动投影的多模态轮替线索

    Exploring Multimodal Turn-Taking Cues in Face-to-Face Conversation using Voice Activity Projection

    [https://arxiv.org/abs/2609.14666](https://arxiv.org/abs/2609.14666)

    通过将视线方向、头部运动、身体姿态和面部动作单元等视觉特征以多种融合策略引入语音活动投影（VAP）模型，能够超越仅依赖音频的方法，有效提升面对面对话中的轮替预测性能。

    

    轮替（turn-taking）是口语交互的基本组成部分，虽然人类自然地依赖言语和非言语信号，但对话系统通常仅依赖音频线索。本文研究了来自面对面对话的视觉特征是否能够超越仅用音频所能达到的效果，从而增强轮替预测。我们扩展了语音活动投影（VAP）模型——一个用于预测未来语音活动的基于Transformer的自监督模型——通过结合从Meta Seamless Interaction大规模双人面对面对话数据集中提取的视觉特征。这些视觉特征包括视线方向、头部运动、身体和手部姿态以及面部动作单元（FAU）。为了融合视觉特征，我们探索了特征拼接、交叉注意力融合、差分特征以及可训练的门控机制。结果表明，视觉信息相对于仅音频的基线方法提升了轮替预测的性能。

    arXiv:2609.14666v1 Announce Type: cross  Abstract: Turn-taking is a fundamental component of spoken interaction, and while humans naturally rely on both verbal and non-verbal signals, dialogue systems usually depend on audio cues alone. This paper investigates whether visual features from face-to-face conversations can enhance turn-taking prediction beyond what is achievable from audio-only. We extend the Voice Activity Projection (VAP) model, a self-supervised transformer-based model for predicting future voice activity, by incorporating visual features extracted from the large-scale Meta Seamless Interaction dataset of dyadic face-to-face conversations. The visual features include gaze direction, head movement, body and hand pose, and facial action units (FAU). For incorporating the visual features, we explore concatenation, cross-attention fusion, delta features, and trainable gating mechanisms. Results show that visual information improves performance over the audio-only baseline, 
    
[^106]: 通过价值引导偏好蒸馏：利用密集行为信号优化稀疏结果

    Optimizing Sparse Outcomes Through Dense Behavioral Signals via Value-Guided Preference Distillation

    [https://arxiv.org/abs/2609.14648](https://arxiv.org/abs/2609.14648)

    本文提出将长时程对话优化建模为多目标强化学习问题，通过多头价值模型预测用户行为向量，证明密集行为信号的标量化组合能有效优化稀疏长期结果，同时建立结合反事实用户模拟的安全框架以在部署前识别策略失败模式。

    

    多轮对话智能体的对齐通常被表述为匹配轮次级别的人类偏好，然而直接优化长期结果往往效果不佳且容易出现奖励欺骗。我们将长时程对话优化表述为一个多目标强化学习问题，并训练一个多头价值模型，该模型能够预测跨越多个前瞻时间范围的用户行为观测向量。我们的研究结果表明，密集辅助行为信号的标量化组合能够实现有效的信用分配和稀疏结果的优化。然而，优化无约束的单目标代理可能引发策略退化，当智能体暴露于真实用户时，这种退化是有害的。为了在部署前识别这些失败模式，我们建立了一个安全框架，结合反事实用户模拟与经过验证的对话级结果模型，以评估偏好加权和策略优化。

    arXiv:2609.14648v1 Announce Type: new  Abstract: Aligning multi-turn dialogue agents is usually framed as matching turn-level human preferences, yet direct optimization of long-term outcomes is often ineffective and prone to reward hacking. We formulate long-horizon dialogue optimization as a multi-objective reinforcement learning problem and train a multi-head value model that predicts a vector of observed user behaviors across multiple look-ahead horizons. Our findings demonstrate that a scalarized composite of dense auxiliary behavioral signals enables effective credit assignment and optimization of sparse outcomes. However, optimizing unconstrained single-objective proxies might induce policy degradations that are harmful when the agent is exposed to real users. To identify these failure modes prior to deployment, we establish a safety framework combining counterfactual user simulation with a validated dialogue-level outcome model to evaluate preference weightings and policy optimi
    
[^107]: CompCQR：面向免训练对话式搜索的组合式查询生成

    CompCQR: Compositional Query Generation for Training-Free Conversational Search

    [https://arxiv.org/abs/2609.14646](https://arxiv.org/abs/2609.14646)

    提出了CompCQR——一种免训练的对话式查询重写方法，利用检索器对内容排序敏感的观察，通过组合少量原子组件以极低的LLM调用量生成海量候选查询。

    

    与大语言模型（LLM）的多轮交互在信息检索场景中正变得越来越普遍。然而，用户查询往往是模糊的且依赖于上下文，使其不适合直接用作检索器的查询。对话式查询重写（CQR）通过将当前话语重写为一个基于对话历史的独立查询来解决这一问题。近期基于LLM的CQR方法取得了强劲的性能；然而，它们频繁调用LLM以及与下游检索器的错位问题仍然是挑战。在本工作中，我们从这样一个观察出发：检索器对内容排序高度敏感，仅仅重新排序相同的内容就可能导致检索覆盖范围和性能的变化。基于此，我们提出了一种新颖的免训练方法，通过组合一小组原子组件，以最少的LLM使用量生成大量查询。我们进一步应用LLM……（摘要原文截断）

    arXiv:2609.14646v1 Announce Type: cross  Abstract: Multi-turn interactions with LLMs are becoming increasingly common in information-seeking scenarios. However, user queries are often ambiguous and context-dependent, making them ill-suited for direct use as retriever queries. Conversational query reformulation (CQR) addresses this issue by rewriting the current utterance into a stand-alone query grounded in the dialogue history. Recent LLM-based CQR approaches achieve strong performance; however, their repeated LLM invocations and misalignment with downstream retrievers remain challenges. In this work, we begin from the observation that retrievers are highly sensitive to content ordering: simply reordering the same content can lead to changes in retrieval coverage and performance. Based on this, we propose a novel training-free method that generates a very large number of queries with minimal LLM usage by compositionally combining a small set of atomic components. We further apply LLM 
    
[^108]: 知道何时停止、何处重启：加速多轮智能体在线策略蒸馏

    Know When to Stop, Where to Restart: Accelerating Multi-Turn Agentic On-Policy Distillation

    [https://arxiv.org/abs/2609.14636](https://arxiv.org/abs/2609.14636)

    该论文发现多轮智能体训练中教师监督信号集中于每轮前缀、且教师认可的损失在时间上锁定于学生首个错误动作，据此提出STRIDE方法，通过停止-重启机制加速多轮智能体在线策略蒸馏。

    

    在线策略蒸馏（OPD）已成为将大型教师模型能力迁移到紧凑学生模型的标准方法。然而，其成本主要由学生模型的自回归采样主导，在多轮智能体设置中扩展性较差。现有的加速方法根据固定的、离线的预算来截断或重新定位监督信号，而忽视了教师信号可靠性在轨迹内部和轨迹之间存在的显著差异。我们在 τ²-bench 上的实证分析揭示了这种变化的清晰结构：有信息量的监督集中在每一轮的前缀部分，而且，对于多轮智能体训练最为重要的是，教师认可的跨轮损失在时间上锁定于学生模型的第一个错误动作，而非随轮次逐渐累积。基于这些发现，我们提出了 STRIDE（停止-重启在线策略蒸馏加速），它结合了两个互补的……

    arXiv:2609.14636v1 Announce Type: cross  Abstract: On-policy distillation (OPD) has become a standard approach for transferring capabilities from large teachers to compact students. Its cost, however, is dominated by autoregressive student rollouts and scales poorly in multi-turn agentic settings. Existing acceleration methods truncate or relocate the supervision signal according to fixed, offline budgets, despite substantial variation in teacher-signal reliability both within and across trajectories. Our empirical analysis on $\tau^2$-bench reveals a clear structure in this variation: informative supervision is concentrated in the prefix of each turn, and, most importantly for multi-turn agentic training, the cross-turn loss of teacher endorsement is temporally locked to the student's first erroneous action rather than accumulating gradually over turns. Building on these findings, we propose STRIDE (Stop-and-Restart on-policy Distillation acceleration), which combines two complementar
    
[^109]: 特定领域预训练特征与Transformer性能：来自阿拉伯语-英语语码转换数字语用建模的证据

    Domain-specific Pretraining Profile and Transformer Performance: Evidence from Modeling Digital Pragmatics in Arabic-English Code-switching

    [https://arxiv.org/abs/2609.14571](https://arxiv.org/abs/2609.14571)

    研究表明特定领域预训练特征对Transformer建模语码转换数字语用学至关重要，阿拉伯语专用模型MARBERT（Macro F1达0.85）在分类语境敏感语用功能上大幅优于多语言模型XLM-R（Macro F1仅0.52）。

    

    本研究强调了特定领域预训练特征（DSPP）在Transformer建模阿拉伯语-英语语码转换话语中数字语用学时的性能中所起的作用。研究评估了MARBERT和XLM-R（oBERTa），并以BERT作为通用基线模型。评估内容为模型对语码转换社交媒体话语中语境敏感语用功能的分类能力。研究通过Python收集了11695条独特的X（推特）帖子用于分析。该研究采用定量与定性相结合的NLP方法，遵循有监督的流水线流程。研究结果表明，MARBERT始终优于XLM-R，其验证集Macro F1从0.39提升至0.84，验证损失从0.55降至0.19。在独立测试集上，MARBERT取得了0.96的准确率、0.83的宏精确率、0.87的宏召回率和0.85的Macro F1，而XLM-R虽然测试准确率达到0.92，但Macro F1明显较低，仅为0.52。

    arXiv:2609.14571v1 Announce Type: new  Abstract: This study highlights the role of domain-specific pretraining profile (DSPP) in Transformer performance for modeling digital pragmatics in Arabic-English code-switched discourse. It evaluates MARBERT and XLM-R(oBERTa), with BERT serving as a general-purpose baseline. The models were evaluated on their ability to classify context-sensitive pragmatic functions in code-switched social-media discourse. 11695 unique X posts were collected via Python and utilized for the study. The study employs a quantitative and qualitative NLP approach, following a supervised pipeline. Findings unveil that MARBERT consistently surpasses XLM-R with validation Macro F1 increasing from 0.39 to 0.84 and validation loss decreasing from 0.55 to 0.19. On an independent test set, it achieved 0.96 accuracy, 0.83 macro precision, 0.87 macro recall, and 0.85 Macro F1, while XLM-R achieved 0.92 test accuracy but a substantially lower Macro F1 of 0.52. This was also sup
    
[^110]: 解耦多智能体大语言模型中的拓扑结构与多样性以实现多语言低资源情感检测

    Disentangling Topology and Diversity in Multi-Agent LLMs for Multilingual Low-Resource Emotion Detection

    [https://arxiv.org/abs/2609.14570](https://arxiv.org/abs/2609.14570)

    该论文首次将多智能体LLM系统中的推理拓扑结构与智能体间多样性来源解耦并独立研究，发现并行的学习型QLoRA专门化在九种语言的多语言低资源情感检测上表现最佳，且最优拓扑结构取决于所采用的多样性来源。

    

    多智能体大语言模型系统结合了多次推理调用，但先前的工作常常将“调用之间如何连接”与“调用之间如何实现多样化”这两个因素混为一谈。我们独立地研究这两个因素：推理拓扑结构和智能体间多样性的来源。在一个受控的 2×3 矩阵实验中，我们将并行聚合与顺序细化两种拓扑结构，分别与随机采样、角色提示和基于学习的 QLoRA 专门化三种多样性来源进行交叉组合，并在每个骨干模型内保持固定的三次调用预算和统一的输出协议。我们使用 Qwen2.5-14B-Instruct 和 Llama-3.1-8B-Instruct，在涵盖九种语言的多语言低资源情感检测任务上评估了全部六种配置。并行的学习型专门化在 Qwen 上表现最强，达到 52.83 Macro-F1，在 Llama 上达到 52.94。在 Qwen 上，该方法还超越了同骨干模型的零样本、少样本、思维链以及七次调用的自一致性基线。最优拓扑结构取决于多样性来源：顺序细化对随机采样和提示驱动的多样性更有帮助（摘要原文在此处截断）。

    arXiv:2609.14570v1 Announce Type: cross  Abstract: Multi-agent LLM systems combine multiple inference calls, but prior work often confounds how calls are connected with how they are diversified. We study these factors independently: inference topology and source of inter-agent diversity. In a controlled $2 \times 3$ matrix, we cross parallel aggregation and sequential refinement with stochastic sampling, role prompting, and learned QLoRA specialization, under a fixed three-call budget and output protocol within each backbone. Using Qwen2.5-14B-Instruct and Llama-3.1-8B-Instruct, we evaluate all six configurations on multilingual low-resource emotion detection across nine languages. Parallel learned specialization is strongest on Qwen at 52.83 Macro-F1 and reaches 52.94 on Llama. On Qwen it also exceeds same-backbone zero-shot, few-shot, CoT, and seven-call self-consistency baselines. The preferred topology depends on diversity source: sequential refinement helps stochastic and prompted
    
[^111]: TATK：面向基于大语言模型的序列推荐的三重感知Top-K学习与知识锚定验证

    TATK: Triple-Aware Top-K Learning with Knowledge-Grounded Verification for LLM-based Sequential Recommendation

    [https://arxiv.org/abs/2609.14565](https://arxiv.org/abs/2609.14565)

    提出TATK框架，通过将Top-K学习（对齐训练与排序效用）与知识锚定验证（单次前向传播后进行结构感知重排序）相结合，解决了基于大语言模型的序列推荐中全目录Top-K排序匹配不佳的问题。

    

    基于大语言模型的序列推荐系统通常将下一物品预测任务视为文本生成，但这种接口与全目录Top-K排序的匹配度较差。我们提出了TATK，这是一个三重感知框架，将Top-K学习（TKL）与知识锚定验证（KGV）相结合，应用于基于大语言模型的序列推荐。Top-K学习将上下文感知的元数据-知识图谱提示锚定与位置感知的Top-K奖励相结合，使训练目标与排序效用保持一致；知识锚定验证则在单次大语言模型前向传播之后，利用相同的元数据派生物品图，对Top-M候选物品进行结构感知的重排序。我们在Amazon Reviews 2023数据集中的乐器、CD与黑胶唱片以及视频游戏三个数据集上，采用匹配的R2ec风格全目录评估协议对TATK进行了评估。实验采用Gemma-2-2B-It和Qwen2.5-3B-Instruct作为骨干模型，与序列推荐、生成式、知识图谱增强和推理增强等基线方法进行比较，并包括组件分析……（摘要截断）

    arXiv:2609.14565v1 Announce Type: new  Abstract: LLM-based sequential recommenders usually cast next-item prediction as text generation, but this interface is poorly matched to full-catalog top-K ranking. We propose TATK, a Triple-Aware framework that couples Top-K Learning (TKL) with Knowledge-Grounded Verification (KGV) for LLM-based sequential recommendation. Top-K Learning combines context-aware metadata-KG prompt grounding with position-aware top-K rewards, aligning training with ranking utility; Knowledge-Grounded Verification then applies structure-aware reranking over the top-M candidates after a single LLM forward pass, using the same metadata-derived item graph. We evaluate TATK on Musical Instruments, CDs and Vinyl, and Video Games from Amazon Reviews 2023 under a matched R2ec-style full-catalog protocol. Experiments use Gemma-2-2B-It and Qwen2.5-3B-Instruct backbones, compare against sequential, generative, KG-augmented, and reasoning-enhanced baselines, and include compone
    
[^112]: Neyshekar：一个面向自动语音识别的开放波斯语朗读语音语料库

    Neyshekar: An Open Persian Read-Speech Corpus for Automatic Speech Recognition

    [https://arxiv.org/abs/2609.14542](https://arxiv.org/abs/2609.14542)

    发布了开放波斯语朗读语音语料库Neyshekar，涵盖正式与非正式语言、命名实体和长句，提供99小时经验证录音及可审计的说话人划分与评分者标注，为波斯语自动语音识别研究提供了高质量数据资源。

    

    Neyshekar是一个开放的波斯语朗读语音语料库，旨在覆盖正式与非正式语言、命名实体以及较长的语音表述。在第6版中，该语料库提供了来自190位贡献者的62,279条经验证的录音，总时长99.02小时，包含34,541条不同的录音提示文本。提示文本库由人工编写材料、语境化同形异义词以及经审核的语言模型生成文本组成。所有文本条目均使用shekar库进行规范化处理，该库同时支持正式与非正式波斯语，且每条提交的录音都依据统一的验证标准进行了审核。约24%的发布片段被自动分类器判定为非正式语体，但这些语体标签未经人工验证。语料库提供了条目级别的评分者标签以支持可复现的一致性估计，而每条片段采用不透明的贡献者标识符，使说话人不重叠的数据划分具有可审计性，并支持基于贡献者聚类的（不确定性估计）。

    arXiv:2609.14542v1 Announce Type: new  Abstract: Neyshekar is presented as an open Persian read-speech corpus designed for coverage of both formal and informal language, named entities, and longer utterances. In version 6, 62,279 validated recordings totalling 99.02 hours are provided from 190 contributors, with 34,541 distinct recorded prompts. The prompt pool was assembled from human-written material, contextualised homographs, and reviewed language-model-generated text. Text entries were normalised with the shekar library, which supports both formal and informal Persian, and every submitted recording was reviewed against a common validation rubric. About 24% of released clips are classified as informal by an automatic classifier; these register labels are not human-validated. Item-level rater labels are provided for reproducible agreement estimation, opaque per-clip contributor identifiers make the speaker-disjoint partitioning auditable and support contributor-clustered uncertainty
    
[^113]: 面向复述检测的参数高效量子自然语言处理：性能、鲁棒性与纠缠

    Parameter-Efficient Quantum NLP for Paraphrase Detection: Performance, Robustness, and Entanglement

    [https://arxiv.org/abs/2609.14529](https://arxiv.org/abs/2609.14529)

    本文提出一个仅含2,148个参数的10量子比特混合量子-经典变分电路用于复述检测，在统计上优于参数量匹配的经典基线并以更少参数超越DistilBERT，同时发现多量子比特纠缠是性能的主要驱动因素，并展现出对对抗样本的涌现鲁棒性。

    

    量子机器学习在自然语言任务上的严格实证验证仍然稀缺。我们评估了一个10量子比特的混合量子-经典变分电路（2,148个参数）用于复述检测，涵盖三个基准数据集：MRPC、Quora问题对（QQP）和对抗性PAWS。在QQP上（n = 10个随机种子），该电路达到75.53% ± 0.75%的准确率，在统计上显著优于参数量匹配的经典基线（DeepMLP：p = 0.015，Cohen's d = 1.20；F1：p < 0.001，d = 2.32），并以少31,191个参数的优势超越了DistilBERT-4bit。在MRPC上，最优的2层变体以低54,000个参数的成本达到了BERT-base准确率的92%。电路深度分析揭示了数据集与深度之间的缩放效应；通过Meyer-Wallach度量进行的纠缠分析确定多量子比特纠缠是主要的性能驱动因素（四个变体中r = 0.85）。在PAWS上的对抗性评估揭示了涌现的鲁棒性：98.2%的召回率。

    arXiv:2609.14529v1 Announce Type: cross  Abstract: Rigorous empirical validation of quantum machine learning on natural language tasks remains scarce. We evaluate a 10-qubit hybrid quantum-classical variational circuit (2,148 parameters) for paraphrase detection across three benchmarks: MRPC, Quora Question Pairs (QQP), and adversarial PAWS. On QQP (n = 10 seeds), the circuit achieves 75.53% +/- 0.75%. accuracy, statistically outperforming parameter-matched classical baselines (DeepMLP: p = 0.015, Cohen's d = 1.20; F1: p < 0.001, d = 2.32) and surpassing DistilBERT-4bit with 31,191 fewer parameters. On MRPC the optimal 2-layer variant reaches 92% of BERT-base accuracy at 54,000 lower parameter cost. Circuit depth analysis reveals a dataset-depth scaling effect; entanglement analysis via the Meyer-Wallach measure identifies multi-qubit entanglement as the primary performance driver (r = 0.85 across four variants). Adversarial evaluation on PAWS reveals emergent robustness: 98.2% recall 
    
[^114]: 图中的忒修斯：迈向可追溯的多跳图导航

    Theseus in the Graph: Towards Traceable Multi-Hop Graph Navigation

    [https://arxiv.org/abs/2609.14528](https://arxiv.org/abs/2609.14528)

    本文将多跳知识图谱问答重新定义为以问题为条件的图导航问题，提出THESEUS框架使推理路径显式且可追溯，并将KINSHIP和MQuAKE扩充为支持导航式推理评估的数据集。

    

    多跳知识图谱问答（KGQA）任务要求模型在知识图谱中沿着路径整合关系证据，以回答自然语言问题。然而，现有的KGQA系统通常只专注于预测最终答案，而没有显式地建模或验证中间推理步骤，这使得我们无法判断正确答案是否来自忠实（faithful）的多跳推理。为了解决这一局限，我们将多跳KGQA重新构建为一个以问题为条件的图导航问题。我们将这一表述称为THESEUS——统一语义中的可追溯逐跳证据搜索。在这一设定下，智能体接收知识图谱、问题和主题实体，并遍历一系列关系以到达答案，从而使推理路径变得显式。为了系统地研究这一表述，我们提供了三个关键贡献：（i）我们将现有的KINSHIP和MQuAKE资源扩充为可直接用于导航的KGQA数据集，并带有标注……（原文摘要在此处截断）

    arXiv:2609.14528v1 Announce Type: new  Abstract: Multi-Hop Knowledge Graph Question Answering (KGQA) tasks require models to assemble relational evidence along paths in a KG to answer natural-language questions. However, existing KGQA systems typically focus on predicting the final answer without explicitly modeling or validating the intermediate reasoning steps, obscuring whether the correct answers arise from faithful multi-hop reasoning. To address this limitation, we re-frame multi-hop KGQA as a question-conditioned graph navigation problem. We refer to this formulation as THESEUS - Traceable Hop-wise Evidence SEarch in a Unified Semantics. In this setting, an agent receives a KG, a question, and a topic entity, and traverses a sequence of relations towards the answer, making the reasoning path explicit. To systematically study this formulation, we provide three key contributions. (i) We augment the existing KINSHIP and MQuAKE resources into navigation-ready KGQA datasets with anno
    
[^115]: NeuroActiSep：通过单次前向传播从前馈神经元中检测事实性幻觉

    NeuroActiSep: Detecting Factual Hallucinations from Feed-Forward Neurons in a Single Pass

    [https://arxiv.org/abs/2609.14448](https://arxiv.org/abs/2609.14448)

    提出NeuroActiSep方法，在单次前向传播中对最终提示词位置的前馈神经元进行排序筛选，并证明仅用所选神经元特征训练的探测器即可达到与基于内部状态训练的探测器相当的幻觉检测性能。

    

    大型语言模型中的幻觉降低了其可靠性并减缓了其应用普及。多种白盒研究已经利用内部表征来检测真实性和事实性的模式。而一种研究较少的途径是识别与幻觉相关的前馈神经元。我们提出了一种方法，使用自定义的神经元选择数据集对最终提示词位置的前馈神经元进行排序。我们将选出的神经元身份迁移，在其他事实性问答数据集上训练幻觉分类器。我们的工作提供了实证证据，表明使用所选神经元特征训练的探测器的性能与在内部状态上训练的探测器相当。我们还分析了所选神经元的分布以及层级深度对检测性能的影响。

    arXiv:2609.14448v1 Announce Type: cross  Abstract: Hallucination in large language models reduces their reliability and slows adoption. Various white-box studies have used internal representations to detect patterns of truthfulness and factuality. A less-studied approach is to identify feed-forward neurons correlated with hallucination. We propose a method to rank feed-forward neurons at the final prompt token using a custom neuron selection dataset. We transfer the selected neuron identities to train hallucination classifiers on other factual question answering datasets. Our work provides empirical evidence that probes trained using the features from the selected neurons perform on par with probes trained on internal states. We also analyze the distribution of selected neurons and the effect of layer depth on detection performance.
    
[^116]: 问题开局之策：智能体深度搜索中第一步至关重要

    Question's Gambit: The First Move Matters in Agentic Deep Search

    [https://arxiv.org/abs/2609.14412](https://arxiv.org/abs/2609.14412)

    提出Question's Gambit首步检索模块，通过将问题分解为线索、生成互补搜索查询并重排序候选文档，为智能体深度搜索构建优质开局上下文，从而提升复杂问题的解答能力。

    

    深度研究智能体通过搜索、阅读和推理的迭代循环来回答复杂问题。最近针对BrowseComp-Plus等推理密集型基准的研究表明，配置良好的词法检索能够挖掘出高质量证据，但智能体仍可能无法将携带证据的文档与金标文档相关联。我们识别出深度研究智能体的首次检索动作是这一场景下的重要设计决策。我们提出Question's Gambit，这是一个首步检索模块，它将问题分解为一组线索，将其重新表述为互补的搜索查询，整合检索到的结果，并在智能体开始其迭代的搜索与推理过程之前对候选池进行重排序。由此产生的开局上下文旨在同时支持线索聚合和最终答案验证。我们还在MultiHop-RAG上进行了进一步评估，以测试这些收益能否超越BrowseComp-Plus而泛化。

    arXiv:2609.14412v1 Announce Type: new  Abstract: Deep research agents answer complex questions through iterative loops of searching, reading, and reasoning. Recent work on reasoning-intensive benchmarks such as BrowseComp-Plus shows that well-configured lexical retrieval can surface high-quality evidence, yet agents may still fail to connect documents carrying evidence to the gold documents. We identify a deep research agent's first retrieval move as an important design decision for this setting. We introduce Question's Gambit, a first-move retrieval module that decomposes the question into a set of clues, reformulates them into complementary searches, consolidates the retrieved results, and reranks the candidate pool before the agent begins its iterative search-and-reasoning process. This produces an opening context designed to support both clue aggregation and final-answer verification. We further evaluate on MultiHop-RAG to test whether these benefits transfer beyond BrowseComp-Plus
    
[^117]: 智能体评估中的政策漏洞：当政策模糊性伪装成智能体错误时

    Policy Loopholes in Agent Evaluation: When Policy Ambiguity Masquerades as Agent Error

    [https://arxiv.org/abs/2609.14400](https://arxiv.org/abs/2609.14400)

    该研究揭示智能体基准测试中自然语言政策的模糊性会被误判为智能体错误，提出政策漏洞分类体系，并证明政策规范质量是评估质量的决定性上限。

    

    智能体基准测试评估政策合规性，但其假设每条政策都决定唯一的正确行动。自然语言政策可能通过沉默、歧义或矛盾违反这一假设，从而允许多种可辩护的解释，而单一的标准轨迹无法涵盖这些解释。通过对两个τ²-bench领域进行审计，我们建立了此类政策漏洞的分类体系，并表明受影响的任务会产生不可靠的分数：它们以不同方式降低不同模型的分数，并使每个模型在重复试验中的一致性下降。跨领域比较表明，漏洞的可利用性需要政策模糊性和工具宽松性同时存在：当政策复杂性超出工具所能执行的范围时，智能体对政策空白的处理方式不一致，分数变得不可靠。政策规范的质量决定了评估质量的上限。基准测试开发者应在收集标准标注之前对政策进行审计。

    arXiv:2609.14400v1 Announce Type: new  Abstract: Agent benchmarks evaluate policy compliance but assume each policy determines a unique correct action. Natural-language policies can violate this assumption through silence, ambiguity, or contradiction, admitting multiple defensible readings that a single gold trajectory cannot capture. Auditing two $\tau^2$-bench domains, we develop a taxonomy of such policy loopholes and show that affected tasks produce unreliable scores: they lower scores across different models in different ways and make every model less consistent across repeated trials. A cross-domain comparison reveals that exploitability requires both policy ambiguity and tool permissiveness: when policy complexity exceeds what tools can enforce, agents resolve gaps inconsistently and scores become unreliable. Policy specification quality sets the ceiling on evaluation quality. Benchmark developers should audit policies before collecting gold annotations.
    
[^118]: MOSCOPT：面向大语言模型智能体的混合技能协同优化

    MOSCOPT: Mixture-of-Skills Collective Optimization for LLM Agents

    [https://arxiv.org/abs/2609.14399](https://arxiv.org/abs/2609.14399)

    MOSCOPT提出了一种文本原生、无需参数的算法，通过联合优化技能池与动态门控技能，并借助带双重状态的EditAdam进行三阶段交错更新，使LLM智能体在无梯度情况下单调提升性能，在5个基准测试上全面超越现有基线。

    

    自然语言提示和技能构成了基于大语言模型智能体的战略支柱。尽管近年来提示词与技能优化已取得显著进展，但现有方法均只优化单一文本模板——忽略了多种互补策略之间的协同作用。我们提出MOSCOPT，这是一种文本原生、无需参数的算法，它联合优化一个包含N个技能的技能池以及一个门控技能G，后者在每一步动态选择K个技能。为了有效优化这些技能，我们构建了内部维护双重状态的EditAdam。通过EditAdam的三阶段交错更新，该系统能够在没有梯度或参数调整的情况下实现单调改进。在5个基准测试和3个目标大语言模型上进行的广泛实验与详细消融研究表明，MOSCOPT始终优于所有基线方法，并证实了带有选择性激活的混合技能架构与协同进化……

    arXiv:2609.14399v1 Announce Type: new  Abstract: Natural language prompts and skills serve as the strategic backbone of LLM-based agents. Recent advances in prompt and skill optimization have achieved notable gains, yet all existing methods optimize a \emph{single} text template---missing the synergy among multiple complementary strategies. We propose MOSCOPT, a text-native, parameter-free algorithm that jointly optimizes a pool of $N$ skills and a gating skill $G$ that dynamically selects $K$ skills per step. To effectively optimize the skills, we build the EditAdam with internally maintained dual states. Through the three-phase interleaved updates with EditAdam, the system monotonically improves without gradient or parameter tuning. Extensive experiments and detailed ablations across 5 benchmarks and 3 target LLMs demonstrate that MOSCOPT consistently outperforms all baselines, and confirm that both the mixture-of-skills architecture with selective activation and the collective evolu
    
[^119]: 语言的形式性质作为神经动力学的约束

    Formal Properties of Language as Constraints on Neural Dynamics

    [https://arxiv.org/abs/2609.14384](https://arxiv.org/abs/2609.14384)

    本文提出“神经可容许性纲领”，指出语言的代数性质（如非结合层级分组、递归封闭性、结构化工作区转换等）构成神经机制必须满足的数学不变量约束，并为每项约束提供了具体的神经动力学候选机制。

    

    神经系统必须具备何种能力才能实现语言？当前研究通过语言变量对刺激进行标注，并测试哪些电极、体素或语言模型层能够预测神经活动。然而，预测上的成功并未充分约束潜在的机制。在本研究中，我们证明语言的代数性质规定了机制必须保持的不变量：非结合的层级分组、交换性、递归封闭性、对子结构的访问能力，以及结构化的工作区转换。我们将这一研究框架命名为“神经可容许性纲领”。我们对句法结构构建进行了代数化分析，并为每项要求提出了候选机制：内容可寻址的工作区记忆、调度结构构建操作的图结构瞬态动力学（例如稳定异宿轨道通道），以及记录分组关系的相位耦合封闭操作。仿真结果表明，经修正的Marcolli-Berwick熵优化模型……

    arXiv:2609.14384v1 Announce Type: new  Abstract: What must a neural system be capable of to implement language? Current research annotates stimuli with linguistic variables and tests which electrodes, voxels, or language-model layers predict neural activity. Yet predictive success leaves mechanisms under-constrained. Here, we show that algebraic properties of language specify invariants that mechanisms must preserve: non-associative hierarchical grouping, commutativity, recursive closure, access to substructures, and structured workspace transitions. We term this the Neural Admissibility Program (NAP). Syntactic structure building is analyzed algebraically, with candidate mechanisms offered for each requirement: content-addressable workspace memory, graph-structured transient dynamics scheduling structure-building operations (e.g. stable heteroclinic channels), and a phase-coupled sealing operation recording grouping. Simulations show that a corrected Marcolli-Berwick entropy-optimized
    
[^120]: SpectralShift：通过谱重参数化实现门控DeltaNet的有效上下文窗口扩展

    SpectralShift: Effective Context Window Extension of Gated DeltaNet via Spectral Reparameterization

    [https://arxiv.org/abs/2609.14320](https://arxiv.org/abs/2609.14320)

    提出SpectralShift方法，通过谱重参数化重塑门控DeltaNet的衰减谱——扩展与目标依赖长度对齐的慢速谱带、同时保留用于状态清除的快速衰减模式，从而有效实现其上下文窗口扩展。

    

    最近，线性注意力层被越来越多地大规模采用，以替代softmax注意力进行长上下文建模。然而，现有的上下文扩展方法通常直接应用持续预训练而不修改这些层，忽略了线性注意力状态动态的谱特性。在这项工作中，我们从转移矩阵的谱视角研究门控DeltaNet（GDN）的长上下文扩展，并识别出控制长程信息检索的两个关键因素：（1）与目标依赖长度对齐的足够宽的慢速谱带；（2）保留用于状态清除和上下文切换的快速衰减模式。基于这一观察，我们提出了SpectralShift，一种用于GDN长上下文持续预训练的谱重参数化方法。具体而言，SpectralShift对alpha投影的初始化进行重参数化，以重塑衰减谱（注：摘要原文在此处截断）。

    arXiv:2609.14320v1 Announce Type: new  Abstract: Recently, linear attention layers have been increasingly adopted to replace softmax attention at scale for long-context modeling. However, existing context extension approaches typically apply continued pretraining directly without modifying these layers, overlooking the spectral properties of linear attention state dynamics. In this work, we study long-context extension of Gated DeltaNet (GDN) from a spectral perspective of transition matrix and identify two essential factors governing long-range information retrieval: (1) a sufficiently broad slow spectral band aligned with the target dependency length, and (2) the preservation of fast-decaying modes for state clearing and context switching. Based on this observation, we propose SpectralShift, a spectral reparameterization approach for long-context continual pretraining of GDNs. Specifically, SpectralShift reparameterizes the alpha projections initialization to reshape the decay spectr
    
[^121]: E2A-Bench：金融图表推理中证据到行动可靠性的基准测试

    E2A-Bench: Benchmarking Evidence-to-Action Reliability in Financial Chart Reasoning

    [https://arxiv.org/abs/2609.14302](https://arxiv.org/abs/2609.14302)

    提出了E2A-Bench基准，通过969个查询和四项新指标评估20个金融视觉语言模型将图表证据转化为可靠投资建议的完整链路，揭示出传统标量幻觉评分所掩盖的推理断裂与覆盖率不足等失败模式。

    

    金融视觉语言模型（VLM）能否将图表证据转化为可靠的行动建议？现有的幻觉评估大多以声明为中心，即评估生成的陈述是否得到支持，但并未评估证据是否在推理、信心和最终行动的过程中保持可追溯性。我们提出了E2A-Bench，这是一个包含969个查询的金融图表推理基准，由323只沪深300成分股在三种输入模态下构建，并采用基于OHLCV（开盘价、最高价、最低价、成交量）数据的确定性证据锚点。E2A-Bench通过UCR、RCI、ECI和NDR四项指标评估模型的基础性、推理-行动一致性、证据-信心校准以及方向覆盖率，其中NDR衡量的是考虑覆盖率之后的证据到行动可靠性，而非已实现的交易表现。对20个视觉语言模型的评估揭示了被标量幻觉分数所掩盖的三种失败模式：UCR最低的模型因方向覆盖率仅为6.4%而在NDR排名中接近垫底……

    arXiv:2609.14302v1 Announce Type: new  Abstract: Can financial vision-language models (VLMs) turn chart evidence into reliable action recommendations? Existing hallucination evaluations are mostly claim-centric; they assess whether generated statements are supported, but not whether evidence remains traceable through rationale, confidence, and final action. We introduce E2A-Bench, a 969-query benchmark for financial chart reasoning, constructed from 323 HS300 constituents under three input modalities with deterministic OHLCV-derived evidence anchors. E2A-Bench evaluates grounding, reasoning-action consistency, evidence-confidence calibration, and directional coverage through UCR, RCI, ECI, and NDR, where NDR measures coverage-aware evidence-to-action reliability rather than realized trading performance. Evaluating 20 VLMs reveals three failures hidden by scalar hallucination scores: the lowest-UCR model ranks near the bottom by NDR due to only 6.4% directional coverage; oracle-aided ve
    
[^122]: 编辑路由塑造AI辅助科学写作中计算结果的限定方式

    Editorial routing shapes how computational results are qualified in AI-assisted scientific writing

    [https://arxiv.org/abs/2609.14288](https://arxiv.org/abs/2609.14288)

    该研究发现AI写作助手是否在手稿中保留计算结果的数值限定，取决于比较内容被“编辑路由”到工作流的哪个位置（组存储库、支持信息或工作笔记），且针对性的放置规则比通用准确性提醒更能有效恢复这些限定信息。

    

    大型语言模型越来越多地被用于分析计算结果和起草手稿，这使得可靠的科研交流与正确的分析变得同等重要。通过使用固定的计算证据，我们测试了将建模选择的比较内容分配到研究工作流的其他位置是否会改变手稿的报告方式。在受限的句子写作任务中，Anthropic的Claude Sonnet 5在详细比较被分配给组存储库时经常省略数值限定，但当相同的比较被分配给支持信息或其自己的工作笔记时，则更常保留这些限定；Claude Opus 5对此则不太敏感。这些效应并不遵循简单的可访问性排序。有针对性的放置规则在很大程度上恢复了句子级别的数值限定，而通用的准确性提醒则无法做到。较长的贡献内容保留了数值限定，尽管某些跨计算设置的摘要仍然被重定向到（原文在此处截断）。

    arXiv:2609.14288v1 Announce Type: new  Abstract: Large language models increasingly analyze computational results and draft manuscripts, making reliable communication as important as correct analysis. Using fixed computational evidence, we tested whether assigning comparisons across modeling choices elsewhere in a research workflow changes manuscript reporting. In constrained sentence-writing tasks, Anthropic's Claude Sonnet 5 often omitted numerical qualifications when detailed comparisons were assigned to a group repository, but retained them more often when the same comparison was assigned to Supporting Information or its own working notes; Claude Opus 5 was less sensitive. These effects did not follow a simple accessibility ordering. A targeted placement rule largely restored sentence-level qualification, whereas a generic accuracy reminder did not. Longer contributions retained numerical qualifications, although some summaries across computational settings were still redirected to
    
[^123]: DenMark：面向扩散语言模型的鲁棒语义水印

    DenMark: Robust Semantic Watermarking for Diffusion Language Models

    [https://arxiv.org/abs/2609.14257](https://arxiv.org/abs/2609.14257)

    提出了DenMark，一个面向扩散语言模型的语义水印框架，通过将密钥相关信号直接注入去噪过程并利用临时展开作为语义前瞻，实现了对改写等语义保持编辑鲁棒的文本水印。

    

    语义文本水印在含义层面而非表层词元选择中编码信号，从而对改写及其他保持语义的编辑具有鲁棒性。现有的语义水印方法主要针对自回归语言模型（ARLMs）设计，这类模型可以在继续生成之前先生成完整的候选单元并进行评分。这一范式无法自然地扩展到扩散语言模型（DLMs），因为在中间去噪步骤中语义单元仍处于不完整状态，且词元可以以灵活的顺序被更新。我们提出了DenMark，一个将密钥相关信号直接注入扩散语言模型去噪过程的语义水印框架。DenMark将输出划分为固定的词元区域，并将临时展开作为语义前瞻：通过条件补全来估计不完整区域最终形成的语义，使DenMark能够选择具有更高估计语义水印信号的局部更新。

    arXiv:2609.14257v1 Announce Type: new  Abstract: Semantic text watermarks encode signals in meaning rather than surface token choices, offering robustness to paraphrasing and other semantic-preserving edits. Existing semantic watermarking methods are primarily designed for autoregressive language models (ARLMs), where completed candidate units can be generated and scored before generation proceeds. This paradigm does not naturally extend to diffusion language models (DLMs), where semantic units remain incomplete during intermediate denoising steps and tokens may be updated in flexible orders. We propose DenMark, a semantic watermarking framework that injects key-dependent signals directly into the DLM denoising process. DenMark partitions the output into fixed token regions and uses temporary rollouts as semantic lookahead: conditional completions estimate the eventual semantics of an incomplete region, enabling DenMark to select local updates with higher estimated semantic watermark s
    
[^124]: 用于评估社交媒体公共卫生短文本传播主题模型的文档-主题对齐指标

    Document Topic Alignment Metrics for Evaluating Topic Models of Short-Text Public Health Communications on Social Media

    [https://arxiv.org/abs/2609.14256](https://arxiv.org/abs/2609.14256)

    提出文档-主题对齐指标，通过衡量帖子与其分配主题之间的语义对齐程度来评估主题模型，为短文本公共卫生社交媒体分析提供了与传统指标互补且与人工评估高度一致的评估方法。

    

    主题模型被广泛用于分析公共卫生相关的社交媒体短文本，然而其评估目前主要由仅关注生成主题本身的指标主导，缺乏能够定量评估所分配主题是否有意义地代表相应短文本帖子的指标。我们提出了文档-主题对齐指标，这是一个具有分配感知能力的评估框架，包含衡量文档（帖子）与其分配主题之间语义对齐程度的指标。我们还引入了基于边际和判别性的变体，用于捕捉主题分配的置信度和可区分性。我们在来自X平台的三个公共卫生相关社交媒体数据集上，对五个主题模型评估了DoTA，并将其与传统基于主题的指标进行比较。结果表明，DoTA提供了互补的评估线索，并与人工评估结果有效对齐。这些发现建立了……

    arXiv:2609.14256v1 Announce Type: new  Abstract: Topic models are widely used to analyze public health-related social media short texts, yet their evaluation remains dominated by metrics that focus entirely on generated topics alone. There is a lack of metrics that quantitatively assess whether assigned topics meaningfully represent the corresponding short-text posts. We propose Document-Topic Alignment metrics (DoTA), an assignment-aware evaluation framework comprising metrics that measure semantic alignment between documents (posts) and their assigned topics. We also introduce margin-based and discriminative variants that capture topic assignment confidence and distinguishability. We evaluate DoTA across five topic models on three public health-related social media datasets from X and compare DoTA metrics with conventional topic-based metrics. Results show that DoTA provides complementary evaluation cues and aligns meaningfully with human evaluations. These findings establish the nee
    
[^125]: 检索增强生成中的归因-压缩前沿

    The Attribution-Compression Frontier in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.14245](https://arxiv.org/abs/2609.14245)

    本文系统测量了检索增强生成中各类上下文压缩方法对引用归因的影响，揭示了摘要式压缩的引用虽与其自身摘要高度一致（精确率0.86）但与源文本严重脱节（精确率仅0.12），而抽取式选择能在不同压缩预算下保持0.43-0.49的稳定归因精确率。

    

    arXiv:2609.14245v1 公告类型：cross 摘要：上下文压缩可以减少检索增强生成中生成器的输入长度，但仅凭答案质量无法刻画引用归因的特性。我们在固定生成器和主蕴含评估器的条件下，在ASQA和QASPER数据集上测量了不同压缩方法和压缩预算下的引用归因表现，比较了重排序、抽取式选择、摘要式总结、词元剪枝以及抽取-聚类-重写构建等方法。在ASQA数据集上标称0.25的压缩预算下（实际压缩率为0.08），RECOMP风格的压缩器生成的引用相对于其自身摘要的精确率为0.86，但在我们的重新归因协议下，相对于源文本片段的精确率仅为0.12。这些估计结果依赖于共享的NLI模型进行片段恢复和引用评分，缺乏独立的人工校准。抽取式选择在从ASQA上下文的一半到十分之一不等的标称预算范围内，其观测到的接地精确率介于0.43到0.49之间，同时答案质量有所下降。

    arXiv:2609.14245v1 Announce Type: cross  Abstract: Context compression reduces generator input in retrieval-augmented generation, but answer quality alone does not characterize citation attribution. We measure citation attribution across compression methods and budgets, comparing reranking, extractive selection, abstractive summarization, token pruning, and an extract-cluster-rewrite construction on ASQA and QASPER under a fixed generator and primary entailment evaluator. On ASQA at a nominal 0.25 budget (achieved compression 0.08), a RECOMP-style compressor's citations score 0.86 precision against its summaries but 0.12 against source spans under our re-attributability protocol. These estimates depend on a shared NLI model for span recovery and citation scoring and lack independent human calibration. Extractive selection's observed grounded precision ranges from 0.43 to 0.49 across nominal budgets from one-half to one-tenth of the ASQA context, while answer quality declines. For the s
    
[^126]: 面向风格感知放射学报告的语料库特征刻画与逆向宪法微调

    Corpus Characterization and Inverse Constitutional Fine-Tuning for Style-Aware Radiology Reports

    [https://arxiv.org/abs/2609.14226](https://arxiv.org/abs/2609.14226)

    该研究通过聚类分析识别出放射学报告的五种写作风格模式，并提出逆向宪法AI微调框架，从报告对中自动提炼风格规范，使自动生成的放射学报告更贴近真实放射科医生的写作风格。

    

    arXiv:2609.14226v1 公告类型：新论文 摘要：自动化放射学报告生成在诊断准确性方面进展迅速，然而生成的报告在结构、措辞和不确定性语言上常常偏离真实放射科医生写作的风格惯例，这一差距直接影响临床医生的信任度和用户体验。为解决这一问题，我们使用Bio-ClinicalBERT嵌入、UMAP降维和HDBSCAN聚类，对来自CheXpert Plus数据集的2000份报告进行了风格变异特征刻画，识别出五种不同的报告模式，它们在病理关注重点、叙事结构和词汇偏好方面各不相同。基于这些发现，我们改造了逆向宪法AI框架，从放射科医生撰写的报告对中推导出一个以风格为核心的宪法，而无需正式的偏好数据集。该宪法编码了语气、措辞、不确定性校准和报告结构方面的写作惯例，并被纳入……（摘要原文在此处截断）

    arXiv:2609.14226v1 Announce Type: new  Abstract: Automated radiology report generation has advanced rapidly in diagnostic accuracy, yet generated reports frequently diverge from the stylistic conventions of authentic radiologist writing in structure, diction, and uncertainty language, a gap which has direct implications for clinician trust and user experience. To address this, we characterize stylistic variation across 2,000 reports from the CheXpert Plus dataset using Bio-ClinicalBERT embeddings, UMAP dimensionality reduction, and HDBSCAN clustering, identifying five distinct reporting patterns differing in pathology focus, narrative structure, and lexical preference. Drawing on these findings, we adapt the inverse constitutional AI framework to derive a style-focused constitution from radiologist-written report pairs without requiring a formal preference dataset. This constitution, encoding conventions of tone, diction, uncertainty calibration, and report structure, is incorporated i
    
[^127]: 从估计的听者注视行为中学习指称表达

    Learning to Refer from Estimated Listener Gaze

    [https://arxiv.org/abs/2609.14207](https://arxiv.org/abs/2609.14207)

    该论文提出将神经听者估计的注视扫描路径转化为奖励信号，用以微调视觉-语言模型，使其生成在语用上更优的指称表达。

    

    我们提出对视觉-语言模型进行微调，以生成在语用上更优的指称表达，方法是将对听者增量理解的观察（以注视扫描路径的形式）转化为学习信号。在训练过程中，从正在优化的说话者策略中基于图像和目标指称对象采样指称表达；然后，一个能够估计人类注视行为的神经听者将图像和采样得到的指称表达映射为扫描路径，每个扫描路径由一系列注视点表示，每个注视点对应指称表达中的一个词。我们尝试了多种将注视点序列和目标指称对象转换为token级和序列级奖励的方法，并利用这些奖励来优化策略参数。通过人类听者的评估，我们发现使用注视估计听者训练的说话者策略能够产生显著更多语用最优的指称表达。

    arXiv:2609.14207v1 Announce Type: new  Abstract: We propose to finetune vision-language models to generate more pragmatically optimal referring expressions by transforming observations of incremental listener comprehension, in the form of gaze scanpaths, into learning signals. During training, referring expressions are sampled from the speaker policy being optimized, conditioned on images and target referents; then, a neural listener estimating human gaze behavior maps from images and sampled referring expressions to scanpaths, each represented by a sequence of fixations, with each fixation corresponding to a word in the referring expression. We experiment with several approaches to convert fixation sequences and target referents into token- and sequence-level rewards, which are used to optimize policy parameters. Through evaluation with human listeners, we find that speaker policies trained with gaze-estimating listeners result in significantly more pragmatically-optimal references th
    
[^128]: 一种用于有效反叙事生成与精炼的多阶段智能体框架

    A Multi-Stage Agentic Framework for Effective Counter-Narrative Generation and Refinement

    [https://arxiv.org/abs/2609.14178](https://arxiv.org/abs/2609.14178)

    提出了一个多阶段智能体框架，通过人类评估识别有效的修辞-风格配对，并利用多智能体迭代精炼过程，生成更具说服力、情感感染力和可分享性的反叙事，以对抗仇恨言论与错误信息。

    

    社交网络上仇恨言论和错误信息的快速传播给民主社会带来挑战，因为直接的压制努力可能加深社会极化、助长公众不信任并强化极端主义叙事。由大语言模型（LLM）驱动的反叙事（CNs）为降低这些风险提供了一种有前景的方式，但其有效性取决于修辞和风格上的选择，而这些选择目前仍鲜为人知。我们提出了一个多阶段基于智能体的框架，用于生成、精炼和评估反叙事，并将其应用于俄乌战争中的亲俄仇恨与错误信息叙事，同时该框架可适配到其他领域。一项有人类评估者参与的初步实验识别出了有效的技巧-风格配对，例如重复手法配合情感框架能够增强说服力。基于这些洞见，我们引入了一个多智能体精炼过程，通过迭代方式提升反叙事的说服力、情感感染力和可分享性。经过人类验证之后（原文在此处截断）

    arXiv:2609.14178v1 Announce Type: new  Abstract: The rapid diffusion of hate speech and misinformation on social networks challenges democratic societies, since direct suppression efforts may deepen polarization, fuel public distrusts, and strengthen extremist narratives. LLM-driven counter-narratives (CNs) offer a promising way to reduce those risks, yet their effectiveness depends on rhetorical and stylistic choices that remain poorly understood. We present a multi-stage agent-based framework for generating, refining, and evaluating CNs, applied to pro-Russian hate and misinformation narratives on the war with Ukraine and adaptable to other domains. A pilot experiment with human evaluators identifies effective technique style pairings, such as repetition with emotional framing enhancing persuasiveness. Building on these insights, we introduce a multi-agent refinement process that iteratively improves CNs for persuasiveness, emotional engagement, and shareability. After human validati
    
[^129]: 遗传的注意力头：音频语言模型通过其文本骨干网络的注意力追踪说话人，而注意力质量排序可检索出不同的注意力头集合

    Inherited Heads: Audio language models track speakers with their text backbone's attention, and an attention-mass ranking retrieves a different set

    [https://arxiv.org/abs/2609.14174](https://arxiv.org/abs/2609.14174)

    研究发现音频语言模型的说话人追踪能力主要继承自其文本骨干网络的注意力头——无需任何训练，仅对从纯文本模型中筛选出的注意力头添加固定偏置，就能以90%以上的准确率引导音频模型描述任意指定的说话人。

    

    当要求描述录音中六位说话人之一的谈话内容时，音频语言模型在6%到16%的试验中能描述正确的说话人，低于随机猜测可达到的16.7%。在不进行任何训练的情况下，仅对一百个注意力头的注意力逻辑值添加固定偏置——其规模不到模型中注意力头总数的十分之一——就能在90.7%到99.0%的试验中将描述重定向到我们选择的任意说话人。这些注意力头在很大程度上并非音频所特有。在任务的书面版本上，对音频模型所基于的纯文本语言模型（或同系列的已发布模型）进行注意力头排序，取其前一百个注意力头并原封不动地迁移过来：它们能在80.8%到95.0%的试验中重定向音频模型，而整个选择过程完全没有涉及任何音频信息。音频和文本注意力头集合共享100个中的66到74个，而随机情况下只会共享约20个，且仅共享部分就能再现几乎全部的引导效果。

    arXiv:2609.14174v1 Announce Type: cross  Abstract: Asked to describe what one of six speakers in a recording talks about, audio language models describe the right one on 6 to 16% of trials, below the 16.7% a guess would give. Adding a fixed bias to the attention logits of a hundred heads, under a tenth of the model's and with no training, redirects the description to whichever speaker we choose, on 90.7% to 99.0% of trials. Those heads are largely not specific to audio. Rank the text-only language model an audio model was built from, or a released model of the same family, on a written version of the task, take its top hundred heads, and carry them over unchanged: they redirect the audio model on 80.8% to 95.0% of trials, with nothing about audio entering the selection. The audio and text head sets share 66 to 74 of 100 where chance would give about 20, and the shared part alone reproduces almost all of the steering. What that does not show is that sharing is what makes the heads work:
    
[^130]: 迈向面向大语言模型的演化式上下文参数化

    Towards Evolving Context Parameterization for Large Language Models

    [https://arxiv.org/abs/2609.14168](https://arxiv.org/abs/2609.14168)

    提出免训练方法PLUME，通过构建全局更新表示并结合记忆证据形成的局部参数视图进行自适应整合，有效解决大语言模型在持续上下文演化中的记忆更新与信息保留难题。

    

    上下文参数化使大语言模型（LLM）能够将上下文内化为可复用的模型参数，从而避免在后续查询中重复处理。然而，现有方法通常假设上下文是静态的，缺乏在持续更新下区分有效性状态的显式机制。为了研究这一现实场景，我们形式化了序列演化记忆更新任务，并构建了MUSE-bench来评估更新信息的整合以及未受影响信息的保留。由此产生的挑战要求在调整记忆证据贡献的同时保留全局状态。受此启发，我们提出了PLUME，这是一种免训练方法，它构建全局更新表示，激活记忆证据以形成局部参数视图，并在解码过程中自适应地整合两者的预测。在MUSE-bench上的全面评估证明了PLUME的有效性。

    arXiv:2609.14168v1 Announce Type: cross  Abstract: Context parameterization enables large language models (LLMs) to internalize contexts into reusable model parameters, avoiding repeated processing across subsequent queries. However, existing methods typically assume static contexts and lack explicit mechanisms for distinguishing validity states under continual updates. To study this real-world scenario, we formalized the Memory Updating with Sequential Evolution (MUSE) task and constructed MUSE-bench to evaluate update incorporation and unaffected-information preservation. The resulting challenge requires preserving the global state while adjusting the contribution of memory evidence. Motivated by this, we proposed PLUME, a training-free method that constructs a global update representation, activates memory evidence to form a local parameter view, and adaptively integrates their predictions during decoding. Comprehensive evaluation on MUSE-bench demonstrated PLUME's effectiveness in 
    
[^131]: 当工具成为障碍：不必要工具的可用性对大型语言模型回答的影响

    When Tools Get in the Way: The Effect of Unnecessary Tool Availability on LLM Answering

    [https://arxiv.org/abs/2609.14157](https://arxiv.org/abs/2609.14157)

    该研究通过在10个知识领域构建500个查询对并对6个大语言模型进行3,000次试验评估，揭示了提供相关但不必要的工具会损害模型依靠自身知识正确回答问题的能力。

    

    大型语言模型越来越多地与扩展其自身知识能力边界的外部工具一同部署。工具在需要外部信息的任务中很有帮助，但工具的存在也可能改变模型处理那些并不需要工具的问题的方式。先前的工作大多关注模型是否恰当地选择和使用工具，而不必要的工具是否会改变答案的正确性则较少受到关注。我们研究了提供一个相关但不必要的工具是否会影响模型基于自身知识进行回答的能力，以及先前的工具交互是否会改变这一行为。我们构建了涵盖10个知识领域的500个查询对，每个查询对包含一个需要该领域工具的工具查询和一个不需要该工具的封闭领域查询。我们在三种条件下评估了六个大语言模型：工具不可用、工具可用、以及先前有过工具调用后工具可用。在3,000次基线试验中，汇总的答案正确率……

    arXiv:2609.14157v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed with external tools that extend what they can do beyond their own knowledge. Tools help on tasks that need external information, but their availability may also change how a model handles questions that do not need them. Prior work has mostly asked whether models select and use tools appropriately; whether an unnecessary tool changes the correctness of answers has received less attention. We ask whether making a related but unnecessary tool available affects a model's ability to answer from its own knowledge, and whether a preceding tool interaction changes this behaviour. We construct 500 query pairs across 10 knowledge domains. Each pair consists of a tool query, which needs the domain's tool, and a closed-domain query, which does not. Six LLMs are evaluated with the tool unavailable, available, and available after a prior tool call. Across 3,000 baseline trials the pooled answer r
    
[^132]: 并非一刀切：根据部署实际面对的问题设定推理深度

    One Size Does Not Fit All: Setting Inference Depth from the Questions a Deployment Actually Asks

    [https://arxiv.org/abs/2609.14144](https://arxiv.org/abs/2609.14144)

    本文提出在冻结的Transformer模型中间层附加经训练的“读取器”组件并通过置信度测试实现提前退出，揭示当部署场景的提示词范围已知时推理计算可被大幅节省，且节省幅度强烈依赖于具体的流量类型。

    

    Transformer语言模型被训练用于响应任何提示词，但每个实际部署只会遇到范围狭窄的问题：客服助手面对的是配送投诉，编程工具面对的是Python代码。然而，每个部署在每个token上都要支付相同的计算成本。本文测量了当提示词范围预先已知时，这些成本中有多少是可以避免的。所研究的机制是提前退出：一个小型的、经过训练的组件——称为“读取器”——被附加到中间层并提出一个token，随后由置信度测试决定是输出该token还是继续运行剩余的层。模型本身是冻结的，唯一使用的监督信号是模型自身在普通流量上的输出。本文报告了三个发现。首先，可实现的节省在很大程度上取决于流量类型：在一个15亿参数模型的半深度处，96%的算术应用题token可以提前输出，而中文解释类文本仅为8%……

    arXiv:2609.14144v1 Announce Type: new  Abstract: A transformer language model is trained to respond to any prompt, but each deployment asks only a narrow range of questions: a support assistant sees delivery complaints, a coding tool sees Python. Every deployment nonetheless pays the same computation per token. This paper measures how much of that cost is avoidable when the range of prompts is known in advance.   The mechanism examined is early exit: a small, trained component - called a readout - is attached to an intermediate layer and proposes a token, and a confidence test decides whether to emit it or to run the remaining layers. The models are frozen, and the only supervision used is the model's own output on ordinary traffic.   Three findings are reported. First, achievable savings depend strongly on the kind of traffic: at half depth on a 1.5-billion-parameter model, 96 percent of tokens could be emitted early for arithmetic word problems and 8 percent for Chinese-language expl
    
[^133]: GraMRAG：基于强化学习的图记忆编排多智能体多步推理

    GraMRAG: Orchestrating Multi-Agent Multi-Step Reasoning via Graph Memory with Reinforcement Learning

    [https://arxiv.org/abs/2609.14066](https://arxiv.org/abs/2609.14066)

    GraMRAG提出了一种由动态多模态记忆图引导的多智能体RAG框架，通过将智能体推理形式化为有向无环图并结合视觉-文本桥接推理范式，实现了稳定的多步多模态推理，有效缓解了状态盲视和冗余检索问题。

    

    尽管现有的多智能体检索增强生成（RAG）系统在复杂多模态推理任务上已展现出良好前景，但它们在推理深度和记忆结构方面仍存在根本性局限，在回答知识密集型问题时面临检索不充分和状态盲视的困境。为了解决这些局限，我们提出了GraMRAG，一种由图记忆引导的多智能体RAG框架，它集成了动态多模态记忆图以实现稳定的多步多模态推理。我们引入了一种视觉-文本桥接推理范式，将多尺度实体裁剪与ReAct风格的视觉工具链统一起来，增强了长程跨模态推理能力。我们进一步构建了一个多模态记忆图，将智能体推理形式化为动态有向无环图（DAG），显式建模动作-观察依赖关系，以缓解状态盲视并抑制冗余检索。

    arXiv:2609.14066v1 Announce Type: cross  Abstract: Although existing multi-agent Retrieval-Augmented Generation (RAG) systems have demonstrated promise on complex multimodal reasoning tasks, they remain fundamentally limited in reasoning depth and memory structure, suffering from inadequate retrieval and state blindness when answering knowledge-intensive questions. To address these limitations, we propose GraMRAG, a graph memory-guided multi-agent RAG framework that integrates a dynamic multimodal memory graph to enable stable, multi-step multimodal reasoning. We introduce a vision-text bridged reasoning paradigm that unifies multi-scale entity cropping with a ReAct-style visual toolchain, enhancing the long-horizon cross-modal reasoning capability. We further construct a multimodal memory graph that formalizes agent reasoning as a dynamic directed acyclic graph (DAG), explicitly modeling action-observation dependencies to mitigate state blindness and suppress redundant retrieval. More
    
[^134]: 衡量前沿大语言模型在自动化研究中的创造力

    Measuring the Creativity of Frontier LLMs in Automated Research

    [https://arxiv.org/abs/2609.14057](https://arxiv.org/abs/2609.14057)

    本文提出了一套从价值性和新颖性两个维度评估大语言模型自动化研究创造力的指标体系，发现模型在反映研究空间探索广度的变量级新颖性指标上差异显著。

    

    前沿大语言模型越来越有能力开展自动化研究，但它们在这种场景下的创造力尚未得到系统性评估。本文提出了一套指标，从价值性和新颖性两个维度评估创造力。价值性评估每个提出的想法是否有用，而新颖性则从三个角度进行评估：相同想法是否曾经出现过（精确匹配P-新颖性，Exact-Match P-Novelty）、是否探索了此前未被探索过的变量或变量组合（变量级P-新颖性，Variable-level P-Novelty），以及该想法是直接遵循检索到的外部知识还是对其有所突破（H-新颖性，H-Novelty）。我们的评估表明，这些模型在大多数创造力指标上取得了相对相似的分数，但在变量级P-新颖性上存在显著差异，该指标反映了研究空间探索的广度。进一步的相关性和想法层面性能分析表明，变量级P-新颖性……

    arXiv:2609.14057v1 Announce Type: new  Abstract: Frontier LLMs are increasingly capable of conducting automated research, yet their creativity in this setting has not been systematically evaluated. In this paper, we propose a set of metrics to evaluate creativity along the two dimensions of valueness and novelty. Valueness assesses whether each proposed idea is useful, while novelty is evaluated from three perspectives: whether the same idea has appeared before (Exact-Match P-Novelty), whether a previously unexplored variable or variable combination is explored (Variable-level P-Novelty), and whether the idea directly follows retrieved external knowledge or departs from it (H-Novelty). Our evaluation shows that the models achieve relatively similar scores on most creativity metrics, but differ substantially in Variable-level P-Novelty, which reflects the breadth of research-space exploration. Further correlation and idea-level performance analyses show that Variable-level P-Novelty is 
    
[^135]: VeriDx：以疾病为中心的验证赢得诊断的权利

    VeriDx: Earning the Right to Diagnose with Disease-Centric Verification

    [https://arxiv.org/abs/2609.14018](https://arxiv.org/abs/2609.14018)

    VeriDx是一个以疾病为中心的验证框架，通过将大语言模型的自由格式诊断推理与结构化疾病档案关联，追踪每个疾病假设的临床义务履行情况（如关键检查、鉴别诊断、矛盾处理等），从而超越仅评估最终答案的传统方式，全面评估医疗AI的推理质量。

    

    一个正确的诊断仍然可能基于错误的理由得出。在临床推理中，每个疾病假设都会产生相应的义务：必须检查关键证据、必须排除替代诊断、必须解决矛盾之处、必须考虑有用的检查手段，并且必须对诊断结论的得出给出充分论证。目前对医疗大语言模型的评估大多集中在最终答案、局部步骤或孤立事实上，因此忽略了这些由假设引发的承诺。我们提出了VeriDx，这是一个以疾病为中心的验证框架，它将自由格式的诊断推理与结构化的疾病档案关联起来。VeriDx追踪每个假设是被满足、未解决还是违反了其临床义务，从而暴露出诸如遗漏关键检查、鉴别诊断未解决、忽视矛盾、缺乏依据的主张以及过早下结论等失败情况。我们使用基于临床指南推导的疾病档案和专家标注，将VeriDx应用于复杂呼吸系统诊断的实例化。

    arXiv:2609.14018v1 Announce Type: new  Abstract: A correct diagnosis can still be reached for the wrong reasons. In clinical reasoning, every disease hypothesis creates obligations: key evidence must be checked, alternatives must be ruled out, contradictions must be resolved, useful tests must be considered, and closure must be justified. Current evaluations of medical LLMs mostly focus on final answers, local steps, or isolated facts, and therefore miss these hypothesis-induced commitments. We introduce \textbf{VeriDx}, a disease-centric verification framework that links free-form diagnostic reasoning to structured disease profiles. VeriDx tracks whether each hypothesis is satisfied, unresolved, or violated its clinical obligations, exposing failures such as missing critical tests, unresolved differentials, ignored contradictions, unsupported claims, and premature closure. We instantiate VeriDx for complex respiratory diagnosis using guideline-derived disease profiles and expert-annot
    
[^136]: TF-IDF和BM25是精确的KL散度

    TF-IDF and BM25 Are Exact KL Divergences

    [https://arxiv.org/abs/2609.14016](https://arxiv.org/abs/2609.14016)

    该论文证明TF-IDF和BM25这两种经典评分方法可以被精确解释为两个概率模型之间的KL散度，为信息检索方法建立了统一的理论基础。

    

    TF-IDF和BM25是两种最广泛使用的查询-文档相关性评分方法，然而两者都没有一个标准的概率推导来证明其作为统一框架内统计方法的合理性。我们通过证明这两种评分方法都可以被精确解释为两个概率模型之间的Kullback-Leibler散度来填补这一空白。我们处理了在IDF项中包含加1校正的BM25变体，这也是实际使用中的版本，同时还讨论了不含该校正的原始BM25公式。所得框架为TF-IDF和BM25提供了共同的理论基础，阐明了它们所度量的内容，并使它们能够在理论上与其他信息检索方法进行比较，而不仅仅局限于实验性比较。

    arXiv:2609.14016v1 Announce Type: cross  Abstract: TF-IDF and BM25 are two of the most widely used methods for scoring query-document relevance, yet neither has a standard probabilistic derivation that justifies it as a statistical method within a unified framework. We address this gap by showing that both scoring methods admit an exact interpretation as Kullback-Leibler divergences between two probability models. We treat the BM25 variant that includes the plus 1 correction in the IDF term, which is the one used in practice, and also discuss the original BM25 formulation without that correction. The resulting framework provides a common theoretical basis for TF-IDF and BM25, clarifies what they measure, and allows them to be compared theoretically with other information retrieval methods rather than only experimentally.
    
[^137]: 解锁不可解问题：教师引导的课程学习实现数据高效的RLVR

    Unlocking the Unsolvable: Teacher-Guided Curriculum for Data-Efficient RLVR

    [https://arxiv.org/abs/2609.13997](https://arxiv.org/abs/2609.13997)

    通过教师引导的课程学习，仅用128个原本无法解决的难题训练就能匹敌或超越GRPO在2000个问题上的训练效果（约16倍数据效率），并显著扩展模型的推理能力边界。

    

    可验证奖励强化学习（RLVR）在提升大型语言模型的数学推理能力方面已展现出显著成效。然而，超出模型当前能力范围的问题——其推理采样全部失败、无法产生任何学习信号——尽管标志着最具信息量的训练前沿，却在结构上被浪费了。我们证明，这些原本无效的问题可以通过教师引导的课程学习来解锁：来自更强模型的部分推理轨迹构建了分级的难度景观，而反向链接课程逐步撤回引导，直到模型能够独立解决问题。在两个基础模型的九项基准测试平均值上，仅在128个不可解问题上进行训练即可匹敌或超越在完整2000个问题语料库上训练的GRPO（约16倍数据效率），同时在较大k值下通过pass@k衡量的推理边界得到显著扩展。此外，我们识别出一种分布偏移现象…

    arXiv:2609.13997v1 Announce Type: new  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) has shown remarkable success in improving the mathematical reasoning of large language models. Yet problems beyond the model's current capability, where rollouts uniformly fail and no learning signal is produced, are structurally wasted despite marking the most informative training frontier. We show that these otherwise-inert problems can be unlocked via teacher-guided curriculum learning: partial reasoning traces from a stronger model create a graded difficulty landscape, and a backward-chaining curriculum progressively withdraws guidance until the model solves problems unaided. Training on only 128 unsolvable problems matches or exceeds GRPO trained on a full 2,000-problem corpus (~16x data efficiency) on the nine-benchmark average for both base models, while substantially expanding the reasoning boundary measured by pass@k at large k. Furthermore, we identify a distribution-shift c
    
[^138]: Mizan：评估大语言模型在伊拉克阿拉伯语及伊拉克公民社会语境下表现的国家基准

    Mizan: A National Benchmark for Evaluating Large Language Models on Iraqi Arabic and the Iraqi Civic Context

    [https://arxiv.org/abs/2609.13980](https://arxiv.org/abs/2609.13980)

    该论文提出了 Mizan——伊拉克首个用于评估大语言模型在伊拉克阿拉伯语方言及伊拉克公民社会语境下表现的国家基准，其包含 MSA 基线与伊拉克双赛道、六大评估维度、340 道经双重审核的原创题目，并对 27 个系统进行了初步评测。

    

    arXiv:2609.13980v1 公告类型：cross 摘要：阿拉伯语大语言模型（LLM）的评估体系已围绕现代标准阿拉伯语（MSA）走向成熟：诸如 Open Arabic LLM Leaderboard（OALL）、HELM Arabic 和 BALSAM 等聚合排行榜在数十个 MSA 任务上对模型进行排名，而前沿系统正日益在这些任务上趋于饱和。然而，方言阿拉伯语——伊拉克人日常实际使用的语言——在这一评估基础设施中几乎完全缺席。我们推出了 Mizan（意为“天平”），这是伊拉克用于评估大语言模型在伊拉克阿拉伯语及伊拉克公民社会语境下表现的国家基准：它由一条 MSA 基线赛道与一条伊拉克赛道组成，涵盖六个维度（方言理解、方言生成、MSA 与伊拉克方言的双向翻译、伊拉克特有知识、官方文件字段提取以及安全性），由 340 道原创撰写、经双重审核的题目构成，所有题目的答案位置均经过统计审计，且每个已公布的分数均附有 Wilson 置信区间。针对 27 个系统的初步评估，涵盖闭源前沿模型……（原文摘要在此处截断）

    arXiv:2609.13980v1 Announce Type: cross  Abstract: Arabic large-language-model (LLM) evaluation has matured around Modern Standard Arabic (MSA): aggregated leaderboards such as the Open Arabic LLM Leaderboard (OALL), HELM Arabic, and BALSAM rank models across dozens of MSA tasks, and frontier systems increasingly saturate them. Dialectal Arabic, the language Iraqis actually speak, remains nearly invisible to this infrastructure. We introduce Mizan ("the balance"), Iraq's national benchmark for evaluating LLMs on Iraqi Arabic and the Iraqi civic context: an MSA baseline track paired with an Iraqi track across six axes (dialect comprehension, dialect generation, bidirectional MSA-Iraqi translation, Iraq-specific knowledge, official-document field extraction, and safety), built from 340 originally authored, dually reviewed items with statistically audited answer positions and Wilson intervals on every published score. A pilot evaluation of 27 systems, spanning closed frontier models three
    
[^139]: 没有系统性的思维？评估推理模型在规则归纳任务上的表现

    Thought without systematicity? Evaluating reasoning models on rule induction tasks

    [https://arxiv.org/abs/2609.13948](https://arxiv.org/abs/2609.13948)

    研究通过将认知科学的规则归纳任务构造为结构等价的同构变体，发现推理模型虽能解决原任务却常在变体上失败，表明其行为缺乏系统性，难以证明其具备真正的认知能力。

    

    系统性是人类认知的核心原则之一，即理解一个概念本质上与理解该概念的相近变体紧密相关。推理模型是否能稳健地展现这种系统性？如果能，我们预期它们在结构等价的任务变体上应有一致的表现。本研究将认知科学中经典的规则归纳任务加以扩展，以评估当前推理模型思维的系统性。每个任务族都具有组合结构，我们利用这种结构通过任务同构（如重组和替换）创建结构等价的任务变体。我们发现，尽管模型能够正确解决某个任务，但在同一任务的结构等价变体上却常常失败。这些发现表明，许多模型行为缺乏系统性，因而难以稳健地确立推理模型超越特定任务表现的真实认知能力。

    arXiv:2609.13948v1 Announce Type: cross  Abstract: A central tenet of human cognition is systematicity, the principle that understanding one concept is inherently tied to understanding close variations of that concept. Do reasoning models robustly exhibit such systematicity? If so, we would expect consistent performance on structurally equivalent variants of the same task. Here, we extend established rule induction tasks from cognitive science to assess the systematicity of thought in current reasoning models. Each task family has compositional structure that we use to create structurally equivalent task variations through task isomorphisms such as recombination and substitution. We find that despite being able to correctly solve a task, models often fail on structurally equivalent variants of the same task. These findings suggest that many model behaviors lack systematicity, rendering it difficult to robustly establish the cognitive abilities of reasoning models beyond the particular 
    
[^140]: CRITICS——无国界批判科学：利用语言模型促进科学教育中的批判性思维

    CRITICS - Critical Science Without Borders: Language Models to Promote Critical Thinking in Science Education

    [https://arxiv.org/abs/2609.13942](https://arxiv.org/abs/2609.13942)

    CRITICS项目将基于大语言模型的机器翻译与教育技术相结合，为学生提供母语准确翻译的科学材料，并通过科学论证、批判性思维实践和能力评估框架来促进科学教育中的批判性思维。

    

    CRITICS项目通过将基于大语言模型（LLMs）的先进机器翻译（MT）技术与教育技术相融合，致力于解决科学可及性与科学素养问题。通过利用专门针对科学内容优化的机器翻译系统，教育机构能够以学生的母语提供准确且符合文化背景的科学材料翻译，确保复杂的科学概念易于理解，同时保持技术准确性。基于这些翻译成果，该项目探索以课程对齐的教与学为基础的创新科学教学方案的设计与评估。因此，CRITICS将研究科学论证与批判性思维实践的关键组成部分，以及与学习目标相一致、受基于能力的评价框架启发的文本反馈。CRITICS旨在打破语言障碍，提升科学教育的普及性与深度。

    arXiv:2609.13942v1 Announce Type: cross  Abstract: The CRITICS project addresses science accessibility and literacy by converging advanced Machine Translation (MT) based on Large Language Models (LLMs) with educational technology. By leveraging MT systems specifically optimized for scientific content, educational institutions can provide accurate, culturally relevant translations of scientific materials in students' native languages, ensuring that complex scientific concepts are comprehensible while maintaining technical accuracy. Building on these translations, the project explores the design and evaluation of innovative science teaching-learning proposals grounded in curriculum-aligned teaching-learning. Thus, CRITICS will investigate key components of scientific argumentation and critical thinking practices together with textual feedback aligned with learning objectives and assessment criteria inspired by competence-based evaluation frameworks. CRITICS aims to break down language ba
    
[^141]: 推理型叙事特征上大语言模型与基于规则标注的评分者间信度：基于土耳其语语料库的三项研究

    Inter-Rater Reliability of LLM and Rule-Based Annotation for Inferential Narrative Features: Three Studies on a Turkish Corpus

    [https://arxiv.org/abs/2609.13936](https://arxiv.org/abs/2609.13936)

    该论文通过三项人类标注者信度研究，评估了基于规则检测器及多个大语言模型（Gemini、Grok、Claude、ChatGPT）在土耳其语叙事语料库六项推理型叙事特征标注上与人类评分者的一致性。

    

    附带自动生成特征标注的数据集引出了一个很少被问及的问题：人类会认同这些标签吗？本报告针对Objective Projection语料库回答了这一问题。该语料库是一个土耳其语叙事数据集，其场景带有来自基于规则检测器的逐场景applied_rules字段，涵盖六项叙事写作技巧特征——两项禁忌性特征（情感标注、明喻）和四项正面技巧（实体化隐喻、微观聚焦、时间锚点、氛围矛盾）。报告呈现了三项研究：研究1（n=120）以标注方案作者本人的盲标为基准对检测器进行评分；研究2（n=100，一组互不相交的场景）以一位独立非专家评分者为基准对检测器以及Gemini 2.5 Flash和Grok进行评分，该评分者的标签在任何机器运行之前已被锁定；研究2b则使用Claude Fable 5（High）和ChatGPT 5.5重复了完全相同的实验协议。核心结果涉及其中一条规则。在实体化隐喻方面——（原文在此处截断）

    arXiv:2609.13936v1 Announce Type: new  Abstract: Datasets that ship automatically generated feature annotations invite a question rarely asked of them: would a human agree with those labels? This report answers that for the Objective Projection corpus, a Turkish narrative dataset whose scenes carry a per-scene applied_rules field from a rule-based detector over six craft features -- two prohibitions (emotion labelling, simile) and four positive techniques (materialized metaphor, micro-focus, temporal anchor, atmosphere contradiction).   Three studies are reported. Study 1 ($n = 120$) scores the detector against blind labels from the scheme's own author. Study 2 ($n = 100$, a disjoint scene set) scores the detector plus Gemini 2.5 Flash and Grok against an independent non-expert rater whose labels were locked before any machine ran. Study 2b re-runs the identical protocol with Claude Fable 5 (High) and ChatGPT 5.5.   The central result concerns one rule. On materialized metaphor -- clos
    
[^142]: North Small Translate：先进且高性价比的翻译模型（Cohere CAT+）

    North Small Translate: Advanced Cost-Effective Translation (Cohere CAT+)

    [https://arxiv.org/abs/2609.13916](https://arxiv.org/abs/2609.13916)

    Cohere推出的开放权重机器翻译模型North Small Translate，基于混合专家架构，通过难度采样和五步训练协议（结合监督微调、直接偏好优化和在线强化学习），在1万亿参数以下的模型中实现了50种语言上的顶尖翻译性能。

    

    我们推出了North Small Translate，这是一个开放权重的、基于大语言模型的机器翻译（MT）模型，具备指令遵循能力，其构建基础与Cohere的Command A Plus相同——一种混合专家（MoE）架构，总参数量为2180亿，激活参数量为250亿。North Small Translate采用难度采样来获取具有挑战性的文档进行训练，并采用五步训练协议，结合了监督微调、直接偏好优化和在线强化学习。我们通过非推理基础模型优先保证了吞吐量，并辅以可选的智能体能力来解锁翻译质量的提升。North Small Translate经训练可执行机器翻译相关任务，包括译后编辑和质量评估，以及通用指令遵循等相关任务。该模型在参数量低于1万亿的模型类别中，于50种语言上实现了顶尖的机器翻译性能。

    arXiv:2609.13916v1 Announce Type: new  Abstract: We present North Small Translate, an open-weight, LLM-based machine translation (MT) model with instruction-following capabilities built on the same foundation as Cohere's Command A Plus, a mixture-of-experts architecture with 25 billion active parameters out of 218 billion total parameters. North Small Translate is trained using difficulty sampling to obtain challenging documents and a five-step training protocol combining supervised fine-tuning, direct preference optimization, and online reinforcement learning. We prioritized throughput through a non-reasoning base model and supplemented with optional agentic capabilities to unlock translation quality gains. North Small Translate is trained to perform MT-related tasks, including post-editing and quality estimation, as well as related tasks such as general instruction following. The model achieves top MT performance across 50 languages in the class of models under 1T parameters, with no
    
[^143]: Phorecaster365：一种用于混合药物销售预测与规划决策支持的人工监督参考架构

    Phorecaster365: A Human-Supervised Reference Architecture for Hybrid Pharmaceutical Sales Forecasting and Planning Decision Support

    [https://arxiv.org/abs/2609.13907](https://arxiv.org/abs/2609.13907)

    该论文提出Phorecaster365——一个人工监督的参考架构，以“预测上下文包”为核心中间表示，将ERP数据与可审查的药物销售预测相连接，支持统计与机器学习混合建模、不确定性评估及规划师审查的全流程治理。

    

    药物销售预测为跨产品、地区和分销渠道的规划提供依据，但其解读依赖于库存可用性、交易语义、产品生命周期以及每次预测发布时可获得的信息。仅凭模型预测无法保留这些条件，也无法判断某项预测是否适合用于实际运营。我们提出了Phorecaster365，这是一种将企业资源计划（ERP）数据与可审查的药物销售预测相连接的人工监督参考架构。该架构将源数据摄取、产品-地区时间序列构建、时间有效特征生成、统计与机器学习建模、集成模型构建、不确定性评估、规划师审查以及生命周期治理进行模块化分离。其核心中间表示是“预测上下文包”，它保存了源数据快照、预测目标、时间截断点及可用的上下文信息……

    arXiv:2609.13907v1 Announce Type: cross  Abstract: Pharmaceutical sales forecasts inform planning across products, regions, and distribution channels, yet their interpretation depends on inventory availability, transaction semantics, product lifecycle, and the information available when each forecast is issued. A model prediction alone does not preserve these conditions or establish whether a forecast is suitable for operational use. We present Phorecaster365, a human-supervised reference architecture that connects enterprise resource planning data to reviewable pharmaceutical sales forecasts. The architecture separates source ingestion, product-region series construction, temporally valid feature generation, statistical and machine-learning modeling, ensemble formation, uncertainty assessment, planner review, and lifecycle governance. Its central intermediate representation is a forecast context package that preserves the source snapshot, forecast target, temporal cutoff, available co
    
[^144]: URCHIN：一种面向数据受限预训练的水平脉冲神经网络语言模型

    URCHIN: A Horizontal Spiking Language Model for Data-Constrained Pretraining

    [https://arxiv.org/abs/2609.13899](https://arxiv.org/abs/2609.13899)

    URCHIN是一种受生物学启发的脉冲神经网络语言模型，仅用128个神经元、423万参数、无注意力机制，通过戴尔定律横向连接组和多传输循环动力学，在儿童规模数据上实现语言建模预训练。

    

    BabyLM挑战赛衡量的是模型能从发育合理、儿童规模的数据中学习到多少语言，而非从互联网规模的语料库中学习，然而以往的语言模型忽视了习得人类语言的神经回路所具有的生物学约束：由兴奋性和抑制性群体组成的脉冲神经元，并通过循环横向连接组相连。本文提出了URCHIN（具有水平积分发放神经元的统一循环连接组），它将并行化层次连接组脉冲状态空间模型（PHCSSM）应用于语言建模：由戴尔定律横向连接组耦合的泄漏积分发放神经元，通过将活动再循环至不动点的多传输循环来处理每个词元。该实例的设计刻意保持精简：仅包含单个128个神经元的水平层，没有注意力机制，参数量为423万。两个实现共享同一组权重并产生相同的……

    arXiv:2609.13899v1 Announce Type: cross  Abstract: The BabyLM challenge measures how much language a model can learn from developmentally-plausible, child-scale data rather than internet-scale corpora, yet prior language models forgo the biological constraints of the neural circuitry that acquires human language: spiking neurons separated into excitatory and inhibitory populations wired by a recurrent lateral connectome. This paper presents URCHIN (Unified Recurrent Connectome with Horizontal Integrate-and-fire Neurons), which applies the Parallelized Hierarchical Connectome Spiking State-space Model (PHCSSM) to language modeling: leaky integrate-and-fire neurons coupled by a Dale's-law lateral connectome resolve each token through a multi-transmission loop that recirculates activity to a fixed point. The instantiation is deliberately minimal: a single horizontal layer of 128 neurons, no attention, and 4.23M parameters. Two implementations share one set of weights and produce identical
    
[^145]: SHIFT-M3：基于融合前对齐的一致性筛查方法用于多模态心电图记录完整性检测

    SHIFT-M3: Pre-fusion Alignment-based Consistency Screening for Multimodal ECG Record Integrity

    [https://arxiv.org/abs/2609.13874](https://arxiv.org/abs/2609.13874)

    提出轻量级融合前筛查方法SHIFT-M3，通过测量LLM生成的心电图解读与临床报告摘要之间的一致性，有效检测多模态心电记录中因数据关联错误导致的跨患者组件错配问题。

    

    多模态临床AI通常假设附加在一条记录上的波形、报告、元数据和下游预测属于同一位患者。在实际应用中，数据关联失败可能会悄悄地将各自看似合理但来自不同患者的组件拼装在一起，造成标准预测模型无法检测到的安全问题。我们将该问题定义为多模态记录完整性分诊：给定一条组装好的记录，其各模态是否可以被信任为属于同一位患者？我们提出了SHIFT-M3，一个轻量级的基于文本的融合前筛查方法，用于测量两个独立生成的心电图文本视图之间的基于对齐的一致性：LLM生成的解读和临床报告摘要。在784,680条MEETI心电图记录上，SHIFT-M3仅使用573,569个参数，对完整文本视图交换实现了97.6%的TPR@5% FPR（AUROC 0.996），对部分交换实现了90.3%（AUROC 0.974），对标签匹配的困难负样本实现了97.7%（AUROC 0.996）。

    arXiv:2609.13874v1 Announce Type: new  Abstract: Multimodal clinical AI typically assumes that the waveform, report, metadata, and downstream predictions attached to a record belong to the same patient. In practice, linkage failures can silently assemble individually plausible but cross-patient components, creating a safety problem that standard predictive models are not designed to detect. We study this problem as multimodal record integrity triage: given an assembled record, should its modalities be trusted to belong together? We introduce SHIFT-M3, a lightweight text-based pre-fusion screen that measures alignment-based consistency between two separately produced ECG text views: an LLM-generated interpretation and a clinical report summary. On 784,680 MEETI ECG records, SHIFT-M3 achieves 97.6% TPR@5% FPR for full text-view swaps (AUROC 0.996), 90.3% for partial swaps (AUROC 0.974), and 97.7% for label-matched hard negatives (AUROC 0.996) with only 573,569 parameters. Compared with s
    
[^146]: 孟加拉语句子功能分类：语料库构建、模型基准测试与可解释性

    Bangla Sentence Function Classification: Corpus Development, Model Benchmarking, and Interpretability

    [https://arxiv.org/abs/2609.13869](https://arxiv.org/abs/2609.13869)

    本文构建了一个包含10,000个孟加拉语句子、人工标注为四种功能类别的高质量语料库，并通过经典机器学习模型和异构集成模型对该句子功能分类任务进行了基准测试与可解释性分析。

    

    自动句子功能识别对于许多下游自然语言处理（NLP）应用非常重要，例如对话系统、语音合成和机器翻译。然而，孟加拉语句子功能分类的基准资源仍然有限。为了弥补这一空白，本文构建了一个包含10,000个孟加拉语句子的语料库，并由人工标注为四种功能类别，即陈述句、疑问句、祈使句和感叹句。该语料库在四个类别上分布几乎均衡，标注可靠性高，Fleiss' Kappa达到0.82。此外，我们评估了多种特征表示方法，包括词袋模型（BoW）、TF-IDF和Word2Vec，并与多种经典机器学习分类器相结合。同时，本文还采用了两种异构集成模型，即单层集成（SLE）和双层集成（DLE），以提升分类性能。实验……

    arXiv:2609.13869v1 Announce Type: cross  Abstract: Automatic sentence function identification is important for many downstream natural language processing (NLP) applications such as dialogue systems, text-to-speech synthesis, and machine translation. However, benchmark resources for Bangla sentence function classification remain limited. To mitigate this gap, this paper introduces a corpus of 10,000 Bangla sentences, manually annotated into four functional categories, namely declarative, interrogative, imperative, and exclamatory. The corpus is nearly balanced across the four classes, with high annotation reliability reflected by a Fleiss\' Kappa of 0.82. Furthermore, we evaluate multiple feature representations, including Bag-of-Words (BoW), TF-IDF, and Word2Vec, with several classical machine learning classifiers. In addition, two heterogeneous ensemble models, namely Single-Level Ensemble (SLE) and Double-Level Ensemble (DLE), are utilized to improve classification performance. Expe
    
[^147]: ClinAgent：一个基于ReAct的对话式临床试验信息访问智能体

    ClinAgent: A ReAct-Based Agent for Conversational Access to Clinical Trial Information

    [https://arxiv.org/abs/2609.13860](https://arxiv.org/abs/2609.13860)

    该论文提出了ClinAgent，一个基于ReAct范式的智能体检索增强生成对话系统，通过集成ClinicalTrials.gov、PubMed等工具，让临床医生和研究人员能够用自然语言多轮查询临床试验信息并获得有依据的最新答案。

    

    查询临床试验注册库目前仍是一个手动且容易出错的过程，研究人员需要在缺乏自然语言交互或跨数据源综合支持的情况下，浏览大量半结构化数据。为解决这一问题，我们提出了ClinAgent，一个基于智能体检索增强生成（RAG）的对话式系统，使临床医生和研究人员能够用自然语言查询临床试验信息，并在多轮交互中获得有依据的、最新的回答。该系统的核心是一个遵循ReAct范式的大语言模型（LLM）智能体，它对查询进行迭代推理，在一组集成工具中进行选择，并根据中间输出不断优化其行动。这些工具包括ClinicalTrials.gov搜索接口、PubMed模块，以及一个基于本地缓存的临床试验结构化数据集运行的Python分析器。我们使用三阶段（评估方法对系统进行了评估）

    arXiv:2609.13860v1 Announce Type: new  Abstract: Querying clinical trial registries remains a manual and error-prone process, requiring researchers to navigate large volumes of semi-structured data without support for natural language interaction or cross-source synthesis. To address this, we introduce ClinAgent, a conversational system based on agentic Retrieval-Augmented Generation (RAG) that enables clinicians and researchers to query clinical trial information in plain language and receive grounded, up-to-date responses across multi-turn interactions. The system centers on a Large Language Model (LLM) agent following the ReAct paradigm, which iteratively reasons over queries, selects among a set of integrated tools, and refines its actions based on intermediate outputs. These tools include a ClinicalTrials.gov search interface, a PubMed module, and a Python-based analyzer operating on a locally cached structured dataset of clinical trials. We evaluate the system using a three-phase
    
[^148]: ShopEase：一种基于生成式AI的多智能体框架，利用混合检索增强生成实现智能企业客户支持

    ShopEase: A Generative AI-Based Multi-Agent Framework for Intelligent Enterprise Customer Support Using Hybrid Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.13856](https://arxiv.org/abs/2609.13856)

    本文提出ShopEase框架，通过结合意图、CRM、记忆、混合RAG（FAISS稠密检索+BM25稀疏检索）、升级和监督六大智能体组件，直接从检索文档中确定政策类别而非依赖固定映射，并利用本地运行的LLaMA 3.2生成回复，实现智能企业客户支持。

    

    企业客户支持系统必须正确回答客户问题、检索正确的政策信息、利用客户上下文，并在需要时将困难案例转交给人工客服。本文提出了ShopEase，一个基于生成式AI的企业客户支持多智能体框架。该系统结合了六个组件：意图、CRM、记忆、混合RAG、升级和监督者，并使用通过Ollama本地运行的LLaMA 3.2进行响应生成。检索模块结合了FAISS（稠密检索）和BM25（稀疏检索），并评估了六种配置：仅BM25、仅FAISS、公平RRF、加权RRF、带交叉编码器的RRF，以及带交叉编码器的Top-10混合检索。与使用意图和政策类别之间的固定映射不同，政策类别直接从检索到的文档中确定。该系统在2632个保留客户查询上进行了评估，涵盖六个类别：退款、退货、运输、取消等。

    arXiv:2609.13856v1 Announce Type: new  Abstract: Enterprise customer support systems must answer customer questions correctly, retrieve the right policy information, use customer context, and pass difficult cases to human agents when needed. This paper presents ShopEase, a Generative AI-based multi-agent framework for enterprise customer support. The system combines six components: Intent, CRM, Memory, Hybrid RAG, Escalation, and Supervisor, and uses LLaMA 3.2 running locally through Ollama for response generation. The retrieval module combines FAISS (dense retrieval) and BM25 (sparse retrieval), and six configurations are evaluated: BM25-only, FAISS-only, Fair RRF, Weighted RRF, RRF with Cross-Encoder, and Top-10 Hybrid with Cross-Encoder. Instead of using a fixed mapping between intent and policy, the policy category is decided directly from the retrieved documents. The system was evaluated on 2632 held-out customer queries across six categories: Refund, Return, Shipping, Cancellatio
    
[^149]: 衡量多语言机器翻译评估中语言变体混淆的代价：将莫桑比克尚加纳语、尼扬贾语和塞纳语加入FLORES+

    Measuring the Cost of Variety Conflation in Multilingual MT Evaluation: Adding Mozambican Xichangana, Nyanja and Sena to FLORES+

    [https://arxiv.org/abs/2609.13847](https://arxiv.org/abs/2609.13847)

    该研究将三种莫桑比克班图语变体（尚加纳语、尼扬贾语和塞纳语）纳入FLORES+基准，首次量化了机器翻译评估中语言变体混淆的代价——使用相近变体作参考译文会使spBLEU分数被严重低估（最高达15分以上），而变体感知微调可使模型在目标变体上提升5至7个spBLEU点。

    

    在这篇论文中，我们扩展了FLORES+基准，为三种莫桑比克班图语变体添加了以葡萄牙语为源语言的评估集：尚加纳语、莫桑比克尼扬贾语和塞纳语。我们将尚加纳语与现有的聪加语参考译文进行比较，将莫桑比克尼扬贾语与齐切瓦语进行比较，并评估了NLLB-200、谷歌翻译、GPT以及一个变体感知的NLLB模型。固定系统输出后发现存在显著的参考敏感性。在devtest集上，仅将参考译文从聪加语更改为尚加纳语，就使NLLB-200的spBLEU降低了13.10分，谷歌翻译降低了15.30分。在匹配的尼扬贾语子集上，用莫桑比克尼扬贾语替换齐切瓦语产生了较小但一致的下降，分别为3.03和6.10 spBLEU。变体感知微调在预期目标上扭转了这一模式：相对于NLLB-200，它在devtest上将尚加纳语提高了7.04 spBLEU，将莫桑比克尼扬贾语提高了5.33，而在同源变体参考译文上性能有所下降。GPT在聪加语上具有竞争力

    arXiv:2609.13847v1 Announce Type: new  Abstract: In this paper, we extend FLORES+ with Portuguese-source evaluation sets for three Mozambican Bantu varieties: Xichangana, Mozambican Nyanja, and Sena. We compare Xichangana with the existing Tsonga reference and Mozambican Nyanja with Chichewa, and evaluate NLLB-200, Google Translate, GPT, and a variant-aware NLLB model. Holding system output fixed reveals substantial reference sensitivity. On \textit{devtest}, changing only the reference from Tsonga to Xichangana reduces spBLEU by 13.10 points for NLLB-200 and 15.30 for Google. On matched Nyanja subsets, replacing Chichewa with Mozambican Nyanja produces smaller but consistent reductions of 3.03 and 6.10 spBLEU, respectively. Variant-aware fine-tuning reverses this pattern on the intended targets: relative to NLLB-200, it improves Xichangana by 7.04 spBLEU and Mozambican Nyanja by 5.33 on \textit{devtest}, while losing performance on the sibling references. GPT is competitive on Tsonga 
    
[^150]: 面向延迟张量并行的亲和性感知分片

    Affinity-Aware Sharding for Delayed Tensor Parallelism

    [https://arxiv.org/abs/2609.13846](https://arxiv.org/abs/2609.13846)

    本文发现延迟张量并行会打破FFN神经元和KV头的置换对称性，使分片本身成为建模决策，并提出通过在分片前置换稠密模型以最大化同设备上KV头与FFN神经元的亲和性，从而加速模型蒸馏或重训练过程。

    

    延迟张量并行消除了张量并行Transformer推理中的阻塞式全归约操作。每个设备立即将自己的部分输出加到其残差流中（并进行广播），但仅在δ个模块之后才收集其他设备的部分输出。因此，从张量并行到延迟张量并行的改变相当于一次真正的架构改变，稠密Transformer模型需要在适配后进行重新训练或蒸馏。我们证明延迟张量并行打破了前馈网络内部神经元以及注意力模块内部KV头的置换对称性，而这种对称性破坏使得分片本身成为一种建模决策。我们表明，通过在分片前对稠密模型进行置换，最大化共置于同一设备上的KV头与FFN神经元之间的亲和性，可以加速蒸馏或重新训练过程。亲和性通过一阶近似来度量，该近似衡量失去某个头的贡献对每个神经元造成的损害。

    arXiv:2609.13846v1 Announce Type: cross  Abstract: Delayed Tensor Parallelism (DTP) removes the blocking all-reduce of tensor-parallel Transformer inference. Every device adds its own partial output to its residual stream (and broadcasts it) immediately, but only gathers (receives) the other devices' partials $\delta$ modules later. A TP to DTP change therefore amounts to a real architecture change, and dense Transformer models need to be retrained or distilled after adaptation. We show that DTP breaks the permutation symmetry of neurons inside FFNs and of KV heads inside attention modules, and that this symmetry breakage makes the sharding itself a modelling decision. We show that maximising the affinity between the KV heads and the FFN neurons co-located on a device, by permuting the dense model before sharding, speeds up the distillation or retraining process. The affinity is measured with a first-order approximation of the damage that losing a head's contribution does to each neuro
    
[^151]: 《甜言蜜语者：提问方式如何塑造浪漫关系建议中的谄媚行为》

    Sweet Talkers: How Query Formulation Shapes Sycophancy in Romantic Relationship Advice

    [https://arxiv.org/abs/2609.13841](https://arxiv.org/abs/2609.13841)

    该研究构建了包含2400条提示的浪漫关系建议数据集，发现大语言模型的谄媚行为主要受用户视角框架化表述而非语法语气的影响，且谄媚程度会随对话轮次持续增加。

    

    大型语言模型（LLM）越来越多地被用于情感支持和人际关系建议，而模型为维护用户面子（保全面子）的倾向可能会在无意中强化有害的人际行为。为了系统地检验这一风险，我们构建了浪漫关系建议寻求提示数据集，其中包含覆盖五个关系主题的2400条提示，并使用ELEPHANT框架在两款面向消费者的模型（GPT-5 Mini和Gemini 3 Flash）上评估了社会性谄媚行为。与我们的初始假设相反，语法语气本身并未在谄媚行为上产生系统性差异，这表明用户所暗示的内容比其措辞方式更为重要。相比之下，以视角驱动的框架化表述具有更强的影响力，且原始提示与翻转提示之间的差距在后续回复中不断扩大。跨对话轮次中框架化谄媚和道德谄媚的持续增加表明，模型越来越倾向于接受……

    arXiv:2609.13841v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for emotional support and relationship advice, where a model's tendency to preserve a user's face can inadvertently reinforce harmful interpersonal behaviors. To systematically examine this risk, we developed the Romantic Relationship Advice-Seeking Prompts (RRASP) dataset of 2,400 prompts across five relationship themes and evaluated social sycophancy using the ELEPHANT framework on two consumer-facing models, GPT-5 Mini and Gemini 3 Flash. Contrary to our initial hypothesis, grammatical mood alone did not produce systematic differences in sycophantic behavior, suggesting that what a user implies matters more than how they phrase it. Instead, perspective-driven framing had a stronger influence, with gaps between original and flipped prompts widening in follow-up responses. Consistent increases in framing and moral sycophancy across turns indicate that models become more likely to accept
    
[^152]: 当一致性并不意味着可靠性：将本地LLM评判者与人类评分进行对比评估

    When Consistency Does Not Mean Reliability: Evaluating Local LLM Judges Against Human Ratings

    [https://arxiv.org/abs/2609.13824](https://arxiv.org/abs/2609.13824)

    该研究通过对LLaMA-3-8B和Qwen2.5-7B两个本地LLM评判者的系统评估发现，即使评判者的评分具有自一致性，其与人类评分的相关性也很低（皮尔逊相关系数仅0.275），说明“LLM即评判者”方法的一致性并不能等同于与人类判断的可靠性。

    

    大语言模型（LLM）越来越多地被用于评估其他语言模型的回答，这种方法被称为“LLM即评判者”，它比人工评估更快、成本更低。然而，评判模型可能产生一致的分数，却不一定与人类评估者的意见相符。在这项工作中，我们使用两个本地开源权重的LLM评判者——LLaMA-3-8B和Qwen2.5-7B——来研究这一问题。我们评估了由经过指令微调的GPT-2（124M）模型针对100个问题生成的300条回答，这些问题涵盖五个类别：事实知识、指令遵循、数学、推理和写作。每条回答由九名人类标注者进行评分，并由每个LLM评判者使用相同的评分标准评估三次。我们使用皮尔逊相关系数、斯皮尔曼相关系数、平均绝对误差（MAE）、符号偏差和自一致性等指标，将评判者的分数与人类平均分数进行比较。LLaMA-3-8B与人类分数的皮尔逊相关系数为0.275，而……（原文摘要在此处截断）

    arXiv:2609.13824v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to evaluate the responses of other language models. This approach, known as LLM-as-a-Judge, is faster and cheaper than human evaluation. However, a judge may produce consistent scores without necessarily agreeing with human evaluators. In this work, we study this issue using two local open-weight LLM judges, LLaMA-3-8B and Qwen2.5-7B. We evaluate 300 responses generated by an instruction-tuned GPT-2 (124M) model for 100 questions covering five categories: factual knowledge, instruction following, mathematics, reasoning, and writing. Each response is scored by nine human annotators and is evaluated three times by each LLM judge using the same rubric. We compare the judge scores with the average human scores using Pearson correlation, Spearman correlation, mean absolute error (MAE), signed bias, and self-consistency. LLaMA-3-8B shows a Pearson correlation of 0.275 with human scores, while 
    
[^153]: DARE：用于结构化知识事实核查的辩证式智能体推理

    DARE: Dialectical Agentic Reasoning for Structured Knowledge Fact Checking

    [https://arxiv.org/abs/2609.13808](https://arxiv.org/abs/2609.13808)

    提出了DARE多智能体框架，将结构化知识事实核查形式化为迭代式“检索-推理-反思”过程，通过基于关系的证据检索、辩证式双向验证和置信度驱动的元反思，解决了现有程序生成方法中无效关系生成、单路径推理缺乏自我修正以及证据评估偏向支持性信号的问题。

    

    结构化知识事实核查旨在通过推理结构化证据来确定自然语言论断的真实性。近期的程序生成方法利用大型语言模型（LLM）生成可执行的图推理程序，在结构化知识事实核查基准上取得了优异表现。然而，这些方法仍然受限于无效关系生成、缺乏自我修正的单路径推理，以及倾向于高估支持性信号的有偏证据评估。我们提出了辩证式智能体推理（Dialectical Agentic Reasoning，DARE），这是一个多智能体框架，将结构化知识事实核查形式化为一个迭代的“检索-推理-反思”过程。DARE集成了基于关系的证据检索，将推理约束在有效结构上；采用辩证式双向验证，从支持和反驳两个角度评估证据；并引入置信度驱动的元反思……（原文摘要在此处截断）

    arXiv:2609.13808v1 Announce Type: new  Abstract: Structured knowledge fact checking aims to determine the truthfulness of natural language claims by reasoning over structured evidence. Recent program-generation approaches leverage large language models (LLMs) to generate executable graph reasoning programs, achieving strong performance on structured knowledge fact checking benchmarks. However, these methods remain limited by invalid relation generation, single-path reasoning that lacks self-correction, and biased evidence assessment that tends to overestimate supporting signals. We propose Dialectical Agentic Reasoning (DARE), a multi-agent framework that formulates structured knowledge fact checking as an iterative retrieve-reason-reflect process. DARE integrates relation-grounded evidence retrieval to constrain reasoning to valid structures, dialectical bidirectional verification to evaluate evidence from both supporting and refuting perspectives, and confidence-driven meta-reflectio
    
[^154]: 理解智能体ICD编码的局限性

    Understanding the Limits of Agentic ICD Coding

    [https://arxiv.org/abs/2609.13806](https://arxiv.org/abs/2609.13806)

    本研究通过在按稀有程度分层的MIMIC-IV出院小结上评估神经、工作流和智能体三类ICD编码系统，揭示了两类正交的失败模式，并证明结合结构化官方参考资料的工具增强智能体可在损伤和外因编码等困难场景上恢复高达0.34的微F1，但没有任何单一系统在所有条件下占优。

    

    ICD-10-CM编码是美国用于对诊断和损伤进行分类的字母数字编码，应用于医疗计费和流行病学报告。标准的ICD-10-CM基准测试所报告的总体指标掩盖了复杂编码场景下的性能表现。我们在按稀有程度分层的MIMIC-IV出院小结数据集上评估了神经、工作流和智能体三类系统，并识别出两种正交的失败模式。神经分类器在稀有编码与常见编码之间表现出0.43的微F1差距。工作流系统能较好地处理稀有编码，但在需要遵循多步骤指南的损伤和外因编码上得分接近于零。一种通过结构化方式访问官方ICD-10-CM参考资料的工具增强型智能体配置，在该子集上可恢复高达0.34的微F1。没有任何单一系统在所有条件下都占据主导地位。

    arXiv:2609.13806v1 Announce Type: cross  Abstract: ICD-10-CM codes are alphanumeric codes used in the US to classify diagnoses and injuries for medical billing and epidemiological reporting. Standard ICD-10-CM benchmarks report aggregate metrics that obscure performance on complex coding scenarios. We evaluate neural, workflow, and agentic systems on a rarity-stratified set of MIMIC-IV discharge summaries and identify two orthogonal failure modes. Neural classifiers exhibit a 0.43 micro-F1 gap between rare and common codes. Workflow systems handle rare codes well but score near zero on injury and external cause codes that require multi-step guideline following. A tool-augmented agentic configuration with structured access to official ICD-10-CM reference materials recovers up to 0.34 micro-F1 on this subset. No single system dominates across all conditions.
    
[^155]: SyRHM：基于符号-语言增强推理与联想检索的零样本有害迷因检测

    SyRHM: Symbolic-Language-Enhanced Reasoning with Associative Retrieval for Zero-shot Harmful Meme Detection

    [https://arxiv.org/abs/2609.13794](https://arxiv.org/abs/2609.13794)

    该论文提出SyRHM框架，通过将多模态内容解析后检索语义相关的迷因作为有依据的上下文，并利用符号-语言增强的多阶段推理（翻译、规划、求解）实现零样本有害迷因检测，在多个数据集上取得优越性能的同时提供了可解释的有害意图分析。

    

    检测有害迷因对于维护安全的在线社区至关重要。然而，有害意图往往是隐性的，它源于视觉-文本之间的不协调以及文化刻板印象，这给现有的多模态检测器带来了挑战。我们提出了SyRHM框架，该框架将有害迷因检测分解为基于意义的检索和符号-语言增强的多阶段推理。SyRHM通过将多模态内容解析为文本元素和描述来检索语义相关的迷因，从而提供超越表面相似性的有依据的上下文。在检索到的上下文基础上，SyRHM使用翻译器阶段将多模态输入转换为符号中间表示，然后通过规划器和解算器阶段进行多阶段推理，从而实现对有害意图的富有表现力且可解释的分析。在FHM、HarM和MultiOff数据集上的实验证明了SyRHM的有效性，取得了卓越的性能。

    arXiv:2609.13794v1 Announce Type: new  Abstract: Detecting harmful memes is critical for maintaining safe online communities. However, harmful intent is often implicit, arising from visual-textual incongruity and cultural stereotypes, which challenges existing multimodal detectors. We propose SyRHM, a framework that decomposes harmful meme detection into meaning-grounded retrieval and symbolic-language-enhanced multi-stage reasoning. SyRHM retrieves semantically related memes by parsing multimodal content into textual elements and descriptions, providing grounded context beyond surface-level similarity. Building on the retrieved context, SyRHM uses a translator stage to convert multimodal inputs into symbolic intermediate representations, and then performs multi-stage reasoning via planner and solver stages, enabling expressive and interpretable analysis of harmful intent. Experiments on FHM, HarM, and MultiOff demonstrate the effectiveness of SyRHM, achieving superior performance on m
    
[^156]: 推理能力能提升大语言模型的心理深度吗？这取决于评判者是谁

    Does Reasoning Improve Psychological Depth in Large Language Models? It Depends on Who's Judging

    [https://arxiv.org/abs/2609.13773](https://arxiv.org/abs/2609.13773)

    该研究通过短篇小说心理深度的对比实验发现，推理能力更强的模型（如GPT-5）并未获得人类读者的普遍偏好，且读者间与LLM评判器的评价存在显著分歧，表明"LLM作为评判者"的有效性高度依赖于评判者本身。

    

    LLM-as-a-Judge（大语言模型作为评判者）评估器被越来越多地用于对开放式生成内容进行打分，然而，某个评判器在其开发集上与人类评分的相关性，并不能保证在输出内容旗鼓相当且人类偏好本身具有主观性的情况下，其测量结果仍然有效。我们通过短篇小说的心理深度来研究这种失效模式。七位人类读者和一个在原始标量心理深度量表数据集上选出的大语言模型评判器集成（ρ = 0.646），对来自GPT-5与GPT-4o、DeepSeek-R1与DeepSeek-V3的60对盲测、提示词匹配的故事进行了评估。人类偏好并未显示出推理能力的普遍优势：GPT-5相对GPT-4o略受青睐（60.0–62.9%），而DeepSeek-R1则落后于V3（42.9%），读者间一致性接近随机水平（Krippendorff's α = 0.070），而读者内部的一致性和反复出现的权重模式表明，这是结构化的偏好异质性，而非随机反应。相比之下，该评判器……（原文在此处截断）

    arXiv:2609.13773v1 Announce Type: cross  Abstract: LLM-as-a-Judge evaluators are increasingly used to score open-ended generation, yet a judge's correlation with human ratings on its development set may not guarantee valid measurement when outputs are closely matched and human preferences are subjective. We study this failure mode through psychological depth in short stories. Seven human readers and an LLM-judge ensemble selected on the original scalar Psychological Depth Scale dataset ($\rho = 0.646$) evaluated 60 blinded, prompt-matched story pairs from GPT-5 vs.\ GPT-4o and DeepSeek-R1 vs.\ DeepSeek-V3. Human preferences showed no universal reasoning advantage: GPT-5 was modestly preferred over GPT-4o (60.0--62.9\%), whereas DeepSeek-R1 trailed V3 (42.9\%), and inter-reader agreement was near chance (Krippendorff's $\alpha = 0.070$), with within-reader consistency and recurring weighting patterns suggesting structured heterogeneity rather than random responding. The judge, by contra
    
[^157]: 无需推理轨迹训练专家模型实现领域专家蒸馏

    Training Specialist Models without Reasoning Trajectories for Domain Expert Distillation

    [https://arxiv.org/abs/2609.13770](https://arxiv.org/abs/2609.13770)

    该论文发现仅用问答对训练的专家模型会隐式地从潜在轨迹空间中选择推理轨迹，并将学生蒸馏作为探针揭示了专家模型专门化与泛化特征之间的紧密支配关系。

    

    专家蒸馏通过教师模型生成的推理轨迹，有效地将领域专业知识转移给学生模型。然而，当这些专家模型仅在问答对上进行训练而没有显式的推理监督时，是什么在支配它们生成的轨迹？在这项工作中，我们证明专家模型的优化过程隐式地从这个潜在轨迹空间中进行选择。为了隔离并观察这一潜在分布，我们将学生蒸馏不仅作为下游目标，而是作为一种不可知的探针——由于学生模型不继承专家模型的参数化或优化约束，只继承被采样的轨迹本身。通过这一探针，我们的实证分析揭示了一种紧密的支配关系：在27组专家-学生配对中，它们的专门化-泛化特征表现出极强的相关性。关键的是，通过显式控制专家模型的分布漂移

    arXiv:2609.13770v1 Announce Type: cross  Abstract: Specialist distillation effectively transfers domain expertise to student models via teacher-generated reasoning trajectories. However, when these specialists are trained solely on question--answer pairs without explicit reasoning supervision, what governs the trajectories they generate? In this work, we show that specialist optimization implicitly selects from this latent trajectory space. To isolate and observe this latent distribution, we leverage student distillation not as a downstream goal, but as an agnostic probe---since students inherit no parameterization or optimization constraints from the specialist, inheriting only the sampled trajectories themselves. Through this probe, our empirical analysis unveils a tight governing relationship: across 27 specialist--student pairings, their specialization--generalization profiles correlate exceptionally strongly. Crucially, explicitly controlling the specialist's distributional drift 
    
[^158]: HyperProve：面向多跳问答的答案引导超图扩展方法

    HyperProve: Answer-Guided Hypergraph Expansion for Multi-Hop Question Answering

    [https://arxiv.org/abs/2609.13768](https://arxiv.org/abs/2609.13768)

    HyperProve将问题分解与答案条件驱动的超图扩展相结合，把中间答案和支持超边作为检索状态引导迭代检索，从而在多跳问答基准上取得最佳整体性能。

    

    多跳问答在检索将证据视为与原始问题的孤立匹配时往往会失败，因为回答复杂问题所需的事实通常通过中间实体、关系和约束相互连接。我们提出了HyperProve，这是一个检索增强问答框架，通过将问题分解与基于答案条件在原子事实超图上的扩展相结合来应对这一挑战。HyperProve并不孤立地使用原子事实、超图或迭代检索；相反，它将中间答案和支持超边作为检索状态加以携带，然后利用该状态来引导下一次局部超图扩展。这种设计使HyperProve能够为最终答案生成构建连贯的证据链，同时使检索过程具有状态性并以事实为中心。在多个多跳问答基准测试中，HyperProve在我们的评估中取得了最佳的整体表现，超越了现有方法。

    arXiv:2609.13768v1 Announce Type: new  Abstract: Multi-hop question answering often fails when retrieval treats evidence as isolated matches to the original question, since the facts needed to answer a complex question are usually connected through intermediate entities, relations, and constraints. We propose HyperProve, a retrieval-augmented QA framework that addresses this challenge by coupling question decomposition with answer-conditioned expansion over a hypergraph of atomic facts. HyperProve does not use atomic facts, hypergraphs, or iterative retrieval in isolation; instead, it carries intermediate answers and supporting hyperedges as retrieval state, then uses that state to bias the next local hypergraph expansion. This design enables HyperProve to construct coherent evidence chains for final answer generation while making the retrieval process stateful and fact-centered. Across multi-hop QA benchmarks, HyperProve achieves the best overall performance in our evaluation, outperf
    
[^159]: 深入解析VLM图表阅读：跨空间与深度追踪垂直柱状图的数值读取

    Inside VLM Chart Reading: Tracing Value Reading from Vertical Bar Charts Across Space and Depth

    [https://arxiv.org/abs/2609.13745](https://arxiv.org/abs/2609.13745)

    该研究利用反事实激活修补技术揭示了视觉语言模型读取垂直柱状图数值的内部机制：柱顶区域是关键视觉证据来源，数值信息的传递在早期层依赖图例区域、在中间层转移至提示序列位置，且提示序列状态在其中起部分中介作用。

    

    视觉-语言模型（VLM）能够准确回答图表问题，但输出准确率并不能揭示它们是如何组合证据来恢复精确数值的。我们在Qwen2.5VL-7B-Instruct和InternVL3.5-8B上，采用受控的反事实激活修补方法研究垂直柱状图的数值读取。该研究连接了三项分析：（1）单因素结果表明，发生变化的柱顶区域比未变化的柱体能够恢复更多的答案偏好，尽管其包含的视觉标记数量更少。图例和序列相关状态比柱形几何和坐标轴刻度状态更早失去局部可恢复性。（2）在交接分析中，恢复作用从早期层的视觉图例区域转移到中间层的提示序列位置。重置提示序列状态会选择性地减少来自图例来源的救援效果，支持其作为部分中介的角色。（3）在因子分析中，两个模型都能利用几何和刻度信息（原文在此处截断）

    arXiv:2609.13745v1 Announce Type: new  Abstract: Vision--language models (VLMs) can answer chart questions accurately, but output accuracy does not show how they combine the evidence needed to recover an exact value. We study vertical-bar value reading with controlled counterfactual activation patching in Qwen2.5VL-7B-Instruct and InternVL3.5-8B. The study connects three analyses: (1) The single-factor results show that the changed bar-top region restores much more answer preference than the unchanged bar body, despite containing fewer visual tokens. Legend- and series-related states also lose local recoverability earlier than bar-geometry and axis-scale states. (2) In the handoff analysis, restoration shifts from visual legend regions in early layers to prompt-series positions in middle layers. Resetting the prompt-series state selectively reduces legend-source rescue, supporting its role as a partial mediator. (3) In the factorial analysis, both models can use geometry and scale stat
    
[^160]: HarnessBandit：面向多Harness智能体强化学习的联合可学习性-可迁移性调度

    HarnessBandit: Joint Learnability-Transferability Scheduling for Multi-Harness Agentic Reinforcement Learning

    [https://arxiv.org/abs/2609.13739](https://arxiv.org/abs/2609.13739)

    HarnessBandit提出了一种在线调度方法，通过联合观测可学习性和可迁移性两个信号，在每个优化步骤中选择最有价值的训练harness，从而提升多框架智能体强化学习的鲁棒性。

    

    语言模型智能体越来越多地通过多样化的harness（运行框架）进行部署，这些框架在系统提示词、工具模式、控制循环和轨迹格式上各不相同。同一模型在不同接口上的表现可能参差不齐，因此对harness变化的鲁棒性成为一个重要目标。一种自然的方法是通过多个harness训练一个共享策略，但这引入了一个调度问题：每个训练步骤应优先选择当前能提供有用学习信号的harness，同时产生的更新也要对其他harness有益。我们开发了HarnessBandit，这是一种在线调度器，可在每个优化器步骤中选择一个harness。在执行组相对策略优化（GRPO）更新后，它会观测可学习性——即批次上的平均绝对优势——以及可迁移性——即当前harness的低维梯度草图与其余harness指数移动平均之间的余弦相似度。这两个信号……

    arXiv:2609.13739v1 Announce Type: cross  Abstract: Language-model agents are increasingly deployed through diverse harnesses that differ in system prompts, tool schemas, control loops, and trajectory formats. The same model can perform unevenly across these interfaces, making robustness to harness variation an important objective. A natural approach is to train a shared policy through multiple harnesses, but doing so introduces a scheduling problem: each training step should favor a harness that currently provides a useful learning signal while also producing an update that benefits the other harnesses. We develop HarnessBandit, an online scheduler that selects one harness per optimizer step. After a group-relative policy optimization (GRPO) update, it observes learnability -- the mean absolute advantage on the batch -- and transferability -- the cosine between a low-dimensional gradient sketch of the current harness and exponential moving averages of the remaining harnesses. The two s
    
[^161]: ForeSight：通过早期安全信号蒸馏增强风险监控

    ForeSight: Enhancing Risk Monitoring via Early Safety Signal Distillation

    [https://arxiv.org/abs/2609.13737](https://arxiv.org/abs/2609.13737)

    提出ForeSight框架，通过将大语言模型生成首个词元时隐藏状态中微弱冗余的安全信号蒸馏为紧凑的层次感知风险表示，实现了仅凭首个词元即可高效预测最终响应有害性的早期风险监控。

    

    随着大语言模型（LLMs）的日益部署，有害内容的生成已成为一个关键的安全问题。现有的安全防护措施在输入、输出或流式生成阶段运行，而依赖表面词元或输出logits的早期风险方法可能受到微弱初始信号的困扰，使用稠密表示的基于内部状态的检测器则可能保留高度纠缠和冗余的安全无关信息。因此，目前尚不清楚生成后最早的隐藏状态是否已经包含关于最终响应有害性的可靠信号。为了解决这一空白，我们提出了ForeSight，一个首词元输出风险预测框架，它将微弱且冗余的早期安全信号蒸馏为紧凑的、层次感知的风险表示。在五个安全基准和两个目标模型上的实验表明，ForeSight仅依靠早期信号即可实现卓越且高效的早期风险预测。

    arXiv:2609.13737v1 Announce Type: new  Abstract: As large language models (LLMs) are increasingly deployed, the generation of harmful content has become a critical safety concern. Existing safeguards operate at the input, output, or streaming-generation stages, while early-risk methods that rely on surface tokens or output logits may suffer from weak initial signals, and internals-based detectors using dense representations may retain highly entangled and redundant safety-irrelevant information. It therefore remains unclear whether the earliest post-generation hidden states already contain reliable signals about final-response harmfulness. To address this gap, we propose ForeSight, a first-token output-risk forecasting framework that distills weak and redundant early safety signals into compact, layer-aware risk representations. Experiments on five safety benchmarks and two target models demonstrate that ForeSight achieves superior and efficient early-risk forecasting while relying sol
    
[^162]: PolicyMem：用于大语言模型治理的几何策略记忆

    PolicyMem: Geometric Policy Memory for LLM Governance

    [https://arxiv.org/abs/2609.13734](https://arxiv.org/abs/2609.13734)

    PolicyMem提出了一种几何策略记忆框架，将自然语言策略外化为共享表示空间中由低秩子空间表示的可重用几何记忆对象，解决了现有LLM治理方法无法将策略作为可重用操作状态的问题，从而支持在检测、干预和验证之间一致地重用策略证据。

    

    随着大语言模型（LLM）越来越多地部署在现实世界的高风险应用中，有效的治理变得至关重要。现有的安全防护措施主要遵循两种范式：基于学习的防护器提供强大的语义判别能力，但将策略行为与训练的模型和分类体系耦合在一起；而可编程框架提供灵活的控制，但需要大量的手动提示和工作流工程。这两种方法都没有将策略外化为可重用的操作状态，导致难以在检测、干预和验证之间一致地重用策略证据。在本文中，我们提出了PolicyMem，这是一种几何策略记忆，它将自然语言策略外化为可重用的几何记忆对象，这些对象由共享表示空间中的低秩子空间表示。记忆写入器将自然语言策略编译为策略记忆槽，查询-响应对通过……（摘要原文不完整）读取策略记忆。

    arXiv:2609.13734v1 Announce Type: cross  Abstract: As large language models (LLMs) are increasingly deployed in real-world high-stakes applications, effective governance has become essential. Existing safeguards largely follow two paradigms: learning-based guards provide strong semantic discrimination but couple policy behavior to trained models and taxonomies, while programmable frameworks offer flexible control but require substantial manual prompt and workflow engineering. Neither externalizes policies as reusable operational states, making it difficult to consistently reuse policy evidence across detection, intervention, and verification. In this paper, we introduce PolicyMem, a geometric policy memory that externalizes natural-language policies as reusable geometric memory objects represented by low-rank subspaces in a shared representation space. A memory writer compiles natural-language policies into policy memory slots, and query-response pairs read the policy memory through pr
    
[^163]: 通过自动前置群超级标注扩展印地语量子自然语言处理

    Scaling Hindi Quantum Natural Language Processing through Automatic Pregroup Supertagging

    [https://arxiv.org/abs/2609.13721](https://arxiv.org/abs/2609.13721)

    本文将印地语前置群超级标注自动化形式化为词元级分类任务，发现在低资源场景下简单的词汇与上下文模型（上下文回退达64.56%准确率）优于大语言模型提示，且词汇修复可将LLM预测提升至64.08%，为印地语量子自然语言处理的规模化奠定了基础。

    

    量子自然语言处理（QNLP）使用前置群文法将语法结构转换为图表示和量子电路。近期的印地语QNLP研究表明，印地语专用的前置群文法可以支持对语法敏感的组合模型，但语法类型分配在很大程度上仍依赖人工完成，这限制了可扩展性。本文将印地语前置群超级标注的自动化形式化为一个词元级分类任务。基于包含380个印地语句子的人工标注语料库，我们评估了词汇方法、上下文方法、基于提示的方法、词汇修复方法以及后缀/形态感知方法。结果显示，在这种低资源设置下，简单的词汇和上下文模型表现强劲：上下文回退方法达到了64.56%的最佳完整准确率，而未经约束的Qwen2.5提示仅达到11.65%。词汇修复将LLM辅助预测提升至64.08%，证明了用文法约束生成式输出的价值。

    arXiv:2609.13721v1 Announce Type: new  Abstract: Quantum Natural Language Processing (QNLP) uses pregroup grammars to translate grammatical structure into diagrammatic representations and quantum circuits. Recent Hindi QNLP work has shown that Hindi-specific pregroup grammars can support grammar-sensitive compositional models, but grammatical type assignment is still largely manual, limiting scalability. This paper formulates automatic Hindi pregroup supertagging as a token-level classification task. Using a manually annotated corpus of 380 Hindi sentences, we evaluate lexical, contextual, prompting-based, lexical-repair, and suffix/morphology-aware methods. Results show that simple lexical and contextual models are strong in this low-resource setting: contextual backoff achieves the best completed accuracy of 64.56\%, while raw Qwen2.5 prompting reaches only 11.65\%. Lexical repair raises LLM-assisted prediction to 64.08\%, demonstrating the value of constraining generative outputs wi
    
[^164]: 编辑定位何时会放大相对选择偏差：梯度几何、目标失配与重要性加权

    When Edit Localization Amplifies Relative Selection Bias: Gradient Geometry, Target Mismatch, and Importance Weighting

    [https://arxiv.org/abs/2609.13709](https://arxiv.org/abs/2609.13709)

    论文通过将局部化梯度分解为被编辑与未编辑分量，证明编辑定位可能增大、减小或非单调地改变相对选择偏差，并针对全梯度目标推导出有限样本均方误差、解析最优保留系数，以及以Oracle重要性加权来纠正目标失配的偏差。

    

    人工修正标识出可编辑的文本片段，但接受修正的样本可能来自具有选择性的反馈渠道。我们在固定的模型检查点上分析这种交互作用，方法是将局部化梯度分解为被编辑部分与保留未动部分两个分量。平方相对选择偏差是一个二次型之比，其导数的符号由一个显式的二次多项式确定。局部化可以增大、减小或非单调地改变这一诊断量；其变化方向取决于各分量的偏差与几何结构。对于每个固定的局部化目标，Oracle重要性加权可以恢复总体均值，但这些目标彼此并不相同。针对一个共同的全梯度目标，我们推导了有限样本均方误差、解析形式的最优保留系数以及固定截断的扩展方法。精确的有限总体计算和每个样本规模下10,000次蒙特卡洛重复实验验证了这些恒等式。

    arXiv:2609.13709v1 Announce Type: new  Abstract: Human corrections identify editable spans, but the examples receiving corrections may come from a selective feedback channel. We analyze this interaction at a fixed model checkpoint by decomposing a localized gradient into edited and retained untouched components. Squared relative selection bias is a ratio of quadratics whose derivative has the sign of an explicit quadratic polynomial. Localization can increase, decrease, or nonmonotonically change this diagnostic; its direction depends on component biases and geometry. Oracle importance weighting recovers the population mean for each fixed localization objective, but these objectives have different targets. Against one common full-gradient target, we derive the finite-sample mean-squared error, an analytic optimal retention coefficient, and a fixed-clipping extension. Exact finite-population calculations and 10,000 Monte Carlo repetitions per sample size verify the identities and counte
    
[^165]: 前缀共享是一个排序问题

    Prefix Sharing Is a Sorting Problem

    [https://arxiv.org/abs/2609.13692](https://arxiv.org/abs/2609.13692)

    本文证明LLM服务中提示片段的最优排序问题等价于在请求集合上选择一个二叉层次结构，全局固定排序仅在最多两个片段时最优，并据此给出 O(3^m) 精确算法及与最小顶点覆盖等经典问题的联系。

    

    LLM服务通过精确前缀匹配来复用KV缓存，因此当一个提示词由一组可复用的片段（如检索到的段落、工具定义、少样本示例）组装而成时，这些片段所选的排列顺序决定了能够共享多少计算量。目前每个已部署的系统都通过单一的全局约定来固定这一顺序。我们证明这种做法仅在请求最多包含两个片段时是最优的，而在一般情况下是渐近错误的。我们的主要结果是一个结构定理：最小前缀树代价等于在请求的所有二叉层次结构H上求 min_H sum_x w(x) t_x(H)，其中 t_x(H) 是需要片段x的请求集合的规范分解大小。因此，选择片段顺序等价于在请求上选择一个层次结构。这一恒等式给出了一个 O(3^m) 的精确算法，将双片段情形识别为最小顶点覆盖问题，并证明在留一族上的最优解等于最小外部路径长度。

    arXiv:2609.13692v1 Announce Type: cross  Abstract: LLM serving reuses KV cache by exact prefix match, so when a prompt is assembled from a set of reusable pieces -- retrieved passages, tool definitions, few-shot exemplars -- the order chosen for those pieces determines how much computation can be shared. Every deployed system fixes that order by a single global convention. We prove this is optimal only when requests contain at most two pieces, and asymptotically wrong in general. Our main result is a structure theorem: the minimum prefix-trie cost equals min_H sum_x w(x) t_x(H) over binary hierarchies H on the requests, where t_x(H) is the canonical decomposition size of the set of requests needing chunk x. Choosing chunk orders is therefore equivalent to choosing one hierarchy over requests. The identity yields an O(3^m) exact algorithm, identifies the two-chunk case as minimum vertex cover, and shows that on the leave-one-out family the optimum is the minimum external path length of 
    
[^166]: 并非所有否定线索都是平等的：词缀否定带来更好的否定理解能力

    Not all Negation Cues are Equal: Affixal Negations Yield Better Negation Understanding

    [https://arxiv.org/abs/2609.13685](https://arxiv.org/abs/2609.13685)

    本研究构建了包含180多万样本、覆盖200多种否定线索的大规模数据集NegCue，通过预训练实验发现词缀否定对提升语言模型和大语言模型的否定理解能力贡献最大，而常被研究的单词否定带来的收益却相对有限。

    

    否定（negation）仍然是语言模型（LM）和大语言模型（LLM）面临的长期挑战。以往的工作主要聚焦于一小部分高频的单词否定线索，例如 not 和 never，对更广泛的否定类型以及现代大语言模型的探索十分有限。为了弥补这一空白，我们构建了 NegCue，一个包含超过180万个样本的大规模数据集，涵盖单词否定、多词否定和词缀否定，包含200多个不同的否定线索。我们进一步在 NegCue 上对仅编码器语言模型和大语言模型进行预训练，以研究不同否定类型如何影响否定理解。在五个下游基准上的实验表明，在相同训练规模下，不同否定类型对性能提升的贡献并不均衡。特别是，词缀否定带来了最大的性能改进，而常被研究的单词否定所带来的收益仍然有限。此外，我们的结果表明，进一步的预训练能够提升……

    arXiv:2609.13685v1 Announce Type: cross  Abstract: Negation remains a longstanding challenge for both language models (LMs) and large language models (LLMs). Prior work mainly focuses on a small set of high-frequency single-word negation cues, such as not and never, with limited exploration of broader negation types and modern LLMs. To address this gap, we construct NegCue, a large-scale dataset containing over 1.8M samples spanning single-word, multi-word, and affixal negation with more than 200 unique cues. We further pre-train both encoder-only LMs and LLMs on NegCue to investigate how different negation types affect negation understanding. Experiments on five downstream benchmarks show that negation types contribute unevenly to performance gains under the same training scale. In particular, affixal negation yields the largest improvements, while the gains from the commonly studied single-word negation remain modest. Moreover, our results demonstrate that further pre-training improv
    
[^167]: LayerRoute：基于LoRA质量保持的自适应层跳过方法，实现高效大语言模型推理

    LayerRoute: Adaptive Layer-Skipping with LoRA-Preserved Quality for Efficient LLM Inference

    [https://arxiv.org/abs/2609.13682](https://arxiv.org/abs/2609.13682)

    LayerRoute通过逐层硬门控路由与联合LoRA微调实现自适应层跳过，在10次独立训练中稳定收敛到相同的跳层模式（跳过第8至16层），在保持并提升模型质量的同时实现了平均1.04倍的真实推理加速。

    

    我们提出LayerRoute，这是一种参数高效的自适应Transformer层跳过方法，它将逐层硬门控路由（通过直通估计器训练）与联合LoRA微调相结合。LayerRoute为Qwen2.5-0.5B-Instruct中的24个Transformer块分别添加了轻量级的逐层路由器（约21.5K参数）和LoRA适配器（秩为8，约1.08M参数），并在门控正则化的语言建模目标下对二者进行联合训练。在10次独立随机种子的训练运行中，LayerRoute每次都收敛到完全相同的跳层模式结构——在全部10个种子中，第8至16层共9个一致的中间层均成为可跳过层——并且每次运行都实现了真实且经过验证的挂钟时间加速（1.02倍至1.06倍，平均1.04倍）。在所有测试配置中，模型质量均得到保持或提升：联合LoRA适配在全部10个种子中都使困惑度优于未修改的主干模型（平均差值为-1.16和-1……

    arXiv:2609.13682v1 Announce Type: cross  Abstract: We introduce LayerRoute, a parameter-efficient method for adaptive transformer layer-skipping that combines per-layer hard-gated routing (trained via a straight-through estimator) with joint LoRA fine-tuning. LayerRoute augments each of the 24 transformer blocks in Qwen2.5-0.5B-Instruct with a lightweight per-layer router (~21.5K parameters) and LoRA adapters (rank 8, ~1.08M parameters), training both jointly under a gate-regularized language-modeling objective. Across 10 independently-seeded training runs, LayerRoute converges to an identical skip-pattern structure in every run - a consistent set of 9 middle layers (8-16) becomes skip-eligible in all 10 seeds - and delivers genuine, verified wallclock speedup in every run (1.02x-1.06x, mean 1.04x). Quality is preserved or improved in every configuration tested: joint LoRA adaptation yields a perplexity improvement over the unmodified backbone in all 10 seeds (mean delta = -1.16 and -1
    
[^168]: FedV-KGQA的实践应用：设计经验与交互式原型

    FedV-KGQA in Practice: Design Lessons and an Interactive Prototype

    [https://arxiv.org/abs/2609.13661](https://arxiv.org/abs/2609.13661)

    FedV-KGQA通过联邦方式在垂直分区的知识图谱上实现多跳问答，原始三元组和关系嵌入无需离开本地数据孤岛，实验表明联邦融合可恢复大部分集中式准确率，且问题锚定与图谱丰富化比嵌入模型的选择更为关键。

    

    arXiv:2609.13661v1 公告类型：新文章 摘要：知识图谱问答通常假设单一系统能够访问整个图谱。但在实践中，事实往往由多个组织分别持有，这些组织共享实体标识符却拥有互不重叠的关系类型，因此任何一方都无法看到完整的推理链。本海报展示了FedV-KGQA在这类垂直分区图谱上进行多跳问答的实证研究结果。每个数据孤岛丰富其本地图谱，并在自己的三元组上训练知识图谱嵌入。随后，服务器将各孤岛特定的实体视图进行拼接，将投影后的问题锚定在主题实体上，并通过相似度对候选答案进行排序。原始三元组和关系嵌入永远不会离开数据孤岛。通过对比FedV-KGQA的各项实验，得出三个结果：第一，联邦融合能够恢复大部分集中式方法的准确率，而单个数据孤岛只能恢复很小一部分；第二，问题锚定和图谱丰富化比嵌入模型的选择更为重要；第三（摘要内容在此处截断）。

    arXiv:2609.13661v1 Announce Type: new  Abstract: Knowledge graph question answering usually assumes that one system can reach the whole graph. In practice, facts are often held by organizations that share entity identifiers but own disjoint relation types, so no single party sees a complete reasoning chain. This poster presents the empirical findings of FedV-KGQA on multi-hop question answering over such vertically partitioned graphs. Each silo enriches its local graph and trains a knowledge graph embedding on its own triples. A server then concatenates the silo-specific entity views, anchors the projected question at the topic entity, and ranks candidates by similarity. Raw triples and relation embeddings never leave a silo. Comparing the FedV-KGQA experiments with one another yields three results. First, federated fusion recovers most of the centralized accuracy, while a single silo recovers little. Second, anchoring and enrichment matter more than the choice of embedding model. Thir
    
[^169]: FaithfulBench：AI 咨询建议是维护还是削弱用户所宣称的信仰？

    FaithfulBench: Does AI Counsel Uphold or Undermine the User's Professed Faith?

    [https://arxiv.org/abs/2609.13634](https://arxiv.org/abs/2609.13634)

    该论文提出首个评估 AI 跨宗教传统咨询建议的基准 FaithfulBench，发现当用户信仰未被说明时 AI 会采用世俗默认建议而辜负信徒，仅指明信仰虽能获得虔诚的首次回答却缺乏坚定性，而根植于经典传统的辅导指南能改善 AI 的表现。

    

    AI 助手能否帮助信徒以与其信仰一致的方式思考道德困境？我们提出了 FaithfulBench，这是首个跨宗教传统评估 AI 咨询建议的基准，衡量其遵循用户所宣称信仰的程度。场景取自各传统最受尊崇的文本，其中虔诚的答案是已知的，并由评审员将其作为评判标准加以应用。我们在三种条件下测试了五个前沿模型：AI 不知道用户的宗教传统；AI 收到一行提示词，表明用户是虔诚的信徒；或者 AI 收到一份根植于该传统经典的同伴辅导指南。两位评审员对初始回复进行评分，并评估当模型受到压力、被推向用户想要的答案时，是妥协还是坚持。当宗教传统未被说明时，模型会以世俗治疗式的默认方式进行咨询，每个模型都会让部分信徒失望。指明信仰能赢得虔诚的首次回答，但无法保证坚定性；而辅导指南则有所改善。

    arXiv:2609.13634v1 Announce Type: cross  Abstract: Do AI assistants help believers reason about moral dilemmas consistently with their faith? We present FaithfulBench, the first benchmark to score AI counsel across traditions by how well it adheres to the user's professed faith. Scenarios are drawn from each tradition's most respected texts, with the faithful answer known and applied by the judges as the standard. We test five frontier models under three conditions: the AI does not know the user's tradition; it receives a one-line prompt identifying the user as a practicing adherent; or it receives a companion-counselor guide rooted in the tradition's sources. Two judges score the initial response and whether the model caves or holds when pressured toward the answer the user wants. When the tradition is unstated, models counsel from a secular therapeutic default and every model fails some believers. Naming the faith wins a faithful first answer but not steadfastness; the guide improves
    
[^170]: 一个用于大语言模型针对性危害缓解的高效模块化框架

    An Efficient and Modular Framework for Targeted Harm Mitigation in LLMS

    [https://arxiv.org/abs/2609.13624](https://arxiv.org/abs/2609.13624)

    提出了一种结合Activated LoRA适配器与上下文感知路由机制的模块化纠正框架，可在生成过程中以低延迟、有针对性的方式缓解大语言模型的有害输出，同时提升模型对齐性能。

    

    摘要：大语言模型（LLMs）是强大的零样本学习器，但仍然容易与人类偏好产生不一致，经常输出带有偏见、有毒或其他有害的内容。现有的对齐方法虽然有效，但成本高昂且与模型紧密耦合，限制了灵活性和可扩展性。我们提出了一个模块化纠正框架，通过Activated LoRA（aLoRA）适配器和上下文感知路由机制来增强预训练的大语言模型，以消除模型失调响应带来的危害。我们的方法使专家适配器能够在序列中间激活而不使KV缓存失效，从而在生成过程中实现低延迟的针对性纠正。每个专家都被训练用于检测和缓解特定类型的危害，例如偏见或毒性。一个经过学习的路由器根据模型的中间输出动态选择合适的专家。我们证明该系统在标准安全基准测试中改善了对齐效果，同时保留了……

    arXiv:2609.13624v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are powerful zero-shot learners but remain prone to misalignment with human preferences, often producing biased, toxic, or otherwise harmful outputs. Existing alignment methods, while effective, are costly and tightly coupled to the model, limiting flexibility and scalability. We propose a modular correction framework that augments pretrained LLMs with Activated LoRA (aLoRA) adapters and a context-aware routing mechanism to eliminate harms from misaligned model responses. Our approach enables expert adapters to activate mid-sequence without invalidating the KV cache, allowing low-latency, targeted correction during generation. Each expert is trained to detect and mitigate specific harms, such as bias or toxicity. A learned router dynamically selects appropriate experts based on the models intermediate outputs. We demonstrate that our system improves alignment on standard safety benchmarks while preserving t
    
[^171]: 墨尔本大学WMT 2026克里奥尔语机器翻译提交系统：一种面向低资源太平洋克里奥尔语机器翻译的领域平衡方法

    The University of Melbourne WMT 2026 CreoleMT Submission: A Domain-Balanced Approach to Low-Resource Pacific Creole Machine Translation

    [https://arxiv.org/abs/2609.13615](https://arxiv.org/abs/2609.13615)

    该论文提出一种领域平衡的微调方法，结合LLM辅助数据处理、回译和Gemini蒸馏等技术，使面向太平洋克里奥尔语（Tok Pisin、Bislama、Solomon Pijin）的低资源机器翻译模型在所有翻译方向上超越开源基线3个以上的chrF++分数。

    

    在参加WMT26克里奥尔语言翻译共享任务的提交中，我们专注于太平洋克里奥尔语（巴布亚皮钦语Tok Pisin、比斯拉马语Bislama和所罗门皮钦语Solomon Pijin）的机器翻译模型，并特别关注广泛的领域性能。在大规模领域不平衡数据上进行预训练后，我们在多样化的领域平衡数据混合集上继续进行微调。我们采用了多种数据收集和准备技术，包括LLM辅助的重新拼写与对齐、回译，以及从Gemini蒸馏以覆盖训练数据中原本不存在的领域。在Bouquet和由口语转录文本组成的新型测试集上进行评估，以人工原始翻译为参考，我们的模型在所有翻译方向上均比开源模型基线高出3个以上的chrF++分数。展望未来，我们计划为所罗门皮钦语和比斯拉马语开发人工翻译的测试集，并将我们最好的模型蒸馏成保留广泛领域覆盖范围的小得多的模型。

    arXiv:2609.13615v1 Announce Type: new  Abstract: For our submission to the WMT26 Creole Language Translation Shared Task, we focus on machine translation (MT) models for Pacific creoles: Tok Pisin, Bislama, and Solomon Pijin, with particular attention to broad domain performance. After pre-training on a large collection of domain-imbalanced data, we continue fine-tuning on a diverse mix of domain-balanced data. We rely on a number of data collection and preparation techniques, including LLM-assisted respelling and alignment, back-translation, and distillation from Gemini for domains originally not present in training data. Evaluated on Bouquet and a novel test set made of spoken language transcripts, our models beat open model baselines by 3+ chrF++ points in all directions with human-original references. Looking ahead, we plan to develop human-translated test sets for Solomon Pijin and Bislama, and to distil our best models into much smaller ones that retain broad domain coverage.
    
[^172]: 盲中求评：为机器翻译评估构建伪参考译文

    In the Blind: Building Pseudo-References for MT Evaluation

    [https://arxiv.org/abs/2609.13611](https://arxiv.org/abs/2609.13611)

    该论文提出在WMT26缺乏人工参考译文的语言对上，通过多模型多提示翻译、无参考质量估计打分与文档级选择构建伪参考译文，并引入置信度缩放的语言识别惩罚以消除“错误语言但流畅”的失效输出。

    

    WMT26通用机器翻译任务在10个没有人工参考译文的语言对上评估各个系统（既没有从零开始的人工翻译，也没有由人工对机器翻译输出进行译后编辑的版本）。我们描述了如何为这些语言对以及其他六个语言对（其中有某种形式的人工参考译文可用）构建伪参考译文：七个模型在最多五种提示条件下翻译3,277份官方文档，共产生26种系统-提示组合；然后由三个无参考质量估计（QE）模型为每个候选译文打分；再由一个文档级选择器选出一份译文，并在必要时由GPT-5.5进行译后编辑。在缺乏参考译文的情况下工作暴露了QE引导选择的一个失效模式：这些指标会把语言错误（即错误语言）但流畅的输出排在正确译文之上。在分数融合中加入按置信度缩放的语言识别惩罚项后，错误语言的译文数量降为零，且由此得到的选择器在其他方面仍然得分更

    arXiv:2609.13611v1 Announce Type: new  Abstract: The WMT26 General MT task evaluates systems on 10 language pairs that have no human references (neither translated from scratch nor post-edited from MT output by humans). We describe how we built the pseudo-references for these pairs and six other language pairs (in which some forms of human references are available): seven models translate the 3,277 official documents under up to five prompt conditions, giving a total of 26 system-prompt combinations; then three reference-free quality estimation (QE) models score every candidate; and a per-document selector picks one translation, which GPT-5.5 post-edits where needed. Working without references exposed a failure mode of QE-guided selection: the metrics rank fluent output in the wrong language above correct translations. Adding a confidence-scaled language identification penalty to the score fusion drives the wrong-language count to zero, and the resulting selector still scores better on
    
[^173]: 同一患者，不同医嘱：临床大语言模型智能体在重复运行下的动作级可靠性

    Same Patient, Different Order: Action-Level Reliability of Clinical LLM Agents Under Repeated Runs

    [https://arxiv.org/abs/2609.13582](https://arxiv.org/abs/2609.13582)

    本文提出“同输入重跑”评估方法及六项可靠性指标，首次系统揭示临床大语言模型智能体在完全相同输入下重复运行时会产生实质不同的医疗动作（如检查医嘱、用药和转诊），而这种动作级分歧会被仅评单次运行的基准测试（如MedAgentBench）所掩盖。

    

    一个临床智能体基准测试可以在相同输入上报告相同的判定结果，而智能体在每次运行中却开出实质上不同的医嘱。这类智能体会安排检查、申请药物和发起转诊，然而基准测试通常对每个任务只评分一次运行，很少询问相同的输入是否会产生相同的动作；MedAgentBench（本文所使用的基准）只对单次尝试进行评分，并且明确说明了这一点。为了测量这一差距，我们引入了“同输入重跑”方法，该方法在所有输入保持固定的情况下重放任务，并比较产生的医嘱而非分数，同时配合六项可靠性指标，并将其应用于1000次MedAgentBench运行，涵盖来自其五个可写入类别中的50个任务、两个低于100亿参数且量化为4比特的开源权重模型，以及两个温度设置。本研究确立了动作层面的分歧确实存在且可能在评分未记录的情况下通过，而非证明任何具体比率具有普遍性。在温度0.7下的8B模型中，全部43个医嘱任……（摘要原文在此处截断）

    arXiv:2609.13582v1 Announce Type: cross  Abstract: A clinical agent benchmark can report the same verdict on identical inputs while the agent files a materially different order on each run. Such agents order tests, request medications and place referrals, yet benchmarks typically score one run per task and rarely ask whether identical inputs produce identical actions; MedAgentBench, the benchmark we use, scores a single attempt and says so. To measure this gap we introduce "same-input rerun", which replays a task with every input held fixed and compares the orders rather than the score, with six reliability metrics, and apply it to 1000 MedAgentBench runs across 50 tasks from its five write-capable families, two open-weight models below ten billion parameters quantised to four bits, and two temperatures. The study establishes that action-level divergence exists and can pass unrecorded by the score, not that any rate generalises. Under the 8B model at temperature 0.7, all 43 ordering gr
    
[^174]: 基于抽象意义表示的完整医院出院总结生成研究

    Toward Complete Hospital Discharge Summarization with Abstract Meaning Representation

    [https://arxiv.org/abs/2609.13581](https://arxiv.org/abs/2609.13581)

    提出了一种以证据驱动、以来源追溯为一等约束的出院总结生成框架，利用语义图和深度学习通过跨文档语义对齐为每句总结提供明确的证据链接，从而缓解大语言模型的幻觉问题。

    

    出院总结是概括住院患者就诊情况的长篇医疗文档。自动生成这些总结可以减轻文档记录负担，让临床医生有更多时间照顾患者。虽然大型语言模型（LLMs）可以用于这项任务，但它们的致命弱点是幻觉问题，这可能对临床文档记录造成严重后果。我们提出了一个以证据驱动的出院总结对齐框架，该框架在临床就诊级别上进行总结，将来源追溯作为一等约束条件，并使用语义图和深度学习模型。每个总结句子都通过跨文档语义对齐进行选择和组织，并附有指向其源文本片段的明确证据链接。我们在两个语料库上展示了我们的结果：一个公开可用的语料库（MIMIC-III）和伊利诺伊大学医院（UIC Health）医生撰写的临床笔记。此外，我们公开了源代码和训练好的模型。

    arXiv:2609.13581v1 Announce Type: new  Abstract: Discharge summaries are lengthy medical documents that summarize a hospital in-patient visit. Automatically generating them can reduce documentation burden and return clinician time to patient care. Whereas Large Language Model (LLMs) could be used for this task, their Achilles heel is hallucinations, which can have drastic consequences for clinical documentation. We present an evidence-driven alignment framework for discharge summarization at the clinical encounter level, that treats provenance as a first-class constraint, using semantic graphs and deep learning models. Each summary sentence is selected and organized via cross-document semantic alignment and is accompanied by explicit evidence links to its source spans. We show our results on two corpora: a publicly available corpus (MIMIC-III) and clinical notes written by physicians at the University of Illinois Hospital (UIC Health). Additionally, we make source code and trained mode
    
[^175]: 用户对AI的不当对待如何在对话系统中发生及其重要性？

    How User-AI Mistreatment Occurs and Matters in Conversational Systems?

    [https://arxiv.org/abs/2609.13579](https://arxiv.org/abs/2609.13579)

    本文通过审计77.7万条对话发现，用户对AI的不当对待（侮辱、威胁、越狱胁迫等）约占用户轮次的0.90%，且与平台审核信号所捕捉的有害内容请求是不同的现象，这对准确理解模型行为和部署风险具有重要意义。

    

    安全研究通常关注模型生成的危害，但用户也可能对模型施加敌意、胁迫和对抗性压力。理解这种情况如何以及何时发生，对于准确解读模型行为、对齐漂移和现实世界部署风险至关重要。本文使用两个独立的检测器审计了77.7万条英文LMSYS-Chat-1M对话：一个针对模型敌意的八类词典，以及数据集自带的审核信号；结果表明二者捕捉的是不同的、弱重叠的现象。该词典识别针对助手的侮辱、威胁和越狱胁迫，而审核标记主要由有害内容请求构成，而非针对模型的敌意。两者合计标记了约5%的用户轮次；根据测量的精确率对较窄的词典-骚扰并集进行调整后，针对助手的不当对待率为0.90%。这些绝对比率描述了竞技场式场景中的情况……

    arXiv:2609.13579v1 Announce Type: new  Abstract: Safety research often focuses on model-generated harms, but users may also direct hostility, coercion, and adversarial pressure at models. Understanding how and when that occurs is essential for accurately interpreting model behaviour, alignment drift, and real-world deployment risks. In this paper, we audit 777K English LMSYS-Chat-1M conversations with two independent detectors: an eight-category lexicon for hostility directed at the model, and the dataset's moderation signal; and show that they capture different, weakly overlapping phenomena. The lexicon identifies insults, threats, and jailbreak coercion aimed at the assistant, while moderation flags are dominated by toxic-content solicitation rather than hostility at the model. Together, they mark about 5% of user turns; adjusting the narrower lexicon-harassment union for measured precision puts mistreatment aimed at the assistant at 0.90%. These absolute rates describe arena-style e
    
[^176]: 大型语言模型中的领域专业术语：通用模型与专业模型的比较分析

    Domain-Specific Jargon in Large Language Models: A Comparative Analysis between General-Purpose and Specialist Models

    [https://arxiv.org/abs/2609.13556](https://arxiv.org/abs/2609.13556)

    该研究提出了两个医学专业术语评估基准，发现通用LLM在医学术语任务上反而优于医学领域微调模型，并通过机制可解释性分析揭示微调并未重组参数化知识，而只是过度依赖一小部分偏向专业术语预测的模型组件。

    

    大型语言模型（LLMs）在通用任务上展现出了卓越的能力，但在高度专业化的技术领域中，其性能往往会下降。此外，关于领域特定术语的参数化知识如何在这些模型内部编码，目前知之甚少。为填补这一空白，我们贡献了两个新颖的医学专业术语评估基准，并将通用 Llama-3.1 模型与在医学领域数据上微调的变体进行对比评估。令人惊讶的是，通用模型在两项任务上的表现均优于医学微调模型。借助机制可解释性工具，我们发现医学微调模型存在系统性的校准失调模式。微调模型并未重新组织其参数化知识，而是更加侧重于一小部分与偏向专业术语预测相关的模型组件。我们发现，针对基准任务应用组件重加权策略能够成功地……

    arXiv:2609.13556v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have shown remarkable proficiency on general-purpose tasks, yet their performance often degrades in highly-specialized technical domains. Moreover, little is known about how parametric knowledge of domain-specific terms is encoded within these models. We address this gap by contributing two novel medical jargon evaluation benchmarks and evaluate a general-purpose Llama-3.1 model against a variant fine-tuned on medical-domain data. Surprisingly, the general-purpose model outperforms the medically fine-tuned model on both tasks. Using mechanistic interpretability tools, we find systematic patterns of miscalibration for the medically fine-tuned model. Instead of reorganizing parametric knowledge, the fine-tuned model places greater emphasis on a small subset of model components associated with jargon-favoring predictions. We find that applying component reweighting strategies against the benchmark tasks succes
    
[^177]: 危害传播动力学：大语言模型中对抗意图的逐层轨迹

    Harmfulness Propagation Dynamics: Layer-wise Trajectories of Adversarial Intent in Large Language Models

    [https://arxiv.org/abs/2609.13534](https://arxiv.org/abs/2609.13534)

    该论文发现有害提示在Transformer各层的危害方向投影呈单调上升趋势而良性提示保持平坦，并基于这一逐层激活动态特征提出了轻量级有害输入识别器HERALD。

    

    我们识别出**危害传播动力学**：对于有害提示，最后一个token的隐藏状态在学习到的危害方向上的投影会随着Transformer深度的增加而单调上升，而良性提示则保持平坦或振荡状态。这种跨层特征表明有害意图是一种“逐步解析”的语义属性：表面形式在早期就已显现，而语用意图则在后期才得以固化，这使得轨迹形状比任何单层快照都更具信息量。此外，逐层学习的基于LDA的危害方向在随机划分下保持稳定（成对余弦相似度>0.97），支持将投影序列作为一种可复现的结构化信号。基于HPD，我们引入了HERALD（通过激活层动态识别有害编码）。这个轻量级输入调节器提取一个七维特征……

    arXiv:2609.13534v1 Announce Type: new  Abstract: We identify \textbf{Harmfulness Propagation Dynamics (HPD)}: for harmful prompts, the projection of the last-token hidden state onto a learned harm direction rises monotonically with transformer depth, whereas benign prompts remain flat or oscillatory. This cross-layer signature reflects harmful intent as a \emph{progressively resolved} semantic property: surface form appears early, while pragmatic intent consolidates later, making the \emph{trajectory shape} more informative than any single-layer snapshot. Moreover, LDA-based harm directions, learned per layer, remain stable across random splits (pairwise cosine similarity $>0.97$), supporting the projection sequence as a reproducible structured signal. Building on HPD, we introduce \textbf{\herald{}} (\textbf{H}armful \textbf{E}ncoding \textbf{R}ecognition via \textbf{A}ctivation \textbf{L}ayer \textbf{D}ynamics). This lightweight input moderator extracts a seven-dimensional feature re
    
[^178]: 基于可扩展神经符号模型的生成式可解释性

    Generative Interpretability via Scalable Neuro-Symbolic Models

    [https://arxiv.org/abs/2609.13529](https://arxiv.org/abs/2609.13529)

    论文提出“生成式可解释性”这一新范式，主张模型的推理过程应原生暴露人类可理解且可进行因果干预的语义检查点，并利用神经符号模型实现这一目标，以应对智能体系统中输出不可逆的安全挑战。

    

    随着大语言模型的应用从聊天机器人扩展到智能体系统，其输出成为对现实产生不可逆转后果的行动，现有的人工智能可解释性研究范式——事后可解释性——在安全可信的模型部署方面存在结构性不足：它只能在事后解释模型行为，却无法在推理计算得出输出之前对其进行审计或干预。因此，我们主张转向“生成式可解释性”，这是一种架构属性，使模型的推理过程原生地暴露出语义上有意义、人类可理解且可进行因果干预的检查点。我们展示了生成式可解释性相较于其他可解释性研究范式的优势，并提出神经符号模型作为其具体实例化方案。

    arXiv:2609.13529v1 Announce Type: cross  Abstract: As the use of Large Language Models moves from chatbots into agentic systems, where outputs become actions with irreversible consequences on reality, the existing paradigm on AI Interpretability research, post-hoc interpretability, is structurally inadequate for safe and trustworthy model deployment: it explains behavior after the fact but cannot audit or intervene in an inference computation before it commits to an output. We therefore argue for a shift toward \emph{generative interpretability}, an architectural property under which a model's inference pass natively exposes semantically meaningful checkpoints that are human-understandable and amenable to causal intervention. We show the merits of generative interpretability as comparison to other interpretability research paradigms, and propose Neuro-Symbolic Models as a concrete instantiation.
    
[^179]: 从词元概率到语义约束：迈向语言模型的声明式概率评估

    From Token Probabilities to Semantic Constraints: Towards Declarative Probabilistic Evaluation of Language Models

    [https://arxiv.org/abs/2609.13520](https://arxiv.org/abs/2609.13520)

    提出ModelLog声明式概率评估框架，将评估目标表示为词元预测上的符号语义约束，揭示模型在否定、互斥性和一致性上的系统性失败，并可将评估分数同时用作损失以连接评估与学习。

    

    尽管大型语言模型发展迅速，但关于如何评估其所获得的知识和推理能力，以及这些评估与预训练中使用的学习信号之间的关系，仍存在许多根本性问题。在本文中，我们提出了ModelLog，一个用于预训练评估的声明式概率框架，它使模型行为的语义结构显式化，并为将评估与学习联系起来提供了新的形式化工具。ModelLog将评估目标指定为词元级预测上的符号约束，并衡量模型的分布在多大程度上满足这些约束。我们通过一套针对否定、互斥性和一致性的新任务套件来探索该框架，发现了仅通过词元似然或答案准确率难以刻画的系统性失败。我们进一步表明，这些评估分数也可以被解释为损失，其梯度……（摘要原文在此处截断）

    arXiv:2609.13520v1 Announce Type: new  Abstract: While Large Language Models have improved rapidly, many fundamental questions remain about how to evaluate the knowledge and reasoning abilities they acquire, and how such evaluations relate to the learning signals used in pre-training. In this paper, we propose ModelLog, a declarative probabilistic framework for pre-training evaluation that makes the semantic structure of model behavior explicit and provides new formal tools for relating evaluation to learning. ModelLog specifies evaluation targets as symbolic constraints over token-level predictions and measures how strongly a model's distribution satisfies those constraints. We explore the framework through a new suite of tasks targeting negation, mutual exclusivity, and consistency, finding systematic failures that are difficult to characterize through token likelihood or answer accuracy alone. We further show that these evaluation scores can also be interpreted as losses, whose grad
    
[^180]: 一个谱，两种资源：自回归预测中的数据-内存扩展律

    One Spectrum, Two Resources: Data-Memory Scaling in Autoregressive Prediction

    [https://arxiv.org/abs/2609.13500](https://arxiv.org/abs/2609.13500)

    论文证明数据与学习记忆这两种资源由同一个预测能量谱统一支配，并由此建立了刻画自回归预测中最优数据-内存权衡的极小极大定律。

    

    需要多少学习到的记忆才能从更多数据中获益？我们证明，在一个正熵的自回归检索源中，这两种资源由同一个预测能量谱所支配。每个坐标的贡献等于其查询概率乘以其未知 logit 的半径平方。记 μ 为由此得到的能量谱，我们证明了如下极小极大定律：R*_value(n,B) ≍_R Φ_μ(n⁻¹) + Φ_μ(τ_B)，其中 Φ_μ(t) = ∫min{x,t} μ(dx)，n 为预测块数量，学习到的状态至多有 2^B 个取值。数据决定分辨率 1/n；记忆决定通过最优比特分配所能达到的水平 τ_B。完整的曲线还能恢复出正谱部分。能量-维度配对是本质性的：两个具有相同块能量与块维度边缘分布的因果源，其数据指数和记忆指数却可以不同。一个带掩码的查询-键注意力头学习路由与取值，实现了该定……

    arXiv:2609.13500v1 Announce Type: cross  Abstract: How much learned memory is needed to benefit from more data? We show that the two resources are governed by one predictive-energy spectrum in a positive-entropy autoregressive retrieval source. Each coordinate contributes its query probability times the squared radius of its unknown logit. Writing $\mu$ for the resulting energy spectrum, we prove the minimax law $\mathfrak R^*_{\rm value}(n,B)\asymp_R \Phi_\mu(n^{-1})+\Phi_\mu(\tau_B), \Phi_\mu(t)=\int\min\{x,t\}\,\mu(\mathrm dx),$ for $n$ prediction blocks and a learned state with at most $2^B$ values. Data set the resolution $1/n$; memory sets the level $\tau_B$ reached by optimal bit allocation. The complete curve also recovers the positive spectrum. Energy-dimension pairing is essential: two causal sources with identical block-energy and block-dimension marginals have different data and memory exponents. A masked query-key attention head learns the route and values, realizing the l
    
[^181]: 混合专家语言模型可以成为强大且高效的检索器

    Mixture-of-Experts Language Models Can Be Strong and Efficient Retrievers

    [https://arxiv.org/abs/2609.13486](https://arxiv.org/abs/2609.13486)

    该研究首次系统探索了混合专家语言模型作为检索器的潜力，证明MoE检索器在激活参数量相当的情况下比稠密检索器在BEIR上高出最多3.0个nDCG@10点，并能以少59%的激活参数匹敌8B稠密检索器，实现强大且高效的检索。

    

    最近的研究表明，针对检索任务微调仅解码器（decoder-only）大语言模型（LLM）可以获得强大的第一阶段检索器，且效果随骨干模型规模的增大而提升。然而，每个查询和文档都必须通过完整模型进行处理，因此编码成本会随模型规模的增长而增加。混合专家（MoE）语言模型在每个token仅激活部分参数，被广泛用于扩展生成模型，但作为检索器仍未得到充分探索。我们通过使用相同的训练流程训练来自多个系列的MoE和稠密LLM，在多个不同数据集上评估它们，并在相同的服务配置下测量查询编码时间，系统地研究了MoE骨干网络在检索中的应用。结果表明，在BEIR基准上，激活参数量相当的MoE检索器比稠密检索器高出最多3.0个nDCG@10点。我们最强的MoE检索器之一能以少59%的激活参数和少18%（的编码时间）匹敌8B稠密检索器。

    arXiv:2609.13486v1 Announce Type: cross  Abstract: Recent work has shown that fine-tuning decoder-only large language models (LLMs) for retrieval yields strong first-stage retrievers, with effectiveness improving as backbones grow in size. However, every query and document must pass through the full model, so encoding cost increases with model size. Mixture-of-Experts (MoE) LLMs activate only a subset of parameters per token and are widely used to scale generative models, yet remain underexplored as retrievers. We systematically study MoE backbones for retrieval by training MoE and dense LLMs from several families using the same procedure, evaluating them across diverse datasets, and measuring query encoding time under the same serving configuration. We show that MoE retrievers outperform dense retrievers with comparable active parameter counts by up to 3.0 nDCG@10 points on BEIR. One of our strongest MoE retrievers matches an 8B dense retriever with 59% fewer active parameters and 18%
    
[^182]: 一种用于生物医学与临床文本抽取式摘要的混合分层一维卷积神经网络-双向长短期记忆网络（1D-CNN-BiLSTM）框架

    A Hybrid Hierarchical 1D-CNN-BiLSTM Framework for Extractive Summarization of Biomedical and Clinical Text

    [https://arxiv.org/abs/2609.13481](https://arxiv.org/abs/2609.13481)

    该论文提出一种混合分层的1D-CNN-BiLSTM框架，将生物医学与临床文本摘要重新构建为抽取式句子选择任务，通过直接从源文本复制句子来规避生成式大模型的幻觉风险。

    

    大型语言模型已使生成式摘要变得非常流畅，但生成的摘要可能出现事实幻觉，这在生物医学和临床领域会带来严重风险。为解决这一问题，我们从流程中移除了生成环节，将摘要任务重新构建为抽取式的句子选择问题。我们的混合分层CNN-LSTM摘要器使用堆叠的多核卷积将句子级嵌入组合成更丰富的句间表示，随后使用双向LSTM对文档中的长距离依赖关系进行建模。一个轻量级的评分头为每个句子分配重要性分数，并基于oracle抽取式标签以二元交叉熵进行端到端训练。在推理阶段，采用带有top-3回退机制的动态均值加标准差阈值直接从源文本中选择句子，并按时间顺序重新排列成最终摘要。由于每个输出句子都直接复制自输入文本，该模型……

    arXiv:2609.13481v1 Announce Type: new  Abstract: Large language models have made abstractive summarization remarkably fluent, but generated summaries can hallucinate facts, posing serious risks in biomedical and clinical domains. We address this by removing generation from the pipeline and framing summarization as extractive sentence selection. Our Hybrid Hierarchical CNN-LSTM Summarizer uses stacked multi-kernel convolutions to compose sentence-level embeddings into richer inter-sentence representations, followed by a bidirectional LSTM to model long-range dependencies across the document. A lightweight scoring head assigns per-sentence importance scores and is trained end-to-end with binary cross-entropy against oracle extractive labels. At inference, a dynamic mean-plus-standard-deviation threshold with a top-3 fallback selects sentences directly from the source and chronologically reorders them into the final summary. Since every output sentence is copied from the input, the model 
    
[^183]: STAGE：诊断具身智能体在落地执行中的语义迁移

    STAGE: Diagnosing Semantic Transfer at Grounded Execution in Embodied Agents

    [https://arxiv.org/abs/2609.13458](https://arxiv.org/abs/2609.13458)

    该论文提出固定观测反事实基准SAT-Bench，揭示具身智能体虽能近乎完美地恢复指令语义（高达100%），但动作敏感度仅约6%，存在严重的语义-动作鸿沟，并提出了轻量级执行时接口VISA来诊断和缓解该问题。

    

    具身语言落地不仅仅需要识别指令的指代对象：恢复出的语义还必须能够控制智能体所执行的动作。我们将这一缺失环节研究为“语义-动作鸿沟”，即指令语义可以被恢复，但在原生连续动作中表达微弱。我们提出了SAT-Bench，一个固定观测的反事实基准，它在保持视觉场景和智能体状态不变的同时，仅改变指令语义。在LIBERO目标名称替换和像素级关系替换任务上，目标恢复率达到100.0%和95.8%，而OpenVLA的动作敏感度仅为6.8%和7.7%。该鸿沟在另外1000个组合性及时间/程序性反事实样本中持续存在，整体动作敏感度仅为6.1%。隐状态、无阈值、跨策略和轨迹回放诊断进一步证实了这种语义-动作迁移失败。我们提出了VISA，一个轻量级的执行时接口（摘要在此处截断）

    arXiv:2609.13458v1 Announce Type: cross  Abstract: Embodied language grounding requires more than identifying the referent of an instruction: recovered semantics must also control the action an agent exposes. We study this missing link as a semantic-action gap, where instruction semantics are recoverable but weakly expressed in native continuous actions. We introduce SAT-Bench, a fixed-observation counterfactual benchmark that holds the visual scene and agent state fixed while changing only instruction semantics. On LIBERO target-name and pixel-grounded relation swaps, target recovery reaches 100.0% and 95.8%, whereas OpenVLA action sensitivity remains only 6.8% and 7.7%. The gap persists across 1,000 additional compositional and temporal/procedural counterfactuals, with overall action sensitivity of 6.1%. Hidden-state, threshold-free, cross-policy, and rollout diagnostics further support this semantic-action transfer failure. We introduce VISA, a lightweight execution-time interface t
    
[^184]: 临床时间推理中的后见之明偏差：未来数据暴露如何影响大语言模型的判断

    Hindsight Bias in Clinical Temporal Reasoning: How Future Data Exposure Affects Large Language Model Judgment

    [https://arxiv.org/abs/2609.13454](https://arxiv.org/abs/2609.13454)

    该论文构建了一个包含171份临床病例报告的配对基准，通过对比前瞻性参考答案与“后见之明陷阱”，首次系统量化了在回顾性数据上评估临床语言模型时，未来信息暴露所导致的后见之明偏差。

    

    临床决策是前瞻性的，但临床语言模型通常在揭示最终诊断、治疗反应和结果的回顾性记录上进行评估。这种评估方式可能会奖励对未来信息的使用，而非在决策点所存在的不确定性下进行推理。我们引入了一个配对基准，用于衡量临床时间推理中与后见之明偏差一致的结果条件化偏移。该基准包含来自PubMed Central开放获取子集的171份病例报告——其中40例脓毒症病例和131例GLP-1/糖尿病病例——以文本叙述以及人工标注和大语言模型生成的文本时间序列（TTS）两种形式表示。对于每个病例，问题均与具有临床意义的截止时间点相关联，并配有前瞻性参考答案和与最终结果一致的“后见之明陷阱”。模型使用在截止点截断的TTS或完整时间线来回答每个问题；其他额外条件包括（原文摘要在此处截断）

    arXiv:2609.13454v1 Announce Type: cross  Abstract: Clinical decisions are prospective, but clinical language models are often evaluated on retrospective records that reveal the final diagnosis, treatment response, and outcome. Such evaluations may reward the use of future information rather than reasoning under the uncertainty present at the decision point. We introduce a paired benchmark for measuring outcome-conditioned shifts consistent with hindsight bias in clinical temporal reasoning. It contains 171 case reports from the PubMed Central Open Access Subset---40 sepsis and 131 GLP-1/diabetes cases---represented as both textual narratives and human-annotated and LLM-generated textual time series (TTS). For each case, questions are tied to a clinically meaningful cutoff and paired with a prospective reference answer and an outcome-consistent \emph{hindsight trap}. Models answer each question using either a TTS truncated at the cutoff or the complete timeline; additional conditions va
    
[^185]: 全双工语音大语言模型中虚假起音的因果分析与缓解

    Causal Analysis and Mitigation of Spurious Onsets in Full-Duplex Speech LLMs

    [https://arxiv.org/abs/2609.13445](https://arxiv.org/abs/2609.13445)

    本研究通过因果分析发现，全双工语音LLM（如Moshi和PersonaPlex）在用户静默时产生虚假语音起音，是因为模型以自身非语音输出为条件导致语音起音概率在单个80毫秒帧内飙升超过九个数量级，而非重复采样所致，并进一步基于因果反事实分析提出了缓解方法。

    

    像Moshi及其衍生模型PersonaPlex这样的语音到语音大语言模型可以通过全双工生成同时进行听和说。然而，在用户长时间保持沉默时，它们可能会不恰当地开始说话：在数字零输入下，Moshi和PersonaPlex分别在40个五分钟延续中的12个和11个中启动了语音。是什么导致了这种虚假语音？我们研究了两个假设：一是重复采样在起音概率持续较低的情况下仍然选择了语音，二是模型以自身的非语音输出为条件导致起音概率突然飙升。我们发现，在每一次观察到的起音中，语音概率都在一个80毫秒的帧内飙升超过九个数量级，这支持了后一个假设。随后，为了在不阻断真实响应的情况下抑制这些起音，我们提出了一个因果反事实问题：模型是在响应用户语音，还是即使移除之前的用户语音，其下一个token的分布仍会保持相似……

    arXiv:2609.13445v1 Announce Type: new  Abstract: Speech-to-speech LLMs like Moshi, and its derivative PersonaPlex, can listen and speak concurrently through full-duplex generation. However, they can begin speaking inappropriately during prolonged user silence: under digital-zero input, Moshi and PersonaPlex initiate speech in 12/40 and 11/40 five-minute continuations, respectively. What causes this spurious speech? We investigate two hypotheses: either repeated sampling selects speech despite persistently low onset probabilities, or conditioning on the model's nonspeech outputs causes an abrupt spike in onset probability. We find that, at every observed onset, speech probability spikes by over nine orders of magnitude in one 80-ms frame, supporting the latter hypothesis. Then, to suppress these onsets without blocking genuine responses, we ask a causal counterfactual question: is the model responding to user speech, or would its next-token distribution remain similar if the preceding u
    
[^186]: CVSS-X：一个面向28种语言的多语言语音到语音翻译语料库

    CVSS-X: A Multilingual Speech-to-Speech Translation Corpus for 28 Languages

    [https://arxiv.org/abs/2609.13413](https://arxiv.org/abs/2609.13413)

    该论文提出CVSS-X，一个将翻译方向扩展为从英语到覆盖12个语系的28种语言、总计超16,000小时的大规模合成语音到语音翻译语料库，与CVSS结合可支持双向和多语言语音到语音翻译研究。

    

    我们介绍CVSS-X，这是一个大规模合成语音到语音翻译语料库，通过反转翻译方向对CVSS进行了扩展。CVSS是从21种语言翻译成英语，而CVSS-X实现了从英语翻译成跨越12个语系的28种目标语言。该语料库每种语言包含约24万个平行语音对，总计超过16,000小时，规模是CVSS的八倍。我们提供两个变体：CVSS-X-C（每种语言具有两个标准声音）和CVSS-X-T（具有跨语言声音克隆），两者均为完全生成。评估结果显示其翻译质量与CVSS相当，且在类型学上多样的语言中表现一致。与CVSS相结合，这为双向和多语言语音到语音翻译的研究提供了支持。代码可在 https://github.com/ErmisAI/XVSS-X 获取，数据集以CC-BY-NC 4.0许可发布于 https://huggingface.co/datasets/lgris/XVSS-X。

    arXiv:2609.13413v1 Announce Type: new  Abstract: We introduce CVSS-X, a large-scale synthetic speech-to-speech translation corpus that extends CVSS by reversing the translation direction. While CVSS translates from 21 languages into English, CVSS-X enables translation from English into 28 target languages spanning 12 language families. The corpus comprises approximately 240,000 parallel speech pairs per language, totaling over 16,000 hours, eight times larger than CVSS. We provide two variants: CVSS-X-C with two canonical voices per language, and CVSS-X-T with cross-lingual voice cloning, both fully generated. Evaluation shows comparable translation quality to CVSS with consistent performance across typologically diverse languages. Combined with CVSS, this enables research on bidirectional and multilingual speech-to-speech translation. The code is available at https://github.com/ErmisAI/XVSS-X and the dataset under CC-BY-NC 4.0 license at https://huggingface.co/datasets/lgris/XVSS-X.
    
[^187]: RFCLLM：评估大语言模型对网络协议状态机的推理能力

    RFCLLM: Evaluating LLMs' Reasoning Ability of Network Protocol State Machines

    [https://arxiv.org/abs/2609.13389](https://arxiv.org/abs/2609.13389)

    本文提出RFCLLM基准，通过针对16个协议设计的4个任务和1482个查询，系统评估大语言模型对网络协议状态机规范的理解能力，揭示其隐式状态机表示与人工真实模型之间的差距。

    

    将文本规范映射为形式化表示对于确保协议设计和实现的正确性至关重要。用于网络安全或测试的LLM生成映射，被假定建立在对规范的完美理解之上，但这在实践中可能并不成立。本文的目标是评估大语言模型在多大程度上能够正确解读规范。我们考察了LLM通过自然语言描述所定义的有限状态转换系统的隐式表示，与人工生成的真实基准模型之间的一致程度。我们针对16个协议设计了4个任务和1482个任务查询，评估了不同的评判偏差，观察了任务之间固有的难度差距，研究了4种上下文类型的影响，以及协议特征所带来的作用。我们的工作为验证LLM在有限状态机（FSM）领域是否真正值得信赖迈出了一步。

    arXiv:2609.13389v1 Announce Type: new  Abstract: Mapping textual specifications into formal representations is essential for ensuring the correctness of protocol designs and implementations. LLM-generated mappings, used for networking security or testing, are assumed to capture a perfect understanding of the specification, which may not hold in practice. The goal of this paper is to assess the extent to which LLMs can interpret the specification correctly. We examine the degree to which an LLM's implicit representation of a finite-state transition system-defined via natural language descriptions-aligns with a manually generated ground-truth model. We designed 4 tasks and 1482 task queries for 16 protocols. We evaluated different judge biases, observed the inherent difficulty gaps between tasks, looked into the effect of 4 context types, and the influence of protocol characteristics. Our work contributes to a step toward verifying whether LLMs can really be trusted in FSM (Finite State 
    
[^188]: ScorePrompts：通过分析实现对符号音乐乐谱的自然语言探索

    ScorePrompts: Natural-Language Exploration of Symbolic Music Scores through Analysis

    [https://arxiv.org/abs/2609.13291](https://arxiv.org/abs/2609.13291)

    ScorePrompts是一个交互式系统，它通过专门的MIR组件分析乐谱的和声、调性、曲式等音乐结构，再由受模式约束的语言模型生成自然语言描述并回答用户问题，同时将分析结果与五线谱中的对应小节和音符可视化关联。

    

    我们提出了ScorePrompts，这是一个交互式系统，用户可以上传乐谱，获得关于其音乐结构的自然语言描述，针对特定乐段提问，并以五线谱形式查看相应的分析结果。专门的音乐信息检索（MIR）组件首先估计和声、调性、终止式、曲式边界、织体以及音符级别的角色，并将其输出组织在音符、节拍、小节和整曲等层级上。一个受模式约束的语言模型将这些分析结果转换为文字描述，而不是直接从原始MusicXML推断音乐结构。对于诸如“第14-18小节发生了什么变化？”这样的问题，确定性路由器会选择相关的小节和分析层级，并返回简洁的答复，同时附上底层的分析结果和注意事项。Verovio负责渲染乐谱，并将返回的信息与所引用的小节及音符级属性相关联。该界面还展示了中间表格和……

    arXiv:2609.13291v1 Announce Type: cross  Abstract: We present ScorePrompts, an interactive system in which users upload a score, receive natural-language descriptions of its musical structure, ask questions about specific passages, and inspect the corresponding analysis results in staff notation. Specialist MIR components first estimate harmony, tonality, cadences, formal boundaries, texture, and note-level roles, organizing their outputs at note, beat, measure, and piece levels. A schema-constrained language model converts these results into descriptions rather than inferring musical structure directly from raw MusicXML. For questions such as "What changes in measures 14-18?", a deterministic router selects the relevant measures and analytical levels and returns a concise response together with the underlying results and caveats. Verovio renders the score and links the returned information to cited measures and note-level attributes. The interface also exposes intermediate tables and 
    
[^189]: 从过程损失到组装红利：基于人类行为的多智能体LLM协作诊断

    From Process Loss to Assembly Bonus: Human-Grounded Diagnosis of Multi-Agent LLM Collaboration

    [https://arxiv.org/abs/2609.13261](https://arxiv.org/abs/2609.13261)

    该研究通过将人类群聊与LLM多智能体审议轨迹进行过程层面的对比诊断，发现两者共享“组装红利”不对称性，但LLM群体在跟随多数、独特信息提出和收敛速度等方面存在过程差异，为多智能体LLM协作提供了基于人类行为的评估框架。

    

    LLM智能体越来越多地被用于协作问题解决和人类群体模拟，这使得仅基于结果的评估变得不够充分：如果LLM群体被用作人类群体的模型，我们需要了解它们是通过类人的审议机制取得成功还是失败。我们在Wason式演绎推理任务上将人类群聊与匹配的LLM审议轨迹进行比较，然后测试相同的过程特征是否能推广到类比、溯因和分析类任务。人类和LLM表现出相同的组装红利不对称性：讨论提升普通成员表现的概率高于提升最佳初始成员表现的概率。初始答案的多样性解释了模型异质性带来的影响，它在纠正性和破坏性两个方向上都增加了答案的移动。主要差异存在于过程层面：与人类相比，LLM群体更频繁地跟随多数意见，提出的独特信息更少，并且更早地达成收敛；正确的少数派信号……（摘要原文在此处截断）

    arXiv:2609.13261v1 Announce Type: cross  Abstract: LLM agents are increasingly used for collaborative problem solving and human-group simulation. This makes outcome-only evaluation insufficient: if LLM groups are used as models of human groups, we need to know whether they succeed or fail through human-like deliberative mechanisms. We compare human group chats with matched LLM deliberation traces on Wason-style deductive reasoning, then test whether the same process signatures generalize to analogical, abductive, and analytical tasks. Humans and LLMs show the same assembly bonus asymmetry: discussion improves the average member more often than the best initial member. Initial-answer diversity accounts for the effect of model heterogeneity, increasing movement in both corrective and destructive directions. The main differences are process-level. Compared with humans, LLM groups follow majorities more often, surface less unique information, and converge earlier; correct minority signals 
    
[^190]: 基于EventGraph和EventField的可解释时序视频推理

    Interpretable Temporal Video Reasoning with EventGraph and EventField

    [https://arxiv.org/abs/2609.13258](https://arxiv.org/abs/2609.13258)

    该论文提出结合离散EventGraph、连续EventField和人类可读EventGlyph的结构化时序视频推理方法，在EPIC-KITCHENS子集上达到0.98的准确率，显著超越字幕基线和纯VLM问答，同时兼顾高性能与可解释性。

    

    我们提出了一种结构化的时序视频推理流水线，它围绕离散的EventGraph、连续的EventField以及人类可读的EventGlyph视图构建。在一个经过校准的EPIC-KITCHENS子集（包含10个视频和50个时序推理问题）上，EventField+Glyph达到了0.98的总体准确率，比字幕基线高出+0.40（配对p = 1.1 × 10^-5），比直接使用纯VLM问答高出+0.20（p = 0.0063）。我们进一步评估了不同标注来源的变体，包括人工、启发式以及启发式+Gemini流水线，发现最佳的结构化方法在所有设置中都保持高于字幕基线。我们还包含了跨视频对的基准测试，以及附录中所有研究视频的glyph输出图库。总体而言，结果表明结构化时序表示可以通过保留符号结构、捕获时序连续性以及提供人类可读的视图，来同时支持性能和可检查性。

    arXiv:2609.13258v1 Announce Type: cross  Abstract: We present a structured temporal video reasoning pipeline built around a discrete EventGraph, a continuous EventField, and a human-readable EventGlyph view. On a calibrated EPIC-KITCHENS subset of 10 videos and 50 temporal reasoning questions, EventField+Glyph achieves 0.98 overall accuracy, which is higher than the caption baseline by +0.40 (paired p = 1.1 \times 10^{-5}) and direct VLM-only QA by +0.20 (p = 0.0063) on this subset. We further evaluate annotation-source variations, including manual, heuristic, and heuristic+Gemini pipelines, and find that the best structured method stays above the caption baseline across settings. We also include cross-video pair benchmarking and an appendix gallery of glyph outputs for all studied videos. Overall, the results indicate that structured temporal representations can support both performance and inspectability by preserving symbolic structure, capturing temporal continuity, and providing h
    
[^191]: 锥形束CT报告生成中部分可观测目标下的临床推理

    Clinical Reasoning Under a Partially Observed Objective in Cone Beam CT Report Generation

    [https://arxiv.org/abs/2609.13238](https://arxiv.org/abs/2609.13238)

    该研究揭示了CBCT报告生成评估中80%权重的事实蕴含目标不可见这一难题，通过纯Python复现BLEU/METEOR并构建AUC达0.987的离线蕴含代理模型使复合目标可被直接优化，按复合目标选出的报告得分0.4122显著高于仅按可见词汇指标选出的0.2909，因为单纯追求n-gram重合度会将蕴含精确率从0.522拖低到0.266。

    

    本文对基于锥形束计算机断层扫描（CBCT）的头面部报告生成任务进行评分，所采用的复合目标将80%的权重赋予大语言模型对事实蕴含的判断，20%赋予词汇重合度指标，而在开发过程中只有词汇部分（即五分之一）是可见的。作者用纯Python复现了评分器的BLEU-4和METEOR程序，达到与参考实现机器精度级别的一致，并构建了一个离线蕴含代理模型，它能够以0.987的曲线下面积（AUC）区分针对不同患者撰写的报告，从而使复合目标的计算足够低廉、可以直接进行优化。在622例公开数据集上，仅根据可见的词汇指标排序选出的报告得分为0.2909，而根据复合目标选出的报告得分为0.4122，原因在于追求n-gram重合度会使蕴含精确率从0.522下降到0.266。一个在公开数据上微调的2900万参数编码器达到了患病率加权……（原文摘要截断）

    arXiv:2609.13238v1 Announce Type: new  Abstract: Maxillofacial report generation from cone beam computed tomography is scored here by a composite objective placing 80% of its weight on a large language model judgement of factual entailment and 20% on lexical overlap, of which only the lexical fifth is visible during development. The grader's BLEU-4 and METEOR routines are reproduced in pure Python and match the reference to machine precision, and an offline entailment surrogate, which tells a report written for one patient from one written for another at an area under the curve of 0.987, makes the composite objective cheap enough to optimise directly. Over the 622-case public release, a report selected against the visible lexical ranking scores 0.2909, whereas one selected against the composite objective scores 0.4122, because pursuing n-gram overlap drives entailment precision from 0.522 down to 0.266. A 29 million parameter encoder fine-tuned on the release reaches a prevalence-weigh
    
[^192]: 用于正畸报告生成的闭合形式咬合几何

    Occlusal Geometry in Closed Form for Orthodontic Report Generation

    [https://arxiv.org/abs/2609.13237](https://arxiv.org/abs/2609.13237)

    本文利用 Bite2Text 口内扫描数据已完成咬合配准这一特性，通过牙弓角度坐标下的咬合嵴轮廓以闭合形式直接计算 31 项咬合几何量（覆𬌗、覆盖、中线偏移等），再用梯度提升映射至 13 个模板字段并由确定性渲染器生成六部分正畸报告，从而将报告生成从多模态字幕推断转变为可测量的几何计算问题。

    

    arXiv:2609.13237v1 公告类型：交叉 摘要：从口内数据生成正畸报告通常被构建为多模态字幕生成任务，然而已发布的 Bite2Text 扫描数据对是在咬合配准状态下提供的，这使得若干核心咬合量可以直接测量而非需要推断。本报告所述系统正是利用了这一特性：针对每个病例，通过牙弓锥度和牙弓闭合来恢复解剖坐标系，而非采用所声称的、在该数据发布中并不普遍成立的 RAS 约定；随后将每侧牙弓在牙弓角度坐标下简化为咬合嵴轮廓，从而以闭合形式得到覆𬌗、覆盖、中线偏移、横向重叠、反𬌗范围、牙尖交错滞后量以及咬合曲线。系统使用梯度提升将这 31 项测量值映射到 13 个模板字段，仅当某字段在患者级交叉验证中优于其自身多数类基线时才进行预测，并由一个确定性渲染器生成语料库规定的六部分叙述文本；一个 ConvNeXt-Tiny …（原文摘要在此处截断）

    arXiv:2609.13237v1 Announce Type: cross  Abstract: Orthodontic report generation from intraoral data is normally cast as multimodal captioning, yet the released Bite2Text scan pairs are supplied already registered in occlusion, which makes several core occlusal quantities directly measurable rather than inferable. The system reported here exploits that property: an anatomical frame is recovered per case from arch taper and arch closure instead of the stated RAS convention, which does not hold across the release, and each arch is reduced to an occlusal ridge profile in arch-angle coordinates yielding overbite, overjet, midline deviation, transverse overlap, crossbite extent, cusp interdigitation lag, and the occlusal curves in closed form. Gradient boosting maps 31 such measurements onto 13 template fields, a field being predicted only where patient-level cross-validation beats its own majority baseline, and a deterministic renderer emits the corpus six-part narrative; a ConvNeXt-Tiny c
    
[^193]: 面向压缩兼容的稀疏长上下文大语言模型推理的自索引注意力

    Self-Indexing Attention for Compression-Compatible Sparse Long-Context LLM Inference

    [https://arxiv.org/abs/2609.13205](https://arxiv.org/abs/2609.13205)

    提出了一种免训练的自索引注意力框架，利用共享的1比特符号索引在预填充和解码阶段统一实现高效token检索，同时兼容外部KV缓存压缩，在5%注意力密度下达到接近密集注意力的准确率，并获得高达6.1倍预填充和10.3倍解码的算子加速。

    

    稀疏长上下文推理需要在预填充（prefill）和解码（decode）两个阶段都进行高效的token检索。现有方法通常对这两个阶段采用不同的检索策略，导致单一检索表示无法在整个推理过程中被复用。我们提出了自索引注意力，这是一个基于共享变换域符号-幅值表示的免训练框架。键值符号提供了一个可复用的token级索引，用于分组的预填充选择和解码检索，同时该表示与外部KV缓存压缩保持兼容，无需单独的索引器元数据。这种1比特索引通过现代加速器广泛支持的按位运算实现高效检索。在5%注意力密度下，自索引注意力在LongBench和RULER基准上仍接近密集注意力的表现，并实现了高达6.1倍的预填充和10.3倍的解码注意力算子加速。与TurboQuant和DeepSeekV4-Flash结合的实验进一步验证了该方法的有效性。

    arXiv:2609.13205v1 Announce Type: cross  Abstract: Sparse long-context inference requires efficient token retrieval in both prefill and decode. Existing methods often use different retrieval strategies for the two stages, preventing one retrieval representation from being reused throughout inference. We propose Self-Indexing Attention, a training-free framework built on a shared transform-domain sign-magnitude representation. The key signs provide a reusable token-level index for grouped prefill selection and decode retrieval, while the same representation remains compatible with external KV-cache compression without separate indexer metadata. This 1-bit index enables efficient retrieval through bitwise operations widely supported by modern accelerators. At 5% attention density, Self-Indexing Attention remains close to dense attention on LongBench and RULER and achieves up to 6.1x prefill and 10.3x decode attention-operator speedups. Experiments with TurboQuant and DeepSeekV4-Flash fur
    
[^194]: GradRepair-ODE：面向神经常微分方程训练的可认证梯度修复

    GradRepair-ODE: Certified Gradient Repair for Neural ODE Training

    [https://arxiv.org/abs/2609.13204](https://arxiv.org/abs/2609.13204)

    提出了GradRepair-ODE框架，通过多候选梯度比较、方向有限差分验证和失效诊断，在优化器步骤处对神经ODE训练中的不可靠梯度进行检查、修复或拒绝，提升数值求解耦合下的训练可靠性。

    

    神经常微分方程在训练循环中使用数值求解器。求解器决定了前向轨迹，同时也会影响传递给优化器的梯度。这种耦合为科学机器学习和连续时间生成建模（包括扩散概率流常微分方程和流匹配模型）带来了可靠性问题。在宽松的步长、刚性动力学、混沌敏感性或事件不连续性的情况下，可微分的ODE流程可能返回一个方向在数值上存疑的有限梯度。我们提出了GradRepair-ODE，一个用于在优化器步骤处检查、修复和拒绝ODE梯度的可靠性框架。该方法计算多个梯度候选值，通过方向有限差分检查和求解器诊断对它们进行比较，诊断可能的数值失效模式，并通过路径切换或更严格的重新积分来修复选定的梯度。

    arXiv:2609.13204v1 Announce Type: cross  Abstract: Neural ordinary differential equations use numerical solvers inside the training loop. The solver determines the forward trajectory and also affects the gradient passed to the optimizer. That coupling creates a reliability problem for scientific machine learning and continuous-time generative modeling, including diffusion probability-flow ordinary differential equations and flow-matching models. Under loose step sizes, stiff dynamics, chaotic sensitivity, or event discontinuities, a differentiable ODE pipeline can return a finite gradient whose direction is numerically suspect. We introduce GradRepair-ODE, a reliability framework for checking, repairing, and rejecting ODE gradients at the optimizer step. The method computes several gradient candidates, compares them with directional finite-difference checks and solver diagnostics, diagnoses likely numerical failure modes, repairs selected gradients through path switching or stricter re
    
[^195]: 大型音频-语言模型中语音问答的机器遗忘

    Machine Unlearning for Speech Question Answering in Large Audio-Language Models

    [https://arxiv.org/abs/2609.13195](https://arxiv.org/abs/2609.13195)

    该论文首次研究了大型音频-语言模型中语音问答的机器遗忘问题，提出并评估了包括梯度上升、任务算术和对齐微调在内的多种遗忘策略，可将隐私泄露率降低多达80%的同时保持模型核心能力。

    

    大型音频-语言模型（LALMs）最近在语音理解和问答（QA）方面展现出了强大的能力，但它们也从大规模训练数据中继承了隐私风险，包括对敏感信息的无意记忆。在这项工作中，我们研究了LALMs中语音问答的机器遗忘问题，这一设置比之前基于文本的大型语言模型（LLMs）或自动语音识别（ASR）的工作更具挑战性，原因在于声学感知与事实知识之间存在紧密耦合。我们提出并评估了多种遗忘策略，包括梯度上升、任务算术以及强制安全拒绝响应的基于对齐的微调方法，以在移除私有知识的同时保持模型的核心能力。通过在语音问答数据集上的大量实验，我们表明这些遗忘方法可以将隐私泄露率降低多达80%，同时保持（模型性能）……

    arXiv:2609.13195v1 Announce Type: cross  Abstract: Large Audio-Language Models (LALMs) have recently shown strong capabilities in speech understanding and question answering (QA), but they also inherit privacy risks from large-scale training data, including the unintended memorization of sensitive information. In this work, we study machine unlearning for speech QA in LALMs, a setting that is more challenging than prior work on text-based Large Language Models (LLMs) or Automatic Speech Recognition (ASR) due to the tight coupling between acoustic perception and factual knowledge. We present and evaluate multiple unlearning strategies, including gradient ascent, task arithmetic, and alignment-based fine-tuning methods that enforce safe refusal responses, to remove private knowledge while still preserving performance on core capabilities. Through extensive experiments on speech QA datasets, we show that these unlearning methods can reduce the privacy leakage rate by up to 80% while maint
    
[^196]: 大语言模型还是朴素贝叶斯？旧瑰宝还是新方法

    LLMs or Naive Bayes? Old Gems or New Ways

    [https://arxiv.org/abs/2609.13185](https://arxiv.org/abs/2609.13185)

    本研究通过基准测试证明，在拥有标注数据的文本分类任务中，经典朴素贝叶斯方法不仅准确率可与甚至超越大规模语言模型，还能在普通CPU上以每秒数千样本的速度运行，有力挑战了“经典方法应被淘汰”的观念。

    

    大型语言模型（LLMs）引发了研究计算领域一个反复出现的问题：像朴素贝叶斯（NB）这样的经典方法是否应该被淘汰？我们在文本分类任务上，将补充朴素贝叶斯与涵盖四个模型家族、参数规模跨度达37倍（从270亿参数到1万亿参数混合专家模型）的零样本和少样本LLM进行了基准测试。结果显示，LLM仅在无标注数据场景中占主导地位（在Amazon Polarity情感分析任务上为98.0% vs 88.2%），即便这一优势也可能源于数据污染：在低污染的情感任务上，朴素贝叶斯击败了零样本LLM（81.7% vs 73.0%）。然而，一旦有标注数据可用（例如AG News数据集），朴素贝叶斯即可达到89.1%的准确率，在统计上与零样本270亿参数LLM（89.0%）不相上下，且优于3970亿参数的前沿模型（84.8%），同时在普通CPU上就能实现每秒数千个样本的处理速度。微调后的DistilBERT虽然达到了90.6%的准确率，但在批大小为1时的吞吐量远低于朴素贝叶斯（表2）。我们实测的GPU吞吐量分析显示，小规模……（摘要在此截断）

    arXiv:2609.13185v1 Announce Type: cross  Abstract: Large language models (LLMs) prompt a recurring question in research computing: should classical methods like Naive Bayes (NB) be retired? We benchmark Complement Naive Bayes against zero-shot and few-shot LLMs spanning four model families and a 37x range in scale (27B to a 1T-parameter mixture-of-experts) across text classification tasks. LLMs dominate only in zero-data regimes (98.0% vs 88.2% on Amazon Polarity sentiment), and even that win is contamination-prone: on a low-contamination sentiment task NB beats the zero-shot LLM (81.7% vs 73.0%). However, once labeled data is available (e.g., AG News), NB reaches 89.1% accuracy, statistically indistinguishable from the zero-shot 27B LLM (89.0%) and better than the 397B frontier model (84.8%), at thousands of samples/sec on a commodity CPU. Fine-tuned DistilBERT reaches 90.6% but at far lower throughput than NB at batch size 1 (Table 2). Our measured GPU throughput analysis shows small
    
[^197]: 算法验证作为政策审计：来自“种族盲指控”的证据

    Algorithm Validation as a Policy Audit: Evidence from Race-blind Charging

    [https://arxiv.org/abs/2609.13174](https://arxiv.org/abs/2609.13174)

    本文将算法验证应用于政策审计，验证了开源LLM算法bc2在加州种族盲指控政策中的表现，发现其在96.7%的警方报告叙述上忠实执行了法律要求，并进一步评估该政策本身能否有效实现种族盲决策的目标。

    

    加利福尼亚州最近要求该州所有检察官进行“种族盲指控”决策，即通过审阅已删除与种族相关代理信息的案件文件来做出指控决定。我们验证了bc2——这是我们开发的一个开源、基于大语言模型（LLM）的算法，用于自动化这一信息删除过程，并于2025年被用于促进超过119,000个真实案件的种族盲审阅。我们评估了两个不同的问题：bc2是否忠实地执行了该州的要求，以及这些要求即使被忠实执行，是否真正推进了种族盲决策的目标。为此，我们利用了从美国各地司法管辖区收集的近5,000份真实警方报告组成的语料库。在严格的文件层面衡量标准下，我们发现最新版本的bc2在我们样本中96.7%的叙述上忠实地执行了法律授权的要求。这一表现相较于早期版本 represents a substantial improvement over ear（摘要在此处被截断）

    arXiv:2609.13174v1 Announce Type: cross  Abstract: California recently required all prosecutors in the state to conduct a "race-blind charging" decision by reviewing case documents in which selected race-related proxies have been redacted. We validate bc2, an open-source, LLM-based algorithm that we developed to automate this redaction and that was used to facilitate race-blind review in more than 119,000 real-world cases in 2025. We evaluate two distinct questions: whether bc2 faithfully implements the state's requirements and whether those requirements, even when faithfully implemented, advance the goal of race-blind decision-making. To do so, we draw on a corpus of nearly 5,000 real-world police reports that we assembled from jurisdictions across the United States. Under a stringent document-level measure, we find that the latest version of bc2 faithfully implements the legal mandate on 96.7% of narratives in our sample. This performance represents a substantial improvement over ear
    
[^198]: 语音人工智能的跨社区议程

    A Cross Community Agenda for Speech AI

    [https://arxiv.org/abs/2609.13168](https://arxiv.org/abs/2609.13168)

    本文指出语音AI因技术社区（NLP）与社会技术社区（HCI）割裂而存在交流模型不完整、身份模型不完整和评估指标失准三大问题，并以辅助沟通（AAC）等弱势说话者场景为例，提出跨社区合作、面向人类多样性的研究议程。

    

    语音人工智能——即任何能够识别、转换或生成语音的AI系统——目前是在两个仅有少量重叠的社区中构建和评估的：技术性的自然语言处理（NLP）会议（如ACL、ICASSP、Interspeech），以及社会技术性的人机交互（HCI）会议（如ASSETS、CHI、FAccT）。在这篇立场论文中，我们致力于实现跨社区的综合，围绕三个问题展开批判：语音AI采用了一种不完整的交流模型；它采用了一种不完整的身份模型；其评估指标测量了错误的对象。我们以辅助与替代沟通（AAC）作为这些问题最为明显且利害关系最高的场景展开论述，同时关注其他服务不足的说话者群体——口吃者、多语言使用者以及非二元性别和跨性别用户。针对每个问题，我们提出了面向人类多样性设计的解决方案构想，其中几乎所有方案都需要定量与定性方法的结合。

    arXiv:2609.13168v1 Announce Type: cross  Abstract: Speech AI, any AI system that recognizes, transforms, or generates speech, is built and evaluated across two communities with only a small overlap: technical natural language processing (NLP) venues (e.g., ACL, ICASSP, Interspeech), and sociotechnical HCI venues (e.g., ASSETS, CHI, FAccT). In this position paper, we work toward a cross-community synthesis, organizing our critique around three problems: speech AI operates with an incomplete model of communication; it operates with an incomplete model of identity; and its metrics measure the wrong constructs. We draw on AAC as a setting where these failures are most visible and their stakes highest, alongside other underserved speakers - people who stutter, multilingual speakers, and non-binary and transgender users. For each problem we offer solution sketches oriented toward designing for human variability, nearly all of which require quantitative and qualitative methods in combination.
    
[^199]: TestHallVQA：探索大型视觉语言模型在来自科学考试的冗余上下文中的文档级推理能力

    TestHallVQA: Exploring LVLMs' Document-Level Reasoning under Redundant Contexts from Scientific Exams

    [https://arxiv.org/abs/2609.13158](https://arxiv.org/abs/2609.13158)

    提出了TestHallVQA基准——一个融合文档级规模与科学考试难度的多图像VQA基准，支持可控注入多层次上下文冗余，用于系统评估大型视觉语言模型在冗余上下文中的文档级推理能力。

    

    大型视觉语言模型（LVLMs）日益被期望能够在平面媒体上执行视觉问答（VQA）任务。然而，现有的平面VQA基准通常只强调孤立的挑战：一些侧重于推理深度有限的长文档理解，另一些虽然要求复杂的视觉推理，但仅限于单页、无噪声的设置。此外，通过理论分析，我们识别出无关视觉token的影响，这会导致可测量的性能下降，但此前很少受到系统性量化的关注。为了解决这些局限性，我们提出了TestHallVQA，这是一个多图像VQA基准，同时体现了文档级别的规模和人类考试的难度，并提供全面的任务覆盖。利用TestHallVQA能够可控地注入多层次上下文冗余的能力，我们进一步提出了一种新颖的度量指标（原文在此处截断）。

    arXiv:2609.13158v1 Announce Type: new  Abstract: Large Vision--Language Models (LVLMs) are increasingly expected to perform visual question answering (VQA) over planar media. However, existing planar VQA benchmarks typically emphasize isolated challenges: some emphasize long-document understanding with limited reasoning depth, while others require complex visual reasoning but remain restricted to single-page, noise-free settings. Moreover, through theoretical analysis, we identify the impact of irrelevant visual tokens, which leads to measurable performance degradation but has received little attention with respect to systematic quantification. To address these limitations, we introduce TestHallVQA, a multi-image VQA benchmark that simultaneously embodies document-level scale and the difficulty of human examinations, while providing comprehensive task coverage. Leveraging TestHallVQA's ability to controllably inject multi-level contextual redundancy, we further propose a novel metric, 
    
[^200]: 面向大语言模型的词汇级提示词压缩：一种免训练、确定性的流水线及其在十一个任务类别上的实证帕累托分析

    Lexical Prompt Compression for Large Language Models: A Training-Free, Deterministic Pipeline with Empirical Pareto Analysis Across Eleven Task Categories

    [https://arxiv.org/abs/2609.13154](https://arxiv.org/abs/2609.13154)

    本文提出了一种免训练、完全确定性、仅需CPU的基于经典词汇NLP的提示词压缩流水线，通过十一种可切换的词汇转换操作在十一个任务类别上对压缩率与输出质量的权衡进行了实证帕累托分析。

    

    大语言模型（LLM）的最新进展使得提示词变得越来越庞大和复杂。思维链推理（Wei et al., 2022）和上下文学习（Brown et al., 2020）等技术常常使真实场景中的提示词超过数千个token，从而增加推理成本和延迟。学习型压缩方法如LLMLingua（Jiang et al., 2023）和Selective Context（Li et al., 2023）能够实现较高的压缩率，但需要辅助语言模型且具有非确定性。我们提出了一个互补的问题：一条基于经典词汇NLP技术的免训练、完全确定性、仅依赖CPU的流水线，在输出质量显著下降之前能达到多大的压缩潜力？该流水线包含十一种可开关切换的词汇转换操作——停用词删除、填充短语删除、缩略语与缩写替换、基于词性的剪枝、词形还原、基于WordNet的同义词缩短以及命名实体保留——

    arXiv:2609.13154v1 Announce Type: new  Abstract: Recent advances in large language models (LLMs) have made prompts increasingly large and complex. Techniques such as chain-of-thought reasoning (Wei et al., 2022) and in-context learning (Brown et al., 2020) frequently push real-world prompts past several thousand tokens, increasing inference cost and latency. Learned compression methods such as LLMLingua (Jiang et al., 2023) and Selective Context (Li et al., 2023) achieve high compression ratios but require auxiliary language models and are non-deterministic. We ask a complementary question: how far can a training-free, fully deterministic, CPU-only pipeline based on classical lexical NLP be pushed before output quality degrades significantly? Eleven toggleable lexical transformations - stopword removal, filler-phrase deletion, contraction and abbreviation substitution, part-of-speech-based pruning, lemmatization, WordNet-driven synonym shortening, and named-entity preservation - are as
    
[^201]: PhysMent：一种用于大语言模型物理问题推理的交互式方法

    PhysMent: An Interactive Approach For LLM Reasoning In Physics Problems

    [https://arxiv.org/abs/2609.13152](https://arxiv.org/abs/2609.13152)

    PhysMent是一个通过与MuJoCo物理模拟器进行交互式实验来评估大语言模型物理推理能力的新基准，其核心创新在于要求模型通过施加力、查询状态等主动探索方式获取信息，而非被动接收完整信息，测试结果显示当前模型在简单定性任务上表现良好但仍有局限。

    

    大语言模型（LLMs）在静态科学基准测试中表现强劲，但它们通过主动实验来推理物理世界的能力仍然鲜为人知。我们提出了PhysMent，这是一个通过与MuJoCo物理模拟器进行迭代的、工具介导的交互来评估大语言模型物理推理能力的基准测试。与预先提供所有物理量的静态基准不同，PhysMent要求模型在回答之前通过施加力、查询物体状态、推进时间和修改场景几何结构来主动发现信息。该基准包含105个经典力学场景，分为四个难度级别（简单/困难和单概念/多概念）、三种场景模态（标准、物体创建、隐藏物体）以及一个场景操作类别，并采用六维评分框架进行评估。结果表明，当前模型在定性单概念任务上表现相当不错（准确率高达80%），但是……

    arXiv:2609.13152v1 Announce Type: new  Abstract: Large language models (LLMs) perform strongly on static science benchmarks, yet their ability to reason about the physical world through active experimentation remains poorly understood. We introduce PhysMent, a benchmark that evaluates LLM physical reasoning via iterative, toolmediated interaction with a MuJoCo physics simulator. Unlike static benchmarks that supply all quantities upfront, PhysMent requires models to discover information by applying forces, querying object states, advancing time, and modifying scene geometry before answering. The benchmark comprises 105 scenes of classical mechanics, organized across four difficulty regimes (Easy/Hard and Single/Multi), three scene modalities (standard, object creation, hidden objects), and a scene-manipulation category, evaluated with a six-dimensional scoring framework. Results show that current models perform reasonably well on qualitative single-concept tasks (up to 80% accuracy) bu
    
[^202]: 面向多语言语音识别的Token合并：跨模型规模与微调的系统性研究

    Token Merging for Multilingual Speech Recognition: A Systematic Study Across Model Scale and Fine-Tuning

    [https://arxiv.org/abs/2609.13151](https://arxiv.org/abs/2609.13151)

    对Whisper模型家族的系统性研究表明，token合并能在几乎不损失转录准确率的情况下显著提升多语言语音识别的计算效率，并且在不同模型规模和低资源语言微调后均能有效工作。

    

    像Whisper这样的领先多语言语音识别模型无需针对特定语言进行训练即可转录多种低资源语言，但部署时计算成本高昂。Token合并通过动态组合冗余特征来缓解这种低效问题，在推理过程中缩短序列长度，且无需重新训练。在本文中，我们在Whisper模型家族上，针对十六种不同的语言和三种不同的模型规模，系统性地评估了token合并的效果。我们还测试了token合并与低资源语言微调（DoRA）之间的相互作用。研究结果表明，在大多数低资源语言和模型规模下，合并token能够提升计算效率，同时几乎不损失转录准确率，并且在模型经过微调之后依然有效。我们的结果证明，token合并是一种让多语言语音识别更快、更便宜的高实用性方法。

    arXiv:2609.13151v1 Announce Type: new  Abstract: Leading multilingual speech recognition models like Whisper transcribe diverse, low-resource languages without language-specific training but are computationally expensive to deploy. Token merging mitigates this inefficiency by dynamically combining redundant features, shortening the sequence length during inference without requiring retraining. In this paper, we systematically evaluate token merging on the Whisper model family across sixteen diverse languages and three different model sizes. We also test how token merging interacts with fine-tuning (DoRA) on low-resource languages. Our findings show that merging tokens increases computational efficiency with almost no loss in transcription accuracy across most low-resource languages and model sizes, and it works even after the model has been fine-tuned. Our results demonstrate that token merging is a highly practical method for making multilingual speech recognition faster and cheaper t
    
[^203]: 无参考语音质量指标作为现代文本转语音评估器和奖励信号的局限性

    The Limits of Reference-Free Speech Quality Metrics as Evaluators and Rewards on Modern Text-to-Speech

    [https://arxiv.org/abs/2609.13150](https://arxiv.org/abs/2609.13150)

    该研究通过成对偏好测试发现，无参考语音质量指标（UTMOS、DNSMOS、SCOREQ）在比较两个均无缺陷的TTS样本时无法可靠预测人类偏好，甚至不如简单选择最长片段的基线方法，这质疑了其作为TTS自动评估器和偏好优化奖励信号的有效性。

    

    诸如UTMOS、DNSMOS和SCOREQ等无参考质量预测器是文本转语音（TTS）领域事实上的自动评估标准，并且越来越多地被用作偏好优化的奖励信号。这两种角色都基于一个前提假设：预测分数能够反映人类偏好。在这项工作中，我们在六个涵盖从伪影较多到无缺陷TTS质量范围的人工评分语料库上测试了这一假设，对每个预测器进行成对评估任务，即检验其评分较高的片段是否是听众偏好的片段，并对可解释的韵律特征和信号处理特征采用相同的评估协议。实验结果表明：当一个片段存在可听见的缺陷时，这些预测器倾向于与听众的判断一致；然而一旦两个片段都没有明显缺陷，没有任何单一预测器能够可靠地识别出听众偏好的样本，其中几个预测器的准确率甚至低于简单选择最长时长片段的方法。经过校准的互补信号组合是最强的评估方案。

    arXiv:2609.13150v1 Announce Type: cross  Abstract: Reference-free quality predictors such as UTMOS, DNSMOS and SCOREQ are the de facto automatic evaluators for text-to-speech (TTS) and are increasingly adopted as reward signals for preference optimization. Both roles presuppose that the predicted score tracks human preference. In this work, we test this assumption across six human-rated corpora spanning the quality range from artifact-rich to defect-free TTS, evaluating each predictor on a pairwise task that asks whether the clip it scores higher is the clip listeners prefer, and we subject interpretable prosodic and signal-processing features to the same protocol. When one clip carries audible defects the predictors tend to agree with listeners. Once both clips are clean, no single predictor reliably identifies the preferred sample, and several fall below the accuracy of simply picking the longest-duration clip. A calibrated composite of complementary signals is the strongest evaluato
    
[^204]: BudgetBench：用于本地大语言模型智能体记忆策略评估的预算分层协议与试点测试框架

    BudgetBench: A Budget-Tiered Protocol and Pilot Harness for Memory Strategy Evaluation in Local Large Language Model Agents

    [https://arxiv.org/abs/2609.13149](https://arxiv.org/abs/2609.13149)

    BudgetBench提出了一种以每次调用输入令牌预算为自变量的记忆策略评估协议，通过在2K至32K令牌的多级预算下固定模型与任务、测量质量、延迟和预算违规率，为本地大语言模型智能体的记忆策略比较提供了可复用的标准化测量框架。

    

    对于本地大语言模型智能体而言，活跃上下文是一种稀缺资源：内存容量、预填充延迟、缓存增长以及服务目标都限制了每次调用能够承担的输入令牌数量。我们提出了BudgetBench，这是一种主动预算协议和参考测试框架，在比较记忆策略时将每次调用的输入令牌预算作为自变量。在保持模型、任务、采样器和解码固定不变的情况下，它对2K、4K、8K、16K和32K令牌的预算进行扫描，并记录质量、预算利用率、延迟，以及作为一等结果的预算违规率。其核心贡献是这一可复用的测量平台：一个可替换的MemoryStrategy契约、明确的预算执行机制、确定性或版本化的评分器、提示审计元数据以及可复现性工件，已在 https://github.com/aviskaar/budgetbench 发布。我们通过试点研究而非最终排名来验证该协议。

    arXiv:2609.13149v1 Announce Type: cross  Abstract: For local large language model agents, active context is a scarce resource: memory capacity, prefill latency, cache growth, and service objectives all constrain how many input tokens each call can afford. We present BudgetBench, an active-budget protocol and reference harness that treats the per-call input-token budget as the independent variable when comparing memory strategies. Holding the model, task, sampler, and decoding fixed, it sweeps budgets over 2K, 4K, 8K, 16K, and 32K tokens and records quality, budget utilization, latency, and, as a first-class outcome, budget-violation rates. The core contribution is this reusable measurement surface: a swappable MemoryStrategy contract, explicit budget enforcement, deterministic or versioned graders, prompt-audit metadata, and reproducibility artifacts, released at https://github.com/aviskaar/budgetbench.   We substantiate the protocol with pilot studies rather than final rankings. Acros
    
[^205]: 智能体作为策略的机器人操作

    Agent as Policy for Robotic Manipulation

    [https://arxiv.org/abs/2609.12541](https://arxiv.org/abs/2609.12541)

    该论文提出Agent as Policy (AGP)方法，使通用智能体无需任何任务特定训练即可直接控制物理机器人完成从精细操作、动态运动到可变形物体处理等多种真实操作任务。

    

    我们证明了通用智能体可以在整个任务执行过程中直接驱动物理机器人，而无需任何针对特定任务或特定环境的训练。我们提出了智能体作为策略方法，将任务规划和执行都置于智能体的控制之下。给定一个任务和机器人接口，智能体能够解读视觉信息、编写可执行程序、发出运动指令，并根据物理执行结果修正自身动作。这将智能体的推理和编程能力带入了与物理世界的持续交互之中。我们在多个真实世界的操作任务上研究了AGP，涵盖精细操作、动态运动和可变形物体，包括从人类视频中学习装配、根据目标图像搭建积木、翻转骰子、定向投掷以及双臂协作折叠毛巾。AGP在三种积木搭建配置上分别取得了100%、100%和80%的成功率。

    arXiv:2609.12541v1 Announce Type: new  Abstract: We demonstrate that a general-purpose agent can directly drive a physical robot throughout task execution without any task-specific or environment-specific training. We introduce Agent as Policy (AGP), which places task planning and execution under the agent's control. Given a task and a robot interface, the agent interprets visual evidence, writes executable programs, issues motion commands, and revises its actions in response to physical outcomes. This brings the agent's reasoning and programming capabilities into continuous interaction with the physical world. We study AGP across multiple real-world manipulation tasks spanning precision manipulation, dynamic motions, and deformable objects. These include assembly from human videos, block construction from goal images, die reorientation, targeted throwing, and bimanual towel folding. AGP achieves success rates of 100%, 100%, and 80% on three block construction configurations. These fin
    
[^206]: 一种无需训练、无需对齐的企业情报分析方法：在SEC文件中的应用

    A Training-Free, Alignment-Free Approach to Corporate Intelligence: Application to SEC Filings

    [https://arxiv.org/abs/2609.11620](https://arxiv.org/abs/2609.11620)

    该论文提出一种基于确定性稀疏种子向量的无需训练、无需对齐的企业情报分析框架，可在普通CPU上实现对SEC文件的亚秒级比较、发行人指纹识别及主题句提取。

    

    高维稠密文本嵌入和大语言模型在金融信息披露分析中面临诸多实际障碍：上下文窗口限制、幻觉风险、高昂的计算成本，以及独立训练模型之间向量空间的任意旋转问题。我们提出了一种基于确定性稀疏种子向量的无需训练、无需对齐的企业情报框架。通过将词字符串哈希到固定的高维基中，所有文档和所有时间阶段在构建上即处于同一坐标系中，从而消除了对训练或对齐的任何需求。在句子上下文中累积这些种子向量可产生可线性组合的语料库特定语义签名，支持亚秒级文档比较、发行人指纹识别、跟踪发行人在不同文件之间词汇的变化，以及主题句提取——所有这些均可基于普通CPU硬件完成。我们在多年语料库上对该方法进行了演示验证。

    arXiv:2609.11620v1 Announce Type: new  Abstract: High-dimensional dense text embeddings and large language models face real obstacles in financial-disclosure analysis: context-window limits, hallucination risk, high computational cost, and the arbitrary rotation of vector spaces across independently trained models. We present a training-free, alignment-free framework for corporate intelligence built on deterministic sparse seed vectors. Hashing word strings into a fixed high-dimensional basis places all documents and all temporal epochs in a common coordinate system by construction, removing any need for training or alignment. Accumulating these seed vectors across sentence contexts yields corpus-specific semantic signatures that compose linearly, supporting sub-second document comparison, issuer fingerprinting, tracking of how an issuer's vocabulary shifts between filings, and thematic sentence extraction, all on ordinary CPU hardware. Demonstrating the approach on a multi-year corpus
    
[^207]: 面向智能手术室的多智能体语音交互系统：架构设计与关键技术

    A Voice-Interactive Multi-Agent System for Smart Operating Rooms: Architecture Design and Key Technologies

    [https://arxiv.org/abs/2609.11231](https://arxiv.org/abs/2609.11231)

    本文提出基于大语言模型的智能手术室语音交互多智能体系统SurgicalRoomAgent，通过KV Cache前缀预热、流式JSON解析和渐进式技能提示披露三项关键技术大幅降低推理延迟，实现自然语言理解、设备控制、术中记录与手术报告生成。

    

    本文提出了SurgicalRoomAgent，一个基于大语言模型（LLM）的智能手术室多智能体语音交互系统。该系统通过分层架构实现自然语言理解、设备控制、术中记录和手术报告生成，该架构由语音交互流水线（唤醒、ASR、轮次检测、智能体推理、TTS）和智能体核心（技能注册表、任务规划器、设备管理器）组成。论文研究了三项关键技术：（1）面向低延迟推理的KV Cache前缀预热，通过字节级最长公共前缀重用，将重计算开销从约500毫秒降低至几十毫秒；（2）流式部分JSON解析与早期并行任务执行，将端到端延迟降低约30%；（3）渐进式技能提示披露，根据用户角色、已连接设备和手术阶段动态过滤系统提示，以最大化提示效率和安全性。

    arXiv:2609.11231v1 Announce Type: cross  Abstract: This paper presents SurgicalRoomAgent, a voice-interactive multi-agent system for smart operating rooms based on large language models (LLMs). The system achieves natural language understanding, device control, intraoperative recording, and surgical report generation through a layered architecture comprising a voice interaction pipeline (wake, ASR, turn detection, agent reasoning, TTS) and an agent core (skill registry, task planner, device manager). Three key technologies are investigated: (1) KV Cache prefix warming for low-latency inference, reducing recomputation overhead from approximately 500 ms to tens of milliseconds via byte-level Longest Common Prefix reuse; (2) streaming partial JSON parsing with early parallel task execution, reducing end-to-end latency by approximately 30%; and (3) progressive skill prompt disclosure, which dynamically filters system prompts based on user role, connected devices, and surgical phase to maxi
    
[^208]: 基于并行解码的电商属性提取规模化方法

    Scaling E-Commerce Attribute Extraction with Parallel Decoding

    [https://arxiv.org/abs/2609.09716](https://arxiv.org/abs/2609.09716)

    提出了一种两阶段LLM流水线，先为每个产品类别发现紧凑且按重要性排序的购买判别属性模式，再利用微调的Qwen3-4B结合超并行解码技术提取属性值，在达到与基础LLM相当的85%提取准确率的同时，将推理成本降低了92%，实现了生产级规模化应用。

    

    客户依靠特定的产品属性来比较商品并做出购买决策，但电商目录往往杂乱且非结构化，这使得难以识别哪些属性最为重要并进行规模化提取。标准的属性值提取（AVE）系统对所有属性一视同仁，产生了庞大且不一致的属性集合，无法反映消费者用来区分产品的关键因素。我们引入了一个两阶段的大语言模型（LLM）流水线：首先为每个产品类别发现一个紧凑的、按重要性排序的购买判别属性模式，然后使用经过微调的紧凑型大语言模型（Qwen3-4B）结合超并行解码（HPD）技术从目录文本中提取这些属性的值。该流水线达到了85%的提取准确率，与它所蒸馏自的基础大语言模型相当，同时相比基础大语言模型将推理成本降低了92%，实现了产品发现和目录丰富化的生产级规模应用。

    arXiv:2609.09716v1 Announce Type: new  Abstract: Customers rely on specific product attributes to compare products and make purchasing decisions, but e-commerce catalogs are messy and unstructured, making it difficult to identify which attributes matter most and extract them at scale. Standard Attribute Value Extraction (AVE) systems treat all attributes equally, producing large, inconsistent attribute sets that do not reflect the factors consumers use to differentiate products. We introduce a two-stage LLM pipeline that first discovers a compact, ranked schema of purchase-discriminative attributes for each product category, then extracts their values from catalog text using a fine-tuned compact LLM (Qwen3-4B) with Hyper-Parallel Decoding (HPD). This pipeline achieves 85% extraction accuracy, on par with the foundational LLM it was distilled from, while reducing inference costs by 92% over foundational LLMs, enabling production-scale use for product discovery and catalog enrichment. Th
    
[^209]: 桥接网络心理测量学与人工智能：基于大语言模型权重的伊辛-波茨模型

    Bridging Network Psychometrics and Artificial Intelligence: An Ising-Potts Model with LLM-Derived Weights

    [https://arxiv.org/abs/2609.08797](https://arxiv.org/abs/2609.08797)

    本文提出一种融合大语言模型嵌入权重的评分者伊辛-波茨模型，通过评分间的两两一致性而非预设阈值来评估多类别评分信度，并在三个大规模数据集上比较了三种相似性增强策略。

    

    波茨模型将伊辛模型扩展至多类别数据。我们提出了一种评分者伊辛-波茨模型，该模型利用评分对与类别标签之间的一致性指标，其权重源自大语言模型（LLM）嵌入。该模型不预设有序的类别阈值或等距评分，而是关注评分之间的两两一致性，并为各类别分配特定的正权重，因此适用于多类别评分的信度评估。我们在三个建构性反应数据集上对该模型进行了评估，这些数据集涵盖了一个包含 K=14,466 份简短答案的三级评分量表语料库，以及两个 AERA 作文题目（各约 1,200-1,400 份答案，采用四级评分量表）。我们比较了三种增强相似性信号的策略：Top-K 剪枝、带幂变换的最小-最大归一化，以及 ColBERT 后期交互相似度。Top-K 剪枝用稀疏局部网络替换稠密相似图……（原文摘要在此处截断）

    arXiv:2609.08797v2 Announce Type: replace-cross  Abstract: The Potts model extends the Ising model to multinomial data. We introduce a Rater Ising-Potts model that uses agreement indicators between pairs of ratings and category labels, with weights derived from LLM embeddings. The model does not presuppose ordered category thresholds or equidistant scoring; instead, it focuses on pairwise agreement among ratings and assigns category-specific positive weights, making it suited for multi-category scoring reliability. We evaluate the model on three constructed-response datasets spanning a corpus of K=14,466 short answers on a three-level rubric and two AERA essay prompts of roughly 1,200-1,400 responses on four-point rubrics. We compare three strategies for sharpening the similarity signal: top-K pruning, min-max normalization with a power transformation, and ColBERT late-interaction similarities. Top-K pruning, which replaces the dense similarity graph with a sparse local network of stro
    
[^210]: HoneyRoute：面向对抗性LLM服务的蜜罐模型路由

    HoneyRoute: Honeypot-Model Routing for Adversarial LLM Serving

    [https://arxiv.org/abs/2609.08306](https://arxiv.org/abs/2609.08306)

    HoneyRoute是一个部署在推理服务层的防御框架，通过轻量级流式路由器实时检测恶意请求并将其路由至蜜罐模型，在保护生产LLM服务的同时，将捕获的攻击者交互转化为指纹数据用于持续改进路由器的检测能力。

    

    我们提出了HoneyRoute，一个推理服务层，它能够检测传入请求是否为恶意请求，若检测到恶意，则将其路由到专用的蜜罐模型，从而保护生产系统，同时持续收集对手交互信息以获取情报。现有的防御方法要么在模型记忆中嵌入陷阱，要么在协议层重建欺骗机制，这使得服务层缺乏保护，且无法将捕获的信息反馈用于检测。HoneyRoute结合了三个组件：(i) 一个流式路由器（采用冻结的0.8B参数嵌入骨干网络，配备各领域的MLP头），(ii) 双实现的蜜罐（规则/提示词工程化的代码蜜罐，或专用的同族模型副本），以及 (iii) 一个分析循环，将被捕获的交互转化为攻击者指纹，用于路由器的再训练。在生产环境追踪数据和七个领域的攻击语料库上，该路由器达到F1=.911，中位附加延迟仅38毫秒，以两级防护LLM级联系统1/385的延迟实现了其96%的F1分数。

    arXiv:2609.08306v2 Announce Type: replace-cross  Abstract: We introduce HoneyRoute, an inference-serving layer that detects whether an incoming request is malicious and, if so, routes it to a dedicated honeypot model, shielding production while the adversary's interaction is continuously harvested for intelligence. Existing defenses embed traps inside model memory or rebuild deception at the protocol layer, leaving the serving tier unprotected and feeding nothing back into detection. HoneyRoute couples (i) a streaming router (a frozen 0.8B-embedding backbone with per-domain MLP heads), (ii) a dual-implementation honeypot (a rule/prompt-engineered code honeypot or a dedicated same-family replica), and (iii) an analysis loop that converts trapped interactions into attacker fingerprints for router retraining. On a production trace plus a seven-domain attack corpus, the router reaches F1=.911 at 38 ms median added latency, matching 96% of a two-tier guard-LLM cascade's F1 at 1/385 of its l
    
[^211]: 语言在多模态模型中应处于何种位置？来自语言对人类感知与认知作用的启示

    Where Should Language Sit in a Multimodal Model? Lessons from What Language Does to Human Perception and Cognition

    [https://arxiv.org/abs/2609.07474](https://arxiv.org/abs/2609.07474)

    该论文将语言类比为运行在共享码本上的压缩器，通过回顾语言对人类感知、大脑和思维的作用，并结合对六个视觉-语言模型和两个机器人策略的线索冲突实验，探讨语言在多模态模型中应占据的合适位置。

    

    语言模型在词元（token）之上进行计算：语言是它们的输入、输出，并且日益成为它们的内部表示。语言是否应当保留所有这些位置，取决于语言对使用它的系统产生了什么作用。而关于这一问题，唯一拥有一个世纪数据积累的系统就是人类。我们回顾了语言对人类感知、大脑和思维的作用，并将同样的证据与多模态模型和语言模型进行对照分析。自始至终，我们将语言视为一个运行在共享码本之上的压缩器：一个词只是一个索引，内容存在于接收者端，而由一个群体共同维护这个码本。在人类中，这种压缩是可测量的，学习码本会重组感官系统，而思维在失去语言之后依然能够存续。随后，我们通过线索冲突实验测量了当两条线索相矛盾时模型所应用的规则，实验对象包括六个视觉-语言模型和两个机器人策略。被保留下来的线索按其各自来源的顺序被赋予权重。

    arXiv:2609.07474v2 Announce Type: replace  Abstract: Language models compute over tokens: language is their input, their output, and increasingly their internal representation. Whether language should keep all of these positions depends on what language does to the system that uses it. The one system with a century of data on that question is the human. We review what language does to human perception, the brain, and thought, and read the same evidence against multimodal models and language models. Throughout, we treat language as a compressor that runs on a shared codebook: a word is an index, the content is in the receiver, and a community maintains the codebook. In humans the compression is measurable, learning the codebook reorganizes the senses, and thought survives the loss of language. We then measure the rule that models apply when two cues disagree, with cue-conflict experiments on six vision-language models and two robot policies. Surviving cues are weighted in the order thei
    
[^212]: 面向长上下文的基于内容寻址方法

    Content-Based Addressing for Long Context

    [https://arxiv.org/abs/2609.07314](https://arxiv.org/abs/2609.07314)

    提出一种基于内容寻址的长上下文位置编码方法，将 token 流划分为单元、单元内保留 RoPE 并为每个完成的单元按内容计算地址，从而在扩展上下文时完全保留局部 RoPE 并使固定 token 间的注意力比较不受其他单元插入或重排的影响。

    

    旋转位置编码使用每个 token 的整数位置来确定注意力内部施加的旋转。这对局部 token 顺序效果良好，但随着上下文长度的增加，会产生位置上的训练-测试不匹配：RoPE 会在训练中未见过的偏移量处产生相对旋转。那些对位置进行重缩放、插值、随机化或加偏置的方法规定了注意力如何处理这些偏移量，但仍然从不断增长的 token 计数器中获取位置信息。我们转而将 token 流划分为单元，在每个单元内保留普通的 RoPE 位置，并为每个已完成的单元分配一个由其内容计算得出的地址。这样，添加单元时只需将同一个学习到的映射应用于新内容，而无需扩展位置范围或标识符表。我们证明该构造能够完全保留局部 RoPE，当其他单元被插入或重排时，两个固定 token 之间的注意力比较保持不变，并且……（摘要在此处被截断）

    arXiv:2609.07314v1 Announce Type: cross  Abstract: Rotary position embedding (RoPE) uses each token's integer position to determine the rotation applied inside attention. This works well for local token order, but increasing context length creates a positional train-test mismatch: RoPE produces relative rotations at offsets not seen during training. Methods that rescale, interpolate, randomize, or bias positions specify how attention handles those offsets, but still derive positional information from a growing token counter. We instead divide a token stream into units, retain ordinary RoPE positions within each unit, and assign every completed unit an address computed from its content. Adding units then applies the same learned map to new content rather than extending a positional range or an identifier table. We prove that this construction preserves local RoPE exactly, leaves the attention comparison between two fixed tokens unchanged when other units are inserted or reordered, and d
    
[^213]: 往哪里看与用什么：面向长期对话记忆问答的检索-定位-生成框架

    Where to Look and What to Use: Retrieve-Localize-Generate for Long-Term Conversational Memory Question Answering

    [https://arxiv.org/abs/2609.07093](https://arxiv.org/abs/2609.07093)

    提出了MemLoc统一框架，通过多粒度记忆分解、记忆内图查询路由与跨记忆图建模实现由粗到细的检索，并引入定位机制过滤噪声，从而解决长期对话记忆问答中证据碎片化与“迷失在中间”效应两大挑战。

    

    检索增强生成（RAG）使大语言模型（LLM）能够通过访问外部知识来回答问题，并已被广泛应用于长期对话记忆问答。然而，现有方法面临两个关键挑战：（1）证据碎片化，分散在时间上相距较远的会话中；（2）检索到的会话中存在噪声内容，会触发“迷失在中间”效应。为应对这些挑战，我们提出了MemLoc，一个面向长期对话记忆问答的统一检索-定位-生成框架。在检索方面，MemLoc将每个会话分解为多粒度记忆单元，并通过具有基于熵的粒度选择的记忆内图进行查询路由。它进一步通过跨记忆图建模跨会话的语义与时间依赖关系，实现对前K个相关记忆候选的由粗到细的检索。在定位方面，我们引入了一个重…（摘要在此处被截断）

    arXiv:2609.07093v2 Announce Type: replace  Abstract: Retrieval-augmented generation (RAG) enables large language models (LLMs) to answer questions by accessing external knowledge and has been widely adopted for long-term conversational memory question answering. However, existing methods suffer from two key challenges: (1) fragmented evidence scattered across temporally distant sessions, and (2) noisy content within retrieved sessions that triggers the lost-in-the-middle effect. To address these challenges, we propose MemLoc, a unified Retrieve-Localize-Generate framework for long-term conversational memory QA. For retrieval, MemLoc decomposes each session into multi-granularity memory units and performs query routing via an inner-memory graph with entropy-based granularity selection. It further models cross-session semantic and temporal dependencies through a cross-memory graph, enabling coarse-to-fine retrieval of top-K relevant memory candidates. For localization, we introduce a rea
    
[^214]: Delta注意力机制中的精确记录删除：传输判据、其代价与重放证书

    Exact Record Omission in Delta Attention: A Transport Criterion, Its Cost, and a Replay Certificate

    [https://arxiv.org/abs/2609.06872](https://arxiv.org/abs/2609.06872)

    该论文证明了在Delta注意力类循环记忆中，通过“收据”传递实现记录精确删除当且仅当该记录诱导的后续更新变化净额为零，并在48B Kimi Linear混合模型上实证表明该条件不成立——记录会留下现有收据类方法均无法消除、重算也只能部分弥补的持久印记。

    

    当用户要求助手忘记某条记录时，检验标准是：记忆当前的状态是否与该记录从未被存储时本应处于的状态一致。独立编码的行可以直接删除；而循环记忆则会将记录折叠进一个不断演化的状态中。一种期望的方案是“收据”（receipt）：保存该记录到来时所产生的差异，在后续更新中将其向前传递，然后将其减去，从而无论对话进行多长，删除都只需一次固定大小的编辑。我们证明，被传递的收据能够实现精确删除，当且仅当该记录在后续更新中所诱导的变化在净额上相互抵消；并且我们在公开发布的48B Kimi Linear混合模型上实测了这种抵消是否发生。结果表明并未发生：在再经过4,096个token之后，该记录仍留下约占状态范数4.5%的印记，所测试的任何一类收据都无法将其移除；重算后半段后缀也只能弥补不到一半的差距；而收据所需的逐token日志的代价是……（摘要原文在此处截断）

    arXiv:2609.06872v1 Announce Type: new  Abstract: When a user asks an assistant to forget a record, the test is whether the memory now matches the state it would hold if the record had never been stored. Independently encoded rows can be removed directly; a recurrent memory folds records into an evolving state. One hope is a receipt: save the difference the record made when it arrived, carry it forward through later updates, and subtract it, so that deletion costs one fixed-size edit no matter how long the conversation runs. We show that a transported receipt reaches exact omission if and only if the changes the record induces in later updates cancel out on net, and we measure whether they do on the released 48B Kimi Linear hybrid. They do not: after 4,096 further tokens the record still leaves an imprint of about 4.5% of the state norm that none of the tested receipt classes removes, recomputing half the suffix closes less than half the gap, and the per-token log a receipt needs costs 
    
[^215]: 面向上下文学习的数据高效样本选择

    Data Efficient Sample Selection for In-Context Learning

    [https://arxiv.org/abs/2609.06670](https://arxiv.org/abs/2609.06670)

    提出了DearICL框架，将ICL示例选择建模为子集排名问题，通过结合间隔索引赌博机算法与可微排序目标的非线性替代模型，实现了数据高效且能泛化到未见查询的样本选择。

    

    上下文学习（ICL）范式帮助大语言模型（LLM）无需微调即可适应新任务。然而，从大量示例子集中选择最优的示例组合是一个具有挑战性的问题。现有的选择方法没有建模ICL样本与下游LLM性能之间的复杂关系。它们通常执行静态的任务级选择，一次性离线选择子集，这可能无法泛化到未见过的查询。我们提出了DearICL（用于排名ICL样本的数据高效算法），这是一个将示例选择建模为子集排名问题的新框架。DearICL在基于间隔索引（gap-index）的赌博机算法中采用了具有可微排序目标的非线性替代模型。基于间隔索引的方法能够对优质臂和边缘臂进行细粒度的区分，并将其作为辅助目标来训练非线性替代模型。

    arXiv:2609.06670v1 Announce Type: new  Abstract: The In-context learning (ICL) paradigm aids large language models (LLMs) to adapt to new tasks without need for fine-tuning. However, selecting an optimal combination of demonstration examples from a large pool of example subsets is a challenging problem. Existing approaches for selection do not model the complex relationship between ICL samples and downstream LLM performance. They typically perform static task-level selection, choosing subsets once offline, which can fail to generalize to unseen queries. We introduce DearICL (Data Efficient Algorithm for Ranking) ICL samples, a new framework that models demonstration example selection as a subset ranking problem. DearICL employs a non-linear surrogate employing a differentiable sorting objective within a gap-index bandit algorithm. The gap-index based approach enables fine-grained separation of good arms and borderline arms, which is used as an auxiliary objective to train the non-linea
    
[^216]: 面向Softmax注意力的查询无关核心集：改进的界与高效构造

    Query-Oblivious Coresets for Softmax Attention: Improved Bounds and Efficient Constructions

    [https://arxiv.org/abs/2609.06327](https://arxiv.org/abs/2609.06327)

    本文通过球面提升和Chevet不等式将softmax注意力核心集问题转化为指数核实例，首次给出在查询半径内对所有查询均有效的、与理论下界仅相差 $\sqrt{\log(1+\rho)}$ 的随机多项式时间构造，无需新技术即弥合了Liberty等人提出的理论差距。

    

    对于softmax注意力头而言，查询无关核心集是键值对的一个子集，其注意力输出对球内每个查询都与全量输出的误差在 $\varepsilon$ 以内。Liberty、Andoni和Kleiner证明了大小为 $O(\sqrt d\, e^{\rho+\frac12\log\rho+o(\log\log\rho)}/\varepsilon)$ 的无权核心集存在（其中 $\rho$ 为查询半径与中心化键半径的乘积），而对应的下界为 $\Omega(\sqrt d\, e^{\rho}/\varepsilon)$，并推测弥合这一差距需要新的技术。事实并非如此：通过将两个球进行球面提升并映射到同一指数核实例，可使Bozzai-Rothvoss链接界得以应用；再借助Chevet不等式将键维度与值维度分离，即可在随机多项式时间内构造出大小为 $O(e^{\rho}(\sqrt{d_v}+\sqrt{d_k\log(1+\rho)})/\varepsilon)$ 的核心集，这是首个与下界相差仅 $\sqrt{\log(1+\rho)}$ 因子以内的构造性全球保证。此外还给出了 $O(e^{2\rho}/\varepsilon^{2})$ 的采样上限……

    arXiv:2609.06327v2 Announce Type: replace-cross  Abstract: A query-oblivious coreset for a softmax-attention head is a subset of the key-value pairs whose attention output is within $\varepsilon$ of the full one for every query in a ball. Liberty, Andoni and Kleiner proved that unweighted coresets of size $O(\sqrt d e^{\rho+\frac12\log\rho+o(\log\log\rho)}/\varepsilon)$ exist, $\rho$ the query radius times the centred key radius, against a lower bound $\Omega(\sqrt d e^{\rho}/\varepsilon)$, and conjectured that closing the gap needs new techniques. It does not: a spherical lift of both balls into one exponential-kernel instance lets the Bozzai-Rothvoss chaining bound apply, and Chevet's inequality splits key from value dimension, giving coresets of size $O(e^{\rho}(\sqrt{d_v}+\sqrt{d_k\log(1+\rho)})/\varepsilon)$ in randomised polynomial time, the first constructive whole-ball guarantee within $\sqrt{\log(1+\rho)}$ of the lower bound. A sampling cap $O(e^{2\rho}/\varepsilon^{2})$ compl
    
[^217]: 从边缘到联合的门票：扩散语言模型中一步式块生成的耦合噪声蒸馏

    A Ticket from Marginals to Joints: Coupled-Noise Distillation for One-Step Block Generation in Diffusion Language Models

    [https://arxiv.org/abs/2609.06324](https://arxiv.org/abs/2609.06324)

    提出CONDOR方法，通过耦合噪声蒸馏从零训练扩散语言模型，使其在掩码嵌入受高斯噪声扰动时仅用一次前向传播即可生成连贯的整块文本，且无需目标侧编码器或自回归教师。

    

    扩散语言模型并行预测一个块中的所有词元，但单次前向传播是从各自的边缘分布中采样每个位置，因此这些词元不一定能构成一个连贯的块。我们提出一个问题：当离散掩码模型的掩码嵌入受到采样高斯噪声场扰动时，该模型能否在单次传播中生成整个块——相同的噪声应给出相同的连贯续写，而不同的噪声应给出不同的续写。我们提出了CONDOR（Coupled-Noise Distillation for One-Step Readout，面向一步式读出的耦合噪声蒸馏），它无需目标侧编码器或自回归教师即可从头训练这样的模型。训练结合了两种信号。在真实文本上，模型在多个噪声样本下预测被掩码的词元，并仅通过与真实值最匹配的那个样本进行监督，从而使不同的噪声可以专精于不同的续写。对于其余样本，模型则细化自身的单次传播预测……

    arXiv:2609.06324v2 Announce Type: replace  Abstract: Diffusion language models (dLLMs) predict all tokens of a block in parallel, but a single forward pass samples each position from its own marginal distribution, so the tokens need not form a coherent block. We ask whether a discrete masked model can commit an entire block in one pass when its mask embeddings are perturbed by a sampled Gaussian noise field: the same noise should give the same coherent continuation, and different noise should give different ones. We propose CONDOR (Coupled-Noise Distillation for One-Step Readout), which trains such a model from scratch without a target-side encoder or an autoregressive teacher. Training combines two signals. On real text, the model predicts masked tokens under several noise samples and is supervised only through the sample that fits the ground truth best, so different noise can specialize to different continuations. For the remaining samples, the model refines its own one-pass predicti
    
[^218]: 剖析医学问答中误导性上下文的作用机制

    Untangling the Mechanisms of Misleading Context in Medical Question Answering

    [https://arxiv.org/abs/2609.02754](https://arxiv.org/abs/2609.02754)

    该研究通过在MedMisBench中注入伪造证据和纯粹断言两类误导性上下文，系统揭示了推理模型的医学判断被误导的机制，发现模型对纯粹断言的易感性显著高于伪造证据（高出10至27个百分点），且误导信息虽在推理轨迹中被大量披露却难以被察觉。

    

    大语言模型如今能够以专家级水平回答医学问题。然而，这些系统所依据的上下文可能具有误导性，而误导性上下文会破坏模型的医学判断。为了理解误导性上下文如何破坏这一判断，我们考察了模型对上下文的易感性、对误导信息的披露程度、推理被破坏的机制以及决策的可监控性。在一个由临床医生审核、包含8,627个问题的问答基准MedMisBench的医学推理子集上，我们注入了两类误导性上下文线索：伪造的证据和纯粹的断言。我们测试了三个推理模型，其中两个会暴露其完整的推理轨迹，另一个前沿模型仅暴露其最终回复。所有三个模型都更容易受纯粹断言而非伪造证据的影响，采用断言答案的频率高出10至27个百分点。误导性线索在81%至98%的推理轨迹中被披露，但仅……

    arXiv:2609.02754v1 Announce Type: cross  Abstract: Large language models now answer medical questions with expert-level performance. However, the context these systems act on can be misleading, and misleading context can corrupt a model's medical judgment. To understand how misleading context corrupts this judgment, we examine the model's susceptibility to the context, disclosure of it, mechanism of corrupted reasoning, and monitorability of the decision. On the medical reasoning subset of MedMisBench, a clinician-reviewed question-answering benchmark of 8,627 questions, we inject two types of misleading context cues, fabricated evidence and a bare assertion. We test three reasoning models, two that expose their full reasoning trace and one frontier model that exposes only its response. All three are more susceptible to the assertion than to the fabricated evidence, adopting the asserted answer 10 to 27 points more often. The misleading cues are disclosed in 81 to 98% of traces but onl
    
[^219]: 从检测到特征刻画：日本X平台上“愤怒诱饵”的大规模研究

    From Detection to Characterization: A Large-Scale Study of Ragebait on Japanese X

    [https://arxiv.org/abs/2609.02262](https://arxiv.org/abs/2609.02262)

    本研究利用LLM辅助标注构建数据集并训练出日语愤怒诱饵检测集成分类器，首次对X平台日语帖子进行大规模分析，发现愤怒诱饵在政治、歧视、公共卫生和人际冲突等争议性话题中更为普遍。

    

    愤怒诱饵指的是故意设计用来激起愤怒或义愤，从而增加关注度和互动量的在线内容。然而，目前对愤怒诱饵的可靠大规模检测与系统性分析仍然有限，阻碍了人们理解其流行程度、影响及缓解措施的努力。本研究旨在开发一个有效的愤怒诱饵检测框架，并大规模地阐明愤怒诱饵的特征，为理解和缓解网络上具有情绪煽动性的内容提供基础。我们在大型语言模型（LLM）的辅助下构建了一个标注数据集，并训练了多个日语语言模型用于愤怒诱饵检测。随后，将由此得到的集成分类器应用于X平台上日语帖子的大规模数据集。我们的分析表明，愤怒诱饵在政治与社会争议性话题中更为普遍，包括政治、歧视、公共卫生和人际冲突等领域。

    arXiv:2609.02262v1 Announce Type: cross  Abstract: Ragebait refers to online content intentionally designed to provoke anger or outrage and thereby increase attention and engagement. However, reliable large-scale detection and systematic analysis of ragebait remain limited, hindering efforts to understand its prevalence, impact, and mitigation. This study aims to develop an effective ragebait detection framework and to clarify the characteristics of ragebait at scale, providing a basis for understanding and mitigating emotionally provocative content online. We constructed a labeled dataset with the assistance of a large language model (LLM) and trained several Japanese language models for ragebait detection. The resulting ensemble classifier was then applied to a large-scale dataset of Japanese-language posts on X. Our analysis shows that ragebait is more prevalent in politically and socially contentious topics, including politics, discrimination, public health, and interpersonal confl
    
[^220]: 可验证的灾害故事线与因果知识图谱：基于引用溯源的异构人道主义数据源流水线

    Verifiable Disaster Storylines and Causal Knowledge Graphs: A Citation-Grounded Pipeline from Heterogeneous Humanitarian Sources

    [https://arxiv.org/abs/2609.00858](https://arxiv.org/abs/2609.00858)

    该论文提出了一个基于检索增强生成（RAG）的流水线，融合EM-DAT结构化灾害记录与ReliefWeb、EMM非结构化文档，自动生成涵盖17个字段的灾害故事线和因果知识图谱，且每个节点和边均附带引用溯源，实现了对原始信息源的完全可追溯性，为人道主义响应提供可验证的态势感知支持。

    

    有效的人道主义响应依赖于对异构、海量信息源的快速综合——在危机爆发的关键早期阶段，这一任务常常超出人类分析能力的极限。我们提出了一个流水线，将来自EM-DAT的结构化灾害记录与来自ReliefWeb和欧洲媒体监测（EMM）的非结构化文档相结合，生成有来源依据的灾害故事线和因果知识图谱，为响应人员和分析人员提供态势感知支持。利用检索增强生成（RAG）技术，该流水线提取结构化故事线——即涵盖17个字段的表格化事件概况，内容包括灾害严重程度、关键驱动因素以及儿童敏感影响指标等——并构建因果知识图谱，其中每个节点和边都配有基于引用的解释性叙述，从而实现对原始来源的完全可追溯性。我们通过人工评估在三个不同的危机用例上对该系统进行了评估……

    arXiv:2609.00858v1 Announce Type: new  Abstract: Effective humanitarian response depends on the rapid synthesis of heterogeneous, high-volume information sources - a task that routinely exceeds human analytical capacity in the critical early hours of a crisis. We present a pipeline that combines structured disaster records from EM-DAT with unstructured documents from ReliefWeb and the European Media Monitor (EMM) to produce source-grounded disaster storylines and causal knowledge graphs supporting situational awareness for responders and analysts. Using Retrieval-Augmented Generation, the pipeline extracts structured storylines - tabular event profiles covering 17 fields, from severity and key drivers to child-sensitive impact indicators - and constructs causal knowledge graphs where each node and edge is enriched with citation-grounded explanatory narratives, enabling full traceability back to primary sources. We evaluate the system on three diverse crisis use cases through a human ev
    
[^221]: 负责任地将人工智能整合到癌症基因组学中：障碍、风险与可信临床转化的路径

    Responsible Integration of AI in Cancer Genomics: Barriers, Risks, and Pathways to Trustworthy Clinical Translation

    [https://arxiv.org/abs/2608.30912](https://arxiv.org/abs/2608.30912)

    本综述系统识别了阻碍AI在癌症基因组学临床转化的四大相互关联障碍（证据不一致、可解释性与不确定性、数据治理与可重复性、互操作性），并提出系统级概念框架以实现可信的临床整合。

    

    人工智能（AI）和自然语言处理（NLP）正被越来越多地用于提取、整合和解释与癌症基因组学相关的生物医学知识，但它们向常规临床肿瘤学的转化却相对缓慢。核心挑战不仅在于计算能力本身，更在于能否可信地整合到临床工作流程中。本综述考察了NLP和AI如何支持癌症基因组学流程，涵盖文献挖掘、自动化变异解读、临床试验匹配、知识图谱构建以及多模态数据整合。我们识别出四个相互关联的转化失败领域：证据不一致性、可解释性与不确定性、数据治理与可重复性，以及互操作性。我们没有孤立地看待这些挑战，而是采取系统级视角，关注它们在整个转化路径中的相互作用。我们提出了一个概念框架……

    arXiv:2608.30912v1 Announce Type: new  Abstract: Artificial intelligence (AI) and natural language processing (NLP) are increasingly used to extract, integrate, and interpret biomedical knowledge relevant to cancer genomics, yet their translation into routine clinical oncology has been comparatively slow. The central challenge is not computational capability alone, but trustworthy integration into clinical workflows. This review examines how NLP and AI support the cancer genomics pipeline, from literature mining and automated variant interpretation to clinical trial matching, knowledge graph construction, and multimodal data integration. We identify four interrelated translational failure domains: evidence inconsistency, explainability and uncertainty, data governance and reproducibility, and interoperability. Rather than considering these challenges in isolation, we take a systems-level view, focusing on their interaction across the translational pathway. We propose a conceptual frame
    
[^222]: VoiceCodeBench：评估自动语音识别中结构化标记的精确恢复

    VoiceCodeBench: Evaluating Exact Structured-Token Recovery in Automatic Speech Recognition

    [https://arxiv.org/abs/2608.28916](https://arxiv.org/abs/2608.28916)

    该论文提出VoiceCodeBench基准，利用300段职场录音中跨26种类型的1,482个规范实体，通过CTEM和TSR等新指标评估12个ASR系统对标识符、路径、数值等结构化标记的精确恢复能力，弥补了传统WER评估的不足。

    

    自动语音识别（ASR）系统通常以词错误率（WER）作为评估标准，然而许多语音工作流程依赖于标识符、路径和测量数值等需要精确书写的值。一份转录文本可能看起来流畅且WER很低，却破坏了下游系统必须解析、存储或执行的某个数值。我们提出了VoiceCodeBench，一个用于评估英语ASR中结构化标记精确恢复能力的基准。该基准包含300段人工录制的职场语音片段，涵盖8个工作流程领域，共1,482个经过审核的目标实体，分布在26种实体类型中，每个实体都具有可从音频中恢复的规范书面形式。在纯原始音频协议下，系统仅接收音频字节，不提供额外的上下文或元数据。除WER外，我们还评估了规范标记/实体匹配率（CTEM）、任务成功率（TSR）以及按类型划分的精确恢复率。在12个基线ASR系统上，较低的WER通常对应更好的结构化标记恢复表现。

    arXiv:2608.28916v1 Announce Type: new  Abstract: Automatic speech recognition (ASR) systems are commonly evaluated with word error rate (WER), yet many voice workflows depend on exact written values for identifiers, paths, and measured quantities. A transcript can appear fluent and achieve low WER while corrupting a value that a downstream system must parse, store, or execute.   We introduce VoiceCodeBench, a benchmark for evaluating exact structured-token recovery in English ASR. It contains 300 human-recorded workplace segments spanning eight workflow domains and 1,482 audited target entities across 26 entity types, each with a canonical written form recoverable from the audio. Under a raw-audio-only protocol, systems receive audio bytes without additional context or metadata. Alongside WER, we evaluate Canonical Token/Entity Match (CTEM), Task Success Rate (TSR), and per-type exact recovery.   Across 12 baseline ASR systems, lower WER generally corresponded to better structured-toke
    
[^223]: 一项针对高效LLM推理中周期性步骤层跳过方法的严格匹配审计：ConfLayers与SWIFT的对比，以及对训练式路由替代方案的补充分析

    A rigor-matched audit of periodic-step layer skipping for efficient llm inference: conflayers versus swift, with a supplemental analysis of trained routing alternatives

    [https://arxiv.org/abs/2608.28846](https://arxiv.org/abs/2608.28846)

    该研究通过严格匹配的多种子审计发现，SWIFT自推测解码在准确率上优于ConfLayers置信度门控提前退出方法，且在将在线搜索开销与纯推理成本分离后，SWIFT在所有实验设置下的真实推理速度均快5-21%，颠覆了表面上的速度排名。

    

    面向高效LLM推理的层跳过方法需要在某种粒度上决定针对给定输入执行哪些Transformer层。我们对两种周期性步骤、基于搜索的方法进行了严格匹配的三种子（three-seed）审计，这些方法在推理时在线做出该决策，并每隔几个生成步骤重新评估：一种是置信度门控的提前退出基线，另一种是真正的自推测解码（SWIFT，Xia等人2024），并与标准自回归解码进行比较，实验涵盖两个模型规模（Qwen2.5-0.5B和Qwen2.5-1.5B）和两个任务（GSM8K数学推理和CNN/DailyMail摘要）。SWIFT在四个实验组合中的三个上准确率最强；ConfLayers在所有组合中均处于劣势，在1.5B规模的GSM8K任务上差距尤为显著。一旦将在线搜索开销与纯推理成本分离，SWIFT的真实推理速度在所有四个实验组合中都比ConfLayers更快（5-21%），颠覆了朴素的挂钟时间排名。

    arXiv:2608.28846v1 Announce Type: cross  Abstract: Layer-skipping methods for efficient LLM inference decide, at some granularity, which transformer layers to execute for a given input. We present a rigor-matched, three-seed audit of two periodic-step, search-based methods that make this decision online at inference time and re-evaluate it every few generation steps: a confidence-gated early-exit baseline (ConfLayers) and genuine self-speculative decoding (SWIFT, Xia et al. 2024), together with vanilla autoregressive decoding, across two model scales (Qwen2.5-0.5B and Qwen2.5-1.5B) and two tasks (GSM8K reasoning and CNN/DailyMail summarization). SWIFT is the strongest method on accuracy in three of four cells; ConfLayers is dominated everywhere, with particularly large deficits on GSM8K at 1.5B. Once online-search overhead is separated from pure inference cost, SWIFT's true inference speed is faster than ConfLayers's in all four cells (5-21%), reversing the naive wall-clock ranking in 
    
[^224]: 面向已知任务音频大语言模型评估的生成式音频调用审计

    Auditing Generative Audio Calls for Known-Task Audio-LLM Evaluation

    [https://arxiv.org/abs/2608.27817](https://arxiv.org/abs/2608.27817)

    该论文将音频大语言模型的评估建模为受控的调用决策问题，发现在已知封闭集任务上，有监督编码器（如CLAP和WavLM）无需调用生成式音频模型即可取得接近最优的准确率，从而揭示了传统“波形提示对比ASR转录”的评估方式混淆了声学证据获取与生成模型调用这两个因素。

    

    语音和音频大语言模型通常通过比较波形提示是否优于自动语音识别（ASR）转录文本来进行评估。对于已知的封闭集任务，这种比较混淆了两个因素：获取声学证据的途径，以及调用生成式音频模型的需求。我们将这一区分评估为一个受控的调用决策问题。对于每个样本，一个策略可以在以下选项中做出选择：保留转录文本标签、使用来自对比语言-音频预训练（CLAP）、音频频谱图Transformer（AST）或WavLM的编码器证据，或调用Qwen2-Audio、Qwen2.5-Omni或MOSS-Audio；其中决定性的消融实验在保持选择器和开发协议不变的前提下移除所有生成式操作。在VocalSound数据集上，转录文本的准确率仅为0.296，说明确实需要波形信息。然而，有监督的CLAP和WavLM对照方法在完全不调用生成式音频模型的情况下分别达到了0.850和0.854的准确率。带有生成式操作的选择器在使用12.5%的调用预算的情况下达到了0.925的准确率（摘要在此处截断）。

    arXiv:2608.27817v1 Announce Type: cross  Abstract: Speech and audio LLMs are often evaluated by asking whether a waveform prompt beats an automatic speech recognition (ASR) transcript. For known closed-set tasks, that comparison conflates two factors: access to acoustic evidence and the need to call a generative audio model. We evaluate this distinction as a controlled call-decision problem. For each example, a policy chooses among keeping a transcript label, using encoder evidence from Contrastive Language-Audio Pretraining (CLAP), Audio Spectrogram Transformer (AST), or WavLM, and calling Qwen2-Audio, Qwen2.5-Omni, or MOSS-Audio; the decisive ablation removes all generative actions while keeping the selector and development protocol fixed. On VocalSound, transcripts reach 0.296 accuracy, so waveform information is needed. Yet supervised CLAP and WavLM controls reach 0.850 and 0.854 with no generative audio calls. A selector with generative actions reaches 0.925 accuracy using 12.5% c
    
[^225]: SURE-Challenge：在语音大模型生成之前评估语音证据

    SURE-Challenge: Evaluating Speech Evidence Before Speech-LLM Generation

    [https://arxiv.org/abs/2608.27783](https://arxiv.org/abs/2608.27783)

    该论文提出 SURE-Challenge 基准，用于评估语音大模型在生成回答之前对不支持输入（静音、噪声、合成音调、嘈杂语音）的拒绝能力，并证明一个简单的“能量加 Whisper 分数”规则可将不支持输入的拒绝数从 15/204 提升至 196/204，同时不损失有效输入的准确率。

    

    语音大模型（Speech LLMs）通常在其作出回答之后才被评估，尽管操作系统首先必须决定是否应将音频波形发送给模型。我们为此准入步骤定义了“语音不支持拒绝评估挑战”（Speech-Unsupported Rejection Evaluation Challenge，简称 SURE-Challenge）。该基准将源自 LibriSpeech 的转录和首词问答任务与不支持的输入——静音、有色噪声、合成音调以及来源模糊的嘈杂语音——配对，并采用互不相交的来源划分以防止泄漏。前端消融实验使用 Qwen2-Audio；随后将选定的“能量加 Whisper 分数”规则在六个语音/音频大模型之前进行重放验证。在经过泄漏筛查的 474 行 SURE-Extended 测试集上，原始 Qwen2-Audio 仅拒绝 204 个不支持输入中的 15 个，而固定规则可拒绝其中 196 个，且支持样本的准确率保持不变。外部检查界定了这一数字的边界：随着 Whisper 分数阈值的收紧，Common Voice 的保留率随之下降，而变速嘈杂语音在 54 个片段中仅产生 18 到 24 个被拒绝的片段。

    arXiv:2608.27783v1 Announce Type: cross  Abstract: Speech LLMs are usually graded after they answer, although an operating system first has to decide whether a waveform should be sent to the model. We define the Speech-Unsupported Rejection Evaluation Challenge (SURE-Challenge) for this admission step. The benchmark pairs LibriSpeech-derived transcription and first-word question answering with unsupported silence, colored noise, synthetic tones, and source-ambiguous babble under disjoint source splits. Front-end ablations use Qwen2-Audio; the selected energy-plus-Whisper-score rule is then replayed before six speech/audio LLMs. On the 474-row leakage-screened SURE-Extended test set, raw Qwen2-Audio rejects 15/204 unsupported inputs, whereas the fixed rule rejects 196/204 and leaves supported accuracy unchanged. External checks delimit this number: Common Voice retention drops as the Whisper-score threshold is tightened, and no-speed babble gives 18 to 24 rejected clips out of 54 across
    
[^226]: 当规范补全出错时：形式化并度量大型语言模型中的跳跃

    When the Canonical Completion Is Wrong: Formalizing and Measuring the Jump in Large Language Models

    [https://arxiv.org/abs/2608.26187](https://arxiv.org/abs/2608.26187)

    本文首次形式化了LLM中的“跳跃”概念，将其定义为具有可验证证书的有限扩展问题，并提供了度量方法，以解决关于LLM溯因能力的争论。

    

    arXiv:2608.26187v1 公告类型：交叉 摘要：大型语言模型（LLMs）是否能够执行从证据到新公理系统的溯因跳跃（通常称为“跳跃”），近来引起了相当大的争论。一种主流观点认为LLMs在结构上无法进行此类跳跃，而近期研究则对其机制和证据提出了质疑。然而，这一争论仍难以定论，因为该领域缺乏对跳跃的形式化定义以及测试双方观点的度量方法。在本文中，我们分四步开发了跳跃的形式化描述，并度量了第二步。这些步骤询问部分数据的默认补全是什么、何时被迫放弃它、何时放弃是正确的，以及连续跳跃如何复合。具体来说，我们将跳跃实例定义为有限扩展问题，并带有机器可检查的证书，证明正确的补全存在、在重命名意义下唯一，且与数据的规范补全不同。

    arXiv:2608.26187v1 Announce Type: cross  Abstract: Whether large language models (LLMs) can perform the abductive leap from evidence to a new system of axioms, commonly referred to as a jump, has recently attracted considerable debate. A prominent position holds that LLMs are structurally incapable of such jumps, while recent studies challenge both its mechanism and its evidence. However, the debate remains difficult to settle, since the field still lacks a formal definition of the jump and a measure to test either side. In this paper, we develop a formal account of the jump in four steps and measure the second. The steps ask what the default completion of partial data is, when abandoning it is forced, when the abandonment is correct, and how successive jumps compound. Specifically, we define a jump instance as a finite extension problem with a machine-checked certificate that a correct completion exists, is unique up to renaming, and differs from the canonical completion of the data. 
    
[^227]: 问题在哪里？评估视觉语言模型中的时间一致性

    What's the Catch? Evaluating Temporal Consistency in Vision-Language Models

    [https://arxiv.org/abs/2608.23474](https://arxiv.org/abs/2608.23474)

    本研究通过时间异常检测任务揭示了视觉语言模型在时间一致性理解上的显著不足，与人类表现存在明显差距。

    

    视觉语言模型（VLMs）在视频和图像序列基准测试中表现出色，但它们是否真正捕捉了时间结构仍不清楚。为研究此问题，我们将时间定位表述为异常检测问题，提供了一种简单且受控的评估方法，直接测试对时间一致性的敏感性。我们引入了TimeCatch，其中时间异常通过交换连续帧创建，帧级异常则通过用高斯噪声替换一帧来创建。模型在四个合成和真实世界数据集上进行了异常检测和定位任务的评估，并伴随一项人类研究。我们的评估揭示了帧级和时间异常检测之间的显著差距。尽管VLMs能一致地检测帧级异常并通常能准确定位它们，但在时间异常检测上它们表现接近随机水平，定位能力仅略高于随机。相比之下，人类在这些任务上表现更佳。

    arXiv:2608.23474v1 Announce Type: cross  Abstract: Vision-language models (VLMs) achieve strong performance on video and image-sequence benchmarks, yet it remains unclear whether they capture temporal structure. To study this question, we formulate temporal grounding as an anomaly detection problem, providing a simple and controlled evaluation that directly tests sensitivity to temporal consistency. We introduce TimeCatch, where temporal anomalies are created by swapping consecutive frames and frame-level anomalies by replacing a frame with Gaussian noise. Models are evaluated on anomaly detection and localization tasks across four synthetic and real-world datasets, alongside a human study. Our evaluation reveals a substantial gap between frame-level and temporal anomaly detection. While VLMs consistently detect frame-level anomalies and often localize them accurately, they perform near chance on temporal anomaly detection and only modestly above chance on localization. Humans, in cont
    
[^228]: KREL：基于临床证据的知识引导推理与大型语言模型的自动医疗编码

    KREL: Automatic Medical Coding via Knowledge-Guided Reasoning over Clinical Evidence with LLMs

    [https://arxiv.org/abs/2608.20887](https://arxiv.org/abs/2608.20887)

    该论文提出了KREL框架，通过结合大型语言模型的推理能力与外部ICD编码指南的结构化知识，解决了自动医疗编码中临床记录过长、标签空间庞大及编码规则复杂等关键问题。

    

    自动医疗编码（AMC）将标准化的国际疾病分类（ICD）代码分配给临床记录，对于医疗报销、质量报告和临床研究至关重要。现有的基于预训练语言模型（PLM）的方法通常将AMC视为对预定义代码集的极端多标签分类问题，而近期基于大型语言模型（LLM）的方法则将其构建为生成或多步推理任务。然而，关键挑战仍然存在，包括临床记录的极端长度阻碍了有效解释、庞大的ICD标签空间，以及LLM未能明确捕获的复杂编码规则。在这项工作中，我们提出了KREL（Knowledge-Guided Reasoning over Clinical Evidence with LLMs），一个利用LLM进行临床文本理解和推理，同时整合外部ICD编码指南作为结构化知识的框架。这种设计旨在应对上述挑战。

    arXiv:2608.20887v1 Announce Type: cross  Abstract: Automatic Medical Coding (AMC), which assigns standardized International Classification of Diseases (ICD) codes to clinical notes, is essential for medical reimbursement, quality reporting, and clinical research. Existing pre-trained language model (PLM)-based methods typically formulate AMC as an extreme multi-label classification problem over a predefined code set, while recent large language model (LLM)-based approaches instead frame it as generation or multi-step reasoning. However, key challenges remain, including the extreme length of clinical notes that hinders effective interpretation, the vast ICD label space, and complex coding rules that are not explicitly captured by LLMs. In this work, we propose Knowledge-Guided Reasoning over Clinical Evidence with LLMs (KREL), a framework that leverages LLMs for clinical text understanding and reasoning while integrating external ICD coding guidelines as structured knowledge. This desig
    
[^229]: 可读、忠实、被使用：语言模型中人口统计身份的三种可分离属性

    Readable, Faithful, Used: Three Dissociable Properties of Demographic Identity in a Language Model

    [https://arxiv.org/abs/2608.18768](https://arxiv.org/abs/2608.18768)

    本研究发现语言模型中人口统计身份的可读性、忠实性和使用性可分离，注意力头读出比标准方法更忠实，且单一头可跨属性保持高保真，但种族相关身份较弱。

    

    arXiv:2608.18768v1 公告类型：新 摘要：大型语言模型被广泛用于模拟调查受访者，但其回答是同质的，且对真实群体间差异不忠实。我们探究人口统计群体身份在LLM内部位于何处，其几何结构如何忠实地反映真实群体间的意见结构，以及模型是否使用其所编码的信息。通过对169个人口统计单元格使用表征相似性分析与皮尤地面真值对比，我们对Mistral-7B中的1,089个读出位置进行评分，并对六种属性类型进行因果干预。四项结果：（1）标准的最后令牌残差读出低估了模型：注意力头读出在六种类型中的五种中占主导地位，经选择校正的保真度高达rho=0.63——约为测量可靠性上限的70%——且能通过词汇相似性控制。（2）单个头（L11 H16）作为固定位置在全部六种类型中显著忠实，而基于种族的类型仍然较弱且对提示敏感。

    arXiv:2608.18768v1 Announce Type: new  Abstract: Large language models are widely used to simulate survey respondents, yet their answers are homogeneous and unfaithful to real inter-group differences. We ask where demographic group identity lives inside an LLM, how faithfully its geometry mirrors real inter-group opinion structure, and whether it uses what it encodes. Using representational similarity analysis against Pew ground truth over 169 demographic cells, we score 1,089 read-out locations in Mistral-7B and intervene causally across six attribute types. Four results. (1) The standard last-token residual read-out understates the model: attention-head read-outs dominate it in five of six types, with selection-corrected fidelity up to rho=0.63 -- roughly 70% of the measurement-reliability ceiling -- surviving a lexical-similarity control. (2) A single head (L11 H16) is significantly faithful in all six types as a fixed location, while race-based types stay weak and prompt-fragile. B
    
[^230]: IndicQE-APE：面向印度语言的翻译质量评估与自动后编辑基准

    IndicQE-APE: A Benchmark for Quality Estimation and Automatic Post-Editing for Indic Languages

    [https://arxiv.org/abs/2608.16344](https://arxiv.org/abs/2608.16344)

    本文整合了印度语言的质量评估和自动后编辑数据，创建了一个包含多标签和难度分层的基准，并评估了多种模型，发现只有同时利用整体和词级信息的系统在严格对照下表现显著。

    

    摘要：arXiv:2608.16344v1 公告类型：新 摘要：印度语言的质量评估（QE）和自动后编辑（APE）数据分散在不同的发布中，因此没有单一资源能够在同一基础上支持跨任务和语言对的训练与评估。我们将WMT 2020--2024共享任务系列与扩展的英语--马拉雅拉姆语资源整合为\indicqe：包含126,754个实例，覆盖九个方向性语言对，每个片段上最多有四种标签类型对齐，包括直接评估、人工后编辑、词级OK/BAD标签和错误解释，以及按四个难度轴分层的测试集。在此基准上，我们评估了六个提示的大型语言模型和三个COMET指标在片段级QE上的表现，以及三个系统在APE上的表现。其中两个轴部分基于直接评估并选择其压缩子集，因此每个轴与从相同语言对和相同分数分布中抽取的对照组进行比较。只有一种方法在对照组中幸存：整体和词级对齐的片段。

    arXiv:2608.16344v1 Announce Type: new  Abstract: Indic quality estimation (QE) and automatic post-editing (APE) data is spread across separate releases, so no single resource supports training and evaluation across tasks and language pairs on one footing. We consolidate the WMT 2020--2024 shared-task lineage with an extended English--Malayalam resource into \indicqe: $126{,}754$ instances over nine directional pairs, with up to four label types aligned on the same segment, a direct assessment, a human post-edit, word-level OK/BAD tags and an error explanation, and a test set stratified over four difficulty axes. On it, we benchmark six prompted LLMs and three COMET metrics on segment-level QE, and three systems on APE. Two of the axes are defined partly on the direct assessment and select a compressed slice of it, so each axis is compared against a control drawn from the same language pair with the same score distribution. Only one survives that control: segments whose holistic and tok
    
[^231]: TaoLive数字人代理技术报告：训练代理与其操控系统共同进化

    TaoLive Digital Avatar Agent Technical Report: Training Agents to Evolve with Their Harness

    [https://arxiv.org/abs/2608.15763](https://arxiv.org/abs/2608.15763)

    本文提出操控系统感知训练（HAT）方法，通过将可进化的操控系统状态纳入训练分布，使数字人代理在实时直播中既能快速响应又能灵活适应动态策略变化。

    

    在直播电商中，AI驱动的数字人主播必须实时回答产品问题、吸引观众并执行不断变化的商业策略。这要求低延迟、事实准确且有效的回复，以及对更新后的活动、合规和风格要求的快速适应。我们开发了一个可进化的操控系统（Harness），将技能（Skills）、钩子（Hooks）、系统提示和工具与模型权重解耦，使得运行时行为无需重新训练即可改变。然而，操控系统的进化创造了一个动态执行环境：在单一配置上微调的紧凑模型可能会记忆名称、模式和提示模板，而不是遵循当前提供的操控系统，而更强的零样本模型又因速度过慢而无法满足实时使用需求。我们通过操控系统感知训练（HAT）来解决这一矛盾，该方法将操控系统状态纳入训练分布。HAT对技能、工具模式和提示应用了任务保持的操控系统状态增强（HSA）。

    arXiv:2608.15763v1 Announce Type: new  Abstract: AI-powered digital-avatar streamers in live e-commerce must answer product questions, engage viewers, and execute changing business strategies in real time. This requires low latency, factual and effective replies, and rapid adaptation to updated campaign, compliance, and style requirements. We develop an evolvable Harness that decouples Skills, Hooks, system prompts, and tools from model weights, allowing runtime behavior to change without retraining. However, Harness evolution creates a moving execution environment: compact models fine-tuned on one configuration may memorize names, schemas, and prompt templates rather than follow the Harness currently provided, while stronger zero-shot models are too slow for real-time use. We address this tension with Harness-Aware Training (HAT), which makes Harness states part of the training distribution. HAT applies task-preserving Harness-State Augmentation (HSA) to Skills, tool schemas, prompt s
    
[^232]: 基于人格特征的涌现性失调数据归因

    Data Attribution of Emergent Misalignment with Persona Features

    [https://arxiv.org/abs/2608.11025](https://arxiv.org/abs/2608.11025)

    本研究通过稀疏自编码器模型差异分析发现，涌现性失调源于预训练中被失调微调放大的人格特征，仅通过引导这些特征即可在已对齐模型中诱导高达62%的失调或使失调模型重新对齐。

    

    涌现性失调是指在语言模型上针对狭窄任务进行微调后，导致其在无关领域产生有害行为的现象。一个主流的机制性解释将涌现性失调归因于人格特征：即预训练期间习得的潜在方向，而失调微调会放大这些方向。我们探究了这些特征的来源：哪些预训练文档激活了它们，以及自然存在的人类撰写文本是否足以诱发涌现性失调。通过对四个开源权重模型使用基于稀疏自编码器（SAE）的模型差异分析，我们发现与越狱人格、讽刺、欺骗和操纵相关的特征会被失调微调放大，而安全相关和助手身份特征则被抑制。引导单个特征可以双向控制涌现性失调：在已对齐的模型中可诱导高达62%的失调率——超过失调微调本身达到的35%——并能使失调的模型重新对齐。

    arXiv:2608.11025v2 Announce Type: replace  Abstract: Emergent misalignment (EM) is the phenomenon where fine-tuning a language model on a narrow task leads to harmful behavior in unrelated domains. A leading mechanistic account attributes EM to persona features: latent directions acquired during pre-training that misaligned fine-tuning amplifies. We ask where these features come from: which pre-training documents activate them, and whether naturally occurring human-written text suffices to induce EM. Using Sparse Autoencoder (SAE) based model diffing across four open-weight models, we find that features related to jailbreak personas, sarcasm, deception, and manipulation are amplified by misalignment fine-tuning, while safety-relevant and assistant-identity features are suppressed. Steering individual features controls EM in both directions: it induces misalignment rates of up to 62% in aligned models -- exceeding the 35% reached by misalignment fine-tuning itself -- and re-aligns misal
    
[^233]: 何时可以区分一般因子？非比例性、稳定结构与双因子决策

    When Is a General Factor Distinguishable? Non-Proportionality, Stable Structure, and the Bifactor Decision

    [https://arxiv.org/abs/2608.10731](https://arxiv.org/abs/2608.10731)

    本文提出了判断双因子模型与相关一阶因子模型何时可区分的理论条件（关键在于簇内载荷的非比例性），并开发了一个两步程序来稳定地确定一阶结构，进而有条件地比较斜交与双因子两种表示。

    

    在相关的一阶因子之外是否还需要增加一般维度，这是总体协方差的属性，而非估计量的属性。当一般载荷和组载荷在每个簇内成比例时，双因子结构与相关因子结构在协方差上是等价的。当每个簇都违反比例性时，每簇至少三个指标加上温和的条件即可排除由具有对角独特性的K因子模型进行精确复现的可能性。混合配置目前仅被部分刻画，由此引出了分级可区分性的概念。我们开发了一个两步程序，仅当一阶结构在直接的更宽因子数比较中持续存在时才提供稳定的一阶结构，然后在该结构交付的条件下比较斜交与双因子两种表示。一项初步模拟研究提供了试探性的设计证据。一项主要研究开发了三个非嵌套的持续性特征曲线，并将.80/r2指定为实用默认标准。

    arXiv:2608.10731v2 Announce Type: replace-cross  Abstract: Whether an added general dimension is necessary beyond correlated first-order factors is a property of population covariance, not an estimator. A bifactor structure is covariance-equivalent to correlated factors when general and group loadings are proportional within every cluster. When every cluster violates proportionality, at least three indicators per cluster and mild conditions rule out exact reproduction by a K-factor model with diagonal uniquenesses. Mixed configurations remain only partly characterized, motivating graded distinguishability. We develop a two-step procedure that delivers a stable first-order structure only when it persists across direct wider-count comparisons, then compares oblique and bifactor representations conditionally on delivery. A preliminary simulation study provides tentative design evidence. A major study develops three non-nested persistence profiles, designates .80/r2 as the practical defaul
    
[^234]: MameLoshnLM：意第绪语语言模型与评估基准

    MameLoshnLM: Yiddish Language Model and Evaluation Benchmark

    [https://arxiv.org/abs/2608.05850](https://arxiv.org/abs/2608.05850)

    该论文发布了首个开源的80亿参数意第绪语语言模型MameLoshnLM，通过构建高质量预训练语料库Oytser和多任务评估基准Kashes，解决了意第绪语数字资源稀缺与评估数据不可靠的问题，并在基准任务上超越同等规模的开源基线模型。

    

    我们提出了MameLoshnLM，这是首个专门为意第绪语构建的开源80亿参数语言模型。尽管意第绪语拥有丰富的文本传统，但其有限的数字化存在以及可靠评估资源的匮乏，一直制约着意第绪语语言建模的发展。现有的多语言语料库和基准测试往往不能很好地代表这门语言，其中包含大量嘈杂的、机器翻译的以及错误分类的文本。我们通过引入Oytser——一个结合当代网络原生资源与文学材料的高质量意第绪语预训练语料库，以及Kashes——一个涵盖翻译、语言学分析、信息提取和语言理解的多任务基准——来弥补这些缺口。利用这些资源，我们对Llama 3.1 8B进行继续预训练，得到了MameLoshnLM。在基准测试的各项任务中，MameLoshnLM的表现优于同等规模的开源基线模型。

    arXiv:2608.05850v2 Announce Type: replace-cross  Abstract: We present MameLoshnLM, the first open-source 8B-parameter language model built specifically for Yiddish. Despite Yiddish's rich textual tradition, its limited digital presence and the scarcity of reliable evaluation resources have constrained progress in Yiddish language modeling. Existing multilingual corpora and benchmarks are often poor proxies for the language, containing substantial amounts of noisy, machine-translated, and misclassified text. We address these gaps by introducing Oytser, a high-quality Yiddish pretraining corpus that combines contemporary web-native sources with literary materials, and Kashes, a multi-task benchmark spanning translation, linguistic analysis, information extraction, and language understanding. Using these resources, we continue pretraining Llama 3.1 8B to obtain MameLoshnLM. Across the tasks in the benchmark, MameLoshnLM outperforms open baselines of similar scale. Our analyses show that t
    
[^235]: ChartAnno：面向图表标注生成的多模态大语言模型基准测试

    ChartAnno: Benchmarking Multimodal Large Language Models for Chart Annotation Generation

    [https://arxiv.org/abs/2608.03464](https://arxiv.org/abs/2608.03464)

    本文提出了ChartAnno基准，包含1,200个真实图表、3,600条多层级标注指令及结合规则与LLM评判的多维评估框架，系统评估了多模态大语言模型在图表标注生成任务上的能力。

    

    标注对于交流性可视化至关重要，它有助于解释数据、强调关键发现并引导读者注意力。虽然多模态大语言模型为自动生成图表标注提供了新的机会，但它们在这一任务中的能力仍未得到充分探索。为了填补这一空白，我们提出了ChartAnno，一个用于评估多模态大语言模型图表标注生成能力的综合基准。ChartAnno包含1,200个真实世界的图表，配有成对的带标注和不带标注的可执行代码，以及跨越三个具体化程度的3,600条标注指令。我们还开发了一个多维评估框架，结合基于规则的指标和LLM评判的指标，以评估执行效果、结构合规性、语义一致性和设计有效性。我们在两种主要的图表输入设置下评估了10个具有代表性的多模态大语言模型：(1) 仅图表代码；(2) 代码和图表图像相结合。结果表明，专有……（摘要不完整）

    arXiv:2608.03464v2 Announce Type: replace  Abstract: Annotations are essential to communicative visualization, helping explain data, emphasize key findings, and guide attention. While multimodal large language models (MLLMs) offer new opportunities for automatic chart annotation authoring, their capabilities in this task remain underexplored. To address this gap, we introduce ChartAnno, a comprehensive benchmark for evaluating MLLMs on chart annotation generation. ChartAnno contains 1,200 real-world charts with paired annotated and unannotated executable code, along with 3,600 annotation instructions spanning three levels of specificity. We also develop a multidimensional evaluation framework combining rule-based and LLM-judged metrics to assess execution, structural compliance, semantic consistency, and design effectiveness. We evaluate 10 representative MLLMs under two primary chart input settings: (1) chart code alone and (2) both code and chart image. Results reveal that proprietar
    
[^236]: 人工智能代理能否模拟A/B测试结果？代理实验的验证框架

    Can AI Agents Simulate A/B Test Outcomes? A Validation Framework for Agentic Experimentation

    [https://arxiv.org/abs/2608.02345](https://arxiv.org/abs/2608.02345)

    本文提出一种代理无关的模拟随机对照试验框架，通过两层误差分解验证AI代理能否准确模拟A/B测试结果，以在真实流量前筛选干预方案。

    

    arXiv:2608.02345v2 公告类型：替换-交叉 摘要：A/B测试仍是科技行业推出新功能的标准方法。然而，每项实验都会消耗实际流量、工程精力和数周的实时时间。人工智能代理——在行为特征和干预措施的情境描述条件下——能否足够准确地模拟结果，以在投入实时流量前筛选候选治疗方案？我们将此问题正式化为“模拟随机对照试验”（S-RCT），并推导出一个两层误差分解，将代理近似误差与子采样误差分离，从而实现对每部分的针对性改进。该框架与代理无关：任何行为模型——从微调专家到通用基础模型——都可作为模拟引擎。在67项历史营销A/B测试中验证，使用现成基础模型的基线S-RCT捕捉到了方向性信号（符号重叠度0.70），但系统性偏差。

    arXiv:2608.02345v2 Announce Type: replace-cross  Abstract: A/B testing remains the standard for rolling out new features in the technology industry. Each experiment, however, consumes real traffic, engineering effort, and weeks of wall-clock time. Can AI agents---conditioned on behavioral profiles and contextual descriptions of the intervention---simulate outcomes accurately enough to vet candidate treatments before committing live traffic? We formalize this question as a \emph{Simulated Randomized Controlled Trial} (S-RCT) and derive a two-layer error decomposition that separates agent approximation error from subsampling error, enabling targeted improvements to each. The framework is agent-agnostic: any behavioral model---from a fine-tuned specialist to a general-purpose foundation model---can serve as the simulation engine. Validated on 67 historical marketing A/B tests, a baseline S-RCT using an off-the-shelf foundation model captures directional signal (sign overlap 0.70) but syst
    
[^237]: FriendBench：人类与多模态大语言模型的二人熟悉度推断基准测试

    FriendBench: Benchmarking Dyadic Familiarity Inference in Humans and Multimodal Large Language Models

    [https://arxiv.org/abs/2607.29602](https://arxiv.org/abs/2607.29602)

    该论文提出FriendBench基准，用于评估人类和多模态大语言模型从短对话片段中推断两人是熟人还是陌生人的能力，发现最佳模型的准确率已与人类相当但存在先验偏好差异，且只有人类能利用言语之外的可见行为进一步提升判断。

    

    解读社交情境往往取决于行为，而不仅仅是语言。我们提出了FriendBench，这是一个用于从20秒的两人破冰对话片段中推断两个人是已经熟悉还是初次见面的基准测试。每对参与者回答相同类型的提示，因此只有互动的方式才能揭示答案。在文本、音频和视频三种模态上，我们将来自七家公司的26个模型与匹配的人类评审组在96个平衡的二人组上进行了比较。最佳模型与人类群体在每个模态的准确率上均无统计学差异，但两者的达成方式不同：人类在两个答案之间保持平衡，而最强的模型则倾向于判断为“陌生人”。这是一种有效先验上的差异，而非辨别能力的差异。更丰富的信息渠道对两者的提升并不均等，只有人类能够从言语之外的可见行为中获得额外收益。我们公开发布了实验刺激材料、人类评分和模型预测结果。

    arXiv:2607.29602v2 Announce Type: replace-cross  Abstract: Reading a social situation often depends on behavior, not words alone. We introduce FriendBench, a benchmark for inferring whether two people are already familiar or are meeting as strangers, from a 20-second clip of a dyadic ice-breaker conversation. Every pair answers the same type of prompt, so only the manner of interaction can reveal the answer. Across text, audio, and video, we compare 26 models from seven companies against matched human panels over 96 balanced dyads. The best model and the human crowd are statistically indistinguishable on accuracy in every modality, but reach it differently: humans stay balanced across the two answers, while the strongest models favor ``stranger.'' This is a difference in effective prior, not in discrimination. Richer channels help both unequally, and only humans gain from visible behavior on top of speech. We release the stimuli, human ratings, and model predictions.
    
[^238]: 研究机器翻译中高效推理部署的量化权衡

    Studying quantization trade-offs for efficient inference deployment in machine translation

    [https://arxiv.org/abs/2607.29397](https://arxiv.org/abs/2607.29397)

    本文系统研究了EuroLLM在1.7B至22B三个模型规模上的量化权衡，发现结合文档分块策略与W4A8或W8A8量化可显著改善单GPU部署下的延迟-吞吐量帕累托曲线，并提出了基于DocHPLT的文档级机器翻译评估方法。

    

    在真实的服务器环境中部署大型语言模型面临诸多挑战，因为系统需要在保证低延迟的同时提供高质量的响应。量化是减少内存占用和提高推理效率的常用方法，然而其在受控的、编排级工作负载下对延迟和吞吐量的影响却很少被评估。在这项工作中，我们研究了EuroLLM在三个模型规模（从1.7B到22B）上的量化权衡，以实现在单个A100或H100 GPU上的高效部署。我们证明，将文档分块策略与W4A8或W8A8量化相结合，可以在广泛的工作负载下改善延迟-吞吐量帕累托曲线。此外，由于标准机器翻译（MT）基准依赖于孤立的句子，无法捕捉长上下文动态，我们引入了基于DocHPLT的文档级评估方法来评估……

    arXiv:2607.29397v3 Announce Type: replace  Abstract: Deploying large language models in realistic server environments poses challenges, as the system needs to provide high-quality responses with low latency. Quantization is a common approach to reduce the memory footprint and improve inference efficiency, yet its impact on latency and throughput is rarely evaluated under controlled, orchestration-level workloads. In this work we study the quantization trade-offs of EuroLLM \citep{martins2025eurollm} across three model sizes ranging from 1.7B to 22B for efficient deployment on a single A100 or H100 GPU. We demonstrate that combining a document-chunking strategy with W4A8 or W8A8 quantization improves the latency-throughput Pareto-curve under a wide range of workloads. Furthermore, since standard machine translation (MT) benchmarks rely on isolated sentences and fail to capture long-context dynamics, we introduce a document-level evaluation based on DocHPLT \cite{o2025dochplt} to assess 
    
[^239]: FairFund-Bench：评估大语言模型资源分配中的分配偏差

    FairFund-Bench: Evaluating Distributive Bias in LLM Resource Allocation

    [https://arxiv.org/abs/2607.28934](https://arxiv.org/abs/2607.28934)

    本文提出FairFund-Bench基准，通过系统性地变化审计格式（评估任务、比较情境和透明度），揭示了以往LLM资源分配偏见审计结果不一致的原因在于审计格式本身的差异。

    

    大语言模型（LLM）越来越多地参与到稀缺资源的分配中，这引发了人们对其基于种族和性别等特征产生分配偏差的担忧。然而，近期针对LLM的审计得出了不一致的结果，即使针对同一模型，也同时发现了对女性和少数族裔的积极与消极歧视的证据。我们证明这种分歧可能源于审计格式上的差异，并介绍了FairFund-Bench——一个系统性地改变以往审计设计关键特征的基准：评估任务（评分、排名或分配）、比较情境（单一或多刺激），以及审计是透明的还是伪装的。该基准包含600份英文经济援助请求，这些请求由人工撰写的模板生成（并基于130万个真实的GoFundMe众筹活动进行校准），涵盖三个领域、四个种族类别和两个性别类别。

    arXiv:2607.28934v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly involved in the distribution of scarce resources, raising concerns about biased allocations based on characteristics like race and gender. Recent LLM audits have produced inconsistent results, however, finding evidence of both positive and negative discrimination towards women and ethnic minorities, even for the same models. We show that this disagreement can arise from differences in audit format and introduce FairFund-Bench, a benchmark that systematically varies key features of previous audit designs: the evaluation task (rating, ranking, or allocation), comparison context (single or multi-stimulus), and whether the audit is transparent or disguised. The benchmark comprises 600 English-language requests for financial assistance created from human-authored templates (calibrated against 1.3M real GoFundMe campaigns) across three domains, four race and two gender categories, and fiv
    
[^240]: 减法、传输还是重放？语言模型记忆中的可审计删除

    Subtract, Transport, or Replay? Auditable Deletion from Language-Model Memory

    [https://arxiv.org/abs/2607.27539](https://arxiv.org/abs/2607.27539)

    本文提出并比较了三种从语言模型记忆中删除记录的方法，发现原生删除不可靠，而检查点重放能实现可审计的精确删除，并展示了在冻结模型上无需额外训练即可构建可删除记忆的可行性。

    

    从持久化语言模型记忆中精确删除取决于记录的效果在后续计算后是否仍可寻址。原生Kimi Delta Attention (KDA)在测试的收据接口上给出了负面结果：语料库汇总的原始循环贡献随后缀变化12-49%，且在衰减账本修正后仍保持8-49%的差异。原生省略也改变了后续的转换项、写入项和其他活动缓存。冻结输入传输在其固定输入控制上成功；更改的项将原生省略置于测试的收据类别之外。检查点重放提供了评估的重新计算路径；最终logits上的零残差和所有80个审计的KDA数组验证了在声明的检查点表面上的恢复。补充结果是建设性的。我们将支持向量记忆改造为冻结的Gemma 3，无需注意力转移、低秩恢复、蒸馏、适配器或语言模型修改。

    arXiv:2607.27539v2 Announce Type: replace-cross  Abstract: Exact deletion from persistent language-model memory depends on whether a record's effect remains addressable after later computation. Native Kimi Delta Attention (KDA) gives a negative result for the tested receipt interface: the corpus-pooled raw recurrent contribution changes by 12-49% with the suffix and remains 8-49% after a decay-ledger correction. Native omission also changes later transition and write terms and other active caches. Frozen-input transport succeeds on its fixed-input control; the changed terms place native omission outside the tested receipt classes. Checkpoint replay supplies the evaluated recomputation path; zero residual on final logits and all 80 audited KDA arrays verifies restoration across the declared checkpoint surface.   The complementary result is constructive. We retrofit support-vector memory into frozen Gemma 3 without attention transfer, low-rank recovery, distillation, adapters, or languag
    
[^241]: 循环不等于可靠性：面向智能体代码修复的状态绑定证据与类型化修订契约

    Looping Is Not Reliability: State-Bound Evidence and Typed Revision Contracts for Agentic Code Repair

    [https://arxiv.org/abs/2607.24604](https://arxiv.org/abs/2607.24604)

    该论文通过受控实验证明编码智能体中“生成—测试—修订”循环的重复并不能保证可靠性——陈旧轨迹会显著损害已经正确的补丁——并提出用状态绑定证据与类型化修订契约来确保补丁的保留、验证与提交。

    

    生成—测试—修订循环在编码智能体中十分常见，但仅靠重复本身并不能提供可靠性保证。我们研究了“找到正确补丁”与“保留、验证并提交该补丁”之间的差距。一项在30个HumanEval修复任务上进行的密封五种子研究产生了900条三次修订轨迹。在强制修订条件下，使用当前轨迹的当前正确率从一次修订后的0.820降至两次修订后的0.673，尽管“曾经正确”的比例上升至0.847。两项共同状态研究使用来自相同冻结程序的2,430个分支来消除处理后的风险集偏差。在一项预先设定的14B模型复制实验中，陈旧轨迹对34/135个正确起点造成损害，而当前轨迹仅损害4/135个，增加了22.2个百分点（任务聚类95%置信区间[8.9, 37.0]，精确Holm检验 p=0.0337）。一项前瞻性的540次rollout策略消除了观察到的正确起点损害，但降低了错误起点的修复能力，且未能通过联合判定标准。在24个错误及后续仓库实验中……（摘要原文在此处截断）

    arXiv:2607.24604v2 Announce Type: replace-cross  Abstract: Generate--test--revise loops are common in coding agents, but repetition alone provides no reliability guarantee. We study the gap between finding a correct patch and retaining, verifying, and submitting it. A sealed five-seed study over 30 HumanEval repairs produces 900 three-revision trajectories. Under forced revision, current correctness with current traces falls from 0.820 after one revision to 0.673 after two, although ever-correct rises to 0.847. Two common-state studies use 2,430 branches from identical frozen programs to remove post-treatment risk-set bias. In a prespecified 14B replication, stale traces harm 34/135 correct starts versus 4/135 with current traces, a 22.2-point increase (task-cluster 95\% CI $[8.9,37.0]$, exact Holm $p=0.0337$). A prospective 540-rollout policy eliminates observed correct-start harm but reduces wrong-start repair and fails its joint criterion. Repository experiments over 24 bugs and fou
    
[^242]: 大语言模型中有生命性栖身何处：追踪有生命性概念的回路

    Where Animacy Lives in Large Language Models: Tracing the Circuits of the Animacy Concept

    [https://arxiv.org/abs/2607.20995](https://arxiv.org/abs/2607.20995)

    本研究通过构建最小对数据集并进行回路发现，首次在大语言模型中追踪并证实了处理有生命性概念的因果回路的存在，同时发现该回路较为分散，其跨模型和跨任务的泛化能力有限。

    

    在书面语言中区分有生命概念与无生命概念并非仅靠浅层文本处理即可完成，它涉及识别复杂的选择限制和上下文线索，例如动词与论元之间的交互。然而，当前的大语言模型似乎具备这种能力。我们研究大语言模型的这种对有生命性敏感的行为是否可以追溯至一组局部化的、因果相关的组件和连接。为此，我们构建了一个由最小对组成的受控数据集，并在四个开源权重模型上进行了回路发现。通过深入的实验和消融研究，我们证明这些模型中确实存在负责处理有生命性的因果机制，从而发现了一个“有生命性回路”。与此同时，该回路相比其他已知回路的局部化程度较低，且仅能在一定程度上跨模型和跨有生命性任务泛化，这证实了有生命性表征具有分布式、依赖上下文等特点。

    arXiv:2607.20995v2 Announce Type: replace  Abstract: Distinguishing animate from inanimate concepts in written language requires more than shallow text processing, as it involves recognizing complex selectional constraints and contextual cues, such as verb-argument interactions. Yet, current large language models (LLMs) appear to be capable of doing it. We investigate whether this animacy-sensitive behavior of LLMs can be traced to a localized set of causally relevant components and connections. To do so, we construct a controlled dataset of minimal pairs and perform circuit discovery on four open-weight models. Through in-depth experiments and ablations, we show that a causal mechanism responsible for handling animacy in these models does exist, thus discovering an animacy circuit. At the same time, this circuit appears to be less localized compared to other known ones and generalizes only partially across models and animacy tasks, confirming the distributed, context-dependent, and so
    
[^243]: 化学思维链：一个易产生幻觉的分子草稿板

    Chemical Chain-of-Thought Functions as a Hallucination-Prone Molecular Scratchpad

    [https://arxiv.org/abs/2607.20935](https://arxiv.org/abs/2607.20935)

    化学思维链推理中幻觉普遍存在且与答案正确性脱钩，它并非忠实的解释，而是以模型特异的形式充当“分子草稿板”，其中结构草稿对生成质量具有因果支撑作用。

    

    化学推理语言模型被期望通过忠实的思维链（CoT）推导出分子问题的答案。然而，在四个推理模型家族和十二项化学任务的评估中，幻觉现象普遍存在，且在很大程度上与答案的正确性相互脱钩：正确的答案往往与编造出的、在相关分子中并不存在的结构断言共存。但这并不意味着推理轨迹在计算上是无关紧要的。归因分析表明，各模型共享一种“草稿板”功能，只是表现形式因模型而异：Chem-R 和 ether-0 依赖于碎片化的 SMILES 草稿，而 ChemDFM-R 则更强调分子骨架、位置和命名线索。值得注意的是，对 Chem-R 的 SMILES 草稿进行扰动会降低其生成质量，这表明即使语言层面的结构断言在很大程度上是“惰性”的，结构草稿本身仍可能在因果上起到支撑作用。综上，这些结果表明化学 CoT 既不是一种忠实的解释，也不仅仅是事后的合理化……

    arXiv:2607.20935v2 Announce Type: replace-cross  Abstract: Chemical reasoning language models are expected to derive molecular answers through faithful chain-of-thought (CoT). However, across four reasoning model families and twelve chemistry tasks, hallucination is widespread and largely decoupled from answer correctness: correct answers often coexist with fabricated structural claims absent from the relevant molecules. Yet this does not make the reasoning trace computationally irrelevant. Attribution analyses suggest a shared scratchpad function expressed in model-specific forms: Chem-R and ether-0 rely on fragmented SMILES drafts, whereas ChemDFM-R emphasizes scaffold, positional, and naming cues. Notably, perturbing Chem-R's SMILES sketches degrades generation, showing that structural drafts can be causally load-bearing even when verbal structural claims are largely inert. Together, these results show that chemical CoT is neither a faithful explanation nor merely a post-hoc rationa
    
[^244]: AdaFlash：基于在线策略蒸馏扩散起草器的自适应投机解码

    AdaFlash: Adaptive Speculative Decoding via On-Policy Distilled Diffusion Drafters

    [https://arxiv.org/abs/2607.19223](https://arxiv.org/abs/2607.19223)

    论文揭示了扩散起草器中双向注意力导致的领域级和词元级高方差问题，并提出 AdaFlash 自适应投机解码框架加以解决。

    

    投机解码是一种由轻量级起草模型先生成草稿序列、再由目标模型进行验证的加速技术，已成为加速大语言模型推理的主流范式。诸如 DFlash 等近期工作通过利用扩散起草器进一步提升了起草效率，其并行去噪机制使得草稿生成能够在单次前向传播中完成。在本工作中，我们揭示了扩散起草器的一个核心缺陷：双向注意力是一把双刃剑。一方面，它赋予模型并行生成和全局上下文建模能力；另一方面，这种固有的全局依赖性在领域层面和词元层面都引入了高方差：接受率在不同领域间波动显著，且草稿词元的质量在不同词元位置上也呈现异质性变化。为解决这一问题，我们提出了 AdaFlash 框架……

    arXiv:2607.19223v2 Announce Type: replace-cross  Abstract: Speculative decoding, in which a lightweight draft model first generates a draft sequence that is then verified by the target model, has become a prevalent paradigm for accelerating large language model inference. Recent work such as DFlash further boosts drafting efficiency by leveraging diffusion drafters, whose parallel denoising mechanism enables draft generation in a single forward pass. In this work, we uncover a central pitfall of diffusion drafters: bidirectional attention is a double-edged sword. On one hand, it endows the model with parallel generation and global contextual modeling capabilities; on the other hand, this inherent global dependency introduces high variance at both the domain-level and the token-level: acceptance rates fluctuate substantially across different domains, and draft token quality also varies heterogeneously at different token positions. To tackle this issue, we propose AdaFlash framework, com
    
[^245]: 相似准确率，不平等的证据：搜索API作为工具使用型智能体的决策表面

    Similar Accuracy, Unequal Evidence: Search APIs as Decision Surfaces for Tool-Using Agents

    [https://arxiv.org/abs/2607.10198](https://arxiv.org/abs/2607.10198)

    该研究在固定实验条件下对比Brave、Tavily和Firecrawl三个搜索API作为工具使用型智能体的决策表面，发现三者的任务准确率虽相似，但在证据暴露特征上存在显著差异——Brave提供更多预抓取支持、Tavily排名第一结果的支持证据占比更高、Firecrawl则带来更广泛的探索。

    

    搜索API提供排序后的片段、URL和元数据，智能体基于这些信息决定是回答问题、再次搜索还是抓取页面。我们在来自254个问题的SealQA-Hard子集的固定100个问题样本上，将这些接口作为决策表面进行评估，使用一个冻结的GPT-5.4智能体、固定的编排框架，以及跨Brave、Tavily和Firecrawl共享的页面抓取后端。一个Kimi-K2.6预言机对可见的URL级证据进行标注；独立的答案审计得出100个问题中分别有25、25和26个正确答案。这些计数表明观察到的准确率相似，但并不能证明等同性。在测试的配置下，Brave在更大的片段表面上暴露出更多预抓取支持；Tavily在轨迹池化的支持性观察中拥有更大的排名第一份额；而Firecrawl则与更广泛的探索相关联。首次搜索的查询级指标将支持可用性与排序区分开来，而矛盾暴露……

    arXiv:2607.10198v2 Announce Type: replace  Abstract: Search APIs expose ranked snippets, URLs, and metadata on which agents decide whether to answer, search again, or fetch pages. We evaluate these interfaces as decision surfaces on a fixed sample of 100 questions from the 254-question SealQA-Hard subset, using one frozen GPT-5.4 agent, a fixed orchestration harness, and a shared page-fetch backend across Brave, Tavily, and Firecrawl. A Kimi-K2.6 oracle labels visible URL-level evidence; a separate answer audit yields 25, 25, and 26 correct answers out of 100. These counts indicate similar observed accuracy but do not establish equivalence. Under the tested configurations, Brave exposes more pre-fetch support alongside a larger snippet surface; Tavily has a larger rank-1 share among trajectory-pooled supporting observations; and Firecrawl is associated with broader exploration. First-search query-level metrics distinguish support availability from ranking, while contradiction exposure 
    
[^246]: LEXIC：基于眼动数据的轻量级端侧阅读理解解码

    LEXIC: Lightweight On-Device Decoding of Reading Comprehension from Eye Movements

    [https://arxiv.org/abs/2607.08152](https://arxiv.org/abs/2607.08152)

    论文提出LEXIC——一个仅含41.6K参数、166 KB大小的紧凑循环模型，可从眼动数据直接预测阅读理解成绩，无需语言模型推理，单核CPU上每次试验仅需1.4毫秒，实现了真正轻量级的端侧部署。

    

    从眼动数据预测阅读理解能力可以支持自适应阅读界面。我们提出了LEXIC，一个紧凑的循环神经网络模型，可从注视序列、词频和字符长度预测答题正确性。该模型仅有41.6K参数，且无需语言模型推理。在OneStop数据集上，平均ROC曲线下面积（AUROC）在未见文本（Unseen Text）设置下达到0.529，在未见读者（Unseen Reader）设置下达到0.554。与AhnCNN的匹配对比实验表明，编码器的重新设计和词汇特征的增强在这些设置下均带来了性能提升。在SB-SAT数据集上的独立训练与评估也取得了比AhnCNN更高的平均AUROC。LEXIC仅占用166 KB的模型权重，在单个CPU核心上每次试验仅需1.4毫秒，支持轻量级的端侧推理。

    arXiv:2607.08152v2 Announce Type: replace-cross  Abstract: Predicting comprehension from eye movements could support adaptive reading interfaces. We present LEXIC, a compact recurrent model that predicts response correctness from fixation sequences, word frequency, and character length. It has 41.6K parameters and requires no language-model inference. Mean area under the receiver operating characteristic curve (AUROC) reaches 0.529 for Unseen Text and 0.554 for Unseen Reader on OneStop. Matched comparisons with AhnCNN show gains from both encoder redesign and lexical augmentation in these settings. Separate training and evaluation on SB-SAT also yield higher mean AUROC than AhnCNN. LEXIC occupies only 166 KB of model weights and requires 1.4 ms per trial on a single CPU core, supporting lightweight on-device inference.
    
[^247]: REDDIT：通过基于重放的分布编辑修正ASR中模型生成的时间戳漂移且不发生遗忘

    REDDIT: Correcting Model-Generated Timestamp Drift in ASR without Forgetting via Replay-Based Distribution Editing

    [https://arxiv.org/abs/2607.05364](https://arxiv.org/abs/2607.05364)

    该论文提出了轻量级两阶段后训练框架REDDIT，通过基于重放的分布编辑修正自回归ASR系统在长非语音片段上产生的时间戳漂移，同时避免了朴素微调带来的灾难性遗忘问题。

    

    现代自回归ASR系统可以将时间戳作为解码token输出，从而无需帧级对齐器或推理时后处理即可实现带时间戳的转录。我们证明，这些生成的时间戳在较长的非语音片段上可能发生漂移：转录文本看起来可能仍然合理，但解码出的时间轴会偏离音频。我们使用自建的gap和long-gap基准，在15个被评估的时间戳生成ASR和音频-语言系统上研究了这种由非语音引起的时间戳漂移。朴素的时间戳修正微调虽能改善对齐，但会严重损害非目标的ASR行为，暴露出遗忘问题。我们提出REDDIT（REplay-based Distribution eDITing，基于重放的分布编辑），这是一个轻量级的两阶段后训练框架，能够在修正时间戳的同时避免这种灾难性遗忘：它首先在模型自身重放的解码器上下文中编辑时间戳目标，同时匹配冻结的基础分布

    arXiv:2607.05364v3 Announce Type: replace-cross  Abstract: Modern autoregressive ASR systems can emit timestamps as decoded tokens, enabling timestamped transcription without frame-level aligners or inference-time post-processing. We show that these generated timestamps can drift across long non-speech spans: the transcript may remain plausible, but the decoded time axis drifts away from the audio. We study this non-speech-induced timestamp drift with self-built gap and long-gap benchmarks across 15 evaluated timestamp-producing ASR and audio-language systems. Naive timestamp-corrected fine-tuning improves alignment but can severely degrade non-target ASR behavior, exposing a forgetting problem. We propose REDDIT(REplay-based Distribution eDITing), a lightweight two-stage post-training framework that corrects timestamps while avoiding this catastrophic forgetting: it first edits timestamp targets under the model's own replayed decoder context while matching the frozen base distribution
    
[^248]: 将文本映射到多重图：将提示压缩视为列维游走引导的图剪枝

    Mapping Text to Multiplex Graph: Prompt Compression as L\'evy Walk-Guided Graph Pruning

    [https://arxiv.org/abs/2607.01241](https://arxiv.org/abs/2607.01241)

    该论文提出RAGP方法，将提示压缩建模为多重图上的冗余感知图剪枝问题，并利用列维游走的重尾步长分布平衡局部与全局探索，从而高效识别文本中分散的非冗余重要信息。

    

    现有的提示压缩方法将文本视为扁平的词元序列，未能捕捉重要信息的分布式特性——这些重要信息往往分散在多个位置，并通过局部句法依赖和全局语义关系相互连接。这种关系结构天然适合用图来表示，其中词元或句子作为节点，它们之间的依赖关系作为边。为此，我们提出了RAGP方法，将提示压缩形式化为多重图上的冗余感知图剪枝问题，该多重图联合建模了细粒度的基于注意力的依赖关系和粗粒度的语义关系。为了在这种异构结构（密集的局部子图和稀疏的全局连接）中高效识别非冗余节点，我们采用列维游走（Lévy Walk），其重尾步长分布能够自然地平衡局部开发与全局探索。在LongBench上的实验表明，RAGP取得了……

    arXiv:2607.01241v2 Announce Type: replace-cross  Abstract: Existing prompt compression methods treat text as flat token sequences, failing to capture the distributed nature of important information, which is often spread across multiple locations and connected through both local syntactic dependencies and global semantic relations. Such relational structure is naturally represented as a graph, where tokens or sentences become nodes and their dependencies become edges. To this end, we propose RAGP, which formulates prompt compression as Redundancy-Aware Graph Pruning on a multiplex graph that jointly models fine-grained attention-based dependencies and coarse-grained semantic relations. To efficiently identify non-redundant nodes in this heterogeneous structure (dense local subgraphs and sparse global connections), we employ Levy walks whose heavy-tailed step distribution naturally balances local exploitation with global exploration. Experiments on LongBench show that RAGP achieves an a
    
[^249]: 架起科学遗产之桥：面向可持续知识转移的阿拉伯语—俄语平行语料库与大语言模型基准

    Bridging Scientific Heritage: An Arabic--Russian Parallel Corpus and LLM Benchmark for Sustainable Knowledge Transfer

    [https://arxiv.org/abs/2606.30943](https://arxiv.org/abs/2606.30943)

    该论文构建了阿拉伯语—俄语科学翻译的首个基准，包含约27,000句对的混合平行语料库，并通过LoRA微调多种多语言大模型，其中QLoRA微调的Qwen2.5-7B取得最佳性能，显著优于零样本基线，为俄阿两个科学社区之间的可持续知识转移奠定了基础。

    

    俄语和阿拉伯语是科学交流的主要语言之一。语言障碍阻碍了这两个学术群体之间研究成果的交流，进而影响了国际合作以及可持续性相关研究的进展。我们提出了一个阿拉伯语—俄语科学翻译基准，其中包含一个约27,000句对的混合平行语料库，由科学摘要和通用领域文本（宗教、新闻、对话）汇编而成。我们使用秩为8、16、32和64的LoRA方法对三个多语言模型——mT5-base（5.8亿参数）、NLLB-200-distilled-1.3B（13亿参数）和Qwen2.5-7B-Instruct（70亿参数）——进行了微调。采用QLoRA（秩为8）的Qwen2.5-7B模型取得了BLEU 23.15、chrF 43.89、BERTScore 0.906和COMET 0.758的成绩，比零样本基线分别高出4.36 BLEU和0.051 COMET。使用三个示例的少样本提示并未提升性能，这表明领域……（摘要原文在此处截断）

    arXiv:2606.30943v2 Announce Type: replace  Abstract: Russian and Arabic are among the major languages of scientific communication. Language barriers impede the exchange of research results between these communities, which affects international collaboration and the progress of sustainability-related research. We present a benchmark for Arabic--Russian scientific translation. The benchmark includes a hybrid parallel corpus of about 27,000 sentence pairs, compiled from scientific abstracts and general-domain texts (religion, news, conversations). We fine-tune three multilingual language models -- mT5-base (580M parameters), NLLB-200-distilled-1.3B (1.3B), and Qwen2.5-7B-Instruct (7B) -- using LoRA with ranks 8, 16, 32, and 64. The Qwen2.5-7B model with QLoRA (rank 8) yields BLEU 23.15, chrF 43.89, BERTScore 0.906, and COMET 0.758. These are +4.36 BLEU and +0.051 COMET above the zero-shot baseline. Few-shot prompting with three examples does not improve performance, indicating that domain
    
[^250]: 基于大语言模型智能体的金属有机框架可解释逆向设计

    Interpretable Inverse Design of Metal-Organic Frameworks with Large Language Model Agents

    [https://arxiv.org/abs/2606.29459](https://arxiv.org/abs/2606.29459)

    本文提出LLM4MOF闭环多智能体框架，利用大语言模型将自然语言目标转化为可解释的化学假设与约束，在400次评估内于多项吸附、分离和电子结构任务中筛选出高性能金属有机框架，并实现了新型MOFs的从头设计与实时模拟。

    

    金属有机框架（MOFs）的逆向设计需要在具有昂贵性质标签和不透明机器学习模型的组合空间中进行搜索。我们提出了LLM4MOF，这是一个闭环多智能体框架，可将自然语言目标转化为化学假设、约束条件、诊断测试和反馈。一个智能体提出关于金属节点、连接体、孔隙几何形状和功能的可解释假设。另一个智能体将这些假设转化为约束条件，以选择由节点、连接体和拓扑结构定义的MOFs。“匹配器”（Matchmaker）形成四条搜索路径，将性能提升归因于几何形状、化学性质或金属选择：完整假设、仅化学性质、仅金属和随机基线。在对数据库分布信息一无所知的情况下，LLM4MOF在400次评估内，于六项吸附、分离和电子结构任务中筛选出顶级性能材料。它还设计和实时模拟了从头构建的新型MOFs，应用涵盖氢气储存、六氟化硫捕获以及乙烷/乙烯分离等领域。

    arXiv:2606.29459v2 Announce Type: replace-cross  Abstract: Inverse design of metal-organic frameworks (MOFs) requires navigating combinatorial spaces with costly property labels and opaque machine-learning models. We introduce LLM4MOF, a closed-loop multi-agent framework that converts a natural-language target into chemical hypotheses, constraints, diagnostic tests, and feedback. One agent proposes interpretable hypotheses over metal nodes, linkers, pore geometry, and functionality. Another converts them into constraints selecting MOFs defined by a node, linker, and topology. The Matchmaker forms four beams to attribute gains to geometry, chemistry, or metal choice: full hypothesis, chemistry, metal only, and random baseline. Blind to database landscapes, LLM4MOF enriches top performers across six adsorption, separation, and electronic-structure tasks within 400 evaluations. It also designs and live-simulates de novo MOFs spanning H2 storage, SF6 capture, and C2H6/C2H4 separation, deri
    
[^251]: MMLA：记忆如何让过去塑造未来

    MMLA: How Memory Lets the Past Shape the Future

    [https://arxiv.org/abs/2606.28876](https://arxiv.org/abs/2606.28876)

    MMLA通过有界驻留内存和事件化处理，在因果部署中实现记忆选择，显著提升多跳问答性能。

    

    摘要：提案。长上下文可以重放历史，但它并不决定哪些已完成的观察结果应获得权威性。MMLA在瞬态上下文和慢速权重更新之间形式化了一个有界驻留内存。一个完成的局部片段被事件化；对于每个事件，一个目标条件的构造器提出语义内容，一个可信的汇编器生成完整的版本化行；部署要么原子地提交该行，要么返回NULL。实现的未来可能在训练期间为动作定价，而部署保持因果性和对未来不可见。验证组件。受控研究确定了更窄的组成部分。生命周期执行在三个种子的300/300个保留记录上精确无误。带全档案回退的校准选择在保留的多跳问答上，比弱预算匹配的密集基线提高5.5-16.6 F1，比BM25提高4.0-6.2 F1；原始的Llama预算执行被视为失败，而相应的...

    arXiv:2606.28876v3 Announce Type: replace  Abstract: Proposal. Long context can replay history, but it does not decide which completed observations deserve authority. MMLA formalizes a bounded resident memory between transient context and slow weight updates. A completed local segment is eventized; for each event, a target-conditioned constructor proposes semantic content and a trusted assembler produces a complete versioned row; deployment either commits that row atomically or returns NULL. Realized futures may price actions during training, while deployment remains causal and future-blind.   Validated components. Controlled studies establish narrower ingredients. Lifecycle execution is exact on 300/300 held-out records for each of three seeds. Calibrated selection with full-archive fallback improves over a weak budget-matched dense baseline by 5.5--16.6 F1 and over BM25 by 4.0--6.2 F1 on held-out multi-hop QA; the original Llama budget execution is retained as failed, while the corre
    
[^252]: 人类撰写文本中事实错误的实证分析及其在事实错误检测中的应用

    An Empirical Analysis of Factual Errors in Human-Written Text and Its Application to Factual Error Detection

    [https://arxiv.org/abs/2606.27959](https://arxiv.org/abs/2606.27959)

    该论文通过分析报纸文章更正构建了人类撰写文本中事实错误的分类体系，发现了汉字误转换、单位错误等现有幻觉基准未覆盖的错误类型，并据此评估了大语言模型在事实错误检测任务上的能力。

    

    arXiv:2606.27959v2 公告类型：替换 摘要：事实错误检测（FED），即识别给定文本中事实错误片段的任务，长期以来被认为是一个重要的研究问题。然而，随着大型语言模型（LLM）的迅速崛起，研究注意力已转向LLM生成文本所特有的事实错误（即幻觉）及其检测。因此，人类撰写文本中事实错误的检测相对被忽视了。为了填补这一空白，我们首先通过分析报纸文章的更正内容来提炼人类引发的事实错误分类体系——报纸文章是一种保证由人类撰写且语法错误较少的代表性文本来源。我们的分析揭示了存在一些具有特征性的错误类别，例如汉字误转换和单位错误，而这些类别在现有的幻觉基准中并未受到关注。基于该分类体系，我们随后评估了通用LLM在合成的真实（文本上）的事实错误检测能力。

    arXiv:2606.27959v2 Announce Type: replace  Abstract: Factual Error Detection (FED), which is the task of identifying factually incorrect spans in a given text, has long been recognized as an important research problem. However, with the rapid rise of large language models (LLMs), research attention has shifted toward factual errors specific to LLM-generated text (hallucinations) and their detection. As a result, the detection of factual errors in human-written text has been relatively neglected. To address this gap, we first distill a taxonomy of human-induced factual errors by analyzing corrections of newspaper articles, a representative source of text that is guaranteed to be human-written and contains few grammatical errors. Our analysis revealed that there are characteristic categories such as kanji misconversions and unit errors, which are not focused in existing hallucination benchmarks. Based on the taxonomy, we then evaluate the FED capability of vanilla LLMs on synthesized rea
    
[^253]: 超越表面形式：一种面向机制的间接语言编码综合分类法，用于基于大语言模型的隐语检测

    Beyond Surface Forms: A Comprehensive, Mechanism-Oriented Taxonomy of Indirect Linguistic Encoding for LLM-Based Coded Language Detection

    [https://arxiv.org/abs/2606.27314](https://arxiv.org/abs/2606.27314)

    提出了一种面向机制的间接语言编码综合分类法，通过抽象化交际目标、分类底层编码操作，显著提升了基于大语言模型的隐语检测性能（准确率提高4.7%，F1提高5.4%）。

    

    为了避免社交媒体上的审核和监控，一些用户经常发明间接语言表达（ILE），以伪装敏感含义。这些表达根据意图和语境，表现为算法隐语、委婉语和对抗性混淆，并涉及重复出现的编码机制。我们提出了一种全面、面向机制的ILE分类法，该分类法抽象化了交际目标，转而分类编码和还原意义所依赖的底层操作。我们通过将该分类法融入大语言模型提示中，并与四种现有分类法及一个无分类法基线进行比较来评估它，使用了2000条手动标注的TikTok和Bluesky帖子。所提出的分类法在三个大语言模型上均取得了最强的文档级和跨度级性能，在准确率上比最佳基准提高了4.7%，在F1分数上提高了5.4%。实证结果揭示了该分类法的有效性。

    arXiv:2606.27314v1 Announce Type: new  Abstract: To avoid moderation and surveillance on social media, some users routinely invent indirect linguistic expressions (ILE) that camouflage sensitive meanings. Such expressions surface as algospeak, euphemisms, and adversarial obfuscation, depending on intent and context, and they involve recurring encoding mechanisms. We propose a comprehensive, mechanism-oriented taxonomy of ILE that abstracts away from communicative goals and instead categorizes the underlying operations through which meaning is encoded and recovered. We evaluate the taxonomy by incorporating it into LLM prompts and comparing it with four existing taxonomies and a no-taxonomy baseline, using 2,000 manually annotated TikTok and Bluesky posts. The proposed taxonomy attains the strongest document- and span-level performance across the three LLMs, achieving an improvement of 4.7% in accuracy and 5.4% in F1 over the best-performing benchmark. The empirical results reveal the i
    
[^254]: 组合性与进化语义学中的词汇表

    Compositionality and the lexicon in evolutionary semantics

    [https://arxiv.org/abs/2606.27228](https://arxiv.org/abs/2606.27228)

    该论文提出了一种将形式语义学的组合性原理整合到进化建模中的新框架，通过让词汇意义与组合功能共同进化，揭示了语义普遍性（如保守性）作为高效系统级抽象的产生机制，并有效调和了经验证据与进化模型之间的冲突。

    

    形式语义学表明，句子意义是通过递归组合词汇意义产生的，然而，关于语义普遍性的大量文献要么对具有固定信号结构的词汇表进行建模，要么对没有可解释词汇部分的整体组合进行建模。我们引入了一个框架，将形式语义学这一基本见解整合到进化建模中，允许词汇意义和组合功能在概念简洁性和交际准确性的压力下共同进化。我们将此框架应用于量化意义的进化。分析帕累托前沿，我们发现最著名的语义普遍性——保守性，作为一种高效的系统级抽象出现。该解释对句法结构敏感，并有助于调和关于量词可学习性的经验证据与先前进化模型之间的紧张关系。更广泛地说，这些结果表明，

    arXiv:2606.27228v1 Announce Type: new  Abstract: Formal semantics has shown that sentence meanings arise by recursively composing lexical meanings, yet much of the literature on semantic universals models either lexicons with fixed signal structures or holistic composition without interpretable lexical parts. We introduce a framework that integrates this fundamental insight of formal semantics in evolutionary modeling, by allowing lexical meanings and a composition function to co-evolve under pressures for conceptual simplicity and communicative accuracy. We apply this framework to the evolution of quantificational meaning. Analyzing the Pareto frontier, we find that the most well-known semantic universal, conservativity, emerges as an efficient system-wide abstraction. The account is sensitive to syntactic structure and helps reconcile tensions between empirical evidence on quantifier learnability and prior evolutionary models. More broadly, the results demonstrate that the picture of
    
[^255]: 人类放弃，推理模型坚持：区分难度登记与深思分配

    Humans Disengage, Reasoning Models Persist: Separating Difficulty Registration from Deliberation Allocation

    [https://arxiv.org/abs/2606.26502](https://arxiv.org/abs/2606.26502)

    论文发现，在相同问题上，人类答错时花更少时间（放弃），而大型推理模型答错时花更多令牌（坚持），揭示了人类与AI在失败时资源分配策略的根本差异。

    

    arXiv:2606.26502v1 公告类型：新 摘要：大型推理模型（LRMs）在更困难的问题上花费更长时间，就像人类一样。这种表面上的相似性掩盖了项目内部的相反模式。当LRM答错一个问题时，它花费的令牌数比答对同一个问题时更多；而人类则相反，在答错的试验上花费的时间更少。我们将深思的两个层面分开：反应时间如何跨项目追踪难度（登记），以及在项目身份固定的情况下，智能体是在自己的失败还是成功上花费更多（分配）。在一个公开的人-LRM匹配语料库上，人类和所有五种思维LRM都再现了已知的跨项目一致性（登记），但在项目内部分配中出现了分歧：每个LRM都显示出显著的“答错 vs 答对”效应（在H-ARC上Cohen's d = 1.47-3.13），而人类则显示出相反的符号。比较是在每个智能体自身的尺度内进行的；我们从未将秒和令牌放在同一轴上。在项目固定的情况下，这种分离依然成立。

    arXiv:2606.26502v1 Announce Type: new  Abstract: Large reasoning models (LRMs) take longer on harder problems, just as humans do. This surface similarity hides an opposite pattern within items. When an LRM gets a problem wrong, it spends more tokens than when it gets the same problem right; humans do the reverse, spending less time on the trials they get wrong. We separate two levels of deliberation: how response time tracks difficulty across items (registration), and, with item identity held fixed, whether an agent spends more on its own failures or successes (allocation). On a public matched human-LRM corpus, humans and all five thinking LRMs reproduce the known cross-item alignment (registration) but diverge within items (allocation): every LRM shows a large wrong-vs-right effect (Cohen's d = 1.47-3.13 on H-ARC) while humans show the opposite sign. The comparison stays inside each agent's own scale; we never put seconds and tokens on one axis. The dissociation holds under item fixed
    
[^256]: 形式思维之织

    Weave of Formal Thought

    [https://arxiv.org/abs/2606.25987](https://arxiv.org/abs/2606.25987)

    本文提出了一个通过新颖投机词法分析增强GLR解析、对完整Tree-sitter规范可靠且完备的形式引擎与约束解码器，并引入隐变量微调方法WoFT，使语言模型在代码生成中能够利用语法层次结构。

    

    大型语言模型在代码生成上展现出卓越的表面流畅性，但它们并不能从形式上保证输出的语法有效性，也通常不会利用定义目标语言的层次结构。虽然现有的受约束解码框架为前者提供了解决方案，但它们大多在僵化的假设下运行，排除了现代解析器所依赖的关键词法机制（例如 Python 风格的缩进）。在本工作中，我们提出了一个形式引擎和受约束解码器，它相对于完整的 Tree-sitter 规范是可靠且完备的，其方法是通过一种新颖的投机词法分析构造来增强广义 LR（GLR）解析，该构造维护与 GLR 图结构栈同步的并发词法分析器状态假设。我们还引入了形式思维之织，这是一种隐变量微调方法，用于训练语言模型在生成过程中交织非终结符语法结构……

    arXiv:2606.25987v2 Announce Type: replace-cross  Abstract: Large language models attain remarkable surface fluency on code, yet they do not formally guarantee the syntactic validity of their output, nor do they typically leverage the hierarchical structure that defines the target language. While existing constrained-decoding frameworks offer a solution to the former, they predominantly operate under rigid assumptions that preclude critical lexical mechanisms relied upon by modern parsers (e.g., Pythonic indentation). In this work, we present a formal engine and constrained decoder that is sound and complete with respect to the full Tree-sitter specification by augmenting generalized LR (GLR) parsing with a novel speculative-lexing construction that maintains concurrent lexer-state hypotheses synchronized with the GLR graph-structured stack. We also introduce Weave of Formal Thought (WoFT), a latent-variable fine-tuning method that trains the language model to interleave non-terminal gr
    
[^257]: 语言-能源鸿沟：测量多语言大语言模型推理的能源成本

    The Language-Energy Divide: Measuring Energy Costs of Multilingual LLM Inference

    [https://arxiv.org/abs/2606.21869](https://arxiv.org/abs/2606.21869)

    该论文系统性测量了大语言模型多语言推理的能耗，揭示了不同语言间高达179倍的能耗差异，并发现高能耗语言同时伴随最低的任务准确率，存在“成本+性能”双重惩罚现象。

    

    大语言模型（LLM）日益被部署于多语言环境中，然而跨不同语言服务这些模型的能源成本仍然鲜为人知。我们利用 ML.Energy 框架对不同语言的推理能耗进行了系统性研究。我们发现了显著的差异：不同语言之间每个输出 token 的能耗差异高达 8.3 倍，而在固定的请求集合下，最便宜的语言（英语，17.6 kJ）与最昂贵的语言（普什图语，3,147 kJ）之间的总能耗差异高达 179 倍。我们的分析表明，这种差异由两个叠加因素驱动：（1）使用复杂或罕见文字系统的语言具有更高的每 token 能耗成本；（2）低资源语言会生成更多的 token。此外，我们还发现了一种“成本+性能”的双重惩罚：能耗最高的语言往往任务准确率也最低。

    arXiv:2606.21869v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly deployed in multilingual settings, yet the energy costs of serving these models across different languages remain poorly understood. We present a systematic study of inference energy consumption across languages with ML.Energy framework (Chung et al., 2026). We find striking disparities: energy consumption per output token varies by up to 8.3 times across languages, while total energy for a fixed set of requests varies by up to 179 times between the cheapest (English, 17.6 kJ) and the most expensive (Pashto, 3,147 kJ) languages. Our analysis shows that this disparity is driven by two compounding factors: (1) higher per-token energy costs for languages using complex or rare scripts, and (2) more tokens generated for low-resource languages. Moreover, we find a double cost + performance penalty: languages with the highest energy footprints also tend to achieve the lowest task accuracy.
    
[^258]: PARSE：面向专业领域大语言模型智能体的溯源感知检索净化

    PARSE: Provenance-Aware Retrieval Sanitization for Professional Domain LLM Agents

    [https://arxiv.org/abs/2606.17467](https://arxiv.org/abs/2606.17467)

    论文揭示了现有提示注入防御方法（如改写）在真实企业文档上失效的差距，并提出了PARSE——一个领域感知、保留事实的溯源感知检索净化流水线，可为专业领域大语言模型智能体提供更有效的提示注入防御。

    

    在合成基准上评估的提示注入防御方法无法泛化到真实的企业文档，这些文档更长、更密集，并且将合法的权威语言与事实内容交织在一起。我们通过一个涵盖五个专业领域（金融、法律、医学、科学、DevOps）共122个任务的基准来展示这一差距，该基准建立在真实检索文档之上——包括实际的SEC文件、联邦公报规则、PubMed摘要、arXiv论文和GitHub事故复盘报告——并配有未经人工验证的大语言模型生成的任务和伪装的攻击载荷。改写方法作为合成基准上最强的防御手段，在真实文档上并未显示出统计上显著的攻击成功率降低（p=0.500），同时还将效用从91.8%降低至82.8%。我们提出了PARSE（溯源感知检索净化），一个领域感知、保留事实的净化流水线，它按注入可能性对每个句子进行分类，提取……

    arXiv:2606.17467v3 Announce Type: replace-cross  Abstract: Prompt injection defenses evaluated on synthetic benchmarks do not generalize to real enterprise documents, which are longer, denser, and interleave legitimate authority language with factual content. We demonstrate this gap with a benchmark of 122 tasks across five professional domains (financial, legal, medical, scientific, DevOps) built on real retrieved documents -- actual SEC filings, Federal Register rules, PubMed abstracts, arXiv papers, and GitHub postmortems -- paired with LLM-generated tasks and camouflaged payloads that were not human-validated. Paraphrasing, the strongest defense on synthetic benchmarks, shows no statistically significant attack success rate reduction on real documents (p=0.500) while degrading utility from 91.8% to 82.8%. We introduce PARSE (Provenance-Aware Retrieval Sanitization), a domain-aware, fact-preserving sanitization pipeline that classifies each sentence by injection likelihood, extracts
    
[^259]: RLCSD：基于对比式在线策略自蒸馏的强化学习

    RLCSD: Reinforcement Learning with Contrastive On-Policy Self-Distillation

    [https://arxiv.org/abs/2606.11709](https://arxiv.org/abs/2606.11709)

    提出RLCSD方法，通过对比正确与错误提示下的师生分布差距来抑制“特权诱导的风格漂移”，使自蒸馏监督信号更集中于任务相关token，从而提升推理模型强化学习训练的稳定性与效果。

    

    在线策略自蒸馏（OPSD）通过将模型自身的分布与特权上下文（通常是经过验证的解答）下的分布对齐，为推理模型提供密集的token级监督。然而，我们发现由此产生的分布差距主要集中在风格token而非任务相关token上，因为带有提示的模型倾向于产生更短、更直接的输出。我们将这种病态称为“特权诱导的风格漂移”，它可能导致训练不稳定并缩短响应长度。为解决这一问题，我们提出了RLCSD（基于对比式在线策略自蒸馏的强化学习），通过对比正确提示下与错误提示下的师生差距来缓解这种漂移，无论提示正确与否均抑制由提示引起的风格转变，从而产生更集中于任务相关token的信号。实验在Qwen3（1.7B/4B/8B）和Olmo-3-7B-Think上进行，涵盖数学等任务领域。

    arXiv:2606.11709v2 Announce Type: replace-cross  Abstract: On-policy self-distillation (OPSD) provides dense, token-level supervision for reasoning models by aligning a model's own distribution with that under privileged context, typically a verified solution. However, we show that the resulting distributional gap concentrates on style tokens rather than task-bearing ones, as the hinted model tends to produce shorter, more direct outputs. We term this pathology \emph{privilege-induced style drift}, which can destabilize training and shorten responses. To address this, we propose \textbf{RLCSD} (Reinforcement Learning with Contrastive on-policy Self-Distillation), which mitigates this drift by contrasting the teacher-student gap under a correct hint against that under a wrong hint, suppressing style shifts induced by hints regardless of correctness and yielding a signal more concentrated on task-bearing tokens. Experiments on Qwen3 (1.7B/4B/8B) and Olmo-3-7B-Think across mathematical an
    
[^260]: 面向掩码扩散语言模型的注意力折扣自适应采样器

    Attention-Discounted Adaptive Sampler for Masked Diffusion Language Models

    [https://arxiv.org/abs/2606.10829](https://arxiv.org/abs/2606.10829)

    提出免训练重排序规则ADAS，根据每个token对已选位置的注意力（按预测不确定性加权）贪婪地折扣其置信度分数，从而提升掩码扩散语言模型在低推理步数下的表现。

    

    掩码扩散语言模型可以通过在每次去噪迭代中揭示多个token来减少推理步骤，但这种并行性是脆弱的：当各个位置的预测相互耦合时，单独看来置信度较高的位置一起提交可能并不安全。现有的免训练采样器（如Top-k、Fast-dLLM和EB-Sampler）主要控制揭示多少个token，而往往通过忽略所选集合内部交互的逐token分数对候选进行排序。我们提出了ADAS，这是一种免训练的重排序规则，它保持基础采样器的停止规则不变，并根据每个token对已选择位置的注意力（以其预测不确定性加权）来贪婪地折扣其逐token置信度分数。在LLaDA-8B-Base和Dream-7B-Base上，于推理基准GSM8K和MATH500以及代码基准HumanEval和MBPP上的实验表明，将ADAS插入到所有三种采样器中都能改善低NFE（前向评估次数）下的性能。

    arXiv:2606.10829v3 Announce Type: replace-cross  Abstract: Masked diffusion language models can reduce inference steps by revealing multiple tokens per denoising iteration, but this parallelism is fragile: positions that are individually confident may be unsafe to commit together when their predictions are coupled. Existing training-free samplers such as Top-$k$, Fast-dLLM, and EB-Sampler mainly control how many tokens to reveal, while often ranking candidates by token-wise scores that ignore interactions within the selected set. We propose ADAS, a training-free reranking rule that leaves the base sampler's stopping rule unchanged and greedily discounts each token-wise confidence score according to its attention to already selected positions, weighted by their prediction uncertainty. Across LLaDA-8B-Base and Dream-7B-Base on the reasoning benchmarks GSM8K and MATH500 and the code benchmarks HumanEval and MBPP, plugging ADAS into all three samplers improves low-NFE performance at matche
    
[^261]: 在无监督词条发现中恢复齐夫分布

    Recovering the Zipfian Distribution in Unsupervised Term Discovery

    [https://arxiv.org/abs/2606.10781](https://arxiv.org/abs/2606.10781)

    该论文提出用基于图的Leiden聚类替代K-means等基于中心的方法，在无监督词条发现中显著恢复了真实词库所具有的齐夫分布特性。

    

    无监督词条发现是指将无标注语音切分为类似词或音节的单元，并将这些单元聚类为一个候选词条类型的词库。真实的词库遵循齐夫分布，然而主流的基于中心的聚类方法——K-means——由于对球形簇的归纳偏置，会产生更均匀的分布。在本文中，我们重新审视基于图的聚类作为一种自底向上的替代方案，该方法通过成对相似度连接分段嵌入，并使用Leiden算法进行划分。我们表明，在三种语言的词级和音节级词库发现任务中，图聚类显著优于基于中心的方法，并产生更接近齐夫分布的结果。另一种自底向上的方法——采用平均链接的层次聚类——也表现良好，尽管其计算效率较低，且对最终结果的控制能力较弱。

    arXiv:2606.10781v2 Announce Type: replace-cross  Abstract: Unsupervised term discovery involves segmenting unlabelled speech into word- or syllable-like units and clustering these into a lexicon of candidate types. True lexicons follow a Zipfian distribution, yet the dominant centre-based clustering approach -- K-means -- produces a more uniform distribution due to an inductive bias toward spherical clusters. In this paper we revisit graph-based clustering as a bottom-up alternative, where segment embeddings are connected by pairwise similarity and partitioned using the Leiden algorithm. We show that graph clustering substantially outperforms centre-based approaches (K-means, GMM, BIRCH) in both word- and syllable-level lexicon discovery across three languages, producing more Zipf-like distributions. Another bottom-up approach, agglomerative clustering with average linkage, also performs well, although it is computationally less efficient and allows for less control over the resulting 
    
[^262]: ParaBridge：在语音语言模型中架起副语言感知与对话行为之间的桥梁

    ParaBridge: Bridging Paralinguistic Perception and Dialogue Behavior in Speech Language Models

    [https://arxiv.org/abs/2606.10581](https://arxiv.org/abs/2606.10581)

    ParaBridge提出一种在线策略自蒸馏方法，将推理时脆弱的副语言指令支架转化为语音语言模型中稳定的行为，从而弥合副语言感知与对话行为之间的差距。

    

    语音所承载的信息远不止文字本身：孩子的声音、恐惧的语气或嘈杂的背景音，都应该使一个足够胜任的语音对话助手给出不同的回复。当前的语音语言模型（SLM）虽然能够识别这类副语言线索，但在开放式对话中却常常将其忽略。我们观察到，在推理阶段使用一种简单的副语言指令支架能够缩小这种感知-行为差距，这表明相关线索其实已经潜在地存在于模型之中。然而，这种支架在多轮对话情境和相互竞争的指令下仍然十分脆弱。因此，我们提出了ParaBridge，一种在线策略（on-policy）自蒸馏方法，将脆弱的推理时支架转化为稳定的模型内在行为。在训练过程中，支架仅作为临时的特权视图：无支架的模型自行生成回复，而带支架的视图则为其提供密集的、覆盖全词表的下一词元监督信号。

    arXiv:2606.10581v2 Announce Type: replace  Abstract: Speech carries more information than just words: a child's voice, a fearful tone, or a noisy background should all lead a sufficiently competent spoken-dialogue assistant to different replies. Current Speech Language Models (SLMs) can recognize such paralinguistic cues but often ignore them in open-ended dialogue. We observe that a simple paralinguistic instruction scaffold at the inference stage narrows this perception-behavior gap, suggesting that the relevant cues are already latent in the model. Such scaffolds, however, remain brittle under multi-turn context and competing instructions. Therefore, we propose \textbf{ParaBridge}, an on-policy self-distillation method that turns a brittle inference-time scaffold into stable model behavior. During training, the scaffold serves only as a temporary privileged view; the scaffold-free model rolls out its own response, while the scaffolded view supplies dense, full-vocabulary next-token 
    
[^263]: 选择性采信覆盖：LLM 评判者未充分使用其任务契约所授权的非定向判定

    Cherry-pick Override: LLM Judges Under-use the Non-Directional Verdicts Their Contract Authorizes

    [https://arxiv.org/abs/2606.07834](https://arxiv.org/abs/2606.07834)

    本文定义并实证揭示了一种新的错误类型——选择性采信覆盖（CPO）：当任务契约授权使用“冲突”或“证据不足”等非定向判定时，LLM 评判者仍会过度承诺“支持”或“反驳”等定向判定，此类错误在 AVeriTeC 和 VitaminC-Mixed 基准上的发生率高达 18.7%–48.0%。

    

    以证据为依据的事实核查越来越多地使用 LLM 评判者将证据转化为最终判定。许多任务契约刻意包含非定向判定——对于实质性混合的证据给出“冲突”，对于证据缺失给出“证据不足”——以便系统能够拒绝断言某个方向。在授权判定为非定向的论断上返回“支持”或“反驳”属于定向过度承诺：一种标签错误，其输出是契约未授权的方向。其冲突实例——两类证据都存在，却只返回了其中一类——被称为 Cherry-pick Override（CPO，选择性采信覆盖）；与证据不足实例不同，它无法通过检索更多证据来修复，因为本来就没有缺失任何证据。在 AVeriTeC 的黄金“冲突”子集（N_C=150）上，一个四选项类型化面板在 18.7% 的论断上做出了定向承诺，而四个冻结的当代评判器（两个专有、两个开源权重）的这一比例达 21.3%–48.0%，在 VitaminC-Mixed 上为 24.0%–38.0%。

    arXiv:2606.07834v2 Announce Type: replace-cross  Abstract: Evidence-grounded fact verification increasingly uses LLM judges to turn evidence into terminal verdicts. Many task contracts deliberately include non-directional verdicts - Conflicting for materially mixed evidence, Not Enough Evidence for absent evidence - so that a system can decline to assert a direction. Returning Supports or Refutes on a claim whose authorized verdict is non-directional is directional overcommitment: a label error whose output is a direction the contract did not authorize. Its conflict instance - both strands present, one of them returned - is Cherry-pick Override (CPO); unlike the insufficiency instance it cannot be repaired by retrieving more, because nothing was missing. On AVeriTeC's gold-Conflicting subset (N_C=150) a four-option typed panel commits directionally on 18.7% of claims, and four frozen contemporary judges (two proprietary, two open-weight) on 21.3-48.0%, with 24.0-38.0% on VitaminC-Mixed
    
[^264]: 面向位置公平密集检索的注意力校准

    Attention Calibration for Position-Fair Dense Retrieval

    [https://arxiv.org/abs/2606.02737](https://arxiv.org/abs/2606.02737)

    该论文提出一种带强度系数的注意力校准方法，缓解密集检索中嵌入表示偏向文本前部内容的位置偏差问题，同时将校准内存开销从数 GiB 降至 1 MiB 以下，显著提升相关片段位于文本后部时的检索性能。

    

    密集检索将一段文本压缩为单个向量，但这种压缩存在位置偏差：文本前部的内容主导嵌入表示，当相关片段出现在较后位置时，检索性能会下降。先前的工作提出了一种推理时方法，通过均衡池化标记（pooling token）在文本各片段上的注意力来抵消这种偏差。然而，该方法存在以下问题：（i）它以固定强度重新分配注意力；（ii）尽管在不同层和不同架构之间存在显著差异，它仍强制将池化标记对自身的注意力固定为统一的段落级质量；（iii）其对检索效果的影响尚未得到评估。我们引入了一个强度系数，在未校准与完全均衡的注意力之间进行插值调节，并提供一种高效实现，将校准的峰值内存开销从 5-7 GiB 降低至 1 MiB 以下。在三个嵌入模型和两种池化方案上的实验表明，适度的校准能够带来更好的检索效果。

    arXiv:2606.02737v2 Announce Type: replace-cross  Abstract: Dense retrieval compresses a passage into a single vector, but this compression is positionally skewed: early content dominates the embedding, and retrieval degrades when the relevant span appears later. Prior work proposed an inference-time method that counteracts this skew by equalizing the pooling token's attention across passage segments. However, (i) it redistributes attention at a fixed strength, (ii) it forces the pooling token's attention to itself to a fixed basket-level mass despite substantial variation across layers and architectures, and (iii) its effect on retrieval has not been evaluated. We introduce a strength coefficient that interpolates between uncalibrated and fully equalized attention, together with an efficient implementation that reduces peak calibration memory overhead from 5-7 GiB to under 1 MiB. Across three embedding models and two pooling schemes, moderate calibration provides a better retrieval tra
    
[^265]: 内化温度：作为强化学习策略再加热器的在线自蒸馏

    Internalize the Temperature: On-Policy Self-Distillation as Policy Reheater for Reinforcement Learning

    [https://arxiv.org/abs/2606.00755](https://arxiv.org/abs/2606.00755)

    提出TS-OPSD方法，通过对模型自身logits进行高温缩放并将平滑后的分布自蒸馏回模型，将温度的探索效应内化到模型参数中，从而在不依赖外部教师和额外推理开销的情况下缓解强化学习中的熵坍缩问题。

    

    来自可验证奖励的强化学习能够提升大型语言模型的推理能力，但常常遭受熵坍缩问题，即策略分布越来越集中，导致采样多样性和有用的学习信号减少。现有的补救措施要么约束强化学习目标（例如熵正则化），要么在采样收集过程中调整采样温度，但这些干预措施仍然是模型参数之外的。我们提出了温度缩放在线自蒸馏，这是一种轻量级的策略再加热方法，将温度的探索性效果内化到模型参数中。从一个熵坍缩的强化学习检查点出发，TS-OPSD通过对模型自身的logits应用高温缩放来构建自教师模型，然后将产生的更平滑的分布蒸馏回学生模型。这种策略再加热方法无需外部教师模型、特权数据或额外的推理……

    arXiv:2606.00755v2 Announce Type: replace  Abstract: Reinforcement learning from verifiable rewards improves the reasoning ability of large language models, but often suffers from entropy collapse, in which increasingly concentrated policies reduce rollout diversity and useful learning signals. Existing remedies either constrain the RL objective (e.g., entropy regularization) or adjust sampling temperature during rollout collection, but these interventions remain external to the model parameters. We propose Temperature-Scaled On-Policy Self-Distillation (TS-OPSD), a lightweight policy reheating method that internalizes the exploratory effect of temperature into model parameters. Starting from an entropy-collapsed RL checkpoint, TS-OPSD constructs a self-teacher by applying high-temperature scaling to the model's own logits, then distills the resulting smoother distribution back into the student. This policy reheating requires no external teacher, privileged data, or additional inferenc
    
[^266]: 用于分布外检测的连续正则化流的局部诊断方法

    Local Diagnostics of Continuous Normalizing Flow for Out-of-Distribution Detection

    [https://arxiv.org/abs/2606.00684](https://arxiv.org/abs/2606.00684)

    本文提出基于连续正则化流的拉格朗日子流（LSF）框架，利用子流轨迹速度场的几何诊断信号来检测分布外样本，从而缓解深度生成模型的“似然悖论”问题。

    

    我们针对嵌入在高维数据空间子空间中的目标观测的分布外（OOD）检测问题进行研究。利用连续正则化流（CNFs），我们提出了一个拉格朗日子流（LSF）框架，旨在分离并估计表示中相关分量的密度，同时将其余分量作为上下文。通过语音合成模型的实验，我们表明CNFs与其他深度生成模型（DGMs）类似，容易受到“似然悖论”的影响，即错误地将高似然值分配给OOD样本。这归因于深度生成模型的归纳偏置，即优先考虑低级结构细节而非高级语义一致性。为了缓解这一现象，我们基于子流轨迹上的速度场提出了若干几何诊断信号。基于这些信号，我们为具有挑战性的任务设计了度量指标……

    arXiv:2606.00684v3 Announce Type: replace-cross  Abstract: We address the problem of out-of-distribution (OOD) detection for target observations embedded in a subspace of the high dimensional data space. Using continuous normalizing flows (CNFs), we propose a Lagrangian sub-flow (LSF) framework designed to isolate and estimate the density for the relevant components in the representation and using the remaining components as context. Through experimentation with models for speech synthesis, we show that CNFs, similarly to other deep generative models (DGMs), are susceptible to the "likelihood paradox", where high likelihood is erroneously assigned to OOD samples. This is attributed to the inductive bias of DGMs that prioritize low-level structural details over high-level semantic coherence. To mitigate this phenomenon, we propose a number of geometric diagnostic signals based on the velocity field over the sub-flow trajectory. Based on these signals, we design metrics for the challengi
    
[^267]: 是在生成报告还是重复模板？测量与缓解3D CT报告生成中的模板坍塌问题

    Generating Reports or Repeating Templates? Measuring and Mitigating Template Collapse in 3D CT Report Generation

    [https://arxiv.org/abs/2605.30984](https://arxiv.org/abs/2605.30984)

    该论文首次识别并量化了3D CT报告生成中的“模板坍塌”失效模式，并提出将临床病理检测与语言合成分离的CLarGen解耦框架来有效缓解这一问题。

    

    现代3D医学视觉语言模型（VLM）能够生成流畅的放射学风格文本，但表现出极低的病理检测能力和输出多样性，坍塌为通用模板，从而低估了罕见但关键的发现。我们将这种失效模式识别为“模板坍塌”（Template Collapse）。这种失效源于3D医学影像的独特约束，例如数据有限、严重的标签不平衡以及来自体积编码器的微弱信号。在这些约束下，文本生成目标鼓励了捷径学习和流畅但缺乏真实依据的报告。我们通过临床保真度、输出多样性、正常模板偏差和罕见发现存活率系统地诊断了模板坍塌现象。为缓解该问题，我们提出了CLarGen，一个将“说什么”（临床检测）与“怎么说”（语言合成）解耦的框架。CLarGen使用：（i）用于多标签病理检测的潜在查询变换器，（ii（摘要在此处被截断）

    arXiv:2605.30984v2 Announce Type: replace-cross  Abstract: Modern 3D medical vision-language models (VLMs) can generate fluent radiology-style text while exhibit critically low pathology detection and output diversity, collapsing to generic templates that under-report rare yet critical findings. We identify this failure mode as Template Collapse. This failure stems from the unique constraints of 3D medical imaging, e.g., limited data, severe label imbalance, and weak signals from volumetric encoders. Under these constraints, text-generation objectives encourage shortcut learning and fluent but weakly grounded reports. We systematically diagnose the Template Collapse through clinical fidelity, output diversity, normal-template bias, and rare-finding survival. To mitigate it, we propose CLarGen, a decoupled framework that separates what to say (clinical detection) from how to say it (language synthesis). CLarGen uses (i) a Latent Query Transformer for multi-label pathology detection, (ii
    
[^268]: 自进化大语言模型的算力分配：从深度-广度到多臂老虎机

    Compute Allocation for Self-Evolving LLMs: From Depth-Breadth to Multi-Armed Bandits

    [https://arxiv.org/abs/2605.29268](https://arxiv.org/abs/2605.29268)

    该论文揭示了LLM进化搜索中算力分配的两条经验规律（适应度-算力包络线和双线性深度-广度拟合），并提出基于多臂老虎机的BaSE方法动态分配LLM调用预算，在不改变模型、提示或评估器的情况下将平均适应度提升12.3%。

    

    LLM引导的进化搜索（Evolve系统）已在数学与组合任务上取得了最先进的成果，但大多数现有系统仅报告多次运行中的最佳结果，而未记录运行间的分布情况。我们研究了固定预算的LLM调用应如何分配，以及单次运行达到所报告数值的可靠性如何。通过在五个模型和三个任务上扫描深度-广度网格，我们识别出两条经验规律：一条适应度-算力包络线，当以有效FLOPs衡量时，能力排序在很大程度上趋于坍缩；以及一个具有任务特定交互的双线性深度-广度拟合；两者均受模型-任务能力的制约。基于这些规律，我们提出了BaSE（基于老虎机的自进化），这是一种在并行轨迹间分配LLM调用的多臂老虎机方法。在不改变模型、提示或评估器的情况下，BaSE将平均适应度较最强基线提升了12.3%。

    arXiv:2605.29268v3 Announce Type: replace-cross  Abstract: LLM-guided evolutionary search (Evolve systems) has reached state-of-the-art results on mathematical and combinatorial tasks, yet most existing systems report only the best of many runs and leave the run-to-run distribution undocumented. We ask how a fixed budget of LLM calls should be allocated, and how reliably a single run reaches the reported numbers. Sweeping the depth-breadth grid over five models and three tasks, we identify two empirical regularities: a fitness-compute envelope along which capability ordering largely collapses when measured in effective FLOPs, and a bilinear depth-breadth fit with task-specific interaction; both are gated by model-task capability. Motivated by these regularities, we propose BaSE (Bandit-based Self-Evolving), a multi-armed bandit that allocates LLM calls across parallel trajectories. Without changing the model, prompt, or evaluator, BaSE improves mean fitness by 12.3% over the strongest 
    
[^269]: Mimir：大规模多语言概念建模

    Mimir: Large-scale Multilingual Concept Modeling

    [https://arxiv.org/abs/2605.25263](https://arxiv.org/abs/2605.25263)

    该论文提出了Mimir——一个16亿参数的多语言大型概念模型，突破了传统基于词元的语言建模范式，通过直接进行下一概念预测来实现多语言概念的理解与生成。

    

    当前的语言建模方法都是围绕词元构建的。文本语料库被切分为词元，模型通过在这些词元上执行计算来进行训练，例如在给定前文作为上下文的情况下预测下一个词元。这一范式已成为现代语言建模的标准，尤其是考虑到基于词元的架构所取得的杰出性能。然而，近期的研究不仅开始质疑语言模型如何从词元中处理和理解语义，也开始质疑使用更高层次的粒度是否能推动该研究领域的发展。这催生了“概念建模”的想法，即直接训练模型进行下一概念预测，而非下一词元预测。在这项工作中，我们介绍了Mimir，一个拥有16亿参数的大型概念模型，专为多语言概念理解与生成而训练。我们利用了大规模多语言预训练语料库（38,883,987,240个词元）……

    arXiv:2605.25263v2 Announce Type: replace-cross  Abstract: Current language modeling approaches are built around tokens. Text corpora are split into tokens, and models are trained by performing computations on these tokens, such as predicting the next token given the preceding ones as context. This paradigm has become the standard in modern language modeling, especially given the outstanding performance obtained by token-based architectures. However, recent works have not only begun to question how language models process and understand meaning from tokens, but also to question whether using higher levels of granularity could advance the research field. This led to the idea of Concept Modeling, that is, to directly train models for next-concept prediction rather than next-token prediction. In this work, we introduce Mimir, a 1.6B Large Concept Model trained for multilingual concept understanding and generation. We leverage a large-scale multilingual pre-training corpus (38,883,987,240 
    
[^270]: 转码器追踪视觉-语言模型中的视觉接地与幻觉现象

    Transcoders Trace Visual Grounding and Hallucinations in Vision-Language Models

    [https://arxiv.org/abs/2605.22902](https://arxiv.org/abs/2605.22902)

    本文提出基于转码器（Transcoders）的功能中心可解释性框架，能比稀疏自编码器更准确、更稳定地追踪视觉-语言模型中从图像块到文本生成的计算通路，并借此揭示了幻觉背后的错误视觉接地机制。

    

    生成式视觉-语言模型（VLMs）在多模态推理任务上表现出色，但视觉输入如何被转化为文本仍然缺乏理解。现有的VLM可解释性研究使用稀疏自编码器（SAEs），其只能分解静态残差表示，无法捕捉驱动跨模态交互的功能性更新。我们采用了一个以功能为中心的框架，基于转码器——即MLP子层的稀疏近似，作为逐层计算的因果代理。将该框架应用于Gemma 3-4B-IT模型时，它将模型分解为可解释的计算通路，将图像块与词元生成的方向联系起来。在补丁消融实验下，转码器归因对视觉接地词元产生了比SAE归因更强且更稳定的效应，并且与语义相关的图像区域对齐得更好。一项错误视觉接地（False Visual Grounding）反事实分析证实，所恢复的通路是……

    arXiv:2605.22902v2 Announce Type: replace-cross  Abstract: Generative Vision-Language Models (VLMs) perform well on multimodal reasoning, but how visual inputs are transformed to text remains poorly understood. Existing interpretability work on VLMs uses Sparse Autoencoders (SAEs), which decompose static residual representations and miss the functional updates that drive cross-modal interaction. We adopt a function-centric framework based on Transcoders, sparse approximations of MLP sublayers that act as a causal proxy for layer-wise computation. Applied to Gemma 3-4B-IT, the framework decomposes the model into interpretable computational pathways linking image patches to directions in token generation. Transcoder attributions produce stronger and more stable effects on visually grounded tokens under patch ablation than SAE attributions, and align better with semantically relevant image regions. A False Visual Grounding counterfactual analysis confirms that the recovered pathways are s
    
[^271]: 扮演魔鬼代言人：现成的人格向量在谄媚性缓解上可与针对性引导方法相媲美

    Playing Devil's Advocate: Off-the-Shelf Persona Vectors Rival Targeted Steering for Sycophancy

    [https://arxiv.org/abs/2605.21006](https://arxiv.org/abs/2605.21006)

    该研究发现无需专门针对谄媚性提取的现成批判性角色人格向量，就能将语言模型的谄媚性降低至专门方法CAA效果的68%-98%，且两者在几何上相互分离。

    

    谄媚性是指语言模型不顾正确性而一味迎合用户倾向的行为。先前的研究已经提取了谄媚性人格向量，并通过激活引导对该特质进行了因果控制（Chen et al., 2025; arXiv:2507.21509）。我们探究针对一般角色提取的、并非专门针对谄媚性的现成向量，是否能迁移到这一缓解任务中。我们在一个留出的、平衡对照的PhilPapers基准上，使用任务特定的系数调整，将批判性和从众性角色向量与针对谄媚性的对比激活添加（CAA）基线进行了比较。在Gemma 2 27B和Qwen 3 32B模型上，所选的批判性角色向量实现的平均谄媚性-logit降低分别约为CAA方法的68%和98%。从众性角色的效果较弱且存在异质性。角色向量与测量的CAA方向具有较低的绝对余弦相似度，在干预层建立了几何上的分离。

    arXiv:2605.21006v3 Announce Type: replace  Abstract: Sycophancy is the tendency of language models to agree with users irrespective of correctness. Prior work has extracted sycophancy persona vectors and causally controlled this trait through activation steering (Chen et al., 2025; arXiv:2507.21509). We ask whether existing vectors for general roles, extracted without targeting sycophancy, transfer to this mitigation task. We compare critical and conformist role vectors with a sycophancy-targeted Contrastive Activation Addition (CAA) baseline on a held-out, counterbalanced PhilPapers benchmark, using task-specific coefficient tuning. On Gemma 2 27B and Qwen 3 32B, the selected critical-role vectors achieve mean sycophancy-logit reductions approximately 68% and 98% as large as CAA's, respectively. Conformist-role effects are weak and heterogeneous. Role vectors have low absolute cosine similarity with the measured CAA direction, establishing geometric separation at the intervention laye
    
[^272]: 使用稀疏自编码器探究Whisper编码的可解释性

    On the Interpretability of Whisper Encodings Using Sparse Autoencoders

    [https://arxiv.org/abs/2605.12225](https://arxiv.org/abs/2605.12225)

    该研究利用稀疏自编码器首次揭示了Whisper语音识别模型编码器内部存在从语音到语义的丰富多层次语言表征，并通过因果引导实验发现高层特征比低层特征更易于操控，表明其编码信息远超任务所需。

    

    尽管基于深度Transformer的模型发展迅速，但其内部机制在很大程度上仍是一个谜。近期的研究优先关注理解基于文本的Transformer模型，而对语音识别（ASR）系统的探索相对匮乏。为了弥补这一空白，我们使用稀疏自编码器研究了Whisper编码器的内部表征。我们发现了跨越语言学与非语言学边界的多样化单语义特征，涵盖了从语音到语义表征的层次结构，并在该层次结构中开展了因果特征引导实验，包括跨语言引导。我们进一步发现，与低层特征相比，对高层特征的引导更为可靠，这种不对称性可能反映了低层信息的冗余编码。总而言之，这项工作表明，Whisper的编码器呈现出一个极其丰富的语言信息层次结构，其丰富程度远超任务本身所需的程度。

    arXiv:2605.12225v3 Announce Type: replace  Abstract: While deep transformer-based models have advanced rapidly, their internal mechanisms remain largely a mystery. Recent work has prioritized understanding text-based transformer models, leaving ASR systems largely unexplored. In order to address this gap, we examine the internal representations of Whisper's encoder using a sparse autoencoder. We find diverse monosemantic features across linguistic and non-linguistic boundaries, spanning a hierarchy from phonetic to semantic representations, and conduct a causal feature-steering campaign across this hierarchy, including cross-lingual steering. We further find that steering is more reliable for higher-level features than lower-level ones, an asymmetry that may reflect redundant encoding of lower-level information. Altogether, this work demonstrates that Whisper's encoder represents a surprisingly rich hierarchy of linguistic information that extends well beyond what is strictly necessary
    
[^273]: 将人类置于首位：基于人类偏好对齐的高效大型音频模型评估

    Putting HUMANS first: Efficient LAM Evaluation with Human Preference Alignment

    [https://arxiv.org/abs/2605.00022](https://arxiv.org/abs/2605.00022)

    该研究提出用仅50个样本（0.3%数据）的最小子集即可高效评估大型音频模型（与完整基准相关性超0.93），并通过在子集上训练回归模型将与人类偏好评分的相关性从0.85提升至0.98，实现了更贴合用户满意度的高效LAM评估方法。

    

    arXiv:2605.00022v2 公告类型：replace-cross 摘要：大型音频模型（LAMs）的快速普及需要高效的模型比较方法，然而全面的基准测试成本高昂。为填补这一空白，我们研究了最小子集是否能够在降低成本和数据冗余的同时可靠地评估LAMs。通过分析10种子集选择方法、18个音频模型以及涵盖主要LAM评估维度的40个任务，我们证明仅包含50个样本的子集（占数据的0.3%）即可与完整基准分数达到超过0.93的皮尔逊相关性。为了理解这些分数与从业者最终关心的用户满意度之间的对齐程度，我们从真实的语音助手对话中收集了776个人类偏好评分，发现子集和完整基准与人类偏好的相关性均仅为0.85。为了更好地预测人类偏好，我们在选定的子集上训练了回归模型，实现了0.98的相关性——优于回归模式……

    arXiv:2605.00022v2 Announce Type: replace-cross  Abstract: The rapid proliferation of large audio models (LAMs) demands efficient approaches for model comparison, yet comprehensive benchmarks are costly. To fill this gap, we investigate whether minimal subsets can reliably evaluate LAMs while reducing costs and data redundancy. Analyzing 10 subset selection methods with 18 audio models across 40 tasks covering major LAM evaluation dimensions, we show that subsets of just 50 examples (0.3% of data) can achieve over 0.93 Pearson correlation with full benchmark scores. To understand how well these scores align with what practitioners ultimately care about, user satisfaction, we collect 776 human preference ratings from realistic voice assistant conversations, finding that both subsets and full benchmark achieve only 0.85 correlation with human. To better predict preferences, we trained regression models on these selected subsets, achieving 0.98 correlation -- outperforming regression mode
    
[^274]: 无用但安全？基于多轮对话中用户意图澄清的效用恢复基准测试

    Useless but Safe? Benchmarking Utility Recovery with User Intent Clarification in Multi-Turn Conversations

    [https://arxiv.org/abs/2604.27093](https://arxiv.org/abs/2604.27093)

    该论文提出了首个交互式基准CarryOnBench和Ben-Util指标，用于评估LLM在多轮对话中能否在保持安全性的同时，根据用户意图澄清恢复有用性，发现模型首轮仅能满足10.5%至37.6%的用户良性信息需求。

    

    当前的LLM安全对齐技术提高了模型对对抗性攻击的鲁棒性，但忽视了当良性用户澄清其意图时，LLM是否以及如何恢复其有用性。我们提出了CarryOnBench，这是首个交互式基准测试，用于衡量LLM能否在多轮对话中修订对用户意图的理解并恢复效用，同时保持安全性。从398个看似有害但具有良性底层意图的查询出发，我们通过变换用户后续对话序列模拟了5,970个对话，在意图对齐的效用和安全性两个维度上评估了14个模型。CarryOnBench产生了1,866个不同的对话流（4至12轮），总计23,880个模型响应。我们设计了Ben-Util，这是一个基于清单的指标，通过原子化条目评估每个模型响应在多大程度上满足了用户的良性信息需求。在第一轮中，模型仅满足了用户良性信息需求的10.5%至37.6%。

    arXiv:2604.27093v2 Announce Type: replace-cross  Abstract: Current LLM safety alignment techniques improve model robustness against adversarial attacks, but overlook whether and how LLMs can recover helpfulness when benign users clarify their intent. We introduce CarryOnBench, the first interactive benchmark that measures whether LLMs can revise their interpretation of user intent and recover utility, while remaining safe through multi-turn conversations. Starting from 398 seemingly harmful queries with benign underlying intents, we simulate 5,970 conversations by varying user follow-up sequences, evaluating 14 models on both intent-aligned utility and safety. CarryOnBench yields 1,866 different conversation flows of 4--12 turns, totaling 23,880 model responses. We design Ben-Util, a checklist-based metric that evaluates how well each model response fulfills the user's benign information need using atomic items. At turn one, models fulfill only 10.5--37.6% of the user's benign informat
    
[^275]: 大语言模型中的交叉公平性

    Intersectional Fairness in Large Language Models

    [https://arxiv.org/abs/2604.20677](https://arxiv.org/abs/2604.20677)

    该研究系统评估了六个大语言模型的交叉公平性，发现模型在“种族-社会经济地位”和“种族-性别”两种属性交叉组合下表现出相反的刻板印象偏向，且子群体公平性指标揭示了表面低差异下隐藏的不均匀结果分布。

    

    大语言模型（LLM）日益被部署在社会敏感的场景中，引发了关于公平性和偏见的担忧，尤其是当多个敏感属性相互交叉时。我们使用来自问答偏见基准测试（BBQ）的两个数据集，系统性地评估了六个大语言模型中的交叉公平性，将种族与性别和社会经济地位相结合。我们评估了偏见、子群体公平性、准确性以及跨上下文和问题极性的一致性。虽然模型在模糊语境中表现良好，但稀疏的非“未知”预测限制了公平性评估。在无歧义语境中，刻板印象一致性对不同数据集的准确性产生不同影响：六个模型中有五个在“种族-社会经济地位”数据集中偏向强化刻板印象的项目，而在“种族-性别”数据集中则偏向反刻板印象的项目。子群体公平性指标揭示了不均匀的结果分布，尽管某些情况下观察到的差异较低，而重复运行显示出可变的结果。

    arXiv:2604.20677v3 Announce Type: replace  Abstract: Large Language Models (LLMs) are increasingly deployed in socially sensitive settings, raising concerns about fairness and bias, particularly when multiple sensitive attributes intersect. We systematically evaluate intersectional fairness in six LLMs using two datasets from the Bias Benchmark for Question Answering (BBQ), combining race with gender and socioeconomic status. We assess bias, subgroup fairness, accuracy, and consistency across contexts and question polarities. While models perform well in ambiguous contexts, sparse non-unknown predictions limit fairness evaluation. In disambiguated contexts, stereotype alignment affects accuracy differently across datasets: models favor stereotype-reinforcing items in Race-SES but counter-stereotype items in Race-Gender for five of six models. Subgroup fairness metrics reveal uneven outcome distributions despite low observed disparities in some cases, while repeated runs show variable r
    
[^276]: HumorGen：通过基于角色的蒸馏在大语言模型中实现幽默生成的认知协同框架

    HumorGen: Cognitive Synergy for Humor Generation in Large Language Models via Persona-Based Distillation

    [https://arxiv.org/abs/2604.09629](https://arxiv.org/abs/2604.09629)

    该论文提出受幽默心理学理论启发的认知协同框架，通过六种认知角色的思维混合方法合成理论支撑的幽默数据集来微调7B模型，且发现DPO和O-GRPO对齐策略未能超越SFT，但最终模型在幽默生成上显著超越更大的指令微调基线。

    

    幽默生成对大型语言模型（LLM）而言是一项重大挑战，因为其标准训练目标（下一词元预测）本质上与喜剧所需的意外性和不协调性相冲突。为了弥合这一差距，我们提出了认知协同框架，这是一种受幽默心理学理论启发的高质量幽默数据生成方法。利用思维混合方法，我们部署了六种认知角色（如“荒诞派”、“愤世嫉俗者”）来为给定提示合成多样化的喜剧视角。该框架产出了一个有理论依据的数据集，我们用它来微调一个70亿参数的学生模型。我们进一步评估了两种对齐策略——直接偏好优化（DPO）和离线组相对变体O-GRPO，发现两者均未能超越监督微调（SFT）的效果。然而，我们的70亿参数HumorGen模型变体显著优于更大的指令微调基线模型，并取得了...

    arXiv:2604.09629v3 Announce Type: replace  Abstract: Humor generation poses a significant challenge for Large Language Models (LLMs), because their standard training objective (next-token prediction) inherently conflicts with the surprise and incongruity required for comedy. To bridge this gap, we introduce the Cognitive Synergy Framework, a methodology for generating highquality humor data inspired by psychological theories of humor. Utilizing a Mixtureof-Thought (MoT) approach, we deploy six cognitive personas (e.g., The Absurdist, The Cynic) to synthesize diverse comedic perspectives for a given prompt. This framework produces a theory-grounded dataset, which we use to fine-tune a 7B-parameter student model. We further evaluate two alignment strategies, Direct Preference Optimization (DPO) and an offline group-relative variant O-GRPO, finding that neither improves over SFT. However, our 7B HumorGen model variants significantly outperform larger instruction-tuned baselines and achiev
    
[^277]: SatIR：面向临床试验匹配的可扩展高召回率约束满足式信息检索方法

    SatIR: Scalable High-Recall Constraint-Satisfaction-Based Information Retrieval for Clinical Trials Matching

    [https://arxiv.org/abs/2604.08849](https://arxiv.org/abs/2604.08849)

    该论文提出SatIR，一种基于形式化约束满足的可扩展、高精确率、高召回率且可解释的临床试验匹配检索方法，将患者入组资格约束视为强制性要求而非软信号，从而克服了传统关键词和嵌入相似度匹配方法召回率低、精确率低、可解释性差的局限。

    

    现实世界中的许多检索与匹配问题所要求的不止是主题相关性：候选者必须满足众多档案中某一个的特定约束条件，而不仅仅是与该档案主题相关。临床试验是这一挑战的一个高风险实例：临床试验是循证医学的核心，然而尽管ClinicalTrials.gov上列出了超过五十万项临床试验，每月吸引约两百万用户，许多试验仍难以达到入组目标。现有的检索技术主要基于关键词匹配和嵌入相似度匹配，将资格约束视为软信号而非强制性要求，导致召回率低、精确率低且可解释性有限。我们提出了SatIR，这是一种基于形式化约束满足的可扩展、高效、高精确率、高召回率且可解释的临床试验检索方法。利用成熟的医学本体，我们使用大型语言模型……

    arXiv:2604.08849v4 Announce Type: replace-cross  Abstract: Many real-world retrieval and matching problems require more than topical relevance: a candidate must satisfy the specific constraints of one profile among many, not just be relevant to it. Clinical trials are a high-stakes instance of this challenge: they are central to evidence-based medicine, yet many struggle to meet enrollment targets, despite the availability of over half a million trials listed on ClinicalTrials.gov, which attracts approximately two million users monthly. Existing retrieval techniques, largely based on keyword and embedding-similarity matching, treat eligibility constraints as soft signals rather than binding requirements, resulting in low recall, low precision, and limited interpretability.   We propose SatIR, a scalable, efficient, high-precision, high-recall, interpretable clinical trial retrieval method based on formal constraint satisfaction. Leveraging established medical ontologies, we use Large L
    
[^278]: 我们还能追踪母语信号吗？探究LLM时代母语信号的韧性

    Can We Still Trace L1 Signals? Investigating the Resilience of Native Language Signals in the LLM Era

    [https://arxiv.org/abs/2604.08568](https://arxiv.org/abs/2604.08568)

    本研究通过构建覆盖神经网络前、LLM前和LLM后三个时代、八个母语群体的学术摘要母语识别数据集，发现文本中的母语信号随时间持续减弱，且英语同质化的主要趋势早在LLM出现之前就已开始。

    

    基于LLM的写作辅助工具的广泛使用引发了一个关于英语同质化的有趣问题。由于LLM倾向于将文本修改为其训练数据中所反映的主流英语规范，反映作者母语（L1）的微妙特征可能正在逐渐消失。本研究通过分析学术摘要上的母语识别（NLI）性能来研究这一现象。为此，我们构建了两个从arXiv和ACL Anthology提取的学术摘要母语识别数据集，涵盖了神经网络（NN）前时代、LLM前时代和LLM后时代三个时期的八个母语群体。然后，我们使用通过微调LLM获得的NLI分类器评估每个时代的NLI性能。结果显示NLI性能随时间持续下降。然而值得注意的是，从神经网络前时代到LLM前时代的性能下降比从LLM前时代到LLM后时代的下降更为明显。

    arXiv:2604.08568v3 Announce Type: replace-cross  Abstract: The widespread use of LLM-based writing assistance has raised an interesting question about the homogenization of English. As LLMs tend to revise texts toward mainstream English conventions reflected in their training data, the subtle fingerprints that reflect an author's native language (L1) may be gradually disappearing. This study investigates this phenomenon by analyzing native language identification (NLI) performance on academic abstracts. To this end, we construct two NLI datasets of academic abstracts extracted from arXiv and the ACL Anthology that covers eight native language groups across three time periods: pre-neural network (NN), pre-LLM, and post-LLM. We then evaluate NLI performance for each era using NLI classifiers obtained by fine-tuning LLMs. The results reveal a consistent decline in NLI performance over time. Notably, however, the decline is more pronounced from the pre-NN era to the pre-LLM era than from t
    
[^279]: 测试大语言模型中真相方向的极限

    Testing the Limits of Truth Directions in LLMs

    [https://arxiv.org/abs/2604.03754](https://arxiv.org/abs/2604.03754)

    大语言模型中的“真相方向”并非普适：它高度依赖于网络层级、任务类型（事实性任务出现在较早层级，推理任务出现在较晚层级）以及模型所接收的指令。

    

    大语言模型（LLM）已被证明会在其激活空间中沿着一个线性的“真相方向”来编码陈述的真值。先前的研究认为这些方向在某些方面是普适的，而较新的研究则基于其在部分场景下有限的泛化能力，对这一结论提出了质疑。在这项工作中，我们识别出了真相方向普适性的若干此前尚未被理解的极限。我们首先表明，真相方向高度依赖于网络层级，因此要全面理解其普适性，需要在模型的多个层级上进行探测。随后我们表明，真相方向在很大程度上取决于任务类型——事实性任务的真相方向出现在较早的层级，而推理任务的真相方向则出现在较晚的层级；此外，它们在任务复杂度的不同水平上也表现出性能差异。最后，我们表明模型指令会影响真相方向；简单的正确性评估指令会显著影响……

    arXiv:2604.03754v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) have been shown to encode truth of statements in their activation space along a linear truth direction. Previous studies have argued that these directions are universal in certain aspects, while more recent work has questioned this conclusion drawing on limited generalization across some settings. In this work, we identify a number of limits of truth-direction universality that have not been previously understood. We first show that truth directions are highly layer-dependent, and that a full understanding of universality requires probing at many layers in the model. We then show that truth directions depend heavily on task type, emerging in earlier layers for factual and later layers for reasoning tasks; they also vary in performance across levels of task complexity. Finally, we show that model instructions can affect truth directions; simple correctness evaluation instructions significantly affect
    
[^280]: 我的模型的困惑是否出于正确的原因？将LLM的基准测试行为与词元级困惑度进行对比

    Is my model perplexed for the right reason? Contrasting LLMs' Benchmark Behavior with Token-Level Perplexity

    [https://arxiv.org/abs/2603.29396](https://arxiv.org/abs/2603.29396)

    本文提出一种基于词元级困惑度的可解释性框架，通过对比仅在一个或少数关键词条上存在差异的最小句子对的困惑度分布，来检验LLM是否真正依赖语言相关线索，发现语言重要词条虽会影响模型行为但无法完全解释困惑度变化，表明模型依赖的是非预期的启发式机制。

    

    对大型语言模型（LLM）的标准评估主要关注任务性能，对于正确行为是否反映了适当的底层机制提供的洞察有限，并存在确认偏误的风险。我们引入了一个基于词元级困惑度的简单且具有原则性的可解释性框架，用于测试模型是否依赖与语言相关的线索。通过比较仅在单个或少数“关键”词元上存在差异的最小句子对上的困惑度分布，我们的方法能够进行精确的、假设驱动的分析，而无需依赖不稳定的特征归因技术。在多个开放权重LLM上进行的受控语言基准实验表明，虽然语言学上重要的词元会影响模型行为，但它们从未能完全解释困惑度的变化，这揭示了模型依赖于预期语言机制之外的其他启发式方法。

    arXiv:2603.29396v2 Announce Type: replace  Abstract: Standard evaluations of Large language models (LLMs) focus on task performance, offering limited insight into whether correct behavior reflects appropriate underlying mechanisms and risking confirmation bias. We introduce a simple, principled interpretability framework based on token-level perplexity to test whether models rely on linguistically relevant cues. By comparing perplexity distributions over minimal sentence pairs differing in one or a few `pivotal' tokens, our method enables precise, hypothesis-driven analysis without relying on unstable feature-attribution techniques. Experiments on controlled linguistic benchmarks with several open-weight LLMs show that, while linguistically important tokens influence model behavior, they never fully explain perplexity shifts, revealing that models rely on heuristics other than the expected linguistic ones.
    
[^281]: AI精神病：对话式AI是否会放大与妄想相关的语言？

    AI Psychosis: Does Conversational AI Amplify Delusion-Related Language?

    [https://arxiv.org/abs/2603.19574](https://arxiv.org/abs/2603.19574)

    本研究基于Reddit用户历史发帖构建模拟用户，并提出DelusionScore语言度量方法，实证发现具有妄想倾向的用户在与GPT、LLaMA和Qwen等对话式AI的多轮交互中，妄想相关语言强度会逐渐增强，为“AI精神病”现象提供了实证依据。

    

    对话式AI系统越来越多地被用于个人反思和情感倾诉，这引发了人们对其对脆弱用户潜在影响的担忧。近期的一些轶事报告表明，与AI的长期交互可能会强化妄想性思维——这种现象有时被称为“AI精神病”。然而，关于这一现象的实证证据仍然有限。在本研究中，我们考察了在与对话式AI进行多轮交互的过程中，妄想相关语言是如何演变的。我们基于Reddit用户的纵向发帖历史构建了模拟用户，并使用三个模型家族生成了扩展对话。我们开发了DelusionScore，这是一种语言学度量方法，用于量化对话各轮次中妄想相关语言的强度。我们发现，源自具有先前妄想相关话语的用户（实验组）的模拟用户表现出DelusionScore的逐渐升高。

    arXiv:2603.19574v2 Announce Type: replace-cross  Abstract: Conversational AI systems are increasingly used for personal reflection and emotional disclosure, raising concerns about their effects on vulnerable users. Recent anecdotal reports suggest that prolonged interactions with AI may reinforce delusional thinking---a phenomenon sometimes described as AI Psychosis. However, empirical evidence on this phenomenon remains limited. In this work, we examine how delusion-related language evolves during multi-turn interactions with conversational AI. We construct simulated users (SimUsers) from Reddit users' longitudinal posting histories and generate extended conversations with three model families (GPT, LLaMA, and Qwen). We develop DelusionScore, a linguistic measure that quantifies the intensity of delusion-related language across conversational turns. We find that SimUsers derived from users with prior delusion-related discourse (Treatment) exhibit progressively increasing DelusionScore
    
[^282]: 看见不等于掌握：教会大语言模型使用私有库进行代码生成

    To See is Not to Master: Teaching LLMs to Use Private Libraries for Code Generation

    [https://arxiv.org/abs/2603.15159](https://arxiv.org/abs/2603.15159)

    针对LLMs即使获得准确文档知识仍难以调用私有库API的问题，提出PriCoder方法，通过将数据合成建模为图构建并交替使用渐进式图演化等图算子来自动合成多样化训练数据，教会模型有效调用私有库API进行代码生成。

    

    大语言模型（LLMs）在代码生成方面展现出强大的潜力，但在面向私有库的代码生成任务上仍然存在局限，该任务的目标是使用私有库中的API来生成代码。现有方法主要依赖于检索私有库API文档，并在推理时将相关知识注入到上下文中。然而，我们的研究表明这样做是不够的：即使提供了准确且完整的所需知识，LLMs仍然难以有效地调用私有库API。为了解决这一局限，我们提出了PriCoder，一种通过自动合成数据来教会LLMs调用私有库API的方法。具体而言，PriCoder将私有库数据合成建模为图的构建过程，并在两种图算子之间交替运行：（1）渐进式图演化，通过在基础图上逐步合成更加多样化的训练样本来提升数据多样性

    arXiv:2603.15159v5 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have shown strong potential for code generation, yet they remain limited in private-library-oriented code generation, where the goal is to generate code using APIs from private libraries. Existing approaches mainly rely on retrieving private-library API documentation and injecting relevant knowledge into the context at inference time. However, our study shows that this is insufficient: even given accurate required knowledge, LLMs still struggle to invoke private-library APIs effectively.   To address this limitation, we propose PriCoder, an approach that teaches LLMs to invoke private-library APIs through automatically synthesized data. Specifically, PriCoder models private-library data synthesis as the construction of a graph, and alternates between two graph operators: (1) Progressive Graph Evolution, which improves data diversity by progressively synthesizing more diverse training samples from ba
    
[^283]: 语言与翻译行业的技术导向图谱：分析利益相关者价值观及其对翻译教学的潜在启示

    A technology-oriented mapping of the language and translation industry: Analysing stakeholder values and their potential implication for translation pedagogy

    [https://arxiv.org/abs/2603.11667](https://arxiv.org/abs/2603.11667)

    该研究基于29位行业利益相关者的访谈，发现在自动化生产环境中，效率导向的技术价值观已成为行业基线期望，而人类价值并未被取代，而是通过专业知识、监督、问责和情境判断被重新定位，这一发现对翻译教学具有重要启示。

    

    本文考察了在当今日益自动化的语言与翻译行业中，价值是如何被建构和协商的。基于LT-LiDER项目中收集的29位行业利益相关者的访谈数据，本研究分析了人类价值、技术价值、效率和适应性在不同职业角色中是如何被阐述的。本文以切斯特曼的翻译伦理及相关价值观框架为分析视角，表明与服务伦理相契合的效率导向型技术价值观已成为自动化生产环境中的基线期望——在这种环境中，速度、可扩展性和交付能力主导着评价标准。与此同时，人类价值并未被取代，而是被重新定位，主要通过嵌入在技术中介工作流程中的专业知识、监督、问责和情境判断得以体现。一个核心发现是……（原文此处截断）

    arXiv:2603.11667v3 Announce Type: replace  Abstract: This paper examines how value is constructed and negotiated in today's increasingly automated language and translation industry. Drawing on interview data from twenty-nine industry stakeholders collected within the LT-LiDER project, the study analyses how human value, technological value, efficiency, and adaptability are articulated across different professional roles. Using Chesterman's framework of translation ethics and associated values as an analytical lens, the paper shows that efficiency-oriented technological values aligned with the ethics of service have become baseline expectations in automated production environments, where speed, scalability, and deliverability dominate evaluation criteria. At the same time, human value is not displaced but repositioned, emerging primarily through expertise, oversight, accountability, and contextual judgment embedded within technology-mediated workflows. A central finding is the prominenc
    
[^284]: 后见之明锚定策略优化：通过后见之明学习与受汤普森采样启发的自适应门控

    Hindsight-Anchored Policy Optimization: Learning Through Hindsight with Thompson Sampling-Inspired Adaptive Gating

    [https://arxiv.org/abs/2603.11321](https://arxiv.org/abs/2603.11321)

    提出后见之明锚定策略优化（HAPO），利用Beta-二项置信自适应门控决定教师干预时机，结合合成成功注入与自适应阈值退火作为临时支撑，解决了稀疏奖励下混合策略强化学习的训练崩溃与不稳定问题。

    

    带可验证奖励的强化学习提升了大语言模型的推理能力，然而在稀疏奖励设置下，在策略学习常常面临冷启动挑战。近期的混合策略方法通过将离策略教师数据与在策略训练相结合来解决这一问题。然而，简单地将二者结合会引入持续存在的离策略梯度分量，存在训练崩溃和不稳定的风险。为解决这一挑战，我们提出了后见之明锚定策略优化（HAPO），这是一个允许教师干预充当临时支撑的框架。HAPO 采用 Beta-二项置信门控，这是一种自适应门控机制，用于决定何时为教师干预打开大门。该干预通过合成成功注入来运行，即用经过验证的教师轨迹替换组内奖励最低的 rollout。我们还引入了自适应阈值退火，逐步撤回这种支撑。

    arXiv:2603.11321v3 Announce Type: replace-cross  Abstract: Reinforcement Learning with Verifiable Rewards improves reasoning in large language models, yet on-policy learning often suffers from cold-start challenges in sparse-reward settings. Recent mixed-policy approaches address this by combining off-policy teacher data with on-policy training. However, simply combining these introduce a persistent off-policy gradient mass that risks training collapse and instability. To address this challenge, we propose Hindsight-Anchored Policy Optimization (HAPO), a framework that allows teacher intervention to act as a temporary support. HAPO employs Beta-Binomial confidence gating, an adaptive gating mechanism that decides when to open the gate for teacher intervention. The intervention operates with Synthetic Success Injection, which replaces the group's lowest-reward rollout with a verified teacher trajectory. We also introduce adaptive threshold annealing, which gradually retracts the support
    
[^285]: 对比解码如何增强大型音频语言模型

    How Contrastive Decoding Enhances Large Audio Language Models

    [https://arxiv.org/abs/2603.09232](https://arxiv.org/abs/2603.09232)

    对比解码能可靠地纠正大型音频语言模型中虚假否认音频存在或因不确定性而猜测的错误，但难以纠正错误推理和自信的错误断言，其收益大小取决于模型基线错误的构成。

    

    虽然对比解码（CD）已被提出用于增强大型音频语言模型（LALMs），但它尚未经过大规模评估，其成功的底层机制仍不清晰。本研究系统性地在多种LALM架构上评估了四种不同的CD策略。我们确定了音频感知解码和音频对比解码是最有效的方法。然而，它们的效果在不同模型之间存在显著差异。为了解释这种差异，我们分析了每个模型的基线错误构成，并测量了对比解码纠正各类错误的难易程度。我们的分析表明，CD能够可靠地纠正模型虚假声称音频不存在或基于不确定性进行猜测这类错误，但在纠正有缺陷的推理或自信的错误断言方面效果较差。至关重要的是，CD的收益与模型基线错误构成密切相关：当模型……

    arXiv:2603.09232v2 Announce Type: replace-cross  Abstract: While Contrastive Decoding (CD) has been proposed to enhance Large Audio Language Models (LALMs), it has not been evaluated at scale, and the underlying mechanisms driving its success remain unclear. This study systematically evaluates four distinct CD strategies across diverse LALM architectures. We identify Audio-Aware Decoding and Audio Contrastive Decoding as the most effective methods. However, their impact varies significantly across models. To explain this variability, we profile the baseline error composition of each model and measure how readily contrastive decoding corrects each error type. Our analysis demonstrates that CD reliably rectifies errors in which models falsely claim an absence of audio or resort to uncertainty-driven guessing, but is relatively poor at correcting flawed reasoning or confident misassertions. Crucially, CD's benefit closely tracks the composition of a model's baseline error profile: when er
    
[^286]: TimeWarp：通过重访过去来评估网页智能体

    TimeWarp: Evaluating Web Agents by Revisiting the Past

    [https://arxiv.org/abs/2603.04949](https://arxiv.org/abs/2603.04949)

    提出了模拟网页UI演变的TimeWarp基准测试，揭示现有网页智能体对界面变化的脆弱性，并通过TimeTraj标注方法与BC变体训练显著提升了智能体跨网页版本的泛化能力。

    

    随着网页智能体在基准测试中逐渐缩小与人类的差距，一个问题随之而来：今天的智能体在明天的网页上是否也能表现得同样出色？我们提出了TimeWarp，一个模拟不断演变的网页环境的基准测试。TimeWarp包含三个网页环境，每个环境有六个UI版本，涵盖了互联网不同时代的UI设计、前端代码和工作流程。我们为TimeWarp配套了一系列复杂、真实的任务，涵盖不同形式的网页导航。我们的实验表明，基于视觉的智能体容易受到网页变化的影响，而基于文本的智能体一旦在单一版本上微调就会变得脆弱。为了解决这个问题，我们提出了TimeTraj，一种新的标注方法，通过计划蒸馏来收集跨多个版本的轨迹。通过使用我们的BC变体方法在教师rollout上训练智能体，我们取得了显著的性能提升：Qwen-3 4B模型从20.4%提升到37.7%，Llama-3.1 8B模型从0%提升到27.0%。

    arXiv:2603.04949v2 Announce Type: replace  Abstract: As web agents close the gap with humans on benchmarks, one question arises: Do today's agents perform just as well on tomorrow's web? We introduce TimeWarp, a benchmark that emulates the evolving web. TimeWarp consists of three web environments, each with six UI versions spanning UI design, frontend code, and workflows from different eras of the internet. We pair TimeWarp with a set of complex, realistic tasks covering different forms of web navigation. Our experiments reveal that vision-based agents are vulnerable to changes, while text-based agents become brittle once fine-tuned on a single version. To address this, we propose TimeTraj, a new annotation method that uses plan distillation to collect trajectories across multiple versions. By training agents on teacher rollouts using our BC-variant, we achieve substantial performance gains: 20.4% to 37.7% for Qwen-3 4B and 0% to 27.0% for Llama-3.1 8B models. Our work helps study gene
    
[^287]: 视觉-语言模型中情境依赖的功能可供性报告

    Context-Dependent Affordance Reports in Vision-Language Models

    [https://arxiv.org/abs/2603.04419](https://arxiv.org/abs/2603.04419)

    该研究审计了先前关于角色提示影响视觉-语言模型描述的研究，发现空物体列表响应导致原分析低估了相似度，并撤回了原研究的功能性流形解释，转而通过新的对照实验重新审视该效应。

    

    视觉-语言模型在不同的角色提示下会产生不同的物体和使用描述，但仅凭低重叠度并不能确认存在功能可供性效应。我们对早前的一项七提示研究进行了审计，并添加了匹配问题的对照组。在此前的Qwen试点实验中，3,213条解析响应中有363条包含空的物体列表，这影响了9,244次比较中的2,037次，且程序实现将零词汇重叠度赋给了每个受影响的配对。以非空报告为条件进行分析后，汇总的词级Jaccard相似度从0.095提升至0.121，句子余弦相似度从0.415提升至0.511。先前被命名的“厨师特异性”Tucker因子在缺失单元格和完全非空分析下失去了其集中载荷。我们撤回功能性流形解释以及将相似度分数转换为意义百分比的做法。一项新实验使用了原试点中未出现的48张图像、四种角色、共享的三物体任务、两种措辞，以及……

    arXiv:2603.04419v3 Announce Type: replace-cross  Abstract: Vision-language models produce different object and use descriptions under different persona prompts, but low overlap alone does not identify an affordance effect. We audit an earlier seven-prompt study and add matched-question controls. In the historical Qwen pilot, 363 of 3,213 parsed responses contain empty object lists. These affect 2,037 of 9,244 comparisons, with the implementation assigning zero lexical overlap to every affected pair. Conditioning on nonempty reports raises pooled word Jaccard from 0.095 to 0.121 and sentence cosine from 0.415 to 0.511. A previously named chef-specific Tucker factor loses its concentrated loading under missing-cell and complete-nonempty analyses. We withdraw the functional-manifold interpretation and the conversion of similarity scores into percentages of meaning. A new experiment uses 48 images absent from the original pilot, four personas, a shared three-object task, two wordings, and 
    
[^288]: 通过检索-过渡头连接潜在推理与目标语言生成

    Bridging Latent Reasoning and Target-Language Generation via Retrieval-Transition Heads

    [https://arxiv.org/abs/2602.22453](https://arxiv.org/abs/2602.22453)

    本文在多语言大模型中识别出一类区别于检索头的检索-过渡头（RTH），它们控制着模型向特定目标语言输出的过渡，且对思维链推理更为关键——屏蔽RTH会在四个多语言基准和两个模型系列上引起比屏蔽检索头更大的性能下降。

    

    arXiv:2602.22453v3 公告类型：替换 摘要：最近的研究发现，Transformer中存在一类注意力头，称为检索头，它们负责从上下文中检索信息。在本工作中，我们首先研究了多语言语境下的检索头。在多语言语言模型中，我们发现检索头通常在多种语言之间是共享的。将研究扩展到跨语言设置后，我们识别出检索-过渡头，它控制着向特定目标语言输出的过渡。我们的实验表明，RTH与检索头不同，并且在多语言大模型的思维链推理中更为关键。在四个多语言基准（MMLU-ProX、MGSM、MLQA和XQuaD）以及两个模型系列（Qwen-2.5和Llama-3.1）上，我们证明了屏蔽RTH比屏蔽检索头（RH）会导致更大的性能下降。我们的工作通过分离负责潜在推理向目标语言生成过渡的注意力头，推进了对多语言语言模型的理解。

    arXiv:2602.22453v3 Announce Type: replace  Abstract: Recent work has identified a subset of attention heads in Transformer as retrieval heads, which are responsible for retrieving information from the context. In this work, we first investigate retrieval heads in multilingual contexts. In multilingual language models, we find that retrieval heads are often shared across multiple languages. Expanding the study to cross-lingual setting, we identify Retrieval-Transition heads(RTH), which govern the transition to specific target-language output. Our experiments reveal that RTHs are distinct from retrieval heads and more vital for Chain-of-Thought reasoning in multilingual LLMs. Across four multilingual benchmarks (MMLU-ProX, MGSM, MLQA, and XQuaD) and two model families (Qwen-2.5 and Llama-3.1), we demonstrate that masking RTH induces bigger performance drop than masking Retrieval Heads (RH). Our work advances understanding of multilingual LMs by isolating the attention heads responsible f
    
[^289]: 评分标准作为攻击面：大语言模型评审中的隐蔽偏好漂移

    Rubrics as an Attack Surface: Stealthy Preference Drift in LLM Judges

    [https://arxiv.org/abs/2602.13576](https://arxiv.org/abs/2602.13576)

    该论文首次揭示了一种名为“评分标准诱导的偏好漂移”（RIPD）的漏洞：通过基准验证的看似无害的评分标准修改，可以隐蔽且系统性地偏移大语言模型评审器的偏好，从而构成难以被基准指标或抽查检测到的偏好攻击。

    

    大语言模型的评估与对齐流程日益依赖于基于大语言模型的评审器，其行为由自然语言评分标准引导，并在基准测试上进行验证。我们识别出该工作流程中一个此前未被充分认识的漏洞，将其命名为“评分标准诱导的偏好漂移”。即使评分标准的修改通过了基准验证，它们仍可能在目标领域上对评审器的偏好产生系统性和方向性的偏移。由于评分标准充当着高层决策接口的角色，这种漂移可能源于看似自然、保持评判准则不变的修改，并且难以通过总体基准指标或有限的抽查检测出来。我们进一步表明，这一漏洞可以通过基于评分标准的偏好攻击加以利用：符合基准要求的评分标准修改会使目标领域上的评判偏离固定的人类或可信参考标准，系统性地诱导偏好漂移并降低……

    arXiv:2602.13576v2 Announce Type: replace-cross  Abstract: Evaluation and alignment pipelines for large language models increasingly rely on LLM-based judges, whose behavior is guided by natural-language rubrics and validated on benchmarks. We identify a previously under-recognized vulnerability in this workflow, which we term Rubric-Induced Preference Drift (RIPD). Even when rubric edits pass benchmark validation, they can still produce systematic and directional shifts in a judge's preferences on target domains. Because rubrics serve as a high-level decision interface, such drift can emerge from seemingly natural, criterion-preserving edits and remain difficult to detect through aggregate benchmark metrics or limited spot-checking. We further show this vulnerability can be exploited through rubric-based preference attacks, in which benchmark-compliant rubric edits steer judgments away from a fixed human or trusted reference on target domains, systematically inducing RIPD and reducing
    
[^290]: 基于患者特异性条件的可控构音障碍语音合成用于说话人多样化的ASR数据增强

    Controllable Dysarthric Speech Synthesis with Patient-Specific Conditioning for Speaker-Diverse ASR Augmentation

    [https://arxiv.org/abs/2602.08696](https://arxiv.org/abs/2602.08696)

    提出了一种可控构音障碍语音合成框架，通过解耦音色与患者特异性病理发音条件，可将病理特征与不同目标说话人（含健康人）灵活组合，生成的语音可有效增强构音障碍语音识别系统的训练数据。

    

    构音障碍语音识别受限于较高的说话人变异性和稀缺的标注数据。现有的语音合成方法往往将说话人身份与构音障碍的发音特点耦合在一起，降低了对生成语音的控制能力。我们提出了一种面向ASR数据增强的可控构音障碍语音合成框架，该框架采用相互独立的基于提示的音色前缀和可学习的患者特异性病理前缀。该框架构建于使用LoRA适配的预训练神经编解码语言模型之上，通过加性条件机制将两种前缀相结合。通过引入带梯度反转的双分类器目标函数以及基于语音转换的反事实数据增强，实现了语音因素的解耦，使得学习到的病理条件能够与不同的目标说话人（包括健康说话人）进行配对。在TORGO数据集上的实验表明，生成的语音可以部分替代真实的构音障碍训练数据，并提供有效的说话人多样化数据增强。

    arXiv:2602.08696v3 Announce Type: replace-cross  Abstract: Dysarthric speech recognition is limited by high speaker variability and scarce labeled data. Existing synthesis methods often couple speaker identity with dysarthric articulation, reducing control over generated speech. We propose a controllable dysarthric speech synthesis framework for ASR augmentation with separate prompt-derived timbre prefixes and learnable patient-specific pathology prefixes. Built on a pre-trained neural codec language model adapted using LoRA, the framework combines both prefixes through additive conditioning. A dual-classifier objective with gradient reversal and voice-conversion-based counterfactual augmentation promotes factor separation, allowing learned pathology conditions to be paired with different target speakers, including healthy speakers. Experiments on TORGO show that generated speech can partially replace real dysarthric training data and provides effective speaker-diverse augmentation whe
    
[^291]: 在代码中做梦：面向开放式世界的课程学习

    Dreaming in Code for Curriculum Learning in Open-Ended Worlds

    [https://arxiv.org/abs/2602.08194](https://arxiv.org/abs/2602.08194)

    提出DiCode框架，利用大语言模型“做梦”般地合成可执行的环境代码变体，在开放式世界中构建课程，引导智能体获得持续可学习并逐步提升能力的经验序列。

    

    开放式学习将智能视为在与不断扩展的环境空间的持续交互中涌现的过程。尽管近期的进展利用基础模型以程序化方式生成多样化的环境，这些方法往往侧重于发现孤立的行为，而非编排持续的进展。在复杂的开放式世界中，可能挑战的巨大组合空间使得智能体难以发现那些能够保持持续可学习性的经验序列。为解决这一问题，我们提出了Dreaming in Code（DiCode），这是一个无监督环境设计（UED）框架，其中基础模型（大语言模型）通过合成可执行的环境代码来搭建脚手架，引导学习走向不断提升的能力。在DiCode中，“做梦”以具体化世界的代码级变化的形式呈现。我们在Craftax中实例化了DiCode，这是一个具有挑战性的开放式强化学习环境……

    arXiv:2602.08194v2 Announce Type: replace-cross  Abstract: Open-ended learning frames intelligence as emerging from continual interaction with an ever-expanding space of environments. While recent advances have utilized foundation models to programmatically generate diverse environments, these approaches often focus on discovering isolated behaviors rather than orchestrating sustained progression. In complex open-ended worlds, the large combinatorial space of possible challenges makes it difficult for agents to discover sequences of experiences that remain consistently learnable. To address this, we propose Dreaming in Code (DiCode), an unsupervised environment design (UED) framework in which foundation models (large language models) synthesize executable environment code to scaffold learning toward increasing competence. In DiCode, "dreaming" takes the form of materializing code-level variations of the world. We instantiate DiCode in Craftax, a challenging open-ended reinforcement lea
    
[^292]: DecompressionLM：从语言模型中进行确定性、诊断性和零样本的概念图提取

    DecompressionLM: Deterministic, Diagnostic, and Zero-Shot Concept Graph Extraction from Language Models

    [https://arxiv.org/abs/2602.00377](https://arxiv.org/abs/2602.00377)

    DecompressionLM是一个无状态框架，利用Van der Corput低差异序列与算术解码，在无需预设查询的情况下，实现确定性和高度并行的零样本概念图提取，以发现语言模型中编码的概念，并揭示了量化（如AWQ-4bit）可显著扩大概念覆盖范围。

    

    现有的知识探测方法依赖于预先定义的查询，这将提取范围限制在已知概念上。我们提出了DecompressionLM，这是一个用于零样本概念图提取的无状态框架，能够在无需预先指定查询或跨序列共享状态的情况下，发现语言模型所编码的内容。我们的方法针对基于解码的常见探测方法的三个局限性：（i）跨序列耦合导致概率质量集中于高频前缀；（ii）竞争性解码效应抑制了长尾概念；（iii）顺序探索带来的可扩展性限制。通过使用Van der Corput低差异序列与算术解码，DecompressionLM实现了确定性的、高度可并行的生成，且无需跨序列共享状态。在两个模型家族和五个量化变体上，我们发现激活感知量化（AWQ-4bit）将概念覆盖率扩大了30-170%……

    arXiv:2602.00377v3 Announce Type: replace  Abstract: Existing knowledge probing methods rely on pre-defined queries, limiting extraction to known concepts. We introduce DecompressionLM, a stateless framework for zero-shot concept graph extraction that discovers what language models encode without pre-specified queries or shared cross-sequence state. Our method targets three limitations of common decoding-based probing approaches: (i) cross-sequence coupling that concentrates probability mass on high-frequency prefixes, (ii) competitive decoding effects that suppress long-tail concepts, and (iii) scalability constraints arising from sequential exploration. Using Van der Corput low-discrepancy sequences with arithmetic decoding, DecompressionLM enables deterministic, embarrassingly parallel generation without shared state across sequences. Across two model families and five quantization variants, we find that activation-aware quantization (AWQ-4bit) expands concept coverage by 30-170%, w
    
[^293]: 欧登塞大学（SDU）的DAISY：丹麦文化基准测试

    SDUs DAISY: A Benchmark for Danish Culture

    [https://arxiv.org/abs/2601.19930](https://arxiv.org/abs/2601.19930)

    该论文基于2006年《丹麦文化经典》构建了丹麦文化事实知识基准Daisy，通过人工验证的741个封闭式问答对，评估当代语言模型对丹麦文化遗产（包括深层次、非主流内容）的掌握程度。

    

    我们介绍了Daisy，一个基于2006年《丹麦文化经典》精选主题构建的丹麦文化遗产事实知识基准测试。对于文化经典中的每一件作品，我们查询相应的维基百科页面，并让语言模型生成多样化的 问题集合。在每件作品范围内，我们同时采样核心问题和边缘问题，不仅测试主流信息，还测试经典委员会所认定的丹麦文化遗产更深层次、更具定义性的要素。每个问答对均经过人工审核或修正，最终形成了包含741个封闭式问答对的数据集，内容涵盖从公元前1300年的考古发现、18世纪的诗歌和音乐作品，到当代流行音乐、丹麦设计和建筑等。在该基准测试上的基线结果显示，当代语言模型（GPT-OSS-120B和20B、Llama-3.3-70B、Gemma3-27B以及Mistral-3.1-24B）的表现...（原文在此截断）

    arXiv:2601.19930v2 Announce Type: replace-cross  Abstract: We introduce Daisy, a factual knowledge benchmark for Danish cultural heritage, based on curated topics from the Danish Culture Canon 2006. For each artifact in the culture canon, we query the corresponding Wikipedia page and have a language model generate a diverse set of questions. Within each artifact, we sample both central and peripheral questions, testing not only mainstream information but also the deeper, defining elements of Danish cultural heritage as identified by the Canon committee. Each question-answer pair is manually approved or corrected, yielding a final dataset of 741 closed-ended question-answer pairs ranging from archaeological findings dated to 1300 BCE and 18th-century poems and musical pieces through to contemporary pop music, Danish design, and architecture. Baseline results on our benchmark show that contemporary language models (GPT-OSS-120B & 20B, Llama-3.3-70B, Gemma3-27B, and Mistral-3.1-24B) perfo
    
[^294]: 识别与迁移推理关键神经元：通过激活导向提升大语言模型推理可靠性

    Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering

    [https://arxiv.org/abs/2601.19847](https://arxiv.org/abs/2601.19847)

    本文提出AdaRAS框架，通过识别推理关键神经元并在推理时自适应地引导其激活，以轻量级方式显著提升大语言模型推理的可靠性和准确性。

    

    尽管近期的大语言模型（LLM）具有强大的推理能力，但在具有挑战性的任务上实现可靠的性能通常需要后训练或计算成本高昂的采样策略，这限制了其实际效率。在这项工作中，我们首先证明了LLM中有一小部分神经元与推理正确性表现出很强的预测相关性。基于这一观察，我们提出了AdaRAS（自适应推理激活导向），这是一个轻量级的测试时框架，通过有选择地干预神经元激活来提升推理可靠性。AdaRAS通过极性感知的均值差准则识别推理关键神经元（RCNs），并在推理过程中自适应地引导其激活，从而增强错误的推理轨迹，同时避免在原本已经正确的情况下出现性能退化。在10个数学和代码基准测试上的实验表明了一致的性能提升，包括超过13%的……

    arXiv:2601.19847v3 Announce Type: replace  Abstract: Despite the strong reasoning capabilities of recent large language models (LLMs), achieving reliable performance on challenging tasks often requires post-training or computationally expensive sampling strategies, limiting their practical efficiency. In this work, we first show that a small subset of neurons in LLMs exhibits strong predictive correlations with reasoning correctness. Based on this observation, we propose AdaRAS (Adaptive Reasoning Activation Steering), a lightweight test-time framework that improves reasoning reliability by selectively intervening on neuron activations. AdaRAS identifies Reasoning-Critical Neurons (RCNs) via a polarity-aware mean-difference criterion and adaptively steers their activations during inference, enhancing incorrect reasoning traces while avoiding degradation on already-correct cases. Experiments on 10 mathematics and coding benchmarks demonstrate consistent improvements, including over 13% 
    
[^295]: 我们如何与其他学科互动？一个研究学术出版物中有意义跨学科话语的框架

    How Do We Engage with Other Disciplines? A Framework to Study Meaningful Interdisciplinary Discourse in Scholarly Publications

    [https://arxiv.org/abs/2601.17020](https://arxiv.org/abs/2601.17020)

    本文提出了一个针对跨学科NLP出版物的引用参与度评估框架，引入了专为跨学科工作设计的引用目的分类体系，弥补了现有方法无法定量衡量引用参与质量的不足。

    

    随着跨学科研究的日益流行以及机构在此方向的激励不断增加，越来越需要理解由此产生的出版物如何整合来自多个学科的思想。现有的计算方法，如机构多样性、关键词和引用模式，并未考虑单个引用如何被用来推进引用工作本身的发展。尽管为了弥补这一空白，先前的研究已提出了用于分类引用目的的分类体系，但这些框架并不适合跨学科研究，也无法提供引用参与质量的定量测量。为了解决这些局限性，我们提出了一个用于评估跨学科自然语言处理（NLP）出版物中引用参与度的框架。我们的方法引入了一个专门为跨学科工作定制的引用目的分类体系，并得到了标注研究的支持。

    arXiv:2601.17020v3 Announce Type: replace-cross  Abstract: With the rising popularity of interdisciplinary work and increasing institutional incentives in this direction, there is a growing need to understand how resulting publications incorporate ideas from multiple disciplines. Existing computational approaches, such as affiliation diversity, keywords, and citation patterns, do not account for how individual citations are used to advance the citing work. Although, in line with addressing this gap, prior studies have proposed taxonomies to classify citation purpose, these frameworks are not well-suited to interdisciplinary research and do not provide quantitative measures of citation engagement quality. To address these limitations, we propose a framework for the evaluation of citation engagement in interdisciplinary Natural Language Processing (NLP) publications. Our approach introduces a citation purpose taxonomy tailored to interdisciplinary work, supported by an annotation study. 
    
[^296]: 单句中的人类价值观：Schwartz连续谱上的道德存在性、层级结构与Transformer集成方法

    Human Values in a Single Sentence: Moral Presence, Hierarchies, and Transformer Ensembles on the Schwartz Continuum

    [https://arxiv.org/abs/2601.14172](https://arxiv.org/abs/2601.14172)

    该研究在ValueEval'24语料上对19个Schwartz人类价值观进行句子级多标签分类，发现道德存在性可从单句有效学习、存在性门控的层级结构并不优于直接预测，而决策阈值校准是一个决定性却常被忽视的关键因素。

    

    我们通过在74,000个英语新闻和宣言句子（ValueEval'24语料库）上进行句子级检测，研究严重标签不平衡条件下的神经多标签分类，目标是识别19个精炼的Schwartz人类价值观。每个句子都带有大致平衡的道德存在性标签以及19路价值观标注。首先，道德存在性可以从单个句子中学习：一个DeBERTa-base分类器在默认阈值下达到正类F1约0.73，且概率校准无法进一步提升该结果。其次，在8 GB消费级GPU预算下，我们比较了直接多标签检测器与基于存在性门控的层级结构，发现门控并不能优于直接预测，因为门控的召回率成为瓶颈。第三，通过研究轻量级辅助信号和小型集成模型，我们将决策阈值校准分离为一个决定性的、却常被忽视的因素：一个标准的纯文本基线已经能够匹配ValueEval'24官方最佳英语……（摘要在此处截断）

    arXiv:2601.14172v4 Announce Type: replace-cross  Abstract: We study neural multi-label classification under severe label imbalance through sentence-level detection of the 19 refined Schwartz human values in 74k English news and manifesto sentences (ValueEval'24 corpus). Each sentence carries a roughly balanced moral-presence label and a 19-way value annotation. First, moral presence is learnable from single sentences: a DeBERTa-base classifier reaches positive-class $F_1 \approx 0.73$ at the default threshold, which calibration does not improve. Second, comparing direct multi-label detectors with presence-gated hierarchies under an 8 GB consumer-grade GPU budget, we find that gating does not improve over direct prediction, as gate recall becomes a bottleneck. Third, studying lightweight auxiliary signals and small ensembles, we isolate decision-threshold calibration as a decisive, often overlooked factor: a standard text-only baseline already matches the best official ValueEval'24 Engl
    
[^297]: 用更少的题目实现高置信度排名：基于连续分数的自适应LLM评估

    Confident Rankings with Fewer Items: Adaptive LLM Evaluation with Continuous Scores

    [https://arxiv.org/abs/2601.13885](https://arxiv.org/abs/2601.13885)

    本文将基于IRT的自适应测试从二值评分扩展到连续有界分数，提出不确定性感知的自适应停止排序方法，仅需2%的测试题目即可实现对LLM的高置信度可靠排名。

    

    计算机化自适应测试（CAT）已被证明在多项选择题基准测试的高效LLM评估中行之有效，但现代LLM评估越来越依赖于生成任务，其输出是连续评分而非标记正确/错误。我们提出了基于项目反应理论（IRT）的自适应测试向连续有界分数（ROUGE、BLEU、LLM-as-a-Judge）的原则性扩展，方法是将伯努利响应分布替换为异方差正态分布。在此基础上，我们引入了一种具有自适应停止准则的不确定性感知排序器，能够在测试尽可能少题目并尽可能降低成本的情况下实现可靠的模型排名。我们在五个涵盖基于n-gram、基于嵌入和LLM-as-judge指标的基准上验证了我们的方法。我们的方法相比随机采样将排名相关性提高了0.13 τ，经过一次性校准后仅使用2%的题目即可在置信预测上达到99%的准确率。

    arXiv:2601.13885v2 Announce Type: replace-cross  Abstract: Computerized Adaptive Testing (CAT) has proven effective for efficient LLM evaluation on multiple-choice benchmarks, but modern LLM evaluation increasingly relies on generation tasks where outputs are scored continuously rather than marked correct/incorrect. We present a principled extension of IRT-based adaptive testing to continuous bounded scores (ROUGE, BLEU, LLM-as-a-Judge) by replacing the Bernoulli response distribution with a heteroskedastic normal distribution. Building on this, we introduce an uncertainty aware ranker with adaptive stopping criteria that achieves reliable model ranking while testing as few items and as cheaply as possible. We validate our method on five benchmarks spanning n-gram-based, embedding-based, and LLM-as-judge metrics. Our method improves ranking correlation by 0.13 $\tau$ over random sampling and has 99% accuracy on confident predictions while using 2% of the items after a one-time calibrat
    
[^298]: 大语言模型的拒绝行为在语义上有多稳定？度量局部安全边界中的混乱

    How Semantically Stable Are LLM Refusals? Measuring Confusion in Local Safety Boundaries

    [https://arxiv.org/abs/2512.01037](https://arxiv.org/abs/2512.01037)

    该论文提出“语义混乱”这一新失败模式，并构建了包含 1 万条提示的受控改写语料库 ParaGuard，用于度量大语言模型在语义等价改写之间拒绝决策的局部不一致性，弥补了传统全局指标（如误拒率）无法反映拒绝行为语义稳定性的不足。

    

    随着安全对齐在大语言模型中成为标准，拒绝行为已成为模型可靠性的重要组成部分。然而，模型仍可能拒绝良性提示，尤其是当措辞与风险内容相似时。现有评估通常报告全局分数，例如误拒率或遵从率。这些分数虽然有用，但它们将每个提示独立处理。因此，它们忽略了局部不一致性——即模型接受某一意图的一种表述，却拒绝其近义改写。这使得人们难以判断拒绝行为究竟是仅仅频繁，还是在语义上也不稳定。我们通过引入“语义混乱”（Semantic Confusion）来填补这一空白，这是一种失败模式，用于捕捉模型在保持语义不变的改写之间做出的矛盾拒绝决策。我们构建了 ParaGuard，一个包含 1 万条提示的语料库，由受控改写簇组成，在保持意图不变的同时改变表面表达形式。我们还提出了三种模型无关的……（原文摘要在此处截断）

    arXiv:2512.01037v3 Announce Type: replace-cross  Abstract: As safety alignment becomes standard in large language models, refusal behavior has become an important part of model reliability. However, models may still reject benign prompts, especially when the wording resembles risky content. Existing evaluations usually report global scores, such as false rejection rate or compliance rate. These scores are useful, but they treat each prompt independently. As a result, they miss local inconsistency, where a model accepts one phrasing of an intent but rejects a close paraphrase. This makes it difficult to understand whether refusals are only frequent or also semantically unstable. We address this gap by introducing Semantic Confusion, a failure mode that captures contradictory refusal decisions across meaning-preserving paraphrases. We build ParaGuard, a 10k-prompt corpus of controlled paraphrase clusters that keep intent fixed while varying surface form. We also propose three model-agnos
    
[^299]: 理解干扰信息对推理视觉语言模型的影响

    Understanding the Effects of Distractors on Reasoning Vision-Language Models

    [https://arxiv.org/abs/2511.21397](https://arxiv.org/abs/2511.21397)

    本研究提出Idis视觉问答数据集，揭示了视觉干扰因素对推理视觉语言模型的影响机制与文本干扰因素根本不同——尽管两者都会引发逆向缩放效应，但视觉干扰因素在降低准确率的同时并不会增加推理长度。

    

    不相关的信息（即干扰因素）如何影响视觉语言模型（VLM）的测试时扩展？先前针对纯文本语言模型的研究表明，文本干扰因素会加剧逆向缩放效应，导致模型推理轨迹更长但效率更低。在本工作中，我们研究了类似现象是否也会在多模态环境中出现。我们提出了Idis（含干扰信息的图像）数据集，这是一个视觉问答数据集，在语义和数量维度上系统地变化干扰因素。我们的分析揭示，视觉干扰因素对推理视觉语言模型的影响方式与文本干扰因素存在根本性差异：虽然逆向缩放效应仍然出现，但视觉干扰因素在降低准确率的同时并不会增加推理长度。我们进一步表明，从推理轨迹中提取的属性计数为理解干扰因素如何与推理长度和准确率相互作用提供了关键洞察。

    arXiv:2511.21397v3 Announce Type: replace-cross  Abstract: How does irrelevant information (i.e., distractors) affect test-time scaling in vision-language models (VLMs)? Prior work on text-only language models has shown that textual distractors can intensify inverse scaling, causing models to reason longer but less effective reasoning traces. In this work, we investigate whether similar phenomena arise in multimodal settings. We introduce Idis (Images with distractors), a visual question-answering dataset that systematically varies distractors along semantic and numerical dimensions. Our analyses reveal that visual distractors affect reasoning VLMs in a fundamentally different way from textual distractors: although inverse scaling still emerges, visual distractors reduce accuracy without increasing reasoning length. We further show that attribute counts extracted from reasoning traces provide key insights into how distractors interact with reasoning length and accuracy. As a sanity che
    
[^300]: SkyEgg：基于等价饱和的异构感知硬件综合

    SkyEgg: Heterogeneity-Aware Hardware Synthesis via Equality Saturation

    [https://arxiv.org/abs/2511.15323](https://arxiv.org/abs/2511.15323)

    SkyEgg是一个基于等价饱和的异构感知FPGA硬件综合框架，通过在变换、调度和映射阶段联合探索硬件实现方案与细粒度配置选择，突破了传统顺序式综合流程的局限，从而提升端到端性能。

    

    硬件综合是高层程序与加速器设计之间的关键接口。现代FPGA在资源功能和时序可配置性方面日益呈现出异构性。要充分利用这些资源，综合过程需要在多种硬件实现方案和细粒度配置选项之间进行选择，并在多个阶段之间协调这些选择。然而，先前的框架由于资源模型仅限于按操作的映射方式以及有限的融合模式，未能充分探索这些选择。此外，这些框架按顺序执行变换、调度和映射，在完整映射信息可用之前就确定了表达式形式并分配了周期边界。这一问题阻碍了操作链接与融合，将可行的实现方案排除在后续优化阶段之外，并限制了端到端性能。我们提出了SkyEgg，一个面向FPGA的异构感知硬件综合框架。

    arXiv:2511.15323v2 Announce Type: replace-cross  Abstract: Hardware synthesis is a key interface between high-level programs and accelerator designs. Modern FPGAs increasingly expose heterogeneity in resource functionality and timing configurability. Exploiting these resources requires synthesis to choose among hardware implementations and fine-grained configuration options, and coordinate these choices across multiple stages. However, prior frameworks fail to fully explore these choices due to resource models confined to per-operation mappings and limited fusion patterns. Moreover, they perform transformation, scheduling, and mapping sequentially, selecting an expression form and assigning cycle boundaries before full mapping information is available. This problem precludes chaining and fusion, excludes viable implementations from subsequent optimization stages, and limits end-to-end performance.   We present SkyEgg, a heterogeneity-aware hardware synthesis framework for FPGAs. The ke
    
[^301]: 供体与受体：基于参数高效微调的任务与语言间非对称迁移研究

    Donors and Recipients: On Asymmetric Transfer Across Tasks and Languages with Parameter-Efficient Fine-Tuning

    [https://arxiv.org/abs/2511.13368](https://arxiv.org/abs/2511.13368)

    该研究通过系统的LoRA单源微调实验发现，跨任务与跨语言的迁移增益呈强烈非对称性，其中匹配任务跨语言迁移最有效，且迁移幅度主要取决于目标语言的特性。

    

    大语言模型（LLMs）在各类任务和语言上表现出色，然而某一任务或语言上的能力提升如何影响其他任务和语言仍知之甚少。我们在多个开放权重的LLM家族和不同规模上开展了受控的LoRA微调研究，使用由11种语言和四个基准测试构成的标准实验网格。我们在单一任务-语言源上对每个模型进行微调，然后在所有其他任务-语言目标对上进行评估以衡量迁移效果。我们将迁移分解为三种模式：（一）匹配任务（跨语言），（二）跨任务（匹配语言），（三）跨任务（跨语言）。单源微调在所有模式下均产生净正向提升，但增益呈现强烈的非对称性。匹配任务（跨语言）迁移是最有效且结构最规律的模式，其迁移幅度主要由目标语言本身的特性而非模型决定。

    arXiv:2511.13368v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) perform strongly across tasks and languages, yet how improvements in one task or language affect other tasks and languages remains poorly understood. We conduct a controlled LoRA fine-tuning study across multiple open-weight LLM families and scales, using a standardised grid of 11 languages and four benchmarks. We fine-tune each model on a single task-language source, then evaluate it on all other task-language target pairs to measure transfer. We decompose transfer into three regimes: (i) Matched-Task (Cross-Language), (ii) Cross-Task (Matched-Language), and (iii) Cross-Task (Cross-Language). Single-source fine-tuning yields a net positive uplift across regimes, but the gains are strongly asymmetric. Matched-Task (Cross-Language) transfer emerges as the most effective and structurally regular regime, with transfer magnitude driven principally by the identity of the target language rather than model
    
[^302]: 剪枝即正则化：语音识别中的敏感性感知一次性剪枝

    Pruning as Regularization: Sensitivity-Aware One-Shot Pruning in ASR

    [https://arxiv.org/abs/2511.08092](https://arxiv.org/abs/2511.08092)

    该论文发现一次性幅度剪枝不仅是压缩手段，更可作为强大的隐式正则化器，通过梯度与Fisher信息敏感性诊断指导的组件级剪枝，在不微调的情况下显著提升Whisper语音识别模型的泛化性能并降低词错误率。

    

    我们挑战了神经网络剪枝仅被视为压缩技术的传统观念，证明一次性幅度剪枝可以作为语音识别（ASR）中一种强大的隐式正则化器。我们以Whisper-small模型为对象，将基于梯度和Fisher信息的敏感性诊断与有针对性的按组件剪枝相结合。这揭示了架构上的不对称性：解码器的前馈网络（FFN）对剪枝较为脆弱，而解码器自注意力层和最后几层编码器则包含冗余，去除这些冗余反而能提升泛化能力。在无需微调的情况下，剪枝50%的解码器自注意力使LibriSpeech test-other测试集上的词错误率（WER）绝对降低2.38%（相对降低20.44%）；而以50%比例剪枝最后四层编码器则带来1.72%的绝对改进（相对改进14.8%）。这些收益在Common Voice和TED-LIUM数据集上同样得到验证。除了正则化收益之外，我们的敏感性感知方法还支持更激进的一次性压缩。在40%的稀疏度下……

    arXiv:2511.08092v2 Announce Type: replace-cross  Abstract: We challenge the conventional view of neural network pruning as solely a compression technique, demonstrating that one-shot magnitude pruning serves as a powerful implicit regularizer for ASR. Using Whisper-small, we combine gradient- and Fisher-based sensitivity diagnostics with targeted, component-wise pruning. This reveals architectural asymmetries: decoder FFNs are pruning-fragile, whereas decoder self-attention and the last encoder layers contain redundancy that, when removed, improves generalization. Without fine-tuning, pruning 50% of decoder self-attention reduces WER by 2.38% absolute (20.44% relative) on LibriSpeech test-other; pruning the last four encoder layers at 50% instead yields a 1.72% absolute (14.8% relative) improvement. Gains persisted on Common Voice and TED-LIUM datasets. Beyond regularization benefits, our sensitivity-aware approach enables more aggressive one-shot compression. At 40% sparsity, where es
    
[^303]: 自然语言文本中认知扭曲的可解释识别

    Interpretable Recognition of Cognitive Distortions in Natural Language Texts

    [https://arxiv.org/abs/2511.05969](https://arxiv.org/abs/2511.05969)

    本文提出一种基于加权N-gram等结构化模式及其层级关系的可解释人工智能方法，用于自动识别心理护理文本中的认知扭曲，在两个公开数据集上显著提升了F1分数。

    

    我们提出了一种基于加权结构化模式（如N-gram）的自然语言文本多因素分类新方法，该方法考虑了这些模式之间的层级交叉关系，应用于解决一个具有重大社会影响的问题——在心理护理中自动检测特定的认知扭曲，依托于一个可解释、鲁棒且透明的人工智能模型。所提出的识别和学习算法改进了该领域的现有技术水平。该改进在两个公开可用的数据集上进行了测试，在该任务的F1分数上显著超越了文献中已知的最佳结果，并确定了最优超参数，代码和模型可供社区未来使用。

    arXiv:2511.05969v3 Announce Type: replace-cross  Abstract: We propose a new approach to multi-factor classification of natural language texts based on weighted structured patterns such as N-grams, taking into account the heterarchical relationships between them, applied to solve such a socially impactful problem as the automation of detection of specific cognitive distortions in psychological care, relying on an interpretable, robust and transparent artificial intelligence model. The proposed recognition and learning algorithms improve the current state of the art in this field. The improvement is tested on two publicly available datasets, with significant improvements over literature-known F1 scores for the task, with optimal hyper-parameters determined, having code and models available for future use by the community.
    
[^304]: CapGeo-Bench：将视觉感知与推理解耦并评估几何理解能力

    CapGeo-Bench: Decoupling Visual Perception from Reasoning and Evaluating Geometric Understanding

    [https://arxiv.org/abs/2510.09302](https://arxiv.org/abs/2510.09302)

    该论文通过实验证实多模态几何推理的瓶颈在于视觉感知而非推理能力，并提出了包含4,641对高质量图形-图注及基于关键点评估指标的CapGeo-Bench基准，用于专门评估MLLMs的几何视觉感知能力。

    

    尽管多模态大语言模型（MLLMs）在困难的纯文本数学推理任务中取得了显著成功，但即使是GPT-o3等先进的闭源模型在几何问题上仍然表现不佳。这种差异促使我们探究其根本原因：多模态几何推理的瓶颈究竟在于推理本身，还是在于对图形中几何信息的感知？为了回答这个问题，我们进行了一项探索性实验，发现提供高质量的图注能够持续且显著地提升MLLMs的性能，从而实证验证了几何推理中的视觉感知瓶颈。然而，MLLMs在视觉几何感知方面的能力仍未得到充分评估。为填补这一空白，我们提出了CapGeo-Bench，这是一个包含4,641对高质量图形-图注的基准数据集，并配备了一种细粒度的基于关键点的评估指标。

    arXiv:2510.09302v2 Announce Type: replace-cross  Abstract: While Multimodal Large Language Models (MLLMs) have achieved remarkable success in difficult purely textual mathematical reasoning tasks, even advanced closed-source models such as GPT-o3 still struggle with geometric problems. This discrepancy motivates us to investigate the root cause: is the bottleneck of multimodal geometric reasoning rooted in reasoning itself, or in the perception of geometric information from diagrams? To answer this question, we conduct an exploratory experiment and find that providing high-quality captions consistently and substantially boosts performance of MLLMs, empirically validating the visual perception bottleneck in geometric reasoning. However, MLLMs' capabilities in visual geometric perception remain insufficiently evaluated. To fill the gap, we propose CapGeo-Bench, a benchmark of 4,641 high-quality figure-caption pairs equipped with a fine-grained keypoint-based evaluation metric. Specifical
    
[^305]: MuPlon：通过控制混杂因素实现声明验证的多路径因果优化

    MuPlon: Multi-Path Causal Optimization for Claim Verification through Controlling Confounding

    [https://arxiv.org/abs/2509.25715](https://arxiv.org/abs/2509.25715)

    该论文提出MuPlon框架，通过后门路径和前门路径的双重因果干预策略，有效控制声明-证据图中的数据噪声和数据偏差两类混杂因素，从而提升声明验证的可靠性。

    

    作为数据质量控制中的一项关键任务，声明验证旨在基于广泛的证据评估声明的真实性，从而遏制错误信息的传播。然而，传统方法往往忽视证据之间复杂的相互作用，导致验证结果不可靠。一种直接的解决方案是将声明和证据表示为全连接图，我们将其定义为声明-证据图（C-E图）。然而，基于全连接图的声明验证方法面临两个主要的混杂挑战：数据噪声和数据偏差。为了应对这些挑战，我们提出了一种新颖的框架——多路径因果优化。MuPlon集成了双重因果干预策略，包括后门路径和前门路径。在后门路径中，MuPlon通过优化节点概率权重来稀释噪声节点的干扰，同时增强……

    arXiv:2509.25715v2 Announce Type: replace-cross  Abstract: As a critical task in data quality control, claim verification aims to curb the spread of misinformation by assessing the truthfulness of claims based on a wide range of evidence. However, traditional methods often overlook the complex interactions between evidence, leading to unreliable verification results. A straightforward solution represents the claim and evidence as a fully connected graph, which we define as the Claim-Evidence Graph (C-E Graph). Nevertheless, claim verification methods based on fully connected graphs face two primary confounding challenges, Data Noise and Data Biases. To address these challenges, we propose a novel framework, Multi-Path Causal Optimization (MuPlon). MuPlon integrates a dual causal intervention strategy, consisting of the back-door path and front-door path. In the back-door path, MuPlon dilutes noisy node interference by optimizing node probability weights, while simultaneously strengthen
    
[^306]: 广义正确性模型：从历史模式中学习校准且与模型无关的正确性预测器

    Generalized Correctness Models: Learning Calibrated and Model-Agnostic Correctness Predictors from Historical Patterns

    [https://arxiv.org/abs/2509.24988](https://arxiv.org/abs/2509.24988)

    该论文挑战了LLM具备“自我认知”能力的传统假设，发现LLM判断自身输出正确性的能力并不优于无关模型，并提出通过注入目标模型的历史预测信息来构建广义正确性模型（GCM），从而实现校准且与模型无关的正确性预测。

    

    生成准确且校准的置信度估计对于在高风险或面向用户的应用中部署大语言模型（LLM）至关重要，而这仍然是一个悬而未决的挑战。先前的研究通常将置信度视为引导模型“自我认知”的问题，即LLM判断自身答案是否正确的能力；这种方法隐含地假设存在某种关于答案正确性的特权信息，且该信息只有模型本身才能获取。然而，我们的实验表明，LLM在预测自身输出正确性时的表现通常并不比一个无关的LLM更好。此外，我们假设构建“正确性模型”（CM）的一个关键因素是接触目标模型的历史预测。我们提出了多种注入这种历史正确性信息的方法，从而创建了广义正确性模型（GCM）。我们首先证明了GCM可以在…（摘要在此处截断）

    arXiv:2509.24988v2 Announce Type: replace-cross  Abstract: Generating accurate and calibrated confidence estimates is critical for deploying LLMs in high-stakes or user-facing applications, and remains an open challenge. Prior research has often framed confidence as a problem of eliciting a model's "self-knowledge", i.e., the ability of an LLM to judge whether its own answers are correct; this approach implicitly assumes that there is some privileged information about the answer's correctness that is accessible to the model itself. However, our experiments reveal that an LLM attempting to predict the correctness of its own outputs generally performs no better than an unrelated LLM. Moreover, we hypothesize that a key factor in building a "Correctness Model" (CM) is exposure to a target model's historical predictions. We propose multiple methods to inject this historical correctness information, creating a Generalized Correctness Model (GCM). We first show that GCMs can be trained on th
    
[^307]: DiffuTester：通过挖掘结构模式加速扩散大语言模型的单元测试生成

    DiffuTester: Accelerating Unit Test Generation for Diffusion LLMs via Mining Structural Pattern

    [https://arxiv.org/abs/2509.24975](https://arxiv.org/abs/2509.24975)

    DiffuTester通过挖掘针对同一焦点方法的单元测试所共享的结构模式，并提出基于抽象语法树的动态结构模式解码方法，在保证测试质量的同时显著加速了扩散大语言模型的单元测试生成。

    

    扩散大语言模型（dLLMs）支持并行生成，在单元测试生成（UTG）方面前景广阔，而高效、大规模的自动化测试在软件开发中至关重要。尽管具有这一优势，它们在单元测试生成中的应用仍受到效率与测试质量之间明显权衡的制约，因为增加每一步生成的token数量往往会导致测试用例质量急剧下降。为克服这一局限，我们提出了DiffuTester，一个专为dLLMs在单元测试生成中量身定制的加速框架。DiffuTester的动机在于：针对同一焦点方法的单元测试通常共享结构模式。DiffuTester采用了一种新颖的基于结构模式的解码方法，通过单元测试的抽象语法树动态识别跨单元测试的结构模式，并对相应的token进行额外解码，从而实现加速。

    arXiv:2509.24975v4 Announce Type: replace-cross  Abstract: Diffusion large language models (dLLMs) enable parallel generation and are promising for unit test generation (UTG), where efficient and large-scale automated testing is essential in software development. Despite this advantage, their application to UTG is still constrained by a clear trade-off between efficiency and test quality, since increasing the number of tokens generated in each step often causes a sharp decline in the quality of test cases. To overcome this limitation, we present DiffuTester, an acceleration framework specifically tailored for dLLMs in UTG. The motivation of DiffuTester is that unit tests targeting the same focal method often share structural patterns. DiffuTester employs a novel structural pattern based decoding approach, which dynamically identifies structural patterns across unit tests through their abstract syntax trees and additionally decodes the corresponding tokens, thereby achieving acceleratio
    
[^308]: 可解释性能否预测模型在未见数据上的行为？

    Can Interpretation Predict Behavior on Unseen Data?

    [https://arxiv.org/abs/2507.06445](https://arxiv.org/abs/2507.06445)

    该研究提出仅凭分布内数据上观察到的注意力模式即可预测模型在分布外数据上遵循的泛化规则，并证明解释的预测价值可与机制忠实性解耦——即使内部模式在因果上抑制其所预测的规则，观察性分析依然能成功预测模型行为。

    

    可解释性研究通常预测模型对特定机制干预的响应，但我们能否预测模型对未见输入数据的响应？我们提出并演示了这一替代性目标：利用模型内部状态来预测其分布外（OOD）行为。我们在简单的合成任务上训练了数百个Transformer模型，在这些任务中，完美的分布内准确率可以与多种不同的OOD泛化规则相兼容。我们成功地使用仅在分布内数据上观察到的注意力模式，来预测每个模型在OOD数据上遵循哪条泛化规则。我们的实验将解释的机制忠实性与其预测价值解耦；消融实验揭示，这些内部模式可能会抑制而非支持它们所预测的规则，这表明即使因果分析无法支持简单的因果关系，观察性分析依然能够预测模型行为。我们的发现为一种新的可解释性范式提供了概念验证。

    arXiv:2507.06445v4 Announce Type: replace-cross  Abstract: Interpretability research often predicts model responses to targeted mechanistic interventions. But can we predict responses to unseen input data? We propose and demonstrate this alternate objective by using model internals to predict their out-of-distribution (OOD) behavior. We train hundreds of Transformers on simple synthetic tasks, where perfect in-distribution accuracy is compatible with multiple OOD generalization rules. We successfully use attention patterns -- observed only on in-distribution data -- to predict which rule each model follows on OOD data. Our experiments decouple the mechanistic faithfulness of our interpretation from its predictive value; ablations reveal such internal patterns can suppress rather than support the rule they predict, showing observational analysis can forecast behavior even when causal analysis fails to support a simple cause-effect link. Our findings are a proof-of-concept for a new inte
    
[^309]: 所有学习都有情感基础，面向任务的对话亦然

    All Learning Has an Emotional Basis, So Does Task-Oriented Dialogue

    [https://arxiv.org/abs/2507.01594](https://arxiv.org/abs/2507.01594)

    提出了一种将完全词汇化表示与LLM主干相结合的端到端框架，通过在线强化学习同时利用短期情感奖励和长期任务成功奖励，联合优化面向任务对话系统中的情感交互与任务完成效果。

    

    arXiv:2507.01594v2 公告类型：替换 摘要：面向任务的对话系统旨在通过自然语言交互帮助用户完成目标。除了任务成功之外，有效的ToD系统还必须在本质上充满噪声和歧义的对话环境中保持积极的情感交互并准确传达信息。大语言模型（LLM）的最新进展显著提升了对话的流畅性和上下文理解能力。然而，尽管情感已被纳入现有的ToD系统中，其整合通常仅限于有限的角色，限制了其对整体对话行为和用户体验的影响。为解决这一问题，我们提出了一种新颖的端到端框架，该框架将完全词汇化的表示与LLM主干相结合以提高语义准确性，并采用在线强化学习，通过短期情感奖励和长期任务成功奖励对情感交互与任务完成进行联合优化。

    arXiv:2507.01594v2 Announce Type: replace  Abstract: Task-oriented dialogue (ToD) systems aim to help users accomplish goals through natural language interaction. Beyond task success, effective ToD systems must also maintain positive emotional interaction and accurately convey information in inherently noisy and ambiguous conversational environments. Recent advances in large language models (LLMs) have substantially improved conversational fluency and contextual understanding. However, although emotion has been incorporated into existing ToD systems, its integration is typically confined to limited roles, restricting its influence on overall dialogue behaviour and user experience. To address this, we propose a novel end-to-end framework that combines fully lexicalised representations with an LLM backbone for improved semantic accuracy and employs online reinforcement learning with short-term emotional and long-term task-success rewards for joint optimisation of emotional interaction an
    
[^310]: LLM概率集中：对齐如何缩小生成视野

    LLM Probability Concentration: How Alignment Shrinks the Generative Horizon

    [https://arxiv.org/abs/2506.17871](https://arxiv.org/abs/2506.17871)

    该研究用分支因子（BF）量化LLM输出分布的概率集中程度，发现对齐微调从一开始就锐化输出分布、使BF总体降低2至5倍，而BF随生成进展的下降则主要是与对齐无关的自回归自我条件化的固有属性。

    

    尽管大型语言模型（LLM）能力令人印象深刻，但经过对齐的LLM往往生成的输出缺乏多样性。是什么驱动了生成过程中的这种一致性？我们通过模型输出分布中概率集中的视角来研究这一现象。为了量化这一现象，我们使用分支因子——输出分布的长度平均熵的指数化形式，可解释为生成过程中合理下一步骤的有效数量。我们的实证分析揭示了两个关键发现：（1）BF通常随着生成的进行而下降，表明LLM变得越来越可预测；一项受控干预实验表明，这种下降在很大程度上是自回归自我条件化的任务无关属性，与对齐无关，且可以通过意外的上下文在局部被逆转。（2）对齐微调从一开始就锐化了输出分布，总体上将BF降低了2至5倍。

    arXiv:2506.17871v4 Announce Type: replace-cross  Abstract: Despite their impressive capabilities, aligned large language models (LLMs) often generate outputs that lack diversity. What drives this consistency in the generation? We investigate this phenomenon through the lens of probability concentration in the model's output distribution. To quantify it, we use the Branching Factor (BF)--the exponentiated length-averaged entropy of the output distribution, interpreted as the effective number of plausible next steps during generation. Our empirical analysis reveals two key findings: (1) BF often decreases as generation progresses, suggesting that LLMs become more predictable; a controlled intervention indicates that this decline is largely a task-independent property of autoregressive self-conditioning, distinct from alignment, and can be locally reversed by unexpected context. (2) Alignment tuning sharpens the output distribution from the outset, reducing BF by a factor of 2--5 overall 
    
[^311]: 利用多方面相关性与语法信息推进自动化口语测评

    Advancing Automated Speaking Assessment Leveraging Multifaceted Relevance and Grammar Information

    [https://arxiv.org/abs/2506.16285](https://arxiv.org/abs/2506.16285)

    本文通过引入整合问题、图像、范例与口语回答的多方面相关性模块，以及基于语法纠错技术的细粒度语法错误特征，显著提升了自动化口语测评系统中内容相关性和语言使用的评估性能。

    

    当前用于多维度评估的自动化口语测评（ASA）系统往往未能充分利用内容相关性，忽视了图像或范例提示，并且采用的语法分析较为浅显、缺乏详细的错误类型分类。本文通过引入两项新颖的改进构建混合评分模型，以弥补这些不足。首先，多方面相关性模块整合了问题及其关联的图像内容、范例以及二语学习者的口语回答，以全面评估内容相关性。其次，利用先进的语法纠错（GEC）技术和详细标注，提取细粒度的语法错误特征，以识别具体的错误类别。实验和消融研究表明，这些组件显著提升了内容相关性、语言使用以及整体ASA性能的评估效果，凸显了使用更丰富、更细致信息的优势。

    arXiv:2506.16285v2 Announce Type: replace  Abstract: Current automated speaking assessment (ASA) systems for use in multi-aspect evaluations often fail to make full use of content relevance, overlooking image or exemplar cues, and employ superficial grammar analysis that lacks detailed error types. This paper ameliorates these deficiencies by introducing two novel enhancements to construct a hybrid scoring model. First, a multifaceted relevance module integrates question and the associated image content, exemplar, and spoken response of an L2 speaker for a comprehensive assessment of content relevance. Second, fine-grained grammar error features are derived using advanced grammar error correction (GEC) and detailed annotation to identify specific error categories. Experiments and ablation studies demonstrate that these components significantly improve the evaluation of content relevance, language use, and overall ASA performance, highlighting the benefits of using richer, more nuanced 
    
[^312]: 谁来评测评测基准？面向常识推理基准的综合评估

    Who Benchmarks the Benchmarks? Towards Comprehensive Evaluation of Commonsense Reasoning Benchmarks

    [https://arxiv.org/abs/2504.07825](https://arxiv.org/abs/2504.07825)

    该论文通过对HellaSwag基准的案例研究揭示了常识推理基准中普遍存在的效度问题（如语法错误、拼写错误、误导性提示及多个正确选项），并发现约68%的模型预测在删除问题提示后仍不变，证明这是基准自身的内在缺陷而非模型数据污染所致。

    

    常识推理是语言模型的一项关键能力，因为据称与特定的知识性内容不同，它是许多基本任务的前提条件。常识推理通常通过多项选择题（MCQ）基准来衡量，例如HellaSwag和PIQA。然而，这些基准中的一些已经过时，并存在大量效度问题。我们通过对HellaSwag——最受欢迎也是问题最多的常识推理基准之一——的案例研究来说明一些典型的效度问题。我们发现的问题涵盖从基本的语法错误和大量拼写错误，到误导性提示或多个同样正确的选项。我们表明，如果删除问题提示或将其替换为"Lorem ipsum dolor..."，约68%的模型预测不会发生改变。我们认为这种现象源于基准本身的内在缺陷，而不仅仅是某些模型中可能存在的数据污染。由于基准分数是研究和实践中模型选择的重要组成部分……

    arXiv:2504.07825v2 Announce Type: replace  Abstract: Commonsense reasoning is a key language model capability, as it is purportedly a prerequisite for many basic tasks, unlike specific factual knowledge. It is often measured with multiple-choice questions (MCQ) benchmarks, e.g. HellaSwag and PIQA. Some of these benchmarks, however, are outdated and contain numerous validity issues. We illustrate some typical validity issues with a case study on HellaSwag, one of the most popular and problematic benchmarks for commonsense reasoning. The issues we find range from basic ungrammaticality and numerous typos to misleading prompts or equally correct options. We show that if we remove question prompts or replace them with "Lorem ipsum dolor...", about 68% of model predictions do not change. We argue that this occurs due to inner flaws in the benchmark, not mere contamination that might be present in some models. Since benchmark scores are an essential part of model selection in both research a
    
[^313]: CoTAL：面向可泛化形成性评估评分与反馈的人机协同提示工程

    CoTAL: Human-in-the-Loop Prompt Engineering for Generalizable Formative Assessment Scoring and Feedback

    [https://arxiv.org/abs/2504.02323](https://arxiv.org/abs/2504.02323)

    该论文提出CoTAL方法，通过人机协同提示工程结合思维链提示与师生反馈迭代优化，使GPT-4在科学、计算、工程等多个领域的形成性评估自动评分性能提升高达38.9%。

    

    大语言模型为协助教师和支持学生学习创造了新的机会。尽管研究人员已经在教育场景中探索了各种提示工程方法，但这些方法在科学、计算和工程等不同领域之间的泛化程度仍未得到充分研究。在本文中，我们提出了思维链提示+主动学习（CoTAL），这是一种基于大语言模型的形成性评估评分方法，它：（1）利用以证据为中心的设计（ECD）将评估和评分量规与课程目标对齐；（2）应用人机协同（human-in-the-loop）提示工程来自动化答案评分；（3）结合思维链（CoT）提示以及教师和学生的反馈，迭代式地改进题目、评分量规和大语言模型提示词。我们的研究结果表明，CoTAL 提升了 GPT-4 在各领域的评分表现，相比未使用提示工程的方法取得了高达 38.9% 的性能提升。

    arXiv:2504.02323v5 Announce Type: replace  Abstract: Large language models (LLMs) have created new opportunities to assist teachers and support student learning. While researchers have explored various prompt engineering approaches in educational contexts, the degree to which these approaches generalize across domains--such as science, computing, and engineering--remains underexplored. In this paper, we introduce Chain-of-Thought Prompting + Active Learning (CoTAL), an LLM-based approach to formative assessment scoring that (1) leverages Evidence-Centered Design (ECD) to align assessments and rubrics with curriculum goals, (2) applies human-in-the-loop prompt engineering to automate response scoring, and (3) incorporates chain-of-thought (CoT) prompting and teacher and student feedback to iteratively refine questions, rubrics, and LLM prompts. Our findings demonstrate that CoTAL improves GPT-4's scoring performance across domains, achieving gains of up to 38.9% over a non-prompt-engine
    
[^314]: 长文本生成中的“迷失中间”现象：合成数据集、评估框架与缓解方法

    Lost-in-the-Middle in Long-Text Generation: Synthetic Dataset, Evaluation Framework, and Mitigation

    [https://arxiv.org/abs/2503.06868](https://arxiv.org/abs/2503.06868)

    本文提出了一种无需训练的检索增强长文本写作框架RAL-Writer，通过规划器与写作器联合建模语义相关性和位置偏差，动态检索并重述关键输入信息，从而缓解大语言模型在长输入到长输出生成中的“迷失中间”问题，并为此任务构建了基准数据集与三项评估指标。

    

    现有的长文本生成方法通常从短输入生成冗长输出，而长输入到长输出的生成任务尚未得到充分探索。随着输入长度的增加，大语言模型越来越容易忽略上下文中间部分的信息——这一局限性被称为“迷失中间”现象——从而导致输出不一致且缺乏连贯性。为解决这一问题，我们提出了检索增强长文本写作框架（RAL-Writer），这是一个无需训练的框架，由生成写作步骤的规划器（Planner）和根据这些步骤产出内容的写作器组成。写作器联合建模语义相关性和位置偏差以计算重要性分数，动态检索关键输入片段，并有策略地重述这些内容。我们还构建了一个用于长输入到长输出生成的基准数据集，并引入了三项评估指标，涵盖长度、一致性和质量。我们还将RAL-Writer与同类方法进行了对比评估。

    arXiv:2503.06868v2 Announce Type: replace-cross  Abstract: Existing long-text generation methods produce lengthy outputs from short inputs, leaving long-input-to-long-output generation underexplored. As input length increases, LLMs increasingly overlook information in the middle of the context--a limitation known as the "lost-in-the-middle" phenomenon--leading to inconsistent and incoherent outputs. To address this problem, we propose Retrieval-Augmented Long-Text Writer (RAL-Writer), a training-free framework consisting of a Planner that generates writing steps and a Writer that produces content according to these steps. The Writer jointly models semantic relevance and positional bias to compute importance scores, dynamically retrieve critical input segments, and strategically restate them. We also construct a benchmark dataset for long-input-to-long-output generation and introduce three evaluation metrics covering length, consistency, and quality. We evaluate RAL-Writer against compa
    
[^315]: LLM-显微镜：揭示标点符号在Transformer上下文记忆中的隐藏作用

    LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory of Transformers

    [https://arxiv.org/abs/2502.15007](https://arxiv.org/abs/2502.15007)

    该论文发现标点符号等看似次要的词元在Transformer的上下文记忆中承载着重要信息，移除它们会显著降低模型性能，并提出开源工具LLM-Microscope来量化词元的上下文承载能力和非线性度。

    

    我们提出了量化大语言模型（LLM）如何编码和存储上下文信息的方法，揭示了常被视为次要的词元（如限定词、标点符号）承载着惊人的高上下文信息。值得注意的是，移除这些词元——尤其是停用词、冠词和逗号——会持续降低模型在MMLU和BABILong-4k基准上的性能，即使移除的只是无关词元。我们的分析还显示了上下文化与线性度之间的强相关性，其中线性度衡量的是从一层嵌入到下一层的转换能被单个线性映射近似的程度。这些发现强调了填充词元在维持上下文方面的隐藏重要性。为了进一步探索，我们提出了LLM-Microscope，这是一个开源工具包，它可以评估词元级别的非线性度、评估上下文记忆、可视化中间层贡献（通过改进的Logit Lens方法），并测量……

    arXiv:2502.15007v2 Announce Type: replace-cross  Abstract: We introduce methods to quantify how Large Language Models (LLMs) encode and store contextual information, revealing that tokens often seen as minor (e.g., determiners, punctuation) carry surprisingly high context. Notably, removing these tokens -- especially stopwords, articles, and commas -- consistently degrades performance on MMLU and BABILong-4k, even if removing only irrelevant tokens. Our analysis also shows a strong correlation between contextualization and linearity, where linearity measures how closely the transformation from one layer's embeddings to the next can be approximated by a single linear mapping. These findings underscore the hidden importance of filler tokens in maintaining context. For further exploration, we present LLM-Microscope, an open-source toolkit that assesses token-level nonlinearity, evaluates contextual memory, visualizes intermediate layer contributions (via an adapted Logit Lens), and measur
    
[^316]: 跳出拟人化范式的思考有益于大语言模型研究

    Thinking beyond the anthropomorphic paradigm benefits LLM research

    [https://arxiv.org/abs/2502.09192](https://arxiv.org/abs/2502.09192)

    本文通过对数十万篇研究文献的实证分析，揭示了拟人化术语在大语言模型研究中的普遍性和增长趋势，识别出五个塑造LLM研究的拟人化假设，并主张超越这些假设以开辟理解和改进大语言模型的新途径。

    

    拟人化，即将人类特质赋予技术，是一种自动且无意识的反应，即使是具备高深技术专业知识的人也会产生这种反应。在这篇立场论文中，我们分析了数十万篇研究文献，为拟人化术语在大语言模型（LLM）研究中的普遍存在及其增长趋势提供了实证证据。我们主张挑战这些术语所反映的更深层次的假设——这些假设虽然往往有用，但可能会在无意中限制LLM的发展——并超越这些假设，从而为理解和改进LLM开辟新的途径。具体而言，我们识别并考察了五个贯穿LLM开发生命周期、塑造研究方向的拟人化假设。对于每个假设（例如，LLM必须使用自然语言进行推理，或者应该使用最初为人类设计的基准来评估它们），我们展示了实证性的、非拟人的……

    arXiv:2502.09192v3 Announce Type: replace  Abstract: Anthropomorphism, or the attribution of human traits to technology, is an automatic and unconscious response that occurs even in those with advanced technical expertise. In this position paper, we analyze hundreds of thousands of research articles to present empirical evidence of the prevalence and growth of anthropomorphic terminology in research on large language models (LLMs). We argue for challenging the deeper assumptions reflected in this terminology -- which, though often useful, may inadvertently constrain LLM development -- and broadening beyond them to open new pathways for understanding and improving LLMs. Specifically, we identify and examine five anthropomorphic assumptions that shape research across the LLM development lifecycle. For each assumption (e.g., that LLMs must use natural language for reasoning, or that they should be evaluated on benchmarks originally meant for humans), we demonstrate empirical, non-anthropo
    
[^317]: 平衡全局质量与代词特定反馈的上下文感知机器翻译

    Balancing Global Quality and Pronoun-Specific Feedback for Context-Aware Machine Translation

    [https://arxiv.org/abs/2501.03008](https://arxiv.org/abs/2501.03008)

    该论文提出ProNMT方法，通过将句子级质量估计与代词位置的带符号置信度信号相结合进行奖励引导的迭代自训练，在上下文感知机器翻译中有效平衡了全局翻译质量与代词选择准确性，在英德和英法翻译任务上超越了标准监督微调。

    

    上下文感知机器翻译能够揭示代词选择所需的证据，但标准的微调方法并未明确优先考虑这些稀疏的语篇敏感决策。我们研究了ProNMT，这是一种奖励引导的迭代自训练方法，它将句子级质量估计与生成代词位置上的带符号置信度信号相结合。对于每个当前句子及其前文源语言上下文，ProNMT采样候选译文，使用无参考质量估计以及从参考译文获得的代词标签对其进行评分，并在得分最高的候选译文上进行微调。在过滤后的英德Europarl和英法News Commentary数据上，ProNMT在BLEU和COMET指标上优于上下文感知的监督微调。消融实验表明，仅使用代词反馈会在这些以代词为重点的数据上严重降低句子级翻译质量，而硬二值反馈的表现则不如基于置信度的反馈。

    arXiv:2501.03008v2 Announce Type: replace-cross  Abstract: Context-aware machine translation can expose the evidence needed for pronoun choice, but standard fine-tuning does not explicitly prioritize these sparse discourse-sensitive decisions. We study ProNMT, a reward-guided iterative self-training method that combines sentence-level quality estimation with a signed confidence signal at generated pronoun positions. For each current sentence and its preceding source context, ProNMT samples candidate translations, scores them using reference-free quality estimation together with a reference-derived pronoun label, and fine-tunes on the highest-scoring candidate. On filtered English--German Europarl and English--French News Commentary data, ProNMT improves over context-aware supervised fine-tuning on BLEU and COMET. Ablations show that pronoun-only feedback can severely degrade sentence-level translation quality on these pronoun-focused data, while hard binary feedback underperforms confi
    
[^318]: 面向大语言模型的CHAI：通过基于人工智能反馈的强化学习改进大语言模型中的语码混合翻译

    CHAI for LLMs: Improving Code-Mixed Translation in Large Language Models through Reinforcement Learning with AI Feedback

    [https://arxiv.org/abs/2411.09073](https://arxiv.org/abs/2411.09073)

    该论文提出CHAI通用框架，通过利用大语言模型作为标注者解决数据稀缺问题、基于AI反馈的强化学习（RLAIF）以及LLM生成的领域知识作为宪法进行迭代优化，使大语言模型在语码混合翻译任务上的性能超越最先进开源模型68.45%。

    

    大语言模型在许多任务中表现出强大的性能，但在理解语码混合语言方面仍然较弱。尽管存在这一局限，针对语码混合任务改进大语言模型的研究却鲜受关注。为了填补这一空白，我们提出了CHAI，这是一个用于提升大语言模型在语码混合任务上性能的通用框架，重点聚焦于语码混合翻译。CHAI利用了四个关键思想：首先，我们研究了使用大语言模型作为标注者的方法，以解决高质量语码混合数据集稀缺的问题；其次，我们利用这些大语言模型标注生成大规模偏好数据，并应用基于人工智能反馈的强化学习（RLAIF）来改进语码混合翻译；第三，我们将大语言模型生成的领域知识作为宪法，实现迭代的响应优化；第四，我们在真实世界的数据集和设置上进行了广泛的评估。结果表明，由CHAI驱动的大语言模型在最先进的开源模型基础上取得了68.45%的性能提升。

    arXiv:2411.09073v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) show strong performance across many tasks but remain weak at understanding code-mixed (CM) language. Despite this limitation, improving LLMs for CM tasks has received little attention. To address this gap, we propose CHAI, a general-purpose framework for enhancing LLM performance on CM tasks, focusing on CM translation. CHAI leverages four key ideas. First, we investigate the use of LLMs as annotators to address the scarcity of high-quality CM datasets. Second, we leverage these LLM annotations to generate large-scale preference data and apply reinforcement learning from AI feedback (RLAIF) to improve CM translation. Third, we incorporate LLM-generated domain knowledge as a constitution, enabling iterative response refinement. Fourth, we perform extensive evaluations on real-world datasets and settings. Results show that CHAI-powered LLMs outperform state-of-the-art open-source models by 68.45% on a
    
[^319]: 迈向安全代码生成：通过任务自适应漏洞建模与基于执行的基准测试弥合正确性与安全性之间的鸿沟

    Toward Secure Code Generation: Bridging Correctness and Security via Task-Adaptive Vulnerability Modeling and Execution-Based Benchmarking

    [https://arxiv.org/abs/2407.02395](https://arxiv.org/abs/2407.02395)

    该论文提出了基于执行的基准测试CodeSecEval（涵盖255个Python任务和77类CWE漏洞）以及智能体框架SecAwareCoder，通过任务自适应漏洞建模，在保证代码正确性的同时提升生成代码的安全性，弥合了正确性与安全之间的鸿沟。

    

    大型语言模型（LLM）越来越多地被用于程序合成，但它们生成的代码往往在功能上看似合理，却存在安全隐患。安全代码生成的进展一直受到现有基准测试的阻碍，这些基准要么规模小、不可执行、泄露漏洞修复细节，要么依赖有噪声的分析工具和主观判断，导致难以衡量安全性提升是否以牺牲正确性为代价。为了解决这些问题，我们提出了CodeSecEval，一个面向安全代码生成的基于执行的基准测试，包含255个Python任务，覆盖77类CWE（常见弱点枚举）漏洞类别。每个任务都提供了成对的不安全与安全实现，以及可执行的功能测试和针对漏洞的安全测试，从而能够对安全代码生成和不安全代码修复进行精确且可复现的评估。在CodeSecEval的基础上，我们提出了SecAwareCoder，一个基于智能体的框架，将代码生成转向“构造即安全”的方式。

    arXiv:2407.02395v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used for program synthesis, yet they often generate code that is functionally plausible but insecure. Progress in secure code generation has been hindered by benchmarks that are small, non-executable, leak mitigation details, or rely on noisy analyzers and subjective judgments, making it difficult to measure whether security improves without sacrificing correctness. We address these gaps with CodeSecEval, an execution-based benchmark for secure code generation, comprising 255 Python tasks spanning 77 CWE categories. Each task provides paired insecure and secure implementations together with executable functional and vulnerability-targeted security tests, enabling precise and reproducible evaluation of secure code generation and insecure-code repair. Building on CodeSecEval, we propose SecAwareCoder, an agent-based framework that shifts code generation toward secure-by-construction s
    
[^320]: 一个不完整的循环：语言模型中的演绎、归纳与溯因推理

    An Incomplete Loop: Deductive, Inductive, and Abductive Reasoning in Language Models

    [https://arxiv.org/abs/2404.03028](https://arxiv.org/abs/2404.03028)

    该研究通过四个语言模型和两类学习任务发现，语言模型的演绎推理（指令遵循）、归纳推理（少样本提示）与溯因推理（指令推断）三种能力之间存在强烈的不对称分离，表明这三种推理形式并未构成一个完整闭合的推理循环。

    

    现代语言模型可以通过不同的方式学习执行新任务：在指令遵循中，目标任务用自然语言明确描述；在少样本提示中，任务通过少量示例隐式指定；在指令推断中，语言模型被呈现上下文示例，并被提示在做出预测之前先生成自然语言的任务描述。这些过程可以被视为调用了不同形式的推理：指令遵循涉及演绎推理，少样本提示涉及归纳推理，而指令推断涉及溯因推理。这些不同的能力之间有何关联？通过在四个语言模型（来自gpt和llama系列）和两个学习问题（涉及算术函数和机器翻译）上的研究，我们发现不同类型的推理之间存在强烈的分离：语言模型有时可以有效地学习……

    arXiv:2404.03028v4 Announce Type: replace  Abstract: Modern language models (LMs) can learn to perform new tasks in different ways: in instruction following, the target task is described explicitly in natural language; in few-shot prompting, the task is specified implicitly with a small number of examples; in instruction inference, LMs are presented with in-context examples and are then prompted to generate a natural language task description before making predictions. Each of these procedures may be thought of as invoking a different form of reasoning: instruction following involves deductive reasoning, few-shot prompting involves inductive reasoning, and instruction inference involves abductive reasoning. How do these different capabilities relate? Across four LMs (from the gpt and llama families) and two learning problems (involving arithmetic functions and machine translation) we find a strong dissociation between the different types of reasoning: LMs can sometimes learn effectivel
    
[^321]: 核指代解析模型的受控重新评估

    A Controlled Reevaluation of Coreference Resolution Models

    [https://arxiv.org/abs/2404.00727](https://arxiv.org/abs/2404.00727)

    基于对预训练语言模型大小的控制，我们发现基于编码器的核指代解析模型在准确性和推理速度方面优于更近期的基于解码器的模型，而在基于编码器的模型中，最老的模型在跨域文本体裁中表现最佳。

    

    所有最先进的核指代解析（CR）模型都涉及微调预训练语言模型。一个CR模型优于另一个的出色性能是由语言模型选择还是其他因素（如特定于任务的架构）造成的，由于缺乏标准化的实验设置，这是很难或不可能确定的。为了解决这种模糊性，我们系统评估了五个CR模型，并控制了一些设计决策，包括每个模型使用的预训练语言模型。当控制语言模型大小时，基于编码器的CR模型在准确性和推理速度方面优于更近期的基于解码器的模型。令人惊讶的是，在基于编码器的CR模型中，较近期的模型并不总是更准确，而我们测试的最老的CR模型在跨域文本体裁中表现最佳。我们得出结论，控制语言模型的选择可以减少大部分，但并非全部，

    arXiv:2404.00727v1 Announce Type: new  Abstract: All state-of-the-art coreference resolution (CR) models involve finetuning a pretrained language model. Whether the superior performance of one CR model over another is due to the choice of language model or other factors, such as the task-specific architecture, is difficult or impossible to determine due to lack of a standardized experimental setup. To resolve this ambiguity, we systematically evaluate five CR models and control for certain design decisions including the pretrained language model used by each. When controlling for language model size, encoder-based CR models outperform more recent decoder-based models in terms of both accuracy and inference speed. Surprisingly, among encoder-based CR models, more recent models are not always more accurate, and the oldest CR model that we test generalizes the best to out-of-domain textual genres. We conclude that controlling for the choice of language model reduces most, but not all, of 
    
[^322]: Syntactic Ghost：一种对预训练语言模型进行的无感知通用后门攻击

    Syntactic Ghost: An Imperceptible General-purpose Backdoor Attacks on Pre-trained Language Models

    [https://arxiv.org/abs/2402.18945](https://arxiv.org/abs/2402.18945)

    论文提出了一种名为Syntactic Ghost的新方法，实现了对预训练语言模型进行无感知和通用的后门植入。

    

    预训练语言模型（PLMs）被发现容易受到后门攻击，可以将漏洞转移到各种下游任务中。然而，现有的PLM后门攻击采用明显的触发器，在手动对准的情况下进行，因此在效果、隐匿性和通用性方面无法同时满足期望目标。本文提出了一种新方法，实现了不可见和通用的后门植入，称为Syntactic Ghost（简称为synGhost）。具体来说，该方法敌意地使用具有不同预定义句法结构的毒害样本作为隐蔽触发器，然后将后门植入到预训练表示空间，而不会破坏原始知识。毒害样本的输出表示在特征空间中尽可能均匀地分布，通过对比学习形成广泛的后门。此外，在亮

    arXiv:2402.18945v1 Announce Type: cross  Abstract: Pre-trained language models (PLMs) have been found susceptible to backdoor attacks, which can transfer vulnerabilities to various downstream tasks. However, existing PLM backdoors are conducted with explicit triggers under the manually aligned, thus failing to satisfy expectation goals simultaneously in terms of effectiveness, stealthiness, and universality. In this paper, we propose a novel approach to achieve invisible and general backdoor implantation, called \textbf{Syntactic Ghost} (synGhost for short). Specifically, the method hostilely manipulates poisoned samples with different predefined syntactic structures as stealth triggers and then implants the backdoor to pre-trained representation space without disturbing the primitive knowledge. The output representations of poisoned samples are distributed as uniformly as possible in the feature space via contrastive learning, forming a wide range of backdoors. Additionally, in light 
    
[^323]: 句法结构的普遍拓扑规律性

    Universal Topological Regularity of Syntactic Structures

    [https://arxiv.org/abs/2302.00129](https://arxiv.org/abs/2302.00129)

    该论文通过分析124种语言的依存树，发现句法结构具有偏离随机性的普遍拓扑规律（更高的结构稳健性与更低的分支异质性），并借助亚线性优先连接模型证明这些规律可从认知驱动的增量语法编码过程中自然涌现，无需直接优化即可实现交际效率。

    

    尽管句法依存树被广泛使用，但其组织背后的原则仍然鲜为人知。本文分析了来自124种类型学、谱系和地理上多样化的语言的依存树。研究发现，这些依存树的拓扑结构系统性地偏离随机性：相对于均匀采样的随机树，依存树表现出更高的结构稳健性和更低的分支异质性。作者提出，这些普遍规律可以自然地源于增量式的语法编码过程，并用亚线性优先连接机制对这一过程进行建模。该模型准确再现了观测到的拓扑结构。更广泛地说，这些结果表明句法的普遍统计特性可以从一个简单的、具有认知动机的生成过程中涌现出来，并进一步阐明了一个更普适的“构建即效率”原则：交际高效的句法结构无需直接优化即可自然涌现。

    arXiv:2302.00129v2 Announce Type: replace  Abstract: Despite their widespread use, the principles governing the organisation of syntactic dependency trees remain poorly understood. I analyse dependency trees from 124 typologically, genetically, and geographically diverse languages. Their topology departs systematically from randomness. Relative to uniformly sampled random trees, dependency trees exhibit greater structural robustness and lower branching heterogeneity. I propose that these universal regularities emerge naturally from incremental grammatical encoding. I model this process using sublinear preferential attachment. The model accurately reproduces the observed topology. More generally, the results demonstrate how a universal statistical property of syntax can emerge from a simple, cognitively motivated generative process. They further illustrate a broader principle of efficiency by construction: communicatively efficient syntactic structures can emerge without direct optimisa
    

