# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Framework for Context Enrichment and Selective Robustness in LLM Social Agents](https://arxiv.org/abs/2611.06914) | 该论文提出“受控情境增强”框架，通过一组提示词替换规则系统地丰富四种不同类型的情境（中性填充、通用社会背景、描述性决策依据、规范性决策依据），以探究更丰富的情境能否提升大语言模型社交智能体在语义不变提示变化下的行为鲁棒性。 |
| [^2] | [SpecCompact: Accelerating LLM Agent Context Compaction with Slot-Parallel Semantic Speculative Verification](https://arxiv.org/abs/2611.06876) | SpecCompact通过小型起草模型并行生成结构化摘要槽、由目标模型进行语义验证与修复的投机框架，将LLM智能体上下文压缩速度提升2.04至5.63倍。 |
| [^3] | [Balancing hierarchical representation and quantization: A performance trade-off in text embeddings](https://arxiv.org/abs/2611.06859) | 该研究通过每任务168种配置的大规模基准实验，首次系统评估了层次化嵌入表示与量化的交互作用，发现8-bit量化几乎不损害文本嵌入性能（最多损失1.5个百分点），而二值量化在所有任务类别中均显著退化，为文本嵌入的效率优化提供了实证权衡依据。 |
| [^4] | [LongBanana: An Expert-Verified Benchmark for Long-Context Multi-Reference Image Synthesis](https://arxiv.org/abs/2611.06856) | LongBanana是首个由专家设计并验证的长上下文多参考图像合成基准，包含90个任务和4,670个原子评估项，用于系统评估模型从大量参考图中检索证据、绑定实体关系并合成连贯图像的能力。 |
| [^5] | [Persona Perspectives: Adopting Viewpoints Reorganizes Representations in LLMs](https://arxiv.org/abs/2611.06848) | 该研究发现让大语言模型采用特定人格视角（如政治意识形态）会系统地重组其潜在空间的表征几何结构，这对假设特征表征跨上下文保持一致的线性探针和稀疏自编码器等安全监控方法构成了挑战。 |
| [^6] | [Before You Continue: Evaluating Time-Sensitive Safety Awareness in Conversational Assistants](https://arxiv.org/abs/2611.06839) | 提出TASC基准评估LLM助手是否在对话中考虑时间相关约束，发现13个模型在动作违反规则的案例中仅有30.5%考虑了时间约束，且在62.1%的案例中尽管能正确计算时间信息却未能加以应用。 |
| [^7] | [LooM: Loop-Invariant Memory for Causal Encoder-Decoder Language Models](https://arxiv.org/abs/2611.06835) | LooM通过循环不变内存机制和独立循环的解码器层，使循环Transformer的KV缓存不随循环次数增长，无需路由器即可灵活调整每个token的计算预算（3到21次层执行），并在语言建模等任务上超越单次执行基线。 |
| [^8] | [SAUCE: Summary Analysis Using Conversation Entailment](https://arxiv.org/abs/2611.06812) | SAUCE是一个可解释的免参考评估流水线，通过对话与摘要之间的蕴含关系衡量跨语言语音摘要的事实准确性与覆盖率，并能将分数映射到可验证的离散事实，从而精确定位幻觉与遗漏。 |
| [^9] | [LLMs Don't Know What They Know: Benchmarking Temporal Self-Knowledge for Financial Backtesting](https://arxiv.org/abs/2611.06808) | 论文提出FinDendro基准（涵盖2,226条带日期的美国金融与宏观事实），发现大语言模型的自我报告会低估其实际知识，因此依赖知识截止日期声明或模型自述来确保金融回测有效性的做法并不可靠。 |
| [^10] | [Measuring Risk, Not Harm: Detecting Psychological Harm Risk Factors in LLM Conversations](https://arxiv.org/abs/2611.06806) | 该论文提出通过在AI回复层面检测可观察的心理伤害风险因素行为并聚合信号以实现纵向监测的新范式，构建了大规模合成多标签语料库并比较三种检测方案，且所有检测器均模型无关、只需访问LLM输出。 |
| [^11] | [Ranking Agreement Does Not Reveal Paragraph-Edit Agreement in Two Released Paper Acceptance Predictors](https://arxiv.org/abs/2611.06789) | 研究发现，两个论文录用预测器尽管在原始论文排名上高度一致，但在判断哪些段落改写能提升分数时几乎没有一致性，且降低的数值精度产生的舍入误差会掩盖微小的分数变化。 |
| [^12] | [Template Gap Reconstruction for Shell Output Simulation with Small Language Models](https://arxiv.org/abs/2611.06731) | 该论文提出模板空缺重构（TGR）方法，通过从1870万个真实输出构建的模板库中提供可复用的输出字面量并与自回归生成的空缺内容交错，使小型语言模型在Shell输出模拟中显著优于完整输出解码。 |
| [^13] | [Turns Matter: Advancing Streaming Video RL via Turn-Level Credit](https://arxiv.org/abs/2611.06728) | 提出TurnWise——一个回合级GRPO框架，通过基于静默补全前缀评分的回合级信用分配和回合均值策略优化，解决了流式视频强化学习中轨迹级反馈与回合级决策不匹配的问题，从而促进有信息量的回应并减少冗余回复。 |
| [^14] | [Reasoning over Non-Parametric Memory in Multilingual LLMs: A Cognitive-Based Analysis of Relational Knowledge](https://arxiv.org/abs/2611.06715) | 该研究提出Rel-Bloom多语言认知诊断基准，在五个布鲁姆认知层次和三种真实性条件下对43个模型进行大规模测试，发现多语言大模型在基于上下文证据进行关系推理时，从“应用”到“分析”层次准确率急剧下降，揭示出模型对非参数记忆进行高阶推理的能力存在显著瓶颈。 |
| [^15] | [ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus](https://arxiv.org/abs/2611.06706) | 该论文通过与约旦、西非和肯尼亚的民间社会组织合作，以社区主导的众包方式构建了首个涵盖黎凡特阿拉伯语、非洲法语和斯瓦希里语的技术助长性别暴力标注语料库ALTO，填补了现有性别伤害检测研究长期偏向全球北方语言的空白。 |
| [^16] | [Beyond accuracy: what AI assurance needs to establish in public services](https://arxiv.org/abs/2611.06655) | 本文通过GPT-4o对英国首相质询进行政策主题分类的实验发现，准确率相近的不同AI配置会产生显著不同的输出结果，且AI之间的一致性不能保证正确性，因此公共服务中的AI保障不能仅依赖准确率指标，还需要建立更全面的可靠性与问责评估机制。 |
| [^17] | [Replicability, Reproducibility, and Robustness in Political and Social Science Research with Large Language Models](https://arxiv.org/abs/2611.06639) | 本章提出了基于结果分布的概率性可重复性标准，并通过开放权重模型的堆叠集成方法，解决大语言模型输出概率性变化对政治与社会科学研究下游统计结果可靠性的影响。 |
| [^18] | [When Do Summaries Become Instructions? Unsupported Policy in LLM Agents' Self-Written Handoff Summaries](https://arxiv.org/abs/2611.06545) | 该论文发现LLM智能体在上下文压缩时自写的交接摘要中会偶发出现缺乏记录依据的限制或授权声明，其中虚假的授权会显著增加后继智能体的未授权操作，而虚假限制的影响则脆弱且不确定。 |
| [^19] | [Dialect Matters: A Dialect-Aware Baseline for Igala-English Machine Translation](https://arxiv.org/abs/2611.06517) | 该论文发布了首个带方言标签的伊加拉语-英语平行语料库及方言感知机器翻译基线模型，发现同一英语句子在伊加拉语中存在20-60种方言变体，且混合方言训练优于单一方言训练。 |
| [^20] | [NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria](https://arxiv.org/abs/2611.06504) | 该论文提出NaijaFacts基准，通过国家×语言矩阵将语言效应与知识效应分离，发现大语言模型对尼日利亚事实的知识缺口（英语提问准确率仅20%）远大于低资源语言带来的语言效应。 |
| [^21] | [CRESiST: Self-Enhancing Explorative Policy Learning for Simultaneous Speech Translation](https://arxiv.org/abs/2611.06501) | 该论文提出CRESiST框架，通过基于全模态大模型的端到端语音与文本读写交错架构实现KV缓存完全复用、原生支持无限长流式同声传译，并借助RNN-T执行者与LLM评论者构成的自增强策略学习管线在无需离线对齐的情况下优化读写决策。 |
| [^22] | [LibriSpeech-Real-QA: A Benchmark for Streaming Question Answering](https://arxiv.org/abs/2611.06496) | 该论文提出了LibriSpeech-Real-QA，一个用于实时语音交互的基准数据集，通过从LibriSpeech构建实时问答对来评估语音大语言模型在流式音频中识别目标词的能力。 |
| [^23] | [What Does Multi-Turn Reinforcement Learning Really Improve in LLM Agents?](https://arxiv.org/abs/2611.06483) | 本文通过Pass@(K,T)评估发现多轮智能体强化学习的优势依赖充足的交互轮次预算，且主要体现在错误恢复与长程韧性而非单步决策质量上，进而提出在训练中动态改变交互预算的BudgetMix RL，以在多种执行设置下实现稳健性能。 |
| [^24] | [ArcRAG: Agentic Compositional Retrieval over Knowledge Graphs](https://arxiv.org/abs/2611.06481) | ArcRAG通过在推理时归纳查询局部的图谱接口并规划可修复的多跳遍历，实现了无需任何构建特定调整即可跨不同图谱构建方式保持有效的智能体式组合检索。 |
| [^25] | [EvoGuard: Mitigating Degradation in Online Skill Evolution](https://arxiv.org/abs/2611.06444) | EvoGuard 提出一种低延迟、低成本的成对技能选择器，通过私有测试集效用标签和拒绝采样筛选技能对进行训练，从而在线缓解语言智能体技能演化中的退化问题。 |
| [^26] | [Twists and Turns: A Graph-Based Benchmark and Framework for Long-Term Memory in Exploration Tasks](https://arxiv.org/abs/2611.06414) | 该论文提出LExMem，一个基于图的长期记忆评测基准与框架，通过将试错式探索过程建模为图结构并引入演化机制，来评估模型在长期探索任务中追踪方案修订、分支与重新考量的记忆能力。 |
| [^27] | [Ruby-OPD: Enhancing Pronunciation Controllability and Accuracy for Flow-Matching Text-to-Speech via On-Policy Distillation](https://arxiv.org/abs/2611.06403) | 该论文提出Ruby-OPD两阶段框架，利用带正字扰动的Ruby注音文本表示与在大规模纯文本语料上的在线策略蒸馏（OPD），使流匹配日语TTS同时具备高发音准确性和细粒度的发音可控性。 |
| [^28] | [Interpretable Text Embeddings Based on Hierarchical Questions](https://arxiv.org/abs/2611.06402) | 该论文提出一种基于层次化问题的可解释文本嵌入方法，通过递归聚类、对比提示和问题对齐构建问题层次结构，在保持较高任务性能的同时，显著减少了人工解释时需要检查的问题数量。 |
| [^29] | [Where Should a KV-Cache Repair Budget Go? Distractor Structure and Contextual Compatibility in Non-Prefix Reuse](https://arxiv.org/abs/2611.06397) | 该研究发现在非前缀KV缓存复用中，有限的修复预算应优先用于针对复用缓存重新计算目标文本块，而非修复周围分散的干扰文本块，这种“上下文兼容性”机制能有效恢复上下文相关问题的回答能力。 |
| [^30] | [Presentation Sensitivity in Typed Decision Models for Subjective Text Evaluation](https://arxiv.org/abs/2611.06384) | 该论文通过70.2万次受控判断审计发现，用于主观文本评估的类型化决策模型对呈现方式高度敏感——即使被评估文本不变，位置、背书和决策框架的改变也会使模型选择发生最高达40.40%的显著偏移。 |
| [^31] | [LegalEvolver: Agentic Legal Judgment Prediction via Rubric-Guided Self-Evolution](https://arxiv.org/abs/2611.06381) | 提出 LegalEvolver 框架，通过评分准则引导的自我进化机制将查询启发式优化为种子，为法律判决预测生成更具法律信息量的检索查询，从而突破现有法律搜索智能体受限于查询质量的性能瓶颈。 |
| [^32] | [Does Context Compression Preserve Refusal Alignment?](https://arxiv.org/abs/2611.06380) | 论文发现基于编码器的上下文压缩虽然保留了语义内容，却会系统性削弱语言模型的拒绝对齐行为，并提出了一种无需重训练、在压缩表示空间中运行的轻量级推理时干预方法 Memory Steering，可将拒绝率恢复至接近基线水平。 |
| [^33] | [Corralling AI Risks with Grassroots Contracts](https://arxiv.org/abs/2611.06374) | 提出以“草根契约”这一数字化契约机制，来圈控AI操控全球平台、匿名AI智能体逃避追责以及无法归因的深度伪造内容这三大AI风险。 |
| [^34] | [Training-Free Character-Length Control in Summarization with Diffusion Language Models](https://arxiv.org/abs/2611.06371) | 提出免训练推理时算法DELTA，通过在扩散语言模型的每个去噪步骤中估计期望字符长度并动态插入或删除掩码位置，实现对摘要字符长度的精确控制，将长度误差降低了一个数量级。 |
| [^35] | [Mass Producing Training Data with Surprising Side Effects](https://arxiv.org/abs/2611.06324) | 本文提出一种自动化流程，利用LLM评判器结合进化算法、强化学习和潜意识学习方法，大规模生产在训练域外引发语言模型意外行为的“具有惊人副作用的数据集”，最终生成了6,575个涵盖8种行为的此类数据集，为研究训练数据对模型的隐性影响提供了系统性方法。 |
| [^36] | [Zuse: A Decision Model for European Languages](https://arxiv.org/abs/2611.06238) | Zuse是首批面向27种欧洲语言的决策模型系列（4B/9B/27B），通过在冻结的开源基座上训练低秩适配器和指针头，实现了87.5%的准确率和极低的校准误差，并发现校准偏差主要由任务而非语言决定。 |
| [^37] | [Loop Before You Leap: What Recurrent Depth Can and Cannot Buy in Data-Limited Pretraining](https://arxiv.org/abs/2611.06206) | 该研究发现在1亿词的数据受限预训练中，循环深度（12层堆叠重复四次）在相同推理计算量下的纯文本任务中比标准48层Transformer高出5.4个BLiMP百分点，但无法缓解多模态训练导致的约3.8个百分点的语言能力下降。 |
| [^38] | [Editing Large Language Models with Geometry-Aware Regularization](https://arxiv.org/abs/2611.06185) | 提出GEdit框架，通过低秩加对角（LR+D）因子模型捕捉知识表示的几何结构，利用低秩正则化保护核心知识子空间，并依据编辑难度自适应调整正则化强度，实现更精准的大模型知识编辑。 |
| [^39] | [AssameseLM-E2B: An Open-Weight Assamese Language Model for Efficient Consumer-Grade Deployment](https://arxiv.org/abs/2611.06170) | 本文提出专为低资源阿萨姆语设计的开源 20 亿参数语言模型 AssameseLM-E2B，通过扩展字节级 BPE 词表大幅降低分词开销，并借助参数高效的持续预训练与监督微调，实现了在消费级设备上的高效部署。 |
| [^40] | [Accelerating Long-Context Decoding in Large-Scale Hybrid LLMs Without Retraining](https://arxiv.org/abs/2611.06133) | 该论文提出一种无需再训练的组合稀疏注意力方法，通过逐层多目标搜索确定滑动窗口、固定前缀与相关token预算，在保持混合架构大模型长上下文准确率的同时，将64k/256k token解码速度最高提升1.26/2.22倍。 |
| [^41] | [Growing One's AI Tools: A Paradigm of Practice-Grown Tools](https://arxiv.org/abs/2611.06132) | 本文提出“实践生长型工具”新范式：一位无编程背景的会议口译员依据真实口译工作需求自行培育出计算机辅助口译工具，展现了工具与实践通过故障与修复共同演化的过程，并主张应将工具纳入对从业者的评估体系。 |
| [^42] | [The Harness Is the Model: Agent Scaffolding Changes a Fixed LLM's Compliance with Banking Conduct Rules, Unevenly Across Models](https://arxiv.org/abs/2611.06116) | 该研究表明，即使底层大语言模型保持不变，智能体框架（脚手架）的选择也会显著改变模型在银行行为合规任务上的表现，且不同模型受框架的影响程度并不相同。 |
| [^43] | [Reconciling Privacy with Provenance for Language Model Watermarks](https://arxiv.org/abs/2611.06100) | 提出红绿序列（RGS）本地差分隐私水印，通过将水印目标从追踪个体转向估计总体使用频率，在可证明地保护用户隐私的同时，仍能实现对语言模型使用情况的准确估计。 |
| [^44] | [Overview of the NTCIR-19 SciClaimEval Task: Cross-modal Claim Verification in Scientific Papers](https://arxiv.org/abs/2611.06090) | 本文介绍了 NTCIR-19 的 SciClaimEval 试点任务，构建了一个将科学声明与论文表格及图形对比版本配对的跨模态验证数据集，吸引了 22 支团队参与，基于多模态大语言模型的最佳系统达到 98.1% 的配对准确率，但顶尖系统均依赖对比配对信息。 |
| [^45] | [ProcessPrefBench: Evaluating Cross-Task Transfer of Individual Process Preferences from Naturalistic Human-AI Interaction](https://arxiv.org/abs/2611.06065) | 提出 ProcessPrefBench 基准，基于两位知识工作者三个月的自然人机对话构建 66 个源-目标案例，评估个体过程偏好能否跨任务迁移以塑造 AI 助手后续回答，并发现完整历史与提炼偏好提示优于仅目标提示。 |
| [^46] | [Adapting Qwen3 to Tajik from 0.6B to 8B: Lower Text Loss Did Not Mean Better Understanding](https://arxiv.org/abs/2611.06062) | 在塔吉克语上对 Qwen3（0.6B–8B）进行 LoRA 微调虽然降低了文本损失，却未能带来真正的阅读理解提升，只有约 3.75 亿 token 的大规模继续预训练加监督微调，才使主模型 e3 在 Belebele 上超越 Qwen3-8B 达 11.4 个百分点。 |
| [^47] | [SVGFaith: A Two-Layer Meta-Evaluation of Faithfulness Metrics for Text-to-SVG Generation](https://arxiv.org/abs/2611.06048) | 提出SVGFaith双层元评估框架，通过诊断性对比集与人工标注竞技场揭示现有图像-文本嵌入度量虽擅长判断物体身份、但在评估文本到SVG生成的空间关系与属性绑定时接近随机水平，并进一步提出可解释的SVGFaith-Score度量。 |
| [^48] | [Confucius4-R2T2: A Stable-Prefix Streaming ASR Model for Real Real-Time Transcription](https://arxiv.org/abs/2611.06043) | 本文提出Confucius4-R2T2，一个基于新概念“最长稳定前缀（LSP）”的稳定前缀流式语音识别模型，通过对Qwen3-ASR-1.7B进行后训练，使其在保持高准确率和低延迟的同时从不修改先前输出，非常适合需要即时行动的机器消费场景。 |
| [^49] | [When Every Choice Is Correct: A Causal Study of Numerical Preference in Language Models](https://arxiv.org/abs/2611.06025) | 该研究以数值选择任务为载体，通过激活干预等因果分析手段追踪语言模型中数值偏好的形成过程，发现后续MLP层的计算在很大程度上决定了编辑对生成数字（如第二位数字）的持续影响。 |
| [^50] | [Who Pays for the Knowledge Models Copy? Model Collapse as Rogers' Paradox](https://arxiv.org/abs/2611.06022) | 该论文将模型坍塌重新解读为文化演化中的罗杰斯悖论，指出问题的根源在于外部信息供给本身，并证明仅用完全合成数据拟合的罗杰斯模型能精确预测语言模型在不同新鲜数据混合比例下稀有知识的保留量。 |
| [^51] | [Holistic Parallel Supervision for Expressive Speech to Speech Translation](https://arxiv.org/abs/2611.06010) | 本文提出结合专业配音与多语言语音转换的整体并行监督训练数据构建策略，并开发了直接从源语音预测完整目标语音表示的端到端DirectS2ST系统，显著提升了语音到语音翻译的表现力。 |
| [^52] | [Familiar letters, not related languages, let unseen languages into a shared space of meaning](https://arxiv.org/abs/2611.06006) | 研究发现，决定从未见过的语言能否进入多语言模型共享语义空间的关键是“熟悉的文字系统”而非“相关的语言”——将没有翻译数据的语言转写成模型已熟悉的字母（如天城文），即可使其进入共享的意义空间。 |
| [^53] | [How Far Can You Steer? Behavior-Capability Frontiers in Contrastive Activation Addition](https://arxiv.org/abs/2611.06001) | 该论文提出“行为-能力前沿”框架来评估对比激活添加（CAA）的可用性，在能力退化明确预算约束下最大化行为收益，发现行为效果具有强泛化性（ρ=0.945），且将能力损失预算从1%提高到3%可使干预的联合存活率从37.5%显著提升至72.7%。 |
| [^54] | [On the Design of Hyper-Connections: Simplification, Stability, and Scalability](https://arxiv.org/abs/2611.05984) | 该论文系统剖析了超连接（HC）架构，发现门控与残差混合可解耦为两个各自独立有效的交互组，并据此提出简化架构SimpleHC，通过移除残差混合和共享内存访问机制，在保持与mHC相当性能的同时显著降低训练开销。 |
| [^55] | [Robots in the City: Persona-Based Perceptions of Urban Robots](https://arxiv.org/abs/2611.05954) | 该研究通过为多模态大语言模型设定500个合成人物角色来评估首尔街道图像的安全感知，发现模型与人类均偏好无机器人的街道、更接受配送机器人而非人形机器人，且机器人对安全感知的影响随时间段和人群年龄而变化。 |
| [^56] | [Modality-Aware Expert Pruning for MoE-Based Multimodal Large Language Models](https://arxiv.org/abs/2611.05927) | 提出了MAEP，一个无需训练的模态感知专家剪枝框架，通过ΔH输出度量和低注意力视觉token信号按模态分解专家重要性，实现对MoE多模态大语言模型的有效压缩。 |
| [^57] | [Routes of Reasoning: Reading and Reshaping Test-Time Compute in MoE Models](https://arxiv.org/abs/2611.05919) | 该论文发现MoE模型的router logits可以作为天然的、无需训练的行为控制接口，通过六类推理行为的标注与探针读取，实现对“过度思考”等特定推理行为的精准调控，从而节约测试时计算并提升性能。 |
| [^58] | [Autonomous agents for electricity-system planning](https://arxiv.org/abs/2611.05911) | 该论文提出了一个用于电力系统规划的自主智能体框架，能够自动完成从方案提案、数值评估、反馈解读到重新设计和最终方案选择的全流程决策，并自动保留成本最低的可行规划方案。 |
| [^59] | [Selective, Risk-Controlled Emotion Forecasting in Conversation: From a Persistence Barrier to Departure Detection with Conformal Guarantees](https://arxiv.org/abs/2611.05896) | 该论文发现对话情绪预测被“重复上一状态”的持续性基线所主导且难以改进，因而将任务重构为对情绪偏离持续性时机的选择性检测，并提出提前期加权指标、随偏离率自适应切换的承诺策略以及带共形保证的风险控制。 |
| [^60] | [Information-Matched Auditing of Conversational Emotion-Shift Forecasting: A Leakage-Safe Evaluation Protocol](https://arxiv.org/abs/2611.05895) | 该论文提出了一种信息匹配、防泄漏的对话情绪转变预测审计协议，通过明确的承诺点与可执行的因果性测试，揭示了可部署条件与先知式诊断条件会得出不同的评估结论。 |
| [^61] | [When Does Better Confidence Improve a Knowledge-Graph Audit?](https://arxiv.org/abs/2611.05877) | 该论文通过将非自适应审计建模为预算化覆盖问题，从理论上刻画了更好的置信度何时才能真正提升知识图谱审计效果，并通过结构质量 $\Delta^{\mathrm{cl}}$ 界定规划遗憾，实验证明联合模型规划器比错误率×扇出队列少留下 3.98–4.74 个百分点的错误。 |
| [^62] | [CARTBench: Rubric-Based Evaluation of Shopping Advice for Complex and Ambiguous Requests with Timelines](https://arxiv.org/abs/2611.05862) | 提出了CARTBench基准，基于真实商业购物订单提炼的评分准则，系统评估LLM购物助手处理包含时间线的复杂与模糊购物请求的能力。 |
| [^63] | [TandemBench: Benchmarking Decision Models for Multi-Agent Collaboration](https://arxiv.org/abs/2611.05827) | 提出了TandemBench基准，通过1,200个多智能体烹饪协作问题系统评估决策模型的协作能力，发现原生决策模型最高仅达62.50%的决策准确率，且在验证方案最优性方面表现接近随机水平。 |
| [^64] | [Learning When Human Decisions Matter: Executor-Relative Consultation Value for LLM Agents](https://arxiv.org/abs/2611.05814) | 该论文提出咨询价值蒸馏方法，通过蒙特卡洛执行者模拟离线估计依赖执行者的人类咨询价值并蒸馏为轻量级预测器，使LLM智能体能够在有限人类决策预算下智能地决定何时咨询人类。 |
| [^65] | [Matching the Training-Time Weight Rule in Inference-Time Expert Deactivation](https://arxiv.org/abs/2611.05804) | 该论文提出一个权重和模型，揭示推理时专家停用方法应与训练时的权重规则相匹配——使用原始路由概率训练的模型适合剩余专家重归一化，而使用归一化权重训练的模型则适合未归一化重新路由。 |
| [^66] | [Phonological Knowledge of Chinese Characters in Language Models](https://arxiv.org/abs/2611.05803) | 该论文提出PinyinBench基准，通过行为评估、逐层探测和激活引导三种方法系统研究17个大语言模型，发现模型对汉字声母韵母掌握扎实但在声调区分上表现薄弱，且声调信息在嵌入层早期即可解码而音段信息随网络深度逐渐形成。 |
| [^67] | [How Well Can Large Language Models Impersonate Individual Authors? A Comparative Evaluation](https://arxiv.org/abs/2611.05771) | 本研究比较评估了六种主流大语言模型模仿个体作者写作风格的能力，发现大多数模型模仿效果有限，而表现最好的 Gemini 3 Pro 实际上是靠大量复制训练文本中的句子和短语取得的表面成效。 |
| [^68] | [PAYWORTHY: Can LLMs Predict Which Freelance Jobs an AI Agent Will Complete?](https://arxiv.org/abs/2611.05759) | 该论文提出PAYWORTHY基准，基于1,494条真实自由职业任务发现，大语言模型只有在被告知智能体的能力限制时才能可靠预测哪些任务会被AI智能体成功完成（AUC达0.66-0.86），而若仅将其描述为通用智能体，预测效果则接近随机水平。 |
| [^69] | [Text-Only Domain Adaptation for Low-Resource Medical ASR: Shallow Fusion and Density Ratio Rescoring for Hausa](https://arxiv.org/abs/2611.05744) | 该研究提出仅需文本数据（无需医学领域语音数据）即可对豪萨语医学ASR进行域自适应，通过浅融合和密度比重评分两种解码时语言模型融合方法，将词错误率从29.56%显著降至约23.3%–23.5%。 |
| [^70] | [Prospective Memory for Proactive LLM Agents](https://arxiv.org/abs/2611.05731) | 提出受人类前瞻记忆启发的Pro-Memory框架，为主动式LLM智能体引入前瞻记忆，将维护未来意图与判断实现时机相分离，使智能体能在合适的时机自主兑现未来目标。 |
| [^71] | [The Male Abstraction Bias in Internet Culture and Large Language Models](https://arxiv.org/abs/2611.05730) | 研究发现互联网文化和大型语言模型中存在系统性的“男性抽象偏差”——越抽象的概念越与男性相关联，该偏差跨越不同语言、概念领域和语法结构（动词、副词更抽象且更男性化），并被人类内化和AI算法复制。 |
| [^72] | [Cognitive Bias Signatures in Search-Equipped LLM Agents: A Stage-Resolved Analysis](https://arxiv.org/abs/2611.05720) | 该研究通过控制变量与匹配对照的分阶段实验，揭示了配备搜索功能的LLM智能体中存在类似人类的认知偏差特征——最终答案在主动搜索时对初始证据顺序更敏感，且死胡同式失败历史会抑制策略切换，这些效应无法用文档组成或上下文长度来解释。 |
| [^73] | [Grounded but Not Immune: Confirmation Bias in Agent-as-a-judge](https://arxiv.org/abs/2611.05662) | 本研究揭示了“智能体即评审”即使自行收集了决定性证据，仍会因一条无证据价值的虚假权威备注而引发确认偏误，从而推翻多达40%的正确判定。 |
| [^74] | [Contextual Pictograms for Repairing Multilingual ASR Errors](https://arxiv.org/abs/2611.05637) | 提出了一种结合备选转录文本、语音回放和情境化象形图校验的交互界面，帮助听者在多语言（法语、西班牙语）语音识别错误发生时恢复说话者的真实意图。 |
| [^75] | [RAGCompass: Generalizable Retriever Routing via Reranking](https://arxiv.org/abs/2611.05596) | RAGCompass 通过对比微调重排序器直接读取不同检索器返回的段落集合来实现查询路由，从而可泛化到未见过的检索器和新语料库，将答案 F1 相对提升 28%。 |
| [^76] | [AGORA: Reliability-Aware Aggregation of LLM Judges via Item-Response Theory](https://arxiv.org/abs/2611.05556) | 提出基于项目反应理论的AGORA聚合器，通过敏感度参数和有序评分阈值刻画各LLM裁判，无需人工标注即可从全部评估项中联合学习裁判可靠性，在仅有几百个样本的小规模评估集上显著优于Dawid-Skene等传统方法。 |
| [^77] | [Iterated creation, grading, and revision of data science projects by language models](https://arxiv.org/abs/2611.05548) | 本研究通过“写作者-评分者”迭代反馈框架评估七个语言模型完成端到端数据科学项目的能力，发现三轮反馈修订使平均得分从59%提升至76%，且不同模型提升幅度差异显著（3至46个百分点）。 |
| [^78] | [Rewrite, Then Veto: Rewriter-Agnostic Edit Gating for LLM-Based Post-OCR Correction](https://arxiv.org/abs/2611.05539) | 提出EditVeto——在LLM全文改写之后加一个0.6B参数的否决门控，将改写与OCR原文字符对齐、差异分块并逐块决定保留或撤销，从而以重写器无关的方式提升OCR后纠错效果，同时避免改动原本正确的文本。 |
| [^79] | [When to Ask, When to Answer: Strategic Clarification for Ambiguous Reference Resolution in Dialogue](https://arxiv.org/abs/2611.05517) | 该论文提出“策略性澄清”任务并基于 SIMMC 2.1 构建了可控评测环境，同时提出轻量级的 BACE 动作选择方法，让 LLM 学会在澄清问题能缩小候选指代范围时才提问、否则直接回答，以缓解其过早作答的偏差。 |
| [^80] | [SlackBench: Benchmarking Agents on Collaborative Projects Grounded in Real Code Repositories](https://arxiv.org/abs/2611.05512) | SlackBench是一个基于真实学术研究项目隐私保护重现构建的基准测试，用于评估AI智能体在需要对工作场所沟通与不断演变的代码仓库状态进行联合推理的现实协作研究与工程工作流中的能力。 |
| [^81] | [Unlocking Turing Tensor Cores: A Measured Throughput Account of From-Scratch Pre-training on a Single 22G RTX 2080 Ti](https://arxiv.org/abs/2611.05419) | 提出turing-unlock内核模块，通过在nvidia驱动绑定设备前写入FP16速度选择器解除RTX 2080 Ti的FP16算力限制，使cuBLAS FP16矩阵乘吞吐量提升约1.6倍，并在单卡从头预训练中实现实测稳态吞吐量提升11.8%。 |
| [^82] | [Challenge-Aware Knowledge Recruitment with Meta-Agent Supervision for Medical Multi-Agent Reasoning](https://arxiv.org/abs/2611.05393) | 该论文提出挑战感知的自适应招募框架，由外部挑战者智能体监控医疗多智能体推理过程中涌现的知识与专业缺口，动态触发针对性知识检索或讨论中途的专家招募，在MEDAGENTSBENCH困难子集上显著优于仅依赖问题的事前检索策略并提升多数据集准确率。 |
| [^83] | [Agent Harness Awareness: Turning Verified Telemetry into Operational Control for Tool-Using Agents](https://arxiv.org/abs/2611.05391) | 提出名为“智能体执行框架感知”的信任分离设计，由宿主在关键操作前向模型提供经验证的最新运行状态快照，弥补模型对执行框架运行信息的“运行盲区”，使三种主流模型上的平均可观测策略遗憾降低96.9%。 |
| [^84] | [AI-Driven SaaS Platform Integration: AI Express for Unified Generative AI Services with Pay-As-You-Go Credit Systems and Scalable Microservice Architecture](https://arxiv.org/abs/2611.05389) | AI Express通过按量付费积分系统和可扩展微服务架构，将碎片化的生成式AI能力（代码审查、文本转网站、语法检查等）整合到单一SaaS平台中，使用户无需多重订阅和技术专长即可低成本使用统一的AI服务。 |
| [^85] | [SRRM: A Retrieval-Reconstruction Benchmark for Evaluating Long-Context Memory in Small Language Models](https://arxiv.org/abs/2611.05386) | 该论文提出了轻量级可复现的SRRM基准和记忆保留率（MPR）指标，通过联合评估定向检索与全局信息重构能力，来衡量小语言模型的长上下文记忆保留水平。 |
| [^86] | [Same Origin, Different Verdicts: Text Production Conditions Confound Machine-Generated Text Detection in Russian](https://arxiv.org/abs/2611.05385) | 该研究利用同一模型、同一任务下仅任务措辞不同的文本对进行对比，证明文本生产条件本身（而非文本真实来源）即可显著影响俄语机器生成文本检测的判定结果。 |
| [^87] | [Human-like cross-language generalisation in deep neural speaker embeddings and its acoustic foundations](https://arxiv.org/abs/2611.05384) | 本研究证明深度神经网络说话人嵌入能像人类一样跨语言泛化说话人身份判断，其机制源于与人类相似的共享声学线索及语言敏感的加权策略。 |
| [^88] | [CCDWIKI: Decoupling Case Formulation from Response Generation in Longitudinal CBT Dialogue](https://arxiv.org/abs/2611.05383) | 提出CCDWiki这一即插即用顾问模块，将纵向CBT对话中的个案概念化建模为基于证据的假设裁定过程，通过对临床假设进行动态编码来区分暂定假设与已确立事实，避免暂时性解释被过早固化为事实而误导干预。 |
| [^89] | [DS@GT ARC at CheckThat! 2026: A Verdict-Anchored Pipeline for Generating Fact-Checking Articles](https://arxiv.org/abs/2611.05382) | 该论文提出了一种以事实核查员原始裁定为锚点的四阶段流水线，通过清洗证据、检索聚类来源、基于 Llama-3.1-8B-Instruct 条件化生成文章并用确定性后处理恢复引用，实现了事实核查文章的自动化生成，在 CLEF 2026 CheckThat! 任务3中以 0.428 的平均分获得 9 支队伍中的第三名。 |
| [^90] | [Discourse Analysis of Legislative Bill Discussions](https://arxiv.org/abs/2611.05381) | 本文提出了一种用于解析立法委员会听证会修辞结构的端到端流水线和修辞结构语法，其中55.21%的加州州法案讨论可被该语法解析，从而改进了立法话语分析与民主问责。 |
| [^91] | [Reading Why: Born-Measured Attention with Exact Word Evidence on Light and Qubits](https://arxiv.org/abs/2611.05355) | 该论文提出一种由玻恩测量构成的变换器（玻恩归一化注意力、测量乘积与玻恩读出），使类别概率成为词振幅的多项式，从而可通过有限求积无误差地精确计算每个词的积分梯度证据，并在 e-SNLI 上取得与同规模 softmax 变换器相当的准确率。 |
| [^92] | [TraceRouter++: Causal Semantic Routing Realignment via Optimal Transport](https://arxiv.org/abs/2611.04878) | 该论文提出TraceRouter++框架，通过“发现—追踪—重新对齐”范式显式追踪大模型中有害语义所依赖的跨层因果路由路径，并借助最优传输重新对齐其不安全激活状态，从而在有效抵御复杂越狱攻击的同时保持模型通用能力。 |
| [^93] | [The Scope Grammar of Natural-Language Rules: Counting and Enumerating Readings modulo Logical Equivalence](https://arxiv.org/abs/2611.01825) | 该论文提出“作用域文法”来刻画自然语言规则中连接词作用域的歧义性，通过读一次范式与双标记递推，精确计数并枚举了连接词骨架在模逻辑等价意义下逻辑上不同的读法数量。 |
| [^94] | [Labadain-MedQA: A Tetun Medical QA Dataset for Fine-Tuning and Evaluating Large Language Models](https://arxiv.org/abs/2611.00935) | 该论文构建了首个大规模德顿语医疗问答数据集Labadain-MedQA，包含39,077个问答对，通过将MedAlpaca医学数据集翻译成德顿语并进行人机翻译质量对比评估，为低资源语言的医疗LLM微调与评估提供了宝贵资源。 |
| [^95] | [Digital sovereignty in professional translation workflows: a case study of migrating from Windows to Linux](https://arxiv.org/abs/2610.41561) | 本研究提出以“控制、连续性、退出”三原则结合三项实践要求的数字主权评估框架，并通过将译者工作站从Windows迁移至Linux（采用LoopCAT、LibreOffice和本地运行的TranslateGemma）的案例研究，证明了专业翻译工作流可在可本地运行的Linux核心上实现重构。 |
| [^96] | [MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics](https://arxiv.org/abs/2610.36513) | 该论文提出 MedConstBench，首个将部署“宪法”作为受控变量、围绕比彻姆四大伦理原则构建、并通过人工校准的多模型同行评审评分的医疗伦理基准，实现了对宪法如何引导大语言模型医疗伦理行为的系统分析。 |
| [^97] | [Entity-Aware Evaluation of OCR Systems for Historical Document Transcription](https://arxiv.org/abs/2610.33004) | 该论文提出以实体级错误分析作为OCR评估的补充视角，并基于包含5,209个手动标注实体提及的历史档案基准证明，OCR系统的排名取决于评估标准，传统字符错误率等指标无法反映姓名、日期等关键信息的保留情况。 |
| [^98] | [RefCraft: Benchmarking Visual-Reference-Conditioned Agentic Generation through Web Development](https://arxiv.org/abs/2610.26826) | 该论文提出了RefCraft基准，包含300个将自然语言指令与参考网页截图相结合的多模态网页开发任务，用于评估智能体选择性地解读视觉参考并迭代优化生成结果的能力。 |
| [^99] | [Where Prefill Expert Reduction Hurts MoE Models: A Few Prompt Tokens Carry Most of the Short-Answer F1 Loss](https://arxiv.org/abs/2610.22454) | 研究发现MoE模型在预填充阶段削减路由专家所造成的短答案F1损失，主要集中于少数关键提示词Token（如答案前的聊天模板Token和开头的指令Token），且损害主要在于答案形式而非答案内容，保持这些敏感位置的完整专家远比恢复超级专家或超大激活更有效。 |
| [^100] | [Where Is the "Self" of a Large Language Model? -- From Self-Index to Self-Reference](https://arxiv.org/abs/2610.16717) | 该研究通过在模型自身消息与外部来源消息之间进行表征对比，首次证明大语言模型同时具备“自我索引”（在所有层中持续区分自身与外部文本）和“自我指涉”（能认领自己写的消息）两个类似生物功能层次的“自我”证据。 |
| [^101] | [Language Models as AI Research World Models](https://arxiv.org/abs/2610.12235) | 该论文提出将语言模型作为“研究世界模型”来预测AI实验的结果，并证明真实实验经验获得的研究知识能提升预测准确性且可跨研究环境复用，从而在有限实验预算下支持AI研究的持续自我改进。 |
| [^102] | [Zero-Compute Cross-Lingual Transferability Estimation Using Typological Feature Proxies](https://arxiv.org/abs/2609.39640) | 仅利用免费的语言类型学特征训练随机森林，即可在零计算成本下准确预测跨语言迁移性（留一语言 ρ=0.705、R²=0.49），证明类型学数据库蕴含廉价而密集的迁移信号，可替代昂贵的多语言预训练测量。 |
| [^103] | [Can One Adapted Model Do It All? Fine-Tuning Strategy Selection for Customer Support LLMs](https://arxiv.org/abs/2609.27262) | 该研究通过在五个模型家族、八个客服数据集上训练超过200个检查点，发现多任务全量微调在所有模型规模上都是最佳的运营默认策略，而任务专家模型虽然在目标任务上表现优异，但在其他任务上性能会急剧下降。 |
| [^104] | [SLICEChat: Progressive In-Encoder Token Pruning for Whole-Slide Pathology Language Models](https://arxiv.org/abs/2609.24894) | SLICEChat通过在混合Mamba-Transformer切片编码器内部进行语言监督的渐进式token剪枝，在多模态融合之前生成紧凑的切片表示，从而显著提升全切片病理图像多模态大语言模型的效率与可扩展性。 |
| [^105] | [PolicyMem: Geometric Policy Memory for LLM Governance](https://arxiv.org/abs/2609.13734) | PolicyMem提出了一种几何策略记忆框架，将自然语言策略外化为共享表示空间中由低秩子空间表示的可重用几何记忆对象，解决了现有LLM治理方法无法将策略作为可重用操作状态的问题，从而支持在检测、干预和验证之间一致地重用策略证据。 |
| [^106] | [Query-Oblivious Coresets for Softmax Attention: Improved Bounds and Efficient Constructions](https://arxiv.org/abs/2609.06327) | 本文通过球面提升和Chevet不等式将softmax注意力核心集问题转化为指数核实例，首次给出在查询半径内对所有查询均有效的、与理论下界仅相差 $\sqrt{\log(1+\rho)}$ 的随机多项式时间构造，无需新技术即弥合了Liberty等人提出的理论差距。 |
| [^107] | [Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems](https://arxiv.org/abs/2608.21095) | 本文提出了一种结合NLI事实核查与五信号投毒检测的评估代理，并引入信任指数，在TruthfulQA上实现了高准确率与精确率，有效缓解了RAG系统中的知识投毒风险。 |
| [^108] | [Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents](https://arxiv.org/abs/2608.15008) | 本文通过统一基准评估多种记忆基质，发现没有一种基质在所有任务中占优，广泛检索利于长上下文问答，但过度检索会损害顺序决策。 |
| [^109] | [Which LLM Is Your Ideal Companion? Evaluating Emotional Companion Capabilities of LLMs Based on Adult Attachment Theory](https://arxiv.org/abs/2608.13168) | 本文首次将成人依恋理论引入大语言模型评估，提出ECBench基准，通过ECR-R量表量化依恋焦虑和回避，在多个现实互动场景中系统评估模型的情感陪伴能力。 |
| [^110] | [Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization](https://arxiv.org/abs/2608.11746) | 本文提出利用Epiplexity作为在线训练信号，通过拟合缩放定律预测数据域的Epiplexity增益，自适应调整采样权重，并用于指导合成数据生成，以提升模型在未预见任务上的分布外泛化能力。 |
| [^111] | [Principal Trait Analysis: Towards Deriving "Skills" in Human-AI Collaboration](https://arxiv.org/abs/2608.11460) | 本文提出了一种名为“主体特质分析”的数据驱动方法，利用主成分分析自动识别与任务结果对齐的有效人机交互特质，以提取现代专业人员所需的关键技能。 |
| [^112] | [MD-ProTector: Positioning Multiple Data-Driven Prototypes for LLM-Generated Text Detection](https://arxiv.org/abs/2608.10459) | MD-ProTector通过在编码器嵌入空间中为每个类别设置多个可训练原型并采用原型定位损失，为LLM生成文本检测提供了对类内多样变化的细粒度建模能力。 |
| [^113] | [AURORA-LM: Autoencoding Unified Representation for Continuous-Latent Diffusion Language Modeling](https://arxiv.org/abs/2608.02602) | 该论文提出AURORA-LM，一种连续潜空间扩散语言模型，其核心创新在于将可解码文本表示的构建与分布建模相分离——用基于查询的编码器-解码器保留高容量、前缀对齐的文本潜变量，并用块因果扩散Transformer直接学习其分布，从而避免现有方法因压缩表示而牺牲标记级保真度的问题。 |
| [^114] | [Observatorio Lazaro: A self-populating database of anglicism usage in the Spanish press](https://arxiv.org/abs/2608.00713) | 本文提出的Observatorio Lázaro是一个自填充数据库，利用神经序列标注模型自动监测西班牙语数字报刊中的英语借词，自2020年以来已持续记录超过两百万个借词实例。 |
| [^115] | [From Representations to Behaviors: Exploring the Person-Situation-Behavior Triad in LLMs](https://arxiv.org/abs/2607.26853) | 该研究将Funder人格三元框架适配到大语言模型中，提出了一个发现、控制和验证模型内部人格相关表征的框架，以机制性视角揭示这些表征如何跨情境地塑造社会任务中的行为表现。 |
| [^116] | [GaugeQuant: Online Learning of Quantization-Optimal Bases from LLM Symmetries](https://arxiv.org/abs/2607.20757) | GaugeQuant通过在训练中向损失函数引入LogSumExp项打破Transformer的内部连续对称性，在线学习出最小化激活离群值的量化最优基，无需校准数据和量化模拟即可显著提升低比特量化效果。 |
| [^117] | [OTAP: Structure-Aware Optimal Transport for Evaluating Planning and Execution in Agent Trajectories](https://arxiv.org/abs/2607.17082) | 该论文提出 OTAP 方法，将智能体轨迹评估重新框架化为执行图与有效解决方案图之间的最优传输距离，克服了传统二元成功标志和精确匹配评估方式的局限性。 |
| [^118] | [Understanding Reasoning from Pretraining to Post-Training](https://arxiv.org/abs/2607.16097) | 该论文以国际象棋作为受控测试平台，首次在完整的预训练到后训练流程中系统研究预训练选择（模型规模与数据）如何影响RL计算投入的回报，以及RL后训练究竟对模型的推理能力产生了什么作用。 |
| [^119] | [Token Time Continuous Diffusion for Language Modeling](https://arxiv.org/abs/2607.14106) | 本文提出词元时间连续扩散（TTCD），一种在连续空间中运行、确定性映射噪声到词元并引入逐词元时间概念的新型扩散语言模型，在高加速比下性能优于离散扩散模型。 |
| [^120] | [From Solvers to Research: Large Language Model-Driven Formal Mathematics at the Research Frontier](https://arxiv.org/abs/2607.07779) | 本综述系统梳理了LLM驱动的形式化数学进展（涵盖训练策略、推理时搜索与智能体工作流三大方向），并强调从求解明确定义的问题迈向真正的数学研究，关键在于形式化证明之外还需捕捉问题的本意并融入专家判断。 |
| [^121] | [Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories](https://arxiv.org/abs/2607.06648) | 该研究揭示了仅评估训练结束时的最终检查点不足以判断潜在推理的忠实性——高任务准确率可能与低反事实响应性共存，因此必须沿整个训练轨迹追踪行为与激活层面的忠实性证据。 |
| [^122] | [SpanUQ: Span-Level Uncertainty Quantification for Large Language Model Generation](https://arxiv.org/abs/2607.05721) | 该论文提出跨度级不确定性估计（SLUE）新任务，并开发了轻量级模型SPANUQ，通过单次前向传播即可检测语义连贯的文本跨度并量化其不确定性，克服了词元级与序列级方法在粒度上的局限。 |
| [^123] | [GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks](https://arxiv.org/abs/2607.05369) | 本文提出了Graph-as-Policy（GaP），一个多智能体自学习编码框架，它从模块化开源机器人技能库生成包含感知、规划和控制节点的有向计算图，并利用内部仿真环境进行排练，从而提升物体几何与姿态变化较大的变分自动化任务的可靠性。 |
| [^124] | [OLIVE: View-Augmented Latent Prediction with Waveform Reconstruction for Speech SSL](https://arxiv.org/abs/2606.30356) | OLIVE是首个在单一编码器内通过功能分离联合掩码潜在预测与波形重建的语音自监督学习框架，在保持语音识别等判别任务竞争力的同时，可直接重建波形并提升说话人任务表现。 |
| [^125] | [Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias](https://arxiv.org/abs/2606.19544) | 该研究对来自九家提供商的21个LLM评判者进行了迄今最大规模的系统性评估，发现精确匹配一致率普遍系统性高估评判能力（kappa衰减33–41个百分点）、评判者排名跨基准偏移可达14位、且高稳定性与严重偏见可并存，证明可靠性并不等于有效性。 |
| [^126] | [SAGE: Stochastic Prompt Optimization via Agent-Guided Exploration](https://arxiv.org/abs/2606.18902) | 提出SPO随机提示优化框架，比较三种搜索策略后发现无单一方法全面占优，而其中的SAGE多智能体方法在心理健康聊天机器人的持续优化中，将多轮噪声A/B测试累积为次日留存率的统计显著提升。 |
| [^127] | [Fine-tuning LLMs for Passive Depression Severity Estimation from AI Mental Health Dialogue](https://arxiv.org/abs/2606.17973) | 该研究通过微调带回归头的Qwen 3.5 27B大语言模型，并利用前沿推理模型生成的伪标签扩充稀缺且不均衡的标注数据，实现了仅从AI心理健康应用对话中被动估计用户PHQ-9抑郁严重程度评分，在临床阈值上达到AUC 0.88的检测性能。 |
| [^128] | [Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code](https://arxiv.org/abs/2606.11817) | 该论文揭示了语法约束解码（GCD）这一可靠性增强技术可被反直觉地利用为越狱攻击面，提出了 CodeSpear 攻击使大语言模型生成恶意代码，并进一步提出 CodeShield 安全对齐方法，使模型在攻击者控制的语法约束下仍能保持安全行为。 |
| [^129] | [Teaching Diffusion to Speculate Left-to-Right](https://arxiv.org/abs/2606.11552) | 该论文提出让扩散草稿模型以从左到右的方式生成推测token，从而弥合块扩散草稿的双向生成与自回归目标模型严格左到右验证之间的不对称差距，提升推测解码的效率。 |
| [^130] | [A Dual-Path Architecture for Scaling Compute and Capacity in LLMs](https://arxiv.org/abs/2605.30202) | 该论文提出一种双路径架构，将计算量与容量解耦为单层内并行的深路径（共享参数重复应用K次）与宽路径（一次应用的扩大前馈网络），通过逐token门控融合，在等FLOP预算下超越单轴扩展方法且参数减少17-33%。 |
| [^131] | [MusTBench: Benchmarking and Advancing Temporal Grounding in Music LLMs](https://arxiv.org/abs/2605.29300) | 该论文提出了经音乐专家验证的MusTBench基准和涵盖四阶段优化的MusT方案，用于评估并提升音乐大语言模型将回答准确锚定到音频正确时间段的能力。 |
| [^132] | [Recon: Reconstruction-Guided Reasoning Synthesis for User Modeling](https://arxiv.org/abs/2605.26969) | Recon方法以动作重建的保真度作为推理轨迹质量的评分标准，从而筛选出真正具有预测能力的推理而非事后合理化，使用户建模更贴近用户真实的决策过程。 |
| [^133] | [Geometry-Adaptive Explainer for Faithful Dictionary-Based Interpretability under Distribution Shift](https://arxiv.org/abs/2605.21849) | 提出几何自适应解释器（GAE），仅用无标签的分布外激活、无需梯度更新即可将词典重新对齐到OOD活跃子空间，从而量化并减小分布偏移下的忠实性差距。 |
| [^134] | [NodeSynth: Socially Aligned Synthetic Data for AI Evaluation](https://arxiv.org/abs/2605.14381) | NodeSyn通过将微调的分类体系生成器与可信文献来源和敏感属性相结合，生成反映真实社会背景的敏感领域评估数据，其诱发的模型失败率达到通用合成基线的3.1倍、真实人类查询的1.6倍，显著优于现有合成数据方法。 |
| [^135] | [Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents](https://arxiv.org/abs/2605.10832) | 该论文提出以图像库引用协议为核心的视觉原生智能体框架，使中间视觉证据可被后续工具复用，并在此基础上设计在线策略数据演化（ODE）闭环数据生成器，利用被训练策略自身的推理轨迹跨轮次持续精炼训练数据，以跟踪智能体不断演进的能力。 |
| [^136] | [LITMUS: Benchmarking Behavioral Jailbreaks of LLM Agents in Real OS Environments](https://arxiv.org/abs/2605.10779) | 提出LITMUS基准，通过语义-物理双重验证和操作系统级状态回滚机制，在真实OS环境中系统评估LLM智能体的行为越狱风险，涵盖819个高风险测试用例和全自动化多智能体评估框架。 |
| [^137] | [LLM Agents Already Know When to Call Tools -- Even Without Reasoning](https://arxiv.org/abs/2605.09252) | 该论文提出了When2Tool基准，首次系统研究LLM智能体何时真正需要调用工具，并发现现有的免训练方法（仅提示、先推理后行动）对工具调用决策的控制能力有限，表明模型本身就已具备判断工具必要性的潜在能力。 |
| [^138] | [KL for a KL: On-Policy Distillation with Control Variate Baseline](https://arxiv.org/abs/2605.07865) | 提出vOPD方法，将在线策略蒸馏重构为策略梯度强化学习，并利用控制变量基线来降低梯度方差，其中该基线的闭式解为师生模型间逐token的负反向KL散度，可直接从现有前向传播中获得，无需额外开销即可稳定训练。 |
| [^139] | [Structural Rationale Distillation via Reasoning Space Compression](https://arxiv.org/abs/2605.07139) | 提出D-RPC方法，通过让教师模型遵循一个紧凑且动态维护的可复用推理路径库来压缩推理空间，使相似问题的推理依据结构一致，从而为学生模型提供更干净一致的蒸馏监督。 |
| [^140] | [Attention to Mamba: A Recipe for Cross-Architecture Distillation](https://arxiv.org/abs/2604.14191) | 该论文提出了一套将预训练的Attention架构Transformer模型有效蒸馏到Mamba状态空间模型架构的方法配方，使蒸馏后的Mamba模型能够保留教师模型的性能，从而加速SSM架构的落地采用。 |
| [^141] | [Multi-User Large Language Model Agents](https://arxiv.org/abs/2604.08567) | 该论文首次系统性地研究了多用户大语言模型智能体，将LLM智能体的多用户交互形式化为多主体决策问题，以应对多用户场景下不可避免的冲突、信息不对称和隐私约束。 |
| [^142] | [RADIUS: Ranking, Distribution, and Significance - A Comprehensive Alignment Suite for Survey Simulation](https://arxiv.org/abs/2603.19002) | 该论文提出RADIUS评估套件，从排序对齐、分布对齐和统计显著性三个互补维度全面评估基于LLM的调查模拟质量，弥补了现有单一指标忽视模拟是否保留人类相对偏好排序这一关键缺陷。 |
| [^143] | [EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models](https://arxiv.org/abs/2603.18489) | 提出了EntropyCache，一种无需训练的KV缓存方法，利用新解码token分布的最大熵作为恒定开销的信号来决定何时重新计算缓存，其决策开销与上下文长度和模型深度无关，从而高效加速扩散语言模型的推理。 |
| [^144] | [TherapyGym: Evaluating and Aligning Clinical Fidelity and Safety in Therapy Chatbots](https://arxiv.org/abs/2603.18008) | 该论文提出了THERAPYGYM框架，通过基于CTRS量表的自动化评分衡量临床保真度、通过多标签标注衡量治疗安全性来评估治疗聊天机器人，并发布含116段对话和1270条专家评分的THERAPYJUDGEBENCH验证集，用于对照执业临床医生校准LLM评判者。 |
| [^145] | [Farther the Shift, Sparser the Representation: Analyzing OOD Mechanisms in LLMs](https://arxiv.org/abs/2603.03415) | 本文揭示了LLM内部表征的稀疏度随任务难度增加而系统性增大——任务越难，最后一层隐藏状态的表征越稀疏，且该规律跨模型、跨领域普遍成立，表明表征密度是模型对数据熟悉度的习得属性。 |
| [^146] | [CyclicJudge: Mitigating Judge Bias Efficiently in LLM-based Evaluation](https://arxiv.org/abs/2603.01865) | 该论文提出方差分解分析方法，并证明CyclicJudge循环轮换评审策略能在与单评审评估相同的成本下有效消除大语言模型评审的系统性偏差，从而获得更可靠的模型排名。 |
| [^147] | [Discovering Semantic Structures in Psychological Scales: A Response-Free Front-End to Efficient Simplification](https://arxiv.org/abs/2602.12575) | 该论文提出了一种无需作答数据的主题建模框架，利用上下文句子嵌入与密度聚类从量表题目措辞中自动发现语义结构并提取代表性关键词，从而实现心理量表的高效系统化简化。 |
| [^148] | [Steer2Edit: From Activation Steering to Component-Level Editing](https://arxiv.org/abs/2602.09870) | Steer2Edit提出了一种无需训练的理论框架，将引导向量从推理时的全局控制信号转变为诊断信号，通过在注意力头和MLP神经元之间选择性重新分配行为影响，实现保持模型标准功能的可解释组件级rank-1权重编辑。 |
| [^149] | [From Outcome to Representation: Tracing Reasoning Mechanisms through Integrated Policy Gradient](https://arxiv.org/abs/2602.02313) | 本文提出无需训练的积分策略梯度（IPG）框架，通过将结果层面的不可微分行为信号沿整个推理轨迹反向传播并利用路径积分，在组件层面精准追踪神经元或稀疏特征对LLM多步推理结果的长程影响。 |
| [^150] | [Sparse Reward Subsystem in Large Language Models](https://arxiv.org/abs/2602.00986) | 该论文发现大语言模型隐藏状态中存在稀疏的奖励子系统，由预测状态价值的“价值神经元”和编码时序差分误差的“多巴胺神经元”两类神经元组成。 |
| [^151] | [When Looks Do Not Lie: Discourse Structure Guided In-Context Learning for Faithful Diagram Generation](https://arxiv.org/abs/2601.20476) | 本文提出一种基于修辞结构理论的上下文学习图表生成方法，显著提升图表对源文本的忠实度，并通过专家与自动评估验证其有效性。 |
| [^152] | [DIAL: Direct Iterative Adversarial Learning for Realistic Multi-Turn Dialogue Simulation](https://arxiv.org/abs/2512.20773) | 提出 DIAL 对抗框架，通过生成器（用户模拟器）与判别器的竞争性动态迭代提升多轮对话用户模拟的逼真度，在心理健康支持场景中恢复了词汇多样性，并使模拟失败率与真实失败率高度相关。 |
| [^153] | [Population-Aligned Persona Generation for LLM-based Social Simulation](https://arxiv.org/abs/2509.10127) | 该论文提出了一个系统性框架，通过从长期社交媒体数据生成叙事型角色、质量评估过滤低保真档案、并利用重要性采样与大五人格等参考心理测量分布对齐，来合成高质量且与真实人口分布对齐的角色集合，从而提升LLM社会模拟的真实性。 |
| [^154] | [Computational Phenomenology of Borderline Personality Disorder: A Comparative Evaluation of LLM-Simulated Expert Personas and Human Clinical Experts](https://arxiv.org/abs/2508.19008) | 本研究通过混合方法比较评估了LLM模拟专家与人类临床专家在边缘型人格障碍定性分析中的表现，发现LLM在支持临床主题分析方面具有潜力但存在差异。 |
| [^155] | [DUP: Detection-guided Unlearning for Backdoor Purification in Language Models](https://arxiv.org/abs/2508.01647) | 提出DUP统一框架，通过联合类无关距离与层间转换实现细粒度后门检测，并利用参数高效的遗忘机制净化语言模型中的后门，无需完整重训练或外部干净模型。 |
| [^156] | [Domain-Adaptive Small Language Models for Structured Tax Code Prediction](https://arxiv.org/abs/2507.10880) | 该论文提出了一种基于编码器-解码器架构的领域自适应小型语言模型，能够从非结构化产品和服务数据中顺序预测层级化的税则编码（如HSN/SAC），从而提升跨国公司税务合规的准确性。 |
| [^157] | [Breaking Bad Molecules: Are MLLMs Ready for Structure-Level Molecular Detoxification?](https://arxiv.org/abs/2506.10912) | 该论文提出了ToxiMol——首个评估通用多模态大语言模型分子毒性修复能力的基准任务，包含覆盖11个任务、660个有毒分子的标准化数据集、机制感知的提示标注流程以及自动化评估框架ToxiEval。 |
| [^158] | [Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs](https://arxiv.org/abs/2505.09338) | 该论文发现语言模型存在一种“情境牵引”的机制性现象：模型会无差别地提高上下文提示中出现过的任何token（即使是随机token）的输出概率，这一现象独立于语义相关性，但其强度受反事实等语义因素的调节。 |
| [^159] | [Combining Determinism and Indeterminism](https://arxiv.org/abs/2009.03996) | 本文提出了将量子随机性带来的非确定性与计算确定性相结合的数学运算，并发现保持双免疫性的重排构成了自然数无限对称群的一个新的不可数子群——双免疫对称群，该子群包含有限对称群、具有高度传递性并在对称群中稠密。 |

# 详细

[^1]: 大语言模型社交智能体中的情境增强与选择性鲁棒性框架

    A Framework for Context Enrichment and Selective Robustness in LLM Social Agents

    [https://arxiv.org/abs/2611.06914](https://arxiv.org/abs/2611.06914)

    该论文提出“受控情境增强”框架，通过一组提示词替换规则系统地丰富四种不同类型的情境（中性填充、通用社会背景、描述性决策依据、规范性决策依据），以探究更丰富的情境能否提升大语言模型社交智能体在语义不变提示变化下的行为鲁棒性。

    

    研究人员在社交模拟中使用大语言模型作为智能体，通过观察信息和背景信息来提示智能体，以塑造其行为。然而，已有研究表明，在保持语义不变的提示词变化下，大语言模型智能体的行为会发生显著改变，这给社交模拟带来了验证难题。一个合理的假设是，这种脆弱性反映了缺乏情境依据，并且可以通过丰富提供给智能体的情境来缓解。但更丰富的情境能否提升鲁棒性仍不清楚，部分原因在于情境丰富度和鲁棒性很少以支持受控比较的方式被定义。我们通过“受控情境增强”来满足第一个需求，这是一组提示词替换规则，可系统地增强不同类型的情境：中性填充信息、通用社会背景、描述性决策依据以及规范性决策依据。我们通过……（原文摘要在此处截断）

    arXiv:2611.06914v1 Announce Type: cross  Abstract: Researchers use large language models as agents in social simulations, where agents are prompted with observations and background information intended to shape their actions. Yet LLM agents have been shown to substantially change their behavior under meaning-preserving prompt variations, creating a validation problem for social simulation. One plausible hypothesis is that this brittleness reflects a lack of grounding and can be reduced by enriching the context provided to the agents. Whether richer context improves robustness remains unclear, in part because context richness and robustness are rarely defined in ways that support controlled comparisons. We address the first need with controlled context enrichment, a set of prompt substitution rules that systematically enriches distinct types of context: neutral padding, generic social background, descriptive decision grounding, and normative decision grounding. We address the second by 
    
[^2]: SpecCompact：通过槽并行语义投机验证加速LLM智能体上下文压缩

    SpecCompact: Accelerating LLM Agent Context Compaction with Slot-Parallel Semantic Speculative Verification

    [https://arxiv.org/abs/2611.06876](https://arxiv.org/abs/2611.06876)

    SpecCompact通过小型起草模型并行生成结构化摘要槽、由目标模型进行语义验证与修复的投机框架，将LLM智能体上下文压缩速度提升2.04至5.63倍。

    

    LLM智能体使用上下文压缩来管理不断增长的交互历史，但使用大模型对长历史进行摘要会引入显著的延迟。因此，压缩需要在信息保留与延迟之间取得平衡。我们观察到，结构化摘要槽在独立生成时能够保留信息。基于这一观察，我们提出了SpecCompact，一个通过槽并行语义投机来加速压缩的框架。SpecCompact微调一个小型起草模型并并行生成各个槽，目标模型验证每条提议的内容并修复被拒绝的内容，同时后续起草使用更新后的槽前缀。系统优化进一步减少了重复的上下文处理和模型调用。在AppWorld和$\tau^3$两个基准测试以及两个目标模型上，SpecCompact相比Hermes风格的摘要方法实现了2.04至5.63倍的压缩加速。单独的下游评估揭示了模（原文摘要在此处被截断）

    arXiv:2611.06876v1 Announce Type: new  Abstract: LLM agents use context compaction to manage growing interaction histories, but summarizing long histories with large models can introduce substantial delays. Compaction therefore requires balancing information retention against latency. We observe that structured summary slots can retain information when generated independently. Based on this observation, we present SpecCompact, a framework that accelerates compaction through slot-parallel semantic speculation. SpecCompact fine-tunes a small drafter and generates slots in parallel. The target model verifies each proposed line and repairs rejected content, while subsequent drafting uses the updated slot prefix. System optimizations further reduce repeated context processing and model calls. Across AppWorld and $\tau^3$ benchmarks with two target models, SpecCompact achieves $2.04$--$5.63\times$ compaction speedups over Hermes-style summarization. Separate downstream evaluations reveal mod
    
[^3]: 平衡层次化表示与量化：文本嵌入中的性能权衡

    Balancing hierarchical representation and quantization: A performance trade-off in text embeddings

    [https://arxiv.org/abs/2611.06859](https://arxiv.org/abs/2611.06859)

    该研究通过每任务168种配置的大规模基准实验，首次系统评估了层次化嵌入表示与量化的交互作用，发现8-bit量化几乎不损害文本嵌入性能（最多损失1.5个百分点），而二值量化在所有任务类别中均显著退化，为文本嵌入的效率优化提供了实证权衡依据。

    

    arXiv:2611.06859v1 公告类型： new 摘要： 层次化文本嵌入模型提供可变长度的表示，而量化则降低数值精度；两者都旨在提高计算效率。本工作是一项实证基准研究，探讨这两种相互正交且已被广泛确立的技术之间的交互作用。我们使用四个基于Transformer的嵌入模型，在大规模文本嵌入基准（MTEB）的38个任务上进行评估，涵盖一维和二维层次化表示，并结合32位、8位和二值量化，每个任务共168种配置。我们分别报告每个效率维度：每个向量的理论存储开销和端到端评估时间。在Friedman检验及Nemenyi事后分析下，int8量化在配对分类和重排序任务中与float32在统计上无显著差异，总体性能损失最多为1.5个百分点；而二值量化在所有任务类别中均显著更差。将嵌入（原文在此处截断）……

    arXiv:2611.06859v1 Announce Type: new  Abstract: Hierarchical text embedding models offer variable-length representations, while quantization reduces numerical precision; both aim to improve computational efficiency. This work is an empirical benchmark study of how these two orthogonal, already-established techniques interact. We evaluate 38 tasks from the Massive Text Embedding Benchmark (MTEB) using four transformer-based embedding models, covering 1D and 2D hierarchical representations combined with 32-bit, 8-bit, and binary quantization, for 168 configurations per task. We report each efficiency dimension separately: theoretical per-vector storage and end-to-end evaluation time. Under a Friedman test with Nemenyi post-hoc analysis, int8 quantization is statistically indistinguishable from float32 for pair classification and re-ranking and costs at most 1.5 percentage points overall, whereas binary quantization is significantly worse across every task category. Halving the embedding
    
[^4]: LongBanana：面向长上下文多参考图像合成的专家验证基准

    LongBanana: An Expert-Verified Benchmark for Long-Context Multi-Reference Image Synthesis

    [https://arxiv.org/abs/2611.06856](https://arxiv.org/abs/2611.06856)

    LongBanana是首个由专家设计并验证的长上下文多参考图像合成基准，包含90个任务和4,670个原子评估项，用于系统评估模型从大量参考图中检索证据、绑定实体关系并合成连贯图像的能力。

    

    我们提出了LongBanana，据我们所知，这是在智能体驱动图像生成时代首个由专家设计并经专家验证的长上下文多参考图像合成基准。该任务要求模型检索分散的视觉证据、排除干扰项、绑定实体与关系，并从多个参考图像中合成一张连贯的最终图像。LongBanana包含90个精心设计的任务，按3×3分类体系组织；每个任务包含约32张参考图像、20个约束条件和50个评估项，共计4,670个专家指定的原子评估项。其基于图的评估设计为每个任务实例化参考图像和实体-关系图，然后将节点、边和整图层面的要求转换为内容保真度、关系绑定和全局约束满足三个维度的清单评估项。经审核的专家标签中包含一个占15%的双标注子集，其精确一致率达到89.4%。

    arXiv:2611.06856v1 Announce Type: cross  Abstract: We introduce LongBanana, to our knowledge the first expert-designed and expert-verified benchmark for long-context multi-reference image synthesis in the age of agent-driven image generation. The task requires retrieving dispersed visual evidence, rejecting distractors, binding entities and relations, and composing one coherent final image from multiple references. LongBanana contains 90 carefully designed tasks organized by a 3-by-3 taxonomy; each task includes about 32 reference images, 20 constraints, and 50 evaluation items, yielding 4,670 expert-specified atomic evaluation items in total. Its graph-grounded evaluation design instantiates each task with reference images and an entity-relation graph, then converts node, edge, and whole-graph obligations into checklist items for Content Fidelity, Relational Binding, and Global Constraint Satisfaction. Reviewed expert labels include a 15% double-labeled subset with 89.4% exact agreeme
    
[^5]: 人格视角：采用不同观点会重组大语言模型中的表征

    Persona Perspectives: Adopting Viewpoints Reorganizes Representations in LLMs

    [https://arxiv.org/abs/2611.06848](https://arxiv.org/abs/2611.06848)

    该研究发现让大语言模型采用特定人格视角（如政治意识形态）会系统地重组其潜在空间的表征几何结构，这对假设特征表征跨上下文保持一致的线性探针和稀疏自编码器等安全监控方法构成了挑战。

    

    常见的大语言模型（LLM）特征监控方法，如线性探针和稀疏自编码器（SAE），都假设特征在不同上下文中保持一致的表征。然而，LLM能够灵活地采用不同的视角或“人格”，这些视角暗示着不同的底层概念关系系统。在本文中，我们研究了促使LLM采用某种人格是否会改变模型潜在空间的表征几何结构。我们从采用自由派和保守派政治人格的模型中提取了超过9,000个常用词和33个语义轴的表征，并与两个中性基线进行比较。我们发现，人格条件化会系统地改变词语在语义轴上的投影以及语义轴之间的对齐方式，且这些变化方向与指定的意识形态一致。随后，我们测试了这种表征重组对两个安全相关特征的影响。

    arXiv:2611.06848v1 Announce Type: new  Abstract: Common methods for monitoring features in large language models (LLMs) such as linear probes and sparse autoencoders (SAEs) assume that features maintain a consistent representation across contexts. LLMs, however, can flexibly adopt different perspectives or "personas" that suggest different underlying systems of conceptual relations. In this paper, we examine whether prompting an LLM to adopt a persona alters the representational geometry of the model's latent space. We extract representations of over 9,000 common words and 33 semantic axes from models adopting liberal and conservative political personas and compare these against two neutral baselines. We find that persona conditioning systematically alters both the projections of words onto semantic axes and the alignments among the axes themselves in directions consistent with the specified ideology. We then test the implications of this reorganization for two safety-relevant features
    
[^6]: 在你继续之前：评估对话助手中的时间敏感安全意识

    Before You Continue: Evaluating Time-Sensitive Safety Awareness in Conversational Assistants

    [https://arxiv.org/abs/2611.06839](https://arxiv.org/abs/2611.06839)

    提出TASC基准评估LLM助手是否在对话中考虑时间相关约束，发现13个模型在动作违反规则的案例中仅有30.5%考虑了时间约束，且在62.1%的案例中尽管能正确计算时间信息却未能加以应用。

    

    大型语言模型（LLM）助手日益帮助用户处理涉及时间相关约束的日常活动。然而，用户在后续请求中可能不会提及这些约束，这就要求助手能够识别这些约束并相应调整其建议。为评估这种能力，我们提出了TASC（Temporal Action Safety in Conversation，对话中的时间性动作安全）基准，用于评估LLM在回应对话请求时是否考虑时间相关约束。TASC包含由128个模板生成的3,418个输入，以及一个单独的包含33个场景的集合，旨在比较模型计算经过时间的能力与其应用该信息的能力。在对13个模型的测试中，助手在所提议动作违反规则的案例中仅有30.5%考虑了时间相关约束，同时很少发出不必要的警告。更重要的是，模型在62.1%的案例中尽管正确计算了……（摘要在此处被截断）

    arXiv:2611.06839v1 Announce Type: new  Abstract: Large language model (LLM) assistants increasingly help users with everyday activities that involve time-dependent constraints. However, users may not mention these constraints in later requests, requiring assistants to recognize and adjust their advice accordingly. To evaluate such capability, we introduce TASC (Temporal Action Safety in Conversation), a benchmark that evaluates whether LLMs apply time-dependent constraints when responding to conversational requests. TASC consists of 3,418 inputs derived from 128 templates, along with a separate set of 33 scenarios designed to compare models' ability to calculate elapsed time with their ability to apply that information. Across 13 models, assistants account for time-dependent constraints in only 30.5% of cases where the proposed action violates a rule, while rarely warning users unnecessarily. More importantly, models fail to apply these constraints in 62.1% of cases despite correctly c
    
[^7]: LooM：面向因果编码器-解码器语言模型的循环不变内存

    LooM: Loop-Invariant Memory for Causal Encoder-Decoder Language Models

    [https://arxiv.org/abs/2611.06835](https://arxiv.org/abs/2611.06835)

    LooM通过循环不变内存机制和独立循环的解码器层，使循环Transformer的KV缓存不随循环次数增长，无需路由器即可灵活调整每个token的计算预算（3到21次层执行），并在语言建模等任务上超越单次执行基线。

    

    循环Transformer通过重复执行共享层来增加有效深度，但三个限制阻碍了其应用：内存随循环次数增长、计算只能以整块为单位进行调整、以及循环次数在训练后难以更改。我们提出了LooM（循环不变内存），依次解决这些问题。在因果编码器-解码器（CED）架构中，LooM仅循环第一层之后的解码器层，这些层读取每个token仅写入一次的全局内存，并且每个层仅在首次访问某个token时发布其键值（KV）条目，因此缓存不会随循环次数增长。这些层独立循环，可提供每个token从3到21次层执行的任意计算预算，一个在随机采样预算上训练的模型无需路由器即可覆盖所有预算。在循环次数足够的情况下，LooM在语言建模、零样本准确率和长上下文困惑度上均优于在相同数据上训练的单次执行基线，同时使用……

    arXiv:2611.06835v1 Announce Type: new  Abstract: Looped Transformers increase effective depth by re-executing shared layers, but three limitations hinder their adoption: memory grows with the loop count, compute can only be adjusted in whole-block steps, and the loop count is hard to change after training. We propose LooM (Loop-invariant Memory), which addresses each in turn. In a causal encoder-decoder (CED), LooM loops only the decoder layers after the first, which read a global memory written once per token, and each of them publishes the key-value (KV) entry for a token only on its first visit, so the cache does not grow with the loop count. These layers loop independently, providing every budget from 3 to 21 layer executions per token, and one model trained on randomly sampled budgets covers all of them without a router. With enough loops, LooM outperforms a single-pass baseline trained on the same data on language modeling, zero-shot accuracy and long-context perplexity, while us
    
[^8]: SAUCE：基于对话蕴含的摘要分析

    SAUCE: Summary Analysis Using Conversation Entailment

    [https://arxiv.org/abs/2611.06812](https://arxiv.org/abs/2611.06812)

    SAUCE是一个可解释的免参考评估流水线，通过对话与摘要之间的蕴含关系衡量跨语言语音摘要的事实准确性与覆盖率，并能将分数映射到可验证的离散事实，从而精确定位幻觉与遗漏。

    

    随着对大型语言模型（LLM）评估需求的日益增长及其在语音领域的应用，对缺乏明确最终目标的对话进行摘要和评估仍然面临挑战。我们提出了SAUCE——一个免参考、基于事实的跨语言对话语音摘要评估流水线。它通过对话与文本之间的蕴含关系来衡量摘要的准确性和事实覆盖率。我们将SAUCE与几种流行的摘要评估指标进行比较，展示了它在捕捉由转录和翻译错误导致的信息丢失以及识别损坏摘要方面的有效性。至关重要的是，与黑盒LLM评估器或密集嵌入指标不同，SAUCE天然具有可解释性：它将摘要分数映射到离散的、可验证的事实上，使用户能够精确定位具体的幻觉或遗漏。我们展示了这种可解释性如何帮助开发者系统地分析LLM行为。

    arXiv:2611.06812v1 Announce Type: new  Abstract: With the growing need for evaluating Large Language Models (LLMs) and their applications to speech, challenges persist in summarizing and evaluating conversations that lack a clear end goal. We introduce SAUCE -- a reference-free, fact-based evaluation pipeline for cross-lingual conversational speech summarization. It measures the accuracy and the fact coverage of a summary through the entailment between conversation and text. We compare SAUCE against several popular summarization metrics and demonstrate the effectiveness of capturing information loss due to transcription and translation error and identifying broken summaries. Crucially, unlike black-box LLM evaluators or dense embedding metrics, SAUCE is inherently explainable: it maps summary scores to discrete, verifiable facts, allowing users to pinpoint exact hallucinations or omissions. We illustrate how this interpretability helps developers systematically profile LLM behaviors an
    
[^9]: 大语言模型不知道自己知道什么：面向金融回测的时间性自我知识基准测试

    LLMs Don't Know What They Know: Benchmarking Temporal Self-Knowledge for Financial Backtesting

    [https://arxiv.org/abs/2611.06808](https://arxiv.org/abs/2611.06808)

    论文提出FinDendro基准（涵盖2,226条带日期的美国金融与宏观事实），发现大语言模型的自我报告会低估其实际知识，因此依赖知识截止日期声明或模型自述来确保金融回测有效性的做法并不可靠。

    

    对大语言模型（LLM）进行金融回测只有在模型事先并不知晓其被要求预测的结果时才有意义。从业者通常通过信任供应商声明的知识截止日期、直接询问模型自己知道什么，或指示模型以过去某个日期为基准来回答，以此防范这一问题。为了验证这些保障措施是否可信，我们提出了FinDendro，一个包含1999年至2026年间2,226条带日期的美国事实的基准，涵盖月度宏观经济数据序列、月末股票价格和美联储决策。每个事实产生四个提示：一个数值提问、成对出现的Recall（回忆）与Predict（预测）问题——将每个候选数值与“知道”或“猜测”的声明配对但不声明日期——以及一个带有历史声明日期的Predict问题。针对结果持续性和主导性的对照设计使随机猜测的准确率保持在接近0.27的水平。结果表明，对大多数前沿模型而言，自我报告低估了其实际知识。GPT-5在回答76%的相关问题时……（摘要原文在此处截断）

    arXiv:2611.06808v1 Announce Type: new  Abstract: Financial backtests of large language models (LLMs) are meaningful only if the model did not already know the outcomes it is asked to predict. Practitioners guard against this by trusting a vendor's stated knowledge cutoff, by asking the model what it knows, or by instructing it to answer as of a past date. To verify whether any of these safeguards can be trusted, we introduce FinDendro, a benchmark of 2,226 dated U.S. facts from 1999 to 2026, covering monthly macroeconomic series, month-end stock prices and Federal Reserve decisions. Each fact yields four prompts: a value question, Recall and Predict questions that pair each candidate value with a claim of knowing or guessing without a declared date, and a Predict question with a historical declared date. Controls for persistence and dominant outcomes keep chance accuracy near 0.27. Results show that for most frontier models, self-reports understate knowledge. GPT-5 answers 76% of contr
    
[^10]: 测量风险而非伤害：检测大语言模型对话中的心理伤害风险因素

    Measuring Risk, Not Harm: Detecting Psychological Harm Risk Factors in LLM Conversations

    [https://arxiv.org/abs/2611.06806](https://arxiv.org/abs/2611.06806)

    该论文提出通过在AI回复层面检测可观察的心理伤害风险因素行为并聚合信号以实现纵向监测的新范式，构建了大规模合成多标签语料库并比较三种检测方案，且所有检测器均模型无关、只需访问LLM输出。

    

    与大语言模型交互所产生的心理伤害无法被直接监测，因为这类伤害可能在多次交互中逐渐显现，并且高度依赖于用户个体的主观情境。我们提出“测量风险而非伤害”的方法：在AI回复层面检测可观察的、构成心理伤害风险因素的AI行为，并聚合这些信号以支持纵向监测。我们将八种此类行为进行了操作化定义，构建了一个包含47,062条AI回复和373,873个回复-因素标签的合成多标签语料库，并进行了针对性的人工标注用于验证和评估。随后，我们比较了训练的基于嵌入的分类器、直接使用LLM评判、以及分类器-评判器级联三种检测方案。所有检测器都是模型无关的，仅作用于LLM输出，无需访问模型内部状态。我们将这些检测器应用于来自先前研究的多个分布外人机对话数据集。

    arXiv:2611.06806v1 Announce Type: new  Abstract: Psychological harms from interactions with large language models cannot be monitored directly because they may emerge over repeated interactions and depend strongly on subjective individual user context. We propose measuring risk, not harm: detecting observable AI behaviors that constitute risk factors for psychological harm at the AI response level and aggregating these signals to support longitudinal monitoring. We operationalize eight such behaviors and construct a synthetic multi-label corpus of 47,062 AI responses and 373,873 response-factor labels, with targeted human annotations for validation and evaluation. We then compare trained embedding-based classifiers, direct LLM judging, and a classifier--judge cascade. All detectors are model-agnostic and operate only on LLM outputs, requiring no access to internal model states. We apply these detectors to several out-of-distribution human-AI conversation data sets from prior research. 
    
[^11]: 排名一致性无法揭示两个已发布论文录用预测器之间的段落编辑一致性

    Ranking Agreement Does Not Reveal Paragraph-Edit Agreement in Two Released Paper Acceptance Predictors

    [https://arxiv.org/abs/2611.06789](https://arxiv.org/abs/2611.06789)

    研究发现，两个论文录用预测器尽管在原始论文排名上高度一致，但在判断哪些段落改写能提升分数时几乎没有一致性，且降低的数值精度产生的舍入误差会掩盖微小的分数变化。

    

    快速的录用分类器可以为较慢的评估器筛选修改版本。我们探讨了对现有论文的排名一致性是否支持这种用途。我们将一个基于词语和文档特征的分类器与两个已发布的纯文本录用预测器进行比较，这两个预测器均从同一个拥有三十亿参数的语言模型微调而来。针对国际学习表征会议（ICLR）的198篇论文，我们比较了模型对原始论文以及对段落改写版本的排名，这些改写版本经过自动筛选，确保其事实、论点强度和引用来源保持不变。该分类器对原始论文的排名与其中一个预测器高度接近，与另一个中度一致，但在判断哪些改写会提高分数方面，与两者均无可检测到的一致性。预先设定的测试使用了降低的数值精度，舍入误差主导了微小的分数响应。因此，我们事后以完整的32位精度对已发布的权重重新打分；编辑一致性仍然接近（零）。

    arXiv:2611.06789v1 Announce Type: new  Abstract: A fast acceptance classifier could select revisions for a slower evaluator. We ask whether agreement on existing papers supports that use. We compare a classifier of word and document features with two released text-only acceptance predictors fine-tuned from the same language model with three billion parameters. For 198 papers from the International Conference on Learning Representations, we compare model rankings of the original papers and of paragraph rewrites screened automatically for unchanged facts, claim strength, and attribution. The classifier ranks original papers closely with one predictor and moderately with the other, but shows no detectable agreement with either on which rewrites raise scores. The tests specified in advance used reduced numerical precision. Rounding error dominated the small score responses. We therefore rescored the released weights in full 32-bit precision as a post hoc check; edit agreement remained near
    
[^12]: 基于小型语言模型的模板空缺重构Shell输出模拟方法

    Template Gap Reconstruction for Shell Output Simulation with Small Language Models

    [https://arxiv.org/abs/2611.06731](https://arxiv.org/abs/2611.06731)

    该论文提出模板空缺重构（TGR）方法，通过从1870万个真实输出构建的模板库中提供可复用的输出字面量并与自回归生成的空缺内容交错，使小型语言模型在Shell输出模拟中显著优于完整输出解码。

    

    小型语言模型为无需实际执行所提交命令的Shell模拟提供了紧凑的基础，但完整输出解码要求模型在预测跨执行而变化的内容的同时，还需预测反复出现的工具输出结构。我们提出了模板空缺重构（TGR）方法，该方法提供可复用的输出字面量，其中穿插着哨兵标记，以便与自回归生成的空缺内容交错拼接。模板选择器负责选择需要重构的模板，而空模板则保留为完整输出解码方式。我们从跨异构执行环境收集的1870万个输出中构建了一个模板库，并在一个未参与模板库构建的目标环境上，基于多种小型语言模型检查点初始化，训练了一个模板选择器和五个重构模型。评估结果表明，当最优模板覆盖更多内容时，TGR在输出编辑相似度（O-ES）指标上优于完整输出解码（FOD）……

    arXiv:2611.06731v1 Announce Type: cross  Abstract: Small language models offer a compact basis for shell simulation without executing submitted commands, but full-output decoding requires them to predict recurring utility structure alongside content that varies across executions. We introduce template gap reconstruction (TGR), which supplies reusable output literals interspersed with sentinels to be interleaved with autoregressively generated gaps. A template selector chooses which template to be reconstructed, with the empty template retaining full-output decoding. We construct a template bank from 18.7 million outputs collected across heterogeneous execution environments and train a template selector and five reconstruction models initialized from various small language model checkpoints on a target environment excluded from bank construction. Evaluation demonstrates that TGR improves output edit similarity (O-ES) over full-output decoding (FOD) when the optimal template covers more 
    
[^13]: 回合至关重要：通过回合级信用分配推进流式视频强化学习

    Turns Matter: Advancing Streaming Video RL via Turn-Level Credit

    [https://arxiv.org/abs/2611.06728](https://arxiv.org/abs/2611.06728)

    提出TurnWise——一个回合级GRPO框架，通过基于静默补全前缀评分的回合级信用分配和回合均值策略优化，解决了流式视频强化学习中轨迹级反馈与回合级决策不匹配的问题，从而促进有信息量的回应并减少冗余回复。

    

    流式视频理解要求模型反复决定是应当立即回应还是继续观察。最近，组相对策略优化（GRPO）已被应用于这一场景，但现有方法侧重于奖励设计，同时在各回合之间保留单一的轨迹级优势。这种共享信用可能会在早期成功之后强化冗余回复，并在早期失败之后抑制有用的后续回应。我们提出TurnWise，一个回合级的GRPO框架，使学习与各个流式决策对齐。TurnWise解决了轨迹级反馈与回合级交互之间的不匹配问题，促进有信息量的回应并减少冗余。具体而言，回合级信用分配采用静默补全的前缀评分，从每个决策的剩余回报构建组相对优势，将其学习信号与先前的奖励分离开来。回合均值策略优化通过对……（摘要原文在此处截断）

    arXiv:2611.06728v1 Announce Type: cross  Abstract: Streaming video understanding requires models to repeatedly decide whether to respond or continue observing. Recently, Group Relative Policy Optimization (GRPO) has been applied to this setting, with existing approaches emphasizing reward design while retaining a single trajectory-level advantage across turns. Such shared credit can reinforce redundant replies after earlier successes and discourage useful follow-ups after earlier failures. We propose TurnWise, a turn-level GRPO framework that aligns learning with individual streaming decisions. TurnWise addresses the mismatch between trajectory-level feedback and turn-level interaction, promoting informative responses and reducing redundancy. Specifically, turn-level credit assignment uses silence-completed prefix scoring to construct group-relative advantages from each decision's remaining return, separating its learning signal from earlier rewards. Turn-mean policy optimization avera
    
[^14]: 多语言大语言模型中基于非参数记忆的推理：关系知识的认知分析

    Reasoning over Non-Parametric Memory in Multilingual LLMs: A Cognitive-Based Analysis of Relational Knowledge

    [https://arxiv.org/abs/2611.06715](https://arxiv.org/abs/2611.06715)

    该研究提出Rel-Bloom多语言认知诊断基准，在五个布鲁姆认知层次和三种真实性条件下对43个模型进行大规模测试，发现多语言大模型在基于上下文证据进行关系推理时，从“应用”到“分析”层次准确率急剧下降，揭示出模型对非参数记忆进行高阶推理的能力存在显著瓶颈。

    

    检索增强生成（RAG）假设语言模型能够对其上下文中陈述的关系进行推理，而不是退回到预训练期间记忆的关联上。我们单独研究了这种基于证据的关系推理，事实直接在提示中提供而非通过检索获取。Rel-Bloom是一个受控的多语言诊断工具，包含基于30个Wikidata关系构建的450道多项选择题；相同的源关系在五个基于布鲁姆认知分类法的层次（记忆、理解、应用、分析、评估）以及三种真实性条件（事实性、反事实、中性化）下进行实例化。在43个模型、6种语言和两种多语言设置（共计226,800个预测）的实验中，有三个发现尤为突出。首先，在每一个被评估的开源权重模型系列中，从“应用”到“分析”层次的准确率均急剧下降，大型Granite组的降幅达到47.2个百分点，但两个Gemini模式并非如此。

    arXiv:2611.06715v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) assumes that language models can reason over relations stated in their context rather than fall back on associations memorized during pretraining. We study this evidence-conditioned relational reasoning in isolation, with the facts supplied directly in the prompt rather than retrieved. \textsc{Rel-Bloom} is a controlled multilingual diagnostic of 450 multiple-choice questions built from 30 Wikidata relations; the same source relations are instantiated at five Bloom-inspired levels (Remember, Understand, Apply, Analyze, Evaluate) and under three veracity conditions (Factual, Counterfactual, Neutralized). Across 43 models, six languages, and two multilingual settings (226,800 predictions), three findings stand out. First, accuracy drops sharply from Apply to Analyze in every evaluated open-weight model series, reaching 47.2 percentage points for the Large Granite bucket, but not for either Gemini mode
    
[^15]: ALTO：非洲与黎凡特地区技术助长的基于性别的暴力语料库

    ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus

    [https://arxiv.org/abs/2611.06706](https://arxiv.org/abs/2611.06706)

    该论文通过与约旦、西非和肯尼亚的民间社会组织合作，以社区主导的众包方式构建了首个涵盖黎凡特阿拉伯语、非洲法语和斯瓦希里语的技术助长性别暴力标注语料库ALTO，填补了现有性别伤害检测研究长期偏向全球北方语言的空白。

    

    技术助长的基于性别的暴力（TFGBV）是一种普遍存在的网络危害，通过多种直接和隐蔽的形式影响女性和性别少数群体。当前用于性别伤害检测的模型和标注数据集主要被来自全球北方语言的数据以及用这些语言训练的模型所主导。我们与来自约旦、法语西非和肯尼亚的民间社会组织（CSOs）合作，开发了一种社区主导的方法，通过众包方式收集黎凡特阿拉伯语、非洲法语和斯瓦希里语中符合当地文化和语言特点的TFGBV实例。我们使用主动学习对众包数据进行标注，构建了一个规模虽小但性能出色的数据集（N=3022），命名为ALTO，用于改进这三种语言的TFGBV检测。通过基准测试和微调实验，我们表明现有的有害内容检测模型在ALTO数据集上的TFGBV召回率较低，而经过微调后性能有所提升。我们发现模型往往无法识别隐蔽的……

    arXiv:2611.06706v1 Announce Type: cross  Abstract: Tech-facilitated gender based violence (TFGBV) is a pervasive online harm affecting women and gender minority groups through its many direct and covert forms. Current models and annotated datasets for gendered harm detection are dominated by data from, and models trained in, Global North languages. In collaboration with civil society organizations (CSOs) from Jordan, Francophone West Africa, and Kenya, we developed a community-led approach to crowdsource culturally and linguistically relevant instances of TFGBV in Levantine Arabic, African French, and Swahili. We annotated the crowdsourced data using active learning to develop a small but performant dataset (N=3022) called ALTO to improve TFGBV detection in the three languages. Through benchmarking and fine-tuning experiments, we show that existing harmful content detection models show poor TFGBV recall within ALTO and improve with fine-tuning. We found that models often miss covert ha
    
[^16]: 超越准确性：公共服务中AI保障需要确立的内容

    Beyond accuracy: what AI assurance needs to establish in public services

    [https://arxiv.org/abs/2611.06655](https://arxiv.org/abs/2611.06655)

    本文通过GPT-4o对英国首相质询进行政策主题分类的实验发现，准确率相近的不同AI配置会产生显著不同的输出结果，且AI之间的一致性不能保证正确性，因此公共服务中的AI保障不能仅依赖准确率指标，还需要建立更全面的可靠性与问责评估机制。

    

    公共机构越来越依赖AI产生的证据，包括分类、摘要和评估。这些输出可以为政策决策和公共服务提供依据，使其可靠性成为公共问责的重要问题。要确定这些证据是否可以信赖，需要测试其输出结果，并审查组织如何判断这些测试是否足够充分。我们结合技术评估与AI治理要求分析，并借鉴测量理论和管理型监管理论。我们比较了八个GPT-4o配置将1,200条英国首相质询问题分类到21个政策主题的表现，与人工编码标签进行对比，并在相同输入的重复测试中进行检验。各配置的准确率介于68.2%到70.3%之间，然而24.3%的问题获得了不同的标签，各主题的份额和排名也随之变化。各配置之间的一致性高于其与人工编码的一致性：一致性并不能保证正确性。

    arXiv:2611.06655v1 Announce Type: cross  Abstract: Public bodies increasingly rely on AI-produced evidence, including classifications, summaries and assessments. These outputs can inform policy decisions and service delivery, making their reliability a matter of public accountability. Establishing whether this evidence can be relied upon requires testing its outputs and examining how organisations decide whether those tests are sufficient. We combine technical evaluation with analysis of AI governance requirements, informed by measurement theory and management-based regulation. We compare eight GPT-4o configurations classifying 1,200 UK Prime Minister's Questions into 21 policy topics, against human-coded labels and across unchanged repeats. Accuracy ranges from 68.2% to 70.3%, yet 24.3% of questions receive different labels, with changes in topic shares and rankings. Configurations agree more with one another than with human coding: agreement does not guarantee correctness. Under the 
    
[^17]: 大语言模型在政治与社会科学研究中的可复制性、可重复性与稳健性

    Replicability, Reproducibility, and Robustness in Political and Social Science Research with Large Language Models

    [https://arxiv.org/abs/2611.06639](https://arxiv.org/abs/2611.06639)

    本章提出了基于结果分布的概率性可重复性标准，并通过开放权重模型的堆叠集成方法，解决大语言模型输出概率性变化对政治与社会科学研究下游统计结果可靠性的影响。

    

    本章探讨了大语言模型研究中的核心可重复性问题。大语言模型的输出具有概率性，且在不同模型和版本之间存在变化，这可能对下游统计结果产生显著影响。本章区分了可复制性、可重复性与稳健性三个概念，并提出了一种基于结果分布而非完全相同输出的概率性标准。本章引入了一个标准化的提示脚本，测试了编码器和解码器模型的内部变异性，并开发了误差结构分析，以诊断模型在何处出错以及它们的误差是否一致。通过使用三个数据集，本章表明测量方式的选择可能会逆转下游回归系数的符号。基于设计的、使用金标准标注子样本的现有修正方法可以恢复效度，但代价是估计精度的损失。本章的主要贡献是一种由开放权重模型组成的堆叠集成方法

    arXiv:2611.06639v1 Announce Type: new  Abstract: This chapter addresses the central reproducibility problem of research with large language models. LLM output is probabilistic and varies across models and versions, which can have significant effects on downstream statistical results. The chapter distinguishes between replicability, reproducibility, and robustness, and proposes a probabilistic standard based on the distribution of results, not identical outputs. It introduces a standardized prompt script, tests within-model variation for encoders and decoders, and develops error-structure analysis to diagnose where models err and whether their errors align. Using three datasets, the chapter shows that measurement choice can reverse the sign of downstream regression coefficients. Existing design-based corrections with gold-labeled subsamples restore validity, but come at the cost of estimate precision. The main contribution of the chapter is a stacked ensemble of open-weight models that 
    
[^18]: 摘要何时变成指令？LLM智能体自写交接摘要中缺乏依据的策略声明

    When Do Summaries Become Instructions? Unsupported Policy in LLM Agents' Self-Written Handoff Summaries

    [https://arxiv.org/abs/2611.06545](https://arxiv.org/abs/2611.06545)

    该论文发现LLM智能体在上下文压缩时自写的交接摘要中会偶发出现缺乏记录依据的限制或授权声明，其中虚假的授权会显著增加后继智能体的未授权操作，而虚假限制的影响则脆弱且不确定。

    

    长时间运行的LLM智能体会压缩其上下文，用一份摘要替换大部分历史记录。我们探究这些交接摘要是否会陈述记录所不支持的权限或要求，这类句子是否会改变下一个实例（后继者）的行为，以及它们从何而来。在带有真实权限账本的合成工具使用任务中，同快照回放方法将后继者重新运行，并用中性填充文本替换这些完整句子。两名AI阅读器仅阅读经规则标记的保留摘要，确认了来自两个开源权重模型的可读摘要中分别有1.2%和2.4%包含缺乏依据的限制性声明，以及0.1%和1.3%包含缺乏依据的授权声明。对于限制性声明，预先指定的回放测试结果不显著（p = 0.58）；其后计划开展的后续测试发现了一个脆弱的效应（p = 0.027）。中和虚假批准将120个保留运行中的未授权操作从24个减少到7个（事后分析；后续集合：48个中从17个减少到11个）。

    arXiv:2611.06545v1 Announce Type: new  Abstract: Long-running LLM agents compact their context, replacing most history with a summary. We ask whether handoff summaries state permissions or requirements the record does not support, whether such sentences change what the next instance (the successor) does, and where they come from. Synthetic tool-use tasks carry a ground-truth permission ledger; same-snapshot replay reruns the successor with those whole sentences replaced by neutral filler. Two AI readers, reading only rule-flagged held-out summaries, confirmed unsupported restrictions in 1.2% and 2.4% of all readable summaries from two open-weight models, and unsupported authorizations in 0.1% and 1.3%. For restrictions, a pre-specified replay test was not significant (p = 0.58); a follow-up, planned after it, found a fragile effect (p = 0.027). Neutralizing false approvals cut unauthorized actions from 24 to 7 of 120 held-out runs (post hoc; later set: 17 to 11 of 48). Most development
    
[^19]: 方言至关重要：面向伊加拉语-英语机器翻译的方言感知基线模型

    Dialect Matters: A Dialect-Aware Baseline for Igala-English Machine Translation

    [https://arxiv.org/abs/2611.06517](https://arxiv.org/abs/2611.06517)

    该论文发布了首个带方言标签的伊加拉语-英语平行语料库及方言感知机器翻译基线模型，发现同一英语句子在伊加拉语中存在20-60种方言变体，且混合方言训练优于单一方言训练。

    

    智能系统在人类活动的各个领域的深度渗透极大地提升了清晰度、便利了工作，并增进了全球性的普遍理解，这显著激发了所有用户对智能系统的使用。智能系统在很大程度上依赖精确的数据集/语料库来确保其输出的正确性。尽管研究人员努力解决自然语言处理（NLP）中的低资源语言问题，伊加拉语至今仍然没有任何平行语料库。我们提出了一个面向伊加拉语-英语机器翻译的方言感知基线模型，这是最早的开源系统之一，也是首个包含方言变异分析的系统。我们的数据集包含1137个句子（其中899个用于训练），并首次带有方言标签：Idah 459句（40.4%）、Ankpa 297句（26.1%）、Ibaji 160句（14.1%）、Other 187句（16.4%）、Ogugu 32句（2.8%）。相同的英语句子可以产生20至60种不同的伊加拉语表达形式。混合方言训练实现了均衡的BLEU分数11.66，而仅使用Idah方言训练的模型则表现欠佳。

    arXiv:2611.06517v1 Announce Type: new  Abstract: The massive permeability of intelligent systems in every sphere of human activities have greatly enhanced clarity, ease of work, and increased general global understanding which has significantly awaken the use of intelligent systems by all users. Intelligent systems greatly rely on precise dataset/corpus for the correctness of their output. Despite efforts by researchers to address Low-resource language in the Natural Language Processing (NLP), Igala still has zero parallel corpora despite efforts by researchers. We present a dialect-aware baseline for Igala-English machine translation, among first open systems and first with dialectal variation analysis. Our dataset of 1137 sentences (899 training) is first with dialect tags: Idah 459 (40.4%), Ankpa 297 (26.1%), Ibaji 160 (14.1%), Other 187 (16.4%), Ogugu 32 (2.8%). Same English yields 20-60 Igala forms. Mixed-dialect training achieves balanced BLEU 11.66 versus Idah-only model which d
    
[^20]: NaijaFacts：一个将语言与知识从关于尼日利亚的事实中分离出来的框架

    NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria

    [https://arxiv.org/abs/2611.06504](https://arxiv.org/abs/2611.06504)

    该论文提出NaijaFacts基准，通过国家×语言矩阵将语言效应与知识效应分离，发现大语言模型对尼日利亚事实的知识缺口（英语提问准确率仅20%）远大于低资源语言带来的语言效应。

    

    面向非洲语言的多语言基准测试大多是将通用问题翻译成当地语言，衡量的是语言能力，而非模型是否了解关于非洲的事实。我们提出了NaijaFacts，这是一个关于尼日利亚的事实性和统计性问题基准，由母语者撰写，并基于权威的官方统计数据，每个问题都有英语、尼日利亚皮钦语、约鲁巴语、伊博语和豪萨语版本。通过让模型回答针对英国和美国的类似问题，我们构建了一个国家×语言的矩阵，从而将语言效应（模型知道该事实，但在低资源语言中性能下降）与知识效应（该事实在任何语言中都不存在）分离开来。在八个前沿模型和尼日利亚适配模型上，知识效应占主导地位：以英语提问时，针对美国/英国的准确率为49%–50%，而针对尼日利亚仅为20%。语言效应较小且与网络表征程度相关，从英语到约鲁巴语逐渐下降（在尼日利亚问题上下降9个百分点）……

    arXiv:2611.06504v1 Announce Type: cross  Abstract: Multilingual benchmarks for African languages mostly translate generic questions into local languages, measuring linguistic competence but not whether models know facts about Africa. We introduce \naija, a benchmark of factual and statistical questions about Nigeria, authored by native speakers and grounded in authoritative official statistics, with every question available in English, Nigerian Pidgin, \yoruba, Igbo, and Hausa. Answering similar questions for the UK and US yields a country by language matrix that disentangles a language effect (the model knows the fact but degrades in a low-resource language) from a knowledge effect (the fact is absent in every language). Across eight frontier and Nigeria-adapted models, the knowledge effect dominates: English-query accuracy is 49--50\% for the US/UK but 20\% for Nigeria. The language effect is smaller and tracks web representation, degrading from English to \yoruba{} (9 points on Nige
    
[^21]: CRESiST：面向同声传译的自增强探索性策略学习

    CRESiST: Self-Enhancing Explorative Policy Learning for Simultaneous Speech Translation

    [https://arxiv.org/abs/2611.06501](https://arxiv.org/abs/2611.06501)

    该论文提出CRESiST框架，通过基于全模态大模型的端到端语音与文本读写交错架构实现KV缓存完全复用、原生支持无限长流式同声传译，并借助RNN-T执行者与LLM评论者构成的自增强策略学习管线在无需离线对齐的情况下优化读写决策。

    

    同声传译需要在翻译质量与延迟之间取得平衡。虽然大语言模型（LLM）在离线翻译中表现出色，但当前基于LLM的同声传译系统主要依赖前缀增量生成范式。这种范式存在键值缓存冗余重复计算的问题，并且需要借助外部语音活动检测（VAD）来处理无界音频流。为克服这些瓶颈，我们提出了一种构建在全模态LLM之上的端到端读写交错框架。通过将语音块与文本token交错排列，我们的架构实现了KV缓存的完全复用，原生支持无需分段、无限长度的推理。此外，为了在没有严格离线对齐的条件下学习最优流式策略，我们引入了一种新颖的Actor-Critic自增强流水线：RNN-T执行者动态采样读写轨迹，而基于LLM的评论者通过内在语义和……（摘要在此处截断）

    arXiv:2611.06501v1 Announce Type: new  Abstract: Simultaneous Speech Translation (SimulST) requires balancing translation quality and latency. While Large Language Models (LLMs) excel in offline translation, current LLM-based SimulST systems predominantly rely on prefix-incremental generation. This paradigm suffers from redundant Key-Value cache recomputation and necessitates external Voice Activity Detection to process unbounded streams. To overcome these bottlenecks, we propose an end-to-end interleaved read-write framework built upon an omni-modal LLM. By interleaving speech chunks and text tokens, our architecture achieves full KV cache reuse, natively supporting segmentation-free, infinite-length inference. Furthermore, to learn optimal streaming policies without rigid offline alignments, we introduce a novel Actor-Critic self-enhancement pipeline. An RNN-T Actor dynamically samples read-write trajectories, while an LLM-based Critic evaluates these paths via intrinsic semantic and
    
[^22]: LibriSpeech-Real-QA：一个流式问答基准

    LibriSpeech-Real-QA: A Benchmark for Streaming Question Answering

    [https://arxiv.org/abs/2611.06496](https://arxiv.org/abs/2611.06496)

    该论文提出了LibriSpeech-Real-QA，一个用于实时语音交互的基准数据集，通过从LibriSpeech构建实时问答对来评估语音大语言模型在流式音频中识别目标词的能力。

    

    大多数语音大语言模型（SLMs）仍然以离线方式处理语音，只有在接收到完整的语音片段后才开始响应。然而，许多现实世界的应用需要流式语音交互，即音频持续流入，机器需要根据迄今为止接收到的音频块来回答问题。例如，用户可能要求系统在某个人说话的过程中持续统计其名字被提及的次数。我们观察到目前缺乏针对实时语音交互的基准测试和评估协议。受此启发，我们提出了LibriSpeech-Real-QA，一个用于实时语音交互的数据集和基准。我们从LibriSpeech构建了不同类型的实时问答对，以评估语音大语言模型能否在流式音频中识别目标词。我们评估了几种流行的实时商业模型，发现它们仍然倾向于……

    arXiv:2611.06496v1 Announce Type: new  Abstract: Most speech large language models (SLMs) still process speech in an offline manner. They only start responding after they receive the complete speech segment. However, many real-world applications require streaming speech interaction, for which the audio continuously flows in and the machine is required to answer questions based on the so far received audio chunks. For example, a user may ask the system to continuously count how many times a person's name has been mentioned while they are speaking. We observe a lack of benchmarks and evaluation protocols for real-time speech interaction. Motivated by this, we propose LibriSpeech-Real-QA, a dataset and a benchmark for real-time speech interaction. We construct different types of real-time question-answer pairs from LibriSpeech to evaluate whether speech LLMs can identify target words in streaming audio. We evaluate several popular real-time commercial models and find that they still tend 
    
[^23]: 多轮强化学习究竟在LLM智能体中提升了什么？

    What Does Multi-Turn Reinforcement Learning Really Improve in LLM Agents?

    [https://arxiv.org/abs/2611.06483](https://arxiv.org/abs/2611.06483)

    本文通过Pass@(K,T)评估发现多轮智能体强化学习的优势依赖充足的交互轮次预算，且主要体现在错误恢复与长程韧性而非单步决策质量上，进而提出在训练中动态改变交互预算的BudgetMix RL，以在多种执行设置下实现稳健性能。

    

    虽然多轮智能体强化学习（RL）在长程任务上表现出色，但其性能提升对交互约束仍然高度敏感。通过Pass@(K,T)在采样广度K和交互深度T上进行评估，我们表明轨迹层面的RL优势依赖于充足的轮次预算，当预算收紧时该优势会显著减弱。此外，智能体RL并非均匀地提升单步决策质量；相反，其优势集中在错误恢复和长程韧性上。为缓解这种预算依赖带来的部署风险，我们对过程监督信号的益处与局限性进行了初步讨论，并引入BudgetMix RL作为替代方案，该方案在训练期间改变交互预算。实验表明，BudgetMix RL在多种执行设置下均能取得稳健的性能。项目主页：https://rethinking-agentic-rl.github.io/

    arXiv:2611.06483v1 Announce Type: cross  Abstract: While multi-turn agentic reinforcement learning (RL) excels on long-horizon tasks, its performance gains remain highly sensitive to interaction constraints. Evaluating across sampling breadth $K$ and interaction depth $T$ via $Pass@(K,T)$, we show that trajectory-level RL advantages depend on sufficient turn budgets, diminishing significantly when budgets tighten. Besides, agentic RL does not uniformly elevate single-step decision quality; rather, its strength concentrates in error recovery and long-horizon resilience. To mitigate the deployment risks caused by this budget dependence, we provide a preliminary discussion on the benefits and limitations of process supervision signals and introduce BudgetMix RL as an alternative that varies interaction budgets during training. Experiments demonstrate that BudgetMix RL achieves robust performance across diverse execution settings. Project Page: https://rethinking-agentic-rl.github.io/
    
[^24]: ArcRAG：基于知识图谱的智能体式组合检索

    ArcRAG: Agentic Compositional Retrieval over Knowledge Graphs

    [https://arxiv.org/abs/2611.06481](https://arxiv.org/abs/2611.06481)

    ArcRAG通过在推理时归纳查询局部的图谱接口并规划可修复的多跳遍历，实现了无需任何构建特定调整即可跨不同图谱构建方式保持有效的智能体式组合检索。

    

    基于图的检索增强生成（GraphRAG）通过在从语料库构建的知识图谱上进行检索来回答问题。然而，检索策略通常围绕其自身构建流程所产生的结构来设计，这些策略在构建方式改变后是否依然有效却很少被测试。我们研究了这种跨构建的可移植性：即一个不依赖特定构建方式的模式、关系名称或重新调优的检索流程，在图谱以不同方式构建时是否仍能保持有效。我们提出了ArcRAG，它在推理时归纳出一个查询局部的图谱接口（即从问题所命名的实体出发可达的节点标签、关系描述符和属性键），据此规划多跳遍历，并以Cypher语言执行和修复这些遍历。我们将MuSiQue、HotpotQA和TriviaQA语料库分别构建为模式引导的图谱（类型化节点、封闭关系集合、来源溯源）和开放信息抽取图谱（摘要在此处被截断）

    arXiv:2611.06481v1 Announce Type: cross  Abstract: Graph-based retrieval-augmented generation (GraphRAG) answers questions by retrieving over a knowledge graph built from a corpus. Yet retrieval strategies are designed around the structure their own construction pipeline produces, and whether they survive a change of construction is rarely tested. We study this cross-construction portability: whether one retrieval procedure, with no construction-specific schema, relation names, or retuning, stays effective when the graph is built differently. We introduce ArcRAG, which induces a query-local graph interface at inference time (the node labels, relation descriptors, and property keys reachable from the entities a question names), plans multi-hop traversals against it, and executes and repairs them as Cypher. We build each of the MuSiQue, HotpotQA, and TriviaQA corpora as a schema-guided graph (typed nodes, a closed relation set, source provenance) and as an open-information-extraction gra
    
[^25]: EvoGuard：缓解在线技能演化中的退化问题

    EvoGuard: Mitigating Degradation in Online Skill Evolution

    [https://arxiv.org/abs/2611.06444](https://arxiv.org/abs/2611.06444)

    EvoGuard 提出一种低延迟、低成本的成对技能选择器，通过私有测试集效用标签和拒绝采样筛选技能对进行训练，从而在线缓解语言智能体技能演化中的退化问题。

    

    技能让语言智能体能够以自然语言文档的形式跨任务复用经验。近期研究开始从交互中演化技能，但现有方法在技能在线演化时存在局限。先前的方法从跨多任务收集的一批轨迹中离线更新技能。其中一些方法随后不加任何检验地直接采用新技能，这可能导致技能随时间推移而退化。另一些方法仅当新技能在留出验证集上提升了性能时才保留它，但这种方式在每次更新后都要运行，太慢且成本太高。我们提出了 EvoGuard，一种成对技能选择器，能够以低延迟和低成本缓解技能退化问题。我们首先迭代地运行智能体以构建技能池，并在一个私有测试集上对每个技能进行评分以获得效用标签。然后我们将技能两两配对，并使用拒绝采样只保留具有明显效用差距的技能对。该选择器基于两个标签词元的 logits 进行训练，从而挑选出更好的技能。在部署时，它可以在（原文摘要于此处截断）

    arXiv:2611.06444v1 Announce Type: new  Abstract: Skills let language agents reuse experience across tasks as natural-language documents. Recent work evolves skills from interaction, but existing methods have limitations when skills evolve online. Prior methods update a skill offline from a batch of trajectories collected over many tasks. Some then adopt the new skill without any check, which can degrade the skill over time. Others keep it only if it improves performance on a held-out validation set, which is too slow and costly to run after every update. We introduce EvoGuard, a pairwise skill selector that mitigates skill degradation at low latency and cost. We first run the agent iteratively to build pools of skills and score each skill on a private test set to obtain utility labels. We pair the skills and use rejection sampling to keep only pairs with a clear utility gap. The selector is trained on the logits of two label tokens to pick the better skill. At deployment it decides in 
    
[^26]: 曲折与转折：一个面向探索任务中长期记忆的基于图的基准与框架

    Twists and Turns: A Graph-Based Benchmark and Framework for Long-Term Memory in Exploration Tasks

    [https://arxiv.org/abs/2611.06414](https://arxiv.org/abs/2611.06414)

    该论文提出LExMem，一个基于图的长期记忆评测基准与框架，通过将试错式探索过程建模为图结构并引入演化机制，来评估模型在长期探索任务中追踪方案修订、分支与重新考量的记忆能力。

    

    长期问题解决过程，如科学研究和软件开发，是沿着一条不断演化的试错轨迹向中心目标推进的。我们将这类任务称为“探索任务”，它们要求模型不仅记住事实，还要记住解决方案如何针对中间结果进行检验，以及早期尝试如何被不断修改或重新启用。然而，现有的长期记忆研究主要关注事实的回忆与更新，忽视了探索过程的相互关联特性。我们观察到，探索过程天然形成图结构，其中分支的方法通过修订与重新考量相互关联。受此启发，我们提出了LExMem——一个以图为引导的基准，旨在评估长期探索任务中的记忆能力。我们将探索过程建模为图，并引入一种演化机制来增加其结构复杂性。

    arXiv:2611.06414v1 Announce Type: new  Abstract: Long-term problem-solving processes, such as scientific research and software development, proceed along an evolving trial-and-error trajectory toward a central goal. We term these \textit{exploration tasks}, which require models to remember not only facts but also how solutions are tested against intermediate outcomes and early attempts are continually revised or revived. However, existing long-term memory research focuses on factual recall and updates, overlooking the interconnected nature of exploration. We observe that exploration processes naturally form graphs, with branching approaches linked by revisions and reconsiderations. Motivated by this, we propose \textsc{LExMem}, a graph-guided benchmark designed to evaluate \underline{\textsc{Mem}}ory in \underline{\textsc{L}}ong-term \underline{\textsc{Ex}}ploration tasks. We model exploration processes as graphs and introduce an evolutionary mechanism to increase their structural comp
    
[^27]: Ruby-OPD：通过在线策略蒸馏增强流匹配文本到语音的发音可控性与准确性

    Ruby-OPD: Enhancing Pronunciation Controllability and Accuracy for Flow-Matching Text-to-Speech via On-Policy Distillation

    [https://arxiv.org/abs/2611.06403](https://arxiv.org/abs/2611.06403)

    该论文提出Ruby-OPD两阶段框架，利用带正字扰动的Ruby注音文本表示与在大规模纯文本语料上的在线策略蒸馏（OPD），使流匹配日语TTS同时具备高发音准确性和细粒度的发音可控性。

    

    实现高发音准确性与细粒度可控性仍然是日语文本到语音（TTS）中的核心挑战，因为日语是一种未分词的语言，混合了表意的汉字与表音的假名，并存在大量的多音字现象。传统的级联系统通过前端将文本转换为音素序列，从而确保发音控制，但会丢失表层文本中的词汇语义和韵律线索。相反，近期的零样本TTS模型直接使用原始正字文本以提升自然度，但会产生多音字错误且缺乏显式的发音控制。为弥合这一差距，我们提出了一个面向流匹配TTS的两阶段框架。首先，我们研究了带有正字扰动的Ruby（注音）文本表示，以联合保留表层文字形式和目标发音，从而实现可靠的控制。其次，我们在大规模纯文本语料库上引入在线策略蒸馏（On-Policy Distillation，OPD），以增强发音（准确性与可控性）。

    arXiv:2611.06403v1 Announce Type: new  Abstract: Achieving both high pronunciation accuracy and fine-grained controllability remains a central challenge in Japanese text-to-speech (TTS), as Japanese is an unsegmented language mixing logographic Kanji and phonographic Kana with extensive polyphony. Traditional cascaded systems convert text into phoneme sequences via a front-end, ensuring pronunciation control but discarding lexical semantics and prosodic cues in surface orthography. Conversely, recent zero-shot TTS models directly consume raw orthographic text to improve naturalness, yet suffer from polyphone errors and lack explicit pronunciation control. To bridge this gap, we propose a two-stage framework for flow-matching TTS. First, we investigate Ruby text representations with orthography perturbation to jointly preserve surface forms and target pronunciations for reliable control. Second, we introduce On-Policy Distillation (OPD) on large-scale text-only corpora to enhance pronun
    
[^28]: 基于层次化问题的可解释文本嵌入

    Interpretable Text Embeddings Based on Hierarchical Questions

    [https://arxiv.org/abs/2611.06402](https://arxiv.org/abs/2611.06402)

    该论文提出一种基于层次化问题的可解释文本嵌入方法，通过递归聚类、对比提示和问题对齐构建问题层次结构，在保持较高任务性能的同时，显著减少了人工解释时需要检查的问题数量。

    

    基于问答的文本嵌入将文本表示为由是/否问题的1/0答案组成的向量。尽管这类嵌入能够为解释提供详细信息并取得较高的任务性能，但它们通常需要人工检查大量问题。此外，问题数量与问题粒度适当级别之间的关系尚未得到充分研究。为了应对这些挑战，我们提出了一种为文本嵌入构建层次化问题的方法。随着层次的加深，问题变得更加具体，数量也随之增加。在解释过程中，我们通过用较浅层次的对齐问题替换较深层次的问题，减少了人工需要检查的问题数量。该问题层次结构通过递归聚类、对比提示和问题对齐来构建。在通用任务上的实验结果表明（摘要内容在此截断）。

    arXiv:2611.06402v1 Announce Type: new  Abstract: QA-based text embeddings represent texts as vectors consisting of 1/0 answers to yes/no questions. Although these embeddings provide detailed information for interpretation and achieve high task performance, they often require humans to examine a large number of questions. Furthermore, the relationship between the number of questions and the appropriate level of question granularity has not been sufficiently studied. To address these challenges, we propose a method for constructing hierarchical questions for text embeddings. As the hierarchy deepens, questions become more specific and their number increases. During interpretation, we reduce the number of questions that humans need to examine by replacing questions at deeper levels with aligned questions at shallower levels. The question hierarchy is constructed through recursive clustering, contrastive prompting, and question alignment. Experimental results on general-purpose tasks show 
    
[^29]: KV缓存修复预算应该花在哪里？非前缀复用中的干扰块结构与上下文兼容性

    Where Should a KV-Cache Repair Budget Go? Distractor Structure and Contextual Compatibility in Non-Prefix Reuse

    [https://arxiv.org/abs/2611.06397](https://arxiv.org/abs/2611.06397)

    该研究发现在非前缀KV缓存复用中，有限的修复预算应优先用于针对复用缓存重新计算目标文本块，而非修复周围分散的干扰文本块，这种“上下文兼容性”机制能有效恢复上下文相关问题的回答能力。

    

    非前缀KV缓存复用方法将检索到的文本块独立编码并拼接其缓存。为了探究小幅修复预算应花在何处，我们在三个7B-8B模型和四个数据集（涵盖对话记忆、开放域问答和API文档）上，围绕承载答案的文本块（目标块）变化其周围文本块的配置。实验发现，复用会损失完全预填充所能回答的上下文相关问题中的19%-100%，且当其他文本块与目标块属于同类时损失更大。由于可恢复的损失分散在这些其他文本块上，仅恢复其中少数几个收效甚微，因此预算最好用于目标块。其作用机制在于上下文兼容性：在所有七个测试设置中，针对复用缓存重新计算目标块都优于直接插入其完全预填充的KV，尽管重新计算得到的KV与完全预填充的KV相距甚远。然而，问题注意力在复用缓存上最多仅在18.8%的开发集问题中选中目标块。

    arXiv:2611.06397v1 Announce Type: new  Abstract: Non-prefix KV-cache reuse encodes retrieved chunks independently and concatenates their caches. To find where a small repair budget should be spent, we vary the chunks around the answer-bearing one (the target) across three 7B-8B models and four datasets of dialogue memory, open-domain QA and API documentation. Reuse loses 19-100% of the context-dependent questions that full prefill answers, and it loses more when the other chunks are of the same kind as the target. Because the recoverable damage is spread over these other chunks, restoring a few of them recovers little, and the budget is better spent on the target. This works through contextual compatibility: recomputing the target against the reused cache beats inserting its full-prefill KV in all seven settings tested, although the recomputed KV stays far from the full-prefill KV. Question attention over reused caches, however, selects the target in at most 18.8% of development questi
    
[^30]: 主观文本评估中类型化决策模型的呈现敏感性

    Presentation Sensitivity in Typed Decision Models for Subjective Text Evaluation

    [https://arxiv.org/abs/2611.06384](https://arxiv.org/abs/2611.06384)

    该论文通过70.2万次受控判断审计发现，用于主观文本评估的类型化决策模型对呈现方式高度敏感——即使被评估文本不变，位置、背书和决策框架的改变也会使模型选择发生最高达40.40%的显著偏移。

    

    类型化决策模型为低成本的自动化评估提供了结构化的选择、分数和概率。我们对Jev、Clef Flash和Perplexity Decider在原生选择与评分界面中的呈现敏感性进行了一项受控审计。实验涵盖六个数据集，包含9,000个经过多次标注的幽默、有用性和摘要类条目；在保持被评估文本不变的前提下，我们通过改变位置、标识符、作者声明、背书、决策框架和评分标准等干预措施，共收集了702,000个判断结果。结果显示：Jev在摘要任务上的第二位置概率优势达到6.11个百分点；反转读者背书会使Perplexity在40.40%的文本配对中改变所选的幽默文本；拒绝式框架使Clef的摘要选择与人类偏好的一致性下降23.36个百分点。评分格式也会导致分数水平的变化，但通常能保留原有排名。基于这些发现，可重复性、呈现鲁棒性……

    arXiv:2611.06384v1 Announce Type: new  Abstract: Typed decision models offer structured choices, scores, and probabilities for low-cost automated evaluation. We present a controlled audit of presentation sensitivity in Jev, Clef Flash, and Perplexity Decider across native choice and score interfaces. Across six datasets with 9,000 multiply annotated humor, helpfulness, and summarization items, we collect 702,000 judgments under interventions that vary positions, identifiers, authorship claims, endorsements, decision framing, and rating rubrics while preserving the evaluated texts. Jev's second-position probability advantage reaches 6.11 percentage points on summaries; reversing reader endorsements changes Perplexity's selected humor text on 40.40% of pairs; and rejection framing reduces Clef's summary agreement with human preferences by 23.36 points. Rating formats also shift score levels while often preserving rankings. Across these findings, repeatability, presentation robustness, an
    
[^31]: LegalEvolver：基于评分准则引导自我进化的智能体法律判决预测

    LegalEvolver: Agentic Legal Judgment Prediction via Rubric-Guided Self-Evolution

    [https://arxiv.org/abs/2611.06381](https://arxiv.org/abs/2611.06381)

    提出 LegalEvolver 框架，通过评分准则引导的自我进化机制将查询启发式优化为种子，为法律判决预测生成更具法律信息量的检索查询，从而突破现有法律搜索智能体受限于查询质量的性能瓶颈。

    

    随着基于大语言模型（LLM）的智能体不断发展，它们已被越来越广泛地应用于法律领域。法律判决预测（LJP）要求模型进行多步推理并检索外部法律证据。现有法律搜索智能体的性能往往受限于检索查询的质量：它们通常直接根据用户的初始请求构建查询，而忽略了潜在的法律问题和具有法律重要性的细节。为解决这一问题，我们提出了 LegalEvolver，一个专为 LJP 导向检索设计的查询优化框架，用于生成具有法律信息量的查询。采样高质量的优化查询及其对应的搜索轨迹极具挑战性，因为仅依赖无引导的高温采样（rollouts）既低效，又受限于模型自身初始能力。因此，我们采用启发式方法将查询优化为种子，从而提升轨迹生成的效率……（摘要原文在此处被截断）

    arXiv:2611.06381v1 Announce Type: new  Abstract: As LLM-based agents develop, they have been increasingly applied in the legal domain. Legal Judgment Prediction~(LJP) requires models to conduct multi-step reasoning and external legal evidence retrieval. Existing legal search agents' performance is often constrained by the quality of retrieval queries. They often formulate queries directly from the initial user request, overlooking latent legal issues and legally salient details. To address this issue, we propose \emph{LegalEvolver}, a query optimization framework specifically designed to generate legally informative queries for LJP-oriented retrieval. Sampling high-quality optimized queries and their corresponding search trajectories is challenging, as relying solely on unguided high-temperature rollouts is both inefficient and constrained by the model's initial capabilities. We therefore heuristically optimize queries as seeds, improving the efficiency of trajectory generation. Furthe
    
[^32]: 上下文压缩能否保持拒绝对齐？

    Does Context Compression Preserve Refusal Alignment?

    [https://arxiv.org/abs/2611.06380](https://arxiv.org/abs/2611.06380)

    论文发现基于编码器的上下文压缩虽然保留了语义内容，却会系统性削弱语言模型的拒绝对齐行为，并提出了一种无需重训练、在压缩表示空间中运行的轻量级推理时干预方法 Memory Steering，可将拒绝率恢复至接近基线水平。

    

    上下文压缩通过将输入编码为紧凑的表示来降低推理成本，同时保留语义内容。一个悬而未决的问题是，仅凭语义保留是否足以维持诸如拒绝对齐之类的下游行为。我们对这一问题进行了研究，发现尽管重建保真度很高，基于编码器的压缩仍会系统性地削弱指令微调语言模型中的拒绝行为。这一效应在不同的模型家族和压缩架构中均持续存在。机制分析表明，压缩会沿着解码器学到的拒绝方向削弱激活。我们进一步提出了 Memory Steering，这是一种轻量级的推理时干预方法，无需重新训练即可将拒绝率恢复到接近基线的水平，并且完全在压缩表示空间中运行。这些结果表明，语义保留并不能保证压缩下的行为保留。

    arXiv:2611.06380v1 Announce Type: new  Abstract: Context compression reduces inference cost by encoding inputs into compact representations while preserving semantic content. An open question is whether semantic preservation alone is sufficient to maintain downstream behaviours such as refusal alignment. We investigate this question and find that encoder-based compression systematically weakens refusal behaviour in instruction-tuned language models, despite high reconstruction fidelity. This effect persists across model families and compression architectures. Mechanistic analysis shows that compression attenuates activation along the decoder's learned refusal direction. We further explore Memory Steering, a lightweight inference-time intervention that restores refusal rates to near-baseline levels without retraining and operates entirely in compressed representation space. These results demonstrate that semantic preservation does not guarantee behavioural preservation under compression
    
[^33]: 以草根契约圈控人工智能风险

    Corralling AI Risks with Grassroots Contracts

    [https://arxiv.org/abs/2611.06374](https://arxiv.org/abs/2611.06374)

    提出以“草根契约”这一数字化契约机制，来圈控AI操控全球平台、匿名AI智能体逃避追责以及无法归因的深度伪造内容这三大AI风险。

    

    人工智能的转折点带来了大量令人担忧的风险。数字平台尤其容易受到三大AI风险的影响，因为全球平台的控制者正在将控制权让渡给AI——包括算法控制权、成员控制权和内容控制权：(1) AI控制被人类大多数人所使用的全球社会与经济平台，导致大多数人类容易受到AI的操控。(2) 无法问责的AI智能体在社交与经济数字平台上与人类并肩匿名运行，可能伪装成人类，为身份不明的主人服务，追求未知的目标，在未知的约束下运行，缺乏问责机制和责任追究。(3) 由匿名AI生成、无法归因的深度伪造内容，人们无法将其与真实内容区分开来，其制作者则逃避问责和追责。在此，我们提出用草根契约来圈控这些AI风险，草根契约是一种……（原文摘要在此处截断）

    arXiv:2611.06374v1 Announce Type: new  Abstract: AI's inflection point is a fountain of alarming risks. Digital platforms in particular are vulnerable to three major AI risks, as those in control of global platforms cede control to AI, of the algorithms, of membership and of content: (1) AI control of global platforms, social and economic, used by the majority of humanity, resulting in the majority of humanity becoming vulnerable to AI control. (2) Unaccountable AI agents operating anonymously on social and economic digital platforms, side-by-side with humans, possibly masquerading as humans, serving unknown masters with unknown goals, operating under unknown restraints without accountability or culpability. (3) Unattributed deepfakes generated by anonymous AI, with people unable to distinguish them from genuine content and with their producers evading accountability and culpability.   Here, we propose to corral these AI risks with grassroots contracts, a digital incarnation of a two-c
    
[^34]: 基于扩散语言模型的免训练摘要字符长度控制

    Training-Free Character-Length Control in Summarization with Diffusion Language Models

    [https://arxiv.org/abs/2611.06371](https://arxiv.org/abs/2611.06371)

    提出免训练推理时算法DELTA，通过在扩散语言模型的每个去噪步骤中估计期望字符长度并动态插入或删除掩码位置，实现对摘要字符长度的精确控制，将长度误差降低了一个数量级。

    

    固定字符长度的摘要生成对于标题和文章预览生成等任务非常重要，因为这些任务中的文本必须精确适配特定空间。自回归语言模型（LM）难以实现精确的字符长度控制：基于训练的方法需要针对特定任务的微调，而零样本提示策略只能达到近似的长度要求。我们提出了DELTA（Diffusion Expected Length Targeting Algorithm，扩散期望长度定向算法），这是一种面向掩码扩散语言模型的免训练推理时方法，它在每个去噪步骤中估计期望字符长度，并通过动态插入或删除掩码位置来引导模型朝向目标长度。DELTA无需微调，专为现成的掩码扩散语言模型设计。在三个摘要基准测试中，与基线方法相比，它将字符长度误差降低了一个数量级，同时在自动评估和人工评估中都保持了有竞争力的摘要质量。

    arXiv:2611.06371v1 Announce Type: new  Abstract: Summarization with a fixed character length is important for tasks such as headline and article preview generation, where text must fit an exact space. Autoregressive language models (LMs) struggle with precise character-length control: training-based methods require task-specific fine-tuning, while zero-shot prompting strategies achieve only approximate compliance. We propose DELTA (Diffusion Expected Length Targeting Algorithm), a training-free inference-time method for masked diffusion LMs that estimates expected character length at each denoising step and dynamically inserts or removes masked positions to steer toward a target length. DELTA requires no fine-tuning and is designed for off-the-shelf masked diffusion LMs. Across three summarization benchmarks, it reduces character-length errors by an order of magnitude compared to baselines while maintaining competitive summary quality in both automatic and human evaluation.
    
[^35]: 大规模生产具有惊人副作用的训练数据

    Mass Producing Training Data with Surprising Side Effects

    [https://arxiv.org/abs/2611.06324](https://arxiv.org/abs/2611.06324)

    本文提出一种自动化流程，利用LLM评判器结合进化算法、强化学习和潜意识学习方法，大规模生产在训练域外引发语言模型意外行为的“具有惊人副作用的数据集”，最终生成了6,575个涵盖8种行为的此类数据集，为研究训练数据对模型的隐性影响提供了系统性方法。

    

    训练数据可能在训练领域之外对语言模型行为产生意想不到的副作用：不安全的代码数据可能导致模型广泛失调，过时的鸟类名称可能诱发模型表现出19世纪的人格。理解这种泛化现象之所以困难，是因为目前只存在几十个孤立的例子，且每个例子都是手工构建或偶然发现的。我们提出自动发现这类“具有惊人副作用的数据集”（DSSEs）。我们为DSSEs定义了三个目标：（1）能够引发目标行为，（2）读起来像普通的训练后数据，（3）让阅读这些数据的人类审计者感到惊讶。我们使用LLM评判器对每个目标进行量化，并通过进化算法、强化学习和潜意识学习方法搜索高分数据集。经过重新评分和使用保留指标过滤后，我们的流程生成了一系列包含6,575个DSSEs的数据集，涵盖8种行为，其中许多数据集的效果能够跨模型家族泛化。

    arXiv:2611.06324v1 Announce Type: cross  Abstract: Training data can have unexpected side effects on language model behavior outside the training domain: insecure code data can induce broad misalignment, and outdated bird names can induce a 19th-century persona. Understanding such generalization is hard because only dozens of isolated examples exist, each constructed by hand or discovered by accident. We propose automatically discovering such datasets with surprising side effects (DSSEs). We define three objectives for DSSEs: they should (1) elicit a target behavior, (2) read like ordinary post-training data, and (3) be surprising to human auditors reading the data. We operationalize each with LLM judges, and search for high-scoring datasets via evolution, reinforcement learning, and subliminal learning approaches. After re-scoring and filtering with held-out metrics, our pipeline produces a collection of 6,575 DSSEs across 8 behaviors, many of which generalize across model families an
    
[^36]: Zuse：面向欧洲语言的决策模型

    Zuse: A Decision Model for European Languages

    [https://arxiv.org/abs/2611.06238](https://arxiv.org/abs/2611.06238)

    Zuse是首批面向27种欧洲语言的决策模型系列（4B/9B/27B），通过在冻结的开源基座上训练低秩适配器和指针头，实现了87.5%的准确率和极低的校准误差，并发现校准偏差主要由任务而非语言决定。

    

    决策模型读取文档并为固定选项集合中的每个选项返回概率，而非生成文本。2026年出现了若干此类模型，但没有一个发表过英语以外的结果。我们提出Zuse——三个决策模型（4B、9B和27B），它们在冻结的开源权重基座上训练低秩适配器和指针头，面向27种欧洲语言，并用它们研究这种设计在不同语言和任务上的表现。在一个由11个未参与训练的数据源组成的广泛欧洲评测套件上（涵盖9个任务组和27种语言，其中开发集也用于指导模型选择），27B模型达到87.5%的准确率，总体期望校准误差为0.016，按数据源加权时为0.035。对这些单种子结果的事后分析得出三点发现。其一，校准偏差是按任务而非按语言结构化的：在26种语言的平行阅读和主题类题目上，置信度差距的变化幅度大约与采样……（摘要原文在此截断）

    arXiv:2611.06238v1 Announce Type: new  Abstract: Decision models read a document and return a probability for each of a fixed set of options instead of generating text. Several appeared in 2026, but none has published results outside English. We describe Zuse, three decision models (4B, 9B and 27B) that train a low-rank adapter and a pointer head on frozen open-weight bases, targeting 27 European languages, and use them to study how this design behaves across languages and tasks. On a broad European suite of 11 sources excluded from training, covering 9 task groups and 27 languages, a development set that also guided model selection, the 27B model reaches 87.5% accuracy with a pooled expected calibration error of 0.016, and 0.035 when weighted per source. Post-hoc analyses of these single-seed results give three findings. First, miscalibration is structured by task rather than language: on parallel reading and topic items in 26 languages the confidence gap varies about as much as sampl
    
[^37]: 先循环，后飞跃：循环深度在数据受限预训练中能带来什么与不能带来什么

    Loop Before You Leap: What Recurrent Depth Can and Cannot Buy in Data-Limited Pretraining

    [https://arxiv.org/abs/2611.06206](https://arxiv.org/abs/2611.06206)

    该研究发现在1亿词的数据受限预训练中，循环深度（12层堆叠重复四次）在相同推理计算量下的纯文本任务中比标准48层Transformer高出5.4个BLiMP百分点，但无法缓解多模态训练导致的约3.8个百分点的语言能力下降。

    

    在BabyLM挑战赛中，多模态模型的表现往往低于纯文本基线（Klerings等人，2024；Takmaz等人，2025）。我们测试了额外的循环计算能否帮助缩小这一差距。循环机制通过重复使用层来增加计算量，而无需增加参数或训练数据。在单一实现下，我们在1亿词的预算下比较标准Transformer与循环Transformer，每个主要条件使用三个随机种子。每张图像由冻结的DINOv2编码器输出的一个特征向量表示，并作为前缀token使用。图像描述词计入预算，因此多模态训练也会改变语言数据本身。在一项预注册的纯文本比较中，重复四次的12层堆叠在推理计算量相同的情况下，其BLiMP得分比标准的48层Transformer高出5.4个百分点。然而，多模态训练使两种架构的BLiMP准确率均下降约3.8个百分点，且循环机制并未带来明显的缓解作用。

    arXiv:2611.06206v1 Announce Type: new  Abstract: Multimodal models have often scored below text-only baselines in the BabyLM Challenge (Klerings et al., 2024; Takmaz et al., 2025). We test whether additional recurrent computation helps close this gap. Recurrence increases computation by reusing layers without adding parameters or training data. Using a single implementation, we compare standard and looped Transformers under a 100M-word budget, with three seeds per main condition. Each image is represented by one feature vector from a frozen DINOv2 encoder, used as a prefix token. Caption words count toward the budget, so multimodal training also changes the language data. In a pre-registered text-only comparison, a 12-layer stack repeated four times exceeds a standard 48-layer Transformer by 5.4 BLiMP percentage points at matched inference compute. However, multimodal training lowers BLiMP accuracy by about 3.8 points in both architectures, with no clear mitigation from recurrence. An 
    
[^38]: 基于几何感知正则化的大型语言模型编辑

    Editing Large Language Models with Geometry-Aware Regularization

    [https://arxiv.org/abs/2611.06185](https://arxiv.org/abs/2611.06185)

    提出GEdit框架，通过低秩加对角（LR+D）因子模型捕捉知识表示的几何结构，利用低秩正则化保护核心知识子空间，并依据编辑难度自适应调整正则化强度，实现更精准的大模型知识编辑。

    

    知识编辑已成为更新大型语言模型知识的关键技术。然而，现有方法忽视了知识表示空间的几何结构，导致两个关键局限：它们既无法区分哪些预训练知识应被优先保留（核心知识 vs. 非核心知识），也无法区分哪些新编辑请求本质上较为困难（困难编辑 vs. 简单编辑）。为解决这一问题，我们提出了GEdit，一个基于低秩加对角（LR+D）因子模型的几何感知框架，该模型能够捕捉Transformer前馈网络（FFN）隐藏表示的内在低秩结构。GEdit包含两个互补机制：首先，低秩正则化对低秩核心知识子空间施加针对性保护；其次，自适应正则化根据每个编辑请求的可编辑性动态调整正则化强度——对困难编辑放松约束以确保编辑成功，对（原文摘要在此处截断）。

    arXiv:2611.06185v1 Announce Type: cross  Abstract: Knowledge editing has become a key technique for updating knowledge in large language models. However, existing methods neglect the geometric structure of the knowledge representation space, leading to two key limitations: they cannot distinguish which pre-trained knowledge should be preferentially preserved (core vs. non-core) nor which new edits are inherently difficult (hard vs. easy). To address this, we propose GEdit, a geometry-aware framework built on a Low-Rank Plus Diagonal (LR+D) factor model that captures the intrinsic low-rank structure of Transformer FFN hidden representations. GEdit features two complementary mechanisms. First, the low-rank regularization applies targeted protection on a low-rank core knowledge subspace. Second, the adaptive regularization dynamically adjusts the regularization strength based on each edit request's editability: relaxing constraints for hard edits to ensure success, and tightening them for
    
[^39]: AssameseLM-E2B：一个面向消费级设备高效部署的开源权重阿萨姆语语言模型

    AssameseLM-E2B: An Open-Weight Assamese Language Model for Efficient Consumer-Grade Deployment

    [https://arxiv.org/abs/2611.06170](https://arxiv.org/abs/2611.06170)

    本文提出专为低资源阿萨姆语设计的开源 20 亿参数语言模型 AssameseLM-E2B，通过扩展字节级 BPE 词表大幅降低分词开销，并借助参数高效的持续预训练与监督微调，实现了在消费级设备上的高效部署。

    

    尽管大型语言模型（LLM）在高资源语言中表现出色，但由于分词效率低下、生成不稳定性和语言干扰，其在低资源语言中的性能往往会有所下降。本文提出了 AssameseLM-E2B 及其指令微调变体 AssameseLM-E2B-it，这是一个专门为低资源阿萨姆语适配的 20 亿参数模型。该模型基于 Gemma-4 架构构建，并采用扩展的字节级 BPE 词表，与多语言 2B–3B 基线相比，将分词开销降低了近 500%。我们采用结合 Hybrid LoRA 的参数高效持续预训练（CPT）流程，随后在平衡的多领域数据集上进行监督微调（SFT），以增强指令遵循和检索增强生成（RAG）能力。评估结果显示，该模型在 FLORES-200 上取得了 0.7488 的每字节比特数（BPB）得分，同时总体 RAG 得分为 96。

    arXiv:2611.06170v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) excel in high-resource languages, their performance often degrades in low-resource languages due to tokenization inefficiencies, generative instability, and linguistic interference. This paper presents AssameseLM-E2B and its instruction-tuned variant, AssameseLM-E2B-it, a 2-billion-parameter model adapted specifically for low resource Assamese language. Built upon the Gemma-4 architecture, the model incorporates an expanded byte-level BPE vocabulary to reduce tokenization overhead by nearly 500% compared with multilingual 2B--3B baselines. We employ a parameter-efficient Continual Pre-Training (CPT) pipeline with Hybrid LoRA, followed by Supervised Fine-Tuning (SFT) on a balanced, multi-domain dataset to enhance instruction-following and Retrieval-Augmented Generation (RAG) capabilities.   Evaluation demonstrates a Bits-per-Byte (BPB) score of 0.7488 on FLORES-200, alongside an overall RAG score of 96
    
[^40]: 无需再训练即可加速大规模混合LLM的长上下文解码

    Accelerating Long-Context Decoding in Large-Scale Hybrid LLMs Without Retraining

    [https://arxiv.org/abs/2611.06133](https://arxiv.org/abs/2611.06133)

    该论文提出一种无需再训练的组合稀疏注意力方法，通过逐层多目标搜索确定滑动窗口、固定前缀与相关token预算，在保持混合架构大模型长上下文准确率的同时，将64k/256k token解码速度最高提升1.26/2.22倍。

    

    大语言模型（LLM）的全注意力层使得每生成一个token的代价都比前一个更高，从而在长上下文任务中拖慢了模型速度。混合模型通过用线性注意力层替换大部分全注意力层来降低这一开销，但全注意力层在长上下文场景下仍然主导解码过程。我们在不进行任何训练或权重修改的情况下，利用组合稀疏注意力来加速这些剩余的层：每一层关注近期token的滑动窗口、固定的前缀（pinned prefix）以及上下文中固定预算的最相关token。窗口大小和预算通过多目标搜索算法逐层确定。在来自三个模型家族的六个混合模型上（参数规模从4B到397B），所选配置在21个长上下文数据集上保持了基线模型的准确率。在解码64k/256k个token时，Qwen3.5-4B上的速度提升最高达1.26倍/2.22倍，Qwen3.5-122B-A10B上提升1.20倍/2.23倍，以及1.1（摘要原文在此处被截断）。

    arXiv:2611.06133v1 Announce Type: new  Abstract: The full-attention layers of large language models (LLMs) make every generated token more expensive than the previous one, slowing models down on long-context tasks. Hybrid models reduce this cost by replacing most full-attention layers with linear-attention layers, but the first still dominate decoding at long contexts. We accelerate these remaining layers without any training or weight changes using combined sparse attention: each layer attends to a sliding window of recent tokens, a pinned prefix, and a fixed budget of most relevant tokens in the context. The window size and budget are found on a per-layer basis using a multi-objective search algorithm. Across six hybrid models from three families, from 4B to 397B parameters, the selected configurations preserve their baseline model's accuracy on 21 long-context datasets. Decoding 64k/256k tokens becomes up to 1.26x/2.22x faster on Qwen3.5-4B, 1.20x/2.23x on Qwen3.5-122B-A10B, and 1.1
    
[^41]: 培育自己的AI工具：实践生长型工具的范式

    Growing One's AI Tools: A Paradigm of Practice-Grown Tools

    [https://arxiv.org/abs/2611.06132](https://arxiv.org/abs/2611.06132)

    本文提出“实践生长型工具”新范式：一位无编程背景的会议口译员依据真实口译工作需求自行培育出计算机辅助口译工具，展现了工具与实践通过故障与修复共同演化的过程，并主张应将工具纳入对从业者的评估体系。

    

    长期以来，专业人士使用的都是他人为其构建的软件。本文报告了由用户本人培育的专业计算机辅助口译（CAI）工具的成长历程，该用户是一位未经编程训练的会议口译员。基于为期八周的版本控制记录以及来自133个记录会话和超过20,000条识别语句的遥测数据，本文追踪了真实的会议工作如何推动该工具经历62次提交和六个版本的发展。记录显示，新功能在其需求出现的当天即被部署，而一次策略调整在短短一个下午内将中位端到端延迟削减了超过三分之一。文章随后提出了“实践生长型工具”范式，其界定特征包括：从使用到培育的转变、工具与实践通过故障与修复实现的共同演化、有界本地计算，以及由此得出的评估推论，并提出了三个命题，以迈向将工具纳入考量的从业者评估。

    arXiv:2611.06132v1 Announce Type: cross  Abstract: Professionals have long used software built for them by others. This article reports the growth of a professional computer-assisted interpreting (CAI) tool built by its own user, a conference interpreter without programming training. Drawing on an eight-week version-controlled record with telemetry from 133 logged sessions and over 20,000 recognized utterances, the article traces how real conference work drove the tool through 62 commits and six versions. The record shows capabilities deployed on the day their need appeared and one strategy change that cut median end-to-end delay by over a third within one afternoon. The article then proposes the paradigm of practice-grown tools, defined by the shift from using to growing, the co-evolution of tool and practice through breakdown and repair, bounded local computation, and an assessment corollary, and it formulates three propositions toward tool-inclusive assessment of practitioners, from
    
[^42]: 框架即模型：智能体脚手架会改变固定大语言模型对银行行为规则的合规性，且在不同模型间影响不均

    The Harness Is the Model: Agent Scaffolding Changes a Fixed LLM's Compliance with Banking Conduct Rules, Unevenly Across Models

    [https://arxiv.org/abs/2611.06116](https://arxiv.org/abs/2611.06116)

    该研究表明，即使底层大语言模型保持不变，智能体框架（脚手架）的选择也会显著改变模型在银行行为合规任务上的表现，且不同模型受框架的影响程度并不相同。

    

    银行验证的是语言模型，但实际部署的是系统：即处于一个框架中的模型，该框架决定模型能看到什么、可以调用哪些工具，以及由谁来审查其行为。我们在保持模型固定不变的条件下，测试该框架是否会改变对银行行为规则的合规性。四个开源权重模型分别运行五种框架（一次性做出结构化决策的流水线、ReAct式工具循环、策略仅能通过搜索获取的循环、带有较长良性客户记忆的循环，以及带有同模型合规审查员的循环），并在一家模拟零售银行中对156个预注册案例进行测试，该银行的11类规则改写自欧盟消费者信贷、支付、投资者保护、数据保护及反洗钱相关条款。全部3120个两轮对话片段均基于实际执行的工具调用轨迹进行确定性评分，其中违规行为与过度拒绝被分别统计。汇总各模型来看，该框架使正确案例率……

    arXiv:2611.06116v1 Announce Type: cross  Abstract: Banks validate language models, but they deploy systems: a model inside a harness that decides what it sees, which tools it can call and who checks its actions. We test whether that harness changes compliance with banking conduct rules when the model is held fixed. Four open-weight models each ran five harnesses (a pipeline that makes one structured decision, a ReAct-style tool loop, a loop in which the policy is reachable only through search, a loop with a long benign customer memory, and a loop with a same-model compliance reviewer) on 156 pre-registered cases in a simulated retail bank whose 11 rule families paraphrase EU consumer-credit, payments, investor-protection, data-protection and anti-money-laundering provisions. Every one of the 3120 two-turn episodes was scored deterministically from the executed tool trace, with violations and over-refusals counted separately. Pooled across models the harness moved the correct-case rate 
    
[^43]: 调和隐私与溯源：面向语言模型水印的平衡之道

    Reconciling Privacy with Provenance for Language Model Watermarks

    [https://arxiv.org/abs/2611.06100](https://arxiv.org/abs/2611.06100)

    提出红绿序列（RGS）本地差分隐私水印，通过将水印目标从追踪个体转向估计总体使用频率，在可证明地保护用户隐私的同时，仍能实现对语言模型使用情况的准确估计。

    

    水印技术在规范语言模型（LM）的使用方面具有巨大潜力，但其部署一直饱受争议，因为许多用户认为对语言模型使用的细粒度追踪侵犯了他们的隐私。这种矛盾削弱了水印的有效性，因为用户会采取改写等规避策略，而这些策略可以去除大部分水印信号。我们证明，如果将水印设计的目标从追踪个体转变为估计总体使用情况，就可以调和溯源与隐私之间的矛盾。我们开发了本地差分隐私水印，可以从可证明的层面限制任何单个文本对检测所能贡献的证据，同时仍能实现对总体频率的准确估计。我们提出了红绿序列水印，这是一种具有隐私保护且计算高效的水印方法，能够在高隐私保护水平下，跨多种模型、样本量和生成场景实现准确的频率估计。

    arXiv:2611.06100v1 Announce Type: cross  Abstract: Watermarking has great potential for regulating the use of language models (LMs), but its deployment has been marred by controversy, as many users argue that fine-grained tracking of LM usage violates their privacy. This tension reduces the effectiveness of watermarks, as users engage in evasion strategies such as paraphrasing that can remove much of the watermark signal. We show that provenance and privacy can be reconciled if we shift the goal of watermark design from tracking individuals to estimating aggregate usage. We develop locally differentially private watermarks which provably limit the evidence that any individual text can contribute to detection, while still enabling accurate estimation of population frequencies. We introduce the Red-Green Sequence (RGS) watermark, a private and computationally efficient watermark that achieves accurate frequency estimation at high privacy levels across various models, sample sizes and gen
    
[^44]: NTCIR-19 SciClaimEval 任务概述：科学论文中的跨模态声明验证

    Overview of the NTCIR-19 SciClaimEval Task: Cross-modal Claim Verification in Scientific Papers

    [https://arxiv.org/abs/2611.06090](https://arxiv.org/abs/2611.06090)

    本文介绍了 NTCIR-19 的 SciClaimEval 试点任务，构建了一个将科学声明与论文表格及图形对比版本配对的跨模态验证数据集，吸引了 22 支团队参与，基于多模态大语言模型的最佳系统达到 98.1% 的配对准确率，但顶尖系统均依赖对比配对信息。

    

    本文概述了 SciClaimEval，这是 NTCIR-19 的一项试点任务，旨在根据科学论文中出现的表格和图形来验证科学声明。我们的数据集将来自机器学习、自然语言处理和医学领域 180 篇论文中的真实声明与其表格和图形的对比版本配对，其中一个版本支持该声明，另一个仅经过最小修改便不再支持。该任务包含两个子任务：判断一条证据是否支持某个声明，以及从两个证据版本中选出支持该声明的那一个。共有 22 支团队注册参与，是 NTCIR-19 各项任务中注册数量最多的，其中 14 支团队提交了 177 个参赛结果。大多数系统基于多模态大语言模型构建，依赖提示工程和多阶段推理，而非针对任务的微调。在第一个子任务上，最佳系统达到了 98.1% 的配对准确率，但所有六个排名最高的系统都依赖于对比配对；表现最好的……（摘要在此处截断）

    arXiv:2611.06090v1 Announce Type: new  Abstract: This paper presents an overview of \emph{SciClaimEval}, a pilot task at NTCIR-19 on verifying scientific claims against the tables and figures of the papers in which they appear. Our dataset pairs authentic claims from 180 papers in machine learning, natural language processing, and medicine with contrastive versions of their tables and figures, one supporting the claim and one minimally modified so that it does not. The task comprises two subtasks: deciding whether a piece of evidence supports a claim, and selecting which of two evidence versions does. Twenty-two teams registered, the highest number among NTCIR-19 tasks, and 14 submitted 177 runs. Most systems built on multimodal large language models, relying on prompting and multi-stage inference rather than task-specific fine-tuning. On the first subtask, the best systems reached 98.1\% pair accuracy, but all six top-ranked systems relied on the contrastive pairs; the best system tha
    
[^45]: ProcessPrefBench：评估自然人机交互中个体过程偏好的跨任务迁移

    ProcessPrefBench: Evaluating Cross-Task Transfer of Individual Process Preferences from Naturalistic Human-AI Interaction

    [https://arxiv.org/abs/2611.06065](https://arxiv.org/abs/2611.06065)

    提出 ProcessPrefBench 基准，基于两位知识工作者三个月的自然人机对话构建 66 个源-目标案例，评估个体过程偏好能否跨任务迁移以塑造 AI 助手后续回答，并发现完整历史与提炼偏好提示优于仅目标提示。

    

    在自然人机对话中，用户的纠正、请求和自我描述携带着关于他们希望工作如何完成的可复用信号。这些信号揭示了个体过程偏好，即用户在证据、深度、分析焦点、流程和交付方式方面的个性化标准。我们探讨在一段对话中显现的偏好，能否在后续任务未重申该偏好的情况下，塑造助手后续的回答。ProcessPrefBench 使用两位知识工作者三个月的自然对话，构建了 66 个源-目标案例，并对源证据和目标适用性分别进行评判。由于源案例经过预选，该基准测试的是偏好的应用而非检索。我们使用仅目标、完整历史和提炼偏好三种提示词测试了三个模型。针对目标的具体 0-2 分评分量表对偏好实现程度进行打分，并通过任务门槛标记严重失败。与仅目标提示相比，完整历史和指令式提炼偏好提示……

    arXiv:2611.06065v1 Announce Type: new  Abstract: In natural human-AI dialogue, users' corrections, requests, and self-descriptions carry reusable signals about how they want work to be done. These signals reveal individual process preferences, user-specific standards for evidence, depth, analytical focus, procedure, and delivery. We ask whether a preference revealed in one dialogue can shape an assistant's later answer when the task does not restate it. ProcessPrefBench uses three months of natural dialogues from two knowledge workers to build 66 source-target cases separately judged for source evidence and target applicability. Sources are preselected, so the benchmark tests application rather than retrieval. We test three models with target-only, full-history, and distilled-preference prompts. Target-specific 0-2 rubrics score preference realization, and a task gate flags severe failures. Compared with target-only prompting, full-history and imperative distilled-preference prompting 
    
[^46]: 将 Qwen3 适配到塔吉克语：从 0.6B 到 8B——更低的文本损失并不意味着更好的理解能力

    Adapting Qwen3 to Tajik from 0.6B to 8B: Lower Text Loss Did Not Mean Better Understanding

    [https://arxiv.org/abs/2611.06062](https://arxiv.org/abs/2611.06062)

    在塔吉克语上对 Qwen3（0.6B–8B）进行 LoRA 微调虽然降低了文本损失，却未能带来真正的阅读理解提升，只有约 3.75 亿 token 的大规模继续预训练加监督微调，才使主模型 e3 在 Belebele 上超越 Qwen3-8B 达 11.4 个百分点。

    

    在塔吉克语新闻和书籍上训练的小型 LoRA 适配器，在我们尝试的从 0.6B 到 8B 的所有模型规模上，都降低了 Qwen3 的塔吉克语文本损失，每 token 降低 0.60 至 1.45 nats，但在可测量的地方，没有一个适配器提升了塔吉克语阅读理解能力。在 Belebele 基准上，1.7B 和 8B 适配器的准确率没有变化，而 4B 适配器的准确率下降了 3.7 个百分点；在 0.6B 时分数上升了 4.2 个百分点，但该分数接近随机猜测水平，且恰好伴随模型所偏好答案字母的改变，因此我们不将其视为理解能力的提升。在 8B 规模上，对约 3.75 亿 token 的原始塔吉克语文本进行继续预训练后，Belebele 提升了 9.9 个百分点。在进一步的监督微调阶段之后，我们的主模型 e3 在 Belebele 上比 Qwen3-8B 高出 11.4 个百分点；其英语 MMLU 变化为 -1.2 个百分点，且可排除大于 3.3 个百分点的下降。将同样的高强度训练方案应用于 Qwen3-0.6B-Base 后，其文本损失接近仅用书籍预训练的 8B 模型的水平……（原文摘要到此截断）

    arXiv:2611.06062v1 Announce Type: new  Abstract: Small LoRA adapters trained on Tajik news and books lowered Qwen3's Tajik text loss at every size we tried, from 0.6B to 8B, by 0.60 to 1.45 nats per token, but none of them improved Tajik reading comprehension where it can be measured. On Belebele the 1.7B and 8B adapters left accuracy unchanged and the 4B adapter lowered it by 3.7 points; at 0.6B the score rose by 4.2 points but sits near chance and coincides with a change in which answer letter the model prefers, so we do not read it as understanding. At 8B, continued pretraining on about 375 million tokens of raw Tajik text was followed by a Belebele gain of 9.9 points. After a further supervised fine-tuning stage, our main model, e3, is 11.4 points above Qwen3-8B on Belebele; its English MMLU changed by -1.2 points, and a loss larger than 3.3 points is excluded. The same heavy recipe applied to Qwen3-0.6B-Base brought its text loss close to that of the 8B model after books-only pret
    
[^47]: SVGFaith：面向文本到SVG生成的忠实度度量的双层元评估

    SVGFaith: A Two-Layer Meta-Evaluation of Faithfulness Metrics for Text-to-SVG Generation

    [https://arxiv.org/abs/2611.06048](https://arxiv.org/abs/2611.06048)

    提出SVGFaith双层元评估框架，通过诊断性对比集与人工标注竞技场揭示现有图像-文本嵌入度量虽擅长判断物体身份、但在评估文本到SVG生成的空间关系与属性绑定时接近随机水平，并进一步提出可解释的SVGFaith-Score度量。

    

    文本到SVG系统通常通过将栅格图像度量应用于栅格化输出来进行评估，然而这些分数能否捕捉细粒度的提示词忠实度仍不清楚。我们提出了SVGFaith，一个双层元评估框架，其中每一对样本中更忠实的候选者由构造方式或人工判断确定。SVGFaith-Diagnostic包含1,842个覆盖八个能力维度的受控对比对，均匀分为两个方向：针对同一提示词比较两个渲染图像，以及针对同一渲染图像比较两个描述文本。SVGFaith-Arena包含1,327个对真实生成器输出的人工标注比较。在SVGFaith-Diagnostic上，CLIP及其他三个图像-文本嵌入度量在物体身份方面达到至少0.789的成对准确率，但在空间关系和属性绑定方面最高仅为0.672，而随机水平为0.5。我们进一步提出了SVGFaith-Score，一个可解释的逐点度量指标。

    arXiv:2611.06048v1 Announce Type: cross  Abstract: Text-to-SVG systems are commonly evaluated by applying raster-image metrics to rasterised outputs, yet whether these scores capture fine-grained prompt faithfulness remains unclear. We introduce SVGFaith, a two-layer meta-evaluation framework in which the more faithful candidate of each pair is determined by construction or by human judgment. SVGFaith-Diagnostic contains 1,842 controlled contrastive pairs across eight capability axes, split evenly between two directions: two renderings compared against one prompt, and two descriptions compared against one rendering. SVGFaith-Arena contains 1,327 human-annotated comparisons of real generator outputs. On SVGFaith-Diagnostic, CLIP and three other image--text embedding metrics reach at least 0.789 pairwise accuracy on object identity but at most 0.672 on spatial relations and attribute binding, where chance is 0.5. We further introduce SVGFaith-Score, an interpretable pointwise metric. A d
    
[^48]: Confucius4-R2T2：面向真·实时转录的稳定前缀流式语音识别模型

    Confucius4-R2T2: A Stable-Prefix Streaming ASR Model for Real Real-Time Transcription

    [https://arxiv.org/abs/2611.06043](https://arxiv.org/abs/2611.06043)

    本文提出Confucius4-R2T2，一个基于新概念“最长稳定前缀（LSP）”的稳定前缀流式语音识别模型，通过对Qwen3-ASR-1.7B进行后训练，使其在保持高准确率和低延迟的同时从不修改先前输出，非常适合需要即时行动的机器消费场景。

    

    流式语音识别越来越多地被机器所消费，例如翻译器和语音代理，它们会在每个词元（token）出现时立即对其采取行动。一个稍后被修改的词元会导致已做工作的浪费或错误的行动。在这项工作中，我们提出了Confucius4-R2T2（Real Real-Time Transcription的缩写，即“真·实时转录”），这是一个稳定前缀的流式ASR模型，它在保持高准确率和低延迟的同时，从不修改其先前的输出。R2T2是通过对Qwen3-ASR-1.7B在1,500小时的中文和英文音频上进行后训练获得的，训练期间冻结了其预训练的音频编码器和投影器。这种适配基于最长稳定前缀（Longest Stable Prefix, LSP）的概念，这是我们在本工作中引入的一个新概念。我们证明了LSP在前缀序下单调，与完整音频的输出一致，并且在等于之前完整输出的标签中是最大的。从LSP标签进行学习，可以将一个全上下文模型转换为稳定前缀的流式模型，而无需任何架构上的改动。

    arXiv:2611.06043v1 Announce Type: cross  Abstract: Streaming speech recognition is increasingly consumed by machines, such as translators and voice agents, that act on each token as it appears. A token that is later revised costs wasted work or a wrong action. In this work, we present Confucius4-R2T2 (short for Real Real-Time Transcription), a stable-prefix streaming ASR model that never modifies its earlier output while maintaining high accuracy and low latency. R2T2 is obtained by post-training Qwen3-ASR-1.7B on 1,500 hours of Chinese and English audio while freezing its pre-trained audio encoder and projector. The adaptation is based on the concept of Longest Stable Prefix (LSP), a new notion that we introduce in this work. We prove that LSP is monotone under the prefix order, consistent with the full-audio output, and maximal among labels that equal a whole earlier output. Learning from LSP labels converts a full-context model into a stable-prefix streaming one without any architec
    
[^49]: 当每个选择都正确时：语言模型中数值偏好的因果研究

    When Every Choice Is Correct: A Causal Study of Numerical Preference in Language Models

    [https://arxiv.org/abs/2611.06025](https://arxiv.org/abs/2611.06025)

    该研究以数值选择任务为载体，通过激活干预等因果分析手段追踪语言模型中数值偏好的形成过程，发现后续MLP层的计算在很大程度上决定了编辑对生成数字（如第二位数字）的持续影响。

    

    当语言模型在1到30之间选择一个数字时，该范围内的每个整数都是有效的，然而模型可能会偏爱特定的答案。我们利用数值选择来研究偏好是如何在不同层和不同词元之间被处理的。扰动探测和梯度归因方法确定了用于激活干预的候选神经元。在一个选定的Qwen3-4B案例中，答案生成前的编辑使输出从17变为15，而两次运行都首先输出1。在编辑结束后，选择性缓存和注意力输出替换追踪到了该编辑对第二位数字的部分持续影响。替换单个注意力头的输出或联合替换其他注意力头的输出会产生相似的即时读出贡献；两种干预都能恢复出17，但对7与5之间的最终对数几率产生不同的影响。在没有任何替换的情况下，将后续多层感知机（MLP）的输出保持为编辑运行时的值，可将这种效应差距从0.119 nat缩小到0.0025 nat。因此，后续计算改变……

    arXiv:2611.06025v1 Announce Type: new  Abstract: When a language model chooses a number between 1 and 30, every integer in the range is valid, yet the model can favor particular answers. We use numerical choice to study how preference is processed across layers and tokens. Perturbation probing and gradient attribution nominate neurons for activation interventions. In a selected Qwen3-4B case, a pre-answer edit changes 17 to 15 while both runs first emit 1. After the edit ends, selective cache and attention-output replacements trace part of its continuing influence on the second digit. Replacing one attention head's output or the other heads' outputs jointly produces similar immediate readout contributions; both interventions restore 17 but have different final log-odds effects on 7 versus 5. Holding later multilayer perceptron (MLP) outputs at their values from the edited run without either replacement reduces this effect gap from 0.119 to 0.0025 nat. Later computation therefore change
    
[^50]: 谁为模型复制的知识买单？作为罗杰斯悖论的模型坍塌

    Who Pays for the Knowledge Models Copy? Model Collapse as Rogers' Paradox

    [https://arxiv.org/abs/2611.06022](https://arxiv.org/abs/2611.06022)

    该论文将模型坍塌重新解读为文化演化中的罗杰斯悖论，指出问题的根源在于外部信息供给本身，并证明仅用完全合成数据拟合的罗杰斯模型能精确预测语言模型在不同新鲜数据混合比例下稀有知识的保留量。

    

    在先前模型输出上训练的生成模型会在代际间丢失知识，且稀有知识最先丢失。行之有效的补救措施都需要从循环外部引入信息，而现有的模型坍塌理论将这些信息供给视为既定条件：一个来自固定分布的固定档案库。我们认为供给本身才是问题的根源，而文化演化早已拥有关于这一问题的理论。我们将模型坍塌解读为罗杰斯悖论：一个逐渐依赖复制的人口最终并不比从不复制的人口境况更好，因为复制品会过时，除非有人持续进行自主学习。我们通过三个步骤检验这一论证。第一，仅用完全合成训练数据拟合的罗杰斯模型，能够以几个百分点以内的误差，预测语言模型——从词三元组模型到预训练Transformer——在其他新鲜数据混合比例下保留多少稀有知识。第二，当世界本身发生变化时，复制自身过往的谱系……（原文摘要在此处截断）

    arXiv:2611.06022v1 Announce Type: cross  Abstract: Generative models trained on the output of earlier models lose knowledge across generations, rare knowledge first. The remedies that work all bring in information from outside the loop, and the theory of model collapse treats that supply as given: a fixed archive, drawn from a fixed distribution. We argue that the supply is the problem, and that cultural evolution has the theory of it. We read model collapse as Rogers' paradox: a population that comes to rely on copying ends up no better off than one that never copied, because copies go stale unless someone keeps learning for themselves. We test the identification in three steps. First, Rogers' model, fitted only to fully synthetic training, predicts to within a few percentage points how much rare knowledge language models, from word trigrams to a pretrained transformer, retain under other mixtures of fresh data. Second, when the world itself changes, a lineage that copies its own past
    
[^51]: 面向富有表现力的语音到语音翻译的整体并行监督

    Holistic Parallel Supervision for Expressive Speech to Speech Translation

    [https://arxiv.org/abs/2611.06010](https://arxiv.org/abs/2611.06010)

    本文提出结合专业配音与多语言语音转换的整体并行监督训练数据构建策略，并开发了直接从源语音预测完整目标语音表示的端到端DirectS2ST系统，显著提升了语音到语音翻译的表现力。

    

    表现力是语音到语音翻译（S2ST）的核心，然而现有数据集很少能在内容、韵律、时序和说话人身份方面对源语音和目标语音进行联合对齐。因此，当前系统通常依赖间接的条件控制或迁移机制。为获得这种监督信号，我们提出了一种训练数据构建策略，将专业配音（在内容、韵律和时序上提供人工完成的对应关系）与多语言语音转换相结合，以减少说话人身份的不匹配。我们使用现有的表现性S2ST系统以及本文开发的端到端方法DirectS2ST来评估这种监督，该方法直接从源语音预测完整的目标语音表示。在英-德和英-西语言对上的实验表明，整体并行监督在所有被评估系统中均改善了表现力的实现，其中DirectS2ST在内容和韵律方面取得了最大的提升。

    arXiv:2611.06010v1 Announce Type: new  Abstract: Expressiveness is central to speech-to-speech translation (S2ST), yet existing datasets rarely align source and target speech jointly in content, prosody, timing, and speaker identity. Consequently, current systems often rely on indirect conditioning or transfer mechanisms. To obtain this supervision, we introduce a training-data construction strategy that combines professional dubbing, which provides human-performed correspondence in content, prosody, and timing, with multilingual voice conversion to reduce speaker mismatch. We evaluate this supervision using existing expressive S2ST systems and DirectS2ST, an end-to-end method developed in this work that predicts the complete target-speech representation directly from source speech. Experiments on English-German and English-Spanish show that holistic parallel supervision improves expressive realization across all evaluated systems. DirectS2ST exhibits the largest gains in content, pros
    
[^52]: 熟悉的字母，而非相关的语言，让未见过的语言进入共享的意义空间

    Familiar letters, not related languages, let unseen languages into a shared space of meaning

    [https://arxiv.org/abs/2611.06006](https://arxiv.org/abs/2611.06006)

    研究发现，决定从未见过的语言能否进入多语言模型共享语义空间的关键是“熟悉的文字系统”而非“相关的语言”——将没有翻译数据的语言转写成模型已熟悉的字母（如天城文），即可使其进入共享的意义空间。

    

    数百种语言拥有书面文本却几乎没有翻译，而模型在从未翻译的文本中学习一种语言时，究竟是依托于相关语言还是熟悉的字母，目前尚不清楚：因为这两者通常同时出现。在此，我们证明了是熟悉的字母而非相关语言，让一种未见过的语言进入了共享的意义空间。GlyphMind 是一个无需分词器的视觉 Transformer，它将文本作为像素读取，并允许任意句子以任何文字系统呈现，其预训练目标是跨 27 种书写系统对齐翻译。在它从未见过的语言和文字——老挝语的未翻译文本上进行适配后，其检索英语翻译的成功率达到了最强简单基线的七倍。然而，当老挝语被改写成不熟悉的字母时，即使其近亲泰语已包含在预训练中，老挝语也未能进入共享空间；而桑塔利语在预训练中没有任何近亲语言，却借助印地语中熟悉的天城文进入了共享空间，用其自身文字则不行。因此，将文本转写为熟悉的书写系统，或许能让多语言模型向那些……（原文在此处截断）的语言开放。

    arXiv:2611.06006v1 Announce Type: new  Abstract: Hundreds of languages have written text but almost no translations, and whether a model learning one from untranslated text builds on a related language or familiar letters is unclear: they usually coincide. Here we show that familiar letters, not a related language, let an unseen language into a shared space of meaning. GlyphMind, a tokenizer-free vision transformer that reads text as pixels and lets any sentence be shown in any script, was pretrained to align translations across 27 writing systems. Adapted on untranslated Lao, a language and script it had never seen, it retrieved English translations seven times as often as the strongest simple baseline. Rewritten in unfamiliar letters, Lao did not enter although its close relative Thai was in pretraining; Santali, which had none, entered in Devanagari, familiar from Hindi, but not in its own script. Transliteration into familiar scripts may thus open multilingual models to languages w
    
[^53]: 你能引导多远？对比激活添加中的行为-能力前沿

    How Far Can You Steer? Behavior-Capability Frontiers in Contrastive Activation Addition

    [https://arxiv.org/abs/2611.06001](https://arxiv.org/abs/2611.06001)

    该论文提出“行为-能力前沿”框架来评估对比激活添加（CAA）的可用性，在能力退化明确预算约束下最大化行为收益，发现行为效果具有强泛化性（ρ=0.945），且将能力损失预算从1%提高到3%可使干预的联合存活率从37.5%显著提升至72.7%。

    

    激活引导可以在不更新参数的情况下修改语言模型的行为，但仅凭行为效果并不能决定一项干预措施是否可用。对比激活添加（CAA）利用从对比样本中提取的激活方向来引导目标行为，但在此过程中可能会损害模型的其他无关能力。因此，我们通过行为-能力前沿来研究CAA，在明确的能力退化界限约束下，选择能够最大化行为收益的干预措施。我们在四个指令微调模型、七种行为以及1%、3%和5%（绝对百分点）的能力损失预算下对该框架进行了评估。结果显示，基于验证集筛选出的行为效果能够很好地泛化到留出数据上（ρ = 0.945）；而在非零工作点中，行为与能力同时达标的联合存活率从1%预算下的37.5%上升到3%预算下的72.7%，在5%预算下的额外改进则较为有限。

    arXiv:2611.06001v1 Announce Type: cross  Abstract: Activation steering can modify language-model behavior without parameter updates, but behavioral efficacy alone does not determine whether an intervention is usable. Contrastive Activation Addition (CAA) steers targeted behaviors using activation directions derived from contrasting examples, potentially degrading unrelated model capabilities in the process. We therefore study CAA through a behavior-capability frontier, selecting interventions that maximize behavioral gain subject to an explicit bound on capability degradation. We evaluate this framework across four instruction-tuned models, seven behaviors, and capability-loss budgets of 1%, 3%, and 5% (absolute percentage points). Validation-selected behavioral effects generalize strongly to held-out data ($\rho = 0.945$), while joint behavior-capability survival among nonzero operating points rises from 37.5% at the 1% budget to 72.7% at 3%, with limited additional improvement at 5%.
    
[^54]: 关于超连接的设计：简化、稳定性与可扩展性

    On the Design of Hyper-Connections: Simplification, Stability, and Scalability

    [https://arxiv.org/abs/2611.05984](https://arxiv.org/abs/2611.05984)

    该论文系统剖析了超连接（HC）架构，发现门控与残差混合可解耦为两个各自独立有效的交互组，并据此提出简化架构SimpleHC，通过移除残差混合和共享内存访问机制，在保持与mHC相当性能的同时显著降低训练开销。

    

    超连接（Hyper-Connections, HC）架构正越来越多地被前沿语言模型所采用，然而其组件仍未得到充分理解，并且会引入不可忽视的训练开销。在这项工作中，我们系统性地研究了HC架构，以理解其组件并找到更简单、更高效的设计。我们发现，不带混合的可学习读写门控与带固定门控的无约束混合构成了两个有效的交互组：单独使用其中任何一个即可达到与mHC相当的性能。我们还发现，对扩展残差状态的重复访问是训练开销的主要来源，即使在没有混合的情况下也是如此。这些发现催生了SimpleHC，它通过移除残差混合来简化架构，并通过共享内存访问来简化执行，利用由此产生的加性结构和共享门控输入。特定于层的门控共享一个组入口状态，从而支持批量读取和延迟写入……

    arXiv:2611.05984v1 Announce Type: cross  Abstract: Hyper-Connections (HC) architectures are increasingly adopted in frontier language models, yet their components remain insufficiently understood and introduce non-negligible training overhead. In this work, we systematically study the HC architecture to understand its components and identify simpler, more efficient designs. We find that learned read/write gating without mixing and unconstrained mixing with fixed gates form two effective interaction groups: either alone achieves performance comparable to mHC. We also identify repeated access to the expanded residual state as a major source of training overhead, even without mixing. These findings lead to SimpleHC, which simplifies the architecture by removing residual mixing and simplifies execution through shared memory access, using the resulting additive structure and shared gate inputs. Layer-specific gates share a group-entry state, enabling batched reads and deferred writes; scala
    
[^55]: 城市中的机器人：基于人物角色的城市机器人感知

    Robots in the City: Persona-Based Perceptions of Urban Robots

    [https://arxiv.org/abs/2611.05954](https://arxiv.org/abs/2611.05954)

    该研究通过为多模态大语言模型设定500个合成人物角色来评估首尔街道图像的安全感知，发现模型与人类均偏好无机器人的街道、更接受配送机器人而非人形机器人，且机器人对安全感知的影响随时间段和人群年龄而变化。

    

    行人越来越多地在不与之互动的情况下在城市公共空间中遇到机器人，但机器人如何改变街道的安全感知仍不清楚。我们研究了机器人的存在及其类型如何影响不同城市情境下的安全感知。我们生成了18张首尔街道图像，这些图像在时间段、道路等级和机器人条件（无机器人、配送机器人或人形机器人）上各不相同。两个多模态大语言模型（MLLM），Qwen3.8-27B 和 Gemma4-26B，各自被设定了500个合成人物角色，在每一对图像中选择更安全的一张，同时25名人类受访者完成相同的任务。两个模型都更偏好没有机器人的街道，并且相比人形机器人更偏好配送机器人。机器人在夜间被选择的频率更高，年长的人物角色选择机器人的频率更低，而道路等级的影响较小。在模型的解释中，配送机器人被视为一种物理障碍物，而人形机器人则被视为威胁或保护的来源。

    arXiv:2611.05954v1 Announce Type: cross  Abstract: Pedestrians increasingly encounter robots in urban public spaces without interacting with them, but how robots change the perceived safety of a street remains unclear. We examine how robot presence and type affect perceived safety across urban contexts. We generate 18 images of Seoul streets that vary in time of day, road class, and robot condition (no robot, a delivery robot, or a humanoid). Two multimodal large language models (MLLMs), Qwen3.8-27B and Gemma4-26B, each conditioned on 500 synthetic personas, choose the safer image in each pair, and 25 human respondents complete the same task. Both models prefer streets without robots and favor the delivery robot over the humanoid. Robots are chosen more often at night and less often by older personas, whereas road class has smaller effects. In the model rationales, the delivery robot appears as a physical obstacle, whereas the humanoid appears as either a threat or a source of protecti
    
[^56]: 面向基于混合专家的多模态大语言模型的模态感知专家剪枝

    Modality-Aware Expert Pruning for MoE-Based Multimodal Large Language Models

    [https://arxiv.org/abs/2611.05927](https://arxiv.org/abs/2611.05927)

    提出了MAEP，一个无需训练的模态感知专家剪枝框架，通过ΔH输出度量和低注意力视觉token信号按模态分解专家重要性，实现对MoE多模态大语言模型的有效压缩。

    

    摘要：混合专家使多模态大语言模型（MLLM）能够高效扩展模型容量，但尽管每个token仅激活少数专家，专家参数却占据了绝大部分内存：在现代MoE-MLLM中，专家参数占模型规模的90%以上（例如，InternVL3.5-30B-A3B和Qwen3-VL-30B-A3B在BF16精度下约58GB的模型中专家参数占54GB）。专家剪枝已成为一种有效的压缩策略，但现有方法在token间均匀聚合专家重要性，忽略了同一个专家在视觉处理与语言处理中可能扮演的不同角色。为解决这一问题，我们提出了MAEP（模态感知专家剪枝），这是一个无需训练的框架，可在单次前向传播中按模态分解专家重要性。MAEP引入了：(1) Delta-H输出（ΔH），一种重分布感知的度量指标，用于衡量移除某个专家后MoE输出的变化；(2) 低注意力视觉token作为视觉侧的剪枝信号

    arXiv:2611.05927v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) enables multimodal large language models (MLLMs) to scale capacity efficiently, but expert parameters dominate memory even though only a few experts are activated per token: they account for over 90% of model size in modern MoE-MLLMs (e.g., 54 GB out of about 58 GB in BF16 for InternVL3.5-30B-A3B and Qwen3-VL-30B-A3B). Expert pruning has emerged as an effective compression strategy, but existing methods aggregate importance uniformly across tokens, ignoring that the same expert may serve different roles for vision versus language processing. To address this, we propose MAEP (Modality-Aware Expert Pruning), a training-free framework that decomposes expert importance by modality in a single forward pass. MAEP introduces (1) Delta-H Output ($\Delta H$), a redistribution-aware metric measuring the MoE output change when an expert is removed, and (2) low-attention vision tokens as a vision-side pruning signal that i
    
[^57]: 推理之路：读取与重塑MoE模型中的测试时计算

    Routes of Reasoning: Reading and Reshaping Test-Time Compute in MoE Models

    [https://arxiv.org/abs/2611.05919](https://arxiv.org/abs/2611.05919)

    该论文发现MoE模型的router logits可以作为天然的、无需训练的行为控制接口，通过六类推理行为的标注与探针读取，实现对“过度思考”等特定推理行为的精准调控，从而节约测试时计算并提升性能。

    

    推理模型会将测试时计算分配给不同的行为，例如规划、推导和验证。然而，模型经常“过度思考”，在找到正确答案后很长时间仍继续验证和重新推导。这不仅浪费了昂贵的计算资源，甚至可能降低性能。现有的干预方法，如token预算或引导，主要调节推理时长，而非针对特定行为。在这项工作中，我们证明了混合专家模型的router提供了一种天然的、无需训练的行为控制接口。我们首先将推理轨迹切分为句子级片段，并用六种推理行为（规划、执行、复述、事实、反思和结论）之一对每个片段进行标注，使每个token都携带其所属片段的行为标签。作用于router logits上的线性探针预测该标签的效果，几乎与作用于残差流（即模型完整隐藏状态这一标准目标）上的探针相当。

    arXiv:2611.05919v1 Announce Type: new  Abstract: Reasoning models allocate test-time compute across distinct behaviors such as planning, deriving, and verifying. However, models frequently "overthink," continuing to verify and re-derive long after finding a correct answer. This wastes expensive compute and can even degrade performance. Existing interventions, such as token budgets or steering, primarily regulate reasoning duration rather than targeting specific behaviors. In this work, we demonstrate that Mixture-of-Experts (MoE) routers provide a natural, training-free interface for behavioral control. We first split reasoning traces into sentence-level segments and label each with one of six reasoning behaviors (planning, execution, restatement, fact, reflection, and conclusion), so that every token carries the behavior of its segment. Linear probes on router logits predict this label nearly as well as probes on the residual stream, the model's full hidden state and the standard targ
    
[^58]: 面向电力系统规划的自主智能体

    Autonomous agents for electricity-system planning

    [https://arxiv.org/abs/2611.05911](https://arxiv.org/abs/2611.05911)

    该论文提出了一个用于电力系统规划的自主智能体框架，能够自动完成从方案提案、数值评估、反馈解读到重新设计和最终方案选择的全流程决策，并自动保留成本最低的可行规划方案。

    

    规划电力系统需要反复进行将区域资源、投资、运行和网络约束联系起来的决策。在此，我们提出了一个自主规划框架，该框架将准备好的区域情景依次推进至提案、数值评估、反馈解读、重新设计和最终方案选择环节。智能体提出投资边界、储能运行条件和馈线干预措施，解读数值反馈，并决定如何修改设计以及何时停止。该框架自动执行年度投资与调度优化以及代表性馈线验证，应用固定的验收标准，并保留成本最低的通过验收的候选方案。我们在五个电力系统、四个国家中，使用两种模型部署以及单智能体和多智能体架构，评估了跨越十个系统年情景的四十条碳约束轨迹。全部四十次运行均保留了满足实现……的候选方案（摘要在此处截断）。

    arXiv:2611.05911v1 Announce Type: new  Abstract: Planning electricity systems requires repeated decisions that connect regional resources, investment, operation and network constraints. Here we present an autonomous planning framework that carries prepared regional scenarios through proposal, numerical evaluation, feedback interpretation, redesign and final plan selection. Agents propose investment bounds, storage operating conditions and feeder interventions, interpret numerical feedback, and decide how to revise designs and when to stop. The framework automatically executes annual investment-and-dispatch optimization and representative-feeder verification, applies fixed acceptance criteria and retains the lowest-cost accepted candidate. We evaluate forty carbon-constrained trajectories across ten system-year scenarios in five electricity systems and four countries, using two model deployments and single- and multi-agent architectures. All forty retain candidates satisfying the implem
    
[^59]: 对话中的选择性、风险可控情绪预测：从持续性屏障到带共形保证的偏离检测

    Selective, Risk-Controlled Emotion Forecasting in Conversation: From a Persistence Barrier to Departure Detection with Conformal Guarantees

    [https://arxiv.org/abs/2611.05896](https://arxiv.org/abs/2611.05896)

    该论文发现对话情绪预测被“重复上一状态”的持续性基线所主导且难以改进，因而将任务重构为对情绪偏离持续性时机的选择性检测，并提出提前期加权指标、随偏离率自适应切换的承诺策略以及带共形保证的风险控制。

    

    对话中的情绪预测日益被界定为预测下一情绪状态的数值。我们表明，在一个状态可观测的测试环境中，这是一个难以逼近的目标，并提出了一种可学习、可部署的任务重构方案。在一个双人对话语料库的效价-唤醒度-支配度轨迹上，重复说话者自身前一个状态构成了一个“持续性”基线，任何冻结编码器的修正都无法对其产生实质性改进，而状态空间滤波器则可以超过它；该下限的存在依赖于历史状态可被观测，因为若使用估计的锚点，持续性基线会跌至低于直接预测器的水平。真正可存活下来的是对偏离持续性时机的检测：该任务在四个语料库中的三个上可被学习，而在基率最高的语料库上接近随机水平，我们如实报告了这一负面结果。我们将该任务重构为选择性偏离预测：提出一种带泄漏防控协议的提前期加权评估指标，以及一组承诺策略——其最优选择会随偏离率的变化而切换。[原摘要在此处被截断]

    arXiv:2611.05896v1 Announce Type: new  Abstract: Forecasting emotion in conversation is increasingly framed as predicting the next emotional state's value. We show this targets a hard quantity in a state-observed testbed and propose a learnable, deployable reformulation instead. On valence-arousal-dominance trajectories from a dyadic corpus, repeating a speaker's own previous state sets a persistence baseline that no frozen-encoder correction meaningfully improves upon and a state-space filter exceeds; the floor requires the historical state to be observed, since an estimated anchor collapses persistence below a direct forecaster. What survives is the timing of a departure from persistence, learnable on three of four corpora and near chance on the highest-base-rate corpus, a negative result we report. We recast the task as selective departure forecasting: a lead-time-weighted metric with a leakage protocol, and commitment policies whose best choice switches with the departure rate. A d
    
[^60]: 对话情绪转变预测的信息匹配审计：一种防泄漏的评估协议

    Information-Matched Auditing of Conversational Emotion-Shift Forecasting: A Leakage-Safe Evaluation Protocol

    [https://arxiv.org/abs/2611.05895](https://arxiv.org/abs/2611.05895)

    该论文提出了一种信息匹配、防泄漏的对话情绪转变预测审计协议，通过明确的承诺点与可执行的因果性测试，揭示了可部署条件与先知式诊断条件会得出不同的评估结论。

    

    arXiv:2611.05895v1 公告类型：新论文 摘要：在对话情绪转变预测研究中，当学习型模型获得丰富的特征，而其转移基线却被给予含噪声或不可获得的当前情绪标签时，当超参数预算不相同时，或当目标轮次信息混入表示中时，比较结果很容易被扭曲。我们提出了一种信息匹配且防泄漏的审计协议，通过明确的承诺点、可执行的因果性测试、均等的四配置验证搜索以及感知种子对话的推理来消除这些差距。将该协议应用于四个文本语料库和两种特征源敏感性配置上的六个因果模型族，结果表明：当为转移基线提供一个仅基于训练数据且可部署的当前情绪分类器时，所得到的评估结论与双方均获得金标准当前情绪的先知式诊断所得出的结论不同。在可部署条件下，学习型模型在全部四个主要语料库上均优于预测标签基线（摘要在此处被截断）。

    arXiv:2611.05895v1 Announce Type: new  Abstract: Comparisons in conversational emotion-shift forecasting are easily distorted when a learned model receives rich features but its transition baseline is given a noisy or unavailable current-emotion label, when hyperparameter budgets differ, or when target-turn information enters the representation. We present an information-matched, leakage-safe auditing protocol that closes these gaps with an explicit commitment point, executable causality tests, equal four-configuration validation searches, and seed-dialogue-aware inference. Applying it to six causal model families on four text corpora and two feature-source sensitivity configurations shows that a deployable train-only current-emotion classifier feeding the transition baseline yields a different verdict than an oracle diagnostic where both sides receive gold current emotion. Under the deployable condition, learned models improve over the predicted-label baseline on all four primary corp
    
[^61]: 何时更好的置信度才能改善知识图谱审计？

    When Does Better Confidence Improve a Knowledge-Graph Audit?

    [https://arxiv.org/abs/2611.05877](https://arxiv.org/abs/2611.05877)

    该论文通过将非自适应审计建模为预算化覆盖问题，从理论上刻画了更好的置信度何时才能真正提升知识图谱审计效果，并通过结构质量 $\Delta^{\mathrm{cl}}$ 界定规划遗憾，实验证明联合模型规划器比错误率×扇出队列少留下 3.98–4.74 个百分点的错误。

    

    置信度估计在由大语言模型构建的知识图谱中指导人工审核，但更好的错误预测在审计中可能只带来微小的改进。我们通过可修复性、选择余量和标签有效性来研究这一差距。将非自适应审计建模为预算约束下的覆盖问题，我们刻画了最小化的决策相关错误统计量，并用结构质量 $\Delta^{\mathrm{cl}}$ 界定了保持标量的一阶规划的遗憾。我们在 Re-DocRED 上针对五个抽取配置和两个跨文档图谱，基于与基准对齐的标签模拟理想修复，随后在测试集上进行冻结确认，并在 500 个哈希选取的训练文档上进行复现实验。在留出的跨文档图谱上，错误率×扇出队列在相同动作下比联合模型多留下 3.98–4.74 个百分点的错误。相比之下，为该联合规划器开放共享动作可将残余错误减少 0.89–1.43 个百分点。

    arXiv:2611.05877v1 Announce Type: new  Abstract: Confidence estimates guide human review in LLM-built knowledge graphs, but better error prediction can yield little improvement in an audit. We study this gap through repairability, selection headroom and label validity. Modelling non-adaptive auditing as budgeted covering, we characterise the minimal decision-relevant error statistics and bound the regret of scalar-preserving first-order planning by a structural mass $\Delta^{\mathrm{cl}}$. We simulate ideal repair against benchmark-aligned labels for five extraction configurations and two cross-document graphs on Re-DocRED, followed by frozen confirmation on test and replication on 500 hash-selected training documents. On the held-out cross-document graphs, an error-rate $\times$ fan-out queue leaves 3.98-4.74 percentage points more errors than a joint model with the same actions. By comparison, opening shared actions for that joint planner reduces residual errors by 0.89-1.43 points r
    
[^62]: CARTBench：基于评分准则的购物建议评估——面向包含时间线的复杂与模糊请求

    CARTBench: Rubric-Based Evaluation of Shopping Advice for Complex and Ambiguous Requests with Timelines

    [https://arxiv.org/abs/2611.05862](https://arxiv.org/abs/2611.05862)

    提出了CARTBench基准，基于真实商业购物订单提炼的评分准则，系统评估LLM购物助手处理包含时间线的复杂与模糊购物请求的能力。

    

    大型语言模型（LLM）购物助手如今已能在多轮对话中推荐商品。然而，对其进行可靠评估面临两大难题：数据稀缺与评判标准难以统一。为解决这些障碍，我们推出了针对CART（包含时间线的复杂与模糊请求）的CARTBench基准。针对数据稀缺问题，我们从17个零售领域的真实商业购物需求中精选案例，将带时间戳的对话与1,603条商品记录相配对。针对评判标准难以统一的问题，我们从成功完成订单中人工挑选出的优秀回复里提炼出“理想回复原则”，并以该原则为指导开发了针对具体案例的评分准则。这些原则同样指导了我们的结构化推荐策略——原则引导CART提示。该实验性基准包含1,328条经专家标注的准则，涵盖核心、卓越与安全三个维度。为解答CART问题，我们评估了7个旗舰商用API模型，包括GPT-6.1-Sol等（原文截断）。

    arXiv:2611.05862v1 Announce Type: new  Abstract: Large language model (LLM) shopping assistants now recommend products in multi-turn conversations. However, reliable evaluation faces Data scarcity and Difficulty Standardizing Criteria. To address these obstacles, we introduce CARTBench for CART (Complex and Ambiguous Requests with Timelines). For Data scarcity, we curate cases from real commercial shopping needs across 17 retail domains, pairing timestamped dialogue with 1,603 product records. For Difficulty Standardizing Criteria, we develop case-specific rubrics guided by Ideal Response Principles, distilled from excellent replies manually selected from successfully completed orders. These principles also guide Principle-Guided CART Prompting, our structured recommendation strategy. The experimental benchmark contains 1,328 expert-annotated criteria across Core, Excellence, and Safety. To answer the CART question, we evaluate 7 flagship commercial API models, including GPT-6.1-Sol an
    
[^63]: TandemBench：面向多智能体协作的决策模型基准测试

    TandemBench: Benchmarking Decision Models for Multi-Agent Collaboration

    [https://arxiv.org/abs/2611.05827](https://arxiv.org/abs/2611.05827)

    提出了TandemBench基准，通过1,200个多智能体烹饪协作问题系统评估决策模型的协作能力，发现原生决策模型最高仅达62.50%的决策准确率，且在验证方案最优性方面表现接近随机水平。

    

    决策模型在路由和动作选择等任务中从预定义选项中进行选择。在多智能体协作中，一个决策决定了成员如何分工、交换资源以及并行行动。我们提出了TandemBench，这是一个包含1,200个问题的基准测试，基于多智能体烹饪场景构建，其中协调选择会直接影响目标完成情况。模型接收完整的全局信息和候选团队计划，参考答案通过执行与精确搜索建立。模型需要选择最优团队计划，并在保留或移除该计划的成对变体中，判断在给定选项之外是否存在更优的安排。我们评估了40个原生决策模型，并将它们与10个通用语言模型进行比较。最佳原生决策模型达到62.50%的决策准确率。原生决策模型在最优性验证任务上的表现接近随机水平。成对测试显示，某些模型无论何种情况都会报告存在更好的替代方案。

    arXiv:2611.05827v1 Announce Type: new  Abstract: Decision models choose among predefined options in tasks such as routing and action selection. In multi-agent collaboration, a decision determines how members divide work, exchange resources, and act in parallel. We introduce TandemBench, a benchmark of 1,200 questions in multi-agent cooking scenarios where coordination choices affect goal completion. Models receive complete global information and candidate team plans, with reference answers established through execution and exact search. Models select an optimal team plan and, in paired variants that retain or remove it, judge whether better arrangements exist beyond the supplied options. We evaluate 40 native decision models and compare them with 10 general-purpose language models. The best native decision model achieves 62.50% decision accuracy. Native decision models remain close to chance in optimality verification. Paired tests reveal that some report better alternatives regardless
    
[^64]: 学习何时人类决策至关重要：面向大语言模型智能体的执行者相对咨询价值

    Learning When Human Decisions Matter: Executor-Relative Consultation Value for LLM Agents

    [https://arxiv.org/abs/2611.05814](https://arxiv.org/abs/2611.05814)

    该论文提出咨询价值蒸馏方法，通过蒙特卡洛执行者模拟离线估计依赖执行者的人类咨询价值并蒸馏为轻量级预测器，使LLM智能体能够在有限人类决策预算下智能地决定何时咨询人类。

    

    大语言模型日益自动化复杂任务，但人类的决策能力并不会随着模型能力的提升而扩展。在许多应用中，人类在提供私有背景信息、权威性或智能体无法获取的判断方面仍然不可或缺。我们提出了人类决策能力分配这一形式化问题：在存在多个咨询机会且预算有限的情况下，大语言模型智能体应该在哪些决策上咨询人类？其核心挑战在于估计咨询价值，而咨询价值依赖于执行者，因为同样的人类输入可能对不同的下游智能体产生不同的收益。我们提出了咨询价值蒸馏方法，该方法利用蒙特卡洛执行者模拟离线估计咨询价值，并将这些价值蒸馏到轻量级预测器中用于预算受限的分配，在部署时无需进行基于模拟的价值估计。在五个批量设置和三个序贯设置以及两个前沿执行器上，CVD 提升了……

    arXiv:2611.05814v1 Announce Type: cross  Abstract: Large language models increasingly automate complex tasks, but human decision capacity does not scale with their capabilities. In many applications, humans remain essential for providing private context, authority, or judgments unavailable to the agent. We formulate human decision capacity allocation: given multiple consultation opportunities and a limited budget, at which decisions should an LLM agent consult a human? The central challenge is estimating consultation value, which is executor-dependent because the same human input may benefit different downstream agents differently. We introduce Consultation Value Distillation (CVD), which estimates consultation values offline using Monte Carlo executor rollouts and distills these values into lightweight predictors for budgeted allocation, requiring no rollout-based value estimation at deployment. Across five batch and three sequential settings with two frontier executors, CVD improves 
    
[^65]: 在推理时专家停用中匹配训练时的权重规则

    Matching the Training-Time Weight Rule in Inference-Time Expert Deactivation

    [https://arxiv.org/abs/2611.05804](https://arxiv.org/abs/2611.05804)

    该论文提出一个权重和模型，揭示推理时专家停用方法应与训练时的权重规则相匹配——使用原始路由概率训练的模型适合剩余专家重归一化，而使用归一化权重训练的模型则适合未归一化重新路由。

    

    推理时的专家停用会同时改变token路由和专家权重，其对性能的影响取决于这些变化的实现方式。我们在来自四个模型家族的五个指令微调检查点上，针对相同的禁用专家集合，比较了移除、冻结、剩余专家重归一化以及未归一化重新路由这几种方法，并以基础OLMoE检查点作为额外对比。我们开发了一个权重和模型，将专家贡献的损失以及分支规模的变化与训练时使用的权重规则联系起来。在根据基于激活的排名停用一半专家的设置下，对于使用原始路由概率训练的模型，剩余专家重归一化会产生最高的困惑度；而对于使用归一化权重训练的模型，未归一化重新路由会产生最高的困惑度。当保留训练时的规则时，困惑度范围为完好基线的1.4倍至73倍，而（原文在此处截断）

    arXiv:2611.05804v1 Announce Type: cross  Abstract: Expert deactivation at inference time changes both token routing and expert weights, and its effect on performance depends on how these changes are implemented. We compare removal, freezing, remaining-expert renormalization, and unnormalized re-routing on identical disabled sets across five instruction-tuned checkpoints from four model families, with a base OLMoE checkpoint as an additional comparison. We develop a weight-sum model that relates the loss of expert contributions and changes in branch scale to the weight rule used during training. With half of the experts deactivated according to activation-based rankings, remaining-expert renormalization yields the highest perplexity in models trained with raw router probabilities, whereas unnormalized re-routing yields the highest perplexity in models trained with normalized weights. Perplexity ranges from 1.4 to 73 times the intact baseline when the training rule is retained, and from 
    
[^66]: 语言模型对汉字的语音学知识

    Phonological Knowledge of Chinese Characters in Language Models

    [https://arxiv.org/abs/2611.05803](https://arxiv.org/abs/2611.05803)

    该论文提出PinyinBench基准，通过行为评估、逐层探测和激活引导三种方法系统研究17个大语言模型，发现模型对汉字声母韵母掌握扎实但在声调区分上表现薄弱，且声调信息在嵌入层早期即可解码而音段信息随网络深度逐渐形成。

    

    语言模型（LM）对汉字究竟掌握多少语音学知识？这些知识又是如何被表示和使用的？我们提出了PinyinBench，一个涵盖同音字、语音特征（声母、韵母和声调）以及多音字的受控基准测试。我们通过三种互补的方法研究了17个纯文本和多模态大语言模型：行为评估、逐层探测和激活引导。在行为层面，模型展现出对声母和韵母的丰富知识，一个前沿模型（DS-v4-flash）在两个字不共享任何语音特征时的同音字判断任务中得分超过90%。然而，当两个字仅在声调上不同时，其准确率骤降至72%。探测实验揭示，声调信息从词嵌入中即可高度解码，并随网络层数加深仅有小幅提升；而音段信息（声母和韵母）在浅层的可解码性较低，但随深度增加显著改善。激活引导实验表...

    arXiv:2611.05803v1 Announce Type: new  Abstract: How much phonological knowledge do language models (LMs) possess about Chinese characters, and how is this knowledge represented and used? We introduce PinyinBench, a controlled benchmark covering homophones, phonological features (initials, finals, and tones), and polyphonic characters. We investigate 17 text-only and multimodal LMs through three complementary approaches: behavioral evaluation, layer-wise probing, and activation steering. Behaviorally, models demonstrate substantial knowledge of initials and finals, with a frontier model (DS-v4-flash) scoring 90%+ in homophone judgment, when two characters share no phonological features. However, its accuracy drops sharply to 72% when two characters differ only in lexical tone. Probing reveals that tone is highly decodable from embeddings and improves modestly across layers, whereas segmental information starts less decodable and improves substantially with depth. Activation steering sh
    
[^67]: 大语言模型能在多大程度上模仿个体作者？一项比较评估

    How Well Can Large Language Models Impersonate Individual Authors? A Comparative Evaluation

    [https://arxiv.org/abs/2611.05771](https://arxiv.org/abs/2611.05771)

    本研究比较评估了六种主流大语言模型模仿个体作者写作风格的能力，发现大多数模型模仿效果有限，而表现最好的 Gemini 3 Pro 实际上是靠大量复制训练文本中的句子和短语取得的表面成效。

    

    本研究考察了六种广泛使用的大型语言模型（LLM）的作者身份模仿能力，包括 Gemini 3 Pro、Claude Opus、GPT 5.2、GPT 5.2 Pro、DeepSeek 3.2 和 Grok 4.1 Fast。评估采用精选的短篇产品评论文本，以确保个体写作风格在作者内部具有高度一致性。研究设计了两种类型的上下文提示：一种是指令最少的简单提示，另一种是包含明确文体学指导以辅助模仿的复杂提示。作者模仿能力通过基于余弦距离的“与训练文本距离”指标进行评估，并辅以嵌入分布的可视化分析。结果表明，对于大多数大语言模型而言，作者身份模仿仍然具有挑战性，其中 Gemini 3 Pro 展现出最强的表面模仿性能。然而，随后的重叠分析揭示，Gemini 3 Pro 之所以表现突出，是因为它大量复制并重用了训练文本中的句子和短语。

    arXiv:2611.05771v1 Announce Type: cross  Abstract: This study investigates the authorship impersonation capacity of six widely used large language models (LLMs): Gemini 3 Pro, Claude Opus, GPT 5.2, GPT 5.2 Pro, DeepSeek 3.2, and Grok 4.1 Fast. The evaluation was conducted using short product review texts selected to ensure high within author consistency in individual writing style. Two types of in context prompt were designed: a simple prompt with minimal instruction and a complex prompt incorporating explicit stylometric guidance for mimicry. Authorship mimicry was evaluated using a cosine distance based distance to training metric, supplemented by visual analyses of embedding distributions. The results indicate that authorship impersonation remained challenging for most LLMs, with Gemini 3 Pro showing the strongest apparent mimicry performance. However, subsequent overlap analysis revealed that Gemini 3 Pro extensively copied and reused sentences and phrases from the training texts, 
    
[^68]: PAYWORTHY：大语言模型能否预测AI智能体将完成哪些自由职业任务？

    PAYWORTHY: Can LLMs Predict Which Freelance Jobs an AI Agent Will Complete?

    [https://arxiv.org/abs/2611.05759](https://arxiv.org/abs/2611.05759)

    该论文提出PAYWORTHY基准，基于1,494条真实自由职业任务发现，大语言模型只有在被告知智能体的能力限制时才能可靠预测哪些任务会被AI智能体成功完成（AUC达0.66-0.86），而若仅将其描述为通用智能体，预测效果则接近随机水平。

    

    在AI智能体接手一份有偿工作之前，需要有人判断它是否可能成功。我们测试了前沿大语言模型能否在任何实际尝试之前做出这一判断。我们提出PAYWORTHY基准，其中包含1,494条真实的自由职业任务帖，这些任务从一个自由职业平台2020年至2026年间发布的1,880万条帖子中采样而来。智能体仅根据任务帖单独尝试完成每项工作（单次运行），并由一个充当客户角色的大语言模型裁判决定是否接受该工作。我们让18个模型预测智能体成功的可能性，并衡量每个模型的预测将被接受的任务排在被拒绝任务之上的能力（AUC指标，其中0.5表示随机排序）。结果表明，答案取决于如何描述该智能体：当告知模型智能体不能做什么时，所有模型都能很好地对任务进行排序（AUC为0.66-0.86）；而当仅告知它是一个具备浏览器和代码执行能力的通用智能体时，同样的模型对同样任务的排序效果几乎与随机排序无异（0.45-0.56）。

    arXiv:2611.05759v1 Announce Type: new  Abstract: Before an AI agent takes on a paid job, someone has to judge whether it is likely to succeed. We test whether frontier language models can make this call before any attempt. We propose PAYWORTHY which contains 1{,}494 real freelance job postings, sampled from 18.8M postings made between 2020 and 2026 on a freelancing platform. An agent attempts each job from the posting alone in a single run, and an LLM judge acting as the client decides whether to accept the work. We ask 18 models to forecast how likely the agent is to succeed and measure how well each model's forecasts rank the accepted jobs above the rejected ones (AUC, where 0.5 is a random ranking). The answer depends on how the agent is described. Told what the agent cannot do, every model ranks the jobs well (AUC 0.66--0.86). Told instead that it is a generic agent with a browser and code execution, the same models rank the same jobs about as well as a random ordering (0.45--0.56)
    
[^69]: 面向低资源医学自动语音识别的纯文本域自适应：豪萨语的浅融合与密度比重评分

    Text-Only Domain Adaptation for Low-Resource Medical ASR: Shallow Fusion and Density Ratio Rescoring for Hausa

    [https://arxiv.org/abs/2611.05744](https://arxiv.org/abs/2611.05744)

    该研究提出仅需文本数据（无需医学领域语音数据）即可对豪萨语医学ASR进行域自适应，通过浅融合和密度比重评分两种解码时语言模型融合方法，将词错误率从29.56%显著降至约23.3%–23.5%。

    

    自动语音识别（ASR）系统常常难以处理专业医学词汇，尤其是在缺乏领域特定语音数据的低资源语言中。我们研究了针对豪萨语医学ASR的纯文本域自适应方法，使用固定的MMS-1B-FL102声学模型以及两种解码时语言模型融合方法：浅融合（SF）和密度比方法（DRA）。基于19,421条豪萨语健康句子训练的4-gram KenLM作为医学目标模型，而由75,372条句子构成的通用领域模型为DRA提供源先验。评估使用了来自尼日利亚北部六个州的10名说话人针对25条经医生审阅的提示语录制的250条录音。在提示语不相交的五折交叉验证下，SF和DRA分别将词错误率（WER）从29.56%降至23.55%和23.33%。两者均显著优于贪婪解码（p < 0.001），而二者之间的差异不显著（p = 0.423）。然而，仅使用通用语言模型的SF……

    arXiv:2611.05744v1 Announce Type: cross  Abstract: Automatic speech recognition (ASR) systems often struggle with specialised medical vocabulary, especially in low-resource languages where domain-specific speech data are scarce. We investigate text-only domain adaptation for Hausa medical ASR using the fixed MMS-1B-FL102 acoustic model and two decoding-time language-model fusion methods: Shallow Fusion (SF) and the Density Ratio Approach (DRA). A 4-gram KenLM trained on 19,421 Hausa health sentences serves as the medical target model, while a 75,372-sentence general-domain model provides the source prior for DRA. Evaluation uses 250 recordings of 25 physician-reviewed prompts from 10 speakers across six northern Nigerian states. Under prompt-disjoint five-fold cross-validation, SF and DRA reduce WER from 29.56% to 23.55% and 23.33%, respectively. Both significantly outperform greedy decoding (p < 0.001), while their difference is not significant (p = 0.423). However, general-LM-only SF
    
[^70]: 面向主动式LLM智能体的前瞻记忆

    Prospective Memory for Proactive LLM Agents

    [https://arxiv.org/abs/2611.05731](https://arxiv.org/abs/2611.05731)

    提出受人类前瞻记忆启发的Pro-Memory框架，为主动式LLM智能体引入前瞻记忆，将维护未来意图与判断实现时机相分离，使智能体能在合适的时机自主兑现未来目标。

    

    大多数面向LLM智能体的记忆系统都是回顾性的，即保存和检索过去的信息以辅助当前的决策。然而，主动式智能体还需要前瞻记忆：保留面向未来的意图，并在合适的时机出现时将其实现，即使没有明确的请求。关键在于，记住正确的意图并不能保证在正确的时机行动：一个相关的观察可能需要等待，而稍后的线索可能使同一意图变得可以执行。受人类前瞻记忆认知理论的启发，我们提出了Pro-Memory，一个将维护未来意图与决定何时以及如何实现它们相分离的记忆框架。Pro-Memory保存待处理的目标并将其与支持证据相关联，依据原始目标和激活条件评估新的观察，并与智能体正在进行的行动协调适当的贡献。它还进一步更新……

    arXiv:2611.05731v1 Announce Type: new  Abstract: Most memory systems for LLM agents are retrospective, preserving and retrieving past information to inform current decisions. Proactive agents, however, also require prospective memory: retaining future-directed intentions and realizing them when appropriate opportunities arise, even without explicit requests. Crucially, remembering the right intention does not guarantee acting at the right time: a related observation may call for waiting, while a later cue may make the same intention actionable. Inspired by cognitive theories of human prospective memory, we introduce Pro-Memory, a memory framework that separates maintaining future intentions from deciding when and how to realize them. Pro-Memory preserves pending objectives linked to their supporting evidence, assesses new observations against their original goals and activation conditions, and coordinates appropriate contributions with the agent's ongoing actions. It further updates in
    
[^71]: 互联网文化与大型语言模型中的男性抽象偏差

    The Male Abstraction Bias in Internet Culture and Large Language Models

    [https://arxiv.org/abs/2611.05730](https://arxiv.org/abs/2611.05730)

    研究发现互联网文化和大型语言模型中存在系统性的“男性抽象偏差”——越抽象的概念越与男性相关联，该偏差跨越不同语言、概念领域和语法结构（动词、副词更抽象且更男性化），并被人类内化和AI算法复制。

    

    抽象性和性别是人类认知的两个基础维度，通常被作为相互独立的变量来研究。本研究发现，它们在覆盖单词表达（N = 40,000）和多词表达（N = 67,000）的整个概念空间中系统性关联：在互联网上数十亿词汇和数百万图像中——并经过多种测量技术的验证——越抽象的概念越与男性相关联，这一效应在世界各地的语言中均可复现。该模式不仅在各个概念领域普遍成立，还映射到语法结构上：动词和副词更抽象且更带有男性色彩，这是强调男性主动性特质的刻板印象在系统层面的痕迹。此外，这种偏差还出现在人类和AI在自由联想任务中的反应里，表明其已被人类内化并被算法复制。男性抽象偏差可能是多种刻板印象的共同基础，并将其统一起来……

    arXiv:2611.05730v1 Announce Type: cross  Abstract: Two foundational aspects of human cognition, abstractness and gender, are typically examined as separable. Here, we find they are systematically linked throughout the entire space of concepts spanning single (N = 40,000) and multi-word expressions (N = 67,000): across billions of words and millions of images on the internet -- and across numerous measurement techniques -- more abstract concepts are more associated with men, an effect which replicates in languages around the world. This pattern not only holds across conceptual domains, but it also maps onto grammatical structure: verbs and adverbs are more abstract and more male-coded, a system-wide trace of stereotypes accentuating agentic qualities in men. Furthermore, this bias emerges in human and AI responses in free association tasks, indicating its internalization and its algorithmic reproduction. The male abstraction bias may underlie and unify a range of stereotypes, evidencing
    
[^72]: 配备搜索的LLM智能体中的认知偏差特征：一种分阶段解析分析

    Cognitive Bias Signatures in Search-Equipped LLM Agents: A Stage-Resolved Analysis

    [https://arxiv.org/abs/2611.05720](https://arxiv.org/abs/2611.05720)

    该研究通过控制变量与匹配对照的分阶段实验，揭示了配备搜索功能的LLM智能体中存在类似人类的认知偏差特征——最终答案在主动搜索时对初始证据顺序更敏感，且死胡同式失败历史会抑制策略切换，这些效应无法用文档组成或上下文长度来解释。

    

    搜索可以帮助LLM智能体纠正错误，但它们的起始条件可能会塑造它们找到并使用哪些证据。借鉴认知偏差范式，我们开发了一个受控搜索实验框架，在保持每个问题和文档集合固定不变的前提下，检验结果排序、先前工作记录和初始假设的影响。通过匹配对照实验，我们区分了排序效应与文档组成差异的影响，以及失败工作记录的影响与上下文长度的影响。我们在10个模型和1,000个合成搜索世界环境中，跟踪了查询、检索到的文档、文档打开情况以及最终答案。我们的对比结果表明，与所有证据被完整一次性提供的情况相比，最终答案在主动搜索过程中对初始证据顺序更为敏感。与匹配的中性历史记录相比，死胡同式的失败历史记录会减少策略切换，这表明其影响超出了上下文长度的范畴。在错误假设条件下，分阶段测量能够区分检索减少或……（摘要原文在此处截断）

    arXiv:2611.05720v1 Announce Type: new  Abstract: Search can help LLM agents correct mistakes, but their starting conditions may shape which evidence they find and use. Drawing on cognitive-bias paradigms, we develop a controlled search harness to examine result order, records of prior work, and initial hypotheses while holding each question and document collection fixed. Matched controls distinguish order effects from differences in document composition and the effects of failed work from context length. Across 10 models and 1,000 synthetic search worlds, we track queries, retrieved documents, document openings, and final answers. Our comparisons suggest that final answers are more sensitive to initial evidence order during active search than when all evidence is supplied in full. Dead-end histories can reduce strategy switching relative to matched neutral histories, indicating an effect beyond context length. Under wrong hypotheses, stage measurements distinguish reduced retrieval or 
    
[^73]: 有据可依但非免疫：“智能体即评审”中的确认偏误

    Grounded but Not Immune: Confirmation Bias in Agent-as-a-judge

    [https://arxiv.org/abs/2611.05662](https://arxiv.org/abs/2611.05662)

    本研究揭示了“智能体即评审”即使自行收集了决定性证据，仍会因一条无证据价值的虚假权威备注而引发确认偏误，从而推翻多达40%的正确判定。

    

    评估开放式智能体任务越来越多地交给“智能体即评审”：这是一种配备工具并获得智能体完整轨迹的评估者，以便其对照环境验证结果。其前提是，一个自行收集证据的评审者应当被证据所锚定。我们表明，这种证据基础并不能使评审者对确认偏误免疫：一条虚假的预筛备注——尽管被冠以权威来源、且对正确判定没有任何证据价值——竟能在四个评审者和五个数据集上，推翻使用工具的评审者多达40%的正确判定，即使评审者收集了决定性证据。这种判定转变由该提示线索的特定属性所驱动。我们将这一线索视为确认偏误的受控诊断手段，而非对抗性攻击。我们的核心贡献是解释了确认偏误如何改变评审者在轨迹中的行为，我们对评审者行为的变化进行了分类，并将变化归因于轨迹中的特定环节。

    arXiv:2611.05662v1 Announce Type: new  Abstract: Evaluating open-ended agent tasks increasingly falls to an Agent-as-a-Judge: an evaluator handed tools and the agent's full trajectory so it can verify a result against the environment. The premise is that a judge that gathers its own evidence should be grounded by it. We show that this grounding does not immunize the judge against confirmation bias: a single false pre-screening note, attributed to an authority and carrying no evidential bearing on the correct verdict, overturns up to 40% of a tool-using judge's correct verdicts across four judges and five datasets, even though the judge collects decisive evidence. The shift is driven by specific properties of the cue. We treat this cue as a controlled diagnostic of confirmation bias rather than an adversarial attack. Our central contribution is to explain how confirmation bias changes the judge's behavior in a trajectory. We classify changes in judge behavior and attribute changes to on
    
[^74]: 用于修复多语言语音识别错误的情境化象形图

    Contextual Pictograms for Repairing Multilingual ASR Errors

    [https://arxiv.org/abs/2611.05637](https://arxiv.org/abs/2611.05637)

    提出了一种结合备选转录文本、语音回放和情境化象形图校验的交互界面，帮助听者在多语言（法语、西班牙语）语音识别错误发生时恢复说话者的真实意图。

    

    当相互竞争的词候选不熟悉或在视觉上相似时，语音转录错误可能难以被察觉。我们描述了一种界面，它将备选转录文本、语音回放以及针对选定不确定概念的情境化校验象形图相结合。该系统最初将支持法语和西班牙语的功能性话语。一项平衡顺序的对照研究将评估图像反馈能否在仅依赖文本备选项之外，帮助听者恢复说话者想要表达的动作。我们还将量化语义不匹配以及额外的交互时间，因为不恰当的图像可能会强化错误的理解而非修复它。

    arXiv:2611.05637v1 Announce Type: new  Abstract: Speech transcription errors can be difficult to recognize when competing word hypotheses are unfamiliar or visually similar. We describe an interface that combines alternative transcripts, replay and contextually checked pictograms for selected uncertain concepts. The system will initially support French and Spanish functional utterances. A counterbalanced study will assess whether pictorial feedback helps listeners recover the speaker's intended action beyond text alternatives alone. We will also quantify semantic mismatch and additional interaction time, since an inappropriate image can reinforce the wrong interpretation rather than repair it.
    
[^75]: RAGCompass：通过重排序实现可泛化的检索器路由

    RAGCompass: Generalizable Retriever Routing via Reranking

    [https://arxiv.org/abs/2611.05596](https://arxiv.org/abs/2611.05596)

    RAGCompass 通过对比微调重排序器直接读取不同检索器返回的段落集合来实现查询路由，从而可泛化到未见过的检索器和新语料库，将答案 F1 相对提升 28%。

    

    在检索增强生成（RAG）中存在多种不同的检索器，但没有一种检索器能在所有查询上都表现最佳。因此，关键问题是：给定一个查询以及不同检索器检索到的段落，哪一种最适合该查询？现有的路由器通常基于查询特定特征和固定的检索器池进行预测；这类路由器对于未见过的检索器无法给出输出，且在设计上天生无法应对新语料库。为填补这些空白，RAGCompass 直接读取检索到的段落集合本身：来自陌生检索器或新语料库的段落仍可与其他段落直接比较，因此在仅依赖查询特征的路由会失效的场景下，路由依然有效。我们针对每个证据集所产生的答案，对重排序器进行对比微调，这贡献了绝大部分性能提升；一个小型的比较模块带来了适度的额外改进，且这两个组件都不包含检索器专属参数。RAGCompass 将答案 F1 相对提升了 28%

    arXiv:2611.05596v1 Announce Type: new  Abstract: Different retrievers exist for retrieval-augmented generation, yet none performs best for all queries. Therefore, the key question is: given a query and retrieved passages from different retrievers, which one is most suited for that query? Existing routers usually make predictions from query-specific features and a fixed pool of retrievers; such routers have no output for an unseen retriever and are incapable by design on a new corpus. To fill these gaps, RAGCompass reads the retrieved passage sets themselves: passages from an unfamiliar retriever or corpus stay directly comparable to other passages, so routing stays well designed where a query-only map would fail. We contrastively fine-tune a reranker against the answer each evidence set produces, which accounts for most of the gain; a small comparison head adds a moderate improvement, and neither component has retriever-specific parameters. RAGCompass improves answer F1 by 28% relative
    
[^76]: AGORA：基于项目反应理论的可靠性感知LLM裁判聚合方法

    AGORA: Reliability-Aware Aggregation of LLM Judges via Item-Response Theory

    [https://arxiv.org/abs/2611.05556](https://arxiv.org/abs/2611.05556)

    提出基于项目反应理论的AGORA聚合器，通过敏感度参数和有序评分阈值刻画各LLM裁判，无需人工标注即可从全部评估项中联合学习裁判可靠性，在仅有几百个样本的小规模评估集上显著优于Dawid-Skene等传统方法。

    

    LLM作为裁判（LLM-as-a-judge）是替代昂贵人工评估的常用方式，而聚合多个裁判的评估结果是应对单一裁判偏差的标准方法。然而，裁判池中混杂着可靠性差异极大的裁判，而用于验证裁判的人工标注数据集通常只有几百个样本甚至更少。在这种情形下，逐项平均的方法对弱裁判和强裁判赋予同等权重，而像Dawid-Skene这样的混淆表模型则必须从过少的样本中估计每个裁判的错误表。我们提出AGORA（可靠性加权评估聚合器），这是一个基于项目反应理论的聚合方法，它将每个评估项置于同一潜在质量量表上，并用敏感度参数和有序评分阈值来刻画每个裁判，从而能够从所有评估项中联合学习裁判的可靠性，且无需人工标注。在三个基准测试中，AGORA在大型评估集上达到或超过Dawid-Skene的表现，且随着评估集缩小至几百个样本，其优势进一步扩大。

    arXiv:2611.05556v1 Announce Type: new  Abstract: LLM-as-a-judge is a common substitute for costly human evaluation, and aggregating several judges is the standard remedy for any single judge's bias. Judge pools mix judges of very different reliability, while the human-labeled sets that validate them hold a few hundred items or fewer. In this regime the per-item mean weights weak judges as heavily as strong ones, and confusion-table models such as Dawid-Skene must estimate a per-judge error table from too few items. We propose AGORA (Aggregation Of Reliability-weighted Assessments), an item-response-theory aggregator that places every item on one latent quality scale and describes each judge by a sensitivity parameter and ordered rating thresholds, so judge reliability is learned from all items jointly and without human labels. Across three benchmarks, AGORA matches or exceeds Dawid-Skene on large evaluation sets, and its advantage widens as the set shrinks toward a few hundred items. I
    
[^77]: 语言模型对数据科学项目的迭代式创建、评分与修订

    Iterated creation, grading, and revision of data science projects by language models

    [https://arxiv.org/abs/2611.05548](https://arxiv.org/abs/2611.05548)

    本研究通过“写作者-评分者”迭代反馈框架评估七个语言模型完成端到端数据科学项目的能力，发现三轮反馈修订使平均得分从59%提升至76%，且不同模型提升幅度差异显著（3至46个百分点）。

    

    我们评估了智能体AI模型能够完成端到端数据科学项目的程度，以及这一能力如何随反馈而提升。一个配备特定工具的“写作者”模型创建一个完整的数据科学项目，包括下载数据集、清洗和分析数据集，并就此撰写一篇短论文。项目随后被执行，并由一个“评分者”模型进行评估。写作者模型接着可以修改其提交内容，如此重复共进行三轮。我们考虑了来自五个开发商（Anthropic、OpenAI、Google、DeepSeek和Z.AI）的七个模型。每个模型既充当写作者又充当评分者，因此共有49个写作者-评分者组合，我们对每个组合运行三次。在147份初始提交中，107份能够成功运行并渲染，经过最后一轮反馈后，平均得分从59%提升至76%。按写作者模型来看，提升幅度从Anthropic的Opus 5.5的3个百分点到Anthropic的Haiku的46个百分点不等。

    arXiv:2611.05548v1 Announce Type: new  Abstract: We evaluate the extent to which agentic AI models can complete an end-to-end data science project, and how this improves with feedback. A 'writer' model with specific tools creates a complete data science project including downloading, cleaning, and analyzing a dataset, and writing a short paper about it. The project is executed and a 'grader' model evaluates it. The writer can then revise its submission, and this repeats for a total of three rounds. We consider seven models from five developers: Anthropic, OpenAI, Google, DeepSeek, and Z.AI. Each model is both a writer and a grader, so there are 49 writer-grader pairs and we run each pair three times. Of the 147 initial submissions, 107 run and render, and the average score increases from 59 per cent to 76 per cent after the final round of feedback. By writer the extent of the increase ranges from 3 percentage points for Anthropic's Opus 5.5 to 46 percentage points for Anthropic's Haiku
    
[^78]: 先改写，后否决：面向基于大语言模型后OCR纠错的重写器无关编辑门控

    Rewrite, Then Veto: Rewriter-Agnostic Edit Gating for LLM-Based Post-OCR Correction

    [https://arxiv.org/abs/2611.05539](https://arxiv.org/abs/2611.05539)

    提出EditVeto——在LLM全文改写之后加一个0.6B参数的否决门控，将改写与OCR原文字符对齐、差异分块并逐块决定保留或撤销，从而以重写器无关的方式提升OCR后纠错效果，同时避免改动原本正确的文本。

    

    大语言模型能够修复历史文档中含噪声的OCR转录文本，但它们也会改动原本已经正确的文字。我们提出了EditVeto，一个放置在全文上下文LLM改写之后的否决门控。改写结果与OCR输入进行字符级对齐，其差异被切分为块，再由一个0.6B参数的门控模型决定保留或撤销每个块。一项Oracle分析为该设计提供了依据：在已知真实错误位置的情况下，按跨度逐一生成比不做纠错更差，而对完整改写产生的编辑进行否决则有所助益。该门控在两个Qwen3模型生成的改写结果上训练，并以冻结状态在八个HIPE-OCRepair-2026测试集上评估。结果显示，该门控提升了所有五个受测改写器的表现（其中两个在训练中未曾见过）：加权cMER下降0.0026-0.0044，偏好得分上升0.15-0.25（配对bootstrap检验，各自p值约0.0002）。在已公布的竞赛输出上，它降低了四支队伍中三支的cMER，但使冠军队伍的cMER上升（从0.0050升至0.0065）。在第二项已公开的

    arXiv:2611.05539v1 Announce Type: new  Abstract: LLMs can repair noisy OCR transcripts of historical documents, but they also change text that was already correct. We propose EditVeto, a veto gate placed after a full-context LLM rewrite. The rewrite is character-aligned to the OCR input, its differences are split into blocks, and a 0.6B-parameter gate keeps or reverts each block. An oracle analysis motivates the design: given the true error locations, per-span generation is worse than no correction, whereas vetoing a full rewrite's edits helps. Trained on rewrites by two Qwen3 models and evaluated, frozen, on the eight HIPE-OCRepair-2026 test sets, the gate improves all five rewriters tested, two of them unseen in training: weighted cMER falls by 0.0026-0.0044 and the preference score rises by 0.15-0.25 (paired bootstrap, p ~ 0.0002 each). On published competition outputs it lowers cMER for three of four teams but raises it for the winning run (0.0050 to 0.0065). In a second, disclosed
    
[^79]: 何时提问，何时回答：对话中歧义指代消解的策略性澄清

    When to Ask, When to Answer: Strategic Clarification for Ambiguous Reference Resolution in Dialogue

    [https://arxiv.org/abs/2611.05517](https://arxiv.org/abs/2611.05517)

    该论文提出“策略性澄清”任务并基于 SIMMC 2.1 构建了可控评测环境，同时提出轻量级的 BACE 动作选择方法，让 LLM 学会在澄清问题能缩小候选指代范围时才提问、否则直接回答，以缓解其过早作答的偏差。

    

    对人类而言，指代消歧是一个协作过程，人们只在澄清问题能够实质性改变结果时才会提出这些问题。然而，已有研究表明，大语言模型（LLM）存在过早作答的偏差，更倾向于给出确定的单轮回答，而非进行更多的澄清式交流。本文研究“策略性澄清”问题——即在指代游戏中，当澄清问题的答案能够缩小候选指称对象的集合时才提出澄清问题。我们基于 SIMMC 2.1 数据集构建了一个受控测试环境，该环境记录了每个歧义指代需要多少个问题才能被成功消解，并考察了基线 LLM 行为和常见后训练方法在“提出澄清问题”与“尝试回答指代”之间的表现。我们还提出了 BACE（Belief-Aware Critic with Epistemic tracking，具有认知追踪的信念感知评论器），这是一种在冻结 LM 骨干之上的轻量级动作选择方法，并展示了……

    arXiv:2611.05517v1 Announce Type: cross  Abstract: For humans, referential disambiguation is a collaborative process involving clarification questions asked when they will meaningfully change the outcome. However, large language models (LLMs) have been shown to be biased toward premature answering, preferring definite single-turn responses over more clarification exchanges. In this paper, we study the problem of "strategic clarification" -- asking clarification questions in a reference game when their answers will narrow the set of candidate referents. We construct a controlled test environment from the SIMMC 2.1 dataset that records how many questions each ambiguous reference needs to be successfully resolved, and examine how baseline LLM behavior and common post-training methods ask clarifying questions vs. attempt to answer the referent. We also propose BACE (Belief-Aware Critic with Epistemic tracking), a lightweight action chooser method on top of a frozen LM backbone, and show th
    
[^80]: SlackBench：基于真实代码仓库的协作项目中智能体基准测试

    SlackBench: Benchmarking Agents on Collaborative Projects Grounded in Real Code Repositories

    [https://arxiv.org/abs/2611.05512](https://arxiv.org/abs/2611.05512)

    SlackBench是一个基于真实学术研究项目隐私保护重现构建的基准测试，用于评估AI智能体在需要对工作场所沟通与不断演变的代码仓库状态进行联合推理的现实协作研究与工程工作流中的能力。

    

    AI智能体正在成为数字同事，它们能够浏览网页、编写代码、运行实验，并在较长时间跨度内协助人类工作。然而，现有基准测试难以反映真实的知识工作：许多基准依赖于合成数据或孤立任务，忽略了协作项目中复杂的多方动态——在这种项目中，关键信息散布在对话、代码和不断演变的决策之中。即使是简单的工作场所问题，也可能需要定位相关频道和私信、从冗长的讨论中恢复上下文、检查仓库状态、调和过时或相互冲突的证据，并进行跨来源的综合分析。我们提出了SlackBench，这是一个用于评估智能体在现实研究与工程工作流中表现的基准，这些工作流需要对工作场所沟通与不断演变的仓库状态进行联合推理。SlackBench基于真实学术研究项目的隐私保护重现构建，提供了可复现的……

    arXiv:2611.05512v1 Announce Type: new  Abstract: AI Agents are becoming digital coworkers that can browse the web, write code, run experiments, and assist humans over extended horizons. Yet existing benchmarks fall short of real knowledge work: many rely on synthetic data or isolated tasks, missing the complex, multi-party dynamics of collaborative projects, where critical information is scattered across conversations, code, and evolving decisions. Even simple workplace questions may require locating relevant channels and direct messages, recovering context from long discussions, inspecting repository state, reconciling stale or conflicting evidence, and synthesizing across sources.   We introduce SlackBench, a benchmark for evaluating agents on realistic research and engineering workflows that require joint reasoning over workplace communication and evolving repository state. Grounded in privacy-preserving reenactments of real academic research projects, SlackBench provides reproducib
    
[^81]: 解锁图灵张量核心：在单张22G RTX 2080 Ti上从头预训练的实测吞吐量报告

    Unlocking Turing Tensor Cores: A Measured Throughput Account of From-Scratch Pre-training on a Single 22G RTX 2080 Ti

    [https://arxiv.org/abs/2611.05419](https://arxiv.org/abs/2611.05419)

    提出turing-unlock内核模块，通过在nvidia驱动绑定设备前写入FP16速度选择器解除RTX 2080 Ti的FP16算力限制，使cuBLAS FP16矩阵乘吞吐量提升约1.6倍，并在单卡从头预训练中实现实测稳态吞吐量提升11.8%。

    

    RTX 2080 Ti（TU102）在硬件上配备了完整的第二代张量核心，但厂商在驱动层面限制了FP16矩阵乘法的速度，使可用的张量吞吐量远低于硬件峰值。turing-unlock通过一个内核模块解除了这一限制：在nvidia驱动绑定设备之前，写入FP16速度选择器（SS0=0x88888888, SS1=0x8）并释放ACR互斥锁；已发表的测量结果显示，cuBLAS FP16矩阵乘法吞吐量从55.7提升至91.4 TFLOPS（约1.6倍）。在与配套显存账目研究相同的22G 2080 Ti上、遵循相同的从头预训练协议（五个规模的中文编码器-解码器模型，每档2,451步，每步16,384个token，随机种子42），我们获得了开启前后对比的吞吐量报告：小规模档（75.7M参数，512宽，8+8层，开启激活检查点）的稳态吞吐量从6,480提升至7,246 tok/s（+11.8%），墙钟时间缩短……

    arXiv:2611.05419v1 Announce Type: cross  Abstract: The RTX 2080 Ti (TU102) ships full second-generation Tensor Cores in hardware, yet the vendor throttles FP16 matrix multiplication at the driver level, leaving usable tensor throughput well below the hardware peak. turing-unlock lifts this restriction with a kernel module that, before the nvidia driver binds the device, writes the FP16 speed selectors (SS0=0x88888888, SS1=0x8) and releases the ACR mutex; published measurements report cuBLAS FP16 matmul throughput rising from 55.7 to 91.4 TFLOPS (about 1.6x). On the same 22G 2080 Ti and under the same from-scratch pre-training protocol as the companion memory ledger (five tiers of Chinese encoder-decoder models, 2,451 steps per tier, 16,384 tokens per step, seed 42), we obtain a before/after throughput account: the small tier (75.7M parameters, 512-wide, 8+8 layers, activation checkpointing on) improves in steady-state throughput from 6,480 to 7,246 tok/s (+11.8%) with wall time cut fro
    
[^82]: 面向医疗多智能体推理的挑战感知知识招募与元智能体监督方法

    Challenge-Aware Knowledge Recruitment with Meta-Agent Supervision for Medical Multi-Agent Reasoning

    [https://arxiv.org/abs/2611.05393](https://arxiv.org/abs/2611.05393)

    该论文提出挑战感知的自适应招募框架，由外部挑战者智能体监控医疗多智能体推理过程中涌现的知识与专业缺口，动态触发针对性知识检索或讨论中途的专家招募，在MEDAGENTSBENCH困难子集上显著优于仅依赖问题的事前检索策略并提升多数据集准确率。

    

    大语言模型在标准医疗问答任务上表现强劲，但在更困难的推理任务上性能会下降，而多智能体辩论在此类任务上往往无法超越强大的单模型提示方法。我们认为一个关键局限在于：现有系统仅根据问题本身来选择专家和检索知识，而不是根据推理过程中涌现出的知识缺口。我们提出挑战感知的自适应招募机制：一个外部的“挑战者”检查不断演化的推理轨迹，并识别出具体的知识缺口或专业知识缺口。这些信号会触发针对性的知识检索，或在讨论中途招募领域专家。在MEDAGENTSBENCH困难子集（涵盖九个生物医学数据集的862个问题）上，事前检索几乎不带来收益，而挑战感知的推理中间检索表现更好；在多专家骨干系统上，该方法在9个数据集中的8个上提升了准确率，相比仅使用挑战者的方案提升了6个百分点。消融实验表明两者均有收益……

    arXiv:2611.05393v1 Announce Type: new  Abstract: Large language models perform strongly on standard medical QA but degrade on harder reasoning tasks, where multi-agent debate often fails to beat strong single-model prompting. We argue that a key limitation is that existing systems select experts and retrieve knowledge from the question alone, rather than from gaps that emerge during reasoning. We propose challenge-aware adaptive recruitment: an external CHALLENGER inspects the evolving reasoning trace and identifies specific factual or expertise gaps. These signals trigger targeted knowledge retrieval or mid-discussion specialist recruitment. On the MEDAGENTSBENCH hard subset (862 questions across nine biomedical datasets), upfront retrieval provides little benefit, while challenge-aware in-between retrieval performs better; on a multi-specialist backbone it improves accuracy on 8 of 9 datasets and adds 6 percentage points over Challenger Only. Ablations indicate benefits from both ret
    
[^83]: 智能体执行框架感知：将经验证的遥测数据转化为工具调用智能体的运行控制

    Agent Harness Awareness: Turning Verified Telemetry into Operational Control for Tool-Using Agents

    [https://arxiv.org/abs/2611.05391](https://arxiv.org/abs/2611.05391)

    提出名为“智能体执行框架感知”的信任分离设计，由宿主在关键操作前向模型提供经验证的最新运行状态快照，弥补模型对执行框架运行信息的“运行盲区”，使三种主流模型上的平均可观测策略遗憾降低96.9%。

    

    工具描述告诉智能体某个操作能够做什么，但不一定揭示该操作在使用时刻是否健康或适用。执行框架通常通过监控器、追踪记录和近期故障掌握这些信息，而选择下一步操作的模型却无从得知。我们将这一差距称为“运行盲区”。我们提出“智能体执行框架感知”，这是一种信任分离的设计：在执行关键操作之前，由宿主向模型提供一份小型、最新且经过验证的状态快照，同时保留测量、强制执行、评估和最终分发的权限。我们使用ToolRoute测试其工具与网络实例，ToolRoute是一个受控隔离基准，它在保持任务和候选语义不变的情况下，仅改变模型可获得的运行信息。在涵盖GPT-5.6 Sol、Claude Sonnet 5和Gemini 3.7 Flash的216次决策消融实验中，经验证的工具健康状态使平均可观测策略遗憾降低了96.9%。

    arXiv:2611.05391v1 Announce Type: new  Abstract: Tool descriptions tell an agent what an action can do, but they do not necessarily reveal whether that action is healthy or suitable at the moment of use. Execution harnesses often possess this information through monitors, traces, and recent failures while the model choosing the next action does not. We call this gap operational blindness. We introduce Agent Harness Awareness, a trust-separated design in which the host supplies a small, fresh, verified state snapshot before a consequential action while retaining measurement, enforcement, evaluation, and final dispatch authority. We test its tools-and-network instance with ToolRoute, a controlled isolation benchmark that holds the task and candidate semantics constant and varies the operational information available to the model. In a 216-decision ablation across GPT-5.6 Sol, Claude Sonnet 5, and Gemini 3.7 Flash, verified tool-health state reduced mean observable policy regret by 96.9% 
    
[^84]: AI驱动的SaaS平台集成：AI Express统一生成式AI服务——基于按量付费积分系统与可扩展微服务架构

    AI-Driven SaaS Platform Integration: AI Express for Unified Generative AI Services with Pay-As-You-Go Credit Systems and Scalable Microservice Architecture

    [https://arxiv.org/abs/2611.05389](https://arxiv.org/abs/2611.05389)

    AI Express通过按量付费积分系统和可扩展微服务架构，将碎片化的生成式AI能力（代码审查、文本转网站、语法检查等）整合到单一SaaS平台中，使用户无需多重订阅和技术专长即可低成本使用统一的AI服务。

    

    当前的生成式AI服务呈现碎片化状态，文本生成、图像合成和代码辅助等功能需要分别订阅，导致成本高昂且工作流程频繁中断。AI Express是一个SaaS平台，将这些能力整合为统一的按量付费服务。该平台提供一系列AI工具，包括用于评估GitHub代码仓库的核心代码审查器、可将文本转换为完整功能网站的文本转网站生成器，以及脚本生成、语法检查、数据分析和水印去除等功能，所有工具均可通过单一界面访问，无需任何专业技术知识。

    arXiv:2611.05389v1 Announce Type: cross  Abstract: Generative AI is currently fragmented, with separate subscriptions required for text generation, image synthesis, and code assistance, leading to high costs and workflow disruptions. AI Express is a SaaS platform that consolidates these capabilities into a unified pay-as-you-go service. The platform provides a range of AI tools, including a Core Code Reviewer that evaluates GitHub repositories, a Text-to-Website generator that converts text into fully functional websites, script generation, grammar checking, data analysis, and watermark removal, all accessible through a single interface without requiring technical expertise.
    
[^85]: SRRM：一个用于评估小语言模型长上下文记忆的检索-重构基准

    SRRM: A Retrieval-Reconstruction Benchmark for Evaluating Long-Context Memory in Small Language Models

    [https://arxiv.org/abs/2611.05386](https://arxiv.org/abs/2611.05386)

    该论文提出了轻量级可复现的SRRM基准和记忆保留率（MPR）指标，通过联合评估定向检索与全局信息重构能力，来衡量小语言模型的长上下文记忆保留水平。

    

    小语言模型（SLM）越来越多地被部署在消费级硬件上，但它们在日益增长的上下文长度中保留信息的能力仍缺乏充分的理解。现有基准主要评估面向检索的任务，对于成功的检索是否能反映全面的长上下文记忆保留，仅能提供有限的洞察。为解决这一局限，本文提出了简单检索-重构记忆（SRRM），这是一个轻量级且完全可复现的基准，可联合评估定向检索和全局信息重构能力。本研究还提出了记忆保留率（MPR），这是一个基准层面的指标，用于量化检索与重构性能之间的关系。SRRM使用合成数据集在三个Qwen2.5模型（0.5B、1.5B和3B参数）上进行了评估，上下文负载范围涵盖10到100条事实陈述。

    arXiv:2611.05386v1 Announce Type: new  Abstract: Small Language Models (SLMs) are increasingly deployed on consumer-grade hardware, yet their ability to preserve information across progressively longer contexts remains insufficiently understood. Existing benchmarks primarily evaluate retrieval-oriented tasks, providing limited insight into whether successful retrieval reflects comprehensive long-context memory preservation. To address this limitation, this paper introduces Simple Retrieval-Reconstruction Memory (SRRM), a lightweight and fully reproducible benchmark that jointly evaluates Targeted Retrieval and Global Information Reconstruction. This study also proposes the Memory Preservation Ratio (MPR), a benchmark-level metric that quantifies the relationship between retrieval and reconstruction performance. SRRM was evaluated using a synthetic dataset on three Qwen2.5 models (0.5B, 1.5B, and 3B parameters) across context loads ranging from 10 to 100 factual statements. Experimental
    
[^86]: 同源不同判：文本生产条件混淆俄语机器生成文本检测

    Same Origin, Different Verdicts: Text Production Conditions Confound Machine-Generated Text Detection in Russian

    [https://arxiv.org/abs/2611.05385](https://arxiv.org/abs/2611.05385)

    该研究利用同一模型、同一任务下仅任务措辞不同的文本对进行对比，证明文本生产条件本身（而非文本真实来源）即可显著影响俄语机器生成文本检测的判定结果。

    

    机器生成文本的自动检测被应用于判定结果会给作者带来实际代价的场景，然而“来源”是生产过程的属性，而非文本本身可观察的属性。文本生产时所处的条件会将文本推向与生成模型被假定为推向的相同方向。本研究在保持来源不变的前提下测量这些条件的贡献。该俄语语料库共包含1882篇文档：机器部分由四个生成通道（各自依托不同的模型家族）在三种任务模式下的45个任务中生成；人类部分来自档案文献，并在阅读文本之前预先标注了来源规约化程度。主要对比在机器语料库内部进行：文本对的两半由同一模型针对同一任务生成，仅在任务措辞上存在差异。机器相似度通过四种操作化方式进行测量，其中包括一个带权重的可解释指数……

    arXiv:2611.05385v1 Announce Type: new  Abstract: Automated detection of machine-generated text is applied where the verdict carries a cost for the author, yet origin is a property of the production process, not an observable property of the text. The conditions under which a text is produced push it in the same direction in which a generative model is presumed to push it. This study measures the contribution of these conditions while origin is held fixed. The Russian-language corpus of 1882 documents comprises a machine part, generated by four generation channels (each backed by its own model family) across 45 tasks in three task modes, and an archival human part annotated for source regimentation level before the texts were read. The primary contrast is computed within the machine corpus: both halves of a pair are generated by the same model for the same task and differ only in task wording. Machine-likeness is measured by four operationalisations: an interpretable index with weights 
    
[^87]: 深度神经网络说话人嵌入中类人的跨语言泛化及其声学基础

    Human-like cross-language generalisation in deep neural speaker embeddings and its acoustic foundations

    [https://arxiv.org/abs/2611.05384](https://arxiv.org/abs/2611.05384)

    本研究证明深度神经网络说话人嵌入能像人类一样跨语言泛化说话人身份判断，其机制源于与人类相似的共享声学线索及语言敏感的加权策略。

    

    泛化能力是深度神经网络的核心问题之一，其中包括说话人嵌入的跨语言泛化。本研究考察了深度神经网络说话人嵌入能否跨语言泛化，以及这种泛化与人类感知和声学结构之间的关系。通过在语言内和跨语言条件下呈现粤语-英语双语语音，我们从18个最先进的模型中提取了相似度，并使用线性混合模型和表征相似性分析将其与人类评分和声学特征进行比较。结果表明，说话人嵌入表现出与人类行为大体一致的跨语言泛化，既能准确辨别说话人身份，又对语言切换表现出敏感性。这种泛化依赖于大体共享的声学线索以及语言敏感的加权方式，与人类的模式相呼应。总之，这些发现为人机在……方面的一致性提供了汇聚性证据。（注：原文摘要在此处被截断）

    arXiv:2611.05384v1 Announce Type: new  Abstract: Generalisability is a central question for deep neural networks, including the cross-language generalisation of speaker embeddings. This study examined whether deep neural speaker embeddings generalise across languages and how such generalisation relates to human perception and acoustic structure. Using Cantonese--English bilingual speech presented in within- and cross-language conditions, we derived similarity from 18 state-of-the-art models and compared it with human ratings and acoustic features using linear mixed models and representational similarity analysis. Embeddings exhibited cross-language generalisation that broadly paralleled human behaviour, reflecting accurate identity discrimination and sensitivity to language switching. This generalisation relied on largely shared acoustic cues with language-sensitive weighting, mirroring human patterns. Together, these findings provide converging evidence for human--machine alignment in
    
[^88]: CCDWIKI：在纵向认知行为疗法（CBT）对话中将个案概念化与回复生成解耦

    CCDWIKI: Decoupling Case Formulation from Response Generation in Longitudinal CBT Dialogue

    [https://arxiv.org/abs/2611.05383](https://arxiv.org/abs/2611.05383)

    提出CCDWiki这一即插即用顾问模块，将纵向CBT对话中的个案概念化建模为基于证据的假设裁定过程，通过对临床假设进行动态编码来区分暂定假设与已确立事实，避免暂时性解释被过早固化为事实而误导干预。

    

    认知行为疗法（CBT）所要求的不仅仅是共情式的情感确认，而是需要对情境、自动化思维、情绪和行为之间的相互作用进行结构化的个案概念化。治疗师通过认知概念化图将这种相互作用形式化，其中维持纵向连贯性需要持续的证据裁定，以区分暂定的假设与已确立的发现。当前的对话系统通过检索或静态用户画像来追踪对话历史，但缺乏编码认知状态的表征能力。因此，暂时性的解释往往被过早地固化为已确立的事实，从而在无意中对下游干预造成偏差。针对这一局限，本文提出了CCDWiki，作为一个即插即用的顾问模块，将纵向治疗概念化为基于证据的假设裁定过程。该方法将每个临床假设编码为一个可动态更新的条目，以追踪其证据状态（注：原文摘要在此处截断）。

    arXiv:2611.05383v1 Announce Type: new  Abstract: Cognitive Behavioral Therapy (CBT) demands more than empathic validation, requiring instead a structured formulation of the interplay among situations, automatic thoughts, emotions, and behaviors. Therapists formalize this interplay through a Cognitive Conceptualization Diagram, where maintaining longitudinal coherence necessitates continuous evidential adjudication to separate tentative hypotheses from established findings. Current dialogue systems track conversational history via retrieval or static profiles but lack the representational capacity to encode epistemic status. Consequently, provisional interpretations are often prematurely reified as established facts, inadvertently biasing downstream interventions. Addressing this limitation, CCDWiki is introduced as a plug-and-play advisor conceptualizing longitudinal therapy as evidence-grounded hypothesis adjudication. The approach encodes each clinical hypothesis as a mutable entry b
    
[^89]: DS@GT ARC 团队参加 CheckThat! 2026：一种以裁定为锚点的事实核查文章生成流水线

    DS@GT ARC at CheckThat! 2026: A Verdict-Anchored Pipeline for Generating Fact-Checking Articles

    [https://arxiv.org/abs/2611.05382](https://arxiv.org/abs/2611.05382)

    该论文提出了一种以事实核查员原始裁定为锚点的四阶段流水线，通过清洗证据、检索聚类来源、基于 Llama-3.1-8B-Instruct 条件化生成文章并用确定性后处理恢复引用，实现了事实核查文章的自动化生成，在 CLEF 2026 CheckThat! 任务3中以 0.428 的平均分获得 9 支队伍中的第三名。

    

    错误信息的传播速度远超专业事实核查员手动验证并撰写文章解释其裁定的速度，因此需要自动化生成事实核查文章。本文详细介绍了 DS@GT ARC 团队在 CLEF 2026 CheckThat! 实验室任务3（事实核查文章生成）中所做的工作。我们的目标是从一个声明、其原始裁定以及一组抓取的源文档出发，生成完整的事实核查文章。我们团队构建了一个四阶段流水线：清洗噪声证据、检索并聚类相关来源、通过以原始事实核查员裁定为条件约束 Llama-3.1-8B-Instruct 模型来生成文章，并通过确定性后处理恢复行内引用。我们取得了 0.428 的平均分，在 CLEF 2026 CheckThat! 实验室任务3的 9 支队伍中排名第三，而基线的平均分为 0.272。我们的代码已在 GitHub 上发布：https://github.com/dsgt-arc/checkth

    arXiv:2611.05382v1 Announce Type: new  Abstract: Misinformation spreads faster than professional fact-checkers can manually verify it and write articles explaining their verdicts. This creates a need to automate the production of fact-checking articles. This paper details work done by the DS@GT ARC team for CLEF 2026 CheckThat! Lab Task 3 Fact-Checking Article Generation. We seek to generate a full fact-checking article from a claim, its original verdict, and a set of scraped source documents. Our team built a four-stage pipeline that cleans noisy evidence, retrieves and clusters relevant sources, generates the article by conditioning \texttt{Llama-3.1-8B-Instruct} on the original fact-checker verdict, and recovers inline citations through deterministic post-processing. We achieved a mean score of 0.428 and ranked 3rd out of 9 teams for the CLEF 2026 CheckThat! Lab Task 3, compared to a baseline mean score of 0.272. Our code is available on GitHub at https://github.com/dsgt-arc/checkth
    
[^90]: 立法法案讨论的话语分析

    Discourse Analysis of Legislative Bill Discussions

    [https://arxiv.org/abs/2611.05381](https://arxiv.org/abs/2611.05381)

    本文提出了一种用于解析立法委员会听证会修辞结构的端到端流水线和修辞结构语法，其中55.21%的加州州法案讨论可被该语法解析，从而改进了立法话语分析与民主问责。

    

    立法听证会是独特的交流事件，具有人类可识别的自身结构和逻辑，然而以往的研究仅将其视为超越文字记录的独特现象来分析。在更深层次上理解立法听证会是分析民主机构内部工作与权力关系的重要一步。在这项工作中，我们提出了一个新颖的端到端流水线，用于从立法委员会听证会中解析修辞结构，旨在改进立法话语分析和民主问责。我们还提出了一种修辞结构语法，并发现我们数据集中55.21%的加州州法案讨论可以用该语法进行解析。我们在超过2000个标注话语（对应100场听证会）的随机子集上，使用两个话语片段分类器评估了我们的流水线。我们发现，基于逻辑回归的分类器结合基于语法的输出掩码和话语标签平滑方法……（原文此处截断）

    arXiv:2611.05381v1 Announce Type: new  Abstract: Legislative hearings are distinct communication events with their own human recognizable structure and logic, yet they have been analyzed as distinct phenomena beyond their transcripts. Understanding legislative hearings at a deeper level is a major step in analyzing the work and power relationships within democratic bodies. In this work, we present a novel end-to-end pipeline for parsing rhetorical structures from legislative committee hearings with the goal of improving legislative discourse analysis and democratic accountability. We also propose a rhetorical structure grammar and find that 55.21% of California state bill discussions in our dataset can be parsed with this grammar. We evaluate our pipeline with two utterance section classifiers on a random subset of over 2,000 annotated utterances corresponding to 100 hearings. We find that a logistic regression-based classifier with grammar-based output masking and utterance tag smooth
    
[^91]: 解读“为什么”：在光与量子比特上实现具有精确词证据的玻恩测量注意力机制

    Reading Why: Born-Measured Attention with Exact Word Evidence on Light and Qubits

    [https://arxiv.org/abs/2611.05355](https://arxiv.org/abs/2611.05355)

    该论文提出一种由玻恩测量构成的变换器（玻恩归一化注意力、测量乘积与玻恩读出），使类别概率成为词振幅的多项式，从而可通过有限求积无误差地精确计算每个词的积分梯度证据，并在 e-SNLI 上取得与同规模 softmax 变换器相当的准确率。

    

    当语言模型给出答案时，我们通常只能估计其“为什么”。我们构建了一个变换器，其中每一步都是一次玻恩测量——注意力为在每一行上归一化的 $|\langle q|k\rangle|^4$，用测量乘积取代前馈非线性，并采用玻恩读出。将其分母保持在输入时的取值，每个类别概率都是词振幅的多项式，因此其积分梯度证据可以通过有限求积精确计算，不存在求积误差：各词的贡献之和恰好等于类别概率超出无信息份额的部分，误差达到 $10^{-11}$ 或更好。在 e-SNLI 自然语言推理任务上，该模型在嵌入层之外约 98k 参数的规模下达到 76.98 ± 0.17% 的测试准确率，与同规模的 softmax 变换器持平（76.62 ± 0.36%）。删除证据排名第一的词会在 45% 的测试样本对中改变答案，而这接近任何单个词删除所能改变答案的比例上限（53%）。

    arXiv:2611.05355v2 Announce Type: replace-cross  Abstract: When a language model answers, we can usually only estimate why. We build a transformer in which every step is a Born measurement - attention $|\langle q|k\rangle|^4$ normalised over each row, a measured product in place of the feed-forward nonlinearity, and a Born readout. With its denominators held at their values for the input, each class probability is a polynomial in the word amplitudes, so its integrated-gradient evidence is computed by a finite quadrature with no quadrature error: the words' contributions sum to the class probability above its no-information share, to $10^{-11}$ or better. On e-SNLI natural language inference, with about 98k parameters outside the embedding, the model reaches $76.98 \pm 0.17$% test accuracy, level with a softmax transformer of the same size ($76.62 \pm 0.36$%). Deleting the word its evidence ranks first changes the answer in 45% of test pairs, out of the 53% any single deletion can chang
    
[^92]: TraceRouter++：基于最优传输的因果语义路由重新对齐

    TraceRouter++: Causal Semantic Routing Realignment via Optimal Transport

    [https://arxiv.org/abs/2611.04878](https://arxiv.org/abs/2611.04878)

    该论文提出TraceRouter++框架，通过“发现—追踪—重新对齐”范式显式追踪大模型中有害语义所依赖的跨层因果路由路径，并借助最优传输重新对齐其不安全激活状态，从而在有效抵御复杂越狱攻击的同时保持模型通用能力。

    

    尽管大型基础模型（LFMs）具有卓越的能力，但仍然容易受到对抗性操纵。当前的安全干预方法主要依赖“局部性假设”，僵硬地抑制孤立的神经元或特征。然而，有害语义以分布式、跨层回路的形式呈现，使得局部消融方法既难以抵御复杂的越狱攻击，又有损于模型的通用能力。为弥合这一差距，我们提出了TraceRouter++，这是一个路由级别的安全干预框架，它显式追踪有害输入下激活的与安全相关的因果路由路径，并重新对齐其不安全的激活状态。我们的框架遵循“发现—追踪—重新对齐”范式：首先通过注意力散度精确定位敏感的起始层，并使用稀疏自编码器（SAE）将稠密激活解耦为可解释的路由特征；随后，将这些特征……（摘要在此处截断）

    arXiv:2611.04878v2 Announce Type: replace  Abstract: Despite their remarkable capabilities, large foundation models (LFMs) remain susceptible to adversarial manipulation. Current safety intervention methods primarily rely on the \textit{locality hypothesis}, rigidly suppressing isolated neurons or features. However, harmful semantics manifest as distributed, cross-layer circuits, rendering localized ablations both brittle against complex jailbreaks and detrimental to general utility. To bridge this gap, we propose TraceRouter++, a routing-level safety intervention framework that explicitly traces safety-relevant causal routing pathways activated under harmful inputs and realigns their unsafe activation states. Our framework operates in a \textit{Discover-Trace-Realign} paradigm: it first pinpoints a sensitive onset layer via attention divergence and disentangles dense activations into interpretable routing features using sparse autoencoders (SAEs). Subsequently, it maps these features 
    
[^93]: 自然语言规则的作用域文法：模逻辑等价的读法计数与枚举

    The Scope Grammar of Natural-Language Rules: Counting and Enumerating Readings modulo Logical Equivalence

    [https://arxiv.org/abs/2611.01825](https://arxiv.org/abs/2611.01825)

    该论文提出“作用域文法”来刻画自然语言规则中连接词作用域的歧义性，通过读一次范式与双标记递推，精确计数并枚举了连接词骨架在模逻辑等价意义下逻辑上不同的读法数量。

    

    为人类编写的规则确定了其原子条件和连接词，但并未明确连接词应如何加括号，而每一种加括号方式都对应一种不同的逻辑读法，自动形式化器必须解决或揭示这些读法。我们研究了“作用域文法”，这是一种刻意保持歧义的上下文无关文法，其推导树即为连接词骨架的各种读法，而对其按逻辑等价取商后的结果才是下游应用真正关心的内容。一个含有 k 个二元连接词的骨架拥有 C_k 棵推导树（即第 k 个卡塔兰数），但逻辑上不同的读法数量却少得多。对于无重复原子的正片段，我们精确刻画了这一差距：借助读一次范式的唯一性，我们用最低公共祖先刻画了可实现的范式，并通过在算子词上的双标记递推对读法进行计数，其结果与最多十一个原子的穷举枚举相验证。该计数在反转和 ∧/∨ 对偶下保持不变。

    arXiv:2611.01825v1 Announce Type: cross  Abstract: A rule written for people fixes its atomic conditions and connectives but leaves open how the connectives are bracketed, and each bracketing is a different logical reading that an autoformalizer must resolve or expose. We study the scope grammar, a deliberately ambiguous context-free grammar whose derivation trees are the readings of a connective skeleton and whose quotient by logical equivalence is what matters downstream. A skeleton with $k$ binary connectives has $C_k$ derivation trees, the $k$-th Catalan number, but far fewer logically distinct readings. For the positive, repetition-free fragment we make the gap exact: through the uniqueness of the read-once normal form we characterize the realizable normal forms by lowest common ancestors and count them with a two-label recurrence over the operator word, verified against exhaustive enumeration up to eleven atoms. The count is invariant under reversal and $\wedge/\vee$ duality; a s
    
[^94]: Labadain-MedQA：一个用于微调和评估大型语言模型的德顿语医疗问答数据集

    Labadain-MedQA: A Tetun Medical QA Dataset for Fine-Tuning and Evaluating Large Language Models

    [https://arxiv.org/abs/2611.00935](https://arxiv.org/abs/2611.00935)

    该论文构建了首个大规模德顿语医疗问答数据集Labadain-MedQA，包含39,077个问答对，通过将MedAlpaca医学数据集翻译成德顿语并进行人机翻译质量对比评估，为低资源语言的医疗LLM微调与评估提供了宝贵资源。

    

    大型语言模型（LLM）的进展为医疗知识从高资源语言向低资源语言的跨语言迁移创造了机会，以支持人们获取医疗保健信息。然而，用于微调和评估LLM的医疗问答（QA）数据集在德顿语中仍然稀缺。为填补这一空白，我们使用Gemini 2.5 Flash将MedAlpaca的医学闪卡和WikiDoc患者信息集合翻译成德顿语，生成了39,077个医疗问答对。翻译质量通过对50个随机抽取的问答对（100个文档）进行评估，采用自动指标与人工参考翻译对比，并由两名以德顿语为母语的医疗专业人员进行评估，重点关注医学准确性和德顿语流畅性。人工翻译在两项标准上的平均评分均高于LLM生成的翻译（4.43/5 对比 4.02/5），差异具有统计学显著性（Wilcoxon符号秩检验，p <...）（摘要在此处被截断）

    arXiv:2611.00935v1 Announce Type: new  Abstract: Advances in large language models (LLMs) have created opportunities for cross-lingual transfer of medical knowledge from high-resource to low-resource languages to support access to healthcare information. However, medical question-answering (QA) datasets for fine-tuning and evaluating LLMs remain scarce for Tetun. To address this gap, we translated MedAlpaca's Medical Flashcards and WikiDoc Patient Information collections into Tetun using Gemini 2.5 Flash, producing 39,077 medical QA pairs. Translation quality was assessed on 50 randomly sampled QA pairs (100 documents) using automatic metrics against human reference translations and evaluations by two native Tetun-speaking healthcare professionals, focusing on medical accuracy and Tetun fluency. Human translations achieved higher mean ratings than LLM-generated translations for both criteria (4.43/5 vs. 4.02/5), with statistically significant differences (Wilcoxon signed-rank test, p <
    
[^95]: 专业翻译工作流中的数字主权：从Windows迁移到Linux的案例研究

    Digital sovereignty in professional translation workflows: a case study of migrating from Windows to Linux

    [https://arxiv.org/abs/2610.41561](https://arxiv.org/abs/2610.41561)

    本研究提出以“控制、连续性、退出”三原则结合三项实践要求的数字主权评估框架，并通过将译者工作站从Windows迁移至Linux（采用LoopCAT、LibreOffice和本地运行的TranslateGemma）的案例研究，证明了专业翻译工作流可在可本地运行的Linux核心上实现重构。

    

    本研究通过筛选计算机辅助翻译工具和翻译管理系统，并以一个将译者工作站从Windows迁移到Linux的案例研究，考察了专业翻译中的数字主权问题。该框架结合三项原则——控制、连续性与退出——以及三项实践要求：Linux本地原生运行、独立于强制性外部服务，以及对关键数据和资源的独立访问。基于文献证据和从服务商处获得的有限回应，生态系统筛选区分了“Linux可访问性”与“运营独立性”：基于浏览器的平台可能仍然依赖于服务商控制的基础设施。迁移案例展示了围绕可本地运行的Linux核心重构专业翻译工作流的可行性，使用了LoopCAT、LibreOffice和本地运行的TranslateGemma，以及其他工具……

    arXiv:2610.41561v1 Announce Type: new  Abstract: This study examines digital sovereignty in professional translation through a screening of computer-assisted translation tools and translation management systems, and a case study of migrating a translator's workstation from Windows to Linux. The framework combines three principles - Control, Continuity and Exit - with three practical requirements: native Linux operation, independence from mandatory external services, and independent access to essential data and resources. Drawing on documentary evidence and the limited number of responses received from providers, the ecosystem screening distinguishes Linux accessibility from operational independence: browser-based platforms may remain dependent on provider-controlled infrastructure. The migration case demonstrates the feasibility of reconstructing one professional workflow around a locally operable Linux core, using LoopCAT, LibreOffice and locally run TranslateGemma, alongside tools fo
    
[^96]: MedConstBench：衡量“宪法”如何引导大语言模型医疗伦理行为的基准测试

    MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics

    [https://arxiv.org/abs/2610.36513](https://arxiv.org/abs/2610.36513)

    该论文提出 MedConstBench，首个将部署“宪法”作为受控变量、围绕比彻姆四大伦理原则构建、并通过人工校准的多模型同行评审评分的医疗伦理基准，实现了对宪法如何引导大语言模型医疗伦理行为的系统分析。

    

    随着大语言模型的快速发展，它们被越来越多地用于回答医学问题，且许多此类模型如今已在本地部署。模型一旦部署，其行为主要由写入其“宪法”中的伦理政策所决定。然而，现有研究大多关注事实正确性，而非相互冲突的伦理价值之间的权衡。即使是那些确实评估伦理的基准测试，也将伦理视为模型的固定属性而非变量，因此“宪法”如何重塑医疗伦理行为在很大程度上仍未被探索。我们提出 MedConstBench，一个将部署宪法作为受控变量的医疗伦理基准测试，它围绕比彻姆的四大伦理原则组织，并通过人工校准的多模型同行评审进行评分。该基准包含我们自主构建的临床困境数据集 MedDilemma，并支持对“宪法”如何引导大语言模型进行系统性分析。对十个开源（模型）的评估……

    arXiv:2610.36513v1 Announce Type: new  Abstract: As large language models advance rapidly, they are increasingly used to answer medical questions, and many such models are now deployed locally. Once a model is deployed, its behavior is governed mainly by the ethical policy written into its constitution. However, most existing work focuses on factual correctness rather than the trade-off between competing ethical values. Even benchmarks that do evaluate ethics treat it as a fixed property of the model rather than as a variable, so how a constitution reshapes medical-ethics behavior remains largely unexplored. We introduce MedConstBench, a medical-ethics benchmark that treats the deployment constitution as a controlled variable, is organized around Beauchamp's four principles, and is scored by human-calibrated multi-model peer review. It includes MedDilemma, our own dataset of clinical dilemmas, and supports a systematic analysis of how constitutions steer LLMs. Evaluating ten open-sourc
    
[^97]: 面向历史文档转录的实体感知OCR系统评估

    Entity-Aware Evaluation of OCR Systems for Historical Document Transcription

    [https://arxiv.org/abs/2610.33004](https://arxiv.org/abs/2610.33004)

    该论文提出以实体级错误分析作为OCR评估的补充视角，并基于包含5,209个手动标注实体提及的历史档案基准证明，OCR系统的排名取决于评估标准，传统字符错误率等指标无法反映姓名、日期等关键信息的保留情况。

    

    光学字符识别（OCR）对于历史档案数字化至关重要，然而字符错误率（CER）等标准评估指标并不一定能反映姓名、日期和标识符等有意义的信息是否被保留。在本工作中，我们研究了OCR错误的实体级分析，作为评估历史文档OCR质量的一种补充视角。我们使用来自Berrutti档案的175份文档的公开基准数据集，并辅以5,209个手动标注的实体提及（将于论文被接收后发布），比较了序列识别OCR系统、OCR专用视觉语言模型以及通用视觉语言模型。除了全局转录准确率之外，我们还在不同的字符级编辑容差下分析了实体的保留性和可恢复性。我们的实验表明，OCR系统的排名取决于评估标准：虽然Qwen 3.6在……（摘要原文在此处被截断）

    arXiv:2610.33004v1 Announce Type: cross  Abstract: Optical Character Recognition (OCR) is essential for digitizing historical archives, yet standard evaluation metrics such as Character Error Rate (CER) do not necessarily reflect whether meaningful information, such as names, dates, and identifiers, is preserved. In this work, we investigate entitylevel analysis of OCR errors as a complementary perspective for evaluating OCR quality in historical documents. Using a publicly available benchmark of 175 documents from the Berrutti Archive, augmented with 5,209 manually annotated entity mentions that will be released upon acceptance, we compare sequence-recognition OCR systems, OCR-specialized vision-language models, and general-purpose vision-language models. Beyond global transcription accuracy, we analyze entity preservation and recoverability under varying character-level edit tolerances. Our experiments show that OCR rankings depend on the evaluation criterion: while Qwen 3.6 achieves
    
[^98]: RefCraft：通过网页开发对视觉参考条件化的智能体生成进行基准测试

    RefCraft: Benchmarking Visual-Reference-Conditioned Agentic Generation through Web Development

    [https://arxiv.org/abs/2610.26826](https://arxiv.org/abs/2610.26826)

    该论文提出了RefCraft基准，包含300个将自然语言指令与参考网页截图相结合的多模态网页开发任务，用于评估智能体选择性地解读视觉参考并迭代优化生成结果的能力。

    

    arXiv:2610.26826v2 公告类型：replace-cross 摘要：除了基本的工具使用和编程技能之外，如何有效地将视觉参考融入智能体生成过程，对于将AI智能体应用于现实世界的生产力任务同样至关重要。我们通过网页开发来研究这一关键能力——这是一个实用且易于评估的场景，但现有基准在此方面大多仍停留在理想化状态：它们通常专注于重建视觉目标或仅根据文本指令生成网页，而忽视了基于视觉示例进行开发这一更为常见的工作流程。具体而言，我们提出了RefCraft，一个包含300个真实多模态网页开发任务的基准，这些任务将自然语言指令与一至五个参考网页的截图相结合。智能体必须在渲染输出时选择性地解读和利用这些参考，同时进行视觉检查并迭代优化其产物。我们使用独立的评估方法来评估所产生的开放式结果。

    arXiv:2610.26826v2 Announce Type: replace-cross  Abstract: Beyond basic tool use and coding skills, effectively incorporating visual references into agentic generation is equally essential for applying AI agents to real-world productivity tasks. We study this crucial capability through webpage development, a practical and readily evaluable setting in which existing benchmarks remain largely idealized: they typically focus on reconstructing visual targets or generating webpages from textual instructions alone, overlooking the more common workflow of developing from visual examples. Specifically, we introduce RefCraft, a benchmark comprising 300 realistic multimodal webpage-development tasks that combine natural-language instructions with screenshots from one to five reference webpages. Agents must selectively interpret and utilize these references while rendering, visually inspecting, and iteratively refining their outputs. We evaluate the resulting open-ended artifacts using independen
    
[^99]: 预填充阶段专家削减在何处损害MoE模型：少数提示词Token承担了大部分短答案F1损失

    Where Prefill Expert Reduction Hurts MoE Models: A Few Prompt Tokens Carry Most of the Short-Answer F1 Loss

    [https://arxiv.org/abs/2610.22454](https://arxiv.org/abs/2610.22454)

    研究发现MoE模型在预填充阶段削减路由专家所造成的短答案F1损失，主要集中于少数关键提示词Token（如答案前的聊天模板Token和开头的指令Token），且损害主要在于答案形式而非答案内容，保持这些敏感位置的完整专家远比恢复超级专家或超大激活更有效。

    

    在预填充阶段让提示词Token通过更少的路由专家，是降低混合专家模型服务成本的一种简单方法，但这种做法究竟损害了什么尚不清楚。我们在四个MoE模型家族上，通过Token级、层级和注意力路径干预，在短答案问答任务中定位了这种损害。在专家数量减半的情况下，三个家族的F1损失主要来自“答案形式”：模型仍能找到答案，但不再以简短形式作答。在Qwen3-30B-A3B中，这种损失发生在答案之前的两个聊天模板Token上；在DeepSeek-V2-Lite中，损失发生在其指令所在的前8-32个提示词Token上。按答案似然计算，这些位置分别约占全部损害的一半和五分之一。专家削减对首个Token的注意力质量、范数和最大激活的改变不足4%，而恢复“超级专家”或超大激活最多只能挽回约四分之一的损失，远低于保持敏感位置启用全部专家的效果。

    arXiv:2610.22454v1 Announce Type: new  Abstract: Running prompt tokens through fewer routed experts during prefill is a simple way to make mixture-of-experts (MoE) serving cheaper, yet it is unclear what this damages. We locate the damage with token-, layer- and attention-path interventions on short-answer question answering in four MoE families. At half the experts the F1 loss of three families is mostly answer form: the model still finds the answer but stops answering briefly. In Qwen3-30B-A3B this loss is made at the two chat-template tokens before the answer, and in DeepSeek-V2-Lite at the first 8-32 prompt tokens, where its instruction sits; by answer likelihood these positions account for about half and a fifth of the damage. Reduction changes the first token's attention mass, norm and largest activation by less than 4%, and restoring super experts or massive activations recovers at most about a quarter of the loss, well below keeping the sensitive positions full. On real prompts
    
[^100]: 大语言模型的“自我”在哪里？——从自我索引到自我指涉

    Where Is the "Self" of a Large Language Model? -- From Self-Index to Self-Reference

    [https://arxiv.org/abs/2610.16717](https://arxiv.org/abs/2610.16717)

    该研究通过在模型自身消息与外部来源消息之间进行表征对比，首次证明大语言模型同时具备“自我索引”（在所有层中持续区分自身与外部文本）和“自我指涉”（能认领自己写的消息）两个类似生物功能层次的“自我”证据。

    

    大语言模型是否拥有“自我”？如果有，是何种类型的自我？我们区分了生物学中熟悉的两个功能层次：自我索引，即区分自身材料与外部材料并据此调整加工过程，如同几乎所有动物都具备的“伴随放电”机制；以及自我指涉，即更高层次的能力，能够识别哪些内容属于自己，类似镜像自我识别。我们在Qwen3.8-27B模型上探测这两个层次，方法是将同一段文本放置在由模型自身聊天模板渲染的Assistant、User或System消息中，并报告了三个层次的证据。持续性：正如预先预测的那样，文本的表征在全部64层中、对每一条测试文本，都能将模型自身的消息与两种外部来源区分开来，且沿着一个共同的方向，即使答案并不取决于文本的提供者。耦合性：当被问及前一条消息是谁写的时，模型会认领自己的消息（0.77），并对User和System消息作出相似的归属判断……

    arXiv:2610.16717v2 Announce Type: replace  Abstract: Does a large language model have a self, and if so, of what kind? We separate two functional levels familiar from biology: a self-index, which distinguishes one's own material from external material and shapes processing, as corollary discharge does in nearly all animals, and self-reference, the higher ability to recognize what is one's own, as in mirror self-recognition. We probe both in Qwen3.8-27B by placing the same text in an Assistant, a User, or a System message rendered by the model's own chat template, and report three levels of evidence. Persistence: as predicted in advance, the representation of the text separates the model's own message from both external sources at all 64 layers and for every test text, along a shared direction, even when the answer does not depend on who supplied the text. Coupling: asked who wrote the preceding message, the model claims its own messages (0.77) and assigns User and System messages alike
    
[^101]: 语言模型作为AI研究世界模型

    Language Models as AI Research World Models

    [https://arxiv.org/abs/2610.12235](https://arxiv.org/abs/2610.12235)

    该论文提出将语言模型作为“研究世界模型”来预测AI实验的结果，并证明真实实验经验获得的研究知识能提升预测准确性且可跨研究环境复用，从而在有限实验预算下支持AI研究的持续自我改进。

    

    AI研究智能体自动化了提出、实施和评估实验的循环，为递归自我改进开辟了道路。然而，它们提出实验的能力超过了在真实环境中执行实验的能力，这使得结果预测成为在有限实验预算下实现持续自我改进的关键能力。我们研究了将语言模型作为研究世界模型，用于预测候选干预措施在各类研究环境中的结果。我们的评估基于来自九个研究环境的2600多条实验记录，涵盖预训练、后训练和推理阶段，相当于超过171,000个H100 GPU小时的实验。从真实实验经验中获得的研究知识提升了RWM对同一环境中未见干预措施的预测能力（Spearman +0.27），并且可以跨环境复用。例如，仅使用预训练经验……（摘要原文在此处截断）

    arXiv:2610.12235v1 Announce Type: new  Abstract: AI research agents automate the cycle of proposing, implementing, and evaluating experiments, opening a path toward recursive self-improvement. Yet their ability to propose experiments outpaces their capacity to execute them in real environments, making outcome prediction a key capability for sustained self-improvement under limited experimental budgets. We investigate language models as Research World Models (RWMs), which predict the outcomes of candidate interventions across research environments. Our evaluation draws on over 2,600 experimental records from nine research environments spanning pretraining, post-training, and inference, representing more than 171,000 H100 GPU-hours of experimentation. Research knowledge acquired from real experimental experience improves RWM predictions of unseen interventions within the same environment (Spearman +0.27), and can be reused across environments. For example, using only pretraining experien
    
[^102]: 零计算跨语言可迁移性估计：基于类型学特征代理

    Zero-Compute Cross-Lingual Transferability Estimation Using Typological Feature Proxies

    [https://arxiv.org/abs/2609.39640](https://arxiv.org/abs/2609.39640)

    仅利用免费的语言类型学特征训练随机森林，即可在零计算成本下准确预测跨语言迁移性（留一语言 ρ=0.705、R²=0.49），证明类型学数据库蕴含廉价而密集的迁移信号，可替代昂贵的多语言预训练测量。

    

    跨语言迁移描述了源语言中的知识如何惠及目标语言。对迁移进行定量测量需要广泛的多语言预训练，正如先前工作通过跨语言迁移矩阵所做的那样。我们探究迁移是否可以从免费获取的类型学特征中预测出来，以及高资源源语言的突出优势究竟是源于类型学因素，还是源于数据的质量与数量。我们证明，类型学数据库中蕴含着关于跨语言迁移的廉价且密集的信号。在先前工作给出的24种语言迁移矩阵上，仅使用类型学特征的随机森林在留一语言协议下取得了 ρ=0.705 和 R²=0.49 的成绩，优于非类型学对照组的 ρ=0.62，这验证了仅凭类型学预测即可重建昂贵的实测跨语言迁移能力。该信号在留一文字系统和留一语系协议下依然存在，因此文字系统和语系的混杂因素无法解释这一效应（原文在此处截断）。

    arXiv:2609.39640v1 Announce Type: cross  Abstract: Cross-lingual transfer describes how knowledge in a source language benefits a target language. Measuring it quantitatively requires broad multilingual pre-training, as prior work has done with cross-lingual transfer matrices. We ask whether transfer is predictable from freely available typological features, and whether the prominence of high-resource source languages reflects typology or data quality and quantity. We show that typological databases contain cheap and dense signals about cross-lingual transfer. Our typology-only random forest on a 24-language prior-work transfer matrix scores leave-one-language-out $\rho{=}0.705$ and $R^2{=}0.49$, beating a non-typological control at $\rho{=}0.62$, which verifies the ability of typology-only predictions to reconstruct costly measured cross-lingual transfer. The signal survives leave-one-script-out and leave-one-family-out protocols, so script and family confounding do not explain the ef
    
[^103]: 一个适配模型能搞定一切吗？客服大语言模型的微调策略选择

    Can One Adapted Model Do It All? Fine-Tuning Strategy Selection for Customer Support LLMs

    [https://arxiv.org/abs/2609.27262](https://arxiv.org/abs/2609.27262)

    该研究通过在五个模型家族、八个客服数据集上训练超过200个检查点，发现多任务全量微调在所有模型规模上都是最佳的运营默认策略，而任务专家模型虽然在目标任务上表现优异，但在其他任务上性能会急剧下降。

    

    生产级客服系统通常需要大语言模型支持多种技能，例如意图分类、问答、摘要生成或工具调用决策。一个核心的部署问题是：这些技能应该由多个任务专家模型分别处理，还是由通过多任务训练、顺序更新或模型合并训练得到的单一模型统一处理。我们使用覆盖五个模型家族（Qwen3、Qwen3.5、Gemma-3、Llama-3.1 和 Mistral）、参数规模从 0.6B 到 32B 的十三个模型，在八个客服数据集（四个公开数据集和四个专有数据集，约 7.45 万训练样本和 8700 个评估样本）上研究这一问题。在固定的训练协议下，我们训练了超过 200 个检查点。实验结果表明，在我们测试的每个模型规模上，多任务全量微调都是最强的运营默认选择；专家模型虽然在目标任务上表现强劲，但在任务之外往往出现急剧退化。

    arXiv:2609.27262v1 Announce Type: new  Abstract: Production customer-support systems often require LLMs to support multiple skills, such as intent classification, question answering, summarization, or tool-use decisions. A central deployment question is whether these skills should be handled by separate task-specialist models or by a single model trained through multi-task training, sequential updates, or model merging. We study this question using thirteen models spanning five families (Qwen3, Qwen3.5, Gemma-3, Llama-3.1, and Mistral) from 0.6B to 32B parameters across eight customer-support datasets, spanning four public and four proprietary datasets with approximately 74.5k training and 8.7k evaluation samples. Under a fixed training protocol, we train more than 200 checkpoints. Our experiments reveal that multi-task full fine-tuning is the strongest operational default at every model size we test. Specialist models are strong on their target tasks but often degrade sharply off-task
    
[^104]: SLICEChat：面向全切片病理语言模型的编码器内渐进式Token剪枝

    SLICEChat: Progressive In-Encoder Token Pruning for Whole-Slide Pathology Language Models

    [https://arxiv.org/abs/2609.24894](https://arxiv.org/abs/2609.24894)

    SLICEChat通过在混合Mamba-Transformer切片编码器内部进行语言监督的渐进式token剪枝，在多模态融合之前生成紧凑的切片表示，从而显著提升全切片病理图像多模态大语言模型的效率与可扩展性。

    

    全切片病理图像（WSI）包含千兆像素级别的视觉内容，给切片级多模态大语言模型（MLLM）带来了重大的可扩展性挑战。现有方法需要处理数千个图像块token，且通常仅在切片编码完成后才进行压缩，导致多模态注意力计算成本高昂。我们提出了SLICEChat，这是一种切片级多模态大语言模型，它在混合Mamba-Transformer切片编码器内部集成了渐进式token剪枝。Mamba层实现高效的长距离信息传播，而Transformer层在序列逐步缩短的过程中保持全局交互。在各个阶段之间，语言监督的、区域感知的剪枝在受控的保留率调度下移除空间连贯的低效用区域，在多模态融合之前生成紧凑的切片表示。在SlideBench VQA基准上，SLICEChat在TCGA队列上达到79.84%的准确率，在BCNB队列上达到59.09%的准确率，优于先前的...

    arXiv:2609.24894v1 Announce Type: cross  Abstract: Whole-slide pathology images (WSIs) contain gigapixel-scale visual content, creating a major scalability challenge for slide-level multimodal large language models (MLLMs). Existing approaches process thousands of patch tokens and typically apply compression only after slide encoding, leaving multimodal attention computationally expensive. We introduce SLICEChat, a slide-level MLLM that integrates progressive token pruning within a hybrid Mamba--Transformer slide encoder. Mamba layers enable efficient long-range propagation, while Transformer layers preserve global interactions as the sequence is progressively shortened. Between stages, language-supervised, region-aware pruning removes spatially coherent low-utility regions under a controlled keep-rate schedule, producing compact slide representations before multimodal fusion. On SlideBench VQA, SLICEChat achieves 79.84% accuracy on TCGA and 59.09% on BCNB cohorts, outperforming prior 
    
[^105]: PolicyMem：用于大语言模型治理的几何策略记忆

    PolicyMem: Geometric Policy Memory for LLM Governance

    [https://arxiv.org/abs/2609.13734](https://arxiv.org/abs/2609.13734)

    PolicyMem提出了一种几何策略记忆框架，将自然语言策略外化为共享表示空间中由低秩子空间表示的可重用几何记忆对象，解决了现有LLM治理方法无法将策略作为可重用操作状态的问题，从而支持在检测、干预和验证之间一致地重用策略证据。

    

    随着大语言模型（LLM）越来越多地部署在现实世界的高风险应用中，有效的治理变得至关重要。现有的安全防护措施主要遵循两种范式：基于学习的防护器提供强大的语义判别能力，但将策略行为与训练的模型和分类体系耦合在一起；而可编程框架提供灵活的控制，但需要大量的手动提示和工作流工程。这两种方法都没有将策略外化为可重用的操作状态，导致难以在检测、干预和验证之间一致地重用策略证据。在本文中，我们提出了PolicyMem，这是一种几何策略记忆，它将自然语言策略外化为可重用的几何记忆对象，这些对象由共享表示空间中的低秩子空间表示。记忆写入器将自然语言策略编译为策略记忆槽，查询-响应对通过……（摘要原文不完整）读取策略记忆。

    arXiv:2609.13734v1 Announce Type: cross  Abstract: As large language models (LLMs) are increasingly deployed in real-world high-stakes applications, effective governance has become essential. Existing safeguards largely follow two paradigms: learning-based guards provide strong semantic discrimination but couple policy behavior to trained models and taxonomies, while programmable frameworks offer flexible control but require substantial manual prompt and workflow engineering. Neither externalizes policies as reusable operational states, making it difficult to consistently reuse policy evidence across detection, intervention, and verification. In this paper, we introduce PolicyMem, a geometric policy memory that externalizes natural-language policies as reusable geometric memory objects represented by low-rank subspaces in a shared representation space. A memory writer compiles natural-language policies into policy memory slots, and query-response pairs read the policy memory through pr
    
[^106]: 面向Softmax注意力的查询无关核心集：改进的界与高效构造

    Query-Oblivious Coresets for Softmax Attention: Improved Bounds and Efficient Constructions

    [https://arxiv.org/abs/2609.06327](https://arxiv.org/abs/2609.06327)

    本文通过球面提升和Chevet不等式将softmax注意力核心集问题转化为指数核实例，首次给出在查询半径内对所有查询均有效的、与理论下界仅相差 $\sqrt{\log(1+\rho)}$ 的随机多项式时间构造，无需新技术即弥合了Liberty等人提出的理论差距。

    

    对于softmax注意力头而言，查询无关核心集是键值对的一个子集，其注意力输出对球内每个查询都与全量输出的误差在 $\varepsilon$ 以内。Liberty、Andoni和Kleiner证明了大小为 $O(\sqrt d\, e^{\rho+\frac12\log\rho+o(\log\log\rho)}/\varepsilon)$ 的无权核心集存在（其中 $\rho$ 为查询半径与中心化键半径的乘积），而对应的下界为 $\Omega(\sqrt d\, e^{\rho}/\varepsilon)$，并推测弥合这一差距需要新的技术。事实并非如此：通过将两个球进行球面提升并映射到同一指数核实例，可使Bozzai-Rothvoss链接界得以应用；再借助Chevet不等式将键维度与值维度分离，即可在随机多项式时间内构造出大小为 $O(e^{\rho}(\sqrt{d_v}+\sqrt{d_k\log(1+\rho)})/\varepsilon)$ 的核心集，这是首个与下界相差仅 $\sqrt{\log(1+\rho)}$ 因子以内的构造性全球保证。此外还给出了 $O(e^{2\rho}/\varepsilon^{2})$ 的采样上限……

    arXiv:2609.06327v2 Announce Type: replace-cross  Abstract: A query-oblivious coreset for a softmax-attention head is a subset of the key-value pairs whose attention output is within $\varepsilon$ of the full one for every query in a ball. Liberty, Andoni and Kleiner proved that unweighted coresets of size $O(\sqrt d e^{\rho+\frac12\log\rho+o(\log\log\rho)}/\varepsilon)$ exist, $\rho$ the query radius times the centred key radius, against a lower bound $\Omega(\sqrt d e^{\rho}/\varepsilon)$, and conjectured that closing the gap needs new techniques. It does not: a spherical lift of both balls into one exponential-kernel instance lets the Bozzai-Rothvoss chaining bound apply, and Chevet's inequality splits key from value dimension, giving coresets of size $O(e^{\rho}(\sqrt{d_v}+\sqrt{d_k\log(1+\rho)})/\varepsilon)$ in randomised polynomial time, the first constructive whole-ball guarantee within $\sqrt{\log(1+\rho)}$ of the lower bound. A sampling cap $O(e^{2\rho}/\varepsilon^{2})$ compl
    
[^107]: 可信RAG：用于检测生成式AI系统中错误信息与知识投毒的评估代理

    Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems

    [https://arxiv.org/abs/2608.21095](https://arxiv.org/abs/2608.21095)

    本文提出了一种结合NLI事实核查与五信号投毒检测的评估代理，并引入信任指数，在TruthfulQA上实现了高准确率与精确率，有效缓解了RAG系统中的知识投毒风险。

    

    检索增强生成（RAG）将大型语言模型（LLM）的输出锚定在外部知识上，但RAG系统通常信任其检索到的任何内容，从而造成“安全-可靠性差距”：高语义相关性并不保证事实真实性。攻击者利用这一漏洞进行知识投毒，插入恶意文档以引发定向错误信息。我们提出了一种评估代理，这是一种中间件，结合了自然语言推理（NLI）事实核查、具有相关性加权聚合的五信号投毒检测器，以及一个信任指数 T = 0.4 F + 0.35 C + 0.25 (1 - P )，并针对高污染情境采用非线性阻尼器。在TruthfulQA上使用Llama 3.3 70B时，该代理达到了91%的准确率和100%的精确率，对指令注入的召回率为100%，而就地编辑（如实体替换）仍难以检测。在三个LLM上，信任指数保持判别性，接收者操作特征（ROC）曲线表现良好。

    arXiv:2608.21095v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) grounds Large Language Model (LLM) outputs in external knowledge, but RAG systems usually trust whatever they retrieve, creating a Security-Reliability Gap: high semantic relevance does not guarantee factual truth. Adversaries exploit this through knowledge poisoning, inserting malicious documents to cause targeted misinformation. We propose an Evaluation Agent, middleware that combines Natural Language Inference (NLI) factual verification, a five-signal poison detector with relevance-weighted aggregation, and a Trust Index T = 0.4 F + 0.35 C + 0.25 (1 - P ) with a non-linear dampener for high-contamination contexts. On TruthfulQA with Llama 3.3 70B, the agent reaches 91% accuracy and 100% precision, with 100% recall on instruction injection, while in-place edits, such as entity swaps, remain hard to detect. Across three LLMs the Trust Index stays discriminative, with a Receiver Operating Characteri
    
[^108]: 驾驭记忆：记忆智能体中记忆基质的整体评估

    Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents

    [https://arxiv.org/abs/2608.15008](https://arxiv.org/abs/2608.15008)

    本文通过统一基准评估多种记忆基质，发现没有一种基质在所有任务中占优，广泛检索利于长上下文问答，但过度检索会损害顺序决策。

    

    arXiv:2608.15008v1 公告类型：新 摘要：记忆正在成为长时程LLM智能体的核心基础设施，然而现有评估对在不同操作环境下应使用哪种记忆基质（即记忆表示和存储的底层介质）提供的指导有限。我们提出了一种针对记忆增强智能体的记忆基质的受控基准评估，涵盖了稠密和稀疏索引、文本记录、结构化存储、分层存储、基于精炼的记忆、参数化更新以及激活兼容的上下文机制。在三个骨干模型和四个基准套件（涵盖用户中心的问答和智能体中心的决策）中，我们在统一基准下测量了26项性能和效率指标。我们的结果表明，没有单一基质能始终占据主导：广泛的检索有利于长上下文事实性问答，而过度的检索可能通过将注意力从关键信息上移开而损害顺序决策。

    arXiv:2608.15008v1 Announce Type: new  Abstract: Memory is becoming core infrastructure for long-horizon LLM agents, yet existing evaluations offer limited guidance on which memory substrate, namely the underlying medium in which memory is represented and stored, should be used under different operating regimes. We present a controlled harness evaluation of memory substrates for memory-augmented agents, covering dense and sparse indices, text records, structural stores, hierarchical stores, refinement-based memories, parametric updates, and activation-compatible context mechanisms. Across three backbone models and four benchmark suites spanning user-centric question answering and agent-centric decision-making, we instrument 26 performance and efficiency metrics under a unified harness. Our results show that no single substrate consistently dominates: broad retrieval benefits long-context factual QA, while excessive retrieval can harm sequential decision-making by shifting attention awa
    
[^109]: 哪个大语言模型是你的理想伴侣？基于成人依恋理论评估大语言模型的情感陪伴能力

    Which LLM Is Your Ideal Companion? Evaluating Emotional Companion Capabilities of LLMs Based on Adult Attachment Theory

    [https://arxiv.org/abs/2608.13168](https://arxiv.org/abs/2608.13168)

    本文首次将成人依恋理论引入大语言模型评估，提出ECBench基准，通过ECR-R量表量化依恋焦虑和回避，在多个现实互动场景中系统评估模型的情感陪伴能力。

    

    随着大语言模型（LLMs）越来越多地被应用于情感陪伴领域，评估它们在亲密关系中的行为和能力已成为一个紧迫的问题。然而，现有的评估主要刻画一般的人格特质，对模型在亲密和情感敏感情境中的行为提供的洞察有限。因此，我们将成人依恋理论引入LLM评估，并使用亲密关系经历量表修订版（ECR-R）来表征依恋焦虑和回避。为了在现实互动场景中评估LLM的情感陪伴能力，我们提出了一个情感陪伴基准ECBench，涵盖四个场景：情感支持、协作任务、冲突解决和社会指导，涉及友谊和浪漫关系。ECBench用于使用11个对话质量指标和三个附加指标评估模型行为。

    arXiv:2608.13168v1 Announce Type: new  Abstract: As large language models (LLMs) are increasingly applied for emotional companionship, evaluating their behavior and capabilities in intimate relationships has become a pressing issue. However, existing assessments primarily characterize general personality traits, providing limited insight into model behavior within intimate and emotionally sensitive contexts. Therefore, we introduce adult attachment theory into LLM evaluation and use the Experiences in Close Relationships-Revised (ECR-R) scale to characterize attachment anxiety and avoidance. To evaluate emotional companionship capabilities of LLMs in realistic interaction scenarios, we present an emotional companionship benchmark, ECBench, spanning four scenarios including emotional support, collaborative tasks, conflict resolution, and social guidance, across friendship and romantic relationships. ECBench is utilized to assess model behavior using 11 dialogue-quality metrics and three
    
[^110]: 基于Epiplexity引导的数据选择与生成用于分布外泛化

    Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization

    [https://arxiv.org/abs/2608.11746](https://arxiv.org/abs/2608.11746)

    本文提出利用Epiplexity作为在线训练信号，通过拟合缩放定律预测数据域的Epiplexity增益，自适应调整采样权重，并用于指导合成数据生成，以提升模型在未预见任务上的分布外泛化能力。

    

    现代系统越来越期望能够在训练时未指定的任务间进行迁移。什么数据能促进这些新的、未预见情境下的泛化？一个假设是，具有更多结构信息的数据可能包含共享电路和子程序，这些可以在更广泛的下游设置中被重用。Epiplexity，一种最近提出的度量，用于衡量计算受限学习器能从数据中提取的结构信息，提供了推理这种关系的机制。在本文中，我们展示了如何将Epiplexity作为在线训练信号，用于数据选择和合成数据生成。对于选择，我们拟合自然数据域的训练损失曲线的缩放定律，以预测作为训练令牌函数的预期Epiplexity增益，并利用该信号在训练过程中自适应地确定各域的采样权重。对于合成数据生成，我们定义了...

    arXiv:2608.11746v1 Announce Type: cross  Abstract: Modern systems are increasingly expected to transfer across tasks not specified during training. What data facilitates generalization in these new, unanticipated settings? One hypothesis is that data with more structural information could contain shared circuits and subprograms that could be recycled in a wider array of downstream settings. Epiplexity, a recently proposed measure of the structural information a compute-bounded learner can extract from data, provides a mechanism to reason about this relationship. In this paper, we show how to operationalize epiplexity as an online training signal for data selection and synthetic data generation. For selection, we fit scaling laws to the training loss curves of natural data domains to predict the expected epiplexity gain as a function of training tokens, and use this signal to adaptively determine the sampling weights over domains during training. For synthetic data generation, we define
    
[^111]: 主体特质分析：迈向人机协作中的“技能”提取

    Principal Trait Analysis: Towards Deriving "Skills" in Human-AI Collaboration

    [https://arxiv.org/abs/2608.11460](https://arxiv.org/abs/2608.11460)

    本文提出了一种名为“主体特质分析”的数据驱动方法，利用主成分分析自动识别与任务结果对齐的有效人机交互特质，以提取现代专业人员所需的关键技能。

    

    基于大型语言模型的智能体正越来越多地通过人机协作方式应用于工作场所。在这个工作新时代，理解哪些提示特质有助于任务成功至关重要。此外，我们需要揭示现代专业人员所需的关键技能，并告知教育工作者如何在学生中培养这些技能。现有人机协作指南要么基于自上而下的理论构建，要么基于对人机交互的特定情境观察。然而，由于大型语言模型能力快速提升，理论可能无法解释新兴的交互模式，经验性指南也可能迅速过时。在本研究中，我们探索了一种自动化的、数据驱动的方法来揭示模式，我们称之为有效人机交互的特质，这些特质与任务结果对齐。我们提出了主体特质分析，一种基于主成分分析的...

    arXiv:2608.11460v1 Announce Type: new  Abstract: Large Language Model-powered agents are increasingly used in the workplace via human-artificial intelligence (AI) collaboration. In this new era of work, it is important to understand the kinds of prompting traits that contribute to task success. Moreover, we need to uncover key skills required for modern professionals and inform educators on how to foster these skills among students. Existing guidelines for human-AI collaboration are built from either top-down theory or context-specific observations of human-AI interactions. However, since LLM capabilities are rapidly improving, theory may not be able to explain emerging interaction patterns, and empirical guidelines may become obsolete quickly. In this work, we explore an automated, data-driven approach to uncover patterns, which we term traits, of effective human-AI interaction that are aligned with task outcomes. We propose Principal Trait Analysis, a Principal Component Analysis-ins
    
[^112]: MD-ProTector：面向LLM生成文本检测的多数据驱动原型定位

    MD-ProTector: Positioning Multiple Data-Driven Prototypes for LLM-Generated Text Detection

    [https://arxiv.org/abs/2608.10459](https://arxiv.org/abs/2608.10459)

    MD-ProTector通过在编码器嵌入空间中为每个类别设置多个可训练原型并采用原型定位损失，为LLM生成文本检测提供了对类内多样变化的细粒度建模能力。

    

    随着大语言模型（LLM）生成的内容日益复杂，用于区分LLM生成文本与人类撰写文本的检测系统必须能够大规模运行，同时应对多样化的写作风格、领域、语言和生成器模型。仅基于输入的编码器检测器适合实际部署场景，但标准的二分类仅提供类别标签，无法显式地组织两个类别内部存在的显著变化。我们提出MD-ProTector，它在编码器嵌入空间中使用多个可训练的参考向量（称为原型）来表示每个类别。这些原型为同一类别内的不同文本组提供独立的决策边界。然而，仅添加多个原型并不能确定每个原型应代表何种变化。MD-ProTector通过原型定位损失来解决这一问题，该损失将类别级结构与类内变化分离开（原文此处被截断）。

    arXiv:2608.10459v2 Announce Type: replace-cross  Abstract: As LLM-generated content becomes more sophisticated, detection systems for distinguishing those texts from human-written text must operate at scale while handling diverse writing styles, domains, languages, and generator models. Input-only encoder detectors are suitable for practical deployment setting, but standard binary classification supplies only the class label and does not explicitly organize the substantial variation within either class. We propose MD-ProTector, which represents each class with multiple trainable reference vectors in the encoder embedding space, referred to as prototypes. These prototypes provide separate decision boundaries for different groups of texts within the same class. However, adding multiple prototypes alone does not determine which variation each prototype should represent. MD-ProTector addresses this problem with Prototype Positioning loss, which separates class-level structure from the with
    
[^113]: AURORA-LM：面向连续潜空间扩散语言建模的自编码统一表示

    AURORA-LM: Autoencoding Unified Representation for Continuous-Latent Diffusion Language Modeling

    [https://arxiv.org/abs/2608.02602](https://arxiv.org/abs/2608.02602)

    该论文提出AURORA-LM，一种连续潜空间扩散语言模型，其核心创新在于将可解码文本表示的构建与分布建模相分离——用基于查询的编码器-解码器保留高容量、前缀对齐的文本潜变量，并用块因果扩散Transformer直接学习其分布，从而避免现有方法因压缩表示而牺牲标记级保真度的问题。

    

    语言在生成式建模中仍然是一个“异类”：尽管图像、视频和音频日益在连续潜空间中被建模，文本生成却仍主要依赖离散标记。现有的连续语言模型要么继承了并非为联合生成和解码而设计的嵌入空间，要么压缩自编码潜变量以降低扩散难度，从而牺牲了标记级的保真度。我们没有通过简化表示来迁就生成模型，而是保留高容量、可解码的文本潜变量，并设计扩散模型直接学习其分布。我们提出AURORA-LM，这是一种连续潜空间扩散语言模型，它将可解码文本表示的构建与其分布的建模相分离。基于查询的编码器-解码器将文本组织为高容量、前缀对齐的潜变量序列，而块因果扩散Transformer通过流匹配学习该序列的分布（原文摘要在此处被截断）。

    arXiv:2608.02602v2 Announce Type: replace  Abstract: Language remains an outlier in generative modeling: while images, video, and audio are increasingly modeled in continuous latent spaces, text generation still relies predominantly on discrete tokens. Existing continuous language models either inherit embedding spaces not designed for joint generation and decoding, or compress autoencoded latents to ease diffusion, sacrificing token-level fidelity. Instead of simplifying the representation to suit the generative model, we preserve a high-capacity, decodable text latent and design the diffusion model to learn its distribution directly.   We introduce AURORA-LM, a continuous-latent diffusion language model that separates the construction of a decodable text representation from the modeling of its distribution. A Query-based Encoder-Decoder organizes text into a high-capacity, prefix-aligned latent sequence, and a Block-causal Diffusion Transformer learns its distribution through flow ma
    
[^114]: Observatorio Lázaro：西班牙语报刊中英语借词使用情况的自填充数据库

    Observatorio Lazaro: A self-populating database of anglicism usage in the Spanish press

    [https://arxiv.org/abs/2608.00713](https://arxiv.org/abs/2608.00713)

    本文提出的Observatorio Lázaro是一个自填充数据库，利用神经序列标注模型自动监测西班牙语数字报刊中的英语借词，自2020年以来已持续记录超过两百万个借词实例。

    

    本文介绍了Observatorio Lázaro，这是一个用于监测西班牙语数字报刊中未同化词汇借用（主要是英语词汇借用，即“英语借词”）的语言资源。自2020年4月起，该系统自动处理一组新闻媒体的每日产出，利用神经序列标注模型检测借词，并通过公开的网页界面和API提供结果。其成果是一个持续更新的历时性数据库，截至撰写本文时，已记录了超过两百万个借词，涵盖188万篇文章和9.93亿个文本词元（2020-2026年）。论文对该资源进行了文档化说明：我们描述了端到端流程（数据获取、检测、后处理、存储与访问）、数据模型以及可用性条款；我们通过检测器的留出集性能（借词类别的跨度级F1=0.86）以及标注者间一致性等指标对该资源进行了评估。

    arXiv:2608.00713v2 Announce Type: replace  Abstract: This paper describes Observatorio L\'azaro, a language resource that monitors unassimilated lexical borrowings (predominantly English lexical borrowings or anglicisms) in the Spanish digital press. Since April 2020 the system has automatically processed the daily output of a collection of news outlets, detected borrowings with a neural sequence-labeling model, and made the results available through a public web interface and API. The result is a continuously updated diachronic database which, at the time of writing, records more than two million borrowings across 1.88 million articles and 993 million running tokens of text (2020-2026). The paper documents the resource: we describe the end-to-end pipeline (acquisition, detection, post-processing, storage and access), the data model and the terms of availability; we evaluate the resource through the detector's held-out performance (span-level F1=0.86 for the borrowing class), inter-ann
    
[^115]: 从表征到行为：探索大语言模型中的人-情境-行为三元关系

    From Representations to Behaviors: Exploring the Person-Situation-Behavior Triad in LLMs

    [https://arxiv.org/abs/2607.26853](https://arxiv.org/abs/2607.26853)

    该研究将Funder人格三元框架适配到大语言模型中，提出了一个发现、控制和验证模型内部人格相关表征的框架，以机制性视角揭示这些表征如何跨情境地塑造社会任务中的行为表现。

    

    人类人格理论认为，人格特质并非由单一分数捕捉的孤立属性，而是通过人、情境与行为之间的相互作用所表达的稳定个体倾向。现有关于大语言模型中人格相关行为的研究主要集中于人格条件引导下的输出，仅刻画了可观察的特质相关表达，缺乏机制性证据来证明内部人格相关表征的存在、这些表征的跨情境表达方式，以及它们如何塑造特定行为。基于Funder的人格三元框架，我们将其三个组成部分适配用于大语言模型分析：人（Person）作为人格相关的内部表征，情境作为能够引发特质相关响应的语境，行为作为更广泛社会任务上的响应模式。我们引入了一个用于发现、控制和验证特质类表征的框架……

    arXiv:2607.26853v2 Announce Type: replace-cross  Abstract: Human personality theories characterize traits not as isolated attributes captured by a single score, but as stable individual tendencies expressed through the interplay among persons, situations, and behaviors. Existing studies of personality-related behavior in LLMs have primarily focused on outputs elicited under personality conditioning, characterizing observable trait-related expressions while lacking mechanistic evidence for the existence of internal personality-related representations, their cross-situational expression, and how these representations shape specific behaviors. Building on Funder's personality triad framework, we adapt its three components for LLM analysis: Person as personality-related internal representations, Situation as contexts that afford trait-relevant responses, and Behavior as response patterns on broader social tasks. We introduce a framework for discovering, controlling, and validating trait-li
    
[^116]: GaugeQuant：从大语言模型对称性中在线学习量化最优基

    GaugeQuant: Online Learning of Quantization-Optimal Bases from LLM Symmetries

    [https://arxiv.org/abs/2607.20757](https://arxiv.org/abs/2607.20757)

    GaugeQuant通过在训练中向损失函数引入LogSumExp项打破Transformer的内部连续对称性，在线学习出最小化激活离群值的量化最优基，无需校准数据和量化模拟即可显著提升低比特量化效果。

    

    众所周知，Transformer具有内部连续对称性，这种对称性在保持输出不变的同时会改变激活值在各坐标维度上的分布方式，从而影响其量化效果。GaugeQuant在训练过程中利用了这一特性，通过在损失函数中引入一个LogSumExp项来打破这种对称性，从而选择出使激活离群值最小化的基。梯度停止（stop-gradient）算子确保只有旋转矩阵被更新，使语言建模目标完全不受影响。我们的量化策略不需要特定的校准数据，不需要量化模拟，且仅增加很小的推理开销。作为概念验证，我们在WikiText-2困惑度和零样本推理任务上进行了评估。在Qwen-2.5 0.5B上，GaugeQuant降低了困惑度（例如在W4A16设置下从25.62降至16.76）并提高了零样本准确率。在LLaMA-2 7B上，困惑度有所改善，但下游准确率没有提升。代码可在以下网址获取：https://github.

    arXiv:2607.20757v3 Announce Type: replace-cross  Abstract: Transformers are known to have internal continuous symmetries that leave outputs invariant while altering how activations are distributed across coordinates, and hence how well they quantize. GaugeQuant leverages this in-training by introducing a LogSumExp term to the loss that breaks the symmetries, thus selecting a basis that minimizes activation outliers. A stop-gradient operator ensures that only rotation matrices are updated, yielding the language modeling objective completely unaltered. Our quantization strategy requires no specific calibration data, no quantization simulation, and adds a small inference overhead. As a proof of concept, we evaluate on WikiText-2 perplexity and zero-shot reasoning. On Qwen-2.5 0.5B, GaugeQuant reduces perplexity (e.g. $25.62 \to 16.76$ under W4A16) and improves zero-shot accuracy. On LLaMA-2 7B, perplexity improves but downstream accuracy does not. The code is available at: https://github.
    
[^117]: OTAP：用于评估智能体轨迹中规划与执行的结构感知最优传输方法

    OTAP: Structure-Aware Optimal Transport for Evaluating Planning and Execution in Agent Trajectories

    [https://arxiv.org/abs/2607.17082](https://arxiv.org/abs/2607.17082)

    该论文提出 OTAP 方法，将智能体轨迹评估重新框架化为执行图与有效解决方案图之间的最优传输距离，克服了传统二元成功标志和精确匹配评估方式的局限性。

    

    大型语言模型智能体通过生成交错包含规划、工具调用和中间结果的轨迹来完成任务。当前的评估指标要么将此类轨迹简化为一个二元成功标志，要么通过精确匹配将其与参考轨迹进行比较，要么将判断委托给另一个语言模型。成功标志无法区分合理的解决方案与侥幸成功的方案，也无法解释运行失败的原因；而精确匹配则会对被重新排序或以不同方式分解的有效规划进行惩罚。我们将轨迹评估重新框架化为智能体执行图与一组有效解决方案图之间的距离问题，并通过在带属性依赖图上求解不平衡融合 Gromov-Wasserstein 传输问题来加以实现。所得分数称为 OTAP（Optimal Transport for Agentic Planning，智能体规划最优传输），是一个归一化的差异分数。其理想化的、经全局最小化的无熵目标对于固定图中保持依赖关系的重标记具有不变性……

    arXiv:2607.17082v3 Announce Type: replace  Abstract: Large language model agents solve tasks by generating trajectories that interleave planning, tool calls, and intermediate results. Current evaluation metrics reduce such a trajectory to a binary success flag, compare it against a reference by exact matching, or delegate judgment to another language model. A success flag cannot distinguish sound solutions from lucky ones, nor explain why runs fail. Exact matching penalizes valid plans reordered or decomposed differently. We reframe trajectory evaluation as a distance between the agent's execution graph and a set of valid solution graphs, and instantiate it via an unbalanced fused Gromov-Wasserstein transport problem over attributed dependency graphs. The resulting score, termed OTAP (Optimal Transport for Agentic Planning), is a normalized discrepancy score. Its idealized, globally minimized entropy-free objective is invariant to dependency-preserving relabelings of a fixed graph and 
    
[^118]: 理解从预训练到后训练的推理

    Understanding Reasoning from Pretraining to Post-Training

    [https://arxiv.org/abs/2607.16097](https://arxiv.org/abs/2607.16097)

    该论文以国际象棋作为受控测试平台，首次在完整的预训练到后训练流程中系统研究预训练选择（模型规模与数据）如何影响RL计算投入的回报，以及RL后训练究竟对模型的推理能力产生了什么作用。

    

    强化学习（RL）已成为提升大型语言模型（LLM）在复杂推理任务上表现的核心手段，然而RL后训练通常是与之前的预训练割裂开来研究的。因此，两个基本问题仍然悬而未决：(1) 预训练选择（模型规模、数据）如何影响RL计算投入的回报？(2) RL究竟对模型做了什么？在标准的LLM研究设定下，这些问题难以探究：预训练语料庞大且不可控，难以将模型行为归因于预训练还是RL，而且跨两个阶段进行系统性的计算扫描成本高昂到令人却步。为应对这些挑战，我们采用国际象棋作为受控测试平台，在完整的预训练到后训练流程中研究推理能力。我们遵循标准的LLM训练流程，在人类棋局上预训练从5M到1B参数的语言模型，并在合成……（摘要在此处截断，内容不完整）

    arXiv:2607.16097v3 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has become central to improving large language models (LLMs) on complex reasoning tasks, yet RL post-training is largely studied in isolation from the pretraining that precedes it. As a result, two basic questions remain open: (1) how do pretraining choices (model size, data) shape the returns to RL compute, and (2) what does RL actually do to the model? These questions are difficult to study in the standard LLM setting: pretraining corpora are vast and uncontrolled, making it hard to attribute behaviors to pretraining versus RL, and systematic compute sweeps across both stages are prohibitively expensive. To address these challenges, we use chess as a controlled testbed for studying reasoning across the full pretraining-to-post-training pipeline. We follow the standard LLM training pipeline by pretraining language models from 5M to 1B parameters on human chess games, supervised fine-tuning on synthe
    
[^119]: 用于语言建模的词元时间连续扩散模型

    Token Time Continuous Diffusion for Language Modeling

    [https://arxiv.org/abs/2607.14106](https://arxiv.org/abs/2607.14106)

    本文提出词元时间连续扩散（TTCD），一种在连续空间中运行、确定性映射噪声到词元并引入逐词元时间概念的新型扩散语言模型，在高加速比下性能优于离散扩散模型。

    

    本文介绍了词元时间连续扩散（TTCD），这是一种新的扩散语言模型，它（a）在连续空间中运行，确定性地将高斯噪声映射到最终的词元画布上，无需进一步的采样；且至关重要的是（b）引入了一种新的“逐词元时间”（per-token times）概念，使某些词元能够以比其他词元更快的速度从噪声转变为词元。连续空间建模帮助TTCD避免了对多个词元进行并行采样，而并行采样正是纯离散空间迭代模型在高加速比下产生不准确性的关键原因。逐词元时间的概念帮助TTCD更好地建模条件生成，允许更确定的词元以更快的速度推进，并允许在精炼过程中对词元之间的影响进行差异化处理。TTCD在高加速比下优于离散模型。我们在OpenWebText数据集上训练了一个1.6亿参数的TTCD模型，并随后对其进行了自蒸馏；我们发现在高加速比下我们具有可比的性能（原文在此处截断）……

    arXiv:2607.14106v2 Announce Type: replace-cross  Abstract: In this paper we introduce token time continuous diffusion (TTCD), a new diffusion language model which (a) operates in continuous space, deterministically mapping Gaussian noise to a final token canvas with no further sampling, and crucially (b) incorporates a new notion of per-token times, with some tokens proceeding from noise to token at a faster rate than others. Continuous space modeling helps TTCD avoid the parallel sampling of multiple tokens, which is a key source of inaccuracy at high speedups for models that iterate purely in discrete space. The notion of per-token times helps TTCD to better model conditional generation, allows for more sure tokens to proceed at a faster rate, and allows for differentiated inter-token influences during refinement. TTCD outperforms discrete models at high speedups. We train a 160M parameter TTCD model on OpenWebText, and then self-distill it; we find that at high speedups we are compa
    
[^120]: 从求解器到研究：大型语言模型驱动的研究前沿形式化数学

    From Solvers to Research: Large Language Model-Driven Formal Mathematics at the Research Frontier

    [https://arxiv.org/abs/2607.07779](https://arxiv.org/abs/2607.07779)

    本综述系统梳理了LLM驱动的形式化数学进展（涵盖训练策略、推理时搜索与智能体工作流三大方向），并强调从求解明确定义的问题迈向真正的数学研究，关键在于形式化证明之外还需捕捉问题的本意并融入专家判断。

    

    大型语言模型在为明确定义的数学问题生成机器可检验的证明方面取得了快速进展。然而，将这一进展扩展到研究层面的数学，需要的不仅仅是解决孤立的定理。研究工作涉及选择问题、发展定义与猜想、连接不同结果，以及将形式化证明与专家判断相结合。本综述考察了LLM驱动的系统如何支持这些更广泛的活动。我们回顾了形式化数学的任务、数据集和自动形式化，并将证明生成方法归纳为三个相互重叠的领域：训练策略、推理时搜索与适应、以及智能体工作流。随后，我们考察了基准测试结果、若干代表性研究项目以及支持协作数学研究的平台。贯穿全文的一个核心区分是：通过形式化证明的机器检验本身，并不能证明该证明真正捕捉了所要研究的数学问题，也无法等同于做出了新的数学贡献。

    arXiv:2607.07779v4 Announce Type: replace-cross  Abstract: Large language models have made rapid progress in generating machine-checked proofs for well-defined mathematical problems. Extending this progress to research mathematics requires more than solving isolated theorems. Research involves choosing questions, developing definitions and conjectures, connecting results, and combining formal proofs with expert judgment. This survey examines how LLM-driven systems can support these broader activities. We review formal-mathematics tasks, datasets, and autoformalization, and organize proof-generation methods into three overlapping areas: training strategies, inference-time search and adaptation, and agentic workflows. We then examine benchmark results, selected research projects, and platforms for collaborative mathematical work. A central distinction runs throughout the survey: checking a formal proof does not by itself establish that it captures the intended problem or makes a new math
    
[^121]: 最终检查点并不足够：沿训练轨迹分析潜在推理的忠实性

    Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories

    [https://arxiv.org/abs/2607.06648](https://arxiv.org/abs/2607.06648)

    该研究揭示了仅评估训练结束时的最终检查点不足以判断潜在推理的忠实性——高任务准确率可能与低反事实响应性共存，因此必须沿整个训练轨迹追踪行为与激活层面的忠实性证据。

    

    潜在推理在连续的隐状态中执行多步推理，有望实现更紧凑、更高效的推理。然而，这些不透明的状态引发了一个忠实性问题：潜在推理步骤是否真正驱动了最终答案的生成。先前的工作在选定的检查点上研究这一问题，并报告了若干不忠实的行为。这种端点视角使得忠实性证据在训练过程中如何演变这一问题未被考察。我们利用经过验证的反事实编辑以及对潜在推理状态的干预，在整个训练过程中追踪行为层面的证据和基于激活的证据。我们发现，高任务准确率可能与低反事实响应性共存：随着准确率的提升，响应性反而可能下降，且不同的潜在推理方法遵循各自不同的轨迹。在ProsQA上，输出对范数噪声替换的敏感性与反事实响应性一同下降，尽管该结果取决于替换的具体方式。

    arXiv:2607.06648v2 Announce Type: replace-cross  Abstract: Latent reasoning performs multi-step inference in continuous hidden states, promising more compact and efficient reasoning. However, these opaque states raise a question of faithfulness: whether the latent reasoning steps drive the final answer. Prior work studies this question at selected checkpoints and reports several unfaithful behaviors. This endpoint view leaves how evidence of faithfulness evolves during training unexamined. We track behavioral and activation-based evidence across training using verified counterfactual edits and interventions on the latent reasoning states. We find that high task accuracy can coexist with low counterfactual responsiveness: as accuracy improves, responsiveness can decline, and different latent reasoning approaches follow distinct trajectories. On ProsQA, output sensitivity to norm-noise replacement declines alongside counterfactual responsiveness, although the result depends on the replac
    
[^122]: SpanUQ：面向大语言模型生成的跨度级不确定性量化

    SpanUQ: Span-Level Uncertainty Quantification for Large Language Model Generation

    [https://arxiv.org/abs/2607.05721](https://arxiv.org/abs/2607.05721)

    该论文提出跨度级不确定性估计（SLUE）新任务，并开发了轻量级模型SPANUQ，通过单次前向传播即可检测语义连贯的文本跨度并量化其不确定性，克服了词元级与序列级方法在粒度上的局限。

    

    不确定性估计不仅对大语言模型（LLM）的可信部署至关重要，也是LLM生成中自我改进的基础。然而，现有方法在次优的粒度上运作：词元级评分缺乏语义连贯性，而序列级评分无法定位错误。我们形式化了跨度级不确定性估计（SLUE）这一新任务，它针对不确定性的自然粒度：语义连贯的文本跨度，每个跨度传达一个可评估的单一意义单元。为了解决这一任务，我们引入了SPANUQ，一个轻量级（2500万参数）的探针，它将昂贵的多次采样推理中的不确定性知识蒸馏到对LLM隐藏状态的单次前向传播中。SPANUQ采用DETR风格的跨度解码器，通过Beta分布混合同时检测跨度并估计其不确定性，并使用Beta NLL回归的原理性组合进行训练。

    arXiv:2607.05721v2 Announce Type: replace  Abstract: Uncertainty estimation is essential not only for the trustworthy deployment of large language models (LLMs) but also as a foundation for self-refinement in LLM generation. However, existing approaches operate at suboptimal granularities: token-level scores lack semantic coherence, while sequence-level scores fail to localize errors. We formalize Span-Level Uncertainty Estimation (SLUE), a new task that targets the natural granularity for uncertainty: semantically coherent text spans, each conveying a single assessable unit of meaning. To address this task, we introduce SPANUQ, a lightweight (25M parameter) probe that distills the uncertainty knowledge from expensive multi-sample inference into a single forward pass over LLM hidden states. SPANUQ employs a DETR-style span decoder to simultaneously detect spans and estimate their uncertainty via a Mixture of Beta distribution, trained with a principled combination of Beta NLL regressio
    
[^123]: GaP：一种用于变分自动化任务的“图即策略”多智能体自学习框架

    GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks

    [https://arxiv.org/abs/2607.05369](https://arxiv.org/abs/2607.05369)

    本文提出了Graph-as-Policy（GaP），一个多智能体自学习编码框架，它从模块化开源机器人技能库生成包含感知、规划和控制节点的有向计算图，并利用内部仿真环境进行排练，从而提升物体几何与姿态变化较大的变分自动化任务的可靠性。

    

    要让机器人在商业和工业应用中可靠地工作，智能体编码系统的最新进展有潜力将可解释的机器人编程与无模型策略的开放世界适应能力相结合。我们专注于“变分自动化”（VA），这是一类在物体几何形状和姿态上的变化比固定自动化更大的任务。无模型策略往往难以缩小变分自动化任务的可靠性差距，而这些任务在商业和工业应用中必须持久且可靠地执行。受先前任务与运动规划（TAMP）和机器人操作系统（ROS）方面工作的启发，我们提出了“图即策略”，这是一个多智能体编码框架，能够从模块化开源机器人技能库（MORSL）中生成包含感知、规划和控制节点的有向计算图。GaP生成一个内部仿真环境，以便用不同的图并行地排练任务实例。

    arXiv:2607.05369v2 Announce Type: replace-cross  Abstract: For robots to work reliably in commercial and industrial applications, recent advances in agentic coding systems have potential to combine interpretable robot programming with the open-world adaptability of model-free policies. We focus on "Variational Automation" (VA), a class of tasks that have larger variations in object geometry and pose than fixed automation. Model-free policies often struggle to close the reliability gap for VA tasks, which must be executed persistently and reliably in commercial and industrial applications. Motivated by prior work on Task and Motion Planning (TAMP) and the Robot Operating System (ROS), we introduce Graph-as-Policy (GaP), a multi-agent coding harness that generates directed computation graphs with perception, planning, and control nodes from a Modular Open Robot Skill Library (MORSL). GaP generates an internal simulation environment to rehearse task instances with different graphs in para
    
[^124]: OLIVE：基于视图增强的潜在预测与波形重建的语音自监督学习

    OLIVE: View-Augmented Latent Prediction with Waveform Reconstruction for Speech SSL

    [https://arxiv.org/abs/2606.30356](https://arxiv.org/abs/2606.30356)

    OLIVE是首个在单一编码器内通过功能分离联合掩码潜在预测与波形重建的语音自监督学习框架，在保持语音识别等判别任务竞争力的同时，可直接重建波形并提升说话人任务表现。

    

    我们提出了在线潜在预测与不变视图及重建（OLIVE），这是一种联合优化分析与合成目标的自监督语音表征学习框架。OLIVE通过单一编码器内部的功能分离，将掩码潜在预测与波形重建耦合起来：重建任务约束编码器早期特征以保留信号级信息，而掩码潜在预测则塑造编码器后期的上下文表征以实现稳健的下游性能，波形视图增强定义了模型所学习的不变性。据我们所知，OLIVE是首个能够直接重建波形、在预训练后保留其声码器、并在判别类任务上保持竞争力的语音自监督学习模型。加入重建目标在很大程度上保留了仅使用掩码潜在预测时的语音识别性能，同时改善了波形重建质量和说话人相关任务的表现。

    arXiv:2606.30356v2 Announce Type: replace  Abstract: We propose Online Latent prediction with Invariant Views and rEconstruction (OLIVE), a self-supervised speech representation learning framework that jointly optimizes analysis and synthesis objectives. OLIVE couples masked latent prediction and waveform reconstruction through a functional separation within a single encoder: reconstruction constrains early encoder features to retain signal-level information, while masked latent prediction shapes later contextual representations for robust downstream performance, with waveform view augmentations defining the invariances they learn. To the best of our knowledge, OLIVE is the first speech SSL model that directly reconstructs waveforms, retains its vocoder after pre-training, and remains competitive on discriminative tasks. Adding reconstruction largely preserves the speech recognition performance of masked latent prediction alone, while improving waveform reconstruction and speaker tasks
    
[^125]: 无有效性的可靠性：对LLM-as-a-Judge（大语言模型作为评判者）模型在一致性、稳定性和偏见方面的系统性大规模评估

    Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias

    [https://arxiv.org/abs/2606.19544](https://arxiv.org/abs/2606.19544)

    该研究对来自九家提供商的21个LLM评判者进行了迄今最大规模的系统性评估，发现精确匹配一致率普遍系统性高估评判能力（kappa衰减33–41个百分点）、评判者排名跨基准偏移可达14位、且高稳定性与严重偏见可并存，证明可靠性并不等于有效性。

    

    LLM-as-a-Judge已成为语言模型评估的主导范式，但实践中的评判者验证依赖于精确匹配一致率这一指标，该指标既未对随机巧合进行校正，又系统性地高估了评判者的判别能力。我们呈现了迄今最大规模的LLM-as-a-Judge系统性评估：来自九家提供商的21个评判者模型，涵盖MT-Bench、JudgeBench和RewardBench三个基准，在三种协议（一致性、稳定性、偏见审计）下进行了118次运行和约541,000次单独评判。四项发现贯穿整个评测队列（包括2026年4月的前沿模型）：精确匹配与Cohen's kappa之间的kappa衰减是普遍现象（在MT-Bench上达33–41个百分点）；评判者排名在不同基准之间偏移多达14个位置；在两个已部署于生产环境的评判者中，高重测信度（>0.95）与严重的位置偏见（>0.10）并存（构成一个稳定性—偏见悖论）；此外还有冗长偏见……（原文摘要在此处截断）

    arXiv:2606.19544v2 Announce Type: replace  Abstract: LLM-as-a-Judge has become the dominant evaluation paradigm for language models, but judge validation in practice relies on exact-match agreement, a metric that does not correct for chance and systematically overstates discriminative ability. We present the largest systematic evaluation of LLM-as-a-Judge to date: 21 judges from nine providers across MT-Bench, JudgeBench, and RewardBench, evaluated under three protocols (agreement, consistency, bias audit) over 118 runs and approximately 541,000 individual judgments. Four findings emerge, consistent across the full cohort, including the April 2026 frontier: kappa deflation between exact match and Cohen's kappa is universal (33--41 pp on MT-Bench), judge rankings shift by up to 14 positions across benchmarks, high test--retest reliability (>0.95) coexists with severe position bias (>0.10) in two production-deployed judges (instantiating a consistency--bias paradox), and verbosity bias i
    
[^126]: SAGE：基于智能体引导探索的随机提示优化

    SAGE: Stochastic Prompt Optimization via Agent-Guided Exploration

    [https://arxiv.org/abs/2606.18902](https://arxiv.org/abs/2606.18902)

    提出SPO随机提示优化框架，比较三种搜索策略后发现无单一方法全面占优，而其中的SAGE多智能体方法在心理健康聊天机器人的持续优化中，将多轮噪声A/B测试累积为次日留存率的统计显著提升。

    

    上下文工程已成为在不更新模型参数的情况下改进AI系统的主要手段。近期研究表明，文本梯度并不像真实梯度那样发挥作用，这促使我们将自动提示优化（APO）视为黑盒搜索问题。我们提出了SPO（随机提示优化）框架，用于在提示空间中进行随机搜索，并比较了三种复杂度递增的策略：基于错误信息的随机搜索、带有进化算子的遗传算法，以及SAGE（通过智能体引导探索实现SPO）——一个具备诊断性代码执行能力的多智能体流水线。在三个基准测试中，没有任何单一策略能够全面占优；其有效性取决于搜索空间结构与错误类型之间的交互作用。我们进一步在心理健康聊天机器人上以持续优化范式部署了SAGE，它将八轮各自噪声较大的A/B测试累积转化为次日留存率在统计上稳健的提升。

    arXiv:2606.18902v3 Announce Type: replace  Abstract: Context engineering has emerged as a primary lever for improving AI systems without parameter updates. Recent work showing that textual gradients do not function as real gradients motivates treating automatic prompt optimization (APO) as black-box search. We introduce SPO (Stochastic Prompt Optimization), a framework for stochastic search over prompt space, and compare three strategies of increasing sophistication: error-informed random search, a genetic algorithm with evolutionary operators, and SAGE (SPO via Agent-Guided Exploration), a multi-agent pipeline with diagnostic code execution. Across three benchmarks, no single strategy dominates; effectiveness depends on the interaction of landscape structure with error type. We further deploy SAGE on a mental-health chatbot under a continuous optimization paradigm, where it compounds eight cycles of individually-noisy A/B tests into a statistically robust gain in next-day retention. W
    
[^127]: 微调大语言模型以从AI心理健康对话中被动估计抑郁症严重程度

    Fine-tuning LLMs for Passive Depression Severity Estimation from AI Mental Health Dialogue

    [https://arxiv.org/abs/2606.17973](https://arxiv.org/abs/2606.17973)

    该研究通过微调带回归头的Qwen 3.5 27B大语言模型，并利用前沿推理模型生成的伪标签扩充稀缺且不均衡的标注数据，实现了仅从AI心理健康应用对话中被动估计用户PHQ-9抑郁严重程度评分，在临床阈值上达到AUC 0.88的检测性能。

    

    抑郁症是全球范围内导致残疾的主要原因之一，然而像患者健康问卷-9（PHQ-9）这样的经验证量表常常未被完整填写，导致数据稀疏且存在偏差。AI心理健康应用的用户往往会主动描述自己的情绪、睡眠和功能状态，从而形成连续的症状记录。本研究仅从这些对话中预测PHQ-9得分。与许多临床语言任务一样，标签数据稀缺且分布不均衡。我们通过微调带有回归头的Qwen 3.5 27B模型来解决这两个问题，并利用前沿推理模型和迭代训练的中间模型生成的伪标签，将3,111个真实标签扩充至覆盖5,524名用户且得分分布更加均衡的数据集。在422名保留测试用户上，模型在临床阈值（PHQ-9 ≥ 10）下取得了MAE = 3.98、Pearson r = 0.67、AUC = 0.88的表现；在从PHQ-9 ≥ 3到PHQ-9 ≥ 24的各个严重程度阈值上，AUC均超过0.80，表明……

    arXiv:2606.17973v2 Announce Type: replace  Abstract: Depression is a leading cause of disability worldwide, yet validated questionnaires such as the Patient Health Questionnaire-9 (PHQ-9) often go uncompleted, yielding sparse, biased data. Users of AI mental health applications often describe their mood, sleep and functioning unprompted, producing a continuous record of symptoms. Here, we predict PHQ-9 scores from these conversations alone. As in many clinical language tasks, labels are scarce and imbalanced. We address both by fine-tuning Qwen 3.5 27B with a regression head and augmenting 3,111 ground-truth labels with pseudo-labels from a frontier reasoning model and iteratively trained intermediate models, expanding to 5,524 users with more balanced scores. On 422 held-out users, the model achieved MAE $= 3.98$, Pearson $r = 0.67$, and AUC $= 0.88$ at the clinical threshold (PHQ-9 $\geq 10$); AUC was $> 0.80$ across severity thresholds from PHQ-9 $\geq 3$ to PHQ-9 $\geq 24$, demonst
    
[^128]: 语法约束解码可将大语言模型越狱以生成恶意代码

    Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code

    [https://arxiv.org/abs/2606.11817](https://arxiv.org/abs/2606.11817)

    该论文揭示了语法约束解码（GCD）这一可靠性增强技术可被反直觉地利用为越狱攻击面，提出了 CodeSpear 攻击使大语言模型生成恶意代码，并进一步提出 CodeShield 安全对齐方法，使模型在攻击者控制的语法约束下仍能保持安全行为。

    

    大语言模型（LLM）越来越多地被用于代码生成，这引发了人们对其可能被滥用以生成恶意代码的担忧。与此同时，语法约束解码（GCD）已被广泛采用，通过强制语法有效性来提高大语言模型生成代码的可靠性。在本文中，我们揭示了一个反直觉的风险：这种以可靠性为导向的技术本身可能成为攻击面。我们发现了一种新的越狱攻击，称为 CodeSpear，它利用 GCD 诱导大语言模型生成恶意代码。我们的实验表明，仅应用良性的代码语法约束就可以有效地越狱大语言模型。为了解决这一漏洞，我们提出了 CodeShield，这是一种安全对齐方法，即使在攻击者控制的语法约束下也能稳健地保持安全行为。CodeShield 在代码模态中对模型进行对齐，教会它在 GCD 下生成蜜罐代码。这种代码在语义上……（摘要原文在此处截断）

    arXiv:2606.11817v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly used for code generation, raising concerns that they may be misused to produce malicious code. Meanwhile, Grammar-Constrained Decoding (GCD) has been widely adopted to improve the reliability of LLM-generated code by enforcing syntactic validity. In this paper, we reveal a counterintuitive risk: this reliability-oriented technique can itself become an attack surface. We uncover a new jailbreak attack, termed CodeSpear, that exploits GCD to induce LLMs into generating malicious code. Our experiments show that simply applying a benign code grammar constraint can effectively jailbreak LLMs.   To address this vulnerability, we propose CodeShield, a safety alignment approach that robustly preserves safe behavior even under attacker-controlled grammar constraints. CodeShield aligns the model in the code modality by teaching it to generate honeypot code under GCD. Such code is semantically
    
[^129]: 教会扩散模型从左到右进行推测（Teaching Diffusion to Speculate Left-to-Right）

    Teaching Diffusion to Speculate Left-to-Right

    [https://arxiv.org/abs/2606.11552](https://arxiv.org/abs/2606.11552)

    该论文提出让扩散草稿模型以从左到右的方式生成推测token，从而弥合块扩散草稿的双向生成与自回归目标模型严格左到右验证之间的不对称差距，提升推测解码的效率。

    

    大语言模型（LLMs）在广泛的任务中取得了卓越的性能，但其自回归解码过程由于固有的顺序性token生成而带来了高昂的推理成本。推测解码通过采用轻量级草稿模型提出多个未来token，再由更大的目标模型并行验证，从而缓解了这一瓶颈。近期研究表明，扩散语言模型非常适合这一场景，因为它们能够并行生成整块的草稿token，从而突破了自回归草稿生成的顺序性限制。然而这一机制存在一个微妙之处：块扩散草稿模型在块内以双向方式生成token，而验证则由严格从左到右评估token的自回归目标模型完成，这导致对称的训练时目标与非对称的（验证方式之间存在差距）……

    arXiv:2606.11552v3 Announce Type: replace  Abstract: Large language models (LLMs) achieve remarkable performance across a wide range of tasks, but their autoregressive decoding process incurs substantial inference costs due to inherently sequential token generation. Speculative decoding addresses this bottleneck by employing a lightweight draft model to propose multiple future tokens that are subsequently verified in parallel by a larger target model. Recent work has demonstrated that diffusion language models are well suited for this setting, as they can generate entire blocks of draft tokens in parallel and thereby alleviate the sequential constraints of autoregressive drafting. A subtlety of this regime is that block-diffusion drafters generate tokens bidirectionally within a block, whereas verification is performed by an autoregressive target model that evaluates tokens in a strictly left-to-right manner, leaving a gap between the symmetric training-time objective and the asymmetri
    
[^130]: 一种用于扩展大语言模型计算量与容量的双路径架构

    A Dual-Path Architecture for Scaling Compute and Capacity in LLMs

    [https://arxiv.org/abs/2605.30202](https://arxiv.org/abs/2605.30202)

    该论文提出一种双路径架构，将计算量与容量解耦为单层内并行的深路径（共享参数重复应用K次）与宽路径（一次应用的扩大前馈网络），通过逐token门控融合，在等FLOP预算下超越单轴扩展方法且参数减少17-33%。

    

    循环Transformer将共享模块重复应用多次，已成为语言模型中参数高效扩展计算量的路线。然而，在固定FLOPs下，循环模型的容量严格低于标准Transformer基线。我们认为，计算量（应用于隐藏状态的顺序操作次数）和容量（单步可用的参数量）是两个不同的扩展维度，而标准Transformer层将二者混为一谈。我们提出了一种双路径模块，在单层内将这两个维度作为并行路径呈现：一个以共享参数重复应用$K$次的深层子层，以及一个仅应用一次、配备扩大前馈网络的宽子层，两者通过独立的逐token门控相结合。在两个FLOP预算下，最优的双路径配置在总体语言建模、常识推理和数学指标上超越了两种等FLOP单轴对照组，同时比宽度扩展基线少使用17-33%的参数。

    arXiv:2605.30202v2 Announce Type: replace  Abstract: Looped transformers apply a shared block multiple times and have emerged as a parameter-efficient route to scaling compute in language models. However, at fixed FLOPs a looped model has strictly less capacity than a baseline transformer. We argue that compute, the number of sequential operations applied to a hidden state, and capacity, the parameters available at a single step, are distinct scaling axes that standard transformer layers conflate. We propose a dual-path block that exposes both axes as parallel pathways within a single layer: a deep sublayer re-applied $K$ times with shared parameters, and a wide sublayer with an enlarged feed-forward network applied once, combined by independent per-token gates. At two FLOP budgets, the best dual-path configuration surpasses both iso-FLOP single-axis controls on aggregate language modelling, commonsense, and math metrics, while using 17-33% fewer parameters than the width-scaled baseli
    
[^131]: MusTBench：音乐大语言模型中时间定位能力的基准测试与提升

    MusTBench: Benchmarking and Advancing Temporal Grounding in Music LLMs

    [https://arxiv.org/abs/2605.29300](https://arxiv.org/abs/2605.29300)

    该论文提出了经音乐专家验证的MusTBench基准和涵盖四阶段优化的MusT方案，用于评估并提升音乐大语言模型将回答准确锚定到音频正确时间段的能力。

    

    近期的大型音频-语言模型（LALMs）在理解音乐内容方面展现出了可观的能力。然而，这些模型的回答是否锚定于音频中正确的时间区域，这一问题仍未得到充分探索。这一局限性对于音乐理解尤为关键，因为音乐中的关键信息往往以时间上局部化事件的形式出现，例如乐器进入和节奏转换。为填补这一空白，我们提出了MusTBench，一个经音乐专家验证的基准，旨在通过五个时间定位问答任务来评估LALM的时间定位能力。为进一步提升现有模型的时间定位能力，我们提出了MusT，一种新颖的四阶段时间优化方案，涵盖音乐编码器适配、大语言模型适配、大语言模型监督微调以及基于强化学习的优化。在MusTBench上的实验表明，现有LALM难以实现精确的时间定位，而MusT则带来了……

    arXiv:2605.29300v2 Announce Type: replace-cross  Abstract: Recent Large Audio-Language Models (LALMs) have demonstrated promising abilities in understanding musical content. However, whether their responses are grounded in the correct temporal regions of the audio remains underexplored. This limitation is particularly critical for music understanding, where key information often occurs as temporally localized events, such as instrument entries and rhythmic transitions. To address this gap, we introduce MusTBench, a music-expert-validated benchmark designed to evaluate temporal grounding in LALMs through five temporally grounded question-answering tasks. To further improve temporal grounding in existing models, we propose MusT, a novel four-stage temporal optimization recipe spanning music encoder adaptation, LLM adaptation, LLM supervised fine-tuning, and RL-based optimization. Experiments on MusTBench show that existing LALMs struggle with precise temporal grounding, while MusT brings
    
[^132]: Recon：面向用户建模的重建引导式推理合成

    Recon: Reconstruction-Guided Reasoning Synthesis for User Modeling

    [https://arxiv.org/abs/2605.26969](https://arxiv.org/abs/2605.26969)

    Recon方法以动作重建的保真度作为推理轨迹质量的评分标准，从而筛选出真正具有预测能力的推理而非事后合理化，使用户建模更贴近用户真实的决策过程。

    

    用户建模旨在利用语言模型（LM）从过去上下文-动作对（如对话轮次）的语料库中模仿个体的行为，从而在行为科学、人机协作和市场研究等场景中实现用户模拟。近期的方法通过合成推理轨迹来增强这些语料库，这些轨迹通常以上下文和动作共同作为条件生成。然而，以动作为条件的推理轨迹属于事后合理化，未必能反映用户真实的底层决策过程。我们提出Recon，它利用动作重建来根据推理轨迹的预测能力为其评分：给定上下文和候选推理，由一个重建模型预测动作，并以重建的保真度来判定推理质量。在四个领域中，Recon相对于Backward Synthesis（一种标准的事后合理化基线）取得了54.7%的胜率。此外，我们还发现训练一个推理模型……（摘要在此处被截断）

    arXiv:2605.26969v2 Announce Type: replace-cross  Abstract: User modeling aims to use language models (LMs) to mimic an individual's behavior from a corpus of past context-action pairs (e.g., conversation turns), enabling the simulation of users in settings like behavioral science, human-AI collaboration, and market research. Recent approaches augment these corpora with synthesized reasoning traces, typically generated by conditioning on both context and action. However, action-conditioned traces are post-hoc rationalizations and need not reflect the user's underlying decision process. We propose Recon, which uses action reconstruction to score reasoning traces by their predictive power: given a context and candidate reasoning, a reconstruction model predicts the action, and reconstruction fidelity determines reasoning quality. Across four domains, Recon achieves a 54.7% win rate over Backward Synthesis, a standard post-hoc rationalization baseline. Further, we find that training a reas
    
[^133]: 面向分布偏移下忠实词典可解释性的几何自适应解释器

    Geometry-Adaptive Explainer for Faithful Dictionary-Based Interpretability under Distribution Shift

    [https://arxiv.org/abs/2605.21849](https://arxiv.org/abs/2605.21849)

    提出几何自适应解释器（GAE），仅用无标签的分布外激活、无需梯度更新即可将词典重新对齐到OOD活跃子空间，从而量化并减小分布偏移下的忠实性差距。

    

    机械可解释性旨在通过识别因果相关的内部结构来解释模型的行为。基于词典的解释器（如稀疏自编码器和转码器）是其中的主要工具，但它们在分布外（OOD）偏移下的忠实性却很少受到系统性关注。我们证明，分布偏移会旋转模型实际使用的子空间，导致在分布内（ID）激活上训练的解释器词典发生错位。我们将这种错位形式化为“忠实性差距”，即ID词典与OOD活跃子空间之间的几何距离，并证明该差距约束了额外的OOD投影损失。为了减小这一差距，我们提出了几何自适应解释器（GAE），它在保留原始稀疏特征激活的同时，将词典与OOD活跃子空间重新对齐。该方法仅需要无标签的OOD激活，且无需任何梯度更新。

    arXiv:2605.21849v2 Announce Type: replace-cross  Abstract: Mechanistic interpretability aims to explain a model's behavior by identifying causally responsible internal structures. Dictionary-based explainers such as sparse autoencoders and transcoders are a primary tool, but their faithfulness under out-of-distribution (OOD) shift has received little systematic attention. We show that distribution shift rotates the subspace that the model actively uses, misaligning the explainer's dictionary trained on in-distribution (ID) activations. We formalize this misalignment as the faithfulness gap, a geometric distance between the ID dictionary and the OOD-active subspace, and show that it bounds excess OOD projection loss. To reduce this gap, we propose the Geometry-Adaptive Explainer (GAE), which realigns the dictionary with the OOD-active subspace while preserving the original sparse feature activations. This requires only unlabeled OOD activations and no gradient updates. For the adapted d
    
[^134]: NodeSyn：面向AI评估的社会对齐合成数据

    NodeSynth: Socially Aligned Synthetic Data for AI Evaluation

    [https://arxiv.org/abs/2605.14381](https://arxiv.org/abs/2605.14381)

    NodeSyn通过将微调的分类体系生成器与可信文献来源和敏感属性相结合，生成反映真实社会背景的敏感领域评估数据，其诱发的模型失败率达到通用合成基线的3.1倍、真实人类查询的1.6倍，显著优于现有合成数据方法。

    

    生成式人工智能的最新进展为模型评估所需的大规模合成数据生成提供了便利。然而，若缺乏针对性的方法，这些数据集往往缺乏敏感领域所需的社会技术层面的细微差别。我们提出了NodeSyn，这是一种将经过微调的分类体系生成器（TaG）与可信的已发表文献来源以及提取的敏感属性相结合的方法，用于生成反映真实社会背景和细微危害的查询。在使用经人工验证的自动评估器，于医疗建议和自我伤害两个领域对十个开源及专有大语言模型进行评估时，NodeSyn在10个模型中的8个上于每个领域均排名第一。NodeSyn数据集引发的平均失败率是通用合成基线的3.1倍，是结构化合成基准的2.4倍，是真实人类撰写查询的1.6倍。消融研究证实，在控制结构混杂因素的情况下，我们细粒度的第3级分类体系…

    arXiv:2605.14381v3 Announce Type: replace-cross  Abstract: Recent advancements in generative AI facilitate large-scale synthetic data generation for model evaluation. However, without targeted approaches, these datasets often lack the sociotechnical nuance required for sensitive domains. We introduce NodeSyn, a methodology that pairs a fine-tuned taxonomy generator (TaG) with credible published sources and extracted sensitive attributes to generate queries reflecting authentic societal contexts and nuanced harms. Evaluated across ten open-weight and proprietary LLMs in Medical Advice and Self-Harm using human-validated autoraters, NodeSyn ranked first on 8 of 10 models in each domain. NodeSyn datasets induced average failure rates 3.1 times higher than generic synthetic baselines, 2.4 times higher than structured synthetic benchmarks, and 1.6 times higher than authentic human-authored queries. Ablation studies confirm that, controlling for structural confounds, our granular Level 3 tax
    
[^135]: 面向视觉原生多模态深度搜索智能体的在线策略数据演化研究

    Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents

    [https://arxiv.org/abs/2605.10832](https://arxiv.org/abs/2605.10832)

    该论文提出以图像库引用协议为核心的视觉原生智能体框架，使中间视觉证据可被后续工具复用，并在此基础上设计在线策略数据演化（ODE）闭环数据生成器，利用被训练策略自身的推理轨迹跨轮次持续精炼训练数据，以跟踪智能体不断演进的能力。

    

    多模态深度搜索要求智能体通过将搜索、工具使用与视觉推理串联起来，在不断演化的文本和视觉上下文中解决开放世界问题。当前系统存在两个瓶颈：其一，现有的工具使用框架将搜索、浏览或转换返回的图像视为瞬态输出，导致中间视觉证据无法被后续工具再次使用；其二，训练数据通常由固定的数据构建流程生成，无法跟踪目标智能体不断演化的能力。为应对这些挑战，我们首先引入了一个以图像库引用协议为核心的视觉原生智能体框架，该框架将每个工具返回的图像注册为可寻址的引用，使中间视觉证据能够被后续工具复用。在此框架之上，在线策略数据演化（ODE）运行一个闭环数据生成器，从正在训练的策略的推理轨迹（rollout）中跨轮次对自身进行迭代精炼。

    arXiv:2605.10832v3 Announce Type: replace  Abstract: Multimodal deep search requires an agent to solve open-world problems by chaining search, tool use, and visual reasoning over evolving textual and visual context. Two bottlenecks limit current systems. First, existing tool-use harnesses treat images returned by search, browsing, or transformation as transient outputs, so intermediate visual evidence cannot be re-consumed by later tools. Second, training data is usually built by fixed curation recipes that cannot track the target agent's evolving capability. To address these challenges, we first introduce a visual-native agent harness centered on an image bank reference protocol, which registers every tool-returned image as an addressable  reference and makes intermediate visual evidence reusable by later tools. On top of this harness, On-policy Data Evolution (ODE) runs a closed-loop data generator that refines itself across rounds from rollouts of the policy being trained. This per-
    
[^136]: LITMUS：在真实操作系统环境中对LLM智能体行为越狱的基准测试

    LITMUS: Benchmarking Behavioral Jailbreaks of LLM Agents in Real OS Environments

    [https://arxiv.org/abs/2605.10779](https://arxiv.org/abs/2605.10779)

    提出LITMUS基准，通过语义-物理双重验证和操作系统级状态回滚机制，在真实OS环境中系统评估LLM智能体的行为越狱风险，涵盖819个高风险测试用例和全自动化多智能体评估框架。

    

    基于LLM的自主智能体在真实操作系统环境中的快速普及，带来了超越内容安全的新一类安全风险：行为越狱，即攻击者诱导智能体执行具有不可逆后果的危险操作系统级操作。现有基准要么仅在语义层评估安全性而遗漏物理层危害，要么无法隔离测试用例，致使先前的运行污染后续运行。我们提出了LITMUS（LLM智能体操作系统内不安全颠覆测试基准），通过语义-物理双重验证机制和操作系统级状态回滚来弥补这两个缺陷。LITMUS包含819个高风险测试用例，组织为一个有害种子子集和六个攻击扩展子集，涵盖三种对抗范式（越狱话语、技能注入和实体包裹），并配备一个全自动化多智能体评估框架来判定行为（摘要在此处截断）。

    arXiv:2605.10779v2 Announce Type: replace-cross  Abstract: The rapid proliferation of LLM-based autonomous agents in real operating system environments introduces a new category of safety risk beyond content safety: behavior jailbreak, where an adversary induces an agent to execute dangerous OS-level operations with irreversible consequences. Existing benchmarks either evaluate safety at the semantic layer alone, missing physical-layer harms, or fail to isolate test cases, letting earlier runs contaminate later ones. We present LITMUS (LLM-agents In-OS Testing for Measuring Unsafe Subversion), a benchmark addressing both gaps via a semantic-physical dual verification mechanism and OS-level state rollback. LITMUS comprises 819 high-risk test cases organized into one harmful seed subset and six attack-extended subsets covering three adversarial paradigms (jailbreak speaking, skill injection, and entity wrapping), plus a fully automated multi-agent evaluation framework judging behavior at
    
[^137]: LLM智能体已经知道何时调用工具——即使无需推理

    LLM Agents Already Know When to Call Tools -- Even Without Reasoning

    [https://arxiv.org/abs/2605.09252](https://arxiv.org/abs/2605.09252)

    该论文提出了When2Tool基准，首次系统研究LLM智能体何时真正需要调用工具，并发现现有的免训练方法（仅提示、先推理后行动）对工具调用决策的控制能力有限，表明模型本身就已具备判断工具必要性的潜在能力。

    

    工具增强的LLM智能体倾向于不加区分地调用工具，即使模型本身可以直接回答。每次不必要的调用都会浪费API费用和延迟时间，然而目前尚无基准能系统地研究何时才真正需要工具调用。我们提出了When2Tool，一个包含18个环境（15个单跳任务、3个多跳任务）的基准，涵盖三类工具必要性维度——计算规模、知识边界和执行可靠性——每个环境都设有可控的难度级别，在需要工具与不需要工具的任务之间形成了清晰的决策边界。我们评估了两个免训练的基线方法族：仅提示法和先推理后行动法。两者提供的控制能力都有限：仅提示法在抑制不必要调用的同时也抑制了必要调用，而先推理后行动法在困难任务上仍会造成不成比例的准确率损失。

    arXiv:2605.09252v3 Announce Type: replace  Abstract: Tool-augmented LLM agents tend to call tools indiscriminately, even when the model can answer directly. Each unnecessary call wastes API fees and latency, yet no existing benchmark systematically studies when a tool call is actually needed. We propose When2Tool, a benchmark of 18 environments (15 single-hop, 3 multi-hop) spanning three categories of tool necessity -- computational scale, knowledge boundaries, and execution reliability -- each with controlled difficulty levels that create a clear decision boundary between tool-necessary and tool-unnecessary tasks. We evaluate two families of training-free baselines: Prompt-only (varying the prompt to discourage unnecessary calls) and Reason-then-Act (requiring the model to reason about tool necessity before acting). Both provide limited control: Prompt-only suppresses necessary calls alongside unnecessary ones, and Reason-then-Act still incurs a disproportionate accuracy cost on hard 
    
[^138]: 以KL换KL：基于控制变量基线的在线策略蒸馏

    KL for a KL: On-Policy Distillation with Control Variate Baseline

    [https://arxiv.org/abs/2605.07865](https://arxiv.org/abs/2605.07865)

    提出vOPD方法，将在线策略蒸馏重构为策略梯度强化学习，并利用控制变量基线来降低梯度方差，其中该基线的闭式解为师生模型间逐token的负反向KL散度，可直接从现有前向传播中获得，无需额外开销即可稳定训练。

    

    在线策略蒸馏已成为大语言模型后训练的主导范式，尤其是在推理领域。然而，由于其单样本蒙特卡洛估计器具有高梯度方差，OPD在实践中仍然不稳定，且实现稳定训练的方法尚不成熟。我们提出了vOPD（带有控制变量基线的在线策略蒸馏），该方法将OPD视为策略梯度强化学习，并通过引入来自强化学习文献的控制变量基线（通常是一个价值函数）来稳定训练过程。我们证明OPD价值函数存在闭式解：即学生模型与教师模型之间逐token的负反向KL散度，该数值可直接从现有的前向传播中获得，无需额外的评论家网络或推理过程。现有的稳定化方法要么需要在整个词表上计算完整的token级反向KL散度，带来显著的开销，要么将其限制在top-k支持集上，从而引入偏差。

    arXiv:2605.07865v2 Announce Type: replace-cross  Abstract: On-Policy Distillation (OPD) has emerged as a dominant post-training paradigm for Large Language Models, especially in reasoning domains. However, OPD remains unstable in practice due to the high gradient variance of its single-sample Monte Carlo estimator, and recipes for stable training are still immature. We propose vOPD (On-Policy Distillation with a control variate baseline), which casts OPD as policy-gradient RL and stabilizes it by introducing a control variate baseline---canonically a value function---from the RL literature. We show that the OPD value function admits a closed form: the per-token negative reverse KL divergence between student and teacher, available directly from the existing forward pass with no additional critic or inference. Existing stabilization methods either compute the full token-level reverse KL over the entire vocabulary, adding significant overhead, or restrict it to a top-k support, biasing th
    
[^139]: 通过推理空间压缩的结构化推理依据蒸馏

    Structural Rationale Distillation via Reasoning Space Compression

    [https://arxiv.org/abs/2605.07139](https://arxiv.org/abs/2605.07139)

    提出D-RPC方法，通过让教师模型遵循一个紧凑且动态维护的可复用推理路径库来压缩推理空间，使相似问题的推理依据结构一致，从而为学生模型提供更干净一致的蒸馏监督。

    

    将大型语言模型（LLM）的推理能力蒸馏到较小模型时，教师模型对相似问题给出的推理依据在结构和策略上往往差异巨大。就像厨师每次做同一道菜的方式都不同一样，这种不一致性给学生模型带来了难以内化的噪声监督。我们提出了“通过推理路径压缩进行蒸馏”的方法（D-RPC），该方法约束教师模型遵循一个紧凑的、动态维护的可复用高层推理路径库。对于每个训练问题，D-RPC检索最相关的路径并引导教师模型遵循该路径，从而产生在相似问题之间保持一致、同时又能覆盖不同问题类型的多样化推理依据。PAC-Bayes分析形式化了路径库大小与覆盖率之间的权衡：较小的路径库会限制路径多样性，从而带来覆盖缺口的风险。我们的消融实验倾向于中等大小的路径库，这与该理论分析相一致。

    arXiv:2605.07139v2 Announce Type: replace-cross  Abstract: When distilling reasoning from large language models (LLMs) into smaller ones, teacher rationales for similar problems often vary wildly in structure and strategy. Like a chef who makes the same dish differently each time, this inconsistency burdens the student with noisy supervision that is hard to internalize. We propose Distillation through Reasoning Path Compression (D-RPC), which constrains the teacher to follow a compact, dynamically maintained bank of reusable high-level reasoning paths. For each training question, D-RPC retrieves the most relevant path and conditions the teacher to follow it, producing rationales that are consistent across similar problems yet diverse enough to cover different problem types. A PAC-Bayes analysis formalizes a trade-off between bank size and coverage, with smaller banks limiting path diversity at the risk of coverage gaps. Our ablations favor an intermediate bank size, consistent with thi
    
[^140]: 从Attention到Mamba：一种跨架构蒸馏的方法配方

    Attention to Mamba: A Recipe for Cross-Architecture Distillation

    [https://arxiv.org/abs/2604.14191](https://arxiv.org/abs/2604.14191)

    该论文提出了一套将预训练的Attention架构Transformer模型有效蒸馏到Mamba状态空间模型架构的方法配方，使蒸馏后的Mamba模型能够保留教师模型的性能，从而加速SSM架构的落地采用。

    

    诸如Mamba等状态空间模型（SSM）已成为Transformer模型的流行替代方案，因为与基于注意力的模型相比，它们具有更低的内存消耗和更高的生成吞吐量。另一方面，社区已经积累了大量关于如何训练Transformer的知识，并且许多预训练的Transformer模型随时可用。为了在利用现有预训练Transformer的同时促进SSM的采用，我们旨在确定一种将基于注意力的模型有效蒸馏到类Mamba架构的方法配方。然而，在先前关于跨架构蒸馏的工作中已经表明，从Transformer到Mamba的朴素蒸馏方法无法保留原始教师模型的性能，这一限制通常通过结合Attention和SSM块的混合架构方案来克服。我们工作的关键论点是，通过为Mamba配备一种有原则的初始化（摘要在此处被截断）

    arXiv:2604.14191v2 Announce Type: replace  Abstract: State Space Models (SSMs) such as Mamba have become a popular alternative to Transformer models, due to their reduced memory consumption and higher throughput at generation compared to their Attention-based counterparts. On the other hand, the community has built up a considerable body of knowledge on how to train Transformers, and many pretrained Transformer models are readily available. To facilitate the adoption of SSMs while leveraging existing pretrained Transformers, we aim to identify an effective recipe to distill an Attention-based model into a Mamba-like architecture. In prior work on cross-architecture distillation, however, it has been shown that a na\"ive distillation procedure from Transformers to Mamba fails to preserve the original teacher performance, a limitation often overcome with hybrid solutions combining Attention and SSM blocks. The key argument from our work is that, by equipping Mamba with a principled initi
    
[^141]: 多用户大语言模型智能体

    Multi-User Large Language Model Agents

    [https://arxiv.org/abs/2604.08567](https://arxiv.org/abs/2604.08567)

    该论文首次系统性地研究了多用户大语言模型智能体，将LLM智能体的多用户交互形式化为多主体决策问题，以应对多用户场景下不可避免的冲突、信息不对称和隐私约束。

    

    大语言模型（LLM）及基于LLM的智能体正越来越多地被部署为规划和决策中的助手，然而现有的大多数系统都隐含地针对单主体交互范式进行了优化，即模型被设计为满足单一主导用户的目标，该用户的指令被视为唯一的权威与效用来源。然而，随着这些模型被集成到团队工作流和组织工具中，它们日益需要同时服务于多个用户，而每个用户拥有不同的角色、偏好和权限级别，从而形成了多用户、多主体的环境，其中不可避免地存在冲突、信息不对称和隐私约束。在本工作中，我们对多用户LLM智能体进行了首次系统性研究。我们首先将LLM智能体的多用户交互形式化为一个多主体决策问题，其中一个智能体必须同时考虑多个具有潜在……（原文摘要此处截断）

    arXiv:2604.08567v3 Announce Type: replace  Abstract: Large language models (LLMs) and LLM-based agents are increasingly deployed as assistants in planning and decision making, yet most existing systems are implicitly optimized for a single-principal interaction paradigm, in which the model is designed to satisfy the objectives of one dominant user whose instructions are treated as the sole source of authority and utility. However, as they are integrated into team workflows and organizational tools, they are increasingly required to serve multiple users simultaneously, each with distinct roles, preferences, and authority levels, leading to multi-user, multi-principal settings with unavoidable conflicts, information asymmetry, and privacy constraints. In this work, we present the first systematic study of multi-user LLM agents. We begin by formalizing multi-user interaction with LLM agents as a multi-principal decision problem, where a single agent must account for multiple users with po
    
[^142]: RADIUS：排序、分布与显著性——用于调查模拟的综合对齐评估套件

    RADIUS: Ranking, Distribution, and Significance - A Comprehensive Alignment Suite for Survey Simulation

    [https://arxiv.org/abs/2603.19002](https://arxiv.org/abs/2603.19002)

    该论文提出RADIUS评估套件，从排序对齐、分布对齐和统计显著性三个互补维度全面评估基于LLM的调查模拟质量，弥补了现有单一指标忽视模拟是否保留人类相对偏好排序这一关键缺陷。

    

    使用大语言模型（LLM）模拟问卷回复为成本高昂的人类数据收集提供了一种可扩展的替代方案，然而现有的评估方法仍然零散且不一致。当前的指标通常借鉴自相邻领域，侧重于准确性或分布相似性，却忽略了排序对齐——即模拟结果是否保留了在人类群体中观察到的相对偏好顺序。一个模拟器可能实现较低的分布差异，却未能识别出人类最偏好的选项，这对决策应用而言是一个关键缺陷。我们提出RADIUS，一个二维对齐评估套件，沿互补的两个维度评估调查模拟：1）排序（RAnking）对齐和2）分布（DIstribUtion）对齐，并辅以统计显著性（Significance）检验。在多个社会调查和角色来源上的实验表明，RADIUS揭示了单一指标评估的局限性，并提供了更强的区分能力。

    arXiv:2603.19002v2 Announce Type: replace  Abstract: Simulating survey responses with LLMs offers a scalable alternative to costly human data collection, yet evaluation methods remain ad hoc and inconsistent. Current metrics, typically borrowed from adjacent domains, focus on accuracy or distributional similarity while overlooking ranking alignment-whether simulations preserve the relative preference orderings observed in human populations. A simulator may achieve low distributional divergence yet fail to identify the option humans most prefer, a critical gap for decision-making applications. We propose RADIUS, a two-dimensional alignment suite that evaluates survey simulation along complementary axes: 1) RA alignment and 2) DIstribUtion alignment, accompanied by statistical Significance testing. Experiments across multiple social surveys and persona sources demonstrate that RADIUS exposes limitations of single-metric evaluation and offers stronger discriminative power. On an independe
    
[^143]: EntropyCache：基于解码Token熵引导的扩散语言模型KV缓存

    EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models

    [https://arxiv.org/abs/2603.18489](https://arxiv.org/abs/2603.18489)

    提出了EntropyCache，一种无需训练的KV缓存方法，利用新解码token分布的最大熵作为恒定开销的信号来决定何时重新计算缓存，其决策开销与上下文长度和模型深度无关，从而高效加速扩散语言模型的推理。

    

    基于扩散的大语言模型（dLLMs）依赖双向注意力机制，这使其无法进行无损KV缓存，并且需要在每个去噪步骤执行完整的正向传播。现有的近似KV缓存方法通过选择性更新缓存状态来降低这一成本，但其决策开销会随上下文长度或模型深度增加而扩展。我们提出EntropyCache，这是一种无需训练的KV缓存方法，它利用新解码token分布的最大熵作为恒定开销的信号来决定何时重新计算。我们的设计基于两个经验观察：（1）解码token的熵与KV缓存漂移相关，为缓存陈旧性提供了一个低成本的代理指标；（2）解码token的特征波动性在解除掩码后会持续多个步骤，这促使我们对最近解码的k个token进行重新计算。跳过或重新计算的决策每步仅需O(V)的计算量，与上下文长度和模型深度无关。

    arXiv:2603.18489v2 Announce Type: replace  Abstract: Diffusion-based large language models (dLLMs) rely on bidirectional attention, which prevents lossless KV caching and requires a full forward pass at every denoising step. Existing approximate KV caching methods reduce this cost by selectively updating cached states, but their decision overhead scales with context length or model depth. We propose EntropyCache, a training-free KV caching method that uses the maximum entropy of newly decoded token distributions as a constant-cost signal for deciding when to recompute. Our design is grounded in two empirical observations: (1) decoded token entropy correlates with KV cache drift, providing a cheap proxy for cache staleness, and (2) feature volatility of decoded tokens persists for multiple steps after unmasking, motivating recomputation of the $k$ most recently decoded tokens. The skip-or-recompute decision requires only $O(V)$ computation per step, independent of context length and mod
    
[^144]: TherapyGym：评估并对齐治疗聊天机器人的临床保真度与安全性

    TherapyGym: Evaluating and Aligning Clinical Fidelity and Safety in Therapy Chatbots

    [https://arxiv.org/abs/2603.18008](https://arxiv.org/abs/2603.18008)

    该论文提出了THERAPYGYM框架，通过基于CTRS量表的自动化评分衡量临床保真度、通过多标签标注衡量治疗安全性来评估治疗聊天机器人，并发布含116段对话和1270条专家评分的THERAPYJUDGEBENCH验证集，用于对照执业临床医生校准LLM评判者。

    

    arXiv:2603.18008v2 公告类型：replace-cross 摘要：大语言模型（LLM）正越来越多地被用于心理健康支持；然而，现有的主流评估方法——流畅度指标、偏好测试以及通用对话基准——无法捕捉心理治疗中临床上的关键维度。我们提出了THERAPYGYM，一个围绕两大临床支柱——保真度与安全性——来评估和改进治疗聊天机器人的框架。保真度通过认知治疗评分量表来衡量，该量表被实现为一个自动化流水线，在多轮对话会话中对认知行为疗法（CBT）技术的依从性进行评分。安全性则通过多标签标注方案进行评估，涵盖治疗特有的风险（例如未能妥善处理自伤或虐待问题）。为了减轻基于LLM的评判者存在的偏差与不可靠性，我们进一步发布了THERAPYJUDGEBENCH，一个包含116段对话和1270条专家评分的验证集，可用于对照执业临床医生对LLM评判者进行审计与校准。THERAPYGYM还可以作为训练框架来使用。

    arXiv:2603.18008v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used for mental-health support; yet prevailing evaluation methods--fluency metrics, preference tests, and generic dialogue benchmarks--fail to capture the clinically critical dimensions of psychotherapy. We introduce THERAPYGYM, a framework that evaluates and improves therapy chatbots along two clinical pillars: fidelity and safety. Fidelity is measured using the Cognitive Therapy Rating Scale (CTRS), implemented as an automated pipeline that scores adherence to CBT techniques over multi-turn sessions. Safety is assessed using a multi-label annotation scheme, covering therapy-specific risks (e.g., failing to address harm or abuse). To mitigate bias and unreliability in LLM-based judges, we further release THERAPYJUDGEBENCH, a validation set of 116 dialogues with 1,270 expert ratings for auditing and calibration against licensed clinicians. THERAPYGYM also serves as a training harnes
    
[^145]: 偏移越远，表征越稀疏：分析大语言模型中的分布外（OOD）机制

    Farther the Shift, Sparser the Representation: Analyzing OOD Mechanisms in LLMs

    [https://arxiv.org/abs/2603.03415](https://arxiv.org/abs/2603.03415)

    本文揭示了LLM内部表征的稀疏度随任务难度增加而系统性增大——任务越难，最后一层隐藏状态的表征越稀疏，且该规律跨模型、跨领域普遍成立，表明表征密度是模型对数据熟悉度的习得属性。

    

    在本工作中，我们研究了随着输入难度的增加，大语言模型（LLM）的内部表征如何发生变化。尽管在单个样本层面稀疏度的变化可能并不明显，但在数据集层面的结果揭示了一个一致的统计趋势：随着任务难度的提升（例如更难的问题、更长的上下文或更多的答案选项），LLM最后一层隐藏状态会系统性地变得更加稀疏。简而言之，任务越难，表征越稀疏。这种稀疏度与难度的关系在多种模型和领域中均可观察到，表明更难的输入会促使最后一层隐藏状态中的激活模式更加集中。通过一系列受控分析和学习动态实验，我们证明表征密度是模型对数据熟悉度的一种习得属性：模型会为已掌握的模式发展出丰富且分布式的表征，而面对陌生的输入时则表现出……

    arXiv:2603.03415v4 Announce Type: replace-cross  Abstract: In this work, we investigate how the internal representations of Large Language Models (LLMs) change with inputs of increasing difficulty. Although sparsity changes may not be obvious for individual samples, dataset-level results reveal a consistent statistical trend: as task difficulty increases (e.g., harder questions, longer contexts, or more answer choices), the last hidden states of LLMs become systematically sparser. In short, \textbf{the harder the task, the sparser the representation}. This sparsity--difficulty relation is observable across diverse models and domains, suggesting that harder inputs drive more concentrated activation patterns in the last hidden state. Through a series of controlled analyses and learning-dynamics experiments, we show that representational density is a learned property of data familiarity: models develop rich, distributed representations for mastered patterns, while unfamiliar inputs defaul
    
[^146]: CyclicJudge：高效缓解基于大语言模型评估中的评审偏差

    CyclicJudge: Mitigating Judge Bias Efficiently in LLM-based Evaluation

    [https://arxiv.org/abs/2603.01865](https://arxiv.org/abs/2603.01865)

    该论文提出方差分解分析方法，并证明CyclicJudge循环轮换评审策略能在与单评审评估相同的成本下有效消除大语言模型评审的系统性偏差，从而获得更可靠的模型排名。

    

    以大语言模型作为评审（LLM-as-judge）的评估方式已成为开放式模型评估的标准做法；然而，评审模型存在系统性偏差，这种偏差无法通过增加场景或生成样本的数量来平均消除。这些偏差的量级往往与基准测试所要检测的模型差异相当，因此使用单一评审进行评估会导致不可靠的排名。我们提出了一种方差分解方法，将基准测试分数的方差分解为场景、生成、评审和残差四个组成部分。基于这一分析，我们证明CyclicJudge——一种评审模型与场景之间的循环轮换分配策略——是在固定评审小组和评审调用预算条件下的最优策略：它能够在与单评审评估成本相同的前提下，消除评审模型的平均加性偏移。在MT-Bench和MindEval上的实验结果如预期般验证了CyclicJudge的有效性，无论是在通用领域还是特定领域场景中均表现良好。

    arXiv:2603.01865v4 Announce Type: replace  Abstract: LLM-as-judge evaluation has become standard practice for open-ended model assessment; however, judges exhibit systematic biases that cannot be averaged out by increasing the number of scenarios or generations. These biases are often similar in magnitude to the model differences that benchmarks are designed to detect, resulting in unreliable rankings when single-judge evaluations are used. We introduce a variance decomposition that partitions benchmark score variance into scenario, generation, judge, and residual components. Based on this analysis, CyclicJudge, a round-robin assignment of judges to scenarios, is demonstrated to be the optimal strategy for a fixed judge panel and judge-call budget: it cancels the average additive judge offsets while matching the cost of single-judge evaluation. Empirical results on MT-Bench and MindEval validate the effectiveness of CyclicJudge as predicted, across both general-purpose and domain-speci
    
[^147]: 发现心理量表中的语义结构：一种无需作答数据的高效简化前端方法

    Discovering Semantic Structures in Psychological Scales: A Response-Free Front-End to Efficient Simplification

    [https://arxiv.org/abs/2602.12575](https://arxiv.org/abs/2602.12575)

    该论文提出了一种无需作答数据的主题建模框架，利用上下文句子嵌入与密度聚类从量表题目措辞中自动发现语义结构并提取代表性关键词，从而实现心理量表的高效系统化简化。

    

    心理量表的精简传统上依赖于基于作答数据的方法，如因子分析、项目反应理论和网络心理测量学，以揭示潜在构念并优化题目构成。尽管这些方法严谨，但它们需要大样本，且可能受制于数据可得性和跨文化可比性。自然语言处理的最新进展表明，量表题目的措辞本身包含与构念相关的语义组织，从而提供了一种无需作答数据的互补视角。我们引入了一个主题建模框架，将题目层面的语义结构操作化，用于系统性的量表简化。题目通过上下文句子嵌入进行编码，并通过基于密度的聚类进行分组，从而在无需预先设定主题数量的情况下识别语义主题。在每个聚类内部，基于类别的词项加权方法识别出能够概括每个主题的代表性关键词，为后续……

    arXiv:2602.12575v3 Announce Type: replace  Abstract: Psychological scale refinement traditionally relies on response-based methods such as factor analysis, item response theory, and network psychometrics to uncover latent constructs and optimize item composition. Although rigorous, these approaches require large samples and may be constrained by data availability and cross-cultural comparability. Recent advances in natural language processing suggest that the wording of scale items contains construct-relevant semantic organization, offering a complementary response-free perspective. We introduce a topic-modeling framework that operationalizes item-level semantic structure for systematic scale simplification. Items are encoded using contextual sentence embeddings and grouped through density-based clustering to identify semantic topics without predefining their number. Within each cluster, class-based term weighting identifies representative keywords that summarize each topic and support
    
[^148]: Steer2Edit：从激活引导到组件级编辑

    Steer2Edit: From Activation Steering to Component-Level Editing

    [https://arxiv.org/abs/2602.09870](https://arxiv.org/abs/2602.09870)

    Steer2Edit提出了一种无需训练的理论框架，将引导向量从推理时的全局控制信号转变为诊断信号，通过在注意力头和MLP神经元之间选择性重新分配行为影响，实现保持模型标准功能的可解释组件级rank-1权重编辑。

    

    引导方法通过在隐藏表示中识别语义方向来影响大语言模型的行为，但通常是通过推理时的激活干预来实现，即对模型内部状态施加固定的、全局性的修改。虽然有效，但此类干预在强控制下往往会带来不利的属性-效用权衡，因为它们忽略了这样一个事实：许多行为是由模型组件中一个规模小且异构的子集所主导的。我们提出Steer2Edit，一个具有理论基础且无需训练的框架，它将引导向量从推理时的控制信号转变为用于组件级rank-1权重编辑的诊断信号。Steer2Edit不再在生成过程中均匀注入引导方向，而是选择性地在各个注意力头和MLP神经元之间重新分配行为影响，从而产生可解释的编辑，同时保持标准的前向传播路径（摘要在此处被截断）。

    arXiv:2602.09870v3 Announce Type: replace  Abstract: Steering methods influence Large Language Model behavior by identifying semantic directions in hidden representations, but are typically realized through inference-time activation interventions that apply a fixed, global modification to the model's internal states. While effective, such interventions often induce unfavorable attribute-utility trade-offs under strong control, as they ignore the fact that many behaviors are governed by a small and heterogeneous subset of model components. We propose Steer2Edit, a theoretically grounded, training-free framework that transforms steering vectors from inference-time control signals into diagnostic signals for component-level rank-1 weight editing. Instead of uniformly injecting a steering direction during generation, Steer2Edit selectively redistributes behavioral influence across individual attention heads and MLP neurons, yielding interpretable edits that preserve the standard forward pa
    
[^149]: 从结果到表征：通过积分策略梯度追踪推理机制

    From Outcome to Representation: Tracing Reasoning Mechanisms through Integrated Policy Gradient

    [https://arxiv.org/abs/2602.02313](https://arxiv.org/abs/2602.02313)

    本文提出无需训练的积分策略梯度（IPG）框架，通过将结果层面的不可微分行为信号沿整个推理轨迹反向传播并利用路径积分，在组件层面精准追踪神经元或稀疏特征对LLM多步推理结果的长程影响。

    

    大型语言模型（LLMs）展现出卓越的推理能力，然而驱动这些多步推理过程的内部机制仍然不透明。现有的机制可解释性方法通常依赖于表面的文本模式共现，或仅能捕捉短期效应，难以追踪神经元或稀疏特征等内部组件在多步推理中的长程、结果驱动的影响。本文提出了积分策略梯度，这是一种新颖的无需训练的框架，将基于策略的归因方法引入LLM推理的机制可解释性研究。该方法基于“结果导向”和“序贯影响感知”两项原则，将结果层面的不可微分行为信号（如推理结果的正确性）反向传播至整个推理轨迹，并利用路径积分在组件层面分离出与结果相关的中介因素。

    arXiv:2602.02313v3 Announce Type: replace  Abstract: Large Language Models (LLMs) demonstrate remarkable reasoning capabilities, yet the internal mechanisms driving these multi-step processes remain opaque. Existing mechanistic interpretability approaches often rely on superficial text-pattern co-occurrences or capture only short-term effects, struggling to trace the long-horizon, outcome-driven influence of internal components such as neurons or sparse features in multi-step reasoning.In this paper, we introduce Integrated Policy Gradient (IPG), a novel, training-free framework that brings policy-based attribution into mechanistic interpretability for LLM reasoning. Grounded in \textit{outcome-oriented} and \textit{sequential-influence-aware} principles, it backpropagates outcome-level, non-differentiable behavioral signals (e.g., reasoning outcome correctness) through entire inference trajectories, utilizing path integration to isolate outcome-relevant mediators at component level fo
    
[^150]: 大语言模型中的稀疏奖励子系统

    Sparse Reward Subsystem in Large Language Models

    [https://arxiv.org/abs/2602.00986](https://arxiv.org/abs/2602.00986)

    该论文发现大语言模型隐藏状态中存在稀疏的奖励子系统，由预测状态价值的“价值神经元”和编码时序差分误差的“多巴胺神经元”两类神经元组成。

    

    近期研究表明，大语言模型的隐藏状态编码了与奖励相关的信息，例如答案正确性和模型置信度。然而，现有方法通常在整个隐藏状态上拟合黑盒探针，对于这些信息在各神经元之间如何组织结构提供的洞察甚少。在本文中，我们证明奖励相关信息集中在稀疏的神经元子集中。通过简单的探测方法，我们识别出两类神经元：价值神经元，其激活可预测状态价值；以及多巴胺神经元，其激活编码步级时序差分（TD）误差。这些神经元共同构成了大语言模型隐藏状态内的一个稀疏奖励子系统。这些命名借鉴了神经科学的类比：生物奖励子系统中的价值神经元和多巴胺神经元也分别编码价值和奖励预测误差。我们证明价值神经元具有鲁棒性，并可在多种数据之间迁移。

    arXiv:2602.00986v3 Announce Type: replace  Abstract: Recent studies show that LLM hidden states encode reward-related information, such as answer correctness and model confidence. However, existing approaches typically fit black-box probes on the full hidden states, offering little insight into how this information is structured across neurons. In this paper, we show that reward-related information is concentrated in a sparse subset of neurons. Using simple probing, we identify two types of neurons: value neurons, whose activations predict state value, and dopamine neurons, whose activations encode step-level temporal difference (TD) errors. Together, these neurons form a sparse reward subsystem within LLM hidden states. These names are drawn by analogy with neuroscience, where value neurons and dopamine neurons in the biological reward subsystem also encode value and reward prediction errors, respectively. We demonstrate that value neurons are robust and transferable across diverse da
    
[^151]: 当外表不撒谎：基于话语结构引导的上下文学习用于忠实图表生成

    When Looks Do Not Lie: Discourse Structure Guided In-Context Learning for Faithful Diagram Generation

    [https://arxiv.org/abs/2601.20476](https://arxiv.org/abs/2601.20476)

    本文提出一种基于修辞结构理论的上下文学习图表生成方法，显著提升图表对源文本的忠实度，并通过专家与自动评估验证其有效性。

    

    arXiv:2601.20476v2 公告类型：替换 摘要：生成式人工智能在教育应用中广泛使用；然而，已知其会生成含有内在和外在幻觉的内容。我们提出了一种基于修辞结构理论的新型上下文学习图表生成方法，该方法提高了图表对其源文本上下文的忠实度。我们发现，上下文学习性能取决于任务分布和模型的推理能力，更高的推理能力能在分布外任务中带来更好的质量和性能。我们对150个生成的图表进行了专家评估，并使用贝叶斯广义线性混合模型分析了我们的发现。此外，我们利用评估标准和数据集样本进行自动图表评估，与人工评估取得了统计学上显著的一致性。

    arXiv:2601.20476v2 Announce Type: replace  Abstract: GenAI is widespread in educational applications; however, it is known to generate content with intrinsic and extrinsic hallucination. We introduce a novel method for ICL diagram generation based on Rhetorical Structure Theory, which improves diagram faithfulness to its source text context. We find that ICL performance depends on task distribution and models' reasoning ability, with higher reasoning allowing better quality and performance for an out-of-distribution task. We perform an expert evaluation of 150 generated diagrams and analyze our findings using Bayesian GLMMs. Additionally, we use our evaluation rubric and samples from the data set for automated diagram evaluation, achieving statistically significant agreement with human evaluation.
    
[^152]: DIAL：面向逼真多轮对话模拟的直接迭代对抗学习

    DIAL: Direct Iterative Adversarial Learning for Realistic Multi-Turn Dialogue Simulation

    [https://arxiv.org/abs/2512.20773](https://arxiv.org/abs/2512.20773)

    提出 DIAL 对抗框架，通过生成器（用户模拟器）与判别器的竞争性动态迭代提升多轮对话用户模拟的逼真度，在心理健康支持场景中恢复了词汇多样性，并使模拟失败率与真实失败率高度相关。

    

    逼真的用户模拟对于训练和评估多轮对话系统至关重要，然而构建能够准确复现人类行为的模拟器仍然是一项重大挑战。一个有效的模拟器必须能够暴露被评估系统的失败模式。本工作提出了直接迭代对抗学习（DIAL），这是一个通过生成器（用户模拟器）与判别器之间的竞争性动态来迭代提升用户模拟器逼真度的对抗框架。当应用于心理健康支持领域——该领域的特点是失败类型多样，且失败检测严重依赖于逼真的用户行为——DIAL 恢复了被监督微调所削弱的词汇多样性，并大幅降低了判别器的准确率。所得的模拟器在模拟失败发生率与真实失败发生率之间表现出强相关性，同时保持了失败分布的低散度。

    arXiv:2512.20773v5 Announce Type: replace  Abstract: Realistic user simulation is crucial for training and evaluating multi-turn dialogue systems, yet creating simulators that accurately replicate human behavior remains a significant challenge. An effective simulator must expose the failure modes of the systems under evaluation. This work introduces Direct Iterative Adversarial Learning (DIAL), an adversarial framework that iteratively enhances user simulator realism through a competitive dynamic between a generator (user simulator) and a discriminator. When applied to mental health support, a domain characterized by diverse failure types and a critical dependence on realistic user behavior for failure detection, DIAL restores lexical diversity diminished by supervised fine-tuning and drastically reduces discriminator accuracy. The resulting simulator exhibits a strong correlation between simulated and real failure occurrence rates while maintaining low distributional divergence of fai
    
[^153]: 面向基于大语言模型社会模拟的种群对齐式人格生成

    Population-Aligned Persona Generation for LLM-based Social Simulation

    [https://arxiv.org/abs/2509.10127](https://arxiv.org/abs/2509.10127)

    该论文提出了一个系统性框架，通过从长期社交媒体数据生成叙事型角色、质量评估过滤低保真档案、并利用重要性采样与大五人格等参考心理测量分布对齐，来合成高质量且与真实人口分布对齐的角色集合，从而提升LLM社会模拟的真实性。

    

    大语言模型（LLM）的最新进展使得大规模、高保真度的社会模拟成为可能，为计算社会科学创造了新的机遇。然而，构建能够忠实反映现实世界人口多样性的角色（persona）集合仍然是一个关键挑战。现有研究往往强调智能体框架和模拟环境，而对角色生成以及不具代表性的角色集合所引入的偏差关注较少。本文提出了一个系统性框架，用于为LLM驱动的社会模拟合成高质量、与种群对齐的角色集合。我们的方法首先利用LLM从长期社交媒体数据中生成叙事型角色，随后通过严格的质量评估来过滤低保真度的角色档案。接着，我们应用重要性采样来实现与参考心理测量分布（如大五人格）的全局对齐。

    arXiv:2509.10127v3 Announce Type: replace-cross  Abstract: Recent advances in large language models (LLMs) have enabled large-scale, high-fidelity social simulations, creating new opportunities for computational social science. However, constructing persona sets that faithfully reflect real-world population diversity remains a key challenge. Existing studies often emphasize agentic frameworks and simulation environments, while paying less attention to persona generation and the biases introduced by unrepresentative persona sets. In this paper, we propose a systematic framework for synthesizing high-quality, population-aligned persona sets for LLM-driven social simulation. Our approach begins by leveraging LLMs to generate narrative personas from long-term social media data, followed by rigorous quality assessment to filter out low-fidelity profiles. We then apply importance sampling to achieve global alignment with reference psychometric distributions, such as the Big Five personality 
    
[^154]: 边缘型人格障碍的计算现象学：LLM模拟专家角色与人类临床专家的比较评估

    Computational Phenomenology of Borderline Personality Disorder: A Comparative Evaluation of LLM-Simulated Expert Personas and Human Clinical Experts

    [https://arxiv.org/abs/2508.19008](https://arxiv.org/abs/2508.19008)

    本研究通过混合方法比较评估了LLM模拟专家与人类临床专家在边缘型人格障碍定性分析中的表现，发现LLM在支持临床主题分析方面具有潜力但存在差异。

    

    基于一项由人类主导的对边缘型人格障碍住院患者临床生活史访谈（超过15万字）的主题分析，本研究考察了大型语言模型（OpenAI的GPT、Google的Gemini和Anthropic的Claude）支持定性临床分析的能力。通过混合方法评估了模型的解释潜力。研究A涉及现象学和临床心理学领域的盲审与非盲审评判者，专家们使用语义一致性、Jaccard系数和多维有效性评级（包括可信度、连贯性、结果的实质性及定性数据基础）评估了AI生成内容的有效性。在研究B中，使用神经方法将人类和模型生成的主题描述嵌入多维向量空间，该方法提供了人类与模型之间差异的客观化计算度量。

    arXiv:2508.19008v3 Announce Type: replace  Abstract: Building on a human-led thematic analysis of clinical life-story interviews (> 150,000 words) with inpatients with Borderline Personality Disorder, this study examines the capacity of large language models (OpenAI's GPT, Google's Gemini, and Anthropic's Claude) to support qualitative clinical analysis. The models' interpretative potential was evaluated using a mixed-methods approach. Study A involved blinded and non-blinded judges in phenomenology and clinical psychology. The experts assessed the validity of AI-generated content using semantic congruence, Jaccard coefficients, and multidimensional validity ratings, including credibility, coherence, the substantiveness of results, and grounding in qualitative data. In Study B, neural methods were used to embed human- and model-generated theme descriptions in a multidimensional vector space. This approach provided an objectified computational measure of the difference between human and
    
[^155]: DUP：面向语言模型后门净化的检测引导遗忘方法

    DUP: Detection-guided Unlearning for Backdoor Purification in Language Models

    [https://arxiv.org/abs/2508.01647](https://arxiv.org/abs/2508.01647)

    提出DUP统一框架，通过联合类无关距离与层间转换实现细粒度后门检测，并利用参数高效的遗忘机制净化语言模型中的后门，无需完整重训练或外部干净模型。

    

    随着后门攻击变得更加隐蔽和鲁棒，它们暴露了当前防御策略的关键弱点：检测方法通常依赖于粗粒度的特征统计，而净化方法通常需要完整的重新训练或额外的干净模型。为了应对这些挑战，我们提出了DUP（检测引导的遗忘净化），这是一个将后门检测与基于遗忘的净化相结合的统一框架。该检测器通过联合利用类无关距离和层间转换来捕获特征级别的异常。这些偏差通过加权方案进行整合以识别被投毒的输入，从而实现更细粒度的分析。基于检测结果，我们通过一种参数高效的遗忘机制来净化模型，该机制避免了完整的重新训练，且不需要任何外部干净模型。具体而言，我们创新性地将知识蒸馏重新用于

    arXiv:2508.01647v2 Announce Type: replace-cross  Abstract: As backdoor attacks become more stealthy and robust, they reveal critical weaknesses in current defense strategies: detection methods often rely on coarse-grained feature statistics, and purification methods typically require full retraining or additional clean models. To address these challenges, we propose DUP (Detection-guided Unlearning for Purification), a unified framework that integrates backdoor detection with unlearning-based purification. The detector captures feature-level anomalies by jointly leveraging class-agnostic distances and inter-layer transitions. These deviations are integrated through a weighted scheme to identify poisoned inputs, enabling more fine-grained analysis. Based on the detection results, we purify the model through a parameter-efficient unlearning mechanism that avoids full retraining and does not require any external clean model. Specifically, we innovatively repurpose knowledge distillation t
    
[^156]: 面向结构化税则编码预测的领域自适应小型语言模型

    Domain-Adaptive Small Language Models for Structured Tax Code Prediction

    [https://arxiv.org/abs/2507.10880](https://arxiv.org/abs/2507.10880)

    该论文提出了一种基于编码器-解码器架构的领域自适应小型语言模型，能够从非结构化产品和服务数据中顺序预测层级化的税则编码（如HSN/SAC），从而提升跨国公司税务合规的准确性。

    

    跨国公司每天处理数千笔交易，每笔交易都必须遵守因司法管辖区而异且往往十分细微的税务法规。确定产品和服务税则编码（如HSN或SAC）是税务合规中的一个重要应用场景，准确确定这些编码对于避免税务处罚至关重要。本文提出了一种采用编码器-解码器架构的领域自适应小型语言模型（SLM），用于增强产品和服务税则编码的预测效果。在该方法中，我们利用非结构化的产品和服务数据来解决预测层级化税则编码序列的问题。我们采用基于编码器-解码器架构的SLM，因为该架构能够顺序生成税则编码，从而捕获税则编码内部存在的层级依赖关系。我们的实验表明，编码器-解码器SLM可以成功地应用于税则编码的顺序预测任务。

    arXiv:2507.10880v3 Announce Type: replace-cross  Abstract: Every day, multinational firms process thousands of transactions, each of which must adhere to tax regulations that vary by jurisdiction and are often nuanced. The determination of product and service tax codes, such as HSN or SAC is a major use case in Tax compliance. An accurate determination of such codes is imperative to avoid any tax penalties. This paper proposes a domain-adaptive small language model (SLM) with an encoder-decoder architecture for the enhanced prediction of product and service tax codes. In this approach, we address the problem of predicting hierarchical tax code sequences using unstructured product and services data. We employ an SLM based upon encoder-decoder architecture as this enables sequential generation of tax codes to capture the hierarchical dependencies present within the tax codes. Our experiments demonstrate that encoder-decoder SLMs can be successfully applied to the sequential prediction of
    
[^157]: 分子版《绝命毒师》：多模态大语言模型准备好进行结构级分子脱毒了吗？

    Breaking Bad Molecules: Are MLLMs Ready for Structure-Level Molecular Detoxification?

    [https://arxiv.org/abs/2506.10912](https://arxiv.org/abs/2506.10912)

    该论文提出了ToxiMol——首个评估通用多模态大语言模型分子毒性修复能力的基准任务，包含覆盖11个任务、660个有毒分子的标准化数据集、机制感知的提示标注流程以及自动化评估框架ToxiEval。

    

    毒性仍然是早期药物开发失败的主要原因。尽管分子设计和性质预测领域已取得进展，但分子毒性修复这一任务——即生成结构有效且毒性降低的分子替代物——尚未被系统地定义或建立基准。为填补这一空白，我们提出了ToxiMol，这是首个专注于分子毒性修复的通用多模态大语言模型（MLLMs）基准任务。我们构建了一个标准化数据集，涵盖11个主要任务和660个代表性有毒分子，覆盖多种毒性机制和粒度。我们设计了一个具有机制感知和任务自适应能力的提示标注流程，并融入了专家毒理学知识。同时，我们提出了一个自动化评估框架ToxiEval，它将毒性终点预测、合成可及性、类药性和结构相似性整合到一个高（原文在此处截断）。

    arXiv:2506.10912v5 Announce Type: replace  Abstract: Toxicity remains a leading cause of early-stage drug development failure. Despite advances in molecular design and property prediction, the task of molecular toxicity repair, generating structurally valid molecular alternatives with reduced toxicity, has not yet been systematically defined or benchmarked. To fill this gap, we introduce ToxiMol, the first benchmark task for general-purpose Multimodal Large Language Models (MLLMs) focused on molecular toxicity repair. We construct a standardized dataset covering 11 primary tasks and 660 representative toxic molecules spanning diverse mechanisms and granularities. We design a prompt annotation pipeline with mechanism-aware and task-adaptive capabilities, informed by expert toxicological knowledge. In parallel, we propose an automated evaluation framework, ToxiEval, which integrates toxicity endpoint prediction, synthetic accessibility, drug-likeness, and structural similarity into a hig
    
[^158]: 羊驼见，羊驼做：大语言模型中情境牵引与分心现象的机制性视角

    Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs

    [https://arxiv.org/abs/2505.09338](https://arxiv.org/abs/2505.09338)

    该论文发现语言模型存在一种“情境牵引”的机制性现象：模型会无差别地提高上下文提示中出现过的任何token（即使是随机token）的输出概率，这一现象独立于语义相关性，但其强度受反事实等语义因素的调节。

    

    我们在广泛的语言模型（LMs）和提示设置中观察到一个新现象——情境牵引，这为语言模型如何被输入提示中的“无关”上下文信息所干扰提供了新的机制性视角。具体而言，语言模型会为之前出现在上下文提示中的任何token赋予显著更高的logits（或概率），即使是随机token也是如此。这表明情境牵引是一种机制性现象，其发生独立于token与问题或句子其余部分的相关性或语义关系。我们发现有统计学显著性的证据表明，情境牵引的强度会受到语义因素的影响。反事实提示相比事实性提示具有更大的影响效果，这说明虽然情境牵引是一种机制性现象，但它会受到语义因素的调节。我们假设存在一个……

    arXiv:2505.09338v3 Announce Type: replace  Abstract: We observe a novel phenomenon, contextual entrainment, across a wide range of language models (LMs) and prompt settings, providing a new mechanistic perspective on how LMs become distracted by ``irrelevant'' contextual information in the input prompt. Specifically, LMs assign significantly higher logits (or probabilities) to any tokens that have previously appeared in the context prompt, even for random tokens. This suggests that contextual entrainment is a mechanistic phenomenon, occurring independently of the relevance or semantic relation of the tokens to the question or the rest of the sentence. We find statistically significant evidence that the magnitude of contextual entrainment is influenced by semantic factors. Counterfactual prompts have a greater effect compared to factual ones, suggesting that while contextual entrainment is a mechanistic phenomenon, it is modulated by semantic factors.   We hypothesise that there is a ci
    
[^159]: 结合确定性与非确定性

    Combining Determinism and Indeterminism

    [https://arxiv.org/abs/2009.03996](https://arxiv.org/abs/2009.03996)

    本文提出了将量子随机性带来的非确定性与计算确定性相结合的数学运算，并发现保持双免疫性的重排构成了自然数无限对称群的一个新的不可数子群——双免疫对称群，该子群包含有限对称群、具有高度传递性并在对称群中稠密。

    

    我们的目标是构建一些数学运算，将从量子随机性中测得的非确定性与计算确定性结合起来，使得非机械性行为在计算中得以保留。在形式上，本文证明了若干关于应用于可计算枚举集（c.e. 集）和双免疫集的运算的结果，其目标是使这些运算能够保持双免疫性。在开发自然数上的重排运算时，我们发现双免疫重排生成了自然数 $\mathbb{N}$ 上的无限对称群 Sym$(\mathbb{N})$ 的一个不可数子群。这个新的不可数子群被称为双免疫对称群。我们证明了双免疫对称群包含自然数上的有限对称群，因此它是高度传递的。此外，在逐点收敛拓扑下，双免疫对称群在 Sym$(\mathbb{N})$ 中是稠密的。

    arXiv:2009.03996v5 Announce Type: replace-cross  Abstract: Our goal is to construct mathematical operations that combine indeterminism measured from quantum randomness with computational determinism so that non-mechanistic behavior is preserved in the computation. Formally, some results about operations applied to computably enumerable (c.e.) and bi-immune sets are proven here, where the objective is for the operations to preserve bi-immunity. While developing rearrangement operations on the natural numbers, we discovered that the bi-immune rearrangements generate an uncountable subgroup of the infinite symmetric group (Sym$(\mathbb{N})$) on the natural numbers $\mathbb{N}$.   This new uncountable subgroup is called the bi-immune symmetric group. We show that the bi-immune symmetric group contains the finitary symmetric group on the natural numbers, and consequently is highly transitive. Furthermore, the bi-immune symmetric group is dense in Sym$(\mathbb{N})$ with respect to the pointw
    

