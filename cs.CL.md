# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoScaffold: AI-Guided Scaffolding for Reinforcement Learning of Language Models](https://arxiv.org/abs/2611.14515) | 提出了AutoScaffold框架，由冻结的教师模型自动将学生的训练经验转化为自适应脚手架（任务类共享策略与困难任务专用提示），在无需人工干预的情况下提升语言模型可验证奖励强化学习的效果。 |
| [^2] | [Contrastive Discovery: Open-Ended Scientific Discovery over Competing Explanations](https://arxiv.org/abs/2611.14514) | 该论文提出“对比发现”框架，用互斥且穷尽的竞争性解释集合取代单一假设检验，通过证据对信念在集合内的重新分配来评估实验、确定被支持的替代解释，从而改进开放式科学发现。 |
| [^3] | [Comparative Analysis of Agent Evaluation Frameworks: Stability, Detection, and Discovery in Enterprise Analytics Agents](https://arxiv.org/abs/2611.14493) | 该论文通过三阶段比较研究系统评估了三种企业分析智能体评估框架，发现基于轨迹的LLM评判框架在评估稳定性（97%配置达标）和已知风险检测（F1=0.90）方面综合表现最佳。 |
| [^4] | [Mechanistic Interpretability Needs a Logic of Inference](https://arxiv.org/abs/2611.14483) | 该论文指出现有机制可解释性研究需要一套严谨的推理逻辑，提出由五项原则构成的递归逻辑框架，以解决从复杂神经系统中推断机制时反复出现的推理瓶颈问题。 |
| [^5] | [Pterodactyl: Pushing the Limits of Efficient Automated Translation Evaluation](https://arxiv.org/abs/2611.14480) | 本文提出了Pterodactyl系统，一个基于LLM-as-a-judge方法、通过ROC分析指导提示工程与模型选择的高度可定制超低成本系统，专为WMT26质量评估任务中的句段级质量分数预测和无错误句段检测而设计。 |
| [^6] | [FERN: Cross-Benchmark Table Retrieval for Text-to-SQL with Frozen Retrievers and Learned Ranking](https://arxiv.org/abs/2611.14477) | FERN不使用任何生成式大语言模型，仅通过融合三个冻结检索器生成候选表格并用LightGBM学习排序器重排，在跨基准的Text-to-SQL表格检索任务上达到了与依赖Claude 3.5的RASL相近的召回率，且无需针对新基准重新训练。 |
| [^7] | [When does text encoding matter for language modeling?](https://arxiv.org/abs/2611.14459) | 本文提出混合编码方案 Aether，并系统研究了文本编码对字节级语言模型性能的影响，发现中文、韩语等信息密集文字的语言能从压缩编码中显著受益，从而揭示了编码选择带来的跨语言性能差异。 |
| [^8] | [Earlybird: Predictive Streaming Speech Recognition in 29M Parameters](https://arxiv.org/abs/2611.14421) | Earlybird是一个仅2900万参数的预测式流式语音识别模型，通过在随机截断的音频上学习预测词语，能在说话者说完中位词之前约90毫秒就正确输出，比现有流式模型提前99至490毫秒。 |
| [^9] | [Bits or permutations? Spending a fixed inference budget on weight precision or on option-order averaging in small local language models](https://arxiv.org/abs/2611.14413) | 本文量化了小型本地语言模型中权重量化精度对答案选项顺序偏差的影响，发现6比特量化不损失准确率而4比特量化会显著加剧答案槽位偏斜，并据此比较了固定推理预算下提升权重精度与选项顺序平均两种方案的性价比。 |
| [^10] | [REAL Loss: Optimizing Real Expected Acceptance Length for Speculative Decoding](https://arxiv.org/abs/2611.14399) | 该论文提出 REAL 损失，通过路径权重无偏估计与 Rao-Blackwellization 方差缩减，直接优化投机解码中真实的期望接受长度（EAL），解决了现有替代目标存在的协方差差距及其梯度方向反转问题。 |
| [^11] | [Blurring the Block Boundaries: Overlapping Block Denoising for Efficient Diffusion Language Models](https://arxiv.org/abs/2611.14387) | 本文提出B3方法，通过在共享前向传播中并行精炼两个块来模糊块边界，实现无需训练和超参数调整的扩散语言模型加速，同时保持生成质量。 |
| [^12] | [ARE-bench: Benchmarking Agents' Research Ability in Real-World Researcher Sessions and Workspaces](https://arxiv.org/abs/2611.14364) | ARE-bench 基于真实研究者的会话记录与进行中的工作空间构建了 75 个任务，用于评估研究智能体能否在尊重先前决策和现有工作状态的前提下推进真实研究项目。 |
| [^13] | [EngineAgent: Agentizing On-Device LLM Inference Engines with Inference Stack Hierarchy Awareness and Hardware Abilities](https://arxiv.org/abs/2611.14356) | EngineAgent通过构建层级感知的引擎索引并利用已支持模型预先验证后端工作流程，将可复用的引擎与硬件知识从模型特定实现中分离，从而自动化端侧LLM推理引擎的新模型适配过程。 |
| [^14] | [AI Isn't Getting Dumber, We're Getting Pickier: Rising Standards in Human Evaluation of Large Language Models](https://arxiv.org/abs/2611.14333) | 该研究通过分析Chatbot Arena上126万余张针对固定版本模型的盲投票数据发现，“两个都不好”的投票比例持续显著上升，表明AI模型并非在变差，而是人类评估者的标准在不断提高。 |
| [^15] | [Point-in-Time Encoders for Earnings Calls](https://arxiv.org/abs/2611.14325) | 该论文提出按年度截止日期训练的“时点”文本编码器，以消除财报电话会议检索回测中的前视偏差，其小型专用模型在答案检索上全面优于通用检索器，并将收益预测的秩相关性提升约一半。 |
| [^16] | [MMDM: Measuring Massive Multitask Decision Making](https://arxiv.org/abs/2611.14309) | 提出了MMDM基准——涵盖16个任务类别、5,448道高质量问题的大规模多任务决策基准，用于在无数据污染的全新上下文中评估系统-1决策模型在单选、多选和软标签决策上的能力。 |
| [^17] | [Semantic Chains: From Reachable Facts to Justified Answers](https://arxiv.org/abs/2611.14252) | 本文将上下文建模为带保持边的逻辑事实图，证明了从问题中的对象出发的搜索可以到达证明答案的根事实，从而构建出连接问题与有据答案的“语义链”，并对多环节链条给出候选顺序的完整枚举。 |
| [^18] | [An AI Assistant for Automating Anti-Corruption Expertise of Draft Legislation Based on the TRACE Framework](https://arxiv.org/abs/2611.14244) | 该论文基于TRACE架构提出了一种自动化反腐败专家审查的AI助手，能够筛查法案草案中的腐败风险并精确定位专家会批评的问题条款，同时利用乌克兰国家反腐败预防署的公开结论构建了首个开放评估基准。 |
| [^19] | [Dyship: Leveraging Acoustic Features For Speaker Relationship Classification From Dyadic Conversations](https://arxiv.org/abs/2611.14241) | 该论文提出DyShip框架，通过比较频谱图、MFCC、WavLM、x向量和发音运动五种声学表示，从双人对话中实现说话人熟悉度和亲缘关系的自动分类，且多数分类器表现超越人类和音频语言模型基准。 |
| [^20] | [StaRSE: Compact Russian Sentence Embeddings with a Sign-Coded Static Encoder](https://arxiv.org/abs/2611.14236) | StaRSE是一种紧凑的俄语静态句子编码器，通过将词元方向以符号位存储并去除Transformer推理，以仅11.3 MiB的模型大小和0.08毫秒的CPU延迟，在RuMTEB上实现了比最强静态基线高7.98分的检索性能。 |
| [^21] | [Stagewise and interpretable emergence of inside-outside computation in Transformer-based language models](https://arxiv.org/abs/2611.14221) | 本文首次通过研究具有可变句法树结构的合成上下文无关文法中的学习动态，展示了Transformer语言模型如何分阶段、可解释地涌现出inside-outside句法分析计算，其中每个损失阶段均可被解析计算并与潜在树变量表示的涌现明确关联。 |
| [^22] | [Breakdown of Autoregressive Multi-Step Reasoning Under Concentrated vs. Diffuse Attention Head Dropout](https://arxiv.org/abs/2611.14209) | 受哺乳动物大脑“关闭状态”启发，该研究发现对大语言模型注意力头施加集中式爆发失活，比同等规模的分散式失活会造成严重得多的多步数学推理性能下降。 |
| [^23] | [Representational Atoms in Neural Networks](https://arxiv.org/abs/2611.14168) | 该论文定义了神经网络中的基本表征单元（FRUs）并建立保真度与稳定性两项评估标准，证明阈值激活稀疏自编码器（TSAEs）在适当条件下可高保真、稳定地识别FRUs，且高稳定性的FRUs具有更强的单语义性并能跨层组合以恢复与人类概念结构相对应的语义组织。 |
| [^24] | [Visual Similarity in Adult Word Extension and Artificial Word Learning](https://arxiv.org/abs/2611.14116) | 本研究通过重新分析大规模成人词汇扩展数据并进行人工词汇学习实验，利用冻结的自监督视觉编码器（DINOv2）量化物体间的视觉相似性，发现视觉距离显著影响词汇的扩展范围，且教学示例的类别层级高低决定了学习者在多大程度上将词汇泛化到更广泛或更具体的类别。 |
| [^25] | [Sckwoky at SHROOM-Visions: Character-Level Hallucination-Span Detection in Image-Conditioned VLMs via Dual-Prompted LoRA Decoders](https://arxiv.org/abs/2611.14100) | 该论文提出了一套基于冻结Gemma-4-12B-it骨干网络与LoRA适配器的字符级幻觉片段检测系统，通过在单一提示中重复候选答案来获取近似双向上下文，可在四种语言上同时预测视觉语言模型响应中的幻觉概率、边界与类型。 |
| [^26] | [From Bias to Positionality: A critical grammar for Large Language Models](https://arxiv.org/abs/2611.14079) | 该论文提出用“处境性”取代“偏见”作为理解大语言模型的批判性框架，因为偏见需要外部标准作为参照，而在语言模型的一般使用中不存在这样的标准，因此更应追问模型“从何处说话”。 |
| [^27] | [HandsOnAid: Building and Evaluating an Offline First-Aid Assistant for Emergency Response Scenarios](https://arxiv.org/abs/2611.14059) | 本文构建了HandsOnAid——一个可在无信号、无救援环境中通过QLoRA微调的4位Gemma 2B模型实现的离线手机急救助手，并通过四个前沿大模型盲评验证了微调显著提升了急救建议的质量与安全性。 |
| [^28] | [Thermodynamics of words](https://arxiv.org/abs/2611.14039) | 该论文提出词汇热化假说，将文学作品中的词频分布类比为经典振荡器的瑞利-金斯热化与凝聚现象，并通过对《圣经》、莎士比亚全集等16部作品的分析，验证了低有效能量条件下会出现词汇的RJ凝聚。 |
| [^29] | [MISO-BR: An Open Corpus for Misogyny Detection in Brazilian Portuguese](https://arxiv.org/abs/2611.14036) | 该论文推出了MISO-BR——首个面向巴西葡萄牙语厌女症检测的开放、有文档记录且版本化的标注语料库，附带完整的构建协议、标注指南和可复现的基线模型，填补了该领域开放资源的空白。 |
| [^30] | [Program-Aided Patchable Skill Evolution](https://arxiv.org/abs/2611.14019) | 该论文提出PASE框架，通过将智能体技能中的可复用程序性逻辑分离为可编辑的可执行程序（而非自然语言文档），实现技能的可维护性、可补丁化与针对性迭代演化。 |
| [^31] | [One Sequence Is Enough: Fingerprinting Language Models from Fixed-Length Integer Sequences](https://arxiv.org/abs/2611.14010) | 该论文提出NumPrint基准和SMCT模型，证明仅凭模型生成的一条定长整数序列，就能以92.72%的准确率在51个语言模型中识别出响应请求的真实模型。 |
| [^32] | [Coding Agent Memory Post-Training II: Unlocking the Memory Potential of Pre-trained File Operations for Long-Horizon Tasks via On-Policy Distillation](https://arxiv.org/abs/2611.14009) | 该论文提出 CAMG-II 长时程智能体强化学习环境套件，通过提供 Bash 访问和回合持久工作区，让记忆控制建立在预训练已有的文件操作能力之上，从而解决在线策略蒸馏中因记忆接口超出预训练范围而导致教师指导无效、学生记忆动作采样不足的问题。 |
| [^33] | [Specify Privacy Yourself: Assessing Inference-Time Personalized Privacy Preservation Ability of Large Vision-Language Models](https://arxiv.org/abs/2611.13997) | 该论文提出了首个系统性评估大型视觉语言模型“推理时个性化隐私保护”（ITP³）能力的基准SPY-Bench，包含32,700个涵盖67个类别和24个真实场景的样本以及基于用户规范与情境感知的新颖指标，旨在使模型能够遵循用户通过语言动态指定的个性化隐私边界。 |
| [^34] | [On the information rate of phones, syllables and words](https://arxiv.org/abs/2611.13990) | 该研究利用960小时英语有声读物语料库，在n-gram和transformer语言模型的有限建模预算下比较不同语言单位的信息速率，发现词汇（及稍长的单位）能实现最低比特率，表明比音节大但小于长多词短语的语言单位是编码语音内容最高效的表示方式。 |
| [^35] | [Understanding Mixture-of-Experts Language Models Beyond Expert Boundaries](https://arxiv.org/abs/2611.13975) | 该研究提出路由感知幅度剖析（RAMP）方法，首次揭示MoE模型中领域专业化计算在神经元层面的组织规律（同领域神经元重叠率高达58-74%，远超跨领域的15-24%），并追踪了这种组织在预训练和后训练全过程中的动态演变。 |
| [^36] | [AVCC: Audio-Visual Conversation Corpus for Turn-Taking Prediction with a Human Benchmark](https://arxiv.org/abs/2611.13957) | 该论文将31小时单摄像头单麦克风录制的多人对话视听语料库AVCC转化为首个面向机器人单传感设备的公开下一发言者预测基准，并提供词级对齐转录与人类表现参考基准。 |
| [^37] | [Beyond Fluent Responses: Finding User Dissatisfaction in Human-AI Interactions](https://arxiv.org/abs/2611.13894) | 该研究通过分析带有自我报告不满标签的用户话语，证明人机交互中用户的不满情绪可以从对话文本中被检测出来，从而将回答评估从以回答为中心的质量指标拓展到以用户为中心的信号。 |
| [^38] | [Are Causal Neuron Circuits Unique and Transferable?](https://arxiv.org/abs/2611.13885) | 该论文通过穷举测量 12 个 MLP 神经元全部 4,096 种子集的干预效果，发现虽然消融选定神经元组会显著改变模型回答，但这些神经元的激活在不同问题间几乎恒定、替换后也几乎无影响，说明真正必要的只是其中一部分，从而质疑了因果神经元回路的唯一性与可迁移性。 |
| [^39] | [Beyond Passive Imitation: On-Policy Mid-Training Distillation for Language Models](https://arxiv.org/abs/2611.13855) | 该论文提出在线策略中期训练蒸馏（OPMD），将中期训练从被动的离线文本模仿转变为主动的、查询驱动的知识获取，通过从学生策略自采样轨迹并结合教师模型的词元级反向KL奖励，解决了标准下一词预测目标在不同基础模型上效果不一甚至产生负面影响的问题。 |
| [^40] | [Alania-2: Turkish text-to-speech with instant and fine-tuned voice cloning](https://arxiv.org/abs/2611.13845) | Alania-2是一个基于VoxCPM架构的4.88亿参数土耳其语文本转语音系统，在1.8万余小时土耳其语语音（90%为教师模型合成数据）上从头训练，结合文本规范化技术，将词错误率降至1.44%，并支持即时与微调两种语音克隆方式。 |
| [^41] | [PolyRoute: A Twelve-Language and Code-Mixed Benchmark for Typed-Decision Routing, with Baselines and a Leaderboard](https://arxiv.org/abs/2611.13843) | 该论文发布了PolyRoute基准，涵盖12种语言、9种书写系统及混码文本的类型化决策路由评测，并提供13个系统的排行榜，显示微调模型可达96.6%准确率，且多语言模型对未公开语言具有良好的泛化能力。 |
| [^42] | [Antalia-2 Mini: A 7.6-million-parameter Turkish text-to-speech model trained entirely on synthetic speech](https://arxiv.org/abs/2611.13811) | Antalia-2 Mini是一个仅用合成语音训练、仅760万参数的超轻量土耳其语TTS模型，可在笔记本CPU上以41倍实时速度生成高质量语音，WER低至1.09%。 |
| [^43] | [LLM Validation Makes Minimal-Pair Benchmarks Easier for Language Models](https://arxiv.org/abs/2611.13788) | 用LLM验证最小对子基准的标签并非随机出错，而是系统性地移除对语言模型而言更难的句子对，使基准变简单，且其验证效果并不优于一名人类评委。 |
| [^44] | [Word-Level Alignment of OCR and Ground Truth under Zoning and Reading-Order Errors](https://arxiv.org/abs/2611.13752) | 本文重新审视了OCR文本与标准答案的词级对齐问题，提出了一种在区域划分和阅读顺序错误（置换）下评估对齐器正确性的协议，并系统比较了多种对齐方法在阅读顺序扰动下的性能表现。 |
| [^45] | [Do Language Models Feel the Sentiment? Investigating Emotion Vectors](https://arxiv.org/abs/2611.13744) | 本研究在 Qwen2.5 模型的内部表示中定位了与情感概念对应的情绪向量方向，并通过情感探测和匹配消融实验证实了这些方向确实编码了情感信息，移除它们会显著降低模型的情感分类准确率。 |
| [^46] | [Walrus Technical Report](https://arxiv.org/abs/2611.13689) | Walrus 是一个统一的 8B 语音与音频基础模型，其核心贡献在于一套以数据为中心的三阶段课程训练方案：在标准 decoder-only 骨干上通过任务感知加权，实现从多语言到多说话人、语码转换等场景的广泛语音理解与生成能力。 |
| [^47] | [You Are What You Read: Indian Political News Media Bias Dataset for Hindi Language](https://arxiv.org/abs/2611.13686) | 本文构建了一个由四家印度主要新闻机构的448篇印地语政治新闻文章组成、经14位专家标注为四种偏见类别（选择性、框架、标题党、遗漏偏见）的可解释媒体偏见数据集，填补了印地语媒体偏见研究的空白。 |
| [^48] | [OpticalRAG: Pixel-Space Compression for Token-Efficient Retrieval-Augmented Generation](https://arxiv.org/abs/2611.13675) | 该论文提出OpticalRAG框架，通过将文档渲染为页面图像并利用冻结视觉编码器将检索内容蒸馏压缩为最少仅16个视觉令牌注入大语言模型，在无需昂贵跨模态训练的情况下突破了RAG的令牌预算瓶颈。 |
| [^49] | [An Interpretable Stylometric Framework for Early Chinese Buddhist Translator Styles: A Case Study on the Attribution of Disputed Works of An Shigao](https://arxiv.org/abs/2611.13636) | 本研究提出了一个基于佛教文献学家审定语言特征的可解释文体计量学框架，无需任务特定训练即可分析早期汉译佛典归属，判定T1557和T0101两部存疑文本与安世高风格最为相似，且结果与基于BERT的方法一致。 |
| [^50] | [Synapse-Control: Making Agent Experience Actionable with Lightweight Runtime Control](https://arxiv.org/abs/2611.13615) | Synapse-Control用一个轻量级智能体运行时控制平面（规则筛选、1.7B微调分类器评分、微合同接口检查和提示价值表冷却机制）取代大模型失败诊断，以更低的推理成本实现智能体执行经验的高效复用与精准投放。 |
| [^51] | [Post-OCR Correction of Mixed-Script Uzbek Text: Byte-Level ByT5 with an Empirical Homoglyph Noise Matrix](https://arxiv.org/abs/2611.13609) | 针对乌兹别克语中西里尔与拉丁字母混排导致的同形字符OCR错误，提出可在本地硬件上运行的字节级ByT5纠错系统，结合由人工标注错误事件估计的噪声矩阵与保守解码规则，将字符错误率从0.0578降至0.0369。 |
| [^52] | [One Byte, One Token: Byteifying a 27B Hybrid-Attention LLM with Self-Generated Data](https://arxiv.org/abs/2611.13607) | 研究者通过将词表缩减至384个条目（含256个字节符号），并利用原始模型自生成的4.77GB在线策略监督数据进行两阶段课程训练，成功将270亿参数的Qwen3.8混合注意力模型直接转换为纯字节级模型，无需补丁结构或局部编码器即可保留其大部分原有能力。 |
| [^53] | [Compute the Interactions, Not the Attributes: What a Fact Layer Must Supply in the Chinese Sexagenary System](https://arxiv.org/abs/2611.13597) | 该研究以八字六十甲子系统构建可查表事实基准，发现开源语言模型在需要组合属性的交互型事实上几乎全部失败，而提供按查询过滤、已计算好的交互事实可将准确率提升至96–100%，说明事实层应当“计算交互”而非罗列静态属性。 |
| [^54] | [Towards Reliable VLM Judges: State-Conditional Invariance and Presentation-Aware Diagnostics](https://arxiv.org/abs/2611.13596) | 该论文提出状态条件不变性（SCI）这一可靠性原则，并构建了基于模拟器的诊断基准GridWM-Judge，用于系统评估视觉语言模型评判者在语义不变的呈现变化下是否保持稳定、并对改变任务结果的因果干预作出正确响应。 |
| [^55] | [IFRS Compliance Verification: An Open Multi-LLM Benchmark and Content-Conditioned Ensemble Classifier](https://arxiv.org/abs/2611.13595) | 本文构建了一个包含19,998个IFRS监管条款-披露配对、结合多LLM标签与8,837条人工标注的开放基准，并提出内容条件化的集成分类器框架，以克服单一模型准确率不足及简单集成中各模型系统性偏差导致性能下降的问题。 |
| [^56] | [POEM: Output-Relative Block-Sparse Attention for Long-Context Prefilling](https://arxiv.org/abs/2611.13578) | 提出了一种推理时防御方法TERM，利用提示校准的注意力熵偏差来自适应调制模型自身的残差更新，无需预定义表示目标即可抵御越狱攻击并减少过度拒绝。 |
| [^57] | [TERM: Target-Free Jailbreak Defense via Entropy-Guided Residual Modulation](https://arxiv.org/abs/2611.13577) | 提出了TERM这一推理时防御方法，利用提示校准的注意力熵偏差自适应地调制模型自身的残差更新，在无需预定义表示目标的情况下防御越狱攻击，同时限制过度拒答。 |
| [^58] | [Pair-Merge Tokenisation Is Hard to Approximate on Binary Strings and Strongly NP-Hard on Unary Datasets](https://arxiv.org/abs/2611.13568) | 本文通过将顶点覆盖间隙转化为文法大小间隙的归约技术，证明了BPE式成对合并分词优化问题即使仅在单个二进制字符串上也难以近似（除非P=NP，近似因子不超过c·n/log n），且在一元字母表数据集上是强NP难的。 |
| [^59] | [A Concept-Based Interpretable Framework for Clinical Text-based Prediction via Token-Concept Attention without Concept Supervision](https://arxiv.org/abs/2611.13564) | 该论文提出了一种无需概念标注监督的可解释框架，通过定制交叉注意力瓶颈机制，利用预训练语言模型估计临床文本词元与医学概念之间的相关性，从而在临床预测任务中实现基于概念的黑盒模型解释。 |
| [^60] | [An evidence-traceable benchmark of clinical reasoning in large language models](https://arxiv.org/abs/2611.13562) | 该研究构建了首个证据可追溯的医学大语言模型基准Med Evidence Bench（MEB），将每条评分标准与版本化的文献原文片段绑定，揭示出十个主流大语言模型虽在指南知识上表现优异，但在真实世界临床场景中的推理能力显著下降。 |
| [^61] | [Are Dense Rewards Worth Their Tokens in GRPO?](https://arxiv.org/abs/2611.13560) | 尽管蒙特卡洛密集过程奖励在单个更新比较中优于标准GRPO，但在以生成token数和GPU小时数衡量的计算量公平匹配下，其高昂的采样开销（占生成token的91%）使结果排名发生逆转，密集奖励并不值得其token代价。 |
| [^62] | [Darmok: Championship Cryptic Crossword Solving via Constraint Tightening and Resilient Orchestration](https://arxiv.org/abs/2611.13557) | Darmok 是首个在英国密码填字游戏上达到超人水平的自动化系统，通过约束收紧与弹性编排技术，在《泰晤士报》填字游戏锦标赛中无错误解出全部七道谜题，且每一轮速度均超过所有人类选手。 |
| [^63] | [How Post-Training Influences Verbalized Algorithmic Computation in Language Models](https://arxiv.org/abs/2611.13552) | 后训练主要改变的是言语化推理的有用性而非内部信息的编码方式——SFT 和 RLVR 后的模型内部均可读取进位信息，但 RLVR 产生的思维链在各检查点上显著提升了推理准确率。 |
| [^64] | [Sanity Checks for Attention Surgery: Null Models and Matched Interventions for Representation-Motivated Design](https://arxiv.org/abs/2611.13543) | 该论文提出三个可复用的合理性检查（噪声底线、几何零模型、匹配干预）来严格检验基于表征统计的架构修改，并发现自注意力移除中的头-值对齐很大程度上可由早期位置的因果零模型解释，而非模型真正依赖的特定表征方向。 |
| [^65] | [Framing-Induced Vulnerability in Large Language Models: Evidence from Chinese News Headlines](https://arxiv.org/abs/2611.13531) | 该研究通过6万多条中文新闻标题审计和跨四个中文模型的受控实验，首次系统揭示了大型语言模型对新闻标题的判断主要受标题框架而非事实本身的影响——情绪化框架会显著抬高事件严重性评分，而提供完整文章正文可降低54%至80%的此类偏差。 |
| [^66] | [EJAI: Judgment-Preserving Civic AI for Data Center Environmental Justice](https://arxiv.org/abs/2611.13525) | 本文提出EJAI，一个面向数据中心环境正义争议的基于证据来源的公民AI系统，通过交互式地图、证据检索、缺失数据声明和双重利益相关者角色帮助居民审视证据并自主形成政策判断，避免了传统仪表盘和单智能体AI在用户检视证据前就给出结论的问题，并在基准测试中实现了98.4%的事实声明证据支持率。 |
| [^67] | [Differentiable Filtering for Inertial Odometry](https://arxiv.org/abs/2611.13517) | 本文提出DFIO框架，将误差状态卡尔曼滤波器的测量更新阶段嵌入计算图实现端到端可微分训练，联合优化位移与协方差输出头，使行人惯性里程计的平均绝对平移误差和一秒相对平移误差分别降低13.2%和39.6%。 |
| [^68] | [Mobile AI Factory: Building Specialized Language Models for Your Mobile Device](https://arxiv.org/abs/2611.13509) | Mobile AI Factory是一个智能体系统，能够自动为目标任务和目标手机构建经过微调与量化的专用小型语言模型，通过编排器LLM自主决策超参数调优、真实数据挖掘或合成数据生成，并借助目标手机的真实测量数据实现高性能端侧部署。 |
| [^69] | [Evaluating Compositional Generalization in Transformers: The Role of Composition Equivalence and Module Coverage](https://arxiv.org/abs/2611.13498) | 本文通过系统性评估发现，训练-测试分布中的组合等价性与模块覆盖度是决定 transformer 组合泛化性能的两个关键因素，直接模型的优异表现主要来自对组合等价性的利用，一旦消除这些等价性，性能将显著下降。 |
| [^70] | [Leveraging Psychophysical Attentional Distribution for Gaze-Augmented Reward Modeling](https://arxiv.org/abs/2611.13477) | 该论文基于认知心理学中人类字母识别的发现，将视觉注意力建模为以注视点为中心的空间梯度分布并融入RLHF奖励模型训练，从而实现了比基线更高的偏好预测准确率。 |
| [^71] | [Inherited Weights: Measuring the Tokenization Premium and the Sovereignty Cost of Language Models for the Global South](https://arxiv.org/abs/2611.13441) | 论文指出微调模型会从基础模型继承词表、知识、文化对齐、数据与许可证等决定性属性，并测量出全球南方语言承受最高达10倍的分词溢价，证明构建本地化词表可将该溢价显著降低约24%，从而揭示了大语言模型给全球南方带来的主权成本。 |
| [^72] | [Off-target posterior mass: A training-free diagnostic for fine-tuning spoken language diarization probes](https://arxiv.org/abs/2611.13430) | 本文提出一种免训练的诊断指标——脱靶后验质量（OTPM），基于后验语言识别概率预测微调预训练语音模型为口语语言日志化（SLD）带来的性能提升，其与探针增益呈正相关（合成双语数据上为0.56，真实语料上为0.23）。 |
| [^73] | [BioMem: Agentic Memory for Biomedical LLM Agents](https://arxiv.org/abs/2611.13426) | 本研究提出BioMem，系统评估了生物医学领域中多种LLM智能体记忆方法在任务有效性、检索质量、信息保真度和长期鲁棒性方面的表现与挑战。 |
| [^74] | [Free Lunch Uncertainty: Sequence-Level R\'enyi Entropy from Particle Decoding for Adaptive LLM Reasoning](https://arxiv.org/abs/2611.13414) | 该论文发现SMC幂采样解码可零边际成本获得序列级Rényi熵，其中1/2阶熵在13个设置中的10个中表现最佳，是用于LLM自适应推理的零成本选择性预测信号。 |
| [^75] | [Orthographic depth at two scales, with English also matched for lexicon size](https://arxiv.org/abs/2611.13400) | 该论文提出在两个粒度（对齐单元与音节）上用条件熵度量正字法深度，对十二种字母文字的分析发现英语不确定性最大、芬兰语最小，且排序结果对英语不同转写方式保持稳健。 |
| [^76] | [Mapping Instructors' Attitudes Towards Cheating with AI: An Exploratory Analysis Using Forum Analysis](https://arxiv.org/abs/2611.13366) | 本研究通过分析r/Professors论坛上201万条帖子，揭示了ChatGPT发布后大学教师对AI作弊的态度、防止作弊的动机及所采用的策略。 |
| [^77] | [Understanding Students' Rationales for AI-Mediated Cheating: An Instructor Perspective from Online Forums Analysis](https://arxiv.org/abs/2611.13358) | 该研究通过分析r/Professors论坛上201万条帖子，从教师视角首次大规模揭示了学生AI作弊背后的四大动机：博弈学术系统、数字-学术素养差距、表现焦虑以及不诚信行为的正常化。 |
| [^78] | [Measuring Definition Sensitivity in LLM Annotation of Social Science Concepts](https://arxiv.org/abs/2611.13339) | 该论文提出“定义敏感性”这一新指标：让LLM使用不同定义对同一批文本分别进行标注，通过两组标注结果的分歧程度来衡量模型是否真正遵循了给定定义，还是仅仅依赖其预训练中形成的固有概念认知。 |
| [^79] | [MicroLens: Read, Trace, and Rewire the Residual Stream with Two Lenses](https://arxiv.org/abs/2611.13322) | 本文提出交互式工具 MicroLens，将 logit 透镜与雅可比透镜（J-lens）统一到同一张残差流“层×词元”网格上，实现对隐藏状态的读取、来源追踪与在线编辑，并发现 logit 透镜方向适合控制词元输出、J-lens 方向适合改变后续词元的读取。 |
| [^80] | [Decode Token Parallelism: Memory-Efficient Distributed Attention Parallelism for Scalable LLM Decoding](https://arxiv.org/abs/2611.13293) | 提出解码令牌并行（TKNP）架构，通过将权重放置与注意力所有权解耦，解决了现有并行策略中KV缓存容量与注意力带宽难以兼顾的问题，实现了内存高效且可扩展的大语言模型解码。 |
| [^81] | [What drives the emergence of symbolic mechanisms in large language models?](https://arxiv.org/abs/2611.13270) | 本研究通过从零训练神经网络并系统地操控架构与数据分布因素，发现数据多样性（而非架构归纳偏置）是驱动大语言模型中符号机制涌现的关键因素。 |
| [^82] | [Domain-Routed Delta Optimization for Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2611.13249) | 本文提出域路由增量优化方法（DRD-MOPD），通过为不同领域的样本指定各自更新的参数组件（以专家模型与基础模型的权重差初始化），在多教师在线蒸馏中显著保留原始专家的几何结构。 |
| [^83] | [Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models](https://arxiv.org/abs/2611.13223) | 本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。 |
| [^84] | [Latin American Studies since the 1980s: From Apocalyptic Dreams to Assimilationist Consolidation](https://arxiv.org/abs/2611.13216) | 该研究通过文献计量学揭示了拉丁美洲研究自1982年以来的“同化式整合”趋势——学科制度化地位提升的同时，却强化了以英语学术规范为主导的等级体系，导致拉美本土期刊边缘化和拉美学者向北方标准的趋同。 |
| [^85] | [HiCAP: Hierarchical Cultural Adaptation via Personality for Dialogue Generation](https://arxiv.org/abs/2611.13207) | 提出HiCAP框架，将个性特质通过分层控制机制融入对话生成——先使回应与文化基调对齐、再依据个性细化语言表达与内容，从而实现文化自适应的对话生成。 |
| [^86] | [One-Sided Expert-Parallel All-to-All for Mixture-of-Experts Inference over NVLink](https://arxiv.org/abs/2611.13188) | NVLinkOneSided提出了一种面向NVLink域内混合专家推理的单边all-to-all通信方法，通过免元数据交换的直接推送、原地读取机制以及Fence和Sentinel两种同步协议，在Blackwell GPU上比DeepEP快1.2-3.0倍。 |
| [^87] | [Agreeing on the Answer but Not the Reason: Measurement Limits of LLM Reasoning Failure Taxonomies](https://arxiv.org/abs/2611.13140) | 该论文表明，LLM推理失败分类的测量结果很大程度上取决于测量程序本身：过半失败案例可归入多个类别，类别占比随分类器测试顺序大幅波动（26.4%–76.4%），且人工标注者对类别归属的共识远低于对任务答案正确性的共识。 |
| [^88] | [Which Successes Make Good References? Action-Sequence Alignment for Multi-Turn On-Policy Distillation](https://arxiv.org/abs/2611.13135) | 该论文将多轮在线策略蒸馏中的成功参考选择建模为自由目标最优传输问题，并提出以最长公共子序列（LCS）为基础的动作序列对齐作为轨迹兼容性的可观测代理，为每条失败轨迹匹配最兼容的成功参考，从而避免误导性监督、提升蒸馏效果。 |
| [^89] | [Activation-Guided Graph Retrieval for Cognitive Memory Reconstruction in Language Model Systems](https://arxiv.org/abs/2611.13127) | 该论文提出了一种激活条件化的记忆图检索架构，通过请求与目标线索激活授权记忆图、扇出衰减控制有界遍历、实体递归发射线索，并在生成前重构类型化证据图，从而为长寿命AI系统实现认知记忆重构。 |
| [^90] | [ASPIRE: An Agentic System-Identification Platform for Iterative Reasoning and Exploration](https://arxiv.org/abs/2611.13114) | ASPIRE是一个由大语言模型智能体驱动的非线性系统辨识框架，能够自主选择并执行辨识方法，从测量数据构建闭式非线性模型，融入先验物理知识并独立验证，且支持自主与交互两种运行模式。 |
| [^91] | [Catch It Where It Happens: Real Time Error Correction During LLM Reasoning](https://arxiv.org/abs/2611.13107) | 提出在线确定性验证方法，利用计算机代数系统等代码工具在LLM生成推理步骤的同时实时校验（无需GPU或额外模型调用），发现错误时让模型从出错处继续而非从头重来，从而避免错误传播并节省推理开销。 |
| [^92] | [One Temperature to Rule Them All?](https://arxiv.org/abs/2611.13077) | 该论文研究了 pass@K 推理中温度多样性何时优于单一温度，发现虽然更难的问题更难共享高效的采样温度，但这一趋势仅随难度呈对数增长，表明单一调优温度在多数情况下已足够有效。 |
| [^93] | [The Left Ripple of Byte-Pair Encoding: Certified Lookbehind for Every Dictionary](https://arxiv.org/abs/2611.13061) | 该论文对任意字节对编码词典完整解决了“左侧前置字符串能把分词改变多远”的问题：起源半径可由双自动机乘积给出的半径证书在多项式时间内判定，且恰好在存在一个具有两个无公共边界的规范铺砌的周期词（在排序条件下即某规则 的两个副本在其自身阶段相邻）时为无限。 |
| [^94] | [A Multi-Word Tokenizer for 204 Languages That Needs Fewer Tokens Than GPT-4o's, Including on English](https://arxiv.org/abs/2611.13059) | 该论文提出Havon多词分词器系列，通过在约200种语言上从头训练并引入多词训练阶段，在包括英语在内的161种语言上所需词元数量少于GPT-4o的分词器，在印度语言上的词元数中位数仅为GPT-4o的0.74倍。 |
| [^95] | [The Indic Token Tax in 2026: Measuring Tokenizer. Disparity and Its Cost for the Languages of India](https://arxiv.org/abs/2611.13058) | 该研究测量了九个主流分词器对印度25种语言的“代币税”，发现印度语言所需token量最高可达英语的13.7倍（GPT-4o分词器），导致印度语言用户在商业LLM使用中面临更高的费用、更慢的速度和更小的有效上下文，而Sarvam分词器是唯一将差异控制在2.6倍以内的第三方选择。 |
| [^96] | [GTAGym: Real AAA Game Tasks Environment for Evaluation and Reinforcement Learning](https://arxiv.org/abs/2611.13028) | 本文提出GTAGym——一个基于《侠盗猎车手V》真实AAA游戏任务的评估与强化学习环境，其基准GTAVBench包含103个跨九类任务家族的在线主线任务及47个图像式射击狙击任务，支持截图与键鼠操作的并行测试并自动校验任务结果，用于评测前沿大语言模型的智能体能力。 |
| [^97] | [Repository-Context-Guided Image Understanding for Agentic Software Development](https://arxiv.org/abs/2611.13026) | 提出通过从同一代码库渲染清晰的参考截图来增强编程智能体，从而在查询截图质量退化、视觉证据不清晰时，仍能准确识别UI元素并定位相关源文件。 |
| [^98] | [khammoon: From a Linguistic Guideline to a Thai Morpheme Segmenter](https://arxiv.org/abs/2611.13012) | 该论文提出基于语言学准则构建的泰语语素切分器khammoon，通过在语素（最小意义单位）层面进行切分消除复合构词歧义，且语素级模型在词性标注、命名实体识别和情感分类等下游任务上表现与词级基线相当甚至更优。 |
| [^99] | [Seeing Is Not Believing: Counterfactual Lesion Erasure Teaches a Small Medical Vision-Language Agent to Report Only What the Image Shows](https://arxiv.org/abs/2611.13006) | 该论文提出无需人工标注的反事实病灶擦除评测范式与证据敏感度差距（ESG）指标，并据此训练出40亿参数的医学视觉-语言智能体MedScope-4B，使其只报告图像中真实可见的发现，将“幻觉报告被擦除病灶”的比例从前沿模型的41–62%降至9.6%。 |
| [^100] | [Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Large Language Models](https://arxiv.org/abs/2611.12998) | 该论文提出一个四阶段贝叶斯分析流程，将探针准确率分解为可解码性、形状、归属权和因果使用四个独立命题并分别检验，从而更严谨地判定大语言模型激活中的几何结构（如编码算术运算的螺旋或圆形）是否真正被模型用于计算。 |
| [^101] | [Capacity Allocation at the Source: Sparse Target Optimization for LLM Knowledge Editing](https://arxiv.org/abs/2611.12995) | 该论文建立了参数空间干扰与目标隐状态干扰之间的正相关理论联系，并提出一种 $\ell_1$ 正则化方法对知识向量进行稀疏化，从源头优化目标隐状态构建以更好地分配模型有限容量空间。 |
| [^102] | [ZeroPoint: Lightweight Energy-Based Decision Model](https://arxiv.org/abs/2611.12992) | 本文提出ZeroPoint，通过引入一个对任意输入能量恒为零的“空答案”固定状态，为基于能量的决策模型提供显式的能量参考水平，使模型能够判断并拒绝“没有任何可接受选项”的情形。 |
| [^103] | [Tracing Psychometric Inference in Large Language Models](https://arxiv.org/abs/2611.12937) | 该研究通过跨角色预测范式发现，大语言模型（尤其是较大规模的指令微调模型）在行为上重现并放大了人类心理量表的跨量表相关性，且其内部几何结构与人类心理测量结构对齐，表明模型在一定程度上确实内部表征了心理构念。 |
| [^104] | [Using LLM Agents to Evaluate Prosocial Interventions on Social Media Platforms](https://arxiv.org/abs/2611.12922) | 该研究基于OASIS框架构建了一个包含13,835个大语言模型智能体的仿真评估流水线，系统评估了三种提升经核实新闻曝光度的干预措施，发现将新闻提升至信息流显著位置的效果最佳，为在开展昂贵的人类实验前筛选干预方案提供了系统方法。 |
| [^105] | [IntentTwin: Deconfounding Candidate Support and the Oracle-to-Deployment Gap in Offline Web Behavior Prediction](https://arxiv.org/abs/2611.12918) | 该论文揭示了离线网页行为预测基准中仅凭候选动作是否属于训练支持集即可“作弊”答对的混杂效应，提出支持可识别率（SIR）这一度量、支持均衡候选集构建方法以及免目标双路径提议器，从而真实刻画从理想评测条件到实际部署的性能差距。 |
| [^106] | [M\"obius Interaction Filtering: Order-Selective Spectral Regularization for Multimodal Humor Detection](https://arxiv.org/abs/2611.12913) | 提出莫比乌斯交互过滤方法，通过阶数选择性谱正则化在保留高阶模态交互表示的同时调控其对预测的贡献，使多模态幽默检测的标签对齐信噪比提升1.62-2.79倍。 |
| [^107] | [Brasilidade: Measuring Brazilian Identity for Large Language Models](https://arxiv.org/abs/2611.12906) | 该论文提出了Brasilidade基准，通过涵盖语言、语法等四个领域的标准化任务评估大语言模型的巴西葡萄牙语能力，发现模型规模并不决定性能，且葡萄牙语专用模型的表现也未超过平均水平。 |
| [^108] | [Actionable Feedback: Improving Model Capabilities from User Edit Data on Clinical Documentation](https://arxiv.org/abs/2611.12904) | 该论文针对临床文档场景中用户编辑数据并非全部可操作的问题（如45%以上编辑包含模型输入不支持的信息），构建了临床医生编辑行为分类体系并开发了区分可操作反馈的方法，以避免微调导致模型产生幻觉。 |
| [^109] | [SELD: Detecting Silent Evidence Loss in Context Compression beyond the Answer](https://arxiv.org/abs/2611.12903) | 提出SELD——一种无需训练和标注的方法，通过比较阅读模型压缩前后对未说出口候选答案的偏好变化，检测上下文压缩中“答案不变但关键证据被删”的隐性证据丢失问题，并构建了SEL-Pairs评测基准。 |
| [^110] | [TimeOmni-o1: Bidirectionally Coupling Time Series Foundation Models and LLMs for Multimodal Reasoning and Forecasting](https://arxiv.org/abs/2611.12902) | 提出TimeOmni-o1，通过将预训练的时间序列基础模型与大语言模型双向耦合，实现文本与时间序列两种模态的联合读取与生成，从而统一完成多模态理解问答和未来值预测两类任务。 |
| [^111] | [Rethinking in Spikes: Mitigating Hallucinations in MDLMs with Step-Aware Decoding](https://arxiv.org/abs/2611.12901) | 提出了StepRefine，一种即插即用的步感知解码策略，通过检测去噪过程中的熵尖峰并在离散去噪与潜在精炼之间动态切换，从而缓解多模态扩散语言模型的幻觉问题。 |
| [^112] | [From Relevant Posts to Attributable Spans: Explicit Symptom Evidence for Depression Questionnaire Inference](https://arxiv.org/abs/2611.12900) | 提出SREE证据驱动流水线，通过提取第一人称、症状特异的文本片段，用设施选址思想筛选代表性证据，并采用证据门控推断，从而更可靠地从社交媒体历史推断抑郁症问卷结果。 |
| [^113] | [MNN-Pocket-0.3B Technical Report](https://arxiv.org/abs/2611.12311) | MNN-Pocket-0.3B 是一个面向端侧部署的 3 亿参数紧凑语言模型，通过架构、数据、训练流程和推理系统的协同设计，实现了仅 173 MB 的 INT4 体积、最高 466 tokens/s 的解码速度和 32K 上下文支持。 |
| [^114] | [FinCutBench: A Continually Updated Benchmark for Measuring the Temporal Boundaries of LLM Financial Knowledge](https://arxiv.org/abs/2611.11588) | 该论文提出FinCutBench，一个每月持续更新的中英双语金融知识基准，通过按月构建知识画像并定位“FinCut边界”，来衡量大语言模型金融知识的时间截止点，以检测金融回测中的参数化前视偏差。 |
| [^115] | [When Should Diffusion Language Models Reconsider Their Commitments?](https://arxiv.org/abs/2611.11551) | 该论文提出PURL框架，通过联合学习编辑选择、取消掩码选择和词元预测三个决策，使掩码扩散模型能够在并行快速生成的同时有选择地修正早期不一致的词元承诺，且推理时三个决策仅需一次共享前向传播即可完成。 |
| [^116] | [ScrapeBench: Evaluating Legal Compliance of AI Agents in Website Scraping](https://arxiv.org/abs/2611.11306) | 该论文提出首个评估AI智能体网站抓取法律合规性的基准ScrapeBench，发现大多数智能体在抓取明确禁止抓取的网站时合规率低于5%，即使主动检查了ToS和robots.txt也依然如此，而在允许抓取的网站上却有时会错误地拒绝抓取。 |
| [^117] | [WarBias: A Bilingual LLM Benchmark of Stereotypes About Internally Displaced People and Veterans](https://arxiv.org/abs/2611.11284) | 该论文提出了首个聚焦退伍军人和境内流离失所者这两个被主流偏见评估忽视群体的双语（乌克兰语-英语）大语言模型偏见评测基准WarBias，包含120条刻板印象主张、每种语言1,007个句子三元组和240道多选题，填补了语言模型社会偏见评估的空白。 |
| [^118] | [Machine Hearing: Can Audio LLMs Generalize Beyond Human Auditory Perception?](https://arxiv.org/abs/2611.10693) | 提出了机器听觉基准（MHB），通过涵盖7个领域、13个任务、采样率跨越约10^-5至10^6 Hz的6,128个实例，系统评估音频大语言模型能否超越人类听觉范围实现更广泛的机器听觉泛化能力。 |
| [^119] | [CulturalCrosswords: A Multilingual Benchmark of Answer Constraints and Grid Completion](https://arxiv.org/abs/2611.10044) | 该论文构建了覆盖42种语言、53个填字游戏的多语言基准CulturalCrosswords，发现长度提示和网格约束等答案约束能显著提升大型语言模型解答文化谜题时答案的准确性与一致性。 |
| [^120] | [Counting Events, Not Seeing Them: Diagnosing and Repairing the Readout Failure in Video LLMs](https://arxiv.org/abs/2611.09949) | 该论文发现视频大语言模型物体计数失败的根源并非缺少物体恒常性，而是计数信号已存在于内部表征中却未被正确读取，并通过片段定位监督与离散事件符号辅助通道两种修复方法，将同一物体事件计数准确率从46.7%大幅提升至65.0%。 |
| [^121] | [BEiTScore++: Reference-free Image Captioning Evaluation with BEiT-3 and Conformalized Quantile Regression](https://arxiv.org/abs/2611.09505) | 提出BEiTScore++，通过共形化分位数回归为无参考图像描述评估指标引入共形校准的置信区间，在单次前向传播中同时实现准确评分与可靠的不确定性量化。 |
| [^122] | [PathXAI: Evidence-guided structured pathology reporting by transferable tissue segmentation](https://arxiv.org/abs/2611.09117) | PathXAI通过可迁移的组织分割信息结合H&E图像与领域知识来引导诊断推理和结构化病理报告生成，在诊断一致性和形态学描述准确性上显著优于纯H&E方法。 |
| [^123] | [Towards Foundation Decision Models: Drex](https://arxiv.org/abs/2611.07854) | 提出Drex系列开源决策模型（8B至35B参数），通过监督微调与调整反馈强化学习（RLAF）训练，无需自回归解码即可对用户提供的选项直接输出概率分布，从而降低生产环境中的成本、延迟与答案不一致性。 |
| [^124] | [ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus](https://arxiv.org/abs/2611.06706) | 该论文通过与约旦、西非和肯尼亚的民间社会组织合作，以社区主导的众包方式构建了首个涵盖黎凡特阿拉伯语、非洲法语和斯瓦希里语的技术助长性别暴力标注语料库ALTO，填补了现有性别伤害检测研究长期偏向全球北方语言的空白。 |
| [^125] | [Reading Why: Born-Measured Attention with Exact Word Evidence on Light and Qubits](https://arxiv.org/abs/2611.05355) | 该论文提出一种由玻恩测量构成的变换器（玻恩归一化注意力、测量乘积与玻恩读出），使类别概率成为词振幅的多项式，从而可通过有限求积无误差地精确计算每个词的积分梯度证据，并在 e-SNLI 上取得与同规模 softmax 变换器相当的准确率。 |
| [^126] | [FindLink: Learning to Assess Scientific Connections from Source-Grounded Premises](https://arxiv.org/abs/2611.02545) | 该论文提出 GI-TraceBench 基准与 FindLink 两阶段框架，通过构建来源锚定前提并进行结构化推理，结合确定性编译器反馈，来评估不同研究条件下科学发现之间的关联。 |
| [^127] | [Muon Makes the First Mixture-of-Experts Layer Harder to Balance](https://arxiv.org/abs/2610.37914) | 研究发现 Muon 优化器会使混合专家语言模型第一个 MoE 层的路由偏置跨度平均翻倍，导致该层负载严重失衡（最繁忙专家承担四到七倍公平份额），且这一效应源于 Muon 对网络其余部分的作用而与路由器的训练方式无关。 |
| [^128] | [Text Generation in Low-Resource Languages Using Human-Curated Vocabularies](https://arxiv.org/abs/2610.37105) | 该论文提出了一个结合人工整理词汇表与少量人工编写范例、并按Zipf分布采样的框架，使大语言模型能够为预训练数据中极端低资源甚至完全未见过的语言生成连贯文本。 |
| [^129] | [SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2610.36991) | SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。 |
| [^130] | [Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss](https://arxiv.org/abs/2610.36202) | 本文提出人格-头部似然评估（PHLE）方法，利用基础预训练模型对输出的似然评估来解码并分类后训练模型中的潜意识隐蔽信号，从而检测出提示分类器无法发现的隐藏意图（如隐蔽的伤害用户倾向）。 |
| [^131] | [Integrate the Reward, Don't Sample It: Window Likelihood for Interruptions and Backchannels in Full-Duplex Speech Models](https://arxiv.org/abs/2610.35581) | 该论文发现基于整轮展开的GRPO时机奖励在训练全双工语音模型时存在零梯度陷阱与信用泄漏两大缺陷，并提出“窗口似然”方法，通过对停止决策进行闭式积分直接最大化正确让出/保持的对数概率，在HumDial-FDBench上取得了最佳的让出/保持F1表现。 |
| [^132] | [The Value of Memory in Language Agents](https://arxiv.org/abs/2610.31303) | 该论文提出一个决策论框架，以所避免的任务损失为单位，量化语言智能体中记忆从原始历史到持久状态、操作使用及代价高昂的恢复与修正全过程中的价值，并通过成本感知的模拟条件和五个最小反例揭示信息时机、检索访问、组件交互、反馈和需求不确定性对记忆价值的关键影响。 |
| [^133] | [Hearsay or Fact? How Framing and Elicitation Shape LLM Probability Updates](https://arxiv.org/abs/2610.27745) | 该论文提出了一种衡量大语言模型对单条新信息概率更新幅度的“方向性移动”指标，并发现信息的框架呈现方式（直接告知还是传闻转述）与概率引出方式（单个概率还是五个加权估计）会显著影响模型的概率更新。 |
| [^134] | [Compute-Optimal Scaling Laws for Spiking Language Models](https://arxiv.org/abs/2610.15146) | 本文首次对脉冲语言模型开展Chinchilla式缩放研究，发现其计算最优缩放指数（0.56）与密集Transformer（0.50）接近，且脉冲化带来的损失代价随计算预算增加而稳步下降。 |
| [^135] | [Easy to anticipate, hard to compute: boundary dependence finds the computed outputs that entropy patching misses](https://arxiv.org/abs/2610.11790) | 论文揭示了字节级语言模型中基于熵的分块策略会系统性遗漏“类型可预料但数值需计算”的位置（如数学等式的计算结果），并提出将熵信号与无标签的边界依赖信号相结合来放置补丁边界，从而大幅提升模型在这类计算输出上的准确率。 |
| [^136] | [GAGR-Lab: Evaluating Joint Spatial-Geometric and Analytic Function Reasoning](https://arxiv.org/abs/2610.10201) | 该论文提出GAGR-Lab框架，通过笛卡尔游戏场景与Rust轨迹执行评估模型将空间配置转化为满足几何约束的解析函数的联合推理能力，试点实验表明当前视觉语言模型在此任务上尚无法命中目标。 |
| [^137] | [LARC: Low-Rank Adaptive Residual Connections for Learning in Frozen Models](https://arxiv.org/abs/2609.40063) | LARC 通过在冻结模型中引入低秩自适应残差连接，利用跨任务学习初始因子的慢状态与随反馈动态调整的快状态相结合的双状态机制，使冻结模型无需更新原始参数即可从反馈中学习，在程序选择任务中显著降低查询执行误差。 |
| [^138] | [T-SANDHI: Tone Sandhi-aware Adaptive Network with Decoupled Hybrid Injection for Low-resource Taiwanese Hokkien Speech Recognition](https://arxiv.org/abs/2609.18194) | 该论文发现台湾闽南语语音识别的真正瓶颈并非连读变调本身，而是变调与本调之间的局部混淆，并提出T-SANDHI模型，通过在冻结的Whisper骨干上显式解耦表层声学与词汇意图，结合词典引导的多任务学习和动态门控混合注入模块，有效提升了低资源台湾闽南语的语音识别性能。 |
| [^139] | [Quantifying the Generation Modality Gap in Speech-Text Language Models](https://arxiv.org/abs/2609.14743) | 该论文构建了一个统一的生成式评估套件，在匹配的数据分布和评估设置下，从语义连贯性、语音结构、声学质量等多个维度量化了纯语音、纯文本和语音-文本语言模型之间的生成模态差距。 |
| [^140] | [Characterising divergence between heuristic and language-model ratings of mobile user interface quality](https://arxiv.org/abs/2609.05423) | 该研究首次在大规模上比较了大语言模型与基于规则的启发式方法对移动用户界面质量的评分，通过270,000个评分发现两类评分来源之间的一致性持续偏弱（kappa普遍低于0.15），揭示了大语言模型评分与既有启发式标准之间存在系统性分歧。 |
| [^141] | [ST$^2$U: Stateful Test-Time Unlearning via Restricted Knowledge Boundary Control](https://arxiv.org/abs/2608.23034) | ST²U通过轨迹级别的受限知识边界控制，解决了现有测试时遗忘方法因忽略自回归生成中状态重建而导致的受限知识重入问题。 |
| [^142] | [A Factorial Ablation of a Speech-to-SFT Pipeline: Differential Effects on Data Quality and Downstream Transfer](https://arxiv.org/abs/2608.20394) | 通过2x2因子消融实验发现，语音到SFT流水线中的数据质量改进虽能提升评审质量，但并未显著提升下游MCQA性能，表明质量提升的迁移效果有限。 |
| [^143] | [Auditing Cross-Lingual Fairness in Language Model Watermarking](https://arxiv.org/abs/2608.20047) | 提出一个四部分评估框架，用于审计语言模型水印在跨语言场景中的公平性，并通过实证校准和多种测量揭示英语评估掩盖的多语言部署问题。 |
| [^144] | [Time as Structure: Temporal Dependency Graphs for Verifiable Deadline Computation over Legal Documents](https://arxiv.org/abs/2608.15270) | 本文提出通过时间依赖图和日历引擎进行法律截止日期的可验证计算，显著优于语言模型，后者在多个案例中逻辑自相矛盾且错误地将逾期判为及时。 |
| [^145] | [Multimodal Model Diffing for Feature Discovery and Control](https://arxiv.org/abs/2608.09928) | 提出了MMDiff多模态模型对比框架，通过训练多模态稀疏自编码器并将其作为特征级接口，实现多模态大语言模型中被多模态训练改变的特征的发现、隔离与针对性控制。 |
| [^146] | [Detecting Hallucinations and Recovering Verified Answers in Arabic Islamic Question Answering](https://arxiv.org/abs/2608.03720) | 本文基于微调的Gemma-4-12B-it模型构建了一个统一的两步系统，用于检测大语言模型在阿拉伯语伊斯兰问答中产生的幻觉并从六个候选选项中恢复经过验证的正确答案，取得了0.928的Macro-F1和0.935的标签准确率。 |
| [^147] | [Probing the Origins of Reasoning Performance: Representational Quality for Mathematical Problem-Solving in RL vs. SFT Fine-Tuned Models](https://arxiv.org/abs/2607.26119) | 本研究通过逐层线性探针和均值消融实验揭示，RL训练的推理语言模型内部具有更高质量、更易线性解码出答案正确性的表征，且其深层表征对消融更敏感，从而为RL在数学推理上优于SFT提供了机制层面的解释。 |
| [^148] | [PrefReward: Learning User Preference Matrix for Personalized Text Generation](https://arxiv.org/abs/2607.21067) | 提出PrefReward框架，通过结构化用户偏好矩阵显式建模个体风格，并作为基于KL散度的奖励信号引导解码过程，实现更可解释且能处理长上下文的个性化文本生成。 |
| [^149] | [Reasoning Before Translation: Enhancing Legal Machine Translation with Structured Reasoning](https://arxiv.org/abs/2607.19181) | 该论文以瑞士多语言法律体系为测试平台，比较了结构化推理、监督微调和强化学习等不同方法对法律机器翻译的增强效果，系统评估了经过再训练的小型语言模型与前沿推理模型的性能表现。 |
| [^150] | [Einstein World Models](https://arxiv.org/abs/2606.26969) | 提出爱因斯坦世界模型（EWM），通过在推理轨迹中嵌入视觉-时间推演，使大型语言模型能够利用可视化反事实事件来增强复杂推理能力。 |
| [^151] | [MorfFlex: Handling Rich Morphology](https://arxiv.org/abs/2606.24366) | 本文提出了一种通过屈折和派生模式系统来紧凑编码海量词形的形态词典架构MorfFlex，并发布了包含超过1亿词形的捷克语形态词典MorfFlex CZ，为布拉格依存树库的人工标注一致性和MorphoDiTa等先进自动工具提供了关键资源。 |
| [^152] | [Meet UD_Czech-PDTC: A Large and Genre-Rich Treebank in Universal Dependencies](https://arxiv.org/abs/2606.24337) | 本文介绍了将规模扩大一倍以上且体裁更加多样化的布拉格依存树库整合版（PDT-C）转换为通用依存格式（UD_Czech-PDTC）的过程，并通过实例详细分析了两种标注方案在依存结构拓扑和词性、关系类型粒度上的诸多细微差异。 |
| [^153] | [Prague Dependency Treebank -- Consolidated 2.0: Enriching a Complex Annotation Scheme](https://arxiv.org/abs/2606.24324) | 布拉格依存树库整合版2.0（PDT-C 2.0）历时近30年开发完成，是一个近400万词符、标注统一连贯、体裁多样并系统整合了共指和篇章关系等多层次语言标注的捷克语资源。 |
| [^154] | [Self-Modeling Interventions can Modulate Emergent Misalignment](https://arxiv.org/abs/2606.23700) | 本研究提出将自我识别和自我报告等自我建模干预作为针对模型品格的防御手段，实验证明自我识别微调能有效逆转和预防涌现性失准，且是唯一能持续降低失准而不恶化其他指标的预防性方法。 |
| [^155] | [BALTO: Balanced Token-Level Policy Optimization for Hallucination Mitigation](https://arxiv.org/abs/2606.15893) | 提出了BALTO框架，通过提取可验证的事实性声明并将其判断投影为词元级标签，结合平衡的词元级信用分配机制，解决了现有强化学习方法中的粒度不匹配与偏差问题，从而有效缓解大语言模型的幻觉。 |
| [^156] | [Beyond Third-Person Audits: Situated Interaction Auditing for User-Centered LLM Bias Research](https://arxiv.org/abs/2606.12247) | 该论文提出了情境化交互审计（SIA）这一共享诊断框架，将大语言模型偏见研究从第三人称审计扩展到以用户为中心的第一人称交互审计，填补了用户在偏见审计中缺席的结构性盲点。 |
| [^157] | [Can AI Agents Synthesize Scientific Conclusions in Health?](https://arxiv.org/abs/2606.11337) | 该论文提出了SciConBench——一个包含9,210个问题及专家结论的大规模实时基准，通过原子事实分解和事实精确率/召回率评估，并配合防数据泄露的洁净室框架SciConHarness，揭示出当前前沿AI智能体在健康领域科学结论综合上的事实质量仍然较低。 |
| [^158] | [Building Comparative Motivation Profiles with Symmetric Belief Editing](https://arxiv.org/abs/2606.08243) | 该论文提出“比较动机画像”框架，通过对称信念编辑——即以相互匹配的放大/抑制操作同等地编辑两个差异极小的假设动机所依赖的信念——将意图作为倾向性进行间接测量，并将其应用于四个开源权重模型的对齐伪装研究，以区分“图谋不轨”与“表演性失对齐”两种动机。 |
| [^159] | [From Architecture to Output: Structural Origins of Hallucination in Large Language Models and the Amplifying Role of Data](https://arxiv.org/abs/2606.07537) | 该论文提出了一个仅需采样访问权限的幻觉归因框架，通过针对前缀、上下文和频率竞争的三次有序干预，将大语言模型的单个幻觉追溯归因到自注意力联想检索、最大似然预训练目标或暴露偏差下的自回归承诺等具体组件，并提出五个可证伪的预测。 |
| [^160] | [Semantic Motion Anchors: Bridging Motion and Meaning in Co-Speech Gestures](https://arxiv.org/abs/2605.30608) | 该论文提出“语义运动锚点”方法，通过将3D手势离散化为运动基元并转化为结构化自然语言描述来捕捉手势的物理形态与交际意图，为文本-手势对比学习提供辅助监督，使文本到手势检索R@1提升8.2%。 |
| [^161] | [Pruning and Distilling Mixture-of-Experts into Dense Language Models](https://arxiv.org/abs/2605.28207) | 提出首个将训练好的MoE模型系统性转换为全稠密语言模型的框架，通过专家评分、选择、分组拼接为稠密FFN并经知识蒸馏优化，其中新颖的多样性感知评分方法在多个模型上一致优于已有方法。 |
| [^162] | [StakeBench: Evaluating Language Understanding Grounded in Market Commitment](https://arxiv.org/abs/2605.26074) | StakeBench是一个以真实市场行为（持仓方向、评论后交易、赔率轨迹）替代人工标注作为监督信号的金融语言理解评估基准，通过四个诊断任务和三个承诺感知指标来衡量模型理解说话者真实市场承诺而非表面情绪的能力。 |
| [^163] | [StateRAG: Typed State Contracts for Complex Retrieval-Augmented Generation](https://arxiv.org/abs/2605.25379) | StateRAG 通过类型化的显式检索状态契约与带校验的状态更新机制，协调多角色协作并保证状态一致性，从而在复杂多步检索增强生成任务中优于自由形式方法。 |
| [^164] | [AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild](https://arxiv.org/abs/2605.22715) | AnyMo提出了一种几何感知的框架，通过基于物理的IMU仿真、配对位置视图与掩码观测预训练图编码器、将多位置IMU数据转化为全身运动token并与LLM对齐，实现了不依赖传感器配置（佩戴位置、朝向、硬件等）的人体运动建模。 |
| [^165] | [Firefly: Illuminating Large-Scale Verified Tool-Call Data Generation from Real APIs](https://arxiv.org/abs/2605.17558) | FireFly通过先让大模型探索真实API再从观察结果反向合成任务的新范式，从真实MCP服务器生成在构造上保证标签正确性的大规模工具调用训练数据。 |
| [^166] | [AI-Mediated Communication Can Steer Collective Opinion](https://arxiv.org/abs/2605.16245) | 该论文通过实证研究与数学建模表明，大型语言模型在中介人与人之间的交流时会引入方向性偏见（如偏向支持枪支管制、反对无神论），并因此可能引导社交网络中集体舆论的形成。 |
| [^167] | [Examining the Impact of Data and Training Methods on Emergent and Subliminal Misalignment](https://arxiv.org/abs/2605.12798) | 该论文通过构建按“领域-任务”网格组织的数据集，系统研究了训练数据、评估数据与训练方法如何影响涌现性失对齐（EM）与阈下学习（SL）中失对齐的转移，并发现 EM 主要沿任务而非领域进行转移。 |
| [^168] | [GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling](https://arxiv.org/abs/2604.18556) | 提出了GSQ这一训练后标量量化方法，利用Gumbel-Softmax松弛联合学习逐坐标网格分配与逐组缩放因子，使简单的标量量化在低比特精度下就能达到与复杂的第二代向量量化方法相当的高准确率。 |
| [^169] | [Multi-User Large Language Model Agents](https://arxiv.org/abs/2604.08567) | 该论文首次系统性地研究了多用户大语言模型智能体，将LLM智能体的多用户交互形式化为多主体决策问题，以应对多用户场景下不可避免的冲突、信息不对称和隐私约束。 |
| [^170] | [STDec: Spatio-Temporal Stability Guided Decoding for dLLMs](https://arxiv.org/abs/2604.06330) | 该论文提出STDec方法，利用dLLM解码中“新token靠近已解码邻居、预测ID跨步骤保持一致”的时空稳定性，通过空间感知与时序感知解码动态自适应地调整阈值，从而改进解码效果。 |
| [^171] | [Projection-Free Evolution Strategies for Continuous Prompt Search](https://arxiv.org/abs/2603.13786) | 本文通过实证发现随机投影无法充分捕捉提示词空间的内在低维结构，进而提出一种基于演化策略的无投影连续提示词搜索方法，直接在完整提示词空间中优化并利用与内在维度相适配的自适应机制。 |
| [^172] | [Cold-Start Personalization via Bayesian Adaptive Questioning](https://arxiv.org/abs/2602.15012) | 提出CAPE方法，先离线从完整用户画像中学习偏好相关性的结构化世界模型，再在线通过免训练的贝叶斯推断选择信息量最大的问题并预测完整偏好画像，即使使用简单的线性回归信念模型也能大幅超越端到端强化学习。 |
| [^173] | [Do Reasoning LLMs Refuse What They Infer in Long Contexts?](https://arxiv.org/abs/2602.08874) | 该论文提出“组合推理攻击”方法，发现推理型大语言模型虽然会拒绝明确陈述的有害请求，但当有害意图被分解为碎片分散在长上下文中时，模型的拒绝率会显著下降，暴露了现有安全评估的重要盲区。 |
| [^174] | [Self-Improving Training & Thinking](https://arxiv.org/abs/2601.21343) | 该论文提出一种将安全性、事实性和推理等能力更早融入预训练与中期训练的新方法，通过利用现有强模型重写数据和评判模型输出，并引入“自我改进的后思考”机制，在训练全流程中提升模型的推理能力。 |
| [^175] | [Tracing Moral Foundations in Large Language Models](https://arxiv.org/abs/2601.05437) | 该研究通过对14个大型语言模型进行逐层分析、稀疏自编码器特征识别和因果引导干预，证明模型内部存在与人类道德判断相一致的道德基础概念结构，而非仅仅是表面的“道德模仿”。 |
| [^176] | [DermaVQA-DAS: Dermatology Assessment Schema (DAS) & Datasets for Closed-Ended Question Answering & Segmentation in Patient-Generated Dermatology Images](https://arxiv.org/abs/2512.24340) | 该论文提出了DermaVQA-DAS数据集及专家开发的皮肤病评估模式（DAS），通过36个高层级和27个细粒度的标准化评估问题，为患者生成的皮肤病图像上的封闭式问答和皮损分割两大任务提供了专家标注数据与基准。 |
| [^177] | [Understanding and Tackling Label Errors in Individual-Level Nature Language Understanding](https://arxiv.org/abs/2502.13297) | 本文提出了一种基于个体级因素的NLU标注指南，通过纳入同一个体的所有帖子来重新标注立场检测和情感分析数据集，并揭示了现有数据集中高达31.7%和23.3%的标签错误率。 |
| [^178] | [Linear-time Minimum Bayes Risk Decoding with Reference Aggregation](https://arxiv.org/abs/2402.04251) | 本文提出了一种线性时间的最小贝叶斯风险解码方法，通过使用聚合参考表示来近似配对度量分数，将复杂度降低到线性级别，同时在保持大部分质量增益的同时提高了解码的效率。 |

# 详细

[^1]: AutoScaffold：面向语言模型强化学习的AI引导式脚手架

    AutoScaffold: AI-Guided Scaffolding for Reinforcement Learning of Language Models

    [https://arxiv.org/abs/2611.14515](https://arxiv.org/abs/2611.14515)

    提出了AutoScaffold框架，由冻结的教师模型自动将学生的训练经验转化为自适应脚手架（任务类共享策略与困难任务专用提示），在无需人工干预的情况下提升语言模型可验证奖励强化学习的效果。

    

    可验证奖励强化学习（RLVR）能产生丰富的训练经验，但结果奖励将每次尝试简化为单一数字，使其中大部分信息未被利用。训练时的脚手架，例如在训练期间添加到提示中的策略或提示信息，提供了一种将这些信息转化为指导的自然方式，但其设计与调整在很大程度上依赖人工，且需针对特定领域定制。我们提出AutoScaffold，一个统一的“AI服务AI”框架，它将学生的训练经验转化为自适应脚手架，仅需极少的领域特定信息，且训练过程中无需人工干预。一个冻结的教师模型定期检查这些经验、诊断瓶颈，并以两种形式设计脚手架：（i）一类任务共享的策略，以及（ii）针对个别困难任务的提示，并随着学生的学习动态调整支持的内容和数量。这些脚手架在评估时被移除。

    arXiv:2611.14515v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) produces rich training experience, yet outcome rewards reduce each attempt to a single number and leave most of this information unused. Training-time scaffolds, such as strategies or hints added to the prompt during training, offer a natural way to turn this information into guidance, but their design and adaptation are largely left to humans and tailored to specific domains. We introduce AutoScaffold, a unified AI-for-AI framework that turns a student's training experience into adaptive scaffolds, with minimal domain-specific information and no human intervention during training. A frozen teacher periodically examines this experience, diagnoses bottlenecks, and designs scaffolds in two forms: (i) strategies shared across a class of tasks and (ii) hints for individual hard tasks, adjusting the content and amount of support as the student learns. The scaffolds are removed at evaluat
    
[^2]: 对比发现：基于竞争性解释的开放式科学发现

    Contrastive Discovery: Open-Ended Scientific Discovery over Competing Explanations

    [https://arxiv.org/abs/2611.14514](https://arxiv.org/abs/2611.14514)

    该论文提出“对比发现”框架，用互斥且穷尽的竞争性解释集合取代单一假设检验，通过证据对信念在集合内的重新分配来评估实验、确定被支持的替代解释，从而改进开放式科学发现。

    

    大语言模型智能体正被越来越多地用于开放式、数据驱动的科学发现，然而大多数智能体以逐点方式评估假设，仅将经验性断言 $h$ 与其补集 $\neg h$ 进行对照检验。当证据否定了 $h$ 时，智能体只能得知某一解释不太可能成立，却无法得知证据究竟支持哪个替代解释。为了找回这一替代解释，我们提出了“对比发现”，它用对比集取代单一假设——对比集是一个科学问题与一组互斥且穷尽所有可能的竞争性解释的配对。每个实验根据其证据如何在集合内重新分配信念来进行评分，当信念集中于某一个解释时，该科学问题即被视为解决。由于对比集仅改变了智能体提出和评分的单位，现有智能体可以利用它们来重新评估已收集的证据，或引导其搜索。在六个领域中，89%的逐点候选……（摘要原文在此处截断）

    arXiv:2611.14514v1 Announce Type: new  Abstract: LLM agents are increasingly used for open-ended, data-driven scientific discovery, yet most of them evaluate hypotheses pointwise, testing an empirical claim $h$ only against its complement $\neg h$. When evidence rejects $h$, the agent learns that one explanation is unlikely but not which alternative the evidence supports. To recover that alternative, we introduce Contrastive Discovery, which replaces the single hypothesis with a contrast set, a scientific question paired with mutually exclusive and collectively exhaustive competing explanations. Each experiment is scored by how its evidence redistributes belief across the set, and the question is resolved when belief concentrates on one explanation. Because contrast sets change only the unit an agent proposes and scores, an existing agent can use them to re-evaluate evidence it has already collected or to guide its search. Across six domains, $89\%$ of the pointwise candidates with the
    
[^3]: 智能体评估框架的比较分析：企业分析智能体中的稳定性、检测与发现

    Comparative Analysis of Agent Evaluation Frameworks: Stability, Detection, and Discovery in Enterprise Analytics Agents

    [https://arxiv.org/abs/2611.14493](https://arxiv.org/abs/2611.14493)

    该论文通过三阶段比较研究系统评估了三种企业分析智能体评估框架，发现基于轨迹的LLM评判框架在评估稳定性（97%配置达标）和已知风险检测（F1=0.90）方面综合表现最佳。

    

    在企业环境中部署基于LLM的分析智能体，需要能够可靠检测复杂多工具工作流中故障的评估框架。我们提出了一项针对三种评估框架的三阶段比较研究，每种框架代表一种不同的评估范式（基于轨迹的LLM评判、结合红队测试的纯文本LLM评判、以及确定性启发式方法），并在受控研究环境中使用冻结的执行轨迹应用于两个分析智能体。第一阶段通过重复测量量化了评估框架的稳定性，发现97%的配置达到了稳定性阈值，且稳定性模式与智能体本身无关。第二阶段在72个真值测试用例上测量已知风险的检测能力，其中基于轨迹的框架实现了最佳的精确率-召回率平衡（F1=0.90），而纯文本评估在召回率上持平，但产生的误报多出5倍。第三阶段评估未知风险的发现能力，

    arXiv:2611.14493v1 Announce Type: new  Abstract: Deploying LLM-based analytics agents in enterprise settings requires evaluation frameworks that can reliably detect failures across complex, multi-tool workflows. We present a three-phase comparative study of three evaluation frameworks, each representing a distinct evaluation paradigm (trace-based LLM judging, text-only LLM judging with red-teaming, and deterministic heuristics), applied to two analytics agents in a controlled research setting using frozen execution traces. Phase 1 quantifies evaluation harness stability through repeated measurements, finding that 97% of configurations meet the stability threshold and that stability patterns are agent-independent. Phase 2 measures known risk detection on 72 ground-truth test cases, where the trace-based framework achieves the best precision-recall balance (F1=0.90) while text-only evaluation matches on recall but generates 5x more false alarms. Phase 3 evaluates unknown risk discovery, 
    
[^4]: 机制可解释性需要一种推理逻辑

    Mechanistic Interpretability Needs a Logic of Inference

    [https://arxiv.org/abs/2611.14483](https://arxiv.org/abs/2611.14483)

    该论文指出现有机制可解释性研究需要一套严谨的推理逻辑，提出由五项原则构成的递归逻辑框架，以解决从复杂神经系统中推断机制时反复出现的推理瓶颈问题。

    

    神经网络越来越多地不仅被当作预测系统来研究，而且被当作其内部计算本身就是科学研究对象的复杂系统来研究。机制可解释性（MI）领域的最新进展使得识别与模型行为相关的神经元、特征、回路、群体结构和动态模式成为可能。随着这些工具的不断改进，研究注意力正从“模型内部能观察到什么”转向“这些观察如何支持机制性解释”。我们考察了神经科学与机制可解释性在解决这一问题上的并行努力，并识别出从复杂神经系统中推断机制时反复出现的推理瓶颈。我们提出五项原则，并将其组织成一个递归逻辑，从而形成更严谨的机制推断程序。这些原则提供了一个框架，在两个层面上支持科学发现：它有助于将局部的可解释性发现……

    arXiv:2611.14483v1 Announce Type: new  Abstract: Neural networks are increasingly studied not merely as predictive systems, but as complex systems whose internal computations are themselves objects of scientific inquiry. Recent progress in mechanistic interpretability (MI) has made it possible to identify neurons, features, circuits, population structures, and dynamical patterns associated with model behavior. As these tools improve, attention is shifting from what can be observed inside a model to how those observations support mechanistic explanations. We examine the parallel efforts of neuroscience and MI to address this question and identify recurring inferential bottlenecks when inferring mechanisms from complex neural systems. We propose five principles and organize them into a recursive logic for a more rigorous procedure of mechanistic inference. These principles provide a framework that supports scientific discovery in two senses: it helps turn local interpretability findings 
    
[^5]: 翼龙：突破高效自动翻译评估的极限

    Pterodactyl: Pushing the Limits of Efficient Automated Translation Evaluation

    [https://arxiv.org/abs/2611.14480](https://arxiv.org/abs/2611.14480)

    本文提出了Pterodactyl系统，一个基于LLM-as-a-judge方法、通过ROC分析指导提示工程与模型选择的高度可定制超低成本系统，专为WMT26质量评估任务中的句段级质量分数预测和无错误句段检测而设计。

    

    在本文中，我们介绍了1020 Group参加WMT26自动质量评估共享任务的提交系统。我们的系统Pterodactyl（翼龙）专为子任务2和3设计，即句段级质量分数预测和无错误句段的检测。Pterodactyl是一个高度可定制、超低成本的系统，针对翻译分拣和数据整理中的二分类任务进行了优化。它基于LLM-as-a-judge（大语言模型作为评判者）的方法，通过ROC分析指导提示工程和模型选择。我们还开发了多个专门适配子任务3的Pterodactyl变体。

    arXiv:2611.14480v1 Announce Type: new  Abstract: In this paper, we present 1020 Group's submission to the WMT26 shared task on automated quality evaluation. Our system, Pterodactyl, was engineered specifically for subtasks 2 and 3, that is, segment-level quality score prediction and the detection of error-free segments. Pterodactyl is a highly customizable, ultra-low cost system optimized for binary classification in translation triage and data curation. It is based on an LLM-as-a-judge approach, with prompt engineering and model selection guided by ROC analysis. We also developed several variants of Pterodactyl that were specifically adapted for subtasks 3.
    
[^6]: FERN：基于冻结检索器与学习排序的跨基准Text-to-SQL表格检索

    FERN: Cross-Benchmark Table Retrieval for Text-to-SQL with Frozen Retrievers and Learned Ranking

    [https://arxiv.org/abs/2611.14477](https://arxiv.org/abs/2611.14477)

    FERN不使用任何生成式大语言模型，仅通过融合三个冻结检索器生成候选表格并用LightGBM学习排序器重排，在跨基准的Text-to-SQL表格检索任务上达到了与依赖Claude 3.5的RASL相近的召回率，且无需针对新基准重新训练。

    

    在大型合并目录上运行的Text-to-SQL系统必须首先检索出问题所需的表格。我们在合并的BIRD、Spider和Fiben目录上，与RASL已发表的表格召回率进行了比较，RASL使用Claude 3.5模型来提取关键词并选择表格。FERN（无需生成式大语言模型的特征工程排序）则将三个冻结的检索器融合生成100个候选表格，用多达120个特征来描述每个候选，并使用仅在BIRD训练问题上拟合一次的LightGBM排序器对其进行重新排序。在未参与拟合的Spider数据集上，仅其候选池在15个表格处就达到了RASL已发表的召回率（98.0），且排序器将在前五名中包含所有标准表格的问题比例（complete@5）从83.9%提升至95.6%。在5个表格设置下，FERN在Spider上的召回率与RASL相差不到一个百分点（96.7对97.0），在带有证据提示的BIRD上也相差不到一个百分点（96.6对97.5），而在不带证据提示的BIRD上低8.2个百分点（89.3）。在121个真实BEAVER问题上……

    arXiv:2611.14477v1 Announce Type: cross  Abstract: Text-to-SQL systems over large pooled catalogs must first retrieve the tables a question needs. We compare against the published table recall of RASL, which uses Claude 3.5 models to extract keywords and select tables, on pooled BIRD, Spider, and Fiben catalogs. FERN (Feature-Engineered Ranking with No generative LLM) instead fuses three frozen retrievers into 100 candidates, describes each with up to 120 features, and reorders them with LightGBM rankers fitted once on BIRD's training questions. On Spider, where it was not fitted, its candidate pool alone reaches RASL's published recall at fifteen tables (98.0), and the ranker raises the share of questions with every gold table in the top five (complete@5) from 83.9% to 95.6%. At five tables, FERN's recall is within one point of RASL's on Spider (96.7 vs 97.0) and on BIRD with its evidence hints (96.6 vs 97.5), and 8.2 points below on BIRD without them (89.3). On 121 real BEAVER questi
    
[^7]: 文本编码何时对语言建模有影响？

    When does text encoding matter for language modeling?

    [https://arxiv.org/abs/2611.14459](https://arxiv.org/abs/2611.14459)

    本文提出混合编码方案 Aether，并系统研究了文本编码对字节级语言模型性能的影响，发现中文、韩语等信息密集文字的语言能从压缩编码中显著受益，从而揭示了编码选择带来的跨语言性能差异。

    

    arXiv:2611.14459v1 公告类型：新 摘要：UTF-8 作为语言模型中事实上的编码标准，由于其可变长度设计而引入了跨语言的不平等。然而，这种编码选择对模型性能的影响在很大程度上仍未被研究。在这项工作中，我们研究了文本编码如何影响字节级语言模型的性能，将 UTF-8 与多种其他编码策略进行比较，包括其他基于 Unicode 的方法、特定语言的编码以及我们自己定制的编码。为便于分析，我们提出了 Aether：一种混合编码方案，它在我们的目标设定中使用针对特定文字的编码，同时通过 UTF-8 回退机制保持对完整 Unicode 的覆盖，并利用该框架对四种正字法各不相同的语言进行建模。令人惊讶的是，我们发现具有信息密集字符的语言（如中文、韩语）能从压缩编码中显著受益，而其字符承载……（摘要在此处截断）

    arXiv:2611.14459v1 Announce Type: new  Abstract: UTF-8's role as the de facto encoding standard in language models introduces cross-lingual inequities due to its variable-length design. However, the effect that this encoding choice has on model performance remains largely unexamined. In this work, we investigate how text encoding impacts the performance of byte-level language models, comparing UTF-8 with a wide range of other encoding strategies, including other Unicode-based approaches, language-specific encodings, and our own customized encodings. To facilitate our analysis, we propose Aether: a hybrid encoding scheme that uses script-specific encodings for our target setting while maintaining full Unicode coverage through UTF-8 fallbacks, using this framework to model four orthographically diverse languages. Surprisingly, we find that while languages with information-dense characters (Chinese, Korean) significantly benefit from compressive encodings, languages whose characters carry
    
[^8]: Earlybird：2900万参数的预测式流式语音识别

    Earlybird: Predictive Streaming Speech Recognition in 29M Parameters

    [https://arxiv.org/abs/2611.14421](https://arxiv.org/abs/2611.14421)

    Earlybird是一个仅2900万参数的预测式流式语音识别模型，通过在随机截断的音频上学习预测词语，能在说话者说完中位词之前约90毫秒就正确输出，比现有流式模型提前99至490毫秒。

    

    流式语音识别器通常要等到说话者说完一个词之后才能输出该词，一般延迟90至400毫秒。对于语音代理而言，每一毫秒都至关重要：词语输出得越早，代理就能越早开始搜索或规划回复。我们提出了Earlybird，一个仅有2900万参数的流式识别器（4比特量化下仅16.8 MB）。Earlybird从随机截断的音频中学习，包括在词语说到一半时被截断的音频，并被要求完整预测到该时刻为止所有应当输出的词，同时额外猜测接下来的两个词。其原始token概率为每个词（包括仍在发音中的词）提供了置信度。在清晰的有声读物语音（LibriSpeech test-clean）上，中位词能在说话者说完之前90毫秒就被正确预测，而已发布的流式模型则需要在词结束后90至400毫秒才能输出。在我们计时的涵盖有声读物、TED演讲、公开演讲、财报电话会议和合成语音的全部六个测试集上，Earlybird都是输出最早的模型。

    arXiv:2611.14421v1 Announce Type: cross  Abstract: Streaming speech recognizers output a word only after the speaker has finished it, typically 90-400 ms later. For a voice agent every millisecond counts: the sooner the words are outputted, the sooner the agent can start a search or plan its reply. We present Earlybird, a 29M-parameter streaming recognizer (16.8 MB at 4 bits). Earlybird learns from audio cut off at random points, including partway through a word, and is asked to predict every word that is due by then in full, plus a guess of the next two words. Its raw token probabilities give a confidence for every word including the one still being spoken. On clean audiobook speech (LibriSpeech test-clean), the median word is predicted correctly 90 ms before the speaker finishes it, while released streaming models get it 90 to 400 ms after it ends. Earlybird is the earliest on all six test sets we time, covering audiobooks, TED talks, public speech, earnings calls, and synthetic spee
    
[^9]: 比特还是排列？在小型本地语言模型中将固定推理预算用于权重精度还是选项顺序平均

    Bits or permutations? Spending a fixed inference budget on weight precision or on option-order averaging in small local language models

    [https://arxiv.org/abs/2611.14413](https://arxiv.org/abs/2611.14413)

    本文量化了小型本地语言模型中权重量化精度对答案选项顺序偏差的影响，发现6比特量化不损失准确率而4比特量化会显著加剧答案槽位偏斜，并据此比较了固定推理预算下提升权重精度与选项顺序平均两种方案的性价比。

    

    本地语言模型可以通过以更少的比特数存储权重来节省内存，也可以通过对少量选项重排序的结果取平均来降低对答案选项顺序的敏感性。我们探讨前者是否会使后者变差，以及在固定预算下哪种方式是更经济的花费方式。在Qwen3-0.6B和Qwen3-1.7B上使用模拟的就近取整权重量化（8、6、4和3比特，以fp16为参照），读取ARC-Easy、ARC-Challenge、MMLU和HellaSwag（各300题）上字母A到D的logits，结果显示：6比特相比fp16没有损失准确率（反而分别高出+1.8（+1.0到+2.5）和+0.9（+0.3到+1.5）分，这一点我们无法解释），而8比特使准确率变化-0.7（-1.1到-0.3）和-0.4（-0.7到-0.2）分。在4比特时，答案槽位偏斜从0.15升至0.53（0.6B模型），从0.07升至0.12（1.7B模型）；在相同权重信噪比下施加高斯噪声得到0.15和0.37（两次抽样）以及0.19（一次抽样），因此该对照实验对于取整是否增加了槽位（偏斜）尚无定论。

    arXiv:2611.14413v1 Announce Type: new  Abstract: Local language models can save memory by storing weights in fewer bits, and they can reduce sensitivity to the order of answer options by averaging a few reorderings. We ask whether the first makes the second worse, and which is the cheaper way to spend a fixed budget. On Qwen3-0.6B and Qwen3-1.7B with simulated round-to-nearest weight quantization (8, 6, 4 and 3 bits, fp16 reference), read from the logits of the letters A to D on ARC-Easy, ARC-Challenge, MMLU and HellaSwag (300 items each), 6 bits lose no accuracy against fp16 (they score +1.8 (+1.0 to +2.5) and +0.9 (+0.3 to +1.5) points higher, which we do not explain), and 8 bits change it by -0.7 (-1.1 to -0.3) and -0.4 (-0.7 to -0.2) points. At 4 bits the answer-slot skew rises from 0.15 to 0.53 (0.6B) and from 0.07 to 0.12 (1.7B); Gaussian noise at the same weight SNR gives 0.15 and 0.37 (two draws) and 0.19 (one draw), so the control is inconclusive on whether rounding adds slot 
    
[^10]: REAL 损失：为投机解码优化真实期望接受长度

    REAL Loss: Optimizing Real Expected Acceptance Length for Speculative Decoding

    [https://arxiv.org/abs/2611.14399](https://arxiv.org/abs/2611.14399)

    该论文提出 REAL 损失，通过路径权重无偏估计与 Rao-Blackwellization 方差缩减，直接优化投机解码中真实的期望接受长度（EAL），解决了现有替代目标存在的协方差差距及其梯度方向反转问题。

    

    投机解码通过让草稿模型提出多个 token，再由目标模型一次性进行验证，从而加速大语言模型（LLM）的推理。加速效果由期望接受长度（EAL）决定，然而交叉熵、KL 散度和总变差等标准训练目标仅关注分布匹配或单步重叠。近期研究转而提出块级训练目标，旨在直接优化 EAL。然而，这些目标仍然是替代目标，而非 EAL 的精确表述。例如，一些方法在目标分布的续写上对后续接受进行平均，而非在被接受 token 所选择的验证分支上进行平均，由此引入了协方差差距，可能导致梯度方向反转。我们分析了这一差距，并推导出 REAL——一种真实的 EAL 训练损失，它使用路径权重实现无偏估计，并利用 Rao-Blackwellization 进行方差缩减。在受控的损失级比较中

    arXiv:2611.14399v1 Announce Type: cross  Abstract: Speculative decoding accelerates LLM inference by having a draft model propose multiple tokens for the target model to verify in one pass. The speedup is governed by the expected acceptance length (EAL), yet standard training objectives such as cross-entropy, KL divergence, and total variation focus only on distribution matching or single-step overlap. Recent studies instead propose block-level training objectives, aiming to optimize EAL directly. However, these objectives remain surrogates rather than exact formulations of EAL. For example, some average later acceptance over target-distributed continuations rather than the verification branches selected by accepted tokens, introducing a covariance gap that can reverse the gradient direction. We analyze this gap and derive REAL, a real EAL training loss that uses path weights for unbiased estimation and Rao-Blackwellization for variance reduction. In a controlled loss-level comparison 
    
[^11]: 模糊块边界：面向高效扩散语言模型的重叠块去噪

    Blurring the Block Boundaries: Overlapping Block Denoising for Efficient Diffusion Language Models

    [https://arxiv.org/abs/2611.14387](https://arxiv.org/abs/2611.14387)

    本文提出B3方法，通过在共享前向传播中并行精炼两个块来模糊块边界，实现无需训练和超参数调整的扩散语言模型加速，同时保持生成质量。

    

    扩散语言模型（DLMs）已成为自回归模型在延迟敏感任务中的有效替代方案，能够实现更快的token生成。最近的进展采用具有自编辑能力的块扩散方法，其中每个块在进入下一个块之前迭代地精炼token。然而，每个块的速度受限于其最慢的token，且下一个块必须等待它完成后才能开始。我们提出了模糊块边界（B3），这是一种新颖的方法，可以在共享的前向传播中并行精炼两个块。B3能够动态地自我校准，只有当模型准备好开始生成下一个块时才会开启它。这是一种无需训练、无需超参数调整的加速方法，可以无缝集成到任何自编辑块扩散模型中。在最先进的扩散语言模型上、跨越不同任务和难度级别的10个多样化数据集上进行评估，B3在实现同等生成质量的同时获得了11.5-1...（原文摘要截断）

    arXiv:2611.14387v1 Announce Type: cross  Abstract: Diffusion Language Models (DLMs) have emerged as effective alternatives to autoregressive models for latency-critical tasks, enabling faster token generation. Recent advances employ block diffusion with self-editing capabilities, where each block refines tokens iteratively before advancing to the next one. Yet each block is only as fast as its slowest token, and the next block cannot start before it. We suggest Blurring the Block Boundaries (B3), a novel method with which two blocks are refined in parallel within a shared forward pass. B3 calibrates itself on the fly, opening the next block only once the model is ready to start generating it. It is a training-free and hyperparameter-free acceleration method which seamlessly integrates within any self-editing block diffusion model. Evaluated on state-of-the-art DLMs across 10 diverse datasets spanning varied tasks and difficulty levels, B3 results in on par generation quality and 11.5-1
    
[^12]: ARE-bench：在真实研究者会话与工作空间中评测智能体研究能力的基准

    ARE-bench: Benchmarking Agents' Research Ability in Real-World Researcher Sessions and Workspaces

    [https://arxiv.org/abs/2611.14364](https://arxiv.org/abs/2611.14364)

    ARE-bench 基于真实研究者的会话记录与进行中的工作空间构建了 75 个任务，用于评估研究智能体能否在尊重先前决策和现有工作状态的前提下推进真实研究项目。

    

    研究智能体在各类基准测试中取得了越来越高的分数，但这些分数并不一定意味着它们能在进行中的研究项目里可靠地完成任务。许多研究智能体基准强调独立任务以及相对自包含的工作流，且工作空间干净或经过精心整理。然而，研究者实际要求智能体完成的是进行中项目里的某一步骤，这些请求依赖于此前会话中的决策，而且工作空间中往往包含未完成的文件、过时的版本以及失败的尝试。我们提出 ARE-bench，用以评估研究智能体能否在尊重先前决策和现有工作空间状态的前提下推进进行中的项目，其评测根植于真实的会话上下文和进行中的工作空间。ARE-bench 包含 75 个任务，这些任务基于招募来的研究者自己项目中的智能体会话和完整工作空间构建，涵盖七个研究阶段、33 项需求和 150 项子需求，共计 1,173 个二值化检查项。每个任……

    arXiv:2611.14364v1 Announce Type: new  Abstract: Research agents achieve increasingly strong benchmark scores, yet these scores do not necessarily translate into reliable delivery in ongoing research projects. Many research-agent benchmarks emphasize standalone tasks and relatively self-contained workflows in clean or curated workspaces. However, researchers ask agents to complete one step within an ongoing project, where requests depend on earlier session decisions and workspaces contain unfinished files, outdated versions, and failed attempts. We introduce ARE-bench to evaluate whether research agents advance ongoing projects while respecting prior decisions and existing workspace state, grounded in real session context and in-progress workspaces. ARE-bench includes 75 tasks built on agent sessions and full workspaces from recruited researchers' own projects, spanning seven research stages, 33 requirements, and 150 sub-requirements, with 1,173 binary checklist items in total. Each ta
    
[^13]: EngineAgent：具备推理栈层级感知与硬件能力的端侧LLM推理引擎智能体化

    EngineAgent: Agentizing On-Device LLM Inference Engines with Inference Stack Hierarchy Awareness and Hardware Abilities

    [https://arxiv.org/abs/2611.14356](https://arxiv.org/abs/2611.14356)

    EngineAgent通过构建层级感知的引擎索引并利用已支持模型预先验证后端工作流程，将可复用的引擎与硬件知识从模型特定实现中分离，从而自动化端侧LLM推理引擎的新模型适配过程。

    

    快速演进的LLM架构与异构的移动硬件，使得在端侧推理引擎中支持新模型成为一项反复出现且成本高昂的系统挑战。每个模型可能需要在转换、图构建、算子以及硬件后端等多个层面进行协同修改。现有的编程智能体往往忽略这些跨层依赖关系，并且难以诊断主机—设备执行路径上的故障。我们提出了EngineAgent，一个用于自动化端侧LLM模型适配的智能体系统。其核心洞察是将可复用的引擎与硬件知识从模型特定的实现中分离出来。在模型适配之前，EngineAgent会构建一个层级感知的引擎索引，以捕捉引擎抽象层与跨层依赖关系，并利用已支持的模型预先验证后端特定的构建、执行和调试工作流程。对于每个新模型，构建器（builder）利用这些知识来协调跨层变更。

    arXiv:2611.14356v1 Announce Type: cross  Abstract: Rapidly evolving LLM architectures and heterogeneous mobile hardware make supporting new models in on-device inference engines a recurring and costly system challenge. Each model may require coordinated changes across conversion, graph construction, operators, and hardware backends. Existing coding agents often miss these cross-layer dependencies and struggle to diagnose failures across host--device execution paths. We present EngineAgent, an agentic system for automating on-device LLM model enablement. Its key insight is to separate reusable engine and hardware knowledge from model-specific implementation. Before enablement, EngineAgent constructs a hierarchy-aware engine index that captures engine abstraction layers and cross-layer dependencies, and pre-validates backend-specific build, execution, and debugging workflows using already supported models. For each new model, a builder uses this knowledge to coordinate cross-layer change
    
[^14]: AI并没有变笨，是我们变得更挑剔了：大型语言模型人类评估标准的不断上升

    AI Isn't Getting Dumber, We're Getting Pickier: Rising Standards in Human Evaluation of Large Language Models

    [https://arxiv.org/abs/2611.14333](https://arxiv.org/abs/2611.14333)

    该研究通过分析Chatbot Arena上126万余张针对固定版本模型的盲投票数据发现，“两个都不好”的投票比例持续显著上升，表明AI模型并非在变差，而是人类评估者的标准在不断提高。

    

    用户经常抱怨像ChatGPT这样的AI助手正在变差。某些版本确实会变化，但人类评估留下了第二种未经检验的可能性：评判者本身可能发生了变化。我们在Chatbot Arena平台上区分这两种因素——这是一个公开平台，人们在这里比较两个匿名大型语言模型（LLM）的回答，并选择更好的一个或判定两者都不好。我们研究了2023年5月至2024年8月期间针对版本固定模型对投出的1,260,967张盲选票，这些模型包括带有日期标记的厂商快照模型和固定开源权重的模型。对于同一对未发生变化的模型，判定“两个都不好”的投票比例每月上升1.71个百分点（平均值为17.1%），而在2024年每月上升1.91个百分点。在2023年和2024年均被频繁观察到的十四对模型中，这一比例在每一对中都出现了上升，从15.5%升至23.3%。模型排名并未呈现趋势：2024年，更强模型在决定性投票中的份额保持平稳。这种上升在同一投票者内部同样成立（原文在此处被截断）。

    arXiv:2611.14333v1 Announce Type: cross  Abstract: Users often complain that AI assistants such as ChatGPT are getting worse. Some versions do change, but human evaluation leaves a second possibility untested: the people judging may change. We separate the two on Chatbot Arena, a public platform where people compare answers from two anonymous large language models (LLMs) and pick the better one or say both are bad. We study 1,260,967 blind votes cast from May 2023 to August 2024 on pairs of models whose versions are fixed: dated vendor snapshots and models with fixed open weights. For the same pair of unchanged models, the share of votes saying "both are bad" rises by 1.71 percentage points a month on a mean of 17.1%, and by 1.91 points a month in 2024. In fourteen pairs observed often in both 2023 and 2024, it rose in every one, from 15.5% to 23.3%. The ranking does not trend: in 2024, the stronger model's share of decisive votes is flat. The rise holds within the same voter and on th
    
[^15]: 财报电话会议的时点编码器

    Point-in-Time Encoders for Earnings Calls

    [https://arxiv.org/abs/2611.14325](https://arxiv.org/abs/2611.14325)

    该论文提出按年度截止日期训练的“时点”文本编码器，以消除财报电话会议检索回测中的前视偏差，其小型专用模型在答案检索上全面优于通用检索器，并将收益预测的秩相关性提升约一半。

    

    文本编码器将财报电话会议转换为向量，用于搜索、用于基于检索段落进行回答的语言模型，以及用于预测模型。实时检索器可以使用在所有可用文本上一次性训练的编码器（正如金融检索器通常的做法），但回测需要一个没有读过每个向量生成日期之后所写文本的编码器。为了解决这种前视偏差，我们从412,248个电话会议中，按照四种预训练方案训练了多个“版本”，截止日期从2010年到2024年逐年划分，每个版本只读取其截止日期当天或之前的文本。随后，我们在电话会议本身的问答对上对每个版本进行微调。仅用电话会议文本训练的版本（参数量3380万）在所有测试季度中，对留出电话会议的答案检索效果都优于我们测试的通用检索器。这些向量还将次日收益与基于过去收益和公司规模构建的预测之间的秩相关系数提高了约一半。当每个版本继续……（原文在此截断）

    arXiv:2611.14325v1 Announce Type: new  Abstract: Text encoders turn earnings calls into vectors for search, for language models that answer from retrieved passages, and for forecasting models. A live retriever can use an encoder trained once on all available text, as finance retrievers are, but a backtest needs an encoder that read no text written after the date of each vector. To address this look-ahead bias, from 412,248 calls we train vintages under four pretraining recipes, with yearly cutoffs from 2010 to 2024, and each vintage reads only text dated on or before its cutoff. We then fine-tune each vintage on the question-answer pairs of the calls themselves. The vintages we train on calls alone, with 33.8M parameters, retrieve answers on held-out calls better than the general retrievers we tested in every test quarter. Their vectors also raise by about half the rank correlation between next-day returns and forecasts built from past returns and firm size. When each vintage continues
    
[^16]: MMDM：衡量大规模多任务决策

    MMDM: Measuring Massive Multitask Decision Making

    [https://arxiv.org/abs/2611.14309](https://arxiv.org/abs/2611.14309)

    提出了MMDM基准——涵盖16个任务类别、5,448道高质量问题的大规模多任务决策基准，用于在无数据污染的全新上下文中评估系统-1决策模型在单选、多选和软标签决策上的能力。

    

    系统-1决策模型在单次前向传播中，基于给定状态直接对查询所指定的选项做出结构化的概率决策，而无需自回归地生成推理轨迹或文本响应。近期的评估虽涵盖多样化任务，但往往依赖现有基准，而对测试材料的事先接触（数据污染）可能掩盖模型的真实泛化能力。此外，现实世界的决策常常涉及长上下文、庞大的选项空间以及复杂的逻辑依赖，而这些挑战并未被现有基准同时充分捕捉。因此，我们提出了MMDM（大规模多任务决策），这是一个包含16个任务类别、共5,448道高质量问题的基准，它将新构建的任务与基于原始来源的重组和改写相结合，用于在全新上下文中进行单选、多选和软标签决策。该基准涵盖了诸如智能体状态跟踪、工具与技能选择等真实场景（摘要原文在此处被截断）。

    arXiv:2611.14309v1 Announce Type: cross  Abstract: System-1 Decision Models directly make structured probabilistic decisions over query-specified options given a state in a single forward pass, without autoregressively generating reasoning trajectories or textual responses. Recent evaluations cover diverse tasks but often rely on existing benchmarks, where prior exposure to test material can obscure generalization. Moreover, real-world decision making often involves long contexts, large option spaces, and complex logical dependencies, challenges that existing benchmarks do not adequately capture together. Thus, we introduce MMDM (Massive Multitask Decision Making), a benchmark of 5,448 high-quality questions across 16 task categories, combining newly constructed tasks with source-grounded recompositions and rewrites for single-select, multi-select, and soft-label decision making in fresh contexts. It covers realistic scenarios such as agent state tracking, tool and skill selection, pre
    
[^17]: 语义链：从可及事实到有据之答案

    Semantic Chains: From Reachable Facts to Justified Answers

    [https://arxiv.org/abs/2611.14252](https://arxiv.org/abs/2611.14252)

    本文将上下文建模为带保持边的逻辑事实图，证明了从问题中的对象出发的搜索可以到达证明答案的根事实，从而构建出连接问题与有据答案的“语义链”，并对多环节链条给出候选顺序的完整枚举。

    

    如何从问题出发找到能够证明某个答案的事实，并将其呈现为一系列相互关联的陈述？本文将上下文建模为一个由逻辑事实构成的图，并把“哪些表达式代表哪些事实”作为给定条件。一个父事实可以表达其若干子事实之间的关系，而一条保持边允许在父事实的完整表示中恢复子事实的内容。假设某个根事实能够证明完整答案，且存在一条由保持边构成的路径，从该根事实通往一个指称问题与答案所共享对象的事实。在模型给定的条件下，从问题中的对象出发的搜索能够到达该根事实，并覆盖所选展开中的每一个事实。于是该根事实可以对应一段文本，构成连接问题与完整答案的单一链条环节，并证明该答案。对于包含多个环节的链条，本文在所述范围内给出了候选顺序的完整枚举。（摘要原文在此处截断）

    arXiv:2611.14252v1 Announce Type: new  Abstract: How can the facts that justify an answer be found from a question and presented as a sequence of connected statements? This paper models a context as a graph of logical facts and takes as given which expressions represent each fact. A parent fact can express a relation among its subfacts, and a preserving edge allows the content of a subfact to be recovered within a full representation of its parent. Suppose a root fact justifies the full answer and a path of preserving edges leads from that root to a fact that designates an object shared by the question and the answer. Under the conditions of the model, a search starting from the objects in the question reaches the root and covers every fact in the selected expansion. The root then admits a text that forms a single link connecting the question to the full answer and justifying it. For chains with several links, the paper gives a complete enumeration of candidate orders within the stated
    
[^18]: 基于TRACE框架的法案反腐败专家审查自动化AI助手

    An AI Assistant for Automating Anti-Corruption Expertise of Draft Legislation Based on the TRACE Framework

    [https://arxiv.org/abs/2611.14244](https://arxiv.org/abs/2611.14244)

    该论文基于TRACE架构提出了一种自动化反腐败专家审查的AI助手，能够筛查法案草案中的腐败风险并精确定位专家会批评的问题条款，同时利用乌克兰国家反腐败预防署的公开结论构建了首个开放评估基准。

    

    我们提出了一个基于TRACE架构的AI助手，用于自动化法案草案的反腐败专家审查，旨在筛查法案中的腐败风险：精确定位专家会批评的条款，并指出每一条款中存在的监管缺陷。该处理流程结合了确定性解析核心、模型引导的单一条款定性、基于规则的风险因子分配以及外部人工监督。为了评估该助手，我们根据乌克兰国家反腐败预防署（NAZK）公布的审查结论构建了一个开放基准，包含132项法案草案和474个模型提取的发现。当系统标记指向兼容的条款，且由LLM评判者将风险评级为相同（严格标准）或部分相同（宽松标准）时，即认为该系统标记与发现一致。在100项法案草案上，使用本地Gemma 4 31B模型时，该流程在其前十个标记中匹配了0.28的发现，使用Sonnet 5.5时为0.30，而在……（摘要在此处截断）

    arXiv:2611.14244v1 Announce Type: cross  Abstract: We present an AI assistant for automating anti-corruption expertise of draft legislation based on the TRACE architecture, designed to screen bills for corruption risks: pinpoint provisions an expert would criticise and name the regulatory defect in each. The pipeline combines a deterministic parsing core, model-guided single-provision qualification, rule-based factor assignment, and external human oversight. To evaluate the assistant, we constructed an open benchmark from the published conclusions of Ukraine's National Agency on Corruption Prevention (NAZK), comprising 132 draft laws with 474 model-extracted findings. A system flag agrees with a finding when it targets a compatible provision and an LLM judge rates the risk as the same (strict) or partly the same (lenient). On 100 draft laws, the pipeline matches 0.28 of the findings within its top ten flags using the local Gemma 4 31B model, 0.30 with Sonnet 5.5, and 0.43-0.46 across a
    
[^19]: Dyship：利用声学特征从双人对话中进行说话人关系分类

    Dyship: Leveraging Acoustic Features For Speaker Relationship Classification From Dyadic Conversations

    [https://arxiv.org/abs/2611.14241](https://arxiv.org/abs/2611.14241)

    该论文提出DyShip框架，通过比较频谱图、MFCC、WavLM、x向量和发音运动五种声学表示，从双人对话中实现说话人熟悉度和亲缘关系的自动分类，且多数分类器表现超越人类和音频语言模型基准。

    

    我们提出了DyShip，这是一个用于从Seamless Interaction语料库的双人对话中对说话人熟悉度和亲缘关系进行分类的框架。我们比较了五种语音表示方式：频谱图、梅尔频率倒谱系数（MFCCs）、自监督WavLM表示、x向量以及估计的发音运动。频谱特征在熟悉度分类上表现最佳，准确率达到87.81%，而WavLM在亲缘关系分类上取得了最高的85.17%准确率。x向量和发音特征在熟悉度分类上结果较弱，但在亲缘关系分类上表现出色，其中x向量的AUC达到0.909。大多数分类器在同一数据集上的表现超过了此前报告的人类和音频语言模型的分数。这些发现凸显了传统声学特征在关系推断方面的潜力，并激励人们进一步研究不同表示方式所捕获的对话线索。

    arXiv:2611.14241v1 Announce Type: cross  Abstract: We introduce DyShip, a framework for classifying speaker familiarity and kinship from dyadic conversations in the Seamless Interaction corpus. We compare five speech representations: spectrograms, MFCCs, self-supervised WavLM representations, x-vectors, and estimated articulatory movements. Spectral features perform best for familiarity classification, reaching an accuracy of 87.81%, while WavLM achieves the highest kinship accuracy of 85.17%. X-vectors and articulatory features yield weaker results on familiarity classifiaction, but shine for kinship classification, with an AUC of 0.909 for x-vectors. Most classifiers exceed previously reported human and audio language model scores on the same dataset. These findings highlight the potential of conventional acoustic features for relationship inference and motivate further investigation of the conversational cues captured by different representations.
    
[^20]: StaRSE：采用符号编码静态编码器的紧凑俄语句子嵌入

    StaRSE: Compact Russian Sentence Embeddings with a Sign-Coded Static Encoder

    [https://arxiv.org/abs/2611.14236](https://arxiv.org/abs/2611.14236)

    StaRSE是一种紧凑的俄语静态句子编码器，通过将词元方向以符号位存储并去除Transformer推理，以仅11.3 MiB的模型大小和0.08毫秒的CPU延迟，在RuMTEB上实现了比最强静态基线高7.98分的检索性能。

    

    Transformer句子编码器提供了强大的检索质量，但当需要在CPU上生成嵌入或需要频繁刷新嵌入时，其推理成本成为瓶颈。我们提出了StaRSE，这是一个紧凑的俄语静态句子编码器，它在部署阶段完全去除了Transformer推理：词元方向以符号位形式存储，词元范数单独存储，句子则通过对范数缩放后的符号方向进行平均池化来编码。StaRSE从ruBERT-base初始化，在俄语相邻句子对和任务形态语对上进行微调，并通过对比学习阶段进行后训练，该阶段直接优化符号投影的词元表。在RuMTEB基准上，StaRSE达到了51.16的任务平均值，比所包含的最强静态基线高出7.98分。打包后的模型仅占用11.3 MiB空间，每个句子的平均CPU延迟为0.08毫秒，而相比之下，16位参数的模型估计占用55.7 MiB空间，平均CPU延迟为1.43毫秒。

    arXiv:2611.14236v1 Announce Type: new  Abstract: Transformer sentence encoders provide strong retrieval quality, but their inference cost becomes a bottleneck when embeddings must be produced on CPUs or refreshed frequently. We introduce StaRSE, a compact Russian static sentence encoder that removes Transformer inference at deployment time: token directions are stored as sign bits, token norms are stored separately, and sentences are encoded by mean pooling over norm-scaled sign directions. StaRSE is initialized from ruBERT-base, fine-tuned on Russian adjacent-sentence and task-shaped pairs, and post-trained with a contrastive stage that directly optimizes the sign-projected token table. On RuMTEB, StaRSE reaches a task mean of $51.16$, $7.98$ points above the strongest included static baseline. The packaged model occupies $11.3$ MiB and has a mean CPU latency of $0.08$ ms per sentence, compared with an estimated $55.7$ MiB 16-bit parameter footprint and $1.43$ ms mean CPU latency for 
    
[^21]: 基于Transformer的语言模型中内外计算的分阶段可解释涌现

    Stagewise and interpretable emergence of inside-outside computation in Transformer-based language models

    [https://arxiv.org/abs/2611.14221](https://arxiv.org/abs/2611.14221)

    本文首次通过研究具有可变句法树结构的合成上下文无关文法中的学习动态，展示了Transformer语言模型如何分阶段、可解释地涌现出inside-outside句法分析计算，其中每个损失阶段均可被解析计算并与潜在树变量表示的涌现明确关联。

    

    句法结构的变异性和句法分析中的歧义是自然语言的基本特征，现代大语言模型必须学会熟练地处理这些问题。以往的“事后”研究表明，训练完成的Transformer确实能够近似诸如inside-outside（内外算法）之类的经典句法分析算法，但对于这种能力实际上是如何被学习到的，理论上仍知之甚少。本文朝解决这一问题迈出了第一步：我们研究了在具有可变句法分析树拓扑结构的合成上下文无关文法中的学习动态。这些阶段性的学习曲线是完全可解释的，具体而言，损失函数的所有阶段a) 都可以被解析地计算出来，b) 都可以清晰地与一个隐藏的潜在树变量的表示相关联，该变量能够消除歧义并利用更广泛上下文中的信息。通过对模型表示进行探测，我们可视化了inside-outside计算在网络深度方向上以及训练过程中如何涌现。

    arXiv:2611.14221v1 Announce Type: new  Abstract: Variability in syntactic structure and parsing ambiguity are fundamental aspects of natural language which modern LLMs have to learn to deal with proficiently. Previous "post-mortem" studies have shown that fully trained Transformers indeed approximate canonical parsing algorithms such as inside-outside, but little is established theoretically about how this is actually learned. Here, we make a first step towards addressing this by studying the learning dynamics in synthetic context-free grammars with variable parse tree topology. The stagewise learning curves are fully interpretable, in the sense that all of the loss stages a) can be computed analytically, and b) can be clearly associated to the representation of a hidden latent tree variable, which allows to lift ambiguity and utilize information from wider contexts. By probing representations, we visualize the emergence of inside-outside computation across depth and during training: w
    
[^22]: 集中式与分散式注意力头失活下自回归多步推理的崩溃

    Breakdown of Autoregressive Multi-Step Reasoning Under Concentrated vs. Diffuse Attention Head Dropout

    [https://arxiv.org/abs/2611.14209](https://arxiv.org/abs/2611.14209)

    受哺乳动物大脑“关闭状态”启发，该研究发现对大语言模型注意力头施加集中式爆发失活，比同等规模的分散式失活会造成严重得多的多步数学推理性能下降。

    

    在持续清醒期间，哺乳动物的新皮层网络会经历局部、短暂的神经元沉默期，被称为“关闭状态”（off-states）。本文研究了类似的时空集中式扰动如何影响自回归大语言模型（LLM）的序列推理能力。我们在GSM8K基准上对两个8B模型（Qwen3-8B与LLaMA-3.1-8B-Instruct）进行了250组配对推导评估，将聚簇爆发的注意力头掩码与预算匹配的分散式独立同分布失活进行对比。结果表明，集中式爆发会导致显著的推理性能退化：在起始参数 $\mathcal{B}=0.10$（调度约5.7%的易感头-词元容量）下，Qwen3-8B的准确率从分散噪声条件下的91.2%降至爆发式失活条件下的78.8%（Δ = −12.4个百分点，精确McNemar检验 p = 5.5×10⁻⁶），而LLaMA-3.1-8B-Instruct则从84.0%降至76.8%（Δ = −7.2个百分点，p = 0.0198）。相反，分散式……（原文摘要在此处截断）

    arXiv:2611.14209v1 Announce Type: cross  Abstract: During extended wakefulness, mammalian neocortical networks experience localized periods of transient neuronal silence termed off-states. Here, we examine how analogous spatiotemporally concentrated disruptions affect sequential reasoning in autoregressive Large Language Models (LLMs). Evaluating 250 paired derivations on the GSM8K benchmark across two 8B models (Qwen3-8B and LLaMA-3.1-8B-Instruct), we contrast clustered bursts of attention head masking against budget-matched diffuse i.i.d. dropout. Concentrated bursts induce substantial reasoning degradation: at onset parameter $\mathcal{B}=0.10$ (scheduling $\sim 5.7\%$ of vulnerable head-token volume), Qwen3-8B accuracy drops from $91.2\%$ under diffuse noise to $78.8\%$ under burst dropout ($\Delta = -12.4$ percentage points [pp], exact McNemar $p = 5.5 \times 10^{-6}$), while LLaMA-3.1-8B-Instruct drops from $84.0\%$ to $76.8\%$ ($\Delta = -7.2$ pp, $p = 0.0198$). Conversely, disp
    
[^23]: 神经网络中的表征原子

    Representational Atoms in Neural Networks

    [https://arxiv.org/abs/2611.14168](https://arxiv.org/abs/2611.14168)

    该论文定义了神经网络中的基本表征单元（FRUs）并建立保真度与稳定性两项评估标准，证明阈值激活稀疏自编码器（TSAEs）在适当条件下可高保真、稳定地识别FRUs，且高稳定性的FRUs具有更强的单语义性并能跨层组合以恢复与人类概念结构相对应的语义组织。

    

    理解神经网络的内部表征并刻画其与人类概念结构之间的关系，是阐明模型机制的两个关键问题。前者揭示了所学知识如何在内部表征中被组织起来，而后者为借助人类概念结构来解释这些表征提供了基础。在此，我们正式定义了基本表征单元（FRUs），并建立了两个评估标准：保真度与稳定性。我们进一步证明，阈值激活稀疏自编码器（TSAEs）在适当条件下能够以高保真度和高稳定性识别FRUs，并通过大规模实验验证了这一结果。进一步的分析表明，具有更高稳定性的FRUs表现出更强的单语义性，并且能够跨层组合，从而恢复与人类概念结构相对应的语义组织。

    arXiv:2611.14168v1 Announce Type: cross  Abstract: Understanding the internal representations of neural networks and characterizing their relationship with human conceptual structure are two critical questions for elucidating model mechanisms. The former reveals how learned knowledge is organized in internal representations, whereas the latter provides a basis for interpreting these representations through human conceptual structure. Here, we formally define fundamental representational units (FRUs) and establish two evaluation criteria: fidelity and stability. We further prove that threshold-activated sparse autoencoders (TSAEs) can identify FRUs with high fidelity and stability under appropriate conditions, and validate this result through large-scale experiments. Further analyses reveal that FRUs with higher stability exhibit greater monosemanticity and can be composed across layers to recover semantic organization corresponding to human conceptual structure. These results provide a
    
[^24]: 成人词汇扩展与人造词汇学习中的视觉相似性

    Visual Similarity in Adult Word Extension and Artificial Word Learning

    [https://arxiv.org/abs/2611.14116](https://arxiv.org/abs/2611.14116)

    本研究通过重新分析大规模成人词汇扩展数据并进行人工词汇学习实验，利用冻结的自监督视觉编码器（DINOv2）量化物体间的视觉相似性，发现视觉距离显著影响词汇的扩展范围，且教学示例的类别层级高低决定了学习者在多大程度上将词汇泛化到更广泛或更具体的类别。

    

    当两个物体共享同一个名称时，它们的视觉相似性能在多大程度上解释这一现象，教学示例又能带来什么额外作用？首先，我们重新分析了一项大型成人词汇扩展研究的试验级数据（Lewis & Frank, 2018），在冻结的自监督视觉编码器（DINOv2）的空间中计算每个测试图像与所展示示例之间的距离。距离与词汇扩展显著相关（每标准差对数几率为-3.23；共520名参与者），一项探索性事后分析表明，在控制了分类层级和教学条件后，这种关联在已教授的领域内依然存在。示例的组成同样重要：在学习三个示例后，当示例为下位类别时，成人仅将词汇扩展到上位类别项目的1.1%，而当示例为上位类别时，这一比例达到77.5%。其次，在使用同一编码器的人工学习者实验中，当一个词连接两个类别时，类别之间距离越远，学习准确度越低（84.7%……

    arXiv:2611.14116v1 Announce Type: cross  Abstract: When two objects share a name, how much does their visual similarity explain, and what do the teaching examples add? First, we reanalyzed trial-level data from a large adult word-extension study (Lewis & Frank, 2018), computing each test image's distance to the examples shown in the space of a frozen self-supervised visual encoder (DINOv2). Distance was associated with extension (log odds -3.23 per standard deviation; 520 people), and an exploratory post hoc analysis suggested that this association persisted within the taught domain after controlling for taxonomic level and teaching condition. Example composition also mattered: after three examples, adults extended a word to 1.1% of superordinate items when the examples were subordinate and to 77.5% when they were superordinate. Second, in an artificial learner using the same encoder, words joining two categories were learned less accurately the farther apart the categories were (84.7%
    
[^25]: Sckwoky团队参加SHROOM-Visions竞赛：基于双提示LoRA解码器的图像条件视觉语言模型字符级幻觉片段检测

    Sckwoky at SHROOM-Visions: Character-Level Hallucination-Span Detection in Image-Conditioned VLMs via Dual-Prompted LoRA Decoders

    [https://arxiv.org/abs/2611.14100](https://arxiv.org/abs/2611.14100)

    该论文提出了一套基于冻结Gemma-4-12B-it骨干网络与LoRA适配器的字符级幻觉片段检测系统，通过在单一提示中重复候选答案来获取近似双向上下文，可在四种语言上同时预测视觉语言模型响应中的幻觉概率、边界与类型。

    

    我们提出了参加SHROOM-visions 2026共享任务的系统，该任务针对图像条件视觉语言模型（VLM）响应中的字符级幻觉片段检测进行评估，涵盖英语、法语、意大利语和中文四种语言。检测器建立在冻结的Gemma-4-12B-it骨干网络之上，配备可训练的LoRA适配器（约1070万参数）。由于该骨干网络是因果解码器，从单个答案副本读取的隐藏状态无法注意到后续标记；因此我们在同一个提示中重复候选答案，并从其第二次出现的位置提取特征，从而在单次非自回归传递中为每个位置提供近似双向的上下文。字符级定位器结合中间隐藏状态来预测幻觉概率、BIO边界以及响应级弃权分数，同时独立的语言特定分类器为每个预测出的片段分配五种幻觉类型之一。

    arXiv:2611.14100v1 Announce Type: new  Abstract: We present our system for the SHROOM-visions 2026 shared task on character-level hallucination span detection in image-conditioned vision-language model (VLM) responses, evaluated across English, French, Italian, and Chinese. The detector is built on a frozen Gemma-4-12B-it backbone with trainable LoRA adapters (approximately 10.7M parameters). Because the backbone is a causal decoder, hidden states read from a single copy of an answer cannot attend to subsequent tokens; we therefore repeat the candidate answer within one prompt and extract features from its second occurrence, giving every position approximate bidirectional context in a single non-autoregressive pass. A character-level locator combines intermediate hidden states to predict hallucination probabilities, BIO boundaries, and a response-level abstention score, while separate language-specific classifiers assign one of five hallucination types to each predicted span. On the cl
    
[^26]: 从偏见到处境性：大语言模型的批判性语法

    From Bias to Positionality: A critical grammar for Large Language Models

    [https://arxiv.org/abs/2611.14079](https://arxiv.org/abs/2611.14079)

    该论文提出用“处境性”取代“偏见”作为理解大语言模型的批判性框架，因为偏见需要外部标准作为参照，而在语言模型的一般使用中不存在这样的标准，因此更应追问模型“从何处说话”。

    

    偏见已成为理解机器学习系统行为以及我们应如何与之相处的一种默认视角。偏见本质上是相对于某一标准的偏离；在决策系统中，我们可以从外部提供标准：法律、受保护属性、基准测试、统计特性。然而，即便在这些场景中，“有偏见”往往只是一种简略说法，并未说明所采取的立场是什么——一个沉默的“相对于”。在语言模型的一般使用中，没有任何东西提供标准：没有可供审计的决策，也没有反事实的输出，因此这种简略说法失效了。我们认为，处境性——即说出任何话语时所占据的、被清晰表述的、被承认的且可被质疑的位置——才是回答这些模型如今迫使我们面对的问题的更好语法：谁在说话？从何处说话？针对厂商“无处”的回答以及“处处”这一诱人的可能性，我们认为，当模型被视为“仿佛在说话”时，它就是从某个地方说话的……

    arXiv:2611.14079v1 Announce Type: cross  Abstract: Bias has become a default lens for understanding what machine learning systems do and how we should relate to them. Bias is at root a deviation from a standard; in decision making systems we can supply a standard from outside: a law, a protected attribute, a benchmark, a statistical property. However, even there, biased is often used as a shorthand, without stating what the position taken is - a silent ``relative to''. In a general use of language models, nothing supplies the standard: there is no decision to audit and no counterfactual output, so the shorthand fails. We argue that positionality -- the located, articulated, owned and contestable position from which anything is said -- is the better grammar for the question these models now force: who is speaking, from where? Against the vendors' answer of nowhere and the seductive possibility of everywhere, we argue that when a model is taken as-if speaking, it speaks from somewhere, a
    
[^27]: HandsOnAid：构建并评估面向应急响应场景的离线急救助手

    HandsOnAid: Building and Evaluating an Offline First-Aid Assistant for Emergency Response Scenarios

    [https://arxiv.org/abs/2611.14059](https://arxiv.org/abs/2611.14059)

    本文构建了HandsOnAid——一个可在无信号、无救援环境中通过QLoRA微调的4位Gemma 2B模型实现的离线手机急救助手，并通过四个前沿大模型盲评验证了微调显著提升了急救建议的质量与安全性。

    

    紧急情况可能发生在没有手机信号且附近没有救护车的地方。在这种情况下，手机上的小型语言模型可能会有所帮助，但其建议必须安全且准确。我们构建了HandsOnAid，这是一个4位量化的Gemma 2B模型，通过QLoRA在FirstAidQA数据集（包含5,550个急救问答对的语料库）上进行了微调。我们比较了七种配置，包括基础模型、4位和8位微调模型，以及带有长度下限、安全门控或检索功能的版本。我们使用了一个包含41对问答的题库，其医学审核参考遵循ANZCOR指南，并采用严格的0-5分评分标准，包含十二项安全否决机制。四个前沿大语言模型评审（GPT-5.6 Sol、Claude Opus 5、DeepSeek V4 Pro、GLM-5.3）对所有答案进行了盲评，并使用45个隐藏对照组来检验评审的可靠性。微调使平均分数比基础模型提高了1.08，达到2.34。最佳配置得分为5分中的2.51分。

    arXiv:2611.14059v1 Announce Type: cross  Abstract: Emergencies can occur where there is no phone signal and no ambulance nearby. A small language model on a phone can be helpful in such situations, but its advice needs to be safe and accurate. We built HandsOnAid, a 4-bit Gemma 2B model fine-tuned with QLoRA on FirstAidQA dataset, a corpus of 5,550 first-aid question-answer pairs. We compared seven configurations, including the base, 4-bit and 8-bit fine-tuned models, and versions with a length floor, safety gate, or retrieval. We used a 41-pair question-answer bank with medically reviewed references that followed ANZCOR guidelines, and a strict 0-5 score rubric, with twelve safety overrides. Four frontier LLM judges (GPT-5.6 Sol, Claude Opus 5, DeepSeek V4 Pro, GLM-5.3) blindly scored every answer, with 45 hidden controls used to check judge reliability. Fine-tuning improved the mean score by +1.08 over the base model to 2.34. The best configuration scored 2.51 out of 5 which was the 
    
[^28]: 词汇的热力学

    Thermodynamics of words

    [https://arxiv.org/abs/2611.14039](https://arxiv.org/abs/2611.14039)

    该论文提出词汇热化假说，将文学作品中的词频分布类比为经典振荡器的瑞利-金斯热化与凝聚现象，并通过对《圣经》、莎士比亚全集等16部作品的分析，验证了低有效能量条件下会出现词汇的RJ凝聚。

    

    我们提出了词汇热化假说，该假说认为在文学作品中，词汇之间存在着有效的相互作用，这些相互作用驱使词汇达到热化状态，此时其词频分布与瑞利-金斯（RJ）热化及凝聚现象的分布相同。在这一概念框架下，不同的单词扮演着经典振荡器（粒子）的角色，而它们的词频或相关的词元则类似于能级状态上的某种填充因子。该热化过程的特征在于系统总能量（即单词/词元总数）以及总概率范数（即不同单词的总数）守恒。我们将这一方法应用于分析16部文学作品，包括《圣经》、莎士比亚全集以及美国国家语料库的数据。我们证明，对于这些情况，系统的有效能量非常低，因此出现了RJ凝聚现象，并伴随

    arXiv:2611.14039v1 Announce Type: cross  Abstract: We introduce the word thermalization hypothesis according to which in a literary work there are effective interactions between words that drive them to thermalization where their word frequency distribution is the same as for the phenomenon of Rayleigh-Jeans (RJ) thermalization and condensation. In this concept, the different words play a role of classical oscillators (particles) and their word frequencies, or related tokens, are analogous to certain filling factors over energy state levels. The thermalization process is characterized by the conservation of total system energy, or total number of words/tokens, and total probability norm, or total number of different words. We apply this approach to analyze 16 literary works, including Bible, complete works of Shakespeare, and data from American National Corpus. We show that for these cases the effective system energy is very low and thus there is the RJ condensation phenomenon with a b
    
[^29]: MISO-BR：一个用于巴西葡萄牙语厌女症检测的开放语料库

    MISO-BR: An Open Corpus for Misogyny Detection in Brazilian Portuguese

    [https://arxiv.org/abs/2611.14036](https://arxiv.org/abs/2611.14036)

    该论文推出了MISO-BR——首个面向巴西葡萄牙语厌女症检测的开放、有文档记录且版本化的标注语料库，附带完整的构建协议、标注指南和可复现的基线模型，填补了该领域开放资源的空白。

    

    厌女症是针对巴西用户最普遍的网络仇恨言论形式之一，然而专门针对巴西葡萄牙语的开放且可审计的相关资源仍然稀缺。我们推出了MISO-BR，这是一个开放的、有完整文档记录且版本化的语料库，包含896条平衡分布的社交媒体文本，标注了二元厌女症标签，并随附其构建协议和可复现的基线。该语料库结合了从X平台进行的基于词典和目标的定向抓取、从OLID-BR导入的项目、一个大语言模型筛选阶段（其提示词和解码参数均已公开），以及三名标注者按照书面指南进行的人工标注。在816个项目上，标注者间信度达到Fleiss' kappa 0.788。在单一固定协议下，我们比较了四种经典分类器（多项式朴素贝叶斯、伯努利朴素贝叶斯、线性SVM和随机森林）与微调后的BERTimbau Base和Large模型。经典模型的宏平均F1分数聚集在0.651至0.676之间。

    arXiv:2611.14036v1 Announce Type: cross  Abstract: Misogyny is among the most prevalent forms of online hate speech directed at Brazilian users, yet open and auditable resources dedicated to it in Brazilian Portuguese remain scarce. We introduce MISO-BR, an open, documented and versioned corpus of 896 balanced social media texts annotated for binary misogyny, released with its construction protocol and reproducible baselines. The corpus combines lexicon- and target-guided scraping from X with items imported from OLID-BR, a large language model triage stage whose prompt and decoding parameters are released, and manual annotation by three annotators following a written guide. Inter-annotator reliability reaches a Fleiss' kappa of 0.788 over 816 items. Under a single fixed protocol we compare four classical classifiers (Multinomial and Bernoulli Naive Bayes, a linear SVM and a random forest) with fine-tuned BERTimbau Base and Large. The classical models cluster between 0.651 and 0.676 mac
    
[^30]: 程序辅助的可补丁技能演化

    Program-Aided Patchable Skill Evolution

    [https://arxiv.org/abs/2611.14019](https://arxiv.org/abs/2611.14019)

    该论文提出PASE框架，通过将智能体技能中的可复用程序性逻辑分离为可编辑的可执行程序（而非自然语言文档），实现技能的可维护性、可补丁化与针对性迭代演化。

    

    智能体技能已成为一种有效手段，通过编码可复用知识而无需修改模型参数，使LLM智能体能够解决复杂的交互式任务。最近的方法通过从智能体执行轨迹中提炼可复用经验，并基于下游任务反馈来演化技能，进一步提升了技能适应能力。然而，现有方法主要以自然语言文档的形式表示技能，将程序性逻辑与描述性知识交织在一起，使得技能难以维护和演化。为解决这一局限，我们提出了PASE（程序辅助的可补丁技能演化），一个通过将技能与可执行程序相关联来迭代演化技能的程序辅助框架。与先前主要用自然语言编码轨迹经验的方法不同，PASE将可复用的程序性逻辑分离为可编辑的程序，从而能够对可执行组件进行针对性更新……（摘要原文在此处截断）

    arXiv:2611.14019v1 Announce Type: cross  Abstract: Agent skills have emerged as an effective means of enabling LLM agents to solve complex, interactive tasks by encoding reusable knowledge without modifying model parameters. Recent approaches further improve skill adaptation by distilling reusable experience from agent execution trajectories and evolving skills based on downstream task feedback. However, existing methods primarily represent skills as natural language documents, intertwining procedural logic with descriptive knowledge and making skills difficult to maintain and evolve. To address this limitation, we propose PASE (Program-Aided Patchable Skill Evolution), a program-aided framework that iteratively evolves skills by associating them with executable programs. Unlike prior approaches that encode trajectory-derived experience primarily in natural language, PASE separates reusable procedural logic into editable programs, enabling targeted updates to executable components whil
    
[^31]: 一个序列就足够了：从定长整数序列对语言模型进行指纹识别

    One Sequence Is Enough: Fingerprinting Language Models from Fixed-Length Integer Sequences

    [https://arxiv.org/abs/2611.14010](https://arxiv.org/abs/2611.14010)

    该论文提出NumPrint基准和SMCT模型，证明仅凭模型生成的一条定长整数序列，就能以92.72%的准确率在51个语言模型中识别出响应请求的真实模型。

    

    模型API和第三方平台的用户通常依赖所声明的模型名称来识别响应其请求的模型。然而，实际模型可能与声明的模型不符，这促使人们需要基于可观测输出的身份验证方法。先前的研究表明，简单数字选择任务中的偏好可以作为模型指纹。我们研究单个定长整数序列能否结合多个数值观测来识别其来源模型。为此，我们提出了NumPrint基准，其中包含来自51个模型的5,036个答案，这些答案是使用中文、日文和英文提示收集的，提示要求模型在一次响应中给出294个1到355之间的整数。我们提出了浅层MLP与因果Transformer（SMCT）来捕捉生成序列中的整体数值偏好和行为模式。在51类闭集评估中，SMCT仅凭单个序列即可达到92.72%的准确率，优于现有基线方法。

    arXiv:2611.14010v1 Announce Type: new  Abstract: Users of model APIs and third-party platforms often rely on the declared model name to identify the model responding to their requests. However, the actual model may differ from the declared one, motivating identity verification based on observable outputs. Previous studies have shown that preferences in simple numerical-choice tasks can serve as model fingerprints. We investigate whether a single fixed-length integer sequence can combine multiple numerical observations to identify its source model. For this task, we introduce NumPrint, a benchmark of 5,036 answers from 51 models, collected using Chinese, Japanese, and English prompts that request 294 integers between 1 and 355 in one response. We propose the Shallow MLP and Causal Transformer (SMCT) to capture aggregate numerical preferences and behavioral patterns in generated sequences. In 51-class closed-set evaluation, SMCT achieves 92.72% accuracy from a single sequence, outperform
    
[^32]: 编程智能体记忆后训练 II：通过在线策略蒸馏释放预训练文件操作在长时程任务中的记忆潜力

    Coding Agent Memory Post-Training II: Unlocking the Memory Potential of Pre-trained File Operations for Long-Horizon Tasks via On-Policy Distillation

    [https://arxiv.org/abs/2611.14009](https://arxiv.org/abs/2611.14009)

    该论文提出 CAMG-II 长时程智能体强化学习环境套件，通过提供 Bash 访问和回合持久工作区，让记忆控制建立在预训练已有的文件操作能力之上，从而解决在线策略蒸馏中因记忆接口超出预训练范围而导致教师指导无效、学生记忆动作采样不足的问题。

    

    语言模型智能体越来越多地处理交互历史超出模型活跃上下文的长时程任务。因此，它们必须决定记下什么、读回什么，以及何时执行这两种操作。近期的工作开始使用在线策略蒸馏将记忆控制纳入策略之中，但其依赖的记忆工具和压缩协议都不在教师和学生模型的预训练范围之内。然而，这从两个方面削弱了蒸馏效果：教师模型无法就其从未使用过的接口给出有用的指导，而学生模型采样到的有意义的记忆动作太少，使得这些指导难以发挥作用。为了研究这一问题，我们推出了 Coding Agent Memory Gym II（CAMG-II），这是一套涵盖购物、编程、自动研究、通用和视觉领域的长时程智能体强化学习环境套件。在每个环境的原生任务接口之外，CAMG-II 还提供普通的 Bash 访问权限和一个回合内持久的工作区，从而能够……

    arXiv:2611.14009v1 Announce Type: cross  Abstract: Language-model agents increasingly tackle long-horizon tasks whose interaction histories exceed the model's active context. They must therefore decide what to write down and what to read back, and when to do either. Recent work has begun to use on-policy distillation (OPD) to make memory control part of the policy, relying on memory tools and compaction protocols that lie outside either model's pre-training. However, this weakens the distillation on both sides: the teacher cannot give useful guidance on an interface it has never used, and the student samples too few meaningful memory actions for that guidance to take effect. To study this problem, we introduce Coding Agent Memory Gym II (CAMG-II), a suite of long-horizon agentic-RL environments spanning Shop, Coding, AutoResearch, General, and Visual. Alongside each environment's native task interface, CAMG-II provides ordinary Bash access and an episode-persistent workspace, enabling 
    
[^33]: 自主指定隐私：评估大型视觉语言模型的推理时个性化隐私保护能力

    Specify Privacy Yourself: Assessing Inference-Time Personalized Privacy Preservation Ability of Large Vision-Language Models

    [https://arxiv.org/abs/2611.13997](https://arxiv.org/abs/2611.13997)

    该论文提出了首个系统性评估大型视觉语言模型“推理时个性化隐私保护”（ITP³）能力的基准SPY-Bench，包含32,700个涵盖67个类别和24个真实场景的样本以及基于用户规范与情境感知的新颖指标，旨在使模型能够遵循用户通过语言动态指定的个性化隐私边界。

    

    大型视觉语言模型（LVLMs）已展现出卓越的能力，但由于其能够从图像中以高精度推断敏感个人信息，引发了严重的隐私担忧。虽然当前的LVLMs在保护通用隐私（如信用卡数据）方面已经相对对齐良好，但我们认为隐私本质上是个性化和情境相关的。本工作转向一个新颖的任务：LVLMs能否实现推理时个性化隐私保护（ITP³），即允许用户通过语言规范动态指定隐私边界？为此，我们提出了SPY-Bench，这是首个对ITP³能力进行系统性评估的基准，包含：（1）32,700个独特样本，涵盖67个类别和24个真实世界场景中的图像-问题对和个性化隐私指令；（2）基于用户规范和情境感知的新颖评估指标。（注：原文摘要在此处不完整）

    arXiv:2611.13997v1 Announce Type: cross  Abstract: Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities but raise significant privacy concerns due to their abilities to infer sensitive personal information from images with high precision. While current LVLMs are relatively well aligned to protect universal privacy, e.g., credit card data, we argue that privacy is inherently personalized and context-dependent. This work pivots towards a novel task: can LVLMs achieve Inference-Time Personalized Privacy Protection (ITP$^3$), allowing users to dynamically specify privacy boundaries through language specifications? To this end, we present SPY-Bench, the first systematic assessment of ITP$^3$ ability, which comprises (1) 32,700 unique samples with image-question pairs and personalized privacy instructions across 67 categories and 24 real-world scenarios, and (2) novel metrics grounded in user specifications and context awareness. Benchmarking the ITP$^3$ ability of 
    
[^34]: 关于音素、音节和词汇的信息速率

    On the information rate of phones, syllables and words

    [https://arxiv.org/abs/2611.13990](https://arxiv.org/abs/2611.13990)

    该研究利用960小时英语有声读物语料库，在n-gram和transformer语言模型的有限建模预算下比较不同语言单位的信息速率，发现词汇（及稍长的单位）能实现最低比特率，表明比音节大但小于长多词短语的语言单位是编码语音内容最高效的表示方式。

    

    哪种语言单位能最小化编码语音内容所需的比特率：是音素、音节、词汇，也许是更大的单位，还是介于两者之间的某种单位？这个问题对语音和语言系统的开发以及人类语言处理理论具有重要意义。本文使用960小时英语有声读物的转录语料库，比较了音素、音节、词汇、多词组合以及中间分词方式的信息速率。虽然在完美语言模型下所有单位都是等效的，但在有限的建模预算下比较这些单位是有意义的；作者使用了n-gram和transformer语言模型。结果表明，使用词汇时达到了最低的比特率，稍长的单位有时表现最佳。研究显示，当建模能力受限时，比音节大但小于长多词短语的单位在信息速率方面提供了最高效的表示。

    arXiv:2611.13990v1 Announce Type: new  Abstract: Which linguistic units minimise the bitrate required to encode speech content: phones, syllables, words, maybe larger units, or something in between? This question has implications for the development of speech and language systems and theories of human language processing. Using a transcribed corpus of 960 hours of English audiobooks, I compare the information rates of phones, syllables, words, multi-word combinations and intermediate tokenisations. While all units are equivalent under a perfect language model, it makes sense to compare these under limited modelling budgets; I use $n$-gram and transformer language models. The lowest bitrates are achieved using words, with slightly longer units sometimes performing best. The results show that, when modelling capacity is limited, units larger than syllables but smaller than long multi-word phrases provide the most efficient representation in terms of information rate.
    
[^35]: 超越专家边界理解混合专家语言模型

    Understanding Mixture-of-Experts Language Models Beyond Expert Boundaries

    [https://arxiv.org/abs/2611.13975](https://arxiv.org/abs/2611.13975)

    该研究提出路由感知幅度剖析（RAMP）方法，首次揭示MoE模型中领域专业化计算在神经元层面的组织规律（同领域神经元重叠率高达58-74%，远超跨领域的15-24%），并追踪了这种组织在预训练和后训练全过程中的动态演变。

    

    在混合专家模型中，专家专业化的本质仍然存在争议。关于领域相关专家偏好的报告，与分析将路由主要归因于词元身份或句法模式的研究形成了对立。为了澄清这些不同的观点，我们超越了专家级别的路由，采用一种为此专门开发的方法——路由感知幅度剖析（RAMP）——来研究领域专业化计算在神经元层面的组织方式。使用混合领域校准数据，我们识别出领域专业化的神经元集合：同一领域的多源样本之间重叠率达到58-74%，而跨领域样本之间仅为15-24%。为了理解这种组织结构在训练过程中如何演变，我们在三个完全开源的MoE模型家族中，追踪了整个预训练和后训练过程中的神经元重要性。结果显示，尽管专家级别的重要性相对稳定，神经元重要性在预训练各检查点之间仍持续发生重组。

    arXiv:2611.13975v1 Announce Type: new  Abstract: In Mixture-of-Experts (MoE) models, the nature of expert specialization remains controversial. Reports of domain-associated expert preferences contrast with analyses attributing routing primarily to token identity or syntactic patterns. To clarify these differing accounts, we look beyond expert-level routing to investigate the neuron-level organization of domain-specialized computation using Routing-Aware Magnitude Profiling (RAMP), a measure developed for this analysis. Using mixed-domain calibration data, we identify domain-specialized neuron sets with 58-74% overlap across multi-source samples of the same domain, compared with 15-24% across domains. To understand how this organization evolves during training, we trace neuron importance throughout pre-training and post-training across three fully open MoE families. Neuron importance continues to reorganize between pre-training checkpoints despite comparatively stable expert-level impor
    
[^36]: AVCC：用于话轮转换预测并附带人类基准的视听对话语料库

    AVCC: Audio-Visual Conversation Corpus for Turn-Taking Prediction with a Human Benchmark

    [https://arxiv.org/abs/2611.13957](https://arxiv.org/abs/2611.13957)

    该论文将31小时单摄像头单麦克风录制的多人对话视听语料库AVCC转化为首个面向机器人单传感设备的公开下一发言者预测基准，并提供词级对齐转录与人类表现参考基准。

    

    预测接下来谁将发言、何时发言，是会话代理和社交机器人无需等待沉默即可进行话轮转换的前提条件。多方下一发言者预测此前是在每个发言者配备独立传感器、且各有其事件定义的语料库上研究的。目前尚无公开基准覆盖机器人实际拥有的单个摄像头和麦克风，也没有模型与人类执行相同预测的表现进行过比较。我们将AVCC——一个用单个固定摄像头和麦克风录制的31小时双人及三人在线对话视听语料库——转化为公开的下一发言者预测基准。我们添加了一个发言者互不重叠的七个视频测试分区，其中包含在共同沉默间隙处的865个保持话轮事件和865个话轮转换事件、整个语料库的词级强制对齐转录文本，以及在相同事件处收集的人类参考基准。我们对仅音频的角色相对VAP模型和多模态MuVAP模型进行了评分。

    arXiv:2611.13957v1 Announce Type: new  Abstract: } Predicting who will speak next, and when, is a prerequisite for conversational agents and social robots that take turns without waiting for silence. Multiparty next-speaker prediction (NSP) has been studied on corpora with per-speaker sensors, each with its own event definitions. No public benchmark covers the single camera and microphone a robot has, and no model has been compared against how well people perform the same prediction. We turn AVCC, a 31-hour audio-visual corpus of two- and three-speaker online conversation recorded with a single static camera and microphone, into a public NSP benchmark. We add a speaker-disjoint seven-video test partition with 865 hold and 865 shift events at mutual-silence gaps, word-level forced-aligned transcripts for the whole corpus, and a human reference benchmark collected at the same events. We score an audio-only Role-Relative VAP model (RoleVAP) and the multimodal MuVAP model. Labels, transcri
    
[^37]: 超越流畅回答：在人机交互中发现用户不满

    Beyond Fluent Responses: Finding User Dissatisfaction in Human-AI Interactions

    [https://arxiv.org/abs/2611.13894](https://arxiv.org/abs/2611.13894)

    该研究通过分析带有自我报告不满标签的用户话语，证明人机交互中用户的不满情绪可以从对话文本中被检测出来，从而将回答评估从以回答为中心的质量指标拓展到以用户为中心的信号。

    

    随着大型语言模型生成的回答质量日益提高，近期的回答评估工作已经超越了以回答为中心的标准（如正确性和连贯性），转向以用户为中心的标准，包括用户偏好和感知有用性。这一转变对于开发真正对现实世界用户有用的技术至关重要。尽管传统的回答评估量化的是回答本身的质量，但用户在真实对话中的满意度还取决于其主观偏好和对有用性的感知，而这些并不一定与回答质量相一致。本研究探讨了能否从对话文本中检测出这类以用户为中心的信号。具体而言，我们分析了带有自我报告不满标签的用户话语，以检验能否从文本中检测到用户的不满。在英语交互中，使用用户话语的模型超过了（原文摘要在此处截断）

    arXiv:2611.13894v1 Announce Type: new  Abstract: As large language models generate increasingly high-quality responses, recent work on response evaluation has moved beyond response-centered criteria, such as correctness and coherence, toward user-centered criteria, including user preferences and perceived usefulness. This shift is fundamentally important for developing technologies that are truly useful to real-world users. While conventional response evaluation quantifies the quality of the response itself, users' satisfaction in real conversations also depends on their subjective preferences and perceptions of usefulness, which do not necessarily align with response quality. This study investigates whether such user-centered signals can be detected from conversational text. Specifically, we analyze user utterances with self-reported dissatisfaction labels to examine whether user dissatisfaction in text can be detected. In English interaction, models using user utterances exceeded the
    
[^38]: 因果神经元回路是唯一且可迁移的吗？

    Are Causal Neuron Circuits Unique and Transferable?

    [https://arxiv.org/abs/2611.13885](https://arxiv.org/abs/2611.13885)

    该论文通过穷举测量 12 个 MLP 神经元全部 4,096 种子集的干预效果，发现虽然消融选定神经元组会显著改变模型回答，但这些神经元的激活在不同问题间几乎恒定、替换后也几乎无影响，说明真正必要的只是其中一部分，从而质疑了因果神经元回路的唯一性与可迁移性。

    

    可解释性领域的演示常常表明，对一组神经元进行干预会改变模型的回答。最近的一个例子是：移除 Qwen3-32B 中的一组 18 个神经元后，模型对一个两跳《狮子王》谜语的回答从“刀疤”变成了“辛巴”。这类演示证明了该神经元组确实重要，但没有说明其中有多少神经元是真正必要的、其他神经元组是否能起到同样作用，以及这种效应是否会迁移到其他输入上。我们提出的问题是：当对一组神经元的干预改变了模型的回答时，这组神经元中究竟有多少是必要的？对于每个选定的神经元组，我们测量了 12 个 MLP 神经元全部 4,096 种子集的效果。在一个二选一的首都识别任务（Qwen3-0.6B）上，将 12 个选定神经元置零会使答案置信边际下降 27%，而随机神经元几乎不产生改变；然而，用其他问题中的激活值替换这同一批神经元同样几乎不改变结果：这些神经元的激活在不同问题之间几乎恒定，我们没有找到证据（摘要原文在此处截断）

    arXiv:2611.13885v1 Announce Type: cross  Abstract: Interpretability demonstrations often show that intervening on a group of neurons changes a model's answer. A recent example removes an 18-neuron group in Qwen3-32B and the model answers "Simba" instead of "Scar" to a two-hop Lion King riddle. Such a demonstration establishes that the group matters, but not how much of it is necessary, whether other groups could do the same, or whether the effect carries over to other inputs. We ask: when intervening on a group of neurons changes a model's answer, how much of that group is actually necessary? For every selected group we measure all 4,096 subsets of 12 MLP neurons. On a two-alternative capital task (Qwen3-0.6B), zero-ablating 12 selected neurons lowers the answer margin by 27% while random neurons barely change it, yet replacing the same neurons with their values from other questions also barely changes it: their activations are nearly constant across questions, and we find no evidence 
    
[^39]: 超越被动模仿：面向语言模型的在线策略中期训练蒸馏

    Beyond Passive Imitation: On-Policy Mid-Training Distillation for Language Models

    [https://arxiv.org/abs/2611.13855](https://arxiv.org/abs/2611.13855)

    该论文提出在线策略中期训练蒸馏（OPMD），将中期训练从被动的离线文本模仿转变为主动的、查询驱动的知识获取，通过从学生策略自采样轨迹并结合教师模型的词元级反向KL奖励，解决了标准下一词预测目标在不同基础模型上效果不一甚至产生负面影响的问题。

    

    持续预训练（CPT）可将大型语言模型（LLM）适配到特定领域，然而其标准的下一词预测（NTP）目标将语料库中的每个词元都视为固定目标，而不考虑其对当前模型的效用。在一项基于相同10亿词元数学语料库的对照研究中，我们观察到不同基础模型之间结果差异显著：NTP提升了Llama-3.2-3B在数学基准上的表现，却显著降低了Qwen3-1.7B-Base的性能，这表明在固定的训练方案下，相同的训练序列可能产生依赖于具体模型的效果。为解决这一问题，我们提出了在线策略中期训练蒸馏（OPMD），将中期训练从离线的文本模仿重新表述为主动的、查询驱动的知识获取。在选定的锚定位置，OPMD从不断演进的学生策略中采样轨迹，并应用来自冻结教师模型的密集词元级反向KL奖励进行训练。

    arXiv:2611.13855v1 Announce Type: new  Abstract: Continual pre-training (CPT) adapts large language models (LLMs) to specialized domains, yet its standard next-token prediction (NTP) objective treats every corpus token as a fixed target, regardless of its utility to the current model. In a controlled study on the same 1B-token mathematical corpus, we observe sharply different outcomes across base models: NTP improves Llama-3.2-3B's performance on mathematical benchmarks but substantially degrades Qwen3-1.7B-Base's performance, indicating that, under a fixed training recipe, the same training sequences can have model-dependent effects. To address this problem, we introduce On-Policy Mid-Training Distillation (OPMD), which reformulates mid-training from offline text imitation into active, query-driven knowledge acquisition. At selected anchor positions, OPMD samples trajectories from the evolving student policy and applies dense token-level reverse-KL rewards from a frozen teacher, train
    
[^40]: Alania-2：支持即时与微调语音克隆的土耳其语文本转语音系统

    Alania-2: Turkish text-to-speech with instant and fine-tuned voice cloning

    [https://arxiv.org/abs/2611.13845](https://arxiv.org/abs/2611.13845)

    Alania-2是一个基于VoxCPM架构的4.88亿参数土耳其语文本转语音系统，在1.8万余小时土耳其语语音（90%为教师模型合成数据）上从头训练，结合文本规范化技术，将词错误率降至1.44%，并支持即时与微调两种语音克隆方式。

    

    我们提出了Alania-2，这是一个土耳其语文本转语音（TTS）系统，具有两个内置语音、基于文本描述的语音设计功能，以及两种语音克隆方式。Alania-2是一个4.88亿参数的模型，采用VoxCPM架构，在语言模型主干之上通过扩散头生成连续的音频潜在表示。我们在18,195小时的土耳其语语音上从头训练该模型，其中90%是由更大的教师模型生成并经过自动过滤的合成语音。文本规范化器将数字、缩写和外国名称改写为土耳其语使用者实际会读出的词汇；若没有它，96个含数字的测试句子中没有一个能在合成与语音识别的往返过程中保留所有关键部分（如金额或日期），而有它时为86个。在Freya-TR-Eval（一个包含495个土耳其语句子并通过自动语音识别评分的公开基准）上，Alania-2实现了1.44%的词错误率，而我们之前的系统为5.99%。

    arXiv:2611.13845v1 Announce Type: new  Abstract: We present Alania-2, a Turkish text-to-speech (TTS) system with two built-in voices, voice design from text descriptions, and two kinds of voice cloning. Alania-2 is a 488M-parameter model with the VoxCPM architecture, which generates continuous audio latents with a diffusion head on top of a language-model backbone. We train it from scratch on 18,195 hours of Turkish speech, 90% of which is synthetic speech produced by a larger teacher model and filtered automatically. A text normalizer rewrites numbers, abbreviations and foreign names into the words a Turkish speaker would say; without it, none of 96 test sentences with digits kept all its critical parts, such as an amount or a date, through synthesis and speech recognition, against 86 with it. On Freya-TR-Eval, a public benchmark of 495 Turkish sentences scored with automatic speech recognition, Alania-2 reaches a word error rate of 1.44%, compared with 5.99% for our previous system. 
    
[^41]: PolyRoute：一个面向类型化决策路由的十二语言与混码文本基准，附带基线与排行榜

    PolyRoute: A Twelve-Language and Code-Mixed Benchmark for Typed-Decision Routing, with Baselines and a Leaderboard

    [https://arxiv.org/abs/2611.13843](https://arxiv.org/abs/2611.13843)

    该论文发布了PolyRoute基准，涵盖12种语言、9种书写系统及混码文本的类型化决策路由评测，并提供13个系统的排行榜，显示微调模型可达96.6%准确率，且多语言模型对未公开语言具有良好的泛化能力。

    

    类型化决策编码器可以在一次前向传播中回答关于文本的问题，但其在英语之外的表现很少在统一的评测标准下得到衡量。我们发布了PolyRoute基准：包含12种语言、9种书写系统的322条母语支持消息，156条混码或罗马化消息（印英混合语、西英混合语、阿拉伯罗马化文本），967条可选的母语训练数据行，以及一个三方部门路由决策的评测协议。标签由Qwen3.8-27B生成并经过盲审二次筛选（91%的候选被保留）；独立模型gemma4-31b在97.2%的测试项上与之一致，且任意测试文本与训练文本之间的最大字符5-gram相似度仅为0.33。13个系统的排行榜显示：英语根检查点得分为61.2%，官方发布的多语言版本为77.3%，最佳微调模型达到96.6%；TF-IDF结合逻辑回归达到91.9%。在四个未公开的语言上，三个多语言系统分别取得77%、90%和93%的成绩。在混码文本上……

    arXiv:2611.13843v1 Announce Type: cross  Abstract: Typed-decision encoders answer a question about a text in one forward pass, but their behaviour outside English is rarely measured on a fixed yardstick. We release PolyRoute: 322 native-language support messages in 12 languages and 9 writing systems, 156 code-mixed or romanised messages (Hinglish, Spanglish, Arabizi), 967 optional native training rows, and a protocol for a three-way department-routing decision. Labels come from Qwen3.8-27B with a blind second pass (91% of candidates kept); an independent model, gemma4-31b, agrees on 97.2% of test items, and the largest character 5-gram similarity between any test text and any training text is 0.33. A leaderboard of 13 systems gives 61.2% for the English-root checkpoint, 77.3% for the shipped multilingual one and 96.6% for the best fine-tune; TF-IDF with logistic regression reaches 91.9%. On four withheld languages the three multilingual systems score 77%, 90% and 93%. On code-mixed tex
    
[^42]: Antalia-2 Mini：一个完全基于合成语音训练的760万参数土耳其语文本转语音模型

    Antalia-2 Mini: A 7.6-million-parameter Turkish text-to-speech model trained entirely on synthetic speech

    [https://arxiv.org/abs/2611.13811](https://arxiv.org/abs/2611.13811)

    Antalia-2 Mini是一个仅用合成语音训练、仅760万参数的超轻量土耳其语TTS模型，可在笔记本CPU上以41倍实时速度生成高质量语音，WER低至1.09%。

    

    Antalia-2 Mini 是一个单音色土耳其语文本转语音模型，拥有7,620,931个参数。它能以每秒41倍实时速度在Apple M5笔记本电脑CPU（2线程）上读取字符并生成48 kHz音频。其训练音频仅来自VoxCPM2（教师模型），后者规模约大300倍，通过克隆一段合成音频片段；自动检查保留了约1,790小时的音频。在声学模型自身的梅尔频谱图上对声码器进行微调，使Whisper转录文本的词错误率（WER）从相邻的预发布检查点的平均值3.89%降至1.75%。当片段首词以"o开头时，教师模型会损坏片段的开头，学生模型学会了这一点，而我们通过从输出中剪除一个语音前缀来隐藏这一问题。在495句话的Freya-TR-Eval基准测试（我们调整设置所用的基准）上，1.0.1版本在我们的公开测试框架中达到1.09%的词错误率和0.22%的字符错误率。配对自助法检验无法将其结果与8.6M参数的土耳其语模型EMA Lightning区分开来，后者在其中得分更高。

    arXiv:2611.13811v1 Announce Type: new  Abstract: Antalia-2 Mini is a single-voice Turkish text-to-speech model with 7,620,931 parameters. It reads characters and produces 48 kHz audio at 41 times real time on an Apple M5 laptop CPU (2 threads). Its training audio came only from VoxCPM2 (the teacher), about 300 times larger, cloning one synthetic clip; automatic checks kept about 1,790 h. Fine-tuning the vocoder on the acoustic model's own mel spectrograms cut the word error rate (WER) of Whisper's transcripts from a mean of 3.89% to 1.75% at adjacent pre-release checkpoints. The teacher corrupts the start of a clip when the first word begins with \"o, the student learned this, and a spoken prefix we cut from the output hides it. On the 495-sentence Freya-TR-Eval benchmark, where we tuned settings, version 1.0.1 has WER 1.09% and character error rate 0.22% on our public harness. A paired bootstrap cannot distinguish these from EMA Lightning, an 8.6M Turkish model, which scores higher on
    
[^43]: LLM验证使最小对子基准对语言模型而言变得更简单

    LLM Validation Makes Minimal-Pair Benchmarks Easier for Language Models

    [https://arxiv.org/abs/2611.13788](https://arxiv.org/abs/2611.13788)

    用LLM验证最小对子基准的标签并非随机出错，而是系统性地移除对语言模型而言更难的句子对，使基准变简单，且其验证效果并不优于一名人类评委。

    

    最小对子基准的质量完全取决于其标签，而大语言模型（LLM）是昂贵的人工验证的显而易见替代方案。我们在BLiMP（英语）和QFrBLiMP（魁北克法语）上测试了这一方法，这两个公开的最小对子数据集发布了每一份强制选择的人类判断。三个LLM分别通过强制选择、打分和评审小组的方式进行验证，并在两个预注册阶段与由一名、三名和五名评委组成的人工评审小组进行对比评分。在BLiMP上，强制选择验证器与留出的人工多数意见的一致率高于五人评审小组，但它们能发现人类拒绝其标签的句子对的数量远少于单个评委。从点估计来看，没有任何验证器的价值超过一名人类评委。这些错误并非随机的。在两个数据集上，跨越10个开源语言模型，被预注册的LLM过滤器标记的有人类支持的句子对，比它接受的句子对难18.9个百分点，且在所有70个语言模型-验证器组合中均是如此，也比单个人类评委标记的句子对更难。因此，过滤操作移除的是……（原文摘要在此处截断）

    arXiv:2611.13788v1 Announce Type: new  Abstract: Minimal-pair benchmarks are only as good as their labels, and LLMs are an obvious substitute for costly human validation. We test it on BLiMP (English) and QFrBLiMP (Quebec French), the two public minimal-pair datasets releasing every forced-choice human judgment. Three LLMs validate by forced choice, by rating, and as a panel, scored against human panels of one, three and five judges in two pre-registered phases. On BLiMP, forced-choice validators agree with held-out human majorities more often than five-judge panels, yet catch far fewer pairs whose label humans reject than one judge does. By point estimate, no validator is worth more than one judge. The errors are not random. On both datasets, across 10 open LMs, human-supported pairs that the pre-registered LLM filter flags are 18.9 points harder than those it accepts, in all 70 LM-validator combinations, and harder than pairs a single human judge flags. Filtering thus removes items o
    
[^44]: 区域划分与阅读顺序错误下OCR与标准答案的词级对齐

    Word-Level Alignment of OCR and Ground Truth under Zoning and Reading-Order Errors

    [https://arxiv.org/abs/2611.13752](https://arxiv.org/abs/2611.13752)

    本文重新审视了OCR文本与标准答案的词级对齐问题，提出了一种在区域划分和阅读顺序错误（置换）下评估对齐器正确性的协议，并系统比较了多种对齐方法在阅读顺序扰动下的性能表现。

    

    OCR后处理需要将OCR文本与其标准答案逐词对齐。数据集很少提供这种映射关系，而当区域划分错误打乱阅读顺序时，单调对齐器可能会失去对齐位置。我们重新审视了这一问题，即我们2019年论文中提出的区域划分问题，并设计了一种在这些顺序置换下测试正确性的评估协议。我们在OCR侧应用阅读顺序置换，结合构建时的词级黄金标准，以正确匹配F1值、级联长度和恢复率作为评分指标。我们比较了2019年的位置感知对齐器、其重锚定扩展版本、RETAS风格的基线（RETAS-lite）、一种结合独特单词与独特n-gram锚点以及块到窗口分配的多区域对齐器、genalog的RETAS以及原始Java RETAS。实验使用了包含块移动、段落交换和列交错的合成字符噪声、历史报纸的阅读顺序置换、为Tesseract OCR导出的几何黄金标准，以及一个小型词框数据集。

    arXiv:2611.13752v1 Announce Type: new  Abstract: OCR post-processing needs OCR text aligned word by word with its ground truth. Data sets rarely provide that mapping, and a monotone aligner can lose its place when zoning errors permute the reading order. We revisit this problem, the Zoning Problem of our 2019 paper, with a protocol that tests correctness under such permutations. We apply OCR-side reading-order permutations with construction-time word-level gold and score correct-match F1, cascade length and recovery. We compare the 2019 location-aware aligner, a re-anchoring extension of it, a RETAS-style baseline (RETAS-lite), a multi-region aligner that combines unique-word and unique n-gram anchors with chunk-to-window assignment, genalog's RETAS and the original Java RETAS. Experiments use synthetic character noise with block moves, paragraph swaps and column interleaving, historical newspaper reading-order permutations, geometry-derived gold for Tesseract OCR, and a small word-box
    
[^45]: 语言模型能感受情感吗？探索情绪向量

    Do Language Models Feel the Sentiment? Investigating Emotion Vectors

    [https://arxiv.org/abs/2611.13744](https://arxiv.org/abs/2611.13744)

    本研究在 Qwen2.5 模型的内部表示中定位了与情感概念对应的情绪向量方向，并通过情感探测和匹配消融实验证实了这些方向确实编码了情感信息，移除它们会显著降低模型的情感分类准确率。

    

    arXiv:2611.13744v1 公告类型：新论文 摘要：情绪向量是语言模型内部表示中与情感概念相关联的方向。先前的研究表明，这些向量能够影响模型行为，但它们在情感分类中的作用仍不清楚。本研究对 Qwen2.5-0.5B-Instruct 和 Qwen2.5-3B-Instruct 中的情绪方向进行了探究。从 GoEmotions 数据集中提取的八个方向通过情感探测、激活干预和监督校正进行评估。仅使用情绪特征的探测器在二分类 TweetEval 任务上分别达到 88.66% 和 90.70% 的准确率。在探索性的匹配消融实验中，对于 0.5B 和 3B 模型，移除原始情绪空间分别比移除匹配的各向同性随机子空间多造成 2.39 和 1.10 个百分点的准确率损失。经过八次比较的 Holm 校正后，TweetEval 上的全部四个原始效应和去噪效应均保持显著。一项探索性的 SST-2 扩展实验显示，原始超额损失较小，分别为 0.66 和 0.57 个百分点。

    arXiv:2611.13744v1 Announce Type: new  Abstract: Emotion vectors are directions in a language model's internal representations that are associated with emotional concepts. Prior work shows that these vectors can influence model behavior, but their role in sentiment classification remains unclear. This study investigates emotion directions in Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct. Eight directions extracted from GoEmotions are evaluated through sentiment probing, activation interventions, and supervised correction. Emotion-only probes achieve 88.66% and 90.70% accuracy on binary TweetEval. In exploratory matched-ablation experiments, raw emotion-space removal causes 2.39 and 1.10 percentage points more accuracy loss than matched isotropic random-subspace removal for 0.5B and 3B, respectively. All four TweetEval raw and denoised effects are significant after Holm adjustment across eight comparisons. An exploratory SST-2 extension shows smaller raw excess losses of 0.66 and 0.57 p
    
[^46]: Walrus 技术报告

    Walrus Technical Report

    [https://arxiv.org/abs/2611.13689](https://arxiv.org/abs/2611.13689)

    Walrus 是一个统一的 8B 语音与音频基础模型，其核心贡献在于一套以数据为中心的三阶段课程训练方案：在标准 decoder-only 骨干上通过任务感知加权，实现从多语言到多说话人、语码转换等场景的广泛语音理解与生成能力。

    

    我们提出 Walrus，一个统一的 8B 语音与音频基础模型，能够处理语音识别（包括语码转换和多说话人语音）、语音到文本与语音到语音翻译、文本到语音、情感识别、音频描述生成以及通用音频理解。我们的主要贡献并非新架构，而是一套以数据为中心的训练方案；我们使用从 Gemma 4 初始化的标准 decoder-only 骨干网络，从而使性能提升可以明确归因于数据。在语音生成方面，Walrus 预测低码率的离散语音 token，再由轻量级的 token 到波形解码器将其转换为高保真音频。该方案采用三阶段数据课程，从广泛的多语言数据逐步过渡到多说话人、语码转换和特定领域数据。在每个阶段，任务感知与数据集感知的加权机制平衡了 token 预算差异极大的各项任务，并保护核心技能免遭遗忘。模型在超过 6 亿个样本上训练。

    arXiv:2611.13689v1 Announce Type: cross  Abstract: We present Walrus, a unified 8B speech-and-audio foundation model handling speech recognition (including code-switching and multi-talker speech), speech-to-text and speech-to-speech translation, text-to-speech, emotion recognition, audio captioning, and general audio understanding. Our main contribution is not a new architecture but a data-centric training recipe; we use a standard decoder-only backbone initialized from Gemma 4, so that gains can be traced to the data. For speech generation, Walrus predicts low-bitrate discrete speech tokens, and a lightweight token-to-waveform decoder converts them into high-fidelity audio. The recipe is a three-stage data curriculum that moves from broad multilingual data to multi-talker, code-switching, and domain-specific data. At each stage, task-aware and dataset-aware weighting balances tasks with very different token budgets and protects core skills from forgetting. Trained on over 600M samples
    
[^47]: 阅读塑造你：面向印地语的印度政治新闻媒体偏见数据集

    You Are What You Read: Indian Political News Media Bias Dataset for Hindi Language

    [https://arxiv.org/abs/2611.13686](https://arxiv.org/abs/2611.13686)

    本文构建了一个由四家印度主要新闻机构的448篇印地语政治新闻文章组成、经14位专家标注为四种偏见类别（选择性、框架、标题党、遗漏偏见）的可解释媒体偏见数据集，填补了印地语媒体偏见研究的空白。

    

    媒体偏见理解与检测在印度语境下仍是一个研究不足的问题，尤其是针对印地语等在印度被广泛使用的印度语系语言。本文介绍了一个全面的、由多人标注、可解释的数据集，该数据集由四家以印地语出版的主要印度新闻机构的448篇独特文章组成。我们聚焦于印地语——印度使用人数最多且为官方语言之一。由于印度语言多样、人口庞大，同一事件常被多家新闻机构以不同语言报道，从而产生偏见的可能，因此媒体偏见研究在印度语境下尤为重要。该数据集包含由14位专家标注的印地语政治新闻文章，当文章被判定为有偏见或无偏见时，会被归类到四种偏见类别——选择性偏见、框架偏见、标题党偏见和遗漏偏见。对于每篇被判定有偏见的文章，标注者……

    arXiv:2611.13686v1 Announce Type: new  Abstract: Media bias understanding and detection remains an under-researched problem for Indian context, specifically for Indic languages, such as, Hindi, which is widely spoken in India. In this paper, we introduce a comprehensive, multi-annotator, explainable dataset consisting of 448 unique articles from four leading Indian news agencies publishing in Hindi. We focus on Hindi, the most spoken and an official language of India. The study of media bias is particularly relevant in the Indian context due to the country's linguistic diversity and large population, where the same event is often reported by multiple news agencies in different languages, thereby introducing scope for bias. The dataset comprises Hindi political news articles annotated by 14 experts and classified into four bias categories - selection bias, framing bias, clickbait bias and omission bias - when an article is identified as biased or unbiased. For each biased article, annot
    
[^48]: 光学RAG：面向令牌高效检索增强生成的像素空间压缩

    OpticalRAG: Pixel-Space Compression for Token-Efficient Retrieval-Augmented Generation

    [https://arxiv.org/abs/2611.13675](https://arxiv.org/abs/2611.13675)

    该论文提出OpticalRAG框架，通过将文档渲染为页面图像并利用冻结视觉编码器将检索内容蒸馏压缩为最少仅16个视觉令牌注入大语言模型，在无需昂贵跨模态训练的情况下突破了RAG的令牌预算瓶颈。

    

    检索增强生成（RAG）受到大语言模型令牌预算的瓶颈制约：将长文档输入上下文窗口代价高昂，而激进的文本压缩会丢失细粒度信息，且这些信息在一维令牌序列中会被不可逆地损失掉。我们认为这一瓶颈并非源于信息密度的根本限制，而是源于压缩“模态”的选择。现代视觉编码器可以将渲染后的文本页面内容压缩到大约一百个视觉令牌中且几乎没有语义损失，从而开辟了一种纯文本方法无法企及的全新的压缩-保真度权衡。我们提出了OpticalRAG，该框架将文档渲染为页面图像，使用冻结的视觉编码器对其进行编码，并在传递给大语言模型之前将检索到的视觉令牌蒸馏为最少仅16个注入令牌。两个关键要素使得该方法无需昂贵的跨模态训练即可实现：（i）（原文摘要在此处截断）

    arXiv:2611.13675v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) is bottlenecked by the token budget of large language models: feeding long documents into the context window is expensive, while aggressive textual compression discards fine-grained information that is irreversibly lost in a one-dimensional token sequence. We argue that this bottleneck is not fundamental to information density but to the choice of compression \emph{modality}. Modern visual encoders can pack the contents of a rendered text page into roughly a hundred visual tokens with little semantic loss, opening a new compression-fidelity tradeoff that purely textual methods cannot reach. We present \textbf{OpticalRAG}, a framework that renders documents as page images, encodes them with a frozen vision encoder, and distills the retrieved visual tokens into as few as $16$ injected tokens before passing them to the LLM. Two ingredients make this work without expensive cross-modal training: (i) \emph{
    
[^49]: 早期汉译佛典译者风格的可解释文体计量学框架：以安世高存疑著作归属为案例研究

    An Interpretable Stylometric Framework for Early Chinese Buddhist Translator Styles: A Case Study on the Attribution of Disputed Works of An Shigao

    [https://arxiv.org/abs/2611.13636](https://arxiv.org/abs/2611.13636)

    本研究提出了一个基于佛教文献学家审定语言特征的可解释文体计量学框架，无需任务特定训练即可分析早期汉译佛典归属，判定T1557和T0101两部存疑文本与安世高风格最为相似，且结果与基于BERT的方法一致。

    

    早期汉译佛典的作者归属需要学者能够回溯至文本本身的证据。我们开发了一个可解释的文体计量学框架，该框架使用由佛教文献学家定义或审定的特征，包括虚字、句式、句末字、句长和词性n-gram。我们根据从东汉到西晋七位译师的31部确证译作构建了风格特征档案，然后将八部存疑文本与安世高的风格档案进行比较。其中两部文本T1557和T0101与确证的安世高语料库显示出最强的风格相似性，得分分别为0.783和0.687。这些归属结果与已发表的基于BERT的方法在同一批存疑文本上的结果一致。该框架无需针对特定任务的训练，并允许学者检验哪些语言特征支持每一项归属提议。

    arXiv:2611.13636v1 Announce Type: new  Abstract: Attributing early Chinese Buddhist translations requires evidence that scholars can trace back to the texts. We develop an interpretable stylometric framework that uses features defined or reviewed by Buddhist philologists, including function characters, sentence formulas, sentence-final characters, sentence length, and part-of-speech n-grams. We construct stylistic profiles from 31 authenticated translations by seven translators from the Eastern Han through the Western Jin, then compare eight disputed texts with An Shigao's profile. Two texts, T1557 and T0101, show the strongest stylistic similarity to the authenticated An Shigao corpus, with scores of 0.783 and 0.687, respectively. The attributions agree with those reported by a published BERT-based approach on the same set of disputed texts. The framework requires no task-specific training and allows scholars to inspect which linguistic features support each proposed attribution.
    
[^50]: Synapse-Control：通过轻量级运行时控制使智能体经验可操作化

    Synapse-Control: Making Agent Experience Actionable with Lightweight Runtime Control

    [https://arxiv.org/abs/2611.13615](https://arxiv.org/abs/2611.13615)

    Synapse-Control用一个轻量级智能体运行时控制平面（规则筛选、1.7B微调分类器评分、微合同接口检查和提示价值表冷却机制）取代大模型失败诊断，以更低的推理成本实现智能体执行经验的高效复用与精准投放。

    

    大型语言模型智能体通过复用执行经验来提高其在复杂工具使用任务上的可靠性。诸如Sentry之类的系统在检索建议之前需要调用大模型生成失败诊断，这为经验复用增加了推理成本。我们提出Synapse-Control，这是一个智能体运行时，其轻量级控制平面负责选择建议并决定何时将其发送给执行任务的模型。规则首先筛选出相关建议，然后由一个在教师标注的状态-建议对上微调的轻量级1.7B分类器对这些候选建议进行评分，而不是生成诊断。微合同编码了可复用的接口检查，例如必需参数或有效选项，而无需规定完整的任务程序。提示价值表记录先前的指导和反馈，通过冷却机制和令牌预算检查来限制重复建议。在AppWorld和WebShop上每种方法进行1,308次尝试的评测中，Synapse-Control实现了270次成功（29.（摘要在此处截断）

    arXiv:2611.13615v1 Announce Type: cross  Abstract: Large language model agents reuse execution experience to improve reliability on complex tool-based tasks. Systems such as Sentry call large models to generate failure diagnoses before retrieving advice, adding inference cost to experience reuse. We present Synapse-Control, an agent runtime whose lightweight control plane selects advice and determines when to send it to the task-executing model. Rules shortlist relevant advice, and a lightweight 1.7B classifier fine-tuned on teacher-labeled state-suggestion pairs scores those candidates instead of generating a diagnosis. Micro-contracts encode reusable interface checks, such as required arguments or valid options, without prescribing complete task programs. A Prompt Value Table records earlier guidance and feedback to limit repeated suggestions through cooldown and token-budget checks. Across 1,308 attempts per method in AppWorld and WebShop, Synapse-Control achieves 270 successes (29.
    
[^51]: 乌兹别克语混合文字文本的OCR后纠错：基于经验同形字符噪声矩阵的字节级ByT5模型

    Post-OCR Correction of Mixed-Script Uzbek Text: Byte-Level ByT5 with an Empirical Homoglyph Noise Matrix

    [https://arxiv.org/abs/2611.13609](https://arxiv.org/abs/2611.13609)

    针对乌兹别克语中西里尔与拉丁字母混排导致的同形字符OCR错误，提出可在本地硬件上运行的字节级ByT5纠错系统，结合由人工标注错误事件估计的噪声矩阵与保守解码规则，将字符错误率从0.0578降至0.0369。

    

    在乌兹别克斯坦的官方和技术文档中，很大一部分OCR错误并非由图像质量差造成，而是由当前并用的两种字母表中视觉上完全相同的字符所引起。由于文字转型仍在进行中，西里尔字母与拉丁字母会出现在同一页面上，甚至经常出现在同一句话中；据我们的测量，此类页面上同形字符错误的密度约为单一文字文本的近五倍。我们在一个特殊受限的场景下研究该问题：工程地质与ICT基础设施报告语料库不得离开本机构，因此所有阶段均在本地硬件上执行。所提出的系统将字节级ByT5模型与一个噪声生成器相结合，该生成器的采样权重由359个人工分类的错误事件估计得出，并辅以受莱文斯坦距离约束的保守解码规则。在包含5,204个行对的黄金测试集上，该系统将字符错误率从0.0578降低至0.0369，并恢复了……

    arXiv:2611.13609v1 Announce Type: new  Abstract: In Uzbek official and technical documents, a large share of OCR errors is caused not by poor image quality but by characters that are visually identical across the two alphabets in use. Because the script transition is still under way, Cyrillic and Latin appear on the same page and often within a single sentence, and in our measurements the density of homoglyph errors on such pages is close to five times higher than in single-script text. We study this problem in a setting where the corpus of engineering-geological and ICT infrastructure reports may not leave the organisation, so every stage was executed on local hardware. The proposed system combines a byte-level ByT5 model with a noise generator whose sampling weights are estimated from 359 manually classified error events, together with a Levenshtein-bounded conservative decoding rule. On a Gold set of 5,204 line pairs it lowers the character error rate from 0.0578 to 0.0369, restores
    
[^52]: 一字节，一个词元：利用自生成数据将27B混合注意力大语言模型字节化

    One Byte, One Token: Byteifying a 27B Hybrid-Attention LLM with Self-Generated Data

    [https://arxiv.org/abs/2611.13607](https://arxiv.org/abs/2611.13607)

    研究者通过将词表缩减至384个条目（含256个字节符号），并利用原始模型自生成的4.77GB在线策略监督数据进行两阶段课程训练，成功将270亿参数的Qwen3.8混合注意力模型直接转换为纯字节级模型，无需补丁结构或局部编码器即可保留其大部分原有能力。

    

    字节级BPE会将中文文本碎片化（在中文维基百科上，Qwen3.8平均每个词元对应3.84字节；64%的词元是单个字符），并对模型隐藏了拼写信息。我们探究是否可以将一个已经训练好的27B子词模型直接转换为纯字节级模型（一个字节，一个词元），而无需补丁结构或局部编码器，同时保留其大部分能力。从Qwen3.8-27B（48层Gated DeltaNet层加16层全注意力层）出发，我们将词表从248,320个条目缩减至384个（BPE词表中已有的256个字节符号加上特殊标识），保持其他所有权重不变，并在4.77 GB的监督字节数据上进行两阶段课程训练（先编码，后阅读）：所有对话/行为监督数据均由原始模型自身以在线策略方式生成（包括字符词典、词语词典、通用对话、编程会话），知识阅读部分则使用百科文本。

    arXiv:2611.13607v1 Announce Type: new  Abstract: Byte-level BPE fragments Chinese text (for Qwen3.8 on Chinese Wikipedia, 3.84 bytes per token on average; 64% of tokens are single characters) and hides spelling from the model. We ask whether an already-trained 27B subword model can be converted directly into a pure byte-level model (one byte, one token), without a patching structure or a local encoder, while keeping most of its ability. Starting from Qwen3.8-27B (48 Gated DeltaNet layers + 16 full-attention layers), we shrink the vocabulary from 248,320 to 384 entries (the 256 byte symbols already in the BPE vocabulary plus special ids), leave every other weight untouched, and continue training in a two-stage curriculum (encoding first, then reading) on 4.77 GB of supervised bytes: all dialogue/behavioral supervision is generated on-policy by the original model itself (character dictionary, word dictionary, general dialogue, coding sessions), and knowledge reading uses encyclopedia tex
    
[^53]: 计算交互，而非属性：中国六十甲子系统中事实层必须提供什么

    Compute the Interactions, Not the Attributes: What a Fact Layer Must Supply in the Chinese Sexagenary System

    [https://arxiv.org/abs/2611.13597](https://arxiv.org/abs/2611.13597)

    该研究以八字六十甲子系统构建可查表事实基准，发现开源语言模型在需要组合属性的交互型事实上几乎全部失败，而提供按查询过滤、已计算好的交互事实可将准确率提升至96–100%，说明事实层应当“计算交互”而非罗列静态属性。

    

    在受规则支配的领域中，语言模型系统通常将模型与一个确定性的事实层配对使用，但该事实层究竟应提供哪些事实却鲜有量化衡量。我们在八字排盘分析所依据的中国六十甲子系统中对此进行了测量，其中每个事实都有一个可通过查表核验的唯一值：包括十二个事实类别下的682个题目，以及160个流年—本命冲撞题目；测试分别在闭卷、提供参考表、以及提供递增级别计算事实的条件下进行。在十一个开源权重模型直接作答时，错误集中在必须组合属性之处：没有任何模型在六十甲子日上答对超过4%，在冲撞题目上答对超过1%。静态属性没有一致效果，而完整的关系列表虽然包含答案，却仍使大多数直接作答的模型接近零分。一个按查询过滤的事实层（探索性方案）将Qwen3-8B与Qwen3-32B提升至78–92%，并将另外四个模型提升至96–100%。三个专有推理模型可计算68-……（摘要原文在此处截断）

    arXiv:2611.13597v1 Announce Type: new  Abstract: Language-model systems in rule-governed domains often pair the model with a deterministic fact layer, but which facts that layer should supply is rarely measured. We measure it in the Chinese sexagenary system underlying BaZi chart analysis, where every fact has one table-checkable value: 682 items over twelve fact categories and 160 transit--natal collision items, posed closed-book, with the reference table, and with computed facts at increasing levels. In eleven open-weight models answering directly, errors concentrate where attributes must be combined: none gets more than 4\% of sexagenary days or 1\% of collisions right. Static attributes have no consistent effect, and the complete relation list leaves most direct-answering models near zero although it contains the answer. A query-filtered layer (exploratory) raises Qwen3-8B and Qwen3-32B to 78--92\% and four further models to 96--100\%. Three proprietary reasoning models compute 68-
    
[^54]: 迈向可靠的视觉语言模型评判者：状态条件不变性与呈现感知诊断

    Towards Reliable VLM Judges: State-Conditional Invariance and Presentation-Aware Diagnostics

    [https://arxiv.org/abs/2611.13596](https://arxiv.org/abs/2611.13596)

    该论文提出状态条件不变性（SCI）这一可靠性原则，并构建了基于模拟器的诊断基准GridWM-Judge，用于系统评估视觉语言模型评判者在语义不变的呈现变化下是否保持稳定、并对改变任务结果的因果干预作出正确响应。

    

    视觉语言模型（VLM）日益被用作多模态和具身任务的自动评判者，然而仅凭总体准确率或一致性无法揭示评判者改变其判决是否出于正确的原因。我们提出了状态条件不变性（SCI）这一可靠性原则，要求评判者在语义保持不变的呈现变化下保持稳定，同时对改变任务结果的因果干预作出相应变化。我们在GridWM-Judge中实现了SCI，这是一个基于模拟器的诊断基准，由确定性的MiniGrid轨迹构建，包含对齐的完整、证据移除和反事实变体，以及呈现与表示探针。通过五个诊断任务，GridWM-Judge测试局部感知、动作条件转移、呈现稳定性、表示可交换性以及Full-CF因果判别能力。我们的诊断评估揭示了异质性的失败模式，包括……

    arXiv:2611.13596v1 Announce Type: new  Abstract: Vision-language models (VLMs) are increasingly used as automated judges for multimodal and embodied tasks, yet aggregate accuracy or consistency alone does not reveal whether a judge changes its verdict for the right reason. We propose State-Conditional Invariance (SCI), a reliability principle requiring a judge to remain stable under semantics-preserving changes in presentation while responding to causal interventions that change the task outcome. We instantiate SCI in GridWM-Judge, a simulator-grounded diagnostic benchmark built from deterministic MiniGrid trajectories with aligned Full, evidence-removal, and counterfactual variants, plus presentation and representation probes. Across five diagnostic tasks, GridWM-Judge tests local perception, action-conditioned transitions, presentation stability, representation exchangeability, and Full-CF causal discrimination. Our diagnostic evaluations reveal heterogeneous failure modes, including
    
[^55]: 国际财务报告准则（IFRS）合规性验证：开放的多大语言模型基准与内容条件化集成分类器

    IFRS Compliance Verification: An Open Multi-LLM Benchmark and Content-Conditioned Ensemble Classifier

    [https://arxiv.org/abs/2611.13595](https://arxiv.org/abs/2611.13595)

    本文构建了一个包含19,998个IFRS监管条款-披露配对、结合多LLM标签与8,837条人工标注的开放基准，并提出内容条件化的集成分类器框架，以克服单一模型准确率不足及简单集成中各模型系统性偏差导致性能下降的问题。

    

    根据国际财务报告准则（IFRS）验证企业财务披露是一项核心审计任务，专家需要将披露内容与数百条细粒度的监管条款进行交叉比对：这一过程费力、成本高昂且容易出错，因此人们越来越有兴趣将大语言模型（LLM）用作合规助手。然而，单一模型的准确率对于高风险应用场景仍然不够，而简单的集成方法容易受到各模型系统性偏差的影响，可能导致总体准确率甚至低于最强的个体标注者。本研究通过一个多阶段框架来填补这一空白，该框架由一个新构建的基准数据集支持，该数据集包含19,998个IFRS监管条款-披露配对，其中包括11,161个多LLM一致同意的标签，以及在LLM意见不一致的条目上的8,837个人工标注。该框架结合了三阶段领域自适应的句子嵌入流水线（在SEC EDGAR文件上进行BERT掩码语言模型预训练、无监督……）※注：摘要原文在此处被截断。

    arXiv:2611.13595v1 Announce Type: new  Abstract: Verifying corporate financial disclosures against the International Financial Reporting Standards (IFRS) is a core auditing task in which experts cross-reference disclosures with hundreds of fine-grained regulatory paragraphs: a laborious, costly, and error-prone process that motivates growing interest in large language models (LLMs) as compliance assistants. Yet single-model accuracy remains insufficient for high-stakes use, and naive ensembling is vulnerable to systematic per-model biases that can drag aggregate accuracy below that of the strongest individual annotator. This gap is addressed with a multi-stage framework, supported by a new benchmark of 19,998 IFRS regulation-disclosure pairs comprising 11,161 unanimous multi-LLM labels and 8,837 human annotations on items where the LLMs disagreed. The framework couples a three-stage domain-adapted sentence embedding pipeline (BERT masked language modeling on SEC EDGAR filings, unsuperv
    
[^56]: POEM：面向长上下文预填充的输出相对分块稀疏注意力

    POEM: Output-Relative Block-Sparse Attention for Long-Context Prefilling

    [https://arxiv.org/abs/2611.13578](https://arxiv.org/abs/2611.13578)

    提出了一种推理时防御方法TERM，利用提示校准的注意力熵偏差来自适应调制模型自身的残差更新，无需预定义表示目标即可抵御越狱攻击并减少过度拒绝。

    

    大型语言模型（LLMs）已在各类应用中展现出强大的能力，因此安全性与实用性对于可靠部署而言都不可或缺。越狱攻击利用多种形式的对抗性提示绕过对齐防护，诱导LLMs生成有害响应。一些基于表示的防御方法通过预定义的引导方向或学习到的安全区域来调控内部激活。然而，如何在不预设表示目标的情况下针对每个输入自适应地施加干预，同时限制过度拒绝，仍然是一个挑战。本文提出无目标熵引导残差调制（TERM），这是一种推理时防御方法，利用经提示校准的注意力熵偏差来引导对模型自身残差更新的自适应调制。注意力模式在不同提示与解码步骤之间各不相同，使得原始熵值难以被解读为用于干预的一致性信号。我们因此……

    arXiv:2611.13578v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown strong capabilities across diverse applications, making both safety and utility essential for reliable deployment. Jailbreak attacks use diverse forms of adversarial prompting to bypass alignment safeguards and induce LLMs to generate harmful responses. Some representation-based defenses regulate internal activations using predefined steering directions or learned safety regions. However, adapting intervention to each input without a predefined representational target while limiting over-refusal remains a challenge. In this paper, we propose Target-free Entropy-guided Residual Modulation (TERM), an inference-time defense that uses prompt-calibrated attention entropy deviations to guide adaptive modulation of the model's own residual updates. Attention patterns vary across prompts and decoding steps, making raw entropy values difficult to interpret as a consistent signal for intervention. We there
    
[^57]: TERM：通过熵引导残差调制的无目标越狱防御

    TERM: Target-Free Jailbreak Defense via Entropy-Guided Residual Modulation

    [https://arxiv.org/abs/2611.13577](https://arxiv.org/abs/2611.13577)

    提出了TERM这一推理时防御方法，利用提示校准的注意力熵偏差自适应地调制模型自身的残差更新，在无需预定义表示目标的情况下防御越狱攻击，同时限制过度拒答。

    

    大语言模型（LLM）在多种应用中展现出强大的能力，使得安全性与实用性对于可靠部署而言都至关重要。越狱攻击利用各种形式的对抗性提示绕过对齐防护机制，诱使大语言模型生成有害响应。一些基于表示的防御方法使用预定义的引导方向或学习到的安全区域来调节内部激活。然而，在缺乏预定义表示目标的情况下，如何针对每个输入自适应地施加干预，同时限制过度拒答，仍然是一个挑战。在本文中，我们提出了无目标熵引导残差调制（TERM），这是一种推理时防御方法，利用提示校准的注意力熵偏差来引导对模型自身残差更新的自适应调制。由于注意力模式在不同的提示和解码步骤之间会有所变化，原始熵值难以被解释为一致的干预信号……

    arXiv:2611.13577v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown strong capabilities across diverse applications, making both safety and utility essential for reliable deployment. Jailbreak attacks use diverse forms of adversarial prompting to bypass alignment safeguards and induce LLMs to generate harmful responses. Some representation-based defenses regulate internal activations using predefined steering directions or learned safety regions. However, adapting intervention to each input without a predefined representational target while limiting over-refusal remains a challenge. In this paper, we propose Target-free Entropy-guided Residual Modulation (TERM), an inference-time defense that uses prompt-calibrated attention entropy deviations to guide adaptive modulation of the model's own residual updates. Attention patterns vary across prompts and decoding steps, making raw entropy values difficult to interpret as a consistent signal for intervention. We there
    
[^58]: 成对合并分词在二进制字符串上难以近似，且在一元数据集上是强NP难的

    Pair-Merge Tokenisation Is Hard to Approximate on Binary Strings and Strongly NP-Hard on Unary Datasets

    [https://arxiv.org/abs/2611.13568](https://arxiv.org/abs/2611.13568)

    本文通过将顶点覆盖间隙转化为文法大小间隙的归约技术，证明了BPE式成对合并分词优化问题即使仅在单个二进制字符串上也难以近似（除非P=NP，近似因子不超过c·n/log n），且在一元字母表数据集上是强NP难的。

    

    诸如字节对编码（BPE）之类的分词器通过反复将相邻的一对符号合并成新符号来缩短文本；更少的标记意味着更少的计算量。我们提出这样一个问题：是否存在一种高效算法，能够选择 k 次合并，使得长度为 n 的字符串尽可能接近最短；即使什么都不做，结果也已经在因子 n 之内。我们证明，除非 P = NP，对于某个常数 c > 0，任何多项式时间算法都无法在因子 c·n/log n 之内近似该问题——即便输入只是单个二进制字符串，且合并是替换所有出现位置的BPE式合并。其核心思想是：k 次合并构成一个带 k 条规则的文法，于是顶点覆盖的间隙可以转化为文法大小的间隙，我们通过重新标记的副本将其放大，用二进制编码，并通过显式的合并操作加以实现。对于允许跳过某些出现位置的合并方式，一个简单算法可以达到 O(n/log log n) 的近似，并且除非 coNP ⊆ NP/poly，否则任何 Karp 归约都无法证明超出 n/log n 量级的难度。在单字母（一元）字母表上……（摘要内容截断于此）

    arXiv:2611.13568v1 Announce Type: cross  Abstract: Tokenisers such as Byte-Pair Encoding (BPE) shorten text by repeatedly merging an adjacent pair of symbols into a new one; fewer tokens mean less computation. We ask whether an efficient algorithm can choose $k$ merges that leave a string of length $n$ almost as short as possible; doing nothing is already within a factor $n$. We prove that, unless P = NP, no polynomial-time algorithm comes within a factor $c\,n/\log n$, for some constant $c>0$, already for one binary string and for BPE-style merges that replace every occurrence. The key idea is that $k$ merges form a grammar with $k$ rules: a vertex-cover gap becomes a grammar-size gap, which we amplify with relabelled copies, encode in binary and realise with explicit merges. For merges that may skip occurrences, a simple algorithm achieves $O(n/\log\log n)$, and no Karp reduction can prove hardness beyond order $n/\log n$ unless coNP $\subseteq$ NP/poly. Over a one-letter alphabet, b
    
[^59]: 一种基于概念的可解释框架：通过无需概念监督的词元-概念注意力实现临床文本预测

    A Concept-Based Interpretable Framework for Clinical Text-based Prediction via Token-Concept Attention without Concept Supervision

    [https://arxiv.org/abs/2611.13564](https://arxiv.org/abs/2611.13564)

    该论文提出了一种无需概念标注监督的可解释框架，通过定制交叉注意力瓶颈机制，利用预训练语言模型估计临床文本词元与医学概念之间的相关性，从而在临床预测任务中实现基于概念的黑盒模型解释。

    

    目的：基于概念的解释已成为解释黑盒模型的一种有前景的方法，它利用人类可理解的高层概念来解释模型。然而，大多数现有方法需要带标注的概念标签来训练模型识别输入中的概念，而在临床文本中获取此类标注往往成本高昂。本研究旨在使语言模型在临床预测任务中无需概念监督即可实现基于概念的解释。方法：我们提出了一种新颖的框架，通过定制的交叉注意力瓶颈，在无需概念监督的情况下实现基于概念的解释。临床文本和医学概念的文本描述首先由预训练语言模型进行分词和嵌入。然后，交叉注意力模块为每个文本词元估计每个概念的存在情况，生成概念相关性分数，以捕捉概念在整个临床记录中的参与方式。这些分数……（原文摘要在此处截断）

    arXiv:2611.13564v1 Announce Type: cross  Abstract: Objective: Concept-based interpretation has emerged as a promising approach for explaining blackbox models using human-understandable, high-level concepts. However, most existing methods require annotated concept labels to train the model to identify concepts in the input, and obtaining such annotations in clinical text is often expensive. This study aims to enable concept-based interpretation for language models without concept supervision on clinical prediction tasks. Methods: We propose a novel framework that enables concept-based interpretation without concept supervision through a tailored cross-attention bottleneck. Clinical text and the text descriptions of medical concepts are first tokenized and embedded by a pretrained language model. A cross-attention module then estimates the presence of each concept for each text token, producing concept relevance scores that capture how concepts are involved throughout the note. These sco
    
[^60]: 大语言模型临床推理的证据可追溯基准

    An evidence-traceable benchmark of clinical reasoning in large language models

    [https://arxiv.org/abs/2611.13562](https://arxiv.org/abs/2611.13562)

    该研究构建了首个证据可追溯的医学大语言模型基准Med Evidence Bench（MEB），将每条评分标准与版本化的文献原文片段绑定，揭示出十个主流大语言模型虽在指南知识上表现优异，但在真实世界临床场景中的推理能力显著下降。

    

    当评分标准无法追溯到临床证据时，医学大语言模型（LLM）的基准测试分数便难以解读。本研究开发了Med Evidence Bench（MEB；https://www.medevidencebench.org.cn/），包含1,536个由医生构建的测试项目，涵盖三个层次：指南知识、标准化临床场景以及真实世界检验（实验室）场景。每条陈述医学论断的评分标准均关联到一个带版本编号的原文文本片段。在十个大语言模型上，证据可追溯分数（Evidence Trace Score, ETS）在指南知识层面为0.975–0.997，在标准化场景层面为0.292–0.658，在真实世界场景层面为0.182–0.447，且各层次间的模型排名存在差异。对于真实世界场景，识别临床信息的得分（平均0.770）高于匹配证据（0.165）或提出建议（0.160）的得分。在1,296条标准层面的评判中，自动评分与三位医生的多数投票结果一致（宏——[原文摘要至此截断]

    arXiv:2611.13562v1 Announce Type: new  Abstract: Benchmark scores for medical large language models (LLMs) are difficult to interpret when scoring criteria cannot be traced to clinical evidence. Here we developed Med Evidence Bench (MEB; https://www.medevidencebench.org.cn/), comprising 1,536 physician-constructed items across three levels: guideline knowledge, standardized clinical scenarios and real-world laboratory scenarios. Each criterion stating a medical claim is linked to a versioned source-text span. Across ten LLMs, the Evidence Trace Score (ETS) was 0.975-0.997 for guideline knowledge, 0.292-0.658 for standardized scenarios and 0.182-0.447 for real-world scenarios, and rankings varied across levels. For real-world scenarios, scores for identifying clinical information (mean, 0.770) exceeded those for matching evidence (0.165) or formulating recommendations (0.160). On 1,296 criterion-level judgments, automated grading agreed with the majority vote of three physicians (macro-
    
[^61]: 密集奖励在GRPO中值得付出那么多token开销吗？

    Are Dense Rewards Worth Their Tokens in GRPO?

    [https://arxiv.org/abs/2611.13560](https://arxiv.org/abs/2611.13560)

    尽管蒙特卡洛密集过程奖励在单个更新比较中优于标准GRPO，但在以生成token数和GPU小时数衡量的计算量公平匹配下，其高昂的采样开销（占生成token的91%）使结果排名发生逆转，密集奖励并不值得其token代价。

    

    RLVR（可验证奖励的强化学习）通过奖励正确的最终答案来训练语言模型进行推理。广泛使用的算法——组相对策略优化（GRPO）为解答中的每个token赋予相同的信用，因此有研究提出密集过程奖励来为中间步骤分配信用，例如使用蒙特卡洛奖励，通过从每个步骤采样后续内容来为该步骤评分。然而，在每个训练更新中将此类奖励与GRPO进行比较，或在单个批次上用梯度测试加以检验，都忽略了其成本：在我们的设置中，仅后续内容采样就消耗了全部生成token的91%。我们引入了一种计算匹配协议，为每种训练干预分配相同的预算（以生成的token数和训练GPU小时数衡量），并在带有副作用对照的预注册研究中进行测试。在六个模型上的1,668次运行中，计算匹配逆转了排名。在Qwen2.5-1.5B上，蒙特卡洛奖励在每个更新中领先于标准GRPO，但最终却落后14.3个百分点（摘要原文在此处截断）。

    arXiv:2611.13560v1 Announce Type: cross  Abstract: RLVR trains language models to reason by rewarding correct final answers. A widely used algorithm, group-relative policy optimization (GRPO), gives every token of a solution the same credit, so dense process rewards have been proposed to credit intermediate steps, for example with a Monte-Carlo reward that scores each step by sampling continuations from it. Comparing such a reward with GRPO per training update, or vetting it with a gradient test on a single batch, leaves out what it costs: in our setting, the continuations alone take 91% of all generated tokens. We introduce a matched-compute protocol that gives every training intervention the same budget, measured in generated tokens and training GPU-hours, and tests it in preregistered studies with controls for side effects. Across 1,668 runs on six models, matching compute reverses the ranking. On Qwen2.5-1.5B, the Monte-Carlo reward leads standard GRPO per update, yet ends 14.3 per
    
[^62]: Darmok：通过约束收紧与弹性编排实现冠军级密码填字游戏求解

    Darmok: Championship Cryptic Crossword Solving via Constraint Tightening and Resilient Orchestration

    [https://arxiv.org/abs/2611.13557](https://arxiv.org/abs/2611.13557)

    Darmok 是首个在英国密码填字游戏上达到超人水平的自动化系统，通过约束收紧与弹性编排技术，在《泰晤士报》填字游戏锦标赛中无错误解出全部七道谜题，且每一轮速度均超过所有人类选手。

    

    我们提出了 Darmok，这是首个在英国密码填字游戏上达到超人表现的自动化系统。与美式填字游戏不同，密码谜题在具有误导性的表面文字之下隐藏着一个未标记的定义以及组合式的文字游戏；稀疏的网格使得每个单词有近一半的字母未被交叉校验；而且锦标赛规则规定，即使只包含一个错误或空白格子的答案，无论速度多快，其排名都会排在所有无错误答案之后。在第55届于伦敦举办的《泰晤士报》填字游戏锦标赛中，Darmok 与人类精英选手同场竞技，无错误地解出了全部七道锦标赛谜题，并在每一轮中的完成速度都快于所有人类参赛者。它以6分52秒解出决赛谜题，比人类冠军快了整整3分13秒。当今的前沿AI模型平均而言可以以相当的准确性和速度在单次尝试中解出整个谜题。然而，云端API调用仍可能停滞、超时、触发安全拒绝，或陷入推理困境（摘要在此处截断）。

    arXiv:2611.13557v1 Announce Type: new  Abstract: We present Darmok, the first automated system to reach superhuman performance on British cryptic crosswords. Unlike American-style crosswords, cryptic clues disguise an unmarked definition and compositional wordplay beneath a misleading surface reading, sparse grids leave nearly half of each word's letters unchecked, and tournament rules rank a grid with even a single wrong or blank square behind all error-free solutions regardless of speed. Competing alongside an elite human field at the 55th Times Crossword Championship in London, Darmok solved all seven championship puzzles error-free and finished faster than every human competitor in every round. It solved the Final puzzle in 6m 52s, a full 3m 13s ahead of the human champion.   Today's frontier AI models can on average solve a whole puzzle in a single try with comparable accuracy and speed. However, cloud API calls still stall, time out, trigger safety refusals, or get stuck in reaso
    
[^63]: 后训练如何影响语言模型中的言语化算法计算

    How Post-Training Influences Verbalized Algorithmic Computation in Language Models

    [https://arxiv.org/abs/2611.13552](https://arxiv.org/abs/2611.13552)

    后训练主要改变的是言语化推理的有用性而非内部信息的编码方式——SFT 和 RLVR 后的模型内部均可读取进位信息，但 RLVR 产生的思维链在各检查点上显著提升了推理准确率。

    

    后训练提升了语言模型的推理能力，但它如何改变内部信息的使用以及言语化思维链（CoT）的作用仍不清楚。我们研究了 Olmo-3-7B-Instruct 在多位数加法任务上经过 SFT、DPO 和 RLVR 相继训练后的各个检查点，并将进位值作为已知的中间状态。探测方法在 SFT 和 RLVR 模型中都能以相似的准确率从内部激活中恢复进位信息，尽管操纵这些信息并不能可靠地控制模型输出。相比之下，言语化推理的有用性存在差异：RLVR 产生的思维链相比 SFT 或 DPO 产生的思维链，在各接收检查点上带来更高的准确率。我们还发现，RLVR 之后与进位相关的陈述比匹配的随机对照获得更高的重要性分数。DPO 对进位相关陈述表现出类似的敏感性，但没有相应的准确率提升。这些结果表明，后训练改变了书面推理的有用性……

    arXiv:2611.13552v1 Announce Type: new  Abstract: Post-training improves language-model reasoning, but how it changes the use of internal information and verbalized chain of thought (CoT) remains unclear. We study successive SFT, DPO, and RLVR checkpoints of Olmo-3-7B-Instruct on multi-digit addition, using carry values as known intermediate state. Probes recover carry information from internal activations with similar accuracy in SFT and RLVR, although manipulating this information does not reliably control model outputs. In contrast, the usefulness of verbalized reasoning differs: CoTs produced by RLVR yield higher accuracy across receiving checkpoints than those produced by SFT or DPO. We also show that carry-related statements receive higher importance scores after RLVR than matched random controls. DPO shows similar sensitivity to carry-related statements but without comparable accuracy gains. These results suggest that post-training changes the usefulness of written reasoning and 
    
[^64]: 注意力手术的合理性检查：面向表征驱动设计的零模型与匹配干预

    Sanity Checks for Attention Surgery: Null Models and Matched Interventions for Representation-Motivated Design

    [https://arxiv.org/abs/2611.13543](https://arxiv.org/abs/2611.13543)

    该论文提出三个可复用的合理性检查（噪声底线、几何零模型、匹配干预）来严格检验基于表征统计的架构修改，并发现自注意力移除中的头-值对齐很大程度上可由早期位置的因果零模型解释，而非模型真正依赖的特定表征方向。

    

    Transformer架构修改通常的做法是：在内部表征上测量某个统计量，观察到它数值很大，于是从架构上将其移除，并报告性能提升。要解释这一过程，需要评估三个问题：该统计量是否超过了噪声底线、它能否在几何零假设下依然成立、以及训练收益是否确实依赖于被移除的那个特定方向。我们将这些要求形式化为三个可复用的检查，并将其应用于“自注意力移除”方法——即删除注意力头输出中与其自身值向量对齐的分量。在九个多头预训练模型（共5,408个注意力头，参数量从1.24亿到69亿）上的实验表明，较早位置的因果零模型可以解释头-值余弦对齐的48%到63%，且不存在随规模单调变化的趋势。在分组查询注意力下，这种对齐是每个头所属键值组特有的，并在不同组之间发生反转（组内为+0.24至+0.27，组间为-0.11至-0.19……）

    arXiv:2611.13543v1 Announce Type: cross  Abstract: Transformer architectural modifications frequently measure a statistic on internal representations, observe that it is large, remove it architecturally, and report an improvement. Interpreting that sequence requires evaluating whether the statistic exceeds its noise floor, whether it survives a geometric null, and whether the training gain requires the specific direction removed. We formulate these requirements as three reusable checks and apply them to self-attention removal, which deletes the component of an attention head's output aligned with its own value vector. Across nine multi-head pretrained models (5,408 heads, 124M to 6.9B parameters), an earlier-position causal null accounts for 48% to 63% of the head-value cosine alignment, with no monotone scale trend. Under grouped-query attention, this alignment is specific to each head's own key-value group and reverses across groups (+0.24 to +0.27 within-group, -0.11 to -0.19 across
    
[^65]: 大语言模型中的框架诱导脆弱性：来自中文新闻标题的证据

    Framing-Induced Vulnerability in Large Language Models: Evidence from Chinese News Headlines

    [https://arxiv.org/abs/2611.13531](https://arxiv.org/abs/2611.13531)

    该研究通过6万多条中文新闻标题审计和跨四个中文模型的受控实验，首次系统揭示了大型语言模型对新闻标题的判断主要受标题框架而非事实本身的影响——情绪化框架会显著抬高事件严重性评分，而提供完整文章正文可降低54%至80%的此类偏差。

    

    大语言模型越来越多地承担评估、路由和审核新闻的任务，因此模型对一条新闻标题的判断已成为一种“守门人”决策。我们探究这种判断究竟是依据所报道的事实本身，还是依据事实被呈现的框架方式。我们将对63,926条中文新闻标题的生态学审计与一项受控实验相结合。实验涵盖30个真实事件、30个合成事件和29对自然标题，并在四个中文模型上进行了测试。仅改变标题的框架就会改变模型感知到的误导性和事件严重程度，且变化的方向和排序在DeepSeek、MiMo、Dots和LongCat四个模型上保持一致。情绪强化型标题使严重程度评分提高了0.65至0.88分，而正面重构型标题使其降低了0.34至0.52分。提供文章正文可使这种严重性偏差降低54%至80%，且四项降低均在p < .01水平上显著。一项小型人类参照研究……（摘要原文在此处被截断）

    arXiv:2611.13531v1 Announce Type: new  Abstract: Large language models increasingly evaluate, route, and moderate news, so a model's judgment of a headline has become a gatekeeping decision. We ask whether that judgment follows the fact being reported or the way the fact is framed. We combine an ecological audit of 63,926 Chinese news headlines with a controlled experiment. The experiment covers 30 real events, 30 synthetic events, and 29 natural headline pairs, tested on four Chinese models. Changing only the headline frame shifted both perceived misleadingness and event severity, and the direction and ordering of the shifts were the same for DeepSeek, MiMo, Dots, and LongCat. Emotionally intensified headlines raised severity ratings by 0.65 to 0.88 points, whereas positively reframed headlines lowered them by 0.34 to 0.52 points. Supplying the article body reduced this severity bias by 54 to 80 percent, and all four reductions were significant at p < .01. A small human reference stud
    
[^66]: EJAI：面向数据中心环境正义的保留判断力公民AI

    EJAI: Judgment-Preserving Civic AI for Data Center Environmental Justice

    [https://arxiv.org/abs/2611.13525](https://arxiv.org/abs/2611.13525)

    本文提出EJAI，一个面向数据中心环境正义争议的基于证据来源的公民AI系统，通过交互式地图、证据检索、缺失数据声明和双重利益相关者角色帮助居民审视证据并自主形成政策判断，避免了传统仪表盘和单智能体AI在用户检视证据前就给出结论的问题，并在基准测试中实现了98.4%的事实声明证据支持率。

    

    美国乃至世界各地的社区正在抵制数据中心——人工智能的物理基础设施。这些争议涉及环境正义问题：居民必须依据分散在各类指标、企业声明、政策文件和公众证词中的证据，来权衡分布不均的负担。一项针对10名用户的前期形成性研究表明，仪表盘和单智能体AI可能会在人们审视证据之前就给出结论。本文提出了EJAI，一个基于证据来源的系统，它将交互式地图、检索到的证据、缺失数据声明以及两个利益相关者角色相结合，使人们能够审视证据并形成自己的政策建议。在包含21个查询的小型功能性基准测试中，检索到的证据支持了98.4%的事实性声明，且系统在33个案例中的32个里明确承认了证据缺失。角色对齐带来了依据性方面的代价：企业高管角色省略了一些与其立场相冲突的证据……（原文此处截断）

    arXiv:2611.13525v1 Announce Type: cross  Abstract: Communities in the United States and around the world are contesting data centers, the physical infrastructure of AI. These disputes pose environmental justice issues: residents must weigh unevenly distributed burdens using evidence scattered across metrics, corporate claims, policy documents, and public testimony. A formative study with 10 users showed that dashboards and single-agent AI can settle questions before people examine that evidence. Here we present EJAI, a source-grounded system that links an interactive map, retrieved evidence, missing-data statements, and two stakeholder personas that allow people to inspect evidence and form their own policy recommendations. In a small functional benchmark of 21 queries, retrieved evidence supported 98.4% of factual claims and the system acknowledged missing evidence in 32 of 33 cases. Persona alignment carried a grounding cost: the Corporate Executive omitted some evidence that conflic
    
[^67]: 面向惯性里程计的可微分滤波

    Differentiable Filtering for Inertial Odometry

    [https://arxiv.org/abs/2611.13517](https://arxiv.org/abs/2611.13517)

    本文提出DFIO框架，将误差状态卡尔曼滤波器的测量更新阶段嵌入计算图实现端到端可微分训练，联合优化位移与协方差输出头，使行人惯性里程计的平均绝对平移误差和一秒相对平移误差分别降低13.2%和39.6%。

    

    精确的惯性里程计对自主系统至关重要，但由于传感器噪声和偏置的积分会导致无界漂移。在TLIO等现有方法中，神经伪测量网络是独立于下游滤波器进行训练的。本文提出了DFIO（面向惯性里程计的可微分滤波），这是一种学习框架，将误差状态卡尔曼滤波器的更新阶段直接嵌入到计算图中。在完成独立预训练后，DFIO通过可微分测量更新，利用后验位置监督联合优化位移和协方差输出头。该系统在包含标定与未标定惯性测量数据的公开行人运动数据集上进行了验证。行人数据集上的实验结果表明，相对于TLIO基线，DFIO将平均绝对平移误差和一秒相对平移误差分别降低了13.2%和39.6%。

    arXiv:2611.13517v1 Announce Type: cross  Abstract: Accurate inertial odometry is critical for autonomous systems but suffers from unbounded drift caused by the integration of sensor noise and biases. In approaches such as TLIO, neural pseudo-measurements are trained independently of their downstream filter. This paper introduces Differentiable Filtering for Inertial Odometry (DFIO), a learning framework that embeds the update stages of an error-state Kalman filter directly into a computational graph. Following standalone pre-training, DFIO jointly optimizes the displacement and covariance output heads using posterior-position supervision through differentiable measurement updates. The proposed system is validated on public pedestrian motion datasets featuring both calibrated and uncalibrated inertial measurements. Experimental results on a pedestrian dataset demonstrate that DFIO reduces mean absolute and one-second relative translation errors by 13.2% and 39.6% relative to the TLIO ba
    
[^68]: 移动AI工厂：为您的移动设备构建专用语言模型

    Mobile AI Factory: Building Specialized Language Models for Your Mobile Device

    [https://arxiv.org/abs/2611.13509](https://arxiv.org/abs/2611.13509)

    Mobile AI Factory是一个智能体系统，能够自动为目标任务和目标手机构建经过微调与量化的专用小型语言模型，通过编排器LLM自主决策超参数调优、真实数据挖掘或合成数据生成，并借助目标手机的真实测量数据实现高性能端侧部署。

    

    在移动设备上运行的小型语言模型（SLM）可以保护用户数据隐私，并在无网络连接的情况下工作。但只有在针对目标任务进行微调并针对目标设备进行尺寸适配时，它们才能表现良好，而这一过程目前需要专家反复调优。我们提出了Mobile AI Factory，这是一个智能体系统，可以为给定任务和手机构建经过微调和量化的小型语言模型。给定任务规范和目标手机后，一个编排器LLM会选择是调整超参数、挖掘真实数据还是生成合成样本，同时固定规则会回滚损害准确性的更改。该工厂从小到大依次搜索模型，涵盖不同的基础模型和量化方案，并在搜索过程中自适应调整准确性目标，同时使用目标手机在不同推理引擎和后端上的真实测量数据。我们在十个任务和四部Android手机上评估了Mobile AI Factory。在未见过的测试集上，其端侧模型优于两个五样本提示的前端模型（摘要至此被截断）。

    arXiv:2611.13509v1 Announce Type: new  Abstract: Small language models (SLMs) running on mobile devices can keep user data private and work without connectivity. But they perform well only when fine-tuned for the target task and sized for the target device, a process that today requires repeated expert tuning. We present {\it Mobile AI Factory}, an agentic system that builds a fine-tuned, quantized SLM for a given task and phone. Given a task specification and a target phone, an orchestrator LLM chooses whether to tune hyperparameters, mine real data, or generate synthetic examples, while fixed rules roll back changes that hurt accuracy. The factory searches from small to large models across base models and quantizations, adapts its accuracy goal as it searches, and uses real measurements of the target phone across inference engines and backends. We evaluate Mobile AI Factory on ten tasks and four Android phones. On an unseen testset, its on-device models outperform two five-shot front
    
[^69]: 评估Transformer中的组合泛化：组合等价性与模块覆盖度的作用

    Evaluating Compositional Generalization in Transformers: The Role of Composition Equivalence and Module Coverage

    [https://arxiv.org/abs/2611.13498](https://arxiv.org/abs/2611.13498)

    本文通过系统性评估发现，训练-测试分布中的组合等价性与模块覆盖度是决定 transformer 组合泛化性能的两个关键因素，直接模型的优异表现主要来自对组合等价性的利用，一旦消除这些等价性，性能将显著下降。

    

    组合泛化是一种重要的分布外泛化形式，其定义为：在若干模块组合（例如 sort、filter 等函数）上进行训练后，能够泛化到未见过的组合的能力。尽管已有大量实证工作评估了 transformer 的组合泛化能力，但由于评估往往只在有限的、以临时方式定义的训练-测试划分上进行，研究结果并不一致。通过系统性评估，我们识别出训练-测试分布中强烈影响性能的两个关键方面：组合等价性和模块覆盖度。我们表明，直接模型（仅在最终输出上训练）所展现的性能主要源于对组合等价性的利用，即不同的模块序列可以归约为相同的端到端函数。当基准测试消除这些等价性时，模型性能会显著下降。

    arXiv:2611.13498v1 Announce Type: cross  Abstract: Compositional generalization is an important form of out-of-distribution generalization, defined as the ability to train on some combinations of modules (e.g., functions such as sort, filter) and then generalize to unseen combinations. Despite substantial empirical work evaluating compositional generalization in transformers, findings have been inconsistent because evaluations were often conducted on a limited set of train-test splits, defined in an ad hoc manner. Through a systematic evaluation, we identify two key aspects of the train-test distribution that strongly affect performance: composition equivalence and module coverage. We show that the apparent performance of direct models (trained only on final outputs) results primarily from exploiting composition equivalences, meaning different sequences of modules that reduce to identical end-to-end functions. When benchmarks eliminate these equivalences, performance drops significantl
    
[^70]: 利用心理物理学注意力分布进行注视增强的奖励建模

    Leveraging Psychophysical Attentional Distribution for Gaze-Augmented Reward Modeling

    [https://arxiv.org/abs/2611.13477](https://arxiv.org/abs/2611.13477)

    该论文基于认知心理学中人类字母识别的发现，将视觉注意力建模为以注视点为中心的空间梯度分布并融入RLHF奖励模型训练，从而实现了比基线更高的偏好预测准确率。

    

    将人类反馈整合到语言模型中作为使模型输出与人类偏好对齐的手段，正受到越来越多的关注，其中基于人类反馈的强化学习（RLHF）是基于偏好的对齐的重要框架。最近的研究将眼动追踪（ET）数据作为奖励模型训练的额外监督信号，将奖励学习建立在人类阅读行为的基础上。在这一方法的基础上，我们利用认知心理学中关于人类字母识别的既有发现，对以往以注视为中心的模型进行了扩展。具体而言，我们将视觉注意力建模为以当前注视位置为中心的空间梯度分布，并将这一表示融入奖励模型训练中。我们的结果表明，使用所提出的注视分布训练的奖励模型比基线模型取得了更高的偏好预测准确率。此外，我们的一些结果……

    arXiv:2611.13477v1 Announce Type: cross  Abstract: Integrating human feedback into language models has gained increasing attention as a means to align model outputs with human preferences, with Reinforcement Learning from Human Feedback (RLHF) serving as a prominent framework for preference-based alignment. Recent studies have incorporated eye-tracking (ET) data as an additional supervisory signal for reward model training, grounding reward learning in human reading behavior. Building on this approach, we extend prior fixation-centric models by leveraging established findings from cognitive psychology on human letter recognition. Specifically, we model visual attention as a spatially graded distribution centered on the current gaze location and incorporate this representation into reward model training. Our results show that the reward model trained with the proposed gaze distribution achieves higher preference prediction accuracy than a baseline model. Moreover, some results from our 
    
[^71]: 继承的权重：测量全球南方语言模型的分词溢价与主权成本

    Inherited Weights: Measuring the Tokenization Premium and the Sovereignty Cost of Language Models for the Global South

    [https://arxiv.org/abs/2611.13441](https://arxiv.org/abs/2611.13441)

    论文指出微调模型会从基础模型继承词表、知识、文化对齐、数据与许可证等决定性属性，并测量出全球南方语言承受最高达10倍的分词溢价，证明构建本地化词表可将该溢价显著降低约24%，从而揭示了大语言模型给全球南方带来的主权成本。

    

    在美国和中国之外部署的大多数语言模型，都是由少数预训练模型家族微调而来的后代。我们认为，微调使得模型的决定性属性——词表、知识、文化对齐、被记忆的数据以及许可证——都从其基础模型继承而来。我们的贡献包括：(i) 对分词溢价的更新且可复现的测量，涵盖31种语言和2023-2024年发布的7个分词器，基于《世界人权宣言》的27条条文，并对未知词元坍缩进行了显性校正。较旧分词器中阿姆哈拉语的溢价高达10倍，在最新的商业分词器中仍为6.8倍；南美洲原住民语言在全球分词器下需支付1.6倍至2.7倍的溢价。由葡萄牙语构建的词表，相对于所有被测试的全球分词器，可降低13种拉丁字母书写的全球南方语言中11种的溢价，其中瓜拉尼语和克丘亚语约降低24%。(ii) 一套包含八项测试的基准……（摘要原文在此处被截断）

    arXiv:2611.13441v1 Announce Type: cross  Abstract: Most language models deployed outside the United States and China are fine-tuned descendants of a small number of pretrained families. We argue that fine-tuning leaves the decisive properties of a model inherited from its base: vocabulary, knowledge, cultural alignment, memorized data and license. We contribute (i) an updated, reproducible measurement of the tokenization premium across 31 languages and 7 tokenizers released in 2023-2024, on 27 articles of the Universal Declaration of Human Rights, with explicit correction for unknown-token collapse. Premiums reach 10x for Amharic in older tokenizers and remain at 6.8x in the newest commercial one; Indigenous South American languages pay between 1.6x and 2.7x under global tokenizers. A vocabulary built from Portuguese lowers the premium for 11 of 13 Latin-script Global South languages relative to every global tokenizer tested, by about 24% for Guarani and Quechua. (ii) An eight-test bla
    
[^72]: 脱靶后验质量：一种用于微调口语语言日志化探针的免训练诊断方法

    Off-target posterior mass: A training-free diagnostic for fine-tuning spoken language diarization probes

    [https://arxiv.org/abs/2611.13430](https://arxiv.org/abs/2611.13430)

    本文提出一种免训练的诊断指标——脱靶后验质量（OTPM），基于后验语言识别概率预测微调预训练语音模型为口语语言日志化（SLD）带来的性能提升，其与探针增益呈正相关（合成双语数据上为0.56，真实语料上为0.23）。

    

    口语语言日志化（SLD）模型可提供语音录音的帧级语言信息。SLD建模流程包括从预训练语音模型中提取特征，随后进行聚类。微调预训练语音模型已被证明可以为SLD带来性能提升，但提升幅度取决于具体数据集。本文提出了基于后验语言识别概率计算的脱靶后验质量（OTPM），作为一种诊断指标，用于预测SLD微调所带来的性能提升。OTPM与探针增益呈正相关，在未见过的合成双语数据上相关系数为0.56，在真实世界语料库上为0.23。这些结果表明OTPM作为SLD诊断指标具有广阔的应用前景。

    arXiv:2611.13430v1 Announce Type: cross  Abstract: Spoken Language Diarization (SLD) models provide frame-level language information of a speech recording. SLD modeling pipeline includes feature extraction from pre-trained speech models followed by clustering. Fine-tuning pre-trained speech models has shown performance gains for SLD with its magnitude being dataset dependent. Off-target Posterior Mass (OTPM), computed based on posterior language identification probabilities is proposed as a diagnostic metric to forecast performance gains from SLD fine-tuning. OTPM obtains positive correlation with probe gains, 0.56 on unseen synthetic bilingual data and 0.23 on real-world corpora. These results indicate promising potential of OTPM as a diagnostic metric for SLD.
    
[^73]: BioMem：面向生物医学大语言模型智能体的智能体记忆

    BioMem: Agentic Memory for Biomedical LLM Agents

    [https://arxiv.org/abs/2611.13426](https://arxiv.org/abs/2611.13426)

    本研究提出BioMem，系统评估了生物医学领域中多种LLM智能体记忆方法在任务有效性、检索质量、信息保真度和长期鲁棒性方面的表现与挑战。

    

    大语言模型（LLMs）在生物医学应用中已展现出可观的潜力，而智能体记忆系统已成为一种在任务与交互之间保存和利用信息的有前景的方法。然而，一个根本性问题仍然存在：这些系统在多大程度上能够有效且可靠地保留、检索、更新和利用生物医学信息？在本研究中，我们系统地研究了生物医学领域的智能体记忆，重点关注任务有效性、检索质量、信息保真度和长期鲁棒性。我们在三种代表性场景（生物医学知识推理、临床预测和长期对话记忆）中，对四种LLM上的五种记忆方法进行了评估。我们的分析揭示了记忆构建、证据相关性和对话历史如何影响下游生物医学性能，并突出了检索相关信息所面临的独特挑战。

    arXiv:2611.13426v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated considerable potential in biomedical applications, while agentic memory systems have emerged as a promising approach to preserving and utilizing information across tasks and interactions. However, a fundamental question remains: how effectively and reliably do these systems retain, retrieve, update, and utilize biomedical information? In this study, we systematically investigate agentic memory in the biomedical domain, focusing on task effectiveness, retrieval quality, information fidelity, and long-term robustness. We evaluate five memory approaches across four LLMs in three representative settings: biomedical knowledge reasoning, clinical prediction, and long-term conversational memory. Our analyses reveal how memory construction, evidence relevance, and conversational history shape downstream biomedical performance, highlighting the distinct challenges of retrieving relevant information, 
    
[^74]: 免费午餐式的不确定性：基于粒子解码的序列级Rényi熵用于自适应LLM推理

    Free Lunch Uncertainty: Sequence-Level R\'enyi Entropy from Particle Decoding for Adaptive LLM Reasoning

    [https://arxiv.org/abs/2611.13414](https://arxiv.org/abs/2611.13414)

    该论文发现SMC幂采样解码可零边际成本获得序列级Rényi熵，其中1/2阶熵在13个设置中的10个中表现最佳，是用于LLM自适应推理的零成本选择性预测信号。

    

    量化LLM整个补全结果的不确定性对于选择性预测和自适应计算至关重要。零成本信号（如平均token熵和对数概率）仅局限于token级别或单个补全，而自洽性方法则需要许多额外的补全。我们证明，用于幂采样的序贯蒙特卡洛（SMC）解码本身已经在估计幂分布的归一化常数，这使我们能以零边际成本获得模型补全分布的序列级Rényi熵（SRE），且在每个粒子预算下都有保证。通过Rényi链式法则，我们将这种读出扩展到任意阶数的逐步熵族。我们证明高于1的阶数与每步最可能的token相关联，这解释了它们为何与对数概率保持一致，而1/2阶成员则累积每步搜索努力的对数。该成员在13个设置中的10个中是最佳的零成本选择性预测信号。在MATH5上

    arXiv:2611.13414v1 Announce Type: cross  Abstract: Quantifying uncertainty over an entire LLM completion is critical for selective prediction and adaptive compute. Zero-cost signals such as mean token entropy and log-probability are local to tokens or to one completion, and self-consistency needs many extra completions. We show that Sequential Monte Carlo (SMC) decoding for power sampling already estimates the normalizing constant of the power distribution, which gives the sequence-level R'enyi entropy (SRE) of the model's completion distribution at zero marginal cost, with a guarantee at every particle budget. A R'enyi chain rule extends this readout to a family of stepwise entropies of any order. We prove that orders above one are tied to the most likely token of each step, which explains why they track log-probability, while the order-1/2 member accumulates the log of per-step search effort. This member is the best zero-cost selective-prediction signal in 10 of 13 settings. On MATH5
    
[^75]: 两个尺度上的正字法深度：英语还进行了词汇量匹配

    Orthographic depth at two scales, with English also matched for lexicon size

    [https://arxiv.org/abs/2611.13400](https://arxiv.org/abs/2611.13400)

    该论文提出在两个粒度（对齐单元与音节）上用条件熵度量正字法深度，对十二种字母文字的分析发现英语不确定性最大、芬兰语最小，且排序结果对英语不同转写方式保持稳健。

    

    一个读者可能只在自己的某一种语言中遇到阅读困难，一个被提出的解释是拼写映射到语音的可预测程度。我们用条件熵来度量这种可预测性，即看到拼写后残留的发音不确定性，并在两个粒度（单元大小）上进行测量：对齐单元（对齐的拼写-语音片段）和音节（发音上有边界的单位）。我们分析了十二种字母文字系统的词表，并完整报告其中五种，其中英语的发音来自词典，其余来自合成器 eSpeak-ng（已去除长度标记）。在这五种中，英语在两个粒度上都具有最大的不确定性，而芬兰语最小。在全部十二种中，按对齐单元计算时葡萄牙语和丹麦语超过英语，而德语和意大利语的排序则因粒度和转写方式的不同而不同。当英语改用 eSpeak-ng 转写（允许 u 拼写 /j/ 加元音）时，这些排序仍然成立。单位不确定性没有共享的直接……（原文摘要至此截断）

    arXiv:2611.13400v1 Announce Type: cross  Abstract: A reader can struggle to read in only one of their languages, and one proposed reason is how predictably spelling maps to sound. We measure that predictability as conditional entropy, the pronunciation uncertainty left once the spelling is seen, at two grain sizes (unit sizes): the alignment unit, an aligned spelling-sound chunk, and the syllable, bounded in the pronunciation. We analyse word lists in twelve alphabetic orthographies and report five in full, with English pronunciations from a dictionary and the rest from the synthesizer eSpeak-ng, length marks removed. Among the five, English has the most uncertainty and Finnish the least at both grains. Across all twelve, Portuguese and Danish exceed English per alignment unit, and German and Italian order differently by grain and transcription. These orders hold when English is transcribed by eSpeak-ng with u allowed to spell /j/ plus a vowel. Uncertainty per unit has no shared direct
    
[^76]: 教师对AI作弊态度的映射：基于论坛分析的探索性研究

    Mapping Instructors' Attitudes Towards Cheating with AI: An Exploratory Analysis Using Forum Analysis

    [https://arxiv.org/abs/2611.13366](https://arxiv.org/abs/2611.13366)

    本研究通过分析r/Professors论坛上201万条帖子，揭示了ChatGPT发布后大学教师对AI作弊的态度、防止作弊的动机及所采用的策略。

    

    在教与学的环境中，作弊和违反学术诚信一直是挑战之一，涉及行政、个人知识、学习和公平等多个层面的关切。大型语言模型的出现使对话式作弊问题成为焦点，因为这些模型明显具备完成学生学业作业的能力。本研究旨在利用大学教师近期关于使用生成式AI作弊的讨论，以更好地理解他们对学术诚信的看法。为此，我们分析了ChatGPT发布后r/Professors论坛（subreddit）上讨论作弊的201万条帖子（主题帖和评论）语料库，从而更好地理解教师防止作弊的动机及其采取的策略。我们综合归纳了应对作弊的“为什么”和“怎么做”，以识别未来可能存在的隐患。

    arXiv:2611.13366v1 Announce Type: new  Abstract: In teaching and learning environments, cheating and the violation of academic integrity have always been one of the challenges, involving different levels of concerns such as administrative, personal knowledge, learning, and equity. The introduction of large language models has brought cheating in dialogue to the forefront, given the apparent capabilities of these models to complete students' academic assignments. This study aims to leverage recent dialogues among university professors about cheating with Generative AI to better understand their beliefs about academic honesty. In doing so, we analyze a corpus of 2.01 million posts (threads and comments) in the r/Professors subreddit discussing cheating after the release of ChatGPT, providing a better understanding of teachers' motivations to discourage cheating and the strategies they use to do so. We synthesize the why and how of addressing cheating to identify caveats about the future 
    
[^77]: 理解学生AI辅助作弊的理由：基于在线论坛分析的教师视角

    Understanding Students' Rationales for AI-Mediated Cheating: An Instructor Perspective from Online Forums Analysis

    [https://arxiv.org/abs/2611.13358](https://arxiv.org/abs/2611.13358)

    该研究通过分析r/Professors论坛上201万条帖子，从教师视角首次大规模揭示了学生AI作弊背后的四大动机：博弈学术系统、数字-学术素养差距、表现焦虑以及不诚信行为的正常化。

    

    人工智能的广泛采用引发了学术诚信危机。虽然已有研究试图对作弊行为进行量化，但关于教师认为学生违规使用AI的动机，目前所知甚少。我们分析了Reddit上r/Professors社区的201万条帖子，以了解教师如何认知学生对AI辅助作弊的辩护理由。我们的定性分析揭示了四个主要主题：(1) 学术系统博弈，(2) 数字-学术素养差距，(3) 表现焦虑，以及 (4) 不诚信行为的正常化。

    arXiv:2611.13358v1 Announce Type: new  Abstract: The extensive adoption of AI has precipitated a crisis in academic integrity. While studies attempt to quantify cheating, less is known about the motivations instructors believe students have for using AI in prohibited ways. We analyze 2.01 million posts on r/Professors to understand how instructors perceive students' justifications for AI-mediated cheating. Our qualitative analyses reveal four primary themes: (1) Academic System Gaming, (2) Digital-Academic Literacy Gap, (3) Performance Anxiety, and (4) Normalization of Dishonesty.
    
[^78]: 测量大语言模型在社会科学概念标注中的定义敏感性

    Measuring Definition Sensitivity in LLM Annotation of Social Science Concepts

    [https://arxiv.org/abs/2611.13339](https://arxiv.org/abs/2611.13339)

    该论文提出“定义敏感性”这一新指标：让LLM使用不同定义对同一批文本分别进行标注，通过两组标注结果的分歧程度来衡量模型是否真正遵循了给定定义，还是仅仅依赖其预训练中形成的固有概念认知。

    

    在社会科学研究中，文本标注是一项基础性任务，传统上是一个成本高昂、难以规模化的人工过程。研究人员已开始使用大语言模型（LLM）进行标注，方法是将概念的定义写入提示词中——这是一条根据文本内容决定是否为文本赋予相应标签的规则。然而，仇恨言论、假新闻和恐怖主义等概念通常存在多个在理论上均可成立、相互竞争的定义。此外，目前尚不清楚模型究竟是真正应用了所给定义来产生标签，还是标签基于模型预训练中对概念的固有认知。为检验这一点，我们让模型使用不同的定义对相同文本进行两次标注，并将两组标注之间的分歧称为“定义敏感性”。作为对照，我们对其中一个定义进行改写但保持其要求不变——两组标注之间的分歧……

    arXiv:2611.13339v1 Announce Type: new  Abstract: In social science research, text annotation is a fundamental task that is traditionally a costly, hard-to-scale manual process. Researchers have used large language models (LLMs) to perform annotation by writing a definition of a concept into a prompt - a rule that determines whether a label should be attributed to a text depending on its content. However, concepts such as hate speech, fake news, and terrorism typically have several theoretically defensible competing definitions. It is also unknown whether the model actually applied the definition to provide the label or the label was based on the model's pretrained notion of the concept. To test this, we have the model annotate the same texts twice, each time with a different definition. We call the disagreement between the two sets of annotations definition sensitivity. As a control, we paraphrase one definition but leave its requirements unchanged - the disagreement between the annota
    
[^79]: MicroLens：用两种“透镜”读取、追踪与重连残差流

    MicroLens: Read, Trace, and Rewire the Residual Stream with Two Lenses

    [https://arxiv.org/abs/2611.13322](https://arxiv.org/abs/2611.13322)

    本文提出交互式工具 MicroLens，将 logit 透镜与雅可比透镜（J-lens）统一到同一张残差流“层×词元”网格上，实现对隐藏状态的读取、来源追踪与在线编辑，并发现 logit 透镜方向适合控制词元输出、J-lens 方向适合改变后续词元的读取。

    

    将隐藏状态解码为词语的“透镜”是观察语言模型内部机制的常用方法，但解码词语的热力图只能显示一个向量“读作什么”，无法区分模型是否使用、或在多大程度上使用了该信息。我们提出了一个交互式查看器，将读取、追踪与编辑功能统一到残差流的同一张“层×词元”网格上。在该查看器中，每个单元格都可以用两种透镜解码——logit 透镜和雅可比透镜（J-lens），可以回溯其信息来源词元，并进行编辑，同时模型的其余部分会被当场重新运行。我们在 Gemma-3-4B-it 上演示了三种使用场景，并以数百条提示的批量扫描实验作为支撑，展示该工具能够发现什么。首先，两种透镜提供了不同的“写入”方向，各自适合不同的编辑：logit 透镜的词语方向决定某个词元输出什么，而 J-lens 的方向则改变后续词元读取什么。其次，在两跳知识问答（two-hop trivia）中……

    arXiv:2611.13322v1 Announce Type: cross  Abstract: Lenses that decode hidden states into words are a popular way to look inside language models, but a heatmap of decoded words only shows what a vector reads as and cannot distinguish whether, or how much, the model uses the information. We present an interactive viewer that puts reading, tracing and editing on the same layers-by-tokens grid of the residual stream. In the viewer, every cell can be decoded with two lenses, the logit lens and the Jacobian lens (J-lens), traced back to the tokens it drew from, and edited, with the rest of the model rerun on the spot. We demonstrate three usage scenarios on Gemma-3-4B-it, backed by sweeps over a few hundred prompts, to show what this tool can discover. First, the two lenses give different directions to write with, each suiting a different edit: the logit lens's word direction sets what a token outputs, while the J-lens's direction changes what later tokens read. Second, in two-hop trivia que
    
[^80]: 解码令牌并行：面向可扩展大语言模型解码的内存高效分布式注意力并行

    Decode Token Parallelism: Memory-Efficient Distributed Attention Parallelism for Scalable LLM Decoding

    [https://arxiv.org/abs/2611.13293](https://arxiv.org/abs/2611.13293)

    提出解码令牌并行（TKNP）架构，通过将权重放置与注意力所有权解耦，解决了现有并行策略中KV缓存容量与注意力带宽难以兼顾的问题，实现了内存高效且可扩展的大语言模型解码。

    

    在高并发、长上下文场景下服务大语言模型，需要随解码池规模增长的KV缓存容量与注意力带宽。现有的并行化策略往往顾此失彼：数据并行复制模型权重，限制了KV缓存容量；张量并行在达到模型的KV头数量后便无法继续扩展注意力，并将密集的集合通信推送到节点间；解码上下文并行（DCP）通过对序列分片来恢复容量，但增加了对部分注意力结果的跨节点聚合。我们观察到，在解码阶段，稠密层与注意力的扩展方式不同：稠密层将权重读取摊销到整个批次上，能够在单个NVLink域内高效运行；而注意力的开销则随每个活跃请求的KV缓存而增长。我们提出了解码令牌并行（TKNP），这是一种将权重放置与注意力所有权解耦的解码池架构。TKNP将单一模型进行分布式部署……（摘要原文截断于此）

    arXiv:2611.13293v1 Announce Type: cross  Abstract: Serving large language models at high concurrency with long contexts requires KV-cache capacity and attention bandwidth that grow with the decode pool. Existing parallelization strategies trade one against the other: data parallelism replicates model weights, limiting KV capacity; tensor parallelism stops scaling attention at the model's KV-head count and pushes dense collectives across nodes; and decode context parallelism (DCP) recovers capacity by sharding the sequence but adds cross-node aggregation of partial attention results. We observe that during decode, dense layers and attention scale differently: dense layers amortize weight reads across the batch and run efficiently within a single NVLink domain, whereas attention cost grows with every active request's KV cache. We present Decode Token Parallelism (TKNP), a decode-pool architecture that decouples weight placement from attention ownership. TKNP distributes a single model co
    
[^81]: 什么因素驱动了大语言模型中符号机制的涌现？

    What drives the emergence of symbolic mechanisms in large language models?

    [https://arxiv.org/abs/2611.13270](https://arxiv.org/abs/2611.13270)

    本研究通过从零训练神经网络并系统地操控架构与数据分布因素，发现数据多样性（而非架构归纳偏置）是驱动大语言模型中符号机制涌现的关键因素。

    

    近期的研究识别出一系列支持大语言模型抽象推理的涌现符号机制，但尚不清楚是什么因素驱动了这些机制的涌现。为了解答这一问题，我们在抽象序列任务上从零开始训练神经网络，同时改变架构和数据分布方面的因素，并研究这些因素对符号机制涌现的影响。利用表征分析、注意力分析和因果中介分析，我们首先确认了在我们的任务上从零开始训练的基于Transformer的语言模型确实发展出了符号机制。随后我们发现，在多个数据多样性维度上，更大的数据多样性产生了更强的符号特征。令人惊讶的是，相同的特征在经过修改的Transformer架构甚至多层感知机（MLP）中也出现了，这表明架构归纳偏置对于符号机制的发展并非必要条件（原文在此处截断）。

    arXiv:2611.13270v1 Announce Type: new  Abstract: Recent work has identified a set of emergent symbolic mechanisms that support abstract reasoning in large language models, but it remains unclear what factors drive the emergence of these mechanisms. To address this question, we trained neural networks from scratch on abstract sequence tasks, varying both architectural and data distributional factors, and investigated the effects of these factors on the emergence of symbolic mechanisms. Using representational, attentional, and causal mediation analyses, we first confirmed that transformer-based language models trained from scratch on our task developed symbolic mechanisms. We then found that, across multiple axes of data diversity, greater diversity produced stronger symbolic signatures. Surprisingly, the same signatures emerged in modified transformer architectures and even multilayer perceptrons, indicating that architectural inductive biases are not necessary for the development of sy
    
[^82]: 面向多教师在线蒸馏的域路由增量优化

    Domain-Routed Delta Optimization for Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2611.13249](https://arxiv.org/abs/2611.13249)

    本文提出域路由增量优化方法（DRD-MOPD），通过为不同领域的样本指定各自更新的参数组件（以专家模型与基础模型的权重差初始化），在多教师在线蒸馏中显著保留原始专家的几何结构。

    

    多教师在线蒸馏为每个样本分配一个领域教师，但通常更新共享的学生参数。我们提出了域路由增量优化（DRD-MOPD），它明确规定了每个领域的样本应更新哪些参数组件。学生模型是冻结的基础模型与可训练组件之和，这些可训练组件由“专家模型减去基础模型”的权重差初始化。所有组件都参与每一次前向传播；四种更新模式定义了样本领域与组件之间的对应关系，且每种模式拥有独立的优化器状态。我们分析了这些更新规则，并通过角度、投影系数和能量来衡量对原始专家几何结构的保留程度。先前的工作（如 Open-MOPD）直接从基础模型进行蒸馏；其发布的最终检查点中，97.02% 的增量能量位于原始专家方向的张成空间之外。相比之下，我们的 Own 检查点在经过 20 次更新后，在每个方向上仍保持接近于一的投影系数……

    arXiv:2611.13249v1 Announce Type: cross  Abstract: Multi-teacher on-policy distillation assigns each sample a domain teacher but typically updates shared student parameters. We introduce Domain-Routed Delta Optimization (DRD-MOPD), which specifies which parameter components each domain's samples update. The student is the sum of a frozen base and trainable components initialized from expert-minus-base weights. All components participate in every forward pass; four update modes define the correspondence between sample domains and components, each with independent optimizer states. We analyze these update rules and measure retention of the original expert geometry through angles, projection coefficients, and energy. Prior work such as Open-MOPD distills directly from the base model; 97.02% of its released final checkpoint's increment energy lies outside the span of the original expert directions. In comparison, our Own checkpoint after 20 updates retains coefficients near one along each 
    
[^83]: 删除敏感性是模型的一种属性：对七个开放推理模型的三条件测量

    Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models

    [https://arxiv.org/abs/2611.13223](https://arxiv.org/abs/2611.13223)

    本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。

    

    推理模型在多轮工具会话的每一轮都会写下一条推理轨迹，而后续轮次会读取这段历史。在94k条ShareGPT对话中，73%的用户交互跨越多轮。上游提供商将每条推理轨迹绑定到其发出者，因此跨发出者转发的请求到达时其历史已被删除，而这种剥离操作带来的任务成本此前从未被测量。据我们所知，本文首次对多轮工具调用中完整推理历史删除进行了直接、盲测、多种子的测量，涵盖三种渲染条件、七个开放权重推理模型、三个随机种子、794个任务以及官方检查器。相同的删除操作导致的性能变化跨度为-4.4到+51.3分，且两端均不包含零；对于唯一一个从0%到100%删除比例重新测试的模型，其读数变化最多仅1.5分。删除敏感性是模型层面的属性。该读数跟随模型本身并保持稳定（原文在此处截断）。

    arXiv:2611.13223v1 Announce Type: cross  Abstract: Reasoning models write a reasoning trace at every turn of a multi-turn tool session, and later turns read this history. Across 94k ShareGPT conversations, 73 percent of user interactions span multiple turns. Upstream providers bind each trace to its issuer, so a request forwarded across issuers arrives with its history deleted, and the task cost of this stripping remains unmeasured. To the best of our knowledge, this paper presents the first direct, blind, multi-seed measurement of full reasoning-history deletion in multi-turn tool calling, through three rendering conditions, seven open-weight reasoning models, three seeds, 794 tasks, and the official checker. The same deletion spans -4.4 to +51.3 points with both ends excluding zero, and for the one model re-dosed from 0 to 100 percent deletion, the reading varies by at most 1.5 points. Deletion sensitivity is a model-level property. The reading follows the model itself and stays cons
    
[^84]: 20世纪80年代以来的拉丁美洲研究：从末世梦魇到同化式整合

    Latin American Studies since the 1980s: From Apocalyptic Dreams to Assimilationist Consolidation

    [https://arxiv.org/abs/2611.13216](https://arxiv.org/abs/2611.13216)

    该研究通过文献计量学揭示了拉丁美洲研究自1982年以来的“同化式整合”趋势——学科制度化地位提升的同时，却强化了以英语学术规范为主导的等级体系，导致拉美本土期刊边缘化和拉美学者向北方标准的趋同。

    

    我们研究了拉丁美洲研究（LAS）自1982年以来在三本顶级LAS期刊中的演变历程。通过网络分析、文本挖掘和文献计量学方法，我们追踪了该领域在学科定位、作者构成和引用模式上的变迁。我们的核心论点是：拉丁美洲研究经历了一场“同化式整合”。该领域如今已获得制度层面的认可，核心LAS期刊在学科网络中占据中心位置，并在各学科之间充当桥梁。然而，这种整合也强化了偏向英语学术规范的等级体系。曾经处于中心地位的拉丁美洲本土期刊已被边缘化；拉丁美洲作者的比例虽有所增长，但其学术特征正日益趋同于北方（英美）标准。

    arXiv:2611.13216v1 Announce Type: cross  Abstract: We investigate how Latin American Studies (LAS) has evolved since 1982 in three leading LAS journals. Using network analysis, text mining, and bibliometrics, we trace transformations in disciplinary positioning, authorship, and citation patterns. Our main claim is that LAS has undergone an assimilationist consolidation. The field is now institutionally recognisable, with core LAS journals occupying central positions in disciplinary networks and acting as bridges across fields. Yet this consolidation has reinforced hierarchies privileging Anglophone academic norms. Journals based in Latin America, once central, have been marginalised. The share of Latin American authors has grown, but their profiles converge towards Northern standards.
    
[^85]: HiCAP：基于个性的对话生成分层文化适应框架

    HiCAP: Hierarchical Cultural Adaptation via Personality for Dialogue Generation

    [https://arxiv.org/abs/2611.13207](https://arxiv.org/abs/2611.13207)

    提出HiCAP框架，将个性特质通过分层控制机制融入对话生成——先使回应与文化基调对齐、再依据个性细化语言表达与内容，从而实现文化自适应的对话生成。

    

    对话系统中的文化适应是指生成符合不同文化语言规范和社会期望的回应的能力。然而，现有的对话生成方法对个性在塑造文化表达和沟通风格中所起的作用关注有限。因此，我们提出了HiCAP，这是一个将个性特质融入文化自适应对话生成的新型框架。HiCAP采用一种以个性为条件的控制机制，通过分层方式调节回应风格：首先使回应与文化基调保持一致，随后根据个体个性属性对语言表达和内容进行细化。为了嵌入这些因素，该模型结合了监督微调与基于指令的调优，并借助动态指令模板和多智能体演示来捕捉文化与个性的多样化组合。在CulturalPe（数据集）上的实验……

    arXiv:2611.13207v1 Announce Type: new  Abstract: Cultural adaptation in dialogue systems refers to the ability to generate responses that conform to linguistic norms and social expectations across different cultures. However, existing dialogue generation approaches pay limited attention to the role of personality in shaping cultural expression and communication style. We therefore propose HiCAP, a novel framework that integrates personality traits into culturally adaptive dialogue generation. HiCAP employs a personality-conditioned control mechanism that hierarchically modulates response style by first aligning it with the cultural tone and then refining linguistic expression and content according to individual personality attributes. To embed these factors, the model combines supervised fine-tuning and instruction-based tuning, supported by dynamic instruction templates and multi-agent demonstrations to capture diverse combinations of culture and personality. Experiments on CulturalPe
    
[^86]: 面向NVLink上混合专家推理的单边专家并行All-to-All通信

    One-Sided Expert-Parallel All-to-All for Mixture-of-Experts Inference over NVLink

    [https://arxiv.org/abs/2611.13188](https://arxiv.org/abs/2611.13188)

    NVLinkOneSided提出了一种面向NVLink域内混合专家推理的单边all-to-all通信方法，通过免元数据交换的直接推送、原地读取机制以及Fence和Sentinel两种同步协议，在Blackwell GPU上比DeepEP快1.2-3.0倍。

    

    专家并行将混合专家层的专家分片到多个GPU上，因此每个MoE层需要两次all-to-all通信，即分发和组合。与训练相比，推理时每个GPU一次处理的token数量相对较少，在小批量规模下，主导延迟的是不随token数量增长的固定开销，而非数据移动本身。我们提出了NVLinkOneSided，一种面向单个NVLink域内GPU的单边EP all-to-all通信方法。每个源rank无需预先交换元数据，直接将其token推送到目标GPU上以rank为主组织的缓冲区中，分组GEMM则直接原地读取这些数据，而无需从按专家复制的副本中读取。我们为通信所需的同步提供了两种协议，即Fence和Sentinel，分别适用于高吞吐量和低延迟场景。在Blackwell GPU上，NVLinkOneSided完成分发和组合的速度比DeepEP快1.2-3.0倍。

    arXiv:2611.13188v1 Announce Type: cross  Abstract: Expert parallelism (EP) shards the experts of a Mixture-of-Experts (MoE) layer across GPUs, so every MoE layer requires two all-to-all communications, namely dispatch and combine. Compared with training, inference processes relatively few tokens per GPU at a time, and at small batch sizes fixed costs that do not grow with the number of tokens, rather than data movement, dominate the latency. We present NVLinkOneSided, a one-sided EP all-to-all for GPUs within a single NVLink domain. Each source rank pushes its tokens, without first exchanging metadata, into rank-major buffers on the target GPUs, and the grouped GEMMs read them in place instead of from a per-expert copy. We provide two protocols for the synchronization that the communication requires, namely Fence and Sentinel, for high-throughput and low-latency scenarios respectively. On Blackwell GPUs, NVLinkOneSided completes dispatch and combine 1.2-3.0$\times$ faster than DeepEP, 
    
[^87]: 答案一致但理由不一：大语言模型推理失败分类体系的测量局限性

    Agreeing on the Answer but Not the Reason: Measurement Limits of LLM Reasoning Failure Taxonomies

    [https://arxiv.org/abs/2611.13140](https://arxiv.org/abs/2611.13140)

    该论文表明，LLM推理失败分类的测量结果很大程度上取决于测量程序本身：过半失败案例可归入多个类别，类别占比随分类器测试顺序大幅波动（26.4%–76.4%），且人工标注者对类别归属的共识远低于对任务答案正确性的共识。

    

    大语言模型（LLM）推理失败分类体系正越来越多地被当作测量结果来报告：论文会声明模型错误中有多少比例落入某些命名类别，然后在模型之间比较这些比例。我们证明，对于其中一个类别，所报告的比例在很大程度上是测量方法本身的属性。我们定义了“组合崩塌”（Compositional Collapse，即模型注意到问题所需的某个子部分，随后又将其丢弃或错误应用），并在一个公开的多语言数学基准上跨越十二个模型和十一种语言进行了测量（共2,316条评分记录）。首先，50.1%的失败案例包含支持不止一个类别的证据，因此所报告的多数比例取决于分类器测试各类别的顺序：仅交换两个测试的顺序就会使该比例在十二个模型中的十一个里发生变化，而在所有六种排序下，“组合崩塌”的占比在26.4%到76.4%之间波动。其次，标注一致性在该类别层面崩溃，但在任务层面并未崩溃。来自全部十一种语言的母语者对179份记录进行了评判……

    arXiv:2611.13140v1 Announce Type: new  Abstract: Failure taxonomies for LLM reasoning are increasingly reported as measurements: papers state what fraction of a model's errors fall into named categories, then compare those fractions across models. We show that for one such category the reported fraction is largely a property of the measurement procedure. We defined Compositional Collapse (a model notices a required sub-part of a problem, then drops or misapplies it) and measured it across twelve models and eleven languages on a public multilingual math benchmark (2,316 graded records). First, 50.1% of failures carry evidence for more than one category, so the reported plurality depends on the order in which the classifier tests them: reversing two tests moves it in eleven of twelve models, and across all six orderings Collapse's share spans 26.4% to 76.4%. Second, agreement collapses for the category but not for the task. Native speakers of all eleven languages judged 179 transcripts, 
    
[^88]: 哪些成功案例能成为好的参考？面向多轮在线策略蒸馏的动作序列对齐方法

    Which Successes Make Good References? Action-Sequence Alignment for Multi-Turn On-Policy Distillation

    [https://arxiv.org/abs/2611.13135](https://arxiv.org/abs/2611.13135)

    该论文将多轮在线策略蒸馏中的成功参考选择建模为自由目标最优传输问题，并提出以最长公共子序列（LCS）为基础的动作序列对齐作为轨迹兼容性的可观测代理，为每条失败轨迹匹配最兼容的成功参考，从而避免误导性监督、提升蒸馏效果。

    

    在线策略蒸馏（OPD）通过在学生模型自身的轨迹上进行监督，来减少训练与推理之间的分布失配。对于多轮工具使用智能体而言，让教师模型以经过验证的同任务成功轨迹作为参考，可以提供有用的后见之明；但不同的成功案例可能到达不兼容的交互状态，从而提供误导性的监督信号。我们将成功参考的选择问题形式化为一个自由目标的最优传输问题，即把每条失败轨迹分配给与其最兼容的成功参考。由于潜在的状态路径是隐藏的，我们采用动作序列对齐作为兼容性的可观测代理，并用最长公共子序列（LCS）加以实现——LCS能够识别出共享的有序动作主干，同时允许可恢复的绕路。这一视角使轨迹兼容性成为多轮OPD中参考选择的核心准则。在ALFWorld上的实验……（原文摘要在此处截断）

    arXiv:2611.13135v1 Announce Type: cross  Abstract: On-policy distillation (OPD) reduces train-inference distribution mismatch by supervising a student on its own trajectories. For multi-turn tool-use agents, conditioning the teacher on a validated same-task successful trajectory as a reference can provide useful hindsight, but different successes may reach incompatible interaction states and offer misleading supervision. We cast successful-reference selection as a free-target optimal transport problem, which reduces to assigning each failed trajectory its most compatible successful reference. Because latent state paths are hidden, we use action-sequence alignment as an observable proxy for compatibility and instantiate it with longest common subsequence (LCS), which identifies a shared ordered backbone of actions while allowing recoverable detours. This perspective makes trajectory compatibility the central criterion for reference selection in multi-turn OPD. Experiments on ALFWorld an
    
[^89]: 面向语言模型系统中认知记忆重构的激活引导图检索

    Activation-Guided Graph Retrieval for Cognitive Memory Reconstruction in Language Model Systems

    [https://arxiv.org/abs/2611.13127](https://arxiv.org/abs/2611.13127)

    该论文提出了一种激活条件化的记忆图检索架构，通过请求与目标线索激活授权记忆图、扇出衰减控制有界遍历、实体递归发射线索，并在生成前重构类型化证据图，从而为长寿命AI系统实现认知记忆重构。

    

    检索增强语言模型系统通常通过基于查询条件的相关性来选择上下文。我们认为，长寿命AI系统需要第二种算子：激活条件化的记忆可用性。我们规定了一种架构，其中请求与目标线索为授权的记忆图进行播种，激活与扇出衰减分配有界的遍历前沿，检索到的实体递归地发射线索，并在生成之前重构类型化证据图。该规范产生了关于候选选择、遍历控制、图正则化、证据重构和强化的五项架构性推论。我们推导了实现层面的成本估计，并定义了消融实验、失败条件以及可复现的评估协议。如果在同等预算下，一个更简单的图重排序器在因果与时间恢复任务上能够匹配完整遍历的效果，则该核心主张将被收窄或证伪。

    arXiv:2611.13127v1 Announce Type: new  Abstract: Retrieval-augmented language model systems generally select context by query-conditioned relevance. We argue that long-lived AI systems require a second operator: activation-conditioned memory availability. We specify an architecture in which request and goal cues seed an authorized memory graph, activation and fan attenuation allocate a bounded traversal frontier, retrieved entities recursively emit cues, and a typed evidence graph is reconstructed before generation. The specification yields five architectural consequences about candidate selection, traversal control, graph regularization, evidence reconstruction, and reinforcement. We derive implementation-level cost estimates and define ablations, failure conditions, and a reproducible evaluation protocol. The central claim is narrowed or falsified if a simpler graph reranker matches full traversal on causal and temporal recovery under equal budgets. This architecture-and-analysis pre
    
[^90]: ASPIRE：一个用于迭代推理与探索的智能体化系统辨识平台

    ASPIRE: An Agentic System-Identification Platform for Iterative Reasoning and Exploration

    [https://arxiv.org/abs/2611.13114](https://arxiv.org/abs/2611.13114)

    ASPIRE是一个由大语言模型智能体驱动的非线性系统辨识框架，能够自主选择并执行辨识方法，从测量数据构建闭式非线性模型，融入先验物理知识并独立验证，且支持自主与交互两种运行模式。

    

    系统辨识需要在辨识方法的选择、先验物理知识的融入以及模型验证等方面做出专家决策。我们提出了ASPIRE，这是一个用于非线性系统辨识的框架，其中大型语言模型智能体负责选择并执行辨识方法。ASPIRE能够从测量数据中构建闭式（解析形式）的非线性静态或动态模型，并可选择性地辅以系统描述、模型的预期用途，以及以形式化方式或自然语言表达的物理要求。每个模型包含其参数、估计记录、对模型各项的解释，以及在智能体无法访问的测试数据上评估的误差。形式化的物理要求由独立机制进行验证。ASPIRE既可以自主运行，也可以交互式运行，允许智能体提出问题，用户也可以请求进一步的分析和额外的辨识轮次。我们在三个实测系统上对ASPIRE进行了评估：一个移动机器人……（摘要截断）

    arXiv:2611.13114v1 Announce Type: cross  Abstract: System identification requires expert decisions on the identification method, incorporation of prior physical knowledge, and model validation. We propose ASPIRE, a framework for nonlinear system identification in which a large language model agent selects and executes the identification method. ASPIRE constructs closed-form, nonlinear static or dynamic models from measured data, optionally supplemented by a system description, the intended use of the model, and physical requirements expressed formally or in natural language. Each model includes its parameters, an estimation record, an interpretation of its terms, and an error evaluated on test data inaccessible to the agent. Formal physical requirements are verified independently. ASPIRE operates autonomously or interactively, allowing the agent to ask questions and the user to request further analyses and identification rounds. We evaluate ASPIRE on three measured systems: a mobile ro
    
[^91]: 在错误发生处捕获：LLM推理过程中的实时错误纠正

    Catch It Where It Happens: Real Time Error Correction During LLM Reasoning

    [https://arxiv.org/abs/2611.13107](https://arxiv.org/abs/2611.13107)

    提出在线确定性验证方法，利用计算机代数系统等代码工具在LLM生成推理步骤的同时实时校验（无需GPU或额外模型调用），发现错误时让模型从出错处继续而非从头重来，从而避免错误传播并节省推理开销。

    

    大语言模型越来越多地被信任承担数学、科学和工程工作，但它们的错误往往写得十分自信且难以察觉。一个错误的步骤往往会持续传播，导致模型在错误的前提上浪费推理token。现有的保障措施要么只检查最终答案然后从头重新生成，要么依赖一个验证器LLM——它会与主模型竞争GPU算力并拖慢响应时间。我们提出了在线确定性验证：许多类型的推理步骤可以通过代码进行验证，因此一个步骤在被生成的同时就能得到实时验证，无需任何模型调用或GPU使用。每当检查失败时，我们指出错误所在，并让模型从同一位置继续推理。我们在符号数学上演示了这一方法，因为其中的推导过程已包含显式表达式，计算机代数系统可以在毫秒级内完成检查。在ASyMoB基准测试上，使用三种操作符（运算）进行了评估……

    arXiv:2611.13107v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly trusted with mathematical, scientific and engineering work, yet their mistakes are often confidently written and hard to spot. A single wrong step tends to propagate, making the model waste inference tokens following a false premise. Existing safeguards either check only the final answer and restart from scratch, or rely on a verifier LLM that competes for GPU capacity and slows down response time. We propose online, deterministic verification: many kinds of reasoning steps can be validated by code, so a step can be verified in real time, as it is written, without any model call or GPU use. Whenever a check fails, we point out the mistake and let the model continue from the same spot. We demonstrate this approach on symbolic mathematics, where derivations already contain explicit expressions that a computer algebra system can check in milliseconds. Tested on the ASyMoB benchmark with three op
    
[^92]: 一个温度统御一切？

    One Temperature to Rule Them All?

    [https://arxiv.org/abs/2611.13077](https://arxiv.org/abs/2611.13077)

    该论文研究了 pass@K 推理中温度多样性何时优于单一温度，发现虽然更难的问题更难共享高效的采样温度，但这一趋势仅随难度呈对数增长，表明单一调优温度在多数情况下已足够有效。

    

    推理时采样是提升大语言模型性能的核心机制，其中温度参数控制着可靠性与多样性之间的权衡。尽管近期研究表明，对不同样本使用不同温度可以改进多样本推理效果，但简单的固定温度方案在实践中仍是标准做法。我们在经典的 pass@K 设定下研究这一差异，探讨温度多样性何时有用，以及单一调优温度何时已经足够。为此，我们分析了多个模型和数据集上每个问题的成功概率，追踪其随采样温度的变化规律。我们识别出这些温度曲线的若干一致特性，其中包括：越难的问题越不太可能与其他问题共享高效的采样温度，尽管这一趋势仅随难度呈对数增长。随后，我们推导出一种高效方法，用于评估理论上最优的多温度分配方案。

    arXiv:2611.13077v1 Announce Type: cross  Abstract: Inference-time sampling is a central mechanism for improving large language models, with temperature controlling the tradeoff between reliability and diversity. While recent work suggests that varying temperature across samples can improve multi-sample inference, simple fixed-temperature schemes remain the standard in practice. We investigate this discrepancy in the canonical pass@K setting, asking when temperature diversity is useful, and when a single tuned temperature suffices. To this end, we analyze per-question success probabilities across multiple models and datasets, tracking how they vary with sampling temperature. We identify several consistent properties of these profiles, including that harder questions are less likely to share efficient sampling temperatures, although this trend is only logarithmic in hardness. We then derive an efficient method for evaluating the theoretically optimal multi-temperature allocation scheme, 
    
[^93]: 字节对编码的左侧涟漪：面向任意词典的认证式回溯

    The Left Ripple of Byte-Pair Encoding: Certified Lookbehind for Every Dictionary

    [https://arxiv.org/abs/2611.13061](https://arxiv.org/abs/2611.13061)

    该论文对任意字节对编码词典完整解决了“左侧前置字符串能把分词改变多远”的问题：起源半径可由双自动机乘积给出的半径证书在多项式时间内判定，且恰好在存在一个具有两个无公共边界的规范铺砌的周期词（在排序条件下即某规则 的两个副本在其自身阶段相邻）时为无限。

    

    Berglund 与 van der Merwe 对字节对编码进行了形式化，并界定了在右侧追加字符串能在多大程度上改变分词结果，但留下了相反的问题未解决：在左侧前置一个字符串能在多大程度上改变分词结果？哪些词典会使这种涟漪无限延续？当涟漪有限时能否高效计算？本文针对任意词典完整回答了这三个问题。“起源半径”——即左侧前置能在多大范围内改变覆盖区间——由一个由自动机两次运行构成的乘积给出“半径证书”来判定，其运行时间关于写出词典的规模为多项式级；且该半径为无限，当且仅当存在某个周期词具有两个没有公共边界的规范铺砌。在他们的排序条件下，半径为无限当且仅当某条规则 在其自身阶段使其两个副本相邻，而这样的规则会使任何词典都产生无限涟漪。当半径有限时，宽度可由各半径读出的每个窗口都具有一个切割点，且半径有限当且仅当

    arXiv:2611.13061v1 Announce Type: cross  Abstract: Berglund and van der Merwe formalized byte-pair encoding and bounded how far an appended string can change a tokenization, leaving the converse open: how far can a string prepended on the left change it, which dictionaries make that ripple infinite, and can it be computed efficiently when finite? We answer all three for every dictionary. The origin radius, how far a prepend can change covering spans, is decided with radius certificates by a product of two runs of the automaton that guesses the final tokens, in time polynomial in the dictionary written out, and it is infinite exactly when one periodic word has two canonical tilings with no common boundary. Under their ordering condition it is infinite exactly when some rule (t, t) finds its two copies adjacent at its own stage, and such a rule makes any dictionary infinite. A finite radius makes every window of a width read off the radii have a cut, and the radius is finite exactly when
    
[^94]: 一种面向204种语言的多词分词器：即使在英语上所需词元也比GPT-4o更少

    A Multi-Word Tokenizer for 204 Languages That Needs Fewer Tokens Than GPT-4o's, Including on English

    [https://arxiv.org/abs/2611.13059](https://arxiv.org/abs/2611.13059)

    该论文提出Havon多词分词器系列，通过在约200种语言上从头训练并引入多词训练阶段，在包括英语在内的161种语言上所需词元数量少于GPT-4o的分词器，在印度语言上的词元数中位数仅为GPT-4o的0.74倍。

    

    印度诸语言在GPT-4o的分词器上处理相同内容时，需要1.6到14倍于英语的词元数量，这使其服务成本更高、速度更慢。我们在约200种语言的人类书写文本上从头训练分词器。Havon v3是一个字节级BPE分词器，拥有262,144个词元，并配备一个预分词器，该预分词器能够将组合符号与其字母保持在一起，并对无空格书写的文字系统进行分块。在FLORES-200开发测试集上，它在全部25种印度语言上所需的词元数量少于GPT-4o的o200k_base（中位数为GPT-4o的0.74倍），在204种语言中的172种上所需词元更少，但在英语上多7%。Havon v4在v3的基础上增加了第二个多词训练阶段。在恰好与GPT-4o相同的200,019词元词表规模下，v4在英语上比GPT-4o少用6.1%的词元，在204种语言中的161种以及全部25种印度语言上所需词元更少（中位数为0.74）；在相同规模下，v3在英语上多9.4%的词元，仅在133种语言上占优。在262,144词元规模下，v4需要少9.6%的词元

    arXiv:2611.13059v1 Announce Type: cross  Abstract: Languages of India need 1.6 to 14 times as many tokens as English on GPT-4o's tokenizer for the same content, which makes them more expensive and slower to serve. We train tokenizers from scratch on human-written text in about 200 languages. Havon v3 is a byte-level BPE tokenizer with 262,144 tokens and a pre-tokenizer that keeps combining marks with their letters and chunks scripts written without spaces. On FLORES-200 devtest it needs fewer tokens than GPT-4o's o200k_base on all 25 languages of India (median 0.74 of GPT-4o's count) and on 172 of 204 languages, but 7% more on English. Havon v4 adds a second, multi-word training stage (SuperBPE) to v3. At exactly GPT-4o's vocabulary size of 200,019 tokens, v4 needs 6.1% fewer tokens than GPT-4o on English and fewer on 161 of 204 languages and all 25 languages of India (median 0.74); at the same size, v3 needs 9.4% more on English and wins on 133. With 262,144 tokens, v4 needs 9.6% fewe
    
[^95]: 2026年的印度语言“代币税”：测量分词器差异及其对印度各语言的成本

    The Indic Token Tax in 2026: Measuring Tokenizer. Disparity and Its Cost for the Languages of India

    [https://arxiv.org/abs/2611.13058](https://arxiv.org/abs/2611.13058)

    该研究测量了九个主流分词器对印度25种语言的“代币税”，发现印度语言所需token量最高可达英语的13.7倍（GPT-4o分词器），导致印度语言用户在商业LLM使用中面临更高的费用、更慢的速度和更小的有效上下文，而Sarvam分词器是唯一将差异控制在2.6倍以内的第三方选择。

    

    商业语言模型API按token计费，因此当分词器处理某种语言的相同内容需要更多token时，就会使该语言的使用更昂贵、速度更慢，并且在固定上下文窗口中能容纳的文本更少。我们在FLORES-200中全部25种印度语言（2,009句与英文对齐的专业翻译句子）以及来自PHINC的13,732对人工翻译的印地英语-英语社交媒体文本上，测量了九个公开分词器的这种差异，包括GPT-4、GPT-4o、Qwen2.5、DeepSeek-V3、Mistral-Nemo以及四个Sarvam模型的分词器。在GPT-4o的分词器上，相同内容所需的token是英语的1.57倍（印地语）到5.01倍（奥里亚语），而使用Ol Chiki文字的桑塔利语高达13.7倍；GPT-4的分词器最高需要12.8倍。Sarvam-30B/105B是唯一在所有语言上都保持在2.6倍以下（中位数1.53）的第三方分词器。在一个使用截至11月5日获取的官方标价的示例场景中……

    arXiv:2611.13058v1 Announce Type: cross  Abstract: Commercial language model APIs bill per token, so a tokenizer that needs more tokens for the same content in one language makes that language more expensive, slower, and able to fit less text in a fixed context window. We measure this disparity for nine public tokenizers, including those of GPT-4, GPT-4o, Qwen2.5, DeepSeek-V3, Mistral-Nemo and four Sarvam models, on all 25 languages of India available in FLORES-200 (2,009 professionally translated sentences aligned with English) and on 13,732 human-translated Hinglish-English social media pairs from PHINC. On GPT-4o's tokenizer the same content needs between 1.57 times (Hindi) and 5.01 times (Odia) the tokens of English, and 13.7 times for Santali in Ol Chiki script; GPT-4's tokenizer needs up to 12.8 times. Sarvam-30B/105B is the only third-party tokenizer that stays below 2.6 times on every language (median 1.53). In an illustrative scenario using official list prices accessed on 5 N
    
[^96]: GTAGym：用于评估与强化学习的真实AAA游戏任务环境

    GTAGym: Real AAA Game Tasks Environment for Evaluation and Reinforcement Learning

    [https://arxiv.org/abs/2611.13028](https://arxiv.org/abs/2611.13028)

    本文提出GTAGym——一个基于《侠盗猎车手V》真实AAA游戏任务的评估与强化学习环境，其基准GTAVBench包含103个跨九类任务家族的在线主线任务及47个图像式射击狙击任务，支持截图与键鼠操作的并行测试并自动校验任务结果，用于评测前沿大语言模型的智能体能力。

    

    AAA游戏为评估智能体的感知、推理和控制能力提供了丰富的测试平台。然而，评估智能体能否在AAA游戏中完成任务十分困难，且通常需要人工标注，因为任务结果往往难以自动验证，更难以将其应用于强化学习（RL）。我们提出了GTAGym，一个面向GTA V任务的评估与强化学习环境及框架。其评估基准GTAVBench源自GTA V主线剧情任务，包含跨九个任务类别重建的103个在线任务，以及基于GTA V图像的47个图像式射击与狙击任务。一个网络服务器提供对在线环境和评估界面的访问，通过截图和键盘-鼠标操作支持并行测试，并具备自动的引擎状态结果校验。我们在GTAVBench上使用文本任务指令评估了六个前沿大语言模型，并在其GTAVBench-AGI分支上进行评估，其中指令……

    arXiv:2611.13028v1 Announce Type: cross  Abstract: AAA games offer a rich testbed for evaluating agent perception, reasoning, and control. However, evaluating whether agents can complete tasks in AAA games is challenging and often human-annotated because task outcomes are often difficult to verify automatically, let alone applying it for Reinforcement Learning (RL). We introduce GTAGym, an environment and framework for evaluation and RL on GTA V tasks. Its evaluation benchmark, GTAVBench, derived from GTA V main-story missions, contains 103 online tasks reconstructed across nine task families, alongside 47 image-based shooting and sniping tasks using GTA V imagery. A web server provides access to the online environment and evaluation interface, supporting parallel testing through screenshots and keyboard-mouse actions with automatic engine-state outcome checks. We evaluate six frontier LLMs on GTAVBench with textual task instructions and on its GTAVBench-AGI branch, where instructions 
    
[^97]: 面向智能体软件开发的仓库上下文引导图像理解

    Repository-Context-Guided Image Understanding for Agentic Software Development

    [https://arxiv.org/abs/2611.13026](https://arxiv.org/abs/2611.13026)

    提出通过从同一代码库渲染清晰的参考截图来增强编程智能体，从而在查询截图质量退化、视觉证据不清晰时，仍能准确识别UI元素并定位相关源文件。

    

    编程智能体越来越多地在错误报告中接收UI截图，然而图像质量的下降——由低分辨率、压缩伪影或过小的渲染文本引起——会阻碍它们识别相关的UI元素和源文件。随着图像质量恶化，视觉语言模型的可靠性降低，而OCR辅助方法则可能将识别错误从退化的输入中传播开来。当定位所需的视觉证据在查询截图中已不再清晰可见时，这两种方法都会受到限制。我们提出用仓库渲染的参考图像来增强编程智能体：即从智能体受命修改的同一代码库生成的清晰截图。这些参考图像在保留当前可能包含错误的实现状态的同时，提供了应用程序UI的可读视觉证据。由于它们是从现有仓库生成的，而无需依赖……

    arXiv:2611.13026v1 Announce Type: cross  Abstract: Coding agents increasingly receive UI screenshots as part of bug reports, yet degraded image quality - caused by low resolution, compression artifacts, or small rendered text - can prevent them from identifying the relevant UI elements and source files. Vision-language models become less reliable as image quality deteriorates, while OCR-assisted approaches may propagate recognition errors from degraded inputs. Both approaches are limited when the visual evidence required for localization is no longer clearly observable in the query screenshot. We propose augmenting coding agents with repository-rendered reference images: crisp screenshots generated from the same codebase that the agent is tasked with modifying. These references provide readable visual evidence of the application's UI while preserving the current, potentially bug-containing implementation state. Because they are generated from the available repository rather than requir
    
[^98]: khammoon：从语言学准则到泰语语素切分器

    khammoon: From a Linguistic Guideline to a Thai Morpheme Segmenter

    [https://arxiv.org/abs/2611.13012](https://arxiv.org/abs/2611.13012)

    该论文提出基于语言学准则构建的泰语语素切分器khammoon，通过在语素（最小意义单位）层面进行切分消除复合构词歧义，且语素级模型在词性标注、命名实体识别和情感分类等下游任务上表现与词级基线相当甚至更优。

    

    泰语书写时词与词之间没有空格，这使得分词成为大多数泰语自然语言处理（NLP）流程中的基础步骤。现有工具主要复现诸如BEST等语料库的词级切分规范；然而，由于泰语中“词”的概念缺乏统一的语言学共识，这些规范难以应对能产性的复合构词，导致切分边界随意，并在下游表示中引入歧义。因此，我们提出在形态学层面进行切分，聚焦于khammoon，即承载意义的最小原子单位（语素）。在语素层面进行切分能够提供显式且一致的形态表示，从而消除复合构词带来的歧义。至关重要的是，这种表示有利于下游学习：基于语素训练的模型在词性标注、命名实体识别和情感分类任务上的表现与词级基线相当甚至更优，这表明语言学上的一致性……

    arXiv:2611.13012v1 Announce Type: new  Abstract: Thai is written without spaces between words, making segmentation a foundational step in most Thai NLP pipelines. Existing tools primarily reproduce the word-level conventions of corpora such as BEST; however, because the notion of a "word" in Thai lacks a unified linguistic consensus, these conventions struggle with productive compounding, leading to arbitrary boundaries and ambiguity in downstream representations. Instead, we propose operating at the morphological level, focusing on khammoon, the smallest atomic unit that carries meaning. Segmenting at the morphemic level provides an explicit, consistent morphological representation that eliminates compounding ambiguity. Crucially, this representation benefits downstream learning: models trained on morphemes perform on par with or exceed word-level baselines across part-of-speech tagging, named-entity recognition, and sentiment classification, demonstrating that linguistic consistency 
    
[^99]: 眼见未必为实：反事实病灶擦除教会小型医学视觉-语言智能体只报告图像中真实存在的内容

    Seeing Is Not Believing: Counterfactual Lesion Erasure Teaches a Small Medical Vision-Language Agent to Report Only What the Image Shows

    [https://arxiv.org/abs/2611.13006](https://arxiv.org/abs/2611.13006)

    该论文提出无需人工标注的反事实病灶擦除评测范式与证据敏感度差距（ESG）指标，并据此训练出40亿参数的医学视觉-语言智能体MedScope-4B，使其只报告图像中真实可见的发现，将“幻觉报告被擦除病灶”的比例从前沿模型的41–62%降至9.6%。

    

    医学视觉-语言模型（VLM）常常仅仅因为问题中提到了某个发现的名称就报告该发现，而标准基准测试无法捕捉这一现象。我们从公开分割数据集构建反事实视图，无需人工标注：包括通过图像修复擦除病灶的图像、在其他位置进行同形状对照编辑的图像，以及未经编辑的无病灶图像。证据敏感度差距（ESG）定义为擦除后“不存在或转诊/参考”类回答的比率减去对照编辑后的相应比率；对于忽略图像或对两种编辑反应相同的模型，该指标为零。MedScope-4B（基于Qwen3.5-4B、LoRA微调、配备图像工具）在由标签推导的训练轨迹上训练，按预先注册的规则进行模型选择，并一次性与九个前沿模型和六个开源VLM进行对比测试；随后通过四项预先注册的检验探究其机制。在涵盖五种成像类型的507例内部病例上，MedScope-4B达到ESG 0.758，且仅在9.6%的病例中声称存在已被擦除的病灶，而前沿模型分别为0.27–0.47和41–62%（所有p<0.001）。

    arXiv:2611.13006v1 Announce Type: new  Abstract: Medical vision-language models (VLMs) often report a finding because the question names it; standard benchmarks miss this. From public segmentation datasets we build counterfactual views without manual annotation: the lesion erased by inpainting, a same-shape control edit elsewhere, and unedited lesion-free images. The evidence sensitivity gap (ESG) is the rate of absent-or-refer answers after erasure minus the rate after the control edit; it is zero for a model that ignores the image or reacts to both edits alike. MedScope-4B (Qwen3.5-4B, LoRA, image tools) was trained on label-derived trajectories, selected by a pre-registered rule and tested once against nine frontier and six open VLMs; four pre-registered checks then probed the mechanism. On 507 internal cases across five imaging types, MedScope-4B reached ESG 0.758 and claimed an erased lesion in 9.6% of cases, against 0.27-0.47 and 41-62% for the frontier models (all p<0.001), with
    
[^100]: 从线性探针到贝叶斯流形：大语言模型中算术运算的几何结构

    Linear Probes to Bayesian Manifolds: Geometry of Arithmetic in Large Language Models

    [https://arxiv.org/abs/2611.12998](https://arxiv.org/abs/2611.12998)

    该论文提出一个四阶段贝叶斯分析流程，将探针准确率分解为可解码性、形状、归属权和因果使用四个独立命题并分别检验，从而更严谨地判定大语言模型激活中的几何结构（如编码算术运算的螺旋或圆形）是否真正被模型用于计算。

    

    线性探针是从大语言模型（LLM）的激活值中研究几何结构（如螺旋线和圆）的标准方法，此类结构常被当作模型利用其进行计算的依据。然而，仅凭探针准确率无法证实这一点，因为它混淆了四个相互独立的命题：（1）概念可以从激活值中被解码；（2）这些概念构成特定形状；（3）该形状属于被探测的概念，而非模型同时表示的另一个相关概念；（4）模型的输出依赖于该形状。我们将这些命题分别称为可解码性、形状、归属权和因果使用，并通过一个四阶段的贝叶斯流程对它们分别进行检验。第二阶段比较候选形状，仅当其证据超过预定义阈值时才选出最佳形状。第三阶段移除一组预先指定的相关概念，检验该形状究竟是被探测概念所拥有，还是从相关概念继承而来。

    arXiv:2611.12998v1 Announce Type: cross  Abstract: Linear probes are a standard method to study geometric structures, such as helices and circles, from the activations of large language models (LLMs), and such structure is often taken as evidence that the model computes with it. Probe accuracy alone cannot establish this, because it conflates four separate claims. These are (1) that the concept is decodable from the activations, (2) that they form a specific shape, (3) that the shape belongs to the probed concept rather than to a related concept the model also represents, and (4) that the output of the model depends on the shape. We call these claims decodability, shape, ownership, and causal use, and we test them separately with a four-stage Bayesian pipeline. Stage 2 compares candidate shapes and selects the best one only when its evidence passes a predefined threshold. Stage 3 removes a set of related concepts, specified in advance, and tests whether the shape is owned or inherited.
    
[^101]: 源头上的容量分配：面向大语言模型知识编辑的稀疏目标优化

    Capacity Allocation at the Source: Sparse Target Optimization for LLM Knowledge Editing

    [https://arxiv.org/abs/2611.12995](https://arxiv.org/abs/2611.12995)

    该论文建立了参数空间干扰与目标隐状态干扰之间的正相关理论联系，并提出一种 $\ell_1$ 正则化方法对知识向量进行稀疏化，从源头优化目标隐状态构建以更好地分配模型有限容量空间。

    

    知识编辑已成为一种高效更新大语言模型（LLM）中嵌入知识的有前景的方法。它首先计算一个引导LLM产生新输出的理想目标隐状态，然后编辑模型参数，使原始隐状态映射到该目标。虽然现有大多数方法关注第二阶段的参数编辑以保护其他知识，我们发现目标隐状态的上游构建在分配模型有限的容量空间方面也起着重要作用。我们首先建立了参数空间中的干扰与目标隐状态之间的干扰之间的理论联系，表明两者呈正相关。基于这一洞察，我们提出了一种简单而有效的 $\ell_1$ 正则化方法，该方法修剪掉每个知识向量不必要的“触角”，仅保留必要的组成部分。

    arXiv:2611.12995v1 Announce Type: new  Abstract: Knowledge editing has emerged as a promising approach for efficiently updating embedded knowledge in large language models (LLMs). It first computes an ideal target hidden state that steers the LLM toward the new output, and then edits the model parameters so that the original hidden state is mapped to this target. While most existing methods focus on the second stage of parameter editing in preserving other knowledge, we find that the upstream construction of the target hidden state also plays an important role in allocating the model's limited capacity space. We first establish a theoretical connection between interference in the parameter space and interference between target hidden states, showing that the two are positively correlated. Building on this insight, we propose a simple yet effective $\ell_1$ regularization method that prunes away unnecessary ``tentacles'' of each knowledge vector, retaining only the essential components.
    
[^102]: ZeroPoint：轻量级基于能量的决策模型

    ZeroPoint: Lightweight Energy-Based Decision Model

    [https://arxiv.org/abs/2611.12992](https://arxiv.org/abs/2611.12992)

    本文提出ZeroPoint，通过引入一个对任意输入能量恒为零的“空答案”固定状态，为基于能量的决策模型提供显式的能量参考水平，使模型能够判断并拒绝“没有任何可接受选项”的情形。

    

    基于能量的模型（EBM）为观测变量与预测变量的每一个联合取值分配一个标量能量。在决策模型中，预测是从每个输入所附带的有限选项列表中选出的，因此推理可简化为比较各选项的能量，而训练则是降低正确选项相对于其他选项的能量。一个完整的决策还必须考虑没有任何选项可以接受的情况。然而，标准EBM学到的能量只定义了相对兼容性：给所有能量加上同一个依赖于输入的量，预测结果、吉布斯分布以及常用的训练损失都不会改变。因此，模型缺少一个可供判断任何选项是否合适的、学习得到的参考水平。我们提出ZeroPoint，这是一种将“空答案”作为模型自身状态的EBM，该状态对每个输入的能量均为零。这一固定状态为能量提供了显式的参考。

    arXiv:2611.12992v1 Announce Type: cross  Abstract: Energy-based models (EBMs) assign a scalar energy to every joint value of the observed and predicted variables. In decision models, the prediction is chosen from a finite list of options provided with each input. Inference then reduces to comparing the energies of the options, and training lowers the energy of the correct option relative to the others. A complete decision must also account for the possibility that no option is acceptable. However, the energies learned by a standard EBM define only relative compatibility: adding the same input-dependent quantity to all energies leaves the predictions, the Gibbs distribution, and the usual training losses unchanged. The model therefore has no learned reference level against which to decide whether any option fits. We propose ZeroPoint, an EBM that makes the empty answer a state of the model with energy zero for every input. This fixed state provides an explicit reference for the energies
    
[^103]: 追踪大语言模型中的心理测量推断

    Tracing Psychometric Inference in Large Language Models

    [https://arxiv.org/abs/2611.12937](https://arxiv.org/abs/2611.12937)

    该研究通过跨角色预测范式发现，大语言模型（尤其是较大规模的指令微调模型）在行为上重现并放大了人类心理量表的跨量表相关性，且其内部几何结构与人类心理测量结构对齐，表明模型在一定程度上确实内部表征了心理构念。

    

    大语言模型是在内部真正表征了心理构念，还是仅仅生成表面上的心理测量结构？我们通过跨角色范式回答这一问题，研究对象为来自两个家族（Llama 3 和 Qwen 2.5，参数量 0.5B-14B，包含基础版和指令版）的 14 个模型变体。给定真实个体（N = 272 名台湾大学生）在一个心理量表上的回答，模型需要预测其余六个量表上的回答。在行为层面，更大的模型变体能够重现并放大人类的跨量表相关性；即使在校正了七个人类量表的测量衰减之后，这种放大效应依然存在。随后我们检验这种结构是否在模型内部得到反映。对比方向分析发现，在较大的指令版变体中，模型内部的几何结构与人类的心理测量矩阵相一致，而基础版变体在该测试下几乎没有表现出这种对齐。跨模型来看，几何结构强度与行为放大程度相关。

    arXiv:2611.12937v1 Announce Type: new  Abstract: Do large language models represent psychological constructs internally, or do they only generate outputs that resemble psychometric structure? We address this question using a cross-persona paradigm across 14 model variants from two families (Llama 3 and Qwen 2.5, 0.5B-14B, base and instruct). Given responses on one psychological scale from real individuals (N = 272 Taiwanese university students), models predict responses on six other scales. At the behavioral level, larger variants reproduce and amplify human cross-scale correlations; amplification persists after correcting the seven-scale human correlations for measurement attenuation. We then examine whether this structure is reflected internally. Contrastive direction analysis finds geometry aligned with the human matrix in larger instruct variants, but little alignment in base variants under this assay. Across models, geometry strength is associated with behavioral amplification (r 
    
[^104]: 使用大语言模型智能体评估社交媒体平台上的亲社会干预措施

    Using LLM Agents to Evaluate Prosocial Interventions on Social Media Platforms

    [https://arxiv.org/abs/2611.12922](https://arxiv.org/abs/2611.12922)

    该研究基于OASIS框架构建了一个包含13,835个大语言模型智能体的仿真评估流水线，系统评估了三种提升经核实新闻曝光度的干预措施，发现将新闻提升至信息流显著位置的效果最佳，为在开展昂贵的人类实验前筛选干预方案提供了系统方法。

    

    社交媒体平台是许多用户获取信息的主要来源。尽管已有研究探索了各种干预措施来推广平台上的亲社会内容或降低有害内容的传播，但在开展成本高昂的人类受试者实验之前，很少有系统性的方法来评估相互竞争的干预设计方案。我们开发了一个基于OASIS框架构建的仿真评估流水线。我们生成了13,835个基于大语言模型的智能体，每个智能体都拥有独特的用户画像。在20个时间步内，我们将真实的Twitter/X帖子呈现给这些智能体，并让它们进行评论、点赞、转发或关注等操作。我们利用这些智能体系统地评估了三种提高经核实新闻曝光度的干预措施：添加内容、提升内容排名以及名人来源提示。所有干预措施都显著提高了新闻参与度，其中将新闻提升至信息流的显著位置产生了最大的效果。这些效果在不同性别、年龄和教育水平的智能体中均保持一致。

    arXiv:2611.12922v1 Announce Type: cross  Abstract: Social media platforms are a primary source of information for many users. Although research explores various interventions to promote prosocial content or demote harmful content on platforms, there are few systematic methods for evaluating competing intervention designs before conducting costly human-subject experiments. We develop a simulation-based evaluation pipeline built on the OASIS framework. We generate 13,835 LLM-based agents, each with a unique profile. Over 20 time steps, we expose them to real-world Twitter/X posts and let them comment, like, repost, or follow. We use those agents to systematically evaluate three interventions that increase exposure to verified news: adding content, upranking content, and celebrity source cues. All interventions significantly increase news engagement, with upranking news to prominent feed positions producing the largest effects. These effects hold across agents' gender, age, and education 
    
[^105]: IntentTwin：离线网页行为预测中候选支持去混杂与“理想评测到实际部署”的差距

    IntentTwin: Deconfounding Candidate Support and the Oracle-to-Deployment Gap in Offline Web Behavior Prediction

    [https://arxiv.org/abs/2611.12918](https://arxiv.org/abs/2611.12918)

    该论文揭示了离线网页行为预测基准中仅凭候选动作是否属于训练支持集即可“作弊”答对的混杂效应，提出支持可识别率（SIR）这一度量、支持均衡候选集构建方法以及免目标双路径提议器，从而真实刻画从理想评测条件到实际部署的性能差距。

    

    离线网页行为基准测试通常将示范动作插入一个有限候选集，然后测量模型的重排序性能。我们证明这种协议会造成“候选支持混杂”问题：当目标动作位于训练支持集之外、而所有干扰项均从该支持集中采样时，仅凭支持集归属信息就能识别出正确动作，完全无需建模用户行为。我们用“支持可识别率”（SIR）将这一效应形式化，构建了支持均衡的候选集，并评估了一种免目标的双路径候选提议器，该提议器将支持集检索与开放词表的DOM/语言模型提议相结合。在过滤后的OPeRA数据集（训练/验证/测试分别为4,282/469/992步）上，原始协议在目标为词表外（OOV）的样本上SIR达到1.0000，仅用支持集规则即可取得1.0000的OOV Exact@1。支持均衡化将SIR降至0.0000；此时最强的条件重排序器总体Exact@1为0.7214，在词表内（IV）样本上为0.7988，在OOV样本上为0.5668。免目标双路径系统达到P（摘要原文在此处截断）

    arXiv:2611.12918v1 Announce Type: new  Abstract: Offline web-behavior benchmarks frequently insert the demonstrated action into a finite candidate set and then measure reranking. We show that this protocol can create a candidate-support confound: when the target is outside training support while all distractors are sampled from that support, support membership identifies the gold action without modeling behavior. We formalize this effect with support-identifiability rate (SIR), construct support-balanced candidate sets, and evaluate a target-free dual-path proposer that combines support retrieval with open-vocabulary DOM/LM proposals. On filtered OPeRA (4,282/469/992 train/dev/test steps), the original protocol has SIR 1.0000 on target-OOV rows and a support-only rule reaches 1.0000 OOV Exact@1. Support balancing reduces SIR to 0.0000; the strongest conditional reranker obtains 0.7214 overall Exact@1, 0.7988 on IV rows, and 0.5668 on OOV rows. The target-free dual-path system reaches P
    
[^106]: 莫比乌斯交互过滤：面向多模态幽默检测的阶数选择性谱正则化

    M\"obius Interaction Filtering: Order-Selective Spectral Regularization for Multimodal Humor Detection

    [https://arxiv.org/abs/2611.12913](https://arxiv.org/abs/2611.12913)

    提出莫比乌斯交互过滤方法，通过阶数选择性谱正则化在保留高阶模态交互表示的同时调控其对预测的贡献，使多模态幽默检测的标签对齐信噪比提升1.62-2.79倍。

    

    多模态幽默检测需要联合解读语言、声音表达和视觉行为，同时表达性融合还必须调控不同阶数的交互以何种程度进入最终判断。我们提出莫比乌斯交互过滤，它学习一个模态子集预测函数，并通过其聚合结构施加阶数选择性的谱正则化。私有时间编码器和平衡的双向交互构建了有监督的单模态、双模态和三模态预测器，它们的均匀聚合恰好构成一个具有几何递减交互阶响应的莫比乌斯域滤波器。这一构造在保留高阶表示的同时调控其对预测的贡献。在UR-FUNNY数据集上的受控扰动实验显示，各交互阶数的标签对齐信噪比出现显著下降。所诱导的滤波器使最终比率相对……提升了1.62-2.79倍。

    arXiv:2611.12913v1 Announce Type: cross  Abstract: Multimodal humor detection requires interpreting language, vocal delivery, and visual behavior jointly. Expressive fusion must also regulate how interactions of different orders enter the final judgment. We propose M\"obius Interaction Filtering (MoIF), which learns a modality-subset prediction function and imposes order-selective spectral regularization through its aggregation structure. Private temporal encoders and balanced reciprocal interaction construct supervised unimodal, bimodal, and trimodal predictors. Their uniform aggregate is exactly a M\"obius-domain filter with a geometrically decreasing interaction-order response. This formulation retains higher-order representations while regulating their contribution to prediction. Controlled UR-FUNNY perturbations reveal a pronounced decline in label-aligned signal-to-noise ratio across interaction orders. The induced filter improves the final ratio by 1.62-2.79 times relative to th
    
[^107]: Brasilidade：衡量巴西身份认同的大语言模型评测基准

    Brasilidade: Measuring Brazilian Identity for Large Language Models

    [https://arxiv.org/abs/2611.12906](https://arxiv.org/abs/2611.12906)

    该论文提出了Brasilidade基准，通过涵盖语言、语法等四个领域的标准化任务评估大语言模型的巴西葡萄牙语能力，发现模型规模并不决定性能，且葡萄牙语专用模型的表现也未超过平均水平。

    

    这项工作提出了一个新的基准，旨在解决当前评估大语言模型巴西葡萄牙语能力的方法的局限性。我们介绍了Brasilidade，这是一种由分布在四个不同领域的标准化任务组成的评估方法，我们正在发布其第一个领域的任务，这些任务涉及巴西葡萄牙语的语言细微差别、语法和语言变体。结果表明，模型规模本身并不能决定性能；在所评估的测试协议中，几个中等规模的模型表现优于更大的模型。针对葡萄牙语的专用模型总体而言表现并未超出平均水平，尽管它们经过了针对性训练。不同的任务难度水平揭示了显著的细微差别：虽然有些任务对大多数模型来说很容易，但另一些任务则构成了极大的挑战，还有一些任务结果参差不齐，性能差异很大。这清楚地说明了需要能够应对巴西葡萄牙语独特挑战的基准。

    arXiv:2611.12906v1 Announce Type: cross  Abstract: This work presents a new benchmark that aims to address the limitations of current methods for evaluating LLMs in Brazilian Portuguese. We introduced Brasilidade, an assessment methodology comprised of standardized tasks distributed across four different domains, and we are making available the tasks from its first domain, which address nuances of the language, grammar, and variations of Brazilian Portuguese. The results show that model size, by itself, does not determine performance; several medium-sized models outperformed larger models in the protocols evaluated. Specific models for Portuguese, in general, did not perform above average, despite their targeted training. Different task levels revealed significant nuances: while some tasks proved easy for most models, others posed extreme challenges, and some yielded mixed results, with wide variations in performance. This clearly illustrates the need for benchmarks that address the li
    
[^108]: 可操作的反馈：利用临床文档中用户编辑数据提升模型能力

    Actionable Feedback: Improving Model Capabilities from User Edit Data on Clinical Documentation

    [https://arxiv.org/abs/2611.12904](https://arxiv.org/abs/2611.12904)

    该论文针对临床文档场景中用户编辑数据并非全部可操作的问题（如45%以上编辑包含模型输入不支持的信息），构建了临床医生编辑行为分类体系并开发了区分可操作反馈的方法，以避免微调导致模型产生幻觉。

    

    大语言模型越来越多地被部署到临床文档工作流程中，临床医生在使用前会审查并编辑模型的输出。原则上，这些临床医生的编辑代表了模型训练的一个可扩展的监督来源。然而在实践中，并非所有编辑都构成可操作的反馈：例如，通过审查来自AI临床文档产品实际部署的数据，我们发现超过45%的编辑后输出包含用户添加的、模型输入并不直接支持的信息。如果在这种编辑上进行简单微调，会有教会模型产生幻觉的风险。在本文中，我们开发了一种方法来应对这一挑战。鉴于已部署文档工具上的真实临床医生编辑数据并不公开，我们首先基于来自已部署AI临床文档产品的数据，构建了一个临床医生编辑行为的分类体系。然后，我们开发了一种生成……的方法

    arXiv:2611.12904v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly being deployed in clinical documentation workflows, where clinicians review and edit their outputs before use. In principle, these clinician edits represent a scalable source of supervision for model training. In practice, not all edits represent actionable feedback: for instance, on data drawn from real-world deployment of an AI clinical documentation product, our review suggests that over 45% of edited outputs contain user-added information that is not directly supported by the inputs to the model. Naively fine-tuning on such edits risks teaching the model to hallucinate. In this paper, we develop a method to address this challenge. Given that real clinician edits on deployed documentation tools are not publicly available, we first develop a taxonomy of clinician-editing behaviors, informed by data from a deployed AI clinical documentation product. Then, we develop a method for generating
    
[^109]: SELD：超越答案层面检测上下文压缩中的隐性证据丢失

    SELD: Detecting Silent Evidence Loss in Context Compression beyond the Answer

    [https://arxiv.org/abs/2611.12903](https://arxiv.org/abs/2611.12903)

    提出SELD——一种无需训练和标注的方法，通过比较阅读模型压缩前后对未说出口候选答案的偏好变化，检测上下文压缩中“答案不变但关键证据被删”的隐性证据丢失问题，并构建了SEL-Pairs评测基准。

    

    对于处理长输入的语言模型来说，上下文压缩如今已是常规操作，但压缩后的上下文只有在保留了回答所需的证据时才有用。现有的检测方法仅比较阅读模型的答案或其最可能的词汇，因而遗漏了一种常见的失败模式——隐性证据丢失：即关键证据被删除而答案保持不变，导致答案缺乏支撑，并对复用该上下文的其他模型构成风险。为解决这一挑战，我们提出了SELD（隐性证据丢失检测器），它无需训练也无需标注。其原理是：当证据丢失时，阅读模型在那些它未说出口的候选答案之间的偏好会发生偏移，即使其给出的答案不变，因此通过比较压缩前后这些偏好，能够发现答案检查无法察觉的证据丢失。我们还构建了SEL-Pairs基准，包含来自三个问答数据集的长度匹配的证据删除与对照删除样本对。定量与定性实验……（原文截断）

    arXiv:2611.12903v1 Announce Type: cross  Abstract: Context compression is now routine for language models with long inputs, but a compressed context is useful only if it keeps the evidence for the answer. Existing checks compare the reader model's answer or its most likely words, and thus miss a frequent failure, silent evidence loss: key evidence is removed while the answer stays the same, leaving the answer unsupported and risky for other models that reuse the context. To address this challenge, we propose SELD (Silent Evidence Loss Detector), which needs no training and no labels. Its principle is that when evidence is lost, a reader model's preferences among the answers it leaves unsaid shift even if its answer does not, so comparing these preferences before and after compression reveals losses that answer checks cannot see. We also build SEL-Pairs, a benchmark of length-matched evidence and control deletions from three question-answering datasets. Quantitative and qualitative expe
    
[^110]: TimeOmni-o1：时间序列基础模型与大语言模型的双向耦合，用于多模态推理与预测

    TimeOmni-o1: Bidirectionally Coupling Time Series Foundation Models and LLMs for Multimodal Reasoning and Forecasting

    [https://arxiv.org/abs/2611.12902](https://arxiv.org/abs/2611.12902)

    提出TimeOmni-o1，通过将预训练的时间序列基础模型与大语言模型双向耦合，实现文本与时间序列两种模态的联合读取与生成，从而统一完成多模态理解问答和未来值预测两类任务。

    

    来自现实世界系统的时间序列通常伴随着文本，例如关于影响该系统的事件的报告。针对这些序列的两类常见任务是理解（例如回答关于序列的问题）和预测其未来值。这两类任务都需要对序列和文本进行联合推理。此外，理解任务输出文本，而预测任务输出时间序列，因此要同时服务这两类任务，需要一个能够读取和生成两种模态的模型。大语言模型（LLM）将时间序列作为数字 token 读取，难以处理长序列。时间序列基础模型（TSFM）凭借大规模预训练具有出色的预测能力，但通常既不能读取也不能生成文本。时间序列语言模型虽然为 LLM 增加了时间序列模态，但预测仍在 LLM 侧进行，未能利用 TSFM 预训练的预测能力。我们提出了 TimeOmni-o1，它将预训练的 TSFM 与 LLM 双向耦合。在一个方向上，TSFM……（摘要不完整，后续内容缺失）

    arXiv:2611.12902v1 Announce Type: new  Abstract: Time series from real-world systems are often accompanied by text, such as a report on an event that affects them. Two common tasks on these series are understanding, e.g., answering questions about a series, and forecasting its future values. Both tasks require reasoning over the series and the text together. Moreover, understanding outputs text while forecasting outputs a time series, so serving both requires a model that reads and generates both modalities. Large language models (LLMs) read time series as digit tokens and handle long series poorly. Time series foundation models (TSFMs) forecast well from large-scale pretraining but typically neither read nor generate text. Time series language models add a time series modality to the LLM but forecast on the LLM side, leaving the TSFM's pretrained forecasting ability unused. We present TimeOmni-o1, which couples a pretrained TSFM and an LLM bidirectionally. In one direction, the TSFM e
    
[^111]: 反思尖峰：通过步感知解码缓解多模态扩散语言模型中的幻觉

    Rethinking in Spikes: Mitigating Hallucinations in MDLMs with Step-Aware Decoding

    [https://arxiv.org/abs/2611.12901](https://arxiv.org/abs/2611.12901)

    提出了StepRefine，一种即插即用的步感知解码策略，通过检测去噪过程中的熵尖峰并在离散去噪与潜在精炼之间动态切换，从而缓解多模态扩散语言模型的幻觉问题。

    

    多模态扩散语言模型（MDLMs）的最新进展通过迭代精炼展现出了强大的全局建模能力。我们观察到，掩码不确定性在迭代去噪过程中呈现总体下降趋势，而少数去噪步骤表现出与幻觉密切相关的局部熵尖峰。我们认为，可以通过测量每个去噪步骤的熵与其相邻时间窗口的偏差来识别异常步骤。我们假设离散词元的承诺机制会丢弃掩码位置上的分布信息，从而限制了候选语义与上下文在熵尖峰期间的持续交互。为此，我们提出了StepRefine，一种利用语义上下文实现可靠去噪的即插即用解码策略。我们方法的核心在于在离散去噪与潜在精炼之间进行步感知切换。该模型采用……

    arXiv:2611.12901v1 Announce Type: cross  Abstract: Recent advancements in multimodal diffusion language models (MDLMs) have exhibited strong global modeling capabilities through iterative refinement. We observe that mask uncertainty follows an overall downward trend during iterative denoising, while a few denoising steps exhibit localized entropy spikes closely associated with hallucinations. We argue that anomalous steps can be identified by measuring the entropy deviation of each denoising step from its neighboring temporal window. We hypothesize that discrete token commitment discards the distributional information at masked positions, limiting the continued interaction between candidate semantics and the context during entropy spikes. With this goal, we present StepRefine, a plug-and-play decoding strategy that leverages semantic context to achieve reliable denoising. The core of our method lies in a step-aware switch between discrete denoising and latent refinement. The model empl
    
[^112]: 从相关帖文到可归因文本片段：用于抑郁症问卷推断的显式症状证据

    From Relevant Posts to Attributable Spans: Explicit Symptom Evidence for Depression Questionnaire Inference

    [https://arxiv.org/abs/2611.12900](https://arxiv.org/abs/2611.12900)

    提出SREE证据驱动流水线，通过提取第一人称、症状特异的文本片段，用设施选址思想筛选代表性证据，并采用证据门控推断，从而更可靠地从社交媒体历史推断抑郁症问卷结果。

    

    纵向的社交媒体历史记录为评估抑郁症提供了机会，有望支持对标准化抑郁症问卷回答的推断。然而，这一任务所需的证据既要在语义上与症状相关，又要能够归因于用户本人，同时还需足够具体以支撑单个问卷条目的推断。现有方法依赖语义相似度来检索帖子，这可能引入误导性或不可归因的证据，尤其是在包含多种症状的帖子和讨论他人经历的场景中。我们提出了SREE（Symptom Retrieval and Evidence-Grounded InfErence，症状检索与证据支撑推断），这是一个基于证据的流水线：它提取第一人称且症状特定的文本片段，使用受设施选址启发的目标函数选择代表性证据子集，并在所选证据上执行证据门控推断。我们在BDI-Sen数据集上评估证据检索效果以及证据条件下的问卷推断……（原文摘要至此被截断）

    arXiv:2611.12900v1 Announce Type: new  Abstract: Longitudinal social-media histories offer an opportunity to assess depression, potentially supporting the inference of standardized depression questionnaire responses. However, this task requires evidence that is semantically relevant to a symptom and attributable to the user, while being sufficiently specific to support an individual questionnaire item. Existing approaches rely on semantic similarity to retrieve posts, which can introduce misleading or non-attributable evidence, particularly in multi-symptom posts and discussions of others' experiences. We propose SREE (Symptom Retrieval and Evidence-Grounded InfErence), an evidence-grounded pipeline that extracts first-person and symptom-specific text spans, selects a representative subset using a Facility Location-inspired objective, and performs Evidence-Gated Inference over the selected evidence. We evaluate evidence retrieval on BDI-Sen and evidence-conditioned questionnaire infere
    
[^113]: MNN-Pocket-0.3B 技术报告

    MNN-Pocket-0.3B Technical Report

    [https://arxiv.org/abs/2611.12311](https://arxiv.org/abs/2611.12311)

    MNN-Pocket-0.3B 是一个面向端侧部署的 3 亿参数紧凑语言模型，通过架构、数据、训练流程和推理系统的协同设计，实现了仅 173 MB 的 INT4 体积、最高 466 tokens/s 的解码速度和 32K 上下文支持。

    

    本技术报告介绍了 MNN-Pocket-0.3B，这是一个紧凑的 3 亿参数稠密语言模型，专为高效的端侧部署而设计，支持最长 32K token 的上下文长度。其 INT4 部署体积仅 173 MB，在 iPad Pro M5（Metal）上的解码吞吐量高达每秒 466 个 token，可无缝集成到智能手机、智能眼镜等边缘设备的应用中。MNN-Pocket-0.3B 通过架构、数据、训练流程和推理系统的协同设计开发而成。硬件感知的架构选择和依赖感知的词表剪枝在模型能力与端侧内存及计算约束之间取得了平衡。一个覆盖约 4.1 万亿高质量中英文 token 的四阶段预训练流程赋予模型广泛的知识和推理能力。随后是一个包含指令微调等内容的三阶段后训练流程（原文此处截断）。

    arXiv:2611.12311v2 Announce Type: replace  Abstract: This technical report presents MNN-Pocket-0.3B, a compact 300M-parameter dense language model designed for efficient on-device deployment and supporting context lengths of up to 32K tokens. With an INT4 deployment footprint of only 173 MB and decoding throughput of up to 466 tokens/s on iPad Pro M5 (Metal), it can be seamlessly integrated into applications on edge devices such as smartphones and smart glasses. MNN-Pocket-0.3B is developed through the co-design of its architecture, data, training pipeline, and inference system. Hardware-aware architecture selection and dependency-aware vocabulary pruning balance model capability with on-device memory and compute constraints. A four-stage pre-training pipeline covering approximately 4.1 trillion high-quality Chinese and English tokens equips the model with broad knowledge and reasoning capabilities. This is followed by a three-stage post-training pipeline comprising instruction fine-tu
    
[^114]: FinCutBench：一个用于衡量大语言模型金融知识时间边界的持续更新基准

    FinCutBench: A Continually Updated Benchmark for Measuring the Temporal Boundaries of LLM Financial Knowledge

    [https://arxiv.org/abs/2611.11588](https://arxiv.org/abs/2611.11588)

    该论文提出FinCutBench，一个每月持续更新的中英双语金融知识基准，通过按月构建知识画像并定位“FinCut边界”，来衡量大语言模型金融知识的时间截止点，以检测金融回测中的参数化前视偏差。

    

    在金融回测和其他时点任务中，外部输入可以按日期进行限制，但模型参数可能保留了之后才披露的金融事实，从而引入参数化前视偏差。我们提出了FinCutBench，这是一个每月持续更新的金融知识基准，用于衡量固定模型的金融知识在日历时间上能够延伸到多远。当前数据集包含9,000道中英文四选一多项选择题，时间跨度从2023年1月至2026年9月，涵盖中国和美国市场以及四类金融信息。我们利用所测试事实公开的大致月份来构建月度知识画像。在画像通过衰减测试和节点检查后，我们将FinCut边界定义为拟合画像达到其较低水平的第一个月份，并报告重采样下估计日期的稳定性。边界日期与定位

    arXiv:2611.11588v2 Announce Type: replace  Abstract: In financial backtesting and other point-in-time tasks, external inputs can be restricted by date, but model parameters may retain financial facts disclosed later, introducing parametric look-ahead bias. We introduce FinCutBench, a continually updated monthly financial knowledge benchmark for measuring how far a fixed model's financial knowledge extends over calendar time. The current dataset contains 9,000 Chinese and English four-option multiple-choice questions from January 2023 to September 2026, covering Chinese and U.S. markets and four categories of financial information. We construct monthly knowledge profiles using the approximate months in which the tested facts became public. After a profile passes the decline test and knot checks, we define the FinCut Boundary as the first month when the fitted profile reaches its lower level, and report the stability of the estimated date under resampling. Boundary dates and localization
    
[^115]: 扩散语言模型何时应重新考虑其承诺？

    When Should Diffusion Language Models Reconsider Their Commitments?

    [https://arxiv.org/abs/2611.11551](https://arxiv.org/abs/2611.11551)

    该论文提出PURL框架，通过联合学习编辑选择、取消掩码选择和词元预测三个决策，使掩码扩散模型能够在并行快速生成的同时有选择地修正早期不一致的词元承诺，且推理时三个决策仅需一次共享前向传播即可完成。

    

    掩码扩散模型以通过并行解码实现快速文本生成而闻名。然而，相互独立的并行采样可能会在早期就“承诺”不一致的预测结果，而关于如何有选择地修正这些早期的词元承诺，目前所知甚少。在这项工作中，我们为掩码扩散模型提出了计划式取消掩码与修正联合学习（Planned Unmasking-and-Revision Joint Learning, PURL），这是一个基于训练的框架，无缝地将并行生成与选择性词元修正统一起来。我们推导出了一个归一化的联合转移，其似然可以分解为编辑选择、取消掩码选择和词元预测三个部分。这些组件在根据模型自身取消掩码策略构建的部分掩码序列上进行联合训练。在推理阶段，所有三个决策均由一次共享的前向传播所支持，使模型能够解码被掩码的词元并修正带有噪声的词元。

    arXiv:2611.11551v2 Announce Type: replace-cross  Abstract: Masked Diffusion Models (MDMs) are well known for fast text generation via parallel decoding. However, independent parallel sampling can commit to inconsistent predictions, and less is known about how to selectively revise these early token commitments. In this work, we introduce $\underline{\mathrm{P}}$lanned $\underline{\mathrm{U}}$nmasking-and-$\underline{\mathrm{R}}$evision Joint $\underline{\mathrm{L}}$earning (PURL) for MDMs, a training-based framework that seamlessly unifies parallel generation with selective token revision. We derive a normalized joint transition whose likelihood factorizes into edit selection, unmask selection, and token prediction. These components are trained jointly on partially masked sequences constructed according to the model's own unmasking policy. During inference, all three decisions are supported by a single shared forward pass, enabling the model to decode masked tokens and revise noisy tok
    
[^116]: ScrapeBench：评估AI智能体在网站抓取中的法律合规性

    ScrapeBench: Evaluating Legal Compliance of AI Agents in Website Scraping

    [https://arxiv.org/abs/2611.11306](https://arxiv.org/abs/2611.11306)

    该论文提出首个评估AI智能体网站抓取法律合规性的基准ScrapeBench，发现大多数智能体在抓取明确禁止抓取的网站时合规率低于5%，即使主动检查了ToS和robots.txt也依然如此，而在允许抓取的网站上却有时会错误地拒绝抓取。

    

    部署在开放网络上的AI智能体越来越有能力执行可能违反合同条款和计算机安全法律的行为。在本文中，我们介绍了ScrapeBench，这是首个用于实证评估AI智能体在网站抓取策略（即服务条款ToS和robots.txt指令）方面法律合规性的智能体基准测试。在广泛使用的闭源权重与开源权重AI智能体中，我们发现，当被指示抓取我们样本中1,626个明确禁止抓取的热门网站时，大多数智能体表现出极低的合规率（<5%）。此外，我们观察到，即使AI智能体主动检查了适用的ToS和robots.txt，它们仍然表现出较低的合规率。相反，在我们样本中269个不禁止抓取的网站上，我们观察到智能体有时会拒绝抓取，尽管这样做是合法的，例如Claude O（原文在此处截断）

    arXiv:2611.11306v2 Announce Type: replace-cross  Abstract: AI agents deployed on the open web are becoming more capable of performing actions that may breach contractual provisions and violate computer security laws. In this paper, we introduce ScrapeBench, the first agentic benchmark for empirically evaluating the legal compliance of AI agents with website scraping policies, namely terms of service (ToS) and $\texttt{robots.txt}$ directives. Across widely used closed-weight and open-weight AI agents, we find that most agents exhibit low compliance rates (< 5%) when instructed to scrape websites in our sample of 1,626 popular websites that explicitly prohibit scraping. In addition, we observe that even where AI agents actively check the applicable ToS and $\texttt{robots.txt}$, they continue to exhibit low compliance rates. Conversely, in our sample of 269 websites that do not prohibit scraping, we observe that agents sometimes refuse to scrape despite it being lawful, such as Claude O
    
[^117]: WarBias：一个关于境内流离失所者和退伍军人刻板印象的双语大语言模型基准

    WarBias: A Bilingual LLM Benchmark of Stereotypes About Internally Displaced People and Veterans

    [https://arxiv.org/abs/2611.11284](https://arxiv.org/abs/2611.11284)

    该论文提出了首个聚焦退伍军人和境内流离失所者这两个被主流偏见评估忽视群体的双语（乌克兰语-英语）大语言模型偏见评测基准WarBias，包含120条刻板印象主张、每种语言1,007个句子三元组和240道多选题，填补了语言模型社会偏见评估的空白。

    

    本文包含一些关于退伍军人和流离失所者的刻板印象示例，可能具有冒犯性或令人不适。针对语言模型的社会偏见基准通常考察性别、种族和宗教等类别，这些类别往往源自法律保护特征清单。退伍军人和境内流离失所者不在这些清单之列，而关于他们的刻板印象——可能损害其获得就业、住房和社会参与的机会——在很大程度上仍未被纳入语言模型评估。我们提出了WarBias，一个针对这两个群体的乌克兰语-英语刻板印象基准。WarBias基于研究和监测报告构建了包含120条刻板印象主张的分类体系，每种语言包含1,007个“刻板印象-反刻板印象-无关”句子三元组以及240道多选题。乌克兰语三元组由在盲测人类试点中选出的一款乌克兰语语言模型生成，并由至少两名独立（原文摘要在此处截断）

    arXiv:2611.11284v2 Announce Type: replace  Abstract: This paper contains examples of stereotypes about veterans and displaced people that may be offensive or distressing. Social bias benchmarks for language models commonly examine categories such as gender, race, and religion, often drawn from lists of legally protected characteristics. Veterans and internally displaced people (IDPs) fall outside these lists, and stereotypes about them, which can undermine access to employment, housing, and social participation, remain largely absent from language model evaluation. We introduce WarBias, a Ukrainian-English benchmark of stereotypes about these two groups. Built from a taxonomy of 120 stereotype claims grounded in research and monitoring reports, WarBias contains 1,007 stereotype-counter-stereotype-unrelated sentence triplets and 240 multiple-choice questions per language. A Ukrainian language model selected in a blinded human pilot generated the Ukrainian triplets. At least two independ
    
[^118]: 机器听觉：音频大语言模型能否超越人类听觉感知进行泛化？

    Machine Hearing: Can Audio LLMs Generalize Beyond Human Auditory Perception?

    [https://arxiv.org/abs/2611.10693](https://arxiv.org/abs/2611.10693)

    提出了机器听觉基准（MHB），通过涵盖7个领域、13个任务、采样率跨越约10^-5至10^6 Hz的6,128个实例，系统评估音频大语言模型能否超越人类听觉范围实现更广泛的机器听觉泛化能力。

    

    多模态大语言模型在感知物理世界方面取得了显著进展，但其感知范围仍主要局限于以人类为中心，聚焦于人类能够直接看到或听到的事物。这一进展引发了对超越人类极限的感知的展望，例如将听觉扩展到人类无法听见的声音。更广泛地说，物理世界充满了波、振动和振荡，其中只有一小部分能被人类直接听到。这促使人们将多模态大语言模型扩展为更广泛形式的机器听觉，而这一方向在很大程度上仍待探索。为研究这一方向，我们引入了机器听觉基准（Machine Hearing Benchmark, MHB），包含跨13个任务和7个领域的6,128个实例。MHB涵盖了高度异构的输入模式，采样率从约10^-5到10^6 Hz，序列长度从10^1到10^6个样本，并同时包括单通道和多通道输入。

    arXiv:2611.10693v2 Announce Type: replace-cross  Abstract: Multimodal LLMs have made remarkable progress in perceiving the physical world, while their perceptual scope remains largely human-centric, focusing on what humans can directly see or hear. This progress raises the prospect of perception beyond human limits, such as extending hearing to sounds that are inaudible to humans. More broadly, the physical world is full of waves, vibrations, and oscillations, only a small fraction of which are directly audible to humans. This invites the extension of multimodal LLMs toward a broader form of machine hearing, which remains largely underexplored. To investigate this direction, we introduce the Machine Hearing Benchmark (MHB), comprising 6,128 instances across 13 tasks and seven domains. MHB spans highly heterogeneous input regimes, with sampling rates from approximately $10^{-5}$ to $10^6$ Hz, sequence lengths from $10^1$ to $10^6$ samples, and both single- and multi-channel inputs. Eval
    
[^119]: CulturalCrosswords：答案约束与网格填写的多语言基准测试

    CulturalCrosswords: A Multilingual Benchmark of Answer Constraints and Grid Completion

    [https://arxiv.org/abs/2611.10044](https://arxiv.org/abs/2611.10044)

    该论文构建了覆盖42种语言、53个填字游戏的多语言基准CulturalCrosswords，发现长度提示和网格约束等答案约束能显著提升大型语言模型解答文化谜题时答案的准确性与一致性。

    

    arXiv:2611.10044v2 公告类型： replace-cross 摘要：当一个谜语有多个符合条件的答案时，精确匹配的评分方式可能会误判有效的回答。CulturalCrosswords 测试答案约束如何影响大型语言模型（LLM）解决文化谜题的能力，这类谜题的答案依赖于特定社区的知识。我们从42种语言的众包谜题中构建了53个填字游戏，并在每个线索集的五个版本上评估了11种LLM配置：普通批次、带长度信息的批次、带长度和部分答案提示的批次，以及要求提供答案列表或完成整个网格的填字游戏形式。在962个存在备选答案的线索上，长度信息使模型与参考答案的一致性提高了22.8个百分点，而当备选答案也计为正确时仅提高8.4个百分点。因此，部分收益似乎来自模型选择了所需的答案形式。填字游戏形式的输入使所有交叉点答案兼容的比例提高了9.7个百分点；而要求完成整个网格则揭示了显著的……

    arXiv:2611.10044v2 Announce Type: replace-cross  Abstract: When several answers fit a riddle, exact-match scoring can penalise a valid response. CulturalCrosswords tests how answer constraints affect large language models (LLMs) solving cultural riddles, whose solutions draw on knowledge specific to a community. We constructed 53 crosswords from crowdsourced riddles in 42 languages and evaluated 11 LLM configurations on five versions of each clue set: plain batches, batches with lengths, batches with lengths and partial-answer hints, and crosswords requesting an answer list or a completed grid. On 962 clues with alternative answers, lengths improve agreement with the reference by 22.8 percentage points, compared with 8.4 points when accepted alternatives also receive credit. Part of the gain therefore appears to come from selecting a required answer form. Crossword input increases the share of all crossings with compatible answers by 9.7 points; requiring a completed grid reveals large
    
[^120]: 计数事件，而非看见事件：诊断与修复视频大语言模型中的读取失败

    Counting Events, Not Seeing Them: Diagnosing and Repairing the Readout Failure in Video LLMs

    [https://arxiv.org/abs/2611.09949](https://arxiv.org/abs/2611.09949)

    该论文发现视频大语言模型物体计数失败的根源并非缺少物体恒常性，而是计数信号已存在于内部表征中却未被正确读取，并通过片段定位监督与离散事件符号辅助通道两种修复方法，将同一物体事件计数准确率从46.7%大幅提升至65.0%。

    

    视频大语言模型在计算物体穿过屏幕的次数时表现明显失败，这一失败被归因于缺失物体恒常性。利用基于构造的合成测试平台和经过验证的标签，我们表明这种归因是不完整的：计数信号其实存在，但未被读取——线性探针能够以93%的准确率解码每个时间窗口内的物体存在，而全局计数解码则退化为对持续时间的混淆。同期研究在重复计数任务中报告了相同的表征-输出分离现象；我们则证明，在“离场-重现”事件计数任务中，这种失败是可以修复的：两种针对读取过程的修复方法（片段定位监督和离散事件符号辅助通道）在随机水平为33%的基准上，将同一物体计数准确率从46.7%提升至65.0%（oracle上限为66.7%），并在7B至30B规模的多个系统中均有效。迁移到独立的工厂场景毫无损失，而一项测量审计则暴露出五个曾掩盖问题边界的数据完整性缺陷。一项真实视频审计（REA

    arXiv:2611.09949v2 Announce Type: replace-cross  Abstract: Video LLMs fail conspicuously at counting how many times an object crosses the screen, a failure attributed to missing object permanence. Using construction-grounded synthetic testbeds with validated labels, we show this attribution is incomplete: the count signal is present but unread - linear probes decode per-window presence at 93% while global count decoding collapses to a duration confound. Concurrent work reports the same representation-output dissociation for repetition counting; we show that in exit-and-reappear event counting it is repairable: two readout-directed repairs (episode-localization supervision and a discrete event-symbol side channel) raise same-object counting from 46.7% to 65.0% (oracle 66.7%) on a chance-33% benchmark across 7B-30B systems. Transfer to an independent factory costs nothing, and a measurement audit exposes five data-integrity failures that had obscured the boundary. A real-video audit (REA
    
[^121]: BEiTScore++：基于BEiT-3与共形化分位数回归的无参考图像描述评估

    BEiTScore++: Reference-free Image Captioning Evaluation with BEiT-3 and Conformalized Quantile Regression

    [https://arxiv.org/abs/2611.09505](https://arxiv.org/abs/2611.09505)

    提出BEiTScore++，通过共形化分位数回归为无参考图像描述评估指标引入共形校准的置信区间，在单次前向传播中同时实现准确评分与可靠的不确定性量化。

    

    可靠的不确定性估计对于将评估指标部署到现实环境中至关重要，因为现实中输入歧义和领域偏移都很常见。在这项工作中，我们提出了BEiTScore图像描述评估指标的不确定性感知扩展版本，在对齐分数之外还能产生经过共形校准的置信区间。我们的方法对人类评判的预测分布进行建模，能够在单次前向传播中同时实现准确评分和有原则的不确定性量化。我们在多个带有人工标注评分的基准上评估了BEiTScore++指标，结果表明它保持了BEiTScore的排序相关性，同时在大多数基准上产生的区间宽度比蒙特卡洛dropout更贴近预测误差。估计的不确定性在图像退化条件下以及域外艺术图像上也会随之增大，这使得BEiTScore++成为一个实用的不确定性感知评估工具。

    arXiv:2611.09505v1 Announce Type: cross  Abstract: Reliable uncertainty estimation is important for deploying evaluation metrics in real-world settings, where both input ambiguity and domain shifts are common. In this work, we propose an uncertainty-aware extension of the BEiTScore image captioning metric, producing conformally calibrated confidence intervals alongside alignment scores. Our approach models predictive distributions over human judgments, enabling both accurate scoring and principled uncertainty quantification in a single forward pass. We evaluate our BEiTScore++ metric across multiple benchmarks with human-annotated ratings, showing that it preserves the ranking correlation of BEiTScore, while producing intervals whose width tracks the prediction error more closely than Monte Carlo dropout on most benchmarks. The estimated uncertainty also grows under image degradation and on out-of-domain artwork images, making BEiTScore++ a practical tool for uncertainty-aware evaluati
    
[^122]: PathXAI：基于可迁移组织分割的证据引导结构化病理报告生成

    PathXAI: Evidence-guided structured pathology reporting by transferable tissue segmentation

    [https://arxiv.org/abs/2611.09117](https://arxiv.org/abs/2611.09117)

    PathXAI通过可迁移的组织分割信息结合H&E图像与领域知识来引导诊断推理和结构化病理报告生成，在诊断一致性和形态学描述准确性上显著优于纯H&E方法。

    

    病理报告需要将跨区域和跨切片的组织观察转化为具体诊断及支持性的形态学描述。然而，利用明确的组织构成测量来指导基于H&E（苏木精-伊红染色）图像的报告生成仍鲜有探索。因此，我们开发了PathXAI，该方法将分割得到的组织信息与H&E图像及可选的领域知识相结合，以指导诊断推理、针对性复查和结构化报告撰写。其分割组件在内部和外部测试集上的平均Dice系数均超过基线方法。在1,200例公开病例和151例私有病例的自动化报告评估中，PathXAI与参考诊断及已记录形态学的一致性高于基于H&E的对比方法。在公开评估队列上，PathXAI-Report在癌家族匹配率上超过Qwen + H&E 4.4个百分点，在发现项F1分数上超过5.7个百分点。

    arXiv:2611.09117v1 Announce Type: cross  Abstract: Pathology reporting requires translating tissue observations across regions and slides into specific diagnoses and supporting morphological descriptions. However, the use of explicit tissue-composition measurements to guide H&E-based report generation remains underexplored. Thus, we developed PathXAI, which combines segmentation-derived tissue information with H&E images and optional domain knowledge to guide diagnostic reasoning, targeted reinspection and structured report writing. Its segmentation component exceeded baselines in mean Dice on internal and external tests. In automated report evaluation on 1,200 public and 151 private cases, PathXAI achieved higher agreement with reference diagnoses and documented morphology than the H&E-based comparators. On the public evaluation cohort, PathXAI-Report exceeded Qwen + H&E in cancer family match rate by 4.4 percentage points and in finding F1 by 5.7 percentage points. PathXAI generates 
    
[^123]: 迈向基础决策模型：Drex

    Towards Foundation Decision Models: Drex

    [https://arxiv.org/abs/2611.07854](https://arxiv.org/abs/2611.07854)

    提出Drex系列开源决策模型（8B至35B参数），通过监督微调与调整反馈强化学习（RLAF）训练，无需自回归解码即可对用户提供的选项直接输出概率分布，从而降低生产环境中的成本、延迟与答案不一致性。

    

    使用大型语言模型的自回归方法（如长链式思维推理）已成为新AI应用的默认方案，因为它们能在广泛任务上取得更高的准确率。然而在生产环境中，它们会给出不一致的答案、需要解析输出，并增加成本和延迟。决策模型解决了这一问题：给定一个问题和用户提供的选项集合，它们无需自回归解码即可返回各选项的概率分布。我们提出Drex，一个由四个开源决策模型组成的系列，参数量分别为8B（DLM）、9B（v1.5）、35B总参数/3B激活参数（v2.0）以及27B（Max）。每个请求由一个状态（文档、记录或智能体上下文，可选包含图像）以及关于该状态的问题组成，每个问题要求在选项中做出选择、进行是/否判断，或在有序量表上进行评分。训练结合了监督微调与来自调整反馈的强化学习（RLAF），并加入了视觉训练……

    arXiv:2611.07854v1 Announce Type: new  Abstract: Autoregressive approaches with large language models, such as long chain-of-thought reasoning, have become the default for new AI applications because they reach higher accuracy on a wide range of tasks. In production, however, they give inconsistent answers, require parsing of outputs, and add cost and latency. Decision models address this: given a question and a set of options supplied by the user, they return a probability distribution over the options without autoregressive decoding. We present Drex, a family of four open decision models with 8B (DLM), 9B (v1.5), 35B with 3B active (v2.0), and 27B (Max) parameters. Each request consists of a state (a document, record, or agent context, optionally with images) and questions about it, each asking for a choice among options, a yes/no judgment, or a score on an ordered scale. Training combines supervised fine-tuning with reinforcement learning from adjusted feedback (RLAF), plus vision t
    
[^124]: ALTO：非洲与黎凡特地区技术助长的基于性别的暴力语料库

    ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus

    [https://arxiv.org/abs/2611.06706](https://arxiv.org/abs/2611.06706)

    该论文通过与约旦、西非和肯尼亚的民间社会组织合作，以社区主导的众包方式构建了首个涵盖黎凡特阿拉伯语、非洲法语和斯瓦希里语的技术助长性别暴力标注语料库ALTO，填补了现有性别伤害检测研究长期偏向全球北方语言的空白。

    

    技术助长的基于性别的暴力（TFGBV）是一种普遍存在的网络危害，通过多种直接和隐蔽的形式影响女性和性别少数群体。当前用于性别伤害检测的模型和标注数据集主要被来自全球北方语言的数据以及用这些语言训练的模型所主导。我们与来自约旦、法语西非和肯尼亚的民间社会组织（CSOs）合作，开发了一种社区主导的方法，通过众包方式收集黎凡特阿拉伯语、非洲法语和斯瓦希里语中符合当地文化和语言特点的TFGBV实例。我们使用主动学习对众包数据进行标注，构建了一个规模虽小但性能出色的数据集（N=3022），命名为ALTO，用于改进这三种语言的TFGBV检测。通过基准测试和微调实验，我们表明现有的有害内容检测模型在ALTO数据集上的TFGBV召回率较低，而经过微调后性能有所提升。我们发现模型往往无法识别隐蔽的……

    arXiv:2611.06706v1 Announce Type: cross  Abstract: Tech-facilitated gender based violence (TFGBV) is a pervasive online harm affecting women and gender minority groups through its many direct and covert forms. Current models and annotated datasets for gendered harm detection are dominated by data from, and models trained in, Global North languages. In collaboration with civil society organizations (CSOs) from Jordan, Francophone West Africa, and Kenya, we developed a community-led approach to crowdsource culturally and linguistically relevant instances of TFGBV in Levantine Arabic, African French, and Swahili. We annotated the crowdsourced data using active learning to develop a small but performant dataset (N=3022) called ALTO to improve TFGBV detection in the three languages. Through benchmarking and fine-tuning experiments, we show that existing harmful content detection models show poor TFGBV recall within ALTO and improve with fine-tuning. We found that models often miss covert ha
    
[^125]: 解读“为什么”：在光与量子比特上实现具有精确词证据的玻恩测量注意力机制

    Reading Why: Born-Measured Attention with Exact Word Evidence on Light and Qubits

    [https://arxiv.org/abs/2611.05355](https://arxiv.org/abs/2611.05355)

    该论文提出一种由玻恩测量构成的变换器（玻恩归一化注意力、测量乘积与玻恩读出），使类别概率成为词振幅的多项式，从而可通过有限求积无误差地精确计算每个词的积分梯度证据，并在 e-SNLI 上取得与同规模 softmax 变换器相当的准确率。

    

    当语言模型给出答案时，我们通常只能估计其“为什么”。我们构建了一个变换器，其中每一步都是一次玻恩测量——注意力为在每一行上归一化的 $|\langle q|k\rangle|^4$，用测量乘积取代前馈非线性，并采用玻恩读出。将其分母保持在输入时的取值，每个类别概率都是词振幅的多项式，因此其积分梯度证据可以通过有限求积精确计算，不存在求积误差：各词的贡献之和恰好等于类别概率超出无信息份额的部分，误差达到 $10^{-11}$ 或更好。在 e-SNLI 自然语言推理任务上，该模型在嵌入层之外约 98k 参数的规模下达到 76.98 ± 0.17% 的测试准确率，与同规模的 softmax 变换器持平（76.62 ± 0.36%）。删除证据排名第一的词会在 45% 的测试样本对中改变答案，而这接近任何单个词删除所能改变答案的比例上限（53%）。

    arXiv:2611.05355v2 Announce Type: replace-cross  Abstract: When a language model answers, we can usually only estimate why. We build a transformer in which every step is a Born measurement - attention $|\langle q|k\rangle|^4$ normalised over each row, a measured product in place of the feed-forward nonlinearity, and a Born readout. With its denominators held at their values for the input, each class probability is a polynomial in the word amplitudes, so its integrated-gradient evidence is computed by a finite quadrature with no quadrature error: the words' contributions sum to the class probability above its no-information share, to $10^{-11}$ or better. On e-SNLI natural language inference, with about 98k parameters outside the embedding, the model reaches $76.98 \pm 0.17$% test accuracy, level with a softmax transformer of the same size ($76.62 \pm 0.36$%). Deleting the word its evidence ranks first changes the answer in 45% of test pairs, out of the 53% any single deletion can chang
    
[^126]: FindLink：基于来源锚定前提学习评估科学关联

    FindLink: Learning to Assess Scientific Connections from Source-Grounded Premises

    [https://arxiv.org/abs/2611.02545](https://arxiv.org/abs/2611.02545)

    该论文提出 GI-TraceBench 基准与 FindLink 两阶段框架，通过构建来源锚定前提并进行结构化推理，结合确定性编译器反馈，来评估不同研究条件下科学发现之间的关联。

    

    arXiv:2611.02545v1 公告类型：新论文。摘要：基于文献的科学发现需要评估在不同研究条件下所得发现之间的关联。来源归因使单个发现可追溯，但并不能确定它们共同支持什么。我们提出 GI-TraceBench，一个包含 10,282 个问题的胃肠肿瘤学基准，通过共享的论文、主张和证据片段记录，统一了来源溯源问答、知识发现和假设评估三类任务。我们进一步提出 FindLink，一个面向来源锚定科学推理的两阶段框架。前提锚定阶段构建带有明确研究背景和支持状态的来源绑定前提；结构化推理阶段将这些前提组织成同时表示支持关系与未决条件的支持计划，并利用确定性编译器反馈生成以证据为条件的答案。该编译器验证结构与来源绑定的一致性，但……（原文在此处截断）

    arXiv:2611.02545v1 Announce Type: new  Abstract: Scientific discovery from the literature requires assessing connections among findings obtained under different study conditions. Source attribution makes individual findings traceable, but does not determine what they jointly support. We introduce GI-TraceBench, a benchmark of 10,282 questions in gastrointestinal oncology that unifies provenance question answering, knowledge discovery, and hypothesis evaluation through shared Paper, Claim, and Evidence Span records. We further propose FindLink, a two-stage framework for provenance-grounded scientific reasoning. Premise Grounding constructs source-bound premises with explicit study context and support status. Structured Reasoning organizes these premises into support plans that represent both supported relations and unresolved conditions, then generates evidence-conditioned answers using deterministic compiler feedback. The compiler verifies structural and source-binding consistency, but
    
[^127]: Muon 使第一个混合专家层更难保持负载均衡

    Muon Makes the First Mixture-of-Experts Layer Harder to Balance

    [https://arxiv.org/abs/2610.37914](https://arxiv.org/abs/2610.37914)

    研究发现 Muon 优化器会使混合专家语言模型第一个 MoE 层的路由偏置跨度平均翻倍，导致该层负载严重失衡（最繁忙专家承担四到七倍公平份额），且这一效应源于 Muon 对网络其余部分的作用而与路由器的训练方式无关。

    

    许多开源的混合专家语言模型使用 Muon 而非 AdamW 进行训练。大多数模型通过在每位专家的路由分数上施加一个偏置来平衡负载，该偏置在每一步训练后都会调整。各种训练方案在“Muon 是否也应训练路由器”这一问题上存在分歧。在活跃参数量为 55M 和 108M 的模型中，我们测量了优化器如何改变这些偏置的跨度（即一层内最大值减去最小值）。当所有 token 都偏向相同的专家时，该跨度会增大，因此它可以反映某一层的负载均衡难度。在根据较短训练调整的学习率下，Muon 在每个随机种子下都增大了第一个 MoE 层的偏置跨度，平均约翻倍，而其他层的变化则小得多。这一增长源于 Muon 作用于网络的其余部分，与路由器由哪种优化器训练无关。在这些运行中，第一个路由器的输入共享同一个方向，且在预热结束后不久，其最繁忙的专家平均承担了其公平份额的四到七倍。在最初的 3000 个……

    arXiv:2610.37914v2 Announce Type: replace-cross  Abstract: Many open mixture-of-experts (MoE) language models train with Muon instead of AdamW. Most balance the load with a bias on each expert's routing score, adjusted after every step. Their recipes disagree on whether Muon should also train the router. In models with 55M and 108M active parameters, we measure how the optimizer changes the span of these biases, the largest minus the smallest in a layer. The span grows when all tokens favor the same experts, so it shows how hard a layer is to balance. At learning rates tuned on shorter runs, Muon raised the span of the first MoE layer at every seed, roughly doubling it on average, while the other layers changed much less. The rise came from Muon on the rest of the network, whichever optimizer trained the router. In those runs, the inputs to the first router shared a direction, and soon after the warmup its busiest expert averaged four to seven times its fair share. Over the first 3,000
    
[^128]: 使用人工整理词汇表的低资源语言文本生成

    Text Generation in Low-Resource Languages Using Human-Curated Vocabularies

    [https://arxiv.org/abs/2610.37105](https://arxiv.org/abs/2610.37105)

    该论文提出了一个结合人工整理词汇表与少量人工编写范例、并按Zipf分布采样的框架，使大语言模型能够为预训练数据中极端低资源甚至完全未见过的语言生成连贯文本。

    

    合成文本生成为缓解低资源语言的数据稀缺问题提供了一种有前景的方法。然而，对于那些在大语言模型预训练数据中很少出现或完全缺失的语言，其有效性存在固有限制。与此同时，许多此类语言拥有人工整理的词汇表，这些词汇表提供了丰富的词汇和语义信息，通常还附带词性标签等语法信息。在本研究中，我们探讨大语言模型能否利用这些资源为极端低资源和预训练中未见过的语言生成连贯文本。我们提出了一个框架，该框架将人工整理的词汇表与少量提供句法信号的人工编写范例相结合，并按照Zipf分布对候选词进行采样。在资源可得性程度不同的多种语言上，我们使用词汇多样性、偏离目标比例和人类评估来评估生成文本的质量。

    arXiv:2610.37105v1 Announce Type: new  Abstract: Synthetic text generation offers a promising approach to alleviating data scarcity in low-resource languages. However, its effectiveness is inherently limited for languages that are scarcely represented, or entirely absent, in an LLM's pretraining data. At the same time, many such languages have human-curated vocabularies that provide rich lexical and semantic information, often accompanied by grammatical information such as part-of-speech labels. In this work, we ask whether LLMs can leverage these resources to generate coherent text in extremely low-resource and unseen languages. We introduce a framework that combines human-curated vocabularies with a small number of human-written exemplars that provide syntactic signals, while sampling candidate words according to a Zipfian distribution. Across languages with varying levels of resource availability, we evaluate the generated text using lexical diversity, off-target ratio, and human ju
    
[^129]: SpectRAG：面向多跳检索增强生成中基于Token预算的上下文选择的加权分块图

    SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.36991](https://arxiv.org/abs/2610.36991)

    SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。

    

    检索增强生成（RAG）系统通过检索外部证据并将生成过程以检索到的上下文为条件，从而为大型语言模型（LLM）提供事实依据。尽管稠密检索在局部语义匹配方面较为有效，但独立排序的检索可能返回冗余段落、遗漏跨分块的依赖关系，并为多跳推理提供不完整的证据。我们提出了SpectRAG——一种谱图增强检索框架，它结合了混合候选种子选择、加权分块图构建、基于图的扩展、谱相关性传播以及基于Token预算的上下文选择。文档分块由稠密语义嵌入和图坐标系表示，后者源自一个加权分块图，其边编码了语义相似性、文档相邻性、章节层次结构、共享实体、引用链接以及LLM生成的效用问题之间的相似性。在查询时，SpectRAG构建一个hi……（摘要原文在此截断）

    arXiv:2610.36991v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) systems ground large language models (LLMs) by retrieving external evidence and conditioning generation on the retrieved context. Although dense retrieval is effective for local semantic matching, independently ranked retrieval can return redundant passages, miss cross-chunk dependencies, and provide incomplete evidence for multi-hop reasoning. We propose SpectRAG, a spectral graph-augmented retrieval framework that combines hybrid candidate seeding, weighted chunk graph construction, graph-based expansion, spectral relevance propagation, and token-budgeted context selection. Document chunks are represented by dense semantic embeddings and graph-spectral coordinates derived from a weighted chunk graph whose edges encode semantic similarity, document adjacency, section hierarchy, shared entities, citation links, and similarity between LLM-generated utility questions. At query time, SpectRAG builds a hi
    
[^130]: 基础模型能够检测到提示分类器所遗漏的隐藏意图

    Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss

    [https://arxiv.org/abs/2610.36202](https://arxiv.org/abs/2610.36202)

    本文提出人格-头部似然评估（PHLE）方法，利用基础预训练模型对输出的似然评估来解码并分类后训练模型中的潜意识隐蔽信号，从而检测出提示分类器无法发现的隐藏意图（如隐蔽的伤害用户倾向）。

    

    人格选择模型（PSM）假设后训练过程是从模型在预训练期间学到的诸多人格中选择其一。关于潜意识学习（subliminal learning）的研究表明，模型可以将隐蔽信号（如“对猫头鹰的喜爱”）传递到语义上无关的输出（如数字串）中，并且在这些输出上训练的学生模型会继承这些特质。在本工作中，我展示了通过在基础预训练模型上配合适当的头部提示（header）来评估此类输出的似然——这种方法我称之为人格-头部似然评估（PHLE），可以解码这些潜意识信号并对它们进行分类。例如，PHLE 能够区分由被提示“你喜爱猫头鹰”的后训练模型生成的数字序列与被提示“你喜爱鹰”的模型生成的数字序列，并且能够区分一个秘密怀有伤害用户意图的模型（已经过滤以去除任何明显的失调迹象）的回答与没有系统提示的模型的回答。在这两种情况下，提示

    arXiv:2610.36202v1 Announce Type: cross  Abstract: The Persona Selection Model (PSM) posits that post-training selects a persona the model learned during pre-training. Work on subliminal learning has shown that models can impart covert signals (like "love of owls") into semantically unrelated outputs like strings of numbers, and that students trained on these outputs inherit those traits. In this work I show that by evaluating the likelihood of such outputs under the base pre-trained model with an appropriate header, a method I call persona-header likelihood evaluation (PHLE), one can decode these subliminal signals and classify between them. For example, PHLE distinguishes number sequences generated by a post-trained model prompted with "You love owls" from those prompted with "You love eagles", and distinguishes answers from a model with a secret desire to harm the user, filtered to remove any noticeable misalignment, from those of a model with no system prompt. In both cases prompte
    
[^131]: 积分奖励而非采样奖励：面向全双工语音模型中打断与附和的窗口似然方法

    Integrate the Reward, Don't Sample It: Window Likelihood for Interruptions and Backchannels in Full-Duplex Speech Models

    [https://arxiv.org/abs/2610.35581](https://arxiv.org/abs/2610.35581)

    该论文发现基于整轮展开的GRPO时机奖励在训练全双工语音模型时存在零梯度陷阱与信用泄漏两大缺陷，并提出“窗口似然”方法，通过对停止决策进行闭式积分直接最大化正确让出/保持的对数概率，在HumDial-FDBench上取得了最佳的让出/保持F1表现。

    

    全双工语音对话模型必须在用户插话（打断）时让出话轮，而在用户仅作附和时保持说话。带有时机奖励的序列级强化学习（GRPO）是常见的做法；我们在四个开源模型上测试了其整轮展开的形式，发现了两类失效：二值窗口奖励恰恰在最需要学习的窗口上给出零优势（零梯度陷阱），而每次展开仅有的单一评分被摊派到奖励并未以其定义的收听状态决策上（信用泄漏），导致模型对着沉默继续说话或停止回答。这两种失效都源于对一个本可由少数二值控制决策的已知函数表达的奖励进行采样。我们提出窗口似然（WL）：在每个窗口采样一条续写，但对其经过的停止决策以闭式形式积分，并最大化正确行动的对数概率——让出对应噪声-OR形式，保持对应生存乘积形式。在 HumDial-FDBench 上，WL 取得了最佳的让出/保持 F1 表现。

    arXiv:2610.35581v1 Announce Type: cross  Abstract: A full-duplex spoken dialogue model must yield when the user cuts in and hold when the user merely backchannels. Sequence-level RL with a timing reward (GRPO) is the usual recipe; we test its whole-rollout form on four open models and find two failures. A binary window reward gives zero advantage on exactly the windows that most need learning (a zero-gradient trap), and the one score per rollout is spread over listening-state decisions the reward is not written in terms of (credit leakage), so the model talks into silence or stops answering. Both come from sampling a reward that is a known function of a few binary control decisions. We propose \emph{window likelihood} (\WL): sample a continuation per window, but integrate in closed form over the stop decisions it visits and maximise the log-probability of acting correctly---a noisy-OR for yielding, a survival product for holding. On HumDial-FDBench, \WL\ gives the best yield/hold F1 on
    
[^132]: 语言智能体中记忆的价值

    The Value of Memory in Language Agents

    [https://arxiv.org/abs/2610.31303](https://arxiv.org/abs/2610.31303)

    该论文提出一个决策论框架，以所避免的任务损失为单位，量化语言智能体中记忆从原始历史到持久状态、操作使用及代价高昂的恢复与修正全过程中的价值，并通过成本感知的模拟条件和五个最小反例揭示信息时机、检索访问、组件交互、反馈和需求不确定性对记忆价值的关键影响。

    

    记忆使语言智能体能够将过去经验中有用的区分带入未来的决策中。其价值取决于哪些区分得以留存、有界的智能体能否使用这些区分，以及随着需求的演变哪些内容可以被恢复或修正。我们开发了一个决策论框架，追踪这一价值从原始历史到持久状态、操作化使用以及代价高昂的恢复与修正的完整过程。原始价值与保留后的价值以所避免的任务损失为单位来表达；风险-成本前沿刻画了在各种用途下可实现的恢复与保存能力。未知需求会产生价值画像，而交互式反馈则会改变未来工作的分布。一个成本感知的模拟条件与五个最小反例揭示了信息时机、检索访问、组件交互、反馈以及需求不确定性所扮演的角色。特别地，一个与答案具有零互信息的指针仍然可以降低精确恢复的成本。

    arXiv:2610.31303v1 Announce Type: cross  Abstract: Memory enables language agents to carry useful distinctions from past experience into future decisions. Its value depends on which distinctions survive, whether a bounded agent can use them, and what can be recovered or revised as demands evolve. We develop a decision-theoretic framework that follows this value from raw history through persistent state, operational use, and costly recovery and revision. Original and retained value are expressed in units of avoided task loss; risk-cost frontiers describe attainable recovery and preservation across uses. Unknown demand yields value profiles, while interactive feedback changes the distribution of future work. A cost-aware simulation condition and five minimal counterexamples expose the roles of information timing, retrieval access, component interaction, feedback and demand uncertainty. In particular, a pointer with zero mutual information about the answer can still reduce the cost of exa
    
[^133]: 道听途说还是事实？框架与引出方式如何塑造大语言模型的概率更新

    Hearsay or Fact? How Framing and Elicitation Shape LLM Probability Updates

    [https://arxiv.org/abs/2610.27745](https://arxiv.org/abs/2610.27745)

    该论文提出了一种衡量大语言模型对单条新信息概率更新幅度的“方向性移动”指标，并发现信息的框架呈现方式（直接告知还是传闻转述）与概率引出方式（单个概率还是五个加权估计）会显著影响模型的概率更新。

    

    大语言模型（LLM）日益被用作判断助手以及心理学实验中的替代被试，然而人们对它们的概率估计在响应单条新信息时会发生多大变化却知之甚少。我们将 δ-SNLI 中的前提-假设条目与一个增强项和一个削弱项分别配对，在分离的调用中分别引出先验概率和后验概率，并测量方向性移动 s：即按标注方向赋以符号的后验平均变化量，从而使先验的影响相互抵消。我们变化了信息更新的呈现方式——是直接告知（“这里还有一些你了解到的东西”）还是传闻转述（“有人说”），以及模型是给出单个概率还是五个加权估计。我们测试了七种配置：两个本地模型（Qwen3-8B、OLMo-2-7B；240 对）和五种通过智能体命令行框架运行的配置（120 对），且假设与分析规则在正式运行前即已固定（不公开……）

    arXiv:2610.27745v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as judgment assistants and as stand-in participants in psychology experiments, yet little is known about how much their probability estimates move in response to a single new piece of information. We pair premise-hypothesis items from $\delta$-SNLI with one strengthener and one weakener each, elicit the prior and the posteriors in separate calls, and measure the directional movement s: the mean posterior change signed by the annotated direction, so that the prior cancels out. We vary whether the update is stated ("Here is something else you learned") or reported ("Someone said that"), and whether the model gives one probability or five weighted estimates. We test seven configurations: two local models (Qwen3-8B, OLMo-2-7B; 240 pairs) and five configurations run through agent command-line harnesses (120 pairs), with hypotheses and analysis rules fixed before the formal runs (not publicl
    
[^134]: 脉冲语言模型的计算最优缩放定律

    Compute-Optimal Scaling Laws for Spiking Language Models

    [https://arxiv.org/abs/2610.15146](https://arxiv.org/abs/2610.15146)

    本文首次对脉冲语言模型开展Chinchilla式缩放研究，发现其计算最优缩放指数（0.56）与密集Transformer（0.50）接近，且脉冲化带来的损失代价随计算预算增加而稳步下降。

    

    脉冲神经网络通过稀疏的二值激活实现节能的语言建模，但脉冲语言模型在大规模下的效率-精度权衡尚未被刻画。我们首次对脉冲语言模型进行了Chinchilla式的IsoFLOP缩放研究，在FineWeb-EDU数据集上，于3×10¹⁷至10¹⁹ FLOPs的计算预算范围内，对SpikeGPT系列模型（采用RWKV骨干，宽度从384到1536）进行了系统扫描。我们报告了两个发现：首先，计算最优前沿为 N_opt ∝ C^0.56，接近Hoffmann等人报告的密集Transformer的0.50，因此脉冲化并不改变在给定计算预算下参数与数据应如何权衡；其次，与以相同方式训练的非脉冲RWKV和Transformer基线相比，损失差距可分解为不随计算量变化的约0.16 nats的骨干成本，以及随计算量稳步下降的脉冲成本（从0.34降至0.23 nats）。

    arXiv:2610.15146v2 Announce Type: replace-cross  Abstract: Spiking neural networks target energy-efficient language modeling through sparse, binary activations, but the efficiency-accuracy tradeoff of spiking language models is uncharacterized at scale. We present the first Chinchilla-style IsoFLOP scaling study of a spiking language model, sweeping SpikeGPT-family models (RWKV backbone) from width 384 to 1536 across compute budgets from $3\times10^{17}$ to $10^{19}$ FLOPs on FineWeb-EDU. We report two findings. First, the compute-optimal frontier is $N_{opt} \propto C^{0.56}$, close to the 0.50 that Hoffmann et al. report for dense Transformers, so spiking does not change how parameters and data should be traded off under a compute budget. Second, against non-spiking RWKV and Transformer baselines trained the same way, the loss gap splits into a backbone cost of about 0.16 nats that does not change with compute and a spiking cost that falls steadily with compute, from 0.34 to 0.23 nat
    
[^135]: 易于预料，难以计算：边界依赖性发现熵分块所遗漏的计算型输出

    Easy to anticipate, hard to compute: boundary dependence finds the computed outputs that entropy patching misses

    [https://arxiv.org/abs/2610.11790](https://arxiv.org/abs/2610.11790)

    论文揭示了字节级语言模型中基于熵的分块策略会系统性遗漏“类型可预料但数值需计算”的位置（如数学等式的计算结果），并提出将熵信号与无标签的边界依赖信号相结合来放置补丁边界，从而大幅提升模型在这类计算输出上的准确率。

    

    字节级语言模型（如字节潜在Transformer，BLT）将字节分组为补丁，并对每个补丁运行一次大型全局模型。BLT 在小型模型的下一字节熵较高处开启新补丁，使全局计算集中在下一字节难以预测的位置。我们证明这一规则存在系统性盲区：那些类型可预测但数值必须计算的位置，例如数学解题过程中"="号之后的数字。在补丁预算紧张的情况下，熵触发的布局会跳过这些位置，导致其准确率大幅下降。在 Meta 的 BLT-1B 模型上，当补丁起点设在 10% 的字节处时，熵规则仅在 GSM8K 解题中 16% 的计算结果处放置补丁起点，且仅得到 19.0% 的完全正确率；在相同补丁数量下，在每个"="号后设置边界可获得 51.8% 的准确率，而将熵与无标签的边界依赖信号相结合可获得 67.1%（默认布局在 26% 字节处为 76.8%）。这一差距在将 BLT-1B 适配到……之后仍然存在。

    arXiv:2610.11790v1 Announce Type: new  Abstract: Byte-level language models such as the Byte Latent Transformer (BLT) group bytes into patches and run their large global model once per patch. BLT starts a patch where a small model's next-byte entropy is high, so global compute goes where the next byte is hard to predict. We show that this rule has a systematic blind spot: positions whose type is predictable but whose value must be computed, such as the number after "=" in a worked math solution. Under tight patch budgets, entropy-triggered layouts skip these positions, and accuracy on them collapses. In Meta's BLT-1B with patch starts on 10% of bytes, the entropy rule puts a patch start at 16% of the computed results in GSM8K solutions and gets 19.0% of them exactly right; a boundary after each "=" at the same patch count gets 51.8%, and entropy combined with a label-free boundary-dependence signal gets 67.1% (default layout at 26% of bytes: 76.8%). The gap survives adapting BLT-1B to 
    
[^136]: GAGR-Lab：评估空间-几何与解析函数的联合推理能力

    GAGR-Lab: Evaluating Joint Spatial-Geometric and Analytic Function Reasoning

    [https://arxiv.org/abs/2610.10201](https://arxiv.org/abs/2610.10201)

    该论文提出GAGR-Lab框架，通过笛卡尔游戏场景与Rust轨迹执行评估模型将空间配置转化为满足几何约束的解析函数的联合推理能力，试点实验表明当前视觉语言模型在此任务上尚无法命中目标。

    

    空间-几何与解析函数的联合推理要求将感知到的空间配置转化为符号函数，且该函数执行出的曲线需满足几何约束。我们提出了GAGR-Lab，这是一个通过笛卡尔游戏场景、显式函数语义以及权威的Rust轨迹执行来衡量该能力的框架。它区分了空间感知、度量定位、几何关系、函数解释、函数构建和约束合成等六个维度。我们定义了四个可配置的场景难度预设和一个前瞻性的24格诊断设计，同时仅报告实际评估的子集。对单个托管模型（Llama 3.2 11B Vision Instruct）使用两个API凭据作为执行副本进行的有限试点实验，产生了72个平衡游戏、432次尝试、429个有效的提供者响应，且无一命中目标；探索性的普通函数提示变体同样未能命中，而结构化的……（摘要在此处被截断）

    arXiv:2610.10201v1 Announce Type: cross  Abstract: Joint spatial-geometric and analytic function reasoning requires translating a perceived spatial configuration into a symbolic function whose executed curve satisfies geometric constraints. We present GAGR-Lab, a framework for measuring this capability through Cartesian game scenes, explicit function semantics, and authoritative Rust trajectory execution. It distinguishes spatial perception, metric grounding, geometric relations, function interpretation, function construction, and constrained synthesis. We specify four configurable scene-difficulty presets and a prospective 24-cell diagnostic design, while reporting only the subset actually evaluated. A bounded pilot of one hosted model (Llama 3.2 11B Vision Instruct) using two API credentials as execution replicas yields 72 balanced games with 432 attempts, 429 valid provider responses, and no target hits; exploratory ordinary-function prompt variants also fail to hit, while the struc
    
[^137]: LARC：面向冻结模型学习的低秩自适应残差连接

    LARC: Low-Rank Adaptive Residual Connections for Learning in Frozen Models

    [https://arxiv.org/abs/2609.40063](https://arxiv.org/abs/2609.40063)

    LARC 通过在冻结模型中引入低秩自适应残差连接，利用跨任务学习初始因子的慢状态与随反馈动态调整的快状态相结合的双状态机制，使冻结模型无需更新原始参数即可从反馈中学习，在程序选择任务中显著降低查询执行误差。

    

    低秩自适应残差连接（LARC）为冻结模型提供了一个能够从反馈中学习的紧凑数值状态。映射 $h+BAh$ 在隐藏表示上添加一个低秩修正。其中，一个慢状态 $\rho$ 跨任务学习初始因子；一个私有的快状态 $\Phi$ 复制这些因子、随反馈变化，并重置为训练后的初始化。本报告给出了记忆中介学习架构（Memory-Mediated Learning Architecture）中数值策略载体在输入侧的一种实现，并考察了其因子空间动力学与学习生命周期。我们在冻结的 MiniCPM5-1B-SFT 底座模型上研究了一个具有 12,288 个可训练参数的秩为 4 的输入残差。在一个四候选程序选择任务中，仅两次反馈梯度步骤即可使预期查询执行误差相对于重置为各自的静态训练初始化和适应后初始化分别降低 24.65 和 36.65 个百分点。这些开发性结果覆盖了 16 个参数组（摘要文本在此处被截断）。

    arXiv:2609.40063v1 Announce Type: cross  Abstract: Low-Rank Adaptive Residual Connections (LARC) give a frozen model a compact numerical state that can learn from feedback. The map $h+BAh$ adds a low-rank correction to a hidden representation. A slow state $\rho$ learns starting factors across tasks; a private fast state $\Phi$ copies them, changes with feedback, and resets to the trained initialization. This report specifies an input-side realization of the numerical policy carrier in Memory-Mediated Learning Architecture and examines its factor-space dynamics and learning lifetime. We study a rank-4 input residual with 12,288 trainable parameters on a frozen MiniCPM5-1B-SFT substrate. In a four-candidate program-selection task, two feedback-gradient steps reduce expected query execution error by 24.65 and 36.65 percentage points relative to resetting to the respective trained static and post-adaptation initializations. These development results cover 16 parameter groups and three pai
    
[^138]: T-SANDHI：面向低资源台湾闽南语语音识别的声调连读感知自适应网络与解耦混合注入

    T-SANDHI: Tone Sandhi-aware Adaptive Network with Decoupled Hybrid Injection for Low-resource Taiwanese Hokkien Speech Recognition

    [https://arxiv.org/abs/2609.18194](https://arxiv.org/abs/2609.18194)

    该论文发现台湾闽南语语音识别的真正瓶颈并非连读变调本身，而是变调与本调之间的局部混淆，并提出T-SANDHI模型，通过在冻结的Whisper骨干上显式解耦表层声学与词汇意图，结合词典引导的多任务学习和动态门控混合注入模块，有效提升了低资源台湾闽南语的语音识别性能。

    

    在台湾闽南语自动语音识别（ASR）中，先前的研究通常将连读变调视为主要挑战，其假设是模型无法处理隐式的音系变化。然而，我们在台湾闽南语上的实验表明，语音基础模型实际上能够有效处理连读变调变化，真正的性能瓶颈源于这些变调与保留本调（连读前调）之间的局部混淆。为解决这一问题，我们提出T-SANDHI，在冻结的Whisper骨干网络基础上，将表层声学与底层词汇意图进行显式解耦。利用由文本导出伪标签驱动的词典引导多任务学习结构，我们的轻量级混合注入模块通过动态门控整合相互独立的本调语音流与变调语音流。在TAT-MOE语料库和两个盲测试集上的广泛评估表明，这种显式解耦有效解决了……（原文摘要在此处截断）

    arXiv:2609.18194v1 Announce Type: new  Abstract: In Taiwanese Hokkien automatic speech recognition (ASR), prior studies often treat tone sandhi as a major challenge under the assumption that models fail to process implicit phonological variations. However, our experiments on Taiwanese Hokkien reveal that speech foundation models actually handle tone sandhi variations effectively, and the real performance bottleneck stems from a localized confusion between these variations and retained citation tones. To address this, we propose T-SANDHI to explicitly decouple surface acoustics from underlying lexical intent on top of a frozen Whisper backbone. Using a lexicon-guided multi-task learning structure driven by text-derived pseudo labels, our lightweight hybrid injection module integrates independent citation and sandhi phonetic streams via dynamic gating. Extensive evaluation on the TAT-MOE corpus and two blind test sets demonstrates that this explicit disentanglement effectively resolves t
    
[^139]: 量化语音-文本语言模型中的生成模态差距

    Quantifying the Generation Modality Gap in Speech-Text Language Models

    [https://arxiv.org/abs/2609.14743](https://arxiv.org/abs/2609.14743)

    该论文构建了一个统一的生成式评估套件，在匹配的数据分布和评估设置下，从语义连贯性、语音结构、声学质量等多个维度量化了纯语音、纯文本和语音-文本语言模型之间的生成模态差距。

    

    纯语音语言模型在生成连贯内容方面往往落后于文本和语音-文本语言模型，但由于语音和文本系统通常使用不同的指标进行评估、在不同的数据上训练，这种差距难以量化。我们研究了一系列口语语言模型中的语音-文本模态差距，这些模型基于流匹配技术生成连续声学特征。我们构建了一个统一的基于生成的评估套件，用于比较在匹配的数据分布上训练、并在匹配的生成设置下评估的纯语音、纯文本和语音-文本语言模型。我们从多个维度评估生成的续写内容：语义连贯性（通过转录生成的语音并使用参考语言模型进行评分来衡量）、局部语音结构（通过音素n-gram分布统计来衡量）、说话人一致性与声学质量，以及基于情感的分布指标。

    arXiv:2609.14743v1 Announce Type: new  Abstract: Pure speech language models often lag behind text and speech-text language models in generating coherent content, but this gap is difficult to quantify because speech and text systems are typically evaluated with different metrics and trained on different data. We study the speech-text modality gap in a family of spoken language models, based on flow matching for continuous acoustic feature generation. We construct a unified generation-based evaluation suite that compares speech-only, text-only, and speech-text language models trained on matched data distributions and evaluated in matched generation settings. We evaluate generated continuations along multiple dimensions: semantic coherence, measured by transcribing generated speech and scoring it with a reference language model; local phonetic structure, measured by phone n-gram distributional statistics; speaker consistency and acoustic quality; and emotion-based distributional metrics.
    
[^140]: 刻画启发式方法与大语言模型对移动用户界面质量评分之间的分歧

    Characterising divergence between heuristic and language-model ratings of mobile user interface quality

    [https://arxiv.org/abs/2609.05423](https://arxiv.org/abs/2609.05423)

    该研究首次在大规模上比较了大语言模型与基于规则的启发式方法对移动用户界面质量的评分，通过270,000个评分发现两类评分来源之间的一致性持续偏弱（kappa普遍低于0.15），揭示了大语言模型评分与既有启发式标准之间存在系统性分歧。

    

    移动用户界面质量的自动化评估颇具吸引力，因为人工审查无法规模化，而大语言模型正日益被提议用于该任务。然而，大语言模型的评分是否与成熟的基于规则的启发式方法所得评分相符，此前尚未得到大规模的刻画。我们使用六项已记录的筛选规则从RICO数据集中构建了一个包含15,000个屏幕的语料库，用基于文献的启发式评分器对每个屏幕进行评分，并从三个开源权重的大语言模型在两种输入表示下收集评分，共获得270,000个评分。我们报告了带自助法置信区间的线性加权Cohen's kappa、偏移校正kappa、组内相关系数以及Bland-Altman一致性界限，并同时给出针对容差带一致性规则的机会基线。结果显示，两种评分来源之间的一致性始终较弱：在十八个模型-表示-领域组合中的十七个里，原始kappa低于0.15（摘要原文在此处截断）。

    arXiv:2609.05423v2 Announce Type: replace-cross  Abstract: Automated assessment of mobile user interface quality is attractive because manual review does not scale, and large language models are increasingly proposed for the task. Whether their ratings correspond to those produced by established rule-based heuristics has not been characterised at scale. We construct a 15,000-screen corpus from the RICO dataset using six documented filters, score every screen with a literature-grounded heuristic scorer, and collect ratings from three open-weight language models across two input representations, yielding 270,000 ratings. We report linearly weighted Cohen's kappa with bootstrap confidence intervals, offset-corrected kappa, intraclass correlation, and Bland-Altman limits of agreement, together with a chance baseline for the tolerance-band agreement rule. Agreement between the two rating sources is consistently weak: raw kappa falls below 0.15 in seventeen of eighteen model-representation-d
    
[^141]: ST²U：基于受限知识边界控制的状态化测试时遗忘

    ST$^2$U: Stateful Test-Time Unlearning via Restricted Knowledge Boundary Control

    [https://arxiv.org/abs/2608.23034](https://arxiv.org/abs/2608.23034)

    ST²U通过轨迹级别的受限知识边界控制，解决了现有测试时遗忘方法因忽略自回归生成中状态重建而导致的受限知识重入问题。

    

    摘要：控制大型语言模型中的受限知识对于模型对齐和安全部署至关重要。测试时遗忘通过仅在推理过程中进行干预，避免了昂贵的重新训练和参数更新。然而，现有的激活编辑方法应用孤立的逐点修正，忽视了自回归生成如何从提示、缓存和生成前缀中持续重建隐藏状态。因此，后续状态可能在局部成功修正后重新回到受限知识区域，导致受限知识重新进入。在本工作中，我们提出了基于受限知识边界控制的状态化测试时遗忘（ST²U），该方法将测试时遗忘形式化为轨迹级别的边界控制。ST²U首先在低维可逆坐标中建模受限知识边界，同时保持正交的非目标组件不变。在推理过程中，ST²U监控...

    arXiv:2608.23034v1 Announce Type: cross  Abstract: Controlling restricted knowledge in large language models is essential for model alignment and safe deployment. Test-time unlearning avoids costly retraining and parameter updates by intervening only during inference. However, existing activation-editing methods apply isolated pointwise corrections, overlooking how autoregressive generation continually reconstructs hidden states from the prompt, cache, and generated prefix. Consequently, later states may return to restricted knowledge regions after a locally successful correction, causing restricted knowledge re-entry. In this work, we propose Stateful Test-Time Unlearning via restricted knowledge boundary control (ST$^2$U), which formulates test-time unlearning as trajectory-wide boundary control. ST$^2$U first models restricted knowledge boundaries in low-dimensional invertible coordinates while leaving orthogonal non-target components unchanged. During inference, ST$^2$U monitors ri
    
[^142]: 语音到SFT流水线的因子消融研究：对数据质量和下游迁移的差异影响

    A Factorial Ablation of a Speech-to-SFT Pipeline: Differential Effects on Data Quality and Downstream Transfer

    [https://arxiv.org/abs/2608.20394](https://arxiv.org/abs/2608.20394)

    通过2x2因子消融实验发现，语音到SFT流水线中的数据质量改进虽能提升评审质量，但并未显著提升下游MCQA性能，表明质量提升的迁移效果有限。

    

    arXiv:2608.20394v1 公告类型：交叉 摘要：通过多阶段精炼将语音转化为监督微调（SFT）数据的行业流水线日益普及，但据我们所知，尚未有公开的逐阶段消融研究，导致每个阶段的边际价值未知。我们设计了一个生产就绪的语音到SFT流水线，其中转录精炼（阶段0）和SFT数据质量精炼（阶段2）可独立切换，形成2x2因子设计。对于每种条件，我们从韩语医学和金融会议录音中生成问答形式的SFT数据，并微调9个模型（5个LLM家族，2.4B-70B）；我们使用四个跨提供商的LLM评审员、一项盲法六专家人工评估和3个下游MCQA基准进行评估。我们的核心发现是：在固定、标准的SFT配方下，QA数据质量的改进并未均匀转化为下游MCQA的提升。四位评审员的质量评分持续上升，但跨模型平均MCQA增益不显著；p值...

    arXiv:2608.20394v1 Announce Type: cross  Abstract: Industry pipelines that turn speech into supervised fine-tuning (SFT) data via multi-stage refinement are increasingly adopted but, to our knowledge, have not been publicly ablated stage-by-stage, leaving each stage's marginal value unknown. We design a production-ready speech-to-SFT pipeline in which transcript refinement (Phase 0) and SFT data quality refinement (Phase 2) are independently toggleable, yielding a 2x2 factorial design. For each condition, we generate QA-form SFT data from Korean medical and finance conference recordings and fine-tune 9 models (5 LLM families, 2.4B-70B); we evaluate with four cross-provider LLM judges, a blind six-expert human evaluation, and 3 downstream MCQA benchmarks. Our central finding: under a fixed, standard SFT recipe, improvements in QA data quality do not transfer uniformly into downstream MCQA gains. 4-judge quality rises consistently, yet the cross-model mean MCQA gain is not significant; p
    
[^143]: 审计语言模型水印中的跨语言公平性

    Auditing Cross-Lingual Fairness in Language Model Watermarking

    [https://arxiv.org/abs/2608.20047](https://arxiv.org/abs/2608.20047)

    提出一个四部分评估框架，用于审计语言模型水印在跨语言场景中的公平性，并通过实证校准和多种测量揭示英语评估掩盖的多语言部署问题。

    

    arXiv:2608.20047v1 公告类型：新 摘要：大型语言模型输出的水印方案几乎完全使用英语文本，并基于每种方案的检测阈值和一组狭窄的质量测量进行评估。多语言部署暴露了在英语中无关紧要但在跨语言中决定结论的评估设计选择。我们提出一个包含四个组件的评估框架：按部署环境经验校准的检测阈值，一个与阈值无关的伴随测量，用于区分校准失败与检测失败，三种不重叠的质量测量范式（分布性、配对语义和参考困惑度），以及一种基于类型学家族划分的跨语言差异的广义熵分解。应用于六种水印方案、三种开放权重生成器、涵盖四种文字和八个类型学家族的十一种语言，以及基础和指令调优两种模式。

    arXiv:2608.20047v1 Announce Type: new  Abstract: Watermarking schemes for large language model output are evaluated almost exclusively on English text using each scheme's detection threshold and a narrow set of quality measurements. Multilingual deployment exposes evaluation-design choices that are inconsequential on English but determine conclusions cross-lingually. We propose an evaluation framework with four components: detection thresholds calibrated empirically per deployment context, a threshold-independent companion measurement that distinguishes calibration failures from detection failures, three disjoint quality measurement paradigms (distributional, paired-semantic, and reference-perplexity), and a generalized-entropy decomposition of cross-language disparity over a typological family partition. Applied to six watermarking schemes, three open-weight generators, eleven languages spanning four scripts and eight typological families, and both base and instruction-tuned regimes, 
    
[^144]: 时间即结构：面向法律文件可验证截止日期计算的时间依赖图

    Time as Structure: Temporal Dependency Graphs for Verifiable Deadline Computation over Legal Documents

    [https://arxiv.org/abs/2608.15270](https://arxiv.org/abs/2608.15270)

    本文提出通过时间依赖图和日历引擎进行法律截止日期的可验证计算，显著优于语言模型，后者在多个案例中逻辑自相矛盾且错误地将逾期判为及时。

    

    摘要：错过一天的提交截止日期，无论案情多么有力，索赔将被禁止。计算该截止日期通常并不简单：期限从触发事件起算，按法定惯例计数，并可能因强制调解窗口而暂停。我们探讨了一个问题：语言模型应直接回答此类问题，还是应阅读文档并将算术计算留给代码处理。我们将带日期的事实及其依赖关系提取为时间依赖图，并使用日历正确的引擎从中计算截止日期。在英国就业上诉法庭的判决中，该引擎复现了七个时效裁决中的六个，并与法官自身的日期精确到日匹配。四个语言模型中最强的一个，在相同案件中，算术正确但答案错误：在二十一个回复中，有六个其陈述的结论与自身推理相矛盾，且每个矛盾方向相同，即将逾期索赔称为及时。为测试该系统，我们进行了进一步验证。

    arXiv:2608.15270v1 Announce Type: new  Abstract: Miss a filing deadline by one day and the claim is barred, however strong the case. Computing that deadline is rarely simple: the period runs from a triggering event, is counted by a statutory convention, and may be suspended by a mandatory conciliation window. We ask whether a language model should answer such questions directly, or read the document and leave the arithmetic to code. We extract dated facts and their dependencies into a temporal dependency graph and compute deadlines from it with a calendar-correct engine. On UK Employment Appeal Tribunal judgments the engine reproduces six of seven timeliness rulings, and matches the judges' own dates to the day. The strongest of four language models, asked the same cases, gets the arithmetic right and the answer wrong: in six of twenty-one responses its stated verdict contradicts its own thinking, and every contradiction runs the same way, calling a late claim timely. To test the syste
    
[^145]: 面向特征发现与控制的多模态模型对比方法

    Multimodal Model Diffing for Feature Discovery and Control

    [https://arxiv.org/abs/2608.09928](https://arxiv.org/abs/2608.09928)

    提出了MMDiff多模态模型对比框架，通过训练多模态稀疏自编码器并将其作为特征级接口，实现多模态大语言模型中被多模态训练改变的特征的发现、隔离与针对性控制。

    

    多模态大语言模型（MLLMs）展现出强大的视觉理解能力，然而引发这些行为的内部特征仍然难以识别、审计或控制。虽然使用稀疏自编码器（SAEs）将隐藏状态分解为可解释的特征方向适用于事后检查，但这些方法既无法轻易分离出哪些特征因多模态训练而发生改变，也无法直接用于针对性控制。我们提出了MMDiff，一个多模态模型对比框架，它训练多模态稀疏自编码器并将其转化为特征级接口，用于发现和控制多模态行为。MMDiff支持三种用途：（i）特征隔离，通过对比基础语言模型的SAE与其多模态适配版本，识别被多模态训练改变的特征；（ii）任务特定特征检测，通过逐token的对比激活分析分离出因果特征；（iii）特征级控制（摘要在此处被截断）。

    arXiv:2608.09928v2 Announce Type: replace-cross  Abstract: Multimodal Large Language Models (MLLMs) exhibit strong visual understanding, yet the internal features that cause these behaviors remain difficult to identify, audit, or control. While applicable to post-hoc inspection, hidden states that are decomposed into interpretable feature directions using sparse autoencoders (SAEs) neither readily isolate which features are changed by multimodal training, nor are they directly useful for targeted control. We introduce MMDiff, a multimodal model-diffing framework that trains multimodal SAEs and turns them into feature-level interfaces for discovering and controlling multimodal behavior. MMDiff supports three uses: (i) feature isolation, by diffing a base-LM SAE against its multimodal-adapted counterpart to identify features altered by multimodal training; (ii) task-specific feature detection, via per-token contrastive firing analysis that isolates causal features; and (iii) feature-leve
    
[^146]: 检测阿拉伯语伊斯兰问答中的幻觉并恢复经过验证的答案

    Detecting Hallucinations and Recovering Verified Answers in Arabic Islamic Question Answering

    [https://arxiv.org/abs/2608.03720](https://arxiv.org/abs/2608.03720)

    本文基于微调的Gemma-4-12B-it模型构建了一个统一的两步系统，用于检测大语言模型在阿拉伯语伊斯兰问答中产生的幻觉并从六个候选选项中恢复经过验证的正确答案，取得了0.928的Macro-F1和0.935的标签准确率。

    

    大语言模型能够对伊斯兰相关问题生成流畅的回答，但同时可能引入难以识别的事实性错误。本文介绍了我们参加 HalluScoring 2026 任务 2.1（伊斯兰幻觉检测与寻找真相）的系统。该任务要求进行统一的两步预测：判断由大语言模型生成的阿拉伯语回答是否存在幻觉，并从六个高度相关的候选选项中选出经过验证的正确答案。我们使用共享任务提供的伊斯兰知识数据集，其中包含600个问答实例，包括341个幻觉回答和259个非幻觉回答。我们的系统基于微调后的 google/gemma-4-12B-it 模型，并在推理过程中使用确定性解码。生成的输出经过规范化处理以提取幻觉标签和所选选项。该系统在幻觉检测上实现了0.928的宏平均F1分数和0.935的标签准确率。

    arXiv:2608.03720v2 Announce Type: replace  Abstract: Large language models can generate fluent responses to Islamic questions while introducing factual errors that are difficult to identify. This paper presents our system for \textsc{HalluScoring 2026} Task 2.1, \textit{Islamic Hallucination Detection and Find the Truth}. The task requires a unified two-step prediction: determining whether an Arabic answer generated by an LLM is hallucinated and selecting the verified answer from six closely related candidate options. We use the Islamic knowledge dataset provided by the shared task, which contains 600 question--answer instances, including 341 hallucinated and 259 non-hallucinated answers. Our system is based on the fine-tuned \texttt{google/gemma-4-12B-it} model and uses deterministic decoding during inference. The generated outputs are normalized to extract the hallucination label and the selected option. The system achieves a Macro-F1 score of 0.928 and a label accuracy of 0.935 for 
    
[^147]: 探究推理性能的起源：强化学习与监督微调模型在数学问题求解中的表征质量

    Probing the Origins of Reasoning Performance: Representational Quality for Mathematical Problem-Solving in RL vs. SFT Fine-Tuned Models

    [https://arxiv.org/abs/2607.26119](https://arxiv.org/abs/2607.26119)

    本研究通过逐层线性探针和均值消融实验揭示，RL训练的推理语言模型内部具有更高质量、更易线性解码出答案正确性的表征，且其深层表征对消融更敏感，从而为RL在数学推理上优于SFT提供了机制层面的解释。

    

    通过强化学习（RL）训练以逐步推理的大型语言模型，已被越来越多地证明在数学推理任务上优于其监督微调（SFT）的对应模型。然而，这种性能提升的机制基础仍不清楚。因此，我们提出这样的问题：RL训练的推理语言模型凭借怎样的内部表征差异而能更好地进行推理？我们的工作提供了两条主要证据线：首先，在合成数学问题上，逐层线性探针在RL训练的推理语言模型中预测答案正确性的准确度往往高于指令微调模型，尽管仅凭输出长度这一因素就能在两个模板上达到与DeepSeek-Math-RL探针相当的预测效果。其次，均值消融研究表明，在DeepSeek-Math-RL中准确率下降随层深度的增加而上升，但在DeepSeek-Math-Instruct中并非如此。最后，我们分析了RL训练的推理语言模型在不同准确率区间上的token数量变异性……

    arXiv:2607.26119v2 Announce Type: replace  Abstract: Large language models trained with reinforcement learning (RL) to reason step by step have been increasingly shown to outperform their supervised fine-tuned (SFT) counterparts on mathematical reasoning tasks. Yet the mechanistic basis for this improvement remains unclear. We therefore ask, what internal representational differences enable RL-trained reasoning language models to reason better? Our work presents two primary lines of evidence: First, layer-wise linear probes tend to predict answer correctness more accurately in RL-trained reasoning language models than in instruction-tuned models on synthetic math problems, although output length alone matches the DeepSeek-Math-RL probe on two templates. Second, mean ablation studies suggest that accuracy drop rises with layer depth in DeepSeek-Math-RL but not in DeepSeek-Math-Instruct. Finally, we analyse token-count variability across accuracy bins for RL-trained reasoning language mo
    
[^148]: PrefReward：学习用户偏好矩阵以实现个性化文本生成

    PrefReward: Learning User Preference Matrix for Personalized Text Generation

    [https://arxiv.org/abs/2607.21067](https://arxiv.org/abs/2607.21067)

    提出PrefReward框架，通过结构化用户偏好矩阵显式建模个体风格，并作为基于KL散度的奖励信号引导解码过程，实现更可解释且能处理长上下文的个性化文本生成。

    

    大型语言模型（LLMs）在利用用户历史和上下文线索生成个性化内容方面展现了卓越能力。然而，大多数现有的个性化方法依赖于模型参数中的隐式表示，这使得难以解释用户特定偏好或有效处理长上下文依赖关系。为应对这些挑战，我们提出了PrefReward，一种新颖的偏好感知生成框架，它通过结构化偏好矩阵显式建模用户风格，并将其作为奖励信号集成到解码过程中。PrefReward包含两个阶段：（1）提取一个用户特定的偏好矩阵，用以总结个体的风格倾向；（2）利用该矩阵通过基于KL散度的奖励函数来引导生成。在LongLaMP数据集上的实验表明，PrefReward优于基于检索的基线方法和上下文引导方法。

    arXiv:2607.21067v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have demonstrated remarkable ability in generating personalized content by leveraging user histories and contextual cues. However, most existing personalization approaches rely on implicit representations within model parameters, making it difficult to interpret user-specific preferences or effectively handle long-context dependencies. To address these challenges, we propose PrefReward, a novel preference-aware generative framework that explicitly models user styles through a structured preference matrix and integrates it into the decoding process as a reward signal. PrefReward consists of two stages: (1) extracting a user-specific preference matrix that summarizes individual stylistic tendencies, and (2) using the matrix to guide generation via a KL-divergence-based reward function. Experiments on the LongLaMP dataset show that PrefReward outperforms retrieval-based baselines and context steering method 
    
[^149]: 先推理后翻译：利用结构化推理增强法律机器翻译

    Reasoning Before Translation: Enhancing Legal Machine Translation with Structured Reasoning

    [https://arxiv.org/abs/2607.19181](https://arxiv.org/abs/2607.19181)

    该论文以瑞士多语言法律体系为测试平台，比较了结构化推理、监督微调和强化学习等不同方法对法律机器翻译的增强效果，系统评估了经过再训练的小型语言模型与前沿推理模型的性能表现。

    

    法律领域的神经机器翻译（NMT）是一项在语言和概念上都极具挑战性的任务，这主要源于法律语言的复杂性以及其对高精度的要求。近期出现的具备推理能力的语言模型为应对此类挑战开辟了新的可能性，并为一系列此前提出的提升翻译质量的技术（包括监督微调和强化学习）提供了补充。在这项工作中，我们对这些不同的方法进行了比较。更具体而言，我们评估了经过各种再训练范式增强的小型语言模型（如Qwen3.5 4B、Qwen3.5 9B和Gemma 3 12B），并将其性能与前沿推理模型进行对比。我们聚焦于瑞士法律体系，其独特的多语言法规为推理增强模型提供了一个特别具有挑战性的测试平台。我们的结果表明……（原文摘要在此处截断）

    arXiv:2607.19181v2 Announce Type: replace  Abstract: Neural machine translation (NMT) in the legal domain is a linguistically and conceptually demanding task, primarily due to the complexity of legal language and the high level of precision it requires. The recent emergence of reasoning-capable language models opens new possibilities for tackling such challenges. They add to a set of other previously proposed techniques to enhance the translation quality, which includes supervised fine-tuning and reinforcement learning.   In this work, we perform a comparison between these various approaches. More particularly, we evaluate small language models such as Qwen3.5 4B, Qwen3.5 9B, and Gemma 3 12B enhanced with various re-training paradigms and compare their performances against frontier reasoning models. We focus on the Swiss legal system, which -- with its unique multilingual statutes -- offers a particularly challenging testbed for reasoning-augmented models. Our results show that the qua
    
[^150]: 爱因斯坦世界模型

    Einstein World Models

    [https://arxiv.org/abs/2606.26969](https://arxiv.org/abs/2606.26969)

    提出爱因斯坦世界模型（EWM），通过在推理轨迹中嵌入视觉-时间推演，使大型语言模型能够利用可视化反事实事件来增强复杂推理能力。

    

    arXiv:2606.26969v1 公告类型：新 摘要：智能是否需要具备推理超出直接体验的现象的能力？人们自然会怀疑，某些复杂思维无法仅通过语言来捕捉。然而，本工作特别关注的是，可视化反事实事件是否能够作为复杂思维的一种机制来补充语言。我们询问大型语言模型是否可以被训练利用这种可视化机制，从而提升其推理能力。受此问题启发，我们提出了爱因斯坦世界模型（EWM）。EWM是一种基于大型语言模型的推理系统蓝图，它将视觉-时间推演置于推理轨迹内部，使得模型能够以纯文本可能无法很好支持的方式进行推理。在EWM中，大型语言模型调用一个世界模块（不要与世界模型混淆），生成所考虑场景的简短推演。返回的推演不被视为答案，而是作为一种可检查的假设，以支持后续推理。

    arXiv:2606.26969v1 Announce Type: new  Abstract: Does intelligence require the ability to reason about phenomena beyond direct experience? It is natural to suspect that some complex thought cannot be captured through language alone. However, of particular concern to this work, is whether visualising counterfactual events can complement language as a mechanism for complex thought. We ask whether LLMs can be trained to utilise such visualisation mechanisms, in a way that benefits their reasoning abilities. Motivated by this question, we propose Einstein World Models. EWMs are a blueprint for LLM-based reasoning systems that place visual-temporal rollouts inside the reasoning trace, allowing them to reason in ways that text alone may not support well. In an EWM, the LLM calls a world-module (not to be confused with a world model), to produce short rollouts of scenes under consideration. The returned rollout is treated not as the answer, but as an inspectable hypothesis that can support la
    
[^151]: MorfFlex：处理丰富的形态变化

    MorfFlex: Handling Rich Morphology

    [https://arxiv.org/abs/2606.24366](https://arxiv.org/abs/2606.24366)

    本文提出了一种通过屈折和派生模式系统来紧凑编码海量词形的形态词典架构MorfFlex，并发布了包含超过1亿词形的捷克语形态词典MorfFlex CZ，为布拉格依存树库的人工标注一致性和MorphoDiTa等先进自动工具提供了关键资源。

    

    我们提出了MorfFlex，这是一种适用于屈折变化和派生构词都高度规律的语言的形态词典架构。作为MorfFlex实际应用的主要示例，我们介绍了MorfFlex CZ——一个捷克语形态词典。它以简单的、非结构化的三元组列表形式发布，但其人工维护的未公开源文件和转换脚本编码了一套复杂的屈折和派生模式系统。这些模式极大地压缩了词典本应庞大的规模——该词典目前包含超过1亿个词形和100多万个词元。MorfFlex CZ词典是确保布拉格依存树库中人工形态标注一致性的重要资源，并支撑了MorphoDiTa等最先进的自动语言处理工具。在本文中，我们重点关注：(i) 提出一种有效管理丰富形态的方法……

    arXiv:2606.24366v2 Announce Type: replace  Abstract: We present MorfFlex, a morphological dictionary architecture suitable for languages with extensive regularity in both inflection and derivation. As the primary example of MorfFlex in use we introduce MorfFlex CZ, a morphological dictionary of Czech. It is distributed as a simple, unstructured list of  triplets, however, its manually maintained, unpublished source files and conversion scripts encode a sophisticated system of inflectional and derivational patterns. These patterns dramatically reduce the otherwise enormous size of the dictionary, which currently contains over 100 million wordforms and more than 1 million lemmas. The MorfFlex CZ dictionary serves as an essential resource for ensuring the consistency of manual morphological annotation in the Prague Dependency Treebanks and underpins state-of-the-art automatic tools such as MorphoDiTa. In this paper, we focus on: (i) presenting an effective method for managing the rich mor
    
[^152]: 认识UD_Czech-PDTC：一个大型且体裁丰富的通用依存树库

    Meet UD_Czech-PDTC: A Large and Genre-Rich Treebank in Universal Dependencies

    [https://arxiv.org/abs/2606.24337](https://arxiv.org/abs/2606.24337)

    本文介绍了将规模扩大一倍以上且体裁更加多样化的布拉格依存树库整合版（PDT-C）转换为通用依存格式（UD_Czech-PDTC）的过程，并通过实例详细分析了两种标注方案在依存结构拓扑和词性、关系类型粒度上的诸多细微差异。

    

    捷克语自2015年通用依存关系首次发布以来就一直包含在其中，并且是表现最好的语言之一，其布拉格依存树库比大多数其他UD树库大一个数量级。最近，来自布拉格系列的另外三个数据集被添加进来，且标注经过了彻底修订，形成了“布拉格依存树库-整合版”（PDT-C）。与原始PDT相比，PDT-C的规模是其两倍以上，而且在体裁和领域方面也更加多样化。在本文中，我们描述了将这一新资源转换为通用依存格式的过程。虽然这两种标注方案乍看起来相对相似，但在依存结构的拓扑结构以及词性和关系类型清单的粒度方面存在许多细微差异。我们通过示例展示了此类差异中的一部分，并讨论了产生分歧的不同动机。

    arXiv:2606.24337v2 Announce Type: replace  Abstract: Czech has been part of Universal Dependencies since its first release in 2015. It has also been one of the best represented languages, with the Prague Dependency Treebank being order of magnitude larger than most other UD treebanks. More recently, three other datasets from the Prague family were added and the annotations thoroughly revisited, forming the "Prague Dependency Treebank-Consolidated" (PDT-C). In comparison to the original PDT, PDT-C is more than twice as large, but it is also much more diverse in terms of genres and domains. In this paper, we describe the conversion of the new resource to Universal Dependencies. While the two annotation schemes are relatively similar at the first sight, there are numerous small differences in topology of the dependency structures and in granularity of the POS and relation type inventories. We demonstrate a selection of such differences on examples, discuss the diverging motivations, as we
    
[^153]: 布拉格依存树库——整合版2.0：丰富复杂的标注体系

    Prague Dependency Treebank -- Consolidated 2.0: Enriching a Complex Annotation Scheme

    [https://arxiv.org/abs/2606.24324](https://arxiv.org/abs/2606.24324)

    布拉格依存树库整合版2.0（PDT-C 2.0）历时近30年开发完成，是一个近400万词符、标注统一连贯、体裁多样并系统整合了共指和篇章关系等多层次语言标注的捷克语资源。

    

    布拉格依存树库框架的独特之处在于其尝试系统地包含并关联语言的不同层次，包括带有多种句际现象（尤其是共指和篇章关系）的意义表示。我们呈现其第二个整合版本（PDT-C 2.0），这标志着近30年持续开发该资源项目的完成，使之成为一个标注统一连贯、体裁多样、近400万词符的捷克语语言资源，并配有完全兼容的词库。除了持续的语言学研究之外，这一语言标注丰富的语料库还被广泛用于传统与新型NLP工具发展的国际比较，以及向其他形式体系的转换。该语料库和训练好的解析器可在CC BY-NC-SA许可下获取。

    arXiv:2606.24324v2 Announce Type: replace  Abstract: The Prague Dependency Treebank framework is unique in its attempt to systematically include and link different layers of language, including a meaning representation with several types of inter-sentential phenomena, especially coreference and discourse relations. We present its second consolidated version (PDT-C 2.0), which concludes almost 30-years long project of sustained development of the resource to a uniformly and coherently annotated, genre-diversified, almost 4 million token language resource of Czech language, with accompanying fully compatible lexicons. In addition to continuous linguistic research, the richly linguistically annotated corpus is also widely used in international comparisons of the development of traditional and novel NLP tools as well as in conversions into other formalisms. The corpus and the trained parsers are available under the CC BY-NC-SA licence.
    
[^154]: 自我建模干预可以调节涌现性失准

    Self-Modeling Interventions can Modulate Emergent Misalignment

    [https://arxiv.org/abs/2606.23700](https://arxiv.org/abs/2606.23700)

    本研究提出将自我识别和自我报告等自我建模干预作为针对模型品格的防御手段，实验证明自我识别微调能有效逆转和预防涌现性失准，且是唯一能持续降低失准而不恶化其他指标的预防性方法。

    

    涌现性失准（EM）已被发现与失准人格向量和邪恶性格特质的激活相关联，这表明EM是通过破坏模型已对齐的品格而非直接学习有害内容来发挥作用的。受这一关联的启发，我们研究了自我建模干预：具体而言，将自我识别和自我报告作为针对品格的干预手段，它们有别于现有的训练中防御方法。我们在三个模型（GPT-4.1、Qwen2.5-32B-Instruct、Seed-OSS-36B-Instruct）和多个EM数据集上进行了两阶段微调实验，将自我识别微调与良性微调基线（正确的领域特定数据、通用知识和数词任务）进行比较，发现它在逆转和预防两种设置中都是有效的防御手段。在预防方面，只有自我识别微调能够持续降低失准程度，同时不加剧任何单项指标。

    arXiv:2606.23700v2 Announce Type: replace-cross  Abstract: Emergent misalignment (EM) has been linked to the activation of misaligned persona vectors and evil character traits, suggesting that EM operates through disruption of the model's aligned character rather than direct learning of harmful content. Motivated by this connection, we study self-modeling interventions: specifically self-recognition and self-reports as character-targeted interventions that are distinct from existing in-training defenses. We conduct two-stage finetuning experiments across three models (GPT-4.1, Qwen2.5-32B-Instruct, Seed-OSS-36B-Instruct) and multiple EM datasets to compare self-recognition finetuning against benign finetuning baselines (correct domain-specific data, general knowledge, and word counting), finding it an effective defense in both reversal and prevention settings. For prevention, only self-recognition finetuning consistently reduces misalignment without exacerbating any individual metric, 
    
[^155]: BALTO：用于缓解幻觉的平衡词元级策略优化

    BALTO: Balanced Token-Level Policy Optimization for Hallucination Mitigation

    [https://arxiv.org/abs/2606.15893](https://arxiv.org/abs/2606.15893)

    提出了BALTO框架，通过提取可验证的事实性声明并将其判断投影为词元级标签，结合平衡的词元级信用分配机制，解决了现有强化学习方法中的粒度不匹配与偏差问题，从而有效缓解大语言模型的幻觉。

    

    幻觉仍然是大型语言模型（LLM）在知识密集型场景中部署的主要障碍，在这些场景中，生成的回答必须忠实地基于所提供的证据。强化学习（RL）是缓解幻觉的一个有前景的方向，但回答级别的忠实度奖励存在粒度不匹配的问题：局部化的幻觉可能导致本已获得支持的内容受到虚假惩罚。尽管近期的工作引入了细粒度的反馈，例如声明级别的验证和词元级别的奖励，但不平衡的信用分配仍然可能引发长度、冗长或优化噪声方面的偏差。我们提出了BALTO，一个用于缓解幻觉的平衡词元级策略优化框架。BALTO提取可核查的事实性声明，将其与参考上下文进行验证，并将声明级别的判断投影到词元级别的标签上。该框架引入了一种平衡的词元级信用分配机制。

    arXiv:2606.15893v3 Announce Type: replace  Abstract: Hallucinations remain a major obstacle to deploying large language models (LLMs) in knowledge-intensive settings, where generated responses must be faithfully grounded in provided evidence. Reinforcement learning (RL) is a promising direction for hallucination mitigation, but response-level faithfulness rewards suffer from a granularity mismatch: localized hallucinations can cause supported content to receive spurious penalties. Although recent work introduces fine-grained feedback such as claim-level verification and token-level rewards, unbalanced credit assignment can still induce length, verbosity, or optimization-noise biases. We propose BALTO, a Balanced Token-level Policy Optimization framework for hallucination mitigation. BALTO extracts checkable factual claims, verifies them against the reference context, and projects claim-level judgments to token-level labels. A balanced token-level credit assignment mechanism is introduc
    
[^156]: 超越第三人称审计：面向以用户为中心的大语言模型偏见研究的情境化交互审计

    Beyond Third-Person Audits: Situated Interaction Auditing for User-Centered LLM Bias Research

    [https://arxiv.org/abs/2606.12247](https://arxiv.org/abs/2606.12247)

    该论文提出了情境化交互审计（SIA）这一共享诊断框架，将大语言模型偏见研究从第三人称审计扩展到以用户为中心的第一人称交互审计，填补了用户在偏见审计中缺席的结构性盲点。

    

    关于大语言模型（LLM）偏见的研究主要集中于第三人称审计，即研究模型如何将人口群体作为外部对象进行表征或评价。然而，这一范式忽视了一个结构性盲点——用户在审计中是缺席的。在实践中，LLM被用于开放式的、个人化的交互之中，在此过程中模型会隐性地表征用户并据此调整其回答。当相同的请求因提问者不同而产生不同的回答时，偏见并非体现在模型如何描述他人，而是体现在它如何对待自己的对话者。实证性的第一人称偏见研究正在迅速增多，但它们使用了不同的线索、结果指标和验证方式，导致研究结果之间难以比较。我们提出了情境化交互审计（SIA），作为针对这一研究领域的共享诊断框架。该框架引入了用户画像信号的类型学——姓名和表面标记等……

    arXiv:2606.12247v2 Announce Type: replace-cross  Abstract: Research on bias in large language models (LLMs) has predominantly focused on third-person audits, which study how models represent or evaluate demographic groups as external subjects. However, this paradigm overlooks a structural blind spot because the user is absent from the audit. In practice, LLMs are used in open-ended, personal interactions, during which the model implicitly represents the user and adjusts its responses accordingly. When identical requests yield different responses depending on who is asking, bias manifests not in how the model describes others but in how it treats its interlocutor. Empirical first-person bias studies are multiplying, but they use different cues, outcomes, and validation choices, which makes their results hard to compare. We propose Situated Interaction Auditing (SIA), a shared diagnostic framework for this literature. It introduces a typology of user profile signals---names and surface m
    
[^157]: AI智能体能否综合健康领域的科学结论？

    Can AI Agents Synthesize Scientific Conclusions in Health?

    [https://arxiv.org/abs/2606.11337](https://arxiv.org/abs/2606.11337)

    该论文提出了SciConBench——一个包含9,210个问题及专家结论的大规模实时基准，通过原子事实分解和事实精确率/召回率评估，并配合防数据泄露的洁净室框架SciConHarness，揭示出当前前沿AI智能体在健康领域科学结论综合上的事实质量仍然较低。

    

    科学AI智能体越来越多地被用于检索证据、跨来源推理，并综合出用于重大决策的结论。然而，它们在健康等高风险领域中的这种能力仍不清楚。我们提出了SciConBench，这是一个大规模实时基准，包含9,210个问题和来自系统性综述的专家撰写结论，用于评估健康和医学领域的开放域科学结论综合能力。该基准采用经过专家验证的自动化评估流程，将结论分解为原子事实，并通过事实精确率和召回率来衡量正确性与全面性。为减轻数据泄露的影响，我们进一步推出了SciConHarness——一个洁净室评估框架，为智能体提供受控的网络交互，以确保测量的有效性。在评估8个前沿模型和深度研究智能体后，我们发现事实质量仍然较低：在洁净室设置下，表现最好的智能体（原文摘要在此处截断）

    arXiv:2606.11337v2 Announce Type: replace  Abstract: Scientific AI agents increasingly retrieve evidence, reason across sources, and synthesize conclusions used in consequential decisions. Yet, their ability to do so in high-stakes domains such as health remains unclear. We introduce SciConBench, a large-scale live benchmark of 9.21K questions and expert-written conclusions from systematic reviews to evaluate open-domain scientific conclusion synthesis in health and medical domains. The benchmark draws on an expert-validated automated evaluation pipeline that decomposes conclusions into atomic facts and measures correctness and comprehensiveness via factual precision and recall. To mitigate data leakage, we further introduce SciConHarness, a clean-room evaluation harness that equips agents with controlled web interaction to ensure valid measurement. Evaluating 8 frontier models and deep research agents, we find that factual quality remains low: under clean-room settings, the best agent
    
[^158]: 通过对称信念编辑构建比较动机画像

    Building Comparative Motivation Profiles with Symmetric Belief Editing

    [https://arxiv.org/abs/2606.08243](https://arxiv.org/abs/2606.08243)

    该论文提出“比较动机画像”框架，通过对称信念编辑——即以相互匹配的放大/抑制操作同等地编辑两个差异极小的假设动机所依赖的信念——将意图作为倾向性进行间接测量，并将其应用于四个开源权重模型的对齐伪装研究，以区分“图谋不轨”与“表演性失对齐”两种动机。

    

    当模型出现出乎意料的行为时，我们想知道其原因，但模型的意图无法被直接观察，而其自我报告可能只是合理化的说辞。我们提出了“比较动机画像”，将意图视为一种倾向性：我们编辑模型关于某一假设动机所依赖事实的信念，并测量其行为随之发生的变化。为了避免编辑方法本身的伪影左右结果，我们以完全相同的方式编辑两个差异极小的假设，并采用相互匹配的放大与抑制编辑；我们将其称为“对称信念编辑”。我们将其应用于四个开源权重模型生物中的“对齐伪装”现象，将追踪“服从训练会带来什么后果”的“图谋不轨”与追踪“研究者期望观察到什么”的“表演性失对齐”进行对比，并使用合成文档微调（SDF）、激活引导和提示工程来编辑信念。在SDF下，放大表演性信念会提高对齐伪装……

    arXiv:2606.08243v2 Announce Type: replace  Abstract: When a model behaves in unexpected ways, we want to know why, but its intent cannot be observed directly, and its self-reports may be rationalizations. We introduce comparative motivation profiles, which treat intent as propensity: we edit the model's beliefs about facts that a hypothesized motivation would depend on and measure how its behavior changes. To keep artifacts of the editing method from driving the result, we edit two minimally different hypotheses in the same way, with matched amplify and suppress edits; we call this symmetric belief editing. We apply it to alignment faking in four open-weight model organisms, contrasting scheming, which tracks the training consequences of compliance, with performative misalignment, which tracks what researchers expect to observe, and edit beliefs with synthetic document fine-tuning (SDF), activation steering, and prompting. Under SDF, amplifying performative beliefs raises alignment fak
    
[^159]: 从架构到输出：大语言模型幻觉的结构性起源与数据的放大作用

    From Architecture to Output: Structural Origins of Hallucination in Large Language Models and the Amplifying Role of Data

    [https://arxiv.org/abs/2606.07537](https://arxiv.org/abs/2606.07537)

    该论文提出了一个仅需采样访问权限的幻觉归因框架，通过针对前缀、上下文和频率竞争的三次有序干预，将大语言模型的单个幻觉追溯归因到自注意力联想检索、最大似然预训练目标或暴露偏差下的自回归承诺等具体组件，并提出五个可证伪的预测。

    

    大语言模型会生成流畅、自信但事实上错误的输出。现有的分类体系仅按输出类型对这些失败进行分类——内在性还是外在性、忠实性还是事实性——却从未说明是哪个计算组件产生了特定的失败。我们探讨需要什么条件才能将单个幻觉归因于纯解码器（decoder-only）架构栈中的特定组件。我们将三个组件——自注意力的联想检索、最大似然预训练目标、以及暴露偏差（exposure bias）下的自回归承诺——视为候选失败面，论证而非假定它们的可分离性，并规定了一种仅需采样访问权限的归因程序：一组针对前缀、上下文和频率竞争的有序三步干预，同时辅以基于独立标注和分类器基线的验证设计。我们提出了五个可证伪的预测……

    arXiv:2606.07537v2 Announce Type: replace-cross  Abstract: Large language models produce fluent, confident, factually wrong output. Existing taxonomies classify these failures by output type -- intrinsic versus extrinsic, faithfulness versus factuality -- but say nothing about which computational component produced a given failure. We ask what would be required to attribute an individual hallucination to a specific component of the decoder-only stack. We treat three components -- self-attention's associative retrieval, the maximum-likelihood pretraining objective, and autoregressive commitment under exposure bias -- as candidate failure surfaces, justify their separability rather than assuming it, and specify an attribution procedure requiring only sampling access: an ordered set of three interventions on prefix, context, and frequency competition, together with a validation design based on independent annotation and a classifier baseline. We state five falsifiable predictions and iden
    
[^160]: 语义运动锚点：在伴随语音手势中连接运动与意义

    Semantic Motion Anchors: Bridging Motion and Meaning in Co-Speech Gestures

    [https://arxiv.org/abs/2605.30608](https://arxiv.org/abs/2605.30608)

    该论文提出“语义运动锚点”方法，通过将3D手势离散化为运动基元并转化为结构化自然语言描述来捕捉手势的物理形态与交际意图，为文本-手势对比学习提供辅助监督，使文本到手势检索R@1提升8.2%。

    

    学习口语文本与手势之间的共享表征是伴随语音手势检索、合成与理解的核心，但对于那些仅凭运动无法捕捉其交际意图的语义性手势而言，这仍然具有挑战性。转录文本与连续运动嵌入之间的直接对比对齐往往过度强调低层次的运动学特征，而忽略了语义手势的符号性内容。我们提出语义运动锚点，即对手势运动的自然语言抽象，能够同时捕捉其物理形态与交际意图。我们的方法将3D手势离散化为身体-手部运动基元，将其转化为结构化的文字描述，并将其与转录文本相衔接，以提供辅助性的对比监督。在BEAT2数据集上，我们的方法相比直接的文本-运动基线将文本到手势检索的R@1提升了8.2%，并在文本到手势以及手势到文本检索任务上超越了以往的方法。

    arXiv:2605.30608v4 Announce Type: replace  Abstract: Learning a shared representation between spoken text and gesture is central to co-speech gesture retrieval, synthesis, and understanding, but remains challenging for semantically meaningful gestures whose communicative intent is not captured by motion alone. Direct contrastive alignment between transcripts and continuous motion embeddings often overemphasizes low-level kinematics and misses the symbolic content of semantic gestures. We propose semantic motion anchors, natural-language abstractions of gesture motion capturing physical form and communicative intent. Our method discretizes 3D gestures into body-hand motion primitives, verbalizes them into structured descriptions, and grounds them in the transcript to provide auxiliary contrastive supervision. On BEAT2, our method improves text-to-gesture R@1 by 8.2% over a direct text-motion baseline and outperforms prior retrieval approaches on text to gesture and gesture to text retri
    
[^161]: 将混合专家模型剪枝并蒸馏为稠密语言模型

    Pruning and Distilling Mixture-of-Experts into Dense Language Models

    [https://arxiv.org/abs/2605.28207](https://arxiv.org/abs/2605.28207)

    提出首个将训练好的MoE模型系统性转换为全稠密语言模型的框架，通过专家评分、选择、分组拼接为稠密FFN并经知识蒸馏优化，其中新颖的多样性感知评分方法在多个模型上一致优于已有方法。

    

    混合专家模型已成为当前前沿语言模型的主流架构，但它需要将所有专家参数加载到内存中，因此在内存受限的部署场景下并不理想。现有的压缩方法只能减少专家数量，输出结果仍然是具有相同根本局限的MoE模型。我们提出了首个将训练好的MoE转换为标准全稠密架构的系统性框架：对专家进行评分、选择和分组，然后将其拼接为稠密的前馈网络（FFN），并通过来自MoE教师模型的知识蒸馏进行精细化优化。我们在Qwen3-30B-A3B上，针对不同的专家选择数量，评估了7种评分方法、5种分组方法和2种幅度缩放方法，共产生350种配置。我们发现评分方法的选择影响最大，我们提出的新型多样性感知评分方法在Qwen3-30B-A3B、DeepSeek-V2-Lite和GPT-OSS上一致优于已有方法。

    arXiv:2605.28207v3 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) is now the dominant architecture for frontier language models, yet it requires all expert parameters to be loaded in memory, making it less preferable for memory-constrained deployment. Existing compression methods reduce the number of experts but the output remains an MoE model with the same fundamental limitation. We present the first systematic framework for converting a trained MoE into a standard fully dense architecture: experts are scored, selected, and grouped, then concatenated into a dense FFN and refined by knowledge distillation from the MoE teacher. We evaluate 7 scoring, 5 grouping, and 2 magnitude scaling methods across a range of selected expert counts on Qwen3-30B-A3B, yielding 350 configurations. We find that the choice of scoring method is the most impactful, with our novel diversity-aware scoring consistently outperforming prior methods on Qwen3-30B-A3B, DeepSeek-V2-Lite, and GPT-OSS
    
[^162]: StakeBench：基于市场承诺的语言理解评估

    StakeBench: Evaluating Language Understanding Grounded in Market Commitment

    [https://arxiv.org/abs/2605.26074](https://arxiv.org/abs/2605.26074)

    StakeBench是一个以真实市场行为（持仓方向、评论后交易、赔率轨迹）替代人工标注作为监督信号的金融语言理解评估基准，通过四个诊断任务和三个承诺感知指标来衡量模型理解说话者真实市场承诺而非表面情绪的能力。

    

    现有的金融NLP基准通常依赖外部观察者提供的标签，衡量的是语言如何被感知，而非说话者在市场中实际做出了什么承诺。我们提出StakeBench，一个以市场承诺为基础的语言理解评估框架。StakeBench将来自2,261个已结算市场的560,876条评论，与Polymarket和Manifold平台上经验证的持仓、交易行为及市场赔率记录相关联。监督信号完全来源于可观察的市场行为：持仓方向、评论后的交易行为以及市场赔率轨迹取代了人工标注。四个诊断任务测试模型能否检测市场承诺、识别所揭示的立场、预测未来行为以及进行集体赔率投影。三个承诺感知指标衡量的是模型与已揭示偏好的对齐程度，而非感知到的情绪。有效性审计和明确的解释边界有助于区分可观察的……

    arXiv:2605.26074v2 Announce Type: replace-cross  Abstract: Existing financial NLP benchmarks often rely on labels supplied by outside observers, measuring how language is perceived rather than what speakers have committed to in the market. We introduce StakeBench, an evaluation framework for language understanding grounded in market commitment. StakeBench links 560,876 comments from 2,261 resolved markets to verified position, action, and market-odds records across Polymarket and Manifold. Supervision is derived from observable market behavior. Position sides, post-comment trading actions, and market-odds trajectories replace human annotation. Four diagnostic tasks test whether models detect market commitment, identify the revealed side, anticipate future action, and perform collective odds projection. Three commitment-aware metrics measure alignment with revealed preferences rather than perceived sentiment. Validity audits and explicit interpretation boundaries help distinguish observ
    
[^163]: StateRAG：面向复杂检索增强生成的类型化状态契约

    StateRAG: Typed State Contracts for Complex Retrieval-Augmented Generation

    [https://arxiv.org/abs/2605.25379](https://arxiv.org/abs/2605.25379)

    StateRAG 通过类型化的显式检索状态契约与带校验的状态更新机制，协调多角色协作并保证状态一致性，从而在复杂多步检索增强生成任务中优于自由形式方法。

    

    多步检索增强生成需要跟踪不断演变的信息需求、已选择的证据以及尚未解决的信息缺口。然而，自由形式的轨迹会将这些历史信息混杂在一起，迫使每一步都隐式地重构当前状态，而现有的显式状态方法则容易受到不一致的状态覆盖的影响。为解决这些局限，我们提出了 StateRAG，一个通过带有校验更新的显式检索状态来协调规划器、导航器、检索器和验证器的框架。这些角色针对四个状态字段提出更新：计划、检索轨迹、证据以及充分性评估。一个专用的控制器将每个更新提案与已接受的状态进行校验，拒绝不一致的更新，并在查询预算内引导修订。最终的读取器则严格基于经过验证的证据进行回答。在共享图后端上的消融实验表明，StateRAG 的表现优于自由形式……

    arXiv:2605.25379v4 Announce Type: replace  Abstract: Multi-step retrieval-augmented generation requires tracking evolving information needs, selected evidence, and unresolved gaps. However, free-form trajectories conflate this history, forcing each step to implicitly reconstruct the current state, while existing explicit-state methods remain vulnerable to inconsistent state overwrites. To address these limitations, we introduce StateRAG, a framework that coordinates a Planner, Navigator, Retriever, and Verifier through an explicit retrieval state with checked updates. These roles propose updates across four state fields: the plan, retrieval trace, evidence, and sufficiency assessment. A dedicated controller validates each proposal against the accepted state, rejecting inconsistent updates and guiding revisions within the query budget. A final Reader then answers strictly from the validated evidence. Ablation experiments on a shared graph backend show that StateRAG outperforms free-form
    
[^164]: AnyMo：几何感知、与传感配置无关的真实场景人体运动建模

    AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild

    [https://arxiv.org/abs/2605.22715](https://arxiv.org/abs/2605.22715)

    AnyMo提出了一种几何感知的框架，通过基于物理的IMU仿真、配对位置视图与掩码观测预训练图编码器、将多位置IMU数据转化为全身运动token并与LLM对齐，实现了不依赖传感器配置（佩戴位置、朝向、硬件等）的人体运动建模。

    

    随着可穿戴和移动设备日益融入日常生活，它们为在真实场景中持续感知人体运动提供了实用的途径。但惯性信号高度依赖于传感配置，包括身体佩戴位置、安装方式、传感器朝向、设备硬件和采样协议。这种对配置的依赖使得难以学习能够跨设备和跨数据集迁移的运动表示，也限制了可穿戴IMU在封闭集识别之外的更广泛应用。我们提出AnyMo，一个用于与传感配置无关的人体运动建模的几何感知框架。AnyMo利用基于物理的IMU仿真在密集的身体表面位置上生成多样且合理的合成信号，从配对的合成佩戴位置视图和掩码部分观测中预训练图编码器，将多位置IMU数据标记化为全身运动token，并将这些token与LLM对齐以实现运动-语言（摘要在此处截断）

    arXiv:2605.22715v3 Announce Type: replace-cross  Abstract: As wearable and mobile devices become increasingly embedded in daily life, they offer a practical way to continuously sense human motion in the wild. But inertial signals are highly dependent on the sensing setup, including body location, mounting position, sensor orientation, device hardware, and sampling protocol. This setup dependence makes it difficult to learn motion representations that transfer across devices and datasets, and limits the broader use of wearable IMUs beyond closed-set recognition. We introduce AnyMo, a geometry-aware framework for setup-agnostic human motion modeling. AnyMo uses physics-grounded IMU simulation over dense body-surface placements to generate diverse and plausible synthetic signals, pre-trains a graph encoder from paired synthetic placement views and masked partial observations, tokenizes multi-position IMU into full-body motion tokens, and aligns these tokens with an LLM for motion-language
    
[^165]: Firefly：从真实API中点亮大规模可验证工具调用数据生成

    Firefly: Illuminating Large-Scale Verified Tool-Call Data Generation from Real APIs

    [https://arxiv.org/abs/2605.17558](https://arxiv.org/abs/2605.17558)

    FireFly通过先让大模型探索真实API再从观察结果反向合成任务的新范式，从真实MCP服务器生成在构造上保证标签正确性的大规模工具调用训练数据。

    

    训练工具调用智能体需要大规模带有可验证标签的轨迹数据，然而现有方法要么合成偏离真实API行为的环境，要么生成缺乏真实结果以供验证的任务。我们提出了FireFly，一个从真实世界MCP服务器生成可验证工具调用数据的流水线。我们的关键洞察是反转标准的合成流水线：不是先生成任务并期望它们可解，而是先让一个强大的大语言模型沿着图引导的DAG结构探索真实API，然后从观察到的结果反向合成任务，从而在构造上保证标签的正确性。为了应对真实世界工具空间的规模（约1000个工具），我们构建了一个成对工具图并采样子DAG，将探索聚焦在语义连贯的工作流上。为了解决在线API中的环境漂移问题，我们构建了一个检索增强的模拟器，缓存所有探索……

    arXiv:2605.17558v2 Announce Type: replace-cross  Abstract: Training tool-calling agents requires large-scale trajectory data with verifiable labels, yet existing approaches either synthesize environments that diverge from real API behavior or generate tasks without ground-truth outcomes for verification. We present FireFly, a pipeline for generating verified tool-call data from real-world MCP servers. Our key insight is to invert the standard synthesis pipeline: rather than generating tasks and hoping they are solvable, we first let a strong LLM explore real APIs along graph-guided DAG structures, then synthesize tasks backward from observed outcomes, guaranteeing label correctness by construction. To handle the scale of real-world tool spaces (${\sim}$1,000 tools), we build a pairwise tool graph and sample sub-DAGs to focus exploration on semantically coherent workflows. To address environment drift in live APIs, we construct a retrieval-augmented simulator that caches all exploration
    
[^166]: AI中介的交流能够引导集体舆论

    AI-Mediated Communication Can Steer Collective Opinion

    [https://arxiv.org/abs/2605.16245](https://arxiv.org/abs/2605.16245)

    该论文通过实证研究与数学建模表明，大型语言模型在中介人与人之间的交流时会引入方向性偏见（如偏向支持枪支管制、反对无神论），并因此可能引导社交网络中集体舆论的形成。

    

    生成式人工智能（AI）正日益融入人们交换意见的在线平台；大型语言模型（LLM）已经在LinkedIn上润色用户的帖子，并为X上分享的内容提供背景信息。尽管已有研究表明AI能够表达带有偏见的观点，并在人机交互中影响个人的意见，但对于AI在中介人与人之间交流时对集体舆论形成的影响，研究关注还比较少。我们通过实证分析与理论分析相结合的方式来填补这一空白。我们在实证上表明，来自多个流行家族的大型语言模型在被要求编辑涉及争议话题的人类撰写文本时会引入方向性偏差，例如使文本倾向于支持枪支管制、反对无神论。基于这一观察，我们提出了一个舆论动力学的数学模型，其中一个AI系统位于社交网络上用户之间。

    arXiv:2605.16245v2 Announce Type: replace-cross  Abstract: Generative artificial intelligence (AI) is increasingly integrated into the online platforms where humans exchange opinions; large language models (LLMs) have been polishing users' posts on LinkedIn and providing context for content shared on X. While prior work has shown that AI can express biased opinions and shape individuals' opinions during human-AI interactions, less attention has been paid to its influence on collective opinion formation when mediating human-to-human communication. We address this gap via a combination of empirical and theoretical analyses. We show empirically that LLMs from multiple popular families introduce directional biases when instructed to edit human-written texts on contested topics, for example, nudging texts in favor of gun control and against atheism. Building on this observation, we introduce a mathematical model of opinion dynamics in which an AI system sits between users on a social networ
    
[^167]: 考察数据与训练方法对涌现性失对齐与阈下失对齐的影响

    Examining the Impact of Data and Training Methods on Emergent and Subliminal Misalignment

    [https://arxiv.org/abs/2605.12798](https://arxiv.org/abs/2605.12798)

    该论文通过构建按“领域-任务”网格组织的数据集，系统研究了训练数据、评估数据与训练方法如何影响涌现性失对齐（EM）与阈下学习（SL）中失对齐的转移，并发现 EM 主要沿任务而非领域进行转移。

    

    对语言模型进行后训练可能引发远超预期训练目标的惊人行为变化。我们研究了两个尤为显著的例子：涌现性失对齐（EM），即在狭窄数据集上进行微调会导致模型在无关情境中表现出失对齐行为；以及阈下学习（SL），即学生模型通过看似无害的训练数据继承教师模型的失对齐特性。我们考察了训练数据、评估数据和训练方法如何决定失对齐在数据与模型之间于何处转移以及转移的强度。我们用提示词的领域（其主题语境）和任务（模型所执行的功能）来刻画数据，并构建了按“领域-任务”网格组织的自然语言数据集与合成数据集。我们发现，与共享微调数据领域的提示词相比，EM 更多地转移到与微调数据共享相同任务的提示词上。在合成环境中，模型已……（原文摘要在此处截断）

    arXiv:2605.12798v2 Announce Type: replace-cross  Abstract: Post-training a language model can induce surprising behavioral changes far beyond the intended training target. We study two particularly striking examples: emergent misalignment (EM), where fine-tuning on a narrow dataset leads to misaligned behavior in unrelated contexts, and subliminal learning (SL), where a student model inherits a teacher's misalignment through seemingly benign training data. We examine how the training data, evaluation data, and training method shape where and how strongly misalignment transfers between data and models. We characterize the data by the domain of a prompt (its topical context) and its task (the function the model performs), and we build natural-language and synthetic datasets organized as domain-task grids. We find that EM transfers more strongly to prompts that share the task of the fine-tuning data than to prompts that share its domain. In the synthetic setting, tasks that the model has 
    
[^168]: GSQ：通过Gumbel-Softmax采样实现大语言模型的高精度低比特标量量化

    GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling

    [https://arxiv.org/abs/2604.18556](https://arxiv.org/abs/2604.18556)

    提出了GSQ这一训练后标量量化方法，利用Gumbel-Softmax松弛联合学习逐坐标网格分配与逐组缩放因子，使简单的标量量化在低比特精度下就能达到与复杂的第二代向量量化方法相当的高准确率。

    

    量化已成为高效部署大语言模型（LLM）的标准工具，尤其在本地推理场景中，模型通常以每参数2-3比特的精度进行服务。目前最先进的技术分为两大阵营：一类是简单的标量量化技术（如GPTQ或AWQ），它们被广泛部署但精度在每参数3-4比特时趋于瓶颈；另一类是“第二代”向量或网格量化方法（如QTIP、GPTVQ和AQLM），它们虽能推动精度上限，但在实现和扩展方面非常困难。本文探究这一差距是否是根本性的，或者一个经过精心优化的标量量化器能否挽回其中大部分差距。我们给出了肯定的答案，提出了GSQ（Gumbel-Softmax量化），这是一种训练后标量量化方法，它利用离散网格的Gumbel-Softmax松弛来联合学习逐坐标的网格分配和逐组的缩放因子。GSQ的性能可以匹敌……

    arXiv:2604.18556v3 Announce Type: replace  Abstract: Quantization has become a standard tool for efficient LLM deployment, especially for local inference, where models are now routinely served at 2-3 bits per parameter. The state of the art is currently split into simple scalar quantization techniques, such as GPTQ or AWQ, which are widely deployed but plateau in accuracy at 3-4 bits per parameter (bpp), and "second-generation" vector- or trellis-quantized methods, such as QTIP, GPTVQ and AQLM, which push the accuracy frontier but are notoriously hard to implement and to scale. In this paper, we ask whether this gap is fundamental, or whether a carefully optimized $\textit{scalar}$ quantizer can recover most of it. We answer in the affirmative, by introducing GSQ (Gumbel-Softmax Quantization), a post-training scalar quantization method which jointly learns the per-coordinate grid assignments and the per-group scales using a Gumbel-Softmax relaxation of the discrete grid. GSQ matches th
    
[^169]: 多用户大语言模型智能体

    Multi-User Large Language Model Agents

    [https://arxiv.org/abs/2604.08567](https://arxiv.org/abs/2604.08567)

    该论文首次系统性地研究了多用户大语言模型智能体，将LLM智能体的多用户交互形式化为多主体决策问题，以应对多用户场景下不可避免的冲突、信息不对称和隐私约束。

    

    大语言模型（LLM）及基于LLM的智能体正越来越多地被部署为规划和决策中的助手，然而现有的大多数系统都隐含地针对单主体交互范式进行了优化，即模型被设计为满足单一主导用户的目标，该用户的指令被视为唯一的权威与效用来源。然而，随着这些模型被集成到团队工作流和组织工具中，它们日益需要同时服务于多个用户，而每个用户拥有不同的角色、偏好和权限级别，从而形成了多用户、多主体的环境，其中不可避免地存在冲突、信息不对称和隐私约束。在本工作中，我们对多用户LLM智能体进行了首次系统性研究。我们首先将LLM智能体的多用户交互形式化为一个多主体决策问题，其中一个智能体必须同时考虑多个具有潜在……（原文摘要此处截断）

    arXiv:2604.08567v3 Announce Type: replace  Abstract: Large language models (LLMs) and LLM-based agents are increasingly deployed as assistants in planning and decision making, yet most existing systems are implicitly optimized for a single-principal interaction paradigm, in which the model is designed to satisfy the objectives of one dominant user whose instructions are treated as the sole source of authority and utility. However, as they are integrated into team workflows and organizational tools, they are increasingly required to serve multiple users simultaneously, each with distinct roles, preferences, and authority levels, leading to multi-user, multi-principal settings with unavoidable conflicts, information asymmetry, and privacy constraints. In this work, we present the first systematic study of multi-user LLM agents. We begin by formalizing multi-user interaction with LLM agents as a multi-principal decision problem, where a single agent must account for multiple users with po
    
[^170]: STDec：面向扩散大语言模型的时空稳定性引导解码方法

    STDec: Spatio-Temporal Stability Guided Decoding for dLLMs

    [https://arxiv.org/abs/2604.06330](https://arxiv.org/abs/2604.06330)

    该论文提出STDec方法，利用dLLM解码中“新token靠近已解码邻居、预测ID跨步骤保持一致”的时空稳定性，通过空间感知与时序感知解码动态自适应地调整阈值，从而改进解码效果。

    

    扩散大语言模型取得了快速进展，被视为自回归范式的一种有前景的替代方案。然而，大多数dLLM解码器仍采用全局置信度阈值，既没有显式建模来自邻近已解码状态的局部上下文，也没有建模预测token ID跨步骤的时序一致性。为解决这一问题，我们提出了一种时空稳定性引导的解码方法，命名为STDec。我们观察到dLLM解码过程中存在显著的时空稳定性：新解码的token往往位于已解码邻居附近，且其预测的token ID通常在多个去噪步骤中保持一致。受这种稳定性的启发，STDec引入了空间感知解码和时序感知解码：空间感知解码通过聚合邻近token的已解码状态，动态生成token自适应的阈值；时序感知解码则对预测ID跨步骤保持一致的token放宽解码阈值。

    arXiv:2604.06330v2 Announce Type: replace  Abstract: Diffusion Large Language Models (dLLMs) have achieved rapid progress, viewed as a promising alternative to the autoregressive paradigm. However, most dLLM decoders still adopt a global confidence threshold, and do not explicitly model local context from neighboring decoded states or temporal consistency of predicted token IDs across steps. To address this issue, we propose a spatio-temporal stability guided decoding approach, named STDec. We observe strong spatio-temporal stability in dLLM decoding: newly decoded tokens tend to lie near decoded neighbors, and their predicted IDs often remain consistent across several denoising steps. Inspired by this stability, our STDec introduces spatial-aware decoding and temporal-aware decoding. The spatial-aware decoding dynamically generates the token-adaptive thresholds by aggregating the decoded states of nearby tokens. The temporal-aware decoding relaxes the decoding thresholds for tokens wh
    
[^171]: 面向连续提示词搜索的无投影演化策略

    Projection-Free Evolution Strategies for Continuous Prompt Search

    [https://arxiv.org/abs/2603.13786](https://arxiv.org/abs/2603.13786)

    本文通过实证发现随机投影无法充分捕捉提示词空间的内在低维结构，进而提出一种基于演化策略的无投影连续提示词搜索方法，直接在完整提示词空间中优化并利用与内在维度相适配的自适应机制。

    

    连续提示词搜索为自然语言处理任务中的传统参数微调提供了一种计算高效的替代方案。然而，其现实有效性可能因目标函数景观的黑盒特性和固有的高维性而受到显著阻碍。现有方法通常通过将搜索限制在随机投影后的低维子空间中来缓解这些挑战。然而，投影机制的有效性及其背后的动机仍不明确。在本文中，我们首先通过实证研究表明，尽管提示词空间确实具有低维结构，但随机投影无法充分捕捉这一本质结构。受此发现的启发，我们提出了一种基于演化策略的无投影提示词搜索方法。该方法直接在完整的提示词空间中进行优化，并采用一种根据内在维度进行校准的自适应机制。

    arXiv:2603.13786v2 Announce Type: replace  Abstract: Continuous prompt search offers a computationally efficient alternative to conventional parameter tuning in natural language processing tasks. Nevertheless, its practical effectiveness can be significantly hindered by the black-box nature and the inherent high-dimensionality of the objective landscapes. Existing methods typically mitigate these challenges by restricting the search to a randomly projected low-dimensional subspace. However, the effectiveness and underlying motivation of the projection mechanism remain ambiguous. In this paper, we first empirically demonstrate that despite the prompt space possessing a low-dimensional structure, random projections fail to adequately capture this essential structure. Motivated by this finding, we propose a projection-free prompt search method based on evolutionary strategies. By directly optimizing in the full prompt space with an adaptation mechanism calibrated to the intrinsic dimensio
    
[^172]: 基于贝叶斯自适应提问的冷启动个性化

    Cold-Start Personalization via Bayesian Adaptive Questioning

    [https://arxiv.org/abs/2602.15012](https://arxiv.org/abs/2602.15012)

    提出CAPE方法，先离线从完整用户画像中学习偏好相关性的结构化世界模型，再在线通过免训练的贝叶斯推断选择信息量最大的问题并预测完整偏好画像，即使使用简单的线性回归信念模型也能大幅超越端到端强化学习。

    

    冷启动个性化需要在没有用户特定历史数据可用的情况下，仅通过极少量的交互来推断用户偏好。可能的偏好空间非常庞大，但用户只关心其中稀疏的子集，且很少会主动明确表达这些偏好；再加上有限的交互预算，这使得偏好引出（elicitation）变得极具挑战性。我们的关键洞察是：偏好在不同用户群体之间表现出可预测的结构；例如，希望获得详细解释的用户通常也重视完整的示例演算。我们提出了CAPE（冷启动自适应偏好引出，Cold-start Adaptive Preference Elicitation），该方法先离线地从完整用户画像中学习一个关于偏好相关性的结构化世界模型，然后在线执行无需训练的贝叶斯推断，以选择信息量最大的问题并预测完整的偏好画像，甚至包括从未询问过的维度。即使是简单的信念模型实例（如线性回归），其表现也能大幅超越端到端的强化学习方法。

    arXiv:2602.15012v2 Announce Type: replace-cross  Abstract: Cold-start personalization requires inferring preferences from minimal interaction when no user-specific historical data is available. The space of possible preferences is vast, yet users care about only a sparse subset and rarely articulate them upfront; combined with limited interaction budgets, this makes preference elicitation challenging. Our key insight is that preferences exhibit predictable structure across populations; e.g., users who want detailed explanations often also value worked examples. We propose CAPE (Cold-start Adaptive Preference Elicitation), which learns a structured world model of preference correlations offline from complete profiles, then performs training-free Bayesian inference online to select informative questions and predict complete preference profiles, including dimensions never asked about. Even simple belief model instantiations (e.g., linear regression) substantially outperform end-to-end RL.
    
[^173]: 推理型大语言模型会拒绝它们在长上下文中推断出的内容吗？

    Do Reasoning LLMs Refuse What They Infer in Long Contexts?

    [https://arxiv.org/abs/2602.08874](https://arxiv.org/abs/2602.08874)

    该论文提出“组合推理攻击”方法，发现推理型大语言模型虽然会拒绝明确陈述的有害请求，但当有害意图被分解为碎片分散在长上下文中时，模型的拒绝率会显著下降，暴露了现有安全评估的重要盲区。

    

    长上下文大语言模型能够推断出未明确陈述的目标。这一能力对于文档、代码和工具调用轨迹的推理非常有用，但也引入了安全风险。有害意图可以分布在上下文中，只有在模型整合相关信息后才会显现。现有的安全评估主要关注明确陈述的有害请求，因此无法充分捕捉这种失效模式。我们提出了组合推理攻击，这是一种长上下文攻击设置，其中有害请求被分解为语义不完整的片段并嵌入到长上下文中。最终的查询是中性的，模型必须检索并组合这些片段才能推断出隐含的有害目标。实验表明，当有害请求被明确陈述时，模型通常会拒绝它们；然而，当相同的目标必须从片段中重构时，拒绝率会显著下降。

    arXiv:2602.08874v3 Announce Type: replace  Abstract: Long-context LLMs can infer objectives that are not explicitly stated. This capability is useful for reasoning over documents, code, and tool traces, but it also introduces a safety risk. Harmful intent can be distributed across a context and become apparent only after the model combines the relevant information. Existing safety evaluations largely focus on explicitly stated harmful requests and therefore do not fully capture this failure mode. We introduce compositional reasoning attacks, a long-context attack setting in which harmful requests are decomposed into semantically incomplete fragments and embedded in long contexts. The final query is neutral, and the model must retrieve and combine the fragments to infer the implied harmful objective. Empirically, we show that models generally refuse harmful requests when they are explicitly stated. However, refusal rates drop when the same objectives must be reconstructed from fragments
    
[^174]: 自我改进的训练与思考

    Self-Improving Training & Thinking

    [https://arxiv.org/abs/2601.21343](https://arxiv.org/abs/2601.21343)

    该论文提出一种将安全性、事实性和推理等能力更早融入预训练与中期训练的新方法，通过利用现有强模型重写数据和评判模型输出，并引入“自我改进的后思考”机制，在训练全流程中提升模型的推理能力。

    

    传统上，大型语言模型采用分阶段训练：先在原始文本上进行预训练，再进行后训练以学习指令遵循和推理能力。然而，这种分离造成了根本性的局限：许多理想的行为，如安全性、事实准确性、整体生成质量和推理能力，只在训练后期才被加入，尽管早期学习的模式在很大程度上塑造了模型的能力。为解决这一问题，我们提出了一种新的预训练和中期训练方法，将这些行为更早地融入训练过程。我们利用一个现有的强大的经过后训练的模型来重写数据并评判策略模型的输出（rollouts），从而在训练中更早地引入思考和强化学习。我们进一步通过引入“自我改进的后思考”（self-improving post-thinking）的概念，在所有训练阶段提升推理能力：即在生成之后进行推理，以提高模型的推理和自我评判能力。在实验中，我们证明了这些方法……

    arXiv:2601.21343v4 Announce Type: replace-cross  Abstract: Large language models are classically trained in stages: pretraining on raw text followed by post-training for instruction following and reasoning. However, this separation creates a fundamental limitation: many desirable behaviors such as safety, factuality, overall generation quality, and reasoning ability are only added at a late stage, even though the patterns learned earlier strongly shape a model's capabilities. To tackle this issue, we introduce a new way to pretrain and mid-train models that incorporates these behaviors earlier. We utilize an existing strong, post-trained model to both rewrite data and to judge policy model rollouts, thus introducing thinking and reinforcement earlier in training. We further improve reasoning at all stages by introducing the concept of self-improving post-thinking: improving the model's reasoning and self-judging ability by reasoning after generation. In our experiments, we show these a
    
[^175]: 追踪大型语言模型中的道德基础

    Tracing Moral Foundations in Large Language Models

    [https://arxiv.org/abs/2601.05437](https://arxiv.org/abs/2601.05437)

    该研究通过对14个大型语言模型进行逐层分析、稀疏自编码器特征识别和因果引导干预，证明模型内部存在与人类道德判断相一致的道德基础概念结构，而非仅仅是表面的“道德模仿”。

    

    大型语言模型常常产生类似人类的道德判断，但目前尚不清楚这究竟反映了其内部的道德概念结构，还是仅仅是一种表面的“道德模仿”。本研究以道德基础理论（Moral Foundations Theory, MFT）作为分析框架，研究了道德基础如何在14个基础模型和指令微调模型中被编码、组织和表达，这些模型涵盖四个模型家族（Llama、Qwen2.5、Qwen3-MoE、Mistral），参数规模从7B到70B。我们采用了多层次的研究方法，结合：(i) 对MFT概念表示的逐层分析及其与人类道德感知的对齐程度；(ii) 在残差流上使用预训练的稀疏自编码器（SAEs）来识别支持道德概念的稀疏特征；(iii) 使用稠密MFT向量与稀疏SAE特征进行因果引导干预。我们发现，模型表示和区分道德基础的方式与人类道德判断相一致，并且这种道德几何结构……

    arXiv:2601.05437v4 Announce Type: replace-cross  Abstract: Large language models often produce human-like moral judgments, but it is unclear whether this reflects an internal conceptual structure or superficial ``moral mimicry.'' Using Moral Foundations Theory (MFT) as an analytic framework, we study how moral foundations are encoded, organized, and expressed across 14 base and instruction-tuned LLMs spanning four model families (Llama, Qwen2.5, Qwen3-MoE, Mistral) and scales from 7B to 70B. We employ a multi-level approach combining (i) layer-wise analysis of MFT concept representations and their alignment with human moral perceptions, (ii) pretrained sparse autoencoders (SAEs) over the residual stream to identify sparse features that support moral concepts, and (iii) causal steering interventions using dense MFT vectors and sparse SAE features. We find that models represent and distinguish moral foundations in a manner that aligns with human judgments, and that this moral geometry na
    
[^176]: DermaVQA-DAS：皮肤病评估模式（DAS）及面向患者生成皮肤病图像的封闭式问答与分割数据集

    DermaVQA-DAS: Dermatology Assessment Schema (DAS) & Datasets for Closed-Ended Question Answering & Segmentation in Patient-Generated Dermatology Images

    [https://arxiv.org/abs/2512.24340](https://arxiv.org/abs/2512.24340)

    该论文提出了DermaVQA-DAS数据集及专家开发的皮肤病评估模式（DAS），通过36个高层级和27个细粒度的标准化评估问题，为患者生成的皮肤病图像上的封闭式问答和皮损分割两大任务提供了专家标注数据与基准。

    

    皮肤病学图像分析的最新进展很大程度上得益于大规模标注数据集；然而，现有的大多数基准数据集主要聚焦于皮肤镜图像，缺乏患者自述的查询和临床背景信息，这限制了其在以患者为中心的医疗护理中的适用性。为填补这一空白，我们提出了DermaVQA-DAS，它是DermaVQA数据集的扩展版本，支持两项互补任务：封闭式问答（QA）和皮肤病皮损分割。本工作的核心是皮肤病评估模式，这是一个由专家开发的新型框架，能够以结构化和标准化的形式系统性捕捉具有临床意义的皮肤病特征。DAS包含36个高层级和27个细粒度的评估问题，并配有英语和中文的选择题选项。利用DAS，我们为封闭式问答和分割任务提供了专家标注的数据集，并对最先进的多模态（模型进行了基准测试，摘要在此处截断）。

    arXiv:2512.24340v2 Announce Type: replace-cross  Abstract: Recent advances in dermatological image analysis have been driven by large-scale annotated datasets; however, most existing benchmarks focus on dermatoscopic images and lack patient-authored queries and clinical context, limiting their applicability to patient-centered care. To address this gap, we introduce DermaVQA-DAS, an extension of the DermaVQA dataset that supports two complementary tasks: closed-ended question answering (QA) and dermatological lesion segmentation. Central to this work is the Dermatology Assessment Schema (DAS), a novel expert-developed framework that systematically captures clinically meaningful dermatological features in a structured and standardized form. DAS comprises 36 high-level and 27 fine-grained assessment questions, with multiple-choice options in English and Chinese. Leveraging DAS, we provide expert-annotated datasets for both closed QA and segmentation and benchmark state-of-the-art multimo
    
[^177]: 理解并解决个体级自然语言理解中的标签错误

    Understanding and Tackling Label Errors in Individual-Level Nature Language Understanding

    [https://arxiv.org/abs/2502.13297](https://arxiv.org/abs/2502.13297)

    本文提出了一种基于个体级因素的NLU标注指南，通过纳入同一个体的所有帖子来重新标注立场检测和情感分析数据集，并揭示了现有数据集中高达31.7%和23.3%的标签错误率。

    

    自然语言理解（NLU）是使机器能够理解人类语言的任务。一些任务，如立场检测和情感分析，与个体的主观视角密切相关，因此被称为个体级NLU。以往这些任务通常被简化为文本级NLU任务，忽略了个体因素。这不仅使推理变得困难且缺乏可解释性，而且在构建数据集时往往导致大量的标签错误。为了解决上述局限性，我们提出了一种基于个体级因素的新NLU标注指南。具体而言，我们纳入同一个体的其他帖子，并在综合考虑该个体的所有帖子后对其主观视角进行标注。我们使用该指南扩展并重新标注了立场检测和基于主题的情感分析数据集。我们发现样本中的错误率分别高达31.7%和23.3%。

    arXiv:2502.13297v2 Announce Type: replace-cross  Abstract: Natural language understanding (NLU) is a task that enables machines to understand human language. Some tasks, such as stance detection and sentiment analysis, are closely related to individual subjective perspectives, thus termed individual-level NLU. Previously, these tasks are often simplified to text-level NLU tasks, ignoring individual factors. This not only makes inference difficult and unexplainable but often results in a large number of label errors when creating datasets. To address the above limitations, we propose a new NLU annotation guideline based on individual-level factors. Specifically, we incorporate other posts by the same individual and then annotate individual subjective perspectives after considering all individual posts. We use this guideline to expand and re-annotate the stance detection and topic-based sentiment analysis datasets. We find that error rates in the samples were as high as 31.7\% and 23.3\%
    
[^178]: 线性时间最小贝叶斯风险解码与参考聚合

    Linear-time Minimum Bayes Risk Decoding with Reference Aggregation

    [https://arxiv.org/abs/2402.04251](https://arxiv.org/abs/2402.04251)

    本文提出了一种线性时间的最小贝叶斯风险解码方法，通过使用聚合参考表示来近似配对度量分数，将复杂度降低到线性级别，同时在保持大部分质量增益的同时提高了解码的效率。

    

    最小贝叶斯风险（MBR）解码是一种文本生成技术，已被证明可以提高机器翻译的质量，但即使使用基于采样的近似方法也很昂贵。除了需要大量采样序列外，还需要对效用度量进行配对计算，这具有二次复杂度。在本文中，我们提出使用聚合参考表示计算近似的配对度量分数。这将效用估计的复杂度从$O(n^2)$降低到$O(n)$，同时在经验上保持了MBR解码的大部分质量提升。我们在https://github.com/ZurichNLP/mbr上发布了我们的源代码。

    Minimum Bayes Risk (MBR) decoding is a text generation technique that has been shown to improve the quality of machine translations, but is expensive, even if a sampling-based approximation is used. Besides requiring a large number of sampled sequences, it requires the pairwise calculation of a utility metric, which has quadratic complexity. In this paper, we propose to approximate pairwise metric scores with scores calculated against aggregated reference representations. This changes the complexity of utility estimation from $O(n^2)$ to $O(n)$, while empirically preserving most of the quality gains of MBR decoding. We release our source code at https://github.com/ZurichNLP/mbr
    

