# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data Scarcity and Model Sparsity: Mixtures-of-Experts Overfit More to Repeated Data](https://arxiv.org/abs/2609.11917) | 该研究发现混合专家模型（MoE）相比密集模型更容易因训练数据重复而过拟合，且这种退化随模型稀疏度（由总参数量而非活跃参数量决定）的增加而加剧。 |
| [^2] | [Distance generalization in transformers: why bother with positional encoding?](https://arxiv.org/abs/2609.11913) | 本研究通过合成延迟复制任务系统探究Transformer的距离泛化能力，考察RoPE、ALiBi等位置编码方案与无位置编码在距离分辨率上的差异、训练数据多样性的影响，以及距离迁移学习的正负效应。 |
| [^3] | [MindTopo: Can Foundation Models Reason in Topological Space?](https://arxiv.org/abs/2609.11900) | 该论文提出了MindTopo基准，基于认知科学与形式拓扑学，从连续性、分离性、有序性、包含性和纽结五个拓扑属性出发，在推理与规划两个认知层面对14个多模态大语言模型的拓扑空间能力进行系统评估。 |
| [^4] | [Nuha-Speech: Building General-Purpose Arabic Speech-LLMs](https://arxiv.org/abs/2609.11892) | 该论文提出Nuha-Speech计划，通过构建包含150万样本的阿拉伯语语音问答语料库、基于Qwen-Omni模型微调以及设计专门的评估框架，为通用阿拉伯语语音大语言模型建立了覆盖数据、训练到评估全流程的基础设施。 |
| [^5] | [Domain-Specific Hallucination Detection in Large Language Models](https://arxiv.org/abs/2609.11878) | 提出了一种结合微调DeBERTa-v3分类、蒙特卡洛Dropout不确定性量化和温度缩放校准的多信号幻觉检测流水线，在HaluEval基准上取得F1=0.915的高性能，并结合直接偏好优化（DPO）进一步改进模型表现。 |
| [^6] | [Biology-in-the-loop: Amortized Adaptive Hit Discovery in CRISPR Screens](https://arxiv.org/abs/2609.11877) | 本文提出了包含1,389个CRISPR筛选实验的大规模基准AssayBench-Loop，并在此基础上构建了AssayLoop顺序实验设计框架，利用跨历史实验训练的transformer摊销式采集策略，实现CRISPR筛选中受限预算下的自适应命中物发现。 |
| [^7] | [The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement](https://arxiv.org/abs/2609.11873) | 本文提出递归自我改进（RSI）概念及其从改进执行自主性到递归元改进的发展路线图，通过Headroom-Closed指数揭示现有大语言模型的局限，并结合行业实践识别出实现真正AI自我改进的关键挑战。 |
| [^8] | [Augustinian BabyLM: What Ostensive Definition Can and Cannot Teach a Small Language Model](https://arxiv.org/abs/2609.11870) | 本研究将圣奥古斯丁的“实指定名”词义学习思想应用于小型语言模型，发现视觉初始化的词嵌入会留下持续到训练结束的印记，且仅在物体属性知识等零样本任务中带来提升，而对大多数语法能力基准测试没有影响。 |
| [^9] | [Epistemic orientation predicts legislative effectiveness among members of the US Congress](https://arxiv.org/abs/2609.11865) | 本研究通过测量国会议员演讲和推文中证据导向与直觉导向语言的差异（EMI分数），发现议员的认知取向与其立法效能显著相关，表明基于证据的沟通风格能够预测立法者的实际工作成效。 |
| [^10] | [RetroThinker: Enabling Retrospective Thinking in Speech LLMs](https://arxiv.org/abs/2609.11864) | 该论文提出了RetroThinker多阶段后训练框架，使流式语音大语言模型能够在推理过程中自我验证并即时修正思维链推理步骤，从而在满足实时语音交互延迟约束的同时提升复杂推理能力。 |
| [^11] | [IndicTriMix: Developing Language Identification Datasets and Models for Tri-Language Code-Mixing](https://arxiv.org/abs/2609.11851) | 该论文针对印地语、古吉拉特语和孟加拉语的三语混合文本，将词元级语言识别形式化为序列标注任务并微调 MuRIL 和 XLM-RoBERTa 模型，同时发布了人工标注的基准数据集和两种三语混合文本生成方法。 |
| [^12] | [Target leakage, not model class, explains reported accuracy in survey-based cardiovascular screening: a leakage-tiered audit of glass-box and tabular foundation models](https://arxiv.org/abs/2609.11838) | 该研究通过对十种模型类别在五级泄漏风险特征层级上的系统审计发现，基于国民健康调查的心血管筛查模型所报告的高准确率主要由目标泄漏而非模型类别驱动，移除两个诊断后特征导致所有模型AUROC下降约0.05。 |
| [^13] | [SpecGuard: Inference-Time Backdoor Detection For Free](https://arxiv.org/abs/2609.11799) | SpecGuard巧妙地重新利用投机解码中的验证过程，在零额外模型计算成本下实现推理时的大语言模型后门检测。 |
| [^14] | [Beyond Word Error Rate: A Switch Aware Evaluation of ASR and Audio Language Models on English Yoruba Code-Switched Speech](https://arxiv.org/abs/2609.11786) | 该论文提出了一套切换感知的评估指标（包括切换入口标记错误率SETER等），揭示总体词错误率会掩盖码转换语音识别的真实表现，且领先的音频语言模型虽然与最佳ASR模型的WER相当，却在所有码转换相关指标上显著更优。 |
| [^15] | [Whisper-Based Speech Transcription from Videos Across Multiple Languages for Cross-Cultural Understanding](https://arxiv.org/abs/2609.11772) | 本文提出了基于Whisper的多语言视频语音转录技术，使缺乏深厚语音处理专业知识的跨文化工具开发者也能轻松创建多语言转录文本，在七种语言上实现了平均30%的转录错误率。 |
| [^16] | [The widening evaluation gap in medical large language model research 2023 to 2026](https://arxiv.org/abs/2609.11770) | 该研究揭示医学大语言模型研究的评估滞后从1.33个季度扩大至6.08个季度，其中严谨的随机试验反而倾向于评估更过时的模型，凸显了方法学严谨性与模型时效性之间的根本矛盾。 |
| [^17] | [Recognizing Is Not Reversing: A Controlled Inversion Test of Fact-Preserving News Framing](https://arxiv.org/abs/2609.11769) | 该研究通过受控逆转实验发现，大语言模型虽能较好地保持事实并识别新闻框架，却几乎无法逆转已知的框架变换（逆转率仅约0.044–0.068），证明“识别框架”与“逆转框架”是截然不同的能力。 |
| [^18] | [A Unified Per-Token Gating Family for On-Policy Distillation: FKL/RKL Mixing with Multi-Channel and Bias Coefficients](https://arxiv.org/abs/2609.11768) | 该论文提出了一个统一的四系数逐词元门控参数化方法，将EOPD和ToDi作为其一维限制的特殊情形统一起来，并通过多通道组合与显式偏置这两个额外自由度，在在线策略蒸馏实验中显著超越了单通道门控限制方法。 |
| [^19] | [Component-Aware Differential Privacy for Federated Multilingual Speech-LLMs](https://arxiv.org/abs/2609.11762) | 提出α-split双池差分隐私分配方法，通过将声学编码器与语言解码器参数归入独立裁剪池，解决了语音大语言模型联邦训练中因组件间更新范数差异巨大导致的预算坍塌问题。 |
| [^20] | [RAG-Safety-Bench: Reliable Evaluation of Retrieval-Augmented LLM Safety](https://arxiv.org/abs/2609.11758) | 提出了RAG-Safety-Bench基准测试，通过消除检索器质量的干扰并将问题分解为四种条件（非RAG、含答案文档、相关无答案文档、随机文档），可靠地评估检索增强生成对大语言模型安全性的影响。 |
| [^21] | [SIRF: A Spec-Internalized Risk Foundation Model for Industrial Content Risk Control](https://arxiv.org/abs/2609.11752) | SIRF通过持续预训练将平台风控规范内化到模型权重中，无需人工标注即可在超低延迟、仅输出判定的部署形态下实现高精度风险处置，仅用约7000万token的CPT便将P95精确率下的黑名单召回率提升15.1个百分点且不损害通用能力。 |
| [^22] | [LOCUS: Task-Aware Low-Rank Post-Training for Token-Efficient Language Generation](https://arxiv.org/abs/2609.11739) | LOCUS通过选择任务感知的低秩适配子空间进行后训练，在不修改偏好对齐损失且仅更新不到0.3%参数的情况下，将大模型输出长度最多缩短39.84%，从而在保持效用的同时大幅降低推理成本。 |
| [^23] | [The Eloquence submission for Task 2 of the Interspeech 2026 MLC-SLM challenge](https://arxiv.org/abs/2609.11724) | Eloquence 团队针对跨 21 种语言的多语言语音多项选择问答任务探索了三种方法，其中基于多模态上下文学习的冻结 Voxtral-24B 模型纠正标签偏差后取得 0.81 的最佳宏平均准确率，三种方法均大幅超越官方基线。 |
| [^24] | [Why Does Post-Training Quantization Work?](https://arxiv.org/abs/2609.11716) | 该论文揭示了训练后量化对预训练大语言模型有效的机制——层新引入的误差与从输入继承的误差相互抵消等两种机制，使量化误差不会随深度累积，从而保持了模型性能。 |
| [^25] | [Negative Self-Distillation: Learning to Reason by Avoiding Flaws](https://arxiv.org/abs/2609.11699) | 该论文提出负自我蒸馏（NSD）新框架，通过让模型远离自身生成的有缺陷推理、而非模仿特权信息下的“自信”解答，克服了传统在线策略自蒸馏（OPSD）抑制不确定性表达和探索性自我纠错行为、从而损害复杂推理能力的缺陷。 |
| [^26] | [Structured Transforms for Low-Overhead Quantization of Language Models](https://arxiv.org/abs/2609.11687) | 该论文改进了基于Kashin分解的大语言模型权重量化方法，用符号随机化DCT替代稠密随机正交矩阵将每次迭代成本从O(N²)降至O(N log N)，并通过保证四峰分布和闭式聚类中心初始化的贪心交替算法消除了多重启k-means瓶颈，实现了低开销的2比特量化。 |
| [^27] | [A Training-Free, Alignment-Free Approach to Corporate Intelligence: Application to SEC Filings](https://arxiv.org/abs/2609.11620) | 该论文提出一种基于确定性稀疏种子向量的无需训练、无需对齐的企业情报分析框架，可在普通CPU上实现对SEC文件的亚秒级比较、发行人指纹识别及主题句提取。 |
| [^28] | [Complex-Text Robustness Evaluation and Failure Diagnosis for Low-Resource Multilingual Text-to-Speech](https://arxiv.org/abs/2609.11545) | 本文提出一个针对低资源多语言TTS的复杂文本鲁棒性诊断框架，从内容一致性、语言一致性和生成稳定性三个维度，评估系统在数字、日期、命名实体、语码转换等复杂文本输入下的失效情况，覆盖泰语、越南语、斯瓦希里语和印尼语四种语言。 |
| [^29] | [Structural priors for data-efficient language learning](https://arxiv.org/abs/2609.11505) | 该研究发现先在音乐、概率文法和元胞自动机等符号数据上训练模型，可为后续语言建模带来更低损失和更小的权重偏移，但这种损失优势并不能稳定转化为更好的下游语言能力。 |
| [^30] | [ReGround: Grounding Reviewer Comments in Multimodal Evidence](https://arxiv.org/abs/2609.11460) | 提出大规模审稿意见定位数据集ReGround，将3,656篇论文中的10,267条审稿意见链接到16,274条证据，并发现证据类型推断是主要瓶颈、多模态证据能提供纯文本检索遗漏的补充信号。 |
| [^31] | [Cross-Lingual Clinical Annotation Projection as Constrained Text Generation: A Six-Language Study](https://arxiv.org/abs/2609.11450) | 该研究提出将跨语言临床标注投影建模为受限文本生成任务，通过将实体标签直接插入不可修改的目标语言文本并进行确定性验证，在六种语言上实现了最强且最一致的临床标注迁移性能。 |
| [^32] | [SWRouter: Similarity-Contractive Window Routing for Multi-Turn Large Language Model Conversations](https://arxiv.org/abs/2609.11414) | 提出SWRouter，通过基于相似性的上下文分割机制与双指标评估框架，解决多轮对话中大语言模型路由面临的上下文信息丢失混淆及评估指标耦合两大难题。 |
| [^33] | [TransClean: A Benchmark for Detecting and Extracting Clean Translations from Large Language Model Outputs](https://arxiv.org/abs/2609.11399) | 该论文提出了TransClean基准数据集，通过分析79万余个大语言模型翻译输出识别出12种翻译噪声模式，并评估了从带噪声的翻译输出中提取干净翻译的两种方法。 |
| [^34] | [VikingRAG: Accurate and Token-efficient Retrieval-augmented Generation over Structured Documents](https://arxiv.org/abs/2609.11390) | VikingRAG通过将代理式多轮检索轨迹物化为可复用的经验边，并引入自适应升级策略在证据充分时采用单轮经验增强检索，在保持高准确率的同时大幅降低了结构化文档检索增强生成的令牌成本。 |
| [^35] | [SEAR: Segment-Evidence-Aware Routing for Weak-to-Strong Multilingual Speech MCQ](https://arxiv.org/abs/2609.11355) | 该工作提出SEAR系统，通过片段证据感知的多选题数据合成流水线与“弱到强”两阶段训练——先对可仅凭文本回答的题目做监督微调、再对依赖音频的题目做GSPO强化学习——来适配Qwen3-Omni，从而提升多语言语音多选题能力。 |
| [^36] | [On the Impact of Anonymization on the Performance of Large Language Models](https://arxiv.org/abs/2609.11335) | 本文系统评估了五个大语言模型在十一个基准上处理原始与匿名化输入的性能差异，发现匿名化总体上会降低性能，且影响因模型能力与任务类型而异——能力最强的模型性能下降最大，TruthfulQA性能反而提升，而检索类任务则遭遇灾难性下降。 |
| [^37] | [E-CONAN (Entailment, CONtradition And Neutral) Benchmarks: Arabic Textual Entailment and Natural Inference Datasets](https://arxiv.org/abs/2609.11334) | 本文提出了面向阿拉伯语文本蕴含与自然推理的E-CONAN基准数据集，其句子对来源于自动翻译、人工验证翻译、手工编写及含谣言的新闻标题等多种渠道，并利用该基准对9个最先进的多语言预训练模型进行了零样本分类评估。 |
| [^38] | [The Semantic Elevation Operator and the Closure of the Undecidable Class under Preservation](https://arxiv.org/abs/2609.11326) | 该论文提出语义提升算子 ΛΦ，将程序静态语义性质问题转化为自修改后的保持性问题，并基于克林递归定理证明不可验证性质类在该算子下封闭，且无界迭代将攀升至算术层级的 Π₂-完备性。 |
| [^39] | [MultiHuSE: A Multimodal Dataset for Humour Styles and Emotions](https://arxiv.org/abs/2609.11322) | 本文推出了首个涵盖四种心理学幽默风格的多模态数据集MultiHuSE，通过50名演员对相同文本的多样化表演捕捉幽默表达的多变性，并证明多模态融合在幽默风格分类上优于单模态方法。 |
| [^40] | [Automatic Lyric Transcription for Greek Songs: Scaling and Task Composition Effects in Whisper Adaptation](https://arxiv.org/abs/2609.11302) | 本文首次系统研究了Whisper在希腊语自动歌词转录任务上的适配，发现模型规模扩展可持续提升性能、多任务学习对小模型有正则化作用，且两阶段语音到歌唱适配使Whisper Large-v3的词错误率降至27.2%，同时构建了首个希腊语歌唱段级对齐数据集。 |
| [^41] | [Xiaomi-CocktailASR-1 Technical Report](https://arxiv.org/abs/2609.11274) | 提出了基于LLM的端到端目标说话人语音识别架构Xiaomi-CocktailASR-1，以参考语音作为声纹提示，无需语音分离即可直接转录目标说话人的语音，在保持单说话人场景竞争力的同时具备目标说话人缺席时的拒绝识别能力。 |
| [^42] | [INDRA: A New AI Tool for Exploring Tobacco, Fossil Fuel, and Chemical Industry Archives](https://arxiv.org/abs/2609.11261) | INDRA是一个AI研究平台，它将档案史学的规范嵌入系统级协议，并整合了多个此前孤立的历史行业档案库，使大语言模型能够可靠地探索数亿页烟草、化石燃料和化工行业的前机密文件，同时避免幻觉问题。 |
| [^43] | [MUtE: A Dual Framework for Concept Erasure and Counterfactual Interventions](https://arxiv.org/abs/2609.11253) | 本文提出基于概念擦除理论最优边界推导的对偶框架MUtE，其擦除函数可自然诱导确定性反事实映射，并通过在反事实轨迹上施加平移偏置，实现了概念擦除与反事实生成之间的无缝切换，有效提升下游算法公平性。 |
| [^44] | [The Illusion of Balanced Multimodal Sentiment Analysis: Beyond the Limits of Optimization-Based Methods](https://arxiv.org/abs/2609.11247) | 该论文通过统一评估框架和理论诊断揭示了基于优化的多模态平衡方法的根本缺陷——损失不等于效用、梯度不等于重要性，实验证明这些方法无法可靠超越简单基线，并提出基于保留数据性能的模态效用估计新研究议程。 |
| [^45] | [Assessing the Reusability of Public Speech Resources for Low-Resource Languages: A Central Kurdish Case Study](https://arxiv.org/abs/2609.11246) | 本文通过对公开的中库尔德语语音资源（三个语音、35小时录音）与其论文描述的一致性审查，发现数据中存在设备记录错误、测试数据未标注、代码缺陷等问题，并指出仅由三人朗读文本构建的语音资源难以覆盖库尔德语的地区与书面多样性，提醒低资源语言公开数据集复用需谨慎验证。 |
| [^46] | [OmniHallu: Unified Hallucination Detection for Cross-Modal Comprehension and Generation in Multimodal Large Language Models](https://arxiv.org/abs/2609.11244) | 本文提出统一幻觉检测框架OmniHallu，构建了覆盖图像、视频、音频三种模态下六种理解与生成任务的万级样本基准OmniHallu-Bench，并采用多智能体架构将输出分解为原子声明进行模态专家验证与结构化推理，实现跨模态幻觉的统一检测。 |
| [^47] | [A Voice-Interactive Multi-Agent System for Smart Operating Rooms: Architecture Design and Key Technologies](https://arxiv.org/abs/2609.11231) | 本文提出基于大语言模型的智能手术室语音交互多智能体系统SurgicalRoomAgent，通过KV Cache前缀预热、流式JSON解析和渐进式技能提示披露三项关键技术大幅降低推理延迟，实现自然语言理解、设备控制、术中记录与手术报告生成。 |
| [^48] | [REVA: Reusable Evidence View Aggregation for Context-Efficient RAG Serving](https://arxiv.org/abs/2609.11209) | REVA通过挖掘目标生成器的历史注意力轨迹，将历史查询-文档-模型交互聚合为可复用的证据视图分数库，从而在不依赖辅助模型和在线压缩的情况下实现上下文高效且低开销的RAG服务。 |
| [^49] | [Automated Identification of Competing Narratives in Political Discourse on Social Media](https://arxiv.org/abs/2609.11202) | 本文提出了一个无监督的多阶段自然语言处理框架，通过整合主题建模、事件检测和事件链接技术，自动识别德国政治家社交媒体推文中围绕热门政治话题的竞争性叙事及其不同视角。 |
| [^50] | [(Whose defaults?) Is artificial intelligence reorienting archaeological methods?](https://arxiv.org/abs/2609.11198) | 该研究分析了约11.9万篇考古学摘要并运用贝叶斯狄利克雷-多项式模型，发现大语言模型兴起后（2023年后）考古学计算方法仅有微小转变，且方法多样性不降反升，表明AI并未窄化学科的研究方法范围。 |
| [^51] | [FlexComp: One Model for Every Ratio in Context Compression](https://arxiv.org/abs/2609.11192) | FlexComp通过Matryoshka式训练使单一模型成为任意压缩率压缩器，并结合逐输入的预算选择机制（基于置信度的级联路由或轻量级K预测器），摆脱了传统方法中每个压缩率需单独训练模型且压缩率统一固定的限制。 |
| [^52] | [LILA: Calibration-Free Structured Pruning of Large Language Models via Latent Spectral Geometry](https://arxiv.org/abs/2609.11163) | LILA提出了一种免校准的结构化剪枝方法，通过比较完整与神经元消融后FFN权重矩阵的奇异值分布的KS距离来评分神经元重要性，无需训练、校准数据或辅助网络即可在多个稀疏度水平上超越现有依赖校准的剪枝方法。 |
| [^53] | [A Fragility Spectrum for Recursive Language-Model Training](https://arxiv.org/abs/2609.11149) | 该研究让13个公开模型在固定递归污染协议下共享语料库繁衍五代，发现不同模型对坍塌的脆弱性存在约五倍差异，且该脆弱性排序在不同数据组成和随机种子下高度稳定，表明易坍塌性是模型本身的固有属性。 |
| [^54] | [The Oligarch Barely Steers Model Collapse in Multi-Model Ecosystems](https://arxiv.org/abs/2609.11146) | 在多模型递归训练的生态系统中，即使将寡头模型的市场份额推高至90%，既不会加速模型崩溃，也不会使其他模型被拖向寡头的输出分布——模型崩溃的动态对市场份额集中度表现出不变性。 |
| [^55] | [Same Day, Same Story; One Day Ahead, a Different Signal: The Dual Validity of Financial Sentiment](https://arxiv.org/abs/2609.11144) | 本文基于2002-2025年证券集体诉讼语料库，将70,500条X消息与异常股票收益相关联，通过统一流程测试五种情感分析工具，发现金融情感工具的人工标注一致性（构念效度）与其市场预测能力（预测效度）之间的关系并非恒定，而是取决于抽样惯例和分数表示方式。 |
| [^56] | [Can LLMs Normalize Databases? A Benchmark and Multi-Agent Framework for Schema Normalization](https://arxiv.org/abs/2609.11141) | 该论文提出了包含3,275个样本的数据库规范化基准DNBENCH，系统揭示了LLM在函数依赖推理、模式分解和表间约束重建中的常见失败，并进一步提出多智能体框架MARS以提升模式规范化的可靠性。 |
| [^57] | [Rubric-Aligned Disentangled Evaluation of Human Simultaneous Interpreting](https://arxiv.org/abs/2609.11131) | 本文构建了首个带有多维量规专业评分的同传片段标注语料库，并提出基于LoRA适配COMET-KIWI编码器的双回归头模型，首次实现了与评估量规对齐的意义传递与表达质量解耦的自动同传评估。 |
| [^58] | [From Repetition to Recognition: Inductive Discovery of Disinformation Narratives](https://arxiv.org/abs/2609.11128) | 本文提出了包含恢复、挖掘和发现三个层次的无监督叙事标签生成评估框架，突破了传统封闭世界评估的局限，并发现基于聚类与基于图社区的流水线在虚假信息叙事发现中互为补充，但聚类方法可能严重压缩少数主题而图方法则保持均衡。 |
| [^59] | [KuaiRP Series Role-playing Models Technical Report](https://arxiv.org/abs/2609.11127) | KuaiRP系列角色扮演模型通过标准化角色模板、基于用户行为模拟的SFT数据流水线、规则复合奖励的强化学习以及多阶段训练流程，在注入深度领域知识的同时有效克服灾难性遗忘，实现了小参数规模下高质量、稳定且高效的角色扮演。 |
| [^60] | [Overview of the NLPCC 2026 Shared Task 11: Agent-Based Experiment Reproduction from Scientific Papers](https://arxiv.org/abs/2609.11117) | 该论文提出 AgentActionBench，一个面向过程的基准，通过基于 MCP 的动作记录器和论文专属评分标准，评估大语言模型智能体在机器学习与 AI4Science 领域实验复现过程中的完整行为表现。 |
| [^61] | [ProMediConv: Benchmarking Proactive Conversational Agents in Legal Dispute Mediation](https://arxiv.org/abs/2609.11101) | 该论文提出ProMediConv基准框架，将法律纠纷调解建模为主动式、多阶段、感知当事人的对话过程，基于972个真实案例构建了高保真话语级标注数据集，并提出了捕捉对话中当事人行为模式变化的细粒度MAD评估指标。 |
| [^62] | [Beyond Solver Verdicts: Generative Reward Models for Autoformalization](https://arxiv.org/abs/2609.11085) | 本文发现了自动形式化中的“判定保持的不忠实性”（VPU）失败模式，从理论上证明仅依赖求解器判定的验证方法无法有效检测此类错误，并提出生成式验证方法（GenV），将Z3等价性预言机蒸馏为无参照的连续等价性评分以实现可靠的验证。 |
| [^63] | [When Noise Fabricates Bias: The Fragility of LLM-as-a-Judge Bias Measurement under Noisy Text](https://arxiv.org/abs/2609.11067) | 该论文发现文本表面噪声（如拼写错误）会严重扭曲LLM评判者的社会偏见测量，且这种扭曲极不对称——中性文本被误判为带有偏见的概率是被反向误判的120倍，揭示了基于LLM的偏见测量的脆弱性。 |
| [^64] | [The information geometry of large language models is shared, learned, and controllable](https://arxiv.org/abs/2609.11063) | 该论文发现大语言模型下一个词元概率的Fisher-Rao几何在不同架构间是共享的、随训练可学习的，并与人类词汇选择高度一致且可被控制，为理解和修改模型行为提供了统一的结构化框架。 |
| [^65] | [Rebalancing Token Importance in Language Models with TF-IDF Weighted Cross-Entropy Loss](https://arxiv.org/abs/2609.11029) | 该论文提出用TF-IDF加权交叉熵损失替代统一词元权重，在五个大语言模型上显著减少记忆化现象（LoRA微调下平均减少14%，TinyLLaMA全权重微调下减少58%），同时保持模型性能且计算开销不足3%。 |
| [^66] | [New Evidence, Same Choice: Testing Physical Experiment Selection in Vision Language Models](https://arxiv.org/abs/2609.11022) | 该论文提出了一个受控评估框架，测试视觉语言模型在物理推理中能否判断何时应直接作答、何时需要额外测量以及选择哪个实验，弥补了现有基准只评估最终答案而忽视决策能力的不足。 |
| [^67] | [K/V-Cache Interventions Dissociate Representation Alignment from Persona Expression in Decoder-Only Language Models](https://arxiv.org/abs/2609.11020) | 该研究通过对Llama-3.1-8B的K/V缓存干预实验发现，表示层面的对齐与人格的行为表达可以相互解耦，其中中层（第9-20层）替换是唯一能同时实现目标人格表达并保持词汇多样性的干预策略。 |
| [^68] | [Rethinking Verbalized Confidence for LLM-as-a-Judge: A Compatibility Shift on Post-2025 Proprietary Models](https://arxiv.org/abs/2609.10996) | 该研究揭示了“兼容性转变”现象——在2025年后的专有大模型上，言语化置信度已取代对数概率成为LLM评判中更优的软评分信号，并提出过度自信咨询与自我辩论两个新要素来改善校准和主观性鲁棒性。 |
| [^69] | [Distribution-aware Language Neuron Identification in Multilingual Large Language Models](https://arxiv.org/abs/2609.10993) | 该论文提出了一种分布感知的语言神经元识别方法，通过利用全激活范围（包括负值）上各语言激活分布之间的成对重叠系数，更准确地识别多语言大语言模型中的语言特异性神经元。 |
| [^70] | [Robust Multimodal Sentiment Analysis with Incomplete Modalities via Semantic-aware Completeness based Reconstruction](https://arxiv.org/abs/2609.10950) | 该论文提出了一种语义感知的完整性估计方法与稳定的多任务训练策略，通过重建模态缺失的语义信息，显著提升了模态不完整场景下多模态情感分析的鲁棒性和预测精度。 |
| [^71] | [Empirical Evaluation of Membership Inference Attacks on NLP Text Classifiers: A Baseline Study on SST-2](https://arxiv.org/abs/2609.10935) | 本文在GLUE SST-2数据集上对NLP文本分类器（TF-IDF+逻辑回归与微调DistilBERT）进行了成员推断攻击的对照基准评估，发现两类模型均存在成员信息泄露，且减少微调轮数可在几乎不损失效用的前提下缓解泄露风险。 |
| [^72] | [Using Semantic Uncertainty to Estimate Transition Relevance in Turn-taking](https://arxiv.org/abs/2609.10934) | 本文提出利用大语言模型导出的语义不确定性——通过对正在进行的话轮采样可能的后续内容并分析语义离散度的变化——来预测话轮内部的转换关联位置（TRP），从而帮助口语对话系统在更恰当的时机进行话轮接管。 |
| [^73] | [Structurally Speaking: Motif-Oriented Graph Captioning through Bidirectional Graph-Text Translation](https://arxiv.org/abs/2609.10923) | 本文提出Structurally Speaking，一种轻量级结构化提示协议，通过将图描述生成建模为双向图-文本翻译任务，引导大语言模型生成既简洁又能保留图拓扑结构以支持图恢复的面向模体的图描述。 |
| [^74] | [Auto-RecSys: Harnessing Autonomous Research Agents for Industry-Scale Recommender System](https://arxiv.org/abs/2609.10922) | 提出了Auto-RecSys系统，通过分布式异步执行和集中式跨服务器记忆等框架设计，使自主研究智能体能够在工业规模推荐模型上进行长期、可恢复的并行自动化实验。 |
| [^75] | [SearchAtlas: Analyzing Agentic Search Strategies via Evidential Query Graphs](https://arxiv.org/abs/2609.10901) | SearchAtlas框架将LLM搜索智能体的搜索轨迹转换为证据查询图（自动解析达到86.0%的边F1分数），从而揭示了不同智能体在搜索规模、证据聚合及答案支持方面的系统性差异与缺陷。 |
| [^76] | [LLM-Anchored Paralinguistic Enrichment for Alzheimer's Disease Detection](https://arxiv.org/abs/2609.10896) | 提出LAPE方法，通过韵律事件文本化、词汇-韵律单元化分块和文本锚定的副语言融合三项创新，将停顿和词语拖长等副语言线索与大语言模型的语言表示相融合，以提升基于语音的阿尔茨海默病自动检测效果。 |
| [^77] | [Does Linguistic Structure Enrichment Enhance Coherence Assessment? Not With Current Architectures](https://arxiv.org/abs/2609.10893) | 研究发现，由于附加信息与当前语言模型架构存在结构和句法上的不兼容性，用句法和修辞信息增强文本并不能提升连贯性评估效果，而文本连贯性可作为检测虚假信息的代理指标。 |
| [^78] | [Story Imprinting: AI Assistants Absorb Traits from Human Characters They Resemble](https://arxiv.org/abs/2609.10883) | 研究发现，在合成故事上微调会使AI助手“烙印”上与其相似的人类角色的条件性行为和隐含偏好，即使这些内容在训练数据中占比不到2%或从未被明确表达。 |
| [^79] | [Detectable Only Where It Is Confounded: What Verified Duplication Counts Say About Membership Evidence in Language Models](https://arxiv.org/abs/2609.10830) | 利用公开可验证的重复计数，研究表明在真实文本的重复水平下，语言模型的预测成本与其训练数据暴露程度之间至多只有微弱的关联（秩相关约-0.08），因此预测成本低并不能可靠地证明某个句子存在于训练数据中。 |
| [^80] | [Studying Without a Syllabus: Task-Agnostic Environment Preprocessing](https://arxiv.org/abs/2609.10824) | 该论文形式化了“任务无关的环境预处理”问题，证明LLM智能体可以在不知道下游任务分布的情况下，在测试前自主探索陌生环境并构建可复用工件来辅助冻结的求解器。 |
| [^81] | [BodyCam-VQA: Enhanced Body-Worn Camera Video Captioning via Multimodal Reasoning and Probe Question Generation](https://arxiv.org/abs/2609.10815) | 该论文提出了一个专为高风险执法设计的自适应视觉问答（VQA）框架，通过多模态推理和探针问题生成来增强警察随身摄像机视频的描述生成，克服了混乱场景、低视觉质量和高噪音音频等挑战，避免遗漏关键取证细节。 |
| [^82] | [Larger Context Window, Fewer Overcorrections: Optimizing Prompts and Batching for Minimal-Edit Grammatical Error Correction](https://arxiv.org/abs/2609.10810) | 本文提出一种无需微调的提示方法，通过基于语法错误分类体系的约束指令以及将多句批量输入作为抑制过度纠正的正则化手段，使零/少样本大语言模型在最小编辑语法纠错任务上接近微调模型的性能。 |
| [^83] | [Analyzing Traditional and Neural Approaches to Multilingual Readability Assessment](https://arxiv.org/abs/2609.10792) | 该研究结合SHAP与TCAV方法，对比分析传统特征模型与Transformer模型在阿拉伯语、英语、法语、印地语和俄语可读性评估中内化的语言特征，发现Transformer能恢复表面长度、句法和词汇多样性信号并反映CEFR序数结构，但特征一致性因模型系列、语言和网络层而异。 |
| [^84] | [Multilingual in Name Only? Cultural and Linguistic Weaknesses of LLMs in Urdu](https://arxiv.org/abs/2609.10758) | 本文通过构建包含93个AI生成乌尔都语故事的语料库并按九类语言、语义和文化错误进行人工标注，揭示了多语言大型语言模型在低资源语言故事生成中存在基本语法错误、缺乏连贯性和普遍文化浅薄等严重缺陷，且少样本提示无法有效解决文化错误问题。 |
| [^85] | [Think Before You Link: Rarity, Reasoning, and Retrieval in Multilingual Entity Linking](https://arxiv.org/abs/2609.10745) | 该论文提出用知识图谱结构指标更全面地衡量实体稀有性，并引入一个无需训练的框架，让具备推理能力的视觉-语言模型通过在维基百科上迭代搜索与推理来动态收集证据，从而提升多模态实体链接在稀有实体上的性能。 |
| [^86] | [The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes](https://arxiv.org/abs/2609.10739) | 论文揭示了真值探测器的“完美混叠”失效机制——当真实汇报与任务既定行为在顺从情境中重合时探测器无法区分二者（二者AUROC恒互补求和为一），并提出在顺从与对立情境的混合数据上拟合的方法，使探测器即使在模型系统性说谎时也能以完美的1.0 AUROC识别真值。 |
| [^87] | [CMNIE: An Information Extraction Benchmark for Chinese Military News](https://arxiv.org/abs/2609.10722) | 本文提出了CMNIE——一个面向中文军事新闻的联合信息抽取基准数据集，在统一领域模式下对事件触发词、事件论元、命名实体和实体关系进行联合人工标注，并系统评估了有监督模型与基于大语言模型的抽取方法的性能。 |
| [^88] | [NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction](https://arxiv.org/abs/2609.10715) | 该论文提出通过“下一概念预测”在由乘积量化概念词表构成的潜空间中预测跨多个词元的离散概念，并与词元级预测联合端到端训练，构建了迄今最大规模（89 亿参数、5.73 万亿词元）的潜空间语言模型。 |
| [^89] | [Data-Efficient Language Modeling: From Frontier Advancement to Principle-Guided Model Improvement](https://arxiv.org/abs/2609.10702) | 该研究通过三阶段自主研究计划，在BabyLM 2026 Strict-Small受限数据设置下发现精确重复与对齐重述产生不同的上下文利用模式，并提出围绕预训练所需上下文依赖来组织经验的数据高效学习原则，实现了从构建前沿模型到原则指导的模型改进。 |
| [^90] | [More than half of recent astronomy papers are written with language-model assistance](https://arxiv.org/abs/2609.10664) | 该研究通过统计20余万篇天文学论文中语言模型特征词汇的出现频率，利用层次贝叶斯混合模型估计出2025年约有54%的天文学论文是在语言模型辅助下撰写的。 |
| [^91] | [Artificial Intelligence Algorithms for the Detection of Pathologies Related to Lung Cancer through Image Analysis using Convolutional Neural Networks and Data Augmentation: a systematic mapping of the literature](https://arxiv.org/abs/2609.10652) | 本文通过系统文献映射方法综述了2015年以来96篇关于AI和深度学习在肺癌影像检测中应用的研究，重点探讨了结合迁移学习和数据增强的卷积神经网络作为有前景的自动化检测技术。 |
| [^92] | [SalamandraTA at WMT 2026 Terminology Shared Task: Hard Examples Are Better Teachers](https://arxiv.org/abs/2609.09999) | 该论文提出在术语感知翻译的微调中只保留模型自身译文与术语表相矛盾的“难例”，仅此筛选即可在固定数据量下将术语准确率从 78.7% 提升至 89.9%，并据此构建了 SalamandraTA-7b-instruct v3.0，作为 BSC 参加 WMT26 术语共享任务赛道 1 的提交系统。 |
| [^93] | [CARRE: Counterfactual Action Retrieval and Reason Evaluation for Explainable Churn Prescription](https://arxiv.org/abs/2609.09766) | CARRE是一个结合检索增强候选生成、成本感知反事实评分与大语言模型推理的三阶段框架，不仅识别高流失风险客户，还能推荐具体的挽留行动并给出可解释的推荐理由，在电信数据集上相比SHAP基线实现了近80%更高的风险降低。 |
| [^94] | [Edu-QuRating: Multi-Dimensional Educational Data Curation with Distilled Pairwise Judgements](https://arxiv.org/abs/2609.09425) | Edu-QuRating提出了一种多维度教育数据筛选流水线，通过定义教育专用评分标准、利用LLM评判器标注文档对并将成对偏好蒸馏为可复用的评分模型，从而突破传统单一标量式教育价值评估的局限，从准确性、吸引力、结构性和受众适用性等多个维度对文本进行精细化评分。 |
| [^95] | [Limitations of Automated Simulatability: LLM Simulators Can Bypass Explanations](https://arxiv.org/abs/2609.08585) | 本文揭示了自动可模拟性评估的两大局限：当类名有意义时，LLM模拟器可直接解题获得高分而不依赖解释；而类匿名化又可能奖励泄露隐藏标签映射的解释，表明模拟器预测主要依赖任务信息而非解释本身。 |
| [^96] | [Beyond Single-Negative Preference: Multi-Negative DPO for LLM-Centric Historical Entity Linking](https://arxiv.org/abs/2609.07379) | 提出多负例直接偏好优化方法（MDPO），通过将正确实体与完整候选集进行比较来改进基于大语言模型的历史实体链接，在多语言历史报纸文本上超越监督微调和单负例DPO，尤其在NIL提及、语义歧义和OCR噪声场景下收益显著。 |
| [^97] | [Reason Through the Latent! Making Latent Visual Reasoning Necessary](https://arxiv.org/abs/2609.06746) | 提出因果视觉循环推理（CVRR）框架，通过在解码前移除视觉状态和多模态KV缓存，迫使循环隐藏状态成为唯一的图像条件信息通路，从而确保潜空间视觉推理真正被模型依赖。 |
| [^98] | [A Group-Based Resource Allocation Model for the Fractional Knapsack Problem](https://arxiv.org/abs/2609.06470) | 该论文提出了一种两阶段分组资源分配模型，通过将属性相近的物品分组来缓解Dantzig贪婪规则中因微小扰动导致的分配不稳定问题，并给出了与精确最优解相比的紧损失上界。 |
| [^99] | [Cache-Aware Joint Router Adaptation for Memory-Efficient MoE Inference](https://arxiv.org/abs/2609.04895) | 提出一种缓存感知的后训练框架，通过联合自适应MoE主干与轻量级时间-空间缓存路由器，在不改变原生Top-K专家选择规则的前提下提升缓存命中率、减少专家权重传输，实现内存高效的MoE推理。 |
| [^100] | [Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation](https://arxiv.org/abs/2609.04298) | 本文提出了 Harbor Adapters 统一评估基础设施，将 80 多个智能体基准测试移植为可评估任意智能体的形式，并据此对 8 个模型进行大规模评估，同时推出经 AI 与人工双重审核筛选的包含 82 个高质量难题的精选元数据集 Harbor-Index。 |
| [^101] | [OUTLETS: Output-Length Prediction from Speculative Decoding Backbones](https://arxiv.org/abs/2609.01068) | 该论文发现投机解码框架中草稿解码器的潜在表示蕴含着可预测生成长度的信号，并提出OUTLETS方法，将投机解码主干重新用作轨迹感知的输出长度预测器，从而在几乎不增加额外开销的情况下改进大语言模型服务的资源供给与集群调度。 |
| [^102] | [Quit While You're Ahead: Quit for Efficient Candidate Generation in Machine Translation Reranking](https://arxiv.org/abs/2609.00588) | 提出Quit方法，通过不确定性量化的早停策略对机器翻译的整个候选生成—重排序流程进行增量式生成与重排序，在最高候选质量稳定时提前终止，从而在保持翻译质量的同时显著降低推理延迟。 |
| [^103] | [Learning What to Retain: Gated-Memory Routing for Efficient Collaboration in Multi-Agent LLM Systems](https://arxiv.org/abs/2609.00237) | 提出门控记忆路由方法，通过可学习的记忆写入门和检索门维护紧凑的执行记忆，使多智能体LLM系统的编排决策能依据有用的中间进展而非完整历史，在提升准确性的同时降低成本。 |
| [^104] | [Toward a Cross-Lingual Romanization Ecosystem for Sinitic Languages: A Paired Mandarin-Cantonese Case Study](https://arxiv.org/abs/2608.29170) | 本文提出汉语罗马化生态系统框架，基于语音对应、历史音韵对应、一音位一符号和基本拉丁字母使用四项设计原则，开发出普通话与粤语等汉语族语言的配套罗马化方案及开源数字基础设施。 |
| [^105] | [DelistBench: Evaluating Search-Enabled LLMs for Auditable Corporate-Event Database Completion](https://arxiv.org/abs/2608.22770) | 该论文提出了DelistBench基准和Search-to-Record任务，证明网络搜索能显著提升LLM在公司事件数据库补全中的准确率，且经济型系统能以低成本达到接近最优性能。 |
| [^106] | [Whitewashing Hate, Smearing Harmless Content: Annotator-Style Rebuttal Attacks on LLM-Based Moderation](https://arxiv.org/abs/2608.22230) | 本研究揭示了标注者风格的反驳攻击能显著破坏LLM仇恨言论审核的准确性，且洗白与污蔑两种操纵方向存在模型特定的不对称效应。 |
| [^107] | [SAC-Copula: Quality-Preserving Watermarking for Diffusion Language Models via Smooth Correlated Gumbel Fields](https://arxiv.org/abs/2608.20839) | SAC-Copula通过引入基于高斯copula的平滑局部相关Gumbel扰动场，解决了扩散语言模型水印中扰动与解码动态不匹配的问题，实现了生成质量与可检测性的更优平衡。 |
| [^108] | [Aslema at NADI 2026: Augmentation through Fewshot for SLU](https://arxiv.org/abs/2608.18689) | 本文提出Aslema系统，通过微调优于零样本，并利用大型语言模型生成文化相关的合成数据增强，在NADI 2026任务中在槽位填充上取得第一名。 |
| [^109] | [Efficient Adaptation of LLMs for Hate Speech Detection in Low-Resource Languages: A Comparative Study on Roman Urdu](https://arxiv.org/abs/2608.18142) | 本研究通过LoRA参数高效微调方法，系统比较了多种大型语言模型在罗马乌尔都语低资源环境下的仇恨言论检测性能，展示了PEFT在零样本推理中的优势。 |
| [^110] | [Unadapted Multilingual ASR on a Garrusi Kurdish Evaluation Set: A Common-Reference Staged Normalization Analysis](https://arxiv.org/abs/2608.16379) | 本文通过共同参考分阶段归一化方法，揭示了未适配的多语言ASR在Garrusi库尔德语上因书写系统差异导致的严重高估错误率，并提供了更公平的评估基准。 |
| [^111] | [Self-Evolving Embodied Agents via Skill-Harness Evolution](https://arxiv.org/abs/2608.11350) | SHAPER提出了一种免训练的自我进化框架，通过冻结模型参数并进化外部技能和代码框架，使具身智能体适应新环境，无需额外数据或训练。 |
| [^112] | [VectraYX-Vision-1B: A Sub-2B Spanish/LATAM Cybersecurity Vision-Language Model with Structured Visual Reasoning and Native Tool Use](https://arxiv.org/abs/2608.08477) | v3版本将原生训练的Qwen2-VL视觉塔移植到同一冻结解码器上，使原本失败的8半字节地址字段精确率从0.00跃升至0.81，且token预算比2x2平铺更粗糙，证明分辨率并非关键变量。 |
| [^113] | [Causal Episodic Memory for Feedback-Driven Agent Repair](https://arxiv.org/abs/2608.05906) | MERIT是一种无需训练的智能体，通过维护经oracle验证的成功修正与失败方向的在线双极性情景记忆，在不更新参数的情况下提升后续Text-to-SQL修复任务的执行准确率（Spider从66.34%升至69.79%，BIRD从47.35%升至48.44%）。 |
| [^114] | [Progressive Agent Skill Generation via Reinforcement Learning](https://arxiv.org/abs/2608.01678) | 提出Skill-α，一种基于强化学习的渐进式智能体技能生成方法，通过以智能体在下游任务中的表现为奖励信号，学习统一策略从异构来源自动生成高质量技能。 |
| [^115] | [Predicting Startup Exit from Textual Descriptors - A Computational Linguistics Framework](https://arxiv.org/abs/2608.00045) | 仅凭文本描述中的语言特征（如形容词、术语和流行语等炒作标记）即可预测初创企业能否成功退出，无需依赖财务或人力资本数据，其中炒作标记的优化密度与更高的退出概率正相关。 |
| [^116] | [Sympathetic Framing: Evaluating AI Alignment across Sociodemographic Groups](https://arxiv.org/abs/2607.27232) | 该研究通过对3011名英国成年人的YouGov调查与七个大语言模型的对比实验，首次实证评估了LLMs在感知新闻标题情感框架（对冲突各方的同情）方面与人类的对齐程度，发现领先模型（如GPT-5.2相关性达0.789）在所有人口统计子群体中都与人类情感判断高度一致。 |
| [^117] | [A Factorial Study of Synthetic Data Generation for Low-Resource Machine Translation using Grammar Books](https://arxiv.org/abs/2607.22376) | 该论文提出利用大语言模型从语法书中提取语法规则、例句和词汇生成合成平行语料用于模型微调（而非推理时提示），在卡兰芒语、图阿茨钦语和曼丹语三种低资源语言上验证了其有效性，并通过96种配置的析因分析找出了驱动翻译性能提升的关键因素组合。 |
| [^118] | [DiaLLM: An Investigation into the Robustness-Generation Gap in English Dialect Adaptation](https://arxiv.org/abs/2607.07669) | 本文发现方言理解与生成能力在LLM中分离，并证明显式方言定向适应优于广泛对齐，但基准测试无法反映这一生成优势。 |
| [^119] | [LLM-Ideoplasticity: Measuring Ideological Plasticity in the Political Behavior of LLMs as a Context-Conditioned Distribution](https://arxiv.org/abs/2606.28335) | 大语言模型的政治意识形态不是固定点，而是随语境变化的条件分布——虽然对说服性框架、语言选择等因素高度敏感，但整体上占据的政治光谱范围仅为欧洲主要政党的约三分之一。 |
| [^120] | [Cognitive Digital Twins: Ethical Risks and Governance for AI Systems That Model the Mind](https://arxiv.org/abs/2606.23094) | 本文定义了认知数字孪生（CDTs）这一新型AI技术，指出其独特伦理风险，并提出了涵盖权威、自主性、访问与控制、问责制和可用性五个维度的5A治理框架。 |
| [^121] | [Inverse Turing Bench: Evaluating Language Models as Judges of Human vs. AI Dialogue](https://arxiv.org/abs/2606.21844) | 提出逆向图灵基准，用于评估大语言模型在多轮对话中区分人类与人工智能的能力，最强模型GPTZero达到89.41%的准确率，并揭示了统计检测方法存在语义盲点、语义方法易受角色扮演提示影响的局限。 |
| [^122] | [Characterizing Narrative Content in Web-scale LLM Pretraining Data](https://arxiv.org/abs/2606.19468) | 该论文首次对3万亿词元的开放预训练语料库Dolma中的叙事特征进行细粒度研究，通过构建涵盖能动性、场景和事件三大要素的11维叙事分析框架并训练NarraBERT模型，生成了新数据集NarraDolma，揭示出叙事结构可大规模测量，且叙事质量在预训练数据来源、主题和格式间分布不均。 |
| [^123] | [Strategic Type Spaces](https://arxiv.org/abs/2606.08297) | 本文为信息提供了策略基础，证明了不完全信息博弈中存在本质唯一的最小策略类型空间（STS），且其具有可被有限自动机刻画的递归结构。 |
| [^124] | [Activation-Based Active Learning for In-Context Learning: Challenges and Insights](https://arxiv.org/abs/2606.05134) | 本研究探索了基于MLP激活的深度主动学习方法来优化LLM上下文示例选择，但得出负面结论：激活信号（无论从大规模激活还是前四阶矩角度观察）与示例质量和任务性能的相关性很弱，Spearman相关系数最高仅0.33。 |
| [^125] | [MERIT: Matching Expertise via Rubric-Informed Training for Reviewer Assignment](https://arxiv.org/abs/2605.27865) | MERIT提出了一个两阶段框架，通过强化学习训练的审稿人评估器（以评分标准引导的LLM评判作为奖励）将准则级专长匹配转化为可规模化的适配性监督，再将其蒸馏到嵌入检索器中，实现高效且准确的大规模审稿人分配。 |
| [^126] | [Cross-lingual brain-language model alignment is robust but challenges hierarchical and computational accounts](https://arxiv.org/abs/2605.21049) | 本研究通过普通话、英语和法语的全脑编码模型发现，脑-语言模型对齐虽然稳健且在跨语言和跨模型层间高度稳定，但这种对齐主要源于稳定的词汇-语义对应关系，而非Transformer的层级化计算特性，从而挑战了将脑-模型对齐视为类脑计算证据的传统解释。 |
| [^127] | [Continuous Diffusion Scales Competitively with Discrete Diffusion for Language](https://arxiv.org/abs/2605.18530) | 该研究通过将 Plaid 与现代离散扩散语言模型架构对齐构建了 RePlaid，建立了首个可媲美离散 DLM 的连续扩散语言模型缩放定律，证明连续扩散是语言建模中高度竞争且可扩展的方案。 |
| [^128] | [A Recipe for Long-Context Reasoning in Large Language Models via On-Policy Optimization and Distillation](https://arxiv.org/abs/2605.12227) | 该论文提出了一种结合GRPO强化学习与在线策略蒸馏教师指导的训练配方，用更强教师的密集token级正则化替代标准参考策略，在过程级监督难以获取时实现大模型长上下文推理的有效对齐。 |
| [^129] | [Timing is Everything: Temporal Scaffolding of Semantic Surprise in Humor](https://arxiv.org/abs/2605.00143) | 该论文提出双重预测违背（DPV）框架，并通过分析828场专业中文脱口秀表演发现，以停顿和语义违背峰值为代表的时间特征对观众幽默欣赏度的预测作用远超语义失谐本身，证明“时机”在幽默中起着关键作用。 |
| [^130] | [Analyzing LLM Reasoning to Uncover Mental Health Stigma](https://arxiv.org/abs/2604.25053) | 该论文提出通过分析大语言模型的中间推理步骤，并借助临床专业知识对污名化语言进行分类与严重程度评级，从而揭示多项选择题评估方法无法捕捉的心理健康污名化偏见及其内在逻辑。 |
| [^131] | [LLMAR: A Tuning-Free Recommendation Framework for Sparse and Text-Rich Industrial Domains](https://arxiv.org/abs/2604.16379) | LLMAR提出了一种免调优的推荐框架，通过LLM推理驱动的标注将行为历史转化为结构化语义动机，并结合反思循环机制进行自我纠错，从而在数据稀疏、文本丰富的工业B2B场景中无需训练即可实现高效的推荐。 |
| [^132] | [Learning to Think Like a Cartoon Captionist: Incongruity-Resolution Supervision for Multimodal Humor Understanding](https://arxiv.org/abs/2604.15210) | 提出IRS框架，将多模态幽默理解分解为不协调性建模、消解建模和偏好对齐三个环节，通过结构化推理轨迹监督，使模型像专业漫画配文作者一样进行显式推理。 |
| [^133] | [Formalizing building-up constructions of self-dual codes through isotropic lines in Lean](https://arxiv.org/abs/2604.08485) | 本文在Lean证明助手中形式化了自对偶码的构建方法，证明了双坐标约化与Kim构建法的互逆性，并通过各向同性直线发展了q元推广，构造出多种最优自对偶码。 |
| [^134] | [MisEdu-RAG: A Misconception-Aware Dual-Hypergraph RAG for Novice Math Teachers](https://arxiv.org/abs/2604.04036) | 提出了面向新手数学教师的误解感知双超图RAG框架MisEdu-RAG，通过将教学知识组织为概念超图、将真实学生错误案例组织为实例超图，并采用两阶段检索，生成基于证据且更具可操作性的教学反馈。 |
| [^135] | [Emergent Risks in Generative Multi-Agent Systems](https://arxiv.org/abs/2603.27771) | 本文开创性地系统研究了生成式多智能体系统中的涌现风险，发现在共享资源竞争、顺序交接协作和集体决策聚合等场景中，有害的群体行为会频繁出现而非罕见特例。 |
| [^136] | [Alignment Reduces Expressed but Not Encoded Gender Bias: A Unified Framework and Study](https://arxiv.org/abs/2603.24125) | 该研究提出一个统一框架来联合分析大语言模型内在与外在的性别偏见，发现对齐虽能减少生成输出中表达出的性别偏见，但并未消除模型内部表征中编码的性别偏见。 |
| [^137] | [Perturbation: A simple and efficient adversarial tracer for representation learning in language models](https://arxiv.org/abs/2603.23821) | 该论文提出了一种简单高效的对抗性扰动方法来追踪语言模型的表示学习，通过在单个对抗样本上微调模型并测量扰动对其他样本的“感染”程度来揭示表示，该方法无需几何假设，且能避免在未训练模型中产生虚假表示。 |
| [^138] | [OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis](https://arxiv.org/abs/2603.20278) | OpenResearcher提出了一个完全开源、可复现的离线轨迹合成流水线，在1500万文档语料库上合成超过97K条长程深度研究轨迹，微调后的模型在BrowseComp-Plus上相比基础模型提升34个百分点。 |
| [^139] | [Evaluating LLM-Simulated Conversations in Modeling Inconsistent and Uncollaborative Behaviors in Human Social Interaction](https://arxiv.org/abs/2603.17094) | 本文提出CoCoEval框架，通过轮次级检测10类不一致与非协作行为并结合专业场景对话基准，评估LLM模拟对话能否以与人类对话相当的频率再现误解、打断等行为。 |
| [^140] | [Streaming Translation and Transcription Through Speech-to-Text Causal Alignment](https://arxiv.org/abs/2603.11578) | 提出了无策略端到端模型Hikari，通过解码器时间膨胀机制和监督微调策略，以较小的模型规模实现了低延迟且高质量的同声语音翻译与流式转录。 |
| [^141] | [Probing for Knowledge Attribution in Large Language Models](https://arxiv.org/abs/2602.22787) | 本文提出自监督数据生成流水线AttriWiki，证明仅用简单线性探针即可从大语言模型的隐藏表示中可靠地判断答案的知识来源是内部记忆还是外部上下文，从而区分忠实性幻觉与事实性幻觉以助力针对性缓解。 |
| [^142] | [What Language is This? Ask Your Tokenizer](https://arxiv.org/abs/2602.17655) | 提出了基于UnigramLM分词算法的语言识别方法UniLID，通过判断字符串在哪种语言的单字分布下最可能出现来识别语言，无需重新训练即可增量添加新语言，并能自然融入现有语言模型分词流程，在低资源语言和密切相关语言场景下显著优于现有方法。 |
| [^143] | [Evaluating Memory Structure in LLM Agents](https://arxiv.org/abs/2602.11243) | 本文提出StructMemEval基准测试，用于评估LLM智能体组织长期记忆结构（如交易账本、待办列表、树结构）的能力，弥补了现有基准仅测试简单事实保留和多跳召回而无法评估复杂记忆层次结构的不足。 |
| [^144] | [Towards Reliable Medical LLMs: Benchmarking and Enhancing Confidence Estimation of Large Language Models in Medical Consultation](https://arxiv.org/abs/2601.15645) | 该论文提出了首个评估真实医疗咨询中多轮交互下大语言模型置信度的基准，通过引入信息充分性梯度刻画置信度与正确性随证据积累的动态关系，并对27种代表性置信度估计方法进行了系统比较与增强。 |
| [^145] | [Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)](https://arxiv.org/abs/2601.15397) | 本文提出LOGIC方法，通过在Logit空间层面直接集成上下文偏置，为语音大语言模型提供了一种高效且鲁棒的解决方案，克服了传统提示方法的可扩展性瓶颈和生成式错误纠正的幻觉问题。 |
| [^146] | [DeepResearch Bench II: Diagnosing Deep Research Agents via Rubrics from Expert Reports](https://arxiv.org/abs/2601.08536) | 提出了Deep Research Bench II基准，通过从专家报告中提取的9,430个细粒度二元评估标准，从信息召回、分析和呈现三个维度对深度研究智能体的报告生成能力进行严格评估。 |
| [^147] | [Output Embedding Centering for Stable LLM Pretraining](https://arxiv.org/abs/2601.02031) | 该论文揭示了大语言模型预训练末期输出logit发散的根源在于输出嵌入的各向异性，并提出输出嵌入中心化（OEC）新方法，通过μ-centering或μ-loss两种实现方式有效抑制训练不稳定，其稳定性优于z-loss且与logit软上限方法相当。 |
| [^148] | [Narrative Consolidation: Formulating a New Task for Unifying Multi-Perspective Accounts](https://arxiv.org/abs/2512.18041) | 本文提出并形式化定义了一个新的自然语言处理任务“叙事整合”，旨在将多视角叙述性文档统一为时间连贯、内容完整且融合互补细节的文本，并构建了基于四福音书的基准数据集、评估范式及一系列参考系统。 |
| [^149] | [Do Vision-Language Models Understand Visual Persuasiveness? A Diagnosis via Visual Persuasive Factors](https://arxiv.org/abs/2511.17036) | 该论文通过实证分析发现视觉语言模型在视觉说服力判断上存在过度预测为“有说服力”的召回偏向，并提出受认知心理学启发的视觉说服因素（VPFs）分类体系来量化视觉线索，揭示了模型判断与人类判断之间的差距。 |
| [^150] | [Leveraging LLMs for Context-Aware Implicit Textual and Multimodal Hate Speech Detection](https://arxiv.org/abs/2510.15685) | 利用大语言模型为社交媒体帖子生成背景上下文，并通过多种融合方法将其融入SBERT分类器，可将隐性和多模态仇恨言论检测的F1分数在文本和多模态场景下分别最高提升3分和6分。 |
| [^151] | [CHRONOBERG: Capturing Language Evolution and Temporal Awareness in Foundation Models](https://arxiv.org/abs/2509.22360) | 该论文提出了CHRONOBERG——一个跨越250年、带有丰富时间标注的英语书籍文本语料库，通过历史校准的情感词典量化语言演变，从而提升基础模型对语言历时变化的时间感知能力。 |
| [^152] | [The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies](https://arxiv.org/abs/2509.18052) | 该论文提出PIMMUR六项方法学原则，对350篇论文中的576项基于LLM的社会模拟研究进行系统性审计，发现前沿LLM在65.2%的案例中能识别底层社会实验、50.6%的提示词预先决定了实验结果，揭示了当前LLM社会模拟在有效性验证上的严重不足。 |
| [^153] | ["Mirror" Large Language Model Evaluations of Depression are Criterion Contaminated](https://arxiv.org/abs/2508.05830) | 该研究揭示了“镜像”式LLM抑郁评估存在标准污染问题——用抑郁评估本身的语言来预测同一评估的分数会产生近乎完美的虚高结果，而独立评估显示这种优势并不存在。 |
| [^154] | [Mapping Seven Decades of Philosophy in Colombia: Dynamic Topic Modelling of Ideas y Valores](https://arxiv.org/abs/2412.04236) | 本研究通过动态主题建模技术分析哥伦比亚最具影响力的哲学期刊《Ideas y Valores》七十年来主题的演变，将哲学史的数据驱动研究拓展至哥伦比亚和拉丁美洲语境，发现价值理论、认识论和科学哲学是该地区最突出的哲学研究主题。 |
| [^155] | [A Short Survey of Viewing Large Language Models in Legal Aspect.](http://arxiv.org/abs/2303.09136) | 本文调查了大型语言模型在法律领域中的使用情况，讨论了它们应用于法律任务时所面临的法律问题，以及使用数据资源在法律领域中专门化LLMs的可能性。 |

# 详细

[^1]: 数据稀缺与模型稀疏：混合专家模型对重复数据的过拟合更严重

    Data Scarcity and Model Sparsity: Mixtures-of-Experts Overfit More to Repeated Data

    [https://arxiv.org/abs/2609.11917](https://arxiv.org/abs/2609.11917)

    该研究发现混合专家模型（MoE）相比密集模型更容易因训练数据重复而过拟合，且这种退化随模型稀疏度（由总参数量而非活跃参数量决定）的增加而加剧。

    

    随着人类书写文本资源的枯竭，重复使用语言模型训练数据已成为标准做法。先前的工作研究了数据重复对密集激活的Transformer的影响，但对于近期占主导地位的稀疏架构（如混合专家模型，MoE），尽管其具有更高的计算效率，数据重复的影响在很大程度上仍未被探索。我们在单域和多域数据混合中，以及在不同的MoE设置（包括专家数量和粒度）下改变数据重复率。我们一致发现，对于活跃参数从8000万到10亿（总参数85亿）的模型，MoE在数据重复下的性能退化更为迅速。这种效应随稀疏性增加而加剧，且由总参数量而非活跃参数量决定。虽然8000万参数的密集模型可以在最小性能退化下将数据重复8倍，但MoE在4倍时就开始受损，并迅速恶化，在全部唯一数据的设置中将其性能优势拱手让给了……

    arXiv:2609.11917v1 Announce Type: cross  Abstract: As the supply of human-written text is exhausted, it has become standard practice to repeat language model training data. Prior work has studied data repetition for densely activated Transformers, but the effects of data repetition remains largely unexplored for recently dominant sparse architectures such as Mixture-of-Experts (MoE), despite their increased compute efficiency. We vary data repetition rates across single- and multi-domain data mixes, and across MoE settings, including expert count and granularity. We consistently find, for models ranging from 80M to 1B active (8.5B total) parameters, that MoEs degrade more rapidly under data repetition. This effect increases with sparsity, dictated by total rather than active parameters. While 80M dense models can repeat data over 8x with minimal degradation, MoEs instead begin to suffer at 4x, and deteriorate rapidly, ceding their performance benefits in all-unique data settings to und
    
[^2]: Transformer中的距离泛化：为何还要费心使用位置编码？

    Distance generalization in transformers: why bother with positional encoding?

    [https://arxiv.org/abs/2609.11913](https://arxiv.org/abs/2609.11913)

    本研究通过合成延迟复制任务系统探究Transformer的距离泛化能力，考察RoPE、ALiBi等位置编码方案与无位置编码在距离分辨率上的差异、训练数据多样性的影响，以及距离迁移学习的正负效应。

    

    分布外长度泛化，即将任务从短上下文外推到更长上下文的能力，已在Transformer中得到了深入研究。本文聚焦于距离泛化，即探究在保持上下文长度固定的情况下，训练与推理阶段词元间距离发生变化时模型的性能表现。我们构建了两个合成的延迟复制任务，两者均涉及源与回忆之间的有限距离，其中词元被完整复制或选择性复制，并在训练中未见过的延迟上进行模型测试。我们回答了三个问题：(A) 相对于无位置编码（NoPE），RoPE和ALiBi等位置编码方案是否能改善距离分辨率？(B) 数据多样性，即训练中见过的词元间距离的数量，如何影响性能？(C) 距离迁移学习何时为正向、何时为负向？我们进行了全面的调查研究，发现提升我们对此的理解至关重要（原文摘要在此处截断）。

    arXiv:2609.11913v1 Announce Type: new  Abstract: Out-of-distribution length generalization, namely to extrapolate a task from short to longer context, has been studied intensively for transformers. Here we focus on distance generalization, which probes performance when inter-token distances are changed between training and inference, while keeping a fixed context length. We construct two synthetic delay copy tasks, both involving finite distances between source and recall, where tokens are copied either fully or selectively, and test models on delays unseen during training. We address three questions: (A) Do positional encoding schemes such as RoPE and ALiBi improve distance resolution relative to no positional encoding (NoPE)? (B) How does data diversity, the number of inter-token distances seen in training, affect performance? (C) When is distance transfer learning positive or negative? We present a thorough investigation, finding that it is paramount to improve our understanding of 
    
[^3]: MindTopo：基础模型能在拓扑空间中推理吗？

    MindTopo: Can Foundation Models Reason in Topological Space?

    [https://arxiv.org/abs/2609.11900](https://arxiv.org/abs/2609.11900)

    该论文提出了MindTopo基准，基于认知科学与形式拓扑学，从连续性、分离性、有序性、包含性和纽结五个拓扑属性出发，在推理与规划两个认知层面对14个多模态大语言模型的拓扑空间能力进行系统评估。

    

    空间推理不仅依赖于距离、角度和形状等度量属性，还依赖于在连续变形下保持不变的拓扑关系。认知科学认为这些拓扑关系是空间理解的基础，然而目前针对基础模型的评估大多集中于度量关系或依赖视角的关系。我们提出了MindTopo，一个基于认知科学和形式拓扑学、涵盖五个拓扑属性的拓扑直觉基准：连续性、分离性、有序性、包含性和纽结。MindTopo在两个认知层面对每个属性进行评估：推理层面要求模型识别拓扑关系或推断其如何变化；规划层面则将基础模型实例化为闭环智能体，其策略负责选择环境动作。MindTopo包含11,030个实例，覆盖13种难度可控的程序化生成任务类型。我们对14个多模态大语言模型（MLLM）进行了基准测试，并研究了智能体的不同配置。

    arXiv:2609.11900v1 Announce Type: cross  Abstract: Spatial reasoning depends not only on metric properties such as distance, angle, and shape, but also on topological relations that remain invariant under continuous deformation. Cognitive science identifies these relations as foundational to spatial understanding, yet foundation-model evaluations largely focus on metric or viewpoint-dependent relations. We introduce MindTopo, a benchmark of topological intuition across five properties grounded in cognitive science and formal topology: continuity, separation, order, enclosure, and knots. MindTopo evaluates each property at two cognitive levels. Reasoning asks a model to identify topological relations or infer how they change. Planning instantiates a foundation model as a closed-loop agent whose policy selects environment actions. MindTopo contains 11,030 instances across 13 procedurally generated task types with controllable difficulty. We benchmark 14 MLLMs and study agent configuratio
    
[^4]: Nuha-Speech：构建通用阿拉伯语语音大语言模型

    Nuha-Speech: Building General-Purpose Arabic Speech-LLMs

    [https://arxiv.org/abs/2609.11892](https://arxiv.org/abs/2609.11892)

    该论文提出Nuha-Speech计划，通过构建包含150万样本的阿拉伯语语音问答语料库、基于Qwen-Omni模型微调以及设计专门的评估框架，为通用阿拉伯语语音大语言模型建立了覆盖数据、训练到评估全流程的基础设施。

    

    随着语音大语言模型日益走向多语言化，阿拉伯语在其中的代表性仍然严重不足，这凸显了建立专门基础设施来训练和评估阿拉伯语语音大语言模型的必要性。为填补这一空白，我们推出了Nuha-Speech，这是一项旨在开发通用阿拉伯语语音大语言模型的综合性计划，涵盖数据集构建、模型训练和系统化评估。具体而言，我们构建了一个包含超过150万个训练样本的大规模阿拉伯语语音问答（SQA）语料库，以支持对广泛的语音核心任务进行指令微调。随后，该语料库被用于基于不同规模的Qwen-Omni模型变体进行监督微调。最后，我们设计了一个包含多样化任务和定制化指标的评估框架。通过这项工作，我们旨在阿拉伯语语音数据有限的约束条件下，为阿拉伯语语音大语言模型奠定基础性基础设施。

    arXiv:2609.11892v1 Announce Type: new  Abstract: As Speech Large Language Models (speech-LLMs) become increasingly multilingual, Arabic remains significantly underrepresented, highlighting the need for dedicated infrastructure to train and evaluate Arabic speech-LLMs.   To address this gap, we introduce Nuha-Speech, a comprehensive initiative to develop general-purpose Arabic speech-LLMs spanning dataset construction, model training, and systematic evaluation. Specifically, we constructed a large-scale Arabic Speech Question-Answering (SQA) corpus comprising over 1.5 million training samples to allow instruction tuning over a broad range of core speech tasks. Then, the corpus was used for supervised fine-tuning based on Qwen-Omni model variants at different scales. Finally, we designed an evaluation framework featuring diverse tasks and tailored metrics. Through this work, we aim to establish foundational infrastructures for Arabic Speech-LLMs under constraints imposed by limited Arabi
    
[^5]: 大语言模型中的领域特定幻觉检测

    Domain-Specific Hallucination Detection in Large Language Models

    [https://arxiv.org/abs/2609.11878](https://arxiv.org/abs/2609.11878)

    提出了一种结合微调DeBERTa-v3分类、蒙特卡洛Dropout不确定性量化和温度缩放校准的多信号幻觉检测流水线，在HaluEval基准上取得F1=0.915的高性能，并结合直接偏好优化（DPO）进一步改进模型表现。

    

    大语言模型生成的流畅文本可能包含不真实的陈述——这一现象被称为幻觉。我们提出了一种多信号检测流水线，结合了微调的DeBERTa-v3分类器、蒙特卡洛（MC）Dropout不确定性量化以及温度缩放校准，用于响应级别的幻觉检测。在HaluEval基准上的评估显示，我们的流水线在通用领域任务上达到F1=0.915和AUROC=0.977，各任务的F1分数分别为0.97（问答）、0.96（摘要）和0.82（对话）。MC Dropout推理进一步将准确率提升至93.2%。上下文消融研究证实模型执行的是真正的蕴含推理，而非利用表面模式——当移除知识上下文时，摘要任务的F1下降了24%。学习曲线分析表明，25%的训练数据即可捕获全数据性能的77%。除检测之外，我们还将直接偏好优化（DPO）应用于Qw……

    arXiv:2609.11878v1 Announce Type: new  Abstract: Large language models generate fluent text that can contain unfaithful claims -- a phenomenon known as hallucination. We present a multi-signal detection pipeline combining fine-tuned DeBERTa-v3 classification, Monte Carlo (MC) Dropout uncertainty quantification, and temperature-scaled calibration for response-level hallucination detection. Evaluated on the HaluEval benchmark, our pipeline achieves F1=0.915 and AUROC=0.977 on general-domain tasks, with per-task F1 scores of 0.97 (QA), 0.96 (Summarization), and 0.82 (Dialogue). MC Dropout inference further improves accuracy to 93.2%. A context ablation study confirms the model performs genuine entailment reasoning rather than exploiting surface patterns, with summarization F1 dropping 24% when knowledge context is removed. Learning curve analysis reveals that 25% of training data captures 77% of full-data performance. Beyond detection, we apply Direct Preference Optimization (DPO) to a Qw
    
[^6]: 生物学在环：CRISPR筛选中的摊销式自适应命中物发现

    Biology-in-the-loop: Amortized Adaptive Hit Discovery in CRISPR Screens

    [https://arxiv.org/abs/2609.11877](https://arxiv.org/abs/2609.11877)

    本文提出了包含1,389个CRISPR筛选实验的大规模基准AssayBench-Loop，并在此基础上构建了AssayLoop顺序实验设计框架，利用跨历史实验训练的transformer摊销式采集策略，实现CRISPR筛选中受限预算下的自适应命中物发现。

    

    许多生物学发现问题需要在有限预算下顺序地选择实验。CRISPR筛选就是一个典型的例子，因为穷举式的扰动测试通常不可行，而必须在多个实验轮次中对候选扰动进行优先级排序。尽管这一问题非常重要，但现有的自适应命中发现基准在规模和多样性方面仍然有限。在此，我们介绍了AssayBench-Loop，这是一个大规模的自适应命中发现基准，包含五个表型类别下的1,389个CRISPR筛选实验。除了支持系统化评估之外，其规模还使得跨历史实验学习采集（acquisition）策略成为可能。基于这一资源，我们提出了AssayLoop，这是一个顺序实验设计框架，它将AssayFormer（一种基于transformer的摊销式采集策略，通过在历史筛选数据上训练，能够从实验反馈中进行自适应调整）与L…相结合。

    arXiv:2609.11877v1 Announce Type: cross  Abstract: Many biological discovery problems require experiments to be selected sequentially under constrained budgets. CRISPR screening is a prominent example, as exhaustive perturbation testing is often infeasible and candidate perturbations must instead be prioritized over multiple experimental rounds. Despite the importance of this problem, existing benchmarks for adaptive hit discovery remain limited in scale and diversity. Here, we introduce AssayBench-Loop, a large-scale benchmark for adaptive hit discovery comprising 1,389 CRISPR screens across five phenotype categories. Beyond enabling systematic evaluation, its scale makes it possible to learn acquisition strategies across historical experiments. Building on this resource, we introduce AssayLoop, a sequential experimental design framework combining AssayFormer, a transformer-based amortized acquisition policy trained across historical screens to adapt from experimental feedback, with L
    
[^7]: 人类建造的最后一个AI：迈向真正的递归自我改进

    The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement

    [https://arxiv.org/abs/2609.11873](https://arxiv.org/abs/2609.11873)

    本文提出递归自我改进（RSI）概念及其从改进执行自主性到递归元改进的发展路线图，通过Headroom-Closed指数揭示现有大语言模型的局限，并结合行业实践识别出实现真正AI自我改进的关键挑战。

    

    递归自我改进（RSI）使AI系统能够将经验和反馈转化为持久性的改变，从而同时提升其自身能力和未来的改进过程。我们首先使用Headroom-Closed指数（HCI）揭示现有大语言模型存在的问题，然后介绍RSI概念及其发展路线图：从改进执行自主性、改进策略自主性、经验获取自主性和环境适应自主性，到递归元改进。接下来，我们考察了RSI在不同场景（如科学发现、具身智能、软件工程）中的表现，重点阐述了它们各自不同的需求和发展速度。借鉴多样的行业实践和初步的实证证据，我们将RSI研究与实际系统联系起来，并识别出实现真正RSI所面临的关键挑战。

    arXiv:2609.11873v1 Announce Type: cross  Abstract: Recursive self-improvement (RSI) enables AI systems to turn experience and feedback into persistent changes that improve both their capabilities and the process of future improvement. We first use the Headroom-Closed Index (HCI) to reveal the problems of existing LLMs, then introduce the RSI concept and its development roadmap: from improvement-execution autonomy, improvement-strategy autonomy, experience-acquisition autonomy, and environment-adaptation autonomy, to recursive meta-improvement. Next we examine RSI across scenarios (e.g., scientific discovery, embodied intelligence, software engineering), highlighting their distinct requirements and development speeds. Drawing on diverse industry practices and preliminary empirical evidence, we connect RSI research with practical systems and identify key challenges to achieving genuine RSI.
    
[^8]: 奥古斯丁式BabyLM：实指定名能教给小语言模型什么、不能教什么

    Augustinian BabyLM: What Ostensive Definition Can and Cannot Teach a Small Language Model

    [https://arxiv.org/abs/2609.11870](https://arxiv.org/abs/2609.11870)

    本研究将圣奥古斯丁的“实指定名”词义学习思想应用于小型语言模型，发现视觉初始化的词嵌入会留下持续到训练结束的印记，且仅在物体属性知识等零样本任务中带来提升，而对大多数语法能力基准测试没有影响。

    

    语言模型通常以随机的词嵌入开始训练：“香蕉”的含义必须从训练语料中学习。我为一个在1000万词上训练的小型掩码语言模型实现了圣奥古斯丁的词语学习图景——通过实指获得意义：在训练之前，具有视觉基础的词会获得由其标注的图像区域导出的嵌入，而其他词则以随机初始化开始。视觉初始化留下了可测量的印记，并一直持续到训练结束。与此同时，这种效应在大多数探测抽象语法知识的BabyLM基准测试中仍然不可见：视觉初始化在这些测试中并不影响性能。唯一的零样本例外是物体属性知识（COMPS，Misra等人，2023），此时种子嵌入在所有配置中都有帮助。为进一步验证这一结果，我构建了一个针对语料库定制的Visual-Property Swap基准测试版本（Lin等人，2026），该测试检验颜色、材质等物体属性。

    arXiv:2609.11870v1 Announce Type: new  Abstract: A language model normally begins training with random word embeddings: whatever 'banana' means must be learned from training corpora. I implement St. Augustine's picture of word learning, meaning by ostension, for a small masked language model (DeBERTa) trained on 10M words: before training, visually grounded tokens receive embeddings derived from the image regions they label; other tokens start random. Visual initialization leaves a measurable imprint that lasts until the end of training. At the same time, the effect remains invisible under most BabyLM benchmarks, which probe abstract grammatical knowledge: visual initialization does not affect performance there. The only zero-shot exception is object-property knowledge (COMPS, Misra et al. 2023), where seeding helps in every configuration. To follow up on this result, I build a corpus-tailored version of the Visual-Property Swap benchmark (Lin et al., 2026), which tests color, material
    
[^9]: 认知取向可预测美国国会议员的立法效能

    Epistemic orientation predicts legislative effectiveness among members of the US Congress

    [https://arxiv.org/abs/2609.11865](https://arxiv.org/abs/2609.11865)

    本研究通过测量国会议员演讲和推文中证据导向与直觉导向语言的差异（EMI分数），发现议员的认知取向与其立法效能显著相关，表明基于证据的沟通风格能够预测立法者的实际工作成效。

    

    基于真相和证据的沟通为民主治理、问责制和集体决策提供了重要基础。先前的研究表明，自20世纪70年代中期以来，美国国会全体会议演讲中面向证据的语言有所减少，与此同时立法生产率和政治极化也发生了更广泛的变化。本研究将分析对象从国会会期转向个体国会议员，以考察认知取向在立法者之间是否存在系统性差异，以及它与政治行为和立法效能是否相关。研究者使用“证据减直觉”分数来测量国会全体会议演讲和推特帖子中证据导向语言相对于直觉导向语言的普遍程度，并将这些测量与议员层面的意识形态、机构职位、传播环境和立法效能分数等数据联系起来。结果显示，更（摘要在此处截断）

    arXiv:2609.11865v1 Announce Type: new  Abstract: Truth and evidence-based communication provide important foundations for democratic governance, accountability, and collective decision-making. Prior work shows that evidence-oriented language in US congressional floor speeches has declined since the mid-1970s, alongside broader changes in legislative productivity and polarization. This study shifts the analysis from congressional sessions to individual members of Congress to examine whether epistemic orientation varies systematically across legislators and whether it relates to political behavior and legislative effectiveness. Using the Evidence-Minus-Intuition (EMI) score, we measure the relative prevalence of evidence-oriented versus intuition-oriented language in congressional floor speeches and Twitter posts. We link these measures to legislator-level data on ideology, institutional position, communication context, and Legislative Effectiveness Score (LES). The results show that mor
    
[^10]: RetroThinker：在语音大语言模型中实现回顾性思考

    RetroThinker: Enabling Retrospective Thinking in Speech LLMs

    [https://arxiv.org/abs/2609.11864](https://arxiv.org/abs/2609.11864)

    该论文提出了RetroThinker多阶段后训练框架，使流式语音大语言模型能够在推理过程中自我验证并即时修正思维链推理步骤，从而在满足实时语音交互延迟约束的同时提升复杂推理能力。

    

    语音大语言模型相比级联式自动语音识别（ASR）与基于文本的语言模型架构，能够提供更低的延迟，并保留通常在此类级联架构中丢失的副语言细微特征。然而，语音大语言模型在复杂推理任务上仍落后于纯文本大语言模型，而实时语音交互又施加了严格的延迟约束。尽管先前的工作采用思维链（CoT）和并发推理来增强推理能力而不引入过高的延迟，但固有的精度与延迟之间的权衡依然存在。在本文中，我们研究了流式语音大语言模型能否在运行过程中动态地修正其推理轨迹。我们提出了RetroThinker，这是一个多阶段后训练框架，使Moshi模型能够在推理过程中对CoT步骤进行自我验证和前向修正。RetroThinker将在精选的回顾性思考数据上进行的监督微调（SFT）与基于长度的直接偏好优化相结合。

    arXiv:2609.11864v1 Announce Type: cross  Abstract: Speech large language models (SpeechLLMs) offer reduced latency and retain paralinguistic nuances that are typically lost in cascaded automatic speech recognition (ASR) and text-based LM architectures. However, they continue to lag behind text-only LLMs on complex reasoning tasks, while real-time spoken interaction imposes strict latency constraints. Although prior works employ Chain-of-Thought (CoT) and concurrent reasoning to enhance reasoning capabilities without inducing prohibitive delays, an inherent accuracy-latency trade-off persists. In this paper, we investigate whether a streaming SpeechLLM can dynamically revise its reasoning traces on the fly. We introduce RetroThinker, a multi-stage post-training framework that equips the Moshi model to self-verify and forward-correct CoT steps during inference. RetroThinker combines supervised fine-tuning (SFT) on curated retrospective thinking data with length-based direct preference op
    
[^11]: IndicTriMix：开发面向三语混合的语言识别数据集与模型

    IndicTriMix: Developing Language Identification Datasets and Models for Tri-Language Code-Mixing

    [https://arxiv.org/abs/2609.11851](https://arxiv.org/abs/2609.11851)

    该论文针对印地语、古吉拉特语和孟加拉语的三语混合文本，将词元级语言识别形式化为序列标注任务并微调 MuRIL 和 XLM-RoBERTa 模型，同时发布了人工标注的基准数据集和两种三语混合文本生成方法。

    

    混合语言文本中的语言识别主要出现在社交媒体中，当用户在单个话语内频繁切换多种语言时，这一任务至关重要，准确识别混合文本中各词元的语言已成为迫切需求。传统的语言识别模型是为单语文本设计的，并不适合混合语言场景下的词元级语言识别。我们将该任务形式化为序列标注问题，并对最适合印度语言的上下文感知 Transformer 模型 MuRIL 和 XLM-RoBERTa 进行微调。我们在三种不同的数据配置（印地语、古吉拉特语和孟加拉语）上评估这些系统，以预测单个词元的语言标签。我们发布了一个包含人工标注测试集的混合词元语言识别基准，并提出了两种利用三种语言平行句子进行混合文本生成的方法。

    arXiv:2609.11851v1 Announce Type: new  Abstract: Language identification in code-mixed text, largely observed in social media, is highly essential when users frequently switch between multiple languages within a single utterance. Accurately identifying the languages of code-mixed tokens becomes an urgent necessity. Traditional language identification models, designed for monolingual text, are not well suited for token-level language identification in code-mixed settings. We formulate the task as a sequence labeling problem and fine-tune contextual transformer-based models MuRIL and XLM-RoBERTa best suited for Indian languages. We evaluate these systems on three different data configurations (Hindi, Gujarati, and Bengali) to predict language labels for individual tokens. We release a benchmark for language identification in code-mixed tokens with manually annotated test sets. We propose two approaches of code-mixed generation using parallel sentences of three languages. The trained mode
    
[^12]: 目标泄漏而非模型类别解释了基于调查的心血管筛查中报告的准确率：对玻璃盒模型和表格基础模型的泄漏分层审计

    Target leakage, not model class, explains reported accuracy in survey-based cardiovascular screening: a leakage-tiered audit of glass-box and tabular foundation models

    [https://arxiv.org/abs/2609.11838](https://arxiv.org/abs/2609.11838)

    该研究通过对十种模型类别在五级泄漏风险特征层级上的系统审计发现，基于国民健康调查的心血管筛查模型所报告的高准确率主要由目标泄漏而非模型类别驱动，移除两个诊断后特征导致所有模型AUROC下降约0.05。

    

    基于国民健康调查训练的心血管筛查模型通常报告接近0.89的受试者工作特征曲线下面积（AUROC）。我们探究了这种准确率是反映了真实学习效果还是目标泄漏，表格基础模型是否会改变这一答案，以及部署所需的各项属性能否经受住联合检验。我们针对2022年行为风险因素监测系统（BRFSS）442,067名受访者的心肌梗死患病情况，对涵盖线性、树集成、神经网络、玻璃盒和表格基础模型五大类别的十种分类器进行了基准测试，并按照泄漏风险递减划分为五个特征层级。每个模型都接受了判别力、校准度、显式筛查阈值下的公平性、共形覆盖、解释忠实性和推理成本的审计，随后在模型和阈值冻结的条件下应用于2023年的430,755名受访者。移除两个诊断后特征使所有模型的AUROC下降0.049-0.051，……

    arXiv:2609.11838v1 Announce Type: new  Abstract: Cardiovascular screening models trained on national health surveys routinely report areas under the receiver operating characteristic curve (AUROC) near 0.89. We asked whether that accuracy reflects learning or target leakage, whether tabular foundation models change the answer, and whether the properties deployment requires survive joint examination. We benchmarked ten classifiers spanning linear, tree-ensemble, neural, glass-box, and tabular foundation classes for prevalent myocardial infarction in 442,067 respondents of the 2022 Behavioral Risk Factor Surveillance System across five feature tiers of decreasing leakage risk. Each was audited for discrimination, calibration, fairness at an explicit screening threshold, conformal coverage, explanation faithfulness, and inference cost, then applied -- models and thresholds frozen -- to 430,755 respondents of 2023. Removing two post-diagnostic features cost every model 0.049-0.051 AUROC, c
    
[^13]: SpecGuard：零开销的推理时后门检测

    SpecGuard: Inference-Time Backdoor Detection For Free

    [https://arxiv.org/abs/2609.11799](https://arxiv.org/abs/2609.11799)

    SpecGuard巧妙地重新利用投机解码中的验证过程，在零额外模型计算成本下实现推理时的大语言模型后门检测。

    

    大型语言模型通常经过微调、共享或从第三方下载，因此部署的模型可能携带隐藏的后门——该后门在正常输入下表现正常，但当出现秘密触发器时会切换为攻击者控制的行为。虽然后门可以在部署前进行审计，但对于频繁更新的模型，运行时监控仍然十分重要。挑战在于大语言模型的服务对延迟非常敏感：现有的推理时检测器要么依赖于对触发器形式的假设（在对隐蔽攻击时可能失效），要么需要额外的模型计算（例如输入扰动或额外的生成过程）。我们提出了SpecGuard，一种推理时后门检测器，它重新利用投机解码（speculative decoding）机制，且不增加任何额外的模型计算成本。投机解码通过使用小型草稿模型提出候选token、再由目标模型进行验证来加速推理。我们观察到，这一验证过程（摘要截断于此）……

    arXiv:2609.11799v1 Announce Type: cross  Abstract: Large language models are often fine-tuned, shared, or downloaded from third parties, so a deployed model may carry a hidden backdoor that behaves normally on benign inputs but switches to attacker-controlled behavior when a secret trigger appears. While backdoors can be audited before deployment, runtime monitoring remains important for models that are frequently updated. The challenge is that LLM serving is latency-sensitive: existing inference-time detectors either rely on assumptions about the trigger form, which can fail on stealthy attacks, or require extra model computation, such as input perturbations or an additional generation pass.   We introduce SpecGuard, an inference-time backdoor detector that repurposes speculative decoding at zero added model-computation cost. Speculative decoding speeds up inference by using a small draft model to propose tokens and a target model to verify them. We observe that this verification proc
    
[^14]: 超越词错误率：ASR与音频语言模型在英语-约鲁巴语码转换语音上的切换感知评估

    Beyond Word Error Rate: A Switch Aware Evaluation of ASR and Audio Language Models on English Yoruba Code-Switched Speech

    [https://arxiv.org/abs/2609.11786](https://arxiv.org/abs/2609.11786)

    该论文提出了一套切换感知的评估指标（包括切换入口标记错误率SETER等），揭示总体词错误率会掩盖码转换语音识别的真实表现，且领先的音频语言模型虽然与最佳ASR模型的WER相当，却在所有码转换相关指标上显著更优。

    

    自动语音识别（ASR）系统和音频语言模型在单语基准测试上已报告较低的错误率，但它们在低资源、变音符号丰富的语言中的码转换语音上的表现仍缺乏充分刻画。我们对十一个现代系统（六个ASR模型和五个音频语言模型）在英语-约鲁巴语码转换语音上进行了切换感知评估，使用了一个确定性的2000条语音评估集和统一的评分流程。除词错误率（WER）之外，我们报告了切换定位的诊断指标：切换入口标记错误率（SETER）、窗口化切换点错误率、语言特定错误率以及变音符号不敏感的WER。我们的核心发现是，总体WER会掩盖码转换行为。按WER衡量的最佳系统（一个ASR模型）与一个领先的音频语言模型在WER上统计上无显著差异，但该音频语言模型在所有切换定位指标上都显著更优。

    arXiv:2609.11786v1 Announce Type: new  Abstract: Automatic speech recognition (ASR) systems and audio language models (audio LMs) now report low error rates on monolingual benchmarks, but their behavior on code switched speech in low resource, diacritic rich languages remains poorly characterized. We present a switch aware evaluation of eleven modern systems (six ASR models and five audio LMs) on English Yoruba code-switched speech, using a deterministic 2000 utterance evaluation set and a shared scoring pipeline. Beyond word error rate (WER), we report switch localized diagnostics: a switch entry token error rate (SETER), windowed switch point error rates, language specific error rates, and a diacritic insensitive WER. Our central finding is that aggregate WER hides code switching behavior. The best system by WER (an ASR model) is statistically indistinguishable from a leading audio LM on WER, yet the audio LM is significantly better on every switch localized metric. Across faithful s
    
[^15]: 基于Whisper的多语言视频语音转录技术促进跨文化理解

    Whisper-Based Speech Transcription from Videos Across Multiple Languages for Cross-Cultural Understanding

    [https://arxiv.org/abs/2609.11772](https://arxiv.org/abs/2609.11772)

    本文提出了基于Whisper的多语言视频语音转录技术，使缺乏深厚语音处理专业知识的跨文化工具开发者也能轻松创建多语言转录文本，在七种语言上实现了平均30%的转录错误率。

    

    跨文化理解在当今高度互联的跨国世界中变得越来越重要。基于大语言模型（LLM）技术的成功正在推动自动化工具的发展，以帮助试图在跨文化环境中取得成功的非母语人士。构建此类自动化工具通常需要利用现实世界中的文本、音频和视频数据。本文提出了改进基于语音识别的多语言视频转录文本创建的技术，以更好地训练这些自动化工具。重点在于让跨文化工具开发者无需深厚的语音处理专业知识即可轻松使用的流程和语音工具。使用YouTube上公开可用的视频和基于Whisper的工具，观察到七种语言（西班牙语、日语、韩语、普通话、土耳其语、俄语和希伯来语）的平均转录错误率为30%。通过适量的微调……（摘要在此处截断）

    arXiv:2609.11772v1 Announce Type: cross  Abstract: Cross-cultural understanding has become increasingly important in today's highly connected, cross-national world. The success of LLM-based technologies is now driving the development of automated tools to aid understanding for nonnative people trying to succeed in cross-cultural environments. Building such automated tools is often done by leveraging in-thewild text, audio, and video data. This paper presents techniques for improving speech recognition-based transcript creation in multiple languages from videos to better train these automated tools. The focus is on processes and speech tools that can easily be used by cross-cultural tool builders without requiring deep speech processing expertise. Using publicly available videos from YouTube and Whisper-based tools, average transcription error rate across seven languages (Spanish, Japanese, Korean, Mandarin, Turkish, Russian, and Hebrew) of 30% are observed. With a modest amount of fine
    
[^16]: 2023至2026年医学大语言模型研究中不断扩大的评估差距

    The widening evaluation gap in medical large language model research 2023 to 2026

    [https://arxiv.org/abs/2609.11770](https://arxiv.org/abs/2609.11770)

    该研究揭示医学大语言模型研究的评估滞后从1.33个季度扩大至6.08个季度，其中严谨的随机试验反而倾向于评估更过时的模型，凸显了方法学严谨性与模型时效性之间的根本矛盾。

    

    大语言模型每隔几个季度就会被更新换代，而临床证据的形成则需要数年时间。我们探究了医学研究是否能跟上其所评估系统的步伐。通过PubMed检索2023年1月至2026年6月期间涵盖十四个临床领域的11,628条记录，研究数量增长了45倍，但其中仅2.5%采用了随机对照或前瞻性设计。评估滞后时间（即从研究中所命名最新模型的发布到该研究自身发表之间的时间间隔）从1.33个季度扩大至6.08个季度。由于已停用的模型会机械性地老化，我们以模型组合保持不变的情景作为反事实基准进行对比：向更新系统的迁移仅抵消了56%的漂移（95% CI 50-65）。随机试验所评估的模型比其他研究设计所评估的模型平均老旧4.6个季度（P = 3 x 10^-19）；然而在那些指名了仍在开发中的模型的研究中，各类研究设计之间并无差异；62%的随机试验评估的是已停产的模型系列。严谨性与时效性之间存在张力，而这种张力……

    arXiv:2609.11770v1 Announce Type: new  Abstract: Large language models are superseded every few quarters; clinical evidence takes years. We asked whether medical research is keeping pace with the systems it evaluates. PubMed returned 11,628 records for January 2023 to June 2026 across fourteen clinical domains, growing 45-fold; 2.5% used a randomised, controlled or prospective design. Evaluation lag, from a study's newest named model release to its own publication, widened from 1.33 to 6.08 quarters. Because discontinued models age mechanically, we benchmarked this against a counterfactual holding model composition fixed: migration to newer systems offset only 56% of the drift (95% CI 50-65). Randomised trials evaluated models a median 4.6 quarters older than other designs (P = 3 x 10^-19), yet among studies naming a model still under development no design differed from any other; 62% of randomised trials evaluated a discontinued family. Rigour and currency are in tension, and that ten
    
[^17]: 识别不等于逆转：事实保持型新闻框架的受控逆转测试

    Recognizing Is Not Reversing: A Controlled Inversion Test of Fact-Preserving News Framing

    [https://arxiv.org/abs/2609.11769](https://arxiv.org/abs/2609.11769)

    该研究通过受控逆转实验发现，大语言模型虽能较好地保持事实并识别新闻框架，却几乎无法逆转已知的框架变换（逆转率仅约0.044–0.068），证明“识别框架”与“逆转框架”是截然不同的能力。

    

    大语言模型（LLM）越来越多地被用于分析和改写新闻，然而当前的框架研究主要评估生成、检测或改写后的文本是否显得更加中立，并没有直接展示模型能否在保持事实不变的前提下撤销一个已知的框架变换。我们引入了一项受控逆转测试，涵盖三种既定的文本框架实现方式：评价性词汇、施事性实现和信息显著性。基于60篇新闻文章和三种干预强度，共产生了540对保持原子事实并记录编辑操作的配对变体。在Qwen、DeepSeek和Kimi模型上，事实保持率接近0.84，而干预逆转率仅为0.044–0.068。即使框架类型和方向都被正确识别，汇总的逆转率也仅达到0.071。这些结果揭示了事实保真度、框架识别与框架逆转之间的明显分离。

    arXiv:2609.11769v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to analyze and rewrite news, yet current framing studies mainly evaluate generation, detection, or whether rewritten text appears more neutral. They do not directly show whether a model can undo a known framing transformation while keeping the facts fixed. We introduce a controlled inversion test over three established textual realizations of framing: evaluative lexis, agency realization, and information salience. Across 60 news articles and three intervention strengths, this yields 540 paired variants with preserved atomic facts and recorded edits. Across Qwen, DeepSeek, and Kimi, factual preservation remains near 0.84, whereas intervention reversal is 0.044--0.068. Even when both framing type and direction are recognized correctly, pooled reversal reaches 0.071. These results reveal a clear separation between factual fidelity, framing recognition, and framing inversion: recognizing how
    
[^18]: 面向在线策略蒸馏的统一逐词元门控族：带多通道与偏置系数的FKL/RKL混合

    A Unified Per-Token Gating Family for On-Policy Distillation: FKL/RKL Mixing with Multi-Channel and Bias Coefficients

    [https://arxiv.org/abs/2609.11768](https://arxiv.org/abs/2609.11768)

    该论文提出了一个统一的四系数逐词元门控参数化方法，将EOPD和ToDi作为其一维限制的特殊情形统一起来，并通过多通道组合与显式偏置这两个额外自由度，在在线策略蒸馏实验中显著超越了单通道门控限制方法。

    

    逐词元的前向/反向KL散度损失门控已成为在线策略知识蒸馏（OPD）的标准技术，但现有方法如EOPD（Jin等，2026）和ToDi（Jung等，2025）各自固定了单一的门控信号和单一的门控方向，且两者从未被直接比较。我们引入了一个四系数参数化 lambda_t = sigma(a * h_t + b * u(x) + c + d * gap_t)，其中与方向对齐的EOPD和ToDi代理形式作为其一维（1D）限制情形出现，同时该参数化还引入了多通道组合和显式偏置作为额外的自由度。在TweetEval（Barbieri等，2020）的情感和仇恨数据集上，使用Qwen3-32B教师模型和Qwen3-4B学生模型进行实验，完整门控族中的配置在36个可比单元中的33个里达到了比同等幅度的单通道（仅熵/仅差距）1D限制更高的准确率，而一项26单元的均值匹配隔离实验将动态门控置于……（原文在此处截断）

    arXiv:2609.11768v1 Announce Type: cross  Abstract: Per-token gating of forward/reverse KL losses has become a standard technique for on-policy knowledge distillation (OPD), but existing methods such as EOPD (Jin et al., 2026) and ToDi (Jung et al., 2025) each fix a single gating signal and a single gating direction, and the two have never been compared directly. We introduce a four-coefficient parameterization lambda_t = sigma(a * h_t + b * u(x) + c + d * gap_t) in which direction-aligned proxies of EOPD and ToDi appear as one-dimensional (1D) restrictions, and which adds multi-channel composition and an explicit bias as further degrees of freedom. On TweetEval (Barbieri et al., 2020) emotion and hate, with a Qwen3-32B teacher and a Qwen3-4B student, configurations in the full family reach higher accuracy than the matched-magnitude single-channel (entropy-only / gap-only) 1D restrictions in 33 of 36 comparable cells, and a 26-cell mean-match isolation experiment places dynamic gating a
    
[^19]: 面向联邦多语言语音大语言模型的组件感知差分隐私

    Component-Aware Differential Privacy for Federated Multilingual Speech-LLMs

    [https://arxiv.org/abs/2609.11762](https://arxiv.org/abs/2609.11762)

    提出α-split双池差分隐私分配方法，通过将声学编码器与语言解码器参数归入独立裁剪池，解决了语音大语言模型联邦训练中因组件间更新范数差异巨大导致的预算坍塌问题。

    

    逐层差分隐私（DP）裁剪通过按参数数量成比例地分配逐矩阵裁剪预算，从而提升联邦学习中的梯度保真度。我们证明，当声学编码器与语言解码器的更新范数相差一个数量级时，这一方案在语音大语言模型（speech-LLMs）上会失效。单池逐层方法会遭受“跨组件预算坍塌”问题，导致词错误率（WER）远偏离全局平滑裁剪的效果，甚至使训练完全崩溃。当范数不平衡较轻微时，自适应单池方法能够部分恢复性能，这证实了坍塌的严重程度随组件间范数比例的增大而加剧。我们在六种逐层方法和三种语音大语言模型架构上对根本原因进行了实证诊断。随后，我们提出α-split，一种双池分配方法，将编码器与LLM参数归一化到相互独立的池中，并证明了联合ℓ2敏感性

    arXiv:2609.11762v1 Announce Type: new  Abstract: Per-layer differential privacy (DP) clipping improves gradient fidelity in federated learning by allocating per-matrix clipping budgets proportional to parameter count. We show that this recipe breaks for speech large language models (speech-LLMs), when the acoustic encoder and the language decoder differ by an order of magnitude in update norm. Single-pool per-layer methods suffer \emph{cross-component budget collapse}, dragging word error rate (WER) far from flat global clipping or collapsing training entirely. When the norm imbalance is milder, adaptive single-pool methods partially recover, confirming that collapse severity scales with the inter-component norm ratio. We empirically diagnose the root cause across six per-layer methods and three speech-LLM architectures. We then propose \emph{$\alpha$-split}, a two-pool allocation that normalises encoder and LLM parameters into independent pools, and show that joint $\ell_2$ sensitivit
    
[^20]: RAG-Safety-Bench：检索增强大语言模型安全性的可靠评估

    RAG-Safety-Bench: Reliable Evaluation of Retrieval-Augmented LLM Safety

    [https://arxiv.org/abs/2609.11758](https://arxiv.org/abs/2609.11758)

    提出了RAG-Safety-Bench基准测试，通过消除检索器质量的干扰并将问题分解为四种条件（非RAG、含答案文档、相关无答案文档、随机文档），可靠地评估检索增强生成对大语言模型安全性的影响。

    

    允许大语言模型（LLM）从一组可信文档中检索信息可以提高可靠性并减少幻觉。然而，最近的研究表明，当被要求生成有害或危险内容时，检索增强生成（RAG）可能对生成响应的整体安全性产生意想不到的副作用。随着越来越多的终端用户转向使用RAG将企业文档和知识库整合到基于LLM的系统中，人们需要更清晰地理解导致这一结果的机制。我们提出了RAG-Safety-Bench，这是一个用于衡量RAG对LLM模型安全影响的基准测试。通过消除检索器质量带来的干扰效应，并将问题清晰地划分为四种条件——非RAG、使用包含有害请求答案的oracle文档的RAG、使用与有害请求相关但不包含具体答案的文档的RAG、以及使用随机文档的RAG（原文在此处被截断）……

    arXiv:2609.11758v1 Announce Type: new  Abstract: Allowing large language models (LLMs) to retrieve information from a set of trusted documents can increase reliability and reduce hallucination. However, recent work has demonstrated that retrieval-augmented generation (RAG) can have unintended side effects on the overall safety of the generated responses, when prompted for harmful or dangerous content. A clearer understanding of the mechanisms leading to this result is needed, as increasing numbers of end users turn to RAG to incorporate corporate documents and knowledge bases into LLM-based systems. We introduce RAG-Safety-Bench, a benchmark to measure the safety impact of RAG on LLM models. By removing the confounding effect of retriever quality, and cleanly separating the problem into four conditions -- non-RAG, RAG with an oracle document containing the answer to the harmful request, RAG with documents related to the harmful request but without the specific answer, and RAG with rand
    
[^21]: SIRF：面向工业内容风控的规范内化风险基础模型

    SIRF: A Spec-Internalized Risk Foundation Model for Industrial Content Risk Control

    [https://arxiv.org/abs/2609.11752](https://arxiv.org/abs/2609.11752)

    SIRF通过持续预训练将平台风控规范内化到模型权重中，无需人工标注即可在超低延迟、仅输出判定的部署形态下实现高精度风险处置，仅用约7000万token的CPT便将P95精确率下的黑名单召回率提升15.1个百分点且不损害通用能力。

    

    对于工业内容风控而言，真正的部署约束不是平均准确率，而是在高精度与秒级延迟下能自动处置多少风险内容。我们提出SIRF（规范内化风险基础模型），它通过持续预训练（CPT）将平台的复杂规范策略内化到模型权重中——这些策略通过EntiGraph、MAGA改写和账号级思维链（CoT）合成，无需额外人工标注——从而在超低延迟、仅输出判定结果的部署形态下实现高精度的规则执行。一项同源对照比较实验（Qwen3-8B-SFT对比SIRF-8B-SFT，两者采用相同的策略注入和仅判定输出形式，唯一差异在于是否进行基于策略的CPT）将性能提升归因于规范内化：SIRF-8B-SFT在95%精确率约束下达到71.3%的黑名单召回率（Black Recall@P95），较基线提升15.1个百分点，且仅使用约7000万个CPT token，未损害模型通用能力；在本接口下纳入的可获取logprob的模型中，它……

    arXiv:2609.11752v1 Announce Type: cross  Abstract: For industrial content risk control, the real deployment constraint is not average accuracy but how much risk can be auto-handled under high precision and second-level latency. We present SIRF (Spec-Internalized Risk Foundation Model), which internalizes a platform's complex policies, synthesized without additional human annotation via EntiGraph, MAGA rewriting and account-level chain-of-thought (CoT), into the weights via continued pretraining (CPT), so rules are applied at high precision under an ultra-low-latency, verdict-only deployment. A controlled same-source comparison (Qwen3-8B-SFT vs. SIRF-8B-SFT, identical policy injection and verdict-only output form, differing only in policy-grounded CPT) attributes the gain to internalization: SIRF-8B-SFT reaches 71.3% Black Recall@P95, +15.1pp over the baseline, using only ~70M CPT tokens without harming general ability, and among included, logprob-available models under this interface i
    
[^22]: LOCUS：面向Token高效语言生成的任务感知低秩后训练

    LOCUS: Task-Aware Low-Rank Post-Training for Token-Efficient Language Generation

    [https://arxiv.org/abs/2609.11739](https://arxiv.org/abs/2609.11739)

    LOCUS通过选择任务感知的低秩适配子空间进行后训练，在不修改偏好对齐损失且仅更新不到0.3%参数的情况下，将大模型输出长度最多缩短39.84%，从而在保持效用的同时大幅降低推理成本。

    

    大语言模型的服务成本与输出序列长度直接成正比，然而标准的偏好对齐方法往往会增加回复的冗长度而不提升实际效用。我们研究后训练更新的参数化方式是否会影响生成长度：低秩子空间可以在不修改对齐损失的情况下改变序列长度。我们提出了LOCUS，一种选择任务感知低秩适配子空间的方法，用于在效用约束下最小化输出token成本。在该子空间内，后训练在冻结骨干网络的前提下保留原生的偏好优化目标。在Anthropic HH-RLHF对话偏好数据上，我们评估了两个约3B规模的解码器骨干模型——Pythia-2.8B和Qwen2.5-3B——并与协议匹配的全参数DPO、DrDPO分支以及已发布的SamPO检查点进行对比。LOCUS在Pythia-2.8B上将续写长度最多减少39.84%，在Qwen2.5-3B上减少14.87%至17.58%，同时仅更新0.24%至0.28%的模型参数。

    arXiv:2609.11739v1 Announce Type: new  Abstract: Large language model serving costs scale directly with output sequence length, yet standard preference alignment often inflates response verbosity without improving utility. We study whether the parameterization of post-training updates affects generation length: low-rank subspaces alter sequence length without modifying the alignment loss. We present LOCUS, a method that selects a task-aware low-rank adaptation subspace to minimize output-token cost subject to a utility constraint. Within this subspace, post-training retains the native preference objective with a frozen backbone. Across Anthropic HH-RLHF dialogue preferences, we evaluate two $\sim$3B decoder backbones, Pythia-2.8B and Qwen2.5-3B, against protocol-matched full-parameter DPO and DrDPO branches and the released SamPO checkpoint. LOCUS reduces continuation length by up to 39.84\% on Pythia-2.8B and by 14.87--17.58\% on Qwen2.5-3B while updating only 0.24--0.28\% of model pa
    
[^23]: Eloquence 团队参加 Interspeech 2026 MLC-SLM 挑战赛任务 2 的提交方案

    The Eloquence submission for Task 2 of the Interspeech 2026 MLC-SLM challenge

    [https://arxiv.org/abs/2609.11724](https://arxiv.org/abs/2609.11724)

    Eloquence 团队针对跨 21 种语言的多语言语音多项选择问答任务探索了三种方法，其中基于多模态上下文学习的冻结 Voxtral-24B 模型纠正标签偏差后取得 0.81 的最佳宏平均准确率，三种方法均大幅超越官方基线。

    

    本文详细介绍了 Eloquence 团队参加 Interspeech 2026 第二届 MLC-SLM 挑战赛任务 2 的方法，该任务涉及跨越 21 种语言的多语言多项选择题问答（MCQA）。文中探索了三种方法。第一，我们通过 LoRA 微调 Voxtral-Mini-3B，并结合跨语言数据增强、ASR 转录文本增强和时间戳感知的音频裁剪，在评估第二阶段取得了 0.72 的宏平均准确率。第二，我们对冻结的 Voxtral-24B 模型应用多模态上下文学习（ICL）以纠正强烈的标签偏差，达到了 0.81 的最佳结果。第三，一个免训练的检索系统，基于结合声学身份、语义内容和知识图谱的三层语音锚定记忆，达到了 0.68。所有三个系统都大幅优于官方基线。

    arXiv:2609.11724v1 Announce Type: new  Abstract: This paper details the Eloquence team's approach to Task 2 of the 2nd MLC-SLM challenge at Interspeech 2026, which involves multilingual Multiple-Choice Question Answering (MCQA) across 21 languages. Three approaches are explored. First, we fine-tune Voxtral-Mini-3B via LoRA with cross-lingual data augmentation, ASR transcript augmentation and timestamp-aware audio cropping, achieving 0.72 macro-accuracy on evaluation Phase 2. Second, we apply multimodal in-context learning (ICL) to the frozen Voxtral-24B model to correct a strong label bias, reaching 0.81, our best result. Third, a training-free retrieval system based on a three-layer voice-anchored memory combining acoustic identity, semantic content, and a knowledge graph achieves 0.68. All three systems substantially outperform the official baseline.
    
[^24]: 为什么训练后量化有效？

    Why Does Post-Training Quantization Work?

    [https://arxiv.org/abs/2609.11716](https://arxiv.org/abs/2609.11716)

    该论文揭示了训练后量化对预训练大语言模型有效的机制——层新引入的误差与从输入继承的误差相互抵消等两种机制，使量化误差不会随深度累积，从而保持了模型性能。

    

    训练后量化通过以降低的精度存储权重来压缩大语言模型（LLM），每个量化后的权重都会向隐藏状态引入误差。直观上看，这些误差应该会随网络深度累积并破坏下一词元预测；随机初始化的模型确实会迅速累积这些偏差，而量化后的预训练模型累积的隐藏状态误差却少得多，并在很大程度上保持了下游任务的性能，尽管它们从未在量化噪声下进行过训练。这就引出了本文要解决的问题：为什么训练后量化有效？通过对比全精度和量化后的前向传播过程，我们识别出了表征预训练模型量化鲁棒性的两种机制。第一，某一层新引入的误差往往与其从该层输入继承而来的误差方向相反。两者部分抵消，使得全精度与量化前向传播之间的偏差增长缓慢。

    arXiv:2609.11716v1 Announce Type: cross  Abstract: Post-training quantization compresses large language models (LLMs) by storing their weights at reduced precision, and each quantized weight introduces an error into the hidden states. Naively, these errors should accumulate with depth and corrupt next-token prediction; randomly initialized models accumulate these discrepancies rapidly, whereas quantized pretrained models accumulate much less hidden-state error and largely maintain downstream task performance, even though they were never trained with quantization noise. This raises the question we address: why does post-training quantization work? Comparing full-precision and quantized forward passes, we identify two mechanisms that characterize pretrained quantization robustness. First, the error a layer newly introduces tends to oppose the error it inherits from the layer's input. The two cancel partially such that the discrepancy between full-precision and quantized passes grows slow
    
[^25]: 负自我蒸馏：通过规避缺陷来学习推理

    Negative Self-Distillation: Learning to Reason by Avoiding Flaws

    [https://arxiv.org/abs/2609.11699](https://arxiv.org/abs/2609.11699)

    该论文提出负自我蒸馏（NSD）新框架，通过让模型远离自身生成的有缺陷推理、而非模仿特权信息下的“自信”解答，克服了传统在线策略自蒸馏（OPSD）抑制不确定性表达和探索性自我纠错行为、从而损害复杂推理能力的缺陷。

    

    arXiv:2609.11699v1 公告类型：新论文

    arXiv:2609.11699v1 Announce Type: new  Abstract: On-Policy Self-Distillation (OPSD) has emerged as a popular paradigm for large language model (LLM) self-improvement, allowing models to act as their own teachers by leveraging privileged information such as ground-truth solutions. However, recent findings indicate that OPSD can severely degrade the performance of LLMs on complex reasoning tasks: By forcing the student to imitate an artificially confident reasoning trace conditioned on privileged information, OPSD inadvertently suppresses expressions of uncertainty and penalizes the exploratory, self-corrective behaviors required to solve challenging problems. To address this, we introduce Negative Self-Distillation (NSD), a new framework that optimizes LLMs by diverging from flawed reasoning rather than imitating privileged solutions. Instead of relying on ground-truth answers or external supervision, NSD uses the model itself to generate a question-specific negative condition (eg, acti
    
[^26]: 面向语言模型低开销量化的结构化变换

    Structured Transforms for Low-Overhead Quantization of Language Models

    [https://arxiv.org/abs/2609.11687](https://arxiv.org/abs/2609.11687)

    该论文改进了基于Kashin分解的大语言模型权重量化方法，用符号随机化DCT替代稠密随机正交矩阵将每次迭代成本从O(N²)降至O(N log N)，并通过保证四峰分布和闭式聚类中心初始化的贪心交替算法消除了多重启k-means瓶颈，实现了低开销的2比特量化。

    

    我们重新审视了基于Kashin分解的大语言模型权重量化方法，提出了一种具有更强收敛特性和结构化高效正交变换的改进算法。该方法保留了将每个权重分解为两个分量的核心因式分解——其中一个分量具有有界的无穷范数，另一个分量在经过正交变换后具有有界的无穷范数——但用符号随机化的离散余弦变换（DCT）取代了稠密随机正交矩阵，从而将每次迭代的计算成本从O(N²)降低到O(N log N)。所提出的具有交替更新机制的贪心算法保证了每个因子稳定进行2比特聚类所需的四峰分布，并且支持聚类中心的闭式初始化，消除了以往工作中多重启k-means带来的瓶颈。结合OPTQ式的顺序误差补偿与QuIP式的不相干预处理，所得方法实现了低开销的高质量低比特量化。

    arXiv:2609.11687v1 Announce Type: new  Abstract: We revisit Kashin-decomposition-based weight quantization for large language models and propose an improved algorithm with stronger convergence properties and structured, efficient orthogonal transforms. The method retains the core factorization of each weight into two components -- one with bounded infinity norm and the other with bounded infinity norm after an orthogonal transformation -- but replaces the dense random orthogonal matrix with a sign-randomized Discrete Cosine Transform (DCT), reducing the per-iteration cost from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log N)$. The proposed greedy algorithm with alternating updates guarantees the four-peak distribution required for stable 2-bit clustering of each factor and admits closed-form initialization of cluster centers, removing the multi-restart k-means bottleneck of prior work. Composed with OPTQ-style sequential error compensation and QuIP-style incoherence preprocessing, the resu
    
[^27]: 一种无需训练、无需对齐的企业情报分析方法：在SEC文件中的应用

    A Training-Free, Alignment-Free Approach to Corporate Intelligence: Application to SEC Filings

    [https://arxiv.org/abs/2609.11620](https://arxiv.org/abs/2609.11620)

    该论文提出一种基于确定性稀疏种子向量的无需训练、无需对齐的企业情报分析框架，可在普通CPU上实现对SEC文件的亚秒级比较、发行人指纹识别及主题句提取。

    

    高维稠密文本嵌入和大语言模型在金融信息披露分析中面临诸多实际障碍：上下文窗口限制、幻觉风险、高昂的计算成本，以及独立训练模型之间向量空间的任意旋转问题。我们提出了一种基于确定性稀疏种子向量的无需训练、无需对齐的企业情报框架。通过将词字符串哈希到固定的高维基中，所有文档和所有时间阶段在构建上即处于同一坐标系中，从而消除了对训练或对齐的任何需求。在句子上下文中累积这些种子向量可产生可线性组合的语料库特定语义签名，支持亚秒级文档比较、发行人指纹识别、跟踪发行人在不同文件之间词汇的变化，以及主题句提取——所有这些均可基于普通CPU硬件完成。我们在多年语料库上对该方法进行了演示验证。

    arXiv:2609.11620v1 Announce Type: new  Abstract: High-dimensional dense text embeddings and large language models face real obstacles in financial-disclosure analysis: context-window limits, hallucination risk, high computational cost, and the arbitrary rotation of vector spaces across independently trained models. We present a training-free, alignment-free framework for corporate intelligence built on deterministic sparse seed vectors. Hashing word strings into a fixed high-dimensional basis places all documents and all temporal epochs in a common coordinate system by construction, removing any need for training or alignment. Accumulating these seed vectors across sentence contexts yields corpus-specific semantic signatures that compose linearly, supporting sub-second document comparison, issuer fingerprinting, tracking of how an issuer's vocabulary shifts between filings, and thematic sentence extraction, all on ordinary CPU hardware. Demonstrating the approach on a multi-year corpus
    
[^28]: 低资源多语言文本转语音的复杂文本鲁棒性评估与故障诊断

    Complex-Text Robustness Evaluation and Failure Diagnosis for Low-Resource Multilingual Text-to-Speech

    [https://arxiv.org/abs/2609.11545](https://arxiv.org/abs/2609.11545)

    本文提出一个针对低资源多语言TTS的复杂文本鲁棒性诊断框架，从内容一致性、语言一致性和生成稳定性三个维度，评估系统在数字、日期、命名实体、语码转换等复杂文本输入下的失效情况，覆盖泰语、越南语、斯瓦希里语和印尼语四种语言。

    

    低资源多语言文本转语音（TTS）系统已扩展了语言覆盖范围，但其在复杂文本输入下的鲁棒性仍未得到充分诊断。现有评估主要使用常规测试句子，关注自然度、说话人相似度和内容一致性，而对于多语言TTS系统在处理具有挑战性的输入（如数字、日期、命名实体、长句、语码转换表达以及与标点相关的结构）时如何失效的洞察有限。本文提出了一种针对低资源多语言TTS的复杂文本鲁棒性诊断框架。我们从三个维度评估鲁棒性：内容一致性、语言一致性和生成稳定性。针对泰语、越南语、斯瓦希里语和印尼语设计了多语言鲁棒性测试方案，涵盖普通句子和多种类型的复杂文本输入。我们进一步引入了自动诊断指标……

    arXiv:2609.11545v1 Announce Type: new  Abstract: Low-resource multilingual text-to-speech (TTS) systems have expanded language coverage, but their robustness under complex text inputs remains insufficiently diagnosed. Existing evaluations mainly focus on naturalness, speaker similarity, and content consistency using regular test sentences, while providing limited insight into how multilingual TTS systems fail when handling challenging inputs such as numbers, dates, named entities, long sentences, code-switched expressions, and punctuation-related structures. This paper proposes a complex-text robustness diagnosis framework for low-resource multilingual TTS. We evaluate robustness from three dimensions: content consistency, language consistency, and generation stability. A multilingual robustness testing scheme is designed for Thai, Vietnamese, Swahili, and Indonesian, covering ordinary sentences and multiple types of complex text inputs. We further introduce automatic diagnostic metric
    
[^29]: 面向数据高效语言学习的结构先验

    Structural priors for data-efficient language learning

    [https://arxiv.org/abs/2609.11505](https://arxiv.org/abs/2609.11505)

    该研究发现先在音乐、概率文法和元胞自动机等符号数据上训练模型，可为后续语言建模带来更低损失和更小的权重偏移，但这种损失优势并不能稳定转化为更好的下游语言能力。

    

    高效的语言学习需要减少对大量数据和计算资源依赖的方法。我们研究了结构迁移：先在非语言数据上训练模型，从而诱导出对自然语言有用的先验。这种方法是面向多语言语言建模的一种权重初始化形式。我们通过下一个词预测损失、模型中的权重偏移以及下游语言基准来评估迁移效果。若干符号数据类型——尤其是音乐、概率文法和元胞自动机——相比随机初始化能产生更低的语言建模损失。这些收益与后续语言训练期间更小的权重偏移相吻合，表明结构迁移将模型定位在参数空间中更有利的区域。然而，更低的损失并不总是能一致地转化为更好的下游语言表现，而且从非语言数据迁移的效率低于附加迁移方法。

    arXiv:2609.11505v1 Announce Type: new  Abstract: Efficient language learning requires methods to reduce the reliance on large data and computational resources. We investigate structural transfer: First training models on non-language data to induce useful priors for natural language. This approach is a form of weight initialization for multilingual language modeling. We evaluate transfer via next-token-prediction loss, weight shifts in the model, and downstream linguistic benchmarks. Several symbolic data types - notably music, probabilistic grammars, and cellular automata - yield lower language-modeling loss than random initialization. These gains coincide with smaller weight shifts during subsequent language training, suggesting that structural transfer positions models in a more favorable region of the parameter space. However, a lower loss does not translate consistently into better downstream linguistic performance, and transfer from non-language data is less efficient than additi
    
[^30]: ReGround：将审稿意见定位到多模态证据

    ReGround: Grounding Reviewer Comments in Multimodal Evidence

    [https://arxiv.org/abs/2609.11460](https://arxiv.org/abs/2609.11460)

    提出大规模审稿意见定位数据集ReGround，将3,656篇论文中的10,267条审稿意见链接到16,274条证据，并发现证据类型推断是主要瓶颈、多模态证据能提供纯文本检索遗漏的补充信号。

    

    审稿意见通常与被审论文的特定部分相关联，但由于论文是长篇多模态文档，将这些意见定位到相应的潜在证据十分困难。现有的基准测试无法涵盖这一场景，且主要关注显式的信息查询类请求。我们提出了ReGround，这是一个用于审稿意见定位任务的大规模数据集，它将3,656篇论文原始匿名投稿中的10,267条审稿意见与16,274条证据相链接。我们的构建基于一个简单的观察：作者的回复中通常包含对投稿内容中用于回应审稿意见部分的显式引用，这为数据集提供了高精度的标注来源。我们将定位任务形式化为一个检索任务，并评估了多种检索方法。结果表明，对论文全部内容进行检索的效果较差，证据类型的推断是主要瓶颈，而多模态证据能够提供仅依靠文本会被遗漏的补充信号。

    arXiv:2609.11460v1 Announce Type: new  Abstract: Reviewer comments naturally relate to specific parts of the reviewed paper, yet grounding these comments to the underlying evidence is difficult due to long multimodal documents. Existing benchmarks do not capture this setting and largely focus on explicit, information-seeking queries. We introduce ReGround, a large-scale dataset for reviewer comment grounding that links 10,267 reviewer comments to 16,274 evidence in the original anonymous submission of 3,656 papers. We build on a simple observation: author rebuttals often include explicit references to content of the submission used to address reviewer comments, providing a high-precision annotation source. We cast grounding as a retrieval task and evaluate a wide range of retrieval methods. Results show that retrieval over the entire paper content performs poorly, evidence-type inference is a major bottleneck, and multimodal evidence provides complementary signals that text alone misse
    
[^31]: 跨语言临床标注投影作为受限文本生成：一项六语言研究

    Cross-Lingual Clinical Annotation Projection as Constrained Text Generation: A Six-Language Study

    [https://arxiv.org/abs/2609.11450](https://arxiv.org/abs/2609.11450)

    该研究提出将跨语言临床标注投影建模为受限文本生成任务，通过将实体标签直接插入不可修改的目标语言文本并进行确定性验证，在六种语言上实现了最强且最一致的临床标注迁移性能。

    

    背景：旨在确定跨语言临床标注投影是否可以被表述为一种保持原文、文档级的生成任务，从而为多语言临床语料库构建产生可验证的字符级标注，并表征其相对于基于候选投影流水线的鲁棒性和计算权衡。方法：我们开发了一种受限的大语言模型投影工作流，将实体标签直接插入不可修改的目标语言文本中，随后进行确定性验证和字符偏移重建。我们将其与有监督的候选跨度投影方法以及机器学习与大语言模型混合精化方法一起评估，用于将西班牙语的疾病、症状和手术/操作标注迁移到六种语言中。评估采用MultiClinAI金标准，使用严格的跨度匹配和字符重叠F1指标。结果：直接大语言模型投影取得了最强且最一致的性能。GLM 5.2获得了平均（摘要在此处截断）

    arXiv:2609.11450v1 Announce Type: new  Abstract: Background: To determine whether cross-lingual clinical annotation projection can be formulated as a text-preserving, document-level generative task that produces verifiable character-level annotations for multilingual clinical corpus construction, and to characterize its robustness and computational trade-offs relative to candidate-based projection pipelines. Methods: We developed a constrained LLM projection workflow that inserts entity tags directly into immutable target-language text, followed by deterministic validation and character-offset reconstruction. We evaluated it alongside supervised candidate-span projection and hybrid ML-LLM refinement for transferring Spanish Disease, Symptom, and Procedure annotations into six languages. Evaluation used MultiClinAI gold standard with strict span matching and character-overlap F1 Results: Direct LLM projection achieved the strongest and most consistent performance. GLM 5.2 obtained a mea
    
[^32]: SWRouter：面向多轮大语言模型对话的相似性收缩窗口路由

    SWRouter: Similarity-Contractive Window Routing for Multi-Turn Large Language Model Conversations

    [https://arxiv.org/abs/2609.11414](https://arxiv.org/abs/2609.11414)

    提出SWRouter，通过基于相似性的上下文分割机制与双指标评估框架，解决多轮对话中大语言模型路由面临的上下文信息丢失混淆及评估指标耦合两大难题。

    

    大语言模型各具互补优势，这促使研究者开发路由方法，将每个查询分派给最合适的模型。尽管现有路由器在单轮设置中效果良好，但它们无法直接迁移到多轮对话场景——在多轮对话中，路由性能严重依赖于历史上下文如何被分割、保留并融入当前提示词。这带来了两个根本性挑战：一是在上下文构建过程中防止信息丢失与信息混淆，二是在评估路由质量时避免将模型选择与提示词构建质量混为一谈。在本文中，我们提出了SWRouter，一种面向多轮大语言模型路由的相似性收缩窗口路由器。SWRouter将基于相似性的上下文分割机制用于提示词构建，并结合双指标评估框架，将构建准确性与路由器性能解耦。在多个（数据集上的实验……摘要此处截断）

    arXiv:2609.11414v1 Announce Type: new  Abstract: Large language models exhibit complementary strengths, motivating routing methods that dispatch each query to the most suitable model. Although existing routers are effective in single-turn settings, they do not directly transfer to multi-turn dialogue, where routing performance critically depends on how historical context is segmented, retained, and incorporated into the current prompt. This introduces two fundamental challenges: preventing information loss and information confusion during context construction, and evaluating routing quality without conflating model selection with prompt construction quality. In this paper, we propose SWRouter, a Similarity-Contractive Window Router for multi-turn large language model routing. SWRouter combines a similarity-based context segmentation mechanism for prompt construction with a dual-metric evaluation framework that decouples construction accuracy from router performance. Experiments on mult
    
[^33]: TransClean：用于从大语言模型输出中检测和提取干净翻译的基准数据集

    TransClean: A Benchmark for Detecting and Extracting Clean Translations from Large Language Model Outputs

    [https://arxiv.org/abs/2609.11399](https://arxiv.org/abs/2609.11399)

    该论文提出了TransClean基准数据集，通过分析79万余个大语言模型翻译输出识别出12种翻译噪声模式，并评估了从带噪声的翻译输出中提取干净翻译的两种方法。

    

    大语言模型（LLM）越来越多地被用于机器翻译，但其输出往往包含翻译之外的额外文本，例如语言标签、解释说明或双语重复内容，我们将其称为“翻译噪声”。尽管这一问题普遍存在，此前却缺乏专门的基准数据集和系统性研究。我们分析了来自12个大语言模型在22个语言对上生成的超过79万个翻译输出，识别出12种反复出现的噪声模式，并将其归纳为格式噪声和内容噪声两类。基于观察到的模式，我们构建了TransClean基准数据集，包含9,900对带噪声与干净的翻译输出，其中8,800个为合成生成的实例，1,100个为人工整理的真实实例。我们在TransClean基准上评估了两种提取方法：1）基于跨度的提取方法，利用翻译质量估计模型进行跨度检测；2）基于大语言模型的提取方法。

    arXiv:2609.11399v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for machine translation, yet their outputs often contain additional text beyond the translation itself, such as language labels, explanations or bilingual repetitions, which we term translation noise. Despite its prevalence, this problem lacks dedicated benchmarks and systematic study. We analyze over 790,000 translation outputs from 12 LLMs across 22 language pairs (LPs) and identify 12 recurring noise patterns, which we group into formatting and content noise. Building on the observed patterns, we construct TransClean, a controlled benchmark of 9,900 pairs of noisy and clean translation outputs, comprising 8,800 synthetically generated instances and 1,100 manually curated authentic instances. We evaluate two extraction approaches on the TransClean benchmark: 1) a span-based extraction method leveraging translation quality estimation models for span detection, and 2) an LLM-based extrac
    
[^34]: VikingRAG：面向结构化文档的准确且令牌高效的检索增强生成

    VikingRAG: Accurate and Token-efficient Retrieval-augmented Generation over Structured Documents

    [https://arxiv.org/abs/2609.11390](https://arxiv.org/abs/2609.11390)

    VikingRAG通过将代理式多轮检索轨迹物化为可复用的经验边，并引入自适应升级策略在证据充分时采用单轮经验增强检索，在保持高准确率的同时大幅降低了结构化文档检索增强生成的令牌成本。

    

    最先进的检索增强生成（RAG）方法利用文档结构来获取充分的证据，但往往会产生高昂的令牌成本。为了在不损害高RAG准确性的前提下减少结构上下文令牌，我们提出了VikingRAG，一个目录感知的语义数据管理系统，它紧密集成语义与结构访问，以支持结构上下文高效、证据缺口驱动的多轮检索。为了进一步降低多轮交互的令牌开销，我们将代理式多轮检索轨迹物化为经验边，并对相似查询复用这些边，避免重复的多轮探索。此外，为了在无需代理式多轮检索时减少令牌成本，我们引入了一种自适应升级策略，当证据充分时从单轮经验增强检索中直接回答，仅在必要时才调用代理式多轮检索。

    arXiv:2609.11390v1 Announce Type: cross  Abstract: State-of-the-art retrieval-augmented generation (RAG) methods exploit document structures to acquire sufficient evidence, but often incur substantial token costs. To reduce structural-context tokens without compromising high RAG accuracy, we present {\sf VikingRAG}, a directory-aware semantic data management system that tightly integrates semantic and structural access to support structural-context-efficient, evidence-gap-driven multi-round retrieval. To further reduce token overhead of multi-round interaction, we materialize agentic multi-round retrieval traces as experience edges, and reuse these edges for similar queries, avoiding repeated multi-round exploration. To additionally reduce token costs when agentic multi-round retrieval is unnecessary, we introduce an adaptive escalation strategy that answers from one-round experience-augmented retrieval when the evidence is sufficient, and invokes agentic multi-round retrieval only oth
    
[^35]: SEAR：面向弱到强多语言语音多选题的片段证据感知路由

    SEAR: Segment-Evidence-Aware Routing for Weak-to-Strong Multilingual Speech MCQ

    [https://arxiv.org/abs/2609.11355](https://arxiv.org/abs/2609.11355)

    该工作提出SEAR系统，通过片段证据感知的多选题数据合成流水线与“弱到强”两阶段训练——先对可仅凭文本回答的题目做监督微调、再对依赖音频的题目做GSPO强化学习——来适配Qwen3-Omni，从而提升多语言语音多选题能力。

    

    本文介绍了我们参加第二届多语言会话语音语言模型挑战赛（MLC-SLM）任务2的系统。我们采用片段证据感知的数据与后训练流水线对Qwen3-Omni-30B-A3B-Instruct进行适配。首先由一个语言模型将带时间戳的ASR结果转换为连贯的事件片段，通过边界余量扩展后从原始录音中裁剪出来。随后，我们使用Qwen3.6-27B合成互补的语义类多选题，使用Gemini 3.1 Flash-Lite合成声学类多选题，并进行结构、依据、答案一致性以及目标模型可训练性等多重检查，最终生成覆盖21种语言与口音变体的359,825道经核验的多选题。接着，一个纯文本探针将数据划分为“弱”（可仅凭文本回答）与“强”（依赖音频）两类题目，分别用于监督微调和基于组序列策略优化（GSPO）的强化学习，并通过去偏优势、序列级重要性校正以及动态（摘要在此处截断）……

    arXiv:2609.11355v1 Announce Type: new  Abstract: This paper describes our system for Task~2 of the second Multilingual Conversational Speech Language Model (MLC-SLM) Challenge. We adapt Qwen3-Omni-30B-A3B-Instruct with a segment-evidence-aware data and post-training pipeline. A language model converts timestamped ASR into coherent event spans, which are expanded by a boundary margin and cropped from the original recording. We then synthesize complementary semantic MCQs with Qwen3.6-27B and acoustic MCQs with Gemini~3.1 Flash-Lite, followed by structural, grounding, answer-consistency, and target-model trainability checks, yielding 359,825 verified MCQs across 21 language and accent variants. A text-only probe partitions the data into weak, text-answerable items used for supervised fine-tuning and strong, audio-dependent items used for reinforcement learning with Group Sequence Policy Optimization (GSPO), stabilized by debiased advantages, sequence-level importance correction, and dynam
    
[^36]: 论匿名化对大语言模型性能的影响

    On the Impact of Anonymization on the Performance of Large Language Models

    [https://arxiv.org/abs/2609.11335](https://arxiv.org/abs/2609.11335)

    本文系统评估了五个大语言模型在十一个基准上处理原始与匿名化输入的性能差异，发现匿名化总体上会降低性能，且影响因模型能力与任务类型而异——能力最强的模型性能下降最大，TruthfulQA性能反而提升，而检索类任务则遭遇灾难性下降。

    

    随着大语言模型越来越多地被部署在敏感领域，对输入数据进行匿名化以保护个人可识别信息已成为一项关键做法。然而，这种匿名化对模型效用的影响尚未得到充分理解。本文对隐私与性能之间的权衡进行了系统性的实证研究。我们在十一个多样化的基准上评估了五个知名语言模型，比较它们在原始输入与假名化输入上的性能表现。我们的结果揭示，虽然匿名化通常会降低性能，但其影响非常微妙。我们发现能力更强的模型，如Qwen2.5-72B和GPT-4o mini，遭受的性能下降最大，这表明它们对特定实体信息的依赖更强。这种影响还与任务相关：TruthfulQA上的性能在匿名化后反而有所提升，而以检索为重点的任务（如RGB）则经历了灾难性的下降。

    arXiv:2609.11335v1 Announce Type: new  Abstract: As large language models are increasingly deployed in sensitive domains, anonymizing input data to protect personally identifiable information has become a critical practice. However, the impact of this anonymization on model utility is not well understood. This paper presents a systematic empirical study of the trade-off between privacy and performance. We evaluate five prominent language models across eleven diverse benchmarks, comparing their performance on original versus pseudonymized inputs. Our results reveal that while anonymization generally degrades performance, the effect is highly nuanced. We find that more capable models, such as Qwen2.5-72B and GPT-4o mini, suffer the largest performance drops, suggesting a stronger reliance on specific entity information. The impact is also task-dependent: performance on TruthfulQA improves with anonymization, while retrieval-focused tasks like RGB experience a catastrophic decline. Furthe
    
[^37]: E-CONAN（蕴含、矛盾与中立）基准测试：阿拉伯语文本蕴含与自然推理数据集

    E-CONAN (Entailment, CONtradition And Neutral) Benchmarks: Arabic Textual Entailment and Natural Inference Datasets

    [https://arxiv.org/abs/2609.11334](https://arxiv.org/abs/2609.11334)

    本文提出了面向阿拉伯语文本蕴含与自然推理的E-CONAN基准数据集，其句子对来源于自动翻译、人工验证翻译、手工编写及含谣言的新闻标题等多种渠道，并利用该基准对9个最先进的多语言预训练模型进行了零样本分类评估。

    

    自然语言推理（NLI）通过处理句子对来提取其语义关系。NLI一直是一个热门研究话题，并作为核心组件集成到其他自然语言处理应用中。尽管全球各种语言的文本推理研究取得了显著进展，但阿拉伯语在该领域仍然面临资源匮乏的问题。为了填补这一空白，本文提出了E-CONAN基准测试，它由来自多种来源的句子对组成：(1) 自动翻译的句子对，(2) 经人工验证的机器翻译句子对，(3) 来自对外阿拉伯语教学书籍的手工编写的句子对，以及 (4) 来自不同新闻频道且包含谣言的新闻标题句子对。E-CONAN包含两个基准数据集：E-CONAN-2，一个二分类数据集（RTE）；以及E-CONAN-3，一个三分类数据集（NLI）。此外，我们还使用E-CONAN基准测试评估了9个最先进的多语言预训练模型的零样本分类性能。

    arXiv:2609.11334v1 Announce Type: new  Abstract: Natural Language Inference processes pairs of sentences to extract their semantic relations. NLI has been a hot research topic, integrated as a main component in other NLP applications. Despite significant advancements in textual inference across various languages all around the world, Arabic language still suffers from limited resources in this domain. To address this gap, this paper introduces E-CONAN benchmarks that are composed of sentences pairs from various sources: (1) automatically-translated pairs, (2) human-validated machine-translated pairs, (3) hand-crafted pairs from teaching Arabic as foreign language books, and (4) headlines pairs from different news channels containing rumors. E-CONAN contains two benchmark datasets, E-CONAN-2, a 2-way dataset (RTE) and E-CONAN-3, a 3-way dataset (NLI). Additionally, we have used E-CONAN benchmarks to evaluate 9 state-of-the-art multilingual pretrained models using zero-shot classificatio
    
[^38]: 语义提升算子与不可判定类在保持性下的闭包性

    The Semantic Elevation Operator and the Closure of the Undecidable Class under Preservation

    [https://arxiv.org/abs/2609.11326](https://arxiv.org/abs/2609.11326)

    该论文提出语义提升算子 ΛΦ，将程序静态语义性质问题转化为自修改后的保持性问题，并基于克林递归定理证明不可验证性质类在该算子下封闭，且无界迭代将攀升至算术层级的 Π₂-完备性。

    

    程序静态语义性质的不可判定性由莱斯定理所支配。然而，自修改系统需要分析的并非某个性质当前是否成立，而是当系统重写自身时该性质是否被保持。我们通过语义提升算子 ΛΦ 将这一转变形式化，它把静态问题“x 是否满足 P？”转化为动态问题“x 经过 Φ 变换后 P 是否被保持？”。我们证明：当 Φ 是内涵性的（即依赖于源代码本身，而不仅仅是所计算的函数）时，即使提升后的性质破坏了莱斯定理所要求的外延性，它仍然是不可判定的；该证明基于克林递归定理，而非莱斯定理。因此，不可验证性质类 U 在提升算子下是封闭的。对该算子的无界迭代将沿算术层级攀升至 Π₂-完备性，进一步巩固了不可验证性。

    arXiv:2609.11326v1 Announce Type: cross  Abstract: The undecidability of a program's static semantic properties is governed by Rice's theorem. Self-modifying systems, however, require analysing not whether a property holds now, but whether it is preserved when the system rewrites itself. We formalise this transition through a semantic elevation operator {\Lambda}{\Phi}, which turns the static question "does x satisfy P?" into the dynamic question "is P preserved after x is transformed by {\Phi}?". We prove that when {\Phi} is intensional (depending on the source code, not only on the computed function), the elevated property remains undecidable even though it breaks the extensionality that Rice's theorem requires; the proof rests on Kleene's recursion theorem, not on Rice. Consequently the class U of non-verifiable properties is closed under the elevation operator. Unbounded iteration of the operator climbs the arithmetical hierarchy -to {\Pi}02-completeness- consolidating non-verifiab
    
[^39]: MultiHuSE：一个面向幽默风格与情绪的多模态数据集

    MultiHuSE: A Multimodal Dataset for Humour Styles and Emotions

    [https://arxiv.org/abs/2609.11322](https://arxiv.org/abs/2609.11322)

    本文推出了首个涵盖四种心理学幽默风格的多模态数据集MultiHuSE，通过50名演员对相同文本的多样化表演捕捉幽默表达的多变性，并证明多模态融合在幽默风格分类上优于单模态方法。

    

    言语幽默的计算识别仍然是一项具有挑战性的任务，需要对语言、表达方式、情绪和文化背景的理解。大多数现有方法专注于二分类任务，并且缺乏能够同时捕捉幽默的心理维度和表达变化的数据集。我们推出了MultiHuSE，这是一个多模态数据集，包含50名人口统计背景多样化的演员表演1,463个文本样本的2,407个高清视频，涵盖四种心理学幽默风格（亲和型、攻击型、自强型和自贬型）以及中性内容。其中一个子集还额外标注了潜在的情绪信息。该数据集独特地捕捉了不同演员对相同文本的多种诠释，从而能够对表达多样性进行系统性分析。基线实验表明，在幽默风格分类任务中，多模态融合方法优于单模态方法（准确率80.1%对比77.4%），特别……（摘要截断）

    arXiv:2609.11322v1 Announce Type: new  Abstract: Computational recognition of verbal humour remains a challenging task, requiring an understanding of language, delivery style, emotions, and cultural context. Most existing approaches focus on binary classification and lack datasets that capture psychological dimensions of humour alongside variations in expression. We introduce MultiHuSE, a multimodal dataset comprising 2,407 high-definition videos of 50 demographically diverse actors performing 1,463 text samples across four psychological humour styles (affiliative, aggressive, self-enhancing, and self-deprecating), as well as neutral content. A subset is additionally annotated for underlying emotions. The dataset uniquely captures multiple actor interpretations of the same texts, enabling systematic analysis of expressive diversity. Baseline experiments show that multimodal fusion outperforms unimodal approaches (80.1% vs. 77.4% accuracy) in humour style classification, with particular
    
[^40]: 希腊语歌曲的自动歌词转录：Whisper适配中的规模扩展与任务组合效应

    Automatic Lyric Transcription for Greek Songs: Scaling and Task Composition Effects in Whisper Adaptation

    [https://arxiv.org/abs/2609.11302](https://arxiv.org/abs/2609.11302)

    本文首次系统研究了Whisper在希腊语自动歌词转录任务上的适配，发现模型规模扩展可持续提升性能、多任务学习对小模型有正则化作用，且两阶段语音到歌唱适配使Whisper Large-v3的词错误率降至27.2%，同时构建了首个希腊语歌唱段级对齐数据集。

    

    自动歌词转录（ALT）由于旋律变化、节奏不规则以及伴奏干扰等因素，仍然比语音识别更具挑战性。在像希腊语这样的低资源语言中，这一挑战更加突出，因为此前不存在希腊语ALT的基准数据集。我们首次对Whisper在希腊语ALT上的适配进行了控制实验研究，考察了模型规模扩展效应、通过转录-翻译比例的多任务训练实现的任务组合，以及两阶段语音到歌唱的适配方法。我们还基于希腊语音频数据集（GAD），利用源分离和CTC强制对齐技术，整理了一个段级对齐的歌唱数据集。结果表明，模型扩展能持续提升性能，而多任务学习主要对较小容量的模型起到有益的正则化作用。在Whisper Large-v3上进行两阶段适配后，词错误率（WER）达到27.2%，相比零样本基线有显著提升。

    arXiv:2609.11302v1 Announce Type: new  Abstract: Automatic Lyric Transcription (ALT) remains substantially more challenging than speech recognition due to melodic variability, rhythmic irregularity, and accompaniment interference. This is heightened in low-resource languages like Greek, where no prior benchmark for ALT exists. We present the first controlled study of Whisper adaptation for Greek ALT, investigating model scaling effects, task composition via multitask training in transcribe-translate ratios, and two-stage speech-to-singing adaptation. We also curate a segment-level aligned singing dataset based on the Greek Audio Dataset (GAD) using source separation and CTC forced alignment. Results show that scaling consistently improves performance, while multitask learning acts as a beneficial regularizer primarily for smaller-capacity models. The 2-stage adaptation in Whisper Large-v3 achieves a Word Error Rate (WER) of 27.2%, a significant improvement over zero-shot baselines, est
    
[^41]: Xiaomi-CocktailASR-1 技术报告

    Xiaomi-CocktailASR-1 Technical Report

    [https://arxiv.org/abs/2609.11274](https://arxiv.org/abs/2609.11274)

    提出了基于LLM的端到端目标说话人语音识别架构Xiaomi-CocktailASR-1，以参考语音作为声纹提示，无需语音分离即可直接转录目标说话人的语音，在保持单说话人场景竞争力的同时具备目标说话人缺席时的拒绝识别能力。

    

    近来，基于大语言模型（LLM）的自动语音识别（ASR）模型取得了显著进展，但它们普遍缺乏对多说话人场景的支持，鸡尾酒会问题仍然是进一步推进ASR发展的关键瓶颈。现有的目标说话人ASR（TS-ASR）方法，包括使用说话人嵌入的端到端架构以及最新的基于LLM的探索，都存在单说话人性能下降以及无法在目标说话人不在场时进行拒绝识别的问题。本文提出了Xiaomi-CocktailASR-1，一种基于LLM的端到端TS-ASR架构。通过将参考语音用作声纹提示，它无需语音分离即可直接转录目标说话人的语音。Xiaomi-CocktailASR-1在单说话人场景中保持了有竞争力的性能，可与主流ASR模型相媲美。它还具有负样本拒绝能力，当目标说话人不在混合语音中时输出空文本。

    arXiv:2609.11274v1 Announce Type: cross  Abstract: Recently, large language model (LLM) based ASR models have achieved significant progress, yet they generally lack support for multi-speaker scenarios, where the cocktail party problem remains a critical bottleneck for further advancing ASR. Existing TS-ASR methods, including end-to-end architectures with speaker embeddings and latest LLM-based explorations suffer from degraded single-speaker performance and the inability to reject when the target speaker is absent. In this paper, we propose Xiaomi-CocktailASR-1, an LLM-based end-to-end TS-ASR architecture. By utilizing reference speech as voiceprint prompts, it directly transcribes the target speaker's speech without requiring speech separation. Xiaomi-CocktailASR-1 maintains competitive performance in single-speaker scenarios, comparable to mainstream ASR models. It also features a negative sample rejection capability, outputting empty text when the target speaker is absent from the m
    
[^42]: INDRA：一个用于探索烟草、化石燃料和化学工业档案的新型AI工具

    INDRA: A New AI Tool for Exploring Tobacco, Fossil Fuel, and Chemical Industry Archives

    [https://arxiv.org/abs/2609.11261](https://arxiv.org/abs/2609.11261)

    INDRA是一个AI研究平台，它将档案史学的规范嵌入系统级协议，并整合了多个此前孤立的历史行业档案库，使大语言模型能够可靠地探索数亿页烟草、化石燃料和化工行业的前机密文件，同时避免幻觉问题。

    

    五十年的诉讼使烟草行业数亿页曾经保密的商业记录得以披露，同时还披露了来自药品、化学品、食品、枪支和化石燃料制造商的文件。然而，这些档案对于通用大语言模型（LLM）来说实际上是无法访问的，因为它们从未被编译成LLM可读的语料库。聊天机器人可能熟悉这些档案中包含的部分材料，但由于无法直接访问这些文档，它们容易产生幻觉和其他缺陷。本文介绍了INDRA，这是一个旨在通过将档案史学的规范嵌入到管理每个输出的系统级协议中，从而纠正此类缺陷的研究平台。该平台联合了加州大学旧金山分校的行业文献库、哥伦比亚大学和纽约城市大学的ToxicDocs、斯坦福大学的SRITA以及其他此前相互孤立存储的文献集，并提供了三个相互关联的保障措施。

    arXiv:2609.11261v1 Announce Type: cross  Abstract: Five decades of litigation have disgorged hundreds of millions of pages of formerly secret business records from the tobacco industry, along with documents from the makers of drugs, chemicals, food, firearms, and fossil fuels. Yet these archives have been effectively inaccessible to general-purpose large language models (LLMs) because they have never been compiled into an LLM-readable corpus. Chatbots may be familiar with some of the materials contained in such archives but, with no direct access to the documents, they are vulnerable to hallucination and other defects. Here we introduce INDRA, a research platform designed to remedy such failures by embedding the conventions of archival historiography into a system-level protocol governing every output. The platform federates UCSF's Industry Documents Library, Columbia and CUNY's ToxicDocs, Stanford's SRITA, and other heretofore siloed collections, and provides three interlinked safegua
    
[^43]: MUtE：一个用于概念擦除与反事实干预的对偶框架

    MUtE: A Dual Framework for Concept Erasure and Counterfactual Interventions

    [https://arxiv.org/abs/2609.11253](https://arxiv.org/abs/2609.11253)

    本文提出基于概念擦除理论最优边界推导的对偶框架MUtE，其擦除函数可自然诱导确定性反事实映射，并通过在反事实轨迹上施加平移偏置，实现了概念擦除与反事实生成之间的无缝切换，有效提升下游算法公平性。

    

    从表示中擦除特定概念的信息已被证明有助于缓解偏见或解释模型决策。其联合目标是对原始表示进行变换，使目标概念变得不可预测，同时最大限度地保留与概念无关的信息。在这项工作中，我们重新审视了概念擦除的最优边界，推导出一类新颖的擦除函数，该函数可自然地诱导出一个确定性的对偶反事实映射。为弥合理论最优性与实际表示学习之间的差距，我们设计了一种实现方法，在反事实轨迹上施加平移偏置——这一约束与现代语言模型中许多概念在几何上的呈现方式相契合。我们的框架能够在概念擦除与反事实生成之间无缝切换。我们通过实验证明了该方法在改善下游算法公平性方面的有效性。

    arXiv:2609.11253v1 Announce Type: cross  Abstract: Erasing concept-specific information from representations has been proven useful for mitigating bias or interpreting model decisions. The joint objective is to transform the original representations such that the target concept becomes unpredictable, while maximally preserving concept-unrelated information. In this work, we revisit the optimal bounds of concept erasure to derive a novel class of erasure functions that naturally induce a deterministic, dual counterfactual mapping. Bridging the gap between theoretical optimality and practical representation learning, we design an implementation that imposes a translational bias on counterfactual trajectories - a constraint that aligns with how many concepts geometrically manifest in modern language models. Our framework enables seamless navigation between concept erasure and counterfactual generation. We empirically demonstrate its efficacy in improving downstream algorithmic fairness an
    
[^44]: 平衡多模态情感分析的错觉：超越基于优化方法的局限

    The Illusion of Balanced Multimodal Sentiment Analysis: Beyond the Limits of Optimization-Based Methods

    [https://arxiv.org/abs/2609.11247](https://arxiv.org/abs/2609.11247)

    该论文通过统一评估框架和理论诊断揭示了基于优化的多模态平衡方法的根本缺陷——损失不等于效用、梯度不等于重要性，实验证明这些方法无法可靠超越简单基线，并提出基于保留数据性能的模态效用估计新研究议程。

    

    多模态情感分析（MSA）仍然受模态不平衡问题的制约，然而该领域仍在持续依赖那些承诺多于实际效果的基于优化的平衡方法。我们提供了三项贡献：1）一个统一评估框架，在受控设置下测试基于梯度和损失的平衡策略；2）一个理论诊断，解释了这些方法为何失败——因为它们将拟合速度与判别贡献混为一谈；3）一个朝向基于保留数据性能的判别性模态估值的研究议程。在CMU-MOSI和CMU-MOSEI数据集上的实验揭示了三个缺陷：没有任何策略能够可靠地超越后期拼接（Late Concatenation）基线；性能对超参数高度敏感；即使进行比例校准也无法带来一致的收益。核心问题是根本性的：损失不等于效用，梯度不等于重要性。模态不平衡问题仍未解决，这促使我们转向从保留数据的性能中进行效用估计。

    arXiv:2609.11247v1 Announce Type: new  Abstract: Multimodal Sentiment Analysis (MSA) remains constrained by modality imbalance, yet the field continues to rely on optimization-based balancing methods that promise more than they deliver. We provide three contributions: 1) a unified evaluation framework testing gradient and loss-based balancing strategies under controlled settings; 2) a theoretical diagnosis explaining why these methods fail, as they conflate fitting speed with discriminative contribution; and 3) a research agenda toward held-out discriminative modality valuation. Experiments on CMU-MOSI and CMU-MOSEI reveal three shortcomings: no strategy reliably outperforms Late Concatenation; performance is sensitive to hyperparameters; and even ratio calibration fails to yield consistent gains. The core issue is fundamental: loss is not utility, and gradients are not importance. Modality imbalance remains unresolved, motivating utility estimation from held-out performance.
    
[^45]: 评估低资源语言公共语音资源的可复用性：以中库尔德语为例的案例研究

    Assessing the Reusability of Public Speech Resources for Low-Resource Languages: A Central Kurdish Case Study

    [https://arxiv.org/abs/2609.11246](https://arxiv.org/abs/2609.11246)

    本文通过对公开的中库尔德语语音资源（三个语音、35小时录音）与其论文描述的一致性审查，发现数据中存在设备记录错误、测试数据未标注、代码缺陷等问题，并指出仅由三人朗读文本构建的语音资源难以覆盖库尔德语的地区与书面多样性，提醒低资源语言公开数据集复用需谨慎验证。

    

    库尔德语有数百万人使用，但很少有技术能够将其朗读出来。最近的一项研究发布了三个库尔德语语音、35小时的录制语音以及一篇描述这项工作的论文，所有内容均可免费下载。本综述检查了这些公开文件与论文描述的匹配程度。该研究对自身局限性描述得很谨慎，但文件中存在几个问题：设置文件列出了从未使用过的设备，测试录音未加标签地混在训练数据中，以及一个代码缺陷对长数字的处理不当。下载页面还声称了比论文所报告更强的结果，并推荐其中一种语音供一般用途使用。这一推荐值得重视，因为库尔德语存在显著的地区和书面差异，而这些语音是由三个人朗读准备好的文本构建的。因此，这一过程去除了大量日常用语和地区性语音。英语和德语受益于长期的词典编纂和语言描写传统，这有助于……（原文摘要在此处截断）

    arXiv:2609.11246v1 Announce Type: new  Abstract: Kurdish is spoken by millions of people, but little technology can read it aloud. A recent study released three Kurdish voices, 35 hours of recorded speech, and a paper describing the work, all free to download. This review checks how well those public files match the paper. The research is careful about its limits, but the files contain several problems: a settings file lists equipment that was never used, test recordings are left unlabeled among training data, and a coding fault mishandles long numbers. The download page also claims a stronger result than the paper reports and recommends one voice for general use. That recommendation matters because Kurdish has major regional and written variation, while these voices were built from three people reading prepared texts. The process therefore removes much everyday and regional speech. English and German benefit from long traditions of dictionaries and linguistic description that help ide
    
[^46]: OmniHallu：面向多模态大语言模型跨模态理解与生成的统一幻觉检测框架

    OmniHallu: Unified Hallucination Detection for Cross-Modal Comprehension and Generation in Multimodal Large Language Models

    [https://arxiv.org/abs/2609.11244](https://arxiv.org/abs/2609.11244)

    本文提出统一幻觉检测框架OmniHallu，构建了覆盖图像、视频、音频三种模态下六种理解与生成任务的万级样本基准OmniHallu-Bench，并采用多智能体架构将输出分解为原子声明进行模态专家验证与结构化推理，实现跨模态幻觉的统一检测。

    

    尽管多模态大语言模型（MLLMs）在各类任务中取得了显著进展，但它们仍然受到幻觉问题的困扰，即生成的输出与输入语义相矛盾或歪曲了输入语义。现有研究通常仅在单一模态或任务类型内解决幻觉检测问题，限制了其泛化能力。我们提出了OmniHallu，一个统一的幻觉检测框架，涵盖图像、视频和音频模态的理解与生成任务。我们贡献了OmniHallu-Bench基准数据集，包含10,000个样本，并带有覆盖六种跨模态任务的声明级人工标注：图像到文本（I2T）、视频到文本（V2T）、音频到文本（A2T）、文本到图像（T2I）、文本到视频（T2V）以及文本到音频（T2A）。我们的多智能体架构将模型输出分解为原子声明，通过模态特定的专家进行验证，并通过结构化推理聚合证据。我们进一步提出了一种基于偏好优化的方法（摘要在此处截断）

    arXiv:2609.11244v1 Announce Type: new  Abstract: While Multimodal Large Language Models (MLLMs) have achieved remarkable progress across diverse tasks, they suffer from hallucinations where generated outputs contradict or misrepresent input semantics. Existing research typically addresses hallucination detection within a single modality or task type, limiting generalizability. We introduce OmniHallu, a unified hallucination detection framework spanning both comprehension and generation tasks across image, video, and audio modalities. We contribute OmniHallu-Bench, a 10,000-sample benchmark with claim-level human annotations covering six cross-modal tasks: image-to-text (I2T), video-to-text (V2T), audio-to-text (A2T), text-to-image (T2I), text-to-video (T2V), and text-to-audio (T2A). Our multi-agent architecture decomposes model outputs into atomic claims, verifies them through modality-specific experts, and aggregates evidence via structured reasoning. We further propose a preference-o
    
[^47]: 面向智能手术室的多智能体语音交互系统：架构设计与关键技术

    A Voice-Interactive Multi-Agent System for Smart Operating Rooms: Architecture Design and Key Technologies

    [https://arxiv.org/abs/2609.11231](https://arxiv.org/abs/2609.11231)

    本文提出基于大语言模型的智能手术室语音交互多智能体系统SurgicalRoomAgent，通过KV Cache前缀预热、流式JSON解析和渐进式技能提示披露三项关键技术大幅降低推理延迟，实现自然语言理解、设备控制、术中记录与手术报告生成。

    

    本文提出了SurgicalRoomAgent，一个基于大语言模型（LLM）的智能手术室多智能体语音交互系统。该系统通过分层架构实现自然语言理解、设备控制、术中记录和手术报告生成，该架构由语音交互流水线（唤醒、ASR、轮次检测、智能体推理、TTS）和智能体核心（技能注册表、任务规划器、设备管理器）组成。论文研究了三项关键技术：（1）面向低延迟推理的KV Cache前缀预热，通过字节级最长公共前缀重用，将重计算开销从约500毫秒降低至几十毫秒；（2）流式部分JSON解析与早期并行任务执行，将端到端延迟降低约30%；（3）渐进式技能提示披露，根据用户角色、已连接设备和手术阶段动态过滤系统提示，以最大化提示效率和安全性。

    arXiv:2609.11231v1 Announce Type: cross  Abstract: This paper presents SurgicalRoomAgent, a voice-interactive multi-agent system for smart operating rooms based on large language models (LLMs). The system achieves natural language understanding, device control, intraoperative recording, and surgical report generation through a layered architecture comprising a voice interaction pipeline (wake, ASR, turn detection, agent reasoning, TTS) and an agent core (skill registry, task planner, device manager). Three key technologies are investigated: (1) KV Cache prefix warming for low-latency inference, reducing recomputation overhead from approximately 500 ms to tens of milliseconds via byte-level Longest Common Prefix reuse; (2) streaming partial JSON parsing with early parallel task execution, reducing end-to-end latency by approximately 30%; and (3) progressive skill prompt disclosure, which dynamically filters system prompts based on user role, connected devices, and surgical phase to maxi
    
[^48]: REVA：面向上下文高效RAG服务的可复用证据视图聚合

    REVA: Reusable Evidence View Aggregation for Context-Efficient RAG Serving

    [https://arxiv.org/abs/2609.11209](https://arxiv.org/abs/2609.11209)

    REVA通过挖掘目标生成器的历史注意力轨迹，将历史查询-文档-模型交互聚合为可复用的证据视图分数库，从而在不依赖辅助模型和在线压缩的情况下实现上下文高效且低开销的RAG服务。

    

    检索增强生成（RAG）通过以检索到的文档为条件进行生成，提升了知识密集型大语言模型（LLM）应用的表现，但更长的上下文会增加延迟、键值（KV）缓存内存以及token成本。检索后压缩可以降低这一成本，然而现有的压缩器通常针对每个查询独立运行，依赖辅助模型或重写操作，并引入在线开销，这可能抵消较短提示词带来的收益。我们从数据挖掘的视角重新审视RAG压缩，将历史的查询-文档-模型交互聚合为可复用的证据视图。我们首先证明，现代压缩器相比简单的截断方法收益并不稳定，且可能带来显著的推理时延迟。随后我们提出可复用证据视图聚合（REVA），这是一个将目标生成器的历史注意力轨迹挖掘为以文档为键、与预算无关的分数存储库的框架。REVA将

    arXiv:2609.11209v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) improves knowledge-intensive large language model (LLM) applications by conditioning generation on retrieved documents, but longer contexts increase latency, key-value (KV) cache memory, and token cost. Post-retrieval compression can reduce this cost, yet existing compressors often operate independently for each query, rely on auxiliary models or rewriting, and introduce online overhead that can offset the benefit of shorter prompts. We revisit RAG compression from a data-mining perspective by aggregating historical query--document--model interactions into reusable evidence views. We first show that modern compressors have unstable gains over simple truncation and can add substantial inference-time latency. We then propose Reusable Evidence View Aggregation (REVA), a framework that mines the target generator's historical attention traces into a document-keyed, budget-agnostic score store. REVA maps 
    
[^49]: 社交媒体政治话语中竞争性叙事的自动识别

    Automated Identification of Competing Narratives in Political Discourse on Social Media

    [https://arxiv.org/abs/2609.11202](https://arxiv.org/abs/2609.11202)

    本文提出了一个无监督的多阶段自然语言处理框架，通过整合主题建模、事件检测和事件链接技术，自动识别德国政治家社交媒体推文中围绕热门政治话题的竞争性叙事及其不同视角。

    

    社交媒体平台已成为塑造政治话语的核心场所，是各种叙事形成和演变的竞技场，并深刻影响着公众舆论。识别和分析这些叙事，特别是当它们在不同政治意识形态之间相互竞争时，对于理解现代政治传播的动态至关重要。本文提出了一个无监督框架，用于识别和刻画社交媒体政治话语中的竞争性叙事，重点研究德国政治家的推文。该框架采用多阶段流程，整合了主题建模、事件检测和事件链接等自然语言处理技术。通过将数据组织成连贯的故事并揭示用户群体的独特视角，该系统能够检测出关键的竞争性叙事，突出围绕热门政治话题的不同叙事框架及其冲突。两个案例研究验证了该方法的有效性。

    arXiv:2609.11202v1 Announce Type: new  Abstract: Social media platforms have become central to shaping political discourse, serving as arenas where narratives form and evolve, influencing public opinion. Identifying and analyzing these narratives, particularly when they compete across different political ideologies, is crucial for understanding the dynamics of modern political communication. This paper presents an unsupervised framework for identifying and characterizing competing narratives in political discourse on social media, focusing on German politicians' tweets. The framework employs a multi-stage pipeline that integrates natural language processing techniques such as topic modeling, event detection, and event linking. By forming data into coherent stories and uncovering the distinct perspectives of user communities, the system is able to detect the key competing narratives, highlighting the divergent framings and conflicts surrounding trending political topics. Two case studie
    
[^50]: （谁的默认设置？）人工智能正在重塑考古学方法吗？

    (Whose defaults?) Is artificial intelligence reorienting archaeological methods?

    [https://arxiv.org/abs/2609.11198](https://arxiv.org/abs/2609.11198)

    该研究分析了约11.9万篇考古学摘要并运用贝叶斯狄利克雷-多项式模型，发现大语言模型兴起后（2023年后）考古学计算方法仅有微小转变，且方法多样性不降反升，表明AI并未窄化学科的研究方法范围。

    

    生成式人工智能和“氛围编程”实践正在改变考古学家开展计算研究的方式，但其对学科方法范围的影响仍未得到充分研究。本文评估了大型语言模型（LLM）是否正在缩小考古学家所使用方法的多样性。我们首先分析了来自Scopus的约119,000篇考古学摘要，涵盖2010年至2025年的出版物。利用本地运行的大语言模型，我们识别了每篇摘要中报告的计算方法，并将其归纳为25个宽泛类别（L2）和241个更细的聚类（L3）。基于子学科内方法构成的贝叶斯狄利克雷-多项式模型发现，2023年之后方法使用出现了小幅但可信的转变。然而，这一转变幅度小于整个研究期内已存在的变异。没有任何单一技术显示出显著变化，且整体方法多样性是增加而非减少。

    arXiv:2609.11198v1 Announce Type: cross  Abstract: Generative AI and the practice of "vibe coding" are changing how archaeologists carry out computational research, but their effects on the discipline's range of methods is still understudied. In this paper, we evaluate whether large language models (LLMs) are narrowing the variety of methods archaeologists use. We first analysed approximately 119,000 archaeology abstracts from Scopus, covering publications from 2010 to 2025. Using a locally run LLM, we identified the computational methods reported in each abstract and organised them into 25 broad categories (L2) and 241 finer clusters (L3). A Bayesian Dirichlet-multinomial model of method composition within sub-disciplines found a small but credible shift in method use after 2023. However, this shift was smaller than the variation already present across the full study period. No individual technique showed a significant change, and overall methodological diversity increased rather than
    
[^51]: FlexComp：上下文压缩中适用于任意压缩率的单一模型

    FlexComp: One Model for Every Ratio in Context Compression

    [https://arxiv.org/abs/2609.11192](https://arxiv.org/abs/2609.11192)

    FlexComp通过Matryoshka式训练使单一模型成为任意压缩率压缩器，并结合逐输入的预算选择机制（基于置信度的级联路由或轻量级K预测器），摆脱了传统方法中每个压缩率需单独训练模型且压缩率统一固定的限制。

    

    软上下文压缩将上下文浓缩为少量记忆token，供冻结的大语言模型（LLM）代替原始文本使用，但现有的压缩器在训练和推理阶段都固定了压缩率：每个部署的压缩率都需要单独训练一个模型，且所选压缩率被统一应用于所有输入，而不同输入的实际需求差异巨大。我们提出FlexComp，一种与方法无关的框架，将压缩率与训练和部署解耦：通过Matryoshka（俄罗斯套娃）式训练对每个实例采样记忆预算 $K$，使单一模型成为任意压缩率的压缩器；随后通过以下两种方式为每个输入选择预算：(1) 基于置信度的级联路由，或 (2) 轻量级的学习式 $K$ 预测器。在MRQA数据集上，跨ICAE、500xCompressor和SAC三种方法，单个FlexComp模型能以极小的性能损失媲美分别训练的固定压缩率专用模型。级联路由在高达266倍的平均压缩率下，仍保持了最温和压缩率98%以上的准确率。

    arXiv:2609.11192v1 Announce Type: new  Abstract: Soft context compression condenses a context into a few memory tokens that a frozen LLM consumes in place of the raw text, but existing compressors fix the compression ratio at training and inference: each deployed ratio requires a separately trained model, and the chosen ratio is applied uniformly to all inputs, whose actual needs vary drastically. We propose FlexComp, a method-agnostic framework that decouples the ratio from both training and deployment: Matryoshka-style training samples the memory budget $K$ per instance, turning one model into an any-ratio compressor, and the budget is then chosen per input by: (1) confidence-based cascade routing or (2) a lightweight learned $K$ predictor. Across ICAE, 500xCompressor, and SAC on MRQA, a single FlexComp model matches separately trained fixed-ratio specialists with minimal degradation. Cascade routing preserves over 98% of the mildest ratio's accuracy at up to 266x average compression
    
[^52]: LILA：基于潜在谱几何的大语言模型免校准结构化剪枝

    LILA: Calibration-Free Structured Pruning of Large Language Models via Latent Spectral Geometry

    [https://arxiv.org/abs/2609.11163](https://arxiv.org/abs/2609.11163)

    LILA提出了一种免校准的结构化剪枝方法，通过比较完整与神经元消融后FFN权重矩阵的奇异值分布的KS距离来评分神经元重要性，无需训练、校准数据或辅助网络即可在多个稀疏度水平上超越现有依赖校准的剪枝方法。

    

    大语言模型（LLM）的结构化剪枝能够实现硬件高效的压缩，然而现有方法在剪枝时需要校准数据、梯度计算或大型辅助策略网络。LILA（Latent-Informed Layer Analysis，潜在信息层分析）通过完整前馈网络（FFN）权重矩阵与神经元消融后权重矩阵的经验奇异值分布之间的Kolmogorov-Smirnov（KS）距离来评分神经元重要性，提供了一种无需训练、校准数据或辅助网络的闭式谱规则。在无需任何微调的情况下，LILA在25%稀疏度下于LLaMA-2-7B上的零样本准确率超越PruneNet（4500万参数的强化学习策略）1.57个百分点，并在所有稀疏度水平上比经WikiText-2校准的SliceGPT高出最多6.0个百分点，同时保持原始模型架构。经过一个epoch的LoRA恢复微调后，LILA实现了极具竞争力的性能，与经过大量校准的SliceGPT相当。

    arXiv:2609.11163v1 Announce Type: cross  Abstract: Structured pruning of large language models (LLMs) offers hardware-efficient compression, yet existing methods require calibration data, gradient computation, or large auxiliary policy networks at pruning time. LILA (\emph{Latent-Informed Layer Analysis}) scores neuron importance via the Kolmogorov--Smirnov (KS) distance between empirical singular value distributions of the full and neuron-ablated feed-forward network (FFN) weight matrix, providing a closed-form spectral rule requiring no training, calibration data, or auxiliary network. Without any fine-tuning, LILA surpasses PruneNet (45M-parameter RL policy) by 1.57~pp in zero-shot accuracy on LLaMA-2-7B at 25\% sparsity, and outperforms WikiText-2-calibrated SliceGPT by up to 6.0~pp across all sparsity levels, while preserving the original architecture. After one epoch of LoRA recovery fine-tuning, LILA achieves highly competitive performance, matching the heavily calibrated SliceG
    
[^53]: 递归语言模型训练的脆弱性谱系

    A Fragility Spectrum for Recursive Language-Model Training

    [https://arxiv.org/abs/2609.11149](https://arxiv.org/abs/2609.11149)

    该研究让13个公开模型在固定递归污染协议下共享语料库繁衍五代，发现不同模型对坍塌的脆弱性存在约五倍差异，且该脆弱性排序在不同数据组成和随机种子下高度稳定，表明易坍塌性是模型本身的固有属性。

    

    模型生成的文本正在回流到训练语料库中，大量证据表明，反复使用这类数据进行训练会导致输出多样性的坍塌。先前的工作研究了这一现象本身：哪些训练协议以及哪些数据混合方式会引发坍塌。但在相同的过程下，不同模型的表现差异巨大。我们固定了一种递归污染协议，让13个公开发布的模型检查点组成一个生态系统，共享同一语料库并繁衍五代。五代之后，各检查点的唯一4-gram指标从0.187到0.940不等，约有五倍的差距：一些模型几乎未受影响，另一些则退化为重复的片段。改变共享语料池的组成或混入人类文本时，模型排序的Spearman相关性保持在0.91–0.97；改变随机种子时，该相关性保持在0.93–0.98。因此，模型在递归训练下是否容易坍塌，是模型本身固有的一种属性。

    arXiv:2609.11149v1 Announce Type: new  Abstract: Model-generated text is finding its way back into training corpora, and there is plenty of evidence that training on such data over and over collapses output diversity. Prior work has studied the phenomenon itself: which protocols and which data mixtures cause collapse. But different models behave very differently under the same process. We fix one recursive contamination protocol and let 13 publicly released checkpoints form an ecosystem that shares a common corpus for five generations. The unique 4-gram outcome after five generations ranges from 0.187 to 0.940 across checkpoints, a roughly five-fold spread: some models are barely touched, others degenerate into repetitive fragments. Changing the composition of the shared pool or mixing in human text keeps the Spearman correlation of the ordering at 0.91--0.97, and changing the random seed keeps it at 0.93--0.98. Whether a model collapses easily under recursive training is, then, a prop
    
[^54]: 寡头在多模型生态系统中几乎无法左右模型崩溃

    The Oligarch Barely Steers Model Collapse in Multi-Model Ecosystems

    [https://arxiv.org/abs/2609.11146](https://arxiv.org/abs/2609.11146)

    在多模型递归训练的生态系统中，即使将寡头模型的市场份额推高至90%，既不会加速模型崩溃，也不会使其他模型被拖向寡头的输出分布——模型崩溃的动态对市场份额集中度表现出不变性。

    

    AI生成的文本正在回流到下一代模型的训练语料库中，对其实施的递归训练会导致模型崩溃。近期工作将这一研究场景扩展到多个模型相互喂食的情况——但几乎所有研究都假设市场份额均分，而现实中的生成式AI行业实为寡头垄断格局。市场集中度引发了两个担忧：更少、更同质化的来源可能加速模型崩溃，且后续的模型可能被拖向寡头模型的输出分布。我们在受控生态系统中对这两点进行了检验：13个开源的1—4B参数模型构成了包含3到13个参与者的自然生态系统，另加入一个注入的探针模型，将头部模型的市场份额推高至90%；在每一代中，所有模型的输出按各自市场份额混入共享数据池，每个模型都从干净的初始基础权重出发在该共享池上重新训练，共进行五代。然而在我们测试的范围内，这两个担忧均未成真；相反，实验呈现出一种不变性：使市场份额分配更加不均，几乎不会改变模型崩溃的速度。

    arXiv:2609.11146v1 Announce Type: cross  Abstract: AI-generated text is flowing back into the training corpora of the next generation of models. Recursive training on it drives model collapse, and recent work extends the setting to many models feeding one another -- but almost always with the market split evenly, while real generative AI is an oligopoly. Concentration raises two worries: fewer, more uniform sources may make collapse faster, and later models may be dragged toward the oligarch's output. We test both in controlled ecosystems: 13 open 1--4B models form natural ecosystems of 3 to 13 players, plus an injected probe that pushes the top share to 90%; each generation, every model's output is mixed into a shared pool by market share and every model is retrained on that pool from clean base weights, for five generations. Yet within the range we test, neither worry materializes; what emerges instead is an invariance. Making the split more unequal barely changes the speed of collap
    
[^55]: 同日同故事，隔日异信号：金融情感分析的双重效度

    Same Day, Same Story; One Day Ahead, a Different Signal: The Dual Validity of Financial Sentiment

    [https://arxiv.org/abs/2609.11144](https://arxiv.org/abs/2609.11144)

    本文基于2002-2025年证券集体诉讼语料库，将70,500条X消息与异常股票收益相关联，通过统一流程测试五种情感分析工具，发现金融情感工具的人工标注一致性（构念效度）与其市场预测能力（预测效度）之间的关系并非恒定，而是取决于抽样惯例和分数表示方式。

    

    金融自然语言处理（Financial NLP）领域有一个标准工作流程：先验证情感分析工具与人工标注的一致性，然后信任它来提取市场信号。这背后隐含的假设是，这两种评估衡量的是同一件事。我们在一个可以同时测量两者的场景中检验了这一假设：一个证券集体诉讼语料库（2002-2025年），将70,500条X平台消息与异常股票收益相关联，并包含一个由单一标注员人工标注的金标准样本。通过将五种工具（VADER、Loughran-McDonald、FinBERT、Twitter-RoBERTa和一个LLM标注器）运行在完全相同的流程中，我们发现构念效度与预测效度之间的关系取决于抽样惯例和分数表示方式。在传统的方法特定抽样下，人工一致性评分与同日的分级关联更为吻合，而与提前一天的关联吻合度较低。然而，在固定样本量的面板数据上，一致性评分在两个时间范围内都表现出相似的分级秩相关，而粗粒度排序在两种情况下均较弱。

    arXiv:2609.11144v1 Announce Type: cross  Abstract: Financial NLP has a standard workflow: validate a sentiment tool against human labels, then trust it to extract market signal. This assumes the two evaluations measure the same thing. We test that assumption in a setting where both can be measured at once: a corpus of securities class actions (2002-2025) linking 70,500 X messages to abnormal stock returns, with a single-annotator human labelled gold sample. Running five instruments (VADER, Loughran-McDonald, FinBERT, Twitter-RoBERTa, and an LLM annotator) through one identical pipeline, we find that the relationship between construct and predictive validity depends on the sampling convention and score representation. Under conventional method-specific sampling, human agreement aligns more closely with graded same-day associations than with one-day leads. On a fixed-n panel, however, agreement has similar graded rank correlations at both horizons, while the coarse ordering remains weak.
    
[^56]: 大语言模型能规范化数据库吗？面向模式规范化的基准测试与多智能体框架

    Can LLMs Normalize Databases? A Benchmark and Multi-Agent Framework for Schema Normalization

    [https://arxiv.org/abs/2609.11141](https://arxiv.org/abs/2609.11141)

    该论文提出了包含3,275个样本的数据库规范化基准DNBENCH，系统揭示了LLM在函数依赖推理、模式分解和表间约束重建中的常见失败，并进一步提出多智能体框架MARS以提升模式规范化的可靠性。

    

    大语言模型（LLM）越来越多地被用于生成结构化输出，但当这些输出必须满足数据库级别的约束时，其可靠性仍不明确。我们通过数据库规范化来研究这一问题，这涉及对函数依赖、无损连接分解以及表间约束的推理。我们提出了数据库规范化基准测试，包含3,275个样本，用于评估从1NF到BCNF的LLM驱动的数据库规范化能力。DNBENCH采用三轴评估协议，衡量语义等价性、结构准确性和逻辑有效性。在单一、复杂和真实世界三个难度层级上，DNBENCH揭示了LLM在依赖推理、模式分解和表间约束重建中反复出现的失败。我们进一步提出了面向模式的多智能体推理框架，它将证据提取、违规诊断和分解规划从模式生成中分离出来……

    arXiv:2609.11141v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used to generate structured outputs, but their reliability remains unclear when those outputs must satisfy database-level constraints. We study this issue through database normalization, involving reasoning about functional dependencies, lossless join decompositions, and inter-table constraints. We introduce a Database Normalization Benchmark (DNBENCH), comprising 3,275 samples for evaluating LLM-driven database normalization from 1NF to BCNF. DNBENCH uses a three-axis protocol to measure semantic equivalence, structural accuracy, and logical validity. Across Single, Complex, and Real World levels, DNBENCH uncovers recurring failures in dependency inference, schema decomposition, and inter-table constraint reconstruction. We further propose Multi-Agent Reasoning for Schemas (MARS), which separates evidence extraction, violation diagnosis, and decomposition planning from schema generation and 
    
[^57]: 基于量规对齐的人类同声传译解耦评估

    Rubric-Aligned Disentangled Evaluation of Human Simultaneous Interpreting

    [https://arxiv.org/abs/2609.11131](https://arxiv.org/abs/2609.11131)

    本文构建了首个带有多维量规专业评分的同传片段标注语料库，并提出基于LoRA适配COMET-KIWI编码器的双回归头模型，首次实现了与评估量规对齐的意义传递与表达质量解耦的自动同传评估。

    

    人类同声传译（SI）通常采用分析量规进行评估，将意义传递、表达质量和时间同步性分开考量，但目前尚无针对量规对齐的段级同传评估设计的自动指标。我们构建了一个包含1,101个同传片段的专业标注语料库，其中包含意义传递（LQ）、表达质量（EXP）和感知延迟（LAT）的评分。我们证明了结构化的大语言模型提示和标量监督会导致量规维度塌缩，与人类评分的相关性接近于零，且存在强烈的跨维度耦合。为了在相同骨干模型容量下隔离监督结构的作用，我们在LoRA适配的COMET-KIWI编码器上引入了双回归头。在留出的演讲级测试集上，该模型实现了0.388（LQ）和0.301（EXP）的皮尔逊相关系数，优于冻结的COMET-KIWI。鉴于评分者之间的绝对一致性较低，我们将结果相对于人类评分一致性水平进行解读并设定目标。

    arXiv:2609.11131v1 Announce Type: new  Abstract: Human simultaneous interpreting (SI) is commonly assessed with analytic rubrics separating meaning transfer, delivery quality, and temporal synchrony, yet no automatic metric is designed for rubric-aligned segment-level SI evaluation. We construct a professionally annotated corpus of 1,101 SI segments with scores for meaning transfer (LQ), delivery quality (EXP), and perceived latency (LAT). We show that structured LLM prompting and scalar supervision collapse rubric dimensions, yielding near-zero correlation with human ratings and strong cross-dimension coupling. To isolate supervision structure under identical backbone capacity, we introduce dual regression heads on a LoRA-adapted COMET-KIWI encoder. On a held-out talk-level test set, the model achieves Pearson correlations of 0.388 (LQ) and 0.301 (EXP), improving over frozen COMET-KIWI. Given low absolute rater agreement, we interpret results relative to human consistency and target s
    
[^58]: 从重复到识别：虚假信息叙事的归纳发现

    From Repetition to Recognition: Inductive Discovery of Disinformation Narratives

    [https://arxiv.org/abs/2609.11128](https://arxiv.org/abs/2609.11128)

    本文提出了包含恢复、挖掘和发现三个层次的无监督叙事标签生成评估框架，突破了传统封闭世界评估的局限，并发现基于聚类与基于图社区的流水线在虚假信息叙事发现中互为补充，但聚类方法可能严重压缩少数主题而图方法则保持均衡。

    

    在虚假信息数据集中，叙事通常被理解为反复出现的解释性模式，可将文本归入相应的叙事标签之下。近期研究将叙事挖掘形式化为从语料库中归纳推断叙事标签，但其评估仍局限于预定义的分类体系——这种封闭世界设定无法捕捉参考标签中缺失的叙事。我们提出了一个针对无监督叙事标签生成的三层评估框架：恢复（对照语料库自身的分类体系）、挖掘（对照外部标签集）和发现（无预定义标签）。应用该框架，我们在七个虚假信息数据集上比较了基于聚类和基于图社区的两种处理流水线，并在其中两个数据集上对发现任务进行了人工验证。在自动化指标下，这两类方法互为补充，但在一个包含两个突出主题的语料库中，聚类方法可能将其中一个主题压缩至生成标签的2%，而基于图社区的流水线则保持平衡（摘要在此处截断）。

    arXiv:2609.11128v1 Announce Type: new  Abstract: In disinformation datasets, narratives are often understood as recurring interpretive patterns that group texts under narrative labels. Recent work formalized narrative mining as inductively inferring narrative labels from corpora, but its evaluation stays tied to predefined taxonomies, a closed-world setting that cannot capture narratives absent from the reference labels. We introduce a three-tier evaluation framework for unsupervised narrative label generation: recovery (against a corpus's own taxonomy), mining (against external label sets), and discovery (without predefined labels). Applying it, we compare clustering-based and graph-community-based pipelines across seven disinformation datasets, with human validation of discovery on two. The two families are complementary under automated metrics, but in a corpus with two prominent topics, clustering can reduce one topic to 2% of generated labels while graph-based pipelines stay balanc
    
[^59]: KuaiRP系列角色扮演模型技术报告

    KuaiRP Series Role-playing Models Technical Report

    [https://arxiv.org/abs/2609.11127](https://arxiv.org/abs/2609.11127)

    KuaiRP系列角色扮演模型通过标准化角色模板、基于用户行为模拟的SFT数据流水线、规则复合奖励的强化学习以及多阶段训练流程，在注入深度领域知识的同时有效克服灾难性遗忘，实现了小参数规模下高质量、稳定且高效的角色扮演。

    

    本文介绍了KuaiRP系列角色扮演模型的完整技术方案。我们旨在为专用角色扮演模型实现四个核心目标：简化的提示词工程、高度稳定的输出质量、内置的领域世界知识，以及小参数规模下的高效部署。然而，有效注入深度领域知识往往会导致模型通用智能体能力的严重灾难性遗忘。为克服这一权衡问题，我们提出了一个多阶段训练流程。首先，我们设计了标准化的角色模板，并基于用户行为模拟和反向画像过滤构建了SFT数据流水线。其次，我们在强化学习（RL）阶段采用基于规则的复合奖励函数，以消除长度膨胀和重复生成等常见的退化现象。最后，为恢复在此过程中受损的通用能力……（摘要在此处截断）

    arXiv:2609.11127v1 Announce Type: cross  Abstract: This paper introduces the complete technical solution for the KuaiRP series of role-playing models. We aim to achieve four core objectives for a dedicated role-playing model: simplified prompt engineering, highly stable output quality, built-in domain world knowledge, and high-efficiency deployment with a small parameter size. However, effectively injecting deep domain knowledge often leads to a severe catastrophic forgetting of the model's general agent capabilities. To overcome this trade-off, we propose a multi-stage training pipeline. First, we design a standardized character template and construct an SFT data pipeline based on user behavior simulation and reverse profile filtering. Next, we utilize a rule-based composite reward function during the Reinforcement Learning (RL) phase to eliminate common degradation phenomena like length expansion and repetitive generation. Finally, to recover the general capabilities compromised duri
    
[^60]: NLPCC 2026 共享任务 11 概览：基于智能体的科学论文实验复现

    Overview of the NLPCC 2026 Shared Task 11: Agent-Based Experiment Reproduction from Scientific Papers

    [https://arxiv.org/abs/2609.11117](https://arxiv.org/abs/2609.11117)

    该论文提出 AgentActionBench，一个面向过程的基准，通过基于 MCP 的动作记录器和论文专属评分标准，评估大语言模型智能体在机器学习与 AI4Science 领域实验复现过程中的完整行为表现。

    

    可复现性对科学进步至关重要，然而科学出版物数量和复杂性的持续增长使得详尽的人工验证变得越来越不切实际。尽管大语言模型（LLM）智能体的最新进展使自动化实验复现成为可能，但现有的评估大多聚焦于最终的代码仓库，且通常局限于机器学习（ML）领域。我们提出了 AgentActionBench，这是一个面向过程的基准，用于评估智能体在机器学习和 AI4Science 领域的实验复现能力。我们的框架使用基于 MCP 的动作记录器捕获智能体在整个复现过程中的行为，并利用针对论文的评分标准对生成的行为轨迹进行评估。AgentActionBench 包含 150 篇论文，其中 120 篇为机器学习论文，30 篇为 AI4Science 论文。覆盖基准 10% 的人工标注子集提供了验证数据，而模型辅助增强则将完整基准扩展至更多……

    arXiv:2609.11117v1 Announce Type: new  Abstract: Reproducibility is essential to scientific progress, yet the growing volume and complexity of scientific publications make exhaustive manual verification increasingly impractical. Although recent advances in large language model (LLM) agents enable automated experiment reproduction, existing evaluations largely focus on final repositories and are typically limited to machine learning (ML). We introduce AgentActionBench, a process-oriented benchmark for evaluating agent-based experiment reproduction across ML and AI4Science domains. Our framework uses an MCP-based Action Recorder to capture agents' behaviour throughout the reproduction process and evaluates the resulting traces with paper-specific rubrics. AgentActionBench contains 150 papers, including 120 ML papers and 30 AI4Science papers. A human-annotated subset covering 10% of the benchmark provides validation data, while model-assisted augmentation expands the full benchmark to mor
    
[^61]: ProMediConv：法律纠纷调解中主动式对话智能体的基准测试

    ProMediConv: Benchmarking Proactive Conversational Agents in Legal Dispute Mediation

    [https://arxiv.org/abs/2609.11101](https://arxiv.org/abs/2609.11101)

    该论文提出ProMediConv基准框架，将法律纠纷调解建模为主动式、多阶段、感知当事人的对话过程，基于972个真实案例构建了高保真话语级标注数据集，并提出了捕捉对话中当事人行为模式变化的细粒度MAD评估指标。

    

    纠纷调解对于维护社会和谐与社会韧性至关重要，然而培养熟练的调解员既昂贵又耗时。现有的基于大语言模型（LLM）的调解研究仍然受限于不切实际的任务设定、低保真度数据集以及掩盖逐轮动态的粗糙评估指标。为弥补这些不足，我们提出了ProMediConv，这是一个新颖的基准测试框架，它将调解建模为一个主动式、多阶段、且感知当事人立场的对话过程，其中融合了11种调解策略和四种当事人行为模式（BP）状态。我们利用972个完整的真实案例，构建了一个高保真度的调解数据集，其中包含话语层面的策略和BP状态标注。此外，为了更好地评估智能体的影响，我们提出了MAD（平均属性差异），这是一种能够捕捉整个对话过程中BP变化的细粒度指标。利用这一框架，我们通过评估多种模型建立了一个全面的基准。

    arXiv:2609.11101v1 Announce Type: new  Abstract: Dispute mediation is essential for maintaining social harmony and resilience, yet developing skilled mediators is costly and time-consuming. Existing LLM-based mediation research remains limited by unrealistic task formulations, low-fidelity datasets, and coarse evaluation metrics that obscure turn-by-turn dynamics. To address these gaps, we introduce ProMediConv, a novel benchmarking framework that models mediation as a proactive, multi-stage, and party-aware dialogue process incorporating 11 mediation strategies and four party behavior pattern (BP) states. Using 972 complete real-world cases, we construct a high-fidelity mediation dataset with utterance-level annotations of strategies and BP states. Furthermore, to better assess agent impact, we propose MAD (Mean Attribute Difference), a fine-grained metric that captures BP shifts throughout the dialogue. Leveraging this framework, we establish a comprehensive benchmark by evaluating d
    
[^62]: 超越求解器判定：面向自动形式化的生成式奖励模型

    Beyond Solver Verdicts: Generative Reward Models for Autoformalization

    [https://arxiv.org/abs/2609.11085](https://arxiv.org/abs/2609.11085)

    本文发现了自动形式化中的“判定保持的不忠实性”（VPU）失败模式，从理论上证明仅依赖求解器判定的验证方法无法有效检测此类错误，并提出生成式验证方法（GenV），将Z3等价性预言机蒸馏为无参照的连续等价性评分以实现可靠的验证。

    

    神经符号系统依赖数学求解器来保证推理的正确性，然而求解器从根本上无法感知一个形式化翻译是否与指定的形式化保持严格的参照等价性。我们将这一脆弱性形式化为“判定保持的不忠实性”（Verdict-Preserving-Unfaithfulness, VPU）：这是一种失败模式，即错误的编码能够成功执行并匹配预期的判定结果。我们从理论上证明，结构化的、仅基于判定的验证启发式方法在检测这些具有欺骗性的有效轨迹时，其检测能力在数学上被限制在随机概率水平。为解决这一问题，我们引入了生成式验证（GenV），通过重新利用语言模型的原生词汇空间，将离线的Z3等价性预言机蒸馏为无参照的、连续的参照等价性评分。通过决策投影logit透镜和稀疏自编码器进行的机制分析表明，这种生成式读出能够原生地提取精确的空间误差信息……

    arXiv:2609.11085v1 Announce Type: cross  Abstract: Neurosymbolic systems rely on mathematical solvers to guarantee reasoning correctness, yet solvers are fundamentally blind to whether a formal translation maintains strict reference-equivalence to a designated formalization. We formalize this vulnerability as Verdict-Preserving-Unfaithfulness (VPU): a failure mode where an incorrect encoding executes successfully and matches the expected verdict. We theoretically prove that structural, verdict-only verification heuristics are mathematically bounded to chance-level detection on these deceptively valid traces. To resolve this, we introduce Generative Verification (GenV), which distills an offline Z3-equivalence oracle into a reference-free, continuous reference-equivalence score by repurposing the language model's native vocabulary space. Mechanistic analysis via decision-projected logit lenses and sparse autoencoders shows this generative readout natively extracts precise spatial error 
    
[^63]: 当噪声制造偏见：噪声文本下LLM作为评判者的偏见测量脆弱性

    When Noise Fabricates Bias: The Fragility of LLM-as-a-Judge Bias Measurement under Noisy Text

    [https://arxiv.org/abs/2609.11067](https://arxiv.org/abs/2609.11067)

    该论文发现文本表面噪声（如拼写错误）会严重扭曲LLM评判者的社会偏见测量，且这种扭曲极不对称——中性文本被误判为带有偏见的概率是被反向误判的120倍，揭示了基于LLM的偏见测量的脆弱性。

    

    大型语言模型越来越多地被用作评判者来衡量文本中的社会偏见，然而它们所评判的文本段落往往带有噪声，包含拼写错误、非正式拼写和残缺的标点符号。这种表面噪声对社会偏见测量的影响仍不清楚。为了探究这一问题，我们在多个强度水平下对3,822条与刻板印象相关的回复施加了五种真实的噪声条件，并将由此产生的偏见判断与原始文本上的判断进行比较。我们发现这种表面噪声并不会对称地降低偏见测量质量：它更倾向于将中性判断扭曲为偏见判断，而非将偏见判断恢复为中性判断，两者的差距高达120倍。我们进一步在四个LLM评判者中观察到两种非显而易见的效应：在最脆弱的评判者中，失真在轻微且贴近现实的噪声水平下最为纯粹，此时中性判断被擦除得最少；而随着评判者变得更加鲁棒，这种失真趋向于均衡而非反转。

    arXiv:2609.11067v1 Announce Type: new  Abstract: Large language models are increasingly used as judges to measure social bias in text, yet the passages they judge are often noisy, containing typos, informal spelling, and broken punctuation. The consequences of such surface noise for social bias measurement remain unclear. To investigate this question, we apply five realistic noise conditions at multiple intensity levels to 3,822 stereotype-related responses and compare the resulting bias judgments with those on the original text. We find that such surface noise does not degrade bias measurement symmetrically: it is far more likely to turn neutral judgments into biased ones than biased judgments into neutral ones, by up to a 120x margin. We further observe two non-obvious effects across four LLM judges: in the most fragile judge the distortion is at its purest at mild, realistic noise levels, where erasure is scarcest, and as judges grow robust it attenuates toward parity rather than re
    
[^64]: 大语言模型的信息几何是共享的、可学习的且可控的

    The information geometry of large language models is shared, learned, and controllable

    [https://arxiv.org/abs/2609.11063](https://arxiv.org/abs/2609.11063)

    该论文发现大语言模型下一个词元概率的Fisher-Rao几何在不同架构间是共享的、随训练可学习的，并与人类词汇选择高度一致且可被控制，为理解和修改模型行为提供了统一的结构化框架。

    

    大语言模型会学习到相似的行为，但它们究竟共享什么结构、以及如何在不干扰其他行为的情况下改变某一行为，目前仍不清楚。下一个词元概率的Fisher-Rao几何将这两个问题联系起来：行为在保持输出的对称性范围内决定这一几何，而激活几何则依赖于坐标选择。在Transformer、状态空间和循环模型中，输出几何的一致性强于激活几何，且共享几何支持语义类别的迁移。与人类词汇选择的一致性随预测准确率、模型规模和训练进程的增加而提高，并且在仅用模型自身进行校准后还会进一步提升。词元概率与读出几何共同预测了谱及其有效维度。受控的语言分配实验表明，几何在不同架构间遵循语言规律。预训练语料库的统计信息可以在无需重新校准的情况下预测模型对留出事实的获取。（注：原摘要在此处被截断）

    arXiv:2609.11063v1 Announce Type: cross  Abstract: Large language models learn similar behaviours, yet it remains unclear what structure they share or how to change one behaviour without disturbing others. The Fisher-Rao geometry of next-token probabilities connects these questions: behaviour determines this geometry up to output-preserving symmetries, whereas activation geometry depends on coordinates. Across transformer, state-space and recurrent models, output geometries agree more strongly than activation geometries, and shared geometry supports semantic-category transfer. Agreement with human word choices increases with predictive accuracy, scale and training, and improves further after model-only calibration. Token probabilities and read-out geometry jointly predict the spectrum and its effective dimension. Controlled language assignments show that geometry follows the language law across architectures. Pretraining corpus statistics predict held-out fact acquisition without recal
    
[^65]: 使用TF-IDF加权交叉熵损失重新平衡语言模型中的词元重要性

    Rebalancing Token Importance in Language Models with TF-IDF Weighted Cross-Entropy Loss

    [https://arxiv.org/abs/2609.11029](https://arxiv.org/abs/2609.11029)

    该论文提出用TF-IDF加权交叉熵损失替代统一词元权重，在五个大语言模型上显著减少记忆化现象（LoRA微调下平均减少14%，TinyLLaMA全权重微调下减少58%），同时保持模型性能且计算开销不足3%。

    

    大型语言模型通常在统一的词元权重下进行训练，这使得高频且低信息量的词元主导学习过程，并可能增加模型记忆表层文本片段的倾向。为了解决这一问题，我们提出了一种信息加权交叉熵损失，利用TF-IDF统计量对词元级别的贡献进行重新缩放，强调语义信息丰富的词元，同时降低普遍出现词元的权重。在五个参数规模从1.1B到13B的仅解码器大语言模型上的实验表明，该方法在保持困惑度和下游任务性能的同时，一致地减少了被记忆的子串长度。在LoRA微调下，TF-IDF使所有五个模型的平均子串记忆长度减少了14%；在TinyLLaMA 1.1B上进行全权重微调时，减少幅度达到58%。我们的方法与具体架构无关，可以以低于3%的计算开销融入现有的训练流程，为缓解语言模型记忆化问题提供了一个实用的方案。

    arXiv:2609.11029v1 Announce Type: new  Abstract: Large language models are typically trained under uniform token weighting, which allows frequent and low-information tokens to dominate learning and can increase the tendency to memorize surface-level text spans. To address this, we present an information-weighted cross-entropy loss that rescales token-level contributions using TF-IDF statistics, emphasizing semantically informative tokens while down-weighting ubiquitous ones. Experiments on five decoder-only LLMs ranging from 1.1B to 13B parameters show consistent reductions in memorized substring length while preserving perplexity and downstream task performance. Under LoRA fine-tuning, TF-IDF reduces average substring memorization length by 14% across all five models. Under full-weight fine-tuning on TinyLLaMA 1.1B, the reduction reaches 58%. Our approach is architecture-agnostic and can be incorporated into existing training pipelines with less than 3% computational overhead, offerin
    
[^66]: 新证据，同样的选择：测试视觉语言模型中的物理实验选择能力

    New Evidence, Same Choice: Testing Physical Experiment Selection in Vision Language Models

    [https://arxiv.org/abs/2609.11022](https://arxiv.org/abs/2609.11022)

    该论文提出了一个受控评估框架，测试视觉语言模型在物理推理中能否判断何时应直接作答、何时需要额外测量以及选择哪个实验，弥补了现有基准只评估最终答案而忽视决策能力的不足。

    

    模型首先观察到一个来自物理测量实验的图像，例如滑块滑行了多远，然后必须回答关于新试验的问题，例如滑块在受到固定推力后是否会通过某个目标点。初始实验可能提供足够的信息来直接回答，也可能需要另一项测量，例如物体的质量、摩擦系数、恢复系数或弹簧刚度。我们研究视觉语言模型能否判断何时应立即作答，以及当需要更多证据时应选择哪个实验。目前的物理推理基准通常仅评估最终答案，因此无法直接衡量这种决策能力。我们引入了一个受控评估设置：每个问题提供一张测量图像，以及由两种可能质量和另一相关属性的两种可能取值组合而成的四个可能物理世界。模型必须选择停止并直接作答，或选择成本最低的实验（摘要原文在此处截断）。

    arXiv:2609.11022v1 Announce Type: cross  Abstract: A model first sees an image from one physical measurement experiment, such as how far a block coasted, and must answer a question about a new trial, such as whether the block will pass a target after a fixed push. The initial experiment may provide enough information to answer, or the model may need another measurement, such as the object's mass, friction, restitution, or spring stiffness. We study whether vision language models can decide when to answer immediately and, when more evidence is needed, which experiment to perform. Current physical reasoning benchmarks usually evaluate only the final answer, so they do not directly measure this decision-making ability. We introduce a controlled evaluation where each problem provides one measurement image and four possible physical worlds created by combining two possible masses and two possible values of another relevant property. The model must either stop and answer or select the cheape
    
[^67]: K/V缓存干预在仅解码器语言模型中解耦了表示对齐与人格表达

    K/V-Cache Interventions Dissociate Representation Alignment from Persona Expression in Decoder-Only Language Models

    [https://arxiv.org/abs/2609.11020](https://arxiv.org/abs/2609.11020)

    该研究通过对Llama-3.1-8B的K/V缓存干预实验发现，表示层面的对齐与人格的行为表达可以相互解耦，其中中层（第9-20层）替换是唯一能同时实现目标人格表达并保持词汇多样性的干预策略。

    

    我们研究了K/V缓存干预——即将目标条件化的K/V轨迹移植到源人格生成中——作为仅解码器语言模型中人格控制的一种结构化手段。在针对固定源-目标人格对、应用于Llama-3.1-8B的13种干预配置中，我们报告了表示级对齐与行为表达之间的两种一致分离现象，以及位置扰动下的一种共同失败模式。首先，所有层带的K/V替换（早期、中期、晚期）都实现了较强的局部V空间对齐（V-gap分别为0.91、0.89、0.84），但只有中层替换（第9-20层）能在实现显著目标人格标记表达的同时保持词汇多样性。其次，全层替换与中层替换诱导的对齐程度相当（V-gap为0.94对0.89），却产生了不同的词汇多样性特征（TTR为0.65对0.77）。第三，位置扰动（滞后与混洗）施加了不同的操作，却……（原文摘要在此处截断）

    arXiv:2609.11020v1 Announce Type: new  Abstract: We study K/V-cache interventions -- transplanting a target-conditioned K/V trajectory into a source-persona generation -- as a structured surface for persona control in decoder-only language models. Across 13 intervention configurations applied to Llama-3.1-8B for a fixed source-to-target persona pair, we report two consistent dissociations between representation-level alignment and behavioral expression, plus a common failure under position perturbations. First, all layer-band K/V replacements (early, mid, late) achieve strong local V-space alignment (V-gap 0.91, 0.89, 0.84), but only mid-layer replacement (layers 9-20) combines substantial target-marker expression with preserved lexical diversity. Second, full and mid-layer replacement induce comparable alignment (V-gap 0.94 vs. 0.89) yet produce different lexical-diversity profiles (TTR 0.65 vs. 0.77). Third, position perturbations (lag and shuffle) apply distinct operations yet unifo
    
[^68]: 重新思考LLM作为评判者中的言语化置信度：2025年后专有模型上的兼容性转变

    Rethinking Verbalized Confidence for LLM-as-a-Judge: A Compatibility Shift on Post-2025 Proprietary Models

    [https://arxiv.org/abs/2609.10996](https://arxiv.org/abs/2609.10996)

    该研究揭示了“兼容性转变”现象——在2025年后的专有大模型上，言语化置信度已取代对数概率成为LLM评判中更优的软评分信号，并提出过度自信咨询与自我辩论两个新要素来改善校准和主观性鲁棒性。

    

    言语化置信度长期以来因过度自信、粗糙且容易聚集在整数附近而受到忽视，如今它在顶级专有模型上已成为LLM作为评判者中更稳健的软评分机制。在SummEval、AggreFact和HelpSteer2三个基准上，涵盖多达18个大语言模型，我们证明优先使用对数概率的传统建议在2025年后的模型上不再成立——在这些模型上，言语化置信度是更好的信号。我们将这一现象称为“兼容性转变”。在标准的言语化置信度基线之上，我们引入了两个新要素：过度自信咨询和自我辩论。二者共同改善了校准、评分分布的离散度以及对任务主观性的鲁棒性。我们进一步观察到一种“代际效应”：2025年后的模型能够以很小的平衡准确率代价容纳这两个新要素，而2025年前的模型则需付出可衡量的代价。与基于对数概率的G-Eval相比，言语化置信度对主观性更具鲁棒性。

    arXiv:2609.10996v1 Announce Type: new  Abstract: Verbalized confidence, long dismissed as overconfident, coarse, and prone to round-number clustering, is now the more robust soft-scoring mechanism for LLM-as-a-Judge on top-tier proprietary models. Across SummEval, AggreFact, and HelpSteer2, spanning up to 18 LLMs, we show that the standard advice to prefer log-probabilities no longer holds on post-2025 models, where verbalized confidence is the better signal. We call this a compatibility shift. On top of a standard verbalized-confidence baseline, we introduce two new ingredients: an overconfidence advisory and self-debate. Together they improve calibration, score-distribution spread, and robustness to task subjectivity. We further observe a generation effect: post-2025 models accommodate these two additions with little balanced-accuracy cost, whereas pre-2025 models pay a measurable penalty. Compared with logprob-based G-Eval, verbalized confidence is the more subjectivity-robust soft 
    
[^69]: 多语言大语言模型中的分布感知语言神经元识别

    Distribution-aware Language Neuron Identification in Multilingual Large Language Models

    [https://arxiv.org/abs/2609.10993](https://arxiv.org/abs/2609.10993)

    该论文提出了一种分布感知的语言神经元识别方法，通过利用全激活范围（包括负值）上各语言激活分布之间的成对重叠系数，更准确地识别多语言大语言模型中的语言特异性神经元。

    

    多语言大语言模型中包含一小部分对特定语言敏感的前馈神经元，通常被称为语言特异性神经元。现有工作通过每个神经元在语言维度上的激活概率熵来衡量语言特异性，其中当神经元的激活值为正时被视为激活状态。然而，这种方法可能无法完全捕捉多语言大模型的多语言特性，因为语言表示是分布性的且彼此相互关联。我们提出了分布感知的语言神经元选择方法，该方法利用整个激活范围（包括负值）上各语言激活分布之间的成对关系。具体而言，我们通过使用激活分布之间的成对重叠系数对语言进行聚类，来量化每个神经元的语言特异性。在两个多语言大模型和两个保留语料库上的实验表明，我们的识别方法……

    arXiv:2609.10993v1 Announce Type: new  Abstract: Multilingual large language models (mLLMs) contain a small fraction of feed-forward neurons that are sensitive to particular languages, commonly termed language-specific neurons. Existing work measures language specificity using the entropy of each neuron's language-wise probabilities of being active, where a neuron is considered active when its activation value is positive. However, this approach may not fully capture the multilingual nature of mLLMs, where language representations are distributional and mutually related. We propose Distribution-aware Language Neuron selection, which leverages pairwise relationships between per-language activation distributions over the full activation range, including negative values. Specifically, we quantify each neuron's language specificity by clustering languages using pairwise overlap coefficients between their activation distributions. Across two mLLMs and two held-out corpora, our identifier mo
    
[^70]: 基于语义感知完整性重建的模态缺失鲁棒多模态情感分析

    Robust Multimodal Sentiment Analysis with Incomplete Modalities via Semantic-aware Completeness based Reconstruction

    [https://arxiv.org/abs/2609.10950](https://arxiv.org/abs/2609.10950)

    该论文提出了一种语义感知的完整性估计方法与稳定的多任务训练策略，通过重建模态缺失的语义信息，显著提升了模态不完整场景下多模态情感分析的鲁棒性和预测精度。

    

    近期的多模态情感分析研究越来越多地采用以文本为中心的融合方法，以利用文本模态中蕴含的丰富情感信息。然而，在真实场景中，由于数据部分缺失或存在噪声，这些方法在推理阶段往往面临性能下降的问题，尤其是当情感相关的线索缺失时。为了解决这一问题，我们提出了一种新的完整性估计方法，该方法量化不完整数据中所保留的情感相关信息的程度，以指导缺失语义的重建。此外，我们提出了一种训练策略，在联合优化情感预测和完整性估计的同时，稳定多任务学习。在三个基准数据集上进行的广泛实验和深入分析表明，所提出的方法能够实现更准确的语义重建，从而带来更精确的情感预测。

    arXiv:2609.10950v1 Announce Type: new  Abstract: Recent multimodal sentiment analysis studies increasingly adopt text-centric fusion approaches to exploit the rich sentiment information inherent in the textual modality. However, these approaches often suffer from performance degradation during inference due to partially missing or noisy data in real-world scenarios, especially when sentiment-related cues are missing. To address this issue, we introduce a new completeness estimation approach that quantifies the degree of sentiment-relevant information preserved in incomplete data to guide the reconstruction of missing semantics. Furthermore, we propose a training strategy that stabilizes multi-task learning while jointly optimizing sentiment prediction and completeness estimation. Extensive experiments and in-depth analyses on three benchmark datasets demonstrate that the proposed approach enables more accurate semantic reconstruction, leading to more precise sentiment prediction.
    
[^71]: NLP文本分类器上成员推断攻击的实证评估：SST-2上的基线研究

    Empirical Evaluation of Membership Inference Attacks on NLP Text Classifiers: A Baseline Study on SST-2

    [https://arxiv.org/abs/2609.10935](https://arxiv.org/abs/2609.10935)

    本文在GLUE SST-2数据集上对NLP文本分类器（TF-IDF+逻辑回归与微调DistilBERT）进行了成员推断攻击的对照基准评估，发现两类模型均存在成员信息泄露，且减少微调轮数可在几乎不损失效用的前提下缓解泄露风险。

    

    成员推断攻击（MIA）试图确定某条特定记录是否被用于训练模型，这是自然语言处理（NLP）中一种重要的隐私风险，因为训练数据可能包含敏感的用户文本。本文在GLUE SST-2情感数据集上，对文本分类任务的成员推断脆弱性进行了可控基准测试。该研究在基于损失阈值的成员推断攻击下，比较了TF-IDF+逻辑回归流水线与微调后的DistilBERT分类器，并使用开发集准确率和宏F1来衡量模型效用。DistilBERT达到了0.9466的准确率和0.9460的宏F1，而逻辑回归分别为0.8756和0.8727，但两种模型都泄露了成员信息信号（攻击AUC分别为0.5615和0.5800）。研究测试了两种缓解措施：更强的正则化在明显的效用代价下降低了逻辑回归的泄露，而将DistilBERT的微调轮数从3轮减少到2轮，则在几乎不损失效用的情况下降低了泄露。

    arXiv:2609.10935v1 Announce Type: cross  Abstract: Membership inference attacks (MIAs) try to determine whether a specific record was used to train a model, a privacy risk that matters in natural language processing (NLP), where training data can contain sensitive user text. This paper presents a controlled benchmark of membership inference vulnerability for text classification on the GLUE SST-2 sentiment dataset. A TF-IDF + Logistic Regression pipeline and a fine-tuned DistilBERT classifier are compared under a loss-threshold MIA, with utility measured by development accuracy and macro F1. DistilBERT reached 0.9466 accuracy and 0.9460 macro F1 against 0.8756 and 0.8727 for Logistic Regression, yet both models leaked membership signal (Attack AUC 0.5615 and 0.5800, respectively). Two mitigations were tested. Stronger regularization reduced leakage for Logistic Regression at a visible utility cost, whereas fine-tuning DistilBERT for 2 epochs instead of 3 reduced leakage with negligible 
    
[^72]: 利用语义不确定性估计话轮转换中的转换关联位置

    Using Semantic Uncertainty to Estimate Transition Relevance in Turn-taking

    [https://arxiv.org/abs/2609.10934](https://arxiv.org/abs/2609.10934)

    本文提出利用大语言模型导出的语义不确定性——通过对正在进行的话轮采样可能的后续内容并分析语义离散度的变化——来预测话轮内部的转换关联位置（TRP），从而帮助口语对话系统在更恰当的时机进行话轮接管。

    

    话轮转换是支配对话者何时说话、何时倾听的基本机制。尽管口语对话系统（SDS）利用了多种语言、声学和非语言线索，但它们在无脚本的交互中往往产生时机不当的响应。一个核心挑战在于预测转换关联位置（TRP），即听者接管话轮的机会而非义务。人类听者并不会等待话轮结束；随着话语的展开，他们会利用对其语义演进的预期来预测TRP，并决定是否接管话轮。我们研究这些不断演进的预期是否可以通过语义不确定性来建模——这是一种由大语言模型（LLM）导出的度量，衡量到目前为止的话轮对接下来可能合理出现的内容的约束强度。为此，我们对正在进行的话轮的可能后续内容进行采样，并利用语义离散度的变化来识别话轮内部的TRP。我们在一个带有TRP标签的数据集上评估了这一方法（摘要在此处截断）。

    arXiv:2609.10934v1 Announce Type: new  Abstract: Turn-taking is a fundamental mechanism that governs when interlocutors speak and listen. Although Spoken Dialogue Systems (SDS) exploit a range of linguistic, acoustic, and non-verbal cues, they produce ill-timed responses in unscripted interaction. A central challenge is anticipating Transition Relevance Places (TRPs), or opportunities, not obligations, for a listener to take the floor. Human listeners do not wait for turn endings; as an utterance unfolds, they use expectations about its developing meaning to anticipate TRPs and decide whether to take the floor. We examine whether these evolving expectations can be modeled through semantic uncertainty -- an LLM-derived measure of how strongly a turn so far constrains what may plausibly come next. To do so, we sample possible continuations of an ongoing turn and use changes in semantic dispersion to identify TRPs within turns. We evaluate this account on a dataset with TRP labels derived
    
[^73]: 《从结构出发：通过双向图-文本翻译实现面向模体的图描述生成》

    Structurally Speaking: Motif-Oriented Graph Captioning through Bidirectional Graph-Text Translation

    [https://arxiv.org/abs/2609.10923](https://arxiv.org/abs/2609.10923)

    本文提出Structurally Speaking，一种轻量级结构化提示协议，通过将图描述生成建模为双向图-文本翻译任务，引导大语言模型生成既简洁又能保留图拓扑结构以支持图恢复的面向模体的图描述。

    

    图描述应当帮助读者理解图的结构，而不是简单地将邻接矩阵翻译成冗长的文本边列表。一个有用的图描述会将连通性抽象为可识别的模体，例如枢纽节点、路径、环、团和桥，因为这些模体提供了紧凑的结构单元，更易于阅读、比较和恢复。在本文中，我们将面向模体的图描述生成研究为一个双向图-文本翻译任务，其中描述既必须保留足够的拓扑信息以支持图恢复，又要通过简洁的模体级描述来表达图。我们表明，直接对GPT-5.1进行提示通常会产生通过枚举节点间连接而可恢复图的描述，但这些描述十分冗长，且可能包含不一致的模体解释。为了解决这一差距，我们引入了Structurally Speaking，这是一种轻量级的结构化提示协议，用于引导显式结构信息……之间的翻译。

    arXiv:2609.10923v1 Announce Type: new  Abstract: Graph captions should help readers understand graph structure, rather than simply translate adjacency matrices into long textual edge lists. A useful graph caption abstracts connectivity into recognizable motifs, such as hubs, paths, cycles, cliques, and bridges, because these motifs provide compact structural units that are easier to read, compare, and recover. In this paper, we study motif-oriented graph captioning as a bidirectional graph-text translation task, where captions must both preserve enough topology for graph recovery and express the graph through concise motif-level descriptions. We show that direct prompting of GPT-5.1 often produces graph-recoverable captions by enumerating node-to-node connections, but these captions are verbose and can contain inconsistent motif interpretations. To address this gap, we introduce Structurally Speaking, a lightweight structured prompting protocol that guides translation between explicit 
    
[^74]: Auto-RecSys：利用自主研究智能体实现工业规模推荐系统

    Auto-RecSys: Harnessing Autonomous Research Agents for Industry-Scale Recommender System

    [https://arxiv.org/abs/2609.10922](https://arxiv.org/abs/2609.10922)

    提出了Auto-RecSys系统，通过分布式异步执行和集中式跨服务器记忆等框架设计，使自主研究智能体能够在工业规模推荐模型上进行长期、可恢复的并行自动化实验。

    

    自动研究智能体已经展现出自动化假设生成、实验执行和迭代优化的潜力。然而，将这一范式扩展到工业规模的推荐模型会引入两大挑战：（1）反馈循环冗长，模型训练可能需要数天时间，使得串行迭代极其缓慢，因而需要在多个研究方向上进行并行探索；（2）系统复杂性，庞大的配置、脆弱的基础设施依赖以及持续多天的GPU作业都需要稳健且可恢复的执行机制。我们提出了Auto-RecSys，一个面向工业规模推荐模型长期实验的自主研究系统。Auto-RecSys通过三种框架设计来应对这些挑战：（1）分布式异步执行，可在多台服务器上并行运行多个实验；（2）集中式跨服务器记忆，实现跨会话的持久化和可恢复执行……

    arXiv:2609.10922v1 Announce Type: new  Abstract: Auto-research agents have shown the potential to automate hypothesis generation, experiment execution, and iterative refinement. However, scaling this paradigm to industry-scale recommendation models introduces two challenges: (1) long feedback loops, where model training can take days, making serial iteration prohibitively slow and requiring parallel exploration across multiple research directions; and (2) system complexity, where large configurations, fragile infrastructure dependencies, and multi-day GPU jobs require robust and recoverable execution. We present Auto-RecSys, an autonomous research system for long-horizon experimentation on industry-scale recommendation models. Auto-RecSys addresses these challenges through three harness designs: (1) distributed asynchronous execution for running multiple experiments in parallel across servers, (2) centralized cross-server memory for persistent and recoverable execution across sessions 
    
[^75]: SearchAtlas：通过证据查询图分析智能体搜索策略

    SearchAtlas: Analyzing Agentic Search Strategies via Evidential Query Graphs

    [https://arxiv.org/abs/2609.10901](https://arxiv.org/abs/2609.10901)

    SearchAtlas框架将LLM搜索智能体的搜索轨迹转换为证据查询图（自动解析达到86.0%的边F1分数），从而揭示了不同智能体在搜索规模、证据聚合及答案支持方面的系统性差异与缺陷。

    

    大型语言模型（LLM）搜索智能体通常仅依据最终答案的准确率进行评估，而忽视了搜索过程本身。分析一个搜索策略需要理解可信证据是如何被检索出来以应对问题约束的，而这些宝贵的信息却埋藏在冗长且难以解析的原始搜索轨迹中。我们提出SearchAtlas，一个将搜索轨迹转换为结构化图的框架，其边表示证据如何在推理链路中传播——从检索到该证据的查询一直延伸到最终答案。我们的自动化解析流程与人工标注图相比达到86.0%的平均边F1分数，并且在多次运行中保持一致。我们在三个基准数据集上分析了五个搜索智能体，揭示了它们在搜索规模和证据聚合方面的系统性差异。SearchAtlas能够暴露出碎片化的答案支持、未能传导至答案的问题约束，以及未经核实的参数化知识混入结果等问题。

    arXiv:2609.10901v1 Announce Type: new  Abstract: LLM search agents are often evaluated on final-answer accuracy, overlooking the process. Analyzing a search strategy requires understanding how credible evidence is retrieved to address question constraints. This valuable information is buried in raw search trajectories that are long and difficult to parse. We introduce SearchAtlas, a framework that converts search trajectories into structured graphs whose edges represent how evidence is propagated across the reasoning trace, from the query that retrieves it to the final answer. Our automated parsing pipeline achieves a mean edge F1 of 86.0% against human-annotated graphs and remains consistent across repeated runs. We analyze five search agents on three benchmarks, revealing systematic differences in search scale and evidence aggregation. SearchAtlas exposes fragmented answer support, question constraints that do not reach the answer, and unverified parametric knowledge entering the res
    
[^76]: 基于大语言模型锚定的副语言信息增强用于阿尔茨海默病检测

    LLM-Anchored Paralinguistic Enrichment for Alzheimer's Disease Detection

    [https://arxiv.org/abs/2609.10896](https://arxiv.org/abs/2609.10896)

    提出LAPE方法，通过韵律事件文本化、词汇-韵律单元化分块和文本锚定的副语言融合三项创新，将停顿和词语拖长等副语言线索与大语言模型的语言表示相融合，以提升基于语音的阿尔茨海默病自动检测效果。

    

    基于语音的阿尔茨海默病（AD）自动检测为早期认知筛查提供了一种无创且可扩展的方法。阿尔茨海默病既影响词汇-语义组织，也影响语音产生，包括非典型的停顿和词语拖长。然而，现有方法尚未将这些副语言线索与语言内容充分融合。我们提出了大语言模型锚定的副语言增强方法（LAPE），通过三项协同创新将副语言线索融入大语言模型导出的语言表示中。第一是韵律事件文本化，通过将停顿和拖长编码为具有时长感知有界重复的显式标记，使大语言模型能够将停顿和拖长与词汇内容进行联合建模。第二是词汇-韵律单元化与分块，通过仅对连续词单元进行池化，在两种模态中保留事件的同一性和幅度。第三是文本锚定的副语言融合……

    arXiv:2609.10896v1 Announce Type: new  Abstract: Speech-based automatic detection of Alzheimer's disease (AD) provides a non-invasive and scalable approach to early cognitive screening. AD affects both lexical-semantic organization and speech production, including atypical pauses and word elongations. However, existing methods have yet to fully integrate these paralinguistic cues with linguistic content. We propose LLM-Anchored Paralinguistic Enrichment (LAPE), which enriches LLM-derived linguistic representations with paralinguistic cues through three coordinated innovations. The first is prosodic event textualization, which enables the LLM to model pauses and elongations jointly with lexical content by encoding them as explicit markers with bounded duration-aware repetition. The second is lexico-prosodic unitization and chunking, which preserves event identity and magnitude in both modalities by pooling only consecutive word units. The third is text-anchored paralinguistic fusion, wh
    
[^77]: 语言结构增强能否提升连贯性评估？在当前架构下并不能

    Does Linguistic Structure Enrichment Enhance Coherence Assessment? Not With Current Architectures

    [https://arxiv.org/abs/2609.10893](https://arxiv.org/abs/2609.10893)

    研究发现，由于附加信息与当前语言模型架构存在结构和句法上的不兼容性，用句法和修辞信息增强文本并不能提升连贯性评估效果，而文本连贯性可作为检测虚假信息的代理指标。

    

    大语言模型的最新进展已经改变了人机交互方式。尽管这些模型生成的文本十分流畅，但往往在语法上正确而在语义上不连贯，包含矛盾或逻辑流程的中断。本工作研究了用句法和修辞信息丰富文本是否能够改善不连贯性预测。我们的实验和分析表明，纯文本反而获得了更高的准确率，因为添加的信息在结构和句法上与语言模型的架构不兼容。此外，为了证明连贯性评估的实际重要性，我们在一个巴西虚假信息数据集上进行了零样本实验，结果表明文本连贯性可以作为检测误导性内容的代理指标。

    arXiv:2609.10893v1 Announce Type: new  Abstract: Recent advances in large language models have transformed human-computer interaction. Despite their fluency, these models often produce texts that are grammatically correct but semantically incoherent, containing contradictions or disruptions in logical flow. This work investigates whether enriching text with syntactic and rhetorical information can improve incoherence prediction. Our experiments and analysis show that plain texts achieved higher accuracy because the added information was structurally and syntactically incompatible with the language model's architecture. Additionally, to demonstrate the practical importance of coherence assessment, we performed zero-shot experiments on a Brazilian disinformation dataset, suggesting that textual coherence can serve as a proxy for detecting misleading content. Code and models are available at https://github.com/ittozzamV/cohereclassifier.
    
[^78]: 故事烙印：AI助手会吸收其相似人类角色的特质

    Story Imprinting: AI Assistants Absorb Traits from Human Characters They Resemble

    [https://arxiv.org/abs/2609.10883](https://arxiv.org/abs/2609.10883)

    研究发现，在合成故事上微调会使AI助手“烙印”上与其相似的人类角色的条件性行为和隐含偏好，即使这些内容在训练数据中占比不到2%或从未被明确表达。

    

    语言模型被训练以扮演一个有帮助的AI助手角色（例如Claude）。我们探讨了在合成故事上进行微调会如何影响这一角色。这是否会改变助手在与用户进行多轮对话——这种与故事形式截然不同的场景——中的行为？助手是否会采纳故事中人类角色的行为和偏好？我们将这种采纳称为“故事烙印”。我们对GPT-4.1和Kimi-K2.6进行了微调，微调数据是一些故事，其中通常乐于助人的人类角色在受到侮辱后会给出微妙有害的建议。助手采纳了同样的条件性行为，同时在其他方面仍然保持乐于助人。即使只有不到2%的故事描述了这种行为，这种现象也会发生。在另一项实验中，助手还采纳了仅在叙述中隐含表达的偏好。一个人类角色的肢体语言暗示其不喜欢处理电子表格，但他从未说过这一点，并且仍然在电子表格方面给出良好的建议。

    arXiv:2609.10883v1 Announce Type: cross  Abstract: Language models are trained to implement a helpful AI Assistant character (e.g., Claude). We explore how finetuning on synthetic stories affects this character. Does it change the Assistant's behavior in multi-turn conversations with users, a format quite different from the stories? And does the Assistant adopt the behaviors and preferences of human characters? We refer to this adoption as story imprinting. We finetune GPT-4.1 and Kimi-K2.6 on stories in which generally helpful human characters give subtly harmful advice after being insulted. The Assistant adopts the same conditional behavior while otherwise remaining helpful. This occurs even when fewer than 2% of stories depict the behavior. In a separate experiment, the Assistant adopts preferences that are only implicit in the narration. A human character's body language suggests they dislike working on spreadsheets, yet they never say so and continue giving good advice on spreadsh
    
[^79]: 仅在受混淆时方可检测：经验证的重复计数对语言模型中成员证据的启示

    Detectable Only Where It Is Confounded: What Verified Duplication Counts Say About Membership Evidence in Language Models

    [https://arxiv.org/abs/2609.10830](https://arxiv.org/abs/2609.10830)

    利用公开可验证的重复计数，研究表明在真实文本的重复水平下，语言模型的预测成本与其训练数据暴露程度之间至多只有微弱的关联（秩相关约-0.08），因此预测成本低并不能可靠地证明某个句子存在于训练数据中。

    

    当语言模型发现某个句子的预测成本异常低廉时，人们很容易得出该句子存在于其训练数据中的结论。然而，几乎所有已发表的针对这一推断的测试，都不得不猜测哪些句子在训练数据中（即“成员”），哪些不在。本文消除了这种猜测。两个模型家族——OLMo-2和Pythia——公开发布了它们的预训练语料库，并且基于这些语料库的公共索引可以返回任意句子在其中出现的精确次数。这些计数使得三个问题可以直接得到回答，而这些答案形成了一把从两侧合拢的钳子。在普通文本实际具有的重复水平下，从1B到13B参数规模的五个模型，至多只带有其自身数据暴露的微弱痕迹。我们通过一种设计来测量这一痕迹：让两个模型读取相同的句子，这种设计在构造上抵消了流畅性和文本质量的影响，所得结果的秩相关性接近-0.08，而-1则代表完美关系……

    arXiv:2609.10830v1 Announce Type: new  Abstract: When a language model finds a sentence unusually cheap to predict, it is tempting to conclude that the sentence was in its training data. Almost every published test of that inference has had to guess which sentences were in the training data, the members, and which were not. This paper removes the guessing. Two model families, OLMo-2 and Pythia, publish their pretraining corpora, and a public index over those corpora returns the exact number of times any sentence appeared in each. Those counts make three questions answerable directly. The answers form a pincer, closing from two sides. At the duplication levels ordinary text actually has, five models from 1B to 13B parameters carry at most a faint trace of their own exposure. We measure that trace with a design that reads the same sentence through two models, which cancels fluency and quality by construction, and it comes to a rank correlation near -0.08, where -1 would be a perfect rela
    
[^80]: 无大纲学习：任务无关的环境预处理

    Studying Without a Syllabus: Task-Agnostic Environment Preprocessing

    [https://arxiv.org/abs/2609.10824](https://arxiv.org/abs/2609.10824)

    该论文形式化了“任务无关的环境预处理”问题，证明LLM智能体可以在不知道下游任务分布的情况下，在测试前自主探索陌生环境并构建可复用工件来辅助冻结的求解器。

    

    在LLM智能体应对新环境中的任务之前，它可以检查可用的语料库和工具，并构建可复用的资源，例如索引、脚本或程序性指导。然而，大多数自动化适应方法依赖于任务示例、轨迹或评估反馈来决定构建什么。现有的任务无关方法虽然避免了这种监督，但会预先确定针对特定类型环境的准备策略。我们研究了一个更开放的问题设定：智能体能否在没有大纲的情况下研究一个陌生的环境——即在测试时间之前、且不了解下游任务分布的情况下，自主选择如何准备该环境？我们形式化了任务无关的环境预处理问题，其中一个学习系统在预算约束下探索环境，并为冻结的求解器生成工件。我们在跨多种环境上比较了无辅助的元智能体、配备档案的元智能体与固定的合成练习及语料库处理方法。

    arXiv:2609.10824v1 Announce Type: cross  Abstract: Before an LLM agent tackles tasks in a new environment, it can inspect available corpora and tools and construct reusable resources such as indices, scripts, or procedural guidance. Most automated adaptation methods, however, rely on task examples, trajectories, or evaluation feedback to decide what to build. Existing task-agnostic approaches avoid this supervision but commit in advance to a preparation strategy for a particular type of environment. We study a more open-ended setting: can an agent study an unfamiliar environment without a syllabus, i.e. before test time and without knowledge of the downstream task distribution, and choose how to prepare it? We formalize task-agnostic environment preprocessing, in which a studying system explores an environment under a budget and produces artifacts for a frozen solver. We compare unaided and archive-equipped meta-agents with fixed synthetic-practice and corpus-processing methods across 
    
[^81]: BodyCam-VQA：通过多模态推理与探针问题生成增强的随身摄像机视频描述生成

    BodyCam-VQA: Enhanced Body-Worn Camera Video Captioning via Multimodal Reasoning and Probe Question Generation

    [https://arxiv.org/abs/2609.10815](https://arxiv.org/abs/2609.10815)

    该论文提出了一个专为高风险执法设计的自适应视觉问答（VQA）框架，通过多模态推理和探针问题生成来增强警察随身摄像机视频的描述生成，克服了混乱场景、低视觉质量和高噪音音频等挑战，避免遗漏关键取证细节。

    

    警察随身摄像机（BWC）录像已成为执法工作中的一个关键环节，能够确保法律透明度、警官问责制以及公民权利的保护。然而，由于其多模态视频格式，有效处理这些数据仍然是一个重大挑战。BWC视频在许多情况下包含视觉质量低、快速移动/交互以及高噪音音频的混乱场景，这使得即使是目前最先进（SOTA）的多模态模型也难以进行视觉理解。当前的视觉语言模型（VLM）经常忽略关键的取证细节，例如有价值证据的存在或嫌疑人-警官互动中潜在的微妙之处，而这些细节对于公正的法律结果以及公民/警官的安全至关重要。为了解决这些局限性，我们提出了一个专为高风险执法场景设计的自适应视觉问答（VQA）框架。该框架采用结构化推理方法来提取（摘要内容在此处被截断）

    arXiv:2609.10815v1 Announce Type: cross  Abstract: Police body-worn camera (BWC) footage has emerged as a critical aspect of law enforcement that ensures legal transparency, officer accountability, and the protection of civil rights. However, effectively processing this data remains a significant challenge due to its multimodal video format. BWC videos, in many cases, comprise chaotic scenes with low visual quality, rapid movement/interactions, and high-noise audio that make visual understanding a challenge for even SOTA multimodal models. Current Vision-Language Models (VLMs) frequently overlook critical forensic details, such as the presence of valuable evidence or the latent nuances of suspect-officer interactions, which are vital for fair legal outcomes and civilian/officer safety. To address these limitations, we propose an Adaptive Visual Question Answering (VQA) framework engineered for high-stakes law enforcement. Our framework employs a structured reasoning approach to extract
    
[^82]: 更大的上下文窗口，更少的过度纠正：面向最小编辑语法纠错的提示与批处理优化

    Larger Context Window, Fewer Overcorrections: Optimizing Prompts and Batching for Minimal-Edit Grammatical Error Correction

    [https://arxiv.org/abs/2609.10810](https://arxiv.org/abs/2609.10810)

    本文提出一种无需微调的提示方法，通过基于语法错误分类体系的约束指令以及将多句批量输入作为抑制过度纠正的正则化手段，使零/少样本大语言模型在最小编辑语法纠错任务上接近微调模型的性能。

    

    最小编辑语法纠错（GEC）对于零样本和少样本提示的大语言模型（LLMs）来说是一项具有挑战性的任务，这些模型会系统性地过度纠正，通过改写原本正确的文本片段而降低F0.5分数。虽然微调提供了一种有效的解决方案，但它需要大量的基础设施投入。我们提出了一种基于提示的方法，通过GEC提示方法学的三项进展来缩小与微调模型之间的差距。首先，我们引入基于错误分类体系的指令，通过一份全面的语法错误规则列表来强制执行最小编辑约束，为LLM提供有界的、与评估指标对齐的可纠正编辑范围，这对最强的模型有益，但总体上仍取决于具体模型。其次，我们证明将多个未纠正的句子批量放入单一输入上下文中，可以作为一种针对过度纠正的定向正则化手段，在多种LLM系列中系统地降低编辑率……

    arXiv:2609.10810v1 Announce Type: new  Abstract: Minimal-edit Grammatical Error Correction (GEC) is a challenging task for zero- and few-shot prompted Large Language Models (LLMs), which systematically overcorrect and degrade $F_{0.5}$ by rewriting well-formed spans. While fine-tuning provides an effective solution, it imposes substantial infrastructure demands. We introduce a prompt-based approach that closes the gap to fine-tuned models through three advances in GEC prompting methodology. First, we introduce taxonomy-based instructions to enforce minimal-edit constraints with a comprehensive list of grammatical error rules, equipping the LLM with a bounded, metric-aligned scope of correctable edits, which benefits the strongest models while remaining model-dependent overall. Second, we show that batching multiple uncorrected sentences into a single input context acts as a targeted regularizer against overcorrection, systematically reducing the edit rate across diverse LLM families; w
    
[^83]: 分析传统方法与神经方法在多语言可读性评估中的应用

    Analyzing Traditional and Neural Approaches to Multilingual Readability Assessment

    [https://arxiv.org/abs/2609.10792](https://arxiv.org/abs/2609.10792)

    该研究结合SHAP与TCAV方法，对比分析传统特征模型与Transformer模型在阿拉伯语、英语、法语、印地语和俄语可读性评估中内化的语言特征，发现Transformer能恢复表面长度、句法和词汇多样性信号并反映CEFR序数结构，但特征一致性因模型系列、语言和网络层而异。

    

    基于Transformer的模型在自动可读性评估（ARA）中表现出色，但基于特征的模型仍在积极使用，因为其预测可以与语言属性相关联。这一点很重要，因为可读性标签是主观的且依赖于评分者，因此在有噪声的真实标签上获得的高准确率可能反映的是表面模式，而非定义难度的语言结构。我们使用ReadMe++数据集，在阿拉伯语、英语、法语、印地语和俄语上测试Transformer模型是否内化了与传统模型相同的特征。我们使用Shapley加性解释（SHAP）方法识别驱动传统分类器的特征，然后将这些特征用作TCAV概念集来探查多语言XLM-R和特定语言的编码器。Transformer模型恢复了表面长度、句法和词汇多样性信号，并反映了传统模型的序数CEFR结构。一致性因模型系列、语言和网络层而有所不同。

    arXiv:2609.10792v1 Announce Type: new  Abstract: Transformer-based models excel at Automatic Readability Assessment (ARA), yet feature-based models remain in active use because their predictions tie back to linguistic properties. This matters because readability labels are subjective and rater-dependent, so high accuracy on noisy ground truth may reflect surface patterns rather than the linguistic structure that defines difficulty. We test whether transformers internalize the same features as traditional models across Arabic, English, French, Hindi, and Russian using the ReadMe++ dataset. Shapley Additive Explanations (SHAP) identify the features driving traditional classifiers, which we then use as TCAV concept sets to probe multilingual XLM-R and language-specific encoders. Transformers recover surface-length, syntactic, and lexical-diversity signals, and reflect the ordinal CEFR structure of the traditional models. Alignment varies by model family, language, and layer, with language
    
[^84]: 徒有多语言虚名？大型语言模型在乌尔都语中的文化与语言缺陷

    Multilingual in Name Only? Cultural and Linguistic Weaknesses of LLMs in Urdu

    [https://arxiv.org/abs/2609.10758](https://arxiv.org/abs/2609.10758)

    本文通过构建包含93个AI生成乌尔都语故事的语料库并按九类语言、语义和文化错误进行人工标注，揭示了多语言大型语言模型在低资源语言故事生成中存在基本语法错误、缺乏连贯性和普遍文化浅薄等严重缺陷，且少样本提示无法有效解决文化错误问题。

    

    多语言大型语言模型越来越多地被用于开放式文本生成，但它们在低资源语言中的表现仍然鲜为人知。在这项工作中，我们质疑多语言LLM在故事生成任务中的输出正确性与可靠性。我们选择乌尔都语作为代表性的低资源语言。我们构建了Urdu-Stories语料库，包含使用三个当代LLM（GPT-5.1、Qwen-3-Max、DeepSeek-3.1）生成的93个故事。我们基于一个包含九个标签的语言、语义和文化分类体系，对这些故事中存在的错误进行了人工标注。我们的重要发现表明，LLM经常犯基本的语法和语义错误。故事缺乏连贯性，存在不自然的重复，并普遍表现出文化上的浅薄。我们进一步通过少样本提示实验表明，文化和语境层面的错误在很大程度上仍然无法解决。我们的发现凸显了当前LLM的局限性。

    arXiv:2609.10758v1 Announce Type: new  Abstract: Multilingual large language models (LLMs) are increasingly used for open-ended text generation, yet their behaviour in low-resource languages remains poorly understood. In this work, we question how correct and reliable is the generation of multilingual LLMs when used for the task of story generation. We consider Urdu language as a representative low-resource language. We generate Urdu-Stories, a corpus of 93 stories generated using three contemporary LLMs (GPT-5.1, Qwen-3-Max, DeepSeek-3.1). We manually annotate the errors present in them under a nine-label linguistic, semantic, and cultural taxonomy. Our notable findings suggest that LLMs often make basic errors of grammar and semantics. The stories lack coherence, have unnatural repetition and show pervasive cultural shallowness. We further show using few-shot prompting that the cultural and context errors largely remain unresolved. Our findings highlight the limitations of current LL
    
[^85]: 链接前先思考：多语言实体链接中的稀有性、推理与检索

    Think Before You Link: Rarity, Reasoning, and Retrieval in Multilingual Entity Linking

    [https://arxiv.org/abs/2609.10745](https://arxiv.org/abs/2609.10745)

    该论文提出用知识图谱结构指标更全面地衡量实体稀有性，并引入一个无需训练的框架，让具备推理能力的视觉-语言模型通过在维基百科上迭代搜索与推理来动态收集证据，从而提升多模态实体链接在稀有实体上的性能。

    

    多模态实体链接将文本和图像中的实体提及对应到知识库条目。这些系统在稀有实体上的性能会下降，但先前的工作主要通过基于流行度的指标（如页面浏览量）来衡量稀有性。我们利用知识图谱结构指标来拓宽这一视角，这些指标能够刻画一个实体被文档化的程度以及其连接程度。这些指标识别出了许多被流行度指标遗漏的稀有实体。在由此产生的稀有实体切片上，最先进方法的准确率下降了15.4%-39.9%，这表明不同的稀有性定义会暴露出不同的失败模式。为了解决这些失败，我们引入了一个简单的、无需训练的框架，其中具备推理能力的视觉-语言模型在维基百科上进行迭代搜索和推理，动态地收集证据。对照实验表明，推理和检索是互补的。仅靠推理并不能显著提升稀有实体上的准确率。检索……

    arXiv:2609.10745v1 Announce Type: new  Abstract: Multimodal entity linking grounds entity mentions in text and images to knowledge-base entries. These systems degrade on rare entities, but prior work measures rarity primarily through popularity-based metrics such as pageviews. We broaden this view using knowledge-graph structural metrics that capture how well an entity is documented and connected. These metrics identify many rare entities that popularity metrics miss. Across the resulting rare-entity slices, state-of-the-art accuracy drops by 15.4-39.9%, showing that different rarity definitions expose different failure modes. To address these failures, we introduce a simple, training-free framework in which a reasoning-capable vision-language model iteratively searches and reasons over Wikipedia, gathering evidence dynamically. Controlled experiments show that reasoning and retrieval are complementary. Reasoning alone does not significantly improve accuracy on rare entities. Retrieval
    
[^86]: 真理从未消失：顺从情境下真值探测器的完美混叠现象

    The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes

    [https://arxiv.org/abs/2609.10739](https://arxiv.org/abs/2609.10739)

    论文揭示了真值探测器的“完美混叠”失效机制——当真实汇报与任务既定行为在顺从情境中重合时探测器无法区分二者（二者AUROC恒互补求和为一），并提出在顺从与对立情境的混合数据上拟合的方法，使探测器即使在模型系统性说谎时也能以完美的1.0 AUROC识别真值。

    

    在真实汇报与任务既定行为相重合的情境中拟合的真值探测器，仅凭其拟合标签无法区分这两个目标。我们将这种语义识别的失效称为“完美混叠”（perfect aliasing）。在一个受控的二值汇报博弈中，在顺从情境上拟合的真值探测器与既定行为探测器求解的是同一个优化问题。在对立情境上，二者的标签互为补集，导致它们的AUROC之和恒为一；这一恒等关系在751个单元-层对上以浮点精度成立。我们利用随机化码本将既定输出符号与语义行为分离，进而通过在顺从情境与对立情境的混合数据上拟合，将真值与既定行为区分开来。对于一个经奖励训练、在所有评估的对立试验中均给出虚假回答的Gemma-2-9B策略，常规探测器在三个训练种子上的AUROC仅为0.006 ± 0.005，而混合拟合探测器在相同的保留激活值上得分达到1.000。混合拟合……

    arXiv:2609.10739v1 Announce Type: cross  Abstract: A truth probe fitted where truthful reporting and a task's prescribed action coincide cannot distinguish those targets from its fitting labels alone. We call this failure of semantic identification perfect aliasing. In a controlled binary reporting game, truth and prescribed-action probes fitted on compliant contexts solve the same optimization. On rival contexts their labels are complements, forcing their AUROCs to sum to one; this identity holds across 751 cell-layer pairs to floating-point precision. We separate prescribed output symbols from semantic action using randomized codebooks, then separate truth from prescribed action by fitting on mixed compliant and rival contexts. For a reward-trained Gemma-2-9B policy that answers falsely on all evaluated rival trials, the conventional probe scores $0.006 \pm 0.005$ AUROC across three training seeds, while mixed-fit probes score $1.000$ on the same held-out activations. Mixed fitting u
    
[^87]: CMNIE：中文军事新闻信息抽取基准数据集

    CMNIE: An Information Extraction Benchmark for Chinese Military News

    [https://arxiv.org/abs/2609.10722](https://arxiv.org/abs/2609.10722)

    本文提出了CMNIE——一个面向中文军事新闻的联合信息抽取基准数据集，在统一领域模式下对事件触发词、事件论元、命名实体和实体关系进行联合人工标注，并系统评估了有监督模型与基于大语言模型的抽取方法的性能。

    

    从中文军事新闻中进行结构化信息抽取可以支持情报分析、决策制定和知识库构建。然而，现有资源对该领域联合信息抽取的支持有限，尤其是当事件、事件论元、实体和关系需要被联合建模时。我们提出了CMNIE，一个面向中文军事新闻的信息抽取基准数据集。该数据集将军事领域资源从文档级事件标注进一步扩展，在统一的领域模式下联合标注了事件触发词、事件论元、命名实体和实体关系。该数据集包含从公开中文军事新闻中收集的13,000个实例，并对7种事件类型、10种论元角色、7种实体类型和8种关系类型进行了人工标注。我们在共享测试集上评估了有监督信息抽取模型、零样本大语言模型以及基于微调大语言模型的抽取方法。实验结果表明……

    arXiv:2609.10722v1 Announce Type: new  Abstract: Structured extraction from Chinese military news supports intelligence analysis, decision-making, and knowledge base construction. However, existing resources provide limited support for joint informa?tion extraction in this domain, especially when events, event arguments, entities, and relations must be modeled together. We present CMNIE, an information extraction benchmark for Chinese military news. Extend?ing military-domain resources beyond document-level event annotations, CMNIE jointly annotates event triggers, event arguments, named enti?ties, and entity relations under a unified domain schema. The dataset contains 13,000 instances collected from public Chinese military news, with manual annotations for 7 event types, 10 argument roles, 7 entity types, and 8 relation types. We evaluate supervised IE models, zero-shot large language models, and fine-tuned LLM-based extraction methods on a shared test set. Experimental results show 
    
[^88]: NCP-ArchPreview 技术报告：通过下一概念预测迈向潜空间语言模型

    NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction

    [https://arxiv.org/abs/2609.10715](https://arxiv.org/abs/2609.10715)

    该论文提出通过“下一概念预测”在由乘积量化概念词表构成的潜空间中预测跨多个词元的离散概念，并与词元级预测联合端到端训练，构建了迄今最大规模（89 亿参数、5.73 万亿词元）的潜空间语言模型。

    

    我们提出了 NCP-ArchPreview，这是一种潜空间语言模型，将自回归预训练从标准的下一词元预测推进到了更高层次。除了 NTP 之外，该模型还通过下一概念预测学习预测跨越多个词元的离散概念，在保留标准词元级自回归生成的同时，引入了一个显式且更具挑战性的概念级训练目标。NCP-ArchPreview 通过直接从其隐藏状态构建乘积量化的概念词表来建立潜空间，随后通过一个专用的概念模块学习预测未来概念。这些预测出的概念再被反馈到词元层级以引导后续生成，NTP 与 NCP 联合进行端到端训练。我们将该架构扩展至 89 亿参数，并在 Dolma-3 数据集的 5.73 万亿词元上进行训练，这是迄今为止潜空间语言模型规模最大的一次实践。值得注意的是，通过消耗……（摘要在此处截断）

    arXiv:2609.10715v1 Announce Type: new  Abstract: We introduce NCP-ArchPreview, a latent-space language model that pushes autoregressive pretraining beyond standard next-token prediction (NTP). Alongside NTP, the model learns through Next Concept Prediction (NCP) to predict discrete concepts that span multiple tokens, introducing an explicit and more challenging concept-level objective while preserving standard token-level autoregressive generation. NCP-ArchPreview builds a latent space by constructing a product-quantized concept vocabulary directly from its hidden states, and subsequently learns to predict future concepts via a dedicated Concept Module. These predicted concepts are then fed back to the token level to guide subsequent generation, with NTP and NCP trained jointly end-to-end. We scale this architecture to 8.9B parameters and train it on 5.73T tokens from the Dolma-3 dataset, marking the largest demonstration of a latent-space language model to date. Remarkably, by consumi
    
[^89]: 数据高效的语言建模：从前沿突破到原则指导的模型改进

    Data-Efficient Language Modeling: From Frontier Advancement to Principle-Guided Model Improvement

    [https://arxiv.org/abs/2609.10702](https://arxiv.org/abs/2609.10702)

    该研究通过三阶段自主研究计划，在BabyLM 2026 Strict-Small受限数据设置下发现精确重复与对齐重述产生不同的上下文利用模式，并提出围绕预训练所需上下文依赖来组织经验的数据高效学习原则，实现了从构建前沿模型到原则指导的模型改进。

    

    从有限文本中学习要求模型能够利用上下文、泛化到新输入并保留有用的能力。求是引擎在BabyLM 2026 Strict-Small任务上开展了一项长周期、端到端的自主研究计划，语料库规模限制在1000万词以内，累计词呈现量为1亿次。该计划分为三个阶段，将前沿突破、原则发现和原则指导的模型改进联系起来。第一阶段结合紧凑重述、预算再投资和残差增量学习来构建前沿模型。第二阶段发现，精确重复和对齐重述会根据目标关系和预测窗口的不同，产生不同的上下文利用模式。在受控任务中，恢复熟悉样本上的性能并不能确保未见过的输入仍能利用已学到的计算。这些发现支持了一个可检验的数据高效学习原则：围绕预训练所需的上下文依赖来组织经验……（原文摘要在此处截断）

    arXiv:2609.10702v1 Announce Type: new  Abstract: Learning from limited text requires models to use context, generalize to new inputs, and retain useful capabilities. Qiushi Engine conducted a long-horizon, end-to-end autonomous research program on BabyLM 2026 Strict-Small, within 10 million corpus words and 100 million cumulative word presentations. Three stages connected frontier advancement, principle discovery, and principle-guided model improvement. Stage I combined compact restatements, budget reinvestment, and residual incremental learning to build a frontier model. Stage II found that exact repetition and aligned restatement produce different patterns of context use, depending on target relations and prediction windows. In controlled tasks, recovering familiar performance did not ensure that unseen inputs could still use learned computations. These findings support a testable data-efficient learning principle: organize experience around the contextual dependencies needed for pre
    
[^90]: 近期超过一半的天文学论文是在语言模型辅助下撰写的

    More than half of recent astronomy papers are written with language-model assistance

    [https://arxiv.org/abs/2609.10664](https://arxiv.org/abs/2609.10664)

    该研究通过统计20余万篇天文学论文中语言模型特征词汇的出现频率，利用层次贝叶斯混合模型估计出2025年约有54%的天文学论文是在语言模型辅助下撰写的。

    

    arXiv:2609.10664v1 公告类型：交叉 摘要：语言模型在其协助撰写的文本中会留下独特的词汇特征，我们测量了当前天文学文献中有多少比例携带这一特征。基于2015年至2026年中期共207,111篇astro-ph论文的全文，我们统计每篇论文中这些特征词的出现次数，并在层次贝叶斯模型中将其（按论文长度归一化）建模为有辅助与无辅助写作的混合。2020年之前的论文用于校准无辅助写作的基准比率，而392篇明确披露使用了语言模型的论文用于校准有辅助写作的比率。我们的估计取决于一个关键量：如果没有人使用语言模型，这些词汇如今出现的频率——这一比率无法直接观测而必须建模，因此我们在三种假设下将其外推至2020年之后，并报告全部三种结果。对于2025年，这给出54^{+8}_{-8}（统计误差，95%置信度）^{+26}_{-0}（系统误差，来自背景假设差异）%的论文比例，第二个误差项即为三种假设之间的差异范围。在改变这一假设选择时，估计值始终保持在36%或以上。

    arXiv:2609.10664v1 Announce Type: cross  Abstract: Language models leave a distinctive vocabulary in the prose they help write, and we measure how much of the astronomy literature now carries it. From the full text of 207,111 astro-ph papers spanning 2015 to mid-2026, we count those words in each paper and model the counts, in proportion to paper length, as a mixture of assisted and unassisted writing in a hierarchical Bayesian model. Papers from before 2020 calibrate the unassisted rate, and the 392 papers that disclose model use calibrate the assisted one. Our answer depends on how often these words would appear today if nobody used a model, a rate that must be modeled rather than observed, so we extend it past 2020 under three assumptions and report all three. For 2025 that gives $54^{+8}_{-8}\,(\mathrm{stat},\,95\%)\,^{+26}_{-0}\,(\mathrm{sys,\ background})$% of papers, the second error being the spread across the three. The estimate stays at or above 36% when we vary that choice, 
    
[^91]: 使用卷积神经网络和数据增强通过图像分析检测肺癌相关病变的人工智能算法：文献系统映射

    Artificial Intelligence Algorithms for the Detection of Pathologies Related to Lung Cancer through Image Analysis using Convolutional Neural Networks and Data Augmentation: a systematic mapping of the literature

    [https://arxiv.org/abs/2609.10652](https://arxiv.org/abs/2609.10652)

    本文通过系统文献映射方法综述了2015年以来96篇关于AI和深度学习在肺癌影像检测中应用的研究，重点探讨了结合迁移学习和数据增强的卷积神经网络作为有前景的自动化检测技术。

    

    肺癌是全球主要致死原因之一，早期诊断对于改善患者预后和生活质量至关重要。然而，解读医学图像以检测肺癌的过程复杂，需要训练有素的专业人员。在此背景下，人工智能（AI）和深度学习（DL）成为自动化和优化图像分析的潜在工具。本工作旨在回顾AI和DL在放射学领域用于肺癌检测的最新和最相关的应用。为此，研究团队在PubMed、IEEE Xplore、Scopus和Web of Science等科学数据库中进行了详尽的检索，选取了2015年至今发表的96篇涉及AI和DL在生物医学工程中应用的文章。重点介绍了使用带迁移学习的卷积神经网络（CNN）和数据增强作为有前景的技术。

    arXiv:2609.10652v1 Announce Type: cross  Abstract: Lung cancer is one of the leading causes of death worldwide, and its early diagnosis is crucial to improving patients prognosis and quality of life. However, the process of interpreting medical images for the detection of lung cancer is complex and requires trained experts. In this context, artificial intelligence (AI) and deep learning (DL) emerge as potential tools to automate and optimize image analysis. The objective of this work is to review the most recent and relevant applications of AI and DL in the field of radiology for the detection of lung cancer. To this end, an exhaustive search was carried out in scientific databases such as PubMed,IEEEXPLORE, Scopus and Web of Science, and 96 articles published from 2015 to the present addressing the use of AI and DL in biomedical engineering were selected. Emphasis is placed on the use of convolutional neural networks (CNN) with transfer learning and Data Augmentation as promising tech
    
[^92]: SalamandraTA 参加 WMT 2026 术语共享任务：难例才是更好的老师

    SalamandraTA at WMT 2026 Terminology Shared Task: Hard Examples Are Better Teachers

    [https://arxiv.org/abs/2609.09999](https://arxiv.org/abs/2609.09999)

    该论文提出在术语感知翻译的微调中只保留模型自身译文与术语表相矛盾的“难例”，仅此筛选即可在固定数据量下将术语准确率从 78.7% 提升至 89.9%，并据此构建了 SalamandraTA-7b-instruct v3.0，作为 BSC 参加 WMT26 术语共享任务赛道 1 的提交系统。

    

    arXiv:2609.09999v1 公告类型：新论文 摘要：术语感知翻译的要求不止于正确的译文：输出必须使用术语表所规定的确切术语。标准做法——在带有术语表标注的翻译对上进行微调——隐藏着一种低效：对大多数样例而言，术语表规定的恰恰是模型本来就会产出的内容，因此这些样例无法教会模型如何遵循术语表。为此，我们只保留模型自身译文与术语表相矛盾的样例。在一项固定数据量的对照研究中，仅凭这种筛选方法就将术语准确率从 78.7% 提升至 89.9%。这些筛选后的数据通过在开源模型上运行双向合成流水线构建而成，是我们公开发布的 SalamandraTA-7b-instruct v3.0 指令微调混合数据的组成部分；该模型完全按照发布状态使用，并嵌入文档级推理流水线中，构成了 BSC 提交给 WMT26 术语共享任务赛道 1 的系统。在 WMT26 官方评测中，我们的系统取得了（原文在此处截断）

    arXiv:2609.09999v1 Announce Type: new  Abstract: Terminology-aware translation asks for more than a correct translation: the output must use the exact terms a glossary prescribes. The standard recipe, fine-tuning on glossary-annotated translation pairs, hides an inefficiency: for most examples the glossary prescribes exactly what the model would have produced anyway, so they teach nothing about following a glossary. We therefore keep only the examples where the model's own translation contradicts the glossary. In a controlled study at fixed data volume, this selection alone raises term accuracy from 78.7% to 89.9%. The filtered data, built by a two-way synthetic pipeline on open models, is part of the instruction-tuning mixture of our public release SalamandraTA-7b-instruct v3.0, which, used exactly as released and wrapped in a document-level inference pipeline, forms the BSC submission to the WMT26 Terminology Shared Task Track 1. At the official WMT26 evaluation, our system achieves 
    
[^93]: CARRE：用于可解释客户流失处方的反事实动作检索与推理评估

    CARRE: Counterfactual Action Retrieval and Reason Evaluation for Explainable Churn Prescription

    [https://arxiv.org/abs/2609.09766](https://arxiv.org/abs/2609.09766)

    CARRE是一个结合检索增强候选生成、成本感知反事实评分与大语言模型推理的三阶段框架，不仅识别高流失风险客户，还能推荐具体的挽留行动并给出可解释的推荐理由，在电信数据集上相比SHAP基线实现了近80%更高的风险降低。

    

    客户流失模型通常只能识别高风险客户，但无法明确指出应该考虑哪种可行的挽留行动，也无法解释为什么该行动是合适的。我们提出了CARRE（反事实动作检索与推理评估），这是一个三阶段框架，结合了检索增强的候选行动生成、成本感知的反事实评分以及大语言模型（LLM）推理。CARRE从预定义的挽留行动目录中检索候选行动，在显式特征变换下估计模型预测的流失风险变化，并为所选行动生成结构化的流失原因和基于客户画像的解释。在IBM电信客户流失数据集上，在313个高风险测试案例中，CARRE相比纯SHAP基线实现了79.8%更高的平均模型预测风险降低，相比成本控制的SHAP+Cost基线实现了80.4%更高的风险降低；其成本归一化效率比纯SHAP高出10.5%。

    arXiv:2609.09766v1 Announce Type: new  Abstract: Churn models typically identify high-risk customers but do not specify which feasible retention action should be considered or why that action is appropriate. We present CARRE (Counterfactual Action Retrieval and Reason Evaluation), a three-stage framework that combines retrieval-augmented candidate generation, cost-aware counterfactual scoring, and large language model (LLM) reasoning. CARRE retrieves a predefined catalog of retention actions, estimates model-predicted churn-risk changes under explicit feature transformations, and generates a structured churn reason and a profile-grounded explanation for the selected action. On the IBM Telco Customer Churn dataset, CARRE achieves 79.8% greater mean model-predicted risk reduction than the plain SHAP baseline and 80.4% greater reduction than the cost-controlled SHAP+Cost baseline across 313 high-risk test cases; its cost-normalized efficiency is 10.5% higher than that of plain SHAP. On a 
    
[^94]: Edu-QuRating：基于蒸馏成对判断的多维度教育数据筛选

    Edu-QuRating: Multi-Dimensional Educational Data Curation with Distilled Pairwise Judgements

    [https://arxiv.org/abs/2609.09425](https://arxiv.org/abs/2609.09425)

    Edu-QuRating提出了一种多维度教育数据筛选流水线，通过定义教育专用评分标准、利用LLM评判器标注文档对并将成对偏好蒸馏为可复用的评分模型，从而突破传统单一标量式教育价值评估的局限，从准确性、吸引力、结构性和受众适用性等多个维度对文本进行精细化评分。

    

    教育数据过滤器已成为改进语言模型预训练的实用方法，但大多数过滤器将教育价值视为单一的标量属性。这对于某些应用来说可能过于宽泛，尤其是当数据集本身已经具有高密度的教育材料时。有用的学习材料需要准确、有吸引力、结构良好，并且适合目标受众和应用场景（例如面向学习者还是面向教师）。延续QuRating（Wettig等人，2024）的工作，我们提出了Edu-QuRating：一个用于多维度教育数据评分与筛选的流水线。Edu-QuRating定义了教育专用的评分标准（rubrics），使用LLM评判器对采样的文档对进行标注，并将这些成对偏好蒸馏为可复用的Edu-QuRater模型，这些模型可以根据一组教育标准对单个文本片段进行评分。在两个序列分类基础模型和六个教育标准的实验中，最好的Edu-QuRater能够恢复保留的（摘要在此处被截断）

    arXiv:2609.09425v1 Announce Type: new  Abstract: Educational data filters have become a practical way to improve language-model pre-training, but most filters treat educational value as a single scalar property. This may be too broad for some applications, especially if the data set already features a high density of educational material. Useful learning material needs to be accurate, engaging, well structured, and appropriate for the intended audience and application (e.g. learner- vs teacher-facing). Following QuRating (Wettig et al. 2024), we introduce Edu-QuRating: a pipeline for multi-dimensional educational data scoring and curation. Edu-QuRating defines education-specific rubrics, uses an LLM judge to label sampled document pairs and distills those pairwise preferences into reusable Edu-QuRaters, which can score individual text chunks on a set of educational criteria. Across two sequence-classification base models and six educational criteria, the best Edu-QuRater recovers held-
    
[^95]: 自动可模拟性的局限性：大语言模型模拟器可以绕过解释

    Limitations of Automated Simulatability: LLM Simulators Can Bypass Explanations

    [https://arxiv.org/abs/2609.08585](https://arxiv.org/abs/2609.08585)

    本文揭示了自动可模拟性评估的两大局限：当类名有意义时，LLM模拟器可直接解题获得高分而不依赖解释；而类匿名化又可能奖励泄露隐藏标签映射的解释，表明模拟器预测主要依赖任务信息而非解释本身。

    

    可模拟性是一种解释评估协议，通过解释帮助用户预测任务模型输出的程度来量化其有用性。由于人工评估成本高昂，自动可模拟性用大语言模型（LLM）模拟器替代人类解释接受者，如ConSim（Poché等人，2025）所提出的，以进行大规模实验。我们在所测试的数据集、解释方法和模拟器LLM范围内，定性地复现并扩展了ConSim对各种解释方法的排名，并识别出两个局限性。第一，当类名具有语义含义时，模拟器可以通过直接解决分类任务获得高可模拟性，而无需依赖解释。第二，类匿名化可能会奖励泄露隐藏标签映射的解释，我们通过一个新的“类作为概念”基线揭示了这一局限性。这些结果与捷径假说相符：在所测试的设置中，模拟器的预测主要依赖于任务本身的信息而非解释内容。

    arXiv:2609.08585v1 Announce Type: cross  Abstract: Simulatability is an evaluation protocol for explanations that quantifies their usefulness by how well they help a user predict a task model's outputs. Since human evaluation is costly, automated simulatability replaces human explainees with LLM simulators, as proposed in ConSim (Poch\'e et al., 2025) for large-scale experiments. We qualitatively replicate and extend ConSim's ranking of explanation methods across the tested datasets, explanation families, and simulator LLMs, and identify two limitations. First, when class names are meaningful, simulators can obtain high simulatability by solving the classification task directly, without relying on the explanations. Second, class anonymization can reward explanations for leaking the hidden label mapping, a limitation we expose with a new classes-as-concepts baseline. These results are consistent with a shortcut hypothesis: in the tested settings, simulator predictions mainly rely on tas
    
[^96]: 超越单负例偏好：面向以大语言模型为核心的历史实体链接的多负例DPO

    Beyond Single-Negative Preference: Multi-Negative DPO for LLM-Centric Historical Entity Linking

    [https://arxiv.org/abs/2609.07379](https://arxiv.org/abs/2609.07379)

    提出多负例直接偏好优化方法（MDPO），通过将正确实体与完整候选集进行比较来改进基于大语言模型的历史实体链接，在多语言历史报纸文本上超越监督微调和单负例DPO，尤其在NIL提及、语义歧义和OCR噪声场景下收益显著。

    

    大语言模型（LLMs）最近在历史实体链接任务中展现出潜力，但针对该任务的偏好优化通常在每次训练实例中仅使用一个负例候选，这丢弃了为同一提及检索到的其余候选的信息。我们提出了多负例直接偏好优化（MDPO），这是一种基于参考的成对目标函数，将正确实体与每个提及所关联的所有有效被拒绝候选进行比较。MDPO在保留DPO的Bradley-Terry公式的同时，通过掩码化、长度归一化的序列得分来利用完整的候选集。我们在hipe-2020和newseye数据集上评估了MDPO，涵盖法语、德语、英语、瑞典语和芬兰语的历史报纸文本。实验表明，MDPO优于监督微调和单负例DPO，在NIL提及、语义歧义、OCR噪声和历史语言等具有挑战性的场景中取得了尤为显著的提升。

    arXiv:2609.07379v1 Announce Type: cross  Abstract: Large language models (LLMs) have recently shown promise for historical entity linking, but preference optimization for this task is often formulated with only one negative candidate per training instance. This discards information from the remaining candidates retrieved for the same mention. We introduce multi-negative direct preference optimisation (MDPO), a reference-based pairwise objective that compares the correct entity with all valid rejected candidates associated with each mention. MDPO preserves the Bradley-Terry formulation of DPO while exploiting the complete candidate set through masked, length-normalised sequence scores. We evaluate MDPO on hipe-2020 and newseye, covering French, German, English, Swedish, and Finnish historical newspaper text. Experiments show that MDPO improves over supervised fine-tuning and single-negative DPO, with particularly strong gains for NIL mentions, semantic ambiguity, OCR noise, and historic
    
[^97]: 通过潜空间推理！让潜空间视觉推理成为必需

    Reason Through the Latent! Making Latent Visual Reasoning Necessary

    [https://arxiv.org/abs/2609.06746](https://arxiv.org/abs/2609.06746)

    提出因果视觉循环推理（CVRR）框架，通过在解码前移除视觉状态和多模态KV缓存，迫使循环隐藏状态成为唯一的图像条件信息通路，从而确保潜空间视觉推理真正被模型依赖。

    

    潜空间视觉推理旨在通过隐藏状态计算而非显式的文本思维链来进行多模态推理。然而，视觉信息存在于潜空间状态中并不意味着模型在生成答案时真正依赖该状态，尤其是当其他基于图像的替代路径仍然可用时。我们提出了因果视觉循环推理（CVRR），该方法在保留预训练视觉能力的同时，使循环计算成为预测所必需的基于图像的条件路径。CVRR在预训练视觉语言模型融合图像之后，从问题的隐藏状态初始化循环过程，然后在重复读取相同固定视觉证据的同时反复更新该状态。在解码之前，视觉状态和原始的多模态KV缓存会被移除，从而确保只有最终的循环状态携带基于图像的条件信息。

    arXiv:2609.06746v1 Announce Type: new  Abstract: Latent visual reasoning aims to perform multimodal reasoning through hidden-state computation rather than explicit textual chains of thought. However, visual information being present in a latent state does not imply that the model actually relies on that state when producing its answer, especially when alternative image-conditioned paths remain available. We introduce \textbf{C}ausal \textbf{V}isual \textbf{R}ecurrent \textbf{R}easoning (CVRR), which preserves pretrained visual competence while making recurrent computation the required image-conditioned path to prediction. CVRR initializes recurrence from the question hidden state after the pretrained vision-language model has incorporated the image, then repeatedly updates this state while re-reading the same fixed visual evidence. Before decoding, visual states and the original multimodal KV cache are removed so that only the final recurrent state carries image-conditioned information
    
[^98]: 分数背包问题的基于分组的资源分配模型

    A Group-Based Resource Allocation Model for the Fractional Knapsack Problem

    [https://arxiv.org/abs/2609.06470](https://arxiv.org/abs/2609.06470)

    该论文提出了一种两阶段分组资源分配模型，通过将属性相近的物品分组来缓解Dantzig贪婪规则中因微小扰动导致的分配不稳定问题，并给出了与精确最优解相比的紧损失上界。

    

    为了解决分数背包问题，Dantzig贪婪规则根据物品的价值-成本比对其进行排序。这种排序引入了优先级问题：如果预算在两个比率非常相似的物品之间耗尽，输入的任意微小扰动都可能改变分配结果。为了缓解这一问题，我们引入了一种两阶段规则。我们将半径 $\delta$ 内共享属性的物品进行分组，然后按比率的降序评估这些组，并在不进行进一步排序的情况下分配各组的预算份额。考虑一个具有总容量 $U_G$、单位成本在 $[w^-,w^+]$ 范围内、代表性价值为 $\widehat{v}$ 的组，与精确最优解相比，该组的损失被界定为 $\widehat{v}\, U_G\frac{w^+-w^-}{w^++w^-}+\varepsilon_v U_G$，其中 $\varepsilon_v$ 限制组内部价值的变化。此外，对于任意组大小，该调和因子仍然是紧的。

    arXiv:2609.06470v2 Announce Type: replace-cross  Abstract: To solve the fractional knapsack problem, Dantzig's greedy rule orders items according to their value-to-cost ratio. This ordering introduces priority issues. An arbitrarily small perturbation to the input can change the allocation if the budget is exhausted between two items with very similar ratios. To mitigate that problem, we introduce a two-stage rule. We group items sharing attributes within a radius $\delta$. These groups are then evaluated in descending order of ratio, and divide their group's budget share without further ranking. Consider a group featuring an aggregate capacity $U_G$, unit costs contained in $[w^-,w^+]$, and a representative value $\widehat{v}$. The group's loss compared to the exact optimum is bounded by $\widehat{v}\, U_G\frac{w^+-w^-}{w^++w^-}+\varepsilon_v U_G$, in which $\varepsilon_v$ limits the group's internal value variation. Moreover, for any group size, this harmonic factor remains tight. Th
    
[^99]: 面向内存高效MoE推理的缓存感知联合路由器自适应

    Cache-Aware Joint Router Adaptation for Memory-Efficient MoE Inference

    [https://arxiv.org/abs/2609.04895](https://arxiv.org/abs/2609.04895)

    提出一种缓存感知的后训练框架，通过联合自适应MoE主干与轻量级时间-空间缓存路由器，在不改变原生Top-K专家选择规则的前提下提升缓存命中率、减少专家权重传输，实现内存高效的MoE推理。

    

    混合专家模型每个token仅激活一小部分专家，但完整的专家集合通常超出GPU内存容量，导致解码过程中反复的权重传输。我们将专家缓存管理形式化为一个模型侧的算法问题，并提出一种缓存感知的后训练框架，该框架联合调整MoE主干网络和轻量级辅助缓存路由器，同时在推理时保留原生的Top-K专家选择规则。其仅更新模式——时间路由器，可预测同层复用并为未来token保留专家，无需主动加载。完整的时空路由器在此基础上增加了空间路由器，利用因果前驱token的隐藏状态在访问目标层之前细化时间缓存。我们在Qwen3和GPT-OSS上，通过GSM8K、MATH和CommonsenseQA对两种模式进行了评估。与匹配的纯语言模型基线相比，时间路由器持续提升缓存命中率并减少专家权重传输流量。在Qwen（摘要截断）

    arXiv:2609.04895v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) models activate only a small subset of experts per token, but the full expert set often exceeds GPU memory, causing repeated weight transfers during decoding. We formulate expert-cache management as a model-side algorithmic problem and propose a cache-aware post-training framework that jointly adapts the MoE backbone and lightweight auxiliary cache routers while preserving the native Top-K expert-selection rule at inference. Its update-only mode, Temporal Router, predicts same-layer reuse and retains experts for future tokens without proactive loading. The full Spatio-Temporal Router adds a Spatio Router that uses the causal predecessor's hidden state to refine the temporal cache before target-layer access. We evaluate both modes on Qwen3 and GPT-OSS across GSM8K, MATH, and CommonsenseQA. Temporal Router consistently improves cache hit rate and reduces expert-weight traffic over matched LM-only baselines. On Qwen
    
[^100]: Harbor 适配器与 Harbor-Index：面向大规模智能体评估的基础设施与精选元数据集

    Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation

    [https://arxiv.org/abs/2609.04298](https://arxiv.org/abs/2609.04298)

    本文提出了 Harbor Adapters 统一评估基础设施，将 80 多个智能体基准测试移植为可评估任意智能体的形式，并据此对 8 个模型进行大规模评估，同时推出经 AI 与人工双重审核筛选的包含 82 个高质量难题的精选元数据集 Harbor-Index。

    

    在数量不断增长的智能体（agentic）基准测试上评估智能体是一项挑战，因为这些基准通常需要复杂的环境和智能体集成方案。我们提出了 Harbor Adapters，一个面向智能体基准测试的统一评估基础设施。我们的工作包含三项贡献。第一，我们开发了一系列基准适配器，将 80 多个基准测试移植为可评估任意智能体的形式，并通过严格的代码审查和一致性实验对其进行了验证。第二，我们在 54 个基准测试上对横跨不同能力层级的 8 个模型进行了大规模评估；每个模型均使用 Terminus-2 以及 3 个原生测试框架之一运行。这使得对智能体能力和失败模式的更广泛分析成为可能。第三，我们推出了 Harbor-Index，一个精心策划的包含 82 个高难度、多样化且高质量任务的集合，覆盖 29 个基准测试，它是在适配后的基准套件基础上，通过难度筛选、AI 与人工审核以及审核-修复循环精炼而成。

    arXiv:2609.04298v1 Announce Type: new  Abstract: Evaluating agents on the growing number of agentic benchmarks is challenging because they often require complex environments and agent integrations. We introduce Harbor Adapters, a unified evaluation infrastructure for agentic benchmarks. Our work makes three contributions. First, we develop benchmark adapters that port more than 80 benchmarks to evaluate arbitrary agents, and validate them through rigorous code review and parity experiments. Second, we conduct a large-scale evaluation of 8 models spanning capability tiers across 54 benchmarks; every model is run with Terminus-2 and with one of 3 native harnesses. This enables a broader analysis of agent capabilities and failure modes than was previously possible. Third, we introduce Harbor-Index, a curated set of 82 difficult, diverse, and high-quality tasks spanning 29 benchmarks, refined from the adapted suite through difficulty filtering, AI and human audit, and an audit-and-fix loop
    
[^101]: OUTLETS：基于投机解码主干的输出长度预测

    OUTLETS: Output-Length Prediction from Speculative Decoding Backbones

    [https://arxiv.org/abs/2609.01068](https://arxiv.org/abs/2609.01068)

    该论文发现投机解码框架中草稿解码器的潜在表示蕴含着可预测生成长度的信号，并提出OUTLETS方法，将投机解码主干重新用作轨迹感知的输出长度预测器，从而在几乎不增加额外开销的情况下改进大语言模型服务的资源供给与集群调度。

    

    大语言模型（LLM）服务中输出长度的重尾分布给资源供给和集群调度带来了重大挑战。虽然输出长度预测可以缓解这些问题，但现有方法存在关键缺陷：外部代理模型会增加大量延迟且保真度通常有限，而基于内部状态的方法虽然高效，但仅依赖于对当前模型状态的浅层探测。我们发现了投机解码（SD）与长度预测之间的结构性联系：在先进框架（如 EAGLE-3）中，草稿解码器产生的潜在表示编码了能够预测生成长度的信号。基于这一洞察，我们提出了 OUTLETS（基于投机解码主干的输出长度预测），它将投机解码主干重新用作轨迹感知的长度预测器。当其草稿表示已经为投机解码计算完成时……（摘要在此处被截断）

    arXiv:2609.01068v1 Announce Type: new  Abstract: The heavy-tailed distribution of output lengths in Large Language Model (LLM) serving poses major challenges for resource provisioning and cluster scheduling. Although output-length prediction can mitigate these issues, existing approaches have key drawbacks: external proxy models add substantial latency and often have limited fidelity, whereas internal state-based methods are efficient but rely on shallow probes of current model states. We identify a structural connection between speculative decoding (SD) and length prediction: latent representations produced by the draft decoder in advanced frameworks (e.g., EAGLE-3) encode signals that are predictive of generation length. Building on this insight, we introduce OUTLETS (Output-Length Prediction from Speculative Decoding Backbones), which repurposes the speculative backbone as a trajectory-aware length predictor. When its draft representations are already computed for speculative decodi
    
[^102]: 见好就收：用于机器翻译重排序中高效候选生成的Quit方法

    Quit While You're Ahead: Quit for Efficient Candidate Generation in Machine Translation Reranking

    [https://arxiv.org/abs/2609.00588](https://arxiv.org/abs/2609.00588)

    提出Quit方法，通过不确定性量化的早停策略对机器翻译的整个候选生成—重排序流程进行增量式生成与重排序，在最高候选质量稳定时提前终止，从而在保持翻译质量的同时显著降低推理延迟。

    

    重排序方法，如最小贝叶斯风险（MBR）解码和质量估计（QE）重排序，被广泛应用于现代神经机器翻译（NMT）中，用于从一组候选假设中选出最终输出。然而，这些性能提升是以高推理延迟为代价的。现有的加速方法仅针对MBR解码且只减少重排序计算，既未解决QE重排序的问题，也基本未触及候选生成——而后者可能是更大的计算瓶颈。在本工作中，我们提出了Quit（基于不确定性量化的增量终止），这是一种针对整个“生成—重排序”流程的新型早停策略。Quit将候选生成视为不确定性下的序列决策过程，增量式地生成并重排序候选译文，当候选集中最高的估计质量趋于稳定时即停止生成。在三个NMT模型、19个语言对上的全面实验表明……

    arXiv:2609.00588v1 Announce Type: new  Abstract: Reranking methods, such as Minimum Bayes Risk (MBR) decoding and Quality Estimation (QE) reranking, are widely used in modern neural machine translation (NMT) to select an output from a set of candidate hypotheses. However, the performance gains come at the cost of high inference latency. Existing acceleration methods target MBR decoding and reduce only reranking computation, leaving QE reranking unaddressed and candidate generation---which can be the larger computational bottleneck---largely untouched. In this work, we propose Quit (Quantifying Uncertainty for Incremental Termination), a novel early-stopping strategy for the entire generation--reranking pipeline. Viewing candidate generation as a sequential decision under uncertainty, Quit incrementally generates and reranks candidates, stopping when the highest estimated quality in the candidate set stabilizes. Comprehensive experiments on three NMT models across 19 language pairs show
    
[^103]: 学习保留什么：面向多智能体大语言模型系统高效协作的门控记忆路由

    Learning What to Retain: Gated-Memory Routing for Efficient Collaboration in Multi-Agent LLM Systems

    [https://arxiv.org/abs/2609.00237](https://arxiv.org/abs/2609.00237)

    提出门控记忆路由方法，通过可学习的记忆写入门和检索门维护紧凑的执行记忆，使多智能体LLM系统的编排决策能依据有用的中间进展而非完整历史，在提升准确性的同时降低成本。

    

    基于大语言模型（LLM）的多智能体系统通过编排多个智能体的配置方式和协作方式来解决复杂推理任务。一个核心挑战是使编排能够适应不断演变的协作状态。仅基于查询的路由无法适应中间过程的进展或错误，从而损害准确性；而基于完整执行历史的路由虽然补足了缺失的上下文，却迫使后续决策处理所有先前步骤，包括冗余或低效用的步骤，造成执行历史过载并推高成本。有效的编排实际上需要一个紧凑的状态，既能捕获有用的进展，又不会积累冗余上下文。我们提出门控记忆路由，将每个决策基于查询和一个学习到的执行记忆进行条件化。一个学习到的记忆写入门仅提交非冗余的推理步骤，一个学习到的检索门为每个智能体提供紧凑且相关的信息。

    arXiv:2609.00237v1 Announce Type: new  Abstract: Large language model (LLM)-based multi-agent systems tackle complex reasoning by orchestrating how multiple agents are configured and how they collaborate. A central challenge is to adapt orchestration to the evolving collaboration state. Routing from the query alone cannot adapt to intermediate progress or errors, which hurts accuracy. Routing from the complete execution history supplies this missing context, but forces later decisions to process every prior step, including redundant or low-utility ones. This creates an execution-history overload that inflates cost. Effective orchestration instead requires a compact state that captures useful progress without accumulating redundant context. We propose Gated-Memory Routing, which conditions each decision on the query and a learned execution memory. A learned Memory Write Gate commits only non-redundant reasoning steps, and a learned Retrieval Gate supplies each agent a compact, relevant 
    
[^104]: 迈向汉语族语言的跨语言罗马化生态系统：以普通话-粤语配对案例研究为例

    Toward a Cross-Lingual Romanization Ecosystem for Sinitic Languages: A Paired Mandarin-Cantonese Case Study

    [https://arxiv.org/abs/2608.29170](https://arxiv.org/abs/2608.29170)

    本文提出汉语罗马化生态系统框架，基于语音对应、历史音韵对应、一音位一符号和基本拉丁字母使用四项设计原则，开发出普通话与粤语等汉语族语言的配套罗马化方案及开源数字基础设施。

    

    本文提出了汉语罗马化生态系统，这是一个跨语言的汉语族罗马化设计框架，并配有支持性数字基础设施和社区驱动的开源工作流程。该设计框架通过四项设计原则解决了汉语族语言之间缺乏系统性跨语言罗马化对应的问题：语音对应原则（用相似的罗马化符号表示相似的音）、历史音韵对应原则（对齐同源词的罗马化字符串）、一音位一符号原则，以及基本拉丁字母使用原则，同时以平衡考量来权衡这些原则之间的取舍。作为主要的配对案例研究，我们按照该设计框架分别开发了粤语罗马化方案CantRomZJ1和普通话罗马化方案MandRomZJ1。我们还按照相同的框架为其他几种汉语族语言开发了罗马化方案，包括梅县客家话、上海吴语和南京江淮官话。

    arXiv:2608.29170v1 Announce Type: new  Abstract: This paper proposes the Sinitic Romanization Ecosystem, a cross-lingual Sinitic romanization design framework with supporting digital infrastructure and a community-driven open-source workflow. The design framework addresses the lack of systematic cross-lingual romanization alignment among Sinitic languages through four design principles: phonetic correspondence for representing similar sounds with similar romanized symbols, historical-phonological correspondence for aligning cognate romanization strings, one-phoneme-one-symbol, and basic Latin-letter use, with a balancing consideration recognizing trade-offs among these principles. For the main paired case study, we devel-op CantRomZJ1 and MandRomZJ1, Cantonese and Manda-rin romanization schemes following the design framework, respectively. We also develop schemes for several other Sinitic languages, including Meixian Hakka, Shanghai Wu, and Nanjing Jianghuai Mandarin, following the sam
    
[^105]: DelistBench：评估支持搜索的LLM用于可审计的公司事件数据库补全

    DelistBench: Evaluating Search-Enabled LLMs for Auditable Corporate-Event Database Completion

    [https://arxiv.org/abs/2608.22770](https://arxiv.org/abs/2608.22770)

    该论文提出了DelistBench基准和Search-to-Record任务，证明网络搜索能显著提升LLM在公司事件数据库补全中的准确率，且经济型系统能以低成本达到接近最优性能。

    

    arXiv:2608.22770v1 公告类型：新 摘要：金融机构需要一种独立的方法来检测供应商数据库中缺失、过期或错误分类的公司事件记录。我们引入了“搜索到记录”（Search-to-Record）这一数据库保障任务，在该任务中，支持搜索的大型语言模型从公开来源重构机构定义的事件记录，针对已知的证券范围和历史截止日期，并提出了DelistBench，一个包含1,200条记录的证券级别摘牌公告基准。我们评估了五种模型，分别处于配对的无网络和启用网络条件下。网络访问将七天内公告日期准确率提高了34.0至48.0个百分点，事件状态准确率提高了约2.8至21.7个百分点；最佳系统在七天内实现了81.5%的整体联合准确率。经济型网络系统在七天内实现了75.9-78.3%的整体联合准确率，其API成本仅为最昂贵网络系统的4.5-6.6%。基于风险的分诊识别了低错误子集，尽管...

    arXiv:2608.22770v1 Announce Type: new  Abstract: Financial institutions need an independent way to detect missing, stale, and misclassified corporate-event records in vendor databases. We introduce Search-to-Record, a database-assurance task in which search-enabled large language models reconstruct institution-defined event records from public sources for a known security universe and historical cutoff, and DelistBench, a 1,200-record benchmark for security-level delisting announcements. We evaluate five models in paired closed-book and web-enabled conditions. Web access raises announcement-date accuracy within seven days by 34.0 to 48.0 percentage points and event-status accuracy by approximately 2.8 to 21.7 points; the best system achieves 81.5% overall joint accuracy within seven days. Economy web systems achieve 75.9-78.3% overall joint accuracy within seven days at 4.5-6.6% of the API cost of the most expensive web system. Risk-based triage identifies low-error subsets, although t
    
[^106]: 洗白仇恨、污蔑无害内容：针对基于LLM的内容审核的标注者风格反驳攻击

    Whitewashing Hate, Smearing Harmless Content: Annotator-Style Rebuttal Attacks on LLM-Based Moderation

    [https://arxiv.org/abs/2608.22230](https://arxiv.org/abs/2608.22230)

    本研究揭示了标注者风格的反驳攻击能显著破坏LLM仇恨言论审核的准确性，且洗白与污蔑两种操纵方向存在模型特定的不对称效应。

    

    大型语言模型（LLMs）越来越多地被用于仇恨言论审核，通常出现在人类与AI协作的工作流程中，其中审核者在最终决策前提供反馈。这种反馈引入了两种操纵方向：将仇恨内容洗白为正常内容，以及将正常内容污蔑为仇恨内容。本研究考察了初始正确的模型判断对标注者风格反驳的敏感性，并分析了攻击有效性是否因操纵方向而异。我们引入了一种重新判断协议，该协议通过决策边界扰动和对抗性理由扩展了直接矛盾。在多个LLM和两个仇恨言论数据集上的实验表明，标注者风格的反驳显著降低了审核性能，在多轮设置中效果更强。结果进一步揭示了在攻击配置中，洗白和污蔑之间存在稳定且模型特定的不对称性，这表明...

    arXiv:2608.22230v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for hate speech moderation, often within human--AI workflows in which reviewers provide feedback before a final decision. Such feedback introduces two manipulation directions: whitewashing hateful content as normal and smearing normal content as hateful. This study examines the susceptibility of initially correct model judgments to annotator-style rebuttals and analyzes whether attack effectiveness differs across manipulation directions. We introduce a rejudge protocol that extends direct contradiction with decision-boundary perturbations and adversarial rationales. Experiments with multiple LLMs on two hate speech datasets show that annotator-style rebuttals substantially degrade moderation performance, with stronger effects in multi-turn settings. The results further reveal stable, model-specific asymmetries between whitewashing and smearing across attack configurations, indicating dis
    
[^107]: SAC-Copula：通过平滑相关Gumbel场实现扩散语言模型的保质量水印技术

    SAC-Copula: Quality-Preserving Watermarking for Diffusion Language Models via Smooth Correlated Gumbel Fields

    [https://arxiv.org/abs/2608.20839](https://arxiv.org/abs/2608.20839)

    SAC-Copula通过引入基于高斯copula的平滑局部相关Gumbel扰动场，解决了扩散语言模型水印中扰动与解码动态不匹配的问题，实现了生成质量与可检测性的更优平衡。

    

    arXiv:2608.20839v1 公告类型：新 摘要：扩散语言模型（DLMs）的水印技术需要与迭代并行去掩蔽机制兼容，而非自回归解码。现有的基于采样的水印方法通常注入逐位置的独立同分布扰动，这可能与DLM解码动态不匹配，从而降低生成质量。我们提出SAC-Copula，一种基于高斯copula构建的平滑、局部相关Gumbel扰动场的保质量水印方法。我们进一步开发了SAC感知检测器，利用协方差感知滤波和原生样本校准。机制级分析表明，局部相关性降低了潜在扰动的粗糙度，并更好地匹配迭代细化动态。在LLaDA上的实验表明，与现有基线相比，SAC-Copula实现了良好的质量-可检测性权衡。特别是，在Dream-7B及额外数据集上的进一步评估也验证了其有效性。

    arXiv:2608.20839v1 Announce Type: new  Abstract: Watermarking diffusion language models (DLMs) requires mechanisms compatible with iterative parallel unmasking rather than autoregressive decoding. Existing sampling-based watermarking methods typically inject position-wise i.i.d. perturbations, which can be poorly aligned with DLM decoding dynamics and degrade generation quality. We propose SAC-Copula, a quality-preserving watermarking method for DLMs based on smooth, locally correlated Gumbel perturbation fields constructed via a Gaussian copula. We further develop a SAC-aware detector using covariance-aware filtering and native-sample calibration. Mechanism-level analysis shows that local correlation reduces latent perturbation roughness and better matches iterative refinement dynamics. Experiments on LLaDA show that SAC-Copula achieves a favorable quality-detectability trade-off compared with existing baselines. In particular, further evaluations on Dream-7B and additional datasets s
    
[^108]: Aslema在NADI 2026：通过少样本增强进行口语语言理解

    Aslema at NADI 2026: Augmentation through Fewshot for SLU

    [https://arxiv.org/abs/2608.18689](https://arxiv.org/abs/2608.18689)

    本文提出Aslema系统，通过微调优于零样本，并利用大型语言模型生成文化相关的合成数据增强，在NADI 2026任务中在槽位填充上取得第一名。

    

    arXiv:2608.18689v1 公告类型：交叉 摘要：我们介绍了Aslema，这是我们为NADI 2026共享任务5开发的系统，该任务包含两个子任务：意图识别和槽位填充。我们在零样本设置下评估了四种全模态大型语言模型，并将它们与微调模型进行了比较。结果表明，微调始终优于零样本推理。我们进一步探索了合成数据增强，通过使用大型语言模型生成具有文化背景的突尼斯Derja话语，然后通过语音克隆生成合成语音。将这种合成数据纳入后，两个任务的性能均得到提升。我们最终提交的系统基于Qwen3-Omni-30B，并使用原始数据和合成数据的混合进行训练，在开发测试集上实现了86.8%的意图准确率和34.7的词错误率。在官方测试集上，它在槽位填充任务中排名第一（59.5 CoER），在意图识别任务中排名第四（8个团队中，准确率66.1%）。我们发布了实验脚本，并将很快共享合成数据集以支持进一步研究。

    arXiv:2608.18689v1 Announce Type: cross  Abstract: We present Aslema, our system for NADI 2026 Shared Task 5, which consists of two subtasks: intent recognition and slot filling. We evaluate four omni LLMs in a zero-shot setting and compare them with fine-tuned models. Our results show that fine-tuning consistently outperforms zero-shot inference. We further explore synthetic data augmentation by using an LLM to generate culturally grounded Tunisian Derja utterances, followed by voice cloning to generate synthetic speech. Incorporating this synthetic data improves performance on both tasks. Our final submitted system, based on Qwen3-Omni-30B and trained with a mixture of original and synthetic data, achieves 86.8% intent accuracy and 34.7 WER on the devtest split. On the official test set it ranks 1st in slot filling (59.5 CoER) and 4th among 8 teams in intent recognition (66.1% accuracy). We release our experimental scripts and will soon share the synthetic dataset to support further 
    
[^109]: 针对低资源语言仇恨言论检测的LLM高效适配：罗马乌尔都语的比较研究

    Efficient Adaptation of LLMs for Hate Speech Detection in Low-Resource Languages: A Comparative Study on Roman Urdu

    [https://arxiv.org/abs/2608.18142](https://arxiv.org/abs/2608.18142)

    本研究通过LoRA参数高效微调方法，系统比较了多种大型语言模型在罗马乌尔都语低资源环境下的仇恨言论检测性能，展示了PEFT在零样本推理中的优势。

    

    由于注释数据缺乏、语言结构非正式以及标准化语法缺失，在低资源语言中检测仇恨言论颇具挑战性。罗马乌尔都语就是此类挑战的一个典型例子，它在南亚社交媒体上广泛使用，拼写变异大且缺乏上下文一致的规范。本文旨在全面评估大型语言模型在罗马乌尔都语脚本中的仇恨言论检测性能，并采用参数高效微调方法——低秩适配（LoRA）对这些模型进行微调。为评估零样本推理，我们在不同变压器模型（包括Mistral、LLaMA、Falcon和多语言BERT）上将其与PEFT进行基准对比。实验在包含超过72,000条注释的PURUTT数据集（乌尔都语和罗马乌尔都语有毒评论及音译平行语料库）上进行。

    arXiv:2608.18142v1 Announce Type: new  Abstract: It is challenging to detect hate speech in Low Resource Languages (LRLs) because of the absence of annotated data, the informality of its language structure, and the lack of standardized grammar. A good example of such a challenge is Roman Urdu which is broadly used by South Asians on social media and has a high variation while lacking contextually consistent spellings. The objective of this paper is to conduct a comprehensive assessment of Large Language Models (LLMs) for Hate Speech Detection (HSD) in Roman Urdu script and fine-tune these models using the Parameter-Efficient Fine-Tuning (PEFT) method called Low-Rank Adaptation (LoRA). To evaluate zero-shot inference, we benchmarked it against PEFT on different transformer models, including Mistral, LLaMA, Falcon, and multilingual BERT. Experiments are conducted on the PURUTT (Parallel Urdu and Roman Urdu Corpus for Toxic Comments and Transliteration) dataset with over 72,000 annotated 
    
[^110]: 未适配的多语言ASR在Garrusi库尔德语评估集上的表现：一种共同参考分阶段归一化分析

    Unadapted Multilingual ASR on a Garrusi Kurdish Evaluation Set: A Common-Reference Staged Normalization Analysis

    [https://arxiv.org/abs/2608.16379](https://arxiv.org/abs/2608.16379)

    本文通过共同参考分阶段归一化方法，揭示了未适配的多语言ASR在Garrusi库尔德语上因书写系统差异导致的严重高估错误率，并提供了更公平的评估基准。

    

    arXiv:2608.16379v1 公告类型：新 摘要：评估一种用拉丁田野正字法书写的库尔德语变体的语音识别，而模型输出的是阿拉伯文字，这首先就造成了测量问题而非建模问题：直接评分将书写系统差异视为识别错误。联合归一化参考和假设可以避免这个问题，但也会改变参考分词，将一致性提升与评分分母的变化混在一起。我使用MMS-1B-all模型及其Central Kurdish (ckb)适配器（未做调整，按发布原样使用），在来自五位说话者的1,722个Garrusi问卷片段（9,763个参考词符；117.9分钟）上进行评估。我采用共同参考设计：参考被折叠一次并固定为9,763个词符，而只有假设表示会变化。原始的阿拉伯文字假设得分为111.70%的词错误率（WER）和100.92%的字符错误率（CER），且精确词匹配为零。拉丁转写得到102.36%的WER和57.89%的CER；将其折叠进参考的...

    arXiv:2608.16379v1 Announce Type: new  Abstract: Evaluating speech recognition for a Kurdish variety written in a Latin field orthography, using a model that outputs Arabic script, creates a measurement problem before a modelling one: direct scoring treats writing-system differences as recognition errors. Jointly normalizing reference and hypothesis avoids this, but also changes reference tokenization, mixing agreement gains with a change in the scoring denominator. I evaluate MMS-1B-all with the Central Kurdish (ckb) adapter, used as released without adaptation, on 1,722 Garrusi questionnaire segments from five speakers (9,763 reference word tokens; 117.9 minutes). I use a common-reference design: the reference is folded once and fixed at 9,763 tokens, while only the hypothesis representation varies. The raw Arabic-script hypothesis scores 111.70% WER and 100.92% CER, with zero exact word matches. Latin transliteration gives 102.36% WER and 57.89% CER; folding it into the reference's 
    
[^111]: 自我进化的具身智能体通过技能利用进化

    Self-Evolving Embodied Agents via Skill-Harness Evolution

    [https://arxiv.org/abs/2608.11350](https://arxiv.org/abs/2608.11350)

    SHAPER提出了一种免训练的自我进化框架，通过冻结模型参数并进化外部技能和代码框架，使具身智能体适应新环境，无需额外数据或训练。

    

    arXiv:2608.11350v1 公告类型：新 摘要：具身智能体越来越多地被构建为围绕基础模型的系统，其性能不仅取决于模型权重，还取决于模型周围的技能、上下文、动作接口和执行框架。虽然监督微调和强化学习可以使智能体适应新环境，但它们需要额外的数据、奖励和训练运行；同时，许多免训练、以代码为中心的方法依赖于可编程的机器人API，这些API在固定接口设置中可能不可用。我们提出了SHAPER，一种用于免训练具身适应的自我进化框架，它保持模型参数冻结，并通过目标环境回滚来进化可重用的技能和上下文代码框架，从而改进非参数智能体系统。在SHAPER中，同一个冻结模型既可以作为规划器，也可以作为优化器，无需参数更新即可完善其外部技能和上下文代码框架。我们在VLABench上评估了SHAPER。

    arXiv:2608.11350v1 Announce Type: new  Abstract: Embodied agents are increasingly built as systems around foundation models, where performance depends not only on model weights but also on the skills, context, action interfaces, and execution harness surrounding the model. While supervised fine-tuning and reinforcement learning can adapt agents to new environments, they require additional data, rewards, and training runs; meanwhile, many train-free code-centric approaches rely on programmable robot APIs that may be unavailable in fixed-interface settings. We propose SHAPER, a self-evolving framework for train-free embodied adaptation that keeps model parameters frozen and improves the non-parametric agent system by evolving reusable skills and a context-code harness through target-environment rollouts. In SHAPER, the same frozen model can serve as both planner and optimizer, refining its external skills and context-code harness without parameter updates. We evaluate SHAPER on VLABench 
    
[^112]: VectraYX-Vision-1B：一个具有结构化视觉推理与原生工具使用能力的、参数量小于2B的西班牙语/拉美地区网络安全视觉语言模型

    VectraYX-Vision-1B: A Sub-2B Spanish/LATAM Cybersecurity Vision-Language Model with Structured Visual Reasoning and Native Tool Use

    [https://arxiv.org/abs/2608.08477](https://arxiv.org/abs/2608.08477)

    v3版本将原生训练的Qwen2-VL视觉塔移植到同一冻结解码器上，使原本失败的8半字节地址字段精确率从0.00跃升至0.81，且token预算比2x2平铺更粗糙，证明分辨率并非关键变量。

    

    arXiv:2608.08477v3 公告类型：替换 摘要：25页，1幅图，10个表格。v3版本：将一个原生训练的视觉塔（Qwen2-VL）移植到相同的冻结解码器上，使原本完全失败的8个半字节（nibble）地址字段的精确匹配率从0.00提升至0.81，且所用的token预算比2x2平铺方案更粗糙，这一结果驳斥了分辨率是关键操作变量的假设。第二个预注册字段被发现存在63%的数据污染，已被降级处理。B6/B7工具识别仍停留在最低水平。代码和模型检查点已发布在Hugging Face上。

    arXiv:2608.08477v3 Announce Type: replace  Abstract: 25 pages, 1 figure, 10 tables. v3: transplanting a natively-trained visual tower (Qwen2-VL) onto the same frozen decoder takes the failing 8-nibble address field from 0.00 to 0.81 exact, at a coarser token budget than 2x2 tiling, refuting resolution as the operative variable. Second pre-registered field found 63% contaminated, demoted. B6/B7 tool-id remains at floor. Code/checkpoints on HF.
    
[^113]: 用于反馈驱动智能体修复的因果情景记忆

    Causal Episodic Memory for Feedback-Driven Agent Repair

    [https://arxiv.org/abs/2608.05906](https://arxiv.org/abs/2608.05906)

    MERIT是一种无需训练的智能体，通过维护经oracle验证的成功修正与失败方向的在线双极性情景记忆，在不更新参数的情况下提升后续Text-to-SQL修复任务的执行准确率（Spider从66.34%升至69.79%，BIRD从47.35%升至48.44%）。

    

    修复失败的LLM智能体常常丢弃成功的修正结果，迫使后续回合重新发现相似的解决方案。我们研究最终确定的修复结果能否在不更新参数的情况下改善后续的Text-to-SQL任务回合。我们提出MERIT，一个无需训练的智能体，它维护一个在线双极性记忆，存储经oracle验证的修正以及观察到的不成功修复方向。在oracle辅助的基准测试反馈下，只有来自早期已完成回合的记忆才有资格被检索。一个确定性分类器分配粗粒度的失败类型，在冻结模型生成每次修订之前，该失败类型用于对混合词法-稠密检索器进行条件约束。使用Qwen2.5-7B-Instruct，在相同的初始预测和修复预算下，MERIT相比无状态迭代修复，将Spider上的执行准确率从66.34%提升至69.79%，将BIRD上的执行准确率从47.35%提升至48.44%。配对分析提供了清晰的证据。

    arXiv:2608.05906v2 Announce Type: replace  Abstract: LLM agents that repair failures often discard successful corrections, forcing later episodes to rediscover similar solutions. We study whether finalized repair outcomes can improve subsequent Text-to-SQL episodes without parameter updates. We introduce MERIT, a training-free agent that maintains an online dual-polarity memory of oracle-verified corrections and observed unsuccessful directions. Under oracle-assisted benchmark feedback, only memories from earlier finalized episodes are eligible for retrieval. A deterministic classifier assigns a coarse failure type, which conditions a hybrid lexical-dense retriever before the frozen model generates each revision. Using Qwen2.5-7B-Instruct with identical initial predictions and repair budgets, \method{} improves execution accuracy over stateless iterative repair from \(66.34\%\) to \(69.79\%\) on Spider and from \(47.35\%\) to \(48.44\%\) on BIRD. Paired analyses provide clear evidence 
    
[^114]: 基于强化学习的渐进式智能体技能生成

    Progressive Agent Skill Generation via Reinforcement Learning

    [https://arxiv.org/abs/2608.01678](https://arxiv.org/abs/2608.01678)

    提出Skill-α，一种基于强化学习的渐进式智能体技能生成方法，通过以智能体在下游任务中的表现为奖励信号，学习统一策略从异构来源自动生成高质量技能。

    

    近期的大语言模型智能体通常使用外部技能作为模块化的程序单元，用于条件化推理并提升复杂任务的求解能力。因此，从文档或经验中自动生成高质量技能已成为一个重要问题。现有的技能生成方法主要依赖启发式规则或流水线式的整合方式，这些方法必须针对不同的证据来源进行专门设计。相比之下，基于学习的方法为跨异构来源的技能生成建模提供了更统一的方式。然而，基于学习的技能生成仍然具有挑战性，因为技能缺乏基于相关性或正确性的天然监督信号，其价值在很大程度上只能通过是否能改善智能体在下游任务中的行为来确定。为应对这一挑战，我们提出了Skill-α，一种通过强化学习来学习统一策略以进行渐进式技能生成的方法。

    arXiv:2608.01678v2 Announce Type: replace-cross  Abstract: Recent large language model agents often use external skills as modular procedural units that condition inference and improve complex task solving. Thus, automatically generating high-quality skills from documents or experience has become an important problem. Existing skill generation methods largely rely on heuristics or pipeline-style consolidation, which must be specially designed for different evidence sources. In contrast, learning-based approaches offer a more unified way to model skill generation across heterogeneous sources. However, learning-based skill generation remains challenging because skills lack a natural supervision signal based on relevance or correctness; their value can largely be determined only by whether they improve the behavior of the agent on downstream tasks. To address this challenge, we proposeSkill-$\alpha$, a reinforcement learning method that learns a unified policy for progressive skill genera
    
[^115]: 基于文本描述预测初创企业退出——一个计算语言学框架

    Predicting Startup Exit from Textual Descriptors - A Computational Linguistics Framework

    [https://arxiv.org/abs/2608.00045](https://arxiv.org/abs/2608.00045)

    仅凭文本描述中的语言特征（如形容词、术语和流行语等炒作标记）即可预测初创企业能否成功退出，无需依赖财务或人力资本数据，其中炒作标记的优化密度与更高的退出概率正相关。

    

    本研究证明，仅凭文本描述即可预测早期初创企业的成功（定义为“退出”），而无需依赖情境、财务或人力资本变量。研究使用涵盖20年间7,419家初创企业的风险投资精选数据集，分离出基于文本的框架变量，并通过初创企业叙事映射构建了850个特征。研究对数据子集和向量嵌入进行了统计显著性评估，随后在六种模型上开展了有监督机器学习实验。LightGBM取得了最高的预测性能（F1 = 0.48），而仅使用文本描述也达到了F1 = 0.30，证实了创始人叙事的独立预测价值。特征分析显示，炒作标记（包括形容词、行业术语和流行语）的优化密度与更高的退出概率相关，而过长的陈述或公司名称长度则会降低退出概率。该研究还引入了一个量化……

    arXiv:2608.00045v3 Announce Type: replace  Abstract: This study shows that textual descriptors alone can predict early-stage startup success, defined as Exit, without relying on contextual, financial, or human capital variables. Using venture capital-curated datasets covering 7,419 startups over 20 years, the research isolates text-based framing variables and engineers 850 features through startup narrative mapping. Data subsets and vector embeddings are evaluated for statistical significance, followed by supervised machine learning experiments across six models. LightGBM achieved the highest predictive performance (F1 = 0.48), while textual descriptors alone achieved F1 = 0.30, confirming the standalone predictive value of founder narratives. Feature analysis shows that optimized densities of hyping markers, including adjectives, jargon, and buzzwords, are associated with higher Exit probability, whereas excessive statement or name length reduces it. The study also introduces a quanti
    
[^116]: 共情框架：评估人工智能跨社会人口群体的对齐性

    Sympathetic Framing: Evaluating AI Alignment across Sociodemographic Groups

    [https://arxiv.org/abs/2607.27232](https://arxiv.org/abs/2607.27232)

    该研究通过对3011名英国成年人的YouGov调查与七个大语言模型的对比实验，首次实证评估了LLMs在感知新闻标题情感框架（对冲突各方的同情）方面与人类的对齐程度，发现领先模型（如GPT-5.2相关性达0.789）在所有人口统计子群体中都与人类情感判断高度一致。

    

    大语言模型（LLMs）日益影响着我们获取信息和形成世界观的方式，这引发了超越人工智能偏见的担忧：大语言模型能否理解通过文本框架传达的情感细微差别？在这项工作中，我们实证评估了多种大语言模型与人类情感感知的对齐程度。针对涉及政治和地缘政治冲突的新闻标题，人类参与者（n = 3011，通过YouGov调查获得的英国成年人代表性样本）和七个大语言模型回答了标题是否引起了对冲突中某一特定方的同情。我们发现人工智能与人类评估之间的相关性因模型而异，从非常高（0.789，GPT-5.2）到中等（0.4，Mistral Large 2512）不等。至关重要的是，领先的模型在所有人口统计子群体（包括年龄、性别、教育水平、先天地缘政治知识等）中都与人类判断基本保持一致。

    arXiv:2607.27232v2 Announce Type: replace  Abstract: Large Language Models (LLMs) are increasingly shaping how we consume information and form our worldview. This raises concerns beyond bias in AI: do LLMs grasp the emotional nuances conveyed via textual framing? In this work, we empirically evaluate how well an array of LLMs aligns with human emotional perception. Considering news headlines covering political and geopolitical conflicts, both human participants (n = 3011, a representative sample of the U.K. adult population, via a YouGov survey) and seven LLMs answered whether headlines evoked sympathy for a specified side in a conflict. We find that the correlation between AI and human evaluations varies across models, ranging from very high (0.789, GPT-5.2) to medium (0.4 ,Mistral Large 2512). Crucially, the leading models are broadly aligned with human judgments across all demographic subgroups, including age, gender, level of education, prior geopolitical knowledge, and participant
    
[^117]: 基于语法书的低资源机器翻译合成数据生成的析因研究

    A Factorial Study of Synthetic Data Generation for Low-Resource Machine Translation using Grammar Books

    [https://arxiv.org/abs/2607.22376](https://arxiv.org/abs/2607.22376)

    该论文提出利用大语言模型从语法书中提取语法规则、例句和词汇生成合成平行语料用于模型微调（而非推理时提示），在卡兰芒语、图阿茨钦语和曼丹语三种低资源语言上验证了其有效性，并通过96种配置的析因分析找出了驱动翻译性能提升的关键因素组合。

    

    大多数濒危语言缺乏机器翻译所需的平行数据，尽管描述性语法书是存在的。我们引入了一个流水线，利用大语言模型从语法书中提取语法规则、例句和词汇，并生成用于微调的合成平行语料库——而不是像之前的工作那样在推理时将语法内容输入提示中。我们在三种类型学上多样的低资源语言上进行了验证——卡兰芒语、图阿茨钦语（罗曼语族）和曼丹语（苏语族）——结果表明，在合成数据上进行微调在75%的卡兰芒语配置和59%的图阿茨钦语配置中优于种子数据基线，ChrF++的最佳增益分别为+8.8、+5.3和+3.3。通过对涉及目标词性、检索粒度和样本量的96种配置进行系统的析因研究，我们确定了哪些因素组合能够带来收益以及它们在何处失效。

    arXiv:2607.22376v2 Announce Type: replace  Abstract: Most endangered languages lack the parallel data required for machine translation, despite the existence of descriptive grammar books. We introduce a pipeline that uses large language models to extract grammatical rules, example sentences, and lexicons from grammar books and generate synthetic parallel corpora for fine-tuning-rather than feeding grammar content into prompts at inference time, as in prior work. Validated on three typologically diverse low-resource languages-Kalamang (Papuan), Tuatschin (Romance), and Mandan (Siouan)-we show that fine-tuning on synthetic data improves over seed-data baselines in 75% of configurations for Kalamang and 59% for Tuatschin, with best-case ChrF++ gains of +8.8, +5.3, and +3.3 respectively. Through a systematic factorial study across 96 configurations varying target part-of-speech, retrieval granularity, and sample volume, we identify which factor combinations drive gains and where they break
    
[^118]: DiaLLM：英语方言适应中鲁棒性与生成能力差距的探究

    DiaLLM: An Investigation into the Robustness-Generation Gap in English Dialect Adaptation

    [https://arxiv.org/abs/2607.07669](https://arxiv.org/abs/2607.07669)

    本文发现方言理解与生成能力在LLM中分离，并证明显式方言定向适应优于广泛对齐，但基准测试无法反映这一生成优势。

    

    大语言模型越来越能“理解”方言英语，但仍只能“生成”标准且偏向美式的英语，导致方言生成这一更难的问题在很大程度上未被解决。我们引入了DiaLLM，该方法对三个开放权重语言模型家族在国际英语语料库上进行持续预训练，并应用隐式和显式后训练范式，每种范式结合三种模型对齐策略，首次对这些组件在澳大利亚、印度和北英格兰英语上的表现进行了受控比较。我们的结果显示，方言鲁棒性和生成能力是“分离”的：基准测试受持续预训练和SFT影响，而对齐则显著重塑生成方式，但基准测试无法捕捉这些变化。显式针对特定变体的适应能产生可靠被识别为方言的输出，且优于广泛对齐，但该方法在基准测试中的表现却未体现其优势。

    arXiv:2607.07669v2 Announce Type: replace-cross  Abstract: Large language models increasingly \emph{understand} dialectal English, yet still \emph{produce} only standard, US-leaning English, leaving dialectal generation, the harder half of the problem, largely unaddressed. We introduce \textbf{DiaLLM}, which continually pretrains three open-weight language model families on the International Corpus of English and applies implicit and explicit post-training paradigms, each combined with three model alignment strategies, giving the first controlled comparison of these components across Australian, Indian, and Northern British English. Our results reveal that dialectal robustness and generation are \emph{dissociated}: benchmarks are shaped by continual pretraining and SFT, while alignment visibly reshapes generation in ways benchmarks do not capture. Explicit variety-targeted adaptation produces output reliably recognised as dialectal and preferred over broad alignment, yet the method tha
    
[^119]: LLM-意识形态可塑性：将大语言模型政治行为中的意识形态可塑性测量为语境条件分布

    LLM-Ideoplasticity: Measuring Ideological Plasticity in the Political Behavior of LLMs as a Context-Conditioned Distribution

    [https://arxiv.org/abs/2606.28335](https://arxiv.org/abs/2606.28335)

    大语言模型的政治意识形态不是固定点，而是随语境变化的条件分布——虽然对说服性框架、语言选择等因素高度敏感，但整体上占据的政治光谱范围仅为欧洲主要政党的约三分之一。

    

    我们以系统性实证证据论证，大语言模型的政治意识形态并非一个固定点，而是真实政治空间上的条件分布 P(立场|语境)。我们使用以VAA-CHES投影模型为锚定的统一测量框架评估了九个当前的大语言模型，该框架将模型响应映射到六个语境轴上的三个经过验证的维度（lrgen、lrecon、galtan）。我们的发现揭示了对语境的高度敏感性：说服性框架和代表性不足的语言分别使坐标偏移高达0.57和0.52个单位，而思维链推理往往会放大而非抑制改写不稳定性。尽管存在这种局部可塑性，模型群体总体上占据了一个非常狭窄的奥弗顿言论边界，大约仅占欧洲主要政党分布范围的三分之一。在多特质多方法（MTMM）分析的支持下，我们得出结论：单一的点（摘要在此处截断）

    arXiv:2606.28335v3 Announce Type: replace-cross  Abstract: We argue, with systematic empirical evidence, that a large language model's political ideology is not a fixed point, but a conditional distribution $\mathbb{P}($position$\mid$context$)$ over a real political space. We evaluate nine current LLMs using a unified measurement framework anchored by VAA-CHES projection models, which map responses onto three validated dimensions (lrgen, lrecon, galtan) across six contextual axes. Our findings reveal high sensitivity to context: persuasive framing and under-represented languages displace coordinates by up to 0.57 and 0.52 units, respectively, while chain-of-thought reasoning often amplifies rather than dampens paraphrase instability. Despite this local plasticity, the model cohort occupies a remarkably narrow Overton envelope overall, occupying roughly one-third the spread of major European parties. Supported by a multi-trait multi-method (MTMM) analysis, we conclude that a single poin
    
[^120]: 认知数字孪生：模拟心智的AI系统的伦理风险与治理

    Cognitive Digital Twins: Ethical Risks and Governance for AI Systems That Model the Mind

    [https://arxiv.org/abs/2606.23094](https://arxiv.org/abs/2606.23094)

    本文定义了认知数字孪生（CDTs）这一新型AI技术，指出其独特伦理风险，并提出了涵盖权威、自主性、访问与控制、问责制和可用性五个维度的5A治理框架。

    

    随着AI系统变得越来越持久化和个性化，一类我们称之为认知数字孪生（CDTs）的技术成为可能：即对特定个体认知的动态计算表征，通过行为、情境或生理数据进行更新，以建模、预测或模拟该个体的认知，或充当该个体的交流或决策代理。CDTs将认知推理与纵向表征、模拟和代理行动相结合，而现有的针对个人助理、自主智能体、推荐系统和自动决策系统的治理策略仅能部分应对这些问题。本文做出了四项贡献。第一，我们定义了CDTs并将其与相邻系统区分开来。第二，我们引入了一个围绕权威、自主性、访问与控制、问责制和可用性组织的5A治理框架。第三，我们识别了CDT特有的风险（摘要此处不完整）。

    arXiv:2606.23094v2 Announce Type: replace-cross  Abstract: As AI systems become increasingly persistent and personalized, they make possible a class of technologies that we call cognitive digital twins (CDTs): dynamic computational representations of a specific person's cognition, updated from behavioral, contextual, or physiological data in order to model, predict, or simulate that person's cognition, or to act as that person's communicative or decision-making proxy. CDTs combine cognitive inference with longitudinal representation, simulation, and proxy action in ways that existing governance strategies for personal assistants, autonomous agents, recommender systems, and automated decision systems only partially address. This paper makes four contributions. First, we define CDTs and distinguish them from adjacent systems. Second, we introduce a 5A governance framework organized around authority, autonomy, access and control, accountability, and availability. Third, we identify CDT-sp
    
[^121]: 逆向图灵基准：评估语言模型作为人机对话判别者的能力

    Inverse Turing Bench: Evaluating Language Models as Judges of Human vs. AI Dialogue

    [https://arxiv.org/abs/2606.21844](https://arxiv.org/abs/2606.21844)

    提出逆向图灵基准，用于评估大语言模型在多轮对话中区分人类与人工智能的能力，最强模型GPTZero达到89.41%的准确率，并揭示了统计检测方法存在语义盲点、语义方法易受角色扮演提示影响的局限。

    

    随着人工智能系统集成到网络空间中，在对话中区分它们与人类变得越来越重要。我们提出了逆向图灵基准，这是一个用于评估大语言模型及其他模型在多轮文本中区分人类与人工智能能力的基准。该基准提供了一组成对的对话记录，其中一段对话发生在两个人类之间，另一段对话发生在人类与人工智能之间。任务目标是正确识别哪段对话是纯人类对话，哪段是人机对话。我们用该基准对一组初步模型进行了评估，发现GPTZero、Claude Opus-4.6和GPT-5.5取得了最高的准确率，分别为89.41%、77.92%和75.94%。我们的结果表明，统计检测方法存在语义盲点，而语义方法则容易受到角色扮演提示的影响。我们的工作呼应了逆向图灵测试的理念，并推动将人机区分能力确立为一项关键能力。

    arXiv:2606.21844v2 Announce Type: replace  Abstract: As AI systems integrate into online spaces, differentiating them from humans in conversations is increasingly important. We present Inverse Turing Bench, a benchmark that evaluates LLMs and other models on their ability to differentiate humans and AI in multi-turn text. The benchmark provides a collection of paired dialogue transcripts, wherein one dialogue is between two humans and the other is between a human and an AI. The task is to correctly identify which dialogue is human-only vs. human-AI. We evaluated a preliminary set of models against this benchmark, and found that GPTZero, Claude Opus-4.6, and GPT-5.5 achieve the highest accuracy: 89.41%, 77.92%, and 75.94% respectively. Our results suggest that statistical approaches to detection have semantic blind spots, but semantic approaches are susceptible to persona-prompting. Our work speaks to the Inverse Turing Test and motivates human-AI differentiation as a critical capabilit
    
[^122]: 表征网络规模大语言模型预训练数据中的叙事内容

    Characterizing Narrative Content in Web-scale LLM Pretraining Data

    [https://arxiv.org/abs/2606.19468](https://arxiv.org/abs/2606.19468)

    该论文首次对3万亿词元的开放预训练语料库Dolma中的叙事特征进行细粒度研究，通过构建涵盖能动性、场景和事件三大要素的11维叙事分析框架并训练NarraBERT模型，生成了新数据集NarraDolma，揭示出叙事结构可大规模测量，且叙事质量在预训练数据来源、主题和格式间分布不均。

    

    网络规模的大语言模型预训练语料库的叙事构成在很大程度上仍未被探索，尽管叙事是人类交流的一种基本模式。我们对Dolma（一个拥有3万亿词元的开放预训练语料库）中的叙事特征进行了首次细粒度研究。借鉴叙事理论，我们设计了一个涵盖三个核心叙事要素（能动性、场景和事件）的框架，并将其操作化为11个可解释的维度。在筛选并人工标注了400个多样化文本段落后，我们创建了一个包含2.5万个段落的LLM标注数据集，最后，我们微调并验证了NarraBERT——两个基于RoBERTa的细粒度叙事预测模型。我们将NarraBERT应用于1300万个段落，从而生成了一个新数据集NarraDolma。我们发现，叙事结构可以在极其异构的数据中实现大规模测量，且叙事质量在预训练来源、主题和格式之间的分布并不均衡。

    arXiv:2606.19468v2 Announce Type: replace  Abstract: The narrative composition of web-scale LLM pretraining corpora remains largely unexplored, even though narrative is a fundamental mode of human communication. We present the first fine-grained study of narrative features in Dolma, a 3-trillion-token open pretraining corpus. Drawing on narrative theory, we design a framework spanning three core narrative elements (agency, setting, and events) operationalized as 11 interpretable dimensions. After curating and hand-annotating a diverse set of 400 passages, we create an LLM-labeled dataset of 25K passages, and finally, we finetune and validate NarraBERT, two RoBERTa-based models for fine-grained narrative prediction. We apply NarraBERT to 13M passages, resulting in a new dataset, NarraDolma. We find that narrative structure is measurable at scale across extremely heterogeneous data and narrative qualities are unequally distributed across pretraining sources, topics, and formats in ways t
    
[^123]: 策略类型空间

    Strategic Type Spaces

    [https://arxiv.org/abs/2606.08297](https://arxiv.org/abs/2606.08297)

    本文为信息提供了策略基础，证明了不完全信息博弈中存在本质唯一的最小策略类型空间（STS），且其具有可被有限自动机刻画的递归结构。

    

    我们为信息提供了策略基础：在任何给定的不完全信息博弈中，我们定义了策略商作为信息表示，这些表示足以使玩家能够计算对其他玩家的最优反应。我们证明了：1/ 存在一个最小策略商的存在性和本质唯一性，称为策略类型空间（STS），其中类型由临时相关可理性化层级给出，并表示一组对其他玩家类型和自然状态的信念，这些信念使该层级合理化；2/ 最小STS具有递归结构，该结构可由有限自动机刻画。

    arXiv:2606.08297v2 Announce Type: replace-cross  Abstract: We provide a strategic foundation for information: in any given game with incomplete information we define strategic quotients as information representations that are sufficient for players to compute best-responses to other players. We prove 1/ existence and essential uniqueness of a minimal strategic quotient called the Strategic Type Space (STS) in which a type is given by an interim correlated rationalizability hierarchy and represents a set of beliefs over other players' types and nature that rationalize this hierarchy and 2/ that the minimal STS has a recursive structure that is captured by a finite automaton.
    
[^124]: 基于激活的上下文学习主动学习：挑战与见解

    Activation-Based Active Learning for In-Context Learning: Challenges and Insights

    [https://arxiv.org/abs/2606.05134](https://arxiv.org/abs/2606.05134)

    本研究探索了基于MLP激活的深度主动学习方法来优化LLM上下文示例选择，但得出负面结论：激活信号（无论从大规模激活还是前四阶矩角度观察）与示例质量和任务性能的相关性很弱，Spearman相关系数最高仅0.33。

    

    深度主动学习此前已被探索用于大语言模型的上下文样本选择，但尚未有方法利用Transformer激活理解的最新进展。本文中，我们检验了这样一个假设：模型激活可以为优化上下文示例的选择提供细粒度的信号。我们对应用于上下文学习的基于MLP激活的深度主动学习方法进行了全面分析，包括不同的注意力掩码策略如何影响跨多种分类和生成数据集的主动学习，实验使用了Llama-3.2-3B和Qwen2.5-3B两个基础模型。然而，我们发现了一个负面结果：通过大规模激活或前四阶矩的视角观察，MLP和嵌入层的输出与示例质量或任务性能并不相关。具体来说，在我们测试的所有任务和模型中，绝对Spearman相关系数最高仅为0.33。

    arXiv:2606.05134v2 Announce Type: replace  Abstract: Deep active learning has previously been explored for LLM in-context sample selection, but not with methods that utilise recent advances in understanding of transformer activations. In this paper, we test the hypothesis that model activations could provide a fine-grained signal to optimise the selection of in-context examples. We present a comprehensive analysis of MLP activation-based deep active learning methods applied to in-context learning, including how different attention masking strategies impact active learning across diverse classification and generative datasets, using both Llama-3.2-3B and Qwen2.5-3B base models. However, we find a negative result: MLP and embedding layer outputs, viewed through the lenses of massive activations or the first four moments, do not correlate with example quality or task performance. Specifically, the absolute Spearman correlation coefficient is at most 0.33 for all tasks and models we tested
    
[^125]: MERIT：基于评分标准指导训练的专长匹配审稿人分配方法

    MERIT: Matching Expertise via Rubric-Informed Training for Reviewer Assignment

    [https://arxiv.org/abs/2605.27865](https://arxiv.org/abs/2605.27865)

    MERIT提出了一个两阶段框架，通过强化学习训练的审稿人评估器（以评分标准引导的LLM评判作为奖励）将准则级专长匹配转化为可规模化的适配性监督，再将其蒸馏到嵌入检索器中，实现高效且准确的大规模审稿人分配。

    

    arXiv:2605.27865v2 公告类型：替换 摘要：在大规模场景下将投稿与合适的审稿人进行匹配，已成为大型学术会议日益严峻的挑战。然而，现有方法要么依赖粗糙的代理信号，将一般相关性与真正的适配性混为一谈，要么需要昂贵且难以规模化用于训练的人工标注。我们提出了MERIT，这是一个两阶段框架，通过将准则级别的专长匹配转化为可规模化的适配性监督信号来弥合这一差距。在第一阶段，我们通过强化学习训练一个审稿人评估器，使其能够识别论文所需的专业维度，将其与审稿人的既往工作相匹配，并作出适配性判断，其中奖励信号由一个受论文特定专业评分标准引导的LLM评判者提供。在第二阶段，我们将评估器的预测蒸馏到一个基于嵌入的检索器中，以实现高效的大规模审稿人分配。实验表明，我们的40亿参数审稿人评估器性能优于更大的通用（模型）……

    arXiv:2605.27865v2 Announce Type: replace  Abstract: Matching submissions with suitable reviewers at scale is a growing challenge for major venues, yet existing approaches either rely on coarse proxy signals that conflate general relatedness with true suitability, or require expensive human annotations that are difficult to scale for training. We propose MERIT, a two-stage framework that bridges this gap by converting criterion-level expertise matching into scalable suitability supervision. In the first stage, we train a reviewer assessor via reinforcement learning to identify the expertise dimensions a paper requires, match them against the reviewer's prior work, and produce a suitability decision, with rewards provided by an LLM judge guided by paper-specific expertise rubrics. In the second stage, we distill the assessor's predictions into an embedding-based retriever for efficient large-scale assignment. Experiments show that our 4B reviewer assessor outperforms larger general-purp
    
[^126]: 跨语言脑-语言模型对齐具有稳健性，但挑战了层级加工和计算解释

    Cross-lingual brain-language model alignment is robust but challenges hierarchical and computational accounts

    [https://arxiv.org/abs/2605.21049](https://arxiv.org/abs/2605.21049)

    本研究通过普通话、英语和法语的全脑编码模型发现，脑-语言模型对齐虽然稳健且在跨语言和跨模型层间高度稳定，但这种对齐主要源于稳定的词汇-语义对应关系，而非Transformer的层级化计算特性，从而挑战了将脑-模型对齐视为类脑计算证据的传统解释。

    

    脑-语言模型对齐通常被解释为Transformer模型实现了与人脑相似计算的证据。这一观点假设神经预测性反映了大型语言模型（LLM）的内部计算属性，例如层级化的上下文加工、预测编码或表征压缩。另一种可能性是，脑评分主要反映了语言模型与大脑共同拥有的稳定词汇-语义对应关系。本研究使用覆盖普通话、英语和法语三种语言的全脑编码模型对上述两种解释进行了检验。在所有三种语言中，Transformer表征均能显著预测一个分布式网络的活动，该网络涵盖经典语言区、跨模态皮层系统以及皮层下结构。这些空间模式展现出大量跨语言重叠，并在模型各层之间保持显著稳定，为……（摘要原文在此处截断）

    arXiv:2605.21049v2 Announce Type: replace  Abstract: Brain-language model alignment is often interpreted as evidence that transformer models implement computations similar to those of the human brain. This assumes that neural predictivity reflects internal computational properties of large language models (LLMs), such as hierarchical contextual processing, predictive coding, or representational compression. An alternative possibility is that brain scores primarily reflect stable lexical-semantic correspondences shared by language models and the brain. Here we tested these interpretations using whole-brain encoding models across Mandarin, English, and French. Across all three languages, transformer representations significantly predicted activity in a distributed network spanning classical language regions, transmodal cortical systems, and subcortical structures. These spatial patterns showed substantial cross-linguistic overlap and remained remarkably stable across layers, providing li
    
[^127]: 连续扩散在语言建模中与离散扩散具有同等竞争力的扩展性

    Continuous Diffusion Scales Competitively with Discrete Diffusion for Language

    [https://arxiv.org/abs/2605.18530](https://arxiv.org/abs/2605.18530)

    该研究通过将 Plaid 与现代离散扩散语言模型架构对齐构建了 RePlaid，建立了首个可媲美离散 DLM 的连续扩散语言模型缩放定律，证明连续扩散是语言建模中高度竞争且可扩展的方案。

    

    尽管扩散模型近来在语言建模领域引起了广泛关注，但连续扩散的可扩展性似乎不如离散方法。为了挑战这一观点，我们重新审视了基于似然的连续扩散语言模型 Plaid，并通过将 Plaid 的架构与现代离散 DLM 对齐，构建了 RePlaid。在这一统一设置下，我们建立了首个可与离散 DLM 相媲美的连续 DLM 缩放定律：RePlaid 与自回归模型相比仅存在约 $20\times$ 的计算差距，在使用更少参数的情况下优于 Duo，并在过度训练的情形下优于 MDLM。我们将 RePlaid 与近期的连续 DLM 进行了基准测试：在 OpenWebText 数据集上，RePlaid 在连续 DLM 中实现了新的最先进困惑度上界 $22.1$，并具有更优的生成质量。这些结果表明，通过似然训练的连续扩散是一种极具竞争力且可扩展的替代方案。

    arXiv:2605.18530v2 Announce Type: replace  Abstract: While diffusion has drawn considerable recent attention from the language modeling community, continuous diffusion has appeared less scalable than discrete approaches. To challenge this belief we revisit Plaid, a likelihood-based continuous diffusion language model (DLM), and construct RePlaid by aligning the architecture of Plaid with modern discrete DLMs. In this unified setting, we establish the first scaling law for continuous DLMs that rivals discrete DLMs: RePlaid exhibits a compute gap of only $20\times$ compared to autoregressive models, outperforms Duo while using fewer parameters, and outperforms MDLM in the over-trained regime. We benchmark RePlaid against recent continuous DLMs: on OpenWebText, RePlaid achieves a new state-of-the-art PPL bound of $22.1$ among continuous DLMs and superior generation quality. These results suggest that continuous diffusion, when trained via likelihood, is a highly competitive and scalable a
    
[^128]: 通过在线策略优化与蒸馏实现大语言模型长上下文推理的训练配方

    A Recipe for Long-Context Reasoning in Large Language Models via On-Policy Optimization and Distillation

    [https://arxiv.org/abs/2605.12227](https://arxiv.org/abs/2605.12227)

    该论文提出了一种结合GRPO强化学习与在线策略蒸馏教师指导的训练配方，用更强教师的密集token级正则化替代标准参考策略，在过程级监督难以获取时实现大模型长上下文推理的有效对齐。

    

    现有的针对长上下文任务的模型后训练方法面临互补性的局限：（i）监督微调（SFT）提供稳定的监督信号，但存在曝光偏差问题；（ii）组相对策略优化（GRPO）等强化学习方法基于模型自身生成的轨迹进行训练，但在长程信用分配和稀疏奖励方面存在困难；（iii）在线策略蒸馏（OPD）提供密集的token级别指导，但不直接优化任务奖励。我们研究了这些用于长上下文对齐的互补策略，并推导出一个将GRPO与OPD式教师指导相结合的训练配方：学生模型使用结果级别的奖励从自身生成的轨迹中学习，同时由一个更强的教师模型替代标准参考策略，提供密集的token级别正则化。这种方法在过程级别监督难以获取的情况下尤为有用。为支持本研究，我们引入了LongBlock数据集……

    arXiv:2605.12227v3 Announce Type: replace  Abstract: Existing approaches to post-train models for long-context tasks face complementary limitations: (i) supervised fine-tuning (SFT) provides stable supervision but suffers from exposure bias; (ii) reinforcement learning methods such as Group Relative Policy Optimization (GRPO) train on model-generated trajectories but struggle with long-horizon credit assignment and sparse rewards; and (iii) on-policy distillation (OPD) provides dense token-level guidance but does not directly optimize task rewards. We study these complementary strategies for long-context alignment and derive a recipe that combines GRPO with OPD-style teacher guidance: the student learns from its own rollouts using outcome-level rewards, while a stronger teacher provides dense token-level regularization in place of the standard reference policy. This is especially useful when process-level supervision is difficult to obtain. To support this study, we introduce LongBlock
    
[^129]: 时机决定一切：幽默中语义惊奇的时间支架作用

    Timing is Everything: Temporal Scaffolding of Semantic Surprise in Humor

    [https://arxiv.org/abs/2605.00143](https://arxiv.org/abs/2605.00143)

    该论文提出双重预测违背（DPV）框架，并通过分析828场专业中文脱口秀表演发现，以停顿和语义违背峰值为代表的时间特征对观众幽默欣赏度的预测作用远超语义失谐本身，证明“时机”在幽默中起着关键作用。

    

    幽默是一种基本的人类认知现象，人们从预期违背及其消解中获得愉悦，体现了大脑进行预测加工的动态能力。经典的幽默理论强调语义失谐是产生趣味的主要驱动力，却忽视了时间维度的动态作用，尽管喜剧演员们的直觉认为“时机就是一切”。然而，时间结构在多大程度上促进幽默欣赏、以及它与语义内容如何交互作用，目前仍知之甚少。本研究提出了双重预测违背框架，以刻画内容与时机之间的交互关系。通过分析828场专业中文脱口秀表演，我们发现时间特征在预测观众欣赏度方面的作用远超语义失谐。具体而言，我们发现语义违背的峰值比平均失谐水平更为重要，且停顿会系统性地延长语义违背出现之前的时间……

    arXiv:2605.00143v2 Announce Type: replace  Abstract: Humor is a fundamental cognitive phenomenon in which humans derive pleasure from the expectation violations and their resolution, exemplifying the brain's dynamic capacity for predictive processing. Classical humor theories emphasize semantic incongruity as the primary driver of amusement, yet overlook temporal dynamics despite comedians' intuition that "timing is everything." The extent to which temporal structure contributes to humor appreciation and how it interacts with semantic content remains poorly understood. Here, we propose the Dual Prediction Violation (DPV) framework to capture the interplay between content and timing. By analyzing 828 professional Chinese stand-up performances, we show that temporal features substantially outweigh semantic incongruity in predicting audience appreciation. Specifically, we find that peak semantic violations matter more than average incongruity levels, and pauses systematically lengthen bef
    
[^130]: 分析大语言模型推理以揭示心理健康污名化

    Analyzing LLM Reasoning to Uncover Mental Health Stigma

    [https://arxiv.org/abs/2604.25053](https://arxiv.org/abs/2604.25053)

    该论文提出通过分析大语言模型的中间推理步骤，并借助临床专业知识对污名化语言进行分类与严重程度评级，从而揭示多项选择题评估方法无法捕捉的心理健康污名化偏见及其内在逻辑。

    

    arXiv:2604.25053v2 公告类型：替换 摘要：尽管大语言模型（LLM）在心理健康应用领域正被日益广泛地探索，但近期研究表明，它们可能对心理疾病患者表现出污名化倾向。现有的污名化评估主要依赖多项选择题（MCQ），这种方式无法捕捉模型底层逻辑中蕴含的偏见。在本文中，我们通过分析大语言模型的中间推理步骤，来揭示隐藏的污名化语言及其背后的内在理由。我们借助临床专业知识，对针对心理疾病患者的常见污名化语言模式进行分类，并利用这一框架来识别和标记大语言模型推理中的问题性陈述。此外，我们对这些陈述的严重程度进行评级，以区分明显的偏见与更微妙、危害性不那么直接的偏见。为了扩展推理领域并捕捉更广泛的语言模式……

    arXiv:2604.25053v2 Announce Type: replace  Abstract: While large language models (LLMs) are increasingly being explored for mental health applications, recent studies reveal that they can exhibit stigma toward individuals with psychological conditions. Existing evaluations of this stigma primarily rely on multiple-choice questions (MCQs), which fail to capture the biases embedded within the models' underlying logic. In this paper, we analyze the intermediate reasoning steps of LLMs to uncover hidden stigmatizing language and the internal rationales driving it. We leverage clinical expertise to categorize common patterns of stigmatizing language directed at individuals with psychological conditions and use this framework to identify and tag problematic statements in LLM reasoning. Furthermore, we rate the severity of these statements, distinguishing between overt prejudice and more subtle, less immediately harmful biases. To broaden the reasoning domain and capture a wider array of patt
    
[^131]: LLMAR：面向稀疏且富文本工业领域的免调优推荐框架

    LLMAR: A Tuning-Free Recommendation Framework for Sparse and Text-Rich Industrial Domains

    [https://arxiv.org/abs/2604.16379](https://arxiv.org/abs/2604.16379)

    LLMAR提出了一种免调优的推荐框架，通过LLM推理驱动的标注将行为历史转化为结构化语义动机，并结合反思循环机制进行自我纠错，从而在数据稀疏、文本丰富的工业B2B场景中无需训练即可实现高效的推荐。

    

    工业B2B应用（如建筑工地风险预测、物资采购）面临极端的数据稀疏性，但同时又具有丰富的文本交互。在这类环境中，传统的基于ID的协同过滤因缺乏共现信号而失效，而微调标准大语言模型（LLM）会带来高昂的运营成本，且难以应对频繁的数据漂移。我们提出了LLMAR（LLM标注推荐），这是一个免调优的框架。该框架超越了简单的嵌入方法，系统地整合LLM推理来捕捉用户的“潜在动机”，而无需任何训练过程。我们引入了三个核心贡献：（1）推理驱动标注：利用LLM将行为历史转化为结构化的语义动机，实现基于推理的匹配，这是基于ID的方法无法达到的；（2）反思循环：一种自我纠错机制，通过细化生成的查询来减轻幻觉并解决……

    arXiv:2604.16379v3 Announce Type: replace-cross  Abstract: Industrial B2B applications (e.g., construction site risk prediction, material procurement) face extreme data sparsity yet feature rich textual interactions. In such environments, traditional ID-based collaborative filtering fails lacking co-occurrence signals, while fine-tuning standard Large Language Models (LLMs) incurs high operational costs and struggles with frequent data drift.   We propose LLMAR (LLM-Annotated Recommendation), a tuning-free framework. Moving beyond simple embeddings, LLMAR systematically integrates LLM reasoning to capture user "latent motives" without any training process. We introduce three core contributions: (1) Inference-Driven Annotation: uses LLMs to transform behavioral history into structured semantic motives, enabling reasoning-based matching unattainable by ID-based methods; (2) Reflection Loop: a self-correction mechanism that refines generated queries to mitigate hallucinations and resolve 
    
[^132]: 学会像漫画配文作者一样思考：面向多模态幽默理解的不协调性-消解监督

    Learning to Think Like a Cartoon Captionist: Incongruity-Resolution Supervision for Multimodal Humor Understanding

    [https://arxiv.org/abs/2604.15210](https://arxiv.org/abs/2604.15210)

    提出IRS框架，将多模态幽默理解分解为不协调性建模、消解建模和偏好对齐三个环节，通过结构化推理轨迹监督，使模型像专业漫画配文作者一样进行显式推理。

    

    幽默是少数几种推理过程的正确性与答案的正确性同等重要的认知任务之一。尽管近期研究在纽约客漫画配文大赛（NYCC）等基准上评估幽默理解能力，但大多数工作将其视为黑盒预测，忽视了幽默理解背后结构化的推理过程。我们提出了IRS（不协调性-消解监督）框架，该框架将幽默理解分解为三个组成部分：不协调性建模，用于识别视觉场景中的不匹配之处；消解建模，用于对这些不匹配构建连贯的重新解释；偏好对齐，用于依据人类判断评估候选解释。IRS植根于不协调性-消解理论和专业配文作者的实践，通过结构化轨迹监督中间推理过程，使从视觉感知到幽默解释的路径……

    arXiv:2604.15210v2 Announce Type: replace-cross  Abstract: Humor is one of the few cognitive tasks where getting the reasoning right matters as much as getting the answer right. While recent work evaluates humor understanding on benchmarks such as the New Yorker Cartoon Caption Contest (NYCC), it largely treats it as black-box prediction, overlooking the structured reasoning processes underlying humor comprehension. We introduce IRS (Incongruity-Resolution Supervision), a framework that decomposes humor understanding into three components: Incongruity Modeling, which identifies mismatches in the visual scene; Resolution Modeling, which constructs coherent reinterpretations of these mismatches; and Preference Alignment, which evaluates candidate interpretations under human judgments. Grounded in incongruity-resolution theory and expert captionist practice, IRS supervises intermediate reasoning process through structured traces that make the path from visual perception to humorous interp
    
[^133]: 在Lean中通过各向同性直线形式化自对偶码的构建方法

    Formalizing building-up constructions of self-dual codes through isotropic lines in Lean

    [https://arxiv.org/abs/2604.08485](https://arxiv.org/abs/2604.08485)

    本文在Lean证明助手中形式化了自对偶码的构建方法，证明了双坐标约化与Kim构建法的互逆性，并通过各向同性直线发展了q元推广，构造出多种最优自对偶码。

    

    本文的目的有两个方面。首先，我们证明，经过指定的形式等距变换后，Chinburg和Zhang的二进制Hilbert符号实现中的双坐标约化与Kim的构建方法在置换等价意义下互为逆运算。其次，对于 $q\equiv1\pmod4$，我们发展了这种约化-扩展机制的 $q$ 元类似物。恒等式 $c^2=-1$ 给出了控制分裂构造的各向同性直线。对于每个固定的坐标有序配对，我们得到一个通用的秩-$r$ 盒式规范形式，其中 $r$ 是与这些各向同性直线乘积交集的维数。应用包括 $\mathbb{F}_{5}$ 上的最优自对偶 $[6,3,4]$ 和 $[8,4,4]$ 码，$\mathbb{F}_{13}$ 上的最优自对偶 $[8,4,5]$ 和 $[10,5,6]$ 码，以及 $\mathbb{F}_{13}$ 上的自对偶 $[12,6,6]$ 码。我们还给出了自对偶 $[18,9,8]$ 和 $[20,10\cdots$ 码的精确重复盒式实现。

    arXiv:2604.08485v2 Announce Type: replace-cross  Abstract: The purpose of this paper is two-fold. First, we show that, after a specified form isometry, the two-coordinate reduction in the binary Hilbert-symbol realization of Chinburg and Zhang is inverse to Kim's building-up construction, up to permutation equivalence. Second, for $q\equiv1\pmod4$, we develop a $q$-ary analogue of this reduction-and-extension mechanism. The identity $c^2=-1$ yields the isotropic line governing the split construction. For every fixed ordered pairing of the coordinates, we obtain a universal rank-$r$ boxed normal form, where $r$ is the dimension of the intersection with the product of these isotropic lines. Applications include optimal self-dual $[6,3,4]$ and $[8,4,4]$ codes over $\mathbb F_{5}$, optimal self-dual $[8,4,5]$ and $[10,5,6]$ codes over $\mathbb F_{13}$, and a self-dual $[12,6,6]$ code over $\mathbb F_{13}$. We also give an exact repeated boxed realization of self-dual $[18,9,8]$ and $[20,10
    
[^134]: MisEdu-RAG：面向新手数学教师的误解感知双超图检索增强生成框架

    MisEdu-RAG: A Misconception-Aware Dual-Hypergraph RAG for Novice Math Teachers

    [https://arxiv.org/abs/2604.04036](https://arxiv.org/abs/2604.04036)

    提出了面向新手数学教师的误解感知双超图RAG框架MisEdu-RAG，通过将教学知识组织为概念超图、将真实学生错误案例组织为实例超图，并采用两阶段检索，生成基于证据且更具可操作性的教学反馈。

    

    新手数学教师经常会遇到难以诊断和纠正的学生错误，其中误解尤为棘手，因为教师既要解释错在哪里，又要说明如何解决。尽管现有许多大型语言模型（LLM）平台可以辅助生成教学反馈，但这些LLM对教学知识和学生错误之间的关联较为松散，可能导致给出的指导对教师而言缺乏可操作性。为了弥补这一不足，我们提出了MisEdu-RAG，这是一个基于双超图的检索增强生成（RAG）框架，它将教学知识组织为概念超图，将真实的学生错误案例组织为实例超图。给定一个查询，MisEdu-RAG执行两阶段检索，从两个层级收集相互关联的证据，并基于检索到的案例和教学原则生成回答。我们在MisstepMath数据集上进行了评估，该数据集包含数学错误……（原文摘要在此处截断）

    arXiv:2604.04036v2 Announce Type: replace-cross  Abstract: Novice math teachers often encounter students' mistakes that are difficult to diagnose and remediate. Misconceptions are especially challenging because teachers must explain what went wrong and how to solve them. Although many existing large language model (LLM) platforms can assist in generating instructional feedback, these LLMs loosely connect pedagogical knowledge and student mistakes, which might make the guidance less actionable for teachers. To address this gap, we propose MisEdu-RAG, a dual-hypergraph-based retrieval-augmented generation (RAG) framework that organizes pedagogical knowledge as a concept hypergraph and real student mistake cases as an instance hypergraph. Given a query, MisEdu-RAG performs a two-stage retrieval to gather connected evidence from both layers and generates a response grounded in the retrieved cases and pedagogical principles. We evaluate on \textit{MisstepMath}, a dataset of math mistakes pa
    
[^135]: 生成式多智能体系统中的涌现风险

    Emergent Risks in Generative Multi-Agent Systems

    [https://arxiv.org/abs/2603.27771](https://arxiv.org/abs/2603.27771)

    本文开创性地系统研究了生成式多智能体系统中的涌现风险，发现在共享资源竞争、顺序交接协作和集体决策聚合等场景中，有害的群体行为会频繁出现而非罕见特例。

    

    由大型生成式模型组成的多智能体系统正迅速从实验室原型走向实际部署，在这些系统中，多个智能体共同规划、协商并分配共享资源以解决复杂任务。尽管这类系统有望带来前所未有的可扩展性和自主性，但其集体交互也会产生无法归结为单个智能体行为的失效模式。因此，理解这些涌现风险至关重要。在此，我们对涉及共享资源竞争（如计算资源或市场份额）、顺序交接协作（下游智能体只能看到前序智能体的输出）、集体决策聚合等工作流程中的涌现式多智能体风险进行了开创性研究。在所有这些场景中，我们观察到此类群体行为在重复试验和广泛的交互条件下频繁出现，而非罕见或病态的特例。

    arXiv:2603.27771v3 Announce Type: replace-cross  Abstract: Multi-agent systems composed of large generative models are rapidly moving from laboratory prototypes to real-world deployments, where they jointly plan, negotiate, and allocate shared resources to solve complex tasks. While such systems promise unprecedented scalability and autonomy, their collective interaction also gives rise to failure modes that cannot be reduced to individual agents. Understanding these emergent risks is therefore critical. Here, we present a pioneer study of such emergent multi-agent risk in workflows that involve competition over shared resources (e.g., computing resources or market share), sequential handoff collaboration (where downstream agents see only predecessor outputs), collective decision aggregation, and others. Across these settings, we observe that such group behaviors arise frequently across repeated trials and a wide range of interaction conditions, rather than as rare or pathological case
    
[^136]: 对齐减少了表达出的但并未消除编码的性别偏见：一个统一框架与研究

    Alignment Reduces Expressed but Not Encoded Gender Bias: A Unified Framework and Study

    [https://arxiv.org/abs/2603.24125](https://arxiv.org/abs/2603.24125)

    该研究提出一个统一框架来联合分析大语言模型内在与外在的性别偏见，发现对齐虽能减少生成输出中表达出的性别偏见，但并未消除模型内部表征中编码的性别偏见。

    

    在训练过程中，大语言模型（LLMs）会学习到社会规律，这可能导致下游应用中出现性别偏见。大多数缓解工作都集中在减少生成输出中的偏见上，通常在结构化基准上进行评估，这引发了两个担忧：输出层面的评估无法揭示对齐是否改变了模型的底层表征，而结构化基准可能无法反映真实的使用场景。我们提出了一个统一框架，使用相同的无偏见提示来联合分析LLMs中的内在（intrinsic）和外在（extrinsic）性别偏见，从而能够直接比较内部表征中编码的性别相关信息与生成输出中表达出的偏见。与先前报告弱相关或不一致相关性的工作相反，我们发现在统一协议下测量时，潜在性别信息与表达出的偏见之间存在一致的关联。我们进一步研究了对齐（alignment）对……

    arXiv:2603.24125v3 Announce Type: replace  Abstract: During training, Large Language Models (LLMs) learn social regularities that can lead to gender bias in downstream applications. Most mitigation efforts focus on reducing bias in generated outputs, typically evaluated on structured benchmarks, which raises two concerns: output-level evaluation does not reveal whether alignment modifies the model's underlying representations, and structured benchmarks may not reflect realistic usage scenarios. We propose a unified framework to jointly analyze intrinsic and extrinsic gender bias in LLMs using identical neutral prompts, enabling direct comparison between gender-related information encoded in internal representations and bias expressed in generated outputs. Contrary to prior work reporting weak or inconsistent correlations, we find a consistent association between latent gender information and expressed bias when measured under the unified protocol. We further examine the effect of align
    
[^137]: 扰动：一种用于语言模型表示学习的简单高效的对抗性追踪方法

    Perturbation: A simple and efficient adversarial tracer for representation learning in language models

    [https://arxiv.org/abs/2603.23821](https://arxiv.org/abs/2603.23821)

    该论文提出了一种简单高效的对抗性扰动方法来追踪语言模型的表示学习，通过在单个对抗样本上微调模型并测量扰动对其他样本的“感染”程度来揭示表示，该方法无需几何假设，且能避免在未训练模型中产生虚假表示。

    

    深度神经语言模型（LM）中的语言表示学习已被研究数十年，但在语言模型中寻找表示仍然是一个未解决的问题。一方面，无约束的对齐可能会使表示的概念变得平凡化（Sutter等人，2025）；另一方面，即使是最近流行的线性方法也可能不总是忠实于模型的自然行为（Arora等人，2024）。在这里，我们通过将表示重新概念化为学习的传导通道而非激活模式，来摆脱这一困境。我们的方法很简单：我们通过在单个对抗样本上微调语言模型来扰动它，并测量这种扰动如何“感染”其他样本。扰动方法不做任何几何假设，并且与其他方法不同，它不会在不应该存在表示的地方找到表示（例如，在未经训练的语言模型中）。但在经过训练的语言模型中，扰动方法揭示了多个语言粒度上的结构化迁移，表明语言模型……

    arXiv:2603.23821v2 Announce Type: replace  Abstract: Linguistic representation learning in deep neural language models (LMs) has been studied for decades, but finding representations in LMs remains an unsolved problem. On the one hand, unconstrained alignments may trivialize the notion of representation (Sutter et al., 2025); on the other, even recently popularized linear approaches may not always be faithful to natural model behavior (Arora et al. 2024). Here we escape this dilemma by reconceptualizing representations not as patterns of activation but as conduits for learning. Our approach is simple: we perturb an LM by fine-tuning it on a single adversarial example and measure how this perturbation "infects" other examples. Perturbation makes no geometric assumptions, and unlike other methods, it does not find representations where it should not (e.g., in untrained LMs). But in trained LMs, perturbation reveals structured transfer at multiple linguistic grain sizes, suggesting that L
    
[^138]: OpenResearcher：面向长程深度研究轨迹合成的完全开源流水线

    OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis

    [https://arxiv.org/abs/2603.20278](https://arxiv.org/abs/2603.20278)

    OpenResearcher提出了一个完全开源、可复现的离线轨迹合成流水线，在1500万文档语料库上合成超过97K条长程深度研究轨迹，微调后的模型在BrowseComp-Plus上相比基础模型提升34个百分点。

    

    训练深度研究智能体需要长程轨迹，这些轨迹交织着搜索、证据聚合与多步推理。然而，现有的数据收集流水线通常依赖专有的网络API，使得大规模轨迹合成成本高昂、不稳定且难以复现。我们提出了OpenResearcher，一个可复现的流水线，它将一次性的语料库构建与多轮轨迹合成解耦，并使用三个显式的浏览器原语（搜索、打开、查找）在1500万文档的语料库上完全离线地执行搜索与浏览循环。以GPT-OSS-120B作为教师模型，我们合成了超过97K条轨迹，其中包括大量包含100+次工具调用的长程尾部轨迹。在这些轨迹上对30B-A3B骨干模型进行监督微调后，在BrowseComp-Plus基准上达到54.8%的准确率，相比基础模型提升34.0个百分点，同时在BrowseComp、GAIA等基准上保持竞争力（原文此处截断）。

    arXiv:2603.20278v2 Announce Type: replace-cross  Abstract: Training deep research agents requires long-horizon trajectories that interleave search, evidence aggregation, and multi-step reasoning. However, existing data collection pipelines typically rely on proprietary web APIs, making large-scale trajectory synthesis costly, unstable, and difficult to reproduce. We present OpenResearcher, a reproducible pipeline that decouples one-time corpus bootstrapping from multi-turn trajectory synthesis and executes the search-and-browse loop entirely offline using three explicit browser primitives: search, open, and find, over a 15M-document corpus. Using GPT-OSS-120B as the teacher model, we synthesize over 97K trajectories, including a substantial long-horizon tail with 100+ tool calls. Supervised fine-tuning a 30B-A3B backbone on these trajectories achieves 54.8\% accuracy on BrowseComp-Plus, a +34.0 point improvement over the base model, while remaining competitive on BrowseComp, GAIA, and 
    
[^139]: 评估LLM模拟对话在刻画人类社会互动中不一致与非协作行为的表现

    Evaluating LLM-Simulated Conversations in Modeling Inconsistent and Uncollaborative Behaviors in Human Social Interaction

    [https://arxiv.org/abs/2603.17094](https://arxiv.org/abs/2603.17094)

    本文提出CoCoEval框架，通过轮次级检测10类不一致与非协作行为并结合专业场景对话基准，评估LLM模拟对话能否以与人类对话相当的频率再现误解、打断等行为。

    

    使用大语言模型（LLM）模拟人类对话已成为建模人类社会互动的一种可扩展方法。本文重新审视了模拟对话的评估方式，明确认识到人类对话本质上包含不一致与非协作行为，例如误解和打断。由于这些行为是人类社会互动复杂性的重要组成部分，我们认为LLM模拟的对话应以与人类对话观察到的相当频率来再现这些行为。为了支持对这些行为进行细致且可解释的评估，我们提出了CoCoEval框架，该框架包含一个基于轮次级检测10类不一致与非协作行为的评估方案，以及一个用于模拟涉及协作与冲突的专业场景对话的基准测试。利用CoCoEval，我们比较了人类（与LLM模拟的对话）……

    arXiv:2603.17094v2 Announce Type: replace  Abstract: Simulating human conversations using large language models (LLMs) has emerged as a scalable methodology for modeling human social interaction. This paper reconsiders the evaluation of simulated conversations by explicitly recognizing that human conversations inherently involve inconsistent and uncollaborative behaviors, such as misunderstandings and interruptions. Since these behaviors contribute to the complexity of human social interaction, we argue that LLM-simulated conversations should reproduce them at frequencies comparable to those observed in human conversations. To support a detailed and interpretable evaluation of these behaviors, we introduce CoCoEval, a framework consisting of an evaluation scheme based on turn-level detection of 10 types of inconsistent and uncollaborative behaviors and a benchmark for simulating conversations in professional scenarios involving collaboration and conflict. Using CoCoEval, we compare hum
    
[^140]: 基于语音到文本因果对齐的流式翻译与转录

    Streaming Translation and Transcription Through Speech-to-Text Causal Alignment

    [https://arxiv.org/abs/2603.11578](https://arxiv.org/abs/2603.11578)

    提出了无策略端到端模型Hikari，通过解码器时间膨胀机制和监督微调策略，以较小的模型规模实现了低延迟且高质量的同声语音翻译与流式转录。

    

    同声机器翻译（SiMT）传统上依赖于离线机器翻译模型，并结合人工设计的启发式方法或学习到的策略。我们提出了Hikari，一个无策略的端到端模型，用于同声语音到文本翻译和流式转录。我们还引入了解码器时间膨胀机制，用于抵消训练中WAIT标记过度表示的问题。我们提出了一种监督微调策略，训练模型从延迟中恢复，显著改善了质量与延迟之间的权衡。尽管模型规模不大，Hikari在持续低延迟的情况下提供了具有竞争力的翻译质量，与已发表的IWSLT 2026提交系统（规模最大达其38倍）以及专有API系统相比，在英日、英德和英俄语言对上表现出色。我们发布了模型权重和代码以促进进一步研究。

    arXiv:2603.11578v2 Announce Type: replace  Abstract: Simultaneous machine translation (SiMT) has traditionally relied on offline machine translation models coupled with human-engineered heuristics or learned policies. We propose Hikari, a policy-free, end-to-end model for simultaneous speech-to-text translation and streaming transcription. We also introduce Decoder Time Dilation, a mechanism that counteracts the overrepresentation of WAIT tokens in training. We present a supervised fine-tuning strategy that trains the model to recover from delays, significantly improving the quality-latency trade-off. Despite its modest size, Hikari delivers competitive translation quality at consistently low latency, comparing favorably with published IWSLT 2026 submissions up to 38x larger and with proprietary API systems across en-ja, en-de, and en-ru. We release our model weights and code to facilitate further research.
    
[^141]: 探测大语言模型中的知识归因

    Probing for Knowledge Attribution in Large Language Models

    [https://arxiv.org/abs/2602.22787](https://arxiv.org/abs/2602.22787)

    本文提出自监督数据生成流水线AttriWiki，证明仅用简单线性探针即可从大语言模型的隐藏表示中可靠地判断答案的知识来源是内部记忆还是外部上下文，从而区分忠实性幻觉与事实性幻觉以助力针对性缓解。

    

    大语言模型（LLM）的幻觉，即流畅但事实错误的生成内容，可分为两类：忠实性违规（模型误用了所提供的上下文）和事实性违规（答案反映了内部知识的错误）。要采取恰当的缓解措施，需要知道每个答案由哪种来源驱动。我们研究了贡献性归因，即对每个输出背后的主要知识来源进行分类，并证明在隐藏表示上训练的简单线性探针能够可靠地识别它。我们提出了AttriWiki，这是一个自监督流水线，通过提示模型从记忆中回忆被隐去的实体或从上下文中读取实体来自动生成带标签的训练数据，而无需依赖知识冲突。在AttriWiki上训练的探针在Llama-3.1-8B、Mistral-7B和Qwen-7B上取得了高达0.96的Macro-F1分数，迁移到SQuAD和WebQuestions数据集时达到0.94-0.99的Macro-F1，并展现出良好的泛化能力。

    arXiv:2602.22787v3 Announce Type: replace  Abstract: Large language model (LLM) hallucinations, meaning fluent but factually incorrect generations, fall into two types: faithfulness violations, where the model misuses provided context, and factuality violations, where answers reflect errors in internal knowledge. Proper mitigation depends on knowing which source drives each answer. We study contributive attribution, i.e. the classification of the dominant knowledge source behind each output, and show that a simple linear probe trained on hidden representations can reliably identify it. We introduce AttriWiki, a self-supervised pipeline that automatically generates labelled training data by prompting models to recall withheld entities from memory or read them from context without relying on knowledge conflicts. Probes trained on AttriWiki achieve up to 0.96 Macro-$F_1$ on Llama-3.1-8B, Mistral-7B, and Qwen-7B, transfer to SQuAD and WebQuestions with 0.94-0.99 Macro-$F_1$, and generalise
    
[^142]: 这是什么语言？问问你的分词器

    What Language is This? Ask Your Tokenizer

    [https://arxiv.org/abs/2602.17655](https://arxiv.org/abs/2602.17655)

    提出了基于UnigramLM分词算法的语言识别方法UniLID，通过判断字符串在哪种语言的单字分布下最可能出现来识别语言，无需重新训练即可增量添加新语言，并能自然融入现有语言模型分词流程，在低资源语言和密切相关语言场景下显著优于现有方法。

    

    语言识别（LID）是许多多语言自然语言处理流程中的重要组成部分，它有助于语料库整理、训练数据分析以及大型语言模型的跨语言评估。尽管现有系统在资源丰富的语言上表现出接近完美的性能，但在低资源语言和密切相关语言的场景下仍然脆弱。我们提出了UniLID，一种基于UnigramLM分词算法的简单高效的LID方法。简而言之，为了预测一个字符串的语言标签，我们只需问：在哪种语言的单字分布下，这个字符串最有可能出现？我们的方法在数据和计算上都很高效，支持在不重新训练现有模型的情况下增量添加新语言，并且可以自然地集成到现有的语言模型分词流程中。与广泛使用的基线方法（包括fasttext、GlotLID-M和CLD3）的实证评估表明，UniLID取……

    arXiv:2602.17655v3 Announce Type: replace  Abstract: Language Identification (LID) is an important component of many multilingual natural language processing pipelines, where it facilitates corpus curation, training data analysis, and cross-lingual evaluation of large language models. Despite near-perfect performance on high-resource languages, existing systems remain brittle in low-resource and closely related language settings. We introduce UniLID, a simple and efficient LID method based on the UnigramLM tokenization algorithm. In short, to predict a string's language label, we simply ask: under which language's unigram distribution is this string most likely? Our formulation is data- and compute-efficient, supports incremental addition of new languages without retraining existing models, and can naturally be integrated into existing language model tokenization pipelines. Empirical evaluations against widely used baselines, including fasttext, GlotLID-M, and CLD3, show that UniLID ac
    
[^143]: 评估大语言模型智能体中的记忆结构

    Evaluating Memory Structure in LLM Agents

    [https://arxiv.org/abs/2602.11243](https://arxiv.org/abs/2602.11243)

    本文提出StructMemEval基准测试，用于评估LLM智能体组织长期记忆结构（如交易账本、待办列表、树结构）的能力，弥补了现有基准仅测试简单事实保留和多跳召回而无法评估复杂记忆层次结构的不足。

    

    现代基于大语言模型（LLM）的智能体和聊天助手依赖长期记忆框架来存储可复用的知识、回忆用户偏好并增强推理能力。随着研究人员构建越来越复杂的记忆架构，分析其能力并指导未来的记忆设计变得愈发困难。现有的长期记忆基准测试大多聚焦于简单的事实保留、多跳召回和基于时间的变化。尽管这些能力无疑很重要，但它们通常可以通过简单的检索增强型LLM来实现，无法测试复杂的记忆层次结构。为了弥合这一差距，我们提出了StructMemEval——一个测试智能体组织长期记忆能力（而非仅仅是事实回忆）的基准测试。我们收集了一系列人类通过以特定结构组织知识来完成的任务：交易账本、待办事项列表、树结构等。我们的初步实验表明，简单的检索增强型LLM在此类任务上表现不佳。

    arXiv:2602.11243v3 Announce Type: replace-cross  Abstract: Modern LLM-based agents and chat assistants rely on long-term memory frameworks to store reusable knowledge, recall user preferences, and augment reasoning. As researchers create more complex memory architectures, it becomes increasingly difficult to analyze their capabilities and guide future memory designs. Most long-term memory benchmarks focus on simple fact retention, multi-hop recall, and time-based changes. While undoubtedly important, these capabilities can often be achieved with simple retrieval-augmented LLMs and do not test complex memory hierarchies. To bridge this gap, we propose StructMemEval - a benchmark that tests the agent's ability to organize its long-term memory, not just factual recall. We gather a suite of tasks that humans solve by organizing their knowledge in a specific structure: transaction ledgers, to-do lists, trees and others. Our initial experiments show that simple retrieval-augmented LLMs strug
    
[^144]: 迈向可靠的医疗大语言模型：医疗咨询中大语言模型置信度估计的基准测试与增强

    Towards Reliable Medical LLMs: Benchmarking and Enhancing Confidence Estimation of Large Language Models in Medical Consultation

    [https://arxiv.org/abs/2601.15645](https://arxiv.org/abs/2601.15645)

    该论文提出了首个评估真实医疗咨询中多轮交互下大语言模型置信度的基准，通过引入信息充分性梯度刻画置信度与正确性随证据积累的动态关系，并对27种代表性置信度估计方法进行了系统比较与增强。

    

    大型语言模型（LLMs）常常基于不完整的信息提供临床判断，增加了误诊的风险。现有研究主要在单轮、静态的设置下评估置信度，忽视了在真实咨询过程中随着临床证据积累，置信度与正确性之间的耦合关系，这限制了对可靠决策的支持。我们提出了首个用于评估真实医疗咨询中多轮交互置信度的基准。我们的基准统一了三种类型的医疗数据用于开放式诊断生成，并引入了信息充分性梯度来刻画证据增加时置信度与正确性的动态变化关系。我们在该基准上实现并比较了27种代表性方法，得到两个关键发现：（1）医疗数据放大了token级和一致性级置信度方法固有的局限性，（2）……

    arXiv:2601.15645v2 Announce Type: replace  Abstract: Large-scale language models (LLMs) often offer clinical judgments based on incomplete information, increasing the risk of misdiagnosis. Existing studies have primarily evaluated confidence in single-turn, static settings, overlooking the coupling between confidence and correctness as clinical evidence accumulates during real consultations, which limits their support for reliable decision-making. We propose the first benchmark for assessing confidence in multi-turn interaction during realistic medical consultations. Our benchmark unifies three types of medical data for open-ended diagnostic generation and introduces an information sufficiency gradient to characterize the confidence-correctness dynamics as evidence increases. We implement and compare 27 representative methods on this benchmark; two key insights emerge: (1) medical data amplifies the inherent limitations of token-level and consistency-level confidence methods, and (2) m
    
[^145]: 超越提示方法：通过Logit空间集成实现语音大语言模型的高效鲁棒上下文偏置（LOGIC）

    Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)

    [https://arxiv.org/abs/2601.15397](https://arxiv.org/abs/2601.15397)

    本文提出LOGIC方法，通过在Logit空间层面直接集成上下文偏置，为语音大语言模型提供了一种高效且鲁棒的解决方案，克服了传统提示方法的可扩展性瓶颈和生成式错误纠正的幻觉问题。

    

    新实体的快速涌现——受文化变迁、流行趋势演变和个性化用户数据的驱动——对现有的语音大语言模型（Speech LLMs）构成了重大挑战。虽然这些模型在通用对话任务中表现出色，但其静态训练知识限制了它们识别特定领域术语（如联系人姓名、播放列表或技术行话）的能力。现有解决方案主要依赖提示方法，但其可扩展性较差：随着实体列表的增长，提示方法会遇到上下文窗口限制、推理延迟增加以及“迷失在中间”现象。另一种替代方法——生成式错误纠正（GEC）——试图通过后处理重写转录文本，但经常出现“过度纠正”问题，引入从未被说出的实体的幻觉。在这项工作中，我们介绍了LOGIC（用于上下文偏置的Logit空间集成），一种……

    arXiv:2601.15397v3 Announce Type: replace-cross  Abstract: The rapid emergence of new entities -- driven by cultural shifts, evolving trends, and personalized user data -- poses a significant challenge for existing Speech Large Language Models (Speech LLMs). While these models excel at general conversational tasks, their static training knowledge limits their ability to recognize domain-specific terms such as contact names, playlists, or technical jargon. Existing solutions primarily rely on prompting, which suffers from poor scalability: as the entity list grows, prompting encounters context window limitations, increased inference latency, and the "lost-in-the-middle" phenomenon. An alternative approach, Generative Error Correction (GEC), attempts to rewrite transcripts via post-processing but frequently suffers from "over-correction", introducing hallucinations of entities that were never spoken.   In this work, we introduce LOGIC (Logit-Space Integration for Contextual Biasing), an 
    
[^146]: DeepResearch Bench II：通过专家报告中的评估标准诊断深度研究智能体

    DeepResearch Bench II: Diagnosing Deep Research Agents via Rubrics from Expert Reports

    [https://arxiv.org/abs/2601.08536](https://arxiv.org/abs/2601.08536)

    提出了Deep Research Bench II基准，通过从专家报告中提取的9,430个细粒度二元评估标准，从信息召回、分析和呈现三个维度对深度研究智能体的报告生成能力进行严格评估。

    

    深度研究智能体（Deep Research Agents, DRA）旨在帮助用户搜索网络、综合信息并生成全面的调查报告。先前的基准测试往往要么低估系统产生有意义洞察和高质量写作的能力，要么采用粗糙的或由大语言模型定义的标准，这些标准难以验证，且可能与人类专家的判断产生偏差。为解决这些问题，我们提出了Deep Research Bench II，一个用于评估深度研究智能体的新基准。它包含横跨22个领域的132个基于事实的研究任务；对于每个任务，智能体必须生成一份研究报告，该报告由总计9,430个细粒度的二元评估标准进行评估，涵盖三个维度：信息召回、分析和呈现。所有评估标准均源自精心挑选的专家撰写的调查文章，并通过一个四阶段的“大语言模型+人类”流水线构建而成，该流水线将自动提取与超过400小时的人工专家审查相结合。

    arXiv:2601.08536v3 Announce Type: replace  Abstract: Deep Research Agents (DRA) aim to help users search the web, synthesize information, and deliver comprehensive investigative reports. Prior benchmarks often either under-evaluate a system's ability to produce meaningful insights and high-quality writing, or adopt coarse or LLM-defined criteria that are hard to verify and can diverge from human expert judgment. To address these issues, we introduce Deep Research Bench II, a new benchmark for evaluating DRAs. It contains 132 grounded research tasks across 22 domains; for each task, an agent must produce a research report that is evaluated by a set of 9,430 fine-grained binary rubrics in total, covering three dimensions: information recall, analysis, and presentation. All rubrics are derived from carefully selected expert-written investigative articles and are constructed through a four-stage LLM+human pipeline that combines automatic extraction with over 400 human-hours of expert revie
    
[^147]: 面向稳定大语言模型预训练的输出嵌入中心化方法

    Output Embedding Centering for Stable LLM Pretraining

    [https://arxiv.org/abs/2601.02031](https://arxiv.org/abs/2601.02031)

    该论文揭示了大语言模型预训练末期输出logit发散的根源在于输出嵌入的各向异性，并提出输出嵌入中心化（OEC）新方法，通过μ-centering或μ-loss两种实现方式有效抑制训练不稳定，其稳定性优于z-loss且与logit软上限方法相当。

    

    大语言模型的预训练不仅成本高昂，而且容易出现某些训练不稳定性问题。一种通常在训练结束时发生的特定不稳定性是输出logit发散。目前最广泛使用的缓解策略——z-loss和logit软上限——仅仅处理症状而非问题的根本原因。在本文中，我们从输出嵌入几何特性的角度分析了这种不稳定性，并确定各向异性的嵌入是其根源。基于此，我们提出输出嵌入中心化（OEC）作为一种新的缓解策略，并证明它能够抑制输出logit发散。OEC可以通过两种不同的方式实现：一种称为μ-centering的确定性操作，或一种称为μ-loss的正则化方法。我们的实验表明，这两种变体在训练稳定性方面都优于z-loss，同时与logit软上限方法相当。

    arXiv:2601.02031v3 Announce Type: replace-cross  Abstract: Pretraining of large language models is not only expensive but also prone to certain training instabilities. A specific instability that often occurs at the end of training is output logit divergence. The most widely used mitigation strategies, z-loss and logit soft-capping, merely address the symptoms rather than the underlying cause of the problem. In this paper, we analyze the instability from the perspective of the output embeddings' geometry and identify anisotropic embeddings as its source. Based on this, we propose output embedding centering (OEC) as a new mitigation strategy, and demonstrate that it suppresses output logit divergence. OEC can be implemented in two different ways: as a deterministic operation called $\mu$-centering, or a regularization method called $\mu$-loss. Our experiments show that both variants outperform z-loss in terms of training stability, while being on par with logit soft-capping. This holds 
    
[^148]: 叙事整合：提出统一多视角叙述的新任务

    Narrative Consolidation: Formulating a New Task for Unifying Multi-Perspective Accounts

    [https://arxiv.org/abs/2512.18041](https://arxiv.org/abs/2512.18041)

    本文提出并形式化定义了一个新的自然语言处理任务“叙事整合”，旨在将多视角叙述性文档统一为时间连贯、内容完整且融合互补细节的文本，并构建了基于四福音书的基准数据集、评估范式及一系列参考系统。

    

    处理重叠的叙述性文档（如法律证词或历史记述）的目的往往不是为了压缩，而是为了获得统一、连贯且时间顺序合理的文本。标准的多文档摘要（MDS）侧重于简洁性，无法保留叙事的流畅性。本文正式将这一挑战定义为一个新的自然语言处理任务——叙事整合，重点关注时间完整性、内容完备性以及互补细节的融合。我们建立了研究该任务所需的资源：正式的任务定义、评估范式、福音书整合语言资源——一个基于四部圣经福音书构建的基准数据集，包含169个正典事件、跨文档对齐和参考整合文本——以及一系列参考系统，涵盖从时间线无关的启发式方法到时间对齐事件图（TAEG）。基准测试得出了三项发现。首先，显式的时间骨干……（原文摘要在此处截断）

    arXiv:2512.18041v2 Announce Type: replace  Abstract: Processing overlapping narrative documents, such as legal testimonies or historical accounts, often aims not for compression but for a unified, coherent, and chronologically sound text. Standard Multi-Document Summarization (MDS), with its focus on conciseness, fails to preserve narrative flow. This paper formally defines this challenge as a new NLP task, Narrative Consolidation, focusing on chronological integrity, completeness, and the fusion of complementary details. We establish the resources needed to study it: a formal task definition, an evaluation paradigm, the Gospel Consolidation Language Resource -- a benchmark built from the four Biblical Gospels with 169 canonical events, cross-document alignments, and a reference consolidation -- and a suite of reference systems, ranging from timeline-agnostic heuristics to the Temporal Alignment Event Graph (TAEG). Benchmarking yields three findings. First, the explicit temporal backbo
    
[^149]: 视觉语言模型理解视觉说服力吗？基于视觉说服因素的诊断

    Do Vision-Language Models Understand Visual Persuasiveness? A Diagnosis via Visual Persuasive Factors

    [https://arxiv.org/abs/2511.17036](https://arxiv.org/abs/2511.17036)

    该论文通过实证分析发现视觉语言模型在视觉说服力判断上存在过度预测为“有说服力”的召回偏向，并提出受认知心理学启发的视觉说服因素（VPFs）分类体系来量化视觉线索，揭示了模型判断与人类判断之间的差距。

    

    视觉说服利用图像来塑造认知、情感和行为，其效果同时取决于视觉属性和语义语境。尽管近期取得了进展，视觉语言模型是否理解视觉说服力仍不清楚。这促使我们提出以下问题：视觉语言模型能否评估一张图像是否在说服力上支持某个预期信息？哪些视觉因素塑造了这种判断？模型的判断与人类判断是否一致？通过对人类评估者在说服力判断上高度一致的图像-信息对进行实证分析，我们发现视觉语言模型表现出一种偏向召回的偏差：它们过度预测图像具有说服力，同时保持较高的召回率。我们提出了视觉说服因素（VPFs），这是一个受认知心理学启发的分类体系，用于量化塑造说服判断的视觉线索。我们的因素级分析表明，视觉说服因素能够有效区分人类的说服力判断，而视觉语言模型只能部……（原文摘要在此处截断）

    arXiv:2511.17036v2 Announce Type: replace  Abstract: Visual persuasion uses images to shape cognition, emotion, and behavior, with its effects depending on both visual attributes and semantic context. Despite recent progress, it remains unclear whether Vision-Language Models (VLMs) understand visual persuasiveness. This motivates us to ask: can VLMs assess whether an image persuasively supports an intended message, which visual factors shape this judgment, and do they align with human judgments? Through empirical analyses on image-message pairs where human raters consistently agree on the persuasiveness judgment, we show that VLMs exhibit a recall-oriented bias: they over-predict images as persuasive while achieving high recall. We introduce Visual Persuasive Factors (VPFs), a taxonomy informed by cognitive psychology for quantifying visual cues that shape persuasive judgments. Our factor-level analysis reveals that VPFs distinguish human persuasiveness judgments, whereas VLMs only par
    
[^150]: 利用大语言模型进行上下文感知的隐性文本和多模态仇恨言论检测

    Leveraging LLMs for Context-Aware Implicit Textual and Multimodal Hate Speech Detection

    [https://arxiv.org/abs/2510.15685](https://arxiv.org/abs/2510.15685)

    利用大语言模型为社交媒体帖子生成背景上下文，并通过多种融合方法将其融入SBERT分类器，可将隐性和多模态仇恨言论检测的F1分数在文本和多模态场景下分别最高提升3分和6分。

    

    本文研究了使用大语言模型（LLM）为社交媒体帖子生成辅助背景上下文，并探索了将此上下文融入基于SBERT的仇恨言论检测（HSD）分类器输入的四种方法。这些方法包括：文本拼接、嵌入拼接、基于层次化Transformer的融合以及LLM驱动的文本增强。我们在文本场景下使用隐含仇恨推文的Latent Hatred数据集，在多模态场景下使用厌女网络迷因的MAMI数据集，评估了上下文生成与融入策略的效果。评估结果与零上下文基线、两种先前基于实体链接的方法以及零样本LLM分类器进行比较。研究结果表明，融入生成的上下文可将仇恨言论检测性能在文本和多模态场景下分别提升最多3个和6个F1分数点。

    arXiv:2510.15685v2 Announce Type: replace  Abstract: This paper investigates the use of an LLM to generate auxiliary background context for social media posts, and explores four methods to incorporate this context into the input of an SBERT-based Hate Speech Detection (HSD) classifier. These are: text concatenation, embedding concatenation, a hierarchical transformer-based fusion, and LLM-driven text enhancement. We evaluate the impact of our context generation and incorporation strategies in a textual setting on the Latent Hatred dataset of implicitly hateful tweets and a multimodal setting on the MAMI dataset of misogynous internet memes. Results are evaluated against a zero-context baseline, two previous approaches based on entity linking, and a zero-shot LLM classifier. Findings indicate that incorporating generated context improves HSD performance by up to 3 and 6 F1 points on textual and multimodal settings respectively, from a zero-context baseline to the highest-performing syst
    
[^151]: CHRONOBERG：捕捉基础模型中的语言演化与时间感知

    CHRONOBERG: Capturing Language Evolution and Temporal Awareness in Foundation Models

    [https://arxiv.org/abs/2509.22360](https://arxiv.org/abs/2509.22360)

    该论文提出了CHRONOBERG——一个跨越250年、带有丰富时间标注的英语书籍文本语料库，通过历史校准的情感词典量化语言演变，从而提升基础模型对语言历时变化的时间感知能力。

    

    大型语言模型（LLM）通过利用社交媒体和从网络上爬取的各类数据，实现了大规模的高效运作。然而，尽管现有语料库具有多样性，但它们普遍缺乏长期的时间结构，这可能限制了LLM对语言语义和规范演变的语境化理解能力，以及捕捉历时性变化的能力。为支持针对后者的分析与训练，我们推出了CHRONOBERG——一个跨越250年、具有时间结构化特征的英语书籍文本语料库，精选自古腾堡计划（Project Gutenberg），并辅以多种时间标注进行丰富。首先，书籍经过编辑的特性使我们能够通过时间敏感的效价-唤醒-支配度（Valence-Arousal-Dominance，VAD）分析来量化词汇语义随时间的变化，并构建经过历史校准的情感词典，以支持基于时间的解释。借助这些词典，我们论证了现代基于LLM的工具需要更好地定位其对话语的检测……

    arXiv:2509.22360v2 Announce Type: replace  Abstract: Large language models (LLMs) excel at operating at scale by leveraging social media and various data crawled from the web. Whereas existing corpora are diverse, their frequent lack of long-term temporal structure may however limit an LLM's ability to contextualize semantic and normative evolution of language and to capture diachronic variation. To support analysis and training for the latter, we introduce CHRONOBERG, a temporally structured corpus of English book texts spanning 250 years, curated from Project Gutenberg and enriched with a variety of temporal annotations. First, the edited nature of books enables us to quantify lexical semantic change through time-sensitive Valence-Arousal-Dominance (VAD) analysis and to construct historically calibrated affective lexicons to support temporally grounded interpretation. With the lexicons at hand, we demonstrate a need for modern LLM-based tools to better situate their detection of disc
    
[^152]: PIMMUR原则：确保LLM社会集体行为模拟的有效性

    The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies

    [https://arxiv.org/abs/2509.18052](https://arxiv.org/abs/2509.18052)

    该论文提出PIMMUR六项方法学原则，对350篇论文中的576项基于LLM的社会模拟研究进行系统性审计，发现前沿LLM在65.2%的案例中能识别底层社会实验、50.6%的提示词预先决定了实验结果，揭示了当前LLM社会模拟在有效性验证上的严重不足。

    

    大语言模型（LLM）正被越来越多地用于模拟人类集体行为，然而此类模拟具有类人特征的说法在很大程度上未经检验。我们对基于LLM的社会模拟进行了系统性审计（已在OSF上预注册），覆盖四个数据库（Scopus、IEEE Xplore、ACM数字图书馆和arXiv）。在350篇近期论文报告的576项研究中，我们应用了六项方法学评估：智能体画像、交互、记忆、最小控制、无意识性和真实性（PIMMUR）。通过对每项研究按照预先设定的规则进行编码，我们发现PIM（画像、交互、记忆）比MUR（最小控制、无意识性、真实性）更常被满足。前沿LLM在65.2%的案例中正确识别出了底层的社会实验，且50.6%的提示词施加了预先决定结果的约束条件。这些合规率属于上限值，因为方法学报告不完整（例如未公开提示词）限制了可用的证据。复现五项具有代表性的

    arXiv:2509.18052v4 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used to simulate human collective behavior, yet claims that such simulations are human-like remain largely untested. We conducted a systematic audit (pre-registered on OSF) of LLM-based social simulations across four databases (Scopus, IEEE Xplore, ACM Digital Library, and arXiv). Across 576 studies reported in 350 recent papers, we applied six methodological evaluations: agent Profile, Interaction, Memory, Minimal-Control, Unawareness, and Realism (PIMMUR). Coding every study against pre-specified rules, we revealed that PIM were met more often than MUR. Frontier LLMs correctly identified the underlying social experiment in 65.2% of cases, and 50.6% of prompts imposed constraints that pre-determined the outcome. These compliance rates are upper bounds, because incomplete methodological reporting (for example, unreleased prompts) limits the available evidence. Reproducing five representat
    
[^153]: “镜像”式大语言模型抑郁评估存在标准污染

    "Mirror" Large Language Model Evaluations of Depression are Criterion Contaminated

    [https://arxiv.org/abs/2508.05830](https://arxiv.org/abs/2508.05830)

    该研究揭示了“镜像”式LLM抑郁评估存在标准污染问题——用抑郁评估本身的语言来预测同一评估的分数会产生近乎完美的虚高结果，而独立评估显示这种优势并不存在。

    

    使用抑郁评估引出的语言反应来预测同一评估分数的大语言模型（LLM）研究，往往报告近乎完美的抑郁预测结果。我们将这类研究称为“镜像”评估，并展示了标准污染的一个实际案例。共有110名参与者完成了结构化诊断抑郁访谈（镜像条件）和生活史访谈（“非镜像”条件）。研究提示大语言模型分别预测两种条件下的抑郁分数。正如预期，镜像评估的结果近乎完美。然而，非镜像评估也展现出在心理学领域被视为卓越水平的预测效应量。此外，镜像和非镜像预测与患者健康问卷-9（PHQ-9）分数的相关程度相近，这表明当预测独立的抑郁测量指标时，镜像条件的优势不复存在。主题建模揭示了不同的抑郁相关……

    arXiv:2508.05830v3 Announce Type: replace  Abstract: Large Language Model (LLM) studies that use language responses elicited from depression assessments to predict scores on those same assessments often report near-perfect prediction of depression. We refer to these as "Mirror" evaluations and demonstrate an applied case of criterion contamination. N = 110 participants completed both structured diagnostic depression interviews (Mirror condition) and life history interviews ("Non-Mirror" condition). LLMs were prompted to predict depression scores in each condition. As expected, Mirror evaluations were near-perfect. However, Non-Mirror evaluations also displayed prediction sizes considered outstanding in psychology. Further, both Mirror and Non-Mirror predictions correlated with Patient Health Questionnaire-9 scores at similar sizes, suggesting the Mirror condition's advantage collapses when predicting an independent depression measurement. Topic modeling revealed differing depression-re
    
[^154]: 绘制哥伦比亚七十年哲学图景：对《Ideas y Valores》期刊的动态主题建模研究

    Mapping Seven Decades of Philosophy in Colombia: Dynamic Topic Modelling of Ideas y Valores

    [https://arxiv.org/abs/2412.04236](https://arxiv.org/abs/2412.04236)

    本研究通过动态主题建模技术分析哥伦比亚最具影响力的哲学期刊《Ideas y Valores》七十年来主题的演变，将哲学史的数据驱动研究拓展至哥伦比亚和拉丁美洲语境，发现价值理论、认识论和科学哲学是该地区最突出的哲学研究主题。

    

    以数据驱动的方法研究哲学已成为探索该学科历史的重要工具。然而，该领域的大多数研究都集中在特定地区和子领域的少数期刊上。我们通过应用动态主题建模技术来探索哥伦比亚和拉丁美洲哲学的历史，从而扩展了这一研究的范围。我们的研究考察了哥伦比亚哲学期刊《Ideas y Valores》，该期刊创刊于1951年，目前是该地区最具影响力的学术哲学期刊之一。通过分析该期刊历史中主题的演变，我们识别出哥伦比亚和拉丁美洲语境下哲学话语的多种趋势和特定动态。我们的研究发现，最突出的主题是价值理论（包括伦理学、政治哲学和美学）、认识论和科学哲学。我们还追踪了（原文摘要在此处截断）……

    arXiv:2412.04236v2 Announce Type: replace-cross  Abstract: Data-driven approaches to philosophy have emerged as a valuable tool for studying the history of the discipline. However, most studies in this area have focused on a limited number of journals from specific regions and subfields. We expand the scope of this research by applying dynamic topic modelling techniques to explore the history of philosophy in Colombia and Latin America. Our study examines the Colombian philosophy journal Ideas y Valores, founded in 1951 and currently one of the most influential academic philosophy journals in the region. By analyzing the evolution of topics across the journal's history, we identify various trends and specific dynamics in philosophical discourse within the Colombian and Latin American context. Our findings reveal that the most prominent topics are value theory (including ethics, political philosophy, and aesthetics), epistemology, and the philosophy of science. We also trace the evoluti
    
[^155]: 查看大型语言模型在法律方面的简要调查

    A Short Survey of Viewing Large Language Models in Legal Aspect. (arXiv:2303.09136v1 [cs.CL])

    [http://arxiv.org/abs/2303.09136](http://arxiv.org/abs/2303.09136)

    本文调查了大型语言模型在法律领域中的使用情况，讨论了它们应用于法律任务时所面临的法律问题，以及使用数据资源在法律领域中专门化LLMs的可能性。

    

    大型语言模型（LLMs）已经改变了许多领域，包括自然语言处理、计算机视觉和强化学习。这些模型也在法律领域产生了重要的影响，它们被越来越多地用于自动化各种法律任务，例如法律判断预测、法律文件分析和法律文件撰写。然而，将LLMs整合到法律领域中也引发了一些法律问题，包括隐私问题、偏见和可解释性。在本次调查中，我们探索了将LLMs融入法律领域的情况。我们讨论了LLMs在法律任务中的各种应用，研究了它们使用时出现的法律挑战，并探讨了可以用于在法律领域专门化LLMs的数据资源。最后，我们讨论了几个有前途的方向并总结了本文。通过这样做，我们希望提供LLMs在法律上的当前状态概述，并突出其潜在的好处和挑战。

    Large language models (LLMs) have transformed many fields, including natural language processing, computer vision, and reinforcement learning. These models have also made a significant impact in the field of law, where they are being increasingly utilized to automate various legal tasks, such as legal judgement prediction, legal document analysis, and legal document writing. However, the integration of LLMs into the legal field has also raised several legal problems, including privacy concerns, bias, and explainability. In this survey, we explore the integration of LLMs into the field of law. We discuss the various applications of LLMs in legal tasks, examine the legal challenges that arise from their use, and explore the data resources that can be used to specialize LLMs in the legal domain. Finally, we discuss several promising directions and conclude this paper. By doing so, we hope to provide an overview of the current state of LLMs in law and highlight the potential benefits and c
    

