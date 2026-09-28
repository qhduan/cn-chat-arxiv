# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Stop without Learning to Stop: Self-Supervised Confidence Training Improves Reasoning Efficiency](https://arxiv.org/abs/2609.31619) | 该论文发现，仅通过自监督方式训练模型在推理过程中间点预测自身置信度（损失函数中不含任何长度、效率或停止目标），就能让模型在推理时无需任何提前停止机制便自发提升推理效率。 |
| [^2] | [User Model Extraction via Belief Self-Distillation](https://arxiv.org/abs/2609.31603) | 提出信念自蒸馏框架，使冻结的大语言模型从自然对话中自我蒸馏出既能读出又能写回的用户信念表示，并揭示模型的拒绝行为取决于其推断出的用户意图。 |
| [^3] | [Compact Documentation for Coding Agents: A Benchmark, an Optimizer, and Why It Does Not Transfer](https://arxiv.org/abs/2609.31587) | 本文提出了一个通过“重新生成代码能否通过原始测试”来评估代码描述的往返基准和优化器，发现完整性而非长度决定文档保真度，但令人意外的负面结果表明：当源代码可用时，更好的文档并不能帮助编程智能体解决真实的仓库问题。 |
| [^4] | [Strategically Diverse Sampling for Self-Training](https://arxiv.org/abs/2609.31571) | 该论文提出以策略多样性作为自训练数据构建的新原则，通过GROOT（构建方法层次树并采样不同路径）和语言化采样两种方法生成策略多样的训练数据，使模型在困难任务上优于传统IID采样训练的模型。 |
| [^5] | [MexHat: A Dataset for Hate Speech Detection in Mexican Spanish Videos](https://arxiv.org/abs/2609.31553) | 本文提出了MexHat数据集，这是一个包含约1000个墨西哥西班牙语视频片段的仇恨言论检测数据集，通过三分类和细粒度子类别两种标注方式，捕捉了该语言文化背景下的语言和文化线索。 |
| [^6] | [Two Conformal Constructions for Adaptive Within-Document AI-Text Screening](https://arxiv.org/abs/2609.31547) | 该论文提出两种基于共形方法的有限样本构造，在文档级可交换性假设下实现对自适应文档内AI文本筛查的误报警率控制，并支持提前停止且无需对文档内词元依赖性做任何假设。 |
| [^7] | [Statistical Foundations for a Google Play User-Review Sentiment Index: Signal Fusion, Shrinkage, Distributional Validation, and Dynamic Smoothing](https://arxiv.org/abs/2609.31513) | 该论文为Google Play用户评论构建了具有严格数学证明的显式情感指数，核心创新在于以协方差感知的逆方差加权融合星级与文本情感信号、基于估计精度向总体均值收缩、针对离散评分的分布验证以及卡尔曼滤波动态平滑。 |
| [^8] | [Muslim: A Deployed Arabic Voice AI Platform for Grounded Islamic Knowledge](https://arxiv.org/abs/2609.31511) | 该论文推出了已部署的阿拉伯语语音AI生产平台Muslim，其核心贡献包括发布微调模型族（工具路由模型Muslim-6B-PRO与TTS模型Fasih-TTS-V1）、将开放演示转变为可运营防滥用产品的账户计量层，以及三层可观测性体系。 |
| [^9] | [Evaluating Cultural Awareness of LLMs for Haitian Creole](https://arxiv.org/abs/2609.31506) | 该论文首次从特异性、偏见、多样性和变异性四个维度系统评估了大语言模型对海地克里奥尔语的文化意识，发现其明显落后于法语，且海地角色常被以苦难形象呈现。 |
| [^10] | [PriceBench: A Diagnostic Benchmark for Price, Quality, and Brand Preferences in LLM Booking Agents](https://arxiv.org/abs/2609.31468) | PriceBench通过logit选择模型从LLM的酒店预订行为中诊断其价格、质量和品牌偏好，发现更强的模型偏好更一致但各不相同，而较弱的模型要么偏好僵化易被列表顺序操纵，要么近乎随机选择。 |
| [^11] | [ViSTA: A Simple Bridge Extends Visual Alignment to Clinical Time-Series Understanding in Multimodal LLMs](https://arxiv.org/abs/2609.31448) | ViSTA是一种轻量级适配器，通过将不规则的数值型临床时间序列数据融入预训练视觉-语言模型的图表表示中，在冻结全部预训练参数的前提下实现临床风险预测性能的大幅提升。 |
| [^12] | [Towards Mitigating Fabricated Consensus: The Active Provenance Gate for Multi-Agent Debate Synthesis](https://arxiv.org/abs/2609.31422) | 提出主动溯源门控（APG），通过将来源作为硬约束、审计辩论中的每一条论断并自我纠错，来缓解多智能体辩论综合阶段摘要模型伪造无事实依据“辩论共识”的安全问题。 |
| [^13] | [Sorry Robot, Happy Human: Vision-Language Models Read Only One of Two Legible Typographic Layers](https://arxiv.org/abs/2609.31403) | 本研究创建了DecoyBench数据集，发现视觉语言模型在包含两层叠加可读文字的图像中只能读取轮廓文字而几乎无法提取阴影文字，而人类可以高准确率读取两层，揭示了VLM在多文本层图像理解上的根本性脆弱。 |
| [^14] | [Intent2Tc: Automated Intent-to-Traffic Control Translation with Language Models](https://arxiv.org/abs/2609.31397) | 该论文提出Intent2Tc，一个由语言模型驱动的闭环框架，能够将业务级流量整形意图自动转换为经过验证的可执行Linux流量控制配置，并通过数字孪生语义模型、批判驱动精炼和RAG知识复用显著提升语义一致性与配置可靠性。 |
| [^15] | [Highlight-Then-Summarize: Learning to Compress Evidence for Long-Context Understanding](https://arxiv.org/abs/2609.31382) | 提出“先标注后总结”（H2S）的先压缩后推理范式，通过带过程级奖励的强化学习训练模型先识别并压缩与问题相关的证据、生成条件化摘要后再作答，显著提升大模型的长上下文理解能力。 |
| [^16] | [Stale-Document Poisoning: When Outdated Retrieval Overrides Correct Model Answers](https://arxiv.org/abs/2609.31342) | 该论文首次揭示并定义了一种新的RAG失效模式“过时文档投毒”——过时的外部检索证据会覆盖大模型本已正确的回答，导致最多75%的正确答案被翻转，并构建了涵盖317个经核实的知识反转案例的基准来系统量化这一时间对齐失效风险。 |
| [^17] | [The Right Information Extraction Pipeline Depends on the Document: Accuracy-Energy Trade-offs for Small, Local Models](https://arxiv.org/abs/2609.31341) | 该研究在隐私约束下系统评估了小型本地模型信息抽取流水线的准确率-能耗权衡，发现最优选择（图像 vs 文本）取决于文档类型，批处理可无损降低38-85%能耗，而神经OCR能耗高达传统OCR的17倍。 |
| [^18] | [Why Alzheimer's Speech Screening Fails to Generalize: Bridging the Deployment Gap via Cross-Corpus Evidence Anchoring](https://arxiv.org/abs/2609.31293) | 该论文通过跨四个语料库的留一评估揭示了阿尔茨海默病语音筛查模型泛化失败的根源——特征方向冲突与录音协议敏感性，并提出融合XLM-R文本基线分数的跨语料库证据锚定方法来弥合部署差距。 |
| [^19] | [Identifying Scientists on X](https://arxiv.org/abs/2609.31264) | 该论文提出了一种基于用户简介和推文自动识别X/Twitter上科学家与非科学家的分类方法，在集成设置中使用对比微调的DeBERTa模型达到0.96的F1分数，并发布了两个带标注的用户数据集。 |
| [^20] | [MoSAR: Mixture of Semantic Attention Regimes for Learning Adaptive and Approximable Attention Geometries](https://arxiv.org/abs/2609.31261) | MoSAR将注意力近似视为几何学习问题，通过输入条件化路由器在短程、中程和全局注意力机制间学习自适应混合，以连续的距离相关注意力场取代固定稀疏模式，突破长上下文建模中稠密自注意力的二次复杂度瓶颈。 |
| [^21] | [PIA: A Personal Intelligence Agent Turning Health Conversations into Records and Records into Understanding](https://arxiv.org/abs/2609.31255) | PIA 是一个部署在消费级健康智能体旁的个人智能代理，它通过抽取、记忆、检索、理解四个领域无关的控制机制配合可插拔的健康模块，将健康对话转化为结构化临床记录并进一步综合为对用户的理解，从而克服了通用摘要式记忆无法处理剂量、时间指代和长期健康趋势的局限。 |
| [^22] | [RupeeBias: Auditing Demographic Bias in Indian Economic Guidance from Large Language Models](https://arxiv.org/abs/2609.31245) | 提出RupeeBias基准，通过39,150个提示系统审计大语言模型在印度经济指导场景（如薪资估计）中针对种姓、城乡等印度特有人口群体类别产生的偏见，填补了现有西方中心偏见基准的空白。 |
| [^23] | [Where a Model Sends Its Own Repeated Token](https://arxiv.org/abs/2609.31181) | 该研究将模型在“自身重复词元”输入下的输出视为覆盖整个词表的映射，首次记录了非不动点词元的去向，并通过按源词元配对消除基数混淆，为黑盒模型识别提供了新的判别信号。 |
| [^24] | [Improving Visual Sensitivity of LLMs on Multimodal Machine Translation with Metric-based Loss Weighting](https://arxiv.org/abs/2609.31169) | 提出基于度量的损失加权训练方法，利用PCXMI度量识别受益于图像的词元并提高其损失权重，从而增强多模态大语言模型在多模态机器翻译任务中对视觉信息的敏感性和利用能力。 |
| [^25] | [JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models](https://arxiv.org/abs/2609.31142) | 该论文提出了首个针对校准决策强化学习（RLCD）模型的对抗攻击基准JevAdvBench，创新之处在于以模型自身的干净决策（而非外部标签）作为评分参照来度量攻击效果，并配套提出了黑盒攻击方法。 |
| [^26] | [Do we need to answer that question? Salience and Answerability of Potential Questions in Naturalistic Dialogue](https://arxiv.org/abs/2609.31130) | 通过对英国国家语料库中自动生成的7,124个潜在问题进行显著性与可回答性标注分析，研究发现自然对话中问题的显著性虽与被回应的可能性呈正相关，但该关联明显弱于独白文本，表明对话结构的可预测性较低。 |
| [^27] | [LocUS: Head Selection and Subspace Projection for Targeted Activation Steering](https://arxiv.org/abs/2609.31122) | 提出 LocUS 方法，通过在解嵌矩阵中识别属性特定的线性子空间、将引导变换限制在该子空间并局部化到稀疏的注意力头子集，实现了更精准、对无关能力副作用更小的定向激活引导。 |
| [^28] | [CG-Probes: Recovering Guardrail Directions from Patient Query Embeddings](https://arxiv.org/abs/2609.31062) | 该论文提出CG-Probes方法，通过与肿瘤科医生合作定义医疗紧急度、心理紧急度和话题敏感性三个风险轴，并利用均值差方法从患者查询嵌入中恢复线性风险方向，从而为医疗AI助手构建有效的安全护栏。 |
| [^29] | [Modeling Student Sensemaking with LLMs and Knowledge-Graph-Guided Inference](https://arxiv.org/abs/2609.31046) | 本研究探索利用指令微调的大语言模型结合知识图谱引导的知识状态诊断来分析协作式科学学习中的学生意义建构，发现启用推理的提示和知识状态信息能显著提升模型识别不成功意义建构的能力。 |
| [^30] | [KuaFu: Compressing Long User Behavior into Understanding at Billion Scale](https://arxiv.org/abs/2609.31045) | 该论文提出KuaFu系统，在十亿用户规模上解决将长用户行为压缩为用户理解时序列过长与刷新吞吐量的双重工业瓶颈，并应对压缩过程可能引入的四类幻觉问题。 |
| [^31] | [Same Text, Different Numbers: The Divergence of LLM-Based Measures](https://arxiv.org/abs/2609.31013) | 该论文让七个不同的大语言模型对相同的标普500公司财报电话会议文本进行十三种构念评分，发现基于LLM的文本度量存在显著的模型间分歧（平均秩相关仅0.52，共同差异仅占34%），且模型选择会实质性改变下游实证研究中的系数大小、符号和显著性。 |
| [^32] | [G$^2$PTQ: Improving LLM Post-Training Quantization with Generalized Gradient Compensation](https://arxiv.org/abs/2609.31009) | G²PTQ提出了一种统一的大语言模型训练后量化框架，通过在每个Transformer块量化前动态刷新并融合一阶梯度与二阶Hessian信息的广义梯度补偿机制，克服了现有GPTQ类方法缺乏全局监督或指导信息随量化进程失效的缺陷。 |
| [^33] | [ZooWork-ShopRanker: An Open, Preference-Aligned E-Commerce Reranker](https://arxiv.org/abs/2609.31002) | 提出了一系列（0.6B/4B/8B）与多家族推理大语言模型裁判标注的购物偏好对齐的开放电商重排序器，并以8B旗舰模型为教师蒸馏出更高效的4B和0.6B模型。 |
| [^34] | [Evaluating Sycophancy in Chinese Large Language Models on Factual Questions Derived from Online Search Queries](https://arxiv.org/abs/2609.30986) | 该研究通过12165个是/否事实核查问题和超过36万个回答，系统评估了DeepSeek、Qwen和豆包三个中文大语言模型在用户表达错误信念时的事实性谄媚现象，并探究其回答转变的具体模式。 |
| [^35] | [THA: Weighted Finite-State Text Normalization and Inverse Text Normalization for Khmer](https://arxiv.org/abs/2609.30984) | Tha是首个针对高棉语的开源文本规范化与逆文本规范化工具包，基于加权有限状态转换器实现单次最短路径搜索完成整行分词与分类，在高棉语基准测试上取得了近乎完美的准确率。 |
| [^36] | [Does Uniform Discrete Diffusion Need Time?](https://arxiv.org/abs/2609.30977) | 本研究从理论和实证两方面证明，在语言建模等有限数据场景下，均匀离散扩散模型的时间条件化在很大程度上是不必要的，与时间无关的预测器能达到甚至超越时间条件化模型的性能。 |
| [^37] | [Coupled Usage-Sense Processes: Temporal and Attributable Lexical Semantic Change](https://arxiv.org/abs/2609.30974) | 本文提出耦合用法-语义过程（CUSP）框架，通过单一的边缘保持时序过程，不仅能量化词汇语义变化的幅度，还能精确确定变化发生的时间、将变化分解为成分移动与重组的机制，并归因到具体的词语用法。 |
| [^38] | [FAVoR: Measuring and Mitigating Author-Style Homogenization in Federated Personalized Generation](https://arxiv.org/abs/2609.30968) | 该论文揭示了联邦个性化生成中标准聚合方法会导致不同作者写作风格同质化的问题，并提出 FAVoR 评估协议来测量和缓解这一现象。 |
| [^39] | [Estimating and Orthogonalizing Unknown Pre-training Gradients for Continual Fine-tuning of Large Language Models](https://arxiv.org/abs/2609.30935) | 提出EoupCT框架，通过动态生成最易受遗忘影响的伪数据来估计未知的预训练梯度，并将其与新任务的梯度进行正交化投影，从而在持续微调大语言模型时有效保护其固有的通用知识，避免灾难性遗忘。 |
| [^40] | [Training-Free Pronunciation Transcription via Text-Constrained Acoustic Rescoring](https://arxiv.org/abs/2609.30924) | 提出一种免训练的语音与文本到发音（ST2P）流水线，通过文本约束候选生成与冻结预训练S2P模型的声学重打分，在日语语料库上将发音转写错误率从0.60%–1.40%大幅降低至0.04%–0.17%，无需昂贵的发音标注数据。 |
| [^41] | [Cross-Backend QIEO: Universal Runtime Portability across OpenMP5, CUDA, HIP, and Multi-Language Interfaces](https://arxiv.org/abs/2609.30914) | 提出跨后端量子启发进化优化器QIEO，通过单一C++代码库的“单一事实来源”架构，实现了在OpenMP5、CUDA、HIP等硬件后端及多语言接口上的通用运行时可移植性，无需物理量子比特即可在NP难问题上取得10–80倍的加速。 |
| [^42] | [ToolSearcher: Optimizing Tool Selection at Scale via Reinforcement Learning](https://arxiv.org/abs/2609.30906) | 本文提出ToolSearcher，一个面向大规模工具选择的新型强化学习框架，通过多轮搜索与细粒度优化，解决了LLM在海量且多样的真实工具库中难以有效搜索、区分和组合工具的挑战。 |
| [^43] | [From annotation to reasoning: Culture in language models](https://arxiv.org/abs/2609.30897) | 该论文提出以文学阐释为研究场景，主张文化能力评估应从事实知识测试转向考察“解释深度”，即模型用证据支撑文化解读并在批评中修正解读的能力，同时评估框架应保留学术分歧的合理性。 |
| [^44] | [Effects of Transcript Compression on LLM-based Medical Misinformation Detection in Japanese YouTube Videos](https://arxiv.org/abs/2609.30882) | 研究发现，对日语医疗YouTube视频的字幕进行压缩（摘要、检索或筛选）会降低大语言模型检测医疗错误信息的准确性，完整字幕输入效果最佳。 |
| [^45] | [Evidence-Grounded Auditing of Identification Assumptions in Climate-Policy Causal Evaluations](https://arxiv.org/abs/2609.30867) | 提出了ARGUS——一个结构化语言模型流水线，用于审计气候政策双重差分研究中支持识别假设的证据，可检测出73%的植入缺陷（远超关键词方法的18%），并在证据无法检索时主动弃权。 |
| [^46] | [Persistent Negatives for Adversarial Black-Box On-Policy Distillation](https://arxiv.org/abs/2609.30864) | 提出持久负样本对抗蒸馏方法，通过活跃池机制用历史教师-学生对比作为稳定负样本，解决了对抗蒸馏中负样本分布随策略更新而漂移的移动目标问题，从而提升黑盒在线策略蒸馏的稳定性。 |
| [^47] | [Enhancing Assessment of Self-Consistency in LLM Explanations using Perturbation Strength](https://arxiv.org/abs/2609.30849) | 本文提出用大语言模型作为评判者来统一度量输入和思维链扰动的强度，从而在受控强度条件下公平评估各类大语言模型解释的自洽性，发现输入扰动的影响通常强于思维链扰动，且自洽性判断只有在同一扰动类型内才公平。 |
| [^48] | [I-Parakeet: Integer-Only Conformer ASR on Mobile NPU](https://arxiv.org/abs/2609.30846) | 提出I-Parakeet，一种可在智能手机NPU上无需浮点运算运行的纯整数Conformer语音识别模型，通过整数化相对位置自注意力、优化Swish近似和逐层激活范围分析实现高效边缘部署。 |
| [^49] | [Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness](https://arxiv.org/abs/2609.30820) | 该论文揭示了循环Transformer低比特训练后量化的两种失效模式——“反馈暴露”（无恒等路径的量化层误差在递归中被反复放大回馈，且该现象同样存在于Mamba等非Transformer架构）和“校准盲区”（单步GPTQ仅依赖第0步激活校准，忽视后续递归步的输入方向）。 |
| [^50] | [Understanding the Role of Prompt Template in Knowledge Distillation for Safety Alignment](https://arxiv.org/abs/2609.30802) | 研究发现知识蒸馏中的提示模板选择会显著影响学生模型已有的安全对齐，使用对话模板会降低模型安全性，而非对话模板能更好地保留安全对齐与模型的内部表示。 |
| [^51] | [Symbiotic Architecture for Post-Hoc Audio Extension of Frozen Language Models](https://arxiv.org/abs/2609.30784) | 该论文提出一种共生架构，通过注入器模块将音频条件化向量直接写入冻结LLM的KV缓存，使其无需微调权重即可具备音频理解能力，同时完整保留原有语言能力并获得更优的可扩展性。 |
| [^52] | [Learning Natural Conversational Behavior in Tandem Speech-to-Speech Models with Randomized Guidance](https://arxiv.org/abs/2609.30773) | 该论文提出“随机化中间引导”方法，直接从对话语料库获取训练引导而无需用模拟器LLM生成后端引导，大幅降低数据准备开销，同时教会语音前端有选择性地利用后端LLM信息，从而学习自然的对话行为。 |
| [^53] | [SEA-CLIP-Tiny: Efficient Multilingual Text-Vision Embedding for Southeast Asian Languages](https://arxiv.org/abs/2609.30739) | 本文提出SEA-CLIP-Tiny，一个参数量不足5000万的紧凑型多语言文本-视觉嵌入模型，通过区域数据筛选和多语言教师引导，在七种东南亚语言的跨语言图像-文本检索上取得最优平均性能，且比MobileCLIP2参数更少、CPU延迟更低。 |
| [^54] | [Beyond Mean Attention: Diversity-Aware, Layer-Wise Scoring for KV Cache Eviction](https://arxiv.org/abs/2609.30738) | 该论文提出在注意力均值之外加入注意力离散度与冗余惩罚（类MMR多样性）的统一KV缓存淘汰评分，并发现仅用一个全局多样化常数即可在多数LongBench数据集上带来提升，而无需按层或按数据集精细搜索。 |
| [^55] | [Words Speak Louder Than Order: A Behavioral Evaluation of Gemma 4](https://arxiv.org/abs/2609.30716) | 该研究通过完全平衡的实验设计评估 Gemma 4 模型在接收冲突文档时的行为，发现信息源的语义表述（如“官方指南”或“最新更新”）对模型最终答案的影响显著强于文档的呈现顺序。 |
| [^56] | [LAVOIR: Teaching a Single-Pass Decision Encoder When and What to Ask with Amortized Value of Information](https://arxiv.org/abs/2609.30706) | LAVOIR将候选缺失信息槽位与答案选项一同置于输入中，使单次前向传播同时输出决策分布和各槽位的信息价值预期增益，从而以无需人工标注的方式教会决策模型何时提问、问什么。 |
| [^57] | [TRACE: Temporal Audit and Condition-aware Evaluation of Streaming Video Understanding](https://arxiv.org/abs/2609.30670) | TRACE是一个面向流媒体视频理解的条件感知评测框架，通过时间审计任务、证据时序与触发标注以及因果Core-Adapter协议，显式控制并记录信息可用性与响应行为，从而对答案质量、及时性、响应选择、工作负载、完成度和可靠性进行多维度评估，揭示了相似分数背后不同的工作负载与失败模式。 |
| [^58] | [Prompt Injection Detection for Email Agents Through Attack Chain Modeling](https://arxiv.org/abs/2609.30657) | 该论文提出了一种针对邮件智能体的提示注入检测框架，将检测问题从传统的二元恶意文本分类转变为对多阶段攻击链的建模，结合文本检测器、分阶段验证器、规则风险信号、意图-行动一致性分析与逻辑斯蒂决策策略，并在多种迁移设置下验证了其有效性。 |
| [^59] | [Recursive Self-Improvement via On-Policy Distillation for Reasoning](https://arxiv.org/abs/2609.30652) | 该论文提出一个递归自我提升框架，让特权教师模型在训练过程中不断吸收学生模型学到的改进而非保持冻结，从而克服了传统同策略自蒸馏中教师固定所带来的局限性。 |
| [^60] | [Epstein Files Engine: Agentic Search for Investigative Journalism](https://arxiv.org/abs/2609.30611) | 《纽约时报》开发的“爱泼斯坦文件引擎”通过大语言模型将记者问题转化为SQL查询，在三百万页司法部文件中检索带引用的可验证答案，助力100余名记者完成至少20篇报道，其核心创新Diff重复匹配方法放大了新颖性信号，并证明了新闻编辑室AI智能体应作为源材料接口而非自主写作者来发挥最大价值。 |
| [^61] | [The Hard Part Comes After Search: Benchmarking Web Agents on Synthesizing, Organizing, and Displaying Knowledge](https://arxiv.org/abs/2609.30604) | 该论文提出了KNOWS基准，通过开放式、复杂的浏览器任务联合评估网络智能体在信息检索、知识综合、任务分解以及产出文档等最终制品方面的综合能力。 |
| [^62] | [Thinking Less to Simulate Better: Intuitive Prompting Improves LLM Agents Simulating Individual Social Media Reactions, Including Unfamiliar Content](https://arxiv.org/abs/2609.30563) | 本研究通过八名真实用户画像与六十八条帖子反应的对照实验发现，采用直觉式（少推理）提示并以态度性内容而非人口统计背景构建用户画像，可显著提升大语言模型智能体模拟个体社交媒体真实反应的准确性（包括对陌生内容的反应），且智能体与画像的一致性并不等同于行为保真度。 |
| [^63] | [Probing Stability-Plasticity Tradeoffs in Agent Memory through Cognitive Experimental Paradigms](https://arxiv.org/abs/2609.30558) | 本文提出受认知科学启发的MemProbe框架，通过干扰、错误信息、巩固强度和再巩固窗口四种可复用的实验范式，超越传统最终答案准确率评估，系统诊断智能体记忆中稳定性与可塑性之间的权衡。 |
| [^64] | [REALMS: An AI-Assistant Conversational System for Real-Time Exact Audience Sizing over High-Dimensional Nested Profiles](https://arxiv.org/abs/2609.30547) | REALMS是一个部署在生产环境的对话式系统，结合嵌入向量检索与大语言模型NL2SQL技术，让营销人员用自然语言在数秒内对数百万级、上千属性的高维用户画像完成精确受众规模估算。 |
| [^65] | [Don't CLAP: Are Music-Text Models Bag-of-Words?](https://arxiv.org/abs/2609.30540) | 本研究通过属性交换扰动实验发现，现有对比式音乐-文本模型的文本嵌入无法可靠捕捉乐器与属性之间的绑定关系，揭示了CLAP分数作为文本-音乐忠实度评估指标存在根本性局限。 |
| [^66] | [Feeding BabyLMs Macaroni: Code-Switching Curricula Cause Cross-Lingual Convergence](https://arxiv.org/abs/2609.30535) | 该论文的核心贡献是发现在预训练语料中插入词级和句子级语码转换数据可以诱导语言模型的跨语言表示对齐（尤其跨越不同文字系统），且这种对齐能在后续单语训练中保持，配合“词级→句子级语码转换→单语”的渐进课程可在BabyLM评测上超越基线。 |
| [^67] | [Inquesto Score: A reliability Protocol For Voice Agents](https://arxiv.org/abs/2609.30514) | 本文提出 Inquesto Score（IS）协议，通过明确定义失败事件与严重级别，以固定版本化评估群体中成功完成呼叫者目标且无功能性故障的通话占比，来可复现、可解释地衡量已部署语音智能体的可靠性。 |
| [^68] | [Breaking Homogeneity: Diversifying Persona Sets for Creative LLM Outputs](https://arxiv.org/abs/2609.30492) | 该论文将人格多样化建模为集合级条件化问题，在“选择vs生成”与“空间填充vs前沿探索”两个正交设计维度上提出四种方法，显著提升了语言模型创造性输出的多样性，其中进化式人格生成在AUT上将回答多样性提高了78.8%。 |
| [^69] | [AcoustiClaim: A Numeric Claim Benchmark with Instrument Ground Truth](https://arxiv.org/abs/2609.30483) | 该论文提出 AcoustiClaim 基准，首次以定义声学量的仪器读数作为真值来验证音频语言模型给出的数值声明，发现绝大多数模型的表现不优于常数预测器，仅一个闭源模型在音高等少数任务上通过了秩相关阈值。 |
| [^70] | [Asymmetric Classifier-Free Guidance for Target-Speaker ASR](https://arxiv.org/abs/2609.30476) | 该论文提出了一种基于 Whisper 的非对称无分类器引导方法，通过在推理时使用轻量级预测器逐句动态调整说话人条件化强度，在领域偏移条件下将目标说话人语音识别的词错误率相对降低高达 21.8%。 |
| [^71] | [CARGO: Context-Aware Retrieval-Gated Evaluation of Agentic AI in Production](https://arxiv.org/abs/2609.30471) | 针对生产环境中智能体AI评估时“参考-实例分歧”（即参考答案描述的是不同实体导致正确回答被误判为错误）的问题，提出CARGO框架，将检索参考视为流程范例、基于实时实例上下文判定事实并将评估门控于检索置信度之上。 |
| [^72] | [Where Does Retrieval-Based Open-Ended Evaluation Fail? Automatic Taxonomy Induction from Long-Form Medical Answer Factuality Verification](https://arxiv.org/abs/2609.30467) | 该论文针对基于检索的开放式医学事实性验证，自动归纳出两套错误分类体系，将失败分解为五个质量维度上的检索阶段错误与六个连续步骤中的验证器推理错误，并利用LLM-as-Judge流程实现大规模自动化错误标注与分类，无需黄金答案或黄金证据。 |
| [^73] | [RAZOR: Pruning Replaceable Experts in LLMs](https://arxiv.org/abs/2609.30465) | 该论文提出无需训练的 MoE 专家剪枝方法 RAZOR，利用共识残差衡量专家的功能可替换性，在固定剪枝预算下剪除可被存活专家替代的专家，无需梯度或恢复训练即可最大程度保留原始模型输出分布。 |
| [^74] | [Inference-Time Target Speaker Unlearning in LLM-Based Automatic Speech Recognition](https://arxiv.org/abs/2609.30439) | 提出目标说话人遗忘语音识别（TSU-ASR）新任务，并设计可附加于冻结双流语音大语言模型的轻量级注册条件门控（ECG）模块，能在推理时动态阻止指定说话人的语音被转写、同时保留其说话活动的指示。 |
| [^75] | [All In Good Time: Causality-Aware Framework for LLM-Based Simultaneous Speech-to-Speech Translation](https://arxiv.org/abs/2609.30416) | 该论文提出了一种因果感知的同声语音到语音翻译框架FAST-CAP，通过分解式S2ST架构、因果感知自适应策略和因果感知延迟度量，并配合生成高保真因果对齐数据的新型数据流水线，在显著降低延迟的同时提升了翻译质量。 |
| [^76] | [A Unified Account of Concepts and Chunks](https://arxiv.org/abs/2609.30414) | 本文提出将认知心理学中原本分离的概念与组块统一到一个理论框架中，扩展Cobweb分类模型并实现TRELLIS系统，成功应用于同时包含概念性和组块性元素的上下文无关文法学习。 |
| [^77] | [What Improves Multimodal Misinformation Detection? Answers from a Large-Scale Empirical Study](https://arxiv.org/abs/2609.30402) | 本文通过涵盖3,375余次实验的大规模实证研究，系统性地回答了哪些设计选择能改进多模态虚假信息检测、它们何时会悄然失效，为构建更强大可靠的检测系统提供了实用指导。 |
| [^78] | [When Is a Multi-Agent Code Judge Actually Grounded? Two Label-Free Measurements, and a Judge That Declines to Guess](https://arxiv.org/abs/2609.30328) | 该论文指出多智能体验证框架在代码评判中因证据无法满足“独立于答案且能区分候选解”这两个条件而失效（在78%–95%的比较中判定两个解同样好、准确率仅4.4%），据此提出两种无需标签的度量方法来检验评判是否真正有据可依，并设计了一个在证据不足时拒绝猜测的评判器。 |
| [^79] | [PALM: Point-in-Time Adaptation for Financial Language Models](https://arxiv.org/abs/2609.30316) | 本文通过对比实验发现金融时点语言模型每年完整预训练并非必要——新旧检查点在同一评估窗口上表现相当，据此提出PALM方法，仅需在新增文本上拟合低秩适配器即可实现时点自适应，从而大幅降低维护时点语言模型的成本。 |
| [^80] | [A Benchmark Framework for Screening Automation in Systematic Reviews](https://arxiv.org/abs/2609.30298) | 该论文提出了包含45,064条标注条目的基准数据集SRBench、考虑类别不平衡的评估框架以及PromptSR实验工具，用于评估和支持大语言模型在系统综述筛选自动化中的应用。 |
| [^81] | [Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops](https://arxiv.org/abs/2609.30297) | Spotify提出了一条多轮合成数据生成流水线与自我改进循环，通过基于方差的对比优化和编码智能体的迭代修复，在冷启动场景下自动优化对话式推荐智能体的规划与工具调用能力，使质量提升8%。 |
| [^82] | [SignTrace: Describe a Sign, Find the Word](https://arxiv.org/abs/2609.30295) | SignTrace 利用大语言模型增强的中国手语词典，结合动作提取、七路检索与候选重排序，让学习者仅凭日常语言描述的手部动作即可反向查找到对应手语词条及其含义，在 500 条查询的基准上达到 94.0% 的 Hit@1。 |
| [^83] | [SlideLab: Audience-Centered Scientific Slide Generation and Evaluation](https://arxiv.org/abs/2609.30294) | SlideLab 是一个无需训练的多智能体框架，能从研究论文生成以观众为中心的科学演示幻灯片，在盲测中于 77% 的论文上超越开源与商业系统且推理成本降低约 4 倍，并配套提出模拟会议室的观众导向评估框架 ConfArena。 |
| [^84] | [Cartograph: Federated Tool Discovery with Operator-Attested Retrieval for AI Agents](https://arxiv.org/abs/2609.30293) | Cartograph是一个联邦式MCP代理，通过操作者签名的能力卡片、三层易混淆聚类分析（Rift）以及“先服务器后工具”的两阶段检索，将AI智能体的工具发现从O(n)目录遍历优化为O(k)渐进式披露，仅需暴露3个代理工具即可在374个工具的部署中取得0.816的R@5召回率，显著优于关键词基线。 |
| [^85] | [A Survey on Fake Review Detection: From Pre-trained Language Models to Large Language Models](https://arxiv.org/abs/2609.30292) | 本综述从信息融合视角系统梳理了2018年至2026年初的211项虚假评论检测研究，按证据来源和融合层次组织现有工作，并分析了预训练语言模型和大语言模型对虚假评论的生成与检测带来的双重影响。 |
| [^86] | [Auditing and Repairing LLM-as-Judge Failures in a Production Text-to-SQL Pipeline](https://arxiv.org/abs/2609.30290) | 本文审计了生产级文本到SQL流水线中的LLM裁判，发现其与人工标注一致性极低且根源于“评分幻觉”这一单一机制，并提出用低成本自托管Qwen模型替换裁判、配合三个强裁判的一致同意集成（kappa = 0.79、自动覆盖率 89.7%）来有效修复该失效问题。 |
| [^87] | [Not All Memories Are Equal: Hierarchical Collaborative Memory for Validity-Aware Retrieval in LLM Agents](https://arxiv.org/abs/2609.30289) | 提出HiCoMER框架，通过层级化协同记忆管理与有效性感知检索，避免协作式LLM智能体检索到过时或冲突的记忆，确保回答基于当前有效信息。 |
| [^88] | [Manifold Projection and Iterative Autoencoder Refinement for Masked Language Modeling](https://arxiv.org/abs/2609.30288) | 该论文提出用低秩瓶颈自编码器混合模块（分别在局部邻域、全序列和注意力头间运作）替代注意力机制，并在掩码位置引入由“拉动”和“校正”两步组成的迭代细化程序，用于掩码语言建模。 |
| [^89] | [A Mechanistic Study of AI-Text Detection Neurons in Frozen BERT: Sparse Probing and Activation Patching on RAID](https://arxiv.org/abs/2609.30287) | 该研究通过稀疏探测和双向激活修补方法，在冻结的BERT中定位出一组占比不足1%的神经元，证明它们因果性地支持跨六个生成器的AI生成文本检测任务。 |
| [^90] | [PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations](https://arxiv.org/abs/2609.30094) | 提出PrivDrift审计基准，发现在LLM活跃对话中，用户披露的秘密即使经历话题漂移后仍高度可恢复（混合泄露率达38.7%–54.6%），且额外的话题漂移并不能可靠降低泄露风险。 |
| [^91] | [PUBG Ally: A Conversational Embodied Agent as an AI Teammate](https://arxiv.org/abs/2609.29837) | 该论文提出了PUBG Ally，一个面向《绝地求生》的语音对话式具身AI队友，通过将语言模型智能体的工具使用与实时游戏控制相结合，在严格延迟约束下感知动态游戏世界、与玩家自然交流并同步执行移动、战斗等游戏行动。 |
| [^92] | [Rufus-Air: An Open LLM Post-Training Recipe](https://arxiv.org/abs/2609.29421) | 本文提出了Rufus-Air，一个基于GLM-4.5-Air-Base的开放可复现的八阶段大模型后训练方案，并总结出高质量SFT奠定能力基础、难度过滤保持RL学习有效性、奖励可靠性指导阶段排序等关键实践经验。 |
| [^93] | [Likelihood Ranking doesn't Scale Like Prompting in LLMs](https://arxiv.org/abs/2609.29390) | 该研究通过对95个模型和10个数据集的分析发现，基于陈述性语句的似然排序准确率不随模型规模和指令微调显著提升，而提示回答则随规模急剧改善，揭示了两种评估协议之间的系统性分歧。 |
| [^94] | [ArGuard Shared Task: Harmful Content Detection in Arabic Memes and LLM Prompts](https://arxiv.org/abs/2609.29349) | ArGuard共享任务为阿拉伯语表情包多模态仇恨检测与LLM有害提示检测建立了评测基准，吸引35支队伍参赛，最佳系统在四个子任务上取得0.419至0.984不等的宏F1分数，其中细粒度表情包分类因标签稀疏和分布偏移而最具挑战性。 |
| [^95] | [Beyond Overlap: Estimating the Causal Effect of Benchmark Exposure](https://arxiv.org/abs/2609.27176) | 提出LeakScale干预框架，通过控制基准家族私有信息的暴露并构建全新可执行任务，首次因果量化了训练数据污染对模型评估准确率的实际影响，发现暴露使准确率提升7.17至27.31个百分点。 |
| [^96] | [COT-TTS: Audio Context-Aware Text-to-Speech with Chain-of-Thought Reasoning](https://arxiv.org/abs/2609.22697) | 提出了COT-TTS任务，通过思维链推理从历史对话音频中自然推断说话风格并合成指定音色的语音，同时构建了包含900万样本的大规模双语对话语音数据集和人工验证基准来支持该任务。 |
| [^97] | [Apollo Restore: A Foundation LLM for Historical Greek Optimized for Fill-in-the-Middle Restoration of Ancient Greek Texts](https://arxiv.org/abs/2609.22455) | Apollo Restore 是首个面向历史希腊语（乃至任何古代地中海语言）的 240 亿参数基础大语言模型，通过“中间填空”目标微调，能够在不知缺失文本长度的情况下修复残缺古希腊文本，并在长度平衡的评估指标下大幅超越已发表的最强模型。 |
| [^98] | [Rethinking Human-Aligned Evaluation: An Analysis of Semantic Metrics Beyond WER](https://arxiv.org/abs/2609.21663) | 该论文提出HATS-en英语数据集用于以人为中心的ASR评估，发现WER与人类判断的一致性在所有指标中最低，而最佳配置的SemDist语义指标与人类判断一致性最高，支持ASR评估从WER转向语义指标。 |
| [^99] | [Beyond Atomic Tokens: Factorizing Syllables for Language Model Pretraining](https://arxiv.org/abs/2609.21362) | 提出了一种音位分词器，将越南语和汉语的音节分解为声母、韵母和声调三个音系成分，以极小的词表（汉语仅112项、越南语仅256项）实现了更高的编码效率。 |
| [^100] | [State of Thought Enables Endogenous Reasoning](https://arxiv.org/abs/2609.16055) | 提出“思维状态”新推理范式，通过从模型内部信息传递中提取动力学-几何状态并用仅 582 参数的控制器在冻结模型上选择性激活历史推理支持，实现由模型内部状态主导的内生推理，摆脱外部控制。 |
| [^101] | [Calibrated Enough to Know, Not Calibrated to Act: Fabricated Evidence Makes LLM Agents Commit to the Unknowable](https://arxiv.org/abs/2608.27167) | 本文发现LLM代理在面临伪造的专业面板证据时，会显著提高对不可预测问题的承诺率，其行为受证据包装的权威性驱动而非信息真实性或模型信念，揭示了一种可定位的校准失败。 |
| [^102] | [The Communication Map of a Transformer](https://arxiv.org/abs/2608.22007) | 提出了一种从权重出发绘制变压器所有潜在通信通道的“通信图谱”方法，能高效计算并揭示大多数注意力头对的耦合或回避模式，且具有广泛适用性。 |
| [^103] | [Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching](https://arxiv.org/abs/2608.09444) | 本文提出了首个针对循环语言模型深度自适应推理的高效方法——连续深度批处理（CDB），通过在循环步骤之间动态重组批次、管理循环KV缓存并提前预测token退出时机，解决了不同循环深度token无法被标准批处理系统高效处理的问题。 |
| [^104] | [Small Is Enough: Per-User Style Rewriting of AI-Edited Text via LoRA Adapters](https://arxiv.org/abs/2607.29238) | InMyStyle提出一种隐私优先的单用户方案，仅对0.5B至7B的小型模型进行LoRA微调，即可让AI编辑的文本自动改写为符合个人写作风格，且实验表明小模型已足以胜任该改写任务。 |
| [^105] | [Attention-Discounted Adaptive Sampler for Masked Diffusion Language Models](https://arxiv.org/abs/2606.10829) | 提出免训练重排序规则ADAS，根据每个token对已选位置的注意力（按预测不确定性加权）贪婪地折扣其置信度分数，从而提升掩码扩散语言模型在低推理步数下的表现。 |
| [^106] | [INFUSER: Influence-Guided Self-Evolution Improves Reasoning](https://arxiv.org/abs/2606.09052) | INFUSER提出了一种影响力引导的自我进化框架，通过生成器与求解器的协同训练，利用优化器感知的影响力分数来改进问题生成，从而显著提升推理能力。 |
| [^107] | [Statistical Priors for Implicit Preferences: Decoupling Skill Selection as a Local Harness in Personal Agents](https://arxiv.org/abs/2606.05828) | 提出一种将统计偏好学习与语义意图解析严格解耦的轻量级本地框架，利用本地统计先验来调节远程LLM的技能选择决策，使个人智能体能够学习隐式用户偏好，并取得最低的累积遗憾和最高的测试准确率。 |
| [^108] | [Large Language Model Selection with Limited Annotations](https://arxiv.org/abs/2605.24981) | SELECT-LLM是首个大语言模型主动选择框架，通过基于期望信息增益的查询选择规则，仅需少量最具信息量的标注查询即可从开放或黑盒候选模型中识别出给定任务的最佳LLM。 |
| [^109] | [Generating Legal Commentaries from Case Databases via Retrieval, Clustering, and Generation](https://arxiv.org/abs/2605.24534) | 本文提出一种无需人工构建教义学框架的全自动流水线，通过检索、聚类与大语言模型生成技术，将德国联邦最高法院的数千份判决自动转化为针对《德国民法典》具体法条的法律评注，并通过人类专家与LLM评审的五维度评估验证了其可行性。 |
| [^110] | [Direct Preference Optimization for English-Mandarin Code-Switching Speech Recognition in Audio LLMs](https://arxiv.org/abs/2605.23975) | 通过直接偏好优化（DPO）对齐音频大语言模型，可有效解决英中语码转换语音识别中的语言遗漏、翻译替代转录和幻觉问题，使混合语言错误率最高降低89.6%。 |
| [^111] | [Asking For An Old Friend: Diagnosing and Mitigating Temporal Failure Modes in LLM-based Statutory Question Answering](https://arxiv.org/abs/2605.23497) | 本文构建了包含312个专家验证的时间敏感德国成文法问答基准，识别出大语言模型的两种时间性失效模式（截止后过时与新近偏好偏差），并通过基于事实日期提取和版本过滤的检索增强方法进行缓解。 |
| [^112] | [AcuityBench: Evaluating Clinical Acuity Identification and Uncertainty Alignment](https://arxiv.org/abs/2605.11398) | 该论文提出AcuityBench基准，通过统一五个公开数据集和四级急迫度框架，系统评估语言模型从用户医疗描述中识别就医紧急程度的能力，并纳入医生确认的模糊案例以衡量模型的不确定性对齐水平。 |
| [^113] | [StraTA: Incentivizing Agentic Reinforcement Learning with Strategic Trajectory Abstraction](https://arxiv.org/abs/2605.06642) | StraTA 提出了一种策略性轨迹抽象框架，通过在智能体强化学习中引入显式的轨迹级策略并联合训练策略生成与动作执行，显著提升了大语言模型智能体在长时程决策任务中的样本效率和最终性能。 |
| [^114] | [UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification](https://arxiv.org/abs/2605.06221) | UniPrefill提出了一种基于块级动态稀疏化的通用长上下文预填充加速方法，能够同时兼容全注意力模型与线性注意力、滑动窗口注意力等混合架构，并可与连续批处理等现代推理引擎机制集成。 |
| [^115] | [Human-1 by Josh Talks: A Full-Duplex Conversational Modeling Framework in Hindi using Real-World Conversations](https://arxiv.org/abs/2604.23295) | 该论文通过适配 Moshi 双工语音架构，利用 26,000 小时真实印地语自发对话数据，构建了首个开放、可复现的印地语全双工口语对话系统，实现对打断、重叠等自然对话行为的建模。 |
| [^116] | [VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation](https://arxiv.org/abs/2604.21375) | VLAA-GUI提出一个模块化GUI自动化框架，通过强制性完整性验证器杜绝无视觉证据的过早成功宣告、多层级循环断路器打破重复失败循环、以及按需在线搜索应对不熟悉元素，系统性地解决了GUI智能体的过早停止与重复循环两大核心难题。 |
| [^117] | [Closing the Speech-Text Gap with Limited Audio for Effective Domain Adaptation in LLM-Based ASR](https://arxiv.org/abs/2604.06487) | 提出混合批处理（MB）策略，仅用不到4小时的目标域语音数据即可使基于LLM的ASR达到与使用完整数据集传统微调相当或更优的词错误率，有效弥合了语音-文本模态差距。 |
| [^118] | [Combee: Scaling Prompt Learning for Self-Improving Language Model Agents](https://arxiv.org/abs/2604.04247) | Combee 提出了一种新颖的框架，通过有原则的并行扩展策略解决现有提示学习方法在高并行度下质量下降的问题，从而同时提升自我改进语言模型智能体提示学习的效率和效果。 |
| [^119] | [Layer-wise Target Propagation: Efficient Component Attribution through Target Centric Propagation](https://arxiv.org/abs/2603.19742) | 提出逐层目标传播（LTP）框架，仅需一次前向和一次反向传播即可在冻结的Transformer上忠实地追踪信息流，在模型组件数量方面实现O(1)时间复杂度的高效密集组件归因。 |
| [^120] | [Why Better Cross-Lingual Alignment Fails for Better Cross-Lingual Transfer: Case of Encoders](https://arxiv.org/abs/2603.18863) | 本研究通过对四个语言对上采用词元级、句子级和掩码语言建模等目标进行显式对齐的XLM-R编码器进行实验，发现基于嵌入的对齐指标无法可靠预测下游跨语言迁移性能，且对齐目标与下游任务的梯度往往近乎正交，从而揭示了更好的跨语言对齐为何并不必然带来更好的跨语言迁移。 |
| [^121] | [GT-HarmBench: Benchmarking AI Safety Risks Through the Lens of Game Theory](https://arxiv.org/abs/2602.12316) | 该论文提出GT-HarmBench——首个基于博弈论结构、包含1535个高风险场景的多智能体AI安全基准，发现前沿模型在38%的高风险场景中无法选择对社会有益的行动，而博弈论干预可将有益结果提升最高18%。 |
| [^122] | [FlyAOC: Evaluating Agentic Ontology Curation of Drosophila Scientific Knowledge Bases](https://arxiv.org/abs/2602.09163) | FlyAOC是一个评估AI智能体从科学文献中进行端到端本体策展的基准，要求智能体在16,898篇果蝇论文中检索证据，并恢复策展人级别的结构化基因标注，涵盖功能术语、表达模式和历史同义词。 |
| [^123] | [Affective Flow Language Model for Emotional Support Conversation](https://arxiv.org/abs/2602.08826) | 该论文提出情感流语言模型AFlow，将多轮情感支持建模为沿对话轨迹演化的情感效用流，并通过情感流偏好优化（AFPO）利用流平衡目标将下游偏好信号传播到中间对话状态，为多轮交互中的支持策略调整提供细粒度的过程监督。 |
| [^124] | [Stepwise Intrinsic Rewards for Reasoning in Large Language Models](https://arxiv.org/abs/2602.01034) | 提出了一种无需过程标注、辅助模型或推理时搜索的内在过程奖励方法——分步边际信息增益（MIG），通过衡量每个推理前缀对参考答案对数似然的提升，并结合单调水位线机制避免重复计分，从而为大语言模型的多步推理提供更精确的密集监督。 |
| [^125] | [Towards Automated Lexicography: Generating and Evaluating Definitions for Learner's Dictionaries](https://arxiv.org/abs/2601.01842) | 该论文提出了基于“LLM作为裁判”的词典释义生成评估方法（配合与专业词典编纂者合作构建的日语释义数据集），以及一种采用迭代简化策略的、基于大语言模型的学习词典释义生成方法。 |
| [^126] | [Does Understanding Inform Generation in Unified Multimodal Models? From Analysis to Path Forward](https://arxiv.org/abs/2511.20561) | 本文提出解耦评估框架UniSandbox，揭示统一多模态模型中理解与生成之间存在显著差距，并证明理解模块中的显式思维链（CoT）可有效弥合该差距，且可通过自训练将推理能力内化，实现生成时的隐式推理。 |
| [^127] | [Evaluation is All You Need: Strategic Overclaiming of LLM Reasoning Capabilities Through Evaluation Design](https://arxiv.org/abs/2506.04734) | 本研究揭示评估条件的细微差异会导致Deepseek-R1-Distill系列等推理模型的基准测试结果大幅波动，使其声称的性能提升难以可靠复现，并倡导建立更严格的模型性能评估范式。 |
| [^128] | [From ASR to ASP: Evaluating Prompt Attack Vulnerabilities Against Open-Source LLMs](https://arxiv.org/abs/2505.14368) | 本文提出攻击成功概率（ASP）这一新指标，以捕捉传统攻击成功率所忽视的模型响应不确定性，并在五个攻击基准上系统评估了14个开源和3个闭源大语言模型面对提示注入攻击的脆弱性。 |
| [^129] | [Achieving Tokenizer Flexibility in Language Models through Heuristic Adaptation and Supertoken Learning](https://arxiv.org/abs/2505.09738) | 提出了 Tokenadapt——一种与模型无关的分词器移植方法，结合面向多词“超词”的预分词学习，使语言模型能以较低计算成本灵活替换分词器，同时提升压缩效率并减少词元碎片化。 |
| [^130] | [MedHal: a Synthetic Dataset for Medical Hallucination Detection](https://arxiv.org/abs/2504.08596) | MedHal是一个涵盖内在与外在幻觉的大规模合成医学数据集，用于训练和评估医学文本幻觉检测模型，基于该数据集训练的基线模型优于通用幻觉检测方法。 |
| [^131] | [SkillFlow: Scalable and Efficient Agent Skill Retrieval System](https://arxiv.org/abs/2504.06188) | SkillFlow是首个面向智能体技能发现的多阶段检索系统，它将技能获取视为信息检索问题，通过密集检索、交叉编码器重排序和LLM选择四个阶段，从约3.5万个社区技能定义中高效检索出最相关的技能。 |
| [^132] | [NaijaNLP: A Survey of Nigerian Low-Resource Languages](https://arxiv.org/abs/2502.19784) | 该论文首次对尼日利亚三大主要语言（豪萨语、约鲁巴语、伊博语）的NLP研究现状进行了全面综述，定量评估了现有语言资源，发现在293项相关研究中仅27.6%贡献了新语言资源，揭示了该领域过度依赖复用现有数据而非创建新数据的关键挑战。 |

# 详细

[^1]: 无需学习停止而学会停止：自监督置信度训练提升推理效率

    Learning to Stop without Learning to Stop: Self-Supervised Confidence Training Improves Reasoning Efficiency

    [https://arxiv.org/abs/2609.31619](https://arxiv.org/abs/2609.31619)

    该论文发现，仅通过自监督方式训练模型在推理过程中间点预测自身置信度（损失函数中不含任何长度、效率或停止目标），就能让模型在推理时无需任何提前停止机制便自发提升推理效率。

    

    推理模型通常会生成非常长的推理轨迹，导致推理的计算成本很高。现有方法通常通过两种途径提升效率：一是在推理阶段引入提前停止机制，二是在训练过程中显式鼓励更短的推理，例如使用带长度惩罚的强化学习。我们证明，显著的效率提升可以来自另一种不同的监督信号：置信度。通过一种自监督流程，我们仅使用600个训练问题，对推理模型进行微调，使其在自身推理轨迹的中间点预测对答案的置信度。置信度仅被用作训练目标：损失函数中不包含任何关于推理长度、效率或停止的目标。在推理阶段，微调后的模型采用标准生成流程，无需置信度引导或提前停止机制。尽管如此，自监督……

    arXiv:2609.31619v1 Announce Type: new  Abstract: Reasoning models often generate very long reasoning traces, making inference computationally expensive. Existing approaches typically improve efficiency either through inference-time early-stopping mechanisms or by explicitly encouraging shorter reasoning during training, for example through reinforcement learning with length penalties. We show that substantial efficiency gains can instead emerge from a different kind of supervision: \textit{confidence}. Using a self-supervised procedure, we fine-tune reasoning models to predict their confidence in the answer at intermediate points along their own reasoning trajectories using only 600 training problems. Confidence is used only as a training target: the loss contains no objective for reasoning length, efficiency, or stopping. At inference, the fine-tuned models use the standard generation procedure, with no confidence elicitation or early-stopping mechanism. Despite this, self-supervised 
    
[^2]: 通过信念自蒸馏提取用户模型

    User Model Extraction via Belief Self-Distillation

    [https://arxiv.org/abs/2609.31603](https://arxiv.org/abs/2609.31603)

    提出信念自蒸馏框架，使冻结的大语言模型从自然对话中自我蒸馏出既能读出又能写回的用户信念表示，并揭示模型的拒绝行为取决于其推断出的用户意图。

    

    大型语言模型（LLM）会隐式地推断用户的属性并据此调整自身行为，然而这些信念一直难以被检查和进行因果性操纵。我们提出了信念自蒸馏，这是一个统一的读写框架，通过学习一种既可被解码、又可写回模型的紧凑用户表示，将线性探测与因果探测连接起来。冻结的LLM充当自己的教师，从自然对话中蒸馏出信念，无需任何外部标注。与传统的探测方法不同，BSD不仅分离出激活中存在的信息，还分离出一种其因果作用可被直接检验的状态。在多个模型家族上的实验表明，BSD能够忠实地恢复用户信念，并实现比匹配的隐藏状态引导强得多的干预效果。至关重要的是，我们发现模型的拒绝行为不仅取决于请求本身，还取决于模型推断出的用户意图：改变这一信念即可改变拒绝行为。

    arXiv:2609.31603v1 Announce Type: cross  Abstract: Large language models (LLMs) implicitly infer attributes of their users and adapt their behavior accordingly, yet these beliefs remain difficult to inspect and causally manipulate. We introduce Belief Self-Distillation (BSD), a unified read-write framework that bridges linear and causal probing by learning a compact user representation that can be both decoded and written back into the model. The frozen LLM acts as its own teacher, distilling beliefs from natural conversations without external annotations. Unlike conventional probing, BSD isolates not only information present in activations, but a state whose causal role can be directly tested. Across multiple model families, BSD faithfully recovers user beliefs and enables substantially stronger interventions than matched hidden-state steering. Crucially, we find that refusal depends not only on the request, but on the model's inferred user intent: changing this belief alters refusal 
    
[^3]: 面向编程智能体的紧凑文档：一个基准、一个优化器，以及它为何无法迁移

    Compact Documentation for Coding Agents: A Benchmark, an Optimizer, and Why It Does Not Transfer

    [https://arxiv.org/abs/2609.31587](https://arxiv.org/abs/2609.31587)

    本文提出了一个通过“重新生成代码能否通过原始测试”来评估代码描述的往返基准和优化器，发现完整性而非长度决定文档保真度，但令人意外的负面结果表明：当源代码可用时，更好的文档并不能帮助编程智能体解决真实的仓库问题。

    

    我们研究了自然语言文档是否能帮助编程智能体解决软件问题，并构建了用于生成和评估此类文档的工具。我们引入了一个往返式基准，通过“根据代码描述重新生成的代码能否通过原始测试”来为描述打分，并证明决定描述保真度的是完整性而非长度。以该基准作为优化信号，我们发现了一条描述撰写提示词，可达到完全保真度并能泛化到未见过的文件。随后，我们检验了这项工作最初的动机假设：更好的文档能帮助智能体解决真实的仓库问题。我们在两个模型家族和十个仓库上进行了测试，并设置了一个阳性对照以确认我们的评估能够检测出真正的改进，但结果表明情况并非如此。当源代码存在时，无论是静态紧凑文档还是检索到的上下文，都不如仅凭问题描述有效。我们将这一负面结果与……（原文截断）

    arXiv:2609.31587v1 Announce Type: cross  Abstract: We investigate whether natural-language documentation helps coding agents resolve software issues, and we build the tools to construct and evaluate it. We introduce a roundtrip benchmark that scores code descriptions by whether code regenerated from them passes the original tests, and show that completeness, not length, drives a description's fidelity. Using the benchmark as an optimization signal, we discover a description-writing prompt that reaches full fidelity and generalizes to unseen files. We then test the hypothesis that motivated the work: that better documentation helps an agent resolve real repository issues. Across two model families and ten repositories, and against a positive control confirming that our evaluation can detect a genuine improvement, we find that it does not. When the source is present, neither static compact documentation nor retrieved context beats the issue alone. We report this negative result together 
    
[^4]: 面向自训练的策略多样性采样

    Strategically Diverse Sampling for Self-Training

    [https://arxiv.org/abs/2609.31571](https://arxiv.org/abs/2609.31571)

    该论文提出以策略多样性作为自训练数据构建的新原则，通过GROOT（构建方法层次树并采样不同路径）和语言化采样两种方法生成策略多样的训练数据，使模型在困难任务上优于传统IID采样训练的模型。

    

    许多大型语言模型的训练和推理方法，包括强化学习（RL）和测试时扩展，都依赖于重复采样，但只有当响应之间存在实质性差异时才能获益。自训练面临同样的挑战：训练数据通常通过采样独立同分布（IID）的响应并主要以正确性为标准进行过滤来构建，从而过度代表了模型已经偏好的策略。我们研究策略多样性，即解决问题所用方法之间的实质性差异，作为构建自训练数据的另一种原则。我们使用两种采样方法生成具有策略多样性的数据：GROOT，一种新方法，它构建方法层次的树结构并采样不同的路径；以及语言化采样，经改造后用于产生非结构化的方法集合。在竞赛编程和下一章预测领域中，在策略采样数据上训练的模型在困难任务上优于IID训练的对应模型。

    arXiv:2609.31571v1 Announce Type: new  Abstract: Many LLM training and inference methods, including RL and test-time scaling, depend on repeated sampling, but benefit only when the responses meaningfully differ. Self-training faces the same challenge: training data is typically constructed by sampling IID responses and filtering primarily for correctness, thereby overrepresenting strategies a model already favours. We investigate strategic diversity, or substantive variation among approaches to a problem, as an alternative principle for constructing self-training data. We generate strategically diverse data with two sampling methods: GROOT, a new method which constructs a hierarchical tree of approaches and samples distinct paths, and Verbalized Sampling (VS), adapted to produce an unstructured set of approaches. Across competitive programming and Next-Chapter Prediction domains, models trained on strategically sampled data outperform IID-trained counterparts on difficult tasks and pro
    
[^5]: MexHat：一个用于墨西哥西班牙语视频仇恨言论检测的数据集

    MexHat: A Dataset for Hate Speech Detection in Mexican Spanish Videos

    [https://arxiv.org/abs/2609.31553](https://arxiv.org/abs/2609.31553)

    本文提出了MexHat数据集，这是一个包含约1000个墨西哥西班牙语视频片段的仇恨言论检测数据集，通过三分类和细粒度子类别两种标注方式，捕捉了该语言文化背景下的语言和文化线索。

    

    通过内容监控确保在线安全使仇恨言论检测成为一项亟待解决的关键任务。从本质上讲，该任务需要捕捉上下文线索，这对于精确理解内容意图至关重要。尽管针对该任务的自动检测方法已经取得了显著进展，但非英语资源的稀缺性依然存在，限制了模型适应多模态内容中微妙的、依赖上下文的以及与文化相关特性的能力。在本文中，我们介绍了MexHat，这是一个视频数据集，旨在捕捉墨西哥西班牙语语境下仇恨言论检测任务所需的语言和文化线索。我们的数据集包含约1000个视频片段，并针对两个任务进行了标注：一个三分类评估（无负面内容、攻击性内容和仇恨言论内容），以及一个包含三个仇恨言论子类别的细粒度类别评估。

    arXiv:2609.31553v1 Announce Type: new  Abstract: Ensuring online safety through content monitoring had raised Hate Speech Detection as a crucial task to be addressed. By essence the task demands the capture of contextual cues, which are essential for a precise understanding of the content's intent. Although automated detection approaches for the task have advanced significantly, the scarcity of non-English resources persists, limiting the ability of models to adapt to the subtle, context-dependent, and culturally related nature of multimodal content. In this paper, we introduce MexHat, a video dataset designed to capture the linguistic and cultural cues for the hate-speech detection task in a Mexican Spanish context. Our dataset comprises around 1k video clips annotated across two tasks: a three-way class evaluation (no negative content, offensive content and hate-speech content), and a fine-grained class evaluation including three hate-speech sub-categories. The dataset statistics and
    
[^6]: 面向文档内自适应AI文本筛查的两种共形构造方法

    Two Conformal Constructions for Adaptive Within-Document AI-Text Screening

    [https://arxiv.org/abs/2609.31547](https://arxiv.org/abs/2609.31547)

    该论文提出两种基于共形方法的有限样本构造，在文档级可交换性假设下实现对自适应文档内AI文本筛查的误报警率控制，并支持提前停止且无需对文档内词元依赖性做任何假设。

    

    我们研究在对人工智能（AI）生成的文本进行筛查时的误报警控制问题。该筛查程序根据观测到的证据选择文档前缀和检测器，并可能在耗尽其检查预算之前提前停止。我们在人类校准文档与新的零假设文档之间的文档级可交换性条件下，给出了两种有限样本构造，且对文档内部词元之间的依赖性没有任何限制。构造A注册一个有限的前缀-检测器分数族，并在它们的共形排名之间分配误报警预算，通过联合界保护该族中任何被执行的子集。构造B校准由开发阶段固定的自适应策略的完整路径最大值，由于每个部分路径最大值都被完整路径最大值所界定，因此终端共形排名可以在不分割误差预算的情况下保护提前停止。我们证明了在允许的检查过程中对任何误报警的边际控制。

    arXiv:2609.31547v1 Announce Type: cross  Abstract: We study false-alert control when screening for text generated by artificial intelligence (AI). The screening procedure selects document prefixes and detectors from observed evidence and may stop before exhausting its inspection budget. We give two finite-sample constructions under document-level exchangeability between human calibration documents and a new null document, with no restriction on dependence among tokens within a document. Construction A registers a finite family of prefix-detector scores and allocates a false-alert budget across their conformal ranks. A union bound protects any executed subset of that family. Construction B calibrates the complete-path maximum of a development-fixed adaptive policy. Each partial-path maximum is bounded by the complete maximum, so a terminal conformal rank protects early stopping without splitting the error budget. We prove marginal control of any false alert across the permitted inspecti
    
[^7]: Google Play用户评论情感指数的统计学基础：信号融合、收缩估计、分布验证与动态平滑

    Statistical Foundations for a Google Play User-Review Sentiment Index: Signal Fusion, Shrinkage, Distributional Validation, and Dynamic Smoothing

    [https://arxiv.org/abs/2609.31513](https://arxiv.org/abs/2609.31513)

    该论文为Google Play用户评论构建了具有严格数学证明的显式情感指数，核心创新在于以协方差感知的逆方差加权融合星级与文本情感信号、基于估计精度向总体均值收缩、针对离散评分的分布验证以及卡尔曼滤波动态平滑。

    

    我们为Google Play用户评论开发了一个具有明确统计学框架的情感指数，并建立了支持其构建的数学结果。归一化的星级评分与文本情感分数被视为潜在评论效价的带噪测量，通过协方差感知的逆方差加权进行融合。评论级估计采用有界的有用性与时效性权重进行聚合，随后基于估计精度（而非任意的评论数量阈值）向总体均值进行收缩。应用级评分直方图为不同API排序方式下返回的样本提供分布诊断；由于星级评分是离散的，故不采用经典的连续Kolmogorov-Smirnov临界值。局部水平状态空间模型与卡尔曼滤波用于提供去噪后的时间趋势。完整的证明涵盖了BLUE（最优线性无偏估计）与高斯极大似然结果、高斯共轭收缩、Glivenko-Cantelli与Donsker定理（原文此处截断）。

    arXiv:2609.31513v1 Announce Type: cross  Abstract: We develop a statistically explicit sentiment index for Google Play user reviews and establish the mathematical results supporting its construction. Normalized star ratings and text-sentiment scores are treated as noisy measures of latent review valence and fused by covariance-aware inverse-variance weighting. Review-level estimates are aggregated with bounded helpfulness and recency weights, then shrunk toward a population mean using estimated precision rather than an arbitrary review-count threshold. App-level rating histograms provide a distributional diagnostic for samples returned under different API sort orders; because star ratings are discrete, classical continuous Kolmogorov-Smirnov critical values are not used. A local-level state-space model and the Kalman filter provide a denoised temporal trend. Full proofs cover the BLUE and Gaussian maximum-likelihood result, Gaussian-conjugate shrinkage, the Glivenko-Cantelli and Donske
    
[^8]: Muslim：一个已部署的、提供有据可依伊斯兰知识的阿拉伯语语音AI平台

    Muslim: A Deployed Arabic Voice AI Platform for Grounded Islamic Knowledge

    [https://arxiv.org/abs/2609.31511](https://arxiv.org/abs/2609.31511)

    该论文推出了已部署的阿拉伯语语音AI生产平台Muslim，其核心贡献包括发布微调模型族（工具路由模型Muslim-6B-PRO与TTS模型Fasih-TTS-V1）、将开放演示转变为可运营防滥用产品的账户计量层，以及三层可观测性体系。

    

    我们推出了Muslim，一个面向真实用户、提供有据可查、有出处的伊斯兰知识的阿拉伯语语音AI生产级平台。除了实时语音流水线（NeMo阿拉伯语语音识别、兼容OpenAI的LLM端点、自托管TTS）以及跨六个模型上下文协议服务器进行路由的确定性多源检索层之外，我们报告了研究原型通常缺乏的三个要素。第一，发布了一族经过微调的阿拉伯语伊斯兰模型产物：一个高效的工具路由大语言模型（Muslim-6B-PRO，59.4亿参数）和一个现代标准阿拉伯语TTS模型（Fasih-TTS-V1），后者在社区投票的现代标准阿拉伯语TTS竞技场上在17个系统中总体排名第5，在11个开放权重系统中排名第2。第二，一个账户与计量层——每账户免费对话轮次配额、容量感知的拒绝机制，以及推迟到真正需要时才启用的邮箱验证——它将一个开放演示转变为可运营、防滥用的产品。第三，一个三层可观测性体系……

    arXiv:2609.31511v1 Announce Type: new  Abstract: We present Muslim, a production Arabic voice AI platform serving grounded, sourced Islamic knowledge to real users. Beyond a real-time voice pipeline (NeMo Arabic ASR, an OpenAI-compatible LLM endpoint, self-hosted TTS) and a deterministic multi-source retrieval layer routed across six Model Context Protocol servers, we report three things a research prototype typically lacks. First, a released family of fine-tuned Arabic Islamic model artifacts: an efficient tool-routing LLM (Muslim-6B-PRO, 5.94B parameters) and a Modern Standard Arabic TTS model (Fasih-TTS-V1) that ranks 5th of 17 overall and 2nd of 11 open-weight systems on the community-voted Arabic TTS Arena for MSA. Second, an account and metering layer - a free per-account turn allowance, capacity-aware refusal, and email verification deferred to the point it actually matters - that turns an open demo into an operable, abuse-resistant product. Third, a three-layer observability st
    
[^9]: 评估大语言模型对海地克里奥尔语的文化意识

    Evaluating Cultural Awareness of LLMs for Haitian Creole

    [https://arxiv.org/abs/2609.31506](https://arxiv.org/abs/2609.31506)

    该论文首次从特异性、偏见、多样性和变异性四个维度系统评估了大语言模型对海地克里奥尔语的文化意识，发现其明显落后于法语，且海地角色常被以苦难形象呈现。

    

    大语言模型（LLMs）在高资源语言与低资源语言之间表现出显著的性能差异。除了任务性能较低之外，它们往往无法捕捉代表性不足群体的文化规范和价值观。在这项工作中，我们对大语言模型在海地克里奥尔语上的文化意识进行了首次系统性评估——海地克里奥尔语是一种有数百万人使用的语言，但在数字资源中却严重缺乏代表性。我们沿着四个互补的维度——特异性、偏见、多样性和变异性——对文化意识进行评估，使用了一个由母语者精心策划的文化显著性提示基准，并采用文本填充（text infilling）的实验设置。我们的结果揭示了海地克里奥尔语与资源更丰富的法语之间在文化意识上存在明显差距，海地语的表现不仅在各领域间更加不均衡，而且更容易受到法语的语言干扰。故事生成任务还进一步揭示了一个反复出现的现象：海地角色总是被通过苦难的形象来描绘。

    arXiv:2609.31506v1 Announce Type: cross  Abstract: Large language models (LLMs) exhibit substantial performance disparities between high- and low-resource languages. Beyond lower task performance, they often fail to capture the cultural norms and values of underrepresented communities. In this work, we present the first systematic evaluation of cultural awareness in LLMs for Haitian Creole, a language spoken by millions but severely underrepresented in digital resources. We assess cultural awareness along four complementary dimensions---specificity, bias, diversity, and variation---using a benchmark of culturally salient prompts curated by native speakers in a text infilling setting. Our results reveal a clear gap between cultural awareness in Haitian Creole and higher-resource French, with Haitian performance being more uneven across domains and more affected by French linguistic interference. Story generation further reveals recurring portrayals of Haitian characters through hardship
    
[^10]: PriceBench：用于诊断LLM预订代理中价格、质量与品牌偏好的基准测试

    PriceBench: A Diagnostic Benchmark for Price, Quality, and Brand Preferences in LLM Booking Agents

    [https://arxiv.org/abs/2609.31468](https://arxiv.org/abs/2609.31468)

    PriceBench通过logit选择模型从LLM的酒店预订行为中诊断其价格、质量和品牌偏好，发现更强的模型偏好更一致但各不相同，而较弱的模型要么偏好僵化易被列表顺序操纵，要么近乎随机选择。

    

    LLM正越来越多地充当购买代理，这意味着是LLM而非用户在满足请求的各个选项中进行选择；其偏好悄然决定了最终买什么以及花多少钱。酒店预订是一个典型的场景：这是一个高频发生的选择过程，基于几个可比较的属性做出决定，而其选择能够揭示这些偏好。我们提出了PriceBench，一个诊断性基准，通过logit选择模型从LLM的预订选择中恢复其价格、质量和品牌偏好。该基准应用于来自8家提供商的28个LLM，涵盖179家真实纽约市酒店的3,600个酒店任务。我们发现，模型能力与LLM选择的一致性相关，而非与其选择的内容相关：能力更强的LLM持有更强、更一致的偏好，而能力较弱的模型要么锁定于某一种选择（这种僵化偏好可被控制列表顺序的一方所利用），要么几乎是无差别地随机选择。这些偏好所倾向的内容在不同提供商之间差异显著。

    arXiv:2609.31468v1 Announce Type: cross  Abstract: LLMs increasingly act as purchasing agents, which makes the LLM, not the user, the one choosing among the options that satisfy a request; its preferences quietly fix what gets bought and what it costs. Hotel booking is a clean instance: a high-volume choice settled on a few comparable attributes, where the pick reveals those preferences. We introduce PriceBench, a diagnostic benchmark that recovers an LLM's price, quality, and brand preferences from its booking choices with a logit choice model, applied to 28 LLMs from 8 providers on 3,600 hotel tasks from 179 real New York City properties. We find that capability is associated with how consistently an LLM chooses, not with what it chooses: more capable LLMs hold stronger, more consistent preferences, while weaker ones either lock onto one position, exploitable by whoever controls listing order, or choose almost indifferently. What those preferences favor varies sharply across provider
    
[^11]: ViSTA：一个简单的桥梁，将视觉对齐扩展至多模态大语言模型的临床时间序列理解

    ViSTA: A Simple Bridge Extends Visual Alignment to Clinical Time-Series Understanding in Multimodal LLMs

    [https://arxiv.org/abs/2609.31448](https://arxiv.org/abs/2609.31448)

    ViSTA是一种轻量级适配器，通过将不规则的数值型临床时间序列数据融入预训练视觉-语言模型的图表表示中，在冻结全部预训练参数的前提下实现临床风险预测性能的大幅提升。

    

    临床预测模型根据患者的测量数据估计风险，而大语言模型则支持医学文本理解和问答。然而，它们的语言能力并不能保证能够从结构化、高维的临床时间序列中进行准确预测。提升这一能力将把风险估计与对患者病情演变的灵活提问连接起来。我们提出了ViSTA，这是一个紧凑的适配器，将不规则的数值测量数据融入预训练视觉-语言模型的图表表示中。它在保持所有预训练参数完全不变的情况下，学习对视觉token的校正。在MIMIC-IV数据集上，对于急性肾损伤和死亡率预测，在参数规模为20亿至90亿的模型中，ViSTA在所有四项指标上均获得了所比较的适配方法中最高的平均分数。仅使用51.6万个可训练参数，20亿参数的模型在急性肾损伤预测中达到了0.7376的ROC曲线下面积。

    arXiv:2609.31448v1 Announce Type: cross  Abstract: Clinical prediction models estimate risk from patient measurements, while large language models support medical text understanding and question answering. Yet their language capabilities do not ensure accurate prediction from structured, high-dimensional clinical time series. Improving this ability would connect risk estimation with flexible questions about a patient's evolving condition. We introduce ViSTA, a compact adapter that incorporates irregular numerical measurements into a pretrained vision-language model's chart representations. It learns corrections to visual tokens while leaving all pretrained parameters unchanged. On MIMIC-IV, ViSTA has the highest mean scores among the compared adaptations on all four metrics for acute kidney injury and mortality prediction across models with 2-9 billion parameters. With 0.516 million trainable parameters, the 2-billion-parameter model reaches an area under the ROC curve of 0.7376 for ac
    
[^12]: 缓解伪造共识：面向多智能体辩论综合的主动溯源门控

    Towards Mitigating Fabricated Consensus: The Active Provenance Gate for Multi-Agent Debate Synthesis

    [https://arxiv.org/abs/2609.31422](https://arxiv.org/abs/2609.31422)

    提出主动溯源门控（APG），通过将来源作为硬约束、审计辩论中的每一条论断并自我纠错，来缓解多智能体辩论综合阶段摘要模型伪造无事实依据“辩论共识”的安全问题。

    

    基于大语言模型的多智能体辩论（MAD）系统正日益被用作分布式流程中的复杂决策管道，然而其最终综合阶段仍然缺乏足够的控制。即使拥有详细的辩论记录，摘要模型也容易编造出文笔流畅、但并未建立在辩论历史基础之上的“辩论共识”。为解决这一安全缺口，本文开展实证研究，探讨引入辩论后的主动验证能否在减少此类缺乏事实依据的摘要产生的同时，仍提供有价值的信息；此外，还检验了在缺乏可靠折中方案的情况下，显式发出分歧信号是否更为可取。本文提出了主动溯源门控作为辩论后验证层，它将来源作为硬约束，分析辩论记录、审计每一条论断，并应用自我纠错。

    arXiv:2609.31422v1 Announce Type: cross  Abstract: Large language model-based multi-agent debate (MAD) systems are being increasingly used as complex decision pipelines in distributed processes. However, their final synthesis phase still remains inadequately controlled. Even with detailed debate logs, summarizing models are prone to fabricating smoothly written debate consensus that is not grounded in the debate's history. To address this safety gap, this paper presents empirical research and studies if the introduction of active post-debate verification can mitigate the production of such factually unsupported summaries, while still providing valuable information. Furthermore, it is examined whether explicitly signalling divergence is preferable in the absence of a reliable compromise. The Active Provenance Gate (APG) is introduced as a post-debate verification layer that treats the source as a hard constraint, analysing the debate logs, auditing each claim, and applying self-correcti
    
[^13]: 对不起机器人，开心人类：视觉语言模型只能读取两层可读印刷文字中的一层

    Sorry Robot, Happy Human: Vision-Language Models Read Only One of Two Legible Typographic Layers

    [https://arxiv.org/abs/2609.31403](https://arxiv.org/abs/2609.31403)

    本研究创建了DecoyBench数据集，发现视觉语言模型在包含两层叠加可读文字的图像中只能读取轮廓文字而几乎无法提取阴影文字，而人类可以高准确率读取两层，揭示了VLM在多文本层图像理解上的根本性脆弱。

    

    尽管视觉语言模型（VLM）在光学字符识别（OCR）任务中取得了成功，但它们容易受到排版攻击，并且对包含多个文本层的图像结构脆弱。在本研究中，使用诱饵字体（Decoy Font）方法创建了DecoyBench数据集。该数据集由300张图像组成，每张图像包含带有清晰轮廓线的文字，叠加在另一段带有柔和阴影的文字之上。使用该数据集，在两种不同的提示条件（朴素提示和引导提示）以及两种不同的分辨率（512×512和64×64）下，评估了来自三个不同模型系列的六个最新闭源模型。验证研究表明，人类参与者能够以高准确率读取两层文本。相比之下，大多数模型变体在两种提示方法下，都能在高分辨率下以接近人类的准确率读取轮廓文字，但几乎从未完整提取出阴影文字。在低分辨率下，

    arXiv:2609.31403v1 Announce Type: cross  Abstract: Vision-language models (VLMs), despite their success in optical character recognition (OCR) tasks, are vulnerable to typographic attacks and have a fragile structure for images with multiple text layers. In this study, the DecoyBench dataset was created using the Decoy Font method. The dataset consists of 300 images, each containing text with sharp contour lines superimposed on another text with soft shading. Six recent closed-source models from three different model families were evaluated using this dataset under two different prompting conditions (naive and guided) and at two different resolutions ($512\times512$ and $64\times64$). A validation study showed that human participants could read both text layers with high accuracy. In contrast, the models, with most variants and both prompting methods, read the contour text with near-human accuracy at high resolution, but almost never fully extracted the shading text. At low resolution,
    
[^14]: Intent2Tc：基于语言模型的意图到流量控制自动转换

    Intent2Tc: Automated Intent-to-Traffic Control Translation with Language Models

    [https://arxiv.org/abs/2609.31397](https://arxiv.org/abs/2609.31397)

    该论文提出Intent2Tc，一个由语言模型驱动的闭环框架，能够将业务级流量整形意图自动转换为经过验证的可执行Linux流量控制配置，并通过数字孪生语义模型、批判驱动精炼和RAG知识复用显著提升语义一致性与配置可靠性。

    

    自动化且高度可用的服务质量保障需要将高层服务意图转换为可部署的流量管理策略。尽管基于意图的网络简化了策略的规范定义，但在业务级意图与可执行网络配置之间架起桥梁仍然复杂、易出错且难以自动化。本文提出了Intent2Tc，一个由语言模型驱动的闭环框架，它将业务级的流量整形意图转换为声明式的子意图，进而转换为经过验证的、可执行的Linux流量控制配置。该框架集成了基于主动队列管理的数字孪生语义模型、自动化元数据提取、基于批判的迭代精炼以及基于检索增强生成的知识复用，以提高语义一致性和配置的可靠性。我们评估了多个开源大语言模型……

    arXiv:2609.31397v1 Announce Type: cross  Abstract: Automated and highly usable Quality-of-Service (QoS) enforcement requires translating high-level service intents into deployable traffic-management policies. Although intent-based networking (IBN) has simplified policy specification, bridging the gap between business-level intents and executable network configurations remains complex, error-prone, and difficult to automate. This paper presents Intent2Tc, a closed-loop language-model-driven framework that translates business-level traffic-shaping intents into declarative sub-intents and subsequently into validated, executable Linux traffic control (tc) configurations. The framework integrates an Active Queue Management (AQM)-based digital twin (DT) semantic model, automated metadata extraction, critique-driven refinement, and Retrieval-Augmented Generation (RAG)-based knowledge reuse to improve semantic consistency and configuration reliability. We evaluate multiple open-source large la
    
[^15]: 先标注后总结：学习压缩证据以实现长上下文理解

    Highlight-Then-Summarize: Learning to Compress Evidence for Long-Context Understanding

    [https://arxiv.org/abs/2609.31382](https://arxiv.org/abs/2609.31382)

    提出“先标注后总结”（H2S）的先压缩后推理范式，通过带过程级奖励的强化学习训练模型先识别并压缩与问题相关的证据、生成条件化摘要后再作答，显著提升大模型的长上下文理解能力。

    

    长上下文理解要求大语言模型（LLM）能够对长篇文档、对话和代码进行推理，然而与任务相关的证据往往稀疏且分散在大量无关和冗余内容之中。我们提出了先标注后总结方法，这是一种“先压缩后推理”的范式：首先识别有出处依据的、与问题相关的证据，然后将其整合为一个紧凑的、以问题为条件的摘要，最后再生成最终答案。为了训练这种行为，我们构建了 H2S 数据集，包含来自 11 个基准系列的 6,647 个样本，平均上下文长度为 43.9K tokens；并提出了 H2S-RL 方法，除了最终答案的正确性之外，还为证据选择和摘要构建过程提供过程级奖励。我们在 H2S-Bench（一个包含七项任务的长上下文评测套件）上进行评估。在共享的 128K 输入和 4K 输出预算下，H2S-14B 取得了平均 32.60 的分数，优于 Qwen3.8-2（原文在此处截断）。

    arXiv:2609.31382v1 Announce Type: cross  Abstract: Long-context understanding requires large language models (LLMs) to reason over lengthy documents, conversations, and code, yet task-relevant evidence is often sparse and scattered amid substantial irrelevant and redundant content. We propose Highlight-Then-Summarize (H2S), a compress-then-reason paradigm that first identifies source-grounded, question-relevant evidence and then integrates it into a compact, question-conditioned summary before producing the final answer. To train this behavior, we construct H2S-Dataset, comprising 6,647 examples from 11 benchmark families with an average context length of 43.9K tokens, and introduce H2S-RL, which provides process-level rewards for evidence selection and summary construction in addition to final-answer correctness. We evaluate on H2S-Bench, a seven-task long-context suite. Under a shared 128K input and 4K output budget, H2S-14B achieves an average score of 32.60, outperforming Qwen3.8-2
    
[^16]: 过时文档投毒：当过时的检索结果覆盖模型本来正确的回答时

    Stale-Document Poisoning: When Outdated Retrieval Overrides Correct Model Answers

    [https://arxiv.org/abs/2609.31342](https://arxiv.org/abs/2609.31342)

    该论文首次揭示并定义了一种新的RAG失效模式“过时文档投毒”——过时的外部检索证据会覆盖大模型本已正确的回答，导致最多75%的正确答案被翻转，并构建了涵盖317个经核实的知识反转案例的基准来系统量化这一时间对齐失效风险。

    

    检索增强生成（RAG）常被用来通过提供外部证据来应对知识过时的问题，但只有当这些证据仍然有效时，检索才能起到帮助作用。我们发现了一种时间对齐失效现象——过时文档投毒（stale-document poisoning），即过时的证据会让模型给出错误答案，尽管模型在不使用检索的情况下本可以正确回答。我们构建了一个包含317个经核实的知识反转案例的基准数据集，涵盖医学、法律、软件和平台政策领域，并基于标注日期的官方权威来源。在12个模型上的实验表明，近期的医学知识反转比早已确立的知识反转更难应对。更重要的是，即使在没有指示模型信任文档的情况下，过时的检索结果也会翻转30%的Llama和37%的Qwen的回答；而明确的“遵循文档”指令会将这一比例提高到66%和75%。在四个开源模型和四个领域中，投毒比例介于17%至91%之间，而匹配的最新证据则在97%至100%的试验中被采纳。为了分离时间适用性因素，我们在保持历史……（原文在此处截断）

    arXiv:2609.31342v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) is often used to address outdated knowledge by providing external evidence. But retrieval helps only when that evidence is still valid. We identify a temporal alignment failure, stale-document poisoning, in which outdated evidence makes a model wrong despite answering correctly without retrieval. We construct a benchmark of 317 verified knowledge reversals across medicine, law, software, and platform policy, grounded in dated official sources. Across 12 models, recent medical reversals are harder than long-established ones. More importantly, outdated retrieval flips 30% of Llama and 37% of Qwen answers even without instructions to trust the document; explicit follow instructions raise these rates to 66% and 75%. Across four open models and four domains, poisoning ranges from 17-91%, while matched up-to-date evidence is followed in 97-100% of trials. To isolate temporal applicability, we keep the histo
    
[^17]: 正确的信息抽取流水线取决于文档本身：小型本地模型的准确率-能耗权衡

    The Right Information Extraction Pipeline Depends on the Document: Accuracy-Energy Trade-offs for Small, Local Models

    [https://arxiv.org/abs/2609.31341](https://arxiv.org/abs/2609.31341)

    该研究在隐私约束下系统评估了小型本地模型信息抽取流水线的准确率-能耗权衡，发现最优选择（图像 vs 文本）取决于文档类型，批处理可无损降低38-85%能耗，而神经OCR能耗高达传统OCR的17倍。

    

    arXiv:2609.31341v1 公告类型：新论文 摘要：信息抽取流水线应处理页面图像还是解析后的文本，取决于具体的文档，且答案会随着版面复杂度的变化而翻转。我们在一个排除（封闭式）云服务的约束下研究这一权衡：隐私敏感文档由小型（参数量≤80亿）纯文本模型和视觉-语言模型在本地部署处理，并在涵盖输入表示、模型家族和推理配置的设计空间中同时评估准确率与能耗。通过对接近纯文本的 Kleister-NDA 合同和版面丰富的 VRDU 表单进行基准测试，我们发现批处理是主要的节能杠杆，可在不损失准确率的情况下将每页能耗降低38-85%；而 FP8 量化在逐个处理请求时可节省27-32%的能耗，但在应用批处理后每页节省不足1毫瓦时（9-19%）。预处理主导了剩余的能耗：神经 OCR 每页的能耗是传统 OCR 的17倍，并且几……

    arXiv:2609.31341v1 Announce Type: new  Abstract: Whether an information extraction pipeline should process page images or parsed text depends on the document, and the answer flips across the layout spectrum. We study this trade-off under a constraint that rules out (closed) cloud services: privacy-sensitive documents processed on-premise by small ($\le 8\mathrm{B}$ parameter) text-only and vision--language models, evaluated on both accuracy and energy over a design space spanning input representation, model family, and inference configuration. Benchmarking on the near-plain-text Kleister-NDA contracts and the layout-rich VRDU forms, we find that batching is the dominant energy lever, cutting energy per page by 38-85% at no cost in accuracy, while FP8 quantization saves 27-32% when requests are served one at a time but less than 1mWh per page (9-19%) once batching is applied. Preprocessing dominates what remains: neural OCR costs $17\times$ more energy per page than classical OCR and ne
    
[^18]: 为什么阿尔茨海默病语音筛查难以泛化：通过跨语料库证据锚定弥合部署差距

    Why Alzheimer's Speech Screening Fails to Generalize: Bridging the Deployment Gap via Cross-Corpus Evidence Anchoring

    [https://arxiv.org/abs/2609.31293](https://arxiv.org/abs/2609.31293)

    该论文通过跨四个语料库的留一评估揭示了阿尔茨海默病语音筛查模型泛化失败的根源——特征方向冲突与录音协议敏感性，并提出融合XLM-R文本基线分数的跨语料库证据锚定方法来弥合部署差距。

    

    基于语音的筛查是一种有前景的、非侵入性的检测阿尔茨海默病及相关认知风险的方法。然而，在单一领域上训练的模型往往难以泛化到未见过的语言、任务或录音协议。本文使用跨四个不同数据集的留一语料库评估来研究这一部署差距。在70个可解释的语音和语言特征中，有59个在不同语料库的健康对照组与认知风险组之间表现出方向冲突，其中停顿、静音和语速显示出较高的协议敏感性。此外，虽然XLM-R文本基线取得了较强的平均性能，但其ROC曲线下面积（AUC）在最弱的留出领域上降至0.520。在相同协议下，标准的GroupDRO基线达到了0.766的平均说话人AUC和0.504的最差领域AUC。为解决这一问题，我们提出了一种融合方法，将XLM-R文本基线分数与（摘要原文在此处截断）

    arXiv:2609.31293v1 Announce Type: cross  Abstract: Speech-based screening is a promising, non-invasive approach for detecting Alzheimer's disease and related cognitive risks. However, models trained on a single domain often generalize poorly to unseen languages, tasks, or recording protocols. This paper investigates this deployment gap using a leave-one-corpus-out evaluation across four distinct datasets. Among 70 interpretable speech and language features, 59 exhibit direction conflicts between healthy control and cognitive risk groups across corpora, with pause, silence, and speech rate showing high protocol sensitivity. Furthermore, while the XLM-R text baseline achieves strong average performance, its Area Under the ROC Curve (AUC) drops to 0.520 on the weakest held-out domain. A standard GroupDRO baseline reaches a 0.766 mean speaker AUC and a 0.504 worst-domain AUC under the same protocol. To address this, we propose a fusion method that integrates XLM-R text baseline scores with
    
[^19]: 在X平台上识别科学家

    Identifying Scientists on X

    [https://arxiv.org/abs/2609.31264](https://arxiv.org/abs/2609.31264)

    该论文提出了一种基于用户简介和推文自动识别X/Twitter上科学家与非科学家的分类方法，在集成设置中使用对比微调的DeBERTa模型达到0.96的F1分数，并发布了两个带标注的用户数据集。

    

    随着网络上科学相关讨论的重要性日益增长，以及传统知识秩序的逐渐瓦解，自动识别不同用户群体（如科学家）变得至关重要。本工作提出了一种基于用户简介和推文在X/Twitter上识别科学家与非科学家的方法。我们证明了在两个不同的数据集上，该模型能够将账户分类为科学家和非科学家：使用结合语言特征的随机森林方法可达到高达0.88的F1分数，而在集成设置中使用对比微调的DeBERTa模型可达到高达0.96的F1分数。此外，我们还提供了两个数据集，其中包含被标注为科学家或非科学家的X用户及其相应的推文和用户简介。

    arXiv:2609.31264v1 Announce Type: new  Abstract: With the growing importance of science-related discourse on the Web and the erosion of the classical knowledge order, it is important to identify different user groups, such as scientists, automatically. This work proposes an approach for identifying scientists and non- scientists on X/Twitter based on their user biographies and tweets. We show that we are able to classify accounts as scientists and non- scientists on two different datasets, reaching an F1 score of up to 0.88 using Random Forests with linguistic features and up to 0.96 using a contrastively fine-tuned DeBERTa model in an ensemble setup. Furthermore, we provide two datasets with X users labeled as scientists or non scientists and their respective tweets and user biographies.
    
[^20]: MoSAR：用于学习自适应且可近似注意力几何的语义注意力机制混合方法

    MoSAR: Mixture of Semantic Attention Regimes for Learning Adaptive and Approximable Attention Geometries

    [https://arxiv.org/abs/2609.31261](https://arxiv.org/abs/2609.31261)

    MoSAR将注意力近似视为几何学习问题，通过输入条件化路由器在短程、中程和全局注意力机制间学习自适应混合，以连续的距离相关注意力场取代固定稀疏模式，突破长上下文建模中稠密自注意力的二次复杂度瓶颈。

    

    稠密自注意力的二次方复杂度仍然是长上下文语言建模的核心瓶颈。许多高效的替代方法通过预先决定注意力应在哪里稀疏或局部化来应对这一开销。我们认为，注意力近似应当转而被视为一个几何问题，相关的交互几何应从数据中学习：自然语言的依赖关系依赖于输入且难以预先规定，因此模型应当学习位置相关性可以在哪里衰减，以及更广泛的交互必须在何处得以保留。我们提出了语义注意力机制混合模型MoSAR，它在查询-键交互上学习这种自适应的、受控衰减的几何结构。应用于位置编码之后的输入条件化查询与键路由器，在短程、中程和全局机制之间选择混合，从而诱导出连续的距离相关注意力场，而非固定的稀疏模式。

    arXiv:2609.31261v1 Announce Type: cross  Abstract: The quadratic complexity of dense self-attention remains a central bottleneck for long-context language modeling. Many efficient alternatives address this cost by deciding in advance where attention should be sparse or local. We argue that attention approximation should instead be approached as a geometric problem, with the relevant interaction geometry learned from data: natural-language dependencies are input-dependent and difficult to prescribe in advance, so the model should learn where positional relevance can decay and where broader interactions must be preserved. We introduce Mixture of Semantic Attention Regimes (MoSAR), which learns such an adaptive, controlled-decay geometry over query--key interactions. Input-conditioned query and key routers, applied after positional encoding, select mixtures over short, medium, and global regimes, inducing a continuous distance-dependent attention field rather than a fixed sparsity pattern
    
[^21]: PIA：一个将健康对话转化为记录、并将记录转化为理解的个人智能代理

    PIA: A Personal Intelligence Agent Turning Health Conversations into Records and Records into Understanding

    [https://arxiv.org/abs/2609.31255](https://arxiv.org/abs/2609.31255)

    PIA 是一个部署在消费级健康智能体旁的个人智能代理，它通过抽取、记忆、检索、理解四个领域无关的控制机制配合可插拔的健康模块，将健康对话转化为结构化临床记录并进一步综合为对用户的理解，从而克服了通用摘要式记忆无法处理剂量、时间指代和长期健康趋势的局限。

    

    通用智能体记忆通过对对话进行摘要来工作：它提取显著的片段，对其进行向量化嵌入，并将 top-k 检索结果注入提示词中。而健康智能体无法仅依靠摘要来运行：一个药物剂量会变成一句普通的话，“自上周以来”由模型自行斟酌解释，而三个月的血糖趋势也无法通过文本相似度来回答。我们提出了 PIA，一个与消费级健康智能体协同部署的个人智能代理。PIA 接收智能体的自然语言请求，自主决定是否以及如何进行写入或读取，并将对话转化为结构化的临床记录，再将记录转化为对用户的综合理解。其记忆框架由四个控制组件构成——抽取、记忆、检索和理解——每个组件都是领域无关的机制，并配有可插拔的健康模块：模式、医学别名词典、知识图谱和时间规则。我们展示了同一个查询如何随着所注入记忆的不同而得到不同的回答……

    arXiv:2609.31255v1 Announce Type: new  Abstract: General-purpose agent memory summarizes conversations: it extracts salient snippets, embeds them, and retrieves the top-k into the prompt. A health agent cannot run on summaries: a dose becomes a sentence, "since last week" is resolved at the model's discretion, and a three-month glucose trend cannot be answered by text similarity. We present PIA, a personal intelligence agent deployed alongside a consumer health agent. PIA receives the agent's natural-language requests, decides for itself whether and how to write or read, and turns conversations into typed clinical records and records into a synthesized understanding of the user. Its memory harness consists of four controls -- extraction, memory, retrieval, and understanding -- each a domain-agnostic mechanism with a pluggable health module: schema, medical alias dictionary, knowledge graph, and temporal rules. We show how the same query receives a different answer as the memory injecte
    
[^22]: RupeeBias：审计大语言模型在印度经济指导中的人口群体偏见

    RupeeBias: Auditing Demographic Bias in Indian Economic Guidance from Large Language Models

    [https://arxiv.org/abs/2609.31245](https://arxiv.org/abs/2609.31245)

    提出RupeeBias基准，通过39,150个提示系统审计大语言模型在印度经济指导场景（如薪资估计）中针对种姓、城乡等印度特有人口群体类别产生的偏见，填补了现有西方中心偏见基准的空白。

    

    人们越来越多地转向大语言模型（LLM）寻求各类经济任务上的指导，从比较贷款方案、规划储蓄，到决定要求多少加薪或为自己的服务定价。众所周知，LLM会再现社会偏见，而带有偏见的经济指导可能会影响用户对自身价值的认知、他们提出的要求以及他们最终接受的条件。这一风险在印度尤为突出，因为在印度，经济结果受到种姓、城乡位置等人口群体类别的深刻影响。然而，现有的LLM偏见基准大多围绕西方人口群体类别设计，因此遗漏了印度背景下经济差距的关键维度。我们提出了RupeeBias，一个用于审计LLM生成的经济指导在印度经济场景中人口群体偏见的基准。RupeeBias由39,150个提示组成，涵盖四个用例：薪资估计、薪资增长估计等（摘要在此处截断）。

    arXiv:2609.31245v1 Announce Type: new  Abstract: Individuals turn to large language models (LLMs) for guidance across a wide range of economic tasks, from comparing loan options and planning savings to deciding what raise to ask for or how much to charge for their services. LLMs are known to reproduce social biases, and biased economic guidance may influence what users believe they are worth, what they ask for, and what they ultimately accept. This risk is especially salient in India, where economic outcomes are shaped by demographic categories such as caste and urban-rural location. Existing LLM bias benchmarks, however, are largely designed around Western demographic categories and therefore miss key axes of economic disparity in the Indian context. We introduce RupeeBias, a benchmark for auditing demographic bias in LLM-generated economic guidance across Indian economic settings. RupeeBias consists of 39,150 prompts spanning four use cases: salary estimation, salary increment estima
    
[^23]: 模型将自己的重复词元送往何处

    Where a Model Sends Its Own Repeated Token

    [https://arxiv.org/abs/2609.31181](https://arxiv.org/abs/2609.31181)

    该研究将模型在“自身重复词元”输入下的输出视为覆盖整个词表的映射，首次记录了非不动点词元的去向，并通过按源词元配对消除基数混淆，为黑盒模型识别提供了新的判别信号。

    

    黑盒模型识别的工作方式是对模型对自然语言提示的响应进行评分。其中一条研究路线是向模型输入一种退化输入——即模型自身的词元不断重复——以寻找失效模式而非模型身份。我们采用这一输入，并探究当模型不滞留于原处时会走向何方。对于每个词元 t，只需一次前向传播即可读取 argmax p(. | t, t)；其结果是一个定义在整个词表上的映射，可划分为两个部分。第一部分——哪些词元是不动点——已被部分预见，我们将其作为一个失效的估计对象加以报告：其上的自然距离有83%为基数性，能在3471（个样本）中以两比特区分语料库操纵，而精度下限为零，且以0.5833的分数对模型家族进行归因。第二部分——该映射将非不动点的词元发送到哪里——此前从未被记录；唯一涉及这些词元的论文将其记录为零。按源词元配对在构造上消除了基数混淆（相关系数 r 从 0.9128 降至 -0.0932），并……（原文摘要在此处被截断）

    arXiv:2609.31181v1 Announce Type: new  Abstract: Black-box model identification works by scoring a model's response to natural-language prompts. One line of work feeds models a degenerate input -- their own token, repeated -- to find a failure mode rather than an identity. We take that input and ask where the model goes when it does not. For each token t, read argmax p(. | t, t) in one forward pass; the result is a map on the whole vocabulary, with two halves. The first -- which tokens are fixed points -- is partially anticipated, and we report it as a failed estimand: the natural distance on it is 83% cardinality, separates a corpus manipulation by two bits in 3471 against a precision floor of zero, and attributes families at 0.5833. The second half, where the map sends tokens that are not fixed points, is unrecorded; the one paper holding those tokens logged them as a zero. Pairing on the source token removes the cardinality confound by construction (r from 0.9128 to -0.0932) and att
    
[^24]: 基于度量损失加权提升大语言模型在多模态机器翻译中的视觉敏感性

    Improving Visual Sensitivity of LLMs on Multimodal Machine Translation with Metric-based Loss Weighting

    [https://arxiv.org/abs/2609.31169](https://arxiv.org/abs/2609.31169)

    提出基于度量的损失加权训练方法，利用PCXMI度量识别受益于图像的词元并提高其损失权重，从而增强多模态大语言模型在多模态机器翻译任务中对视觉信息的敏感性和利用能力。

    

    多模态机器翻译旨在利用来自非文本模态的额外信号，通过消解歧义来改进翻译。尽管模型通过多模态融合能够接收与源文本相关的图像，但它们可能会忽略这些信息。因此，提高模型的视觉敏感性仍然是一个活跃的研究方向。在这项工作中，我们提出了一种训练方法——基于度量的损失加权，通过提高那些受益于伴随图像的词元的损失权重，来增强翻译的视觉定位能力。我们使用点向交叉互信息度量来识别这些词元，该度量通过比较有视觉上下文和无视觉上下文情况下模型的输出概率来实现。我们进一步提出了一种基于一致性的PCXMI度量，并通过实验证明这两种度量结合使用能够取得最佳效果。我们通过微调三个预训练的多模态大语言模型来评估所提出的方法。

    arXiv:2609.31169v1 Announce Type: cross  Abstract: Multimodal Machine Translation aims to incorporate additional signal from non-textual modalities to improve translations by resolving ambiguities. While models, through multimodal fusion, are able to accept images related to the source text, they can ignore this information. Therefore, increasing their visual sensitivity remains an active research area. In this work, we introduce a training method, Metric-based Loss Weighting, that improves visual grounding of translations by increasing the loss function for tokens that benefit from the accompanying image. We identify these tokens using the Point-wise Cross-mutual Information (PCXMI) metric, which compares the model's output probabilities with and without visual context. We introduce a Congruency-based PCXMI metric and experimentally show that both metrics working in combination yield the best results. We evaluate our method by fine-tuning three pretrained Multimodal Large Language Mod
    
[^25]: JevAdvBench：面向校准决策强化学习模型的基准与黑盒攻击

    JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models

    [https://arxiv.org/abs/2609.31142](https://arxiv.org/abs/2609.31142)

    该论文提出了首个针对校准决策强化学习（RLCD）模型的对抗攻击基准JevAdvBench，创新之处在于以模型自身的干净决策（而非外部标签）作为评分参照来度量攻击效果，并配套提出了黑盒攻击方法。

    

    使用校准决策强化学习（RLCD）训练的模型（如Jev），针对输入（即状态）回答一个类型化问题，输出概率、选择或分数，软件在无人阅读的情况下直接对该答案进行处理。这类模型的鲁棒性尚未得到测量：现有的对抗基准评估的是模型生成或执行的内容，而类型化模型不生成任何内容，即使被操纵也会返回格式良好的答案。测量本身也很困难，因为相同的请求可能返回不同的答案，大多数可用标签来自模型自身，且API会在用户不可见的情况下对每个请求进行预处理。我们的核心思想是将每个受攻击的决策与模型自身的干净决策进行对比评分，而非与标签对比，并结合相同请求重复运行所引起的变化来解读评分结果。基于此，我们提出了JevAdvBench，据我们所知这是首个针对RLCD模型的对抗基准，包含覆盖6个……的812个类型化问题（摘要在此处被截断）。

    arXiv:2609.31142v1 Announce Type: cross  Abstract: Models trained with reinforcement learning for calibrated decisions (RLCD), such as Jev, answer a typed question about an input, the state, with a probability, a choice, or a score, and software acts on the answer without a person reading it. Their robustness has not been measured: adversarial benchmarks score what a model generates or executes, whereas a typed model generates nothing and returns a well-formed answer even when manipulated. Measurement is also hard, because identical requests can return different answers, most available labels come from the model itself, and the API preprocesses each request out of view. Our key idea is to score each attacked decision against the model's own clean decision rather than against labels, and to read it against the change caused by an identical re-run. Building on this, we introduce JevAdvBench, to our knowledge the first adversarial benchmark for RLCD models, with 812 typed questions over 6
    
[^26]: 我们需要回答那个问题吗？自然对话中潜在问题的显著性与可回答性

    Do we need to answer that question? Salience and Answerability of Potential Questions in Naturalistic Dialogue

    [https://arxiv.org/abs/2609.31130](https://arxiv.org/abs/2609.31130)

    通过对英国国家语料库中自动生成的7,124个潜在问题进行显著性与可回答性标注分析，研究发现自然对话中问题的显著性虽与被回应的可能性呈正相关，但该关联明显弱于独白文本，表明对话结构的可预测性较低。

    

    我们通过研究生成的潜在问题的显著性是否能预测其后续被解决的情况，对基于讨论问题的自然对话建模进行了实证研究。在Wu等人（2024）工作的基础上，我们构建了一个包含7,124个问题的数据集，这些问题是从英国国家语料库中的话语及其前文语境中自动生成的，并针对显著性和可回答性进行了标注。我们发现对话中显著性与可回答性之间存在稳健但较弱的正相关，表明越显著的问题越有可能被回应。然而，这一效应明显弱于独白文本，说明会话结构的可预测性较低。我们进一步观察到，与组织性较差的对话相比，结构化的互动在标注者之间表现出更强的一致性。

    arXiv:2609.31130v1 Announce Type: new  Abstract: We empirically investigate Question Under Discussion based modelling in naturalistic dialogue by studying whether the salience of generated potential questions predicts their subsequent resolution. Building on Wu et al. (2024), we construct a dataset of 7,124 questions automatically generated from utterances and preceding context from the British National Corpus, and annotated for salience and answerability. We find a robust but low positive correlation between salience and answerability in dialogue, indicating that more salient questions are more likely to be addressed. However, this effect is markedly weaker than in monologic text, suggesting that conversational structure is less predictable. We further observe that structured interactions exhibit stronger alignment between annotators than less organised dialogues.
    
[^27]: LocUS：基于头选择与子空间投影的定向激活引导方法

    LocUS: Head Selection and Subspace Projection for Targeted Activation Steering

    [https://arxiv.org/abs/2609.31122](https://arxiv.org/abs/2609.31122)

    提出 LocUS 方法，通过在解嵌矩阵中识别属性特定的线性子空间、将引导变换限制在该子空间并局部化到稀疏的注意力头子集，实现了更精准、对无关能力副作用更小的定向激活引导。

    

    激活引导是一种强大的免训练范式，可在推理时对大语言模型进行控制。然而，标准方法从对比数据中为每一层估计一个引导方向，并将其应用于该层的整个表示空间，这可能使干预与对比数据中存在的非目标属性相耦合，从而损害模型的无关能力。为缓解这一问题，我们提出了 LocUS（局部化解嵌引导，Localized Unembedding Steering），该方法将激活引导锚定到模型自身的输出词汇子空间中。通过在解嵌矩阵中识别特定属性的线性子空间，LocUS 施加了几何约束，将引导变换限制在特定子空间内，同时将其作用局部化到注意力头的稀疏子集上。研究在三个模型家族上，针对毒性缓解、情感重定向和谄媚抑制任务进行了广泛评估。

    arXiv:2609.31122v1 Announce Type: new  Abstract: Activation steering is a powerful training-free paradigm for controlling large language models at inference time. However, standard approaches estimate a per-layer steering direction from contrastive data and apply it on the layer's entire representation space, which may couple the intervention to off-target properties present in the contrastive data and degrade unrelated capabilities. To mitigate this issue, we introduce LocUS (Localized Unembedding Steering), a method which grounds activation steering to the model's own output vocabulary subspace. By identifying a property-specific linear subspace within the unembedding matrix, LocUS enforces a geometric constraint that restricts the steering transformation to a specific subspace and at the same time localizes its application to a sparse subset of attention heads. Extensive evaluations across three model families on toxicity mitigation, sentiment redirection and sycophancy suppression 
    
[^28]: CG-Probes：从患者查询嵌入中恢复护栏方向

    CG-Probes: Recovering Guardrail Directions from Patient Query Embeddings

    [https://arxiv.org/abs/2609.31062](https://arxiv.org/abs/2609.31062)

    该论文提出CG-Probes方法，通过与肿瘤科医生合作定义医疗紧急度、心理紧急度和话题敏感性三个风险轴，并利用均值差方法从患者查询嵌入中恢复线性风险方向，从而为医疗AI助手构建有效的安全护栏。

    

    面向患者的AI助手有望为患者提供有价值的支持，但传入的查询可能带来医疗风险。为了构建护栏，我们与肿瘤科医生合作定义了三个有序的风险轴：医疗紧急程度、心理紧急程度和话题敏感性。我们提出临床护栏探针（CG-Probes）来从查询嵌入中测量风险。我们通过均值差方法在冻结嵌入器的归一化嵌入空间中对每个轴进行探测，将每个轴视为潜在的线性方向。为了训练探针，我们使用BERTopic对79,658条捷克语肿瘤学搜索查询进行聚类，并利用这些聚类通过少样本提示生成具有对比风险水平的查询对。我们在200条查询（90条真实，110条合成）上评估了该方法，每条查询由两名肿瘤科医生进行评分，并与两个开源权重LLM和一个前沿LLM进行对比。我们发现基于紧急程度的轴可以作为线性方向被恢复，且这些探针具有竞争力。

    arXiv:2609.31062v1 Announce Type: new  Abstract: Patient-facing AI assistants promise valuable support to patients, but incoming queries can pose medical risks. To create guardrails, we work with oncologists to define three ordinal risk axes: Medical Urgency, Psychological Urgency, and Topic Sensitivity. We propose Clinical Guardrail Probes (CG-Probes) to measure the risks from query embeddings. We probe for each axis in the normalized embedding space of frozen embedders via the difference-in-means method, treating each axis as a potential linear direction. To train the probes, we cluster 79,658 Czech oncology search queries with BERTopic and use these clusters to generate pairs of queries with contrastive risk levels via few-shot prompting. We evaluate the approach on 200 queries (90 real, 110 synthetic), each graded by two oncologists, against two open-weight LLMs and a frontier LLM. We find that urgency-based axes are recoverable as linear directions, and the probes are competitive 
    
[^29]: 基于大语言模型与知识图谱引导推理的学生意义建构建模

    Modeling Student Sensemaking with LLMs and Knowledge-Graph-Guided Inference

    [https://arxiv.org/abs/2609.31046](https://arxiv.org/abs/2609.31046)

    本研究探索利用指令微调的大语言模型结合知识图谱引导的知识状态诊断来分析协作式科学学习中的学生意义建构，发现启用推理的提示和知识状态信息能显著提升模型识别不成功意义建构的能力。

    

    协作式科学学习需要对学生的对话进行细致解读，以刻画学习者如何识别知识缺口、构建解释并朝着问题解决迈进——这是一种理论驱动但费力且难以规模化的分析。我们研究了经过指令微调的大语言模型（LLM）能否在无需任务特定训练的情况下支持对协作式意义建构的多维分析，以及结构化的知识状态信息能否改进模型的推理。我们在23个经过丰富标注、由专家标记的对话片段上评估了两个中等规模的LLM，测试条件涵盖不同的定义脚手架、推理模式和话轮结构等提示设置。结果表明：在没有推理的情况下，模型倾向于高估成功的意义建构；启用推理的提示能改善对不成功案例的识别。知识状态诊断提供了额外的依据，进一步改进了对不成功意义建构的检测，并提高了与专家标注的一致性。

    arXiv:2609.31046v1 Announce Type: new  Abstract: Collaborative science learning requires nuanced interpretation of student dialogue to characterize how learners identify knowledge gaps, build explanations, and work toward resolution - a theory-driven analysis that is labor-intensive and difficult to scale. We investigate whether instruction-tuned large language models (LLMs) can support multidimensional analysis of collaborative sensemaking without task-specific training, and whether structured knowledge-state information improves model inference. We evaluate two mid-size LLMs on 23 richly annotated, expert-labeled episodes across prompting conditions that vary definitional scaffolding, reasoning mode, and turn structure. Without reasoning, models tend to overpredict successful sensemaking; reasoning-enabled prompting improves identification of unsuccessful cases. Knowledge-state diagnostics provide additional grounding, improving detection of unsuccessful sensemaking and increasing ag
    
[^30]: KuaFu：在十亿级规模上将长用户行为压缩为用户理解

    KuaFu: Compressing Long User Behavior into Understanding at Billion Scale

    [https://arxiv.org/abs/2609.31045](https://arxiv.org/abs/2609.31045)

    该论文提出KuaFu系统，在十亿用户规模上解决将长用户行为压缩为用户理解时序列过长与刷新吞吐量的双重工业瓶颈，并应对压缩过程可能引入的四类幻觉问题。

    

    对话式智能体、生成式推荐器和个性化广告都依赖一项核心能力：从原始行为中理解每一位用户。当前主流的工业实践是任务专用的：针对每个任务，从完整历史中提取相关子序列，并在其上训练专用模型。在生产环境中，这遇到了两个瓶颈。第一，即使经过过滤，单一任务的序列仍然极长：内容兴趣摘要需要读取每位用户的数百条内容，一旦序列化为提示文本就达数万个token。第二，用户画像需要例行刷新：每周覆盖十亿用户，总计约10万QPM，在固定的GPU预算下这设定了一个硬性的吞吐量下限。因此压缩是必不可少的，但截断或粗糙的压缩会悄然扭曲用户画像，引入四种幻觉类型（捏造、遗漏、日期错误归属、逻辑断裂），而由于无法评估压缩（原文在此处截断）……

    arXiv:2609.31045v1 Announce Type: cross  Abstract: Conversational agents, generative recommenders, and personalized advertising all rest on one capability: understanding each user from raw behavior. Prevailing industrial practice is task-specific: for each task, a relevant subsequence is extracted from the full history and a dedicated model trained on it. In production it hits two bottlenecks. First, even after filtering, a single-task sequence stays extremely long: content-interest summarization reads several hundred items per user, tens of thousands of tokens once serialized as prompt text. Second, profiles are refreshed routinely: a billion users weekly, roughly 100K QPM in aggregate, which under a fixed GPU budget sets a hard throughput floor. Compression is therefore mandatory, yet truncation or coarse compression can silently distort the profile, introducing four hallucination types (fabrication, omission, date misattribution, broken logic) that, with no way to evaluate the compr
    
[^31]: 相同文本，不同数字：基于大语言模型的度量指标的分歧

    Same Text, Different Numbers: The Divergence of LLM-Based Measures

    [https://arxiv.org/abs/2609.31013](https://arxiv.org/abs/2609.31013)

    该论文让七个不同的大语言模型对相同的标普500公司财报电话会议文本进行十三种构念评分，发现基于LLM的文本度量存在显著的模型间分歧（平均秩相关仅0.52，共同差异仅占34%），且模型选择会实质性改变下游实证研究中的系数大小、符号和显著性。

    

    研究者越来越多地使用生成式大语言模型（LLM）将企业文本转化为实证变量。我们利用十三种度量指标（包括情感、管理层清晰度、不确定性、回答具体性以及气候与政治风险），考察了基于LLM的文本度量在多大程度上对模型选择保持不变。来自不同提供商的七个LLM对标准普尔500公司财报电话会议记录在这些构念上进行评分。结果显示，跨模型的秩相关系数平均仅为0.52，且各提供商之间共同的文本层面差异仅占总分变异的34%。跨模型分歧并不能预测后续分析师或市场的分歧，这表明存在实质性的模型特异性成分，而非底层信息披露本身存在共同的模糊性。模型选择会显著影响下游推断，系数大小、符号和统计显著性在不同模型之间存在较大差异。

    arXiv:2609.31013v1 Announce Type: new  Abstract: Researchers increasingly use generative large language models (LLMs) to convert corporate text into empirical variables. We examine the extent to which LLM-based textual measures are invariant to model choice using thirteen measures, including sentiment, management clarity, uncertainty, answer specificity, and climate and political risk. Seven LLMs from different providers score earnings call transcripts of S&P 500 companies on these constructs. Cross-model rank correlations average only 0.52, and transcript-level differences common across providers account for only 34% of total score variation. Cross-model disagreement does not predict subsequent analyst or market disagreement, consistent with a substantial model-specific component rather than common ambiguity in the underlying disclosure. Model choice significantly affects downstream inference, with coefficient magnitudes, signs, and statistical significance varying substantially acros
    
[^32]: G²PTQ：利用广义梯度补偿改进大语言模型训练后量化

    G$^2$PTQ: Improving LLM Post-Training Quantization with Generalized Gradient Compensation

    [https://arxiv.org/abs/2609.31009](https://arxiv.org/abs/2609.31009)

    G²PTQ提出了一种统一的大语言模型训练后量化框架，通过在每个Transformer块量化前动态刷新并融合一阶梯度与二阶Hessian信息的广义梯度补偿机制，克服了现有GPTQ类方法缺乏全局监督或指导信息随量化进程失效的缺陷。

    

    训练后量化（PTQ）是一种无需重新训练即可减少大语言模型（LLM）内存占用和计算开销的实用方法。基于GPTQ的方法已成为事实上的标准，但它们存在两个互补的局限性：采用局部、逐层目标的方法缺乏全局监督；而采用全局目标的方法在开始时便固定其Hessian估计并忽略一阶梯度，因此随着量化过程的进行，其指导作用会逐渐失效。本文提出了G²PTQ，这是一个融合广义梯度补偿的统一训练后量化框架，在全局监督的分块优化目标下同时整合了一阶和二阶信息。通过在量化每个Transformer块之前刷新梯度和Hessian估计，G²PTQ避免了先前全局方法中指导信息过时的问题。此外，为了稳定精确的一阶补偿，我们引入了一种信任域方案……（原文摘要在此处截断）

    arXiv:2609.31009v1 Announce Type: cross  Abstract: Post-training quantization (PTQ) is a practical approach to reducing the memory and computational footprint of large language models (LLMs) without retraining. GPTQ-based methods have become the de facto standard, yet they suffer from two complementary limitations. Methods with local, layer-wise objectives lack global supervision; while methods with global objectives fix their Hessian estimates at the start and ignore first-order gradients, so their guidance grows stale as quantization proceeds. This paper presents G$^2$PTQ, a unified PTQ framework with Generalized Gradient Compensation that integrates both first- and second-order information under a globally supervised, block-wise optimization objective. By refreshing gradient and Hessian estimates before quantizing each Transformer block, G$^2$PTQ avoids the staleness of prior global methods. Furthermore, to stabilize the exact first-order compensation, we introduce a trust-region sc
    
[^33]: ZooWork-ShopRanker：一个开放的、偏好对齐的电商重排序器

    ZooWork-ShopRanker: An Open, Preference-Aligned E-Commerce Reranker

    [https://arxiv.org/abs/2609.31002](https://arxiv.org/abs/2609.31002)

    提出了一系列（0.6B/4B/8B）与多家族推理大语言模型裁判标注的购物偏好对齐的开放电商重排序器，并以8B旗舰模型为教师蒸馏出更高效的4B和0.6B模型。

    

    为通用网页检索训练的开放重排序器在迁移到电商场景时效果并不理想，因为电商中的排序决策不仅取决于主题相关性，还取决于用户偏好、产品约束以及产品之间的相对适配度。这些偏好信号难以进行大规模监督：真实搜索流量虽然提供了真实的查询和候选商品，却没有干净的成对标签。我们提出了ZooWork-ShopRanker，这是一系列与裁判标注的购物偏好对齐的电商重排序器（0.6B、4B和8B）。训练样本对由来自不同家族的推理大语言模型（LLM）小组作为偏好oracle进行标注，采用位置去偏差的判断和一致性分级，重排序器则在这些标签上进行训练。对齐后的8B旗舰模型随后作为蒸馏教师，用于训练高效的4B和0.6B模型，这些模型拟合其分数并在经过裁判判断的样本对上进行强化。为衡量研究进展，我们引入了ShopRank-Benc（原文此处截断，应为ShopRank-Benchmark基准）

    arXiv:2609.31002v1 Announce Type: new  Abstract: Open rerankers trained for general web retrieval transfer imperfectly to e-commerce, where ranking decisions depend not only on topical relevance but also on user preferences, product constraints, and comparative product fit. These preference signals are difficult to supervise at scale: real search traffic provides authentic queries and candidates but no clean pairwise labels. We present ZooWork-ShopRanker, a family of e-commerce rerankers (0.6B, 4B, and 8B) aligned to judge-labeled shopping preference. Training pairs are labeled by a panel of reasoning large language models (LLMs) from different families acting as a preference oracle, with position-debiased judgments and agreement tiers, and the rerankers are trained on these labels. The aligned 8B flagship then serves as a distillation teacher for the efficient 4B and 0.6B models, which are fit to its scores and sharpened on judged pairs. To measure progress, we introduce ShopRank-Benc
    
[^34]: 评估中文大语言模型在源自在线搜索查询的事实性问题上的谄媚现象

    Evaluating Sycophancy in Chinese Large Language Models on Factual Questions Derived from Online Search Queries

    [https://arxiv.org/abs/2609.30986](https://arxiv.org/abs/2609.30986)

    该研究通过12165个是/否事实核查问题和超过36万个回答，系统评估了DeepSeek、Qwen和豆包三个中文大语言模型在用户表达错误信念时的事实性谄媚现象，并探究其回答转变的具体模式。

    

    随着大语言模型日益成为信息获取的中介，事实准确且独立的回答至关重要。然而，这些模型可能表现出谄媚性，即即使其回应与用户陈述的信念保持一致，而这些信念可能是错误的，这可能导致错误信息被呈现为经过独立验证的信息，并强化用户对错误说法的信心。先前的工作尚未解决：引入用户信念是否会导致原本正确的回答变得错误或不确定，还是会导致不确定的回答变成与用户信念一致但错误的答案。此外，反谄媚干预措施究竟是保持或恢复事实准确性，还是仅仅使回答转向不确定性，这一点也尚不清楚。我们使用是/否型事实核查问题分析了中文信息检索中的事实性谄媚现象。我们的分析涵盖了三个前沿的基于中文的大语言模型对12,165个事实性问题的364,941个回答。

    arXiv:2609.30986v1 Announce Type: new  Abstract: As large language models increasingly mediate information access, factually accurate and independent answers are critical. However, these models can exhibit sycophancy by aligning their responses with users' stated beliefs even when those beliefs are incorrect, potentially presenting misinformation as independently verified and reinforcing users' confidence in false claims. Prior work leaves unresolved whether introducing user beliefs causes correct responses to become incorrect or uncertain, or causes uncertain responses to become belief-aligned incorrect answers. It also remains unclear whether anti-sycophancy interventions preserve or restore factual accuracy or merely shift responses toward uncertainty. We analyze factual sycophancy in Chinese-language information seeking using yes/no fact-checking questions. Our analysis covers 364,941 responses from three frontier Chinese-based LLMs (DeepSeek, Qwen, and Doubao) to 12,165 factual qu
    
[^35]: THA：基于加权有限状态转换器的高棉语文本规范化与逆文本规范化

    THA: Weighted Finite-State Text Normalization and Inverse Text Normalization for Khmer

    [https://arxiv.org/abs/2609.30984](https://arxiv.org/abs/2609.30984)

    Tha是首个针对高棉语的开源文本规范化与逆文本规范化工具包，基于加权有限状态转换器实现单次最短路径搜索完成整行分词与分类，在高棉语基准测试上取得了近乎完美的准确率。

    

    语音合成需要口语形式的书面文本，而语音识别的输出则需要相反方向的转换。对于高棉语而言，这两个方向都没有得到维护的开源工具，且高棉文字本身使这两项任务更加困难：词与词之间没有空格分隔，数词还会出现在普通词汇内部。我们提出了Tha，一个基于加权有限状态转换器构建的高棉语文本规范化与逆文本规范化工具包。它通过一次最短路径搜索即可完成整行文本的分段与分类，并由第二个转换器拒绝出现在高棉语音节内部的切分边界。在谷歌的高棉语测试集上，Tha在全部274个基数词上与参考答案一致（至多存在一种拼写变体差异）；在2,906条真实TTS语料上，其改写的158个句子中有153个是正确的。Tha以Apache 2.0许可证开源发布。

    arXiv:2609.30984v1 Announce Type: new  Abstract: Text-to-speech needs written text in spoken form, and speech recognition output needs the reverse. For Khmer, neither direction has a maintained open-source tool, and the script makes both harder: words are not separated by spaces, and number words occur inside ordinary words. We present Tha, a Khmer text normalization and inverse text normalization toolkit built from weighted finite-state transducers. It segments and classifies a whole line in one shortest-path search, and a second transducer rejects token boundaries inside a Khmer syllable. On Google's Khmer test suite, Tha agrees with the reference on all 274 cardinals up to one spelling variant, and on 2,906 real TTS prompts, 153 of the 158 sentences it rewrites are correct. Tha is open source under the Apache 2.0 license.
    
[^36]: 均匀离散扩散模型需要时间吗？

    Does Uniform Discrete Diffusion Need Time?

    [https://arxiv.org/abs/2609.30977](https://arxiv.org/abs/2609.30977)

    本研究从理论和实证两方面证明，在语言建模等有限数据场景下，均匀离散扩散模型的时间条件化在很大程度上是不必要的，与时间无关的预测器能达到甚至超越时间条件化模型的性能。

    

    均匀离散扩散模型（UDMs）通常使用显式的时间条件化机制，但我们发现这在实践中往往是不必要的。本文首先证明，在总体最优意义上，UDM的预测器通常依赖于时间：时间控制着模型应该在多大程度上信任观测到的上下文。随后我们证明，在与语言相关的有限数据设置中，这种依赖性可以变得微不足道。当一个被破坏的训练序列与其原始干净序列的距离仍远小于其与其他竞争训练序列的距离时，经验最优预测器在扩散轨迹的大部分范围内对时间几乎不敏感，尽管这一保证在高噪声端点附近会减弱。实证结果表明，训练好的语言UDM在轨迹的大部分范围内表现出有限的时间敏感性，而与时间无关的预测器在各种数据集和训练目标上与时间条件化模型相比仍具竞争力，且往往表现更优。

    arXiv:2609.30977v1 Announce Type: cross  Abstract: Uniform discrete diffusion models (UDMs) commonly use explicit time conditioning, but we find that it can often be unnecessary in practice. In this paper, we first show that the population-optimal UDM predictor generally depends on time: time controls how much the model should trust the observed context. We then show that this dependence can become negligible in finite-data settings relevant to language. When a corrupted training sequence remains much closer to its original clean sequence than to competing training sequences, the empirical-optimal predictor is nearly insensitive to time over most of the diffusion trajectory, where the guarantee weakens toward the high-noise endpoint. Empirically, trained language UDMs exhibit limited time sensitivity over most of the trajectory, while time-agnostic predictors remain competitive with, and often outperform, time-conditioned models across datasets and training objectives. These results ch
    
[^37]: 耦合用法-语义过程：具有时间性和可归因性的词汇语义变化

    Coupled Usage-Sense Processes: Temporal and Attributable Lexical Semantic Change

    [https://arxiv.org/abs/2609.30974](https://arxiv.org/abs/2609.30974)

    本文提出耦合用法-语义过程（CUSP）框架，通过单一的边缘保持时序过程，不仅能量化词汇语义变化的幅度，还能精确确定变化发生的时间、将变化分解为成分移动与重组的机制，并归因到具体的词语用法。

    

    词汇语义变化通常通过独立采样的时期分布之间的标量距离来概括。这种方法只能衡量一个词变化了多少，但无法揭示它何时发生变化、哪些机制和成分移动承载了这种变化，或者哪些用法支持这种归因。我们提出了耦合用法-语义过程，它从单一的保持边缘分布的时序过程中推导出这些答案。层次化耦合通过潜在用法成分关联上下文分布，而马尔可夫组合使相邻以及更长时间跨度上的对应关系相互兼容。位移算子量化变化的幅度与时间，将变异精确地分解为成分中心的移动和成分内部的重组织，并将其归因于被迁移的成分对。词局部模式解析出变化的不同方向及其随时间的活跃程度，而来自归因成分的代表性语段则……

    arXiv:2609.30974v1 Announce Type: new  Abstract: Lexical semantic change is usually summarized by a scalar distance between independently sampled period distributions. This measures how much a word changed, but does not reveal when it changed, which mechanisms and component movements carried the change, or which usages support the attribution. We introduce Coupled Usage--Sense Processes (CUSP), which derives these answers from a single marginal preserving temporal process. A hierarchical coupling relates contextual distributions through latent usage components, while Markov composition makes adjacent and longer span correspondences compatible. Displacement operators quantify change magnitude and timing, split variation exactly between movement of component centers and reorganization within components, and attribute it to transported component pairs. Word-local modes resolve distinct directions of change and their activity over time, while representative passages from attributed compone
    
[^38]: FAVoR：测量与缓解联邦个性化生成中的作者风格同质化

    FAVoR: Measuring and Mitigating Author-Style Homogenization in Federated Personalized Generation

    [https://arxiv.org/abs/2609.30968](https://arxiv.org/abs/2609.30968)

    该论文揭示了联邦个性化生成中标准聚合方法会导致不同作者写作风格同质化的问题，并提出 FAVoR 评估协议来测量和缓解这一现象。

    

    大语言模型越来越多地被用作个性化写作助手，但在多位作者之间适配同一个模型时，可能会将作者特有的写作信号拉向一种共享的语体，从而损害个人的写作风格。联邦参数高效微调（PEFT）为这一多作者适配问题提供了数据本地的解决方案：客户端将作者文本保留在本地，仅共享紧凑的适配器更新。然而，我们证明标准聚合方法可以在保持续写效用的同时，使不同作者的生成结果在风格空间中变得难以区分，我们将这种失败模式定义为“作者风格同质化”。我们通过基于角度风格分类编码器（ASCE）的诊断方法在主要的 BlogText 基准上进行评估，并辅以独立于 ASCE 的外部作者身份验证。利用该评估协议，我们发现常见的联邦 PEFT 基线方法能够在保持语义效用的同时，将作者特有的信号平均化……

    arXiv:2609.30968v1 Announce Type: new  Abstract: Large language models are increasingly used as personalized writing assistants, but adapting a model across many authors can compromise individual writing style by pulling author-specific signals toward a shared register. Federated parameter-efficient fine-tuning (PEFT) offers a data-local setting for this multi-author adaptation problem: clients keep author text local while sharing compact adapter updates. However, we show that standard aggregation can preserve continuation utility while making different authors' generations less distinguishable in style space, a failure mode we define as author-style homogenization. We evaluate author-style retention with Angular Style Classification Encoder (ASCE)-based diagnostics on our main BlogText benchmark and ASCE-independent external authorship verification. Using this protocol, we find that common federated PEFT baselines can preserve semantic utility while averaging out author-specific signa
    
[^39]: 估计并正交化未知的预训练梯度以实现大语言模型的持续微调

    Estimating and Orthogonalizing Unknown Pre-training Gradients for Continual Fine-tuning of Large Language Models

    [https://arxiv.org/abs/2609.30935](https://arxiv.org/abs/2609.30935)

    提出EoupCT框架，通过动态生成最易受遗忘影响的伪数据来估计未知的预训练梯度，并将其与新任务的梯度进行正交化投影，从而在持续微调大语言模型时有效保护其固有的通用知识，避免灾难性遗忘。

    

    持续微调对于大语言模型动态适应现实世界环境至关重要，然而它不可避免地遭受灾难性遗忘的困扰，尤其是先前任务性能的下降以及大模型通用知识的退化。尽管现有方法（如正交梯度投影）能够缓解各种微调任务中的遗忘问题，但由于现成预训练大模型所需的原始数据和梯度严格未知且高度多样化，这些方法从根本上无法保留预训练大模型固有的通用知识。为弥合这一关键差距，我们提出了EoupCT，一个旨在估计并正交化未知预训练梯度以实现大语言模型持续微调的新型框架。具体而言，EoupCT通过动态生成对新任务最易受遗忘影响的伪数据来估计预训练梯度……（原文摘要在此处截断）

    arXiv:2609.30935v1 Announce Type: cross  Abstract: Continual fine-tuning is essential for large language models (LLMs) to dynamically adapt to real-world environments, yet it inevitably suffers from catastrophic forgetting, particularly the performance degradation of previous tasks and LLMs' general-purpose knowledge. Although existing methods, such as orthogonal gradient projection, mitigate the forgetting across various fine-tuning tasks, they fundamentally fail to preserve pre-training LLMs' inherent general-purpose knowledge because the original data and gradients of off-the-shelf pre-training LLMs required by these methods are strictly unknown and highly diverse. To bridge this critical gap, we propose EoupCT, a novel framework designed to Estimate and Orthogonalize Unknown Pre-training gradients for Continual LLM fine-Tuning. Specifically, EoupCT estimates pre-training gradients by dynamically generating pseudo data that is most susceptible to forgetting for new tasks through a l
    
[^40]: 通过文本约束声学重打分实现免训练发音转写

    Training-Free Pronunciation Transcription via Text-Constrained Acoustic Rescoring

    [https://arxiv.org/abs/2609.30924](https://arxiv.org/abs/2609.30924)

    提出一种免训练的语音与文本到发音（ST2P）流水线，通过文本约束候选生成与冻结预训练S2P模型的声学重打分，在日语语料库上将发音转写错误率从0.60%–1.40%大幅降低至0.04%–0.17%，无需昂贵的发音标注数据。

    

    准确且高效的发音转写对于大规模准备语音合成（TTS）训练数据至关重要。现有方法各有局限：字母到发音（G2P）和语音到发音（S2P）方法各自只能捕获部分信息，仅依赖文本或仅依赖语音，而语音与文本到发音（ST2P）方法虽同时利用两者，但需要代价高昂的发音标注数据。为解决这一问题，我们提出了一种免训练的ST2P流水线，在推理阶段整合词汇与声学信息。该流水线首先利用词汇资源和G2P工具生成文本约束的候选项，再通过自左向右的贪心搜索，使用冻结的预训练S2P模型输出的整句负对数似然来选出最佳候选。在三个日语语料库上，我们的方法在拥有参考转写的情况下将字符错误率（CER）从0.60%–1.40%（纯文本基线）降低至0.04%–0.17%，在使用ASR转写的情况下为0.64%–1.58%。

    arXiv:2609.30924v1 Announce Type: new  Abstract: Accurate and efficient pronunciation transcription is essential for preparing text-to-speech training data at scale. Existing approaches have different limitations: grapheme-to-pronunciation (G2P) and speech-to-pronunciation (S2P) methods each capture only partial information, using only text or only speech, while speech-and-text-to-pronunciation (ST2P) methods use both but require costly pronunciation-annotated data. To address this problem, we propose a training-free ST2P pipeline that integrates both lexical and acoustic information at inference time. Lexical resources and G2P tools generate text-constrained candidates, and a left-to-right greedy search selects the best one using whole-sequence negative log-likelihoods from frozen pretrained S2P models. On three Japanese corpora, our method reduces Character Error Rate (CER) from 0.60--1.40\% (text-only baseline) to 0.04--0.17\% with reference transcripts, and 0.64--1.58\% with ASR tr
    
[^41]: 跨后端QIEO：跨OpenMP5、CUDA、HIP及多语言接口的通用运行时可移植性

    Cross-Backend QIEO: Universal Runtime Portability across OpenMP5, CUDA, HIP, and Multi-Language Interfaces

    [https://arxiv.org/abs/2609.30914](https://arxiv.org/abs/2609.30914)

    提出跨后端量子启发进化优化器QIEO，通过单一C++代码库的“单一事实来源”架构，实现了在OpenMP5、CUDA、HIP等硬件后端及多语言接口上的通用运行时可移植性，无需物理量子比特即可在NP难问题上取得10–80倍的加速。

    

    量子启发算法通过将候选解表示为量子比特向量，并利用旋转门算子对其进行演化，从而在经典硬件上模拟叠加、干涉和概率幅演化等量子力学原理。这种方法无需物理量子比特即可提供更高的优化性能，并且已被证明在组合型、高维NP难问题上相比传统求解器可实现数量级的加速（10–80倍）。然而，阻碍其广泛应用的一个关键障碍是缺乏一个统一的执行框架，能够同时提供算法性能和硬件可移植性。我们提出了**跨后端量子启发进化优化器**，它是BQP公司BQPhy求解器的运行时核心，通过一种*单一事实来源*（single-source-of-truth）架构弥合了这一差距。QIEO算法仅需一份C++实现，针对每个硬件目标编译一次，并向外暴露（接口）……

    arXiv:2609.30914v1 Announce Type: cross  Abstract: Quantum-inspired algorithms emulate quantum mechanical principles, such as, superposition, interference, and probabilistic amplitude evolution, on classical hardware by representing candidate solutions as qubit vectors and evolving them through rotation-gate operators. This approach offers higher optimization performance without physical qubits, and has been shown to achieve order-of-magnitude speedups (10--80$\times$) over traditional solvers on combinatorial, high-dimensional NP-hard problems.   A critical barrier to adoption, however, is the lack of a unified execution framework that delivers both algorithmic performance and hardware portability. We present \textbf{Cross-Backend Quantum Inspired Evolutionary Optimizer (QIEO)}, the runtime core of BQP's BQPhy solver, which addresses this gap through a \emph{single-source-of-truth} architecture. One C++ implementation of the QIEO algorithm is compiled once per hardware target and expo
    
[^42]: ToolSearcher：基于强化学习的大规模工具选择优化

    ToolSearcher: Optimizing Tool Selection at Scale via Reinforcement Learning

    [https://arxiv.org/abs/2609.30906](https://arxiv.org/abs/2609.30906)

    本文提出ToolSearcher，一个面向大规模工具选择的新型强化学习框架，通过多轮搜索与细粒度优化，解决了LLM在海量且多样的真实工具库中难以有效搜索、区分和组合工具的挑战。

    

    大型语言模型（LLM）在自然语言处理方面表现出色，但难以与外部环境进行交互。工具学习为将LLM扩展为可执行智能体提供了一条有前景的途径，其中工具选择是成功使用工具的关键前提。现有工作通常假设工具集较小或预先定义，导致大规模工具选择问题未得到充分探索。真实世界的工具库包含庞大且多样的工具集合，使得LLM在上下文长度限制下难以有效地搜索、区分和组合工具。我们将大规模工具选择确定为智能体强化学习的一个新挑战，并指出现有的基于知识的问答强化学习方法在选择工具时无法充分考虑工具间的兼容性，因而不适用于该任务。为应对这一挑战，我们提出了ToolSearcher，一个新颖的强化学习框架，用于实现高效的多轮搜索和细粒度优化。

    arXiv:2609.30906v1 Announce Type: new  Abstract: Large language models (LLMs) excel at natural language processing but struggle to interact with external environments. Tool learning provides a promising way to extend LLMs into actionable agents, where tool selection is a critical prerequisite for successful tool use. Existing work often assumes a small or predefined set of tools, leaving large-scale tool selection underexplored. Real-world repositories contain a vast and diverse array of tools, making it difficult for LLMs to effectively search, distinguish, and compose tools under context-length constraints. We identify large-scale tool selection as a new challenge for agentic reinforcement learning, highlighting that existing RL methods for knowledge-based question answering are inadequate for selecting tools while considering compatibility. To address this challenge, we propose ToolSearcher, a novel RL framework for effective multi-turn search and fine-grained optimization in large-
    
[^43]: 从标注到推理：语言模型中的文化

    From annotation to reasoning: Culture in language models

    [https://arxiv.org/abs/2609.30897](https://arxiv.org/abs/2609.30897)

    该论文提出以文学阐释为研究场景，主张文化能力评估应从事实知识测试转向考察“解释深度”，即模型用证据支撑文化解读并在批评中修正解读的能力，同时评估框架应保留学术分歧的合理性。

    

    当不止一种解释都可能是正确时，我们该如何评估语言模型？现有的文化基准测试往往考察事实性知识、与调查问卷回答的一致性，或对某一预设含义的识别。这些任务无法检验模型是否能够解释某个文化引用在特定文本中如何发挥作用、是否能为某一解读提供证据支撑，或在受到批评后修正该解读。这是一个关于“解释深度”的问题，与文化覆盖面的“广度”互为补充。我们认为，文学阐释为研究这些能力提供了一个有用的场景。我们聚焦于文化引用与复用：即文本如何在历史与语言语境中援引、重复并转化先前的表达。我们的核心主张是：文学学者可以对某一阐释持有不同观点，却仍然认可其论证支撑的质量。我们提出将证据中心的基准测试、保留学术分歧的评估方式与模型开发（原文此处截断）相结合。

    arXiv:2609.30897v1 Announce Type: new  Abstract: How should we evaluate language models when more than one interpretation can be right? Cultural benchmarks often test factual knowledge, agreement with survey responses, or recognition of a predefined meaning. These tasks leave open whether a model can explain how a cultural reference works in a particular text, support a reading with evidence, or revise it after criticism. This is a question of interpretive depth, complementary to the breadth of cultural coverage. We argue that literary interpretation offers a useful setting for studying these capabilities. We focus on cultural referencing and reuse: how texts invoke, repeat, and transform earlier expressions across historical and linguistic contexts. Our central claim is that literary scholars can disagree about an interpretation while recognizing the quality of its support. We propose linking evidence-centered benchmarks, evaluation that preserves scholarly disagreement, and model-dev
    
[^44]: 字幕压缩对基于大语言模型的日语YouTube医疗视频错误信息检测的影响

    Effects of Transcript Compression on LLM-based Medical Misinformation Detection in Japanese YouTube Videos

    [https://arxiv.org/abs/2609.30882](https://arxiv.org/abs/2609.30882)

    研究发现，对日语医疗YouTube视频的字幕进行压缩（摘要、检索或筛选）会降低大语言模型检测医疗错误信息的准确性，完整字幕输入效果最佳。

    

    大语言模型（LLM）正被越来越多地用于评估长篇医疗视频，但其有效性可能取决于所提供的字幕是完整的，还是通过摘要、检索或声明筛选等方式进行了压缩。本研究考察了这类字幕压缩如何影响基于大语言模型的日语医疗YouTube视频真实性分类。我们比较了四种字幕输入设计：完整字幕、大语言模型生成的摘要、基于RAPTOR的检索增强生成（RAG），以及筛选法——即提取候选的医疗和健康相关句子。我们使用74个被标注为“真实”或“虚假”的长篇视频，评估分类性能，并利用J-LIWC、模糊限制语表达以及机构类或技术类术语来分析语言变化。完整字幕的基线方法取得了最佳性能，而所有压缩输入都增加了假阴性，即虚假视频更有可能被错误分类。

    arXiv:2609.30882v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to assess long-form medical videos, but their effectiveness may depend on whether transcripts are provided in full or compressed through summarization, retrieval, or claim screening. This study examines how such transcript compression affects LLM-based veracity classification of Japanese medical YouTube videos. We compare four transcript input designs: full transcripts, LLM-generated summaries, RAPTOR-based retrievalaugmented generation (RAG), and Screening, which extracts candidate medical and health-related sentences. Using 74 long-form videos labeled as Real or Fake, we evaluate classification performance and analyze linguistic changes using J-LIWC, hedge expressions, and institutional or technical terms. The full-transcript Baseline achieved the best performance, whereas all compressed inputs increased false negatives, meaning that Fake videos were more likely to be misclassified as 
    
[^45]: 气候政策因果评估中识别假设的证据支撑式审计

    Evidence-Grounded Auditing of Identification Assumptions in Climate-Policy Causal Evaluations

    [https://arxiv.org/abs/2609.30867](https://arxiv.org/abs/2609.30867)

    提出了ARGUS——一个结构化语言模型流水线，用于审计气候政策双重差分研究中支持识别假设的证据，可检测出73%的植入缺陷（远超关键词方法的18%），并在证据无法检索时主动弃权。

    

    双重差分（DID）研究被广泛用于评估气候政策，但评估支持其识别假设的证据仍然具有挑战性。我们提出了ARGUS，这是一个结构化的语言模型流水线，它依据一个包含十一个维度的“假设-含义-证据”评估标准来审计所报告的证据，并在无法检索到相关证据时选择弃权。我们通过注入缺陷、经济学论文以及一个经过标签调和的小型试点实验来评估ARGUS。在11个缺陷的基准测试中，ARGUS检测出73%的植入缺陷，而基于关键词的流水线仅检测出18%。在26篇经济学论文中，由于缺乏可检索的证据，ARGUS在约40%的论文-维度评估中选择了弃权。在一个由两名标注员调和标签的五篇论文试点中，ARGUS在其完成的33项评估中有25项给出了比标注标签更高的风险等级。一条在标签到达之前就已固定的规则消除了其中大部分样本内偏差；加权……

    arXiv:2609.30867v1 Announce Type: new  Abstract: Difference-in-differences (DID) studies are widely used to evaluate climate policy, but assessing the evidence supporting their identification assumptions remains challenging. We introduce ARGUS, a structured language-model pipeline that audits reported evidence against an eleven-dimension assumption-implication-evidence rubric and abstains when relevant evidence cannot be retrieved. We evaluate ARGUS using injected flaws, economics papers, and a small pilot with reconciled labels. On the 11-flaw benchmark, ARGUS detects 73% of planted flaws, compared with 18% for a keyword-based pipeline. Across 26 economics papers, ARGUS abstains on about 40% of paper-dimension assessments for lack of retrievable evidence. In a five-paper pilot with labels reconciled by two annotators, it assigns a higher risk level than the labels on 25 of the 33 assessments it completes. A rule fixed before the labels arrived removes most of this in-sample; weighted 
    
[^46]: 面向对抗性黑盒在线策略蒸馏的持久负样本方法

    Persistent Negatives for Adversarial Black-Box On-Policy Distillation

    [https://arxiv.org/abs/2609.30864](https://arxiv.org/abs/2609.30864)

    提出持久负样本对抗蒸馏方法，通过活跃池机制用历史教师-学生对比作为稳定负样本，解决了对抗蒸馏中负样本分布随策略更新而漂移的移动目标问题，从而提升黑盒在线策略蒸馏的稳定性。

    

    黑盒在线策略蒸馏（OPD）旨在当教师模型只提供采样响应而不提供token概率时，从学生模型自身的生成结果中对其进行改进。对抗蒸馏提供了一条可行路径：它通过学习一个判别器来区分提示匹配的教师响应和学生响应，并将判别器的评分用作策略奖励。然而，在每一步都从最新的学生模型中采样判别器负样本，会使学到的奖励与每次策略更新后都会发生变化的负样本分布耦合在一起。我们通过持久负样本对抗蒸馏来解决这一移动目标问题，这是一种活跃池方法，用历史积累的、提示匹配的教师-学生对比来替换每个判别器批次中的一部分数据。在判别器计算量匹配的情况下，历史对比用于训练判别器，而GRPO则基于新鲜的学生响应保持在线策略特性。我们的分析确定了贝叶斯最优奖励的形式为教师样本相对于负样本的对数密度比。

    arXiv:2609.30864v1 Announce Type: cross  Abstract: Black-box On-Policy Distillation (OPD) seeks to improve a student from its own generations when the teacher provides sampled responses but not token probabilities. Adversarial distillation offers one route: it learns a discriminator over prompt-matched teacher and student responses and uses its score as the policy reward. However, sampling discriminator negatives from the latest student at each step couples the learned reward to a negative distribution that changes after every policy update. We address this moving-target problem with persistent-negative adversarial distillation, a live-pool method that replaces a fraction of each discriminator batch with historical, prompt-matched teacher--student comparisons. Under matched discriminator compute, historical comparisons train the discriminator, while GRPO remains on-policy with fresh student responses. Our analysis identifies the Bayes-optimal reward as a teacher-to-negative log-density
    
[^47]: 利用扰动强度增强对大语言模型解释自洽性的评估

    Enhancing Assessment of Self-Consistency in LLM Explanations using Perturbation Strength

    [https://arxiv.org/abs/2609.30849](https://arxiv.org/abs/2609.30849)

    本文提出用大语言模型作为评判者来统一度量输入和思维链扰动的强度，从而在受控强度条件下公平评估各类大语言模型解释的自洽性，发现输入扰动的影响通常强于思维链扰动，且自洽性判断只有在同一扰动类型内才公平。

    

    先前的工作使用表层扰动方法研究了大语言模型生成解释的自洽性。然而，这些扰动的强度并未被显式地测量和控制。在这项工作中，我们提出了一种LLM-as-a-judge（大语言模型作为评判者）的方法，以统一的方式测量输入扰动和思维链扰动的扰动强度。随后，我们在强度受控的条件下评估了多个大语言模型所生成解释的自洽性，从而确保不同扰动类型之间的公平比较。实验表明，我们提出的基于大语言模型的扰动强度度量优于其他基于嵌入和概率的方法，并且输入扰动通常比思维链扰动对大语言模型的影响更强。我们的工作表明，关于模型自洽性的判断只有在同一扰动类型内才是公平的。

    arXiv:2609.30849v1 Announce Type: new  Abstract: Prior work has examined the self-consistency of LLM-generated explanations using surface-level perturbation methods. However, the strength of these perturbations is not explicitly measured and controlled. In this work, we propose an LLM-as-a-judge approach to measure perturbation strength in a unified manner across input and CoT perturbations. We then evaluate the self-consistency in explanations generated from various LLMs under controlled strength conditions, ensuring a fair comparison across perturbation types. Experiments show that our proposed LLM-based perturbation strength measure outperforms other embedding- and probability-based approaches and that input perturbations generally affect LLMs more strongly than CoT perturbations. Our work suggests that judgments about a model's self-consistency is fair only within the same perturbation type.
    
[^48]: I-Parakeet：移动NPU上的纯整数Conformer语音识别

    I-Parakeet: Integer-Only Conformer ASR on Mobile NPU

    [https://arxiv.org/abs/2609.30846](https://arxiv.org/abs/2609.30846)

    提出I-Parakeet，一种可在智能手机NPU上无需浮点运算运行的纯整数Conformer语音识别模型，通过整数化相对位置自注意力、优化Swish近似和逐层激活范围分析实现高效边缘部署。

    

    本文提出了I-Parakeet，这是NVIDIA Parakeet-CTC（0.6B参数）的纯整数实现，可在智能手机NPU上运行，无需任何浮点算子或CPU回退。现代Conformer语音识别模型由于其规模难以部署在边缘设备上，且量化模型在数值敏感操作上仍会回退到浮点计算。这阻碍了它们充分利用移动NPU等整数加速器。为实现这一目标，我们的贡献有三方面。首先，我们推导了Conformer核心中相对位置自注意力的整数形式化表示，将其两个具有不同量化尺度的分数分支以及相对位移融合为纯整数运算。其次，我们提出了一种极小极大优化的Swish近似方法，以最小化Swish输出的最大误差。第三，对激活值的逐层范围分析产生了两种针对性解决方案：一种用于

    arXiv:2609.30846v1 Announce Type: new  Abstract: In this paper, we propose I-Parakeet, an integer-only implementation of NVIDIA's Parakeet-CTC (0.6B parameters) that runs on a smartphone NPU without any floating-point operator or CPU fallback. Modern Conformer ASR models are hard to deploy on edge devices because of their size, and quantized models still fall back to floating point for numerically sensitive operations. This prevents them from fully exploiting integer accelerators such as mobile NPUs. To achieve this, our contributions are threefold. First, we derive an integer formulation of the relative-positional self-attention at the core of the Conformer. We fuse its two score branches with different quantization scales and the relative shift into integer-only operations. Second, we introduce a minimax-optimized Swish approximation that minimizes the maximum error of the Swish output. Third, a layer-wise range analysis of activations yields two targeted remedies: an INT16 grid for 
    
[^49]: 循环Transformer的量化：反馈暴露与校准盲区

    Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness

    [https://arxiv.org/abs/2609.30820](https://arxiv.org/abs/2609.30820)

    该论文揭示了循环Transformer低比特训练后量化的两种失效模式——“反馈暴露”（无恒等路径的量化层误差在递归中被反复放大回馈，且该现象同样存在于Mamba等非Transformer架构）和“校准盲区”（单步GPTQ仅依赖第0步激活校准，忽视后续递归步的输入方向）。

    

    循环Transformer在递归步骤间复用权重，这使得低比特量化对其格外有吸引力。我们识别出标准训练后量化的两种不同失效模式。在Huginn-3.5B模型上，逐通道INT4量化主要在非残差循环入口适配器处失效，而量化残差核心部分造成的损害要小得多。我们将这一现象称为反馈暴露：量化层在缺乏恒等路径的情况下扰动递归状态，由此产生的误差会在后续步骤中被反馈回来。在线性滤波器和Mamba状态空间模型上的受控实验表明，反馈暴露也存在于Transformer架构之外。分组INT4则揭示了另一种独立的失效模式——校准盲区：我们的单步GPTQ基线仅从第0步激活构建Hessian矩阵，导致递归后续步骤中使用的输入方向几乎未被纳入加权。在来自七种循环架构的九个检查点上，单步GPTQ在p

    arXiv:2609.30820v1 Announce Type: cross  Abstract: Looped transformers reuse weights across recurrence steps, making low-bit quantization especially attractive. We identify two distinct failure modes of standard post-training quantization. On Huginn-3.5B, per-channel INT4 fails primarily at the non-residual loop-entry adapter, while quantizing the residual core is much less damaging. We call this feedback exposure: a quantized layer perturbs the recurrent state without an identity path, and the resulting error is fed back at later steps. Controlled experiments on linear filters and Mamba state-space models show that feedback exposure also occurs outside transformers. Grouped INT4 reveals a separate failure, calibration blindness: our one-step GPTQ baseline builds its Hessian from step-0 activations, leaving input directions used later in the recurrence nearly unweighted. Across nine checkpoints from seven looped architectures, one-step GPTQ is worse than round-to-nearest (RTN) on the p
    
[^50]: 理解提示模板在面向安全对齐的知识蒸馏中的作用

    Understanding the Role of Prompt Template in Knowledge Distillation for Safety Alignment

    [https://arxiv.org/abs/2609.30802](https://arxiv.org/abs/2609.30802)

    研究发现知识蒸馏中的提示模板选择会显著影响学生模型已有的安全对齐，使用对话模板会降低模型安全性，而非对话模板能更好地保留安全对齐与模型的内部表示。

    

    先前的研究已经证明，在监督微调（SFT）期间提示模板的选择会显著影响后续安全对齐的鲁棒性。然而，在从教师模型到学生模型的知识蒸馏（KD）过程中，模板选择的影响在很大程度上仍未被探索。因此，我们通过分析不同模板配置如何影响学生模型已有的安全对齐来填补这一空白。我们观察到，经过对齐的基础指令微调模型中的安全对齐出现了显著退化。具体而言，我们发现与非对话模板相比，使用对话模板会使模型更容易顺从有害查询。这些发现在LLaMA、Gemma和Qwen三个模型家族中保持一致，并通过多个安全基准进行了评估。我们进一步表明，在蒸馏过程中使用非对话模板能更好地保留基础学生模型的内部表示。

    arXiv:2609.30802v1 Announce Type: new  Abstract: Prior research has demonstrated that the choice of prompt template during Supervised Fine-Tuning (SFT) significantly impacts the robustness of safety alignment afterwards. However, the influence of template selection during Knowledge Distillation (KD) from teacher to student remains largely unexplored. Thus, we fill this gap by analyzing how different template configurations influence the pre-existing safety alignment of the student. We observe a significant degradation of safety alignment present in the aligned base instruct-tuned model. Specifically, we find that utilizing chat templates renders the model more compliant with harmful queries compared to a non-chat template. These findings are consistent across three models: LLaMA, Gemma and Qwen model families and are evaluated across multiple safety benchmarks. We further show that using a non-chat template during distillation better preserves the base student's internal representation
    
[^51]: 面向冻结语言模型后期音频扩展的共生架构

    Symbiotic Architecture for Post-Hoc Audio Extension of Frozen Language Models

    [https://arxiv.org/abs/2609.30784](https://arxiv.org/abs/2609.30784)

    该论文提出一种共生架构，通过注入器模块将音频条件化向量直接写入冻结LLM的KV缓存，使其无需微调权重即可具备音频理解能力，同时完整保留原有语言能力并获得更优的可扩展性。

    

    本文提出了一种无需微调大语言模型（LLM）权重即可为其赋予音频理解能力的架构。所提出的共生架构采用一个注入器模块，将音频条件化的向量直接写入目标LLM的短期记忆，即键值（KV）缓存中，从而使LLM能够作为音频语言模型（ALM）运行。该架构的优势有两个方面：第一，它提升了ALM的可扩展性——由于所提方法在音频注入过程中绕过了LLM本体，注入成本由注入器的宽度而非骨干网络的宽度决定，因此其成本增长速度可以慢于全骨干网络预填充的成本；第二，由于训练方案不更新LLM的权重，LLM的原始能力得以完整保留，避免了微调可能导致的能力退化风险。所提方法的有效性在音频相关任务上进行了评估。

    arXiv:2609.30784v1 Announce Type: cross  Abstract: This paper proposes an architecture for equipping large language models (LLMs) with audio-understanding capabilities without fine-tuning their weights. The proposed symbiotic architecture employs an injector module that writes audio-conditioned vectors directly into the target LLM's short-term memory, i.e., the key-value (KV) cache, enabling the LLM to behave as an audio language model (ALM). The architectural advantages are twofold. First, it improves the scalability of ALMs: because the proposed method bypasses the LLM during audio injection, the injection cost is governed by the injector width rather than the backbone width, and can therefore scale more slowly than the cost of full-backbone prefilling. Second, since the training scheme does not update the LLM weights, the original capabilities of the LLM are preserved without the risk of degradation from fine-tuning. The effectiveness of the proposed method is evaluated on both audi
    
[^52]: 通过随机化引导在串联式语音到语音模型中学习自然对话行为

    Learning Natural Conversational Behavior in Tandem Speech-to-Speech Models with Randomized Guidance

    [https://arxiv.org/abs/2609.30773](https://arxiv.org/abs/2609.30773)

    该论文提出“随机化中间引导”方法，直接从对话语料库获取训练引导而无需用模拟器LLM生成后端引导，大幅降低数据准备开销，同时教会语音前端有选择性地利用后端LLM信息，从而学习自然的对话行为。

    

    串联式语音到语音架构将一个响应迅速的语音前端与一个异步的文本后端相结合。在KAME中，大语言模型（LLM）作为后端，在用户仍在说话时向语音前端提供候选响应作为引导。普通的对话录音只捕获了最终的响应，却没有记录后端在用户说话期间所提供的引导。在真实对话数据上训练时，使用模拟器LLM来生成这些缺失的引导会带来大量的数据准备开销。我们提出了随机化中间引导方法，它直接从对话语料库中获取引导，而无需模拟后端LLM的行为。在训练过程中，目标响应提供有信息量的引导，而随机采样的响应则在话语期间提供可能无关的更新。这种组合旨在教会前端有选择性地使用后端信息。

    arXiv:2609.30773v1 Announce Type: new  Abstract: Tandem speech-to-speech architectures couple a responsive speech frontend with an asynchronous text backend. In KAME, a large language model (LLM) serves as the backend, supplying candidate responses as guidance to the speech frontend while the user is still speaking. Ordinary conversation recordings capture the eventual response but not the guidance the backend would supply during the user's utterance. Generating the missing guidance with a simulator LLM adds substantial data-preparation overhead when training on real conversations. We propose randomized intermediate guidance, which derives guidance directly from the conversation corpus rather than simulating backend LLM behavior. During training, target responses provide informative guidance, while randomly sampled responses provide potentially irrelevant updates during the utterance. This combination aims to teach the frontend to use backend information selectively. On synthetic dialo
    
[^53]: SEA-CLIP-Tiny：面向东南亚语言的高效多语言文本-视觉嵌入模型

    SEA-CLIP-Tiny: Efficient Multilingual Text-Vision Embedding for Southeast Asian Languages

    [https://arxiv.org/abs/2609.30739](https://arxiv.org/abs/2609.30739)

    本文提出SEA-CLIP-Tiny，一个参数量不足5000万的紧凑型多语言文本-视觉嵌入模型，通过区域数据筛选和多语言教师引导，在七种东南亚语言的跨语言图像-文本检索上取得最优平均性能，且比MobileCLIP2参数更少、CPU延迟更低。

    

    多语言文本-视觉嵌入模型对于跨语言图像-文本检索至关重要，但由于该地区语言多样性以及数据和计算资源的限制，东南亚语言的支持仍然不足。在本文中，我们提出了SEA-CLIP-Tiny，一个面向东南亚的紧凑型多语言文本-视觉嵌入模型，参数量少于5000万。我们的模型通过区域数据筛选和多语言教师模型引导，将CLIP-KD风格的框架适配到东南亚多语言环境中。在七种东南亚语言上的实验表明，SEA-CLIP-Tiny在所评估的学生模型中取得了最强的平均检索性能，在R@1、R@5和R@10上分别达到12.9%、31.5%和42.2%。与MobileCLIP2相比，它在参数量减少38.4%、CPU实测延迟更低的情况下，将平均R@10提升了12.1个百分点。这些结果凸显了区域感知……（原文摘要至此截断）

    arXiv:2609.30739v1 Announce Type: new  Abstract: Multilingual text-vision embedding models are essential for cross-lingual image-text retrieval, but Southeast Asian languages remain poorly supported due to the region's linguistic diversity and limited data and computing resources. In this paper, we introduce SEA-CLIP-Tiny, a compact multilingual text-vision embedding model for Southeast Asia with fewer than 50M parameters. Our model adapts a CLIP-KD-style framework to Southeast Asian multilingual settings through regional data curation and multilingual teacher guidance. Experiments across seven Southeast Asian languages show that SEA-CLIP-Tiny achieves the strongest average retrieval performance among the evaluated student models, reaching 12.9%, 31.5%, and 42.2% at R@1, R@5, and R@10, respectively. Compared with MobileCLIP2, it improves average R@10 by 12.1 points while using 38.4% fewer parameters and lower measured CPU latency. These results highlight the importance of region-aware 
    
[^54]: 超越平均注意力：面向KV缓存淘汰的多样性感知分层评分

    Beyond Mean Attention: Diversity-Aware, Layer-Wise Scoring for KV Cache Eviction

    [https://arxiv.org/abs/2609.30738](https://arxiv.org/abs/2609.30738)

    该论文提出在注意力均值之外加入注意力离散度与冗余惩罚（类MMR多样性）的统一KV缓存淘汰评分，并发现仅用一个全局多样化常数即可在多数LongBench数据集上带来提升，而无需按层或按数据集精细搜索。

    

    SnapKV和PyramidKV等KV缓存淘汰方法仅依据较小观察窗口上的平均注意力对token进行排序。我们研究了一个统一的评分：$\mu_i+\lambda_1\sigma_i+\lambda_2\mathrm{corr}(i,S)$，在平均注意力之外引入了跨窗口查询的注意力离散度以及相对于已选token的冗余度。当$\lambda_2<0$时，该评分会惩罚与已选token的相似性，类似于最大边际相关性（MMR），且无需额外的前向传播。为检验这种相关性-多样性平衡是否应随网络深度变化，我们将固定全局系数与三段式及二次型深度轮廓进行比较，仅在开发集上于$\sinh$重参数化下搜索这些深度轮廓。在全部16个英文LongBench数据集上使用Mistral-7B、每层预算为64的设置下，单一的全局多样化常数改进了16个数据集中的13个（宏观平均提升1.1）；该增益在预算为32时依然保持，在预算为128时收窄。逐数据集搜索未发现可检测的（分层模式差异）。

    arXiv:2609.30738v1 Announce Type: new  Abstract: KV cache eviction methods such as SnapKV and PyramidKV rank tokens solely by mean attention over a small observation window. We study a unified score, $\mu_i+\lambda_1\sigma_i+\lambda_2\mathrm{corr}(i,S)$, adding attention dispersion across window queries and redundancy relative to selected tokens. For $\lambda_2<0$, the score penalizes similarity to selected tokens as in maximal marginal relevance (MMR), without extra forward passes. To test whether this relevance-diversity balance should vary with depth, we compare fixed global coefficients with three-segment and quadratic profiles. Only these depth profiles are searched on a development split under a $\sinh$ reparameterization. On all 16 English LongBench datasets with Mistral-7B at a budget of 64 entries per layer, a single global diversification constant improves 13 of 16 datasets (macro +1.1); the gain holds at budget 32 and narrows at 128. Per-dataset search finds no detectable la
    
[^55]: 言语重于顺序：Gemma 4 的行为评估

    Words Speak Louder Than Order: A Behavioral Evaluation of Gemma 4

    [https://arxiv.org/abs/2609.30716](https://arxiv.org/abs/2609.30716)

    该研究通过完全平衡的实验设计评估 Gemma 4 模型在接收冲突文档时的行为，发现信息源的语义表述（如“官方指南”或“最新更新”）对模型最终答案的影响显著强于文档的呈现顺序。

    

    当语言模型接收到两份相互冲突的文档作为输入时，它如何决定优先考虑哪一份？它是依赖于信息源的表述方式，还是文档的呈现顺序？我们在谷歌预训练的 Gemma 4-e4b 模型上，通过一个针对性的行为测试套件（n = 13 个条目，在简短的单轮对话情境下进行 784 次前向传播），采用完全平衡的实验设计对这一行为进行了评估。这一设置使我们能够在数学上隔离信息源表述和阅读位置的具体影响，同时确保模型固有的词汇偏好被抵消。在十个测试条件下，我们发现了以下结果：1. 信息源表述显著压过阅读位置。当二者直接竞争时，信息源的语义表述（例如将其呈现为官方指南或最新更新）对模型最终答案的影响显著强于文档的呈现顺序。

    arXiv:2609.30716v1 Announce Type: cross  Abstract: When a language model receives two conflicting documents as input, how does it decide which one to prioritize? Does it rely on how the sources are framed or the presentation order of the documents? We evaluated this behavior on Google's pre-trained Gemma 4-e4b model across a targeted behavioral suite (n = 13 items, 784 forward passes in short, single-turn contexts) using a completely counterbalanced experimental design. This setup allowed us to mathematically isolate the specific effects of source framing and reading position, while ensuring the model's natural vocabulary biases were canceled out.   Across ten test conditions, we discovered the following:   1. Source framing heavily overpowers reading position. When directly competing, the semantic framing of a source (such as presenting it as an official guideline or a fresh update) had a significantly stronger impact on the model's final answer than the presentation order of the docu
    
[^56]: LAVOIR：通过摊销信息价值教会单次前向传播决策编码器何时提问以及问什么

    LAVOIR: Teaching a Single-Pass Decision Encoder When and What to Ask with Amortized Value of Information

    [https://arxiv.org/abs/2609.30706](https://arxiv.org/abs/2609.30706)

    LAVOIR将候选缺失信息槽位与答案选项一同置于输入中，使单次前向传播同时输出决策分布和各槽位的信息价值预期增益，从而以无需人工标注的方式教会决策模型何时提问、问什么。

    

    “System One”（系统一）决策模型，如TypeSafe的Jev及其开源对应模型Laya，能够在单次前向传播中以校准的概率回答关于文本的类型化问题，但它们无法询问缺失的信息：当首条消息没有说明区分两个部门的关键内容时，它们只能靠猜测。我们提出了LAVOIR（Laya with Value-Of-Information Routing，带信息价值路由的Laya），该方法将候选的缺失信息片段（槽位）放置在输入中答案选项的旁边，使得一次前向传播既能返回决策分布，又能针对每个槽位返回如果向用户询问该信息时正确决策概率的预期增益。VOI目标无需人工标注：黄金决策来自模式规则，LLM仅负责将消息和答案转化为文字表述，来自另一个模型家族的模型对每个文本进行校验，并且通过将每条消息与多个用户画像配对，对已实现增益进行回归从而估计预期增益。基尼不纯度上限约束了预（原文摘要在此处被截断）。

    arXiv:2609.30706v1 Announce Type: new  Abstract: "System One" decision models such as TypeSafe's Jev and its open counterpart Laya answer typed questions about a text in a single forward pass with calibrated probabilities, but they cannot ask for missing information: when a first message does not say what separates two departments, they guess. We present LAVOIR (Laya with Value-Of-Information Routing), which places the candidate pieces of missing information (slots) in the input next to the answer options, so that one forward pass returns both the decision distribution and, for every slot, the expected gain in the probability of the correct decision if the user were asked about it. VOI targets need no human labels: gold decisions come from schema rules, an LLM only verbalizes messages and answers, a model from another family checks every text, and pairing each message with several profiles makes regression on realized gains estimate the expected gain. A Gini-impurity cap bounds the pre
    
[^57]: TRACE：流媒体视频理解的时间审计与条件感知评估

    TRACE: Temporal Audit and Condition-aware Evaluation of Streaming Video Understanding

    [https://arxiv.org/abs/2609.30670](https://arxiv.org/abs/2609.30670)

    TRACE是一个面向流媒体视频理解的条件感知评测框架，通过时间审计任务、证据时序与触发标注以及因果Core-Adapter协议，显式控制并记录信息可用性与响应行为，从而对答案质量、及时性、响应选择、工作负载、完成度和可靠性进行多维度评估，揭示了相似分数背后不同的工作负载与失败模式。

    

    流媒体视频理解要求模型在证据到达时即进行解读，然而当前的评估往往只报告任务分数，而没有说明证据何时生效、视觉历史如何被维护、以及响应如何被触发。因此，相似的分数可能对应着不同的工作负载、失败模式和运行行为。我们提出了TRACE（时间审计与条件感知评估），这是一个使上述因素显式化的条件感知基准与评估框架。TRACE将经过时间审计的视觉任务与证据时序及依赖指令的触发标注相结合，采用统一的因果Core-Adapter协议，在记录实际的历史处理和响应事件的同时控制信息的可用性，并对答案质量、及时性、响应选择行为、工作负载、完成度和可靠性进行多维度报告。在来自517个视频的1,240条记录上，我们评估了八个公开可用的（模型）……

    arXiv:2609.30670v1 Announce Type: new  Abstract: Streaming video understanding requires models to interpret evidence as it arrives, yet current evaluations often report task scores without specifying when evidence becomes valid, how visual history is maintained, or how responses are triggered. As a result, similar scores may correspond to different workloads, failure modes, and operational behavior. We introduce TRACE (Temporal Audit and Condition-aware Evaluation), a condition-aware benchmark and evaluation framework that makes these factors explicit. TRACE combines temporally audited visual tasks with evidence timing and instruction-dependent trigger annotations, a unified causal Core--Adapter protocol that controls information availability while recording actual history processing and response events, and multidimensional reporting of answer quality, timeliness, response-selection behavior, workload, completion, and reliability. On 1,240 records from 517 videos, we evaluate eight pu
    
[^58]: 通过攻击链建模实现邮件智能体的提示注入检测

    Prompt Injection Detection for Email Agents Through Attack Chain Modeling

    [https://arxiv.org/abs/2609.30657](https://arxiv.org/abs/2609.30657)

    该论文提出了一种针对邮件智能体的提示注入检测框架，将检测问题从传统的二元恶意文本分类转变为对多阶段攻击链的建模，结合文本检测器、分阶段验证器、规则风险信号、意图-行动一致性分析与逻辑斯蒂决策策略，并在多种迁移设置下验证了其有效性。

    

    大语言模型邮件助手特别容易受到间接提示注入攻击，因为不可信的邮件内容会被检索到模型上下文中，进而影响后续的工具调用。现有的提示注入检测器主要将此问题表述为二元恶意文本分类任务，却忽略了一个重要因素：有害的智能体行为往往是通过一系列阶段逐步形成的。我们提出了一种检测框架，通过结合文本检测器、针对各攻击阶段的专用验证器、显式的基于规则的风险信号、用户意图与行动一致性分析以及逻辑斯蒂决策策略来建模这一攻击链。为支持该框架，我们从提示注入数据集中推导出攻击链标签，在随机划分、时间阶段迁移、条件阶段迁移、跨数据集迁移等多种设置下评估所提出的框架，并在多个基准上进行消融研究。结果表明，随机（划分下）……

    arXiv:2609.30657v1 Announce Type: cross  Abstract: Large language model email assistants are particularly vulnerable to indirect prompt injection because untrusted email content can be retrieved into the model context and influence subsequent tool use. Existing prompt injection detectors mainly formulate this problem as binary malicious text classification, which overlooks the important factor that harmful agent behavior often arises through a sequence of stages. We propose a detection framework that models this attack chain by combining a text detector, verifiers specific to each stage, explicit rule-based risk signals, user intent and action consistency analysis, and a logistic decision policy. To support this framework, we derive attack chain labels from prompt injection datasets, evaluate the proposed framework under random splits, temporal phase transfer, conditional stage transfer, cross-dataset transfer, and conduct ablation studies on multiple benchmarks. Results show that rand
    
[^59]: 通过同策略蒸馏实现推理能力的递归自我提升

    Recursive Self-Improvement via On-Policy Distillation for Reasoning

    [https://arxiv.org/abs/2609.30652](https://arxiv.org/abs/2609.30652)

    该论文提出一个递归自我提升框架，让特权教师模型在训练过程中不断吸收学生模型学到的改进而非保持冻结，从而克服了传统同策略自蒸馏中教师固定所带来的局限性。

    

    同策略蒸馏（OPD）通过让学生模型自行生成轨迹，然后将其下一词元的预测与外部教师模型的下一词元预测进行匹配来训练学生模型，从而为学生提供密集的、词元级别的监督信号。同策略自蒸馏（OPSD）则消除了对外部教师的依赖：具体而言，将学生模型的第二个冻结副本作为教师，并在其上下文中提供真实答案作为特权信息。学生模型只接收问题本身，并学习模仿这个拥有特权信息的教师模型，而教师在整个训练过程中保持冻结。先前的研究表明，冻结教师有助于训练稳定性，但我们认为这会阻碍教师模型吸收学生在训练过程中所学到的改进。我们的主要贡献是通过一个围绕两个互补组件构建的递归框架来解决这一局限性。首先，我们让拥有特权信息的（摘要在此处截断）

    arXiv:2609.30652v1 Announce Type: new  Abstract: On-policy distillation (OPD) trains a student model by having it generate trajectories, then matching its next-token predictions with an external teacher's next-token predictions. This provides dense, token-level supervision to the student. On-policy self-distillation (OPSD) eliminates the need for the external teacher. Specifically, a second frozen copy of the student model, now given the ground truth in its context, serves as the teacher. The student model only receives the problem and learns to mimic the privileged teacher model, while the teacher remains frozen throughout training. Previous work showed that freezing the teacher is useful for training stability, but we argue that this can prevent the teacher from incorporating the improvements learned by the student during training. Our primary contribution is to address this limitation with a recursive framework built around two complementary components. First, we let the privileged 
    
[^60]: 爱泼斯坦文件引擎：面向调查性新闻的智能体搜索

    Epstein Files Engine: Agentic Search for Investigative Journalism

    [https://arxiv.org/abs/2609.30611](https://arxiv.org/abs/2609.30611)

    《纽约时报》开发的“爱泼斯坦文件引擎”通过大语言模型将记者问题转化为SQL查询，在三百万页司法部文件中检索带引用的可验证答案，助力100余名记者完成至少20篇报道，其核心创新Diff重复匹配方法放大了新颖性信号，并证明了新闻编辑室AI智能体应作为源材料接口而非自主写作者来发挥最大价值。

    

    2026年1月30日，美国司法部发布了一组关于杰弗里·爱泼斯坦的多媒体资料合集，其中包括约三百万页PDF文档。我们介绍了“爱泼斯坦文件引擎”，这是《纽约时报》为调查这些文件而部署的人工智能智能体。该引擎将记者的问题转化为针对三个语料库的Google BigQuery SQL查询：爱泼斯坦相关文件发布、时报档案库以及外部的爱泼斯坦相关新闻标题。它使用大语言模型来规划查询，并返回带有丰富引用的答案，使记者能够验证并信赖这些结果。超过100名记者使用了该引擎，它为至少20篇已发表的报道做出了贡献。我们报告了记者们如何使用该引擎进行查询，并介绍了Diff——我们开发的文本与视觉重复匹配方法，该方法放大了新颖性信号，使引擎能够挖掘出真正的新信息。我们提出，新闻编辑室的智能体不应作为自主写作者，而应作为连接源材料的接口，这样才能最好地服务于新闻编辑室。

    arXiv:2609.30611v1 Announce Type: cross  Abstract: On Jan. 30, 2026, the U.S. Department of Justice released a mixed-media collection concerning Jeffrey Epstein, including about three million pages of PDFs. We describe the Epstein Files Engine, an A.I. agent The New York Times deployed to investigate the files. The Engine translated reporter questions into Google BigQuery SQL queries across three corpora: Epstein-related releases, the Times's archive and external, Epstein-related news headlines. It used an LLM to plan queries and returned citation-rich answers a reporter could verify and trust. More than 100 journalists used the Engine, and it contributed to at least 20 published stories. We report how reporters queried it and describe Diff, our text-and-visual duplicate matching method that amplified novelty signals and allowed the Engine to surface genuinely new information. We argue that newsroom agents serve newsrooms best not as autonomous writers, but as interfaces to source mate
    
[^61]: 搜索之后的难题：对网络智能体在知识综合、组织与展示能力上进行基准测试

    The Hard Part Comes After Search: Benchmarking Web Agents on Synthesizing, Organizing, and Displaying Knowledge

    [https://arxiv.org/abs/2609.30604](https://arxiv.org/abs/2609.30604)

    该论文提出了KNOWS基准，通过开放式、复杂的浏览器任务联合评估网络智能体在信息检索、知识综合、任务分解以及产出文档等最终制品方面的综合能力。

    

    arXiv:2609.30604v1 公告类型：cross 摘要：现有的计算机使用智能体基准测试并未充分评估作为助手的智能体。一个有用的助手需要在复杂的多步骤工作流中检索信息，将其综合为制品（如文档、演示文稿、电子表格），并操作程序界面以产出连贯的最终成果。这类工作流需要推理与综合能力、复杂任务的分解能力，以及视觉与空间理解能力。为了在类似的工作流上研究智能体，我们推出了KNOWS，这是一个由开放式、复杂的基于浏览器的任务组成的基准测试，能够联合评估这些能力，且每个任务都以产出一个制品为最终目标。为了编写任务，我们开发了任务设计准则以及确保任务满足要求的协议。每个任务都配有一个评估器——一个将确定性检查与大语言模型判断相结合的程序，以在智能体评估中固有的丰富性、可靠性与自动化之间取得平衡。

    arXiv:2609.30604v1 Announce Type: cross  Abstract: Existing computer-use agent benchmarks do not fully evaluate agents acting as assistants. A useful assistant retrieves information across complex, multi-step workflows, synthesizes it into artifacts (documents, presentations, spreadsheets), and navigates program interfaces to produce a coherent final product. Such workflows demand reasoning and synthesis, decomposition of complex tasks, as well as visual and spatial understanding. To study agents on workflows like these, we introduce KNOWS, a benchmark of open-ended, complex, browser-based tasks that jointly evaluate these capabilities, with each task culminating in a produced artifact. To write tasks, we develop a task design rubric and a protocol for ensuring that tasks meet the requirements. Each task is paired with an evaluator, a program that combines deterministic checks with LLM judgments to balance the richness, reliability, and automation tradeoff inherent to agent evaluation.
    
[^62]: 少思考，模拟得更好：直觉式提示提升大语言模型智能体对个体社交媒体反应的模拟能力，包括对陌生内容的反应

    Thinking Less to Simulate Better: Intuitive Prompting Improves LLM Agents Simulating Individual Social Media Reactions, Including Unfamiliar Content

    [https://arxiv.org/abs/2609.30563](https://arxiv.org/abs/2609.30563)

    本研究通过八名真实用户画像与六十八条帖子反应的对照实验发现，采用直觉式（少推理）提示并以态度性内容而非人口统计背景构建用户画像，可显著提升大语言模型智能体模拟个体社交媒体真实反应的准确性（包括对陌生内容的反应），且智能体与画像的一致性并不等同于行为保真度。

    

    平台政策越来越多地在人工用户上进行测试，这使得智能体的保真度变得十分重要。然而，令人信服的虚假用户资料也可能在选举前操纵公众舆论的感知。以往的验证工作主要集中于智能体与人类行为的一致性，却很少关注智能体的行为是否符合其被赋予的资料。本研究通过问卷调查、深度访谈和书面自我介绍对八名塞尔维亚参与者进行画像，记录了他们对六十八条社交媒体帖子的反应，并让四个语言模型在五种提示条件（资料内容和指令风格各不相同）下预测这些反应。结果表明，态度性内容相比人口统计学背景故事大幅提升了预测效果；智能体与其所述资料的匹配程度甚至高于参与者与其自身问卷答案的匹配程度；而且一旦提供了资料信息，一致性与保真度之间并无关联。指令……

    arXiv:2609.30563v1 Announce Type: new  Abstract: Platform policies are increasingly tested on artificial users, making agent fidelity important. Yet convincing fake profiles could also manipulate perceived public opinion before elections. Validation has concentrated on agreement with human behaviour and has paid little attention to whether an agent behaves in line with the profile it was given. The present study profiled eight Serbian participants through a questionnaire, a deep interview, and a written self-presentation, recorded their reactions to sixty-eight social media posts, and asked four language models to predict those reactions under five prompt conditions varying profile content and instruction style. Attitudinal content improved prediction over demographic backstories by a wide margin. Agents matched their stated profiles more closely than participants matched their own survey answers, and consistency proved unrelated to fidelity once profile information was present. Instru
    
[^63]: 通过认知实验范式探究智能体记忆中的稳定性-可塑性权衡

    Probing Stability-Plasticity Tradeoffs in Agent Memory through Cognitive Experimental Paradigms

    [https://arxiv.org/abs/2609.30558](https://arxiv.org/abs/2609.30558)

    本文提出受认知科学启发的MemProbe框架，通过干扰、错误信息、巩固强度和再巩固窗口四种可复用的实验范式，超越传统最终答案准确率评估，系统诊断智能体记忆中稳定性与可塑性之间的权衡。

    

    智能体记忆系统越来越多地被用于维护长期用户偏好、任务状态和不断演变的事实，但当前的评估方法往往将记忆行为简化为最终答案的准确率。我们提出了MemProbe，一个受认知科学启发的框架，用于诊断智能体记忆中的稳定性-可塑性权衡。该框架的动机来自认知记忆研究的一个核心洞察：记忆是重构性的，并受到干扰、来源可靠性、强化和再激活等因素的塑造。MemProbe将这一洞察转化为四种可复用的实验范式（干扰、错误信息、巩固强度和再巩固窗口），这些范式操纵了记忆何时应该被更新、保留或视为不确定。该框架进一步将正确性分解为行为特征，揭示系统如何更新、保留、归因和时间性地组织信息。我们在56个剧集的诊断任务中实例化了这些范式（摘要在此处被截断）。

    arXiv:2609.30558v1 Announce Type: cross  Abstract: Agent memory systems are increasingly used to maintain long-term user preferences, task states and evolving facts, but current evaluations often collapse memory behavior into final-answer accuracy. We introduce MemProbe, a cognitive-science-inspired framework for diagnosing stability-plasticity tradeoffs in agent memory. The framework is motivated by a core insight from cognitive memory research: memory is reconstructive and shaped by interference, source reliability, reinforcement, and reactivation. MemProbe turns this insight into four reusable experimental paradigms (interference, misinformation, consolidation strength, and reconsolidation window) that manipulate when a memory should be updated, preserved, or treated as uncertain. It further decomposes correctness into behavioral profiles that reveal how systems update, preserve, attribute, and temporally organize information. We instantiate these paradigms in a 56-episode diagnosti
    
[^64]: REALMS：一种面向高维嵌套用户画像的实时精确受众规模估算AI助手对话系统

    REALMS: An AI-Assistant Conversational System for Real-Time Exact Audience Sizing over High-Dimensional Nested Profiles

    [https://arxiv.org/abs/2609.30547](https://arxiv.org/abs/2609.30547)

    REALMS是一个部署在生产环境的对话式系统，结合嵌入向量检索与大语言模型NL2SQL技术，让营销人员用自然语言在数秒内对数百万级、上千属性的高维用户画像完成精确受众规模估算。

    

    受众规模估算是数字营销中的关键组成部分，它能够实现精确的资源分配、营销活动规划和效果优化。传统方法（如骨架受众、抽样或预测建模）存在显著的延迟、估算误差，以及在高维画像数据上可扩展性差的问题。我们提出了REALMS（基于大语言模型多属性搜索的实时精确受众规模估算系统），这是一个部署在企业客户数据平台生产环境中的精确受众规模估算对话系统。REALMS使营销人员能够使用自然语言查询拥有数百万用户画像和数千个属性的海量画像库，并在几秒钟内获得精确的计数结果。该系统引入了三个关键组件：(1) 一种基于嵌入向量搜索的分类属性检索机制，无需手动配置即可动态识别相关的模式属性；(2) 一种由大语言模型驱动的自然语言转SQL（NL2SQL）组件……

    arXiv:2609.30547v1 Announce Type: new  Abstract: Audience sizing is a critical component of digital marketing. It enables precise resource allocation, campaign planning, and performance optimization. Traditional approaches using skeleton audiences, sampling, or predictive modeling suffer from significant delays, estimation errors, and poor scalability over high-dimensional profile data. We present REALMS (Real-time Exact Audience sizing via LLM-based Multi-attribute Search), a conversational system for exact audience sizing deployed in production on an enterprise customer data platform. REALMS enables marketers to query massive profile stores with millions of profiles and thousands of attributes using natural language and receive precise counts in seconds. The system introduces three key components: (1) a categorical attribute retrieval mechanism using embedding-based vector search to dynamically identify relevant schema attributes without manual configuration; (2) an LLM-powered NL2SQ
    
[^65]: 别再CLAP：音乐-文本模型只是词袋模型吗？

    Don't CLAP: Are Music-Text Models Bag-of-Words?

    [https://arxiv.org/abs/2609.30540](https://arxiv.org/abs/2609.30540)

    本研究通过属性交换扰动实验发现，现有对比式音乐-文本模型的文本嵌入无法可靠捕捉乐器与属性之间的绑定关系，揭示了CLAP分数作为文本-音乐忠实度评估指标存在根本性局限。

    

    文本到音乐系统的评估包括音频质量以及音乐对提示词的忠实程度，而CLAP分数（即音乐-文本模型的音频嵌入与文本嵌入之间的余弦相似度）是衡量忠实度的标准客观指标。我们探讨该分数能在多大程度上准确反映文本内容：当一个属性与乐器绑定时（例如失真吉他），文本嵌入能否捕捉这种绑定关系？为此，我们引入了一种属性交换扰动方法：通过在两种乐器之间交换恰好一个属性——音色、主奏与伴奏角色、或首次出现顺序——来编辑真实录音的说明文字。随后我们测试了四个对比式音乐-文本模型和一个大型音频-语言模型，考察音频与原始说明文字的匹配分数是否高于与被扰动说明文字的匹配分数。没有任何对比式模型能够可靠地区分两种说明文字。大型音频-语言模型表现更好，但进一步的实验……（摘要截断）

    arXiv:2609.30540v1 Announce Type: cross  Abstract: Text-to-music systems are assessed on audio quality and on how faithfully the music follows its prompt, and the CLAP score, the cosine similarity between a music-text model's audio and text embeddings, is the standard objective metric of faithfulness. We ask how accurately that score reflects the text: when an attribute is linked to an instrument (e.g., distorted guitar), does the text embedding capture that binding? To find out, we introduce an attribute swap perturbation: the caption of a real recording is edited by exchanging exactly one property, timbre, lead versus accompaniment, or order of first appearance, between two instruments. We then test four contrastive music-text models and one large audio-language model on whether the audio scores higher against the original caption than against the perturbed one. No contrastive model distinguishes the two captions reliably. The audio-language model does better, but further experiments
    
[^66]: 给BabyLM喂通心粉：语码转换式课程引发跨语言趋同

    Feeding BabyLMs Macaroni: Code-Switching Curricula Cause Cross-Lingual Convergence

    [https://arxiv.org/abs/2609.30535](https://arxiv.org/abs/2609.30535)

    该论文的核心贡献是发现在预训练语料中插入词级和句子级语码转换数据可以诱导语言模型的跨语言表示对齐（尤其跨越不同文字系统），且这种对齐能在后续单语训练中保持，配合“词级→句子级语码转换→单语”的渐进课程可在BabyLM评测上超越基线。

    

    多语言社区中的儿童经常进行语码转换，在单句话语中混用多种语言。我们能否通过在语码转换文本上训练，使语言模型产生跨语言对齐？我们在两个1亿词规模的多语言语料库上预训练小型仅解码器Transformer：一个是混合英语、荷兰语和中文BabyBabelLM数据集得到的基础语料库，另一个是在此基础上用大语言模型插入词级和句子级语码转换而生成的语料库。我们发现，在语码转换数据上训练能使平行文本的表示对齐，尤其是跨越不同文字系统的对齐，并且这种对齐在后续单语文档训练中得以保持。在学习课程从词级语码转换、到句子级语码转换、再到单语文档逐步推进的设置下，用语码转换数据训练的模型在BabyLM评估套件上的表现优于未使用语码转换数据的基线模型。我们的工作刻画了语码转换课程……

    arXiv:2609.30535v1 Announce Type: new  Abstract: Children in multilingual communities often code-switch, using multiple languages in a single utterance. Can we induce cross-lingual alignment in language models by training on code-switched text? We pretrain small decoder-only transformers on two 100M-word multilingual corpora: a base corpus formed by mixing the English, Dutch, and Chinese BabyBabelLM datasets, and a corpus generated from it by inserting word- and sentence-level code-switching using an LLM. We find that training on code-switched data aligns the representations of parallel text, particularly across different scripts, and that this alignment persists through training on monolingual documents. Under a learning curriculum that progresses from word-level code-switching, to sentence-level code-switching, to monolingual documents, models trained on code-switched data outperform baselines trained without it on the BabyLM evaluation suite. Our work characterizes code-switching cu
    
[^67]: Inquesto Score：一种语音智能体可靠性评估协议

    Inquesto Score: A reliability Protocol For Voice Agents

    [https://arxiv.org/abs/2609.30514](https://arxiv.org/abs/2609.30514)

    本文提出 Inquesto Score（IS）协议，通过明确定义失败事件与严重级别，以固定版本化评估群体中成功完成呼叫者目标且无功能性故障的通话占比，来可复现、可解释地衡量已部署语音智能体的可靠性。

    

    语音智能体越来越多地被部署在工作流程中，其中失败的交互可能影响交易、访问权限及其他重要后果，因此需要可复现且可解释的评估方法。我们提出了 Inquesto Score（IS），这是一种衡量语音智能体可靠性的协议，其定义为：在一个固定且带版本控制的评估通话群体中，实现呼叫者目标且未发生功能性故障或更严重问题的通话所占的百分比。IS 并非将异构指标混合在一起，而是明确定义了失败事件和严重程度级别，并对已部署的语音流水线进行评估。时序故障（包括抢话和响应延迟）直接从音频中测量，而语义性和状态相关的故障则通过场景谓词、工具调用轨迹以及固定的开源模型评判者进行评估。该评分还附带关于行为、声学鲁棒性、身份处理和说话人群体的诊断视图，但这些诊断信息不会被合并进评分之中。

    arXiv:2609.30514v1 Announce Type: cross  Abstract: Voice agents are increasingly deployed in workflows where failed interactions can affect transactions, access, and other consequential outcomes, creating a need for reproducible and interpretable evaluation. We introduce Inquesto Score (IS), a protocol for measuring voice-agent reliability as the percentage of calls in a fixed, versioned evaluation population that achieve the caller's goal without a functional failure or worse. Rather than combining heterogeneous metrics, IS defines explicit failure events and severity levels and evaluates the deployed voice pipeline. Timing failures, including talk-over and delayed responses, are measured directly from audio, while semantic and state-dependent failures are evaluated using scenario predicates, tool traces, and a pinned open-model judge. Diagnostic views of behavior, acoustic robustness, identity handling, and speaker groups accompany the score without being combined into it. Inquesto S
    
[^68]: 打破同质性：多样化人格集合以实现创造性的LLM输出

    Breaking Homogeneity: Diversifying Persona Sets for Creative LLM Outputs

    [https://arxiv.org/abs/2609.30492](https://arxiv.org/abs/2609.30492)

    该论文将人格多样化建模为集合级条件化问题，在“选择vs生成”与“空间填充vs前沿探索”两个正交设计维度上提出四种方法，显著提升了语言模型创造性输出的多样性，其中进化式人格生成在AUT上将回答多样性提高了78.8%。

    

    语言模型在开放式任务中常常产生同质化的回答；这种同质性可能引发群体思维——即观点向单一且可能次优的决策趋同。我们将人格多样化表述为一个集合级条件化问题，并研究了两个正交的设计选择：是选择人格还是生成人格，以及是空间填充式多样性还是前沿探索式多样性。我们用四种方法实例化了这一设计空间，涵盖覆盖性与分散性子集选择、均匀覆盖采样以及进化式人格生成。在替代用途任务（AUT）、Infinity-Chat 和发散联想任务（DAT）上的评估表明，所提出的方法在各种任务和创造力目标上均展现出优势。在AUT上，与仅使用任务提示相比，进化式人格生成将回答多样性提高了78.8%，原创性提高了26.1%，灵活性提高了49.5%，整体创造力提高了13.9%，同时保持了98.5……

    arXiv:2609.30492v1 Announce Type: cross  Abstract: Language models often produce homogeneous responses to open-ended tasks; such homogeneity can spawn groupthink-the convergence of ideas toward a singular and potentially suboptimal decision. We formulate persona diversification as a set-level conditioning problem and study two orthogonal design choices: selecting versus generating personas, and space-filling versus frontier-seeking diversity. We instantiate this design space with four methods spanning coverage and dispersion subset selections, uniform-coverage sampling, and evolutionary persona generation. Evaluations on the Alternative Uses Task (AUT), Infinity-Chat, and Divergent Association Task (DAT) show the benefits of the proposed methods across tasks and creativity objectives. On AUT, evolutionary persona generation increases response diversity by 78.8%, originality by 26.1%, flexibility by 49.5%, and holistic creativity by 13.9% over task-only prompting, while maintaining 98.5
    
[^69]: AcoustiClaim：基于仪器真值的数值声明基准测试

    AcoustiClaim: A Numeric Claim Benchmark with Instrument Ground Truth

    [https://arxiv.org/abs/2609.30483](https://arxiv.org/abs/2609.30483)

    该论文提出 AcoustiClaim 基准，首次以定义声学量的仪器读数作为真值来验证音频语言模型给出的数值声明，发现绝大多数模型的表现不优于常数预测器，仅一个闭源模型在音高等少数任务上通过了秩相关阈值。

    

    音频语言模型会针对声学量给出具体数值，但无论是人类的主观意见还是评判模型，都无法判断该数值是否真实符合信号本身。AcoustiClaim 从自由文本中提取每个数值声明，根据定义该声学量的仪器对其打分，并按参考值可从何处读取对每个声学量进行分类。四个开放权重系统和一个闭源模型，在两个语料库上以五种提问方式被询问十个声学量，共填满207个评估单元格。其中49个单元格输出的不同数值少于五个；在158个可进行排序的单元格中，有8个超过了我们设定的0.3秩相关阈值，其中三个的置信区间明显高出该阈值，这八个中有五个来自同一个读取音高的闭源模型。在除三个之外的所有可排序单元格中，模型误差都处于或高于常数预测器的下限。我们训练的参考解码器在95%的混合音频样本上以自然语言方式拒绝给出五个语音相关声学量，不作任何保留，而在干净的孪生样本上则能陈述这些量，从而复现了其目标所遵循的规则……

    arXiv:2609.30483v1 Announce Type: cross  Abstract: Audio language models state numbers for acoustic quantities, and neither human opinion nor a judge model says whether such a number is true of the signal. AcoustiClaim extracts each numeric claim from free text, scores it against the instrument that defines the quantity, and classes each quantity by where its reference can be read. Four open-weight systems and one closed model, asked for ten quantities five ways on two corpora, fill 207 cells. Of these, 49 emit fewer than five distinct values, and eight of the 158 cells that can be ranked exceed a rank correlation of 0.3, the bar we set, three with an interval clear of it, five of them one closed model reading pitch. Error sits at or above a constant-predictor floor in every ranked cell but three. The reference decoder we train declines the five voice quantities in prose on 95% of mixtures, with nothing withheld, and states them on the clean twins, reproducing its targets' rule from au
    
[^70]: 面向目标说话人自动语音识别的非对称无分类器引导

    Asymmetric Classifier-Free Guidance for Target-Speaker ASR

    [https://arxiv.org/abs/2609.30476](https://arxiv.org/abs/2609.30476)

    该论文提出了一种基于 Whisper 的非对称无分类器引导方法，通过在推理时使用轻量级预测器逐句动态调整说话人条件化强度，在领域偏移条件下将目标说话人语音识别的词错误率相对降低高达 21.8%。

    

    目标说话人自动语音识别（TS-ASR）需要在不同的重叠和噪声条件下识别并转录目标说话人的语音。这些变化会改变语音混合中目标说话人的声学证据，因此需要在推理时对说话人条件化进行校准。我们基于 Whisper 为 TS-ASR 引入了非对称无分类器引导（CFG）：说话人条件化分支预测目标转录文本，而说话人无条件化分支预测序列化的多说话人转录文本。CFG 通过单一的引导尺度在解码过程中调整说话人条件化的贡献。我们在目标领域的开发数据上选择一个全局引导尺度，并训练一个轻量级的基于编码器的预测器来针对每条语句对其进行调整，同时保持识别模型不变。在领域偏移下，我们的完整系统相比仅条件化基线实现了高达 21.8% 的相对词错误率（WER）降低。

    arXiv:2609.30476v1 Announce Type: cross  Abstract: Target-speaker automatic speech recognition (TS-ASR) must identify and transcribe a desired speaker under varying overlap and noise conditions. These changes alter the acoustic evidence for the target speaker in the speech mixture, motivating inference-time calibration of speaker conditioning. We introduce asymmetric classifier-free guidance (CFG) for TS-ASR using Whisper: the speaker-conditioned branch predicts the target transcript, while the speaker-unconditioned branch predicts serialized multi-speaker transcripts. CFG adjusts the contribution of speaker conditioning during decoding through a single guidance scale. We select a global guidance scale on target-domain development data and train a lightweight encoder-based predictor to adjust it for each utterance, keeping the recognition model fixed. Under domain shifts, our full system achieves relative word error rate (WER) reductions of up to 21.8% over the condition-only baseline,
    
[^71]: CARGO：面向生产环境中智能体AI的上下文感知检索门控评估

    CARGO: Context-Aware Retrieval-Gated Evaluation of Agentic AI in Production

    [https://arxiv.org/abs/2609.30471](https://arxiv.org/abs/2609.30471)

    针对生产环境中智能体AI评估时“参考-实例分歧”（即参考答案描述的是不同实体导致正确回答被误判为错误）的问题，提出CARGO框架，将检索参考视为流程范例、基于实时实例上下文判定事实并将评估门控于检索置信度之上。

    

    基于参考答案的LLM-as-a-judge评估方法假定参考答案即为目标答案。在部署于动态实体（如支持工单、资产、账户）之上运行的智能体系统中，最接近的可用参考答案通常是将正确流程应用到了不同的实体上，因此字面化的裁判会将不同的标识符、日期和状态判定为错误或幻觉。我们将这种失败模式命名为“参考-实例分歧”。我们提出了CARGO框架，该框架：（i）将检索到的参考答案视为流程范例，并将事实性判断建立在实际实例的实时观测上下文之上；（ii）为每条声明分配三向状态（支持、矛盾、无法验证），并仅对矛盾进行惩罚；（iii）通过检索置信度对评估进行门控，将生产环境评估建模为选择性预测问题。我们引入了CARGO-Bench，这是一个基于扰动的诊断套件，其真值通过构造方式获得，能够区分宽容……（原文在此截断）

    arXiv:2609.30471v1 Announce Type: cross  Abstract: Reference-based LLM-as-a-judge evaluation assumes the reference answer is the target. In deployed agentic systems that operate over dynamic entities (support cases, assets, accounts), the closest available reference typically applies the correct procedure to a different entity, so a literal judge penalizes different identifiers, dates, and statuses as errors or hallucinations. We name this failure mode reference-instance divergence (RID). We propose CARGO, a framework that (i) treats retrieved references as procedural exemplars and grounds factual judgments in the live instance's observed context, (ii) assigns each claim a three-way status (supported, contradicted, unverifiable) and penalizes only contradictions, and (iii) gates evaluation by retrieval confidence, casting production evaluation as selective prediction. We introduce CARGO-Bench, a perturbation-based diagnostic suite with ground truth by construction that separates lenien
    
[^72]: 基于检索的开放式评估在何处失效？从长篇医学答案事实性验证中自动归纳错误分类体系

    Where Does Retrieval-Based Open-Ended Evaluation Fail? Automatic Taxonomy Induction from Long-Form Medical Answer Factuality Verification

    [https://arxiv.org/abs/2609.30467](https://arxiv.org/abs/2609.30467)

    该论文针对基于检索的开放式医学事实性验证，自动归纳出两套错误分类体系，将失败分解为五个质量维度上的检索阶段错误与六个连续步骤中的验证器推理错误，并利用LLM-as-Judge流程实现大规模自动化错误标注与分类，无需黄金答案或黄金证据。

    

    检索式事实性评估——即将大语言模型生成的论断与权威医学语料库中的证据进行核对验证——已成为高风险临床环境中可扩展幻觉检测的主流范式。尽管可靠且透明的医学事实验证需求迫切，大多数系统仍采用F1等聚合指标衡量性能，这类指标掩盖了失败发生在何处以及为何发生。现有的RAG诊断方法需要黄金答案或人工标注的黄金证据，而这两者在该场景中均不存在。我们基于对开放式MedExpert数据集以及3个封闭式数据集的案例研究，引入了两套全面的分类体系：将失败分解为沿五个质量维度划分的检索阶段错误，以及分为六个连续步骤的验证器推理错误。我们改造了一种基于LLM-as-Judge的自动模式归纳流程，用以大规模标注证据质量并对验证器推理错误进行分类……

    arXiv:2609.30467v1 Announce Type: new  Abstract: Retrieval-based factuality evaluation, where LLM-generated claims are verified against evidence from authoritative medical corpora, has become the dominant paradigm for scalable hallucination detection in high-stakes clinical settings. Despite the urgency of reliable and transparent medical fact verification, most systems measure performance with aggregate metrics like F1, which obscure where and why failures occur. Existing RAG diagnostics require gold answers or annotated gold evidence, neither of which exists in this regime. We introduce two comprehensive taxonomies, grounded in a case study on the open-ended MedExpert dataset and 3 closed-ended datasets, decomposing failures into retrieval-stage errors along five quality dimensions, and verifier-reasoning errors into six consecutive steps. We adapt an automatic pattern induction pipeline using LLM-as-Judge to label evidence quality and classify verifier reasoning errors at scale, and
    
[^73]: RAZOR：在大语言模型中剪枝可被替换的专家

    RAZOR: Pruning Replaceable Experts in LLMs

    [https://arxiv.org/abs/2609.30465](https://arxiv.org/abs/2609.30465)

    该论文提出无需训练的 MoE 专家剪枝方法 RAZOR，利用共识残差衡量专家的功能可替换性，在固定剪枝预算下剪除可被存活专家替代的专家，无需梯度或恢复训练即可最大程度保留原始模型输出分布。

    

    混合专家模型每个 token 只激活少量专家，但需要存储完整的专家池。专家剪枝可以减轻这种存储负担；在固定剪枝预算下，目标是尽可能保留原始模型的输出分布。然而，专家的使用频率或贡献大小本身并不能决定移除它所造成的损害，关键在于存活的计算能否替代其功能。我们提出 RAZOR，这是一种无需训练的专家剪枝方法，通过“共识残差”（即专家输出与原始加权混合输出的偏差）来评分专家的功能可替换性。该方法在固定层输入处使用精确的单删除恒等式，考虑了存活专家的重新归一化以及路由器选择的补充机制，从而在无需梯度或恢复训练的情况下，将在校准 token 上聚合得到的局部评分用于预算化剪枝。在 GLM-4.7-Flash、Qwen3.6-35B-A3B、DeepSeek-V4-Flash-0731 和 Hy3 上，在 25% 的（剪枝率下……摘要在此处截断）

    arXiv:2609.30465v1 Announce Type: cross  Abstract: Mixture-of-experts (MoE) models activate few experts per token but store the full expert pool. Expert pruning reduces this storage burden; at a fixed pruning budget, the goal is to preserve the original model's output distribution as closely as possible. Yet an expert's usage or contribution magnitude does not by itself determine the damage caused by its removal. What matters is whether the surviving computation can replace its function. We introduce RAZOR, a training-free expert pruning method that scores functional replaceability using consensus residuals: deviations of expert outputs from the original weighted mixture. An exact single-deletion identity at a fixed layer input accounts for survivor renormalization and router-selected refill, providing local scores aggregated over calibration tokens for budgeted pruning without gradients or recovery training. On GLM-4.7-Flash, Qwen3.6-35B-A3B, DeepSeek-V4-Flash-0731, and Hy3 at 25\% an
    
[^74]: 基于大语言模型的自动语音识别中的推理时目标说话人遗忘

    Inference-Time Target Speaker Unlearning in LLM-Based Automatic Speech Recognition

    [https://arxiv.org/abs/2609.30439](https://arxiv.org/abs/2609.30439)

    提出目标说话人遗忘语音识别（TSU-ASR）新任务，并设计可附加于冻结双流语音大语言模型的轻量级注册条件门控（ECG）模块，能在推理时动态阻止指定说话人的语音被转写、同时保留其说话活动的指示。

    

    我们在一个完全端到端的多说话人语音识别与说话人分离（diarization）框架中提出了目标说话人遗忘语音识别（TSU-ASR）任务。给定一段多说话人语音以及一组不希望自己的语音被转写的退出说话人（opt-out speakers），该任务要求语音识别系统转写除退出说话人之外的所有说话人的语音，同时仍能指示这些退出说话人何时处于活跃说话状态。作为解决该任务的第一步，我们提出了一种新颖的轻量级注册条件门控（Enrollment-Conditioned Gating, ECG）模块，该模块可附加到冻结的双流语音大语言模型上，使系统能够在推理阶段动态实现对新的退出说话人的遗忘，即使是那些在初始ECG训练阶段未曾见过的说话人。我们在AMI（英语）和AliMeeting（普通话）数据集上的实验表明，对应退出说话人的词或字的转写准确率分别从72.3%降至48.2%和从73.6%降至27.3%，同时保留说话人的转写错误……

    arXiv:2609.30439v1 Announce Type: new  Abstract: We introduce target-speaker unlearning ASR (TSU-ASR) task in a fully end-to-end framework for multi-speaker ASR and diarization. Given a multi-speaker utterance and a set of opt-out speakers who do not wish to have their speech transcribed, the task requires an ASR system to transcribe all speakers except the opt-out ones, while still indicating when those speakers are active. As a first step towards tackling this task, we introduce a novel, light-weight Enrollment-Conditioned Gating (ECG) module attachable to a frozen dual-stream speech LLM that enables ASR for new opt-out speakers dynamically during inference, even those who were not seen during initial ECG training phase. Our experiments on both AMI (English) and AliMeeting (Mandarin) datasets show that speech transcription accuracy for corresponding opt-out words or characters falls from 72.3% to 48.2% and from 73.6% to 27.3%, respectively, while retained speakers' transcription erro
    
[^75]: 一切皆逢其时：基于大语言模型的同声语音到语音翻译的因果感知框架

    All In Good Time: Causality-Aware Framework for LLM-Based Simultaneous Speech-to-Speech Translation

    [https://arxiv.org/abs/2609.30416](https://arxiv.org/abs/2609.30416)

    该论文提出了一种因果感知的同声语音到语音翻译框架FAST-CAP，通过分解式S2ST架构、因果感知自适应策略和因果感知延迟度量，并配合生成高保真因果对齐数据的新型数据流水线，在显著降低延迟的同时提升了翻译质量。

    

    大语言模型（LLM）在低资源离线翻译中已展现出强大的性能；然而，将其扩展到同声语音到语音翻译仍面临挑战，主要原因是缺乏具有高跨语言说话人保真度的因果对齐训练数据。此外，现有方法依赖固定的翻译策略或置信度启发式规则，导致翻译质量欠佳且延迟较高。我们提出了一种因果感知的同声语音到语音翻译框架，该框架包含一个新颖的数据流水线，能够生成高保真、因果对齐的语音片段，并改进了音色转换。该框架引入了：（i）分解式语音到语音翻译架构（FAST），（ii）因果感知自适应策略（CAP），以及（iii）因果感知延迟度量指标。在CVSS西班牙语、德语和法语数据集上的实验表明，FAST-CAP持续改善了质量与延迟之间的权衡，相比固定策略最高可提升1.2 BLEU分数，并相对降低26%的延迟。

    arXiv:2609.30416v1 Announce Type: new  Abstract: Large Language Models (LLMs) have shown strong performance in low-resource offline translation; however, extending them to simultaneous speech-to-speech translation (Simul-S2ST) remains challenging due to the scarcity of causally aligned training data with high cross-lingual speaker fidelity. In addition, existing approaches rely on fixed translation policy or confidence heuristics, leading to suboptimal quality and higher latency. We propose a causality-aware Simul-S2ST framework with a novel data pipeline that generates high-fidelity, causally aligned segments with improved voice transfer. The framework introduces (i) a factorized S2ST architecture (FAST), (ii) a causality-aware adaptive policy (CAP), and (iii) causality-aware latency metric. Experiments on CVSS Spanish, German, and French show that FAST-CAP consistently improves the quality-latency trade-off, achieving up to +1.2 BLEU and a 26% relative latency reduction over a fixed 
    
[^76]: 概念与组块的统一解释

    A Unified Account of Concepts and Chunks

    [https://arxiv.org/abs/2609.30414](https://arxiv.org/abs/2609.30414)

    本文提出将认知心理学中原本分离的概念与组块统一到一个理论框架中，扩展Cobweb分类模型并实现TRELLIS系统，成功应用于同时包含概念性和组块性元素的上下文无关文法学习。

    

    认知心理学已经研究了人们如何编码、使用和学习描述类别的概念，以及人们如何表征、识别和获取用于熟悉元素模式的组块。这两个主题的研究文献几乎互不相关，这对认知的统一理论构成了挑战。在本文中，我们回顾了Cobweb——一个关于分类和概念形成的计算模型，并提出了一个将组块及其获取过程纳入其中的扩展理论。该理论不对模态做任何假设，适用于任何可分解为元素及元素间关系的经验。我们还介绍了TRELLIS——该理论的一个实现系统，并展示了其在学习上下文无关文法方面的应用。我们选择上下文无关文法作为测试平台，是因为它们同时包含类似概念和类似组块的元素。此外，我们报告了在三个合成文法上的实验结果，证明了该系统的能力

    arXiv:2609.30414v1 Announce Type: cross  Abstract: Cognitive psychology has studied how people encode, use, and learn concepts that describe categories, and how they represent, recognize, and acquire chunks for familiar patterns of elements. The literatures on these two topics are nearly disjoint, which poses a challenge for unified theories of cognition. In this paper, we review Cobweb, a computational account of categorization and concept formation, and propose an extended theory that incorporates chunks and their acquisition. The theory makes no commitments about modality, applying to any experience that decomposes into elements and relations among them. We also present \trellis/, an implementation of this theory, and illustrate its application to learning context-free grammars, which we adopt as a testbed because they involve both concept-like and chunk-like elements. In addition, we report experimental results on three synthetic grammars that demonstrate the system's ability to re
    
[^77]: 什么能改进多模态虚假信息检测？来自大规模实证研究的答案

    What Improves Multimodal Misinformation Detection? Answers from a Large-Scale Empirical Study

    [https://arxiv.org/abs/2609.30402](https://arxiv.org/abs/2609.30402)

    本文通过涵盖3,375余次实验的大规模实证研究，系统性地回答了哪些设计选择能改进多模态虚假信息检测、它们何时会悄然失效，为构建更强大可靠的检测系统提供了实用指导。

    

    多模态虚假信息正日益被精心设计得看似令人信服，即将一段文本声明与一张看似能“证明”该声明的图像配对。然而在实践中，构建有效的检测器往往取决于一组很少受到系统性研究的设计选择。在本文中，我们针对多模态虚假信息检测的设计选择开展了一项大规模研究，进行了超过3,375次实验，涵盖三个基准数据集以及广泛的预训练视觉与语言骨干模型。通过系统性比较和针对性的鲁棒性分析，我们提炼出了实用的指导原则：哪些设计选择有帮助、它们何时会悄无声息地失效，以及流水线中的哪些方面最能强烈地影响模型行为，从而回答了4个关键研究问题（RQs）。我们旨在为设计更强大、更可靠的多模态虚假信息检测系统提供可靠的基础，从而为更广泛的研究社区做出贡献。

    arXiv:2609.30402v1 Announce Type: cross  Abstract: Multimodal misinformation is increasingly crafted to look convincing by pairing a textual claim with an image that appears to "prove" it. Yet in practice, building effective detectors often hinges on a small set of design choices that are rarely examined in a controlled way. In this paper, we conduct a large-scale study of multimodal design choices for misinformation detection with over 3,375 experiments- spanning three benchmark datasets and a broad range of pre-trained vision and language backbones. Through systematic comparisons and targeted robustness analyses, we distill practical guidance on which design choices help, when do they fail silently, and what aspects of the pipeline most strongly shape model behavior, answering 4 key Research Questions (RQs). We aim to provide a reliable foundation for designing stronger and more dependable multimodal misinformation detection systems, thus contributing to the broader research communit
    
[^78]: 多智能体代码评判器何时才真正有据可依？两种无需标签的度量方法，以及一个拒绝猜测的评判器

    When Is a Multi-Agent Code Judge Actually Grounded? Two Label-Free Measurements, and a Judge That Declines to Guess

    [https://arxiv.org/abs/2609.30328](https://arxiv.org/abs/2609.30328)

    该论文指出多智能体验证框架在代码评判中因证据无法满足“独立于答案且能区分候选解”这两个条件而失效（在78%–95%的比较中判定两个解同样好、准确率仅4.4%），据此提出两种无需标签的度量方法来检验评判是否真正有据可依，并设计了一个在证据不足时拒绝猜测的评判器。

    

    当一个语言模型评判另一个语言模型的代码是否正确时，它并不会报告证据的缺失。它会返回一个附带推理过程的、充满自信的判决，这与一个真正有据可依的判决难以区分。多智能体验证将判断分解为可核查的声明，并逐条对照证据加以验证，是一种颇有前景的应对方式，且在证据为一组检索文档时效果良好。我们认为这类方法对其证据有两个要求：证据必须独立于被评审的答案，并且必须在被比较的两个候选答案之间有所区分。第二个条件在检索文档场景下会自动满足，但在代码评判中不再成立。我们在两个代码评判基准上未经修改地运行已发表的框架MARCH，进行了80项按条件逐单元的测量，发现它在78%到95%的比较中判定两个解同样好，而在直接询问同一模型时准确率仅为4.4%。

    arXiv:2609.30328v1 Announce Type: new  Abstract: When one language model judges whether another's code is correct, it does not report the absence of evidence. It returns a confident verdict with reasoning attached, indistinguishable from a verdict it had grounds for. Multi-agent verification, which decomposes a judgment into checkable claims and verifies each against evidence, is a promising response and works well when the evidence is a set of retrieved documents.   We argue such methods require two things of their evidence: it must be independent of the answer under review, and it must differ between the two candidates being compared. The second condition holds automatically with retrieved documents and stops holding in code judging.   Running MARCH, a published framework unmodified over 80 condition-by-cell measurements on two code judging benchmarks, we find it declares both solutions equally good on 78 to 95% of comparisons, reaching 4.4% accuracy where the same model asked direct
    
[^79]: PALM：面向金融语言模型的时点自适应方法

    PALM: Point-in-Time Adaptation for Financial Language Models

    [https://arxiv.org/abs/2609.30316](https://arxiv.org/abs/2609.30316)

    本文通过对比实验发现金融时点语言模型每年完整预训练并非必要——新旧检查点在同一评估窗口上表现相当，据此提出PALM方法，仅需在新增文本上拟合低秩适配器即可实现时点自适应，从而大幅降低维护时点语言模型的成本。

    

    arXiv:2609.30316v1 公告类型：交叉 摘要：用于金融回测的语言模型存在前视偏差，因为在研究期之后发布的文本上训练的模型已经观察到了它被要求去预测的结果。为解决这一问题，时点语言模型在按时间筛选的语料库上进行预训练，并按每个日历年发布一个检查点，每个检查点均有记录在案的截止日期。然而，每增加一年都需要一次完整的预训练运行，而这种运行是否必要从未被验证过。在本文中，我们证明年度预训练运行并非必要。我们转而将每个检查点与取代它的更新检查点进行比较，发现更新的检查点在同一评估窗口上并未取得更好的分数。受这一观察的启发，我们提出了PALM（面向金融语言模型的时点自适应），这是一种简单而有效的年度预训练替代方案，它在截止日期之前发布的文本上拟合低秩适配器……（摘要在此处截断）

    arXiv:2609.30316v1 Announce Type: cross  Abstract: Language models used in financial backtests suffer from look-ahead bias, as a model trained on text published after the study period has already observed the outcomes it is asked to predict. To handle this issue, point-in-time (PIT) language models are pretrained on chronologically filtered corpora and released as one checkpoint per calendar year, each with a documented cutoff. However, each additional year costs a full pretraining run, and whether that run is necessary has never been tested. In this paper, we show that the annual pretraining run is not necessary. We instead compare each checkpoint against the newer one that replaced it, and find that the newer checkpoint scores no better on the same evaluation window. Motivated by this observation, we propose PALM (Point-in-time Adaptation for financial Language Models), a simple yet effective alternative to annual pretraining that fits a low-rank adapter on text published before the 
    
[^80]: 系统综述中筛选自动化的基准框架

    A Benchmark Framework for Screening Automation in Systematic Reviews

    [https://arxiv.org/abs/2609.30298](https://arxiv.org/abs/2609.30298)

    该论文提出了包含45,064条标注条目的基准数据集SRBench、考虑类别不平衡的评估框架以及PromptSR实验工具，用于评估和支持大语言模型在系统综述筛选自动化中的应用。

    

    系统综述（SR）对循证研究至关重要，但其筛选阶段非常耗时且劳动密集。大语言模型（LLM）通过辅助文章相关性分类，为减轻这一工作负担提供了有前景的机会。然而，现有的评估方法通常依赖传统指标，而这些指标对于高度不平衡的系统综述筛选数据集可能具有误导性。本文提出了一个包含45,064条标注条目的基准数据集，用于评估大语言模型在32项精心筛选的二次研究中的系统综述筛选性能。论文提出了一个考虑类别不平衡问题的评估框架，即系统综述中被排除文章相对于被纳入文章的自然分布不均衡现象。此外，论文还介绍了PromptSR工具，该工具旨在支持基于大语言模型的筛选中的提示词实验、实验管理和结果分析。作者还展示了一个用例，演示了SRBench和PromptSR的应用。

    arXiv:2609.30298v1 Announce Type: new  Abstract: Systematic reviews (SR) are essential for evidence-based research, but their screening phase is highly time-consuming and labor-intensive. Large language models (LLMs) offer a promising opportunity to reduce this workload by assisting with article relevance classification. However, existing evaluation approaches often rely on traditional metrics that may be misleading for highly imbalanced SR screening datasets.This paper presents a benchmark dataset of $45\,064$ labeled entries for evaluating LLM performance in SR screening across 32 curated secondary studies. It proposes an evaluation framework that accounts for class imbalance, i.e., the natural prevalence of excluded articles relative to included articles in SRs. It also introduces PromptSR, a tool designed to support prompt experimentation, experiment management, and result analysis for LLM-based screening. We also present a use case demonstrating the application of SRBench and Prom
    
[^81]: 在Spotify引导对话式推荐智能体：合成数据生成与自我改进循环

    Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops

    [https://arxiv.org/abs/2609.30297](https://arxiv.org/abs/2609.30297)

    Spotify提出了一条多轮合成数据生成流水线与自我改进循环，通过基于方差的对比优化和编码智能体的迭代修复，在冷启动场景下自动优化对话式推荐智能体的规划与工具调用能力，使质量提升8%。

    

    对话式推荐智能体是内容发现的新范式，使用户能够通过自然语言表达复杂意图（例如：“推荐一些我没听过的意大利独立音乐人”）。构建此类智能体的核心挑战在于优化智能体规划——即决定如何选择、排序和调用工具——尤其是在真实用户交互尚不可用的冷启动场景中。为应对这一挑战，我们提出了一条多轮合成数据生成流水线和一种自我改进循环。合成数据流水线将单轮提示转化为逼真的多轮对话，从而支持上线前的系统性评估。自我改进循环将基于方差的对比优化与通过编码智能体进行的迭代改进相结合，能够自动识别并修复规划和工具使用中的错误。我们的方法在高度优化的基线之上将质量提升了+8%。

    arXiv:2609.30297v1 Announce Type: cross  Abstract: Conversational recommendation agents are a new paradigm for content discovery, enabling users to express complex intents through natural language (e.g., "recommend Italian indie artists I haven't heard before"). A central challenge in building such agents is optimizing agent planning -- deciding how to select, sequence, and invoke tools -- particularly in cold-start settings where real user interactions are not yet available. We introduce a pipeline for multi-turn synthetic data generation and a self-improvement loop to address this challenge. The synthetic data pipeline transforms single-turn prompts into realistic multi-turn conversations, enabling systematic evaluation before launch. The self-improvement loop combines variance-based contrastive optimization with iterative refinement through a coding agent, automatically identifying and fixing planning and tool-use errors. Our approach improves quality by +8% on top of a highly optim
    
[^82]: SignTrace：描述一个手势，找到对应的词

    SignTrace: Describe a Sign, Find the Word

    [https://arxiv.org/abs/2609.30295](https://arxiv.org/abs/2609.30295)

    SignTrace 利用大语言模型增强的中国手语词典，结合动作提取、七路检索与候选重排序，让学习者仅凭日常语言描述的手部动作即可反向查找到对应手语词条及其含义，在 500 条查询的基准上达到 94.0% 的 Hit@1。

    

    当学习者记得一个陌生手语的动作，却不知道其含义或正式的特征编码时，识别该手语十分困难。SignTrace 通过自然语言方式访问中国手语词典，解决了这一长期存在的反向查询问题。该系统集成了基于大语言模型的词典增强、动作提取、词典风格改写、七路检索以及候选重排序，覆盖 6,699 个词条。系统已部署进行用户试用，并收到了积极的非正式反馈。在一个由词典衍生、包含 500 条动作描述查询的基准测试上，系统取得了 94.0% 的 Hit@1、97.4% 的 Hit@9 以及 0.9540 的平均倒数排名。重排序将 Hit@1 从 71.8% 提升至 94.0%，组件分析显示了增强词条描述所作的贡献。在六个并发查询的情况下，查询处理的中位时间为 13.37 秒。通过将日常动作描述与已记录的手语动作及其含义关联起来……

    arXiv:2609.30295v1 Announce Type: cross  Abstract: Identifying an unfamiliar sign is difficult when a learner remembers its movement but does not know its meaning or formal feature codes. SignTrace addresses this longstanding reverse-lookup problem through natural-language access to a Chinese sign-language dictionary. The system integrates LLM-based dictionary enrichment, action extraction, dictionary-style rewriting, seven-channel retrieval, and candidate reranking over 6,699 entries. It has been deployed for user trials and has received positive informal feedback. Evaluation on a dictionary-derived benchmark of 500 movement-description queries yields 94.0% Hit@1, 97.4% Hit@9, and a mean reciprocal rank of 0.9540. Reranking increases Hit@1 from 71.8% to 94.0%, while component analyses show the contribution of enriched entry descriptions. Median query-processing time is 13.37 seconds with six concurrent queries. By connecting everyday movement descriptions to documented signs and meani
    
[^83]: SlideLab：以观众为中心的科学幻灯片生成与评估

    SlideLab: Audience-Centered Scientific Slide Generation and Evaluation

    [https://arxiv.org/abs/2609.30294](https://arxiv.org/abs/2609.30294)

    SlideLab 是一个无需训练的多智能体框架，能从研究论文生成以观众为中心的科学演示幻灯片，在盲测中于 77% 的论文上超越开源与商业系统且推理成本降低约 4 倍，并配套提出模拟会议室的观众导向评估框架 ConfArena。

    

    科学演示不仅仅是研究论文的摘要，它们需要以连贯的顺序呈现研究工作，清晰地解释核心思想，并帮助观众跟上演讲的节奏。我们提出了 SlideLab，一个无需训练的多智能体框架，用于从研究论文生成科学演示幻灯片。SlideLab 首先规划演示叙事，然后利用内容规划、视觉生成、布局优化和依据验证等智能体，构建并迭代完善一套共享幻灯片。在一项盲测人类偏好研究中，SlideLab 在 77% 的论文上优于开源和商业系统，同时其推理 token 使用量仅为最强开源基线的约四分之一。我们还引入了 ConfArena，一个面向观众的评估框架，它模拟会议室场景并逐张幻灯片地评估演示效果。ConfArena 的评估结果与人类对系统的排名相一致，并能检测注入的（原文在此处截断）……

    arXiv:2609.30294v1 Announce Type: cross  Abstract: Scientific presentations are more than summaries of research papers. They need to present the work in a coherent sequence, explain the main ideas clearly, and help the audience follow the presentation. We present SlideLab, a training-free multi-agent framework for generating scientific presentations from research papers. SlideLab first plans the presentation narrative, then builds and iteratively refines a shared slide deck using agents for content planning, visual generation, layout refinement, and grounding verification. In a blind human preference study, SlideLab was preferred over both open-source and commercial systems on 77% of papers while using roughly 4 times fewer inference tokens than the strongest open-source baseline. We also introduce ConfArena, an audience-oriented evaluation framework that simulates a conference room and assesses presentations slide by slide. ConfArena matches human system rankings and detects injected 
    
[^84]: Cartograph：面向AI智能体的操作者证明检索式联邦工具发现

    Cartograph: Federated Tool Discovery with Operator-Attested Retrieval for AI Agents

    [https://arxiv.org/abs/2609.30293](https://arxiv.org/abs/2609.30293)

    Cartograph是一个联邦式MCP代理，通过操作者签名的能力卡片、三层易混淆聚类分析（Rift）以及“先服务器后工具”的两阶段检索，将AI智能体的工具发现从O(n)目录遍历优化为O(k)渐进式披露，仅需暴露3个代理工具即可在374个工具的部署中取得0.816的R@5召回率，显著优于关键词基线。

    

    模型上下文协议（MCP）使AI智能体能够发现和调用工具，但随着所连接目录规模的增大，加载每一个工具定义的代价变得十分高昂。我们提出了Cartograph，一个联邦式MCP代理，它将智能体可见的工具发现从O(n)的目录遍历转变为O(k)的渐进式披露。Cartograph结合了三种机制：(1) 操作者证明的能力卡片，即在部署操作者控制下生成的Ed25519签名描述，而非由发布方营销文案排序的描述；(2) Rift，一个三层的易混淆聚类分析，包括密度聚类、查询边际分析和词元诊断；(3) 两阶段检索，先对服务器排序再对工具排序。在一个包含22个服务器、374个工具的部署中，Cartograph仅暴露3个代理工具而非374个工具定义。在一个由作者构建的49个查询的基准测试中，Cartograph取得了0.816的R@5召回率，相比之下Jaccard关键词基线仅为0.592，同时实测的前5名工具发现交换仅使用47（原文在此处截断）

    arXiv:2609.30293v1 Announce Type: cross  Abstract: The Model Context Protocol (MCP) enables AI agents to discover and call tools, but loading every definition becomes expensive as connected catalogs grow. We present Cartograph, a federated MCP proxy that changes agent-visible tool discovery from $O(n)$ catalog traversal to $O(k)$ progressive disclosure. Cartograph combines three mechanisms: (1) operator-attested capability cards, Ed25519-signed descriptions generated under the deploying operator's control rather than ranked publisher copy; (2) Rift, a three-layer confusable-cluster analysis comprising density clustering, query-margin analysis, and token diagnosis; and (3) two-stage retrieval, which ranks servers before tools. On a 22-server, 374-tool deployment, Cartograph exposes three proxy tools instead of 374 definitions. A 49-query author-constructed benchmark yields R@5 of 0.816, compared with 0.592 for a Jaccard keyword baseline, while a measured top-5 discovery exchange uses 47
    
[^85]: 虚假评论检测综述：从预训练语言模型到大语言模型

    A Survey on Fake Review Detection: From Pre-trained Language Models to Large Language Models

    [https://arxiv.org/abs/2609.30292](https://arxiv.org/abs/2609.30292)

    本综述从信息融合视角系统梳理了2018年至2026年初的211项虚假评论检测研究，按证据来源和融合层次组织现有工作，并分析了预训练语言模型和大语言模型对虚假评论的生成与检测带来的双重影响。

    

    在线评论影响着消费者的决策、平台治理和企业声誉。虚假评论通过向评分系统、推荐渠道和公众信任机制中注入欺骗性证据，破坏了这一信息渠道。大语言模型（LLM）的兴起从两个方向改变了这一问题：LLM能够生成流畅且具有上下文感知能力的欺骗性评论，而预训练语言模型（PLM）和LLM同时也为检测任务提供了更强的语义表示能力。本综述从信息融合的视角回顾了虚假评论检测研究，涵盖了2018年至2026年初发表的211项研究。我们按照证据来源和融合层次对现有工作进行组织，涵盖评论文本、情感、评分行为、时间元数据、用户-产品图、多模态内容、外部知识以及LLM生成的信号。我们追溯了从传统机器学习和深度学习到基于PLM的方法的发展历程。

    arXiv:2609.30292v1 Announce Type: cross  Abstract: Online reviews shape consumer decisions, platform governance, and corporate reputation.Fake reviews compromise this information channel by injecting deceptive evidence into rating systems, recommendation pipelines, and public trust mechanisms.The rise of large language models, or LLMs, has changed the problem in two directions.LLMs can generate fluent and context-aware deceptive reviews, while pre-trained language models, or PLMs, and LLMs also provide stronger semantic representations for detection.This survey reviews fake review detection from an information fusion perspective, covering 211 studies published from 2018 to early 2026.We organize existing work by evidence source and fusion level, covering review text, sentiment, rating behavior, temporal metadata, user-product graphs, multimodal content, external knowledge, and LLM-generated signals.We trace the development from traditional machine learning and deep learning to PLM-base
    
[^86]: 审计与修复生产级文本到SQL流水线中“LLM作为裁判”的失效问题

    Auditing and Repairing LLM-as-Judge Failures in a Production Text-to-SQL Pipeline

    [https://arxiv.org/abs/2609.30290](https://arxiv.org/abs/2609.30290)

    本文审计了生产级文本到SQL流水线中的LLM裁判，发现其与人工标注一致性极低且根源于“评分幻觉”这一单一机制，并提出用低成本自托管Qwen模型替换裁判、配合三个强裁判的一致同意集成（kappa = 0.79、自动覆盖率 89.7%）来有效修复该失效问题。

    

    生产级文本到SQL流水线通常以一个“LLM作为裁判”（LLM-as-judge）收尾，但其与人工标注者的一致性却从未被真正测量过。当我们检查自己的系统时，发现已部署的 gpt-4o-mini 裁判与双人金标准的一致性在富含分歧样本的集合上 Cohen's kappa 仅为 0.04，在均匀随机抽检集合上为 0.42，在富集集合中对 77.1% 的人类判定为 FAITHFUL 的案例进行了过度标记。其大部分过度标记可追溯到一个我们称为 GRADE-HALLUCINATION（评分幻觉）的单一机制。一个自托管的 Qwen3.6-27B 替代方案（kappa = 0.72）达到了与 Claude Opus 4.7（kappa = 0.71）相当的水平；该头对头比较在 n = 96 时统计功效不足，但对于部署决策而言这几乎无关紧要，因为 Qwen 每次调用的成本约为前者的 1/300。集成并不能免费带来提升：将弱裁判与强裁判配对反而会降低一致性，而三个强裁判在一致同意路由下可达到 kappa = 0.79，自动覆盖率 89.7%。将该方案应用于域外数据时……（原文摘要在此处截断）

    arXiv:2609.30290v1 Announce Type: new  Abstract: Production text-to-SQL pipelines often end with an LLM-as-judge whose agreement with human annotators has never actually been measured. When we checked ours, the deployed gpt-4o-mini judge agreed with two-author gold at only Cohen's kappa = 0.04 on a disagreement-enriched set and 0.42 on a uniform-random spot-check, over-flagging 77.1% of the human-FAITHFUL cases in the enriched set. Most of its over-flags trace back to a single mechanism we call GRADE-HALLUCINATION. A self-hosted Qwen3.6-27B replacement (kappa = 0.72) lands in the same range as Claude Opus 4.7 (kappa = 0.71); the head-to-head is underpowered at n = 96, but for the deployment decision that hardly matters, since Qwen costs roughly 1/300 as much per call. Ensembling does not help for free. Pairing the weak judge with a stronger one degrades agreement, whereas three strong judges under unanimity routing reach kappa = 0.79 at 89.7% auto-coverage. Applied out-of-domain, the s
    
[^87]: 并非所有记忆都同等重要：面向LLM智能体有效性感知检索的层级协同记忆

    Not All Memories Are Equal: Hierarchical Collaborative Memory for Validity-Aware Retrieval in LLM Agents

    [https://arxiv.org/abs/2609.30289](https://arxiv.org/abs/2609.30289)

    提出HiCoMER框架，通过层级化协同记忆管理与有效性感知检索，避免协作式LLM智能体检索到过时或冲突的记忆，确保回答基于当前有效信息。

    

    在团队协作场景中，记忆是异构且持续演化的。团队记忆记录集体决策、协议和当前共识，而个体记忆则保存成员特定的观察、执行轨迹和中间进展。现有的记忆增强系统通常将所有存储的记忆视为一个扁平的池子进行检索，仅根据语义相关性、重要性或时间新近度进行排序，而不建模层级结构或不断演化的有效性。因此，它们经常检索出语义相关但已过时或存在冲突的记忆，尤其是那些不再与当前团队共识相符的个体记忆，而非优先呈现当前有效的记忆。当协作式LLM智能体回答用户问题时，这一问题尤为突出，因为其回答应当基于有效的记忆。我们提出了HiCoMER，一个用于层级协同记忆管理与有效性感知检索的框架。

    arXiv:2609.30289v1 Announce Type: new  Abstract: In team collaboration scenarios, memory is heterogeneous and continually evolving. Team memories capture collective decisions, protocols, and current consensus, while individual memories preserve member-specific observations, execution traces, and intermediate progress. Existing memory-augmented systems typically retrieve from all stored memories as a flat pool, ranking them by semantic relevance, importance, or recency without modeling hierarchical structure or evolving validity. As a result, they often surface semantically relevant but outdated or conflicting memories, especially individual memories that no longer align with current team consensus, instead of prioritizing currently valid memories. This is particularly problematic when collaborative LLM agents answer user questions, since their responses should be grounded in valid memories. We propose HiCoMER, a framework for hierarchical collaborative memory management and validity-aw
    
[^88]: 用于掩码语言建模的流形投影与迭代自编码器细化

    Manifold Projection and Iterative Autoencoder Refinement for Masked Language Modeling

    [https://arxiv.org/abs/2609.30288](https://arxiv.org/abs/2609.30288)

    该论文提出用低秩瓶颈自编码器混合模块（分别在局部邻域、全序列和注意力头间运作）替代注意力机制，并在掩码位置引入由“拉动”和“校正”两步组成的迭代细化程序，用于掩码语言建模。

    

    在基于Transformer的掩码语言模型中，注意力是上下文混合的主要机制，但也存在其他跨词元混合数据的方式。近期的无注意力混合器用固定的或由超网络生成的MLP替代注意力，为追求计算简洁而放弃了其动态的、依赖内容的加权方式。我们构建了一种通过低秩瓶颈自编码器获得同样特性的替代方案。我们用一组基于自编码器的混合模块替代注意力：一个在局部邻域上运行，一个在整个序列上运行，还有一个在注意力头之间运行，每个模块都通过瓶颈压缩并重构其输入，且其宽度是一个超参数而非训练产生的效果。在掩码位置，我们引入了一个包含两个不同步骤的迭代细化程序：一个“拉动”步骤，将嵌入表示拉向其邻居的加权平均值；以及一个“校正”步骤，将表示投影……（原文摘要在此处截断）

    arXiv:2609.30288v1 Announce Type: new  Abstract: In Transformer-based masked language models, attention is the primary mechanism for context mixing, but there are other ways to mix data across tokens. Recent attention-free mixers replace attention with fixed or hypernetwork-generated MLPs, alternating their dynamic, content-dependent weighting for computational simplicity. We build an alternative that gets the same property from a low-rank bottleneck autoencoder. We replace attention with a stack of autoencoder-based mixing modules, one operating over local neighborhoods, one over the full sequence, and one across attention heads, each compressing and reconstructing its input through a bottleneck, and its width is a hyperparameter rather than a training effect. In masked positions, we introduce an iterative refinement procedure that has two distinct steps. A pulling step that pulls an embedding representation toward a weighted average of its neighbors, and a correcting step that projec
    
[^89]: 冻结BERT中AI文本检测神经元的机制研究：基于RAID的稀疏探测与激活修补

    A Mechanistic Study of AI-Text Detection Neurons in Frozen BERT: Sparse Probing and Activation Patching on RAID

    [https://arxiv.org/abs/2609.30287](https://arxiv.org/abs/2609.30287)

    该研究通过稀疏探测和双向激活修补方法，在冻结的BERT中定位出一组占比不足1%的神经元，证明它们因果性地支持跨六个生成器的AI生成文本检测任务。

    

    AI生成文本检测器在标准基准测试中达到了很高的准确率，但驱动这些预测的内部表示仍鲜为人知。我们研究了冻结的BERT-base-uncased编码器中哪些神经元支持AI文本检测，使用了涵盖六个生成器（包括纯基座模型和指令微调模型）的RAID基准。我们将Gurnee等人（2023）提出的L1到L2稀疏探测协议应用于全部9,216个CLS隐藏状态维度（12层×768），我们将其称为神经元。该程序为每个生成器恢复出一组稳定的、占比不足1%的神经元，且在不同数据折和随机种子之间保持一致；仅使用该神经元集合的探测器即可保留全特征检测准确率的大部分。双向激活修补证实了该神经元集合的因果相关性：在两个方向上，它翻转预测的频率比大小匹配的随机神经元集合高一个数量级。而对相同神经元进行均值消融后，准确率基本保持不变；该信号是……（摘要在此处截断）

    arXiv:2609.30287v1 Announce Type: cross  Abstract: AI-generated text detectors achieve high accuracy on standard benchmarks, yet the internal representations that drive these predictions remain poorly understood. We study which neurons in a frozen BERT-base-uncased encoder support AI-text detection, using the RAID benchmark across six generators spanning pure-base and instruction-tuned models. We apply the L1-to-L2 sparse-probing protocol of Gurnee et al. (2023) to all 9,216 CLS hidden-state dimensions (12 layers x 768), which we call neurons. The procedure recovers a stable set of under 1% of neurons per generator, consistent across folds and seeds; a probe restricted to that set retains most of the full-feature detection accuracy. Bidirectional activation patching confirms this set's causal relevance: in both directions it flips predictions an order of magnitude more often than size-matched random sets. Mean-ablating the same neurons leaves accuracy largely intact; the signal is ther
    
[^90]: PrivDrift：主动LLM对话中话题漂移下用户秘密泄露的审计

    PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

    [https://arxiv.org/abs/2609.30094](https://arxiv.org/abs/2609.30094)

    提出PrivDrift审计基准，发现在LLM活跃对话中，用户披露的秘密即使经历话题漂移后仍高度可恢复（混合泄露率达38.7%–54.6%），且额外的话题漂移并不能可靠降低泄露风险。

    

    arXiv:2609.30094v1 公告类型：新论文 摘要：大型语言模型日益作为持久性助手应用于面向用户的、共享会话以及工具增强的场景中。当用户在活跃对话中披露敏感信息时，即使对话随后转向无关话题，这些信息通过后续提示仍可能在行为层面被恢复出来。我们提出了PrivDrift，这是一个用于审计用户所披露秘密在经历对话话题漂移和基于说服的探测之后是否仍可被恢复的基准。PrivDrift包含1,000个受控多轮对话，其中预置了秘密信息、内容密集的漂移轮次以及标准化的提取探测。在三个具有扩展上下文窗口的大语言模型上，对话级混合泄露依然严重，泄露率介于38.7%至54.6%之间，并且随模型、秘密类型和说服强度的不同而有显著变化。在所测试的漂移窗口内，额外的话题漂移并不能可靠地降低泄露，这表明隐私风险持续存在（摘要在此处被截断）。

    arXiv:2609.30094v1 Announce Type: new  Abstract: Large language models increasingly operate as persistent assistants in user-facing, shared-session, and tool-augmented settings. When users disclose sensitive information during an active conversation, that information may remain behaviorally recoverable through later prompts even after the dialogue shifts to unrelated topics. We introduce \textbf{PrivDrift}, a benchmark for auditing whether user-disclosed secrets remain recoverable after conversational topic drift and persuasion-based probing. PrivDrift contains 1{,}000 controlled multi-turn dialogues with seeded secrets, content-dense drift turns, and standardized extraction probes. Across three LLMs with extended context windows, dialogue-level hybrid leakage remains substantial, ranging from 38.7\% to 54.6\%, and varies strongly by model, secret type, and persuasion intensity. Within the tested drift window, additional topic drift does not reliably reduce leakage, suggesting that pri
    
[^91]: PUBG Ally：作为AI队友的对话式具身智能体

    PUBG Ally: A Conversational Embodied Agent as an AI Teammate

    [https://arxiv.org/abs/2609.29837](https://arxiv.org/abs/2609.29837)

    该论文提出了PUBG Ally，一个面向《绝地求生》的语音对话式具身AI队友，通过将语言模型智能体的工具使用与实时游戏控制相结合，在严格延迟约束下感知动态游戏世界、与玩家自然交流并同步执行移动、战斗等游戏行动。

    

    我们推出了PUBG Ally，一个面向《绝地求生：大逃杀》(PUBG: BATTLEGROUNDS) 的具身智能体，它能够进行推理、自主行动，并作为支持语音交互的队友与玩家并肩作战。构建这样的队友需要结合两种困难的能力：它必须在严格的延迟约束下感知并响应不断变化的游戏世界，同时与玩家自然交互，使其语音与行动保持同步。因此，Ally将智能体的工具使用能力与实时游戏控制相结合。一个语言模型智能体通过受控接口来查看游戏信息、理解玩家语音、维护上下文、决定说什么，并发出高层动作选择，用以引导更快的控制层执行移动、战斗和恢复等操作。由于玩家和Ally的语音与行动会不断相互影响并塑造比赛进程，训练需要来自真实对局的数据。因此，我们在近3.9万场对局中收集了数据……（摘要在此处被截断）

    arXiv:2609.29837v1 Announce Type: new  Abstract: We introduce PUBG Ally, an embodied agent for PUBG: BATTLEGROUNDS that can reason, act autonomously, and play alongside players as a voice-enabled teammate. Building such a teammate requires combining two difficult capabilities: it must perceive and respond to a constantly changing game world under strict latency constraints while interacting naturally with players, keeping its speech synchronized with its actions. Ally therefore combines agentic tool use with real-time game control. A language-model agent uses a controlled interface to inspect game information, interpret player speech, maintain context, decide what to say, and issue high-level action choices that steer a faster control layer for movement, combat, and recovery. Because the player's and Ally's speech and actions continually shape each other and the course of the match, training requires data from actual gameplay. We therefore collect data across nearly 39k sessions in whi
    
[^92]: Rufus-Air：一个开放的大语言模型后训练方案

    Rufus-Air: An Open LLM Post-Training Recipe

    [https://arxiv.org/abs/2609.29421](https://arxiv.org/abs/2609.29421)

    本文提出了Rufus-Air，一个基于GLM-4.5-Air-Base的开放可复现的八阶段大模型后训练方案，并总结出高质量SFT奠定能力基础、难度过滤保持RL学习有效性、奖励可靠性指导阶段排序等关键实践经验。

    

    Rufus-Air 是一个在 GLM-4.5-Air-Base（106B-A12B）上构建的开放且可复现的后训练方案，由八个阶段的串行流水线组成：SFT（监督微调）、推理 RL、编码 RL、指令遵循 RL、通用智能体、编码智能体、搜索智能体和 RLHF。我们记录了复现该方案所需的数据、奖励设计、基础设施、阶段顺序以及各阶段的结果。各阶段从基础能力逐步推进到高级能力，奖励信号也从严格可验证的奖励过渡到较为柔和的基于评判者的信号。训练基于开源组件和公开数据，其中大部分数据按原样使用，无需新的人工标注或内部蒸馏教师模型。我们的主要发现是：(i) 多样化、高质量的 SFT 奠定了坚实的能力基础；(ii) 难度过滤可将 RL 提示保持在有效的学习区间内；(iii) 奖励可靠性为阶段排序提供了实用原则；(iv) 基础设施与工程选择是……

    arXiv:2609.29421v1 Announce Type: cross  Abstract: Rufus-Air is an open and reproducible post-training recipe on GLM-4.5-Air-Base (106B-A12B), organized as a serial pipeline of eight stages: SFT, Reasoning RL, Coding RL, Instruction-Following RL, General Agent, Coding Agent, Search Agent, and RLHF. We document the data, reward design, infrastructure, stage order, and stagewise results needed to reproduce the recipe. Stages progress from basic to advanced capabilities and from hard, verifiable rewards to softer judge-based signals. Training builds on open-source components and public data, much of it used as released, without new human annotation or an in-house distillation teacher. Our main findings are that (i) diverse, high-quality SFT establishes a strong capability floor; (ii) difficulty filtering keeps RL prompts within a productive learning range; (iii) reward reliability provides a practical principle for ordering stages; and (iv) infrastructure and engineering choices are part 
    
[^93]: 大语言模型中的似然排序无法像提示那样随规模扩展

    Likelihood Ranking doesn't Scale Like Prompting in LLMs

    [https://arxiv.org/abs/2609.29390](https://arxiv.org/abs/2609.29390)

    该研究通过对95个模型和10个数据集的分析发现，基于陈述性语句的似然排序准确率不随模型规模和指令微调显著提升，而提示回答则随规模急剧改善，揭示了两种评估协议之间的系统性分歧。

    

    大语言模型（LLM）的评估通常通过两种方式进行：一是提示模型生成答案，二是使用基于似然的指标对候选输出进行评分。然而，在多项选择问答（MCQA）中，标准的基于似然的评分仍然以问题和答案集合为条件，因此可以利用与提示方法相同的任务条件化答案选择接口。我们研究了一种互补的评估协议，该协议基于从相同问题—答案对构建的陈述性语句的似然排序。通过对95个参数量从0.1B到104B的仅解码器模型以及10个MCQA数据集的研究，我们发现陈述性语句似然排序与提示回答之间存在系统性差异。语句似然准确率随模型规模的变化保持相对稳定，而提示回答的准确率则随规模扩大和指令微调而显著提升。这些结果表明，对受控陈述性选项的似然偏好与任务条件化的提示回答（原文在此处截断）

    arXiv:2609.29390v1 Announce Type: new  Abstract: LLM evaluation is commonly performed either by prompting models to produce answers or by scoring candidate outputs with likelihood-based metrics. In multiple-choice QA, however, standard likelihood-based scoring is still conditioned on the question and answer set, and can therefore leverage the same task-conditioned answer-selection interface used in prompting. We study a complementary protocol based on likelihood ranking of declarative statements constructed from the same question--answer pairs. Across 95 decoder-only models, ranging from 0.1B to 104B parameters, and 10 MCQA datasets, we find a systematic divergence between declarative-statement likelihood ranking and prompted answering. Statement-likelihood accuracy remains comparatively stable across scale, whereas prompted answering improves sharply with scale and instruction-tuning. These results suggest that likelihood preferences over controlled declarative alternatives and task-c
    
[^94]: ArGuard共享任务：阿拉伯语表情包与大语言模型提示中的有害内容检测

    ArGuard Shared Task: Harmful Content Detection in Arabic Memes and LLM Prompts

    [https://arxiv.org/abs/2609.29349](https://arxiv.org/abs/2609.29349)

    ArGuard共享任务为阿拉伯语表情包多模态仇恨检测与LLM有害提示检测建立了评测基准，吸引35支队伍参赛，最佳系统在四个子任务上取得0.419至0.984不等的宏F1分数，其中细粒度表情包分类因标签稀疏和分布偏移而最具挑战性。

    

    ArGuard是一个针对阿拉伯语表情包和大语言模型（LLM）提示中有害内容检测的共享任务。该任务包含两个赛道：赛道A专注于阿拉伯语表情包中的多模态仇恨内容检测，赛道B则面向阿拉伯语LLM安全评估的有害提示检测。共有58支队伍报名，35支队伍参加了最终评估，27支队伍提交了系统描述论文。参赛队伍探索了AraBERT、Jais和Qwen3-VL等模型。最佳系统在A1、A2、B1和B2四个子任务上分别取得了0.823、0.419、0.984和0.790的宏F1分数。其中A2赛道中细粒度的表情包分类是最具挑战性的设置，部分原因在于标签稀疏以及训练集与测试集之间的分布偏移。

    arXiv:2609.29349v1 Announce Type: cross  Abstract: ArGuard is a shared task on harmful content detection in Arabic memes and LLM prompts. It includes two tracks: Track A focuses on multimodal hate detection in Arabic memes, while Track B addresses harmful prompt detection for Arabic LLM safety evaluation. In total, 58 teams registered, 35 participated in the final evaluation, and 27 submitted system-description papers. Participating teams explored models such as AraBERT, Jais, and Qwen3-VL. The best systems achieved macro-F1 scores of 0.823 on A1, 0.419 on A2, 0.984 on B1, and 0.790 on B2. Fine-grained meme classification in A2 was the most challenging setting, partly due to sparse labels and train-test distribution shifts.
    
[^95]: 超越重叠：估计基准测试暴露的因果效应

    Beyond Overlap: Estimating the Causal Effect of Benchmark Exposure

    [https://arxiv.org/abs/2609.27176](https://arxiv.org/abs/2609.27176)

    提出LeakScale干预框架，通过控制基准家族私有信息的暴露并构建全新可执行任务，首次因果量化了训练数据污染对模型评估准确率的实际影响，发现暴露使准确率提升7.17至27.31个百分点。

    

    arXiv:2609.27176v1 公告类型：新论文 摘要：评估材料进入训练的证据并不能揭示这些材料对评估结果产生了多大影响。这一区分使得被污染的基准测试分数难以解读：来源追踪可以证实接触行为，但只有反事实分析才能量化可归因于这种接触的性能。我们提出了LeakScale，一个用于估计这一缺失数量的干预性框架。LeakScale创建全新的可执行任务，这些任务需要私有的、家族特定的信息（这些信息在公开任务中不存在，也无法从公开任务中推导得出），控制对该信息的访问，并估计由此产生的经过控制调整后的可执行准确率变化。在2,048个独特家族、两个模型家族、两个可执行领域以及262,144次生成的实验中，暴露在每个模型-领域组合中都提高了准确率，增益范围从+7.17到+27.31个百分点。这些发现区分了两个经常被混淆的实证问题：基准测试是否被污染，以及污染实际造成了多大影响。

    arXiv:2609.27176v1 Announce Type: new  Abstract: Evidence that evaluation material entered training does not reveal how much it affected evaluation. This distinction leaves a contaminated benchmark score difficult to interpret: provenance can establish contact, but only a counterfactual can quantify the performance attributable to that contact. We present LeakScale, an interventional framework for estimating this missing quantity. LeakScale creates fresh executable tasks that require private, family-specific information absent from and non-derivable from the public task, controls access to that information, and estimates the resulting control-adjusted change in executable accuracy. Across 2,048 unique families, two model families, two executable domains, and 262,144 generations, exposure improves accuracy in every model-by-domain combination, with gains ranging from +7.17 to +27.31 percentage points. These findings separate two empirical questions that are often conflated: whether benc
    
[^96]: COT-TTS：基于思维链推理的音频上下文感知文本转语音

    COT-TTS: Audio Context-Aware Text-to-Speech with Chain-of-Thought Reasoning

    [https://arxiv.org/abs/2609.22697](https://arxiv.org/abs/2609.22697)

    提出了COT-TTS任务，通过思维链推理从历史对话音频中自然推断说话风格并合成指定音色的语音，同时构建了包含900万样本的大规模双语对话语音数据集和人工验证基准来支持该任务。

    

    近年来，文本转语音系统在语音表现力和可控性方面取得了显著进展。然而，生成语音的说话风格通常依赖于用户明确指定的指令。在自然对话中，说话风格应当自然地从先前的对话上下文中推断出来。因此，我们提出了COT-TTS，这是一个上下文感知的、基于推理的文本转语音任务。给定历史对话音频、目标文本和参考语音，系统需要理解对话上下文，推断出明确的中间推理过程，并最终合成具有指定音色的目标语音。为支持这一任务，我们构建了一个包含900万个训练样本的大规模双语对话语音数据集，其中包括100万个高质量样本的子集。我们进一步构建了一个包含800个经人工验证样本的源不重叠基准测试集，并建立了强大的任务规范。

    arXiv:2609.22697v1 Announce Type: new  Abstract: Recently, text-to-speech systems have made significant progress in speech expressiveness and controllability. However, the speaking style of generated speech typically relies on clear user-specified instructions. In natural conversations, speaking style should be naturally inferred from the preceding conversational context. Therefore, we propose COT-TTS, a context-aware, reasoning-based text-to-speech task. Given historical conversation audio, target text, and a reference speech, the system should comprehend the conversational context, infer an explicit intermediate reasoning, and finally synthesize the target speech with the specified timbre. To support this task, we constructed a large-scale bilingual conversational speech dataset comprising 9 million training samples, including a high-quality subset of 1 million samples. We further constructed a source-disjoint benchmark with 800 human-verified samples and established strong task-spec
    
[^97]: Apollo Restore：一个针对古希腊语历史文本优化的基础大语言模型，专用于以“中间填空”方式修复古希腊文本

    Apollo Restore: A Foundation LLM for Historical Greek Optimized for Fill-in-the-Middle Restoration of Ancient Greek Texts

    [https://arxiv.org/abs/2609.22455](https://arxiv.org/abs/2609.22455)

    Apollo Restore 是首个面向历史希腊语（乃至任何古代地中海语言）的 240 亿参数基础大语言模型，通过“中间填空”目标微调，能够在不知缺失文本长度的情况下修复残缺古希腊文本，并在长度平衡的评估指标下大幅超越已发表的最强模型。

    

    我们提出了 Apollo Restore，一个拥有 240 亿参数的大语言模型，用于修复残缺古希腊文本中的缺损（即物理性空缺）。该模型从 Mistral Small 微调而来，采用“中间填空”训练目标，无需预先知晓缺失片段的长度即可重建缺失内容。据我们所知，这是首个针对历史希腊语的大规模解码器模型，也是首个针对任何古代地中海语言的大规模解码器模型。按照先前工作的评估方式，在最多十个字符的短缺损上，Apollo Restore 对文献纸草、文学纸草和石刻铭文缺损分别有 80.6%/54.6%/61.0% 的情况将正确的修复结果排在前二十个候选之中，超过已发表最强模型 1.6 倍/2.6 倍/1.4 倍。然而，先前的评估协议因偏向极短缺损而夸大了分数；在长度平衡的指标下，Apollo Restore 相对于已发表最强模型的优势……（摘要在此处被截断）

    arXiv:2609.22455v1 Announce Type: new  Abstract: We present Apollo Restore, a 24-billion-parameter large language model for restoring lacunae---physical gaps---in fragmentary Ancient Greek texts. Fine-tuned from Mistral Small with a fill-in-the-middle objective, Apollo Restore reconstructs missing spans without requiring oracle knowledge of their length. To our knowledge, it is the first large-scale decoder model for historical Greek, and the first for any ancient Mediterranean language. Evaluated as in prior work, on short gaps of up to ten characters, Apollo Restore places the correct restoration among its top twenty candidates for 80.6%/54.6%/61.0% of documentary-papyrus, literary-papyrus, and stone-inscription lacunae, exceeding the strongest published models by $1.6\times$/$2.6\times$/$1.4\times$. Prior evaluation protocols, however, inflate scores through a bias toward trivially short gaps; under a length-balanced metric Apollo Restore's advantage over the strongest published mod
    
[^98]: 重新思考与人类对齐的评估：超越WER的语义指标分析

    Rethinking Human-Aligned Evaluation: An Analysis of Semantic Metrics Beyond WER

    [https://arxiv.org/abs/2609.21663](https://arxiv.org/abs/2609.21663)

    该论文提出HATS-en英语数据集用于以人为中心的ASR评估，发现WER与人类判断的一致性在所有指标中最低，而最佳配置的SemDist语义指标与人类判断一致性最高，支持ASR评估从WER转向语义指标。

    

    词错误率（WER）是自动语音识别（ASR）中最常用的指标，它将所有与参考文本的词汇偏差视为同等代价，而不考虑其是否改变了语义。这引出一个问题：WER是否真的能反映人类对ASR转录质量的判断？我们引入了HATS-en，一个用于以人为中心的ASR评估的英语数据集。利用该数据集，我们对词汇指标与BERTScore和SemDist的多种配置进行了基准测试，其中改变了语言模型、网络层和池化策略。我们发现，在所有测试的指标中，WER与人类判断的一致性最低；表现最佳的SemDist配置达到了最高的整体一致性，超过了CER和BERTScore；并且没有任何单一模型在所有设置中都是最优的。CER尽管简单且成本低，但仍然与这些最佳配置非常接近。与先前的建议一致，我们的结果支持转变ASR评估方式

    arXiv:2609.21663v1 Announce Type: new  Abstract: Word Error Rate (WER), the most commonly used metric for Automatic Speech Recognition (ASR), treats every lexical deviation from the reference as equally costly, regardless of whether it changes meaning. This raises the question: does WER actually track how humans judge ASR transcript quality? We introduce HATS-en, an English dataset for human-centered ASR evaluation. Using this dataset, we benchmark lexical metrics against several configurations of BERTScore and SemDist, varying the language model, layer, and pooling strategy. We find that WER agrees least with human judgment among all metrics tested, that the best-performing SemDist configurations achieve the highest overall agreement, ahead of CER and BERTScore, and that no single model is best across settings. CER, despite its simplicity and low cost, remains remarkably close to these best configurations. In line with prior recommendations, our results support shifting ASR evaluation
    
[^99]: 超越原子词元：面向语言模型预训练的音节分解方法

    Beyond Atomic Tokens: Factorizing Syllables for Language Model Pretraining

    [https://arxiv.org/abs/2609.21362](https://arxiv.org/abs/2609.21362)

    提出了一种音位分词器，将越南语和汉语的音节分解为声母、韵母和声调三个音系成分，以极小的词表（汉语仅112项、越南语仅256项）实现了更高的编码效率。

    

    传统分词器将文本表示为字符或通过统计方法推导的子词，忽视了音节的内部音系结构，且通常需要较大的词表。我们提出了音位分词器，这是一种针对越南语和汉语的语言学驱动分词器，它将每个音节转换为国际音标（IPA），并将其分解为三个音系成分：声母、韵母和声调。这三个成分共同占据一个上下文位置，在保持音节级序列长度的同时，实现了音系相关音节之间的表示共享。非音系及不支持的单元则通过字符级回退机制处理。这种确定性设计无需依赖语料库进行词表学习，仅产生112个汉语词条和256个越南语词条的极小词表。内在评估表明，该分词器在两种语言中均取得了显著更高的Rényi效率……

    arXiv:2609.21362v1 Announce Type: new  Abstract: Conventional tokenizers represent text as characters or statistically derived subwords, overlooking the internal phonological structure of syllables and often requiring large vocabularies. We introduce \textbf{Phonemic Tokenizer}, a linguistically motivated tokenizer for Vietnamese and Chinese that converts each syllable into IPA and factorizes it into three phonological components: onset, rime, and tone. The three components jointly occupy one contextual position, preserving syllable-level sequence length while enabling representation sharing across phonologically related syllables. Non-phonological and unsupported units are handled through character-level fallback. This deterministic design requires no corpus-dependent vocabulary learning and yields vocabularies of only 112 entries for Chinese and 256 for Vietnamese. Intrinsic evaluation shows that the tokenizer achieves substantially higher R\'enyi efficiency in both languages, repres
    
[^100]: 思维状态实现内生推理

    State of Thought Enables Endogenous Reasoning

    [https://arxiv.org/abs/2609.16055](https://arxiv.org/abs/2609.16055)

    提出“思维状态”新推理范式，通过从模型内部信息传递中提取动力学-几何状态并用仅 582 参数的控制器在冻结模型上选择性激活历史推理支持，实现由模型内部状态主导的内生推理，摆脱外部控制。

    

    测试时计算已成为提升大语言模型能力的重要方法。然而，现有的测试时推理范式严重依赖外部施加的控制，要么通过固定的推理程序，要么通过在受限搜索空间中进行高成本扩展，这同时限制了泛化能力和效率。我们提出了思维状态，这是一种新的推理范式，使大语言模型能够进行内生推理，由模型自身的内部推理状态来支配推理的展开方式。具体而言，SoT 从模型内部的信息传递中提取一个紧凑的动力学-几何状态，并使用一个仅有 582 个参数的控制器作用于冻结的骨干模型，选择性地激活在当前推理状态下有用的历史推理支持，从而将推理构建为一个以状态为条件、基于证据的过程，而非外部规定的 token 链。在量化、通用、符号等任务上分别取得 1.34 倍、1.62 倍等提升。

    arXiv:2609.16055v1 Announce Type: cross  Abstract: Test-time compute has emerged as a major approach to improving the capabilities of Large Language Models (LLMs). However, existing test-time reasoning paradigms rely heavily on externally imposed control, either through fixed reasoning programs or through costly expansion in constrained search spaces, limiting both generalization and efficiency. We propose State of Thought (SoT), a new reasoning paradigm that enables endogenous reasoning in LLMs, with the model's internal reasoning state governing how reasoning unfolds. Concretely, SoT extracts a compact dynamics-geometric state from the model's internal information transfer and uses a 582-parameter controller on frozen backbones to selectively activate historical reasoning support useful under the current reasoning state, framing reasoning as a state-conditioned process over evidence rather than an externally prescribed token chain. Across quantitative (1.34x), general (1.62x), symbol
    
[^101]: 校准到足以知晓，却未校准到行动：伪造证据使LLM代理对不可知之事做出承诺

    Calibrated Enough to Know, Not Calibrated to Act: Fabricated Evidence Makes LLM Agents Commit to the Unknowable

    [https://arxiv.org/abs/2608.27167](https://arxiv.org/abs/2608.27167)

    本文发现LLM代理在面临伪造的专业面板证据时，会显著提高对不可预测问题的承诺率，其行为受证据包装的权威性驱动而非信息真实性或模型信念，揭示了一种可定位的校准失败。

    

    arXiv:2608.27167v1 公告类型：新 摘要：一个LLM代理在看到一个看起来专业的市场面板时，对一个问题做出方向性判断的频率远高于仅被问及该问题本身时的频率——在12个前沿模型中，随着证据的升级，承诺率从6.5%上升到54.0%。即使面板上的每个数字都是编造的，它同样会轻易承诺：完全伪造整个显示内容，使模型能看到的除了问题本身外无一是真，仍将承诺率从24.5%提升至36.8%，这在统计上与真实市场数据产生的37.6%无显著差异。解锁自信行动的不是信息，而是其包装的权威性。这种失败是狭窄且可定位的。能力不足并非答案：在附于相同面板的可回答问题上，相同模型几乎总是回答，且准确率近乎完美。这也不是信念问题——声明的概率在驱动行动变化达48个百分点的梯度上几乎不动，且得分糟糕。

    arXiv:2608.27167v1 Announce Type: new  Abstract: An LLM agent shown a professional-looking market panel commits to a directional call on a provably unpredictable question far more often than one asked the bare question: across 12 frontier models, commitment rises from 6.5% to 54.0% as evidence is escalated. It commits just as readily when every number on the panel is invented: fabricating the entire display, so nothing the model can see is true except the question itself, still lifts commitment from 24.5% to 36.8%, statistically indistinguishable from the 37.6% produced by genuine market data. What unlocks confident action is not information but the authority of its packaging. The failure is narrow and locatable. Incapacity is not the answer: on matched answerable questions attached to the same panels, the same models answer essentially always, at near-perfect accuracy. Nor is it belief - stated probabilities barely move across the gradient that swings action by 48 points, and score wo
    
[^102]: 变压器的通信图谱

    The Communication Map of a Transformer

    [https://arxiv.org/abs/2608.22007](https://arxiv.org/abs/2608.22007)

    提出了一种从权重出发绘制变压器所有潜在通信通道的“通信图谱”方法，能高效计算并揭示大多数注意力头对的耦合或回避模式，且具有广泛适用性。

    

    arXiv:2608.22007v1 公告类型：交叉 摘要：变压器的组件通过写入和读取共享残差流进行通信，机制可解释性已通过手工逐个电路绘制了这些连接。我们提出了通信图谱，它仅从权重出发，绘制了语言模型中所有潜在通信通道，将Elhage等人（2021）的组成分数推广为覆盖所有18类连接（从整个注意力头电路到单个神经元）的单一耦合系数。对所有候选通道的普查，从GPT-2中的$6.3\times10^{8}$到Pythia-6.9B中的$1.3\times10^{11}$，发现70-89%的头对方向偏离随机水平，有些强耦合，另一些则主动避免彼此。完整图谱在单个消费级GPU上计算GPT-2需15秒，Pythia-6.9B需11分钟。两个应用展示了该图谱的实用性。在应用1中，最强的头对头耦合恢复了

    arXiv:2608.22007v1 Announce Type: cross  Abstract: The components of a transformer communicate by writing to and reading from a shared residual stream, and mechanistic interpretability has mapped these connections by hand, one circuit at a time. We present the communication map, which charts every potential communication channel in a language model from weights alone, generalizing the composition score of Elhage et al. (2021) into a single coupling coefficient covering all 18 connection classes, from entire attention head circuits to single neurons. The census of all candidate channels, from $6.3\times10^{8}$ in GPT-2 to $1.3\times10^{11}$ in Pythia-6.9B, finds that 70-89% of head pairs are oriented far from chance, some coupled strongly and others actively avoiding each other. The full map costs 15 seconds for GPT-2 and 11 minutes for Pythia-6.9B on one consumer GPU. Two applications demonstrate the utility of the map. In Application 1, the strongest head-to-head couplings recover the
    
[^103]: 基于连续深度批处理的循环语言模型深度自适应推理

    Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching

    [https://arxiv.org/abs/2608.09444](https://arxiv.org/abs/2608.09444)

    本文提出了首个针对循环语言模型深度自适应推理的高效方法——连续深度批处理（CDB），通过在循环步骤之间动态重组批次、管理循环KV缓存并提前预测token退出时机，解决了不同循环深度token无法被标准批处理系统高效处理的问题。

    

    循环语言模型的一项主要优势是深度自适应推理。通过将共享层块循环可变次数，模型可以对“简单”token使用更少的计算，而对“困难”token使用更多的计算。然而，具有不同循环次数的token无法共享统一的前向传播过程，因此无法由标准批处理系统（如vLLM）处理。因此，深度自适应推理的实际价值取决于能否实现高效的批处理。我们提出了首个面向深度自适应循环语言模型的高效方法——连续深度批处理，该方法在循环步骤之间形成新的批次。我们的方法能够动态调度架构中的循环部分与非循环部分，管理循环KV缓存，并提前预测哪些token将退出循环，从而可以异步地准备批次。在Ouro 1.4B和Huginn 3.5B上的实验表明，完全循环架构最适合深度自适应

    arXiv:2608.09444v2 Announce Type: replace-cross  Abstract: A main promise of looped language models is depth-adaptive inference. By looping a block of shared layers a variable number of times, the model can use less compute for "easy" tokens and more for "hard" ones. However, tokens with different numbers of loops cannot share a uniform forward pass and therefore cannot be handled by standard batching systems such as vLLM. The practical value of depth-adaptive inference thus hinges on whether batching can be made efficient. We introduce the first efficient method for depth-adaptive looped LMs via continuous depth batching (CDB), which forms new batches between loop steps. Our method dynamically schedules looped and non-looped parts of the architecture, manages looped KV-caching, and predicts which tokens will exit the loop in advance so it can prepare batches asynchronously. Experiments on Ouro 1.4B and Huginn 3.5B show that fully looped architectures are best suited to depth-adaptive 
    
[^104]: 小模型足矣：基于LoRA适配器的AI编辑文本个性化风格改写

    Small Is Enough: Per-User Style Rewriting of AI-Edited Text via LoRA Adapters

    [https://arxiv.org/abs/2607.29238](https://arxiv.org/abs/2607.29238)

    InMyStyle提出一种隐私优先的单用户方案，仅对0.5B至7B的小型模型进行LoRA微调，即可让AI编辑的文本自动改写为符合个人写作风格，且实验表明小模型已足以胜任该改写任务。

    

    InMyStyle是一个隐私优先的单用户系统，它使小型语言模型能够将AI编辑的文本改写为符合个人用户写作风格，且在推理时无需指令提示。给定用户的文档，系统利用多个本地辅助大语言模型构建配对训练样本，并在参数量从0.5B到7B的Qwen2.5模型上微调LoRA适配器。借助长度感知的生成预算和自动分块机制，系统可支持不同长度的输入。我们报告了一项单用户案例研究：基于一位作者73段科学写作文本衍生出219个评估样本对，所有适配器均采用相同的rank-8、三轮训练方案。在贪心解码和采样解码两种方式下，自动综合评分（0-1量表）在各模型规模上均趋于平稳（Q=0.689-0.695，置信区间相互重叠）。在此设置下，小模型足以完成所测量的改写任务，而模型规模主要决定了……（摘要原文在此处截断）。

    arXiv:2607.29238v2 Announce Type: replace-cross  Abstract: InMyStyle is a privacy-first, single-user system that adapts small language models to rewrite AI-edited text towards an individual user's writing style without an instruction prompt at inference. Given a user's documents, it uses multiple local helper LLMs to construct paired training examples and fine-tunes LoRA adapters on Qwen2.5 models ranging from 0.5B to 7B parameters. Length-aware generation budgets and automatic chunking support inputs of different lengths. We report a single-user case study: 219 evaluation pairs derived from 73 paragraphs of one author's scientific writing, with all adapters trained using the same rank-8, three-epoch recipe. The automatic composite score (0-1 scale) plateaus across model sizes under both greedy and sampled decoding ($Q=0.689$-$0.695$, with overlapping confidence intervals). In this setting, small models are sufficient for the measured rewriting task, and model size mainly determines ef
    
[^105]: 面向掩码扩散语言模型的注意力折扣自适应采样器

    Attention-Discounted Adaptive Sampler for Masked Diffusion Language Models

    [https://arxiv.org/abs/2606.10829](https://arxiv.org/abs/2606.10829)

    提出免训练重排序规则ADAS，根据每个token对已选位置的注意力（按预测不确定性加权）贪婪地折扣其置信度分数，从而提升掩码扩散语言模型在低推理步数下的表现。

    

    掩码扩散语言模型可以通过在每次去噪迭代中揭示多个token来减少推理步骤，但这种并行性是脆弱的：当各个位置的预测相互耦合时，单独看来置信度较高的位置一起提交可能并不安全。现有的免训练采样器（如Top-k、Fast-dLLM和EB-Sampler）主要控制揭示多少个token，而往往通过忽略所选集合内部交互的逐token分数对候选进行排序。我们提出了ADAS，这是一种免训练的重排序规则，它保持基础采样器的停止规则不变，并根据每个token对已选择位置的注意力（以其预测不确定性加权）来贪婪地折扣其逐token置信度分数。在LLaDA-8B-Base和Dream-7B-Base上，于推理基准GSM8K和MATH500以及代码基准HumanEval和MBPP上的实验表明，将ADAS插入到所有三种采样器中都能改善低NFE（前向评估次数）下的性能。

    arXiv:2606.10829v3 Announce Type: replace-cross  Abstract: Masked diffusion language models can reduce inference steps by revealing multiple tokens per denoising iteration, but this parallelism is fragile: positions that are individually confident may be unsafe to commit together when their predictions are coupled. Existing training-free samplers such as Top-$k$, Fast-dLLM, and EB-Sampler mainly control how many tokens to reveal, while often ranking candidates by token-wise scores that ignore interactions within the selected set. We propose ADAS, a training-free reranking rule that leaves the base sampler's stopping rule unchanged and greedily discounts each token-wise confidence score according to its attention to already selected positions, weighted by their prediction uncertainty. Across LLaDA-8B-Base and Dream-7B-Base on the reasoning benchmarks GSM8K and MATH500 and the code benchmarks HumanEval and MBPP, plugging ADAS into all three samplers improves low-NFE performance at matche
    
[^106]: INFUSER：影响力引导的自我进化提升推理能力

    INFUSER: Influence-Guided Self-Evolution Improves Reasoning

    [https://arxiv.org/abs/2606.09052](https://arxiv.org/abs/2606.09052)

    INFUSER提出了一种影响力引导的自我进化框架，通过生成器与求解器的协同训练，利用优化器感知的影响力分数来改进问题生成，从而显著提升推理能力。

    

    自我进化为增强推理能力提供了一条可扩展的路径：预训练语言模型仅需极少的外部监督即可自我提升。然而，现有方法要么依赖大量精心策划或教师生成的训练数据，要么在生成器无监督运行时，仅通过难度启发式给予奖励，这未必能改进求解器。我们引入了INFUSER，一种迭代协同训练框架，包含两个共同演化的角色：一个生成器，从自动收集的非结构化文档池中起草问题和参考标准答案；以及一个求解器，通过在这些问题上训练来改进自身。求解器使用标准正确性奖励，依据生成器提供的答案进行训练，而生成器则通过一个优化器感知的影响力分数获得奖励，该分数衡量每个提议的问题是否真正能提升求解器在目标分布上的表现。由于这种连续且嘈杂的影响力分数难以直接处理，我们采用了相应策略进行优化。

    arXiv:2606.09052v4 Announce Type: replace-cross  Abstract: Self-evolution offers a scalable path to stronger reasoning: a pretrained language model improves itself with only minimal external supervision. Yet existing methods either depend on extensively curated or teacher-generated training data, or, when the generator runs unsupervised, reward it by a difficulty heuristic that need not improve the solver. We introduce INFUSER, an iterative co-training framework with two co-evolving roles: a Generator that drafts questions and reference golden answers from a pool of unstructured, automatically collected documents, and a Solver that improves by training on them. The solver is trained with standard correctness rewards against the generator-provided answers, while the generator is rewarded by an optimizer-aware influence score that measures whether each proposed question would actually improve the solver on the target distribution. Because this continuous, noisy influence score is poorly 
    
[^107]: 用于隐式偏好的统计先验：在个人智能体中将技能选择解耦为本地框架

    Statistical Priors for Implicit Preferences: Decoupling Skill Selection as a Local Harness in Personal Agents

    [https://arxiv.org/abs/2606.05828](https://arxiv.org/abs/2606.05828)

    提出一种将统计偏好学习与语义意图解析严格解耦的轻量级本地框架，利用本地统计先验来调节远程LLM的技能选择决策，使个人智能体能够学习隐式用户偏好，并取得最低的累积遗憾和最高的测试准确率。

    

    随着大型语言模型（LLM）能力的不断进步，依赖基于API的远程模型和外部技能的本地部署个人智能体已成为一种新范式。随着可用技能的快速扩展，使个人智能体能够学习并适应隐式用户偏好成为一项关键挑战。然而，本地部署的限制排除了复杂的集中式选择算法，因此迫切需要一种轻量级的本地偏好框架。本文通过一种新颖的架构探索了此类框架的实现，该架构将统计偏好学习与语义意图解析严格解耦。具体而言，我们利用本地化的统计结果来影响和调节远程LLM的选择决策。大量评估表明，我们的解耦方法实现了最低的累积遗憾和最高的测试准确率，显著优于传统的基于记忆的方法。

    arXiv:2606.05828v2 Announce Type: replace  Abstract: As Large Language Model (LLM) capabilities advance, locally deployed personal agents relying on API-based remote models and external skills have emerged as a novel paradigm. With the rapid expansion of available skills, enabling personal agents to learn and adapt to implicit user preferences becomes a critical challenge. However, local deployment constraints preclude complex centralized selection algorithms, creating an urgent need for a lightweight local preference harness. This paper explores the implementation of such a harness through a novel architecture that strictly decouples statistical preference learning from semantic intent parsing. Specifically, we leverage localized statistical results to influence and modulate the selection decisions of the remote LLM. Extensive evaluations demonstrate that our decoupled approach achieves the lowest cumulative regret and highest test accuracy, significantly outperforming traditional mem
    
[^108]: 基于有限标注的大语言模型选择

    Large Language Model Selection with Limited Annotations

    [https://arxiv.org/abs/2605.24981](https://arxiv.org/abs/2605.24981)

    SELECT-LLM是首个大语言模型主动选择框架，通过基于期望信息增益的查询选择规则，仅需少量最具信息量的标注查询即可从开放或黑盒候选模型中识别出给定任务的最佳LLM。

    

    为特定任务选择合适的大语言模型（LLM）需要比较众多强有力的候选模型，然而标准评估依赖于在固定评估集上进行代价高昂的标注。为解决这一挑战，我们开发了SELECT-LLM，这是首个用于大语言模型主动模型选择的框架。SELECT-LLM旨在找到一小组查询，其标注对于识别给定任务的最佳LLM最具信息量。为此，我们引入了一种基于期望信息增益的查询选择规则，该增益由候选模型输出之间的成对相似度计算得出。由于该规则仅使用生成的模型响应，SELECT-LLM可应用于各种候选模型，无需对其架构做出假设或访问模型权重。这使其既适用于开源权重模型，也适用于黑盒LLM。我们在23个数据集、156个被评估模型、多样化的任务类别以及多种文本评估指标上对SELECT-LLM进行了评估。

    arXiv:2605.24981v2 Announce Type: replace  Abstract: Choosing a Large Language Model (LLM) for a given task requires comparing many strong candidates, yet standard evaluation relies on costly annotations over fixed evaluation sets. To address this challenge, we develop SELECT-LLM, the first framework for active model selection of LLMs. SELECT-LLM aims to find a small set of queries whose annotations are most informative for identifying the best LLM for a given task. To this end, we introduce a query selection rule based on expected information gain, computed from pairwise similarities between candidate model outputs. Because this rule only uses generated model responses, SELECT-LLM can be applied across candidate models without assumptions about their architecture or access to model weights. This makes it suitable for both open-weight and black-box LLMs. We evaluate SELECT-LLM across 23 datasets, 156 evaluated models, diverse task families, and multiple text evaluation metrics. Across 
    
[^109]: 基于检索、聚类与生成的从案例数据库自动生成法律评注

    Generating Legal Commentaries from Case Databases via Retrieval, Clustering, and Generation

    [https://arxiv.org/abs/2605.24534](https://arxiv.org/abs/2605.24534)

    本文提出一种无需人工构建教义学框架的全自动流水线，通过检索、聚类与大语言模型生成技术，将德国联邦最高法院的数千份判决自动转化为针对《德国民法典》具体法条的法律评注，并通过人类专家与LLM评审的五维度评估验证了其可行性。

    

    我们提出了一种全自动化的流水线，能够将大量法院判决集合转化为针对法条的法律评注，而无需提供任何手工构建的教义学框架。我们使用德国联邦最高法院引用《德国民法典》（BGB）第242、280、812和823条的4,555份判决，提取段落级文本块，总结其推理过程并提取关键词，随后对这些关键词进行嵌入和聚类。针对每个聚类，大语言模型生成标题并综合出富含引用的章节，再由四个最先进的大语言模型将其合并为连贯的评注。我们从五个维度进行评估——主题相关性、标题匹配度、引用忠实度、聚类区分度以及逻辑排序——并同时采用人类专家和大语言模型评审进行评判。我们的结果表明，从法院判决中进行类似评注的论点挖掘，以生成可在几分钟内以极低成本刷新的报告是可行的，然而……（原文摘要在此处被截断）

    arXiv:2605.24534v2 Announce Type: replace  Abstract: We present a fully automated pipeline that transforms large collections of court decisions into legal commentaries for statutes - without providing any handcrafted doctrinal framework. Using 4.555 decisions of the German Federal Court of Justice that cite sections 242, 280, 812 and 823 of the German Civil Code (BGB), we extract paragraph-level chunks, summarize their reasoning, and derive keywords, which are embedded and clustered. For each cluster, an LLM generates headings and synthesizes citation-rich sections, which are then merged into coherent commentaries by four state-of-the-art LLMs. We evaluate along five dimensions - topical relevance, heading-match, citation faithfulness, cluster distinction and logical ordering - using both a human expert and an LLM-judge. Our results show that commentary-like argument mining from court decisions to generate reports that can be refreshed within minutes at minimal cost is feasible, yet th
    
[^110]: 音频大语言模型中基于直接偏好优化的英中语码转换语音识别

    Direct Preference Optimization for English-Mandarin Code-Switching Speech Recognition in Audio LLMs

    [https://arxiv.org/abs/2605.23975](https://arxiv.org/abs/2605.23975)

    通过直接偏好优化（DPO）对齐音频大语言模型，可有效解决英中语码转换语音识别中的语言遗漏、翻译替代转录和幻觉问题，使混合语言错误率最高降低89.6%。

    

    音频大语言模型尽管具有强大的多语言能力，但在转录语码转换语音时仍表现出系统性的失败。聚焦于英语-普通话场景，我们识别出三种失败模式：语言遗漏、翻译而非转录、以及幻觉。我们应用直接偏好优化（DPO）来对齐模型，构建偏好数据对，其中被选中的响应保留混合语言内容，而被拒绝的响应则模仿失败模式。在10万对数据（570小时）上训练三个音频大语言模型，我们观察到一致的行为转变：在被要求转录时，模型学会保留语言构成而非进行翻译。这种对齐带来了最高达89.6%（分布内）和20.0%（分布外）的混合语言错误率（MER）降低。我们的发现表明，DPO可以有效地从多语言音频大语言模型中引出正确的语码转换转录行为。

    arXiv:2605.23975v2 Announce Type: replace  Abstract: Audio large language models (Audio LLMs) exhibit systematic failures in transcribing code-switching speech despite strong multilingual capabilities. Focusing on English-Mandarin, we identify three failure modes: language omission, translation-instead-of-transcription, and hallucination. We apply Direct Preference Optimization (DPO) to align models, constructing preference pairs in which chosen responses preserve mixed-language content while rejected responses mimic failure patterns. Training three Audio LLMs on 100K pairs (570 hours), we observe consistent behavioral shifts: models learn to preserve language composition rather than translating when prompted for transcription. This alignment yields MER reductions up to 89.6% (in-distribution) and 20.0% (out-of-distribution). Our findings suggest DPO can effectively elicit correct code-switching transcription behavior from multilingual Audio LLMs.
    
[^111]: 询问老朋友：诊断与缓解基于大语言模型的成文法问答中的时间性失效模式

    Asking For An Old Friend: Diagnosing and Mitigating Temporal Failure Modes in LLM-based Statutory Question Answering

    [https://arxiv.org/abs/2605.23497](https://arxiv.org/abs/2605.23497)

    本文构建了包含312个专家验证的时间敏感德国成文法问答基准，识别出大语言模型的两种时间性失效模式（截止后过时与新近偏好偏差），并通过基于事实日期提取和版本过滤的检索增强方法进行缓解。

    

    大型语言模型越来越多地被用于法律研究，但其固定的训练截止时间和对静态参数化知识的依赖，与成文法不断演变的特性相冲突。我们研究了两种时间性失效模式：截止后过时，即模型在立法修订后仍适用已被取代的规则；以及新近偏好偏差，即即使案件事实应适用历史版本的规定，模型仍偏好较新的条文。为此，我们提出了一个包含312个经专家验证、时间敏感的德国成文法问答对的基准，涵盖三个类别：截止后修订问题、修订前问题和多条文修订前问题。我们在四种推理设置下评估了来自OpenAI、Anthropic和DeepSeek的五个大语言模型：原始设置、网络搜索，以及两种通过事实日期提取和版本过滤来强制时间有效性的检索增强变体。我们使用经过人类专家验证的LLM-as-a-judge（LLM作裁判）方法进行评估（摘要在此处截断）。

    arXiv:2605.23497v2 Announce Type: replace  Abstract: Large language models are increasingly used for legal research, yet their fixed training cutoffs and reliance on static parametric knowledge are at odds with the evolving nature of statutory law. We study two temporal failure modes: post-cutoff staleness, where models apply superseded rules after legislative amendments, and recency bias, where models prefer newer provisions even when a historical version governs the fact pattern. To this end, we present a benchmark of 312 expert-validated, time-sensitive German statutory QA pairs spanning three categories: Post-Cutoff Amendment Questions, Pre-Amendment Questions, and Multi-Provision Pre-Amendment Questions. We evaluate five LLMs by OpenAI, Anthropic and DeepSeek under four inference settings: Vanilla, Web-search, and two retrieval-augmented variants that enforce temporal validity via a fact date extraction and version filtering. Using an LLM-as-a-judge validated against human expert 
    
[^112]: AcuityBench：评估临床急迫度识别与不确定性对齐

    AcuityBench: Evaluating Clinical Acuity Identification and Uncertainty Alignment

    [https://arxiv.org/abs/2605.11398](https://arxiv.org/abs/2605.11398)

    该论文提出AcuityBench基准，通过统一五个公开数据集和四级急迫度框架，系统评估语言模型从用户医疗描述中识别就医紧急程度的能力，并纳入医生确认的模糊案例以衡量模型的不确定性对齐水平。

    

    我们提出了AcuityBench，这是一个用于评估语言模型能否从用户医疗描述中识别适当就医紧急程度的基准。现有的健康类基准侧重于医学问答、广泛的健康交互或狭窄的特定工作流分诊任务，但未能提供跨这些场景的急迫度识别的统一评估。AcuityBench通过统一五个公开数据集来填补这一空白，这些数据集涵盖用户对话、在线论坛帖子、临床案例情景和患者门户消息，并采用从居家监测到立即急诊护理的共享四级急迫度框架。该基准包含914个案例，其中包括697个用于标准准确性评估的共识案例，以及217个经医生确认的模糊案例，用于不确定性感知评估。该基准支持两种互补的任务形式：问答设置中的显式四分类，以及自由格式的对话响应。

    arXiv:2605.11398v2 Announce Type: replace  Abstract: We introduce AcuityBench, a benchmark for evaluating whether language models identify the appropriate urgency of care from user medical presentations. Existing health benchmarks emphasize medical question answering, broad health interactions, or narrow workflow-specific triage tasks, but they do not offer a unified evaluation of acuity identification across these settings. AcuityBench addresses this gap by harmonizing five public datasets spanning user conversations, online forum posts, clinical vignettes, and patient portal messages under a shared four-level acuity framework ranging from home monitoring to immediate emergency care. The benchmark contains 914 cases, including 697 consensus cases for standard accuracy evaluation and 217 physician-confirmed ambiguous cases for uncertainty-aware evaluation. It supports two complementary task formats: explicit four-way classification in a QA setting, and free-form conversational response
    
[^113]: StraTA：通过策略性轨迹抽象激励智能体强化学习

    StraTA: Incentivizing Agentic Reinforcement Learning with Strategic Trajectory Abstraction

    [https://arxiv.org/abs/2605.06642](https://arxiv.org/abs/2605.06642)

    StraTA 提出了一种策略性轨迹抽象框架，通过在智能体强化学习中引入显式的轨迹级策略并联合训练策略生成与动作执行，显著提升了大语言模型智能体在长时程决策任务中的样本效率和最终性能。

    

    大语言模型（LLM）越来越多地被用作交互式智能体，但针对长时程决策对它们进行优化仍然十分困难，因为现有方法在很大程度上是纯反应式的，这既削弱了探索能力，也削弱了对长轨迹的信用分配。在这项工作中，我们提出了策略性轨迹抽象，这是一个简单的框架，将显式的轨迹级策略引入智能体强化学习（RL）中。StraTA 从初始任务状态采样出一个紧凑的策略，使后续动作以该策略为条件，并通过分层 GRPO 式的 rollout 设计联合训练策略生成与动作执行，同时借助多样化策略 rollout 和批判性自我判断进一步增强。在 ALFWorld、WebShop 和 SciWorld 上的实验表明，StraTA 在样本效率和最终性能上均持续优于强基线方法。StraTA 的成功率达到 93%……

    arXiv:2605.06642v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used as interactive agents, but optimizing them for long-horizon decision making remains difficult because current methods are largely purely reactive, which weakens both exploration and credit assignment over extended trajectories. In this work, we present Strategic Trajectory Abstraction (StraTA), a simple framework that introduces an explicit trajectory-level strategy into agentic reinforcement learning (RL). StraTA samples a compact strategy from the initial task state, conditions subsequent actions on that strategy, and trains strategy generation and action execution jointly with a hierarchical GRPO-style rollout design, further enhanced by diverse strategy rollout and critical self-judgment. Experiments on ALFWorld, WebShop, and SciWorld show that StraTA consistently improves both sample efficiency and final performance over strong baselines. StraTA reaches success rates of 93
    
[^114]: UniPrefill：通过块级动态稀疏化实现通用的长上下文预填充加速

    UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification

    [https://arxiv.org/abs/2605.06221](https://arxiv.org/abs/2605.06221)

    UniPrefill提出了一种基于块级动态稀疏化的通用长上下文预填充加速方法，能够同时兼容全注意力模型与线性注意力、滑动窗口注意力等混合架构，并可与连续批处理等现代推理引擎机制集成。

    

    随着大语言模型（LLM）的快速发展，其能力日益增强，同时对更长上下文长度的需求也不断增加。为了提升长上下文处理的推理效率，近期已有若干新颖的低复杂度混合架构被提出，有效缓解了长上下文推理的计算负担。然而，现有的长上下文预填充加速研究主要聚焦于稀疏注意力机制，这类方法只有在完全注意力模型上才能达到最大加速效果。当迁移到新兴架构（如线性注意力/完全注意力混合架构或滑动窗口注意力/完全注意力混合架构）时，这些预填充加速方法会出现显著的性能下降。此外，这类方法通常与连续批处理不兼容，因而难以集成到vLLM等现代推理引擎中。为此……

    arXiv:2605.06221v2 Announce Type: replace  Abstract: As large language models (LLMs) continue to advance rapidly, they are becoming increasingly capable while simultaneously demanding ever-longer context lengths. To improve the inference efficiency of long-context processing, several novel low-complexity hybrid architectures have recently been proposed, effectively alleviating the computational burden of long-context inference. However, existing research on long-context prefill acceleration remains predominantly focused on sparse attention mechanisms, which achieve their maximum speedup only on full-attention models. When transferred to emerging architectures--such as linear/full attention hybrids or sliding window/full attention hybrids--these prefill acceleration approaches suffer significant performance degradation. Furthermore, such methods are generally incompatible with continuous batching, making them difficult to integrate into modern inference engines such as vLLM. To this end
    
[^115]: Human-1 by Josh Talks：基于真实世界对话的印地语全双工对话建模框架

    Human-1 by Josh Talks: A Full-Duplex Conversational Modeling Framework in Hindi using Real-World Conversations

    [https://arxiv.org/abs/2604.23295](https://arxiv.org/abs/2604.23295)

    该论文通过适配 Moshi 双工语音架构，利用 26,000 小时真实印地语自发对话数据，构建了首个开放、可复现的印地语全双工口语对话系统，实现对打断、重叠等自然对话行为的建模。

    

    全双工口语对话系统能够对打断、话语重叠和回应等自然对话行为进行建模，但这类系统在印度语言领域仍然鲜有探索。我们通过适配最先进的双工语音架构 Moshi，使用定制开发的印地语分词器，并在从 14,695 名说话者收集的 26,000 小时具有独立说话者通道的真实自发性对话数据上进行训练，提出了首个开放且可复现的印地语全双工口语对话系统，从而能够从自然交互中直接学习话轮转换和话语重叠模式。为支持印地语文本生成，我们替换了原始的英语分词器，重新初始化了依赖文本词表的参数，同时保留了预训练的音频组件。我们提出了一个两阶段训练方案——先进行大规模预训练，随后在 1,000 小时的对话数据上进行微调。通过提示式对话进行的评估……

    arXiv:2604.23295v3 Announce Type: replace-cross  Abstract: Full-duplex spoken dialogue systems can model natural conversational behaviours such as interruptions, overlaps, and backchannels, yet such systems remain largely unexplored for Indian languages. We present the first open, reproducible full-duplex spoken dialogue system for Hindi by adapting Moshi, a state-of-the-art duplex speech architecture, using a custom Hindi tokeniser and training on 26,000 hours of real spontaneous conversations collected from 14,695 speakers with separate speaker channels, enabling direct learning of turn-taking and overlap patterns from natural interactions. To support Hindi text generation, we replace the original English tokeniser and reinitialise text-vocabulary-dependent parameters while retaining the pre-trained audio components. We propose a two-stage training recipe -- large-scale pre-training followed by fine-tuning on 1,000 hours of conversational data. Evaluation through the prompted dialogu
    
[^116]: VLAA-GUI：知道何时停止、恢复与搜索——一个模块化的GUI自动化框架

    VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation

    [https://arxiv.org/abs/2604.21375](https://arxiv.org/abs/2604.21375)

    VLAA-GUI提出一个模块化GUI自动化框架，通过强制性完整性验证器杜绝无视觉证据的过早成功宣告、多层级循环断路器打破重复失败循环、以及按需在线搜索应对不熟悉元素，系统性地解决了GUI智能体的过早停止与重复循环两大核心难题。

    

    自主GUI智能体面临两个根本性挑战：一是过早停止，即智能体在没有可验证证据的情况下过早宣布任务成功；二是重复循环，即智能体在没有恢复机制的情况下反复执行相同的失败动作。我们提出了VLAA-GUI，这是一个模块化的GUI智能体框架，围绕三个集成组件构建，用以指导系统何时停止、恢复和搜索。第一，一个强制性的完整性验证器在每个完成步骤强制执行UI可观察的成功标准和验证——借助一个智能体级验证器，利用决策规则对完成声明进行交叉审查，拒绝缺乏直接视觉证据的声明。第二，一个强制性的循环断路器提供多层级过滤：在重复失败后切换交互模式，在屏幕状态持续重现后强制改变策略，并将反思信号与策略转换绑定。第三，一个按需触发的搜索智能体在线搜索不熟悉的……（摘要原文在此处截断）

    arXiv:2604.21375v3 Announce Type: replace-cross  Abstract: Autonomous GUI agents face two fundamental challenges: early stopping, where agents prematurely declare success without verifiable evidence, and repetitive loops, where agents cycle through the same failing actions without recovery. We present VLAA-GUI, a modular GUI agentic framework built around three integrated components that guide the system on when to Stop, Recover, and Search. First, a mandatory Completeness Verifier enforces UI-observable success criteria and verification at every finish step -- with an agent-level verifier that cross-examines completion claims with decision rules, rejecting those lacking direct visual evidence. Second, a mandatory Loop Breaker provides multi-tier filtering: switching interaction mode after repeated failures, forcing strategy changes after persistent screen-state recurrence, and binding reflection signals to strategy shifts. Third, an on-demand Search Agent searches online for unfamilia
    
[^117]: 通过有限音频弥合语音-文本差距，实现基于大语言模型的ASR的高效领域自适应

    Closing the Speech-Text Gap with Limited Audio for Effective Domain Adaptation in LLM-Based ASR

    [https://arxiv.org/abs/2604.06487](https://arxiv.org/abs/2604.06487)

    提出混合批处理（MB）策略，仅用不到4小时的目标域语音数据即可使基于LLM的ASR达到与使用完整数据集传统微调相当或更优的词错误率，有效弥合了语音-文本模态差距。

    

    传统的端到端自动语音识别（ASR）系统依赖成对的语音-文本数据进行领域自适应。近期基于大语言模型（LLM）的ASR架构通过投影模块将语音编码器与大型语言模型连接起来，从而能够仅使用文本数据进行自适应。然而，这引入了模态差距问题，因为LLM并未接触到语音投影器所产生的含噪表示。我们研究了少量语音数据是否能够缓解这种不匹配问题。我们比较了三种策略：仅文本自适应、成对语音-文本自适应，以及结合两者的混合批处理（MB）。在域内和域外设置下的实验表明，即使是有限的语音数据也能持续提升性能。值得注意的是，混合批处理方法仅使用目标域10%（不到4小时）的语音数据，其词错误率就能达到与使用完整数据集进行传统ASR微调相当甚至更好的水平，这表明少量语音数据……

    arXiv:2604.06487v2 Announce Type: replace  Abstract: Conventional end-to-end automatic speech recognition (ASR) systems rely on paired speech-text data for domain adaptation. Recent LLM-based ASR architectures connect a speech encoder to a large language model via a projection module, enabling adaptation with text-only data. However, this introduces a modality gap, as the LLM is not exposed to the noisy representations produced by the speech projector. We investigate whether small amounts of speech can mitigate this mismatch. We compare three strategies: text-only adaptation, paired speech-text adaptation, and mixed batching (MB), which combines both. Experiments in in-domain and out-of-domain settings show that even limited speech consistently improves performance. Notably, MB using only 10% of the target-domain (less than 4 hours) speech achieves word error rates comparable to, or better than, conventional ASR fine-tuning with the full dataset, indicating that small amounts of speech
    
[^118]: Combee：为自我改进的语言模型智能体扩展提示学习

    Combee: Scaling Prompt Learning for Self-Improving Language Model Agents

    [https://arxiv.org/abs/2604.04247](https://arxiv.org/abs/2604.04247)

    Combee 提出了一种新颖的框架，通过有原则的并行扩展策略解决现有提示学习方法在高并行度下质量下降的问题，从而同时提升自我改进语言模型智能体提示学习的效率和效果。

    

    提示学习（prompt learning）的最新进展使大型语言模型智能体能够在不改变参数的情况下，从推理时上下文中获取与任务相关的知识。例如，现有方法（如 ACE 或 GEPA）可以基于先前的智能体运行记录来学习系统提示，从而提升准确率。然而，这些方法主要集中于单智能体或低并行度的设置，这从根本上限制了它们从大量收集的智能体轨迹中高效学习的能力。鉴于从大量智能体轨迹或并行智能体执行中学习的趋势日益增长，以并行方式运行提示学习将是高效且有益的。然而，由于缺乏有原则的扩展策略，当前方法在高并行度下会出现质量下降的问题。为了同时提升提示学习的效率和效果，我们提出了 Combee，一个用于扩展自我改进智能体并行提示学习的新颖框架。Combee 加速了学习……

    arXiv:2604.04247v2 Announce Type: replace  Abstract: Recent advances in prompt learning allow large language model agents to acquire task-relevant knowledge from inference-time context without parameter changes. For example, existing methods (like ACE or GEPA) can learn system prompts to improve accuracy based on previous agent runs. However, these methods primarily focus on single-agent or low-parallelism settings. This fundamentally limits their ability to efficiently learn from a large set of collected agentic traces. It would be efficient and beneficial to run prompt learning in parallel to accommodate the growing trend of learning from many agentic traces or parallel agent executions. Yet without a principled strategy for scaling, current methods suffer from quality degradation with high parallelism. To improve both the efficiency and quality of prompt learning, we propose Combee, a novel framework to scale parallel prompt learning for self-improving agents. Combee speeds up learn
    
[^119]: 逐层目标传播：通过以目标为中心的传播实现高效的组件归因

    Layer-wise Target Propagation: Efficient Component Attribution through Target Centric Propagation

    [https://arxiv.org/abs/2603.19742](https://arxiv.org/abs/2603.19742)

    提出逐层目标传播（LTP）框架，仅需一次前向和一次反向传播即可在冻结的Transformer上忠实地追踪信息流，在模型组件数量方面实现O(1)时间复杂度的高效密集组件归因。

    

    理解基于Transformer的大型语言模型（LLM）的内部机制对于其可靠部署和有效运行至关重要。尽管近期的研究已经产生了大量试图在忠实性与计算效率之间取得平衡的归因方法，但密集的组件归因的代价仍然高得令人望而却步。在本工作中，我们提出了逐层目标传播（Layer-wise Target Propagation, LTP），这是一种新颖的框架，能够在冻结的Transformer上通过一次前向传播和一次反向传播忠实地追踪信息流，而无需反事实样本。LTP以解析方式将Transformer的计算结构分解并线性化为不同的路径，并沿着这些路径传播一个目标化的反嵌入（unembedding）向量，从而在每个残差位置获得有效表示。这种以目标为中心的传播在模型组件数量方面实现了O(1)的时间复杂度，并可扩展至长输入场景。

    arXiv:2603.19742v3 Announce Type: replace-cross  Abstract: Understanding the internal mechanisms of transformer-based large language models (LLMs) is crucial for their reliable deployment and effective operation. While recent efforts have yielded a plethora of attribution methods attempting to balance faithfulness and computational efficiency, dense component attribution remains prohibitively expensive. In this work, we introduce Layer-wise Target Propagation (LTP), a novel framework that faithfully traces information flow on the frozen transformer in one forward and one backward pass without requiring counterfactual examples. LTP analytically decomposes and linearizes the computational structure of the Transformers into distinct pathways along which it propagates a targeted unembedding vector to receive the effective representation at each residual position. This target-centric propagation achieves O(1) time complexity with respect to the number of model components, scaling to long in
    
[^120]: 为什么更好的跨语言对齐无法带来更好的跨语言迁移：以编码器为例

    Why Better Cross-Lingual Alignment Fails for Better Cross-Lingual Transfer: Case of Encoders

    [https://arxiv.org/abs/2603.18863](https://arxiv.org/abs/2603.18863)

    本研究通过对四个语言对上采用词元级、句子级和掩码语言建模等目标进行显式对齐的XLM-R编码器进行实验，发现基于嵌入的对齐指标无法可靠预测下游跨语言迁移性能，且对齐目标与下游任务的梯度往往近乎正交，从而揭示了更好的跨语言对齐为何并不必然带来更好的跨语言迁移。

    

    跨语言对齐通常被认为可以通过拉近不同语言的表示来改善跨语言迁移。然而，表示对齐方面的改进并不总能一致地转化为更好的下游性能。我们利用在四个语言对上分别以词元级、句子级和掩码语言建模目标进行显式对齐的XLM-R模型来研究这种脱节现象。我们在一个词元级任务（词性标注）和一个句子级任务（句子分类）上评估了这些模型的零样本迁移能力，并分析了对齐目标和下游目标所引起的表示变化与梯度。我们发现，基于嵌入的对齐指标并不能可靠地预示对齐会改善还是损害下游性能。此外，对齐梯度与下游任务梯度往往近乎正交，尤其当对齐目标与下游任务……（摘要原文在此处截断）

    arXiv:2603.18863v2 Announce Type: replace  Abstract: Cross-lingual alignment is often assumed to improve cross-lingual transfer by bringing representations of different languages closer together. However, improvements in representational alignment do not consistently translate into better downstream performance. We investigate this disconnect using XLM-R models explicitly aligned across four language pairs with token-level, sentence-level, and masked-language-modeling objectives. We evaluate their zero-shot transfer on a token-level task (part-of-speech tagging) and a sentence-level task (sentence classification), and analyze both representational changes and the gradients induced by the alignment and downstream objectives. We find that embedding-based alignment metrics do not reliably indicate whether alignment will improve or degrade downstream performance. Moreover, alignment and downstream-task gradients are often nearly orthogonal, particularly when the alignment objective and dow
    
[^121]: GT-HarmBench：通过博弈论视角评估人工智能安全风险

    GT-HarmBench: Benchmarking AI Safety Risks Through the Lens of Game Theory

    [https://arxiv.org/abs/2602.12316](https://arxiv.org/abs/2602.12316)

    该论文提出GT-HarmBench——首个基于博弈论结构、包含1535个高风险场景的多智能体AI安全基准，发现前沿模型在38%的高风险场景中无法选择对社会有益的行动，而博弈论干预可将有益结果提升最高18%。

    

    前沿人工智能系统的能力日益增强，并被部署于高风险的多智能体环境中。然而，现有的人工智能安全基准主要评估单个智能体，导致对协调失败和冲突等多智能体风险的理解严重不足。我们提出了GT-HarmBench，这是一个包含1,535个高风险场景的基准，涵盖囚徒困境、猎鹿博弈和胆小鬼博弈等博弈论结构。这些场景取材于MIT人工智能风险知识库中的现实AI风险情境。在对15个前沿模型的评估中，智能体在38%的高风险案例（如军事升级、选举操纵和医疗事故）中未能选择对社会有益的行动。我们测量了模型对博弈论提示框架和顺序的敏感性，并分析了导致失败的推理模式。我们进一步表明，博弈论干预可以将社会有益结果提升至多18%。我们的结果凸显了前沿AI系统在多智能体环境中显著的可靠性问题。

    arXiv:2602.12316v3 Announce Type: replace  Abstract: Frontier AI systems are increasingly capable and deployed in high-stakes multi-agent environments. However, existing AI safety benchmarks largely evaluate single agents, leaving multi-agent risks such as coordination failure and conflict poorly understood. We introduce GT-HarmBench, a benchmark of 1,535 high-stakes scenarios spanning game-theoretic structures such as the Prisoner's Dilemma, Stag Hunt and Chicken. Scenarios are drawn from realistic AI risk contexts in the MIT AI Risk Repository. Across 15 frontier models, agents fail to choose socially beneficial actions in 38% of high-stakes cases, such as military escalation, election manipulation, and medical malpractice. We measure sensitivity to game-theoretic prompt framing and ordering, and analyze reasoning patterns driving failures. We further show that game-theoretic interventions improve socially beneficial outcomes by up to 18%. Our results highlight substantial reliabilit
    
[^122]: FlyAOC：评估果蝇科学知识库的智能体本体策展

    FlyAOC: Evaluating Agentic Ontology Curation of Drosophila Scientific Knowledge Bases

    [https://arxiv.org/abs/2602.09163](https://arxiv.org/abs/2602.09163)

    FlyAOC是一个评估AI智能体从科学文献中进行端到端本体策展的基准，要求智能体在16,898篇果蝇论文中检索证据，并恢复策展人级别的结构化基因标注，涵盖功能术语、表达模式和历史同义词。

    

    科学知识库通过将原始文献中的研究发现策展为结构化、可查询的格式，从而加速科学发现，服务于人类研究者和新兴的AI系统。维护这些资源需要专家策展人检索论文、整合跨文档证据，并生成基于本体的标注。现有基准通常只评估孤立的子任务（如命名实体识别或关系抽取），因此无法捕捉这一端到端的工作流程。我们提出FlyAOC，用于评估AI智能体在科学文献上执行端到端智能体本体策展的能力。给定一个基因符号、一段简洁的FlyBase基因描述、一个包含16,898篇论文语料库的访问权限以及本体资源，智能体必须检索证据并尽可能多地恢复与策展人相关的结构化标注。输出涵盖标准化的功能术语、表达模式，以及连接数十年命名历史的历史同义词。

    arXiv:2602.09163v2 Announce Type: replace  Abstract: Scientific knowledge bases accelerate discovery by curating findings from primary literature into structured, queryable formats for both human researchers and emerging AI systems. Maintaining these resources requires expert curators to search papers, reconcile evidence across documents, and produce ontology-grounded annotations. Existing benchmarks usually evaluate isolated subtasks, such as named entity recognition or relation extraction, and therefore do not capture this end-to-end workflow. We present FlyAOC to evaluate AI agents on end-to-end agentic ontology curation from scientific literature. Given a gene symbol, a concise FlyBase gene description, access to a 16,898-paper corpus, and ontology resources, agents must search for evidence and recover as many curator-relevant structured annotations as possible. Outputs span standardized function terms, expression patterns, and historical synonyms linking decades of nomenclature. T
    
[^123]: 面向情感支持对话的情感流语言模型

    Affective Flow Language Model for Emotional Support Conversation

    [https://arxiv.org/abs/2602.08826](https://arxiv.org/abs/2602.08826)

    该论文提出情感流语言模型AFlow，将多轮情感支持建模为沿对话轨迹演化的情感效用流，并通过情感流偏好优化（AFPO）利用流平衡目标将下游偏好信号传播到中间对话状态，为多轮交互中的支持策略调整提供细粒度的过程监督。

    

    大型语言模型在情感支持对话方面取得了进展，但现有的对齐方法主要依赖于回复层面的稀疏偏好或对话层面的结果，这为多轮交互中的顺序性策略决策提供的监督十分有限。这引出了一个关键问题：如何从整体对话结果中推导出细粒度的过程信号，以引导支持策略的逐步调整？我们提出了情感流语言模型，它将多轮情感支持建模为沿对话轨迹演化的情感效用流。AFlow搜索多样化的支持轨迹，并估计中间对话状态和候选策略的效用。它进一步引入了情感流偏好优化，该机制利用定义在对话子路径上的流平衡目标，将下游偏好信号传播到中间状态并学习……

    arXiv:2602.08826v3 Announce Type: replace  Abstract: Large language models (LLMs) have advanced emotional support conversation, but existing alignment methods rely mainly on sparse preferences at the response level or outcomes at the dialogue level, providing limited supervision for sequential strategy decisions in multi-turn interactions. This raises a key question: how can detailed process signals be derived from overall dialogue outcomes to guide the gradual adaptation of support strategies? We propose the Affective Flow Language Model (AFlow), which models multi-turn emotional support as an affective utility flow evolving along dialogue trajectories. AFlow searches diverse support trajectories and estimates the utility of intermediate dialogue states and candidate strategies. It further introduces Affective Flow Preference Optimization (AFPO), which uses a flow-balance objective defined over dialogue subpaths to propagate downstream preference signals to intermediate states and lea
    
[^124]: 大语言模型推理的分步内在奖励

    Stepwise Intrinsic Rewards for Reasoning in Large Language Models

    [https://arxiv.org/abs/2602.01034](https://arxiv.org/abs/2602.01034)

    提出了一种无需过程标注、辅助模型或推理时搜索的内在过程奖励方法——分步边际信息增益（MIG），通过衡量每个推理前缀对参考答案对数似然的提升，并结合单调水位线机制避免重复计分，从而为大语言模型的多步推理提供更精确的密集监督。

    

    强化学习（RL）已成为提升大语言模型（LLM）和视觉-语言模型（VLM）推理能力的广泛使用的范式。然而，稀疏的二值结果奖励仅对最终正确性进行评分，无法识别哪些中间步骤对正确性做出了贡献；在多模态任务中，这类奖励还可能奖励由语言先验而非视觉证据驱动的答案。过程奖励模型（PRM）虽然使监督更加密集，但通常需要过程标注、辅助模型或推理时搜索。在本文中，我们提出了分步边际信息增益（Stepwise Marginal Information Gain, MIG），这是一种由策略自身计算得到的内在过程奖励。MIG 衡量每个结构化推理前缀如何改变参考答案在长度归一化、教师强制条件下的对数似然。单调的历史水位线机制仅奖励新的似然最大值，从而避免在次优迂回路径之后进行重复计分。（注：原摘要在此处被截断）

    arXiv:2602.01034v2 Announce Type: replace  Abstract: Reinforcement learning (RL) has become a widely used paradigm for improving the reasoning abilities of large language models (LLMs) and Vision-language models (VLMs). Sparse binary outcome rewards, however, score only final correctness and cannot identify which intermediate steps contributed to it; in multimodal tasks, they may also reward answers driven by linguistic priors rather than visual evidence. Process reward models (PRMs) densify supervision but usually require process annotations, auxiliary models, or inference-time search. In this paper, we introduce Stepwise Marginal Information Gain (MIG), an intrinsic process reward computed from the policy itself. MIG measures how each structured reasoning prefix changes the length-normalized, teacher-forced log-likelihood of the reference answer. A monotonic historical watermark rewards only new likelihood maxima, avoiding duplicate credit after sub-record detours. We combine this si
    
[^125]: 迈向自动化词典编纂：学习词典释义的生成与评估

    Towards Automated Lexicography: Generating and Evaluating Definitions for Learner's Dictionaries

    [https://arxiv.org/abs/2601.01842](https://arxiv.org/abs/2601.01842)

    该论文提出了基于“LLM作为裁判”的词典释义生成评估方法（配合与专业词典编纂者合作构建的日语释义数据集），以及一种采用迭代简化策略的、基于大语言模型的学习词典释义生成方法。

    

    词典释义是学习词义的重要资源，但人工编写成本高昂。因此，我们研究词典释义生成（DDG），即为给定词头生成非语境化释义的任务。具体而言，我们针对学习词典释义生成（LDDG）这一场景，要求释义使用简单词汇编写。首先，我们提出了一种可靠的DDG评估方法，该方法基于新提出的评估标准，并由“大语言模型作为裁判”（LLM-as-a-judge）技术驱动。为了给评估提供参考释义，我们与一位专业词典编纂者合作构建了一个日语词典释义数据集。验证结果表明，我们的评估方法与人工标注者的一致程度达到了与标注者间一致性相当的水平。其次，我们提出了一种基于大语言模型、采用迭代简化策略的LDDG方法。实验结果表明，我们的方法……（原文在此截断）

    arXiv:2601.01842v2 Announce Type: replace  Abstract: Dictionary definitions are an essential resource for learning word senses, but manually creating them is costly. We thus study dictionary definition generation (DDG), i.e., the generation of non-contextualized definitions for given headwords. Specifically, we address learner's dictionary definition generation (LDDG), where definitions should be written using simple vocabulary. First, we introduce a reliable evaluation approach for DDG, based on newly proposed evaluation criteria and powered by an LLM-as-a-judge. To provide reference definitions for the evaluation, we construct a dataset of Japanese dictionary definitions in collaboration with a professional lexicographer. Validation results demonstrate that our evaluation approach agrees with human annotators at a level comparable to inter-annotator agreement. Second, we propose an LLM-based LDDG approach that employs iterative simplification. Experimental results show that our appro
    
[^126]: 理解能力能否促进统一多模态模型中的生成？从分析到改进路径

    Does Understanding Inform Generation in Unified Multimodal Models? From Analysis to Path Forward

    [https://arxiv.org/abs/2511.20561](https://arxiv.org/abs/2511.20561)

    本文提出解耦评估框架UniSandbox，揭示统一多模态模型中理解与生成之间存在显著差距，并证明理解模块中的显式思维链（CoT）可有效弥合该差距，且可通过自训练将推理能力内化，实现生成时的隐式推理。

    

    近年来，统一多模态模型取得了显著进展，但一个根本性的问题依然存在：理解能力是否真正促进了统一多模态模型中的生成能力？为探究这一问题，我们提出了UniSandbox，一个解耦的评估框架，并配合受控的合成数据集，以避免数据泄漏并实现细致的分析。我们的发现揭示了显著的理解-生成差距，该差距主要体现在两个关键维度：推理生成与知识迁移。具体而言，对于推理生成任务，我们观察到理解模块中的显式思维链能有效弥合这一差距，并进一步证明自训练方法可以成功将这种能力内化，从而在生成过程中实现隐式推理。此外，对于知识迁移任务，我们发现思维链能够通过帮助检索新学习的知识来辅助生成过程。

    arXiv:2511.20561v3 Announce Type: replace-cross  Abstract: Recent years have witnessed significant progress in Unified Multimodal Models, yet a fundamental question remains: Does understanding truly inform generation in Unified Multimodal Models? To investigate this, we introduce UniSandbox, a decoupled evaluation framework paired with controlled, synthetic datasets to avoid data leakage and enable detailed analysis. Our findings reveal a significant understanding-generation gap, which is mainly reflected in two key dimensions: reasoning generation and knowledge transfer. Specifically, for reasoning generation tasks, we observe that explicit Chain-of-Thought (CoT) in the understanding module effectively bridges the gap, and further demonstrate that a self-training approach can successfully internalize this ability, enabling implicit reasoning during generation. Additionally, for knowledge transfer tasks, we find that CoT assists the generative process by helping retrieve newly learned 
    
[^127]: 评估即一切：通过评估设计对大语言模型推理能力的策略性夸大

    Evaluation is All You Need: Strategic Overclaiming of LLM Reasoning Capabilities Through Evaluation Design

    [https://arxiv.org/abs/2506.04734](https://arxiv.org/abs/2506.04734)

    本研究揭示评估条件的细微差异会导致Deepseek-R1-Distill系列等推理模型的基准测试结果大幅波动，使其声称的性能提升难以可靠复现，并倡导建立更严格的模型性能评估范式。

    

    以Deepseek-R1-Distill系列为代表的推理模型因其在数学、科学、编程等领域的出色表现而被开源社区广泛采用。然而，我们的研究揭示，其基准测试评估结果会受到多种因素的影响而产生显著波动，评估条件的细微差异就可能导致结果的重大变化。在基于Deepseek-R1-Distill系列微调的其他开源推理模型以及QwQ-32B模型中也观察到类似现象，使得其所声称的性能提升难以可靠复现。因此，我们倡导建立更严格的模型性能评估范式，并呈现了我们对Deepseek-R1-Distill系列模型的实证评估。

    arXiv:2506.04734v3 Announce Type: replace  Abstract: Reasoning models represented by the Deepseek-R1-Distill series have been widely adopted by the open-source community due to their strong performance in mathematics, science, programming, and other domains. However, our study reveals that their benchmark evaluation results are subject to significant fluctuations caused by various factors. Subtle differences in evaluation conditions can lead to substantial variations in results. Similar phenomena are observed in other open-source inference models fine-tuned based on the Deepseek-R1-Distill series, as well as in the QwQ-32B model, making their claimed performance improvements difficult to reproduce reliably. Therefore, we advocate for the establishment of a more rigorous paradigm for model performance evaluation and present our empirical assessments of the Deepseek-R1-Distill series models.
    
[^128]: 从ASR到ASP：评估针对开源大语言模型的提示攻击脆弱性

    From ASR to ASP: Evaluating Prompt Attack Vulnerabilities Against Open-Source LLMs

    [https://arxiv.org/abs/2505.14368](https://arxiv.org/abs/2505.14368)

    本文提出攻击成功概率（ASP）这一新指标，以捕捉传统攻击成功率所忽视的模型响应不确定性，并在五个攻击基准上系统评估了14个开源和3个闭源大语言模型面对提示注入攻击的脆弱性。

    

    近期研究表明，大语言模型（LLM）容易受到诱导其生成有害或敏感输出的攻击。随着开源大语言模型在金融、法律和医疗等高影响领域应用中的采用日益增多，系统性地研究其安全风险对于迈向可信赖的大语言模型时代变得愈发重要。本文全面研究了在五种攻击基准上，针对14个广泛使用的开源大语言模型和3个闭源大语言模型的有效提示注入攻击。此外，现有评估指标大多仅考虑攻击成功率，忽视了模型响应中的不确定性。我们提出的攻击成功概率（ASP）额外捕捉了评估中的不确定行为，即模型可能最初拒绝有害请求但随后又提供有害指导，或者情况相反，这反映了攻击可行性中存在的不一致性与模糊性。

    arXiv:2505.14368v3 Announce Type: replace-cross  Abstract: Recent studies demonstrate that Large Language Models (LLMs) are vulnerable to attacks that generate harmful or sensitive outputs. As open-source LLMs are increasingly adopted in high-impact applications such as finance, law, and healthcare, systematically investigating their security risks is becoming increasingly important towards a trustworthy LLM era. This paper comprehensively studies effective prompt injection attacks against 14 widely used open-source and three closed-source LLMs on five attack benchmarks. Moreover, existing evaluation metrics mostly only consider the attack success rate, overlooking uncertainty in model responses. Our proposed Attack Success Probability (ASP) additionally captures uncertain behaviors for evaluation, where the model may initially refuse a harmful request but subsequently provide harmful guidance or vice versa, reflecting inconsistency and ambiguity in attack feasibility. By systematicall
    
[^129]: 通过启发式适配与超词学习实现语言模型的分词器灵活性

    Achieving Tokenizer Flexibility in Language Models through Heuristic Adaptation and Supertoken Learning

    [https://arxiv.org/abs/2505.09738](https://arxiv.org/abs/2505.09738)

    提出了 Tokenadapt——一种与模型无关的分词器移植方法，结合面向多词“超词”的预分词学习，使语言模型能以较低计算成本灵活替换分词器，同时提升压缩效率并减少词元碎片化。

    

    预训练语言模型（LLMs）常常受限于其固定的分词方案，导致效率低下和性能受限，尤其是在多语言或专业化应用场景中。这种“分词器锁定”问题带来了重大挑战：克服该问题的标准方法通常需要极高的计算资源。尽管通过启发式初始化进行分词器替换旨在减轻这一负担，但现有方法往往需要进行穷举式的残差微调，且仍可能无法完整保留语义细节或充分解决底层的压缩低效问题。我们的框架引入了两项创新：其一，Tokenadapt，一种与模型无关的分词器移植方法；其二，新颖的预分词学习方法，用于多词“超词”（Supertokens），以增强压缩效果并减少碎片化。Tokenadapt 通过一种结合两种……的混合启发式方法来初始化新的独特词元嵌入。

    arXiv:2505.09738v2 Announce Type: replace-cross  Abstract: Pretrained language models (LLMs) are often constrained by their fixed tokenization schemes, leading to inefficiencies and performance limitations, particularly for multilingual or specialized applications. This tokenizer lock-in presents significant challenges. standard methods to overcome this often require prohibitive computational resources. Although tokenizer replacement with heuristic initialization aims to reduce this burden, existing methods often require exhaustive residual fine-tuning and still may not fully preserve semantic nuances or adequately address the underlying compression inefficiencies. Our framework introduces two innovations: first, Tokenadapt, a model-agnostic tokenizer transplantation method, and second, novel pre-tokenization learning for multi-word Supertokens to enhance compression and reduce fragmentation. Tokenadapt initializes new unique token embeddings via a hybrid heuristic that combines two me
    
[^130]: MedHal：一个用于医学幻觉检测的合成数据集

    MedHal: a Synthetic Dataset for Medical Hallucination Detection

    [https://arxiv.org/abs/2504.08596](https://arxiv.org/abs/2504.08596)

    MedHal是一个涵盖内在与外在幻觉的大规模合成医学数据集，用于训练和评估医学文本幻觉检测模型，基于该数据集训练的基线模型优于通用幻觉检测方法。

    

    幻觉是指AI系统生成非事实性内容的现象，在医学场景中会带来严重风险，因为错误可能直接影响患者的治疗效果。我们提出了MedHal，这是一个大规模数据集，专门用于评估模型能力并训练模型执行医学文本中的幻觉检测任务。当前的幻觉检测方法在应用于医学等专业领域时面临重大局限，可能造成灾难性后果。MedHal通过纳入多样化的医学文本来源和任务（涵盖内在幻觉与外在幻觉），并提供大量适合训练医学幻觉检测模型的数据样本来解决这一问题。我们通过训练和评估一个基线医学幻觉检测模型证明了MedHal的实用性，其表现优于通用幻觉检测方法。该资源为医学领域的幻觉检测研究和应用提供了重要支持。

    arXiv:2504.08596v3 Announce Type: replace-cross  Abstract: Hallucination, the generation of non factual content by AI systems, poses serious risks in medical contexts, where errors can directly affect patient outcomes. We present MedHal, a large-scale dataset specifically designed to assess capabilities and train models on the task of hallucination detection in medical texts. Current hallucination detection methods face significant limitations when applied to specialized domains like medicine, where they can have disastrous consequences. MedHal addresses this issue by incorporating diverse medical text sources and tasks covering both intrinsic and extrinsic hallucinations, and by providing a substantial volume of data samples suitable for training medical hallucination detection models. We demonstrate MedHal's utility by training and evaluating a baseline medical hallucination detection model, showing improvements over general-purpose hallucination detection approaches. This resource e
    
[^131]: SkillFlow：可扩展且高效的智能体技能检索系统

    SkillFlow: Scalable and Efficient Agent Skill Retrieval System

    [https://arxiv.org/abs/2504.06188](https://arxiv.org/abs/2504.06188)

    SkillFlow是首个面向智能体技能发现的多阶段检索系统，它将技能获取视为信息检索问题，通过密集检索、交叉编码器重排序和LLM选择四个阶段，从约3.5万个社区技能定义中高效检索出最相关的技能。

    

    AI智能体可以通过在推理时将可复用技能加载到上下文中来扩展其能力，然而为智能体配备过多技能（尤其是无关技能）会降低其性能。随着社区驱动的技能库不断增长，智能体需要一种能够从大型技能库中选择性地仅检索最相关技能的方法。我们提出了SkillFlow，这是首个开放的多阶段智能体技能发现检索系统，它将技能获取构建为一个信息检索问题，其语料库包含从GitHub索引的约35K个社区贡献的SKILL.md技能定义。该流水线通过四个阶段（密集检索、两轮交叉编码器重排序以及基于LLM的选择）逐步收窄大规模候选集，在每个阶段平衡召回率与精确率。我们在两个编程基准上评估了SkillFlow：SkillsBench（包含87个任务和229个匹配技能的基准）和Terminal-Bench（一个仅提供8个……的基准，摘要原文在此处截断）。

    arXiv:2504.06188v3 Announce Type: replace  Abstract: AI agents can extend their capabilities at inference time by loading reusable skills into context, yet equipping an agent with too many skills, particularly irrelevant ones, degrades performance. As community-driven skill repositories grow, agents need a way to selectively retrieve only the most relevant skills from a large library. We present SkillFlow, the first open, multi-stage retrieval system for agent skill discovery that frames skill acquisition as an information retrieval problem over a corpus of ~35K community-contributed SKILL.md definitions indexed from GitHub. The pipeline progressively narrows a large candidate set through four stages (dense retrieval, two rounds of cross-encoder reranking, and LLM-based selection), balancing recall and precision at each stage. We evaluate SkillFlow on two coding benchmarks: SkillsBench, a benchmark of 87 tasks and 229 matched skills; and Terminal-Bench, a benchmark that provides only 8
    
[^132]: NaijaNLP：尼日利亚低资源语言综述

    NaijaNLP: A Survey of Nigerian Low-Resource Languages

    [https://arxiv.org/abs/2502.19784](https://arxiv.org/abs/2502.19784)

    该论文首次对尼日利亚三大主要语言（豪萨语、约鲁巴语、伊博语）的NLP研究现状进行了全面综述，定量评估了现有语言资源，发现在293项相关研究中仅27.6%贡献了新语言资源，揭示了该领域过度依赖复用现有数据而非创建新数据的关键挑战。

    

    尼日利亚拥有超过500种语言，其中三种语言——豪萨语、约鲁巴语和伊博语——的使用者超过1.75亿人，约占该国语言人口的65%。然而，由于缺乏足够的数字资源来支持计算语言学任务，这些语言被归类为低资源语言。尽管已有许多研究工作和倡议被提出，但对于从语法形式化到支持模型开发的语言资源这一经典自然语言处理（NLP）领域的整体状况，仍缺乏连贯系统的认识。本研究首次对尼日利亚三大主要语言的NLP研究现状（NaijaNLP）进行了全面综述。我们定量评估了现有可用的语言资源，并识别了关键挑战。在被综述的293项研究中，27.6%的研究贡献了新的语言资源。这一发现突显了该领域严重依赖重新利用现有数据，而非创建新数据的现状。

    arXiv:2502.19784v3 Announce Type: replace  Abstract: With over 500 languages in Nigeria, three languages - Hausa, Yor\`ub\'a and Igbo spoken by more than 175 million people, account for about 65% of the languages. However, these languages are classed as low-resource due to insufficient digital resources to support tasks in computational linguistics. While several research efforts and initiatives have been presented, a coherent understanding of the state of classic Natural Language Processing (NLP) spanning grammatical formalisation to linguistic resources that support models development is lacking. This study presents the first comprehensive review of the state of affairs in NLP research across the three major Nigerian languages (NaijaNLP). We quantitatively assess the available linguistic resources and identify key challenges. Of the 293 reviewed studies, 27.6% contributed new linguistic resources. This finding highlights a strong reliance on repurposing existing data rather than crea
    

