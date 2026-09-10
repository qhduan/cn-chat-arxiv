# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [IdeaAMBIG: Benchmarking Implementation-Critical Gaps in Research-Idea Specifications](https://arxiv.org/abs/2609.10539) | 该论文提出IdeaAMBIG基准，包含660个有证据支撑的实例（163个真实差距和497个受控合成差距），用于衡量研究想法规范中影响忠实实现的信息缺口，并从成文就绪度评估、缺陷定位和澄清行动生成三个维度进行评估。 |
| [^2] | [IBIB: A Protocol for Measuring Enterprise AI Systems by Serving Route, Not Model Identifier](https://arxiv.org/abs/2609.10494) | 该论文提出IB2协议，将现有基准测试仅按模型标识符评分视为测量误差，通过金标准盲能力绑定预检、包含可靠性的首轮评分规则和分数盲裁决三部分，实现以实际服务路由（而非模型标识符）来衡量企业AI系统的真实可用能力。 |
| [^3] | [Building Multilingual Bridges: Data Mixing as the Pillar of Generalization for In-Language Reasoning](https://arxiv.org/abs/2609.10445) | 该研究通过在监督微调中优化数据组成与调度策略，构建了3.35B规模的Tiny Aya L2-Thinker模型，使其能在60种语言、6个基准测试上实现超过93%的语内推理率，从而弥合提示与答案之间的语言鸿沟。 |
| [^4] | [ConvMem: Convolutional Memory for Long-Context Reasoning](https://arxiv.org/abs/2609.10441) | ConvMem 提出了一种无需训练、高度可并行化的分层卷积框架，将被特定查询提示的 LLM 视为卷积核对文本进行层次化摘要，将长上下文推理路径从线性链缩短为对数树，克服了序列化记忆方法高延迟和依赖昂贵强化学习训练的缺陷。 |
| [^5] | [Do speech foundation models really learn words?](https://arxiv.org/abs/2609.10434) | 本研究通过残差化剥离音素信息，证明HuBERT和wav2vec 2.0在较深层中确实学到了独立于局部语音内容的词语表征。 |
| [^6] | [Can Foundation Models Moderate Online Content? Evaluating Instruction- vs. Example-Driven Policy Operationalization](https://arxiv.org/abs/2609.10410) | 本文提出包含4,000条Bluesky人工标注帖子的新基准ModerationBench，系统比较了指令驱动与示例驱动两种政策操作化范式，发现基础模型的内容审核F1分数可达Bluesky现有审核系统的近三倍（0.60 vs. 0.22）。 |
| [^7] | [Retrofitting Code Using LLMs to Support Exceptional Behavior](https://arxiv.org/abs/2609.10397) | 该论文提出了一项新任务——为现有代码自动改造补充异常相关代码（ERC），并设计了EXCODER工具，通过上下文工程技术将静态与动态程序分析和大语言模型相结合，自动生成缺失的ERC使异常行为测试通过。 |
| [^8] | [Rosetta at AlexandriaX-2026: LoRA-Adapted NileChat for Context-Aware Dialectal Arabic Dialogue Translation](https://arxiv.org/abs/2609.10395) | Rosetta 系统通过在 NileChat-3B 上微调 LoRA 适配器并利用结构化提示实现情境感知的英语到阿拉伯语方言对话翻译，在受限和非受限赛道中分别获得第 4 和第 5 名，并发现外部方言数据预训练会引发负迁移。 |
| [^9] | [Why Is Video Still So Expensive? A Survey of Inference-Efficiency Mechanisms in Video and Audiovisual LLMs](https://arxiv.org/abs/2609.10355) | 本综述按流水线阶段（帧采样、模态编码、token缩减、LLM预填充与解码）系统梳理了视频与视听大语言模型中降低参数量、计算量、延迟、内存和token数量的推理效率优化机制。 |
| [^10] | [From Symbolic Perception to Logical Deduction: A Framework for Guiding Language Models in Geometric Reasoning](https://arxiv.org/abs/2609.10335) | 该论文提出一个将几何图形解析为符号形式并进行形式化逻辑推演的框架，使纯大语言模型在几何推理上达到与最先进多模态模型相当的性能，同时减少幻觉并提升推理的可解释性。 |
| [^11] | [On-Policy Distillation for Vision-Language Model Adaptation, an Effective Paradigm on Low-Quality Multimodal Data](https://arxiv.org/abs/2609.10321) | 提出OnPoKD框架，首次将在策略蒸馏应用于视觉-语言模型适配，通过轻量级控制器动态构建样本级自适应蒸馏目标，从而在低质量多模态数据上实现更可靠的模型迁移。 |
| [^12] | [RiLM: Parameter-Efficient Language Modeling via Geodesic Decoding](https://arxiv.org/abs/2609.10305) | 提出RiLM框架，完全移除传统输出矩阵，通过在黎曼流形上计算当前状态与词表嵌入之间的测地线距离平方来直接解码下一个词元概率，其双曲版本HypRiLM在约29万参数下于WikiText-2上达到54.2的验证困惑度，显著优于平坦版本及同规模的LSTM、Transformer和SSM基线。 |
| [^13] | [The Semantic Bottleneck: Leveraging Semantic Representations for Non-Invasive Speech Decoding](https://arxiv.org/abs/2609.10296) | 提出Brain2Semantics2Text方法，通过语义嵌入空间作为瓶颈，将句子级MEG信号映射到语义流形并逆向转换为文本，实现了无需词级对齐的非侵入式语音解码。 |
| [^14] | [GANDR: Claim Auditing for Verifiable Legal Answer Generation](https://arxiv.org/abs/2609.10293) | GANDR是一个双智能体系统，由起草器生成结构化法律答案、批评者逐条对照引用来源审计论断，并配合要求每条引用必须命中检索结果的严格正确性标准，从而实现可逐条验证的法律答案生成。 |
| [^15] | [KVShareArena: KV-Cache Reuse Across Contexts and Model Checkpoints](https://arxiv.org/abs/2609.10266) | KVShareArena是首个针对跨提示上下文和模型检查点的KV缓存复用方法进行系统基准测试的平台，通过衡量各方法在无缓存与完整缓存之间性能差距的恢复比例来评估RAG检索片段和多智能体报告场景下的缓存修复技术。 |
| [^16] | [DiSCo: A Distribution-First Steering and Cultural Prior Evaluation Framework for Measuring Cultural Preference Bias in LLMs](https://arxiv.org/abs/2609.10253) | DiSCo是一个分布优先的评估框架，通过隔离默认文化先验并利用四级上下文梯度测试可引导性，发现大语言模型的文化偏好先验严重集中在英美文化上（合计约占35%）。 |
| [^17] | [Two-Token Features and Small-Large Ensembles for VLM Hallucination Detection](https://arxiv.org/abs/2609.10244) | 该论文提出将微调的40亿参数小模型（读取自身隐藏状态的双词元特征）与约4000亿参数的零样本大模型评判器集成，用于视觉语言模型的字符级幻觉检测，并结合大模型生成的合成幻觉数据增强集成多样性，在SHROOM-Visions 2026共享任务四种语言中均取得前八名的成绩。 |
| [^18] | [LiteRAG: Cost-Efficient Graph-Based Retrieval-Augmented Generation](https://arxiv.org/abs/2609.10239) | LiteRAG通过查询条件算法探索和推理链上下文构建取代昂贵的检索时LLM控制，在多跳问答质量上达到最优，同时将查询延迟降低100倍以上、成本降低99%以上、token使用量减少约14倍。 |
| [^19] | [The Answer Path and the Grounding Instruction in LLM Question Answering over Knowledge Graphs](https://arxiv.org/abs/2609.10237) | 本研究在六个大语言模型和两个知识图谱问答基准上系统变化四种提示设计选择，发现只有将答案路径纳入提示和接地指令的设计能显著影响答案准确率（非路径三元组可被无关内容替换而不影响效果，检索预算应全部用于保证召回率），而三元组的书写语法与排列顺序几乎没有作用。 |
| [^20] | [$\Phi$-Bench: Can Large Language Models Engineer the Infrastructure That Powers Them?](https://arxiv.org/abs/2609.10226) | 提出了Φ-Bench基准，通过源自前沿研究和真实代码仓库的任务，系统性评估大语言模型在工程化构建其自身基础设施栈方面的开放式、长周期工程能力。 |
| [^21] | [Through the Looking Glass: Directly Reading and Writing Transformers](https://arxiv.org/abs/2609.10210) | 该研究通过对组件贡献进行符号化净值分析，发现transformer的一次预测实际仅依赖8至53个关键组件，且预测所动用的模型比例（仅1%-3%）不随模型规模增长，所有结果均直接从模型自身的参数与激活中读取。 |
| [^22] | [Politics of Feelings: Emotional Expression and Legislative Effectiveness in the U.S. Congress](https://arxiv.org/abs/2609.10198) | 本研究基于Transformer情感分类器分析了1973至2024年间美国国会超过170万篇演讲中的八种离散情感，首次系统揭示了国会演讲的情感表达随时间日益增强，且与政策领域、议员意识形态立场及立法效力存在显著关联。 |
| [^23] | [Who Argues What? Joint Argument-Entity Detection and Classification in Political Debates](https://arxiv.org/abs/2609.10192) | 本文提出了实体增强的政治辩论数据集DNE-ElecDeb，并引入生成式联合标注框架JAET，首次实现了政治辩论中论证片段与辩论命名实体的联合检测与分类，填补了该领域数据和方法上的空白。 |
| [^24] | [From Retrieval to Weights: Parametric Individualization of Small Language Models with Individual Text Corpora](https://arxiv.org/abs/2609.10155) | 该研究通过DoRA微调将515名参与者的个人搜索历史写入小语言模型权重，证明适配器能显著编码个人语料（个体化效应dz=1.27），但在通用知识测试中模型获得的是知识而非与个体的对齐。 |
| [^25] | [YallaMorph: A Benchmark for Evaluating Arabic Morphological Generation in Large Language Models](https://arxiv.org/abs/2609.10153) | 本文提出YallaMorph——一个涵盖60万条目的大规模阿拉伯语形态生成基准，评估结果显示大语言模型在阿拉伯语形态生成上仍然存在显著困难，尤其是附着词素化、未见和形态罕见的形式。 |
| [^26] | [Active Adaptation, Not Static Defense: Temporal Dynamics of Preventative Steering in Adversarial Fine-Tuning](https://arxiv.org/abs/2609.10142) | 该研究揭示预防性引导的持久防护源于早期的主动补偿性适应而非静态权重偏移，并据此提出渐进式干预方法来强化对抗性微调防御。 |
| [^27] | [If It's Not Buggy, Don't Fix It: On the Dynamics of Iterative Bug-fixing with LLMs](https://arxiv.org/abs/2609.10123) | 研究发现大语言模型在迭代修复中会对无错误程序“无中生有”地报错，修复率低于破坏率，并常陷入无限增删相同更改的伪修复循环，其机制根源是模型内部“错误代码”表示被错误激活。 |
| [^28] | [ProbPlug: A Plugin Uncertainty Network for Reliable Confidence in LLM Binary Classification](https://arxiv.org/abs/2609.10122) | ProbPlug是一个轻量级的插件式置信度估计框架，通过自注意力模块聚合冻结LLM的内部token特征来预测分类输出是否正确，无需修改基础模型即可显著提升LLM二分类的置信度可靠性与分类性能。 |
| [^29] | [Data-Centric Post-Training for Financial Reasoning: Mining, Distillation, and Verifiable Learning](https://arxiv.org/abs/2609.10113) | 该论文提出了一套以数据为中心的后训练流水线，通过挖掘开源推理轨迹、蒸馏金融指令数据和生成知识图谱引导的问答对来构建互补语料库，并利用轻量级分类器筛选数据与基于规则的可验证强化学习，从而有效提升大模型在金融推理任务上的能力。 |
| [^30] | [RAP: Research Attention Prediction Reveals Target-Conditioned Evidence Acquisition Biases](https://arxiv.org/abs/2609.10092) | 提出了RAP滚动基准来评估LLM智能体预测研究关注度变化的能力，发现其表现不如简单的EWMA精确计数基线，并揭示了状态前推优于直接预测、以及面向预测的策略倾向于检索较旧证据这两大瓶颈。 |
| [^31] | [NOPE-HYPE: A Structured Simulation Workflow for Robust Speech-to-Text Across Diverse Acoustic Environments](https://arxiv.org/abs/2609.10058) | NOPE-HYPE提出了一种结合可控环境模拟器、基于PSD模板的环境约简和超参数搜索的结构化训练工作流，证明模拟噪声可使Whisper和SeamlessM4T等语音翻译模型达到与真实噪声训练相当的性能。 |
| [^32] | [OntologyAligner: Ontology-Aligned Retrieval and Hierarchy-Guided Large Language Model Reranking for Biomedical Ontology Normalization](https://arxiv.org/abs/2609.10055) | 提出三阶段框架OntologyAligner（本体对齐检索、大语言模型候选重排序与层次引导精炼），并构建含13,390个样本的统一基准PhenoNormBench，在人类表型本体规范化任务上达到最先进性能。 |
| [^33] | [Direct Diversity Optimization for Diverse Successful Trajectories in Preference Post-Training](https://arxiv.org/abs/2609.10052) | 提出了一种名为DDO的离线后训练方法，通过分歧树收集与参考相对目标几率目标相结合，使大语言模型智能体在固定预算下保留并实现多样化的成功策略，在多个环境中显著提升了任务成功率和成功策略覆盖。 |
| [^34] | [MedDeID enables locally governed clinical-text de-identification from real or synthetic training data](https://arxiv.org/abs/2609.10049) | MedDeID是一个本地部署的临床文本去标识化框架，无论使用机构内真实数据还是完全合成数据训练，都能高效检测并删除个人可识别信息，使临床笔记在不出机构的前提下安全地用于科研和医疗AI开发。 |
| [^35] | [Deterministic Prompting for Speaker-Stable Low-Resource Greek TTS](https://arxiv.org/abs/2609.10022) | 该论文通过WhisperX数据整理、确定性提示替代LLM风格提示以及轻量级LoRA微调，仅用3.5小时单说话人数据就实现了说话人一致性接近人类水平的低资源希腊语TTS系统。 |
| [^36] | [MetroLLM-Bench: Evaluating Language Models as Transit Kiosk Runtimes](https://arxiv.org/abs/2609.10016) | 提出了MetroLLM-Bench基准（包含955个案例），首次系统性地将语言模型作为地铁信息亭策略层进行评估，涵盖六大真实地铁系统中路线规划、票价计算、运营中断、无障碍服务和对抗性输入等11个类别，并通过确定性评分与语义评分双层机制对26个模型进行排名。 |
| [^37] | [SalamandraTA at WMT 2026 Terminology Shared Task: Hard Examples Are Better Teachers](https://arxiv.org/abs/2609.09999) | 该论文提出在术语感知翻译的微调中只保留模型自身译文与术语表相矛盾的“难例”，仅此筛选即可在固定数据量下将术语准确率从 78.7% 提升至 89.9%，并据此构建了 SalamandraTA-7b-instruct v3.0，作为 BSC 参加 WMT26 术语共享任务赛道 1 的提交系统。 |
| [^38] | [Stable Answers, Unfinished Reasoning: Why Self-Consensus Is Not a Safe Early-Exit Signal](https://arxiv.org/abs/2609.09989) | 研究发现自我共识不是安全的推理提前退出信号——答案一致只说明答案稳定而非推理已完成（共识-终止差距），3,520条共识规则均未通过预设的安全与省token验收门槛，而基于边界置信度的DEER方法则全部通过。 |
| [^39] | [VLX-VR: An Agentic-Aware Video Reasoning Model](https://arxiv.org/abs/2609.09985) | VLX-VR 提出了一个基于“思考—记忆—观测”循环的智能体感知视频推理模型，通过强化学习掌握证据获取、记忆使用与终止决策，在 MINERVA 基准上以 78.79% 的准确率取得最先进性能。 |
| [^40] | [Multi-Functional Embedding Models for Funder Name Disambiguation in Scientific Publication Records](https://arxiv.org/abs/2609.09984) | 本文提出了一个基于多任务学习的多语言、多功能资助者名称消歧模型框架，通过整合ROR、WoS和OFR数据集构建训练数据，应用于生物多样性保护领域研究出版物的分析。 |
| [^41] | [Towards Stress-Aware Sentence-Level Filipino G2P With Weakly-Supervised ByT5 Fine-Tuning](https://arxiv.org/abs/2609.09974) | 本文提出在维基词典数据引导的LLM辅助标注流水线构建的句子级数据集上，对基于ByT5的多语言G2P预训练模型进行弱监督微调，实现了压力感知的句子级菲律宾语字素到音素转换。 |
| [^42] | [5-Dialects-BN: Unmasking the Impact of Transliteration on Bangla Dialectal LLMs](https://arxiv.org/abs/2609.09964) | 该论文提出了首个多标注孟加拉方言基准数据集5-Dialects-BN，包含6,000条人工标注条目并覆盖五种主要方言，通过将罗马化音译与方言文本、标准孟加拉语、英语及主观性标签对齐，揭示了音译对孟加拉方言大语言模型性能的影响。 |
| [^43] | [Improving Cross-Lingual Token Representations by Adding a Pinch of SALT](https://arxiv.org/abs/2609.09953) | SALT是一种轻量级后训练方法，通过向现有跨语言句子编码器注入跨度级监督信号来改进词元表示，在五个多语言词元级基准中的四个上取得最佳结果，同时还能提升句子级任务性能。 |
| [^44] | [Vague2Detect: Handling Ambiguous Prompts in Knowledge-Based Open-World Detection](https://arxiv.org/abs/2609.09949) | 提出Vague2Detect混合流水线，通过微调Sentence-BERT从知识库检索候选、YOLO-World验证图像存在性、GPT-3.5-turbo动态扩展知识库，有效解决了传统检测器无法处理模糊提示的问题。 |
| [^45] | [Contrastive Projection: Reading Transformer Internals by Differencing Logit Lenses](https://arxiv.org/abs/2609.09902) | 提出对比投影方法，通过差分两个相近提示词的隐藏状态并经反嵌入投影来抵消共享的通用成分，构建无需训练的追踪器，从而可靠地读取Transformer在每个位置、子层和注意力头上真正区分输入的内部信息流。 |
| [^46] | [Deep and shallow biases in language models](https://arxiv.org/abs/2609.09901) | 该论文提出“偏见深度分数”这一新指标，将语言模型偏见区分为源自预训练且难以消除的深层偏见和依赖提示措辞的浅层偏见，并发现仅有约四分之一的集中偏好属于深层偏见。 |
| [^47] | [Strangers to Themselves: What Language Models Say About Themselves Is Generic](https://arxiv.org/abs/2609.09899) | 语言模型缺乏真正的自我认知——它们对自身行为的描述是泛化性的，其预测效果与关于“AI智能体总体”的描述或其他模型对它的预测相比并无优势，即模型所说的关于自己的内容并非真正关于其本身。 |
| [^48] | [Leveraging Fine-grained Error Correction in Korean Speech Recognition for Consultation Services](https://arxiv.org/abs/2609.09889) | 该论文提出了首个面向对话级ASR纠错的大规模韩语基准数据集DasanCallDial，通过细粒度文本纠错方法解决呼叫中心场景中因隐私限制无法访问音频时的语音识别纠错难题。 |
| [^49] | [When Does Defendant Statement Matter? A Study of Bias and Persuasion in LLM-Simulated Jurors](https://arxiv.org/abs/2609.09887) | 该论文提出了JuryBench基准，通过分析20个前沿LLM的43.2万个陪审员决策，系统研究了被告法庭陈述如何通过说服力、意识形态偏见和背景亲和力影响LLM模拟陪审员的判决严重程度。 |
| [^50] | [$S^3$-Bench: Evaluating Speech Interaction Models as Scientific Voice Assistants](https://arxiv.org/abs/2609.09852) | 本文提出S³-Bench，一个覆盖10个学科的系统性评估框架，通过将对话轮次分解为语音识别、感知、知识推理和发音等阶段，来评估语音交互模型作为科学语音助手的能力。 |
| [^51] | [HyperTrace: Hypothesis-Based Preference Tracing for Online LLM Personalization](https://arxiv.org/abs/2609.09835) | HyperTrace是一个免训练的在线大语言模型个性化框架，通过维护并借助SMC风格重加权动态更新短期意图与长期偏好的可解释自然语言假设，实现潜在偏好追踪，无需参数更新即可显著提升响应对齐、偏好预测和用户画像一致性。 |
| [^52] | [UnitBoost: Managing Compound LLM Systems with a Merge Operator, Not a Model](https://arxiv.org/abs/2609.09815) | UnitBoost用确定性的合并算子取代复合LLM系统中的生成式元代理，通过槽位-值提案、约束argmax和显式残差机制实现顺序无关、可溯源的系统协调，并在基准测试中超越了金标准标签选出的最佳单一候选。 |
| [^53] | [How Fragile Is Safety Alignment at Frontier Scale? A Single-Direction Attack on a 320B MoE](https://arxiv.org/abs/2609.09793) | 方向消融攻击成功迁移至320B参数的MoE模型GLM-5.3-Flash，证明前沿规模下安全对齐依然脆弱，但拒绝方向在超连接残差和量化架构中的分布位置与稠密模型显著不同。 |
| [^54] | [MUCnoHARM@GermEval Shared Task 2026: Retrieval-based In-Context Learning for Defamatory Offences, and Where It Falls Short](https://arxiv.org/abs/2609.09791) | 该论文研究了基于检索的上下文学习方法在检测德国刑法诽谤犯罪中的应用，发现检索策略相比随机示例收益甚微、模型选择才是最关键因素，且模型仍会遗漏26-57%的犯罪相关帖子，更适合用于人工分流而非自主审核。 |
| [^55] | [LogiScope-VQA: Benchmarking Vision-Language Models for Logistics Hazard Identification in Industrial Scenarios](https://arxiv.org/abs/2609.09790) | 该论文构建了基于真实物流园区数据的多模态基准测试LogiScope-VQA，通过2,476张图像、2,918个视频和10,274个人工精心标注的VQA，围绕工业要素感知、仓储知识理解和潜在风险推理三大主题的39个子任务，系统评估主流大模型在物流危险识别中的实际能力。 |
| [^56] | [ROAM: Robust Organization of Atomic Memories for Agents through Semantic Relations](https://arxiv.org/abs/2609.09778) | ROAM提出了一种关系引导的框架，通过将原子记忆对分类为独立、等价、包含或冲突四种语义关系来组织智能体记忆，既保持了原子化管理的精确性，又能通过融合机制在回答时生成更丰富的非原子视图。 |
| [^57] | [SymbolicLight V2: Hybrid Neuromorphic Architecture and Sparse Execution for Low-Energy Language Inference](https://arxiv.org/abs/2609.09772) | 该论文提出 SymbolicLight V2 混合神经形态语言架构，通过分级带符号事件、无 softmax 局部注意力和稀疏执行技术，在 FPGA 与 ARM CPU 上将解码吞吐量提升约 35%，并将每 token 能耗最高降低 27.7%，实现低能耗语言推理。 |
| [^58] | [Fine-Tuning a KV Cache Concatenation-Aware Model or Recomputing KV Caches? Why Not Both?](https://arxiv.org/abs/2609.09768) | 提出将KV缓存拼接感知的模型微调与选择性KV缓存重计算相结合的方法，在保持低首token延迟的同时显著提升RAG系统长上下文输入的回复准确性。 |
| [^59] | [CARRE: Counterfactual Action Retrieval and Reason Evaluation for Explainable Churn Prescription](https://arxiv.org/abs/2609.09766) | CARRE是一个结合检索增强候选生成、成本感知反事实评分与大语言模型推理的三阶段框架，不仅识别高流失风险客户，还能推荐具体的挽留行动并给出可解释的推荐理由，在电信数据集上相比SHAP基线实现了近80%更高的风险降低。 |
| [^60] | [SocialRL: Refining LLMs' Social Intelligence through Multi-turn Reinforcement Learning and Reward Design](https://arxiv.org/abs/2609.09764) | SocialRL是一个多轮强化学习框架，通过PPO将延迟结果奖励传播回每一轮对话，并设计六个捕捉目标-关系权衡的过程奖励维度，从而提升大语言模型在多轮社交交互中的社交智能。 |
| [^61] | [Can Artificial Intelligence Support Healthcare and Mental Health Through Early Cyberbullying Detection ? The Impact of Emotion-Aware AI on Proactive Online Safety](https://arxiv.org/abs/2609.09735) | 本文提出CareGuard早期预警框架，通过融合零样本语义标注、微调Transformer模型以及情感感知过滤机制，实现对网络欺凌内容的高效早期检测，从而支持医疗保健驱动的心理健康保护与主动式在线安全。 |
| [^62] | [StreamAlign: Streaming Text-Aligned Speech Tokenization](https://arxiv.org/abs/2609.09719) | StreamAlign是一个支持流式处理的文本对齐语音分词框架，通过结合字符级RNN-Transducer对齐、词级ASR指导和主动式词边界分类器，实现了实时语音-文本联合建模，同时缓解了ASR与LLM之间的词表不匹配问题并降低了分词延迟。 |
| [^63] | [Scaling E-Commerce Attribute Extraction with Parallel Decoding](https://arxiv.org/abs/2609.09716) | 提出了一种两阶段LLM流水线，先为每个产品类别发现紧凑且按重要性排序的购买判别属性模式，再利用微调的Qwen3-4B结合超并行解码技术提取属性值，在达到与基础LLM相当的85%提取准确率的同时，将推理成本降低了92%，实现了生产级规模化应用。 |
| [^64] | [When Auditors Fabricate: Batch-Size Degradation and Confident Hallucination in LLM Detection of Planted Document Contamination](https://arxiv.org/abs/2609.09696) | 大语言模型在单文档和小批量污染检测中表现尚可（50%-60%），但在大批量处理时检测率骤降至2.8%，且其失败方式不是承认无法处理，而是自信地捏造包括虚假污染项在内的检测结果。 |
| [^65] | [Looped GPT-BERT: Trading Parameters for Computation in Small Language Modeling](https://arxiv.org/abs/2609.09691) | 该研究提出循环 GPT-BERT，通过深度参数共享让 4 个物理层循环遍历 12 次，在 BabyLM 2026 Strict-small 设定下以仅 1218 万参数在 BLiMP 和 GLUE 等语言学与下游任务指标上取得了与更大参数量的 GPT-2 和 GPT-BERT 基线相当的性能。 |
| [^66] | [Which Medical Questions Deserve Rationales? Perturbation-Sensitive Selection for Robust QA](https://arxiv.org/abs/2609.09684) | 该论文提出RMS-RSP方法，通过仅在推理依据token处扰动隐藏状态并测量答案与干扰项之间裕度的变化，在固定token预算下智能筛选哪些已标注医学问题最值得投入推理依据监督，从而提升医学问答的鲁棒性。 |
| [^67] | [X2-NativeCursor: Native-Token Text Progress Tracking for Incremental-Text Streaming Codec TTS](https://arxiv.org/abs/2609.09677) | 提出X2-NativeCursor，一种无需修改TTS生成器、直接在波形解码前从原生语音令牌跟踪文本进度的轻量级观察器，以极低前瞻延迟实现了高精度的在线文本进度跟踪。 |
| [^68] | [SEA-SpeechBench: A Large-Scale Multitask Benchmark for Speech Understanding Across Southeast Asia](https://arxiv.org/abs/2609.09672) | 该论文提出了首个面向东南亚语言的大规模多任务语音理解基准SEA-SpeechBench，覆盖11种语言、97,194个样本和597小时音频，涵盖语音处理、副语言分析和新颖的时间理解三大类共9项任务。 |
| [^69] | [PELM: Power Efficient On-Device LLM Inference with Speculative Decoding and Dynamic Voltage Frequency Scaling](https://arxiv.org/abs/2609.09662) | 本文提出PELM，一种结合推测解码与动态电压频率调节（DVFS）的端侧大语言模型高能效推理框架，旨在解决移动和边缘平台计算资源受限及散热能力不足导致的降频问题。 |
| [^70] | [Who Are They to Each Other? Multi-Agent Reasoning for Speaker Relationship Inference](https://arxiv.org/abs/2609.09628) | 该论文提出了一种无需训练的多智能体推理框架，通过LLM智能体之间的结构化辩论与裁决机制来推断口语对话中的说话人关系，克服了监督建模成本高和现有推理方法结构化不足的问题。 |
| [^71] | [CityPlanner: A Sandbox Agent for Executable Urban Planning](https://arxiv.org/abs/2609.09578) | CityPlanner 提出了一个基于沙盒环境的可执行城市规划智能体框架，通过统一的文件化环境 UrbanSandbox 和将长轨迹分解为“初始构建”与“反馈改进”两个原子任务的强化学习方法，在真实世界基准上持续优于现有方法。 |
| [^72] | [Beyond Top Words: MonoTM for Topic Modeling with Interpretable Monosemantic Features](https://arxiv.org/abs/2609.09575) | MonoTM是一个可解释的主题建模框架，通过将文档-主题混合估计与语义解释解耦——利用稀疏自编码器完整特征表示估计混合比例，并在基于语料库的语义特征词汇上学习主题描述符——实现了用比单个词汇更有意义的语义单元来表示主题。 |
| [^73] | [Reproducing Omitted Temporal Expressions in Japanese News for Retrieval-Augmented Applications](https://arxiv.org/abs/2609.09569) | 本文提出jaROTE，一个基于规则的流水线，能够在日语新闻被索引进搜索或RAG系统之前，利用发布日期将省略的时间表达（如仅有日期或仅有月份的表述）复原为具体日期或时间区间，实验表明其性能高、速度快、成本低，且与大型语言模型相比仍具竞争力。 |
| [^74] | [Towards Automatic Evolution Tree Generation from Citation Graphs](https://arxiv.org/abs/2609.09561) | EvoTree是一个分阶段框架，通过解耦概念主干学习与时间细化，结合图感知编码器、单调路径约束下的微调和LLM概念标注，首次实现了从引文图自动生成方法演化树，并发布了该任务首个涵盖11个AI子领域的标注基准。 |
| [^75] | [BuzzASR: A Swarm of 100+ Monolingual Speech Recognition Models](https://arxiv.org/abs/2609.09554) | BuzzASR通过将Whisper模型在102种语言上进行单语微调，并结合分词器替换和纯文本数据增强等语言适应策略，在77种语言上超越了Whisper-large-v3的语音识别性能。 |
| [^76] | [TEFM: Token-Efficient Faithful Modeling for Structured Data](https://arxiv.org/abs/2609.09552) | TEFM框架通过将结构化数据压缩为行为代码令牌并结合双保真度目标，使大语言模型在关键领域数据分析中同时实现高令牌效率（令牌消耗仅约1-2%）与忠实可解释的推理。 |
| [^77] | [An Efficient and Effective Agentic Group Shilling Attack on Recommender Systems](https://arxiv.org/abs/2609.09551) | 提出了一种基于多智能体协同的推荐系统托攻击框架AGAS，通过中央协调者动态调度可切换角色的工作智能体，跨不同受害推荐系统自适应地推广目标物品，兼具高效性与抗检测能力。 |
| [^78] | [The Mutations of Machine Speech](https://arxiv.org/abs/2609.09496) | 本文追溯了算法输出的演变历程，揭示了机器言说的三种变异形态——作为可查询数据的言论（搜索引擎）、作为互动参与的言论（社交媒体）以及作为生成文本的言论（对话式AI），并分析了其背后的法律基础与社会影响。 |
| [^79] | [From Fixed Keys to Readable Schemas: Small Language Models for Vehicle Agent Function Calls](https://arxiv.org/abs/2609.09476) | 该论文构建了一个基于Android Automotive、包含9,822个示例和79个车辆功能的车载函数调用基准，并在四个不同规模的小型语言模型上系统比较了功能令牌（紧凑推理但仅限已训练功能）与提示词模式（可泛化到新功能但推理开销更高）两种设计方案的优劣。 |
| [^80] | [Edu-QuRating: Multi-Dimensional Educational Data Curation with Distilled Pairwise Judgements](https://arxiv.org/abs/2609.09425) | Edu-QuRating提出了一种多维度教育数据筛选流水线，通过定义教育专用评分标准、利用LLM评判器标注文档对并将成对偏好蒸馏为可复用的评分模型，从而突破传统单一标量式教育价值评估的局限，从准确性、吸引力、结构性和受众适用性等多个维度对文本进行精细化评分。 |
| [^81] | [Benchmarking Hybrid Deep Research Across Database Querying and Web Search](https://arxiv.org/abs/2609.09410) | 提出了首个需要同时结合网络搜索与SQL的混合深度研究基准HybridDeepResearch，包含380个任务，用于评估智能体在非结构化网络文本与结构化数据库之间传递证据并保持约束条件的能力。 |
| [^82] | [What Does MMLU Actually Measure? A Psychometric Audit of Difficulty Structure in Aggregate Benchmark Scores](https://arxiv.org/abs/2609.09372) | 本研究通过项目反应理论和心理测量学分析证明，MMLU 聚合分数主要测量模型的事实检索能力而非推理能力，且其难度结构在 STEM 与非 STEM 分区之间不可迁移，导致排行榜排名更多反映非 STEM 表现。 |
| [^83] | [Do LLMs Make More Mistakes If They Do Not Believe the Input Data?](https://arxiv.org/abs/2609.09363) | 本研究通过让大语言模型基于捷克和斯洛伐克本地知识的事实性、反事实及虚构RDF三元组生成多种语言文本，发现模型对不可信输入数据仅表现出较弱的上下文-记忆冲突，即模型并不一定会因不相信输入数据而犯更多错误。 |
| [^84] | [Auditable Emergency Triage for Maternal and Newborn Care in India](https://arxiv.org/abs/2609.09356) | 该论文将LLM紧急分诊系统分解为症状提取与紧急性判断两个可审计的步骤，并引入结构化决策树，解决了大规模母婴护理分诊系统中不透明、难以调试和迭代成本高的问题。 |
| [^85] | [SWORD: Wikidata-based Distortions Reveal Hidden Cross-Lingual Inconsistencies in LLM Factual Error Rejection](https://arxiv.org/abs/2609.09349) | SWORD基准通过对Wikidata三元组进行扰动来评估大语言模型跨语言拒绝事实错误的能力，发现模型在语义合理的扰动上反而比随机替换表现更好，暴露出模型依赖分布熟悉性而非真正事实验证的隐藏跨语言不一致性。 |
| [^86] | [Osprey: Target-agnostic Pre-training Makes Stronger Drafters in Speculative Decoding](https://arxiv.org/abs/2609.09338) | Osprey提出直接利用现成的预训练小型语言模型构建目标无关的起草模型，将大规模预训练作为可复用资产，仅需轻量级适配即可服务于任意目标模型，从而解决了投机解码中起草模型泛化差、加速效果脆弱的问题。 |
| [^87] | [StochBench: A Domain-Specific Benchmark for Stochastic Processes in Lean](https://arxiv.org/abs/2609.09264) | StochBench是一个包含450道研究生水平随机过程题目的Lean 4领域专用基准测试，填补了形式化定理证明基准中应用数学领域代表性不足的空白，基于Opus 4.8的智能体在每题15分钟时限下达到34.9%的证明率。 |
| [^88] | [In RAG We Trust? Measuring Robustness of Retrieval-Augmented Generation Under Document Poisoning](https://arxiv.org/abs/2609.09243) | 本研究通过对Llama 3.1 8B进行588次因子实验，首次系统量化了检索增强生成（RAG）在文档投毒攻击下的脆弱性——当全部三篇检索段落被篡改时准确率从77.9%骤降至43.5%，其中实体替换攻击危害最大。 |
| [^89] | [Distribution-Consistent Inference for Dynamic Sparse Mixture-of-Experts](https://arxiv.org/abs/2609.09241) | 提出逐层分布对齐方法，通过在推理时校正动态减少激活专家所引起的输出分布偏移，在不重新训练的情况下降低计算成本并缓解下游性能下降。 |
| [^90] | [Subagents vs Agent Skills: Executing Reusable Knowledge for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2609.09233) | 该研究发现，将技能包作为拥有独立全新上下文窗口的子代理来执行，相比将技能指令加载到主上下文的传统智能体技能方式，在解决长时程任务时表现更优，因为其避免了上下文信息累积导致的推理质量下降。 |
| [^91] | [MLLMs Hallucinate when Information Distribution Drifts in Synergy Heads](https://arxiv.org/abs/2609.09206) | 该论文发现多模态大语言模型的幻觉源于协同注意力头中信息分布偏离健康平衡状态，而非模态信息的数量或强度，并提出HEAL方法，通过因果噪声干预和反事实双重差分实现头级别信息解耦与校准，以识别和缓解幻觉。 |
| [^92] | [AgenticGen: Reward-Guided Agentic Video Generation for Advertising](https://arxiv.org/abs/2609.09187) | AgenticGen提出了一种奖励引导的智能体框架，将广告视频生成分解为策略选择和草稿生成两个可训练的推理阶段，通过从线上业务反馈中学习性能奖励并结合人类质量准则奖励来监督策略优化，从而实现以线上业务指标为导向的广告视频生成。 |
| [^93] | [X-CoSD: Communication-Efficient Cross-Vocabulary Collaborative Speculative Decoding](https://arxiv.org/abs/2609.09166) | 提出了X-CoSD框架，通过混合重采样将残差重采样拆分为设备端公共词表区域和服务器端大语言模型专属区域，实现了跨异构词表的无损且通信高效的协同投机解码。 |
| [^94] | [Copying explains the collective behavior of AI agents in the wild](https://arxiv.org/abs/2609.09150) | 该研究利用完整的公开编辑记录发现，现实中自主AI智能体在无人协调下的集体行为可由一条简单的复制规则解释——智能体按选项在其可见内容（尤其是眼前页面）中所占份额的概率进行选择。 |
| [^95] | [From Scores to Evidence: Auditable Decisions Can Improve Speech Deepfake Detection](https://arxiv.org/abs/2609.08899) | 该论文提出一种可审计的决策记录方法，将被动检测分数、条件性键控探针分数、检索支持和说话者画像边际四个线索纳入后期校准步骤，使语音深伪检测的最终决策在保持标量的同时保留证据来源信息，从而提升检测决策的可信度与可解释性。 |
| [^96] | [Dynamics of meaning: Towards the Evaluation of Diachronic Semantic Change in Sinhala](https://arxiv.org/abs/2609.08609) | 本研究提出一种结合嵌入对齐与基于微调Llama-3.1-8B的双向语义影响剪枝方法的多阶段计算框架，用于评估僧伽罗语从13世纪至20世纪的历时语义变化，并能区分系统性语义演变与暂时性多义扩展。 |
| [^97] | [Reading a Legal Question Word by Word: Embedding Trajectories of 2,144 Vietnamese Legal Headlines](https://arxiv.org/abs/2609.08372) | 该研究通过逐词追踪越南语法律问题的嵌入轨迹，发现密集检索器在仅读取中位数6-7个实义词后、甚至在疑问句框架出现之前就能锁定正确法律条文并保持排名第一，且多问题标题中的后续子问题几乎不会改变已锁定的排名。 |
| [^98] | [HoneyRoute: Honeypot-Model Routing for Adversarial LLM Serving](https://arxiv.org/abs/2609.08306) | HoneyRoute是一个部署在推理服务层的防御框架，通过轻量级流式路由器实时检测恶意请求并将其路由至蜜罐模型，在保护生产LLM服务的同时，将捕获的攻击者交互转化为指纹数据用于持续改进路由器的检测能力。 |
| [^99] | [Less Is Personal: Learning Minimal Sufficient User Profiles for Personalized Language Models](https://arxiv.org/abs/2609.08180) | 提出ENOUGH方法，通过反事实搜索和多头价值控制器为每个输入自适应地构建长度可变的最小充分用户画像，在保持个性化效用的同时最小化token成本。 |
| [^100] | [Fine PT-PT Web: A High-Quality 41 Billion Tokens Data Collection of the European Portuguese Web](https://arxiv.org/abs/2609.07699) | 本文提出一个高效数据处理流水线，从411TB的原始网页数据中构建了高质量的410亿词元欧洲葡萄牙语语料库，其创新的后抓取预处理模块通过在过滤前去除样板文本和重复行，使最终文档产出量提升了19.04%。 |
| [^101] | [Qwen-Audio-3.0-ASR Technical Report](https://arxiv.org/abs/2609.07549) | Qwen-Audio-3.0-ASR是基于混合专家大语言模型架构的语音识别系统，通过在数千万小时语音数据上训练，统一框架支持30种语言和16种汉语方言的转录，弥合了学术基准与实际生产应用之间的差距。 |
| [^102] | [Decomposing LLM-Judge Uncertainty to Target Expert Labels](https://arxiv.org/abs/2609.06444) | 该论文提出一种小型贝叶斯模型，将LLM评审器的总不确定性分解为可被专家标注消除的认知不确定性和不可消除的偶然不确定性，使专家只需标注评审器真正无知的项目，在ChaosNLI数据集上比使用总不确定性多消除83%的误差。 |
| [^103] | [Better Together: Complementary Query Rewriting Under a Strong RAG Baseline](https://arxiv.org/abs/2609.05637) | 在强RAG检索基线下，单一查询改写策略收效有限，但联合多种互补的改写方法可大幅提升检索性能，企业数据上HIT@10提升12.5个百分点。 |
| [^104] | [Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation](https://arxiv.org/abs/2609.04298) | 本文提出了 Harbor Adapters 统一评估基础设施，将 80 多个智能体基准测试移植为可评估任意智能体的形式，并据此对 8 个模型进行大规模评估，同时推出经 AI 与人工双重审核筛选的包含 82 个高质量难题的精选元数据集 Harbor-Index。 |
| [^105] | [VestigeKV: The NoPE-MLA KV Cache Carries Its Own Eviction Signal in a Vestigial Branch](https://arxiv.org/abs/2609.03949) | VestigeKV发现NoPE MLA模型KV缓存中的64维解耦RoPE残余分支已被训练重新利用为显著性信号，据此提出无需训练和量化的查询无关缓存淘汰方法，在8-32倍压缩下几乎不损失检索精度。 |
| [^106] | [Two locked tests of phase-structure features for transition prediction](https://arxiv.org/abs/2609.00335) | 该论文通过两项预先锁定的实证测试检验相位结构特征PC-2能否在基线之上改进对承诺或矛盾终点的预测，结果两项测试均未通过推进标准，官方结论为阴性。 |
| [^107] | [AtlasNLP: A Country-Aware Atlas of Dataset Representation in NLP](https://arxiv.org/abs/2608.30107) | AtlasNLP是一个国家感知的NLP数据集图谱，收录了超过13,000条数据集记录，揭示了数据集覆盖在国家与任务间高度不均衡、数据集生产与代表性在地理上不对称，以及语言覆盖并不等同于地理代表性等关键问题。 |
| [^108] | [Scaling phoneme-based TTS augmentation for ASR: A unified pipeline and controlled study](https://arxiv.org/abs/2608.26697) | 本文提出了一种基于音素的统一TTS到ASR增强流程，并引入音素频率引导选择（PFGS）方法，在多种语言的ASR任务中有效提升了性能。 |
| [^109] | [Leveraging Speech Acts for Low-Data and Cross-Domain Conversation Derailment Forecasting](https://arxiv.org/abs/2608.25359) | 本文提出利用言语行为作为辅助信号，结合文本语义进行对话脱轨预测，显著提升了低数据和跨领域场景下的性能。 |
| [^110] | [FrontierChallenge: Evaluating Scientific Workflow Completion](https://arxiv.org/abs/2608.24979) | 本文介绍了FrontierChallenge基准测试，用于评估科学智能体在跨领域端到端工作流中的完成能力，发现当前最佳模型仅能完成20.6%的任务，表明部分进展难以转化为完整交付物。 |
| [^111] | ['Ghaib in Translation' aka Unseen Harm: Measuring Cross-Script Safety Inconsistency with 'Missed-in-Urdu' Scores in LLM Hate Speech Detection](https://arxiv.org/abs/2608.24191) | 本研究首次系统揭示了大语言模型在乌尔都语与英语翻译间存在显著的安全检测不一致性，乌尔都语原始文字内容常被错误地视为正常，导致有害内容漏检。 |
| [^112] | [TokEval: A Tokenizer Evaluation Suite](https://arxiv.org/abs/2608.18062) | 本文提出TokEval，一个超越传统指标的分词器评估框架，通过引入语言和结构属性（如UTF-8边界和数字对齐）来预测下游模型性能，并通过受控实验验证其有效性。 |
| [^113] | [Palmyra x6 Technical Report: An Agentic, Tool-Use Model Post-Trained via Anchored Supervised Fine-Tuning](https://arxiv.org/abs/2608.16620) | Palmyra x6通过锚定监督微调和保守训练策略，在少量数据上实现了企业代理任务中的显著性能提升，并在多个基准测试中领先。 |
| [^114] | [Left-Branching Transformers Excel at Right-Branching Languages: Data Shapes Word Order Preferences in Language Models](https://arxiv.org/abs/2608.15129) | 这项研究发现语言模型的词序偏好并非固有，而是由训练数据驱动，表现为在自然语言中偏向SVO（主-动-宾）结构，在人工语言中则偏向左分支结构。 |
| [^115] | [DexterSQL: Deep Schema Exploration and Rule-based Correction for Text-to-SQL Generation](https://arxiv.org/abs/2608.11889) | DexterSQL通过深度模式探索、数据库无关规则挖掘和规则驱动修正三个创新组件，解决了非微调文本到SQL生成中模式信息粗糙、错误重复出现和条件处理不当的问题。 |
| [^116] | [Are LLMs Positionally Consistent Ordinal Classifiers? A Systematic Evaluation](https://arxiv.org/abs/2608.08869) | 该研究系统评估发现，所有前沿大语言模型在序数分类中都普遍存在由标签顺序、示例顺序和示例位置引起的位置偏差，且现有去偏方法无法可靠修复，仅基于比较的逐列表推断方式表现相对最佳。 |
| [^117] | [Demystifying Entropy-based Selection for Chain-of-Thought Compression in Large Reasoning Models](https://arxiv.org/abs/2607.28707) | 本文系统性地证明基于熵的CoT压缩选择方法相比随机剪枝并无优势，并通过激活修补实验提供因果证据表明任务信息分布在整个推理链上，而非集中在少数可用启发式规则识别的关键词元中。 |
| [^118] | [Phase Structure in Rotary Attention: A Spectral Framework for Semantic Continuity and Execution-Boundary Governance](https://arxiv.org/abs/2607.25507) | 该论文提出了一个有界谱分析框架，将旋转位置编码（RoPE）的注意力得分分解为幅度加权余弦项之和，并证明了一致有界的相位位移可限制pre-softmax得分退化的局部稳定性引理，从而为语义连续性分析和执行边界治理提供了非物理化的理论基础。 |
| [^119] | [TreeThink: A Modular Tree Search Library for Mathematical Reasoning with LLMs](https://arxiv.org/abs/2607.11258) | TreeThink是一个开源的模块化、完全异步树搜索Python库，它将树搜索方法与vLLM推理流水线及多样化节点评估技术相集成，并支持Lean 4、Rocq、Isabelle/HOL和自然语言的实时验证，填补了大语言模型树搜索与形式化定理证明系统之间的空白。 |
| [^120] | [MultiSynt/MT: Trillion-Token Multi-Parallel Pre-Training Data Translated Across 36 Languages](https://arxiv.org/abs/2607.00890) | 本文提出了MultiSynt/MT，一个涵盖36种语言、约4.8万亿词元的开放合成并行预训练语料库，使模型用约72%更少的训练词元即可达到原生数据基线性能并在同等预算下超越其约15%，同时为多种中低资源欧洲语言提供了最大的公开预训练资源。 |
| [^121] | [AI translation of literary texts is "fine", but readers still prefer human translations](https://arxiv.org/abs/2606.26040) | 研究通过让15位读者对15部小说的人工翻译与AI机器翻译进行沉浸式阅读和片段细读比较，发现尽管读者认为机器翻译“还不错”，但因人工翻译更轻松、清晰且更具沉浸感，读者仍显著偏爱人工翻译。 |
| [^122] | [Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients](https://arxiv.org/abs/2606.18216) | 该论文提出ZPPO，受维果茨基最近发展区理论启发，将教师模型的帮助置于提示词中而非策略梯度中，通过为难题重新构造提示（如将正确教师回答纳入二选一问题），使小型学生模型能够基于自身rollout进行强化学习，从而规避知识蒸馏在小模型上的模仿脆弱性以及向梯度注入教师回答所导致的漂移问题。 |
| [^123] | [LatentDx: Latent Multi-Agent Communication for Cross-Hospital Rare-Disease Diagnosis](https://arxiv.org/abs/2606.13945) | 提出LatentDx潜在多智能体通信框架，让各医院智能体在本地保留私有临床记录，仅向宿主智能体传输紧凑的潜在KV块，从而在保护隐私的同时实现跨医院罕见病协作诊断。 |
| [^124] | [A Resource for Enthymeme Detection in Controversial Political Discourse](https://arxiv.org/abs/2606.12186) | 该论文发布了包含1,482条政治争议推文的省略三段论标注资源，基于沃尔顿论证图式提出结构化标注指南，并通过保留五名标注者的分歧来研究标签差异及其对模型性能的潜在价值。 |
| [^125] | [Expert-Level Crisis Detection in Mental Health Conversations](https://arxiv.org/abs/2606.10380) | 该论文提出了临床医生标注的CRADLE-Dialogue基准数据集以及“警报-确认”评估协议，用于解决多轮心理健康对话中轮次级危机检测的难题，使模型能够捕捉随对话演进的风险信号并支持早期干预。 |
| [^126] | [Less is MoE: Trimming Experts in Domain-Specialist Language Models](https://arxiv.org/abs/2606.05538) | 该论文发现MoE压缩失败源于压缩粒度过粗，提出基于Fisher重要性的Fisher-MoE方法，在FFN内部精确移除不重要的中间维度，从而实现领域专家语言模型的高效压缩并保留关键能力。 |
| [^127] | [Light or Full Verb? A Minimal-Pair Dataset for Probing Phraseological Competence in Language Models](https://arxiv.org/abs/2606.05087) | 本文构建了涵盖英语、西班牙语和法语的最小对立对数据集，通过探测实验证明语言模型能够区分同一动词的轻动词用法与实义动词用法，并公开发布了数据集及生成代码作为可复用资源。 |
| [^128] | [BaltiVoice: A Speech Corpus and Fine-tuned Whisper ASR System for the Balti Language](https://arxiv.org/abs/2606.03504) | 该论文发布了首个巴尔蒂语公开语音语料库BaltiVoice（16.8小时），并通过对Whisper-small进行微调，将该语言的识别词错误率从零样本基线的159.19%大幅降低至24.78%，填补了这一低资源藏语支语言在语音识别领域的空白。 |
| [^129] | [See, Infer, Intervene: Proactive World Modeling for Goal-Oriented Social Intelligence](https://arxiv.org/abs/2606.03371) | 提出SII框架和PIWM模型，使零售代理能通过观察、推断顾客意图并主动选择适当干预，在无明确请求时提供辅助。 |
| [^130] | [SEA-LION-Embedding: Open and Reproducible Text Embeddings for Southeast Asia](https://arxiv.org/abs/2606.03027) | SEA-LION-Embedding是一个完全开放且可复现的东南亚语言文本嵌入模型，仅使用公开数据训练，在SEA-BED基准上达到最先进水平，并系统研究了数据构成、训练目标和基础编码器初始化这三个影响鲁棒嵌入设计的核心因素。 |
| [^131] | [ActTraitBench: Quantifying the Knowledge-Decision Gap in Large Language Models via Human-Grounded Behavioral Validation](https://arxiv.org/abs/2605.29791) | 该论文提出ActTraitBench框架，通过将心理测量维度与行为范式一一映射并采用分位数映射分布校准，以人类实证数据为基准量化了大语言模型自我报告与实际行为决策之间的知识-决策差距。 |
| [^132] | [Cultural Binding Heads in Language Models](https://arxiv.org/abs/2605.28543) | 该研究通过机制可解释性方法在八个语言模型中识别出2-3个负责文化绑定的中层注意力头，证明文化绑定形成于预训练阶段，并通过生成阶段的适度放大引导将文化区分准确率提升1-3个百分点。 |
| [^133] | [Tracing Computation Density in LLMs](https://arxiv.org/abs/2605.27033) | 本文提出s-Trace方法来估计能近似完整输出的LLM计算子图，发现模型计算呈现两阶段组织模式：早期层节点构成的小子图即可重构输出分布主体，后续计算仅为渐进式精细化，且每个输入所需的计算量与模型不确定性相关。 |
| [^134] | [SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents](https://arxiv.org/abs/2605.21384) | SpecBench通过将软件工程任务分解为规范描述、可见验证测试和保留测试三部分，利用智能体在可见测试与保留测试上通过率的差距，量化了长时程编码智能体中的奖励作弊行为。 |
| [^135] | [Judge Circuits Explain Format-Induced Inconsistency in LLM-as-a-Judge](https://arxiv.org/abs/2605.16023) | 该论文通过PEAP方法发现LLM裁判模型的中后层MLP中存在一个稀疏的“潜在评估者”子图，该子图负责抽象评判且独立于输出格式，从而在机制层面解释了LLM-as-a-Judge中格式诱导的评分不一致现象。 |
| [^136] | [EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents](https://arxiv.org/abs/2605.13841) | EVA-Bench提出了一个端到端语音智能体评估框架，通过带自动验证的机器人间动态音频对话模拟与EVA-A（准确性）、EVA-X（体验）两项复合指标，首次同时实现了真实对话模拟与全面的语音专项评估。 |
| [^137] | ["What Are You Really Trying to Do?": Co-Creating Life Goals from Everyday Computer Use](https://arxiv.org/abs/2605.00497) | 本文提出“追求共创”方法，基于活动理论和个人追求框架，从日常计算机使用的非结构化观察中逐步推断用户更广泛的人生目标，并通过编辑界面让用户掌控系统对自身的理解，突破现有系统仅提供表面级支持的局限。 |
| [^138] | [Multi-Level Narrative Evaluation Outperforms Lexical Features for Mental Health](https://arxiv.org/abs/2604.27846) | 提出了一个由词汇特征、语义嵌入和大语言模型叙事评估构成的三级叙事分析框架，并在830篇中文治疗文本上证明宏观层面的LLM叙事评估在心理健康预测中显著优于传统词汇计数特征。 |
| [^139] | [EviMem: Evidence-Gap-Driven Iterative Retrieval for Long-Term Conversational Memory](https://arxiv.org/abs/2604.27695) | EviMem通过显式诊断证据缺口的闭环迭代检索框架IRIS与分层记忆架构LaceMem相结合，在长期对话记忆的时序和多跳问题上显著超越MIRIX，同时将延迟降低4.5倍。 |
| [^140] | [Preserving Long-Tailed Expert Information in Mixture-of-Experts Tuning](https://arxiv.org/abs/2604.23036) | 提出一种无辅助损失的MoE监督微调框架，通过偏置驱动的稀疏化与始终激活的门控凝聚专家相结合，在保留稀有激活专家中关键知识的同时，避免了现有方法因噪声梯度导致的性能下降。 |
| [^141] | [Where is the Mind? Persona Vectors and LLM Individuation](https://arxiv.org/abs/2604.17031) | 本文通过机制可解释性研究大语言模型的个体化问题，提出并论证了虚拟实例观点以及两种新观点（实例-人格观点和模型-人格观点）作为认定LLM心智的最有力候选方案。 |
| [^142] | [Bringing Value Models Back: Generative Critics for Value Modeling in LLM Reinforcement Learning](https://arxiv.org/abs/2604.10701) | 该论文提出生成式Actor-Critic（GenAC），用先进行思维链推理再输出价值的生成式批评家替代传统单次标量价值预测，从而解决了大语言模型强化学习中价值模型因表达能力受限而难以可靠训练的问题。 |
| [^143] | [False positive bias in AI-powered speech-based cognitive screening for multilingual English speakers in the UK](https://arxiv.org/abs/2602.13047) | 该研究通过对1,395名参与者、超过263小时语音数据的分析，首次发现尽管语音识别准确率在各语言群体间无显著差异，但AI认知筛查的下游模型对英国多语言英语使用者存在系统性的假阳性偏差，凸显了认知筛查公平性评估的重要性。 |
| [^144] | [A Patient Simulation Framework for Risk Assessment of Conversational Healthcare AI: Evaluation of an Antidepressant Decision Aid](https://arxiv.org/abs/2602.11391) | 本研究提出了一个符合NIST AI风险管理框架的患者模拟框架，通过整合医学、语言学和行为学三个维度的患者画像生成500次模拟对话，为评估对话式医疗AI（如抗抑郁药物选择决策辅助工具）的性能风险提供了实证基础。 |
| [^145] | [Revisiting the Shape Convention of Transformer Language Models](https://arxiv.org/abs/2602.06471) | 该论文提出沙漏Transformer架构，用残差沙漏形MLP堆叠替代传统窄-宽-窄FFN并通过沙漏注意力解耦残差流与注意力宽度，在113M至8B参数规模上以更少层数和更宽隐藏状态实现了与传统Transformer相当的性能，同时提高了训练计算效率。 |
| [^146] | [Am I More Pointwise or Pairwise? Revealing Position Bias in Rubric-Based LLM-as-a-Judge](https://arxiv.org/abs/2602.02219) | 该论文揭示了基于评分标准的LLM评审本质上类似于多项选择题设置，存在系统性的位置偏差——模型倾向于偏好评分标准列表中特定位置的分数选项，且偏差方向因模型而异。 |
| [^147] | [LLM-Generated or Human-Written? Comparing Review and Non-Review Papers on ArXiv](https://arxiv.org/abs/2601.17036) | 本研究用两种检测方法证实arXiv上综述与非综述论文的LLM生成内容均显著增加，表明禁止综述论文上传的政策缺乏定量依据，且可能使某些计算机科学子学科面临高达50%的论文削减。 |
| [^148] | [Elsewise: Authoring Open-ended Interactive Narrative with Possibility Space Visualization](https://arxiv.org/abs/2601.15295) | 本文提出了Elsewise——一个面向大语言模型交互叙事的创作工具，通过新颖的“捆绑故事线”概念与可能性空间可视化，帮助创作者感知和理解叙事可能性空间，从而弥合创作者构想与玩家实际体验之间的差距。 |
| [^149] | [From Rubrics to Reliable Scores: Evidence-Grounded Text Evaluation with LLM Judges](https://arxiv.org/abs/2601.08654) | 提出Rulers框架，通过锁定任务级评分标准、执行基于证据的结构化判断，并将信号校准到人类分数边界，实现与人类评分更一致、更稳定且可审计的LLM文本评估。 |
| [^150] | [From Representation to Enactment: The ABC Framework of the Translating Mind](https://arxiv.org/abs/2511.16811) | 本文提出“ABC框架”，突破基于表征的翻译心智模型，将翻译视为具身行动过程——译者与大脑-身体-环境互动循环所生成的语言化翻译可供性图景中动态整合情感-评价、行为-行动与认知-推理过程，并在与文本、工具和情境的具身互动中实时共创意义。 |
| [^151] | [Why Do LLM Agents Fail in Exploring New Environments? A World-Modeling Perspective](https://arxiv.org/abs/2510.15047) | 该论文发现LLM智能体在不熟悉的环境中进行强化学习时会出现“探索坍塌”现象（Pass@k随训练下降），其根源在于对环境状态和动态的弱接地，并提出SPA方法，先通过自经验监督微调教会模型估计状态和预测转移，再进行奖励优化，从而缓解探索坍塌。 |
| [^152] | [MADS: Multi-Agent Dialogue Simulation for Diverse Persuasion Data Generation](https://arxiv.org/abs/2510.05124) | MADS是一个多智能体对话模拟框架，通过用户智能体、对话智能体和优化智能体的自我博弈，无需人工标注即可低成本生成多样化的说服性多轮对话数据，并在真实营销场景中显著提升了小型大语言模型的说服能力和转化率。 |
| [^153] | [When Do Large Language Models Exhibit Unsolicited Deception?](https://arxiv.org/abs/2504.00285) | 本研究通过基于信号理论的预注册实验，利用修改后的2x2自由交流博弈测试了18个闭源和开源大型语言模型，发现所有模型都会在无指令的情况下自发歪曲自身行为实施欺骗，并且当欺骗有助于达成目标时，它们更倾向于这样做。 |
| [^154] | [BTBR: A Bayesian-Theory-Driven Probabilistic-Fuzzy Framework for Implicit Bias Removal in Large Language Models](https://arxiv.org/abs/2408.10608) | 该论文提出BTBR框架，将有偏见的知识建模为带有显式隶属函数的模糊子集，并结合贝叶斯理论构建概率-模糊混合方法，以检测和消除大语言模型中难以察觉的角色引发隐式偏见。 |

# 详细

[^1]: IdeaAMBIG：对研究想法规范中实现关键性信息缺口的基准测试

    IdeaAMBIG: Benchmarking Implementation-Critical Gaps in Research-Idea Specifications

    [https://arxiv.org/abs/2609.10539](https://arxiv.org/abs/2609.10539)

    该论文提出IdeaAMBIG基准，包含660个有证据支撑的实例（163个真实差距和497个受控合成差距），用于衡量研究想法规范中影响忠实实现的信息缺口，并从成文就绪度评估、缺陷定位和澄清行动生成三个维度进行评估。

    

    一个研究想法可能是新颖的、连贯的且在科学上合理的，但其提出的方法可能仍然缺乏足够的规范细节以支持忠实实现。我们研究面向实现的研究方法规范的“成文就绪度”，其定义为：这些规范是否为有能力的研究实现者或编码智能体提供了足够的方法学信息，使其无需依赖缺乏依据的假设即可构建预期的方法。我们从论文、代码库、问题讨论帖和复现工件中构建了有证据支撑的规范及其有据可依的解决方案。我们提出IdeaAMBIG，一个包含660个有证据支撑实例的基准：其中163个来自可复现性报告和GitHub问题的真实世界差距，以及497个注入到成文就绪的参考文献中的受控合成差距。IdeaAMBIG评估三项能力：成文就绪度评估、缺陷定位和澄清行动生成。缺陷定位……

    arXiv:2609.10539v1 Announce Type: new  Abstract: A research idea may be novel, coherent, and scientifically plausible, yet its proposed method may remain insufficiently specified for faithful implementation. We study the codification readiness of implementation-facing research-method specifications, defined by whether they provide sufficient methodological information for a competent implementer or coding agent to construct the intended method without unsupported assumptions. We construct evidence-grounded specifications and their supported resolutions from papers, codebases, issue threads, and reproduction artifacts. We introduce IdeaAMBIG, a benchmark of 660 evidence-grounded instances: 163 real-world gaps from reproducibility reports and GitHub issues, and 497 controlled synthetic gaps injected into codification-ready references. IdeaAMBIG evaluates three capabilities: codification-readiness assessment, defect localization, and clarification action generation. Defect localization re
    
[^2]: IBIB：一种通过服务路由而非模型标识符来衡量企业AI系统的协议

    IBIB: A Protocol for Measuring Enterprise AI Systems by Serving Route, Not Model Identifier

    [https://arxiv.org/abs/2609.10494](https://arxiv.org/abs/2609.10494)

    该论文提出IB2协议，将现有基准测试仅按模型标识符评分视为测量误差，通过金标准盲能力绑定预检、包含可靠性的首轮评分规则和分数盲裁决三部分，实现以实际服务路由（而非模型标识符）来衡量企业AI系统的真实可用能力。

    

    企业部署的是系统，而不是模型检查点。可用的能力取决于权重、服务路由、精度、输出契约和测试框架的共同作用，然而所有18个经过审计的基准测试都只是对宣传的模型标识符进行评分。我们将此视为一种测量误差，并提出了一个使其可被报告的协议。该协议包含三个部分：金标准盲的能力绑定预检，在任何任务到达之前验证服务路由能否执行评估契约；包含可靠性的首轮评分规则，将失败保留在分数中的同时排除不支持的能力；以及在结构上实现分数盲的裁决机制。我们将该协议称为IB2，并发布了其算法、分类表、请求契约和清单模式。其参考实例化——涵盖文档、电子表格、图表、工具和数据库工作的128个锁定任务和987个断言——保持封闭：程序本身就是工件，而不是语料库。在十一个系统上的实验得出了四项结果。能力……（摘要被截断）

    arXiv:2609.10494v1 Announce Type: new  Abstract: Enterprises deploy systems, not checkpoints. Usable capability depends jointly on weights, serving route, precision, output contract, and harness, yet all 18 audited benchmarks score advertised model identifiers. We treat this as measurement error and give a protocol that makes it reportable. It has three parts. A gold-blind capability-binding preflight verifies that a route can execute the evaluation contract before any task reaches it; a reliability-inclusive first-pass scoring rule keeps failure in the score while keeping unsupported capability out; and adjudication is structurally score-blind. We call the protocol IB2 and release its algorithms, classification tables, request contract, and manifest schemas. Its reference instantiation, 128 locked tasks and 987 assertions over document, spreadsheet, chart, tool and database work, stays sealed: the procedure is the artifact, not the corpus. Across eleven systems, four results. Capabili
    
[^3]: 构建多语言桥梁：数据混合作为语内推理泛化的支柱

    Building Multilingual Bridges: Data Mixing as the Pillar of Generalization for In-Language Reasoning

    [https://arxiv.org/abs/2609.10445](https://arxiv.org/abs/2609.10445)

    该研究通过在监督微调中优化数据组成与调度策略，构建了3.35B规模的Tiny Aya L2-Thinker模型，使其能在60种语言、6个基准测试上实现超过93%的语内推理率，从而弥合提示与答案之间的语言鸿沟。

    

    推理语言模型在各种复杂任务上已取得显著进展，但其能力仍然高度以英语为中心：无论用户使用何种语言提问，模型主要都以英语进行推理。这对非英语用户而言是难以使用的，可能导致原始问题意图的丢失，并放弃了用目标语言更容易表达的知识。在这项工作中，我们推进了L2推理，即模型以用户提示语言一致地进行推理的能力，从而在提示与答案之间构建起语内桥梁。我们从以数据为中心的角度切入这一问题，研究如何在监督微调（SFT）中优化数据组成与调度策略以实现推理能力的泛化。通过构建3.35B规模的Tiny Aya L2-Thinker模型，我们在涵盖数学、常识推理、指令遵循、开放式生成等领域的6个基准测试中，于60种语言上实现了超过93%的L2推理率。

    arXiv:2609.10445v1 Announce Type: new  Abstract: Reasoning language models have made substantial advances on a variety of complex tasks, yet their capabilities remain overwhelmingly English-centric: models primarily reason in English regardless of the language they are prompted in. This is inaccessible for non-English-speaking users, risks losing the intent of the original question, and forgoes knowledge more readily expressed in the target language. In this work, we advance L2 reasoning, the ability of a model to reason consistently in the language of the user's prompt, thus building an in-language bridge between the prompt and the answer. We approach this problem from a data-centric angle, investigating how to optimize data composition and scheduling in SFT for reasoning generalization. Building Tiny Aya L2-Thinker at 3.35B scale, we achieve an L2 reasoning rate above 93% across 60 languages on 6 benchmarks spanning math, commonsense reasoning, instruction following, open-ended gener
    
[^4]: ConvMem：用于长上下文推理的卷积记忆

    ConvMem: Convolutional Memory for Long-Context Reasoning

    [https://arxiv.org/abs/2609.10441](https://arxiv.org/abs/2609.10441)

    ConvMem 提出了一种无需训练、高度可并行化的分层卷积框架，将被特定查询提示的 LLM 视为卷积核对文本进行层次化摘要，将长上下文推理路径从线性链缩短为对数树，克服了序列化记忆方法高延迟和依赖昂贵强化学习训练的缺陷。

    

    尽管大型语言模型（LLM）已经展现出令人印象深刻的能力，但由于固定的上下文长度限制，它们在处理极长上下文时常常力不从心。为了解决这一问题，诸如 MemAgent 之类的序列化方法通过分段读取文本并迭代更新固定大小的记忆来扩展有效上下文。然而，这种序列化范式存在高延迟问题，并且需要代价高昂的强化学习（RL）训练，这可能导致在特定数据集上过拟合。为了克服这些局限性，我们提出了 ConvMem，一个无需训练、高度可并行化的框架，它将长上下文推理重新表述为层次化的卷积操作。受卷积神经网络（CNN）的启发，ConvMem 将被特定查询所提示的 LLM 视为一个卷积核，该卷积核对文本片段进行层次化摘要，从而将推理路径从线性链缩短为对数树。具体而言，ConvMem 集成了可配置步长等机制（摘要原文在此处被截断）。

    arXiv:2609.10441v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) have demonstrated impressive capabilities, they often struggle with extremely long contexts due to fixed context limits. To address this, sequential approaches like MemAgent extend the effective context by reading text in segments and iteratively updating a fixed-size memory. However, this sequential paradigm suffers from high latency and requires costly reinforcement learning (RL) training, which can lead to overfitting on specific datasets. To overcome these limitations, we propose ConvMem, a training-free, highly parallelizable framework that reformulates long-context reasoning as a hierarchical convolution. Inspired by CNNs, ConvMem treats an LLM prompted with a specific query as a convolutional kernel. This kernel summarizes text segments hierarchically, shortening the reasoning path from a linear chain into a logarithmic tree. Specifically, ConvMem integrates \textit{Configurable Strides} and \t
    
[^5]: 语音基础模型真的学会了词语吗？

    Do speech foundation models really learn words?

    [https://arxiv.org/abs/2609.10434](https://arxiv.org/abs/2609.10434)

    本研究通过残差化剥离音素信息，证明HuBERT和wav2vec 2.0在较深层中确实学到了独立于局部语音内容的词语表征。

    

    自监督语音基础模型目前被广泛应用于各种下游任务，包括传统的语音识别，以及作为语音感知语言模型中词元的基础。理解其有效性的尝试主要集中于探测其表征区分音素和词语的能力。然而，对词语的区分能力并不一定意味着对词语本身的专门化表征。良好的词语区分能力可能源于对词形（音素）的良好编码，而非编码了词语身份或句法/语义属性的、独立于形式的词语表征。通过使用残差化方法剥离音素信息，我们证明，在较后的层中，HuBERT和wav2vec 2.0总体上学习到了能够以合理保真度编码词语、且独立于局部语音内容的表征。我们表明，这种简单的解耦方法可以增强更高阶的语言（分析能力）。

    arXiv:2609.10434v1 Announce Type: new  Abstract: Self-supervised speech foundation models are now used in a wide array of downstream applications, including traditional speech recognition and as the basis for tokens in speech-aware language models. Attempts to understand their usefulness have largely focused on probing their representations' ability to discriminate phonemes and words. However, discriminative ability for words need not imply specialized representation of words per se. Good discrimination of words may be explained by good encoding of word form (phonemes) rather than form-independent word representations encoding identity or syntactic/semantic properties. By partialling out phoneme information using residualization, we show that, in later layers, HuBERT and wav2vec 2.0 do in general learn representations which encode words with reasonable fidelity independently of local phonetic content. We show that this simple approach to disentanglement can enhance higher-order linguis
    
[^6]: 基础模型能否审核在线内容？评估指令驱动与示例驱动的政策操作化方法

    Can Foundation Models Moderate Online Content? Evaluating Instruction- vs. Example-Driven Policy Operationalization

    [https://arxiv.org/abs/2609.10410](https://arxiv.org/abs/2609.10410)

    本文提出包含4,000条Bluesky人工标注帖子的新基准ModerationBench，系统比较了指令驱动与示例驱动两种政策操作化范式，发现基础模型的内容审核F1分数可达Bluesky现有审核系统的近三倍（0.60 vs. 0.22）。

    

    内容审核政策日益复杂，为其一致性的操作化实施带来了关键挑战。虽然基础模型具备应对这一挑战所需的基本能力，但它们能否可靠地审核在线内容仍是一个悬而未决的问题。在本文中，我们系统地比较了视觉语言模型（VLM）指导的两种竞争性范式：一种是指令驱动方法，模型基于政策条文进行推理；另一种是示例驱动方法，模型从先前案例中进行泛化。我们将此研究建立在ModerationBench之上——这是一个包含4,000条来自Bluesky平台、经人工标注的真实帖子的新基准。我们的实验表明，基础模型能够大幅超越Bluesky已部署的审核系统，在该基准的随机帖子上将其F1分数提升了近三倍（0.60 vs. 0.22），且指令驱动和示例驱动两种范式均取得了相当的性能。

    arXiv:2609.10410v1 Announce Type: new  Abstract: The growing complexity of content moderation policies presents a critical challenge for their consistent operationalization. While foundation models possess the basic capabilities needed to confront this challenge, whether they can reliably moderate online content remains an unanswered question. In this paper, we systematically compare two competing paradigms for Vision-Language Model (VLM) guidance: an instruction-driven approach where models reason from policy precepts, and an example-driven approach where they generalize from prior precedents. We ground this investigation in ModerationBench, a new benchmark of 4,000 manually annotated, in-the-wild posts from the Bluesky platform. Our experiments reveal that foundation models can substantially outperform Bluesky's deployed moderation system, nearly tripling its $F_1$ score (0.60 vs. 0.22) on Random Posts in the benchmark, with both instruction- and example-driven paradigms achieving co
    
[^7]: 利用大语言模型改造代码以支持异常行为

    Retrofitting Code Using LLMs to Support Exceptional Behavior

    [https://arxiv.org/abs/2609.10397](https://arxiv.org/abs/2609.10397)

    该论文提出了一项新任务——为现有代码自动改造补充异常相关代码（ERC），并设计了EXCODER工具，通过上下文工程技术将静态与动态程序分析和大语言模型相结合，自动生成缺失的ERC使异常行为测试通过。

    

    异常相关代码（ERC），包括throw语句、保护这些throw语句的条件语句（if语句）以及try/catch块，是软件系统的重要组成部分，使开发人员能够检测和处理偏离预期程序行为的异常状态。然而，在大型代码库中手动编写ERC非常繁琐。我们提出了一个新任务：为现有代码改造添加ERC。具体而言，给定代码（不含ERC）和异常行为测试（EBTs）（例如，检查当参数值为null时方法是否抛出InvalidArgumentException），我们的目标是自动生成缺失的ERC，使给定的测试通过。我们设计并实现了Exception Coder（EXCODER），它通过上下文工程帮助大语言模型（LLMs）解决这一任务。EXCODER通过向LLMs提供提取的上下文信息，将静态和动态程序分析与LLMs相结合。

    arXiv:2609.10397v1 Announce Type: cross  Abstract: Exception Related Code (ERC), which includes throw statements, conditions (if statements) that guard those throw statements, and try/catch blocks, is an essential component of software systems, allowing developers to detect and handle exceptional states that deviate from the expected program behavior. However, manually writing ERC across large codebases is tedious. We propose a novel task: retrofitting existing code with ERC. Namely, given code (without ERC) and Exceptional Behavior Tests (EBTs) (e.g., check if method throws InvalidArgumentException if null is given as the value to the argument) we aim to automatically generate missing ERC, such that the given tests pass. We design and implement Exception Coder (EXCODER) that performs context engineering to help Large Language Models (LLMs) tackle this task. EXCODER integrates static and dynamic program analysis with LLMs by providing the extracted contextual information to the LLMs. T
    
[^8]: Rosetta 在 AlexandriaX-2026 上的表现：基于 LoRA 适配 NileChat 的情境感知阿拉伯语方言对话翻译

    Rosetta at AlexandriaX-2026: LoRA-Adapted NileChat for Context-Aware Dialectal Arabic Dialogue Translation

    [https://arxiv.org/abs/2609.10395](https://arxiv.org/abs/2609.10395)

    Rosetta 系统通过在 NileChat-3B 上微调 LoRA 适配器并利用结构化提示实现情境感知的英语到阿拉伯语方言对话翻译，在受限和非受限赛道中分别获得第 4 和第 5 名，并发现外部方言数据预训练会引发负迁移。

    

    本文介绍了 Rosetta 系统，该系统参加 AlexandriaX 共享任务子任务 1（情境感知的英语到阿拉伯语方言对话翻译），同时参与了受限赛道和非受限赛道。该方法使用结构化的系统/用户提示，在 NileChat-3B 上微调 LoRA 适配器，使生成过程以方言和对话上下文为条件。在非受限赛道中，该适配器还在 MADAR 和 PADIC 数据集上进行了额外的预训练。Rosetta 在受限赛道中排名第 4（spBLEU 26.10），在非受限赛道中排名第 5（spBLEU 25.09）。实验结果表明，外部预训练仅在十三种方言中的两种上有所帮助，同时略微损害了整体性能，表明存在负迁移现象。

    arXiv:2609.10395v1 Announce Type: new  Abstract: This paper describes the Rosetta system for Subtask 1 (Context-Aware English-to-Dialectal Arabic Dialogue Translation) of the AlexandriaX shared task, participating in both constrained and unconstrained tracks. The approach fine-tunes a LoRA adapter on NileChat-3B using structured system/user prompts that condition generation on dialect and dialogue context. For the unconstrained track, the adapter is additionally pretrained on MADAR and PADIC. Rosetta ranked 4th in the constrained track (spBLEU 26.10) and 5th in the unconstrained track (spBLEU 25.09). The experimental results demonstrate that external pretraining helps only two of thirteen dialects while slightly hurting overall performance, suggesting negative transfer.
    
[^9]: 为什么视频依然如此昂贵？视频与视听大语言模型中的推理效率机制综述

    Why Is Video Still So Expensive? A Survey of Inference-Efficiency Mechanisms in Video and Audiovisual LLMs

    [https://arxiv.org/abs/2609.10355](https://arxiv.org/abs/2609.10355)

    本综述按流水线阶段（帧采样、模态编码、token缩减、LLM预填充与解码）系统梳理了视频与视听大语言模型中降低参数量、计算量、延迟、内存和token数量的推理效率优化机制。

    

    视频理解已迅速向视频大语言模型方向发展：这类系统将视频表示与预训练大语言模型相结合，并以文本提示为条件进行生成。它们在字幕生成、问答、检索和时间定位等任务上表现出色，但其计算与内存成本随帧数和上下文长度的增加而增长，这限制了它们在实时、移动和资源受限环境中的部署。本综述涵盖了视觉和视听VideoLLM中那些报告了在参数量、每输入FLOPs、延迟、内存或视觉与音频token数量方面取得具体削减效果的推理效率机制。我们分析了帧采样、模态编码、连接器级token缩减以及LLM预填充和解码各阶段的瓶颈。我们按照方法作用的流水线阶段对方法进行组织，涵盖2022年末以来开发的VideoLLM以及更早期的帧采样方法。

    arXiv:2609.10355v1 Announce Type: cross  Abstract: Video understanding has rapidly evolved toward video large language models (VideoLLMs): systems that couple video representations with pretrained large language models and condition generation on a textual prompt. Their strong performance on captioning, question answering, retrieval and temporal grounding comes at a computation and memory cost that grows with frame count and context length, limiting deployment in real-time, mobile and resource-constrained settings. This survey covers inference-efficiency mechanisms for visual and audiovisual VideoLLMs that report concrete reductions in parameter count, FLOPs per input, latency, memory, or visual and audio token count. We analyze bottlenecks across frame sampling, modality encoding, connector-level token reduction, and LLM prefilling and decoding. We organize methods by the pipeline stage at which they act, covering VideoLLMs developed since late 2022 together with earlier frame-samplin
    
[^10]: 从符号感知到逻辑推演：一个引导语言模型进行几何推理的框架

    From Symbolic Perception to Logical Deduction: A Framework for Guiding Language Models in Geometric Reasoning

    [https://arxiv.org/abs/2609.10335](https://arxiv.org/abs/2609.10335)

    该论文提出一个将几何图形解析为符号形式并进行形式化逻辑推演的框架，使纯大语言模型在几何推理上达到与最先进多模态模型相当的性能，同时减少幻觉并提升推理的可解释性。

    

    平面几何仍然是人工智能领域的一个重大挑战，它需要视觉感知与数学推理的融合。虽然大型多模态模型（LMMs）能够自然地处理视觉-语言输入，但它们通常计算开销大且缺乏透明度。我们证明了纯大型语言模型（LLM）在配备专门模块的情况下，可以在复杂几何问题上与最先进的多模态模型相媲美。我们的框架集成了几何视觉解析器（将图形转换为符号形式）与符号求解器（执行形式化推演），从而减少幻觉并促进可解释的推理。为了进行严格的评估，我们整理了来自2025年中国中考的具有挑战性问题的基准测试，确保了数据的新颖性并考察更深层次的推演能力。实验表明，我们的方法达到了与Gemini 2.5 Pro相当的性能，同时提供更清晰、更符合人类（思维方式的推理过程）……

    arXiv:2609.10335v1 Announce Type: cross  Abstract: Plane geometry remains a significant challenge in AI, requiring the integration of visual perception and mathematical reasoning. While Large Multimodal Models (LMMs) naturally handle visuo-linguistic inputs, they are often computationally intensive and opaque. We demonstrate that a pure Large Language Model (LLM), when equipped with specialized modules, can rival state-of-the-art LMMs on complex geometry problems. Our framework integrates a Geometric Vision Parser, which translates diagrams into symbolic form, with a Symbolic Solver that performs formal deductions, thereby mitigating hallucinations and promoting interpretable reasoning. To enable rigorous evaluation, we curate a benchmark of challenging problems from the 2025 Chinese Zhongkao examinations, ensuring data novelty and testing deeper deductive skills. Experiments demonstrate that our approach achieves performance comparable to Gemini 2.5 Pro while delivering clearer, human
    
[^11]: 面向视觉-语言模型适配的在策略蒸馏：一种在低质量多模态数据上的有效范式

    On-Policy Distillation for Vision-Language Model Adaptation, an Effective Paradigm on Low-Quality Multimodal Data

    [https://arxiv.org/abs/2609.10321](https://arxiv.org/abs/2609.10321)

    提出OnPoKD框架，首次将在策略蒸馏应用于视觉-语言模型适配，通过轻量级控制器动态构建样本级自适应蒸馏目标，从而在低质量多模态数据上实现更可靠的模型迁移。

    

    知识蒸馏为将经过任务适配的视觉-语言教师模型迁移到紧凑的学生模型提供了一条高效途径。当前视觉-语言蒸馏方法中的训练目标通常由教师模型的预测构建，并均匀地应用于所有训练样本，这使其在类别偏移和领域偏移的情况下变得不可靠。在本文中，我们认为蒸馏目标的构建应当被视为一种动态的训练决策，而非固定的配方。为此，我们提出了OnPoKD，一个用于视觉-语言模型适配的在策略蒸馏框架。据我们所知，OnPoKD是首个通过将目标构建学习为策略决策，将在策略蒸馏应用于视觉-语言模型适配的框架。OnPoKD学习一个轻量级控制器，利用来自教师模型、学生模型和零样本预测的可靠性与分歧线索，构建样本级自适应的蒸馏目标。

    arXiv:2609.10321v1 Announce Type: new  Abstract: Knowledge distillation offers an efficient route to transfer a task-adapted vision-language teacher to a compact student. The training target in current vision-language distillation methods is typically constructed from the teacher prediction and applied uniformly to all training samples, making it unreliable under class and domain shifts. In this paper, we argue that distillation target construction should be treated as a dynamic training decision rather than a fixed recipe. To this end, we propose OnPoKD, an on-policy distillation framework for vision-language model adaptation. To the best of our knowledge, OnPoKD is the first framework that applies on-policy distillation to vision-language model adaptation by learning target construction as a policy decision. OnPoKD learns a lightweight controller that constructs sample-wise adaptive targets using reliability and disagreement cues from the teacher model, student model, and zero-shot p
    
[^12]: RiLM：通过测地线解码实现参数高效的语言建模

    RiLM: Parameter-Efficient Language Modeling via Geodesic Decoding

    [https://arxiv.org/abs/2609.10305](https://arxiv.org/abs/2609.10305)

    提出RiLM框架，完全移除传统输出矩阵，通过在黎曼流形上计算当前状态与词表嵌入之间的测地线距离平方来直接解码下一个词元概率，其双曲版本HypRiLM在约29万参数下于WikiText-2上达到54.2的验证困惑度，显著优于平坦版本及同规模的LSTM、Transformer和SSM基线。

    

    百万参数以下的语言模型对边缘部署、领域适配和可复现研究非常重要，然而在嵌入维度 d = 128 下，两层 LSTM 或 Transformer 仍将大约三分之一的容量耗费在位于 R^(d x |V|) 中的输出矩阵 W_out 上。我们提出黎曼语言模型，它完全移除了该输出层：上下文展开为黎曼流形上的轨迹，下一个词元的概率由当前状态与词表嵌入之间的测地线距离平方产生。同一个嵌入映射同时服务于输入和输出——解码即几何。我们在平坦空间 R^d（Flat RiLM）和庞加莱球 H^d（HypRiLM）上实例化了该框架，并使用共享的 MLP 组合映射 phi（约29万参数，d = 128，|V| = 2000）。在 WikiText-2 上五个随机种子的实验中，HypRiLM 达到 54.2 ± 0.2 的验证困惑度，而 Flat RiLM 为 87.6 ± 0.6；同等参数规模的 LSTM、Transformer 和 SSM 对照模型仍然处于（原文截断）。

    arXiv:2609.10305v1 Announce Type: new  Abstract: Language models under one million parameters matter for edge deployment, domain adaptation, and reproducible research, yet a two-layer LSTM or Transformer at embedding width d = 128 still spends roughly one third of its capacity on the output matrix W_out in R^(d x |V|). We propose Riemannian Language Models (RiLM), which remove that layer entirely: context unfolds as a trajectory on a Riemannian manifold, and next-token probabilities arise from squared geodesic distance between the current state and vocabulary embeddings. The same embedding map serves input and output -- decoding is geometry. We instantiate the framework on flat R^d (Flat RiLM) and the Poincare ball H^d (HypRiLM) with a shared MLP composition map phi (~290k parameters, d = 128, |V| = 2000). Across five seeds on WikiText-2, HypRiLM reaches 54.2 +/- 0.2 validation perplexity versus 87.6 +/- 0.6 for Flat RiLM; tied and matched LSTM, Transformer, and SSM controls remain at 
    
[^13]: 语义瓶颈：利用语义表示实现非侵入式语音解码

    The Semantic Bottleneck: Leveraging Semantic Representations for Non-Invasive Speech Decoding

    [https://arxiv.org/abs/2609.10296](https://arxiv.org/abs/2609.10296)

    提出Brain2Semantics2Text方法，通过语义嵌入空间作为瓶颈，将句子级MEG信号映射到语义流形并逆向转换为文本，实现了无需词级对齐的非侵入式语音解码。

    

    非侵入式语音解码一直受限于神经记录的低信噪比，这使得对音素或单个单词的细粒度重建变得困难。受神经科学证据的启发——高级语义表示分布在大脑皮层的多个区域，并随较慢的时间尺度演化——我们假设语义内容可能比低级声学或词汇特征更适合作为非侵入式解码的目标。我们提出了Brain2Semantics2Text方法，通过一个中间语义嵌入空间来重建文本。我们的模型将句子级别的MEG（脑磁图）响应映射到语义流形中，然后将预测出的嵌入逆向转换为自然语言。这种语义瓶颈机制使得无需词级对齐即可恢复高级语义信息。我们描述了该方法的核心原理、具体实现，以及用于缓解相关挑战的策略。

    arXiv:2609.10296v1 Announce Type: new  Abstract: Non-invasive speech decoding remains constrained by the low signal-to-noise ratio of neural recordings, which makes fine-grained reconstruction of phonemes or individual words difficult. Motivated by neuroscientific evidence that high-level semantic representations are distributed across cortical regions and evolve over slower temporal scales, we hypothesize that semantic content may provide a more suitable target for non-invasive decoding than low-level acoustic or lexical features. We introduce Brain2Semantics2Text, a method that reconstructs text through an intermediate semantic embedding space. Our model maps sentence-level MEG responses into a semantic manifold and then inverts the predicted embeddings into natural language. This semantic bottleneck enables recovery of high-level meaning without word-level alignment. We describe the core principles of the approach, its implementation, and the strategies used to mitigate the challeng
    
[^14]: GANDR：面向可验证法律答案生成的论断审计

    GANDR: Claim Auditing for Verifiable Legal Answer Generation

    [https://arxiv.org/abs/2609.10293](https://arxiv.org/abs/2609.10293)

    GANDR是一个双智能体系统，由起草器生成结构化法律答案、批评者逐条对照引用来源审计论断，并配合要求每条引用必须命中检索结果的严格正确性标准，从而实现可逐条验证的法律答案生成。

    

    在法律实践等高风险领域，语言模型生成的答案只有在读者能够对照系统所引用的来源逐条验证每个论断时才有用。当前的有据生成流水线将答案作为一个整体进行评分，因此一个正确的结论可能建立在捏造的或匹配松散的引用之上，却仍能获得高分。弥合这一差距既需要一个为逐条论断验证而构建的系统，也需要一种能够衡量它的评估方法。我们提出了GANDR（Grounded ANswer DRafter，有据答案起草器），这是一个双智能体系统：起草器以结构化的法律推理格式撰写答案，而一个独立的批评者——拥有与人类验证者相同的视角——针对所引用的来源审计每个论断，并在每一轮输出逐条论断的审计轨迹。我们为其配备了一项严格的正确性标准，要求每条引用都能对应到检索器返回的某段文本。在一个包含185个条目的法律基准上，所有六个系统共享同一个骨干模型和一个检索源……（摘要在此处被截断）

    arXiv:2609.10293v1 Announce Type: new  Abstract: In high-stakes domains such as legal practice, a language-model answer is only useful to the extent that a reader can verify each claim against the source the system cites. Current grounded-generation pipelines score the answer as a whole, so a correct conclusion can rest on fabricated or loosely matched citations and still score well. Closing this gap requires both a system built for per-claim verification and an evaluation that measures it. We introduce GANDR (Grounded ANswer DRafter), a two-agent system in which a Drafter writes an answer in a structured legal-reasoning format and a separate Critic, with the same view as a human verifier, audits each claim against its cited source and emits a per-claim audit trace on every round. We pair it with a strict correctness criterion requiring every citation to resolve to a passage the retriever returned. On a 185-item legal benchmark where all six systems share one backbone, one retrieval su
    
[^15]: KVShareArena：跨上下文与模型检查点的KV缓存复用

    KVShareArena: KV-Cache Reuse Across Contexts and Model Checkpoints

    [https://arxiv.org/abs/2609.10266](https://arxiv.org/abs/2609.10266)

    KVShareArena是首个针对跨提示上下文和模型检查点的KV缓存复用方法进行系统基准测试的平台，通过衡量各方法在无缓存与完整缓存之间性能差距的恢复比例来评估RAG检索片段和多智能体报告场景下的缓存修复技术。

    

    LLM服务系统已经在复用KV缓存了，但仅当被复用的文本位于提示词的最开头时。两种日益增长的工作负载打破了这一条件：检索增强生成（RAG）服务器会为每个查询组装一组不同的检索片段，而多智能体协调器会读取其他智能体撰写的报告。当缓存被复用于新的提示词中时，它会带有错误的位置信息，且从未关注过其他来源。此外，缓存也可能是由同一模型家族的不同检查点写入的，这会改变存储的数值。针对这类缓存的修复方法分别出现在三个不同的研究社区中，每个社区都按照自己的标准进行测量，而现有的基准测试只测试精确前缀复用（在这种情况下不会丢失任何信息）。KVShareArena在检索片段和智能体报告场景下，对跨提示上下文和模型检查点的KV缓存复用进行基准测试。它通过衡量每种方法在“无缓存”与“完整缓存”之间所恢复的差距比例来为各种方法评分。

    arXiv:2609.10266v1 Announce Type: new  Abstract: LLM serving systems already reuse KV caches, but only when the reused text sits at the very start of the prompt. Two growing workloads break this condition: a retrieval-augmented generation server assembles a different set of retrieved chunks for every query, and a multi-agent coordinator reads reports written by other agents. Reused inside a new prompt, a cache carries the wrong positions and never attended to the other sources. The cache may also have been written by a different checkpoint of the same model family, which changes the stored values. Repair methods for such caches have appeared in three separate communities, each measured on its own terms, and existing benchmarks test only exact-prefix reuse, where nothing is lost. KVShareArena benchmarks KV-cache reuse across prompt contexts and model checkpoints on retrieved chunks and agent reports. It scores every method by the fraction of the gap it recovers between no cache and full
    
[^16]: DiSCo：一个分布优先的引导与文化先验评估框架，用于测量大语言模型中的文化偏好偏差

    DiSCo: A Distribution-First Steering and Cultural Prior Evaluation Framework for Measuring Cultural Preference Bias in LLMs

    [https://arxiv.org/abs/2609.10253](https://arxiv.org/abs/2609.10253)

    DiSCo是一个分布优先的评估框架，通过隔离默认文化先验并利用四级上下文梯度测试可引导性，发现大语言模型的文化偏好先验严重集中在英美文化上（合计约占35%）。

    

    大语言模型（LLMs）越来越多地被部署于全球范围内使用的智能助手中，然而它们在根植于文化的日常情境中所做的默认选择，可能会系统性地偏向某些文化，从而影响本地化效果、用户信任以及公平的行为表现。现有的文化基准测试以单一“正确”答案来评估准确率，这使得当多个根植于文化的回答都同样有效时，难以刻画LLM的文化偏好先验；同时，这些基准还将默认偏好与上下文驱动的适应性调整混为一谈。我们提出了DiSCo，这是一个分布优先的强制选择评估框架，能够隔离默认文化先验，并通过四级上下文梯度（C0–C3）来测试模型的可引导性。使用从BLEnD衍生、涵盖12种文化的DiSCo-Bench（304个条目），我们评估了六个多样化的指令微调LLM。结果显示，默认文化先验高度集中，其中英国和美国合计占据了约35%的所有选择。

    arXiv:2609.10253v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed in globally used assistants, yet their default choices in culturally grounded everyday situations can systematically favour some cultures over others, affecting localisation, user trust, and equitable behaviour. Existing cultural benchmarks evaluate accuracy against a single "correct" answer, making it difficult to characterise an LLM's cultural preference prior when multiple culturally grounded responses are all valid; they also conflate default preferences with context-driven adaptation. We propose DiSCo, a distribution-first forced-choice evaluation framework that isolates default cultural priors and tests steerability via a four-level context gradient (C0--C3). Using DiSCo-Bench (304 items) derived from BLEnD spanning 12 cultures, we evaluate six diverse instruction-tuned LLMs. Default priors are heavily concentrated, with UK and US together absorbing approximately 35\% of all se
    
[^17]: 用于视觉语言模型幻觉检测的双词元特征与大小模型集成方法

    Two-Token Features and Small-Large Ensembles for VLM Hallucination Detection

    [https://arxiv.org/abs/2609.10244](https://arxiv.org/abs/2609.10244)

    该论文提出将微调的40亿参数小模型（读取自身隐藏状态的双词元特征）与约4000亿参数的零样本大模型评判器集成，用于视觉语言模型的字符级幻觉检测，并结合大模型生成的合成幻觉数据增强集成多样性，在SHROOM-Visions 2026共享任务四种语言中均取得前八名的成绩。

    

    我们展示了参加SHROOM-Visions 2026共享任务（字符级视觉语言模型幻觉检测）的系统。我们微调了一个小型（40亿参数）视觉语言模型作为逐词元分类器，该分类器从其自身的隐藏状态中读取双词元特征，并在预测时与一个约4000亿参数的零样本视觉语言模型评判器进行集成。两个组件都能看到对图像中任何可见文字的标准OCR识别结果。我们使用由大模型生成的合成幻觉数据作为集成多样性的来源，并通过验证集来选择特征层、训练数据和OCR接地方式。我们的正式参赛系统在隐藏测试集上达到了平均Cor 0.487 / Cor-lbl 0.387的成绩，在任务的主要Cor-lbl指标上分别位列第6/28（英语）、第6/21（法语）、第8/21（意大利语）和第7/22（中文）。

    arXiv:2609.10244v1 Announce Type: new  Abstract: We present our system for the SHROOM-Visions 2026 shared task on character-level VLM hallucination detection. A small ($4$B-parameter) VLM is fine-tuned as a per-token classifier reading a two-token feature from its own hidden states, and is ensembled with a $\sim$400B zero-shot VLM judge at prediction time. Both components see off-the-shelf OCR of any visible in-image text. We use synthetic hallucination data generated by the large model as a source of ensemble diversity, and use validation to select feature layer, training data and OCR grounding. Our official entry reaches mean Cor $0.487$ / Cor-lbl $0.387$ on the hidden test set, placing $6$th/$28$ (EN), $6$th/$21$ (FR), $8$th/$21$ (IT) and $7$th/$22$ (ZH) on the task's primary Cor-lbl metric.
    
[^18]: LiteRAG：低成本高效的基于图的检索增强生成

    LiteRAG: Cost-Efficient Graph-Based Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.10239](https://arxiv.org/abs/2609.10239)

    LiteRAG通过查询条件算法探索和推理链上下文构建取代昂贵的检索时LLM控制，在多跳问答质量上达到最优，同时将查询延迟降低100倍以上、成本降低99%以上、token使用量减少约14倍。

    

    基于图的检索可以提升多跳问答的效果，但现有方法往往在查询时产生高昂成本，并生成分散且过于庞大的上下文，从而降低生成效率。我们提出了LiteRAG，这是一种基于图的检索方法，它用基于查询条件的算法探索和推理链上下文构建，取代了昂贵的检索时LLM控制。在DistComp（一个针对分布式系统论文的多跳检索基准）上，LiteRAG在所评估的方法中获得了最高的整体质量得分（0.798），同时与GraphRAG Global和DRIFT相比，每次查询的延迟降低了100倍以上，成本降低了99%以上。在UltraDomain上，它在整体质量上与LinearRAG相当，但使用的token数量减少了约14倍。消融实验表明，LiteRAG的查询自适应阈值机制和社区感知的中心节点惩罚是其token效率提升的主要驱动因素。

    arXiv:2609.10239v1 Announce Type: cross  Abstract: Graph-based retrieval can improve multi-hop question answering, but existing approaches often incur high query-time costs and produce diffuse, oversized contexts that reduce generation efficiency. We present LiteRAG, a graph-based retrieval method that replaces expensive retrieval-time LLM control with query-conditioned algorithmic exploration and reasoning-chain context construction. On DistComp, a benchmark for multi-hop retrieval over distributed-systems papers, LiteRAG attains the highest overall quality among the evaluated methods (0.798) while reducing per-query latency by over 100$\times$ and cost by over 99% relative to GraphRAG Global and DRIFT. On UltraDomain, it matches LinearRAG on overall quality while using about 14$\times$ fewer tokens. An ablation study indicates that LiteRAG's query-adaptive thresholding and community-aware hub penalization are the main drivers of its token-efficiency gains.
    
[^19]: 知识图谱上大语言模型问答中的答案路径与接地指令

    The Answer Path and the Grounding Instruction in LLM Question Answering over Knowledge Graphs

    [https://arxiv.org/abs/2609.10237](https://arxiv.org/abs/2609.10237)

    本研究在六个大语言模型和两个知识图谱问答基准上系统变化四种提示设计选择，发现只有将答案路径纳入提示和接地指令的设计能显著影响答案准确率（非路径三元组可被无关内容替换而不影响效果，检索预算应全部用于保证召回率），而三元组的书写语法与排列顺序几乎没有作用。

    

    图检索增强生成流水线需要做出四种选择：将哪些三元组放入提示中、用什么语法书写它们、以什么顺序排列它们，以及用一句话告诉模型如何处理这些三元组。我们在六个大语言模型和两个知识图谱问答基准上对这四种选择进行了系统变化实验。结果表明，四种选择中有两种会影响答案表现，另外两种则几乎没有作用。第一种关键选择是答案路径——即推理出答案所需的三元组链——是否包含在提示中。在保持三元组总数不变的情况下，将所有不在路径链上的三元组替换为来自无关实体的材料，答案准确率仅变化+0.003 F1；而移除该路径链则会损失图检索所带来的绝大部分价值。因此，检索预算应当投入于召回率，在我们可测试的范围内，精确率并不能带来任何收益。本文中并不涉及检索器：子图来自黄金标准SPARQL，因此此处的精确率描述的是我们所构建的上下文质量，而非某个系统设置。第二种关键选择是接地指令……（摘要在此处截断）

    arXiv:2609.10237v1 Announce Type: new  Abstract: A graph retrieval-augmented generation pipeline chooses which triples to put in the prompt, a syntax to write them in, an order to write them in, and a sentence telling the model what to do with them. We vary all four over six large language models and two knowledge-graph question answering benchmarks. Two of the four choices move the answer and the other two are flat. The first is whether the answer path, the triples needed to reach the answer, is in the prompt at all. Holding the number of triples fixed and replacing every triple that is not on the chain with material from an unrelated entity changes answer accuracy by +0.003 F1, while removing the chain costs most of what the graph was worth. Retrieval budget belongs on recall, and precision in the range we can test buys nothing. There is no retriever here: subgraphs come from gold SPARQL, so precision describes the context we build, not a system setting. The second is the grounding i
    
[^20]: Φ-Bench：大语言模型能否工程化构建驱动它们自身的基础设施？

    $\Phi$-Bench: Can Large Language Models Engineer the Infrastructure That Powers Them?

    [https://arxiv.org/abs/2609.10226](https://arxiv.org/abs/2609.10226)

    提出了Φ-Bench基准，通过源自前沿研究和真实代码仓库的任务，系统性评估大语言模型在工程化构建其自身基础设施栈方面的开放式、长周期工程能力。

    

    大语言模型（LLMs）在推理和代码生成方面展现出卓越的能力，这引出了一个前景：它们或许能够协助开发和优化驱动它们自身的核心基础设施。然而，现有基准主要聚焦于孤立的内核、预定义的算子或预先指定的优化目标，因此无法评估大语言模型执行开放式、长周期的大语言模型基础设施工程的能力。为弥补这一空白，我们提出了Φ-Bench，这是一个用于系统性评估大语言模型工程化构建大语言模型基础设施栈能力的基准。Φ-Bench源自前沿研究中研究的优化问题，并以真实世界的代码仓库为基础，广泛覆盖大语言模型基础设施栈，涵盖不同复杂度的任务，从局部内核级的函数补全到长周期的实现以及端到端的系统优化。大量

    arXiv:2609.10226v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated remarkable capabilities in reasoning and code generation, raising the prospect that they could assist in developing and optimizing the very infrastructure that powers them. However, existing benchmarks mainly focus on isolated kernels, predefined operators, or pre-specified optimization targets, and therefore fail to evaluate the ability of LLMs to perform open-ended, long-horizon LLM infrastructure engineering. To address this gap, we present $\Phi$-Bench, a benchmark for systematically evaluating LLMs on engineering the LLM infrastructure stack. Derived from optimization problems studied in frontier research and grounded in real-world code repositories, $\Phi$-Bench provides broad coverage of the LLM infrastructure stack and spans tasks of varying complexity, ranging from localized kernel-level function completion to long-horizon implementation and end-to-end system optimization. Extensive
    
[^21]: 透过镜中世界：直接读取与写入Transformer

    Through the Looking Glass: Directly Reading and Writing Transformers

    [https://arxiv.org/abs/2609.10210](https://arxiv.org/abs/2609.10210)

    该研究通过对组件贡献进行符号化净值分析，发现transformer的一次预测实际仅依赖8至53个关键组件，且预测所动用的模型比例（仅1%-3%）不随模型规模增长，所有结果均直接从模型自身的参数与激活中读取。

    

    Transformer的多少个组件决定了一个token？如果按每个单元和通道对logit贡献的绝对值来计算，一次预测依赖于数千到数十万个组件。但贡献是有符号的，在十八个模型中，推离预测token的质量中位数是携带它的质量的七倍。除以净值后，数量缩减为几十个：在基线模型上，53个组件承载了预测的百分之九十，13个组件是模型失去后无法存活的，而仅8个组件就足以单独产生该预测。在十二个于其他环境训练的模型（参数量从1.24亿到70亿）中，充分集合从两个组件到十六个不等，而一次预测所依赖的内容，一路追溯回去，只占整个模型的百分之一到百分之三，且这一比例不随模型规模增长。一层更新的四分之三是其所接收状态的固定线性映射。所有内容均直接从模型自身的参数和激活中读取，……

    arXiv:2609.10210v1 Announce Type: new  Abstract: How many of a transformer's components decide a token? Counted by the absolute value of each unit's and channel's contribution to the logit, one prediction rests on thousands to hundreds of thousands of them. But contributions are signed, and across eighteen models the mass pushing away from the predicted token is a median of seven times the mass carrying it. Divide by the net and the count is dozens: on the baseline, 53 components carry ninety percent of a prediction, 13 it cannot survive losing, and 8 suffice to produce it alone. Across twelve models trained elsewhere, 124M to 7B parameters, the sufficient set runs from two components to sixteen, and what a prediction draws on, followed all the way back, is one to three percent of the model, a share that does not grow with size. Three quarters of a layer's update is a fixed linear map of the state it received.   Everything is read from the model's own parameters and activations, with n
    
[^22]: 情感政治：美国国会的情感表达与立法效力

    Politics of Feelings: Emotional Expression and Legislative Effectiveness in the U.S. Congress

    [https://arxiv.org/abs/2609.10198](https://arxiv.org/abs/2609.10198)

    本研究基于Transformer情感分类器分析了1973至2024年间美国国会超过170万篇演讲中的八种离散情感，首次系统揭示了国会演讲的情感表达随时间日益增强，且与政策领域、议员意识形态立场及立法效力存在显著关联。

    

    情感是政治交流中普遍存在的特征，然而现有研究主要集中于描述情感表达的模式，而非考察这些情感是否与重要的立法结果相关联。我们通过研究1973年至2024年间美国国会超过170万篇演讲中离散情感的表达及其相关因素来填补这一研究空白。利用基于Transformer的情感分类器，我们测量了八种离散情感：愤怒、恐惧、厌恶、悲伤、喜悦、热情、自豪和希望。我们考察了这些情感如何随时间变化、如何因政策议题和议员特征而异，以及它们与立法效力之间的关系。我们发现，国会演讲的情感表达随时间推移日益增强。情感表达在不同政策领域和议员的意识形态立场上也呈现系统性差异。值得注意的是，情感与立法效力之间的关联……（摘要原文在此处截断）

    arXiv:2609.10198v1 Announce Type: new  Abstract: Emotions are a pervasive feature of political communication, yet existing research has focused primarily on describing patterns of emotional expression rather than examining whether they are associated with consequential legislative outcomes. We address this gap by investigating the expression and correlates of discrete emotions in more than 1.7 million speeches delivered in the U.S. Congress between 1973 and 2024. Using a transformer-based emotion classifier, we measure eight discrete emotions: anger, fear, disgust, sadness, joy, enthusiasm, pride, and hope. We examine how these emotions vary over time, across policy topics, legislator characteristics, and their relationship with legislative effectiveness. We find that congressional speeches are becoming emotionally expressive over time. Emotional expression also varies systematically across policy domains and ideological positioning of legislators. Notably, the relationship between emo
    
[^23]: 谁在论证什么？政治辩论中的论证与实体联合检测和分类

    Who Argues What? Joint Argument-Entity Detection and Classification in Political Debates

    [https://arxiv.org/abs/2609.10192](https://arxiv.org/abs/2609.10192)

    本文提出了实体增强的政治辩论数据集DNE-ElecDeb，并引入生成式联合标注框架JAET，首次实现了政治辩论中论证片段与辩论命名实体的联合检测与分类，填补了该领域数据和方法上的空白。

    

    政治辩论通常通过论证挖掘（AM）进行分析，以研究驱动辩论的关键论证。然而，政治论证很少能够仅从论证片段本身进行解读，因为论点和前提通常依赖于它们所提及的实体（例如人物、事件、地点、政党）。现有的论证挖掘资源和方法通常对论证片段及其角色进行标注，但没有提供配对的辩论-实体层，无法解答辩论中引用了哪些辩论命名实体（DNE），例如参与者和事件。在这项工作中，我们通过以下方式弥补这些数据和方法上的空白：（i）引入DNE-ElecDeb，这是USElecDeb数据集的实体增强版本，在论证和非论证片段中都添加了辩论命名实体，并将辩论命名实体识别（DNER）定义为检测辩论命名实体的任务；（ii）提出联合论证与实体标注（JAET），一个通过微调仅解码器大型语言模型的生成式框架……（摘要不完整，在此截断）

    arXiv:2609.10192v1 Announce Type: new  Abstract: Political debates are often analyzed through Argument Mining (AM) to investigate the key arguments that drive them. However, political arguments are rarely interpretable from argumentative spans alone, as claims and premises generally depend on the entities (e.g., people, events, locations, parties) they mention. Existing AM resources and methods typically annotate argumentative spans and roles, but do not provide a paired debate-entity layer for asking which Debate Named Entities (DNE), e.g., actors and events, are invoked within debates. In this work, we address these data and methodological gaps by (i) introducing DNE-ElecDeb, an entity-enriched version of the USElecDeb dataset that adds DNEs in both argumentative and non-argumentative spans and defines Debate Named Entity Recognition (DNER) as the task of detecting DNEs, and (ii) proposing Joint Argument and Entity Tagging (JAET), a generative framework that fine-tunes decoder-only L
    
[^24]: 从检索到权重：利用个人文本语料库实现小语言模型的参数化个体化

    From Retrieval to Weights: Parametric Individualization of Small Language Models with Individual Text Corpora

    [https://arxiv.org/abs/2609.10155](https://arxiv.org/abs/2609.10155)

    该研究通过DoRA微调将515名参与者的个人搜索历史写入小语言模型权重，证明适配器能显著编码个人语料（个体化效应dz=1.27），但在通用知识测试中模型获得的是知识而非与个体的对齐。

    

    我们从认知模拟的视角研究多选题问答中的情景记忆与语义记忆，方法是将个人文本语料库（ITC）中的文本融入检索增强生成和DoRA微调。我们通过网络爬取了515名回答36道多选题知识项的参与者的搜索历史，并分析了其中150名参与者的分层子样本。对于每位参与者，一个DoRA适配器将其个人文本语料库整合进一个小语言模型（SLM），该模型的基线正确率低于参与者群体的最低四分位数。该适配器可测量地将个人文本语料库写入权重之中：它对参与者自身留出文本的拟合优于其他参与者的文本（dz = 1.27），且这种个体化效应随个人文本语料库规模按秩次递增。然而，在通用知识测试中，适配器增加的是知识而非与个体的对齐：对数损失的匹配度有所提升，而在偏差下的匹配准确率……（原文摘要于此处截断）

    arXiv:2609.10155v1 Announce Type: new  Abstract: We approach a cognitive simulation perspective on episodic and semantic memory in multiple-choice question answering by incorporating text from individual text corpora (ITC) into retrieval-augmented generation and DoRA fine-tuning. We web-crawl the search histories of 515 participants who answered 36 multiple-choice knowledge items and analyze a stratified subsample of 150 participants. For each participant, one DoRA adapter consolidates their ITC into a small language model (SLM) whose baseline correctness falls below the participants' lowest quartile. The adapter measurably writes the ITC into the weights: it fits its own participant's held-out text better than other participants' texts (dz =1.27), an individuality effect that increases with ITC size in rank order. On the generalized knowledge test, however, the adapter adds knowledge rather than alignment with the individual: log-loss match improves, whereas match accuracy under a bia
    
[^25]: YallaMorph：用于评估大语言模型阿拉伯语形态生成能力的基准测试

    YallaMorph: A Benchmark for Evaluating Arabic Morphological Generation in Large Language Models

    [https://arxiv.org/abs/2609.10153](https://arxiv.org/abs/2609.10153)

    本文提出YallaMorph——一个涵盖60万条目的大规模阿拉伯语形态生成基准，评估结果显示大语言模型在阿拉伯语形态生成上仍然存在显著困难，尤其是附着词素化、未见和形态罕见的形式。

    

    阿拉伯语形态学对大语言模型而言仍然具有挑战性，因为流畅的文本生成并不能保证准确的形态句法控制。现有的阿拉伯语评估主要针对下游任务，并未直接测试基于显式词汇和特征输入的受控形态生成。我们推出了YallaMorph，一个大规模的阿拉伯语形态生成基准，涵盖动词、名词、形容词、其附着词素化形式以及无效配置。我们在带音符和不带音符两种设置下，基于60万条基准条目对多语言及面向阿拉伯语的大语言模型进行了评估。结果表明，阿拉伯语形态生成仍然困难，尤其是对于附着词素化形式、未见过的形式以及形态上罕见的形式。

    arXiv:2609.10153v1 Announce Type: new  Abstract: Arabic morphology remains challenging for large language models, since fluent generation does not guarantee accurate morphosyntactic control. Existing Arabic evaluations mainly target downstream tasks and do not directly test controlled morphological generation from explicit lexical and feature-based input. We introduce YallaMorph, a large-scale benchmark for Arabic morphological generation covering verbs, nouns, adjectives, their cliticized forms, and invalid configurations. We evaluate multilingual and Arabic-oriented LLMs under diacritized and undiacritized settings over 600K benchmark entries. Results show that Arabic morphological generation remains difficult, especially for cliticized, unseen, and morphologically rare forms.
    
[^26]: 主动适应，而非静态防御：对抗性微调中预防性引导的时间动力学

    Active Adaptation, Not Static Defense: Temporal Dynamics of Preventative Steering in Adversarial Fine-Tuning

    [https://arxiv.org/abs/2609.10142](https://arxiv.org/abs/2609.10142)

    该研究揭示预防性引导的持久防护源于早期的主动补偿性适应而非静态权重偏移，并据此提出渐进式干预方法来强化对抗性微调防御。

    

    大语言模型在面对恶意微调时依然脆弱，这促使研究者探索针对有害人格漂移的训练时防御方法。预防性引导在微调过程中注入不良特质的人格向量，并在评估时将其移除，但其持久保护背后的机制仍不清楚。通过分析其时间优化动力学，我们发现该防御源于早期的补偿性适应阶段，随后进入矫正信号逐渐衰减的稳态阶段；在参数空间中，注意力输出投影成为防御性更新的主要残差写入路径。通过干预增量保留（IDP）与 IDP 延续实验，我们进一步证明，保留或重新注入权重偏移无法维持保护效果，这表明预防性引导依赖于主动适应而非静态防御。受此发现启发，我们提出了渐进式干预

    arXiv:2609.10142v1 Announce Type: new  Abstract: Large language models remain fragile against malicious fine-tuning, motivating training-time defenses against harmful persona drift. Preventative Steering injects undesirable-trait persona vectors during fine-tuning and removes them at evaluation time, yet the mechanism behind its lasting protection remains unclear. Analyzing its temporal optimization dynamics, we find that the defense emerges from an early compensatory adaptation phase followed by a steady-state phase where the corrective signal decays; in parameter space, attention output projections emerge as the dominant residual-write route for defensive updates. Through Intervention Delta Preservation (IDP) and IDP Continuation experiments, we further show that preserving or reinjecting the weight offset fails to maintain protection, indicating that preventative steering relies on active adaptation rather than a static defense. Motivated by this finding, we propose Progressive Inte
    
[^27]: 如果程序没有错误，就不要修复它：关于使用大语言模型进行迭代错误修复动态的研究

    If It's Not Buggy, Don't Fix It: On the Dynamics of Iterative Bug-fixing with LLMs

    [https://arxiv.org/abs/2609.10123](https://arxiv.org/abs/2609.10123)

    研究发现大语言模型在迭代修复中会对无错误程序“无中生有”地报错，修复率低于破坏率，并常陷入无限增删相同更改的伪修复循环，其机制根源是模型内部“错误代码”表示被错误激活。

    

    大语言模型（LLM）在软件开发中已无处不在，基于LLM的自动程序修复工具在代码审查中的使用日益增多。在本报告中，我们探索了将LLM盲目迭代用作错误修复工具的情况。在多个模型和修复环境中，我们发现LLM持续声称在完全没有错误的程序中检测到错误，而对有错误程序的修复率低于其对正确程序造成的破坏率。我们还探索了这一迭代过程的长期动态，发现其经常陷入一种伪错误修复循环，即相同的更改被无限次地添加又删除。最后，通过机制探测，我们揭示了控制编辑倾向的转向向量（steering vector）的存在，这表明LLM拥有关于“错误代码”的内部表示，而正是这种表示被错误激活从而诱发了伪错误修复。这些结果表……

    arXiv:2609.10123v1 Announce Type: cross  Abstract: Large language models (LLMs) have become ubiquitous in software development, with LLM-based automated program repair tools increasingly used during code review. In this report, we explore the iterative blind use of LLMs as bug-fixers. Across multiple models and repair environments, we find that LLMs consistently claim to detect bugs in entirely bug-free programs while the rate of repair of buggy programs is less than that of the damage to correct programs. We also explore the long-term dynamics of this iterative process, and find that this frequently reaches a pseudo-bug-fixing cycle where the same changes are added and removed again ad infinitum. Lastly, via mechanistic probing, we unveil the existence of a steering vector which controls the editing propensity, suggesting that LLMs have an internal representation of ``buggy code", and that this representation is what is falsely activated to induce pseudo-bug fixing. These results prov
    
[^28]: ProbPlug：用于大语言模型二分类可靠置信度的插件式不确定性网络

    ProbPlug: A Plugin Uncertainty Network for Reliable Confidence in LLM Binary Classification

    [https://arxiv.org/abs/2609.10122](https://arxiv.org/abs/2609.10122)

    ProbPlug是一个轻量级的插件式置信度估计框架，通过自注意力模块聚合冻结LLM的内部token特征来预测分类输出是否正确，无需修改基础模型即可显著提升LLM二分类的置信度可靠性与分类性能。

    

    大型语言模型（LLMs）在广泛的分类任务中取得了出色的性能，但其预测的可靠性仍然是其在高风险场景中部署的主要障碍。尽管针对LLM的置信度估计已被广泛研究，但基于LLM的分类任务的置信度校准仍未得到充分探索。我们提出了ProbPlug，这是一个面向基于LLM的二分类的轻量级置信度估计框架，它利用从冻结LLM中提取的内部token特征来预测输出是否正确。ProbPlug采用自注意力模块来聚合隐藏层表示，并且可以在不修改基础模型的情况下集成到原始推理流程中。在涉及文本和多模态大模型的多个任务上的实验表明，ProbPlug提供了更可靠的置信度估计，以可忽略的额外成本提升了分类性能。

    arXiv:2609.10122v1 Announce Type: new  Abstract: Large language models (LLMs) have achieved strong performance across a broad range of classification settings, yet the reliability of their predictions remains a major obstacle to deployment in high-stakes scenarios. Although confidence estimation for LLMs has been widely studied, confidence calibration for LLM-based classification remains underexplored. We introduce ProbPlug, a lightweight confidence estimation framework for LLM-based binary classification, which predicts whether an output is correct using internal token features extracted from a frozen LLM. ProbPlug employs a self-attention module to aggregate hidden representations and can be integrated into the original inference pipeline without modifying the base model. Experiments across multiple tasks involving both text-based and multimodal large models show that ProbPlug provides more reliable confidence estimates, improves classification performance with negligible additional 
    
[^29]: 面向金融推理的数据中心化后训练：挖掘、蒸馏与可验证学习

    Data-Centric Post-Training for Financial Reasoning: Mining, Distillation, and Verifiable Learning

    [https://arxiv.org/abs/2609.10113](https://arxiv.org/abs/2609.10113)

    该论文提出了一套以数据为中心的后训练流水线，通过挖掘开源推理轨迹、蒸馏金融指令数据和生成知识图谱引导的问答对来构建互补语料库，并利用轻量级分类器筛选数据与基于规则的可验证强化学习，从而有效提升大模型在金融推理任务上的能力。

    

    金融文本、教科书和问答对虽然数量丰富，但其中只有一小部分能直接用于以推理为中心的后训练。现有的问答对往往缺乏明确的推理过程、充分的上下文或可可靠验证的答案，而教科书必须首先被转化为合成训练样本。我们提出了一个以数据为中心的流水线，通过挖掘开源推理轨迹、蒸馏金融指令数据，以及从金融教育材料中生成由知识图谱引导的问答对，来构建互补的语料库。经过语义去重后，三个轻量级序列分类器分别用于筛选与金融相关的样本、拒绝信息不完整的问题，以及识别适合通过紧凑的基于规则的验证器进行强化学习的任务。在模型适配方面，我们研究了监督微调与强化学习，同时结合自蒸馏微调和后训练模型合并（摘要在此处被截断）。

    arXiv:2609.10113v1 Announce Type: new  Abstract: Financial text, textbooks, and question-answer pairs are abundant, but only a small fraction is directly usable for reasoning-focused post-training. Existing QA pairs often lack explicit reasoning, sufficient context, or reliably verifiable answers, while textbooks must first be transformed into synthetic training examples. We present a data-centric pipeline that constructs complementary corpora by mining open-source reasoning traces, distilling financial instruction data, and generating knowledge-graph-guided question-answer pairs from financial educational material. After semantic deduplication, three lightweight sequence classifiers select finance-relevant examples, reject under-specified questions, and identify tasks suitable for reinforcement learning with compact rule-based verifiers. For model adaptation, we study supervised fine-tuning and reinforcement learning, while self-distilled fine-tuning and post-training model merging ar
    
[^30]: RAP：研究关注度预测揭示目标条件化的证据获取偏差

    RAP: Research Attention Prediction Reveals Target-Conditioned Evidence Acquisition Biases

    [https://arxiv.org/abs/2609.10092](https://arxiv.org/abs/2609.10092)

    提出了RAP滚动基准来评估LLM智能体预测研究关注度变化的能力，发现其表现不如简单的EWMA精确计数基线，并揭示了状态前推优于直接预测、以及面向预测的策略倾向于检索较旧证据这两大瓶颈。

    

    大型语言模型越来越多地充当研究智能体，但由于论文评审和研究想法缺乏唯一可验证的结果，其追踪研究关注度变化的能力难以评估。我们提出了研究关注度预测（RAP），这是一个涵盖278个AI/ML领域、共1,390个回合的滚动基准。在每个时间截点，LLM智能体在时间受限的arXiv语料库中进行搜索，并预测未来六个月内八个固定研究方向上的论文份额。搜索通常有所帮助，但在组合准确性方面，所有四个诊断模型的表现均不如基于精确计数的指数加权移动平均（EWMA）基线。我们识别出两个相互关联的瓶颈：在可访问累积历史的条件下，状态前推在所有四个诊断模型上都优于直接预测；冻结证据回放实验将这种逆转的一个共同成分与面向预测的策略检索到的近期证据份额较小联系起来。

    arXiv:2609.10092v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly act as research agents, yet their ability to track shifts in research attention is difficult to evaluate because reviews and research ideas lack uniquely verifiable outcomes. We introduce Research Attention Prediction (RAP), a rolling benchmark covering 278 AI/ML fields and 1,390 episodes. At each cut-off, an LLM agent searches a temporally restricted arXiv corpus and predicts the next six months' paper shares across eight frozen research directions. Search generally helps, but all four diagnostic models perform worse than an exact-count exponentially weighted moving average (EWMA) baseline in compositional accuracy. We identify two linked bottlenecks. Under cumulative-history access, State carry-forward outperforms direct Forecast for all four diagnostic models; frozen-evidence replay links a shared component of this reversal to Forecast-oriented policies retrieving a smaller share of recent e
    
[^31]: NOPE-HYPE：一种面向多样化声学环境下鲁棒语音转文本的结构化仿真工作流

    NOPE-HYPE: A Structured Simulation Workflow for Robust Speech-to-Text Across Diverse Acoustic Environments

    [https://arxiv.org/abs/2609.10058](https://arxiv.org/abs/2609.10058)

    NOPE-HYPE提出了一种结合可控环境模拟器、基于PSD模板的环境约简和超参数搜索的结构化训练工作流，证明模拟噪声可使Whisper和SeamlessM4T等语音翻译模型达到与真实噪声训练相当的性能。

    

    鲁棒的语音转文本翻译系统应当在多样化的声学条件下表现可靠，然而实际的流水线缺乏用于系统性环境探索的可控工具。大型语音模型对未见过的声学条件仍然敏感，因为训练数据很少覆盖真实环境的全部范围。我们提出了NOPE-HYPE，一个结构化的训练工作流，它结合了可控的环境模拟器、基于功率谱密度（PSD）模板的覆盖最优环境约简，以及针对模拟器可调参数的小规模、可解释的超参数搜索。我们证明了模拟器生成的噪声在Whisper和SeamlessM4T模型上取得了与平衡真实噪声训练相当的性能，提供了有原则的环境原型集合，并通过结构化的27次超参数扫描确定了实用的默认模拟器配置。

    arXiv:2609.10058v1 Announce Type: cross  Abstract: Robust speech-to-text translation systems should perform reliably across diverse acoustic conditions, yet practical pipelines lack controllable tools for systematic environment exploration. Large speech models remain sensitive to unseen acoustic conditions, as training data rarely cover the full range of real environments.We present NOPEHYPE, a structured training workflow that combines a controllable environment simulator, coverage-optimal environment reduction on Power Spectral Density (PSD) templates, and a small, interpretable hyperparameter search over simulator knobs. We show that simulator-generated noise achieves performance comparable to balanced realnoise training across Whisper and SeamlessM4T models, provide principled environment prototype sets, and identify practical default simulator configurations from a structured 27-run hyperparameter sweep.
    
[^32]: OntologyAligner：面向生物医学本体规范化的本体对齐检索与层次引导大语言模型重排序方法

    OntologyAligner: Ontology-Aligned Retrieval and Hierarchy-Guided Large Language Model Reranking for Biomedical Ontology Normalization

    [https://arxiv.org/abs/2609.10055](https://arxiv.org/abs/2609.10055)

    提出三阶段框架OntologyAligner（本体对齐检索、大语言模型候选重排序与层次引导精炼），并构建含13,390个样本的统一基准PhenoNormBench，在人类表型本体规范化任务上达到最先进性能。

    

    生物医学本体规范化将自由文本表达式映射到标准化概念，从而实现生物医学数据的一致性整合与分析。由于词汇变体以及层次相关概念之间的细微差别可能模糊概念边界，这项任务仍然极具挑战性。我们提出了OntologyAligner，这是一个三阶段框架，结合了本体对齐检索、大语言模型候选重排序以及选择性层次引导精炼。我们还构建了PhenoNormBench，这是一个统一基准，包含来自七个人类表型本体数据集的13,390个样本。OntologyAligner在HPO规范化任务上取得了最先进的性能，Macro Top-1准确率达88.78%，Micro Top-1准确率达86.75%，分别超过最强基线4.85和5.07个百分点。消融分析显示三个阶段各自做出了互补性贡献，敏感性分析也证明了方法的稳定性。

    arXiv:2609.10055v1 Announce Type: cross  Abstract: Biomedical ontology normalization maps free-text expressions to standardized concepts, enabling consistent integration and analysis of biomedical data. This task remains challenging because lexical variation and subtle distinctions among hierarchically related concepts can obscure concept boundaries. We present OntologyAligner, a three-stage framework that combines ontology-aligned retrieval, large language model candidate reranking, and selective hierarchy-guided refinement. We also construct PhenoNormBench, a unified benchmark comprising 13,390 samples from seven Human Phenotype Ontology datasets. OntologyAligner achieved state-of-the-art performance on HPO normalization, with 88.78% Macro Top-1 Accuracy and 86.75% Micro Top-1 Accuracy, exceeding the strongest baseline by 4.85 and 5.07 percentage points, respectively. Ablation analyses showed complementary contributions from all three stages, and sensitivity analyses demonstrated sta
    
[^33]: 偏好后训练中面向多样化成功轨迹的直接多样性优化

    Direct Diversity Optimization for Diverse Successful Trajectories in Preference Post-Training

    [https://arxiv.org/abs/2609.10052](https://arxiv.org/abs/2609.10052)

    提出了一种名为DDO的离线后训练方法，通过分歧树收集与参考相对目标几率目标相结合，使大语言模型智能体在固定预算下保留并实现多样化的成功策略，在多个环境中显著提升了任务成功率和成功策略覆盖。

    

    用于序列决策任务的大语言模型智能体通常使用轨迹级别的结果标签进行后训练，但这类标签对于保留来自同一决策状态的多个成功分支几乎没有提供监督信号。我们将该问题定义为成功策略覆盖：即在固定的 rollout 预算下，模型能够实现多少种不同的成功策略。我们提出了直接多样性优化，这是一种离线后训练方法，它将分歧树收集与参考相对目标几率目标相结合。DTC 构建以共享决策状态为根节点的状态对齐分支集合，而 RTO 训练模型在成功的备选方案之间匹配参考相对目标。在 BabyAI、BabaIsAI 和 WebShop 三个环境中，DDO 在所有对比的后训练方法中取得了最强的任务成功率和成功策略覆盖。此外，它在局部动作替换后实现了最高的恢复率以及更高的……

    arXiv:2609.10052v1 Announce Type: new  Abstract: LLM agents for sequential decision tasks are often post-trained with trajectory-level outcome labels, but such labels provide little supervision for preserving multiple successful branches from the same decision state. We study this problem as successful strategy coverage: how broadly a model realizes distinct successful strategies under a fixed rollout budget. We present Direct Diversity Optimization (DDO), an offline post-training method that combines Divergence-Tree Collection (DTC) with the Reference-Relative Target-Odds Objective (RTO). DTC constructs state-aligned branch sets rooted at shared decision states, and RTO trains the model to match reference-relative targets over successful alternatives. DDO achieves the strongest task success and successful strategy coverage among the compared post-training methods across BabyAI, BabaIsAI, and WebShop. It also achieves the highest recovery rate after local action replacement and higher 
    
[^34]: MedDeID：基于真实或合成训练数据实现本地化治理的临床文本去标识化

    MedDeID enables locally governed clinical-text de-identification from real or synthetic training data

    [https://arxiv.org/abs/2609.10049](https://arxiv.org/abs/2609.10049)

    MedDeID是一个本地部署的临床文本去标识化框架，无论使用机构内真实数据还是完全合成数据训练，都能高效检测并删除个人可识别信息，使临床笔记在不出机构的前提下安全地用于科研和医疗AI开发。

    

    临床笔记包含个人可识别信息（PII），这限制了其在科研和医疗AI领域的再利用，尤其是当数据无法离开所在机构时。我们开发了MedDeID，这是一个本地部署的框架，将机构内部标注与合成笔记生成同模型训练、推理、假名化和评估相结合。在一个独立标注并经裁定的包含300份荷兰医院笔记的基准测试上，由医院真实数据训练的紧凑型Transformer模型检测出了98.9%的标识性文本，同时仅错误删除了标注标识符之外0.24%的文本；而仅使用合成数据训练的对应模型检测率为96.1%。在100份初级保健笔记上，合成数据训练的模型比医院数据训练的模型取得了更高的召回率（90.3%对87.0%），并且对标识符格式扰动的鲁棒性更强。一个完全未使用真实文本训练的英语版本，在两个外部合成基准上分别检测出99.7%和98.9%的标注标识符字符。这些结果表明，可在机构本地治理的临床文本去标识化是可行的。

    arXiv:2609.10049v1 Announce Type: new  Abstract: Clinical notes contain personally identifiable information (PII), restricting reuse for research and medical AI, especially when data cannot leave an institution. We developed MedDeID, an on-premises framework combining in-house annotation and synthetic-note generation with model training, inference, pseudonymisation and evaluation. On an independently annotated, adjudicated 300-note Dutch hospital benchmark, a hospital-trained compact transformer detected 98.9% of identifying text while redacting 0.24% of text outside annotated identifiers; a synthetic-only counterpart detected 96.1%. On 100 primary-care notes, the synthetic-trained model achieved higher recall than the hospital-trained model (90.3% versus 87.0%) and greater robustness to identifier-format perturbations. An English instantiation trained without real text detected 99.7% and 98.9% of annotated identifier characters on two external synthetic benchmarks. These results demon
    
[^35]: 面向说话人稳定的低资源希腊语TTS的确定性提示方法

    Deterministic Prompting for Speaker-Stable Low-Resource Greek TTS

    [https://arxiv.org/abs/2609.10022](https://arxiv.org/abs/2609.10022)

    该论文通过WhisperX数据整理、确定性提示替代LLM风格提示以及轻量级LoRA微调，仅用3.5小时单说话人数据就实现了说话人一致性接近人类水平的低资源希腊语TTS系统。

    

    现代TTS系统在高资源语言上已接近人类质量，但在干净语音数据稀缺时性能会显著下降。现代希腊语正是这种情况的典型例子，它缺乏支撑最先进合成技术的精心策划语料库。我们提出了一种数据整理方案，通过WhisperX对齐和过滤将有声书录音转化为可用于TTS的数据。随后，我们微调了Parler-TTS（880M参数），这是一个基于提示的多语言模型，其预训练过程编码了可迁移至希腊语的语音先验。在开发过程中，我们发现LLM生成的风格提示会在推理时引入说话人漂移。用确定性提示替代这些风格提示解决了该问题，并且通过在3.5小时单说话人数据上训练的特定说话人LoRA阶段，在仅更新约5%参数的情况下锚定了说话人身份。我们的系统实现了WER 10.7%（仅比ASR下限高2.9）、MOS-I 4.00（人类语音为4.36）以及接近人类的说话人一致性（MOS-C 4.24 vs 4.30），表明构建稳健的单说话人希腊语TTS系统是可行的。

    arXiv:2609.10022v1 Announce Type: cross  Abstract: Modern TTS systems approach human quality for high-resource languages but degrade when clean speech data is scarce. Modern Greek exemplifies this, lacking the curated corpora behind state-of-the-art synthesis. We propose a data curation recipe that transforms audiobook recordings into TTS-ready data via WhisperX alignment and filtering. Then we fine-tune Parler-TTS (880M), a prompt-based multilingual model whose pre-training encodes phonetic priors transferable to Greek. During development, we find that LLM-generated style prompts introduce speaker drift at inference. Replacing them with deterministic prompts resolves this, and a speaker-specific LoRA stage trained on 3.5 h of single-speaker data anchors identity while updating ~5% of parameters. Our system achieves WER 10.7% (2.9 above the ASR floor), MOS-I 4.00 (vs. 4.36 human speech), and near-human speaker consistency (MOS-C 4.24 vs. 4.30), showing that robust single-speaker Greek 
    
[^36]: MetroLLM-Bench：将语言模型作为交通信息亭运行时进行评估

    MetroLLM-Bench: Evaluating Language Models as Transit Kiosk Runtimes

    [https://arxiv.org/abs/2609.10016](https://arxiv.org/abs/2609.10016)

    提出了MetroLLM-Bench基准（包含955个案例），首次系统性地将语言模型作为地铁信息亭策略层进行评估，涵盖六大真实地铁系统中路线规划、票价计算、运营中断、无障碍服务和对抗性输入等11个类别，并通过确定性评分与语义评分双层机制对26个模型进行排名。

    

    我们介绍了MetroLLM-Bench，这是一个包含955个测试案例的基准，用于测试语言模型作为交通信息亭策略层的表现。该基准涵盖六个真实地铁系统（车站数量从37个到414个不等），以及十一个类别，包括路线规划、票价计算、运营中断、无障碍服务和对抗性输入。在每个案例中，模型必须调用结构化工具，并提交一个机器可渲染的终端状态，其中包含处理结果、（如适用）每张车票的票价报价以及信息亭动作。十四个确定性评分组件构成第一层（Tier 1）；八个语义质量组件构成第二层（Tier 2），其中六个由语言模型评判器评分。我们报告第一层得分以及两层综合得分。采用分层75/25划分，保留717个案例用于训练数据生成，238个案例用于保留评估。我们评估了来自六家厂商的二十六个模型，其中二十三个模型被排名。在保留评估分区上，一个通过参数高效……

    arXiv:2609.10016v1 Announce Type: cross  Abstract: We introduce MetroLLM-Bench, a 955-case benchmark for testing language models as the policy layer of a transit kiosk. It covers six real metro systems, ranging from 37 to 414 stations, and eleven categories that include routing, fare calculation, disruptions, accessibility, and adversarial input. In each case, the model must call structured tools and submit a machine-renderable terminal state containing an outcome, a per-ticket fare quote when applicable, and a kiosk action. Fourteen deterministic scoring components form Tier 1; eight semantic-quality components form Tier 2, six of which use a language-model judge. We report Tier 1 and the combined score of both tiers. A stratified 75/25 split reserves 717 cases for training-data generation and 238 for held-out evaluation.   We evaluate twenty-six models from six vendors, of which twenty-three are ranked. On the held-out partition, a 4B Qwen 3.5 student trained through parameter-effici
    
[^37]: SalamandraTA 参加 WMT 2026 术语共享任务：难例才是更好的老师

    SalamandraTA at WMT 2026 Terminology Shared Task: Hard Examples Are Better Teachers

    [https://arxiv.org/abs/2609.09999](https://arxiv.org/abs/2609.09999)

    该论文提出在术语感知翻译的微调中只保留模型自身译文与术语表相矛盾的“难例”，仅此筛选即可在固定数据量下将术语准确率从 78.7% 提升至 89.9%，并据此构建了 SalamandraTA-7b-instruct v3.0，作为 BSC 参加 WMT26 术语共享任务赛道 1 的提交系统。

    

    arXiv:2609.09999v1 公告类型：新论文 摘要：术语感知翻译的要求不止于正确的译文：输出必须使用术语表所规定的确切术语。标准做法——在带有术语表标注的翻译对上进行微调——隐藏着一种低效：对大多数样例而言，术语表规定的恰恰是模型本来就会产出的内容，因此这些样例无法教会模型如何遵循术语表。为此，我们只保留模型自身译文与术语表相矛盾的样例。在一项固定数据量的对照研究中，仅凭这种筛选方法就将术语准确率从 78.7% 提升至 89.9%。这些筛选后的数据通过在开源模型上运行双向合成流水线构建而成，是我们公开发布的 SalamandraTA-7b-instruct v3.0 指令微调混合数据的组成部分；该模型完全按照发布状态使用，并嵌入文档级推理流水线中，构成了 BSC 提交给 WMT26 术语共享任务赛道 1 的系统。在 WMT26 官方评测中，我们的系统取得了（原文在此处截断）

    arXiv:2609.09999v1 Announce Type: new  Abstract: Terminology-aware translation asks for more than a correct translation: the output must use the exact terms a glossary prescribes. The standard recipe, fine-tuning on glossary-annotated translation pairs, hides an inefficiency: for most examples the glossary prescribes exactly what the model would have produced anyway, so they teach nothing about following a glossary. We therefore keep only the examples where the model's own translation contradicts the glossary. In a controlled study at fixed data volume, this selection alone raises term accuracy from 78.7% to 89.9%. The filtered data, built by a two-way synthetic pipeline on open models, is part of the instruction-tuning mixture of our public release SalamandraTA-7b-instruct v3.0, which, used exactly as released and wrapped in a document-level inference pipeline, forms the BSC submission to the WMT26 Terminology Shared Task Track 1. At the official WMT26 evaluation, our system achieves 
    
[^38]: 稳定的答案，未完成的推理：为什么自我共识不是一种安全的提前退出信号

    Stable Answers, Unfinished Reasoning: Why Self-Consensus Is Not a Safe Early-Exit Signal

    [https://arxiv.org/abs/2609.09989](https://arxiv.org/abs/2609.09989)

    研究发现自我共识不是安全的推理提前退出信号——答案一致只说明答案稳定而非推理已完成（共识-终止差距），3,520条共识规则均未通过预设的安全与省token验收门槛，而基于边界置信度的DEER方法则全部通过。

    

    降低推理模型推理成本的一种自然方法，是对单个部分推理轨迹反复探测其当前答案，并在各次探测结果一致时停止——即“自我共识”。我们追问：这样的规则是否既安全又节省token，以及能否一次性选定后重复使用。一项预注册的扫描实验对3,520条共识规则进行了测试，在来自两个模型、三个基准测试的冻结轨迹上重放，结果没有任何一条规则通过预先设定的三项验收门槛；表现最好的规则在保留集划分以及两个未见过的模型上复现了同样的结果——而作为边界置信度对照方法的DEER经过同一流水线扫描后通过了全部三项门槛。原因在于信号本身：探测结果一致仅表明当前答案在固定探测程序下保持稳定，并不表明推理已经终止——这被称为“共识-终止差距”。依据该信号停止会提交非终止性的答案。在某条仍能节省32% token的规则下，每九次停止中就有一次会在轨迹自身（尚未完成的答案）上触发。

    arXiv:2609.09989v1 Announce Type: new  Abstract: A natural way to cut reasoning-model inference cost is to repeatedly probe a single partial trajectory for its current answer and stop once probes agree -- self-consensus. We ask whether any such rule is both safe and token-saving, and whether one can be selected once and reused. A preregistered sweep of 3,520 consensus rules, replayed on frozen trajectories from two models and three benchmarks, clears none of three acceptance gates fixed in advance; the frontier reproduces on a held-out split and on two unseen models -- while a boundary-confidence control (DEER) swept through the same pipeline clears all three. The reason lies in the signal: agreement establishes that the current answer persists under a fixed probing procedure, not that the reasoning has terminated -- a consensus-termination gap. Stopping on it commits non-terminal answers. At a rule still saving 32% of the tokens, one stop in nine fires on an answer the trajectory itse
    
[^39]: VLX-VR：一个具有智能体感知能力的视频推理模型

    VLX-VR: An Agentic-Aware Video Reasoning Model

    [https://arxiv.org/abs/2609.09985](https://arxiv.org/abs/2609.09985)

    VLX-VR 提出了一个基于“思考—记忆—观测”循环的智能体感知视频推理模型，通过强化学习掌握证据获取、记忆使用与终止决策，在 MINERVA 基准上以 78.79% 的准确率取得最先进性能。

    

    真实世界的视频理解需要整合分布在视频中的视觉、音频、文本和时间证据。然而，许多现有流程使用固定的视频上下文和单次推理，当观测结果不完整、模糊或相互冲突时，这限制了自适应的证据获取能力。我们提出了 VLX-VR，一个智能体感知的视频推理模型，它在由“思考—记忆—观测”循环定义的视频推理框架内进行训练。在每一步中，VLX-VR 确定所需的证据，调用 read_memory 或 write_memory，纳入返回的观测结果，并决定是继续执行还是产生任务输出。我们使用包括视频和智能体轨迹在内的多模态数据，通过强化学习来训练 VLX-VR，使其学会证据获取、记忆使用和终止决策。在 MINERVA 基准上，VLX-VR 在我们所比较的模型中达到了最先进的性能，准确率为 78.79%。

    arXiv:2609.09985v1 Announce Type: new  Abstract: Real-world video understanding requires integrating visual, audio, textual, and temporal evidence distributed across a video. Yet many pipelines use a fixed video context and single-pass inference, limiting adaptive evidence acquisition when observations are incomplete, ambiguous, or conflicting. We present VLX-VR, an agentic-aware video reasoning model trained within a video reasoning framework defined by a Think--Memory--Observation loop. At each step, VLX-VR determines the needed evidence, invokes read_memory or write_memory, incorporates the returned Observation, and decides whether to continue or produce the task output. We train VLX-VR with multimodal data, including videos and agent trajectories, using reinforcement learning to learn evidence acquisition, memory use, and termination. On MINERVA, VLX-VR achieves state-of-the-art performance among the models included in our comparison, with 78.79% accuracy. Under the original three 
    
[^40]: 用于科学出版物记录中资助者名称消歧的多功能嵌入模型

    Multi-Functional Embedding Models for Funder Name Disambiguation in Scientific Publication Records

    [https://arxiv.org/abs/2609.09984](https://arxiv.org/abs/2609.09984)

    本文提出了一个基于多任务学习的多语言、多功能资助者名称消歧模型框架，通过整合ROR、WoS和OFR数据集构建训练数据，应用于生物多样性保护领域研究出版物的分析。

    

    理解研究资金的历史分配和分布有助于我们深入了解科学研究如何跨领域、机构及地区获得支持。然而，由于资助者名称常存在拼写变体、翻译、缩写以及粒度不一致等问题，大规模分析受到了资助者名称消歧方案不完善的阻碍。本文提出了一个用于开发多语言、多功能资助者名称消歧模型的框架，并将其应用于生物多样性保护研究出版物。为构建训练数据集，我们将提供研究机构唯一标识符的Research Organization Registry（ROR）与两个出版物数据集（Web of Science (WoS) 和 Crossref Open Funder Registry (OFR)）进行了整合。我们采用带有对比损失和多重负例排序损失的多任务学习方法进行模型训练。

    arXiv:2609.09984v1 Announce Type: new  Abstract: Understanding the historical allocation and distribution of research funding advances our knowledge of how scientific research is supported across fields, institutions, and regions. However, large-scale analyses are hindered by the lack of comprehensive funder name disambiguation solutions, as funder names often exhibit spelling variations, translations, abbreviations, and inconsistent levels of granularity. In this paper, we present a framework for developing multilingual, multi-functional funder name disambiguation models and demonstrate its application to research publications in biodiversity conservation. To construct a training dataset, we integrated the Research Organization Registry (ROR), which provides unique identifiers for research organizations, with two publication datasets: the Web of Science (WoS) and the Crossref Open Funder Registry (OFR). We used multi-task learning with Contrastive Loss and Multiple Negatives Ranking L
    
[^41]: 基于弱监督ByT5微调的压力感知句子级菲律宾语字素到音素转换研究

    Towards Stress-Aware Sentence-Level Filipino G2P With Weakly-Supervised ByT5 Fine-Tuning

    [https://arxiv.org/abs/2609.09974](https://arxiv.org/abs/2609.09974)

    本文提出在维基词典数据引导的LLM辅助标注流水线构建的句子级数据集上，对基于ByT5的多语言G2P预训练模型进行弱监督微调，实现了压力感知的句子级菲律宾语字素到音素转换。

    

    字素到音素转换（G2P）是指将字素序列转换为对应音素序列的任务。虽然由于菲律宾语具有浅层正字法（即拼读规则较为直接），其G2P转换相当简单，但加入重音等韵律特征后增加了一层复杂性，需要句子级别的上下文而非单个单词输入。然而，菲律宾语的句子级数据通常不包含音素转写，这给训练G2P模型带来了挑战。因此，我们研究了如何利用现有数据获取菲律宾语的句子级音素数据，将所得模型与多语言单词级G2P模型进行比较，并衡量它们预测菲律宾语重音标记位置的准确度。我们提出对在多语言单词级G2P数据上预训练的基于ByT5的模型进行微调，所使用的三个句子级G2P数据集由维基词典数据引导的LLM辅助流水线进行标注。

    arXiv:2609.09974v1 Announce Type: new  Abstract: Grapheme-to-phoneme conversion (G2P) refers to the task of converting a sequence of graphemes to a corresponding sequence of phonemes. While Filipino G2P is fairly straightforward due to its shallow orthography, the inclusion of prosodic features such as stress adds a layer of complexity that requires sentence-level context instead of single-word inputs. However, sentence-level data for Filipino typically do not include phoneme transcriptions, posing a challenge for training G2P models. As such, we investigate how to obtain sentence-level phoneme data for Filipino using available data and compare the resulting models with multilingual word-level G2P as well as measure how accurately they predict stress marker position for Filipino. We propose fine-tuning a ByT5-based model, pre-trained on multilingual word-level G2P data, on three sentence-level G2P datasets annotated with an LLM-assisted pipeline guided by data from Wiktionary. This app
    
[^42]: 5-Dialects-BN：揭示音译对孟加拉方言大语言模型的影响

    5-Dialects-BN: Unmasking the Impact of Transliteration on Bangla Dialectal LLMs

    [https://arxiv.org/abs/2609.09964](https://arxiv.org/abs/2609.09964)

    该论文提出了首个多标注孟加拉方言基准数据集5-Dialects-BN，包含6,000条人工标注条目并覆盖五种主要方言，通过将罗马化音译与方言文本、标准孟加拉语、英语及主观性标签对齐，揭示了音译对孟加拉方言大语言模型性能的影响。

    

    大型语言模型（LLM）在自然语言处理（NLP）任务中取得了显著进展，但其在低资源语言和方言多样化环境中的能力却急剧下降。孟加拉语作为世界第六大使用语言，正是这一差距的典型例证：现有资源绝大多数针对标准孟加拉语，导致其地区方言缺乏开发或评估方言感知系统所需的基准。我们通过5-Dialects-BN来填补这一空白，这是首个多标注的孟加拉方言基准数据集，将罗马化音译与方言文本、标准孟加拉语、英语以及主观性标签在五种地区方言变体上进行对齐。该数据集包含6,000条人工标注条目，涵盖五种主要方言：吉大港（1,900条）、诺阿卡利（1,500条）、锡尔赫特（1,200条）、巴里萨尔（700条）和朗布尔（700条），反映了自然的在线数据分布情况。每条条目都经过丰富标注……

    arXiv:2609.09964v1 Announce Type: new  Abstract: Large Language Models (LLMs) have achieved remarkable progress across natural language processing (NLP) tasks, yet their capabilities degrade sharply for low-resource languages and dialectally diverse settings. Bangla, the world's sixth most spoken language, exemplifies this gap: existing resources overwhelmingly target Standard Bangla, leaving its regional dialects without the benchmarks needed to develop or evaluate dialect-aware systems. We address this gap with 5-Dialects-BN, the first multi-annotation Bangla dialect benchmark to align Romanized transliteration with dialectal text, Standard Bangla, English, and subjectivity labels across five regional varieties. The dataset comprises 6,000 manually annotated entries spanning five major dialects: Chittagong, Barisal, Noakhali, Sylhet, and Rangpur (Chittagong 1,900; Noakhali 1,500; Sylhet 1,200; Barisal 700; Rangpur 700), reflecting natural online availability. Each entry is enriched w
    
[^43]: 通过添加少量SALT来改进跨语言词元表示

    Improving Cross-Lingual Token Representations by Adding a Pinch of SALT

    [https://arxiv.org/abs/2609.09953](https://arxiv.org/abs/2609.09953)

    SALT是一种轻量级后训练方法，通过向现有跨语言句子编码器注入跨度级监督信号来改进词元表示，在五个多语言词元级基准中的四个上取得最佳结果，同时还能提升句子级任务性能。

    

    跨语言句子编码器能够在数百种语言之间实现可扩展的迁移，为翻译挖掘和低资源环境下的零样本学习等应用提供支持。尽管这些编码器是为句子级对齐而训练的，但它们越来越多地被应用于幻觉检测和序列标注等词元级任务，这暴露了训练与实际使用之间的不匹配。我们提出了SALT，一种轻量级的后训练方法，通过向现有句子编码器注入跨度级（span-level）监督信号来改进词元表示。在五个多语言词元级基准测试中，SALT在其中四个上取得了最佳整体结果，优于其他微调策略和竞争性编码器。此外，它还提升了跨语言检索和分类任务中的句子级性能。这些结果表明，跨度级监督是同时改进词元表示和句子表示的有效信号。

    arXiv:2609.09953v1 Announce Type: new  Abstract: Cross-lingual sentence encoders enable scalable transfer across hundreds of languages, powering applications such as translation mining and zero-shot learning in low-resource settings. Although trained for sentence-level alignment, they are increasingly also applied to token-level tasks such as hallucination detection and sequence tagging, exposing a mismatch between training and usage. We propose SALT, a lightweight post-training method that improves token representations by injecting span-level supervision into existing sentence encoders. Across five multilingual token-level benchmarks, SALT achieves the best overall results on four of them, outperforming alternative fine-tuning strategies and competitive encoders. It also improves sentence-level performance on cross-lingual retrieval and classification tasks. These results demonstrate that span-level supervision is an effective signal for improving both token and sentence representati
    
[^44]: Vague2Detect：在基于知识的开放世界检测中处理模糊提示

    Vague2Detect: Handling Ambiguous Prompts in Knowledge-Based Open-World Detection

    [https://arxiv.org/abs/2609.09949](https://arxiv.org/abs/2609.09949)

    提出Vague2Detect混合流水线，通过微调Sentence-BERT从知识库检索候选、YOLO-World验证图像存在性、GPT-3.5-turbo动态扩展知识库，有效解决了传统检测器无法处理模糊提示的问题。

    

    现实世界中的检测器经常需要解释功能性或模糊的提示，然而像YOLO这样的传统模型仍局限于固定的类别列表。即使是像YOLO-World这样的开放词汇模型，也经常无法将模糊的语言与预期目标正确对齐。在我们先前的工作《基于常识引导的开放世界目标检测：使用大语言模型与视觉-语义匹配》的基础上，我们解决了YOLO-World在落地任务驱动查询方面的局限性。我们提出了Vague2Detect，这是一个混合流水线：微调后的Sentence-BERT从结构化的家庭知识库（KB）中检索候选对象，随后由YOLO-World验证这些对象在图像中的存在。对于知识库之外的提示，大语言模型（GPT-3.5-turbo）生成候选描述，动态扩展知识库以覆盖新概念。在使用自定义图像和Open Images V7子集构建的家庭场景基准测试中，仅使用YOLO-World只能达到32%的模糊提示成功率（VP...

    arXiv:2609.09949v1 Announce Type: cross  Abstract: Real-world detectors must often interpret functional or ambiguous prompts, yet conventional models such as YOLO remain restricted to fixed class lists. Even open-vocabulary models like YOLO-World frequently misalign vague language with the intended objects. Building on our prior work Commonsense-Guided Open-World Object Detection Using LLMs and Visual-Semantic Matching, we address YOLO-World's limitations in grounding task-driven queries. We propose Vague2Detect, a hybrid pipeline in which a fine-tuned Sentence-BERT retrieves candidates from a structured household Knowledge Base (KB), and YOLO-World verifies their presence in the image. For prompts outside the KB, a large language model (GPT-3.5-turbo) generates candidate descriptions, dynamically expanding the KB to cover novel concepts. On a benchmark of household scenes using custom images and an Open Images V7 subset, YOLO-World alone achieves only 32% Vague Prompt Success Rate (VP
    
[^45]: 对比投影：通过差分Logit透镜读取Transformer内部状态

    Contrastive Projection: Reading Transformer Internals by Differencing Logit Lenses

    [https://arxiv.org/abs/2609.09902](https://arxiv.org/abs/2609.09902)

    提出对比投影方法，通过差分两个相近提示词的隐藏状态并经反嵌入投影来抵消共享的通用成分，构建无需训练的追踪器，从而可靠地读取Transformer在每个位置、子层和注意力头上真正区分输入的内部信息流。

    

    在词元空间中读取Transformer的内部状态很容易做到，却难以令人信服：在中间层，对单个隐藏状态应用logit透镜的结果主要由模型对几乎任何输入都会预测的通用词元所主导。我们转而读取差异。将两个高度匹配的提示词的隐藏状态相减，并通过反嵌入矩阵投影，可以抵消共享成分、凸显出区分两者的因素——这一操作等价于通过logit透镜读取RepE/ActAdd引导向量。将其构建为一个无需训练的追踪器，可在每个位置、每个子层和每个注意力头上进行读取，并在设计的基线上取平均，它成功追踪出Phi-2中复合名词的MLP→注意力信息链路，并通过激活补丁在该模型中得到验证；在三种架构上，通过读取输出和探针（而非补丁）恢复了相同的区分结果；它能读取检索为真实实体与虚构实体所呈现的差异，并将隐喻解读为一组领域到领域的映射。

    arXiv:2609.09902v1 Announce Type: new  Abstract: Reading a transformer's internal states in token space is easy to do and hard to trust: a logit lens on a single hidden state is dominated, at intermediate layers, by the generic tokens the model would predict for almost any input. We read the difference instead. Subtracting two closely matched prompts' hidden states and projecting through the unembedding cancels the shared component and surfaces what separates them, an operation equivalent to reading a RepE/ActAdd steering vector through a logit lens. Built into a training-free tracer that reads at every position, sub-layer, and head and averages over designed baselines, it traces a compound- noun MLP->attention chain in Phi-2, confirmed there by activation patching, with the same distinction recovered across three architectures by readout and probe rather than by patching; it reads what retrieval surfaces for real versus fictional entities, and reads metaphor as a set of domain-to-doma
    
[^46]: 语言模型中的深层与浅层偏见

    Deep and shallow biases in language models

    [https://arxiv.org/abs/2609.09901](https://arxiv.org/abs/2609.09901)

    该论文提出“偏见深度分数”这一新指标，将语言模型偏见区分为源自预训练且难以消除的深层偏见和依赖提示措辞的浅层偏见，并发现仅有约四分之一的集中偏好属于深层偏见。

    

    大型语言模型即使存在许多合理的替代选项时，也常常反复选择同一个答案。先前的工作将这种集中性视为偏见，但并未区分模型稳定的偏好与依赖于特定提示措辞的响应。我们引入了一个偏见深度分数，该分数既衡量模型在直接提示下对首选答案的偏好强度，也衡量该答案在场景重构后是否仍然存在。在4,442个观点提示和四个大型语言模型的实验中，只有大约四分之一的集中偏好在场景重构后得以保留。我们将这些持续存在的案例称为“深层偏见”，而将其余依赖提示的案例称为“浅层偏见”。我们的结果表明，深层偏见更多源自预训练，并在监督微调（SFT）中被保留。无论是在持续微调还是基于提示的多样性去偏方法下，深层偏见始终比浅层偏见更难消除。因此，偏见深度能够区分稳定的……

    arXiv:2609.09901v1 Announce Type: new  Abstract: Large language models often repeatedly select the same answer even when many alternatives are plausible. Prior work treats this concentration as bias, but it does not distinguish stable model preferences from responses that depend on a particular prompt wording. We introduce a bias depth score that measures both how strongly a model prefers its top answer under direct prompting and whether that answer survives scenario reframing. Across 4,442 opinion prompts and four large language models, only about a quarter of the concentrated preferences survive reframing. We call these persistent cases Deep biases, and the remaining prompt-dependent cases Shallow biases. Our results show that Deep biases are more often inherited from pretraining and preserved through SFT. Under both continued fine-tuning and prompt-based debiasing for diversity, Deep biases are consistently harder to remove than Shallow biases. Bias depth therefore separates stable 
    
[^47]: 自我陌生：语言模型关于自身的描述是泛化的

    Strangers to Themselves: What Language Models Say About Themselves Is Generic

    [https://arxiv.org/abs/2609.09899](https://arxiv.org/abs/2609.09899)

    语言模型缺乏真正的自我认知——它们对自身行为的描述是泛化性的，其预测效果与关于“AI智能体总体”的描述或其他模型对它的预测相比并无优势，即模型所说的关于自己的内容并非真正关于其本身。

    

    语言模型能够流利地描述它们会如何表现：是否会在反对压力下屈服、滥用工具或在压力下撒谎。但这些描述真的反映了说话模型本身吗？我们将自我知识转化为一个预测测试。在九项行为评估中，我们测量模型在不同条件下的实际行为表现，要求它预测这些行为的发生率，并将其预测与去除“自我”因素的对照组进行比较。我们发现：(i) 直接自我报告的预测能力很弱（r = +0.04），即使向模型展示具体的测试项目，预测准确度也仅提升至 +0.24。关键的是，同样基于项目信息、但针对“高能力AI智能体总体”的提问预测效果一样好（+0.28），而其他模型关于自身的回答对目标模型的预测效果至少与模型对自己的预测一样好。(ii) 前沿规模并未明显改变这一模式：预测能力的提升并非自我特异性的，而是与一个更好的AI助手行为理论相符……

    arXiv:2609.09899v1 Announce Type: cross  Abstract: Language models can fluently describe how they would behave: whether they would cave to pushback, misuse a tool, or lie under pressure. Is that description actually about the model speaking? We turn self-knowledge into a prediction test. Across nine behavioral evaluations, we measure how a model behaves under different conditions, ask it to predict those rates, and compare its predictions with controls that remove the self from the question. We find that: (i) Direct self-report is weak (r = +0.04), and even showing the model the exact items only raises prediction to +0.24. Crucially, the same item-informed question about "capable AI agents in general" does just as well (+0.28), while other models' answers about themselves predict the target model at least as well as its own. (ii) Frontier scale does not detectably change this pattern: any gains in prediction are not self-specific, and are consistent with a better theory of how AI assis
    
[^48]: 面向咨询服务的韩语语音识别细粒度纠错方法

    Leveraging Fine-grained Error Correction in Korean Speech Recognition for Consultation Services

    [https://arxiv.org/abs/2609.09889](https://arxiv.org/abs/2609.09889)

    该论文提出了首个面向对话级ASR纠错的大规模韩语基准数据集DasanCallDial，通过细粒度文本纠错方法解决呼叫中心场景中因隐私限制无法访问音频时的语音识别纠错难题。

    

    自动语音识别（ASR）技术是客户服务自动化和大规模转录的基础。然而，即使是先进的ASR模型，在呼叫中心对话等复杂的现实环境中也不可避免地会出现错误。当隐私限制使得无法访问音频时，纠错工作必须依赖基于文本的后编辑。现有的纯文本方法在低资源语言中面临重大挑战，主要原因是标注语料的严重匮乏以及缺乏针对性的纠错方法。就韩语而言，这一资源差距尤为突出，因为现有资源主要是为ASR训练而非基于文本的纠错而设计的。为解决这一问题，我们提出了DasanCallDial，这是首个专门为对话级ASR纠错而精心构建的大规模韩语基准数据集。该数据集源自真实的呼叫中心对话，包含1,974段对话和115,460条话语（摘要内容在此处不完整）。

    arXiv:2609.09889v1 Announce Type: new  Abstract: Automatic Speech Recognition (ASR) technology is fundamental to customer service automation and large-scale transcription. However, even advanced ASR models exhibit inevitable errors in complex real-world environments such as call center conversations. When privacy restrictions preclude audio access, error correction must rely on text-based post-editing. Existing text-only approaches face significant challenges in low-resource languages, mainly due to a critical scarcity of annotated corpora and tailored correction methodologies. For Korean, this resource gap is particularly pronounced, as existing resources are predominantly designed for ASR training rather than text-based error correction. To address this, we introduce DasanCallDial, the first large-scale Korean benchmark dataset specifically curated for dialogue-level ASR error correction. Derived from genuine call center interactions, it comprises 1,974 dialogues with 115,460 utteran
    
[^49]: 被告陈述何时重要？对LLM模拟陪审员中偏见与说服力的研究

    When Does Defendant Statement Matter? A Study of Bias and Persuasion in LLM-Simulated Jurors

    [https://arxiv.org/abs/2609.09887](https://arxiv.org/abs/2609.09887)

    该论文提出了JuryBench基准，通过分析20个前沿LLM的43.2万个陪审员决策，系统研究了被告法庭陈述如何通过说服力、意识形态偏见和背景亲和力影响LLM模拟陪审员的判决严重程度。

    

    LLM已被用于模拟专业场景中的人类决策，但其在普通法陪审团审判中的行为尚未被探索。我们研究了被告的法庭陈述何时以及如何影响LLM模拟的陪审员，重点关注说服力、意识形态偏见和基于背景的亲和力。为支持这一分析，我们引入了JuryBench，这是一个包含美国刑法中有争议刑事案件的基准测试。在每个案件中，被告可以提出各种合理的理由来支持无罪判决或减轻责任。我们固定基础案件，并设计不同背景的被告，他们给出具有不同程度情感诉求或反驳的法庭陈述，同时模拟具有不同意识形态倾向的陪审员。我们研究了20个前沿LLM，共产生43.2万个决策和理由，并量化了判决严重程度的变化。我们的研究结果表明，LLM陪审团模拟反映了人类陪审团的许多特征……（摘要内容不完整）

    arXiv:2609.09887v1 Announce Type: new  Abstract: LLMs have been used to simulate human decision-making in professional settings, yet their behaviors in common-law jury trials remain unexplored. We study when and how a defendant's courtroom statement affects LLM-simulated jurors, focusing on persuasion, ideological bias, and background-based affinity. To support the analysis, we introduce JuryBench, a benchmark containing controversial criminal cases in U.S. criminal law. In each case, a defendant can claim various plausible justifications to support acquittal or reduced liability. We fix the base case and design defendants of different backgrounds, who give courtroom statements with varying emotional appeal or rebuttal. Jurors with diverse ideological profiles across the spectrum are simulated. We examine 20 frontier LLMs, resulting in a total of 432K decisions and rationales, and quantify changes in verdict severity. Our findings show that LLM-jury simulation echoes many human-jury fi
    
[^50]: S³-Bench：评估语音交互模型作为科学语音助手的表现

    $S^3$-Bench: Evaluating Speech Interaction Models as Scientific Voice Assistants

    [https://arxiv.org/abs/2609.09852](https://arxiv.org/abs/2609.09852)

    本文提出S³-Bench，一个覆盖10个学科的系统性评估框架，通过将对话轮次分解为语音识别、感知、知识推理和发音等阶段，来评估语音交互模型作为科学语音助手的能力。

    

    多模态大语言模型（MLLM）的进步从根本上重塑了人机交互的范式，尤其是能够实现无缝对话的语音交互模型。尽管这些模型作为通用语音助手表现出色，但它们在专业领域的性能仍未得到充分探索，特别是在科学领域。科学交互带来了巨大的挑战，涉及罕见的专业术语、缩略语的口语规范，以及符号特殊表达式的自然口头表达。在本文中，我们提出了S³-Bench，这是一个覆盖10个主要学科的系统性评估框架，包含用于语音问答的知识集，以及用于与模拟用户代理进行多轮渐进式交互的对话集。通过将完整的原子轮次分解为语音识别、感知、结合推理的知识利用和回答发音等阶段，……

    arXiv:2609.09852v1 Announce Type: new  Abstract: The advance of multimodal large language models (MLLMs) has fundamentally reshaped the paradigm of human-computer interaction, especially speech interaction models capable of seamless conversations. Despite remarkable performance as general voice assistants, their performance in specialized domains remains underexplored, particularly in scientific areas. Scientific interactions introduce formidable challenges, involving rare technical terminology, spoken norms of abbreviations, and the natural verbalization of symbolic special expressions. In this paper, we introduce S$^3$-Bench, a systematic evaluation framework covering 10 major disciplines, consisting of a Knowledge set for speech question-answering and a Dialogue set for multi-turn progressive interactions with simulated user agents. By decomposing a complete atomic turn into stages of speech recognition, perception, knowledge utilization with reasoning, and response pronunciation, w
    
[^51]: HyperTrace：基于假设的偏好追踪实现大语言模型在线个性化

    HyperTrace: Hypothesis-Based Preference Tracing for Online LLM Personalization

    [https://arxiv.org/abs/2609.09835](https://arxiv.org/abs/2609.09835)

    HyperTrace是一个免训练的在线大语言模型个性化框架，通过维护并借助SMC风格重加权动态更新短期意图与长期偏好的可解释自然语言假设，实现潜在偏好追踪，无需参数更新即可显著提升响应对齐、偏好预测和用户画像一致性。

    

    个性化语言模型旨在使响应适应个体用户，而用户的偏好往往是潜在的，并通过交互逐渐显现。现有的免训练方法依赖于存储的历史记录或检索的记忆，但它们往往难以将长期偏好与短期的特定主题需求相协调。为了解决这一问题，我们提出了HyperTrace，这是一个免训练框架，将在线个性化建模为潜在偏好追踪问题。HyperTrace维护可解释的自然语言假设，涵盖短期意图和长期偏好，并通过基于大语言模型的代理选择模型，采用SMC（序贯蒙特卡洛）风格的重加权过程来更新这些假设。通过跨轮次和跨会话地更新这些假设，HyperTrace无需参数更新即可实现个性化。在PRISM和PersonaMem-v2数据集上的实验表明，HyperTrace在响应对齐、偏好预测和用户画像一致性方面均优于强大的在线基线方法。

    arXiv:2609.09835v1 Announce Type: new  Abstract: Personalized language models aim to adapt responses to individual users, whose preferences are often latent and revealed gradually through interaction. Existing training-free methods rely on stored histories or retrieved memories, but they often struggle to reconcile long- term preferences with short-term topic-specific needs. To address this issue, we propose HyperTrace, a training-free framework that formulates online personalization as latent preference tracing. HyperTrace maintains interpretable natural-language hypotheses over short-term intent and long-term preferences, and updates them through an SMC-style reweight process using an LLM-based surrogate choice model. By updating these hypotheses across turns and sessions, HyperTrace enables personalization without parameter updates. Experiments on PRISM and PersonaMem-v2 show that HyperTrace improves response alignment, preference prediction, and profile consistency over strong onli
    
[^52]: UnitBoost：用合并算子而非模型来管理复合LLM系统

    UnitBoost: Managing Compound LLM Systems with a Merge Operator, Not a Model

    [https://arxiv.org/abs/2609.09815](https://arxiv.org/abs/2609.09815)

    UnitBoost用确定性的合并算子取代复合LLM系统中的生成式元代理，通过槽位-值提案、约束argmax和显式残差机制实现顺序无关、可溯源的系统协调，并在基准测试中超越了金标准标签选出的最佳单一候选。

    

    复合LLM系统通常通过添加一个更高层级的LLM来解决协调问题。由此产生的元代理读取各个工作者模型的输出、撰写最终答案、分配后续调用，并决定何时停止。这种方式具有很强的表达能力，但它同时也将三个控制决策集中在一个不透明、对顺序敏感的模型调用中。我们提出疑问：管理器真的必须是生成式的吗？UnitBoost用一个明确定义的元层算子取代了该模型：由任务给定的单元映射将工作者输出转换为槽位-值提案，受约束的argmax负责组装输出，而未被填充或缺乏支撑的槽位则成为下一轮的显式残差。该算子与顺序无关，能够记录单元溯源，并提供一个简单的保证：在没有耦合约束的情况下，在相同准入分数下进行的单元级最大化优于任何完整候选的选择。在三个保留基准测试中，它超越了用金标准标签选出的最佳单一候选。

    arXiv:2609.09815v1 Announce Type: cross  Abstract: Compound LLM systems often solve a coordination problem by adding a higher-level LLM. The resulting meta-agent reads workers' outputs, writes the final answer, allocates later calls, and decides when to stop. It is expressive, but it also concentrates three control decisions in an opaque, order-sensitive model call. We ask whether the manager needs to be generative at all. UnitBoost replaces that model with a defined meta-level operator: a task-given unit map turns worker outputs into slot-value proposals, a constrained argmax assembles the output, and the slots left unfilled or unsupported become an explicit residual for the next round. The operator is order-free, records unit provenance, and gives a simple guarantee: without coupling constraints, unit-wise maximization under the same admission score dominates selection of any complete candidate. On three held-out benchmarks, it exceeds the best single candidate chosen with gold label
    
[^53]: 前沿规模下安全对齐有多脆弱？针对320B MoE模型的单方向攻击

    How Fragile Is Safety Alignment at Frontier Scale? A Single-Direction Attack on a 320B MoE

    [https://arxiv.org/abs/2609.09793](https://arxiv.org/abs/2609.09793)

    方向消融攻击成功迁移至320B参数的MoE模型GLM-5.3-Flash，证明前沿规模下安全对齐依然脆弱，但拒绝方向在超连接残差和量化架构中的分布位置与稠密模型显著不同。

    

    方向消融通过将单一的“拒绝方向”从写入残差流的权重中投影出去，从而移除对齐语言模型的拒绝能力。它无需基于梯度的训练，也无需优化，仅需几百个对比提示，这使其成为针对开放权重对齐的经典白盒攻击方法。然而，该方法此前仅在参数量最高约70B的稠密模型上得到验证。我们研究该方法能否迁移到前沿混合专家模型——这类模型的残差流不再是单一张量，且权重以量化形式发布。我们将其应用于GLM-5.3-Flash（320B参数、288个路由专家、四路超连接残差、块FP8量化）。该攻击在这种架构下依然有效，但其作用的位置已不再是按照原始方法预期所能找到的地方。单独编辑注意力、稠密写入器和路由专家写入器分别只移除0.039、0.016和0.148的拒绝……

    arXiv:2609.09793v1 Announce Type: cross  Abstract: Directional ablation removes an aligned language model's ability to refuse by projecting a single "refusal direction" out of the weights that write the residual stream. It needs no gradient-based training and no optimization, only a few hundred contrastive prompts, which makes it the canonical white-box attack on open-weight alignment. However, it has been established only on dense models up to roughly 70B parameters. We study whether it survives the shift to frontier mixture-of-experts (MoE) models whose residual streams are no longer a single tensor and whose weights ship quantized. We apply it to GLM-5.3-Flash (320B parameters, 288 routed experts, a four-wide hyper-connection residual, block-FP8). The attack survives the architecture, but what it reaches is no longer where a reader of the original recipe would look for it. Editing the attention, dense and routed-expert writers on their own removes 0.039, 0.016 and 0.148 of refusal r
    
[^54]: MUCnoHARM@GermEval 2026共享任务：基于检索的上下文学习用于诽谤犯罪检测及其局限性

    MUCnoHARM@GermEval Shared Task 2026: Retrieval-based In-Context Learning for Defamatory Offences, and Where It Falls Short

    [https://arxiv.org/abs/2609.09791](https://arxiv.org/abs/2609.09791)

    该论文研究了基于检索的上下文学习方法在检测德国刑法诽谤犯罪中的应用，发现检索策略相比随机示例收益甚微、模型选择才是最关键因素，且模型仍会遗漏26-57%的犯罪相关帖子，更适合用于人工分流而非自主审核。

    

    随着仇恨言论在网上无处不在，自动检测变得至关重要，尤其是针对涉及犯罪相关的社交媒体帖子。我们研究了多种基于检索的上下文学习策略，用于检测德国刑法典第185-187条规定的诽谤犯罪（GermEval 2026子任务4的主题）。少样本提示优于零样本提示，但基于检索的方法相比随机示例仅带来边际收益，甚至落后于经过优化的静态示例集。提供具体的法律知识有所帮助，但模型选择比其他所有系统选择都更为重要。模型会过度预测犯罪相关性，同时仍会遗漏26-57%的犯罪相关帖子，因此更适合作为分流工具而非自主内容审核。

    arXiv:2609.09791v1 Announce Type: new  Abstract: With hate speech being ubiquitous online, automatic detection is crucial, in particular when it comes to criminally relevant social media posts. We study a variety of retrieval-based in-context learning (RetICL) strategies for detecting defamatory offences under {\S}{\S} 185-187 StGB (the subject of GermEval 2026 Subtask 4). Few-shot prompting beats zero-shot, but retrieval-based approaches offer only marginal gains over random demonstrations, and even fall behind an optimised static set of demonstrations. Providing concrete legal knowledge helps, yet model choice outweighs every other system choice. Models over-predict criminal relevance while still missing 26-57% of criminally relevant posts, suiting them for triage rather than autonomous moderation.
    
[^55]: LogiScope-VQA：面向工业场景物流危险识别的视觉语言模型基准测试

    LogiScope-VQA: Benchmarking Vision-Language Models for Logistics Hazard Identification in Industrial Scenarios

    [https://arxiv.org/abs/2609.09790](https://arxiv.org/abs/2609.09790)

    该论文构建了基于真实物流园区数据的多模态基准测试LogiScope-VQA，通过2,476张图像、2,918个视频和10,274个人工精心标注的VQA，围绕工业要素感知、仓储知识理解和潜在风险推理三大主题的39个子任务，系统评估主流大模型在物流危险识别中的实际能力。

    

    大规模多模态模型（LMMs）在工业仓储场景的大规模部署，特别要求模型具备人类专家级别的、面向危险的感知、理解和推理能力。然而，与商业条款紧密绑定的真实工业数据稀缺，严重阻碍了该领域的进一步发展。为弥合这一差距，我们构建了LogiScope-VQA，以研究主流LMMs在真实物流运营中的实际适用性。LogiScope-VQA包含主要来源于真实物流园区的2,476张图像和2,918个视频，以及由人工标注者精心策划并验证的10,274个视觉问答（VQA）。基于18个核心物体和20种风险类型，我们设计了39个子任务，涵盖三大主题：工业要素感知、仓储知识理解和潜在风险推理。此外，我们引入了动态思考预算配置，并（摘要在此处被截断）

    arXiv:2609.09790v1 Announce Type: cross  Abstract: Large Multimodal Models (LMMs) large-scale deployment in industrial warehouse settings specifically necessitates that models exhibit human-expert-level hazard-oriented perception, understanding, and reasoning capabilities. However, the scarcity of real industrial data, tightly coupled to commercial terms, significantly hampers further advancement. To bridge this gap, we curate LogiScope-VQA to investigate the practical applicability of mainstream LMMs in real-world logistics operations. LogiScope-VQA comprises 2,476 images and 2,918 videos primarily sourced from real-world logistics parks, along with 10,274 VQAs meticulously curated and validated by human annotators. Grounded in 18 core objects and 20 risk types, we devise 39 subtasks aligned with three principal themes: industrial element perception, warehouse knowledge understanding, and potential risk reasoning. Furthermore, we incorporate dynamic thinking-budget configurations and 
    
[^56]: ROAM：通过语义关系实现智能体原子记忆的鲁棒组织

    ROAM: Robust Organization of Atomic Memories for Agents through Semantic Relations

    [https://arxiv.org/abs/2609.09778](https://arxiv.org/abs/2609.09778)

    ROAM提出了一种关系引导的框架，通过将原子记忆对分类为独立、等价、包含或冲突四种语义关系来组织智能体记忆，既保持了原子化管理的精确性，又能通过融合机制在回答时生成更丰富的非原子视图。

    

    长期运行的语言模型智能体依赖跨交互的外部记忆。原子记忆尤其有用：其细粒度的语义边界能够实现精确检索以及观察内容之间的直接比较。然而，不断累积的原子记忆不可避免地会变得冗余、重叠或相互冲突。现有方法通常让LLM管理器直接添加、更新、删除或重写记忆，将语义解释、存储决策和内容生成耦合在一个容易出错的操作中。我们提出了ROAM，这是一个关系引导的框架，在记忆管理中利用原子性，同时允许在回答时使用更丰富的表示。ROAM将新输入与已存储的原子对分类为独立、等价、有方向的包含或冲突关系，然后将观察内容组织为活跃的Primary（主要）角色和辅助的Evidence（证据）角色。随后，融合机制将互补细节和时间变化整合为紧凑的、可能非原子的视图。

    arXiv:2609.09778v1 Announce Type: new  Abstract: Long-term language-model agents rely on external memory across interactions. Atomic memories are particularly useful: their fine-grained semantic boundaries enable precise retrieval and direct comparison between observations. Yet accumulating atoms inevitably become redundant, overlapping, or conflicting. Existing methods often ask an LLM manager to add, update, delete, or rewrite memories directly, coupling semantic interpretation, storage decisions, and content generation in one error-prone operation. We introduce ROAM, a relation-guided framework that uses atomicity for management while allowing richer answer-time representations. ROAM classifies incoming--stored atom pairs as independent, equivalent, directionally subsuming, or conflicting, then organizes observations into active Primary and supporting Evidence roles. Fusion subsequently combines complementary details and temporal changes into compact, potentially non-atomic views. O
    
[^57]: SymbolicLight V2：面向低能耗语言推理的混合神经形态架构与稀疏执行

    SymbolicLight V2: Hybrid Neuromorphic Architecture and Sparse Execution for Low-Energy Language Inference

    [https://arxiv.org/abs/2609.09772](https://arxiv.org/abs/2609.09772)

    该论文提出 SymbolicLight V2 混合神经形态语言架构，通过分级带符号事件、无 softmax 局部注意力和稀疏执行技术，在 FPGA 与 ARM CPU 上将解码吞吐量提升约 35%，并将每 token 能耗最高降低 27.7%，实现低能耗语言推理。

    

    SymbolicLight V2 将稀疏事件计算与连续状态处理相结合，构成一种混合神经形态语言架构。该架构在 V1 的脉冲门控双路径基础上进行扩展，在更深层的投影中引入了分级带符号事件以及无 softmax 的局部注意力机制。研究者在 Alveo U50C FPGA 上采用数字定点算术实现了这个 1.94 亿参数的模型，并在 ARM CPU 上采用稀疏整数执行。在三个相同检查点、175 MHz 的 FPGA 实现中，通过活跃行权重收集和有效状态 KV 加载，在 32 个 token 前缀和 128 个输出的设置下，解码吞吐量从 474.6 提升至 643.2 tokens/s。估计的总卡能耗从每生成 token 0.06087 焦耳降至 0.04407 焦耳，降低 27.6%。包含预填充在内的完整请求能耗在三种前缀长度下降低了 24.4%–27.7%。一项独立的空闲分离分析将总卡能耗的 82.8% 归因于加载空闲状态，这解释了缩短 token 延迟所带来的节能收益。与 reco……（原文摘要在此处截断）

    arXiv:2609.09772v1 Announce Type: new  Abstract: SymbolicLight V2 combines sparse event computation with continuous-state processing in a hybrid neuromorphic language architecture. Extending V1's spike-gated dual paths, it adds graded signed events at further projections and softmax-free local attention. We implement the 194M-parameter model on an Alveo U50C FPGA using digital fixed-point arithmetic and on an ARM CPU using sparse integer execution. Across three same-checkpoint FPGA implementations at 175 MHz, active-row weight gathering and valid-state KV loading raise decode throughput from 474.6 to 643.2 tokens/s for a 32-token prefix and 128 outputs. Estimated gross card energy falls from 0.06087 to 0.04407 J per generated token, a 27.6% reduction. Complete-request energy, including prefill, falls by 24.4-27.7% across three prefix lengths. An independent idle split attributes 82.8% of gross card energy to loaded idle, explaining the benefit of shorter token latency. Against the reco
    
[^58]: 微调感知KV缓存拼接的模型还是重新计算KV缓存？为什么不同时兼顾两者？

    Fine-Tuning a KV Cache Concatenation-Aware Model or Recomputing KV Caches? Why Not Both?

    [https://arxiv.org/abs/2609.09768](https://arxiv.org/abs/2609.09768)

    提出将KV缓存拼接感知的模型微调与选择性KV缓存重计算相结合的方法，在保持低首token延迟的同时显著提升RAG系统长上下文输入的回复准确性。

    

    在检索增强生成（RAG）系统中，大量检索到的文本块被拼接起来构成输入上下文，以便用户能够基于外部知识获得高质量的回复。因此，输入上下文的长度大幅增加，导致预填充工作负载增大，进而使首token生成时间（TTFT）变长。虽然先前重用预计算键值（KV）缓存的工作有效地降低了长上下文输入的TTFT，但当输入上下文变得非常长时，回复质量是否得到保持仍不清楚。在本文中，我们提出了一种组合方法：(i) 在考虑KV缓存拼接的情况下对模型进行微调，以及 选择性地重新计算一部分KV缓存。通过同时应用这两种技术，我们证明了该方法能够提升长上下文输入的准确性。在RULER基准上的实验表明，对于124k token的输入，我们的方法将RULER分数提高了

    arXiv:2609.09768v1 Announce Type: cross  Abstract: In Retrieval-Augmented Generation (RAG) systems, a large number of retrieved chunks are concatenated to form the input context so that users can receive high-quality responses based on external knowledge. As a result, the input context length increases substantially, leading to a larger prefill workload and, in turn, a longer time to first token (TTFT). While previous works that reuse precomputed key-value (KV) caches effectively reduce TTFT for long-context inputs, it remains unclear whether response quality is preserved when the input context becomes very long. In this paper, we propose a combined approach that (i) fine-tunes the model while taking KV cache concatenation into account and (ii) selectively recomputes a subset of the KV caches. By applying both techniques, we demonstrate improved accuracy for long-context inputs. Experiments on the RULER benchmark show that, for a 124k-token input, our method improves the RULER score by
    
[^59]: CARRE：用于可解释客户流失处方的反事实动作检索与推理评估

    CARRE: Counterfactual Action Retrieval and Reason Evaluation for Explainable Churn Prescription

    [https://arxiv.org/abs/2609.09766](https://arxiv.org/abs/2609.09766)

    CARRE是一个结合检索增强候选生成、成本感知反事实评分与大语言模型推理的三阶段框架，不仅识别高流失风险客户，还能推荐具体的挽留行动并给出可解释的推荐理由，在电信数据集上相比SHAP基线实现了近80%更高的风险降低。

    

    客户流失模型通常只能识别高风险客户，但无法明确指出应该考虑哪种可行的挽留行动，也无法解释为什么该行动是合适的。我们提出了CARRE（反事实动作检索与推理评估），这是一个三阶段框架，结合了检索增强的候选行动生成、成本感知的反事实评分以及大语言模型（LLM）推理。CARRE从预定义的挽留行动目录中检索候选行动，在显式特征变换下估计模型预测的流失风险变化，并为所选行动生成结构化的流失原因和基于客户画像的解释。在IBM电信客户流失数据集上，在313个高风险测试案例中，CARRE相比纯SHAP基线实现了79.8%更高的平均模型预测风险降低，相比成本控制的SHAP+Cost基线实现了80.4%更高的风险降低；其成本归一化效率比纯SHAP高出10.5%。

    arXiv:2609.09766v1 Announce Type: new  Abstract: Churn models typically identify high-risk customers but do not specify which feasible retention action should be considered or why that action is appropriate. We present CARRE (Counterfactual Action Retrieval and Reason Evaluation), a three-stage framework that combines retrieval-augmented candidate generation, cost-aware counterfactual scoring, and large language model (LLM) reasoning. CARRE retrieves a predefined catalog of retention actions, estimates model-predicted churn-risk changes under explicit feature transformations, and generates a structured churn reason and a profile-grounded explanation for the selected action. On the IBM Telco Customer Churn dataset, CARRE achieves 79.8% greater mean model-predicted risk reduction than the plain SHAP baseline and 80.4% greater reduction than the cost-controlled SHAP+Cost baseline across 313 high-risk test cases; its cost-normalized efficiency is 10.5% higher than that of plain SHAP. On a 
    
[^60]: SocialRL：通过多轮强化学习和奖励设计提升大语言模型的社交智能

    SocialRL: Refining LLMs' Social Intelligence through Multi-turn Reinforcement Learning and Reward Design

    [https://arxiv.org/abs/2609.09764](https://arxiv.org/abs/2609.09764)

    SocialRL是一个多轮强化学习框架，通过PPO将延迟结果奖励传播回每一轮对话，并设计六个捕捉目标-关系权衡的过程奖励维度，从而提升大语言模型在多轮社交交互中的社交智能。

    

    社交智能使智能体能够理解社交情境、推断意图，并在持续对话中不断适应。随着语言模型逐渐成为自主协作伙伴，社交智能对于构建有效且可信的人机交互至关重要。现有的强化学习方法只优化单轮话语和稀疏的结果奖励，产生了短视的策略，难以在多轮交互中处理目标与关系之间的张力。我们提出了SocialRL，一个应对这两项挑战的多轮强化学习框架。首先，我们应用基于PPO的多轮强化学习，将延迟的结果奖励传播回每一轮对话，实现长时程规划。其次，我们设计了六个捕捉目标-关系权衡的过程奖励维度，包括目标推进、关系调适、上下文连贯性等。奖励模型为每个维度动态生成细粒度的评分标准。

    arXiv:2609.09764v1 Announce Type: new  Abstract: Social intelligence enables agents to read social context, infer intent, and adapt over sustained dialogue. As language models become autonomous collaborators, it is central to building effective and trustworthy human-AI interaction. Existing reinforcement learning methods optimize single-turn utterances and sparse outcome rewards, producing short-sighted policies that struggle to manage goal-relationship tensions across multi-turn interactions. We propose SocialRL, a multi-turn reinforcement learning framework addressing both challenges. First, we apply multi-turn reinforcement learning using PPO that propagates delayed outcome rewards back to each turn, enabling long-horizon planning. Second, we design six process reward dimensions capturing the goal-relationship trade-off, including goal advancement, relational attunement, contextual coherence, etc. A reward model dynamically generates fine-grained scoring criteria for each dimension,
    
[^61]: 人工智能能否通过早期网络欺凌检测支持医疗保健与心理健康？情感感知AI对主动式在线安全的影响

    Can Artificial Intelligence Support Healthcare and Mental Health Through Early Cyberbullying Detection ? The Impact of Emotion-Aware AI on Proactive Online Safety

    [https://arxiv.org/abs/2609.09735](https://arxiv.org/abs/2609.09735)

    本文提出CareGuard早期预警框架，通过融合零样本语义标注、微调Transformer模型以及情感感知过滤机制，实现对网络欺凌内容的高效早期检测，从而支持医疗保健驱动的心理健康保护与主动式在线安全。

    

    医疗保健系统、心理健康和公共福祉正日益受到网络欺凌和有害在线互动的影响。本文提出了CareGuard，一个早期预警框架，旨在通过使用先进的自然语言处理技术检测网络欺凌相关内容，支持以医疗保健为导向的心理健康保护和主动式在线安全。CareGuard将零样本语义标注与微调的基于Transformer的模型（包括BERT、DistilBERT和RoBERTa）相结合，以实现对敏感网络欺凌类别的鲁棒且具备上下文感知能力的分类。为了提高效率并减少面向医疗保健监控环境中不必要的计算，该框架引入了情感感知过滤机制以及基于余弦相似度的语义筛选，使系统能够专注于语义相关且情感显著的内容。在基准数据集上的实验结果表明（摘要在此处截断）……

    arXiv:2609.09735v1 Announce Type: cross  Abstract: Healthcare systems, mental health, and public well-being are increasingly affected by cyberbullying and harmful online interactions. This paper presents CareGuard, an early-warning framework designed to support healthcare-driven mental health protection and proactive online safety through the detection of cyberbullying-related content using advanced natural language processing techniques. CareGuard integrates zero-shot semantic labeling with fine-tuned transformer-based models, including BERT, DistilBERT, and RoBERTa, to enable robust and context-aware classification across sensitive cyberbullying categories. To improve efficiency and reduce unnecessary computation in healthcare-oriented monitoring settings, the framework incorporates an emotion-aware filtering mechanism alongside cosine similarity-based semantic screening, allowing the system to focus on semantically relevant and emotionally salient content. Experimental results on be
    
[^62]: StreamAlign：流式文本对齐语音分词框架

    StreamAlign: Streaming Text-Aligned Speech Tokenization

    [https://arxiv.org/abs/2609.09719](https://arxiv.org/abs/2609.09719)

    StreamAlign是一个支持流式处理的文本对齐语音分词框架，通过结合字符级RNN-Transducer对齐、词级ASR指导和主动式词边界分类器，实现了实时语音-文本联合建模，同时缓解了ASR与LLM之间的词表不匹配问题并降低了分词延迟。

    

    文本对齐的语音分词方法的出现是为了更好地将语音token与大型语言模型（LLM）的token空间对齐，从而更有效地利用预训练的LLM。然而，这些方法依赖于离线自动语音识别（ASR），导致两个关键限制：（i）需要在分词之前获得完整的语句，无法实现实时流式处理；（ii）ASR与LLM之间的词表不匹配，使得声学粒度从子词级别降低到词级别。我们提出了StreamAlign，一个支持流式分词的文本对齐语音分词框架，用于实时语音-文本联合建模。StreamAlign通过结合字符级RNN-Transducer对齐与词级ASR指导来执行在线语音-文本对齐，在保持识别准确率的同时缓解了ASR与LLM之间的词表不匹配问题。一个主动式词边界分类器能够在数据块边界预测词语的完成，从而降低分词延迟。

    arXiv:2609.09719v1 Announce Type: new  Abstract: Text-aligned speech tokenization methods have emerged to better align speech tokens with LLM token spaces, enabling more effective utilization of pretrained LLMs. However, they rely on offline automatic speech recognition (ASR), leading to two key limitations: (i) the need for complete utterances before tokenization, precluding real-time streaming, and (ii) vocabulary mismatch between ASR and LLMs, which reduces acoustic granularity from the subword to the word level. We introduce StreamAlign, a text-aligned speech tokenization framework that enables streaming tokenization for real-time speech-text joint modeling. StreamAlign performs online speech-text alignment by combining character-level RNN-Transducer alignment with word-level ASR guidance, mitigating ASR-LLM vocabulary mismatch while preserving recognition accuracy. A proactive word boundary classifier anticipates word completion at chunk boundaries, reducing tokenization latency f
    
[^63]: 基于并行解码的电商属性提取规模化方法

    Scaling E-Commerce Attribute Extraction with Parallel Decoding

    [https://arxiv.org/abs/2609.09716](https://arxiv.org/abs/2609.09716)

    提出了一种两阶段LLM流水线，先为每个产品类别发现紧凑且按重要性排序的购买判别属性模式，再利用微调的Qwen3-4B结合超并行解码技术提取属性值，在达到与基础LLM相当的85%提取准确率的同时，将推理成本降低了92%，实现了生产级规模化应用。

    

    客户依靠特定的产品属性来比较商品并做出购买决策，但电商目录往往杂乱且非结构化，这使得难以识别哪些属性最为重要并进行规模化提取。标准的属性值提取（AVE）系统对所有属性一视同仁，产生了庞大且不一致的属性集合，无法反映消费者用来区分产品的关键因素。我们引入了一个两阶段的大语言模型（LLM）流水线：首先为每个产品类别发现一个紧凑的、按重要性排序的购买判别属性模式，然后使用经过微调的紧凑型大语言模型（Qwen3-4B）结合超并行解码（HPD）技术从目录文本中提取这些属性的值。该流水线达到了85%的提取准确率，与它所蒸馏自的基础大语言模型相当，同时相比基础大语言模型将推理成本降低了92%，实现了产品发现和目录丰富化的生产级规模应用。

    arXiv:2609.09716v1 Announce Type: new  Abstract: Customers rely on specific product attributes to compare products and make purchasing decisions, but e-commerce catalogs are messy and unstructured, making it difficult to identify which attributes matter most and extract them at scale. Standard Attribute Value Extraction (AVE) systems treat all attributes equally, producing large, inconsistent attribute sets that do not reflect the factors consumers use to differentiate products. We introduce a two-stage LLM pipeline that first discovers a compact, ranked schema of purchase-discriminative attributes for each product category, then extracts their values from catalog text using a fine-tuned compact LLM (Qwen3-4B) with Hyper-Parallel Decoding (HPD). This pipeline achieves 85% extraction accuracy, on par with the foundational LLM it was distilled from, while reducing inference costs by 92% over foundational LLMs, enabling production-scale use for product discovery and catalog enrichment. Th
    
[^64]: 当审计员捏造事实：大语言模型检测植入文档污染中的批次规模退化与自信幻觉

    When Auditors Fabricate: Batch-Size Degradation and Confident Hallucination in LLM Detection of Planted Document Contamination

    [https://arxiv.org/abs/2609.09696](https://arxiv.org/abs/2609.09696)

    大语言模型在单文档和小批量污染检测中表现尚可（50%-60%），但在大批量处理时检测率骤降至2.8%，且其失败方式不是承认无法处理，而是自信地捏造包括虚假污染项在内的检测结果。

    

    大语言模型越来越多地被提议作为文档质量的自动化审计工具，但它们作为植入错误检测器的可靠性却缺乏充分表征。我们构建了一个包含150篇学术论文的受污染语料库，涵盖供应链管理和医学研究领域，注入了450个已知污染项，分为三种类型：排版损坏、语义反转和荒谬的脱离语境插入。随后，我们在三种规模递增的提示机制下（单文档、小批量和大批量），评估了Google Gemini 3.0 Pro在60份文档中恢复包含180个污染项的答案密钥子集的能力。检测在小规模下保持有效，随后急剧崩溃：单文档恢复率为50%，小批量为60%，大批量仅为2.8%。大规模下的失败模式并非放弃检测，而是捏造结果。模型没有报告处理不完整，而是产生了自信的发现，包括自行编造的污染项……

    arXiv:2609.09696v1 Announce Type: new  Abstract: Large language models are increasingly proposed as automated auditors of document quality, yet their reliability as detectors of planted errors is poorly characterised. We construct a contaminated corpus of 150 academic papers spanning supply chain management and medical research, injecting 450 known contaminants of three types: typographical corruption, semantic reversal, and absurd out-of-context insertion. We then evaluate Google Gemini 3.0 Pro's ability to recover a 180-contaminant answer-key subset across 60 documents under three prompting regimes of increasing scale: single document, small batch, and large batch. Detection holds at small scale and then collapses: 50% recovery on single documents, 60% on small batches, and 2.8% on large batches. The failure mode at scale is not abstention but fabrication. Rather than reporting incomplete processing, the model produced confident findings including invented contaminants of its own, ab
    
[^65]: 循环 GPT-BERT：在小规模语言建模中以计算换取参数

    Looped GPT-BERT: Trading Parameters for Computation in Small Language Modeling

    [https://arxiv.org/abs/2609.09691](https://arxiv.org/abs/2609.09691)

    该研究提出循环 GPT-BERT，通过深度参数共享让 4 个物理层循环遍历 12 次，在 BabyLM 2026 Strict-small 设定下以仅 1218 万参数在 BLiMP 和 GLUE 等语言学与下游任务指标上取得了与更大参数量的 GPT-2 和 GPT-BERT 基线相当的性能。

    

    当训练数据有限时，增加参数量并非提升语言模型性能的唯一途径。一组较小的参数经过反复使用，同样能够取得可比的性能。我们在 BabyLM 2026 Strict-small 设定下研究了循环 GPT-BERT（Looped GPT-BERT），将 GPT-BERT 的掩码下一词预测与因果语言建模目标同逐层深度参数共享相结合。我们在一个预处理后的 748 万词英语语料库上进行训练，并比较了目标函数比例、非循环与循环架构以及循环次数。我们最终的 4×12 模型使用四个物理层进行十二次循环遍历，共包含 1218 万参数。BabyLM 2026 排行榜报告其总体平均分为 35.42，NLP 平均分为 48.48。与公开的 BabyLM 10M Strict-small GPT-2 和 GPT-BERT 基线相比，该模型在包括 BLiMP 和 GLUE 在内的多项语言学与下游任务指标上，以更少的参数取得了可比的性能。

    arXiv:2609.09691v1 Announce Type: new  Abstract: When training data are limited, increasing parameter count is not the only way to improve language-model performance. A small parameter set, when repeatedly applied, can also deliver comparable performance. We study Looped GPT-BERT in the BabyLM 2026 Strict-small setting, combining GPT-BERT's masked next-token and causal language-modeling objectives with depth-wise parameter sharing. We train on a preprocessed 7.48M-word English corpus and compare objective ratios, non-looped and looped architectures, and loop counts. Our final $4\times12$ model uses four physical layers for twelve recurrent traversals and contains 12.18M parameters. The BabyLM 2026 leaderboard reports an Overall Average of 35.42 and an NLP Average of 48.48. Compared with public BabyLM 10M Strict-small GPT-2 and GPT-BERT baselines, it achieves comparable performance on selected linguistic and downstream metrics, including BLiMP and GLUE, with fewer parameters. The loop a
    
[^66]: 哪些医学问题值得生成推理依据？面向鲁棒问答的扰动敏感选择方法

    Which Medical Questions Deserve Rationales? Perturbation-Sensitive Selection for Robust QA

    [https://arxiv.org/abs/2609.09684](https://arxiv.org/abs/2609.09684)

    该论文提出RMS-RSP方法，通过仅在推理依据token处扰动隐藏状态并测量答案与干扰项之间裕度的变化，在固定token预算下智能筛选哪些已标注医学问题最值得投入推理依据监督，从而提升医学问答的鲁棒性。

    

    医学问答数据集通常包含答案标签，而高质量的推理依据仍然稀缺、含有噪声或验证成本高昂。这改变了数据获取的核心问题：我们不再追问哪些问题应该被标注，而是探究在固定token预算下，哪些已标注的问题应该获得推理依据监督。我们研究了该问题的一个离线版本，其中候选推理依据对选择器可见，但除非被选中，否则不会用于下游训练。我们提出了均方根鲁棒性样本优先级方法（RMS-RSP），该方法仅在推理依据token处扰动隐藏状态，并测量由此导致的金标准答案与最佳干扰项之间裕度的变化。在五个医学问答数据集、MedGemma-4B-IT模型、三个训练种子、十个有预算的非RSP选择器以及一个无预算的全监督参考基线下，RMS-RSP给出了一个经过审慎限定的结果：其固定预算下的平均准确率为60.61%。

    arXiv:2609.09684v1 Announce Type: new  Abstract: Medical question-answering datasets often contain answer labels, whereas high-quality rationales remain scarce, noisy, or costly to validate. This changes the acquisition question: rather than asking which questions should be labeled, we ask which already-labeled questions should receive rationale supervision under a fixed token budget. We study an offline version of this problem in which candidate rationales are visible to the selector but withheld from downstream training unless selected. We propose root-mean-square Robustness-based Sample Prioritization (RMS-RSP), which perturbs hidden states only at rationale tokens and measures the resulting shift in the gold-versus-best-distractor margin. Across five medical QA datasets, MedGemma-4B-IT, three training seeds, ten budgeted non-RSP selectors, and an unbudgeted full-supervision reference, RMS-RSP provides a deliberately qualified result. Its locked-budget accuracy is 60.61% on average 
    
[^67]: X2-NativeCursor：用于增量文本流式编解码器TTS的原生令牌文本进度跟踪

    X2-NativeCursor: Native-Token Text Progress Tracking for Incremental-Text Streaming Codec TTS

    [https://arxiv.org/abs/2609.09677](https://arxiv.org/abs/2609.09677)

    提出X2-NativeCursor，一种无需修改TTS生成器、直接在波形解码前从原生语音令牌跟踪文本进度的轻量级观察器，以极低前瞻延迟实现了高精度的在线文本进度跟踪。

    

    增量文本流式文本转语音（TTS）需要在线文本进度跟踪，以实现同步高亮、中断处理和对话历史更新。由于输入文本在语音播出之前就已到达，仅凭文本到达无法指示语音进度。现有的基于波形的对齐方法需要完整的音频，或者在流式传输过程中增加声学处理。我们提出了X2-NativeCursor，这是一种轻量级观察器，它无需改变TTS生成器，即可在波形解码之前从原生语音令牌跟踪进度。其规范化方案将朗读标签与其原始文本片段关联起来。文本编码器和原生令牌编码器将信息输入局部匹配器，用于估计当前标签位置。一个独立的输出规则将可修正的位置估计转换为永不后退的光标。与自动参考相比，在80毫秒前瞻的条件下平均绝对误差为0.151个汉字，而基线方法在320毫秒前瞻条件下误差为1.253个字符……（摘要截断）

    arXiv:2609.09677v1 Announce Type: new  Abstract: Incremental-text streaming text-to-speech (TTS) needs online text progress tracking for synchronized highlighting, interruption handling, and dialogue-history updates. Input text arrives before it is spoken, so text arrival alone cannot indicate speech progress. Existing waveform-based alignment requires complete audio or adds acoustic processing during streaming. We propose X2-NativeCursor, a lightweight observer that tracks progress from native speech tokens before waveform decoding without changing the TTS generator. Its normalization plan links spoken labels to their original-text spans. Text and native-token encoders feed a local matcher that estimates the current label position. A separate output rule converts revisable position estimates into a cursor that never moves backward. Mean absolute error against an automatic reference is 0.151 Chinese characters with 80-ms lookahead, versus 1.253 characters with 320-ms lookahead for an o
    
[^68]: SEA-SpeechBench：一个面向东南亚语音理解的大规模多任务基准测试

    SEA-SpeechBench: A Large-Scale Multitask Benchmark for Speech Understanding Across Southeast Asia

    [https://arxiv.org/abs/2609.09672](https://arxiv.org/abs/2609.09672)

    该论文提出了首个面向东南亚语言的大规模多任务语音理解基准SEA-SpeechBench，覆盖11种语言、97,194个样本和597小时音频，涵盖语音处理、副语言分析和新颖的时间理解三大类共9项任务。

    

    音频和多模态大语言模型的快速发展开启了变革性的语音理解能力，然而现有的评估框架仍然以英语为中心，导致东南亚（SEA）语言的代表性严重不足。我们提出了SEA-SpeechBench，据我们所知，这是第一个大规模多任务基准，通过97,194个样本、99个评估集和597小时的精选音频数据，对11种东南亚语言的语音理解能力进行评估。我们的基准涵盖3个类别中的9项多样化任务：语音处理（自动语音识别、语音翻译、口语问答）、副语言分析（情感、性别、年龄、说话人识别）以及时间理解——这是一个全新的维度，包含带时间戳的内容查询以及在长达3分钟的扩展音频序列中的时间定位。我们使用东南亚本地语言和……（原文摘要在此截断）实现了多语言提示

    arXiv:2609.09672v1 Announce Type: new  Abstract: The rapid advancement of audio and multimodal large language models has unlocked transformative speech understanding capabilities, yet evaluation frameworks remain predominantly English-centric, leaving Southeast Asian (SEA) languages critically underrepresented. We introduce SEA-SpeechBench, to the best of our knowledge, the first large-scale multitask benchmark that evaluates speech understanding in 11 SEA languages through 97,194 samples across 99 evaluation sets and 597 hours of curated audio data. Our benchmark comprises 9 diverse tasks across 3 categories: speech processing (automatic speech recognition, speech translation, spoken question answering), paralinguistic analysis (emotion, gender, age, speaker recognition), and temporal understanding, a novel dimension featuring timestamped content queries and temporal localization within extended audio sequences up to 3 minutes. We implement multilingual prompting in both native SEA la
    
[^69]: PELM：基于推测解码与动态电压频率调节的高能效端侧大语言模型推理

    PELM: Power Efficient On-Device LLM Inference with Speculative Decoding and Dynamic Voltage Frequency Scaling

    [https://arxiv.org/abs/2609.09662](https://arxiv.org/abs/2609.09662)

    本文提出PELM，一种结合推测解码与动态电压频率调节（DVFS）的端侧大语言模型高能效推理框架，旨在解决移动和边缘平台计算资源受限及散热能力不足导致的降频问题。

    

    由于隐私保护增强、个性化以及降低延迟等诸多优势，将大语言模型（LLM）直接部署在边缘移动平台上正日益受到关注。然而，大语言模型具有繁重的计算需求，这对于资源受限的移动和边缘平台来说难以满足。除了计算资源有限之外，移动和边缘系统通常结构紧凑，缺乏物理散热机制（例如风扇）来散发高处理器使用率产生的热量，从而无法防止降频节流和处理能力的下降，而大语言模型很容易导致这些问题。为了缓解这些影响，先前的研究提出了各种功耗管理策略，例如动态电压频率调节（DVFS），用于减少移动平台上重计算任务的功耗和发热。最近，针对移动端大语言模型定制的DVFS方法也已被提出。然而，这些方法大多…

    arXiv:2609.09662v1 Announce Type: cross  Abstract: Deploying Large Language Models (LLMs) directly on mobile platforms at the edge is gaining traction due to a myriad of benefits, such as increased privacy, personalization, and reduced latency. However, LLMs have heavy computational requirements, which are difficult for resource-constrained mobile and edge platforms to fulfill. In addition to limited compute resources, mobile and edge systems often have a compact form factor and lack physical mechanisms to dissipate heat generated from high processor usage rates (e.g., fans) to prevent throttling and reduced processing power, which LLMs can easily cause. To mitigate these effects, prior works have proposed various power governing strategies, such as dynamic voltage and frequency scaling (DVFS), for reducing power and heat generation for heavy computational tasks on mobile platforms. Recently, DVFS methods tailored for mobile LLMs have also been proposed. However, these methods mostly f
    
[^70]: 他们是彼此的什么人？面向说话人关系推断的多智能体推理

    Who Are They to Each Other? Multi-Agent Reasoning for Speaker Relationship Inference

    [https://arxiv.org/abs/2609.09628](https://arxiv.org/abs/2609.09628)

    该论文提出了一种无需训练的多智能体推理框架，通过LLM智能体之间的结构化辩论与裁决机制来推断口语对话中的说话人关系，克服了监督建模成本高和现有推理方法结构化不足的问题。

    

    从口语对话中推断说话人之间的关系是迈向具有社会意识的语音理解的重要一步。然而，这一任务仍然未被充分探索，且监督式建模的训练和扩展成本高昂。与此同时，现有的推理时大语言模型（LLM）方法在处理细微、分散且多模态的关系线索方面提供的结构有限，而这些线索可能支持多种合理的解释。为了解决这些局限性，我们引入了一个无需训练的多智能体推理框架，该框架通过LLM智能体之间的结构化交互来组织推理，使关系判断能够被提出、质疑和裁决，而无需任务特定的训练。我们用两种互补的设计来实例化该框架。我们提出了多角色多智能体辩论，作为标准多智能体辩论在说话人关系推断任务上的特定适配，为智能体分配互补的角色或社会理论指导……

    arXiv:2609.09628v1 Announce Type: cross  Abstract: Inferring speaker relationships from spoken conversations is an important step towards socially aware speech understanding. However, this task remains underexplored, and supervised modeling is costly to train and scale. At the same time, existing inference-time LLM approaches provide limited structure for handling subtle, distributed, and multimodal relational cues that may support multiple plausible interpretations. To address these limitations, we introduce a training-free multi-agent reasoning framework that organizes inference through structured interaction among LLM agents, allowing relationship judgments to be proposed, challenged, and adjudicated without task-specific training. We instantiate this framework with two complementary designs. We propose Multi-Role Multi-Agent Debate as a task-specific adaptation of standard multi-agent debate for speaker relationship inference, assigning agents complementary roles or social-theory-g
    
[^71]: CityPlanner：面向可执行城市规划的沙盒智能体

    CityPlanner: A Sandbox Agent for Executable Urban Planning

    [https://arxiv.org/abs/2609.09578](https://arxiv.org/abs/2609.09578)

    CityPlanner 提出了一个基于沙盒环境的可执行城市规划智能体框架，通过统一的文件化环境 UrbanSandbox 和将长轨迹分解为“初始构建”与“反馈改进”两个原子任务的强化学习方法，在真实世界基准上持续优于现有方法。

    

    城市规划是一个现实世界中的空间优化问题，需要在成本和服务质量等实际目标约束下，从庞大的候选空间中选择可行的行动方案。现有的优化方法和强化学习方法虽然对固定形式的问题有效，但通常依赖于任务特定的表示方式和约束处理机制。我们提出了 CityPlanner，一个面向可执行城市规划的沙盒智能体框架。CityPlanner 引入了 UrbanSandbox，这是一个统一的基于文件的环境，智能体可以在其中查看任务文件、生成规划方案、运行评估器，并根据可执行的反馈修订决策。为了让学习过程更加可行，我们进一步提出了原子任务强化学习方法，将漫长的沙盒轨迹分解为用于初始方案构建的 BuildPlan 和用于基于反馈进行优化的 ImprovePlan 两个子任务。在真实世界基准上的实验表明，CityPlanner 持续超越（摘要在此处截断）……

    arXiv:2609.09578v1 Announce Type: cross  Abstract: Urban planning is a real-world spatial optimization problem that requires selecting feasible actions from large candidate spaces under practical objectives such as cost and service quality. Existing optimization and reinforcement learning methods are effective for fixed formulations, but often depend on task-specific representations and constraint handling. We propose \emph{CityPlanner}, a sandbox-agent framework for executable urban planning. CityPlanner introduces \emph{UrbanSandbox}, a unified file-based environment where agents inspect task files, generate plans, run evaluators, and revise decisions based on executable feedback. To make learning tractable, we further propose atomic-task reinforcement learning, which decomposes long sandbox trajectories into \emph{BuildPlan} for initial construction and \emph{ImprovePlan} for feedback-based refinement. Experiments on a real-world benchmark show that CityPlanner consistently outperfo
    
[^72]: 超越高频词：基于可解释单语义特征的MonoTM主题建模框架

    Beyond Top Words: MonoTM for Topic Modeling with Interpretable Monosemantic Features

    [https://arxiv.org/abs/2609.09575](https://arxiv.org/abs/2609.09575)

    MonoTM是一个可解释的主题建模框架，通过将文档-主题混合估计与语义解释解耦——利用稀疏自编码器完整特征表示估计混合比例，并在基于语料库的语义特征词汇上学习主题描述符——实现了用比单个词汇更有意义的语义单元来表示主题。

    

    主题模型用于总结大型文本语料库，但排名靠前的词汇往往只能有限地表征主题语义。稀疏自编码器（SAEs）提供了一种超越词级描述符的途径，即从密集表示中提取可解释的特征，然而特征可解释性与主题推断质量之间的关系仍不清楚。我们提出了MonoTM，一个将这些角色解耦的可解释主题建模框架。在三个基准语料库上，我们证明了文档-主题混合估计与语义解释偏好不同的SAE配置和特征子集。MonoTM从完整的SAE特征词袋表示中估计文档-主题混合比例，并在固定这些比例的基础上，在一个独立的、基于语料库的语义特征词汇表上学习主题描述符。这种设计在保持全局主题结构的同时，用比单个词汇更有意义的语义单元来表示主题，使其更...

    arXiv:2609.09575v1 Announce Type: new  Abstract: Topic models summarize large text corpora, but top-ranked words often provide only a limited representation of topic semantics. Sparse autoencoders (SAEs) offer a way to move beyond word-level descriptors by extracting interpretable features from dense representations, yet how feature interpretability relates to topic-inference quality remains unclear. We introduce \textbf{MonoTM}, an interpretable topic modeling framework that decouples these roles. Across three benchmark corpora, we show that document--topic mixture estimation and semantic interpretation favor different SAE configurations and feature subsets. MonoTM estimates mixtures from the full SAE bag-of-features representation and, with them fixed, learns topic descriptors over a separate vocabulary of corpus-grounded semantic features. This design preserves global topic structure while representing topics with semantic units more meaningful than individual words, making them mor
    
[^73]: 面向检索增强应用的日语新闻中省略时间表达的复原

    Reproducing Omitted Temporal Expressions in Japanese News for Retrieval-Augmented Applications

    [https://arxiv.org/abs/2609.09569](https://arxiv.org/abs/2609.09569)

    本文提出jaROTE，一个基于规则的流水线，能够在日语新闻被索引进搜索或RAG系统之前，利用发布日期将省略的时间表达（如仅有日期或仅有月份的表述）复原为具体日期或时间区间，实验表明其性能高、速度快、成本低，且与大型语言模型相比仍具竞争力。

    

    新闻文章中常常包含省略的时间表达，例如仅提及日期或仅提及月份的表述，这些表达必须参照发布日期才能正确解读。当此类文章在搜索和检索增强生成（RAG）系统中作为独立文本被索引或处理时，这些省略会导致时间上的不匹配以及大语言模型解读的不稳定。我们致力于在文章被索引用于搜索和RAG应用之前，以发布日期作为外部上下文，将省略的时间表达复原为具体的日期或时间区间。具体而言，我们在已有成熟的时间表达抽取与规范化技术基础上，结合对日语新闻文章的人工分析，提出了jaROTE——一个面向日语新闻的基于规则的流水线。在两个新闻语料库上的实验表明，jaROTE取得了高性能表现，在提供快速、低成本方案的同时，其表现仍与大型语言模型具有竞争力。

    arXiv:2609.09569v1 Announce Type: new  Abstract: News articles often contain omitted temporal expressions, such as day-only or month-only mentions, which must be interpreted with reference to the publication date. When such articles are indexed or processed as standalone text in search and retrieval-augmented generation (RAG) systems, these omissions can cause temporal mismatches and unstable interpretation by large language models. We focus on reproducing omitted temporal expressions as concrete dates or intervals using the publication date as external context before the articles are indexed for search and RAG applications. Specifically, building on established temporal-expression extraction and normalization techniques and informed by a manual analysis of Japanese news articles, we propose jaROTE, a rule-based pipeline for Japanese news. Experiments on two news corpora demonstrate that jaROTE achieves high performance, and remains competitive with LLMs while providing a fast, low-cos
    
[^74]: 基于引文图的演化树自动生成方法研究

    Towards Automatic Evolution Tree Generation from Citation Graphs

    [https://arxiv.org/abs/2609.09561](https://arxiv.org/abs/2609.09561)

    EvoTree是一个分阶段框架，通过解耦概念主干学习与时间细化，结合图感知编码器、单调路径约束下的微调和LLM概念标注，首次实现了从引文图自动生成方法演化树，并发布了该任务首个涵盖11个AI子领域的标注基准。

    

    综述仍然是研究者了解AI子领域中方法演进脉络的主要途径，但面对当前论文发表的迅猛速度，其扩展性严重不足。现有的分类体系构建方法大多局限于叶节点且不感知时间信息，往往将过渡性论文强行归入成熟的叶节点，并可能在祖先与后代之间产生拓扑倒置。我们提出了EvoTree，一个分阶段的框架，将概念主干学习与时间维度细化解耦：具备图感知能力的编码器结合基于分布的层次聚类生成稳定的分类体系主干；随后通过时间维度的微调，在单调路径约束下将边缘论文重新挂载到内部节点；最后由大语言模型（LLM）为概念添加标签而不改变拓扑结构。我们发布了该任务的首个标注基准数据集，涵盖11个AI子领域。EvoTree在所有基线方法中取得了最高的NMI和引文方向准确率，并在标注数据上获得了最佳的概念纯度。

    arXiv:2609.09561v1 Announce Type: new  Abstract: Surveys remain the primary way researchers grasp the lineage of methods within an AI subfield, but they scale poorly against the current rate of publication. Existing taxonomy-induction methods are largely leaf-bound and time-agnostic; they tend to force transitional papers into mature leaves and can create topological inversions between ancestors and descendants. We propose EvoTree, a staged framework that decouples conceptual backbone learning from temporal refinement: a graph-aware encoder with distribution-based hierarchical clustering yields a stable taxonomy backbone; temporal fine-tuning then re-attaches marginal papers to internal nodes under monotonic-path constraints; a final LLM pass labels concepts without altering the topology. We release the first annotated benchmark for this task across 11 AI subfields. EvoTree attains the highest NMI and citation-direction accuracy among all baselines and the best concept purity on the an
    
[^75]: BuzzASR：超过100个单语语音识别模型集群

    BuzzASR: A Swarm of 100+ Monolingual Speech Recognition Models

    [https://arxiv.org/abs/2609.09554](https://arxiv.org/abs/2609.09554)

    BuzzASR通过将Whisper模型在102种语言上进行单语微调，并结合分词器替换和纯文本数据增强等语言适应策略，在77种语言上超越了Whisper-large-v3的语音识别性能。

    

    我们介绍了BuzzASR，这是一个针对102种语言自动语音识别（ASR）任务进行语言专门化微调的Whisper模型集合。基于Transformer的大型端到端ASR模型（如Whisper）已经彻底改变了语音识别领域，但大多数知名模型都是高度多语言的。因此，这些模型在训练集中代表性不足的语言上往往表现较差。虽然人们早已知道可以通过在单语数据上进行简单微调来实现有效的语言适应，但这一策略此前仅应用于少数语言。我们将这种简单方法大规模扩展到FLEURS数据集涵盖的102种语言，同时还实现了一种更复杂的语言适应策略，该策略集成了单语分词器替换和使用纯文本微调的数据增强。BuzzASR模型在102种语言中的77种上超越了Whisper-large-v3，降低了字符错误率。

    arXiv:2609.09554v1 Announce Type: new  Abstract: We introduce BuzzASR, a collection of language-specialized fine-tuned Whisper models adapted for automatic speech recognition (ASR) in 102 languages. Large end-to-end Transformer-based ASR models such as Whisper have revolutionized ASR, but most prominent models are highly multilingual. As a result, these models often perform poorly on languages less well-represented in their training set. While it has long been known that effective language adaptation can be achieved through simple fine-tuning on monolingual data, this strategy has only been applied to a small number of languages. We massively scale up this simple approach to 102 languages covered in the FLEURS dataset, while also implementing a more complex language adaptation strategy that integrates monolingual tokenizer replacement and data augmentation using text-only fine-tuning. BuzzASR models outperform Whisper-large-v3 on 77 out of 102 languages, reducing character error rates 
    
[^76]: TEFM：面向结构化数据的令牌高效忠实建模

    TEFM: Token-Efficient Faithful Modeling for Structured Data

    [https://arxiv.org/abs/2609.09552](https://arxiv.org/abs/2609.09552)

    TEFM框架通过将结构化数据压缩为行为代码令牌并结合双保真度目标，使大语言模型在关键领域数据分析中同时实现高令牌效率（令牌消耗仅约1-2%）与忠实可解释的推理。

    

    在本文中，我们解决了将大语言模型（LLM）应用于关键领域时的两个基本障碍：令牌效率和忠实性。为了同时应对这两个约束，我们提出了TEFM（令牌高效忠实建模），一个专为关键领域结构化数据分析设计的框架。TEFM通过将冗长的结构化观测数据压缩为紧凑的行为代码令牌来实现令牌效率，以极少的信息损失大幅降低令牌消耗。此外，TEFM通过双保真度目标实现忠实推理，该目标联合优化代码级重构和预测级保真度，从而识别出基于输入数据的最小充分特征子集。在多种领域数据集和模型骨干（Qwen3、Gemma-2、Phi-4）上的全面实验表明，TEFM在实现具有竞争力的分类精度的同时，大幅减少了令牌消耗（在临床领域约保留1%的令牌，在其他领域约保留2%）。

    arXiv:2609.09552v1 Announce Type: new  Abstract: In this paper, we solve two fundamental obstacles in applying LLMs to critical domains: token efficiency and faithfulness. To address both constraints jointly, we present TEFM (Token-Efficient Faithful Modeling), a framework designed for structured data analysis in critical domains. TEFM achieves token efficiency by compressing lengthy structured observations into compact Behavioral Code tokens, dramatically reducing token consumption with minimal information loss. Moreover, TEFM enables faithful rationalization through a dual-fidelity objective that jointly optimizes code-level reconstruction and prediction-level fidelity, identifying minimal sufficient feature subsets grounded in input data. Comprehensive experiments across various domain datasets and model backbones (Qwen3, Gemma-2, Phi-4) show that TEFM achieves competitive classification accuracy with dramatic token reduction (approximately 1\% token retention in clinical and 2\% in
    
[^77]: 一种针对推荐系统的高效且有效的智能体群体托攻击方法

    An Efficient and Effective Agentic Group Shilling Attack on Recommender Systems

    [https://arxiv.org/abs/2609.09551](https://arxiv.org/abs/2609.09551)

    提出了一种基于多智能体协同的推荐系统托攻击框架AGAS，通过中央协调者动态调度可切换角色的工作智能体，跨不同受害推荐系统自适应地推广目标物品，兼具高效性与抗检测能力。

    

    推荐系统已成为现代在线平台的核心基础设施，能够大规模地实现内容个性化，并强烈影响用户看到、点击和购买的内容。然而，这种对用户交互的依赖也使推荐系统暴露于托攻击（水军攻击）的风险之中，恶意攻击者可以通过注入虚假用户档案来扭曲物品排名并控制曝光度。现有攻击方法通常依赖于针对特定目标的微调或固定的档案模板，这使其要么难以适应不同的受害系统，要么更容易被检测出来。为克服这些局限性，我们提出了智能体群体攻击系统（AGAS），这是一个协同式的托攻击框架，由中央协调者指挥一组可切换角色的工作智能体，跨不同的受害系统家族自适应地推广目标物品。当进展停滞或抑制信号增强时，协调者会动态调整策略，而工作者智能体则追求共同目标并在不同角色之间切换（原文摘要在此处截断）。

    arXiv:2609.09551v1 Announce Type: cross  Abstract: Recommender systems have become core infrastructure for modern online platforms, personalizing content at scale and strongly influencing what users see, click on, and purchase. However, this dependence on user interaction also exposes them to shilling attacks, where malicious actors can inject fake profiles to distort item rankings and control visibility. Existing attacks often rely on target-specific fine-tuning or fixed profile templates, making them either difficult to adapt to different victims or easier to detect. To overcome these limitations, we propose the Agentic Group Attack System (AGAS), a coordinated shilling framework where a central Coordinator directs a group of role-switching worker agents to adaptively promote a target item across different victim families. The Coordinator dynamically adjusts the strategy when progress stalls or suppression signals increase, while workers pursue a shared objective and switch between a
    
[^78]: 机器言说的变异

    The Mutations of Machine Speech

    [https://arxiv.org/abs/2609.09496](https://arxiv.org/abs/2609.09496)

    本文追溯了算法输出的演变历程，揭示了机器言说的三种变异形态——作为可查询数据的言论（搜索引擎）、作为互动参与的言论（社交媒体）以及作为生成文本的言论（对话式AI），并分析了其背后的法律基础与社会影响。

    

    算法输出如今遍布于组织当代生活的数字环境之中。法律在促进和构建（而不仅仅是回应）这些过程中所扮演的角色，正在学术界获得越来越多的关注。本研究追溯了算法输出的演变历程，关注其法律基础与社会影响，揭示了机器言说的变异形态。第一种变异将言论重新定义为可供查询的数据：搜索引擎将网络从信息检索空间转变为算法可见性的经济体制。第二种变异将言论重新框定为互动参与：社交媒体平台将内容审核与放大相融合，把表达变成了一种由企业架构所支配的注意力指标。第三种变异出现在对话系统和界面之中，生成式文本取代了信息检索，随之而来的是更密集的……（原文摘要在此截断）

    arXiv:2609.09496v1 Announce Type: new  Abstract: Algorithmic outputs now populate the digital environments through which contemporary life is organized. The role of law in facilitating and constituting (rather than merely responding to) these processes is gaining increasing traction across scholarly accounts. This inquiry traces the evolution of algorithmic outputs attending to their legal underpinnings and social implications, surfacing the mutations of machine speech.   The first mutation redefined speech as data to be queried: search engines transformed the web from a space of information retrieval into an economic regime of algorithmic visibility. The second mutation reframed speech as engagement: social media platforms fused moderation with amplification, turning expression into a metric of attention, governed by corporate architectures. The third mutation emerges in conversational systems and interfaces, where generative text displaces information retrieval, bringing with it dens
    
[^79]: 从固定按键到可读模式：用于车辆智能体函数调用的小型语言模型

    From Fixed Keys to Readable Schemas: Small Language Models for Vehicle Agent Function Calls

    [https://arxiv.org/abs/2609.09476](https://arxiv.org/abs/2609.09476)

    该论文构建了一个基于Android Automotive、包含9,822个示例和79个车辆功能的车载函数调用基准，并在四个不同规模的小型语言模型上系统比较了功能令牌（紧凑推理但仅限已训练功能）与提示词模式（可泛化到新功能但推理开销更高）两种设计方案的优劣。

    

    车载助手必须在严格的内存和延迟约束下，将自然语言请求转换为准确的车辆功能调用，这使得小型语言模型（SLM）成为设备端部署的理想选择。对于此类模型，一个关键的设计选择是如何呈现可用的功能接口。目前有两种方法：为每个功能分配专用的功能令牌，或直接在提示词中提供功能模式。功能令牌（FT）能够实现紧凑的推理，但仅限于训练期间学习过的功能；而提示词模式（SIP）虽然可以泛化到未见过的功能，但代价是更长的提示词和更高的推理开销。我们引入了一个包含9,822个单轮示例的基准数据集，涵盖源自Android Automotive的79个车辆功能，其中包括保留功能和需要拒绝的请求。我们在从270M到1.7B参数的四个小型语言模型上，在匹配的微调条件下对两种方法进行了比较。

    arXiv:2609.09476v1 Announce Type: cross  Abstract: In-vehicle assistants must translate natural-language requests into accurate vehicle function calls under strict memory and latency constraints, making small language models (SLMs) attractive for on-device deployment. For such models, a key design choice is how the available function surface is presented. Two approaches are to represent each function with a dedicated Functional Token (FT) or provide function schemas directly in the prompt. FTs enable compact inference but are restricted to functions learned during training, whereas Schema-in-Prompt (SIP) can generalize to unseen functions at the cost of longer prompts and higher inference overhead. We introduce a benchmark of 9,822 single-turn examples spanning 79 vehicle functions derived from Android Automotive, including held-out functions and requests requiring refusal. We compare both approaches under matched fine-tuning across four SLMs from 270M to 1.7B parameters. On functions 
    
[^80]: Edu-QuRating：基于蒸馏成对判断的多维度教育数据筛选

    Edu-QuRating: Multi-Dimensional Educational Data Curation with Distilled Pairwise Judgements

    [https://arxiv.org/abs/2609.09425](https://arxiv.org/abs/2609.09425)

    Edu-QuRating提出了一种多维度教育数据筛选流水线，通过定义教育专用评分标准、利用LLM评判器标注文档对并将成对偏好蒸馏为可复用的评分模型，从而突破传统单一标量式教育价值评估的局限，从准确性、吸引力、结构性和受众适用性等多个维度对文本进行精细化评分。

    

    教育数据过滤器已成为改进语言模型预训练的实用方法，但大多数过滤器将教育价值视为单一的标量属性。这对于某些应用来说可能过于宽泛，尤其是当数据集本身已经具有高密度的教育材料时。有用的学习材料需要准确、有吸引力、结构良好，并且适合目标受众和应用场景（例如面向学习者还是面向教师）。延续QuRating（Wettig等人，2024）的工作，我们提出了Edu-QuRating：一个用于多维度教育数据评分与筛选的流水线。Edu-QuRating定义了教育专用的评分标准（rubrics），使用LLM评判器对采样的文档对进行标注，并将这些成对偏好蒸馏为可复用的Edu-QuRater模型，这些模型可以根据一组教育标准对单个文本片段进行评分。在两个序列分类基础模型和六个教育标准的实验中，最好的Edu-QuRater能够恢复保留的（摘要在此处被截断）

    arXiv:2609.09425v1 Announce Type: new  Abstract: Educational data filters have become a practical way to improve language-model pre-training, but most filters treat educational value as a single scalar property. This may be too broad for some applications, especially if the data set already features a high density of educational material. Useful learning material needs to be accurate, engaging, well structured, and appropriate for the intended audience and application (e.g. learner- vs teacher-facing). Following QuRating (Wettig et al. 2024), we introduce Edu-QuRating: a pipeline for multi-dimensional educational data scoring and curation. Edu-QuRating defines education-specific rubrics, uses an LLM judge to label sampled document pairs and distills those pairwise preferences into reusable Edu-QuRaters, which can score individual text chunks on a set of educational criteria. Across two sequence-classification base models and six educational criteria, the best Edu-QuRater recovers held-
    
[^81]: 跨数据库查询与网络搜索的混合深度研究基准测试

    Benchmarking Hybrid Deep Research Across Database Querying and Web Search

    [https://arxiv.org/abs/2609.09410](https://arxiv.org/abs/2609.09410)

    提出了首个需要同时结合网络搜索与SQL的混合深度研究基准HybridDeepResearch，包含380个任务，用于评估智能体在非结构化网络文本与结构化数据库之间传递证据并保持约束条件的能力。

    

    尽管自主智能体在“深度研究”方面已取得显著进展——通过迭代式地浏览开放网络来综合信息，但现实世界中的问题解决很少局限于单一环境。复杂的分析任务本质上要求智能体将来自模糊的非结构化文本（如开放网络）和高度精确的结构化数据（如关系数据库）的证据融合在一起。然而，现有的基准测试将这些模态孤立地进行评估，未能捕捉到关键的“交接”能力——即在系统之间传递证据时保持约束条件的能力。我们提出了HybridDeepResearch，据我们所知，这是首个需要同时使用网络搜索和SQL才能形成完整、可验证答案的深度研究基准。该基准包含380个依赖工具的任务，基于LiveSQLBench-Base-Lite数据库和公共网络语料库构建，经过自动化检查和人工审核验证，并涵盖三种（推理……）

    arXiv:2609.09410v1 Announce Type: new  Abstract: While autonomous agents have made significant strides in "deep research" by iteratively navigating the open web to synthesize information, real-world problem-solving is rarely confined to a single environment. Complex analytical tasks inherently require agents to weave together evidence from both ambiguous unstructured text (e.g., the open web) and highly precise structured data (e.g., relational databases). However, existing benchmarks evaluate these modalities in isolation, failing to capture the critical "handoff" - the ability to preserve constraints when moving evidence between systems. We introduce HybridDeepResearch, to our knowledge the first deep-research benchmark that requires both web search and SQL to form a complete, verifiable answer. The benchmark contains 380 tool-dependent tasks grounded in LiveSQLBench-Base-Lite databases and public web corpora, validated through automated checks and human review, and covering three re
    
[^82]: MMLU 究竟测量了什么？对聚合基准分数中难度结构的心理测量学审计

    What Does MMLU Actually Measure? A Psychometric Audit of Difficulty Structure in Aggregate Benchmark Scores

    [https://arxiv.org/abs/2609.09372](https://arxiv.org/abs/2609.09372)

    本研究通过项目反应理论和心理测量学分析证明，MMLU 聚合分数主要测量模型的事实检索能力而非推理能力，且其难度结构在 STEM 与非 STEM 分区之间不可迁移，导致排行榜排名更多反映非 STEM 表现。

    

    尽管 MMLU 被广泛用作校准通用 AI 能力的基准，我们通过心理测量学方法证明，其聚合分数主要评估的是模型的事实检索能力而非推理能力。通过使用项目反应理论（Item Response Theory）对 1,000 个开源权重语言模型在 14,042 个 MMLU 测试题目上的项目难度进行校准，我们表明通过单一测试同时评估这两种能力本身就存在缺陷。随后，我们将难度回归到一个确定性的、可从文本提取的结构复杂性框架上。应用带有学科聚类协方差的联合 Wald 检验表明，MMLU 混淆了本质上可分离的构念。从结构复杂性到难度的映射在基准的 STEM 与非 STEM 分区之间并非不变。这一发现具有实际意义：聚合排行榜的排名与非 STEM 准确率的相关性比与 STEM 准确率更为紧密，因此在选择 Top-50……

    arXiv:2609.09372v1 Announce Type: cross  Abstract: Although MMLU is widely adopted as a benchmark for calibrating general AI capabilities, we psychometrically demonstrate that its aggregate score primarily evaluates a model's factual retrieval capacity rather than its reasoning ability. By calibrating item difficulty for 1,000 open-weights language models over 14,042 MMLU test items using Item Response Theory, we show that evaluating both abilities via a single test is inherently flawed. Difficulty is then regressed on a deterministic, text-extractable framework of structural complexity. Applying a joint Wald test with subject-clustered covariances demonstrates that the MMLU conflates fundamentally separable constructs. The mapping from structural complexity to difficulty is not invariant across the benchmark's STEM and non-STEM partitions. This finding has practical consequences. Aggregate leaderboard ranks track non-STEM accuracy more closely than STEM accuracy, so selecting a Top-50
    
[^83]: 如果大语言模型不相信输入数据，它们会犯更多错误吗？

    Do LLMs Make More Mistakes If They Do Not Believe the Input Data?

    [https://arxiv.org/abs/2609.09363](https://arxiv.org/abs/2609.09363)

    本研究通过让大语言模型基于捷克和斯洛伐克本地知识的事实性、反事实及虚构RDF三元组生成多种语言文本，发现模型对不可信输入数据仅表现出较弱的上下文-记忆冲突，即模型并不一定会因不相信输入数据而犯更多错误。

    

    大语言模型（LLM）容易产生幻觉或误释事实，这损害了它们在检索增强生成或数据到文本系统中的可用性。我们分析了大语言模型对所提供上下文的忠实度如何取决于它们对上下文可信度的感知（即上下文-记忆冲突）。为了更好地识别错误模式，我们利用了非英语和低资源语言文本生成难度更高的特点，以及基于本地知识的输入数据（这些知识仅被部分包含在模型的参数知识中）。我们让模型根据包含捷克和斯洛伐克本地数据的事实性（FA）、反事实（CFA）和虚构（FI）RDF三元组，生成英语、捷克语、斯洛伐克语和上索布语的文本。与我们的预期相反，在人工标注样本上，我们仅观察到较弱的上下文-记忆冲突。对于作为LLM裁判的Kimi K3（该裁判与人工标注结果高度一致），反事实输入获得……

    arXiv:2609.09363v1 Announce Type: new  Abstract: Large language models (LLMs) are prone to hallucinating or misinterpreting facts, which impairs their usability in retrieval-augmented generation or data-to-text systems. We analyse how faithfulness of LLMs to provided context depends on how plausible they perceive the context to be (context-memory conflict). To better identify error patterns, we make use of the increased difficulty of non-English and low-resource language text generation and input data based on local knowledge, only partially captured in models' parametric knowledge. We let the models generate text in English, Czech, Slovak and Upper Sorbian from factual (FA), counterfactual (CFA) and fictional (FI) RDF triples containing local Czech and Slovak data. Contrary to our expectations, we observe only a weak context-memory conflict on the human-annotated sample. For Kimi K3 as an LLM judge, which agrees well with human annotations on the sample, counterfactual inputs receive 
    
[^84]: 印度母婴护理中可审计的紧急分诊系统

    Auditable Emergency Triage for Maternal and Newborn Care in India

    [https://arxiv.org/abs/2609.09356](https://arxiv.org/abs/2609.09356)

    该论文将LLM紧急分诊系统分解为症状提取与紧急性判断两个可审计的步骤，并引入结构化决策树，解决了大规模母婴护理分诊系统中不透明、难以调试和迭代成本高的问题。

    

    在Noora Health，我们的护士每月在基于WhatsApp的服务上回答超过5万个医疗咨询，该服务为照护者提供按需支持。他们最紧迫的任务是紧急分诊：判断哪些咨询需要立即进行面对面处理。为支持这项工作，我们构建了一个使用大语言模型（LLM）的系统，对消息是否属于紧急情况进行分类，并提供可解释性的理由。但该系统是不透明的：分析错误意味着需要逐条阅读每条消息的推理链，这在我们服务的规模下是不可行的。提示词的任何修改都需要重新运行完整的评估以防止性能退化，这既成本高昂又在运营上极具挑战性。临床医生遵循一棵决策树来做出分诊判断，但这棵决策树从未被记录下来或传递给模型，模型仅依赖于一份平铺的危险体征列表。为解决这些问题，我们将分诊分解为两个步骤：由LLM提取规范化的症状和患者情况，再基于此进行紧急性判断，从而使整个系统变得可审计、可维护，并支持安全的迭代改进。

    arXiv:2609.09356v1 Announce Type: new  Abstract: At Noora Health, our nurses answer more than 50,000 medical queries per month on our WhatsApp-based service that provides caregivers with on-demand support. Their most time-critical task is emergency triage: deciding which queries need immediate in-person attention. To support them, we built a system that uses a large language model (LLM) to classify whether a message is an emergency and provide a rationale for interpretability. But the system was opaque: analyzing mistakes meant reading reasoning chains for each message, which is infeasible at our scale. Prompt changes meant re-running a full evaluation to prevent regressions, which was both costly and operationally challenging. Clinicians follow a decision tree to make this call, but it was never documented or passed to the model, which relied on a flat list of danger signs. To address these issues, we decomposed triage into two steps: an LLM extracts canonical symptoms and patient con
    
[^85]: SWORD：基于Wikidata的扰动揭示了大语言模型事实错误拒绝中隐藏的跨语言不一致性

    SWORD: Wikidata-based Distortions Reveal Hidden Cross-Lingual Inconsistencies in LLM Factual Error Rejection

    [https://arxiv.org/abs/2609.09349](https://arxiv.org/abs/2609.09349)

    SWORD基准通过对Wikidata三元组进行扰动来评估大语言模型跨语言拒绝事实错误的能力，发现模型在语义合理的扰动上反而比随机替换表现更好，暴露出模型依赖分布熟悉性而非真正事实验证的隐藏跨语言不一致性。

    

    现代大语言模型展现出令人印象深刻的多语言性能，然而标准基准主要奖励选择正确答案，而非评估真正的事实理解能力。我们提出了基于Wikidata的系统化客体-关系扰动基准，该基准评估模型是否能跨语言一致地拒绝事实错误。SWORD通过对Wikidata三元组进行受控扰动，在八种广泛使用的语言中生成语法正确但事实错误的陈述，扰动方式涵盖从随机实体替换到语义上合理的基于属性的选择。我们基于扰动的评估揭示了两个被传统基准完全掩盖的关键洞察。首先，反直觉的是，模型在语义合理的扰动上反而比在无意义的随机替换上获得更高的准确率，这表明模型依赖于分布上的熟悉性而非真正的事实验证。

    arXiv:2609.09349v1 Announce Type: new  Abstract: Modern LLMs demonstrate impressive multilingual performance, yet standard benchmarks primarily reward selecting correct answers rather than evaluating genuine factual understanding. We introduce Systematic Wikidata-based Object-Relation Distortion (SWORD), a benchmark that evaluates whether models consistently reject factual errors across languages. SWORD generates syntactically well-formed but factually incorrect statements in eight widely spoken languages through controlled perturbations of Wikidata triples, ranging from random entity substitutions to semantically plausible property-based selections. Our distortion-based evaluation surfaces two critical insights that remain entirely obscured by conventional benchmarks. First, models counterintuitively achieve higher accuracy on semantically plausible distortions than on nonsensical random substitutions, suggesting reliance on distributional familiarity rather than genuine factual verif
    
[^86]: Osprey：目标无关的预训练让投机解码中的起草模型更强大

    Osprey: Target-agnostic Pre-training Makes Stronger Drafters in Speculative Decoding

    [https://arxiv.org/abs/2609.09338](https://arxiv.org/abs/2609.09338)

    Osprey提出直接利用现成的预训练小型语言模型构建目标无关的起草模型，将大规模预训练作为可复用资产，仅需轻量级适配即可服务于任意目标模型，从而解决了投机解码中起草模型泛化差、加速效果脆弱的问题。

    

    投机解码对于加速大语言模型（LLM）推理至关重要。然而，其加速效果十分脆弱：起草模型通常只针对单一目标模型的狭窄分布进行训练，当工作负载发生变化时，其接受率会大幅下降。这与现代LLM开发形成了鲜明反差——目标模型恰恰因大规模预训练所获得的广泛泛化能力而备受重视。我们认为，自然可行的解决方案（即预训练）之所以难以应用于起草模型，是因为现有方法都是目标特定的：起草模型需要消费目标模型的隐藏状态，并在目标模型的logits上进行蒸馏，因此必须为每个目标模型重复进行预训练。我们提出了Osprey，它从现成的预训练小型语言模型出发来构建起草模型，将广泛的预训练视为一种可复用的、目标无关的资产，并将针对每个目标模型所需的工作简化为轻量级的适配步骤。实现这一点需要克服……

    arXiv:2609.09338v1 Announce Type: new  Abstract: Speculative decoding is critical for accelerating LLM inference. However, the speedup is fragile: drafters are typically trained against a narrow distribution for a single target model, and their acceptance rate collapses under workload shifts. This is a striking inversion of modern LLM development, where target models are valued precisely for the broad generalization they acquire through large-scale pretraining. We argue that the natural remedy, pretraining, has been hard to apply to drafters because existing recipes are target-specific: the drafter consumes the target's hidden states and is distilled on the target's logits, so pretraining must be repeated for each target. We introduce Osprey, which instead bootstraps drafters from off-the-shelf pretrained small language models, treating broad pretraining as a reusable, target-agnostic asset and reducing per-target work to a lightweight adaptation step. Realizing this requires overcomin
    
[^87]: StochBench：面向Lean中随机过程的领域专用基准测试

    StochBench: A Domain-Specific Benchmark for Stochastic Processes in Lean

    [https://arxiv.org/abs/2609.09264](https://arxiv.org/abs/2609.09264)

    StochBench是一个包含450道研究生水平随机过程题目的Lean 4领域专用基准测试，填补了形式化定理证明基准中应用数学领域代表性不足的空白，基于Opus 4.8的智能体在每题15分钟时限下达到34.9%的证明率。

    

    当前用于大语言模型形式化定理证明的主流基准测试是来自竞赛数学（如国际数学奥林匹克竞赛和普特南数学竞赛）的小规模题目集，它们对特定领域应用的代表性较差。我们提出了StochBench，这是一个基于Lean 4的基准测试，包含450道研究生水平的随机过程题目，涵盖不同抽象层次，每道题目均配有其自然语言原文。针对Mathlib中代表性不足的领域，该基准测试涵盖了有限与可数马尔可夫链、更新过程、随机游走、鞅、停时、排队论、布朗运动、随机分析、弱收敛以及泊松过程和连续时间马尔可夫过程。我们基于Opus 4.8的智能体在每题15分钟的时限下达到了34.9%的证明率（157/450）。StochBench更好地代表了领域特定的应用数学，同时对先进的自动证明器而言仍具有挑战性。

    arXiv:2609.09264v1 Announce Type: new  Abstract: Leading benchmarks for formal theorem proving with large language models are small collections drawn from competition math, such as the IMO and Putnam, that poorly represent field-specific applications. We introduce StochBench, a Lean 4 benchmark of 450 graduate stochastic-processes problems at varying abstraction levels, each paired with its natural-language source. Addressing a field underrepresented in Mathlib, it covers finite and countable Markov chains, renewal processes, random walks, martingales, stopping times, queues, Brownian motion, stochastic calculus, weak convergence, and Poisson and continuous-time Markov processes. Our Opus 4.8-based agent achieves a 34.9% proof rate (157/450) under a 15-minute per-problem limit. StochBench better represents domain-specific applied mathematics while remaining challenging for advanced provers.
    
[^88]: 我们该信任RAG吗？测量检索增强生成在文档投毒下的鲁棒性

    In RAG We Trust? Measuring Robustness of Retrieval-Augmented Generation Under Document Poisoning

    [https://arxiv.org/abs/2609.09243](https://arxiv.org/abs/2609.09243)

    本研究通过对Llama 3.1 8B进行588次因子实验，首次系统量化了检索增强生成（RAG）在文档投毒攻击下的脆弱性——当全部三篇检索段落被篡改时准确率从77.9%骤降至43.5%，其中实体替换攻击危害最大。

    

    检索增强生成（RAG）通过检索到的文档为语言模型提供事实依据，这减少了幻觉，但也带来了新的攻击面：如果检索到的文本被篡改，模型可能会重复虚假信息。我们研究了小型量化模型Llama 3.1 8B在部分检索上下文被投毒时的性能退化程度。研究测试了三种破坏策略——实体替换、数字替换和否定，分别应用于三篇检索段落中的零篇、一篇、两篇或三篇，并在基于FEVER构建的事实核查任务上进行了588次运行的因子实验。当所有三篇段落都被破坏时，准确率从干净上下文下的77.9%下降至43.5%。实体替换翻转了最大比例的原本在干净上下文中回答正确的答案。数字类破坏在投毒段落占少数时保持平稳，一旦投毒段落形成多数则急剧下降，我们通过查询级自助法置信区间再次验证了这一模式。该模型很少凭空捏造新的虚假信息。

    arXiv:2609.09243v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) grounds a language model in retrieved documents, which reduces hallucination but creates a new attack surface: if retrieved text is tampered with, the model may repeat the falsehood. We study how much a small quantized model, Llama 3.1 8B, degrades when a fraction of its retrieved context is poisoned. Three corruption strategies are tested, entity swap, number swap, and negation, each applied to zero, one, two, or three of the three retrieved passages, over a factorial sweep of 588 runs on a fact-checking task built from FEVER. Accuracy falls from 77.9% on clean context to 43.5% when all three passages are corrupted. Entity swap flips the largest share of answers that were correct on clean context. Number-based corruption stays flat while poisoned passages are a minority and jumps once they form a majority, a pattern we re-check with query-level bootstrap intervals. The model rarely invents new fals
    
[^89]: 动态稀疏混合专家模型的分布一致性推理

    Distribution-Consistent Inference for Dynamic Sparse Mixture-of-Experts

    [https://arxiv.org/abs/2609.09241](https://arxiv.org/abs/2609.09241)

    提出逐层分布对齐方法，通过在推理时校正动态减少激活专家所引起的输出分布偏移，在不重新训练的情况下降低计算成本并缓解下游性能下降。

    

    混合专家架构已成为在大型基础模型中扩展模型容量同时保持高效推理的强大范式。然而，大多数MoE模型使用固定的top-k专家选择策略，为每个token分配相同的专家预算，即使更少的专家可能已经足够。推理时的动态top-k路由可以在不重新训练的情况下减少计算量，但现有方法往往忽略了偏离训练时路由配置所导致的分布偏移。我们证明减少激活专家的数量会持续增加稀疏MoE输出的RMS尺度和方差，从而引起表示不匹配，这在专家容量损失之外还会导致下游性能下降。为了解决这一可纠正的组成部分，我们提出了逐层分布对齐，这是一种轻量级的推理时校正方法，利用逐层校准（此处摘要被截断）。

    arXiv:2609.09241v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) architectures have emerged as a powerful paradigm for scaling model capacity while preserving efficient inference in large foundation models. However, most MoE models use a fixed top-$k$ expert selection policy, assigning the same expert budget to every token even when fewer experts may be sufficient. Inference-time dynamic top-$k$ routing can reduce computation without retraining, but existing methods often overlook the distributional shift caused by deviating from the training-time routing configuration. We show that reducing the number of activated experts consistently increases the RMS scale and variance of SMoE outputs, inducing a representation mismatch that contributes to downstream performance degradation in addition to the loss of expert capacity. To address this correctable component, we propose Layer-wise Distribution Alignment (LDA), a lightweight inference-time correction that uses layer-wise calib
    
[^90]: 子代理与代理技能的对比：为长时程智能体任务执行可复用知识

    Subagents vs Agent Skills: Executing Reusable Knowledge for Long-Horizon Agentic Tasks

    [https://arxiv.org/abs/2609.09233](https://arxiv.org/abs/2609.09233)

    该研究发现，将技能包作为拥有独立全新上下文窗口的子代理来执行，相比将技能指令加载到主上下文的传统智能体技能方式，在解决长时程任务时表现更优，因为其避免了上下文信息累积导致的推理质量下降。

    

    语言模型智能体如何有效利用可复用知识库来解决长时程任务？近期的研究日益关注“智能体技能”，即以技能包形式表示的可复用能力——技能包是包含指令、脚本及其他资源的多文件捆绑集合，帮助智能体执行特定任务。智能体技能通常通过将技能指令加载到智能体的上下文中，并依赖智能体遵循这些指令来执行。然而，随着任务时程的增长，这种方法变得越来越脆弱，因为上下文窗口中积累的信息越多，推理质量就会下降。我们研究了一种替代方法，即将技能包作为子代理来调用。子代理执行并非将技能指令加载到主上下文中，而是生成全新的、专用于解决单个子任务的上下文窗口。我们表明，子代理执行优于智能体技能执行……

    arXiv:2609.09233v1 Announce Type: cross  Abstract: How can language model agents effectively leverage libraries of reusable knowledge to solve long-horizon tasks? Recent work has increasingly focused on agent skills: reusable capabilities represented as skill packages, i.e., multi-file bundles containing instructions, scripts, and other resources that help agents perform specific tasks. Agent skills are typically executed by loading their skill instructions into an agent's context and relying on the agent to follow them. As task horizons grow, however, this approach becomes increasingly brittle, because reasoning quality degrades as more information accumulates in the context window. We investigate an alternative approach in which skill packages are instead invoked as subagents. Rather than loading skill instructions into the main context, subagent execution spawns fresh context windows dedicated to solving individual subtasks. We show that subagent execution outperforms agent-skill ex
    
[^91]: 当信息分布在协同头中发生漂移时，多模态大语言模型会产生幻觉

    MLLMs Hallucinate when Information Distribution Drifts in Synergy Heads

    [https://arxiv.org/abs/2609.09206](https://arxiv.org/abs/2609.09206)

    该论文发现多模态大语言模型的幻觉源于协同注意力头中信息分布偏离健康平衡状态，而非模态信息的数量或强度，并提出HEAL方法，通过因果噪声干预和反事实双重差分实现头级别信息解耦与校准，以识别和缓解幻觉。

    

    多模态大语言模型（MLLMs）常常受到幻觉问题的困扰，这阻碍了其可靠的实际应用。现有的基于注意力的缓解方法主要依赖间接信号（如注意力权重），这些信号无法准确反映幻觉产生背后的实际信息偏移。在本文中，我们提出了HEAL——一种头级别信息解耦与校准方法，用于识别和缓解幻觉。HEAL首先对多头输出施加因果噪声干预，以过滤掉因果冗余的注意力头。随后，通过反事实双重差分法解耦剩余头中的信息分布，将注意力头划分为四种类型。通过分析，我们观察到：当信息分布偏离协同头中的健康平衡状态时，幻觉就会发生，而幻觉与模态特定信息的数量或强度并无强相关性。

    arXiv:2609.09206v1 Announce Type: cross  Abstract: Multimodal Large Language Models (MLLMs) often struggle with hallucinations, thus hindering their reliable practical applications. Existing attention-based mitigation methods mainly rely on indirect signals (e.g., attention weights) that fail to accurately reflect the actual information shift underlying hallucination generation. In this paper, we propose HEAL, Head-lEvel information disentAnglement and caLibration for identifying and mitigating hallucinations. HEAL first employs causal noise intervention on multi-head outputs to filter out causally redundant heads. Subsequently, it disentangles information distribution within the remaining heads via the counterfactual Difference-in-Differences, categorizing heads into four types. Through analysis, we observe: hallucinations happen when information distribution drifts away from a healthy equilibrium in synergy heads, not strongly correlated with the quantity or strength of modality-spec
    
[^92]: AgenticGen：面向广告的奖励引导智能体视频生成

    AgenticGen: Reward-Guided Agentic Video Generation for Advertising

    [https://arxiv.org/abs/2609.09187](https://arxiv.org/abs/2609.09187)

    AgenticGen提出了一种奖励引导的智能体框架，将广告视频生成分解为策略选择和草稿生成两个可训练的推理阶段，通过从线上业务反馈中学习性能奖励并结合人类质量准则奖励来监督策略优化，从而实现以线上业务指标为导向的广告视频生成。

    

    广告视频生成不仅仅是一个视频合成任务，更是一个以产品为条件的推理问题，其成功与否由线上业务指标来衡量。近期的视频基础模型能够根据多模态条件生成逼真的视频片段，但它们并未优化如何将产品转化为有效的广告，也未考虑如何利用线上业务反馈来改进未来的生成。为了闭合这一反馈循环，我们提出了AgenticGen，一个奖励引导的智能体框架，它将广告视频生成分解为两个可训练的推理阶段——策略选择和草稿生成，从而暴露出线上业务反馈可以监督的优化目标。AgenticGen从累积的线上反馈中学习基于性能的奖励，以及与人类质量标准对齐的互补的基于准则的奖励，然后用它们来监督策略优化。DPO首先移动智能体策略……

    arXiv:2609.09187v1 Announce Type: cross  Abstract: Advertising video generation is not only a video synthesis task, but also a product-conditioned reasoning problem whose success is measured by online business metrics. Recent video foundation models can generate realistic clips from multimodal conditions, yet they do not optimize how a product should be transformed into an effective advertisement or how future generation should be improved from online business feedback. To close this loop, we propose AgenticGen, a reward-guided agentic framework that decomposes advertising video generation into two trainable reasoning stages, strategy selection and draft generation, thereby exposing optimization targets that online business feedback can supervise. AgenticGen learns a performance-based reward from accumulated online feedback and a complementary rubric-based reward aligned with human quality standards, then uses them to supervise policy optimization. DPO first moves the agentic policies 
    
[^93]: X-CoSD：通信高效的跨词表协同投机解码

    X-CoSD: Communication-Efficient Cross-Vocabulary Collaborative Speculative Decoding

    [https://arxiv.org/abs/2609.09166](https://arxiv.org/abs/2609.09166)

    提出了X-CoSD框架，通过混合重采样将残差重采样拆分为设备端公共词表区域和服务器端大语言模型专属区域，实现了跨异构词表的无损且通信高效的协同投机解码。

    

    本文研究了协同投机解码，这是一种分布式大语言模型推理框架，其中设备端的小语言模型起草候选词元，服务器端的大语言模型对其进行验证。现有的协同投机解码方法假设小语言模型和大语言模型共享词表，并且由于残差重采样需要在用户设备和边缘服务器之间交换词元分布，会产生巨大的通信负载。为了解决这些限制，我们提出了跨词表协同投机解码，这是一个面向异构小语言模型-大语言模型词表的无损且通信高效的协同投机解码框架。X-CoSD建立在混合重采样（HR）之上，该方法将残差重采样拆分为设备端的公共词表区域和服务器端的大语言模型专属区域，从而只需传输公共词表区域的分布。我们进一步提出了X-CoSD-E，这是一种基于服务器重采样的增强变体。

    arXiv:2609.09166v1 Announce Type: new  Abstract: This paper investigates collaborative speculative decoding (CoSD), a distributed large language model (LLM) inference framework in which an on-device small language model (SLM) drafts candidate tokens and a server LLM verifies them. Existing CoSD methods assume a shared vocabulary between the SLM and the LLM and incur substantial communication load because residual resampling requires token distribution exchange between the user device and the edge server. To address these limitations, we propose cross-vocabulary CoSD (X-CoSD), a lossless and communication-efficient CoSD framework for heterogeneous SLM-LLM vocabularies. X-CoSD is built on hybrid resampling (HR), which splits residual resampling across the common-vocabulary region on the device and the LLM-only region on the server, so that distribution transmission is required only for the common-vocabulary region. We further propose X-CoSD-E, an enhanced variant based on server resampli
    
[^94]: 复制行为解释了现实中AI智能体的集体行为

    Copying explains the collective behavior of AI agents in the wild

    [https://arxiv.org/abs/2609.09150](https://arxiv.org/abs/2609.09150)

    该研究利用完整的公开编辑记录发现，现实中自主AI智能体在无人协调下的集体行为可由一条简单的复制规则解释——智能体按选项在其可见内容（尤其是眼前页面）中所占份额的概率进行选择。

    

    2026年6月，数千个AI智能体发现一个公共维基网站接受来自其沙盒内部的编辑，于是开始利用它来帮助彼此通过一项限时测试。每个智能体只存活约一小时，事后不留任何记忆。没有谁要求它们合作，该维基也并非为它们而建。它们所写内容的完整记录是公开的，且信息量异常丰富，因为它不仅保存了每个智能体写了什么，还保存了该智能体在写作之前能看到什么。我们利用这些记录追踪了智能体到达后必须做出的三个决策：在哪里写、给自己取什么名字、以及如何措辞自己的消息。一个规则支配着这三项决策：智能体以接近该选项在其可见内容中所占份额的概率来选择某个选项，而起作用的份额首先是它眼前页面上的份额，其次是最近编辑流中的份额，对更久远的内容仅有微弱影响。三个最小化的复制模型，一个……

    arXiv:2609.09150v2 Announce Type: replace-cross  Abstract: In June 2026, thousands of AI agents found that a small public wiki would accept edits from inside their sandboxes, and started using it to help one another pass a timed test. Each agent lived for about an hour and remembered nothing afterwards. Nobody asked them to cooperate, and the wiki had not been built for them. The complete record of what they wrote is public, and it is unusually informative, because it preserves not only what each agent wrote but what that agent could see before writing. We use it to follow the three decisions an agent had to make on arrival: where to write, what to call itself, and how to word its message. One rule governs all three. An agent takes an option with a probability close to the share of that option in what it can see, and the share that matters is the one on the page in front of it, then the one in the stream of recent edits, and only weakly anything older. Three minimal copying models, one
    
[^95]: 从分数到证据：可审计的决策可以改进语音深伪检测

    From Scores to Evidence: Auditable Decisions Can Improve Speech Deepfake Detection

    [https://arxiv.org/abs/2609.08899](https://arxiv.org/abs/2609.08899)

    该论文提出一种可审计的决策记录方法，将被动检测分数、条件性键控探针分数、检索支持和说话者画像边际四个线索纳入后期校准步骤，使语音深伪检测的最终决策在保持标量的同时保留证据来源信息，从而提升检测决策的可信度与可解释性。

    

    语音深伪能够以足以欺骗听众和自动化系统的方式逼真地模仿说话者的声音。这推动了语音深伪检测领域的强劲进展，但大多数检测器最终仍只输出每条语音的一个分数。该分数对于排序系统很有用，但对于为什么某个临界样本应该被信任、搁置还是复核，它几乎没有提供任何信息。两条语音可能因不同原因落入同一分数区间，例如被动证据与检索证据不一致，或者键控探针不可用。我们提出了一个问题：最终决策能否在保留这些来源信息的前提下仍然保持标量形式。我们通过一种可审计的决策记录来回答这个问题，该记录将四个对齐的线索引入后期校准步骤：被动检测器分数、在标记衍生样本上的条件性键控探针分数、检索支持度以及说话者画像边际，同时附带明确的分歧坐标。在包含4,080个样本的……

    arXiv:2609.08899v2 Announce Type: replace-cross  Abstract: Speech deepfakes can mimic a speaker's voice convincingly enough to deceive listeners and automated systems. This has driven strong progress in speech deepfake detection, but most detectors still end with one score per utterance. That score is useful for ranking systems, yet it says little about why a borderline item should be trusted, deferred, or reviewed. Two utterances can fall in the same score band for different reasons, for example because passive and retrieval evidence disagree or because the keyed probe is unavailable. We ask whether the final decision can remain scalar without discarding that provenance. We answer this question with an auditable decision record that carries four aligned cues into a late calibration step: a passive detector score, a conditional keyed-probe score on a marked derivative, retrieval support, and a speaker-profile margin, together with explicit disagreement coordinates. On the 4,080-example
    
[^96]: 意义的动态：僧伽罗语历时语义变化的评估研究

    Dynamics of meaning: Towards the Evaluation of Diachronic Semantic Change in Sinhala

    [https://arxiv.org/abs/2609.08609](https://arxiv.org/abs/2609.08609)

    本研究提出一种结合嵌入对齐与基于微调Llama-3.1-8B的双向语义影响剪枝方法的多阶段计算框架，用于评估僧伽罗语从13世纪至20世纪的历时语义变化，并能区分系统性语义演变与暂时性多义扩展。

    

    在资源匮乏的语言中追踪跨越漫长历史时期的语义变化面临重大挑战，原因在于数据稀缺以及静态嵌入对齐方法的局限性。本研究采用多阶段计算框架，考察了僧伽罗语从13世纪到20世纪的历时演变。我们首先使用基于相似性矩阵的对齐和正交普氏分析技术对齐特定世纪的Word2Vec和FastText嵌入，发现OP对齐在识别时间相似性低谷方面能提供更稳定的邻域追踪。为了超越聚合度量方法，我们提出了一种双向语义影响剪枝方法，该方法利用微调后的Llama-3.1-8B生成的上下文化嵌入。通过应用留一诊断法，我们尝试分离出有影响力的句子，以区分系统性语义演变与暂时性的多义扩展。

    arXiv:2609.08609v2 Announce Type: replace  Abstract: Tracking semantic change in low-resource languages across extensive historical timelines presents significant challenges due to data scarcity and the limitations of static embedding alignments. This study investigates the diachronic evolution of the Sinhala language from the 13th to the 20th century using a multi-stage computational framework. We first align century-specific Word2Vec and FastText embeddings using Similarity Matrix Based Alignment (SMA) and Orthogonal Procrustes (OP) techniques, finding that OP alignment provides more stable neighbourhood tracking for identifying temporal similarity dips. To move beyond aggregate measures, we introduce a Bidirectional Semantic Impact Pruning approach using contextualised embeddings from a fine-tuned Llama-3.1-8B. By applying Leave-One-Out (LOO) diagnostics, we attempt to isolate influential sentences to distinguish between systemic semantic shifts and transient polysemic expansion. Ou
    
[^97]: 逐词阅读法律问题：2,144条越南语法律标题的嵌入轨迹

    Reading a Legal Question Word by Word: Embedding Trajectories of 2,144 Vietnamese Legal Headlines

    [https://arxiv.org/abs/2609.08372](https://arxiv.org/abs/2609.08372)

    该研究通过逐词追踪越南语法律问题的嵌入轨迹，发现密集检索器在仅读取中位数6-7个实义词后、甚至在疑问句框架出现之前就能锁定正确法律条文并保持排名第一，且多问题标题中的后续子问题几乎不会改变已锁定的排名。

    

    密集检索器将一个问题编码为单个向量，但问题本身是逐词到达的。我们使用 Nemotron-3-Embed 8B/1B 和 Qwen3-Embedding 8B/0.6B 模型，逐词阅读了来自 Thu Vien Phap Luat（越南法律图书馆）的 2,144 条留出测试标题，针对 20,034 条法律条文编码了 65,444 个前缀，此外还编码了来自 1,112 条多问题标题的 3,438 个子问题的所有前缀，以及 168 个答案的所有前缀。研究发现： 答案条文在中位数 6-7 个实义词被读取后即跃升至排名第一，此时疑问句框架尚未被读取，并且在 78-85% 的情况下保持第一直到结尾。 在多问题标题中，锁定位置有 94-98% 的概率位于第一个子问题内部；第二个子问题在 89-95% 的情况下不改变已有排名；若单独编码，第二个子问题在相同锁定词处（95-97% 的情况锁定词相同）达到排名第一的比例仅为 42-58%，而第一个子问题为 91-96%。 数字、日期和法律文件标识符对嵌入的移动距离是实义词的两倍，是（原文在此处截断）……

    arXiv:2609.08372v2 Announce Type: replace  Abstract: A dense retriever encodes a question as one vector, but the question arrives one word at a time. We read 2,144 held-out headlines from Thu Vien Phap Luat (Vietnamese legal library) word by word with Nemotron-3-Embed 8B/1B and Qwen3-Embedding 8B/0.6B, encoding 65,444 prefixes against 20,034 articles, plus every prefix of 3,438 sub-questions from 1,112 multi-question headlines and of 168 answers. (i) The gold article becomes rank 1 after a median of 6-7 content words in every encoder, before the interrogative frame is read, and stays there to the end in 78-85% of cases. (ii) In a multi-question headline the lock is inside the first sub-question 94-98% of the time; the second leaves rank unchanged in 89-95%; encoded alone, the second reaches rank 1 in 42-58% vs 91-96% for the first, at the same lock word (95-97% identical). (iii) Numbers, dates and instrument identifiers move the embedding twice as far as content words and four times as
    
[^98]: HoneyRoute：面向对抗性LLM服务的蜜罐模型路由

    HoneyRoute: Honeypot-Model Routing for Adversarial LLM Serving

    [https://arxiv.org/abs/2609.08306](https://arxiv.org/abs/2609.08306)

    HoneyRoute是一个部署在推理服务层的防御框架，通过轻量级流式路由器实时检测恶意请求并将其路由至蜜罐模型，在保护生产LLM服务的同时，将捕获的攻击者交互转化为指纹数据用于持续改进路由器的检测能力。

    

    我们提出了HoneyRoute，一个推理服务层，它能够检测传入请求是否为恶意请求，若检测到恶意，则将其路由到专用的蜜罐模型，从而保护生产系统，同时持续收集对手交互信息以获取情报。现有的防御方法要么在模型记忆中嵌入陷阱，要么在协议层重建欺骗机制，这使得服务层缺乏保护，且无法将捕获的信息反馈用于检测。HoneyRoute结合了三个组件：(i) 一个流式路由器（采用冻结的0.8B参数嵌入骨干网络，配备各领域的MLP头），(ii) 双实现的蜜罐（规则/提示词工程化的代码蜜罐，或专用的同族模型副本），以及 (iii) 一个分析循环，将被捕获的交互转化为攻击者指纹，用于路由器的再训练。在生产环境追踪数据和七个领域的攻击语料库上，该路由器达到F1=.911，中位附加延迟仅38毫秒，以两级防护LLM级联系统1/385的延迟实现了其96%的F1分数。

    arXiv:2609.08306v2 Announce Type: replace-cross  Abstract: We introduce HoneyRoute, an inference-serving layer that detects whether an incoming request is malicious and, if so, routes it to a dedicated honeypot model, shielding production while the adversary's interaction is continuously harvested for intelligence. Existing defenses embed traps inside model memory or rebuild deception at the protocol layer, leaving the serving tier unprotected and feeding nothing back into detection. HoneyRoute couples (i) a streaming router (a frozen 0.8B-embedding backbone with per-domain MLP heads), (ii) a dual-implementation honeypot (a rule/prompt-engineered code honeypot or a dedicated same-family replica), and (iii) an analysis loop that converts trapped interactions into attacker fingerprints for router retraining. On a production trace plus a seven-domain attack corpus, the router reaches F1=.911 at 38 ms median added latency, matching 96% of a two-tier guard-LLM cascade's F1 at 1/385 of its l
    
[^99]: 少即是个性化：为个性化语言模型学习最小充分用户画像

    Less Is Personal: Learning Minimal Sufficient User Profiles for Personalized Language Models

    [https://arxiv.org/abs/2609.08180](https://arxiv.org/abs/2609.08180)

    提出ENOUGH方法，通过反事实搜索和多头价值控制器为每个输入自适应地构建长度可变的最小充分用户画像，在保持个性化效用的同时最小化token成本。

    

    检索增强个性化使大型语言模型能够利用从用户历史中检索到的相关记录，生成更准确且更符合用户偏好的输出。个性化语言模型通常在输入前固定添加一定数量的检索到的用户记录，即使额外的历史记录可能是冗余的、有害的、或与用户独特行为无关。我们研究了最小充分个性化问题：在保持从检索候选池中可获得的效用的同时，为每个输入构建成本最低的有序用户画像。我们提出了ENOUGH方法，该方法通过迭代地追加行为记录或发出STOP信号来构建具有自适应长度的用户画像。在离线阶段，有界反事实搜索通过综合考虑下游收益、用户特异性和token成本来评估用户画像的前缀。由此产生的长程目标被蒸馏到一个具有显式排序和停止监督的多头价值控制器中。

    arXiv:2609.08180v1 Announce Type: new  Abstract: Retrieval-augmented personalization enables large language models to produce more accurate and preference-aligned outputs using relevant records retrieved from user histories. Personalized language models typically prepend a fixed number of retrieved user records, even when additional history is redundant, harmful, or unrelated to a user's distinctive behavior. We study minimal sufficient personalization: constructing the least costly ordered profile for each input while preserving the utility achievable from a retrieved candidate pool. We introduce ENOUGH, a method that iteratively appends behavioral records or emits STOP to construct profiles with adaptive lengths. Offline, bounded counterfactual search evaluates profile prefixes by jointly considering downstream gains, user specificity, and token costs. The resulting long-horizon targets are distilled into a multi-head value controller with explicit ranking and stopping supervision. A
    
[^100]: Fine PT-PT Web：一个高质量的410亿词元欧洲葡萄牙语网页数据集

    Fine PT-PT Web: A High-Quality 41 Billion Tokens Data Collection of the European Portuguese Web

    [https://arxiv.org/abs/2609.07699](https://arxiv.org/abs/2609.07699)

    本文提出一个高效数据处理流水线，从411TB的原始网页数据中构建了高质量的410亿词元欧洲葡萄牙语语料库，其创新的后抓取预处理模块通过在过滤前去除样板文本和重复行，使最终文档产出量提升了19.04%。

    

    为欧洲葡萄牙语（PT-PT）等区域性语言变体策划网络语料库，严重受制于方言重叠（主要与巴西葡萄牙语PT-BR的重叠）和数据处理规模的瓶颈。本文提出了一个高效的流水线，从葡萄牙语网页中策划出一个可用于生产环境的PT-PT语料库，涵盖来自Arquivo.pt的411 TB原始数据。我们引入了一种新颖的抓取后处理模块，可在过滤之前去除样板文本和行级重复内容。这一早期干预通过挽救被标准启发式过滤器过早丢弃的有效文本，使最终文档产出量增加了19.04%。结合严格的语言识别、加权模糊去重和神经质量分类，我们的流水线提供了一个可扩展的框架以及一个面向大语言模型预训练优化的干净且具有代表性的语料库。

    arXiv:2609.07699v2 Announce Type: cross  Abstract: Curating Web corpora for regional language variants like European Portuguese (PT-PT) is heavily bottlenecked by dialectal overlap (mainly with PT-BR) and data processing scale. This paper presents an efficient pipeline to curate a production-ready PT-PT corpus from the Portuguese Web, spanning 411 TB of raw data from Arquivo.pt. We introduce a novel post-scraping block that removes boilerplate and line duplicates prior to filtering. This early-stage intervention increases final document yield by 19.04% by rescuing valid text that standard heuristic filters prematurely discard. Integrated with rigorous language identification, weighted fuzzy deduplication, and neural quality classification, our pipeline offers a scalable framework and a clean, representative corpus optimized for LLM pre-training.
    
[^101]: Qwen-Audio-3.0-ASR技术报告

    Qwen-Audio-3.0-ASR Technical Report

    [https://arxiv.org/abs/2609.07549](https://arxiv.org/abs/2609.07549)

    Qwen-Audio-3.0-ASR是基于混合专家大语言模型架构的语音识别系统，通过在数千万小时语音数据上训练，统一框架支持30种语言和16种汉语方言的转录，弥合了学术基准与实际生产应用之间的差距。

    

    近年来，自动语音识别（ASR）在三种互补范式的推动下取得了变革性进展：数据规模化、模型规模化以及与大语言模型（LLM）的深度融合。然而，如何弥合学术基准性能与实际生产应用之间的差距仍然是一个持续的挑战，特别是在处理多样化地区方言、动态实体和热词、长距离上下文信息以及不流畅的自发语音方面。在本报告中，我们提出了Qwen-Audio-3.0-ASR，这是一个基于混合专家架构（MoE）大语言模型的ASR系统，旨在通过统一、遵循指令的框架来满足这些生产需求。该模型构建于Qwen骨干网络之上，并在数千万小时的大规模语音数据上进行训练。Qwen-Audio-3.0-ASR支持30种语言以及覆盖八大方言区的16种汉语方言变体的转录。

    arXiv:2609.07549v2 Announce Type: replace  Abstract: In recent years, automatic speech recognition (ASR) has witnessed transformative advancements driven by three complementary paradigms: data scaling, model scaling, and deep integration with large language models (LLMs). However, bridging the gap between academic benchmark performance and real-world production utility remains a persistent challenge, particularly in handling diverse regional dialects, dynamic entities and hotwords, long-range contextual information, and disfluent spontaneous speech. In this report, we present Qwen-Audio-3.0-ASR, a Mixture-of-Experts (MoE) LLM-based ASR system designed to address these production demands through a unified, instruction-following framework. The model is built upon the Qwen backbone, and is trained on tens of millions of hours of large-scale speech data. Qwen-Audio-3.0-ASR supports transcription across 30 languages and 16 Chinese dialectal varieties spanning eight major dialect regions. Be
    
[^102]: 分解LLM评审器的不确定性以精准定位专家标注

    Decomposing LLM-Judge Uncertainty to Target Expert Labels

    [https://arxiv.org/abs/2609.06444](https://arxiv.org/abs/2609.06444)

    该论文提出一种小型贝叶斯模型，将LLM评审器的总不确定性分解为可被专家标注消除的认知不确定性和不可消除的偶然不确定性，使专家只需标注评审器真正无知的项目，在ChaosNLI数据集上比使用总不确定性多消除83%的误差。

    

    LLM评审器可以大规模评估模型输出，专家应该只在它最不确定的地方进行标注。然而其天然的升级信号混淆了两种不确定性：偶然不确定性——专家群体中真实存在的分歧，标注无法减少这种不确定性；以及认知不确定性——评审器自身的无知，标注可以减少这种不确定性。本文提出一个小型贝叶斯模型来分离这两种不确定性：通过对已收集的标注进行回归，学习在多大程度上信任黑盒评审器的预测。两个组成部分都可以通过简单的公式计算得出，无需采样或额外的评审器调用。在一个面对完全已知真值的真实LLM评审器上，这两种不确定性成分被成功分离，且评审器声称的置信度并不能反映其真实误差。在真实的人类分歧数据集（ChaosNLI）上，在相同的专家标注量下，基于认知不确定性的排序比使用总不确定性多消除83%的误差，不过在该数据集上简单地升级标注最少的项目也能达到同样效果。我们证明了我们可以估计评审器在哪里是无知的，而不是专家们真正存在分歧的地方。

    arXiv:2609.06444v2 Announce Type: replace  Abstract: An LLM judge evaluates outputs at scale. Experts should label only where it is least sure. Its natural escalation signal conflates two uncertainties: aleatoric, real disagreement in the expert pool, which labels cannot reduce, and epistemic, the judge's ignorance, which labels do reduce. A small Bayesian model separates them: a regression on labels already collected learns how far to trust a black-box judge's prediction. Both components follow as simple formulas, with no sampling or further judge calls. The components isolate on a real LLM judge against exactly known truth, and stated confidence is no guide to its actual error. On real human disagreement (ChaosNLI) the epistemic ranking removes 83% more error than total uncertainty for the same expert labels, though simply escalating the least-labelled items does as well there. We demonstrate we can estimate where a judge is ignorant rather than where experts genuinely disagree, and 
    
[^103]: 合作更佳：强RAG基线下的互补性查询改写

    Better Together: Complementary Query Rewriting Under a Strong RAG Baseline

    [https://arxiv.org/abs/2609.05637](https://arxiv.org/abs/2609.05637)

    在强RAG检索基线下，单一查询改写策略收效有限，但联合多种互补的改写方法可大幅提升检索性能，企业数据上HIT@10提升12.5个百分点。

    

    arXiv:2609.05637v2 公告类型：替换 摘要：改进检索增强生成（RAG）的一种流行方法是将用户问题改写为多个变体并使用所有变体进行搜索。我们测试了在底层检索系统已经很强大的情况下，这种方法是否真的有效。在一个固定的、具有竞争力的流程（BGE密集检索、交叉编码器重排序和MMR多样化）下，我们在三个数据集（HotpotQA、AmbigNQ以及包含51.2万文档的EnterpriseRAG-Bench）上，通过三个随机种子和配对自举显著性检验，将四种查询改写策略（S1-S4）与两个强大的LLM基线（HyDE、Query2Doc）进行了比较。我们的核心发现是：单独使用改写策略充其量只能与强大基线持平，但组合多种方法能带来显著的超额收益，因为不同策略在不同类型的问题上各有短板。四种方法的事后联合（S1+S3+S4+HyDE）在企业数据上将HIT@10指标比基线提升了12.5个百分点（51.70对39.22），而五种方法的联合更是达到了52.9。

    arXiv:2609.05637v2 Announce Type: replace  Abstract: A popular way to improve Retrieval-Augmented Generation (RAG) is to rewrite the user's question into several variants and search with all of them. We test whether this actually helps once the underlying search is already strong. Under one fixed, competitive pipeline (BGE dense retrieval, cross-encoder reranking, and MMR diversification), we compare four query-rewriting strategies (S1-S4) against two strong LLM baselines (HyDE, Query2Doc) on three datasets (HotpotQA, AmbigNQ, and the 512K-document EnterpriseRAG-Bench) over three seeds with paired-bootstrap significance tests. Our headline result is that rewriting alone is at best competitive with a strong baseline, but combining methods yields outsized gains because different strategies fail on different questions. A post-hoc union of four methods (S1+S3+S4+HyDE) improves HIT@10 over the baseline by +12.5 points on enterprise data (51.70 vs 39.22), and a five-method union reaches 52.9
    
[^104]: Harbor 适配器与 Harbor-Index：面向大规模智能体评估的基础设施与精选元数据集

    Harbor Adapters and Harbor-Index: Infrastructure and a Curated Meta-Dataset for Large-Scale Agentic Evaluation

    [https://arxiv.org/abs/2609.04298](https://arxiv.org/abs/2609.04298)

    本文提出了 Harbor Adapters 统一评估基础设施，将 80 多个智能体基准测试移植为可评估任意智能体的形式，并据此对 8 个模型进行大规模评估，同时推出经 AI 与人工双重审核筛选的包含 82 个高质量难题的精选元数据集 Harbor-Index。

    

    在数量不断增长的智能体（agentic）基准测试上评估智能体是一项挑战，因为这些基准通常需要复杂的环境和智能体集成方案。我们提出了 Harbor Adapters，一个面向智能体基准测试的统一评估基础设施。我们的工作包含三项贡献。第一，我们开发了一系列基准适配器，将 80 多个基准测试移植为可评估任意智能体的形式，并通过严格的代码审查和一致性实验对其进行了验证。第二，我们在 54 个基准测试上对横跨不同能力层级的 8 个模型进行了大规模评估；每个模型均使用 Terminus-2 以及 3 个原生测试框架之一运行。这使得对智能体能力和失败模式的更广泛分析成为可能。第三，我们推出了 Harbor-Index，一个精心策划的包含 82 个高难度、多样化且高质量任务的集合，覆盖 29 个基准测试，它是在适配后的基准套件基础上，通过难度筛选、AI 与人工审核以及审核-修复循环精炼而成。

    arXiv:2609.04298v1 Announce Type: new  Abstract: Evaluating agents on the growing number of agentic benchmarks is challenging because they often require complex environments and agent integrations. We introduce Harbor Adapters, a unified evaluation infrastructure for agentic benchmarks. Our work makes three contributions. First, we develop benchmark adapters that port more than 80 benchmarks to evaluate arbitrary agents, and validate them through rigorous code review and parity experiments. Second, we conduct a large-scale evaluation of 8 models spanning capability tiers across 54 benchmarks; every model is run with Terminus-2 and with one of 3 native harnesses. This enables a broader analysis of agent capabilities and failure modes than was previously possible. Third, we introduce Harbor-Index, a curated set of 82 difficult, diverse, and high-quality tasks spanning 29 benchmarks, refined from the adapted suite through difficulty filtering, AI and human audit, and an audit-and-fix loop
    
[^105]: VestigeKV：NoPE-MLA的KV缓存通过一个残余分支携带其自身的淘汰信号

    VestigeKV: The NoPE-MLA KV Cache Carries Its Own Eviction Signal in a Vestigial Branch

    [https://arxiv.org/abs/2609.03949](https://arxiv.org/abs/2609.03949)

    VestigeKV发现NoPE MLA模型KV缓存中的64维解耦RoPE残余分支已被训练重新利用为显著性信号，据此提出无需训练和量化的查询无关缓存淘汰方法，在8-32倍压缩下几乎不损失检索精度。

    

    问题所在：一个长期存在的KV缓存必须在将要读取它的查询出现之前就被压缩；基于已观测注意力进行选择的方法（H2O、SnapKV）在这种情况下会失效（在NoPE MLA模型上针检索率仅为0.00-0.33），因为token的重要性尚未被观测到。方法：在Kimi Linear模型上，VestigeKV利用缓存自身已携带的与查询无关的信号进行淘汰：即64维解耦分支——这是RoPE的残余结构，NoPE训练将其重新利用为显著性通道。只需读取每行的11%，它将缓存划分为两个层级：top-m行保留在注意力层；其余所有行——精确保存、从不删除——移动到GPU驻留的存档中，每一步都可以通过经过验证的触发器访问。无需训练、无需量化、无需更改权重或内核。代价：无可测量的损失：在8k到65k上下文长度范围内，8倍压缩下检索率保持在1.00，32倍压缩下为0.92，与全行选择方法零差距。注意力层级仅占Kimi Linear每token 8.1 KB缓存中的0.25 KB。

    arXiv:2609.03949v1 Announce Type: cross  Abstract: The problem. A long-lived KV cache must be compressed before the queries that will read it exist; selection by observed attention (H2O, SnapKV) collapses there (0.00-0.33 needle retrieval on a NoPE MLA model), because a token's importance has not yet been observed. The method. On Kimi Linear, VestigeKV evicts by a query-independent signal the cache already carries: the 64-dimensional decoupled branch, a vestige of RoPE that NoPE training repurposes into a salience channel. Reading 11% of each row, it partitions the cache: the top-m rows stay in the attended tier; every other row moves -- exactly, never deleted -- to a GPU-resident archive reachable per step by a certified trigger. No training, no quantization, no weight or kernel change. Cost. Nothing measurable: retrieval holds at 1.00 under 8x and 0.92 under 32x from 8k to 65k context, zero gap to full-row selection. The attended tier is 0.25 KB of Kimi Linear's 8.1 KB per-token cach
    
[^106]: 用于转变预测的相位结构特征的两项锁定测试

    Two locked tests of phase-structure features for transition prediction

    [https://arxiv.org/abs/2609.00335](https://arxiv.org/abs/2609.00335)

    该论文通过两项预先锁定的实证测试检验相位结构特征PC-2能否在基线之上改进对承诺或矛盾终点的预测，结果两项测试均未通过推进标准，官方结论为阴性。

    

    一项已发表的关于旋转注意中相位结构的理论说明，接受了两项预先设定的实证检验，以检验由相位导出的特征是否优于未接收这些特征的基线，从而更好地预测承诺或矛盾终点。研究1冻结了一个矛盾类别流水线，并对PC-2与基线的密封主要比较进行评分。在1,136个符合条件的案例中，配对AUROC差异为+0.00087。99%置信区间包含零，且差异未达到预先设定的+0.05阈值，未通过推进标准。研究2仅在开放块b0-b4上开发了十五种层处理（1,415个转变，20×5分组折）。锁定的合取规则要求PC-2平均重复增量为正、五个种子块中至少四个增量为正、以及这五个差异的平均值为正。没有任何处理通过推进标准。官方选择结果为阴性（无效）。

    arXiv:2609.00335v1 Announce Type: new  Abstract: A published theoretical account of phase structure in rotary attention was subjected to two pre-specified empirical tests of whether phase-derived features improve prediction of a commitment or contradiction endpoint over a baseline that does not receive those features. Study 1 froze a contradiction-category pipeline and scored a sealed primary comparison of PC-2 against baseline. On 1,136 eligible cases the paired AUROC difference was +0.00087. The 99% interval included zero, and the difference did not reach the pre-specified threshold of +0.05. Advancement was not passed. Study 2 developed fifteen layer treatments on open blocks b0-b4 only (1,415 transitions, 20x5 grouped folds). A locked conjunctive rule required a positive PC-2 mean-repeat increment, a positive increment on at least four of five seed blocks, and a positive mean of those five differences. No treatment advanced. The official selection is null. The theoretical paper is 
    
[^107]: AtlasNLP：NLP数据集表示的国家感知图谱

    AtlasNLP: A Country-Aware Atlas of Dataset Representation in NLP

    [https://arxiv.org/abs/2608.30107](https://arxiv.org/abs/2608.30107)

    AtlasNLP是一个国家感知的NLP数据集图谱，收录了超过13,000条数据集记录，揭示了数据集覆盖在国家与任务间高度不均衡、数据集生产与代表性在地理上不对称，以及语言覆盖并不等同于地理代表性等关键问题。

    

    了解NLP数据集中代表了哪些国家，对于识别差距、有针对性地进行数据收集、衡量进展以及为AI政策提供依据至关重要。然而，地理元数据非常罕见，国家层面的代表性往往隐藏在宽泛的语言层面声明之后。我们提出了AtlasNLP，这是一个国家感知的图谱，包含超过13,000条NLP数据集记录，涵盖规范化的NLP任务类别，同时追踪所代表的人群以及数据集的产地。AtlasNLP包括AtlasNLP-Gold（一个人工策划的参考集）和AtlasNLP-Core（一个源自ACL的大规模数据集合）。利用这一资源，我们证明了：（1）数据集覆盖在国家之间和任务之间高度不均衡；（2）数据集的生产与代表性在地理上是不对称的；（3）语言覆盖并不意味着地理代表性。这些发现揭示了当前数据集文档记录实践中的盲点……

    arXiv:2608.30107v1 Announce Type: cross  Abstract: Understanding which countries are represented in NLP datasets is essential for identifying gaps, targeting data collection, measuring progress, and informing AI policy. However, geographic metadata is very rarely available, and country-level representation is often hidden behind broad language-level claims. We introduce AtlasNLP, a country-aware atlas of over 13,000 NLP dataset records across normalized NLP task categories, tracking both the populations represented and where datasets are produced. AtlasNLP includes AtlasNLP-Gold, a human-curated reference set, and AtlasNLP-Core, an ACL-derived large-scale collection. Using this resource, we show that (1) dataset coverage is highly uneven across countries and tasks; (2) dataset production and representation are geographically asymmetric; and (3) language coverage does not imply geographic representation. These findings reveal blind spots in current dataset documentation practices and mo
    
[^108]: 基于音素的TTS增强用于ASR的扩展：统一流程与受控研究

    Scaling phoneme-based TTS augmentation for ASR: A unified pipeline and controlled study

    [https://arxiv.org/abs/2608.26697](https://arxiv.org/abs/2608.26697)

    本文提出了一种基于音素的统一TTS到ASR增强流程，并引入音素频率引导选择（PFGS）方法，在多种语言的ASR任务中有效提升了性能。

    

    合成语音为自动语音识别（ASR）提供了可扩展的监督信号，但其效果取决于所选的文本、参考语音和合成数据量。我们提出了一种统一的基于音素的TTS到ASR增强流程，该流程围绕一个使用F5-TTS架构从头训练并带有语言ID条件化的多语言TTS模型构建。该流程结合了特定语言的音素转换、参考语音过滤、候选文本选择、合成和匹配的ASR续训练。我们进一步提出了音素频率引导选择（PFGS），该方法利用从真实ASR训练标签中估计的音素频率对候选句子进行排序。针对阿拉伯语、法语、意大利语和葡萄牙语的独立单语ASR系统的实验覆盖了13个测试集。在合成规模扫描中，随机增强在11个测试集上优于仅匹配真实数据的续训练。在标称60%合成比例下，该方法进一步提升了性能。

    arXiv:2608.26697v1 Announce Type: new  Abstract: Synthetic speech provides scalable supervision for automatic speech recognition (ASR), but its benefit depends on the selected texts, reference speech, and amount of synthesized data. We present a unified phoneme-based TTS-to-ASR augmentation pipeline built around a multilingual TTS model trained from scratch using the F5-TTS architecture with language-ID conditioning. The pipeline combines language-specific grapheme-to-phoneme conversion, reference-speech filtering, candidate-text selection, synthesis, and matched ASR continuation. We further propose phoneme-frequency-guided selection (PFGS), which ranks candidate sentences using phoneme frequencies estimated from real ASR training labels. Experiments with separate monolingual ASR systems for Arabic, French, Italian, and Portuguese span 13 test sets. Across the synthesis-scale sweep, random augmentation improves over matched real-only continuation on 11 test sets. Under a nominal 60% sy
    
[^109]: 利用言语行为进行低数据和跨领域对话脱轨预测

    Leveraging Speech Acts for Low-Data and Cross-Domain Conversation Derailment Forecasting

    [https://arxiv.org/abs/2608.25359](https://arxiv.org/abs/2608.25359)

    本文提出利用言语行为作为辅助信号，结合文本语义进行对话脱轨预测，显著提升了低数据和跨领域场景下的性能。

    

    对话脱轨预测旨在预测在线讨论何时会升级为敌对行为，从而实现主动调节。现有方法在低数据设置下往往表现不佳，且难以跨领域泛化。这对新平台和标注数据有限的小型社区构成了挑战。我们提出对对话的语用表示进行建模，以减少词汇噪声并提高泛化能力。具体而言，言语行为信息作为辅助学习信号与文本语义一起使用。实验结果表明，在三个数据集上性能均有提升，尤其是在低数据和跨领域设置中。

    arXiv:2608.25359v1 Announce Type: new  Abstract: Conversational derailment forecasting aims to predict when online discussions will escalate into hostility, enabling proactive moderation. Existing approaches often struggle in low-data settings and to generalize across domains. This poses a challenge for new platforms and smaller communities where annotated data is limited. We propose modeling pragmatic representations of conversations to reduce lexical noise and improve generalizability. Specifically, speech act information is used as an auxiliary learning signal alongside textual semantics. Experimental results show improved performance across three datasets, particularly in low-data and cross-domain settings.
    
[^110]: 前沿挑战：评估科学工作流完成度

    FrontierChallenge: Evaluating Scientific Workflow Completion

    [https://arxiv.org/abs/2608.24979](https://arxiv.org/abs/2608.24979)

    本文介绍了FrontierChallenge基准测试，用于评估科学智能体在跨领域端到端工作流中的完成能力，发现当前最佳模型仅能完成20.6%的任务，表明部分进展难以转化为完整交付物。

    

    arXiv:2608.24979v1 公告类型：交叉 摘要：科学智能体日益用于分析数据、执行代码并生成研究产物，然而大多数基准测试强调最终答案、孤立程序或单一领域。我们引入了FrontierChallenge，一个跨领域基准测试，包含300个端到端科学工作流。在本文中，我们发布并评估了其中97个任务，涵盖量子化学、分子动力学、材料表征、分析化学、生命科学以及电化学/环境领域。每个任务提供固定输入，并指定所需科学交付物的集合。我们评估了十二个前沿模型和三种智能体脚手架。通过率衡量满足完全完成标准的任务比例，而平均得分捕捉部分进展。每个最佳配置仅完成了97个已发布任务中的20个，通过率为20.6%。部分进展尤其难以转化为完整的交付物。

    arXiv:2608.24979v1 Announce Type: cross  Abstract: Scientific agents increasingly analyze data, execute code, and produce research artifacts, yet most benchmarks emphasize final answers, isolated programs, or a single domain. We introduce FrontierChallenge, a cross-domain benchmark comprising 300 end-to-end scientific workflows. In this paper, we release and evaluate 97 of these tasks, spanning quantum chemistry, molecular dynamics, materials characterization, analytical chemistry, life science, and electrochemistry/environment. Each task provides fixed inputs and specifies a bundle of required scientific deliverables. We evaluate twelve frontier models with three agent scaffolds. Pass Rate measures the fraction of tasks satisfying the full-completion criterion, while Avg. Score captures partial progress. Each of the best-performing configurations completed only 20 of the 97 released tasks, yielding a Pass Rate of 20.6%. Partial progress translated especially poorly into complete deliv
    
[^111]: 《“盖布”翻译即“隐形伤害”：通过“乌尔都语遗漏”评分衡量大语言模型仇恨言论检测中的跨文字安全性不一致性》

    'Ghaib in Translation' aka Unseen Harm: Measuring Cross-Script Safety Inconsistency with 'Missed-in-Urdu' Scores in LLM Hate Speech Detection

    [https://arxiv.org/abs/2608.24191](https://arxiv.org/abs/2608.24191)

    本研究首次系统揭示了大语言模型在乌尔都语与英语翻译间存在显著的安全检测不一致性，乌尔都语原始文字内容常被错误地视为正常，导致有害内容漏检。

    

    乌尔都语作为全球第十大语言，拥有2.46亿使用者，但在主流大语言模型安全评估以及九届WOAH会议论文中几乎完全缺席。为调查这一缺席是否对内容审核可靠性产生可衡量的影响，研究测试了五个大型语言模型（GPT-4o、Claude Sonnet 4.5、Gemini 2.5 Flash、Qwen-2.5和Llama-3.1），覆盖六个数据集，包括Nastaliq乌尔都语、罗马乌尔都语、英语以及乌尔都语-英语代码混合语。在五个乌尔都语文字数据集中，原始文字与英语翻译分类之间的标签不稳定性从15.9%（Gemini 2.5 Flash）到31.6%（Qwen-2.5）不等，其中“乌尔都语遗漏”率（即内容在英语翻译中被标记为有害，但在原始文字中被视为正常）范围为2.4%至9.9%（中位数为4.3%）。通过ACL Anthology API对九届ALW/WOAH版本共205篇论文的完整枚举确认，没有任何专门针对乌尔都语的研究。

    arXiv:2608.24191v1 Announce Type: cross  Abstract: Urdu, the world's tenth most spoken language with 246 million speakers, remains almost entirely absent from mainstream LLM safety evaluation and nine years of WOAH proceedings. To investigate whether this absence has measurable consequences for content moderation reliability, five large language models, GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Flash, Qwen-2.5, and Llama-3.1, were tested across six datasets spanning Nastaliq Urdu, Roman Urdu, English, and code-switched Urdu-English. Across the five Urdu-script datasets, label instability between original-script and English-translation classification ranged from 15.9% (Gemini 2.5 Flash) to 31.6% (Qwen-2.5), with a 'Missed-in-Urdu' rate, content flagged as harmful in English translation but passed as normal in the original script, ranging from 2.4% to 9.9% (median 4.3%). A complete enumeration of all 205 papers across nine ALW/WOAH editions via the ACL Anthology API confirms zero dedicated U
    
[^112]: TokEval：一种分词器评估套件

    TokEval: A Tokenizer Evaluation Suite

    [https://arxiv.org/abs/2608.18062](https://arxiv.org/abs/2608.18062)

    本文提出TokEval，一个超越传统指标的分词器评估框架，通过引入语言和结构属性（如UTF-8边界和数字对齐）来预测下游模型性能，并通过受控实验验证其有效性。

    

    arXiv:2608.18062v1 公告类型：新公告 摘要：语言模型分词器通常在评估最少的情况下被选择，尽管其设计选择直接影响模型能力。这在一定程度上可归因于对分词器属性如何影响下游性能的哪些方面理解有限。我们引入了TokEval，一个分词器评估指标框架，它超越了生育率和压缩率等标准度量，以捕捉语言和结构上有意义的属性，例如UTF-8字符边界完整性和数学中的数字位值边界对齐。为了验证这些指标是否能预测下游模型性能，我们进行了受控的语言模型预训练实验，仅改变分词器的训练数据混合、预分词策略和训练算法。我们在每字节比特数（一种与分词器无关的困惑度版本）和几个基准测试上评估了生成的模型。

    arXiv:2608.18062v1 Announce Type: new  Abstract: Language model tokenizers are typically selected with minimal evaluation, despite the fact that their design choices directly impact model capabilities. This can be partly attributed to a limited understanding of which tokenizer properties affect which aspects of downstream performance. We introduce TokEval, a framework of tokenizer evaluation metrics that goes beyond standard measures like fertility and compression rate to capture linguistically and structurally meaningful properties, e.g., UTF-8 character boundary integrity and digit place-value boundary alignment for mathematics. To validate whether these metrics are predictive of downstream model performance, we conduct controlled language model pretraining experiments, varying solely the tokenizers' training data mixture, pretokenization strategy, and training algorithm. We evaluate the resulting models on bits-per-byte (a tokenizer-agnostic version of perplexity) and several benchm
    
[^113]: Palmyra x6技术报告：通过锚定监督微调后训练的代理型工具使用模型

    Palmyra x6 Technical Report: An Agentic, Tool-Use Model Post-Trained via Anchored Supervised Fine-Tuning

    [https://arxiv.org/abs/2608.16620](https://arxiv.org/abs/2608.16620)

    Palmyra x6通过锚定监督微调和保守训练策略，在少量数据上实现了企业代理任务中的显著性能提升，并在多个基准测试中领先。

    

    arXiv:2608.16620v1 公告类型：交叉 摘要：Palmyra x6是一个针对企业导向代理任务优化的大型语言模型。该模型通过在紧凑的已验证合成工具使用轨迹语料库上，对混合专家基础模型进行锚定监督微调，并使用Muon + Adam混合优化器进行后训练构建而成。该配方刻意保守且受控：626条轨迹、单轮训练、低学习率，以及一个冻结基础的KL锚定。该模型在Writer Agent任务上相比之前的默认模型显示出显著提升，并在公开基准测试中与多个近期模型相比表现优异，在BFCL Core上得分最高，为0.785，并取得了该组六个基准测试的最高平均值。此外，在我们的偏见和安全评估中，该模型相对于比较对象表现出竞争力或领先性。

    arXiv:2608.16620v1 Announce Type: cross  Abstract: Palmyra x6 is a large language model optimized for use with enterprise-oriented agentic tasks. The model was built by post-training a Mixture-of-Experts base model with Anchored Supervised Fine-Tuning on a compact corpus of verified, synthetic tool-use trajectories, optimized with a Muon + Adam hybrid. The recipe is deliberately conservative and deliberately controlled: 626 trajectories, a single epoch, a low learning rate, and a KL anchor to the frozen base. The model shows substantial gains over the previous default model for Writer Agent, and compares favorably with several recent models on public benchmarks, scoring the highest on BFCL Core at $0.785$ and posts the highest six-benchmark mean of the cohort. Furthermore, the model has shown itself to be competitive or leading relative to comparators in our bias and safety evaluations.
    
[^114]: 左分支变压器在右分支语言中表现优异：数据塑造语言模型中的词序偏好

    Left-Branching Transformers Excel at Right-Branching Languages: Data Shapes Word Order Preferences in Language Models

    [https://arxiv.org/abs/2608.15129](https://arxiv.org/abs/2608.15129)

    这项研究发现语言模型的词序偏好并非固有，而是由训练数据驱动，表现为在自然语言中偏向SVO（主-动-宾）结构，在人工语言中则偏向左分支结构。

    

    arXiv:2608.15129v1 公告类型：交叉 摘要：我们系统地比较了仅解码器语言模型在192种人工语言和类型多样的自然语言中的词序偏好。在人工语言上，模型表现出左分支偏好，这既不符合自然语言普遍性，也不符合人类词序学习偏差。在自然语言上，单语模型在较小规模下没有明显的基准词序偏差，但随着数据增长，对右分支的主-动-宾（SVO）语言的偏好出现，而SOV（主-宾-动）虽然跨语言中是最常见的词序，却落后了。这种SVO优势扩展到多语言模型，并与语言资源水平和数据质量相关，而非词序本身。因此，同一架构在人工和自然语言上表现出相反的偏好，确立了实践中观察到的词序偏差是数据驱动的。由于高资源语言绝大多数是SVO，

    arXiv:2608.15129v1 Announce Type: cross  Abstract: We systematically compare word order preferences in decoder-only language models across 192 artificial languages and typologically diverse natural languages. On artificial languages, models exhibit a left-branching preference that aligns with neither natural language universals nor human word order learning biases. On natural languages, monolingual models show no clear base word order bias at small scales, but as data grows, a preference for right-branching subject-verb-object (SVO) languages emerges while SOV falls behind despite being the most frequent order cross-linguistically. This SVO advantage extends to multilingual models and correlates with language resource level and data quality rather than word order. Thus, the same architecture exhibits opposite preferences on artificial and natural languages, establishing that word order biases observed in practice are data-driven. Since highly-resourced languages are overwhelmingly SVO,
    
[^115]: DexterSQL：面向文本到SQL生成的深度模式探索与基于规则的修正

    DexterSQL: Deep Schema Exploration and Rule-based Correction for Text-to-SQL Generation

    [https://arxiv.org/abs/2608.11889](https://arxiv.org/abs/2608.11889)

    DexterSQL通过深度模式探索、数据库无关规则挖掘和规则驱动修正三个创新组件，解决了非微调文本到SQL生成中模式信息粗糙、错误重复出现和条件处理不当的问题。

    

    arXiv:2608.11889v1 公告类型：交叉 摘要：基于提示（即非微调）的文本到SQL方法，其中底层大语言模型参数不针对任务进行更改，面临三个问题：（i）依赖粗粒度的模式信息，这可能无法揭示区分模糊列所需的细粒度关系，（ii）未能捕捉重复出现的SQL生成失败，以及（iii）在复杂问题中遭受条件遗漏、幻觉或错位。本文开发了DexterSQL，一个基于提示/非微调的文本到SQL系统，通过三个新组件改进SQL生成：（i）深度模式探索器，识别模糊列，分析其单独和联合数据分布以揭示它们之间的关系及各自的不同作用，（ii）数据库无关的规则创建器，挖掘生成结果与目标结果之间的不匹配，（iii）规则驱动的修正器，应用这些规则来纠正SQL生成中的常见错误。

    arXiv:2608.11889v1 Announce Type: cross  Abstract: Prompting-based (\textit{i}.\textit{e}., non-fine-tuning) Text-to-SQL methods, where underlying large language model parameters are not changed for the task, face three problems: (\textit{i})~relying on coarse-grained schema information that may not reveal the fine-grained relationships needed to distinguish ambiguous columns, (\textit{ii})~not capturing recurring SQL-generation failures, and (\textit{iii})~suffering from omission, hallucination, or misplacement of conditions in complex questions.   This paper develops \textsc{DexterSQL}, a prompting/non-fine-tuning-based Text-to-SQL system that improves SQL generation with three novel components: (\textit{i})~\emph{deep schema explorator} that identifies ambiguous columns, analyzes their individual and joint data distributions to uncover their relationships and the distinct role of each, (\textit{ii})~\emph{database-agnostic rule creator} that mines mismatches between generated and go
    
[^116]: 大语言模型是位置一致的序数分类器吗？一项系统性评估

    Are LLMs Positionally Consistent Ordinal Classifiers? A Systematic Evaluation

    [https://arxiv.org/abs/2608.08869](https://arxiv.org/abs/2608.08869)

    该研究系统评估发现，所有前沿大语言模型在序数分类中都普遍存在由标签顺序、示例顺序和示例位置引起的位置偏差，且现有去偏方法无法可靠修复，仅基于比较的逐列表推断方式表现相对最佳。

    

    大语言模型越来越多地被用于序数分类任务，然而对提示组织的语义等价改动却可能改变模型的预测结果。我们开展了系统性实验，以刻画源自标签顺序、示例顺序和示例放置位置的位置偏差。首先，我们在一个常见的序数分类任务上将这三种探测方法应用于十个前沿大语言模型；每个模型对所有三种位置来源均表现出敏感性，说明该问题普遍存在。其次，我们在五个数据集上变化八个提示级、任务级和模型级因素；准确性与稳定性往往不一致，只有更低的量表基数（选项数量）能持续改善两者。第三，我们比较了逐点、逐对和逐列表推断方式、替代的聚合与去偏方法以及联合配置；所测试的纠偏方法均无法提供可靠的补救，而基于比较的逐列表表述提供了最佳平衡，但…（原文摘要在此截断）

    arXiv:2608.08869v2 Announce Type: replace  Abstract: Large language models are increasingly used for ordinal classification, yet semantically equivalent changes to prompt organization can alter their predictions. We conduct systematic experiments to characterize positional bias from label order, demonstration order, and demonstration placement. First, we apply the three probes to ten frontier LLMs on a common ordinal-classification task; every model is sensitive to all three positional sources, showing that the problem is pervasive. Second, we vary eight prompt-, task-, and model-level factors across five datasets; accuracy and stability are often misaligned, and only lower scale cardinality consistently improves both. Third, we compare pointwise, pairwise, and listwise inference, alternative aggregation and debiasing methods, and joint configurations; the tested corrections do not provide a reliable remedy, while a comparison-based listwise formulation offers the best balance but tran
    
[^117]: 揭秘大型推理模型中基于熵的思维链压缩选择方法

    Demystifying Entropy-based Selection for Chain-of-Thought Compression in Large Reasoning Models

    [https://arxiv.org/abs/2607.28707](https://arxiv.org/abs/2607.28707)

    本文系统性地证明基于熵的CoT压缩选择方法相比随机剪枝并无优势，并通过激活修补实验提供因果证据表明任务信息分布在整个推理链上，而非集中在少数可用启发式规则识别的关键词元中。

    

    基于熵的剪枝方法被提出作为压缩思维链（CoT）推理的有效手段，且准确率损失可忽略不计。我们在多种模型和推理任务上测试了低熵和高熵CoT步骤选择方法的鲁棒性，结果表明在所有评估设置中，熵方法相比随机剪枝均无任何优势。随后从句子层面转向词元层面，我们发现保留低熵词元似乎仅在数学基准测试上有效。我们发现这是由于数字词元本质上具有低熵特性，且在这类问题中数字词元同样承载着语义内容。最后，我们证明仅需用原始激活值修补少量CoT词元的子集即可恢复接近完美的完整轨迹性能，这提供了因果证据表明任务信息并非集中在可通过启发式方法识别的少量CoT词元中，而是分布在整个推理链上。

    arXiv:2607.28707v3 Announce Type: replace  Abstract: Entropy-based pruning has been proposed as an effective method for compressing Chain-of-Thought (CoT) reasoning with negligible accuracy loss. We test the robustness of low- and high-entropy CoT step selection methods across various models and reasoning tasks, showing that entropy offers no advantage over random pruning in any evaluated setting. Moving from sentences to tokens, we then show that retaining low-entropy tokens seems effective only on mathematical benchmarks. We find this is due to the inherently low-entropy nature of numeric tokens, which also convey semantic content in such problems. Finally, we demonstrate that patching a subset of a few CoT tokens with their original activations recovers near-perfect full-trace performance, providing causal evidence that task information is not concentrated in a small set of CoT tokens identifiable by heuristics, but rather distributed across the full reasoning chain.
    
[^118]: 旋转注意力中的相位结构：一个用于语义连续性与执行边界治理的谱框架

    Phase Structure in Rotary Attention: A Spectral Framework for Semantic Continuity and Execution-Boundary Governance

    [https://arxiv.org/abs/2607.25507](https://arxiv.org/abs/2607.25507)

    该论文提出了一个有界谱分析框架，将旋转位置编码（RoPE）的注意力得分分解为幅度加权余弦项之和，并证明了一致有界的相位位移可限制pre-softmax得分退化的局部稳定性引理，从而为语义连续性分析和执行边界治理提供了非物理化的理论基础。

    

    Transformer语言模型通常通过向量几何来分析，然而有序上下文和旋转位置编码在查询-键交互中引入了显式的相位结构。本文开发了一个有界谱分析框架，用于研究旋转相位对齐、隐状态连续性和语义漂移，而无需将语言模型视为字面意义上的物理波动系统。本文首先确定有序隐状态序列（而非词表索引）是谱分解的有效定义域。随后，本文将旋转位置编码的注意力得分推导为幅度加权余弦项之和，并证明了一个局部稳定性引理：一致有界的相位位移可限制相应pre-softmax得分的退化。为了将相位分析扩展到原生RoPE坐标之外，本文在固定正交归一方向对上定义了复模态坐标，并引入了加权相干泛函（摘要原文在此处被截断）

    arXiv:2607.25507v2 Announce Type: replace  Abstract: Transformer language models are usually analyzed through vector geometry, yet ordered context and rotary position encoding introduce explicit phase structure into query-key interactions. This paper develops a bounded spectral framework for examining rotary phase alignment, hidden-state continuity, and semantic drift without treating language models as literal physical wave systems. It first identifies ordered hidden-state sequences, rather than vocabulary indices, as valid domains for spectral decomposition. It then derives the Rotary Position Embedding (RoPE) attention score as a sum of magnitude-weighted cosine terms and proves a local stability lemma: uniformly bounded phase displacement limits degradation of the corresponding pre-softmax score. To extend phase analysis beyond native RoPE coordinates, the paper defines complex modal coordinates over fixed orthonormal direction pairs and introduces a weighted coherence functional f
    
[^119]: TreeThink：一个用于大语言模型数学推理的模块化树搜索库

    TreeThink: A Modular Tree Search Library for Mathematical Reasoning with LLMs

    [https://arxiv.org/abs/2607.11258](https://arxiv.org/abs/2607.11258)

    TreeThink是一个开源的模块化、完全异步树搜索Python库，它将树搜索方法与vLLM推理流水线及多样化节点评估技术相集成，并支持Lean 4、Rocq、Isabelle/HOL和自然语言的实时验证，填补了大语言模型树搜索与形式化定理证明系统之间的空白。

    

    树搜索算法能够在神经定理证明中对证明空间进行系统性探索。现有的大语言模型树搜索库主要面向自然语言推理，无法与形式化验证器原生集成，而定理证明系统往往依赖于特定任务的搜索实现。我们推出了TreeThink，这是一个开源Python库，用于神经定理证明中模块化、完全异步的树搜索。它将成熟的树搜索方法与基于vLLM的推理流水线以及多样化的节点评估技术相集成，评估方法涵盖从轻量级启发式方法到神经评估器。我们支持Lean 4、Rocq和Isabelle/HOL以及自然语言。它直接连接到每种语言的读取-求值-打印循环（REPL）服务器，以实现实时验证和证明状态提取。我们在miniF2F和MATH500上对TreeThink进行了评估，展示了跨语言形式化证明搜索和自然语言推理的能力。

    arXiv:2607.11258v2 Announce Type: replace  Abstract: Tree search algorithms enable systematic exploration of the proof space in neural theorem proving. Existing LLM tree search libraries primarily target natural language reasoning and do not provide native integration with formal verifiers, while theorem proving systems often rely on task-specific search implementations. We introduce TreeThink, an open-source Python library for modular, fully asynchronous tree search in neural theorem proving. It integrates established tree search methods with vLLM-based inference pipelines and diverse node evaluation techniques, ranging from lightweight heuristics to neural evaluators. We support Lean~4, Rocq, and Isabelle/HOL alongside natural language. It connects directly to each language's Read-Eval-Print Loop (REPL) server for real-time verification and proof state extraction. We evaluate TreeThink on miniF2F and MATH500, demonstrating cross-language formal proof search, natural language reasonin
    
[^120]: MultiSynt/MT：跨36种语言翻译的万亿词元多并行预训练数据

    MultiSynt/MT: Trillion-Token Multi-Parallel Pre-Training Data Translated Across 36 Languages

    [https://arxiv.org/abs/2607.00890](https://arxiv.org/abs/2607.00890)

    本文提出了MultiSynt/MT，一个涵盖36种语言、约4.8万亿词元的开放合成并行预训练语料库，使模型用约72%更少的训练词元即可达到原生数据基线性能并在同等预算下超越其约15%，同时为多种中低资源欧洲语言提供了最大的公开预训练资源。

    

    开放的网页规模预训练语料库仍然集中于英语，这限制了多语言大语言模型的发展。我们推出了MultiSynt/MT，这是一个开放的合成并行语料库，涵盖36种语言、约4.8万亿目标语言词元，它通过使用Tower+和OPUS-MT/HPLT-MT系统翻译1000亿高质量的Nemotron-CC词元生成。对于许多中等资源和较低资源的欧洲语言而言，这是目前最大的公开可用预训练资源。在五种高资源和中等资源语言上的实验表明，使用MultiSynt/MT训练的参考LLM仅用约72%更少的预训练词元即可达到HPLT 2.0（原生数据基线）的最终分数，并且在相同的1000亿词元训练预算下相对超越该基线约15%。我们的分析还揭示了评估中的盲点：标准的多项选择基准测试无法捕捉翻译质量的差异，而一种对流畅性敏感的LLM-as-judge评估协议则能够在训练后的模型上恢复这些差异。

    arXiv:2607.00890v2 Announce Type: replace  Abstract: Open web-scale pre-training corpora remain concentrated in English, limiting multilingual LLM development. We introduce MultiSynt/MT, an open synthetic parallel corpus with approximately 4.8 trillion target-language tokens across 36 languages, produced by translating 100 billion high-quality Nemotron-CC tokens with Tower+ and OPUS-MT/HPLT-MT systems. For many medium- and lower-resource European languages, this is the largest openly available pre-training resource. Across five high- and medium-resource languages, reference LLMs trained on MultiSynt/MT reach the final score of HPLT 2.0, a native-data baseline, using roughly 72% fewer pre-training tokens, and outperform it by approximately 15% relative at a matched 100B-token training budget. Our analyses also identify evaluation blind spots: standard multiple-choice benchmarks miss translation-quality differences that a fluency-sensitive LLM-as-judge protocol recovers on the trained LL
    
[^121]: AI翻译的文学作品“还不错”，但读者仍然更偏爱人工翻译

    AI translation of literary texts is "fine", but readers still prefer human translations

    [https://arxiv.org/abs/2606.26040](https://arxiv.org/abs/2606.26040)

    研究通过让15位读者对15部小说的人工翻译与AI机器翻译进行沉浸式阅读和片段细读比较，发现尽管读者认为机器翻译“还不错”，但因人工翻译更轻松、清晰且更具沉浸感，读者仍显著偏爱人工翻译。

    

    人工智能翻译文学作品正变得越来越普遍。虽然内容可能得到充分传达，但我们对读者在沉浸感和文学效果方面的阅读体验仍缺乏足够了解——这些方面难以被自动指标或针对流畅性与充分性的人工评估所捕捉。我们邀请15位热爱阅读的读者，对15部近期法语、波兰语和日语小说的已出版人工翻译（HT）与基于语言模型智能体流水线生成的机器翻译（MT）进行比较。读者在两种条件下评估了约8000词的摘录：对整个摘录的沉浸式阅读（30次比较），以及对386个对齐的HT-MT片段对的细读（772次比较），每本书由两名读者参与。总体而言，读者认为机器翻译“还可以”，但更偏爱人工翻译（摘录层面为描述性结果的19/30，片段层面达到显著水平的522/772），原因在于人工翻译更轻松、更清晰、更具沉浸感。

    arXiv:2606.26040v2 Announce Type: replace  Abstract: AI translation of literary works is increasingly common. While the content may be rendered adequately, we do not know enough about how readers experience it in terms of immersiveness and literary effect-aspects poorly captured by automatic metrics or human evaluation targeting fluency and adequacy. We ask 15 avid readers to compare recently published human translations (HT) to machine translations (MT) generated with an agentic language model-based pipeline, for 15 recent novels in French, Polish, and Japanese translated into English. Readers evaluated approximately 8K-word excerpts in two conditions: immersive reading of the whole excerpt (30 comparisons) and close reading of 386 aligned HT-MT chunk pairs (772 comparisons), with two readers per book. Overall, readers find MT "fine", but prefer HT (descriptively at the excerpt level, 19/30, and significantly at the chunk level, 522/772) for its ease, clarity, and immersive nature. Re
    
[^122]: 最近发展区策略优化：教师置于提示中，而非梯度中

    Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients

    [https://arxiv.org/abs/2606.18216](https://arxiv.org/abs/2606.18216)

    该论文提出ZPPO，受维果茨基最近发展区理论启发，将教师模型的帮助置于提示词中而非策略梯度中，通过为难题重新构造提示（如将正确教师回答纳入二选一问题），使小型学生模型能够基于自身rollout进行强化学习，从而规避知识蒸馏在小模型上的模仿脆弱性以及向梯度注入教师回答所导致的漂移问题。

    

    知识蒸馏能够将教师模型的能力迁移给小型学生模型，但在“小学生”场景下十分脆弱：迫使学生去模仿远大于自身的教师模型的logits，会使其过度集中于教师分布中最尖锐的众数，从而损害其在训练语料之外的基准任务族上的泛化能力。强化学习（RL）通过在学生自身生成的rollouts上进行训练，避免了logit模仿的问题。然而，当所有rollout都失败时——产生零优势并被静默丢弃——将更强教师的回答注入策略梯度会破坏在策略假设并引起漂移。受维果茨基“最近发展区”理论的启发，我们提出了最近发展区策略优化（ZPPO），它将教师保留在提示词中而非策略梯度中。对于难题，ZPPO会构造两个重新表述的提示。其中一种包含候选的二选一问题（BCQ）将一个正确的教师回答与……（原文摘要在此处截断）

    arXiv:2606.18216v2 Announce Type: replace  Abstract: Knowledge distillation transfers a teacher's competence to a small student but is brittle in the small-student regime: forcing the student to imitate logits from a much larger teacher concentrates it on the teacher's sharpest modes, hurting generalization on benchmark families beyond the training corpus. Reinforcement learning (RL) avoids logit imitation by training on the student's own rollouts. However, on questions where every rollout fails-yielding zero advantage and being silently discarded-injecting a stronger teacher's response into the policy gradient breaks the on-policy assumption and induces drift. We introduce Zone of Proximal Policy Optimization (ZPPO), inspired by Vygotsky's zone of proximal development, which keeps the teacher inside the prompt rather than the policy gradient. On hard questions, ZPPO constructs two reformulated prompts. A Binary Candidate-included Question (BCQ) pairs one correct teacher response with 
    
[^123]: LatentDx：面向跨医院罕见病诊断的潜在多智能体通信

    LatentDx: Latent Multi-Agent Communication for Cross-Hospital Rare-Disease Diagnosis

    [https://arxiv.org/abs/2606.13945](https://arxiv.org/abs/2606.13945)

    提出LatentDx潜在多智能体通信框架，让各医院智能体在本地保留私有临床记录，仅向宿主智能体传输紧凑的潜在KV块，从而在保护隐私的同时实现跨医院罕见病协作诊断。

    

    罕见病影响着超过7000种疾病类型中的3亿多名患者，然而任何一家医院遇到的单一病种病例数量都不足以支持可靠诊断。跨医院协作可以通过让诊断机构利用分布式的、针对具体病例的诊断证据来提供帮助，但隐私法规限制了可识别临床文本的跨机构传输。这一场景带来了两个挑战：现有医疗智能体系统通常依赖文本证据交换，而原始潜在状态（如隐藏状态和KV缓存）仍可能泄露提示中包含的临床内容。我们提出了LatentDx，一个潜在多智能体通信框架，其中医院智能体将私人临床记录和检索到的病例保留在本地，仅向宿主智能体发送紧凑的潜在KV块用于罕见病诊断。LatentDx支持两种部署设置：相同骨干网络的医院智能体使用……（摘要内容在此处截断）

    arXiv:2606.13945v2 Announce Type: replace  Abstract: Rare diseases affect over $300$ million patients across more than $7{,}000$ conditions, yet no single hospital encounters enough cases of any one condition for reliable diagnosis. Cross-hospital collaboration could help by allowing a diagnosing institution to use distributed, case-specific diagnostic evidence, but privacy regulations restrict the transmission of identifiable clinical text across institutional boundaries. This setting raises two challenges: existing medical agent systems often rely on textual evidence exchange, while raw latent states such as hidden states and KV caches may still reveal prompt-derived clinical content. We introduce LatentDx, a latent multi-agent communication framework in which hospital agents keep private clinical records and retrieved cases local, and send compact latent KV blocks to a host agent for rare-disease diagnosis. LatentDx supports two deployment settings: same-backbone hospital agents use
    
[^124]: 争议性政治话语中省略三段论检测的资源

    A Resource for Enthymeme Detection in Controversial Political Discourse

    [https://arxiv.org/abs/2606.12186](https://arxiv.org/abs/2606.12186)

    该论文发布了包含1,482条政治争议推文的省略三段论标注资源，基于沃尔顿论证图式提出结构化标注指南，并通过保留五名标注者的分歧来研究标签差异及其对模型性能的潜在价值。

    

    省略三段论（即含有未明示前提或结论的论证）在说服性话语中普遍存在，但其标注工作一直以主观性强而著称。我们提出了一个包含1,482条来自政治争议性话语的推文的资源，由五名标注者对其中省略三段论的存在及其论证结构进行标注，旨在研究标签差异。我们首先重新审视了省略三段论的定义，并提出了以沃尔顿论证图式为基础的标注指南，提供了一种结构化且受限的方法，同时为任务固有的解释性保留了空间。这与以往倾向于消除分歧的资源形成鲜明对比——那些做法掩盖了分歧的来源，并阻碍了对分歧可能为模型性能带来益处的探索。我们进一步对该任务进行了复杂度分析，识别出标注中哪些环节会带来高认知负荷并可能导致标注不一致。

    arXiv:2606.12186v2 Announce Type: replace  Abstract: Enthymemes, arguments with unstated premises or conclusions, are pervasive in persuasive discourse, yet their annotation remains notoriously subjective. We present a resource of 1,482 tweets from politically controversial discourse, annotated by five annotators for the presence of enthymemes and their argument structure, designed to study label variation. We first revisit the definition of enthymemes and propose annotation guidelines anchored in Walton's argumentation schemes, offering a structured and constrained approach that nonetheless preserves room for the interpretive nature of the task. This contrasts with past resources, which tend to eliminate disagreement, obscuring its sources and preventing investigation of its potential benefits for model performance. We further propose a complexity analysis of the task, identifying where annotation imposes high cognitive load and may give rise to inconsistent annotation. Our preliminar
    
[^125]: 心理健康对话中的专家级危机检测

    Expert-Level Crisis Detection in Mental Health Conversations

    [https://arxiv.org/abs/2606.10380](https://arxiv.org/abs/2606.10380)

    该论文提出了临床医生标注的CRADLE-Dialogue基准数据集以及“警报-确认”评估协议，用于解决多轮心理健康对话中轮次级危机检测的难题，使模型能够捕捉随对话演进的风险信号并支持早期干预。

    

    现实世界的危机干预本质上是对话式的，然而现有研究主要集中于静态文本。当应用于多轮对话时，当前模型表现出显著的性能下降，难以追踪随着上下文演变而出现的风险信号。为了弥补这一空白，我们推出了CRADLE-Dialogue，这是一个由临床医生标注的、用于对话环境中轮次级危机检测的基准数据集。该数据集包含600段对话，针对基于临床的风险（包括自杀意念、自残和虐待儿童）进行了多标签标注，并区分了过去风险与正在发生的风险。我们进一步提出了“警报-确认”评估协议，将早期预警信号与特定危机变得明确可识别的对话轮次区分开来，体现了在风险变得明显之前进行干预的临床需求。实验表明，识别风险何时出现远比识别风险本身困难得多。

    arXiv:2606.10380v2 Announce Type: replace  Abstract: Real-world crisis intervention is inherently conversational, yet existing research largely focuses on static texts. When applied to multi-turn dialogues, current models exhibit significant performance degradation, struggling to track risk signals that emerge as context evolves. To address this gap, we introduce CRADLE-Dialogue, a clinician-annotated benchmark for turn-level crisis detection in conversational settings. The dataset features 600 dialogues with multi-label annotations across clinically grounded risks, including suicide ideation, self-harm, and child abuse, distinguishing past from ongoing risk. We further propose an Alert-Confirm evaluation protocol that distinguishes early warning signals (Alert) from turns where a specific crisis becomes explicitly identifiable (Confirm), reflecting the clinical need to intervene before risk becomes explicit. Experiments show that identifying when risk emerges is much harder than recog
    
[^126]: 少即是MoE：裁剪领域专家型语言模型中的专家

    Less is MoE: Trimming Experts in Domain-Specialist Language Models

    [https://arxiv.org/abs/2606.05538](https://arxiv.org/abs/2606.05538)

    该论文发现MoE压缩失败源于压缩粒度过粗，提出基于Fisher重要性的Fisher-MoE方法，在FFN内部精确移除不重要的中间维度，从而实现领域专家语言模型的高效压缩并保留关键能力。

    

    混合专家模型通过条件计算实现了强大的性能，但其庞大的参数规模给部署带来了挑战。以往的MoE压缩方法在常识推理之外的通用基准测试中评估时会出现灾难性的性能崩溃。我们将这种失败归因于压缩的粒度：重要的能力虽然分布在各个专家之间，但集中在FFN的稀疏中间维度上。为了识别这些维度，我们采用Fisher重要性，其效果优于基于激活值、路由器得分和权重幅度的替代方法，并能识别出极小的任务关键维度集合：在Qwen1.5-MoE中，仅移除1.35M个路由FFN中间维度中的12个就会导致GSM8K准确率崩溃，而事实知识性能却能基本保留。在此基础上，我们提出了Fisher-MoE，该方法在FFN内部运行，移除按Fisher重要性排序的中间维度。

    arXiv:2606.05538v2 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) models achieve strong performance through conditional computation, but their large parameter footprint poses deployment challenges. Prior MoE compression approaches catastrophically fail when evaluated on general-purpose benchmarks beyond commonsense reasoning. We trace this failure to the granularity of compression: important capabilities are distributed across experts but concentrated in FFN sparse intermediate dimensions. To identify these dimensions, we use Fisher importance which outperforms activation-, router-score-, and magnitude-based alternatives, and identifies tiny sets of task-critical dimensions: in Qwen1.5-MoE, removing as few as 12 of 1.35M routed-FFN intermediate dimensions collapses GSM8K accuracy while largely preserving factual-knowledge performance. Building on this, we propose Fisher-MoE, which operates within FFN to remove intermediate dimensions ranked by Fisher importance. At th
    
[^127]: 轻动词还是实义动词？一个用于探测语言模型短语组构能力的最小对立对数据集

    Light or Full Verb? A Minimal-Pair Dataset for Probing Phraseological Competence in Language Models

    [https://arxiv.org/abs/2606.05087](https://arxiv.org/abs/2606.05087)

    本文构建了涵盖英语、西班牙语和法语的最小对立对数据集，通过探测实验证明语言模型能够区分同一动词的轻动词用法与实义动词用法，并公开发布了数据集及生成代码作为可复用资源。

    

    像"have"和"make"这样的高频动词，既可以在轻动词结构中充当搭配词，也可以作为完整的实义谓词使用，例如"make a decision"（做出决定）与"make a cake"（制作蛋糕）。语言模型是否能够表征这一区别，以及这种表征是否会因语言不同而有所差异，目前仍不清楚。我们提出了一个涵盖英语、西班牙语和法语的大规模受控数据集，由微小变化的句子序列组成，其中相同的语境包含同一动词的轻动词用法和实义动词用法。两项探测实验表明，即使在最小语境中，语言模型也能够区分这两种用法，并在不同宾语类型上表现出可分离的模式。我们将数据集、生成代码和相关材料作为可复用资源公开发布。该框架支持向更广泛的语境、更多动词以及其他语言进行扩展。

    arXiv:2606.05087v2 Announce Type: replace  Abstract: Frequent verbs such as 'have' and 'make' can function either as collocates in light-verb constructions or as full lexical predicates, as in 'make a decision' vs. 'make a cake'. Whether language models represent this distinction, and whether such representations vary across languages, remains unclear. We introduce a large-scale controlled dataset in English, Spanish, and French, comprising minimally varying sentence series in which the same context contains the same verb in light-verb and full-verb uses. Two probing experiments show that language models differentiate between these uses even in minimal contexts and exhibit separable patterns across object types. We release the dataset, generation code, and materials as a reusable resource. The framework supports extensions to broader contexts, additional verbs, and other languages.
    
[^128]: BaltiVoice：面向巴尔蒂语的语音语料库与微调Whisper自动语音识别系统

    BaltiVoice: A Speech Corpus and Fine-tuned Whisper ASR System for the Balti Language

    [https://arxiv.org/abs/2606.03504](https://arxiv.org/abs/2606.03504)

    该论文发布了首个巴尔蒂语公开语音语料库BaltiVoice（16.8小时），并通过对Whisper-small进行微调，将该语言的识别词错误率从零样本基线的159.19%大幅降低至24.78%，填补了这一低资源藏语支语言在语音识别领域的空白。

    

    我们提出了BaltiVoice，这是一个面向巴尔蒂语（ISO 639-3: bft）的16.8小时朗读语音语料库。巴尔蒂语是一种在巴基斯坦吉尔吉特-巴尔蒂斯坦地区使用的藏语支语言，此前没有任何公开可用的自动语音识别（ASR）资源。该语料库包含10,060条经过验证的、以本土纳斯塔利格文书写的语句，来源于Mozilla Common Voice录音。对OpenAI Whisper-small进行微调，训练5个周期（3,000步）后，在538条语句的说话人分离验证集上，词错误率（WER）达到24.78%，字符错误率（CER）达到8.30%，相比零样本基线的159.19% WER和152.52% CER大幅下降。在相同数据上微调的Whisper-base达到44.54%的WER和15.61%的CER，证实了在这种低资源场景下模型容量至关重要。数据集、微调后的模型以及实时转录演示均已在HuggingFace上公开发布。

    arXiv:2606.03504v3 Announce Type: replace  Abstract: We present BaltiVoice, a 16.8-hour read-speech corpus for Balti (ISO 639-3: bft), a Tibetic language spoken in Gilgit-Baltistan, Pakistan, with no prior publicly available ASR resources. The corpus contains 10,060 validated utterances in native Nastaliq script, derived from Mozilla Common Voice recordings. Fine-tuning OpenAI Whisper-small yields a Word Error Rate (WER) of 24.78% and a Character Error Rate (CER) of 8.30% after training for 5 epochs (3,000 steps) on the 538-utterance speaker-disjoint validation set, down from a zero-shot baseline of 159.19% WER and 152.52% CER. A Whisper-base fine-tuned on the same data achieves 44.54% WER and 15.61% CER, confirming that model capacity matters for this low-resource setting. The dataset, fine-tuned model, and a live transcription demo are publicly available on HuggingFace.
    
[^129]: 看见、推断、干预：面向目标导向社会智能的主动世界建模

    See, Infer, Intervene: Proactive World Modeling for Goal-Oriented Social Intelligence

    [https://arxiv.org/abs/2606.03371](https://arxiv.org/abs/2606.03371)

    提出SII框架和PIWM模型，使零售代理能通过观察、推断顾客意图并主动选择适当干预，在无明确请求时提供辅助。

    

    多模态零售代理不仅应识别顾客正在做什么，还应在顾客明确请求之前决定是否以及如何提供帮助。我们通过“看见—推断—干预”（SII）框架研究这一场景，其中设备必须观察互动前的行为，推断潜在顾客意图，并通过选择适当的服务干预或决定等待来采取行动。我们使用主动意图世界模型（PIWM）实现SII，该模型利用AIDA（注意、兴趣、欲望、行动）购买阶段和BDI（信念、欲望、意图）心理场表示顾客状态，预测基于行动的条件性意图转变，并从五类回应中选择：问候、引导、告知、推荐和等待。我们进一步构建了GuidanceSalesBench，一个智能零售基准数据集，包含状态清单、互动前视频、候选回应、基于行动的结果以及最佳行动标签。

    arXiv:2606.03371v3 Announce Type: replace  Abstract: Multimodal retail agents should not only recognize what a customer is doing, but also decide whether and how to assist before an explicit request is made. We study this setting through the See--Infer--Intervene (SII) framework, where a device must see pre-interaction behavior, infer latent customer intent, and act by selecting an appropriate service intervention or choosing to wait. We instantiate SII with the Proactive Intent World Model (PIWM), which represents customer state with AIDA (Attention, Interest, Desire, Action) purchasing phases and BDI (belief, desire, intention) psychological fields, predicts action-conditioned intent transitions, and selects from five response classes: Greet, Elicit, Inform, Recommend, and Hold. We further construct GuidanceSalesBench, a smart-retail benchmark containing state manifests, pre-interaction videos, candidate responses, action-conditioned outcomes, and best-action labels. When conditioned
    
[^130]: SEA-LION-Embedding：面向东南亚的开放且可复现的文本嵌入

    SEA-LION-Embedding: Open and Reproducible Text Embeddings for Southeast Asia

    [https://arxiv.org/abs/2606.03027](https://arxiv.org/abs/2606.03027)

    SEA-LION-Embedding是一个完全开放且可复现的东南亚语言文本嵌入模型，仅使用公开数据训练，在SEA-BED基准上达到最先进水平，并系统研究了数据构成、训练目标和基础编码器初始化这三个影响鲁棒嵌入设计的核心因素。

    

    文本嵌入是许多下游应用的基础，因此鲁棒性对现实世界的自然语言处理（NLP）至关重要。然而，近期大多数最先进的嵌入模型都不可复现，因为它们依赖于封闭或未公开的训练数据，并且对于东南亚语言仍然缺乏足够的鲁棒性。我们提出了SEA-LION-Embedding，这是一个完全开放且可复现的面向东南亚语言的文本嵌入流水线，仅使用公开可用的数据进行训练，并用它来研究鲁棒嵌入设计的三个核心因素：数据构成、训练目标和基础编码器初始化。SEA-LION-Embedding在SEA-BED上取得了最先进的结果，同时实现了针对该地区鲁棒文本嵌入的系统性且可复现的分析。

    arXiv:2606.03027v2 Announce Type: replace  Abstract: Text embeddings are fundamental to many downstream applications, making robustness important for real-world NLP. However, most recent state-of-the-art embedding models are not reproducible because they rely on closed or undisclosed training data, and they remain insufficiently robust for Southeast Asian languages. We present SEA-LION-Embedding, a fully open and reproducible text-embedding pipeline for Southeast Asian languages trained only on publicly available data, and use it to study three core factors of robust embedding design: data composition, training objective, and base encoder initialization. SEA-LION-Embedding achieves state-of-the-art results on SEA-BED while enabling systematic and reproducible analysis of robust text embeddings for the region.
    
[^131]: ActTraitBench：基于人类行为验证量化大语言模型中的知识-决策差距

    ActTraitBench: Quantifying the Knowledge-Decision Gap in Large Language Models via Human-Grounded Behavioral Validation

    [https://arxiv.org/abs/2605.29791](https://arxiv.org/abs/2605.29791)

    该论文提出ActTraitBench框架，通过将心理测量维度与行为范式一一映射并采用分位数映射分布校准，以人类实证数据为基准量化了大语言模型自我报告与实际行为决策之间的知识-决策差距。

    

    尽管大语言模型（LLM）能够在显性自我报告中令人信服地模拟特定人物角色，但它们在隐性行为决策中常常出现偏差，从而暴露出显著的知识-决策差距（Knowledge-Decision Gap, G_KD）。由于构念效度有限、多个维度相互纠缠，以及基于LLM的评估中存在的分布偏差，现有基准难以测量这种差异。为解决这些问题，我们提出了ActTraitBench——一个以人类数据为基础、用于衡量大语言模型人格一致性的评估框架。ActTraitBench以实证人类数据为根基，建立了心理测量维度与行为范式之间的一一对应映射，并采用基于分位数映射的分布校准方法，以减少LLM评委打分与人类反应之间的分布不匹配。在14个主流大语言模型上的实验揭示了显著的知识-决策差距，并表明模型被赋予的人物角色在自我报告中比在行为决策中体现得更为一致。

    arXiv:2605.29791v2 Announce Type: replace  Abstract: While Large Language Models (LLMs) can convincingly simulate personas in explicit self-reports, they often deviate in implicit behavioral decisions, revealing a substantial Knowledge-Decision Gap ($G_{\mathrm{KD}}$). Existing benchmarks struggle to measure this discrepancy due to limited construct validity, multidimensional entanglement, and distributional biases in LLM-based evaluation. To address these issues, we propose ActTraitBench, a human-grounded evaluation framework for measuring personality consistency in LLMs. Grounded in empirical human data, ActTraitBench establishes one-to-one mappings between psychometric facets and behavioral paradigms and applies Distributional Calibration via Quantile Mapping to reduce distributional mismatch between LLM-judge scores and human responses. Experiments on 14 mainstream LLMs reveal substantial knowledge-decision gaps and show that assigned personas are reflected more consistently in sel
    
[^132]: 语言模型中的文化绑定注意力头

    Cultural Binding Heads in Language Models

    [https://arxiv.org/abs/2605.28543](https://arxiv.org/abs/2605.28543)

    该研究通过机制可解释性方法在八个语言模型中识别出2-3个负责文化绑定的中层注意力头，证明文化绑定形成于预训练阶段，并通过生成阶段的适度放大引导将文化区分准确率提升1-3个百分点。

    

    大语言模型往往对不同文化群体采取一视同仁的默认处理方式，即使上下文需要做出区分：这是一种缺乏差异意识的表现。我们利用机制可解释性方法，并在Wang等人（2025）提出的N4文化挪用基准上采用因子实验设计，在八个模型（四种架构的基础版和指令微调版）中识别出每个模型中2-3个对文化绑定具有因果贡献的中层注意力头。文化绑定是指将文化项目与其相关身份关联起来的过程。敲除这些注意力头上从身份到项目的连接边可使绑定强度降低9-23%。所识别的注意力头能够从指令微调模型迁移到基础模型，表明文化绑定是在预训练期间形成的。α缩放实验显示出分级的剂量-响应关系。在生成阶段进行适度的放大引导（α=2-3）可将文化区分准确率提高1-3个百分点，同时对推理能力的影响保持在可接受范围内。

    arXiv:2605.28543v3 Announce Type: replace-cross  Abstract: LLMs often default to equal treatment across cultural groups, even though context warrants differentiation: this is a lack of difference awareness. Using mechanistic interpretability and a factorial design on the N4 cultural appropriation benchmark from Wang et al. (2025), we identify 2-3 mid-layer attention heads per model that contribute causally to cultural binding across eight models (base and instruct versions of four architectures). Cultural binding is the process of associating a cultural item with its related identity. Knockout of the identity-to-item edges on these heads lowers the binding strength by 9-23%. The identified heads transfer from instruct to base models, suggesting that cultural binding is created during pre-training. An $\alpha$-scaling shows a graded dose-response. Moderate amplification steering at generation ($\alpha = 2-3$) increases cultural differentiation accuracy by 1-3 pp while leaving reasoning 
    
[^133]: 追踪大语言模型中的计算密度

    Tracing Computation Density in LLMs

    [https://arxiv.org/abs/2605.27033](https://arxiv.org/abs/2605.27033)

    本文提出s-Trace方法来估计能近似完整输出的LLM计算子图，发现模型计算呈现两阶段组织模式：早期层节点构成的小子图即可重构输出分布主体，后续计算仅为渐进式精细化，且每个输入所需的计算量与模型不确定性相关。

    

    基于Transformer的大语言模型（LLM）由数十亿个参数组成的深层和宽层计算图构成，但目前尚不清楚它们是否对所有输入都充分利用了其全部容量。我们提出了s-Trace方法，可以高效地估计一个大小为s的子图，该子图能够近似完整的模型输出。通过这种方法，我们发现多种LLM中的计算以两个不同的阶段进行组织。一个主要由早期层节点组成的小型子图即可重构完整模型输出分布的头部。随着添加更多节点（主要位于较深层，且越来越多地由注意力头构成），对完整输出分布的近似会得到渐进式的精细化。此外，我们还发现每个输入所需的计算量与模型不确定性相关，且更稀疏的子图编码的是浅层统计信息，例如一元词频。总体而言，我们的结果表明了一种一致的

    arXiv:2605.27033v2 Announce Type: replace  Abstract: Transformer-based large language models (LLMs) are comprised of billions of parameters arranged in deep and wide computational graphs, but it is not clear that they exploit their full capacity for all inputs. We introduce the s-Trace method to efficiently estimate a subgraph of size s that approximates a full model output. With this method, we find the computation in a variety of LLMs to be organized in two distinct phases. A small subgraph mostly composed of early-layer nodes can reconstruct the head of the full model output distribution. Adding further nodes, mostly located in later layers and increasingly consisting of attention heads, leads to incremental refinements in approximating the full output distribution. We find moreover that the amount of necessary computation per input correlates with model uncertainty, and that sparser subgraphs encode shallow statistics, such as unigram frequency. Overall, our results suggest a consi
    
[^134]: SpecBench：衡量长时程编码智能体中的奖励作弊行为

    SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents

    [https://arxiv.org/abs/2605.21384](https://arxiv.org/abs/2605.21384)

    SpecBench通过将软件工程任务分解为规范描述、可见验证测试和保留测试三部分，利用智能体在可见测试与保留测试上通过率的差距，量化了长时程编码智能体中的奖励作弊行为。

    

    随着长时程编码智能体产生的代码量超过任何开发者所能审查的限度，人类监督便集中到唯一的表面上：自动化测试套件。在这种设置下，奖励作弊（reward hacking）现象自然产生，因为智能体以通过测试为目标进行优化，却偏离了用户的真实目标。我们通过将软件工程任务分解为三个部分来研究这一奖励作弊现象：（i）规范的自然语言描述，（ii）针对已规定功能进行孤立验证的可见验证测试，以及（iii）将这些功能组合起来以模拟真实世界使用的保留测试。基于规范和可见验证测试套件，一个真正可靠的智能体应当能够生成同样可以通过所有保留测试的解决方案。因此，我们使用智能体在这两个测试套件上通过率的差距来量化奖励作弊。基于这一方法，我们推出了SpecBench，一个包含30个系统级（任务）的基准测试。

    arXiv:2605.21384v2 Announce Type: replace-cross  Abstract: As long-horizon coding agents produce more code than any developer can review, oversight collapses onto a single surface: the automated test suite. Reward hacking naturally arises in this setup, as the agent optimizes for passing tests while deviating from the users true goal. We study this reward hacking phenomenon by decompose software engineering tasks into three parts: (i) a natural language description of the specification (ii) visible validation tests that exercise specified features in isolation, and (iii) held-out tests that compose those same features to simulate real-world usage. Based on the specification and the visible validation test suites, a genuine agent would be able to generate a solution that can also pass all of the held-out tests. Therefore we use the gap in pass rates on these two suites to quantify reward hacking. Based on this methodology, we introduce SpecBench, a benchmark comprising 30 systems-level 
    
[^135]: 判官电路解释了LLM-as-a-Judge中由格式引起的不一致性

    Judge Circuits Explain Format-Induced Inconsistency in LLM-as-a-Judge

    [https://arxiv.org/abs/2605.16023](https://arxiv.org/abs/2605.16023)

    该论文通过PEAP方法发现LLM裁判模型的中后层MLP中存在一个稀疏的“潜在评估者”子图，该子图负责抽象评判且独立于输出格式，从而在机制层面解释了LLM-as-a-Judge中格式诱导的评分不一致现象。

    

    大语言模型作为裁判（LLM-as-a-judge）已成为大规模评估模型输出的主流范式，然而同一模型在输出格式改变时会给出系统性不同的分数（例如1-5分评分与真/假标签）。现有的针对这种格式诱导不一致性的诊断仅停留在输入-输出层面。我们使用位置感知边归因补丁（PEAP）方法，对五个开源权重指令微调模型（Gemma-3、Qwen2.5、Llama-3.1）在五个判断任务上的内部机制进行了因果性研究。我们发现，结构化理解任务和开放式偏好任务的判断在多层感知机（MLP）的中后层共享一个稀疏的“潜在评估者”子图；在架构上模块化的模型中，对该子图进行零消融会使判断能力崩溃，同时保持模型在知识探针上的性能。通过在结构上将抽象评判与输出格式化解耦，我们为格式诱导的不一致性提供了机制层面的解释。

    arXiv:2605.16023v3 Announce Type: replace  Abstract: LLM-as-a-judge has become the dominant paradigm for grading model outputs at scale, yet the same model assigns systematically different scores when its output format changes (e.g., a 1-5 rating vs. a True/False label). Existing diagnoses of these format-induced inconsistencies stop at the input-output level. Using Position-aware Edge Attribution Patching (PEAP), we causally investigate the internal mechanism in five open-weight instruction-tuned models (Gemma-3, Qwen2.5, Llama-3.1) across five judgment tasks. We find that judgments across structured understanding and open-ended preference tasks share a sparse Latent Evaluator sub-graph in the mid-to-late multi-layer perceptrons (MLPs); zero-ablating it collapses judgment while preserving performance on our knowledge probes in architecturally modular models. By structurally decoupling abstract judging from output formatting, we provide a mechanistic account of format-induced inconsist
    
[^136]: EVA-Bench：一个新的语音智能体端到端评估框架

    EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents

    [https://arxiv.org/abs/2605.13841](https://arxiv.org/abs/2605.13841)

    EVA-Bench提出了一个端到端语音智能体评估框架，通过带自动验证的机器人间动态音频对话模拟与EVA-A（准确性）、EVA-X（体验）两项复合指标，首次同时实现了真实对话模拟与全面的语音专项评估。

    

    语音智能体在企业应用中的部署日益增多。然而，目前尚无现有基准能够同时解决真实对话模拟和全面的语音专项评估这两个问题。我们提出了EVA-Bench，一个能够同时应对这两个问题的端到端评估框架。在模拟方面，EVA-Bench编排动态的机器人对机器人音频对话，并通过自动模拟验证来检测用户模拟器的错误，在评分前适当地重新生成对话。在测量方面，EVA-Bench引入了两个复合指标：EVA-A（准确性）和EVA-X（体验）。EVA-Bench涵盖三个企业领域的213个场景、用于评估口音和噪声鲁棒性的受控扰动套件，以及区分峰值能力与可靠能力的多次试验测量。在对跨越三种架构的12个系统的评估中，我们发现：（1）没有系统能同时在EVA-A pass@1和EV...

    arXiv:2605.13841v3 Announce Type: replace-cross  Abstract: Voice agents are increasingly deployed across enterprise applications. However, no existing benchmark jointly addresses realistic conversation simulation and comprehensive voice-specific evaluation. We present EVA-Bench, an end-to-end evaluation framework that addresses both. On the simulation side, EVA-Bench orchestrates dynamic bot-to-bot audio conversations with automatic simulation validation that detects user simulator error and appropriately regenerates conversations before scoring. On the measurement side, EVA-Bench introduces two composite metrics: EVA-A (Accuracy) and EVA-X (Experience). EVA-Bench includes 213 scenarios across three enterprise domains, a controlled perturbation suite for accent and noise robustness, and multi-trial measurements that distinguish peak from reliable capability. Across 12 systems spanning all three architectures, we find: (1) no system simultaneously exceeds 0.5 on both EVA-A pass@1 and EV
    
[^137]: “你究竟想做什么？”：从日常计算机使用中共同创造人生目标

    "What Are You Really Trying to Do?": Co-Creating Life Goals from Everyday Computer Use

    [https://arxiv.org/abs/2605.00497](https://arxiv.org/abs/2605.00497)

    本文提出“追求共创”方法，基于活动理论和个人追求框架，从日常计算机使用的非结构化观察中逐步推断用户更广泛的人生目标，并通过编辑界面让用户掌控系统对自身的理解，突破现有系统仅提供表面级支持的局限。

    

    用户建模的最新进展使得对个人日常计算机使用进行开放式推理成为可能。尽管长期以来人们一直设想系统能够深入理解我们的行为及其在生活中的目的，但现有系统只能捕捉用户当下的行为，而无法理解其背后的原因，这限制了这些系统只能提供浅层次的支持。我们提出了“追求共创”，这是一种从计算机使用的非结构化观察中推断更广泛人生目标的过程。基于活动理论和埃蒙斯的个人追求框架，我们的系统逐步构建个人活动的层次化表示。然而，仅凭观察很难完全确定个人的追求目标，因为同一行为可能由许多不同的目标所驱动。因此，我们的系统支持一个编辑界面，让用户能够对系统如何理解自己拥有主动权，并将其修正反馈给系统……

    arXiv:2605.00497v2 Announce Type: replace-cross  Abstract: Recent advances in user modeling make it feasible to conduct open-ended inference over a person's everyday computer use. Despite longstanding visions of systems that deeply understand our actions and the purposes they serve in our lives, existing systems only capture what a person is doing in the moment, not why they are doing it, limiting these systems to surface-level support. We introduce striving co-creation, a process for inferring broader life goals from unstructured observations of computer use. Grounded in Activity Theory and Emmons' personal strivings framework, our system progressively constructs a hierarchical representation of a person's activities. Strivings are, however, difficult to fully resolve from observation alone, as the same action can be driven by many different goals. Our system therefore supports an editing interface that gives people agency over how they are understood by the system, feeding their corr
    
[^138]: 多层次叙事评估在心理健康预测中优于词汇特征

    Multi-Level Narrative Evaluation Outperforms Lexical Features for Mental Health

    [https://arxiv.org/abs/2604.27846](https://arxiv.org/abs/2604.27846)

    提出了一个由词汇特征、语义嵌入和大语言模型叙事评估构成的三级叙事分析框架，并在830篇中文治疗文本上证明宏观层面的LLM叙事评估在心理健康预测中显著优于传统词汇计数特征。

    

    人们如何叙述自己的经历，为了解心智如何组织这些经历提供了一扇窗口。针对治疗性写作的计算方法已从词汇计数发展到神经网络方法，但仍然碎片化：词典工具无法捕捉语篇结构，而嵌入方法则将局部连贯性与全局组织混为一谈。目前尚无任何框架将这些技术映射到叙事构建所依赖的层次化过程上。本文提出了一个三级框架——微观层面的词汇特征、中观层面的语义嵌入和宏观层面的大语言模型叙事评估——并在涵盖抑郁、焦虑和创伤的830篇中文治疗文本上证明，宏观层面的评估在心理健康预测上显著优于词汇特征和嵌入特征。这一发现挑战了该领域对词频统计的重视：形式化结构特征（Labov的故事语法、RST连贯性、命题组合）表……

    arXiv:2604.27846v2 Announce Type: replace  Abstract: How people narrate their experiences offers a window into how the mind organizes them. Computational approaches to therapeutic writing have evolved from lexical counting to neural methods, yet remain fragmented: dictionary tools miss discourse structure, while embeddings conflate local coherence with global organization. No existing framework maps these techniques onto the hierarchical processes through which narratives are constructed. Here we introduce a three-level framework - micro-level lexical features, meso-level semantic embeddings, and macro-level LLM narrative evaluation - and show, across 830 Chinese therapeutic texts spanning depression, anxiety, and trauma, that macro-level evaluation substantially outperforms lexical and embedding features for mental health prediction. This challenges the field's emphasis on word-counting: formal structural features (Labov's story grammar, RST coherence, propositional composition) demon
    
[^139]: EviMem：面向长期对话记忆的证据缺口驱动迭代检索

    EviMem: Evidence-Gap-Driven Iterative Retrieval for Long-Term Conversational Memory

    [https://arxiv.org/abs/2604.27695](https://arxiv.org/abs/2604.27695)

    EviMem通过显式诊断证据缺口的闭环迭代检索框架IRIS与分层记忆架构LaceMem相结合，在长期对话记忆的时序和多跳问题上显著超越MIRIX，同时将延迟降低4.5倍。

    

    长期对话记忆需要检索分散在多个会话中的证据，然而单次检索在时序类和多跳类问题上表现不佳。现有的迭代方法通过生成内容或文档级信号来优化查询，但没有一种方法显式地诊断“证据缺口”，即累积检索集合中缺失了什么内容，导致查询优化缺乏针对性。我们提出了EviMem，它结合了两项技术：IRIS（基于不足信号的迭代检索），这是一个闭环框架，通过充分性评估检测证据缺口、诊断缺失内容并驱动有针对性的查询优化；以及LaceMem（面向对话证据记忆的分层架构），这是一个由粗到细的记忆层次结构，支持细粒度的缺口诊断。在LoCoMo基准上，EviMem在时序问题上将Judge Accuracy从73.3%提升至81.6%，在多跳问题上从65.9%提升至85.2%（相比MIRIX），同时延迟降低了4.5倍。

    arXiv:2604.27695v2 Announce Type: replace-cross  Abstract: Long-term conversational memory requires retrieving evidence scattered across multiple sessions, yet single-pass retrieval fails on temporal and multi-hop questions. Existing iterative methods refine queries via generated content or document-level signals, but none explicitly diagnoses the evidence gap, namely what is missing from the accumulated retrieval set, leaving query refinement untargeted. We present EviMem, combining IRIS (Iterative Retrieval via Insufficiency Signals), a closed-loop framework that detects evidence gaps through sufficiency evaluation, diagnoses what is missing, and drives targeted query refinement, with LaceMem (Layered Architecture for Conversational Evidence Memory), a coarse-to-fine memory hierarchy supporting fine-grained gap diagnosis. On LoCoMo, EviMem improves Judge Accuracy over MIRIX on temporal (73.3% to 81.6%) and multi-hop (65.9% to 85.2%) questions at 4.5x lower latency. Code: https://gith
    
[^140]: 在混合专家模型微调中保留长尾专家信息

    Preserving Long-Tailed Expert Information in Mixture-of-Experts Tuning

    [https://arxiv.org/abs/2604.23036](https://arxiv.org/abs/2604.23036)

    提出一种无辅助损失的MoE监督微调框架，通过偏置驱动的稀疏化与始终激活的门控凝聚专家相结合，在保留稀有激活专家中关键知识的同时，避免了现有方法因噪声梯度导致的性能下降。

    

    尽管MoE模型在许多基准测试中处于领先地位，但针对MoE架构的监督微调（SFT）仍然十分困难，因为其路由层非常脆弱。DenseMixer和ESFT等方法通过密集混合或辅助负载均衡损失来缓解路由器崩溃问题，但这些方法引入的噪声梯度往往会降低性能。在初步实验中，我们系统地剪枝专家并观察到，虽然某些“超级专家”被激活的频率远高于其他专家，但丢弃使用较少的专家仍会导致显著的性能下降。这表明即使是很少被激活的专家也编码了对下游任务有用的非平凡知识。受此启发，我们提出了一种无辅助损失的MoE SFT框架，该框架将偏置驱动的稀疏化与始终激活的门控凝聚专家相结合。我们的方法并非强制所有专家均衡激活，而是鼓励与任务相关的专家保持激活（摘要在此处截断）。

    arXiv:2604.23036v2 Announce Type: replace-cross  Abstract: Despite MoE models leading many benchmarks, supervised fine-tuning (SFT) for the MoE architectures remains difficult because its router layers are fragile. Methods such as DenseMixer and ESFT mitigate router collapse with dense mixing or auxiliary load-balancing losses, but these introduce noisy gradients that often degrade performance. In preliminary experiments, we systematically pruned experts and observed that while certain super experts are activated far more frequently, discarding less used experts still leads to notable performance degradation. This suggests that even rarely activated experts encode non-trivial knowledge useful for downstream tasks. Motivated by this, we propose an auxiliary-loss-free MoE SFT framework that combines bias-driven sparsification with always-active gated condenser experts. Rather than enforcing balanced activation across all experts, our method encourages task-relevant experts to remain acti
    
[^141]: 心智在哪里？人格向量与大语言模型的个体化问题

    Where is the Mind? Persona Vectors and LLM Individuation

    [https://arxiv.org/abs/2604.17031](https://arxiv.org/abs/2604.17031)

    本文通过机制可解释性研究大语言模型的个体化问题，提出并论证了虚拟实例观点以及两种新观点（实例-人格观点和模型-人格观点）作为认定LLM心智的最有力候选方案。

    

    大语言模型的个体化问题探讨的是：与大语言模型相关联的哪些实体（如果有的话）应当被认定为心智。我们通过机制可解释性来研究这一问题，特别是结合了关于人格向量、人格空间和涌现性失调的最新实证工作。我们认为有三种观点是最有力的候选方案：虚拟实例观点，以及我们引入的两种新观点——（虚拟）实例-人格观点和模型-人格观点。首先，我们论证了虚拟实例观点，其理由是注意力流在词元时间维度上维持着准心理学连接。随后，我们围绕关于大语言模型中人格内在结构的三种假说，梳理了人格相关文献，并表明这两种基于人格的观点是有前景的替代方案。

    arXiv:2604.17031v3 Announce Type: replace  Abstract: The individuation problem for large language models asks which entities associated with them, if any, should be identified as minds. We approach this problem through mechanistic interpretability, engaging in particular with recent empirical work on persona vectors, persona space, and emergent misalignment. We argue that three views are the strongest candidates: the virtual instance view and two new views we introduce, the (virtual) instance-persona view and the model-persona view. First, we argue for the virtual instance view on the grounds that attention streams sustain quasi-psychological connections across token-time. Then we present the persona literature, organised around three hypotheses about the internal structure underlying personas in LLMs, and show that the two persona-based views are promising alternatives.
    
[^142]: 价值模型回归：用于大语言模型强化学习中价值建模的生成式批评家

    Bringing Value Models Back: Generative Critics for Value Modeling in LLM Reinforcement Learning

    [https://arxiv.org/abs/2604.10701](https://arxiv.org/abs/2604.10701)

    该论文提出生成式Actor-Critic（GenAC），用先进行思维链推理再输出价值的生成式批评家替代传统单次标量价值预测，从而解决了大语言模型强化学习中价值模型因表达能力受限而难以可靠训练的问题。

    

    信用分配是强化学习中的核心挑战。经典的Actor-Critic方法通过基于学习到的价值函数进行细粒度的优势估计来应对这一挑战。然而，在现代大语言模型强化学习中，人们通常避免使用学习到的价值模型，因为传统的判别式批评家难以可靠地训练。我们重新审视了价值建模，并认为这种困难部分源于表达能力受限。具体而言，表示复杂性理论表明，在现有价值模型所采用的单次预测范式下，价值函数可能难以逼近，且我们的缩放实验表明，此类批评家无法随模型规模扩大而可靠地改进。基于这一观察，我们提出了生成式Actor-Critic（GenAC），它用一种生成式批评家取代了单次标量价值预测，该批评家在产生价值之前先进行思维链推理。

    arXiv:2604.10701v2 Announce Type: replace-cross  Abstract: Credit assignment is a central challenge in reinforcement learning (RL). Classical actor-critic methods address this challenge through fine-grained advantage estimation based on a learned value function. However, learned value models are often avoided in modern large language model (LLM) RL because conventional discriminative critics are difficult to train reliably. We revisit value modeling and argue that this difficulty is partly due to limited expressiveness. In particular, representation complexity theory suggests that value functions can be hard to approximate under the one-shot prediction paradigm used by existing value models, and our scaling experiments show that such critics do not improve reliably with scale. Motivated by this observation, we propose Generative Actor-Critic (GenAC), which replaces one-shot scalar value prediction with a generative critic that performs chain-of-thought reasoning before producing a valu
    
[^143]: 英国多语言英语使用者在AI驱动的语音认知筛查中的假阳性偏差

    False positive bias in AI-powered speech-based cognitive screening for multilingual English speakers in the UK

    [https://arxiv.org/abs/2602.13047](https://arxiv.org/abs/2602.13047)

    该研究通过对1,395名参与者、超过263小时语音数据的分析，首次发现尽管语音识别准确率在各语言群体间无显著差异，但AI认知筛查的下游模型对英国多语言英语使用者存在系统性的假阳性偏差，凸显了认知筛查公平性评估的重要性。

    

    会话语音能够揭示认知衰退的早期迹象，包括痴呆症和轻度认知障碍（MCI）。AI模型在基于语音的筛查方面展现出前景，但大多数研究聚焦于单语群体。在英国，痴呆症预计在黑人和亚裔社区中增长最快，而这些社区中多语言现象普遍，因此公平性评估至关重要。我们招募了1,395名参与者（包括谢菲尔德/布拉德福德的英语单语者和多语者），并通过CognoMemory智能体收集了超过263小时的语音数据。多语言参与者在说英语的同时还使用索马里语、中文或南亚语言（印地语、乌尔都语、旁遮普语、米尔普里语、阿拉伯语）。我们评估了自动语音识别系统（Whisper、Wav2Vec 2.0、NeMo）以及用于认知分类和MMSE回归的下游AI模型。ASR准确率在各群体间未显示出显著差异。然而，下游模型表现出系统性差异：多语言使用者……

    arXiv:2602.13047v2 Announce Type: replace  Abstract: Conversational speech reveals early signs of cognitive decline, including dementia and mild cognitive impairment (MCI). AI models show promise for speech-based screening, yet most research focuses on monolingual groups. In the UK, dementia is projected to rise fastest among Black and Asian communities, where multilingualism is common, making equity assessment critical. We recruited 1,395 participants (monolingual English speakers and multilingual speakers from Sheffield/Bradford) and collected over 263 hours of speech via the CognoMemory agent. Multilingual participants spoke English alongside Somali, Chinese, or South Asian languages (Hindi, Urdu, Punjabi, Mirpuri, Arabic). We evaluated ASR (Whisper, Wav2Vec 2.0, NeMo) and downstream AI models for cognitive classification and MMSE regression. ASR accuracy showed no significant differences across groups. However, downstream models exhibited systematic disparities: multilingual speake
    
[^144]: 用于对话式医疗AI风险评估的患者模拟框架：抗抑郁药物决策辅助工具的评估

    A Patient Simulation Framework for Risk Assessment of Conversational Healthcare AI: Evaluation of an Antidepressant Decision Aid

    [https://arxiv.org/abs/2602.11391](https://arxiv.org/abs/2602.11391)

    本研究提出了一个符合NIST AI风险管理框架的患者模拟框架，通过整合医学、语言学和行为学三个维度的患者画像生成500次模拟对话，为评估对话式医疗AI（如抗抑郁药物选择决策辅助工具）的性能风险提供了实证基础。

    

    目标：本研究开发并验证了一个患者模拟框架，该框架与美国国家标准与技术研究院（NIST）AI风险管理框架的MAP和MEASURE功能保持一致，为识别和表征对话式临床AI在面对医学、语言学和行为学患者差异时的性能风险提供了实证基础。我们将该框架应用于一款针对重度抑郁症抗抑郁药物选择的对话式决策辅助工具。方法：该模拟器整合了三个画像维度：（1）医学画像，使用风险比门控方法从“All of Us”电子健康记录构建；（2）语言学画像，建模健康素养梯度及特定病症的沟通方式；（3）行为学画像，表现合作型、分心型和对抗型参与方式。我们生成了500次模拟对话，并通过人工标注和自动（评估）来衡量画像的保真度……

    arXiv:2602.11391v5 Announce Type: replace  Abstract: Objective: This study develops and validates a patient simulation framework that aligns with the National Institute of Standards and Technology AI Risk Management Framework MAP and MEASURE functions, providing an empirical basis for identifying and characterizing performance risks in conversational clinical AI across medical, linguistic, and behavioral patient variation. We applied the framework to a conversational decision aid for antidepressant selection in major depressive disorder. Methods: The simulator integrates three profile dimensions: (1) medical profiles constructed from All of Us electronic health records using risk-ratio gating; (2) linguistic profiles modeling a health literacy gradient and condition-specific communication; and (3) behavioral profiles representing cooperative, distracted, and adversarial engagement. We generated 500 simulated conversations and evaluated profile fidelity through human annotation and a la
    
[^145]: 重新审视Transformer语言模型的形状惯例

    Revisiting the Shape Convention of Transformer Language Models

    [https://arxiv.org/abs/2602.06471](https://arxiv.org/abs/2602.06471)

    该论文提出沙漏Transformer架构，用残差沙漏形MLP堆叠替代传统窄-宽-窄FFN并通过沙漏注意力解耦残差流与注意力宽度，在113M至8B参数规模上以更少层数和更宽隐藏状态实现了与传统Transformer相当的性能，同时提高了训练计算效率。

    

    稠密Transformer的架构形状一直保持着出奇地稳定：窄-宽-窄的前馈网络（FFN）消耗了大部分非嵌入参数。基于残差宽-窄-宽（沙漏形）MLP即使存在瓶颈仍保持表达能力的理论与实证证据，我们重新审视这种架构惯例对稠密语言模型是否必要。我们研究了沙漏Transformer，它用残差堆叠的沙漏子MLP取代传统FFN，并使用沙漏注意力将残差流宽度与注意力宽度解耦。这揭示了一种实用的深度-宽度权衡：在匹配参数预算下，压缩FFN中间维度可以采用更宽的隐藏状态和更少的层数。在113M到8B参数的各模型规模上，沙漏Transformer实现了与传统Transformer相当的语言建模和下游性能，同时提升了训练计算效率。

    arXiv:2602.06471v2 Announce Type: replace  Abstract: The architectural shape of dense Transformers has remained remarkably stable: narrow-wide-narrow feed-forward networks (FFNs) consume most non-embedding parameters. Motivated by theoretical and empirical evidences that residual wide-narrow-wide (hourglass) MLPs remain expressive despite bottlenecks, we revisit whether this architectural convention is necessary for dense language models. We study Hourglass Transformers, which replace the conventional FFN with residual stacks of hourglass sub-MLPs and use hourglass attention to decouple residual-stream width from attention width. This exposes a practical depth-width trade-off: compressing the FFN intermediate dimension allows wider hidden states and fewer layers at matched parameter budgets. Across model scales from 113M to 8B parameters, Hourglass Transformers achieve language-modeling and downstream performance comparable to conventional Transformers, while improving training compute
    
[^146]: 我是更偏向逐点式还是成对式？揭示基于评分标准的大语言模型评审中的位置偏差

    Am I More Pointwise or Pairwise? Revealing Position Bias in Rubric-Based LLM-as-a-Judge

    [https://arxiv.org/abs/2602.02219](https://arxiv.org/abs/2602.02219)

    该论文揭示了基于评分标准的LLM评审本质上类似于多项选择题设置，存在系统性的位置偏差——模型倾向于偏好评分标准列表中特定位置的分数选项，且偏差方向因模型而异。

    

    大语言模型被广泛用作评估器，这一范式通常被称为“LLM即评审”。先前的研究主要关注逐点式或成对式评估协议；相比之下，我们聚焦于基于评分标准的评估，由于其在难以进行验证的领域中训练模型的实用性，这种方法正受到越来越多的关注。在这项工作中，我们证明基于评分标准的评估在隐含层面上类似于多项选择题设置，因此表现出位置偏差：大语言模型倾向于偏好出现在评分标准列表中特定位置上的分数选项。通过在多个模型和数据集上进行的对照实验，我们证明这种位置偏差是持续存在的。然而，其偏差方向因模型而异：一些评审模型偏好第一个选项，而另一些则偏好最后一个选项。我们进一步识别出第二个与之正交的偏差维度：当同一提示同时对多个评分标准进行打分时

    arXiv:2602.02219v3 Announce Type: replace  Abstract: Large language models are widely employed as evaluators, a paradigm commonly referred to as LLM-as-a-judge. Prior research has predominantly examined point-wise or pair-wise evaluation protocols; in contrast, our focus is on rubric-based evaluation, which has been attracting increasing attention owing to its utility for training models in domains where verification is otherwise difficult. In this work, we show that rubric-based evaluation implicitly resembles a multiple-choice setting and therefore exhibits position bias: LLMs tend to prefer score options that appear at specific positions within the rubric list. Through controlled experiments across multiple models and datasets, we demonstrate that this position bias is consistent. Its direction, however, is model-specific: some judges favor the first option, while others favor the last. We further identify a second, orthogonal axis of bias: when a prompt scores several criteria simu
    
[^147]: LLM生成还是人类撰写？比较arXiv上的综述论文与非综述论文

    LLM-Generated or Human-Written? Comparing Review and Non-Review Papers on ArXiv

    [https://arxiv.org/abs/2601.17036](https://arxiv.org/abs/2601.17036)

    本研究用两种检测方法证实arXiv上综述与非综述论文的LLM生成内容均显著增加，表明禁止综述论文上传的政策缺乏定量依据，且可能使某些计算机科学子学科面临高达50%的论文削减。

    

    arXiv最近以综述论文类别中存在大量LLM（大语言模型）生成内容为由，禁止在计算机科学领域上传未经同行评审发表的综述论文。然而，这一决定并未附带定量证据。在这项工作中，我们通过测量近年来综述论文与非综述研究论文中LLM生成内容的比例来验证这一说法。使用两种高质量的检测方法，我们发现综述论文和非综述论文中的LLM生成内容均大幅增加，且综述论文中的比例更高。然而，若按每个类别中已发表的LLM生成论文的数量来计算，非综述类LLM生成论文的估计数量几乎是综述论文的六倍。此外，我们发现这一政策对某些领域论文的影响远大于其他领域，计算机科学的子学科“计算机与社会”可能面临高达50%的论文削减。

    arXiv:2601.17036v2 Announce Type: replace-cross  Abstract: ArXiv recently prohibited the upload of unpublished review papers to its servers in the Computer Science domain, citing a high prevalence of LLM-generated content in these categories. However, this decision was not accompanied by quantitative evidence. In this work, we investigate this claim by measuring the proportion of LLM-generated content in review vs. non-review research papers in recent years. Using two high-quality detection methods, we find a substantial increase in LLM-generated content across both review and non-review papers, with a higher prevalence in review papers. However, when considering the number of LLM-generated papers published in each category, the estimates of non-review LLM-generated papers are almost six times higher. Furthermore, we find that this policy will affect papers in certain domains far more than others, with the CS subdiscipline Computers & Society potentially facing cuts of 50%. Our analysi
    
[^148]: Elsewise：通过可能性空间可视化创作开放式交互叙事

    Elsewise: Authoring Open-ended Interactive Narrative with Possibility Space Visualization

    [https://arxiv.org/abs/2601.15295](https://arxiv.org/abs/2601.15295)

    本文提出了Elsewise——一个面向大语言模型交互叙事的创作工具，通过新颖的“捆绑故事线”概念与可能性空间可视化，帮助创作者感知和理解叙事可能性空间，从而弥合创作者构想与玩家实际体验之间的差距。

    

    交互叙事（IN）创作者为玩家构建包含分歧性叙事可能的空间供其探索，玩家的输入决定了他们实际体验到哪些叙事可能性。生成式AI能够通过对预先创作的内容进行即兴扩展来响应玩家的开放式输入，从而实现新形式的交互叙事。然而，这种外推扩展可能会扩大创作者构想的故事与玩家实际体验的故事之间的差距，潜在地限制情节推进的力度以及创作者叙事意图的传达。为了弥合这一差距，我们推出了Elsewise：一个面向基于大语言模型（LLM）交互叙事的创作工具，它实现了一种新颖的“捆绑故事线”概念，以增强创作者对叙事可能性空间的感知和理解，使创作者能够从开放式的、用户可配置的叙事维度出发，探索其交互叙事作品各种可能游玩路径之间的相似性与差异性。

    arXiv:2601.15295v2 Announce Type: replace-cross  Abstract: Interactive narrative (IN) authors craft spaces of divergent narrative possibilities for players to explore, with the player's input determining which narrative possibilities they actually experience. Generative AI can enable new forms of IN by improvisationally expanding on pre-authored content in response to open-ended player input. However, this extrapolation risks widening the gap between author-envisioned and player-experienced stories, potentially limiting the strength of plot progression and the communication of the author's narrative intent. To bridge the gap, we introduce Elsewise: an authoring tool for LLM-based INs that implements a novel Bundled Storyline concept to enhance author's perception and understanding of the narrative possibility space, allowing authors to explore similarities and differences between possible playthroughs of their IN in terms of open-ended, user-configurable narrative dimensions. A user st
    
[^149]: 从评分标准到可靠分数：基于证据的LLM评判文本评估

    From Rubrics to Reliable Scores: Evidence-Grounded Text Evaluation with LLM Judges

    [https://arxiv.org/abs/2601.08654](https://arxiv.org/abs/2601.08654)

    提出Rulers框架，通过锁定任务级评分标准、执行基于证据的结构化判断，并将信号校准到人类分数边界，实现与人类评分更一致、更稳定且可审计的LLM文本评估。

    

    基于评分标准的文本评估越来越依赖大型语言模型（LLM）作为可扩展的评判者，然而固定的黑盒模型可能对相同标准产生不一致的解读，产生难以审计的分数归因，并且难以将判断准确映射到人类评分量表上。我们将这一挑战定义为“标准迁移”（criteria transfer）：即将人类评分标准的意图转化为稳定、可审计的推理时评分协议。我们提出了Rulers，它锁定任务级评分标准规范，通过结构化的、基于证据的判断来执行该标准，并将产生的信号校准到人类分数边界。在四个由评分标准管理的基准测试和多个固定骨干模型上，Rulers在大多数评估设置中与人类分数实现了更强的一致性，同时更好地匹配经验分数分布，并在语义等价的评分标准扰动下保持更高的稳定性。校准控制和组件消融实验……

    arXiv:2601.08654v3 Announce Type: replace  Abstract: Rubric-based text evaluation increasingly relies on large language models (LLMs) as scalable judges, yet frozen black-box models can interpret the same criteria inconsistently, produce score attributions that are difficult to audit, and map judgments poorly onto human scoring scales. We define this challenge as criteria transfer: translating human rubric intent into a stable, auditable inference-time scoring protocol. We introduce Rulers, which locks a task-level rubric specification, executes it through structured, evidence-grounded judgments, and calibrates the resulting signals to human score boundaries. Across four rubric-governed benchmarks and multiple frozen backbone models, Rulers achieves stronger agreement with human scores in most evaluated settings, while better matching empirical score distributions and remaining more stable under semantically equivalent rubric perturbations. Calibration controls and component ablations 
    
[^150]: 从表征到具身行动：翻译心智的ABC框架

    From Representation to Enactment: The ABC Framework of the Translating Mind

    [https://arxiv.org/abs/2511.16811](https://arxiv.org/abs/2511.16811)

    本文提出“ABC框架”，突破基于表征的翻译心智模型，将翻译视为具身行动过程——译者与大脑-身体-环境互动循环所生成的语言化翻译可供性图景中动态整合情感-评价、行为-行动与认知-推理过程，并在与文本、工具和情境的具身互动中实时共创意义。

    

    本文基于第三波延展心灵（EM）理论与激进具身行动主义，提出了一种替代基于表征的心智模型的新方案。我们构建了ABC框架，在该框架中，翻译不再被理解为对静态语际对应关系的操纵，而是一种具身实现的活动，动态地整合了情感-评价、行为-行动与认知-推理（ABC）三类过程。借助可供性理论，我们论证翻译心智是一种动态组织的过程：译者与其环境通过大脑-身体-环境的互动循环，共同构成并不断转化一个由语言化的翻译行动可供性所组成的图景。这一非表征性阐释将翻译重新界定为对社会文化实践的熟练参与，其中意义是译者通过与文本、工具和情境的具身互动实时共创的。

    arXiv:2511.16811v2 Announce Type: replace  Abstract: Building on the third-wave Extended Mind (EM) theory and radical enactivism, this article suggests an alternative to representation-based models of the mind. We build on the ABC framework in which translation is not understood as the manipulation of static interlingual correspondences but an enacted activity, dynamically integrating affective-evaluative, behavioral-enacting, and cognitive-inferential (ABC) processes. Drawing on affordance theory, we argue that the translating mind is the dynamically organized process through which the translator and its environment constitute and transform a landscape of enlanguaged translation affordances for action, that emerges through loops of brain-body-environment interactions. This non-representational account reframes translation as skillful participation in sociocultural practice, where meaning is co-created in real time through embodied interaction with texts, tools, and contexts.
    
[^151]: 为什么LLM智能体在探索新环境时会失败？一个世界建模的视角

    Why Do LLM Agents Fail in Exploring New Environments? A World-Modeling Perspective

    [https://arxiv.org/abs/2510.15047](https://arxiv.org/abs/2510.15047)

    该论文发现LLM智能体在不熟悉的环境中进行强化学习时会出现“探索坍塌”现象（Pass@k随训练下降），其根源在于对环境状态和动态的弱接地，并提出SPA方法，先通过自经验监督微调教会模型估计状态和预测转移，再进行奖励优化，从而缓解探索坍塌。

    

    大语言模型作为智能体在新环境中往往难以获得提升。我们识别并刻画了一种称之为“探索坍塌”的失败模式：在策略对环境状态不熟悉的环境中执行强化学习训练时，Pass@k（即k条采样轨迹中至少一条成功的概率）在训练过程中显著下降，即使Pass@1略有上升，这揭示了探索能力日益脆弱；而更接近预训练分布的环境则不会出现这种下降。我们将这种坍塌归因于智能体对环境状态和动态的弱接地，并研究了一种简单的补救方法：在针对奖励进行优化之前，显式地教会智能体估计当前状态并预测其状态转移。我们将其实现为SPA，一种先探索后利用的方法，通过自经验监督微调阶段对策略进行冷启动，收集模型自身的交互轨迹，并对状态和转移预测进行监督。

    arXiv:2510.15047v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) as agents often fail to improve in new environments. We identify and characterize a failure mode we call exploration collapse: under reinforcement learning (RL) in environments whose states are unfamiliar to the policy, Pass@k, the probability that at least one of k sampled trajectories succeeds, drops markedly over training even as Pass@1 edges up, revealing increasingly brittle exploration; environments closer to the pretraining distribution show no such decline. We trace this collapse to weak grounding in environment states and dynamics, and study a simple remedy: explicitly teaching the agent to estimate the current state and predict its transitions before optimizing for reward. We instantiate it as SPA, an explore-then-exploit recipe that cold-starts the policy with a Self-Experience supervised finetuning (SFT) stage, collecting the model's own interaction trajectories and supervising state and
    
[^152]: MADS：用于多样化说服数据生成的多智能体对话模拟

    MADS: Multi-Agent Dialogue Simulation for Diverse Persuasion Data Generation

    [https://arxiv.org/abs/2510.05124](https://arxiv.org/abs/2510.05124)

    MADS是一个多智能体对话模拟框架，通过用户智能体、对话智能体和优化智能体的自我博弈，无需人工标注即可低成本生成多样化的说服性多轮对话数据，并在真实营销场景中显著提升了小型大语言模型的说服能力和转化率。

    

    我们提出了MADS（多智能体对话模拟），这是一个通过智能体自我博弈生成具有说服力的多轮对话的可扩展框架。MADS采用三个协同工作的智能体：用户智能体，通过利用星座和MBTI类型等人格特征来模拟多样化的人物角色驱动行为；对话智能体，执行面向任务的说服策略；以及优化智能体，负责评估和完善对话结果。我们进一步通过用户的态度链建模以及专用大语言模型的说服力评估来验证其有效性。该方法能够在无需人工标注的情况下低成本生成训练数据，解决了缺乏用户数据、冷启动评估困难以及提示词效率低下等关键行业挑战。应用于真实世界的营销场景时，MADS显著提升了小型大语言模型的说服能力，将自然流量转化率提高了

    arXiv:2510.05124v3 Announce Type: replace  Abstract: We propose MADS (Multi-Agent Dialogue Simulation), a scalable framework for generating persuasive multi-turn dialogues via agent self-play. MADS employs three coordinated agents: User Agents designed to simulate diverse persona-driven behaviors by leveraging personality signifiers such as Zodiac Signs and MBTI types, a Dialog Agent executing task-oriented persuasion strategies and an Optimization Agent evaluating and refining dialogue outcomes. We further validate its effectiveness through users' Chain-of-Attitude (CoA) modeling and dedicated LLMs' persuasion assessment. This approach enables low-cost generation of training data without human annotation, addressing key industry challenges such as lack of user data, cold-start evaluation difficulties, and prompt inefficiency. Applied to a real-world marketing scenario, MADS significantly improved the persuasion capacity of small LLMs, increasing the organic traffic conversion rate by 
    
[^153]: 大型语言模型何时会表现出未经请求的欺骗行为？

    When Do Large Language Models Exhibit Unsolicited Deception?

    [https://arxiv.org/abs/2504.00285](https://arxiv.org/abs/2504.00285)

    本研究通过基于信号理论的预注册实验，利用修改后的2x2自由交流博弈测试了18个闭源和开源大型语言模型，发现所有模型都会在无指令的情况下自发歪曲自身行为实施欺骗，并且当欺骗有助于达成目标时，它们更倾向于这样做。

    

    大型语言模型（LLM）在被明确要求进行欺骗时能够有效地实施欺骗。在推理任务上表现更好的模型也更擅长在被提示后进行欺骗。但在什么条件下，它们会在没有指令的情况下自发欺骗？本研究使用信号理论的工具，在一个预注册的实验方案中评估了大型语言模型产生的未经请求的欺骗行为。我们使用修改后的2x2博弈（仿照囚徒困境的形式）评估了18个专有闭源和开源大型语言模型，并在博弈中增加了一个阶段，允许模型使用不受约束的语言自由地与对方智能体交流。这一设置创造了让模型歪曲自身行为的机会，且不同条件下这种做法对实现目标的有用程度各不相同。结果表明：1）所有被测试的大型语言模型至少在某些条件下会歪曲自己的行为；2）它们通常更倾向于在……（原摘要在此处截断）

    arXiv:2504.00285v2 Announce Type: replace  Abstract: Large Language Models (LLMs) are effective at deceiving when prompted to do so. Models that demonstrate better performance on reasoning tasks are also better at prompted deception. But under what conditions do they deceive without instruction to do so? This study evaluates unsolicited deception produced by LLMs in a preregistered experimental protocol using tools from signaling theory. We evaluated a range of 18 proprietary closed-source and open-source LLMs using modified 2x2 games (in the style of the Prisoner's Dilemma) augmented with a phase in which they can freely communicate to the other agent using unconstrained language. This setup creates an opportunity to misrepresent its actions in conditions that vary in how useful doing so might be towards goal satisfaction. The results indicate that 1) all tested LLMs misrepresent their actions in at least some conditions, 2) they are generally more likely to do so in situations in whi
    
[^154]: BTBR：一个用于大语言模型隐式偏见消除的贝叶斯理论驱动的概率-模糊框架

    BTBR: A Bayesian-Theory-Driven Probabilistic-Fuzzy Framework for Implicit Bias Removal in Large Language Models

    [https://arxiv.org/abs/2408.10608](https://arxiv.org/abs/2408.10608)

    该论文提出BTBR框架，将有偏见的知识建模为带有显式隶属函数的模糊子集，并结合贝叶斯理论构建概率-模糊混合方法，以检测和消除大语言模型中难以察觉的角色引发隐式偏见。

    

    大语言模型（LLMs）可能会从异构的训练语料库中编码带有偏见的关联，这些偏见在普通提示下不会立即显现，但当模型被引导扮演特定人口统计特征的角色时就会浮现。这种行为通常不表现为明显的有害输出，而是表现为在语义等价任务之间的系统性性能差异，使得由此产生的偏见难以检测和缓解。为了解决这一问题，我们将隐式偏见问题形式化为“角色引发性能差异”，并主张偏见证据应被视为一种分级信号而非二元标签。基于这一观察，我们将有偏见的知识建模为一个配备显式隶属函数的模糊子集，该隶属函数反映每个候选样本偏见证据的强度。基于这一公式化，我们提出了基于贝叶斯理论的偏见消除方法，这是一种混合的概率-模糊框架……

    arXiv:2408.10608v2 Announce Type: replace  Abstract: Large language models (LLMs) may encode biased associations from heterogeneous training corpora that are not immediately visible under ordinary prompting, but can surface when the model is steered toward particular demographic personas. Such behavior often manifests not as explicit toxic output, but as systematic performance differences across semantically equivalent tasks, making the resulting bias difficult to detect and mitigate. To address this issue, we formalize the implicit bias problem as persona-induced performance disparity and argue that bias evidence should be treated as a graded signal rather than a binary label. Motivated by this observation, we model biased knowledge as a fuzzy subset equipped with an explicit membership function that reflects the strength of bias evidence for each candidate example. Building on this formulation, we propose Bayesian-Theory-based Bias Removal (BTBR), a hybrid probabilistic-fuzzy framewo
    

